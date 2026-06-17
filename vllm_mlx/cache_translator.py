"""Trie-storage translator: live mlx-lm cache state ↔ KVLayerSegment.

Exposes the round-trip law (CONTEXT.md "Cache translator"):

    assemble(segment(live_states, policy, group_size), group_size) ≈ live_states

The producers (segment, slice_kv_to_delta) and consumer (assemble) here are
the seam that ADR-0005 governs.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_mlx.cache_types import (
    KVConcatSegment,
    KVLayerSegment,
    KVRotatingSegment,
    KVQuantPolicy,
    RecurrentLayerSegment,
)

def _linearize(tensor: mx.array, offset: int, max_size: int) -> mx.array:
    if offset == max_size:
        return tensor[..., :offset, :]
    return mx.concatenate([tensor[..., offset:, :], tensor[..., :offset, :]], axis=-2)


def _apply_rotating_window(arr: mx.array, keep: int, max_size: int) -> mx.array:
    n = arr.shape[-2]
    if n <= max_size:
        return arr
    if keep <= 0:
        return arr[..., -max_size:, :]
    trim_size = n - max_size
    return mx.concatenate(
        [arr[..., :keep, :], arr[..., trim_size + keep:, :]], axis=-2
    )


def segment(
    live_states: list[dict],
    policy: KVQuantPolicy | None = None,
    group_size: int = 64,
) -> tuple[list[KVLayerSegment | None], list[RecurrentLayerSegment | None]]:
    """Translate live mlx-lm cache states into immutable trie segments.

    Contract: emitted arrays are evaluated and graph-detached
    (mx.eval then mx.stop_gradient). Callers may delete live_states and
    call mx.clear_cache(); segment arrays survive.
    """
    kv_list: list[KVLayerSegment | None] = [None] * len(live_states)
    rec_list: list[RecurrentLayerSegment | None] = [None] * len(live_states)

    for i, state_dict in enumerate(live_states):
        class_name = state_dict["class_name"]
        state = state_dict["state"]
        meta = state_dict.get("meta_state", ())
        bits = policy.bits_for(class_name) if policy is not None else None

        if class_name == "RotatingKVCache":
            kv_list[i] = _segment_rotating(state, meta, i, bits, group_size)
        elif "KVCache" in class_name:
            kv_list[i] = _segment_concat(state, meta, i, class_name, bits, group_size)
        else:
            rec_list[i] = RecurrentLayerSegment(
                arrays=state,
                metadata={
                    "class_name": class_name,
                    "layer_index": i,
                    "class_ref": state_dict.get("class_ref"),
                },
            )

    return kv_list, rec_list


def _segment_rotating(state, meta, i, bits, group_size) -> KVRotatingSegment:
    """Lift of prefix_cache_adapters.py:_segment rotating branch (lines 302-397)."""
    from vllm_mlx.kv_cache import QuantizedArray

    try:
        keep, max_size, offset, _idx = map(int, meta)
    except (TypeError, ValueError):
        max_size = state[0].shape[2]
        offset = max_size
        _idx = max_size
        keep = 0

    # _idx is the ring write position (offset % max_size when wrapped,
    # or max_size when the buffer just became full without wrapping).
    # Using raw offset here would slice out-of-bounds when offset > max_size.
    lin_keys = _linearize(state[0], _idx, max_size)
    lin_values = _linearize(state[1], _idx, max_size)

    # Eagerly cap the buffer to max_size. _update_concat (mlx-lm's
    # multi-token path) leaves the buffer at max_size + S - 1 to
    # give every new token max_size of preceding context — that
    # extra is only needed for in-flight attention, not storage.
    # The first decode tick would trim it via _trim; we do the
    # same here so the trie doesn't pay for transient rows.
    pre_trim = lin_keys.shape[-2]
    lin_keys = _apply_rotating_window(lin_keys, keep, max_size)
    lin_values = _apply_rotating_window(lin_values, keep, max_size)
    if lin_keys.shape[-2] < pre_trim:
        # After trim the ring is full; signal mlx-lm to wrap on
        # the next decode write (matches the post-_trim assignment
        # `self._idx = self.max_size` in _update_in_place).
        _idx = max_size

    if bits is None:
        # Track B: store float arrays, no quantization
        sliced_keys = mx.stop_gradient(lin_keys)
        sliced_values = mx.stop_gradient(lin_values)
        mx.eval(sliced_keys, sliced_values)
        return KVRotatingSegment(
            keys=sliced_keys,
            values=sliced_values,
            layer_index=i,
            n_tokens=lin_keys.shape[-2],
            bits=bits,
            max_size=max_size,
            keep=keep,
            offset=offset,
            idx=_idx,
        )

    # bits is not None: quantize path
    q_keys = QuantizedArray(
        *mx.quantize(lin_keys, group_size=group_size, bits=bits)
    )
    q_values = QuantizedArray(
        *mx.quantize(lin_values, group_size=group_size, bits=bits)
    )
    mx.eval(
        q_keys.packed,
        q_keys.scales,
        q_keys.biases,
        q_values.packed,
        q_values.scales,
        q_values.biases,
    )
    # mx.stop_gradient severs the MLX computation graph so the
    # trie node does not retain a reference to the source float16
    # Metal buffers via the lazy quantize dependency chain.
    q_keys = QuantizedArray(
        packed=mx.stop_gradient(q_keys.packed),
        scales=mx.stop_gradient(q_keys.scales),
        biases=mx.stop_gradient(q_keys.biases),
    )
    q_values = QuantizedArray(
        packed=mx.stop_gradient(q_values.packed),
        scales=mx.stop_gradient(q_values.scales),
        biases=mx.stop_gradient(q_values.biases),
    )

    return KVRotatingSegment(
        keys=q_keys,
        values=q_values,
        layer_index=i,
        n_tokens=lin_keys.shape[-2],
        bits=bits,
        max_size=max_size,
        keep=keep,
        offset=offset,
        idx=_idx,
    )


def _segment_concat(state, meta, i, class_name, bits, group_size) -> KVConcatSegment:
    """Lift of prefix_cache_adapters.py:_segment concat branch (lines 399-486)."""
    from vllm_mlx.kv_cache import QuantizedArray

    try:
        actual_end = int(meta[0]) if meta else (
            state[0].packed.shape[-2] if isinstance(state[0], QuantizedArray)
            else state[0].shape[2]
        )
    except (TypeError, ValueError, IndexError):
        actual_end = (
            state[0].packed.shape[-2] if isinstance(state[0], QuantizedArray)
            else state[0].shape[2]
        )

    if isinstance(state[0], QuantizedArray):
        # Track A: state already quantized — stop_gradient and store as-is
        q_keys = QuantizedArray(
            packed=mx.stop_gradient(state[0].packed[..., :actual_end, :]),
            scales=mx.stop_gradient(state[0].scales[..., :actual_end, :]),
            biases=mx.stop_gradient(state[0].biases[..., :actual_end, :]),
        )
        q_values = QuantizedArray(
            packed=mx.stop_gradient(state[1].packed[..., :actual_end, :]),
            scales=mx.stop_gradient(state[1].scales[..., :actual_end, :]),
            biases=mx.stop_gradient(state[1].biases[..., :actual_end, :]),
        )
        mx.eval(
            q_keys.packed, q_keys.scales, q_keys.biases,
            q_values.packed, q_values.scales, q_values.biases,
        )
    elif bits is None:
        # Track B: float precision — stop_gradient and store as float arrays
        sliced_keys = mx.stop_gradient(state[0][:, :, :actual_end, :])
        sliced_values = mx.stop_gradient(state[1][:, :, :actual_end, :])
        mx.eval(sliced_keys, sliced_values)
        return KVConcatSegment(
            keys=sliced_keys,
            values=sliced_values,
            layer_index=i,
            n_tokens=actual_end,
            bits=bits,
            class_name=class_name,
        )
    else:
        # Track C: quantize float arrays
        sliced_keys = state[0][:, :, :actual_end, :]
        sliced_values = state[1][:, :, :actual_end, :]
        q_keys = QuantizedArray(
            *mx.quantize(sliced_keys, group_size=group_size, bits=bits)
        )
        q_values = QuantizedArray(
            *mx.quantize(sliced_values, group_size=group_size, bits=bits)
        )
        mx.eval(
            q_keys.packed,
            q_keys.scales,
            q_keys.biases,
            q_values.packed,
            q_values.scales,
            q_values.biases,
        )
        # mx.stop_gradient severs the MLX computation graph so the
        # trie node does not retain a reference to the source float16
        # Metal buffers via the lazy quantize dependency chain.
        q_keys = QuantizedArray(
            packed=mx.stop_gradient(q_keys.packed),
            scales=mx.stop_gradient(q_keys.scales),
            biases=mx.stop_gradient(q_keys.biases),
        )
        q_values = QuantizedArray(
            packed=mx.stop_gradient(q_values.packed),
            scales=mx.stop_gradient(q_values.scales),
            biases=mx.stop_gradient(q_values.biases),
        )

    return KVConcatSegment(
        keys=q_keys,
        values=q_values,
        layer_index=i,
        n_tokens=actual_end,
        bits=bits,
        class_name=class_name,
    )


def assemble(
    kv_layers: list[KVLayerSegment],
    recurrent_layers: list[RecurrentLayerSegment],
    group_size: int = 64,
) -> list:
    """Reconstruct live mlx-lm cache objects from segments. Inverse of segment()."""
    result: dict[int, Any] = {}
    for layer in kv_layers:
        if layer is None:
            continue
        result[layer.layer_index] = layer.reconstruct(group_size)
    for rec_layer in recurrent_layers:
        if rec_layer is None:
            continue
        li = rec_layer.metadata["layer_index"]
        class_ref = rec_layer.metadata.get("class_ref")
        # Lifted from prefix_cache_adapters.py:_assemble recurrent branch (lines 629-638).
        if class_ref is not None and hasattr(class_ref, "from_state"):
            result[li] = class_ref.from_state(rec_layer.arrays, ())
        else:
            from mlx_lm.models.cache import ArraysCache

            cache = ArraysCache.from_state(rec_layer.arrays, ())
            result[li] = cache
    return [result[i] for i in sorted(result)]


def slice_kv_to_delta(states: list[dict], prev_end: int) -> list[dict]:
    """Slice KVCache state arrays to the incremental delta [prev_end:actual_end].

    RotatingKVCache state is left untouched (its ring buffer is not cumulative).
    """
    # Direct lift from prefix_cache_adapters.py:_slice_kv_to_delta (lines 812-849).
    if prev_end == 0:
        return states
    from vllm_mlx.kv_cache import QuantizedArray

    result = []
    for s in states:
        cname = s.get("class_name", "")
        if "KVCache" in cname and "Rotating" not in cname:
            state = s["state"]
            meta = s.get("meta_state") or ()
            if isinstance(state[0], QuantizedArray):
                actual_end = int(meta[0]) if meta else state[0].packed.shape[-2]

                def _slice_qa(qa, start, end):
                    return QuantizedArray(
                        packed=mx.contiguous(qa.packed[..., start:end, :]),
                        scales=mx.contiguous(qa.scales[..., start:end, :]),
                        biases=mx.contiguous(qa.biases[..., start:end, :]),
                    )

                sliced_state = (
                    _slice_qa(state[0], prev_end, actual_end),
                    _slice_qa(state[1], prev_end, actual_end),
                )
            else:
                actual_end = int(meta[0]) if meta else state[0].shape[2]
                sliced_state = tuple(
                    mx.contiguous(arr[:, :, prev_end:actual_end, :]) for arr in state[:2]
                )
            new_meta = (actual_end - prev_end,) + tuple(meta[1:])
            s = {**s, "state": sliced_state, "meta_state": new_meta}
        result.append(s)
    return result
