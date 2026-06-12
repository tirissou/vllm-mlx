# SPDX-License-Identifier: Apache-2.0
"""Adapters that bridge concrete prefix cache backends to the CacheManager protocol."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import mlx.core as mx

from vllm_mlx.request import Request
from vllm_mlx.turn_prefix_cache import TurnPrefixCache

if TYPE_CHECKING:
    from vllm_mlx.turn_prefix_cache import TurnNode

from .kv_cache import CacheIndexMap, _BATCH_KV_TYPES, validate_cache, extract_cache_states
from .cache_types import KVLayerSegment, KVQuantPolicy, RecurrentLayerSegment

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _linearize(tensor: "mx.array", offset: int, max_size: int) -> "mx.array":
    """Unwrap a RotatingKVCache ring buffer into a contiguous linear sequence."""
    if offset == max_size:
        return tensor[..., :offset, :]
    return mx.concatenate([tensor[..., offset:, :], tensor[..., :offset, :]], axis=-2)


class CacheManager(ABC):
    """Abstract base class for all prefix cache adapters.

    The Scheduler holds a CacheManager reference. Subclasses implement
    fetch(), store(), and boundaries(). All other methods have no-op defaults.
    """

    _cache_index_map: "CacheIndexMap | None" = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def boundaries(self, request) -> list[int]:
        """Return prefill chunk boundaries adjusted for any cached prefix.

        On a cache hit, boundaries are offset by cached_tokens.
        On a miss, boundaries are the raw turn boundaries from the request.
        """
        ...

    @abstractmethod
    def fetch(self, request) -> bool:
        """Look up a cached prefix for this request.

        On a hit, populates request._cache_state.turn_path and returns True.
        On a miss, populates request._cache_state for miss and returns False.
        """
        ...

    @abstractmethod
    def store(self, request, tokens: list[int], cache: list) -> bool:
        """Store the completed request's cache keyed on the full token sequence."""
        ...

    # ── Default no-ops (override as needed) ──────────────────────────────────

    def release(self, request) -> None:
        pass

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        """Called after each prefill chunk with the absolute token count.

        total_tokens_prefilled = cs.cached_tokens + chunk_tokens_just_processed.
        Adapter decides internally whether this lands on a boundary worth inserting.
        Cache semantic at a boundary: N tokens → cache @ N (no N-1 adjustment).
        """
        pass

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        pass

    def validate(self, cache: list) -> bool:
        """Validate cache state. Returns True if valid and usable."""
        return True

    def extract_cache(self, raw_cache: list) -> list | None:
        """Extract cache state from raw cache objects.

        Returns list of layer state dicts, or None on failure.
        Called during cleanup to prepare cache for storage.
        """
        return None

    def save(self, cache_dir: str) -> bool:
        """Persist cache to disk. Returns True on success."""
        return False

    def load(self, cache_dir: str) -> int:
        """Load cache from disk. Returns entries loaded."""
        return 0

    def close(self) -> None:
        """Cleanup resources (SSD threads, file handles, etc.)."""
        pass

    def _ensure_cache_index_map(self, layers: list) -> "CacheIndexMap":
        if self._cache_index_map is not None:
            return self._cache_index_map

        from mlx_lm.models.cache import BatchRotatingKVCache, ArraysCache

        kv_indices = []
        rotating_indices = []
        recurrent_indices = []

        for i, layer in enumerate(layers):
            if isinstance(layer, dict):
                name = layer.get("class_name", "")
                if "Rotating" in name:
                    rotating_indices.append(i)
                elif "KV" in name or "Quantized" in name:
                    kv_indices.append(i)
                else:
                    recurrent_indices.append(i)
            else:
                if isinstance(layer, BatchRotatingKVCache):
                    rotating_indices.append(i)
                elif isinstance(layer, _BATCH_KV_TYPES):
                    kv_indices.append(i)
                else:
                    recurrent_indices.append(i)

        self._cache_index_map = CacheIndexMap(
            kv_indices=kv_indices,
            rotating_indices=rotating_indices,
            recurrent_indices=recurrent_indices,
        )
        return self._cache_index_map


class TurnCacheManager(CacheManager):
    """Adapts TurnPrefixCache to the CacheManager protocol."""

    def __init__(
        self, inner: TurnPrefixCache, kv_bits: int | None = 8, kv_group_size: int = 64
    ):
        self._inner = inner
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        # request_id -> currently pinned leaf node (Active Leaf invariant).
        # Populated by fetch() on hit, advanced by store(), cleared by release().
        self._pinned_leaves: dict[str, "TurnNode"] = {}

    def boundaries(self, request) -> list[int]:
        cs = request._cache_state
        cached = cs.cached_tokens if cs is not None else 0
        turn_bds = getattr(request, "_turn_boundaries", None) or []
        return sorted(b - cached for b in turn_bds if b > cached)

    @staticmethod
    def messages_to_segments(request) -> list:
        """Split request token sequence into per-message Segment objects."""
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if not _turn_boundaries:
            return []
        B_sys = _turn_boundaries[0]
        if B_sys <= 0:
            return []
        segments: list = [Segment(role="system", token_ids=full_tokens[:B_sys])]
        prev = B_sys
        for B_k in _turn_boundaries[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(
                    Segment(role="conversation", token_ids=full_tokens[prev:B_k])
                )
                prev = B_k
        if prev < len(full_tokens):
            segments.append(Segment(role="user", token_ids=full_tokens[prev:]))
        return segments

    @staticmethod
    def _segment(
        live_states: list[dict],
        policy: "KVQuantPolicy | None" = None,
        group_size: int = 64,
    ) -> tuple[list, list]:
        """Transform live cache states into KVLayerSegment / RecurrentLayerSegment.

        bits per layer is resolved via policy.bits_for(class_name); None means
        store float. Each emitted KVLayerSegment carries metadata['bits'] = bits.
        """
        from .kv_cache import QuantizedArray

        kv_list = [None] * len(live_states)
        rec_list = [None] * len(live_states)

        for i, state_dict in enumerate(live_states):
            class_name = state_dict["class_name"]
            state = state_dict["state"]
            meta = state_dict.get("meta_state", ())
            bits = policy.bits_for(class_name) if policy is not None else None

            if class_name == "RotatingKVCache":
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

                if bits is None:
                    # Track B: store float arrays, no quantization
                    sliced_keys = mx.stop_gradient(lin_keys)
                    sliced_values = mx.stop_gradient(lin_values)
                    mx.eval(sliced_keys, sliced_values)
                    kv_list[i] = KVLayerSegment(
                        keys=sliced_keys,
                        values=sliced_values,
                        metadata={
                            "class_name": "RotatingKVCache",
                            "layer_index": i,
                            "merge_strategy": "last",
                            "n_tokens": lin_keys.shape[-2],
                            "max_size": max_size,
                            "keep": keep,
                            "offset": offset,
                            "_idx": _idx,
                            "bits": bits,
                        },
                    )
                    continue

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

                kv_list[i] = KVLayerSegment(
                    keys=q_keys,
                    values=q_values,
                    metadata={
                        "class_name": "RotatingKVCache",
                        "layer_index": i,
                        "merge_strategy": "last",
                        "n_tokens": lin_keys.shape[-2],
                        "max_size": max_size,
                        "keep": keep,
                        "offset": offset,
                        "_idx": _idx,
                        "bits": bits,
                    },
                )

            elif "KVCache" in class_name:
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
                    kv_list[i] = KVLayerSegment(
                        keys=sliced_keys,
                        values=sliced_values,
                        metadata={
                            "class_name": class_name,
                            "layer_index": i,
                            "merge_strategy": "concatenate",
                            "n_tokens": actual_end,
                            "bits": bits,
                        },
                    )
                    continue
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

                kv_list[i] = KVLayerSegment(
                    keys=q_keys,
                    values=q_values,
                    metadata={
                        "class_name": class_name,
                        "layer_index": i,
                        "merge_strategy": "concatenate",
                        "n_tokens": actual_end,
                        "bits": bits,
                    },
                )

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

    @staticmethod
    def _assemble(
        kv_layers: list,
        recurrent_layers: list,
        group_size: int = 64,
        bits: int | None = 8,
    ) -> list:
        """Reconstruct live cache objects from KVLayerSegment and RecurrentLayerSegment lists."""
        from mlx_lm.models.cache import RotatingKVCache as _RotatingKVCache
        from .kv_cache import QuantizedArray

        result: dict[int, Any] = {}

        for layer in kv_layers:
            li = layer.metadata["layer_index"]
            is_quantized_payload = isinstance(layer.keys, QuantizedArray)
            if layer.metadata["class_name"] == "RotatingKVCache":
                if is_quantized_payload:
                    dq_keys = mx.dequantize(
                        layer.keys.packed,
                        layer.keys.scales,
                        layer.keys.biases,
                        group_size=group_size,
                        bits=bits,
                    )
                    dq_values = mx.dequantize(
                        layer.values.packed,
                        layer.values.scales,
                        layer.values.biases,
                        group_size=group_size,
                        bits=bits,
                    )
                else:
                    dq_keys = layer.keys
                    dq_values = layer.values
                max_size = layer.metadata["max_size"]
                if dq_keys.shape[-2] > max_size:
                    dq_keys = dq_keys[..., -max_size:, :]
                    dq_values = dq_values[..., -max_size:, :]
                _idx = layer.metadata.get("_idx", dq_keys.shape[-2])
                # _segment stores keys in chronological (linearized) order.
                # RotatingKVCache expects ring order: rotate back so the ring write
                # position lands at _idx, matching the live cache layout.
                if 0 < _idx < max_size and dq_keys.shape[-2] == max_size:
                    split = max_size - _idx
                    dq_keys = mx.concatenate(
                        [dq_keys[..., split:, :], dq_keys[..., :split, :]], axis=-2
                    )
                    dq_values = mx.concatenate(
                        [dq_values[..., split:, :], dq_values[..., :split, :]], axis=-2
                    )
                cache = _RotatingKVCache(max_size, layer.metadata.get("keep", 0))
                cache.keys = dq_keys
                cache.values = dq_values
                cache.offset = layer.metadata.get("offset", dq_keys.shape[-2])
                cache._idx = _idx
            else:
                if is_quantized_payload:
                    from .batch_quantized_kv_cache import BatchQuantizedKVCache

                    n_tokens = layer.metadata.get("n_tokens", layer.keys.packed.shape[-2])
                    # Pad to the next `step` boundary so the first decode-step
                    # update_and_fetch lands on the in-place assignment branch
                    # rather than re-allocating + concatenating the whole buffer.
                    # `((n // step) + 1) * step` (rather than the usual round-up)
                    # guarantees pad >= 1 even when n_tokens is already aligned —
                    # an exactly-aligned buffer would otherwise still hit the
                    # expand branch on the very first step.
                    step = BatchQuantizedKVCache.step
                    padded_len = ((n_tokens // step) + 1) * step
                    pad = padded_len - n_tokens

                    def _pad_qa(qa):
                        if pad == 0:
                            return qa
                        return QuantizedArray(*[
                            mx.concatenate(
                                [c, mx.zeros((*c.shape[:-2], pad, c.shape[-1]), dtype=c.dtype)],
                                axis=-2,
                            )
                            for c in qa
                        ])

                    cache = BatchQuantizedKVCache.from_quantized_arrays(
                        keys=_pad_qa(layer.keys),
                        values=_pad_qa(layer.values),
                        n_tokens=n_tokens,
                        group_size=group_size,
                        bits=bits,
                    )
                else:
                    from mlx_lm.models.cache import KVCache as _KVCache

                    n_tokens = layer.metadata.get("n_tokens", layer.keys.shape[-2])
                    # Pre-pad to the next `step` boundary so the first decode-step
                    # update_and_fetch lands on the in-place assignment branch
                    # rather than `mx.concatenate([n_tokens_buffer, step_zeros])` —
                    # the same buffer-doubling pattern fixed in the quantized
                    # branch above. `((n // step) + 1) * step` guarantees pad >= 1
                    # even when n_tokens is already aligned.
                    step = _KVCache.step
                    padded_len = ((n_tokens // step) + 1) * step
                    pad = padded_len - n_tokens

                    k = layer.keys[..., :n_tokens, :]
                    v = layer.values[..., :n_tokens, :]
                    if pad:
                        k = mx.concatenate(
                            [k, mx.zeros((*k.shape[:-2], pad, k.shape[-1]), dtype=k.dtype)],
                            axis=-2,
                        )
                        v = mx.concatenate(
                            [v, mx.zeros((*v.shape[:-2], pad, v.shape[-1]), dtype=v.dtype)],
                            axis=-2,
                        )

                    cache = _KVCache()
                    cache.keys = k
                    cache.values = v
                    # offset is the logical token count; cache.keys.shape[-2] is
                    # padded_len. Attention masking is offset-based, so this is
                    # what the model expects (see mlx_lm KVCache.update_and_fetch
                    # which reads `prev = self.offset`).
                    cache.offset = n_tokens
            result[li] = cache

        for layer in recurrent_layers:
            li = layer.metadata["layer_index"]
            cache_cls = layer.metadata.get("class_ref")
            if cache_cls is not None and hasattr(cache_cls, "from_state"):
                result[li] = cache_cls.from_state(layer.arrays, ())
            else:
                from mlx_lm.models.cache import ArraysCache

                cache = ArraysCache.from_state(layer.arrays, ())
                result[li] = cache

        return [result[li] for li in sorted(result)]

    def _set_miss_state(self, request) -> None:
        """Populate request._cache_state with miss values."""
        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.hit_type = "miss"
            cs.cached_tokens = 0
            cs.turn_path = []
            cs.remaining_tokens = request.prompt_token_ids
            cs.prefill_boundaries = self.boundaries(request)

    def fetch(self, request) -> bool:
        segments = self.messages_to_segments(request)
        if not segments:
            self._set_miss_state(request)
            return False

        path, _ = self._inner.match(segments)
        if not path:
            self._set_miss_state(request)
            return False

        # Active Leaf invariant: match() incremented ref_count on every node
        # in `path`. Release the ancestors so only the leaf retains its +1.
        if len(path) > 1:
            self._inner.release(path[:-1])

        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.turn_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            # Release the remaining leaf pin; this is a miss after all.
            self._inner.release([path[-1]])
            self._pinned_leaves.pop(request.request_id, None)
            self._set_miss_state(request)
            return False

        _probe_req_id = getattr(request, "request_id", None) or getattr(request, "uid", "?")
        _probe_pre_active = mx.get_active_memory()
        kv_data, rec_data = self._inner.collect_path_data(ancestor)
        reconstructed = self._assemble(kv_data, rec_data, self._kv_group_size, self._kv_bits)
        _probe_n_tokens = sum(l.metadata.get("n_tokens", 0) for l in (kv_data or []) if l is not None)
        del kv_data, rec_data
        # Materialize the KVCache arrays now so the lazy computation graph
        # is freed before decode starts.
        arrays_to_eval = []
        for _layer in reconstructed or []:
            for attr in ("keys", "values"):
                v = getattr(_layer, attr, None)
                if v is None or callable(v):
                    continue
                if isinstance(v, mx.array):
                    arrays_to_eval.append(v)
                else:
                    for comp in (v if isinstance(v, (list, tuple)) else []):
                        if isinstance(comp, mx.array):
                            arrays_to_eval.append(comp)
        if arrays_to_eval:
            mx.eval(*arrays_to_eval)
        _probe_post_active = mx.get_active_memory()
        logging.warning(
            "[memprobe:fetch] req=%s n_tokens=%d pre_assemble=%.2fGB "
            "post_assemble=%.2fGB delta=%.2fGB peak=%.2fGB",
            _probe_req_id,
            _probe_n_tokens,
            _probe_pre_active / 1e9,
            _probe_post_active / 1e9,
            (_probe_post_active - _probe_pre_active) / 1e9,
            mx.get_peak_memory() / 1e9,
        )
        if not self.validate(reconstructed):
            # Release the remaining leaf pin; this is a miss after all.
            self._inner.release([path[-1]])
            self._pinned_leaves.pop(request.request_id, None)
            self._set_miss_state(request)
            return False
        cached_tokens = ancestor.n_tokens
        if cs is not None:
            cs.hit_type = "hit"
            cs.cache = reconstructed
            cs.cached_tokens = cached_tokens
            cs.remaining_tokens = list(request.prompt_token_ids[cached_tokens:])
            cs.prefill_boundaries = self.boundaries(request)
        # Record the pinned leaf so release() can find it.
        self._pinned_leaves[request.request_id] = path[-1]
        return True

    def store(self, request, tokens: list[int] = None, cache: list = None) -> bool:
        from .turn_prefix_cache import Segment

        if (
            cache is None
            and isinstance(tokens, list)
            and (not tokens or not isinstance(tokens[0], int))
        ):
            cache = tokens
            tokens = None
        if cache is None:
            cache = []

        segments = self.messages_to_segments(request)
        if not segments or not getattr(request, "output_token_ids", None):
            return False

        cs = getattr(request, "_cache_state", None)
        path = cs.turn_path if cs is not None else []
        matched_depth = len(path)
        parent = path[-1] if path else self._inner.root
        new_segments = segments[matched_depth:]

        if new_segments:
            # Combine the last unmatched prompt segment with the generated output
            # into a single response node.
            response_tokens = list(segments[-1].token_ids) + list(
                request.output_token_ids
            )
        else:
            # Full prompt was already in the trie; record only the new output as
            # a child of the deepest matched node.
            response_tokens = list(request.output_token_ids)

        if cache and not isinstance(cache[0], dict):
            from .kv_cache import extract_layer_state

            cache = [
                d for layer in cache if (d := extract_layer_state(layer)) is not None
            ]

        if cache:
            prev_end = path[-1].n_tokens if path else 0
            cache = self._slice_kv_to_delta(cache, prev_end)
            kv_sparse, rec_sparse = self._segment(
                cache, self._kv_group_size, self._kv_bits
            )
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
        else:
            kv_layers, rec_layers = [], []

        # Snapshot current pinned leaf BEFORE any mutation so we can roll back
        # cleanly on insert failure.
        old_leaf = self._pinned_leaves.get(request.request_id)

        # Insert first; only touch ref_counts / _pinned_leaves after success so
        # that an exception leaves the previous leaf pin intact.
        new_leaf = self._inner.insert(
            parent,
            Segment(role="conversation", token_ids=response_tokens),
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
        )

        # Success path: advance the active leaf — unpin the old, pin the new.
        if old_leaf is not None:
            self._inner.release([old_leaf])
        with self._inner._lock:
            new_leaf.ref_count += 1
        self._pinned_leaves[request.request_id] = new_leaf
        if cs is not None:
            cs.turn_path = self._inner._inorder_path(new_leaf)
        return True

    @staticmethod
    def _slice_kv_to_delta(states: list[dict], prev_end: int) -> list[dict]:
        """Slice KVCache state arrays to the incremental delta [prev_end:actual_end].

        RotatingKVCache is left untouched — its ring buffer is not a cumulative sequence.
        """
        if prev_end == 0:
            return states
        from .kv_cache import QuantizedArray

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

    def release(self, request) -> None:
        leaf = self._pinned_leaves.pop(request.request_id, None)
        if leaf is not None:
            # Go through the inner trie's release() so the node is re-added to
            # the eviction heap if it just became evictable.
            self._inner.release([leaf])
        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.turn_path = []

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if total_tokens_prefilled not in _turn_boundaries:
            return

        try:
            abs_idx = _turn_boundaries.index(total_tokens_prefilled)
        except ValueError:
            return

        segments = self.messages_to_segments(request)
        if abs_idx >= len(segments):
            return

        cs = getattr(request, "_cache_state", None)
        turn_path = cs.turn_path if cs is not None else []
        if len(turn_path) > abs_idx:
            return

        parent = turn_path[-1] if turn_path else self._inner.root
        segment = segments[abs_idx]
        is_sys = segment.role == "system" and abs_idx == 0

        if extracted_cache:
            if not isinstance(extracted_cache[0], dict):
                from .kv_cache import extract_layer_state

                extracted_cache = [
                    d
                    for layer in extracted_cache
                    if (d := extract_layer_state(layer)) is not None
                ]
            prev_end = _turn_boundaries[abs_idx - 1] if abs_idx > 0 else 0
            extracted_cache = self._slice_kv_to_delta(extracted_cache, prev_end)
            kv_sparse, rec_sparse = self._segment(
                extracted_cache, self._kv_group_size, self._kv_bits
            )
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
        else:
            kv_layers, rec_layers = [], []

        # Snapshot the previous pinned leaf BEFORE any mutation so that an
        # exception from insert() leaves the existing pin intact (implicit
        # rollback — same shape as store()).
        old_leaf = self._pinned_leaves.get(request.request_id)

        new_node = self._inner.insert(
            parent,
            segment,
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )

        # Insert succeeded — advance the active leaf: release the old leaf
        # (if any) and pin the new one.
        if old_leaf is not None:
            self._inner.release([old_leaf])
        with self._inner._lock:
            new_node.ref_count += 1
        self._pinned_leaves[request.request_id] = new_node
        # Append rather than rebuild the full path: checkpoint always inserts
        # depth abs_idx in order, so the path is already correct up to abs_idx-1.
        if cs is not None:
            cs.turn_path.append(new_node)

    # ── Updated: save() / load() with error handling ──────────────────────────

    def save(self, cache_dir: str) -> bool:
        try:
            return self._inner.save(cache_dir)
        except Exception:
            return False

    def load(self, cache_dir: str) -> int:
        try:
            self._inner.load(cache_dir)
            return 0
        except Exception:
            return 0

    # ── New: validate, extract_cache, close ──────────────────────────────────

    def validate(self, cache: list) -> bool:
        return validate_cache(cache)

    def extract_cache(self, raw_cache: list) -> list | None:
        return extract_cache_states(raw_cache)

    def close(self) -> None:
        pass
