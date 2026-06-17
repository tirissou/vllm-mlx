# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class KVLayerSegment(ABC):
    """Abstract per-layer KV snapshot stored in a TurnNode.

    Two concrete subclasses dispatch path-merge and reconstruction polymorphically:
    - KVConcatSegment: standard KVCache (incremental accumulation, axis=-2 concat)
    - KVRotatingSegment: RotatingKVCache (ring buffer, only last node's segment used)

    Holds evaluated, graph-detached arrays per the segment contract (CONTEXT.md).
    """

    keys: Any           # QuantizedArray | mx.array
    values: Any         # QuantizedArray | mx.array
    layer_index: int
    n_tokens: int
    bits: int | None    # None = bf16, int = quantized at that precision
    class_name: str     # the live mlx-lm class name this segment was extracted from

    @abstractmethod
    def merge_path(self, path: list["KVLayerSegment"]) -> "KVLayerSegment":
        """Merge a path of same-layer segments. ``path`` includes self at path[-1]."""

    @abstractmethod
    def reconstruct(self, group_size: int) -> Any:
        """Reconstruct a live mlx-lm cache object for this layer."""


@dataclass(frozen=True)
class KVConcatSegment(KVLayerSegment):
    """KV snapshot for a standard KVCache layer (incremental, concatenated)."""

    class_name: str = "KVCache"

    def merge_path(self, path: list[KVLayerSegment]) -> KVLayerSegment:
        return KVConcatSegment.concat(path)  # type: ignore[arg-type]

    def reconstruct(self, group_size: int) -> Any:
        from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
        from vllm_mlx.kv_cache import QuantizedArray

        if isinstance(self.keys, QuantizedArray):
            # Pad to step boundary so the first decode lands on in-place update.
            step = BatchQuantizedKVCache.step
            padded_len = ((self.n_tokens // step) + 1) * step
            pad = padded_len - self.n_tokens

            def _pad_qa(qa: QuantizedArray) -> QuantizedArray:
                if pad == 0:
                    return qa
                return QuantizedArray(*[
                    mx.concatenate(
                        [c, mx.zeros((*c.shape[:-2], pad, c.shape[-1]), dtype=c.dtype)],
                        axis=-2,
                    )
                    for c in (qa.packed, qa.scales, qa.biases)
                ])

            return BatchQuantizedKVCache.from_quantized_arrays(
                keys=_pad_qa(self.keys),
                values=_pad_qa(self.values),
                n_tokens=self.n_tokens,
                group_size=group_size,
                bits=self.bits,
            )
        # Float path: mirror prefix_cache_adapters.py:_assemble float branch (lines 593-626).
        from mlx_lm.models.cache import KVCache

        n_tokens = self.n_tokens
        step = KVCache.step
        padded_len = ((n_tokens // step) + 1) * step
        pad = padded_len - n_tokens

        k = self.keys[..., :n_tokens, :]
        v = self.values[..., :n_tokens, :]
        if pad > 0:
            k = mx.concatenate(
                [k, mx.zeros((*k.shape[:-2], pad, k.shape[-1]), dtype=k.dtype)],
                axis=-2,
            )
            v = mx.concatenate(
                [v, mx.zeros((*v.shape[:-2], pad, v.shape[-1]), dtype=v.dtype)],
                axis=-2,
            )

        cache = KVCache()
        cache.keys = k
        cache.values = v
        cache.offset = n_tokens
        return cache

    @classmethod
    def concat(cls, layers: list["KVConcatSegment"]) -> "KVConcatSegment":
        """Concat same-layer segments along the sequence axis."""
        from vllm_mlx.kv_cache import QuantizedArray

        first_bits = layers[0].bits
        for seg in layers[1:]:
            assert seg.bits == first_bits, (
                f"KVConcatSegment.concat: mismatched bits "
                f"{first_bits!r} vs {seg.bits!r}"
            )

        if isinstance(layers[0].keys, QuantizedArray):
            merged_keys = QuantizedArray(
                packed=mx.concatenate([seg.keys.packed for seg in layers], axis=-2),
                scales=mx.concatenate([seg.keys.scales for seg in layers], axis=-2),
                biases=mx.concatenate([seg.keys.biases for seg in layers], axis=-2),
            )
            merged_values = QuantizedArray(
                packed=mx.concatenate([seg.values.packed for seg in layers], axis=-2),
                scales=mx.concatenate([seg.values.scales for seg in layers], axis=-2),
                biases=mx.concatenate([seg.values.biases for seg in layers], axis=-2),
            )
        else:
            merged_keys = mx.concatenate([seg.keys for seg in layers], axis=-2)
            merged_values = mx.concatenate([seg.values for seg in layers], axis=-2)
        return cls(
            keys=merged_keys,
            values=merged_values,
            layer_index=layers[-1].layer_index,
            n_tokens=sum(seg.n_tokens for seg in layers),
            bits=first_bits,
            class_name=layers[-1].class_name,
        )


@dataclass(frozen=True)
class KVRotatingSegment(KVLayerSegment):
    """KV snapshot for a RotatingKVCache layer (ring buffer)."""

    max_size: int = 0
    keep: int = 0
    offset: int = 0
    idx: int = 0          # ring write position (was _idx in old metadata dict)
    class_name: str = "RotatingKVCache"

    def merge_path(self, path: list[KVLayerSegment]) -> KVLayerSegment:
        # Rotating state is not cumulative — only the deepest node's segment matters.
        return path[-1]

    def reconstruct(self, group_size: int) -> Any:
        from mlx_lm.models.cache import RotatingKVCache
        from vllm_mlx.kv_cache import QuantizedArray

        is_quantized = isinstance(self.keys, QuantizedArray)
        if is_quantized:
            dq_keys = mx.dequantize(
                self.keys.packed, self.keys.scales, self.keys.biases,
                group_size=group_size, bits=self.bits,
            )
            dq_values = mx.dequantize(
                self.values.packed, self.values.scales, self.values.biases,
                group_size=group_size, bits=self.bits,
            )
        else:
            dq_keys = self.keys
            dq_values = self.values

        if dq_keys.shape[-2] > self.max_size:
            dq_keys = dq_keys[..., -self.max_size:, :]
            dq_values = dq_values[..., -self.max_size:, :]

        # Rotate back so the ring write position lands at idx, matching live layout.
        if 0 < self.idx < self.max_size and dq_keys.shape[-2] == self.max_size:
            split = self.max_size - self.idx
            dq_keys = mx.concatenate([dq_keys[..., split:, :], dq_keys[..., :split, :]], axis=-2)
            dq_values = mx.concatenate([dq_values[..., split:, :], dq_values[..., :split, :]], axis=-2)

        cache = RotatingKVCache(self.max_size, self.keep)
        cache.keys = dq_keys
        cache.values = dq_values
        cache.offset = self.offset
        cache._idx = self.idx
        return cache


@dataclass
class RecurrentLayerSegment:
    """Immutable recurrent state snapshot for one layer, stored in a TurnNode."""

    arrays: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys:
    #   class_name: str          — e.g. 'MambaCache', 'ArraysCache'
    #   layer_index: int         — position in the live cache list
    #   class_ref: type | None   — concrete mlx-lm class for from_state() reconstruction
    #                              (None when loaded from disk — use class_name fallback)
    scales: list[list[float]] | None = None


@dataclass(frozen=True)
class KVQuantPolicy:
    """Per-layer-type KV cache quantization policy.

    bits_for(class_name) returns the int bit-width to use for that class, or
    None for "store float — do not quantize". Class dispatch is purely structural:
    'RotatingKVCache' uses sliding_bits, any other class whose name ends in
    'KVCache' uses full_bits, everything else returns None (recurrent caches
    are never quantized). The endswith check intentionally accepts subclasses
    like 'BatchKVCache' while rejecting unrelated names.
    """

    sliding_bits: int | None = None       # bf16 by default
    full_bits: int | None = 8             # q8 by default
    # Provenance — True iff the value came from a user CLI override (vs. defaulted).
    sliding_override: bool = False
    full_override: bool = False

    def bits_for(self, class_name: str) -> int | None:
        if class_name == "RotatingKVCache":
            return self.sliding_bits
        if class_name.endswith("KVCache"):
            return self.full_bits
        return None

    def describe(self) -> str:
        def _label(bits: int | None) -> str:
            return "bf16" if bits is None else f"q{bits}"

        s = _label(self.sliding_bits)
        f = _label(self.full_bits)
        if self.sliding_override:
            s += " (user override)"
        if self.full_override:
            f += " (user override)"
        suffix = "" if (self.sliding_override or self.full_override) else " (smart defaults)"
        return f"sliding={s}, full={f}{suffix}"
