# SPDX-License-Identifier: Apache-2.0
"""BatchQuantizedKVCache — batched KV cache in quantized int4 format.

Drop-in replacement for BatchKVCache that immediately quantizes keys/values
to int4 (group_size=64), reducing memory ~4x for cached tokens.

Storage layout per layer:
  keys/values = QuantizedArray(packed, scales, biases)
  packed : [B, H, T, k_head_dim * bits // 32]  dtype=uint32
  scales : [B, H, T, k_head_dim // group_size]  dtype=bfloat16
  biases : [B, H, T, k_head_dim // group_size]  dtype=bfloat16
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache

from .kv_cache import QuantizedArray


class BatchQuantizedKVCache:
    step = 256

    def __init__(self, left_padding: List[int], group_size: int = 64, bits: int = 4):
        self.keys: QuantizedArray | None = None
        self.values: QuantizedArray | None = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self.group_size = group_size
        self.bits = bits

    # ------------------------------------------------------------------
    # Core cache interface
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx
        el_per_int = 8 * mx.uint32.size // self.bits

        if self.keys is None or (prev + num_steps) > self.keys.packed.shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim) -> QuantizedArray:
                return QuantizedArray(
                    packed=mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    scales=mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    biases=mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(qa: QuantizedArray) -> QuantizedArray:
                return QuantizedArray(*[
                    mx.concatenate(
                        [c, mx.zeros((*shape, c.shape[-1]), dtype=c.dtype)], axis=2
                    )
                    for c in qa
                ])

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = QuantizedArray(*[k[..., :prev, :] for k in self.keys])
                    self.values = QuantizedArray(*[v[..., :prev, :] for v in self.values])
                self.keys = expand_quant(self.keys)
                self.values = expand_quant(self.values)
            else:
                self.keys = init_quant(k_head_dim)
                self.values = init_quant(v_head_dim)

        self.offset += num_steps
        self._idx += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(3):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
            self.values[i][..., prev : self._idx, :] = q_values[i]

        return (
            QuantizedArray(*[k[..., : self._idx, :] for k in self.keys]),
            QuantizedArray(*[v[..., : self._idx, :] for v in self.values]),
        )

    def make_mask(
        self,
        N: int,
        window_size: int | None = None,
        return_array: bool = False,
    ):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, window_size=window_size
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        # right_padding unsupported for quantized caches (no dynamic_roll equivalent)

    def finalize(self):
        pass

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return (
            QuantizedArray(*[k[..., : self._idx, :] for k in self.keys]),
            QuantizedArray(*[v[..., : self._idx, :] for v in self.values]),
        )

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.group_size, self.bits)))

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = QuantizedArray(*[k[batch_indices] for k in self.keys])
            self.values = QuantizedArray(*[v[batch_indices] for v in self.values])
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = QuantizedArray(*[k[..., min_left_pad:, :] for k in self.keys])
                self.values = QuantizedArray(*[v[..., min_left_pad:, :] for v in self.values])
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: BatchQuantizedKVCache):
        # Defensive: if caller passes a BatchKVCache (unquantized), convert it
        # on the fly.  This happens when the older mlx-lm BatchGenerator's
        # _process_prompts produces a BatchKVCache that _quantize_batch_kv_cache
        # didn't reach (e.g. via a code path not covered by the monkey-patch).
        if isinstance(other, BatchKVCache):
            other = BatchQuantizedKVCache.from_batch_kvcache(
                other, group_size=self.group_size, bits=self.bits
            )

        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = self.keys.packed.shape[2] if self.keys is not None else 0
        L2 = other.keys.packed.shape[2] if other.keys is not None else 0
        max_size = max(L1, L2)

        ref_keys = self.keys if self.keys is not None else other.keys
        ref_vals = self.values if self.values is not None else other.values

        def _pad_cache(c):
            left = max_idx - c._idx
            if c.keys is None:
                Bc = c.offset.shape[0]
                k_pads = QuantizedArray(*[
                    mx.zeros((Bc, ref_keys[i].shape[1], max_size, ref_keys[i].shape[-1]), dtype=ref_keys[i].dtype)
                    for i in range(3)
                ])
                v_pads = QuantizedArray(*[
                    mx.zeros((Bc, ref_vals[i].shape[1], max_size, ref_vals[i].shape[-1]), dtype=ref_vals[i].dtype)
                    for i in range(3)
                ])
                return k_pads, v_pads, c.offset, c.left_padding + left

            right = max_size - c.keys.packed.shape[2] - left

            def pad_qa(qa: QuantizedArray) -> QuantizedArray:
                result = []
                for comp in qa:
                    r = right
                    if r < 0:
                        comp = comp[..., :r, :]
                        r = 0
                    if left != 0 or r != 0:
                        comp = mx.pad(comp, [(0, 0), (0, 0), (left, r), (0, 0)])
                    result.append(comp)
                return QuantizedArray(*result)

            return pad_qa(c.keys), pad_qa(c.values), c.offset, c.left_padding + left

        sk, sv, so, slp = _pad_cache(self)
        ok, ov, oo, olp = _pad_cache(other)

        self.keys = QuantizedArray(*[mx.concatenate([s, o], axis=0) for s, o in zip(sk, ok)])
        self.values = QuantizedArray(*[mx.concatenate([s, o], axis=0) for s, o in zip(sv, ov)])
        self.offset = mx.concatenate([so, oo])
        self.left_padding = mx.concatenate([slp, olp])
        self._idx = max_idx

    # ------------------------------------------------------------------
    # Extract / Merge
    # ------------------------------------------------------------------

    def extract(self, idx: int) -> QuantizedKVCache:
        cache = VllmQuantizedKVCache(group_size=self.group_size, bits=self.bits)
        padding = self.left_padding[idx].item()
        # QuantizedKVCache expects keys/values as a list of 3 arrays
        cache.keys = [
            mx.contiguous(k[idx : idx + 1, :, padding : self._idx, :])
            for k in self.keys
        ]
        cache.values = [
            mx.contiguous(v[idx : idx + 1, :, padding : self._idx, :])
            for v in self.values
        ]
        cache.offset = self._idx - padding
        return cache

    @classmethod
    def merge(cls, caches: list) -> BatchQuantizedKVCache:
        """Merge a list of QuantizedKVCache objects into a single batched cache."""
        lengths = [c.offset for c in caches]
        max_length = max(lengths)

        if max_length == 0:
            gs = getattr(caches[0], "group_size", 64) if caches else 64
            b = getattr(caches[0], "bits", 4) if caches else 4
            return cls([0] * len(caches), group_size=gs, bits=b)

        padding = [max_length - l for l in lengths]
        group_size = getattr(caches[0], "group_size", 64)
        bits = getattr(caches[0], "bits", 4)

        def merge_component(comp_idx: int, attr: str) -> mx.array:
            arrays = []
            for p, l, c in zip(padding, lengths, caches):
                comp = getattr(c, attr)[comp_idx]  # [1, H, T_alloc, D']
                comp = comp[..., :l, :]            # slice to actual tokens
                if p > 0:
                    pad_shape = list(comp.shape)
                    pad_shape[2] = p
                    comp = mx.concatenate(
                        [mx.zeros(pad_shape, dtype=comp.dtype), comp], axis=2
                    )
                arrays.append(comp)
            return mx.concatenate(arrays, axis=0)

        result = cls(padding, group_size=group_size, bits=bits)
        result.keys = QuantizedArray(*[merge_component(i, "keys") for i in range(3)])
        result.values = QuantizedArray(*[merge_component(i, "values") for i in range(3)])
        result._idx = max_length
        result.offset += max_length  # -padding + max_length = lengths
        return result

    @classmethod
    def from_batch_kvcache(
        cls, bkv, group_size: int = 64, bits: int = 4
    ) -> BatchQuantizedKVCache:
        """Quantize a BatchKVCache in-place, returning a BatchQuantizedKVCache."""
        left_padding = bkv.left_padding.tolist()
        result = cls(left_padding, group_size=group_size, bits=bits)
        result.offset = bkv.offset
        result._idx = bkv._idx
        if bkv.keys is None:
            return result
        k = bkv.keys[..., : bkv._idx, :]
        v = bkv.values[..., : bkv._idx, :]
        result.keys = QuantizedArray(*mx.quantize(k, group_size=group_size, bits=bits))
        result.values = QuantizedArray(*mx.quantize(v, group_size=group_size, bits=bits))
        return result


class VllmQuantizedKVCache(QuantizedKVCache):
    """Single-sequence quantized KV cache returned by BatchQuantizedKVCache.extract().

    Mirrors the mlx-lm pattern: KVCache.merge → BatchKVCache,
    RotatingKVCache.merge → BatchRotatingKVCache.
    """

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)


def make_quantized_cache(model, left_padding, max_kv_size, group_size: int = 64, bits: int = 4):
    """Like mlx-lm's _make_cache but emits BatchQuantizedKVCache for KV layers.

    ArraysCache (recurrent) and RotatingKVCache layers are left unchanged
    so hybrid Mamba+Transformer models work correctly.
    """
    from mlx_lm.models.cache import (
        ArraysCache,
        BatchRotatingKVCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

    def to_quantized_batch(c):
        if type(c) is KVCache:
            return BatchQuantizedKVCache(left_padding, group_size=group_size, bits=bits)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_quantized_batch(sub) for sub in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        return [to_quantized_batch(c) for c in model.make_cache()]
    if max_kv_size is not None:
        from mlx_lm.models.cache import BatchRotatingKVCache
        return [BatchRotatingKVCache(max_kv_size, left_padding) for _ in model.layers]
    return [BatchQuantizedKVCache(left_padding, group_size=group_size, bits=bits)
            for _ in model.layers]


