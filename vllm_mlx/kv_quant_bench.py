# SPDX-License-Identifier: Apache-2.0
"""KV-cache quantization bench helpers.

Public seam for the bench subcommand (and its tests). This module is
self-contained: it does not import from :mod:`vllm_mlx.mllm_cache` so the
KV-quantization primitives can be reused without pulling in the prefix-cache
machinery.

Exposed symbols:

* :func:`estimate_kv_cache_memory` — shape+dtype memory accounting that does
  not trigger MLX lazy evaluation.
* :func:`_quantize_cache` / :func:`_dequantize_cache` — pair that turn plain
  ``KVCache`` layers into :class:`_QuantizedCacheWrapper` instances and back.
* :class:`_QuantizedCacheWrapper` — opaque holder for the quantized arrays
  alongside the metadata needed to rebuild the original cache type.
"""

from __future__ import annotations

import math
from typing import Any


def _array_memory(arr) -> int:
    """Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing ``.nbytes`` on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute the
    same value.
    """
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """Estimate memory usage of a KV cache in bytes.

    Inspects MLX arrays in the cache and calculates their total memory
    footprint using shape+dtype metadata to avoid triggering lazy
    evaluation (which would cause a VRAM spike).
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict
            keys, values = layer_cache["state"]
            total_bytes += _array_memory(keys)
            total_bytes += _array_memory(values)
        # Handle QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)
            continue
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with state property returning (keys, values)
            try:
                keys, values = layer_cache.state
                total_bytes += _array_memory(keys)
                total_bytes += _array_memory(values)
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

    return total_bytes


class _QuantizedCacheWrapper:
    """Lightweight wrapper storing quantized KV arrays + original cache metadata.

    Unlike ``QuantizedKVCache``, this preserves enough info to reconstruct
    the *original* cache type (KVCache, RotatingKVCache, etc.) on dequantize.
    """

    __slots__ = (
        "keys",
        "values",
        "offset",
        "bits",
        "group_size",
        "orig_type",
        "orig_attrs",
    )

    def __init__(self, layer: Any, bits: int, group_size: int):
        import mlx.core as mx

        self.keys = mx.quantize(layer.keys, group_size=group_size, bits=bits)
        self.values = mx.quantize(layer.values, group_size=group_size, bits=bits)
        mx.eval(self.keys, self.values)
        self.offset = layer.offset
        self.bits = bits
        self.group_size = group_size
        self.orig_type = type(layer)
        # Preserve RotatingKVCache-specific attrs
        self.orig_attrs = {}
        for attr in ("max_size", "keep", "step", "_idx"):
            if hasattr(layer, attr):
                self.orig_attrs[attr] = getattr(layer, attr)


def _quantize_cache(cache: list[Any], bits: int = 8, group_size: int = 64) -> list[Any]:
    """Quantize KV cache layers to reduce memory.

    Only plain KVCache layers are quantized. RotatingKVCache (sliding window)
    is left as-is because its internal _idx/rotation state is tightly coupled
    with update_and_fetch logic and cannot survive quantize/dequantize roundtrip.
    RotatingKVCache is typically small (max_size=1024) so skipping it is fine.
    """
    from mlx_lm.models.cache import KVCache

    quantized = []
    for layer in cache:
        if type(layer) is KVCache and getattr(layer, "keys", None) is not None:
            quantized.append(_QuantizedCacheWrapper(layer, bits, group_size))
        else:
            quantized.append(layer)
    return quantized


def _dequantize_cache(cache: list[Any]) -> list[Any]:
    """Dequantize _QuantizedCacheWrapper layers and copy non-quantized layers.

    All layers are copied (never returned by reference) so that the model's
    ``update_and_fetch`` mutations don't corrupt the stored cache entry.
    """
    import mlx.core as mx

    result = []
    for layer in cache:
        if isinstance(layer, _QuantizedCacheWrapper):
            # Reconstruct original cache type from quantized data
            orig_cls = layer.orig_type
            kv = orig_cls.__new__(orig_cls)
            kv.keys = mx.dequantize(
                *layer.keys, group_size=layer.group_size, bits=layer.bits
            )
            kv.values = mx.dequantize(
                *layer.values, group_size=layer.group_size, bits=layer.bits
            )
            kv.offset = layer.offset
            # Slice the dequantized arrays down to offset so that readers
            # which bypass offset (e.g. Gemma 4 KV-shared layers reading
            # cache.state directly) cannot see stale tokens from a previous
            # request.  Mirrors the plain-KVCache slice in
            # _trim_cache_offset — see issue #384.
            if (
                kv.keys is not None
                and hasattr(kv.keys, "shape")
                and len(kv.keys.shape) >= 3
                and kv.offset < kv.keys.shape[-2]
            ):
                kv.keys = kv.keys[..., : kv.offset, :]
                kv.values = kv.values[..., : kv.offset, :]
            # Restore type-specific attrs (max_size, keep, step, _idx)
            for attr, val in layer.orig_attrs.items():
                setattr(kv, attr, val)
            result.append(kv)
        elif hasattr(layer, "keys") and hasattr(layer, "offset"):
            # Deep-copy non-quantized cache layers (e.g. RotatingKVCache)
            # so model's in-place mutations don't corrupt stored entries
            orig_cls = type(layer)
            kv = orig_cls.__new__(orig_cls)
            kv.keys = mx.array(layer.keys) if layer.keys is not None else None
            kv.values = mx.array(layer.values) if layer.values is not None else None
            kv.offset = layer.offset
            for attr in ("max_size", "keep", "step", "_idx"):
                if hasattr(layer, attr):
                    setattr(kv, attr, getattr(layer, attr))
            result.append(kv)
        else:
            result.append(layer)
    return result
