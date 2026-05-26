# SPDX-License-Identifier: Apache-2.0
"""Core KV cache types shared across the vllm-mlx stack."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import mlx.core as mx

logger = logging.getLogger(__name__)


class QuantizedArray(NamedTuple):
    """Typed wrapper around MLX's int4/int8 quantized tensor representation.

    NamedTuple so mx.eval traverses it as a pytree — critical for chunked prefill.
    """

    packed: mx.array   # uint32
    scales: mx.array   # bfloat16
    biases: mx.array   # bfloat16

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes + self.biases.nbytes

    def as_tuple(self):
        return (self.packed, self.scales, self.biases)


@dataclass
class RequestCacheState:
    """All cache-related state for a single request. Lives at request._cache_state."""

    # Set by Scheduler from CacheHit after fetch()
    hit_type: str = "miss"
    cache: list | None = None
    cached_tokens: int = 0
    remaining_tokens: list | None = None
    prefill_boundaries: list = field(default_factory=list)

    # Set by Scheduler during decode / cleanup pipeline
    decoded_cache: list | None = None
    prev_recurrent: list | None = None   # N-1 recurrent snapshot; was set dynamically before

    # Owned by TurnCacheAdapter across the request lifecycle
    turn_path: list = field(default_factory=list)   # list[TurnNode]; typed replacement for adapter_state
    n_minus_one_state: Any = None                   # per-step N-1 tracking (set by update_n_minus_one)


@dataclass
class CacheIndexMap:
    """Index classification for a model's cache layer list.

    Designed for future save/load: plain index lists, cleanly serializable.
    """
    kv_indices: list[int]
    rotating_indices: list[int]
    recurrent_indices: list[int]


@dataclass
class CacheHit:
    """Returned by PrefixCache.fetch on a successful prefix match."""

    cache: list                    # per-layer KV state
    cached_tokens: int
    remaining_tokens: list         # tokens not yet covered by cache
    handle: Any = None             # opaque value passed back to release()
    hit_type: str = "hit"
    prefill_boundaries: list = field(default_factory=list)


@runtime_checkable
class PrefixCache(Protocol):
    """Seam between Scheduler and all prefix cache backends."""

    def fetch(self, request) -> CacheHit | None: ...
    def store(self, request, cache: list) -> bool: ...
    def release(self, handle: Any) -> None: ...
    def get_stats(self) -> dict: ...
    def clear(self) -> None: ...
    def on_prefill_checkpoint(
        self, request: Any, processed_tokens: int, extracted_cache: list
    ) -> None:
        """Called after each prefill chunk with the extracted cache state.

        extracted_cache is a list of dicts as returned by
        Scheduler._extract_cache_states(). Most adapters implement this as
        a no-op; MemoryCacheAdapter stores prefix entries, TurnCacheAdapter
        captures boundary states.
        """
        ...

    def update_n_minus_one(
        self, request: Any, prompt_cache: list, uid_idx: int
    ) -> None:
        """Called before each decode step. Default: no-op."""
        ...


@runtime_checkable
class PersistableCache(PrefixCache, Protocol):
    """Extension for backends that survive process restart."""

    def save(self, cache_dir: str) -> bool: ...
    def load(self, cache_dir: str) -> int: ...


@runtime_checkable
class CacheDiskStore(Protocol):
    """Shared durable store for SSD tiering and save/load persistence.

    Used by SSDOffloadedCache for both runtime spill/promote and
    startup/shutdown save/load. See CONTEXT.md for design rationale.
    """

    def write(self, tokens: tuple[int, ...], layers: list) -> None: ...
    def read(self, tokens: tuple[int, ...]) -> list | None: ...
    def has(self, tokens: tuple[int, ...]) -> bool: ...
    def all_keys(self) -> Iterable[tuple[int, ...]]: ...


@runtime_checkable
class SpillableCache(PrefixCache, Protocol):
    """PrefixCache that can delegate array storage to an external durable store.

    Implemented by MemoryAwarePrefixCache (full eviction) and TurnPrefixCache
    (intra-cache spilling). See CONTEXT.md for the distinction.
    """

    def set_spill_delegate(
        self,
        on_spill: Callable[[tuple[int, ...], list], Any],
        on_promote: Callable[[Any], list | None],
    ) -> None: ...


def validate_cache(cache: Any) -> bool:
    """Validate that a cache object is usable.

    Checks for None references AND shape compatibility.  Restored cache entries
    must have batch_size == 1 (single sequence) so they can be merged into the
    running batch by _merge_caches.  A shape mismatch (e.g. batch=2 from a
    stale entry) would cause a concatenation crash inside _merge_caches.

    Args:
        cache: The cache object to validate

    Returns:
        True if cache is valid and usable
    """
    if cache is None:
        return False

    # Check if it's a list of cache layers
    if isinstance(cache, list):
        if len(cache) == 0:
            return False
        for layer_cache in cache:
            if layer_cache is None:
                return False
            # Check if layer has expected structure
            if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                return False
            if hasattr(layer_cache, "values") and layer_cache.values is None:
                return False
            # Validate batch dimension == 1 for KVCache layers
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                # QuantizedKVCache.keys is a (packed, scales, biases) tuple
                keys_arr = (
                    layer_cache.keys[0]
                    if isinstance(layer_cache.keys, (tuple, list))
                    else layer_cache.keys
                )
                if keys_arr.shape[0] != 1:
                    logger.debug(
                        "validate_cache: batch dim mismatch on keys layer, shape=%s",
                        keys_arr.shape,
                    )
                    return False
            # Validate batch dimension for MambaCache layers
            if hasattr(layer_cache, "cache") and isinstance(layer_cache.cache, list):
                for arr in layer_cache.cache:
                    if arr is not None and arr.shape[0] != 1:
                        logger.debug(
                            "validate_cache: batch dim mismatch on mamba layer, shape=%s",
                            arr.shape,
                        )
                        return False

    # Check BatchKVCache structure
    if hasattr(cache, "caches"):
        if cache.caches is None:
            return False
        for c in cache.caches:
            if c is None:
                return False

    return True


def extract_layer_state(layer) -> dict | None:
    """Extract state dict from a single KV cache layer.

    Converts raw KVCache or BatchKVCache objects to normalized dict form with
    state, meta_state, class_name, and class_ref. Returns None if layer lacks
    the required .state and .meta_state attributes.
    """
    if hasattr(layer, "state") and hasattr(layer, "meta_state"):
        return {
            "state": layer.state,
            "meta_state": layer.meta_state,
            "class_name": type(layer).__name__,
            "class_ref": type(layer),
        }
    return None


def _build_batch_kv_types() -> tuple:
    """Return a tuple of batch-level KV cache types for isinstance checks.

    Covers both standard and quantized batch caches. Used to identify
    recurrent (non-KV) layers in a batch generator's prompt_cache list.
    """
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, QuantizedKVCache
    try:
        from .batch_quantized_kv_cache import BatchQuantizedKVCache
        return (BatchKVCache, BatchRotatingKVCache, QuantizedKVCache, BatchQuantizedKVCache)
    except ImportError:
        return (BatchKVCache, BatchRotatingKVCache, QuantizedKVCache)


_BATCH_KV_TYPES: tuple = _build_batch_kv_types()


def extract_recurrent_state(cache: list) -> list:
    """Return only the non-KV layers from a live cache list.

    KV layers have an `offset` attribute and a `keys` attribute.
    Recurrent layers (Mamba, DeltaRNN) have neither.
    """
    from mlx_lm.models.cache import KVCache, BatchKVCache, RotatingKVCache, QuantizedKVCache
    try:
        from .batch_quantized_kv_cache import BatchQuantizedKVCache as _QuantizedCacheWrapper
        kv_types = (KVCache, BatchKVCache, RotatingKVCache, QuantizedKVCache, _QuantizedCacheWrapper)
    except ImportError:
        kv_types = (KVCache, BatchKVCache, RotatingKVCache, QuantizedKVCache)
    return [layer for layer in cache if not isinstance(layer, kv_types)]


def _is_kv_extracted(layer: dict) -> bool:
    """True if an extracted state dict represents a KV (not recurrent) layer."""
    name = layer.get("class_name", "")
    return "KV" in name or "Quantized" in name or name == "RotatingKVCache"


def extract_cache_states(raw_cache: list) -> list:
    """Extract actual tensor state from each layer cache.

    This extracts the real KV data using mlx-lm's cache.state property,
    allowing the data to be stored and reconstructed later even after
    the BatchGenerator is recreated.

    Returns:
        List of dicts with {state, meta_state, class_name, class_ref}, or []
        if any layer fails extraction.
    """
    if not raw_cache:
        return []

    extracted = []
    for i, layer_cache in enumerate(raw_cache):
        try:
            d = extract_layer_state(layer_cache)
            if d is not None:
                extracted.append(d)
        except Exception as e:
            logger.warning(
                f"Failed to extract state from cache layer {i}/{len(raw_cache)} "
                f"(type={type(layer_cache).__name__}): {e}"
            )

    if len(extracted) != len(raw_cache):
        logger.warning(
            f"extract_cache_states partial: {len(extracted)}/{len(raw_cache)} layers succeeded"
        )
        return []
    return extracted


def reconstruct_cache_from_states(extracted_states: list) -> list | None:
    """Reconstruct cache objects from extracted cache states.

    Inverse of extract_cache_states(). Uses mlx-lm's _BaseCache.from_state()
    to reconstruct any cache type (KVCache, MambaCache, etc.).

    Returns:
        List of cache objects, or None if reconstruction fails.
    """
    if not extracted_states:
        return None

    try:
        caches = []
        for layer_state in extracted_states:
            state = layer_state.get("state")
            meta_state = layer_state.get("meta_state")
            cache_cls = layer_state.get("class_ref")
            if state is None:
                return None

            if cache_cls is not None and hasattr(cache_cls, "from_state"):
                from mlx_lm.models.cache import (
                    BatchKVCache as _BatchKVCache,
                    KVCache as _KVCache,
                )
                if cache_cls is _BatchKVCache:
                    keys, values = state[0], state[1]
                    cache = _KVCache()
                    cache.keys = keys
                    cache.values = values
                    cache.offset = keys.shape[2]
                else:
                    cache = cache_cls.from_state(state, meta_state)
            else:
                from mlx_lm.models.cache import KVCache

                if len(state) != 2:
                    return None
                cache = KVCache()
                cache.keys, cache.values = state
                cache.offset = (
                    int(meta_state[0]) if meta_state else cache.keys.shape[2]
                )

            caches.append(cache)

        return caches

    except Exception as e:
        logger.info(f"[mid_prefill_cache] reconstruct EXCEPTION: {e}")
        return None


def reconstruct_ssd_layers(layer_dicts: list) -> list | None:
    """Reconstruct cache objects from deserialized SSD layer dicts.

    Converts numpy arrays back to MLX arrays and creates KVCache objects.
    """
    try:
        from mlx_lm.models.cache import ArraysCache, KVCache

        result = []
        for ld in layer_dicts:
            if "keys" in ld and "values" in ld:
                kv = KVCache()
                kv.keys = mx.array(ld["keys"])
                kv.values = mx.array(ld["values"])
                kv.offset = ld["offset"]
                for attr in ("max_size", "keep", "step", "_idx"):
                    if attr in ld:
                        setattr(kv, attr, ld[attr])
                result.append(kv)
            elif "state" in ld:
                state_arrays = [mx.array(a) for a in ld["state"]]
                layer_obj = ArraysCache(len(state_arrays))
                layer_obj.state = state_arrays
                result.append(layer_obj)
            else:
                logger.warning(
                    f"[ssd_promote] unknown layer dict format: {list(ld.keys())}"
                )
                return None
        return result
    except Exception as e:
        logger.warning(f"[ssd_promote] reconstruction failed: {e}")
        return None
