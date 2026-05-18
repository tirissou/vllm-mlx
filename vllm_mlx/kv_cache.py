# SPDX-License-Identifier: Apache-2.0
"""Core KV cache types shared across the vllm-mlx stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import mlx.core as mx


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

    # Written by PrefixCache.fetch()
    hit_type: str = "miss"
    cache: list | None = None
    cached_tokens: int = 0
    remaining_tokens: list | None = None

    # Written by Scheduler before calling store()
    store_tokens: list | None = None        # N-1 token key
    decoded_cache: list | None = None       # composed N-1 cache (extracted state dicts)
    prev_recurrent: list | None = None      # recurrent-only snapshot before last decode step

    # Written by MemoryCacheAdapter.on_prefill_checkpoint()
    mid_prefill_last_save: int = 0
    mid_prefill_cache_key: tuple | None = None

    # Written by _fetch_cache_for_request for SSD promotion (memory_aware only)
    ssd_candidate: dict | None = None

    # Opaque per-adapter slot (e.g. turn_cache_path for TurnCacheAdapter)
    adapter_state: Any = None


@dataclass
class CacheHit:
    """Returned by PrefixCache.fetch on a successful prefix match."""

    cache: list                    # per-layer KV state
    cached_tokens: int
    remaining_tokens: list         # tokens not yet covered by cache
    handle: Any = None             # opaque value passed back to release()
    hit_type: str = "hit"


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


@runtime_checkable
class PersistableCache(PrefixCache, Protocol):
    """Extension for backends that survive process restart."""

    def save(self, cache_dir: str) -> bool: ...
    def load(self, cache_dir: str) -> int: ...
