# Cache Adapter Consolidation Design

**Date:** 2026-05-26  
**Branch:** improve-prefill-speed  
**Status:** Approved

## Summary

Remove all cache adapters except `TurnCacheAdapter`. Promote `CacheManager` from a mixin to the Scheduler-facing abstract base class, replacing the `PrefixCache` protocol. Slim `RequestCacheState` to a typed, single-owner dataclass. Make insertion calls explicit.

## Motivation

The `PrefixCache` protocol was a hypothetical seam — one surviving adapter means no real seam. `MemoryCacheAdapter`, `PagedCacheAdapter`, and `LegacyCacheAdapter` are pure pass-throughs that add indirection without depth. `RequestCacheState` has 10+ fields owned by different parties (Scheduler, MemoryCacheAdapter, TurnCacheAdapter), including an `adapter_state: Any` grab-bag. `store_tokens` is set by the Scheduler but read by adapters through the state object instead of being passed explicitly.

## Deletion Scope

**Removed from `prefix_cache_adapters.py`:**
- `MemoryCacheAdapter`
- `PagedCacheAdapter`
- `LegacyCacheAdapter`

**Removed from `RequestCacheState` in `kv_cache.py`:**
- `mid_prefill_last_save` — MemoryCacheAdapter-specific
- `mid_prefill_cache_key` — MemoryCacheAdapter-specific
- `adapter_state: Any` — replaced by typed `turn_path`
- `store_tokens` — moved to explicit parameter on `store()`

**Removed from `Request` in `request.py`:**
- `_turn_cache_path` — fallback attribute that existed because `_cache_state` wasn't guaranteed; no longer needed

**Deprecated (not deleted yet):**
- `PrefixCache` protocol in `kv_cache.py`
- `SpillableCache` protocol in `kv_cache.py`

`_turn_boundaries` stays on `Request` — it is request metadata, not cache state.

## `RequestCacheState`

Single `_cache_state` attribute on `Request`, replacing all scattered cache attributes. Fields split by owner:

```python
@dataclass
class RequestCacheState:
    # Set by Scheduler from CacheHit after fetch()
    hit_type: str = "miss"
    cache: list | None = None
    cached_tokens: int = 0
    remaining_tokens: list | None = None
    prefill_boundaries: list = field(default_factory=list)

    # Set by Scheduler during decode/cleanup pipeline
    decoded_cache: list | None = None
    prev_recurrent: list | None = None

    # Owned by TurnCacheAdapter across its request lifecycle
    turn_path: list = field(default_factory=list)   # list[TurnNode], replaces adapter_state: Any
    n_minus_one_state: dict | None = None
```

`store_tokens` is no longer stored on the state — the Scheduler computes it and passes it directly to `store()`.

## `CacheManager` as Scheduler-facing ABC

`CacheManager` is promoted from a mixin to the abstract base class the Scheduler holds (`self._prefix_cache: CacheManager`). The `PrefixCache` protocol is deprecated.

```python
class CacheManager(ABC):

    # --- Abstract: every adapter must implement ---

    @abstractmethod
    def boundaries(self, request) -> list[int]:
        """Return prefill boundaries adjusted for any cached prefix."""
        ...

    @abstractmethod
    def fetch(self, request) -> CacheHit | None:
        ...

    @abstractmethod
    def store(self, request, tokens: list[int], cache: list) -> bool:
        """Store cache for the completed request. tokens is the N-1 key (set by Scheduler)."""
        ...

    # --- Default no-ops: override as needed ---

    def release(self, handle) -> None:
        pass

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        """Called after each prefill chunk. Adapter decides internally whether to act."""
        pass

    # --- Concrete n-minus-one machinery (inherited, not overridden) ---

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        pass

    def _ensure_cache_index_map(self, layers: list): ...
    def _reconstruct(self, request, extracted_cache: list) -> list: ...
```

The Scheduler calls `self._prefix_cache.boundaries(request)` after `fetch()` (hit or miss) and stores the result into `cs.prefill_boundaries`. It no longer reads `request._turn_boundaries` directly.

## Insertion Interface

Two distinct insertion calls from the Scheduler, reflecting two semantically different operations:

**During prefill — at each chunk boundary:**
```python
cache.on_prefill_checkpoint(request, total_tokens_prefilled, extracted_cache)
```
- `total_tokens_prefilled` is the absolute token count (`cs.cached_tokens + chunk_size`), not the relative chunk offset
- The adapter decides internally whether this count lands on a turn boundary worth inserting
- Cache semantic: `N tokens → cache @ N` — the cache is complete at this boundary, no N-1 adjustment

**After decode completes:**
```python
cache.store(request, tokens, decoded_cache)
```
- `tokens` is the N-1 key, computed by the Scheduler: `prompt + output[:-1]`
- `decoded_cache` has already been through `compose_n_minus_1_cache` before this call
- The N-1 composition is a Scheduler/decode concern only; `TurnCacheAdapter.store()` receives the already-composed cache

## `TurnCacheAdapter`

The only surviving adapter. Clean rewrite with no fallback paths.

**`boundaries(request) -> list[int]`**
```python
def boundaries(self, request) -> list[int]:
    cs = request._cache_state
    cached = cs.cached_tokens if cs else 0
    turn_bds = getattr(request, "_turn_boundaries", []) or []
    return sorted(b - cached for b in turn_bds if b > cached)
```
Scheduler calls this once after fetch (hit or miss). Replaces the two different boundary-setting paths currently in the Scheduler.

**`fetch(request)`**  
Unchanged in logic; reads/writes `cs.turn_path` instead of `cs.adapter_state` / `request._turn_cache_path`. Does not set `prefill_boundaries` (moved to `boundaries()`).

**`store(request, tokens, cache)`**  
Receives `tokens` explicitly. Reads `cs.turn_path` for the matched path. No fallback to `_turn_cache_path`.

**`on_prefill_checkpoint(request, total_tokens_prefilled, extracted_cache)`**  
Receives absolute token count. Checks internally if it's a turn boundary. Reads/writes `cs.turn_path`.

**`messages_to_segments(request)`**  
Stays as an internal `@staticmethod`. Not part of `CacheManager`'s interface. Scheduler never calls it.

## `TurnPrefixCache` interface change

`_split_cache_arrays` is promoted to a public method (rename to `split_cache_arrays`). It is called explicitly by `TurnCacheAdapter` in `store()` and `on_prefill_checkpoint()`, so the underscore was a false privacy claim.

## ADR-0003 Amendment

ADR-0003 ("Deepen the existing adapter hierarchy instead of extracting a CacheOrchestrator") will be amended with an addendum:

- The `PrefixCache` protocol seam is deprecated in favour of `CacheManager` as a concrete ABC
- Rationale: with one surviving adapter (`TurnCacheAdapter`), the Protocol was a hypothetical seam — one adapter equals no real seam
- `boundaries(request)` is added as `@abstractmethod` on `CacheManager`, replacing the Scheduler's direct read of `request._turn_boundaries`
- The original decision (no `CacheOrchestrator`) stands unchanged

## What Does Not Change

- `_turn_boundaries` on `Request` — set by `EngineCore`, read by `TurnCacheAdapter` internally
- `CacheHit` namedtuple — returned by `fetch()`, read by Scheduler
- `SSDOffloadedCache` decorator — still wraps a cache; will reference `CacheManager` once `SpillableCache` is fully removed
- `compose_n_minus_1_cache` and `prev_recurrent` tracking — remain in the Scheduler's decode cleanup pipeline
- `update_n_minus_one` — called by Scheduler before each decode step; concrete on `CacheManager`, overridden by `TurnCacheAdapter`
