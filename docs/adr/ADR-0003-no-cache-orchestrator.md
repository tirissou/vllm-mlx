# ADR-0003: Deepen the existing adapter hierarchy instead of extracting a CacheOrchestrator

**Status:** Accepted  
**Date:** 2026-05-19

## Context

`scheduler.py` has 8+ methods that coordinate between `_prefix_cache`, `memory_aware_cache`, and `_ssd_tier`: `_fetch_cache_for_request`, `_try_promote_ssd_for_request`, `_try_promote_ssd_pending`, `promote_from_ssd`, `_reconstruct_ssd_layers`, `_validate_cache`, `_extract_cache_states`, `_reconstruct_cache_from_states`. The two SSD promotion methods are near-identical (~80 lines each, diverging only by a loop wrapper). Cache lifecycle (fetch → promote → validate → use → store → release) is split across five methods with no single home.

The obvious fix is to extract a `CacheOrchestrator` class that coordinates between the three cache objects. We considered this.

## Decision

We rejected `CacheOrchestrator` in favour of deepening the existing adapter hierarchy.

The `PrefixCache` Protocol seam already exists. The problem is that `Scheduler` bypasses it — calling `memory_aware_cache.check_ssd()` directly instead of going through `_prefix_cache.fetch()`. The right fix is to make the adapter's `fetch` genuinely deep, not to add a fourth coordinator on top of three existing objects.

Concrete changes:

- **`SSDOffloadedCache`** — a decorator wrapping any `SpillableCache`. Registers the spill delegate, owns the background promotion loop, exposes `save()`/`load()` via the same `CacheDiskStore`. `_build_prefix_cache` conditionally wraps: `cache = SSDOffloadedCache(inner, store) if ssd_configured else inner`. The Scheduler holds one `PrefixCache` regardless of SSD configuration.
- **`SpillableCache`** — sub-protocol of `PrefixCache` with `set_spill_delegate(on_spill, on_promote)`. Implemented by `MemoryAwarePrefixCache` and `TurnPrefixCache`. The two-part delegate replaces `SSDRef` and `_spill_to_ssd`/`_promote_from_ssd` in `TurnPrefixCache`.
- **`CacheDiskStore`** — shared durable store (`write`, `read`, `all_keys`). Used by both runtime SSD tiering (spill/promote) and startup/shutdown persistence (save/load). Eliminates the duplicated file I/O currently split between `SSDCacheTier` and the persistence path.
- **`validate_cache`** — free function in `kv_cache.py`, called inside the adapter's `fetch`. Scheduler stops seeing invalid caches.

## Consequences

`Scheduler` loses `self.memory_aware_cache`, `self._ssd_tier`, and the eight cache coordination methods. `_schedule_waiting`'s cache section reduces to one `_prefix_cache.fetch(request)` call. `TurnPrefixCache` loses `SSDRef`, `_spill_to_ssd`, `_promote_from_ssd` — SSD-specific code has one home in `SSDOffloadedCache`.

Do not re-propose `CacheOrchestrator`. The coordinator pattern adds a layer without adding depth — it would just wrap three objects that already compose correctly once the adapter is deepened.
