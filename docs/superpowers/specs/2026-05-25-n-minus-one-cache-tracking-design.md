# N-1 Cache State Tracking Design

**Date:** 2026-05-25
**Branch:** improve-prefill-speed

## Problem

`compose_n_minus_1_cache` attempts to derive the N-1 cache state from the N-state at request completion. This is:

- **Correct** for standard KV (trim `offset` by 1)
- **Broken** for `RotatingKVCache` — the `trim_last` flag approach cannot correctly reconstruct the N-1 circular buffer state from the N-state alone
- **Probably correct but wasteful** for recurrent (`ArraysCache`) — snapshots every decode step via `extract_cache_states` even for requests that have no cache backend

**Root cause:** trying to reconstruct N-1 from N is fundamentally wrong for RotatingKV. The correct approach is to capture N-state before advancing to N+1.

**Prefill is already correct** — `insert_segments` splits tokens at turn boundaries so each segment ends at exactly the right token. No N-1 adjustment needed for prefill; this design is decode-only.

---

## Design

### Core Idea

Replace `compose_n_minus_1_cache` (derive N-1 from N, broken for RotatingKV) with per-step N-1 tracking: **capture N-state before advancing to N+1** on every decode step. At request completion, `_cache_state.n_minus_one_state` already holds the correct N-1 state — no reconstruction needed.

### Per-Type Tracking Strategy

| Layer type | Mechanism | Cost per decode step |
|---|---|---|
| Standard KV | Nothing — offset−1 derived at completion | Zero |
| Recurrent (`ArraysCache`) | Save Python refs to current arrays before step | O(1) ref copy |
| `RotatingKVCache` | Shadow instance mirrored one step behind | O(1) — one K/V token write per layer |

Snapshotting runs unconditionally for all requests.

**RotatingKV memory cost:** 2× `max_size` buffers per rotating layer per in-flight request (one live, one shadow). Fixed at scheduling, not per-step.

**Recurrent correctness:** `ArraysCache` decode updates are reference replacements (`cache[e] = new_value`), not in-place mutation. Python refs saved before `next()` remain valid after the step. Must be validated before implementation (see Validation section).

**RotatingKV correctness:** `RotatingKVCache._update_in_place` uses true in-place Metal buffer writes (`self.keys[..., idx, :] = keys`). Shared references would see mutations, so two independent instances are required. The shadow instance receives the K/V written to the live instance one step later. Must be validated for wrapped-buffer correctness (see Validation section).

---

## Changes

### `kv_cache.py`

**`RequestCacheState`:**
- Remove `prev_recurrent`
- Add `n_minus_one_state: Any = None` — opaque, fully owned by the adapter

**`CacheIndexMap`** (new dataclass, serialization-friendly — no live MLX objects or class refs):
```python
@dataclass
class CacheIndexMap:
    kv_indices: list[int]
    rotating_indices: list[int]
    recurrent_indices: list[int]
```

Designed for future save/load: plain index lists, cleanly serializable to JSON or pickle.

**`PrefixCache` Protocol** — one new method:
```python
def update_n_minus_one(self, request: Any, prompt_cache: list, uid_idx: int) -> None:
    """Called before each decode step. Default: no-op."""
    ...
```

**Delete:** `compose_n_minus_1_cache`.

---

### `prefix_cache_adapters.py` — `CacheManager` (new base class)

A mixin base class providing N-1 tracking machinery. All adapters inherit from it. It does **not** implement the full `PrefixCache` protocol — `fetch`, `store`, `release`, `get_stats`, `clear`, and `on_prefill_checkpoint` remain adapter-specific.

```
CacheManager                  # N-1 tracking machinery only
    ├── MemoryCacheAdapter    # implements full PrefixCache
    ├── TurnCacheAdapter      # implements full PrefixCache, overrides update_n_minus_one
    ├── PagedCacheAdapter     # implements full PrefixCache
    └── LegacyCacheAdapter    # implements full PrefixCache
```

**Instance state (on `CacheManager`):**
```python
self._cache_index_map: CacheIndexMap | None = None
```
Shared across all requests. Built lazily on first call to either `update_n_minus_one` or `store()`.

**`_ensure_cache_index_map(self, layers: list) -> CacheIndexMap`** (on `CacheManager`):
- Accepts either live `prompt_cache` objects (from `update_n_minus_one`) or extracted state dicts (from `store()`) — both carry sufficient type information (`class_name` on dicts, `isinstance` checks on live objects)
- Classifies indices into `kv_indices`, `rotating_indices`, `recurrent_indices`
- Stores result in `self._cache_index_map` on first call; returns immediately on subsequent calls

**`update_n_minus_one(self, request, prompt_cache, uid_idx) -> None`** (on `CacheManager`, default no-op):
- `MemoryCacheAdapter`, `PagedCacheAdapter`, `LegacyCacheAdapter` inherit this default

**`_reconstruct(self, request, prompt_cache) -> list`** (on `CacheManager`):
- Uses `self._cache_index_map` to interleave sources back into original layer order:
  - KV positions: extracted from `prompt_cache` with `offset − 1`
  - Rotating positions: extracted from shadow instances in `_cache_state.n_minus_one_state`
  - Recurrent positions: extracted from saved refs in `_cache_state.n_minus_one_state`
- Returns complete N-1 cache list in original layer order, ready for `_split_cache_arrays`
- Replaces `compose_n_minus_1_cache`

---

### `prefix_cache_adapters.py` — `TurnCacheAdapter`

Overrides `update_n_minus_one`:
1. Calls `_ensure_cache_index_map(prompt_cache)`
2. If `_cache_state.n_minus_one_state` is None, initialises shadow `RotatingKVCache` instances (one per rotating layer, same `max_size`/`keep` as live instances)
3. **RotatingKV layers:** reads K/V from the live instance at position `_idx - 1` (the slot written at the previous decode step, before this step's `next()` call overwrites it), writes into shadow instance via `_update_in_place` — O(1) per layer
4. **Recurrent layers:** saves Python refs to current `ArraysCache.cache` lists into `_cache_state.n_minus_one_state`
5. **Standard KV layers:** skip

**`store(request, cache)`:**
- Calls `_ensure_cache_index_map(cache)` using the extracted state dicts (in case `update_n_minus_one` was never called, e.g. prefill-only requests)
- If `_cache_state.n_minus_one_state` is None (no decode steps occurred), skips `_reconstruct` and stores `cache` as-is — prefill-only requests are already at the correct N-1 via `insert_segments`
- Otherwise calls `_reconstruct` to get N-1 cache instead of `compose_n_minus_1_cache`

---

### `scheduler.py`

**Decode loop** — before `batch_generator.next()`, for each running request:
```python
self._prefix_cache.update_n_minus_one(request, _gb.prompt_cache, uid_idx)
```

**Remove:**
- `_recurrent_indices` snapshotting block (lines 1651–1661)
- `compose_n_minus_1_cache` call and import (lines 38, 1422–1424)
- `extract_recurrent_state` import (line 40)

---

## Validation Required Before Implementation

1. **Recurrent reference semantics:** confirm `ArraysCache` decode updates replace references rather than mutate in-place. If any path uses in-place indexing (`cache[e][i:j] = val` rather than `cache[e] = new_val`), saved refs would be corrupted. Write a targeted test.

2. **RotatingKV wrapped-buffer correctness:** confirm one-token mirror writes into the shadow instance produce correct N-1 state when the circular buffer has wrapped (i.e. `offset > max_size`). Write a test covering wrapped and unwrapped cases.

3. **Regression tests:** standard KV N-1, RotatingKV N-1 (wrapped and unwrapped), recurrent N-1, hybrid models (mixed rotating + recurrent layers).

---

## What Is Not Changing

- Prefill boundary handling (`insert_segments`, `on_prefill_checkpoint`) — already correct
- `PrefixCache.store()` interface
- `_split_cache_arrays` in `TurnPrefixCache` — still called from `store()` after `_reconstruct`
- `CacheHit`, `CacheDiskStore`, `SpillableCache` protocols
