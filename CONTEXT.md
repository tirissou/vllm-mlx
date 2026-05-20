# Domain Glossary

Terms used in architecture discussions and code. See ADRs for decisions that constrain future design.

---

## Cache layer

**PrefixCache** — Protocol: `fetch(request) -> CacheHit | None`, `store`, `release`, `clear`. The seam the Scheduler uses for all cache interaction. Never bypass it to access underlying cache objects directly.

**SpillableCache** — Sub-protocol of `PrefixCache`. Caches that can move KV arrays out of RAM to disk under memory pressure, leaving a handle in their place. Exposes `set_spill_delegate(on_spill, on_promote)`. Implemented by `MemoryAwarePrefixCache` (full eviction) and `TurnPrefixCache` (intra-cache spilling).

**CacheDiskStore** — Protocol: `write(tokens, layers)`, `read(tokens)`, `all_keys()`. Shared durable store used by both runtime SSD tiering (spill/promote) and startup/shutdown persistence (save/load). Concrete implementation wraps `SSDCacheTier` file I/O.

**SSDOffloadedCache** — Decorator wrapping any `SpillableCache`. Registers the spill delegate, owns the background promotion loop (for full-eviction caches), and exposes `save()`/`load()` for persistence — all via the same `CacheDiskStore`. The Scheduler holds one `PrefixCache` reference regardless of whether SSD is configured.

**Full eviction** — the cache entry leaves the in-memory structure entirely (`MemoryAwarePrefixCache` pattern). Promotion happens via `SSDOffloadedCache`'s background loop on the next `fetch` miss; the Scheduler re-queues the request and it is scheduled one cycle later.

**Intra-cache spilling** — the cache node stays in the in-memory structure with a disk handle in place of its arrays (`TurnPrefixCache` pattern). Promotion happens synchronously via `on_promote` when the node is accessed.

**EvictableCache** — Superseded by `SpillableCache`. Do not use.

---

## Spill delegate contract

**Spill** — move KV arrays out of RAM to disk under memory pressure, leaving an opaque handle in their place.  
**Promote** — restore spilled arrays from disk back into RAM.

`on_spill(tokens, arrays) -> handle` — called by the cache when spilling. Writes to `CacheDiskStore`, returns an opaque handle the cache stores in place of the arrays.

`on_promote(handle) -> list | None` — called by the cache when it needs spilled arrays back. Reads from `CacheDiskStore` using the handle. For `TurnPrefixCache` (intra-cache spilling) this is synchronous, called on access. For `MemoryAwarePrefixCache` (full eviction) it is never called directly — `SSDOffloadedCache` promotes via its background loop on the next `fetch` miss instead.

---

## Scheduler

**step()** — synchronous. Must remain synchronous: MLX lazy ops (dequantize, reconstruct) are enqueued on a stream tied to the worker thread that runs `step()`. Any await in the scheduling path would land reconstruction on the event loop thread, violating the MLX stream constraint.
