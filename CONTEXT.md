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

**Active Leaf pinning** — invariant enforced by `TurnCacheManager`. For each in-flight request, exactly one trie node (the deepest currently-relevant leaf) is pinned via `ref_count`. Mutators: `fetch` pins the matched leaf, `store` and `on_prefill_checkpoint` insert-then-advance (release old leaf, pin new leaf), `release` unpins the current leaf. The mapping `request_id -> pinned leaf` lives in `TurnCacheManager._pinned_leaves`; the Scheduler never touches it.

**Leaf-only eviction** — invariant enforced by `TurnPrefixCache`. A node is evictable iff `len(children) == 0` AND `ref_count == 0`. Interior nodes are never evicted. This is what makes Active Leaf safe with O(1) per-request bookkeeping: pinning just the leaf is sufficient because everything above it is protected structurally.

---

## Spill delegate contract

**Spill** — move KV arrays out of RAM to disk under memory pressure, leaving an opaque handle in their place.  
**Promote** — restore spilled arrays from disk back into RAM.

`on_spill(tokens, arrays) -> handle` — called by the cache when spilling. Writes to `CacheDiskStore`, returns an opaque handle the cache stores in place of the arrays.

`on_promote(handle) -> list | None` — called by the cache when it needs spilled arrays back. Reads from `CacheDiskStore` using the handle. For `TurnPrefixCache` (intra-cache spilling) this is synchronous, called on access. For `MemoryAwarePrefixCache` (full eviction) it is never called directly — `SSDOffloadedCache` promotes via its background loop on the next `fetch` miss instead.

---

## Trie storage types

**KVLayerSegment** — immutable snapshot of one transformer layer's KV state stored in a `TurnNode`. Holds `keys: QuantizedArray` and `values: QuantizedArray` in mlx-lm's native group-quantized format (`packed: uint32`, `scales: bfloat16`, `biases: bfloat16`), plus a metadata dict (`layer_index`, `merge_strategy`, `bits` (int or None — the precision used to store this layer), and rotating-cache fields `max_size`/`keep`/`offset`). Not a decode buffer — callers must not treat it as one. Compare with `BatchQuantizedKVCache` (live, mutable) and `QuantizedKVCache` (live, single-sequence).

`KVLayerSegment.concat(layers)` — classmethod. Concatenates a list of same-layer segments along the sequence axis (`axis=-2`) by concatenating `packed`, `scales`, and `biases` arrays independently. Called by `collect_path_data()` for `merge_strategy='concatenate'` (standard KV) layers; rotating layers use `layers[-1]` directly.

**KVQuantPolicy** — immutable dataclass in `cache_types.py`. Maps cache-class name → bits-or-None via `bits_for(class_name)`: `'RotatingKVCache'` → `sliding_bits`, any other name ending in `'KVCache'` (e.g. `'KVCache'`, `'BatchKVCache'`) → `full_bits`, everything else → `None`. Owned by `TurnCacheManager` (one per process, set at construction); consulted by `_segment` at write time. Smart defaults: `sliding_bits=None` (bf16), `full_bits=8` (q8). See ADR-0007.

**RecurrentLayerSegment** — immutable snapshot of one recurrent layer's state stored in a `TurnNode`. Holds raw arrays plus `class_ref` (the concrete mlx-lm class) so `_assemble()` can call `class_ref.from_state(arrays, meta_state)` at reconstruction time without a class-name dispatch table.

**merge_strategy** — metadata field on `KVLayerSegment`. `'concatenate'`: incremental KV slices are concatenated across the trie path (standard `KVCache`). `'last'`: only the deepest node's segment is used (rotating `RotatingKVCache` — ring buffer, not an accumulation).

---

## Scheduler

**step()** — synchronous. Must remain synchronous: MLX lazy ops (dequantize, reconstruct) are enqueued on a stream tied to the worker thread that runs `step()`. Any await in the scheduling path would land reconstruction on the event loop thread, violating the MLX stream constraint.
