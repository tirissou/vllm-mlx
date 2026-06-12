# Domain Glossary

Terms used in architecture discussions and code. See ADRs for decisions that constrain future design.

---

## Cache layer

**PrefixCache** — Protocol: `fetch(request) -> CacheHit | None`, `store`, `release`, `clear`. The seam the Scheduler uses for all cache interaction. Never bypass it to access underlying cache objects directly.

**TurnCacheManager** — `CacheManager` for the conversation-turn trie. Owns the in-memory `TurnPrefixCache`, the optional `CacheDiskStore` for persistence, the `KVQuantPolicy`, and the `request_id → pinned leaf` map (`_pinned_leaves`). Installs spill/promote handlers on the trie at construction. Save/load lifecycle methods (no args) are invoked by `BatchedEngine` on graceful shutdown / startup.

**CacheDiskStore** — Protocol defined in `vllm_mlx/cache_disk_store.py`. Methods: `write(key, payload) -> evicted_keys`, `read(key)`, `read_header(key)` (cheap structural), `delete`, `has`, `touch`, `all_keys`, `get_total_bytes`, `close`. One shipping implementation: `FilesystemCacheDiskStore` (safetensors-mlx + JSON sidecar + JSON index with parent-aware LRU).

**SSDRef** — Sentinel that replaces a node's KV/recurrent payload after spill. Holds only the `NodeKey = (parent_hash, token_ids)` needed to fetch on promote. Defined in `cache_disk_store.py`.

**Spill / promote** — `TurnCacheManager._on_spill(node)` is called by the trie during memory-pressure eviction; it writes the full node payload, sets `node.kv_data = node.recurrent_data = SSDRef(key)`, and decrements `_memory_bytes`. `_on_promote(ssd_ref)` is invoked synchronously by `collect_path_data` when it encounters an SSDRef in the path; on `None` (disk miss) it raises `CacheMissDuringWalk`, which `fetch` catches and converts into a partial hit.

**Active Leaf pinning** — invariant enforced by `TurnCacheManager`. For each in-flight request, exactly one trie node (the deepest currently-relevant leaf) is pinned via `ref_count`. The mapping `request_id → pinned leaf` lives in `TurnCacheManager._pinned_leaves`; the Scheduler never touches it.

**Leaf-only eviction** — invariant enforced by `TurnPrefixCache`. A node is evictable iff `len(children) == 0` AND `ref_count == 0`. Spill respects this too — pinned leaves are never spilled.

**Disk LRU ordering** — `FilesystemCacheDiskStore` enforces two invariants. (1) Parent-aware: an entry whose `parent_key` is itself on disk is not evicted before its children. (2) Leaf-first: among leaf candidates (`child_count == 0`), oldest `last_access_ts` wins. `DiskStoreFullError` is raised when no leaf candidate exists.

---

**Spill / promote contract**

`_on_spill(node) -> bool` — manager-side handler installed on the trie. Writes the node's payload to disk, replaces `node.kv_data` / `node.recurrent_data` with a shared `SSDRef`, and returns `True` on success (node stays in the trie) or `False` on disk-full (trie falls back to drop eviction).

`_on_promote(ssd_ref) -> tuple[list, list] | None` — manager-side handler installed on the trie. Reads the payload from disk and returns `(kv_layers, recurrent_layers)` for the trie to slot back into the node. `None` means a disk miss; the trie drops the node and raises `CacheMissDuringWalk` upward.

---

## Trie storage types

**KVLayerSegment** — immutable snapshot of one transformer layer's KV state stored in a `TurnNode`. Holds `keys: QuantizedArray` and `values: QuantizedArray` in mlx-lm's native group-quantized format (`packed: uint32`, `scales: bfloat16`, `biases: bfloat16`), plus a metadata dict (`layer_index`, `merge_strategy`, `bits` (int or None — the precision used to store this layer), and rotating-cache fields `max_size`/`keep`/`offset`). Not a decode buffer — callers must not treat it as one. Compare with `BatchQuantizedKVCache` (live, mutable) and `QuantizedKVCache` (live, single-sequence).

`KVLayerSegment.concat(layers)` — classmethod. Concatenates a list of same-layer segments along the sequence axis (`axis=-2`) by concatenating `packed`, `scales`, and `biases` arrays independently. Called by `collect_path_data()` for `merge_strategy='concatenate'` (standard KV) layers; rotating layers use `layers[-1]` directly.

**KVQuantPolicy** — immutable dataclass in `cache_types.py`. Maps cache-class name → bits-or-None via `bits_for(class_name)`: `'RotatingKVCache'` → `sliding_bits`, any `'*KVCache'` → `full_bits`, everything else → `None`. Owned by `TurnCacheManager` (one per process, set at construction); consulted by `_segment` at write time. Smart defaults: `sliding_bits=None` (bf16), `full_bits=8` (q8). See ADR-0007.

**RecurrentLayerSegment** — immutable snapshot of one recurrent layer's state stored in a `TurnNode`. Holds raw arrays plus `class_ref` (the concrete mlx-lm class) so `_assemble()` can call `class_ref.from_state(arrays, meta_state)` at reconstruction time without a class-name dispatch table.

**merge_strategy** — metadata field on `KVLayerSegment`. `'concatenate'`: incremental KV slices are concatenated across the trie path (standard `KVCache`). `'last'`: only the deepest node's segment is used (rotating `RotatingKVCache` — ring buffer, not an accumulation).

---

## Scheduler

**step()** — synchronous. Must remain synchronous: MLX lazy ops (dequantize, reconstruct) are enqueued on a stream tied to the worker thread that runs `step()`. Any await in the scheduling path would land reconstruction on the event loop thread, violating the MLX stream constraint.
