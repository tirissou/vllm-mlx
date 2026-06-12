# SSD persistence redesign

**Status:** Draft
**Date:** 2026-06-11
**Companion ADR:** ADR-0007 (to be written alongside the implementation)

## Goal

Replace the current broken-and-unused SSD persistence subsystem (`ssd_cache.py`, `memory_cache.py`'s `save_to_disk`/`load_from_disk`, `TurnPrefixCache`'s inline `_spill_to_ssd`/`_promote_from_ssd`/`save`/`load`) with a single coherent design that:

- Fits the `CacheManager` pattern — disk concerns live at the `TurnCacheManager` seam, not buried in the trie data structure.
- Unifies runtime spill/promote with startup/shutdown save/load through one `CacheDiskStore` protocol and one disk format.
- Preserves the trie's structural dedup on disk (per-trie-node granularity, not per-leaf-path).
- Leverages the self-describing `KVLayerSegment.metadata["bits"]` from the per-layer-type quant spec so the disk format inherits the quant policy automatically.
- Deletes the large body of deprecated code that ADR-0003's deprecation amendment left in place.

## Non-goals

- Cross-version disk-format migration (a `_DISK_FORMAT_VERSION` mismatch is fatal-with-instructions; no migration code).
- Multi-process sharing of a single `cache_dir` (single owner; no file locks).
- Network-backed `CacheDiskStore` implementations (protocol allows them; none shipped).
- Background promotion loop (sync promote on access; ADR-0004 forces synchronous `step()`).
- Per-layer spill decisions (whole-node granularity only).

## Architecture

```
                                    ┌─────────────────────────┐
   Scheduler ──────fetch/store────▶│   TurnCacheManager      │
                                    │   (CacheManager ABC)    │
                                    │                         │
                                    │   _segment / _assemble  │
                                    │   pinned_leaves         │
                                    │   spill / promote       │   ◀── new
                                    │   save / load           │   ◀── new
                                    └────────┬───────┬────────┘
                                             │       │
                       (trie ops, refcounts) │       │ (write / read / delete / all_keys)
                                             │       │
                                             ▼       ▼
                            ┌──────────────────┐   ┌────────────────────────┐
                            │ TurnPrefixCache  │   │ CacheDiskStore         │
                            │ (pure in-mem     │   │ (protocol)             │
                            │  trie + LRU      │   │                        │
                            │  + leaf-only     │   │ FilesystemCacheDisk-   │
                            │  eviction)       │   │ Store (impl): safe-   │
                            │                  │   │ tensors + sidecar JSON │
                            └──────────────────┘   │ + LRU on disk          │
                                                   └────────────────────────┘
```

### Responsibilities

**`TurnCacheManager`** — lifecycle and disk coordination.
- `__init__` accepts `disk_store: CacheDiskStore | None`. When non-`None`, installs a spill handler on the trie and a promote handler on the segment walker.
- `save() -> int` — called by the engine on graceful shutdown; persists every in-memory node not already on disk. Returns nodes written.
- `load() -> int` — called by the engine on startup, before the first scheduling cycle; rebuilds the trie from `disk_store.all_keys()` with `SSDRef` placeholders. Returns nodes restored.
- `_on_spill(node) -> bool` — writes the node's segments to disk, replaces `node.segments` with an `SSDRef`, sets `node.bytes = 0`. Returns `True` (kept in trie) on success, `False` on disk-full so the trie falls back to drop-the-node eviction.
- `_on_promote(ssd_ref) -> list | None` — reads from disk; on `None` the trie drops the node.

**`TurnPrefixCache`** — pure in-memory trie. Disk concerns removed entirely.
- New: `set_spill_handler(handler)`, `walk_all_nodes_preorder()`, `insert_prebuilt(parent_key, token_ids, last_access_ts, n_tokens_cumulative, segments)`, `replace_segments(node, new)`, `restore_segments(node, new)`, `_drop_node(node)`.
- Removed: `SSDRef`, `_spill_to_ssd`, `_promote_from_ssd`, `_ssd_path`, `_tokens_to_node`, `save`, `load`.

**`CacheDiskStore`** — protocol in new `vllm_mlx/cache_disk_store.py`. `FilesystemCacheDiskStore` is the only concrete implementation that ships.

### ADR-0003 alignment

ADR-0003 rejected the `SSDOffloadedCache` decorator because it could not unify two cache shapes (`MemoryAwarePrefixCache`'s full-eviction model and `TurnPrefixCache`'s intra-cache-spilling model). With `MemoryAwarePrefixCache` deprecated by the ADR-0003 addendum, only one cache shape remains, and this design is not a decorator — disk concerns move *up* one level from the trie to the existing `CacheManager` seam. ADR-0003's load-bearing argument no longer applies.

A new ADR-0007 documents this redesign. The `CONTEXT.md` "Cache layer" section is rewritten: the stale `SSDOffloadedCache` and `SpillableCache` paragraphs are removed and replaced with the new shape.

## `CacheDiskStore` protocol

```python
NodeKey = tuple[int, tuple[int, ...]]  # (parent_hash, token_ids_for_this_turn)

class CacheDiskStore(Protocol):
    def write(self, key: NodeKey, payload: NodePayload) -> list[NodeKey]: ...
    # Returns NodeKeys evicted by disk-LRU as a side-effect of this write.
    # Caller is responsible for dropping the corresponding in-memory nodes.

    def read(self, key: NodeKey) -> NodePayload | None: ...
    def read_header(self, key: NodeKey) -> NodeHeader | None: ...
    # Cheap: returns structural fields only, no tensor I/O. Used at load().

    def delete(self, key: NodeKey) -> None: ...
    def has(self, key: NodeKey) -> bool: ...
    def touch(self, key: NodeKey) -> None: ...
    def all_keys(self) -> Iterable[NodeKey]: ...
    def get_total_bytes(self) -> int: ...
    def close(self) -> None: ...
```

### Types

```python
@dataclass(frozen=True)
class NodePayload:
    # Structural fields — define the node's place in the trie.
    parent_key: NodeKey | None        # None means "child of trie root"
    token_ids: tuple[int, ...]        # tokens introduced at this node
    n_tokens_cumulative: int          # prefix length to and including this node
    last_access_ts: float             # in-memory LRU recency, persisted across restart
    # KV state.
    segments: list[KVLayerSegment | RecurrentLayerSegment]


@dataclass(frozen=True)
class NodeHeader:
    parent_key: NodeKey | None
    token_ids: tuple[int, ...]
    n_tokens_cumulative: int
    last_access_ts: float
    size_bytes: int                   # for LRU bookkeeping
    child_count: int                  # for parent-aware LRU eviction


@dataclass(frozen=True)
class SSDRef:
    key: NodeKey
```

### Fields explicitly NOT persisted

- `TurnNode.children: dict[...]` — derived during load via topological insert; each child links into its parent's children dict when inserted.
- `TurnNode.ref_count: int` — always `0` at load (no in-flight requests survive a restart).
- The live `parent: TurnNode` reference — derived from `parent_key` lookup.
- `parent_hash: int` — derived from `parent_key` at insert time via the same deterministic hash function.

## `FilesystemCacheDiskStore` — concrete implementation

- One directory per cache instance (`cache_dir/`).
- Per node, two files: `{node_hash}.safetensors` (all layer tensors flattened with `l{i}_{field}` keys) and `{node_hash}.meta.json` (per-layer metadata and the `NodePayload` structural fields).
- `node_hash` = blake2b of `(parent_hash, token_ids)` truncated to 32 hex chars.
- Single root-level `_index.json`: `{"_DISK_FORMAT_VERSION": 1, "entries": {node_hash: NodeHeader, ...}, "total_bytes": N}`. Eliminates directory-scan-on-startup cost. Updated atomically (write-to-temp, rename).
- **No SQLite.** A trie with thousands of nodes does not need a database; a flat JSON index is debuggable, crash-safe via atomic rename, and avoids the lifecycle complexity of `SSDIndex`.
- **No background writer thread.** Spill is called synchronously from the manager during scheduling (already on a worker thread per ADR-0004); per-node write latency is well inside the existing eviction budget.
- **No quarantine subdir.** Corrupt files are deleted with a WARNING log; treated as a normal cache miss thereafter.
- **Tensor I/O uses `safetensors.mlx.save_file` / `load_file`**, not the NumPy-backed variants. NumPy has no native `bfloat16` and awkward `uint32` semantics — both essential for the quant-spec storage formats.

## Disk format

Storage dtypes are a byte-level pass-through of whatever the in-memory `KVLayerSegment` holds. No re-quantize, no dequantize on spill or promote. Per-layer-class dispatch on disk write/read mirrors `_segment` / `_assemble`.

### Full-attention layers (`KVCache`)

- In-memory: `KVLayerSegment(keys=QuantizedArray(packed: uint32, scales: bfloat16, biases: bfloat16), values=...)` plus `metadata={"bits": int, "group_size": int, "merge_strategy": "concatenate", "layer_index": i, ...}`.
- Disk: 6 tensors per layer (`l{i}_k_packed`, `l{i}_k_scales`, `l{i}_k_biases`, `l{i}_v_packed`, `l{i}_v_scales`, `l{i}_v_biases`) + metadata JSON.
- Reconstruction: `QuantizedArray(packed, scales, biases, group_size=metadata["group_size"], bits=metadata["bits"])`.

### Sliding-window layers (`RotatingKVCache`)

- In-memory: `KVLayerSegment(keys: mx.array(bfloat16), values: mx.array(bfloat16))` plus `metadata={"bits": None, "merge_strategy": "last", "max_size": ..., "keep": ..., "offset": ..., "layer_index": i, ...}`.
- Disk: 2 tensors per layer (`l{i}_k`, `l{i}_v`) + metadata JSON.
- Reconstruction: load tensors as `mx.array`. The rotating-cache fields (`max_size`, `keep`, `offset`) feed `_assemble`'s `RotatingKVCache` construction.

### Recurrent layers (`RecurrentLayerSegment`)

- In-memory: `RecurrentLayerSegment(arrays=[mx.array, ...], meta_state={...}, class_ref=<class ...>)`.
- Disk: raw arrays + metadata JSON. `class_ref` is serialized as a module-qualified name string (`"mlx_lm.models.foo.FooRecurrentCache"`) and resolved on load via `importlib.import_module` + `getattr`.

### Metadata schema requirements

Per-layer metadata JSON must contain the keys listed above for its type. Load validates required keys per layer-class and raises a clear error on missing keys — `_assemble` cannot reconstruct without them, and silent reconstruction with defaults would corrupt state.

### Format versioning

`_DISK_FORMAT_VERSION = 1` covers schema/layout only — not quant choices. Quant choices are self-describing per segment via `metadata["bits"]`. A future quant-policy change does not require a version bump; a schema change does.

### Trie identity stability across restart

`_context_hash(parent_hash, token_ids)` must be deterministic across processes — Python's built-in `hash()` is PYTHONHASHSEED-randomized and unsuitable. Implementation MUST use a deterministic algorithm: blake2b truncated to 64 bits is the chosen primitive. (The existing implementation should be audited and replaced if it uses `hash()`.)

## Spill / promote data flow

### Spill — runtime, under memory pressure

`TurnPrefixCache._evict_if_needed_unlocked` continues to identify evictable leaves in LRU order (Active Leaf + leaf-only-eviction invariants unchanged). New: before dropping the leaf, it calls `spill_handler(node)`:

```
1. key = (node.parent_hash, node.token_ids)
2. payload = NodePayload(
       parent_key=node.parent_key,
       token_ids=node.token_ids,
       n_tokens_cumulative=node.n_tokens_cumulative,
       last_access_ts=node.last_access,
       segments=node.segments,
   )
3. evicted_keys = disk_store.write(key, payload)
4. node.segments = SSDRef(key)
5. node.bytes = 0
6. for evicted_key in evicted_keys:
       drop the matching in-memory node, if any (Section: Error handling, case 3)
7. return True
```

Only `segments` leaves RAM. The `TurnNode` itself stays in the trie with all structural fields intact; only its KV payload becomes a handle. The leaf-only-eviction invariant holds because the trie still only ever picks evictable leaves to spill.

If `disk_store.write` raises `DiskStoreFullError` (Section: Error handling, case 7), the handler returns `False` and the trie falls back to its existing drop-the-node behaviour.

### Promote — synchronous, on access

When `collect_path_data` or `_assemble` walks a node whose segments are an `SSDRef`, it calls `promote_handler(ssd_ref)`:

```
1. segments = disk_store.read(ssd_ref.key)
2. if segments is None:
       trie._drop_node(node)
       raise CacheMissDuringWalk(node)
3. node.segments = segments
4. node.bytes = recompute from materialized arrays
5. trie._evict_if_needed_unlocked()   # may spill another node
```

Promote is synchronous because `step()` is synchronous (ADR-0004). A safetensors read for one node is a few ms — within the existing per-step budget. We do not introduce a background promote loop.

`CacheMissDuringWalk` is caught by `collect_path_data`, which returns the partial path it had collected up to that node. The Scheduler treats this as a normal partial cache hit; the request prefills the missing suffix.

## Save / load lifecycle

### Save

Called by the engine on graceful shutdown. Walks the trie preorder (parents before children) and writes any node not already on disk. Spilled nodes get a `touch()` to refresh LRU but no rewrite.

```python
def save(self) -> int:
    written = 0
    for node in self._trie.walk_all_nodes_preorder():
        key = (node.parent_hash, node.token_ids)
        if isinstance(node.segments, SSDRef):
            self._disk_store.touch(key)
            continue
        payload = NodePayload(
            parent_key=node.parent_key,
            token_ids=node.token_ids,
            n_tokens_cumulative=node.n_tokens_cumulative,
            last_access_ts=node.last_access,
            segments=node.segments,
        )
        self._disk_store.write(key, payload)
        written += 1
    return written
```

Interior (non-leaf) nodes are written too. Leaf-only-eviction is a *runtime* spill rule, not a persistence rule — at shutdown nothing is in-flight, so writing the full trie is safe and necessary.

### Load

Called by the engine on startup, before the first scheduling cycle. Reads headers only — no tensor I/O — and rebuilds the in-memory trie with `SSDRef` placeholders. Tensors are read lazily on first access.

```python
def load(self) -> int:
    headers = [(key, self._disk_store.read_header(key))
               for key in self._disk_store.all_keys()]
    self._validate_compat(headers)                 # version + policy checks
    ordered = self._topo_sort_by_parent(headers)   # parents before children
    restored = 0
    for key, header in ordered:
        self._trie.insert_prebuilt(
            parent_key=header.parent_key,
            token_ids=header.token_ids,
            n_tokens_cumulative=header.n_tokens_cumulative,
            last_access_ts=header.last_access_ts,
            segments=SSDRef(key),
        )
        restored += 1
    return restored
```

`insert_prebuilt` is the only new structural method on `TurnPrefixCache`. It shares the trie-insert code path with the normal `insert`, accepting an explicit `last_access_ts` instead of stamping `now()` and skipping the KV-state argument (passing the `SSDRef` directly).

### Compatibility validation at load

Before populating the trie, `load()` validates:

1. `_index.json` exists and `_DISK_FORMAT_VERSION == 1` — else `IncompatibleCacheDirError`.
2. Every header's per-layer `bits` (and `group_size` where applicable) matches the current `KVQuantPolicy` — else `CachePolicyMismatchError`. Mixing policies in one trie would corrupt assembly.
3. For recurrent layers, every persisted `class_ref` path resolves via `importlib.import_module` — else `MissingCacheClassError`.

All three errors are fatal at startup. The engine does not silently ignore a non-empty cache dir.

### Engine plumbing

`BatchedEngine.save_cache_to_disk(cache_dir)` and `load_cache_from_disk(cache_dir)` (lines 1296–1315) are rewired:
- The `cache_dir` argument is removed; the path is configured once via `SchedulerConfig.kv_cache_disk_dir` at engine construction.
- Both methods call `turn_cache_manager.save()` / `load()` respectively.
- The `MemoryAwarePrefixCache.save_to_disk`/`load_from_disk` path is deleted.

## Config surface

New `SchedulerConfig` fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `kv_cache_disk_dir: str \| None` | `None` | Enables persistence + spill when set. `None` = pure in-memory cache with eviction-on-pressure (today's no-SSD behaviour). |
| `kv_cache_disk_max_bytes: int \| None` | `None` | Disk LRU cap. `None` = unbounded (caller responsible for monitoring). |
| `kv_cache_load_on_startup: bool` | `True` | Escape hatch for tests / cold-start benchmarks. |
| `kv_cache_save_on_shutdown: bool` | `True` | Symmetric escape hatch. |

CLI flags mirror the names: `--kv-cache-disk-dir`, `--kv-cache-disk-max-bytes`, `--kv-cache-load-on-startup`, `--kv-cache-save-on-shutdown`. Any legacy `--ssd-*` flag gets a migration error directing the user to the new names.

## Error handling

1. **`read` returns `None` (entry missing).** Treat as a cache miss. Trie's `_drop_node` propagates structural cleanup. `collect_path_data` returns the partial path; Scheduler prefills the missing suffix. No new error path.
2. **Corrupt safetensors on read.** `FilesystemCacheDiskStore.read` catches the safetensors exception, logs WARNING with file path, calls `self.delete(key)`, returns `None`. Caller cannot distinguish from case 1.
3. **Disk LRU eviction during write.** `write` returns the list of `NodeKey`s it evicted. `_on_spill` walks the in-memory trie; any evicted key whose node still holds an `SSDRef` gets dropped via `_drop_node`. One INFO log per dropped node.
4. **Load-time format mismatch.** `_DISK_FORMAT_VERSION != 1` → `IncompatibleCacheDirError(path, found, expected)`. Fatal at startup.
5. **Load-time quant-policy mismatch.** Persisted `bits` per layer disagrees with current `KVQuantPolicy` → `CachePolicyMismatchError`. Fatal.
6. **Class ref no longer resolvable.** `importlib.import_module` fails on a persisted recurrent class path → `MissingCacheClassError(class_path)`. Fatal.
7. **Disk full while writing.** First attempt LRU eviction to make headroom. If still full after evicting down to a minimum-floor, raise `DiskStoreFullError(needed, available)`. `_on_spill` catches, returns `False`, trie falls back to drop-the-node eviction. Rate-limited WARNING log.
8. **Crash mid-write.** Atomic write-to-temp-then-rename for both `.safetensors` and `_index.json`. On next startup, if a `.safetensors` exists without a corresponding `_index.json` entry, rebuild from the sidecar `.meta.json`. If sidecar is also missing/corrupt, delete the orphan safetensors with a WARNING.
9. **Orphan child at load.** A child whose `parent_key` doesn't resolve to a loaded node is dropped with a WARNING. Defense in depth — the parent-aware LRU eviction (below) should prevent this, but the load-time guard catches any leak.

The three custom exceptions inherit from a shared `CacheDiskError` base in `cache_disk_store.py`.

## Disk LRU ordering

Two invariants:

1. **Parent-aware.** An entry whose `parent_key` is itself still on disk must not be evicted before its children. The index tracks per-entry `child_count` (incremented on child `write`, decremented on child `delete`) to enforce this in O(1) per candidate.
2. **Leaf-first.** Among entries with `child_count == 0`, oldest `last_access_ts` wins.

If no leaf candidate exists (every entry has on-disk children), the disk store cannot evict; `write` proceeds to error case 7. This guarantees the on-disk trie is always a closed subtree — no orphans, no surprises at load.

## Cleanup / deletion list

| File | Action | LOC |
|------|--------|-----|
| `vllm_mlx/ssd_cache.py` | Delete entirely | ~1131 |
| `vllm_mlx/memory_cache.py` | Delete entirely (deprecated per ADR-0003 addendum) | ~1418 |
| `vllm_mlx/turn_prefix_cache.py` | Remove `SSDRef`, `_spill_to_ssd`, `_promote_from_ssd`, `_ssd_path`, `_tokens_to_node`, `save`, `load`. Add `set_spill_handler`, `walk_all_nodes_preorder`, `insert_prebuilt`, `replace_segments`, `restore_segments`, `_drop_node`. Net ~−350 LOC; final size ~540 LOC | — |
| `vllm_mlx/prefix_cache_adapters.py` | `TurnCacheManager.__init__` accepts `disk_store`. Adds `save`, `load`, `_on_spill`, `_on_promote`, `_build_payload`. Net +~200 LOC | — |
| `vllm_mlx/cache_disk_store.py` | **New.** Protocol, types, `FilesystemCacheDiskStore`, custom exceptions. ~400 LOC | — |
| `vllm_mlx/engine/batched.py` | Rewire `save_cache_to_disk` / `load_cache_from_disk` to call manager methods; remove `cache_dir` argument | — |
| `vllm_mlx/scheduler.py` | Add the four new config fields; construct `FilesystemCacheDiskStore` in `_build_prefix_cache` when configured | — |
| `vllm_mlx/cli.py` | Add the four new CLI flags; migration error for legacy `--ssd-*` flags | — |
| `tests/test_ssd_cache.py` | Delete | ~1000+ |
| `tests/test_cache_disk_store.py` | Rewrite from scratch against the new protocol | — |
| `tests/test_batched_engine.py` | Update assertions (lines 114, 128) | — |
| `CONTEXT.md` | Rewrite "Cache layer" section: remove stale `SSDOffloadedCache`/`SpillableCache` paragraphs, describe new shape | — |
| `docs/adr/ADR-0007-persistence-redesign.md` | **New** | — |

**Net:** ~3500 LOC out, ~600 LOC in.

## Testing strategy

### Unit — `tests/test_cache_disk_store.py` (new)

- `write` → `read` round-trip for each `NodePayload` shape (full q8, sliding bf16, recurrent). Assert tensor dtype, shape, and metadata survive bit-exact.
- `safetensors.mlx` path verified specifically with `bfloat16` and `uint32` tensors.
- LRU eviction: write `N+1` entries past `max_disk_bytes`, assert leaf-first ordering, assert no parent-before-child eviction. Reproduce case 7 by forcing a full disk.
- Corrupt-file recovery: write entry, truncate file, assert `read` returns `None` and the entry is deleted.
- Crash recovery: simulate partial `.safetensors.tmp` left behind; assert `_rebuild_index_if_stale` recovers.

### Integration — extend `tests/test_turn_prefix_cache_integration.py`

- Spill/promote round-trip: insert N nodes, force memory pressure, assert each evictable leaf becomes an `SSDRef`; access each, assert promote restores arrays bit-exact and trie state.
- Save/load round-trip at process boundary: build a trie, call `save()`, construct a fresh `TurnCacheManager` against the same disk dir, call `load()`, assert every node is restored with `SSDRef` placeholders, then assert `fetch` for any persisted prefix promotes the right tensors.
- Quant-policy mismatch on load: save with one `KVQuantPolicy`, attempt load with a different one, assert `CachePolicyMismatchError`.
- Active Leaf + leaf-only-eviction invariants hold across spill: spill while a leaf is pinned, assert the pinned leaf is not selected.

### End-to-end — extend `tests/test_cache_hit_oom_repro.py`

Variant that runs the existing 60k-token Gemma 4 26B A4B scenario with `kv_cache_disk_dir` set; asserts Metal peak memory stays under budget; asserts the disk_dir grows and shrinks as expected. Skipped on non-Metal hardware.

## Out of scope

- Cross-version migration (`_DISK_FORMAT_VERSION` bumps in the future).
- Multi-process sharing of a single `cache_dir`.
- Cloud / network-backed `CacheDiskStore` implementations.
- Per-layer spill granularity.
- Background promotion loops.
