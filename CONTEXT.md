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

**`pinned_leaf(request_id)`** — public observer on `TurnCacheManager` returning the currently pinned `TurnNode` or `None`. The supported way for tests and diagnostics to read pin state; direct access to `_pinned_leaves` is private. Tests verifying eviction-protection should prefer `TurnNode.is_evictable` (already public) over reading `ref_count` directly.

**Leaf-only eviction** — invariant enforced by `TurnPrefixCache`. A node is evictable iff `len(children) == 0` AND `ref_count == 0`. Interior nodes are never evicted. This is what makes Active Leaf safe with O(1) per-request bookkeeping: pinning just the leaf is sufficient because everything above it is protected structurally. Non-cumulative state — `recurrent_data` **and** `sliding_kv_data` — is dropped from interior non-checkpoint nodes when they gain their first child; only permanent checkpoints and active leaves retain it.

---

## Spill delegate contract

**Spill** — move KV arrays out of RAM to disk under memory pressure, leaving an opaque handle in their place.  
**Promote** — restore spilled arrays from disk back into RAM.

`on_spill(tokens, arrays) -> handle` — called by the cache when spilling. Writes to `CacheDiskStore`, returns an opaque handle the cache stores in place of the arrays.

`on_promote(handle) -> list | None` — called by the cache when it needs spilled arrays back. Reads from `CacheDiskStore` using the handle. For `TurnPrefixCache` (intra-cache spilling) this is synchronous, called on access. For `MemoryAwarePrefixCache` (full eviction) it is never called directly — `SSDOffloadedCache` promotes via its background loop on the next `fetch` miss instead.

---

## Trie storage types

**TurnNode storage fields** — `kv_data` holds full-attention (`KVConcatSegment`) layers only; `sliding_kv_data` holds sliding-window (`KVRotatingSegment`) layers; `recurrent_data` holds recurrent state. The latter two are non-cumulative and are retained only at permanent checkpoints + active leaves (see "Leaf-only eviction" and the checkpoint-stride invariant). Each field may independently be a `list`, an `SSDRef`, or `None`.

**KVLayerSegment** — abstract base class in `cache_types.py` for immutable per-layer KV snapshots stored in a `TurnNode`. Two concrete subclasses (`KVConcatSegment`, `KVRotatingSegment`) dispatch path-merge and reconstruction polymorphically. Common fields: `keys` / `values` (either `QuantizedArray` in mlx-lm's native group-quantized format — `packed: uint32`, `scales: bfloat16`, `biases: bfloat16` — or `mx.array` for float precision), `layer_index: int`, `n_tokens: int`, `bits: int | None` (the precision used to store this layer; `None` = bf16). Frozen dataclass. Not a decode buffer — callers must not treat it as one. Compare with `BatchQuantizedKVCache` (live, mutable) and `QuantizedKVCache` (live, single-sequence).

**KVConcatSegment** — `KVLayerSegment` for standard `KVCache` layers (incremental accumulation along the sequence axis). Implements `merge_path(path)` → `concat(path)` and `reconstruct(group_size)` → `BatchQuantizedKVCache.from_quantized_arrays(...)`.

**KVRotatingSegment** — `KVLayerSegment` for `RotatingKVCache` layers (ring buffer). Adds `max_size: int`, `keep: int`, `offset: int`, `idx: int` (the ring write position; was `_idx` in the old metadata dict — the underscore meant nothing). Implements `merge_path(path)` → `path[-1]` (rotating state is not cumulative) and `reconstruct(group_size)` → live `RotatingKVCache` with ring re-rotation back to `idx`.

`KVConcatSegment.concat(layers)` — classmethod. Concatenates a list of same-layer concat segments along the sequence axis (`axis=-2`) by concatenating `packed`, `scales`, and `biases` arrays independently. All segments must agree on `bits`. Only valid for concat segments; the type system prevents calling concat on rotating segments.

**KVQuantPolicy** — immutable dataclass in `cache_types.py`. Maps cache-class name → bits-or-None via `bits_for(class_name)`: `'RotatingKVCache'` → `sliding_bits`, any other name ending in `'KVCache'` (e.g. `'KVCache'`, `'BatchKVCache'`) → `full_bits`, everything else → `None`. Owned by `TurnCacheManager` (one per process, set at construction); consulted by `_segment` at write time. Smart defaults: `sliding_bits=None` (bf16), `full_bits=8` (q8). See ADR-0007.

**RecurrentLayerSegment** — immutable snapshot of one recurrent layer's state stored in a `TurnNode`. Holds raw arrays plus `class_ref` (the concrete mlx-lm class) so `_assemble()` can call `class_ref.from_state(arrays, meta_state)` at reconstruction time without a class-name dispatch table.

**merge_strategy** — superseded. The two-way `'concatenate'` / `'last'` dispatch now lives as polymorphic `merge_path()` on `KVConcatSegment` / `KVRotatingSegment`. The string was the discriminator in the previous metadata-dict design.

---

## Cache translator (`vllm_mlx/cache_translator.py`)

**`segment(live_states, policy, group_size) -> (kv_list, rec_list)`** — translates a list of live mlx-lm cache states into `KVLayerSegment` / `RecurrentLayerSegment` lists for trie storage. Producers: `TurnCacheManager.fetch` (during reconstruction-validate), `store`, `on_prefill_checkpoint`. Constructs the right `KVLayerSegment` subclass based on the live cache's `class_name`. Note: `segment()` / `assemble()` signatures are unchanged regardless of full vs sliding; the **full/sliding partition happens at the trie-write boundary** (`TurnCacheManager.on_prefill_checkpoint`), which separates `KVRotatingSegment` entries into `sliding_kv_data` and all others into `kv_data`. `collect_path_data` sources full KV by concat-merge across the path and sliding KV from the anchor node's `sliding_kv_data` only (non-cumulative). `assemble` reconstructs by `layer_index` regardless of which field the segments came from.

**Segment contract** — emitted segments hold **evaluated, graph-detached** arrays (`mx.eval` then `mx.stop_gradient` on every output). The trie node never retains a reference to the source MLX computation graph or the source float16 Metal buffers via the lazy quantize dependency chain. Callers may delete the source live state and call `mx.clear_cache()`; segment arrays survive. See `prefix_cache_adapters.py:369-371` for the original motivation. ADR-0005 records why this seam exists.

**`assemble(kv_layers, rec_layers, group_size) -> list[cache]`** — inverse of `segment`. Reconstructs live mlx-lm cache objects (`BatchQuantizedKVCache` for full KV, `RotatingKVCache` for rotating, mlx-lm recurrent classes via `class_ref.from_state` for recurrent). Each subclass owns its own reconstruction via `seg.reconstruct(group_size)` — no string dispatch.

**`slice_kv_to_delta(states, prev_end) -> list[dict]`** — pre-segment input shaper. Slices live `KVCache` state arrays to the incremental delta `[prev_end:actual_end]` so each trie node stores only its slice of the path (not the cumulative prefix). RotatingKVCache state is left untouched (its ring buffer is not a cumulative sequence). Called by `store` and `on_prefill_checkpoint` before `segment`.

**Round-trip law** — for any `live_states` consistent with `policy`:

```
assemble(segment(live_states, policy, group_size), group_size) ≈ live_states
```

Equality is approximate for quantized layers (lossy dequant) and exact for float layers. The trie-storage tests in `test_cache_translator.py` and `test_turn_prefix_cache_integration.py` verify this law.

---

## Scheduler

**step()** — synchronous. Must remain synchronous: MLX lazy ops (dequantize, reconstruct) are enqueued on a stream tied to the worker thread that runs `step()`. Any await in the scheduling path would land reconstruction on the event loop thread, violating the MLX stream constraint.

---

## Architecture layer

**Architecture** — abstract class in `vllm_mlx/architectures/base.py`. One subclass per `model_type`, registered via `architectures.register`. Owns every model-specific decision: install lifecycle (`validate`, `install`), capability queries (`turn_end_token_ids`, `extract_attention_query`, `build_mtp_module`, `vision_processor`), and static capability bits (`has_mtp`, `has_vision`, `has_audio`). Engine call sites read `model_wrapper.architecture` and query or call it instead of branching on `model_type` / `isinstance` / `hasattr`. Not to be confused with `prefix_cache_adapters.py`'s `CacheManager` subclasses, which are sometimes called "cache adapters" — those are a separate, older layer. See ADR-0008.

**install_once()** — idempotent function in `vllm_mlx/global_runtime_fixes.py`. Applies architecture-agnostic mlx-lm fixes (the SDPA patches that affect every architecture identically). Called at server bootstrap *and* defensively at the start of `MLXLanguageModel.load()` / `MLXMultimodalLM.load()`. Distinct from `Architecture.install()`, which patches an mlx-lm *model instance* for one specific architecture.

**validate → install → use lifecycle** — the order an `Architecture` is exercised during model load: `lookup(config)` constructs the instance, `validate(model, config, model_path)` raises `UnsupportedModelConfig` on known-unsupported configs (e.g. `RotatingKVCache.keep > 0`) and may clear `mtp_available` when an MTP sidecar file is missing, then `install(model, config, model_path)` performs the actual mlx-lm patching and weight loading. Capability methods (`turn_end_token_ids`, etc.) are only called after `install()` has returned.

**mtp_available** — per-instance `bool` on `Architecture`, distinct from the static `has_mtp: ClassVar[bool]`. `has_mtp` answers "does this architecture support MTP at all?" `mtp_available` answers "did MTP successfully load for *this specific model*?" The scheduler's MTP gate checks `mtp_available`, not `has_mtp`. Set by `validate`/`install` based on sidecar weight file presence. Preserves today's defensive UX: missing sidecar produces a warning and a non-MTP model, not a load failure.

**Capability `None` vs. exception** — capability queries on `Architecture` return `Optional`; `None` means "not supported by this architecture." Call sites log a warning once and degrade. Exceptions on `Architecture` are reserved for state-changing failures (`validate` raises on bad config; `install` raises on failed mutation). This is a deliberate split: capability declarations are not control-flow exceptions.
