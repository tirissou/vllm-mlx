# ADR-0005: Store KV segments in mlx-lm native group-quantized format

**Status:** Accepted  
**Date:** 2026-05-29

## Context

`TurnNode` stores KV state as `StaticKVData` — per-tensor int8 quantized arrays produced by `CacheTranslator.quantize_kv`. At reconstruction (`collect_path_data`), each node's arrays are dequantized to bfloat16 before concatenation, producing a merged `StaticKVData` with `scales=[1.0, 1.0]` (a sentinel meaning "already bfloat16"). `_assemble` then calls `dequantize_kv` a second time (a no-op) and passes state dicts to `reconstruct_cache_from_states`, which constructs plain `BatchKVCache` objects. The live decode cache is `BatchQuantizedKVCache` (group-int8), so the model requantizes again at the first decode step.

The full round-trip on a cache hit is: `int8 → dequantize → bfloat16 → concat → dequantize (no-op) → BatchKVCache (bf16) → requantize → BatchQuantizedKVCache`. The `scales=[1.0, 1.0]` sentinel leaks quantization state across the `collect_path_data` seam; callers must know it means "already bfloat16, skip dequantize." The intermediate bfloat16 buffers remain live in Metal memory until the first decode step triggers `mx.eval`.

We evaluated two options:

**Option A** — keep per-tensor int8 in the trie; re-quantize to group-int8 inside `_assemble` at reconstruction time. One-time double-quantize penalty per hit; `scales=[1.0, 1.0]` sentinel remains; bfloat16 intermediate still allocated.

**Option B** — store mlx-lm's native group-quantized format (`QuantizedArray(packed: uint32, scales: bfloat16, biases: bfloat16)`) in the trie from the start; concatenate along the sequence axis at reconstruction with no dequantize step; feed directly into `BatchQuantizedKVCache.from_quantized_arrays()`.

The packed-concat concern (uint32 arrays cannot be naively concatenated) does not apply: `mx.quantize` packs along `head_dim` (last axis), so segments are concatenated along the sequence axis (`axis=-2`), which is straightforward — `packed`, `scales`, and `biases` each concatenate independently with no unpacking.

## Decision

Option B. Store `KVLayerSegment(keys: QuantizedArray, values: QuantizedArray)` in trie nodes from the moment of insertion. `CacheTranslator` is deleted. `reconstruct_cache_from_states` is deleted. The seam at `collect_path_data` becomes clean: it returns `list[KVLayerSegment]` with no bfloat16 intermediates and no sentinel conventions.

Concrete changes:

- **`KVLayerSegment`** — new type in `cache_types.py`. Replaces `StaticKVData`. Holds `keys: QuantizedArray`, `values: QuantizedArray`, metadata. `KVLayerSegment.concat(layers)` classmethod owns all QuantizedArray concatenation; `TurnPrefixCache` has no knowledge of quantized array internals.
- **`RecurrentLayerSegment`** — replaces `StaticRecurrentData`. Stores `class_ref` so `_assemble` can call `class_ref.from_state()` directly, fixing a latent bug where the missing `class_ref` caused recurrent reconstruction to silently fall through to a `KVCache` fallback.
- **`BatchQuantizedKVCache.from_quantized_arrays()`** — new classmethod. Accepts pre-merged `QuantizedArray` keys and values and constructs a `BatchQuantizedKVCache` without going through `update_and_fetch`. Closes the construction gap; direct instance-variable assignment is not permitted.
- **`QuantizedKVCache.merge` patched** — `VllmQuantizedKVCache` is deleted. `QuantizedKVCache.merge` is patched at module level in `batch_quantized_kv_cache.py` to return `BatchQuantizedKVCache`, following mlx-lm's `KVCache` ↔ `BatchKVCache` pattern exactly. `BatchQuantizedKVCache.extract()` returns a plain `QuantizedKVCache`.
- **`_assemble` three-path inline** — standard KV via `from_quantized_arrays`; rotating KV via dequantize → `RotatingKVCache` (no quantized rotating type exists in mlx-lm); recurrent via `class_ref.from_state`. `reconstruct_cache_from_states` in `turn_prefix_cache.py` is deleted; the dead import in `scheduler.py` is removed.
- **Single `mx.eval`** — after reconstruction, one `mx.eval(*arrays_to_eval)` call materializes all lazy concatenation graphs before decode starts, handling both bare `mx.array` (RotatingKVCache, recurrent) and `QuantizedArray` fields (`packed`, `scales`, `biases`).
- **`_CACHE_FORMAT_VERSION` bumped to 5** — persisted trie files in per-tensor int8 format (versions ≤ 4) are rejected at load.

## Consequences

The `scales=[1.0, 1.0]` sentinel is eliminated. `CacheTranslator` is deleted. `reconstruct_cache_from_states` is deleted. `VllmQuantizedKVCache` is deleted. The hit path emits `QuantizedKVCache` objects directly — no bfloat16 intermediate, no double-quantize. Rotating KV layers still dequantize at reconstruction (unavoidable: no quantized rotating cache type in mlx-lm).

Do not re-introduce per-tensor int8 storage (Option A). The double-quantize compounds rounding error and the bfloat16 intermediate negates the memory benefit during the reconstruction window.

---

## Addendum (2026-06-23) — `sliding_kv_data` field and cache format v6

`TurnNode` now stores sliding-window (rotating) KV segments in a dedicated `sliding_kv_data: list[KVLayerSegment] | SSDRef | None` field, separate from `kv_data` (full-attention layers only). The partition happens at the trie-write boundary in `TurnCacheManager.on_prefill_checkpoint`: `KVRotatingSegment` entries go to `sliding_kv_data`, all other `KVLayerSegment` entries go to `kv_data`. `assemble()` is unchanged — it reconstructs by `layer_index` regardless of which field the segments originated in.

**Persistence (format version 6):** the `nodes` SQLite table gains a `sliding_file_path TEXT` column. Each node with non-empty `sliding_kv_data` gets its own `sliding_{i}.safetensors` file and a matching `_meta.json`. The KV serialization helpers `_write_kv_segment_arrays` / `_read_kv_segment_arrays` are **type-driven**: `_write_kv_segment_arrays` dispatches on the runtime type of `keys` — `QuantizedArray` → packed/scales/biases tensors; plain `mx.array` → float tensors. `_read_kv_segment_arrays` mirrors it by inspecting which tensor keys are present (`*_keys_packed` → quantized branch; `*_keys_float` → float branch). This means both `kv_data` (production default: q8, byte-identical wire format to v5) and `sliding_kv_data` (production default: bf16 float, `sliding_bits=None`) use the same helpers. Format v5 files are rejected at load.

**Eviction:** `sliding_kv_data` is treated identically to `recurrent_data` for interior-node cleanup — dropped from non-checkpoint nodes when they gain their first child, retained at permanent checkpoints and active leaves.
