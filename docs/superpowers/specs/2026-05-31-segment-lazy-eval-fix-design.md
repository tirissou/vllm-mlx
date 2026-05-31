# Design: Evaluate KVLayerSegment Arrays in `_segment()` to Free Dequantized Float16 Buffers

**Date:** 2026-05-31
**Status:** Approved

## Problem

`TurnCacheManager._segment()` produces `KVLayerSegment` objects whose `packed`, `scales`, and `biases` arrays are lazy MLX quantize computations. These hold computation graph references to intermediate float16 arrays (`lin_keys`/`lin_values` for RotatingKVCache, `sliced_keys`/`sliced_values` for KVCache), which in turn reference the live cache's dequantized float16 Metal buffers from `_assemble()`.

When `_insert_node()` stores the unevaluated `KVLayerSegment` in the trie, the entire dependency chain becomes rooted in the trie and stays live. `mx.clear_cache()` cannot free it because `mx.get_active_memory()` correctly sees it as referenced. The dequantized float16 data lingers in active Metal memory for the lifetime of the trie node — indefinitely, even between requests.

**Observed symptom:** `metal_active_memory_gb` stays at ~50 GB after decode completes and `mx.clear_cache()` is called, even with no active requests. Expected idle baseline is ~26 GB (16 GB model + 10 GB quantized trie).

**Root cause chain:**
```
trie node → KVLayerSegment → lazy packed/scales/biases
         → lin_keys (float16 intermediate)
         → RotatingKVCache.keys (dequantized float16 from _assemble())
         → Metal buffer: never freed
```

## Fix

Add `mx.eval()` on all six quantized components (`packed`, `scales`, `biases` for both keys and values) at the end of each `KVLayerSegment`-creating branch in `_segment()`:

- **RotatingKVCache branch** (after `q_keys` and `q_values` are created from `mx.quantize(lin_keys, ...)` and `mx.quantize(lin_values, ...)`)
- **KVCache branch** (after `q_keys` and `q_values` are created from `mx.quantize(sliced_keys, ...)` and `mx.quantize(sliced_values, ...)`)

This materialises the quantized arrays into concrete Metal buffers before returning, severing the computation graph dependency on the source float16 data. Once the source arrays go out of scope in the caller, they can be freed.

The recurrent branch stores `RecurrentLayerSegment.arrays` which hold raw arrays read from live cache `.state` — these are already evaluated. No change needed there.

## Invariant Being Enforced

> Any `KVLayerSegment` stored in the trie must hold only concretely evaluated Metal buffers — never lazy computations that reference source float16 data.

## Scope

One method (`TurnCacheManager._segment()`), two branches, ~1 `mx.eval()` call each. No changes to the trie, scheduler, `_assemble()`, callers of `_segment()`, or serialisation format.

## Testing

- Add a test: create a RotatingKVCache with float16 arrays, call `_segment()`, delete the source arrays, call `mx.eval()` and `mx.clear_cache()`, then assert that `mx.get_active_memory()` is at or below baseline (i.e., does not include the source float16 size).
- Existing segment/store/fetch round-trip tests must pass unchanged.
