# Canonical Prefill Padding + Decoded-K, V Cleanup

**Date:** 2026-06-18
**Status:** Approved for implementation

## Problem

In multi-turn conversations on Gemma 4 26B MoE with `TurnPrefixCache` (no quantization), generation quality degrades slowly across cache hits. First-decode quality off a fresh prefill is high; quality progressively worsens as subsequent decodes work off cached prefixes.

Empirical investigation (see `vllm_mlx/prefix_cache_adapters.py:_scan_M_regimes` and `_scan_chunking` probes added during debug) identified two compounding root causes:

1. **Small-M kernel-regime drift on partial-tail prefills.** When a request has a small number of new tokens to prefill after a cache hit (e.g., 24 tokens for a brief user follow-up), the model runs a forward pass with `S=24` while production's main chunks run with `S=4096`. On Gemma 4 MoE, M values below ~512 fall into a different MLX kernel regime than the canonical band [512, 1024]. The K, V computed for those small chunks differ measurably from what they would be in canonical regime (e.g., V diffs of ~8 at the last layer between `1x1024` and `4x256` schedules in the production probe). These off-regime K, V then enter the cache and pollute subsequent decodes.

2. **Decoded K, V in the trie.** `TurnCacheManager.store()` currently bundles `request.output_token_ids` (the decoded assistant response) into the trie node it promotes at end-of-turn. Decoded K, V are computed at M=1 — the worst possible kernel regime — and they accumulate across turns. Every cached prefix that spans past an assistant turn contains decoded K, V that don't match any canonical-regime reference.

The two causes interact: even if padding fixed (1), decoded K, V from (2) would continue to drift the cache. Fixing both is necessary.

## Empirical foundation

From the debug probes (Qwen 0.6B and Gemma 4 26B MoE):

- MLX is bit-deterministic at fixed batch shape (`fresh-vs-fresh max_K = 0`).
- The translator round-trip (`vllm_mlx/cache_translator.py`, `cache_types.py:KVConcatSegment`) is bit-exact for the no-quant path.
- On Gemma 4 MoE, chunk sizes `M ∈ {512, 1024}` are bit-identical at every probed layer (full and sliding). `M = 256` and below diverge at downstream layers. `M = 2048` diverges from `M = 1024` at the last full-attention layer.
- The canonical band on Gemma 4 MoE at B=1 is `M ∈ [512, 1024]`. Per-batch-size canonical bands may be narrower at higher B; the per-model canonical M must be the intersection across the production batch-size range.

## Goal

Every prefill forward pass through the model has exactly `S = prefill_step_size` tokens, where `prefill_step_size` is set to a value verified canonical for the deployed model and batch sizes. No K, V from decode steps (M=1) enter the trie. Result: the cache contains only regime-homogeneous K, V, and cross-turn quality stabilizes.

## Architecture

Three components, all under `vllm_mlx/`:

1. **`CanonicalPrefillBatchGenerator`** — a subclass of the existing `_InstrumentedBatchGenerator` (`scheduler.py:174`) that wraps `self.model` with a padding shim. Every prefill forward pass is right-padded to `prefill_step_size`, the cache is per-request trimmed afterward, and sliding-window buffers are sliced to drop pad rows.

2. **`TurnCacheManager.store()` → no-op.** Returns `False` without mutating the trie. All cache promotion happens via `on_prefill_checkpoint()` at turn boundaries during prefill. Decoded K, V never enter the trie.

3. **`scripts/find_canonical_m.py`** — a standalone CLI that probes a model across multiple chunk sizes and batch sizes, identifies the canonical regime, and recommends a `prefill_step_size` value.

`prefill_step_size` is the single source of truth. No new env vars, no new `MLXEngineConfig` fields. Operators set the CLI flag based on the tool's recommendation.

## Component 1 — CanonicalPrefillBatchGenerator

### Where it hooks

Extends `_InstrumentedBatchGenerator` in `vllm_mlx/scheduler.py`. The padding shim wraps the `self.model` field that mlx-lm's `BatchGenerator` uses for forward passes. By wrapping the model attribute rather than overriding a specific method, the shim catches every prefill model call without depending on mlx-lm's internal control flow.

### Shim behavior per call

For input shape `(B, S)` with cache list `cache`:

```
if S == 1:                # decode step
    return self._model(input, cache=cache)

if S >= canonical_M:       # already canonical
    return self._model(input, cache=cache)

pad = canonical_M - S
padded_input = right_pad(input, pad)
out = self._model(padded_input, cache=cache)

for layer in cache:
    layer.trim(pad)
    if isinstance(layer, RotatingKVCache):
        layer.keys   = layer.keys[..., :-pad, :]
        layer.values = layer.values[..., :-pad, :]

return out[..., :S, :]
```

`canonical_M` resolves to the BatchGenerator's `prefill_step_size`. The shim reads it from the parent class at construction time.

### Why the slicing step is needed for sliding layers

In `RotatingKVCache._update_concat` (mlx-lm `cache.py:449`), each prefill chunk temporarily expands the buffer to `max_size + S - 1` to ensure every chunk token gets full sliding-window context. The pre-chunk K, V at positions `[offset - max_size, offset)` are preserved during this expansion (only the single oldest position is dropped). After our padded forward, the expanded buffer holds `[old_oldest+1 .. offset_post_chunk)`, where the last `pad` rows are pad-generated garbage.

`trim(pad)` decrements `offset` and `_idx` correctly but does not change the buffer's physical contents. On the next `_update_in_place` call (the first decode step), the buffer is trimmed to `max_size` by keeping its tail — which includes the pad rows and discards old real K, V. To prevent that, the shim physically slices the pad rows off the buffer before any subsequent call sees it.

### Multi-request batches

The shim sees the batched input tensor `(B, S)`. mlx-lm's `BatchGenerator` left-pads each row so real new tokens for request `i` occupy positions `[S - real_chunk_size_i, S)`. The right-pad applied by the shim extends every row uniformly by `canonical_M - S`, producing a `(B, canonical_M)` input. The cache offsets for all requests advance by `canonical_M` during the forward.

For cleanup, the per-request rewind amount is:

```
pad_i = canonical_M - real_chunk_size_i
```

Two cases follow:

1. **Uniform `real_chunk_size_i` across the batch.** All `pad_i` are equal, so `cache.trim(pad)` with a scalar argument suffices.

2. **Mixed `real_chunk_size_i`.** `pad_i` differs per request. `BatchKVCache.trim` and `BatchRotatingKVCache.trim` must accept a per-batch vector argument so each request rewinds by its own amount. Implementation step: verify upstream supports this. If not, add a small patch in `vllm_mlx/patches/` (the project already has a `patches/` directory pattern).

### Progress accounting for the mid-prefill callback

mlx-lm's `BatchGenerator` increments per-request progress based on its own counter, independent of the cache state. `progress` reflects real tokens, not padded tokens. The mid-prefill callback (which `_InstrumentedBatchGenerator._next` fires) gets the real count, so `on_prefill_checkpoint` matches turn boundaries correctly without additional plumbing.

Implementation step: verify this assumption against mlx-lm's source. If `progress` is incremented by model-input size rather than real tokens, the shim must override that path too.

### What stays out of scope

- No changes to mlx-lm's prefill chunk-boundary logic.
- No changes to `_split_at_boundaries` (`scheduler.py:1312`). Boundary-aware chunking continues to produce sub-canonical chunks; the shim picks them up.
- No changes to the decode path (S=1) — shim early-out skips it.

## Component 2 — TurnCacheManager.store() → no-op

### Current behavior

`TurnCacheManager.store()` (`prefix_cache_adapters.py:790`) fires at end-of-turn (post-decode). It builds `response_tokens = segments[-1].token_ids + request.output_token_ids`, segments the cache, and inserts a new trie node containing both the last unmatched user message AND the decoded assistant response.

### New behavior

`store()` returns `False` without mutating any state. All cache promotion goes through `on_prefill_checkpoint()` (`prefix_cache_adapters.py:887`), which fires at each prefill chunk boundary and promotes only when the cumulative real-token count matches a registered turn boundary.

### Why this works

Turn boundaries (`request._turn_boundaries`) include end-of-user-message positions for every turn. On any subsequent turn that prefils through a prior assistant response, `on_prefill_checkpoint` fires at the assistant-end and user-end positions. Those K, V are now computed by **prefill** (in canonical regime, thanks to Component 1) rather than decode (M=1).

### Cost

Per cache hit, the prior assistant response gets re-prefilled instead of pulled from cache. For a 4k-token system + 200-token user + 500-token assistant + 100-token user-next scenario, the cache-hit prefill is 600 tokens (assistant + user-next) instead of 100 tokens (user-next only). On 26B MoE this is ~6x more work for the cache-hit step — still much less than a full no-cache prefill (4.8k tokens), and worth it for cache-state coherence.

### Migration

`store()` is called by the request lifecycle at end-of-turn. The no-op change is backward-compatible with existing callers — they get `False` instead of `True`, which they already handle (it's the cache-miss return value). No callers need to change.

## Component 3 — find_canonical_m.py CLI

### Invocation

```
python scripts/find_canonical_m.py --model <hf-id-or-local-path> \
    [--n 512,1024,2048,4096] \
    [--batch-sizes 1,2,4,8] \
    [--json]
```

- `--model`: required, mlx-lm-compatible path or HF id.
- `--n`: comma-separated N values to probe (default `512,1024,2048,4096`).
- `--batch-sizes`: comma-separated B values to probe (default `1,2,4,8`). Should cover at least `1` and production's `prefill_batch_size`.
- `--json`: also emit a machine-readable summary alongside the human report.

### What it does

1. Loads the model via mlx-lm's `load(model_path)`.
2. For each `(N, B)` combination: generates synthetic tokens (random vocab-bounded ids), runs probe schedules `1xN, 2x(N/2), 4x(N/4), 8x(N/8)`. Uses `BatchKVCache` / `BatchRotatingKVCache` / `BatchQuantizedKVCache` to match production cache types. Captures K, V at probe layers `{0, 1, n_layers//2, n_layers-1}` after each schedule.
3. Per `(N, B)`: builds a pairwise diff matrix across schedules at each (layer, K|V).
4. Per B: identifies the canonical band — the largest contiguous set of chunk sizes M for which every (layer, K|V) pairwise diff across schedules with chunk size in the band is exactly zero.
5. Computes the **intersection** of per-B canonical bands across all probed B values. The recommended `prefill_step_size` is the largest value in the intersection, capped at the smallest sliding `max_size` observed on the model.

### Output

Human report to stdout:

```
Model: gemma-4-26b-moe
Layers: 30 (KVCache: 5, RotatingKVCache: 25)
Sliding max_size: 1024

Per-batch-size canonical bands:
  B=1  band: M ∈ [512, 1024]
  B=2  band: M ∈ [512, 1024]
  B=4  band: M ∈ [512, 1024]
  B=8  band: M ∈ [1024]

Intersection (canonical across all B): M ∈ [1024]
Sliding max_size constraint: M <= 1024

Recommended: --prefill-step-size 1024
```

With `--json`, additionally emit a structured summary suitable for config ingestion:

```json
{
  "model": "gemma-4-26b-moe",
  "n_layers": 30,
  "n_sliding": 25,
  "n_full": 5,
  "sliding_max_size": 1024,
  "per_batch_bands": {"1": [512, 1024], "2": [512, 1024], "4": [512, 1024], "8": [1024]},
  "canonical_intersection": [1024],
  "recommended_prefill_step_size": 1024
}
```

### Empty-intersection failure mode

If no M is canonical across all probed B, the tool emits a clear failure message naming the per-B bands and recommending either lowering `--prefill-batch-size` or investigating per-B kernel regimes separately. It exits non-zero.

### Shared code with production probes

The CLI reuses the existing `_scan_chunking` comparison logic, refactored into `vllm_mlx/canonical_m_probe.py` as a pure function `run_chunking_probe(model, n, batch_size, layer_indices) -> dict`. The existing `_scan_chunking` method in `prefix_cache_adapters.py` becomes a thin wrapper around this function. The CLI calls the function directly without going through the production code path.

### Cost

At defaults: 4 batch sizes × 4 N values × 4 schedules = 64 probe forward passes. Memory peaks at the largest `(N, B)` combination (`N=4096, B=8` → 32k tokens per forward). Runtime estimate: a few minutes on a 26B MoE model. Operators with tight memory budgets can pass `--batch-sizes 1,prefill_batch_size` to skip intermediate values.

### Out of scope

- No automatic config writes. The operator reads the recommendation and sets `--prefill-step-size` themselves.
- No accuracy validation (perplexity, quality checks). The probe is purely a kernel-regime fingerprint.
- No multi-model batch mode. One model per invocation.

## Testing

### Unit tests

**`tests/test_canonical_prefill_padding.py`** — tests the padding shim against a tiny synthetic model:

- `test_padded_forward_advances_offset_correctly`: feed N real tokens at canonical M, verify cache offset equals N_real after trim(pad).
- `test_rotating_buffer_geometry_after_pad_trim_slice`: build a `RotatingKVCache` in steady state (offset > max_size, buffer at full max_size). Run a padded forward + trim + slice. Verify (a) buffer size is `max_size + N_real`, (b) `_idx == max_size + N_real`, (c) the last `pad` rows are gone, (d) old K, V at displaced positions are still present.
- `test_decode_after_pad_trim_reads_correct_positions`: post-pad-trim, run one decode step and verify the resulting buffer holds positions corresponding to the sliding window of the new decode position.
- `test_no_padding_when_S_equals_canonical_M`: shim early-out for S = canonical_M. Verify zero cache mutation beyond what the unwrapped model would do.
- `test_no_padding_for_decode_step`: shim early-out for S = 1.

**`tests/test_canonical_m_probe.py`** — tests the refactored `run_chunking_probe` function:

- `test_probe_returns_expected_shape`: known schedules × known layers → dict structure matches docstring.
- `test_probe_canonical_band_detection`: feed synthetic K, V data where the bit-stable band is known by construction. Verify recommendation matches.
- `test_probe_runs_at_b_gt_1`: smoke test at B=4.
- `test_canonical_band_intersection_logic`: synthetic per-B bands → expected intersection.

### Integration tests

**`tests/test_turn_prefix_cache_integration.py`** — updated for the `store()` no-op change:

- Tests that promote via `store()` move to `on_prefill_checkpoint()` invocations or get deleted if redundant with checkpoint coverage.
- New `test_store_does_not_promote_decoded_tokens`: simulate a request that just decoded, call `store()`, verify the trie is unchanged.
- New `test_decoded_tokens_re_prefilled_on_next_turn`: two-turn scenario. Turn 1 prefils + decodes. Turn 2 cache-hits at end-of-user-1. Verify the prior assistant response gets re-prefilled (not pulled from cache).

### End-to-end

**`tests/test_canonical_prefill_e2e.py`** (new) — runs the production code path on a small real model:

- `test_multi_turn_cache_hit_quality`: 3-turn dialogue against a small mlx-lm model (Qwen 0.6B used in earlier probes). Compare decoded output between (a) no-cache full prefill reference and (b) cache-hit path with canonical padding. With both components in place, the two paths produce outputs that are token-identical (or within a defined tolerance).

**`tests/test_cache_hit_oom_repro.py`** (existing) — runs unchanged. Peak memory should not regress materially; the pad rows add small constant overhead per chunk.

### CLI smoke

**`tests/test_find_canonical_m_cli.py`** (new) — invokes `scripts/find_canonical_m.py` against Qwen 0.6B. Captures stdout, asserts the report has the expected sections, the JSON parses against a small schema, and the recommended value is one of the expected canonical M's.

### Gate

`pytest tests/` clean (per CLAUDE.md).

## Rollout

### Pre-deployment

1. Run `scripts/find_canonical_m.py --model <production-model-path>` to get the recommended `--prefill-step-size`.
2. Smoke-test locally: replay a 3-turn cache-hit scenario with `--prefill-step-size <recommended>` and observe `[verify_kv]` log lines. The verify probe still reports non-zero diffs (one-shot reference vs chunked production — a known unrelated artifact, expected).
3. Run the full test suite.

### Deployment

No staged rollout, no feature flag. The new code path is always-on once merged.

Rationale:

- `store()` becoming a no-op is a correctness fix.
- The padding shim activates only when `S < prefill_step_size`. For chunks already at step size, the shim early-outs; behavior matches today.
- Operators control behavior via `--prefill-step-size`. Setting it to the old value (e.g., 4096) reverts the effective chunking semantics — the shim never engages because every chunk hits the early-out.

Rollback path: revert the operator's `--prefill-step-size` flag to the pre-fix value. No code rollback needed.

### Post-deployment validation

The user-visible symptom (multi-turn quality degradation) is the acceptance criterion.

- Run the same prompt across N turns. Quality should remain stable across 10+ cache-hit turns instead of slowly degrading.
- Inspect `_log_segment_breakdown` log lines: confirm trie nodes promoted via `on_prefill_checkpoint` have token counts matching turn boundaries. No node sizes should match `output_token_ids` lengths (would indicate `store()` was still promoting).
- Optional: temporarily set `VLLM_MLX_VERIFY_FETCH_KV=1` post-deployment. The `verify_kv` diffs won't go to zero (probe uses one-shot reference, regime-mismatched), but absolute magnitudes should drop meaningfully if decoded K, V no longer pollute the cache.

### Documentation

- CLI help text: explain `--prefill-step-size` as a regime-canonical setting; recommend running `find_canonical_m.py` to pick a value.
- Release note: "Decoded tokens are no longer cached. Cache hits land at prefill boundaries only; the prior turn's assistant response is re-prefilled on the next turn."

## Out of scope for this spec

- Verify-probe re-alignment to use a chunked reference matching production chunking. Deferred to a follow-up spec; would give a meaningful production dashboard but is not required for the quality fix.
- `MLXEngineConfig` field for canonical M. We use `prefill_step_size` directly.
- Multi-model canonical-M cache file consumed at server startup. Operator runs the tool and sets the flag.
- Per-batch-vector trim for `BatchKVCache` if mlx-lm doesn't already support it. If a patch is needed, scope it via the existing `vllm_mlx/patches/` pattern; implementation step decides.

## Implementation steps (to be expanded by the writing-plans skill)

1. Refactor `_scan_chunking` core into `vllm_mlx/canonical_m_probe.py:run_chunking_probe`.
2. Build `scripts/find_canonical_m.py` consuming the probe function with multi-B intersection logic.
3. Add `CanonicalPrefillBatchGenerator` subclass with the padding shim.
4. Verify (and patch if needed) `BatchKVCache.trim` per-batch-vector support.
5. Verify `progress` accounting in mlx-lm `BatchGenerator` counts real tokens.
6. Convert `TurnCacheManager.store()` to a no-op.
7. Update `tests/test_turn_prefix_cache_integration.py` for the no-op `store()`.
8. Add `tests/test_canonical_prefill_padding.py`, `tests/test_canonical_m_probe.py`, `tests/test_canonical_prefill_e2e.py`, `tests/test_find_canonical_m_cli.py`.
9. Wire `CanonicalPrefillBatchGenerator` into the scheduler factory in place of `_InstrumentedBatchGenerator`.
10. Documentation updates (CLI help text, release note).
