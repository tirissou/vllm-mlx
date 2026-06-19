# Chunked Prefill: Correctness Invariants

How to chunk prefill on bf16 GPUs without introducing slow quality drift across cache
hits. Read this before touching the prefill chunk schedule, the trie promotion rules,
or any path that splits a forward call into multiple chunks.

The core problem is **kernel-shape regime drift in bf16**: the same mathematical
operation routed through different GPU matmul kernels can produce results that differ
in their bit pattern, and these differences propagate through transformer layers and
accumulate across cache hits. Chunked prefill is correct only when every chunk is sized
to land in the same kernel regime.

This document records what we measured on two models (Qwen 0.6B dense; Gemma 4 26B MoE),
the three invariants that follow from those measurements, and the procedure to verify
them on a new architecture.

---

## TL;DR

Three rules. Violate any of them and chunked prefill produces K, V that drift from a
canonical reference, slowly degrading decode quality across multi-turn cache hits.

1. **Pin all prefill chunks to a canonical M.** The canonical M is model-specific. For
   Gemma 4 26B MoE we measured `M = 1024`; for Qwen 0.6B any `M ∈ {512, 1024, 2048}`.
2. **Right-pad partial tails to canonical M** with `cache.trim(pad)` after the forward
   pass. Causal masking makes the pad rows invisible to real tokens at zero extra cost.
3. **Promote trie nodes only at prefill boundaries.** Never cache K, V that were
   produced by decode (`M=1`), even if the decoded tokens are part of a multi-turn
   conversation prefix.

---

## 1. Why this matters: bf16 non-associativity meets kernel dispatch

bf16 floating-point arithmetic is non-associative: `(a + b) + c` and `a + (b + c)` can
produce different bit patterns when intermediate results round differently. GPU matmul
kernels dispatch to different tiled implementations based on input shape — the tile size
that the kernel picks determines the accumulation order, and the accumulation order
determines the bit pattern of the output.

Two consequences:

- **Same operation, different shapes → different bits.** Multiplying the same row vector
  by the same matrix can give different bf16 outputs depending on the *batch size of the
  surrounding call*. This is not a bug; it is intrinsic to bf16 + tile-dispatch.
- **Drift accumulates through layers.** A single LSB flip at layer 0 propagates through
  the attention softmax (where it can cross small thresholds), then through MLP
  non-linearities (where it can land on different sides of an activation gate), and so
  on. By layer 30 the original LSB flip is no longer LSB — it can be unit-scale.

The two shape variables that determine which kernel is dispatched:

- **M** — number of rows in the forward call (number of query positions). For prefill,
  this is the chunk size. For decode, `M = 1`.
- **T_kv** — number of cached key/value positions visible to attention. Determines the
  K, V cache dimension that Q attends against.

Both vary depending on how prefill is chunked. The job of canonical chunking is to pin
the kernel call to the same shape across every prefill of every chunk.

---

## 2. The three invariants

### Invariant 1: Canonical chunk size

**Rule.** Every prefill forward call must use `M = M_canonical`, a single global
constant chosen per architecture from the measured canonical band.

**Why.** A model's matmul kernels — both the per-token projections (`W_K`, `W_V`,
`W_Q`, `W_O`, MLP) and the per-position attention (`Q @ K^T`, `softmax @ V`) — fall into
discrete regimes by input shape. Within a regime, output is bit-identical across shape
variation. Across regimes, output differs. The canonical band is the contiguous range
of `M` values for which all relevant kernels stay in one regime.

The band is model-specific (see §4) and has both a lower and upper edge. Below the
lower edge, small-batch kernels engage and drift downward. Above the upper edge,
large-batch kernels engage and drift upward.

**Implementation.** Set `M_canonical` to a single value globally; configure your
scheduler to never produce a chunk with `M ≠ M_canonical`, with one exception (Invariant
2 — partial tails). Prefer the upper end of the canonical band for latency; longer
chunks mean fewer forward calls per prefill.

### Invariant 2: Right-pad partial tails

**Rule.** When `n_real < M_canonical` for the final chunk of a prefill, right-pad the
input to `M_canonical` and call `cache.trim(pad)` after the forward returns:

```python
M = M_canonical
n_real = len(remaining_tokens)
pad = M - n_real

# Right-pad: real tokens FIRST, then pad. Critical for RoPE position assignment.
chunk_input = mx.concatenate([
    remaining_tokens,
    mx.zeros((pad,), dtype=remaining_tokens.dtype),
])  # shape (M,)

_ = model(chunk_input[None], cache=cache)

# Rewind: drops the garbage pad K, V from future reads.
for c in cache:
    c.trim(pad)
```

**Why right-pad, not left-pad.** RoPE assigns each chunk position an absolute position
index of `cache.offset + chunk_pos`. With right-pad, real tokens land at
`chunk_pos ∈ [0, n_real)` → absolute positions `[offset, offset + n_real)` — correct.
With left-pad, real tokens land at the *end* of the chunk → RoPE rotates them for the
wrong absolute position. The model effectively sees the real tokens at positions they
weren't intended for.

**Why padding is invisible to real tokens.** Causal masking blocks real-token attention
from reading chunk positions `[n_real, M_canonical)`. The pad K, V are computed and
written to the cache buffer at offsets `[n_real, M_canonical)`, but the next
`cache.trim(pad)` makes them unreadable by future operations. No explicit pad mask is
needed.

**Why `cache.trim()` not manual offset decrement.** `cache.trim(n)` is the supported
mlx-lm API and handles all cache subclasses uniformly:

| Cache type | `trim(n)` effect |
|---|---|
| `KVCache`, `QuantizedKVCache`, `BatchKVCache` | `offset -= n`. Next `update_and_fetch` writes at the new lower offset, overwriting garbage. ✓ Safe. |
| `RotatingKVCache` | `offset -= n; _idx -= n`. Safe **only if no internal rotation triggered during the padded chunk** (see Pitfall 1). |

Manually setting `cache.offset -= pad` is incorrect for `RotatingKVCache` because it
leaves `_idx` out of sync, corrupting the ring-buffer layout.

### Invariant 3: Trie promotion only at prefill boundaries

**Rule.** A trie node may be promoted (have its K, V stored for cross-turn reuse) only
at positions where the K, V at every position `[0, n_tokens)` of the trie node was
computed by *prefill*. K, V produced during decode (`M=1`) must never enter the trie.

**Why.** Decode has `M = 1` by construction — every step is a single new token. `M = 1`
is the smallest possible kernel shape, and on every model we have tested it sits deep in
the small-M regime, far below the canonical band lower edge. Decoded K, V at any
position therefore drift from canonical-regime K, V at that same position by an amount
that cannot be eliminated by chunking strategy.

If decoded K, V enter the trie, then future cache hits will retrieve K, V that are in a
fundamentally different kernel regime than what a fresh prefill would compute. As the
proportion of decoded K, V in the cached prefix grows across turns, drift compounds and
quality degrades.

**Cost.** Each turn re-prefills the previous assistant's response (which would otherwise
have been cached from the decode path). For typical chat workloads the assistant
response is `O(100–1000)` tokens, a small fraction of total prefill compute.

**Operational consequence.** Between turn `N-1` (which decoded an assistant response)
and turn `N` (which prefills the new user prompt), the prefill that fills the gap
covers `[previous assistant tokens] + [new user prompt]`. The trie state advances by
exactly one new node at the end of turn `N`'s user prompt, regardless of how many
decode steps happened in turn `N-1`.

---

## 3. Empirical methodology

Two probes, implemented in `vllm_mlx/prefix_cache_adapters.py`, gated by environment
variables. Both run during a `fetch()` after a cache hit and emit pairwise-diff matrices
to the log.

### Probe A: M regime scan (`_scan_M_regimes`)

Run with `VLLM_MLX_VERIFY_SCAN=1`. Takes the first 64 real tokens of the cached prefix,
runs the model at each `M ∈ {64, 128, 256, 512, 1024, 2048, 4096}` with zero-padding,
and dumps layer-0 K and V at positions `[0, 64)`. Emits two pairwise max-abs-diff
matrices (one for K, one for V) in fp32.

**What it measures.** The `W_K` and `W_V` projection kernels at layer 0. Layer-0 K, V
depend only on the input embedding — the attention mask is irrelevant because the
projection is purely per-position. Pad tokens contaminate higher-layer K, V via
attention, but never layer 0.

**Interpretation.** Two `M` values whose entries are `0.00e+00` share a regime for the
projection kernel. Two values with non-zero diff straddle a regime boundary. Read down
the M=64 column: any zero entries are in the same regime as M=64; non-zero entries are
in different regimes.

**Limitation.** Only covers projection kernels, not attention. The attention kernel may
have a stricter regime threshold (typically: yes — see §4 findings).

### Probe B: chunking schedule scan (`_scan_chunking`)

Run with `VLLM_MLX_VERIFY_CHUNK_SCAN=1` and optionally
`VLLM_MLX_VERIFY_CHUNK_SCAN_N=<N>` (default 1024). Takes the same physical `N`-token
range and prefills it under four schedules: `1xN, 2x(N/2), 4x(N/4), 8x(N/8)`. Dumps K
and V at probe layers `{0, 1, mid, last}` and emits a pairwise diff matrix per
(layer, K/V).

**What it measures.** The full pipeline effect of chunking at the matched absolute
positions. Layer 0 is a control (chunking-invariant unless the M-regime cliff hits
W_K/W_V). Downstream layers reveal attention-kernel regime effects and any
amplification through MLP / MoE expert routing.

**Interpretation.**

- All-zero matrix at a layer → that layer's K, V are bit-identical across schedules at
  that `N`. The schedules' shared chunk sizes are in one regime end-to-end.
- L=0 zero, L>0 non-zero → projection kernel is fine, but the attention kernel at the
  smaller chunks falls into a different regime. The attention regime threshold is
  *stricter* than the projection regime threshold.
- Diagonal cluster of zeros (e.g., `1xN = 2x(N/2)` but ≠ `4x(N/4)`) → identifies the
  canonical band: chunk sizes producing the same K, V are in the same regime.

**Selecting N.** Run multiple times. `N = 1024` covers sliding-window layers with
`max_size = 1024` (Gemma 4 MoE); `N = 2048` covers wider canonical regimes; `N = 4096`
tests production-scale main-chunk sizes. Sliding layers with `max_size < N` are skipped
("buffer < N").

### Auxiliary: fetch-time KV verification (`_verify_fetched_kv`)

Run with `VLLM_MLX_VERIFY_FETCH_KV=1`. After a cache hit, runs a fresh one-shot prefill
of the cached prefix and diffs against the cached K, V. Emits per-layer max-abs and
mean-abs diffs.

**Caveat.** This probe compares cached K, V (produced by chunked production prefill)
against a single-chunk reference. The diff it reports therefore conflates:

- True translator / quantization drift (target signal).
- Kernel-shape regime mismatch between production chunking and the one-shot reference
  (noise).

On models where M_production ≠ M_reference falls outside the canonical band, the probe
will report large diffs that are *not* a translator bug. To get a clean signal, the
reference forward must use the same chunking schedule as production.

---

## 4. Findings

### Qwen 0.6B (dense, KVCache only, 28 layers)

| Kernel | M values that share a regime |
|---|---|
| L=0 K, V projection | `M ∈ {512, 1024, 2048, 4096}` |
| Deep-layer K, V at L=27 | `M ∈ {512, 1024, 2048}` (4096 untested in chunk scan) |

Canonical band: `M ∈ [512, 2048]`. Pick `M_canonical = 1024` or `2048`.

### Gemma 4 26B MoE (5 full + 25 sliding, sliding `max_size = 1024`)

| Kernel | M values that share a regime |
|---|---|
| L=0 K, V projection (RotatingKVCache) | `M ∈ {256, 512, 1024, 2048, 4096}` |
| Sliding L=1 K, V | `M ∈ {512, 1024}` (M=256 drifts) |
| Sliding L=15 K, V | `M ∈ {512, 1024}` (M=256 drifts) |
| Full L=29 K, V | `M ∈ {512, 1024}` (M=256 drifts; M=2048 differs from M=1024 by ~0.83 K) |

Canonical band: `M ∈ [512, 1024]`. Pick `M_canonical = 1024`.

### Cross-architecture pattern

Two robust observations across both models:

1. **The attention kernel has a stricter regime threshold than the projection kernel.**
   L=0 K, V become canonical at a lower `M` than downstream K, V. This is why a
   projection-only probe (layer-0 M-scan) is necessary but not sufficient — the chunk
   scan at downstream layers is the load-bearing measurement.

2. **The canonical band has an upper edge as well as a lower edge.** On Gemma we
   observed M=1024 and M=2048 in different regimes at the last full-attention layer.
   "Bigger M is always safer" is false. Pick a specific `M_canonical` and stay there.

### What MoE adds (and doesn't)

MoE routing introduces an amplification mechanism: if upstream activations drift enough
to cross a top-k routing boundary, different experts activate, and K, V can change at
unit scale rather than LSB scale. We initially suspected this was happening between
`1x2048` and `2x1024` on Gemma 4 MoE (which showed `0.83` K diff at L=29).

The chunk scan at `N=1024` ruled this out: `1x1024` and `2x512` are bit-identical at
L=29 (and at all probed sliding layers). If MoE routing were flipping at canonical
shapes, we would see non-zero diff here. We do not. The `1x2048` / `2x1024` diff was
purely kernel-regime drift from M=2048 falling outside the canonical band.

**Inference:** MoE routing stays stable *within* the canonical band. Routing flips are
a downstream consequence of regime drift, not an independent failure mode. Fixing
chunk size fixes routing stability automatically.

---

## 5. Pitfalls

### Pitfall 1: Sliding-window rotation during a padded chunk

`RotatingKVCache._update_concat` wraps the ring buffer when `_idx + S > max_size`. A
padded chunk that crosses the rotation boundary mid-chunk evicts real tokens written
earlier in the same chunk, and `cache.trim(pad)` cannot recover them — it only undoes
`_idx -= pad`, not the rotation's eviction.

**Mitigation.** Either guarantee `_idx_before + M_canonical ≤ max_size` (size your
chunks so they fit between rotation boundaries), or align padded-tail chunks so they
either fit entirely before the next rotation or start fresh after it.

**Detection.** Before the padded forward, check `c._idx + M_canonical > c.max_size`
for any `RotatingKVCache` layer.

### Pitfall 2: Verify probes that compare against a one-shot reference

The fetch-time verify probe runs a single-chunk reference forward to compare against
cached K, V. If `M_reference ≠ M_canonical` for any kernel that crosses a regime
boundary, the probe will report drift that is **not a bug** — it is the expected
difference between two equally-valid regime calls.

**Mitigation.** When a verify probe needs to compare cached K, V to a fresh
computation, the fresh computation must use the same chunking schedule that produced
the cached K, V. A one-shot reference is meaningful only when `M_full ≤ band upper edge`.

### Pitfall 3: Heterogeneous regimes across the cached prefix

If the main prefill uses `M_main` and partial tails use `M_tail`, and these are in
different regimes, then the cached prefix has heterogeneous regime history — different
positions were computed by different kernels. Cross-turn comparison breaks: a turn
that recomputes part of the prefix in `M_main` regime will produce K, V that disagree
with what's cached.

**Mitigation.** Use `M_canonical` for *every* prefill, including partial tails (via
Invariant 2). Do not mix chunk sizes.

### Pitfall 4: New-tokens prefill after a cache hit ignores canonical M

The most common source of production drift. When the cache hits at position `H` and
the new user prompt is `n_new` tokens with `n_new < M_canonical`, the naive
implementation forwards `n_new` tokens directly — yielding a forward call with
`M = n_new` (e.g., M=24), which is deep in the small-M regime.

**Symptom.** Quality degrades across cache hits with small new-user prompts. Each turn
adds K, V in a worse and worse regime onto the cached prefix.

**Mitigation.** Apply Invariant 2 to the new-tokens prefill: right-pad to
`M_canonical`, then `cache.trim(pad)`.

### Pitfall 5: Conflating decode K, V with prefill K, V

Decode is `M = 1` by definition. Promoting trie nodes past the last prefill boundary
includes M=1 K, V in the cache. Even with otherwise-perfect chunking, this slowly
poisons the cache.

**Mitigation.** Invariant 3 — promote only at prefill boundaries.

---

## 6. Per-architecture verification protocol

Before deploying chunked prefill on a new architecture, run this protocol:

1. **Run probe A** (`VLLM_MLX_VERIFY_SCAN=1`) on a representative prompt with
   `cached_tokens ≥ 64`. Identify the smallest `M` for which layer-0 K and V matrices
   show `0.00e+00` against all larger `M` values. This is the **projection cliff**.

2. **Run probe B** (`VLLM_MLX_VERIFY_CHUNK_SCAN=1`) at multiple `N` values:
   - `N = max(1024, sliding_max_size)` to cover sliding layers.
   - `N = 2048` to test M ∈ {256, 512, 1024} against M=2048.
   - `N = 4096` (if `cached_tokens ≥ 4096`) to test production-scale chunks.

   At each `N`, identify the contiguous set of schedules whose deep-layer K, V are
   bit-identical. The intersection across `N` values is the **canonical band**.

3. **Pick `M_canonical`** from the upper end of the canonical band, but verify it
   doesn't cross a sliding-window rotation boundary mid-chunk for representative
   prompt lengths.

4. **Confirm sliding behavior.** If any layer is `RotatingKVCache`, verify that
   `M_canonical ≤ max_size - max_expected_idx_before_chunk`. Otherwise, Pitfall 1
   applies.

5. **Re-run probe A and B with the chosen `M_canonical`** as the production setting.
   Confirm `[verify_kv]` reports near-zero diffs (the remaining diff should be
   quantization-only if quantization is on).

6. **Production check.** Run a multi-turn conversation test and confirm decode quality
   is stable across cache hits. If quality drifts, check (in order): is Invariant 2
   applied to the new-tokens prefill? Is Invariant 3 enforced (no decode K, V in
   trie)? Is `M_canonical` actually being used for every forward call?

---

## 7. Generalization across architectures

What's expected to vary:

- **Cliff position** depends on head dimension, head count, and MLX (or other backend)
  kernel dispatch rules. Larger models with larger head dimensions tend to have lower
  cliffs (M=256 for Gemma 4 vs M=512 for Qwen at the projection level), but the
  attention cliff is consistently at `M ≥ 512` on both models we tested.
- **Canonical band width**. Dense models tend to have wide bands; MoE models tend to
  have narrower bands because routing-flip amplification truncates the upper edge
  sooner (we did not directly observe this — Gemma's narrower band may be size-related
  not MoE-related — but it remains a hypothesis to verify per architecture).
- **Sliding-window rotation interactions**. Different models use different sliding
  window sizes (Gemma 4: 1024; some Llamas: 4096). `M_canonical ≤ sliding_max_size` is
  a hard constraint if you want to avoid Pitfall 1.

What's expected to be invariant:

- **Decode (`M=1`) is always non-canonical.** Invariant 3 applies universally.
- **The attention kernel's regime threshold is at least as strict as the projection
  kernel's.** Probe A (projection only) gives an *upper bound* on the canonical lower
  edge; the actual lower edge is determined by probe B (full pipeline).
- **Right-pad + `cache.trim(pad)`** is correct on every architecture for non-sliding
  layers. For sliding layers, Pitfall 1 must be considered.
- **Causal masking handles pad rows for free** — no per-architecture work needed to
  ensure real tokens don't attend to pad tokens.

---

## 8. What we don't claim

- Bit-exactness between chunked prefill and one-shot prefill on all architectures.
  When `M_canonical < model_max_seq_len`, a one-shot forward call uses a different
  kernel than chunked forward calls; the resulting K, V can legitimately differ even
  when both are correct. Equivalence is between *consistently-chunked* prefills.
- Equivalence between decoded K, V and prefilled K, V at the same position. They are
  computed in different kernel regimes and will differ at bf16 precision.
- That `M_canonical` is the same across all model families. It is empirically
  determined per architecture by the probe protocol.
- That MoE routing is bf16-stable in general. Within the canonical band on Gemma 4 MoE
  we observed routing stability, but this should be reverified per MoE architecture.

---

## Related code

- **Probes.** `vllm_mlx/prefix_cache_adapters.py:_scan_M_regimes`,
  `vllm_mlx/prefix_cache_adapters.py:_scan_chunking`,
  `vllm_mlx/prefix_cache_adapters.py:_verify_fetched_kv`.
- **Env flags.** `VLLM_MLX_VERIFY_FETCH_KV`, `VLLM_MLX_VERIFY_SCAN`,
  `VLLM_MLX_VERIFY_CHUNK_SCAN`, `VLLM_MLX_VERIFY_CHUNK_SCAN_N`.
- **Cache types.** `vllm_mlx/cache_types.py` (`KVConcatSegment`, `KVRotatingSegment`),
  `vllm_mlx/cache_translator.py` (`_segment`, `_assemble`, `slice_kv_to_delta`).
- **Cache APIs.** `mlx_lm.models.cache.KVCache.trim`,
  `mlx_lm.models.cache.RotatingKVCache.trim` (note `_idx` interaction),
  `mlx_lm.models.cache.QuantizedKVCache.trim`.
- **Architecture decisions.** ADR-0003 (no cache orchestrator), ADR-0005 (segment
  storage format), ADR-0008 (per-architecture behavior).

## Related dev notes

- `docs/dev/cache-reconstruction-invariants.md` — invariants for `_assemble` and
  `BatchQuantizedKVCache` construction. Read together with this document when touching
  the cache-hit path.
- `docs/dev/mlx-memory-profiling.md` — how to measure Metal peak memory; relevant when
  Invariant 2's tail-padding increases per-chunk allocations.
