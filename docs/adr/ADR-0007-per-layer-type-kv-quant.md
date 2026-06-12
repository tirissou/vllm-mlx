# ADR-0007: Per-layer-type KV cache quantization

**Status:** Accepted
**Date:** 2026-06-11

## Context

`--kv-cache-quantization` used a single global bit-width (`--kv-cache-quantization-bits`, default 8) applied uniformly to every layer. On Gemma 4 26B-A4B-class models, sliding-window layers use full standard RoPE — every K dimension is rotated — producing irregular K distributions that quantize poorly. Full-attention layers use pruned RoPE (~25% of dims rotated), leaving 75% clean content signal that quantizes well. The uniform-q8 policy therefore overpaid quality on the sliding-window side (irregular K → larger error) and also paid an unavoidable quantize-on-write + dequantize-on-read round-trip on every hit (per ADR-0005). Sliding-window cache hits dominate the hot path in long conversations.

## Decision

KV cache precision is selected per layer type by a structural rule keyed on the cache class name in `state_dict["class_name"]`:

| Cache class | Default bits | Rationale |
|---|---|---|
| `RotatingKVCache` | `None` (bf16) | Full RoPE quantizes poorly; storing float removes the round-trip. |
| Any other class name ending in `KVCache` (full-attention) | `8` (q8) | Pruned RoPE; dominant memory consumer; clean quantization signal. |
| Recurrent (everything else) | `None` (bf16) | No quantization path for recurrent caches. |

A new `KVQuantPolicy` dataclass (`vllm_mlx/cache_types.py`) owns the mapping. CLI flags are per-side: `--kv-cache-bits-sliding`, `--kv-cache-bits-full`. The single global `--kv-cache-quantization-bits` flag is removed; argparse rejects it with a migration hint. To make the smart defaults work without ambiguity, the CLI uses a private sentinel default so that "flag omitted" and "flag explicitly set to `none`" are distinguishable — `SchedulerConfig` carries an explicit `*_override` companion field for each side.

The `_segment` write path consults the policy once per layer and stamps `metadata["bits"] = bits` onto each `KVLayerSegment`. The `_assemble` read path drops its `bits` parameter and reads `bits` from per-layer metadata, making segment storage self-describing across the trie. `KVLayerSegment.concat` asserts all input segments agree on `bits` (within a single process a single policy writes the whole path, so a mismatch is a bug).

## Rationale & caveat

The split is architecturally well-motivated and supported by the Qwen3.5 partial-RoPE analogy. No published Gemma 4-specific per-layer ablation has confirmed it directly. The conservative default (sliding=bf16, full=q8) sticks to the architectural argument and does not depend on the unverified Qwen3.5 extrapolation; aggressive choices like `--kv-cache-bits-full 4` emit an INFO line referencing the caveat.

## Rejected alternatives

- **Model-name dispatch** — keying defaults on architecture names (`Gemma`, `Qwen`, …) would couple the cache layer to a per-model lookup that decays as new architectures land. Class-name dispatch keys on a structural fact (presence of `RotatingKVCache` ↔ sliding window) that is already true in the cache state dict.
- **Global uniform escape hatch** — re-adding `--kv-cache-bits-uniform N` would re-introduce the original mistake. Users who actually want every layer at q8 can pass `--kv-cache-bits-sliding 8 --kv-cache-bits-full 8`.

## Seam — self-describing segments

The `bits` value lives on each `KVLayerSegment.metadata` rather than as a `TurnCacheManager` field. This eliminates a hidden assumption (policy at write-time matches policy at read-time) and makes `_assemble` callable from any context where segments are available, including unit tests that don't carry a policy.

## Out of scope

- SSD persistence is broken and unused; this spec is runtime-only. No `_CACHE_FORMAT_VERSION` bump.
- Quantization path for recurrent caches.
- MLLM-side per-layer-type policy. `MLLMSchedulerConfig` carries a single `kv_cache_bits_full` field; `engine/batched.py` builds the upstream `KVQuantPolicy` and forwards `policy.full_bits` only. `--kv-cache-bits-sliding` has no effect on the MLLM path. (Advisory warnings still fire at the args→config boundary, so a user who sets `--kv-cache-bits-sliding 8` against an MLLM model sees the "sensitive to quantization error" warning even though the flag will then be silently dropped on this code path.) Full per-layer-type wiring of the MLLM cache is a future spec.
