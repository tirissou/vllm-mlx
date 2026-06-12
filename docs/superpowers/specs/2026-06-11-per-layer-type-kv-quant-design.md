# Per-layer-type KV cache quantization

**Date:** 2026-06-11
**Status:** Draft

## Goal

When `--kv-cache-quantization` is on, pick KV cache precision per layer type instead of using one global bit-width. The rule is structural — keyed off the cache class, with no model-name or architecture detection — and produces smart defaults that exploit the asymmetry between RoPE-rotated and partial-RoPE layers.

| Cache class in `_segment` | Smart default | Rationale |
|---|---|---|
| `RotatingKVCache` (sliding-window) | bf16 (no quantization) | Full standard RoPE rotates every K dim, producing irregular K distributions that quantize poorly. Today's uniform-q8 also pays a quantize-on-write + dequantize-on-read round-trip on every hit (ADR-0005: "unavoidable"); storing float removes the round-trip entirely. |
| `KVCache` (full-attention) | q8 | In Gemma 4 these layers use pruned RoPE (~25% of dims rotated); 75% of K is clean content signal that quantizes cleanly. They also share KV across full-attention layers, further regularizing distributions. This is the dominant-memory branch, so the q8 default carries most of the savings. |
| Recurrent (`RecurrentLayerSegment`) | bf16 (no quantization) | Matches existing behavior; recurrent state never goes through `mx.quantize`. |

CLI overrides are per-layer-type only: `--kv-cache-bits-sliding` and `--kv-cache-bits-full`. The current `--kv-cache-quantization-bits N` global flag is **removed** (breaking change). Recurrent is hardcoded bf16 with no override flag — there is no quantization path for recurrent caches today and inventing one is out of scope.

### Motivation caveat

The default split is architecturally well-motivated and supported by the Qwen3.5 partial-RoPE analogy (where partial-RoPE layers showed zero measurable regression at low precision), but no published Gemma 4-specific per-layer ablation has confirmed it directly as of mid-2026. The conservative default (sliding=bf16, full=q8) sticks tightly to the architectural argument and does not depend on the unverified Qwen3.5 extrapolation; users who want q4 on full layers must pass `--kv-cache-bits-full 4` explicitly and accept the caveat.

## Non-goals

- No model-name dispatch. The mapping is purely class-name → bits.
- No global uniform escape hatch. If a user wants every layer at q8, they pass `--kv-cache-bits-sliding 8 --kv-cache-bits-full 8`.
- No SSD persistence work. SSD persistence is broken and unused; it is being redesigned in a follow-up spec. This spec is runtime-only — no `_CACHE_FORMAT_VERSION` bump, no v5-compat guarantees.
- No quantization path for recurrent caches.

## Architecture

### New type — `KVQuantPolicy`

Lives in `vllm_mlx/cache_types.py`, next to `KVLayerSegment`, since it co-governs how segments are written.

```python
@dataclass(frozen=True)
class KVQuantPolicy:
    sliding_bits: int | None = None   # bf16
    full_bits: int | None = 8         # q8
    # Provenance of each field, used by describe() and warning logic.
    # True = value came from a user CLI override; False = smart default.
    sliding_override: bool = False
    full_override: bool = False

    def bits_for(self, class_name: str) -> int | None:
        if class_name == "RotatingKVCache":
            return self.sliding_bits
        if "KVCache" in class_name:
            return self.full_bits
        return None  # recurrent and anything else: never quantize

    def describe(self) -> str:
        # Examples:
        #   "sliding=bf16, full=q8 (smart defaults)"
        #   "sliding=q8 (user override), full=q4 (user override)"
        #   "sliding=bf16, full=q4 (user override)"
        ...
```

`None` for `sliding_bits` / `full_bits` means "store float, do not quantize." An int means "quantize to that many bits with the configured `kv_cache_quantization_group_size`." The `*_override` flags carry provenance — they let `describe()` produce the right startup-log label and let warning logic distinguish "user explicitly set this" from "field defaulted."

### Telling "unset" from "explicitly set"

To distinguish a flag the user didn't pass from a flag the user explicitly set to `none` (matters for both the warning logic and the `*_override` flags above), the CLI uses a private sentinel object as the argparse default — not `None`. `SchedulerConfig` gains two boolean companion fields, `kv_cache_bits_sliding_override: bool = False` and `kv_cache_bits_full_override: bool = False`. `cli.py` resolves the sentinel into the pair `(value, override_bool)` for each side: sentinel → `(None, False)`, user-supplied `none` → `(None, True)`, user-supplied int N → `(N, True)`. Both pairs are passed into `SchedulerConfig`. `_build_kv_quant_policy(cfg)` then has full provenance available when emitting warnings and constructing the `KVQuantPolicy`. The sentinel never escapes `cli.py`.

### Touched files

- **`vllm_mlx/cache_types.py`** — add `KVQuantPolicy` dataclass with the `bits_for` and `describe` methods above.

- **`vllm_mlx/scheduler.py` (`SchedulerConfig`)** — replace the field `kv_cache_quantization_bits: int = 8` with four fields: `kv_cache_bits_sliding: int | None = None`, `kv_cache_bits_full: int | None = None`, `kv_cache_bits_sliding_override: bool = False`, `kv_cache_bits_full_override: bool = False`. The `*_override` companions carry the provenance information needed to distinguish "user didn't pass the flag" from "user explicitly passed `none`" (see "Telling 'unset' from 'explicitly set'" above). Add a module-level helper `_build_kv_quant_policy(config) -> KVQuantPolicy | None` that returns `None` when `kv_cache_quantization` is off and otherwise constructs `KVQuantPolicy(...)` with overrides and provenance flags threaded through; this is the single function that owns all CLI-validation warnings (see "Validation and warnings" below).

- **`vllm_mlx/prefix_cache_adapters.py` (`TurnCacheManager`)** — constructor becomes `__init__(self, inner, policy: KVQuantPolicy | None, kv_group_size: int = 64)`. `policy=None` means quantization is disabled entirely (everything stored float). `self._policy` replaces `self._kv_bits`. Update the call site in `_build_prefix_cache` to pass the policy from `_build_kv_quant_policy(config)`.

- **`vllm_mlx/prefix_cache_adapters.py` (`_segment`)** — accepts `policy: KVQuantPolicy | None` instead of `bits: int | None`. Inside each branch (`RotatingKVCache`, `KVCache`/concatenate, recurrent), compute `bits = policy.bits_for(class_name) if policy else None` once. The existing three tracks (A: already-quantized state, B: float storage, C: quantize from float) keep working; the per-class dispatch decides which track each layer takes. The new behavior is the **mix** within a single segment list, not new tracks.

- **`vllm_mlx/prefix_cache_adapters.py` — `KVLayerSegment.metadata`** — `_segment` writes `metadata["bits"] = bits` (the int or `None` actually used for that layer). This makes segment storage self-describing.

- **`vllm_mlx/prefix_cache_adapters.py` (`_assemble`)** — drops the `bits` parameter and reads `bits = layer.metadata["bits"]` per layer. The branch already keys on `isinstance(layer.keys, QuantizedArray)`, so the assemble code barely changes — it just sources `bits` per-layer instead of from a function argument.

- **`vllm_mlx/prefix_cache_adapters.py` (`KVLayerSegment.concat`)** — add an assertion that all segments in the input list agree on `metadata["bits"]`. Within a single process the policy is immutable, so this should always hold; the assertion catches policy-mismatch bugs early.

- **`vllm_mlx/cli.py`** — replace the `--kv-cache-quantization-bits` argument with `--kv-cache-bits-sliding` and `--kv-cache-bits-full` (both `int` or `none`, default unset → smart default). Add a deprecation-error argparse action for `--kv-cache-quantization-bits` that exits with the migration message. Update the startup log line to use `policy.describe()`.

- **`vllm_mlx/engine/batched.py` and `vllm_mlx/mllm_scheduler.py`** — rename the field at the `SchedulerConfig` construction sites; same pattern as `cli.py`.

- **`docs/adr/ADR-0007-per-layer-type-kv-quant.md`** — new ADR recording: (a) the rule, (b) rationale + the Qwen3.5 analogy caveat, (c) rejected alternatives (model-name dispatch, global escape hatch), (d) self-describing `metadata["bits"]` as the seam between `_segment` and `_assemble`.

- **`CONTEXT.md`** — add `KVQuantPolicy` to the glossary; note that `KVLayerSegment.metadata` now carries `bits`.

## Data flow

### Write path (cache miss → segment into trie)

1. `Scheduler.__init__` calls `_build_kv_quant_policy(config)`. If `kv_cache_quantization` is off → returns `None` (after emitting an ignored-flag warning if either override was explicitly set). Else → returns `KVQuantPolicy(sliding_bits=..., full_bits=..., sliding_override=cfg.kv_cache_bits_sliding_override, full_override=cfg.kv_cache_bits_full_override)`. For each side, when `*_override` is False the smart default is used (sliding=None, full=8); when True the user-supplied value (including `None`) is used verbatim. All advisory warnings from "Validation and warnings" are emitted here before the policy is returned.
2. Policy is handed to `TurnCacheManager` once at construction. Immutable for the lifetime of the process.
3. On `store(...)`, `_segment(live_states, policy, group_size)` walks the live caches. For each layer `i`:
   - Resolve `class_name = state_dict["class_name"]`.
   - Compute `bits = policy.bits_for(class_name) if policy else None`.
   - Take the existing Track A / B / C branch based on `(bits, type(state[0]))` — no new tracks are introduced.
   - Write `metadata["bits"] = bits` into the resulting `KVLayerSegment`.

### Read path (cache hit → assemble live cache)

1. `collect_path_data` returns `list[KVLayerSegment]` per layer (unchanged — `concat` still concatenates `packed`/`scales`/`biases` independently along `axis=-2` per ADR-0005).
2. `_assemble(kv_layers, recurrent_layers, group_size)` reads `bits = layer.metadata["bits"]` for each layer. Same three reconstruction branches as today — only the source of `bits` changes.
3. Float-stored sliding layers feed `RotatingKVCache` directly with no dequantize step (this is the speed/quality win on the hit path; today's code dequantizes every rotating layer on every hit). Quantized full-attention layers feed `BatchQuantizedKVCache.from_quantized_arrays` (per ADR-0005, unchanged).

### Concat across the trie path

A path of segments for the same layer is always written by the same policy in a given process, so all segments for one layer share the same `bits` and the same storage shape. `KVLayerSegment.concat` keeps its current logic — concatenate `packed`/`scales`/`biases` for quantized, or arrays directly for float — and adds a cheap assertion that all input segments agree on `metadata["bits"]`.

### Quantization off

When `kv_cache_quantization=False`, `_build_kv_quant_policy` returns `None`, `_segment` sets `bits=None` everywhere, and everything is stored float. Identical to today's "quantization disabled" state.

## Validation and warnings

All emitted via the existing module-level `logger` in `scheduler.py` (or the deprecation action's argparse error for the removed flag). Owned by `_build_kv_quant_policy` so they're unit-testable.

### Hard error (argparse rejects before scheduler starts)

- `--kv-cache-quantization-bits N` — flag is removed. A custom argparse action intercepts it and exits with:
  > *`--kv-cache-quantization-bits` was removed. Use `--kv-cache-bits-sliding` (default: bf16) and/or `--kv-cache-bits-full` (default: q8). See ADR-0007.*

### Warnings in `_build_kv_quant_policy(config)`

| Trigger | Level | Message |
|---|---|---|
| `kv_cache_bits_sliding` or `kv_cache_bits_full` set, but `kv_cache_quantization` is off | WARNING | *"`--kv-cache-bits-{sliding,full}` is ignored because `--kv-cache-quantization` is disabled."* |
| User explicitly passed `--kv-cache-bits-full none` (full_override=True and full_bits=None) | WARNING | *"Full-attention layers are the dominant memory consumer; setting them to bf16 negates the memory benefit of `--kv-cache-quantization`. Did you mean to leave the default (q8)?"* |
| User explicitly passed `--kv-cache-bits-sliding N` where N is an int (sliding_override=True and sliding_bits is an int) | WARNING | *"Sliding-window layers use full RoPE and are sensitive to quantization error. Recommended default is bf16; quantizing them may degrade quality. Measure before relying on this setting."* |
| `full_bits` is an int ≤ 4 (regardless of provenance — applies to both the unlikely future smart default and explicit overrides) | INFO | *"`--kv-cache-bits-full {N}` is supported by the Qwen3.5 partial-RoPE analogy but unverified for Gemma 4. Measure quality before relying on this setting."* |

### Successful-config log line

Current line in `cli.py:286-289`:

```
KV cache quantization: {N}-bit, group_size={G}
```

Becomes one of:

- *"KV cache quantization: sliding=bf16, full=q8 (smart defaults), group_size=64"*
- *"KV cache quantization: sliding=q8, full=q4 (user override), group_size=64"*

Generated by `KVQuantPolicy.describe()`, which reads the `sliding_override` / `full_override` provenance flags carried on the policy itself (see the dataclass definition above). No recomputation, no auxiliary args.

## Testing

`pytest tests/` must stay green before any commit (CLAUDE.md rule, unchanged).

### `tests/test_kv_quant_policy.py` (new) — unit tests for `KVQuantPolicy`

- `bits_for("RotatingKVCache")` returns `sliding_bits`.
- `bits_for("KVCache")` returns `full_bits`.
- `bits_for("BatchKVCache")` returns `full_bits` (any class with "KVCache" in name).
- `bits_for("MambaCache")` (or any other class) returns `None`.
- Smart defaults: `KVQuantPolicy()` gives `sliding_bits=None, full_bits=8, sliding_override=False, full_override=False`.
- `describe()` reflects the `*_override` flags: with both false, the string includes "(smart defaults)"; with `full_override=True`, the full side is labelled "(user override)" independently of the sliding side; with both true, both sides are labelled.

### `tests/test_cache_translator.py` (extend existing)

- **Mixed-policy round-trip**: live-cache list with alternating `KVCache`/`RotatingKVCache` (mimicking Gemma 4's interleave). `_segment` with `KVQuantPolicy(sliding_bits=None, full_bits=8)` produces `KVLayerSegment`s where sliding layers hold raw float arrays and full layers hold `QuantizedArray`. Each segment's `metadata["bits"]` matches what was used.
- **`_assemble` reads bits from metadata**: build a list of `KVLayerSegment` with mixed `metadata["bits"]` directly (no policy in scope), call `_assemble`, assert correct cache types come back. Proves the read path doesn't depend on a policy being passed.
- **`KVLayerSegment.concat` invariant**: concatenating two segments with mismatched `metadata["bits"]` raises; two with matching bits succeeds.
- **`policy=None` disables everything**: `_segment` with `policy=None` produces all-float segments regardless of class.

### `tests/test_kv_quant_policy_config.py` (new) — CLI / SchedulerConfig validation

- `_build_kv_quant_policy(cfg)` returns `None` when `kv_cache_quantization=False`.
- Returns smart-default policy when on with no overrides.
- Applies overrides correctly when fields are set.
- Each warning trigger from the table above fires (`caplog` assertions): overrides-without-master-switch, explicit full=none (self-defeating), explicit sliding=int (sensitive-side opt-in), full≤4 (Qwen3.5-caveat info).
- **Sentinel distinguishes "unset" from "explicit none"**: CLI parsing produces `(sliding_bits=None, sliding_override=False)` when the flag is omitted, and `(sliding_bits=None, sliding_override=True)` when the user passes `--kv-cache-bits-sliding none`. Only the latter trips the self-defeating-combo warning (for the full side) or the sensitive-side warning (for sliding when set to an int).
- Argparse rejects `--kv-cache-quantization-bits` with the migration message.

### `tests/test_cache_hit_oom_repro.py` (existing)

Update the 60k-token Metal peak-memory regression test to run smart-defaults (currently uniform-q8) and re-baseline the expected peak. Sliding layers moving from quantized to float shifts the memory floor (slightly higher per-sliding-layer bytes, but no quantize/dequantize allocation churn on hits). The test verifies the new peak doesn't regress.
