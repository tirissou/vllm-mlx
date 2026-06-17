# Architecture layer — Design

**Date:** 2026-06-16
**Status:** Accepted (see ADR-0008)
**Scope:** Replace today's scattered monkey-patching of mlx-lm with a per-architecture `Architecture` class that centralizes everything model-specific behind one protocol.

> History note: this spec was grilled and tightened in a `grill-with-docs` session on 2026-06-16. Eight design decisions (D1–D8) emerged from that session and are reflected throughout. The accepted decisions are formalized in `docs/adr/ADR-0008-architecture-layer.md`; this spec is the implementation-level companion.

---

## Motivation

vllm-mlx supports Gemma 4, Qwen 3, Qwen 3.5 VL, GLM-4.6V, and Qwen 3-Next today, plus MTP-equipped variants. Support is achieved by monkey-patching mlx-lm classes at several lifecycle points (module import, MLLM init, model load, function scope) and by branching on architecture across the engine (`isinstance` on cache types, `hasattr` capability probes, special-token string sniffing, `model_type` switches). The result is a code base where:

- Adding an architecture requires touching `patches/`, `scheduler.py`, `engine/batched.py`, `specprefill.py`, `utils/tokenizer.py`, and `mllm_batch_generator.py` — with no single place to look.
- Failures degrade silently: a tokenizer without recognizable turn-end tokens makes `_compute_turn_boundaries` return `[]`, which makes the turn cache a no-op without any signal.
- There is no single seam where architecture-specific behavior lives.

This redesign replaces those patterns with one `Architecture` per architecture, registered by `model_type`, owning every model-specific decision. It does **not** change the cache layer (`CacheManager`, `TurnPrefixCache`, `_segment`/`_assemble`, `BatchQuantizedKVCache`), the batched runtime (mlx-lm's `BatchGenerator`), or the request/scheduler flow.

## Constraints decided during brainstorming

- **(a) mlx-lm appetite:** patch-only. mlx-lm stays vendored; the architecture layer organizes existing patches rather than forking. Upstream-PR-friendly code is welcome but no fork.
- **(c) Scope:** dense text LLMs + MTP variants + MLLMs. Hybrid-attention models (mixed full/sliding/recurrent) keep working via the existing structural cache classification; they don't need architecture-side hooks.

## Non-goals

- Forking mlx-lm.
- Touching the cache layer (`CacheManager`, segmentation, assembly, quantization policy, Active Leaf pinning, the `step` padding formulas, `_ensure_cache_index_map`).
- Adding new model support as part of this work.
- Changing the chunked-prefill or continuous-batching mechanism (lives in mlx-lm's `BatchGenerator`).
- Adding a `chunk_size_hint` or other batched-runtime knobs to `Architecture`.
- **Multi-architecture-per-process (D1).** The `Architecture` instance is owned by the model wrapper, but day-1 `install()` implementations patch mlx-lm at the class level. No forward-compatibility scaffolding for instance-level patching ships in v1. If the use case ever materializes, the protocol changes then.

## Module layout

```
vllm_mlx/
  architectures/                  # NEW
    __init__.py                   # registry + lookup
    base.py                       # Architecture protocol + exception types
    gemma4.py                     # Gemma 4 text
    qwen3.py                      # Qwen 3 text
    qwen3_5_vl.py                 # Qwen 3.5 VL (MLLM)
    qwen3_next.py                 # Qwen 3-Next (with MTP)
    glm4v_moe.py                  # GLM-4.6V
    _mtp_common.py                # Shared MTP scaffolding (the 90% common to qwen3_5_mtp + qwen3_next_mtp)
    _testing.py                   # FakeArchitecture for engine tests
  global_runtime_fixes.py         # NEW — architecture-agnostic mlx-lm fixes (SDPA patches)
  patches/                        # RETIRED — content migrates to architectures/* and global_runtime_fixes.py
  models/
    llm.py                        # MLXLanguageModel gains .architecture attribute
    mllm.py                       # MLXMultimodalLM gains .architecture attribute
```

The cache layer (`vllm_mlx/prefix_cache_adapters.py`, `turn_prefix_cache.py`, `batch_quantized_kv_cache.py`, `kv_cache.py`, `cache_types.py`) is unchanged. ADRs 0003, 0004, 0005, 0006, 0007 continue to apply.

**Naming note (D6):** `prefix_cache_adapters.py` already uses the word "adapter" for `CacheManager` subclasses. The new layer is named `Architecture` end-to-end (package: `architectures/`; protocol: `Architecture`; subclasses: `Gemma4Architecture`, `Qwen3Architecture`, etc.; attribute: `model_wrapper.architecture`; test fake: `FakeArchitecture`). The cache layer keeps its existing vocabulary. `grep -rn "adapter"` continues to surface cache-adapter code only.

## Architecture protocol

```python
# vllm_mlx/architectures/base.py

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar
import mlx.core as mx
import mlx.nn as nn


class Architecture(ABC):
    # Identity
    model_type: ClassVar[str]                       # matches HF config.model_type
    aliases: ClassVar[tuple[str, ...]] = ()         # for renames or text_config variants

    # Static capabilities (class-level facts about the architecture)
    has_mtp: ClassVar[bool] = False
    has_vision: ClassVar[bool] = False
    has_audio: ClassVar[bool] = False

    def __init__(self) -> None:
        # Per-instance runtime state set by validate()/install(). Distinct from
        # the static has_mtp classvar: mtp_available answers "did MTP load
        # successfully for *this specific model*?", not "does this architecture
        # support MTP at all?".
        self.mtp_available: bool = self.has_mtp

    # ── Install lifecycle ────────────────────────────────────────────
    def validate(self, model: nn.Module, config: Any, model_path: Path) -> None:
        """Raise UnsupportedModelConfig if this model + config combination is
        known to be unsupported (e.g. RotatingKVCache.keep > 0).

        MTP architectures also check sidecar weight presence here and clear
        self.mtp_available if the file is missing — that case is a warning,
        not a load failure (D8).

        Default: no-op. Called from MLXLanguageModel.load() before install().
        """
        return None

    def install(self, model: nn.Module, config: Any, model_path: Path) -> None:
        """Patch this model so batched prefill/decode work. Idempotent.

        Day-1 implementations patch mlx-lm at the class level (D1 — no
        per-instance patching scaffolding).

        Override to:
          - swap __call__ on attention modules to handle per-batch cache.offset
          - load + attach MTP weights via _mtp_common
          - install vision-tower hooks for MLLM

        Raises if a mutation fails (e.g. MTPWeightsLoadError on a corrupt
        sidecar). Does NOT raise for the missing-sidecar case — that path is
        already gated by self.mtp_available, cleared in validate().
        """
        return None

    # ── Capability queries (D4 — all return Optional, never raise) ───
    def turn_end_token_ids(self, tokenizer) -> set[int] | None:
        """Tokens that mark the end of a conversational turn. Returns None if
        the tokenizer does not expose them; engine logs a warning once and
        disables turn cache for this model.
        """
        return None

    def extract_attention_query(self, attn_layer, layer_idx: int) -> mx.array | None:
        """Post-RoPE query vector for SpecPrefill importance scoring. Returns
        None when SpecPrefill is unavailable for this architecture.
        """
        return None

    def build_mtp_module(self, model, config) -> nn.Module | None:
        """Construct + attach the MTP module if this architecture has MTP.
        Called from install() for has_mtp=True architectures. The architecture
        loads its own sidecar weights inside this method (D5). Returns None
        for non-MTP architectures.
        """
        return None

    def vision_processor(self, config):
        """Return the multimodal processor. None for text-only architectures."""
        return None
```

**Exception types (in `architectures/base.py`):**

```python
class UnsupportedArchitectureError(Exception):
    """No Architecture subclass registered for this config.model_type. Raised
    at model load (D2). Adding an mlx-lm-supported model to vllm-mlx requires
    writing an Architecture subclass — there is no generic fallback."""

class UnsupportedModelConfig(Exception):
    """Architecture exists but this config combination is unsupported.
    Raised from validate() at model load."""

class MTPWeightsLoadError(Exception):
    """install() failed to load or attach MTP sidecar weights despite the
    sidecar file being present. The missing-sidecar case is not an error;
    it sets self.mtp_available = False in validate() and is observable
    through the load-time warning (D8)."""
```

Note (D4): there is no `NoTurnDelimitersError` or `SpecPrefillUnsupportedError`. Capability queries return `Optional`; the engine call sites log warnings on `None` and degrade. Exceptions are reserved for `validate()` (failed precondition) and `install()` (failed mutation).

**Registry (in `architectures/__init__.py`):**

```python
_REGISTRY: dict[str, type[Architecture]] = {}

def register(cls):
    _REGISTRY[cls.model_type] = cls
    for alias in cls.aliases:
        _REGISTRY[alias] = cls
    return cls

def lookup(config) -> Architecture:
    candidates = [
        getattr(config, "model_type", None),
        getattr(getattr(config, "text_config", None), "model_type", None),
    ]
    for mt in candidates:
        if mt in _REGISTRY:
            return _REGISTRY[mt]()
    raise UnsupportedArchitectureError(
        f"No Architecture for model_type in {candidates}. "
        f"Registered: {sorted(_REGISTRY)}"
    )
```

## Lifecycle

**Process startup (server bootstrap):**

```python
from vllm_mlx import global_runtime_fixes
global_runtime_fixes.install_once()
```

**Defensive bootstrap (D3) inside model load:**

```python
def MLXLanguageModel.load(...):
    from vllm_mlx import global_runtime_fixes
    global_runtime_fixes.install_once()           # idempotent — safe to repeat

    model, tokenizer = mlx_lm.load(model_path, ...)
    architecture = architectures.lookup(model.config)
    architecture.validate(model, model.config, Path(model_path))
    architecture.install(model, model.config, Path(model_path))

    self.model = model
    self.tokenizer = tokenizer
    self.architecture = architecture
```

`install_once()` is idempotent and applies the two architecture-agnostic mlx-lm fixes:

- `apply_prefill_flash_sdpa()` — routes quantized prefill through dequantize + `mx.fast.scaled_dot_product_attention` (was `patches/mlx_lm_prefill_flash_sdpa.py`).
- `patch_quantized_sdpa()` — expands 4D mask to 5D for GQA + batch ≥ 2 (was `patches/mlx_lm_quantized_sdpa.py`).

Both fixes affect every architecture identically and are upstream candidates. The module docstring documents this and the import-order requirement (must run before any mlx-lm class is touched at load time because `apply_prefill_flash_sdpa` walks `sys.modules`). Calling from both the server bootstrap *and* `MLXLanguageModel.load()` / `MLXMultimodalLM.load()` is the contract — the second call short-circuits.

**MTP injection** moves into `Architecture.install()`. The two near-duplicate files (`patches/qwen3_5_mtp.py`, `patches/qwen3_next_mtp.py`) consolidate into shared scaffolding in `architectures/_mtp_common.py` plus thin per-architecture implementations of `build_mtp_module()`. Sidecar weight loading happens inside the architecture's `install()` — the protocol exposes `model_path` for exactly this purpose (D5).

**Defensive MTP UX (D8).** `validate()` checks for the MTP sidecar file at `model_path / "mtp" / "weights.safetensors"` or `model_path / "model-mtp.safetensors"`. If absent, it sets `self.mtp_available = False` and logs a warning; the model still loads. The scheduler gates MTP installation on `architecture.mtp_available`, not `has_mtp`. A corrupt-but-present sidecar still raises `MTPWeightsLoadError` and aborts load — that case is a real failure, not an opt-out.

## Engine integration

The pattern at every call site is the same: **read `model_wrapper.architecture` once, query or call it instead of branching.**

**`vllm_mlx/scheduler.py:41–43`** — three eager installs at module import are removed. They move into `global_runtime_fixes.install_once()` (the two SDPA fixes) and the Gemma 4 architecture's `install()` (the offset-snapshot patch).

**`vllm_mlx/scheduler.py:945–957`** — MTP gate becomes a typed capability check:

```python
if self.config.enable_mtp:
    if self.model_wrapper.architecture.mtp_available:        # D8: runtime, not classvar
        _install_mtp(bg, model=self.model, ...)
    else:
        logger.warning(
            f"[MTP] not available for {self.model_wrapper.architecture.model_type} "
            f"(architecture has_mtp={self.model_wrapper.architecture.has_mtp})"
        )
```

**`vllm_mlx/engine/batched.py:_compute_turn_boundaries`** collapses from ~155 lines to ~15. The honest collapsed form preserves the system-message guard (D7) and the MLLM-aware `_apply_chat_template` arguments:

```python
def _compute_turn_boundaries(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    num_images: int = 0,
    num_audios: int = 0,
    chat_template_kwargs: dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
) -> list[int]:
    # System-message guard — same invariant as today: turn cache assumes the
    # system prompt is the stable prefix.
    if not messages or messages[0].get("role") != "system":
        return []

    architecture = self.model_wrapper.architecture
    end_ids = architecture.turn_end_token_ids(self.tokenizer)
    if end_ids is None:
        logger.warning(
            f"[turn_cache] disabled for {architecture.model_type}: "
            f"tokenizer has no turn delimiters"
        )
        return []

    rendered = self._apply_chat_template(
        messages,
        tools=tools,
        num_images=num_images,
        num_audios=num_audios,
        chat_template_kwargs={**(chat_template_kwargs or {}), "add_generation_prompt": False},
        enable_thinking=enable_thinking,
    )
    tokens = self.tokenizer.encode(rendered)
    return [i + 1 for i, tok in enumerate(tokens) if tok in end_ids]
```

The three branches (Qwen `<|im_end|>` scan, Gemma `<turn|>`/`<tool_response|>` scan, O(N²) re-render fallback) consolidate into the engine's single scan plus each architecture's `turn_end_token_ids`. The fallback is **dropped**; a tokenizer without delimiters now produces a `WARNING` and a turn-cache no-op for that model, observable through logs.

The MLLM-aware arguments (`num_images`, `num_audios`, `enable_thinking`, `chat_template_kwargs`) must stay in the signature: the chat template renders differently when multimodal placeholders are present, so the tokenization used for boundary detection must match the one used for prefill.

**`vllm_mlx/specprefill.py:328–344`** — the `model_type → extractor` switch becomes:

```python
query = architecture.extract_attention_query(attn_layer, layer_idx)
if query is None:
    logger.warning(
        f"[specprefill] disabled for {architecture.model_type}: "
        f"extract_attention_query unavailable"
    )
    return None        # caller treats None as "skip importance scoring for this request"
```

Five branches plus a `nemotron_h` fallback move into the respective architectures or a shared default in `architectures/_mtp_common.py` that architectures opt into.

**`vllm_mlx/api/utils.py:is_mllm_model`** — name-based regex becomes typed capability:

```python
def is_mllm_model(model_wrapper) -> bool:
    return model_wrapper.architecture.has_vision or model_wrapper.architecture.has_audio
```

Routing decisions in `model_registry.py:783–795` use this. Resistant to HF renames.

**Vision processor** lookup at MLLM pipeline construction:

```python
processor = architecture.vision_processor(config)
if processor is None and (architecture.has_vision or architecture.has_audio):
    logger.warning(
        f"[mllm] architecture {architecture.model_type} declares vision/audio but "
        f"returned no processor — multimodal inputs will fail"
    )
```

Subsumes the architecture-specific wiring in `mllm_batch_generator.py` and `multimodal_processor.py`.

**`vllm_mlx/utils/tokenizer.py:_try_inject_mtp`** is deleted. MTP injection happens during `architecture.install()` at model load.

**CacheManager — unchanged.** `_ensure_cache_index_map` (`prefix_cache_adapters.py:197`), segmentation, assembly, quantization policy by layer type, the `step` padding formulas, and Active Leaf pinning all stay. The architecture layer stops at the engine; cache internals remain structural and architecture-agnostic.

## Behavior changes (user-visible)

| Today | Redesign |
|---|---|
| Any mlx-lm-supported model loads; unsupported capabilities silently no-op | `UnsupportedArchitectureError` at load for unregistered `model_type` (D2). Adding a model is now a contributor task — write an `Architecture` subclass. |
| `_compute_turn_boundaries` returns `[]` when tokenizer lacks delimiters | `WARNING` log; turn cache disabled for this model (D4). |
| `_compute_turn_boundaries` falls back to O(N²) re-render | Fallback dropped. Architectures must expose `turn_end_token_ids`. |
| `hasattr(model, "mtp")` False because injection never ran | Either: `mtp_available = False` set in `validate()` for missing sidecar (warning, model loads without MTP — D8), or `MTPWeightsLoadError` at model load for corrupt sidecar (real failure). |
| `RotatingKVCache.keep > 0` raises `ValueError` at first cache construction | `UnsupportedModelConfig` from `validate()` at model load. |
| Unknown `model_type` causes downstream `AttributeError`/wrong behavior | `UnsupportedArchitectureError` at model load with registered types listed. |
| SpecPrefill on unknown architecture silently misses or returns wrong queries | `WARNING` log; SpecPrefill skipped for this request (D4). |

## Failure modes

Every silent degradation today becomes either a loud, named exception (state-changing failures: `validate`, `install`) or a warned-then-degraded path (capability declarations: `turn_end_token_ids`, `extract_attention_query`, `build_mtp_module`, `vision_processor`). The split is deliberate (D4): exceptions for things that should abort, `Optional` for things that should be observable but recoverable on the hot path.

## Testing strategy

**1. Per-architecture unit tests** (`tests/architectures/test_<architecture>.py`):

- Construct a mocked model exposing only the bits the architecture touches: `config`, a handful of `nn.Module` attention layers.
- Assert `install()` mutates them correctly (e.g. `model.layers[0].self_attn.__call__` is rebound; `model.mtp is not None` for MTP architectures with a sidecar present).
- Assert `turn_end_token_ids()` returns the right IDs against a real tokenizer fixture (small per-architecture JSON of `{token_string → id}`).
- Assert `validate()` raises `UnsupportedModelConfig` on known-unsupported configs.
- Assert `validate()` clears `mtp_available` and logs a warning when the sidecar file is absent (D8).

**2. Engine tests using `FakeArchitecture`** (`vllm_mlx/architectures/_testing.py`):

- Drop-in `Architecture` with configurable returns: `FakeArchitecture(turn_end_ids={5, 7}, mtp_available=True, ...)`.
- Engine tests for `_compute_turn_boundaries`, MTP gating, SpecPrefill routing use this — no real model needed and no dependence on the registered architecture set.

**3. Existing integration tests unchanged:**

- `tests/test_cache_hit_oom_repro.py` (Metal peak-memory at production scale) validates that the architecture-driven path doesn't regress the OOM fix.
- `tests/test_cache_translator.py`, `tests/test_turn_prefix_cache_integration.py` test the CacheManager surface, which is untouched.
- `tests/test_prefix_cache_adapters.py` continues to test the CacheManager.

**4. One-off migration test (not committed):**

A script that loads each currently-supported model, runs a 4-turn conversation with turn cache on, and asserts hit/miss/save behavior matches what today's code produces. Catches semantic drift during the cutover. Thrown away after.

## Migration plan

1. Land `architectures/base.py` (protocol, exceptions, registry) and `global_runtime_fixes.py` (extracted SDPA patches). No behavior change; nothing imports the new module yet.
2. Wire `MLXLanguageModel.load()` and `MLXMultimodalLM.load()` to call `global_runtime_fixes.install_once()` (D3), then `architectures.lookup()` + `validate()` + `install()`. Architectures still no-op at this point.
3. Land per-architecture classes one at a time, each moving its slice from `patches/` and call-site branches. After each architecture lands, the corresponding `patches/` file is deleted and the call sites in `scheduler.py`, `engine/batched.py`, `specprefill.py`, `api/utils.py` switch to architecture calls for that `model_type`.
4. After all architectures migrate, delete `patches/` and `utils/tokenizer.py:_try_inject_mtp`. Engine call sites now uniformly read `model_wrapper.architecture`.
5. Run the one-off migration test to confirm parity with today's behavior.

Each step is a separately reviewable PR. Step 3 admits per-architecture parallelism.

## Estimated diff size

| File | Lines changed | Nature |
|---|---|---|
| `vllm_mlx/architectures/` (new) | +400 (5 architectures + base + registry + MTP common) | New |
| `vllm_mlx/global_runtime_fixes.py` (new) | +80 | Extracted from `patches/` |
| `vllm_mlx/patches/` (removed) | −600 | Deletion |
| `vllm_mlx/engine/batched.py` | −140 / +30 | `_compute_turn_boundaries` collapses (~15 lines per D7) |
| `vllm_mlx/scheduler.py` | −10 / +5 | MTP gate (`mtp_available`), removing 3 eager installs |
| `vllm_mlx/specprefill.py` | −40 / +5 | Extractor switch becomes method call |
| `vllm_mlx/api/utils.py` | −20 / +5 | `is_mllm_model` |
| `vllm_mlx/models/llm.py`, `mllm.py` | +15 each | `install_once()` + lookup + validate + install at load |
| `vllm_mlx/utils/tokenizer.py` | −60 | `_try_inject_mtp` deleted |

Net: roughly −280 lines of scattered branching for +480 lines of organized architectures. Not a code-size win — a seam win.

## Decisions (summary)

The eight decisions surfaced during the grilling session and reflected throughout this spec:

- **D1** — Multi-architecture-per-process is a v1 non-goal. No `installed_for: WeakSet` scaffolding, no "forward-compat for instance patching" claim. Day-1 `install()` patches mlx-lm classes globally.
- **D2** — Strict failure on unknown `model_type` at model load. No generic fallback `Architecture`. Adding a model is a contributor task.
- **D3** — `global_runtime_fixes.install_once()` is called both at server bootstrap *and* defensively at the start of each `MLXLanguageModel.load()` / `MLXMultimodalLM.load()`. Idempotency is the contract.
- **D4** — Capability queries return `Optional`, never raise. Engine call sites log a warning once and degrade. Exceptions reserved for `validate()` (failed precondition) and `install()` (failed mutation).
- **D5** — `install(model, config, model_path)` and `validate(model, config, model_path)`. `build_mtp_module(model, config)` — no `weights` arg; the architecture loads its own sidecar.
- **D6** — Rename to `Architecture` throughout the new layer. `model_wrapper.architecture`, `Gemma4Architecture`, `FakeArchitecture`. Cache layer keeps its existing vocabulary.
- **D7** — Honest `_compute_turn_boundaries` is ~15 lines, preserves the "system message must be first" guard, and keeps the MLLM-aware `_apply_chat_template` arguments (`num_images`, `num_audios`, `enable_thinking`, `chat_template_kwargs`).
- **D8** — `has_mtp: ClassVar[bool]` is the static architecture capability; `mtp_available: bool` is the per-instance runtime state set in `validate()`/`install()` based on sidecar presence. Scheduler gates on `mtp_available`. Missing sidecar is a warning, not a load failure.

## Open questions

Surface during implementation:

- Whether the `_mtp_common.py` scaffolding can absorb 100% of the duplication between `qwen3_5_mtp` and `qwen3_next_mtp`, or whether per-architecture variation justifies leaving some duplication. Decision deferred to the MTP-architecture PR.
- Whether `vision_processor()` should be cached on the `Architecture` instance or constructed fresh per call. Day-1: fresh per call (today's behavior); if pipeline construction frequency turns out to matter, cache later.
- Exact shape of `FakeArchitecture` — start minimal (just the fields tests need) and grow on demand.
