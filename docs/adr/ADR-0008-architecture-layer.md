# ADR-0008: Architecture layer for model-specific behavior

**Status:** Accepted
**Date:** 2026-06-16

## Context

vllm-mlx supports Gemma 4, Qwen 3, Qwen 3.5 VL, GLM-4.6V, and Qwen 3-Next, plus MTP-equipped variants. Support is achieved today by monkey-patching mlx-lm at several lifecycle points (module import, model load, function scope) and by branching on architecture across the engine (`isinstance` on cache types, `hasattr` capability probes, special-token string sniffing, `model_type` switches across `scheduler.py`, `engine/batched.py`, `specprefill.py`, `utils/tokenizer.py`, `api/utils.py`, `mllm_batch_generator.py`). Failures degrade silently: a tokenizer without recognizable turn-end tokens makes `_compute_turn_boundaries` return `[]` and the turn cache becomes a no-op with no signal.

## Decision

Introduce an `Architecture` class per `model_type`, registered in `vllm_mlx/architectures/`, owning every model-specific decision. Engine call sites read `model_wrapper.architecture` and query or call it instead of branching. The cache layer (ADR-0003, 0004, 0005, 0006, 0007) is untouched.

**Protocol surface** (in `vllm_mlx/architectures/base.py`):

- Identity: `model_type: ClassVar[str]`, `aliases: ClassVar[tuple[str, ...]]`.
- Static capabilities: `has_mtp`, `has_vision`, `has_audio` — all `ClassVar[bool]`.
- Instance runtime state: `mtp_available: bool` — set by `validate()`/`install()` based on sidecar presence.
- Lifecycle: `validate(model, config, model_path)`, `install(model, config, model_path)`. Idempotent. `validate` raises `UnsupportedModelConfig` on precondition failure; `install` raises on mutation failure.
- Capability queries: `turn_end_token_ids(tokenizer) -> set[int] | None`, `extract_attention_query(layer, idx) -> mx.array | None`, `build_mtp_module(model, config) -> nn.Module | None`, `vision_processor(config) -> Any | None`. All return `Optional`; `None` means "not supported"; call sites log a warning once and degrade.

**Process-wide fixes**: `vllm_mlx/global_runtime_fixes.install_once()` applies architecture-agnostic mlx-lm patches (SDPA fixes). Called at server bootstrap *and* defensively at the start of `MLXLanguageModel.load()` / `MLXMultimodalLM.load()`. Idempotency is the contract.

**Registration**: `architectures.lookup(config)` checks `config.model_type` and `config.text_config.model_type`; unknown `model_type` raises `UnsupportedArchitectureError`. There is no generic fallback architecture — adding an mlx-lm-supported model to vllm-mlx is a contributor task that requires writing an `Architecture` subclass.

## Behavior changes

- **Loading an unregistered `model_type` now fails at model load** instead of producing a model with all capabilities silently disabled.
- **`_compute_turn_boundaries` no longer falls back to O(N²) re-render**. A tokenizer without `turn_end_token_ids` produces a `WARNING` and a turn-cache no-op for that model.
- **MTP load failure is still defensive**, but routed through `Architecture.mtp_available`: `validate()` detects missing sidecar, clears `mtp_available`, the scheduler skips MTP installation. Today's "model loads without MTP" UX is preserved; the failure mode is named and observable.
- **SpecPrefill on an architecture that returns `None` from `extract_attention_query`** raises `SpecPrefillUnsupportedError` at the first SpecPrefill request rather than silently miscomputing queries.

## Rejected alternatives

- **Multi-architecture-per-process as a v1 goal.** Considered designing the protocol for per-instance patching with an `installed_for` weak-set guard. Rejected: aspirational only — no current use case loads two architectures in one process, and the day-1 class-level patches don't actually deliver the property regardless. The `Architecture` instance is owned by the model wrapper; if multi-architecture lands later, the protocol changes then.

- **Generic fallback architecture for unknown `model_type`.** Considered registering a `GenericTextArchitecture` with all-default methods so any mlx-lm-supported model loads. Rejected: contradicts the "every silent degradation becomes a loud, named error" thesis. Failing at registration is the loudest possible signal. The cost (every new mlx-lm model needs an adapter PR) is acceptable for a project that already requires per-model expertise.

- **Exception-as-control-flow for capability declarations.** Considered raising `NoTurnDelimitersError`, `SpecPrefillUnsupportedError`, etc. from architecture methods to indicate "not supported." Rejected: capability queries run on the hot path (per-request `_compute_turn_boundaries`); exceptions for control flow there are a smell. Loud failure is achieved by call-site logging on `None`, not by raising. Exceptions are reserved for `validate()` (failed precondition) and `install()` (failed mutation).

- **Renaming the cache layer to break the `adapter` name collision.** `prefix_cache_adapters.py` already uses "adapter" for `CacheManager` subclasses. Considered renaming it to free up the word. Rejected: cache layer is older, structurally load-bearing, and covered by three ADRs. The new layer uses `Architecture` everywhere instead; the cache layer keeps its vocabulary.

## Consequences

- **Module layout**: new `vllm_mlx/architectures/` package; new `vllm_mlx/global_runtime_fixes.py`. The `vllm_mlx/patches/` directory is retired — content migrates into per-architecture `install()` and `global_runtime_fixes`.
- **Engine call sites**: `scheduler.py:945` MTP gate, `engine/batched.py:_compute_turn_boundaries`, `specprefill.py:328` extractor switch, `api/utils.py:is_mllm_model`, and the MLLM vision-processor wiring all read `model_wrapper.architecture` and call into it.
- **`utils/tokenizer.py:_try_inject_mtp` is deleted**. MTP injection happens during `Architecture.install()` at model load.
- **CacheManager is unchanged.** `_ensure_cache_index_map`, segmentation, assembly, quantization policy by layer type, the `step` padding formulas, and Active Leaf pinning all stay. The architecture layer stops at the engine; cache internals remain structural and architecture-agnostic.

## Seam — `Architecture` vs `CacheManager`

The `Architecture` layer answers "what does *this model* need?" The `CacheManager` layer answers "how do we store and reconstruct KV state?" Both layers are structural inside their own scope and do not branch on each other: `CacheManager` keys its decisions on cache class names (ADR-0007), never on `model_type`; `Architecture` never inspects cache internals. The two layers meet only at `model_wrapper`, which carries both.

## Out of scope

- Forking mlx-lm. Patches remain runtime-applied; upstream PRs are welcome but not on this work's critical path.
- Adding new model support. The migration moves existing patches into the new shape; it does not extend the set of supported architectures.
- Hybrid-attention models (mixed full/sliding/recurrent). They keep working via the existing structural cache classification (ADR-0007) and do not need architecture-side hooks.
- Chunked-prefill or continuous-batching mechanics. These live in mlx-lm's `BatchGenerator`; the architecture layer does not extend into the batched runtime.
