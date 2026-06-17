# Architecture Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered monkey-patching of mlx-lm with a single `Architecture` class per `model_type` that owns every model-specific decision, registered in `vllm_mlx/architectures/`, queried by the engine through `model_wrapper.architecture`.

**Architecture:** New `vllm_mlx/architectures/` package with an abstract `Architecture` base class, a registry keyed on `config.model_type`, and one subclass per supported model. Architecture-agnostic mlx-lm fixes extract into `vllm_mlx/global_runtime_fixes.py`. The cache layer (`prefix_cache_adapters.py`, ADRs 0003/0004/0005/0006/0007) is unchanged.

**Tech Stack:** Python 3.11+, mlx, mlx-lm, mlx-vlm, pytest.

## Global Constraints

- **Spec & ADR are source of truth.** `docs/superpowers/specs/2026-06-16-architecture-adapters-design.md` and `docs/adr/ADR-0008-architecture-layer.md`. Decisions D1–D8 are non-negotiable.
- **`Architecture` (not `Adapter`).** The class is `Architecture`; subclasses are `Gemma4Architecture`, `Qwen3Architecture`, `Qwen3_5VLArchitecture`, `Qwen3NextArchitecture`, `GLM4VMoEArchitecture`; attribute is `model_wrapper.architecture`. The cache layer keeps its existing "adapter" vocabulary; do not rename it.
- **Capability methods return `Optional`, never raise.** `turn_end_token_ids`, `extract_attention_query`, `build_mtp_module`, `vision_processor` return `T | None`. Engine call sites log a `WARNING` once and degrade. Exceptions are reserved for `validate()` and `install()`.
- **Strict failure on unknown `model_type`.** `lookup()` raises `UnsupportedArchitectureError`. No generic fallback.
- **Idempotent global fixes.** `global_runtime_fixes.install_once()` must be safe to call repeatedly. Server bootstrap and every `MLXLanguageModel.load()` / `MLXMultimodalLM.load()` both call it.
- **`mtp_available` is per-instance.** `has_mtp: ClassVar[bool]` is static; `self.mtp_available: bool` is set by `validate()`/`install()` based on sidecar presence. Scheduler gates on `mtp_available`.
- **Cache layer untouched.** Do not edit `prefix_cache_adapters.py`, `turn_prefix_cache.py`, `batch_quantized_kv_cache.py`, `kv_cache.py`, `cache_types.py`. If a task seems to require it, stop and re-read ADR-0008.
- **One PR per task.** Each task is a separately reviewable commit. Commit messages should reference the task number and the ADR (e.g., `feat(architectures): add base protocol [ADR-0008, Task 1]`).
- **Tests before code.** Every task lands its tests in the same PR. The migration parity script (Task 20) is the last gate.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `vllm_mlx/architectures/__init__.py` | Registry (`register`, `lookup`, `_REGISTRY`); module-level imports of all concrete architectures so registration runs at import time. |
| `vllm_mlx/architectures/base.py` | `Architecture` abstract class, exception types (`UnsupportedArchitectureError`, `UnsupportedModelConfig`, `MTPWeightsLoadError`). |
| `vllm_mlx/architectures/_testing.py` | `FakeArchitecture` for engine tests. |
| `vllm_mlx/architectures/_mtp_common.py` | Shared MTP scaffolding extracted from the two `patches/*_mtp.py` files. |
| `vllm_mlx/architectures/gemma4.py` | `Gemma4Architecture` — turn-end tokens, attention offset-snapshot patch. |
| `vllm_mlx/architectures/qwen3.py` | `Qwen3Architecture` — turn-end tokens, SpecPrefill query extractor. |
| `vllm_mlx/architectures/qwen3_5_vl.py` | `Qwen3_5VLArchitecture` — MLLM + MTP + SpecPrefill. |
| `vllm_mlx/architectures/qwen3_next.py` | `Qwen3NextArchitecture` — MTP + SpecPrefill. |
| `vllm_mlx/architectures/glm4v_moe.py` | `GLM4VMoEArchitecture` — MLLM. |
| `vllm_mlx/global_runtime_fixes.py` | `install_once()` applying SDPA fixes; idempotent. |
| `tests/architectures/test_base.py` | Registry, `lookup`, exception behavior. |
| `tests/architectures/test_<name>.py` | Per-architecture install/validate/capability-query tests. |
| `tests/architectures/test_fake.py` | `FakeArchitecture` shape used by engine tests. |
| `scripts/migration_parity_check.py` | Throwaway parity script (Task 20). |

**Modified files:**

| Path | Change |
|---|---|
| `vllm_mlx/models/llm.py` | `MLXLanguageModel.load()` calls `install_once()` + `lookup()` + `validate()` + `install()`. Gain `self.architecture`. |
| `vllm_mlx/models/mllm.py` | Same for `MLXMultimodalLM.load()`. |
| `vllm_mlx/scheduler.py` | Remove eager `patches/*` calls at module top; MTP gate uses `architecture.mtp_available`. |
| `vllm_mlx/engine/batched.py` | `_compute_turn_boundaries` collapses to ~15 lines using `architecture.turn_end_token_ids`. |
| `vllm_mlx/specprefill.py` | `_EXTRACTOR_REGISTRY` switch replaced by `architecture.extract_attention_query`. |
| `vllm_mlx/api/utils.py` | `is_mllm_model(model_wrapper)` signature change to use `architecture.has_vision/has_audio`. |
| `vllm_mlx/mllm_batch_generator.py` | Vision processor wiring routes through `architecture.vision_processor(config)`. |
| `vllm_mlx/utils/tokenizer.py` | Delete `_try_inject_mtp` and `_try_inject_mtp_post_load`. |

**Deleted files (Task 19):** `vllm_mlx/patches/__init__.py`, `gemma4_llm.py`, `glm4v_moe_mllm.py`, `mlx_lm_prefill_flash_sdpa.py`, `mlx_lm_quantized_sdpa.py`, `qwen3_5_mllm.py`, `qwen3_5_mtp.py`, `qwen3_next_mtp.py`.

---

## Task Dependency Order

```
Task 1 (base) ─┬─→ Task 3 (FakeArchitecture) ─┐
               │                              ├─→ Task 7 (turn_boundaries collapse)
Task 2 (global_runtime_fixes) ─┐              ├─→ Task 8 (specprefill collapse)
                               ├─→ Task 5, 6  ├─→ Task 9 (is_mllm_model)
Task 4 (skeleton architectures)┘              │
                                              ├─→ Task 10 (Gemma4)
                                              ├─→ Task 11 (Qwen3)
                                              ├─→ Task 12 (_mtp_common)
                                              │     └─→ Task 13 (Qwen3_5VL)
                                              │     └─→ Task 14 (Qwen3Next)
                                              ├─→ Task 15 (GLM4V_MoE)
                                              ├─→ Task 16 (MTP gate)
                                              └─→ Task 17 (vision_processor wiring)
                                                    └─→ Task 18 (delete _try_inject_mtp)
                                                          └─→ Task 19 (delete patches/)
                                                                └─→ Task 20 (parity script)
```

Tasks 10, 11, 15 are parallelizable. Tasks 13 and 14 depend on Task 12.

---

## Task 1: Base protocol, exceptions, registry

**Files:**
- Create: `vllm_mlx/architectures/__init__.py`
- Create: `vllm_mlx/architectures/base.py`
- Create: `tests/architectures/__init__.py`
- Create: `tests/architectures/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Architecture(ABC)` with `model_type: ClassVar[str]`, `aliases: ClassVar[tuple[str, ...]]`, `has_mtp/has_vision/has_audio: ClassVar[bool]`, `mtp_available: bool`, methods `validate(model, config, model_path) -> None`, `install(model, config, model_path) -> None`, `turn_end_token_ids(tokenizer) -> set[int] | None`, `extract_attention_query(layer, idx) -> mx.array | None`, `build_mtp_module(model, config) -> nn.Module | None`, `vision_processor(config) -> Any | None`.
  - `register(cls)` decorator.
  - `lookup(config) -> Architecture`.
  - Exceptions: `UnsupportedArchitectureError`, `UnsupportedModelConfig`, `MTPWeightsLoadError`.

- [ ] **Step 1: Write failing tests for registry and lookup**

`tests/architectures/test_base.py`:

```python
import pytest
from types import SimpleNamespace

from vllm_mlx.architectures import (
    Architecture,
    UnsupportedArchitectureError,
    register,
    lookup,
    _REGISTRY,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_register_adds_class_to_registry():
    @register
    class Dummy(Architecture):
        model_type = "dummy_text"

    assert _REGISTRY["dummy_text"] is Dummy


def test_register_records_aliases():
    @register
    class DummyAliased(Architecture):
        model_type = "dummy_canonical"
        aliases = ("dummy_old_name",)

    assert _REGISTRY["dummy_canonical"] is DummyAliased
    assert _REGISTRY["dummy_old_name"] is DummyAliased


def test_lookup_uses_top_level_model_type():
    @register
    class Top(Architecture):
        model_type = "top_only"

    arch = lookup(SimpleNamespace(model_type="top_only"))
    assert isinstance(arch, Top)


def test_lookup_falls_back_to_text_config_model_type():
    @register
    class Nested(Architecture):
        model_type = "nested_text"

    cfg = SimpleNamespace(
        model_type="wrapper", text_config=SimpleNamespace(model_type="nested_text")
    )
    arch = lookup(cfg)
    assert isinstance(arch, Nested)


def test_lookup_raises_for_unknown_model_type():
    with pytest.raises(UnsupportedArchitectureError) as exc:
        lookup(SimpleNamespace(model_type="nope"))
    assert "nope" in str(exc.value)


def test_mtp_available_defaults_to_has_mtp():
    class WithMtp(Architecture):
        model_type = "with_mtp"
        has_mtp = True

    class WithoutMtp(Architecture):
        model_type = "without_mtp"

    assert WithMtp().mtp_available is True
    assert WithoutMtp().mtp_available is False


def test_default_capability_queries_return_none():
    class Plain(Architecture):
        model_type = "plain"

    a = Plain()
    assert a.turn_end_token_ids(tokenizer=None) is None
    assert a.extract_attention_query(attn_layer=None, layer_idx=0) is None
    assert a.build_mtp_module(model=None, config=None) is None
    assert a.vision_processor(config=None) is None


def test_default_lifecycle_is_noop():
    class Plain(Architecture):
        model_type = "plain2"

    a = Plain()
    a.validate(model=None, config=None, model_path=None)
    a.install(model=None, config=None, model_path=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/architectures/test_base.py -v`
Expected: ImportError (`vllm_mlx.architectures` does not exist yet).

- [ ] **Step 3: Implement `base.py`**

`vllm_mlx/architectures/base.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Architecture protocol — one class per supported model_type.

See docs/adr/ADR-0008-architecture-layer.md for design rationale.
"""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar


class UnsupportedArchitectureError(Exception):
    """No Architecture subclass registered for this config.model_type.

    Raised at model load (D2). Adding an mlx-lm-supported model to vllm-mlx
    requires writing an Architecture subclass — there is no generic fallback.
    """


class UnsupportedModelConfig(Exception):
    """Architecture exists but this config combination is unsupported.

    Raised from validate() at model load (e.g. RotatingKVCache.keep > 0).
    """


class MTPWeightsLoadError(Exception):
    """install() failed to load or attach MTP sidecar weights despite the
    sidecar file being present. The missing-sidecar case is not an error;
    it sets self.mtp_available = False in validate() and is observable
    through the load-time warning (D8).
    """


class Architecture(ABC):
    # Identity
    model_type: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()

    # Static capabilities (class-level facts about the architecture).
    has_mtp: ClassVar[bool] = False
    has_vision: ClassVar[bool] = False
    has_audio: ClassVar[bool] = False

    def __init__(self) -> None:
        # Per-instance runtime state set by validate()/install(). See D8.
        self.mtp_available: bool = self.has_mtp

    # ── Install lifecycle ────────────────────────────────────────────
    def validate(self, model, config, model_path: Path) -> None:
        """Raise UnsupportedModelConfig on known-unsupported configs.
        MTP architectures also clear self.mtp_available here if the sidecar
        weights file is absent (D8). Default: no-op.
        """
        return None

    def install(self, model, config, model_path: Path) -> None:
        """Patch this model so batched prefill/decode work. Idempotent.
        Default: no-op.
        """
        return None

    # ── Capability queries (D4 — all return Optional, never raise) ───
    def turn_end_token_ids(self, tokenizer) -> set[int] | None:
        return None

    def extract_attention_query(self, attn_layer, layer_idx: int):
        return None

    def build_mtp_module(self, model, config):
        return None

    def vision_processor(self, config):
        return None
```

- [ ] **Step 4: Implement registry in `__init__.py`**

`vllm_mlx/architectures/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Architecture registry. See docs/adr/ADR-0008."""

from .base import (
    Architecture,
    MTPWeightsLoadError,
    UnsupportedArchitectureError,
    UnsupportedModelConfig,
)

_REGISTRY: dict[str, type[Architecture]] = {}


def register(cls: type[Architecture]) -> type[Architecture]:
    """Class decorator: add cls to the registry under its model_type and aliases."""
    _REGISTRY[cls.model_type] = cls
    for alias in cls.aliases:
        _REGISTRY[alias] = cls
    return cls


def lookup(config) -> Architecture:
    """Return an Architecture instance for this config.

    Checks config.model_type then config.text_config.model_type.
    Raises UnsupportedArchitectureError on unknown model_type.
    """
    candidates = [
        getattr(config, "model_type", None),
        getattr(getattr(config, "text_config", None), "model_type", None),
    ]
    for mt in candidates:
        if mt is not None and mt in _REGISTRY:
            return _REGISTRY[mt]()
    raise UnsupportedArchitectureError(
        f"No Architecture for model_type in {candidates}. "
        f"Registered: {sorted(_REGISTRY)}"
    )


__all__ = [
    "Architecture",
    "MTPWeightsLoadError",
    "UnsupportedArchitectureError",
    "UnsupportedModelConfig",
    "lookup",
    "register",
]
```

- [ ] **Step 5: Add empty `tests/architectures/__init__.py`** so pytest treats it as a package.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/architectures/test_base.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/architectures/__init__.py vllm_mlx/architectures/base.py \
        tests/architectures/__init__.py tests/architectures/test_base.py
git commit -m "feat(architectures): add base protocol, exceptions, registry [ADR-0008, Task 1]"
```

---

## Task 2: Extract global runtime fixes (SDPA patches)

**Files:**
- Create: `vllm_mlx/global_runtime_fixes.py`
- Create: `tests/architectures/test_global_runtime_fixes.py`
- Modify: `vllm_mlx/scheduler.py:37-43` (remove eager calls, replaced by load-time bootstrap in Task 5; for this task we add a call to `install_once()` to preserve behavior)

**Interfaces:**
- Consumes: existing `patches.mlx_lm_quantized_sdpa.patch_quantized_sdpa` and `patches.mlx_lm_prefill_flash_sdpa.apply` (still imported from `patches/` for now; deleted in Task 19).
- Produces: `global_runtime_fixes.install_once() -> None`. Idempotent.

- [ ] **Step 1: Write failing test for idempotency**

`tests/architectures/test_global_runtime_fixes.py`:

```python
from vllm_mlx import global_runtime_fixes


def test_install_once_is_idempotent(caplog):
    global_runtime_fixes._INSTALLED = False  # reset module state
    global_runtime_fixes.install_once()
    assert global_runtime_fixes._INSTALLED is True

    # Calling again should be a no-op (no exception, no re-application).
    global_runtime_fixes.install_once()
    assert global_runtime_fixes._INSTALLED is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/architectures/test_global_runtime_fixes.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `global_runtime_fixes.py`**

`vllm_mlx/global_runtime_fixes.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic mlx-lm fixes applied once per process.

These patches affect every architecture identically — they are upstream
candidates and live here separately from per-architecture install logic.

Must run before any mlx-lm model class is touched at load time because
apply_prefill_flash_sdpa walks sys.modules. Both server bootstrap and
MLXLanguageModel.load() / MLXMultimodalLM.load() call install_once();
idempotency is the contract.

See docs/adr/ADR-0008.
"""

import logging

logger = logging.getLogger(__name__)

_INSTALLED: bool = False


def install_once() -> None:
    """Apply architecture-agnostic mlx-lm runtime fixes. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Imports are lazy so this module loads cheaply; the actual mlx-lm
    # import is deferred until install_once is first called.
    from .patches.mlx_lm_prefill_flash_sdpa import apply as _apply_prefill_flash_sdpa
    from .patches.mlx_lm_quantized_sdpa import patch_quantized_sdpa

    patch_quantized_sdpa()
    _apply_prefill_flash_sdpa()
    _INSTALLED = True
    logger.info("[global_runtime_fixes] installed")
```

- [ ] **Step 4: Update `scheduler.py` to call `install_once()` instead of the three eager calls**

Modify `vllm_mlx/scheduler.py:37-43`:

```python
# Before:
from .patches.mlx_lm_quantized_sdpa import patch_quantized_sdpa
from .patches.mlx_lm_prefill_flash_sdpa import apply as _apply_prefill_flash_sdpa
from .patches.gemma4_llm import patch_gemma4_attention_for_batching as _patch_gemma4_llm

patch_quantized_sdpa()
_apply_prefill_flash_sdpa()
_patch_gemma4_llm()

# After:
from .global_runtime_fixes import install_once as _install_global_runtime_fixes
from .patches.gemma4_llm import patch_gemma4_attention_for_batching as _patch_gemma4_llm

_install_global_runtime_fixes()
_patch_gemma4_llm()  # stays for now; moves into Gemma4Architecture.install() in Task 10
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/architectures/test_global_runtime_fixes.py tests/architectures/test_base.py -v`
Expected: 9 passed.

- [ ] **Step 6: Run full test suite to confirm no regression**

Run: `pytest tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q`
Expected: all green (Metal-only OOM test skipped).

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/global_runtime_fixes.py vllm_mlx/scheduler.py \
        tests/architectures/test_global_runtime_fixes.py
git commit -m "feat(architectures): extract global SDPA fixes into install_once [ADR-0008, Task 2]"
```

---

## Task 3: FakeArchitecture for engine tests

**Files:**
- Create: `vllm_mlx/architectures/_testing.py`
- Create: `tests/architectures/test_fake.py`

**Interfaces:**
- Consumes: `Architecture` base class.
- Produces: `FakeArchitecture` with constructor args `(model_type="fake", has_mtp=False, has_vision=False, has_audio=False, turn_end_ids=None, attention_query=None, mtp_module=None, processor=None, mtp_available=None)`. Each capability method returns the configured value; `validate`/`install` are no-ops.

- [ ] **Step 1: Write failing test**

`tests/architectures/test_fake.py`:

```python
from vllm_mlx.architectures._testing import FakeArchitecture


def test_fake_returns_configured_capabilities():
    a = FakeArchitecture(
        has_mtp=True,
        turn_end_ids={1, 2, 3},
    )
    assert a.has_mtp is True
    assert a.mtp_available is True
    assert a.turn_end_token_ids(tokenizer=None) == {1, 2, 3}
    assert a.extract_attention_query(attn_layer=None, layer_idx=0) is None


def test_fake_mtp_available_override():
    a = FakeArchitecture(has_mtp=True, mtp_available=False)
    assert a.has_mtp is True
    assert a.mtp_available is False


def test_fake_install_is_noop():
    a = FakeArchitecture()
    a.validate(model=None, config=None, model_path=None)
    a.install(model=None, config=None, model_path=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/architectures/test_fake.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `FakeArchitecture`**

`vllm_mlx/architectures/_testing.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Configurable Architecture stand-in for engine tests.

Avoid in production code paths. Engine tests use this to exercise
_compute_turn_boundaries, MTP gating, SpecPrefill routing, and the
MLLM vision-processor wiring without depending on the registered
architecture set or loading a real model.
"""

from typing import Any

from .base import Architecture


class FakeArchitecture(Architecture):
    model_type = "fake"

    def __init__(
        self,
        *,
        model_type: str = "fake",
        has_mtp: bool = False,
        has_vision: bool = False,
        has_audio: bool = False,
        turn_end_ids: set[int] | None = None,
        attention_query: Any = None,
        mtp_module: Any = None,
        processor: Any = None,
        mtp_available: bool | None = None,
    ) -> None:
        # Override per-instance copies of the classvars so each test can
        # configure capabilities without subclassing.
        self.model_type = model_type
        self.has_mtp = has_mtp
        self.has_vision = has_vision
        self.has_audio = has_audio
        self._turn_end_ids = turn_end_ids
        self._attention_query = attention_query
        self._mtp_module = mtp_module
        self._processor = processor
        self.mtp_available = has_mtp if mtp_available is None else mtp_available

    def turn_end_token_ids(self, tokenizer) -> set[int] | None:
        return self._turn_end_ids

    def extract_attention_query(self, attn_layer, layer_idx: int):
        return self._attention_query

    def build_mtp_module(self, model, config):
        return self._mtp_module

    def vision_processor(self, config):
        return self._processor
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/architectures/test_fake.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/architectures/_testing.py tests/architectures/test_fake.py
git commit -m "feat(architectures): add FakeArchitecture for engine tests [ADR-0008, Task 3]"
```

---

## Task 4: Skeleton concrete Architectures

**Files:**
- Create: `vllm_mlx/architectures/gemma4.py`
- Create: `vllm_mlx/architectures/qwen3.py`
- Create: `vllm_mlx/architectures/qwen3_5_vl.py`
- Create: `vllm_mlx/architectures/qwen3_next.py`
- Create: `vllm_mlx/architectures/glm4v_moe.py`
- Modify: `vllm_mlx/architectures/__init__.py` (import-side-effect registration)
- Create: `tests/architectures/test_skeletons.py`

**Interfaces:**
- Consumes: `Architecture`, `register`.
- Produces: Five registered subclasses with correct `model_type`, `has_mtp/has_vision/has_audio` classvars, but **no overridden methods** — they inherit no-op `validate`/`install` and `None`-returning capability queries. Real behavior lands in Tasks 10–15.

Registration values come from existing patches and HF configs. For Qwen 3.5 VL the `model_type` is `qwen3_5_moe`; for Qwen 3-Next it's `qwen3_next`; for Gemma 4 it's `gemma4_text`; for Qwen 3 it's `qwen3`; for GLM-4.6V it's `glm4v_moe`. These match the strings already used in `specprefill.py:_EXTRACTOR_REGISTRY` and `_try_inject_mtp`.

- [ ] **Step 1: Write failing test verifying registration of all five**

`tests/architectures/test_skeletons.py`:

```python
import vllm_mlx.architectures as arch  # triggers import-side-effect registration

EXPECTED = {
    "gemma4_text": ("Gemma4Architecture", False, False, False),
    "qwen3": ("Qwen3Architecture", False, False, False),
    "qwen3_5_moe": ("Qwen3_5VLArchitecture", True, True, False),
    "qwen3_next": ("Qwen3NextArchitecture", True, False, False),
    "glm4v_moe": ("GLM4VMoEArchitecture", False, True, False),
}


def test_all_skeletons_registered():
    for model_type, (cls_name, has_mtp, has_vision, has_audio) in EXPECTED.items():
        cls = arch._REGISTRY[model_type]
        assert cls.__name__ == cls_name
        assert cls.has_mtp is has_mtp, model_type
        assert cls.has_vision is has_vision, model_type
        assert cls.has_audio is has_audio, model_type
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/architectures/test_skeletons.py -v`
Expected: KeyError on first lookup.

- [ ] **Step 3: Implement skeleton `gemma4.py`**

`vllm_mlx/architectures/gemma4.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from . import register
from .base import Architecture


@register
class Gemma4Architecture(Architecture):
    model_type = "gemma4_text"
```

- [ ] **Step 4: Implement skeleton `qwen3.py`**

```python
# SPDX-License-Identifier: Apache-2.0
from . import register
from .base import Architecture


@register
class Qwen3Architecture(Architecture):
    model_type = "qwen3"
```

- [ ] **Step 5: Implement skeleton `qwen3_5_vl.py`**

```python
# SPDX-License-Identifier: Apache-2.0
from . import register
from .base import Architecture


@register
class Qwen3_5VLArchitecture(Architecture):
    model_type = "qwen3_5_moe"
    aliases = ("qwen3_5",)  # alternate model_type seen in some configs
    has_mtp = True
    has_vision = True
```

- [ ] **Step 6: Implement skeleton `qwen3_next.py`**

```python
# SPDX-License-Identifier: Apache-2.0
from . import register
from .base import Architecture


@register
class Qwen3NextArchitecture(Architecture):
    model_type = "qwen3_next"
    has_mtp = True
```

- [ ] **Step 7: Implement skeleton `glm4v_moe.py`**

```python
# SPDX-License-Identifier: Apache-2.0
from . import register
from .base import Architecture


@register
class GLM4VMoEArchitecture(Architecture):
    model_type = "glm4v_moe"
    has_vision = True
```

- [ ] **Step 8: Wire registration via package import**

Modify `vllm_mlx/architectures/__init__.py` — append after the existing exports:

```python
# Register concrete architectures via import side-effect.
# Must come AFTER `register` is defined.
from . import gemma4, glm4v_moe, qwen3, qwen3_5_vl, qwen3_next  # noqa: E402, F401
```

- [ ] **Step 9: Run tests**

Run: `pytest tests/architectures/ -v`
Expected: all green (Task 1 + 2 + 3 + 4 tests).

- [ ] **Step 10: Commit**

```bash
git add vllm_mlx/architectures/gemma4.py vllm_mlx/architectures/qwen3.py \
        vllm_mlx/architectures/qwen3_5_vl.py vllm_mlx/architectures/qwen3_next.py \
        vllm_mlx/architectures/glm4v_moe.py vllm_mlx/architectures/__init__.py \
        tests/architectures/test_skeletons.py
git commit -m "feat(architectures): add skeleton subclasses with capability classvars [ADR-0008, Task 4]"
```

---

## Task 5: Wire `MLXLanguageModel.load()` to architecture lifecycle

**Files:**
- Modify: `vllm_mlx/models/llm.py:81-115`
- Modify: `tests/test_llm.py` if it exists; otherwise create `tests/test_llm_load_lifecycle.py`

**Interfaces:**
- Consumes: `global_runtime_fixes.install_once`, `architectures.lookup`, `Architecture`.
- Produces: `MLXLanguageModel.architecture: Architecture` attribute populated after `load()`. `load()` calls `install_once()` → `lookup()` → `validate()` → `install()` in that order. Raises `UnsupportedArchitectureError` on unregistered model_type; propagates `UnsupportedModelConfig` / `MTPWeightsLoadError`.

- [ ] **Step 1: Write failing test using a stub model loader**

`tests/test_llm_load_lifecycle.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.architectures import (
    Architecture,
    UnsupportedArchitectureError,
    _REGISTRY,
    register,
)
from vllm_mlx.models.llm import MLXLanguageModel


@pytest.fixture(autouse=True)
def _isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _stub_loader(model_type: str):
    cfg = SimpleNamespace(model_type=model_type)
    model = SimpleNamespace(config=cfg)
    tokenizer = MagicMock()
    return model, tokenizer


def test_load_populates_architecture_attribute():
    calls = []

    @register
    class Stub(Architecture):
        model_type = "stub"
        def validate(self, model, config, model_path):
            calls.append(("validate", model, config, model_path))
        def install(self, model, config, model_path):
            calls.append(("install", model, config, model_path))

    with patch(
        "vllm_mlx.utils.tokenizer.load_model_with_fallback",
        return_value=_stub_loader("stub"),
    ):
        m = MLXLanguageModel(model_name="dummy/stub")
        m.load()

    assert isinstance(m.architecture, Stub)
    assert [c[0] for c in calls] == ["validate", "install"]


def test_load_raises_on_unregistered_model_type():
    with patch(
        "vllm_mlx.utils.tokenizer.load_model_with_fallback",
        return_value=_stub_loader("unregistered_xyz"),
    ):
        m = MLXLanguageModel(model_name="dummy/unknown")
        with pytest.raises(UnsupportedArchitectureError):
            m.load()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_load_lifecycle.py -v`
Expected: `AttributeError: 'MLXLanguageModel' object has no attribute 'architecture'`.

- [ ] **Step 3: Modify `MLXLanguageModel.__init__` to declare `self.architecture`**

In `vllm_mlx/models/llm.py`, find `__init__` and add after `self._loaded = False`:

```python
self.architecture = None  # set by load()
```

- [ ] **Step 4: Modify `MLXLanguageModel.load()` to run the architecture lifecycle**

Replace the body of `load()` in `vllm_mlx/models/llm.py:81-115`:

```python
def load(self) -> None:
    """Load the model and tokenizer, then run the architecture lifecycle."""
    if self._loaded:
        return

    try:
        from pathlib import Path

        from mlx_lm.utils import _download

        from .. import architectures, global_runtime_fixes
        from ..utils.tokenizer import load_model_with_fallback

        # D3: ensure global mlx-lm fixes are installed before model load.
        global_runtime_fixes.install_once()

        logger.info(f"Loading model: {self.model_name}")

        tokenizer_config = {"trust_remote_code": self.trust_remote_code}
        if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
            tokenizer_config["eos_token"] = "<|im_end|>"
            logger.info("Qwen3 detected: setting eos_token to <|im_end|>")

        self.model, self.tokenizer = load_model_with_fallback(
            self.model_name,
            tokenizer_config=tokenizer_config,
        )

        # Run architecture lifecycle: lookup → validate → install.
        model_path = Path(_download(self.model_name))
        self.architecture = architectures.lookup(self.model.config)
        self.architecture.validate(self.model, self.model.config, model_path)
        self.architecture.install(self.model, self.model.config, model_path)

        self._loaded = True
        logger.info(
            f"Model loaded successfully: {self.model_name} "
            f"(architecture={self.architecture.model_type})"
        )

    except ImportError as err:
        raise ImportError(
            "mlx-lm is required for LLM inference. Install with: pip install mlx-lm"
        ) from err
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_llm_load_lifecycle.py tests/architectures/ -v`
Expected: all green.

- [ ] **Step 6: Smoke test loading a real Qwen3 model if a checkpoint is available locally**

Run (only if a Qwen3 checkpoint is in MLX cache):

```bash
python -c "
from vllm_mlx.models.llm import MLXLanguageModel
m = MLXLanguageModel(model_name='mlx-community/Qwen3-0.6B-bf16')
m.load()
print('architecture:', m.architecture.model_type)
"
```

Expected output: `architecture: qwen3`.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/models/llm.py tests/test_llm_load_lifecycle.py
git commit -m "feat(models): wire MLXLanguageModel.load to architecture lifecycle [ADR-0008, Task 5]"
```

---

## Task 6: Wire `MLXMultimodalLM.load()` to architecture lifecycle

**Files:**
- Modify: `vllm_mlx/models/mllm.py:886-920`
- Create: `tests/test_mllm_load_lifecycle.py`

**Interfaces:** Same shape as Task 5 but using `mlx_vlm.load()` instead of `load_model_with_fallback`. `self.architecture` populated; raises on unknown model_type.

- [ ] **Step 1: Write failing test mirroring Task 5**

`tests/test_mllm_load_lifecycle.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.architectures import (
    Architecture,
    UnsupportedArchitectureError,
    _REGISTRY,
    register,
)
from vllm_mlx.models.mllm import MLXMultimodalLM


@pytest.fixture(autouse=True)
def _isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _stub_mlx_vlm_load(model_type):
    cfg = SimpleNamespace(model_type=model_type)
    inner = SimpleNamespace(config=cfg, language_model=MagicMock())
    inner.config = cfg
    processor = MagicMock(tokenizer=MagicMock())
    return inner, processor


def test_load_populates_architecture_attribute():
    @register
    class StubMLLM(Architecture):
        model_type = "stub_mllm"
        has_vision = True

    with patch("mlx_vlm.load", return_value=_stub_mlx_vlm_load("stub_mllm")), \
         patch("mlx_vlm.utils.load_config", return_value={"model_type": "stub_mllm"}):
        m = MLXMultimodalLM(model_name="dummy/stub_mllm")
        m.load()

    assert isinstance(m.architecture, StubMLLM)


def test_load_raises_on_unregistered_model_type():
    with patch("mlx_vlm.load", return_value=_stub_mlx_vlm_load("unknown_mllm")), \
         patch("mlx_vlm.utils.load_config", return_value={"model_type": "unknown_mllm"}):
        m = MLXMultimodalLM(model_name="dummy/unknown_mllm")
        with pytest.raises(UnsupportedArchitectureError):
            m.load()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mllm_load_lifecycle.py -v`
Expected: AttributeError on `m.architecture`.

- [ ] **Step 3: Modify `MLXMultimodalLM.__init__` to add `self.architecture = None`**

In `vllm_mlx/models/mllm.py`, find the `__init__` ending around line 884 and append `self.architecture = None`.

- [ ] **Step 4: Modify `MLXMultimodalLM.load()` to run the architecture lifecycle**

Replace the `try` block in `load()` at `vllm_mlx/models/mllm.py:886-920`:

```python
def load(self) -> None:
    """Load the model and processor, then run the architecture lifecycle."""
    if self._loaded:
        return

    try:
        from pathlib import Path

        from mlx_lm.utils import _download
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        from .. import architectures, global_runtime_fixes

        # D3: ensure global mlx-lm fixes are installed before model load.
        global_runtime_fixes.install_once()

        logger.info(f"Loading MLLM: {self.model_name}")

        self.model, self.processor = load(self.model_name)
        self.config = load_config(self.model_name)

        # Run architecture lifecycle: lookup → validate → install.
        model_path = Path(_download(self.model_name))
        self.architecture = architectures.lookup(self.model.config)
        self.architecture.validate(self.model, self.model.config, model_path)
        self.architecture.install(self.model, self.model.config, model_path)

        self._loaded = True
        self._video_native = hasattr(
            self.model.config, "video_token_id"
        ) or hasattr(self.model.config, "video_token_index")
        logger.info(
            f"MLLM loaded successfully: {self.model_name} "
            f"(architecture={self.architecture.model_type})"
        )
        if self._video_native:
            logger.info("Native video pipeline enabled (temporal 3D conv + M-RoPE)")

    except ImportError:
        raise ImportError(
            "mlx-vlm is required for multimodal inference. "
            "Install with: pip install mlx-vlm"
        )
    except Exception as e:
        logger.error(f"Failed to load MLLM: {e}")
        raise
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_mllm_load_lifecycle.py tests/architectures/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/models/mllm.py tests/test_mllm_load_lifecycle.py
git commit -m "feat(models): wire MLXMultimodalLM.load to architecture lifecycle [ADR-0008, Task 6]"
```

---

## Task 7: Collapse `_compute_turn_boundaries` to use `architecture.turn_end_token_ids`

**Files:**
- Modify: `vllm_mlx/engine/batched.py:988-1145` (the function body)
- Create: `tests/engine/test_compute_turn_boundaries.py`

**Interfaces:**
- Consumes: `self.model_wrapper.architecture.turn_end_token_ids(tokenizer) -> set[int] | None`.
- Produces: behavior matches today for any architecture that returns the right end-token set; logs a single `WARNING` and returns `[]` when the architecture returns `None`.

- [ ] **Step 1: Write tests using `FakeArchitecture`**

`tests/engine/test_compute_turn_boundaries.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_mlx.architectures._testing import FakeArchitecture


class _StubEngine:
    """Minimal stand-in for the BatchedEngine slice we test."""

    def __init__(self, architecture, tokenizer, rendered, tokens):
        self.model_wrapper = SimpleNamespace(architecture=architecture)
        self.tokenizer = tokenizer
        self._rendered = rendered
        self._tokens = tokens

    def _apply_chat_template(self, *args, **kwargs):
        return self._rendered


def _make_tokenizer(tokens):
    tk = MagicMock()
    tk.encode.return_value = tokens
    return tk


def test_returns_empty_when_no_system_message():
    from vllm_mlx.engine.batched import BatchedEngine

    eng = _StubEngine(
        architecture=FakeArchitecture(turn_end_ids={5}),
        tokenizer=_make_tokenizer([1, 2, 3]),
        rendered="hi",
        tokens=[1, 2, 3],
    )
    result = BatchedEngine._compute_turn_boundaries(
        eng, messages=[{"role": "user", "content": "hi"}]
    )
    assert result == []


def test_returns_empty_and_warns_when_arch_has_no_delimiters(caplog):
    from vllm_mlx.engine.batched import BatchedEngine

    eng = _StubEngine(
        architecture=FakeArchitecture(turn_end_ids=None, model_type="silent"),
        tokenizer=_make_tokenizer([1, 2, 3]),
        rendered="x",
        tokens=[1, 2, 3],
    )
    with caplog.at_level("WARNING"):
        result = BatchedEngine._compute_turn_boundaries(
            eng, messages=[{"role": "system", "content": "sys"}]
        )
    assert result == []
    assert any("silent" in r.message and "no turn delimiters" in r.message
               for r in caplog.records)


def test_returns_boundaries_after_each_end_token():
    from vllm_mlx.engine.batched import BatchedEngine

    eng = _StubEngine(
        architecture=FakeArchitecture(turn_end_ids={9}),
        tokenizer=_make_tokenizer([1, 2, 9, 3, 4, 9, 5]),
        rendered="...",
        tokens=[1, 2, 9, 3, 4, 9, 5],
    )
    result = BatchedEngine._compute_turn_boundaries(
        eng, messages=[{"role": "system", "content": "sys"}]
    )
    assert result == [3, 6]  # one past each end-token index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_compute_turn_boundaries.py -v`
Expected: tests fail (current implementation has different shape).

- [ ] **Step 3: Replace `_compute_turn_boundaries`**

Replace `vllm_mlx/engine/batched.py:988-1145` with the honest collapsed version from the spec (D7). Keep the existing signature (`tools`, `num_images`, `num_audios`, `chat_template_kwargs`, `enable_thinking`):

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
    """Compute token boundaries via architecture-supplied turn-end tokens.

    Each segment ends with the turn-end token (no trailing \n); the \n
    becomes the first token of the next segment. Returns [] when:
      - the first message is not the system message (turn cache invariant), or
      - the architecture exposes no turn delimiters (logs a WARNING).

    See docs/adr/ADR-0008 (D7) and the spec for the rationale.
    """
    # Invariant: turn cache assumes the system prompt is the stable prefix.
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
        chat_template_kwargs={
            **(chat_template_kwargs or {}),
            "add_generation_prompt": False,
        },
        enable_thinking=enable_thinking,
    )
    tokens = self.tokenizer.encode(rendered)
    return [i + 1 for i, tok in enumerate(tokens) if tok in end_ids]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/engine/test_compute_turn_boundaries.py tests/architectures/ -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/engine/batched.py tests/engine/test_compute_turn_boundaries.py
git commit -m "refactor(engine): collapse _compute_turn_boundaries through architecture [ADR-0008, Task 7]"
```

---

## Task 8: Collapse SpecPrefill extractor switch

**Files:**
- Modify: `vllm_mlx/specprefill.py:328-344`
- Create: `tests/test_specprefill_architecture.py`

**Interfaces:**
- Consumes: `architecture.extract_attention_query(attn_layer, layer_idx) -> mx.array | None`.
- Produces: `query_extractor` resolution path checks the architecture first; falls back to today's RoPE/non-RoPE auto-detection ONLY if `extract_attention_query` is not callable (e.g. older tests using a model without an architecture). Architecture-returning-`None` means "skip this request, log warning."

Note: the existing call site does `_EXTRACTOR_REGISTRY.get(model_type)` to pick a per-architecture extractor *function*, then later calls it for each layer. The new shape pushes the per-layer call into the architecture. This changes a small contract — instead of a single function reference cached up front, each layer-extraction call goes through the architecture. Architecture authors implementing `extract_attention_query` must be aware they're called per layer.

- [ ] **Step 1: Write test**

`tests/test_specprefill_architecture.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from vllm_mlx.architectures._testing import FakeArchitecture


def test_architecture_extractor_used_when_available():
    fake_q = mx.array([[1.0, 2.0]])
    arch = FakeArchitecture(attention_query=fake_q)
    layer = MagicMock()

    result = arch.extract_attention_query(layer, layer_idx=0)
    assert result is fake_q


def test_specprefill_skips_request_when_extractor_returns_none(caplog):
    # End-to-end at this level is heavy; we just unit-test the gate logic.
    from vllm_mlx.specprefill import _resolve_query_extractor

    arch = FakeArchitecture(attention_query=None, model_type="unsupported")
    with caplog.at_level("WARNING"):
        result = _resolve_query_extractor(arch)
    assert result is None
    assert any("unsupported" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_specprefill_architecture.py -v`
Expected: ImportError for `_resolve_query_extractor`.

- [ ] **Step 3: Refactor `specprefill.py`**

Add a new module-private helper above the current auto-detect block, and replace the `_EXTRACTOR_REGISTRY` block:

```python
def _resolve_query_extractor(architecture):
    """Return a per-layer extractor callable bound to the architecture,
    or None if the architecture doesn't support SpecPrefill.
    """
    # Probe with layer=None, layer_idx=0 only to check support — architectures
    # whose extract_attention_query returns None for any (None, 0) input
    # signal "no SpecPrefill." Architectures implementing the method must
    # tolerate this probe (return None) gracefully.
    if architecture.extract_attention_query(None, 0) is None:
        logger.warning(
            f"[specprefill] disabled for {architecture.model_type}: "
            f"extract_attention_query unavailable"
        )
        return None
    return lambda attn_layer, layer_idx: architecture.extract_attention_query(
        attn_layer, layer_idx
    )
```

Then at the existing extractor-resolution site (`specprefill.py:328-344`):

```python
# Replace the entire _EXTRACTOR_REGISTRY block with:
if query_extractor is None:
    architecture = getattr(model, "_architecture", None)
    if architecture is not None:
        query_extractor = _resolve_query_extractor(architecture)
    if query_extractor is None:
        if _get_rope(attn_obj) is not None:
            query_extractor = _llama_extract_queries
        else:
            query_extractor = _nemotron_h_extract_queries
```

Note: `_architecture` on the raw model is not yet set anywhere; we'll wire that on the caller side. For now, the fallback to `_llama_extract_queries` preserves today's behavior for any code path that hasn't been migrated. Each architecture that lands in Tasks 11–14 sets `model._architecture = self` in its `install()` method so SpecPrefill can find it.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_specprefill_architecture.py tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/specprefill.py tests/test_specprefill_architecture.py
git commit -m "refactor(specprefill): route extractor lookup through architecture [ADR-0008, Task 8]"
```

---

## Task 9: Collapse `is_mllm_model` to use `architecture.has_vision/has_audio`

**Files:**
- Modify: `vllm_mlx/api/utils.py:364-381`
- Modify: `vllm_mlx/model_registry.py:783-795` (call sites passing `model_wrapper`)
- Create: `tests/api/test_is_mllm_model.py`

**Interfaces:**
- Consumes: `model_wrapper.architecture.has_vision`, `architecture.has_audio`.
- Produces: `is_mllm_model(model_wrapper) -> bool`. The string-pattern path is kept under a different name (`is_mllm_model_by_name(model_name: str) -> bool`) for pre-load routing decisions that don't yet have a model_wrapper.

- [ ] **Step 1: Inspect callers**

```bash
grep -rn "is_mllm_model\|is_vlm_model" vllm_mlx tests | head -30
```

Note which callers pass a string vs. a model wrapper.

- [ ] **Step 2: Write tests**

`tests/api/test_is_mllm_model.py`:

```python
from types import SimpleNamespace

from vllm_mlx.api.utils import is_mllm_model, is_mllm_model_by_name
from vllm_mlx.architectures._testing import FakeArchitecture


def test_is_mllm_model_true_for_vision_arch():
    mw = SimpleNamespace(architecture=FakeArchitecture(has_vision=True))
    assert is_mllm_model(mw) is True


def test_is_mllm_model_true_for_audio_arch():
    mw = SimpleNamespace(architecture=FakeArchitecture(has_audio=True))
    assert is_mllm_model(mw) is True


def test_is_mllm_model_false_for_text_only_arch():
    mw = SimpleNamespace(architecture=FakeArchitecture())
    assert is_mllm_model(mw) is False


def test_is_mllm_model_by_name_keeps_string_path():
    assert is_mllm_model_by_name("Qwen/Qwen3.5-VL-7B") is True
    assert is_mllm_model_by_name("Qwen/Qwen3-0.6B") is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/api/test_is_mllm_model.py -v`
Expected: `TypeError` or `AttributeError` on the SimpleNamespace path.

- [ ] **Step 4: Refactor `api/utils.py`**

Replace `is_mllm_model` in `vllm_mlx/api/utils.py`:

```python
def is_mllm_model_by_name(model_name: str) -> bool:
    """Pre-load routing: detect MLLM/VLM by HF model name string.

    Used only when a model_wrapper is not yet available (e.g. router
    dispatching before load()). Once loaded, prefer is_mllm_model.
    """
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


def is_mllm_model(model_wrapper) -> bool:
    """Detect MLLM/VLM via the loaded architecture's capability bits."""
    architecture = getattr(model_wrapper, "architecture", None)
    if architecture is None:
        # Pre-load fallback: route by name if the wrapper hasn't loaded yet.
        return is_mllm_model_by_name(getattr(model_wrapper, "model_name", ""))
    return architecture.has_vision or architecture.has_audio


# Backwards compatibility alias
is_vlm_model = is_mllm_model
```

- [ ] **Step 5: Update callers passing strings**

Find each caller passing a raw model name and migrate to `is_mllm_model_by_name`. Audit:

```bash
grep -rn "is_mllm_model(\|is_vlm_model(" vllm_mlx tests
```

Replace calls that pass `model_name: str` directly with `is_mllm_model_by_name`. Callers passing wrappers stay on `is_mllm_model`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/api/test_is_mllm_model.py tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/api/utils.py vllm_mlx/model_registry.py tests/api/test_is_mllm_model.py
git commit -m "refactor(api): route is_mllm_model through architecture capability bits [ADR-0008, Task 9]"
```

---

## Task 10: Migrate Gemma 4 — turn_end tokens + attention offset patch

**Files:**
- Modify: `vllm_mlx/architectures/gemma4.py` (fill in skeleton)
- Modify: `vllm_mlx/scheduler.py` (remove `_patch_gemma4_llm` eager call)
- Create: `tests/architectures/test_gemma4.py`
- Modify: `tests/architectures/test_skeletons.py` (remove gemma4 from the registration-only list; it has real behavior now)

**Interfaces:**
- Consumes: existing `patches.gemma4_llm._snapshot_cache_offset` logic (inlined into the architecture).
- Produces: `Gemma4Architecture.install()` patches `mlx_lm.models.gemma4_text.Attention.__call__` to snapshot `cache.offset` before in-place mutation. `Gemma4Architecture.turn_end_token_ids(tokenizer)` returns the set of `<turn|>` and `<tool_response|>` ids when both are present. Sets `model._architecture = self` for SpecPrefill.

- [ ] **Step 1: Write tests**

`tests/architectures/test_gemma4.py`:

```python
from unittest.mock import MagicMock

from vllm_mlx.architectures.gemma4 import Gemma4Architecture


def test_turn_end_token_ids_returns_set_when_both_present():
    tk = MagicMock()
    # Map <turn|> → 100, <tool_response|> → 101, unk → 0
    tk.convert_tokens_to_ids.side_effect = lambda s: {
        "<turn|>": 100, "<tool_response|>": 101
    }.get(s, 0)
    tk.unk_token_id = 0

    arch = Gemma4Architecture()
    assert arch.turn_end_token_ids(tk) == {100, 101}


def test_turn_end_token_ids_returns_none_when_neither_present():
    tk = MagicMock()
    tk.convert_tokens_to_ids.return_value = 0
    tk.unk_token_id = 0

    arch = Gemma4Architecture()
    assert arch.turn_end_token_ids(tk) is None


def test_install_sets_model_architecture_attribute():
    arch = Gemma4Architecture()
    fake_model = MagicMock()
    # install() patches mlx_lm.models.gemma4_text.Attention globally; we
    # only assert that the model gets the back-reference for SpecPrefill.
    arch.install(fake_model, config=MagicMock(), model_path=None)
    assert fake_model._architecture is arch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/architectures/test_gemma4.py -v`
Expected: AttributeError (`turn_end_token_ids` returns None — skeleton).

- [ ] **Step 3: Implement `gemma4.py`**

`vllm_mlx/architectures/gemma4.py` — replace skeleton with:

```python
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 architecture.

Handles the offset-snapshot fix for BatchKVCache (cache.offset is an mx.array
that update_and_fetch mutates in place; query RoPE sees the post-update
offset unless we materialize a snapshot first).
"""

import logging
from typing import Any, Optional

import mlx.core as mx

from . import register
from .base import Architecture

logger = logging.getLogger(__name__)

_PATCHED = False  # class-level idempotency guard for the mlx-lm class patch


def _snapshot_cache_offset(cache):
    """Return a defensive copy of cache.offset safe from in-place mutation."""
    if cache is None:
        return 0
    off = cache.offset
    if isinstance(off, int):
        return off
    if isinstance(off, mx.array):
        return off + 0  # new array, same values
    return off


def _patch_gemma4_attention() -> bool:
    """Class-level patch of mlx_lm Gemma4 Attention.__call__. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from mlx_lm.models.gemma4_text import Attention as Gemma4Attention
        from mlx_lm.models.base import scaled_dot_product_attention
    except (ImportError, TypeError):
        logger.debug("[Gemma4] mlx_lm gemma4_text module not available")
        return False

    original_call = Gemma4Attention.__call__

    def patched_call(self, x: mx.array, mask=None, cache=None) -> mx.array:
        # Snapshot the offset BEFORE update_and_fetch mutates it.
        snapshot = _snapshot_cache_offset(cache)
        # Delegate to the original __call__ but with a wrapped cache that
        # exposes the snapshotted offset until after update_and_fetch.
        # The original implementation reads cache.offset for RoPE; we
        # substitute the snapshot via a thin shim.
        return original_call(self, x, mask=mask, cache=_OffsetSnapshotCache(cache, snapshot)
                             if cache is not None else None)

    Gemma4Attention.__call__ = patched_call
    _PATCHED = True
    logger.info("[Gemma4] Attention.__call__ patched with offset snapshot")
    return True


class _OffsetSnapshotCache:
    """Wrapper that returns the snapshotted offset for the duration of one
    Attention.__call__, delegating all other attribute access to the real cache.
    """
    __slots__ = ("_cache", "_snapshot")

    def __init__(self, cache, snapshot):
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_snapshot", snapshot)

    def __getattr__(self, name):
        if name == "offset":
            return self._snapshot
        return getattr(self._cache, name)

    def __setattr__(self, name, value):
        setattr(self._cache, name, value)


@register
class Gemma4Architecture(Architecture):
    model_type = "gemma4_text"

    def install(self, model, config: Any, model_path) -> None:
        _patch_gemma4_attention()
        # Back-reference for SpecPrefill resolution (Task 8).
        model._architecture = self

    def turn_end_token_ids(self, tokenizer) -> Optional[set[int]]:
        unk = getattr(tokenizer, "unk_token_id", None)
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        ids: set[int] = set()
        for tok in ("<turn|>", "<tool_response|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != unk:
                ids.add(tid)
        return ids or None
```

> Note: the `_OffsetSnapshotCache` wrapper is *new* — today's patch reaches into the original `Attention.__call__` body and substitutes the snapshot inline. Read `patches/gemma4_llm.py:patch_gemma4_attention_for_batching` in full before writing the replacement to make sure the substitution shape matches mlx-lm's `Attention.__call__` signature. If the existing patch is a copy-paste of the entire `__call__` body with one line changed, prefer that shape — copy the existing patch body verbatim into `patched_call`, change only the `_snapshot_cache_offset(cache)` line, and drop `_OffsetSnapshotCache`. The wrapper above is a fallback design.

- [ ] **Step 4: Remove eager call from `scheduler.py`**

In `vllm_mlx/scheduler.py`, delete:

```python
from .patches.gemma4_llm import patch_gemma4_attention_for_batching as _patch_gemma4_llm
...
_patch_gemma4_llm()
```

- [ ] **Step 5: Update `test_skeletons.py`** — remove `gemma4_text` from the skeleton list since it now has real behavior. The `EXPECTED` dict keeps `gemma4_text` for the classvar assertions (those are still about identity, not behavior) but Task 10's tests cover the behavior.

- [ ] **Step 6: Run tests**

Run: `pytest tests/architectures/ -v`
Expected: all green.

- [ ] **Step 7: Smoke-test a Gemma 4 model load + batched prefill if a checkpoint is available**

```bash
python -c "
from vllm_mlx.models.llm import MLXLanguageModel
m = MLXLanguageModel(model_name='mlx-community/gemma-3-4b-it-bf16')
m.load()
print('arch:', m.architecture.model_type)
print('attention patched:', hasattr(m.model.layers[0].self_attn.__call__, '__name__'))
"
```

Expected: `arch: gemma4_text` and the attention patch installed.

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/architectures/gemma4.py vllm_mlx/scheduler.py \
        tests/architectures/test_gemma4.py tests/architectures/test_skeletons.py
git commit -m "feat(architectures): migrate Gemma 4 attention patch + turn delimiters [ADR-0008, Task 10]"
```

---

## Task 11: Migrate Qwen 3 — turn_end tokens + attention query extractor

**Files:**
- Modify: `vllm_mlx/architectures/qwen3.py`
- Create: `tests/architectures/test_qwen3.py`

**Interfaces:**
- Consumes: today's `_llama_extract_queries` (in `specprefill.py`) — copy its implementation into the architecture.
- Produces: `Qwen3Architecture.turn_end_token_ids` returns `{<|im_end|>}`. `extract_attention_query(layer, idx)` returns the post-RoPE Q for SpecPrefill.

- [ ] **Step 1: Write test**

`tests/architectures/test_qwen3.py`:

```python
from unittest.mock import MagicMock

from vllm_mlx.architectures.qwen3 import Qwen3Architecture


def test_turn_end_token_ids_returns_im_end():
    tk = MagicMock()
    tk.convert_tokens_to_ids.side_effect = lambda s: {"<|im_end|>": 151645}.get(s, 0)
    tk.unk_token_id = 0

    arch = Qwen3Architecture()
    assert arch.turn_end_token_ids(tk) == {151645}


def test_turn_end_token_ids_returns_none_when_absent():
    tk = MagicMock()
    tk.convert_tokens_to_ids.return_value = 0
    tk.unk_token_id = 0

    arch = Qwen3Architecture()
    assert arch.turn_end_token_ids(tk) is None


def test_extract_attention_query_returns_none_for_probe():
    # The (None, 0) probe used by specprefill._resolve_query_extractor
    # must NOT raise — the architecture must tolerate it and return None
    # when it can't compute (Task 8 contract).
    arch = Qwen3Architecture()
    assert arch.extract_attention_query(None, 0) is None
```

- [ ] **Step 2: Run test**

Run: `pytest tests/architectures/test_qwen3.py -v`
Expected: turn-end tests fail; probe test passes (skeleton already returns None).

- [ ] **Step 3: Implement `qwen3.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Qwen 3 (text) architecture."""

from typing import Optional

import mlx.core as mx

from . import register
from .base import Architecture


@register
class Qwen3Architecture(Architecture):
    model_type = "qwen3"

    def install(self, model, config, model_path) -> None:
        model._architecture = self

    def turn_end_token_ids(self, tokenizer) -> Optional[set[int]]:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        unk = getattr(tokenizer, "unk_token_id", None)
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is None or tid == unk:
            return None
        return {tid}

    def extract_attention_query(self, attn_layer, layer_idx: int):
        if attn_layer is None:
            return None  # tolerate the probe from _resolve_query_extractor
        # Inline the body of specprefill._llama_extract_queries here.
        # This is identical to today's Llama-family extractor: read q_proj
        # output, reshape to [B, num_heads, T, head_dim], apply RoPE if
        # present, return the post-RoPE Q.
        # PASTE THE CURRENT _llama_extract_queries IMPLEMENTATION HERE.
        raise NotImplementedError(
            "Copy specprefill._llama_extract_queries verbatim into this method "
            "during implementation. It already has the right shape."
        )
```

Replace the `NotImplementedError` block by copying the current `_llama_extract_queries` body verbatim from `vllm_mlx/specprefill.py`. Adjust the function signature to match `(self, attn_layer, layer_idx)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/architectures/test_qwen3.py -v`
Expected: turn-end tests pass; extractor test passes (returns None for probe).

- [ ] **Step 5: Smoke test SpecPrefill end-to-end on a Qwen 3 model**

```bash
python -c "
from vllm_mlx.models.llm import MLXLanguageModel
m = MLXLanguageModel(model_name='mlx-community/Qwen3-0.6B-bf16')
m.load()
print('extractor available:', m.architecture.extract_attention_query(
    m.model.layers[0].self_attn, 0) is not None)
"
```

Expected: `extractor available: True`.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/architectures/qwen3.py tests/architectures/test_qwen3.py
git commit -m "feat(architectures): migrate Qwen 3 turn delimiters + attention extractor [ADR-0008, Task 11]"
```

---

## Task 12: Extract shared MTP scaffolding to `_mtp_common.py`

**Files:**
- Create: `vllm_mlx/architectures/_mtp_common.py`
- Create: `tests/architectures/test_mtp_common.py`

**Interfaces:**
- Consumes: today's `patches/qwen3_5_mtp.py:inject_mtp_support` and `patches/qwen3_next_mtp.py:inject_mtp_support`.
- Produces:
  - `find_mtp_sidecar(model_path: Path) -> Path | None` — search canonical locations.
  - `load_mtp_weights(sidecar_path: Path) -> dict[str, mx.array]` — load + dequantize + apply RMSNorm offset fixups.
  - `BuildResult` dataclass with `module: nn.Module, weights_loaded: int`.
  - `build_qwen35_mtp_module(model, config, sidecar_path) -> BuildResult` — extracted from `qwen3_5_mtp.inject_mtp_support`.
  - `build_qwen3_next_mtp_module(model, config, sidecar_path) -> BuildResult` — extracted from `qwen3_next_mtp.inject_mtp_support`.

Read both existing patches end-to-end before extracting; identify the 90% shared scaffolding (sidecar discovery, weight loading, RMSNorm offset fixup, module attachment) and isolate per-architecture differences (module class construction, weight key remapping).

- [ ] **Step 1: Audit the two patches**

```bash
diff -u vllm_mlx/patches/qwen3_5_mtp.py vllm_mlx/patches/qwen3_next_mtp.py | wc -l
```

- [ ] **Step 2: Write test for `find_mtp_sidecar`**

`tests/architectures/test_mtp_common.py`:

```python
from pathlib import Path

from vllm_mlx.architectures._mtp_common import find_mtp_sidecar


def test_find_mtp_sidecar_prefers_mtp_subdir(tmp_path):
    (tmp_path / "mtp").mkdir()
    weights = tmp_path / "mtp" / "weights.safetensors"
    weights.touch()
    (tmp_path / "model-mtp.safetensors").touch()  # fallback also present

    assert find_mtp_sidecar(tmp_path) == weights


def test_find_mtp_sidecar_falls_back_to_root_file(tmp_path):
    fallback = tmp_path / "model-mtp.safetensors"
    fallback.touch()

    assert find_mtp_sidecar(tmp_path) == fallback


def test_find_mtp_sidecar_returns_none_when_absent(tmp_path):
    assert find_mtp_sidecar(tmp_path) is None
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/architectures/test_mtp_common.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `_mtp_common.py`**

Create `vllm_mlx/architectures/_mtp_common.py`. The skeleton:

```python
# SPDX-License-Identifier: Apache-2.0
"""Shared MTP scaffolding for Qwen 3.5 and Qwen 3-Next.

Extracted from patches/qwen3_5_mtp.py and patches/qwen3_next_mtp.py.
Per-architecture builders are imported by the corresponding Architecture.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    module: nn.Module
    weights_loaded: int


def find_mtp_sidecar(model_path: Path) -> Path | None:
    """Return the path to the MTP sidecar weights, or None if absent."""
    candidates = [
        model_path / "mtp" / "weights.safetensors",
        model_path / "model-mtp.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_mtp_weights(sidecar_path: Path) -> dict[str, mx.array]:
    """Load + dequantize MTP sidecar weights and apply RMSNorm offset fixups.

    Body extracted verbatim from patches/qwen3_5_mtp.py (the dequantize loop +
    _apply_qwen_mtp_rmsnorm_offset_fixups). The fixup is the same for both
    architectures.
    """
    # COPY the dequantize loop + RMSNorm offset fixup from
    # patches/qwen3_5_mtp.py:inject_mtp_support lines 230–278 here.
    raise NotImplementedError("Copy from patches/qwen3_5_mtp.py during implementation")


def build_qwen35_mtp_module(model, config: Any, sidecar_path: Path) -> BuildResult:
    """Construct + attach the Qwen 3.5 MTP module. Caller has already verified
    that sidecar_path exists.
    """
    # EXTRACT the Qwen 3.5–specific module construction from
    # patches/qwen3_5_mtp.py:inject_mtp_support. Reuse load_mtp_weights for
    # the parts that are shared.
    raise NotImplementedError(
        "Extract from patches/qwen3_5_mtp.py during implementation"
    )


def build_qwen3_next_mtp_module(model, config: Any, sidecar_path: Path) -> BuildResult:
    """Construct + attach the Qwen 3-Next MTP module."""
    raise NotImplementedError(
        "Extract from patches/qwen3_next_mtp.py during implementation"
    )
```

Replace the three `NotImplementedError` blocks with code extracted from the existing patches. Keep the per-architecture differences minimal — anything identical between Qwen 3.5 and Qwen 3-Next lives in `load_mtp_weights` or as private helpers in this module.

- [ ] **Step 5: Run tests**

Run: `pytest tests/architectures/test_mtp_common.py -v`
Expected: 3 passed (only `find_mtp_sidecar` is tested at unit level; the builders are exercised by Tasks 13/14).

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/architectures/_mtp_common.py tests/architectures/test_mtp_common.py
git commit -m "feat(architectures): extract shared MTP scaffolding [ADR-0008, Task 12]"
```

---

## Task 13: Migrate Qwen 3.5 VL — MLLM + MTP + SpecPrefill

**Files:**
- Modify: `vllm_mlx/architectures/qwen3_5_vl.py`
- Create: `tests/architectures/test_qwen3_5_vl.py`

**Interfaces:**
- Consumes: `_mtp_common.find_mtp_sidecar`, `_mtp_common.build_qwen35_mtp_module`.
- Produces:
  - `Qwen3_5VLArchitecture.validate()` clears `self.mtp_available = False` if no sidecar (D8).
  - `install()` builds the MTP module when `mtp_available`; sets `model._architecture = self`. Patches the model class for batched MLLM as today's `patches/qwen3_5_mllm.py` does (move that logic in too).
  - `turn_end_token_ids` returns `{<|im_end|>}` (same as Qwen 3).
  - `extract_attention_query` returns the post-RoPE Q (same as Qwen 3 — Qwen 3.5 uses the same extractor).
  - `vision_processor(config)` returns the multimodal processor wrapper used by MLLMBatchGenerator (move logic from `patches/qwen3_5_mllm.py` + the architecture branches in `multimodal_processor.py`).

- [ ] **Step 1: Write tests** (similar shape to Task 11; add MTP cases)

`tests/architectures/test_qwen3_5_vl.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.architectures.qwen3_5_vl import Qwen3_5VLArchitecture


def test_mtp_available_cleared_when_sidecar_missing(tmp_path, caplog):
    arch = Qwen3_5VLArchitecture()
    assert arch.mtp_available is True  # default = has_mtp

    with caplog.at_level("WARNING"):
        arch.validate(model=MagicMock(), config=MagicMock(), model_path=tmp_path)

    assert arch.mtp_available is False
    assert any("MTP sidecar" in r.message for r in caplog.records)


def test_mtp_available_preserved_when_sidecar_present(tmp_path):
    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "weights.safetensors").touch()

    arch = Qwen3_5VLArchitecture()
    arch.validate(model=MagicMock(), config=MagicMock(), model_path=tmp_path)
    assert arch.mtp_available is True


def test_install_raises_on_corrupt_sidecar(tmp_path):
    from vllm_mlx.architectures import MTPWeightsLoadError

    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "weights.safetensors").write_bytes(b"not safetensors")

    arch = Qwen3_5VLArchitecture()
    arch.validate(model=MagicMock(), config=MagicMock(), model_path=tmp_path)

    with patch(
        "vllm_mlx.architectures._mtp_common.build_qwen35_mtp_module",
        side_effect=MTPWeightsLoadError("corrupt"),
    ):
        with pytest.raises(MTPWeightsLoadError):
            arch.install(model=MagicMock(), config=MagicMock(), model_path=tmp_path)
```

- [ ] **Step 2: Implement `qwen3_5_vl.py`**

Move the body of `patches/qwen3_5_mllm.py` and the MTP wiring into the architecture:

```python
# SPDX-License-Identifier: Apache-2.0
"""Qwen 3.5 VL (MoE, multimodal) architecture."""

import logging
from pathlib import Path
from typing import Optional

from . import register
from .base import Architecture, MTPWeightsLoadError
from ._mtp_common import (
    BuildResult,
    build_qwen35_mtp_module,
    find_mtp_sidecar,
)

logger = logging.getLogger(__name__)


@register
class Qwen3_5VLArchitecture(Architecture):
    model_type = "qwen3_5_moe"
    aliases = ("qwen3_5",)
    has_mtp = True
    has_vision = True

    def validate(self, model, config, model_path: Path) -> None:
        sidecar = find_mtp_sidecar(model_path) if model_path else None
        if self.has_mtp and sidecar is None:
            logger.warning(
                f"[Qwen3.5VL] MTP sidecar weights not found in {model_path}; "
                f"loading without MTP"
            )
            self.mtp_available = False

    def install(self, model, config, model_path: Path) -> None:
        from ..patches.qwen3_5_mllm import patch_qwen3_5_mllm_for_batching
        patch_qwen3_5_mllm_for_batching(model)  # move body inline if simple

        if self.mtp_available:
            sidecar = find_mtp_sidecar(model_path)
            if sidecar is None:
                # validate() should have caught this — defensive guard.
                self.mtp_available = False
            else:
                try:
                    build_qwen35_mtp_module(model, config, sidecar)
                except Exception as e:
                    raise MTPWeightsLoadError(
                        f"failed to build Qwen 3.5 MTP module: {e}"
                    ) from e

        model._architecture = self

    def turn_end_token_ids(self, tokenizer) -> Optional[set[int]]:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        unk = getattr(tokenizer, "unk_token_id", None)
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is None or tid == unk:
            return None
        return {tid}

    def extract_attention_query(self, attn_layer, layer_idx: int):
        if attn_layer is None:
            return None
        # Same as Qwen 3 — copy specprefill._qwen35_extract_queries here.
        raise NotImplementedError(
            "Copy specprefill._qwen35_extract_queries verbatim into this method"
        )

    def vision_processor(self, config):
        # Move the construction from mllm_batch_generator.py / multimodal_processor.py
        # specific to qwen3_5_moe / qwen3_5_vl here.
        raise NotImplementedError(
            "Extract from mllm_batch_generator.py / multimodal_processor.py "
            "the qwen3_5_moe vision-processor branch."
        )
```

Resolve the two `NotImplementedError` blocks by copying from the named source files.

- [ ] **Step 3: Run tests**

Run: `pytest tests/architectures/test_qwen3_5_vl.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/architectures/qwen3_5_vl.py tests/architectures/test_qwen3_5_vl.py
git commit -m "feat(architectures): migrate Qwen 3.5 VL (MLLM + MTP) [ADR-0008, Task 13]"
```

---

## Task 14: Migrate Qwen 3-Next — MTP + SpecPrefill

Mirror of Task 13 for Qwen 3-Next. Same shape; uses `build_qwen3_next_mtp_module` from `_mtp_common`; text-only (no `vision_processor`).

**Files:**
- Modify: `vllm_mlx/architectures/qwen3_next.py`
- Create: `tests/architectures/test_qwen3_next.py`

- [ ] **Step 1: Write tests** identical in shape to `test_qwen3_5_vl.py` but using `Qwen3NextArchitecture` and `build_qwen3_next_mtp_module`. Drop the `has_vision`-related cases.

- [ ] **Step 2: Implement `qwen3_next.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Qwen 3-Next architecture (text + MTP)."""

import logging
from pathlib import Path
from typing import Optional

from . import register
from .base import Architecture, MTPWeightsLoadError
from ._mtp_common import build_qwen3_next_mtp_module, find_mtp_sidecar

logger = logging.getLogger(__name__)


@register
class Qwen3NextArchitecture(Architecture):
    model_type = "qwen3_next"
    has_mtp = True

    def validate(self, model, config, model_path: Path) -> None:
        sidecar = find_mtp_sidecar(model_path) if model_path else None
        if self.has_mtp and sidecar is None:
            logger.warning(
                f"[Qwen3-Next] MTP sidecar weights not found in {model_path}; "
                f"loading without MTP"
            )
            self.mtp_available = False

    def install(self, model, config, model_path: Path) -> None:
        if self.mtp_available:
            sidecar = find_mtp_sidecar(model_path)
            if sidecar is None:
                self.mtp_available = False
            else:
                try:
                    build_qwen3_next_mtp_module(model, config, sidecar)
                except Exception as e:
                    raise MTPWeightsLoadError(
                        f"failed to build Qwen 3-Next MTP module: {e}"
                    ) from e
        model._architecture = self

    def turn_end_token_ids(self, tokenizer) -> Optional[set[int]]:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        unk = getattr(tokenizer, "unk_token_id", None)
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is None or tid == unk:
            return None
        return {tid}

    def extract_attention_query(self, attn_layer, layer_idx: int):
        if attn_layer is None:
            return None
        # Same Qwen3.5 extractor (the existing _EXTRACTOR_REGISTRY entry was
        # the same function for both qwen3_5* and qwen3_vl).
        raise NotImplementedError(
            "Copy specprefill._qwen35_extract_queries verbatim into this method"
        )
```

- [ ] **Step 3: Run tests + commit**

```bash
pytest tests/architectures/test_qwen3_next.py -v
git add vllm_mlx/architectures/qwen3_next.py tests/architectures/test_qwen3_next.py
git commit -m "feat(architectures): migrate Qwen 3-Next (MTP) [ADR-0008, Task 14]"
```

---

## Task 15: Migrate GLM-4.6V — MLLM

Mirror of Task 13 minus MTP. Move `patches/glm4v_moe_mllm.py` into the architecture's `install()`. Implement `vision_processor` and `turn_end_token_ids` (per GLM-4.6V chat template).

**Files:**
- Modify: `vllm_mlx/architectures/glm4v_moe.py`
- Create: `tests/architectures/test_glm4v_moe.py`

- [ ] **Step 1: Read `patches/glm4v_moe_mllm.py`** to understand what `install()` needs to do.

- [ ] **Step 2: Write tests** following the shape of Task 11 (turn delimiters) and Task 13's vision_processor test.

- [ ] **Step 3: Implement** — copy patch body into `install()`. Determine the GLM-4.6V turn-end token by checking its tokenizer / chat template (likely `<|user|>` and `<|assistant|>` markers; verify with a real model).

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/architectures/test_glm4v_moe.py -v
git add vllm_mlx/architectures/glm4v_moe.py tests/architectures/test_glm4v_moe.py
git commit -m "feat(architectures): migrate GLM-4.6V (MLLM) [ADR-0008, Task 15]"
```

---

## Task 16: Wire scheduler MTP gate to `architecture.mtp_available`

**Files:**
- Modify: `vllm_mlx/scheduler.py:945-957`
- Create: `tests/test_scheduler_mtp_gate.py`

**Interfaces:**
- Consumes: `self.model_wrapper.architecture.mtp_available`.
- Produces: gate uses `mtp_available` instead of `hasattr(model, "mtp")`.

- [ ] **Step 1: Write test**

`tests/test_scheduler_mtp_gate.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_mlx.architectures._testing import FakeArchitecture


def test_mtp_install_skipped_when_unavailable(caplog):
    # Patch the scheduler's _install_mtp to detect whether it was called.
    with patch("vllm_mlx.scheduler._install_mtp") as install_mtp:
        # Construct a minimal scheduler-like object that exercises just the
        # gate. Real scheduler construction is heavy; we test the gate logic
        # in isolation via the same code structure.
        scheduler = MagicMock()
        scheduler.config.enable_mtp = True
        scheduler.model_wrapper = SimpleNamespace(
            architecture=FakeArchitecture(has_mtp=True, mtp_available=False,
                                          model_type="fake_no_mtp")
        )

        # Execute the gate (extracted as a helper in the refactor below).
        from vllm_mlx.scheduler import _maybe_install_mtp
        _maybe_install_mtp(scheduler, bg=MagicMock())

        assert install_mtp.call_count == 0


def test_mtp_install_called_when_available():
    with patch("vllm_mlx.scheduler._install_mtp") as install_mtp:
        scheduler = MagicMock()
        scheduler.config.enable_mtp = True
        scheduler.config.mtp_num_draft_tokens = 2
        scheduler.config.mtp_optimistic = False
        scheduler.model = MagicMock()
        scheduler.model_wrapper = SimpleNamespace(
            architecture=FakeArchitecture(has_mtp=True, mtp_available=True)
        )

        from vllm_mlx.scheduler import _maybe_install_mtp
        _maybe_install_mtp(scheduler, bg=MagicMock())
        assert install_mtp.call_count == 1
```

- [ ] **Step 2: Refactor the gate into a helper**

Replace `vllm_mlx/scheduler.py:945-957` with:

```python
        _maybe_install_mtp(self, bg)
```

And add the helper near the top of `vllm_mlx/scheduler.py` (above `Scheduler` class definition or as a module-level function):

```python
def _maybe_install_mtp(scheduler, bg) -> None:
    """Install MTP on `bg` iff config + architecture both allow it."""
    if not scheduler.config.enable_mtp:
        return
    architecture = scheduler.model_wrapper.architecture
    if architecture.mtp_available:
        _install_mtp(
            bg,
            model=scheduler.model,
            num_draft_tokens=scheduler.config.mtp_num_draft_tokens,
            optimistic=scheduler.config.mtp_optimistic,
        )
    else:
        logger.warning(
            f"[MTP] --enable-mtp set but architecture {architecture.model_type} "
            f"reports mtp_available=False (has_mtp={architecture.has_mtp}); "
            f"MTP disabled for this model"
        )
```

- [ ] **Step 3: Run tests + commit**

```bash
pytest tests/test_scheduler_mtp_gate.py -v
git add vllm_mlx/scheduler.py tests/test_scheduler_mtp_gate.py
git commit -m "refactor(scheduler): gate MTP on architecture.mtp_available [ADR-0008, Task 16]"
```

---

## Task 17: Wire `vision_processor` at MLLM pipeline construction

**Files:**
- Modify: `vllm_mlx/mllm_batch_generator.py` (constructor / processor wiring)
- Modify: `vllm_mlx/multimodal_processor.py` (collapse architecture branches)
- Create: `tests/test_mllm_vision_processor.py`

**Interfaces:**
- Consumes: `architecture.vision_processor(config)`.
- Produces: MLLM pipeline construction routes through the architecture; multimodal_processor branches collapse.

- [ ] **Step 1: Audit existing branches**

```bash
grep -n "model_type\|qwen3_5\|qwen3_vl\|gemma\|glm" vllm_mlx/multimodal_processor.py vllm_mlx/mllm_batch_generator.py | head -30
```

- [ ] **Step 2: Write test** asserting that the architecture's processor is preferred over the legacy lookup.

`tests/test_mllm_vision_processor.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_mlx.architectures._testing import FakeArchitecture


def test_architecture_processor_takes_precedence():
    fake_proc = MagicMock(name="from_arch")
    arch = FakeArchitecture(has_vision=True, processor=fake_proc)
    mw = SimpleNamespace(architecture=arch, config=MagicMock())

    from vllm_mlx.mllm_batch_generator import resolve_vision_processor
    resolved = resolve_vision_processor(mw)
    assert resolved is fake_proc


def test_warning_emitted_when_capability_claimed_but_processor_missing(caplog):
    arch = FakeArchitecture(has_vision=True, processor=None, model_type="claims_vision")
    mw = SimpleNamespace(architecture=arch, config=MagicMock())

    from vllm_mlx.mllm_batch_generator import resolve_vision_processor
    with caplog.at_level("WARNING"):
        resolved = resolve_vision_processor(mw)
    assert resolved is None
    assert any("claims_vision" in r.message for r in caplog.records)
```

- [ ] **Step 3: Implement `resolve_vision_processor`**

In `vllm_mlx/mllm_batch_generator.py`, add module-level:

```python
def resolve_vision_processor(model_wrapper):
    """Return the architecture-supplied vision processor, or None.

    Logs a WARNING when the architecture claims vision/audio capability but
    returns no processor — this indicates a misconfigured architecture, not
    a user error.
    """
    architecture = model_wrapper.architecture
    processor = architecture.vision_processor(model_wrapper.config)
    if processor is None and (architecture.has_vision or architecture.has_audio):
        logger.warning(
            f"[mllm] architecture {architecture.model_type} declares "
            f"vision/audio capability but returned no processor — "
            f"multimodal inputs will fail"
        )
    return processor
```

Replace call sites that currently branch on `model_type` to construct processors.

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/test_mllm_vision_processor.py tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q
git add vllm_mlx/mllm_batch_generator.py vllm_mlx/multimodal_processor.py \
        tests/test_mllm_vision_processor.py
git commit -m "refactor(mllm): route vision processor through architecture [ADR-0008, Task 17]"
```

---

## Task 18: Delete `_try_inject_mtp` from `utils/tokenizer.py`

**Files:**
- Modify: `vllm_mlx/utils/tokenizer.py`
- Modify: any caller of `_try_inject_mtp` or `_try_inject_mtp_post_load`

**Interfaces:** No public-API change. The functions were called internally during `load_model_with_fallback`.

- [ ] **Step 1: Find all callers**

```bash
grep -rn "_try_inject_mtp" vllm_mlx tests
```

- [ ] **Step 2: Remove `_try_inject_mtp`, `_try_inject_mtp_post_load`, and their call sites in `tokenizer.py`**

MTP loading now happens in `Architecture.install()` — these helpers are dead. Specifically:

- Delete `_try_inject_mtp` (lines ~156–177).
- Delete `_try_inject_mtp_post_load` (lines ~179–215).
- Delete the call to `_try_inject_mtp_post_load(model, model_name)` at the end of `load_model_with_fallback`.
- Delete the call to `_try_inject_mtp(model, model_path, config)` in `_load_strict_false`.

- [ ] **Step 3: Run tests including a Qwen 3-Next or Qwen 3.5 VL smoke load**

```bash
pytest tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q
```

If a Qwen 3.5 VL or Qwen 3-Next checkpoint is locally available:

```bash
python -c "
from vllm_mlx.models.llm import MLXLanguageModel
m = MLXLanguageModel(model_name='<qwen3_5 checkpoint>')
m.load()
print('mtp_available:', m.architecture.mtp_available)
print('model.mtp:', getattr(m.model, 'mtp', None) is not None)
"
```

Both should match (True/True if sidecar present; False/False otherwise).

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/utils/tokenizer.py
git commit -m "refactor(tokenizer): delete _try_inject_mtp (moved to Architecture.install) [ADR-0008, Task 18]"
```

---

## Task 19: Delete `vllm_mlx/patches/`

**Files:**
- Delete: `vllm_mlx/patches/__init__.py`
- Delete: `vllm_mlx/patches/gemma4_llm.py`
- Delete: `vllm_mlx/patches/glm4v_moe_mllm.py`
- Delete: `vllm_mlx/patches/qwen3_5_mllm.py`
- Delete: `vllm_mlx/patches/qwen3_5_mtp.py`
- Delete: `vllm_mlx/patches/qwen3_next_mtp.py`
- Delete: `vllm_mlx/patches/mlx_lm_prefill_flash_sdpa.py`
- Delete: `vllm_mlx/patches/mlx_lm_quantized_sdpa.py`
- Modify: `vllm_mlx/global_runtime_fixes.py` — inline the SDPA patch bodies (Task 2 imported from `patches/`; now we copy them in).

- [ ] **Step 1: Inline the two SDPA patches into `global_runtime_fixes.py`**

Copy the body of `patches/mlx_lm_quantized_sdpa.py:patch_quantized_sdpa` and `patches/mlx_lm_prefill_flash_sdpa.py:apply` directly into `global_runtime_fixes.py`. Remove the lazy imports.

- [ ] **Step 2: Verify nothing else imports from `vllm_mlx.patches`**

```bash
grep -rn "from .patches\|from vllm_mlx.patches\|vllm_mlx\.patches" vllm_mlx tests
```

Expected: empty. If anything remains, the migration is incomplete — stop and fix before deleting.

- [ ] **Step 3: Delete the directory**

```bash
git rm -r vllm_mlx/patches/
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -x --ignore=tests/test_cache_hit_oom_repro.py -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A vllm_mlx/global_runtime_fixes.py vllm_mlx/patches
git commit -m "refactor: retire vllm_mlx/patches; inline SDPA fixes into global_runtime_fixes [ADR-0008, Task 19]"
```

---

## Task 20: Migration parity script (not committed)

**Files:**
- Create: `scripts/migration_parity_check.py` (do NOT commit — listed in `.gitignore` or simply not added)

**Interfaces:** Standalone script. Loads each currently-supported model, runs a 4-turn conversation with turn cache enabled, prints hit/miss/save counters, asserts they match a baseline recorded from `main` before the migration.

- [ ] **Step 1: Record baseline on `main`**

Before starting the migration, run on `main`:

```bash
git stash
git checkout main
python scripts/migration_parity_check.py --record baseline.json
git checkout debug-variable-quant
git stash pop
```

(If `main` doesn't yet have this script, run a minimal capture: load each model, run a 4-turn convo, print hit counters.)

- [ ] **Step 2: Implement `scripts/migration_parity_check.py`**

```python
"""Migration parity check — verify the architecture-layer migration produces
the same turn-cache hit/miss/save behavior as the pre-migration code.

Throwaway: deleted after the migration ships. Not committed.
"""
import argparse
import json
from pathlib import Path

MODELS = [
    "mlx-community/Qwen3-0.6B-bf16",
    "mlx-community/gemma-3-4b-it-bf16",
    # Add the local checkpoints for the other architectures.
]

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "What about Germany?"},
    {"role": "assistant", "content": "Berlin."},
    {"role": "user", "content": "Italy?"},
    {"role": "assistant", "content": "Rome."},
    {"role": "user", "content": "Spain?"},
]


def run_one(model_name: str) -> dict:
    from vllm_mlx.models.llm import MLXLanguageModel
    # ... load, run a 4-turn convo, return {"hits": ..., "misses": ..., "saves": ...}
    raise NotImplementedError("Wire up to the actual cache counters")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()

    results = {m: run_one(m) for m in MODELS}

    if args.record:
        args.record.write_text(json.dumps(results, indent=2))
        print(f"Recorded baseline to {args.record}")
        return

    if args.compare:
        baseline = json.loads(args.compare.read_text())
        for model, current in results.items():
            base = baseline[model]
            assert current == base, f"DRIFT on {model}: {current} vs {base}"
        print("Parity verified.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run parity check on the migration branch**

```bash
python scripts/migration_parity_check.py --compare baseline.json
```

Expected output: `Parity verified.`

- [ ] **Step 4: Discard the script (do not commit)**

```bash
rm scripts/migration_parity_check.py baseline.json
```

The parity verification is recorded in the PR description, not in tree.

---

## Self-review

**Spec coverage:** every spec section maps to at least one task —

| Spec section | Task(s) |
|---|---|
| Module layout | Tasks 1, 2, 3, 4 (skeletons) |
| `Architecture` protocol | Task 1 |
| Exception types | Task 1 |
| Registry | Task 1 |
| Lifecycle (server bootstrap + defensive) | Tasks 2, 5, 6 |
| MTP injection moves into install | Tasks 12, 13, 14, 18 |
| Defensive MTP UX (`mtp_available`) | Task 12 (`find_mtp_sidecar`), 13, 14 |
| `scheduler.py:41–43` (remove eager installs) | Tasks 2, 10 |
| `scheduler.py:945–957` (MTP gate) | Task 16 |
| `_compute_turn_boundaries` collapse | Task 7 |
| `specprefill.py` extractor switch | Task 8 |
| `is_mllm_model` | Task 9 |
| Vision processor | Task 17 |
| `_try_inject_mtp` deletion | Task 18 |
| Behavior changes table | Implicit in test assertions (Tasks 7, 8, 11–16) |
| Testing strategy (FakeArchitecture, per-arch units, parity) | Tasks 3, 10–15, 20 |
| Migration plan | Task order matches |

**Placeholder scan:** intentional placeholders flagged with "PASTE THE CURRENT … IMPLEMENTATION HERE" or `NotImplementedError("Copy from …")` exist in Tasks 11, 13, 14, 15. These are deliberate handoff points where the implementer reads a specific source file and copies the body verbatim. The plan names the source file and the function in every case. Acceptable per the project's existing-patch-migration pattern, but called out here so the implementer knows to expect them.

**Type consistency:** `Architecture` class name, `architecture` attribute, `mtp_available` instance attr, `model_path: Path` parameter — all consistent across tasks 1–20.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-16-architecture-layer.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Parallelizable on Tasks 10, 11, 15 (independent per-architecture migrations).

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
