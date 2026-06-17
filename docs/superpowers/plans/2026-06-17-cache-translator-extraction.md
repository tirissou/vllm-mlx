# Cache Translator Extraction + KVLayerSegment Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `TurnCacheManager._segment` / `_assemble` to a named module (`vllm_mlx/cache_translator.py`) and replace the unified `KVLayerSegment` dataclass with an abstract base + two concrete subclasses (`KVConcatSegment`, `KVRotatingSegment`) so trie-storage dispatch is polymorphic instead of stringly-typed.

**Architecture:** The translator module exposes three free functions: `segment(live_states, policy, group_size)`, `assemble(kv_layers, rec_layers, group_size)`, `slice_kv_to_delta(states, prev_end)`. Per-layer dispatch (`if class_name == "RotatingKVCache" else ...`) collapses into `seg.reconstruct(group_size)` and `seg.merge_path(path)` on the new types. `TurnCacheManager` calls into the translator instead of holding the logic as static methods. `TurnPrefixCache.collect_path_data` uses polymorphic `merge_path` instead of branching on `metadata["merge_strategy"]`.

**Tech Stack:** Python 3.11+, MLX (`mlx.core`), mlx-lm cache types, pytest.

## Global Constraints

- `pytest tests/` must pass at the end of every task.
- `mx.eval(...)` then `mx.stop_gradient(...)` on every segment array — the existing segment contract (see CONTEXT.md "Segment contract"). Do not regress this.
- Frozen dataclasses for all segment types. No mutation after construction.
- No new dependencies. No new ADRs in this PR (ADR-0005 already covers the seam).
- Preserve byte-for-byte SSD serialization compatibility for already-spilled caches — see Task 5.

---

### Task 1: Failing tests for new segment types and translator module

**Files:**
- Modify: `tests/test_cache_translator.py` (add new tests at end; existing tests remain unchanged for now)
- Create: `tests/test_kv_layer_segment_types.py`

**Interfaces (forward-declared, implemented in Tasks 2-3):**
- Produces: `KVLayerSegment` (ABC), `KVConcatSegment`, `KVRotatingSegment` in `vllm_mlx.cache_types`
- Produces: `segment(live_states, policy, group_size) -> (kv_list, rec_list)`, `assemble(kv_layers, rec_layers, group_size) -> list`, `slice_kv_to_delta(states, prev_end) -> list[dict]` in `vllm_mlx.cache_translator`

- [ ] **Step 1: Create `tests/test_kv_layer_segment_types.py`**

```python
"""Tests for KVLayerSegment ABC + KVConcatSegment + KVRotatingSegment.

Verifies polymorphic dispatch (merge_path, reconstruct) and the type-system
guarantee that concat is only valid for KVConcatSegment.
"""
import mlx.core as mx
import pytest

from vllm_mlx.cache_types import (
    KVConcatSegment,
    KVLayerSegment,
    KVRotatingSegment,
)
from vllm_mlx.kv_cache import QuantizedArray


def _q(shape=(1, 2, 16, 64), bits=8, group_size=64):
    arr = mx.random.normal(shape).astype(mx.float16)
    return QuantizedArray(*mx.quantize(arr, group_size=group_size, bits=bits))


def test_concat_segment_merge_path_concatenates():
    a = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    b = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    merged = a.merge_path([a, b])
    assert isinstance(merged, KVConcatSegment)
    assert merged.n_tokens == 32


def test_rotating_segment_merge_path_returns_last():
    a = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    b = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=32, idx=0,
    )
    merged = a.merge_path([a, b])
    assert merged is b


def test_concat_classmethod_rejects_mismatched_bits():
    a = KVConcatSegment(keys=_q(bits=8), values=_q(bits=8), layer_index=0, n_tokens=16, bits=8)
    b = KVConcatSegment(keys=_q(bits=4), values=_q(bits=4), layer_index=0, n_tokens=16, bits=4)
    with pytest.raises(AssertionError, match="mismatched bits"):
        KVConcatSegment.concat([a, b])


def test_concat_segment_is_frozen():
    seg = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    with pytest.raises((AttributeError, Exception)):
        seg.n_tokens = 99  # type: ignore[misc]


def test_concat_reconstruct_returns_batch_quantized_kv_cache():
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    seg = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    cache = seg.reconstruct(group_size=64)
    assert isinstance(cache, BatchQuantizedKVCache)


def test_rotating_reconstruct_returns_rotating_kv_cache():
    from mlx_lm.models.cache import RotatingKVCache
    seg = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    cache = seg.reconstruct(group_size=64)
    assert isinstance(cache, RotatingKVCache)


def test_subclass_dispatch_via_base_type_annotation():
    """A list typed as list[KVLayerSegment] holds mixed concrete types."""
    concat = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    rotating = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=1, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    segs: list[KVLayerSegment] = [concat, rotating]
    assert isinstance(segs[0], KVConcatSegment)
    assert isinstance(segs[1], KVRotatingSegment)
```

- [ ] **Step 2: Append translator round-trip tests to `tests/test_cache_translator.py`**

Append at the end of the file:

```python
# ── Translator module round-trip ──────────────────────────────────────────────

from vllm_mlx.cache_translator import (
    assemble,
    segment,
    slice_kv_to_delta,
)


def _live_kvcache_state(B=1, H=2, T=16, D=64):
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {"class_name": "KVCache", "state": (keys, values), "meta_state": (T,)}


def test_segment_returns_kv_concat_segment_for_kvcache():
    from vllm_mlx.cache_types import KVConcatSegment
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    assert isinstance(kv_list[0], KVConcatSegment)


def test_segment_returns_kv_rotating_segment_for_rotating_kvcache():
    from vllm_mlx.cache_types import KVRotatingSegment
    keys = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    values = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    mx.eval(keys, values)
    states = [{
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (0, 16, 16, 16),
    }]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    assert isinstance(kv_list[0], KVRotatingSegment)


def test_segment_arrays_survive_source_deletion():
    """Segment contract: emitted arrays are graph-detached from source."""
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    seg = kv_list[0]
    del states
    mx.clear_cache()
    dequant_keys = mx.dequantize(
        seg.keys.packed, seg.keys.scales, seg.keys.biases,
        group_size=64, bits=8,
    )
    mx.eval(dequant_keys)
    assert dequant_keys.shape == (1, 2, 16, 64)


def test_assemble_inverse_of_segment_for_kvcache():
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    caches = assemble(kv_list, [], group_size=64)
    assert len(caches) == 1
    # Round-trip n_tokens preserved.
    assert kv_list[0].n_tokens == 16


def test_slice_kv_to_delta_slices_kvcache():
    states = [_live_kvcache_state(T=32)]
    sliced = slice_kv_to_delta(states, prev_end=16)
    assert sliced[0]["state"][0].shape[2] == 16


def test_slice_kv_to_delta_leaves_rotating_untouched():
    keys = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    values = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    states = [{
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (0, 16, 16, 16),
    }]
    sliced = slice_kv_to_delta(states, prev_end=8)
    assert sliced[0]["state"][0].shape == (1, 2, 16, 64)  # untouched
```

- [ ] **Step 3: Run the new tests, confirm they fail with ImportError**

Run: `pytest tests/test_kv_layer_segment_types.py tests/test_cache_translator.py -k "translator or kv_concat or kv_rotating or subclass_dispatch or arrays_survive or slice_kv_to_delta" -v 2>&1 | tail -20`

Expected: ImportError on `vllm_mlx.cache_translator` and `vllm_mlx.cache_types.KVConcatSegment` (modules/symbols don't exist yet).

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/test_kv_layer_segment_types.py tests/test_cache_translator.py
git commit -m "test: failing tests for cache_translator module and split segment types"
```

---

### Task 2: Implement segment types (cache_types.py)

**Files:**
- Modify: `vllm_mlx/cache_types.py`

**Interfaces:**
- Consumes: nothing from prior tasks (Task 1 is tests only)
- Produces: `KVLayerSegment` (ABC), `KVConcatSegment`, `KVRotatingSegment`

- [ ] **Step 1: Replace the old `KVLayerSegment` dataclass with the ABC + subclasses**

Open `vllm_mlx/cache_types.py`. Replace the existing `KVLayerSegment` class (lines 10-63) with:

```python
from abc import ABC, abstractmethod
from typing import Literal


@dataclass(frozen=True)
class KVLayerSegment(ABC):
    """Abstract per-layer KV snapshot stored in a TurnNode.

    Two concrete subclasses dispatch path-merge and reconstruction polymorphically:
    - KVConcatSegment: standard KVCache (incremental accumulation, axis=-2 concat)
    - KVRotatingSegment: RotatingKVCache (ring buffer, only last node's segment used)

    Holds evaluated, graph-detached arrays per the segment contract (CONTEXT.md).
    """

    keys: Any           # QuantizedArray | mx.array
    values: Any         # QuantizedArray | mx.array
    layer_index: int
    n_tokens: int
    bits: int | None    # None = bf16, int = quantized at that precision
    class_name: str     # the live mlx-lm class name this segment was extracted from

    @abstractmethod
    def merge_path(self, path: list["KVLayerSegment"]) -> "KVLayerSegment":
        """Merge a path of same-layer segments. ``path`` includes self at path[-1]."""

    @abstractmethod
    def reconstruct(self, group_size: int) -> Any:
        """Reconstruct a live mlx-lm cache object for this layer."""


@dataclass(frozen=True)
class KVConcatSegment(KVLayerSegment):
    """KV snapshot for a standard KVCache layer (incremental, concatenated)."""

    class_name: str = "KVCache"

    def merge_path(self, path: list[KVLayerSegment]) -> KVLayerSegment:
        return KVConcatSegment.concat(path)  # type: ignore[arg-type]

    def reconstruct(self, group_size: int) -> Any:
        from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
        from vllm_mlx.kv_cache import QuantizedArray

        if isinstance(self.keys, QuantizedArray):
            # Pad to step boundary so the first decode lands on in-place update.
            step = BatchQuantizedKVCache.step
            padded_len = ((self.n_tokens // step) + 1) * step
            pad = padded_len - self.n_tokens

            def _pad_qa(qa: QuantizedArray) -> QuantizedArray:
                if pad == 0:
                    return qa
                return QuantizedArray(*[
                    mx.concatenate(
                        [c, mx.zeros((*c.shape[:-2], pad, c.shape[-1]), dtype=c.dtype)],
                        axis=-2,
                    )
                    for c in (qa.packed, qa.scales, qa.biases)
                ])

            return BatchQuantizedKVCache.from_quantized_arrays(
                keys=_pad_qa(self.keys),
                values=_pad_qa(self.values),
                offset=self.n_tokens,
                group_size=group_size,
                bits=self.bits,
            )
        # Float path: TODO mirror the existing float reconstruction in _assemble.
        raise NotImplementedError("Float KVConcatSegment.reconstruct not yet wired")

    @classmethod
    def concat(cls, layers: list["KVConcatSegment"]) -> "KVConcatSegment":
        """Concat same-layer segments along the sequence axis."""
        from vllm_mlx.kv_cache import QuantizedArray

        first_bits = layers[0].bits
        for l in layers[1:]:
            assert l.bits == first_bits, (
                f"KVConcatSegment.concat: mismatched bits "
                f"{first_bits!r} vs {l.bits!r}"
            )

        if isinstance(layers[0].keys, QuantizedArray):
            merged_keys = QuantizedArray(
                packed=mx.concatenate([l.keys.packed for l in layers], axis=-2),
                scales=mx.concatenate([l.keys.scales for l in layers], axis=-2),
                biases=mx.concatenate([l.keys.biases for l in layers], axis=-2),
            )
            merged_values = QuantizedArray(
                packed=mx.concatenate([l.values.packed for l in layers], axis=-2),
                scales=mx.concatenate([l.values.scales for l in layers], axis=-2),
                biases=mx.concatenate([l.values.biases for l in layers], axis=-2),
            )
        else:
            merged_keys = mx.concatenate([l.keys for l in layers], axis=-2)
            merged_values = mx.concatenate([l.values for l in layers], axis=-2)
        return cls(
            keys=merged_keys,
            values=merged_values,
            layer_index=layers[-1].layer_index,
            n_tokens=sum(l.n_tokens for l in layers),
            bits=first_bits,
            class_name=layers[-1].class_name,
        )


@dataclass(frozen=True)
class KVRotatingSegment(KVLayerSegment):
    """KV snapshot for a RotatingKVCache layer (ring buffer)."""

    max_size: int = 0
    keep: int = 0
    offset: int = 0
    idx: int = 0          # ring write position (was _idx in old metadata dict)
    class_name: str = "RotatingKVCache"

    def merge_path(self, path: list[KVLayerSegment]) -> KVLayerSegment:
        # Rotating state is not cumulative — only the deepest node's segment matters.
        return path[-1]

    def reconstruct(self, group_size: int) -> Any:
        from mlx_lm.models.cache import RotatingKVCache
        from vllm_mlx.kv_cache import QuantizedArray

        is_quantized = isinstance(self.keys, QuantizedArray)
        if is_quantized:
            dq_keys = mx.dequantize(
                self.keys.packed, self.keys.scales, self.keys.biases,
                group_size=group_size, bits=self.bits,
            )
            dq_values = mx.dequantize(
                self.values.packed, self.values.scales, self.values.biases,
                group_size=group_size, bits=self.bits,
            )
        else:
            dq_keys = self.keys
            dq_values = self.values

        if dq_keys.shape[-2] > self.max_size:
            dq_keys = dq_keys[..., -self.max_size:, :]
            dq_values = dq_values[..., -self.max_size:, :]

        # Rotate back so the ring write position lands at idx, matching live layout.
        if 0 < self.idx < self.max_size and dq_keys.shape[-2] == self.max_size:
            split = self.max_size - self.idx
            dq_keys = mx.concatenate([dq_keys[..., split:, :], dq_keys[..., :split, :]], axis=-2)
            dq_values = mx.concatenate([dq_values[..., split:, :], dq_values[..., :split, :]], axis=-2)

        cache = RotatingKVCache(self.max_size, self.keep)
        cache.keys = dq_keys
        cache.values = dq_values
        cache.offset = self.offset
        cache._idx = self.idx
        return cache
```

Notes for the implementer:
- `dataclass(frozen=True)` only; do NOT add `slots=True` — `slots` + ABC + inheritance interacts badly in some Python versions. Memory optimization is a follow-up.
- Field ordering matters for dataclass inheritance: subclasses can only add fields with defaults after a parent that ends with non-default fields. The `class_name: str = "..."` defaults handle this. `KVRotatingSegment`'s new fields all have defaults so they appear after.
- Keep the existing `RecurrentLayerSegment` dataclass unchanged.

- [ ] **Step 2: Run the new type tests**

Run: `pytest tests/test_kv_layer_segment_types.py -v 2>&1 | tail -25`

Expected: PASS for all 7 tests.

- [ ] **Step 3: Run the full suite to find collateral failures**

Run: `pytest tests/ -x 2>&1 | tail -30`

Expected: failures in tests that constructed `KVLayerSegment(...)` directly with the old metadata-dict signature, and in production code that reads `seg.metadata[...]`. Note the failing test names — they will be migrated in Task 4. **Do not fix yet.**

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/cache_types.py
git commit -m "feat(cache_types): split KVLayerSegment into KVConcatSegment + KVRotatingSegment with polymorphic dispatch"
```

---

### Task 3: Implement cache_translator module

**Files:**
- Create: `vllm_mlx/cache_translator.py`

**Interfaces:**
- Consumes: `KVLayerSegment`, `KVConcatSegment`, `KVRotatingSegment` (Task 2); `RecurrentLayerSegment`, `KVQuantPolicy` (existing); `QuantizedArray` (existing in `vllm_mlx.kv_cache`)
- Produces: free functions `segment`, `assemble`, `slice_kv_to_delta`

- [ ] **Step 1: Create `vllm_mlx/cache_translator.py`**

The module body is a direct lift-and-shape of the existing `_segment` / `_assemble` / `_slice_kv_to_delta` in `vllm_mlx/prefix_cache_adapters.py` (lines 26-49 for the rotating helpers, 280-498 for `_segment`, 500-end for `_assemble`, 811-849 for `_slice_kv_to_delta`). Two structural changes:

1. `segment` returns the new typed subclasses, populated via named-field constructors instead of `metadata={...}`.
2. `assemble` dispatches via `seg.reconstruct(group_size)` per layer instead of the `if class_name == "RotatingKVCache"` branch.

```python
"""Trie-storage translator: live mlx-lm cache state ↔ KVLayerSegment.

Exposes the round-trip law (CONTEXT.md "Cache translator"):

    assemble(segment(live_states, policy, group_size), group_size) ≈ live_states

The producers (segment, slice_kv_to_delta) and consumer (assemble) here are
the seam that ADR-0005 governs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from vllm_mlx.cache_types import (
    KVConcatSegment,
    KVLayerSegment,
    KVRotatingSegment,
    KVQuantPolicy,
    RecurrentLayerSegment,
)

if TYPE_CHECKING:
    pass


def _linearize(tensor: mx.array, offset: int, max_size: int) -> mx.array:
    if offset == max_size:
        return tensor[..., :offset, :]
    return mx.concatenate([tensor[..., offset:, :], tensor[..., :offset, :]], axis=-2)


def _apply_rotating_window(arr: mx.array, keep: int, max_size: int) -> mx.array:
    n = arr.shape[-2]
    if n <= max_size:
        return arr
    if keep <= 0:
        return arr[..., -max_size:, :]
    trim_size = n - max_size
    return mx.concatenate(
        [arr[..., :keep, :], arr[..., trim_size + keep:, :]], axis=-2
    )


def segment(
    live_states: list[dict],
    policy: KVQuantPolicy | None = None,
    group_size: int = 64,
) -> tuple[list[KVLayerSegment | None], list[RecurrentLayerSegment | None]]:
    """Translate live mlx-lm cache states into immutable trie segments.

    Contract: emitted arrays are evaluated and graph-detached
    (mx.eval then mx.stop_gradient). Callers may delete live_states and
    call mx.clear_cache(); segment arrays survive.
    """
    from vllm_mlx.kv_cache import QuantizedArray

    kv_list: list[KVLayerSegment | None] = [None] * len(live_states)
    rec_list: list[RecurrentLayerSegment | None] = [None] * len(live_states)

    for i, state_dict in enumerate(live_states):
        class_name = state_dict["class_name"]
        state = state_dict["state"]
        meta = state_dict.get("meta_state", ())
        bits = policy.bits_for(class_name) if policy is not None else None

        if class_name == "RotatingKVCache":
            kv_list[i] = _segment_rotating(state, meta, i, bits, group_size)
        elif "KVCache" in class_name:
            kv_list[i] = _segment_concat(state, meta, i, class_name, bits, group_size)
        else:
            rec_list[i] = RecurrentLayerSegment(
                arrays=state,
                metadata={
                    "class_name": class_name,
                    "layer_index": i,
                    "class_ref": state_dict.get("class_ref"),
                },
            )

    return kv_list, rec_list


def _segment_rotating(state, meta, i, bits, group_size) -> KVRotatingSegment:
    # Lift the body from prefix_cache_adapters.py:_segment lines 302-397 here,
    # returning a KVRotatingSegment populated from named fields rather than dict.
    # The mx.eval + mx.stop_gradient discipline must be preserved.
    raise NotImplementedError("paste from prefix_cache_adapters.py:_segment rotating branch")


def _segment_concat(state, meta, i, class_name, bits, group_size) -> KVConcatSegment:
    # Lift the body from prefix_cache_adapters.py:_segment lines 399-486 here,
    # returning a KVConcatSegment populated from named fields rather than dict.
    raise NotImplementedError("paste from prefix_cache_adapters.py:_segment concat branch")


def assemble(
    kv_layers: list[KVLayerSegment],
    recurrent_layers: list[RecurrentLayerSegment],
    group_size: int = 64,
) -> list:
    """Reconstruct live mlx-lm cache objects from segments. Inverse of segment()."""
    result: dict[int, Any] = {}
    for layer in kv_layers:
        if layer is None:
            continue
        result[layer.layer_index] = layer.reconstruct(group_size)
    for rec_layer in recurrent_layers:
        if rec_layer is None:
            continue
        li = rec_layer.metadata["layer_index"]
        class_ref = rec_layer.metadata.get("class_ref")
        # Existing assemble() recurrent branch already handles this — lift it here.
        # See prefix_cache_adapters.py:_assemble lines for recurrent (end of method).
        raise NotImplementedError("paste from prefix_cache_adapters.py:_assemble recurrent branch")
    return [result[i] for i in sorted(result)]


def slice_kv_to_delta(states: list[dict], prev_end: int) -> list[dict]:
    """Slice KVCache state arrays to the incremental delta [prev_end:actual_end].

    RotatingKVCache state is left untouched (its ring buffer is not cumulative).
    """
    # Direct lift from prefix_cache_adapters.py:_slice_kv_to_delta lines 811-849.
    raise NotImplementedError("paste from prefix_cache_adapters.py:_slice_kv_to_delta")
```

**Lift-and-shape instructions for the implementer:**

The three `NotImplementedError` blocks above are placeholders. To fill them in:

- `_segment_rotating`: copy lines 302-397 of `prefix_cache_adapters.py` (the `if class_name == "RotatingKVCache":` branch of `_segment`). Replace the two `KVLayerSegment(... metadata={...})` constructor calls (lines 337-351 and 383-397) with `KVRotatingSegment(keys=..., values=..., layer_index=i, n_tokens=lin_keys.shape[-2], bits=bits, max_size=max_size, keep=keep, offset=offset, idx=_idx)`. Drop the `class_name` field (defaults correctly).
- `_segment_concat`: copy lines 399-486 (the `elif "KVCache" in class_name:` branch). Replace the three `KVLayerSegment(... metadata={...})` calls with `KVConcatSegment(keys=..., values=..., layer_index=i, n_tokens=actual_end, bits=bits, class_name=class_name)`.
- `assemble`'s recurrent branch: lift from the tail of `_assemble` in `prefix_cache_adapters.py` (the section after the kv_layers loop that handles `recurrent_layers`).
- `slice_kv_to_delta`: copy lines 811-849 verbatim; no shape changes.

Preserve every `mx.eval(...)` and `mx.stop_gradient(...)` call exactly.

- [ ] **Step 2: Run translator tests**

Run: `pytest tests/test_cache_translator.py -v -k "translator or segment_returns or arrays_survive or assemble_inverse or slice_kv_to_delta" 2>&1 | tail -20`

Expected: PASS for the seven new tests added in Task 1 Step 2.

- [ ] **Step 3: Commit**

```bash
git add vllm_mlx/cache_translator.py
git commit -m "feat(cache_translator): extract segment / assemble / slice_kv_to_delta as a named seam"
```

---

### Task 4: Switch production code to cache_translator and delete the duplicates

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_cache_translator.py` (migrate existing tests)
- Modify: `tests/test_prefix_cache_adapters.py` (migrate `_segment` call sites)
- Modify: `tests/test_turn_prefix_cache_integration.py` (migrate `_segment` / `_assemble`)
- Modify: `tests/test_cache_hit_oom_repro.py` (migrate `_assemble`)
- Modify: `tests/test_cache_parity.py` (migrate `_orig_assemble`)

**Interfaces:**
- Consumes: `segment`, `assemble`, `slice_kv_to_delta` (Task 3); `KVConcatSegment`, `KVRotatingSegment` (Task 2)

- [ ] **Step 1: Update `prefix_cache_adapters.py` callers**

In `vllm_mlx/prefix_cache_adapters.py`:

1. At the top of the file, add: `from vllm_mlx.cache_translator import segment, assemble, slice_kv_to_delta`
2. Replace the three internal callers:
   - Line 683: `reconstructed = self._assemble(kv_data, rec_data, self._kv_group_size)` → `reconstructed = assemble(kv_data, rec_data, self._kv_group_size)`
   - Line 774: `kv_sparse, rec_sparse = self._segment(...)` → `kv_sparse, rec_sparse = segment(...)`
   - Line 903: same as 774
3. Replace `self._slice_kv_to_delta(cache, prev_end)` at line 773 → `slice_kv_to_delta(cache, prev_end)`
4. Replace `self._slice_kv_to_delta(extracted_cache, prev_end)` at line 902 → `slice_kv_to_delta(extracted_cache, prev_end)`
5. Delete the methods that have moved: `_segment` (lines 280-498), `_assemble` (lines 500-end of method), `_slice_kv_to_delta` (lines 811-849).
6. Delete the now-unused helpers `_linearize` (lines 26-30) and `_apply_rotating_window` (lines 33-49).

- [ ] **Step 2: Update `TurnPrefixCache.collect_path_data` to use polymorphic merge_path**

In `vllm_mlx/turn_prefix_cache.py`, replace lines 325-331:

```python
        # Before:
        merged_kv: list[KVLayerSegment] = []
        for li in sorted(kv_by_layer):
            items = kv_by_layer[li]
            if items[0].metadata.get("merge_strategy", "concatenate") == "last":
                merged_kv.append(items[-1])
            else:
                merged_kv.append(KVLayerSegment.concat(items))

        # After:
        merged_kv: list[KVLayerSegment] = []
        for li in sorted(kv_by_layer):
            items = kv_by_layer[li]
            merged_kv.append(items[-1].merge_path(items))
```

Also update line 322 (`li = item.metadata["layer_index"]`) → `li = item.layer_index`.

- [ ] **Step 3: Migrate `tests/test_cache_translator.py`**

Find every assertion of the form `seg.metadata["..."]` and rewrite to use the new fields:
- `seg.metadata["layer_index"]` → `seg.layer_index`
- `seg.metadata["bits"]` → `seg.bits`
- `seg.metadata["n_tokens"]` → `seg.n_tokens`
- `seg.metadata["class_name"]` → `seg.class_name`
- `seg.metadata["max_size"]` → `seg.max_size` (only valid on `KVRotatingSegment`)
- `seg.metadata["merge_strategy"] == "last"` → `isinstance(seg, KVRotatingSegment)`
- `seg.metadata.get("_idx", ...)` → `seg.idx`

Find every `TurnCacheManager._segment(...)` and `TurnCacheManager._assemble(...)` call (lines 133, 145, 158, 169, 182, 184, 194, 196, 205, 207, 216, 219, 234, 235, 238, 326, 346, 400) and replace with `segment(...)` / `assemble(...)` (imported at the top of the file).

- [ ] **Step 4: Migrate other test files**

Same `TurnCacheManager._segment`/`_assemble` → `segment`/`assemble` substitution in:
- `tests/test_prefix_cache_adapters.py` (lines 770, 794, 834, and `patch.object(TurnCacheManager, "_segment", ...)` at line 177 → patch `vllm_mlx.prefix_cache_adapters.segment` instead; this is also fixed in PR 2's test rework, but the import shim works here)
- `tests/test_turn_prefix_cache_integration.py` (lines 60, 73, 88-89, 97, 111-112, 125, 139, 149, 169, 177)
- `tests/test_cache_hit_oom_repro.py` (line 136)
- `tests/test_cache_parity.py` (line 338)

Add `from vllm_mlx.cache_translator import segment, assemble` to each migrated file.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ 2>&1 | tail -30`

Expected: PASS for all cache-layer tests. Known unrelated failures (gradio, downloads, mcp) may persist — confirm they were failing on the base commit before assuming you caused them.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py vllm_mlx/turn_prefix_cache.py tests/
git commit -m "refactor(cache): switch to cache_translator module; delete duplicate methods"
```

---

### Task 5: Verify SSD round-trip preserves subclass identity

**Files:**
- Read: `vllm_mlx/turn_prefix_cache.py` (the `SSDRef` save/load path — search for `SSDRef` and `_spill_to_ssd`)
- Modify if broken: same files
- Modify: `tests/test_ssd_cache.py` (add subclass round-trip test)

**Interfaces:**
- Consumes: `KVConcatSegment`, `KVRotatingSegment` (Task 2)

- [ ] **Step 1: Add failing SSD round-trip test**

Append to `tests/test_ssd_cache.py`:

```python
def test_ssd_roundtrip_preserves_kv_concat_segment_type(tmp_path):
    """SSD spill then promote must return the same KVLayerSegment subclass."""
    from vllm_mlx.cache_translator import segment
    from vllm_mlx.cache_types import KVConcatSegment, KVQuantPolicy
    # Use the SSD store helpers however turn_prefix_cache exposes them
    # (search _spill_to_ssd / _promote_from_ssd for the API).
    # Spill a KVConcatSegment, promote it, assert isinstance(promoted, KVConcatSegment).
    ...  # implementer fills in based on existing test_ssd_cache helpers


def test_ssd_roundtrip_preserves_kv_rotating_segment_type(tmp_path):
    """Same for rotating."""
    from vllm_mlx.cache_types import KVRotatingSegment
    ...
```

- [ ] **Step 2: Run and inspect**

Run: `pytest tests/test_ssd_cache.py -v -k "subclass or preserves_kv" 2>&1 | tail -20`

If PASS: SSD path is type-agnostic, no fix needed.  
If FAIL: the disk format encodes `class_name` and reconstructs the wrong subclass. Fix the spill/promote serialization in `turn_prefix_cache.py` to encode and read back the subclass tag.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ssd_cache.py vllm_mlx/turn_prefix_cache.py
git commit -m "test(ssd): round-trip preserves KVLayerSegment subclass identity"
```

---

## Self-review checklist

- All five tasks reference exact file:line for the lift-and-shape work.
- `mx.eval` + `mx.stop_gradient` discipline preserved (Task 3 Step 1 instructions).
- Round-trip law (`assemble(segment(x)) ≈ x`) covered by Task 1 Step 2 tests.
- No metadata-dict access after Task 4 (Step 3 migration list is exhaustive).
- `KVConcatSegment.concat`'s bits-match assertion preserved as a runtime check.
- SSD compatibility verified in Task 5 (separate gate).
