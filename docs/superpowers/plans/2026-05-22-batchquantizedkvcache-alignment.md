# BatchQuantizedKVCache mlx-lm Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `BatchQuantizedKVCache` with mlx-lm conventions by subclassing `_BaseCache`, introducing `VllmQuantizedKVCache` to replace the monkey-patch, adding `is_trimmable`/`trim`, right-padding support, and an `isinstance` fix.

**Architecture:** Five targeted changes to `vllm_mlx/batch_quantized_kv_cache.py`, each tested in TDD order. `VllmQuantizedKVCache(QuantizedKVCache)` is a thin subclass with a `merge` classmethod, eliminating the module-level monkey-patch. `BatchQuantizedKVCache` gains `_BaseCache` inheritance and the missing methods that `BatchKVCache` has had all along.

**Tech Stack:** MLX, mlx-lm (`_BaseCache`, `QuantizedKVCache`, `dynamic_roll`), pytest

**Spec:** `docs/superpowers/specs/2026-05-22-batchquantizedkvcache-design.md`

---

## File Map

| File | Change |
|------|--------|
| `vllm_mlx/batch_quantized_kv_cache.py` | All implementation changes |
| `tests/test_kv_cache.py` | New `TestBatchQuantizedKVCacheAlignment` class |
| `tests/test_kv_cache_quantization.py` | One new test in `TestMakeQuantizedCache` |

---

### Task 1: `VllmQuantizedKVCache` — thin subclass with `merge`

Introduces the `VllmQuantizedKVCache` class and changes `extract()` to return it.
Removes the monkey-patch block. Tests verify the engine's call-site pattern works.

**Files:**
- Modify: `vllm_mlx/batch_quantized_kv_cache.py`
- Test: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_kv_cache.py` (after the existing `TestBatchQuantizedKVCacheQuantizedArray` class):

```python
class TestBatchQuantizedKVCacheAlignment:
    """Alignment with mlx-lm _BaseCache conventions."""

    def _make_cache(self, B=2, H=4, T=16, D=64):
        cache = BatchQuantizedKVCache(left_padding=[0] * B)
        keys = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
        values = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)
        return cache

    def test_extract_returns_vllm_quantized_kv_cache(self):
        from vllm_mlx.batch_quantized_kv_cache import VllmQuantizedKVCache
        cache = self._make_cache(B=2)
        extracted = cache.extract(0)
        assert isinstance(extracted, VllmQuantizedKVCache)

    def test_vllm_quantized_kv_cache_merge_returns_batch_quantized(self):
        from vllm_mlx.batch_quantized_kv_cache import VllmQuantizedKVCache
        cache = self._make_cache(B=2, T=16)
        e0, e1 = cache.extract(0), cache.extract(1)
        merged = VllmQuantizedKVCache.merge([e0, e1])
        assert isinstance(merged, BatchQuantizedKVCache)
        assert merged._idx == 16
        assert merged.keys.packed.shape[0] == 2

    def test_polymorphic_merge_matches_engine_call_site(self):
        """Engine calls extracted[0].merge(extracted) — must work via VllmQuantizedKVCache."""
        cache = self._make_cache(B=3)
        extracted = [cache.extract(i) for i in range(3)]
        merged = extracted[0].merge(extracted)
        assert isinstance(merged, BatchQuantizedKVCache)
        assert merged.keys.packed.shape[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tibo/Projects/vllm-mlx/.worktrees/refactor-scheduler-and-cache
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment -v
```

Expected: `ImportError: cannot import name 'VllmQuantizedKVCache'`

- [ ] **Step 3: Add `VllmQuantizedKVCache` to `batch_quantized_kv_cache.py`**

Add this class **after** `BatchQuantizedKVCache` (it must be defined after, since its `merge` references `BatchQuantizedKVCache`):

```python
class VllmQuantizedKVCache(QuantizedKVCache):
    """Single-sequence quantized KV cache returned by BatchQuantizedKVCache.extract().

    Mirrors the mlx-lm pattern: KVCache.merge → BatchKVCache,
    RotatingKVCache.merge → BatchRotatingKVCache.
    """

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)
```

- [ ] **Step 4: Update `extract()` to return `VllmQuantizedKVCache`**

In `BatchQuantizedKVCache.extract()`, change the first line from:

```python
cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
```

to:

```python
cache = VllmQuantizedKVCache(group_size=self.group_size, bits=self.bits)
```

- [ ] **Step 5: Delete the monkey-patch block**

Remove these lines from the bottom of the file (the entire block):

```python
def _qkv_merge(_, caches):
    return BatchQuantizedKVCache.merge(caches)


if not hasattr(QuantizedKVCache, "merge"):
    QuantizedKVCache.merge = classmethod(_qkv_merge)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_extract_returns_vllm_quantized_kv_cache tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_vllm_quantized_kv_cache_merge_returns_batch_quantized tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_polymorphic_merge_matches_engine_call_site -v
```

Expected: all three PASS.

- [ ] **Step 7: Run the full test suite to check for regressions**

```bash
.venv/bin/pytest tests/test_kv_cache.py -v
```

Expected: all pass. The existing `test_extract_merge_round_trip` continues to pass because `VllmQuantizedKVCache` is a `QuantizedKVCache` subclass.

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/batch_quantized_kv_cache.py tests/test_kv_cache.py
git commit -m "feat: add VllmQuantizedKVCache, remove monkey-patch, update extract()"
```

---

### Task 2: `_BaseCache` inheritance + setter stubs

Makes `BatchQuantizedKVCache` a proper mlx-lm cache object.
Enables `isinstance(cache, _BaseCache)` and satisfies the property setter protocol.

**Files:**
- Modify: `vllm_mlx/batch_quantized_kv_cache.py`
- Test: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestBatchQuantizedKVCacheAlignment`:

```python
    def test_is_instance_of_base_cache(self):
        from mlx_lm.models.cache import _BaseCache
        cache = BatchQuantizedKVCache(left_padding=[0])
        assert isinstance(cache, _BaseCache)

    def test_state_setter_raises_not_implemented(self):
        import pytest
        cache = BatchQuantizedKVCache(left_padding=[0])
        with pytest.raises(NotImplementedError):
            cache.state = (None, None)

    def test_meta_state_setter_raises_not_implemented(self):
        import pytest
        cache = BatchQuantizedKVCache(left_padding=[0])
        with pytest.raises(NotImplementedError):
            cache.meta_state = ("0", "64", "4")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_is_instance_of_base_cache tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_state_setter_raises_not_implemented tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_meta_state_setter_raises_not_implemented -v
```

Expected: `test_is_instance_of_base_cache` FAILS (not a subclass); setter tests FAIL (no setter defined).

- [ ] **Step 3: Add `_BaseCache` to imports**

Change the import line in `batch_quantized_kv_cache.py`:

```python
from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache
```

to:

```python
from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache, _BaseCache
```

- [ ] **Step 4: Add `_BaseCache` to the class declaration**

Change:

```python
class BatchQuantizedKVCache:
```

to:

```python
class BatchQuantizedKVCache(_BaseCache):
```

- [ ] **Step 5: Add `state.setter` and `meta_state.setter`**

Directly after the existing `@state.getter` property block and `@meta_state.getter` property block in `BatchQuantizedKVCache`, add:

```python
    @state.setter
    def state(self, v):
        raise NotImplementedError("BatchQuantizedKVCache does not support from_state")

    @meta_state.setter
    def meta_state(self, v):
        raise NotImplementedError("BatchQuantizedKVCache does not support from_state")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_is_instance_of_base_cache tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_state_setter_raises_not_implemented tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_meta_state_setter_raises_not_implemented -v
```

Expected: all three PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/batch_quantized_kv_cache.py tests/test_kv_cache.py
git commit -m "feat: BatchQuantizedKVCache inherits _BaseCache, add setter stubs"
```

---

### Task 3: `is_trimmable` and `trim`

Enables `can_trim_prompt_cache` / `trim_prompt_cache` to work on batched quantized caches.
Matches the `BatchKVCache.trim` contract exactly.

**Files:**
- Modify: `vllm_mlx/batch_quantized_kv_cache.py`
- Test: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestBatchQuantizedKVCacheAlignment`:

```python
    def test_is_trimmable_returns_true(self):
        from mlx_lm.models.cache import can_trim_prompt_cache
        cache = self._make_cache(B=2)
        assert cache.is_trimmable() is True
        assert can_trim_prompt_cache([cache]) is True

    def test_trim_reduces_idx_and_offset(self):
        cache = self._make_cache(B=2, T=16)
        mx.eval(cache.offset)
        offset_before = cache.offset.tolist()
        result = cache.trim(4)
        assert result == 4
        assert cache._idx == 12
        mx.eval(cache.offset)
        assert cache.offset.tolist() == [o - 4 for o in offset_before]

    def test_trim_clamps_to_idx(self):
        cache = self._make_cache(B=1, T=8)
        result = cache.trim(100)
        assert result == 8
        assert cache._idx == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_is_trimmable_returns_true tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_trim_reduces_idx_and_offset tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_trim_clamps_to_idx -v
```

Expected: `test_is_trimmable_returns_true` FAILS (`is_trimmable` returns False from `_BaseCache` default); trim tests FAIL (`AttributeError: 'BatchQuantizedKVCache' has no attribute 'trim'`).

- [ ] **Step 3: Add `is_trimmable` and `trim` to `BatchQuantizedKVCache`**

Add these two methods to `BatchQuantizedKVCache`, after the `size()` method:

```python
    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_is_trimmable_returns_true tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_trim_reduces_idx_and_offset tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_trim_clamps_to_idx -v
```

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/batch_quantized_kv_cache.py tests/test_kv_cache.py
git commit -m "feat: add is_trimmable and trim to BatchQuantizedKVCache"
```

---

### Task 4: Right-padding in `prepare()` / `finalize()`

Closes the gap with `BatchKVCache` — right-padded inputs no longer silently produce wrong cache state.
Uses `dynamic_roll` (from mlx-lm) along the token axis on each `QuantizedArray` component.

**Files:**
- Modify: `vllm_mlx/batch_quantized_kv_cache.py`
- Test: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestBatchQuantizedKVCacheAlignment`:

```python
    def test_right_padding_finalize_adjusts_offset_and_left_padding(self):
        """After right-padded prefill + finalize, offset and left_padding reflect real tokens only."""
        B, H, T, D = 2, 4, 6, 64
        cache = BatchQuantizedKVCache(left_padding=[0, 0])
        cache.prepare(right_padding=[2, 0])

        keys = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
        values = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values, cache.offset, cache.left_padding)

        offset_before = cache.offset.tolist()
        lp_before = cache.left_padding.tolist()

        cache.finalize()
        mx.eval(cache.offset, cache.left_padding)

        offset_after = cache.offset.tolist()
        lp_after = cache.left_padding.tolist()

        # Seq 0: 2 right-padding tokens removed → offset shrinks, left_padding grows
        assert offset_after[0] == offset_before[0] - 2
        assert lp_after[0] == lp_before[0] + 2
        # Seq 1: no right padding → unchanged
        assert offset_after[1] == offset_before[1]
        assert lp_after[1] == lp_before[1]

    def test_finalize_without_right_padding_is_noop(self):
        cache = self._make_cache(B=2, T=10)
        mx.eval(cache.offset, cache.left_padding)
        offset_before = cache.offset.tolist()
        lp_before = cache.left_padding.tolist()

        cache.finalize()
        mx.eval(cache.offset, cache.left_padding)

        assert cache.offset.tolist() == offset_before
        assert cache.left_padding.tolist() == lp_before
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_right_padding_finalize_adjusts_offset_and_left_padding tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_finalize_without_right_padding_is_noop -v
```

Expected: `test_right_padding_finalize_adjusts_offset_and_left_padding` FAILS (offset/left_padding unchanged because `prepare(right_padding=...)` is silently ignored); noop test PASSES (finalize is already a no-op).

- [ ] **Step 3: Add `dynamic_roll` to imports**

Change:

```python
from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache, _BaseCache
```

to:

```python
from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache, _BaseCache, dynamic_roll
```

- [ ] **Step 4: Add `self._right_padding = None` to `__init__`**

In `BatchQuantizedKVCache.__init__`, add after the existing instance variable assignments:

```python
        self._right_padding = None
```

- [ ] **Step 5: Update `prepare()` to store right padding**

Replace the current `prepare` method body:

```python
    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        # right_padding unsupported for quantized caches (no dynamic_roll equivalent)
```

with:

```python
    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
```

- [ ] **Step 6: Update `finalize()` to apply `dynamic_roll`**

Replace the current `finalize` method:

```python
    def finalize(self):
        pass
```

with:

```python
    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding

            def roll_qa(qa):
                return QuantizedArray(*[
                    dynamic_roll(c, padding[:, None], axis=2) for c in qa
                ])

            self.keys = roll_qa(self.keys)
            self.values = roll_qa(self.values)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_right_padding_finalize_adjusts_offset_and_left_padding tests/test_kv_cache.py::TestBatchQuantizedKVCacheAlignment::test_finalize_without_right_padding_is_noop -v
```

Expected: both PASS.

- [ ] **Step 8: Run the full test suite**

```bash
.venv/bin/pytest tests/test_kv_cache.py tests/test_kv_cache_quantization.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add vllm_mlx/batch_quantized_kv_cache.py tests/test_kv_cache.py
git commit -m "feat: add right-padding support to BatchQuantizedKVCache prepare/finalize"
```

---

### Task 5: `isinstance` fix in `make_quantized_cache`

`type(c) is KVCache` misses subclasses. Fixing to `isinstance` ensures any `KVCache`
subclass is correctly replaced with `BatchQuantizedKVCache`.

**Files:**
- Modify: `vllm_mlx/batch_quantized_kv_cache.py`
- Test: `tests/test_kv_cache_quantization.py`

- [ ] **Step 1: Write the failing test**

Add to `TestMakeQuantizedCache` in `tests/test_kv_cache_quantization.py`:

```python
    def test_kvcache_subclass_gets_replaced(self):
        """isinstance check: a KVCache subclass must be converted to BatchQuantizedKVCache."""
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache, make_quantized_cache

        class MyKVCache(KVCache):
            pass

        class FakeModel:
            layers = [None]

            def make_cache(self):
                return [MyKVCache()]

        cache = make_quantized_cache(FakeModel(), [0], max_kv_size=None)
        assert isinstance(cache[0], BatchQuantizedKVCache), (
            f"Expected BatchQuantizedKVCache, got {type(cache[0])}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_kv_cache_quantization.py::TestMakeQuantizedCache::test_kvcache_subclass_gets_replaced -v
```

Expected: FAIL — `MyKVCache` passes through unchanged because `type(c) is KVCache` is False for subclasses.

- [ ] **Step 3: Apply the fix in `make_quantized_cache`**

In `batch_quantized_kv_cache.py`, inside `make_quantized_cache`, in the `to_quantized_batch` inner function, change:

```python
        if type(c) is KVCache:
```

to:

```python
        if isinstance(c, KVCache):
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_kv_cache_quantization.py::TestMakeQuantizedCache::test_kvcache_subclass_gets_replaced -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/batch_quantized_kv_cache.py tests/test_kv_cache_quantization.py
git commit -m "fix: use isinstance in make_quantized_cache to handle KVCache subclasses"
```

---

## Self-Review

**Spec coverage:**
- ✅ `VllmQuantizedKVCache` subclass with `merge` → Task 1
- ✅ Monkey-patch removal → Task 1 Step 5
- ✅ `extract()` returns `VllmQuantizedKVCache` → Task 1 Step 4
- ✅ `_BaseCache` inheritance → Task 2
- ✅ `state.setter` / `meta_state.setter` raise `NotImplementedError` → Task 2
- ✅ `is_trimmable` / `trim` → Task 3
- ✅ Right-padding in `prepare` / `finalize` → Task 4
- ✅ `isinstance` fix in `make_quantized_cache` → Task 5

**Placeholder scan:** clean — no TBDs, all code blocks are complete.

**Type consistency:** `VllmQuantizedKVCache` referenced consistently across Tasks 1 and later. `QuantizedArray` used throughout for quantized storage. `dynamic_roll` imported once in Task 4 and used only there.
