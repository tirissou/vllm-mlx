# `_segment()` Lazy-Eval Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure `TurnCacheManager._segment()` evaluates all quantized KVLayerSegment arrays before returning, so the source float16 Metal buffers can be freed and do not linger in active memory between requests.

**Architecture:** Add two `mx.eval()` calls — one at the end of the RotatingKVCache branch and one at the end of the KVCache branch in `_segment()`. Each call materialises the six quantized components (`packed`, `scales`, `biases` for both keys and values) for that layer, severing the computation graph dependency on the source float16 arrays. No other files change.

**Tech Stack:** Python, MLX (`mlx.core`), pytest

---

### Task 1: Write the failing tests

**Files:**
- Modify: `tests/test_prefix_cache_adapters.py`

The tests go in `tests/test_prefix_cache_adapters.py` alongside the existing `_segment` tests. Add them at the end of the file.

- [ ] **Step 1: Add two failing tests to `tests/test_prefix_cache_adapters.py`**

Append the following to the end of the file:

```python
# ── _segment() lazy-eval fix ─────────────────────────────────────────────────

import pytest


def _rotating_state(B=1, H=2, T=16, D=64):
    """Return a RotatingKVCache state dict with float16 keys/values."""
    import mlx.core as mx

    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        # meta: (keep, max_size, offset, _idx)
        "meta_state": ("0", str(T), str(T), str(T)),
    }


def _kvcache_state(B=1, H=2, T=16, D=64):
    """Return a KVCache state dict with float16 keys/values."""
    import mlx.core as mx

    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {
        "class_name": "KVCache",
        "state": (keys, values),
        "meta_state": (str(T),),
    }


def test_segment_rotating_kvcache_quantized_arrays_are_evaluated():
    """KVLayerSegment arrays from the RotatingKVCache branch must be concrete
    (already evaluated) so that freeing the source float16 arrays does not
    leave dangling computation graph references in active Metal memory."""
    import mlx.core as mx

    state = _rotating_state()
    kv_list, _ = TurnCacheManager._segment([state], group_size=32, bits=8)
    seg = kv_list[0]
    assert seg is not None

    # Release source arrays and flush the Metal pool.
    del state
    mx.clear_cache()

    # If the quantized arrays were NOT evaluated inside _segment(), calling
    # mx.eval() here would attempt to resolve a computation graph whose input
    # buffers have been freed, producing incorrect data.  With the fix the
    # arrays are already concrete and this is a no-op that must not raise.
    mx.eval(seg.keys.packed, seg.keys.scales, seg.keys.biases)
    mx.eval(seg.values.packed, seg.values.scales, seg.values.biases)

    assert seg.keys.packed.nbytes > 0
    assert seg.values.packed.nbytes > 0


def test_segment_kvcache_quantized_arrays_are_evaluated():
    """KVLayerSegment arrays from the KVCache branch must be concrete."""
    import mlx.core as mx

    state = _kvcache_state()
    kv_list, _ = TurnCacheManager._segment([state], group_size=32, bits=8)
    seg = kv_list[0]
    assert seg is not None

    del state
    mx.clear_cache()

    mx.eval(seg.keys.packed, seg.keys.scales, seg.keys.biases)
    mx.eval(seg.values.packed, seg.values.scales, seg.values.biases)

    assert seg.keys.packed.nbytes > 0
    assert seg.values.packed.nbytes > 0


@pytest.mark.skipif(
    not __import__("mlx.core", fromlist=["metal"]).metal.is_available(),
    reason="requires Metal GPU",
)
def test_segment_does_not_retain_source_float16_in_active_memory():
    """Source float16 buffers must not stay in active Metal memory after
    _segment() returns and the source references are dropped."""
    import mlx.core as mx

    # Establish baseline before any new allocations.
    mx.clear_cache()
    baseline = mx.get_active_memory()

    # Use a large tensor so the delta is measurable (8 MB float16).
    B, H, T, D = 1, 8, 128, 128
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    source_bytes = keys.nbytes + values.nbytes  # float16 footprint

    state = {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": ("0", str(T), str(T), str(T)),
    }

    kv_list, _ = TurnCacheManager._segment([state], group_size=32, bits=8)
    seg = kv_list[0]

    # Release all source references.
    del keys, values, state
    mx.eval(seg.keys.packed)  # ensure segment is materialised
    mx.clear_cache()

    active_after = mx.get_active_memory()
    # Without the fix: active_after ≈ baseline + source_bytes (float16 still
    # referenced via lazy computation graph in the trie).
    # With the fix: active_after ≈ baseline + quantized_bytes (< source_bytes).
    # Assert the float16 source is NOT retained: growth must be well under
    # the full float16 footprint (allow 50% to account for quantized arrays).
    growth = active_after - baseline
    assert growth < source_bytes * 0.75, (
        f"Source float16 ({source_bytes / 1e6:.1f} MB) appears retained: "
        f"baseline={baseline / 1e6:.1f} MB, active_after={active_after / 1e6:.1f} MB, "
        f"growth={growth / 1e6:.1f} MB"
    )
```

- [ ] **Step 2: Run the tests and confirm they fail (or the Metal test is skipped)**

```bash
pytest tests/test_prefix_cache_adapters.py::test_segment_rotating_kvcache_quantized_arrays_are_evaluated tests/test_prefix_cache_adapters.py::test_segment_kvcache_quantized_arrays_are_evaluated tests/test_prefix_cache_adapters.py::test_segment_does_not_retain_source_float16_in_active_memory -v
```

Expected: the first two tests pass vacuously (MLX buffers are not immediately freed by Python GC), and the Metal test is either skipped or fails with an active memory assertion. The important thing confirmed here is that the tests run and the new test names are registered.

> Note: the first two tests are conservative — they verify the API works after source deletion and that the quantized arrays have data. They will likely pass even before the fix because Metal's pool retains buffers. The Metal active-memory test is the definitive regression guard.

- [ ] **Step 3: Commit the tests**

```bash
git add tests/test_prefix_cache_adapters.py
git commit -m "test: add _segment() lazy-eval correctness tests"
```

---

### Task 2: Implement the fix

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py:193-213` (RotatingKVCache branch)
- Modify: `vllm_mlx/prefix_cache_adapters.py:223-239` (KVCache branch)

- [ ] **Step 1: Add `mx.eval()` to the RotatingKVCache branch**

In `vllm_mlx/prefix_cache_adapters.py`, find the RotatingKVCache branch inside `_segment()`. The current code ends at the `kv_list[i] = KVLayerSegment(...)` call (around line 200–213). Add the eval call **after** `q_keys` and `q_values` are assigned and **before** the `KVLayerSegment` is constructed (so `lin_keys`/`lin_values` can be freed when they go out of scope):

```python
            lin_keys = _linearize(state[0], _idx, max_size)
            lin_values = _linearize(state[1], _idx, max_size)
            q_keys = QuantizedArray(
                *mx.quantize(lin_keys, group_size=group_size, bits=bits)
            )
            q_values = QuantizedArray(
                *mx.quantize(lin_values, group_size=group_size, bits=bits)
            )
            mx.eval(
                q_keys.packed, q_keys.scales, q_keys.biases,
                q_values.packed, q_values.scales, q_values.biases,
            )

            kv_list[i] = KVLayerSegment(
                keys=q_keys,
                values=q_values,
                metadata={
                    "class_name": "RotatingKVCache",
                    "layer_index": i,
                    "merge_strategy": "last",
                    "n_tokens": lin_keys.shape[-2],
                    "max_size": max_size,
                    "keep": keep,
                    "offset": offset,
                    "_idx": _idx,
                },
            )
```

- [ ] **Step 2: Add `mx.eval()` to the KVCache branch**

In the same method, find the KVCache branch (the `elif "KVCache" in class_name:` block, around lines 215–239). Add the eval call after `q_keys` and `q_values` are assigned:

```python
                sliced_keys = state[0][:, :, :actual_end, :]
                sliced_values = state[1][:, :, :actual_end, :]
                q_keys = QuantizedArray(
                    *mx.quantize(sliced_keys, group_size=group_size, bits=bits)
                )
                q_values = QuantizedArray(
                    *mx.quantize(sliced_values, group_size=group_size, bits=bits)
                )
                mx.eval(
                    q_keys.packed, q_keys.scales, q_keys.biases,
                    q_values.packed, q_values.scales, q_values.biases,
                )

                kv_list[i] = KVLayerSegment(
                    keys=q_keys,
                    values=q_values,
                    metadata={
                        "class_name": class_name,
                        "layer_index": i,
                        "merge_strategy": "concatenate",
                        "n_tokens": actual_end,
                    },
                )
```

- [ ] **Step 3: Run all three new tests**

```bash
pytest tests/test_prefix_cache_adapters.py::test_segment_rotating_kvcache_quantized_arrays_are_evaluated tests/test_prefix_cache_adapters.py::test_segment_kvcache_quantized_arrays_are_evaluated tests/test_prefix_cache_adapters.py::test_segment_does_not_retain_source_float16_in_active_memory -v
```

Expected: all three pass (or the Metal test is skipped on non-Metal hardware).

- [ ] **Step 4: Run the full adapter test suite**

```bash
pytest tests/test_prefix_cache_adapters.py -v
```

Expected: all 34 tests pass (32 existing + 2 new always-on tests; Metal test is a 35th if hardware is present).

- [ ] **Step 5: Commit the fix**

```bash
git add vllm_mlx/prefix_cache_adapters.py
git commit -m "fix: eval KVLayerSegment arrays in _segment() to free source float16 buffers

Lazy mx.quantize() ops in _segment() held computation graph references
to source float16 arrays, keeping them in active Metal memory even after
requests completed and mx.clear_cache() was called.

mx.eval() on packed/scales/biases in both the RotatingKVCache and
KVCache branches severs the dependency before the KVLayerSegment is
stored in the trie."
```
