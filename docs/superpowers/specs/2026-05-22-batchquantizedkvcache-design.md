# BatchQuantizedKVCache Alignment with mlx-lm

**Date:** 2026-05-22  
**Branch:** refactor-scheduler-and-cache  
**Scope:** Approach A — targeted alignment, no structural refactor

## Problem

`BatchQuantizedKVCache` fills a genuine gap in mlx-lm (no batched quantized KV cache
upstream), but its alignment with mlx-lm conventions has drifted:

- No `_BaseCache` inheritance → missing `from_state`, `is_trimmable`, `trim`
- `right_padding` in `prepare()` silently no-ops → correctness risk for callers
- `QuantizedKVCache.merge` added via monkey-patch → fragile if mlx-lm adds it upstream
- `type(c) is KVCache` in `make_quantized_cache` → misses subclasses

## Design

### 1. `VllmQuantizedKVCache` — thin single-sequence subclass

New class in `batch_quantized_kv_cache.py`, mirrors the mlx-lm pattern:

```
KVCache.merge           → BatchKVCache
RotatingKVCache.merge   → BatchRotatingKVCache
VllmQuantizedKVCache.merge → BatchQuantizedKVCache   (ours)
```

```python
class VllmQuantizedKVCache(QuantizedKVCache):
    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)
```

- `BatchQuantizedKVCache.extract()` returns `VllmQuantizedKVCache` instead of `QuantizedKVCache`
- Monkey-patch block at module bottom is deleted entirely
- Everything else inherited from `QuantizedKVCache`

### 2. `BatchQuantizedKVCache(_BaseCache)` — inheritance + missing methods

```python
class BatchQuantizedKVCache(_BaseCache):
```

Add `is_trimmable` and `trim` matching `BatchKVCache`:

```python
def is_trimmable(self):
    return True

def trim(self, n):
    n = min(self._idx, n)
    self._idx -= n
    self.offset -= n
    return n
```

Add `state.setter` and `meta_state.setter` to satisfy `_BaseCache`'s protocol.
Note: `from_state` is intentionally **not** supported for `BatchQuantizedKVCache` —
batch caches are never round-tripped through `save_prompt_cache`/`load_prompt_cache`
(only single-sequence caches are). The setters raise to make this explicit:

```python
@state.setter
def state(self, v):
    raise NotImplementedError("BatchQuantizedKVCache does not support from_state")

@meta_state.setter
def meta_state(self, v):
    raise NotImplementedError("BatchQuantizedKVCache does not support from_state")
```

`nbytes` and `empty` are already implemented — `_BaseCache` stubs satisfied.

### 3. Right-padding in `prepare()` / `finalize()`

`__init__`: add `self._right_padding = None`

`prepare()`: store right padding, same as `BatchKVCache`:

```python
if right_padding is not None and max(right_padding) > 0:
    self._right_padding = mx.array(right_padding)
```

`finalize()`: apply `dynamic_roll` along axis=2 (token axis) to each component
of the `QuantizedArray`. Rolling tokens does not affect the quantization encoding
within each token position.

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

New import: `dynamic_roll` from `mlx_lm.models.cache`.  
Delete comment: `# right_padding unsupported for quantized caches`.

### 4. `make_quantized_cache` isinstance fix

```python
# before
if type(c) is KVCache:
# after
if isinstance(c, KVCache):
```

### 5. AGENTS.md — mlx-lm section

Already written and committed. See `AGENTS.md § mlx-lm Cache Primitives`.

## Files Changed

| File | Change |
|------|--------|
| `vllm_mlx/batch_quantized_kv_cache.py` | Add `VllmQuantizedKVCache`; `_BaseCache` inheritance; `is_trimmable`, `trim`, `state.setter`, `meta_state.setter`; right-padding in `prepare`/`finalize`; `isinstance` fix; delete monkey-patch |
| `AGENTS.md` | Add mlx-lm cache primitives section |

## What This Does NOT Do

- Does not eliminate duplication between `BatchQuantizedKVCache` and `BatchKVCache`
  batch management methods (`filter`, `extend`, `merge`, `extract`). The duplication
  is structural — quantized arrays have a different shape contract and can't share
  the same storage path. This is acceptable until mlx-lm adds `BatchQuantizedKVCache`
  upstream.
- Does not change `BatchQuantizedKVCache.merge` logic.
- Does not rename `QuantizedArray`.
