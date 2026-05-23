# TurnPrefixCache KV-Only Support + Legacy Insert Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `TurnPrefixCache` work with KV-only models (Gemma4, Qwen3) by auto-detecting model type from inserted data, and remove the unused legacy insert path.

**Architecture:** A single `has_recurrent_state: bool` flag on `TurnPrefixCache` starts `False`, is set `True` on the first insert or load that contains recurrent state, and is never reset. `find_checkpoint_ancestor` branches on this flag: KV-only mode returns the deepest node with non-empty `kv_arrays`; hybrid mode is unchanged. The legacy `_insert_legacy` method and its dispatch branch are deleted since no production code uses them.

**Tech Stack:** Python, MLX, pytest. All changes are in `vllm_mlx/turn_prefix_cache.py` and `tests/test_turn_prefix_cache.py`.

---

## File Map

| File | What changes |
|------|-------------|
| `vllm_mlx/turn_prefix_cache.py` | Add `has_recurrent_state` to `__init__`; set in `_insert_node` and `load`; update `find_checkpoint_ancestor`; remove assert in `_retrieve_full_cache`; delete `_insert_legacy` and its dispatch branch in `insert` |
| `tests/test_turn_prefix_cache.py` | Add tests for `has_recurrent_state` flag and KV-only ancestor lookup; convert 3 legacy `insert(segments, extracted)` calls to new API |

---

## Task 1: Add `has_recurrent_state` flag and set it in `_insert_node`

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write failing tests**

Add after the existing `make_cache` helper (around line 154) in `tests/test_turn_prefix_cache.py`:

```python
# --- has_recurrent_state flag ---

def test_has_recurrent_state_starts_false():
    cache = make_cache()
    assert not cache.has_recurrent_state


def test_has_recurrent_state_not_set_for_kv_only_insert():
    cache = make_cache(stride=0)
    kv = _make_kv()
    cache.insert(cache.root, seg([1, 2, 3]), kv, None, None, is_system_prompt=True)
    assert not cache.has_recurrent_state


def test_has_recurrent_state_set_on_hybrid_insert():
    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    cache.insert(cache.root, seg([1, 2, 3]), [], None, state, is_system_prompt=True)
    assert cache.has_recurrent_state


def test_has_recurrent_state_survives_clear():
    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    cache.insert(cache.root, seg([1]), [], None, state, is_system_prompt=True)
    assert cache.has_recurrent_state
    cache.clear()
    assert cache.has_recurrent_state
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tibo/Projects/vllm-mlx/.worktrees/improve-prefill-speed
python -m pytest tests/test_turn_prefix_cache.py::test_has_recurrent_state_starts_false tests/test_turn_prefix_cache.py::test_has_recurrent_state_not_set_for_kv_only_insert tests/test_turn_prefix_cache.py::test_has_recurrent_state_set_on_hybrid_insert tests/test_turn_prefix_cache.py::test_has_recurrent_state_survives_clear -v
```

Expected: FAIL with `AttributeError: 'TurnPrefixCache' object has no attribute 'has_recurrent_state'`

- [ ] **Step 3: Add flag to `__init__` and set it in `_insert_node`**

In `vllm_mlx/turn_prefix_cache.py`, in `TurnPrefixCache.__init__` (around line 309), add after `self._on_promote`:

```python
        self.has_recurrent_state: bool = False
```

In `_insert_node` (around line 531), add inside the lock, just before the `h = _context_hash(...)` line:

```python
            if recurrent_state is not None:
                self.has_recurrent_state = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_has_recurrent_state_starts_false tests/test_turn_prefix_cache.py::test_has_recurrent_state_not_set_for_kv_only_insert tests/test_turn_prefix_cache.py::test_has_recurrent_state_set_on_hybrid_insert tests/test_turn_prefix_cache.py::test_has_recurrent_state_survives_clear -v
```

Expected: PASS (all 4)

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x -q
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: add has_recurrent_state flag to TurnPrefixCache, set on hybrid insert"
```

---

## Task 2: Set `has_recurrent_state` in `load()`

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write failing tests**

Add after the Task 1 tests:

```python
def test_load_sets_has_recurrent_state_for_hybrid_cache(tmp_path):
    # Build a cache with recurrent state and save it.
    cache1 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    state = mx.zeros((2, 3))
    cache1.insert(cache1.root, seg([1, 2, 3], role="system"), [], None, state, is_system_prompt=True)
    cache1.save(str(tmp_path))

    # Load into a fresh cache — flag must be set.
    cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    assert not cache2.has_recurrent_state
    cache2.load(str(tmp_path))
    assert cache2.has_recurrent_state


def test_load_leaves_has_recurrent_state_false_for_kv_only_cache(tmp_path):
    cache1 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    cache1.insert(cache1.root, seg([1, 2, 3], role="system"), kv, None, None, is_system_prompt=True)
    cache1.save(str(tmp_path))

    cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache2.load(str(tmp_path))
    assert not cache2.has_recurrent_state
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_load_sets_has_recurrent_state_for_hybrid_cache tests/test_turn_prefix_cache.py::test_load_leaves_has_recurrent_state_false_for_kv_only_cache -v
```

Expected: FAIL — `cache2.has_recurrent_state` is `False` even after loading hybrid nodes.

- [ ] **Step 3: Add flag scan to `load()`**

In `vllm_mlx/turn_prefix_cache.py`, in `load()`, after the `# Link parent→child INSIDE lock` block (around line 1021), add a scan before the method returns:

```python
        # Auto-detect model type from loaded nodes.
        for node in hash_to_node.values():
            if node is self.root:
                continue
            state = node.recurrent_state
            if (
                state is not None
                and not isinstance(state, SSDRef)
                and isinstance(state, (list, tuple))
                and len(state) > 0
            ):
                self.has_recurrent_state = True
                break
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_load_sets_has_recurrent_state_for_hybrid_cache tests/test_turn_prefix_cache.py::test_load_leaves_has_recurrent_state_false_for_kv_only_cache -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: set has_recurrent_state flag when loading hybrid nodes from disk"
```

---

## Task 3: Update `find_checkpoint_ancestor` for KV-only models

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write failing test**

Add after the Task 2 tests:

```python
def test_find_checkpoint_ancestor_kv_only_returns_deepest_kv_node():
    """In KV-only mode, find_checkpoint_ancestor returns the deepest node with kv_arrays."""
    cache = make_cache(stride=0)
    kv = _make_kv()
    n1 = cache.insert(cache.root, seg([1], role="system"), kv, None, None, is_system_prompt=True)
    n2 = cache.insert(n1, seg([2]), kv, None, None)
    assert not cache.has_recurrent_state  # confirm KV-only mode

    path, _ = cache.match([seg([1], role="system"), seg([2])])
    ancestor = cache.find_checkpoint_ancestor(path)
    cache.release(path)

    assert ancestor is n2


def test_find_checkpoint_ancestor_kv_only_returns_none_when_no_kv():
    """In KV-only mode, returns None if no node in path has kv_arrays."""
    cache = make_cache(stride=0)
    n1 = cache.insert(cache.root, seg([1], role="system"), [], None, None, is_system_prompt=True)
    assert not cache.has_recurrent_state

    path, _ = cache.match([seg([1], role="system")])
    ancestor = cache.find_checkpoint_ancestor(path)
    cache.release(path)

    assert ancestor is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_find_checkpoint_ancestor_kv_only_returns_deepest_kv_node tests/test_turn_prefix_cache.py::test_find_checkpoint_ancestor_kv_only_returns_none_when_no_kv -v
```

Expected: FAIL — `ancestor` is `None` because the hybrid path requires `recurrent_state`.

- [ ] **Step 3: Add KV-only branch to `find_checkpoint_ancestor`**

In `vllm_mlx/turn_prefix_cache.py`, replace the `find_checkpoint_ancestor` method (around line 675) with:

```python
    def find_checkpoint_ancestor(self, path: list[TurnNode]) -> TurnNode | None:
        """Return the deepest node in path that can serve as a prefill resume point.

        Hybrid models: deepest node with real recurrent state.
        KV-only models: deepest node with non-empty, in-memory kv_arrays.
        """
        if not self.has_recurrent_state:
            for node in reversed(path):
                if node.kv_arrays and not isinstance(node.kv_arrays, SSDRef):
                    return node
            return None

        def _has_real_recurrent(node: TurnNode) -> bool:
            return (
                node.recurrent_state is not None
                and not isinstance(node.recurrent_state, SSDRef)
                and len(node.recurrent_state) > 0
            )

        for node in reversed(path):
            if _has_real_recurrent(node):
                return node
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_find_checkpoint_ancestor_kv_only_returns_deepest_kv_node tests/test_turn_prefix_cache.py::test_find_checkpoint_ancestor_kv_only_returns_none_when_no_kv -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x -q
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: find_checkpoint_ancestor returns deepest KV node for KV-only models"
```

---

## Task 4: Remove assert in `_retrieve_full_cache` and add KV-only end-to-end test

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write failing test**

Add after the Task 3 tests:

```python
def test_retrieve_full_cache_kv_only_does_not_crash():
    """_retrieve_full_cache must work when node.recurrent_state is None (KV-only model)."""
    from vllm_mlx.kv_cache import reconstruct_cache_from_states
    from mlx_lm.models.cache import QuantizedKVCache

    trie = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    ext = _make_bf16_kvcache_extracted(n_layers=2, n_tokens=10)
    kv, recur = trie._split_cache_arrays(ext, 0)
    assert recur == []  # confirms this is KV-only data

    node = trie.insert(
        trie.root,
        Segment(role="system", token_ids=list(range(10))),
        kv, None, None,
        is_system_prompt=True,
    )
    assert not trie.has_recurrent_state

    raw_state = trie._retrieve_full_cache(node)
    assert raw_state is not None

    prompt_cache = reconstruct_cache_from_states(raw_state)
    assert prompt_cache is not None
    for layer in prompt_cache:
        assert isinstance(layer, QuantizedKVCache)
        assert layer.offset == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_retrieve_full_cache_kv_only_does_not_crash -v
```

Expected: FAIL with `AssertionError` at `assert node.recurrent_state is not None` (line 456 of `turn_prefix_cache.py`).

- [ ] **Step 3: Remove the assert from `_retrieve_full_cache`**

In `vllm_mlx/turn_prefix_cache.py`, in `_retrieve_full_cache` (around line 452), remove the line:

```python
            assert node.recurrent_state is not None
```

Leave everything else in the method unchanged. For KV-only nodes `node.recurrent_state` is `None`, so `recurrent = node.recurrent_state` will be `None`. Update the line that sets `recurrent`:

The current line (around 477) reads:
```python
            recurrent = node.recurrent_state
```

Change it to pass an empty list when `recurrent_state` is `None` (which is what `_reassemble_cache_fn` expects for a KV-only model):

```python
            recurrent = node.recurrent_state if node.recurrent_state is not None else []
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_retrieve_full_cache_kv_only_does_not_crash -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: _retrieve_full_cache supports KV-only nodes (no recurrent state)"
```

---

## Task 5: Delete `_insert_legacy` and its dispatch branch

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Convert the 3 legacy test calls to the new API**

In `tests/test_turn_prefix_cache.py`, update `test_retrieve_full_cache_produces_quantized_kvcache` (around line 1999):

```python
def test_retrieve_full_cache_produces_quantized_kvcache():
    """Full path: insert with bf16 → _retrieve_full_cache → reconstruct → QuantizedKVCache."""
    from mlx_lm.models.cache import QuantizedKVCache
    from vllm_mlx.kv_cache import reconstruct_cache_from_states

    trie = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sys_extracted = _make_bf16_kvcache_extracted(n_layers=2, n_tokens=10)
    kv, recur = trie._split_cache_arrays(sys_extracted, 0)
    sys_node = trie.insert(
        trie.root,
        Segment(role="system", token_ids=list(range(10))),
        kv, None, recur,
        is_system_prompt=True,
    )

    raw_state = trie._retrieve_full_cache(sys_node)
    assert raw_state is not None

    prompt_cache = reconstruct_cache_from_states(raw_state)
    assert prompt_cache is not None
    for layer in prompt_cache:
        assert isinstance(layer, QuantizedKVCache), f"Expected QuantizedKVCache, got {type(layer)}"
        assert layer.offset == 10
```

Update `test_retrieve_full_cache_concatenates_two_nodes` (around line 2019):

```python
def test_retrieve_full_cache_concatenates_two_nodes():
    """_retrieve_full_cache across two nodes concatenates quantized arrays correctly."""
    from mlx_lm.models.cache import QuantizedKVCache

    trie = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    # Node 1: system, 10 tokens
    ext1 = _make_bf16_kvcache_extracted(n_layers=2, n_tokens=10)
    kv1, recur1 = trie._split_cache_arrays(ext1, 0)
    node1 = trie.insert(
        trie.root,
        Segment(role="system", token_ids=list(range(10))),
        kv1, None, recur1,
        is_system_prompt=True,
    )
    # Node 2: user, 5 more tokens (full cache has 15 tokens, offset starts at 10)
    ext2 = _make_bf16_kvcache_extracted(n_layers=2, n_tokens=15)
    kv2, recur2 = trie._split_cache_arrays(ext2, node1.n_tokens)
    node2 = trie.insert(
        node1,
        Segment(role="user", token_ids=list(range(10, 15))),
        kv2, None, recur2,
    )

    raw_state = trie._retrieve_full_cache(node2)
    from vllm_mlx.kv_cache import reconstruct_cache_from_states
    prompt_cache = reconstruct_cache_from_states(raw_state)

    for layer in prompt_cache:
        assert isinstance(layer, QuantizedKVCache)
        assert layer.offset == 15  # 10 + 5 tokens
```

- [ ] **Step 2: Run the two converted tests to confirm they pass before deleting anything**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_retrieve_full_cache_produces_quantized_kvcache tests/test_turn_prefix_cache.py::test_retrieve_full_cache_concatenates_two_nodes -v
```

Expected: PASS (confirming the new API works for these tests).

- [ ] **Step 3: Delete `_insert_legacy` and its dispatch branch**

In `vllm_mlx/turn_prefix_cache.py`, in the `insert()` method (around line 500), remove the entire legacy dispatch block:

```python
        if isinstance(parent_or_segments, list):
            # Legacy API: insert(segments, extracted_cache, acquire_lock=True)
            actual_lock = kv_arrays if isinstance(kv_arrays, bool) else True
            return self._insert_legacy(
                parent_or_segments, segment_or_extracted, acquire_lock=actual_lock
            )
```

Also delete the entire `_insert_legacy` method (approximately lines 584–640).

- [ ] **Step 4: Run the full test suite to confirm nothing broke**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "refactor: delete _insert_legacy and legacy insert dispatch branch"
```

---

## Final check

- [ ] **Run the complete test suite one final time**

```bash
python -m pytest tests/ -x -q
```

Expected: all tests pass.
