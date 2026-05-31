# Eviction Ref-Count Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `TurnPrefixCache` nodes from self-evicting immediately after insertion, and ensure all nodes placed in `cs.turn_path` by `on_prefill_checkpoint` are pinned for the duration of the request.

**Architecture:** Two coordinated changes. First, `_insert_node` temporarily sets `ref_count=1` on the new node before triggering eviction (preventing it from being its own victim), then resets to 0. Second, `TurnCacheManager.on_prefill_checkpoint` increments `ref_count` on the returned node before appending it to `cs.turn_path`, providing a durable pin until `release()` is called at request completion. All existing release call sites in the scheduler already iterate the full `cs.turn_path`, so no scheduler changes are needed.

**Tech Stack:** Python, MLX, `vllm_mlx.turn_prefix_cache.TurnPrefixCache`, `vllm_mlx.prefix_cache_adapters.TurnCacheManager`

---

## File Map

| File | Change |
|------|--------|
| `tests/test_turn_prefix_cache.py` | Add two new tests (Tasks 1–2) |
| `vllm_mlx/turn_prefix_cache.py` | Temporary pin in `_insert_node` (Task 4) |
| `vllm_mlx/prefix_cache_adapters.py` | Permanent pin in `on_prefill_checkpoint` (Task 5) |

---

### Task 1: Write failing test — self-eviction prevention

**Files:**
- Modify: `tests/test_turn_prefix_cache.py` (append after `test_evicted_node_removed_from_parent_children`)

- [ ] **Step 1: Add the test**

Append this function to `tests/test_turn_prefix_cache.py`:

```python
def test_insert_does_not_self_evict_when_only_evictable_candidate():
    """Node inserted when it is the sole evictable leaf must not self-evict."""
    cache = make_cache(max_gb=100.0)
    n1 = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    n2 = cache.insert(n1, seg([2]), kv_data=_make_kv_data())
    # n1 now has child n2 → is_leaf=False, not evictable
    # n2 is currently the only leaf

    # Shrink budget so inserting n3 pushes memory over the limit
    node_size = _node_data_bytes(n2)
    cache.config.max_memory_gb = (node_size * 2) / (1024**3)

    # After n3 is inserted: n2 gains a child → n2 no longer a leaf; n3 is the only evictable leaf
    n3 = cache.insert(n2, seg([3]), kv_data=_make_kv_data())

    path, _ = cache.match([seg([1]), seg([2]), seg([3])])
    cache.release(path)
    assert len(path) == 3, f"n3 self-evicted (matched {len(path)} nodes, expected 3)"
```

---

### Task 2: Write failing test — `on_prefill_checkpoint` ref_count pinning

**Files:**
- Modify: `tests/test_turn_prefix_cache.py` (append after Task 1's test)

- [ ] **Step 1: Add the test**

```python
def test_on_prefill_checkpoint_pins_inserted_node():
    """on_prefill_checkpoint must set ref_count=1 on the node it appends to turn_path."""
    from vllm_mlx.turn_prefix_cache import TurnPrefixCacheConfig, TurnPrefixCache
    from vllm_mlx.kv_cache import RequestCacheState

    inner = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0, max_memory_gb=100.0))
    manager = TurnCacheAdapter(inner)

    class FakeRequest:
        prompt_token_ids = list(range(100))
        _turn_boundaries = [50]
        _cache_state = RequestCacheState()

    req = FakeRequest()
    manager.on_prefill_checkpoint(req, 50, [])

    assert len(req._cache_state.turn_path) == 1
    node = req._cache_state.turn_path[0]
    assert node.ref_count == 1, f"Expected ref_count=1 after pin, got {node.ref_count}"
```

---

### Task 3: Run failing tests to confirm baseline

**Files:** (read-only)

- [ ] **Step 1: Run the two new tests**

```bash
cd /Users/tibo/Projects/vllm-mlx/cache-translation-layer
pytest tests/test_turn_prefix_cache.py::test_insert_does_not_self_evict_when_only_evictable_candidate tests/test_turn_prefix_cache.py::test_on_prefill_checkpoint_pins_inserted_node -v
```

Expected: **both FAIL**

- `test_insert_does_not_self_evict_when_only_evictable_candidate` — FAIL: `AssertionError: n3 self-evicted (matched 2 nodes, expected 3)`
- `test_on_prefill_checkpoint_pins_inserted_node` — FAIL: `AssertionError: Expected ref_count=1 after pin, got 0`

---

### Task 4: Implement temporary pin in `_insert_node`

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py:226-251`

- [ ] **Step 1: Apply the change**

In `_insert_node`, find the block that handles the new-node path (after `parent.children[h] = node`). Replace the three lines starting at `self._memory_bytes += ...`:

Current (`vllm_mlx/turn_prefix_cache.py:248-251`):
```python
            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            self._evict_if_needed_unlocked()
            return node
```

Replacement:
```python
            self._memory_bytes += _node_data_bytes(node)
            node.ref_count = 1  # prevent self-eviction; caller may re-use or release
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            self._evict_if_needed_unlocked()
            node.ref_count = 0
            return node
```

- [ ] **Step 2: Run the self-eviction test to confirm it now passes**

```bash
pytest tests/test_turn_prefix_cache.py::test_insert_does_not_self_evict_when_only_evictable_candidate -v
```

Expected: **PASS**

- [ ] **Step 3: Confirm no existing eviction tests regressed**

```bash
pytest tests/test_turn_prefix_cache.py -k "evict" -v
```

Expected: all pass.

---

### Task 5: Implement permanent pin in `on_prefill_checkpoint`

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py:516-524`

- [ ] **Step 1: Apply the change**

In `on_prefill_checkpoint`, find the insert + append block near the end of the method. Replace:

Current (`vllm_mlx/prefix_cache_adapters.py:516-524`):
```python
        new_node = self._inner.insert(
            parent,
            segment,
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )
        if cs is not None:
            cs.turn_path.append(new_node)
```

Replacement:
```python
        new_node = self._inner.insert(
            parent,
            segment,
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )
        new_node.ref_count += 1  # pin for duration of request; released by TurnCacheManager.release()
        if cs is not None:
            cs.turn_path.append(new_node)
```

- [ ] **Step 2: Run the ref_count pinning test to confirm it now passes**

```bash
pytest tests/test_turn_prefix_cache.py::test_on_prefill_checkpoint_pins_inserted_node -v
```

Expected: **PASS**

---

### Task 6: Full test suite and commit

**Files:** (read-only + git)

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/test_turn_prefix_cache.py tests/test_prefix_cache_adapters.py -v
```

Expected: all tests pass, including the two new ones.

- [ ] **Step 2: Verify the three scheduler release call sites handle the pinned nodes correctly (read-only audit)**

Open `vllm_mlx/scheduler.py` and confirm these three locations all call `self._prefix_cache.release(turn_path)` or equivalent:

- Abort path: around line 1101 — `self._prefix_cache.release(turn_path)`
- Prefill error retry: around line 1283 — `self._prefix_cache.release(request._cache_state.turn_path)`
- Normal completion: around line 1514 — `self._prefix_cache.release(_handle)` where `_handle = request._cache_state.turn_path`

All three already release the full `cs.turn_path`. Since checkpoint-inserted nodes now have `ref_count=1`, they will correctly drain to `ref_count=0` on release. No code changes needed.

- [ ] **Step 3: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py vllm_mlx/prefix_cache_adapters.py tests/test_turn_prefix_cache.py
git commit -m "fix: pin nodes during and after on_prefill_checkpoint to prevent self-eviction"
```
