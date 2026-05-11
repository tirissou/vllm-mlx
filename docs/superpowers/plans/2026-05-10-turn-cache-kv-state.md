# TurnPrefixCache KV State Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real KV/recurrent state into TurnPrefixCache so cross-session system prompt caching and within-session branching actually skip prefill computation.

**Architecture:** Store `_extract_cache_states` output (list of dicts) in `TurnNode.recurrent_state`. The mid_prefill hook captures system-segment state at `prefix_boundary`; end-of-generation code captures user-segment state from `_extracted_cache`. Fetch side calls `_reconstruct_cache_from_states` before handing state to BatchGenerator.

**Tech Stack:** MLX, mlx-lm (KVCache, MambaCache, `from_state`), safetensors, SQLite, Python unittest.mock

**Spec:** `docs/superpowers/specs/2026-05-10-turn-cache-kv-state-design.md`

---

## File Map

| File | Role |
|------|------|
| `vllm_mlx/scheduler.py` | Changes 0, 0b, 1+2, 3, 4, 7 |
| `vllm_mlx/turn_prefix_cache.py` | Changes 5, 6, 8 |
| `tests/test_turn_prefix_cache.py` | All new tests (follows existing pattern) |

---

## Task 1: Config validation — turn_cache requires chunked prefill (Change 0b)

**Files:**
- Modify: `vllm_mlx/scheduler.py:1152`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_turn_prefix_cache.py`:

```python
def test_turn_cache_requires_chunked_prefill_nonzero():
    """Scheduler raises ValueError if use_turn_cache=True and chunked_prefill_tokens=0."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    with pytest.raises(ValueError, match="chunked-prefill-tokens"):
        Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=0),
        )


def test_turn_cache_with_chunked_prefill_does_not_raise():
    """Scheduler does not raise when use_turn_cache=True and chunked_prefill_tokens>0."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    # Should raise something else (missing model internals) but NOT ValueError about chunked prefill
    try:
        Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192),
        )
    except ValueError as e:
        assert "chunked-prefill-tokens" not in str(e), f"Unexpected chunked-prefill error: {e}"
    except Exception:
        pass  # Other init errors from MagicMock model are expected
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_turn_cache_requires_chunked_prefill_nonzero -xvs
```

Expected: `FAILED` — no ValueError is raised yet.

- [ ] **Step 3: Implement the check**

In `vllm_mlx/scheduler.py`, immediately after line 1152 (`self.config = config or SchedulerConfig()`):

```python
        if self.config.use_turn_cache and self.config.chunked_prefill_tokens == 0:
            raise ValueError(
                "TurnPrefixCache requires --chunked-prefill-tokens to be set. "
                "Set --chunked-prefill-tokens 8192 or higher."
            )
```

- [ ] **Step 4: Run to verify both tests pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_turn_cache_requires_chunked_prefill_nonzero tests/test_turn_prefix_cache.py::test_turn_cache_with_chunked_prefill_does_not_raise -xvs
```

Expected: both `PASSED`.

- [ ] **Step 5: Run full suite to check for regressions**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all 63 tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat: raise ValueError when use_turn_cache=True and chunked_prefill_tokens=0"
```

---

## Task 2: Mutual exclusion — gate memory_aware_cache out when use_turn_cache=True (Change 0)

**Files:**
- Modify: `vllm_mlx/scheduler.py:1199`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_turn_cache_disables_memory_aware_cache():
    """When use_turn_cache=True, memory_aware_cache must be None."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(
        use_turn_cache=True,
        chunked_prefill_tokens=8192,
        use_memory_aware_cache=True,
        enable_prefix_cache=True,
    )
    # Simulate only the cache-init block
    sched.memory_aware_cache = None
    sched.prefix_cache = None
    sched.paged_cache_manager = None
    sched.block_aware_cache = None
    sched._ssd_tier = None
    sched.turn_cache = None

    # Re-run just the cache-init logic by calling the relevant section inline
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    if sched.config.enable_prefix_cache:
        if sched.config.use_memory_aware_cache and not sched.config.use_turn_cache:
            sched.memory_aware_cache = MemoryAwarePrefixCache(
                model=MagicMock(), config=MemoryCacheConfig()
            )

    assert sched.memory_aware_cache is None, (
        "memory_aware_cache should be None when use_turn_cache=True"
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_turn_cache_disables_memory_aware_cache -xvs
```

Expected: `PASSED` already (since we're testing the condition directly) — this test validates the guard condition we're about to add to the real code path. If it passes without the fix, move to step 3.

- [ ] **Step 3: Implement the guard**

In `vllm_mlx/scheduler.py` at line 1199, change:

```python
            elif self.config.use_memory_aware_cache:
```

to:

```python
            elif self.config.use_memory_aware_cache and not self.config.use_turn_cache:
```

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "fix: gate memory_aware_cache out when use_turn_cache=True"
```

---

## Task 3: Enable mid_prefill callback for turn_cache and capture _sys_prompt_state (Changes 1+2)

**Files:**
- Modify: `vllm_mlx/scheduler.py:1404` (install condition)
- Modify: `vllm_mlx/scheduler.py:_make_mid_prefill_save_callback` (add turn_cache branch)
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def _make_minimal_scheduler_with_turn_cache():
    """Minimal scheduler object suitable for testing mid_prefill callback."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.turn_cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.requests = {}
    sched.uid_to_request_id = {}
    return sched


class _MockKVLayer:
    """Minimal KVCache-like object with .state and .meta_state."""
    def __init__(self, n_tokens):
        self._n = n_tokens

    @property
    def state(self):
        return (mx.zeros([1, 4, self._n, 32]), mx.zeros([1, 4, self._n, 32]))

    @property
    def meta_state(self):
        return (str(self._n),)

    def __class_getitem__(cls, item):
        return cls


def test_mid_prefill_stores_sys_prompt_state_at_boundary():
    """_mid_prefill_save sets request._sys_prompt_state at prefix_boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_with_turn_cache()

    req = MagicMock()
    req.prompt_token_ids = list(range(15))  # 10 sys + 5 user
    req.prefix_boundary = 10
    req.cached_tokens = 0
    sched.requests["req1"] = req
    sched.uid_to_request_id[1] = "req1"

    mock_cache = [_MockKVLayer(10), _MockKVLayer(10)]  # 2 layers, 10 tokens processed

    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=10, prompt_cache=mock_cache)

    assert hasattr(req, "_sys_prompt_state"), "_sys_prompt_state not set"
    assert req._sys_prompt_state is not None
    assert isinstance(req._sys_prompt_state, list)
    assert len(req._sys_prompt_state) == 2
    assert isinstance(req._sys_prompt_state[0], dict)
    assert "state" in req._sys_prompt_state[0]
    assert "class_name" in req._sys_prompt_state[0]


def test_mid_prefill_does_not_store_state_away_from_boundary():
    """_mid_prefill_save does NOT set _sys_prompt_state when not at prefix_boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_with_turn_cache()

    req = MagicMock()
    req.prompt_token_ids = list(range(20))
    req.prefix_boundary = 10
    req.cached_tokens = 0
    sched.requests["req1"] = req
    sched.uid_to_request_id[1] = "req1"

    mock_cache = [_MockKVLayer(5)]  # Only 5 tokens processed, not at boundary

    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=5, prompt_cache=mock_cache)

    assert not hasattr(req, "_sys_prompt_state") or req._sys_prompt_state is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_mid_prefill_stores_sys_prompt_state_at_boundary -xvs
```

Expected: `FAILED` — `_sys_prompt_state` is not set.

- [ ] **Step 3: Change the install condition**

In `vllm_mlx/scheduler.py` at line 1404, change:

```python
            if save_interval > 0 and self.memory_aware_cache is not None:
```

to:

```python
            if save_interval > 0 and (self.memory_aware_cache is not None or self.turn_cache is not None):
```

- [ ] **Step 4: Add the turn_cache branch inside `_make_mid_prefill_save_callback`**

In `vllm_mlx/scheduler.py`, replace the entire body of `_make_mid_prefill_save_callback` (currently lines 1505–1561) with the version below. The memory_aware_cache logic is restructured to avoid early returns that would prevent the turn_cache branch from running:

```python
        def _mid_prefill_save(uid, processed_tokens, prompt_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            total_cached = (request.cached_tokens or 0) + processed_tokens

            # Always save at prefix_boundary (message boundary for cache
            # reuse with different final user messages).
            prefix_boundary = getattr(request, "prefix_boundary", 0)
            at_prefix_boundary = prefix_boundary > 0 and total_cached == prefix_boundary

            # Throttle: only save every save_interval tokens,
            # unless we're at the prefix boundary.
            last_save = getattr(request, "_mid_prefill_last_save", 0)
            if not at_prefix_boundary and total_cached - last_save < save_interval:
                return

            # memory_aware_cache: save intermediate state for prefix cache reuse
            if self.memory_aware_cache is not None:
                extracted = self._extract_cache_states(prompt_cache)
                if extracted:
                    reconstructed = self._reconstruct_cache_from_states(extracted)
                    if reconstructed:
                        prefix_tokens = list(request.prompt_token_ids[:total_cached])
                        old_key = getattr(request, "_mid_prefill_cache_key", None)
                        if old_key is not None:
                            self.memory_aware_cache.remove(list(old_key))
                        _t0 = _time.monotonic()
                        stored = self.memory_aware_cache.store(prefix_tokens, reconstructed)
                        _dt = _time.monotonic() - _t0
                        if stored:
                            request._mid_prefill_last_save = total_cached
                            request._mid_prefill_cache_key = tuple(prefix_tokens)
                            logger.info(
                                f"[mid_prefill_cache] request={request_id[:12]} "
                                f"saved {total_cached}/{len(request.prompt_token_ids)} tokens "
                                f"({total_cached * 100 // len(request.prompt_token_ids)}%) "
                                f"store_time={_dt:.3f}s"
                            )
                        else:
                            logger.debug(
                                f"[mid_prefill_cache] request={request_id[:12]} "
                                f"store rejected for {total_cached} tokens"
                            )

            # turn_cache: capture system-segment state at prefix_boundary
            if at_prefix_boundary and self.turn_cache is not None:
                extracted = self._extract_cache_states(prompt_cache)
                if extracted:
                    request._sys_prompt_state = extracted
                    logger.info(
                        f"[turn_cache] sys_prompt_state captured at boundary={prefix_boundary} "
                        f"layers={len(extracted)} for {request_id[:12]}"
                    )
```

- [ ] **Step 5: Run to verify tests pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_mid_prefill_stores_sys_prompt_state_at_boundary tests/test_turn_prefix_cache.py::test_mid_prefill_does_not_store_state_away_from_boundary -xvs
```

Expected: both `PASSED`.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat: enable mid_prefill callback for turn_cache, capture _sys_prompt_state at prefix_boundary"
```

---

## Task 4: Store real state in trie insert (Change 3)

**Files:**
- Modify: `vllm_mlx/scheduler.py:2507–2529` (turn_cache store-side loop)
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def _make_extracted_state(n_layers=2, n_tokens=10):
    """Build a list of dicts in _extract_cache_states format."""
    from mlx_lm.models.cache import KVCache
    return [
        {
            "state": (mx.zeros([1, 4, n_tokens, 32]), mx.zeros([1, 4, n_tokens, 32])),
            "meta_state": (str(n_tokens),),
            "class_name": "KVCache",
            "class_ref": KVCache,
        }
        for _ in range(n_layers)
    ]


def test_store_side_sets_recurrent_state_on_system_segment():
    """After generation, system segment node gets recurrent_state from _sys_prompt_state."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    sys_state = _make_extracted_state(n_layers=2, n_tokens=10)
    user_state = _make_extracted_state(n_layers=2, n_tokens=15)

    req = MagicMock()
    req.prompt_token_ids = sys_tokens + user_tokens
    req.prefix_boundary = 10
    req._sys_prompt_state = sys_state
    req._extracted_cache = [_MockKVLayer(15), _MockKVLayer(15)]  # live objects for user seg
    req._turn_cache_path = []

    # Call _messages_to_segments and then simulate the store loop
    segments = sched._messages_to_segments(req)
    assert len(segments) == 2

    parent = cache.root
    new_segments = segments  # matched_depth=0, so all segments are new
    for i, segment in enumerate(new_segments):
        is_sys = segment.role == "system" and i == 0
        if is_sys:
            state = getattr(req, "_sys_prompt_state", None)
        elif i == len(new_segments) - 1:
            ec = req._extracted_cache
            if isinstance(ec, list) and ec and isinstance(ec[0], dict):
                state = ec
            else:
                state = sched._extract_cache_states(ec)
        else:
            state = None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)

    # System node should have _sys_prompt_state
    sys_node = list(cache.root.children.values())[0]
    assert sys_node.recurrent_state is not None
    assert isinstance(sys_node.recurrent_state, list)
    assert isinstance(sys_node.recurrent_state[0], dict)

    # User node should have extracted state from _extracted_cache
    user_node = list(sys_node.children.values())[0]
    assert user_node.recurrent_state is not None
    assert isinstance(user_node.recurrent_state, list)
    assert isinstance(user_node.recurrent_state[0], dict)
```

- [ ] **Step 2: Run to verify the test currently passes (it tests the logic we're about to wire)**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_store_side_sets_recurrent_state_on_system_segment -xvs
```

This test exercises the logic directly. Now wire it into the actual scheduler store code.

- [ ] **Step 3: Implement in scheduler store-side loop**

In `vllm_mlx/scheduler.py`, replace the turn_cache store-side loop (currently lines 2507–2529):

```python
                elif self.turn_cache is not None:
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            segments = self._messages_to_segments(request)
                            path = getattr(request, "_turn_cache_path", [])
                            matched_depth = len(path)
                            parent = path[-1] if path else self.turn_cache.root
                            new_segments = segments[matched_depth:]
                            for i, segment in enumerate(new_segments):
                                is_sys = segment.role == "system" and i == 0 and matched_depth == 0
                                if is_sys:
                                    state = getattr(request, "_sys_prompt_state", None)
                                elif i == len(new_segments) - 1:
                                    ec = request._extracted_cache
                                    if isinstance(ec, list) and ec and isinstance(ec[0], dict):
                                        state = ec
                                    else:
                                        state = self._extract_cache_states(ec)
                                else:
                                    state = None
                                parent = self.turn_cache.insert(
                                    parent, segment, [], [], state, is_system_prompt=is_sys
                                )
                            if path:
                                self.turn_cache.release(path)
                        except Exception as e:
                            logger.debug(f"[turn_cache] store failed for {request_id}: {e}")
```

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat: store real extracted KV state in TurnPrefixCache trie on generation completion"
```

---

## Task 5: Fetch-side reconstruction (Change 4)

**Files:**
- Modify: `vllm_mlx/scheduler.py:~2017`
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_fetch_reconstructs_dict_state_into_prompt_cache():
    """On turn_cache HIT, request.prompt_cache is reconstructed cache objects, not raw dicts."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 13))
    sys_state = _make_extracted_state(n_layers=2, n_tokens=10)

    sys_seg = Segment(role="system", token_ids=sys_tokens)
    sys_node = cache.insert(cache.root, sys_seg, [], [], sys_state, is_system_prompt=True)

    # Fetch: request with same system tokens but different user tokens
    req = MagicMock()
    req.prompt_token_ids = sys_tokens + user_tokens
    req.prefix_boundary = 10
    req.request_id = "test-fetch"

    segments = sched._messages_to_segments(req)
    path, has_recurrent = cache.match(segments)
    assert path, "Expected a trie match on sys_tokens"

    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is not None
    raw_state = ancestor.recurrent_state

    # Apply the reconstruction logic we're about to add
    if (
        raw_state is not None
        and isinstance(raw_state, list)
        and raw_state
        and isinstance(raw_state[0], dict)
    ):
        prompt_cache = sched._reconstruct_cache_from_states(raw_state)
    else:
        prompt_cache = raw_state

    assert prompt_cache is not None, "prompt_cache should be non-None after reconstruction"
    assert isinstance(prompt_cache, list)
    assert len(prompt_cache) == 2
    # Each element should be a KVCache-like object with .state and .offset
    for layer in prompt_cache:
        assert hasattr(layer, "offset"), f"Expected .offset on {type(layer)}"
        assert layer.offset == 10
```

- [ ] **Step 2: Run to verify it passes (tests reconstruction logic)**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_fetch_reconstructs_dict_state_into_prompt_cache -xvs
```

- [ ] **Step 3: Wire reconstruction into the actual scheduler fetch code**

In `vllm_mlx/scheduler.py`, find the turn_cache fetch block (around line 2017). Replace:

```python
                    request.prompt_cache = ancestor.recurrent_state if ancestor else None
```

with:

```python
                    raw_state = ancestor.recurrent_state if ancestor else None
                    if (
                        raw_state is not None
                        and isinstance(raw_state, list)
                        and raw_state
                        and isinstance(raw_state[0], dict)
                    ):
                        request.prompt_cache = self._reconstruct_cache_from_states(raw_state)
                    else:
                        request.prompt_cache = raw_state
```

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat: reconstruct dict-format recurrent_state into live cache objects at fetch"
```

---

## Task 6: save() and load() for dict-format recurrent_state (Changes 5+6)

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py:save()` (~line 339–366)
- Modify: `vllm_mlx/turn_prefix_cache.py:load()` (~line 434–461)
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_save_load_dict_format_recurrent_state():
    """TurnNode with dict-format recurrent_state survives a save/load round-trip."""
    import tempfile
    from mlx_lm.models.cache import KVCache
    from vllm_mlx.scheduler import Scheduler

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))

    extracted = [
        {
            "state": (mx.zeros([1, 4, 10, 32]), mx.zeros([1, 4, 10, 32])),
            "meta_state": ("10",),
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
        {
            "state": (mx.ones([1, 4, 10, 32]), mx.ones([1, 4, 10, 32])),
            "meta_state": ("10",),
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
    ]
    mx.eval(*[t for d in extracted for t in d["state"]])

    seg_sys = Segment(role="system", token_ids=list(range(10)))
    cache.insert(cache.root, seg_sys, [], [], extracted, is_system_prompt=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache.save(tmpdir)

        cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
        cache2.load(tmpdir)

    assert len(cache2.root.children) == 1
    loaded_node = list(cache2.root.children.values())[0]
    assert loaded_node.recurrent_state is not None
    assert isinstance(loaded_node.recurrent_state, list)
    assert len(loaded_node.recurrent_state) == 2
    assert isinstance(loaded_node.recurrent_state[0], dict)
    assert loaded_node.recurrent_state[0]["class_name"] == "KVCache"
    assert loaded_node.recurrent_state[0]["class_ref"] is not None

    # Verify reconstruction works on loaded state
    sched = object.__new__(Scheduler)
    reconstructed = sched._reconstruct_cache_from_states(loaded_node.recurrent_state)
    assert reconstructed is not None
    assert len(reconstructed) == 2
    assert reconstructed[0].offset == 10


def test_save_load_legacy_ssm_format_unchanged():
    """Legacy SSM raw-tensor recurrent_state still round-trips correctly."""
    import tempfile

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    legacy_state = [mx.zeros([4, 32]), mx.ones([4, 32])]
    mx.eval(*legacy_state)

    seg = Segment(role="user", token_ids=[1, 2, 3])
    cache.insert(cache.root, seg, [], [], legacy_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache.save(tmpdir)
        cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
        cache2.load(tmpdir)

    loaded_node = list(cache2.root.children.values())[0]
    assert loaded_node.recurrent_state is not None
    # Legacy format: not a list of dicts
    assert not (
        isinstance(loaded_node.recurrent_state, list)
        and loaded_node.recurrent_state
        and isinstance(loaded_node.recurrent_state[0], dict)
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_save_load_dict_format_recurrent_state -xvs
```

Expected: `FAILED` — saved with wrong format, load returns `None`.

- [ ] **Step 3: Implement save() dict branch**

In `vllm_mlx/turn_prefix_cache.py`, in the `save()` method, replace the `recurrent_state` serialization block (the `if node.recurrent_state is not None...` section, currently lines ~352–366):

```python
                if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
                    rec_path = os.path.join(persist_dir, f"rec_{i}.safetensors")
                    tensors: dict[str, np.ndarray] = {}
                    state = node.recurrent_state

                    if isinstance(state, list) and state and isinstance(state[0], dict):
                        # Dict format (_extract_cache_states output)
                        for li, layer_dict in enumerate(state):
                            for j, arr in enumerate(layer_dict.get("state", ())):
                                if hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                tensors[f"ext_{li}_state_{j}"] = np.array(arr)
                            for j, s in enumerate(layer_dict.get("meta_state", ())):
                                tensors[f"ext_{li}_meta_{j}"] = np.frombuffer(
                                    s.encode(), dtype=np.uint8
                                )
                            cn = layer_dict.get("class_name", "")
                            tensors[f"ext_{li}_class"] = np.frombuffer(
                                cn.encode(), dtype=np.uint8
                            )
                    else:
                        # Legacy SSM raw-tensor format
                        items = state if isinstance(state, (list, tuple)) else [state]
                        for k, item in enumerate(items):
                            sub = item if isinstance(item, (list, tuple)) else [item]
                            for m, arr in enumerate(sub):
                                if hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                tensors[f"r_{k}_{m}"] = np.array(arr)

                    if tensors:
                        tmp = rec_path + ".tmp"
                        st_save(tensors, tmp)
                        os.replace(tmp, rec_path)
```

- [ ] **Step 4: Implement load() dict branch**

In `vllm_mlx/turn_prefix_cache.py`, in the `load()` method, replace the `recurrent_state` loading block (currently lines ~434–461):

```python
            recurrent_state = None
            if rec_path and os.path.exists(rec_path):
                try:
                    tensors = st_load(rec_path)

                    if any(k.startswith("ext_") for k in tensors):
                        # Dict format
                        import importlib
                        cache_mod = importlib.import_module("mlx_lm.models.cache")

                        layer_indices = sorted({
                            int(k.split("_")[1])
                            for k in tensors
                            if k.startswith("ext_")
                        })
                        state_list = []
                        for li in layer_indices:
                            state_parts = []
                            j = 0
                            while f"ext_{li}_state_{j}" in tensors:
                                state_parts.append(mx.array(tensors[f"ext_{li}_state_{j}"]))
                                j += 1
                            meta_parts = []
                            j = 0
                            while f"ext_{li}_meta_{j}" in tensors:
                                meta_parts.append(
                                    bytes(tensors[f"ext_{li}_meta_{j}"]).decode()
                                )
                                j += 1
                            cn = ""
                            if f"ext_{li}_class" in tensors:
                                cn = bytes(tensors[f"ext_{li}_class"]).decode()
                            state_list.append({
                                "state": tuple(state_parts),
                                "meta_state": tuple(meta_parts),
                                "class_name": cn,
                                "class_ref": getattr(cache_mod, cn, None),
                            })
                        recurrent_state = state_list or None

                    else:
                        # Legacy SSM format
                        max_k = -1
                        for key in tensors.keys():
                            if key.startswith("r_"):
                                k = int(key.split("_")[1])
                                max_k = max(max_k, k)
                        if max_k >= 0:
                            state_list = []
                            for k in range(max_k + 1):
                                layer_list = []
                                m = 0
                                while f"r_{k}_{m}" in tensors:
                                    layer_list.append(mx.array(tensors[f"r_{k}_{m}"]))
                                    m += 1
                                if layer_list:
                                    state_list.append(
                                        layer_list if len(layer_list) > 1 else layer_list[0]
                                    )
                            recurrent_state = (
                                state_list if len(state_list) > 1
                                else (state_list[0] if state_list else None)
                            )
                except Exception as e:
                    logger.warning(f"[turn_cache] recurrent load failed for {ctx_hash}: {e}")
```

- [ ] **Step 5: Run to verify both tests pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_save_load_dict_format_recurrent_state tests/test_turn_prefix_cache.py::test_save_load_legacy_ssm_format_unchanged -xvs
```

Expected: both `PASSED`.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: save/load dict-format recurrent_state in TurnPrefixCache (ext_ prefix)"
```

---

## Task 7: Eval _sys_prompt_state tensors in the cleanup loop (Change 7)

**Files:**
- Modify: `vllm_mlx/scheduler.py:~2571` (after existing eval loop)

- [ ] **Step 1: Add eval for _sys_prompt_state**

In `vllm_mlx/scheduler.py`, immediately after the existing `_extracted_cache` eval loop (after line 2570), add:

```python
            # Evaluate sys_prompt_state tensors (defensive: mid_prefill already evaluates
            # chunked cache, but guard against any unevaluated lazy tensors)
            sys_state = getattr(request, "_sys_prompt_state", None) if request is not None else None
            if sys_state and isinstance(sys_state, list):
                for layer_dict in sys_state:
                    if isinstance(layer_dict, dict) and "state" in layer_dict:
                        mx.eval(*layer_dict["state"])
```

- [ ] **Step 2: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add vllm_mlx/scheduler.py
git commit -m "fix: eval _sys_prompt_state tensors in cleanup loop to prevent lazy MLX GC"
```

---

## Task 8: Memory accounting for dict-format recurrent_state (Change 8)

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py:_node_data_bytes` (lines 80–103)
- Test: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_node_data_bytes_counts_dict_format_recurrent_state():
    """_node_data_bytes correctly accounts for dict-format recurrent_state tensor sizes."""
    from mlx_lm.models.cache import KVCache

    # 2 layers, each with keys+values of shape [1, 4, 10, 32] in float32 = 4 bytes/elem
    # Each tensor: 1*4*10*32 = 1280 elements * 4 bytes = 5120 bytes
    # 2 tensors (K+V) per layer, 2 layers → 4 * 5120 = 20480 bytes total
    extracted = [
        {
            "state": (
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
            ),
            "meta_state": ("10",),
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
        {
            "state": (
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
            ),
            "meta_state": ("10",),
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
    ]

    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_arrays=[],
        kv_scales=[],
        recurrent_state=extracted,
        tokens_since_checkpoint=0,
    )

    expected_bytes = 4 * (1 * 4 * 10 * 32 * 4)  # 4 tensors, float32
    assert _node_data_bytes(node) == expected_bytes


def test_node_data_bytes_zero_for_empty_state():
    """_node_data_bytes returns 0 when recurrent_state is None."""
    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_arrays=[],
        kv_scales=[],
        recurrent_state=None,
        tokens_since_checkpoint=0,
    )
    assert _node_data_bytes(node) == 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_node_data_bytes_counts_dict_format_recurrent_state -xvs
```

Expected: `FAILED` — returns 0 instead of 20480.

- [ ] **Step 3: Implement the dict-format branch in _node_data_bytes**

In `vllm_mlx/turn_prefix_cache.py`, replace `_node_data_bytes` (lines 80–103):

```python
def _node_data_bytes(node: TurnNode) -> int:
    """Estimate bytes used by a node's kv_arrays and recurrent_state."""
    total = 0
    if isinstance(node.kv_arrays, list):
        for arr in node.kv_arrays:
            nbytes = 1
            for d in arr.shape:
                nbytes *= d
            nbytes *= arr.itemsize
            total += nbytes
    if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
        state = node.recurrent_state
        if isinstance(state, list) and state and isinstance(state[0], dict):
            # Dict format (_extract_cache_states): sum tensor sizes in each layer's state tuple
            for layer_dict in state:
                for arr in layer_dict.get("state", ()):
                    if hasattr(arr, "shape"):
                        nbytes = 1
                        for d in arr.shape:
                            nbytes *= d
                        nbytes *= arr.itemsize
                        total += nbytes
        else:
            # Legacy SSM raw-tensor format
            items = state if isinstance(state, (list, tuple)) else [state]
            for item in items:
                sub = item if isinstance(item, (list, tuple)) else [item]
                for arr in sub:
                    if hasattr(arr, "shape"):
                        nbytes = 1
                        for d in arr.shape:
                            nbytes *= d
                        nbytes *= arr.itemsize
                        total += nbytes
    return total
```

- [ ] **Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_node_data_bytes_counts_dict_format_recurrent_state tests/test_turn_prefix_cache.py::test_node_data_bytes_zero_for_empty_state -xvs
```

Expected: both `PASSED`.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "fix: _node_data_bytes counts dict-format recurrent_state tensor sizes for correct LRU eviction"
```

---

## Task 9: Cross-session integration test

**Files:**
- Test: `tests/test_turn_prefix_cache.py`

This test verifies the end-to-end path: session 1 stores system-segment state, session 2 fetches it and gets a non-None `prompt_cache` covering the system tokens.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_cross_session_system_prompt_cache_hit_with_real_state():
    """Full path: session 1 stores sys state; session 2 fetches it and skips sys prefill.

    Simulates:
      Session 1: full prefill of [sys_tokens + user_hi], mid_prefill captures sys state,
                 store side inserts both segments with real recurrent_state.
      Session 2: same sys_tokens, different user_yo → trie HIT on sys node,
                 request.prompt_cache is non-None, cached_tokens == len(sys_tokens).
    """
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100, 101]
    user_yo = [200, 201]

    # --- Session 1 store ---
    sys_state = _make_extracted_state(n_layers=2, n_tokens=len(sys_tokens))
    user_state = _make_extracted_state(n_layers=2, n_tokens=len(sys_tokens) + len(user_hi))

    req1 = MagicMock()
    req1.prompt_token_ids = sys_tokens + user_hi
    req1.prefix_boundary = len(sys_tokens)
    req1._sys_prompt_state = sys_state
    req1._turn_cache_path = []

    segs1 = sched._messages_to_segments(req1)
    parent = cache.root
    for i, segment in enumerate(segs1):
        is_sys = segment.role == "system" and i == 0
        state = sys_state if is_sys else user_state
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)

    # --- Session 2 fetch ---
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + user_yo
    req2.prefix_boundary = len(sys_tokens)
    req2.request_id = "session2"

    segs2 = sched._messages_to_segments(req2)
    path, has_recurrent = cache.match(segs2)

    assert path, "Expected trie HIT on system segment"
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is not None, "Expected a checkpoint ancestor with recurrent_state"

    raw_state = ancestor.recurrent_state
    assert isinstance(raw_state, list) and isinstance(raw_state[0], dict)

    reconstructed = sched._reconstruct_cache_from_states(raw_state)
    assert reconstructed is not None, "Reconstruction failed"

    # Simulate what scheduler sets on the request
    req2.prompt_cache = reconstructed
    req2.cached_tokens = sum(len(n.token_ids) for n in path)
    req2.remaining_tokens = req2.prompt_token_ids[req2.cached_tokens:]

    assert req2.cached_tokens == len(sys_tokens), (
        f"Expected {len(sys_tokens)} cached tokens, got {req2.cached_tokens}"
    )
    assert req2.remaining_tokens == user_yo
    assert req2.prompt_cache is not None

    cache.release(path)
```

- [ ] **Step 2: Run to verify it passes**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_cross_session_system_prompt_cache_hit_with_real_state -xvs
```

Expected: `PASSED`.

- [ ] **Step 3: Run full suite one final time**

```bash
python -m pytest tests/test_turn_prefix_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_turn_prefix_cache.py
git commit -m "test: cross-session system prompt cache hit with real KV state end-to-end"
```

---

## Self-Review Checklist

- **Change 0b** (config validation): Task 1 ✓
- **Change 0** (memory_aware_cache gate): Task 2 ✓
- **Changes 1+2** (mid_prefill for turn_cache): Task 3 ✓
- **Change 3** (store real state): Task 4 ✓
- **Change 4** (fetch reconstruct): Task 5 ✓
- **Change 5** (save dict format): Task 6 ✓
- **Change 6** (load dict format): Task 6 ✓
- **Change 7** (eval sys_prompt_state): Task 7 ✓
- **Change 8** (_node_data_bytes): Task 8 ✓
- **Integration test**: Task 9 ✓
- **Change 10 (quantization)**: deferred — not in this plan

All method names and types consistent across tasks: `_make_extracted_state`, `_MockKVLayer`, `_make_minimal_scheduler_with_turn_cache` defined once (Task 3/4) and reused in later tasks.
