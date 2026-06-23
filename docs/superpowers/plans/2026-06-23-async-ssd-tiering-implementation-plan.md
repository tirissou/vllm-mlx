# Asynchronous SSD Cache Tiering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an asynchronous mechanism to promote data from SSD to RAM without blocking the main engine loop.

**Architecture:** Refactor the `EngineCore` worker thread to run an `asyncio` event loop. The `Scheduler` and `TurnCacheManager` will become natively asynchronous, allowing `await`ing of SSD I/O during cache promotion.

**Tech Stack:** Python `asyncio`, `vllm-mlx` internals, `mlx`.

---

### Task 1: Refactor `SSDCacheTier.async_promote`

**Files:**
- Modify: `vllm_mlx/ssd_cache.py:781-831` (and surrounding)
- Test: `tests/vllm_mlx/test_ssd_cache.py` (create if necessary)

- [ ] **Step 1: Write the failing test**

```python
import pytest
import asyncio
from vllm_mlx.ssd_cache import SSDCacheTier, SSDRef

@pytest.mark.asyncio
async def test_async_promote_with_ref():
    # Setup mock SSD cache and ref
    tier = SSDCacheTier(...) 
    ref = SSDRef(file_path="/tmp/test_cache", size_bytes=1024)
    
    # We want to ensure it calls the internal read method
    result = await tier.async_promote(ref)
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vllm_mlx/test_ssd_cache.py -v`
Expected: FAIL with `async_promote` having wrong signature or not implemented as expected.

- [ ] **Step 3: Write minimal implementation**

```python
# vllm_mlx/ssd_cache.py

async def async_promote(self, ref: SSDRef) -> list | None:
    """Promote an entry from SSD to RAM asynchronously."""
    import asyncio
    
    # Simplified implementation matching the spec
    try:
        return await asyncio.to_thread(self._read_entry, ref.file_path)
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vllm_mlx/test_ssd_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/ssd_cache.py tests/vllm_mlx/test_ssd_cache.py
git commit -m "feat: refactor SSDCacheTier.async_promote to match spec"
```

### Task 2: Refactor `TurnCacheManager.fetch` to be asynchronous

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py:278-328`
- Test: `tests/vllm_mlx/test_turn_cache_manager.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_turn_cache_manager_fetch_async():
    manager = TurnCacheManager(inner=mock_inner, ...)
    # Mock a hit that requires SSD promotion
    mock_inner.match.return_value = (path, SSDRef(...))
    
    result = await manager.fetch(request)
    assert result is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vllm_mlx/test_turn_cache_manager.py -v`
Expected: FAIL with `TypeError: fetch() takes 2 positional arguments but 3 were given` (if called with await) or similar.

- [ ] **Step 3: Write minimal implementation**

```python
# vllm_mlx/prefix_cache_adapters.py

async def fetch(self, request) -> bool:
    # ... existing match logic ...
    if isinstance(path[-1], SSDRef):
        # Implement memory budgeting and async promotion
        ref = path[-1]
        # Note: We need to implement the budget reservation logic here
        # as per the spec.
        async with asyncio.shield(self._perform_promotion(request, ref)):
             # ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vllm_mlx/test_turn_cache_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py tests/vllm_mlx/test_turn_cache_manager.py
git commit -m "feat: make TurnCacheManager.fetch asynchronous"
```

### Task 3: Refactor `Scheduler.step` to be asynchronous

**Files:**
- Modify: `vllm_mlx/scheduler.py:1810-2000`
- Test: `tests/vllm_mlx/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_scheduler_step_async():
    scheduler = Scheduler(...)
    output = await scheduler.step()
    assert output.has_work is True # or whatever the expectation is
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vllm_mlx/test_scheduler.py -v`
Expected: FAIL with `TypeError: object Scheduler can't be used in 'await' expression`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm_mlx/scheduler.py

async def step(self, max_retries: int = 1) -> SchedulerOutput:
    # ... 
    scheduled = await self._schedule_waiting()
    # ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vllm_mlx/test_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/scheduler.py tests/vllm_mlx/test_scheduler.py
git commit -m "feat: make Scheduler.step asynchronous"
```

### Task 4: Refactor `EngineCore` to run an `asyncio` loop in the worker thread

**Files:**
- Modify: `vllm_mlx/engine_core.py:174-200`
- Test: `tests/vllm_mlx/test_engine_core.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_engine_core_worker_loop_async():
    engine = EngineCore(...)
    # Start engine and wait for a few steps
    # ...
    assert engine._steps_executed > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vllm_mlx/test_engine_core.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# vllm_mlx/engine_core.py

def _step_on_worker():
    _bind_worker_streams_once()
    # Run the async scheduler step in the worker's event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    output = loop.run_until_complete(self.scheduler.step())
    self._steps_executed += 1
    # ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vllm_mlx/test_engine_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/engine_core.py tests/vllm_mlx/test_engine_core.py
git commit -m "feat: refactor EngineCore worker loop to use asyncio"
```
