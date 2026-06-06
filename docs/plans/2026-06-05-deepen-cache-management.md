# Deepen Cache Management Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Refactor the cache lifecycle management from the `Scheduler` into a high-leverage `CacheManager` interface.

**Architecture:** Implement a batch-oriented `CacheManager` (Interface C) that encapsulates the complexity of chunking, boundary detection, and lifecycle management (fetch, store, release, cleanup) within the adapter layer.

**Tech Stack:** Python, MLX, Pytest

---

## Phase 1: Foundation (Contract & No-Op)

### Task 1: Update Request Control Plane

**Files:**
- Modify: `vllm_mlx/request.py`
- Modify: `vllm_mlx/kv_cache.py`
- Test: `tests/test_request_cache_state.py` (New)

**Step 1: Add Control Plane fields to Request**

```python
# vllm_mlx/request.py

@dataclass
class Request:
    # ... existing fields ...
    # Control Plane fields (used for telemetry/scheduling decisions)
    next_boundary_distance: int = 0
    cache_hit_type: Optional[str] = None
```

**Step 2: Update RequestCacheState for implementation details**

```python
# vllm_mlx/kv_cache.py

@dataclass
class RequestCacheState:
    # Control Plane (managed by CacheManager)
    hit_type: str = "miss"
    cached_tokens: int = 0
    remaining_tokens: list | None = None
    prefill_boundaries: list = field(default_factory=list)

    # Implementation Details (opaque to Scheduler)
    cache: list | None = None
    decoded_cache: list | None = None
    turn_path: list = field(default_factory=list)
    # ... existing fields ...
```

**Step 3: Write and run test**

```python
# tests/test_request_cache_state.py
from vllm_mlx.request import Request
from vllm_mlx.kv_cache import RequestCacheState

def test_request_cache_state_initialization():
    req = Request(request_id="test", prompt="hello")
    req._cache_state = RequestCacheState()
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.turn_path == []
```

**Step 4: Commit**
`git add vllm_mlx/request.py vllm_mlx/kv_cache.py tests/test_request_cache_state.py`
`git commit -m "feat: add control plane fields to Request and RequestCacheState"`

---

### Task 2: Implement No-Op Cache and Scheduler Wiring

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`
- Modify: `vllm_mlx/scheduler.py`
- Test: `tests/test_scheduler_noop_cache.py` (New)

**Step 1: Define CacheManager (ABC) and NoOpCacheManager**

```python
# vllm_mlx/prefix_cache_adapters.py

class CacheManager(ABC):
    @abstractmethod
    def prepare_batch(self, requests: list[Request], max_chunk_size: int) -> list[list[Segment]]: ...
    @abstractmethod
    def cleanup_batch(self, finished: list[Request], aborted: list[Request]) -> None: ...
    @abstractmethod
    def get_stats(self) -> dict: ...

class NoOpCacheManager(CacheManager):
    def prepare_batch(self, requests, max_chunk_size):
        # Return whole prompt as single segment for each request
        return [[Segment(role="user", token_ids=r.prompt_token_ids)] for r in requests]
    def cleanup_batch(self, finished, aborted):
        pass
    def get_stats(self):
        return {}
```

**Step 2: Wire Scheduler to use CacheManager (initially NoOp)**

```python
# vllm_mlx/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ...
        self._prefix_cache: CacheManager = NoOpCacheManager() # Start with NoOp
```

**Step 3: Verify Scheduler runs with NoOpCacheManager**

Run: `pytest tests/test_scheduler_noop_cache.py`
Expected: PASS (Scheduler should perform normal scheduling without actual cache hits)

**Step 4: Commit**
`git add vllm_mlx/prefix_cache_adapters.py vllm_mlx/scheduler.py tests/test_scheduler_noop_cache.py`
`git commit -m "feat: add CacheManager interface and NoOp implementation"`

---

## Phase 2: Core Implementation (The Deep Adapter)

### Task 3: Implement TurnCacheManager.prepare_batch

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Test: `tests/test_turn_cache_manager_batch.py` (New)

**Step 1: Implement Batch-level coordination in TurnCacheManager**

```python
# vllm_mlx/prefix_cache_adapters.py

class TurnCacheManager(CacheManager):
    def prepare_batch(self, requests, max_chunk_size):
        # 1. Coordination: Batch-wide min(distance)
        distances = [self._inner.get_distance_to_next_boundary(req) for req in requests]
        batch_chunk_size = min(max_chunk_size, min(distances))
        
        all_segments = []
        for req in requests:
            # 2. Fetch (Trie Match)
            hit = self._inner.fetch(req)
            
            # 3. Slicing (Segmentation)
            # Implementation uses the batch_chunk_size and boundaries
            segments = self._get_segments_for_request(req, batch_chunk_size)
            all_segments.append(segments)
            
            # 4. Update Control Plane
            req.next_boundary_distance = distances[requests.index(req)]
            req.cache_hit_type = "hit" if hit else "miss"
            
        return all_segments
```

**Step 2: Run integration tests for boundary-aware batching**

Run: `pytest tests/test_turn_cache_manager_batch.py`
Expected: PASS (Segments correctly align with turn boundaries across the batch)

**Step 3: Commit**
`git add vllm_mlx/prefix_cache_adapters.py tests/test_turn_cache_manager_batch.py`
`git commit -m "feat: implement batch coordination in TurnCacheManager"`

---

### Task 4: Implement TurnCacheManager.cleanup_batch

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`
- Test: `tests/test_turn_cache_manager_cleanup.py` (New)

**Step 1: Implement Batch Cleanup**

```python
# vllm_mlx/prefix_cache_adapters.py

class TurnCacheManager(CacheManager):
    def cleanup_batch(self, finished, aborted):
        for req in finished:
            self._inner.store(req)
            self._inner.release(req)
            self._clear_request_implementation_state(req)
            
        for req in aborted:
            self._inner.release(req)
            self._clear_request_implementation_state(req)

    def _clear_request_implementation_state(self, req):
        # Clear the opaque _cache_state fields
        req._cache_state.cache = None
        req._cache_state.decoded_cache = None
        req._cache_state.turn_path = []
        # ... clear boundary state ...
```

**Step 2: Verify cleanup and memory release**

Run: `pytest tests/test_turn_cache_manager_cleanup.py`
Expected: PASS (Nodes released, request state cleared, no leaks)

**Step 3: Commit**
`git add vllm_mlx/prefix_cache_adapters.py tests/test_turn_cache_manager_cleanup.py`
`git commit -m "feat: implement batch cleanup in TurnCacheManager"`

---

## Phase 3: Integration (The Scheduler Refactor)

### Task 5: Refactor Scheduler Core Loop

**Files:**
- Modify: `vllm_mlx/scheduler.py`
- Test: `tests/test_scheduler_deep_cache.py` (New)

**Step 1: Replace manual orchestration with CacheManager calls**

```python
# vllm_mlx/scheduler.py

# In _schedule_waiting():
# Replace boundary/segmentation loop with:
segments_list = self._prefix_cache.prepare_batch(prefill_batch, self.config.max_num_batched_tokens)
for req, segments in zip(prefill_batch, segments_list):
    self.batch_generator.insert_segments(segments, ...)

# In _cleanup_finished():
# Replace manual store/release/clear with:
self._prefix_cache.cleanup_batch(finished=finished_ids, aborted=aborted_ids)
```

**Step 2: Run full end-to-end integration tests**

Run: `pytest tests/test_scheduler_deep_cache.py`
Expected: PASS (Scheduler works with high-leverage CacheManager)

**Step 3: Final Commit**
`git add vllm_mlx/scheduler.py`
`git commit -m "refactor: complete scheduler refactor to use deep CacheManager"`
