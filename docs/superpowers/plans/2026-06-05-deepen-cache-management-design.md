# Deepen Cache Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor prefix cache management into a high-leverage, batch-oriented `CacheManager` interface to improve locality, depth, and separation of concerns.

**Architecture:** Introduce a `CacheManager` ABC that encapsulates the lifecycle (`fetch` $\to$ `store` $\to$ `release` $\to$ `cleanup`) of cache state. The `Scheduler` will interact solely with this interface. `TurnCacheManager` will implement this by wrapping the existing `TurnPrefixCache` trie, moving batch-level coordination (chunking, boundary alignment, and lifecycle management) into the manager.

**Tech Stack:** Python, mlx, mlx-lm, pytest

---

### Task 1: Refactor Request and CacheState

**Files:**
- Modify: `vllm_mlx/request.py`
- Modify: `vllm_mlx/kv_cache.py`

- [ ] **Step 1: Update `Request` class**
    - Remove all direct prefix cache fields from `Request` (`prompt_cache`, `cached_tokens`, `remaining_tokens`, `_turn_boundaries`, `_boundary_states`, etc.).
    - Add `_cache_state: RequestCacheState` to `Request`.

- [ ] **Step 2: Update `RequestCacheState`**
    - Ensure `RequestCacheState` in `vllm_mlx/kv_cache.py` contains all necessary implementation-specific fields: `hit_type`, `cache`, `cached_tokens`, `remaining_tokens`, `prefill_boundaries`, `decoded_cache`, `turn_path`, etc.

- [ ] **Step 3: Run tests and commit**
    - Run existing tests to ensure no breakage from removing fields (although `Request` attributes might be accessed elsewhere).
    - Commit.

### Task 2: Implement new CacheManager interface

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`

- [ ] **Step 1: Update `CacheManager` ABC**
    - Add `prepare_batch(requests: list[Request], max_chunk_size: int) -> list[list[Segment]]` to the abstract interface.
    - Add `cleanup_batch(finished: list[Request], aborted: list[Request]) -> None` to the abstract interface.

- [ ] **Step 2: Implement `TurnCacheManager.prepare_batch`**
    - Iterate through `requests`.
    - Call `self.fetch(request)` for each request.
    - Use `self.messages_to_segments(request)` to get segments.
    - Use `self.boundaries(request)` to get chunk split points.
    - Slice segments based on boundaries.
    - Return the `list[list[Segment]]`.

- [ ] **Step 3: Implement `TurnCacheManager.cleanup_batch`**
    - For `finished` requests:
        - Call `self.store(request, tokens, cache)`.
        - Call `self.release(request._cache_state.turn_path)`.
        - Clear `request._cache_state`.
    - For `aborted` requests:
        - Call `self.release(request._cache_state.turn_path)`.
        - Clear `request._cache_state`.

- [ ] **Step 4: Run tests and commit**
    - Commit.

### Task 3: Refactor Scheduler

**Files:**
- Modify: `vllm_mlx/scheduler.py`

- [ ] **Step 1: Update initialization and attributes**
    - Replace `self.turn_cache: Optional[TurnPrefixCache]` and `self._prefix_cache` with `self.cache_manager: CacheManager`.
    - Update `_init_cache_bundle()` to instantiate `TurnCacheManager`.

- [ ] **Step 2: Refactor `_schedule_waiting`**
    - Replace the manual cache fetching and segmentation logic with a single call to `self.cache_manager.prepare_batch(requests, max_chunk_size)`.
    - Feed the returned segments into `BatchGenerator`.

- [ ] **Step 3: Refactor cleanup and abort logic**
    - In `_cleanup_finished()`, replace manual store/release logic with `self.cache_manager.cleanup_batch(finished=finished_requests, aborted=[])`.
    - In `_do_abort_request()`, use `self.cache_manager.cleanup_batch(finished=[], aborted=[request])`.

- [ ] **Step 4: Run tests and commit**
    - Commit.

### Task 4: Verification

- [ ] **Step 1: Unit Tests**
    - Add unit tests for `TurnCacheManager` covering full `Request` lifecycle (fetch $\to$ chunks $\to$ store $\to$ cleanup).
- [ ] **Step 2: Integration Tests**
    - Verify `Scheduler` correctly coordinates with `CacheManager` in a simulated batch.
- [ ] **Step 3: Final Run**
    - Run all project tests.
