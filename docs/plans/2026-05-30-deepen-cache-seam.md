# Deepen Cache Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the seam between `Scheduler` and `TurnPrefixCache` by moving the responsibility of populating `request._cache_state` to the `CacheManager` adapter.

**Architecture:**
- Modify `CacheManager` protocol in `vllm_mlx/prefix_cache_adapters.py` to change `fetch` signature from returning `CacheHit | None` to returning `bool`.
- Update `TurnCacheManager.fetch` to perform in-place mutation of `request._cache_state`.
- Refactor `Scheduler._schedule_waiting` in `vllm_mlx/scheduler.py` to use the new `fetch` signature.

**Tech Stack:** Python, `pytest`.

---

### Task 1: Update CacheManager Protocol and TurnCacheManager

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py:1-360`

- [ ] **Step 1: Write the failing test**

Since this is a core change, I'll first update the protocol and implementation. I'll look for existing tests that use `TurnCacheManager`.

```python
def test_turn_cache_manager_fetch_hit():
    # Mocking needed for complete test
    ...
```

Actually, I will do this in one go.

- [ ] **Step 2: Update protocol and TurnCacheManager implementation**

```python
class CacheManager(ABC):
    # ...
    @abstractmethod
    def fetch(self, request: Request) -> bool:
        """Look up a cached prefix for this request.

        On a hit, populates request._cache_state.turn_path and returns True.
        On a miss, populates request._cache_state for miss and returns False.
        """
        ...

# ...

class TurnCacheManager(CacheManager):
    # ...
    def fetch(self, request: Request) -> bool:
        segments = self.messages_to_segments(request)
        if not segments:
            cs = getattr(request, '_cache_state', None)
            if cs is not None:
                cs.hit_type = "miss"
                cs.remaining_tokens = request.prompt_token_ids
                cs.prefill_boundaries = self.boundaries(request)
            return False

        path, _ = self._inner.match(segments)
        if not path:
            self._inner.release(path)
            cs = getattr(request, '_cache_state', None)
            if cs is not None:
                cs.hit_type = "miss"
                cs.remaining_tokens = request.prompt_token_ids
                cs.prefill_boundaries = self.boundaries(request)
            return False

        cs = getattr(request, '_cache_state', None)
        if cs is not None:
            cs.turn_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            self._inner.release(path)
            if cs is not None:
                cs.hit_type = "miss"
                cs.remaining_tokens = request.prompt_token_ids
                cs.prefill_boundaries = self.boundaries(request)
            return False

        kv_data, rec_data = self._inner.collect_path_data(ancestor)
        reconstructed = self._assemble(kv_data, rec_data, self._kv_group_size, self._kv_bits)
        
        if cs is not None:
            cs.hit_type = 'hit'
            cs.cache = reconstructed
            cs.cached_tokens = ancestor.n_tokens
            cs.remaining_tokens = list(request.prompt_token_ids[cs.cached_tokens:])
            cs.prefill_boundaries = self.boundaries(request)
        
        return True
```

- [ ] **Step 3: Run tests and verify**

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py
git commit -m "refactor: update CacheManager protocol and TurnCacheManager to use in-place mutation"
```

### Task 2: Update Scheduler._schedule_waiting

**Files:**
- Modify: `vllm_mlx/scheduler.py:1420-1470`

- [ ] **Step 1: Write the failing test**

I'll use an existing scheduler test.

- [ ] **Step 2: Implement simplified logic in Scheduler._schedule_waiting**

```python
            if request._cache_state.remaining_tokens is None:
                hit_occurred = False
                if self._prefix_cache is not None:
                    hit_occurred = self._prefix_cache.fetch(request)
                    if hit_occurred:
                        self._log_cache_key("get", request.request_id, list(request.prompt_token_ids))
                        logger.info(f"[cache_fetch] request={request.request_id[:12]} HIT ...")
                    else:
                        self._log_cache_key("get", request.request_id, list(request.prompt_token_ids))
                        logger.info(f"[cache_fetch] request={request.request_id[:12]} MISS ...")
                else:
                    request._cache_state.hit_type = "miss"
                    request._cache_state.remaining_tokens = request.prompt_token_ids
                    request._cache_state.prefill_boundaries = []
```

- [ ] **Step 3: Run tests and verify**

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/scheduler.py
git commit -m "refactor: simplify Scheduler cache fetch logic"
```
