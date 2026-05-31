# Design: Deepen the seam between the Scheduler and TurnPrefixCache

Date: 2026-05-30

## Status

Draft

## Context

The current interaction between the `Scheduler` and the `TurnPrefixCache` (via the `TurnCacheManager` adapter) is a shallow seam. The `Scheduler` calls `fetch()` on the adapter, receives a `CacheHit` object, and then manually copies multiple attributes (`hit_type`, `cache`, `cached_tokens`, `remaining_tokens`) from that object into the `request._cache_state`. Furthermore, the `Scheduler` makes a redundant call to `self._prefix_cache.boundaries(request)` because the `CacheHit` result isn't fully used to populate the request state. This causes the `Scheduler` to leak knowledge about the internal structure of the cache response and the request's cache state.

## Decision

We will deepen the seam by moving the responsibility of populating the `request._cache_state` from the `Scheduler` to the `CacheManager` adapter.

### 1. Protocol Change

The `CacheManager` protocol's `fetch` method is modified to update the `request` object in-place.

**Old Signature:**
`fetch(self, request) -> CacheHit | None`

**New Signature:**
`fetch(self, request: Request) -> bool`

The `bool` return value indicates whether a cache hit occurred.

### 2. Scheduler Refactoring

The `Scheduler._schedule_waiting` method is simplified. Instead of manually wiring the `CacheHit` object into the `request._cache_state`, it now simply calls `fetch(request)` and reacts to the boolean result.

```python
# New simplified logic in Scheduler._schedule_waiting
if self._prefix_cache is not None:
    hit_occurred = self._prefix_cache.fetch(request)
    if hit_occurred:
        # request._cache_state is already fully populated by the adapter
        self._log_cache_key("get", request.request_id, list(request.prompt_token_ids))
        logger.info(f"[cache_fetch] request={request.request_id[:12]} HIT ...")
    else:
        # request._cache_state is already set to a valid miss state
        self._log_cache_key("get", request.request_id, list(request.prompt_token_ids))
        logger.info(f"[cache_fetch] request={request.request_id[:12]} MISS ...")
else:
    # fallback for no cache configured
    request._cache_state.hit_type = "miss"
    request._cache_state.remaining_tokens = request.prompt_token_ids
    # ...
```

### 3. TurnCacheManager Implementation

The `TurnCacheManager` implementation of `fetch` is updated to populate the `request._cache_state` in-place.

- On a **hit**: It sets `hit_type`, `cache`, `cached_tokens`, `remaining_tokens`, `prefill_boundaries`, and `turn_path`.
- On a **miss**: It sets `hit_type="miss"`, `remaining_tokens=request.prompt_token_ids`, and `prefill_boundaries`.

## Consequences

### Pros
- **Increased Leverage:** The `Scheduler` becomes thinner and more agnostic to cache implementation details.
- **Improved Locality:** The logic for how a cache interaction affects a request's state is concentrated within the adapter.
- **Reduced Redundancy:** Eliminates redundant calls to `boundaries()`.

### Cons
- **Breaking Change:** The `CacheManager` protocol is changed, requiring all implementing adapters (like `PagedCacheManager` or future ones) to follow the new in-place mutation pattern.
- **Mutation:** The `fetch` method now mutates the `request` object, which is a change in behavior from the previous functional-style `CacheHit` return.

## Implementation Plan

1. Update `CacheManager` protocol in `vllm_mlx/prefix_cache_adapters.py`.
2. Update `TurnCacheManager.fetch` in `vllm_mlx/prefix_cache_adapters.py`.
3. Update `Scheduler._schedule_waiting` in `vllm_mlx/scheduler.py`.
4. Verify with tests.
