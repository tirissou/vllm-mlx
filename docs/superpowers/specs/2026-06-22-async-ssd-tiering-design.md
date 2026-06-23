# Design: Asynchronous SSD Cache Tiering

## Overview
This design introduces an asynchronous tiering mechanism for the SSD cache in `vllm-mlx`. The goal is to allow the system to promote data from the SSD tier to the RAM tier without blocking the main engine loop or stalling the scheduler's execution for other requests.

## Architecture: The "Async Island"
To avoid a full-scale asynchronous refactor of the `EngineCore`, we implement a "Worker-Owned Loop" pattern.

- **EngineCore**: Remains largely synchronous, using a `ThreadPoolExecutor` to run the `Scheduler`.
- **Worker Thread**: Initializes a dedicated `asyncio` event loop.
- **Scheduler**: Becomes natively asynchronous (`async def step()`). It runs within the worker thread's local event loop.
- **Concurrency**: Within the worker thread, the scheduler can `await` multiple concurrent cache promotions using standard `asyncio` primitives.

## Component Interfaces

### `CacheManager` (via `TurnCacheManager`)
The protocol is updated to support asynchronous operations:
- `async def fetch(self, request) -> bool`:
    - Returns `True` if the request is in RAM.
    - If in SSD, calls `await self._ssd_tier.async_promote(...)`, updates the trie, and returns `True`.
    - Returns `False` on a miss.
- `async def store(self, request, tokens, cache) -> bool`:
    - Handles RAM storage and triggers asynchronous "spill" to SSD.

### `Scheduler`
The scheduler's main loop is refactored to be awaitable:
- `async def step(self, max_retries: int = 1) -> SchedulerOutput`: The primary entry point.
- `async def _schedule_waiting(self) -> List[Request]`: Now awaits `self._prefix_cache.fetch(request)`.

### `SSDCacheTier`
Provides the low-level asynchronous I/O:
- `async def async_promote(self, ref: SSDRef) -> KVData`: Handles disk reads and deserialization.

## Data Flow: SSD Promotion
1. `Scheduler.step()` $\to$ `_schedule_waiting()` $\to$ `await prefix_cache.fetch(request)`.
2. `TurnCacheManager.fetch()` identifies `SSDRef` in a `TurnNode`.
3. `TurnCacheManager.fetch()` calls `await ssd_tier.async_promote(ref)`.
4. `SSDCacheTier` performs I/O on the worker thread's event loop.
5. Upon completion, `TurnCacheManager` updates the trie with real `KVData`.
6. `Scheduler` receives `True` and proceeds to schedule the request.

## Error Handling & Reliability

### Disk & I/O Failures
- **Quarantine**: Corrupt or unreadable files are marked in the SQLite index and deleted.
- **Fallback**: If promotion fails, `fetch()` returns `False`, and the `Scheduler` falls back to a standard prefill (treating it as a cache miss).

### RAM Budgeting & OOM Prevention
- **Atomic Reservation**: `TurnCacheManager` must successfully reserve RAM budget via the `MemoryCache` *before* initiating `async_promote`.
- **Guaranteed Release**: `asyncio.shield()` is used to ensure that even if a request is cancelled, the reserved RAM budget is released in a `finally` block.

### Shutdown & Integrity
- **Spill Queue**: The background `_writer_loop` is allowed to drain during shutdown.
- **Consistency**: SQLite WAL mode is used for atomic index updates.
- **Timeouts**: All `async_promote` calls are wrapped in `asyncio.wait_for` to prevent indefinite hangs.
