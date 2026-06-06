# Design: Deepen Cache Management

## Overview
This design aims to refactor the prefix cache management logic in `vllm-mlx`. Currently, the `Scheduler` is burdened with the low-level lifecycle and internal state management of the cache (e.g., `TurnNode` handles, `decoded_cache` tensors, and manual attribute cleanup). This leads to poor **Locality** and low **Depth** in the `CacheManager` interface.

The new design moves all implementation-specific cache logic behind a high-leverage, batch-oriented `CacheManager` interface.

## Architecture

### Core Components

#### `CacheManager` (The Seam)
The `CacheManager` is the primary interface for the `Scheduler`. It is an abstract base class (`ABC`) with concrete implementations like `TurnCacheManager`.

**Key Responsibilities:**
* **Batch-level Coordination**: Calculating optimal chunk sizes for a batch of requests based on boundary alignment.
* **Lifecycle Management**: Encapsulating the `fetch` $\to$ `store` $\to$ `release` $\to$ `cleanup` lifecycle.
* **State Management**: Managing the `Request._cache_state` object.

**Proposed Interface (Interface C - Caller-optimised):**
* `prepare_batch(requests: list[Request], max_chunk_size: int) -> list[list[Segment]]`: Performs batch-level chunking, boundary alignment, and cache lookups.
* `cleanup_batch(finished: list[Request], aborted: list[Request]) -> None`: Handles batch-level commit (store) and resource release (cleanup).

#### `Request` (The Data Carrier)
The `Request` object carries an opaque `_cache_state` (of type `RequestCacheState`). 

**Control Plane properties (accessible to Scheduler):**
* `next_boundary_distance: int`: Calculated by `CacheManager` during `prepare_batch`.
* `cache_hit_type: str`: (e.g., "hit", "miss") for telemetry.
* `cached_tokens: int`: For telemetry.

**Implementation properties (opaque to Scheduler):**
* `turn_path: list[TurnNode]`
* `decoded_cache: list[KVLayerSegment]`
* `_boundary_states`, `_sys_prompt_state`, etc.

#### `TurnPrefixCache` (The Implementation)
The existing trie-based storage mechanism. It remains the core data structure but is now strictly hidden behind the `CacheManager`.

### Data Flow: The Prefill Loop

1. **Batching**: The `Scheduler` identifies $N$ requests for prefill.
2. **Coordination**: The `Scheduler` calls `cache_manager.prepare_batch(requests, max_chunk_size)`.
3. **Internal Logic**:
    * `CacheManager` iterates through requests.
    * It queries the `TurnPrefixCache` to find hits.
    * It calculates `distances` to the next boundary for each request.
    * It calculates the `global_min_chunk_size = min(max_chunk_size, min(distances))`.
    * It slices each request's tokens into `Segment` objects based on this `global_min_chunk_size`.
    * It updates `request._cache_state` (including `turn_path` and `cached_tokens`).
4. **Execution**: The `Scheduler` receives a `list[list[Segment]]` and feeds them into the `BatchGenerator`.

### Data Flow: The Cleanup Loop

1. **Batch Completion**: A set of requests finishes or is aborted.
2. **Cleanup**: The `Scheduler` calls `cache_manager.cleanup_batch(finished, aborted)`.
3. **Internal Logic**:
    * For `finished` requests: The `CacheManager` extracts the `decoded_cache`, stores it in the trie, releases `TurnNode` handles, and clears all implementation state.
    * For `aborted` requests: The `CacheManager` performs an immediate, hard release of all resources.

## Error Handling and Resilience

* **Cache Corruption**: If a `BatchGenerator` error indicates cache corruption, the `Scheduler` triggers a `CacheManager.reset()` which clears the internal trie and all associated `Request` state.
* **Memory Pressure**: The `TurnPrefixCache` continues to handle its own LRU eviction and SSD spilling internally.

## Testing Strategy

1. **Unit Tests (`CacheManager`)**: These will become the primary tests for cache correctness. We will test the full `Request` lifecycle (from `fetch` to `cleanup`) purely through the `CacheManager` interface.
2. **Integration Tests (`Scheduler`)**: These will verify the coordination between the `Scheduler`'s batching logic and the `CacheManager`'s boundary logic (e.g., verifying that chunking correctly respects boundaries).
