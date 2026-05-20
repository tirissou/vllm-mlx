# ADR-0004: Scheduler.step() must remain synchronous

**Status:** Accepted  
**Date:** 2026-05-19

## Context

`Scheduler.step()` runs the full scheduling cycle: abort processing, cache fetch, BatchGenerator insertion, generation, response processing, cleanup. During cache redesign (ADR-0003) we considered making `PrefixCache.fetch()` async so SSD promotion could happen without blocking the worker thread.

## Decision

`step()` must remain synchronous. `PrefixCache.fetch()` must remain synchronous.

The constraint: MLX lazy operations (dequantize, reconstruct, concat) are enqueued on the stream of the thread that creates them. `BatchGenerator.next()` evaluates those ops on the same worker thread. If `fetch` awaits anything, control returns to the event loop and subsequent MLX reconstruction executes on the event loop thread — a different stream. `BatchGenerator.next()` then evaluates ops it did not enqueue, producing incorrect or undefined results.

This is not a performance tradeoff. It is a correctness constraint imposed by MLX's stream model.

## Consequences

SSD promotion cannot block-avoid by going async inside `step()`. The two valid approaches are:

1. **Synchronous disk read inside `fetch`** — simple, already done today in `_try_promote_ssd_for_request`. Blocks the worker thread for the duration of the read.
2. **Background promotion loop outside `step()`** — `SSDOffloadedCache` owns an async loop that pre-loads SSD entries to the RAM cache. `fetch` finds a RAM hit on the next cycle. Worker thread never blocks on disk I/O; SSD-pending requests wait one extra scheduling cycle.

`SSDOffloadedCache` uses approach 2 for `MemoryAwarePrefixCache` (full eviction). `TurnPrefixCache` uses approach 1 via the synchronous `on_promote` hook (same behaviour as today).

Do not re-propose async `fetch`, async `_schedule_waiting`, or any await inside `step()`. The MLX stream constraint rules these out regardless of how the cache layer is structured.
