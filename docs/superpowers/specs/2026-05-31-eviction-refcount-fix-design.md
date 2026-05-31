# Eviction Fix: Ref-Count Nodes Inserted by `on_prefill_checkpoint`

**Date:** 2026-05-31
**Status:** Approved

## Problem

Once `TurnPrefixCache` reaches its memory budget, new insertions can silently corrupt the trie.

### Root cause

`_insert_node` adds the new node to the heap, then immediately calls `_evict_if_needed_unlocked()`. The new node has `ref_count=0` and `is_leaf=True`, making it evictable. When the cache is full of a deep linear chain (every existing node has at least one child, so no existing node is a leaf), the eviction loop skips all of them and the newly inserted node becomes the only valid eviction candidate — it self-evicts.

`_insert_node` returns the now-evicted, orphaned node. `on_prefill_checkpoint` stores it in `cs.turn_path` and uses it as `parent` for the next boundary insertion. That child is reachable from the orphan but not from `root`, so `match()` never finds it. Every subsequent insertion piles onto the orphaned subtree. The cache appears to accept stores but lookups always miss.

### Secondary issue (same fix)

Nodes appended to `cs.turn_path` by `on_prefill_checkpoint` have `ref_count=0`, unlike nodes placed there by `match()` (which increment `ref_count`). A later insert from a concurrent request can evict a `turn_path` node between checkpoint calls, with the same orphaning result.

## Invariant Being Enforced

> Any node referenced by an active request's `cs.turn_path` must have `ref_count >= 1`.

`match()` already upholds this for matched nodes. `on_prefill_checkpoint` must uphold it for newly inserted nodes.

## Changes

### 1. `TurnCacheManager.on_prefill_checkpoint()` — pin the new node

After inserting, increment `ref_count` before appending to `cs.turn_path`:

```python
new_node = self._inner.insert(
    parent, segment,
    kv_data=kv_layers or None,
    recurrent_data=rec_layers or None,
    is_system_prompt=is_sys,
)
new_node.ref_count += 1          # pin for duration of request
if cs is not None:
    cs.turn_path.append(new_node)
```

This makes the new node fail `is_evictable` during the eviction pass triggered inside `insert()`.

### 2. `TurnCacheManager.release()` — no change required

`release()` forwards `cs.turn_path` to `TurnPrefixCache.release()`, which calls `max(0, ref_count - 1)` for each node. After the fix, checkpoint-inserted nodes will have `ref_count=1`, so they drain to 0 correctly on release. The underflow guard handles any edge-case double-release safely.

### 3. `store()` — no change

`store()` receives the completed `cs.turn_path` (which includes checkpoint-pinned nodes) and uses `path[-1]` as parent. The node returned by `store()`'s own `insert()` call is not appended to `turn_path` and not used as a parent again, so it does not need pinning.

## Call-site audit (implementation task)

Confirm `TurnCacheManager.release()` is called on all request exit paths in the scheduler: normal completion, generation error, and request cancellation. The `max(0, ...)` guard in `TurnPrefixCache.release()` makes double-release safe, but a missing release leaks `ref_count` and prevents eviction of those nodes indefinitely.

## What this does NOT change

- Eviction order (LRU, lazy-deletion heap) is unchanged.
- The heap rebuild threshold (200 entries) is unchanged — a separate known issue.
- No changes to `is_permanent_checkpoint` handling — also a separate issue.
- Memory accounting is unchanged.

## Testing

- Add a test: fill cache to exactly budget with a linear chain, then insert one more node. Assert the new node is reachable via `match()` after the insert (i.e., was not self-evicted).
- Add a test: simulate a request that inserts nodes via `on_prefill_checkpoint` at multiple boundaries while at capacity. Assert all boundary nodes remain in the trie and are reachable.
- Existing eviction tests (`test_lru_evicts_oldest_leaf`, `test_pinned_node_not_evicted`, `test_eviction_cascade_to_parent`, `test_cascade_stops_at_sibling`) should continue to pass unchanged.
