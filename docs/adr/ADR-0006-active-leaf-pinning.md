# ADR-0006: Active Leaf pinning + leaf-only eviction as paired invariants

**Status:** Accepted
**Date:** 2026-06-11

## Context

The `TurnCacheManager` seam (see `2026-06-08-deepen-cache-manager-seam-design.md`) introduced an "Active Leaf" pinning model: for each in-flight request, the manager keeps a single entry in `_pinned_leaves` mapping `request_id -> leaf node`. `fetch`, `store`, and `release` advance or clear that pin in lockstep with trie mutations.

Two questions were left unanswered by that work:

1. **Why is pinning just the leaf sufficient?** With no further constraints, an interior node could be evicted out from under a request that holds a downstream leaf pin.
2. **Where does the pin advance during a prefill that spans multiple checkpoints?** The original spec described `fetch`/`store`/`release` but did not formalise `on_prefill_checkpoint`.

The answer to both is the same: leaf-only eviction makes Active Leaf safe with O(1) bookkeeping, and every node-inserting call site (now including `on_prefill_checkpoint`) must advance the pin atomically with the insert. Without writing this down, a future contributor could break either half of the pair — e.g. add a node-inserting code path that forgets to advance the pin, or relax eviction to include interior nodes — and not realise the other half has now become unsafe.

## Decision

The two invariants below are paired and load-bearing. Changing one without re-examining the other is a correctness bug.

### Invariant 1 — Active Leaf

`TurnCacheManager._pinned_leaves: dict[request_id, TurnNode]` is the sole owner of `Request -> pinned node` state. At most one node per request is pinned at any time, and that node is always a leaf of the trie at the moment of pinning.

The four mutators are `fetch`, `store`, `on_prefill_checkpoint`, and `release`. Each one advances or clears the pin in the same shape:

1. Snapshot the current pinned leaf (if any).
2. Mutate the trie (insert / match / nothing).
3. On success, release the snapshot and pin the new leaf; on failure, do not touch ref counts — the snapshot is still correctly pinned because we never released it.

This gives implicit rollback: the insert-then-advance ordering means an exception from the trie mutation leaves `_pinned_leaves` and all `ref_count` values untouched. No try/except needed.

### Invariant 2 — Leaf-only eviction

`TurnPrefixCache` only evicts nodes where `len(children) == 0` AND `ref_count == 0`. Interior nodes are never selected for eviction regardless of `last_used` age.

This is what makes Active Leaf safe: a request's pinned leaf is enough to protect its entire ancestor chain, because the ancestor chain cannot be leaves while the descendant exists. Pinning ancestors would be redundant.

## Consequences

- All node-inserting code paths inside `TurnCacheManager` (currently `store` and `on_prefill_checkpoint`; any future addition) must follow the snapshot → insert → release-old → pin-new shape. Reviews should flag any other shape as a bug.
- Eviction policy in `TurnPrefixCache._evict_if_needed_unlocked` must continue to filter on `is_evictable` (the property that combines leaf-status and ref_count). Adding an "evict any unused node" mode would silently break Active Leaf.
- Scheduler code MUST NOT mutate `_pinned_leaves` or call `inner.match` / `inner.release` directly. The seam established in `2026-06-08-deepen-cache-manager-seam-design.md` owns all of this state.

## Alternatives considered

- **Pin every node in the matched path.** Rejected in the original deepen-the-seam spec because it requires the Scheduler to track full paths and bumps per-request bookkeeping from O(1) to O(path length). Active Leaf + leaf-only eviction give the same protection cheaper.
- **Evict any non-pinned node, including interior.** Rejected because it would force pinning the whole path, defeating Active Leaf.
