# TurnPrefixCache: KV-Only Model Support

**Date:** 2026-05-23  
**Status:** Approved

## Problem

`TurnPrefixCache` was designed for hybrid models (e.g. Mamba) that have both KV attention layers and recurrent SSM layers. It uses the presence of `recurrent_state` on a `TurnNode` as the signal that a node is a valid resume point. This makes it unusable for KV-only models (Gemma4, Qwen3, etc.):

- `find_checkpoint_ancestor` requires `recurrent_state is not None` → always returns `None` for KV-only models
- `TurnCacheAdapter.fetch` sees `ancestor = None` → returns `cached_tokens=0`, cache is never used
- `_retrieve_full_cache` has `assert node.recurrent_state is not None` → crashes if reached

## Design

### Core idea

Add a single boolean `has_recurrent_state` to `TurnPrefixCache`. It starts `False`, is set to `True` the first time a node with non-empty recurrent state is inserted or loaded, and is never reset (not even by `clear()`). This flag becomes the branch point for "what makes a node a valid resume point."

For hybrid models (flag `True`): a node is resumable when it has real recurrent state — same as today.  
For KV-only models (flag `False`): every node with non-empty `kv_arrays` is resumable.

### Write sites (three, all set `True`, never `False`)

**`_insert_node`** — receives `recurrent_state` directly. Set `self.has_recurrent_state = True` when `recurrent_state is not None`. Runs inside `self._lock`.

**`_insert_legacy`** — calls `_split_cache_arrays` which returns `(kv, recur)`. Set `self.has_recurrent_state = True` when `recur` is non-empty. Runs inside `self._lock`.

**`load`** — after all nodes are linked, scan them: if any node has `recurrent_state is not None and not isinstance(recurrent_state, SSDRef) and len(recurrent_state) > 0`, set `True`.

### Read sites (two)

**`find_checkpoint_ancestor`**

```python
# KV-only path
if not self.has_recurrent_state:
    for node in reversed(path):
        if node.kv_arrays and not isinstance(node.kv_arrays, SSDRef):
            return node
    return None

# Hybrid path (unchanged)
for node in reversed(path):
    if _has_real_recurrent(node):
        return node
return None
```

**`_retrieve_full_cache`** — remove `assert node.recurrent_state is not None` (line 456). For a KV-only node, `recurrent` will be `[]`; the existing `_reassemble_cache_fn` already handles this correctly because `recurrent_indices` will be `()`.

### What does not change

- `match()` — `has_recurrent` return value stays as the per-node check it already is; callers already handle `False`
- Intermediate-node pruning (`parent.recurrent_state = None`) — no-op for KV-only since `recurrent_state` is already `None`; no guard needed
- `_split_cache_arrays` — already naturally produces `recurrent=[]` for KV-only models
- `clear()` — does not touch `has_recurrent_state`

## Files affected

| File | Change |
|------|--------|
| `vllm_mlx/turn_prefix_cache.py` | Add `has_recurrent_state` field; update `_insert_node`, `_insert_legacy`, `load`, `find_checkpoint_ancestor`, `_retrieve_full_cache` |

No changes to `prefix_cache_adapters.py`, `scheduler.py`, or any config class.

## Testing

- Unit test: insert KV-only nodes → `has_recurrent_state` stays `False` → `find_checkpoint_ancestor` returns deepest KV node
- Unit test: insert hybrid nodes → `has_recurrent_state` becomes `True` → existing checkpoint logic applies
- Unit test: `clear()` does not reset `has_recurrent_state`
- Unit test: `load()` from disk with recurrent nodes sets flag; without recurrent nodes leaves it `False`
- Integration test: Gemma4 conversation with `TurnCacheAdapter` — second turn gets a cache hit with `cached_tokens > 0`
