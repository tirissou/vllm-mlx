# Batch Prefill: Exchange-Level Turn Boundaries (Option C)

**Date:** 2026-05-21
**Branch:** batch-prefill

## Problem

The turn cache trie has a structural mismatch between how it stores and how it looks up segments.

At **store time** (end of turn N), `response_tokens` is assembled as:

```
segments[-1].token_ids + output_token_ids
```

With per-message (turn-level) boundaries, `segments[-1]` covers only the generation prompt of turn N. So the stored node contains `gen_prompt_N + output_N` — the assistant exchange for turn N.

At **lookup time** (turn N+1), `messages_to_segments` splits `prompt_token_ids` using the same boundaries. The asst-N block now appears as a single segment `full_tokens[B_u(N) : B_a(N)]`. For the trie to find it, its hash must equal the hash of what was stored. This holds when `gen_prompt_N + output_N` equals `full_tokens[B_u(N) : B_a(N)]`, which in turn requires the generation prompt to include the leading `\n` separator — a template-specific assumption.

Separately, the current `_compute_turn_boundaries` main path (Qwen3) emits a boundary after **every** `<|im_end|>`, while the fallback path (non-Qwen3) already only emits boundaries after system and assistant messages. This inconsistency means the main path produces user-turn boundaries that the store/lookup logic never intended to use.

Finally, the init chunk of the boundary-aware prefill loop fires `mid_prefill_save` with `processed_tokens = B_sys` (the exact boundary position), but `on_prefill_checkpoint` needs `total_cached = B - 1` to key `_boundary_states[B]`. So the system node KV state is never captured.

## Solution: Approach 1 — Targeted Patch

Three isolated change sites, no refactoring of unrelated code.

### Change 1 — `_compute_turn_boundaries` main path (`batched.py`)

Add a counter that pairs each `<|im_end|>` token with its corresponding message role. Emit a boundary only when the role is `"system"` or `"assistant"`.

**Before:**
```python
boundaries = []
for i, tok in enumerate(full_tokens):
    if tok == im_end_id:
        boundaries.append(i + 1)
```

**After:**
```python
boundaries = []
im_end_count = 0
for i, tok in enumerate(full_tokens):
    if tok == im_end_id:
        if im_end_count < len(messages):
            role = messages[im_end_count].get("role", "")
            if role in ("system", "assistant"):
                boundaries.append(i + 1)
        im_end_count += 1
```

The fallback path is unchanged — it already implements this correctly.

**Effect on segments:** With a 3-turn conversation `[sys, u1, a1, u2, a2, u3]`, boundaries are now `[B_sys, B_a1, B_a2]` instead of `[B_sys, B_u1, B_a1, B_u2, B_a2, B_u3]`. `messages_to_segments` produces:

```
[sys] [u1+a1] [u2+a2] [u3+gen_prompt]
```

**Why the match is now structural:** `segments[-1] = prompt_token_ids[B_a(N-1):]` which starts with the `\n` separator followed by user N's full block and then the generation prompt. At the next turn, the exchange segment `prompt_token_ids[B_a(N-1):B_aN]` starts at the same position. `response_tokens = segments[-1] + output` equals that segment as long as `gen_prompt_N + output_N = asst_N_block` — which holds whenever the model generates through its stop token (`<|im_end|>`), the same condition both schemes require.

### Change 2 — Init chunk system-boundary capture (`scheduler.py`)

The init chunk lands exactly at `B_sys` and then calls:
```python
mid_prefill_save(uids[0], n_to_process, prompt_cache)
```

`on_prefill_checkpoint` checks `total_cached + 1 in _turn_boundaries`. With `total_cached = B_sys`, it checks `B_sys + 1`, which is not a boundary — so the system node KV is never saved.

Fix: pass `n_to_process - 1` when the init chunk was boundary-capped (`_first_chunk < budget`):

```python
if mid_prefill_save is not None and len(uids) == 1:
    _save_processed = n_to_process
    if _needs_boundary_split and _first_chunk < budget and n_to_process > 0:
        _save_processed = n_to_process - 1
    mid_prefill_save(uids[0], _save_processed, prompt_cache)
```

The condition `_first_chunk < budget` is True only when `_first_chunk` was capped to a boundary, so this is a no-op for prompts that didn't need boundary splitting.

The loop's subsequent iterations are unaffected: `partial["processed"]` stays at `n_to_process = B_sys`, so `_total_pos = B_sys` on the first loop step, `_next_b = B_a1`, and `_dist = (B_a1 - 1) - B_sys` caps correctly.

### Change 3 — Tests

**Two count updates in `tests/test_turn_boundaries_qwen3.py`:**

| Test | Old expected | New expected | Reason |
|------|-------------|-------------|--------|
| `test_exact_boundary_count_two_messages` (sys+user) | 2 | 1 | User boundary no longer emitted |
| `test_exact_boundary_count_multi_turn` (sys+u+a+u) | 4 | 2 | Only sys+asst boundaries |

**New round-trip test:**

Mock-based test (no real model required) that exercises the full fetch→prefill→store→fetch cycle:

1. Build a `TurnCacheAdapter` backed by a `TurnPrefixCache`
2. Construct a mock turn-1 request: `[sys, u1]` with `_turn_boundaries = [B_sys]`, synthetic `_boundary_states`, and `output_token_ids`
3. Call `store(request, cache)` — verifies it returns `True`
4. Construct turn-2 request: `[sys, u1, a1, u2]` with `_turn_boundaries = [B_sys, B_a1]`
5. Call `fetch(request)` — asserts `CacheHit.cached_tokens > 0` and `CacheHit.cache is not None`

## Scope

- `vllm_mlx/engine/batched.py`: ~8 lines changed in `_compute_turn_boundaries`
- `vllm_mlx/scheduler.py`: ~4 lines changed in the init chunk section of `_install_chunked_prefill`
- `tests/test_turn_boundaries_qwen3.py`: 2 integer literals updated
- `tests/test_turn_prefix_cache.py`: 1 new test added (~40 lines)

## Non-goals

- Unifying the main and fallback code paths in `_compute_turn_boundaries`
- Changes to `TurnCacheAdapter`, `TurnPrefixCache`, or `prefix_cache_adapters.py`
- Changes to the loop phase of `_install_chunked_prefill` (already correct)
