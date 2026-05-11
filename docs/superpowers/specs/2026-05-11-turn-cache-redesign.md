# Turn Cache Redesign: Closed-Template Boundary Computation

**Date**: 2026-05-11  
**Status**: Approved for implementation  
**Replaces**: `_compute_prefix_boundary` + `_lcp_end` approach

---

## Motivation

The current TurnPrefixCache boundary computation is fragile in several ways:

- `_lcp_end` depends on knowing specific EOS token IDs (`<|im_end|>`, `<|eot_id|>`) — silently falls back to wrong boundaries if the model uses nonstandard tokens
- The 20-token backward scan window is hardcoded and arbitrary
- The `% budget` alignment trick guarantees only `prefix_boundary` is hit during chunked prefill; `sys_end_boundary` and intermediate `turn_boundaries` are hit by coincidence
- The alignment trick is disabled for batch size > 1, so states are often not captured at all
- Seven separate request fields (`prefix_boundary`, `sys_end_boundary`, `turn_boundaries`, `_conv_end_state`, `_turn_boundary_states`, `_sys_prompt_state`, `_mid_prefill_last_save`) implement what is conceptually one thing

---

## Design

### 1. Boundary Computation

Replace `_compute_prefix_boundary` with `_compute_turn_boundaries`:

```python
def _compute_turn_boundaries(
    messages: list[dict],
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    # Note: `tools` is intentionally absent from this signature.
    # Tool schemas are embedded in the system prompt in this codebase.
    # Any model architecture requiring separate template-level tool injection
    # (i.e. passing `tools` to apply_chat_template) would need to revisit
    # this signature.
```

**Algorithm**: LCP each closed-template tokenization against `full_tokens`.

```
full_tokens = tokenize(template(messages, add_generation_prompt=True))

B_sys = lcp(tokenize(template(messages[:1],       gen_prompt=False)), full_tokens)
B_1   = lcp(tokenize(template(messages[:3],       gen_prompt=False)), full_tokens)
B_2   = lcp(tokenize(template(messages[:5],       gen_prompt=False)), full_tokens)
...
B_k   = lcp(tokenize(template(messages[:2k+1],    gen_prompt=False)), full_tokens)
```

`messages[:2k+1]` = `[sys, u1, a1, ..., uk, ak]` — all messages through the k-th completed turn.

Each `B_k` is the exact token position just after `a_k`'s end-of-message token in `full_tokens`. No EOS token ID lookup, no backward scan, no hardcoded window. Template-specific trailing separators resolve automatically because we verify against the actual token sequence.

**Complexity**: O(N) tokenizer calls where N = number of completed turns. In practice, only turns beyond the current trie match depth need computing, typically 1–2 per request.

**Return value**: `[B_sys, B_1, ..., B_{N-1}]` where `B_{N-1}` corresponds to the last completed assistant turn before the current user message.

---

### 2. Prefill Alignment

Replace the `% budget` alignment trick with a boundary-aware chunk loop. For each chunk, check whether the next boundary falls within the current budget window:

```python
boundaries = sorted(_turn_boundaries)   # adjusted for cached_tokens
pos = cached_tokens

while pos < len(full_tokens):
    next_boundary = first B in boundaries where B > pos   # or None

    if next_boundary is not None and (next_boundary - pos) <= budget:
        chunk_size = next_boundary - pos   # land exactly on boundary
        prefill(full_tokens[pos : pos + chunk_size])
        capture_state()                    # _boundary_states[next_boundary]
    else:
        chunk_size = min(budget, len(full_tokens) - pos)
        prefill(full_tokens[pos : pos + chunk_size])

    pos += chunk_size
```

Properties:
- No chunk ever exceeds `budget` (memory safe)
- Every boundary in `[cached_tokens, len(full_tokens)]` is hit exactly, regardless of where it falls relative to `budget`
- Works for all layer types: transformer KV and Mamba/hybrid recurrent states alike — states are captured at the boundary, not trimmed from a later position
- Boundary-aware chunking applies to single-request prefills; batch prefills skip intermediate boundary capture (response node KV, captured at generation end, is always available regardless)

---

### 3. Trie Structure and State Storage

The trie structure and `_context_hash(parent_hash, segment.token_ids)` keying are **unchanged**.

What changes is how `segment.token_ids` is derived — from exact closed-template boundaries instead of LCP-heuristic boundaries:

```
sys_node.token_ids      = full_tokens[0      : B_sys]
conv_1_node.token_ids   = full_tokens[B_sys  : B_1  ]
conv_2_node.token_ids   = full_tokens[B_1    : B_2  ]
...
user_node.token_ids     = full_tokens[B_{N-1}:      ]   # structural, state=None
response_node.token_ids = full_tokens[B_{N-1}:      ] + output_token_ids
```

The response node concept is unchanged: inserted at generation end as a sibling of the user structural node, under `parent_before_user`, with state = `_extracted_cache`.

---

### 4. Request Lifecycle

**Scheduling (lookup)**

```
full_tokens  = tokenize(template(messages, add_generation_prompt=True))
boundaries   = _compute_turn_boundaries(messages, chat_template_kwargs)

walk trie using segment token slices at boundaries
→ deepest match → set cached_tokens, remaining_tokens, prompt_cache
store boundaries as _turn_boundaries on request
```

**Prefill**

Boundary-aware chunk loop (section 2). At each boundary hit, capture full state (KV + recurrent) into `_boundary_states[B_k]`.

**Generation end**

Extract `_extracted_cache` (covers full prompt + all output tokens). Unchanged from current.

**Cleanup (store)**

```
for each new segment beyond matched depth:
    sys node    → state = _boundary_states.get(B_sys)
    conv_k node → state = _boundary_states.get(B_k)
    user node   → state = None  (structural)

response node:
    token_ids = full_tokens[B_{N-1}:] + output_token_ids
    state     = _extracted_cache
    parent    = parent_before_user
```

**Request fields delta**

| Removed | Added |
|---|---|
| `prefix_boundary` | `_turn_boundaries: list[int]` |
| `sys_end_boundary` | `_boundary_states: dict[int, state]` |
| `turn_boundaries` | |
| `_conv_end_state` | |
| `_turn_boundary_states` | |
| `_sys_prompt_state` | |
| `_mid_prefill_last_save` | |

`_turn_cache_path`, `cached_tokens`, `remaining_tokens`, `prompt_cache`, `_extracted_cache` unchanged.

---

## Constraints and Notes

**System message assumed**: `_compute_turn_boundaries` assumes `messages[0]` is a system message. Requests without a system message fall back to no caching (return `[]`). This matches the existing behaviour of `_compute_prefix_boundary`.

**Thinking stripping**: If a client strips `<think>...</think>` from assistant content before the next turn, the closed-template tokenization of the stored assistant message will not match `full_tokens_N[B_{N-1}:] + output_tokens_N`. This produces a cache miss (not corruption) — the trie walk diverges at the response node and falls back to full prefill. This is a client-side concern outside the server's control.

**Sys prompt bootstrapping**: The sys node KV is populated naturally on the first single-request prefill that sees a given sys prompt. No explicit pre-warming mechanism is required. Under sustained batch load (all prefills at batch size > 1), the sys node may remain structural until a single-request prefill occurs.

**Tool schemas**: Embedded in the system prompt. See comment in `_compute_turn_boundaries`.

---

## Files Affected

- `vllm_mlx/engine/batched.py` — replace `_compute_prefix_boundary` with `_compute_turn_boundaries`
- `vllm_mlx/scheduler.py` — update `_mid_prefill_save` callback → boundary-aware chunk loop; update `_cleanup_finished` store loop; remove seven Request fields
- `vllm_mlx/request.py` — remove `prefix_boundary`, `sys_end_boundary`, `turn_boundaries`, `_conv_end_state`, `_turn_boundary_states`, `_sys_prompt_state`, `_mid_prefill_last_save`; add `_turn_boundaries`, `_boundary_states`
- `tests/test_turn_prefix_cache.py` — extend with boundary computation tests (no real model needed, tokenizer only)
