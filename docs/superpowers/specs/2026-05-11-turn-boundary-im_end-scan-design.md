# Turn Boundary Computation: `<|im_end|>` Token Scan

**Date:** 2026-05-11
**Status:** Approved

## Problem

`_compute_turn_boundaries` in `vllm_mlx/engine/batched.py` computes turn boundaries by repeatedly applying the chat template to message prefixes and comparing tokenized outputs. This approach has two bugs for Qwen3:

1. **System boundary is too late.** The system boundary falls back to `len(one_dummy)` as a character offset into `full_prompt`. But `one_dummy` has an empty user message while `full_prompt` has real user content, so the offset extends past `<|im_start|>user\n` and into the actual user text. In production this causes one separate system trie node per distinct first-user-message opener (6 nodes observed for the same system prompt).

2. **Intermediate boundaries are never recorded.** When the template is applied to a prefix ending on an assistant message, Qwen3 injects `<think>\n\n</think>\n\n` before the assistant content for that final position. This makes `prefix_tokens != full_tokens[:N]`, so `matches = False` for every assistant turn, and no intermediate boundaries are added.

## Solution

Replace the entire boundary computation body with a direct scan of the token sequence for the `<|im_end|>` token. Every `<|im_end|>` in the prompt marks the end of a message (system, user, or assistant) — no template re-application or string comparison needed.

### Boundary position

The Qwen3 template always places `\n` immediately after `<|im_end|>`. Each segment should own its closing delimiter, so the boundary is set at the position **after** `<|im_end|>\n` (i.e. `i+2` when the next token is the newline, `i+1` otherwise as a safe fallback).

Resulting segment structure for an N-turn conversation:

```
[0,          B0) = <|im_start|>system\n{sys}<|im_end|>\n
[B0,         B1) = <|im_start|>user\n{u1}<|im_end|>\n
[B1,         B2) = <|im_start|>assistant\n<think>...<|im_end|>\n
...
[B_{2N-2}, B_{2N-1}) = <|im_start|>user\n{uN}<|im_end|>\n
[B_{2N-1},  end)     = <|im_start|>assistant\n<think>\n  ← gen-prompt tail
```

Each segment is self-contained. The tail (gen-prompt tokens) is short (3–5 tokens) and forms the final "user" segment in the trie.

## Implementation

### `_compute_turn_boundaries` (`batched.py:976`)

Replace the function body with:

```python
tokenizer = self.tokenizer
if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer

if not messages or messages[0].get("role") != "system":
    return []
if not hasattr(tokenizer, "apply_chat_template"):
    return []

try:
    boundary_kwargs = {**(chat_template_kwargs or {}), "add_generation_prompt": False}
    full_prompt = self._apply_chat_template(
        messages, tools=tools,
        num_images=num_images, num_audios=num_audios,
        chat_template_kwargs=boundary_kwargs,
        enable_thinking=enable_thinking,
    )
    full_tokens = tokenizer.encode(full_prompt)
    if not full_tokens:
        return []

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        return []

    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = nl_tokens[0] if nl_tokens else None

    boundaries = []
    for i, tok in enumerate(full_tokens):
        if tok == im_end_id:
            next_i = i + 1
            if nl_id is not None and next_i < len(full_tokens) and full_tokens[next_i] == nl_id:
                next_i += 1
            boundaries.append(next_i)

    logger.info(f"[turn_cache] _compute_turn_boundaries: {len(full_tokens)} tokens, {len(boundaries)} boundaries")
    return boundaries

except Exception as e:
    logger.info(f"[turn_cache] _compute_turn_boundaries exception: {type(e).__name__}: {e}")
    return []
```

### Nothing else changes

- `_messages_to_segments` already iterates all boundaries generically — no changes needed.
- `_boundary_states` save logic triggers on `total_cached in _turn_boundaries` — no changes needed.
- Store phase uses `_turn_boundaries[-1]` as `last_boundary`; `response_tokens = prompt_tokens[last_boundary:] + output_token_ids` now equals gen-prompt tokens + model output, which matches the formatted assistant segment in subsequent requests.

## Expected outcome

All conversations sharing the same system prompt will share a single system trie node (down from one per distinct first-user-message opener). Intermediate assistant-turn boundaries will be recorded correctly, enabling per-turn cache hits for multi-turn conversations.

## Tests to update

- `tests/test_turn_boundaries_qwen3.py` — update expected boundary values to match `<|im_end|>\n` positions
- `tests/test_turn_prefix_cache.py` — verify segment structure with new boundaries
