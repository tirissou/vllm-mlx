# Turn Boundary `<|im_end|>` Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile template-comparison logic in `_compute_turn_boundaries` with a direct scan of `full_tokens` for the `<|im_end|>` token, fixing wrong system boundaries and missing intermediate boundaries.

**Architecture:** Tokenize the full prompt once (without gen prompt), find `<|im_end|>` (token 151645 in Qwen3) followed by `\n` (token 198), and emit a boundary after each pair. Every segment becomes a complete self-contained message block from `<|im_start|>` to `<|im_end|>\n` inclusive. All downstream consumers (`_messages_to_segments`, `_boundary_states`, store phase) are unchanged.

**Tech Stack:** Python, `transformers` AutoTokenizer (Qwen3-0.6B for tests), `pytest`

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `vllm_mlx/engine/batched.py:976–1124` | Replace function body (~150 lines → ~35 lines) |
| Modify | `tests/test_turn_boundaries_qwen3.py` | Add exact-count and same-system-boundary tests; update docstring |

---

### Task 1: Add stricter failing tests

These tests will **fail** with the current implementation and **pass** after the fix.

**Files:**
- Modify: `tests/test_turn_boundaries_qwen3.py` (append to `TestTurnBoundariesQwen3` class)

- [ ] **Step 1.1: Append new test methods to the existing class**

Open `tests/test_turn_boundaries_qwen3.py`. Add these three methods inside `class TestTurnBoundariesQwen3`, after the last existing method (`test_very_long_conversation`):

```python
    def test_exact_boundary_count_two_messages(self):
        """system + user → exactly 2 boundaries (after system, after user)."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)
        assert len(boundaries) == 2, (
            f"Expected 2 boundaries (system, user), got {len(boundaries)}: {boundaries}"
        )

    def test_exact_boundary_count_multi_turn(self):
        """system + user + asst + user → exactly 4 boundaries."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)
        assert len(boundaries) == 4, (
            f"Expected 4 boundaries, got {len(boundaries)}: {boundaries}"
        )

    def test_same_system_boundary_for_different_user_messages(self):
        """Same system prompt → same system boundary regardless of user content."""
        eng = _make_engine_with_qwen3()
        sys_msg = "You are helpful."
        messages_a = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "hey hey"},
        ]
        messages_b = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "write a poem"},
        ]
        ba = eng._compute_turn_boundaries(messages_a)
        bb = eng._compute_turn_boundaries(messages_b)
        assert len(ba) >= 1 and len(bb) >= 1
        assert ba[0] == bb[0], (
            f"System boundary should be identical for same system prompt: "
            f"{ba[0]} != {bb[0]}"
        )

    def test_segments_end_with_im_end_newline(self):
        """Every non-tail segment decoded from boundaries ends with <|im_end|>\\n."""
        eng = _make_engine_with_qwen3()
        tok = eng._tokenizer
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ]
        full_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_tokens = tok.encode(full_prompt)
        boundaries = eng._compute_turn_boundaries(messages)

        prev = 0
        for b in boundaries:
            seg_tokens = full_tokens[prev:b]
            seg_text = tok.decode(seg_tokens)
            assert seg_text.endswith("<|im_end|>\n"), (
                f"Segment [{prev},{b}) should end with '<|im_end|>\\n', "
                f"got: {repr(seg_text[-20:])}"
            )
            prev = b
```

- [ ] **Step 1.2: Run the new tests to confirm they fail**

```bash
cd /Users/tibo/Projects/vllm-mlx && python -m pytest tests/test_turn_boundaries_qwen3.py::TestTurnBoundariesQwen3::test_exact_boundary_count_two_messages tests/test_turn_boundaries_qwen3.py::TestTurnBoundariesQwen3::test_exact_boundary_count_multi_turn tests/test_turn_boundaries_qwen3.py::TestTurnBoundariesQwen3::test_same_system_boundary_for_different_user_messages tests/test_turn_boundaries_qwen3.py::TestTurnBoundariesQwen3::test_segments_end_with_im_end_newline -v 2>&1 | tail -20
```

Expected: 4 failures (AssertionError on boundary counts / segment endings).

- [ ] **Step 1.3: Confirm existing tests still pass (baseline)**

```bash
cd /Users/tibo/Projects/vllm-mlx && python -m pytest tests/test_turn_boundaries_qwen3.py -v 2>&1 | tail -20
```

Expected: existing tests PASS, new 4 tests FAIL.

- [ ] **Step 1.4: Commit the failing tests**

```bash
git add tests/test_turn_boundaries_qwen3.py
git commit -m "test: add exact-count and same-system-boundary assertions for im_end scan"
```

---

### Task 2: Replace `_compute_turn_boundaries`

**Files:**
- Modify: `vllm_mlx/engine/batched.py:976–1124`

The function signature and docstring are kept; only the body changes.

- [ ] **Step 2.1: Replace the function body**

In `vllm_mlx/engine/batched.py`, replace everything from line 984 (the docstring opening `"""`) through line 1124 (the closing `return []` of the outer `except`) with:

```python
        """Compute token boundaries by scanning for <|im_end|> in the full prompt.

        Returns one boundary per message: the token position immediately after
        each <|im_end|>\\n pair in the tokenized prompt (without generation prompt).
        Each segment therefore covers exactly one complete message block, from
        <|im_start|> through <|im_end|>\\n inclusive.

        Returns [] if no system message or tokenizer lacks apply_chat_template.
        """
        if not messages or messages[0].get("role") != "system":
            return []

        tokenizer = self.tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        if not hasattr(tokenizer, "apply_chat_template"):
            return []

        try:
            boundary_kwargs = {**(chat_template_kwargs or {}), "add_generation_prompt": False}
            full_prompt = self._apply_chat_template(
                messages,
                tools=tools,
                num_images=num_images,
                num_audios=num_audios,
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
                    if (
                        nl_id is not None
                        and next_i < len(full_tokens)
                        and full_tokens[next_i] == nl_id
                    ):
                        next_i += 1
                    boundaries.append(next_i)

            logger.info(
                f"[turn_cache] _compute_turn_boundaries: "
                f"{len(full_tokens)} tokens, {len(boundaries)} boundaries"
            )
            return boundaries

        except Exception as e:
            logger.info(
                f"[turn_cache] _compute_turn_boundaries exception: "
                f"{type(e).__name__}: {e}"
            )
            return []
```

The new body replaces lines 984–1124. The function signature on lines 976–983 is untouched.

- [ ] **Step 2.2: Run all turn-boundary tests**

```bash
cd /Users/tibo/Projects/vllm-mlx && python -m pytest tests/test_turn_boundaries_qwen3.py -v 2>&1 | tail -30
```

Expected: all tests PASS (including the 4 new ones).

- [ ] **Step 2.3: Run the broader test suite**

```bash
cd /Users/tibo/Projects/vllm-mlx && python -m pytest tests/test_turn_prefix_cache.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 2.4: Commit the implementation**

```bash
git add vllm_mlx/engine/batched.py
git commit -m "fix: replace template-comparison boundary detection with im_end token scan

Eliminates two bugs:
- System boundary leaked into user message content (6 separate cache nodes
  for the same system prompt in production)
- Intermediate assistant boundaries never recorded for Qwen3 with thinking
  (template asymmetry caused prefix-token mismatch)

New approach: tokenize full prompt once, scan for <|im_end|> followed by
newline, emit a boundary after each pair. O(n), template-agnostic."
```

---

## Self-Review

**Spec coverage:**
- System boundary bug → covered by `test_same_system_boundary_for_different_user_messages` + implementation
- Intermediate boundary bug → covered by `test_exact_boundary_count_multi_turn` (checks 4 boundaries, which requires intermediate ones)
- Boundary position = after `<|im_end|>\n` → covered by `test_segments_end_with_im_end_newline`
- Downstream unchanged → no downstream files modified ✓

**Placeholder scan:** None found.

**Type consistency:** `boundaries` is `list[int]` throughout; `im_end_id` and `nl_id` are both `int | None`; consistent with existing return type annotation `-> list[int]`.
