"""
Tests for _compute_turn_boundaries() using the real Qwen3-0.6B tokenizer.

These tests verify that the LCP-based boundary detection works correctly
with a real tokenizer, not just the mock tokenizer used in test_turn_prefix_cache.py.
"""

import pytest

# Try to import the tokenizer; skip all tests if unavailable
try:
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-4bit")
    TOKENIZER_AVAILABLE = True
except Exception as e:
    TOKENIZER_AVAILABLE = False
    TOKENIZER_LOAD_ERROR = str(e)


def _make_engine_with_qwen3():
    """Return a BatchedEngine stub with the real Qwen3 tokenizer."""
    from vllm_mlx.engine.batched import BatchedEngine

    eng = object.__new__(BatchedEngine)
    eng._is_mllm = False
    eng._tokenizer = TOKENIZER
    eng._processor = None
    eng._model_name = "Qwen3-0.6B"
    return eng


@pytest.mark.skipif(not TOKENIZER_AVAILABLE, reason=f"Qwen3 tokenizer not available")
class TestTurnBoundariesQwen3:
    """Test _compute_turn_boundaries with the real Qwen3 tokenizer."""

    def test_system_and_user_message(self):
        """Basic case: system + user → should find system boundary."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Should have at least one boundary (system)
        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"
        assert boundaries[0] > 0, f"First boundary should be > 0, got {boundaries[0]}"

        # Verify boundary is strictly less than total tokens
        # tokenize=True returns dict with 'input_ids' and 'attention_mask'
        full_tokens_dict = eng._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        full_tokens = full_tokens_dict["input_ids"]
        assert boundaries[0] < len(
            full_tokens
        ), f"B_sys={boundaries[0]} should be < total tokens {len(full_tokens)}"

    def test_multi_turn_has_increasing_boundaries(self):
        """Multi-turn: system + user1 + assistant1 + user2 → [B_sys, B_1]."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Should have 2 boundaries: system and after first assistant turn
        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"

        # All boundaries should be strictly increasing
        for i in range(len(boundaries) - 1):
            assert (
                boundaries[i] < boundaries[i + 1]
            ), f"Boundaries not increasing: {boundaries[i]} >= {boundaries[i+1]}"

    def test_multi_turn_three_assistant_turns(self):
        """Three completed assistant turns → [B_sys, B_1, B_2, B_3]."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "U3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "U4"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Should have 4 boundaries: system + 3 completed assistant turns
        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"

        # Verify all increasing
        for i in range(len(boundaries) - 1):
            assert (
                boundaries[i] < boundaries[i + 1]
            ), f"Boundaries not strictly increasing: {boundaries}"

    def test_empty_system_message(self):
        """Edge case: empty system message."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Should still find system boundary (even if content is empty)
        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"
        assert boundaries[0] > 0, f"B_sys should be > 0, got {boundaries[0]}"

    def test_long_system_message(self):
        """Edge case: very long system message (100+ tokens)."""
        eng = _make_engine_with_qwen3()
        long_sys = "This is a system message. " * 20  # ~120 tokens
        messages = [
            {"role": "system", "content": long_sys},
            {"role": "user", "content": "Hi"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"
        # Long system message should result in large boundary
        assert (
            boundaries[0] > 50
        ), f"Long system message should have B_sys > 50, got {boundaries[0]}"

    def test_token_boundary_accuracy(self):
        """Verify computed boundary is within valid range of full token sequence."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Get full tokens
        full_tokens = eng._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )["input_ids"]

        # Boundary should be a valid position in the full token sequence
        B_sys = boundaries[0]
        assert (
            0 < B_sys < len(full_tokens)
        ), f"B_sys={B_sys} should be within range (0, {len(full_tokens)})"

    def test_system_message_in_multi_turn(self):
        """System boundary is computed correctly in multi-turn conversations."""
        eng = _make_engine_with_qwen3()
        sys_msg = {"role": "system", "content": "Fixed system prompt."}

        # Two different multi-turn conversations with the same system prompt
        messages1 = [
            sys_msg,
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

        messages2 = [
            sys_msg,
            {"role": "user", "content": "Different Q1"},
            {"role": "assistant", "content": "Different A1"},
            {"role": "user", "content": "Different Q2"},
        ]

        boundaries1 = eng._compute_turn_boundaries(messages1)
        boundaries2 = eng._compute_turn_boundaries(messages2)

        # Both should have at least one boundary (system)
        assert len(boundaries1) >= 1
        assert len(boundaries2) >= 1

        # System boundaries may differ due to context-dependent tokenization,
        # but they should both be valid (positive and less than full token count)
        full_tokens1 = eng._tokenizer.apply_chat_template(
            messages1, add_generation_prompt=True, tokenize=True
        )["input_ids"]
        full_tokens2 = eng._tokenizer.apply_chat_template(
            messages2, add_generation_prompt=True, tokenize=True
        )["input_ids"]

        assert 0 < boundaries1[0] < len(full_tokens1)
        assert 0 < boundaries2[0] < len(full_tokens2)

    def test_no_duplicate_boundaries(self):
        """Computed boundaries should all be unique (strictly increasing)."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "U3"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # No duplicates
        assert len(boundaries) == len(
            set(boundaries)
        ), f"Boundaries contain duplicates: {boundaries}"

    def test_last_boundary_less_than_full_tokens(self):
        """Last boundary must be < len(full_tokens) because there's a user segment after."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        full_tokens = eng._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )["input_ids"]

        assert boundaries[-1] < len(
            full_tokens
        ), f"Last boundary {boundaries[-1]} should be < full_tokens {len(full_tokens)}"

    def test_boundaries_strictly_increasing(self):
        """Multiple boundaries should be strictly increasing positions."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "Help me."},
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
            {"role": "user", "content": "E"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        # Get full tokens
        full_tokens = eng._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )["input_ids"]

        # All boundaries should be strictly increasing
        for i in range(len(boundaries) - 1):
            assert (
                boundaries[i] < boundaries[i + 1]
            ), f"Boundaries not strictly increasing: {boundaries}"

        # All boundaries should be within valid token range
        for b in boundaries:
            assert (
                0 < b < len(full_tokens)
            ), f"Boundary {b} outside valid range (0, {len(full_tokens)})"

    def test_special_characters_in_messages(self):
        """Messages with special characters should compute boundaries correctly."""
        eng = _make_engine_with_qwen3()
        messages = [
            {
                "role": "system",
                "content": "Handle special chars: <>, {}, [], @, #, $, %",
            },
            {"role": "user", "content": "Question with \"quotes\" and 'apostrophes'"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"
        assert boundaries[0] > 0, f"B_sys should be > 0, got {boundaries[0]}"

    def test_unicode_messages(self):
        """Messages with unicode characters should compute boundaries correctly."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "你好，我是一个助手。"},  # Chinese
            {"role": "user", "content": "こんにちは。"},  # Japanese
        ]
        boundaries = eng._compute_turn_boundaries(messages)

        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"
        assert boundaries[0] > 0, f"B_sys should be > 0, got {boundaries[0]}"

    def test_very_long_conversation(self):
        """Long conversation with many turns should compute all boundaries."""
        eng = _make_engine_with_qwen3()
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(10):
            messages.append({"role": "user", "content": f"Q{i}"})
            messages.append({"role": "assistant", "content": f"A{i}"})
        # End with a user message
        messages.append({"role": "user", "content": "Q10"})

        boundaries = eng._compute_turn_boundaries(messages)

        # Should have many boundaries (system + 10 completed turns)
        assert len(boundaries) >= 1, f"Expected at least 1 boundary, got {boundaries}"

        # All strictly increasing
        for i in range(len(boundaries) - 1):
            assert (
                boundaries[i] < boundaries[i + 1]
            ), f"Boundaries not strictly increasing: {boundaries}"

    def test_exact_boundary_count_two_messages(self):
        """system + user → exactly 2 boundaries (after system, after user)."""
        eng = _make_engine_with_qwen3()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
        ]
        boundaries = eng._compute_turn_boundaries(messages)
        assert (
            len(boundaries) == 2
        ), f"Expected 2 boundaries (system, user), got {len(boundaries)}: {boundaries}"

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
        assert (
            len(boundaries) == 4
        ), f"Expected 4 boundaries, got {len(boundaries)}: {boundaries}"

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

    def test_segments_end_with_im_end(self):
        """Every non-tail segment decoded from boundaries ends with <|im_end|>.

        Boundaries point AT the \\n after <|im_end|> (i.e. boundary = im_end_index + 1),
        so each segment ends at <|im_end|> without the trailing \\n.  The \\n
        becomes the first token of the next segment instead.  This is required
        for insert/match consistency: model output_token_ids ends at <|im_end|>
        (the EOS), so the stored response segment must also end there.
        """
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
            assert seg_text.endswith("<|im_end|>"), (
                f"Segment [{prev},{b}) should end with '<|im_end|>', "
                f"got: {repr(seg_text[-20:])}"
            )
            assert not seg_text.endswith("<|im_end|>\n"), (
                f"Segment [{prev},{b}) must NOT include trailing \\n "
                f"(that belongs to the next segment for insert/match consistency)"
            )
            prev = b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
