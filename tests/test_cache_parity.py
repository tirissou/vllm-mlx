# SPDX-License-Identifier: Apache-2.0
"""
Regression tests: TurnPrefixCache + Scheduler must produce bit-identical
output_token_ids compared to a no-cache run for the same prompt at temperature=0.

Run: pytest tests/test_cache_parity.py -v
"""

import asyncio

import pytest

from vllm_mlx import AsyncEngineCore, EngineConfig, SamplingParams, SchedulerConfig

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _no_cache_config():
    return EngineConfig(
        scheduler_config=SchedulerConfig(
            enable_prefix_cache=False,
            use_turn_cache=False,
        )
    )


def _turn_cache_config():
    return EngineConfig(
        scheduler_config=SchedulerConfig(
            use_turn_cache=True,
            chunked_prefill_tokens=2048,
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_turn_boundaries(tokenizer, messages):
    """Compute turn boundaries by scanning for <|im_end|> in the full prompt.

    Mirrors BatchedEngine._compute_turn_boundaries without the server-side
    multimodal plumbing.
    """
    if not messages or messages[0].get("role") != "system":
        return []
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if not hasattr(tok, "apply_chat_template"):
        return []
    full_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tok.encode(full_prompt)
    if not full_tokens:
        return []
    im_end_id = (
        tok.convert_tokens_to_ids("<|im_end|>")
        if hasattr(tok, "convert_tokens_to_ids")
        else None
    )
    unk_id = getattr(tok, "unk_token_id", None)
    if im_end_id is None or im_end_id == unk_id:
        return []
    return [i + 1 for i, t in enumerate(full_tokens) if t == im_end_id]


async def _run_chat(engine, tokenizer, messages, max_tokens=20):
    """Submit one chat request; return output_token_ids when finished."""
    turn_boundaries = _compute_turn_boundaries(tokenizer, messages)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    rid = await engine.add_request(prompt, params, turn_boundaries=turn_boundaries)
    async for out in engine.stream_outputs(rid, timeout=30):
        if out.finished:
            return list(out.output_token_ids)
    return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_and_tokenizer():
    try:
        from mlx_lm import load

        return load("mlx-community/Qwen3-0.6B-8bit")
    except Exception as e:
        pytest.skip(f"Model not available: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestCacheParity:

    async def test_simple_conversation(self, model_and_tokenizer):
        """Cache must not alter output_token_ids for a single-turn conversation."""
        model, tokenizer = model_and_tokenizer
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        async with AsyncEngineCore(model, tokenizer, _no_cache_config()) as engine:
            await asyncio.sleep(0.05)
            tokens_no_cache = await _run_chat(engine, tokenizer, messages)

        async with AsyncEngineCore(model, tokenizer, _turn_cache_config()) as engine:
            await asyncio.sleep(0.05)
            # First request populates the cache.
            tokens_miss = await _run_chat(engine, tokenizer, messages)
            # Second identical request hits the cache.
            tokens_hit = await _run_chat(engine, tokenizer, messages)

        assert tokens_miss == tokens_no_cache, (
            f"Cache-miss run differs from no-cache baseline.\n"
            f"no_cache : {tokens_no_cache}\n"
            f"miss     : {tokens_miss}"
        )
        assert tokens_hit == tokens_no_cache, (
            f"Cache-hit run differs from no-cache baseline.\n"
            f"no_cache : {tokens_no_cache}\n"
            f"hit      : {tokens_hit}"
        )

    async def test_multi_turn_after_read(self, model_and_tokenizer):
        """Cache must not alter output_token_ids with two consecutive assistant turns.

        Mirrors the agentic "Read tool call" pattern where the assistant emits a
        tool-call message and then immediately a follow-up reply, producing back-to-back
        assistant turns in the conversation history:

            system → user₁ → assistant₁ (tool call) → assistant₂ (reply after read) → user₂

        Both assistant texts are generated live.  assistant₁ is produced from the
        opening exchange; assistant₂ is produced independently to stand in for the
        post-read continuation.
        """
        model, tokenizer = model_and_tokenizer

        system = {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses short.",
        }
        user_1 = {"role": "user", "content": "Name exactly one planet. One word only."}

        # Generate assistant₁ live (simulates the tool-call turn).
        async with AsyncEngineCore(model, tokenizer, _no_cache_config()) as engine:
            await asyncio.sleep(0.05)
            toks_a1 = await _run_chat(
                engine, tokenizer, [system, user_1], max_tokens=10
            )
        text_a1 = tokenizer.decode(toks_a1)

        # Generate assistant₂ live (simulates the reply after the file was read).
        # Prompted independently; the content just needs to be authentic model output.
        async with AsyncEngineCore(model, tokenizer, _no_cache_config()) as engine:
            await asyncio.sleep(0.05)
            toks_a2 = await _run_chat(
                engine,
                tokenizer,
                [
                    system,
                    {
                        "role": "user",
                        "content": "Is that planet larger than Earth? One word.",
                    },
                ],
                max_tokens=10,
            )
        text_a2 = tokenizer.decode(toks_a2)

        # Two consecutive assistant turns before the final user question.
        messages = [
            system,
            user_1,
            {"role": "assistant", "content": text_a1},
            {"role": "assistant", "content": text_a2},
            {"role": "user", "content": "Does it have rings? Yes or no only."},
        ]

        async with AsyncEngineCore(model, tokenizer, _no_cache_config()) as engine:
            await asyncio.sleep(0.05)
            tokens_no_cache = await _run_chat(engine, tokenizer, messages)

        async with AsyncEngineCore(model, tokenizer, _turn_cache_config()) as engine:
            await asyncio.sleep(0.05)
            tokens_miss = await _run_chat(engine, tokenizer, messages)
            tokens_hit = await _run_chat(engine, tokenizer, messages)

        assert tokens_miss == tokens_no_cache, (
            f"Cache-miss run differs from no-cache baseline (consecutive assistant turns).\n"
            f"no_cache : {tokens_no_cache}\n"
            f"miss     : {tokens_miss}"
        )
        assert tokens_hit == tokens_no_cache, (
            f"Cache-hit run differs from no-cache baseline (consecutive assistant turns).\n"
            f"no_cache : {tokens_no_cache}\n"
            f"hit      : {tokens_hit}"
        )


@pytest.mark.anyio
async def test_invalid_cache_falls_back_to_miss(model_and_tokenizer):
    """A corrupt reconstructed cache must trigger a miss fallback, not a crash."""
    import unittest.mock
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager

    model, tokenizer = model_and_tokenizer
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # In Python 3, accessing a staticmethod via the class returns a plain
    # function, so we just reference it directly (no __func__ needed).
    _orig_assemble = TurnCacheManager._assemble

    def _corrupt_assemble(kv_layers, rec_layers, group_size=64, bits=None):
        result = _orig_assemble(kv_layers, rec_layers, group_size, bits)
        # Corrupt batch dimension so validate() rejects it
        from mlx_lm.models.cache import KVCache
        import mlx.core as mx
        bad = KVCache()
        bad.keys = mx.zeros([2, 4, 3, 64])  # batch=2 → invalid
        bad.values = mx.zeros([2, 4, 3, 64])
        bad.offset = 3
        if result:
            result[0] = bad
        return result

    async with AsyncEngineCore(model, tokenizer, _turn_cache_config()) as engine:
        await asyncio.sleep(0.05)
        # Populate cache on first request
        await _run_chat(engine, tokenizer, messages)
        # Patch _assemble so the second request gets a corrupt cache
        with unittest.mock.patch.object(
            TurnCacheManager, "_assemble", staticmethod(_corrupt_assemble)
        ):
            # Should not crash; falls back to miss and produces valid output
            tokens = await _run_chat(engine, tokenizer, messages)

    assert len(tokens) > 0, "Engine must produce tokens even after corrupt cache fallback"
