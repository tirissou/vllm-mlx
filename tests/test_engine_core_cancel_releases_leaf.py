# SPDX-License-Identifier: Apache-2.0
"""Client-disconnect cancellation must release the cache leaf.

EngineCore submits requests through an async API. When the upstream consumer
cancels (websocket disconnect, abort_request call), the request flow eventually
calls Scheduler.abort_request → _do_abort_request → release(). This test wires
the full chain end-to-end and verifies that the cache observer sees the release.

Strategy: Two-request setup with explicit turn_boundaries.
1. Tokenize a prompt; split into a "system" prefix and "user" suffix via
   turn_boundaries=[B_sys]. Submit with use_turn_cache=True + small stride.
   The prefill will call on_prefill_checkpoint at B_sys, inserting the system
   segment node into the trie (and pinning it).  Completion then calls store(),
   adding the response leaf.
2. Submit the same prompt (same tokens, same turn_boundaries) as a second
   request. fetch() walks the trie, matches the system segment node, and pins
   its leaf for rid2.
3. Pre-assert pinned_leaf(rid2) is not None.
4. abort_request(rid2) → poll until step() drains the abort.
5. Assert pinned_leaf(rid2) is None.
"""

import asyncio

import pytest

TEST_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# A prompt long enough to split into two meaningful segments.
_SYS_TEXT = "You are a helpful assistant. Answer questions concisely and accurately."
_USER_TEXT = "What is the capital of France? Please explain briefly."


@pytest.fixture(scope="module")
def model_and_tokenizer():
    try:
        from mlx_lm import load

        return load(TEST_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load model {TEST_MODEL}: {e}")


@pytest.mark.anyio
async def test_client_disconnect_releases_pinned_leaf(model_and_tokenizer):
    """abort_request after a cache-hit must release the pinned leaf."""
    from vllm_mlx import AsyncEngineCore, SamplingParams
    from vllm_mlx.engine_core import EngineConfig
    from vllm_mlx.scheduler import SchedulerConfig

    model, tokenizer = model_and_tokenizer

    # Tokenize to get the exact boundary between system and user segments.
    sys_ids = tokenizer.encode(_SYS_TEXT)
    user_ids = tokenizer.encode(_USER_TEXT)
    all_ids = sys_ids + user_ids
    B_sys = len(sys_ids)
    turn_boundaries = [B_sys]

    config = EngineConfig(
        scheduler_config=SchedulerConfig(
            use_turn_cache=True,
            turn_cache_stride=8,     # small stride so short prompts trigger checkpoints
            chunked_prefill_tokens=8192,  # required by TurnPrefixCache
        )
    )
    params_full = SamplingParams(max_tokens=16, temperature=0.0)
    params_long = SamplingParams(max_tokens=200, temperature=0.0)  # long enough to abort mid-flight

    async with AsyncEngineCore(model, tokenizer, config) as engine:
        # ------------------------------------------------------------------ #
        # Step 1: seed the cache — run first request to completion so that
        # on_prefill_checkpoint stores the system-segment node and store()
        # stores the response leaf.
        # ------------------------------------------------------------------ #
        rid1 = await engine.add_request(
            all_ids,
            params_full,
            request_id="req-seed",
            turn_boundaries=turn_boundaries,
        )
        async for out in engine.stream_outputs(rid1, timeout=60):
            if out.finished:
                break

        # ------------------------------------------------------------------ #
        # Step 2: submit the same prompt as a second request. fetch() on a
        # cache HIT pins the matching leaf for rid2.
        # ------------------------------------------------------------------ #
        rid2 = await engine.add_request(
            all_ids,
            params_long,
            request_id="req-abort",
            turn_boundaries=turn_boundaries,
        )

        # Wait briefly for the engine to run at least one step so that
        # _schedule_waiting() calls fetch() and pins the leaf.
        # The step_interval is 1ms; 50ms is ~50 steps — enough for fetch()
        # but the request (max_tokens=200) will still be mid-flight.
        await asyncio.sleep(0.05)

        adapter = engine.engine.scheduler._prefix_cache
        assert adapter is not None, (
            "TurnCacheManager was not wired — check SchedulerConfig.use_turn_cache"
        )

        leaf_before = adapter.pinned_leaf(rid2)
        assert leaf_before is not None, (
            f"No leaf pinned for rid2 after cache-hit fetch(). "
            f"Prompt has {len(all_ids)} tokens, B_sys={B_sys}. "
            "Ensure the first request completed and on_prefill_checkpoint seeded the trie."
        )

        # ------------------------------------------------------------------ #
        # Step 3: abort the second request and drain through step().
        # ------------------------------------------------------------------ #
        await engine.abort_request(rid2)

        # Poll up to 2 s for the abort to be processed by scheduler.step(),
        # which calls _process_pending_aborts() → _do_abort_request() → release().
        deadline = asyncio.get_event_loop().time() + 2.0
        while adapter.pinned_leaf(rid2) is not None:
            if asyncio.get_event_loop().time() > deadline:
                break
            await asyncio.sleep(0.05)

        assert adapter.pinned_leaf(rid2) is None, (
            "Pinned leaf was NOT released after abort_request. "
            "EngineCore._cleanup_request must release the pinned leaf before "
            "removing the request from scheduler.requests."
        )
