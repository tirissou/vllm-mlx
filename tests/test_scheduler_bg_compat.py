# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Scheduler compatibility with mlx-lm >=0.31.x."""

import pytest
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture(scope="module")
def qwen3_small():
    try:
        from mlx_lm import load
        return load("mlx-community/Qwen3-0.6B-4bit")
    except Exception:
        pytest.skip("mlx-community/Qwen3-0.6B-4bit not available")


def _run_to_completion(scheduler, max_steps=200):
    """Step until no requests remain or max_steps reached."""
    for _ in range(max_steps):
        output = scheduler.step()
        if not scheduler.has_requests():
            return output
    return None


@pytest.mark.slow
def test_single_request_produces_tokens(qwen3_small):
    model, tokenizer = qwen3_small
    scheduler = Scheduler(model, tokenizer, SchedulerConfig())
    scheduler.add_request(Request(
        request_id="r1",
        prompt="Hello",
        sampling_params=SamplingParams(max_tokens=8),
    ))
    _run_to_completion(scheduler)
    req = scheduler.requests.get("r1") or scheduler._finished_requests.get("r1")
    assert req is not None and req.num_output_tokens > 0


@pytest.mark.slow
def test_mid_prefill_save_fires_before_prefill_completes(qwen3_small):
    """Prefix cache must receive a checkpoint mid-prefill, not just at the end."""
    model, tokenizer = qwen3_small
    checkpoints = []

    config = SchedulerConfig(
        mid_prefill_save_interval=64,
        use_memory_aware_cache=True,
        prefill_step_size=128,
    )
    scheduler = Scheduler(model, tokenizer, config)

    original = scheduler._prefix_cache.on_prefill_checkpoint
    def _recording_checkpoint(request, processed_tokens, cache_states):
        checkpoints.append(processed_tokens)
        original(request, processed_tokens, cache_states)
    scheduler._prefix_cache.on_prefill_checkpoint = _recording_checkpoint

    long_prompt_ids = list(range(512))
    scheduler.add_request(Request(
        request_id="r1",
        prompt=" ".join(str(t) for t in long_prompt_ids),
        prompt_token_ids=long_prompt_ids,
        sampling_params=SamplingParams(max_tokens=4),
    ))

    for _ in range(300):
        output = scheduler.step()
        if "r1" in output.finished_request_ids:
            break

    assert len(checkpoints) >= 2, (
        f"Expected >=2 mid-prefill checkpoints, got {checkpoints}"
    )


@pytest.mark.slow
def test_turn_boundary_checkpoint_saved_at_each_boundary(qwen3_small):
    """With use_turn_cache, the turn cache must record state at each turn boundary."""
    model, tokenizer = qwen3_small
    config = SchedulerConfig(
        use_turn_cache=True,
        use_memory_aware_cache=False,
        chunked_prefill_tokens=256,
        turn_cache_stride=64,
        turn_cache_memory_gb=1.0,
    )
    scheduler = Scheduler(model, tokenizer, config)

    boundary_positions = []
    original_checkpoint = scheduler._prefix_cache.on_prefill_checkpoint
    def _record(request, processed_tokens, cache_states):
        boundary_positions.append(processed_tokens)
        original_checkpoint(request, processed_tokens, cache_states)
    scheduler._prefix_cache.on_prefill_checkpoint = _record

    turn1 = list(range(256))
    turn2 = list(range(256, 512))
    req = Request(
        request_id="r-turns",
        prompt="placeholder",
        prompt_token_ids=turn1 + turn2,
        sampling_params=SamplingParams(max_tokens=4),
    )
    req._turn_boundaries = [256]
    scheduler.add_request(req)

    for _ in range(400):
        output = scheduler.step()
        if "r-turns" in output.finished_request_ids:
            break

    assert any(abs(p - 256) <= 16 for p in boundary_positions), (
        f"No checkpoint near turn boundary 256; got checkpoints at {boundary_positions}"
    )


@pytest.fixture(scope="module")
def qwen3_mtp_model():
    try:
        from mlx_lm import load
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit")
        if not (hasattr(model, "mtp") and model.mtp is not None):
            pytest.skip("loaded model has no MTP head")
        return model, tokenizer
    except Exception:
        pytest.skip("mlx-community/Qwen3-0.6B-4bit not available or has no MTP head")


@pytest.mark.slow
@pytest.mark.hybrid_only
def test_mtp_produces_extra_tokens(qwen3_mtp_model):
    """With MTP enabled, at least one step should return 2 tokens for a request."""
    model, tokenizer = qwen3_mtp_model
    config = SchedulerConfig(enable_mtp=True)
    scheduler = Scheduler(model, tokenizer, config)
    scheduler.add_request(Request(
        request_id="r-mtp",
        prompt="Hello",
        sampling_params=SamplingParams(max_tokens=16),
    ))
    token_counts_per_step = []
    for _ in range(100):
        output = scheduler.step()
        total = sum(len(o.output_token_ids) for o in output.outputs)
        if total > 0:
            token_counts_per_step.append(total)
        if not scheduler.has_requests():
            break
    assert any(c > 1 for c in token_counts_per_step), (
        "MTP never produced more than 1 token in a single step"
    )
