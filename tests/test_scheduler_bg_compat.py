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
