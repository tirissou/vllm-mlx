import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from vllm_mlx.engine_core import EngineCore
from vllm_mlx.scheduler import SchedulerOutput

@pytest.mark.asyncio
async def test_engine_core_worker_loop_async():
    model = MagicMock()
    tokenizer = MagicMock()
    engine = EngineCore(model=model, tokenizer=tokenizer)
    
    # Mock scheduler.step to be an async method.
    # The current implementation calls it synchronously, so it will receive a coroutine object.
    engine.scheduler.step = AsyncMock(return_value=SchedulerOutput(
        scheduled_request_ids=[],
        num_scheduled_tokens=0,
        finished_request_ids=set(),
        outputs=[],
        has_work=False
    ))
    
    # Mock scheduler.has_requests to return True once, then False
    engine.scheduler.has_requests = MagicMock(side_effect=[True, False])
    
    # Start engine and wait for a few steps
    await engine.start()
    # Give it a moment to run the loop
    await asyncio.sleep(0.2)
    await engine.stop()

    assert engine._steps_executed > 0
