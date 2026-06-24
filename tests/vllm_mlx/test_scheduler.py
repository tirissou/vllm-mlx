import pytest
from unittest.mock import MagicMock
from vllm_mlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput

@pytest.mark.asyncio
async def test_scheduler_step_async():
    model = MagicMock()
    tokenizer = MagicMock()
    config = SchedulerConfig()
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    
    output = await scheduler.step()
    assert isinstance(output, SchedulerOutput)
