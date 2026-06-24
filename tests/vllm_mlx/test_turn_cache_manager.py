import pytest
from unittest.mock import MagicMock, AsyncMock
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.turn_prefix_cache import SSDRef, TurnNode, TurnPrefixCache
from vllm_mlx.kv_cache import RequestCacheState

def _make_turn_cache_request(prompt_token_ids, turn_boundaries):
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req._turn_boundaries = turn_boundaries
    req._cache_state = RequestCacheState()
    req.request_id = "test_request"
    return req

@pytest.mark.asyncio
async def test_turn_cache_manager_fetch_async_hit():
    mock_inner = MagicMock(spec=TurnPrefixCache)
    node = MagicMock(spec=TurnNode)
    node.recurrent_data = None
    mock_inner.match.return_value = ([node], True)
    mock_inner.find_checkpoint_ancestor.return_value = MagicMock(spec=TurnNode)
    mock_inner.collect_path_data.return_value = ([], [])
    
    manager = TurnCacheManager(inner=mock_inner)
    request = _make_turn_cache_request(prompt_token_ids=[1, 2, 3], turn_boundaries=[1])
    
    result = await manager.fetch(request)
    assert result is True

@pytest.mark.asyncio
async def test_turn_cache_manager_fetch_async_ssd_promotion():
    mock_inner = MagicMock(spec=TurnPrefixCache)
    
    # Setup a node with SSDRef
    ssd_ref = SSDRef(file_path="/tmp/cache", size_bytes=1024)
    mock_node = MagicMock(spec=TurnNode)
    mock_node.recurrent_data = ssd_ref
    
    mock_inner.match.return_value = ([mock_node], False)
    mock_inner.find_checkpoint_ancestor.return_value = None
    
    # Mock SSD cache on inner
    mock_ssd_cache = MagicMock()
    mock_ssd_cache.async_promote = AsyncMock(return_value=[{"layer": 0}])
    mock_inner.ssd_cache = mock_ssd_cache
    
    manager = TurnCacheManager(inner=mock_inner)
    request = _make_turn_cache_request(prompt_token_ids=[1, 2, 3], turn_boundaries=[1])
    
    # The fetch should trigger promotion
    result = await manager.fetch(request)
    
    # It should return True because promotion makes it a hit
    assert result is True
    # It should have called async_promote
    mock_ssd_cache.async_promote.assert_awaited_once_with(ssd_ref)
    # It should have updated the node's recurrent_data
    assert mock_node.recurrent_data == [{"layer": 0}]
