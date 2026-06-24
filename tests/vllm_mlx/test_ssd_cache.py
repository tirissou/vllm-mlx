import pytest
import asyncio
import tempfile
from unittest.mock import patch
from vllm_mlx.ssd_cache import SSDCacheTier, SSDCacheConfig
try:
    from vllm_mlx.ssd_cache import SSDRef
except ImportError:
    from vllm_mlx.turn_prefix_cache import SSDRef

@pytest.mark.asyncio
async def test_async_promote_with_ref():
    # Setup mock SSD cache and ref
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = SSDCacheConfig(cache_dir=tmp_dir)
        tier = SSDCacheTier(config)
        ref = SSDRef(file_path="/tmp/test_cache", size_bytes=1024)
        
        # We want to ensure it calls the internal read method
        with patch.object(tier, '_read_entry', return_value=[{"data": "dummy"}]) as mock_read:
            result = await tier.async_promote(ref)
            assert result is not None
            assert result == [{"data": "dummy"}]
            # According to spec, it should be called with ONLY ref.file_path
            mock_read.assert_called_once_with(ref.file_path)
