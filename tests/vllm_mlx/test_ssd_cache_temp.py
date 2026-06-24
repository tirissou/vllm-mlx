import pytest
import asyncio
from vllm_mlx.ssd_cache import SSDCacheTier, SSDRef

@pytest.mark.asyncio
async def test_async_promote_with_ref():
    # Setup mock SSD cache and ref
    from vllm_mlx.ssd_cache import SSDCacheConfig
    import tempfile
    import shutil
    
    tmp_dir = tempfile.mkdtemp()
    try:
        config = SSDCacheConfig(cache_dir=tmp_dir)
        tier = SSDCacheTier(config)
        ref = SSDRef(file_path="/tmp/test_cache", size_bytes=1024)
        
        # We want to ensure it calls the internal read method
        result = await tier.async_promote(ref)
        assert result is not None
    finally:
        shutil.rmtree(tmp_dir)
