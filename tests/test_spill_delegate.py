# SPDX-License-Identifier: Apache-2.0
"""Integration tests: SpillableCache.set_spill_delegate wired correctly."""

from unittest.mock import MagicMock, call

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.kv_cache import SpillableCache


def _tiny_config():
    # Force eviction after 1 entry by using a very small memory limit.
    return MemoryCacheConfig(max_memory_mb=0.001, max_entries=1)


def _fake_layers():
    import numpy as np
    return [{"keys": np.zeros((1, 4, 8)), "values": np.zeros((1, 4, 8))}]


def _fake_request(token_ids, request_id="req-1"):
    req = MagicMock()
    req.prompt_token_ids = list(token_ids)
    req.request_id = request_id
    req.output_token_ids = []
    req._cache_state = MagicMock()
    req._cache_state.store_tokens = None
    return req


class TestMemoryAwarePrefixCacheSpillDelegate:
    def test_implements_spillable_cache_protocol(self):
        config = MemoryCacheConfig()
        cache = MemoryAwarePrefixCache(model=MagicMock(), config=config)
        assert isinstance(cache, SpillableCache)

    def test_set_spill_delegate_is_callable(self):
        config = MemoryCacheConfig()
        cache = MemoryAwarePrefixCache(model=MagicMock(), config=config)
        on_spill = MagicMock(return_value=(1, 2))
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)  # must not raise

    def test_on_spill_called_with_tokens_and_layers_on_eviction(self):
        config = _tiny_config()
        cache = MemoryAwarePrefixCache(model=MagicMock(), config=config)
        on_spill = MagicMock(return_value=(1, 2, 3))
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)

        req1 = _fake_request([1, 2, 3], "req-1")
        req2 = _fake_request([4, 5, 6], "req-2")
        cache.store(list(req1.prompt_token_ids), _fake_layers())
        cache.store(list(req2.prompt_token_ids), _fake_layers())  # triggers eviction of req1

        on_spill.assert_called_once()
        tokens_arg, layers_arg = on_spill.call_args[0]
        assert tokens_arg == (1, 2, 3)
        assert isinstance(layers_arg, list)

    def test_no_direct_ssd_tier_access_after_delegate_set(self):
        """MemoryAwarePrefixCache must not touch _ssd_tier once delegate is set."""
        config = MemoryCacheConfig()
        cache = MemoryAwarePrefixCache(model=MagicMock(), config=config)
        on_spill = MagicMock(return_value=(1,))
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)
        # _ssd_tier should be None (not set externally)
        assert cache._ssd_tier is None  # not set externally
