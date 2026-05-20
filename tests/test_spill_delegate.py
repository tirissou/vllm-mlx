# SPDX-License-Identifier: Apache-2.0
"""Integration tests: SpillableCache.set_spill_delegate wired correctly."""

from unittest.mock import MagicMock, call

import mlx.core as mx

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.kv_cache import SpillableCache
from vllm_mlx.turn_prefix_cache import (
    TurnPrefixCacheConfig,
    TurnPrefixCache,
    Segment,
    SSDRef,
)


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


# ── helpers ────────────────────────────────────────────────────────────────

def _seg(token_ids, role="user"):
    return Segment(role=role, token_ids=token_ids)


def _make_turn_layers(n_layers=2):
    return [
        {"state": (mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8)))}
        for _ in range(n_layers)
    ]


def _make_turn_cache(tmp_path=None):
    """Create a TurnPrefixCache with SSD dir (needed for legacy spill path tests)."""
    import tempfile, os
    ssd_dir = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    return TurnPrefixCache(TurnPrefixCacheConfig(
        checkpoint_stride=0,
        max_memory_gb=8.0,
        kv_dtype="bf16",
        ssd_max_gb=10.0,
        ssd_dir=ssd_dir,
    ))


class TestTurnPrefixCacheSpillDelegate:
    def test_implements_spillable_cache_protocol(self):
        """TurnPrefixCache must satisfy the SpillableCache structural protocol."""
        cache = _make_turn_cache()
        assert isinstance(cache, SpillableCache)

    def test_set_spill_delegate_is_callable(self):
        """set_spill_delegate must be present and accept two callables."""
        cache = _make_turn_cache()
        on_spill = MagicMock(return_value=(1, 2))
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)  # must not raise

    def test_spill_calls_on_spill_with_tokens_and_layers(self):
        """When a delegate is set, _spill_to_ssd must call on_spill(tokens, layers)."""
        cache = _make_turn_cache()
        on_spill = MagicMock(return_value=(1, 2))
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)

        kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
        node = cache.insert(cache.root, _seg([1, 2, 3]), kv, [1.0], None)
        cache._spill_to_ssd(node)

        on_spill.assert_called_once()
        tokens_arg, layers_arg = on_spill.call_args[0]
        assert isinstance(tokens_arg, tuple)
        assert isinstance(layers_arg, list)

    def test_node_stores_handle_not_ssd_ref_after_spill(self):
        """After spill with delegate, node.kv_arrays must hold the opaque handle, not SSDRef."""
        cache = _make_turn_cache()
        handle = object()  # distinct opaque handle
        on_spill = MagicMock(return_value=handle)
        on_promote = MagicMock(return_value=None)
        cache.set_spill_delegate(on_spill, on_promote)

        kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
        node = cache.insert(cache.root, _seg([1, 2, 3]), kv, [1.0], None)
        cache._spill_to_ssd(node)

        # The node must hold the opaque handle, not an SSDRef
        assert not isinstance(node.kv_arrays, SSDRef)
        assert node.kv_arrays is handle

    def test_promote_calls_on_promote_with_handle(self):
        """When promoting a delegate-spilled node, on_promote must receive the stored handle."""
        cache = _make_turn_cache()
        handle = (42, 99, 7)
        on_spill = MagicMock(return_value=handle)
        restored_layers = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
        on_promote = MagicMock(return_value=restored_layers)
        cache.set_spill_delegate(on_spill, on_promote)

        kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
        node = cache.insert(cache.root, _seg([1, 2, 3]), kv, [1.0], None)
        cache._spill_to_ssd(node)

        result = cache._promote_from_ssd(node)

        assert result is True
        on_promote.assert_called_once_with(handle)
        assert node.kv_arrays is restored_layers

    def test_tokens_to_node_returns_correct_token_sequence(self):
        """_tokens_to_node must reconstruct the full token path from root to node."""
        cache = _make_turn_cache()
        parent_node = cache.insert(cache.root, _seg([1, 2]), [], None, None)
        child_node = cache.insert(parent_node, _seg([3, 4, 5]), [], None, None)

        tokens = cache._tokens_to_node(child_node)
        assert tokens == (1, 2, 3, 4, 5)
