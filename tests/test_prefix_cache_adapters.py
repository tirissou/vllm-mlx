# SPDX-License-Identifier: Apache-2.0
# tests/test_prefix_cache_adapters.py
from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.prefix_cache_adapters import (
    MemoryCacheAdapter, TurnCacheAdapter, PagedCacheAdapter, LegacyCacheAdapter,
)


def test_request_cache_state_defaults():
    cs = RequestCacheState()
    assert cs.hit_type == "miss"
    assert cs.cache is None
    assert cs.cached_tokens == 0
    assert cs.remaining_tokens is None
    assert cs.store_tokens is None
    assert cs.decoded_cache is None
    assert cs.prev_recurrent is None
    assert cs.mid_prefill_last_save == 0
    assert cs.mid_prefill_cache_key is None
    assert cs.ssd_candidate is None
    assert cs.adapter_state is None


def _make_dummy_adapters():
    return [
        MemoryCacheAdapter(None),
        TurnCacheAdapter(None),
        PagedCacheAdapter(None),
        LegacyCacheAdapter(None),
    ]


def test_all_adapters_implement_on_prefill_checkpoint():
    for adapter in _make_dummy_adapters():
        assert callable(getattr(adapter, "on_prefill_checkpoint", None)), (
            f"{type(adapter).__name__} missing on_prefill_checkpoint"
        )


def test_on_prefill_checkpoint_no_op_does_not_raise():
    """No-op implementations must not raise on None inputs."""
    from unittest.mock import MagicMock
    request = MagicMock()
    for adapter in _make_dummy_adapters():
        adapter.on_prefill_checkpoint(request, 100, [])
