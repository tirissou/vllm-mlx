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


from unittest.mock import MagicMock, call, patch


def _make_turn_cache_request(
    prompt_token_ids, output_token_ids, turn_boundaries, boundary_states=None, path=None
):
    from vllm_mlx.kv_cache import RequestCacheState
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req.output_token_ids = output_token_ids
    req._turn_boundaries = turn_boundaries
    req._boundary_states = boundary_states or {}
    req._cache_state = RequestCacheState(adapter_state=path or [])
    return req


def test_turn_cache_adapter_store_returns_true_on_success():
    """store() must return True now — the old stub returned False."""
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner._split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=5)

    adapter = TurnCacheAdapter(inner)
    # prompt=[1,2,3,4,5] with boundary at 3 (sys=[1,2,3], user=[4,5])
    req = _make_turn_cache_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        output_token_ids=[6, 7],
        turn_boundaries=[3],
    )
    result = adapter.store(req, [])
    assert result is True


def test_turn_cache_adapter_store_calls_inner_insert():
    """store() must call inner.insert() to build the trie node."""
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner._split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=0)

    adapter = TurnCacheAdapter(inner)
    req = _make_turn_cache_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        output_token_ids=[6, 7],
        turn_boundaries=[3],
    )
    adapter.store(req, [])
    assert inner.insert.called


def test_turn_cache_adapter_store_returns_false_when_no_output():
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    adapter = TurnCacheAdapter(inner)
    req = _make_turn_cache_request(
        prompt_token_ids=[1, 2, 3, 4], output_token_ids=[], turn_boundaries=[3]
    )
    assert adapter.store(req, []) is False


def test_turn_cache_adapter_release_calls_inner_release():
    """TurnCacheAdapter.release() must forward to inner.release()."""
    inner = MagicMock()
    adapter = TurnCacheAdapter(inner)
    path = [MagicMock(), MagicMock()]
    adapter.release(path)
    inner.release.assert_called_once_with(path)


def test_turn_cache_adapter_release_noop_on_none():
    inner = MagicMock()
    adapter = TurnCacheAdapter(inner)
    adapter.release(None)  # must not raise
    inner.release.assert_not_called()


def test_memory_cache_adapter_on_prefill_checkpoint_stores_prefix():
    """MemoryCacheAdapter must store a prefix entry on checkpoint."""
    inner = MagicMock()
    inner.store.return_value = True

    adapter = MemoryCacheAdapter(inner)

    from vllm_mlx.kv_cache import RequestCacheState
    request = MagicMock()
    request.prompt_token_ids = list(range(10))
    request._cache_state = RequestCacheState(cached_tokens=0, mid_prefill_last_save=0, mid_prefill_cache_key=None)

    extracted = [{"state": (None, None), "class_name": "KVCache", "class_ref": None}]
    fake_reconstructed = [MagicMock()]
    # The adapter does `from .turn_prefix_cache import reconstruct_cache_from_states`
    # inside the method, so we patch the function in the turn_prefix_cache module.
    with patch("vllm_mlx.turn_prefix_cache.reconstruct_cache_from_states", return_value=fake_reconstructed):
        adapter.on_prefill_checkpoint(request, 5, extracted)

    inner.store.assert_called_once()
    stored_tokens = inner.store.call_args[0][0]
    assert stored_tokens == list(range(5))


def test_turn_cache_adapter_on_prefill_checkpoint_captures_boundary_state():
    """TurnCacheAdapter must record boundary state at boundary-1 positions."""
    from vllm_mlx.kv_cache import RequestCacheState
    inner = MagicMock()
    adapter = TurnCacheAdapter(inner)

    request = MagicMock()
    request._cache_state = RequestCacheState(cached_tokens=0)
    request._turn_boundaries = [5]   # boundary at 5
    request._boundary_states = {}

    extracted = [{"state": (None, None), "class_name": "KVCache"}]
    # checkpoint fires at processed=4 (one before boundary 5, per split-chunk convention)
    adapter.on_prefill_checkpoint(request, 4, extracted)

    # boundary_states keyed by boundary position (5), not checkpoint position (4)
    assert 5 in request._boundary_states
    assert request._boundary_states[5] is extracted


def test_turn_cache_adapter_on_prefill_checkpoint_ignores_non_boundary():
    from vllm_mlx.kv_cache import RequestCacheState
    inner = MagicMock()
    adapter = TurnCacheAdapter(inner)

    request = MagicMock()
    request._cache_state = RequestCacheState(cached_tokens=0)
    request._turn_boundaries = [5]
    request._boundary_states = {}

    adapter.on_prefill_checkpoint(request, 3, [])  # total=3, boundary=5 → 3+1=4 ≠ 5
    assert request._boundary_states == {}


import mlx.core as mx


def _make_kv_cache():
    from mlx_lm.models.cache import KVCache
    c = KVCache()
    c.keys = mx.zeros((1, 8, 4, 64))
    c.values = mx.zeros((1, 8, 4, 64))
    c.offset = 4
    return c


def _make_recurrent_layer():
    """A fake recurrent cache layer (no .offset or .keys attributes)."""
    class FakeRecurrent:
        pass
    return FakeRecurrent()


def test_extract_recurrent_state_returns_only_non_kv_layers():
    from vllm_mlx.kv_cache import extract_recurrent_state as _extract_recurrent_state
    kv = _make_kv_cache()
    recur = _make_recurrent_layer()
    result = _extract_recurrent_state([kv, recur, kv])
    assert len(result) == 1
    assert result[0] is recur


def test_extract_recurrent_state_empty_for_pure_kv():
    from vllm_mlx.kv_cache import extract_recurrent_state as _extract_recurrent_state
    result = _extract_recurrent_state([_make_kv_cache(), _make_kv_cache()])
    assert result == []


def test_compose_n_minus_1_cache_trims_kv_offset():
    from vllm_mlx.kv_cache import compose_n_minus_1_cache as _compose_n_minus_1_cache

    kv_dict = {
        "state": (mx.zeros((1, 8, 4, 64)), mx.zeros((1, 8, 4, 64))),
        "meta_state": (4,),
        "class_name": "KVCache",
        "class_ref": None,
    }
    composed = _compose_n_minus_1_cache([kv_dict], prev_recurrent_extracted=[])
    # offset should be reduced from 4 to 3 (stored as string in meta_state)
    assert int(composed[0]["meta_state"][0]) == 3


def test_compose_n_minus_1_cache_replaces_recurrent_with_snapshot():
    from vllm_mlx.kv_cache import compose_n_minus_1_cache as _compose_n_minus_1_cache

    kv_dict = {
        "state": (mx.zeros((1, 8, 4, 64)), mx.zeros((1, 8, 4, 64))),
        "meta_state": (4,),
        "class_name": "KVCache",
        "class_ref": None,
    }
    snapshot_recur = {"state": "SNAPSHOT", "class_name": "MambaCache", "class_ref": None}
    current_recur = {"state": "CURRENT", "class_name": "MambaCache", "class_ref": None}

    composed = _compose_n_minus_1_cache(
        [kv_dict, current_recur],
        prev_recurrent_extracted=[snapshot_recur],
    )
    assert composed[0]["class_name"] == "KVCache"
    assert composed[1]["state"] == "SNAPSHOT"   # replaced with N-1 snapshot


class TestBuildPrefixCache:
    """_build_prefix_cache selects the right adapter for each SchedulerConfig variant."""

    def _config(self, **kwargs):
        from vllm_mlx.scheduler import SchedulerConfig
        # Disable all backends by default so tests opt-in explicitly
        base = dict(use_paged_cache=False, use_memory_aware_cache=False, use_turn_cache=False)
        base.update(kwargs)
        return SchedulerConfig(**base)

    def test_memory_aware_config_returns_memory_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_inner = MagicMock()
        mock_inner.memory_limit_mb = 1000.0
        with patch("vllm_mlx.scheduler.MemoryAwarePrefixCache", return_value=mock_inner):
            bundle = _build_prefix_cache(self._config(use_memory_aware_cache=True), model=object())

        assert isinstance(bundle.adapter, MemoryCacheAdapter)
        assert bundle.memory_aware_cache is mock_inner
        assert bundle.prefix_cache is None
        assert bundle.turn_cache is None
        assert bundle.ssd_tier is None

    def test_legacy_config_returns_legacy_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_pm = MagicMock()
        with patch("vllm_mlx.scheduler.PrefixCacheManager", return_value=mock_pm):
            bundle = _build_prefix_cache(self._config(), model=object())

        assert isinstance(bundle.adapter, LegacyCacheAdapter)
        assert bundle.prefix_cache is mock_pm
        assert bundle.memory_aware_cache is None
        assert bundle.turn_cache is None

    def test_turn_cache_config_returns_turn_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_tc = MagicMock()
        with patch("vllm_mlx.turn_prefix_cache.TurnPrefixCache", return_value=mock_tc):
            bundle = _build_prefix_cache(self._config(use_turn_cache=True), model=object())

        assert isinstance(bundle.adapter, TurnCacheAdapter)
        assert bundle.turn_cache is mock_tc
        assert bundle.memory_aware_cache is None
        assert bundle.prefix_cache is None

    def test_paged_cache_config_returns_paged_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_pcm = MagicMock()
        mock_bac = MagicMock()
        with patch("vllm_mlx.scheduler.PagedCacheManager", return_value=mock_pcm), \
             patch("vllm_mlx.scheduler.BlockAwarePrefixCache", return_value=mock_bac):
            bundle = _build_prefix_cache(self._config(use_paged_cache=True), model=object())

        assert isinstance(bundle.adapter, PagedCacheAdapter)
        assert bundle.paged_cache_manager is mock_pcm
        assert bundle.block_aware_cache is mock_bac
        assert bundle.memory_aware_cache is None


def test_request_cache_state_is_single_attribute():
    """After wiring, all cache state lives at request._cache_state."""
    from vllm_mlx.kv_cache import RequestCacheState
    cs = RequestCacheState(hit_type="hit", cached_tokens=42)
    class FakeRequest:
        pass
    req = FakeRequest()
    req._cache_state = cs
    assert req._cache_state.hit_type == "hit"
    assert req._cache_state.cached_tokens == 42
