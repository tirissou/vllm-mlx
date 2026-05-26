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
    assert cs.n_minus_one_state is None
    assert cs.mid_prefill_last_save == 0
    assert cs.mid_prefill_cache_key is None
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


def test_turn_cache_adapter_on_prefill_checkpoint_eagerly_inserts_turn():
    """TurnCacheAdapter.on_prefill_checkpoint eagerly inserts the turn into the trie."""
    from vllm_mlx.kv_cache import RequestCacheState
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    new_node = MagicMock(n_tokens=5)
    inner._split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = new_node
    adapter = TurnCacheAdapter(inner)

    request = MagicMock()
    cs = RequestCacheState(cached_tokens=0, adapter_state=[])
    request._cache_state = cs
    request.prompt_token_ids = list(range(10))  # 10 tokens; B_sys=5 → sys=[0-4], user=[5-9]
    request._turn_boundaries = [5]

    extracted = [{"state": (None, None), "class_name": "KVCache"}]
    adapter.on_prefill_checkpoint(request, 5, extracted)

    inner.insert.assert_called_once()
    assert len(cs.adapter_state) == 1
    assert cs.adapter_state[0] is new_node


def test_turn_cache_adapter_on_prefill_checkpoint_ignores_non_boundary():
    from vllm_mlx.kv_cache import RequestCacheState
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner._split_cache_arrays.return_value = ([], None)
    adapter = TurnCacheAdapter(inner)

    request = MagicMock()
    request._cache_state = RequestCacheState(cached_tokens=0, adapter_state=[])
    request._turn_boundaries = [5]
    request.prompt_token_ids = list(range(10))

    adapter.on_prefill_checkpoint(request, 3, [])  # total=3, not in [5] → no insert
    inner.insert.assert_not_called()


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


from vllm_mlx.scheduler import _build_prefix_cache, SchedulerConfig
from vllm_mlx.ssd_offloaded_cache import SSDOffloadedCache
from unittest.mock import MagicMock
import tempfile


class TestBuildPrefixCacheSSDWiring:
    def test_no_ssd_does_not_wrap_with_ssd_offloaded_cache(self):
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            ssd_cache_dir=None,
        )
        model = MagicMock()
        model.layers = [MagicMock()] * 4
        bundle = _build_prefix_cache(config, model)
        assert not isinstance(bundle.adapter, SSDOffloadedCache)

    def test_with_ssd_wraps_adapter_with_ssd_offloaded_cache(self):
        with tempfile.TemporaryDirectory() as ssd_dir:
            config = SchedulerConfig(
                enable_prefix_cache=True,
                use_memory_aware_cache=True,
                ssd_cache_dir=ssd_dir,
            )
            model = MagicMock()
            model.layers = [MagicMock()] * 4
            bundle = _build_prefix_cache(config, model)
            assert isinstance(bundle.adapter, SSDOffloadedCache)
            bundle.adapter.close()


def _make_mock_request_with_cache_state(n_minus_one_state=None):
    from unittest.mock import MagicMock
    from vllm_mlx.kv_cache import RequestCacheState
    req = MagicMock()
    req._cache_state = RequestCacheState()
    req._cache_state.n_minus_one_state = n_minus_one_state
    return req


class TestCacheManagerNMinusOne:
    """Regression tests for per-step N-1 cache tracking via CacheManager."""

    def test_reconstruct_standard_kv_decrements_offset(self):
        """_reconstruct trims standard KV offset by 1."""
        import mlx.core as mx
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[0], rotating_indices=[], recurrent_indices=[]
        )
        kv_state = {
            "state": (mx.zeros((1, 2, 5, 4)), mx.zeros((1, 2, 5, 4))),
            "meta_state": ("5", "64", "4"),
            "class_name": "BatchKVCache",
            "class_ref": None,
        }
        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"recurrent": None}
        )
        result = adapter._reconstruct(req, [kv_state])
        assert result[0]["meta_state"][0] == "4"

    def test_reconstruct_standard_kv_clamps_offset_at_zero(self):
        """Offset at 0 stays 0, does not go negative."""
        import mlx.core as mx
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[0], rotating_indices=[], recurrent_indices=[]
        )
        kv_state = {
            "state": (mx.zeros((1, 2, 1, 4)), mx.zeros((1, 2, 1, 4))),
            "meta_state": ("0",),
            "class_name": "BatchKVCache",
            "class_ref": None,
        }
        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"recurrent": None}
        )
        result = adapter._reconstruct(req, [kv_state])
        assert int(result[0]["meta_state"][0]) == 0

    def test_reconstruct_rotating_kv_sets_trim_last(self):
        """_reconstruct tags rotating KV layers with trim_last=True; no shadow needed."""
        import mlx.core as mx
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[], rotating_indices=[0], recurrent_indices=[]
        )
        rotating_layer = {
            "class_name": "RotatingKVCache",
            "state": (mx.zeros((1, 4, 8, 64)), mx.zeros((1, 4, 8, 64))),
            "meta_state": ("4", "0", "8", "0"),
        }
        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"recurrent": None}
        )
        result = adapter._reconstruct(req, [rotating_layer])

        assert result[0]["trim_last"] is True
        assert result[0]["class_name"] == "RotatingKVCache"
        assert result[0]["meta_state"] == ("4", "0", "8", "0")

    def test_reconstruct_recurrent_uses_saved_refs(self):
        """_reconstruct uses saved ArraysCache refs for recurrent layers."""
        import mlx.core as mx
        import pytest
        from mlx_lm.models.cache import ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[], rotating_indices=[], recurrent_indices=[0]
        )
        saved = ArraysCache(2)
        saved[0] = mx.full((1, 4), 42.0)
        saved[1] = mx.full((1, 4), 43.0)

        recurrent_state = {
            "state": [mx.full((1, 4), 99.0), mx.full((1, 4), 99.0)],
            "meta_state": "",
            "class_name": "ArraysCache",
            "class_ref": ArraysCache,
        }
        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"rotating": [], "recurrent": [saved]}
        )
        result = adapter._reconstruct(req, [recurrent_state])
        mx.eval(*result[0]["state"])
        assert float(result[0]["state"][0][0, 0]) == pytest.approx(42.0)
        assert float(result[0]["state"][1][0, 0]) == pytest.approx(43.0)

    def test_update_n_minus_one_first_call_initializes_recurrent_state(self):
        """First call sets n_minus_one_state with recurrent key only; no rotating shadows."""
        from unittest.mock import MagicMock
        from mlx_lm.models.cache import ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import RequestCacheState

        adapter = TurnCacheAdapter(None)
        arrays_cache = ArraysCache(2)
        import mlx.core as mx
        arrays_cache.cache = [mx.zeros((2, 4)), mx.zeros((2, 4))]
        prompt_cache = [arrays_cache]

        req = MagicMock()
        req._cache_state = RequestCacheState()

        adapter.update_n_minus_one(req, prompt_cache, uid_idx=0)

        state = req._cache_state.n_minus_one_state
        assert state is not None
        assert "rotating" not in state
        assert "recurrent" in state

    def test_update_n_minus_one_saves_recurrent_ref_each_step(self):
        """Each call saves a fresh ArraysCache ref for recurrent layers; no shadow writes."""
        from unittest.mock import MagicMock
        import mlx.core as mx
        import pytest
        from mlx_lm.models.cache import ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import RequestCacheState

        adapter = TurnCacheAdapter(None)

        live = ArraysCache(1)
        live.cache = [mx.full((2, 4), 1.0, dtype=mx.float32)]
        prompt_cache = [live]

        req = MagicMock()
        req._cache_state = RequestCacheState()

        # First call — initialises state
        adapter.update_n_minus_one(req, prompt_cache, uid_idx=0)

        # Simulate model replacing live state (ArraysCache entries are replaced, not mutated)
        live.cache = [mx.full((2, 4), 2.0, dtype=mx.float32)]

        # Second call — saves current live state
        adapter.update_n_minus_one(req, prompt_cache, uid_idx=0)

        saved = req._cache_state.n_minus_one_state["recurrent"]
        assert saved is not None and len(saved) == 1
        mx.eval(saved[0].cache[0])
        assert float(saved[0].cache[0][0, 0]) == pytest.approx(2.0)

    def test_reconstruct_hybrid_model_correct_layer_order(self):
        """_reconstruct handles hybrid models with rotating + recurrent + KV layers."""
        import mlx.core as mx
        import pytest
        from mlx_lm.models.cache import RotatingKVCache, ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[2], rotating_indices=[0], recurrent_indices=[1]
        )
        saved_recurrent = ArraysCache(1)
        saved_recurrent[0] = mx.full((1, 4), 55.0)

        extracted_cache = [
            {
                "state": (mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4))),
                "meta_state": ("0", "4", "2", "2"),
                "class_name": "RotatingKVCache",
                "class_ref": RotatingKVCache,
            },
            {
                "state": [mx.full((1, 4), 99.0)],
                "meta_state": "",
                "class_name": "ArraysCache",
                "class_ref": ArraysCache,
            },
            {
                "state": (mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4))),
                "meta_state": ("3",),
                "class_name": "BatchKVCache",
                "class_ref": None,
            },
        ]
        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"recurrent": [saved_recurrent]}
        )
        result = adapter._reconstruct(req, extracted_cache)

        assert len(result) == 3
        assert result[0]["trim_last"] is True
        mx.eval(*result[1]["state"])
        assert float(result[1]["state"][0][0, 0]) == pytest.approx(55.0)
        assert result[2]["meta_state"][0] == "2"  # KV offset 3 → 2
