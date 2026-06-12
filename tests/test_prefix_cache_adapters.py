# SPDX-License-Identifier: Apache-2.0
# tests/test_prefix_cache_adapters.py
import inspect
from abc import ABC

import pytest

from vllm_mlx.cache_types import KVQuantPolicy
from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.prefix_cache_adapters import TurnCacheManager


def test_cache_manager_is_abstract():
    from vllm_mlx.prefix_cache_adapters import CacheManager

    assert issubclass(CacheManager, ABC)


def test_cache_manager_abstract_methods():
    from vllm_mlx.prefix_cache_adapters import CacheManager

    abstract = {
        name
        for name, val in inspect.getmembers(CacheManager)
        if getattr(val, "__isabstractmethod__", False)
    }
    assert "boundaries" in abstract
    assert "fetch" in abstract
    assert "store" in abstract


def test_cache_manager_cannot_be_instantiated():
    from vllm_mlx.prefix_cache_adapters import CacheManager

    try:
        CacheManager()
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_request_cache_state_defaults():
    cs = RequestCacheState()
    assert cs.hit_type == "miss"
    assert cs.cache is None
    assert cs.cached_tokens == 0
    assert cs.remaining_tokens is None
    assert cs.prefill_boundaries == []
    assert cs.decoded_cache is None
<<<<<<< Updated upstream
    assert cs.prev_recurrent is None
    assert cs.turn_path == []
    # Removed fields must not exist
    assert not hasattr(cs, "n_minus_one_state")
    assert not hasattr(cs, "adapter_state")
    assert not hasattr(cs, "store_tokens")
    assert not hasattr(cs, "mid_prefill_last_save")
    assert not hasattr(cs, "mid_prefill_cache_key")
=======
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
>>>>>>> Stashed changes


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
    req._cache_state = RequestCacheState(turn_path=path or [])
    return req


def test_turn_cache_adapter_store_returns_true_on_success():
    """store() must return True now — the old stub returned False."""
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner.split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=5)

    adapter = TurnCacheManager(inner)
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
    inner.split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=0)

    adapter = TurnCacheManager(inner)
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
    adapter = TurnCacheManager(inner)
    req = _make_turn_cache_request(
        prompt_token_ids=[1, 2, 3, 4], output_token_ids=[], turn_boundaries=[3]
    )
    assert adapter.store(req, []) is False


def test_turn_cache_adapter_release_unpins_recorded_leaf():
    """TurnCacheManager.release(request) unpins the pinned leaf via inner.release()."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    leaf = MagicMock()
    req = MagicMock()
    req.request_id = "r-1"
    req._cache_state = MagicMock(turn_path=[MagicMock(), leaf])
    adapter._pinned_leaves[req.request_id] = leaf
    adapter.release(req)
    inner.release.assert_called_once_with([leaf])
    assert req.request_id not in adapter._pinned_leaves
    assert req._cache_state.turn_path == []


def test_turn_cache_adapter_release_noop_when_no_pinned_leaf():
    """If no leaf is recorded for the request, release is a no-op on the trie."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    req = MagicMock()
    req.request_id = "r-missing"
    req._cache_state = MagicMock(turn_path=[])
    adapter.release(req)  # must not raise
    inner.release.assert_not_called()


def test_turn_cache_adapter_on_prefill_checkpoint_eagerly_inserts_turn():
    """TurnCacheManager.on_prefill_checkpoint eagerly inserts the turn into the trie."""
    from vllm_mlx.kv_cache import RequestCacheState
    from vllm_mlx.cache_types import KVLayerSegment
    from vllm_mlx.kv_cache import QuantizedArray

    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    new_node = MagicMock(n_tokens=5)
    inner.insert.return_value = new_node
    adapter = TurnCacheManager(inner)

    request = MagicMock()
    cs = RequestCacheState(cached_tokens=0, turn_path=[])
    request._cache_state = cs
    request.prompt_token_ids = list(
        range(10)
    )  # 10 tokens; B_sys=5 → sys=[0-4], user=[5-9]
    request._turn_boundaries = [5]

    # Patch _segment so it returns empty sparse lists without needing real arrays
    import mlx.core as mx

    qa = QuantizedArray(
        packed=mx.zeros((1, 1, 1, 1), dtype=mx.uint32),
        scales=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
        biases=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
    )
    kv_placeholder = KVLayerSegment(keys=qa, values=qa, metadata={"layer_index": 0})
    with patch.object(
        TurnCacheManager, "_segment", return_value=([kv_placeholder], [None])
    ):
        extracted = [{"state": (None, None), "class_name": "KVCache"}]
        adapter.on_prefill_checkpoint(request, 5, extracted)

    inner.insert.assert_called_once()
    assert len(cs.turn_path) == 1
    assert cs.turn_path[0] is new_node


def test_turn_cache_adapter_on_prefill_checkpoint_ignores_non_boundary():
    from vllm_mlx.kv_cache import RequestCacheState

    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner.split_cache_arrays.return_value = ([], None)
    adapter = TurnCacheManager(inner)

    request = MagicMock()
    request._cache_state = RequestCacheState(cached_tokens=0, turn_path=[])
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


<<<<<<< Updated upstream
# ════════════════════════════════════════════════════════════════════════════
# Task 1: TurnCacheManager validate, extract_cache, close, error handling
# ════════════════════════════════════════════════════════════════════════════

# ── validate() ───────────────────────────────────────────────────────────────


def test_turn_cache_manager_validate_valid_cache():
    """validate() returns True for valid cache (list of layers with keys/values)."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    valid_cache = [
        MagicMock(
            keys=MagicMock(shape=(1, 8, 4, 64)),
            values=MagicMock(shape=(1, 8, 4, 64)),
        )
    ]
    with patch("vllm_mlx.prefix_cache_adapters.validate_cache", return_value=True):
        assert adapter.validate(valid_cache) is True


def test_turn_cache_manager_validate_none_cache():
    """validate() returns False for None cache."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    with patch("vllm_mlx.prefix_cache_adapters.validate_cache", return_value=False):
        assert adapter.validate(None) is False


def test_turn_cache_manager_validate_empty_cache():
    """validate() returns False for empty list cache."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    with patch("vllm_mlx.prefix_cache_adapters.validate_cache", return_value=False):
        assert adapter.validate([]) is False


def test_turn_cache_manager_validate_none_layer():
    """validate() returns False if any layer is None."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    with patch('vllm_mlx.prefix_cache_adapters.validate_cache', return_value=False):
        assert adapter.validate([None]) is False


# ── extract_cache() ──────────────────────────────────────────────────────────


def test_turn_cache_manager_extract_cache_valid():
    """extract_cache() forwards to extract_cache_states."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    raw_cache = [
        MagicMock(state=(MagicMock(), MagicMock()), meta_state=())
    ]
    with patch(
        "vllm_mlx.prefix_cache_adapters.extract_cache_states",
        return_value=[{"k": "v"}],
    ) as mock_extract:
        result = adapter.extract_cache(raw_cache)
        assert result == [{"k": "v"}]
        mock_extract.assert_called_once_with(raw_cache)


def test_turn_cache_manager_extract_cache_empty():
    """extract_cache() returns None for empty list."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    with patch(
        "vllm_mlx.prefix_cache_adapters.extract_cache_states", return_value=None
    ):
        assert adapter.extract_cache([]) is None


# ── save() / load() (updated with error handling) ────────────────────────────


def test_turn_cache_manager_save_forwards_to_inner():
    """save() forwards to inner.save()."""
    inner = MagicMock()
    inner.save.return_value = True
    adapter = TurnCacheManager(inner)
    result = adapter.save("/tmp/cache")
    assert result is True
    inner.save.assert_called_once_with("/tmp/cache")


def test_turn_cache_manager_save_fails():
    """save() returns False when inner.save() raises."""
    inner = MagicMock()
    inner.save.side_effect = OSError("disk full")
    adapter = TurnCacheManager(inner)
    result = adapter.save("/tmp/cache")
    assert result is False


def test_turn_cache_manager_load_forwards_to_inner():
    """load() forwards to inner.load()."""
    inner = MagicMock()
    inner.load.return_value = 42
    adapter = TurnCacheManager(inner)
    result = adapter.load("/tmp/cache")
    assert result == 0


def test_turn_cache_manager_load_fails():
    """load() returns 0 when inner.load() raises."""
    inner = MagicMock()
    inner.load.side_effect = OSError("disk full")
    adapter = TurnCacheManager(inner)
    result = adapter.load("/tmp/cache")
    assert result == 0


# ── close() ──────────────────────────────────────────────────────────────────


def test_turn_cache_manager_close_is_noop():
    """close() does nothing (TurnPrefixCache has no external resources)."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    adapter.close()  # must not raise
    inner.close.assert_not_called()

=======
>>>>>>> Stashed changes

class TestBuildPrefixCache:
    """_build_prefix_cache selects the right adapter for each SchedulerConfig variant."""

    def _config(self, **kwargs):
        from vllm_mlx.scheduler import SchedulerConfig

        # Disable all backends by default so tests opt-in explicitly
        base = dict(
            use_paged_cache=False, use_memory_aware_cache=False, use_turn_cache=False
        )
        base.update(kwargs)
        return SchedulerConfig(**base)

    def test_turn_cache_config_returns_turn_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_tc = MagicMock()
        with patch("vllm_mlx.turn_prefix_cache.TurnPrefixCache", return_value=mock_tc):
            bundle = _build_prefix_cache(
                self._config(use_turn_cache=True), model=object()
            )

        assert isinstance(bundle.adapter, TurnCacheManager)
        assert bundle.turn_cache is mock_tc
        # Deprecated fields removed
        assert not hasattr(bundle, 'memory_aware_cache')
        assert not hasattr(bundle, 'prefix_cache')


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
from unittest.mock import MagicMock

# ════════════════════════════════════════════════════════════════════════════
# Task 4: TurnCacheManager new interface tests
# ════════════════════════════════════════════════════════════════════════════


<<<<<<< Updated upstream
def _make_request(
    prompt_token_ids, turn_boundaries, output_token_ids=None, cached_tokens=0
):
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req.output_token_ids = output_token_ids or []
    req._turn_boundaries = turn_boundaries
    req._cache_state = RequestCacheState(cached_tokens=cached_tokens)
    return req


def _make_inner():
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner.split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=5)
    return inner


# ── boundaries() ─────────────────────────────────────────────────────────────


def test_boundaries_no_hit_returns_raw_turn_boundaries():
    adapter = TurnCacheManager(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[10, 20],
        cached_tokens=0,
    )
    assert adapter.boundaries(req) == [10, 20]


def test_boundaries_after_hit_offsets_by_cached_tokens():
    adapter = TurnCacheManager(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[10, 20, 30],
        cached_tokens=10,
    )
    # Boundaries > 10, shifted by 10: [20-10, 30-10] = [10, 20]
    assert adapter.boundaries(req) == [10, 20]


def test_boundaries_excludes_boundaries_at_or_below_cached_tokens():
    adapter = TurnCacheManager(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[5, 10, 20],
        cached_tokens=10,
    )
    # Only boundaries strictly > 10: [20-10] = [10]
    assert adapter.boundaries(req) == [10]


def test_boundaries_empty_when_no_turn_boundaries():
    adapter = TurnCacheManager(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[],
        cached_tokens=0,
    )
    assert adapter.boundaries(req) == []


# ── messages_to_segments() ───────────────────────────────────────────────────


def test_messages_to_segments_system_only_prompt_returns_one_segment():
    """System-only prompt (boundary at or beyond token count) should cache the system segment."""
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[5],  # boundary == len(tokens) → system fills whole context
    )
    segments = TurnCacheManager.messages_to_segments(req)
    assert len(segments) == 1
    assert list(segments[0].token_ids) == [1, 2, 3, 4, 5]


def test_messages_to_segments_no_boundaries_returns_empty():
    """No turn boundaries → caching genuinely inapplicable."""
    req = _make_request(prompt_token_ids=[1, 2, 3], turn_boundaries=[])
    assert TurnCacheManager.messages_to_segments(req) == []


def test_messages_to_segments_system_plus_user_returns_two_segments():
    """Normal case: system boundary mid-prompt yields 2 segments."""
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[3],  # sys=[1,2,3], user=[4,5]
    )
    segments = TurnCacheManager.messages_to_segments(req)
    assert len(segments) == 2
    assert list(segments[0].token_ids) == [1, 2, 3]
    assert list(segments[1].token_ids) == [4, 5]


def test_messages_to_segments_empty_tokens_returns_empty():
    req = _make_request(prompt_token_ids=[], turn_boundaries=[3])
    assert TurnCacheManager.messages_to_segments(req) == []


def test_messages_to_segments_boundary_at_zero_returns_empty():
    """Boundary at position 0 means no actual system prefix — skip."""
    req = _make_request(prompt_token_ids=[1, 2, 3], turn_boundaries=[0])
    assert TurnCacheManager.messages_to_segments(req) == []


# ── fetch() ──────────────────────────────────────────────────────────────────


def test_fetch_miss_returns_false():
    inner = _make_inner()
    inner.match.return_value = ([], None)
    adapter = TurnCacheManager(inner)
    req = _make_request(prompt_token_ids=list(range(10)), turn_boundaries=[])
    result = adapter.fetch(req)
    assert result is False
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.remaining_tokens == list(range(10))


def test_fetch_hit_populates_turn_path_on_cache_state():
    inner = _make_inner()
    node = MagicMock()
    node.n_tokens = 5
    inner.match.return_value = ([node], None)
    ancestor = MagicMock()
    ancestor.n_tokens = 5
    inner.find_checkpoint_ancestor.return_value = ancestor
    # Mock collect_path_data to return empty lists (no real MLX arrays needed)
    inner.collect_path_data.return_value = ([], [])
    # Patch _assemble and validate_cache so fetch() doesn't need real MLX arrays
    with patch.object(TurnCacheManager, "_assemble", return_value=[]), patch(
        "vllm_mlx.prefix_cache_adapters.validate_cache", return_value=True
    ):
        adapter = TurnCacheManager(inner)
        req = _make_request(
            prompt_token_ids=list(range(10)),
            turn_boundaries=[5],
            cached_tokens=0,
=======
import pytest


def _make_mock_request_with_cache_state(n_minus_one_state=None):
    from unittest.mock import MagicMock
    from vllm_mlx.kv_cache import RequestCacheState
    req = MagicMock()
    req._cache_state = RequestCacheState()
    req._cache_state.n_minus_one_state = n_minus_one_state
    return req


class TestCacheManagerNMinusOne:
    """Regression tests for per-step N-1 cache state tracking."""

    def test_reconstruct_standard_kv_decrements_offset(self):
        """_reconstruct trims standard KV offset by 1."""
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap
        import mlx.core as mx

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
            n_minus_one_state={"rotating": [], "recurrent": None}
        )
        result = adapter._reconstruct(req, [kv_state])

        assert result[0]["meta_state"][0] == "4"  # offset 5 → 4

    def test_reconstruct_standard_kv_clamps_offset_at_zero(self):
        """Offset at 0 stays 0, does not go negative."""
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap
        import mlx.core as mx

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
            n_minus_one_state={"rotating": [], "recurrent": None}
        )
        result = adapter._reconstruct(req, [kv_state])
        assert int(result[0]["meta_state"][0]) == 0

    def test_reconstruct_rotating_kv_uses_shadow(self):
        """_reconstruct uses shadow RotatingKVCache for rotating layers."""
        from mlx_lm.models.cache import RotatingKVCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap
        import mlx.core as mx

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[], rotating_indices=[0], recurrent_indices=[]
        )

        # Build shadow with 3 tokens (N-1 state)
        shadow = RotatingKVCache(max_size=8, keep=0)
        for i in range(3):
            k = mx.full((1, 2, 1, 4), float(i))
            v = mx.full((1, 2, 1, 4), float(i))
            shadow.update_and_fetch(k, v)
        mx.eval(shadow.keys)
        assert shadow.offset == 3

        # N-state extracted cache (4 tokens)
        rotating_state = {
            "state": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4))),
            "meta_state": ("0", "8", "4", "4"),
            "class_name": "RotatingKVCache",
            "class_ref": RotatingKVCache,
        }

        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"rotating": [shadow], "recurrent": None}
        )
        result = adapter._reconstruct(req, [rotating_state])

        # Result must come from shadow, not from extracted_cache
        # Shadow has 3 tokens; extracted has 4
        assert result[0]["class_name"] == "RotatingKVCache"
        assert int(result[0]["meta_state"][2]) == 3  # shadow offset == 3

    def test_reconstruct_recurrent_uses_saved_refs(self):
        """_reconstruct uses saved ArraysCache refs for recurrent layers."""
        from mlx_lm.models.cache import ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap
        import mlx.core as mx

        adapter = TurnCacheAdapter(None)
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[], rotating_indices=[], recurrent_indices=[0]
        )

        # Saved N-1 ArraysCache (step N-1 values)
        saved = ArraysCache(2)
        saved[0] = mx.full((1, 4), 42.0)
        saved[1] = mx.full((1, 4), 43.0)

        # N-state extracted cache (step N values, different)
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

        # Result state must come from saved (42.0), not extracted (99.0)
        mx.eval(*result[0]["state"])
        assert float(result[0]["state"][0][0, 0]) == pytest.approx(42.0)
        assert float(result[0]["state"][1][0, 0]) == pytest.approx(43.0)

    def test_update_n_minus_one_initializes_on_first_call(self):
        """First call initializes shadow instances but does not mirror (no previous step)."""
        from unittest.mock import MagicMock
        from mlx_lm.models.cache import BatchRotatingKVCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import RequestCacheState
        import mlx.core as mx

        adapter = TurnCacheAdapter(None)

        # Simulate a live BatchRotatingKVCache with 1 token prefilled
        live_rotating = BatchRotatingKVCache(max_size=8, left_padding=[0])
        k = mx.zeros((1, 2, 1, 4))
        v = mx.zeros((1, 2, 1, 4))
        live_rotating.update_and_fetch(k, v)

        req = MagicMock()
        req._cache_state = RequestCacheState()
        assert req._cache_state.n_minus_one_state is None

        adapter.update_n_minus_one(req, [live_rotating], uid_idx=0)

        # After first call: shadow initialized, no tokens mirrored
        state = req._cache_state.n_minus_one_state
        assert state is not None
        assert "rotating" in state
        assert len(state["rotating"]) == 1
        shadow = state["rotating"][0]
        assert shadow.offset == 0  # no tokens mirrored yet

    def test_update_n_minus_one_mirrors_on_second_call(self):
        """Second call mirrors the previous step's write into the shadow."""
        from unittest.mock import MagicMock
        from mlx_lm.models.cache import BatchRotatingKVCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import RequestCacheState
        import mlx.core as mx

        adapter = TurnCacheAdapter(None)

        live_rotating = BatchRotatingKVCache(max_size=8, left_padding=[0])

        req = MagicMock()
        req._cache_state = RequestCacheState()

        # Simulate: first call (init), then a decode step writes token, then second call (mirror)
        adapter.update_n_minus_one(req, [live_rotating], uid_idx=0)

        # Decode step: live gets one token
        k = mx.full((1, 2, 1, 4), 7.0)
        v = mx.full((1, 2, 1, 4), 7.0)
        live_rotating.update_and_fetch(k, v)
        mx.eval(live_rotating.keys)

        # Second call: should mirror the token just written
        adapter.update_n_minus_one(req, [live_rotating], uid_idx=0)

        shadow = req._cache_state.n_minus_one_state["rotating"][0]
        mx.eval(shadow.keys)
        assert shadow.offset == 1
        assert float(shadow.keys[0, 0, 0, 0]) == pytest.approx(7.0)

    def test_reconstruct_hybrid_model_correct_layer_order(self):
        """_reconstruct handles hybrid models with both rotating and recurrent layers."""
        from mlx_lm.models.cache import RotatingKVCache, ArraysCache
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        from vllm_mlx.kv_cache import CacheIndexMap
        import mlx.core as mx

        adapter = TurnCacheAdapter(None)
        # 3 layers: [rotating, recurrent, kv]
        adapter._cache_index_map = CacheIndexMap(
            kv_indices=[2],
            rotating_indices=[0],
            recurrent_indices=[1],
        )

        shadow = RotatingKVCache(max_size=4, keep=0)
        shadow.update_and_fetch(mx.ones((1, 1, 1, 4)), mx.ones((1, 1, 1, 4)))

        saved_recurrent = ArraysCache(1)
        saved_recurrent[0] = mx.full((1, 4), 55.0)

        extracted_cache = [
            {  # rotating, layer 0 — should be replaced by shadow
                "state": (mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4))),
                "meta_state": ("0", "4", "2", "2"),
                "class_name": "RotatingKVCache",
                "class_ref": RotatingKVCache,
            },
            {  # recurrent, layer 1 — should be replaced by saved ref
                "state": [mx.full((1, 4), 99.0)],
                "meta_state": "",
                "class_name": "ArraysCache",
                "class_ref": ArraysCache,
            },
            {  # kv, layer 2 — offset decremented
                "state": (mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4))),
                "meta_state": ("3",),
                "class_name": "BatchKVCache",
                "class_ref": None,
            },
        ]

        req = _make_mock_request_with_cache_state(
            n_minus_one_state={"rotating": [shadow], "recurrent": [saved_recurrent]}
        )
        result = adapter._reconstruct(req, extracted_cache)

        assert len(result) == 3
        # Layer 0: shadow has 1 token
        assert int(result[0]["meta_state"][2]) == 1
        # Layer 1: saved recurrent (55.0)
        mx.eval(*result[1]["state"])
        assert float(result[1]["state"][0][0, 0]) == pytest.approx(55.0)
        # Layer 2: KV offset 3 → 2
        assert result[2]["meta_state"][0] == "2"


class TestBuildPrefixCacheSSDWiring:
    def test_no_ssd_does_not_wrap_with_ssd_offloaded_cache(self):
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            ssd_cache_dir=None,
>>>>>>> Stashed changes
        )
        result = adapter.fetch(req)
    assert result is True
    assert req._cache_state.turn_path == [node]
    assert req._cache_state.hit_type == "hit"
    assert req._cache_state.cached_tokens == 5


def test_fetch_sets_prefill_boundaries_via_boundaries():
    """fetch() now populates prefill_boundaries by calling boundaries()."""
    inner = _make_inner()
    node = MagicMock()
    node.n_tokens = 5
    inner.match.return_value = ([node], None)
    ancestor = MagicMock()
    ancestor.n_tokens = 5
    inner.find_checkpoint_ancestor.return_value = ancestor
    inner.collect_path_data.return_value = ([], [])
    with patch.object(TurnCacheManager, "_assemble", return_value=[]), patch(
        "vllm_mlx.prefix_cache_adapters.validate_cache", return_value=True
    ):
        adapter = TurnCacheManager(inner)
        # Use boundaries [5, 8] so cached=5 excludes 5 but keeps 8 → [8-5]=[3]
        req = _make_request(
            prompt_token_ids=list(range(10)),
            turn_boundaries=[5, 8],
            cached_tokens=0,
        )
        result = adapter.fetch(req)
    # boundaries() adjusts for cached_tokens: [5,8] with cached=5 → [8-5]=[3]
    # Boundary at 5 is excluded because 5 > 5 is False
    assert result is True
    assert req._cache_state.prefill_boundaries == [3]


# ── store() ──────────────────────────────────────────────────────────────────


def test_store_uses_explicit_tokens_not_cache_state():
    """store(request, tokens, cache) must not read store_tokens from cs."""
    inner = _make_inner()
    adapter = TurnCacheManager(inner)
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[3],
        output_token_ids=[6, 7],
        cached_tokens=0,
    )
    req._cache_state.turn_path = []
    explicit_tokens = [1, 2, 3, 4, 5, 6]  # N-1 key passed by Scheduler
    result = adapter.store(req, explicit_tokens, [])
    # Adapter accepts the call; does not crash looking for cs.store_tokens
    assert isinstance(result, bool)


def test_store_reads_turn_path_from_cache_state():
    inner = _make_inner()
    parent_node = MagicMock(n_tokens=3)
    inner.root = MagicMock(n_tokens=0)
    adapter = TurnCacheManager(inner)
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[3],
        output_token_ids=[
            6,
            7,
        ],  # must be non-empty — store() returns False with no output
    )
    req._cache_state.turn_path = [parent_node]
    adapter.store(req, [1, 2, 3, 4, 5, 6], [])
    # insert should be called with parent_node as parent
    assert inner.insert.called
    call_parent = inner.insert.call_args[0][0]
    assert call_parent is parent_node


# ── on_prefill_checkpoint() ───────────────────────────────────────────────────


def test_on_prefill_checkpoint_at_boundary_inserts_node():
    from vllm_mlx.cache_types import KVLayerSegment
    from vllm_mlx.kv_cache import QuantizedArray

    inner = _make_inner()
    new_node = MagicMock(n_tokens=10)
    inner.insert.return_value = new_node
    adapter = TurnCacheManager(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    req._cache_state.turn_path = []
    extracted = [
        {"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("5",)}
    ]
    import mlx.core as mx

    qa = QuantizedArray(
        packed=mx.zeros((1, 1, 1, 1), dtype=mx.uint32),
        scales=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
        biases=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
    )
    kv_placeholder = KVLayerSegment(keys=qa, values=qa, metadata={"layer_index": 0})
    with patch.object(
        TurnCacheManager, "_segment", return_value=([kv_placeholder], [None])
    ):
        adapter.on_prefill_checkpoint(
            req, total_tokens_prefilled=10, extracted_cache=extracted
        )
    assert inner.insert.called
    assert new_node in req._cache_state.turn_path


def test_on_prefill_checkpoint_not_at_boundary_is_noop():
    inner = _make_inner()
    adapter = TurnCacheManager(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    extracted = [
        {"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("5",)}
    ]
    adapter.on_prefill_checkpoint(
        req, total_tokens_prefilled=7, extracted_cache=extracted
    )
    inner.insert.assert_not_called()


def test_on_prefill_checkpoint_does_not_read_n_minus_one_for_prefill():
    """Prefill boundaries store cache @ N, not N-1; n_minus_one_state must be ignored."""
    from vllm_mlx.cache_types import KVLayerSegment
    from vllm_mlx.kv_cache import QuantizedArray

    inner = _make_inner()
    adapter = TurnCacheManager(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    extracted = [
        {"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("10",)}
    ]
    # Verify that on_prefill_checkpoint delegates to _segment (not inner.split_cache_arrays)
    import mlx.core as mx

    qa = QuantizedArray(
        packed=mx.zeros((1, 1, 1, 1), dtype=mx.uint32),
        scales=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
        biases=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
    )
    kv_placeholder = KVLayerSegment(keys=qa, values=qa, metadata={"layer_index": 0})
    with patch.object(
        TurnCacheManager, "_segment", return_value=([kv_placeholder], [None])
    ) as mock_seg:
        adapter.on_prefill_checkpoint(
            req, total_tokens_prefilled=10, extracted_cache=extracted
        )
    # _segment should have been called with extracted_cache plus policy and group_size
    mock_seg.assert_called_once_with(
        extracted, policy=adapter._policy, group_size=adapter._kv_group_size
    )
    # inner.split_cache_arrays must NOT be called (it was the old API)
    inner.split_cache_arrays.assert_not_called()


# ── _segment() lazy-eval fix ─────────────────────────────────────────────────


def _rotating_state(B=1, H=2, T=16, D=64):
    """Return a RotatingKVCache state dict with float16 keys/values."""
    import mlx.core as mx

    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        # meta: (keep, max_size, offset, _idx)
        "meta_state": ("0", str(T), str(T), str(T)),
    }


def _kvcache_state(B=1, H=2, T=16, D=64):
    """Return a KVCache state dict with float16 keys/values."""
    import mlx.core as mx

    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {
        "class_name": "KVCache",
        "state": (keys, values),
        "meta_state": (str(T),),
    }


def test_segment_rotating_kvcache_quantized_arrays_are_evaluated():
    """KVLayerSegment arrays from the RotatingKVCache branch must be concrete
    (already evaluated) so that freeing the source float16 arrays does not
    leave dangling computation graph references in active Metal memory."""
    import mlx.core as mx

    state = _rotating_state()
    kv_list, _ = TurnCacheManager._segment([state], policy=KVQuantPolicy(sliding_bits=8, full_bits=8), group_size=32)
    seg = kv_list[0]
    assert seg is not None

    # Release source arrays and flush the Metal pool.
    del state
    mx.clear_cache()

    # If the quantized arrays were NOT evaluated inside _segment(), calling
    # mx.eval() here would attempt to resolve a computation graph whose input
    # buffers have been freed, producing incorrect data.  With the fix the
    # arrays are already concrete and this is a no-op that must not raise.
    mx.eval(seg.keys.packed, seg.keys.scales, seg.keys.biases)
    mx.eval(seg.values.packed, seg.values.scales, seg.values.biases)

    assert seg.keys.packed.nbytes > 0
    assert seg.values.packed.nbytes > 0


def test_segment_kvcache_quantized_arrays_are_evaluated():
    """KVLayerSegment arrays from the KVCache branch must be concrete."""
    import mlx.core as mx

    state = _kvcache_state()
    kv_list, _ = TurnCacheManager._segment([state], policy=KVQuantPolicy(full_bits=8), group_size=32)
    seg = kv_list[0]
    assert seg is not None

    del state
    mx.clear_cache()

    mx.eval(seg.keys.packed, seg.keys.scales, seg.keys.biases)
    mx.eval(seg.values.packed, seg.values.scales, seg.values.biases)

    assert seg.keys.packed.nbytes > 0
    assert seg.values.packed.nbytes > 0


@pytest.mark.skipif(
    not __import__("mlx.core", fromlist=["metal"]).metal.is_available(),
    reason="requires Metal GPU",
)
def test_segment_does_not_retain_source_float16_in_active_memory():
    """Source float16 buffers must not stay in active Metal memory after
    _segment() returns and the source references are dropped."""
    import mlx.core as mx

    # Establish baseline before any new allocations.
    mx.clear_cache()
    baseline = mx.get_active_memory()

    # Use a large tensor so the delta is measurable (8 MB float16).
    B, H, T, D = 1, 8, 128, 128
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    source_bytes = keys.nbytes + values.nbytes  # float16 footprint

    state = {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": ("0", str(T), str(T), str(T)),
    }

    kv_list, _ = TurnCacheManager._segment([state], policy=KVQuantPolicy(sliding_bits=8, full_bits=8), group_size=32)
    seg = kv_list[0]

    # Release all source references.
    del keys, values, state
    mx.eval(seg.keys.packed)  # ensure segment is materialised
    mx.clear_cache()

    active_after = mx.get_active_memory()
    # Without the fix: active_after ≈ baseline + source_bytes (float16 still
    # referenced via lazy computation graph in the trie).
    # With the fix: active_after ≈ baseline + quantized_bytes (< source_bytes).
    # Assert the float16 source is NOT retained: growth must be well under
    # the full float16 footprint (allow 50% to account for quantized arrays).
    growth = active_after - baseline
    assert growth < source_bytes * 0.75, (
        f"Source float16 ({source_bytes / 1e6:.1f} MB) appears retained: "
        f"baseline={baseline / 1e6:.1f} MB, active_after={active_after / 1e6:.1f} MB, "
        f"growth={growth / 1e6:.1f} MB"
    )
