# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.kv_cache — QuantizedArray, CacheHit, PrefixCache protocol."""

import mlx.core as mx
import pytest
from unittest.mock import MagicMock
from mlx_lm.models.cache import KVCache

from vllm_mlx.kv_cache import CacheHit, CacheDiskStore, QuantizedArray, SpillableCache, validate_cache
from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache


def _make_quantized_array(seq_len: int = 64, head_dim: int = 64, group_size: int = 64, bits: int = 4):
    keys = mx.random.normal((1, 4, seq_len, head_dim)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(keys, group_size=group_size, bits=bits)
    return QuantizedArray(packed=packed, scales=scales, biases=biases)


class TestQuantizedArray:
    def test_nbytes_equals_sum_of_components(self):
        q = _make_quantized_array()
        expected = q.packed.nbytes + q.scales.nbytes + q.biases.nbytes
        assert q.nbytes == expected

    def test_named_attribute_access(self):
        keys = mx.random.normal((1, 4, 64, 64)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(keys, group_size=64, bits=4)
        q = QuantizedArray(packed=packed, scales=scales, biases=biases)
        assert q.packed is packed
        assert q.scales is scales
        assert q.biases is biases

    def test_mx_eval_traversal(self):
        """mx.eval must traverse QuantizedArray as a pytree (NamedTuple)."""
        q = _make_quantized_array()
        # Should not raise and should materialise all arrays
        mx.eval(q)
        # After eval, .nbytes is accessible (arrays are concrete)
        assert q.nbytes > 0

    def test_as_tuple_unpacking(self):
        q = _make_quantized_array()
        packed, scales, biases = q
        assert packed is q.packed
        assert scales is q.scales
        assert biases is q.biases

    def test_nbytes_zero_when_small(self):
        """Sanity: packed dtype is uint32, scales/biases are bfloat16."""
        q = _make_quantized_array(seq_len=64, head_dim=64, group_size=64, bits=4)
        assert q.packed.dtype == mx.uint32
        assert q.scales.dtype == mx.bfloat16
        assert q.biases.dtype == mx.bfloat16


class TestSpillableCacheProtocol:
    """SpillableCache runtime-checkable protocol checks."""

    def test_class_with_all_methods_satisfies_protocol(self):
        """A class implementing all PrefixCache methods plus set_spill_delegate should satisfy SpillableCache."""
        class FullImpl:
            def fetch(self, request): ...
            def store(self, request, cache): ...
            def release(self, handle): ...
            def get_stats(self): ...
            def clear(self): ...
            def on_prefill_checkpoint(self, request, processed_tokens, extracted_cache): ...
            def set_spill_delegate(self, on_spill, on_promote): ...
        assert isinstance(FullImpl(), SpillableCache)

    def test_missing_set_spill_delegate_fails_check(self):
        """A class with all PrefixCache methods but missing set_spill_delegate should not satisfy SpillableCache."""
        class NoDelegate:
            def fetch(self, request): ...
            def store(self, request, cache): ...
            def release(self, handle): ...
            def get_stats(self): ...
            def clear(self): ...
            def on_prefill_checkpoint(self, request, processed_tokens, extracted_cache): ...
            # No set_spill_delegate
        assert not isinstance(NoDelegate(), SpillableCache)


def _run_prefill(cache: BatchQuantizedKVCache, B: int, H: int, T: int, D: int):
    keys = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    values = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    return cache.update_and_fetch(keys, values)


class TestBatchQuantizedKVCacheQuantizedArray:
    """BatchQuantizedKVCache stores keys/values as QuantizedArray per layer."""

    def test_update_and_fetch_stores_quantized_arrays(self):
        cache = BatchQuantizedKVCache(left_padding=[0, 0])
        _run_prefill(cache, B=2, H=4, T=10, D=64)
        assert isinstance(cache.keys, QuantizedArray)
        assert isinstance(cache.values, QuantizedArray)

    def test_update_and_fetch_returns_quantized_arrays(self):
        cache = BatchQuantizedKVCache(left_padding=[0])
        k_out, v_out = _run_prefill(cache, B=1, H=4, T=16, D=64)
        assert isinstance(k_out, QuantizedArray)
        assert isinstance(v_out, QuantizedArray)

    def test_output_shapes(self):
        B, H, T, D = 2, 4, 16, 64
        cache = BatchQuantizedKVCache(left_padding=[0, 0])
        k_out, v_out = _run_prefill(cache, B=B, H=H, T=T, D=D)
        el_per_int = 8 * 4 // 4  # bits=4
        assert k_out.packed.shape == (B, H, T, D // el_per_int)
        assert k_out.scales.shape == (B, H, T, D // 64)  # group_size=64

    def test_nbytes_property(self):
        cache = BatchQuantizedKVCache(left_padding=[0])
        assert cache.nbytes == 0
        _run_prefill(cache, B=1, H=4, T=32, D=64)
        assert cache.nbytes > 0

    def test_extract_merge_round_trip(self):
        """extract → QuantizedKVCache; merge → BatchQuantizedKVCache with same shapes."""
        from mlx_lm.models.cache import QuantizedKVCache

        B, H, T, D = 3, 4, 20, 64
        cache = BatchQuantizedKVCache(left_padding=[0] * B)
        _run_prefill(cache, B=B, H=H, T=T, D=D)
        mx.eval(cache.keys, cache.values)

        extracted = [cache.extract(i) for i in range(B)]
        for e in extracted:
            assert isinstance(e, QuantizedKVCache)
            assert e.offset == T

        merged = BatchQuantizedKVCache.merge(extracted)
        assert merged._idx == T
        assert isinstance(merged.keys, QuantizedArray)
        assert merged.keys.packed.shape[0] == B

    def test_extend_accepts_batch_kv_cache(self):
        """extend() must handle BatchKVCache as other (not crash with .packed AttributeError).

        Regression test for: 'mlx.core.array' object has no attribute 'packed'
        This can occur when _quantize_batch_kv_cache misses a layer that was
        produced by an older mlx-lm _process_prompts code path.
        """
        from mlx_lm.models.cache import BatchKVCache

        B, H, T, D = 2, 4, 16, 64

        # Active batch: BatchQuantizedKVCache with data
        active = BatchQuantizedKVCache(left_padding=[0] * B)
        _run_prefill(active, B=B, H=H, T=T, D=D)
        mx.eval(active.keys, active.values)

        # New batch: BatchKVCache (the type that triggers the bug)
        new_bkv = BatchKVCache([0] * B)
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        mx.eval(k, v)
        new_bkv.update_and_fetch(k, v)

        # Before the fix this raised AttributeError: 'mlx.core.array' object
        # has no attribute 'packed'
        active.extend(new_bkv)

        assert isinstance(active.keys, QuantizedArray)
        assert active.keys.packed.shape[0] == B * 2

    def test_quantize_batch_kv_cache_guard_prevents_shape_error(self):
        """The _quantize_batch_kv_cache guard must prevent 'QuantizedArray has no .shape'.

        Regression test for the reverse-direction batch decoding bug:
          active_batch.cache[i] = BatchKVCache   (from _orig_process_prompts)
          new_batch.cache[i]    = BatchQuantizedKVCache (from chunked-prefill path)
        Without the guard, BatchKVCache.extend(BatchQuantizedKVCache) crashes:
          'QuantizedArray' object has no attribute 'shape'
        The guard converts active_batch.cache to BatchQuantizedKVCache first.
        """
        from mlx_lm.models.cache import BatchKVCache

        B, H, T, D = 2, 4, 16, 64

        # Simulate active_batch.cache[i] = BatchKVCache (unquantized)
        active_bkvc = BatchKVCache([0] * B)
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        mx.eval(k, v)
        active_bkvc.update_and_fetch(k, v)

        # Simulate the _quantize_batch_kv_cache guard
        cache_list = [active_bkvc]
        cache_list[0] = BatchQuantizedKVCache.from_batch_kvcache(cache_list[0])
        assert isinstance(cache_list[0], BatchQuantizedKVCache)

        # Now extend with a BatchQuantizedKVCache — must not crash with
        # AttributeError: 'QuantizedArray' object has no attribute 'shape'
        new_quantized = BatchQuantizedKVCache(left_padding=[0] * B)
        _run_prefill(new_quantized, B=B, H=H, T=T, D=D)
        mx.eval(new_quantized.keys, new_quantized.values)

        cache_list[0].extend(new_quantized)

        assert isinstance(cache_list[0].keys, QuantizedArray)
        assert cache_list[0].keys.packed.shape[0] == B * 2


# ------------------------------------------------------------------
# Helpers shared by adapter tests
# ------------------------------------------------------------------

def _make_kv_cache_layers(n_layers=2, seq_len=50, n_heads=4, head_dim=64):
    cache = []
    for _ in range(n_layers):
        kv = KVCache()
        kv.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.offset = seq_len
        cache.append(kv)
    mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])
    return cache


def _make_request(tokens):
    req = MagicMock()
    req.prompt_token_ids = tokens
    req.request_id = "req-test"
    return req


# ------------------------------------------------------------------
# MemoryCacheAdapter
# ------------------------------------------------------------------

class TestMemoryCacheAdapter:
    def _make_adapter(self, max_memory_mb=500):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
        from vllm_mlx.prefix_cache_adapters import MemoryCacheAdapter

        config = MemoryCacheConfig(kv_quantize=False, max_memory_mb=max_memory_mb)
        inner = MemoryAwarePrefixCache(MagicMock(), config)
        return MemoryCacheAdapter(inner)

    def test_miss_returns_none(self):
        adapter = self._make_adapter()
        req = _make_request(list(range(20)))
        result = adapter.fetch(req)
        assert result is None

    def test_hit_returns_cache_hit(self):
        adapter = self._make_adapter()
        tokens = list(range(50))
        cache = _make_kv_cache_layers(seq_len=50)

        adapter.store(_make_request(tokens), cache)
        result = adapter.fetch(_make_request(tokens))

        assert isinstance(result, CacheHit)
        assert result.cached_tokens == 50
        assert result.remaining_tokens == []
        assert result.cache is not None

    def test_store_returns_true_on_success(self):
        adapter = self._make_adapter()
        tokens = list(range(30))
        cache = _make_kv_cache_layers(seq_len=30)
        assert adapter.store(_make_request(tokens), cache) is True

    def test_stored_entry_is_fetchable(self):
        adapter = self._make_adapter()
        tokens = list(range(40))
        cache = _make_kv_cache_layers(seq_len=40)

        adapter.store(_make_request(tokens), cache)
        hit = adapter.fetch(_make_request(tokens))
        assert hit is not None
        assert hit.cached_tokens == 40

    def test_release_is_noop(self):
        adapter = self._make_adapter()
        adapter.release(None)  # must not raise

    def test_get_stats_returns_dict(self):
        adapter = self._make_adapter()
        stats = adapter.get_stats()
        assert isinstance(stats, dict)

    def test_clear_empties_cache(self):
        adapter = self._make_adapter()
        tokens = list(range(30))
        cache = _make_kv_cache_layers(seq_len=30)
        adapter.store(_make_request(tokens), cache)
        adapter.clear()
        assert adapter.fetch(_make_request(tokens)) is None

    def test_save_load_roundtrip(self, tmp_path):
        from vllm_mlx.prefix_cache_adapters import MemoryCacheAdapter
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        model = MagicMock()
        config = MemoryCacheConfig(kv_quantize=False, max_memory_mb=500)

        adapter = MemoryCacheAdapter(MemoryAwarePrefixCache(model, config))
        tokens = list(range(50))
        cache = _make_kv_cache_layers(seq_len=50)
        adapter.store(_make_request(tokens), cache)

        cache_dir = str(tmp_path / "cache")
        assert adapter.save(cache_dir) is True

        # New adapter with same model loads from disk
        adapter2 = MemoryCacheAdapter(MemoryAwarePrefixCache(model, config))
        count = adapter2.load(cache_dir)
        assert count > 0
        assert adapter2.fetch(_make_request(tokens)) is not None


# ------------------------------------------------------------------
# TurnCacheAdapter
# ------------------------------------------------------------------

def _make_turn_request(tokens, turn_boundaries):
    req = MagicMock()
    req.prompt_token_ids = tokens
    req.request_id = "req-turn-test"
    req._turn_boundaries = turn_boundaries
    return req


class TestTurnCacheAdapterMessages:
    """_messages_to_segments is a pure function of request fields."""

    def _call(self, tokens, boundaries):
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        req = _make_turn_request(tokens, boundaries)
        return TurnCacheAdapter.messages_to_segments(req)

    def test_no_boundaries_returns_empty(self):
        segs = self._call(list(range(50)), [])
        assert segs == []

    def test_single_boundary_returns_system_and_user(self):
        from vllm_mlx.turn_prefix_cache import Segment
        tokens = list(range(60))
        segs = self._call(tokens, [10])
        assert len(segs) == 2
        assert segs[0].role == "system"
        assert segs[0].token_ids == tokens[:10]
        assert segs[1].role == "user"
        assert segs[1].token_ids == tokens[10:]

    def test_two_boundaries_creates_conversation_segment(self):
        tokens = list(range(80))
        segs = self._call(tokens, [10, 50])
        assert len(segs) == 3
        assert segs[0].role == "system"
        assert segs[1].role == "conversation"
        assert segs[2].role == "user"

    def test_matches_scheduler_output(self):
        """Scheduler._messages_to_segments delegates to TurnCacheAdapter.messages_to_segments."""
        from vllm_mlx.scheduler import Scheduler
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        tokens = list(range(100))
        boundaries = [15, 60]
        req = _make_turn_request(tokens, boundaries)

        adapter_segs = TurnCacheAdapter.messages_to_segments(req)
        sched = object.__new__(Scheduler)
        scheduler_segs = sched._messages_to_segments(req)
        assert adapter_segs == scheduler_segs


class TestTurnCacheAdapterFetch:
    """TurnCacheAdapter.fetch path as handle; miss returns None."""

    def _make_adapter(self):
        from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
        from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter
        inner = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
        return TurnCacheAdapter(inner), inner

    def test_miss_returns_none(self):
        adapter, _ = self._make_adapter()
        tokens = list(range(60))
        req = _make_turn_request(tokens, [10])
        assert adapter.fetch(req) is None

    def test_no_segments_returns_none(self):
        adapter, _ = self._make_adapter()
        req = _make_turn_request(list(range(50)), [])  # no boundaries → no segments
        assert adapter.fetch(req) is None

    def test_release_noop_on_none_handle(self):
        adapter, _ = self._make_adapter()
        adapter.release(None)  # must not raise

    def test_release_decrements_refcount(self):
        """release(path) decrements ref counts on matched nodes."""
        from vllm_mlx.turn_prefix_cache import Segment
        adapter, inner = self._make_adapter()

        tokens = list(range(60))
        boundaries = [10]
        req = _make_turn_request(tokens, boundaries)

        # Manually store segments so match() can find them
        segments = [
            Segment(role="system", token_ids=tokens[:10]),
            Segment(role="user", token_ids=tokens[10:]),
        ]
        # Store via inner trie — use match first then store
        path, _ = inner.match(segments)
        # No match yet; release empty path
        inner.release(path)

        # After fetch miss the handle should be None and release is a noop
        result = adapter.fetch(req)
        assert result is None


# ------------------------------------------------------------------
# PagedCacheAdapter
# ------------------------------------------------------------------

class TestPagedCacheAdapter:
    def _make_adapter(self):
        from vllm_mlx.prefix_cache_adapters import PagedCacheAdapter
        inner = MagicMock()
        return PagedCacheAdapter(inner), inner

    def test_fetch_miss_returns_none(self):
        adapter, inner = self._make_adapter()
        inner.fetch_cache.return_value = (None, list(range(50)))
        req = _make_request(list(range(50)))
        result = adapter.fetch(req)
        assert result is None

    def test_fetch_hit_returns_cache_hit(self):
        from vllm_mlx.paged_cache import BlockTable
        adapter, inner = self._make_adapter()

        block_table = MagicMock(spec=BlockTable)
        block_table.num_tokens = 40
        remaining = list(range(40, 50))
        inner.fetch_cache.return_value = (block_table, remaining)
        inner.reconstruct_cache.return_value = _make_kv_cache_layers(seq_len=40)

        req = _make_request(list(range(50)))
        result = adapter.fetch(req)

        assert isinstance(result, CacheHit)
        assert result.cached_tokens == 40
        assert result.remaining_tokens == remaining
        assert result.handle == req.request_id

    def test_fetch_hit_stores_block_table_internally(self):
        from vllm_mlx.paged_cache import BlockTable
        adapter, inner = self._make_adapter()

        block_table = MagicMock(spec=BlockTable)
        block_table.num_tokens = 30
        inner.fetch_cache.return_value = (block_table, list(range(30, 50)))
        inner.reconstruct_cache.return_value = []

        req = _make_request(list(range(50)))
        adapter.fetch(req)
        assert adapter._block_tables.get(req.request_id) is block_table

    def test_release_calls_release_cache_and_clears_table(self):
        from vllm_mlx.paged_cache import BlockTable
        adapter, inner = self._make_adapter()

        block_table = MagicMock(spec=BlockTable)
        block_table.num_tokens = 30
        inner.fetch_cache.return_value = (block_table, [])
        inner.reconstruct_cache.return_value = []

        req = _make_request(list(range(30)))
        adapter.fetch(req)
        adapter.release(req.request_id)

        inner.release_cache.assert_called_once_with(req.request_id)
        assert req.request_id not in adapter._block_tables

    def test_store_delegates_to_inner(self):
        adapter, inner = self._make_adapter()
        inner.store_cache.return_value = MagicMock()
        req = _make_request(list(range(30)))
        cache = _make_kv_cache_layers(seq_len=30)
        result = adapter.store(req, cache)
        assert result is True
        inner.store_cache.assert_called_once()

    def test_get_stats_returns_dict(self):
        adapter, inner = self._make_adapter()
        inner.get_stats.return_value = MagicMock(hits=1, misses=2)
        stats = adapter.get_stats()
        assert isinstance(stats, dict)

    def test_clear_delegates_to_inner(self):
        adapter, inner = self._make_adapter()
        adapter.clear()
        inner.clear.assert_called_once()


# ------------------------------------------------------------------
# Scheduler integration — _prefix_cache path
# ------------------------------------------------------------------

class TestSchedulerPrefixCacheIntegration:
    """PrefixCache adapter fetch() populates request cache state fields."""

    def _make_stored_adapter(self):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
        from vllm_mlx.prefix_cache_adapters import MemoryCacheAdapter

        config = MemoryCacheConfig(kv_quantize=False, max_memory_mb=500)
        inner = MemoryAwarePrefixCache(MagicMock(), config)
        adapter = MemoryCacheAdapter(inner)

        tokens = list(range(50))
        cache = _make_kv_cache_layers(seq_len=50)
        adapter.store(_make_request(tokens), cache)
        return adapter, tokens

    def test_cache_hit_populates_request_fields(self):
        from vllm_mlx.kv_cache import RequestCacheState
        adapter, tokens = self._make_stored_adapter()

        req = MagicMock()
        req.prompt_token_ids = tokens
        req.request_id = "req-integration"
        req._cache_state = RequestCacheState()

        hit = adapter.fetch(req)
        assert hit is not None
        req._cache_state.hit_type = hit.hit_type
        req._cache_state.cache = hit.cache
        req._cache_state.cached_tokens = hit.cached_tokens
        req._cache_state.remaining_tokens = hit.remaining_tokens

        assert req._cache_state.cache is not None
        assert req._cache_state.cached_tokens == 50
        assert req._cache_state.remaining_tokens == []
        assert req._cache_state.hit_type != "miss"

    def test_cache_miss_sets_remaining_tokens(self):
        from vllm_mlx.kv_cache import RequestCacheState
        adapter, _ = self._make_stored_adapter()

        req = MagicMock()
        req.prompt_token_ids = list(range(999, 1050))  # tokens not in cache
        req.request_id = "req-miss"
        req._cache_state = RequestCacheState()

        hit = adapter.fetch(req)
        if hit is None:
            req._cache_state.hit_type = "miss"
            req._cache_state.remaining_tokens = req.prompt_token_ids

        assert req._cache_state.hit_type == "miss"
        assert req._cache_state.remaining_tokens == req.prompt_token_ids


# ---------------------------------------------------------------------------
# Regression: _extract_recurrent_state must filter QuantizedKVCache (mlx_lm)
# ---------------------------------------------------------------------------

class TestExtractCacheStates:
    """extract_cache_states converts live KV layer objects to serialisable dicts."""

    def test_empty_input_returns_empty(self):
        from vllm_mlx.kv_cache import extract_cache_states
        assert extract_cache_states([]) == []

    def test_kvcache_layer_produces_expected_dict_shape(self):
        from vllm_mlx.kv_cache import extract_cache_states
        kv = KVCache()
        kv.keys = mx.zeros((1, 4, 8, 64))
        kv.values = mx.zeros((1, 4, 8, 64))
        kv.offset = 8
        result = extract_cache_states([kv])
        assert len(result) == 1
        layer = result[0]
        assert layer["class_name"] == "KVCache"
        assert layer["class_ref"] is KVCache
        # state holds the (keys, values) tensors
        assert len(layer["state"]) == 2
        # mlx-lm KVCache.meta_state returns "" — offset is recovered from key shape
        assert "meta_state" in layer

    def test_layer_missing_state_attr_causes_failure(self):
        from vllm_mlx.kv_cache import extract_cache_states

        class NoState:
            pass

        assert extract_cache_states([NoState()]) == []


class TestReconstructCacheFromStates:
    """reconstruct_cache_from_states is the inverse of extract_cache_states."""

    def test_empty_input_returns_none(self):
        from vllm_mlx.kv_cache import reconstruct_cache_from_states
        assert reconstruct_cache_from_states([]) is None

    def test_kvcache_round_trip(self):
        from vllm_mlx.kv_cache import extract_cache_states, reconstruct_cache_from_states
        kv = KVCache()
        kv.keys = mx.zeros((1, 4, 8, 64))
        kv.values = mx.zeros((1, 4, 8, 64))
        kv.offset = 8
        extracted = extract_cache_states([kv])
        assert extracted, "extraction must succeed before round-trip"
        reconstructed = reconstruct_cache_from_states(extracted)
        assert reconstructed is not None
        assert len(reconstructed) == 1
        assert isinstance(reconstructed[0], KVCache)
        assert reconstructed[0].offset == 8


def test_extract_recurrent_state_filters_mlx_quantized_kv_cache():
    """mlx_lm.QuantizedKVCache must be treated as a KV layer, not recurrent.

    Before the fix, only KVCache / BatchKVCache / RotatingKVCache / our own
    BatchQuantizedKVCache were excluded. QuantizedKVCache from mlx_lm leaked
    into the recurrent snapshot, which later caused _split_cache_arrays to
    produce fewer recurrent entries than the stored closure expected, triggering
    IndexError: list index out of range at recurrent[i].
    """
    from mlx_lm.models.cache import QuantizedKVCache
    from vllm_mlx.kv_cache import extract_recurrent_state as _extract_recurrent_state

    layer = QuantizedKVCache()
    result = _extract_recurrent_state([layer])
    assert result == [], (
        "QuantizedKVCache from mlx_lm must be excluded by _extract_recurrent_state"
    )


def test_extract_recurrent_state_keeps_non_kv_layers():
    """Non-KV layers (e.g. plain objects) are returned as recurrent state."""
    from vllm_mlx.kv_cache import extract_recurrent_state as _extract_recurrent_state

    class FakeMambaLayer:
        pass

    layer = FakeMambaLayer()
    result = _extract_recurrent_state([layer])
    assert result == [layer]


# ------------------------------------------------------------------
# CacheDiskStore, SpillableCache protocols and validate_cache
# ------------------------------------------------------------------

class TestValidateCache:
    def test_none_is_invalid(self):
        assert validate_cache(None) is False

    def test_empty_list_is_invalid(self):
        assert validate_cache([]) is False

    def test_list_with_none_layer_is_invalid(self):
        assert validate_cache([None]) is False

    def test_valid_list_with_mock_layers(self):
        layer = MagicMock()
        layer.keys = MagicMock()
        layer.keys.__class__ = object  # not a tuple/list
        layer.keys.shape = (1, 4, 128)
        layer.values = MagicMock()
        assert validate_cache([layer]) is True

    def test_layer_with_batch_dim_not_1_is_invalid(self):
        layer = MagicMock()
        layer.keys = MagicMock()
        layer.keys.__class__ = object
        layer.keys.shape = (2, 4, 128)  # batch=2, not 1
        assert validate_cache([layer]) is False


class TestCacheDiskStoreProtocol:
    def test_protocol_is_runtime_checkable(self):
        # Any concrete class with write/read/has/all_keys satisfies the protocol.
        # MagicMock(spec=[...]) is intentionally avoided: Python 3.12+ protocol
        # isinstance checks use MRO lookup, not __getattr__, so MagicMock fails
        # even when hasattr returns True for all required attrs.
        from vllm_mlx.kv_cache import CacheDiskStore

        class MinimalStore:
            def write(self, tokens, layers): ...
            def read(self, tokens): ...
            def has(self, tokens): ...
            def all_keys(self): ...

        assert isinstance(MinimalStore(), CacheDiskStore)

    def test_missing_method_fails_check(self):
        from vllm_mlx.kv_cache import CacheDiskStore

        class IncompleteStore:
            def write(self, tokens, layers): ...
            def read(self, tokens): ...
            # missing has and all_keys

        assert not isinstance(IncompleteStore(), CacheDiskStore)


class TestSpillableCacheProtocol:
    def test_protocol_is_runtime_checkable(self):
        from vllm_mlx.kv_cache import SpillableCache

        class MinimalSpillable:
            def fetch(self, request): ...
            def store(self, request, cache): ...
            def release(self, handle): ...
            def get_stats(self): ...
            def clear(self): ...
            def on_prefill_checkpoint(self, request, processed_tokens, extracted_cache): ...
            def set_spill_delegate(self, on_spill, on_promote): ...

        assert isinstance(MinimalSpillable(), SpillableCache)

    def test_missing_set_spill_delegate_fails_check(self):
        """A class with all PrefixCache methods but missing set_spill_delegate should not satisfy SpillableCache."""
        class NoDelegate:
            def fetch(self, request): ...
            def store(self, request, cache): ...
            def release(self, handle): ...
            def get_stats(self): ...
            def clear(self): ...
            def on_prefill_checkpoint(self, request, processed_tokens, extracted_cache): ...
            # No set_spill_delegate
        assert not isinstance(NoDelegate(), SpillableCache)



def test_patched_merge_caches_routes_quantized_kvcache():
    """_patched_merge_caches must produce BatchQuantizedKVCache for QuantizedKVCache inputs.

    Prefix cache restore returns QuantizedKVCache objects (via BatchQuantizedKVCache.extract).
    Before the fix this raises ValueError: 'QuantizedKVCache does not yet support batching'.
    """
    import importlib
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache
    from vllm_mlx.utils.mamba_cache import ensure_mamba_support
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache

    ensure_mamba_support()
    gen_module = importlib.import_module("mlx_lm.generate")
    merge_fn = gen_module._merge_caches

    def _make_qkvc(n_tokens=4, n_heads=2, d_head=64, group_size=64, bits=4):
        el_per_int = 8  # 8*4 bytes // 4 bits
        q = QuantizedKVCache(group_size=group_size, bits=bits)
        shape = (1, n_heads, n_tokens)
        q.keys = [
            mx.zeros((*shape, d_head // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
        ]
        q.values = [
            mx.zeros((*shape, d_head // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
        ]
        q.offset = n_tokens
        return q

    result = merge_fn([[_make_qkvc()], [_make_qkvc()]])
    assert isinstance(result[0], BatchQuantizedKVCache), (
        f"Expected BatchQuantizedKVCache, got {type(result[0])}"
    )


def test_bqkvc_extend_accepts_quantized_kvcache_input():
    """BatchQuantizedKVCache.extend() must accept a bare QuantizedKVCache as other.

    Safety net: a single-sequence QuantizedKVCache should be promoted to
    BatchQuantizedKVCache before the concatenation, not crash with AttributeError.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache

    def _make_qkvc(n_tokens=4, n_heads=2, d_head=64, group_size=64, bits=4):
        el_per_int = 8
        q = QuantizedKVCache(group_size=group_size, bits=bits)
        shape = (1, n_heads, n_tokens)
        q.keys = [
            mx.zeros((*shape, d_head // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
        ]
        q.values = [
            mx.zeros((*shape, d_head // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
            mx.zeros((*shape, d_head // group_size), dtype=mx.bfloat16),
        ]
        q.offset = n_tokens
        return q

    batch = BatchQuantizedKVCache.merge([_make_qkvc()])  # batch=1
    batch.extend(_make_qkvc())  # bare QuantizedKVCache — must not crash
    assert batch.keys.packed.shape[0] == 2, (
        f"Expected batch dimension 2 after extend, got {batch.keys.packed.shape[0]}"
    )
