# SPDX-License-Identifier: Apache-2.0
"""Tests for SSDOffloadedCache."""

import threading
import time
import pytest
from unittest.mock import MagicMock, call

from vllm_mlx.ssd_offloaded_cache import SSDOffloadedCache
from vllm_mlx.kv_cache import CacheHit


def _make_inner():
    inner = MagicMock()
    inner.fetch.return_value = None
    inner.store.return_value = True
    inner.get_stats.return_value = {}
    inner.set_spill_delegate = MagicMock()
    return inner


def _make_store():
    store = MagicMock()
    store.has.return_value = False
    store.read.return_value = None
    store.all_keys.return_value = iter([])
    return store


def _make_request(token_ids):
    req = MagicMock()
    req.prompt_token_ids = token_ids
    return req


class TestSSDOffloadedCacheInit:
    def test_registers_spill_delegate_on_construction(self):
        inner = _make_inner()
        store = _make_store()
        SSDOffloadedCache(inner, store)
        inner.set_spill_delegate.assert_called_once()

    def test_on_spill_writes_to_store_and_returns_handle(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        layers = [{"keys": [1, 2, 3]}]
        handle = cache._on_spill((1, 2, 3), layers)
        store.write.assert_called_once_with((1, 2, 3), layers)
        assert handle == (1, 2, 3)

    def test_on_promote_reads_from_store(self):
        inner = _make_inner()
        store = _make_store()
        layers = [{"keys": [4, 5]}]
        store.read.return_value = layers
        cache = SSDOffloadedCache(inner, store)
        result = cache._on_promote((1, 2))
        store.read.assert_called_once_with((1, 2))
        assert result == layers


class TestSSDOffloadedCacheFetch:
    def test_returns_inner_hit_when_inner_has_cache(self):
        inner = _make_inner()
        store = _make_store()
        hit = CacheHit(cache=[[1]], cached_tokens=3, remaining_tokens=[], hit_type="hit")
        inner.fetch.return_value = hit
        cache = SSDOffloadedCache(inner, store)
        req = _make_request([1, 2, 3])
        result = cache.fetch(req)
        assert result is hit

    def test_returns_none_on_miss_when_not_on_disk(self):
        inner = _make_inner()
        store = _make_store()
        store.has.return_value = False
        cache = SSDOffloadedCache(inner, store)
        result = cache.fetch(_make_request([1, 2, 3]))
        assert result is None

    def test_enqueues_promotion_when_entry_is_on_disk(self):
        inner = _make_inner()
        store = _make_store()
        store.has.return_value = True
        layers = [{"keys": [1]}]
        store.read.return_value = layers
        cache = SSDOffloadedCache(inner, store)
        cache.start()
        try:
            cache.fetch(_make_request([10, 20]))
            # Give the background thread time to promote
            time.sleep(0.1)
            # Second fetch should return promoted entry
            result = cache.fetch(_make_request([10, 20]))
            assert result is not None
            assert result.hit_type == "ssd_hit"
            assert result.cache == layers
        finally:
            cache.close()

    def test_does_not_double_enqueue_same_tokens(self):
        inner = _make_inner()
        store = _make_store()
        store.has.return_value = True
        store.read.return_value = [{"keys": [1]}]
        cache = SSDOffloadedCache(inner, store)
        cache.start()
        try:
            cache.fetch(_make_request([1, 2]))
            cache.fetch(_make_request([1, 2]))
            time.sleep(0.1)
        finally:
            cache.close()
        # read should have been called at most once per distinct token key
        assert store.read.call_count <= 1


class TestSSDOffloadedCacheDelegation:
    def test_store_delegates_to_inner(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        req = _make_request([1])
        layers = [[1, 2]]
        cache.store(req, layers)
        inner.store.assert_called_once_with(req, layers)

    def test_release_delegates_to_inner(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        cache.release("handle")
        inner.release.assert_called_once_with("handle")

    def test_clear_delegates_to_inner_and_clears_promoted(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        # Manually inject a promoted entry
        cache._promoted[(1, 2)] = [[1]]
        cache.clear()
        inner.clear.assert_called_once()
        assert len(cache._promoted) == 0

    def test_get_stats_delegates_to_inner(self):
        inner = _make_inner()
        store = _make_store()
        inner.get_stats.return_value = {"hits": 5}
        cache = SSDOffloadedCache(inner, store)
        assert cache.get_stats() == {"hits": 5}


class TestSSDOffloadedCacheLifecycle:
    def test_close_stops_background_thread(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        cache.start()
        assert cache._thread is not None and cache._thread.is_alive()
        cache.close()
        cache._thread.join(timeout=1.0)
        assert not cache._thread.is_alive()

    def test_close_before_start_is_a_noop(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        # Must not raise even though start() was never called.
        cache.close()


class TestSSDOffloadedCachePersistence:
    def test_save_writes_promoted_entries_to_store(self):
        inner = _make_inner()
        store = _make_store()
        cache = SSDOffloadedCache(inner, store)
        cache._promoted[(1, 2)] = [{"keys": [1]}]
        cache.save()
        store.write.assert_called_once_with((1, 2), [{"keys": [1]}])

    def test_load_populates_promoted_from_disk(self):
        inner = _make_inner()
        store = _make_store()
        layers = [{"keys": [5]}]
        store.all_keys.return_value = iter([(3, 4)])
        store.read.return_value = layers
        cache = SSDOffloadedCache(inner, store)
        count = cache.load()
        assert count == 1
        assert cache._promoted[(3, 4)] == layers

    def test_failed_promotion_clears_in_flight(self):
        inner = _make_inner()
        store = _make_store()
        store.has.return_value = True
        store.read.return_value = None  # read fails
        cache = SSDOffloadedCache(inner, store)
        cache.start()
        try:
            cache.fetch(_make_request([1, 2, 3]))
            time.sleep(0.1)
            # After failed promotion, token should no longer be in_flight
            assert (1, 2, 3) not in cache._in_flight
        finally:
            cache.close()
