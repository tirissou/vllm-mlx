# SPDX-License-Identifier: Apache-2.0
"""Tests for FilesystemCacheDiskStore."""

import numpy as np
import pytest

from vllm_mlx.ssd_cache import FilesystemCacheDiskStore
from vllm_mlx.kv_cache import CacheDiskStore


@pytest.fixture
def store(tmp_path):
    return FilesystemCacheDiskStore(cache_dir=str(tmp_path))


class TestFilesystemCacheDiskStore:
    def test_implements_protocol(self, store):
        assert isinstance(store, CacheDiskStore)

    def test_has_returns_false_for_missing(self, store):
        assert store.has((1, 2, 3)) is False

    def test_write_then_has(self, store):
        layers = [{"keys": np.zeros((1, 4, 8)), "values": np.zeros((1, 4, 8))}]
        store.write((1, 2, 3), layers)
        assert store.has((1, 2, 3)) is True

    def test_read_roundtrip(self, store):
        layers = [{"keys": np.zeros((1, 4, 8)), "values": np.ones((1, 4, 8))}]
        store.write((10, 20), layers)
        result = store.read((10, 20))
        assert result is not None
        assert len(result) == 1
        np.testing.assert_array_equal(result[0]["values"], np.ones((1, 4, 8)))

    def test_read_missing_returns_none(self, store):
        assert store.read((99, 98, 97)) is None

    def test_all_keys_empty_initially(self, store):
        assert list(store.all_keys()) == []

    def test_all_keys_after_writes(self, store):
        layers = [{"keys": np.zeros((1, 2, 4)), "values": np.zeros((1, 2, 4))}]
        store.write((1, 2), layers)
        store.write((3, 4), layers)
        keys = set(store.all_keys())
        assert (1, 2) in keys
        assert (3, 4) in keys

    def test_different_token_keys_dont_collide(self, store):
        a = [{"keys": np.zeros((1, 1, 4)), "values": np.zeros((1, 1, 4))}]
        b = [{"keys": np.ones((1, 1, 4)), "values": np.ones((1, 1, 4))}]
        store.write((1,), a)
        store.write((2,), b)
        result_a = store.read((1,))
        result_b = store.read((2,))
        np.testing.assert_array_equal(result_a[0]["keys"], np.zeros((1, 1, 4)))
        np.testing.assert_array_equal(result_b[0]["keys"], np.ones((1, 1, 4)))

    def test_persistence_across_instances(self, tmp_path):
        """Entries written by one instance are readable by a new instance (models restart)."""
        layers = [{"keys": np.zeros((1, 2, 4)), "values": np.zeros((1, 2, 4))}]
        store1 = FilesystemCacheDiskStore(cache_dir=str(tmp_path))
        store1.write((5, 6, 7), layers)
        # Construct a fresh instance pointing to the same dir
        store2 = FilesystemCacheDiskStore(cache_dir=str(tmp_path))
        assert store2.has((5, 6, 7))
        result = store2.read((5, 6, 7))
        assert result is not None
        np.testing.assert_array_equal(result[0]["keys"], np.zeros((1, 2, 4)))
