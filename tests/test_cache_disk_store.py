# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx/cache_disk_store.py."""

import pytest


def test_types_importable():
    from vllm_mlx.cache_disk_store import (
        CacheDiskStore,
        CacheDiskError,
        CacheMissDuringWalk,
        CachePolicyMismatchError,
        DiskStoreFullError,
        IncompatibleCacheDirError,
        MissingCacheClassError,
        NodeHeader,
        NodePayload,
        SSDRef,
    )
    assert CacheDiskStore is not None
    assert issubclass(CacheMissDuringWalk, Exception)
    assert issubclass(CachePolicyMismatchError, CacheDiskError)
    assert issubclass(DiskStoreFullError, CacheDiskError)
    assert issubclass(IncompatibleCacheDirError, CacheDiskError)
    assert issubclass(MissingCacheClassError, CacheDiskError)


def test_ssd_ref_holds_key():
    from vllm_mlx.cache_disk_store import SSDRef
    ref = SSDRef(key=(0, (1, 2, 3)))
    assert ref.key == (0, (1, 2, 3))


def test_node_payload_fields():
    from vllm_mlx.cache_disk_store import NodePayload
    payload = NodePayload(
        parent_key=None,
        token_ids=(1, 2),
        n_tokens_cumulative=2,
        last_access_ts=0.0,
        kv_layers=[],
        recurrent_layers=[],
    )
    assert payload.parent_key is None
    assert payload.token_ids == (1, 2)


def test_node_header_fields():
    from vllm_mlx.cache_disk_store import NodeHeader
    header = NodeHeader(
        parent_key=None,
        token_ids=(1, 2),
        n_tokens_cumulative=2,
        last_access_ts=0.0,
        size_bytes=128,
        child_count=0,
        layer_bits=(8, None),
        layer_class_names=("KVCache", "RotatingKVCache"),
        recurrent_class_paths=(),
        kv_group_size=64,
    )
    assert header.size_bytes == 128


import mlx.core as mx


def _make_kv_segment(layer_index: int, n_tokens: int, bits: int = 8):
    """Build a KVLayerSegment with a real q8 QuantizedArray for the given shape."""
    from vllm_mlx.cache_types import KVLayerSegment
    from vllm_mlx.kv_cache import QuantizedArray

    keys = mx.random.normal(shape=(1, 4, n_tokens, 64), dtype=mx.bfloat16)
    values = mx.random.normal(shape=(1, 4, n_tokens, 64), dtype=mx.bfloat16)
    q_keys = QuantizedArray(*mx.quantize(keys, group_size=64, bits=bits))
    q_values = QuantizedArray(*mx.quantize(values, group_size=64, bits=bits))
    mx.eval(q_keys.packed, q_keys.scales, q_keys.biases,
            q_values.packed, q_values.scales, q_values.biases)
    return KVLayerSegment(
        keys=q_keys,
        values=q_values,
        metadata={
            "class_name": "KVCache",
            "layer_index": layer_index,
            "merge_strategy": "concatenate",
            "n_tokens": n_tokens,
            "bits": bits,
        },
    )


class TestFilesystemRoundTripFullAttention:
    def test_write_read_q8_full_attention(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore,
            NodePayload,
        )
        store = FilesystemCacheDiskStore(cache_dir=str(tmp_path), kv_group_size=64)
        key = (0, (1, 2, 3))
        payload = NodePayload(
            parent_key=None,
            token_ids=(1, 2, 3),
            n_tokens_cumulative=3,
            last_access_ts=12345.0,
            kv_layers=[_make_kv_segment(0, 16, bits=8), _make_kv_segment(1, 16, bits=8)],
            recurrent_layers=[],
        )
        evicted = store.write(key, payload)
        assert evicted == []
        assert store.has(key) is True

        roundtrip = store.read(key)
        assert roundtrip is not None
        assert roundtrip.parent_key is None
        assert roundtrip.token_ids == (1, 2, 3)
        assert roundtrip.n_tokens_cumulative == 3
        assert roundtrip.last_access_ts == 12345.0
        assert len(roundtrip.kv_layers) == 2

        for i, seg in enumerate(roundtrip.kv_layers):
            assert seg.metadata["layer_index"] == i
            assert seg.metadata["class_name"] == "KVCache"
            assert seg.metadata["bits"] == 8
            orig = payload.kv_layers[i]
            assert mx.array_equal(seg.keys.packed, orig.keys.packed)
            assert mx.array_equal(seg.keys.scales, orig.keys.scales)
            assert mx.array_equal(seg.keys.biases, orig.keys.biases)
            assert mx.array_equal(seg.values.packed, orig.values.packed)
            assert mx.array_equal(seg.values.scales, orig.values.scales)
            assert mx.array_equal(seg.values.biases, orig.values.biases)

    def test_read_returns_none_on_miss(self, tmp_path):
        from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore
        store = FilesystemCacheDiskStore(cache_dir=str(tmp_path), kv_group_size=64)
        assert store.read((9, (99,))) is None
        assert store.has((9, (99,))) is False
