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


class TestDiskLRU:
    def test_leaf_first_eviction(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore,
            NodePayload,
        )

        # Cap small enough that a third write forces eviction.
        # Each entry is ~9 KB on disk; estimated overhead ~13 KB per write.
        # 28_000 comfortably holds two entries but triggers eviction on the third.
        store = FilesystemCacheDiskStore(
            cache_dir=str(tmp_path), kv_group_size=64, max_bytes=28_000,
        )

        def write_node(parent_key, tokens, ts):
            key = (hash(parent_key), tokens)
            payload = NodePayload(
                parent_key=parent_key, token_ids=tokens,
                n_tokens_cumulative=len(tokens), last_access_ts=ts,
                kv_layers=[_make_kv_segment(0, 16, bits=8)],
                recurrent_layers=[],
            )
            return key, store.write(key, payload)

        ka, _ = write_node(None, (1,), ts=1.0)
        kb, _ = write_node(None, (2,), ts=2.0)
        kc, evicted = write_node(None, (3,), ts=3.0)
        # ka is the oldest leaf — it should be evicted.
        assert ka in evicted

    def test_parent_not_evicted_before_children(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore,
            NodePayload,
        )
        # Each entry is ~9 KB on disk; estimated overhead ~13 KB per write.
        # 28_000 comfortably holds two entries but triggers eviction on the third.
        store = FilesystemCacheDiskStore(
            cache_dir=str(tmp_path), kv_group_size=64, max_bytes=28_000,
        )

        # Parent first, then a child of that parent, then a new leaf to force eviction.
        parent_key = (0, (10,))
        store.write(parent_key, NodePayload(
            parent_key=None, token_ids=(10,), n_tokens_cumulative=1,
            last_access_ts=1.0,
            kv_layers=[_make_kv_segment(0, 16)], recurrent_layers=[],
        ))
        child_key = (hash(parent_key), (11,))
        store.write(child_key, NodePayload(
            parent_key=parent_key, token_ids=(11,), n_tokens_cumulative=2,
            last_access_ts=2.0,
            kv_layers=[_make_kv_segment(0, 16)], recurrent_layers=[],
        ))

        unrelated_key = (0, (99,))
        evicted = store.write(unrelated_key, NodePayload(
            parent_key=None, token_ids=(99,), n_tokens_cumulative=1,
            last_access_ts=3.0,
            kv_layers=[_make_kv_segment(0, 16)], recurrent_layers=[],
        ))
        # Parent has a child on disk → not eligible; child is the only leaf →
        # child must be evicted before parent.
        assert child_key in evicted
        assert parent_key not in evicted


def _make_rotating_segment(layer_index: int, n_tokens: int = 16):
    """RotatingKVCache lives in bf16 (sliding_bits=None default)."""
    from vllm_mlx.cache_types import KVLayerSegment
    keys = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    values = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    mx.eval(keys, values)
    return KVLayerSegment(
        keys=keys, values=values,
        metadata={
            "class_name": "RotatingKVCache",
            "layer_index": layer_index,
            "merge_strategy": "last",
            "n_tokens": n_tokens,
            "max_size": 256,
            "keep": 0,
            "offset": n_tokens,
            "_idx": n_tokens,
            "bits": None,
        },
    )


def _make_recurrent_segment(layer_index: int):
    from vllm_mlx.cache_types import RecurrentLayerSegment
    arrays = [mx.zeros((1, 8, 16), dtype=mx.bfloat16)]
    mx.eval(*arrays)
    from mlx_lm.models.cache import ArraysCache
    return RecurrentLayerSegment(
        arrays=arrays,
        metadata={
            "class_name": "ArraysCache",
            "layer_index": layer_index,
            "class_ref": ArraysCache,
        },
        scales=None,
    )


class TestFilesystemRoundTripSlidingAndRecurrent:
    def test_rotating_bf16_round_trip(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore, NodePayload,
        )
        store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
        key = (0, (1,))
        payload = NodePayload(
            parent_key=None, token_ids=(1,), n_tokens_cumulative=1,
            last_access_ts=0.0,
            kv_layers=[_make_rotating_segment(0)],
            recurrent_layers=[],
        )
        store.write(key, payload)
        rt = store.read(key)
        assert rt is not None
        assert rt.kv_layers[0].metadata["max_size"] == 256
        assert mx.array_equal(rt.kv_layers[0].keys, payload.kv_layers[0].keys)
        assert mx.array_equal(rt.kv_layers[0].values, payload.kv_layers[0].values)

    def test_recurrent_round_trip(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore, NodePayload,
        )
        store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
        key = (0, (2,))
        payload = NodePayload(
            parent_key=None, token_ids=(2,), n_tokens_cumulative=1,
            last_access_ts=0.0,
            kv_layers=[],
            recurrent_layers=[_make_recurrent_segment(0)],
        )
        store.write(key, payload)
        rt = store.read(key)
        assert rt is not None
        assert len(rt.recurrent_layers) == 1
        assert rt.recurrent_layers[0].metadata["class_ref"] is not None
        assert mx.array_equal(
            rt.recurrent_layers[0].arrays[0], payload.recurrent_layers[0].arrays[0]
        )

        header = store.read_header(key)
        assert header is not None
        assert header.recurrent_class_paths[0].endswith("ArraysCache")


class TestCrashAndCorruption:
    def test_corrupt_safetensors_returns_none_and_deletes(self, tmp_path):
        from vllm_mlx.cache_disk_store import (
            FilesystemCacheDiskStore, NodePayload,
        )
        store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
        key = (0, (1,))
        store.write(key, NodePayload(
            parent_key=None, token_ids=(1,), n_tokens_cumulative=1,
            last_access_ts=0.0,
            kv_layers=[_make_kv_segment(0, 16)], recurrent_layers=[],
        ))
        # Corrupt the safetensors file in place.
        st_file = next((tmp_path).glob("*.safetensors"))
        st_file.write_bytes(b"not a real safetensors file")
        assert store.read(key) is None
        assert store.has(key) is False  # delete cascaded.

    def test_orphan_tmp_files_cleaned_on_open(self, tmp_path):
        # Simulate a crash mid-write: leave a .safetensors.tmp behind.
        (tmp_path / "_index.json").write_text(
            '{"_DISK_FORMAT_VERSION": 1, "entries": {}, "total_bytes": 0}'
        )
        (tmp_path / "abc123.safetensors.tmp").write_bytes(b"truncated")

        from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore
        store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
        # Constructor should clean orphan tmp files.
        assert not (tmp_path / "abc123.safetensors.tmp").exists()
