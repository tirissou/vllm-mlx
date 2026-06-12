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
