# SPDX-License-Identifier: Apache-2.0
"""End-to-end save/load round-trip integration test.

Build a trie with several real KV layers, save, construct a fresh manager
pointing at the same dir, load, then fetch — assert all data is restored.
"""

import mlx.core as mx
import pytest

from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore, SSDRef
from vllm_mlx.cache_types import KVLayerSegment, KVQuantPolicy
from vllm_mlx.kv_cache import QuantizedArray
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.turn_prefix_cache import (
    Segment,
    TurnPrefixCache,
    TurnPrefixCacheConfig,
)


def _quantize_kv(n_tokens, bits=8, layer_index=0):
    k = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    v = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    qk = QuantizedArray(*mx.quantize(k, group_size=64, bits=bits))
    qv = QuantizedArray(*mx.quantize(v, group_size=64, bits=bits))
    mx.eval(qk.packed, qk.scales, qk.biases, qv.packed, qv.scales, qv.biases)
    return KVLayerSegment(
        keys=qk,
        values=qv,
        metadata={
            "class_name": "KVCache",
            "layer_index": layer_index,
            "merge_strategy": "concatenate",
            "n_tokens": n_tokens,
            "bits": bits,
        },
    )


def test_save_load_fetch_round_trip(tmp_path):
    policy = KVQuantPolicy(full_bits=8)
    # Pass 1: build + save.
    trie_a = TurnPrefixCache(TurnPrefixCacheConfig())
    store_a = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr_a = TurnCacheManager(
        trie_a, policy=policy, kv_group_size=64, disk_store=store_a,
    )

    n_a = trie_a.insert(
        trie_a.root, Segment(role="user", token_ids=[1, 2]),
        kv_data=[_quantize_kv(2)],
    )
    n_b = trie_a.insert(
        n_a, Segment(role="user", token_ids=[3, 4]),
        kv_data=[_quantize_kv(2)],
    )
    n_c = trie_a.insert(
        n_b, Segment(role="user", token_ids=[5, 6]),
        kv_data=[_quantize_kv(2)],
    )
    _ = n_c.kv_data[0].keys.packed

    written = mgr_a.save()
    assert written == 3
    mgr_a.close()

    # Pass 2: cold load + promote-via-fetch path.
    trie_b = TurnPrefixCache(TurnPrefixCacheConfig())
    store_b = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr_b = TurnCacheManager(
        trie_b, policy=policy, kv_group_size=64, disk_store=store_b,
    )
    restored = mgr_b.load()
    assert restored == 3

    nodes_b = trie_b.walk_all_nodes_preorder()
    assert len(nodes_b) == 3
    assert all(isinstance(n.kv_data, SSDRef) for n in nodes_b)

    # Locate the deepest node (n_c-equivalent) by tokens.
    deepest = next(n for n in nodes_b if n.token_ids == [5, 6])
    kv, _ = trie_b.collect_path_data(deepest)
    assert len(kv) >= 1
    # The reconstructed concat for layer 0 has all three slices, packed bit-exact
    # at the tail.
    assert kv[0].metadata["bits"] == 8
    mgr_b.close()


def test_quant_policy_mismatch_fails_load(tmp_path):
    from vllm_mlx.cache_disk_store import CachePolicyMismatchError
    p1 = KVQuantPolicy(full_bits=8)
    p2 = KVQuantPolicy(full_bits=4)
    t1 = TurnPrefixCache(TurnPrefixCacheConfig())
    s1 = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    m1 = TurnCacheManager(t1, policy=p1, kv_group_size=64, disk_store=s1)
    t1.insert(
        t1.root, Segment(role="user", token_ids=[1]),
        kv_data=[_quantize_kv(2)],
    )
    m1.save()
    m1.close()

    t2 = TurnPrefixCache(TurnPrefixCacheConfig())
    s2 = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    m2 = TurnCacheManager(t2, policy=p2, kv_group_size=64, disk_store=s2)
    with pytest.raises(CachePolicyMismatchError):
        m2.load()
    m2.close()
