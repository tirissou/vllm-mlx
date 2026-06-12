# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx

from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig


def test_manager_accepts_disk_store(tmp_path):
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr = TurnCacheManager(trie, disk_store=store)
    assert mgr._disk_store is store
    # Spill handler must be registered on the trie.
    assert trie._spill_handler is not None
    assert trie._promote_handler is not None


def test_manager_without_disk_store_does_not_register_handler():
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    mgr = TurnCacheManager(trie)
    assert mgr._disk_store is None
    assert trie._spill_handler is None


def _make_kv_seg(layer_index, n_tokens, bits=8):
    from vllm_mlx.cache_types import KVLayerSegment
    from vllm_mlx.kv_cache import QuantizedArray
    k = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    v = mx.random.normal((1, 4, n_tokens, 64), dtype=mx.bfloat16)
    qk = QuantizedArray(*mx.quantize(k, group_size=64, bits=bits))
    qv = QuantizedArray(*mx.quantize(v, group_size=64, bits=bits))
    mx.eval(qk.packed, qk.scales, qk.biases, qv.packed, qv.scales, qv.biases)
    return KVLayerSegment(
        keys=qk, values=qv,
        metadata={"class_name": "KVCache", "layer_index": layer_index,
                  "merge_strategy": "concatenate", "n_tokens": n_tokens, "bits": bits},
    )


def test_on_spill_writes_and_replaces_with_ssd_ref(tmp_path):
    from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore, SSDRef
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr = TurnCacheManager(trie, disk_store=store)

    node = trie.insert(trie.root, Segment(role="user", token_ids=[1, 2]),
                       kv_data=[_make_kv_seg(0, 8)], recurrent_data=None)

    kept = mgr._on_spill(node)
    assert kept is True
    assert isinstance(node.kv_data, SSDRef)
    assert store.has(node.kv_data.key)


def test_on_promote_restores_segments(tmp_path):
    from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr = TurnCacheManager(trie, disk_store=store)

    node = trie.insert(trie.root, Segment(role="user", token_ids=[1, 2]),
                       kv_data=[_make_kv_seg(0, 8)], recurrent_data=None)
    original = node.kv_data
    mgr._on_spill(node)

    result = mgr._on_promote(node.kv_data)
    assert result is not None
    kv, rec = result
    assert len(kv) == 1
    assert mx.array_equal(kv[0].keys.packed, original[0].keys.packed)


def test_save_then_load_round_trip(tmp_path):
    from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore, SSDRef
    from vllm_mlx.cache_types import KVQuantPolicy
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    policy = KVQuantPolicy(full_bits=8, sliding_bits=None)
    trie_a = TurnPrefixCache(TurnPrefixCacheConfig())
    store_a = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr_a = TurnCacheManager(
        trie_a, policy=policy, kv_group_size=64, disk_store=store_a,
    )

    a = trie_a.insert(trie_a.root, Segment(role="user", token_ids=[1]),
                      kv_data=[_make_kv_seg(0, 8, bits=8)])
    b = trie_a.insert(a, Segment(role="user", token_ids=[2]),
                      kv_data=[_make_kv_seg(0, 8, bits=8)])

    written = mgr_a.save()
    assert written == 2

    # Fresh manager pointing at the same disk dir.
    trie_b = TurnPrefixCache(TurnPrefixCacheConfig())
    store_b = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr_b = TurnCacheManager(
        trie_b, policy=policy, kv_group_size=64, disk_store=store_b,
    )
    restored = mgr_b.load()
    assert restored == 2

    nodes = trie_b.walk_all_nodes_preorder()
    assert all(isinstance(n.kv_data, SSDRef) for n in nodes)


def test_load_rejects_policy_mismatch(tmp_path):
    from vllm_mlx.cache_disk_store import (
        CachePolicyMismatchError, FilesystemCacheDiskStore,
    )
    from vllm_mlx.cache_types import KVQuantPolicy
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    # Save with q8.
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr = TurnCacheManager(trie, policy=KVQuantPolicy(full_bits=8),
                           kv_group_size=64, disk_store=store)
    trie.insert(trie.root, Segment(role="user", token_ids=[1]),
                kv_data=[_make_kv_seg(0, 8, bits=8)])
    mgr.save()

    # Load with q4 — must raise.
    import pytest
    trie2 = TurnPrefixCache(TurnPrefixCacheConfig())
    store2 = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr2 = TurnCacheManager(trie2, policy=KVQuantPolicy(full_bits=4),
                            kv_group_size=64, disk_store=store2)
    with pytest.raises(CachePolicyMismatchError):
        mgr2.load()


def test_fetch_promotes_spilled_node_on_access(tmp_path):
    from vllm_mlx.cache_disk_store import FilesystemCacheDiskStore, SSDRef
    from vllm_mlx.cache_types import KVQuantPolicy
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    store = FilesystemCacheDiskStore(str(tmp_path), kv_group_size=64)
    mgr = TurnCacheManager(trie, policy=KVQuantPolicy(full_bits=8),
                           kv_group_size=64, disk_store=store)

    node = trie.insert(trie.root, Segment(role="user", token_ids=[1, 2]),
                       kv_data=[_make_kv_seg(0, 8, bits=8)])
    original_packed = node.kv_data[0].keys.packed
    mgr._on_spill(node)
    assert isinstance(node.kv_data, SSDRef)

    # collect_path_data should auto-promote.
    kv, rec = trie.collect_path_data(node)
    assert len(kv) == 1
    assert mx.array_equal(kv[0].keys.packed, original_packed)
    # Node has its real KV back.
    assert not isinstance(node.kv_data, SSDRef)
