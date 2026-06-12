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
