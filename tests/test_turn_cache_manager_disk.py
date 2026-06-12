# SPDX-License-Identifier: Apache-2.0
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
