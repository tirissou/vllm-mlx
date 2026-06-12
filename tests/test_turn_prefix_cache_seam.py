# SPDX-License-Identifier: Apache-2.0
"""Asserts the trie no longer owns disk concerns."""


def test_no_save_method_on_trie():
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache
    assert not hasattr(TurnPrefixCache, "save")
    assert not hasattr(TurnPrefixCache, "load")
    assert not hasattr(TurnPrefixCache, "_spill_to_ssd")
    assert not hasattr(TurnPrefixCache, "_promote_from_ssd")


def test_ssdref_exported_from_cache_disk_store():
    from vllm_mlx.cache_disk_store import SSDRef
    assert SSDRef is not None


def test_walk_preorder_visits_root_then_descendants():
    from vllm_mlx.turn_prefix_cache import (
        Segment, TurnPrefixCache, TurnPrefixCacheConfig,
    )
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    a = trie.insert(trie.root, Segment(role="user", token_ids=[1]))
    b = trie.insert(a, Segment(role="user", token_ids=[2]))
    c = trie.insert(trie.root, Segment(role="user", token_ids=[3]))
    order = list(trie.walk_all_nodes_preorder())
    # Root NOT included; parents before children.
    assert a in order and b in order and c in order
    assert order.index(a) < order.index(b)


def test_insert_prebuilt_preserves_last_access_and_n_tokens():
    from vllm_mlx.turn_prefix_cache import (
        TurnPrefixCache, TurnPrefixCacheConfig,
    )
    from vllm_mlx.cache_disk_store import SSDRef
    trie = TurnPrefixCache(TurnPrefixCacheConfig())
    node = trie.insert_prebuilt(
        parent_key=None,
        token_ids=(1, 2, 3),
        last_access_ts=42.0,
        n_tokens_cumulative=3,
        kv_data=SSDRef(key=(0, (1, 2, 3))),
        recurrent_data=SSDRef(key=(0, (1, 2, 3))),
    )
    assert node.last_used == 42.0
    assert isinstance(node.kv_data, SSDRef)


def test_context_hash_is_deterministic_across_processes():
    """Smoke test: same inputs → same hash, independent of PYTHONHASHSEED."""
    from vllm_mlx.turn_prefix_cache import _context_hash
    a = _context_hash(0, [1, 2, 3])
    b = _context_hash(0, [1, 2, 3])
    assert a == b
    # Different parent → different hash.
    assert _context_hash(0, [1, 2, 3]) != _context_hash(1, [1, 2, 3])
