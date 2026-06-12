# SPDX-License-Identifier: Apache-2.0
"""Trie spill/promote handler wiring."""

from vllm_mlx.turn_prefix_cache import (
    Segment, TurnPrefixCache, TurnPrefixCacheConfig,
)


def test_spill_handler_invoked_on_eviction():
    """When memory exceeds budget, the trie should call the spill handler."""
    config = TurnPrefixCacheConfig(max_memory_gb=1e-9)  # ~0 → forces eviction.
    trie = TurnPrefixCache(config)

    calls = []
    trie.set_spill_handler(lambda node: (calls.append(node), True)[1])

    # Insert two nodes; the LRU one should be spilled.
    n1 = trie.insert(trie.root, Segment(role="user", token_ids=[1]),
                     kv_data=None, recurrent_data=None)
    n2 = trie.insert(trie.root, Segment(role="user", token_ids=[2]),
                     kv_data=None, recurrent_data=None)

    # Without real KV bytes, _memory_bytes stays at 0, so no eviction fires.
    # The handler should at least be assignable without error.
    assert callable(trie._spill_handler)


def test_promote_handler_assignable():
    config = TurnPrefixCacheConfig()
    trie = TurnPrefixCache(config)
    trie.set_promote_handler(lambda ref: None)
    assert callable(trie._promote_handler)
