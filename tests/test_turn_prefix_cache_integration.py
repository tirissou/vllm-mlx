"""Integration tests: full segment → Trie → collect_path_data → _assemble round-trip.

Tests the spec's invariant:
  TurnCacheManager._assemble(trie.collect_path_data(node)) ≈ reconstructed live cache objects
(within quantization epsilon).
"""

import math
import mlx.core as mx
import pytest

from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig, Segment
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.cache_types import KVLayerSegment, RecurrentLayerSegment


@pytest.fixture
def trie():
    cfg = TurnPrefixCacheConfig(checkpoint_stride=0, max_memory_gb=4.0)
    return TurnPrefixCache(cfg)


def _make_kvcache_states(n_tokens: int, n_layers: int = 2, head_dim: int = 64):
    """Create a list of KVCache state dicts. head_dim must be divisible by group_size=64."""
    states = []
    for i in range(n_layers):
        keys = mx.ones((1, 1, n_tokens, head_dim), dtype=mx.bfloat16) * (i + 1)
        values = mx.ones((1, 1, n_tokens, head_dim), dtype=mx.bfloat16) * (i + 10)
        states.append(
            {
                "class_name": "KVCache",
                "state": (keys, values),
                "meta_state": (n_tokens,),
            }
        )
    return states


def _make_rotating_states(n_tokens: int, max_size: int = 4, keep: int = 0):
    """Create a single RotatingKVCache state dict. head_dim=1 for simplicity."""
    # Use head_dim=64 for group_size compatibility
    keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
    values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * 2
    offset = n_tokens % max_size if n_tokens <= max_size else max_size
    return [
        {
            "class_name": "RotatingKVCache",
            "state": (keys, values),
            "meta_state": (keep, max_size, offset, n_tokens),
        }
    ]


def test_kvcache_round_trip(trie):
    """KVCache: _segment → insert → match → collect_path_data → _assemble restores shape."""
    from mlx_lm.models.cache import QuantizedKVCache

    live_states = _make_kvcache_states(n_tokens=4, n_layers=2)

    kv_sparse, rec_sparse = TurnCacheManager._segment(live_states)
    kv_layers = [kv for kv in kv_sparse if kv is not None]
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1, 2, 3, 4])
    node = trie.insert(
        trie.root, seg, kv_data=kv_layers, recurrent_data=rec_layers or None
    )

    path, _ = trie.match([seg])
    assert len(path) == 1 and path[0] is node

    kv_out, rec_out = trie.collect_path_data(node)
    assembled = TurnCacheManager._assemble(kv_out, rec_out)

    assert len(assembled) == 2  # 2 KV layers
    for cache in assembled:
        # Each assembled layer should be a QuantizedKVCache
        assert isinstance(cache, QuantizedKVCache)
        assert cache.offset == 4
    trie.release(path)


def test_kvcache_concatenates_across_two_nodes(trie):
    """Two KVCache nodes in a path → collect_path_data concatenates their arrays."""
    states1 = _make_kvcache_states(n_tokens=3, n_layers=1)
    states2 = _make_kvcache_states(n_tokens=5, n_layers=1)

    kv1, _ = TurnCacheManager._segment(states1)
    kv2, _ = TurnCacheManager._segment(states2)

    seg1 = Segment(role="system", token_ids=[1, 2, 3])
    seg2 = Segment(role="conversation", token_ids=[4, 5, 6, 7, 8])
    node1 = trie.insert(trie.root, seg1, kv_data=[kv for kv in kv1 if kv])
    node2 = trie.insert(node1, seg2, kv_data=[kv for kv in kv2 if kv])

    kv_out, _ = trie.collect_path_data(node2)
    assembled = TurnCacheManager._assemble(kv_out, [])

    # After concat: offset should be 3 + 5 = 8
    assert len(assembled) == 1
    assert assembled[0].offset == 3 + 5


def test_rotating_kvcache_uses_last_node(trie):
    """RotatingKVCache: collect_path_data uses only the deepest node's buffer."""
    from mlx_lm.models.cache import RotatingKVCache

    states1 = _make_rotating_states(n_tokens=4, max_size=4)
    states2 = _make_rotating_states(n_tokens=4, max_size=4)

    kv1, _ = TurnCacheManager._segment(states1)
    kv2, _ = TurnCacheManager._segment(states2)

    seg1 = Segment(role="system", token_ids=[1, 2, 3, 4])
    seg2 = Segment(role="conversation", token_ids=[5, 6, 7, 8])
    node1 = trie.insert(trie.root, seg1, kv_data=[kv for kv in kv1 if kv])
    node2 = trie.insert(node1, seg2, kv_data=[kv for kv in kv2 if kv])

    kv_out, _ = trie.collect_path_data(node2)
    # merge_strategy='last' → only node2's data
    assert kv_out[0].metadata["layer_index"] == 0
    assert kv_out[0].metadata["merge_strategy"] == "last"

    # Verify full round-trip: _assemble reconstructs a RotatingKVCache
    assembled = TurnCacheManager._assemble(kv_out, [])
    assert len(assembled) == 1
    assert isinstance(assembled[0], RotatingKVCache)
    # RotatingKVCache is dequantized, max_size=4
    assert assembled[0].keys.shape[-2] == 4  # max_size tokens after linearize


def test_recurrent_comes_from_leaf(trie):
    """Recurrent state is taken from the deepest node (not concatenated)."""
    from mlx_lm.models.cache import ArraysCache

    rec_state = (mx.array([[[[1.0]]]], dtype=mx.bfloat16),)
    live_states = [{"class_name": "MambaLayer", "state": rec_state, "meta_state": ()}]

    _, rec_sparse = TurnCacheManager._segment(live_states)
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1])
    node = trie.insert(trie.root, seg, recurrent_data=rec_layers)

    path, has_recurrent = trie.match([seg])
    assert has_recurrent

    _, rec_out = trie.collect_path_data(node)
    assembled = TurnCacheManager._assemble([], rec_out)

    assert len(assembled) == 1
    assert isinstance(assembled[0], ArraysCache)
    trie.release(path)


def test_collect_path_data_layer_ordering(trie):
    """Mixed KV + recurrent layers: _assemble output is ordered by layer_index."""
    from mlx_lm.models.cache import QuantizedKVCache, ArraysCache

    # layer 0 = KVCache, layer 1 = Recurrent
    kv_state = mx.ones((1, 1, 2, 64), dtype=mx.bfloat16)
    rec_raw = (mx.ones((1, 1, 1, 64), dtype=mx.bfloat16),)
    live_states = [
        {"class_name": "KVCache", "state": (kv_state, kv_state), "meta_state": (2,)},
        {"class_name": "Mamba", "state": rec_raw, "meta_state": ()},
    ]

    kv_sparse, rec_sparse = TurnCacheManager._segment(live_states)
    kv_layers = [kv for kv in kv_sparse if kv is not None]
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1, 2])
    node = trie.insert(trie.root, seg, kv_data=kv_layers, recurrent_data=rec_layers)

    kv_out, rec_out = trie.collect_path_data(node)
    assembled = TurnCacheManager._assemble(kv_out, rec_out)

    assert len(assembled) == 2
    # layer 0 = KVCache (QuantizedKVCache), layer 1 = recurrent (ArraysCache)
    assert isinstance(assembled[0], QuantizedKVCache)
    assert isinstance(assembled[1], ArraysCache)
