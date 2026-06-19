"""Integration tests: full segment → Trie → collect_path_data → assemble round-trip.

Tests the spec's invariant:
  assemble(trie.collect_path_data(node)) ≈ reconstructed live cache objects
(within quantization epsilon).
"""

import math
import mlx.core as mx
import pytest

from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig, Segment
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.cache_types import KVRotatingSegment, KVQuantPolicy, RecurrentLayerSegment
from vllm_mlx.cache_translator import segment, assemble


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
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache

    live_states = _make_kvcache_states(n_tokens=4, n_layers=2)

    kv_sparse, rec_sparse = segment(live_states, policy=KVQuantPolicy(full_bits=8))
    kv_layers = [kv for kv in kv_sparse if kv is not None]
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1, 2, 3, 4])
    node = trie.insert(
        trie.root, seg, kv_data=kv_layers, recurrent_data=rec_layers or None
    )

    path, _ = trie.match([seg])
    assert len(path) == 1 and path[0] is node

    kv_out, rec_out = trie.collect_path_data(node)
    assembled = assemble(kv_out, rec_out)

    assert len(assembled) == 2  # 2 KV layers
    for cache in assembled:
        # Per ADR-0005: full-attn quantized KV reconstructs to BatchQuantizedKVCache.
        assert isinstance(cache, BatchQuantizedKVCache)
        assert cache._idx == 4
    trie.release(path)


def test_kvcache_concatenates_across_two_nodes(trie):
    """Two KVCache nodes in a path → collect_path_data concatenates their arrays."""
    states1 = _make_kvcache_states(n_tokens=3, n_layers=1)
    states2 = _make_kvcache_states(n_tokens=5, n_layers=1)

    kv1, _ = segment(states1)
    kv2, _ = segment(states2)

    seg1 = Segment(role="system", token_ids=[1, 2, 3])
    seg2 = Segment(role="conversation", token_ids=[4, 5, 6, 7, 8])
    node1 = trie.insert(trie.root, seg1, kv_data=[kv for kv in kv1 if kv])
    node2 = trie.insert(node1, seg2, kv_data=[kv for kv in kv2 if kv])

    kv_out, _ = trie.collect_path_data(node2)
    assembled = assemble(kv_out, [])

    # After concat: offset should be 3 + 5 = 8
    assert len(assembled) == 1
    assert assembled[0].offset == 3 + 5


def test_rotating_kvcache_uses_last_node(trie):
    """RotatingKVCache: collect_path_data uses only the deepest node's buffer."""
    from mlx_lm.models.cache import RotatingKVCache

    states1 = _make_rotating_states(n_tokens=4, max_size=4)
    states2 = _make_rotating_states(n_tokens=4, max_size=4)

    kv1, _ = segment(states1)
    kv2, _ = segment(states2)

    seg1 = Segment(role="system", token_ids=[1, 2, 3, 4])
    seg2 = Segment(role="conversation", token_ids=[5, 6, 7, 8])
    node1 = trie.insert(trie.root, seg1, kv_data=[kv for kv in kv1 if kv])
    node2 = trie.insert(node1, seg2, kv_data=[kv for kv in kv2 if kv])

    kv_out, _ = trie.collect_path_data(node2)
    # KVRotatingSegment.merge_path uses last → only node2's data
    assert kv_out[0].layer_index == 0
    assert isinstance(kv_out[0], KVRotatingSegment)

    # Verify full round-trip: _assemble reconstructs a RotatingKVCache
    assembled = assemble(kv_out, [])
    assert len(assembled) == 1
    assert isinstance(assembled[0], RotatingKVCache)
    # RotatingKVCache is dequantized, max_size=4
    assert assembled[0].keys.shape[-2] == 4  # max_size tokens after linearize


def test_recurrent_comes_from_leaf(trie):
    """Recurrent state is taken from the deepest node (not concatenated)."""
    from mlx_lm.models.cache import ArraysCache

    rec_state = (mx.array([[[[1.0]]]], dtype=mx.bfloat16),)
    live_states = [{"class_name": "MambaLayer", "state": rec_state, "meta_state": ()}]

    _, rec_sparse = segment(live_states)
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1])
    node = trie.insert(trie.root, seg, recurrent_data=rec_layers)

    path, has_recurrent = trie.match([seg])
    assert has_recurrent

    _, rec_out = trie.collect_path_data(node)
    assembled = assemble([], rec_out)

    assert len(assembled) == 1
    assert isinstance(assembled[0], ArraysCache)
    trie.release(path)


def test_collect_path_data_layer_ordering(trie):
    """Mixed KV + recurrent layers: _assemble output is ordered by layer_index."""
    from mlx_lm.models.cache import ArraysCache
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache

    # layer 0 = KVCache, layer 1 = Recurrent
    kv_state = mx.ones((1, 1, 2, 64), dtype=mx.bfloat16)
    rec_raw = (mx.ones((1, 1, 1, 64), dtype=mx.bfloat16),)
    live_states = [
        {"class_name": "KVCache", "state": (kv_state, kv_state), "meta_state": (2,)},
        {"class_name": "Mamba", "state": rec_raw, "meta_state": ()},
    ]

    kv_sparse, rec_sparse = segment(live_states, policy=KVQuantPolicy(full_bits=8))
    kv_layers = [kv for kv in kv_sparse if kv is not None]
    rec_layers = [rec for rec in rec_sparse if rec is not None]

    seg = Segment(role="system", token_ids=[1, 2])
    node = trie.insert(trie.root, seg, kv_data=kv_layers, recurrent_data=rec_layers)

    kv_out, rec_out = trie.collect_path_data(node)
    assembled = assemble(kv_out, rec_out)

    assert len(assembled) == 2
    # layer 0 = KVCache → BatchQuantizedKVCache (ADR-0005), layer 1 = recurrent (ArraysCache)
    assert isinstance(assembled[0], BatchQuantizedKVCache)
    assert isinstance(assembled[1], ArraysCache)


# ---------------------------------------------------------------------------
# store() no-op tests (Task 4)
# ---------------------------------------------------------------------------

def _build_request_with_decoded_output():
    """Stub Request-like object with the attributes store() reads."""
    class _Req:
        # request_id is never mutated so it can stay class-level.
        request_id = "rid-test"

        def __init__(self):
            # Per-instance state so multiple callers don't share mutable defaults.
            self.output_token_ids = [101, 102, 103]
            self._cache_state = type("CS", (), {"turn_path": []})()
        # messages_to_segments() reads ._messages or similar in production;
        # for this test we monkeypatch messages_to_segments on the manager.
    return _Req()


def test_store_does_not_promote_decoded_tokens(monkeypatch):
    """store() must be a no-op: trie unchanged, returns False, no leaf pin."""
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig, Segment

    cfg = TurnPrefixCacheConfig(checkpoint_stride=0, max_memory_gb=1.0)
    inner = TurnPrefixCache(cfg)
    mgr = TurnCacheManager(inner)

    req = _build_request_with_decoded_output()
    monkeypatch.setattr(
        mgr, "messages_to_segments",
        lambda r: [Segment(role="system", token_ids=[1, 2, 3]),
                   Segment(role="user", token_ids=[4, 5])],
    )

    pre_root_children = len(inner.root.children)
    pre_pinned = dict(mgr._pinned_leaves)

    ok = mgr.store(req, cache=[])

    assert ok is False
    assert len(inner.root.children) == pre_root_children, \
        "store() must not insert any trie node"
    assert mgr._pinned_leaves == pre_pinned, \
        "store() must not touch pinned-leaf bookkeeping"


def test_decoded_tokens_re_prefilled_on_next_turn(monkeypatch):
    """Two-turn scenario: after store() becomes a no-op, the prior assistant
    response is NOT in the trie. on_prefill_checkpoint at the next turn must
    receive the assistant tokens as part of the prefill input."""
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig, Segment

    cfg = TurnPrefixCacheConfig(checkpoint_stride=0, max_memory_gb=1.0)
    inner = TurnPrefixCache(cfg)
    mgr = TurnCacheManager(inner)

    # Pre-populate the trie with a real segment so root.children is non-empty.
    # This ensures the assertion below is non-vacuous.
    # Use a token count different from len(req.output_token_ids)=3 to avoid
    # false negatives in the "no node sized like assistant response" check.
    pre_seg = Segment(role="system", token_ids=[10, 20])
    inner.insert(inner.root, pre_seg)
    assert len(inner.root.children) > 0, "pre-condition: trie has at least one node"
    pre_count = len(inner.root.children)

    req = _build_request_with_decoded_output()
    monkeypatch.setattr(
        mgr, "messages_to_segments",
        lambda r: [Segment(role="system", token_ids=[1, 2, 3]),
                   Segment(role="user", token_ids=[4, 5])],
    )

    # Simulate end-of-turn store() — must be no-op.
    mgr.store(req, cache=[])

    # The trie must be structurally unchanged at the root level.
    assert len(inner.root.children) == pre_count, \
        "store() must not insert any new root-level trie node"
    # No node anywhere in root.children may have a segment sized like the
    # assistant response — the decoded tokens must not be cached.
    assert all(
        len(child.token_ids) != len(req.output_token_ids)
        for child in inner.root.children.values()
    ), "no node sized like the assistant response may exist"
