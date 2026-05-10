import pytest
import mlx.core as mx
import json
import tempfile
from pathlib import Path
from vllm_mlx.turn_prefix_cache import (
    Segment, TurnNode, SSDRef, TurnPrefixCacheConfig, TurnPrefixCache, _context_hash, _node_data_bytes,
    _quantize_kv, _dequantize_kv,
)


def seg(token_ids, role="user"):
    return Segment(role=role, token_ids=token_ids)


def test_context_hash_deterministic():
    assert _context_hash(0, [1, 2, 3]) == _context_hash(0, [1, 2, 3])


def test_context_hash_different_parent():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(1, [1, 2, 3])


def test_context_hash_different_tokens():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(0, [1, 2, 4])


def test_turn_node_is_leaf_when_no_children():
    node = TurnNode(token_ids=[1], context_hash=1, kv_arrays=[], kv_scales=[],
                    recurrent_state=None, tokens_since_checkpoint=0, parent=None)
    assert node.is_leaf


def test_turn_node_not_leaf_when_has_children():
    parent = TurnNode(token_ids=[], context_hash=0, kv_arrays=None, kv_scales=None,
                      recurrent_state=None, tokens_since_checkpoint=0, parent=None)
    child = TurnNode(token_ids=[1], context_hash=1, kv_arrays=[], kv_scales=[],
                     recurrent_state=None, tokens_since_checkpoint=1, parent=parent)
    parent.children[1] = child
    assert not parent.is_leaf


def test_ssdref_is_sentinel():
    ref = SSDRef(file_path="/tmp/foo.safetensors", size_bytes=1024)
    assert ref.file_path == "/tmp/foo.safetensors"
    assert ref.size_bytes == 1024


def test_config_defaults():
    cfg = TurnPrefixCacheConfig()
    assert cfg.checkpoint_stride == 512
    assert cfg.max_memory_gb == 8.0
    assert cfg.kv_dtype == "int8"
    assert cfg.persist_dir is None
    assert cfg.ssd_max_gb == 0.0


from vllm_mlx.turn_prefix_cache import TurnPrefixCache


def make_cache(stride=512, max_gb=8.0, kv_dtype="bf16"):
    return TurnPrefixCache(TurnPrefixCacheConfig(
        checkpoint_stride=stride, max_memory_gb=max_gb, kv_dtype=kv_dtype
    ))


def test_insert_creates_child_of_root():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), kv_arrays=[], kv_scales=[], recurrent_state=None)
    assert node in cache.root.children.values()


def test_insert_sets_token_ids():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.token_ids == [1, 2, 3]


def test_insert_node_is_leaf():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.is_leaf


def test_insert_sets_parent():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.parent is cache.root


def test_insert_idempotent_same_segment():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert n1 is n2  # same node returned


def test_insert_chained():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]), [], [], None)
    n2 = cache.insert(n1, seg([2]), [], [], None)
    assert n2.parent is n1
    assert n2 in n1.children.values()


def test_insert_context_hash_differs_at_different_depths():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(n1, seg([1, 2, 3]), [], [], None)  # same tokens, different parent
    assert n1.context_hash != n2.context_hash


def test_leaf_gets_recurrent_state():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    node = cache.insert(cache.root, seg(list(range(50))), [], [], state)
    assert node.recurrent_state is not None


def test_temp_recurrent_pruned_on_non_stride_inner():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    n1 = cache.insert(cache.root, seg(list(range(50))), [], [], state)
    # n1 is leaf with temp recurrent (50 < 100)
    assert n1.recurrent_state is not None
    # Add child: n1 becomes inner node, tokens_since=50 < 100 → prune
    cache.insert(n1, seg([99]), [], [], state)
    assert n1.recurrent_state is None


def test_permanent_checkpoint_at_stride():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    # 120 tokens >= stride → permanent
    n1 = cache.insert(cache.root, seg(list(range(120))), [], [], state)
    assert n1.is_permanent_checkpoint
    # Add child: should NOT prune recurrent
    cache.insert(n1, seg([999]), [], [], state)
    assert n1.recurrent_state is not None


def test_system_prompt_always_permanent():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    # Only 50 tokens but is_system_prompt=True
    n = cache.insert(cache.root, seg(list(range(50)), role="system"), [], [], state,
                     is_system_prompt=True)
    assert n.is_permanent_checkpoint
    cache.insert(n, seg([99]), [], [], state)
    assert n.recurrent_state is not None


def test_tokens_since_resets_after_permanent():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    # 120 tokens → permanent checkpoint
    n1 = cache.insert(cache.root, seg(list(range(120))), [], [], state)
    assert n1.is_permanent_checkpoint
    # 10 more tokens; tokens_since should count from n1 (permanent), not root
    n2 = cache.insert(n1, seg(list(range(10))), [], [], state)
    assert n2.tokens_since_checkpoint == 10


def test_stride_zero_makes_every_node_permanent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    n1 = cache.insert(cache.root, seg([1, 2]), [], [], state)
    assert n1.is_permanent_checkpoint
    n2 = cache.insert(n1, seg([3]), [], [], state)
    assert n2.is_permanent_checkpoint
    # Neither should have recurrent pruned
    assert n1.recurrent_state is not None
    assert n2.recurrent_state is not None


def test_match_full():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(n1, seg([4, 5]), [], [], None)
    path, _ = cache.match([seg([1, 2, 3]), seg([4, 5])])
    assert path == [n1, n2]


def test_match_partial():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    cache.insert(n1, seg([4, 5]), [], [], None)
    path, _ = cache.match([seg([1, 2, 3]), seg([99])])  # second seg not in trie
    assert path == [n1]


def test_match_empty():
    cache = make_cache()
    path, has_recurrent = cache.match([seg([99])])
    assert path == []
    assert not has_recurrent


def test_match_updates_last_used():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    node.last_used = 0.0
    cache.match([seg([1])])
    assert node.last_used > 0.0


def test_match_reports_has_recurrent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    node = cache.insert(cache.root, seg([1]), [], [], state)
    _, has_recurrent = cache.match([seg([1])])
    assert has_recurrent


def test_match_increments_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    assert node.ref_count == 0
    cache.match([seg([1])])
    assert node.ref_count == 1


def test_release_decrements_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    path, _ = cache.match([seg([1])])
    assert node.ref_count == 1
    cache.release(path)
    assert node.ref_count == 0


def test_release_all_nodes_in_path():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]), [], [], None)
    n2 = cache.insert(n1, seg([2]), [], [], None)
    path, _ = cache.match([seg([1]), seg([2])])
    assert n1.ref_count == 1
    assert n2.ref_count == 1
    cache.release(path)
    assert n1.ref_count == 0
    assert n2.ref_count == 0


def test_find_checkpoint_ancestor_returns_self_if_has_recurrent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    n = cache.insert(cache.root, seg([1]), [], [], state)
    path = [n]
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is n


def test_find_checkpoint_ancestor_walks_up():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    # sys node: permanent checkpoint (is_system_prompt=True)
    n_sys = cache.insert(cache.root, seg(list(range(50)), role="system"),
                          [], [], state, is_system_prompt=True)
    # user node: not a checkpoint (stride not met, temp gets pruned after child added)
    n_user = cache.insert(n_sys, seg([100, 101]), [], [], state)
    # asst node: not a checkpoint, prunes n_user's temp recurrent
    n_asst = cache.insert(n_user, seg([200]), [], [], state)
    # n_user's recurrent was pruned; n_sys still has permanent recurrent
    assert n_user.recurrent_state is None
    path = [n_sys, n_user, n_asst]
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is n_sys


def test_find_checkpoint_ancestor_returns_none_when_no_checkpoint():
    cache = make_cache(stride=10000)
    # No state stored, stride too high → no permanent checkpoints (except root which isn't in path)
    n = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is None


def test_find_checkpoint_ancestor_returns_leaf_if_leaf_has_recurrent():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    n = cache.insert(cache.root, seg([1]), [], [], state)
    # n is a leaf → has temp recurrent
    assert n.recurrent_state is not None
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is n


def test_find_checkpoint_ancestor_skips_ssdref_nodes():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    n1 = cache.insert(cache.root, seg([1]), [], [], state)
    n2 = cache.insert(n1, seg([2]), [], [], state)
    # Simulate n1's state being spilled to SSD
    n1.recurrent_state = SSDRef(file_path="/tmp/state.bin", size_bytes=1024)
    # Should skip n1 and return n2
    ancestor = cache.find_checkpoint_ancestor([n1, n2])
    assert ancestor is n2


def _make_kv(n_tokens=1):
    """Small real KV arrays for memory-tracked tests."""
    return [mx.zeros((1, 4, n_tokens, 256), dtype=mx.bfloat16)]


def _kv_scales():
    return [1.0]


def test_lru_evicts_oldest_leaf():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2 = cache.insert(cache.root, seg([2]), _make_kv(), _kv_scales(), None)
    n1.last_used = 1.0
    n2.last_used = 2.0
    # Set budget to allow only the newer (n2) node, forcing n1's eviction
    node_size = _node_data_bytes(n1)
    cache.config.max_memory_gb = (node_size + 512) / (1024**3)  # Budget for ~1 node plus buffer
    cache._evict_if_needed()
    assert n1.kv_arrays is None      # evicted (oldest)
    assert n2.kv_arrays is not None  # kept (newer)


def test_pinned_node_not_evicted():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    node = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    node.ref_count = 1
    # Set budget to 0 to force eviction attempt; ref_count should protect it
    cache.config.max_memory_gb = 0.0
    cache._evict_if_needed()
    assert node.kv_arrays is not None  # pinned by ref_count, should not be evicted


def test_eviction_cascade_to_parent():
    cache = make_cache(max_gb=0.0)
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2 = cache.insert(n1, seg([2]), _make_kv(), _kv_scales(), None)
    n1.last_used = 0.5
    n2.last_used = 1.0
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert n2.kv_arrays is None  # leaf evicted first
    assert n1.kv_arrays is None  # cascades since n1 now has no children


def test_cascade_stops_at_sibling():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2a = cache.insert(n1, seg([2]), _make_kv(), _kv_scales(), None)
    n2b = cache.insert(n1, seg([3]), _make_kv(), _kv_scales(), None)
    n2a.last_used = 1.0
    n2b.last_used = 2.0
    n1.last_used = 0.5
    # Set budget to allow n1 + n2b but not n2a; cascade stops because n1 still has n2b
    node_size = _node_data_bytes(n2a)
    cache.config.max_memory_gb = (node_size * 2 + 512) / (1024**3)
    cache._evict_if_needed()
    assert n2a.kv_arrays is None      # evicted (oldest leaf)
    assert n1.kv_arrays is not None   # n1 still has n2b, so cascade stops


def test_evicted_node_removed_from_parent_children():
    cache = make_cache(max_gb=0.0)
    node = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    h = node.context_hash
    cache._evict_if_needed()
    assert h not in cache.root.children


def test_quantize_roundtrip_within_tolerance():
    arr = mx.array([[0.5, -0.3, 0.8, -1.2, 0.0, 1.0, -1.0, 0.25]], dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    diff = mx.abs(restored[0].astype(mx.float32) - arr.astype(mx.float32))
    # Max quantization error <= range/127 ≈ 2/127 ≈ 0.016 for this data
    assert mx.max(diff).item() < 0.02


def test_quantize_preserves_sign():
    arr = mx.array([-1.0, 0.0, 1.0], dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    assert restored[0][0].item() < 0
    assert restored[0][2].item() > 0


def test_quantize_zero_array():
    arr = mx.zeros((4, 4), dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    assert mx.max(mx.abs(restored[0])).item() == 0.0


def test_insert_quantizes_when_kv_dtype_int8():
    cache = make_cache(stride=512, kv_dtype="int8")
    kv = [mx.ones((1, 4, 3, 256), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, None, None)
    assert node.kv_arrays is not None
    assert node.kv_arrays[0].dtype == mx.int8
    assert node.kv_scales is not None


def test_insert_skips_quantization_when_bf16():
    cache = make_cache(stride=512, kv_dtype="bf16")
    kv = [mx.ones((1, 4, 3, 256), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    assert node.kv_arrays[0].dtype == mx.bfloat16


def test_save_and_load_roundtrip(tmp_path):
    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    seg1 = seg(list(range(10)), role="system")
    kv = [mx.ones((1, 4, 10, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg1, kv, None, state, is_system_prompt=True)

    cache.save(str(tmp_path))

    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))
    path, has_recurrent = cache2.match([seg1])
    assert len(path) == 1
    assert has_recurrent


def test_load_version_mismatch(tmp_path):
    meta = {"version": 9999, "model_fingerprint": "test"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cache = make_cache()
    cache.load(str(tmp_path))  # must not raise
    assert len(cache.root.children) == 0  # starts empty


def test_load_missing_kv_file_skips_node(tmp_path):
    cache = make_cache(stride=0)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    cache.insert(cache.root, seg([1, 2, 3]), kv, None, None)
    cache.save(str(tmp_path))

    # Delete the KV file
    for f in tmp_path.glob("kv_*.safetensors"):
        f.unlink()
        break

    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))  # must not raise


def make_ssd_cache(tmp_path, stride=512):
    return TurnPrefixCache(TurnPrefixCacheConfig(
        checkpoint_stride=stride,
        max_memory_gb=8.0,
        kv_dtype="bf16",
        ssd_max_gb=10.0,
        ssd_dir=str(tmp_path),
    ))


def test_spill_replaces_kv_with_ssdref(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_arrays, SSDRef)


def test_spill_trie_still_matchable(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    cache._spill_to_ssd(node)
    path, _ = cache.match([seg([1, 2, 3])])
    assert len(path) == 1


def test_promote_restores_arrays(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_arrays, SSDRef)
    success = cache._promote_from_ssd(node)
    assert success
    assert isinstance(node.kv_arrays, list)


def test_promote_returns_false_on_missing_file(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1]), [], [], None)
    node.kv_arrays = SSDRef(file_path="/nonexistent/file.safetensors", size_bytes=0)
    result = cache._promote_from_ssd(node)
    assert result is False


def test_promote_restores_recurrent_state(tmp_path):
    """Regression test: recurrent state is properly reconstructed from SSD."""
    cache = make_ssd_cache(tmp_path)
    # Create nested recurrent state structure (list of layers)
    state = [mx.ones((2, 3)), mx.ones((4, 5))]
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], state)

    # Verify before spill
    assert isinstance(node.recurrent_state, list)
    assert len(node.recurrent_state) == 2

    # Spill and promote
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_arrays, SSDRef)
    assert isinstance(node.recurrent_state, SSDRef)

    success = cache._promote_from_ssd(node)
    assert success

    # Verify recurrent state is restored as list
    assert isinstance(node.recurrent_state, list)
    assert len(node.recurrent_state) == 2
    assert node.recurrent_state[0].shape == (2, 3)
    assert node.recurrent_state[1].shape == (4, 5)


from unittest.mock import MagicMock


def test_scheduler_integration_fetch_hits_cache():
    """Verify that matching path and recurrent state are set on the request."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    sys_seg = seg(list(range(20)), role="system")
    cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)

    # Simulate what _fetch_turn_cache does
    segments = [sys_seg]
    path, has_recurrent = cache.match(segments)
    assert len(path) == 1
    assert has_recurrent

    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is path[0]
    cache.release(path)


def test_scheduler_integration_store_extends_trie():
    """Verify that inserting new segments after match extends the trie."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    sys_seg = seg(list(range(10)), role="system")
    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)

    # Match found sys_node; now store new user segment
    user_seg = seg([100, 101, 102])
    n_user = cache.insert(n_sys, user_seg, [], [], state)
    assert n_user in n_sys.children.values()


def test_concurrent_ref_counts():
    """Two requests sharing a prefix: ref_count=2 while both active."""
    cache = make_cache()
    n = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)

    path1, _ = cache.match([seg([1, 2, 3])])
    path2, _ = cache.match([seg([1, 2, 3])])
    assert n.ref_count == 2
    assert not n.is_evictable

    cache.release(path1)
    assert n.ref_count == 1
    assert not n.is_evictable

    cache.release(path2)
    assert n.ref_count == 0
    assert n.is_evictable


def test_scheduler_stores_and_serves_cache():
    """Verify that segments are stored and served correctly in a realistic flow."""
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    state = mx.zeros((1,))

    # Session 1: Build the cache
    sys_seg = seg(list(range(10)), role="system")
    user_seg = seg([100, 101])

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_user = cache.insert(n_sys, user_seg, [], [], state)

    # Verify nodes were created
    assert n_sys is not None
    assert n_user is not None
    assert n_user.parent is n_sys

    # Session 2: Same prefix should hit both segments
    path, has_recurrent = cache.match([sys_seg, user_seg])
    assert len(path) == 2
    assert path[0] is n_sys
    assert path[1] is n_user
    assert has_recurrent  # leaf has temp recurrent

    # Release and verify can be used again
    cache.release(path)
    assert path[0].ref_count == 0
    assert path[1].ref_count == 0

    # Session 3: Partial match should work
    path_partial, _ = cache.match([sys_seg])
    assert len(path_partial) == 1
    assert path_partial[0] is n_sys
    cache.release(path_partial)
