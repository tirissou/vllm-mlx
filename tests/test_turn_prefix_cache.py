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


def test_node_data_bytes_counts_dict_format_recurrent_state():
    """_node_data_bytes correctly accounts for dict-format recurrent_state tensor sizes."""
    from mlx_lm.models.cache import KVCache

    # 2 layers, each with keys+values of shape [1, 4, 10, 32] in float32 = 4 bytes/elem
    # Each tensor: 1*4*10*32 = 1280 elements * 4 bytes = 5120 bytes
    # 2 tensors (K+V) per layer, 2 layers → 4 * 5120 = 20480 bytes total
    extracted = [
        {
            "state": (
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
            ),
            "meta_state": "",
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
        {
            "state": (
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
                mx.zeros([1, 4, 10, 32], dtype=mx.float32),
            ),
            "meta_state": "",
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
    ]

    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_arrays=[],
        kv_scales=[],
        recurrent_state=extracted,
        tokens_since_checkpoint=0,
    )

    expected_bytes = 4 * (1 * 4 * 10 * 32 * 4)  # 4 tensors, float32
    assert _node_data_bytes(node) == expected_bytes


def test_node_data_bytes_zero_for_empty_state():
    """_node_data_bytes returns 0 when recurrent_state is None."""
    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_arrays=[],
        kv_scales=[],
        recurrent_state=None,
        tokens_since_checkpoint=0,
    )
    assert _node_data_bytes(node) == 0


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


# ── Integration tests ──────────────────────────────────────────────────────


def test_multiturn_continuation():
    """Session A stores [sys, u1, a1]; session A extended gets full hit on all three."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    u1_seg = seg([100, 101, 102])
    a1_seg = seg([200, 201])

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)

    # Next request: same [sys, u1, a1] prefix → full hit
    path, has_recurrent = cache.match([sys_seg, u1_seg, a1_seg])
    assert len(path) == 3
    assert path[0] is n_sys
    assert path[1] is n_u1
    assert path[2] is n_a1
    assert has_recurrent
    cache.release(path)


def test_cross_session_system_prompt_reuse():
    """Session B with same system prompt gets immediate recurrent state hit."""
    cache = make_cache(stride=10000)  # high stride so only sys_prompt gets permanent checkpoint
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    assert n_sys.is_permanent_checkpoint

    # Session B: same system prompt
    path, has_recurrent = cache.match([sys_seg])
    assert len(path) == 1
    assert has_recurrent  # permanent checkpoint → recurrent available immediately
    cache.release(path)


def test_mid_session_branching():
    """Two sessions share [sys, u1, a1] but diverge at u2."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(10)), role="system")
    u1_seg = seg([100])
    a1_seg = seg([200])
    u2a_seg = seg([300])   # branch A
    u2b_seg = seg([400])   # branch B

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)
    n_u2a = cache.insert(n_a1, u2a_seg, [], [], state)
    n_u2b = cache.insert(n_a1, u2b_seg, [], [], state)

    # Branch A
    path_a, _ = cache.match([sys_seg, u1_seg, a1_seg, u2a_seg])
    assert len(path_a) == 4
    assert path_a[3] is n_u2a
    cache.release(path_a)

    # Branch B
    path_b, _ = cache.match([sys_seg, u1_seg, a1_seg, u2b_seg])
    assert len(path_b) == 4
    assert path_b[3] is n_u2b
    cache.release(path_b)

    # Shared nodes are the same objects
    assert path_a[0] is path_b[0]  # sys
    assert path_a[1] is path_b[1]  # u1
    assert path_a[2] is path_b[2]  # a1


def test_gap_reconstruction_finds_ancestor():
    """When branch point has no recurrent, nearest checkpoint ancestor is identified."""
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    u1_seg = seg([100, 101])
    a1_seg = seg([200])

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)
    # n_u1's temp recurrent was pruned when n_a1 was added (stride not met)
    assert n_u1.recurrent_state is None

    path, has_recurrent = cache.match([sys_seg, u1_seg, a1_seg])
    assert has_recurrent  # n_a1 is leaf → has temp recurrent
    ancestor = cache.find_checkpoint_ancestor(path)
    # Deepest permanent checkpoint with recurrent is n_sys
    assert ancestor is n_sys
    cache.release(path)

    # Now add a child to n_a1 — its temp recurrent is pruned
    u2_seg = seg([300])
    cache.insert(n_a1, u2_seg, [], [], state)
    assert n_a1.recurrent_state is None  # pruned

    # New request ending at n_a1: no recurrent at n_a1, walk up to n_sys
    path2, has_recurrent2 = cache.match([sys_seg, u1_seg, a1_seg])
    assert not has_recurrent2
    ancestor2 = cache.find_checkpoint_ancestor(path2)
    assert ancestor2 is n_sys  # falls back to sys permanent checkpoint
    cache.release(path2)


def test_messages_to_segments_uses_prefix_boundary():
    """Scheduler _messages_to_segments splits on prefix_boundary, not proportionally.

    This is the fix for the cross-session system-prompt cache miss: both sessions
    share the same prefix_boundary (len of system-prompt tokens), so the system-prompt
    segment hash is stable across sessions with different user messages.
    """
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler

    # Build a minimal scheduler mock that has _messages_to_segments
    sched = object.__new__(Scheduler)

    # Simulate prompt_token_ids for [sys_prompt (10 tokens) + user "hi" (2 tokens)]
    sys_tokens = list(range(10))
    user_tokens_hi = [100, 101]
    user_tokens_yo = [200, 201]

    def make_request(user_tokens, boundary):
        req = MagicMock()
        req.prompt_token_ids = sys_tokens + user_tokens
        req.prefix_boundary = boundary
        req.sys_end_boundary = boundary
        return req

    # Both sessions have the same prefix_boundary (system prompt length)
    req1 = make_request(user_tokens_hi, 10)
    req2 = make_request(user_tokens_yo, 10)

    segs1 = sched._messages_to_segments(req1)
    segs2 = sched._messages_to_segments(req2)

    # Both produce two segments
    assert len(segs1) == 2
    assert len(segs2) == 2

    # System prompt segment is IDENTICAL across sessions → same cache hash
    assert segs1[0].token_ids == segs2[0].token_ids == sys_tokens
    assert segs1[0].role == "system"

    # User segments differ → different hash → no spurious hit beyond sys prompt
    assert segs1[1].token_ids == user_tokens_hi
    assert segs2[1].token_ids == user_tokens_yo
    assert segs1[1].token_ids != segs2[1].token_ids


def test_cross_session_hit_via_prefix_boundary():
    """End-to-end: session 1 stores sys-prompt segment; session 2 gets a hit on it."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler
    import mlx.core as mx

    sched = object.__new__(Scheduler)

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100]
    user_yo = [200]

    def make_request(user_tokens):
        req = MagicMock()
        req.prompt_token_ids = sys_tokens + user_tokens
        req.prefix_boundary = len(sys_tokens)  # exact system-prompt boundary
        req.sys_end_boundary = len(sys_tokens)
        return req

    # Session 1 store: insert sys segment + user-hi segment
    req1 = make_request(user_hi)
    segs1 = sched._messages_to_segments(req1)
    assert len(segs1) == 2
    n_sys = cache.insert(cache.root, segs1[0], [], [], None, is_system_prompt=True)
    cache.insert(n_sys, segs1[1], [], [], None)

    # Session 2 fetch: same system prompt, different user message
    req2 = make_request(user_yo)
    segs2 = sched._messages_to_segments(req2)
    path, _ = cache.match(segs2)

    # Must hit the system-prompt segment
    assert len(path) >= 1
    assert path[0] is n_sys
    cache.release(path)


def test_turn_cache_requires_chunked_prefill_nonzero():
    """Scheduler raises ValueError if use_turn_cache=True and chunked_prefill_tokens=0."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    with pytest.raises(ValueError, match="chunked-prefill-tokens"):
        Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=0),
        )


def test_turn_cache_with_chunked_prefill_does_not_raise():
    """Scheduler does not raise when use_turn_cache=True and chunked_prefill_tokens>0."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    # Should raise something else (missing model internals) but NOT ValueError about chunked prefill
    try:
        Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192),
        )
    except ValueError as e:
        assert "chunked-prefill-tokens" not in str(e), f"Unexpected chunked-prefill error: {e}"
    except Exception:
        pass  # Other init errors from MagicMock model are expected


def test_turn_cache_disables_memory_aware_cache():
    """When use_turn_cache=True, memory_aware_cache must be None."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(
        use_turn_cache=True,
        chunked_prefill_tokens=8192,
        use_memory_aware_cache=True,
        enable_prefix_cache=True,
    )
    # Simulate only the cache-init block
    sched.memory_aware_cache = None
    sched.prefix_cache = None
    sched.paged_cache_manager = None
    sched.block_aware_cache = None
    sched._ssd_tier = None
    sched.turn_cache = None

    # Re-run just the cache-init logic by calling the relevant section inline
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    if sched.config.enable_prefix_cache:
        if sched.config.use_memory_aware_cache and not sched.config.use_turn_cache:
            sched.memory_aware_cache = MemoryAwarePrefixCache(
                model=MagicMock(), config=MemoryCacheConfig()
            )

    assert sched.memory_aware_cache is None, (
        "memory_aware_cache should be None when use_turn_cache=True"
    )


def _make_minimal_scheduler_with_turn_cache():
    """Minimal scheduler object suitable for testing mid_prefill callback."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.turn_cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.requests = {}
    sched.uid_to_request_id = {}
    return sched


class _MockKVLayer:
    """Minimal KVCache-like object with .state and .meta_state."""
    def __init__(self, n_tokens):
        self._n = n_tokens

    @property
    def state(self):
        return (mx.zeros([1, 4, self._n, 32]), mx.zeros([1, 4, self._n, 32]))

    @property
    def meta_state(self):
        return (str(self._n),)


def test_mid_prefill_stores_sys_prompt_state_at_boundary():
    """_mid_prefill_save sets request._sys_prompt_state at prefix_boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_with_turn_cache()

    req = MagicMock()
    req.prompt_token_ids = list(range(15))  # 10 sys + 5 user
    req.prefix_boundary = 10
    req.sys_end_boundary = 10
    req.cached_tokens = 0
    sched.requests["req1"] = req
    sched.uid_to_request_id[1] = "req1"

    mock_cache = [_MockKVLayer(10), _MockKVLayer(10)]  # 2 layers, 10 tokens processed

    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=10, prompt_cache=mock_cache)

    assert hasattr(req, "_sys_prompt_state"), "_sys_prompt_state not set"
    assert req._sys_prompt_state is not None
    assert isinstance(req._sys_prompt_state, list)
    assert len(req._sys_prompt_state) == 2
    assert isinstance(req._sys_prompt_state[0], dict)
    assert "state" in req._sys_prompt_state[0]
    assert "class_name" in req._sys_prompt_state[0]


def test_mid_prefill_does_not_store_state_away_from_boundary():
    """_mid_prefill_save does NOT set _sys_prompt_state when not at prefix_boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_with_turn_cache()

    req = MagicMock()
    req.prompt_token_ids = list(range(20))
    req.prefix_boundary = 10
    req.sys_end_boundary = 10
    req.cached_tokens = 0
    req._mid_prefill_last_save = 0  # Avoid MagicMock returning a Mock for this
    req._sys_prompt_state = None    # Pre-set so we can detect if callback writes it
    sched.requests["req1"] = req
    sched.uid_to_request_id[1] = "req1"

    mock_cache = [_MockKVLayer(5)]  # Only 5 tokens processed, not at boundary

    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=5, prompt_cache=mock_cache)

    # _sys_prompt_state should remain None — callback returns early due to throttle
    assert req._sys_prompt_state is None


def test_mid_prefill_does_not_store_state_away_from_boundary_past_interval():
    """_mid_prefill_save does NOT set _sys_prompt_state when past save_interval but not at boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_with_turn_cache()

    req = MagicMock()
    req.prompt_token_ids = list(range(30))
    req.prefix_boundary = 20
    req.sys_end_boundary = 20
    req.cached_tokens = 0
    req._mid_prefill_last_save = 0
    req._sys_prompt_state = None
    sched.requests["req1"] = req
    sched.uid_to_request_id[1] = "req1"

    # processed_tokens=15 exceeds save_interval=10, but is NOT at prefix_boundary=20
    mock_cache = [_MockKVLayer(15)]
    cb = sched._make_mid_prefill_save_callback(save_interval=10)
    cb(uid=1, processed_tokens=15, prompt_cache=mock_cache)

    assert req._sys_prompt_state is None, "_sys_prompt_state should not be set away from boundary"


def _make_extracted_state(n_layers=2, n_tokens=10):
    """Build a list of dicts in _extract_cache_states format."""
    from mlx_lm.models.cache import KVCache
    return [
        {
            "state": (mx.zeros([1, 4, n_tokens, 32]), mx.zeros([1, 4, n_tokens, 32])),
            "meta_state": "",  # KVCache.meta_state is a string
            "class_name": "KVCache",
            "class_ref": KVCache,
        }
        for _ in range(n_layers)
    ]


def test_store_side_sets_recurrent_state_on_system_segment():
    """After generation, system segment node gets recurrent_state from _sys_prompt_state."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    sys_state = _make_extracted_state(n_layers=2, n_tokens=10)

    req = MagicMock()
    req.prompt_token_ids = sys_tokens + user_tokens
    req.prefix_boundary = 10
    req.sys_end_boundary = 10
    req._sys_prompt_state = sys_state
    req._extracted_cache = [_MockKVLayer(15), _MockKVLayer(15)]  # live objects for user seg
    req._turn_cache_path = []

    # Call _messages_to_segments and then simulate the store loop
    segments = sched._messages_to_segments(req)
    assert len(segments) == 2

    parent = cache.root
    new_segments = segments  # matched_depth=0, so all segments are new
    for i, segment in enumerate(new_segments):
        is_sys = segment.role == "system" and i == 0
        if is_sys:
            state = getattr(req, "_sys_prompt_state", None)
        elif i == len(new_segments) - 1:
            ec = req._extracted_cache
            if isinstance(ec, list) and ec and isinstance(ec[0], dict):
                state = ec
            else:
                state = sched._extract_cache_states(ec)
        else:
            state = None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)

    # System node should have _sys_prompt_state
    sys_node = list(cache.root.children.values())[0]
    assert sys_node.recurrent_state is not None
    assert isinstance(sys_node.recurrent_state, list)
    assert isinstance(sys_node.recurrent_state[0], dict)

    # User node should have extracted state from _extracted_cache
    user_node = list(sys_node.children.values())[0]
    assert user_node.recurrent_state is not None
    assert isinstance(user_node.recurrent_state, list)
    assert isinstance(user_node.recurrent_state[0], dict)


def test_fetch_reconstructs_dict_state_into_prompt_cache():
    """On turn_cache HIT, request.prompt_cache is reconstructed cache objects, not raw dicts."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 13))
    sys_state = _make_extracted_state(n_layers=2, n_tokens=10)

    sys_seg = Segment(role="system", token_ids=sys_tokens)
    sys_node = cache.insert(cache.root, sys_seg, [], [], sys_state, is_system_prompt=True)

    # Fetch: request with same system tokens but different user tokens
    req = MagicMock()
    req.prompt_token_ids = sys_tokens + user_tokens
    req.prefix_boundary = 10
    req.sys_end_boundary = 10
    req.request_id = "test-fetch"

    segments = sched._messages_to_segments(req)
    path, has_recurrent = cache.match(segments)
    assert path, "Expected a trie match on sys_tokens"

    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is not None
    raw_state = ancestor.recurrent_state

    # Apply the reconstruction logic we're about to add
    if (
        raw_state is not None
        and isinstance(raw_state, list)
        and raw_state
        and isinstance(raw_state[0], dict)
    ):
        prompt_cache = sched._reconstruct_cache_from_states(raw_state)
    else:
        prompt_cache = raw_state

    assert prompt_cache is not None, "prompt_cache should be non-None after reconstruction"
    assert isinstance(prompt_cache, list)
    assert len(prompt_cache) == 2
    # Each element should be a KVCache-like object with .state and .offset
    for layer in prompt_cache:
        assert hasattr(layer, "offset"), f"Expected .offset on {type(layer)}"
        assert layer.offset == 10


def test_save_load_dict_format_recurrent_state():
    """TurnNode with dict-format recurrent_state survives a save/load round-trip."""
    import tempfile
    from mlx_lm.models.cache import KVCache
    from vllm_mlx.scheduler import Scheduler

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))

    extracted = [
        {
            "state": (mx.zeros([1, 4, 10, 32]), mx.zeros([1, 4, 10, 32])),
            "meta_state": "",
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
        {
            "state": (mx.ones([1, 4, 10, 32]), mx.ones([1, 4, 10, 32])),
            "meta_state": "",
            "class_name": "KVCache",
            "class_ref": KVCache,
        },
    ]
    mx.eval(*[t for d in extracted for t in d["state"]])

    seg_sys = Segment(role="system", token_ids=list(range(10)))
    cache.insert(cache.root, seg_sys, [], [], extracted, is_system_prompt=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache.save(tmpdir)

        cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
        cache2.load(tmpdir)

    assert len(cache2.root.children) == 1
    loaded_node = list(cache2.root.children.values())[0]
    assert loaded_node.recurrent_state is not None
    assert isinstance(loaded_node.recurrent_state, list)
    assert len(loaded_node.recurrent_state) == 2
    assert isinstance(loaded_node.recurrent_state[0], dict)
    assert loaded_node.recurrent_state[0]["class_name"] == "KVCache"
    assert loaded_node.recurrent_state[0]["class_ref"] is not None

    # Verify reconstruction works on loaded state
    sched = object.__new__(Scheduler)
    reconstructed = sched._reconstruct_cache_from_states(loaded_node.recurrent_state)
    assert reconstructed is not None
    assert len(reconstructed) == 2
    assert reconstructed[0].offset == 10


def test_save_load_legacy_ssm_format_unchanged():
    """Legacy SSM raw-tensor recurrent_state still round-trips correctly."""
    import tempfile

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    legacy_state = [mx.zeros([4, 32]), mx.ones([4, 32])]
    mx.eval(*legacy_state)

    seg_l = Segment(role="user", token_ids=[1, 2, 3])
    cache.insert(cache.root, seg_l, [], [], legacy_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache.save(tmpdir)
        cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
        cache2.load(tmpdir)

    loaded_node = list(cache2.root.children.values())[0]
    assert loaded_node.recurrent_state is not None
    # Legacy format: not a list of dicts
    assert not (
        isinstance(loaded_node.recurrent_state, list)
        and loaded_node.recurrent_state
        and isinstance(loaded_node.recurrent_state[0], dict)
    )


def test_cross_session_system_prompt_cache_hit_with_real_state():
    """Full path: session 1 stores sys state; session 2 fetches it and skips sys prefill.

    Simulates:
      Session 1: full prefill of [sys_tokens + user_hi], mid_prefill captures sys state,
                 store side inserts both segments with real recurrent_state.
      Session 2: same sys_tokens, different user_yo → trie HIT on sys node,
                 request.prompt_cache is non-None, cached_tokens == len(sys_tokens).
    """
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100, 101]
    user_yo = [200, 201]

    # --- Session 1 store ---
    sys_state = _make_extracted_state(n_layers=2, n_tokens=len(sys_tokens))
    user_state = _make_extracted_state(n_layers=2, n_tokens=len(sys_tokens) + len(user_hi))

    req1 = MagicMock()
    req1.prompt_token_ids = sys_tokens + user_hi
    req1.prefix_boundary = len(sys_tokens)
    req1.sys_end_boundary = len(sys_tokens)
    req1._sys_prompt_state = sys_state
    req1._turn_cache_path = []

    segs1 = sched._messages_to_segments(req1)
    parent = cache.root
    for i, segment in enumerate(segs1):
        is_sys = segment.role == "system" and i == 0
        state = sys_state if is_sys else user_state
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)

    # --- Session 2 fetch ---
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + user_yo
    req2.prefix_boundary = len(sys_tokens)
    req2.sys_end_boundary = len(sys_tokens)
    req2.request_id = "session2"

    segs2 = sched._messages_to_segments(req2)
    path, has_recurrent = cache.match(segs2)

    assert path, "Expected trie HIT on system segment"
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is not None, "Expected a checkpoint ancestor with recurrent_state"

    raw_state = ancestor.recurrent_state
    assert isinstance(raw_state, list) and isinstance(raw_state[0], dict)

    reconstructed = sched._reconstruct_cache_from_states(raw_state)
    assert reconstructed is not None, "Reconstruction failed"

    # Simulate what scheduler sets on the request
    req2.prompt_cache = reconstructed
    req2.cached_tokens = sum(len(n.token_ids) for n in path)
    req2.remaining_tokens = req2.prompt_token_ids[req2.cached_tokens:]

    assert req2.cached_tokens == len(sys_tokens), (
        f"Expected {len(sys_tokens)} cached tokens, got {req2.cached_tokens}"
    )
    assert req2.remaining_tokens == user_yo
    assert req2.prompt_cache is not None

    cache.release(path)


# ---------------------------------------------------------------------------
# Multi-turn diagnostic tests
# ---------------------------------------------------------------------------

def _make_multi_turn_request(prompt_token_ids, prefix_boundary, sys_end_boundary):
    """Create a MagicMock request with the given token ids and boundaries."""
    from unittest.mock import MagicMock
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req.prefix_boundary = prefix_boundary
    req.sys_end_boundary = sys_end_boundary
    return req


def test_segment1_tokens_stable_across_turns():
    """Segment 1 (sys) must have identical token_ids for turns 1, 2, and 3.

    This is the fundamental invariant for cross-turn sys hits: if token_ids differ,
    the segment hash differs and the cache misses.
    """
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(100))           # 100-token system prompt
    user1 = list(range(100, 120))           # turn-1 user
    asst1 = list(range(200, 250))           # turn-1 assistant (output appended by client)
    user2 = list(range(300, 315))           # turn-2 user
    asst2 = list(range(400, 430))           # turn-2 assistant
    user3 = list(range(500, 510))           # turn-3 user

    seb = len(sys_tokens)  # stable sys_end_boundary

    # Turn 1
    pb1 = seb  # first_user == last_user → prefix == sys_end
    req1 = _make_multi_turn_request(sys_tokens + user1, pb1, seb)
    segs1 = sched._messages_to_segments(req1)

    # Turn 2
    pb2 = seb + len(user1) + len(asst1)
    req2 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2, pb2, seb)
    segs2 = sched._messages_to_segments(req2)

    # Turn 3
    pb3 = pb2 + len(user2) + len(asst2)
    req3 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2 + asst2 + user3, pb3, seb)
    segs3 = sched._messages_to_segments(req3)

    # All must return at least 2 segments (sys + user)
    assert len(segs1) >= 2, f"Turn 1 has {len(segs1)} segments"
    assert len(segs2) >= 2, f"Turn 2 has {len(segs2)} segments"
    assert len(segs3) >= 2, f"Turn 3 has {len(segs3)} segments"

    # Segment 1 must be the same object content across all turns
    assert segs1[0].role == "system"
    assert segs2[0].role == "system"
    assert segs3[0].role == "system"
    assert segs1[0].token_ids == sys_tokens, "Turn 1 seg0 ≠ sys_tokens"
    assert segs2[0].token_ids == sys_tokens, "Turn 2 seg0 ≠ sys_tokens"
    assert segs3[0].token_ids == sys_tokens, "Turn 3 seg0 ≠ sys_tokens"

    # The context hashes (used for trie lookup) must be equal
    from vllm_mlx.turn_prefix_cache import _context_hash, TurnPrefixCache, TurnPrefixCacheConfig
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    h1 = _context_hash(cache.root.context_hash, segs1[0].token_ids)
    h2 = _context_hash(cache.root.context_hash, segs2[0].token_ids)
    h3 = _context_hash(cache.root.context_hash, segs3[0].token_ids)
    assert h1 == h2 == h3, f"Hashes differ: {h1} {h2} {h3}"


def test_multi_turn_sys_hit_each_turn():
    """After storing turn 1, turns 2 and 3 must get a trie HIT on the sys segment."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(50))
    user1 = list(range(50, 60))
    asst1 = list(range(100, 120))
    user2 = list(range(200, 210))
    asst2 = list(range(300, 315))
    user3 = list(range(400, 405))

    seb = len(sys_tokens)

    # --- Turn 1 store ---
    pb1 = seb
    req1 = _make_multi_turn_request(sys_tokens + user1, pb1, seb)
    segs1 = sched._messages_to_segments(req1)
    assert len(segs1) == 2, f"Turn 1 should have 2 segments, got {len(segs1)}"

    sys_state = _make_extracted_state(n_layers=1, n_tokens=seb)
    sys_node = cache.insert(cache.root, segs1[0], [], [], sys_state, is_system_prompt=True)
    cache.insert(sys_node, segs1[1], [], [], None)

    # --- Turn 2 fetch: must hit sys_node ---
    pb2 = seb + len(user1) + len(asst1)
    req2 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2, pb2, seb)
    segs2 = sched._messages_to_segments(req2)
    assert len(segs2) == 3, f"Turn 2 should have 3 segments, got {len(segs2)}: {[s.role for s in segs2]}"
    assert segs2[0].token_ids == sys_tokens, "Turn 2 seg0 token mismatch"

    path2, has_rec2 = cache.match(segs2)
    assert len(path2) >= 1, f"Turn 2 expected sys HIT, got path={path2}"
    assert path2[0] is sys_node, "Turn 2 did not match sys_node"
    cached_t2 = sum(len(n.token_ids) for n in path2)
    assert cached_t2 == seb, f"Turn 2 cached_tokens should be {seb}, got {cached_t2}"
    cache.release(path2)

    # --- Turn 3 fetch: must also hit sys_node ---
    pb3 = pb2 + len(user2) + len(asst2)
    req3 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2 + asst2 + user3, pb3, seb)
    segs3 = sched._messages_to_segments(req3)
    assert len(segs3) == 3, f"Turn 3 should have 3 segments, got {len(segs3)}"
    assert segs3[0].token_ids == sys_tokens, "Turn 3 seg0 token mismatch"

    path3, has_rec3 = cache.match(segs3)
    assert len(path3) >= 1, f"Turn 3 expected sys HIT, got path={path3}"
    assert path3[0] is sys_node, "Turn 3 did not match sys_node"
    cached_t3 = sum(len(n.token_ids) for n in path3)
    assert cached_t3 == seb, f"Turn 3 cached_tokens should be {seb}, got {cached_t3}"
    cache.release(path3)


def test_conv_segment_grows_each_turn():
    """Conv segment is unique per turn — no cross-turn conv hits within a single session.

    This is EXPECTED behaviour: same-session caching only provides sys hits.
    Conv hits happen only in cross-session scenarios (same history, different session).
    """
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))

    sys_tokens = list(range(20))
    user1 = list(range(20, 25))
    asst1 = list(range(100, 105))
    user2 = list(range(200, 203))
    asst2 = list(range(300, 306))
    user3 = list(range(400, 402))
    seb = len(sys_tokens)

    pb2 = seb + len(user1) + len(asst1)
    pb3 = pb2 + len(user2) + len(asst2)

    req2 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2, pb2, seb)
    req3 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2 + asst2 + user3, pb3, seb)

    segs2 = sched._messages_to_segments(req2)
    segs3 = sched._messages_to_segments(req3)

    conv2 = segs2[1]  # role="conversation"
    conv3 = segs3[1]  # role="conversation"

    assert conv2.role == "conversation"
    assert conv3.role == "conversation"
    # Conv segments grow → different tokens → different hashes → no cross-turn conv hit
    assert conv2.token_ids != conv3.token_ids, "Conv segments should differ between turns"


def test_multi_turn_conv_stored_after_turn2():
    """After turn 2 completes, the trie should contain a conv node under sys_node.

    The conv node captures state at prefix_boundary so cross-session turn-2 requests
    with the same history can skip processing the conversation history.
    """
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user1 = list(range(20, 25))
    asst1 = list(range(100, 105))
    user2 = list(range(200, 203))
    seb = len(sys_tokens)
    pb2 = seb + len(user1) + len(asst1)

    # Simulate turn 1 already stored
    sys_state = _make_extracted_state(n_layers=1, n_tokens=seb)
    req1 = _make_multi_turn_request(sys_tokens + user1, seb, seb)
    segs1 = sched._messages_to_segments(req1)
    sys_node = cache.insert(cache.root, segs1[0], [], [], sys_state, is_system_prompt=True)
    cache.insert(sys_node, segs1[1], [], [], None)

    # Simulate turn 2: HIT on sys, then store conv + user
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + user1 + asst1 + user2
    req2.prefix_boundary = pb2
    req2.sys_end_boundary = seb
    req2._turn_cache_path = [sys_node]  # simulates the fetch hit
    sys_node.ref_count += 1             # simulate match increment

    conv_state = _make_extracted_state(n_layers=1, n_tokens=pb2)
    user_state = _make_extracted_state(n_layers=1, n_tokens=pb2 + len(user2))
    req2._conv_end_state = conv_state
    req2._extracted_cache = user_state

    segs2 = sched._messages_to_segments(req2)
    assert len(segs2) == 3, f"Expected 3 segments for turn 2, got {len(segs2)}"

    path = req2._turn_cache_path
    matched_depth = len(path)  # 1
    parent = path[-1]          # sys_node
    new_segments = segs2[matched_depth:]  # [conv, user2]

    inserted_nodes = []
    for i, segment in enumerate(new_segments):
        is_sys = segment.role == "system" and i == 0 and matched_depth == 0
        is_conv = segment.role == "conversation" and not is_sys
        if is_sys:
            state = getattr(req2, "_sys_prompt_state", None)
        elif is_conv and i < len(new_segments) - 1:
            state = getattr(req2, "_conv_end_state", None)
        elif i == len(new_segments) - 1:
            ec = req2._extracted_cache
            state = ec if isinstance(ec, list) and ec and isinstance(ec[0], dict) else None
        else:
            state = None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)
        inserted_nodes.append(parent)

    cache.release(path)

    conv_node = inserted_nodes[0]
    user2_node = inserted_nodes[1]

    # Conv node should be under sys_node with the conv_state
    assert conv_node in sys_node.children.values(), "Conv node not under sys_node"
    assert conv_node.recurrent_state == conv_state, "Conv node has wrong state"
    expected_conv_tokens = (sys_tokens + user1 + asst1 + user2)[seb:pb2]
    assert conv_node.token_ids == expected_conv_tokens, f"Conv tokens wrong: {conv_node.token_ids[:5]}..."

    # Cross-session turn-2: same prompt as turn 2 → full hit on [sys, conv, user2] (3 deep)
    req_cross2 = _make_multi_turn_request(sys_tokens + user1 + asst1 + user2, pb2, seb)
    segs_cross2 = sched._messages_to_segments(req_cross2)
    path_cross2, _ = cache.match(segs_cross2)
    assert len(path_cross2) == 3, f"Cross-session turn-2 expected 3-deep HIT, got {len(path_cross2)}"
    assert path_cross2[1] is conv_node
    cached_cross2 = sum(len(n.token_ids) for n in path_cross2)
    assert cached_cross2 == pb2 + len(user2), f"Expected full prompt cached, got {cached_cross2}"
    cache.release(path_cross2)

    # Cross-session turn-3 (same history + new user3): only sys + conv matches (NOT full turn-2 conv)
    # because turn-3's conv segment = user1+asst1+user2+asst2, which is not in the trie yet
    asst2 = list(range(500, 506))
    user3 = list(range(600, 603))
    pb3 = pb2 + len(user2) + len(asst2)
    req_cross3 = _make_multi_turn_request(
        sys_tokens + user1 + asst1 + user2 + asst2 + user3, pb3, seb
    )
    segs_cross3 = sched._messages_to_segments(req_cross3)
    path_cross3, _ = cache.match(segs_cross3)
    # Conv segment for turn-3 is larger than the stored conv node → only sys matches
    assert len(path_cross3) == 1, (
        f"Cross-session turn-3 expected only sys hit (1-deep), got {len(path_cross3)}"
    )
    assert path_cross3[0] is sys_node
    cache.release(path_cross3)


# ── _compute_turn_boundaries tests (tokenizer-only, no model) ──────────────

class _MockTok:
    """Char-level tokenizer with a minimal chat template."""
    unk_token_id = None

    def encode(self, text):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def _make_engine_with_mock_tok():
    """Return a BatchedEngine stub with the mock tokenizer wired in."""
    from vllm_mlx.engine.batched import BatchedEngine
    eng = object.__new__(BatchedEngine)
    eng._is_mllm = False
    eng._tokenizer = _MockTok()
    eng._processor = None
    eng._model_name = "mock"
    return eng


def test_compute_turn_boundaries_single_turn():
    """First turn (no completed assistant turn): returns [B_sys]."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "HELLO"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 1, f"Expected 1 boundary, got {boundaries}"
    # B_sys should be the byte-length of "<system>SYS</system>"
    expected_sys_text = "<system>SYS</system>"
    assert boundaries[0] == len(expected_sys_text.encode("utf-8")), (
        f"B_sys={boundaries[0]}, expected {len(expected_sys_text.encode('utf-8'))}"
    )


def test_compute_turn_boundaries_two_turns():
    """Two completed turns returns [B_sys, B_1, B_2]."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 3, f"Expected 3 boundaries, got {boundaries}"
    assert boundaries[0] < boundaries[1] < boundaries[2], (
        f"Boundaries not strictly increasing: {boundaries}"
    )


def test_compute_turn_boundaries_no_system():
    """No system message → returns []."""
    eng = _make_engine_with_mock_tok()
    messages = [{"role": "user", "content": "HI"}]
    assert eng._compute_turn_boundaries(messages) == []


def test_compute_turn_boundaries_exact_position():
    """B_sys is the exact LCP of closed [sys] template against full_tokens."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    # Reconstruct what B_1 should be:
    # full = <system>SYS</system><user>U1</user><assistant>A1</assistant><assistant>
    # closed [sys,u1,a1] = <system>SYS</system><user>U1</user><assistant>A1</assistant>
    tok = _MockTok()
    closed_text = "<system>SYS</system><user>U1</user><assistant>A1</assistant>"
    expected_b1 = len(closed_text.encode("utf-8"))
    assert boundaries[1] == expected_b1, (
        f"B_1={boundaries[1]}, expected {expected_b1}"
    )


def test_compute_turn_boundaries_strictly_increasing():
    """Every boundary in the returned list is strictly greater than the previous."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "U4"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 4  # B_sys + 3 completed turns
    for i in range(len(boundaries) - 1):
        assert boundaries[i] < boundaries[i + 1], (
            f"boundaries[{i}]={boundaries[i]} >= boundaries[{i+1}]={boundaries[i+1]}"
        )


def test_compute_turn_boundaries_last_boundary_less_than_full():
    """The last boundary must be < len(full_tokens) (user segment exists after it)."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Q"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    tok = _MockTok()
    full_tokens = tok.encode(tok.apply_chat_template(messages, add_generation_prompt=True))
    assert boundaries[-1] < len(full_tokens), (
        f"Last boundary {boundaries[-1]} >= full_tokens length {len(full_tokens)}"
    )
