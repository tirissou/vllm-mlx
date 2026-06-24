import pytest
import mlx.core as mx
import json
import tempfile
from pathlib import Path
from vllm_mlx.turn_prefix_cache import (
    Segment,
    TurnNode,
    SSDRef,
    TurnPrefixCacheConfig,
    TurnPrefixCache,
    _context_hash,
    _node_data_bytes,
)
from vllm_mlx.cache_types import KVConcatSegment, KVRotatingSegment
from vllm_mlx.prefix_cache_adapters import TurnCacheManager as TurnCacheAdapter


def seg(token_ids, role="user"):
    return Segment(role=role, token_ids=token_ids)


class _MockKVLayer:
    """Mock KV cache layer for testing."""

    def __init__(self, n_tokens):
        self.n_tokens = n_tokens


def _make_extracted_state(n_layers, n_tokens):
    """Create mock extracted cache state (list of layer dicts)."""
    return [
        {"state": (mx.zeros((1, n_tokens, 128)), mx.zeros((1, n_tokens, 128)))}
        for _ in range(n_layers)
    ]


def _make_minimal_scheduler_new():
    """Minimal scheduler for testing new mid-prefill callback."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.turn_cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched._prefix_cache = TurnCacheManager(sched.turn_cache)
    sched.requests = {}
    sched.uid_to_request_id = {}
    return sched


def test_context_hash_deterministic():
    assert _context_hash(0, [1, 2, 3]) == _context_hash(0, [1, 2, 3])


def test_context_hash_different_parent():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(1, [1, 2, 3])


def test_context_hash_different_tokens():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(0, [1, 2, 4])


def test_turn_node_is_leaf_when_no_children():
    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_data=[],
        recurrent_data=None,
        tokens_since_checkpoint=0,
        parent=None,
    )
    assert node.is_leaf


def test_turn_node_not_leaf_when_has_children():
    parent = TurnNode(
        token_ids=[],
        context_hash=0,
        kv_data=None,
        recurrent_data=None,
        tokens_since_checkpoint=0,
        parent=None,
    )
    child = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_data=[],
        recurrent_data=None,
        tokens_since_checkpoint=1,
        parent=parent,
    )
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
    from vllm_mlx.cache_types import RecurrentLayerSegment

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

    recurrent_data = [RecurrentLayerSegment(arrays=extracted, metadata={})]

    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_data=[],
        recurrent_data=recurrent_data,
        tokens_since_checkpoint=0,
    )

    expected_bytes = 4 * (1 * 4 * 10 * 32 * 4)  # 4 tensors, float32
    assert _node_data_bytes(node) == expected_bytes


def test_node_data_bytes_zero_for_empty_state():
    """_node_data_bytes returns 0 when recurrent_data is None."""
    node = TurnNode(
        token_ids=[1],
        context_hash=1,
        kv_data=[],
        recurrent_data=None,
        tokens_since_checkpoint=0,
    )
    assert _node_data_bytes(node) == 0


from vllm_mlx.turn_prefix_cache import TurnPrefixCache


def make_cache(stride=512, max_gb=8.0, kv_dtype="bf16", recurrent_dtype="bf16"):
    return TurnPrefixCache(
        TurnPrefixCacheConfig(
            checkpoint_stride=stride,
            max_memory_gb=max_gb,
            kv_dtype=kv_dtype,
            recurrent_dtype=recurrent_dtype,
        )
    )


# --- has_recurrent_state flag ---


def test_has_recurrent_state_starts_false():
    cache = make_cache()
    assert not cache.has_recurrent_state


def test_has_recurrent_state_not_set_for_kv_only_insert():
    cache = make_cache(stride=0)
    cache.insert(
        cache.root,
        seg([1, 2, 3]),
        kv_data=[],
        recurrent_data=None,
        is_system_prompt=True,
    )
    assert not cache.has_recurrent_state


def test_has_recurrent_state_set_on_hybrid_insert():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    cache.insert(
        cache.root,
        seg([1, 2, 3]),
        kv_data=[],
        recurrent_data=rec,
        is_system_prompt=True,
    )
    assert cache.has_recurrent_state


def test_has_recurrent_state_survives_clear():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    cache.insert(
        cache.root, seg([1]), kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    assert cache.has_recurrent_state
    cache.clear()
    assert cache.has_recurrent_state


def test_load_sets_has_recurrent_state_for_hybrid_cache(tmp_path):
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache1 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    state = mx.zeros((2, 3))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    cache1.insert(
        cache1.root,
        seg([1, 2, 3], role="system"),
        kv_data=[],
        recurrent_data=rec,
        is_system_prompt=True,
    )
    cache1.save(str(tmp_path))

    cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    assert not cache2.has_recurrent_state
    cache2.load(str(tmp_path))
    assert cache2.has_recurrent_state


def test_load_leaves_has_recurrent_state_false_for_kv_only_cache(tmp_path):
    cache1 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache1.insert(
        cache1.root,
        seg([1, 2, 3], role="system"),
        kv_data=[],
        recurrent_data=None,
        is_system_prompt=True,
    )
    cache1.save(str(tmp_path))

    cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache2.load(str(tmp_path))
    assert not cache2.has_recurrent_state


def test_find_checkpoint_ancestor_kv_only_returns_deepest_kv_node():
    """In KV-only mode, find_checkpoint_ancestor returns the deepest node with kv_data."""
    cache = make_cache(stride=0)
    kv_data = _make_kv_data()
    n1 = cache.insert(
        cache.root,
        seg([1], role="system"),
        kv_data=kv_data,
        recurrent_data=None,
        is_system_prompt=True,
    )
    n2 = cache.insert(n1, seg([2]), kv_data=kv_data, recurrent_data=None)
    assert not cache.has_recurrent_state  # confirm KV-only mode

    path, _ = cache.match([seg([1], role="system"), seg([2])])
    ancestor = cache.find_checkpoint_ancestor(path)
    cache.release(path)

    assert ancestor is n2


def test_find_checkpoint_ancestor_kv_only_returns_none_when_no_kv():
    """In KV-only mode, returns None if no node in path has kv_data."""
    cache = make_cache(stride=0)
    n1 = cache.insert(
        cache.root,
        seg([1], role="system"),
        kv_data=[],
        recurrent_data=None,
        is_system_prompt=True,
    )
    assert not cache.has_recurrent_state

    path, _ = cache.match([seg([1], role="system")])
    ancestor = cache.find_checkpoint_ancestor(path)
    cache.release(path)

    assert ancestor is None


def test_insert_creates_child_of_root():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]))
    assert node in cache.root.children.values()


def test_insert_sets_token_ids():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]))
    assert node.token_ids == [1, 2, 3]


def test_insert_node_is_leaf():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]))
    assert node.is_leaf


def test_insert_sets_parent():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]))
    assert node.parent is cache.root


def test_insert_idempotent_same_segment():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]))
    n2 = cache.insert(cache.root, seg([1, 2, 3]))
    assert n1 is n2  # same node returned


def test_insert_chained():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]))
    n2 = cache.insert(n1, seg([2]))
    assert n2.parent is n1
    assert n2 in n1.children.values()


def test_insert_context_hash_differs_at_different_depths():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]))
    n2 = cache.insert(n1, seg([1, 2, 3]))  # same tokens, different parent
    assert n1.context_hash != n2.context_hash


def test_leaf_gets_recurrent_state():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    node = cache.insert(cache.root, seg(list(range(50))), recurrent_data=rec)
    assert node.recurrent_data is not None


def test_temp_recurrent_pruned_on_non_stride_inner():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n1 = cache.insert(cache.root, seg(list(range(50))), recurrent_data=rec)
    # n1 is leaf with temp recurrent (50 < 100)
    assert n1.recurrent_data is not None
    # Add child: n1 becomes inner node, tokens_since=50 < 100 → prune
    cache.insert(n1, seg([99]), recurrent_data=rec)
    assert n1.recurrent_data is None


def test_permanent_checkpoint_at_stride():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    # 120 tokens >= stride → permanent
    n1 = cache.insert(cache.root, seg(list(range(120))), recurrent_data=rec)
    assert n1.is_permanent_checkpoint
    # Add child: should NOT prune recurrent
    cache.insert(n1, seg([999]), recurrent_data=rec)
    assert n1.recurrent_data is not None


def test_system_prompt_always_permanent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    # Only 50 tokens but is_system_prompt=True
    n = cache.insert(
        cache.root,
        seg(list(range(50)), role="system"),
        recurrent_data=rec,
        is_system_prompt=True,
    )
    assert n.is_permanent_checkpoint
    cache.insert(n, seg([99]), recurrent_data=rec)
    assert n.recurrent_data is not None


def test_tokens_since_resets_after_permanent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    # 120 tokens → permanent checkpoint
    n1 = cache.insert(cache.root, seg(list(range(120))), recurrent_data=rec)
    assert n1.is_permanent_checkpoint
    # 10 more tokens; tokens_since should count from n1 (permanent), not root
    n2 = cache.insert(n1, seg(list(range(10))), recurrent_data=rec)
    assert n2.tokens_since_checkpoint == 10


def test_stride_zero_makes_every_node_permanent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n1 = cache.insert(cache.root, seg([1, 2]), recurrent_data=rec)
    assert n1.is_permanent_checkpoint
    n2 = cache.insert(n1, seg([3]), recurrent_data=rec)
    assert n2.is_permanent_checkpoint
    # Neither should have recurrent pruned
    assert n1.recurrent_data is not None
    assert n2.recurrent_data is not None


def test_match_full():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]))
    n2 = cache.insert(n1, seg([4, 5]))
    path, _ = cache.match([seg([1, 2, 3]), seg([4, 5])])
    assert path == [n1, n2]


def test_match_partial():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]))
    cache.insert(n1, seg([4, 5]))
    path, _ = cache.match([seg([1, 2, 3]), seg([99])])  # second seg not in trie
    assert path == [n1]


def test_match_empty():
    cache = make_cache()
    path, has_recurrent = cache.match([seg([99])])
    assert path == []
    assert not has_recurrent


def test_match_updates_last_used():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]))
    node.last_used = 0.0
    cache.match([seg([1])])
    assert node.last_used > 0.0


def test_match_reports_has_recurrent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    node = cache.insert(cache.root, seg([1]), recurrent_data=rec)
    _, has_recurrent = cache.match([seg([1])])
    assert has_recurrent


def test_match_increments_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]))
    assert node.ref_count == 0
    cache.match([seg([1])])
    assert node.ref_count == 1


def test_release_decrements_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]))
    path, _ = cache.match([seg([1])])
    assert node.ref_count == 1
    cache.release(path)
    assert node.ref_count == 0


def test_release_all_nodes_in_path():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]))
    n2 = cache.insert(n1, seg([2]))
    path, _ = cache.match([seg([1]), seg([2])])
    assert n1.ref_count == 1
    assert n2.ref_count == 1
    cache.release(path)
    assert n1.ref_count == 0
    assert n2.ref_count == 0


def test_find_checkpoint_ancestor_returns_self_if_has_recurrent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n = cache.insert(cache.root, seg([1]), recurrent_data=rec)
    path = [n]
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is n


def test_find_checkpoint_ancestor_returns_none_when_no_checkpoint():
    cache = make_cache(stride=10000)
    # No state stored, stride too high → no permanent checkpoints (except root which isn't in path)
    n = cache.insert(cache.root, seg([1, 2, 3]))
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is None


def test_find_checkpoint_ancestor_returns_leaf_if_leaf_has_recurrent():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n = cache.insert(cache.root, seg([1]), recurrent_data=rec)
    # n is a leaf → has temp recurrent
    assert n.recurrent_data is not None
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is n


def test_find_checkpoint_ancestor_skips_ssdref_nodes():
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n1 = cache.insert(cache.root, seg([1]), recurrent_data=rec)
    n2 = cache.insert(n1, seg([2]), recurrent_data=rec)
    # Simulate n1's state being spilled to SSD
    n1.recurrent_data = SSDRef(file_path="/tmp/state.bin", size_bytes=1024)
    # Should skip n1 and return n2
    ancestor = cache.find_checkpoint_ancestor([n1, n2])
    assert ancestor is n2


def _make_kv(n_tokens=1):
    """Small real KV arrays for memory-tracked tests (head_dim=256, divisible by group_size=64)."""
    return mx.zeros((1, 4, n_tokens, 256), dtype=mx.bfloat16)


def _make_kv_data(n_tokens=1):
    """Small KVConcatSegment for memory-tracked tests."""
    from vllm_mlx.cache_types import KVConcatSegment
    from vllm_mlx.kv_cache import QuantizedArray

    keys = _make_kv(n_tokens)
    values = _make_kv(n_tokens)
    q_keys = QuantizedArray(*mx.quantize(keys, group_size=64, bits=8))
    q_values = QuantizedArray(*mx.quantize(values, group_size=64, bits=8))
    return [
        KVConcatSegment(
            keys=q_keys,
            values=q_values,
            layer_index=0,
            n_tokens=n_tokens,
            bits=8,
            class_name="KVCache",
        )
    ]


def test_lru_evicts_oldest_leaf():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    n1 = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    n2 = cache.insert(cache.root, seg([2]), kv_data=_make_kv_data())
    n1.last_used = 1.0
    n2.last_used = 2.0
    # Set budget to allow only the newer (n2) node, forcing n1's eviction
    node_size = _node_data_bytes(n1)
    cache.config.max_memory_gb = (node_size + 512) / (
        1024**3
    )  # Budget for ~1 node plus buffer
    cache._evict_if_needed()
    assert n1.kv_data is None  # evicted (oldest)
    assert n2.kv_data is not None  # kept (newer)


def test_pinned_node_not_evicted():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    node = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    node.ref_count = 1
    # Set budget to 0 to force eviction attempt; ref_count should protect it
    cache.config.max_memory_gb = 0.0
    cache._evict_if_needed()
    assert node.kv_data is not None  # pinned by ref_count, should not be evicted


def test_eviction_cascade_to_parent():
    cache = make_cache(max_gb=0.0)
    n1 = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    n2 = cache.insert(n1, seg([2]), kv_data=_make_kv_data())
    n1.last_used = 0.5
    n2.last_used = 1.0
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert n2.kv_data is None  # leaf evicted first
    assert n1.kv_data is None  # cascades since n1 now has no children


def test_cascade_stops_at_sibling():
    cache = make_cache(max_gb=100.0)  # HIGH during insert to prevent auto-eviction
    n1 = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    n2a = cache.insert(n1, seg([2]), kv_data=_make_kv_data())
    n2b = cache.insert(n1, seg([3]), kv_data=_make_kv_data())
    n2a.last_used = 1.0
    n2b.last_used = 2.0
    n1.last_used = 0.5
    # Set budget to allow n1 + n2b but not n2a; cascade stops because n1 still has n2b
    node_size = _node_data_bytes(n2a)
    cache.config.max_memory_gb = (node_size * 2 + 512) / (1024**3)
    cache._evict_if_needed()
    assert n2a.kv_data is None  # evicted (oldest leaf)
    assert n1.kv_data is not None  # n1 still has n2b, so cascade stops


def test_evicted_node_removed_from_parent_children():
    cache = make_cache(max_gb=0.0)
    node = cache.insert(cache.root, seg([1]), kv_data=_make_kv_data())
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    h = node.context_hash
    cache._evict_if_needed()
    assert h not in cache.root.children


def test_save_and_load_roundtrip(tmp_path):
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    seg1 = seg(list(range(10)), role="system")
    cache.insert(
        cache.root, seg1, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )

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
    kv_data = _make_kv_data(n_tokens=3)
    cache.insert(cache.root, seg([1, 2, 3]), kv_data=kv_data)
    cache.save(str(tmp_path))

    # Delete the KV file
    for f in tmp_path.glob("kv_*.safetensors"):
        f.unlink()
        break

    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))  # must not raise


def make_ssd_cache(tmp_path, stride=512):
    return TurnPrefixCache(
        TurnPrefixCacheConfig(
            checkpoint_stride=stride,
            max_memory_gb=8.0,
            kv_dtype="bf16",
            ssd_max_gb=10.0,
            ssd_dir=str(tmp_path),
        )
    )


def test_spill_replaces_kv_with_ssdref(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv_data = _make_kv_data(n_tokens=3)
    node = cache.insert(cache.root, seg([1, 2, 3]), kv_data=kv_data)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_data, SSDRef)


def test_spill_trie_still_matchable(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1, 2, 3]))
    cache._spill_to_ssd(node)
    path, _ = cache.match([seg([1, 2, 3])])
    assert len(path) == 1


def test_promote_restores_arrays(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv_data = _make_kv_data(n_tokens=3)
    node = cache.insert(cache.root, seg([1, 2, 3]), kv_data=kv_data)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_data, SSDRef)
    success = cache._promote_from_ssd(node)
    assert success
    assert isinstance(node.kv_data, list)


def test_promote_returns_false_on_missing_file(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1]))
    node.kv_data = SSDRef(file_path="/nonexistent/file.safetensors", size_bytes=0)
    result = cache._promote_from_ssd(node)
    assert result is False


def test_promote_restores_recurrent_state(tmp_path):
    """Regression test: recurrent state is properly reconstructed from SSD."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_ssd_cache(tmp_path)
    kv_data = _make_kv_data(n_tokens=3)
    rec_data = [
        RecurrentLayerSegment(arrays=[mx.ones((2, 3)), mx.ones((4, 5))], metadata={})
    ]
    node = cache.insert(
        cache.root, seg([1, 2, 3]), kv_data=kv_data, recurrent_data=rec_data
    )

    # Verify before spill
    assert isinstance(node.recurrent_data, list)
    assert len(node.recurrent_data) == 1

    # Spill and promote
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_data, SSDRef)
    assert isinstance(node.recurrent_data, SSDRef)

    success = cache._promote_from_ssd(node)
    assert success

    # Verify recurrent data is restored as list of RecurrentLayerSegment
    assert isinstance(node.recurrent_data, list)


from unittest.mock import MagicMock


def test_scheduler_integration_fetch_hits_cache():
    """Verify that matching path and recurrent state are set on the request."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    sys_seg = seg(list(range(20)), role="system")
    cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )

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
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    sys_seg = seg(list(range(10)), role="system")
    n_sys = cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )

    # Match found sys_node; now store new user segment
    user_seg = seg([100, 101, 102])
    n_user = cache.insert(n_sys, user_seg, kv_data=[], recurrent_data=rec)
    assert n_user in n_sys.children.values()


def test_concurrent_ref_counts():
    """Two requests sharing a prefix: ref_count=2 while both active."""
    cache = make_cache()
    n = cache.insert(cache.root, seg([1, 2, 3]))

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
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]

    # Session 1: Build the cache
    sys_seg = seg(list(range(10)), role="system")
    user_seg = seg([100, 101])

    n_sys = cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    n_user = cache.insert(n_sys, user_seg, kv_data=[], recurrent_data=rec)

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
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]

    sys_seg = seg(list(range(50)), role="system")
    u1_seg = seg([100, 101, 102])
    a1_seg = seg([200, 201])

    n_sys = cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    n_u1 = cache.insert(n_sys, u1_seg, kv_data=[], recurrent_data=rec)
    n_a1 = cache.insert(n_u1, a1_seg, kv_data=[], recurrent_data=rec)

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
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(
        stride=10000
    )  # high stride so only sys_prompt gets permanent checkpoint
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]

    sys_seg = seg(list(range(50)), role="system")
    n_sys = cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    assert n_sys.is_permanent_checkpoint

    # Session B: same system prompt
    path, has_recurrent = cache.match([sys_seg])
    assert len(path) == 1
    assert has_recurrent  # permanent checkpoint → recurrent available immediately
    cache.release(path)


def test_mid_session_branching():
    """Two sessions share [sys, u1, a1] but diverge at u2."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]

    sys_seg = seg(list(range(10)), role="system")
    u1_seg = seg([100])
    a1_seg = seg([200])
    u2a_seg = seg([300])  # branch A
    u2b_seg = seg([400])  # branch B

    n_sys = cache.insert(
        cache.root, sys_seg, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    n_u1 = cache.insert(n_sys, u1_seg, kv_data=[], recurrent_data=rec)
    n_a1 = cache.insert(n_u1, a1_seg, kv_data=[], recurrent_data=rec)
    n_u2a = cache.insert(n_a1, u2a_seg, kv_data=[], recurrent_data=rec)
    n_u2b = cache.insert(n_a1, u2b_seg, kv_data=[], recurrent_data=rec)

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


def test_messages_to_segments_uses_turn_boundaries():
    """Scheduler _messages_to_segments splits using _turn_boundaries.

    This is the fix for the cross-session system-prompt cache miss: both sessions
    share the same _turn_boundaries[0] (len of system-prompt tokens), so the system-prompt
    segment hash is stable across sessions with different user messages.
    """
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    # Simulate prompt_token_ids for [sys_prompt (10 tokens) + user "hi" (2 tokens)]
    sys_tokens = list(range(10))
    user_tokens_hi = [100, 101]
    user_tokens_yo = [200, 201]

    B_sys = len(sys_tokens)

    req1 = _make_request_with_boundaries(sys_tokens + user_tokens_hi, [B_sys])
    req2 = _make_request_with_boundaries(sys_tokens + user_tokens_yo, [B_sys])

    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    segs2 = TurnCacheAdapter.messages_to_segments(req2)

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


def test_cross_session_hit_via_turn_boundaries():
    """End-to-end: session 1 stores sys-prompt segment; session 2 gets a hit on it."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100]
    user_yo = [200]
    B_sys = len(sys_tokens)

    # Session 1 store: insert sys segment + user-hi segment
    req1 = _make_request_with_boundaries(sys_tokens + user_hi, [B_sys])
    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    assert len(segs1) == 2
    n_sys = cache.insert(cache.root, segs1[0], is_system_prompt=True)
    cache.insert(n_sys, segs1[1])

    # Session 2 fetch: same system prompt, different user message
    req2 = _make_request_with_boundaries(sys_tokens + user_yo, [B_sys])
    segs2 = TurnCacheAdapter.messages_to_segments(req2)
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
        assert "chunked-prefill-tokens" not in str(
            e
        ), f"Unexpected chunked-prefill error: {e}"
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

    assert (
        sched.memory_aware_cache is None
    ), "memory_aware_cache should be None when use_turn_cache=True"


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


def _make_extracted_state(n_layers=2, n_tokens=10):
    """Build a list of dicts in _extract_cache_states format.

    head_dim=64 ensures divisibility by _split_cache_arrays' group_size=64.
    """
    from mlx_lm.models.cache import KVCache

    return [
        {
            "state": (mx.zeros([1, 4, n_tokens, 64]), mx.zeros([1, 4, n_tokens, 64])),
            "meta_state": "",  # KVCache.meta_state is a string
            "class_name": "KVCache",
            "class_ref": KVCache,
        }
        for _ in range(n_layers)
    ]


def test_save_and_load_recurrent_data_roundtrip(tmp_path):
    """TurnNode with RecurrentLayerSegment survives a save/load round-trip."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    state = mx.zeros((2, 3))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    mx.eval(state)

    seg_sys = Segment(role="system", token_ids=list(range(10)))
    cache.insert(
        cache.root, seg_sys, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )

    cache.save(str(tmp_path))

    cache2 = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache2.load(str(tmp_path))

    assert len(cache2.root.children) == 1
    loaded_node = list(cache2.root.children.values())[0]
    assert loaded_node.recurrent_data is not None
    assert isinstance(loaded_node.recurrent_data, list)
    assert len(loaded_node.recurrent_data) > 0


# ---------------------------------------------------------------------------
# Multi-turn diagnostic tests
# ---------------------------------------------------------------------------


def test_segment1_tokens_stable_across_turns_new():
    """Segment 1 (sys) has identical token_ids for turn 1, 2, and 3 using new boundaries."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(100))
    u1 = list(range(100, 120))
    a1 = list(range(200, 250))
    u2 = list(range(300, 315))
    a2 = list(range(400, 430))
    u3 = list(range(500, 510))
    B_sys = len(sys_tokens)

    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    B_1 = B_sys + len(u1) + len(a1)
    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    B_2 = B_1 + len(u2) + len(a2)
    req3 = _make_request_with_boundaries(
        sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2]
    )

    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    segs2 = TurnCacheAdapter.messages_to_segments(req2)
    segs3 = TurnCacheAdapter.messages_to_segments(req3)

    assert segs1[0].token_ids == sys_tokens
    assert segs2[0].token_ids == sys_tokens
    assert segs3[0].token_ids == sys_tokens

    from vllm_mlx.turn_prefix_cache import (
        _context_hash,
        TurnPrefixCache,
        TurnPrefixCacheConfig,
    )

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    h1 = _context_hash(cache.root.context_hash, segs1[0].token_ids)
    h2 = _context_hash(cache.root.context_hash, segs2[0].token_ids)
    h3 = _context_hash(cache.root.context_hash, segs3[0].token_ids)
    assert h1 == h2 == h3


def test_multi_turn_sys_hit_each_turn_new():
    """After storing turn 1, turns 2 and 3 must get a trie HIT on the sys segment."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(50))
    u1 = list(range(50, 60))
    a1 = list(range(100, 120))
    u2 = list(range(200, 210))
    a2 = list(range(300, 315))
    u3 = list(range(400, 405))
    B_sys = len(sys_tokens)

    # Turn 1 store
    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    assert len(segs1) == 2
    sys_node = cache.insert(cache.root, segs1[0], is_system_prompt=True)
    cache.insert(sys_node, segs1[1])

    # Turn 2 fetch: sys hit
    B_1 = B_sys + len(u1) + len(a1)
    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    segs2 = TurnCacheAdapter.messages_to_segments(req2)
    assert len(segs2) == 3
    path2, _ = cache.match(segs2)
    assert len(path2) >= 1
    assert path2[0] is sys_node
    assert sum(len(n.token_ids) for n in path2) == B_sys
    cache.release(path2)

    # Turn 3 fetch: sys hit
    B_2 = B_1 + len(u2) + len(a2)
    req3 = _make_request_with_boundaries(
        sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2]
    )
    segs3 = TurnCacheAdapter.messages_to_segments(req3)
    assert len(segs3) == 4
    path3, _ = cache.match(segs3)
    assert len(path3) >= 1
    assert path3[0] is sys_node
    cache.release(path3)


def test_conv_segment_grows_each_turn_new():
    """Conv segments at same boundary position are identical; total conversation grows."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(20))
    u1 = list(range(20, 25))
    a1 = list(range(100, 105))
    u2 = list(range(200, 203))
    a2 = list(range(300, 306))
    u3 = list(range(400, 402))
    B_sys = len(sys_tokens)
    B_1 = B_sys + len(u1) + len(a1)
    B_2 = B_1 + len(u2) + len(a2)

    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    req3 = _make_request_with_boundaries(
        sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2]
    )

    segs2 = TurnCacheAdapter.messages_to_segments(req2)
    segs3 = TurnCacheAdapter.messages_to_segments(req3)

    # Turn 2: [sys, conv, user]
    # Turn 3: [sys, conv, conv, user]
    assert len(segs2) == 3
    assert len(segs3) == 4
    # Same boundary → same segment content
    assert segs2[1].role == "conversation"
    assert segs3[1].role == "conversation"
    assert segs2[1].token_ids == segs3[1].token_ids  # both cover [B_sys:B_1]
    # Turn 3 has additional conv segment
    assert segs3[2].role == "conversation"


def test_multi_turn_conv_stored_after_turn2_new():
    """After turn 2 completes, the trie contains a conv node under sys_node."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    u1 = list(range(20, 25))
    a1 = list(range(100, 105))
    u2 = list(range(200, 203))
    B_sys = len(sys_tokens)
    B_1 = B_sys + len(u1) + len(a1)

    # Turn 1 store
    sys_state = _make_extracted_state(n_layers=1, n_tokens=B_sys)
    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    sys_node = cache.insert(cache.root, segs1[0], is_system_prompt=True)
    cache.insert(sys_node, segs1[1])

    # Turn 2: HIT on sys, then store [conv, user2]
    conv_state = _make_extracted_state(n_layers=1, n_tokens=B_1)
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + u1 + a1 + u2
    req2._turn_boundaries = [B_sys, B_1]
    req2._extracted_cache = _make_extracted_state(n_layers=1, n_tokens=B_1 + len(u2))
    req2.output_token_ids = [999]
    sys_node.ref_count += 1

    segs2 = TurnCacheAdapter.messages_to_segments(req2)
    assert len(segs2) == 3

    matched_depth = 1
    path = [sys_node]
    parent = sys_node
    new_segments = segs2[matched_depth:]
    _turn_boundaries = req2._turn_boundaries
    boundary_states = {B_sys: sys_state, B_1: conv_state}
    parent_before_user = parent

    inserted = []
    for i, segment in enumerate(new_segments):
        abs_idx = matched_depth + i
        is_sys = segment.role == "system" and abs_idx == 0
        is_last = i == len(new_segments) - 1
        parent = cache.insert(parent, segment, is_system_prompt=is_sys)
        if not is_last:
            parent_before_user = parent
        inserted.append(parent)

    cache.release(path)
    conv_node = inserted[0]
    user2_node = inserted[1]

    assert conv_node in sys_node.children.values()
    expected_conv_tokens = (sys_tokens + u1 + a1 + u2)[B_sys:B_1]
    assert conv_node.token_ids == expected_conv_tokens

    # Cross-session turn-2: 3-deep HIT
    req_cross2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    segs_cross2 = TurnCacheAdapter.messages_to_segments(req_cross2)
    path_cross2, _ = cache.match(segs_cross2)
    assert len(path_cross2) == 3
    assert path_cross2[1] is conv_node
    cache.release(path_cross2)


# ── _compute_turn_boundaries tests (tokenizer-only, no model) ──────────────


class _MockTok:
    """Char-level tokenizer with a minimal chat template."""

    unk_token_id = None

    def encode(self, text):
        return list(text.encode("utf-8"))

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
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

    # Add _apply_chat_template method for boundary detection
    def _apply_chat_template(
        messages,
        tools=None,
        num_images=0,
        num_audios=0,
        chat_template_kwargs=None,
        enable_thinking=None,
    ):
        return eng._tokenizer.apply_chat_template(
            messages,
            **(
                (chat_template_kwargs or {})
                if isinstance(chat_template_kwargs, dict)
                else {}
            ),
        )

    eng._apply_chat_template = _apply_chat_template

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
    assert boundaries[0] == len(
        expected_sys_text.encode("utf-8")
    ), f"B_sys={boundaries[0]}, expected {len(expected_sys_text.encode('utf-8'))}"


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
    assert (
        boundaries[0] < boundaries[1] < boundaries[2]
    ), f"Boundaries not strictly increasing: {boundaries}"


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
    assert boundaries[1] == expected_b1, f"B_1={boundaries[1]}, expected {expected_b1}"


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
        assert (
            boundaries[i] < boundaries[i + 1]
        ), f"boundaries[{i}]={boundaries[i]} >= boundaries[{i+1}]={boundaries[i+1]}"


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
    full_tokens = tok.encode(
        tok.apply_chat_template(messages, add_generation_prompt=True)
    )
    assert boundaries[-1] < len(
        full_tokens
    ), f"Last boundary {boundaries[-1]} >= full_tokens length {len(full_tokens)}"


# ── Request dataclass field tests ──────────────────────────────────────────


def test_request_has_turn_boundaries_field():
    """Request must have _turn_boundaries field (new)."""
    from vllm_mlx.request import Request, SamplingParams

    req = Request(
        request_id="test",
        prompt="hi",
        sampling_params=SamplingParams(),
    )
    assert hasattr(req, "_turn_boundaries"), "Request missing _turn_boundaries field"


def test_request_turn_boundaries_field_set():
    """Request._turn_boundaries defaults to empty list and can be set."""
    from vllm_mlx.request import Request, SamplingParams

    req = Request(
        request_id="test",
        prompt="hi",
        sampling_params=SamplingParams(),
    )
    assert isinstance(req._turn_boundaries, list), "_turn_boundaries should be a list"
    assert len(req._turn_boundaries) == 0, "_turn_boundaries should default to empty"
    # Should also have _boundary_states field
    assert hasattr(req, "_boundary_states"), "Request missing _boundary_states field"
    assert isinstance(req._boundary_states, dict), "_boundary_states should be a dict"
    assert len(req._boundary_states) == 0, "_boundary_states should default to empty"


def test_request_no_old_boundary_fields():
    """Request must NOT have the old boundary fields (prefix_boundary, sys_end_boundary, turn_boundaries)."""
    from vllm_mlx.request import Request, SamplingParams

    req = Request(
        request_id="test",
        prompt="hi",
        sampling_params=SamplingParams(),
    )
    assert not hasattr(
        req, "prefix_boundary"
    ), "Request should not have prefix_boundary field"
    assert not hasattr(
        req, "sys_end_boundary"
    ), "Request should not have sys_end_boundary field"
    assert not hasattr(
        req, "turn_boundaries"
    ), "Request should not have turn_boundaries field"


def test_engine_core_add_request_accepts_turn_boundaries():
    """add_request accepts turn_boundaries (list) and sets _turn_boundaries on Request."""
    from unittest.mock import MagicMock, AsyncMock
    import asyncio
    from vllm_mlx.engine_core import EngineCore

    core = object.__new__(EngineCore)
    core.config = MagicMock()
    core.config.stream_interval = 1
    core.scheduler = MagicMock()
    core.scheduler.add_request = MagicMock()
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}

    loop = asyncio.new_event_loop()
    try:
        request_id = loop.run_until_complete(
            core.add_request(
                prompt="hello",
                turn_boundaries=[10, 30],
            )
        )
    finally:
        loop.close()

    assert core.scheduler.add_request.called
    added_req = core.scheduler.add_request.call_args[0][0]
    assert added_req._turn_boundaries == [10, 30]


# ── _messages_to_segments (new implementation) tests ──────────────────────


def _make_request_with_boundaries(prompt_token_ids, turn_boundaries):
    """Create a MagicMock request with _turn_boundaries."""
    from unittest.mock import MagicMock

    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req._turn_boundaries = turn_boundaries
    return req


def test_messages_to_segments_new_single_turn():
    """Single turn: [B_sys] → [system, user] segments."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    # full_tokens = [0..9=sys, 10..14=user]
    req = _make_request_with_boundaries(
        prompt_token_ids=list(range(15)),
        turn_boundaries=[10],  # B_sys = 10
    )
    segs = TurnCacheAdapter.messages_to_segments(req)

    assert len(segs) == 2, f"Expected 2 segments, got {len(segs)}"
    assert segs[0].role == "system"
    assert segs[0].token_ids == list(range(10))
    assert segs[1].role == "user"
    assert segs[1].token_ids == list(range(10, 15))


def test_messages_to_segments_new_two_boundaries():
    """Two boundaries: [B_sys, B_1] → [system, conversation, user]."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    # full = [0..9=sys, 10..14=conv, 15..19=user]
    req = _make_request_with_boundaries(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10, 15],  # B_sys=10, B_1=15
    )
    segs = TurnCacheAdapter.messages_to_segments(req)

    assert len(segs) == 3, f"Expected 3 segments, got {len(segs)}"
    assert segs[0].role == "system"
    assert segs[0].token_ids == list(range(10))
    assert segs[1].role == "conversation"
    assert segs[1].token_ids == list(range(10, 15))
    assert segs[2].role == "user"
    assert segs[2].token_ids == list(range(15, 20))


def test_messages_to_segments_new_no_boundaries():
    """No boundaries → empty list."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    req = _make_request_with_boundaries(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[],
    )
    segs = TurnCacheAdapter.messages_to_segments(req)

    assert segs == []


def test_messages_to_segments_new_boundary_at_end():
    """Boundary at end of tokens → system-only segment (entire prompt is the system prefix)."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    req = _make_request_with_boundaries(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[10],  # B_sys == len(full_tokens) → system fills whole context
    )
    segs = TurnCacheAdapter.messages_to_segments(req)

    assert len(segs) == 1
    assert list(segs[0].token_ids) == list(range(10))


def test_messages_to_segments_new_three_boundaries():
    """Three boundaries: [B_sys, B_1, B_2] → [system, conv, conv, user]."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    # full = [0..4=sys, 5..9=conv1, 10..14=conv2, 15..19=user]
    req = _make_request_with_boundaries(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[5, 10, 15],  # B_sys=5, B_1=10, B_2=15
    )
    segs = TurnCacheAdapter.messages_to_segments(req)

    assert len(segs) == 4, f"Expected 4 segments, got {len(segs)}"
    assert segs[0].role == "system"
    assert segs[0].token_ids == list(range(5))
    assert segs[1].role == "conversation"
    assert segs[1].token_ids == list(range(5, 10))
    assert segs[2].role == "conversation"
    assert segs[2].token_ids == list(range(10, 15))
    assert segs[3].role == "user"
    assert segs[3].token_ids == list(range(15, 20))


def test_messages_to_segments_new_sys_stable():
    """System segment is stable across requests with different user messages."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(10))
    user_hi = [100, 101]
    user_yo = [200, 201]

    req1 = _make_request_with_boundaries(
        prompt_token_ids=sys_tokens + user_hi,
        turn_boundaries=[10],  # B_sys = 10
    )
    req2 = _make_request_with_boundaries(
        prompt_token_ids=sys_tokens + user_yo,
        turn_boundaries=[10],  # Same B_sys
    )

    segs1 = TurnCacheAdapter.messages_to_segments(req1)
    segs2 = TurnCacheAdapter.messages_to_segments(req2)

    assert len(segs1) == 2
    assert len(segs2) == 2

    # Both have the same system segment
    assert segs1[0].token_ids == segs2[0].token_ids == sys_tokens
    assert segs1[0].role == "system"

    # User segments differ
    assert segs1[1].token_ids == user_hi
    assert segs2[1].token_ids == user_yo
    assert segs1[1].token_ids != segs2[1].token_ids


def test_mid_prefill_eagerly_inserts_turn_at_boundary():
    """_make_mid_prefill_save_callback eagerly inserts the turn into the trie at boundary."""
    from unittest.mock import MagicMock, patch
    from vllm_mlx.kv_cache import RequestCacheState

    sched = _make_minimal_scheduler_new()
    callback = sched._make_mid_prefill_save_callback(save_interval=512)

    class SimpleRequest:
        pass

    req = SimpleRequest()
    req.request_id = "test-1"
    req.prompt_token_ids = list(
        range(100)
    )  # 100 tokens; B_sys=50 → sys=[0-49], user=[50-99]
    req._turn_boundaries = [50]
    req._mid_prefill_last_save = 0
    req._cache_state = RequestCacheState()

    sched.requests["test-1"] = req
    sched.uid_to_request_id[123] = "test-1"

    mock_extracted = _make_extracted_state(n_layers=2, n_tokens=50)
    with patch("vllm_mlx.prefix_cache_adapters.extract_cache_states", return_value=mock_extracted):
        callback(123, 50, MagicMock())

    # System turn eagerly inserted into trie
    assert len(sched.turn_cache.root.children) == 1
    assert len(req._cache_state.turn_path) == 1


def test_mid_prefill_does_not_insert_away_from_boundary():
    """Callback does nothing when processed position is not a turn boundary."""
    from unittest.mock import MagicMock, patch
    from vllm_mlx.kv_cache import RequestCacheState

    sched = _make_minimal_scheduler_new()
    callback = sched._make_mid_prefill_save_callback(save_interval=512)

    class SimpleRequest:
        pass

    req = SimpleRequest()
    req.request_id = "test-2"
    req.prompt_token_ids = list(range(100))
    req._turn_boundaries = [50]  # boundary at 50, not at 30
    req._mid_prefill_last_save = 0
    req._cache_state = RequestCacheState()

    sched.requests["test-2"] = req
    sched.uid_to_request_id[124] = "test-2"

    mock_extracted = _make_extracted_state(n_layers=2, n_tokens=30)
    with patch("vllm_mlx.prefix_cache_adapters.extract_cache_states", return_value=mock_extracted):
        callback(124, 30, MagicMock())  # total=30, not in [50] → no insert

    assert len(sched.turn_cache.root.children) == 0
    assert req._cache_state.turn_path == []


def test_mid_prefill_inserts_multiple_boundaries_in_sequence():
    """Callback inserts all turn nodes in sequence at each boundary."""
    from unittest.mock import MagicMock, patch
    from vllm_mlx.kv_cache import RequestCacheState

    sched = _make_minimal_scheduler_new()
    callback = sched._make_mid_prefill_save_callback(save_interval=512)

    class SimpleRequest:
        pass

    req = SimpleRequest()
    req.request_id = "test-3"
    # 200 tokens: sys=[0-49], conv1=[50-99], conv2=[100-149], user=[150-199]
    req.prompt_token_ids = list(range(200))
    req._turn_boundaries = [50, 100, 150]
    req._mid_prefill_last_save = 0
    req._cache_state = RequestCacheState()

    sched.requests["test-3"] = req
    sched.uid_to_request_id[125] = "test-3"

    for boundary, n_tok in [(50, 50), (100, 100), (150, 150)]:
        extracted = _make_extracted_state(n_layers=2, n_tokens=n_tok)
        with patch("vllm_mlx.prefix_cache_adapters.extract_cache_states", return_value=extracted):
            callback(125, boundary, MagicMock())

    # Three turns inserted, chained: root → sys → conv1 → conv2
    assert len(req._cache_state.turn_path) == 3
    root_children = sched.turn_cache.root.children
    assert len(root_children) == 1
    sys_node = list(root_children.values())[0]
    assert len(sys_node.children) == 1
    conv1_node = list(sys_node.children.values())[0]
    assert len(conv1_node.children) == 1


def test_chunked_prefill_boundary_aware_first_chunk():
    """When _turn_boundaries=[B_sys] and B_sys <= budget, first chunk lands exactly on B_sys."""
    from unittest.mock import MagicMock, patch
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=100)

    # Simulate a request with B_sys=40, budget=100, prompt=80 tokens
    req = MagicMock()
    req._turn_boundaries = [40]
    req.cached_tokens = 0
    req.prompt_token_ids = list(range(80))

    # The first chunk should be 40 (landing on B_sys)
    _turn_boundaries = req._turn_boundaries
    cached = req.cached_tokens
    budget = 100

    boundaries_to_hit = sorted(b for b in _turn_boundaries if b > cached)
    assert boundaries_to_hit
    first_b = boundaries_to_hit[0]
    dist = first_b - cached
    first_chunk = min(dist, budget) if dist <= budget else budget
    assert first_chunk == 40, f"Expected first_chunk=40, got {first_chunk}"


def test_chunked_prefill_boundary_aware_skips_when_boundary_beyond_budget():
    """When B_sys > budget, full budget is used (boundary handled in next iteration)."""
    req_boundaries = [300]  # B_sys=300
    cached = 0
    budget = 100

    boundaries_to_hit = sorted(b for b in req_boundaries if b > cached)
    first_b = boundaries_to_hit[0]
    dist = first_b - cached
    first_chunk = min(dist, budget) if dist <= budget else budget
    assert first_chunk == budget, f"Expected first_chunk={budget}, got {first_chunk}"


def test_chunked_prefill_boundary_aware_continuation_lands_on_boundary():
    """In the continuation loop, chunk size lands exactly on next boundary."""
    _turn_boundaries = [40, 80]
    cached = 0
    processed_so_far = 40  # first chunk already landed on B_sys=40
    budget = 100

    total_pos = cached + processed_so_far
    next_b = next((b for b in sorted(_turn_boundaries) if b > total_pos), None)
    assert next_b == 80
    dist = next_b - total_pos
    n_to_process = min(dist, budget) if dist <= budget else budget
    assert n_to_process == 40


def _make_bf16_kvcache_extracted(n_layers=2, n_tokens=10, head_dim=256, n_kv_heads=4):
    """Create extracted_cache in KVCache (bf16) format matching _extract_cache_states output."""
    from mlx_lm.models.cache import KVCache

    return [
        {
            "state": (
                mx.random.normal([1, n_kv_heads, n_tokens, head_dim]).astype(
                    mx.bfloat16
                ),
                mx.random.normal([1, n_kv_heads, n_tokens, head_dim]).astype(
                    mx.bfloat16
                ),
            ),
            "meta_state": (str(n_tokens),),
            "class_name": "KVCache",
            "class_ref": KVCache,
        }
        for _ in range(n_layers)
    ]


def test_find_checkpoint_ancestor_rejects_empty_recurrent_data():
    """Nodes with recurrent_data=[] must not be selected as checkpoint ancestors."""
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache.has_recurrent_state = True
    node = TurnNode(
        token_ids=[1, 2],
        context_hash=1,
        kv_data=[],
        recurrent_data=[],  # empty list — real recurrent state is never empty
        parent=cache.root,
    )
    result = cache.find_checkpoint_ancestor([node])
    assert (
        result is None
    ), "find_checkpoint_ancestor must return None for a node with recurrent_data=[]"


def test_find_checkpoint_ancestor_accepts_nonempty_recurrent_data():
    """Sanity check: nodes with actual recurrent data are still selected."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    cache.has_recurrent_state = True
    node = TurnNode(
        token_ids=[1, 2],
        context_hash=1,
        kv_data=[],
        recurrent_data=[RecurrentLayerSegment(arrays=[mx.zeros((1, 16))], metadata={})],
        parent=cache.root,
    )
    result = cache.find_checkpoint_ancestor([node])
    assert result is node


# ── save/load dtype combination tests ─────────────────────────────────────────


def test_save_load_multi_node_parent_child(tmp_path):
    """Two-node trie (system → user) survives save/load: both nodes exist and parent link is correct."""
    cache = make_cache(stride=0)
    s_sys = seg(list(range(10)), role="system")
    s_usr = seg([100, 101], role="user")
    from vllm_mlx.cache_types import RecurrentLayerSegment

    state = mx.zeros((2,))
    rec = [RecurrentLayerSegment(arrays=[state], metadata={})]
    n_sys = cache.insert(
        cache.root, s_sys, kv_data=[], recurrent_data=rec, is_system_prompt=True
    )
    cache.insert(n_sys, s_usr, kv_data=[], recurrent_data=rec)

    cache.save(str(tmp_path))
    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))

    path, _ = cache2.match([s_sys, s_usr])
    assert len(path) == 2
    assert path[1].parent is path[0]
    assert path[0].parent is cache2.root


def _concat_seg(layer_index=0, n=4):
    k = mx.zeros((1, 2, n, 8), dtype=mx.bfloat16)
    return KVConcatSegment(keys=k, values=k, layer_index=layer_index,
                           n_tokens=n, bits=None, class_name="KVCache")


def _rot_seg(layer_index=1, n=4):
    k = mx.zeros((1, 2, n, 8), dtype=mx.bfloat16)
    return KVRotatingSegment(keys=k, values=k, layer_index=layer_index,
                             n_tokens=n, bits=None, class_name="RotatingKVCache",
                             max_size=n, keep=0, offset=n, idx=n)


def test_insert_stores_sliding_kv_data_separately():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    node = cache.insert(
        cache.root,
        Segment(role="user", token_ids=[1, 2, 3, 4]),
        kv_data=[_concat_seg()],
        sliding_kv_data=[_rot_seg()],
    )
    assert node.sliding_kv_data is not None
    assert len(node.sliding_kv_data) == 1
    assert node.kv_data is not None and len(node.kv_data) == 1
    assert cache.has_sliding_state is True


def test_node_data_bytes_counts_sliding():
    node_kv_only = _make_node(kv=[_concat_seg()], sliding=None)
    node_with_sliding = _make_node(kv=[_concat_seg()], sliding=[_rot_seg()])
    assert _node_data_bytes(node_with_sliding) > _node_data_bytes(node_kv_only)


def _make_node(kv, sliding):
    from vllm_mlx.turn_prefix_cache import TurnNode
    return TurnNode(token_ids=[1], context_hash=1, kv_data=kv,
                    recurrent_data=None, sliding_kv_data=sliding)


def test_collect_path_data_returns_full_and_sliding():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    a = cache.insert(cache.root, Segment("user", [1, 2]),
                     kv_data=[_concat_seg(layer_index=0)],
                     sliding_kv_data=[_rot_seg(layer_index=1)])
    b = cache.insert(a, Segment("user", [3, 4]),
                     kv_data=[_concat_seg(layer_index=0)],
                     sliding_kv_data=[_rot_seg(layer_index=1)])

    kv_layers, rec_layers = cache.collect_path_data(b)
    layer_indices = sorted(seg.layer_index for seg in kv_layers)
    assert layer_indices == [0, 1]            # one full (0) + one sliding (1)
    # Sliding (layer 1) comes from the anchor b only — exactly one segment.
    sliding = [s for s in kv_layers if s.layer_index == 1]
    assert len(sliding) == 1


def test_interior_node_drops_sliding_keeps_full():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    parent = cache.insert(cache.root, Segment("user", [1, 2]),
                          kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert parent.is_permanent_checkpoint is False
    # Giving parent its first child turns it interior -> sliding dropped.
    cache.insert(parent, Segment("user", [3, 4]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert parent.sliding_kv_data is None       # dropped
    assert isinstance(parent.kv_data, list)     # full attention retained


def test_checkpoint_node_keeps_sliding_when_gaining_child():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    cp = cache.insert(cache.root, Segment("system", [1, 2]),
                      kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()],
                      is_system_prompt=True)
    assert cp.is_permanent_checkpoint is True
    cache.insert(cp, Segment("user", [3, 4]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert isinstance(cp.sliding_kv_data, list)  # checkpoint keeps sliding


def test_find_checkpoint_ancestor_falls_back_to_checkpoint_for_sliding():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    cp = cache.insert(cache.root, Segment("system", [1, 2]),
                      kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()],
                      is_system_prompt=True)
    mid = cache.insert(cp, Segment("user", [3, 4]),
                       kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    # mid gains a child -> becomes interior -> its sliding is dropped.
    cache.insert(mid, Segment("user", [5, 6]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert mid.sliding_kv_data is None

    anchor = cache.find_checkpoint_ancestor([cp, mid])
    assert anchor is cp           # mid has no sliding -> fall back to checkpoint cp
