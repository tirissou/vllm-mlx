import pytest
import mlx.core as mx
from vllm_mlx.turn_prefix_cache import (
    Segment, TurnNode, SSDRef, TurnPrefixCacheConfig, _context_hash,
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
