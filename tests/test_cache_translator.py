"""Tests for KVLayerSegment.concat() and the new _segment/_assemble pipeline.

Replaces the old CacheTranslator tests now that linearize/quantize_kv are gone.
"""

import mlx.core as mx
import pytest

from vllm_mlx.cache_types import KVLayerSegment, KVQuantPolicy
from vllm_mlx.kv_cache import QuantizedArray
from vllm_mlx.prefix_cache_adapters import TurnCacheManager, _linearize

# ── _linearize ────────────────────────────────────────────────────────────────


def test_linearize_no_wrap():
    """offset == max_size: ring buffer is full, return as-is (sliced to offset)."""
    data = mx.array(
        [[[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]]]
    )
    result = _linearize(data, offset=10, max_size=10)
    assert result.shape == data.shape
    assert mx.array_equal(result, data)


def test_linearize_with_wrap():
    """offset < max_size: oldest data starts at offset."""
    data = mx.array(
        [[[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]]]
    )
    expected = mx.array(
        [[[[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [0.0], [1.0], [2.0], [3.0]]]]
    )
    result = _linearize(data, offset=4, max_size=10)
    assert mx.array_equal(result, expected)


# ── KVLayerSegment.concat ──────────────────────────────────────────────────────


def _make_kv_segment(
    n_tokens: int, layer_index: int = 0, fill: float = 1.0
) -> KVLayerSegment:
    """Helper: make a KVLayerSegment with n_tokens sequence length."""
    group_size = 64
    bits = 8
    keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * fill
    values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * fill
    q_keys = QuantizedArray(*mx.quantize(keys, group_size=group_size, bits=bits))
    q_values = QuantizedArray(*mx.quantize(values, group_size=group_size, bits=bits))
    return KVLayerSegment(
        keys=q_keys,
        values=q_values,
        metadata={
            "class_name": "KVCache",
            "layer_index": layer_index,
            "merge_strategy": "concatenate",
            "n_tokens": n_tokens,
            "bits": bits,
        },
    )


def test_concat_two_segments_sequence_axis():
    """KVLayerSegment.concat() joins along axis=-2 (sequence axis)."""
    seg1 = _make_kv_segment(n_tokens=3)
    seg2 = _make_kv_segment(n_tokens=5)
    merged = KVLayerSegment.concat([seg1, seg2])
    # packed dim is head_dim * bits // 32 = 64 * 8 // 32 = 16
    assert merged.keys.packed.shape[-2] == 8  # 3 + 5
    assert merged.values.packed.shape[-2] == 8


def test_concat_n_tokens_metadata_sum():
    """concat() updates n_tokens to sum of all segments."""
    seg1 = _make_kv_segment(n_tokens=4)
    seg2 = _make_kv_segment(n_tokens=6)
    merged = KVLayerSegment.concat([seg1, seg2])
    assert merged.metadata["n_tokens"] == 10


def test_concat_preserves_last_metadata():
    """Non-n_tokens metadata comes from the last segment."""
    seg1 = _make_kv_segment(n_tokens=2, layer_index=0)
    seg2 = _make_kv_segment(n_tokens=2, layer_index=0)
    seg2.metadata["class_name"] = "KVCacheVariant"
    merged = KVLayerSegment.concat([seg1, seg2])
    assert merged.metadata["class_name"] == "KVCacheVariant"


def test_concat_single_segment_passthrough():
    """Single-element concat returns a segment with the same shape."""
    seg = _make_kv_segment(n_tokens=7)
    merged = KVLayerSegment.concat([seg])
    assert merged.keys.packed.shape[-2] == seg.keys.packed.shape[-2]
    assert merged.metadata["n_tokens"] == 7


# ── _segment pipeline ─────────────────────────────────────────────────────────


def _make_kvcache_state(n_tokens: int, layer_index_hint: int = 0):
    """Build a KVCache live-state dict."""
    from mlx_lm.models.cache import KVCache

    return {
        "class_name": "KVCache",
        "state": (
            mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16),
            mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16),
        ),
        "meta_state": (n_tokens,),
        "class_ref": KVCache,
    }


def _make_rotating_state(n_tokens: int, max_size: int = 8, keep: int = 0):
    """Build a RotatingKVCache live-state dict (head_dim=64 for group_size compatibility)."""
    keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
    values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * 2
    offset = n_tokens % max_size if n_tokens <= max_size else max_size
    return {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (keep, max_size, offset, n_tokens),
    }


def test_segment_kvcache_produces_kv_layer_segment():
    """_segment on a KVCache state produces a KVLayerSegment."""
    states = [_make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = TurnCacheManager._segment(states, policy=policy, group_size=64)
    assert kv_list[0] is not None
    assert rec_list[0] is None
    assert isinstance(kv_list[0], KVLayerSegment)
    assert kv_list[0].metadata["merge_strategy"] == "concatenate"
    assert kv_list[0].metadata["n_tokens"] == 4


def test_segment_rotating_kvcache_produces_last_strategy():
    """_segment on a RotatingKVCache state produces merge_strategy='last'."""
    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = TurnCacheManager._segment(states, policy=policy, group_size=64)
    assert kv_list[0] is not None
    assert kv_list[0].metadata["merge_strategy"] == "last"


def test_segment_recurrent_produces_recurrent_layer_segment():
    """_segment on a non-KV state produces a RecurrentLayerSegment."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    states = [
        {"class_name": "MambaCache", "state": (mx.zeros((1, 4)),), "meta_state": ()}
    ]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = TurnCacheManager._segment(states, policy=policy, group_size=64)
    assert kv_list[0] is None
    assert rec_list[0] is not None
    assert isinstance(rec_list[0], RecurrentLayerSegment)
    assert rec_list[0].metadata["class_name"] == "MambaCache"


def test_segment_layer_index_matches_position():
    """layer_index in metadata matches position in input list."""
    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = TurnCacheManager._segment(states, policy=policy, group_size=64)
    assert kv_list[0].metadata["layer_index"] == 0
    assert kv_list[1].metadata["layer_index"] == 1


# ── _assemble pipeline ────────────────────────────────────────────────────────


def test_assemble_kvcache_returns_batch_quantized_kv_cache():
    """_assemble on a KVLayerSegment returns a BatchQuantizedKVCache (ADR-0005)."""
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    states = [_make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = TurnCacheManager._segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64)
    assert len(result) == 1
    assert isinstance(result[0], BatchQuantizedKVCache)


def test_assemble_kvcache_offset_matches_n_tokens():
    """Reconstructed cache _idx (logical length) equals original n_tokens."""
    # Reads ._idx directly; refactor to use a public attribute when one is exposed.
    n_tokens = 6
    states = [_make_kvcache_state(n_tokens=n_tokens)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = TurnCacheManager._segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64)
    assert result[0]._idx == n_tokens


def test_assemble_rotating_returns_rotating_kv_cache():
    """_assemble on a RotatingKVCache segment returns a RotatingKVCache."""
    from mlx_lm.models.cache import RotatingKVCache
    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = TurnCacheManager._segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64)
    assert len(result) == 1
    assert isinstance(result[0], RotatingKVCache)


def test_assemble_mixed_layer_ordering():
    """_assemble returns layers sorted by layer_index regardless of input order."""
    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = TurnCacheManager._segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    # Reverse to test sorting
    result = TurnCacheManager._assemble(
        list(reversed(kv_layers)), [], group_size=64
    )
    assert len(result) == 2


# ── round-trip: _segment → KVLayerSegment.concat → _assemble ─────────────────


def test_kvcache_round_trip_shape():
    """KVCache: segment, concat two nodes, assemble → correct sequence length."""
    states1 = [_make_kvcache_state(n_tokens=3)]
    states2 = [_make_kvcache_state(n_tokens=5)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)

    kv1, _ = TurnCacheManager._segment(states1, policy=policy, group_size=64)
    kv2, _ = TurnCacheManager._segment(states2, policy=policy, group_size=64)

    merged = KVLayerSegment.concat([kv1[0], kv2[0]])
    result = TurnCacheManager._assemble([merged], [], group_size=64)
    assert len(result) == 1
    # logical length should be 3 + 5 = 8
    # Reads ._idx directly; refactor to use a public attribute when one is exposed.
    assert result[0]._idx == 8


# ── KVLayerSegment.concat bits-agree invariant ───────────────────────────────


def _make_kv_segment_with_bits(n_tokens: int, bits: int | None) -> KVLayerSegment:
    if bits is None:
        keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
        values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
        return KVLayerSegment(
            keys=keys,
            values=values,
            metadata={
                "class_name": "KVCache",
                "layer_index": 0,
                "merge_strategy": "concatenate",
                "n_tokens": n_tokens,
                "bits": None,
            },
        )
    seg = _make_kv_segment(n_tokens=n_tokens)
    seg.metadata["bits"] = bits
    return seg


def test_concat_mismatched_bits_raises():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=8)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=4)
    with pytest.raises(AssertionError):
        KVLayerSegment.concat([a, b])


def test_concat_matching_bits_succeeds():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=8)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=8)
    merged = KVLayerSegment.concat([a, b])
    assert merged.metadata["bits"] == 8
    assert merged.metadata["n_tokens"] == 8


def test_concat_matching_float_bits_succeeds():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=None)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=None)
    merged = KVLayerSegment.concat([a, b])
    assert merged.metadata["bits"] is None


# ── _segment with KVQuantPolicy ──────────────────────────────────────────────


def _make_rotating_live_state(n_tokens: int, layer_index: int) -> dict:
    """Build a live RotatingKVCache state dict that exercises the float branch."""
    max_size = max(n_tokens, 16)
    keys = mx.ones((1, 1, max_size, 64), dtype=mx.bfloat16)
    values = mx.ones((1, 1, max_size, 64), dtype=mx.bfloat16)
    return {
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (0, max_size, n_tokens, n_tokens),  # keep, max_size, offset, _idx
    }


def _make_kvcache_live_state(n_tokens: int, layer_index: int) -> dict:
    keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
    values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
    return {
        "class_name": "KVCache",
        "state": (keys, values),
        "meta_state": (n_tokens,),
    }


def test_segment_mixed_policy_writes_bits_metadata():
    """Alternating KVCache/RotatingKVCache layers — sliding gets float, full gets quantized."""
    from vllm_mlx.kv_cache import QuantizedArray

    live = [
        _make_kvcache_live_state(n_tokens=128, layer_index=0),
        _make_rotating_live_state(n_tokens=64, layer_index=1),
        _make_kvcache_live_state(n_tokens=128, layer_index=2),
        _make_rotating_live_state(n_tokens=64, layer_index=3),
    ]
    policy = KVQuantPolicy(sliding_bits=None, full_bits=8)

    kv_list, rec_list = TurnCacheManager._segment(live, policy=policy, group_size=64)

    assert rec_list == [None, None, None, None]
    assert kv_list[0].metadata["bits"] == 8
    assert isinstance(kv_list[0].keys, QuantizedArray)
    assert kv_list[1].metadata["bits"] is None
    assert not isinstance(kv_list[1].keys, QuantizedArray)
    assert kv_list[2].metadata["bits"] == 8
    assert kv_list[3].metadata["bits"] is None


def test_segment_policy_none_disables_quantization():
    """policy=None: everything stored as float regardless of class."""
    from vllm_mlx.kv_cache import QuantizedArray

    live = [
        _make_kvcache_live_state(n_tokens=128, layer_index=0),
        _make_rotating_live_state(n_tokens=64, layer_index=1),
    ]

    kv_list, _ = TurnCacheManager._segment(live, policy=None, group_size=64)

    for seg in kv_list:
        assert seg.metadata["bits"] is None
        assert not isinstance(seg.keys, QuantizedArray)


# ── _assemble reads bits from metadata ───────────────────────────────────────


def test_assemble_reads_bits_from_metadata_mixed():
    """Build segments with mixed metadata['bits']; _assemble reconstructs correct cache types."""
    from mlx_lm.models.cache import KVCache as _KVCache, RotatingKVCache as _RotatingKVCache
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    from vllm_mlx.kv_cache import QuantizedArray

    # Full-attention quantized layer (layer 0)
    n = 128
    k = mx.ones((1, 4, n, 128), dtype=mx.bfloat16)
    v = mx.ones((1, 4, n, 128), dtype=mx.bfloat16)
    qk = QuantizedArray(*mx.quantize(k, group_size=64, bits=8))
    qv = QuantizedArray(*mx.quantize(v, group_size=64, bits=8))
    full_seg = KVLayerSegment(
        keys=qk,
        values=qv,
        metadata={
            "class_name": "KVCache",
            "layer_index": 0,
            "merge_strategy": "concatenate",
            "n_tokens": n,
            "bits": 8,
        },
    )

    # Sliding-window float layer (layer 1)
    max_size = 256
    rk = mx.ones((1, 4, max_size, 128), dtype=mx.bfloat16)
    rv = mx.ones((1, 4, max_size, 128), dtype=mx.bfloat16)
    sliding_seg = KVLayerSegment(
        keys=rk,
        values=rv,
        metadata={
            "class_name": "RotatingKVCache",
            "layer_index": 1,
            "merge_strategy": "last",
            "n_tokens": max_size,
            "max_size": max_size,
            "keep": 0,
            "offset": max_size,
            "_idx": max_size,
            "bits": None,
        },
    )

    caches = TurnCacheManager._assemble(
        kv_layers=[full_seg, sliding_seg],
        recurrent_layers=[],
        group_size=64,
    )

    assert isinstance(caches[0], BatchQuantizedKVCache)
    assert isinstance(caches[1], _RotatingKVCache)


# ── TurnCacheManager constructor ───────────────────────────────────────────────


def test_turncachemanager_holds_policy():
    """TurnCacheManager stores a KVQuantPolicy instance (not a raw kv_bits int)."""
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    inner = TurnPrefixCache(TurnPrefixCacheConfig())
    policy = KVQuantPolicy(sliding_bits=None, full_bits=8)
    mgr = TurnCacheManager(inner, policy=policy, kv_group_size=64)

    assert mgr._policy is policy
    assert mgr._kv_group_size == 64


def test_turncachemanager_accepts_none_policy():
    """policy=None means quantization disabled."""
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    inner = TurnPrefixCache(TurnPrefixCacheConfig())
    mgr = TurnCacheManager(inner, policy=None)
    assert mgr._policy is None
