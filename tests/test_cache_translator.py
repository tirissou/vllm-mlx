"""Tests for the cache_translator module: segment/assemble pipeline and KVConcatSegment.concat().

Replaces the old CacheTranslator tests now that linearize/quantize_kv are gone.
"""

import mlx.core as mx
import pytest

from vllm_mlx.cache_types import KVLayerSegment, KVConcatSegment, KVRotatingSegment, KVQuantPolicy
from vllm_mlx.kv_cache import QuantizedArray
from vllm_mlx.cache_translator import segment, assemble, _linearize

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


# ── KVConcatSegment.concat ──────────────────────────────────────────────────────


def _make_kv_segment(
    n_tokens: int, layer_index: int = 0, fill: float = 1.0
) -> KVConcatSegment:
    """Helper: make a KVConcatSegment with n_tokens sequence length."""
    group_size = 64
    bits = 8
    keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * fill
    values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16) * fill
    q_keys = QuantizedArray(*mx.quantize(keys, group_size=group_size, bits=bits))
    q_values = QuantizedArray(*mx.quantize(values, group_size=group_size, bits=bits))
    return KVConcatSegment(
        keys=q_keys,
        values=q_values,
        layer_index=layer_index,
        n_tokens=n_tokens,
        bits=bits,
        class_name="KVCache",
    )


def test_concat_two_segments_sequence_axis():
    """KVConcatSegment.concat() joins along axis=-2 (sequence axis)."""
    seg1 = _make_kv_segment(n_tokens=3)
    seg2 = _make_kv_segment(n_tokens=5)
    merged = KVConcatSegment.concat([seg1, seg2])
    # packed dim is head_dim * bits // 32 = 64 * 8 // 32 = 16
    assert merged.keys.packed.shape[-2] == 8  # 3 + 5
    assert merged.values.packed.shape[-2] == 8


def test_concat_n_tokens_metadata_sum():
    """concat() updates n_tokens to sum of all segments."""
    seg1 = _make_kv_segment(n_tokens=4)
    seg2 = _make_kv_segment(n_tokens=6)
    merged = KVConcatSegment.concat([seg1, seg2])
    assert merged.n_tokens == 10


def test_concat_preserves_last_metadata():
    """Non-n_tokens metadata comes from the last segment."""
    seg1 = _make_kv_segment(n_tokens=2, layer_index=0)
    seg2 = KVConcatSegment(
        keys=seg1.keys,
        values=seg1.values,
        layer_index=0,
        n_tokens=2,
        bits=8,
        class_name="KVCacheVariant",
    )
    merged = KVConcatSegment.concat([seg1, seg2])
    assert merged.class_name == "KVCacheVariant"


def test_concat_single_segment_passthrough():
    """Single-element concat returns a segment with the same shape."""
    seg = _make_kv_segment(n_tokens=7)
    merged = KVConcatSegment.concat([seg])
    assert merged.keys.packed.shape[-2] == seg.keys.packed.shape[-2]
    assert merged.n_tokens == 7


# ── segment pipeline ─────────────────────────────────────────────────────────


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
    """segment on a KVCache state produces a KVConcatSegment."""
    states = [_make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = segment(states, policy=policy, group_size=64)
    assert kv_list[0] is not None
    assert rec_list[0] is None
    assert isinstance(kv_list[0], KVConcatSegment)
    assert not isinstance(kv_list[0], KVRotatingSegment)
    assert kv_list[0].n_tokens == 4


def test_segment_rotating_kvcache_produces_last_strategy():
    """segment on a RotatingKVCache state produces a KVRotatingSegment."""
    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = segment(states, policy=policy, group_size=64)
    assert kv_list[0] is not None
    assert isinstance(kv_list[0], KVRotatingSegment)


def test_segment_recurrent_produces_recurrent_layer_segment():
    """segment on a non-KV state produces a RecurrentLayerSegment."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    states = [
        {"class_name": "MambaCache", "state": (mx.zeros((1, 4)),), "meta_state": ()}
    ]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, rec_list = segment(states, policy=policy, group_size=64)
    assert kv_list[0] is None
    assert rec_list[0] is not None
    assert isinstance(rec_list[0], RecurrentLayerSegment)
    assert rec_list[0].metadata["class_name"] == "MambaCache"


def test_segment_layer_index_matches_position():
    """layer_index matches position in input list."""
    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    assert kv_list[0].layer_index == 0
    assert kv_list[1].layer_index == 1


# ── assemble pipeline ────────────────────────────────────────────────────────


def test_assemble_kvcache_returns_batch_quantized_kv_cache():
    """assemble on a KVConcatSegment returns a BatchQuantizedKVCache (ADR-0005)."""
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    states = [_make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = assemble(kv_layers, [], group_size=64)
    assert len(result) == 1
    assert isinstance(result[0], BatchQuantizedKVCache)


def test_assemble_kvcache_offset_matches_n_tokens():
    """Reconstructed cache _idx (logical length) equals original n_tokens."""
    # Reads ._idx directly; refactor to use a public attribute when one is exposed.
    n_tokens = 6
    states = [_make_kvcache_state(n_tokens=n_tokens)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = assemble(kv_layers, [], group_size=64)
    assert result[0]._idx == n_tokens


def test_assemble_rotating_returns_rotating_kv_cache():
    """assemble on a KVRotatingSegment returns a RotatingKVCache."""
    from mlx_lm.models.cache import RotatingKVCache
    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    result = assemble(kv_layers, [], group_size=64)
    assert len(result) == 1
    assert isinstance(result[0], RotatingKVCache)


def test_assemble_mixed_layer_ordering():
    """assemble returns layers sorted by layer_index regardless of input order."""
    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    kv_layers = [k for k in kv_list if k is not None]
    # Reverse to test sorting
    result = assemble(
        list(reversed(kv_layers)), [], group_size=64
    )
    assert len(result) == 2


# ── round-trip: segment → KVConcatSegment.concat → assemble ─────────────────


def test_kvcache_round_trip_shape():
    """KVCache: segment, concat two nodes, assemble → correct sequence length."""
    states1 = [_make_kvcache_state(n_tokens=3)]
    states2 = [_make_kvcache_state(n_tokens=5)]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)

    kv1, _ = segment(states1, policy=policy, group_size=64)
    kv2, _ = segment(states2, policy=policy, group_size=64)

    merged = KVConcatSegment.concat([kv1[0], kv2[0]])
    result = assemble([merged], [], group_size=64)
    assert len(result) == 1
    # logical length should be 3 + 5 = 8
    # Reads ._idx directly; refactor to use a public attribute when one is exposed.
    assert result[0]._idx == 8


# ── KVConcatSegment.concat bits-agree invariant ───────────────────────────────


def _make_kv_segment_with_bits(n_tokens: int, bits: int | None) -> KVConcatSegment:
    if bits is None:
        keys = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
        values = mx.ones((1, 1, n_tokens, 64), dtype=mx.bfloat16)
        return KVConcatSegment(
            keys=keys,
            values=values,
            layer_index=0,
            n_tokens=n_tokens,
            bits=None,
            class_name="KVCache",
        )
    seg = _make_kv_segment(n_tokens=n_tokens)
    # Reconstruct with different bits value
    return KVConcatSegment(
        keys=seg.keys,
        values=seg.values,
        layer_index=seg.layer_index,
        n_tokens=seg.n_tokens,
        bits=bits,
        class_name=seg.class_name,
    )


def test_concat_mismatched_bits_raises():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=8)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=4)
    with pytest.raises(AssertionError):
        KVConcatSegment.concat([a, b])


def test_concat_matching_bits_succeeds():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=8)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=8)
    merged = KVConcatSegment.concat([a, b])
    assert merged.bits == 8
    assert merged.n_tokens == 8


def test_concat_matching_float_bits_succeeds():
    a = _make_kv_segment_with_bits(n_tokens=3, bits=None)
    b = _make_kv_segment_with_bits(n_tokens=5, bits=None)
    merged = KVConcatSegment.concat([a, b])
    assert merged.bits is None


# ── segment with KVQuantPolicy ──────────────────────────────────────────────


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

    kv_list, rec_list = segment(live, policy=policy, group_size=64)

    assert rec_list == [None, None, None, None]
    assert kv_list[0].bits == 8
    assert isinstance(kv_list[0].keys, QuantizedArray)
    assert kv_list[1].bits is None
    assert not isinstance(kv_list[1].keys, QuantizedArray)
    assert kv_list[2].bits == 8
    assert kv_list[3].bits is None


def test_segment_policy_none_disables_quantization():
    """policy=None: everything stored as float regardless of class."""
    from vllm_mlx.kv_cache import QuantizedArray

    live = [
        _make_kvcache_live_state(n_tokens=128, layer_index=0),
        _make_rotating_live_state(n_tokens=64, layer_index=1),
    ]

    kv_list, _ = segment(live, policy=None, group_size=64)

    for seg_ in kv_list:
        assert seg_.bits is None
        assert not isinstance(seg_.keys, QuantizedArray)


# ── assemble reads bits from typed fields ───────────────────────────────────


def test_assemble_reads_bits_from_typed_fields_mixed():
    """Build segments with mixed bits fields; assemble reconstructs correct cache types."""
    from mlx_lm.models.cache import KVCache as _KVCache, RotatingKVCache as _RotatingKVCache
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    from vllm_mlx.kv_cache import QuantizedArray

    # Full-attention quantized layer (layer 0)
    n = 128
    k = mx.ones((1, 4, n, 128), dtype=mx.bfloat16)
    v = mx.ones((1, 4, n, 128), dtype=mx.bfloat16)
    qk = QuantizedArray(*mx.quantize(k, group_size=64, bits=8))
    qv = QuantizedArray(*mx.quantize(v, group_size=64, bits=8))
    full_seg = KVConcatSegment(
        keys=qk,
        values=qv,
        layer_index=0,
        n_tokens=n,
        bits=8,
        class_name="KVCache",
    )

    # Sliding-window float layer (layer 1)
    max_size = 256
    rk = mx.ones((1, 4, max_size, 128), dtype=mx.bfloat16)
    rv = mx.ones((1, 4, max_size, 128), dtype=mx.bfloat16)
    sliding_seg = KVRotatingSegment(
        keys=rk,
        values=rv,
        layer_index=1,
        n_tokens=max_size,
        bits=None,
        max_size=max_size,
        keep=0,
        offset=max_size,
        idx=max_size,
    )

    caches = assemble(
        kv_layers=[full_seg, sliding_seg],
        recurrent_layers=[],
        group_size=64,
    )

    assert isinstance(caches[0], BatchQuantizedKVCache)
    assert isinstance(caches[1], _RotatingKVCache)


# ── TurnCacheManager constructor ───────────────────────────────────────────────


def test_turncachemanager_holds_policy():
    """TurnCacheManager stores a KVQuantPolicy instance (not a raw kv_bits int)."""
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    inner = TurnPrefixCache(TurnPrefixCacheConfig())
    policy = KVQuantPolicy(sliding_bits=None, full_bits=8)
    mgr = TurnCacheManager(inner, policy=policy, kv_group_size=64)

    assert mgr._policy is policy
    assert mgr._kv_group_size == 64


def test_turncachemanager_accepts_none_policy():
    """policy=None means quantization disabled."""
    from vllm_mlx.prefix_cache_adapters import TurnCacheManager
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    inner = TurnPrefixCache(TurnPrefixCacheConfig())
    mgr = TurnCacheManager(inner, policy=None)
    assert mgr._policy is None


# ── Translator module round-trip ──────────────────────────────────────────────

from vllm_mlx.cache_translator import (
    assemble,
    segment,
    slice_kv_to_delta,
)


def _live_kvcache_state(B=1, H=2, T=16, D=64):
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    mx.eval(keys, values)
    return {"class_name": "KVCache", "state": (keys, values), "meta_state": (T,)}


def test_segment_returns_kv_concat_segment_for_kvcache():
    from vllm_mlx.cache_types import KVConcatSegment
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    assert isinstance(kv_list[0], KVConcatSegment)


def test_segment_returns_kv_rotating_segment_for_rotating_kvcache():
    from vllm_mlx.cache_types import KVRotatingSegment
    keys = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    values = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    mx.eval(keys, values)
    states = [{
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (0, 16, 16, 16),
    }]
    policy = KVQuantPolicy(sliding_bits=8, full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    assert isinstance(kv_list[0], KVRotatingSegment)


def test_segment_arrays_survive_source_deletion():
    """Segment contract: emitted arrays are graph-detached from source."""
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    seg = kv_list[0]
    del states
    mx.clear_cache()
    dequant_keys = mx.dequantize(
        seg.keys.packed, seg.keys.scales, seg.keys.biases,
        group_size=64, bits=8,
    )
    mx.eval(dequant_keys)
    assert dequant_keys.shape == (1, 2, 16, 64)


def test_assemble_inverse_of_segment_for_kvcache():
    states = [_live_kvcache_state()]
    policy = KVQuantPolicy(full_bits=8)
    kv_list, _ = segment(states, policy=policy, group_size=64)
    caches = assemble(kv_list, [], group_size=64)
    assert len(caches) == 1
    # Round-trip n_tokens preserved.
    assert kv_list[0].n_tokens == 16


def test_slice_kv_to_delta_slices_kvcache():
    states = [_live_kvcache_state(T=32)]
    sliced = slice_kv_to_delta(states, prev_end=16)
    assert sliced[0]["state"][0].shape[2] == 16


def test_slice_kv_to_delta_leaves_rotating_untouched():
    keys = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    values = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)
    states = [{
        "class_name": "RotatingKVCache",
        "state": (keys, values),
        "meta_state": (0, 16, 16, 16),
    }]
    sliced = slice_kv_to_delta(states, prev_end=8)
    assert sliced[0]["state"][0].shape == (1, 2, 16, 64)  # untouched
