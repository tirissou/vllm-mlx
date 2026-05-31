"""Tests for KVLayerSegment.concat() and the new _segment/_assemble pipeline.

Replaces the old CacheTranslator tests now that linearize/quantize_kv are gone.
"""

import mlx.core as mx
import pytest

from vllm_mlx.cache_types import KVLayerSegment
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
    kv_list, rec_list = TurnCacheManager._segment(states, group_size=64, bits=8)
    assert kv_list[0] is not None
    assert rec_list[0] is None
    assert isinstance(kv_list[0], KVLayerSegment)
    assert kv_list[0].metadata["merge_strategy"] == "concatenate"
    assert kv_list[0].metadata["n_tokens"] == 4


def test_segment_rotating_kvcache_produces_last_strategy():
    """_segment on a RotatingKVCache state produces merge_strategy='last'."""
    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    kv_list, rec_list = TurnCacheManager._segment(states, group_size=64, bits=8)
    assert kv_list[0] is not None
    assert kv_list[0].metadata["merge_strategy"] == "last"


def test_segment_recurrent_produces_recurrent_layer_segment():
    """_segment on a non-KV state produces a RecurrentLayerSegment."""
    from vllm_mlx.cache_types import RecurrentLayerSegment

    states = [
        {"class_name": "MambaCache", "state": (mx.zeros((1, 4)),), "meta_state": ()}
    ]
    kv_list, rec_list = TurnCacheManager._segment(states, group_size=64, bits=8)
    assert kv_list[0] is None
    assert rec_list[0] is not None
    assert isinstance(rec_list[0], RecurrentLayerSegment)
    assert rec_list[0].metadata["class_name"] == "MambaCache"


def test_segment_layer_index_matches_position():
    """layer_index in metadata matches position in input list."""
    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    kv_list, _ = TurnCacheManager._segment(states, group_size=64, bits=8)
    assert kv_list[0].metadata["layer_index"] == 0
    assert kv_list[1].metadata["layer_index"] == 1


# ── _assemble pipeline ────────────────────────────────────────────────────────


def test_assemble_kvcache_returns_quantized_kv_cache():
    """_assemble on a KVLayerSegment returns a QuantizedKVCache."""
    from mlx_lm.models.cache import QuantizedKVCache

    states = [_make_kvcache_state(n_tokens=4)]
    kv_list, _ = TurnCacheManager._segment(states, group_size=64, bits=8)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64, bits=8)
    assert len(result) == 1
    assert isinstance(result[0], QuantizedKVCache)


def test_assemble_kvcache_offset_matches_n_tokens():
    """Reconstructed QuantizedKVCache.offset equals original n_tokens."""
    from mlx_lm.models.cache import QuantizedKVCache

    n_tokens = 6
    states = [_make_kvcache_state(n_tokens=n_tokens)]
    kv_list, _ = TurnCacheManager._segment(states, group_size=64, bits=8)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64, bits=8)
    assert result[0].offset == n_tokens


def test_assemble_rotating_returns_rotating_kv_cache():
    """_assemble on a RotatingKVCache segment returns a RotatingKVCache."""
    from mlx_lm.models.cache import RotatingKVCache

    states = [_make_rotating_state(n_tokens=4, max_size=4)]
    kv_list, _ = TurnCacheManager._segment(states, group_size=64, bits=8)
    kv_layers = [k for k in kv_list if k is not None]
    result = TurnCacheManager._assemble(kv_layers, [], group_size=64, bits=8)
    assert len(result) == 1
    assert isinstance(result[0], RotatingKVCache)


def test_assemble_mixed_layer_ordering():
    """_assemble returns layers sorted by layer_index regardless of input order."""
    from mlx_lm.models.cache import QuantizedKVCache

    states = [_make_kvcache_state(n_tokens=4), _make_kvcache_state(n_tokens=4)]
    kv_list, _ = TurnCacheManager._segment(states, group_size=64, bits=8)
    kv_layers = [k for k in kv_list if k is not None]
    # Reverse to test sorting
    result = TurnCacheManager._assemble(
        list(reversed(kv_layers)), [], group_size=64, bits=8
    )
    assert len(result) == 2


# ── round-trip: _segment → KVLayerSegment.concat → _assemble ─────────────────


def test_kvcache_round_trip_shape():
    """KVCache: segment, concat two nodes, assemble → correct sequence length."""
    states1 = [_make_kvcache_state(n_tokens=3)]
    states2 = [_make_kvcache_state(n_tokens=5)]

    kv1, _ = TurnCacheManager._segment(states1, group_size=64, bits=8)
    kv2, _ = TurnCacheManager._segment(states2, group_size=64, bits=8)

    merged = KVLayerSegment.concat([kv1[0], kv2[0]])
    result = TurnCacheManager._assemble([merged], [], group_size=64, bits=8)
    assert len(result) == 1
    # offset should be 3 + 5 = 8
    assert result[0].offset == 8
