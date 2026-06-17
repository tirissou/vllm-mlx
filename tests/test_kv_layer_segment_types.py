"""Tests for KVLayerSegment ABC + KVConcatSegment + KVRotatingSegment.

Verifies polymorphic dispatch (merge_path, reconstruct) and the type-system
guarantee that concat is only valid for KVConcatSegment.
"""
import mlx.core as mx
import pytest

from vllm_mlx.cache_types import (
    KVConcatSegment,
    KVLayerSegment,
    KVRotatingSegment,
)
from vllm_mlx.kv_cache import QuantizedArray


def _q(shape=(1, 2, 16, 64), bits=8, group_size=64):
    arr = mx.random.normal(shape).astype(mx.float16)
    return QuantizedArray(*mx.quantize(arr, group_size=group_size, bits=bits))


def test_concat_segment_merge_path_concatenates():
    a = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    b = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    merged = a.merge_path([a, b])
    assert isinstance(merged, KVConcatSegment)
    assert merged.n_tokens == 32


def test_rotating_segment_merge_path_returns_last():
    a = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    b = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=32, idx=0,
    )
    merged = a.merge_path([a, b])
    assert merged is b


def test_concat_classmethod_rejects_mismatched_bits():
    a = KVConcatSegment(keys=_q(bits=8), values=_q(bits=8), layer_index=0, n_tokens=16, bits=8)
    b = KVConcatSegment(keys=_q(bits=4), values=_q(bits=4), layer_index=0, n_tokens=16, bits=4)
    with pytest.raises(AssertionError, match="mismatched bits"):
        KVConcatSegment.concat([a, b])


def test_concat_segment_is_frozen():
    seg = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    with pytest.raises((AttributeError, Exception)):
        seg.n_tokens = 99  # type: ignore[misc]


def test_concat_reconstruct_returns_batch_quantized_kv_cache():
    from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache
    seg = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    cache = seg.reconstruct(group_size=64)
    assert isinstance(cache, BatchQuantizedKVCache)


def test_rotating_reconstruct_returns_rotating_kv_cache():
    from mlx_lm.models.cache import RotatingKVCache
    seg = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    cache = seg.reconstruct(group_size=64)
    assert isinstance(cache, RotatingKVCache)


def test_subclass_dispatch_via_base_type_annotation():
    """A list typed as list[KVLayerSegment] holds mixed concrete types."""
    concat = KVConcatSegment(keys=_q(), values=_q(), layer_index=0, n_tokens=16, bits=8)
    rotating = KVRotatingSegment(
        keys=_q(), values=_q(), layer_index=1, n_tokens=16, bits=8,
        max_size=16, keep=0, offset=16, idx=16,
    )
    segs: list[KVLayerSegment] = [concat, rotating]
    assert isinstance(segs[0], KVConcatSegment)
    assert isinstance(segs[1], KVRotatingSegment)
