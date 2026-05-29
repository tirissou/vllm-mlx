import mlx.core as mx
import pytest

from vllm_mlx.cache_translator import CacheTranslator

def test_linearize_no_wrap():
    # max_size=10, offset=10
    # Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Result should be: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = mx.array([[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]], dtype=mx.float32)
    max_size = 10
    offset = 10
    
    result = CacheTranslator.linearize(data, offset, max_size)
    assert mx.array_equal(result, data)

def test_linearize_with_wrap():
    # max_size=10, offset=4
    # Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Sequence starting from offset 4: [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    data = mx.array([[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]], dtype=mx.float32)
    max_size = 10
    offset = 4
    
    expected = mx.array([[[4], [5], [6], [7], [8], [9], [0], [1], [2], [3]]], dtype=mx.float32)
    result = CacheTranslator.linearize(data, offset, max_size)
    assert mx.array_equal(result, expected)

def test_quantize_kv():
    # Create some dummy KV arrays
    arr1 = mx.array([[[1.0], [-1.0]]], dtype=mx.bfloat16)
    arr2 = mx.array([[[2.0], [0.0]]], dtype=mx.bfloat16)
    
    # max_val for arr1 is 1.0. Scale = 1/127.
    # q = round(arr / scale) = round(arr * 127)
    # arr1 * 127 = [[127], [-127]]
    
    # max_val for arr2 is 2.0. Scale = 2/127.
    # q = round(arr / scale) = round(arr * 127 / 2) = round(arr * 63.5)
    # arr2 * 63.5 = [[127], [0]]
    
    result_q, result_scales = CacheTranslator.quantize_kv([arr1, arr2])
    
    assert len(result_q) == 2
    assert len(result_scales) == 2
    
    # Check arr1
    assert mx.array_equal(result_q[0], mx.array([[[127], [-127]]], dtype=mx.int8))
    assert pytest.approx(result_scales[0]) == 1.0/127.0
    
    # Check arr2
    assert mx.array_equal(result_q[1], mx.array([[[127], [0]]], dtype=mx.int8))
    assert pytest.approx(result_scales[1]) == 2.0/127.0
