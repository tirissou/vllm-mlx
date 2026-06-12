# SPDX-License-Identifier: Apache-2.0


def test_kv_quant_bench_exports():
    from vllm_mlx.kv_quant_bench import (
        _dequantize_cache,
        _quantize_cache,
        estimate_kv_cache_memory,
    )
    assert _dequantize_cache is not None
    assert _quantize_cache is not None
    assert estimate_kv_cache_memory is not None
