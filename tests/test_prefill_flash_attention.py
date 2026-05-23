# SPDX-License-Identifier: Apache-2.0
"""Tests for the prefill flash-attention patch."""

import mlx.core as mx
import mlx_lm.models.base as _base
from mlx_lm.models.cache import QuantizedKVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quantized_context(B=1, H_kv=2, C=64, D=64, bits=4, group_size=64):
    """Return (cache, q_keys, q_values) with C cached tokens."""
    cache = QuantizedKVCache(group_size=group_size, bits=bits)
    k = mx.random.normal((B, H_kv, C, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, C, D)).astype(mx.float16)
    q_keys, q_values = cache.update_and_fetch(k, v)
    mx.eval(q_keys, q_values)
    return cache, q_keys, q_values


class _NoCache:
    """Mimics a non-quantized cache (no .bits attribute)."""
    pass


# ---------------------------------------------------------------------------
# Tests — import the patched function directly (no apply() needed for unit tests)
# ---------------------------------------------------------------------------

def test_prefill_output_matches_dequantize_flash():
    """L > 1: output equals dequantize(KV) + mx.fast.scaled_dot_product_attention."""
    from vllm_mlx.patches.mlx_lm_prefill_flash_sdpa import _patched_sdpa

    mx.random.seed(0)
    B, H_q, H_kv, L, C, D = 1, 4, 2, 16, 64, 64
    cache, q_keys, q_values = _make_quantized_context(B, H_kv, C, D)

    queries = mx.random.normal((B, H_q, L, D)).astype(mx.float16)
    scale = D ** -0.5

    k_f16 = mx.dequantize(*q_keys, group_size=cache.group_size, bits=cache.bits)
    v_f16 = mx.dequantize(*q_values, group_size=cache.group_size, bits=cache.bits)
    expected = mx.fast.scaled_dot_product_attention(
        queries, k_f16, v_f16, scale=scale, mask="causal"
    )
    actual = _patched_sdpa(queries, q_keys, q_values, cache=cache, scale=scale, mask="causal")

    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, atol=1e-5).item()


def test_decode_output_matches_quantized_path():
    """L == 1: output is identical to quantized_scaled_dot_product_attention."""
    from vllm_mlx.patches.mlx_lm_prefill_flash_sdpa import _patched_sdpa

    mx.random.seed(1)
    B, H_q, H_kv, C, D = 1, 4, 2, 64, 64
    cache, q_keys, q_values = _make_quantized_context(B, H_kv, C, D)

    queries_orig = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    scale = D ** -0.5

    # Make copies to avoid in-place modification issues with quantized_scaled_dot_product_attention
    queries_expected = mx.array(queries_orig)
    queries_actual = mx.array(queries_orig)

    expected = _base.quantized_scaled_dot_product_attention(
        queries_expected, q_keys, q_values,
        scale=scale, mask="causal",
        group_size=cache.group_size, bits=cache.bits,
    )
    actual = _patched_sdpa(queries_actual, q_keys, q_values, cache=cache, scale=scale, mask="causal")

    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, atol=1e-5).item()


def test_non_quantized_cache_passthrough():
    """No .bits: delegates to mx.fast.scaled_dot_product_attention unchanged."""
    from vllm_mlx.patches.mlx_lm_prefill_flash_sdpa import _patched_sdpa

    mx.random.seed(2)
    B, H_q, H_kv, L, C, D = 1, 4, 2, 16, 64, 64
    queries = mx.random.normal((B, H_q, L, D)).astype(mx.float16)
    k = mx.random.normal((B, H_kv, C, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, C, D)).astype(mx.float16)
    scale = D ** -0.5

    expected = mx.fast.scaled_dot_product_attention(
        queries, k, v, scale=scale, mask="causal"
    )
    actual = _patched_sdpa(queries, k, v, cache=_NoCache(), scale=scale, mask="causal")

    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, atol=1e-5).item()


def test_apply_patches_module():
    """apply() replaces mlx_lm.models.base.scaled_dot_product_attention."""
    from vllm_mlx.patches.mlx_lm_prefill_flash_sdpa import apply, _patched_sdpa

    original = _base.scaled_dot_product_attention
    try:
        apply()
        assert _base.scaled_dot_product_attention is _patched_sdpa
    finally:
        _base.scaled_dot_product_attention = original


def test_rejects_sinks_with_quantized_cache():
    """Quantized cache + sinks should raise ValueError."""
    import pytest
    from vllm_mlx.patches.mlx_lm_prefill_flash_sdpa import _patched_sdpa

    cache, q_keys, q_values = _make_quantized_context()
    queries = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
    sinks = mx.random.normal((1, 4, 64)).astype(mx.float16)

    with pytest.raises(ValueError, match="does not support attention sinks"):
        _patched_sdpa(queries, q_keys, q_values, cache=cache,
                      scale=1.0, mask=None, sinks=sinks)
