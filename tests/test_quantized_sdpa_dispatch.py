# SPDX-License-Identifier: Apache-2.0
"""Guard: mlx_lm.scaled_dot_product_attention routes quantized caches to the
quantized SDPA kernel and plain caches to the fast SDPA kernel.

If this test fails:
  - A cache class with .bits stopped being treated as quantized (regression in
    mlx_lm.models.base.scaled_dot_product_attention's dispatch heuristic).
  - QuantizedKVCache.update_and_fetch or BatchQuantizedKVCache.update_and_fetch
    started returning dequantized bf16 tensors instead of quantized tuples.
  - A new cache wrapper hides .bits and silently downgrades decode to bf16 SDPA.

This is a dispatch test, not a numerical or perf test — kept small and fast.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from vllm_mlx.batch_quantized_kv_cache import BatchQuantizedKVCache


N_KV_HEADS = 2
N_Q_HEADS = 4
HEAD_DIM = 64
GROUP_SIZE = 64


class _CallCounter:
    """Wraps a callable to count invocations while still calling through."""

    def __init__(self):
        self.count = 0

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            self.count += 1
            return fn(*args, **kwargs)
        return wrapped


def _install_counters(monkeypatch):
    """Patch mx.quantized_matmul and mx.fast.scaled_dot_product_attention to
    count calls. Returns (qmm_counter, fast_sdpa_counter)."""
    qmm = _CallCounter()
    fast = _CallCounter()
    monkeypatch.setattr(mx, "quantized_matmul", qmm(mx.quantized_matmul))
    monkeypatch.setattr(
        mx.fast,
        "scaled_dot_product_attention",
        fast(mx.fast.scaled_dot_product_attention),
    )
    return qmm, fast


def _seed_with_step(cache):
    """Issue one update_and_fetch so cache.keys/values hold real (quantized
    or plain) state. Returns (keys, values) — whatever the cache returns."""
    k = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
    v = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
    return cache.update_and_fetch(k, v)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize(
    "cache_factory",
    [
        pytest.param(lambda: QuantizedKVCache(group_size=GROUP_SIZE, bits=8), id="quantized_kv_8bit"),
        pytest.param(lambda: QuantizedKVCache(group_size=GROUP_SIZE, bits=4), id="quantized_kv_4bit"),
        pytest.param(lambda: BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=8), id="batch_quantized_kv_8bit"),
    ],
)
def test_sdpa_dispatches_to_quantized_kernel(monkeypatch, cache_factory):
    cache = cache_factory()
    keys, values = _seed_with_step(cache)

    qmm, fast_sdpa = _install_counters(monkeypatch)

    queries = mx.zeros((1, N_Q_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
    out = scaled_dot_product_attention(
        queries, keys, values, cache=cache,
        scale=1.0 / math.sqrt(HEAD_DIM), mask=None,
    )
    mx.eval(out)

    # Two quantized matmuls per attention call: Q @ K^T and softmax(scores) @ V.
    assert qmm.count >= 2, (
        f"expected mx.quantized_matmul to fire >=2 times; got {qmm.count}. "
        f"This means scaled_dot_product_attention did NOT take the quantized "
        f"branch for {type(cache).__name__}."
    )
    assert fast_sdpa.count == 0, (
        f"expected mx.fast.scaled_dot_product_attention to NOT fire; got "
        f"{fast_sdpa.count}. The dispatch fell back to bf16 SDPA even though "
        f"the cache reports .bits — decode is silently materializing bf16 K/V."
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_sdpa_dispatches_to_plain_kernel_for_unquantized_cache(monkeypatch):
    cache = KVCache()
    keys, values = _seed_with_step(cache)

    qmm, fast_sdpa = _install_counters(monkeypatch)

    queries = mx.zeros((1, N_Q_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
    out = scaled_dot_product_attention(
        queries, keys, values, cache=cache,
        scale=1.0 / math.sqrt(HEAD_DIM), mask=None,
    )
    mx.eval(out)

    assert qmm.count == 0, (
        f"expected mx.quantized_matmul to NOT fire for plain KVCache; got "
        f"{qmm.count}. The dispatcher is incorrectly routing unquantized "
        f"caches to the quantized kernel."
    )
    assert fast_sdpa.count >= 1, (
        f"expected mx.fast.scaled_dot_product_attention to fire >=1 time; "
        f"got {fast_sdpa.count}."
    )
