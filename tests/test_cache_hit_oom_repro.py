# SPDX-License-Identifier: Apache-2.0
"""Regression test for OOM at decode-from-cache-hit on large prefixes.

TurnCacheManager._assemble used to construct an mlx_lm QuantizedKVCache sized
exactly to n_tokens. The next update_and_fetch then entered the
`(prev + num_steps) > self.keys[0].shape[-2]` branch, which trims and concatenates
a fresh buffer — doubling Metal-memory transiently. At 60k cached tokens on a
Gemma 4 26B-A4B-class model that doubling is GB-scale and OOMs the server.

The fix (see ADR-0005): emit BatchQuantizedKVCache.from_quantized_arrays(...)
with buffers pre-padded to the next `step` boundary so the first update_and_fetch
lands on the in-place assignment branch.

Signal: peak Metal allocation during the first decode step, above the
pre-decode active level. Must stay near zero.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import pytest

from vllm_mlx.cache_types import KVLayerSegment
from vllm_mlx.kv_cache import QuantizedArray
from vllm_mlx.prefix_cache_adapters import TurnCacheManager


# Realistic shape for the full-attention slice of a Gemma 4 26B-A4B-class model.
# 60000 % 256 = 96 — deliberately not step-aligned, like the reported failure.
N_TOKENS = 60_000
N_KV_HEADS = 4
HEAD_DIM = 512
GROUP_SIZE = 64
BITS = 8
EL_PER_INT = 32 // BITS
N_FULL_LAYERS = 7  # ~1 in 5 of 35 layers in Gemma 4 26B-A4B.


def _make_quantized_segment(layer_index: int, n_tokens: int) -> KVLayerSegment:
    packed_shape = (1, N_KV_HEADS, n_tokens, HEAD_DIM // EL_PER_INT)
    scale_shape = (1, N_KV_HEADS, n_tokens, HEAD_DIM // GROUP_SIZE)
    keys = QuantizedArray(
        packed=mx.zeros(packed_shape, dtype=mx.uint32),
        scales=mx.zeros(scale_shape, dtype=mx.bfloat16),
        biases=mx.zeros(scale_shape, dtype=mx.bfloat16),
    )
    values = QuantizedArray(
        packed=mx.zeros(packed_shape, dtype=mx.uint32),
        scales=mx.zeros(scale_shape, dtype=mx.bfloat16),
        biases=mx.zeros(scale_shape, dtype=mx.bfloat16),
    )
    mx.eval(keys.packed, keys.scales, keys.biases,
            values.packed, values.scales, values.biases)
    return KVLayerSegment(
        keys=keys,
        values=values,
        metadata={
            "class_name": "KVCache",
            "layer_index": layer_index,
            "merge_strategy": "concatenate",
            "n_tokens": n_tokens,
        },
    )


def _flatten_arrays(x):
    if isinstance(x, mx.array):
        yield x
    elif hasattr(x, "packed"):
        yield x.packed; yield x.scales; yield x.biases
    else:
        for v in x:
            yield from _flatten_arrays(v)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_assemble_does_not_spike_on_first_decode_step():
    """Reconstruction + first decode step must not allocate a transient buffer
    on the order of the cache size."""
    gc.collect()

    kv_layers = [_make_quantized_segment(i, N_TOKENS) for i in range(N_FULL_LAYERS)]

    caches = TurnCacheManager._assemble(
        kv_layers=kv_layers,
        recurrent_layers=[],
        group_size=GROUP_SIZE,
        bits=BITS,
    )

    eval_args = []
    for cache in caches:
        eval_args.extend(_flatten_arrays(cache.keys))
        eval_args.extend(_flatten_arrays(cache.values))
    mx.eval(*eval_args)
    resident = mx.get_active_memory()

    mx.reset_peak_memory()
    pre_decode = mx.get_active_memory()

    outs = []
    for cache in caches:
        k = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
        v = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
        out_k, out_v = cache.update_and_fetch(k, v)
        outs.extend(_flatten_arrays(out_k))
        outs.extend(_flatten_arrays(out_v))
    mx.eval(*outs)

    peak_decode = mx.get_peak_memory() - pre_decode

    print(
        f"\n[regress] N_TOKENS={N_TOKENS} N_FULL_LAYERS={N_FULL_LAYERS} "
        f"resident={resident/1e6:.1f} MB  first-decode peak={peak_decode/1e6:.1f} MB"
    )

    # Threshold: 10% of the per-layer cache size, summed over layers. A reallocation
    # by `step` slots is bounded by step/N_TOKENS * cache_bytes — for step=256 and
    # N=60000 that's <0.5%. 10% leaves comfortable headroom for MLX bookkeeping
    # while still catching a full-buffer-doubling regression by orders of magnitude.
    per_layer_bytes_keys = (
        (N_KV_HEADS * N_TOKENS * (HEAD_DIM // EL_PER_INT) * 4)
        + 2 * (N_KV_HEADS * N_TOKENS * (HEAD_DIM // GROUP_SIZE) * 2)
    )
    cache_bytes = 2 * N_FULL_LAYERS * per_layer_bytes_keys  # K + V
    threshold = int(0.10 * cache_bytes)
    assert peak_decode < threshold, (
        f"first-decode peak {peak_decode/1e6:.1f} MB exceeds threshold "
        f"{threshold/1e6:.1f} MB (10% of {cache_bytes/1e6:.1f} MB cache). "
        f"Reconstruction is forcing an allocation+concat on the first decode step."
    )
