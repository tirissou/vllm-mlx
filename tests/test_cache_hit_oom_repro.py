# SPDX-License-Identifier: Apache-2.0
"""Regression test for OOM at decode-from-cache-hit on large prefixes.

TurnCacheManager._assemble used to construct an mlx_lm KVCache (or
QuantizedKVCache) sized exactly to n_tokens. The next update_and_fetch
then entered the `(prev + 1) > self.keys.shape[-2]` branch, which trims
and concatenates a fresh buffer — doubling Metal-memory transiently. At
60k cached tokens on a Gemma 4 26B-A4B-class model that doubling is
GB-scale and OOMs the server.

The fix (see ADR-0005): emit a cache pre-padded to the next `step` boundary
so the first update_and_fetch lands on the in-place assignment branch.

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
N_KV_HEADS = 4
HEAD_DIM = 512
GROUP_SIZE = 64
BITS = 8
EL_PER_INT = 32 // BITS
N_FULL_LAYERS = 7  # ~1 in 5 of 35 layers in Gemma 4 26B-A4B.

# Two token counts: one deliberately off the 256-step grid (the original
# failure shape), one exactly on it (the step-aligned edge case that the
# original quantized fix missed).
N_TOKENS_UNALIGNED = 60_000   # 60000 % 256 = 96
N_TOKENS_ALIGNED = 60_160     # 235 * 256


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
    mx.eval(
        keys.packed, keys.scales, keys.biases,
        values.packed, values.scales, values.biases,
    )
    return KVLayerSegment(
        keys=keys,
        values=values,
        metadata={
            "bits": BITS,
            "class_name": "KVCache",
            "layer_index": layer_index,
            "merge_strategy": "concatenate",
            "n_tokens": n_tokens,
        },
    )


def _make_unquantized_segment(layer_index: int, n_tokens: int) -> KVLayerSegment:
    shape = (1, N_KV_HEADS, n_tokens, HEAD_DIM)
    keys = mx.zeros(shape, dtype=mx.bfloat16)
    values = mx.zeros(shape, dtype=mx.bfloat16)
    mx.eval(keys, values)
    return KVLayerSegment(
        keys=keys,
        values=values,
        metadata={
            "bits": None,
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


def _per_layer_bytes_quantized(n_tokens: int) -> int:
    # packed: uint32 (4 bytes) ; scales/biases: bfloat16 (2 bytes each).
    return (
        (N_KV_HEADS * n_tokens * (HEAD_DIM // EL_PER_INT) * 4)
        + 2 * (N_KV_HEADS * n_tokens * (HEAD_DIM // GROUP_SIZE) * 2)
    )


def _per_layer_bytes_unquantized(n_tokens: int) -> int:
    # bfloat16: 2 bytes per element.
    return N_KV_HEADS * n_tokens * HEAD_DIM * 2


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize(
    "kind, n_tokens, make_segment, per_layer_bytes",
    [
        ("quantized_unaligned",   N_TOKENS_UNALIGNED, _make_quantized_segment,   _per_layer_bytes_quantized),
        ("quantized_aligned",     N_TOKENS_ALIGNED,   _make_quantized_segment,   _per_layer_bytes_quantized),
        ("unquantized_unaligned", N_TOKENS_UNALIGNED, _make_unquantized_segment, _per_layer_bytes_unquantized),
        ("unquantized_aligned",   N_TOKENS_ALIGNED,   _make_unquantized_segment, _per_layer_bytes_unquantized),
    ],
)
def test_assemble_does_not_spike_on_first_decode_step(
    kind, n_tokens, make_segment, per_layer_bytes
):
    """Reconstruction + first decode step must not allocate a transient buffer
    on the order of the cache size, regardless of payload kind or whether
    n_tokens is aligned to the cache's `step` boundary."""
    gc.collect()

    kv_layers = [make_segment(i, n_tokens) for i in range(N_FULL_LAYERS)]

    caches = TurnCacheManager._assemble(
        kv_layers=kv_layers,
        recurrent_layers=[],
        group_size=GROUP_SIZE,
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

    cache_bytes = 2 * N_FULL_LAYERS * per_layer_bytes(n_tokens)  # K + V
    threshold = int(0.10 * cache_bytes)

    print(
        f"\n[regress:{kind}] n_tokens={n_tokens} layers={N_FULL_LAYERS} "
        f"resident={resident/1e6:.1f} MB  "
        f"first-decode peak={peak_decode/1e6:.1f} MB  "
        f"threshold={threshold/1e6:.1f} MB"
    )

    assert peak_decode < threshold, (
        f"[{kind}] first-decode peak {peak_decode/1e6:.1f} MB exceeds threshold "
        f"{threshold/1e6:.1f} MB (10% of {cache_bytes/1e6:.1f} MB cache). "
        f"Reconstruction is forcing an allocation+concat on the first decode step."
    )
