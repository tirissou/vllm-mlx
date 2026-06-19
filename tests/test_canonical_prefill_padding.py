"""Unit tests for CanonicalPrefillBatchGenerator's padding shim.

We don't construct a full BatchGenerator (heavy). Instead we test the shim's
core behavior — pad + trim + slice — against real mlx_lm cache instances
wrapped around a tiny toy model.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models.cache import KVCache, RotatingKVCache, BatchRotatingKVCache

from vllm_mlx.scheduler import _make_padding_shim


class _ToyModel(nn.Module):
    """Stores per-position embeddings into cache layers so we can assert
    geometry after pad / trim / slice."""

    def __init__(self, n_layers: int = 2, dim: int = 8, vocab: int = 32):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)

    def __call__(self, inputs, cache=None):
        B, S = inputs.shape
        x = self.embed(inputs)  # (B, S, dim)
        if cache is not None:
            for layer in cache:
                k = x[:, None, :, :]
                v = x[:, None, :, :]
                layer.update_and_fetch(k, v)
        return mx.zeros((B, S, self.embed.weight.shape[0]))


def _make_caches(n_layers=2, sliding_max=16):
    return [
        KVCache() if i % 2 == 0 else RotatingKVCache(max_size=sliding_max)
        for i in range(n_layers)
    ]


def test_no_padding_when_S_equals_canonical_M():
    model = _ToyModel()
    canonical_M = 32
    shim = _make_padding_shim(model, canonical_M)

    cache = _make_caches(n_layers=2, sliding_max=16)
    inputs = mx.zeros((1, canonical_M), dtype=mx.int32)
    shim(inputs, cache=cache)

    # Full M of work landed: full layer has offset == 32, rotating offset == 32.
    assert cache[0].offset == canonical_M
    assert cache[1].offset == canonical_M


def test_no_padding_for_decode_step():
    model = _ToyModel()
    canonical_M = 32
    shim = _make_padding_shim(model, canonical_M)

    cache = _make_caches(n_layers=2, sliding_max=16)
    # Warm with 4 real tokens first.
    shim(mx.zeros((1, canonical_M), dtype=mx.int32), cache=cache)
    pre_offset_full = cache[0].offset
    pre_offset_rot = cache[1].offset

    # Now feed one decode token.
    decode_input = mx.array([[5]], dtype=mx.int32)
    shim(decode_input, cache=cache)

    assert cache[0].offset == pre_offset_full + 1
    assert cache[1].offset == pre_offset_rot + 1


def test_padded_forward_advances_offset_by_real_tokens():
    """Shim runs canonical_M tokens through the model but rewinds cache
    offset to N_real after trim(pad)."""
    model = _ToyModel()
    canonical_M = 32
    shim = _make_padding_shim(model, canonical_M)

    cache = _make_caches(n_layers=2, sliding_max=16)
    N_real = 24
    inputs = mx.arange(N_real, dtype=mx.int32)[None, :]
    shim(inputs, cache=cache)

    assert cache[0].offset == N_real, f"full-attn offset should be N_real after trim"
    # RotatingKVCache.trim only decrements offset/_idx.
    assert cache[1].offset == N_real


def test_rotating_buffer_geometry_after_pad_trim_slice():
    """After pad + trim + physical slice, the RotatingKVCache buffer must:
       - have shape (..., max_size + N_real, ...) along sequence axis
       - have _idx == max_size + N_real
       - not contain the pad rows
    Pre-condition: the buffer is in steady state (offset > max_size).
    """
    model = _ToyModel()
    canonical_M = 32
    sliding_max = 16
    shim = _make_padding_shim(model, canonical_M)

    cache = [RotatingKVCache(max_size=sliding_max)]

    # Warm to steady state: push canonical_M tokens twice so offset > max_size.
    shim(mx.arange(canonical_M, dtype=mx.int32)[None, :], cache=cache)
    shim(mx.arange(canonical_M, 2 * canonical_M, dtype=mx.int32)[None, :], cache=cache)
    assert cache[0].offset >= sliding_max

    pre_keys_shape = cache[0].keys.shape
    pre_offset = cache[0].offset

    # Sub-canonical prefill: N_real = 8 (pad = 24).
    N_real = 8
    inputs = mx.arange(N_real, dtype=mx.int32)[None, :]
    shim(inputs, cache=cache)

    layer = cache[0]
    # After _update_concat the buffer was (max_size + canonical_M - 1, ...).
    # After trim(pad) — pad=24 — and physical slicing of pad rows, the buffer
    # should be (max_size + N_real - 1, ...). The trim decrements _idx by pad
    # then the slice drops pad rows from the buffer.
    assert layer.offset == pre_offset + N_real, "offset advances by real tokens"
    assert layer.keys.shape[-2] == layer._idx, \
        "buffer length must equal _idx after slice"
    assert layer.keys.shape[-2] < pre_keys_shape[-2] + canonical_M, \
        "pad rows must have been sliced off"


def test_decode_after_pad_trim_reads_correct_positions():
    """Post-pad-trim, the first decode step must extend the same logical
    position — no jump caused by stale pad rows."""
    model = _ToyModel()
    canonical_M = 32
    sliding_max = 16
    shim = _make_padding_shim(model, canonical_M)

    cache = [RotatingKVCache(max_size=sliding_max)]
    # Warm to steady state then run a sub-canonical prefill.
    shim(mx.arange(canonical_M, dtype=mx.int32)[None, :], cache=cache)
    shim(mx.arange(canonical_M, 2 * canonical_M, dtype=mx.int32)[None, :], cache=cache)
    N_real = 5
    shim(mx.arange(N_real, dtype=mx.int32)[None, :], cache=cache)
    pre_offset = cache[0].offset

    # One decode token.
    shim(mx.array([[42]], dtype=mx.int32), cache=cache)
    assert cache[0].offset == pre_offset + 1


def test_shim_returns_unpadded_logits():
    """When shim pads input from S to canonical_M, returned tensor must be
    sliced back to S on the sequence axis so downstream sampler sees the
    real-token logits."""
    model = _ToyModel()
    canonical_M = 32
    shim = _make_padding_shim(model, canonical_M)

    cache = _make_caches(n_layers=1, sliding_max=16)
    N_real = 7
    inputs = mx.zeros((1, N_real), dtype=mx.int32)
    out = shim(inputs, cache=cache)
    assert out.shape[-2] == N_real


def test_shim_delegates_unknown_attributes_to_model():
    """The shim must expose model attributes so downstream paths
    (MTP heads, custom forward methods) keep working."""
    model = _ToyModel()
    model.some_arbitrary_method = lambda: "ok"
    shim = _make_padding_shim(model, canonical_M=32)
    assert shim.some_arbitrary_method() == "ok"
    # Cache attributes too
    assert shim.embed is model.embed


def test_shim_skips_physical_slice_for_batch_rotating_with_lengths():
    """When BatchRotatingKVCache._lengths is not None (mixed-length batch),
    the shim must NOT apply the tail-slice: after dynamic_roll the pad rows
    are not at the buffer tail so slicing would corrupt real K/V entries.

    We verify by running the same sub-canonical call twice — once with
    _lengths=None (tail-slice path) and once with _lengths active (skip-slice
    path) — and asserting the buffer is larger in the latter case by exactly
    the pad amount."""
    import copy

    model = _ToyModel()
    canonical_M = 32
    N_real = canonical_M - 8  # pad = 8
    pad = canonical_M - N_real
    shim = _make_padding_shim(model, canonical_M)

    # Construct two identical BatchRotatingKVCache instances.
    def make_warmed_layer():
        layer = BatchRotatingKVCache(max_size=16, left_padding=[0, 0])
        warm_input = mx.zeros((2, canonical_M), dtype=mx.int32)
        shim(warm_input, cache=[layer])
        mx.eval(layer.keys)
        return layer

    layer_no_len = make_warmed_layer()
    layer_with_len = make_warmed_layer()

    # Activate _lengths on one layer to simulate mixed-length batch state.
    layer_with_len._lengths = mx.array([canonical_M, canonical_M - 4])

    sub_input = mx.zeros((2, N_real), dtype=mx.int32)

    # Run shim on both layers.
    shim(sub_input, cache=[layer_no_len])
    shim(sub_input, cache=[layer_with_len])
    mx.eval(layer_no_len.keys, layer_with_len.keys)

    seq_len_no_len = layer_no_len.keys.shape[-2]
    seq_len_with_len = layer_with_len.keys.shape[-2]

    # The _lengths path skips the physical tail-slice, so the buffer is
    # exactly `pad` rows larger than the sliced version.
    assert seq_len_with_len == seq_len_no_len + pad, (
        f"With _lengths active, buffer should be {pad} rows larger than sliced "
        f"path (expected {seq_len_no_len + pad}, got {seq_len_with_len})"
    )
