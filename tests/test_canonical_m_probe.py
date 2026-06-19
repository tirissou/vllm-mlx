"""Unit tests for vllm_mlx.canonical_m_probe pure helpers + tiny-model probe."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_mlx.canonical_m_probe import (
    compute_canonical_band,
    intersect_per_batch_bands,
    run_chunking_probe,
)


def test_compute_canonical_band_all_zero_returns_all_M():
    # All pairwise diffs exactly zero across schedules → every chunk size canonical.
    probe = {
        "schedules": [("1x1024", 1024), ("2x512", 512), ("4x256", 256)],
        "diffs": {
            (0, "K"): {("1x1024", "2x512"): 0.0, ("1x1024", "4x256"): 0.0,
                       ("2x512", "4x256"): 0.0},
            (0, "V"): {("1x1024", "2x512"): 0.0, ("1x1024", "4x256"): 0.0,
                       ("2x512", "4x256"): 0.0},
        },
        "layer_types": {0: "RotatingKVCache"},
    }
    assert compute_canonical_band(probe) == [256, 512, 1024]


def test_compute_canonical_band_drops_offending_M():
    # 256 diverges from 512 and 1024 → canonical band is [512, 1024].
    probe = {
        "schedules": [("1x1024", 1024), ("2x512", 512), ("4x256", 256)],
        "diffs": {
            (0, "V"): {
                ("1x1024", "2x512"): 0.0,
                ("1x1024", "4x256"): 8.0,
                ("2x512", "4x256"): 8.0,
            },
        },
        "layer_types": {0: "RotatingKVCache"},
    }
    assert compute_canonical_band(probe) == [512, 1024]


def test_compute_canonical_band_non_contiguous_returns_largest_contiguous():
    # If {512, 2048} are zero-diff but 1024 is not, the band must be contiguous.
    probe = {
        "schedules": [("1x2048", 2048), ("2x1024", 1024), ("4x512", 512)],
        "diffs": {
            (0, "K"): {
                ("1x2048", "2x1024"): 4.0,  # 1024 ≠ 2048
                ("1x2048", "4x512"): 0.0,
                ("2x1024", "4x512"): 4.0,
            },
        },
        "layer_types": {0: "RotatingKVCache"},
    }
    # 512 is canonical solo; 2048 is canonical solo. Contiguous band of zero-diff = [512].
    assert compute_canonical_band(probe) == [512]


def test_intersect_per_batch_bands_basic():
    per_b = {1: [512, 1024], 2: [512, 1024], 4: [1024]}
    assert intersect_per_batch_bands(per_b) == [1024]


def test_intersect_per_batch_bands_empty():
    per_b = {1: [512], 2: [1024]}
    assert intersect_per_batch_bands(per_b) == []


class _TinyToyModel(nn.Module):
    """Deterministic identity-style model used for probe shape testing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)

    def __call__(self, inputs, cache=None):
        # Return (B, S, vocab) all zeros; populate cache layers with input embeddings.
        x = self.embed(inputs)
        if cache is not None:
            for layer in cache:
                k = x[:, None, :, :]  # (B, 1, S, head)
                v = x[:, None, :, :]
                layer.update_and_fetch(k, v)
        return mx.zeros((inputs.shape[0], inputs.shape[1], 32))


def test_run_chunking_probe_shape():
    from mlx_lm.models.cache import KVCache

    model = _TinyToyModel()
    model.make_cache = lambda: [KVCache()]
    result = run_chunking_probe(model, n_tokens=64, batch_size=1, layer_indices=[0], vocab_size=32)
    assert "schedules" in result
    assert "diffs" in result
    assert "layer_types" in result
    assert (0, "K") in result["diffs"]
    assert (0, "V") in result["diffs"]
    # Schedules at N=64: 1x64, 2x32, 4x16, 8x8 — anything below 8 skipped.
    assert any(name.startswith("1x") for name, _ in result["schedules"])
