# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Gemma4 mlx-lm attention patch."""

import sys
import types
from typing import Any, cast

import mlx.core as mx
import pytest


def _install_fake_gemma4_lm_modules(monkeypatch):
    """Install fake mlx_lm.models.gemma4_text and mlx_lm.models.base modules."""
    rope_call_log: list[dict] = []

    class _IdentityNorm:
        def __call__(self, x: mx.array) -> mx.array:
            return x

    class _RecordingRope:
        """Records the offset passed on each call."""
        def __call__(self, x: mx.array, offset=None) -> mx.array:
            rope_call_log.append({"offset": offset})
            return x

    class DummyCache:
        def __init__(self, offset=0):
            self.offset = offset

        def update_and_fetch(self, keys: mx.array, values: mx.array):
            # Simulate BatchKVCache in-place mutation of offset
            if isinstance(self.offset, mx.array):
                self.offset += 1
            else:
                self.offset += 1
            return keys, values

    class DummyAttention:
        def __init__(self, has_kv: bool = True):
            self.n_heads = 8
            self.n_kv_heads = 8
            self.head_dim = 64
            self.scale = 1.0
            self.has_kv = has_kv
            self.use_k_eq_v = True

            dim = self.n_heads * self.head_dim
            self.q_proj = lambda x: mx.zeros((x.shape[0], x.shape[1], dim), dtype=x.dtype)
            self.k_proj = lambda x: mx.zeros((x.shape[0], x.shape[1], dim), dtype=x.dtype)
            self.v_proj = lambda x: mx.zeros((x.shape[0], x.shape[1], dim), dtype=x.dtype)
            self.o_proj = lambda x: x
            self.q_norm = _IdentityNorm()
            self.k_norm = _IdentityNorm()
            self.v_norm = _IdentityNorm()
            self.rope = _RecordingRope()

    def fake_sdpa(queries, keys, values, **kwargs):
        return queries

    fake_gemma4_text = types.ModuleType("mlx_lm.models.gemma4_text")
    setattr(fake_gemma4_text, "Attention", DummyAttention)

    fake_base = types.ModuleType("mlx_lm.models.base")
    setattr(fake_base, "scaled_dot_product_attention", fake_sdpa)

    monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma4_text", fake_gemma4_text)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", fake_base)

    return DummyAttention, DummyCache, rope_call_log


def test_patch_returns_true_on_success(monkeypatch):
    _install_fake_gemma4_lm_modules(monkeypatch)
    if "vllm_mlx.patches.gemma4_llm" in sys.modules:
        del sys.modules["vllm_mlx.patches.gemma4_llm"]
    from vllm_mlx.patches.gemma4_llm import patch_gemma4_attention_for_batching
    assert patch_gemma4_attention_for_batching() is True


def test_patch_returns_false_when_mlx_lm_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma4_text", None)
    if "vllm_mlx.patches.gemma4_llm" in sys.modules:
        del sys.modules["vllm_mlx.patches.gemma4_llm"]
    from vllm_mlx.patches.gemma4_llm import patch_gemma4_attention_for_batching
    assert patch_gemma4_attention_for_batching() is False


def test_patch_is_idempotent(monkeypatch):
    attention_cls, _, _ = _install_fake_gemma4_lm_modules(monkeypatch)
    if "vllm_mlx.patches.gemma4_llm" in sys.modules:
        del sys.modules["vllm_mlx.patches.gemma4_llm"]
    from vllm_mlx.patches.gemma4_llm import patch_gemma4_attention_for_batching
    assert patch_gemma4_attention_for_batching() is True
    assert patch_gemma4_attention_for_batching() is True
    assert getattr(attention_cls, "_batch_patched", False) is True


def test_patched_call_returns_3_tuple(monkeypatch):
    attention_cls, cache_cls, _ = _install_fake_gemma4_lm_modules(monkeypatch)
    if "vllm_mlx.patches.gemma4_llm" in sys.modules:
        del sys.modules["vllm_mlx.patches.gemma4_llm"]
    from vllm_mlx.patches.gemma4_llm import patch_gemma4_attention_for_batching
    patch_gemma4_attention_for_batching()

    attn = cast(Any, attention_cls())
    x = mx.zeros((2, 7, 512), dtype=mx.float32)
    cache = cache_cls(offset=0)

    result = attn(x, cache=cache)
    assert isinstance(result, tuple)
    assert len(result) == 3
    h, kv, offset = result
    assert isinstance(h, mx.array)
    assert isinstance(kv, tuple) and len(kv) == 2
    assert offset is not None


def test_offset_snapshotted_before_update_and_fetch(monkeypatch):
    """Core regression: queries must use pre-update offset, not post-update."""
    attention_cls, cache_cls, rope_log = _install_fake_gemma4_lm_modules(monkeypatch)
    if "vllm_mlx.patches.gemma4_llm" in sys.modules:
        del sys.modules["vllm_mlx.patches.gemma4_llm"]
    from vllm_mlx.patches.gemma4_llm import patch_gemma4_attention_for_batching
    patch_gemma4_attention_for_batching()

    attn = cast(Any, attention_cls())
    x = mx.zeros((1, 5, 512), dtype=mx.float32)

    # BatchKVCache: offset is an mx.array (per-sequence), starts at 10
    cache = cache_cls(offset=mx.array([10]))

    rope_log.clear()
    attn(x, cache=cache)

    # rope is called twice: once for keys, once for queries
    assert len(rope_log) == 2, f"Expected 2 rope calls, got {len(rope_log)}"

    key_offset, query_offset = rope_log[0]["offset"], rope_log[1]["offset"]

    # Both should be the pre-update value (10), not post-update (11)
    assert int(mx.array(key_offset).flatten()[0]) == 10, (
        f"Key RoPE used wrong offset: {key_offset}"
    )
    assert int(mx.array(query_offset).flatten()[0]) == 10, (
        f"Query RoPE used wrong offset: {query_offset} — in-place mutation not snapshotted"
    )
