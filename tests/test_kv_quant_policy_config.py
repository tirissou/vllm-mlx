# SPDX-License-Identifier: Apache-2.0
"""Tests for _build_kv_quant_policy(SchedulerConfig) and its advisory warnings."""

import logging

import pytest

from vllm_mlx.cache_types import KVQuantPolicy
from vllm_mlx.scheduler import SchedulerConfig, _build_kv_quant_policy


def _cfg(**overrides) -> SchedulerConfig:
    return SchedulerConfig(**overrides)


def test_quantization_off_returns_none():
    cfg = _cfg(kv_cache_quantization=False)
    assert _build_kv_quant_policy(cfg) is None


def test_smart_defaults_when_on_without_overrides():
    cfg = _cfg(kv_cache_quantization=True)
    p = _build_kv_quant_policy(cfg)
    assert p == KVQuantPolicy(
        sliding_bits=None,
        full_bits=8,
        sliding_override=False,
        full_override=False,
    )


def test_full_override_applied():
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
    )
    p = _build_kv_quant_policy(cfg)
    assert p.full_bits == 4
    assert p.full_override is True
    assert p.sliding_bits is None
    assert p.sliding_override is False


def test_warning_when_overrides_set_but_quantization_off(caplog):
    cfg = _cfg(
        kv_cache_quantization=False,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        assert _build_kv_quant_policy(cfg) is None
    assert any("is ignored because" in r.message for r in caplog.records)


def test_warning_when_full_explicitly_none(caplog):
    """Explicit --kv-cache-bits-full none defeats the memory benefit."""
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=None,
        kv_cache_bits_full_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _build_kv_quant_policy(cfg)
    assert any("dominant memory consumer" in r.message for r in caplog.records)


def test_warning_when_sliding_set_to_int(caplog):
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_sliding=8,
        kv_cache_bits_sliding_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _build_kv_quant_policy(cfg)
    assert any("sensitive to quantization error" in r.message for r in caplog.records)


def test_info_when_full_bits_le_4(caplog):
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
    )
    with caplog.at_level(logging.INFO, logger="vllm_mlx.scheduler"):
        _build_kv_quant_policy(cfg)
    assert any("Qwen3.5 partial-RoPE analogy" in r.message for r in caplog.records)


def test_unset_vs_explicit_none_distinct():
    """Provenance flag separates 'flag omitted' from 'flag explicitly = none'."""
    unset = _cfg(kv_cache_quantization=True)
    assert unset.kv_cache_bits_full == 8       # smart default applied
    assert unset.kv_cache_bits_full_override is False

    explicit = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=None,
        kv_cache_bits_full_override=True,
    )
    p_unset = _build_kv_quant_policy(unset)
    p_explicit = _build_kv_quant_policy(explicit)
    assert p_unset.full_bits == 8
    assert p_explicit.full_bits is None
    assert p_explicit.full_override is True
