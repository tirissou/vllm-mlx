# SPDX-License-Identifier: Apache-2.0
"""Tests for _build_kv_quant_policy(SchedulerConfig) and its advisory warnings."""

import logging

import pytest

from vllm_mlx.cache_types import KVQuantPolicy
from vllm_mlx.scheduler import (
    SchedulerConfig,
    _build_kv_quant_policy,
    _warn_about_kv_quant_policy,
)


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
        _warn_about_kv_quant_policy(cfg)
        assert _build_kv_quant_policy(cfg) is None
    # The "ignored because" warning names the specific flag that was set,
    # not a generic `{sliding,full}` brace string.
    matching = [r for r in caplog.records if "is ignored because" in r.message]
    assert matching, caplog.records
    assert "--kv-cache-bits-full" in matching[0].getMessage()
    assert "{" not in matching[0].getMessage()


def test_warning_when_full_explicitly_none(caplog):
    """Explicit --kv-cache-bits-full none defeats the memory benefit."""
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=None,
        kv_cache_bits_full_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _warn_about_kv_quant_policy(cfg)
    assert any("dominant memory consumer" in r.message for r in caplog.records)


def test_warning_when_sliding_set_to_int(caplog):
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_sliding=8,
        kv_cache_bits_sliding_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _warn_about_kv_quant_policy(cfg)
    assert any("sensitive to quantization error" in r.message for r in caplog.records)


def test_info_when_full_bits_le_4(caplog):
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
    )
    with caplog.at_level(logging.INFO, logger="vllm_mlx.scheduler"):
        _warn_about_kv_quant_policy(cfg)
    assert any("Qwen3.5 partial-RoPE analogy" in r.message for r in caplog.records)


def test_build_kv_quant_policy_is_pure(caplog):
    """_build_kv_quant_policy must not emit warnings even on aggressive configs.

    Regression for the duplicate-warning issue where calling it once from the
    startup log line and again from _build_prefix_cache produced two copies
    of every advisory warning.
    """
    cfg = _cfg(
        kv_cache_quantization=True,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
        kv_cache_bits_sliding=8,
        kv_cache_bits_sliding_override=True,
    )
    with caplog.at_level(logging.DEBUG, logger="vllm_mlx.scheduler"):
        for _ in range(3):
            _build_kv_quant_policy(cfg)
    assert caplog.records == []


def test_warn_when_both_sides_off_names_both_flags(caplog):
    """The 'ignored because' warning lists every flag the user passed."""
    cfg = _cfg(
        kv_cache_quantization=False,
        kv_cache_bits_full=4,
        kv_cache_bits_full_override=True,
        kv_cache_bits_sliding=8,
        kv_cache_bits_sliding_override=True,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _warn_about_kv_quant_policy(cfg)
    msgs = [r.getMessage() for r in caplog.records if "is ignored because" in r.message]
    assert msgs, caplog.records
    assert "--kv-cache-bits-sliding" in msgs[0]
    assert "--kv-cache-bits-full" in msgs[0]


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


# ── CLI / argparse ────────────────────────────────────────────────────────────

import subprocess
import sys


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_rejects_removed_flag():
    """--kv-cache-quantization-bits is removed; argparse rejects it with a migration message."""
    result = _run_cli("serve", "--help")  # smoke parser builds
    assert result.returncode == 0

    bad = _run_cli(
        "serve",
        "dummy-model",
        "--continuous-batching",
        "--kv-cache-quantization",
        "--kv-cache-quantization-bits", "4",
    )
    assert bad.returncode != 0
    err = bad.stderr + bad.stdout
    assert "--kv-cache-quantization-bits" in err
    assert "was removed" in err
    assert "--kv-cache-bits-sliding" in err
    assert "--kv-cache-bits-full" in err


def test_cli_parser_accepts_new_flags():
    """CLI parses --kv-cache-bits-sliding none and --kv-cache-bits-full 4."""
    from vllm_mlx.cli import build_parser  # exposed parser builder

    parser = build_parser()
    args = parser.parse_args([
        "serve", "dummy-model",
        "--continuous-batching",
        "--kv-cache-quantization",
        "--kv-cache-bits-sliding", "none",
        "--kv-cache-bits-full", "4",
    ])
    # The CLI maps these to (value, override) pairs at SchedulerConfig
    # construction time; just verify the args namespace holds the raw values.
    assert hasattr(args, "kv_cache_bits_sliding")
    assert hasattr(args, "kv_cache_bits_full")
