# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVQuantPolicy: bits_for() dispatch and describe() formatting."""

from vllm_mlx.cache_types import KVQuantPolicy


def test_smart_defaults():
    p = KVQuantPolicy()
    assert p.sliding_bits is None
    assert p.full_bits == 8
    assert p.sliding_override is False
    assert p.full_override is False


def test_bits_for_rotating_returns_sliding_bits():
    p = KVQuantPolicy(sliding_bits=None, full_bits=8)
    assert p.bits_for("RotatingKVCache") is None
    p2 = KVQuantPolicy(sliding_bits=4, full_bits=8)
    assert p2.bits_for("RotatingKVCache") == 4


def test_bits_for_kvcache_returns_full_bits():
    p = KVQuantPolicy(sliding_bits=None, full_bits=8)
    assert p.bits_for("KVCache") == 8


def test_bits_for_batchkvcache_name_match():
    """Any class with 'KVCache' in its name (but not 'RotatingKVCache') uses full_bits."""
    p = KVQuantPolicy(sliding_bits=None, full_bits=4)
    assert p.bits_for("BatchKVCache") == 4


def test_bits_for_recurrent_returns_none():
    p = KVQuantPolicy(sliding_bits=8, full_bits=8)
    assert p.bits_for("MambaCache") is None
    assert p.bits_for("ArraysCache") is None


def test_describe_smart_defaults():
    p = KVQuantPolicy()  # both overrides False
    assert p.describe() == "sliding=bf16, full=q8 (smart defaults)"


def test_describe_full_override_only():
    p = KVQuantPolicy(sliding_bits=None, full_bits=4, full_override=True)
    assert p.describe() == "sliding=bf16, full=q4 (user override)"


def test_describe_both_overridden():
    p = KVQuantPolicy(
        sliding_bits=8, full_bits=4,
        sliding_override=True, full_override=True,
    )
    assert p.describe() == "sliding=q8 (user override), full=q4 (user override)"


def test_describe_sliding_override_only():
    p = KVQuantPolicy(sliding_bits=8, full_bits=8, sliding_override=True)
    assert p.describe() == "sliding=q8 (user override), full=q8"
