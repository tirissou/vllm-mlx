# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class KVLayerSegment:
    """Immutable KV snapshot for one transformer layer, stored in a TurnNode.

    Keys and values can be either:
    - QuantizedArray(packed, scales, biases) for quantized storage (bits is not None)
    - mx.array for float precision storage (bits is None, Track B)
    Use KVLayerSegment.concat() to merge incremental segments along the sequence axis.
    """

    keys: Any  # QuantizedArray(packed, scales, biases) or mx.array
    values: Any  # QuantizedArray(packed, scales, biases) or mx.array
    metadata: dict[str, Any]
    # metadata keys:
    #   class_name: str          — 'KVCache' or 'RotatingKVCache'
    #   layer_index: int         — position in the live cache list
    #   merge_strategy: str      — 'concatenate' (KVCache) or 'last' (RotatingKVCache)
    #   n_tokens: int            — token count represented by this segment
    #   max_size: int            — (RotatingKVCache) ring-buffer capacity
    #   keep: int                — (RotatingKVCache) attention sink tokens kept
    #   offset: int              — (RotatingKVCache) linearized write-head position

    @classmethod
    def concat(cls, layers: list[KVLayerSegment]) -> KVLayerSegment:
        """Concatenate incremental KV segments along the sequence axis (axis=-2)."""
        from vllm_mlx.kv_cache import QuantizedArray

        # All segments in a concat must agree on quantization bits: a single
        # policy writes the whole path in one process, so a mismatch is a bug.
        first_bits = layers[0].metadata.get("bits")
        for l in layers[1:]:
            assert l.metadata.get("bits") == first_bits, (
                f"KVLayerSegment.concat: mismatched bits "
                f"{first_bits!r} vs {l.metadata.get('bits')!r}"
            )

        if isinstance(layers[0].keys, QuantizedArray):
            merged_keys = QuantizedArray(
                packed=mx.concatenate([l.keys.packed for l in layers], axis=-2),
                scales=mx.concatenate([l.keys.scales for l in layers], axis=-2),
                biases=mx.concatenate([l.keys.biases for l in layers], axis=-2),
            )
            merged_values = QuantizedArray(
                packed=mx.concatenate([l.values.packed for l in layers], axis=-2),
                scales=mx.concatenate([l.values.scales for l in layers], axis=-2),
                biases=mx.concatenate([l.values.biases for l in layers], axis=-2),
            )
        else:
            merged_keys = mx.concatenate([l.keys for l in layers], axis=-2)
            merged_values = mx.concatenate([l.values for l in layers], axis=-2)
        meta = dict(layers[-1].metadata)
        meta["n_tokens"] = sum(l.metadata.get("n_tokens", 0) for l in layers)
        meta["bits"] = first_bits
        return cls(keys=merged_keys, values=merged_values, metadata=meta)


@dataclass
class RecurrentLayerSegment:
    """Immutable recurrent state snapshot for one layer, stored in a TurnNode."""

    arrays: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys:
    #   class_name: str          — e.g. 'MambaCache', 'ArraysCache'
    #   layer_index: int         — position in the live cache list
    #   class_ref: type | None   — concrete mlx-lm class for from_state() reconstruction
    #                              (None when loaded from disk — use class_name fallback)
    scales: list[list[float]] | None = None


@dataclass(frozen=True)
class KVQuantPolicy:
    """Per-layer-type KV cache quantization policy.

    bits_for(class_name) returns the int bit-width to use for that class, or
    None for "store float — do not quantize". Class dispatch is purely structural:
    'RotatingKVCache' uses sliding_bits, any other 'KVCache' uses full_bits,
    everything else returns None (recurrent caches are never quantized).
    """

    sliding_bits: int | None = None       # bf16 by default
    full_bits: int | None = 8             # q8 by default
    # Provenance — True iff the value came from a user CLI override (vs. defaulted).
    sliding_override: bool = False
    full_override: bool = False

    def bits_for(self, class_name: str) -> int | None:
        if class_name == "RotatingKVCache":
            return self.sliding_bits
        if "KVCache" in class_name:
            return self.full_bits
        return None

    def describe(self) -> str:
        def _label(bits: int | None) -> str:
            return "bf16" if bits is None else f"q{bits}"

        s = _label(self.sliding_bits)
        f = _label(self.full_bits)
        if self.sliding_override:
            s += " (user override)"
        if self.full_override:
            f += " (user override)"
        suffix = "" if (self.sliding_override or self.full_override) else " (smart defaults)"
        return f"sliding={s}, full={f}{suffix}"
