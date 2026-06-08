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
