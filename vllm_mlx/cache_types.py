from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import mlx.core as mx


@dataclass
class StaticKVData:
    """Representation-agnostic quantized/linearized KV cache segment for one layer."""
    arrays: list[mx.array]   # [q_keys, q_values] — int8 per-tensor quantized
    metadata: dict[str, Any]
    # metadata keys:
    #   class_name: str          — 'KVCache' or 'RotatingKVCache'
    #   layer_index: int         — position in the original live_states list
    #   merge_strategy: str      — 'concatenate' (KVCache) or 'last' (RotatingKVCache)
    #   scales: list[float]      — [key_scale, value_scale]
    #   actual_end: int          — (KVCache) number of tokens in this node's incremental slice
    #   max_size: int            — (RotatingKVCache) ring-buffer max capacity
    #   keep: int                — (RotatingKVCache) attention sink tokens kept
    #   offset: int              — (RotatingKVCache) current write head position


@dataclass
class StaticRecurrentData:
    """Representation-agnostic recurrent state segment for one layer."""
    arrays: Any                          # raw state (list of dicts or raw tensors)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys:
    #   class_name: str          — e.g. 'MambaLayer'
    #   layer_index: int         — position in the original live_states list
    scales: list[list[float]] | None = None  # per-channel int8 scales if quantized
