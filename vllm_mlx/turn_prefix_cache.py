# SPDX-License-Identifier: Apache-2.0
"""TurnPrefixCache — conversation-turn-level prefix cache trie for hybrid models."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = 1


@dataclass
class Segment:
    role: str
    token_ids: list[int]


@dataclass
class SSDRef:
    file_path: str
    size_bytes: int


@dataclass
class TurnNode:
    token_ids: list[int]
    context_hash: int
    kv_arrays: list[mx.array] | SSDRef | None   # None only for root sentinel
    kv_scales: list[float] | None
    recurrent_state: Any | SSDRef | None         # list of per-layer states, or SSDRef
    tokens_since_checkpoint: int                 # from nearest permanent checkpoint ancestor at creation
    parent: Optional[TurnNode] = field(default=None, repr=False)
    children: dict[int, TurnNode] = field(default_factory=dict)
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    is_permanent_checkpoint: bool = False

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        return self.ref_count == 0 and self.is_leaf


@dataclass
class TurnPrefixCacheConfig:
    checkpoint_stride: int = 512      # tokens between permanent checkpoints; 0 = every node
    max_memory_gb: float = 8.0
    kv_dtype: str = "int8"            # "bf16" or "int8"
    persist_dir: str | None = None    # None = disabled
    ssd_max_gb: float = 0.0           # 0 = disabled
    ssd_dir: str | None = None        # SSD spill directory (defaults to persist_dir/ssd)


def _context_hash(parent_hash: int, token_ids: list[int]) -> int:
    """Context-sensitive hash: same tokens at different trie depths get different hashes."""
    data = struct.pack("<q", parent_hash) + bytes(
        np.array(token_ids, dtype=np.int32).tobytes()
    )
    digest = hashlib.sha256(data).digest()
    return struct.unpack("<q", digest[:8])[0]
