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


def _node_data_bytes(node: TurnNode) -> int:
    """Estimate bytes used by a node's kv_arrays and recurrent_state."""
    total = 0
    if isinstance(node.kv_arrays, list):
        for arr in node.kv_arrays:
            # Use shape+dtype to avoid triggering lazy eval
            nbytes = 1
            for d in arr.shape:
                nbytes *= d
            nbytes *= arr.itemsize
            total += nbytes
    if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
        state = node.recurrent_state
        items = state if isinstance(state, (list, tuple)) else [state]
        for item in items:
            sub = item if isinstance(item, (list, tuple)) else [item]
            for arr in sub:
                if hasattr(arr, "shape"):
                    nbytes = 1
                    for d in arr.shape:
                        nbytes *= d
                    nbytes *= arr.itemsize
                    total += nbytes
    return total


class TurnPrefixCache:
    def __init__(self, config: TurnPrefixCacheConfig) -> None:
        self.config = config
        self.root = TurnNode(
            token_ids=[],
            context_hash=0,
            kv_arrays=None,
            kv_scales=None,
            recurrent_state=None,
            tokens_since_checkpoint=0,
            parent=None,
            is_permanent_checkpoint=True,  # root acts as checkpoint anchor
        )
        self._lock = threading.Lock()
        self._eviction_heap: list[tuple[float, int, TurnNode]] = []
        self._memory_bytes: int = 0

    def insert(
        self,
        parent: TurnNode,
        segment: Segment,
        kv_arrays: list[mx.array],
        kv_scales: list[float],
        recurrent_state: Any | None,
        is_system_prompt: bool = False,
    ) -> TurnNode:
        h = _context_hash(parent.context_hash, segment.token_ids)
        with self._lock:
            if h in parent.children:
                node = parent.children[h]
                node.last_used = time.time()
                return node

            if parent.is_permanent_checkpoint:
                tokens_since = len(segment.token_ids)
            else:
                tokens_since = parent.tokens_since_checkpoint + len(segment.token_ids)

            is_permanent = is_system_prompt or (
                self.config.checkpoint_stride == 0
                or tokens_since >= self.config.checkpoint_stride
            )

            node = TurnNode(
                token_ids=segment.token_ids,
                context_hash=h,
                kv_arrays=kv_arrays,
                kv_scales=kv_scales,
                recurrent_state=recurrent_state,
                tokens_since_checkpoint=tokens_since if not is_permanent else 0,
                parent=parent,
                is_permanent_checkpoint=is_permanent,
            )
            parent.children[h] = node

            # Prune parent's temp recurrent state if parent just became an inner node
            # and its recurrent state was only a temp (leaf) checkpoint.
            if (
                len(parent.children) == 1          # parent just got its first child
                and not parent.is_permanent_checkpoint
                and parent is not self.root
            ):
                parent.recurrent_state = None

            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            self._evict_if_needed_unlocked()
            return node

    def match(self, segments: list[Segment]) -> tuple[list[TurnNode], bool]:
        """Walk trie matching segments. Returns (path, has_recurrent_at_deepest)."""
        path: list[TurnNode] = []
        node = self.root
        now = time.time()
        with self._lock:
            for segment in segments:
                h = _context_hash(node.context_hash, segment.token_ids)
                if h not in node.children:
                    break
                node = node.children[h]
                node.last_used = now
                node.ref_count += 1
                path.append(node)
        has_recurrent = bool(path) and path[-1].recurrent_state is not None
        return path, has_recurrent

    def release(self, path: list[TurnNode]) -> None:
        """Decrement ref_count for all nodes in a matched path. Call on request completion."""
        with self._lock:
            for node in path:
                node.ref_count = max(0, node.ref_count - 1)

    def find_checkpoint_ancestor(self, path: list[TurnNode]) -> TurnNode | None:
        """Return the deepest node in path with a recurrent state.

        First searches for the deepest permanent checkpoint with a recurrent state
        (excluding SSDRef). Falls back to the deepest node with any recurrent state
        if no permanent checkpoint is found.
        """
        # First pass: look for deepest permanent checkpoint with recurrent state
        for node in reversed(path):
            if (node.is_permanent_checkpoint and
                node.recurrent_state is not None and
                not isinstance(node.recurrent_state, SSDRef)):
                return node

        # Fallback: if no permanent checkpoint found, return deepest node with recurrent state
        for node in reversed(path):
            if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
                return node

        return None

    def _walk_nodes(self, node: TurnNode) -> list[TurnNode]:
        """DFS walk to collect all nodes in subtree."""
        result = [node]
        for child in node.children.values():
            result.extend(self._walk_nodes(child))
        return result

    def _evict_node(self, node: TurnNode) -> None:
        """Free a node's data and remove it from its parent. Internal — caller holds lock."""
        to_evict = [node]
        while to_evict:
            current = to_evict.pop()
            self._memory_bytes -= _node_data_bytes(current)
            current.kv_arrays = None
            current.kv_scales = None
            current.recurrent_state = None

            parent = current.parent
            if parent is not None and current.context_hash in parent.children:
                del parent.children[current.context_hash]
                if parent.is_evictable and parent is not self.root:
                    to_evict.append(parent)

    def _evict_if_needed_unlocked(self) -> None:
        """Evict LRU leaves until memory is within budget. Caller must hold lock."""
        max_bytes = int(self.config.max_memory_gb * 1024**3)
        while self._memory_bytes > max_bytes and self._eviction_heap:
            _, _, node = heapq.heappop(self._eviction_heap)
            # Lazy deletion: node may no longer be evictable
            if not node.is_evictable:
                continue
            self._evict_node(node)

        # Rebuild heap periodically if bloated
        # (heap can accumulate skipped nodes from lazy deletion)
        if len(self._eviction_heap) > 200:
            nodes = []
            for child in self.root.children.values():
                nodes.extend(self._walk_nodes(child))
            self._eviction_heap = [
                (n.last_used, id(n), n)
                for n in nodes
                if n.is_evictable and n is not self.root
            ]
            heapq.heapify(self._eviction_heap)

    def _evict_if_needed(self) -> None:
        """Evict LRU leaves until memory is within budget."""
        with self._lock:
            self._evict_if_needed_unlocked()
