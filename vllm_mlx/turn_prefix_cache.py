# SPDX-License-Identifier: Apache-2.0
"""TurnPrefixCache — conversation-turn-level prefix cache trie for hybrid models."""

from __future__ import annotations

import hashlib
import heapq
import logging
import os
import struct
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx.nn.utils import checkpoint
import numpy as np

from vllm_mlx.cache_types import KVLayerSegment, RecurrentLayerSegment
from vllm_mlx.cache_disk_store import CacheMissDuringWalk, SSDRef

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Segment:
    role: str
    token_ids: list[int]


@dataclass
class TurnNode:
    token_ids: list[int]
    context_hash: int
    kv_data: list[KVLayerSegment] | SSDRef | None  # None for root sentinel
    recurrent_data: list[RecurrentLayerSegment] | SSDRef | None
    parent: Optional[TurnNode] = field(default=None, repr=False)
    children: dict[int, TurnNode] = field(default_factory=dict)
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    is_permanent_checkpoint: bool = False
    tokens_since_checkpoint: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        return self.ref_count == 0 and self.is_leaf

    def touch(self, tstamp=None):
        if tstamp is None:
            tstamp = time.time()
        self.last_used = tstamp
        if self.parent:
            self.parent.touch(tstamp)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids) + (self.parent.n_tokens if self.parent else 0)


@dataclass
class TurnPrefixCacheConfig:
    checkpoint_stride: int = 512  # tokens between permanent checkpoints; 0 = every node
    max_memory_gb: float = 8.0


def _context_hash(parent_hash: int, token_ids: list[int]) -> int:
    """Context-sensitive hash: same tokens at different trie depths get different hashes."""
    data = struct.pack("<q", parent_hash) + bytes(
        np.array(token_ids, dtype=np.int32).tobytes()
    )
    digest = hashlib.sha256(data).digest()
    return struct.unpack("<q", digest[:8])[0]


def _arr_bytes(arr) -> int:
    """Return byte size of an mx.array, or 0 for non-arrays."""
    if not hasattr(arr, "itemsize"):
        return 0
    n = arr.itemsize
    for d in arr.shape:
        n *= d
    return n


def _node_data_bytes(node: TurnNode) -> int:
    """Estimate bytes used by a node's kv_data and recurrent_data."""
    total = 0
    if isinstance(node.kv_data, list):
        for kv in node.kv_data:
            total += kv.keys.nbytes + kv.values.nbytes
    if isinstance(node.recurrent_data, list):
        for rec in node.recurrent_data:
            arrays = rec.arrays
            if isinstance(arrays, (list, tuple)):
                for item in arrays:
                    if isinstance(item, dict):
                        for arr in item.get("state", ()):
                            if hasattr(arr, "itemsize"):
                                total += _arr_bytes(arr)
                    elif hasattr(item, "itemsize"):
                        total += _arr_bytes(item)
            elif isinstance(arrays, dict):
                for arr in arrays.get("state", ()):
                    if hasattr(arr, "itemsize"):
                        total += _arr_bytes(arr)
    return total


class TurnPrefixCache:
    def __init__(self, config: TurnPrefixCacheConfig) -> None:
        self.config = config
        self.root = TurnNode(
            token_ids=[],
            context_hash=0,
            kv_data=None,
            recurrent_data=None,
            parent=None,
            is_permanent_checkpoint=True,
        )
        self._lock = threading.RLock()
        self._eviction_heap: list[tuple[float, int, TurnNode]] = []
        self._memory_bytes: int = 0
        self.has_recurrent_state: bool = False
        self._spill_handler = None
        self._promote_handler = None

    def set_spill_handler(self, handler) -> None:
        """Install the manager's spill callback.

        Called during _evict_if_needed_unlocked. Handler signature:
            (TurnNode) -> bool — True if node kept (spilled), False to drop.
        """
        self._spill_handler = handler

    def set_promote_handler(self, handler) -> None:
        """Install the manager's promote callback.

        Called by collect_path_data. Handler signature:
            (SSDRef) -> tuple[list, list] | None — (kv_layers, recurrent_layers) or None on miss.
        """
        self._promote_handler = handler

    # ── SpillableCache / PrefixCache protocol stubs ─────────────────────────
    # TurnPrefixCache is a low-level trie; the PrefixCache protocol is
    # implemented by TurnCacheManager.  These stubs exist solely so that
    # isinstance(cache, SpillableCache) returns True (runtime_checkable).

    def fetch(self, request) -> None:  # type: ignore[override]
        raise NotImplementedError("Use TurnCacheManager.fetch()")

    def store(self, request, cache: list) -> bool:  # type: ignore[override]
        raise NotImplementedError("Use TurnCacheManager.store()")

    def get_stats(self) -> dict:
        return {"memory_bytes": self._memory_bytes}

    def clear(self) -> None:
        with self._lock:
            self.root.children.clear()
            self._eviction_heap.clear()
            self._memory_bytes = 0

    def on_prefill_checkpoint(
        self, request: Any, processed_tokens: int, extracted_cache: list
    ) -> None:
        pass  # no-op; handled by TurnCacheManager

    def _inorder_path(self, node: TurnNode) -> list[TurnNode]:
        path = []
        while node != self.root:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def insert(
        self,
        parent: TurnNode,
        segment: Segment,
        kv_data: list[KVLayerSegment] | None = None,
        recurrent_data: list[RecurrentLayerSegment] | None = None,
        is_system_prompt: bool = False,
        acquire_lock: bool = True,
    ) -> TurnNode:
        rval = self._insert_node(
            parent,
            segment,
            kv_data,
            recurrent_data,
            is_system_prompt,
            acquire_lock,
        )
        logger.info(self.visualize())
        return rval

    def _insert_node(
        self,
        parent: TurnNode,
        segment: Segment,
        kv_data: list[KVLayerSegment] | None,
        recurrent_data: list[RecurrentLayerSegment] | None,
        is_system_prompt: bool = False,
        acquire_lock: bool = True,
    ) -> TurnNode:
        with self._lock if acquire_lock else nullcontext():
            if recurrent_data:
                self.has_recurrent_state = True
            h = _context_hash(parent.context_hash, segment.token_ids)

            if h in parent.children:
                node = parent.children[h]
                node.touch()
                return node

            tokens_since = parent.tokens_since_checkpoint + len(segment.token_ids)
            is_permanent = (
                is_system_prompt
                or self.config.checkpoint_stride == 0
                or tokens_since >= self.config.checkpoint_stride
            )
            node_tsc = 0 if is_permanent else tokens_since

            node = TurnNode(
                token_ids=segment.token_ids,
                context_hash=h,
                kv_data=kv_data,
                recurrent_data=recurrent_data,
                parent=parent,
                is_permanent_checkpoint=is_permanent,
                tokens_since_checkpoint=node_tsc,
            )
            node.touch()
            parent.children[h] = node

            if (
                len(parent.children) == 1
                and not parent.is_permanent_checkpoint
                and parent is not self.root
            ):
                freed = _node_data_bytes(parent)
                parent.recurrent_data = None
                freed -= _node_data_bytes(parent)
                self._memory_bytes -= freed

            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            self._evict_if_needed_unlocked()
            return node

    def match(
        self, segments: list[Segment], acquire_lock=True
    ) -> tuple[list[TurnNode], bool]:
        """Walk trie matching segments. Returns (path, has_recurrent).

        Increments ref_count for all matched nodes (caller must call release()).
        Does NOT include root node in path.
        """
        with self._lock if acquire_lock else nullcontext():
            path: list[TurnNode] = []
            node = self.root
            tstamp = time.time()
            for segment in segments:
                h = _context_hash(node.context_hash, segment.token_ids)
                if h not in node.children:
                    break
                node = node.children[h]
                node.last_used = tstamp
                node.ref_count += 1
                path.append(node)
            has_recurrent = (
                bool(path)
                and path[-1].recurrent_data is not None
                and not isinstance(path[-1].recurrent_data, SSDRef)
            )
            return path, has_recurrent

    def release(self, path: list[TurnNode]) -> None:
        """Decrement ref_count for all nodes in path; re-add newly evictable ones to heap."""
        with self._lock:
            for node in path:
                node.ref_count = max(0, node.ref_count - 1)
                if node.is_evictable:
                    heapq.heappush(
                        self._eviction_heap, (node.last_used, id(node), node)
                    )

    def find_checkpoint_ancestor(self, path: list[TurnNode]) -> TurnNode | None:
        """Return the deepest node in path that can serve as a prefill resume point.

        Hybrid models: deepest node with real recurrent data.
        KV-only models: deepest node with non-empty, in-memory kv_data.
        """
        if not self.has_recurrent_state:
            for node in reversed(path):
                if isinstance(node.kv_data, list) and node.kv_data:
                    return node
            return None

        for node in reversed(path):
            if isinstance(node.recurrent_data, list) and node.recurrent_data:
                return node
        return None

    def collect_path_data(
        self, node: TurnNode
    ) -> tuple[list[KVLayerSegment], list[RecurrentLayerSegment]]:
        """Walk from root to node and merge KV data per layer.

        KVCache layers: concatenated via KVLayerSegment.concat() (incremental).
        RotatingKVCache layers: deepest node only (full ring buffer).
        Recurrent data: leaf node only.
        """
        path = self._inorder_path(node)

        for n in path:
            kv_is_ref = isinstance(n.kv_data, SSDRef)
            rec_is_ref = isinstance(n.recurrent_data, SSDRef)
            if not (kv_is_ref or rec_is_ref):
                continue
            if self._promote_handler is None:
                raise CacheMissDuringWalk(n)
            ref = n.kv_data if kv_is_ref else n.recurrent_data
            result = self._promote_handler(ref)
            if result is None:
                self._drop_subtree(n)
                raise CacheMissDuringWalk(n)
            kv_layers, rec_layers = result
            if kv_is_ref:
                n.kv_data = kv_layers if kv_layers else None
            if rec_is_ref:
                n.recurrent_data = rec_layers if rec_layers else None
            self._memory_bytes += _node_data_bytes(n)

        kv_by_layer: dict[int, list[KVLayerSegment]] = {}
        for n in path:
            if isinstance(n.kv_data, list):
                for item in n.kv_data:
                    li = item.metadata["layer_index"]
                    kv_by_layer.setdefault(li, []).append(item)

        merged_kv: list[KVLayerSegment] = []
        for li in sorted(kv_by_layer):
            items = kv_by_layer[li]
            if items[0].metadata.get("merge_strategy", "concatenate") == "last":
                merged_kv.append(items[-1])
            else:
                merged_kv.append(KVLayerSegment.concat(items))

        leaf = path[-1] if path else None
        recurrent: list[RecurrentLayerSegment] = (
            leaf.recurrent_data
            if leaf and isinstance(leaf.recurrent_data, list)
            else []
        )
        return merged_kv, recurrent

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
            current.kv_data = None
            current.recurrent_data = None

            parent = current.parent
            if parent is not None and current.context_hash in parent.children:
                del parent.children[current.context_hash]
                if parent.is_evictable and parent is not self.root:
                    to_evict.append(parent)

    def _evict_if_needed_unlocked(self) -> None:
        """Evict LRU leaves until memory is within budget. Caller must hold lock."""
        max_bytes = int(self.config.max_memory_gb * 1024**3)
        while self._memory_bytes > max_bytes and self._eviction_heap:
            last_used, _, node = heapq.heappop(self._eviction_heap)
            # Lazy deletion: skip if node was touched after this entry was pushed,
            # or is no longer a leaf/unpinned.
            if node.last_used > last_used or not node.is_evictable:
                continue
            if self._spill_handler is not None:
                kept = self._spill_handler(node)
                if kept:
                    # Manager's _on_spill set kv_data/recurrent_data to SSDRef
                    # and adjusted _memory_bytes. Continue evicting.
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

    def visualize(self, max_depth: int = 20) -> str:
        """Return a compact tree representation of the trie structure.

        Pinned nodes (ref_count > 0) are highlighted in red via ANSI codes.
        """
        lines = []

        def node_label(node: TurnNode, is_root: bool = False) -> str:
            if is_root:
                return "ROOT"
            ntok = len(node.token_ids)
            ckpt = "✓" if node.is_permanent_checkpoint else " "
            has_kv = "K" if isinstance(node.kv_data, list) and node.kv_data else " "
            has_state = (
                "S"
                if isinstance(node.recurrent_data, list) and node.recurrent_data
                else " "
            )
            label = f"[{ntok}t {ckpt}{has_kv}{has_state}]"
            if node.ref_count > 0:
                label = f"\033[31m{label}\033[0m"
            return label

        def visit(
            node: TurnNode,
            self_prefix: str = "",
            child_prefix: str = "",
            is_root: bool = False,
            depth: int = 0,
        ):
            if depth > max_depth:
                return
            lines.append(self_prefix + node_label(node, is_root))
            children = list(node.children.values())
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                if is_last:
                    visit(child, child_prefix + "└ ", child_prefix + "  ", depth=depth + 1)
                else:
                    visit(child, child_prefix + "├ ", child_prefix + "│ ", depth=depth + 1)

        visit(self.root, is_root=True)
        summary = (
            f"Nodes: {self._count_nodes()}, Memory: {self._memory_bytes / 1e9:.2f}GB"
        )
        return "\n".join(lines) + "\n" + summary

    def walk_all_nodes_preorder(self) -> list[TurnNode]:
        """Yield every non-root node in preorder (parent before children)."""
        result = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root:
                result.append(n)
            # Reverse children so the preorder is left-to-right.
            stack.extend(reversed(list(n.children.values())))
        return result

    def insert_prebuilt(
        self,
        parent_key,
        token_ids,
        last_access_ts: float,
        n_tokens_cumulative: int,
        kv_data,
        recurrent_data,
    ) -> TurnNode:
        """Insert a node restored from disk with explicit state.

        ``parent_key`` is ``(parent_hash, parent_token_ids)`` or ``None``
        to attach directly to root.  The matching parent TurnNode is found
        by walking children by hash (cheap when restore is topo-sorted).
        """
        with self._lock:
            if parent_key is None:
                parent_hash = self.root.context_hash
            else:
                parent_hash = parent_key[0]
            parent = self._find_node_by_hash(self.root, parent_hash)
            if parent is None:
                raise ValueError(
                    f"insert_prebuilt: parent with hash {parent_hash} not yet restored"
                )

            token_ids = list(token_ids)
            h = _context_hash(parent.context_hash, token_ids)
            if h in parent.children:
                # Idempotent re-insert (e.g. duplicate restore); return existing.
                return parent.children[h]

            node = TurnNode(
                token_ids=token_ids,
                context_hash=h,
                kv_data=kv_data,
                recurrent_data=recurrent_data,
                parent=parent,
                is_permanent_checkpoint=False,
                tokens_since_checkpoint=0,
            )
            node.last_used = last_access_ts
            parent.children[h] = node
            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            return node

    def _find_node_by_hash(self, root: TurnNode, target_hash: int) -> TurnNode | None:
        """DFS search for a node with the given context_hash."""
        if root.context_hash == target_hash:
            return root
        for child in root.children.values():
            found = self._find_node_by_hash(child, target_hash)
            if found is not None:
                return found
        return None

    def _drop_node(self, node: TurnNode) -> None:
        """Public-by-convention: manager calls this when promote fails or disk LRU evicts a node."""
        with self._lock:
            self._evict_node(node)

    def _drop_subtree(self, node: TurnNode) -> None:
        """Drop ``node`` and every descendant, accounting their bytes.

        ``_evict_node`` only walks UP toward the root, so dropping a path-interior
        node with descendants would leak their bytes from ``_memory_bytes`` and
        leave the descendant Python objects unreachable.
        """
        with self._lock:
            for descendant in self._walk_nodes(node):
                self._memory_bytes -= _node_data_bytes(descendant)
                descendant.kv_data = None
                descendant.recurrent_data = None
            parent = node.parent
            if parent is not None and node.context_hash in parent.children:
                del parent.children[node.context_hash]

    def _count_nodes(self) -> int:
        """Count total nodes in trie."""
        count = 1  # root

        def visit(node):
            nonlocal count
            for child in node.children.values():
                count += 1
                visit(child)

        visit(self.root)
        return count
