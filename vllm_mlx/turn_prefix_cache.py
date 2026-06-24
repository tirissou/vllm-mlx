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
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx.nn.utils import checkpoint
import numpy as np

from vllm_mlx.cache_types import KVLayerSegment, KVConcatSegment, KVRotatingSegment, RecurrentLayerSegment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_CACHE_FORMAT_VERSION = 6


def _kv_segment_to_meta(seg: KVLayerSegment) -> dict:
    """Serialize a typed KVLayerSegment to a JSON-serializable metadata dict."""
    if isinstance(seg, KVRotatingSegment):
        return {
            "class_name": seg.class_name,
            "layer_index": seg.layer_index,
            "merge_strategy": "last",
            "n_tokens": seg.n_tokens,
            "bits": seg.bits,
            "max_size": seg.max_size,
            "keep": seg.keep,
            "offset": seg.offset,
            "_idx": seg.idx,
        }
    # KVConcatSegment (or any other subclass — treat as concat)
    return {
        "class_name": seg.class_name,
        "layer_index": seg.layer_index,
        "merge_strategy": "concatenate",
        "n_tokens": seg.n_tokens,
        "bits": seg.bits,
    }


def _write_kv_segment_arrays(tensors: dict, j: int, kv_item) -> None:
    """Serialize one KV segment's arrays into `tensors`. Dispatches on array type:
    QuantizedArray -> packed/scales/biases tensors; plain mx.array (float, e.g. bf16
    sliding layers with bits=None) -> float tensors.
    """
    from vllm_mlx.kv_cache import QuantizedArray

    keys, values = kv_item.keys, kv_item.values
    if isinstance(keys, QuantizedArray):
        tensors[f"layer_{j}_keys_packed"] = np.array(keys.packed)
        tensors[f"layer_{j}_keys_scales"] = np.array(keys.scales.astype(mx.float32))
        tensors[f"layer_{j}_keys_biases"] = np.array(keys.biases.astype(mx.float32))
        tensors[f"layer_{j}_values_packed"] = np.array(values.packed)
        tensors[f"layer_{j}_values_scales"] = np.array(values.scales.astype(mx.float32))
        tensors[f"layer_{j}_values_biases"] = np.array(values.biases.astype(mx.float32))
    else:
        k = keys.astype(mx.float32) if keys.dtype == mx.bfloat16 else keys
        v = values.astype(mx.float32) if values.dtype == mx.bfloat16 else values
        tensors[f"layer_{j}_keys_float"] = np.array(k)
        tensors[f"layer_{j}_values_float"] = np.array(v)


def _read_kv_segment_arrays(tensors: dict, j: int, item_meta: dict):
    """Reconstruct one KVLayerSegment from saved tensors + metadata. Inverse of
    _write_kv_segment_arrays; picks the quantized or float branch by inspecting
    which tensors are present.
    """
    from vllm_mlx.kv_cache import QuantizedArray

    if f"layer_{j}_keys_packed" in tensors:
        keys = QuantizedArray(
            packed=mx.array(tensors[f"layer_{j}_keys_packed"]),
            scales=mx.array(tensors[f"layer_{j}_keys_scales"]).astype(mx.bfloat16),
            biases=mx.array(tensors[f"layer_{j}_keys_biases"]).astype(mx.bfloat16),
        )
        values = QuantizedArray(
            packed=mx.array(tensors[f"layer_{j}_values_packed"]),
            scales=mx.array(tensors[f"layer_{j}_values_scales"]).astype(mx.bfloat16),
            biases=mx.array(tensors[f"layer_{j}_values_biases"]).astype(mx.bfloat16),
        )
    else:
        keys = mx.array(tensors[f"layer_{j}_keys_float"])
        values = mx.array(tensors[f"layer_{j}_values_float"])
        if item_meta.get("bits") is None:
            keys = keys.astype(mx.bfloat16)
            values = values.astype(mx.bfloat16)
    return _meta_to_kv_segment(keys, values, item_meta)


def _meta_to_kv_segment(keys, values, meta: dict) -> KVLayerSegment:
    """Reconstruct a typed KVLayerSegment from a metadata dict and key/value arrays."""
    merge_strategy = meta.get("merge_strategy", "concatenate")
    if merge_strategy == "last":
        return KVRotatingSegment(
            keys=keys,
            values=values,
            layer_index=meta["layer_index"],
            n_tokens=meta.get("n_tokens", 0),
            bits=meta.get("bits"),
            class_name=meta.get("class_name", "RotatingKVCache"),
            max_size=meta.get("max_size", 0),
            keep=meta.get("keep", 0),
            offset=meta.get("offset", 0),
            idx=meta.get("_idx", meta.get("idx", 0)),
        )
    return KVConcatSegment(
        keys=keys,
        values=values,
        layer_index=meta["layer_index"],
        n_tokens=meta.get("n_tokens", 0),
        bits=meta.get("bits"),
        class_name=meta.get("class_name", "KVCache"),
    )


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
    kv_data: list[KVLayerSegment] | SSDRef | None  # None for root sentinel; full-attention layers only
    recurrent_data: list[RecurrentLayerSegment] | SSDRef | None
    sliding_kv_data: list[KVLayerSegment] | SSDRef | None = None  # sliding-window (rotating); non-cumulative
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
    kv_dtype: str = "int8"  # "bf16" or "int8"
    recurrent_dtype: str = "bf16"  # "none", "fp16", "bf16", or "int8" (per-channel)
    persist_dir: str | None = None  # None = disabled
    ssd_max_gb: float = 0.0  # 0 = disabled
    ssd_dir: str | None = None  # SSD spill directory (defaults to persist_dir/ssd)


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
    """Estimate bytes used by a node's kv_data, sliding_kv_data, and recurrent_data."""
    total = 0
    if isinstance(node.kv_data, list):
        for kv in node.kv_data:
            total += kv.keys.nbytes + kv.values.nbytes
    if isinstance(node.sliding_kv_data, list):
        for kv in node.sliding_kv_data:
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
        self.has_sliding_state: bool = False

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
        sliding_kv_data: list[KVLayerSegment] | None = None,
        recurrent_data: list[RecurrentLayerSegment] | None = None,
        is_system_prompt: bool = False,
        acquire_lock: bool = True,
    ) -> TurnNode:
        rval = self._insert_node(
            parent,
            segment,
            kv_data,
            sliding_kv_data,
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
        sliding_kv_data: list[KVLayerSegment] | None,
        recurrent_data: list[RecurrentLayerSegment] | None,
        is_system_prompt: bool = False,
        acquire_lock: bool = True,
    ) -> TurnNode:
        with self._lock if acquire_lock else nullcontext():
            if recurrent_data:
                self.has_recurrent_state = True
            if sliding_kv_data:
                self.has_sliding_state = True
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
                sliding_kv_data=sliding_kv_data,
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
                parent.sliding_kv_data = None
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
        Sliding-window models: deepest node with in-memory sliding KV data.
        KV-only models: deepest node with non-empty, in-memory kv_data.
        """
        if self.has_recurrent_state:
            for node in reversed(path):
                if isinstance(node.recurrent_data, list) and node.recurrent_data:
                    return node
            return None

        if self.has_sliding_state:
            for node in reversed(path):
                if isinstance(node.sliding_kv_data, list) and node.sliding_kv_data:
                    return node
            return None

        for node in reversed(path):
            if isinstance(node.kv_data, list) and node.kv_data:
                return node
        return None

    def collect_path_data(
        self, node: TurnNode
    ) -> tuple[list[KVLayerSegment], list[RecurrentLayerSegment]]:
        """Walk from root to node and merge KV data per layer.

        Full-attention layers (kv_data): merged via KVConcatSegment.merge_path()
        (incremental concat across the path).
        Sliding-window layers (sliding_kv_data): taken from `node` (the anchor)
        only — non-cumulative, so the deepest copy is authoritative.
        Recurrent data: leaf node only.
        """
        path = self._inorder_path(node)

        kv_by_layer: dict[int, list[KVLayerSegment]] = {}
        for n in path:
            if isinstance(n.kv_data, list):
                for item in n.kv_data:
                    kv_by_layer.setdefault(item.layer_index, []).append(item)

        merged_kv: list[KVLayerSegment] = []
        for li in sorted(kv_by_layer):
            items = kv_by_layer[li]
            merged_kv.append(items[-1].merge_path(items))

        # Sliding-window KV: anchor only (non-cumulative).
        if isinstance(node.sliding_kv_data, list):
            merged_kv.extend(node.sliding_kv_data)

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
            current.sliding_kv_data = None
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

    # ── Disk persistence ───────────────────────────────────────────────────

    def save(self, persist_dir: str) -> None:
        """Save all trie nodes to persist_dir (SQLite index + per-node safetensors + JSON sidecars)."""
        from safetensors.numpy import save_file as st_save

        os.makedirs(persist_dir, exist_ok=True)
        meta = {"version": _CACHE_FORMAT_VERSION, "model_fingerprint": ""}
        with open(os.path.join(persist_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        db_path = os.path.join(persist_dir, "turn_cache_index.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                context_hash INTEGER PRIMARY KEY,
                parent_hash INTEGER,
                token_ids_blob BLOB NOT NULL,
                kv_file_path TEXT NOT NULL,
                recurrent_file_path TEXT,
                sliding_file_path TEXT,
                last_used REAL NOT NULL,
                tokens_since_checkpoint INTEGER NOT NULL,
                is_permanent_checkpoint INTEGER NOT NULL
            )
        """)
        conn.execute("DELETE FROM nodes")

        with self._lock:
            stack = list(self.root.children.values())
            i = 0
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())

                kv_path = os.path.join(persist_dir, f"kv_{i}.safetensors")
                rec_path: str | None = None
                sliding_path: str | None = None

                # Save kv_data: list[KVLayerSegment] — one safetensors file + JSON sidecar per item
                if isinstance(node.kv_data, list) and node.kv_data:
                    tensors: dict[str, np.ndarray] = {}
                    all_meta: list[dict] = []
                    for j, kv_item in enumerate(node.kv_data):
                        _write_kv_segment_arrays(tensors, j, kv_item)
                        all_meta.append(_kv_segment_to_meta(kv_item))
                    tmp = kv_path + ".tmp"
                    st_save(tensors, tmp)
                    os.replace(tmp, kv_path)
                    meta_path_kv = kv_path.replace(".safetensors", "_meta.json")
                    with open(meta_path_kv, "w") as mf:
                        json.dump(all_meta, mf)

                # Save recurrent_data: list[RecurrentLayerSegment]
                if isinstance(node.recurrent_data, list) and node.recurrent_data:
                    rec_path = os.path.join(persist_dir, f"rec_{i}.safetensors")
                    rec_tensors: dict[str, np.ndarray] = {}
                    rec_meta_list: list[dict] = []
                    for j, rec_item in enumerate(node.recurrent_data):
                        arrays = rec_item.arrays
                        if isinstance(arrays, (list, tuple)):
                            for k, arr in enumerate(arrays):
                                if hasattr(arr, "dtype"):
                                    if arr.dtype == mx.bfloat16:
                                        arr = arr.astype(mx.float32)
                                    rec_tensors[f"rec_{j}_arr_{k}"] = np.array(arr)
                        elif isinstance(arrays, dict):
                            for k, arr in enumerate(arrays.get("state", ())):
                                if hasattr(arr, "dtype"):
                                    if arr.dtype == mx.bfloat16:
                                        arr = arr.astype(mx.float32)
                                    rec_tensors[f"rec_{j}_arr_{k}"] = np.array(arr)
                        item_meta = dict(rec_item.metadata)
                        item_meta["_scales"] = rec_item.scales
                        rec_meta_list.append(item_meta)
                    if rec_tensors:
                        tmp = rec_path + ".tmp"
                        st_save(rec_tensors, tmp)
                        os.replace(tmp, rec_path)
                        rec_meta_path = rec_path.replace(".safetensors", "_meta.json")
                        with open(rec_meta_path, "w") as mf:
                            json.dump(rec_meta_list, mf)

                # Save sliding_kv_data: list[KVLayerSegment] (own file)
                if isinstance(node.sliding_kv_data, list) and node.sliding_kv_data:
                    sliding_path = os.path.join(persist_dir, f"sliding_{i}.safetensors")
                    s_tensors: dict[str, np.ndarray] = {}
                    s_meta: list[dict] = []
                    for j, kv_item in enumerate(node.sliding_kv_data):
                        _write_kv_segment_arrays(s_tensors, j, kv_item)
                        s_meta.append(_kv_segment_to_meta(kv_item))
                    tmp = sliding_path + ".tmp"
                    st_save(s_tensors, tmp)
                    os.replace(tmp, sliding_path)
                    s_meta_path = sliding_path.replace(".safetensors", "_meta.json")
                    with open(s_meta_path, "w") as mf:
                        json.dump(s_meta, mf)

                parent_hash = node.parent.context_hash if node.parent is not None else 0
                conn.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        node.context_hash,
                        parent_hash,
                        np.array(node.token_ids, dtype=np.int32).tobytes(),
                        kv_path,
                        rec_path,
                        sliding_path,
                        node.last_used,
                        node.tokens_since_checkpoint,
                        int(node.is_permanent_checkpoint),
                    ),
                )
                i += 1

        conn.commit()
        conn.close()

    def load(self, persist_dir: str) -> None:
        """Load trie from persist_dir. Silently ignores missing/corrupt files."""
        from safetensors.numpy import load_file as st_load

        meta_path = os.path.join(persist_dir, "meta.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("version") != _CACHE_FORMAT_VERSION:
            logger.warning(
                f"[turn_cache] version mismatch: disk={meta.get('version')} "
                f"expected={_CACHE_FORMAT_VERSION} — starting empty"
            )
            return

        db_path = os.path.join(persist_dir, "turn_cache_index.db")
        if not os.path.exists(db_path):
            return

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM nodes").fetchall()
        conn.close()

        # Build hash→node map in one pass
        hash_to_node: dict[int, TurnNode] = {0: self.root}
        for row in rows:
            (
                ctx_hash,
                parent_hash,
                tok_blob,
                kv_path,
                rec_path,
                sliding_path,
                last_used,
                tokens_since,
                is_perm,
            ) = row

            token_ids = list(np.frombuffer(tok_blob, dtype=np.int32))

            # Load kv_data: list[KVLayerSegment]
            kv_data: list[KVLayerSegment] | None = None
            if os.path.exists(kv_path):
                try:
                    tensors = st_load(kv_path)
                    meta_path_kv = kv_path.replace(".safetensors", "_meta.json")
                    if os.path.exists(meta_path_kv):
                        with open(meta_path_kv) as mf:
                            all_meta = json.load(mf)
                        kv_items: list[KVLayerSegment] = [
                            _read_kv_segment_arrays(tensors, j, item_meta)
                            for j, item_meta in enumerate(all_meta)
                        ]
                        kv_data = kv_items if kv_items else None
                except Exception as e:
                    logger.warning(f"[turn_cache] skipping node {ctx_hash}: {e}")
                    continue

            # Load recurrent_data: list[RecurrentLayerSegment]
            recurrent_data: list[RecurrentLayerSegment] | None = None
            if rec_path and os.path.exists(rec_path):
                try:
                    rec_tensors = st_load(rec_path)
                    rec_meta_path = rec_path.replace(".safetensors", "_meta.json")
                    if os.path.exists(rec_meta_path):
                        with open(rec_meta_path) as mf:
                            rec_meta_list = json.load(mf)
                        rec_items: list[RecurrentLayerSegment] = []
                        for j, item_meta in enumerate(rec_meta_list):
                            scales = item_meta.pop("_scales", None)
                            arrays_list: list[mx.array] = []
                            k = 0
                            while f"rec_{j}_arr_{k}" in rec_tensors:
                                arrays_list.append(
                                    mx.array(rec_tensors[f"rec_{j}_arr_{k}"])
                                )
                                k += 1
                            rec_items.append(
                                RecurrentLayerSegment(
                                    arrays=arrays_list,
                                    metadata=item_meta,
                                    scales=scales,
                                )
                            )
                        recurrent_data = rec_items if rec_items else None
                except Exception as e:
                    logger.warning(
                        f"[turn_cache] recurrent load failed for {ctx_hash}: {e}"
                    )

            # Load sliding_kv_data: list[KVLayerSegment]
            sliding_kv_data: list[KVLayerSegment] | None = None
            if sliding_path and os.path.exists(sliding_path):
                try:
                    s_tensors = st_load(sliding_path)
                    s_meta_path = sliding_path.replace(".safetensors", "_meta.json")
                    if os.path.exists(s_meta_path):
                        with open(s_meta_path) as mf:
                            s_meta_list = json.load(mf)
                        s_items: list[KVLayerSegment] = [
                            _read_kv_segment_arrays(s_tensors, j, item_meta)
                            for j, item_meta in enumerate(s_meta_list)
                        ]
                        sliding_kv_data = s_items if s_items else None
                except Exception as e:
                    logger.warning(
                        f"[turn_cache] sliding load failed for {ctx_hash}: {e}"
                    )

            node = TurnNode(
                token_ids=token_ids,
                context_hash=ctx_hash,
                kv_data=kv_data,
                recurrent_data=recurrent_data,
                sliding_kv_data=sliding_kv_data,
                tokens_since_checkpoint=tokens_since,
                is_permanent_checkpoint=bool(is_perm),
                last_used=last_used,
            )
            hash_to_node[ctx_hash] = node

        # Link parent→child INSIDE lock (modifies shared state)
        with self._lock:
            for row in rows:
                ctx_hash, parent_hash = row[0], row[1]
                if ctx_hash not in hash_to_node:
                    continue
                node = hash_to_node[ctx_hash]
                parent = hash_to_node.get(parent_hash, self.root)
                node.parent = parent
                parent.children[ctx_hash] = node
                self._memory_bytes += _node_data_bytes(node)
                if node.is_evictable:
                    heapq.heappush(
                        self._eviction_heap, (node.last_used, id(node), node)
                    )

        for node in hash_to_node.values():
            if node is self.root:
                continue
            if isinstance(node.recurrent_data, list) and node.recurrent_data:
                self.has_recurrent_state = True
            if isinstance(node.sliding_kv_data, list) and node.sliding_kv_data:
                self.has_sliding_state = True

    # ── SSD offloading ─────────────────────────────────────────────────────

    def _tokens_to_node(self, node: TurnNode) -> tuple[int, ...]:
        """Reconstruct the full token sequence from the root down to *node*.

        Walks the parent chain, collecting token_ids from each node, then
        reverses to produce root-to-leaf order.
        """
        segments: list[list[int]] = []
        current = node
        while current.parent is not None:
            segments.append(current.token_ids)
            current = current.parent
        # segments is in leaf-to-root order; reverse to get root-to-leaf
        result: list[int] = []
        for seg in reversed(segments):
            result.extend(seg)
        return tuple(result)

    def _ssd_path(self, node: TurnNode, suffix: str) -> str:
        ssd_dir = self.config.ssd_dir or os.path.join(
            self.config.persist_dir or "/tmp", "ssd"
        )
        os.makedirs(ssd_dir, exist_ok=True)
        return os.path.join(
            ssd_dir,
            f"{node.context_hash & 0xFFFFFFFFFFFFFFFF:016x}_{suffix}.safetensors",
        )

    def _spill_to_ssd(self, node: TurnNode) -> None:
        """Write node's KV (and recurrent if present) to SSD; replace with SSDRef."""
        from safetensors.numpy import save_file as st_save

        if isinstance(node.kv_data, list) and node.kv_data:
            path = self._ssd_path(node, "kv")
            tensors: dict[str, np.ndarray] = {}
            all_meta: list[dict] = []
            for j, kv_item in enumerate(node.kv_data):
                tensors[f"layer_{j}_keys_packed"] = np.array(kv_item.keys.packed)
                tensors[f"layer_{j}_keys_scales"] = np.array(
                    kv_item.keys.scales.astype(mx.float32)
                )
                tensors[f"layer_{j}_keys_biases"] = np.array(
                    kv_item.keys.biases.astype(mx.float32)
                )
                tensors[f"layer_{j}_values_packed"] = np.array(kv_item.values.packed)
                tensors[f"layer_{j}_values_scales"] = np.array(
                    kv_item.values.scales.astype(mx.float32)
                )
                tensors[f"layer_{j}_values_biases"] = np.array(
                    kv_item.values.biases.astype(mx.float32)
                )
                all_meta.append(_kv_segment_to_meta(kv_item))
            tmp = path + ".tmp"
            st_save(tensors, tmp)
            os.replace(tmp, path)
            meta_path = path.replace(".safetensors", "_meta.json")
            with open(meta_path, "w") as mf:
                json.dump(all_meta, mf)
            size = os.path.getsize(path)
            node.kv_data = SSDRef(file_path=path, size_bytes=size)

        if isinstance(node.recurrent_data, list) and node.recurrent_data:
            path = self._ssd_path(node, "rec")
            tensors: dict[str, np.ndarray] = {}
            rec_meta_list: list[dict] = []
            for j, rec_item in enumerate(node.recurrent_data):
                arrays = rec_item.arrays
                if isinstance(arrays, (list, tuple)):
                    for k, arr in enumerate(arrays):
                        if hasattr(arr, "dtype"):
                            if arr.dtype == mx.bfloat16:
                                arr = arr.astype(mx.float32)
                            tensors[f"rec_{j}_arr_{k}"] = np.array(arr)
                elif isinstance(arrays, dict):
                    for k, arr in enumerate(arrays.get("state", ())):
                        if hasattr(arr, "dtype"):
                            if arr.dtype == mx.bfloat16:
                                arr = arr.astype(mx.float32)
                            tensors[f"rec_{j}_arr_{k}"] = np.array(arr)
                item_meta = dict(rec_item.metadata)
                item_meta["_scales"] = rec_item.scales
                rec_meta_list.append(item_meta)
            if tensors:
                tmp = path + ".tmp"
                st_save(tensors, tmp)
                os.replace(tmp, path)
                meta_path = path.replace(".safetensors", "_meta.json")
                with open(meta_path, "w") as mf:
                    json.dump(rec_meta_list, mf)
                size = os.path.getsize(path)
                node.recurrent_data = SSDRef(file_path=path, size_bytes=size)

    def _promote_from_ssd(self, node: TurnNode) -> bool:
        """Load node's KV from SSD back into RAM. Returns False on error."""
        from safetensors.numpy import load_file as st_load

        if isinstance(node.kv_data, SSDRef):
            path = node.kv_data.file_path
            if not os.path.exists(path):
                logger.warning(f"[turn_cache] SSD file missing: {path}")
                return False
            try:
                tensors = st_load(path)
                meta_path = path.replace(".safetensors", "_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as mf:
                        all_meta = json.load(mf)
                    from vllm_mlx.kv_cache import QuantizedArray

                    kv_items: list[KVLayerSegment] = []
                    for j, item_meta in enumerate(all_meta):
                        keys = QuantizedArray(
                            packed=mx.array(tensors[f"layer_{j}_keys_packed"]),
                            scales=mx.array(tensors[f"layer_{j}_keys_scales"]).astype(
                                mx.bfloat16
                            ),
                            biases=mx.array(tensors[f"layer_{j}_keys_biases"]).astype(
                                mx.bfloat16
                            ),
                        )
                        values = QuantizedArray(
                            packed=mx.array(tensors[f"layer_{j}_values_packed"]),
                            scales=mx.array(tensors[f"layer_{j}_values_scales"]).astype(
                                mx.bfloat16
                            ),
                            biases=mx.array(tensors[f"layer_{j}_values_biases"]).astype(
                                mx.bfloat16
                            ),
                        )
                        kv_items.append(
                            _meta_to_kv_segment(keys, values, item_meta)
                        )
                    node.kv_data = kv_items if kv_items else None
                else:
                    node.kv_data = None
            except Exception as e:
                logger.warning(f"[turn_cache] SSD promote failed: {e}")
                return False

        if isinstance(node.recurrent_data, SSDRef):
            path = node.recurrent_data.file_path
            if os.path.exists(path):
                try:
                    rec_tensors = st_load(path)
                    meta_path = path.replace(".safetensors", "_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as mf:
                            rec_meta_list = json.load(mf)
                        rec_items: list[RecurrentLayerSegment] = []
                        for j, item_meta in enumerate(rec_meta_list):
                            scales = item_meta.pop("_scales", None)
                            arrays_list: list[mx.array] = []
                            k = 0
                            while f"rec_{j}_arr_{k}" in rec_tensors:
                                arrays_list.append(
                                    mx.array(rec_tensors[f"rec_{j}_arr_{k}"])
                                )
                                k += 1
                            rec_items.append(
                                RecurrentLayerSegment(
                                    arrays=arrays_list,
                                    metadata=item_meta,
                                    scales=scales,
                                )
                            )
                        node.recurrent_data = rec_items if rec_items else None
                    else:
                        node.recurrent_data = None
                except Exception as e:
                    logger.warning(f"[turn_cache] SSD recurrent promote failed: {e}")
                    node.recurrent_data = None

        return True

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
