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


def _quantize_kv(
    kv_arrays: list[mx.array],
) -> tuple[list[mx.array], list[float]]:
    """Quantize bf16 KV arrays to int8 with per-tensor scale."""
    quantized: list[mx.array] = []
    scales: list[float] = []
    for arr in kv_arrays:
        arr_f32 = arr.astype(mx.float32)
        max_val = mx.max(mx.abs(arr_f32)).item()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        q = mx.clip(mx.round(arr_f32 / scale), -127, 127).astype(mx.int8)
        quantized.append(q)
        scales.append(scale)
    return quantized, scales


def _dequantize_kv(
    kv_int8: list[mx.array], scales: list[float]
) -> list[mx.array]:
    """Dequantize int8 KV arrays back to bf16."""
    return [
        (arr.astype(mx.float32) * scale).astype(mx.bfloat16)
        for arr, scale in zip(kv_int8, scales)
    ]


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

            # Quantize KV if configured
            stored_kv = kv_arrays
            stored_scales = kv_scales
            if self.config.kv_dtype == "int8" and kv_arrays:
                stored_kv, stored_scales = _quantize_kv(kv_arrays)

            node = TurnNode(
                token_ids=segment.token_ids,
                context_hash=h,
                kv_arrays=stored_kv,
                kv_scales=stored_scales,
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

    # ── Disk persistence ───────────────────────────────────────────────────

    def save(self, persist_dir: str) -> None:
        """Save all trie nodes to persist_dir (SQLite index + per-node safetensors)."""
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

                if isinstance(node.kv_arrays, list) and node.kv_arrays:
                    tensors: dict[str, np.ndarray] = {}
                    for j, arr in enumerate(node.kv_arrays):
                        # Convert to float32 if bfloat16 to avoid numpy conversion issues
                        if arr.dtype == mx.bfloat16:
                            arr = arr.astype(mx.float32)
                        tensors[f"kv_{j}"] = np.array(arr)
                        if node.kv_scales:
                            tensors[f"scale_{j}"] = np.array([node.kv_scales[j]], dtype=np.float32)
                    tmp = kv_path + ".tmp"
                    st_save(tensors, tmp)
                    os.replace(tmp, kv_path)

                if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
                    rec_path = os.path.join(persist_dir, f"rec_{i}.safetensors")
                    tensors: dict[str, np.ndarray] = {}
                    state = node.recurrent_state

                    if isinstance(state, list) and state and isinstance(state[0], dict):
                        # Dict format (_extract_cache_states output)
                        for li, layer_dict in enumerate(state):
                            for j, arr in enumerate(layer_dict.get("state", ())):
                                if hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                tensors[f"ext_{li}_state_{j}"] = np.array(arr)
                            meta = layer_dict.get("meta_state", ())
                            if isinstance(meta, (list, tuple)):
                                for j, s in enumerate(meta):
                                    tensors[f"ext_{li}_meta_{j}"] = np.frombuffer(
                                        str(s).encode(), dtype=np.uint8
                                    )
                            cn = layer_dict.get("class_name", "")
                            tensors[f"ext_{li}_class"] = np.frombuffer(
                                cn.encode(), dtype=np.uint8
                            )
                    else:
                        # Legacy SSM raw-tensor format
                        items = state if isinstance(state, (list, tuple)) else [state]
                        for k, item in enumerate(items):
                            sub = item if isinstance(item, (list, tuple)) else [item]
                            for m, arr in enumerate(sub):
                                if hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                tensors[f"r_{k}_{m}"] = np.array(arr)

                    if tensors:
                        tmp = rec_path + ".tmp"
                        st_save(tensors, tmp)
                        os.replace(tmp, rec_path)

                parent_hash = node.parent.context_hash if node.parent is not None else 0
                conn.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                    (
                        node.context_hash,
                        parent_hash,
                        np.array(node.token_ids, dtype=np.int32).tobytes(),
                        kv_path,
                        rec_path,
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
            (ctx_hash, parent_hash, tok_blob, kv_path, rec_path,
             last_used, tokens_since, is_perm) = row

            token_ids = list(np.frombuffer(tok_blob, dtype=np.int32))

            kv_arrays: list[mx.array] = []
            kv_scales: list[float] = []
            if os.path.exists(kv_path):
                try:
                    tensors = st_load(kv_path)
                    j = 0
                    while f"kv_{j}" in tensors:
                        kv_arrays.append(mx.array(tensors[f"kv_{j}"]))
                        if f"scale_{j}" in tensors:
                            kv_scales.append(float(tensors[f"scale_{j}"][0]))
                        j += 1
                except Exception as e:
                    logger.warning(f"[turn_cache] skipping node {ctx_hash}: {e}")
                    continue

            recurrent_state = None
            if rec_path and os.path.exists(rec_path):
                try:
                    tensors = st_load(rec_path)

                    if any(k.startswith("ext_") for k in tensors):
                        # Dict format
                        import importlib
                        cache_mod = importlib.import_module("mlx_lm.models.cache")

                        layer_indices = sorted({
                            int(k.split("_")[1])
                            for k in tensors
                            if k.startswith("ext_")
                        })
                        state_list = []
                        for li in layer_indices:
                            state_parts = []
                            j = 0
                            while f"ext_{li}_state_{j}" in tensors:
                                state_parts.append(mx.array(tensors[f"ext_{li}_state_{j}"]))
                                j += 1
                            meta_parts = []
                            j = 0
                            while f"ext_{li}_meta_{j}" in tensors:
                                meta_parts.append(
                                    bytes(tensors[f"ext_{li}_meta_{j}"]).decode()
                                )
                                j += 1
                            cn = ""
                            if f"ext_{li}_class" in tensors:
                                cn = bytes(tensors[f"ext_{li}_class"]).decode()
                            state_list.append({
                                "state": tuple(state_parts),
                                "meta_state": tuple(meta_parts) if meta_parts else "",
                                "class_name": cn,
                                "class_ref": getattr(cache_mod, cn, None),
                            })
                        recurrent_state = state_list or None

                    else:
                        # Legacy SSM format
                        max_k = -1
                        for key in tensors.keys():
                            if key.startswith("r_"):
                                k = int(key.split("_")[1])
                                max_k = max(max_k, k)
                        if max_k >= 0:
                            state_list = []
                            for k in range(max_k + 1):
                                layer_list = []
                                m = 0
                                while f"r_{k}_{m}" in tensors:
                                    layer_list.append(mx.array(tensors[f"r_{k}_{m}"]))
                                    m += 1
                                if layer_list:
                                    state_list.append(
                                        layer_list if len(layer_list) > 1 else layer_list[0]
                                    )
                            recurrent_state = (
                                state_list if len(state_list) > 1
                                else (state_list[0] if state_list else None)
                            )
                except Exception as e:
                    logger.warning(f"[turn_cache] recurrent load failed for {ctx_hash}: {e}")

            node = TurnNode(
                token_ids=token_ids,
                context_hash=ctx_hash,
                kv_arrays=kv_arrays or None,
                kv_scales=kv_scales or None,
                recurrent_state=recurrent_state,
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
                    heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))

    # ── SSD offloading ─────────────────────────────────────────────────────

    def _ssd_path(self, node: TurnNode, suffix: str) -> str:
        ssd_dir = self.config.ssd_dir or os.path.join(
            self.config.persist_dir or "/tmp", "ssd"
        )
        os.makedirs(ssd_dir, exist_ok=True)
        return os.path.join(ssd_dir, f"{node.context_hash & 0xFFFFFFFFFFFFFFFF:016x}_{suffix}.safetensors")

    def _spill_to_ssd(self, node: TurnNode) -> None:
        """Write node's KV (and recurrent if present) to SSD; replace with SSDRef."""
        from safetensors.numpy import save_file as st_save

        if isinstance(node.kv_arrays, list) and node.kv_arrays:
            path = self._ssd_path(node, "kv")
            tensors: dict[str, np.ndarray] = {}
            for j, arr in enumerate(node.kv_arrays):
                # Convert to float32 if bfloat16 to avoid numpy conversion issues
                if arr.dtype == mx.bfloat16:
                    arr = arr.astype(mx.float32)
                tensors[f"kv_{j}"] = np.array(arr)
                # Only save scale if it exists and we have enough scales
                if node.kv_scales and j < len(node.kv_scales):
                    tensors[f"scale_{j}"] = np.array([node.kv_scales[j]], dtype=np.float32)
            tmp = path + ".tmp"
            st_save(tensors, tmp)
            os.replace(tmp, path)
            size = os.path.getsize(path)
            node.kv_arrays = SSDRef(file_path=path, size_bytes=size)
            node.kv_scales = None

        if (
            node.recurrent_state is not None
            and not isinstance(node.recurrent_state, SSDRef)
        ):
            path = self._ssd_path(node, "rec")
            tensors = {}
            state = node.recurrent_state
            items = state if isinstance(state, (list, tuple)) else [state]
            for k, item in enumerate(items):
                sub = item if isinstance(item, (list, tuple)) else [item]
                for m, arr in enumerate(sub):
                    # Convert to float32 if bfloat16 to avoid numpy conversion issues
                    if hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                        arr = arr.astype(mx.float32)
                    tensors[f"r_{k}_{m}"] = np.array(arr)
            tmp = path + ".tmp"
            st_save(tensors, tmp)
            os.replace(tmp, path)
            size = os.path.getsize(path)
            node.recurrent_state = SSDRef(file_path=path, size_bytes=size)

    def _promote_from_ssd(self, node: TurnNode) -> bool:
        """Load node's KV from SSD back into RAM. Returns False on error."""
        from safetensors.numpy import load_file as st_load

        if isinstance(node.kv_arrays, SSDRef):
            path = node.kv_arrays.file_path
            if not os.path.exists(path):
                logger.warning(f"[turn_cache] SSD file missing: {path}")
                return False
            try:
                tensors = st_load(path)
                kv_arrays: list[mx.array] = []
                kv_scales: list[float] = []
                j = 0
                while f"kv_{j}" in tensors:
                    kv_arrays.append(mx.array(tensors[f"kv_{j}"]))
                    if f"scale_{j}" in tensors:
                        kv_scales.append(float(tensors[f"scale_{j}"][0]))
                    j += 1
                node.kv_arrays = kv_arrays
                node.kv_scales = kv_scales or None
            except Exception as e:
                logger.warning(f"[turn_cache] SSD promote failed: {e}")
                return False

        if isinstance(node.recurrent_state, SSDRef):
            path = node.recurrent_state.file_path
            if os.path.exists(path):
                try:
                    tensors = st_load(path)
                    # Reconstruct nested structure from flat dict keys r_{k}_{m}
                    state_list = []
                    k = 0
                    max_k = -1
                    for key in tensors.keys():
                        if key.startswith("r_"):
                            parts = key.split("_")
                            if len(parts) >= 3:
                                try:
                                    k_idx = int(parts[1])
                                    max_k = max(max_k, k_idx)
                                except ValueError:
                                    pass

                    for k in range(max_k + 1):
                        layer_list = []
                        m = 0
                        while f"r_{k}_{m}" in tensors:
                            layer_list.append(mx.array(tensors[f"r_{k}_{m}"]))
                            m += 1
                        if layer_list:
                            state_list.append(layer_list if len(layer_list) > 1 else layer_list[0])

                    node.recurrent_state = state_list if len(state_list) > 1 else (state_list[0] if state_list else None)
                except Exception as e:
                    logger.warning(f"[turn_cache] SSD recurrent promote failed: {e}")
                    node.recurrent_state = None

        return True
