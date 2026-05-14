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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_CACHE_FORMAT_VERSION = 2


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
    recurrent_scales: list[list[float]] | None   # per-layer, per-tensor per-channel scales (int8 only)
    parent: Optional[TurnNode] = field(default=None, repr=False)
    children: dict[int, TurnNode] = field(default_factory=dict)
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    is_permanent_checkpoint: bool = False

    @property
    def tokens_since_checkpoint(self) -> int:
        if self.is_permanent_checkpoint:
            return 0
        return len(self.token_ids) + (self.parent.tokens_since_checkpoint if self.parent else 0)

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
        if self.parent: self.parent.touch(tstamp)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids) + (self.parent.n_tokens if self.parent else 0)



@dataclass
class TurnPrefixCacheConfig:
    checkpoint_stride: int = 4000 # tokens between permanent checkpoints; 0 = every node
    max_memory_gb: float = 8.0
    kv_dtype: str = "int8"                      # "bf16" or "int8"
    recurrent_dtype: str = "bf16"                # "none", "fp16", "bf16", or "int8" (per-channel)
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


def _arr_bytes(arr) -> int:
    """Return byte size of an mx.array, or 0 for non-arrays."""
    if not hasattr(arr, "itemsize"):
        return 0
    n = arr.itemsize
    for d in arr.shape:
        n *= d
    return n


def _node_data_bytes(node: TurnNode) -> int:
    """Estimate bytes used by a node's kv_arrays and recurrent_state."""
    kv_total = 0
    rec_total = 0
    if isinstance(node.kv_arrays, list):
        for arrs in node.kv_arrays:
            for kv_item in arrs:
                if isinstance(kv_item, (tuple, list)):
                    # quantized format: [packed, scales, biases]
                    kv_total += sum(_arr_bytes(a) for a in kv_item)
                else:
                    kv_total += _arr_bytes(kv_item)
    if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
        layers = node.recurrent_state
        for arrs in layers:
            for arr in arrs:
                arr: mx.array
                if hasattr(arr, "shape"):
                    nbytes = arr.itemsize
                    for d in arr.shape:
                        nbytes *= d
                    rec_total += nbytes
    return kv_total + rec_total


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


def _quantize_recurrent(
    recurrent_state: Any,
    dtype: str,
) -> tuple[Any, list[list[float]] | None]:
    """Quantize/cast recurrent state to the configured dtype.

    For int8, uses per-channel (last-axis) scales for better precision on small tensors.
    Returns (quantized_state, scales) where scales is None for non-int8 dtypes.
    """
    if recurrent_state is None:
        return None, None

    if dtype == "none":
        return recurrent_state, None

    target = mx.float16 if dtype == "fp16" else mx.bfloat16
    if dtype in ("fp16", "bf16"):
        if isinstance(recurrent_state, list) and recurrent_state and isinstance(recurrent_state[0], dict):
            return [
                {
                    **ld,
                    "state": tuple(arr.astype(target) for arr in ld.get("state", ())),
                }
                for ld in recurrent_state
            ], None
        else:
            items = recurrent_state if isinstance(recurrent_state, (list, tuple)) else [recurrent_state]
            return [
                [
                    arr.astype(target) if hasattr(arr, "astype") else arr
                    for arr in (item if isinstance(item, (list, tuple)) else [item])
                ]
                for item in items
            ], None

    # int8 with per-channel (last-axis) scales
    if isinstance(recurrent_state, list) and recurrent_state and isinstance(recurrent_state[0], dict):
        all_scales: list[list[float]] = []
        quantized = []
        for ld in recurrent_state:
            layer_scales: list[float] = []
            quantized_states = []
            for arr in ld.get("state", ()):
                f32 = arr.astype(mx.float32)
                scales = mx.max(mx.abs(f32), axis=tuple(range(f32.ndim - 1)), keepdims=True)
                scale = mx.where(scales > 0, scales / 127.0, 1.0)
                q = mx.clip(mx.round(f32 / scale), -127, 127).astype(mx.int8)
                quantized_states.append(q)
                layer_scales.append(mx.squeeze(scale).tolist())
            all_scales.append(layer_scales)
            quantized.append({**ld, "state": tuple(quantized_states)})
        return quantized, all_scales
    else:
        all_scales = []
        items = recurrent_state if isinstance(recurrent_state, (list, tuple)) else [recurrent_state]
        quantized = []
        for item in items:
            layer_scales = []
            sub = item if isinstance(item, (list, tuple)) else [item]
            quantized_sub = []
            for arr in sub:
                if hasattr(arr, "astype"):
                    f32 = arr.astype(mx.float32)
                    scales = mx.max(mx.abs(f32), axis=tuple(range(f32.ndim - 1)), keepdims=True)
                    scale = mx.where(scales > 0, scales / 127.0, 1.0)
                    q = mx.clip(mx.round(f32 / scale), -127, 127).astype(mx.int8)
                    quantized_sub.append(q)
                    layer_scales.append(mx.squeeze(scale).tolist())
                else:
                    quantized_sub.append(arr)
            all_scales.append(layer_scales)
            quantized.append(quantized_sub if len(quantized_sub) > 1 else quantized_sub[0])
        return (quantized if len(quantized) > 1 else (quantized[0] if quantized else None)), all_scales


def _dequantize_recurrent(
    recurrent_state: Any,
    scales: list[list[float]] | None,
    original_dtype: str = "bf16",
) -> Any:
    """Dequantize recurrent state back to bf16/fp16.

    If scales is None, just cast to target dtype (for fp16/bf16 storage).
    """
    if recurrent_state is None:
        return None

    target = mx.float16 if original_dtype == "fp16" else mx.bfloat16

    if scales is None:
        # Simple cast (fp16/bf16 storage or no quantization)
        if isinstance(recurrent_state, list) and recurrent_state and isinstance(recurrent_state[0], dict):
            return [
                {
                    **ld,
                    "state": tuple(arr.astype(target) for arr in ld.get("state", ())),
                }
                for ld in recurrent_state
            ]
        else:
            items = recurrent_state if isinstance(recurrent_state, (list, tuple)) else [recurrent_state]
            return [
                [
                    arr.astype(target) if hasattr(arr, "astype") else arr
                    for arr in (item if isinstance(item, (list, tuple)) else [item])
                ]
                for item in items
            ]

    # int8 dequantize with per-channel scales
    if isinstance(recurrent_state, list) and recurrent_state and isinstance(recurrent_state[0], dict):
        dequantized = []
        for ld, layer_scales in zip(recurrent_state, scales):
            dq_states = []
            for arr, scale_list in zip(ld.get("state", ()), layer_scales):
                scale = mx.array(scale_list, dtype=mx.float32)
                # Restore the kept dimensions for broadcasting
                arr_f32 = arr.astype(mx.float32)
                broadcast_shape = [1] * (arr_f32.ndim - 1) + [-1]
                scale = mx.reshape(scale, broadcast_shape)
                dq_states.append((arr_f32 * scale).astype(target))
            dequantized.append({**ld, "state": tuple(dq_states)})
        return dequantized
    else:
        items = recurrent_state if isinstance(recurrent_state, (list, tuple)) else [recurrent_state]
        dequantized = []
        for item, layer_scales in zip(items, scales):
            sub = item if isinstance(item, (list, tuple)) else [item]
            dq_sub = []
            for arr, scale_list in zip(sub, layer_scales):
                if hasattr(arr, "astype"):
                    scale = mx.array(scale_list, dtype=mx.float32)
                    arr_f32 = arr.astype(mx.float32)
                    broadcast_shape = [1] * (arr_f32.ndim - 1) + [-1]
                    scale = mx.reshape(scale, broadcast_shape)
                    dq_sub.append((arr_f32 * scale).astype(target))
                else:
                    dq_sub.append(arr)
            dequantized.append(dq_sub if len(dq_sub) > 1 else dq_sub[0])
        return dequantized if len(dequantized) > 1 else (dequantized[0] if dequantized else None)


class TurnPrefixCache:
    def __init__(self, config: TurnPrefixCacheConfig) -> None:
        self.config = config
        self.root = TurnNode(
            token_ids=[],
            context_hash=0,
            kv_arrays=None,
            kv_scales=None,
            recurrent_state=None,
            recurrent_scales=None,
            parent=None,
            is_permanent_checkpoint=True,  # root acts as checkpoint anchor
        )
        self._lock = threading.Lock()
        self._eviction_heap: list[tuple[float, int, TurnNode]] = []
        self._memory_bytes: int = 0
    
    def _split_cache_arrays(self, cache_states: list[Any], offset: int = 0):
        """
        Process the output from Scheduler._extract_cache_states.
        KV arrays are quantized to int4 (group_size=64) before storage.
        """
        from mlx_lm.models.cache import QuantizedKVCache as _QuantizedKVCache
        _KV_GROUP_SIZE = 64
        _KV_BITS = 4

        kv = []
        kv_indices = []
        recurrent = []
        recurrent_indices = []
        recurrent_cls = None
        logs = []
        for i, state in enumerate(cache_states):
            if "KVCache" in state['class_name']:
                meta = state.get('meta_state')
                raw_state = state['state']
                try:
                    actual_end = int(meta[0]) if meta else raw_state[0].shape[2]
                except (ValueError, TypeError, IndexError):
                    actual_end = raw_state[0].shape[2]
                if len(raw_state) > 2:
                    logs.append("Only keeping first 2 arrays in KVCache arrays.")
                state = raw_state[:2]
                if isinstance(state[0], (list, tuple)):
                    # Already quantized: each element is [packed, scales, biases]
                    state = tuple(
                        tuple(mx.contiguous(comp[:, :, offset:actual_end, :]) for comp in arr)
                        for arr in state
                    )
                else:
                    state = tuple(
                        mx.quantize(mx.contiguous(arr[:, :, offset:actual_end, :]), group_size=_KV_GROUP_SIZE, bits=_KV_BITS)
                        for arr in state
                    )
                kv.append(state)
                kv_indices.append(i)
            else:
                if not recurrent_cls: recurrent_cls = state['class_ref']
                assert recurrent_cls == state['class_ref']
                recurrent.append(state['state'])
                recurrent_indices.append(i)

        # Eval quantized tuples and recurrent arrays
        arrays_to_eval = [comp for arrs in kv for q_tuple in arrs for comp in q_tuple]
        arrays_to_eval += [arr for arrs in recurrent for arr in arrs if isinstance(arr, mx.array)]
        if arrays_to_eval:
            mx.eval(*arrays_to_eval)
        mx.clear_cache()

        n = len(cache_states)
        kv_indices = tuple(kv_indices)
        recurrent_indices = tuple(recurrent_indices)

        def reconstruct(kv, recurrent):
            rval: list[Any] = [None] * n
            assert len(kv) == len(kv_indices)
            assert len(recurrent) == len(recurrent_indices)
            for i, out_i in enumerate(kv_indices):
                q_keys, q_values = kv[i]
                n_tokens = q_keys[0].shape[2]  # packed array; axis=2 is token dim
                rval[out_i] = {
                    "state": kv[i],
                    "meta_state": (str(n_tokens), str(_KV_GROUP_SIZE), str(_KV_BITS)),
                    "class_name": _QuantizedKVCache.__name__,
                    "class_ref": _QuantizedKVCache,
                }
            for i, out_i in enumerate(recurrent_indices):
                rval[out_i] = {
                    "state": recurrent[i],
                    "meta_state": '',
                    "class_name": recurrent_cls.__name__,
                    "class_ref": recurrent_cls,
                }
            assert not any(x is None for x in rval)
            return rval

        if not hasattr(self, "_reassemble_cache_fn"):
            self._reassemble_cache_fn = reconstruct

        for log in logs:
            logger.debug(log)

        return kv, recurrent

    def _retrieve_full_cache(self, node: TurnNode):
        # NOTE: Assuming no SSD for now
        """Reconstruct the cache for a node to pipe to _reconstruct_cache_from_states."""
        with self._lock:
            assert node.recurrent_state is not None
            assert hasattr(self, '_reassemble_cache_fn')
            path = self._inorder_path(node)
            kv = [n.kv_arrays for n in path]
            # Each node's kv_arrays: list of per-layer ((pk,sk,bk),(pv,sv,bv)) quantized tuples.
            # zip(*kv) groups by layer; for each layer, zip(*arrs) groups by key/value;
            # then concatenate each of the 3 components along the token axis (axis=2).
            kv = [
                tuple(
                    tuple(mx.concatenate([t[j] for t in kv_group], axis=2) for j in range(3))
                    for kv_group in zip(*arrs)
                )
                for arrs in zip(*kv)
            ]
            n = kv[0][0][0].shape[2]  # layer 0, q_keys, packed component, token dim
            logger.info(f"Rebuilding cache... {n} tokens")
            recurrent = node.recurrent_state
            rval = self._reassemble_cache_fn(kv, recurrent)
            logger.info(f"MLX Cache size: {mx.get_cache_memory() / (1024 ** 3)} GB")
            return rval

    def _inorder_path(self, node: TurnNode) -> list[TurnNode]:
        path = []
        while node != self.root:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def insert(
        self,
        segments: list[Segment],
        extracted_cache,
        acquire_lock=True
    ) -> TurnNode:
        logger.info("Trying to insert!")

        with self._lock if acquire_lock else nullcontext():
            path = self.match(segments, acquire_lock=False)

            # Check if there is already an exact match... hallelujah
            if path and len(path) == len(segments):
                node = path[-1]
                node.touch()
                return node

            is_system_prompt = not path

            parent = self.root if is_system_prompt else path[-1]
            segment = segments[len(path)]
            logger.info(f"Inserting {len(segment.token_ids)} tokens")
            tokens_since = parent.tokens_since_checkpoint + len(segment.token_ids)
            is_permanent = is_system_prompt or (tokens_since >= self.config.checkpoint_stride)

            kv, recur = self._split_cache_arrays(extracted_cache, parent.n_tokens)

            h = _context_hash(parent.context_hash, segment.token_ids)
            node = TurnNode(
                token_ids=segment.token_ids,
                context_hash=h,
                kv_arrays=kv,
                kv_scales=None,
                recurrent_state=recur,
                recurrent_scales=None,
                parent=parent,
                is_permanent_checkpoint=is_permanent,
            )
            node.touch()
            parent.children[h] = node

            # Prune parent's temp recurrent state if parent just became an inner node
            # and its recurrent state was only a temp (leaf) checkpoint.
            if (
                len(parent.children) == 1          # parent just got its first child
                and not parent.is_permanent_checkpoint
                and parent is not self.root
            ):
                freed = _node_data_bytes(parent)  # KV + recurrent
                parent.recurrent_state = None
                parent.recurrent_scales = None
                freed -= _node_data_bytes(parent)  # subtract KV portion → just recurrent
                self._memory_bytes -= freed

            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            self._evict_if_needed_unlocked()
            logger.info(f"MLX Cache size: {mx.get_cache_memory() / (1024 ** 3)} GB")
            self.visualize()
            return node

    def match(self, segments: list[Segment], acquire_lock=True) -> list[TurnNode]:
        """
        Walk trie matching segments. Returns (path, has_recurrent_at_deepest).
        Does NOT include root node in path.
        """
        with self._lock if acquire_lock else nullcontext():
            path: list[TurnNode] = []
            node = self.root
            for segment in segments:
                h = _context_hash(node.context_hash, segment.token_ids)
                if h not in node.children:
                    logger.info(f"Could not match {segment.role=}")
                    break
                node = node.children[h]
                path.append(node)
            has_recurrent = bool(path) and path[-1].recurrent_state is not None

            return path

    def find_checkpoint_ancestor(self, path: list[TurnNode]) -> TurnNode:
        """Return the deepest node in path with a recurrent state.

        Returns the deepest node with recurrent state stored. Including SSDFRef.
        Fallback to `self.root`.
        """
        # First pass: look for deepest permanent checkpoint with recurrent state
        for node in reversed(path):
            if node.recurrent_state:
                return node
        return self.root

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
            current.recurrent_scales = None

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
            # Lazy deletion: node may no longer be evictable
            if node.last_used != last_used or not node.is_evictable:
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
                    has_scales = node.recurrent_scales is not None

                    if isinstance(state, list) and state and isinstance(state[0], dict):
                        # Dict format (_extract_cache_states output)
                        for li, layer_dict in enumerate(state):
                            for j, arr in enumerate(layer_dict.get("state", ())):
                                # int8 arrays: store as int32 numpy + scales
                                if arr.dtype == mx.int8 and has_scales and li < len(node.recurrent_scales):
                                    tensors[f"ext_{li}_state_{j}"] = np.array(arr, dtype=np.int32)
                                    if j < len(node.recurrent_scales[li]):
                                        tensors[f"ext_{li}_scale_{j}"] = np.array(
                                            node.recurrent_scales[li][j], dtype=np.float32
                                        )
                                elif hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                    tensors[f"ext_{li}_state_{j}"] = np.array(arr)
                                else:
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
                                if hasattr(arr, 'dtype') and arr.dtype == mx.int8 and has_scales and k < len(node.recurrent_scales):
                                    tensors[f"r_{k}_{m}"] = np.array(arr, dtype=np.int32)
                                    if m < len(node.recurrent_scales[k]):
                                        tensors[f"r_{k}_scale_{m}"] = np.array(
                                            node.recurrent_scales[k][m], dtype=np.float32
                                        )
                                elif hasattr(arr, 'dtype') and arr.dtype == mx.bfloat16:
                                    arr = arr.astype(mx.float32)
                                    tensors[f"r_{k}_{m}"] = np.array(arr)
                                else:
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

                    # Collect scales alongside state tensors
                    recurrent_scales: list[list[float]] | None = None

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
                        scales_list: list[list[float]] = []
                        for li in layer_indices:
                            state_parts = []
                            layer_scales: list[float] = []
                            j = 0
                            while f"ext_{li}_state_{j}" in tensors:
                                state_parts.append(mx.array(tensors[f"ext_{li}_state_{j}"]))
                                # Check for per-tensor scale (int8 quantization)
                                if f"ext_{li}_scale_{j}" in tensors:
                                    layer_scales.append(
                                        float(tensors[f"ext_{li}_scale_{j}"])
                                    )
                                j += 1
                            if layer_scales:
                                scales_list.append(layer_scales)
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
                        recurrent_scales = scales_list if scales_list else None
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
                            scales_list: list[list[float]] = []
                            for k in range(max_k + 1):
                                layer_list = []
                                layer_scales: list[float] = []
                                m = 0
                                while f"r_{k}_{m}" in tensors:
                                    layer_list.append(mx.array(tensors[f"r_{k}_{m}"]))
                                    if f"r_{k}_scale_{m}" in tensors:
                                        layer_scales.append(
                                            float(tensors[f"r_{k}_scale_{m}"])
                                        )
                                    m += 1
                                if layer_scales:
                                    scales_list.append(layer_scales)
                                if layer_list:
                                    state_list.append(
                                        layer_list if len(layer_list) > 1 else layer_list[0]
                                    )
                            recurrent_scales = scales_list if scales_list else None
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
                recurrent_scales=recurrent_scales,
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
            tensors: dict[str, np.ndarray] = {}
            state = node.recurrent_state
            has_scales = node.recurrent_scales is not None

            if isinstance(state, list) and state and isinstance(state[0], dict):
                # Dict format (_extract_cache_states output)
                for li, layer_dict in enumerate(state):
                    for j, arr in enumerate(layer_dict.get("state", ())):
                        # int8 arrays: store as int32 numpy + scales
                        if arr.dtype == mx.int8 and has_scales and li < len(node.recurrent_scales):
                            tensors[f"ext_{li}_state_{j}"] = np.array(arr, dtype=np.int32)
                            if j < len(node.recurrent_scales[li]):
                                tensors[f"ext_{li}_scale_{j}"] = np.array(
                                    node.recurrent_scales[li][j], dtype=np.float32
                                )
                        elif hasattr(arr, "dtype") and arr.dtype == mx.bfloat16:
                            tensors[f"ext_{li}_state_{j}"] = np.array(arr.astype(mx.float32))
                        else:
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
                        if hasattr(arr, "dtype") and arr.dtype == mx.int8 and has_scales and k < len(node.recurrent_scales):
                            tensors[f"r_{k}_{m}"] = np.array(arr, dtype=np.int32)
                            if m < len(node.recurrent_scales[k]):
                                tensors[f"r_{k}_scale_{m}"] = np.array(
                                    node.recurrent_scales[k][m], dtype=np.float32
                                )
                        elif hasattr(arr, "dtype") and arr.dtype == mx.bfloat16:
                            tensors[f"r_{k}_{m}"] = np.array(arr.astype(mx.float32))
                        else:
                            tensors[f"r_{k}_{m}"] = np.array(arr)

            if tensors:
                tmp = path + ".tmp"
                st_save(tensors, tmp)
                os.replace(tmp, path)
                size = os.path.getsize(path)
                node.recurrent_state = SSDRef(file_path=path, size_bytes=size)
                node.recurrent_scales = None

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

                    if any(k.startswith("ext_") for k in tensors):
                        # Dict format
                        layer_indices = sorted({
                            int(k.split("_")[1])
                            for k in tensors
                            if k.startswith("ext_")
                        })
                        state_list = []
                        scales_list: list[list[float]] = []
                        for li in layer_indices:
                            state_parts = []
                            layer_scales: list[float] = []
                            j = 0
                            while f"ext_{li}_state_{j}" in tensors:
                                state_parts.append(mx.array(tensors[f"ext_{li}_state_{j}"]))
                                if f"ext_{li}_scale_{j}" in tensors:
                                    layer_scales.append(
                                        float(tensors[f"ext_{li}_scale_{j}"])
                                    )
                                j += 1
                            if layer_scales:
                                scales_list.append(layer_scales)
                            meta_parts = []
                            j = 0
                            while f"ext_{li}_meta_{j}" in tensors:
                                meta_parts.append(
                                    bytes(tensors[f"ext_{li}_meta_{j}"]).decode()
                                )
                                j += 1
                            state_list.append({
                                "state": tuple(state_parts),
                                "meta_state": tuple(meta_parts) if meta_parts else "",
                                "class_name": "",
                                "class_ref": None,
                            })
                        node.recurrent_state = state_list or None
                        node.recurrent_scales = scales_list if scales_list else None
                    else:
                        # Legacy SSM format
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

                        state_list = []
                        scales_list: list[list[float]] = []
                        for k in range(max_k + 1):
                            layer_list = []
                            layer_scales: list[float] = []
                            m = 0
                            while f"r_{k}_{m}" in tensors:
                                layer_list.append(mx.array(tensors[f"r_{k}_{m}"]))
                                if f"r_{k}_scale_{m}" in tensors:
                                    layer_scales.append(
                                        float(tensors[f"r_{k}_scale_{m}"])
                                    )
                                m += 1
                            if layer_scales:
                                scales_list.append(layer_scales)
                            if layer_list:
                                state_list.append(layer_list if len(layer_list) > 1 else layer_list[0])

                        node.recurrent_state = state_list if len(state_list) > 1 else (state_list[0] if state_list else None)
                        node.recurrent_scales = scales_list if scales_list else None
                except Exception as e:
                    logger.warning(f"[turn_cache] SSD recurrent promote failed: {e}")
                    node.recurrent_state = None

        return True

    def get_dequantized_recurrent(self, node: TurnNode) -> Any | None:
        """Return the recurrent state dequantized to bf16/fp16.

        Handles both dict format (_extract_cache_states) and legacy SSM format.
        If the state has scales (int8 quantization), dequantizes on demand.
        If the state is an SSDRef, promotes it first.
        """
        state = node.recurrent_state
        scales = node.recurrent_scales

        # Promote from SSD if needed
        if isinstance(state, SSDRef):
            if not self._promote_from_ssd(node):
                return None
            state = node.recurrent_state
            scales = node.recurrent_scales

        if state is None:
            return None

        # Dequantize if scales are present (int8 storage)
        if scales is not None:
            state = _dequantize_recurrent(state, scales, self.config.recurrent_dtype)

        return state

    def visualize(self, max_depth: int = 10, tokenizer=None) -> str:
        """Return a tree representation of the trie structure.

        Args:
            max_depth: Maximum depth to traverse
            tokenizer: Optional tokenizer to decode tokens (must have decode() method)
        """
        lines = []

        def node_label(node: TurnNode, is_root: bool = False) -> str:
            if is_root:
                return "ROOT"
            ntok = len(node.token_ids)
            ckpt = "✓" if node.is_permanent_checkpoint else " "
            has_kv = "K" if node.kv_arrays else " "
            has_state = "S" if node.recurrent_state else " "
            label = f"[{ntok}t {ckpt}{has_kv}{has_state}]"

            if node.token_ids:
                last_tokens = node.token_ids[-10:] if len(node.token_ids) >= 10 else node.token_ids

                # Try to decode tokens to text
                if tokenizer:
                    try:
                        # Handle both direct tokenizer with decode() and wrapped tokenizer
                        tok = tokenizer
                        if hasattr(tokenizer, "tokenizer"):
                            tok = tokenizer.tokenizer

                        if hasattr(tok, "decode"):
                            text = tok.decode(last_tokens)
                            # Escape newlines and limit length for display
                            text = text.replace("\n", "\\n").replace("\r", "\\r")
                            if len(text) > 80:
                                text = text[:47] + "..."
                            label += f" | {text}"
                        else:
                            # Fallback to token IDs if decode not available
                            tokens_str = ",".join(str(t) for t in last_tokens)
                            label += f" ...{tokens_str}"
                    except Exception:
                        # If decode fails, show token IDs
                        tokens_str = ",".join(str(t) for t in last_tokens)
                        label += f" ...{tokens_str}"
                else:
                    # No tokenizer, show token IDs
                    tokens_str = ",".join(str(t) for t in last_tokens)
                    label += f" ...{tokens_str}"

            return label

        def visit(node: TurnNode, prefix: str = "", is_root: bool = False, depth: int = 0):
            if depth > max_depth:
                return
            lines.append(prefix + node_label(node, is_root))
            children = list(node.children.values())
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                ext = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                visit(child, new_prefix, depth=depth+1)

        visit(self.root, is_root=True)
        summary = f"Nodes: {self._count_nodes()}, Memory: {self._memory_bytes / 1e9:.2f}GB"
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
