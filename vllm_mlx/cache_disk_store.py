# SPDX-License-Identifier: Apache-2.0
"""CacheDiskStore — durable backing for TurnCacheManager.

Owns the on-disk persistence format. Used both at runtime
(intra-cache spill/promote) and at process boundaries (save/load).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

# (parent_hash, token_ids_for_this_node) — uniquely identifies a trie node.
NodeKey = tuple[int, tuple[int, ...]]


@dataclass(frozen=True)
class SSDRef:
    """Sentinel that replaces a node's KV payload when spilled to disk."""
    key: NodeKey


@dataclass(frozen=True)
class NodePayload:
    """Full node state for disk write or in-RAM reconstruction."""
    parent_key: "NodeKey | None"
    token_ids: tuple[int, ...]
    n_tokens_cumulative: int
    last_access_ts: float
    kv_layers: list  # list[KVLayerSegment]
    recurrent_layers: list  # list[RecurrentLayerSegment]


@dataclass(frozen=True)
class NodeHeader:
    """Cheap-to-load structural view of a NodePayload — no tensor I/O."""
    parent_key: "NodeKey | None"
    token_ids: tuple[int, ...]
    n_tokens_cumulative: int
    last_access_ts: float
    size_bytes: int
    child_count: int
    layer_bits: tuple
    layer_class_names: tuple
    recurrent_class_paths: tuple
    kv_group_size: int


class CacheDiskError(Exception):
    """Base class for all CacheDiskStore errors that should reach the engine."""


class CacheMissDuringWalk(Exception):
    """Raised in collect_path_data when promote returns None for a node."""
    def __init__(self, node: Any):
        super().__init__("Cache miss during walk")
        self.node = node


class IncompatibleCacheDirError(CacheDiskError):
    def __init__(self, path: str, found: int, expected: int):
        super().__init__(
            f"Cache dir {path!r}: _DISK_FORMAT_VERSION={found}, expected {expected}. "
            f"No automatic migration is supported — wipe the cache dir to proceed."
        )
        self.path = path
        self.found = found
        self.expected = expected


class CachePolicyMismatchError(CacheDiskError):
    def __init__(self, kind: str, expected: str, found: str):
        super().__init__(
            f"Persisted {kind} differs from current config: found={found!r}, "
            f"expected={expected!r}. Mixing in one trie would corrupt assembly."
        )
        self.kind = kind
        self.expected = expected
        self.found = found


class MissingCacheClassError(CacheDiskError):
    def __init__(self, class_path: str):
        super().__init__(
            f"Persisted recurrent class {class_path!r} cannot be resolved via importlib. "
            f"The cache dir was written with a different code version."
        )
        self.class_path = class_path


class DiskStoreFullError(CacheDiskError):
    def __init__(self, needed: int, available: int):
        super().__init__(
            f"Disk LRU full: need {needed} bytes, only {available} available "
            f"after evicting eligible candidates."
        )
        self.needed = needed
        self.available = available


@runtime_checkable
class CacheDiskStore(Protocol):
    """Protocol for durable backing of TurnCacheManager."""

    def write(self, key: NodeKey, payload: NodePayload) -> list[NodeKey]:
        """Persist a node. Returns NodeKeys evicted by disk-LRU as a side effect."""
        ...

    def read(self, key: NodeKey) -> NodePayload | None:
        """Read a node payload. Returns None on miss or corruption."""
        ...

    def read_header(self, key: NodeKey) -> NodeHeader | None:
        """Cheap structural read — no tensor I/O. Used at load()."""
        ...

    def delete(self, key: NodeKey) -> None: ...

    def has(self, key: NodeKey) -> bool: ...

    def touch(self, key: NodeKey) -> None:
        """Refresh disk-LRU recency without rewriting the payload."""
        ...

    def all_keys(self) -> Iterable[NodeKey]: ...

    def get_total_bytes(self) -> int: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

import hashlib
import json
import logging
import os
import struct
from pathlib import Path

import mlx.core as mx

from vllm_mlx.cache_types import KVLayerSegment, RecurrentLayerSegment

_DISK_FORMAT_VERSION = 1
_INDEX_NAME = "_index.json"

logger = logging.getLogger(__name__)


def _node_hash(key: NodeKey) -> str:
    parent_hash, token_ids = key
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<q", parent_hash))
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return h.hexdigest()


def _estimate_payload_bytes(payload: NodePayload) -> int:
    total = 0
    for kv in payload.kv_layers:
        for attr in ("keys", "values"):
            t = getattr(kv, attr)
            if hasattr(t, "packed"):
                total += t.packed.nbytes + t.scales.nbytes + t.biases.nbytes
            else:
                total += t.nbytes
    for rec in payload.recurrent_layers:
        arrays = rec.arrays
        seq = arrays if isinstance(arrays, (list, tuple)) else [arrays]
        for arr in seq:
            if hasattr(arr, "nbytes"):
                total += arr.nbytes
    return total


class FilesystemCacheDiskStore:
    """Concrete CacheDiskStore on a local filesystem.

    Layout::
        cache_dir/
            _index.json            — {_DISK_FORMAT_VERSION, entries, total_bytes}
            <node_hash>.safetensors
            <node_hash>.meta.json
    """

    def __init__(
        self,
        cache_dir: str,
        kv_group_size: int,
        max_bytes: int | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._kv_group_size = kv_group_size
        self._max_bytes = max_bytes
        self._index_path = self._cache_dir / _INDEX_NAME
        self._index: dict[str, Any] = self._load_or_init_index()

    def _load_or_init_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {
                "_DISK_FORMAT_VERSION": _DISK_FORMAT_VERSION,
                "entries": {},
                "total_bytes": 0,
            }
        with open(self._index_path) as f:
            return json.load(f)

    def _flush_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._index, f)
        os.replace(tmp, self._index_path)

    def has(self, key: NodeKey) -> bool:
        return _node_hash(key) in self._index["entries"]

    def _evict_until_fits(self, needed_bytes: int) -> list[NodeKey]:
        if self._max_bytes is None:
            return []
        evicted: list[NodeKey] = []
        while self._index["total_bytes"] + needed_bytes > self._max_bytes:
            candidate = self._pick_eviction_candidate()
            if candidate is None:
                raise DiskStoreFullError(
                    needed=needed_bytes,
                    available=self._max_bytes - self._index["total_bytes"],
                )
            self.delete(candidate)
            evicted.append(candidate)
        return evicted

    def _pick_eviction_candidate(self) -> "NodeKey | None":
        # Leaf-first: only entries with child_count == 0; oldest last_access_ts wins.
        best: "tuple[float, NodeKey] | None" = None
        for entry in self._index["entries"].values():
            if entry["child_count"] != 0:
                continue
            ts = entry["last_access_ts"]
            k: NodeKey = (entry["key"][0], tuple(entry["key"][1]))
            if best is None or ts < best[0]:
                best = (ts, k)
        return best[1] if best is not None else None

    def write(self, key: NodeKey, payload: NodePayload) -> list[NodeKey]:
        needed = _estimate_payload_bytes(payload) + 4096  # JSON overhead.
        evicted = self._evict_until_fits(needed)

        node_hash = _node_hash(key)
        tensor_path = self._cache_dir / f"{node_hash}.safetensors"
        meta_path = self._cache_dir / f"{node_hash}.meta.json"

        tensors: dict[str, mx.array] = {}
        layer_meta: list[dict] = []
        for kv in payload.kv_layers:
            li = kv.metadata["layer_index"]
            if hasattr(kv.keys, "packed"):
                tensors[f"l{li}_k_packed"] = kv.keys.packed
                tensors[f"l{li}_k_scales"] = kv.keys.scales
                tensors[f"l{li}_k_biases"] = kv.keys.biases
                tensors[f"l{li}_v_packed"] = kv.values.packed
                tensors[f"l{li}_v_scales"] = kv.values.scales
                tensors[f"l{li}_v_biases"] = kv.values.biases
            else:
                tensors[f"l{li}_k"] = kv.keys
                tensors[f"l{li}_v"] = kv.values
            layer_meta.append({"kind": "kv", **kv.metadata})

        # mx.save_safetensors appends ".safetensors" automatically;
        # pass a .tmp stem so the written file becomes <stem>.tmp.safetensors,
        # then rename to the final <hash>.safetensors.
        tmp_stem = self._cache_dir / f"{node_hash}.tmp"
        mx.save_safetensors(str(tmp_stem), tensors)
        os.replace(str(tmp_stem) + ".safetensors", tensor_path)

        meta_blob = {
            "parent_key": list(payload.parent_key) if payload.parent_key else None,
            "token_ids": list(payload.token_ids),
            "n_tokens_cumulative": payload.n_tokens_cumulative,
            "last_access_ts": payload.last_access_ts,
            "kv_group_size": self._kv_group_size,
            "layers": layer_meta,
        }
        tmp_meta = str(meta_path) + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta_blob, f)
        os.replace(tmp_meta, meta_path)

        size = tensor_path.stat().st_size + meta_path.stat().st_size
        self._index["entries"][node_hash] = {
            "key": [key[0], list(key[1])],
            "parent_key": meta_blob["parent_key"],
            "token_ids": meta_blob["token_ids"],
            "n_tokens_cumulative": payload.n_tokens_cumulative,
            "last_access_ts": payload.last_access_ts,
            "size_bytes": size,
            "child_count": 0,
            "layer_bits": [m.get("bits") for m in layer_meta],
            "layer_class_names": [m.get("class_name") for m in layer_meta],
            "recurrent_class_paths": [],
            "kv_group_size": self._kv_group_size,
        }
        self._index["total_bytes"] += size

        if payload.parent_key is not None:
            parent_hash = _node_hash(payload.parent_key)
            parent_entry = self._index["entries"].get(parent_hash)
            if parent_entry is not None:
                parent_entry["child_count"] += 1

        self._flush_index()
        return evicted

    def read(self, key: NodeKey) -> NodePayload | None:
        node_hash = _node_hash(key)
        if node_hash not in self._index["entries"]:
            return None
        tensor_path = self._cache_dir / f"{node_hash}.safetensors"
        meta_path = self._cache_dir / f"{node_hash}.meta.json"
        try:
            tensors = mx.load(str(tensor_path))
            with open(meta_path) as f:
                meta_blob = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("Corrupt cache entry %s: %s", tensor_path, e)
            self.delete(key)
            return None

        from vllm_mlx.kv_cache import QuantizedArray

        kv_layers: list[KVLayerSegment] = []
        recurrent_layers: list[RecurrentLayerSegment] = []
        for layer_meta in meta_blob["layers"]:
            if layer_meta["kind"] != "kv":
                continue  # recurrent handled in Task 9.
            li = layer_meta["layer_index"]
            md = {k: v for k, v in layer_meta.items() if k != "kind"}
            if md.get("bits") is None:
                kv_layers.append(KVLayerSegment(
                    keys=tensors[f"l{li}_k"],
                    values=tensors[f"l{li}_v"],
                    metadata=md,
                ))
            else:
                kv_layers.append(KVLayerSegment(
                    keys=QuantizedArray(
                        packed=tensors[f"l{li}_k_packed"],
                        scales=tensors[f"l{li}_k_scales"],
                        biases=tensors[f"l{li}_k_biases"],
                    ),
                    values=QuantizedArray(
                        packed=tensors[f"l{li}_v_packed"],
                        scales=tensors[f"l{li}_v_scales"],
                        biases=tensors[f"l{li}_v_biases"],
                    ),
                    metadata=md,
                ))

        parent_key = (
            (meta_blob["parent_key"][0], tuple(meta_blob["parent_key"][1]))
            if meta_blob["parent_key"] is not None
            else None
        )
        return NodePayload(
            parent_key=parent_key,
            token_ids=tuple(meta_blob["token_ids"]),
            n_tokens_cumulative=meta_blob["n_tokens_cumulative"],
            last_access_ts=meta_blob["last_access_ts"],
            kv_layers=kv_layers,
            recurrent_layers=recurrent_layers,
        )

    def read_header(self, key: NodeKey) -> NodeHeader | None:
        node_hash = _node_hash(key)
        entry = self._index["entries"].get(node_hash)
        if entry is None:
            return None
        parent_key = (
            (entry["parent_key"][0], tuple(entry["parent_key"][1]))
            if entry["parent_key"] is not None
            else None
        )
        return NodeHeader(
            parent_key=parent_key,
            token_ids=tuple(entry["token_ids"]),
            n_tokens_cumulative=entry["n_tokens_cumulative"],
            last_access_ts=entry["last_access_ts"],
            size_bytes=entry["size_bytes"],
            child_count=entry.get("child_count", 0),
            layer_bits=tuple(entry["layer_bits"]),
            layer_class_names=tuple(entry["layer_class_names"]),
            recurrent_class_paths=tuple(entry["recurrent_class_paths"]),
            kv_group_size=entry["kv_group_size"],
        )

    def delete(self, key: NodeKey) -> None:
        node_hash = _node_hash(key)
        entry = self._index["entries"].pop(node_hash, None)
        if entry is None:
            return
        if entry["parent_key"] is not None:
            parent_hash = _node_hash(
                (entry["parent_key"][0], tuple(entry["parent_key"][1]))
            )
            parent_entry = self._index["entries"].get(parent_hash)
            if parent_entry is not None and parent_entry["child_count"] > 0:
                parent_entry["child_count"] -= 1
        self._index["total_bytes"] -= entry["size_bytes"]
        for ext in (".safetensors", ".meta.json"):
            p = self._cache_dir / f"{node_hash}{ext}"
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self._flush_index()

    def touch(self, key: NodeKey) -> None:
        import time
        node_hash = _node_hash(key)
        entry = self._index["entries"].get(node_hash)
        if entry is not None:
            entry["last_access_ts"] = time.time()
            self._flush_index()

    def all_keys(self) -> Iterable[NodeKey]:
        for entry in self._index["entries"].values():
            yield (entry["key"][0], tuple(entry["key"][1]))

    def get_total_bytes(self) -> int:
        return self._index["total_bytes"]

    def close(self) -> None:
        self._flush_index()
