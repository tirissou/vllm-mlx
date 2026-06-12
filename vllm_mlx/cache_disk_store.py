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
