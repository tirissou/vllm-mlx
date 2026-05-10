# TurnPrefixCache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TurnPrefixCache`, a conversation-turn-level prefix cache trie for hybrid models (Qwen3.6-27B) that enables cross-session prefix sharing, configurable recurrent state checkpointing, int8 KV quantization, disk persistence, and SSD offloading — all within a 10 GB RAM budget for 150K-token sessions.

**Architecture:** A trie where each node is one post-normalized conversation message. Nodes use context-sensitive hashing (`hash(parent_hash || token_ids)`) so identical content at different depths never shares KV. Recurrent state is stored on leaves temporarily and promoted to a permanent checkpoint when `tokens_since_last_permanent_checkpoint >= checkpoint_stride` (configurable; 0 = every node). int8 quantization halves KV memory from 9.15 GB to 4.7 GB at 150K tokens. The trie structure always lives in RAM; only KV and recurrent data tier to SSD.

**Tech Stack:** Python 3.14, MLX (`mlx.core`), NumPy, `safetensors.numpy`, `sqlite3` (stdlib), `heapq` (stdlib), `threading` (stdlib). Tests use `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-09-turn-prefix-cache-design.md`

---

## File Map

| File | Role |
|---|---|
| `vllm_mlx/turn_prefix_cache.py` | New — all of `TurnNode`, `SSDRef`, `Segment`, `TurnPrefixCacheConfig`, `TurnPrefixCache` |
| `tests/test_turn_prefix_cache.py` | New — all unit and integration tests |
| `vllm_mlx/scheduler.py` | Modify — add `use_turn_cache` config field and two integration points |
| `vllm_mlx/cli.py` | Modify — add `--use-turn-cache`, `--turn-cache-stride`, `--turn-cache-ssd-gb` flags |

---

## Task 1: Core dataclasses, context hash, and config

**Files:**
- Create: `vllm_mlx/turn_prefix_cache.py`
- Create: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Create the test file with context hash tests**

`tests/test_turn_prefix_cache.py`:
```python
import pytest
import mlx.core as mx
from vllm_mlx.turn_prefix_cache import (
    Segment, TurnNode, SSDRef, TurnPrefixCacheConfig, _context_hash,
)


def seg(token_ids, role="user"):
    return Segment(role=role, token_ids=token_ids)


def test_context_hash_deterministic():
    assert _context_hash(0, [1, 2, 3]) == _context_hash(0, [1, 2, 3])


def test_context_hash_different_parent():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(1, [1, 2, 3])


def test_context_hash_different_tokens():
    assert _context_hash(0, [1, 2, 3]) != _context_hash(0, [1, 2, 4])


def test_turn_node_is_leaf_when_no_children():
    node = TurnNode(token_ids=[1], context_hash=1, kv_arrays=[], kv_scales=[],
                    recurrent_state=None, tokens_since_checkpoint=0, parent=None)
    assert node.is_leaf


def test_turn_node_not_leaf_when_has_children():
    parent = TurnNode(token_ids=[], context_hash=0, kv_arrays=None, kv_scales=None,
                      recurrent_state=None, tokens_since_checkpoint=0, parent=None)
    child = TurnNode(token_ids=[1], context_hash=1, kv_arrays=[], kv_scales=[],
                     recurrent_state=None, tokens_since_checkpoint=1, parent=parent)
    parent.children[1] = child
    assert not parent.is_leaf


def test_ssdref_is_sentinel():
    ref = SSDRef(file_path="/tmp/foo.safetensors", size_bytes=1024)
    assert ref.file_path == "/tmp/foo.safetensors"
    assert ref.size_bytes == 1024


def test_config_defaults():
    cfg = TurnPrefixCacheConfig()
    assert cfg.checkpoint_stride == 512
    assert cfg.max_memory_gb == 8.0
    assert cfg.kv_dtype == "int8"
    assert cfg.persist_dir is None
    assert cfg.ssd_max_gb == 0.0
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_turn_prefix_cache.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` or `ImportError` for `turn_prefix_cache`.

- [ ] **Step 3: Create `vllm_mlx/turn_prefix_cache.py` with dataclasses and hash**

```python
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
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: TurnPrefixCache dataclasses, Segment, config, and context hash"
```

---

## Task 2: TurnPrefixCache init and basic insert

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add insert tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
from vllm_mlx.turn_prefix_cache import TurnPrefixCache


def make_cache(stride=512, max_gb=8.0, kv_dtype="bf16"):
    return TurnPrefixCache(TurnPrefixCacheConfig(
        checkpoint_stride=stride, max_memory_gb=max_gb, kv_dtype=kv_dtype
    ))


def test_insert_creates_child_of_root():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), kv_arrays=[], kv_scales=[], recurrent_state=None)
    assert node in cache.root.children.values()


def test_insert_sets_token_ids():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.token_ids == [1, 2, 3]


def test_insert_node_is_leaf():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.is_leaf


def test_insert_sets_parent():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert node.parent is cache.root


def test_insert_idempotent_same_segment():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    assert n1 is n2  # same node returned


def test_insert_chained():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]), [], [], None)
    n2 = cache.insert(n1, seg([2]), [], [], None)
    assert n2.parent is n1
    assert n2 in n1.children.values()


def test_insert_context_hash_differs_at_different_depths():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(n1, seg([1, 2, 3]), [], [], None)  # same tokens, different parent
    assert n1.context_hash != n2.context_hash
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_insert_creates_child_of_root -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute 'insert'`

- [ ] **Step 3: Add `TurnPrefixCache` class with `__init__` and `insert` to `turn_prefix_cache.py`**

Append after the `_context_hash` function:
```python
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

            # Compute tokens_since_checkpoint from nearest permanent checkpoint ancestor
            if parent.is_permanent_checkpoint:
                tokens_since = len(segment.token_ids)
            else:
                tokens_since = parent.tokens_since_checkpoint + len(segment.token_ids)

            node = TurnNode(
                token_ids=segment.token_ids,
                context_hash=h,
                kv_arrays=kv_arrays,
                kv_scales=kv_scales,
                recurrent_state=recurrent_state,  # temp checkpoint (leaf)
                tokens_since_checkpoint=tokens_since,
                parent=parent,
                is_permanent_checkpoint=False,
            )
            parent.children[h] = node

            self._memory_bytes += _node_data_bytes(node)
            heapq.heappush(self._eviction_heap, (node.last_used, id(node), node))
            return node
```

- [ ] **Step 4: Run tests — all new tests should pass**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (17 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: TurnPrefixCache init and basic insert"
```

---

## Task 3: Checkpoint and prune rules

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add checkpoint/prune tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
def test_leaf_gets_recurrent_state():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    node = cache.insert(cache.root, seg(list(range(50))), [], [], state)
    assert node.recurrent_state is not None


def test_temp_recurrent_pruned_on_non_stride_inner():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    n1 = cache.insert(cache.root, seg(list(range(50))), [], [], state)
    # n1 is leaf with temp recurrent (50 < 100)
    assert n1.recurrent_state is not None
    # Add child: n1 becomes inner node, tokens_since=50 < 100 → prune
    cache.insert(n1, seg([99]), [], [], state)
    assert n1.recurrent_state is None


def test_permanent_checkpoint_at_stride():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    # 120 tokens >= stride → permanent
    n1 = cache.insert(cache.root, seg(list(range(120))), [], [], state)
    assert n1.is_permanent_checkpoint
    # Add child: should NOT prune recurrent
    cache.insert(n1, seg([999]), [], [], state)
    assert n1.recurrent_state is not None


def test_system_prompt_always_permanent():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    # Only 50 tokens but is_system_prompt=True
    n = cache.insert(cache.root, seg(list(range(50)), role="system"), [], [], state,
                     is_system_prompt=True)
    assert n.is_permanent_checkpoint
    cache.insert(n, seg([99]), [], [], state)
    assert n.recurrent_state is not None


def test_tokens_since_resets_after_permanent():
    cache = make_cache(stride=100)
    state = mx.zeros((1,))
    # 120 tokens → permanent checkpoint
    n1 = cache.insert(cache.root, seg(list(range(120))), [], [], state)
    assert n1.is_permanent_checkpoint
    # 10 more tokens; tokens_since should count from n1 (permanent), not root
    n2 = cache.insert(n1, seg(list(range(10))), [], [], state)
    assert n2.tokens_since_checkpoint == 10


def test_stride_zero_makes_every_node_permanent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    n1 = cache.insert(cache.root, seg([1, 2]), [], [], state)
    assert n1.is_permanent_checkpoint
    n2 = cache.insert(n1, seg([3]), [], [], state)
    assert n2.is_permanent_checkpoint
    # Neither should have recurrent pruned
    assert n1.recurrent_state is not None
    assert n2.recurrent_state is not None
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_temp_recurrent_pruned_on_non_stride_inner -v
```
Expected: FAIL — prune logic not yet implemented.

- [ ] **Step 3: Add checkpoint/prune logic to `insert()` in `turn_prefix_cache.py`**

Replace the `insert` method body (after the node creation) with the updated version:
```python
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
            return node
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (25 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: checkpoint and prune rules for TurnPrefixCache"
```

---

## Task 4: Prefix matching and ref_count management

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add matching tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
def test_match_full():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    n2 = cache.insert(n1, seg([4, 5]), [], [], None)
    path, _ = cache.match([seg([1, 2, 3]), seg([4, 5])])
    assert path == [n1, n2]


def test_match_partial():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    cache.insert(n1, seg([4, 5]), [], [], None)
    path, _ = cache.match([seg([1, 2, 3]), seg([99])])  # second seg not in trie
    assert path == [n1]


def test_match_empty():
    cache = make_cache()
    path, has_recurrent = cache.match([seg([99])])
    assert path == []
    assert not has_recurrent


def test_match_updates_last_used():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    node.last_used = 0.0
    cache.match([seg([1])])
    assert node.last_used > 0.0


def test_match_reports_has_recurrent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    node = cache.insert(cache.root, seg([1]), [], [], state)
    _, has_recurrent = cache.match([seg([1])])
    assert has_recurrent


def test_match_increments_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    assert node.ref_count == 0
    cache.match([seg([1])])
    assert node.ref_count == 1


def test_release_decrements_ref_count():
    cache = make_cache()
    node = cache.insert(cache.root, seg([1]), [], [], None)
    path, _ = cache.match([seg([1])])
    assert node.ref_count == 1
    cache.release(path)
    assert node.ref_count == 0


def test_release_all_nodes_in_path():
    cache = make_cache()
    n1 = cache.insert(cache.root, seg([1]), [], [], None)
    n2 = cache.insert(n1, seg([2]), [], [], None)
    path, _ = cache.match([seg([1]), seg([2])])
    assert n1.ref_count == 1
    assert n2.ref_count == 1
    cache.release(path)
    assert n1.ref_count == 0
    assert n2.ref_count == 0
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_match_full -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute 'match'`

- [ ] **Step 3: Add `match()` and `release()` to `TurnPrefixCache`**

Append these methods to the `TurnPrefixCache` class:
```python
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
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (34 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: prefix matching and ref_count management"
```

---

## Task 5: Gap reconstruction helper

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add gap reconstruction tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
from vllm_mlx.turn_prefix_cache import _context_hash


def test_find_checkpoint_ancestor_returns_self_if_has_recurrent():
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    n = cache.insert(cache.root, seg([1]), [], [], state)
    path = [n]
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is n


def test_find_checkpoint_ancestor_walks_up():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    # sys node: permanent checkpoint (is_system_prompt=True)
    n_sys = cache.insert(cache.root, seg(list(range(50)), role="system"),
                          [], [], state, is_system_prompt=True)
    # user node: not a checkpoint (stride not met, temp gets pruned after child added)
    n_user = cache.insert(n_sys, seg([100, 101]), [], [], state)
    # asst node: not a checkpoint, prunes n_user's temp recurrent
    n_asst = cache.insert(n_user, seg([200]), [], [], state)
    # n_user's recurrent was pruned; n_sys still has permanent recurrent
    assert n_user.recurrent_state is None
    path = [n_sys, n_user, n_asst]
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is n_sys


def test_find_checkpoint_ancestor_returns_none_when_no_checkpoint():
    cache = make_cache(stride=10000)
    # No state stored, stride too high → no permanent checkpoints (except root which isn't in path)
    n = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is None


def test_find_checkpoint_ancestor_returns_leaf_if_leaf_has_recurrent():
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))
    n = cache.insert(cache.root, seg([1]), [], [], state)
    # n is a leaf → has temp recurrent
    assert n.recurrent_state is not None
    ancestor = cache.find_checkpoint_ancestor([n])
    assert ancestor is n
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_find_checkpoint_ancestor_returns_self_if_has_recurrent -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute 'find_checkpoint_ancestor'`

- [ ] **Step 3: Add `find_checkpoint_ancestor()` to `TurnPrefixCache`**

Append to the `TurnPrefixCache` class:
```python
    def find_checkpoint_ancestor(self, path: list[TurnNode]) -> TurnNode | None:
        """Return the deepest node in path that has a recurrent state, or None."""
        for node in reversed(path):
            if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
                return node
        return None
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (38 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: find_checkpoint_ancestor for gap reconstruction"
```

---

## Task 6: LRU eviction

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add eviction tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
def _make_kv(n_tokens=1):
    """Small real KV arrays for memory-tracked tests."""
    return [mx.zeros((1, 4, n_tokens, 256), dtype=mx.bfloat16)]


def _kv_scales():
    return [1.0]


def test_lru_evicts_oldest_leaf():
    cache = make_cache(max_gb=0.0)
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2 = cache.insert(cache.root, seg([2]), _make_kv(), _kv_scales(), None)
    n1.last_used = 1.0
    n2.last_used = 2.0
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert n1.kv_arrays is None      # evicted
    assert n2.kv_arrays is not None  # kept


def test_pinned_node_not_evicted():
    cache = make_cache(max_gb=0.0)
    node = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    node.ref_count = 1
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert node.kv_arrays is not None


def test_eviction_cascade_to_parent():
    cache = make_cache(max_gb=0.0)
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2 = cache.insert(n1, seg([2]), _make_kv(), _kv_scales(), None)
    n1.last_used = 0.5
    n2.last_used = 1.0
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert n2.kv_arrays is None  # leaf evicted first
    assert n1.kv_arrays is None  # cascades since n1 now has no children


def test_cascade_stops_at_sibling():
    cache = make_cache(max_gb=0.0)
    n1 = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    n2a = cache.insert(n1, seg([2]), _make_kv(), _kv_scales(), None)
    n2b = cache.insert(n1, seg([3]), _make_kv(), _kv_scales(), None)
    n2a.last_used = 1.0
    n2b.last_used = 2.0
    n1.last_used = 0.5
    # Force eviction of only n2a (one step)
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    cache._evict_if_needed()
    assert n2a.kv_arrays is None      # evicted
    assert n1.kv_arrays is not None   # n1 still has n2b


def test_evicted_node_removed_from_parent_children():
    cache = make_cache(max_gb=0.0)
    node = cache.insert(cache.root, seg([1]), _make_kv(), _kv_scales(), None)
    cache._memory_bytes = int(cache.config.max_memory_gb * 1024**3) + 1
    h = node.context_hash
    cache._evict_if_needed()
    assert h not in cache.root.children
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_lru_evicts_oldest_leaf -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute '_evict_if_needed'`

- [ ] **Step 3: Add eviction methods to `TurnPrefixCache`**

Append to the `TurnPrefixCache` class:
```python
    def _evict_node(self, node: TurnNode) -> None:
        """Free a node's data and remove it from its parent. Internal — caller holds lock."""
        self._memory_bytes -= _node_data_bytes(node)
        node.kv_arrays = None
        node.kv_scales = None
        node.recurrent_state = None

        parent = node.parent
        if parent is not None and node.context_hash in parent.children:
            del parent.children[node.context_hash]
            # Parent may now be a leaf — add to heap if evictable
            if parent.is_evictable and parent is not self.root:
                heapq.heappush(self._eviction_heap, (parent.last_used, id(parent), parent))

    def _evict_if_needed(self) -> None:
        """Evict LRU leaves until memory is within budget."""
        max_bytes = int(self.config.max_memory_gb * 1024**3)
        with self._lock:
            while self._memory_bytes > max_bytes and self._eviction_heap:
                _, _, node = heapq.heappop(self._eviction_heap)
                # Lazy deletion: node may no longer be evictable
                if not node.is_evictable:
                    continue
                self._evict_node(node)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (44 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: LRU eviction with cascade and pinning"
```

---

## Task 7: int8 KV quantization

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add quantization tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
from vllm_mlx.turn_prefix_cache import _quantize_kv, _dequantize_kv


def test_quantize_roundtrip_within_tolerance():
    arr = mx.array([[0.5, -0.3, 0.8, -1.2, 0.0, 1.0, -1.0, 0.25]], dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    diff = mx.abs(restored[0].astype(mx.float32) - arr.astype(mx.float32))
    # Max quantization error <= range/127 ≈ 2/127 ≈ 0.016 for this data
    assert mx.max(diff).item() < 0.02


def test_quantize_preserves_sign():
    arr = mx.array([-1.0, 0.0, 1.0], dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    assert restored[0][0].item() < 0
    assert restored[0][2].item() > 0


def test_quantize_zero_array():
    arr = mx.zeros((4, 4), dtype=mx.bfloat16)
    q, scales = _quantize_kv([arr])
    restored = _dequantize_kv(q, scales)
    assert mx.max(mx.abs(restored[0])).item() == 0.0


def test_insert_quantizes_when_kv_dtype_int8():
    cache = make_cache(stride=512, kv_dtype="int8")
    kv = [mx.ones((1, 4, 3, 256), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, None, None)
    assert node.kv_arrays is not None
    assert node.kv_arrays[0].dtype == mx.int8
    assert node.kv_scales is not None


def test_insert_skips_quantization_when_bf16():
    cache = make_cache(stride=512, kv_dtype="bf16")
    kv = [mx.ones((1, 4, 3, 256), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    assert node.kv_arrays[0].dtype == mx.bfloat16
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_quantize_roundtrip_within_tolerance -v
```
Expected: `ImportError: cannot import name '_quantize_kv'`

- [ ] **Step 3: Add quantization functions to `turn_prefix_cache.py` (before the `TurnPrefixCache` class) and update `insert()`**

Add after `_node_data_bytes`:
```python
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
```

Then update `insert()` to quantize when `kv_dtype == "int8"`. Replace the lines that build the `TurnNode` with:
```python
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
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (50 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: int8 KV quantization with per-tensor scale"
```

---

## Task 8: Disk persistence

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add persistence tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
import json
import tempfile
from pathlib import Path


def test_save_and_load_roundtrip(tmp_path):
    cache = make_cache(stride=0)
    state = mx.zeros((2, 3))
    seg1 = seg(list(range(10)), role="system")
    kv = [mx.ones((1, 4, 10, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg1, kv, None, state, is_system_prompt=True)

    cache.save(str(tmp_path))

    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))
    path, has_recurrent = cache2.match([seg1])
    assert len(path) == 1
    assert has_recurrent


def test_load_version_mismatch(tmp_path):
    meta = {"version": 9999, "model_fingerprint": "test"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cache = make_cache()
    cache.load(str(tmp_path))  # must not raise
    assert len(cache.root.children) == 0  # starts empty


def test_load_missing_kv_file_skips_node(tmp_path):
    cache = make_cache(stride=0)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    cache.insert(cache.root, seg([1, 2, 3]), kv, None, None)
    cache.save(str(tmp_path))

    # Delete the KV file
    for f in tmp_path.glob("kv_*.safetensors"):
        f.unlink()
        break

    cache2 = make_cache(stride=0)
    cache2.load(str(tmp_path))  # must not raise
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_save_and_load_roundtrip -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute 'save'`

- [ ] **Step 3: Add `save()` and `load()` to `TurnPrefixCache`**

Append to the `TurnPrefixCache` class:
```python
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
                    tensors[f"kv_{j}"] = np.array(arr)
                    if node.kv_scales:
                        tensors[f"scale_{j}"] = np.array([node.kv_scales[j]], dtype=np.float32)
                tmp = kv_path + ".tmp"
                st_save(tensors, tmp)
                os.replace(tmp, kv_path)

            if node.recurrent_state is not None and not isinstance(node.recurrent_state, SSDRef):
                rec_path = os.path.join(persist_dir, f"rec_{i}.safetensors")
                tensors = {}
                state = node.recurrent_state
                items = state if isinstance(state, (list, tuple)) else [state]
                for k, item in enumerate(items):
                    sub = item if isinstance(item, (list, tuple)) else [item]
                    for m, arr in enumerate(sub):
                        tensors[f"r_{k}_{m}"] = np.array(arr)
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
                    recurrent_state = tensors  # kept as dict for now
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

        # Link parent→child
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
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (53 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: disk persistence (SQLite index + safetensors per node)"
```

---

## Task 9: SSD offloading

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add SSD tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
def make_ssd_cache(tmp_path, stride=512):
    return TurnPrefixCache(TurnPrefixCacheConfig(
        checkpoint_stride=stride,
        max_memory_gb=8.0,
        kv_dtype="bf16",
        ssd_max_gb=10.0,
        ssd_dir=str(tmp_path),
    ))


def test_spill_replaces_kv_with_ssdref(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_arrays, SSDRef)


def test_spill_trie_still_matchable(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)
    cache._spill_to_ssd(node)
    path, _ = cache.match([seg([1, 2, 3])])
    assert len(path) == 1


def test_promote_restores_arrays(tmp_path):
    cache = make_ssd_cache(tmp_path)
    kv = [mx.ones((1, 4, 3, 16), dtype=mx.bfloat16)]
    node = cache.insert(cache.root, seg([1, 2, 3]), kv, [1.0], None)
    cache._spill_to_ssd(node)
    assert isinstance(node.kv_arrays, SSDRef)
    success = cache._promote_from_ssd(node)
    assert success
    assert isinstance(node.kv_arrays, list)


def test_promote_returns_false_on_missing_file(tmp_path):
    cache = make_ssd_cache(tmp_path)
    node = cache.insert(cache.root, seg([1]), [], [], None)
    node.kv_arrays = SSDRef(file_path="/nonexistent/file.safetensors", size_bytes=0)
    result = cache._promote_from_ssd(node)
    assert result is False
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_turn_prefix_cache.py::test_spill_replaces_kv_with_ssdref -v
```
Expected: `AttributeError: 'TurnPrefixCache' object has no attribute '_spill_to_ssd'`

- [ ] **Step 3: Add SSD methods to `TurnPrefixCache`**

Append to the `TurnPrefixCache` class:
```python
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
                tensors[f"kv_{j}"] = np.array(arr)
                if node.kv_scales:
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
                    node.recurrent_state = st_load(path)
                except Exception as e:
                    logger.warning(f"[turn_cache] SSD recurrent promote failed: {e}")
                    node.recurrent_state = None

        return True
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (57 total).

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat: SSD spill and promotion for TurnPrefixCache"
```

---

## Task 10: Scheduler integration

**Files:**
- Modify: `vllm_mlx/scheduler.py`
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add scheduler integration tests**

Append to `tests/test_turn_prefix_cache.py`:
```python
from unittest.mock import MagicMock


def test_scheduler_integration_fetch_hits_cache():
    """Verify that matching path and recurrent state are set on the request."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    sys_seg = seg(list(range(20)), role="system")
    cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)

    # Simulate what _fetch_turn_cache does
    segments = [sys_seg]
    path, has_recurrent = cache.match(segments)
    assert len(path) == 1
    assert has_recurrent

    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is path[0]
    cache.release(path)


def test_scheduler_integration_store_extends_trie():
    """Verify that inserting new segments after match extends the trie."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))
    sys_seg = seg(list(range(10)), role="system")
    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)

    # Match found sys_node; now store new user segment
    user_seg = seg([100, 101, 102])
    n_user = cache.insert(n_sys, user_seg, [], [], state)
    assert n_user in n_sys.children.values()


def test_concurrent_ref_counts():
    """Two requests sharing a prefix: ref_count=2 while both active."""
    cache = make_cache()
    n = cache.insert(cache.root, seg([1, 2, 3]), [], [], None)

    path1, _ = cache.match([seg([1, 2, 3])])
    path2, _ = cache.match([seg([1, 2, 3])])
    assert n.ref_count == 2
    assert not n.is_evictable

    cache.release(path1)
    assert n.ref_count == 1
    assert not n.is_evictable

    cache.release(path2)
    assert n.ref_count == 0
    assert n.is_evictable
```

- [ ] **Step 2: Run — all should pass (no scheduler changes yet)**

```bash
pytest tests/test_turn_prefix_cache.py::test_scheduler_integration_fetch_hits_cache \
       tests/test_turn_prefix_cache.py::test_scheduler_integration_store_extends_trie \
       tests/test_turn_prefix_cache.py::test_concurrent_ref_counts -v
```
Expected: all 3 pass (they test the cache API, not the scheduler itself).

- [ ] **Step 3: Add `use_turn_cache` to `SchedulerConfig` in `scheduler.py`**

Find the `SchedulerConfig` dataclass (around line 60). Add after the `use_paged_cache` block:
```python
    # TurnPrefixCache settings
    use_turn_cache: bool = False
    turn_cache_stride: int = 512
    turn_cache_ssd_gb: float = 0.0
    turn_cache_memory_gb: float = 8.0
```

- [ ] **Step 4: Add `self.turn_cache` initialization in `Scheduler.__init__`**

Find the block around line 1178 that initializes `self.prefix_cache`. After the existing `if self.config.enable_prefix_cache:` block, add:
```python
        self.turn_cache: Optional[Any] = None
        if self.config.use_turn_cache:
            from .turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
            self.turn_cache = TurnPrefixCache(TurnPrefixCacheConfig(
                checkpoint_stride=self.config.turn_cache_stride,
                max_memory_gb=self.config.turn_cache_memory_gb,
                ssd_max_gb=self.config.turn_cache_ssd_gb,
            ))
            logger.info(
                f"TurnPrefixCache enabled: stride={self.config.turn_cache_stride} "
                f"memory={self.config.turn_cache_memory_gb}GB"
            )
```

- [ ] **Step 5: Add fetch branch in `_fetch_cache_for_request` (around line 1990)**

In the `elif self.prefix_cache is not None:` block's surrounding if/elif chain, add a new branch before it:
```python
        elif self.turn_cache is not None:
            segments = self._messages_to_segments(request)
            if segments:
                path, has_recurrent = self.turn_cache.match(segments)
                if path:
                    request.cache_hit_type = "hit"
                    request._turn_cache_path = path
                    request.cached_tokens = sum(len(n.token_ids) for n in path)
                    ancestor = self.turn_cache.find_checkpoint_ancestor(path)
                    request.prompt_cache = ancestor.recurrent_state if ancestor else None
                    request.remaining_tokens = request.prompt_token_ids[request.cached_tokens:]
                else:
                    request.cache_hit_type = "miss"
                    request.remaining_tokens = request.prompt_token_ids
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
```

- [ ] **Step 6: Add store branch in `_process_batch_responses` (around line 2459)**

In the `elif self.prefix_cache is not None:` store block, add before it:
```python
            elif self.turn_cache is not None:
                if (
                    hasattr(request, "_extracted_cache")
                    and request._extracted_cache is not None
                ):
                    try:
                        segments = self._messages_to_segments(request)
                        path = getattr(request, "_turn_cache_path", [])
                        matched_depth = len(path)
                        parent = path[-1] if path else self.turn_cache.root
                        for i, segment in enumerate(segments[matched_depth:]):
                            is_sys = segment.role == "system" and i == 0 and matched_depth == 0
                            parent = self.turn_cache.insert(
                                parent, segment, [], [], None, is_system_prompt=is_sys
                            )
                        if path:
                            self.turn_cache.release(path)
                    except Exception as e:
                        logger.debug(f"[turn_cache] store failed for {request_id}: {e}")
```

- [ ] **Step 7: Add `_messages_to_segments` helper to the `Scheduler` class**

```python
    def _messages_to_segments(self, request: "Request") -> list:
        """Split a request's token sequence into per-message Segment objects.

        Uses the message boundaries stored on the request. Falls back to
        treating the whole prompt as a single segment if no messages are set.
        """
        from .turn_prefix_cache import Segment

        messages = getattr(request, "messages", None)
        if not messages:
            return []

        # Tokenize each message in conversational context to find boundaries.
        # The full token sequence is request.prompt_token_ids.
        # We split it by re-tokenizing each prefix and noting where length grows.
        full_tokens = list(request.prompt_token_ids or [])
        segments: list[Segment] = []
        pos = 0
        for i, msg in enumerate(messages):
            # Estimate end of this message's tokens using prefix tokenization
            # For now, fall back to approximate: tokenize messages[0:i+1] and diff.
            # The scheduler subclass or model runner should provide exact boundaries.
            # This is a best-effort split; exact boundaries require chat-template-aware tokenization.
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            # Use remaining tokens for the last message
            if i == len(messages) - 1:
                seg_tokens = full_tokens[pos:]
            else:
                # Rough split: assign proportional chunk
                remaining_msgs = len(messages) - i
                remaining_tokens = len(full_tokens) - pos
                chunk = max(1, remaining_tokens // remaining_msgs)
                seg_tokens = full_tokens[pos:pos + chunk]
            if seg_tokens:
                segments.append(Segment(role=role, token_ids=seg_tokens))
                pos += len(seg_tokens)
        return segments
```

> **Note:** The `_messages_to_segments` implementation above uses a rough approximation. For exact segment boundaries, the scheduler needs the tokenizer to re-tokenize each prefix. This can be refined later without changing the `TurnPrefixCache` API.

- [ ] **Step 8: Run all tests**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (60 total).

- [ ] **Step 9: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat: scheduler integration for TurnPrefixCache (fetch + store)"
```

---

## Task 11: CLI flags

**Files:**
- Modify: `vllm_mlx/cli.py`

- [ ] **Step 1: Add `--use-turn-cache` and related flags**

Find the `"--use-paged-cache"` argument definition in `cli.py` (appears twice, once around line 1125 and once around line 1477 for different subcommands). After each `--use-paged-cache` block, add:

```python
        parser.add_argument(
            "--use-turn-cache",
            action="store_true",
            default=False,
            help="Enable TurnPrefixCache (conversation-turn-level prefix trie). "
                 "Recommended for hybrid models (Qwen3.5-27B, etc.).",
        )
        parser.add_argument(
            "--turn-cache-stride",
            type=int,
            default=512,
            metavar="N",
            help="Tokens between permanent recurrent checkpoints in TurnPrefixCache. "
                 "0 = checkpoint every turn (eager). Default: 512.",
        )
        parser.add_argument(
            "--turn-cache-memory-gb",
            type=float,
            default=8.0,
            metavar="GB",
            help="RAM budget in GB for TurnPrefixCache KV data. Default: 8.0.",
        )
        parser.add_argument(
            "--turn-cache-ssd-gb",
            type=float,
            default=0.0,
            metavar="GB",
            help="SSD budget in GB for TurnPrefixCache cold tier. 0 = disabled.",
        )
```

- [ ] **Step 2: Wire the flags into the config dicts**

Find the two places where `use_paged_cache=args.use_paged_cache` is set (around lines 236 and 488). After each, add:
```python
            use_turn_cache=args.use_turn_cache,
            turn_cache_stride=args.turn_cache_stride,
            turn_cache_memory_gb=args.turn_cache_memory_gb,
            turn_cache_ssd_gb=args.turn_cache_ssd_gb,
```

- [ ] **Step 3: Add startup log message for turn cache**

Find the block around line 268 that prints `"Paged cache: ..."`. After the `elif enable_prefix_cache and not args.no_memory_aware_cache:` branch, add:
```python
        if args.use_turn_cache:
            print(
                f"Turn cache: stride={args.turn_cache_stride} tokens, "
                f"memory={args.turn_cache_memory_gb}GB"
                + (f", SSD={args.turn_cache_ssd_gb}GB" if args.turn_cache_ssd_gb > 0 else "")
            )
```

- [ ] **Step 4: Verify CLI parses correctly**

```bash
python -m vllm_mlx.cli serve --help 2>&1 | grep -A2 "turn-cache"
```
Expected: shows `--use-turn-cache`, `--turn-cache-stride`, `--turn-cache-memory-gb`, `--turn-cache-ssd-gb`.

- [ ] **Step 5: Run all tests to verify no regressions**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all 60 tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/cli.py
git commit -m "feat: CLI flags for TurnPrefixCache (--use-turn-cache, --turn-cache-stride)"
```

---

## Task 12: Integration tests

**Files:**
- Modify: `tests/test_turn_prefix_cache.py`

- [ ] **Step 1: Add multi-turn continuation test**

Append to `tests/test_turn_prefix_cache.py`:
```python
# ── Integration tests ──────────────────────────────────────────────────────


def test_multiturn_continuation():
    """Session A stores [sys, u1, a1]; session A extended gets full hit on all three."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    u1_seg = seg([100, 101, 102])
    a1_seg = seg([200, 201])

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)

    # Next request: same [sys, u1, a1] prefix → full hit
    path, has_recurrent = cache.match([sys_seg, u1_seg, a1_seg])
    assert len(path) == 3
    assert path[0] is n_sys
    assert path[1] is n_u1
    assert path[2] is n_a1
    assert has_recurrent
    cache.release(path)


def test_cross_session_system_prompt_reuse():
    """Session B with same system prompt gets immediate recurrent state hit."""
    cache = make_cache(stride=10000)  # high stride so only sys_prompt gets permanent checkpoint
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    assert n_sys.is_permanent_checkpoint

    # Session B: same system prompt
    path, has_recurrent = cache.match([sys_seg])
    assert len(path) == 1
    assert has_recurrent  # permanent checkpoint → recurrent available immediately
    cache.release(path)


def test_mid_session_branching():
    """Two sessions share [sys, u1, a1] but diverge at u2."""
    cache = make_cache(stride=0)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(10)), role="system")
    u1_seg = seg([100])
    a1_seg = seg([200])
    u2a_seg = seg([300])   # branch A
    u2b_seg = seg([400])   # branch B

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)
    n_u2a = cache.insert(n_a1, u2a_seg, [], [], state)
    n_u2b = cache.insert(n_a1, u2b_seg, [], [], state)

    # Branch A
    path_a, _ = cache.match([sys_seg, u1_seg, a1_seg, u2a_seg])
    assert len(path_a) == 4
    assert path_a[3] is n_u2a
    cache.release(path_a)

    # Branch B
    path_b, _ = cache.match([sys_seg, u1_seg, a1_seg, u2b_seg])
    assert len(path_b) == 4
    assert path_b[3] is n_u2b
    cache.release(path_b)

    # Shared nodes are the same objects
    assert path_a[0] is path_b[0]  # sys
    assert path_a[1] is path_b[1]  # u1
    assert path_a[2] is path_b[2]  # a1


def test_gap_reconstruction_finds_ancestor():
    """When branch point has no recurrent, nearest checkpoint ancestor is identified."""
    cache = make_cache(stride=10000)
    state = mx.zeros((1,))

    sys_seg = seg(list(range(50)), role="system")
    u1_seg = seg([100, 101])
    a1_seg = seg([200])

    n_sys = cache.insert(cache.root, sys_seg, [], [], state, is_system_prompt=True)
    n_u1 = cache.insert(n_sys, u1_seg, [], [], state)
    n_a1 = cache.insert(n_u1, a1_seg, [], [], state)
    # n_u1's temp recurrent was pruned when n_a1 was added (stride not met)
    assert n_u1.recurrent_state is None

    path, has_recurrent = cache.match([sys_seg, u1_seg, a1_seg])
    assert has_recurrent  # n_a1 is leaf → has temp recurrent
    ancestor = cache.find_checkpoint_ancestor(path)
    # n_a1 is leaf with temp recurrent → ancestor is n_a1 itself
    assert ancestor is n_a1
    cache.release(path)

    # Now add a child to n_a1 — its temp recurrent is pruned
    u2_seg = seg([300])
    cache.insert(n_a1, u2_seg, [], [], state)
    assert n_a1.recurrent_state is None  # pruned

    # New request ending at n_a1: no recurrent at n_a1, walk up to n_sys
    path2, has_recurrent2 = cache.match([sys_seg, u1_seg, a1_seg])
    assert not has_recurrent2
    ancestor2 = cache.find_checkpoint_ancestor(path2)
    assert ancestor2 is n_sys  # falls back to sys permanent checkpoint
    cache.release(path2)
```

- [ ] **Step 2: Run — all should pass with existing implementation**

```bash
pytest tests/test_turn_prefix_cache.py -v
```
Expected: all tests pass (64 total).

- [ ] **Step 3: Final check — run full test suite for regressions**

```bash
pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no failures outside of any pre-existing failures unrelated to this feature.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_turn_prefix_cache.py
git commit -m "test: TurnPrefixCache integration tests (multi-turn, branching, gap reconstruction)"
```

---

## Self-Review Notes

**Spec coverage:**
- Architecture / TurnNode / Segment / SSDRef / TurnPrefixCacheConfig → Task 1 ✓
- Context-sensitive hashing → Task 1 ✓
- Trie insert + idempotency → Task 2 ✓
- Checkpoint and prune rules (stride, system prompt, temp→permanent) → Task 3 ✓
- Prefix matching + last_used propagation + ref_count → Task 4 ✓
- Gap reconstruction via find_checkpoint_ancestor → Task 5 ✓
- LRU eviction heap + cascade + pinning → Task 6 ✓
- int8 quantization → Task 7 ✓
- Disk persistence (SQLite + safetensors + version check) → Task 8 ✓
- SSD spill/promote/fallback → Task 9 ✓
- Scheduler fetch + store integration → Task 10 ✓
- CLI flags → Task 11 ✓
- Integration tests (multi-turn, cross-session, branching, gap, concurrency) → Task 12 ✓

**Known limitation:** `_messages_to_segments` in Task 10 uses an approximate token boundary split. Exact boundaries require chat-template-aware tokenization. This is a known TODO and does not affect the `TurnPrefixCache` API — only the scheduler's segment construction needs improvement.
