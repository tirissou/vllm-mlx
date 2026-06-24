# Sliding-Window KV Respects Turn Cache Stride — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give sliding-window (rotating) KV its own non-cumulative field on `TurnNode` so it is retained only at permanent checkpoints + active leaves, eliminating the ~200 MB/node bloat under Gemma 4 + turn prefix caching.

**Architecture:** `TurnNode` gains a `sliding_kv_data` field parallel to `recurrent_data`. Rotating segments are partitioned out of `kv_data` (now full-attention-only) at the trie-write boundary. Interior non-checkpoint nodes drop their sliding state on the same trigger that already drops recurrent state; the resume-anchor selector (`find_checkpoint_ancestor`) gains a sliding branch so prefix-divergent fetches fall back to the nearest checkpoint. Persistence stores sliding KV in its own per-node file.

**Tech Stack:** Python 3.14, MLX (`mlx.core`), mlx-lm cache classes, safetensors, SQLite, pytest.

## Global Constraints

- Run `pytest tests/` before any cache-layer commit (per `CLAUDE.md`).
- Emitted segment arrays must stay graph-detached (`mx.eval` + `mx.stop_gradient`) — do not reintroduce live-graph references (CONTEXT.md "Segment contract", ADR-0005).
- The trie's `step()`/scheduler path stays synchronous (ADR-0004) — no async added.
- KV segments stored in mlx-lm native group-quantized format; `_assemble` emits `BatchQuantizedKVCache.from_quantized_arrays` (ADR-0005) — unchanged here.
- KV serialization is **type-driven**: `_write_kv_segment_arrays` / `_read_kv_segment_arrays` (in `turn_prefix_cache.py`) dispatch on the array type of each segment — `QuantizedArray` → packed/scales/biases tensors; plain `mx.array` (float, e.g. bf16) → float tensors. These helpers are used by **both** the `kv_data` and `sliding_kv_data` persistence paths. In production, full-attention `kv_data` uses `KVQuantPolicy.full_bits=8` (quantized, byte-identical wire format to v5); sliding `kv_data` uses `sliding_bits=None` (bf16 float).

## Deviation from the committed spec (2026-06-23 design doc)

The spec proposed a `SegmentedState(full_kv, sliding_kv, recurrent)` dataclass changing the `segment()` / `assemble()` / `collect_path_data()` return signatures. During planning we found those signatures are consumed by ~30 existing test call sites, and `assemble()` is type-agnostic (reconstructs per `layer_index`, indifferent to full vs sliding), so it needs no split. **This plan keeps the translator signatures unchanged** and performs the full/sliding partition at the trie-write boundary (`TurnCacheManager.on_prefill_checkpoint`), sourcing sliding KV from the new `sliding_kv_data` field inside `collect_path_data`. The architectural outcome the spec calls for — a separate, independently droppable/persistable `sliding_kv_data` field on `TurnNode` — is fully preserved with far less churn and risk.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `vllm_mlx/turn_prefix_cache.py` | Trie storage, lifecycle, persistence | `sliding_kv_data` field; insert plumbing; memory accounting; `has_sliding_state`; interior cleanup; `find_checkpoint_ancestor` branch; `collect_path_data`; `save`/`load` + helpers; version bump |
| `vllm_mlx/prefix_cache_adapters.py` | `TurnCacheManager` (PrefixCache impl) | Partition segments into full vs sliding at `on_prefill_checkpoint` |
| `tests/test_turn_prefix_cache.py` | Trie unit tests | Field, cleanup, anchor, accounting tests |
| `tests/test_turn_prefix_cache_integration.py` | Round-trip integration | Sliding-in-own-field round-trip; prefix-divergent resume |
| `tests/test_turn_cache_roundtrip.py` | Persistence round-trip | Sliding save/load + version bump |
| `CONTEXT.md`, `docs/adr/`, `docs/dev/` | Domain docs | Invariant + seam updates |

---

## Task 1: Add `sliding_kv_data` field to `TurnNode` and plumb it through `insert`

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py` (`TurnNode` dataclass ~94-104; `_node_data_bytes` ~156-177; `__init__` ~181-194; `insert` ~229-247; `_insert_node` ~249-301)
- Test: `tests/test_turn_prefix_cache.py`

**Interfaces:**
- Produces:
  - `TurnNode.sliding_kv_data: list[KVLayerSegment] | SSDRef | None` (defaults `None`).
  - `TurnPrefixCache.insert(parent, segment, kv_data=None, sliding_kv_data=None, recurrent_data=None, is_system_prompt=False, acquire_lock=True) -> TurnNode`
  - `TurnPrefixCache._insert_node(parent, segment, kv_data, sliding_kv_data, recurrent_data, is_system_prompt=False, acquire_lock=True) -> TurnNode`
  - `TurnPrefixCache.has_sliding_state: bool`
  - `_node_data_bytes(node)` counts `sliding_kv_data` bytes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
import mlx.core as mx
from vllm_mlx.cache_types import KVConcatSegment, KVRotatingSegment
from vllm_mlx.turn_prefix_cache import (
    Segment, TurnPrefixCache, TurnPrefixCacheConfig, _node_data_bytes,
)


def _concat_seg(layer_index=0, n=4):
    k = mx.zeros((1, 2, n, 8), dtype=mx.bfloat16)
    return KVConcatSegment(keys=k, values=k, layer_index=layer_index,
                           n_tokens=n, bits=None, class_name="KVCache")


def _rot_seg(layer_index=1, n=4):
    k = mx.zeros((1, 2, n, 8), dtype=mx.bfloat16)
    return KVRotatingSegment(keys=k, values=k, layer_index=layer_index,
                             n_tokens=n, bits=None, class_name="RotatingKVCache",
                             max_size=n, keep=0, offset=n, idx=n)


def test_insert_stores_sliding_kv_data_separately():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    node = cache.insert(
        cache.root,
        Segment(role="user", token_ids=[1, 2, 3, 4]),
        kv_data=[_concat_seg()],
        sliding_kv_data=[_rot_seg()],
    )
    assert node.sliding_kv_data is not None
    assert len(node.sliding_kv_data) == 1
    assert node.kv_data is not None and len(node.kv_data) == 1
    assert cache.has_sliding_state is True


def test_node_data_bytes_counts_sliding():
    node_kv_only = _make_node(kv=[_concat_seg()], sliding=None)
    node_with_sliding = _make_node(kv=[_concat_seg()], sliding=[_rot_seg()])
    assert _node_data_bytes(node_with_sliding) > _node_data_bytes(node_kv_only)


def _make_node(kv, sliding):
    from vllm_mlx.turn_prefix_cache import TurnNode
    return TurnNode(token_ids=[1], context_hash=1, kv_data=kv,
                    recurrent_data=None, sliding_kv_data=sliding)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turn_prefix_cache.py::test_insert_stores_sliding_kv_data_separately tests/test_turn_prefix_cache.py::test_node_data_bytes_counts_sliding -v`
Expected: FAIL — `TypeError: insert() got an unexpected keyword argument 'sliding_kv_data'` / `TurnNode.__init__() got an unexpected keyword argument 'sliding_kv_data'`.

- [ ] **Step 3: Add the field to `TurnNode`**

In the `TurnNode` dataclass, add `sliding_kv_data` as a defaulted field placed AFTER the non-default fields (`kv_data`, `recurrent_data`) and before `parent`:

```python
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
```

- [ ] **Step 4: Count sliding bytes in `_node_data_bytes`**

In `_node_data_bytes`, after the existing `kv_data` block and before the `recurrent_data` block, add:

```python
    if isinstance(node.sliding_kv_data, list):
        for kv in node.sliding_kv_data:
            total += kv.keys.nbytes + kv.values.nbytes
```

- [ ] **Step 5: Initialize `has_sliding_state`**

In `TurnPrefixCache.__init__`, alongside `self.has_recurrent_state = False`:

```python
        self.has_recurrent_state: bool = False
        self.has_sliding_state: bool = False
```

- [ ] **Step 6: Plumb `sliding_kv_data` through `insert` / `_insert_node`**

Update `insert` signature and its forwarding call:

```python
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
```

Update `_insert_node` signature, the `has_*` flag setting, and the `TurnNode(...)` construction:

```python
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
            # ... (unchanged through is_permanent / node_tsc) ...
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
```

(Leave the rest of `_insert_node` — touch, eviction-heap push, memory accounting — unchanged.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_turn_prefix_cache.py -v`
Expected: PASS (new tests green; existing tests unaffected — `sliding_kv_data` defaults to `None`).

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn-cache): add sliding_kv_data field to TurnNode"
```

---

## Task 2: Route sliding KV into its own field end-to-end

Moves rotating segments out of `kv_data` (at the producer) and reads them back from `sliding_kv_data` (at `collect_path_data`), keeping reconstruction intact. After this task `kv_data` holds only full-attention concat layers.

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py` (`collect_path_data` ~357-386)
- Modify: `vllm_mlx/prefix_cache_adapters.py` (import block ~21; `on_prefill_checkpoint` segment-handling ~422-448)
- Test: `tests/test_turn_prefix_cache.py`

**Interfaces:**
- Consumes: `TurnPrefixCache.insert(..., sliding_kv_data=...)` (Task 1).
- Produces: `collect_path_data(node) -> (kv_layers, recurrent_layers)` where `kv_layers` is full-attention concat-merged across the path **plus** the anchor's sliding segments. Signature unchanged from today.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py` (reuses `_concat_seg`/`_rot_seg` from Task 1):

```python
def test_collect_path_data_returns_full_and_sliding():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    a = cache.insert(cache.root, Segment("user", [1, 2]),
                     kv_data=[_concat_seg(layer_index=0)],
                     sliding_kv_data=[_rot_seg(layer_index=1)])
    b = cache.insert(a, Segment("user", [3, 4]),
                     kv_data=[_concat_seg(layer_index=0)],
                     sliding_kv_data=[_rot_seg(layer_index=1)])

    kv_layers, rec_layers = cache.collect_path_data(b)
    layer_indices = sorted(seg.layer_index for seg in kv_layers)
    assert layer_indices == [0, 1]            # one full (0) + one sliding (1)
    # Sliding (layer 1) comes from the anchor b only — exactly one segment.
    sliding = [s for s in kv_layers if s.layer_index == 1]
    assert len(sliding) == 1
    assert rec_layers == []
```

Note `Segment("user", [1, 2])` uses positional args matching `Segment(role, token_ids)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turn_prefix_cache.py::test_collect_path_data_returns_full_and_sliding -v`
Expected: FAIL — `collect_path_data` currently only reads `kv_data`; with sliding now in `sliding_kv_data`, layer 1 is missing, so `layer_indices == [0]`.

- [ ] **Step 3: Read sliding from the anchor field in `collect_path_data`**

Replace the body of `collect_path_data`:

```python
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
```

- [ ] **Step 4: Run the new trie test to verify it passes**

Run: `pytest tests/test_turn_prefix_cache.py::test_collect_path_data_returns_full_and_sliding -v`
Expected: PASS.

- [ ] **Step 5: Partition segments at the producer**

In `vllm_mlx/prefix_cache_adapters.py`, add the import (top-of-file import block, ~line 21):

```python
from vllm_mlx.cache_types import KVRotatingSegment
```

Replace the segment-handling block in `on_prefill_checkpoint` (currently ~422-426 plus the `insert` call ~442-448):

```python
            kv_sparse, rec_sparse = segment(
                extracted_cache, policy=self._policy, group_size=self._kv_group_size
            )
            kv_layers, sliding_layers = [], []
            for kv in kv_sparse:
                if kv is None:
                    continue
                if isinstance(kv, KVRotatingSegment):
                    sliding_layers.append(kv)
                else:
                    kv_layers.append(kv)
            rec_layers = [rec for rec in rec_sparse if rec is not None]
            _log_segment_breakdown(
                f"checkpoint rid={getattr(request, 'request_id', '?')} "
                f"abs_idx={abs_idx} tok_seg={len(turn_segment.token_ids)} "
                f"prev_end={prev_end} total={total_tokens_prefilled}",
                kv_layers + sliding_layers,
                rec_layers,
            )
        else:
            kv_layers, sliding_layers, rec_layers = [], [], []
```

And update the `insert` call to pass sliding separately:

```python
        new_node = self._inner.insert(
            parent,
            turn_segment,
            kv_data=kv_layers or None,
            sliding_kv_data=sliding_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )
```

- [ ] **Step 6: Run the integration + adapter tests**

Run: `pytest tests/test_turn_prefix_cache_integration.py tests/test_prefix_cache_adapters.py -v`
Expected: PASS — sliding now lands in `sliding_kv_data`; `collect_path_data` reassembles full + sliding; `assemble(kv_layers, rec_layers)` reconstructs by `layer_index` unchanged.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py vllm_mlx/prefix_cache_adapters.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn-cache): route sliding KV into sliding_kv_data field"
```

---

## Task 3: Drop sliding KV at interior nodes; resume from nearest checkpoint

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py` (interior-cleanup block in `_insert_node` ~288-296; `find_checkpoint_ancestor` ~340-355)
- Test: `tests/test_turn_prefix_cache.py`

**Interfaces:**
- Consumes: `has_sliding_state`, `sliding_kv_data` (Task 1).
- Produces: interior non-checkpoint nodes have `sliding_kv_data is None`; `find_checkpoint_ancestor` returns the deepest path node still holding in-memory `sliding_kv_data` when `has_sliding_state` and not `has_recurrent_state`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_interior_node_drops_sliding_keeps_full():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    parent = cache.insert(cache.root, Segment("user", [1, 2]),
                          kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert parent.is_permanent_checkpoint is False
    # Giving parent its first child turns it interior -> sliding dropped.
    cache.insert(parent, Segment("user", [3, 4]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert parent.sliding_kv_data is None       # dropped
    assert isinstance(parent.kv_data, list)     # full attention retained


def test_checkpoint_node_keeps_sliding_when_gaining_child():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    cp = cache.insert(cache.root, Segment("system", [1, 2]),
                      kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()],
                      is_system_prompt=True)
    assert cp.is_permanent_checkpoint is True
    cache.insert(cp, Segment("user", [3, 4]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert isinstance(cp.sliding_kv_data, list)  # checkpoint keeps sliding


def test_find_checkpoint_ancestor_falls_back_to_checkpoint_for_sliding():
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    cp = cache.insert(cache.root, Segment("system", [1, 2]),
                      kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()],
                      is_system_prompt=True)
    mid = cache.insert(cp, Segment("user", [3, 4]),
                       kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    # mid gains a child -> becomes interior -> its sliding is dropped.
    cache.insert(mid, Segment("user", [5, 6]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    assert mid.sliding_kv_data is None

    anchor = cache.find_checkpoint_ancestor([cp, mid])
    assert anchor is cp           # mid has no sliding -> fall back to checkpoint cp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_turn_prefix_cache.py -k "interior_node_drops_sliding or checkpoint_node_keeps_sliding or falls_back_to_checkpoint" -v`
Expected: FAIL — `test_interior_node_drops_sliding_keeps_full` fails (`parent.sliding_kv_data` still a list); `test_find_checkpoint_ancestor_falls_back_to_checkpoint_for_sliding` fails (returns `mid`, the deepest node with `kv_data`).

- [ ] **Step 3: Drop sliding in the interior-cleanup block**

In `_insert_node`, extend the existing cleanup block:

```python
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
```

- [ ] **Step 4: Add the sliding branch to `find_checkpoint_ancestor`**

Replace `find_checkpoint_ancestor`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_turn_prefix_cache.py -v`
Expected: PASS.

- [ ] **Step 6: Run the round-trip integration to confirm resume correctness**

Run: `pytest tests/test_turn_prefix_cache_integration.py -v`
Expected: PASS — exact-match-to-leaf reconstructs with zero recompute; prefix-divergent fetch resumes from the checkpoint anchor.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn-cache): sliding KV respects checkpoint stride"
```

---

## Task 4: Persist `sliding_kv_data` (float-aware) and bump the cache format version

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py` (`_CACHE_FORMAT_VERSION` ~28; new `_write_sliding_arrays` / `_read_sliding_arrays` helpers near `_kv_segment_to_meta` ~31; `save` schema + write + INSERT ~452-551; `load` unpack + read + `has_sliding_state` ~577-699)
- Test: `tests/test_turn_cache_roundtrip.py`

**Interfaces:**
- Consumes: `node.sliding_kv_data`, `_kv_segment_to_meta`, `_meta_to_kv_segment`.
- Produces: `nodes` table gains a `sliding_file_path TEXT` column; `_CACHE_FORMAT_VERSION == 6`; `load` repopulates `sliding_kv_data` and `has_sliding_state`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_turn_cache_roundtrip.py`:

```python
import os
import mlx.core as mx
from vllm_mlx.cache_types import KVConcatSegment, KVRotatingSegment
from vllm_mlx.turn_prefix_cache import (
    Segment, TurnPrefixCache, TurnPrefixCacheConfig,
)


def _concat_seg(layer_index=0, n=4):
    k = mx.zeros((1, 2, n, 8), dtype=mx.bfloat16)
    return KVConcatSegment(keys=k, values=k, layer_index=layer_index,
                           n_tokens=n, bits=None, class_name="KVCache")


def _rot_seg(layer_index=1, n=4):
    k = mx.ones((1, 2, n, 8), dtype=mx.bfloat16)
    return KVRotatingSegment(keys=k, values=k, layer_index=layer_index,
                             n_tokens=n, bits=None, class_name="RotatingKVCache",
                             max_size=n, keep=0, offset=n, idx=n)


def test_sliding_kv_survives_save_load(tmp_path):
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    cache.insert(cache.root, Segment("user", [1, 2, 3, 4]),
                 kv_data=[_concat_seg()], sliding_kv_data=[_rot_seg()])
    cache.save(str(tmp_path))

    restored = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=10000))
    restored.load(str(tmp_path))

    node = next(iter(restored.root.children.values()))
    assert isinstance(node.sliding_kv_data, list)
    assert len(node.sliding_kv_data) == 1
    seg = node.sliding_kv_data[0]
    assert seg.layer_index == 1
    assert seg.max_size == 4
    assert restored.has_sliding_state is True


def test_format_version_five_cache_is_rejected(tmp_path):
    import json
    with open(os.path.join(tmp_path, "meta.json"), "w") as f:
        json.dump({"version": 5, "model_fingerprint": ""}, f)
    cache = TurnPrefixCache(TurnPrefixCacheConfig())
    cache.load(str(tmp_path))   # version mismatch -> starts empty, no raise
    assert len(cache.root.children) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_turn_cache_roundtrip.py -k "sliding_kv_survives or version_five" -v`
Expected: FAIL — `test_sliding_kv_survives_save_load` fails (`sliding_kv_data is None` after load; column/serialization absent); `test_format_version_five_cache_is_rejected` fails (current version is 5, so a v5 file is accepted).

- [ ] **Step 3: Bump the format version**

```python
_CACHE_FORMAT_VERSION = 6
```

- [ ] **Step 4: Add float-aware sliding serialization helpers**

Add near `_kv_segment_to_meta` (module level):

```python
def _write_sliding_arrays(tensors: dict, j: int, kv_item) -> None:
    """Serialize one sliding KV segment's arrays into `tensors`.

    Handles both quantized (QuantizedArray) and float (mx.array) keys/values,
    because sliding layers default to bf16 (KVQuantPolicy.sliding_bits=None).
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


def _read_sliding_arrays(tensors: dict, j: int, item_meta: dict):
    """Reconstruct one sliding KVLayerSegment from saved tensors + metadata."""
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
```

- [ ] **Step 5: Add the `sliding_file_path` column and write sliding in `save`**

In `save`, update the `CREATE TABLE` to add the column after `recurrent_file_path`:

```python
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
```

Inside the per-node loop, after the recurrent-save block and before the `INSERT`, add (initialize `sliding_path = None` alongside `rec_path = None`):

```python
                # Save sliding_kv_data: list[KVLayerSegment] (own file)
                sliding_path: str | None = None
                if isinstance(node.sliding_kv_data, list) and node.sliding_kv_data:
                    sliding_path = os.path.join(persist_dir, f"sliding_{i}.safetensors")
                    s_tensors: dict[str, np.ndarray] = {}
                    s_meta: list[dict] = []
                    for j, kv_item in enumerate(node.sliding_kv_data):
                        _write_sliding_arrays(s_tensors, j, kv_item)
                        s_meta.append(_kv_segment_to_meta(kv_item))
                    tmp = sliding_path + ".tmp"
                    st_save(s_tensors, tmp)
                    os.replace(tmp, sliding_path)
                    s_meta_path = sliding_path.replace(".safetensors", "_meta.json")
                    with open(s_meta_path, "w") as mf:
                        json.dump(s_meta, mf)
```

Update the `INSERT` to 9 columns (add `sliding_path` after `rec_path`):

```python
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
```

- [ ] **Step 6: Read the column and rebuild sliding in `load`**

Update the row unpack to 9 columns (add `sliding_path` after `rec_path`):

```python
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
```

After the recurrent-load block and before `node = TurnNode(...)`, add:

```python
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
                            _read_sliding_arrays(s_tensors, j, item_meta)
                            for j, item_meta in enumerate(s_meta_list)
                        ]
                        sliding_kv_data = s_items if s_items else None
                except Exception as e:
                    logger.warning(
                        f"[turn_cache] sliding load failed for {ctx_hash}: {e}"
                    )
```

Pass it into the `TurnNode(...)` construction:

```python
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
```

Extend the post-load flag scan (currently sets `has_recurrent_state`) to also set `has_sliding_state`:

```python
        for node in hash_to_node.values():
            if node is self.root:
                continue
            if isinstance(node.recurrent_data, list) and node.recurrent_data:
                self.has_recurrent_state = True
            if isinstance(node.sliding_kv_data, list) and node.sliding_kv_data:
                self.has_sliding_state = True
```

(Remove the `break` so both flags are detected across nodes.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_turn_cache_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py tests/test_turn_cache_roundtrip.py
git commit -m "feat(turn-cache): persist sliding_kv_data; bump cache format to v6"
```

---

## Task 5: Documentation + peak-memory regression guard

**Files:**
- Modify: `CONTEXT.md` (Trie storage types / Cache translator / checkpoint sections)
- Modify: `docs/adr/` (ADR-0005 addendum if storage-format description is now stale)
- Modify: `docs/dev/cache-reconstruction-invariants.md` (note the sliding-field sourcing)
- Verify: `tests/test_cache_hit_oom_repro.py` (Gemma-4-scale peak-memory guard)

**Interfaces:** none (docs + verification).

- [ ] **Step 1: Run the production-scale OOM repro to capture the post-change peak**

Run: `pytest tests/test_cache_hit_oom_repro.py -v`
Expected (Metal hardware): PASS — interior sliding bloat removed; Metal peak memory at or below the test's existing threshold. (Skipped on non-Metal hardware — note the skip and rely on the unit-level accounting tests from Tasks 1/3.)

If the test asserts an explicit peak threshold that is now comfortably beaten, tighten the threshold to lock in the win. Show the change:

```python
# tests/test_cache_hit_oom_repro.py — tighten the regression ceiling to the
# post-fix peak (sliding KV no longer retained at every interior TurnNode).
assert peak_gb < <new_observed_ceiling>
```

- [ ] **Step 2: Update `CONTEXT.md`**

Under **Trie storage types**, add a `sliding_kv_data` entry and narrow `kv_data`:

```markdown
**TurnNode storage fields** — `kv_data` holds full-attention (`KVConcatSegment`)
layers only; `sliding_kv_data` holds sliding-window (`KVRotatingSegment`) layers;
`recurrent_data` holds recurrent state. The latter two are non-cumulative and are
retained only at permanent checkpoints + active leaves (see "Leaf-only eviction"
and the checkpoint-stride invariant). Each field may independently be a `list`,
an `SSDRef`, or `None`.
```

Under **Cache translator**, note that the full/sliding partition happens at the
trie-write boundary (`TurnCacheManager.on_prefill_checkpoint`), not in `segment()`:
`segment()` / `assemble()` signatures are unchanged; `assemble` reconstructs per
`layer_index` regardless of full vs sliding. `collect_path_data` sources full KV by
concat-merge across the path and sliding KV from the anchor node's `sliding_kv_data`.

Under the **Leaf-only eviction / checkpoint** discussion, state that non-cumulative
state — recurrent **and** sliding-window KV — survives only at checkpoints + leaves.

- [ ] **Step 3: Check ADR-0005 and the dev note**

ADR-0005 governs the KV storage format and the `segment`/`assemble` seam. Add a short
addendum if its description of "where KV segments live on a `TurnNode`" is now stale —
note the dedicated `sliding_kv_data` field and the v6 format (per-node `sliding_*`
safetensors, float-aware). In `docs/dev/cache-reconstruction-invariants.md`, add one
line: sliding layers are reconstructed from the anchor's `sliding_kv_data`, full layers
from the concat-merged `kv_data` path.

- [ ] **Step 4: Run the full cache-layer suite (CLAUDE.md gate)**

Run: `pytest tests/ -v`
Expected: PASS (Metal-only tests may skip on non-Metal hardware).

- [ ] **Step 5: Commit**

```bash
git add CONTEXT.md docs/adr docs/dev/cache-reconstruction-invariants.md tests/test_cache_hit_oom_repro.py
git commit -m "docs: sliding_kv_data field + stride invariant; tighten OOM regression"
```

---

## Self-Review

**Spec coverage:**
- Separate `sliding_kv_data` field on `TurnNode` → Task 1.
- Rotating split out of `kv_data`; reconstruction intact → Task 2.
- Interior cleanup respects stride; resume-anchor sliding branch; `has_sliding_state` → Tasks 1 (flag) + 3.
- Persistence (own file/column, float-aware, version bump) → Task 4.
- `match()` `SSDRef` guard: intentionally **not** changed — `match()`'s second return value is ignored by its only caller (`fetch`, `path, _ = self._inner.match(...)`); sliding presence is determined by `find_checkpoint_ancestor`. Documented here to record the deliberate omission (YAGNI).
- SSD independent spill of sliding KV: explicitly out of scope (spec "Out of scope"); `_spill_to_ssd`/`_promote_from_ssd` keep operating on full `kv_data` only.
- Docs (CONTEXT.md, ADR-0005, dev note) + peak-memory regression → Task 5.

**Placeholder scan:** No TBD/TODO; every code step shows full code; the only intentional fill-in is the OOM ceiling value in Task 5 Step 1, which depends on the observed post-fix peak on Metal hardware (with a documented fallback to the existing threshold).

**Type consistency:** `sliding_kv_data` typed identically everywhere (`list[KVLayerSegment] | SSDRef | None`). `insert`/`_insert_node` parameter order consistent (`kv_data, sliding_kv_data, recurrent_data`). `collect_path_data` keeps its `(list[KVLayerSegment], list[RecurrentLayerSegment])` return; `assemble` unchanged. `_write_sliding_arrays`/`_read_sliding_arrays` names match between Task 4 definition and use. SQLite column count (9) consistent across `CREATE TABLE`, `INSERT`, and row unpack.
