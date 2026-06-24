# Sliding-window KV respects the turn cache stride

**Date:** 2026-06-23
**Status:** Approved — ready for implementation plan
**Branch:** `sliding-window-respects-stride`

## Problem

Running Gemma 4 with turn prefix caching, the sliding-window (rotating) KV stores
roughly 200 MB at *every* `TurnNode`. On a deep trie this dominates cache memory and
drives cache-hit OOM at production scale.

Recurrent state already avoids this. In `TurnPrefixCache._insert_node`, when a
non-checkpoint parent gains its first child, `parent.recurrent_data` is dropped — so
recurrent state survives only at permanent checkpoints (every `checkpoint_stride`
tokens) and at active leaves. This is safe because recurrent state is **non-cumulative**.

Sliding-window state is *also* non-cumulative — `KVRotatingSegment.merge_path()` returns
`path[-1]` ("deepest node only"), exactly like recurrent. But rotating segments live
inside each node's `kv_data` list, mixed with the cumulative full-attention
`KVConcatSegment`s, and the interior cleanup never touches them. Every node therefore
retains its ring-buffer state even though only the anchor's copy is ever used.

## Goal

Sliding-window KV becomes a first-class non-cumulative field on `TurnNode`, with the
same lifecycle and storage independence recurrent state already has:

- retained only at permanent checkpoints + active leaves (respects `checkpoint_stride`);
- stored in its own field, independently droppable;
- independently spillable to SSD (the **spill wiring is deferred** — see Out of Scope —
  but the field/persistence separation leaves a clean seam for it).

## Why a separate field (not an `isinstance` filter)

`recurrent_data` is **already** independently spillable today:

```python
kv_data:        list[KVLayerSegment] | SSDRef | None
recurrent_data: list[RecurrentLayerSegment] | SSDRef | None
```

A node can hold `kv_data` in RAM while `recurrent_data` is an `SSDRef` — which is why
`match()` guards with `not isinstance(path[-1].recurrent_data, SSDRef)`. Recurrent state
has independent spill granularity. Sliding KV does not, *only* because it is buried in
the `kv_data` blob that spills as one unit. Giving sliding KV its own field is not new
architecture — it extends the existing recurrent precedent to the other non-cumulative
state. SSD tiering will later want to drop/spill sliding KV independently of full KV;
the separate field is what makes that possible.

## Design

### Storage — `TurnNode` (`turn_prefix_cache.py`)

Add a third field, structurally identical to `recurrent_data`:

```python
kv_data:          list[KVLayerSegment] | SSDRef | None   # full-attention only (concat, cumulative)
sliding_kv_data:  list[KVLayerSegment] | SSDRef | None   # NEW — rotating, non-cumulative
recurrent_data:   list[RecurrentLayerSegment] | SSDRef | None
```

The name `kv_data` is kept (it is the persisted SQLite column; renaming adds churn for
marginal clarity). Its semantics narrow to full-attention layers only — documented in
code and `CONTEXT.md`.

### Translator seam (`cache_translator.py`)

`segment()` is the point where `class_name` distinguishes `RotatingKVCache` from other
KV caches, so the split happens here.

- **`segment()`** changes its return from
  `(kv_list, rec_list)` to a small dataclass:

  ```python
  @dataclass
  class SegmentedState:
      full_kv: list[KVLayerSegment | None]      # concat layers, sparse by layer_index
      sliding_kv: list[KVLayerSegment | None]   # rotating layers, sparse by layer_index
      recurrent: list[RecurrentLayerSegment | None]
  ```

  A dataclass (not a 3-tuple) keeps call sites readable and is extensible if SSD tiers
  grow. The `RotatingKVCache` branch fills `sliding_kv[i]`; the other `KVCache` branch
  fills `full_kv[i]`; the recurrent branch fills `recurrent[i]`.

- **`assemble()`** accepts full + sliding (disjoint `layer_index` sets) + recurrent and
  reconstructs per `layer_index`. Each segment reconstructs polymorphically via
  `seg.reconstruct(group_size)` — no `isinstance` dispatch.

- **Segment contract / round-trip law** are preserved: emitted arrays are still
  `mx.eval` + `mx.stop_gradient` detached, and
  `assemble(segment(live_states)) ≈ live_states` holds across the 3-field split.

Producers updated for the new return shape: `TurnCacheManager.fetch` (reconstruction-
validate), `store`, `on_prefill_checkpoint`.

### Trie behavior (`turn_prefix_cache.py`)

- **Interior cleanup** — the existing block in `_insert_node`
  (`len(parent.children) == 1 and not parent.is_permanent_checkpoint and parent is not root`)
  additionally drops sliding state:

  ```python
  parent.recurrent_data = None
  parent.sliding_kv_data = None   # NEW
  ```

  Full `kv_data` (cumulative concat) is left intact. The existing
  `freed = before − after` memory-accounting bracket captures the reclaimed bytes.

- **`has_sliding_state`** — new flag mirroring `has_recurrent_state`, set `True` when a
  sliding segment is inserted (in `_insert_node` and the `load()` path).

- **`find_checkpoint_ancestor`** — add a sliding branch, strictly parallel to recurrent:

  ```
  if has_recurrent_state:   deepest node with in-memory recurrent_data    (unchanged)
  elif has_sliding_state:   deepest node with in-memory sliding_kv_data    (NEW)
  else:                     deepest node with kv_data                      (unchanged)
  ```

  This is the load-bearing change for Gemma 4 (`has_recurrent_state == False`): the old
  `else` returned the deepest node with *any* `kv_data`, which after cleanup is an
  interior concat-only node that cannot anchor sliding layers. The new branch makes a
  prefix-divergent fetch fall back to the nearest checkpoint.

- **`collect_path_data`** returns three results: full KV (concat merged across the path),
  sliding KV (from the anchor only), recurrent (from the leaf). The previous mixed
  `kv_by_layer` dict simplifies — full and sliding are gathered from separate fields.

- **`match()`** mirrors the recurrent `SSDRef` guard for `sliding_kv_data` when
  computing whether the matched node carries usable in-memory non-cumulative state.

- **`_node_data_bytes`** sums `kv_data` + `sliding_kv_data` + `recurrent_data`.

### Persistence (`save` / `load`)

- Add a nullable `sliding_file_path TEXT` column to the `nodes` table, alongside
  `recurrent_file_path`.
- Write sliding segments to their own per-node safetensors (+ JSON sidecar), mirroring
  the existing kv/recurrent serialization.
- Bump `_CACHE_FORMAT_VERSION` (currently `5` → `6`). Older persisted caches are
  incompatible and rejected by the version check on load.
- `load()` reconstructs `sliding_kv_data` and re-establishes `has_sliding_state`.

### Behavior preserved

- **Exact-match-to-leaf** (common multi-turn case) → leaf retains its sliding state →
  zero recompute.
- **Prefix-divergent fetch** → resume from nearest checkpoint, re-prefill ≤ `stride`
  tokens (approved trade-off; identical to today's recurrent behavior; falls out of
  selecting an earlier anchor — `cached_tokens = ancestor.n_tokens`).
- **`checkpoint_stride == 0`** → every node permanent → cleanup never fires → fully
  correct, no savings.
- **Hybrid recurrent + sliding** → recurrent branch wins the if-chain; both fields are
  cleaned at the same trigger, so the recurrent anchor is also a valid sliding anchor.

## Out of scope (documented follow-up)

Wiring `sliding_kv_data` into the SSD spill delegate (`on_spill` / `on_promote`,
`SSDOffloadedCache`) so sliding KV tiers to SSD independently of full KV. The
stride-driven interior cleanup already reclaims the bulk of the per-node sliding bytes
(interior nodes dominate a deep trie); independent SSD spill of the remaining
checkpoint/leaf sliding KV is a clean follow-up task on the new field + persistence seam.

## Testing

- **Unit (`test_turn_prefix_cache.py`)**
  - Interior transition drops `sliding_kv_data`; permanent checkpoints and active leaves
    retain it.
  - Memory accounting (`_memory_bytes`) decreases by the dropped sliding bytes.
  - `has_sliding_state` is set when a sliding segment is inserted.
- **Round-trip (`test_cache_translator.py`, `test_turn_prefix_cache_integration.py`)**
  - `segment` → `assemble` law holds across the 3-field split (full + sliding +
    recurrent), interleaved layer indices reconstruct in order.
  - A prefix-divergent fetch resumes from the checkpoint and reconstructs correctly.
- **Peak-memory regression (`test_cache_hit_oom_repro.py`)**
  - Gemma-4-scale repro shows the per-node sliding bloat removed; Metal peak memory
    drops. Skipped on non-Metal hardware.
- **Persistence**
  - `save` → `load` round-trips `sliding_kv_data`; version bump rejects format-5 caches.

Run `pytest tests/` before any cache-layer commit (per `CLAUDE.md`).

## Documentation updates

- **`CONTEXT.md`**
  - Translator seam: `segment` / `assemble` now exchange a `SegmentedState`
    (full + sliding + recurrent), not `(kv_list, rec_list)`.
  - Checkpoint / non-cumulative-state invariant: recurrent **and** sliding KV are
    retained only at checkpoints + leaves; `kv_data` is full-attention only.
  - `TurnNode` storage fields: document `sliding_kv_data`.
- **ADR-0005** (KV segments stored in mlx-lm native group-quantized format) — check
  whether the third storage field + format-version bump warrants an addendum; add one if
  the storage-format description would otherwise be stale.
- **`docs/dev/cache-reconstruction-invariants.md`** — note the 3-field split if it
  affects the `_assemble` invariants described there.

## Files touched

| File | Change |
|------|--------|
| `vllm_mlx/turn_prefix_cache.py` | `sliding_kv_data` field, interior cleanup, `has_sliding_state`, `find_checkpoint_ancestor` branch, `collect_path_data`, `match`, `_node_data_bytes`, `save`/`load`, version bump |
| `vllm_mlx/cache_translator.py` | `SegmentedState`, `segment()` return, `assemble()` signature |
| `vllm_mlx/prefix_cache_adapters.py` | `TurnCacheManager` producers/consumers of the new return shape |
| `tests/test_turn_prefix_cache.py` | interior cleanup + flag unit tests |
| `tests/test_cache_translator.py` | 3-field round-trip |
| `tests/test_turn_prefix_cache_integration.py` | prefix-divergent resume |
| `tests/test_cache_hit_oom_repro.py` | peak-memory regression (existing, expect drop) |
| `CONTEXT.md`, `docs/adr/`, `docs/dev/` | invariant + seam docs |
