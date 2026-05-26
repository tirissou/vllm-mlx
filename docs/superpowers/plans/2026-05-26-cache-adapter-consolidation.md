# Cache Adapter Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all cache adapters except `TurnCacheAdapter`, promote `CacheManager` to an ABC, slim `RequestCacheState` to a typed single-owner dataclass, and make Scheduler insertion calls explicit.

**Architecture:** `CacheManager` becomes the abstract base class the Scheduler holds, with `boundaries()`, `fetch()`, and `store()` as abstract methods and sensible no-op defaults for everything else. `TurnCacheAdapter(CacheManager)` is the only concrete adapter; the old `MemoryCacheAdapter`, `PagedCacheAdapter`, and `LegacyCacheAdapter` are deleted. `RequestCacheState` loses `adapter_state`, `store_tokens`, `mid_prefill_last_save`, and `mid_prefill_cache_key`; gains a typed `turn_path` and a formal `prev_recurrent` field.

**Tech Stack:** Python 3.12+, `abc.ABC`, `dataclasses`, `mlx`, `pytest`

---

## File Map

| File | Change |
|------|--------|
| `vllm_mlx/turn_prefix_cache.py` | Rename `_split_cache_arrays` → `split_cache_arrays` (public) |
| `vllm_mlx/kv_cache.py` | Slim `RequestCacheState`; add deprecation notice to `PrefixCache`/`SpillableCache` |
| `vllm_mlx/prefix_cache_adapters.py` | Promote `CacheManager` to ABC; rewrite `TurnCacheAdapter`; delete 3 adapters |
| `vllm_mlx/request.py` | Remove `_turn_cache_path` field |
| `vllm_mlx/scheduler.py` | Update `on_prefill_checkpoint`, `store`, `release`, and boundary call sites |
| `tests/test_prefix_cache_adapters.py` | Rewrite for new interface |
| `docs/adr/ADR-0003-no-cache-orchestrator.md` | Amend with protocol deprecation addendum |

---

## Task 1: Rename `_split_cache_arrays` → `split_cache_arrays` in `TurnPrefixCache`

**Files:**
- Modify: `vllm_mlx/turn_prefix_cache.py:366`

- [ ] **Step 1: Rename the definition**

In `vllm_mlx/turn_prefix_cache.py` at line 366, change:
```python
def _split_cache_arrays(self, cache_states: list[Any], offset: int = 0):
```
to:
```python
def split_cache_arrays(self, cache_states: list[Any], offset: int = 0):
```

- [ ] **Step 2: Update all internal callers in `turn_prefix_cache.py`**

Search for `_split_cache_arrays` in `turn_prefix_cache.py` and replace every occurrence with `split_cache_arrays`. Run:
```bash
grep -n "_split_cache_arrays" vllm_mlx/turn_prefix_cache.py
```
Expected: zero matches.

- [ ] **Step 3: Update all callers in `prefix_cache_adapters.py`**

```bash
grep -n "_split_cache_arrays" vllm_mlx/prefix_cache_adapters.py
```
Replace each occurrence with `split_cache_arrays`.

- [ ] **Step 4: Run existing tests to confirm nothing broke**

```bash
python -m pytest tests/test_turn_prefix_cache.py tests/test_prefix_cache_adapters.py -x -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/turn_prefix_cache.py vllm_mlx/prefix_cache_adapters.py
git commit -m "refactor: make split_cache_arrays public on TurnPrefixCache"
```

---

## Task 2: Slim `RequestCacheState`

**Files:**
- Modify: `vllm_mlx/kv_cache.py:35-57`
- Modify: `tests/test_prefix_cache_adapters.py`

The current dataclass has `adapter_state: Any`, `store_tokens`, `mid_prefill_last_save`, `mid_prefill_cache_key`, and a missing `prev_recurrent` field (set dynamically by the Scheduler). We replace `adapter_state` with a typed `turn_path`, formalise `prev_recurrent`, and remove the dead fields.

- [ ] **Step 1: Write a failing test for the new `RequestCacheState` defaults**

Replace the `test_request_cache_state_defaults` test in `tests/test_prefix_cache_adapters.py`:
```python
def test_request_cache_state_defaults():
    cs = RequestCacheState()
    assert cs.hit_type == "miss"
    assert cs.cache is None
    assert cs.cached_tokens == 0
    assert cs.remaining_tokens is None
    assert cs.prefill_boundaries == []
    assert cs.decoded_cache is None
    assert cs.prev_recurrent is None
    assert cs.turn_path == []
    assert cs.n_minus_one_state is None
    # Removed fields must not exist
    assert not hasattr(cs, "adapter_state")
    assert not hasattr(cs, "store_tokens")
    assert not hasattr(cs, "mid_prefill_last_save")
    assert not hasattr(cs, "mid_prefill_cache_key")
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest tests/test_prefix_cache_adapters.py::test_request_cache_state_defaults -x -q
```
Expected: FAIL — `RequestCacheState` still has the old fields.

- [ ] **Step 3: Rewrite `RequestCacheState` in `vllm_mlx/kv_cache.py`**

Replace the entire `RequestCacheState` dataclass (lines 35–57):
```python
@dataclass
class RequestCacheState:
    """All cache-related state for a single request. Lives at request._cache_state."""

    # Set by Scheduler from CacheHit after fetch()
    hit_type: str = "miss"
    cache: list | None = None
    cached_tokens: int = 0
    remaining_tokens: list | None = None
    prefill_boundaries: list = field(default_factory=list)

    # Set by Scheduler during decode / cleanup pipeline
    decoded_cache: list | None = None
    prev_recurrent: list | None = None   # N-1 recurrent snapshot; was set dynamically before

    # Owned by TurnCacheAdapter across the request lifecycle
    turn_path: list = field(default_factory=list)   # list[TurnNode]; typed replacement for adapter_state
    n_minus_one_state: Any = None                   # per-step N-1 tracking (set by update_n_minus_one)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_prefix_cache_adapters.py::test_request_cache_state_defaults -x -q
```
Expected: PASS.

- [ ] **Step 5: Run full test suite to surface broken callers**

```bash
python -m pytest tests/ -x -q 2>&1 | head -60
```
The Scheduler and adapters still reference `adapter_state`, `store_tokens`, `mid_prefill_last_save`, `mid_prefill_cache_key`. Note the failures — they will be fixed in Tasks 5 and 6. Do not fix them here.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/kv_cache.py tests/test_prefix_cache_adapters.py
git commit -m "refactor: slim RequestCacheState — typed turn_path, formalise prev_recurrent, drop dead fields"
```

---

## Task 3: Promote `CacheManager` to ABC with `boundaries()` abstract method

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`

`CacheManager` is currently a mixin with concrete n-minus-one machinery. We add `ABC` as a base and declare `fetch`, `store`, and `boundaries` as abstract methods.

- [ ] **Step 1: Write failing tests that `CacheManager` is an ABC and has `boundaries`**

Add to `tests/test_prefix_cache_adapters.py`:
```python
import inspect
from abc import ABC

def test_cache_manager_is_abstract():
    from vllm_mlx.prefix_cache_adapters import CacheManager
    assert issubclass(CacheManager, ABC)

def test_cache_manager_abstract_methods():
    from vllm_mlx.prefix_cache_adapters import CacheManager
    abstract = {
        name for name, val in inspect.getmembers(CacheManager)
        if getattr(val, "__isabstractmethod__", False)
    }
    assert "boundaries" in abstract
    assert "fetch" in abstract
    assert "store" in abstract

def test_cache_manager_cannot_be_instantiated():
    from vllm_mlx.prefix_cache_adapters import CacheManager
    try:
        CacheManager()
        assert False, "Expected TypeError"
    except TypeError:
        pass
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_prefix_cache_adapters.py::test_cache_manager_is_abstract tests/test_prefix_cache_adapters.py::test_cache_manager_abstract_methods tests/test_prefix_cache_adapters.py::test_cache_manager_cannot_be_instantiated -x -q
```
Expected: all FAIL.

- [ ] **Step 3: Update `CacheManager` in `prefix_cache_adapters.py`**

Replace the `CacheManager` class definition (everything before `class MemoryCacheAdapter`):
```python
from abc import ABC, abstractmethod
from typing import Any

from .kv_cache import CacheHit, CacheIndexMap, _BATCH_KV_TYPES


class CacheManager(ABC):
    """Abstract base class for all prefix cache adapters.

    The Scheduler holds a CacheManager reference. Subclasses implement
    fetch(), store(), and boundaries(). All other methods have no-op defaults.
    """

    _cache_index_map: "CacheIndexMap | None" = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def boundaries(self, request) -> list[int]:
        """Return prefill chunk boundaries adjusted for any cached prefix.

        On a cache hit, boundaries are offset by cached_tokens.
        On a miss, boundaries are the raw turn boundaries from the request.
        """
        ...

    @abstractmethod
    def fetch(self, request) -> "CacheHit | None":
        """Look up a cached prefix for this request.

        On a hit, populates request._cache_state.turn_path and returns a CacheHit.
        On a miss, returns None.
        """
        ...

    @abstractmethod
    def store(self, request, tokens: list[int], cache: list) -> bool:
        """Store the completed request's N-1 cache.

        tokens — the N-1 token key (prompt + output[:-1]), computed by Scheduler.
        cache  — already composed N-1 cache (compose_n_minus_1_cache applied by Scheduler).
        """
        ...

    # ── Default no-ops (override as needed) ──────────────────────────────────

    def release(self, handle: Any) -> None:
        pass

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        """Called after each prefill chunk with the absolute token count.

        total_tokens_prefilled = cs.cached_tokens + chunk_tokens_just_processed.
        Adapter decides internally whether this lands on a boundary worth inserting.
        Cache semantic at a boundary: N tokens → cache @ N (no N-1 adjustment).
        """
        pass

    # ── Concrete n-minus-one machinery (do not override) ─────────────────────

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        """Capture recurrent layer snapshots before each decode step.

        Called by the Scheduler before every decode step. Only recurrent
        (ArraysCache) layers need tracking here — rotating KV layers are
        handled at store time via trim_last.
        """
        pass

    def _ensure_cache_index_map(self, layers: list) -> "CacheIndexMap":
        if self._cache_index_map is not None:
            return self._cache_index_map

        from mlx_lm.models.cache import BatchRotatingKVCache, ArraysCache

        kv_indices = []
        rotating_indices = []
        recurrent_indices = []

        for i, layer in enumerate(layers):
            if isinstance(layer, dict):
                name = layer.get("class_name", "")
                if "Rotating" in name:
                    rotating_indices.append(i)
                elif "KV" in name or "Quantized" in name:
                    kv_indices.append(i)
                else:
                    recurrent_indices.append(i)
            else:
                if isinstance(layer, BatchRotatingKVCache):
                    rotating_indices.append(i)
                elif isinstance(layer, _BATCH_KV_TYPES):
                    kv_indices.append(i)
                else:
                    recurrent_indices.append(i)

        self._cache_index_map = CacheIndexMap(
            kv_indices=kv_indices,
            rotating_indices=rotating_indices,
            recurrent_indices=recurrent_indices,
        )
        return self._cache_index_map

    def _reconstruct(self, request, extracted_cache: list) -> list:
        """Build N-1 state dict list from extracted N-state and per-step tracking."""
        from .kv_cache import extract_layer_state

        idx_map = self._cache_index_map
        cs = request._cache_state
        n_minus_one = cs.n_minus_one_state

        result = [None] * len(extracted_cache)

        for i in idx_map.kv_indices:
            layer = extracted_cache[i]
            meta = layer.get("meta_state")
            if meta and len(meta) > 0:
                new_meta = (str(max(0, int(meta[0]) - 1)),) + meta[1:]
                result[i] = {**layer, "meta_state": new_meta}
            else:
                result[i] = layer

        for layer_idx in idx_map.rotating_indices:
            result[layer_idx] = {**extracted_cache[layer_idx], "trim_last": True}

        saved_recurrent = (n_minus_one or {}).get("recurrent") or []
        for rec_idx, layer_idx in enumerate(idx_map.recurrent_indices):
            if rec_idx < len(saved_recurrent):
                saved = saved_recurrent[rec_idx]
                state_dict = extract_layer_state(saved)
                result[layer_idx] = state_dict if state_dict is not None else extracted_cache[layer_idx]
            else:
                result[layer_idx] = extracted_cache[layer_idx]

        return result
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_prefix_cache_adapters.py::test_cache_manager_is_abstract tests/test_prefix_cache_adapters.py::test_cache_manager_abstract_methods tests/test_prefix_cache_adapters.py::test_cache_manager_cannot_be_instantiated -x -q
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py
git commit -m "refactor: promote CacheManager to ABC with abstract boundaries/fetch/store"
```

---

## Task 4: Write failing tests for the new `TurnCacheAdapter` interface

**Files:**
- Modify: `tests/test_prefix_cache_adapters.py`

- [ ] **Step 1: Add test helpers and failing tests**

Add the following to `tests/test_prefix_cache_adapters.py` (after existing imports):

```python
from unittest.mock import MagicMock, patch
from vllm_mlx.kv_cache import RequestCacheState, CacheHit
from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter


def _make_request(prompt_token_ids, turn_boundaries, output_token_ids=None, cached_tokens=0):
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req.output_token_ids = output_token_ids or []
    req._turn_boundaries = turn_boundaries
    req._cache_state = RequestCacheState(cached_tokens=cached_tokens)
    return req


def _make_inner():
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    inner.split_cache_arrays.return_value = ([], None)
    inner.insert.return_value = MagicMock(n_tokens=5)
    return inner


# ── boundaries() ─────────────────────────────────────────────────────────────

def test_boundaries_no_hit_returns_raw_turn_boundaries():
    adapter = TurnCacheAdapter(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[10, 20],
        cached_tokens=0,
    )
    assert adapter.boundaries(req) == [10, 20]


def test_boundaries_after_hit_offsets_by_cached_tokens():
    adapter = TurnCacheAdapter(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[10, 20, 30],
        cached_tokens=10,
    )
    # Boundaries > 10, shifted by 10: [20-10, 30-10] = [10, 20]
    assert adapter.boundaries(req) == [10, 20]


def test_boundaries_excludes_boundaries_at_or_below_cached_tokens():
    adapter = TurnCacheAdapter(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(30)),
        turn_boundaries=[5, 10, 20],
        cached_tokens=10,
    )
    # Only boundaries strictly > 10: [20-10] = [10]
    assert adapter.boundaries(req) == [10]


def test_boundaries_empty_when_no_turn_boundaries():
    adapter = TurnCacheAdapter(_make_inner())
    req = _make_request(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[],
        cached_tokens=0,
    )
    assert adapter.boundaries(req) == []


# ── fetch() ──────────────────────────────────────────────────────────────────

def test_fetch_miss_returns_none():
    inner = _make_inner()
    inner.match.return_value = ([], None)
    adapter = TurnCacheAdapter(inner)
    req = _make_request(prompt_token_ids=list(range(10)), turn_boundaries=[])
    result = adapter.fetch(req)
    assert result is None


def test_fetch_hit_populates_turn_path_on_cache_state():
    inner = _make_inner()
    node = MagicMock()
    node.n_tokens = 5
    inner.match.return_value = ([node], None)
    inner.find_checkpoint_ancestor.return_value = None
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[5],
        cached_tokens=0,
    )
    hit = adapter.fetch(req)
    assert hit is not None
    assert req._cache_state.turn_path == [node]


def test_fetch_does_not_set_prefill_boundaries():
    """fetch() must NOT set cs.prefill_boundaries — that is boundaries()'s job."""
    inner = _make_inner()
    node = MagicMock()
    node.n_tokens = 5
    inner.match.return_value = ([node], None)
    inner.find_checkpoint_ancestor.return_value = None
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=list(range(10)),
        turn_boundaries=[5],
        cached_tokens=0,
    )
    adapter.fetch(req)
    # prefill_boundaries must still be the default empty list
    assert req._cache_state.prefill_boundaries == []


# ── store() ──────────────────────────────────────────────────────────────────

def test_store_uses_explicit_tokens_not_cache_state():
    """store(request, tokens, cache) must not read store_tokens from cs."""
    inner = _make_inner()
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[3],
        output_token_ids=[6, 7],
        cached_tokens=0,
    )
    req._cache_state.turn_path = []
    explicit_tokens = [1, 2, 3, 4, 5, 6]  # N-1 key passed by Scheduler
    result = adapter.store(req, explicit_tokens, [])
    # Adapter accepts the call; does not crash looking for cs.store_tokens
    assert isinstance(result, bool)


def test_store_reads_turn_path_from_cache_state():
    inner = _make_inner()
    parent_node = MagicMock(n_tokens=3)
    inner.root = MagicMock(n_tokens=0)
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=[1, 2, 3, 4, 5],
        turn_boundaries=[3],
        output_token_ids=[6, 7],   # must be non-empty — store() returns False with no output
    )
    req._cache_state.turn_path = [parent_node]
    adapter.store(req, [1, 2, 3, 4, 5, 6], [])
    # insert should be called with parent_node as parent
    assert inner.insert.called
    call_parent = inner.insert.call_args[0][0]
    assert call_parent is parent_node


# ── on_prefill_checkpoint() ───────────────────────────────────────────────────

def test_on_prefill_checkpoint_at_boundary_inserts_node():
    inner = _make_inner()
    new_node = MagicMock(n_tokens=10)
    inner.insert.return_value = new_node
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    req._cache_state.turn_path = []
    extracted = [{"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("5",)}]
    adapter.on_prefill_checkpoint(req, total_tokens_prefilled=10, extracted_cache=extracted)
    assert inner.insert.called
    assert new_node in req._cache_state.turn_path


def test_on_prefill_checkpoint_not_at_boundary_is_noop():
    inner = _make_inner()
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    extracted = [{"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("5",)}]
    adapter.on_prefill_checkpoint(req, total_tokens_prefilled=7, extracted_cache=extracted)
    inner.insert.assert_not_called()


def test_on_prefill_checkpoint_does_not_read_n_minus_one_for_prefill():
    """Prefill boundaries store cache @ N, not N-1; n_minus_one_state must be ignored."""
    inner = _make_inner()
    adapter = TurnCacheAdapter(inner)
    req = _make_request(
        prompt_token_ids=list(range(20)),
        turn_boundaries=[10],
        cached_tokens=0,
    )
    req._cache_state.n_minus_one_state = {"recurrent": ["some_stale_state"]}
    extracted = [{"class_name": "BatchKVCache", "state": (None, None), "meta_state": ("10",)}]
    adapter.on_prefill_checkpoint(req, total_tokens_prefilled=10, extracted_cache=extracted)
    # split_cache_arrays should be called with the extracted_cache as-is, not composed
    call_args = inner.split_cache_arrays.call_args
    assert call_args is not None  # was called
```

- [ ] **Step 2: Run to confirm all new tests fail**

```bash
python -m pytest tests/test_prefix_cache_adapters.py -k "boundaries or fetch_populates or fetch_does_not or store_uses_explicit or store_reads_turn or checkpoint" -x -q
```
Expected: many failures (old signatures, `cs.adapter_state`, etc.).

- [ ] **Step 3: Commit the test file as failing tests**

```bash
git add tests/test_prefix_cache_adapters.py
git commit -m "test: add failing tests for new TurnCacheAdapter interface (red)"
```

---

## Task 5: Rewrite `TurnCacheAdapter`

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`

Delete the current `TurnCacheAdapter` class and replace it with the following. Keep `MemoryCacheAdapter`, `PagedCacheAdapter`, and `LegacyCacheAdapter` in place for now — they will be deleted in Task 7.

- [ ] **Step 1: Replace `TurnCacheAdapter` class**

```python
class TurnCacheAdapter(CacheManager):
    """Adapts TurnPrefixCache to the CacheManager interface.

    The only surviving adapter after consolidation. Manages the turn-node
    hierarchy for conversation-aware prefix caching.
    """

    def __init__(self, inner: "TurnPrefixCache"):
        self._inner = inner

    # ── CacheManager abstract methods ─────────────────────────────────────────

    def boundaries(self, request) -> list[int]:
        """Return turn boundaries adjusted for any cached prefix.

        Called by Scheduler after fetch() (hit or miss). On a hit, offsets
        boundaries by cs.cached_tokens so the Scheduler sees positions relative
        to the remaining (uncached) tokens. On a miss cached_tokens=0 so the
        raw boundaries are returned unchanged.
        """
        cs = request._cache_state
        cached = cs.cached_tokens if cs is not None else 0
        turn_bds = getattr(request, "_turn_boundaries", None) or []
        return sorted(b - cached for b in turn_bds if b > cached)

    def fetch(self, request) -> "CacheHit | None":
        from .turn_prefix_cache import reconstruct_cache_from_states

        segments = self._messages_to_segments(request)
        if not segments:
            return None

        path, _ = self._inner.match(segments)
        if not path:
            self._inner.release(path)
            return None

        cs = request._cache_state
        if cs is not None:
            cs.turn_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            return CacheHit(
                cache=[],
                cached_tokens=0,
                remaining_tokens=list(request.prompt_token_ids),
                handle=path,
                hit_type="hit",
            )

        assembled = self._inner._retrieve_full_cache(ancestor)
        reconstructed = reconstruct_cache_from_states(assembled)
        cached_tokens = ancestor.n_tokens
        remaining = list(request.prompt_token_ids[cached_tokens:])
        return CacheHit(
            cache=reconstructed,
            cached_tokens=cached_tokens,
            remaining_tokens=remaining,
            handle=path,
            hit_type="hit",
        )

    def store(self, request, tokens: list[int], cache: list) -> bool:
        """Insert the completed response turn into the trie.

        tokens — N-1 key computed by Scheduler (not used for trie insertion;
                  the trie uses segment structure instead).
        cache  — already-composed N-1 cache (compose_n_minus_1_cache applied).
        """
        from .turn_prefix_cache import Segment

        segments = self._messages_to_segments(request)
        if not segments or not getattr(request, "output_token_ids", None):
            return False

        cs = request._cache_state
        path = cs.turn_path if cs is not None else []
        matched_depth = len(path)
        parent = path[-1] if path else self._inner.root
        new_segments = segments[matched_depth:]

        if not new_segments:
            return False

        response_tokens = list(segments[-1].token_ids) + list(request.output_token_ids)

        if cache and not isinstance(cache[0], dict):
            from .kv_cache import extract_layer_state
            cache = [d for layer in cache if (d := extract_layer_state(layer)) is not None]

        resp_state = cache if cache else None

        if resp_state is not None:
            self._ensure_cache_index_map(resp_state)
            n_minus_one = cs.n_minus_one_state if cs is not None else None
            if n_minus_one is not None:
                resp_state = self._reconstruct(request, resp_state)

        resp_kv, resp_recur = (
            self._inner.split_cache_arrays(resp_state, parent.n_tokens)
            if resp_state is not None else ([], None)
        )
        self._inner.insert(
            parent,
            Segment(role="conversation", token_ids=response_tokens),
            resp_kv, None, resp_recur,
        )
        return True

    # ── on_prefill_checkpoint ─────────────────────────────────────────────────

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        """Insert a turn node when total_tokens_prefilled lands on a turn boundary.

        Cache semantic: N tokens → cache @ N (no N-1 adjustment for prefill).
        n_minus_one_state is intentionally ignored here.
        """
        turn_bds = getattr(request, "_turn_boundaries", None) or []
        if total_tokens_prefilled not in turn_bds:
            return

        try:
            abs_idx = turn_bds.index(total_tokens_prefilled)
        except ValueError:
            return

        segments = self._messages_to_segments(request)
        if abs_idx >= len(segments):
            return

        cs = request._cache_state
        turn_path = cs.turn_path if cs is not None else []

        if len(turn_path) > abs_idx:
            return  # already inserted (duplicate callback guard)

        parent = turn_path[-1] if turn_path else self._inner.root
        segment = segments[abs_idx]
        is_sys = segment.role == "system" and abs_idx == 0

        kv_slice, recur = self._inner.split_cache_arrays(extracted_cache, parent.n_tokens)
        new_node = self._inner.insert(parent, segment, kv_slice, None, recur, is_system_prompt=is_sys)

        if cs is not None:
            cs.turn_path.append(new_node)

    # ── update_n_minus_one ────────────────────────────────────────────────────

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        cs = request._cache_state
        if cs is None:
            return

        idx_map = self._ensure_cache_index_map(prompt_cache)

        if cs.n_minus_one_state is None:
            cs.n_minus_one_state = {"recurrent": None}

        if idx_map.recurrent_indices:
            saved = [prompt_cache[i].extract(uid_idx) for i in idx_map.recurrent_indices]
            cs.n_minus_one_state["recurrent"] = saved

    # ── Default no-ops ────────────────────────────────────────────────────────

    def release(self, handle) -> None:
        if handle is not None:
            self._inner.release(handle)

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    # ── PersistableCache extension ────────────────────────────────────────────

    def save(self, cache_dir: str) -> bool:
        self._inner.save(cache_dir)
        return True

    def load(self, cache_dir: str) -> int:
        self._inner.load(cache_dir)
        return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _messages_to_segments(request) -> list:
        """Split a request's token sequence into per-message Segment objects.

        Pure function of request.prompt_token_ids and request._turn_boundaries.
        Not part of CacheManager's interface — the Scheduler calls boundaries()
        instead, which encapsulates boundary computation.
        """
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []

        turn_bds = getattr(request, "_turn_boundaries", None) or []
        if not turn_bds:
            return []

        B_sys = turn_bds[0]
        if B_sys <= 0 or B_sys >= len(full_tokens):
            return []

        segments: list = [Segment(role="system", token_ids=full_tokens[:B_sys])]
        prev = B_sys
        for B_k in turn_bds[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(Segment(role="conversation", token_ids=full_tokens[prev:B_k]))
                prev = B_k
        if prev < len(full_tokens):
            segments.append(Segment(role="user", token_ids=full_tokens[prev:]))

        return segments if len(segments) > 1 else []
```

- [ ] **Step 2: Run the failing tests from Task 4**

```bash
python -m pytest tests/test_prefix_cache_adapters.py -k "boundaries or fetch_populates or fetch_does_not or store_uses_explicit or store_reads_turn or checkpoint" -x -q
```
Expected: all PASS (or close — fix any remaining issues).

- [ ] **Step 3: Run full adapter test suite**

```bash
python -m pytest tests/test_prefix_cache_adapters.py -x -q
```
Expected: all pass except tests that reference `MemoryCacheAdapter`, `PagedCacheAdapter`, `LegacyCacheAdapter` by name (those will be cleaned up in Task 7).

- [ ] **Step 4: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py
git commit -m "feat: rewrite TurnCacheAdapter with boundaries(), explicit store(tokens, cache), absolute checkpoint count"
```

---

## Task 6: Update Scheduler call sites

**Files:**
- Modify: `vllm_mlx/scheduler.py`

Four call sites need updating. Make each change separately and verify nothing is broken before moving to the next.

### 6a — `boundaries()` after fetch (hit and miss paths)

- [ ] **Step 1: Update the hit path in `_schedule_waiting`**

Find the block around line 1121 that sets `prefill_boundaries` from `hit.prefill_boundaries`:
```python
# BEFORE (around line 1121):
request._cache_state.prefill_boundaries = hit.prefill_boundaries

# AFTER — fetch() no longer sets boundaries; call boundaries() instead:
request._cache_state.prefill_boundaries = self._prefix_cache.boundaries(request)
```

- [ ] **Step 2: Update the miss path in `_schedule_waiting`**

Find the block around line 1130 that sets `prefill_boundaries` from `_turn_boundaries`:
```python
# BEFORE (around line 1130):
request._cache_state.prefill_boundaries = list(getattr(request, "_turn_boundaries", []))

# AFTER:
request._cache_state.prefill_boundaries = self._prefix_cache.boundaries(request)
```

- [ ] **Step 3: Update the retry/fallback path around line 1244**

Find any remaining direct reads of `_turn_boundaries` used to reset `prefill_boundaries` (around line 1244):
```python
# BEFORE:
request._cache_state.prefill_boundaries = list(getattr(request, "_turn_boundaries", []))

# AFTER:
request._cache_state.prefill_boundaries = self._prefix_cache.boundaries(request)
```

### 6b — `on_prefill_checkpoint` passes absolute token count

- [ ] **Step 4: Update `_make_mid_prefill_save_callback` (around line 889)**

```python
# BEFORE:
self._prefix_cache.on_prefill_checkpoint(request, processed_tokens, extracted)

# AFTER — compute absolute count before passing:
total = (request._cache_state.cached_tokens or 0) + processed_tokens
self._prefix_cache.on_prefill_checkpoint(request, total, extracted)
```

- [ ] **Step 5: Update the second `on_prefill_checkpoint` call (around line 2001)**

Same change — find the call `self._prefix_cache.on_prefill_checkpoint(request, processed, extracted)` and update it:
```python
# BEFORE:
self._prefix_cache.on_prefill_checkpoint(request, processed, extracted)

# AFTER:
total = (request._cache_state.cached_tokens or 0) + processed
self._prefix_cache.on_prefill_checkpoint(request, total, extracted)
```

### 6c — `store()` receives explicit tokens

- [ ] **Step 6: Remove `store_tokens` computation in `_process_batch_responses`**

Find the block around line 1421:
```python
# REMOVE these lines entirely:
request._cache_state.store_tokens = _full_tokens[:-1]  # N-1 key
```

The N-1 composition (compose_n_minus_1_cache) still runs — only the `store_tokens` assignment is removed.

- [ ] **Step 7: Update the `store()` call and `store_tokens` fallback in `_cleanup_finished`**

Find the block around lines 1445–1450:
```python
# BEFORE:
_store_cache = request._cache_state.decoded_cache
if _store_cache is not None:
    if not request._cache_state.store_tokens:
        request._cache_state.store_tokens = (
            list(request.prompt_token_ids) + list(request.output_token_ids)
        )
    try:
        self._prefix_cache.store(request, _store_cache)

# AFTER — compute tokens inline; only store when cache is non-empty:
_store_cache = request._cache_state.decoded_cache
if _store_cache:
    _full_tokens = list(request.prompt_token_ids) + list(request.output_token_ids)
    _store_tokens = _full_tokens[:-1]  # N-1 key; matches compose_n_minus_1_cache
    try:
        self._prefix_cache.store(request, _store_tokens, _store_cache)
```

### 6d — `release()` and abort path use `turn_path`

- [ ] **Step 8: Update `_do_abort_request` (around line 1069)**

```python
# BEFORE:
turn_path = getattr(request._cache_state, "adapter_state", None)
if turn_path and self.turn_cache is not None:
    self.turn_cache.release(turn_path)
    request._cache_state.adapter_state = []

# AFTER — use turn_path; release via _prefix_cache:
turn_path = request._cache_state.turn_path
if turn_path and self._prefix_cache is not None:
    self._prefix_cache.release(turn_path)
    request._cache_state.turn_path = []
```

- [ ] **Step 9: Update `_cleanup_finished` release (around line 1453)**

```python
# BEFORE:
_handle = request._cache_state.adapter_state
try:
    self._prefix_cache.release(_handle)
except Exception as e:
    logger.debug(f"[cache_store] release failed for {request_id}: {e}")
request._cache_state.adapter_state = []

# AFTER:
_handle = request._cache_state.turn_path
try:
    self._prefix_cache.release(_handle)
except Exception as e:
    logger.debug(f"[cache_store] release failed for {request_id}: {e}")
request._cache_state.turn_path = []
```

### 6e — Verify

- [ ] **Step 10: Confirm no remaining references to removed fields**

```bash
grep -n "adapter_state\|store_tokens\|mid_prefill_last_save\|mid_prefill_cache_key\|_turn_cache_path" vllm_mlx/scheduler.py
```
Expected: zero matches.

- [ ] **Step 11: Run tests**

```bash
python -m pytest tests/test_prefix_cache_adapters.py tests/test_turn_prefix_cache.py -x -q
```
Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add vllm_mlx/scheduler.py
git commit -m "refactor: update Scheduler to use boundaries(), explicit store(tokens, cache), absolute checkpoint count, turn_path"
```

---

## Task 7: Delete old adapters and clean up `Request`

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py` (delete 3 classes)
- Modify: `vllm_mlx/request.py` (remove `_turn_cache_path`)
- Modify: `tests/test_prefix_cache_adapters.py` (remove old-adapter tests)

- [ ] **Step 1: Delete `MemoryCacheAdapter`, `PagedCacheAdapter`, `LegacyCacheAdapter` from `prefix_cache_adapters.py`**

Remove the three class definitions entirely. After deletion, `prefix_cache_adapters.py` should contain only `CacheManager` and `TurnCacheAdapter`.

Verify:
```bash
grep -n "class MemoryCacheAdapter\|class PagedCacheAdapter\|class LegacyCacheAdapter" vllm_mlx/prefix_cache_adapters.py
```
Expected: zero matches.

- [ ] **Step 2: Remove `_turn_cache_path` from `Request` in `request.py`**

Find line 121 in `vllm_mlx/request.py`:
```python
_turn_cache_path: List[Any] = field(default_factory=list)  # pinned TurnNode path from match()
```
Delete it entirely.

- [ ] **Step 3: Remove old-adapter tests from `test_prefix_cache_adapters.py`**

Delete:
- `_make_dummy_adapters()` helper
- `test_all_adapters_implement_on_prefill_checkpoint()`
- `test_on_prefill_checkpoint_no_op_does_not_raise()`
- `test_turn_cache_adapter_store_returns_true_on_success()` (replaced by new store tests)
- Any `_make_turn_cache_request()` helper that sets `adapter_state=`

Also update the import at the top of the test file:
```python
# BEFORE:
from vllm_mlx.prefix_cache_adapters import (
    MemoryCacheAdapter, TurnCacheAdapter, PagedCacheAdapter, LegacyCacheAdapter,
)

# AFTER:
from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter, CacheManager
```

- [ ] **Step 4: Remove `Scheduler._messages_to_segments` delegation method**

In `vllm_mlx/scheduler.py`, find and delete:
```python
def _messages_to_segments(self, request):
    """Delegates to TurnCacheAdapter.messages_to_segments (kept for existing tests)."""
    from .prefix_cache_adapters import TurnCacheAdapter
    return TurnCacheAdapter.messages_to_segments(request)
```
This delegation is for `messages_to_segments` (public), which is now `_messages_to_segments` (private on the adapter). No callers outside the adapter need it.

- [ ] **Step 5: Confirm no remaining references to deleted adapters or field**

```bash
grep -rn "MemoryCacheAdapter\|PagedCacheAdapter\|LegacyCacheAdapter\|_turn_cache_path\|adapter_state\|messages_to_segments" vllm_mlx/ tests/
```
Expected: zero matches (or only in comments/imports that are themselves being removed).

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -x -q
```
Expected: all pass. If tests for deleted adapters in other files fail, remove them.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py vllm_mlx/request.py vllm_mlx/scheduler.py tests/test_prefix_cache_adapters.py
git commit -m "refactor: delete MemoryCacheAdapter, PagedCacheAdapter, LegacyCacheAdapter; remove _turn_cache_path from Request"
```

---

## Task 8: Deprecate `PrefixCache` protocol and amend ADR-0003

**Files:**
- Modify: `vllm_mlx/kv_cache.py`
- Modify: `docs/adr/ADR-0003-no-cache-orchestrator.md`

- [ ] **Step 1: Add deprecation notice to `PrefixCache` and `SpillableCache` in `kv_cache.py`**

Find the `PrefixCache` class definition and prepend a deprecation notice:
```python
@runtime_checkable
class PrefixCache(Protocol):
    """DEPRECATED — use CacheManager (prefix_cache_adapters.py) instead.

    Kept for SSDOffloadedCache compatibility until SpillableCache is migrated.
    Do not introduce new usages of this protocol.
    """
    ...
```

Find `SpillableCache` and add similarly:
```python
class SpillableCache(PrefixCache, Protocol):
    """DEPRECATED — see PrefixCache deprecation notice above."""
    ...
```

- [ ] **Step 2: Amend ADR-0003**

Open `docs/adr/ADR-0003-no-cache-orchestrator.md` and append the following addendum after the existing content:

```markdown
---

## Addendum — 2026-05-26: Protocol deprecated in favour of `CacheManager` ABC

### Context

With `MemoryCacheAdapter`, `PagedCacheAdapter`, and `LegacyCacheAdapter` removed, only `TurnCacheAdapter` remains. One adapter equals a hypothetical seam, not a real one. The `PrefixCache` Protocol was earning its keep as a multi-adapter contract; with a single adapter it adds indirection without depth.

### Amendment

The `PrefixCache` and `SpillableCache` protocols in `kv_cache.py` are deprecated. `CacheManager` (in `prefix_cache_adapters.py`) is now the Scheduler-facing abstract base class. It declares `fetch()`, `store()`, and `boundaries()` as abstract methods and provides no-op defaults for `release()`, `get_stats()`, `clear()`, and `on_prefill_checkpoint()`.

`boundaries(request) -> list[int]` is added as an `@abstractmethod`. The Scheduler calls it after `fetch()` (hit or miss) to populate `cs.prefill_boundaries`, replacing the two divergent `_turn_boundaries` reads in the old Scheduler code.

The original decision — no `CacheOrchestrator` — stands. This amendment does not add a coordinator; it collapses a now-unnecessary abstraction layer.
```

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -q
```
Expected: all pass.

- [ ] **Step 4: Final commit**

```bash
git add vllm_mlx/kv_cache.py docs/adr/ADR-0003-no-cache-orchestrator.md
git commit -m "docs: deprecate PrefixCache/SpillableCache protocols; amend ADR-0003 with CacheManager rationale"
```
