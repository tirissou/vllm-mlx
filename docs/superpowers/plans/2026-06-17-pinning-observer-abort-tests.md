# Pinning Observer + Abort-Path Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `TurnCacheManager.pinned_leaf(request_id)` as the supported observer for the Active Leaf pinning invariant, migrate flagged implementation-coupled tests to use it (and the already-public `TurnNode.is_evictable`), and add four abort-path tests that verify every code path which terminates a request also releases its pinned leaf.

**Architecture:** One new public method on `TurnCacheManager` (a 3-line lookup into `_pinned_leaves`). Tests that previously asserted on private state (`_pinned_leaves` membership, `ref_count` integers) move to the observer + `is_evictable`. Four new tests exercise specific termination paths: explicit `release()`, scheduler `abort_request()`, decode-time exception inside `step()`, and EngineCore client cancellation. The new tests may **uncover existing bugs** (missing `release` calls on rare paths) — production fixes that surface ride along with the test that revealed them.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio.

## Global Constraints

- `pytest tests/` must pass at the end of every task.
- No production code change in Task 1 except the new observer method.
- If an abort-path test reveals a missing `release` call, fix it in the same task as the test, in the same commit. Do not defer.
- Preserve the synchronous `Scheduler.step()` invariant (ADR-0004) — no `await` in scheduler-path tests that would shift work onto the event loop thread.

---

### Task 1: Add `pinned_leaf` observer and migrate flagged tests

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py` (add observer ~line 246)
- Modify: `tests/test_prefix_cache_adapters.py`
- Modify: `tests/test_cache_manager_lifecycle.py`
- Modify: `tests/test_cache_translator.py`
- Modify: `tests/test_kv_cache.py`

**Interfaces:**
- Produces: `TurnCacheManager.pinned_leaf(request_id: str) -> TurnNode | None`

- [ ] **Step 1: Add the failing observer test**

Append to `tests/test_prefix_cache_adapters.py`:

```python
def test_pinned_leaf_returns_none_when_no_pin():
    adapter = _make_turn_cache_manager()  # whatever helper exists in the file
    assert adapter.pinned_leaf("nonexistent-request-id") is None


def test_pinned_leaf_returns_node_after_fetch_pin():
    adapter, req = _make_adapter_and_pinned_request()  # helper that fetches a hit
    leaf = adapter.pinned_leaf(req.request_id)
    assert leaf is not None
    assert not leaf.is_evictable  # public predicate replacing ref_count check


def test_pinned_leaf_returns_none_after_release():
    adapter, req = _make_adapter_and_pinned_request()
    adapter.release(req)
    assert adapter.pinned_leaf(req.request_id) is None
```

(If the helper names don't exist in the file, write the helpers using existing patterns in `test_cache_manager_lifecycle.py:_make_adapter_and_request`.)

- [ ] **Step 2: Run, confirm fail**

Run: `pytest tests/test_prefix_cache_adapters.py -k pinned_leaf -v 2>&1 | tail -10`

Expected: AttributeError — `pinned_leaf` does not exist.

- [ ] **Step 3: Add the observer**

In `vllm_mlx/prefix_cache_adapters.py`, after the `__init__` of `TurnCacheManager` (around line 246) add:

```python
    def pinned_leaf(self, request_id: str) -> "TurnNode | None":
        """Return the currently pinned leaf for a request, or None.

        Supported observer for the Active Leaf pinning invariant
        (CONTEXT.md). Tests and diagnostics use this; the underlying
        ``_pinned_leaves`` dict remains private.
        """
        return self._pinned_leaves.get(request_id)
```

- [ ] **Step 4: Run, confirm pass**

Run: `pytest tests/test_prefix_cache_adapters.py -k pinned_leaf -v 2>&1 | tail -10`

Expected: 3 PASS.

- [ ] **Step 5: Migrate the smell tests**

Apply the following substitutions across all four test files (`test_prefix_cache_adapters.py`, `test_cache_manager_lifecycle.py`, `test_cache_translator.py`, `test_kv_cache.py`):

| Old (smell) | New |
|---|---|
| `adapter._pinned_leaves[req.request_id]` | `adapter.pinned_leaf(req.request_id)` |
| `adapter._pinned_leaves.get(req.request_id)` | `adapter.pinned_leaf(req.request_id)` |
| `req.request_id not in adapter._pinned_leaves` | `adapter.pinned_leaf(req.request_id) is None` |
| `req.request_id in adapter._pinned_leaves` | `adapter.pinned_leaf(req.request_id) is not None` |
| `assert node.ref_count == 0` | `assert node.is_evictable` (if node is a leaf) |
| `assert node.ref_count == 1` | `assert not node.is_evictable` (when leaf, expressing the invariant) |
| `assert node.ref_count >= 1` | `assert not node.is_evictable` (same — the invariant is "cannot evict") |

Direct **mutations** of `_pinned_leaves` in tests (e.g., `adapter._pinned_leaves[req.request_id] = leaf` at `tests/test_prefix_cache_adapters.py:130`) should be **replaced** with a proper `fetch()` setup. If the original test was using the mutation to skip a `fetch`, write a small helper `_pin_leaf_for_request(adapter, request, leaf)` in the test file that does the equivalent via the public API. Do not leave private-state mutation in any test.

The exact lines flagged in the review (file:line shorthand):
- `test_cache_manager_lifecycle.py:146-152` — `ref_count` reads on `user_leaf` and `sys_node`
- `test_cache_manager_lifecycle.py:244-254` — `leaf_a.children.values()` + `ref_count` on `leaf_a`/`leaf_b`
- `test_cache_manager_lifecycle.py:299-333` — `req._cache_state.turn_path[-1].ref_count`
- `test_prefix_cache_adapters.py:130-134` — direct `_pinned_leaves` mutation + assert
- `test_cache_translator.py:197` and `:133-136`, `:314-335` — `result[0]._idx`, `metadata[...]` reads (these get fixed in the cache-translator PR's Task 4; if that PR has not landed, skip and add a TODO)
- `test_kv_cache.py:97-114`, `:196-210`, `:220-240` — `merged._idx` reads. Use `merged.offset` if exposed, or fail loud with the asserted contract (n_tokens after merge).

For `_idx` reads: the public field on `BatchQuantizedKVCache` / `QuantizedKVCache` is `.offset` (verify in `vllm_mlx/batch_quantized_kv_cache.py`). If `.offset` is correct, swap. If it isn't, the assertion was wrong — replace with the meaningful one (e.g., the `.state[0].shape[-2]` length the test actually cares about).

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -x 2>&1 | tail -30`

Expected: PASS. If any of the migrated tests fail, fix them by reading the public state correctly — never reach back into private fields.

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/prefix_cache_adapters.py tests/
git commit -m "feat(cache): add TurnCacheManager.pinned_leaf observer; migrate ref_count / _pinned_leaves test smells"
```

---

### Task 2: Abort-path test via `Scheduler.abort_request`

**Files:**
- Create: `tests/test_cache_release_on_abort.py`

**Interfaces:**
- Consumes: `TurnCacheManager.pinned_leaf` (Task 1)

- [ ] **Step 1: Write the failing test**

```python
"""Active Leaf pinning lifecycle: every termination path must release the pin.

Covers Scheduler.abort_request (deferred-abort) which fires _do_abort_request
on the executor thread. The contract: after _do_abort_request returns,
pinned_leaf(request_id) is None and the leaf is evictable.
"""
import pytest

from vllm_mlx.scheduler import Scheduler
# Reuse fixtures from existing scheduler tests (see test_scheduler_cache_fetch.py
# for the canonical pattern of building a Scheduler with a TurnCacheManager).


def _make_scheduler_with_pinned_request(tmp_path):
    """Build a Scheduler with a TurnCacheManager, submit a request, force a
    cache fetch hit so a leaf is pinned, return (scheduler, request)."""
    ...  # implementer fills in following test_scheduler_cache_fetch.py patterns


def test_abort_request_releases_pinned_leaf(tmp_path):
    sched, req = _make_scheduler_with_pinned_request(tmp_path)
    adapter = sched._prefix_cache
    assert adapter.pinned_leaf(req.request_id) is not None

    sched.abort_request(req.request_id)
    sched._process_pending_aborts()  # drain on this thread for the test

    assert adapter.pinned_leaf(req.request_id) is None


def test_abort_unknown_request_id_is_noop():
    sched = _make_scheduler()  # no requests
    # Must not raise, must not mutate cache state.
    sched.abort_request("never-submitted-id")
    sched._process_pending_aborts()


def test_abort_running_request_releases_then_evicts():
    """After abort, the previously pinned leaf is evictable."""
    sched, req = _make_scheduler_with_pinned_request_at_capacity()
    adapter = sched._prefix_cache
    leaf = adapter.pinned_leaf(req.request_id)

    sched.abort_request(req.request_id)
    sched._process_pending_aborts()

    # Public invariant check: the leaf is now evictable.
    assert leaf.is_evictable
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_cache_release_on_abort.py -v 2>&1 | tail -20`

Expected: PASS (the abort path at `scheduler.py:1147-1155` already calls `self._prefix_cache.release(request)`). If FAIL, the test has uncovered a missing release call. Fix it: add `self._prefix_cache.release(request)` in the missing branch and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cache_release_on_abort.py vllm_mlx/scheduler.py
git commit -m "test(cache): abort_request releases pinned leaf"
```

---

### Task 3: Decode-time exception releases pinned leaf

**Files:**
- Create: `tests/test_cache_release_on_decode_exception.py`

**Interfaces:**
- Consumes: `TurnCacheManager.pinned_leaf` (Task 1)

- [ ] **Step 1: Write the failing test**

```python
"""When step()'s decode path raises, the cache must still release the leaf.

The scheduler has a try/except around batch_generator.insert at
scheduler.py:1325 that calls release on the "cache_insert_error" branch.
This test verifies that path AND that exceptions raised *during decode*
(after insert succeeds) also release.
"""
import pytest


class _FakeBatchGenerator:
    """Batch generator that raises on step() to simulate a decode-time error."""

    def __init__(self):
        self.removed_uids: list = []

    def insert(self, *args, **kwargs):
        return ["fake-uid"]

    def step(self):
        raise RuntimeError("simulated decode failure")

    def remove(self, uids):
        self.removed_uids.extend(uids)


def test_decode_exception_releases_pinned_leaf(tmp_path):
    sched, req = _make_scheduler_with_pinned_request(tmp_path)
    sched.batch_generator = _FakeBatchGenerator()
    adapter = sched._prefix_cache
    assert adapter.pinned_leaf(req.request_id) is not None

    with pytest.raises(RuntimeError, match="simulated decode failure"):
        sched.step()

    # Contract: the exception propagated, but the cache leaf is released.
    assert adapter.pinned_leaf(req.request_id) is None
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_cache_release_on_decode_exception.py -v 2>&1 | tail -20`

Expected: likely FAIL — there is no documented try/finally around `step()` that calls release on arbitrary exceptions. Inspect `Scheduler.step()` (search for the body around `self.batch_generator.step()`). If no release-on-exception path exists, this test has uncovered a real bug.

- [ ] **Step 3: Fix if uncovered**

If the test fails, wrap the decode call in a try/finally that releases all in-flight requests' pinned leaves before re-raising. The simplest fix is at the outermost `step()` boundary:

```python
def step(self) -> ...:
    try:
        return self._step_inner()  # existing body
    except BaseException:
        if self._prefix_cache is not None:
            for req in list(self.running.values()):
                self._prefix_cache.release(req)
        raise
```

(Match existing code style and the actual control flow in `Scheduler.step()`.)

- [ ] **Step 4: Re-run, commit**

Run: `pytest tests/test_cache_release_on_decode_exception.py tests/ -x 2>&1 | tail -20`

Expected: PASS.

```bash
git add tests/test_cache_release_on_decode_exception.py vllm_mlx/scheduler.py
git commit -m "fix(scheduler): release pinned leaves when step() raises"
```

(If no production fix was needed because release was already wired correctly, drop the scheduler.py from the commit.)

---

### Task 4: EngineCore client-disconnect releases pinned leaf

**Files:**
- Create: `tests/test_engine_core_cancel_releases_leaf.py`

**Interfaces:**
- Consumes: `TurnCacheManager.pinned_leaf` (Task 1)

- [ ] **Step 1: Write the failing test**

Follow the pattern in `tests/test_engine_core_stream_safety.py` / `tests/test_simple_engine_cancel_serialization.py` for EngineCore-level integration.

```python
"""Client-disconnect cancellation must release the cache leaf.

EngineCore submits requests through an async API. When the upstream consumer
cancels (websocket disconnect, asyncio.CancelledError on the output stream),
the request flow eventually calls Scheduler.abort_request. This test wires
the full chain end-to-end and verifies that the cache observer sees release.
"""
import asyncio
import pytest

# Reuse the canonical EngineCore + tiny-model fixture from
# test_engine_core_stream_safety.py.


@pytest.mark.asyncio
async def test_client_disconnect_releases_pinned_leaf(tmp_path):
    engine = await _make_engine_with_turn_cache(tmp_path)
    try:
        # Submit and consume one token so a leaf has been pinned (via fetch).
        request_id = "test-disconnect-req"
        stream = await engine.submit_request(
            request_id=request_id,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],  # enough for a checkpoint
            sampling_params=...,  # mirror existing test
        )
        first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
        assert first_chunk is not None

        adapter = engine.scheduler._prefix_cache
        leaf_before = adapter.pinned_leaf(request_id)
        # Leaf may be None if this prompt was a cache miss with no checkpoint
        # yet; in that case force a checkpoint via a longer prompt or skip.
        if leaf_before is None:
            pytest.skip("no leaf pinned for this fixture; extend prompt to force checkpoint")

        # Simulate client disconnect: cancel the consumer task.
        await engine.abort_request(request_id)
        await asyncio.sleep(0.05)  # let executor thread drain

        assert adapter.pinned_leaf(request_id) is None
    finally:
        await engine.shutdown()
```

- [ ] **Step 2: Run, fix if uncovered, commit**

Run: `pytest tests/test_engine_core_cancel_releases_leaf.py -v 2>&1 | tail -20`

If the existing `EngineCore.abort_request` correctly routes to `Scheduler.abort_request`, this should PASS. If not, fix the routing in EngineCore so cancellation reaches the scheduler.

```bash
git add tests/test_engine_core_cancel_releases_leaf.py vllm_mlx/engine_core.py
git commit -m "test(engine): client disconnect releases pinned leaf end-to-end"
```

(Drop the engine_core.py addition if no fix was needed.)

---

## Self-review checklist

- All four abort paths covered: explicit `release()` (Task 1 Step 5 happy-path migration), `abort_request` (Task 2), decode-time exception (Task 3), client disconnect (Task 4).
- Public observer (`pinned_leaf`) added in one task, then used throughout.
- No private-state assertions remain in migrated tests (Task 1 Step 5 list is exhaustive).
- Production fixes uncovered by tests ride along with the test that revealed them; no orphan tests.
- `pytest tests/` is the gate after every task.
