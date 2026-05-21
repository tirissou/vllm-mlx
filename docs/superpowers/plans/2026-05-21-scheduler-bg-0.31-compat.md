# Scheduler BatchGenerator 0.31.x Compatibility Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the now-broken `_install_chunked_prefill` monkey patch and fix `_install_mtp` to integrate with mlx-lm 0.31.x's redesigned `BatchGenerator`, restoring mid-prefill saves and turn-boundary chunking via the new public API.

**Architecture:** Replace the 570-line `_install_chunked_prefill` with a lean `_InstrumentedBatchGenerator` subclass (~80 lines) that fires mid-prefill callbacks after each `_next()` chunk; use `insert_segments()` for turn-boundary segmentation instead of manual chunk-sizing; rewrite `_install_mtp` to patch `GenerationBatch._step` (the correct hook point in 0.31.x) and handle the new `(prompt_responses, generation_responses)` tuple return from `_next()`.

**Tech Stack:** mlx-lm 0.31.3, `BatchGenerator` / `PromptProcessingBatch` / `GenerationBatch` (all in `mlx_lm.generate`), `pytest --run-slow`, `mlx-community/Qwen3-0.6B-4bit` for local tests, hybrid model validation deferred to a separate machine.

---

## Background: what changed in mlx-lm 0.31.x

`BatchGenerator` was fully redesigned. The old internal attributes the patches hooked into are gone:

| Old (0.30.x) | New (0.31.x) | Impact |
|---|---|---|
| `batch_gen.active_batch` | `batch_gen._generation_batch` | Dead code in `step()` |
| `batch_gen._process_prompts` | Removed | `_install_chunked_prefill` guard evaluates False |
| `batch_gen._step(y, cache, ...)` | `GenerationBatch._step(self)` | `_install_mtp` patches wrong object |
| `_next()` returns `List[Response]` | `_next()` returns `(prompt_responses, gen_responses)` | `_mtp_next` crashes if MTP enabled |

`BatchGenerator` now natively interleaves prefill and decode (generation runs first each `_next()` call; a chunk of at most `prefill_step_size` prompt tokens is processed second). `_install_chunked_prefill` is therefore fully redundant for the interleaving concern — only the two features it provided on top of interleaving need a replacement: mid-prefill KV cache saves, and turn-boundary-aligned segmentation.

---

## File Map

| File | Change |
|---|---|
| `vllm_mlx/scheduler.py` | Delete `_install_chunked_prefill` (lines 178–748); add `_InstrumentedBatchGenerator` class before `Scheduler`; update `_create_batch_generator`; update `step()`; update `_schedule_waiting()`; rewrite `_install_mtp` hook points |
| `tests/test_batching.py` | Remove `_install_chunked_prefill` from import |
| `tests/test_scheduler_bg_compat.py` | New — six integration tests covering the behaviours described in each task |

---

## Task 1: Remove `_install_chunked_prefill` and stale BG references

**Files:**
- Modify: `vllm_mlx/scheduler.py`
- Modify: `tests/test_batching.py`

The 570-line `_install_chunked_prefill` function (lines 178–748) is silently disabled by the `chunked_compatible` guard in `_create_batch_generator`. Deleting it and its call site is pure subtraction; the new BG handles interleaved prefill/decode natively. Two other stale references also need cleanup:

1. `bg.prompt_progress_callback = _prefill_progress` (line 1521) — the attribute was only consumed by `_install_chunked_prefill`.
2. The `active_batch` lazy-eval block (lines 2409–2417) — `GenerationBatch.tokens` is a `List[List[int]]` (not lazy mx arrays), so the eval is a no-op and the `hasattr` guard already suppresses it.
3. The N-1 recurrent state snapshot block (lines 2320–2332) reads `batch_gen.active_batch`; replace with `batch_gen._generation_batch`.

- [ ] **Step 1: Delete `_install_chunked_prefill` from scheduler.py**

Delete lines 178–748 (the entire `_install_chunked_prefill` function). The function starts with `def _install_chunked_prefill(` and ends just before `def _install_mtp(`.

- [ ] **Step 2: Remove the call site and stale callback in `_create_batch_generator`**

Replace the block from line 1501 (`def _prefill_progress`) through line 1564 (the `elif need_chunked and not chunked_compatible` warning) with a single log line:

```python
        # mlx-lm >=0.31.x BatchGenerator natively interleaves prefill and
        # decode — chunked_prefill_tokens now only controls mid-prefill save
        # frequency (wired via _InstrumentedBatchGenerator in Task 2).
        logger.info(
            f"[batch_generator] prefill_step_size={self.config.prefill_step_size} "
            f"(native interleaving)"
        )
```

- [ ] **Step 3: Fix the N-1 recurrent snapshot in `step()`**

Replace lines 2320–2332:

```python
                    _ab = getattr(self.batch_generator, "active_batch", None)
                    if _ab is not None:
                        for _e, _uid in enumerate(_ab.uids):
                            _rid = self.uid_to_request_id.get(_uid)
                            _req = self.running.get(_rid) if _rid else None
                            if _req is not None:
                                _live_cache = _ab.extract_cache(_e)
                                if _live_cache:
                                    _req_recur = extract_recurrent_state(_live_cache)
                                    _req._cache_state.prev_recurrent = (
                                        extract_cache_states(_req_recur)
                                        if _req_recur else []
                                    )
```

with:

```python
                    _gb = getattr(self.batch_generator, "_generation_batch", None)
                    if _gb is not None and _gb.uids:
                        for _e, _uid in enumerate(_gb.uids):
                            _rid = self.uid_to_request_id.get(_uid)
                            _req = self.running.get(_rid) if _rid else None
                            if _req is not None:
                                _live_cache = _gb.extract_cache(_e)
                                if _live_cache:
                                    _req_recur = extract_recurrent_state(_live_cache)
                                    _req._cache_state.prev_recurrent = (
                                        extract_cache_states(_req_recur)
                                        if _req_recur else []
                                    )
```

- [ ] **Step 4: Replace the stale `active_batch` lazy-eval block**

Replace lines 2409–2418:

```python
        self._step_count += 1
        if self._step_count % effective_interval == 0:
            # Evaluate batch tokens to collapse lazy concatenation chains
            if (
                self.batch_generator is not None
                and hasattr(self.batch_generator, "active_batch")
                and self.batch_generator.active_batch is not None
                and hasattr(self.batch_generator.active_batch, "tokens")
            ):
                tokens = self.batch_generator.active_batch.tokens
                if tokens:
                    mx.eval(*tokens)
            mx.clear_cache()
```

with:

```python
        self._step_count += 1
        if self._step_count % effective_interval == 0:
            # GenerationBatch.tokens is List[List[int]] — no lazy eval needed.
            mx.clear_cache()
```

- [ ] **Step 5: Fix import in test_batching.py**

In `tests/test_batching.py`, remove `_install_chunked_prefill` from the import:

```python
# Before
from vllm_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
    _install_chunked_prefill,
)

# After
from vllm_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
)
```

- [ ] **Step 6: Write the tracer-bullet test**

Create `tests/test_scheduler_bg_compat.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Scheduler compatibility with mlx-lm >=0.31.x."""

import pytest
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture(scope="module")
def qwen3_small():
    try:
        from mlx_lm import load
        return load("mlx-community/Qwen3-0.6B-4bit")
    except Exception:
        pytest.skip("mlx-community/Qwen3-0.6B-4bit not available")


def _run_to_completion(scheduler, max_steps=200):
    """Step until no requests remain or max_steps reached."""
    for _ in range(max_steps):
        output = scheduler.step()
        if not scheduler.has_requests():
            return output
    return None


@pytest.mark.slow
def test_single_request_produces_tokens(qwen3_small):
    model, tokenizer = qwen3_small
    scheduler = Scheduler(model, tokenizer, SchedulerConfig())
    scheduler.add_request(Request(
        request_id="r1",
        prompt="Hello",
        sampling_params=SamplingParams(max_tokens=8),
    ))
    _run_to_completion(scheduler)
    req = scheduler.requests.get("r1") or scheduler._finished_requests.get("r1")
    assert req is not None and req.num_output_tokens > 0
```

- [ ] **Step 7: Run the tracer-bullet test**

```bash
cd /Users/tibo/Projects/vllm-mlx
pytest tests/test_scheduler_bg_compat.py::test_single_request_produces_tokens -v --run-slow
```

Expected: PASS

- [ ] **Step 8: Run existing test suite to check for regressions**

```bash
pytest tests/test_batching.py tests/test_continuous_batching.py -v
```

Expected: all previously-passing tests still pass (no import errors from removed `_install_chunked_prefill`).

- [ ] **Step 9: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_batching.py tests/test_scheduler_bg_compat.py
git commit -m "refactor: remove _install_chunked_prefill, fix stale active_batch refs for mlx-lm 0.31.x"
```

---

## Task 2: Add `_InstrumentedBatchGenerator` for mid-prefill saves

**Files:**
- Modify: `vllm_mlx/scheduler.py`

The new `BatchGenerator._next()` processes exactly one chunk of `prefill_step_size` tokens per call before returning, then interleaves generation. After `super()._next()` returns, `self._prompt_batch` contains the sequences still in prefill, and each `PromptProcessingBatch.Response` in `prompt_responses` carries `uid`, `progress=(processed, total)`, and `end_of_prompt: bool`. We subclass `BatchGenerator` to fire a per-sequence callback after each chunk for sequences still in prefill (`not resp.end_of_prompt`).

The callback signature is `(uid: int, processed_tokens: int, per_uid_cache: List[Any])`, matching what `_make_mid_prefill_save_callback` already builds.

- [ ] **Step 1: Add `_InstrumentedBatchGenerator` to `scheduler.py`**

Insert this class immediately before the `_install_mtp` function (after line 177, before the current `_install_mtp` definition, which is now immediately after the deleted `_install_chunked_prefill`):

```python
class _InstrumentedBatchGenerator(BatchGenerator):
    """BatchGenerator subclass that fires a mid-prefill callback after each chunk.

    After every _next() call, for each sequence still in the prompt batch
    (end_of_prompt=False), calls mid_prefill_callback(uid, processed, cache)
    where cache is the per-sequence KV state extracted from the shared batch
    cache.  Throttled by save_interval: only fires when at least save_interval
    new tokens have been processed since the last save for that uid.
    """

    def __init__(self, *args, mid_prefill_callback=None, save_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._mid_prefill_callback = mid_prefill_callback
        self._save_interval = save_interval
        self._uid_last_saved: dict = {}

    def _next(self):
        prompt_responses, gen_responses = super()._next()

        if self._mid_prefill_callback and prompt_responses:
            uid_to_idx = {uid: i for i, uid in enumerate(self._prompt_batch.uids)}
            for resp in prompt_responses:
                if resp.end_of_prompt or resp.uid not in uid_to_idx:
                    continue
                processed = resp.progress[0]
                last = self._uid_last_saved.get(resp.uid, 0)
                if self._save_interval > 0 and (processed - last) < self._save_interval:
                    continue
                idx = uid_to_idx[resp.uid]
                per_uid_cache = self._prompt_batch.extract_cache(idx)
                self._mid_prefill_callback(resp.uid, processed, per_uid_cache)
                self._uid_last_saved[resp.uid] = processed

        # Remove tracking for sequences that have left the prompt batch
        active = set(self._prompt_batch.uids)
        for uid in list(self._uid_last_saved):
            if uid not in active:
                del self._uid_last_saved[uid]

        return prompt_responses, gen_responses
```

- [ ] **Step 2: Wire `_InstrumentedBatchGenerator` in `_create_batch_generator`**

Replace the `BatchGenerator(...)` construction (around line 1510) with the subclass, passing the mid-prefill callback when configured:

```python
        save_interval = self.config.mid_prefill_save_interval
        mid_prefill_cb = None
        if self._prefix_cache is not None and (save_interval > 0 or self.turn_cache is not None):
            mid_prefill_cb = self._make_mid_prefill_save_callback(save_interval)
            logger.info(f"[mid_prefill_cache] enabled, interval={save_interval}")

        bg = _InstrumentedBatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            mid_prefill_callback=mid_prefill_cb,
            save_interval=save_interval,
        )
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_scheduler_bg_compat.py`:

```python
@pytest.mark.slow
def test_mid_prefill_save_fires_before_prefill_completes(qwen3_small):
    """Prefix cache must receive a checkpoint mid-prefill, not just at the end."""
    from unittest.mock import MagicMock, patch

    model, tokenizer = qwen3_small
    checkpoints = []

    config = SchedulerConfig(
        mid_prefill_save_interval=64,
        use_memory_aware_cache=True,
    )
    scheduler = Scheduler(model, tokenizer, config)

    # Intercept on_prefill_checkpoint to record calls
    original = scheduler._prefix_cache.on_prefill_checkpoint
    def _recording_checkpoint(request, processed_tokens, cache_states):
        checkpoints.append(processed_tokens)
        original(request, processed_tokens, cache_states)
    scheduler._prefix_cache.on_prefill_checkpoint = _recording_checkpoint

    # 512-token prompt with interval=64 → expect >=2 checkpoints before completion
    long_prompt_ids = list(range(512))
    scheduler.add_request(Request(
        request_id="r1",
        prompt_token_ids=long_prompt_ids,
        sampling_params=SamplingParams(max_tokens=4),
    ))

    steps_before_done = 0
    for _ in range(300):
        output = scheduler.step()
        if "r1" in output.finished_request_ids:
            break
        steps_before_done += 1

    assert len(checkpoints) >= 2, (
        f"Expected >=2 mid-prefill checkpoints, got {checkpoints}"
    )
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/test_scheduler_bg_compat.py::test_mid_prefill_save_fires_before_prefill_completes -v --run-slow
```

Expected: FAIL — no checkpoints fired (mid_prefill_save_interval not yet plumbed into `_InstrumentedBatchGenerator`).

> **Note:** This test will pass after Step 2 is applied. If Step 2 is already done when you reach this point, verify the test was failing before Step 2 by checking git diff.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_scheduler_bg_compat.py::test_mid_prefill_save_fires_before_prefill_completes -v --run-slow
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_scheduler_bg_compat.py
git commit -m "feat: add _InstrumentedBatchGenerator for mid-prefill KV cache saves"
```

---

## Task 3: Turn-boundary segmentation via `insert_segments()`

**Files:**
- Modify: `vllm_mlx/kv_cache.py`
- Modify: `vllm_mlx/prefix_cache_adapters.py`
- Modify: `vllm_mlx/scheduler.py`

The old `_install_chunked_prefill` computed the first chunk size to land exactly on a turn boundary, then had the mid-prefill callback capture the cache at that point. The new `BatchGenerator.insert_segments()` accepts a list of token-list segments per sequence; the BG processes each segment in order and emits `PromptProcessingBatch.Response(end_of_segment=True)` when it finishes a segment boundary before the last one. We split the prompt at each turn boundary and call `insert_segments()` instead of `insert()`. In `step()`, when `prompt_responses` contains `end_of_segment=True` for a uid, we trigger the turn-cache boundary save.

Boundaries are expressed as positions within `remaining_tokens` (already adjusted for any cache hit offset) so the Scheduler never needs to touch `cached_tokens` arithmetic. `TurnCacheAdapter.fetch()` owns this adjustment; other adapters leave `prefill_boundaries=[]`.

- [ ] **Step 0: Add `prefill_boundaries` to `CacheHit` and `RequestCacheState`**

In `vllm_mlx/kv_cache.py`, add one field to each dataclass:

```python
# CacheHit — append after hit_type
prefill_boundaries: list[int] = field(default_factory=list)
# Positions within remaining_tokens where prefill must pause for a boundary save.
# Computed by the cache adapter; [] means no constraints.

# RequestCacheState — append after remaining_tokens
prefill_boundaries: list[int] = field(default_factory=list)
```

In `vllm_mlx/prefix_cache_adapters.py`, in `TurnCacheAdapter.fetch()`, populate the field before returning. `cached_tokens` and `_turn_boundaries` are both in scope at that point:

```python
prefill_boundaries = sorted(
    b - cached_tokens
    for b in _turn_boundaries
    if b > cached_tokens
)
return CacheHit(
    ...,
    prefill_boundaries=prefill_boundaries,
)
```

In `scheduler.py`'s fetch path (the block that unpacks a `CacheHit` into `request._cache_state`), route the field in both branches — no arithmetic, just assignment:

```python
if hit:
    request._cache_state.prefill_boundaries = hit.prefill_boundaries
else:
    # miss: cached_tokens=0, so absolute positions == relative positions
    request._cache_state.prefill_boundaries = list(getattr(request, "_turn_boundaries", []))
```

- [ ] **Step 1: Add `_split_at_boundaries` helper inside `_schedule_waiting`**

In `_schedule_waiting`, just before the `try:` block that calls `self.batch_generator.insert(...)`, add this helper function and the conditional call:

```python
            def _split_at_boundaries(tokens, boundaries):
                """Split token list at pre-adjusted boundary positions."""
                if not boundaries:
                    return [tokens]
                segments = []
                prev = 0
                for b in sorted(boundaries):
                    if 0 < b < len(tokens):
                        segments.append(tokens[prev:b])
                        prev = b
                segments.append(tokens[prev:])
                return [s for s in segments if s]  # drop empty

            prefill_bds = request._cache_state.prefill_boundaries if request._cache_state else []
            segments = _split_at_boundaries(tokens_to_process, prefill_bds)
            use_segments = len(segments) > 1
```

Then replace the `uids = self.batch_generator.insert([tokens_to_process], **insert_kwargs)` call (and its fallback retry) with:

```python
            try:
                if use_segments:
                    uids = self.batch_generator.insert_segments(
                        [segments],
                        **insert_kwargs,
                    )
                else:
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        **insert_kwargs,
                    )
            except Exception as e:
                if cache_to_use is not None:
                    logger.warning(
                        f"[cache_insert_error] request={request.request_id[:12]} "
                        f"cache insert failed ({e}), retrying without cache"
                    )
                    cache_to_use = None
                    request._cache_state.cache = None
                    request._cache_state.cached_tokens = 0
                    request._cache_state.remaining_tokens = request.prompt_token_ids
                    request._cache_state.prefill_boundaries = list(getattr(request, "_turn_boundaries", []))
                    tokens_to_process = request.prompt_token_ids
                    segments = _split_at_boundaries(tokens_to_process, request._cache_state.prefill_boundaries)
                    use_segments = len(segments) > 1
                    insert_kwargs["caches"] = None
                    if use_segments:
                        uids = self.batch_generator.insert_segments(
                            [segments], **insert_kwargs
                        )
                    else:
                        uids = self.batch_generator.insert(
                            [tokens_to_process], **insert_kwargs
                        )
                else:
                    raise
```

- [ ] **Step 2: Handle `end_of_segment` in `step()`**

In `step()`, replace:

```python
                    if isinstance(result, tuple):
                        responses = result[1]  # generation_responses only
                    else:
                        responses = result
```

with:

```python
                    if isinstance(result, tuple):
                        prompt_responses, responses = result
                        self._handle_prompt_segment_ends(prompt_responses)
                    else:
                        responses = result
```

Add a new private method on `Scheduler`:

```python
    def _handle_prompt_segment_ends(self, prompt_responses) -> None:
        """Save turn-cache state at each completed prompt segment boundary."""
        if self._prefix_cache is None:
            return
        for resp in prompt_responses:
            if not resp.end_of_segment or resp.end_of_prompt:
                continue
            uid = resp.uid
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                continue
            request = self.requests.get(request_id)
            if not request:
                continue
            processed = resp.progress[0]
            # Extract cache from the prompt batch at this boundary
            bg = self.batch_generator
            pb = getattr(bg, "_prompt_batch", None)
            if pb is None or uid not in pb.uids:
                continue
            idx = pb.uids.index(uid)
            per_uid_cache = pb.extract_cache(idx)
            extracted = extract_cache_states(per_uid_cache)
            if extracted:
                cached_offset = (
                    request._cache_state.cached_tokens if request._cache_state else 0
                )
                self._prefix_cache.on_prefill_checkpoint(
                    request, cached_offset + processed, extracted
                )
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_scheduler_bg_compat.py`:

```python
@pytest.mark.slow
def test_turn_boundary_checkpoint_saved_at_each_boundary(qwen3_small):
    """With use_turn_cache, the turn cache must record state at each turn boundary."""
    model, tokenizer = qwen3_small
    config = SchedulerConfig(
        use_turn_cache=True,
        use_memory_aware_cache=False,
        chunked_prefill_tokens=256,  # required by turn cache validation
        turn_cache_stride=64,
        turn_cache_memory_gb=1.0,
    )
    scheduler = Scheduler(model, tokenizer, config)

    boundary_positions = []
    original_checkpoint = scheduler._prefix_cache.on_prefill_checkpoint
    def _record(request, processed_tokens, cache_states):
        boundary_positions.append(processed_tokens)
        original_checkpoint(request, processed_tokens, cache_states)
    scheduler._prefix_cache.on_prefill_checkpoint = _record

    # Simulate a 2-turn conversation: 256 + 256 = 512 tokens, boundary at 256
    turn1 = list(range(256))
    turn2 = list(range(256, 512))
    req = Request(
        request_id="r-turns",
        prompt_token_ids=turn1 + turn2,
        sampling_params=SamplingParams(max_tokens=4),
    )
    req._turn_boundaries = [256]
    scheduler.add_request(req)

    for _ in range(400):
        output = scheduler.step()
        if "r-turns" in output.finished_request_ids:
            break

    # Expect at least one checkpoint at or near position 256 (the boundary)
    assert any(abs(p - 256) <= 16 for p in boundary_positions), (
        f"No checkpoint near turn boundary 256; got checkpoints at {boundary_positions}"
    )
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/test_scheduler_bg_compat.py::test_turn_boundary_checkpoint_saved_at_each_boundary -v --run-slow
```

Expected: FAIL — no boundary checkpoint recorded.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_scheduler_bg_compat.py::test_turn_boundary_checkpoint_saved_at_each_boundary -v --run-slow
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_scheduler_bg_compat.py
git commit -m "feat: use insert_segments() for turn-boundary prefill chunking"
```

---

## Task 4: Fix `_install_mtp` for the 0.31.x `GenerationBatch._step` API

**Files:**
- Modify: `vllm_mlx/scheduler.py`

`_install_mtp` currently patches `batch_gen._step` (which no longer exists on `BatchGenerator`) and `batch_gen._next` (which exists but now returns a tuple). The correct hook points in 0.31.x are:

- **`batch_gen._generation_batch._step`** — the no-arg method on `GenerationBatch` that does the model forward, samples, and returns `(List[int], List[mx.array])`. This is where MTP logic (draft + verify) belongs.
- **`batch_gen._next`** — still exists; wrapping it to augment `gen_responses` with deferred drafts still works, but the wrapper must unwrap the tuple.

`GenerationBatch._step(self)` uses `self._current_tokens` (set at the top of `_step` from `self._next_tokens`), `self.prompt_cache`, `self.model`, `self.samplers`, `self.logits_processors`, and `self.fallback_sampler`. All MTP state (skip-state, deferred-drafts dict) lives in closures inside `_install_mtp` — that is unchanged.

The `GenerationBatch` instance (`batch_gen._generation_batch`) is created once in `BatchGenerator.__init__` and mutated in-place through `extend()` and `filter()` as sequences move through the system. Patching `._step` on the instance once at BG creation is persistent for the lifetime of the BG.

### Key API differences for `_mtp_step`

| Old signature | New signature |
|---|---|
| `_mtp_step(input_tokens, prompt_cache, samplers, logits_processors, tokens)` | `_mtp_step(self)` where `self` is the `GenerationBatch` |
| Returns `(mx.array, List[mx.array])` — primary tokens array + logprobs list | Returns `(List[int], List[mx.array])` — ints list + logprobs list |
| Reads batch size from `input_tokens.shape[0]` | Reads `self._current_tokens` (set before `_step` is called) |
| Model called as `model(input_tokens, cache=prompt_cache, return_hidden=True)` | `self.model(self._current_tokens[:, None], cache=self.prompt_cache, return_hidden=True)` |
| `self.active_batch.uids` for uid list | `self.uids` |

### Key API differences for `_mtp_next`

| Old | New |
|---|---|
| `responses = self._inner_next()` — flat list | `prompt_responses, gen_responses = self._inner_next()` — tuple |
| Iterates `responses` directly | Iterates `gen_responses`; returns `(prompt_responses, augmented_gen)` |
| `r.uid`, `r.finish_reason`, `r.cache_out` | `r.uid`, `r.finish_reason`, `r.prompt_cache` |

`GenerationBatch.Response` fields: `uid`, `token`, `logprobs`, `finish_reason`, `current_state`, `match_sequence`, `prompt_cache`, `all_tokens`.

- [ ] **Step 1: Rewrite `_mtp_step` inside `_install_mtp`**

Replace the existing `_mtp_step` function definition (inside `_install_mtp`, currently ~270 lines) with the following. Keep all closure variables (`_skip_state`, `_deferred_drafts`, `_mtp_stats`, `_draft_sampler`, `_rnn_snapshots` logic) unchanged — only the function signature and how state is accessed changes:

```python
    def _mtp_step(self):
        """Replacement for GenerationBatch._step with MTP always-advance strategy."""
        # Consume _next_tokens (mirroring upstream _step contract)
        self._current_tokens = self._next_tokens
        inputs = self._current_tokens
        batch_size = inputs.shape[0]

        # Skip MTP during prefill (multi-token input) or when cache doesn't
        # belong to the active generation batch (shouldn't happen here, but guard).
        if inputs.shape[0] == 0:
            return _orig_gen_step(self)

        skip = _skip_state[0]
        if skip is not None and skip["logits"].shape[0] != batch_size:
            skip = None
            _skip_state[0] = None

        if skip is not None:
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            _skip_state[0] = None
        else:
            model_output = self.model(inputs[:, None], cache=self.prompt_cache, return_hidden=True)
            if not isinstance(model_output, tuple):
                return _orig_gen_step(self)
            logits, hidden_states = model_output
            logits = logits[:, -1, :]

        # Apply logits processors
        if any(self.logits_processors):
            processed = []
            for e in range(batch_size):
                sl = logits[e : e + 1]
                for proc in self.logits_processors[e]:
                    token_ctx = self._token_context[e] if self._token_context else None
                    sl = proc(token_ctx, sl) if token_ctx is not None else sl
                processed.append(sl)
            logits = mx.concatenate(processed, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(self.samplers):
            samples = [
                (self.samplers[e] or self.fallback_sampler)(logprobs[e : e + 1])
                for e in range(batch_size)
            ]
            primary_tokens = mx.concatenate(samples, axis=0)
        else:
            primary_tokens = self.fallback_sampler(logprobs)

        current_uids = list(self.uids)

        try:
            draft_logits = self.model.mtp_forward(
                hidden_states[:, -1:, :],
                primary_tokens[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft_logprobs = draft_logits - mx.logsumexp(draft_logits, axis=-1, keepdims=True)
            draft_tokens = _draft_sampler(draft_logprobs)

            _rnn_snapshots = {}
            if not optimistic:
                for _ci, _c in enumerate(self.prompt_cache):
                    if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                        if hasattr(_c, "state"):
                            _rnn_snapshots[_ci] = [
                                s.copy() if s is not None else None for s in _c.state
                            ]

            verify_input = mx.concatenate(
                [primary_tokens[:, None], draft_tokens[:, None]], axis=1
            )
            verify_output = self.model(verify_input, cache=self.prompt_cache, return_hidden=True)
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits, verify_hidden = verify_output, None

            if optimistic:
                if verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    mx.async_eval(
                        _skip_state[0]["logits"], _skip_state[0]["hidden"],
                        draft_tokens, verify_lp,
                    )
                    for e in range(batch_size):
                        uid = current_uids[e]
                        _deferred_drafts[uid] = {
                            "token_array": draft_tokens[e : e + 1],
                            "logprobs": verify_lp[e],
                        }
                else:
                    _skip_state[0] = None
                _mtp_stats["accepted"] += 1
            else:
                verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
                mx.eval(verify_pred, draft_tokens)
                pred_list = verify_pred.tolist()
                draft_list = draft_tokens.tolist()
                all_accepted = pred_list == draft_list

                if all_accepted and verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    for e in range(batch_size):
                        _deferred_drafts[current_uids[e]] = {
                            "token": draft_list[e],
                            "logprobs": verify_lp[e],
                        }
                    _mtp_stats["accepted"] += 1
                else:
                    if _rnn_snapshots:
                        for c in self.prompt_cache:
                            if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
                                c.trim(2)
                        for _ci, _snap in _rnn_snapshots.items():
                            self.prompt_cache[_ci].state = _snap
                        rerun = self.model(primary_tokens[:, None], cache=self.prompt_cache, return_hidden=True)
                        if isinstance(rerun, tuple):
                            _, rerun_hidden = rerun
                            _skip_state[0] = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": rerun_hidden[:, -1:, :],
                            }
                            mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                        else:
                            _skip_state[0] = None
                    else:
                        for c in self.prompt_cache:
                            if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
                                c.trim(1)
                        if verify_hidden is not None:
                            _skip_state[0] = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": verify_hidden[:, 0:1, :],
                            }
                            mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                        else:
                            _skip_state[0] = None
                    for uid in current_uids:
                        _deferred_drafts.pop(uid, None)
                    _mtp_stats["rejected"] += 1

        except Exception as e:
            logger.debug(f"[MTP] draft/verify failed: {e}")
            _skip_state[0] = None
            _mtp_stats["errors"] += 1

        # Set _next_tokens to primary (consumed next step) and return current
        self._next_tokens = primary_tokens
        self._next_logprobs = list(logprobs)
        mx.async_eval(self._next_tokens, self._next_logprobs)

        # Add current inputs to tokens list (mirrors upstream _step)
        mx.eval(inputs, self._current_logprobs if self._current_logprobs else [])
        inputs_list = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs_list):
            sti.append(ti)
        return inputs_list, self._current_logprobs if self._current_logprobs else list(logprobs)
```

- [ ] **Step 2: Update the hook installation lines at the bottom of `_install_mtp`**

Replace:

```python
    batch_gen._step = _mtp_step
    batch_gen._next = _mtp_next
```

with:

```python
    _orig_gen_step = batch_gen._generation_batch._step
    batch_gen._generation_batch._step = _mtp_step
    batch_gen._inner_next = batch_gen._next
    batch_gen._next = _mtp_next
```

> `_orig_gen_step` is referenced as the fallback in `_mtp_step` when `return_hidden` is unsupported. Ensure this name is in scope by adding `_orig_gen_step = None` as a placeholder at the top of `_install_mtp`, then assigning it just before the hook installation lines.

- [ ] **Step 3: Update `_mtp_next` to handle the tuple return**

Replace the existing `_mtp_next` function with:

```python
    def _mtp_next(self=batch_gen):
        """Wrapper around _next that emits deferred MTP draft tokens."""
        if not self._generation_batch.uids:
            _skip_state[0] = None
            _deferred_drafts.clear()

        prev_deferred = {}
        if self._generation_batch.uids:
            for uid in self._generation_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        prompt_responses, gen_responses = self._inner_next()

        if not prev_deferred or not gen_responses:
            return prompt_responses, gen_responses

        augmented = []
        draft_end_uids = set()
        for r in gen_responses:
            uid = r.uid
            augmented.append(r)

            if r.finish_reason is not None:
                _deferred_drafts.pop(uid, None)
                prev_deferred.pop(uid, None)
                continue

            if uid in prev_deferred:
                draft_info = prev_deferred.pop(uid)
                draft_t = (
                    draft_info["token"]
                    if "token" in draft_info
                    else draft_info["token_array"].item()
                )
                draft_lp = draft_info["logprobs"]

                from mlx_lm.generate import GenerationBatch as _GB
                draft_finish = None
                gb = self._generation_batch
                if gb is not None and uid in gb.uids:
                    e = gb.uids.index(uid)
                    gb._num_tokens[e] = gb._num_tokens[e] + 1
                    if gb._num_tokens[e] >= gb.max_tokens[e]:
                        draft_finish = "length"
                        draft_end_uids.add(uid)

                from dataclasses import replace as _dc_replace
                draft_r = _dc_replace(
                    r,
                    token=draft_t,
                    logprobs=draft_lp,
                    finish_reason=draft_finish,
                    prompt_cache=None,
                )
                augmented.append(draft_r)

        if draft_end_uids and self._generation_batch.uids:
            keep = [e for e, u in enumerate(self._generation_batch.uids) if u not in draft_end_uids]
            self._generation_batch.filter(keep)

        return prompt_responses, augmented
```

- [ ] **Step 4: Add a `hybrid_only` marker and skeleton MTP test**

Add the marker to `pytest.ini` (or `pyproject.toml`):

```ini
# In pytest.ini or [tool.pytest.ini_options] in pyproject.toml:
markers =
    hybrid_only: requires a hybrid (MTP-capable) model; skip on pure-KV machines
```

Append to `tests/test_scheduler_bg_compat.py`:

```python
@pytest.mark.slow
@pytest.mark.hybrid_only
def test_mtp_produces_extra_tokens(qwen3_mtp_model):
    """With MTP enabled, at least one step should return 2 tokens for a request."""
    # qwen3_mtp_model fixture: loads a Qwen3-Next or Qwen3.5 model with MTP weights
    # and calls inject_mtp_support(). Defined on the hybrid machine's conftest.
    model, tokenizer = qwen3_mtp_model
    config = SchedulerConfig(enable_mtp=True)
    scheduler = Scheduler(model, tokenizer, config)
    scheduler.add_request(Request(
        request_id="r-mtp",
        prompt="Hello",
        sampling_params=SamplingParams(max_tokens=16),
    ))
    token_counts_per_step = []
    for _ in range(100):
        output = scheduler.step()
        total = sum(len(o.output_token_ids) for o in output.outputs)
        if total > 0:
            token_counts_per_step.append(total)
        if not scheduler.has_requests():
            break
    assert any(c > 1 for c in token_counts_per_step), (
        "MTP never produced more than 1 token in a single step"
    )
```

- [ ] **Step 5: Run MTP-unrelated tests to confirm no regression**

```bash
pytest tests/test_scheduler_bg_compat.py -v --run-slow -m "not hybrid_only"
```

Expected: all three non-MTP tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_scheduler_bg_compat.py
git commit -m "fix: rewrite _install_mtp hooks for mlx-lm 0.31.x GenerationBatch._step API"
```

---

## Task 5: Full regression pass

**Files:**
- No code changes — run only.

- [ ] **Step 1: Run full test suite (non-slow)**

```bash
pytest tests/ -v --ignore=tests/test_scheduler_bg_compat.py -x
```

Expected: same pass/fail profile as before this branch.

- [ ] **Step 2: Run the new integration tests**

```bash
pytest tests/test_scheduler_bg_compat.py -v --run-slow -m "not hybrid_only"
```

Expected: 3 tests pass (tracer bullet, mid-prefill save, turn boundary).

- [ ] **Step 3: Verify `ensure_mamba_support` is still a no-op**

```bash
python -c "from vllm_mlx.utils.mamba_cache import ensure_mamba_support; ensure_mamba_support(); print('ok')"
```

Expected output: `ok` (no errors, no patch applied).

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "test: add scheduler BG 0.31.x compat regression suite"
```

---

## Hybrid machine checklist

After porting to the hybrid machine (Qwen3-Next or Qwen3.5 with MTP weights):

- [ ] Add `qwen3_mtp_model` fixture to that machine's `conftest.py` (loads model + calls `inject_mtp_support`)
- [ ] Run `pytest tests/test_scheduler_bg_compat.py -v --run-slow` — all 4 tests should pass
- [ ] Manually verify that `_mtp_stats["accepted"]` / `["rejected"]` appear in logs and acceptance rate is reasonable (>50% for optimistic mode)
- [ ] Verify mid-prefill save fires for hybrid cache layers (`ArraysCache.state` is non-empty in checkpointed cache states)
