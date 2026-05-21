# Batch Prefill: Exchange-Level Turn Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `_compute_turn_boundaries` to emit boundaries only after system and assistant messages (not user messages), capture the system-node KV state in the scheduler init chunk, and add a regression guard for the exchange-level store→fetch round-trip.

**Architecture:** Three independent change sites. (1) `_compute_turn_boundaries` main path (Qwen3 `im_end_id` scan) gains a role-aware counter so it matches the fallback path's existing Option-C behaviour. (2) The scheduler init chunk already calls `mid_prefill_save`; we pass `n_to_process - 1` when boundary-capped so `on_prefill_checkpoint` keys `_boundary_states[B_sys]` correctly. (3) A new round-trip test confirms the adapter's store→match path works with exchange-level boundaries.

**Tech Stack:** Python, mlx, pytest, unittest.mock

---

## File Map

| File | Change |
|------|--------|
| `vllm_mlx/engine/batched.py:1027-1036` | Replace unconditional `im_end_id` scan with role-aware counter |
| `vllm_mlx/scheduler.py:688-689` | Pass `n_to_process - 1` to `mid_prefill_save` when init chunk was boundary-capped |
| `tests/test_turn_boundaries_qwen3.py:309` | `== 2` → `== 1` |
| `tests/test_turn_boundaries_qwen3.py:323` | `== 4` → `== 2` |
| `tests/test_turn_prefix_cache.py` | New test appended at end of file |

---

## Task 1: Update Qwen3 boundary-count tests to new expected values

The two existing tests in `test_turn_boundaries_qwen3.py` assert the OLD counts (every `<|im_end|>`).
Updating them to the correct Option-C counts makes them RED, which drives the implementation.

**Files:**
- Modify: `tests/test_turn_boundaries_qwen3.py:301-325`

- [ ] **Step 1: Change the assertion in `test_exact_boundary_count_two_messages`**

  In `tests/test_turn_boundaries_qwen3.py` at line 309, change:

  ```python
  assert len(boundaries) == 2, (
      f"Expected 2 boundaries (system, user), got {len(boundaries)}: {boundaries}"
  )
  ```

  to:

  ```python
  assert len(boundaries) == 1, (
      f"Expected 1 boundary (system only), got {len(boundaries)}: {boundaries}"
  )
  ```

  Also update the docstring on line 301 from `"system + user → exactly 2 boundaries (after system, after user)."` to `"system + user → exactly 1 boundary (after system only)."`.

- [ ] **Step 2: Change the assertion in `test_exact_boundary_count_multi_turn`**

  At line 323, change:

  ```python
  assert len(boundaries) == 4, (
      f"Expected 4 boundaries, got {len(boundaries)}: {boundaries}"
  )
  ```

  to:

  ```python
  assert len(boundaries) == 2, (
      f"Expected 2 boundaries (system + assistant), got {len(boundaries)}: {boundaries}"
  )
  ```

  Also update the docstring on line 313 from `"system + user + asst + user → exactly 4 boundaries."` to `"system + user + asst + user → exactly 2 boundaries (system + assistant)."`.

- [ ] **Step 3: Run the two tests to confirm they are RED**

  ```bash
  cd /path/to/repo
  pytest tests/test_turn_boundaries_qwen3.py::TestQwen3TurnBoundaries::test_exact_boundary_count_two_messages tests/test_turn_boundaries_qwen3.py::TestQwen3TurnBoundaries::test_exact_boundary_count_multi_turn -v
  ```

  Expected: both FAIL — actual count will be the old value (2 and 4 respectively).

  If either test is SKIPPED (Qwen3 tokenizer not available), install it:
  ```bash
  pip install mlx-lm
  huggingface-cli download Qwen/Qwen3-0.6B-4bit
  ```
  then re-run.

---

## Task 2: Fix `_compute_turn_boundaries` main path

The `im_end_id` scan in `batched.py` currently appends a boundary after every `<|im_end|>`.
Add an `im_end_count` counter to pair each token with `messages[im_end_count].role` and
emit a boundary only for `"system"` and `"assistant"`.

**Files:**
- Modify: `vllm_mlx/engine/batched.py:1022-1036`

- [ ] **Step 1: Replace the scan loop**

  At `vllm_mlx/engine/batched.py` lines 1027–1030, replace:

  ```python
              boundaries = []
              for i, tok in enumerate(full_tokens):
                  if tok == im_end_id:
                      boundaries.append(i + 1)
  ```

  with:

  ```python
              boundaries = []
              im_end_count = 0
              for i, tok in enumerate(full_tokens):
                  if tok == im_end_id:
                      if im_end_count < len(messages):
                          role = messages[im_end_count].get("role", "")
                          if role in ("system", "assistant"):
                              boundaries.append(i + 1)
                      im_end_count += 1
  ```

  Also remove the now-unused `nl_tokens` / `nl_id` lines (1024–1025) since they were a remnant of an earlier approach. The block should now read:

  ```python
              # Try im_end scan for Qwen3-like tokenizers
              if im_end_id is not None and im_end_id != getattr(tokenizer, "unk_token_id", None):
                  boundaries = []
                  im_end_count = 0
                  for i, tok in enumerate(full_tokens):
                      if tok == im_end_id:
                          if im_end_count < len(messages):
                              role = messages[im_end_count].get("role", "")
                              if role in ("system", "assistant"):
                                  boundaries.append(i + 1)
                          im_end_count += 1

                  logger.info(
                      f"[turn_cache] _compute_turn_boundaries: "
                      f"{len(full_tokens)} tokens, {len(boundaries)} boundaries"
                  )
                  return boundaries
  ```

- [ ] **Step 2: Run the two boundary-count tests to confirm they are now GREEN**

  ```bash
  pytest tests/test_turn_boundaries_qwen3.py::TestQwen3TurnBoundaries::test_exact_boundary_count_two_messages tests/test_turn_boundaries_qwen3.py::TestQwen3TurnBoundaries::test_exact_boundary_count_multi_turn -v
  ```

  Expected: both PASS.

- [ ] **Step 3: Run the full Qwen3 boundary test suite to catch regressions**

  ```bash
  pytest tests/test_turn_boundaries_qwen3.py -v
  ```

  Expected: all tests PASS (skips are acceptable where the tokenizer is unavailable).

- [ ] **Step 4: Commit**

  ```bash
  git add vllm_mlx/engine/batched.py tests/test_turn_boundaries_qwen3.py
  git commit -m "fix: emit turn boundaries only after system and assistant messages"
  ```

---

## Task 3: Add exchange-level round-trip regression test

This test confirms that storing a turn-1 exchange via `TurnCacheAdapter.store` and then
calling `cache.match` with turn-2 segments finds both the sys node and the exchange node
(depth ≥ 2). It starts GREEN and guards against future regressions in the adapter's
store→match path.

The test relies on `_make_extracted_state` defined at line 917 of `test_turn_prefix_cache.py`
(the full-format version with `class_name`/`class_ref`). It does NOT pass KV state through
`_split_cache_arrays` (to avoid int4 quantisation shape constraints in tests); it tests trie
structure only.

**Files:**
- Modify: `tests/test_turn_prefix_cache.py` (append after line 2305)

- [ ] **Step 1: Append the test**

  Add at the end of `tests/test_turn_prefix_cache.py`:

  ```python
  def test_exchange_level_round_trip_store_then_fetch():
      """Store turn-1 exchange; cache.match at turn-2 finds sys + exchange node (depth >= 2).

      Regression guard for the Option-C boundary scheme:
      response_tokens = segments[-1].token_ids + output_token_ids must hash-equal
      the next turn's exchange segment so the trie walks to depth >= 2.
      """
      from unittest.mock import MagicMock
      from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
      from vllm_mlx.prefix_cache_adapters import TurnCacheAdapter

      cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
      adapter = TurnCacheAdapter(cache)

      # Token layout — values are arbitrary; lengths chosen to be distinct.
      sys_tokens   = list(range(10))          # [0..9]       B_sys = 10
      u1           = list(range(10, 16))       # [10..15]     6 tokens
      gen_prompt1  = [200, 201]               # 2 tokens
      a1_output    = [300, 301, 302, 303]     # 4 tokens
      u2           = list(range(20, 24))       # [20..23]     4 tokens
      gen_prompt2  = [400, 401]               # 2 tokens

      B_sys = len(sys_tokens)                          # 10
      B_a1  = B_sys + len(u1) + len(gen_prompt1) + len(a1_output)  # 22

      # ── Turn 1: store exchange [sys | u1+gen_prompt1+a1_output] ─────────────
      # No KV state passed — we test trie structure, not KV restoration.
      req1 = MagicMock()
      req1.prompt_token_ids = sys_tokens + u1 + gen_prompt1
      req1._turn_boundaries  = [B_sys]
      req1._boundary_states  = {}   # no system KV checkpoint
      req1._turn_cache_path  = []
      req1._cache_state      = None
      req1.output_token_ids  = a1_output

      stored = adapter.store(req1, [])
      assert stored, "TurnCacheAdapter.store should return True for a valid request"

      # ── Turn 2: match [sys | u1+gen_prompt1+a1_output | u2+gen_prompt2] ─────
      req2 = MagicMock()
      req2.prompt_token_ids = (
          sys_tokens + u1 + gen_prompt1 + a1_output + u2 + gen_prompt2
      )
      req2._turn_boundaries = [B_sys, B_a1]
      req2._cache_state     = None

      segments2 = TurnCacheAdapter.messages_to_segments(req2)
      path, _   = cache.match(segments2)

      assert len(path) >= 2, (
          f"Expected depth-2 trie match (sys + exchange node), got depth {len(path)}: "
          f"{[n.token_ids for n in path]}"
      )
      cache.release(path)
  ```

- [ ] **Step 2: Run the new test to confirm it is GREEN**

  ```bash
  pytest tests/test_turn_prefix_cache.py::test_exchange_level_round_trip_store_then_fetch -v
  ```

  Expected: PASS.

- [ ] **Step 3: Run the full test_turn_prefix_cache suite to catch regressions**

  ```bash
  pytest tests/test_turn_prefix_cache.py -v
  ```

  Expected: all tests PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_turn_prefix_cache.py
  git commit -m "test: add exchange-level round-trip regression guard for TurnCacheAdapter"
  ```

---

## Task 4: Fix scheduler init chunk system-boundary capture

When the init chunk is boundary-capped (`_first_chunk < budget`), it processes exactly
`B_sys` tokens then calls `mid_prefill_save(uids[0], n_to_process, prompt_cache)`.
`on_prefill_checkpoint` checks `total_cached + 1 in _turn_boundaries`, so it needs
`total_cached = B_sys - 1` — i.e., we must pass `n_to_process - 1`.

The loop's subsequent boundary caps (`_dist = (_next_b - 1) - _total_pos`) are unaffected:
`partial["processed"]` still equals `n_to_process = B_sys`, so `_total_pos = B_sys` on the
first loop step and `_next_b = B_a1` is the correct next cap target.

**Files:**
- Modify: `vllm_mlx/scheduler.py:688-689`

- [ ] **Step 1: Patch the `mid_prefill_save` call in the init chunk**

  At `vllm_mlx/scheduler.py` lines 688–689, replace:

  ```python
                  if mid_prefill_save is not None and len(uids) == 1:
                      mid_prefill_save(uids[0], n_to_process, prompt_cache)
  ```

  with:

  ```python
                  if mid_prefill_save is not None and len(uids) == 1:
                      _save_processed = n_to_process
                      if _needs_boundary_split and _first_chunk < budget and n_to_process > 0:
                          _save_processed = n_to_process - 1
                      mid_prefill_save(uids[0], _save_processed, prompt_cache)
  ```

- [ ] **Step 2: Run the full test suite to confirm nothing regressed**

  ```bash
  pytest tests/test_turn_prefix_cache.py tests/test_turn_boundaries_qwen3.py -v
  ```

  Expected: all tests PASS.

- [ ] **Step 3: Commit**

  ```bash
  git add vllm_mlx/scheduler.py
  git commit -m "fix: capture system-node KV state from boundary-capped init chunk"
  ```
