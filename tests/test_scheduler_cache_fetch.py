# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Scheduler._schedule_waiting cache fetch logic.

Verifies that the scheduler correctly consumes the bool return from
CacheManager.fetch() and that _cache_state fields are populated by the
fetch call (in-place mutation), not unpacked from a CacheHit object.
"""

import types
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class _FakeTokenizer:
    """Minimal tokenizer that returns fixed token ids."""

    def __init__(self, token_ids=None):
        self._token_ids = token_ids or [1, 2, 3, 4, 5]

    def encode(self, text, add_bos=True, add_eos=False):
        return self._token_ids


class _FakeModel:
    """Minimal model with an empty layers list."""

    def __call__(self, *args, **kwargs):
        pass

    def __init__(self):
        self.layers = []


def _make_scheduler(config=None):
    """Create a Scheduler with mocked model and tokenizer."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    cfg = config or SchedulerConfig()
    scheduler = Scheduler(model, tokenizer, cfg)
    # Disable the batch generator so _schedule_waiting puts requests back
    # when it can't insert. We only care about the cache-fetch branch.
    scheduler.batch_generator = None
    return scheduler


def _add_request(scheduler, request_id="r1", prompt_token_ids=None):
    """Add a request to the scheduler and return it."""
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3, 4, 5]
    req = Request(
        request_id=request_id,
        prompt=" ".join(str(t) for t in prompt_token_ids),
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=8),
    )
    scheduler.add_request(req)
    return req


# ------------------------------------------------------------------ #
# Test 1: _prefix_cache is None → miss state is set manually
# ------------------------------------------------------------------ #
def test_fetch_with_no_prefix_cache_sets_miss_state():
    """When _prefix_cache is None, the scheduler should set miss state."""
    scheduler = _make_scheduler()
    req = _add_request(scheduler)

    # _prefix_cache is None by default (no config enabling it)
    assert scheduler._prefix_cache is None

    # _cache_state.remaining_tokens starts as None (not yet fetched)
    assert req._cache_state.remaining_tokens is None

    # Call _schedule_waiting — it will try to fetch but _prefix_cache is None
    # It should set miss state manually.
    # Note: batch_generator is None so the request gets put back in waiting.
    # But the cache-fetch branch runs before the batch-generator check.
    scheduler._schedule_waiting()

    # Cache state should be set to miss
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.cached_tokens == 0
    assert req._cache_state.remaining_tokens == [1, 2, 3, 4, 5]
    assert req._cache_state.prefill_boundaries == []


# ------------------------------------------------------------------ #
# Test 2: fetch() returns True → hit state is populated by fetch
# ------------------------------------------------------------------ #
def test_fetch_hit_populates_cache_state_in_place():
    """When fetch() returns True, _cache_state fields are already set."""
    scheduler = _make_scheduler()
    req = _add_request(scheduler)

    def _fetch_and_populate(request):
        """Simulate what the real TurnCacheManager.fetch() does on a hit."""
        cs = request._cache_state
        cs.hit_type = "hit"
        cs.cache = [MagicMock(), MagicMock()]
        cs.cached_tokens = 3
        cs.turn_path = [MagicMock(), MagicMock()]
        cs.remaining_tokens = list(request.prompt_token_ids[3:])
        cs.prefill_boundaries = []
        return True

    mock_cache = MagicMock()
    mock_cache.fetch.side_effect = _fetch_and_populate
    mock_cache.boundaries.return_value = []
    scheduler._prefix_cache = mock_cache

    # _cache_state.remaining_tokens starts as None
    assert req._cache_state.remaining_tokens is None

    with patch("vllm_mlx.scheduler.validate_cache", return_value=True):
        scheduler._schedule_waiting()

    # fetch() should have been called exactly once
    mock_cache.fetch.assert_called_once_with(req)
    # boundaries() should NOT be called separately — fetch() handles it
    mock_cache.boundaries.assert_not_called()

    # _cache_state fields populated by fetch() should survive scheduling.
    # Note: the scheduler clears cache after scheduling (releases reference),
    # but the other fields remain set.
    assert req._cache_state.hit_type == "hit"
    assert req._cache_state.cache is None  # cleared after scheduling
    assert req._cache_state.cached_tokens == 3
    assert req._cache_state.remaining_tokens == [4, 5]
    # turn_path is NOT cleared after scheduling (it's used during decode)


# ------------------------------------------------------------------ #
# Test 3: fetch() returns False → miss state is populated by fetch
# ------------------------------------------------------------------ #
def test_fetch_miss_populates_cache_state_in_place():
    """When fetch() returns False, _cache_state fields are already set."""
    scheduler = _make_scheduler()
    req = _add_request(scheduler)

    def _fetch_miss(request):
        """Simulate what the real TurnCacheManager.fetch() does on a miss."""
        cs = request._cache_state
        cs.hit_type = "miss"
        cs.cached_tokens = 0
        cs.turn_path = []
        cs.remaining_tokens = list(request.prompt_token_ids)
        cs.prefill_boundaries = []
        return False

    mock_cache = MagicMock()
    mock_cache.fetch.side_effect = _fetch_miss
    mock_cache.boundaries.return_value = []
    scheduler._prefix_cache = mock_cache

    assert req._cache_state.remaining_tokens is None

    scheduler._schedule_waiting()

    # fetch() should have been called exactly once
    mock_cache.fetch.assert_called_once_with(req)
    # boundaries() should NOT be called separately — fetch() handles it
    mock_cache.boundaries.assert_not_called()

    # _cache_state should have been populated by fetch() in-place
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.cached_tokens == 0
    assert req._cache_state.remaining_tokens == [1, 2, 3, 4, 5]


# ------------------------------------------------------------------ #
# Test 4: Error-recovery branch after cache insert failure
# ------------------------------------------------------------------ #
def test_cache_insert_error_resets_state_with_turn_path_clear():
    """After a cache insert error, state should reset with turn_path cleared."""
    scheduler = _make_scheduler()
    req = _add_request(scheduler)

    # Pre-populate the cache state with a turn_path to simulate a hit
    req._cache_state.turn_path = [MagicMock(), MagicMock()]
    req._cache_state.cache = [MagicMock()]
    req._cache_state.hit_type = "hit"
    req._cache_state.cached_tokens = 3
    req._cache_state.remaining_tokens = [4, 5]
    req._cache_state.prefill_boundaries = []

    # Mock _prefix_cache with boundaries for the recovery branch
    mock_cache = MagicMock()
    mock_cache.boundaries.return_value = []
    scheduler._prefix_cache = mock_cache

    # Mock batch_generator.insert to raise an error, then succeed on retry
    mock_bg = MagicMock()
    mock_bg.insert.side_effect = [ValueError("shape mismatch"), [123]]
    scheduler.batch_generator = mock_bg

    # Patch _ensure_batch_generator to not recreate the batch generator,
    # otherwise our mock gets replaced with a real BatchGenerator.
    with patch.object(scheduler, "_ensure_batch_generator", return_value=None):
        with patch(
            "vllm_mlx.scheduler.validate_cache", return_value=True
        ):
            result = scheduler._schedule_waiting()

    # The request should have been scheduled after retry
    assert len(result) == 1
    assert req.batch_uid == 123

    # After error recovery, state should be reset to miss with turn_path cleared
    assert req._cache_state.cache is None
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.cached_tokens == 0
    assert req._cache_state.turn_path == []
    assert req._cache_state.remaining_tokens == [1, 2, 3, 4, 5]

    # boundaries() should NOT be called during recovery — fetch() handles it
    mock_cache.boundaries.assert_not_called()
