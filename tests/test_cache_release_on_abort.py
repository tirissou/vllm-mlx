# SPDX-License-Identifier: Apache-2.0
"""Active Leaf pinning lifecycle: every termination path must release the pin.

Covers Scheduler.abort_request (deferred-abort) which fires _do_abort_request
on the executor thread. The contract: after _do_abort_request returns,
pinned_leaf(request_id) is None and the leaf is evictable.
"""
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from vllm_mlx.cache_types import KVLayerSegment
from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig
from vllm_mlx.turn_prefix_cache import Segment, TurnPrefixCache, TurnPrefixCacheConfig


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

class _FakeTokenizer:
    def __init__(self, token_ids=None):
        self._token_ids = token_ids or [1, 2, 3, 4, 5]

    def encode(self, text, add_bos=True, add_eos=False):
        return self._token_ids


class _FakeModel:
    def __call__(self, *args, **kwargs):
        pass

    def __init__(self):
        self.layers = []


def _make_scheduler(config=None):
    """Create a Scheduler with mocked model and tokenizer (no prefix cache)."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    cfg = config or SchedulerConfig()
    scheduler = Scheduler(model, tokenizer, cfg)
    scheduler.batch_generator = None
    return scheduler


def _dummy_kv_layer(idx: int) -> KVLayerSegment:
    return KVLayerSegment(
        keys=mx.zeros((1, 1, 1, 1)),
        values=mx.zeros((1, 1, 1, 1)),
        metadata={
            "class_name": "KVCache",
            "layer_index": idx,
            "merge_strategy": "concatenate",
            "n_tokens": 1,
        },
    )


def _make_turn_cache_manager_with_hit():
    """Build a TurnCacheManager whose trie has a leaf, ready for fetch() to hit.

    Returns (adapter, sys_tokens, user_tokens) so the caller can build a
    matching request.
    """
    trie = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0, kv_dtype="bf16"))
    adapter = TurnCacheManager(trie, policy=None, kv_group_size=64)

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))

    sys_node = trie.insert(
        trie.root,
        Segment(role="system", token_ids=sys_tokens),
        kv_data=[_dummy_kv_layer(0)],
        is_system_prompt=True,
    )
    trie.insert(
        sys_node,
        Segment(role="user", token_ids=user_tokens),
        kv_data=[_dummy_kv_layer(0)],
    )
    return adapter, sys_tokens, user_tokens


def _make_scheduler_with_pinned_request(request_id="req-abort-test"):
    """Build a Scheduler + TurnCacheManager, submit a request, pin a leaf.

    Returns (scheduler, request).  The adapter is accessible via
    scheduler._prefix_cache.
    """
    sched = _make_scheduler()

    adapter, sys_tokens, user_tokens = _make_turn_cache_manager_with_hit()
    all_tokens = sys_tokens + user_tokens

    req = Request(
        request_id=request_id,
        prompt=" ".join(str(t) for t in all_tokens),
        prompt_token_ids=all_tokens,
        sampling_params=SamplingParams(max_tokens=8),
    )
    req._turn_boundaries = [len(sys_tokens)]

    # Wire the adapter before add_request so the request's _cache_state is
    # initialised (add_request sets req._cache_state = RequestCacheState()).
    sched._prefix_cache = adapter
    sched.add_request(req)

    # Manually call fetch() (bypassing _schedule_waiting) to pin the leaf.
    # Stub _assemble and validate so no real MLX KV reconstruction is needed.
    with (
        patch.object(TurnCacheManager, "_assemble", staticmethod(lambda kv, rec, *a, **k: [object()])),
        patch.object(TurnCacheManager, "validate", lambda self, cache: True),
    ):
        hit = adapter.fetch(req)

    assert hit, "test setup: adapter.fetch() must return True to pin a leaf"
    assert adapter.pinned_leaf(req.request_id) is not None, (
        "test setup: leaf must be pinned after fetch()"
    )
    return sched, req


# ------------------------------------------------------------------ #
# Test 1: abort releases the pinned leaf
# ------------------------------------------------------------------ #
def test_abort_request_releases_pinned_leaf():
    sched, req = _make_scheduler_with_pinned_request()
    adapter = sched._prefix_cache
    assert adapter.pinned_leaf(req.request_id) is not None

    sched.abort_request(req.request_id)
    sched._process_pending_aborts()  # drain on this thread for the test

    assert adapter.pinned_leaf(req.request_id) is None


# ------------------------------------------------------------------ #
# Test 2: aborting an unknown id is a no-op
# ------------------------------------------------------------------ #
def test_abort_unknown_request_id_is_noop():
    """Aborting an unknown request_id must not pollute scheduler state or crash.

    Observable behavior: finished_req_ids is updated (audit trail), but
    the unknown id is never added to self.requests.
    """
    unknown_id = "never-submitted-id"

    # Case 1: no cache attached, no requests
    sched = _make_scheduler()
    assert unknown_id not in sched.requests, "test setup: unknown id must not be in requests"
    assert unknown_id not in sched.finished_req_ids, "test setup: unknown id must not be pre-finished"

    sched.abort_request(unknown_id)
    sched._process_pending_aborts()

    # Assertions: no phantom entry, but audit trail is recorded
    assert unknown_id not in sched.requests, "abort of unknown id must not create phantom entry"
    assert unknown_id in sched.finished_req_ids, "abort must record unknown id in finished_req_ids"

    # Case 2: with cache attached, should still not crash or pollute scheduler
    sched2 = _make_scheduler()
    adapter, sys_tokens, user_tokens = _make_turn_cache_manager_with_hit()
    sched2._prefix_cache = adapter

    unknown_id_2 = "never-submitted-either"
    sched2.abort_request(unknown_id_2)
    sched2._process_pending_aborts()

    # Even with cache, no phantom entry and audit trail recorded
    assert unknown_id_2 not in sched2.requests, "abort with cache must not create phantom entry"
    assert unknown_id_2 in sched2.finished_req_ids, "abort with cache must record in finished_req_ids"


# ------------------------------------------------------------------ #
# Test 3: after abort the previously-pinned leaf becomes evictable
# ------------------------------------------------------------------ #
def test_abort_running_request_releases_then_evicts():
    """After abort, the previously pinned leaf is evictable."""
    sched, req = _make_scheduler_with_pinned_request(request_id="req-evict-test")
    adapter = sched._prefix_cache
    leaf = adapter.pinned_leaf(req.request_id)
    assert leaf is not None
    assert not leaf.is_evictable  # pinned ⟹ not evictable

    sched.abort_request(req.request_id)
    sched._process_pending_aborts()

    # Public invariant: the leaf is now evictable (ref_count==0 and is_leaf).
    assert leaf.is_evictable
