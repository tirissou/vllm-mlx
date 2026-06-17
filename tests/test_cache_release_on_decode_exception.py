# SPDX-License-Identifier: Apache-2.0
"""Pinned-leaf lifecycle: decode-time exceptions must release the pin.

After Scheduler.step() returns from a decode-time exception (the Exception
recovery branch in step()), the following must hold:
  - adapter.pinned_leaf(request_id) is None
  - the leaf whose ref_count dropped to 0 is evictable

step() catches Exception and calls _recover_from_generation_error() instead of
re-raising, so these tests must NOT use pytest.raises.
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
# Helpers (mirror pattern from test_cache_release_on_abort.py)
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
    """Build a TurnCacheManager whose trie has a leaf, ready for fetch() to hit."""
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


def _make_scheduler_with_running_pinned_request(request_id="req-decode-exc-test"):
    """Build a Scheduler with a pinned request already in the running dict.

    Returns (scheduler, request, adapter).  The request has a pinned leaf and
    lives in scheduler.running so that _recover_from_generation_error() will
    iterate over it.
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

    # Wire the adapter before add_request so _cache_state is initialised.
    sched._prefix_cache = adapter
    sched.add_request(req)

    # Pin the leaf manually (bypassing _schedule_waiting, which would need a
    # real batch_generator).
    with (
        patch.object(TurnCacheManager, "_assemble", staticmethod(lambda kv, rec, *a, **k: [object()])),
        patch.object(TurnCacheManager, "validate", lambda self, cache: True),
    ):
        hit = adapter.fetch(req)

    assert hit, "test setup: adapter.fetch() must return True to pin a leaf"
    assert adapter.pinned_leaf(req.request_id) is not None, (
        "test setup: leaf must be pinned after fetch()"
    )

    # Move request into running (simulating that _schedule_waiting already ran).
    sched.waiting.remove(req)
    sched.running[req.request_id] = req

    return sched, req, adapter


# ------------------------------------------------------------------ #
# Test 1: decode exception releases the pinned leaf (main contract)
# ------------------------------------------------------------------ #
def test_decode_exception_releases_pinned_leaf():
    """_recover_from_generation_error must release pinned leaves.

    step() does NOT re-raise non-stream-thread exceptions; it calls
    _recover_from_generation_error() and returns normally.
    """
    sched, req, adapter = _make_scheduler_with_running_pinned_request()

    # Pre-assert: leaf is pinned before the failing step.
    assert adapter.pinned_leaf(req.request_id) is not None

    # Install a fake batch_generator whose .next() raises a RuntimeError.
    fake_bg = MagicMock()
    fake_bg.next.side_effect = RuntimeError("simulated decode failure")
    sched.batch_generator = fake_bg

    # step() must return without raising (the Exception recovery branch catches).
    result = sched.step()

    # Post-assert: pin released.
    assert adapter.pinned_leaf(req.request_id) is None, (
        "_recover_from_generation_error must release the pinned leaf"
    )


# ------------------------------------------------------------------ #
# Test 2: after decode exception the leaf becomes evictable
# ------------------------------------------------------------------ #
def test_decode_exception_leaf_becomes_evictable():
    """After step() recovers from a decode error, the leaf is evictable."""
    sched, req, adapter = _make_scheduler_with_running_pinned_request(
        request_id="req-evict-decode-test"
    )

    leaf = adapter.pinned_leaf(req.request_id)
    assert leaf is not None
    assert not leaf.is_evictable, "pinned leaf must not be evictable before step()"

    fake_bg = MagicMock()
    fake_bg.next.side_effect = RuntimeError("simulated decode failure")
    sched.batch_generator = fake_bg

    sched.step()

    # The leaf's ref_count must have returned to 0 (it has no children in this
    # single-segment trie), making it evictable.
    assert leaf.is_evictable, (
        "after decode exception recovery the leaf must be evictable (ref_count==0)"
    )
