# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests for the deepened CacheManager seam ("Active Leaf" model).

These tests assert the externally-visible contract that

  - ``fetch`` pins only the leaf of the matched path,
  - ``release`` unpins that leaf and clears request cache state,
  - ``store`` performs "unpin old leaf -> insert -> pin new leaf",
  - errors during ``store`` roll back to the pre-store leaf pin,
  - ``on_prefill_checkpoint`` advances the active leaf the same way ``store`` does.

The first five tests (fetch/release/store lifecycle) pass on the current
implementation.  The four ``test_checkpoint_*`` tests are intentionally red
until Task 2 (on_prefill_checkpoint active-leaf advancement) is implemented.
"""

from __future__ import annotations

import pytest

from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.turn_prefix_cache import (
    Segment,
    TurnPrefixCache,
    TurnPrefixCacheConfig,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────


def _make_trie() -> TurnPrefixCache:
    """Trie where every node is a permanent checkpoint (stride=0) and stored
    in bf16. Keeps configuration deterministic and removes eviction races."""
    return TurnPrefixCache(
        TurnPrefixCacheConfig(checkpoint_stride=0, kv_dtype="bf16")
    )


def _make_manager(trie: TurnPrefixCache) -> TurnCacheManager:
    return TurnCacheManager(trie, policy=None, kv_group_size=64)


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    turn_boundaries: list[int],
) -> Request:
    """Build a Request with the fields the cache layer reads from."""
    req = Request(
        request_id=request_id,
        prompt=" ".join(str(t) for t in prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=8),
    )
    req.prompt_token_ids = list(prompt_token_ids)
    req.num_prompt_tokens = len(prompt_token_ids)
    req._turn_boundaries = list(turn_boundaries)
    req._cache_state = RequestCacheState()
    return req


def _stub_assemble_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass assemble / validate so hit-path tests do not depend on real
    KV reconstruction. We are testing lifecycle/pinning, not cache contents.
    """
    monkeypatch.setattr(
        "vllm_mlx.prefix_cache_adapters.assemble",
        lambda kv, rec, *a, **k: [object()],
    )
    monkeypatch.setattr(TurnCacheManager, "validate", lambda self, cache: True)


def _insert_two_segment_path(
    trie: TurnPrefixCache, sys_tokens: list[int], user_tokens: list[int]
):
    """Insert a [system, user] path with non-empty kv_data so that
    find_checkpoint_ancestor will return the leaf rather than None.

    Returns (sys_node, user_leaf).
    """
    # A list with no items is truthy-ish? No — an empty list is falsy. We need
    # non-empty kv_data so find_checkpoint_ancestor accepts the node.
    # The kv_data items only have to look like KVLayerSegments for the
    # collect_path_data walk; since we stub assemble, the actual contents
    # do not matter.
    from vllm_mlx.cache_types import KVConcatSegment
    import mlx.core as mx

    def _dummy_layer(idx: int) -> KVConcatSegment:
        return KVConcatSegment(
            keys=mx.zeros((1, 1, 1, 1)),
            values=mx.zeros((1, 1, 1, 1)),
            layer_index=idx,
            n_tokens=1,
            bits=None,
            class_name="KVCache",
        )

    sys_node = trie.insert(
        trie.root,
        Segment(role="system", token_ids=sys_tokens),
        kv_data=[_dummy_layer(0)],
        is_system_prompt=True,
    )
    user_leaf = trie.insert(
        sys_node,
        Segment(role="user", token_ids=user_tokens),
        kv_data=[_dummy_layer(0)],
    )
    return sys_node, user_leaf


# ── Tests ─────────────────────────────────────────────────────────────────


def test_cache_fetch_hit_pins_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a hit, only the leaf of the matched path is pinned (Active Leaf).

    This intentionally fails today: ``TurnPrefixCache.match`` pins every node
    in the path, and ``TurnCacheManager`` does not yet populate
    ``_pinned_leaves``.
    """
    _stub_assemble_and_validate(monkeypatch)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    sys_node, user_leaf = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-hit",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],  # B_sys = 10 → [system, user]
    )

    assert manager.fetch(req) is True
    assert req._cache_state.hit_type == "hit"

    # Active Leaf invariant: only the leaf is pinned.
    assert not user_leaf.is_evictable, (
        f"leaf should be pinned (not evictable), got ref_count={user_leaf.ref_count}"
    )
    assert sys_node.ref_count == 0, (
        "non-leaf ancestors must not be pinned under the Active Leaf model; "
        f"got sys_node.ref_count == {sys_node.ref_count}"
    )

    # The manager must remember which leaf belongs to this request.
    assert manager.pinned_leaf(req.request_id) is not None
    assert manager.pinned_leaf(req.request_id) is user_leaf


def test_cache_fetch_miss_initialises_state() -> None:
    """A miss leaves the trie unpinned and populates miss state on the request."""
    trie = _make_trie()
    manager = _make_manager(trie)

    prompt = list(range(20))
    req = _make_request(
        "req-miss",
        prompt_token_ids=prompt,
        turn_boundaries=[10],  # [system, user] segments, but trie is empty
    )

    assert manager.fetch(req) is False
    assert req._cache_state.hit_type == "miss"
    assert req._cache_state.cached_tokens == 0
    assert req._cache_state.remaining_tokens == req.prompt_token_ids

    # No leaf was pinned for this request.
    assert manager.pinned_leaf(req.request_id) is None


def test_cache_release_unpins_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    """``release`` undoes the leaf pin from ``fetch`` and clears state."""
    _stub_assemble_and_validate(monkeypatch)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, user_leaf = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-release",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],
    )

    assert manager.fetch(req) is True
    # Sanity: hit pinned the leaf.
    assert not user_leaf.is_evictable
    assert manager.pinned_leaf(req.request_id) is user_leaf

    manager.release(req)

    assert user_leaf.is_evictable, "release must unpin the leaf"
    assert manager.pinned_leaf(req.request_id) is None, (
        "release must clear the pin (pinned_leaf returns None)"
    )
    assert req._cache_state.turn_path == [], (
        "release must clear request._cache_state.turn_path"
    )


def test_cache_store_advancement_unpins_old_pins_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store`` advances the active leaf: old leaf is unpinned, new leaf pinned."""
    _stub_assemble_and_validate(monkeypatch)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, leaf_a = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-store",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],
    )

    assert manager.fetch(req) is True
    assert not leaf_a.is_evictable
    assert manager.pinned_leaf(req.request_id) is leaf_a

    # Simulate the model having produced output tokens; ``store`` requires
    # output_token_ids to be non-empty.
    req.output_token_ids = [42, 43]

    # store with no cache layers is sufficient for lifecycle testing; the
    # important effect is that a new node is inserted below leaf_a.
    manager.store(req, tokens=req.output_token_ids, cache=[])

    # The newly inserted child of leaf_a is the new leaf.
    new_children = list(leaf_a.children.values())
    assert len(new_children) == 1, (
        "store should insert exactly one new child under the previous leaf"
    )
    leaf_b = new_children[0]

    assert leaf_a.ref_count == 0, "old leaf must be unpinned by store"
    assert not leaf_b.is_evictable, "new leaf must be pinned by store"
    assert manager.pinned_leaf(req.request_id) is leaf_b
    assert req._cache_state.turn_path[-1] is leaf_b, (
        "request.turn_path must end at the new pinned leaf"
    )


def test_cache_store_rollback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If insertion raises, ``store`` rolls back so the old leaf stays pinned."""
    _stub_assemble_and_validate(monkeypatch)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, leaf_a = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-rollback",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],
    )

    assert manager.fetch(req) is True
    assert not leaf_a.is_evictable
    assert manager.pinned_leaf(req.request_id) is leaf_a

    req.output_token_ids = [42, 43]

    class _BoomError(RuntimeError):
        pass

    def _raise(*args, **kwargs):
        raise _BoomError("simulated insert failure")

    monkeypatch.setattr(TurnPrefixCache, "_insert_node", _raise)

    with pytest.raises(_BoomError):
        manager.store(req, tokens=req.output_token_ids, cache=[])

    # Rollback invariant: leaf_a is still the pinned leaf for this request.
    assert not leaf_a.is_evictable, (
        "old leaf must remain pinned (not evictable) after a failed store"
    )
    assert manager.pinned_leaf(req.request_id) is leaf_a, (
        "pinned_leaf must still map req -> leaf_a after a failed store"
    )


def test_checkpoint_from_miss_pins_new_leaf() -> None:
    """Miss path: the first on_prefill_checkpoint must pin the inserted node
    even though _pinned_leaves had no prior entry.

    Today this fails: on_prefill_checkpoint inserts the node but never touches
    _pinned_leaves, so the new leaf has ref_count == 0 and is evictable.
    """
    trie = _make_trie()
    manager = _make_manager(trie)

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    prompt = sys_tokens + user_tokens
    req = _make_request(
        "req-ckpt-miss",
        prompt_token_ids=prompt,
        turn_boundaries=[len(sys_tokens), len(prompt)],
    )

    # Miss populates cache_state.turn_path == [] and leaves _pinned_leaves empty.
    assert manager.fetch(req) is False
    assert manager.pinned_leaf(req.request_id) is None

    # Drive the first checkpoint at the system-boundary (abs_idx=0).
    manager.on_prefill_checkpoint(req, total_tokens_prefilled=len(sys_tokens),
                                  extracted_cache=[])

    turn_path = req._cache_state.turn_path
    assert len(turn_path) == 1, "checkpoint should append exactly one node"
    new_leaf = turn_path[-1]

    assert not new_leaf.is_evictable, (
        f"new checkpoint leaf must be pinned (not evictable), got ref_count={new_leaf.ref_count}"
    )
    assert manager.pinned_leaf(req.request_id) is new_leaf


def test_checkpoint_advances_active_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hit path: fetch pins leaf A; the next checkpoint inserts B as child of A
    and must unpin A while pinning B.

    Today this fails: A stays pinned (interior-pin is wasted) and B has
    ref_count == 0 (evictable while prefill is still running).
    """
    _stub_assemble_and_validate(monkeypatch)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, leaf_a = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    # Prompt includes a NEW third segment past the matched [system, user] path.
    # Boundaries are [10, 15, 20] for prompt of length 20, so messages_to_segments
    # produces 3 segments (system, conversation, user). The matched path after
    # fetch has depth 2. The early-return guard inside on_prefill_checkpoint is
    # `if len(turn_path) > abs_idx: return`, so we must call it at the boundary
    # whose abs_idx == 2 — i.e. total_tokens_prefilled == len(prompt).
    extra_tokens = list(range(15, 20))
    prompt = sys_tokens + user_tokens + extra_tokens
    req = _make_request(
        "req-ckpt-hit",
        prompt_token_ids=prompt,
        turn_boundaries=[
            len(sys_tokens),
            len(sys_tokens) + len(user_tokens),
            len(prompt),
        ],
    )

    assert manager.fetch(req) is True
    assert not leaf_a.is_evictable
    assert manager.pinned_leaf(req.request_id) is leaf_a

    # Drive a checkpoint at abs_idx=2 (past the matched depth of 2).
    manager.on_prefill_checkpoint(
        req,
        total_tokens_prefilled=len(prompt),
        extracted_cache=[],
    )

    new_children = list(leaf_a.children.values())
    assert len(new_children) == 1, "checkpoint must insert exactly one child under leaf_a"
    leaf_b = new_children[0]

    assert leaf_a.ref_count == 0, "checkpoint must unpin the previous active leaf"
    assert not leaf_b.is_evictable, "checkpoint must pin the newly inserted leaf"
    assert manager.pinned_leaf(req.request_id) is leaf_b
    assert req._cache_state.turn_path[-1] is leaf_b


def test_consecutive_checkpoints_advance_leaf() -> None:
    """Two checkpoints in a row from a miss: each advances the pin to the
    newest leaf; the intermediate node ends at ref_count == 0."""
    trie = _make_trie()
    manager = _make_manager(trie)

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    prompt = sys_tokens + user_tokens
    req = _make_request(
        "req-ckpt-chain",
        prompt_token_ids=prompt,
        turn_boundaries=[len(sys_tokens), len(prompt)],
    )

    assert manager.fetch(req) is False

    manager.on_prefill_checkpoint(req, total_tokens_prefilled=len(sys_tokens),
                                  extracted_cache=[])
    node_sys = req._cache_state.turn_path[-1]
    # These two assertions are also part of the expected red state today
    # (first checkpoint does not pin): both will pass once Task 2 is done.
    assert not node_sys.is_evictable, "first checkpoint must pin the inserted node"
    assert manager.pinned_leaf(req.request_id) is node_sys

    manager.on_prefill_checkpoint(req, total_tokens_prefilled=len(prompt),
                                  extracted_cache=[])
    node_user = req._cache_state.turn_path[-1]

    assert node_user is not node_sys, "second checkpoint must insert a new node"
    assert node_sys.ref_count == 0, "previous checkpoint leaf must be unpinned"
    assert not node_user.is_evictable, "newest checkpoint leaf must be pinned"
    assert manager.pinned_leaf(req.request_id) is node_user


def test_release_after_checkpoint_unpins_latest_leaf() -> None:
    """Abort during prefill: fetch (miss) -> checkpoint -> release should
    unpin the checkpoint leaf, not the (non-existent) original leaf, and
    leave no dangling _pinned_leaves entry."""
    trie = _make_trie()
    manager = _make_manager(trie)

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    prompt = sys_tokens + user_tokens
    req = _make_request(
        "req-ckpt-abort",
        prompt_token_ids=prompt,
        turn_boundaries=[len(sys_tokens), len(prompt)],
    )

    assert manager.fetch(req) is False
    manager.on_prefill_checkpoint(req, total_tokens_prefilled=len(sys_tokens),
                                  extracted_cache=[])
    leaf = req._cache_state.turn_path[-1]
    # This assertion is also red today (checkpoint never pins): it passes once Task 2 is done.
    assert not leaf.is_evictable, "checkpoint must have pinned the leaf before release"

    manager.release(req)

    assert leaf.is_evictable, "release must unpin the checkpoint leaf"
    assert manager.pinned_leaf(req.request_id) is None
    assert req._cache_state.turn_path == []


def test_fetch_failure_no_checkpoint_ancestor_releases_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When find_checkpoint_ancestor returns None after a match, fetch must
    release the leaf pin, drop _pinned_leaves[req], and set miss state."""
    # _assemble and validate are never reached on this code path;
    # find_checkpoint_ancestor returning None short-circuits before them.
    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, user_leaf = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-no-ancestor",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],
    )

    # Force the post-match failure mode: no usable checkpoint ancestor.
    monkeypatch.setattr(TurnPrefixCache, "find_checkpoint_ancestor",
                        lambda self, path: None)

    assert manager.fetch(req) is False
    assert req._cache_state.hit_type == "miss"
    assert user_leaf.is_evictable, "leaf pin must be released on this failure"
    assert manager.pinned_leaf(req.request_id) is None


def test_fetch_failure_validate_returns_false_releases_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When validate(reconstructed) returns False, fetch must release the
    leaf pin, drop _pinned_leaves[req], and set miss state."""
    # Stub assemble (so we don't depend on real reconstruction) but force
    # validate False instead of True.
    monkeypatch.setattr(
        "vllm_mlx.prefix_cache_adapters.assemble",
        lambda kv, rec, *a, **k: [object()],
    )
    monkeypatch.setattr(TurnCacheManager, "validate", lambda self, cache: False)

    trie = _make_trie()
    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    _sys_node, user_leaf = _insert_two_segment_path(trie, sys_tokens, user_tokens)

    manager = _make_manager(trie)
    req = _make_request(
        "req-bad-validate",
        prompt_token_ids=sys_tokens + user_tokens,
        turn_boundaries=[len(sys_tokens)],
    )

    assert manager.fetch(req) is False
    assert req._cache_state.hit_type == "miss"
    assert user_leaf.is_evictable, "leaf pin must be released on validate failure"
    assert manager.pinned_leaf(req.request_id) is None
