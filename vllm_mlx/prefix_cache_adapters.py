# SPDX-License-Identifier: Apache-2.0
"""Adapters that bridge concrete prefix cache backends to the CacheManager protocol."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import mlx.core as mx

from vllm_mlx.request import Request
from vllm_mlx.turn_prefix_cache import TurnPrefixCache

if TYPE_CHECKING:
    from vllm_mlx.turn_prefix_cache import TurnNode

from .kv_cache import CacheIndexMap, _BATCH_KV_TYPES, validate_cache, extract_cache_states
from .cache_types import KVQuantPolicy
from vllm_mlx.cache_translator import segment, assemble, slice_kv_to_delta

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)



def _kv_segment_bytes(seg) -> int:
    """Sum on-device bytes held by one KVLayerSegment (keys + values)."""
    from .kv_cache import QuantizedArray

    total = 0
    for arr in (seg.keys, seg.values):
        if isinstance(arr, QuantizedArray):
            total += int(arr.packed.nbytes) + int(arr.scales.nbytes) + int(arr.biases.nbytes)
        else:
            total += int(arr.nbytes)
    return total


def _log_segment_breakdown(label: str, kv_layers: list, rec_layers: list) -> None:
    """Emit one log line per checkpoint with byte breakdown by layer class.

    Gated by VLLM_MLX_MEMPROBE_SEGMENTS=1. Cheap when off (env check, early return).
    """
    if os.environ.get("VLLM_MLX_MEMPROBE_SEGMENTS") != "1":
        return
    if not kv_layers and not rec_layers:
        logger.warning("[segprobe:%s] no layers", label)
        return

    # Aggregate by (class_name, merge_strategy)
    from .cache_types import KVRotatingSegment as _KVRotatingSegment

    groups: dict[tuple[str, str], dict] = {}
    n_tokens_seen: dict[tuple[str, str], list[int]] = {}
    for seg in kv_layers:
        cname = getattr(seg, "class_name", "?")
        strategy = "last" if isinstance(seg, _KVRotatingSegment) else "concatenate"
        key = (cname, strategy)
        g = groups.setdefault(key, {"count": 0, "bytes": 0, "max": 0})
        b = _kv_segment_bytes(seg)
        g["count"] += 1
        g["bytes"] += b
        g["max"] = max(g["max"], b)
        n_tokens_seen.setdefault(key, []).append(int(getattr(seg, "n_tokens", -1)))

    parts = []
    grand_total = 0
    for (cname, strategy), g in sorted(groups.items()):
        toks = n_tokens_seen[(cname, strategy)]
        tok_min, tok_max = min(toks), max(toks)
        mean_mb = g["bytes"] / g["count"] / 1e6
        max_mb = g["max"] / 1e6
        total_mb = g["bytes"] / 1e6
        grand_total += g["bytes"]
        tok_str = f"{tok_min}" if tok_min == tok_max else f"{tok_min}..{tok_max}"
        parts.append(
            f"{cname}[{strategy}] n={g['count']} tok={tok_str} "
            f"sum={total_mb:.1f}MB mean={mean_mb:.2f}MB max={max_mb:.2f}MB"
        )

    n_rec = len([r for r in rec_layers if r is not None])
    logger.warning(
        "[segprobe:%s] total=%.1fMB rec_layers=%d | %s",
        label,
        grand_total / 1e6,
        n_rec,
        " | ".join(parts),
    )


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
    def fetch(self, request) -> bool:
        """Look up a cached prefix for this request.

        On a hit, populates request._cache_state.turn_path and returns True.
        On a miss, populates request._cache_state for miss and returns False.
        """
        ...

    @abstractmethod
    def store(self, request, tokens: list[int], cache: list) -> bool:
        """Store the completed request's cache keyed on the full token sequence."""
        ...

    # ── Default no-ops (override as needed) ──────────────────────────────────

    def release(self, request) -> None:
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

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        pass

    def validate(self, cache: list) -> bool:
        """Validate cache state. Returns True if valid and usable."""
        return True

    def extract_cache(self, raw_cache: list) -> list | None:
        """Extract cache state from raw cache objects.

        Returns list of layer state dicts, or None on failure.
        Called during cleanup to prepare cache for storage.
        """
        return None

    def save(self, cache_dir: str) -> bool:
        """Persist cache to disk. Returns True on success."""
        return False

    def load(self, cache_dir: str) -> int:
        """Load cache from disk. Returns entries loaded."""
        return 0

    def close(self) -> None:
        """Cleanup resources (SSD threads, file handles, etc.)."""
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


class TurnCacheManager(CacheManager):
    """Adapts TurnPrefixCache to the CacheManager protocol."""

    def __init__(
        self,
        inner: TurnPrefixCache,
        policy: KVQuantPolicy | None = None,
        kv_group_size: int = 64,
    ):
        self._inner = inner
        self._policy = policy
        self._kv_group_size = kv_group_size
        # request_id -> currently pinned leaf node (Active Leaf invariant).
        # Populated by fetch() on hit, advanced by store(), cleared by release().
        self._pinned_leaves: dict[str, "TurnNode"] = {}

    def pinned_leaf(self, request_id: str) -> "TurnNode | None":
        """Return the currently pinned leaf for a request, or None.

        Supported observer for the Active Leaf pinning invariant
        (CONTEXT.md). Tests and diagnostics use this; the underlying
        ``_pinned_leaves`` dict remains private.
        """
        return self._pinned_leaves.get(request_id)

    def boundaries(self, request) -> list[int]:
        cs = request._cache_state
        cached = cs.cached_tokens if cs is not None else 0
        turn_bds = getattr(request, "_turn_boundaries", None) or []
        return sorted(b - cached for b in turn_bds if b > cached)

    @staticmethod
    def messages_to_segments(request) -> list:
        """Split request token sequence into per-message Segment objects."""
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if not _turn_boundaries:
            return []
        B_sys = _turn_boundaries[0]
        if B_sys <= 0:
            return []
        segments: list = [Segment(role="system", token_ids=full_tokens[:B_sys])]
        prev = B_sys
        for B_k in _turn_boundaries[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(
                    Segment(role="conversation", token_ids=full_tokens[prev:B_k])
                )
                prev = B_k
        if prev < len(full_tokens):
            segments.append(Segment(role="user", token_ids=full_tokens[prev:]))
        return segments

    def _set_miss_state(self, request) -> None:
        """Populate request._cache_state with miss values."""
        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.hit_type = "miss"
            cs.cached_tokens = 0
            cs.turn_path = []
            cs.remaining_tokens = request.prompt_token_ids
            cs.prefill_boundaries = self.boundaries(request)

    def fetch(self, request) -> bool:
        segments = self.messages_to_segments(request)
        if not segments:
            self._set_miss_state(request)
            return False

        path, _ = self._inner.match(segments)
        if not path:
            self._set_miss_state(request)
            return False

        # Active Leaf invariant: match() incremented ref_count on every node
        # in `path`. Release the ancestors so only the leaf retains its +1.
        if len(path) > 1:
            self._inner.release(path[:-1])

        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.turn_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            # Release the remaining leaf pin; this is a miss after all.
            self._inner.release([path[-1]])
            self._pinned_leaves.pop(request.request_id, None)
            self._set_miss_state(request)
            return False

        _probe_req_id = getattr(request, "request_id", None) or getattr(request, "uid", "?")
        _probe_pre_active = mx.get_active_memory()
        kv_data, rec_data = self._inner.collect_path_data(ancestor)
        reconstructed = assemble(kv_data, rec_data, self._kv_group_size)
        _probe_n_tokens = sum(l.n_tokens for l in (kv_data or []) if l is not None)
        del kv_data, rec_data
        # Materialize the KVCache arrays now so the lazy computation graph
        # is freed before decode starts.
        arrays_to_eval = []
        for _layer in reconstructed or []:
            for attr in ("keys", "values"):
                v = getattr(_layer, attr, None)
                if v is None or callable(v):
                    continue
                if isinstance(v, mx.array):
                    arrays_to_eval.append(v)
                else:
                    for comp in (v if isinstance(v, (list, tuple)) else []):
                        if isinstance(comp, mx.array):
                            arrays_to_eval.append(comp)
        if arrays_to_eval:
            mx.eval(*arrays_to_eval)
        _probe_post_active = mx.get_active_memory()
        logging.warning(
            "[memprobe:fetch] req=%s n_tokens=%d pre_assemble=%.2fGB "
            "post_assemble=%.2fGB delta=%.2fGB peak=%.2fGB",
            _probe_req_id,
            _probe_n_tokens,
            _probe_pre_active / 1e9,
            _probe_post_active / 1e9,
            (_probe_post_active - _probe_pre_active) / 1e9,
            mx.get_peak_memory() / 1e9,
        )
        if not self.validate(reconstructed):
            # Release the remaining leaf pin; this is a miss after all.
            self._inner.release([path[-1]])
            self._pinned_leaves.pop(request.request_id, None)
            self._set_miss_state(request)
            return False
        cached_tokens = ancestor.n_tokens
        if cs is not None:
            cs.hit_type = "hit"
            cs.cache = reconstructed
            cs.cached_tokens = cached_tokens
            cs.remaining_tokens = list(request.prompt_token_ids[cached_tokens:])
            cs.prefill_boundaries = self.boundaries(request)
        # Record the pinned leaf so release() can find it.
        self._pinned_leaves[request.request_id] = path[-1]
        return True

    def store(self, request, tokens: list[int] = None, cache: list = None) -> bool:
        """No-op: decoded K,V never enter the trie.

        Cache promotion now happens exclusively through on_prefill_checkpoint()
        at turn boundaries during prefill (in canonical kernel regime, thanks
        to CanonicalPrefillBatchGenerator). Decoded K,V are computed at M=1 —
        the worst possible kernel regime — and would pollute the cache; this
        method returning False keeps them out.

        The signature and return type match the previous behavior's cache-miss
        path, so existing callers handle False without change.
        """
        return False

    def release(self, request) -> None:
        leaf = self._pinned_leaves.pop(request.request_id, None)
        if leaf is not None:
            # Go through the inner trie's release() so the node is re-added to
            # the eviction heap if it just became evictable.
            self._inner.release([leaf])
        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.turn_path = []

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if total_tokens_prefilled not in _turn_boundaries:
            return

        try:
            abs_idx = _turn_boundaries.index(total_tokens_prefilled)
        except ValueError:
            return

        segments = self.messages_to_segments(request)
        if abs_idx >= len(segments):
            return

        cs = getattr(request, "_cache_state", None)
        turn_path = cs.turn_path if cs is not None else []
        if len(turn_path) > abs_idx:
            return

        parent = turn_path[-1] if turn_path else self._inner.root
        turn_segment = segments[abs_idx]
        is_sys = turn_segment.role == "system" and abs_idx == 0

        if extracted_cache:
            if not isinstance(extracted_cache[0], dict):
                from .kv_cache import extract_layer_state

                extracted_cache = [
                    d
                    for layer in extracted_cache
                    if (d := extract_layer_state(layer)) is not None
                ]
            prev_end = _turn_boundaries[abs_idx - 1] if abs_idx > 0 else 0
            extracted_cache = slice_kv_to_delta(extracted_cache, prev_end)
            kv_sparse, rec_sparse = segment(
                extracted_cache, policy=self._policy, group_size=self._kv_group_size
            )
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
            _log_segment_breakdown(
                f"checkpoint rid={getattr(request, 'request_id', '?')} "
                f"abs_idx={abs_idx} tok_seg={len(turn_segment.token_ids)} "
                f"prev_end={prev_end} total={total_tokens_prefilled}",
                kv_layers,
                rec_layers,
            )
        else:
            kv_layers, rec_layers = [], []

        # Snapshot the previous pinned leaf BEFORE any mutation so that an
        # exception from insert() leaves the existing pin intact (implicit
        # rollback — same shape as store()).
        old_leaf = self._pinned_leaves.get(request.request_id)

        new_node = self._inner.insert(
            parent,
            turn_segment,
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )

        # Insert succeeded — advance the active leaf: release the old leaf
        # (if any) and pin the new one.
        if old_leaf is not None:
            self._inner.release([old_leaf])
        with self._inner._lock:
            new_node.ref_count += 1
        self._pinned_leaves[request.request_id] = new_node
        # Append rather than rebuild the full path: checkpoint always inserts
        # depth abs_idx in order, so the path is already correct up to abs_idx-1.
        if cs is not None:
            cs.turn_path.append(new_node)

    # ── Updated: save() / load() with error handling ──────────────────────────

    def save(self, cache_dir: str) -> bool:
        try:
            return self._inner.save(cache_dir)
        except Exception:
            return False

    def load(self, cache_dir: str) -> int:
        try:
            self._inner.load(cache_dir)
            return 0
        except Exception:
            return 0

    # ── New: validate, extract_cache, close ──────────────────────────────────

    def validate(self, cache: list) -> bool:
        return validate_cache(cache)

    def extract_cache(self, raw_cache: list) -> list | None:
        return extract_cache_states(raw_cache)

    def close(self) -> None:
        pass
