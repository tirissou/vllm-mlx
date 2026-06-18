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
        verify_model: Any = None,
    ):
        self._inner = inner
        self._policy = policy
        self._kv_group_size = kv_group_size
        # Optional model reference used by the VLLM_MLX_VERIFY_FETCH_KV diagnostic.
        # Holding the model here keeps fetch()'s signature stable while letting us
        # run a reference forward pass on demand.
        self._verify_model = verify_model
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

        if os.environ.get("VLLM_MLX_VERIFY_FETCH_KV") == "1":
            try:
                self._verify_fetched_kv(request, reconstructed, cached_tokens)
            except Exception as e:
                logger.warning("[verify_kv] failed: %s", e)

        return True

    def _verify_fetched_kv(self, request, reconstructed, cached_tokens: int) -> None:
        """Diff the just-reconstructed cache against a fresh prefill of the same prefix.

        Gated by VLLM_MLX_VERIFY_FETCH_KV=1. Expensive — runs an extra forward pass
        over `cached_tokens` tokens at every cache hit. Logs per-layer-class max/mean
        absolute diffs on K and V. If diffs are ~0, the segment/assemble/merge round
        trip is exonerated at this turn; any quality drift comes from elsewhere.
        """
        if self._verify_model is None:
            logger.warning("[verify_kv] no model registered; skipping verification")
            return
        if cached_tokens <= 0:
            return

        from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

        tokens = list(request.prompt_token_ids[:cached_tokens])
        tokens_arr = mx.array(tokens)

        fresh = make_prompt_cache(self._verify_model)
        out = self._verify_model(tokens_arr[None], cache=fresh)
        eval_args: list = [out]
        for layer in fresh:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, mx.array):
                eval_args.append(k)
            if isinstance(v, mx.array):
                eval_args.append(v)
        mx.eval(*eval_args)

        # Determinism control: run a SECOND identical one-shot prefill and diff
        # fresh-vs-fresh2. If non-zero, MLX itself has run-to-run nondeterminism
        # on this batch shape; if zero, the prefill function is deterministic at
        # batch=1 shape and any rec-vs-fresh diff must be coming from elsewhere
        # (chunked prefill kernels, batch-dim kernels, etc.). Gated by a separate
        # env var so it's opt-in (doubles the verify cost again).
        do_determinism_check = os.environ.get("VLLM_MLX_VERIFY_DETERMINISM") == "1"
        fresh2 = None
        if do_determinism_check:
            fresh2 = make_prompt_cache(self._verify_model)
            out2 = self._verify_model(tokens_arr[None], cache=fresh2)
            eval2: list = [out2]
            for layer in fresh2:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if isinstance(k, mx.array):
                    eval2.append(k)
                if isinstance(v, mx.array):
                    eval2.append(v)
            mx.eval(*eval2)
            # Diff fresh vs fresh2 with the same KVCache/RotatingKVCache logic
            # we use below, but inline since it's a self-contained sanity probe.
            self_max_k = 0.0
            self_max_v = 0.0
            for f1, f2 in zip(fresh, fresh2):
                k1 = getattr(f1, "keys", None)
                k2 = getattr(f2, "keys", None)
                v1 = getattr(f1, "values", None)
                v2 = getattr(f2, "values", None)
                if not (isinstance(k1, mx.array) and isinstance(k2, mx.array)):
                    continue
                n1 = int(getattr(f1, "offset", k1.shape[-2]))
                n2 = int(getattr(f2, "offset", k2.shape[-2]))
                n = min(n1, n2)
                if n <= 0:
                    continue
                dk = mx.abs(k1[..., :n, :].astype(mx.float32)
                            - k2[..., :n, :].astype(mx.float32))
                dv = mx.abs(v1[..., :n, :].astype(mx.float32)
                            - v2[..., :n, :].astype(mx.float32))
                mx.eval(dk, dv)
                self_max_k = max(self_max_k, float(dk.max().item()))
                self_max_v = max(self_max_v, float(dv.max().item()))
            del out2, eval2
            logger.warning(
                "[verify_kv:determinism] fresh-vs-fresh max_K=%.3e max_V=%.3e %s",
                self_max_k, self_max_v,
                "(DETERMINISTIC at batch=1)" if (self_max_k == 0.0 and self_max_v == 0.0)
                else "(NONDETERMINISTIC — diff source includes MLX itself)",
            )

        if len(fresh) != len(reconstructed):
            logger.warning(
                "[verify_kv] layer-count mismatch: fresh=%d reconstructed=%d (skipping)",
                len(fresh), len(reconstructed),
            )
            return

        # Per-class aggregation. Beyond abs/mean diffs we track:
        # - relative diff (= abs / max(|rec|, |fresh|) + eps): distinguishes bf16 noise
        #   on large-magnitude attention-sink K values from real round-trip bugs.
        # - worst layer index + position + |K|/|V| magnitudes at that position: lets us
        #   see whether the worst element sits on a known-outlier slot (BOS/sinks) or
        #   on an arbitrary token.
        from collections import defaultdict
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "n": 0, "n_tok": 0,
                "max_k": 0.0, "max_v": 0.0,
                "sum_mean_k": 0.0, "sum_mean_v": 0.0,
                "max_rel_k": 0.0, "max_rel_v": 0.0,
                "worst_k_layer": -1, "worst_k_pos": -1,
                "worst_k_rec_mag": 0.0, "worst_k_fresh_mag": 0.0,
                "worst_v_layer": -1, "worst_v_pos": -1,
                "worst_v_rec_mag": 0.0, "worst_v_fresh_mag": 0.0,
            }
        )

        for layer_idx, (rec_layer, fresh_layer) in enumerate(zip(reconstructed, fresh)):
            cname = type(fresh_layer).__name__
            try:
                if isinstance(fresh_layer, RotatingKVCache):
                    # Rotating layer: compare last `rec._idx` of fresh against the
                    # whole reconstructed buffer (which already holds the most-recent
                    # sliding-window-sized slice in temporal order).
                    n = int(rec_layer._idx)
                    if n <= 0:
                        continue
                    rec_k = rec_layer.keys[..., :n, :]
                    rec_v = rec_layer.values[..., :n, :]
                    fresh_k = fresh_layer.keys[..., -n:, :]
                    fresh_v = fresh_layer.values[..., -n:, :]
                elif isinstance(fresh_layer, KVCache):
                    # Concat layer: compare the first `offset` tokens of both buffers
                    # (both are stored in absolute order).
                    n = int(rec_layer.offset)
                    if n <= 0:
                        continue
                    rec_k = rec_layer.keys[..., :n, :]
                    rec_v = rec_layer.values[..., :n, :]
                    fresh_k = fresh_layer.keys[..., :n, :]
                    fresh_v = fresh_layer.values[..., :n, :]
                else:
                    continue
            except Exception as e:
                logger.warning("[verify_kv] layer %s skipped: %s", cname, e)
                continue

            rk = rec_k.astype(mx.float32)
            fk = fresh_k.astype(mx.float32)
            rv = rec_v.astype(mx.float32)
            fv = fresh_v.astype(mx.float32)

            dk = mx.abs(rk - fk)
            dv = mx.abs(rv - fv)
            denom_k = mx.maximum(mx.abs(rk), mx.abs(fk)) + 1e-6
            denom_v = mx.maximum(mx.abs(rv), mx.abs(fv)) + 1e-6
            rdk = dk / denom_k
            rdv = dv / denom_v

            # Per-position projections: max over all axes except the sequence axis.
            # Tensor shape is (1, H, n, D) — sequence axis is index -2.
            collapse = tuple(i for i in range(dk.ndim) if i != dk.ndim - 2)
            pos_dk = dk.max(axis=collapse)
            pos_dv = dv.max(axis=collapse)
            pos_rk = mx.abs(rk).max(axis=collapse)
            pos_fk = mx.abs(fk).max(axis=collapse)
            pos_rv = mx.abs(rv).max(axis=collapse)
            pos_fv = mx.abs(fv).max(axis=collapse)

            mx.eval(dk, dv, rdk, rdv, pos_dk, pos_dv, pos_rk, pos_fk, pos_rv, pos_fv)

            max_k = float(dk.max().item())
            max_v = float(dv.max().item())
            worst_pos_k = int(pos_dk.argmax().item())
            worst_pos_v = int(pos_dv.argmax().item())

            g = agg[cname]
            g["n"] += 1
            g["n_tok"] = n
            if max_k > g["max_k"]:
                g["max_k"] = max_k
                g["worst_k_layer"] = layer_idx
                g["worst_k_pos"] = worst_pos_k
                g["worst_k_rec_mag"] = float(pos_rk.flatten()[worst_pos_k].item())
                g["worst_k_fresh_mag"] = float(pos_fk.flatten()[worst_pos_k].item())
            if max_v > g["max_v"]:
                g["max_v"] = max_v
                g["worst_v_layer"] = layer_idx
                g["worst_v_pos"] = worst_pos_v
                g["worst_v_rec_mag"] = float(pos_rv.flatten()[worst_pos_v].item())
                g["worst_v_fresh_mag"] = float(pos_fv.flatten()[worst_pos_v].item())
            g["max_rel_k"] = max(g["max_rel_k"], float(rdk.max().item()))
            g["max_rel_v"] = max(g["max_rel_v"], float(rdv.max().item()))
            g["sum_mean_k"] += float(dk.mean().item())
            g["sum_mean_v"] += float(dv.mean().item())

        # Free the reference prefill ASAP — it doubled (or tripled, with the
        # determinism control) active memory for this call.
        del fresh, out, eval_args
        if fresh2 is not None:
            del fresh2
        mx.clear_cache()

        parts = []
        for cname, g in sorted(agg.items()):
            n = g["n"]
            parts.append(
                f"{cname} layers={n} tok={g['n_tok']} | "
                f"abs max_K={g['max_k']:.3e} max_V={g['max_v']:.3e} "
                f"mean_K={g['sum_mean_k']/n:.3e} mean_V={g['sum_mean_v']/n:.3e} | "
                f"rel max_K={g['max_rel_k']:.3e} max_V={g['max_rel_v']:.3e} | "
                f"worst_K L={g['worst_k_layer']} pos={g['worst_k_pos']} "
                f"|rec|={g['worst_k_rec_mag']:.2f} |fresh|={g['worst_k_fresh_mag']:.2f} | "
                f"worst_V L={g['worst_v_layer']} pos={g['worst_v_pos']} "
                f"|rec|={g['worst_v_rec_mag']:.2f} |fresh|={g['worst_v_fresh_mag']:.2f}"
            )
        rid = getattr(request, "request_id", "?")
        logger.warning(
            "[verify_kv] req=%s cached_tokens=%d | %s",
            rid[:12] if isinstance(rid, str) else rid,
            cached_tokens,
            " | ".join(parts) if parts else "(no comparable layers)",
        )

        if os.environ.get("VLLM_MLX_VERIFY_SCAN") == "1":
            try:
                self._scan_M_regimes(tokens_arr, cached_tokens)
            except Exception as e:
                logger.warning("[verify_kv:scan] failed: %s", e)

        if os.environ.get("VLLM_MLX_VERIFY_CHUNK_SCAN") == "1":
            try:
                self._scan_chunking(tokens_arr, cached_tokens)
            except Exception as e:
                logger.warning("[verify_kv:chunk_scan] failed: %s", e)

    def _scan_M_regimes(self, tokens_arr: mx.array, cached_tokens: int) -> None:
        """Scan matmul kernel-shape regimes by varying input length M.

        Takes the first 64 real tokens of the cached prefix and runs the model
        with input lengths M ∈ {64, 128, 256, 512, 1024, 2048, 4096}, padding
        with zeros. Diffs layer-0 K and V at [..., :64, :] pairwise across M
        values: two M values whose results are bit-identical share a matmul
        kernel regime; differing values straddle a regime boundary.

        Layer-0 K, V = W_{K,V} @ RMSNorm(embed[t]) are purely per-position, so
        the attention mask is irrelevant for this measurement (dummy pad tokens
        contaminate higher-layer K, V via attention, but not L=0).

        Gated by VLLM_MLX_VERIFY_SCAN=1.
        """
        if cached_tokens < 64:
            logger.warning(
                "[verify_kv:scan] cached_tokens=%d < 64, skipping", cached_tokens
            )
            return

        from mlx_lm.models.cache import make_prompt_cache

        n_real = 64
        candidates = [64, 128, 256, 512, 1024, 2048, 4096]
        scan_lengths = sorted({M for M in candidates if M >= n_real})
        real_tokens = tokens_arr[:n_real]

        scan_KV: list[tuple[int, mx.array, mx.array]] = []
        layer0_type: str | None = None
        for M in scan_lengths:
            pad = M - n_real
            if pad > 0:
                padded = mx.concatenate(
                    [real_tokens, mx.zeros((pad,), dtype=real_tokens.dtype)]
                )
            else:
                padded = real_tokens
            c = make_prompt_cache(self._verify_model)
            _out = self._verify_model(padded[None], cache=c)
            layer0 = c[0] if len(c) > 0 else None
            if layer0_type is None and layer0 is not None:
                layer0_type = type(layer0).__name__
            keys = getattr(layer0, "keys", None) if layer0 is not None else None
            values = getattr(layer0, "values", None) if layer0 is not None else None
            if isinstance(keys, mx.array) and isinstance(values, mx.array):
                k0 = mx.contiguous(keys[..., :n_real, :].astype(mx.float32))
                v0 = mx.contiguous(values[..., :n_real, :].astype(mx.float32))
                mx.eval(k0, v0)
                scan_KV.append((M, k0, v0))
            del c, _out
            mx.clear_cache()

        if len(scan_KV) < 2:
            logger.warning(
                "[verify_kv:scan] only %d valid M values collected, skipping",
                len(scan_KV),
            )
            return

        for kind_idx, kind in enumerate(("K", "V"), start=1):
            header = "        " + "  ".join(f"M={M:>5}" for M, *_ in scan_KV)
            rows = [header]
            for i in range(len(scan_KV)):
                Mi = scan_KV[i][0]
                Ai = scan_KV[i][kind_idx]
                cells = []
                for j in range(len(scan_KV)):
                    if i == j:
                        cells.append("    .   ")
                        continue
                    Aj = scan_KV[j][kind_idx]
                    d = float(mx.abs(Ai - Aj).max().item())
                    cells.append(f"{d:7.2e}")
                rows.append(f"M={Mi:>5} " + "  ".join(cells))
            logger.warning(
                "[verify_kv:scan] layer-0 %s[0..%d) (%s) pairwise diff (fp32):\n%s",
                kind, n_real, layer0_type or "?", "\n".join(rows),
            )
        del scan_KV
        mx.clear_cache()

    def _scan_chunking(self, tokens_arr: mx.array, cached_tokens: int) -> None:
        """Compare prefill K, V from different chunking schedules at same positions.

        Prefills the same physical token range under different chunking
        schedules and diffs the resulting K, V at matching positions across
        several probe layers.

        Layer 0 K, V depend only on input embedding — invariant to chunking
        unless the M-regime cliff hits W_{K,V}. Layer 1+ K, V depend on
        attention output, which sees different T_kv shapes under different
        schedules — diff here measures chunking-induced drift in production.

        N is set via VLLM_MLX_VERIFY_CHUNK_SCAN_N (default 1024). Lower N
        works on models with sliding-window max_size < 2048 (e.g. Gemma 4 MoE,
        max_size=1024). Schedules are auto-derived: 1xN, 2x(N/2), 4x(N/4),
        8x(N/8) — skipping any chunk size below 64.

        Gated by VLLM_MLX_VERIFY_CHUNK_SCAN=1.
        """
        from mlx_lm.models.cache import make_prompt_cache

        try:
            N = int(os.environ.get("VLLM_MLX_VERIFY_CHUNK_SCAN_N", "1024"))
        except (TypeError, ValueError):
            N = 1024
        if cached_tokens < N:
            logger.warning(
                "[verify_kv:chunk_scan] cached_tokens=%d < %d, skipping",
                cached_tokens, N,
            )
            return

        real_tokens = tokens_arr[:N]
        schedules: list[tuple[str, list[int]]] = []
        for divisor in (1, 2, 4, 8):
            chunk_size = N // divisor
            if chunk_size < 64 or N % divisor != 0:
                continue
            schedules.append((f"{divisor}x{chunk_size}", [chunk_size] * divisor))

        # Probe a small set of representative layers.
        sample_cache = make_prompt_cache(self._verify_model)
        n_layers = len(sample_cache)
        probe_layers = sorted({0, 1, max(1, n_layers // 2), n_layers - 1})
        layer_types: dict[int, str] = {L: type(sample_cache[L]).__name__ for L in probe_layers}
        del sample_cache
        mx.clear_cache()

        # results[i] = (schedule_name, {layer_idx: (K_fp32, V_fp32)})
        results: list[tuple[str, dict[int, tuple[mx.array, mx.array]]]] = []
        for name, chunks in schedules:
            c = make_prompt_cache(self._verify_model)
            cursor = 0
            for chunk_size in chunks:
                chunk = real_tokens[cursor : cursor + chunk_size]
                _ = self._verify_model(chunk[None], cache=c)
                cursor += chunk_size
            layer_dump: dict[int, tuple[mx.array, mx.array]] = {}
            for L in probe_layers:
                if L >= len(c):
                    continue
                layer = c[L]
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if isinstance(keys, mx.array) and isinstance(values, mx.array):
                    # Take only positions in [0, N). For RotatingKVCache the
                    # buffer may be smaller than N when max_size < N — skip
                    # those layers since cross-schedule comparison is undefined.
                    if keys.shape[-2] < N or values.shape[-2] < N:
                        continue
                    k = mx.contiguous(keys[..., :N, :].astype(mx.float32))
                    v = mx.contiguous(values[..., :N, :].astype(mx.float32))
                    mx.eval(k, v)
                    layer_dump[L] = (k, v)
            results.append((name, layer_dump))
            del c
            mx.clear_cache()

        # Emit a pairwise matrix per (layer, K/V).
        for L in probe_layers:
            lt = layer_types.get(L, "?")
            for kind_pos, kind in enumerate(("K", "V")):
                header = "             " + "  ".join(f"{n:>10}" for n, _ in results)
                rows = [header]
                any_missing = False
                for i, (ni, di) in enumerate(results):
                    if L not in di:
                        rows.append(f"{ni:>10} (missing — buffer < {N})")
                        any_missing = True
                        continue
                    cells = []
                    for j, (nj, dj) in enumerate(results):
                        if i == j:
                            cells.append("    .     ")
                            continue
                        if L not in dj:
                            cells.append("    ?     ")
                            continue
                        ai = di[L][kind_pos]
                        aj = dj[L][kind_pos]
                        d = float(mx.abs(ai - aj).max().item())
                        cells.append(f"{d:9.2e}")
                    rows.append(f"{ni:>10}    " + "  ".join(cells))
                if any_missing and all(L not in di for _, di in results):
                    continue
                logger.warning(
                    "[verify_kv:chunk_scan] L=%d (%s) %s @ [0..%d) pairwise diff (fp32):\n%s",
                    L, lt, kind, N, "\n".join(rows),
                )
        del results
        mx.clear_cache()

    def store(self, request, tokens: list[int] = None, cache: list = None) -> bool:
        from .turn_prefix_cache import Segment

        if (
            cache is None
            and isinstance(tokens, list)
            and (not tokens or not isinstance(tokens[0], int))
        ):
            cache = tokens
            tokens = None
        if cache is None:
            cache = []

        segments = self.messages_to_segments(request)
        if not segments or not getattr(request, "output_token_ids", None):
            return False

        cs = getattr(request, "_cache_state", None)
        path = cs.turn_path if cs is not None else []
        matched_depth = len(path)
        parent = path[-1] if path else self._inner.root
        new_segments = segments[matched_depth:]

        if new_segments:
            # Combine the last unmatched prompt segment with the generated output
            # into a single response node.
            response_tokens = list(segments[-1].token_ids) + list(
                request.output_token_ids
            )
        else:
            # Full prompt was already in the trie; record only the new output as
            # a child of the deepest matched node.
            response_tokens = list(request.output_token_ids)

        if cache and not isinstance(cache[0], dict):
            from .kv_cache import extract_layer_state

            cache = [
                d for layer in cache if (d := extract_layer_state(layer)) is not None
            ]

        if cache:
            prev_end = path[-1].n_tokens if path else 0
            cache = slice_kv_to_delta(cache, prev_end)
            kv_sparse, rec_sparse = segment(
                cache, policy=self._policy, group_size=self._kv_group_size
            )
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
            _log_segment_breakdown(
                f"store rid={getattr(request, 'request_id', '?')} "
                f"tok_seg={len(response_tokens)} prev_end={prev_end}",
                kv_layers,
                rec_layers,
            )
        else:
            kv_layers, rec_layers = [], []

        # Snapshot current pinned leaf BEFORE any mutation so we can roll back
        # cleanly on insert failure.
        old_leaf = self._pinned_leaves.get(request.request_id)

        # Insert first; only touch ref_counts / _pinned_leaves after success so
        # that an exception leaves the previous leaf pin intact.
        new_leaf = self._inner.insert(
            parent,
            Segment(role="conversation", token_ids=response_tokens),
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
        )

        # Success path: advance the active leaf — unpin the old, pin the new.
        if old_leaf is not None:
            self._inner.release([old_leaf])
        with self._inner._lock:
            new_leaf.ref_count += 1
        self._pinned_leaves[request.request_id] = new_leaf
        if cs is not None:
            cs.turn_path = self._inner._inorder_path(new_leaf)
        return True

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
