# SPDX-License-Identifier: Apache-2.0
"""Adapters that bridge concrete prefix cache backends to the CacheManager protocol."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx

from vllm_mlx.request import Request
from vllm_mlx.turn_prefix_cache import TurnPrefixCache

from .kv_cache import CacheHit, CacheIndexMap, _BATCH_KV_TYPES
from .cache_types import StaticKVData, StaticRecurrentData
from .cache_translator import CacheTranslator


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
        """Store the completed request's cache keyed on the full token sequence."""
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

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
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

    def __init__(self, inner: TurnPrefixCache):
        self._inner = inner

    def boundaries(self, request) -> list[int]:
        cs = request._cache_state
        cached = cs.cached_tokens if cs is not None else 0
        turn_bds = getattr(request, '_turn_boundaries', None) or []
        return sorted(b - cached for b in turn_bds if b > cached)

    @staticmethod
    def messages_to_segments(request) -> list:
        """Split request token sequence into per-message Segment objects."""
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []
        _turn_boundaries = getattr(request, '_turn_boundaries', None) or []
        if not _turn_boundaries:
            return []
        B_sys = _turn_boundaries[0]
        if B_sys <= 0:
            return []
        segments: list = [Segment(role='system', token_ids=full_tokens[:B_sys])]
        prev = B_sys
        for B_k in _turn_boundaries[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(Segment(role='conversation', token_ids=full_tokens[prev:B_k]))
                prev = B_k
        if prev < len(full_tokens):
            segments.append(Segment(role='user', token_ids=full_tokens[prev:]))
        return segments

    @staticmethod
    def _segment(
        live_states: list[dict[str, Any]]
    ) -> tuple[list[StaticKVData | None], list[StaticRecurrentData | None]]:
        """Transform live cache states into normalized static format.

        Returns two sparse lists parallel to live_states: kv_data_list and rec_data_list.
        Exactly one of kv_data_list[i] / rec_data_list[i] is non-None for each i.
        """
        kv_data_list: list[StaticKVData | None] = [None] * len(live_states)
        rec_data_list: list[StaticRecurrentData | None] = [None] * len(live_states)

        for i, state_dict in enumerate(live_states):
            class_name = state_dict['class_name']
            state = state_dict['state']
            meta = state_dict.get('meta_state', ())

            if class_name == 'RotatingKVCache':
                try:
                    keep, max_size, offset, _ = map(int, meta)
                except (TypeError, ValueError):
                    max_size = state[0].shape[2]
                    offset = max_size
                    keep = 0

                raw_keys, raw_values = state
                lin_keys = CacheTranslator.linearize(raw_keys, offset, max_size)
                lin_values = CacheTranslator.linearize(raw_values, offset, max_size)
                q_arrays, q_scales = CacheTranslator.quantize_kv([lin_keys, lin_values])

                kv_data_list[i] = StaticKVData(
                    arrays=q_arrays,
                    metadata={
                        'class_name': 'RotatingKVCache',
                        'layer_index': i,
                        'merge_strategy': 'last',
                        'max_size': max_size,
                        'keep': keep,
                        'offset': offset,
                        'scales': q_scales,
                    },
                )

            elif 'KVCache' in class_name:
                try:
                    actual_end = int(meta[0]) if meta else state[0].shape[2]
                except (TypeError, ValueError, IndexError):
                    actual_end = state[0].shape[2]

                sliced = [arr[:, :, :actual_end, :] for arr in state[:2]]
                q_arrays, q_scales = CacheTranslator.quantize_kv(sliced)

                kv_data_list[i] = StaticKVData(
                    arrays=q_arrays,
                    metadata={
                        'class_name': class_name,
                        'layer_index': i,
                        'merge_strategy': 'concatenate',
                        'actual_end': actual_end,
                        'scales': q_scales,
                    },
                )

            else:
                rec_data_list[i] = StaticRecurrentData(
                    arrays=state,
                    metadata={'class_name': class_name, 'layer_index': i},
                )

        return kv_data_list, rec_data_list

    @staticmethod
    def _assemble(
        kv_data: list[StaticKVData],
        recurrent_data: list[StaticRecurrentData],
    ) -> list[dict[str, Any]]:
        """Reconstruct live cache states from compact static data lists.

        Input lists are non-sparse (no None entries). Output is ordered by layer_index.
        """
        all_states: dict[int, dict[str, Any]] = {}

        for kv in kv_data:
            li = kv.metadata['layer_index']
            class_name = kv.metadata['class_name']
            scales = kv.metadata['scales']
            dq = CacheTranslator.dequantize_kv(kv.arrays, scales)

            if class_name == 'RotatingKVCache':
                all_states[li] = {
                    'class_name': class_name,
                    'state': (dq[0], dq[1]),
                    'meta_state': (
                        kv.metadata['max_size'],
                        kv.metadata['keep'],
                        kv.metadata['offset'],
                    ),
                }
            else:
                all_states[li] = {
                    'class_name': class_name,
                    'state': tuple(dq),
                    'meta_state': (kv.metadata.get('actual_end', dq[0].shape[-2]),),
                }

        for rec in recurrent_data:
            li = rec.metadata['layer_index']
            all_states[li] = {
                'class_name': rec.metadata['class_name'],
                'state': rec.arrays,
                'meta_state': (),
            }

        return [all_states[li] for li in sorted(all_states)]

    def fetch(self, request) -> CacheHit | None:
        from .turn_prefix_cache import reconstruct_cache_from_states

        segments = self.messages_to_segments(request)
        if not segments:
            return None

        path, _ = self._inner.match(segments)
        if not path:
            self._inner.release(path)
            return None

        cs = getattr(request, '_cache_state', None)
        if cs is not None:
            cs.turn_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            return CacheHit(
                cache=[],
                cached_tokens=0,
                remaining_tokens=list(request.prompt_token_ids),
                handle=path,
                hit_type='hit',
            )

        kv_data, rec_data = self._inner.collect_path_data(ancestor)
        assembled = self._assemble(kv_data, rec_data)
        del kv_data  # free merged bf16 intermediates; graph still rooted in assembled
        reconstructed = reconstruct_cache_from_states(assembled)
        del assembled  # Python wrappers released; graph now rooted only in reconstructed
        # Materialize the KVCache arrays now so the lazy computation graph
        # (int8 trie arrays → dequantize → concat → cast) is freed before decode
        # starts.  Without this every intermediate stays pinned in Metal memory
        # until BatchGenerator happens to touch the cache.
        for _layer in (reconstructed or []):
            _k = getattr(_layer, "keys", None)
            _v = getattr(_layer, "values", None)
            if _k is not None and not callable(_k):
                mx.eval(_k)
            if _v is not None and not callable(_v):
                mx.eval(_v)
        cached_tokens = ancestor.n_tokens
        remaining = list(request.prompt_token_ids[cached_tokens:])
        _turn_boundaries = getattr(request, '_turn_boundaries', None) or []
        prefill_boundaries = sorted(
            b - cached_tokens for b in _turn_boundaries if b > cached_tokens
        )
        return CacheHit(
            cache=reconstructed,
            cached_tokens=cached_tokens,
            remaining_tokens=remaining,
            handle=path,
            hit_type='hit',
            prefill_boundaries=prefill_boundaries,
        )

    def store(self, request, tokens: list[int] = None, cache: list = None) -> bool:
        from .turn_prefix_cache import Segment

        if cache is None and isinstance(tokens, list) and (not tokens or not isinstance(tokens[0], int)):
            cache = tokens
            tokens = None
        if cache is None:
            cache = []

        segments = self.messages_to_segments(request)
        if not segments or not getattr(request, 'output_token_ids', None):
            return False

        cs = getattr(request, '_cache_state', None)
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

        if cache:
            prev_end = path[-1].n_tokens if path else 0
            cache = self._slice_kv_to_delta(cache, prev_end)
            kv_sparse, rec_sparse = self._segment(cache)
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
        else:
            kv_layers, rec_layers = [], []

        self._inner.insert(
            parent,
            Segment(role='conversation', token_ids=response_tokens),
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
        )
        return True

    @staticmethod
    def _slice_kv_to_delta(states: list[dict], prev_end: int) -> list[dict]:
        """Slice KVCache state arrays to the incremental delta [prev_end:actual_end].

        RotatingKVCache is left untouched — its ring buffer is not a cumulative sequence.
        """
        if prev_end == 0:
            return states
        result = []
        for s in states:
            cname = s.get("class_name", "")
            if "KVCache" in cname and "Rotating" not in cname:
                state = s["state"]
                meta = s.get("meta_state") or ()
                actual_end = int(meta[0]) if meta else state[0].shape[2]
                sliced_state = tuple(mx.array(arr[:, :, prev_end:actual_end, :]) for arr in state[:2])
                new_meta = (actual_end - prev_end,) + tuple(meta[1:])
                s = {**s, "state": sliced_state, "meta_state": new_meta}
            result.append(s)
        return result

    def release(self, handle) -> None:
        if handle is not None:
            self._inner.release(handle)

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, total_tokens_prefilled: int, extracted_cache: list
    ) -> None:
        _turn_boundaries = getattr(request, '_turn_boundaries', None) or []
        if total_tokens_prefilled not in _turn_boundaries:
            return

        try:
            abs_idx = _turn_boundaries.index(total_tokens_prefilled)
        except ValueError:
            return

        segments = self.messages_to_segments(request)
        if abs_idx >= len(segments):
            return

        cs = getattr(request, '_cache_state', None)
        turn_path = cs.turn_path if cs is not None else []
        if len(turn_path) > abs_idx:
            return

        parent = turn_path[-1] if turn_path else self._inner.root
        segment = segments[abs_idx]
        is_sys = segment.role == 'system' and abs_idx == 0

        if extracted_cache:
            if not isinstance(extracted_cache[0], dict):
                from .kv_cache import extract_layer_state
                extracted_cache = [
                    d for layer in extracted_cache
                    if (d := extract_layer_state(layer)) is not None
                ]
            prev_end = _turn_boundaries[abs_idx - 1] if abs_idx > 0 else 0
            extracted_cache = self._slice_kv_to_delta(extracted_cache, prev_end)
            kv_sparse, rec_sparse = self._segment(extracted_cache)
            kv_layers = [kv for kv in kv_sparse if kv is not None]
            rec_layers = [rec for rec in rec_sparse if rec is not None]
        else:
            kv_layers, rec_layers = [], []

        new_node = self._inner.insert(
            parent, segment,
            kv_data=kv_layers or None,
            recurrent_data=rec_layers or None,
            is_system_prompt=is_sys,
        )
        if cs is not None:
            cs.turn_path.append(new_node)

    # PersistableCache extension
    def save(self, cache_dir: str) -> bool:
        self._inner.save(cache_dir)
        return True

    def load(self, cache_dir: str) -> int:
        self._inner.load(cache_dir)
        return 0
