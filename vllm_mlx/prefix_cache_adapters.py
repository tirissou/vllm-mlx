# SPDX-License-Identifier: Apache-2.0
"""Adapters that bridge concrete prefix cache backends to the PrefixCache protocol."""

from __future__ import annotations

import logging
from typing import Any

from vllm_mlx.turn_prefix_cache import TurnPrefixCache

from .kv_cache import CacheHit, PrefixCache

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class CacheManager:
    """Mixin providing per-step N-1 tracking machinery for prefix cache adapters."""

    _cache_index_map = None  # CacheIndexMap | None, per-instance (lazily initialized)

    def _ensure_cache_index_map(self, layers: list):
        """Classify layer indices into KV, RotatingKV, and recurrent buckets.

        Accepts either live prompt_cache objects (from update_n_minus_one)
        or extracted state dicts (from store()). Lazy-initializes once.
        """
        if self._cache_index_map is not None:
            return self._cache_index_map

        from .kv_cache import CacheIndexMap, _BATCH_KV_TYPES
        from mlx_lm.models.cache import BatchRotatingKVCache, ArraysCache

        kv_indices = []
        rotating_indices = []
        recurrent_indices = []

        for i, layer in enumerate(layers):
            if isinstance(layer, dict):
                # Extracted state dict
                name = layer.get("class_name", "")
                if "Rotating" in name:
                    rotating_indices.append(i)
                elif "KV" in name or "Quantized" in name:
                    kv_indices.append(i)
                else:
                    recurrent_indices.append(i)
            else:
                # Live cache object
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

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        """Default no-op. Override in adapters that track N-1 state."""
        pass

    def _reconstruct(self, request, extracted_cache: list) -> list:
        """Build N-1 state dict list from extracted N-state and per-step tracking.

        Replaces compose_n_minus_1_cache. Returns a list of state dicts in
        original layer order, ready for TurnPrefixCache._split_cache_arrays.
        """
        from .kv_cache import extract_layer_state

        idx_map = self._cache_index_map
        cs = request._cache_state
        n_minus_one = cs.n_minus_one_state  # {"rotating": [...], "recurrent": [...]}

        result = [None] * len(extracted_cache)

        # Standard KV: offset - 1
        for i in idx_map.kv_indices:
            layer = extracted_cache[i]
            meta = layer.get("meta_state")
            if meta and len(meta) > 0:
                new_meta = (str(max(0, int(meta[0]) - 1)),) + meta[1:]
                result[i] = {**layer, "meta_state": new_meta}
            else:
                result[i] = layer

        # RotatingKV: use shadow instances
        shadows = n_minus_one["rotating"] if n_minus_one else []
        for shadow_idx, layer_idx in enumerate(idx_map.rotating_indices):
            if shadow_idx < len(shadows):
                shadow = shadows[shadow_idx]
                state_dict = extract_layer_state(shadow)
                if state_dict is not None:
                    result[layer_idx] = state_dict
                else:
                    result[layer_idx] = extracted_cache[layer_idx]
            else:
                result[layer_idx] = extracted_cache[layer_idx]

        # Recurrent: use saved ArraysCache refs
        saved_recurrent = (n_minus_one or {}).get("recurrent") or []
        for rec_idx, layer_idx in enumerate(idx_map.recurrent_indices):
            if rec_idx < len(saved_recurrent):
                saved = saved_recurrent[rec_idx]
                state_dict = extract_layer_state(saved)
                if state_dict is not None:
                    result[layer_idx] = state_dict
                else:
                    result[layer_idx] = extracted_cache[layer_idx]
            else:
                result[layer_idx] = extracted_cache[layer_idx]

        return result


class MemoryCacheAdapter(CacheManager):
    """Adapts MemoryAwarePrefixCache to the PrefixCache / PersistableCache protocol."""

    def __init__(self, inner, mid_prefill_save_interval: int = 0):
        self._inner = inner
        self._save_interval = mid_prefill_save_interval

    def fetch(self, request) -> CacheHit | None:
        tokens = list(request.prompt_token_ids)
        cache, remaining = self._inner.fetch(tokens)
        if cache is None:
            return None
        cached_tokens = len(tokens) - len(remaining)
        return CacheHit(
            cache=cache,
            cached_tokens=cached_tokens,
            remaining_tokens=list(remaining),
            handle=None,
            hit_type=getattr(self._inner, "_last_match_type", "hit"),
        )

    def store(self, request, cache: list) -> bool:
        cs = getattr(request, "_cache_state", None)
        _st = (cs.store_tokens if cs is not None else None)
        tokens = _st if isinstance(_st, list) else list(request.prompt_token_ids)
        # Memory cache requires live KV objects; reconstruct from dict form if needed
        if cache and isinstance(cache[0], dict):
            from .turn_prefix_cache import reconstruct_cache_from_states
            cache = reconstruct_cache_from_states(cache) or []
        if not cache:
            return False
        return self._inner.store(tokens, cache, evict_prefixes=False)

    def release(self, handle: Any) -> None:
        pass  # memory cache has no handle lifecycle

    def get_stats(self) -> dict:
        return self._inner.get_stats()

    def clear(self) -> None:
        self._inner.clear()

    def on_prefill_checkpoint(
        self, request, processed_tokens: int, extracted_cache: list
    ) -> None:
        from .turn_prefix_cache import reconstruct_cache_from_states

        cs = getattr(request, "_cache_state", None)
        total_cached = ((cs.cached_tokens if cs is not None else None) or 0) + processed_tokens
        last_save = (cs.mid_prefill_last_save if cs is not None else 0)

        interval = self._save_interval
        if interval > 0 and total_cached - last_save < interval:
            return

        reconstructed = reconstruct_cache_from_states(extracted_cache)
        if not reconstructed:
            return

        prefix_tokens = list((request.prompt_token_ids or [])[:total_cached])
        old_key = (cs.mid_prefill_cache_key if cs is not None else None)
        if old_key is not None:
            self._inner.remove(list(old_key))

        if self._inner.store(prefix_tokens, reconstructed):
            if cs is not None:
                cs.mid_prefill_last_save = total_cached
                cs.mid_prefill_cache_key = tuple(prefix_tokens)

    def set_spill_delegate(self, on_spill, on_promote) -> None:
        """Forward spill delegate registration to the underlying SpillableCache."""
        self._inner.set_spill_delegate(on_spill, on_promote)

    # PersistableCache extension
    def save(self, cache_dir: str) -> bool:
        return self._inner.save_to_disk(cache_dir)

    def load(self, cache_dir: str) -> int:
        return self._inner.load_from_disk(cache_dir)


class TurnCacheAdapter(CacheManager):
    """Adapts TurnPrefixCache to the PrefixCache / PersistableCache protocol."""

    def __init__(self, inner: TurnPrefixCache):
        self._inner = inner

    @staticmethod
    def messages_to_segments(request) -> list:
        """Split a request's token sequence into per-message Segment objects.

        Pure function of request.prompt_token_ids and request._turn_boundaries.
        """
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []

        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if not _turn_boundaries:
            return []

        B_sys = _turn_boundaries[0]
        if B_sys <= 0 or B_sys >= len(full_tokens):
            return []

        segments: list = [Segment(role="system", token_ids=full_tokens[:B_sys])]

        prev = B_sys
        for B_k in _turn_boundaries[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(Segment(role="conversation", token_ids=full_tokens[prev:B_k]))
                prev = B_k

        if prev < len(full_tokens):
            segments.append(Segment(role="user", token_ids=full_tokens[prev:]))

        return segments if len(segments) > 1 else []

    def fetch(self, request) -> CacheHit | None:
        from .turn_prefix_cache import reconstruct_cache_from_states

        segments = self.messages_to_segments(request)
        # logger.debug(f"Fetch segments: {segments}")
        if not segments:
            return None

        path, _ = self._inner.match(segments)
        # logger.debug(f"Fetch matched path: {path}")
        if not path:
            self._inner.release(path)
            return None

        # Store path on request so _cleanup_finished can access it for the store step
        cs = getattr(request, "_cache_state", None)
        if cs is not None:
            cs.adapter_state = path
        else:
            request._turn_cache_path = path

        ancestor = self._inner.find_checkpoint_ancestor(path)
        if ancestor is None:
            return CacheHit(
                cache=[],
                cached_tokens=0,
                remaining_tokens=list(request.prompt_token_ids),
                handle=path,
                hit_type="hit",
            )

        assembled = self._inner._retrieve_full_cache(ancestor)
        reconstructed = reconstruct_cache_from_states(assembled)
        cached_tokens = ancestor.n_tokens
        remaining = list(request.prompt_token_ids[cached_tokens:])
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        prefill_boundaries = sorted(
            b - cached_tokens
            for b in _turn_boundaries
            if b > cached_tokens
        )
        return CacheHit(
            cache=reconstructed,
            cached_tokens=cached_tokens,
            remaining_tokens=remaining,
            handle=path,
            hit_type="hit",
            prefill_boundaries=prefill_boundaries,
        )

    def update_n_minus_one(self, request, prompt_cache: list, uid_idx: int) -> None:
        """Capture N-1 state before each decode step.

        On first call: initialize shadow RotatingKVCache instances.
        On subsequent calls: mirror the previous step's write into shadows,
        and save recurrent refs.
        """
        from mlx_lm.models.cache import RotatingKVCache, BatchRotatingKVCache, ArraysCache

        cs = getattr(request, "_cache_state", None)
        if cs is None:
            return

        idx_map = self._ensure_cache_index_map(prompt_cache)
        just_initialized = (cs.n_minus_one_state is None)

        if just_initialized:
            # Initialize one shadow RotatingKVCache per rotating layer
            shadows = []
            for i in idx_map.rotating_indices:
                live = prompt_cache[i]  # BatchRotatingKVCache
                shadow = RotatingKVCache(max_size=live.max_size, keep=0)
                shadows.append(shadow)
            cs.n_minus_one_state = {"rotating": shadows, "recurrent": None}
            # No mirroring on first call — no previous decode step yet
            return

        # Mirror previous step's RotatingKV write into shadow
        shadows = cs.n_minus_one_state["rotating"]
        for shadow_idx, layer_idx in enumerate(idx_map.rotating_indices):
            live = prompt_cache[layer_idx]  # BatchRotatingKVCache
            shadow = shadows[shadow_idx]
            # _idx-1 is always the last-written slot after the previous next() call
            prev_slot = live._idx - 1
            k_prev = live.keys[uid_idx:uid_idx+1, :, prev_slot:prev_slot+1, :]
            v_prev = live.values[uid_idx:uid_idx+1, :, prev_slot:prev_slot+1, :]
            shadow.update_and_fetch(k_prev, v_prev)

        # Save recurrent refs (valid because ArraysCache uses reference replacement)
        if idx_map.recurrent_indices:
            saved = []
            for layer_idx in idx_map.recurrent_indices:
                live = prompt_cache[layer_idx]  # ArraysCache (batched)
                saved.append(live.extract(uid_idx))
            cs.n_minus_one_state["recurrent"] = saved

    def store(self, request, cache: list) -> bool:
        from .turn_prefix_cache import Segment

        segments = self.messages_to_segments(request)
        if not segments or not getattr(request, "output_token_ids", None):
            return False

        cs = getattr(request, "_cache_state", None)
        path = (cs.adapter_state if cs is not None else None) or getattr(request, "_turn_cache_path", None) or []
        matched_depth = len(path)
        parent = path[-1] if path else self._inner.root
        new_segments = segments[matched_depth:]

        if not new_segments:
            return False

        response_tokens = list(segments[-1].token_ids) + list(request.output_token_ids)
        # Normalize to dict form if raw KV layer objects were passed
        if cache and not isinstance(cache[0], dict):
            from .kv_cache import extract_layer_state
            cache = [d for layer in cache if (d := extract_layer_state(layer)) is not None]

        resp_state = cache if cache else None

        if resp_state is not None:
            self._ensure_cache_index_map(resp_state)
            n_minus_one = cs.n_minus_one_state if cs is not None else None
            if n_minus_one is not None:
                # Decode steps occurred — use per-step N-1 tracking
                resp_state = self._reconstruct(request, resp_state)

        resp_kv, resp_recur = (
            self._inner._split_cache_arrays(resp_state, parent.n_tokens)
            if resp_state is not None else ([], None)
        )
        self._inner.insert(
            parent,
            Segment(role="conversation", token_ids=response_tokens),
            resp_kv, None, resp_recur,
        )
        return True

    def release(self, handle) -> None:
        if handle is not None:
            self._inner.release(handle)

    def get_stats(self) -> dict:
        return {}

    def clear(self) -> None:
        pass

    def on_prefill_checkpoint(
        self, request, processed_tokens: int, extracted_cache: list
    ) -> None:
        cs = getattr(request, "_cache_state", None)
        total_cached = ((cs.cached_tokens if cs is not None else None) or 0) + processed_tokens
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []

        # insert_segments() fires end_of_segment AT boundary B (not B-1).
        if total_cached not in _turn_boundaries:
            return

        try:
            abs_idx = _turn_boundaries.index(total_cached)
        except ValueError:
            return

        segments = self.messages_to_segments(request)
        if abs_idx >= len(segments):
            return

        adapter_state = (cs.adapter_state if cs is not None else None) or getattr(request, "_turn_cache_path", None) or []

        # Guard: skip if this boundary was already inserted (e.g. duplicate callback).
        if len(adapter_state) > abs_idx:
            return

        parent = adapter_state[-1] if adapter_state else self._inner.root
        segment = segments[abs_idx]
        is_sys = segment.role == "system" and abs_idx == 0

        kv_slice, recur = self._inner._split_cache_arrays(extracted_cache, parent.n_tokens)
        new_node = self._inner.insert(parent, segment, kv_slice, None, recur, is_system_prompt=is_sys)

        # Record the new node so store() and subsequent checkpoints can chain off it.
        if cs is not None:
            if not isinstance(cs.adapter_state, list):
                cs.adapter_state = list(cs.adapter_state or [])
            cs.adapter_state.append(new_node)
        else:
            if not isinstance(getattr(request, "_turn_cache_path", None), list):
                request._turn_cache_path = []
            request._turn_cache_path.append(new_node)

    # PersistableCache extension
    def save(self, cache_dir: str) -> bool:
        self._inner.save(cache_dir)
        return True

    def load(self, cache_dir: str) -> int:
        self._inner.load(cache_dir)
        return 0


class PagedCacheAdapter(CacheManager):
    """Adapts BlockAwarePrefixCache to the PrefixCache protocol.

    Maintains an internal {request_id → block_table} mapping so the protocol
    surface stays clean (no block_table on Request).
    """

    def __init__(self, inner):
        self._inner = inner
        self._block_tables: dict = {}

    def fetch(self, request) -> CacheHit | None:
        tokens = list(request.prompt_token_ids)
        block_table, remaining = self._inner.fetch_cache(request.request_id, tokens)
        if block_table is None:
            return None

        self._block_tables[request.request_id] = block_table
        cache = self._inner.reconstruct_cache(block_table)
        cached_tokens = block_table.num_tokens
        return CacheHit(
            cache=cache,
            cached_tokens=cached_tokens,
            remaining_tokens=list(remaining),
            handle=request.request_id,
            hit_type="hit",
        )

    def store(self, request, cache: list) -> bool:
        cs = getattr(request, "_cache_state", None)
        _st = (cs.store_tokens if cs is not None else None)
        tokens = _st if isinstance(_st, list) else list(request.prompt_token_ids)
        self._inner.store_cache(request.request_id, tokens, cache)
        return True

    def release(self, handle) -> None:
        if handle is None:
            return
        self._inner.release_cache(handle)
        self._block_tables.pop(handle, None)

    def get_stats(self) -> dict:
        raw = self._inner.get_stats()
        if isinstance(raw, dict):
            return raw
        return vars(raw) if hasattr(raw, "__dict__") else {}

    def clear(self) -> None:
        self._inner.clear()
        self._block_tables.clear()

    def on_prefill_checkpoint(
        self, request, processed_tokens: int, extracted_cache: list
    ) -> None:
        pass


class LegacyCacheAdapter(CacheManager):
    """Adapts PrefixCacheManager (entry-count based cache) to the PrefixCache protocol."""

    def __init__(self, inner):
        self._inner = inner

    def fetch(self, request) -> CacheHit | None:
        tokens = list(request.prompt_token_ids)
        cache, remaining = self._inner.fetch_cache(tokens)
        if not cache:
            return None
        cached_tokens = len(tokens) - len(remaining)
        return CacheHit(
            cache=cache,
            cached_tokens=cached_tokens,
            remaining_tokens=list(remaining),
            handle=None,
            hit_type="hit",
        )

    def store(self, request, cache: list) -> bool:
        cs = getattr(request, "_cache_state", None)
        _st = (cs.store_tokens if cs is not None else None)
        tokens = _st if isinstance(_st, list) else list(request.prompt_token_ids)
        self._inner.store_cache(tokens, cache)
        return True

    def release(self, handle: Any) -> None:
        pass

    def get_stats(self) -> dict:
        stats = self._inner.get_stats()
        if isinstance(stats, dict):
            return stats
        return vars(stats) if hasattr(stats, "__dict__") else {}

    def clear(self) -> None:
        self._inner.clear()

    def on_prefill_checkpoint(
        self, request, processed_tokens: int, extracted_cache: list
    ) -> None:
        pass
