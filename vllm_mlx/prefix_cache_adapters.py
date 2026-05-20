# SPDX-License-Identifier: Apache-2.0
"""Adapters that bridge concrete prefix cache backends to the PrefixCache protocol."""

from __future__ import annotations

from typing import Any

from .kv_cache import CacheHit


class MemoryCacheAdapter:
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


class TurnCacheAdapter:
    """Adapts TurnPrefixCache to the PrefixCache / PersistableCache protocol."""

    def __init__(self, inner):
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
        if not segments:
            return None

        path, _ = self._inner.match(segments)
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
                cache=None,
                cached_tokens=0,
                remaining_tokens=list(request.prompt_token_ids),
                handle=path,
                hit_type="hit",
            )

        assembled = self._inner._retrieve_full_cache(ancestor)
        reconstructed = reconstruct_cache_from_states(assembled)
        cached_tokens = ancestor.n_tokens
        remaining = list(request.prompt_token_ids[cached_tokens:])
        return CacheHit(
            cache=reconstructed,
            cached_tokens=cached_tokens,
            remaining_tokens=remaining,
            handle=path,
            hit_type="hit",
        )

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
        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        _boundary_states = getattr(request, "_boundary_states", None) or {}

        if not new_segments:
            return False

        for i, segment in enumerate(new_segments[:-1]):
            abs_idx = matched_depth + i
            is_sys = segment.role == "system" and abs_idx == 0
            full_state = (
                _boundary_states.get(_turn_boundaries[abs_idx])
                if abs_idx < len(_turn_boundaries) else None
            )
            kv_slice, recur = (
                self._inner._split_cache_arrays(full_state, parent.n_tokens)
                if full_state else ([], None)
            )
            parent = self._inner.insert(
                parent, segment, kv_slice, None, recur, is_system_prompt=is_sys
            )

        response_tokens = list(segments[-1].token_ids) + list(request.output_token_ids)
        # Normalize to dict form if raw KV layer objects were passed
        if cache and not isinstance(cache[0], dict):
            from .kv_cache import extract_layer_state
            cache = [d for layer in cache if (d := extract_layer_state(layer)) is not None]
        resp_state = cache if cache else None
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

        # Split-chunk convention: checkpoint fires at B-1, boundary key is B.
        if total_cached + 1 not in _turn_boundaries:
            return

        if not hasattr(request, "_boundary_states") or request._boundary_states is None:
            request._boundary_states = {}
        request._boundary_states[total_cached + 1] = extracted_cache

    # PersistableCache extension
    def save(self, cache_dir: str) -> bool:
        self._inner.save(cache_dir)
        return True

    def load(self, cache_dir: str) -> int:
        self._inner.load(cache_dir)
        return 0


class PagedCacheAdapter:
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


class LegacyCacheAdapter:
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
