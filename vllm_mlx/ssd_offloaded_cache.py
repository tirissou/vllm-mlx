# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from typing import Any

from .kv_cache import CacheHit, CacheDiskStore, SpillableCache


class SSDOffloadedCache:
    """Decorator: wraps any SpillableCache with transparent SSD offloading.

    - On spill: writes to CacheDiskStore (via spill delegate registered on inner)
    - On promote (TurnPrefixCache path): reads synchronously via on_promote hook
    - On fetch miss (MemoryAwarePrefixCache path): enqueues background promotion;
      result available on next fetch call
    - save/load: uses same CacheDiskStore for persistence

    See CONTEXT.md and ADR-0003 for design rationale.
    """

    def __init__(self, inner: SpillableCache, store: CacheDiskStore) -> None:
        self._inner = inner
        self._store = store
        # Background-promoted entries waiting to be returned on next fetch.
        self._promoted: dict[tuple[int, ...], list] = {}
        self._promoted_lock = threading.Lock()
        # Tokens currently in the promotion queue (dedup guard).
        self._in_flight: set[tuple[int, ...]] = set()
        self._queue: queue.Queue[tuple[int, ...] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        inner.set_spill_delegate(self._on_spill, self._on_promote)

    # ── Spill delegate ────────────────────────────────────────────────────────

    def _on_spill(self, tokens: tuple[int, ...], layers: list) -> tuple[int, ...]:
        """Called by inner cache when spilling arrays. Writes to disk."""
        self._store.write(tokens, layers)
        return tokens  # handle IS the token key

    def _on_promote(self, handle: tuple[int, ...]) -> list | None:
        """Called by inner cache (TurnPrefixCache) when it needs spilled arrays back."""
        return self._store.read(handle)

    # ── PrefixCache protocol ──────────────────────────────────────────────────

    def fetch(self, request) -> CacheHit | None:
        hit = self._inner.fetch(request)
        if hit is not None:
            return hit

        tokens = tuple(request.prompt_token_ids)

        # Return a completed background promotion if available;
        # also guard the dedup check-then-add under the same lock.
        with self._promoted_lock:
            if tokens in self._promoted:
                layers = self._promoted.pop(tokens)
                self._in_flight.discard(tokens)
                return CacheHit(
                    cache=layers,
                    cached_tokens=len(tokens),
                    remaining_tokens=[],
                    hit_type="ssd_hit",
                )
            # Enqueue for background promotion (dedup).
            if tokens not in self._in_flight and self._store.has(tokens):
                self._in_flight.add(tokens)
                self._queue.put_nowait(tokens)

        return None

    def store(self, request, cache: list) -> bool:
        return self._inner.store(request, cache)

    def release(self, handle: Any) -> None:
        self._inner.release(handle)

    def get_stats(self) -> dict:
        return self._inner.get_stats()

    def clear(self) -> None:
        self._inner.clear()
        with self._promoted_lock:
            self._promoted.clear()
            self._in_flight.clear()
        # Drain any pending promotions.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def on_prefill_checkpoint(
        self, request: Any, processed_tokens: int, extracted_cache: list
    ) -> None:
        self._inner.on_prefill_checkpoint(request, processed_tokens, extracted_cache)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background promotion thread."""
        self._thread = threading.Thread(
            target=self._promotion_loop, daemon=True, name="ssd-promote"
        )
        self._thread.start()

    def close(self) -> None:
        """Stop background promotion thread."""
        if self._thread is None:
            return
        self._queue.put(None)  # sentinel
        self._thread.join()

    def _promotion_loop(self) -> None:
        while True:
            tokens = self._queue.get()
            if tokens is None:
                break
            layers = self._store.read(tokens)
            if layers is not None:
                with self._promoted_lock:
                    self._promoted[tokens] = layers
            else:
                # Read failed — remove from in-flight so future fetches can retry.
                with self._promoted_lock:
                    self._in_flight.discard(tokens)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, cache_dir: str | None = None) -> bool:
        """Flush in-memory promoted entries to disk store.

        `cache_dir` is ignored — the underlying CacheDiskStore owns its path.
        """
        with self._promoted_lock:
            for tokens, layers in self._promoted.items():
                self._store.write(tokens, layers)
        return True

    def load(self, cache_dir: str | None = None) -> int:
        """Pre-populate _promoted from all keys currently on disk.

        `cache_dir` is ignored — the underlying CacheDiskStore owns its path.
        WARNING: loads all disk keys into memory. Only call on small stores.
        """
        count = 0
        for tokens in self._store.all_keys():
            layers = self._store.read(tokens)
            if layers is not None:
                with self._promoted_lock:
                    self._promoted[tokens] = layers
                count += 1
        return count
