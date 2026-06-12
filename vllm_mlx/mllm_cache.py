# SPDX-License-Identifier: Apache-2.0
"""
MLLM (Multimodal Language Model) Prefix Cache Manager.

This module provides advanced caching for MLLM inference, implementing
the LMCache-style approach for multimodal prefix caching:

Features:
- Image content hashing for cache keys (LMCache style)
- Vision embedding caching (skip encoder on hit)
- KV cache state caching with prefix matching
- Token ID tracking for partial prefix reuse
- LRU eviction policy with memory limits
- Stats tracking (hits, misses, tokens saved, encoder skips)

Based on research from:
- LMCache: https://blog.lmcache.ai/2025-07-03-multimodal-models/
- vLLM Prefix Caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- mlx-lm cache_prompt: https://github.com/ml-explore/mlx-lm
"""

import bisect
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_mlx.kv_quant_bench import (
    _QuantizedCacheWrapper,
    _dequantize_cache,
    _quantize_cache,
    estimate_kv_cache_memory,
)

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails
# Bump this when the cache on-disk format or KV semantics change.
# Loading a cache with a different version is rejected automatically.
_CACHE_PERSIST_VERSION = 3


@dataclass
class MLLMCacheStats:
    """Statistics for MLLM cache performance."""

    hits: int = 0
    misses: int = 0
    partial_hits: int = 0  # Prefix matched but not full
    tokens_saved: int = 0
    image_cache_hits: int = 0
    vision_encoder_skips: int = 0  # Times we skipped vision encoder
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "partial_hits": self.partial_hits,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "image_cache_hits": self.image_cache_hits,
            "vision_encoder_skips": self.vision_encoder_skips,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


@dataclass
class MLLMPrefixCacheEntry:
    """
    Enhanced cache entry storing vision embeddings, KV cache, and token IDs.

    This enables:
    1. Skipping vision encoder on image cache hit (saves ~1-2s per image)
    2. Skipping prefix computation on token match (saves ~0.5s per 1k tokens)
    3. Partial prefix reuse for multi-turn conversations
    """

    # Identity
    image_hash: str  # SHA256 of image content
    prompt_hash: str  # SHA256 of formatted prompt

    # Cached states - the key to performance
    vision_embeddings: Any = None  # Output of vision encoder (skip encoder on hit!)
    kv_cache: list[Any] = field(default_factory=list)  # Language model KV states

    # Token tracking for prefix matching
    token_ids: list[int] = field(default_factory=list)  # Full token sequence
    num_image_tokens: int = 0  # e.g., 256 for Gemma 3
    num_text_tokens: int = 0
    prompt_tokens: int = 0  # Total prompt tokens (for stats)

    # Metadata
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    model_name: str = ""

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def memory_size(self) -> int:
        """Estimate memory usage in bytes."""
        size = 0
        if self.vision_embeddings is not None:
            if hasattr(self.vision_embeddings, "nbytes"):
                size += self.vision_embeddings.nbytes
        if self.kv_cache is not None:
            for layer_cache in self.kv_cache:
                if hasattr(layer_cache, "state"):
                    state = layer_cache.state
                    if state is not None:
                        for tensor in state:
                            if hasattr(tensor, "nbytes"):
                                size += tensor.nbytes
        return size

    def get_prefix_match_length(self, new_token_ids: list[int]) -> int:
        """
        Find how many tokens match between cached prefix and new input.

        This is the key to prefix caching - if the first N tokens match,
        we can skip computing KV states for those N tokens.
        """
        match_length = 0
        for i, (cached, new) in enumerate(zip(self.token_ids, new_token_ids)):
            if cached != new:
                break
            match_length = i + 1
        return match_length


def compute_image_hash(image_path: str) -> str:
    """
    Compute hash of image content for cache key.

    Following LMCache approach: hash the actual image bytes, not the path.
    This ensures cache hits even when the same image is loaded from
    different paths or as base64.

    Args:
        image_path: Path to image file

    Returns:
        SHA256 hash of image content (first 16 chars)
    """
    try:
        path = Path(image_path)
        if path.exists():
            # Hash file content - this is the LMCache approach
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        else:
            # Hash the string itself (for URLs or base64)
            return hashlib.sha256(image_path.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Failed to hash image: {e}")
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]


def compute_images_hash(images: list[str]) -> str:
    """
    Compute combined hash for multiple images.

    Args:
        images: List of image paths/URLs

    Returns:
        Combined hash string
    """
    if not images:
        return "no_images"

    hashes = [compute_image_hash(img) for img in images]
    combined = "_".join(sorted(hashes))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class MLLMPrefixCacheManager:
    """
    LRU Cache manager for MLLM prefix states with vision embedding caching.

    Implements the LMCache approach for multimodal caching:
    1. Hash-based identification of image+prompt combinations
    2. Vision embedding caching (skip encoder on hit - saves 1-2s!)
    3. KV cache reuse for matching prefixes
    4. Token ID tracking for partial prefix matching
    5. Memory-based eviction (configurable limit)

    Example:
        >>> cache = MLLMPrefixCacheManager(max_memory_mb=2048)
        >>> # First request - cache miss, full computation
        >>> entry, match_len = cache.fetch(["image.jpg"], prompt, token_ids)
        >>> # ... run full forward pass ...
        >>> cache.store(["image.jpg"], prompt, vision_emb, kv_cache, token_ids)
        >>>
        >>> # Second request with same image - cache hit!
        >>> entry, match_len = cache.fetch(["image.jpg"], prompt, token_ids)
        >>> # entry.vision_embeddings available - skip encoder!
        >>> # match_len > 0 - skip prefix computation!

    Performance (Gemma 3 27B, 256 image tokens):
        - Vision encoder: ~1.5s -> 0s (skip on hit)
        - Prefix computation: ~0.5s/1k tokens -> 0s (skip on match)
        - Multi-turn speedup: 8-12x for subsequent turns
    """

    def __init__(
        self,
        max_entries: int = 50,
        max_memory_mb: int = 2048,
    ):
        """
        Initialize MLLM prefix cache manager.

        Args:
            max_entries: Maximum number of cache entries (default: 50)
            max_memory_mb: Maximum memory in MB (default: 2048)
        """
        self.max_size = max_entries
        self.max_memory = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, MLLMPrefixCacheEntry] = OrderedDict()
        self._current_memory = 0
        self.stats = MLLMCacheStats()

    def _make_cache_key(self, images: list[str], prompt: str) -> str:
        """Create cache key from images and prompt."""
        image_hash = compute_images_hash(images)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"{image_hash}_{prompt_hash}"

    def _make_image_only_key(self, images: list[str]) -> str:
        """Create cache key for image-only lookup (vision embedding reuse)."""
        return compute_images_hash(images)

    def _evict_by_memory(self, required_size: int) -> None:
        """Evict entries until we have enough memory."""
        while self._current_memory + required_size > self.max_memory and self._cache:
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            self._current_memory -= oldest_entry.memory_size
            self.stats.evictions += 1
            logger.debug(f"MLLM cache evicted (memory): {oldest_key[:20]}...")

    def _evict_by_count(self) -> None:
        """Evict entries until we're under max_size."""
        while len(self._cache) >= self.max_size and self._cache:
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            self._current_memory -= oldest_entry.memory_size
            self.stats.evictions += 1
            logger.debug(f"MLLM cache evicted (count): {oldest_key[:20]}...")

    def fetch(
        self,
        images: list[str],
        prompt: str,
        token_ids: list[int] | None = None,
    ) -> tuple[MLLMPrefixCacheEntry | None, int]:
        """
        Fetch cached prefix state with prefix matching.

        This is the main entry point for cache lookups. Returns both
        the cache entry (if found) and the prefix match length.

        Args:
            images: List of image paths
            prompt: Text prompt
            token_ids: Optional token IDs for prefix matching

        Returns:
            Tuple of (entry, prefix_match_length) where:
            - entry: The cache entry if found, None otherwise
            - prefix_match_length: Number of tokens that match (0 if miss)
        """
        self.stats.total_queries += 1
        cache_key = self._make_cache_key(images, prompt)

        if cache_key in self._cache:
            # Full cache hit - exact image+prompt match
            entry = self._cache.pop(cache_key)
            self._cache[cache_key] = entry  # Move to end (LRU)
            entry.hit_count += 1

            self.stats.hits += 1
            if images:
                self.stats.image_cache_hits += 1
            if entry.vision_embeddings is not None:
                self.stats.vision_encoder_skips += 1

            # Calculate prefix match length
            match_length = entry.total_tokens
            if token_ids:
                match_length = entry.get_prefix_match_length(token_ids)
                if match_length < entry.total_tokens:
                    self.stats.partial_hits += 1

            self.stats.tokens_saved += match_length
            logger.debug(
                f"MLLM cache HIT: {cache_key[:32]}..., prefix_match={match_length}"
            )

            return entry, match_length

        # Check for image-only match (can reuse vision embeddings)
        if images:
            image_key = self._make_image_only_key(images)
            for key, entry in self._cache.items():
                if (
                    entry.image_hash == image_key
                    and entry.vision_embeddings is not None
                ):
                    # Image match - can reuse vision embeddings!
                    self.stats.partial_hits += 1
                    self.stats.vision_encoder_skips += 1
                    logger.debug(
                        f"MLLM cache PARTIAL HIT (vision only): image={image_key[:16]}"
                    )

                    # Return entry for vision embeddings, but 0 prefix match
                    # (prompt is different, so KV cache can't be reused)
                    return entry, 0

        self.stats.misses += 1
        logger.debug(f"MLLM cache MISS: {cache_key[:32]}...")
        return None, 0

    def fetch_cache(
        self,
        images: list[str],
        prompt: str,
    ) -> tuple[list[Any] | None, bool]:
        """
        Legacy API: Fetch cached KV state for image+prompt combination.

        For backwards compatibility with existing code.
        """
        entry, match_len = self.fetch(images, prompt)
        # For legacy API, return hit if entry exists (don't require token match)
        if entry is not None and entry.kv_cache is not None:
            return entry.kv_cache, True
        return None, False

    def store(
        self,
        images: list[str],
        prompt: str,
        vision_embeddings: Any,
        kv_cache: list[Any],
        token_ids: list[int],
        num_image_tokens: int = 0,
        model_name: str = "",
    ) -> None:
        """
        Store prefix state in cache.

        Args:
            images: List of image paths
            prompt: Text prompt
            vision_embeddings: Output of vision encoder (can be None for text-only)
            kv_cache: Language model KV cache states
            token_ids: Full token sequence
            num_image_tokens: Number of image tokens (e.g., 256 for Gemma 3)
            model_name: Model name for validation
        """
        cache_key = self._make_cache_key(images, prompt)

        entry = MLLMPrefixCacheEntry(
            image_hash=compute_images_hash(images),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            vision_embeddings=vision_embeddings,
            kv_cache=kv_cache,
            token_ids=token_ids,
            num_image_tokens=num_image_tokens,
            num_text_tokens=len(token_ids) - num_image_tokens,
            prompt_tokens=len(token_ids),
            model_name=model_name,
        )

        # Evict by memory first
        self._evict_by_memory(entry.memory_size)

        # Then evict by count
        self._evict_by_count()

        self._cache[cache_key] = entry
        self._current_memory += entry.memory_size

        logger.debug(
            f"MLLM cache STORED: key={cache_key[:32]}..., "
            f"tokens={len(token_ids)}, vision_emb={vision_embeddings is not None}, "
            f"memory={entry.memory_size / 1024 / 1024:.1f}MB"
        )

    def store_cache(
        self,
        images: list[str],
        prompt: str,
        cache: list[Any] | None,
        num_tokens: int = 0,
    ) -> None:
        """
        Legacy API: Store KV cache for future reuse.

        For backwards compatibility with existing code.
        """
        # Don't store empty or None caches
        if cache is None or (isinstance(cache, list) and len(cache) == 0):
            return

        self.store(
            images=images,
            prompt=prompt,
            vision_embeddings=None,
            kv_cache=cache,
            token_ids=[0] * num_tokens,  # Dummy token IDs
            num_image_tokens=0,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["entries"] = len(self._cache)
        stats["max_entries"] = self.max_size
        stats["memory_used_mb"] = self._current_memory / 1024 / 1024
        stats["max_memory_mb"] = self.max_memory / 1024 / 1024
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = MLLMCacheStats()

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._cache.clear()
        self._current_memory = 0
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __repr__(self) -> str:
        mem_mb = self._current_memory / 1024 / 1024
        return f"<MLLMPrefixCacheManager entries={len(self)} memory={mem_mb:.1f}MB>"


# Short alias for convenience
MLLMCacheManager = MLLMPrefixCacheManager

# Legacy aliases for backwards compatibility
VLMCacheStats = MLLMCacheStats
VLMPrefixCacheEntry = MLLMPrefixCacheEntry
VLMCacheEntry = MLLMPrefixCacheEntry
VLMPrefixCacheManager = MLLMPrefixCacheManager
VLMCacheManager = MLLMPrefixCacheManager

# ---------------------------------------------------------------------------
# Memory-aware prefix cache — public seam for MLLM batch generator / scheduler.
#
# This was previously the ``vllm_mlx.memory_cache`` module; it was inlined here
# in Phase 7 of the SSD persistence redesign so callers can depend on a single
# cache module.  ``MemoryAwarePrefixCache`` is deprecated by the ADR-0003
# addendum but still used by the live MLLM pipeline.
# ---------------------------------------------------------------------------


def _get_available_memory() -> int:
    """Get available system memory in bytes (0 if detection fails)."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers for reduced memory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization to apply.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )

    def compute_memory_limit(self) -> int:
        """Compute the memory limit in bytes."""
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
        )


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create copies of cache layers with the last ``trim_by`` positions removed.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).

    For plain KVCache: reduces offset (surplus data beyond offset is harmless
    since merge slices to ``keys[:, :, :offset, :]``).

    For RotatingKVCache: actually trims the circular buffer — reducing offset
    alone breaks ``size()`` / ``_temporal_order`` invariants.

    Supports KVCache, RotatingKVCache, and _QuantizedCacheWrapper.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import RotatingKVCache

    trimmed: list[Any] = []
    eval_targets: list[Any] = []
    for layer_cache in cache:
        if isinstance(layer_cache, _QuantizedCacheWrapper):
            # Shallow copy with reduced offset
            tc = _QuantizedCacheWrapper.__new__(_QuantizedCacheWrapper)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            tc.bits = layer_cache.bits
            tc.group_size = layer_cache.group_size
            tc.orig_type = layer_cache.orig_type
            tc.orig_attrs = layer_cache.orig_attrs
            trimmed.append(tc)
        elif isinstance(layer_cache, RotatingKVCache):
            if layer_cache.keys is None or trim_by <= 0:
                trimmed.append(layer_cache)
                continue
            # RotatingKVCache: must trim buffer, not just offset.
            # The buffer stores the last min(offset, max_size) tokens in a
            # circular arrangement.  Trimming excess positions from the END
            # means removing the newest entries (chronologically last).
            old_offset = layer_cache.offset
            new_offset = max(old_offset - trim_by, 0)
            old_size = min(old_offset, layer_cache.max_size)
            entries_to_keep = max(0, old_size - trim_by)

            orig_cls = type(layer_cache)
            tc = orig_cls.__new__(orig_cls)
            tc.offset = new_offset
            tc.max_size = layer_cache.max_size
            tc.keep = getattr(layer_cache, "keep", 0)
            tc.step = getattr(layer_cache, "step", layer_cache.max_size)

            if entries_to_keep <= 0:
                # All buffer content is beyond the trim point — clear
                tc.keys = None
                tc.values = None
                tc._idx = 0
                tc.offset = 0
            elif entries_to_keep < old_size:
                # Reorder to temporal order, keep the oldest entries
                ordered_k = layer_cache._temporal_order(layer_cache.keys)
                ordered_v = layer_cache._temporal_order(layer_cache.values)
                kept_k = ordered_k[:, :, :entries_to_keep, :]
                kept_v = ordered_v[:, :, :entries_to_keep, :]

                if new_offset >= tc.max_size:
                    # Invariant: when offset >= max_size, buffer must be
                    # full (keys.shape[2] == max_size).  Left-pad with
                    # zeros to restore the full buffer.  Zeros represent
                    # positions evicted long ago; _idx = max_size so
                    # _temporal_order returns as-is and _update_in_place
                    # rotates to overwrite zeros first.
                    pad_n = tc.max_size - entries_to_keep
                    pad_k = mx.zeros(
                        (kept_k.shape[0], kept_k.shape[1], pad_n, kept_k.shape[3]),
                        dtype=kept_k.dtype,
                    )
                    pad_v = mx.zeros(
                        (kept_v.shape[0], kept_v.shape[1], pad_n, kept_v.shape[3]),
                        dtype=kept_v.dtype,
                    )
                    tc.keys = mx.concatenate([pad_k, kept_k], axis=2)
                    tc.values = mx.concatenate([pad_v, kept_v], axis=2)
                    tc._idx = tc.max_size
                else:
                    if entries_to_keep < new_offset:
                        # Buffer has fewer entries than offset requires.
                        # This happens when old_offset > max_size (rotating)
                        # and the trim brought new_offset below max_size.
                        # Pad with zeros on the left to maintain the invariant
                        # size() == keys.shape[2], preventing merge crashes.
                        pad_n = new_offset - entries_to_keep
                        pad_k = mx.zeros(
                            (
                                kept_k.shape[0],
                                kept_k.shape[1],
                                pad_n,
                                kept_k.shape[3],
                            ),
                            dtype=kept_k.dtype,
                        )
                        pad_v = mx.zeros(
                            (
                                kept_v.shape[0],
                                kept_v.shape[1],
                                pad_n,
                                kept_v.shape[3],
                            ),
                            dtype=kept_v.dtype,
                        )
                        tc.keys = mx.concatenate([pad_k, kept_k], axis=2)
                        tc.values = mx.concatenate([pad_v, kept_v], axis=2)
                        tc._idx = new_offset
                    else:
                        tc.keys = kept_k
                        tc.values = kept_v
                        tc._idx = entries_to_keep
                eval_targets.extend([tc.keys, tc.values])
            else:
                # No entries removed (trim_by == 0 already handled above,
                # this covers entries_to_keep == old_size edge case)
                tc.keys = layer_cache.keys
                tc.values = layer_cache.values
                tc._idx = layer_cache._idx
            trimmed.append(tc)
        elif (
            hasattr(layer_cache, "offset")
            and hasattr(layer_cache, "keys")
            and not isinstance(layer_cache.keys, (list, tuple))
        ):
            orig_cls = type(layer_cache)
            tc = orig_cls.__new__(orig_cls)
            new_offset = max(layer_cache.offset - trim_by, 0)
            keys = layer_cache.keys
            values = layer_cache.values
            # Slice the arrays down to new_offset rather than just shrinking the
            # offset pointer.  Sharing the original (over-sized) array across
            # requests lets attention paths that read the full underlying
            # buffer (e.g. Gemma 4's KV-shared layers, which read cache.state
            # directly instead of going through update_and_fetch) see stale
            # tokens from the previous owner — issue #384.
            if (
                keys is not None
                and hasattr(keys, "shape")
                and len(keys.shape) >= 3
                and new_offset < keys.shape[-2]
            ):
                tc.keys = keys[..., :new_offset, :]
                tc.values = values[..., :new_offset, :]
                eval_targets.extend([tc.keys, tc.values])
            else:
                tc.keys = keys
                tc.values = values
            tc.offset = new_offset
            # Preserve type-specific attrs (max_size, keep, step, _idx)
            for attr in ("max_size", "keep", "step", "_idx"):
                if hasattr(layer_cache, attr):
                    setattr(tc, attr, getattr(layer_cache, attr))
            trimmed.append(tc)
        else:
            trimmed.append(layer_cache)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _needs_kv_trim(layer: Any) -> bool:
    """Check if a cache layer has oversized KV arrays (duck-typed, no MLX import)."""
    keys = getattr(layer, "keys", None)
    offset = getattr(layer, "offset", None)
    if keys is None or offset is None:
        return False
    if isinstance(keys, (list, tuple)):
        return False  # QuantizedKVCache — skip
    shape = getattr(keys, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return 0 < offset < shape[2]


def _trim_to_offset(cache: list[Any]) -> list[Any]:
    """Trim KV arrays to their actual used size (offset) before storage.

    KV arrays are often pre-allocated larger than needed (e.g. 4096 slots
    when only 100 are used).  This slices them down to ``offset`` and
    evaluates the result so the original large buffer can be freed.
    """
    if not any(_needs_kv_trim(layer) for layer in cache):
        return cache

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    trimmed = []
    eval_targets = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            offset = layer.offset
            if offset <= 0 or offset >= layer.keys.shape[2]:
                trimmed.append(layer)
                continue
            tc = KVCache()
            tc.keys = layer.keys[:, :, :offset, :]
            tc.values = layer.values[:, :, :offset, :]
            tc.offset = offset
            eval_targets.extend([tc.keys, tc.values])
            trimmed.append(tc)
        else:
            trimmed.append(layer)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _compute_model_fingerprint(model: Any) -> str:
    """Compute a fingerprint from model architecture for cache compatibility.

    Used to reject disk-persisted caches created by a different model or
    a different quantisation of the same model.  The fingerprint is a
    short hex digest of (num_layers, hidden_size, vocab_size, num_kv_heads,
    head_dim) — lightweight and deterministic.
    """
    import hashlib as _hashlib

    parts: list[str] = []
    # Walk model.config / model.args / direct attributes
    for cfg_attr in ("config", "args", "model_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None:
            break
    if cfg is None:
        cfg = model  # fallback: attributes on the model itself

    for key in (
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "model_type",
    ):
        val = getattr(cfg, key, None)
        if val is not None:
            parts.append(f"{key}={val}")

    fingerprint = _hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    logger.debug(f"[model_fingerprint] {fingerprint} ({', '.join(parts)})")
    return fingerprint


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        This class is NOT thread-safe. Use external locking if needed.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()
        self._model_fingerprint = _compute_model_fingerprint(model)

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Sorted index of token keys for efficient prefix/supersequence lookup.
        # Tuple lexicographic ordering means a prefix key P is always < any
        # extension of P, so bisect gives O(log N) range scans instead of O(N).
        self._sorted_keys: list[tuple[int, ...]] = []

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        self._current_memory = 0
        self._memory_lock = threading.RLock()

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        # Track the match type from the last fetch() call
        self._last_match_type: str | None = None

        # Optional SSD cold tier (set via set_ssd_tier())
        self._ssd_tier = None

        # Spill delegate (set via set_spill_delegate())
        self._on_spill = None  # Callable[[tuple[int,...], list], Any] | None
        self._on_promote = None  # Callable[[Any], list | None] | None

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}"
        )

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches, prefix matches, supersequence
        matches, and longest-common-prefix (LCP) matches.  Uses a sorted key
        index for O(log N) lookup instead of scanning all entries.

        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            self._last_match_type = "miss"
            return None, tokens

        tokens_key = tuple(tokens)

        # --- O(1) exact match ---
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            self._entries.move_to_end(tokens_key)
            self._stats.hits += 1
            self._stats.tokens_saved += len(tokens)
            self._last_match_type = "exact"
            cache_out = (
                _dequantize_cache(entry.cache)
                if self._config.kv_quantize
                else entry.cache
            )
            return cache_out, []

        # --- O(log N) prefix & supersequence match via sorted index ---
        best_match: _CacheEntry | None = None
        best_length = 0
        best_super: _CacheEntry | None = None

        sorted_keys = self._sorted_keys
        if sorted_keys:
            # Find insertion point for tokens_key in the sorted list.
            # Keys that are prefixes of tokens_key or supersequences will be
            # clustered around this position due to lexicographic ordering.
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan backwards from idx to find cached keys that are PREFIXES
            # of tokens_key (shorter cached sequences).  A prefix P of T
            # satisfies P <= T lexicographically, so P is at idx-1 or earlier.
            for i in range(idx - 1, -1, -1):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len >= len(tokens_key):
                    continue  # Not a prefix (same length or longer)
                # Check if cached_key is a prefix of tokens_key
                if tokens_key[:cached_len] == cached_key:
                    if cached_len > best_length:
                        best_match = self._entries[cached_key]
                        best_length = cached_len
                    # Found best prefix — shorter entries can't be longer
                    break
                # Once we go past the prefix range, stop
                if cached_key[0] != tokens_key[0]:
                    break

            # Scan forward from idx to find cached keys that are SUPERSEQUENCES
            # of tokens_key (longer cached sequences starting with tokens_key).
            for i in range(idx, len(sorted_keys)):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len < len(tokens_key):
                    continue
                # Check if tokens_key is a prefix of cached_key
                if cached_key[: len(tokens_key)] == tokens_key:
                    if best_super is None or cached_len > len(best_super.tokens):
                        best_super = self._entries[cached_key]
                else:
                    # Past the supersequence range
                    break

        # --- Supersequence match handling ---
        if best_super is not None:
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                trimmed_cache = _trim_cache_offset(best_super.cache, excess)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, []
            else:
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                cache_out = (
                    _dequantize_cache(best_super.cache)
                    if self._config.kv_quantize
                    else best_super.cache
                )
                return cache_out, []

        # --- Prefix match ---
        if best_match is not None:
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            self._last_match_type = "prefix"
            cache_out = (
                _dequantize_cache(best_match.cache)
                if self._config.kv_quantize
                else best_match.cache
            )
            return cache_out, remaining

        # --- LCP (Longest Common Prefix) for divergent sequences ---
        # This handles the agentic pattern: same system+context prefix
        # but different final user message.  Use the sorted index to find
        # the nearest neighbor which likely shares the longest prefix.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)
            # Check neighbors around insertion point (they share the most
            # common prefix due to lexicographic ordering).
            for i in (idx - 1, idx):
                if i < 0 or i >= len(sorted_keys):
                    continue
                cached_key = sorted_keys[i]
                if cached_key == tokens_key:
                    continue  # Skip exact (already handled)
                min_len = min(len(cached_key), len(tokens_key))
                if min_len <= best_lcp_length:
                    continue
                # Compute LCP length
                lcp = 0
                for j in range(min_len):
                    if cached_key[j] != tokens_key[j]:
                        break
                    lcp = j + 1
                if lcp > best_lcp_length:
                    best_lcp_entry = self._entries[cached_key]
                    best_lcp_length = lcp
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={len(cached_key)} "
                        f"req_len={len(tokens_key)} lcp={lcp}"
                    )

        if best_lcp_entry is not None and best_lcp_length > 0:
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if has_non_trimmable:
                # Hybrid model (SSM+Attention): SSM state can't be rewound.
                # Block LCP for hybrid models — use think-suffix stripping
                # in the engine layer to get clean PREFIX matches instead.
                logger.debug(
                    "[cache_fetch] LCP skipped: non-trimmable cache layers "
                    "(hybrid model, SSM state can't be rewound)"
                )
            else:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                self._last_match_type = "lcp"
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, remaining

        self._stats.misses += 1
        self._last_match_type = "miss"

        return None, tokens

    def store(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.
            evict_prefixes: If True, evict existing entries whose token
                sequence is a strict prefix of ``tokens``.  Set to False
                when storing prompt+output entries to preserve prompt-only
                entries created by prompt_cache_save (those are the entries
                that future requests will actually match).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        with self._memory_lock:
            tokens_key = tuple(tokens)

            # If already cached, just update LRU order (skip expensive trim/quantize)
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

            # Trim oversized KV arrays to actual used size
            cache = _trim_to_offset(cache)

            # Quantize if enabled and sequence is long enough
            if (
                self._config.kv_quantize
                and len(tokens) >= self._config.kv_min_quantize_tokens
            ):
                cache = _quantize_cache(
                    cache, self._config.kv_bits, self._config.kv_group_size
                )

            # Create entry and estimate memory
            entry = _CacheEntry.create(tokens, cache)

            # Check if single entry exceeds limit
            if entry.memory_bytes > self._max_memory:
                logger.warning(
                    f"Cache entry too large: {entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                    f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
                )
                return False

            # Prefix-subset eviction: remove entries whose token sequence
            # is a strict prefix of the new entry.  Uses sorted index for
            # O(log N + K) lookup instead of O(N) scan.
            if evict_prefixes and self._sorted_keys:
                to_remove = []
                idx = bisect.bisect_left(self._sorted_keys, tokens_key)
                # Scan backwards — prefixes of tokens_key are immediately before idx
                for i in range(idx - 1, -1, -1):
                    key = self._sorted_keys[i]
                    klen = len(key)
                    if klen >= len(tokens_key):
                        continue
                    if tokens_key[:klen] == key:
                        to_remove.append(key)
                    elif key[0] != tokens_key[0]:
                        break
                for key in to_remove:
                    old = self._entries.pop(key)
                    self._current_memory -= old.memory_bytes
                    self._stats.evictions += 1
                    self._remove_from_sorted(key)
                    logger.debug(
                        f"[prefix_evict] removed {len(key)} tokens, "
                        f"freed {old.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                        f"new_entry={len(tokens_key)} tokens"
                    )
                if to_remove:
                    self._stats.entry_count = len(self._entries)
                    self._stats.current_memory_bytes = self._current_memory

            # Evict until we have room
            while (
                self._current_memory + entry.memory_bytes > self._max_memory
                or len(self._entries) >= self._config.max_entries
            ) and self._entries:
                self._evict_lru()

            # Store entry
            self._entries[tokens_key] = entry
            self._current_memory += entry.memory_bytes
            bisect.insort(self._sorted_keys, tokens_key)
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Stored cache: {len(tokens)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )

        return True

    def _remove_from_sorted(self, key: tuple[int, ...]) -> None:
        """Remove a key from the sorted index using bisect for O(log N)."""
        idx = bisect.bisect_left(self._sorted_keys, key)
        if idx < len(self._sorted_keys) and self._sorted_keys[idx] == key:
            self._sorted_keys.pop(idx)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        If a spill delegate is set, the entry is passed to the delegate for storage.
        Otherwise, if an SSD tier is attached, the entry is spilled to disk.
        If neither is configured, the entry is discarded.
        """
        with self._memory_lock:
            if not self._entries:
                return

            # popitem(last=False) removes oldest entry (FIFO order = LRU)
            tokens_key, entry = self._entries.popitem(last=False)
            self._current_memory -= entry.memory_bytes
            self._remove_from_sorted(tokens_key)
            self._stats.evictions += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        # Spill to delegate or legacy SSD tier if available
        if self._on_spill is not None:
            self._on_spill(tokens_key, entry.cache)
        elif self._ssd_tier is not None:
            # legacy path: direct SSD write (used when not wrapped by SSDOffloadedCache)
            self._ssd_tier.enqueue_spill(tokens_key, entry.cache, entry.memory_bytes)

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
            f"{'  (spilled via delegate)' if self._on_spill is not None else ''}"
            f"{'  (spilled to SSD)' if self._ssd_tier is not None and self._on_spill is None else ''}"
        )

    def remove(self, tokens: list[int]) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if entry was found and removed.
        """
        with self._memory_lock:
            tokens_key = tuple(tokens)
            entry = self._entries.pop(tokens_key, None)
            if entry is not None:
                self._current_memory -= entry.memory_bytes
                self._remove_from_sorted(tokens_key)
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._memory_lock:
            self._entries.clear()
            self._sorted_keys.clear()
            self._current_memory = 0
            self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents."""
        with self._memory_lock:
            self._stats = CacheStats(
                max_memory_bytes=self._max_memory,
                current_memory_bytes=self._current_memory,
                entry_count=len(self._entries),
            )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def try_reserve_memory(self, nbytes: int) -> bool:
        """Tentatively reserve cache memory for an upcoming promotion."""
        with self._memory_lock:
            if self._current_memory + nbytes > self._max_memory:
                return False
            self._current_memory += nbytes
            self._stats.current_memory_bytes = self._current_memory
            return True

    def release_reserved_memory(self, nbytes: int) -> None:
        """Release memory previously reserved by try_reserve_memory()."""
        with self._memory_lock:
            self._current_memory = max(0, self._current_memory - nbytes)
            self._stats.current_memory_bytes = self._current_memory

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    def set_ssd_tier(self, ssd_tier) -> None:
        """Attach an SSD cache tier for eviction spilling.

        When set, evicted entries are spilled to SSD instead of discarded.

        Args:
            ssd_tier: An SSDCacheTier instance (or None to disable).
        """
        self._ssd_tier = ssd_tier
        if ssd_tier is not None:
            logger.info("[memory_cache] SSD tier attached for eviction spilling")

    def set_spill_delegate(
        self,
        on_spill: Callable[[tuple[int, ...], list], Any],
        on_promote: Callable[[Any], list | None],
    ) -> None:
        """Register a spill/promote delegate.

        When set, evicted entries call ``on_spill(tokens_key, layers)`` instead
        of writing directly to ``_ssd_tier``.  The ``on_promote`` callback is
        stored for use by higher-level wrappers (e.g. SSDOffloadedCache).

        Args:
            on_spill: Callable[[tuple[int,...], list], Any] called on eviction.
            on_promote: Callable[[Any], list | None] called on cache promotion.
        """
        self._on_spill = on_spill
        self._on_promote = on_promote

    def release(self, handle: Any) -> None:
        """No-op: MemoryAwarePrefixCache has no handle lifecycle."""

    def on_prefill_checkpoint(
        self, request: Any, processed_tokens: int, extracted_cache: list
    ) -> None:
        """No-op: mid-prefill checkpointing is handled by the cache adapter layer."""

    def check_ssd(self, tokens: list[int]) -> dict | None:
        """Check if tokens have an SSD cache hit (without reading data).

        Returns metadata dict with 'match_type' ('exact' or 'prefix') if
        found in SSD tier, None if not found. For prefix matches, the dict
        also includes 'matched_tokens' (the count of tokens the SSD entry
        covers).

        This is a fast synchronous call (SQLite lookup only).
        The actual data read happens via the scheduler handoff.
        """
        if self._ssd_tier is None:
            return None

        tokens_key = tuple(tokens)

        # If already in RAM, no SSD needed
        if tokens_key in self._entries:
            return None

        # Check SSD tier — exact match first, then prefix
        candidate = self._ssd_tier.lookup_ssd(tokens_key)
        if candidate is not None:
            candidate["match_type"] = "exact"
            candidate["matched_tokens"] = len(tokens)
            return candidate

        prefix = self._ssd_tier.lookup_ssd_prefix(tokens_key)
        if prefix is not None:
            prefix["match_type"] = "prefix"
            prefix["matched_tokens"] = prefix["num_tokens"]
            return prefix

        return None

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def save_to_disk(self, cache_dir: str) -> bool:
        """Save all cache entries to disk using mlx_lm's safetensors format.

        Directory layout::

            cache_dir/
              index.json          # token keys + metadata per entry
              entry_0.safetensors # KV arrays for entry 0
              entry_1.safetensors
              ...

        Returns True if at least one entry was saved.
        """
        import json
        import os
        import time as _time

        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        t0 = _time.monotonic()
        os.makedirs(cache_dir, exist_ok=True)

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot save")
            return False

        index = {
            "version": _CACHE_PERSIST_VERSION,
            "model_fingerprint": self._model_fingerprint,
            "num_entries": len(self._entries),
            "total_memory_bytes": self._current_memory,
            "entries": [],
        }

        saved = 0
        for i, (tokens_key, entry) in enumerate(self._entries.items()):
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            try:
                # Dequantize _QuantizedCacheWrapper layers before saving.
                # save_prompt_cache requires .state and .meta_state which
                # the wrapper does not provide; dequantizing restores the
                # original cache types that do.
                persist_cache = (
                    _dequantize_cache(entry.cache)
                    if any(isinstance(c, _QuantizedCacheWrapper) for c in entry.cache)
                    else entry.cache
                )
                save_prompt_cache(
                    entry_path,
                    persist_cache,
                    metadata={"num_tokens": str(len(tokens_key))},
                )
                # Save tokens separately (can be 100K+ ints → binary is smaller)
                tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")
                import array as _array

                arr = _array.array("i", tokens_key)  # 32-bit signed ints
                with open(tokens_path, "wb") as f:
                    arr.tofile(f)

                index["entries"].append(
                    {
                        "index": i,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": entry.memory_bytes,
                    }
                )
                saved += 1
                logger.info(
                    f"[cache_persist] saved entry {i}: "
                    f"{len(tokens_key)} tokens, "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                    f"file={entry_path}"
                )
            except Exception as e:
                logger.warning(f"[cache_persist] failed to save entry {i}: {e}")

        index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] SAVED {saved}/{len(self._entries)} entries "
            f"to {cache_dir} in {dt:.1f}s "
            f"({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return saved > 0

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Returns the number of entries successfully loaded.
        """
        import json
        import os
        import time as _time

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        with open(index_path) as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version != _CACHE_PERSIST_VERSION:
            logger.warning(
                f"[cache_persist] version mismatch: disk={version} "
                f"current={_CACHE_PERSIST_VERSION}, discarding stale cache"
            )
            return 0

        disk_fp = index.get("model_fingerprint", "")
        if disk_fp and disk_fp != self._model_fingerprint:
            logger.warning(
                f"[cache_persist] model fingerprint mismatch: "
                f"disk={disk_fp} current={self._model_fingerprint}, "
                f"discarding incompatible cache"
            )
            return 0

        loaded = 0
        for entry_meta in index.get("entries", []):
            i = entry_meta["index"]
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {i}, skipping")
                continue

            try:
                # Load tokens from binary
                import array as _array

                arr = _array.array("i")
                with open(tokens_path, "rb") as f:
                    arr.fromfile(f, entry_meta["num_tokens"])
                tokens = list(arr)

                # Load KV cache
                cache = load_prompt_cache(entry_path)

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                with self._memory_lock:
                    # Check if it fits
                    if self._current_memory + memory > self._max_memory:
                        logger.info(
                            f"[cache_persist] entry {i} would exceed memory limit "
                            f"({(self._current_memory + memory) / _BYTES_PER_MB:.0f}MB > "
                            f"{self._max_memory / _BYTES_PER_MB:.0f}MB), stopping load"
                        )
                        break

                    tokens_key = tuple(tokens)
                    entry = _CacheEntry(
                        tokens=tokens_key,
                        cache=cache,
                        memory_bytes=memory,
                    )
                    self._entries[tokens_key] = entry
                    self._current_memory += memory
                    bisect.insort(self._sorted_keys, tokens_key)
                    loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {i}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                logger.warning(f"[cache_persist] failed to load entry {i}: {e}")

        with self._memory_lock:
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return loaded
