# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-lm's Gemma 4 attention to support BatchKVCache.

Gemma 4 Attention reads cache.offset then calls update_and_fetch, which
uses mx.array.__iadd__ (in-place mutation). For BatchKVCache, cache.offset
is an mx.array (per-sequence), so a naive local reference is silently
mutated before query RoPE runs — giving queries the wrong position.

mlx_lm's implementation uses mx.array(cache.offset) as a snapshot, but
mx.array(x) for an existing mx.array may return a view rather than a copy.
This patch uses `offset + 0` to force materialization of a new array,
guaranteeing independence from subsequent in-place mutations.
"""

import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _snapshot_cache_offset(cache):
    """Return a defensive copy of cache.offset safe from in-place mutation.

    BatchKVCache stores offset as mx.array (per-batch-item).
    mx.array.__iadd__ is in-place, so update_and_fetch mutates the original.
    We return a new computed array to preserve the pre-update value for RoPE.
    """
    if cache is None:
        return 0
    off = cache.offset
    if isinstance(off, int):
        return off
    if isinstance(off, mx.array):
        return off + 0  # new array, same values — immune to in-place mutation
    return off


def patch_gemma4_attention_for_batching() -> bool:
    """Monkey-patch mlx_lm Gemma4 Attention.__call__ to snapshot offset before update.

    Returns True if patch was applied, False if mlx_lm Gemma4 module is not available.
    """
    try:
        from mlx_lm.models.gemma4_text import Attention as Gemma4Attention
        from mlx_lm.models.base import scaled_dot_product_attention
    except (ImportError, TypeError):
        logger.debug("[Gemma4 patch] mlx_lm gemma4_text module not available")
        return False

    if getattr(Gemma4Attention, "_batch_patched", False):
        logger.debug("[Gemma4 patch] Already patched")
        return True

    def _patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv=None,
        offset=None,
    ) -> tuple[mx.array, tuple, Any]:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
            # offset is the snapshotted value returned by the layer that produced shared_kv
        else:
            # Snapshot BEFORE update_and_fetch can mutate cache.offset in-place.
            offset = _snapshot_cache_offset(cache)

            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            keys = self.k_norm(keys)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2]:]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values), offset

    Gemma4Attention.__call__ = _patched_call
    Gemma4Attention._batch_patched = True
    logger.info("[Gemma4 patch] mlx_lm Attention patched for BatchKVCache support")
    return True
