# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-lm's quantized_scaled_dot_product_attention to support
batched GQA (Group Query Attention) with quantized KV cache.

When n_repeats > 1 (GQA), queries are reshaped from 4D (B, n_q, L, D) to 5D
(B, n_kv, n_repeats, L, D), making scores 5D as well. The attention mask,
however, is 4D (B, 1, 1, seq_k). With batch_size=1 this accidentally works
because NumPy broadcasting prepends a leading 1, making all head-dims 1 and
broadcastable. With batch_size>=2 the batch dimension (2) gets right-aligned
against a head dimension (n_kv_heads) and broadcasting fails:

    (2, 1, 1, seq_k) vs (2, n_kv, n_repeats, 1, seq_k)
    → prepend 1 → (1, 2, 1, 1, seq_k)
    → dim 1: 2 vs n_kv → ValueError

Fix: expand the mask to 5D inside the n_repeats > 1 branch so it becomes
(B, 1, 1, 1, seq_k) and broadcasts correctly against (B, n_kv, n_repeats, 1, seq_k).
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def patch_quantized_sdpa() -> bool:
    """Monkey-patch mlx_lm.models.base.quantized_scaled_dot_product_attention.

    Returns True if patch was applied, False if mlx-lm is not installed.
    """
    try:
        import mlx_lm.models.base as base_module
        from mlx.utils import tree_map
    except ImportError:
        logger.debug("[quantized_sdpa patch] mlx-lm base module not available")
        return False

    if getattr(base_module, "_quantized_sdpa_batch_patched", False):
        logger.debug("[quantized_sdpa patch] Already patched")
        return True

    def _patched_quantized_sdpa(
        queries: mx.array,
        q_keys,
        q_values,
        scale: float,
        mask: Optional[mx.array],
        group_size: int = 64,
        bits: int = 8,
    ) -> mx.array:
        B, n_q_heads, L, D = queries.shape
        n_kv_heads = q_keys[0].shape[-3]
        n_repeats = n_q_heads // n_kv_heads

        queries *= scale

        if n_repeats > 1:
            queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
            q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
            q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)
            # Expand 4D mask (B,1,1,seq_k) → 5D (B,1,1,1,seq_k) so it
            # broadcasts correctly against 5D scores (B,n_kv,n_repeats,1,seq_k).
            if mask is not None and not isinstance(mask, str) and mask.ndim == 4:
                mask = mx.expand_dims(mask, axis=-3)

        scores = mx.quantized_matmul(
            queries, *q_keys, transpose=True, group_size=group_size, bits=bits
        )
        if mask is not None:
            if isinstance(mask, str):
                qL, kL = scores.shape[-2:]
                q_indices = mx.arange(kL - qL, kL)
                k_indices = mx.arange(kL)
                mask = q_indices[:, None] >= k_indices[None]
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores += mask
        scores = mx.softmax(scores, axis=-1, precise=True)
        out = mx.quantized_matmul(
            scores, *q_values, transpose=False, group_size=group_size, bits=bits
        )

        if n_repeats > 1:
            out = mx.reshape(out, (B, n_q_heads, L, D))

        return out

    base_module.quantized_scaled_dot_product_attention = _patched_quantized_sdpa
    setattr(base_module, "_quantized_sdpa_batch_patched", True)
    logger.info("[quantized_sdpa patch] Patched for batched GQA support")
    return True
