# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch: route prefill attention (L > 1) through flash attention.

mlx-lm's scaled_dot_product_attention dispatches to quantized_scaled_dot_product_attention
whenever the cache has a .bits attribute. That kernel materialises the full
(B, H_q, L, C) score matrix — O(L × C) memory — which causes a persistent RAM
spike during prefill and disables the fused Metal flash-attention kernel.

Fix: for prefill (queries.shape[-2] > 1), dequantize the quantized KV cache
to float16 and delegate to mx.fast.scaled_dot_product_attention. Peak memory
for the dequantised tensors is H_kv × C × head_dim × 2 bytes per layer (~266 MB
at C=130k for Qwen3.5), vs 4–17 GB for the score matrix.

For decode (queries.shape[-2] == 1) the score matrix is 1 × C — negligible —
so the quantized path is kept unchanged.

The decode branch calls _base.quantized_scaled_dot_product_attention via module
lookup (not a closed-over import) so it transparently picks up any already-applied
patches, including the GQA batch fix in mlx_lm_quantized_sdpa.py.
"""

import logging
from typing import Optional

import mlx.core as mx
import mlx_lm.models.base as _base

logger = logging.getLogger(__name__)


def _patched_sdpa(
    queries: mx.array,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        if queries.shape[-2] > 1:
            # Prefill: dequantize this layer's KV and use flash attention.
            # keys / values are (data, scales, biases) tuples from update_and_fetch.
            keys = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits)
            values = mx.dequantize(
                *values, group_size=cache.group_size, bits=cache.bits
            )
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )
        else:
            # Decode: score matrix is B × H_q × 1 × C — quantized matmul is fine.
            # Look up via module so we get any already-applied patches (e.g. GQA fix).
            return _base.quantized_scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=cache.group_size,
                bits=cache.bits,
            )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )


def apply() -> bool:
    """Replace mlx_lm.models.base.scaled_dot_product_attention with _patched_sdpa.

    Returns True if patched, False if already patched.

    All mlx_lm model modules do `from .base import scaled_dot_product_attention`
    at import time, which binds the original function directly in their own
    namespace. Patching _base alone has no effect on those already-bound names.
    We must also walk sys.modules and overwrite the attribute in every
    already-imported mlx_lm.models.* module.
    """
    import sys

    if getattr(_base, "_prefill_flash_sdpa_patched", False):
        logger.debug("[prefill_flash_sdpa patch] Already patched")
        return False

    _base.scaled_dot_product_attention = _patched_sdpa
    setattr(_base, "_prefill_flash_sdpa_patched", True)

    patched_modules = []
    for mod_name, module in sys.modules.items():
        if (
            mod_name.startswith("mlx_lm.models.")
            and mod_name != "mlx_lm.models.base"
            and hasattr(module, "scaled_dot_product_attention")
        ):
            module.scaled_dot_product_attention = _patched_sdpa
            patched_modules.append(mod_name)

    if patched_modules:
        logger.info(
            "[prefill_flash_sdpa patch] Also patched %d model module(s): %s",
            len(patched_modules),
            patched_modules,
        )

    logger.info("[prefill_flash_sdpa patch] Prefill flash-attention patch applied")
    return True
