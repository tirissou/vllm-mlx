# SPDX-License-Identifier: Apache-2.0
"""KV-cache quantization bench helpers.

Public seam for the bench subcommand (and its tests). Re-exports from
vllm_mlx.memory_cache for now; memory_cache.py is deleted in Phase 7.
"""

from vllm_mlx.memory_cache import (  # noqa: F401
    _dequantize_cache,
    _quantize_cache,
    estimate_kv_cache_memory,
)
