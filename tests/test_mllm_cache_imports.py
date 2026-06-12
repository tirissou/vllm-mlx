# SPDX-License-Identifier: Apache-2.0
"""Sanity test: MLLM cache types live in vllm_mlx.mllm_cache."""


def test_mllm_cache_module_exports():
    from vllm_mlx.mllm_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _trim_cache_offset,
    )
    assert MemoryAwarePrefixCache is not None
    assert MemoryCacheConfig is not None
    assert _trim_cache_offset is not None
