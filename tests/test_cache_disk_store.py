# SPDX-License-Identifier: Apache-2.0
"""Tests for the new CacheDiskStore (vllm_mlx/cache_disk_store.py).

The legacy SSDCacheTier/FilesystemCacheDiskStore tests are removed —
the old subsystem is being deleted. New tests are added in Phase 2.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Rewritten in Phase 2 of the SSD persistence redesign."
)


def test_placeholder():
    assert True
