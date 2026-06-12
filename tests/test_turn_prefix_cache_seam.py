# SPDX-License-Identifier: Apache-2.0
"""Asserts the trie no longer owns disk concerns."""


def test_no_save_method_on_trie():
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache
    assert not hasattr(TurnPrefixCache, "save")
    assert not hasattr(TurnPrefixCache, "load")
    assert not hasattr(TurnPrefixCache, "_spill_to_ssd")
    assert not hasattr(TurnPrefixCache, "_promote_from_ssd")


def test_ssdref_exported_from_cache_disk_store():
    from vllm_mlx.cache_disk_store import SSDRef
    assert SSDRef is not None
