# SPDX-License-Identifier: Apache-2.0
from vllm_mlx.scheduler import SchedulerConfig


def test_new_disk_fields_exist_with_defaults():
    config = SchedulerConfig()
    assert config.kv_cache_disk_dir is None
    assert config.kv_cache_disk_max_bytes is None
    assert config.kv_cache_load_on_startup is True
    assert config.kv_cache_save_on_shutdown is True


def test_legacy_fields_removed():
    config = SchedulerConfig()
    for legacy in ("ssd_cache_dir", "ssd_cache_max_gb", "turn_cache_ssd_gb"):
        assert not hasattr(config, legacy), f"legacy field {legacy} still present"


def test_build_prefix_cache_creates_disk_store(tmp_path):
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import SchedulerConfig, _build_prefix_cache

    config = SchedulerConfig(
        use_turn_cache=True,
        kv_cache_disk_dir=str(tmp_path),
        kv_cache_disk_max_bytes=10_000_000,
        chunked_prefill_tokens=8192,
    )
    bundle = _build_prefix_cache(config, model=MagicMock())
    assert bundle.adapter is not None
    assert bundle.adapter._disk_store is not None
    assert bundle.adapter._disk_store._cache_dir.exists()


def test_build_prefix_cache_no_disk_store_when_dir_none():
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import SchedulerConfig, _build_prefix_cache

    config = SchedulerConfig(
        use_turn_cache=True,
        kv_cache_disk_dir=None,
        chunked_prefill_tokens=8192,
    )
    bundle = _build_prefix_cache(config, model=MagicMock())
    assert bundle.adapter._disk_store is None
