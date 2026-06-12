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
