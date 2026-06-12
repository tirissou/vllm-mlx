# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", *args],
        capture_output=True, text=True,
    )


def test_serve_help_lists_new_flags():
    result = _run_cli("serve", "--help")
    assert "--kv-cache-disk-dir" in result.stdout
    assert "--kv-cache-disk-max-bytes" in result.stdout
    assert "--kv-cache-load-on-startup" in result.stdout
    assert "--kv-cache-save-on-shutdown" in result.stdout


def test_legacy_ssd_flag_returns_migration_error():
    result = _run_cli("serve", "--ssd-cache-dir", "/tmp/foo", "--model", "x")
    assert result.returncode != 0
    assert "--kv-cache-disk-dir" in result.stderr  # migration hint.
