"""Smoke test for scripts/find_canonical_m.py CLI.

Runs the script against a tiny mlx-lm model. Marked slow / requires the model
to be present locally — skipped in CI without it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MODEL = os.environ.get("VLLM_MLX_TEST_SMALL_MODEL", "mlx-community/Qwen2.5-0.5B-4bit")
SCRIPT = Path(__file__).parent.parent / "scripts" / "find_canonical_m.py"

requires_model = pytest.mark.skipif(
    not shutil.which("python") or os.environ.get("VLLM_MLX_SKIP_CLI_SMOKE") == "1",
    reason="CLI smoke disabled or python missing",
)


@requires_model
def test_cli_human_report_has_sections(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", MODEL,
         "--n", "256,512", "--batch-sizes", "1"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode in (0, 1), result.stderr
    assert "Per-batch-size canonical bands:" in result.stdout
    assert "Recommended:" in result.stdout or "FAILURE:" in result.stdout


@requires_model
def test_cli_json_output_parses(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", MODEL,
         "--n", "256,512", "--batch-sizes", "1", "--json"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode in (0, 1)
    # JSON block starts at a line beginning with "{"
    json_start = result.stdout.find("\n{")
    assert json_start != -1, result.stdout
    payload = json.loads(result.stdout[json_start:].strip())
    assert "per_batch_bands" in payload
    assert "canonical_intersection" in payload
    assert "recommended_prefill_step_size" in payload or payload["canonical_intersection"] == []
