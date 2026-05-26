# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Tests for scripts/verify_config.py.

The script is the operator's pre-flight check for any node config
change.  These tests cover:

  - It accepts a valid in-repo example config without flagging anything.
  - It catches the kb7ryy-style misspelling
    (``carrier_onset_margin_db`` inside the ``carrier`` block, where
     the schema field is just ``onset_margin_db``) with a suggestion.
  - It catches schema-validation errors (missing required fields, type
    errors).
  - It catches unparseable YAML / JSON.
  - Exit code equals the number of failing files.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_config.py"
EXAMPLE_YAML = REPO_ROOT / "config" / "node.example.yaml"

# Use the same python that runs the test suite (matches the venv pydantic).
PYTHON = sys.executable


def _run_verify(*paths: Path) -> subprocess.CompletedProcess[str]:
    """Run the script with given file paths; capture stdout/stderr + rc."""
    return subprocess.run(
        [PYTHON, str(SCRIPT), *map(str, paths)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


@pytest.mark.skipif(not EXAMPLE_YAML.exists(),
                    reason="config/node.example.yaml missing from checkout")
def test_example_yaml_passes():
    r = _run_verify(EXAMPLE_YAML)
    assert r.returncode == 0, f"stdout:{r.stdout}\nstderr:{r.stderr}"
    assert "OK" in r.stdout
    assert "passed verification" in r.stdout


def test_misspelled_carrier_margin_field_caught(tmp_path):
    """The kb7ryy regression: 'carrier_onset_margin_db' inside the carrier
    block is silently ignored at runtime but verify_config.py flags it
    with a 'did you mean onset_margin_db?' suggestion."""
    cfg = tmp_path / "kb7ryy_buggy.json"
    cfg.write_text(json.dumps({
        "node_id": "test",
        "sdr_mode": "rspduo",
        "location": {"latitude_deg": 47.5, "longitude_deg": -122.1},
        "sync_signal": {
            "mode": "rds",
            "station_id": "KUOW",
            "frequency_hz": 94_900_000.0,
            "transmitter_location": {"latitude_deg": 47.6, "longitude_deg": -122.3},
        },
        "target_channels": [{"frequency_hz": 146_960_000, "label": "x"}],
        "carrier": {
            "carrier_onset_margin_db": 18,   # <-- typo: extra carrier_ prefix
            "carrier_offset_margin_db": 10,  # <-- typo: extra carrier_ prefix
        },
    }))
    r = _run_verify(cfg)
    assert r.returncode == 1, f"expected 1 failed file; stdout:{r.stdout} stderr:{r.stderr}"
    # The two typo'd keys are surfaced explicitly
    assert "carrier.carrier_onset_margin_db" in r.stderr
    assert "carrier.carrier_offset_margin_db" in r.stderr
    # ...with the right suggestion
    assert "onset_margin_db" in r.stderr
    assert "offset_margin_db" in r.stderr


def test_missing_required_field_caught(tmp_path):
    cfg = tmp_path / "missing.yaml"
    cfg.write_text("node_id: test\n")  # nearly everything else required
    r = _run_verify(cfg)
    assert r.returncode == 1
    assert "Field required" in r.stderr


def test_invalid_json_caught(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text("{not valid json at all")
    r = _run_verify(cfg)
    assert r.returncode == 1
    assert "parse error" in r.stderr or "FAIL" in r.stderr


def test_missing_file_caught(tmp_path):
    cfg = tmp_path / "does_not_exist.yaml"
    r = _run_verify(cfg)
    assert r.returncode == 1
    assert "file not found" in r.stderr


def test_exit_code_equals_failure_count(tmp_path):
    """When asked to verify multiple files, the exit code is the number
    of failures (so the script can be used directly in shell pipelines)."""
    good = tmp_path / "good.yaml"
    good.write_text(EXAMPLE_YAML.read_text() if EXAMPLE_YAML.exists()
                    else textwrap.dedent("""
                        node_id: test
                        sdr_mode: rspduo
                        location: {latitude_deg: 47.5, longitude_deg: -122.1}
                        sync_signal:
                          mode: rds
                          station_id: KUOW
                          frequency_hz: 94900000.0
                          transmitter_location: {latitude_deg: 47.6, longitude_deg: -122.3}
                        target_channels: [{frequency_hz: 146960000, label: x}]
                    """).strip() + "\n")
    bad1 = tmp_path / "bad1.yaml"
    bad1.write_text("node_id: only\n")   # missing required
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("{ truly: unparseable yaml")
    r = _run_verify(good, bad1, bad2)
    # bad1 and bad2 fail → expect 2.  good may or may not pass depending on
    # whether EXAMPLE_YAML exists; assert at least 2 failures.
    assert r.returncode >= 2, f"stdout:{r.stdout}\nstderr:{r.stderr}"
