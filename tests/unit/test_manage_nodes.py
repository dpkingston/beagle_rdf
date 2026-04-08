# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for scripts/manage_nodes.py.

Focuses on the parts that are interesting to break:

* ``cmd_show`` printing of ``config_file_path``
* ``cmd_show --merged`` actually applying the freq_group overlay using the
  same shared helper that the long-poll handler in ``beagle_server.api`` uses
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

# manage_nodes.py is a CLI script in scripts/ — load it as a module by path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "manage_nodes.py"
_spec = importlib.util.spec_from_file_location("manage_nodes", _SCRIPT)
assert _spec is not None and _spec.loader is not None
manage_nodes = importlib.util.module_from_spec(_spec)
sys.modules["manage_nodes"] = manage_nodes
_spec.loader.exec_module(manage_nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: Path) -> sqlite3.Connection:
    """Open a brand-new registry DB using the script's own _open_db (so the
    schema migrations match production)."""
    db = manage_nodes._open_db(str(tmp_path / "registry.db"))
    # node_freq_groups isn't created by manage_nodes' subset schema; create
    # the minimum we need for the overlay tests.
    db.execute("""
        CREATE TABLE IF NOT EXISTS node_freq_groups (
            group_id            TEXT PRIMARY KEY,
            label               TEXT,
            description         TEXT,
            sync_freq_hz        REAL,
            sync_station_id     TEXT,
            sync_station_lat    REAL,
            sync_station_lon    REAL,
            target_channels_json TEXT
        )
    """)
    db.commit()
    return db


def _insert_node(
    db: sqlite3.Connection,
    *,
    node_id: str,
    config_json: str | None = None,
    config_file_path: str | None = None,
    freq_group_id: str | None = None,
) -> None:
    db.execute(
        """
        INSERT INTO nodes
            (node_id, secret_hash, label, registered_at, enabled,
             config_version, config_json, config_file_path, freq_group_id)
        VALUES (?, 'sha256:dummy', ?, ?, 1, 1, ?, ?, ?)
        """,
        (node_id, f"label-{node_id}", time.time(),
         config_json, config_file_path, freq_group_id),
    )
    db.commit()


def _insert_group(
    db: sqlite3.Connection,
    *,
    group_id: str,
    sync_station_id: str = "KISW_99.9",
    sync_freq_hz: float = 99_900_000.0,
    sync_lat: float = 47.6253,
    sync_lon: float = -122.3563,
    target_channels: list[dict] | None = None,
) -> None:
    if target_channels is None:
        target_channels = [
            {"frequency_hz": 460_000_000.0, "label": "T1"},
            {"frequency_hz": 462_500_000.0, "label": "T2"},
        ]
    db.execute(
        """
        INSERT INTO node_freq_groups
            (group_id, label, sync_freq_hz, sync_station_id,
             sync_station_lat, sync_station_lon, target_channels_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (group_id, group_id, sync_freq_hz, sync_station_id,
         sync_lat, sync_lon, json.dumps(target_channels)),
    )
    db.commit()


# ---------------------------------------------------------------------------
# config_file_path display
# ---------------------------------------------------------------------------

def test_show_prints_config_file_path(tmp_path, capsys):
    db = _fresh_db(tmp_path)
    _insert_node(
        db, node_id="n1",
        config_json='{"sync_signal": {}}',
        config_file_path="/etc/beagle/remote_configs/n1.yaml",
    )
    args = argparse.Namespace(node_id="n1", merged=False)
    manage_nodes.cmd_show(db, args)
    out = capsys.readouterr().out
    assert "config_file_path:" in out
    assert "/etc/beagle/remote_configs/n1.yaml" in out


def test_show_prints_inline_when_no_config_file_path(tmp_path, capsys):
    db = _fresh_db(tmp_path)
    _insert_node(db, node_id="n2", config_json='{"sync_signal": {}}')
    args = argparse.Namespace(node_id="n2", merged=False)
    manage_nodes.cmd_show(db, args)
    out = capsys.readouterr().out
    assert "(none - inline / API)" in out


# ---------------------------------------------------------------------------
# --merged overlay
# ---------------------------------------------------------------------------

def test_show_merged_applies_freq_group_overlay(tmp_path, capsys):
    """--merged should print the same overlay-applied JSON the long-poll
    endpoint would serve."""
    db = _fresh_db(tmp_path)
    base_cfg = {
        "sync_signal": {
            "primary_station": {
                "station_id": "OLD",
                "frequency_hz": 88_500_000.0,
                "latitude_deg": 0.0,
                "longitude_deg": 0.0,
            },
            "min_corr_peak": 0.42,
        },
        "target_channels": [{"frequency_hz": 1.0, "label": "OLD"}],
    }
    _insert_group(db, group_id="seattle-fm")
    _insert_node(
        db, node_id="m1",
        config_json=json.dumps(base_cfg),
        freq_group_id="seattle-fm",
    )

    args = argparse.Namespace(node_id="m1", merged=True)
    manage_nodes.cmd_show(db, args)
    out = capsys.readouterr().out

    assert "config_json (merged):" in out
    assert "seattle-fm" in out

    # Extract the JSON block following the "(merged):" header.
    marker = "config_json (merged):"
    idx = out.index(marker)
    # Skip the header line, parse what follows.
    json_text = out[idx + len(marker):].split("\n", 1)[1]
    parsed = json.loads(json_text)
    ps = parsed["sync_signal"]["primary_station"]
    assert ps["station_id"] == "KISW_99.9"
    assert ps["frequency_hz"] == 99_900_000.0
    # min_corr_peak preserved through the merge.
    assert parsed["sync_signal"]["min_corr_peak"] == 0.42
    freqs = sorted(c["frequency_hz"] for c in parsed["target_channels"])
    assert freqs == [460_000_000.0, 462_500_000.0]


def test_show_merged_without_group_is_raw(tmp_path, capsys):
    db = _fresh_db(tmp_path)
    base_cfg = {"sync_signal": {"min_corr_peak": 0.5}}
    _insert_node(db, node_id="m2", config_json=json.dumps(base_cfg))

    args = argparse.Namespace(node_id="m2", merged=True)
    manage_nodes.cmd_show(db, args)
    out = capsys.readouterr().out
    assert "no freq_group assigned" in out
    # The config dump is still printed.
    assert '"min_corr_peak": 0.5' in out


def test_show_default_does_not_apply_overlay(tmp_path, capsys):
    """Without --merged, the raw stored config_json is printed verbatim."""
    db = _fresh_db(tmp_path)
    base_cfg = {"target_channels": [{"frequency_hz": 1.0, "label": "OLD"}]}
    _insert_group(db, group_id="seattle-fm")
    _insert_node(
        db, node_id="m3",
        config_json=json.dumps(base_cfg),
        freq_group_id="seattle-fm",
    )

    args = argparse.Namespace(node_id="m3", merged=False)
    manage_nodes.cmd_show(db, args)
    out = capsys.readouterr().out
    assert "config_json (merged)" not in out
    assert '"frequency_hz": 1.0' in out
    # The group's targets must NOT appear.
    assert "460000000" not in out.replace(",", "").replace(".", "")


def test_show_missing_node_exits(tmp_path):
    db = _fresh_db(tmp_path)
    args = argparse.Namespace(node_id="ghost", merged=False)
    with pytest.raises(SystemExit) as exc_info:
        manage_nodes.cmd_show(db, args)
    assert "ghost" in str(exc_info.value)
