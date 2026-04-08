# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Tests for the per-poll config auto-reload feature.

When a node polls GET /api/v1/nodes/{node_id}/config, the server stats the
node's config_file_path and reloads from disk if the file has changed
since config_file_mtime was last recorded.  This eliminates the need for
the operator to manually press a "Reload Configs" button after editing.

Covered:
  - db.maybe_reload_node_config: unit-style tests for each branch
    (skipped, missing, unchanged, updated, parse_error, validation_error,
    error)
  - parse_error and validation_error leave config_file_mtime unchanged
    so the next call retries
  - Long-poll integration: editing the file causes the next poll to
    return the new config and bump config_version
  - GET /map/nodes surfaces the most recent reload status per node
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from beagle_server import db as db_module
from beagle_server.api import create_app
from beagle_server.config import (
    DatabaseConfig,
    MapConfig,
    PairingConfig,
    ServerConfig,
    ServerFullConfig,
    SolverConfig,
)


# ---------------------------------------------------------------------------
# Test credentials and config dicts
# ---------------------------------------------------------------------------

_ADMIN_TOKEN = "admin-test-token-xyz"
_NODE_ID = "test-reload-node"
_NODE_SECRET = "test-reload-secret-zzzz"

# A minimal valid NodeConfig in dict form (matches the fixture used in
# tests/unit/test_remote_config.py).  Stored in the file at the start of
# each test; mutated to verify reload behaviour.
_VALID_CONFIG: dict = {
    "node_id": _NODE_ID,
    "location": {
        "latitude_deg": 47.6,
        "longitude_deg": -122.3,
        "altitude_m": 50.0,
    },
    "sdr_mode": "freq_hop",
    "freq_hop": {},
    "sync_signal": {
        "primary_station": {
            "station_id": "KISW_99.9",
            "frequency_hz": 99_900_000.0,
            "latitude_deg": 47.625,
            "longitude_deg": -122.356,
        }
    },
    "target_channels": [{"frequency_hz": 155_100_000.0, "label": "TEST"}],
}


def _sha256_hash(plaintext: str) -> str:
    return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()


def _node_hdrs() -> dict:
    return {
        "Authorization": f"Bearer {_NODE_SECRET}",
        "X-Node-ID": _NODE_ID,
    }


def _admin_hdrs() -> dict:
    return {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


# ---------------------------------------------------------------------------
# Fixtures: a registry DB seeded with one node whose config came from a file
# ---------------------------------------------------------------------------

def _make_config(db_path: str, registry_path: str) -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(
            host="127.0.0.1",
            port=8765,
            auth_token=_ADMIN_TOKEN,
            node_auth="nodedb",
        ),
        database=DatabaseConfig(path=db_path, registry_path=registry_path),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=0.05,
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
        ),
        map=MapConfig(output_dir="/tmp/tdoa_test_maps", max_age_s=3600.0),
    )


@pytest.fixture
def config_file(tmp_path) -> Path:
    """Write _VALID_CONFIG to a YAML file and return its path."""
    p = tmp_path / "test-reload-node.yaml"
    p.write_text(yaml.safe_dump(_VALID_CONFIG))
    return p


@pytest.fixture
def registry_path(tmp_path, config_file) -> str:
    """Create a registry DB seeded with one node whose config came from
    config_file.  Stores config_file_path + initial config_file_mtime so
    the auto-reload starts in the steady state.
    """
    db_path = str(tmp_path / "test_registry.db")
    initial_json = json.dumps(_VALID_CONFIG)
    initial_mtime = config_file.stat().st_mtime

    async def _seed() -> None:
        db = await db_module.open_registry_db(db_path)
        now = time.time()
        await db.execute(
            """
            INSERT INTO nodes
                (node_id, secret_hash, label, registered_at, enabled,
                 config_version, config_json, config_file_path, config_file_mtime)
            VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)
            """,
            (_NODE_ID, _sha256_hash(_NODE_SECRET), "Reload Test Node", now,
             initial_json, str(config_file), initial_mtime),
        )
        await db.commit()
        await db.close()

    asyncio.run(_seed())
    return db_path


@pytest.fixture
def client(registry_path, tmp_path) -> TestClient:
    db_path = str(tmp_path / "test_data.db")
    config = _make_config(db_path, registry_path)
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: directly invoke maybe_reload_node_config against a registry DB
# ---------------------------------------------------------------------------

async def _open_and_fetch(registry_path_str: str) -> tuple:
    db = await db_module.open_registry_db(registry_path_str)
    row = await db_module.fetch_node(db, _NODE_ID)
    return db, dict(row)


# ---------------------------------------------------------------------------
# db.maybe_reload_node_config - unit-style branches
# ---------------------------------------------------------------------------

def test_reload_unchanged_returns_unchanged(registry_path: str, config_file: Path) -> None:
    """File mtime matches the recorded value -> status unchanged, no DB write."""
    async def _go():
        db, row = await _open_and_fetch(registry_path)
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "unchanged"
        # config_version should not have moved
        row_after = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row_after["config_version"] == row["config_version"]
        await db.close()
    asyncio.run(_go())


def test_reload_picks_up_file_change(registry_path: str, config_file: Path) -> None:
    """Edit the file -> next call returns 'updated' and bumps config_version."""
    async def _go():
        # Edit the file: change the target_channels label.  Bump mtime to
        # ensure the change is detectable even on filesystems with coarse
        # mtime resolution.
        new_cfg = json.loads(json.dumps(_VALID_CONFIG))  # deep copy
        new_cfg["target_channels"][0]["label"] = "EDITED"
        config_file.write_text(yaml.safe_dump(new_cfg))
        future = time.time() + 5.0
        import os
        os.utime(config_file, (future, future))

        db, row = await _open_and_fetch(registry_path)
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "updated"
        assert result["new_version"] == row["config_version"] + 1

        # Verify the row was actually updated
        row_after = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row_after["config_version"] == result["new_version"]
        cfg_after = json.loads(row_after["config_json"])
        assert cfg_after["target_channels"][0]["label"] == "EDITED"
        await db.close()
    asyncio.run(_go())


def test_reload_missing_file_status(registry_path: str, config_file: Path) -> None:
    """Delete the file -> status missing, config_json untouched."""
    async def _go():
        config_file.unlink()
        db, row = await _open_and_fetch(registry_path)
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "missing"
        assert "message" in result
        # Original config preserved
        row_after = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row_after["config_version"] == row["config_version"]
        assert row_after["config_json"] == row["config_json"]
        await db.close()
    asyncio.run(_go())


def test_reload_parse_error_does_not_update_mtime(
    registry_path: str, config_file: Path
) -> None:
    """Broken YAML -> status parse_error AND config_file_mtime is left
    alone so the next call retries the file when the operator fixes it."""
    async def _go():
        config_file.write_text(":::not valid: yaml: at all:::\n  - [")
        future = time.time() + 5.0
        import os
        os.utime(config_file, (future, future))

        db, row = await _open_and_fetch(registry_path)
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "parse_error"
        assert "parse error" in result["message"]
        # mtime untouched
        row_after = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row_after["config_file_mtime"] == row["config_file_mtime"]
        # config_json untouched
        assert row_after["config_json"] == row["config_json"]
        assert row_after["config_version"] == row["config_version"]
        await db.close()
    asyncio.run(_go())


def test_reload_validation_error_does_not_update_mtime(
    registry_path: str, config_file: Path
) -> None:
    """Valid YAML, invalid NodeConfig schema (e.g. missing required field)
    -> status validation_error and mtime is left alone."""
    async def _go():
        bad = {"this": "is valid yaml", "but": "not a NodeConfig"}
        config_file.write_text(yaml.safe_dump(bad))
        future = time.time() + 5.0
        import os
        os.utime(config_file, (future, future))

        db, row = await _open_and_fetch(registry_path)
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "validation_error"
        assert "validation" in result["message"].lower()
        row_after = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row_after["config_file_mtime"] == row["config_file_mtime"]
        assert row_after["config_json"] == row["config_json"]
        await db.close()
    asyncio.run(_go())


def test_reload_skipped_when_no_path() -> None:
    """A node row with no config_file_path -> status skipped, no error."""
    async def _go():
        # Build an in-memory DB with one node and no path
        db = await db_module.open_registry_db(":memory:")
        await db.execute(
            """
            INSERT INTO nodes
                (node_id, secret_hash, label, registered_at, enabled,
                 config_version, config_json)
            VALUES (?, ?, ?, ?, 1, 0, NULL)
            """,
            ("no-path-node", _sha256_hash("x"), "no path", time.time()),
        )
        await db.commit()
        row = dict(await db_module.fetch_node(db, "no-path-node"))
        result = await db_module.maybe_reload_node_config(db, row)
        assert result["status"] == "skipped"
        await db.close()
    asyncio.run(_go())


def test_reload_retries_after_parse_error_is_fixed(
    registry_path: str, config_file: Path
) -> None:
    """Sequence: edit-broken (parse_error) -> edit-good -> next call
    succeeds.  Confirms that NOT updating config_file_mtime on parse
    failure is enough to make the retry work."""
    async def _go():
        # Step 1: edit to broken
        config_file.write_text("not: [valid: yaml")
        future = time.time() + 5.0
        import os
        os.utime(config_file, (future, future))
        db, row = await _open_and_fetch(registry_path)
        r1 = await db_module.maybe_reload_node_config(db, row)
        assert r1["status"] == "parse_error"

        # Step 2: edit to a valid (different) config
        new_cfg = json.loads(json.dumps(_VALID_CONFIG))
        new_cfg["target_channels"][0]["label"] = "FIXED"
        config_file.write_text(yaml.safe_dump(new_cfg))
        future2 = time.time() + 10.0
        os.utime(config_file, (future2, future2))

        # Step 3: next call should succeed (mtime check sees new timestamp)
        row2 = dict(await db_module.fetch_node(db, _NODE_ID))
        r2 = await db_module.maybe_reload_node_config(db, row2)
        assert r2["status"] == "updated"
        row3 = dict(await db_module.fetch_node(db, _NODE_ID))
        cfg = json.loads(row3["config_json"])
        assert cfg["target_channels"][0]["label"] == "FIXED"
        await db.close()
    asyncio.run(_go())


# ---------------------------------------------------------------------------
# Long-poll integration: editing the file makes the next poll return new config
# ---------------------------------------------------------------------------

def test_long_poll_picks_up_file_change(client: TestClient, config_file: Path) -> None:
    """End-to-end: client polls, then file is edited, then client polls again
    and gets the updated config + bumped version."""
    # First poll: we already know about config_version 1
    r1 = client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 0, "since_version": 0},
    )
    assert r1.status_code == 200
    assert r1.json()["config_version"] == 1
    initial = r1.json()["config"]
    assert initial["target_channels"][0]["label"] == "TEST"

    # Edit the file
    new_cfg = json.loads(json.dumps(_VALID_CONFIG))
    new_cfg["target_channels"][0]["label"] = "EDITED-VIA-LONGPOLL"
    config_file.write_text(yaml.safe_dump(new_cfg))
    import os
    future = time.time() + 5.0
    os.utime(config_file, (future, future))

    # Second poll: same since_version, but the auto-reload at the top of
    # the handler should bump the version and return immediately.
    r2 = client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 0, "since_version": 1},
    )
    assert r2.status_code == 200
    assert r2.json()["config_version"] == 2
    assert r2.json()["config"]["target_channels"][0]["label"] == "EDITED-VIA-LONGPOLL"


def test_long_poll_unchanged_returns_304(client: TestClient, config_file: Path) -> None:
    """If the file hasn't changed and we long-poll with the current version,
    we still get 304 (the auto-reload runs but is a no-op)."""
    r = client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 1, "since_version": 1},
    )
    assert r.status_code == 304


def test_long_poll_validation_error_keeps_old_config(
    client: TestClient, config_file: Path
) -> None:
    """If the operator saves a bad file, the next poll does NOT bump the
    version and the node continues to receive the previous good config."""
    bad = {"this": "is valid yaml", "but": "not a NodeConfig"}
    config_file.write_text(yaml.safe_dump(bad))
    import os
    future = time.time() + 5.0
    os.utime(config_file, (future, future))

    r = client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 0, "since_version": 0},
    )
    assert r.status_code == 200
    # Still version 1, original config
    assert r.json()["config_version"] == 1
    assert r.json()["config"]["target_channels"][0]["label"] == "TEST"


# ---------------------------------------------------------------------------
# /map/nodes surfaces the reload status
# ---------------------------------------------------------------------------

def test_map_nodes_surfaces_reload_error(
    client: TestClient, config_file: Path
) -> None:
    """Trigger a validation_error via a poll, then check /map/nodes shows
    config_reload.status == 'validation_error' for the node."""
    # Save a bad file
    bad = {"this": "is valid yaml", "but": "not a NodeConfig"}
    config_file.write_text(yaml.safe_dump(bad))
    import os
    future = time.time() + 5.0
    os.utime(config_file, (future, future))

    # Trigger a poll so the auto-reload runs
    client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 0, "since_version": 0},
    )

    # Now ask /map/nodes
    resp = client.get("/map/nodes")
    assert resp.status_code == 200
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    assert _NODE_ID in nodes
    cr = nodes[_NODE_ID].get("config_reload")
    assert cr is not None
    assert cr["status"] == "validation_error"
    assert "validation" in cr["message"].lower()


def test_map_nodes_no_reload_field_when_no_polls(client: TestClient) -> None:
    """Before any poll runs, /map/nodes returns config_reload=None for
    the node (the in-memory dict is empty)."""
    resp = client.get("/map/nodes")
    assert resp.status_code == 200
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    assert nodes[_NODE_ID]["config_reload"] is None


def test_map_nodes_surfaces_updated_status(
    client: TestClient, config_file: Path
) -> None:
    """A successful reload shows status='updated' in /map/nodes."""
    new_cfg = json.loads(json.dumps(_VALID_CONFIG))
    new_cfg["target_channels"][0]["label"] = "EDITED"
    config_file.write_text(yaml.safe_dump(new_cfg))
    import os
    future = time.time() + 5.0
    os.utime(config_file, (future, future))

    client.get(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        params={"wait": 0, "since_version": 0},
    )

    resp = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    cr = nodes[_NODE_ID].get("config_reload")
    assert cr is not None
    assert cr["status"] == "updated"
