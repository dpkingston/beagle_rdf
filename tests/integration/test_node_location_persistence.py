# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Tests for persistent node location in the registry.

When a node POSTs a heartbeat (via the long-poll config endpoint), the
server now writes the latitude_deg / longitude_deg from the request body
into new location_lat / location_lon columns on the nodes table.  This
lets the map render markers for known nodes immediately on page load,
even after a server restart wipes the in-memory heartbeats dict.

Covered:
  - update_node_seen() writes lat/lon when supplied; leaves them alone
    when None
  - Long-poll handler persists heartbeat lat/lon on every poll
  - /map/nodes returns location from event > heartbeat > registry,
    in that precedence order
  - /map/nodes sets location_source to indicate where the value came from
  - /map/data emits a node feature for a registered node with persisted
    coordinates even when the in-memory heartbeats dict is empty
    (simulated server restart)
  - The grey "inactive" marker is selected when location_source ==
    "registry" or there's no recent heartbeat
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
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


_ADMIN_TOKEN = "admin-test-token-xyz"
_NODE_ID = "test-loc-node"
_NODE_SECRET = "test-loc-secret-yyyy"
_NODE_LAT = 47.671928
_NODE_LON = -122.404209


def _sha256_hash(plaintext: str) -> str:
    return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()


def _node_hdrs() -> dict:
    return {
        "Authorization": f"Bearer {_NODE_SECRET}",
        "X-Node-ID": _NODE_ID,
    }


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
def registry_path(tmp_path) -> str:
    """Registry DB seeded with one node, no location yet."""
    db_path = str(tmp_path / "test_registry.db")

    async def _seed() -> None:
        db = await db_module.open_registry_db(db_path)
        now = time.time()
        await db.execute(
            """
            INSERT INTO nodes
                (node_id, secret_hash, label, registered_at, enabled,
                 config_version, config_json)
            VALUES (?, ?, ?, ?, 1, 0, NULL)
            """,
            (_NODE_ID, _sha256_hash(_NODE_SECRET), "Loc Test Node", now),
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
# db.update_node_seen unit-style behaviour
# ---------------------------------------------------------------------------

def test_update_node_seen_with_location_writes_lat_lon(registry_path: str) -> None:
    async def _go():
        db = await db_module.open_registry_db(registry_path)
        await db_module.update_node_seen(
            db, _NODE_ID, "127.0.0.1",
            location_lat=_NODE_LAT, location_lon=_NODE_LON,
        )
        row = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row["location_lat"] == pytest.approx(_NODE_LAT)
        assert row["location_lon"] == pytest.approx(_NODE_LON)
        assert row["last_seen_at"] is not None
        await db.close()
    asyncio.run(_go())


def test_update_node_seen_without_location_leaves_existing_alone(
    registry_path: str,
) -> None:
    """If a previously good location is in the registry and the next call
    has location=None, the existing values must NOT be cleared."""
    async def _go():
        db = await db_module.open_registry_db(registry_path)
        # Step 1: write a known good location
        await db_module.update_node_seen(
            db, _NODE_ID, "127.0.0.1",
            location_lat=_NODE_LAT, location_lon=_NODE_LON,
        )
        # Step 2: call again with no location
        await db_module.update_node_seen(db, _NODE_ID, "127.0.0.2")
        row = dict(await db_module.fetch_node(db, _NODE_ID))
        # Original location preserved
        assert row["location_lat"] == pytest.approx(_NODE_LAT)
        assert row["location_lon"] == pytest.approx(_NODE_LON)
        # last_ip was updated
        assert row["last_ip"] == "127.0.0.2"
        await db.close()
    asyncio.run(_go())


def test_update_node_seen_partial_location_treated_as_missing(
    registry_path: str,
) -> None:
    """If only one of (lat, lon) is provided, treat it as missing rather
    than writing a None into the other column."""
    async def _go():
        db = await db_module.open_registry_db(registry_path)
        # Pre-seed a known good
        await db_module.update_node_seen(
            db, _NODE_ID, "127.0.0.1",
            location_lat=_NODE_LAT, location_lon=_NODE_LON,
        )
        # Now call with only lat
        await db_module.update_node_seen(
            db, _NODE_ID, "127.0.0.1",
            location_lat=99.0, location_lon=None,
        )
        row = dict(await db_module.fetch_node(db, _NODE_ID))
        # Original preserved (the function checks both are non-None)
        assert row["location_lat"] == pytest.approx(_NODE_LAT)
        assert row["location_lon"] == pytest.approx(_NODE_LON)
        await db.close()
    asyncio.run(_go())


# ---------------------------------------------------------------------------
# Long-poll handler persists heartbeat lat/lon
# ---------------------------------------------------------------------------

def test_post_config_persists_lat_lon_to_registry(
    client: TestClient, registry_path: str
) -> None:
    """A POST to /api/v1/nodes/{node_id}/config with a heartbeat body
    containing latitude_deg/longitude_deg writes them to the nodes row."""
    resp = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={
            "latitude_deg": _NODE_LAT,
            "longitude_deg": _NODE_LON,
            "sdr_mode": "rspduo",
        },
        params={"wait": 0, "since_version": 0},
    )
    # Status doesn't matter for this assertion (the node has no config
    # assigned, so it'll get "pending"), only that the side effect
    # happened.
    assert resp.status_code == 200

    # Read the row directly to confirm
    async def _check():
        db = await db_module.open_registry_db(registry_path)
        row = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row["location_lat"] == pytest.approx(_NODE_LAT)
        assert row["location_lon"] == pytest.approx(_NODE_LON)
        assert row["last_seen_at"] is not None
        await db.close()
    asyncio.run(_check())


def test_post_config_without_lat_lon_does_not_clobber(
    client: TestClient, registry_path: str
) -> None:
    """First POST with location, then second POST without location -- the
    persisted location must survive."""
    # First POST: with location
    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={"latitude_deg": _NODE_LAT, "longitude_deg": _NODE_LON},
        params={"wait": 0},
    )
    # Second POST: empty body
    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={},
        params={"wait": 0},
    )
    # Registry still has the original location
    async def _check():
        db = await db_module.open_registry_db(registry_path)
        row = dict(await db_module.fetch_node(db, _NODE_ID))
        assert row["location_lat"] == pytest.approx(_NODE_LAT)
        assert row["location_lon"] == pytest.approx(_NODE_LON)
        await db.close()
    asyncio.run(_check())


# ---------------------------------------------------------------------------
# /map/nodes precedence chain: event > heartbeat > registry
# ---------------------------------------------------------------------------

def test_map_nodes_uses_registry_when_no_heartbeat(
    client: TestClient, registry_path: str
) -> None:
    """After persisting a location and then clearing the in-memory
    heartbeats dict, /map/nodes should still report the location from
    the registry, with location_source='registry'."""
    # Push location into the registry
    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={"latitude_deg": _NODE_LAT, "longitude_deg": _NODE_LON},
        params={"wait": 0},
    )
    # Wipe the in-memory heartbeats dict (simulates a server restart
    # without actually restarting the test client)
    client.app.state.heartbeats.clear()

    resp = client.get("/map/nodes")
    assert resp.status_code == 200
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    n = nodes[_NODE_ID]
    assert n["location_lat"] == pytest.approx(_NODE_LAT)
    assert n["location_lon"] == pytest.approx(_NODE_LON)
    assert n["location_source"] == "registry"


def test_map_nodes_prefers_heartbeat_over_registry(
    client: TestClient, registry_path: str
) -> None:
    """If a heartbeat exists in memory with different coordinates than
    the registry, the in-memory value wins (it's fresher)."""
    # Step 1: persist coordinates A
    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={"latitude_deg": _NODE_LAT, "longitude_deg": _NODE_LON},
        params={"wait": 0},
    )
    # Step 2: drop a different value in the in-memory dict
    fresh_lat = _NODE_LAT + 0.5
    fresh_lon = _NODE_LON + 0.5
    client.app.state.heartbeats[_NODE_ID] = {
        "node_id": _NODE_ID,
        "latitude_deg": fresh_lat,
        "longitude_deg": fresh_lon,
        "received_at": time.time(),
    }
    resp = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    n = nodes[_NODE_ID]
    assert n["location_lat"] == pytest.approx(fresh_lat)
    assert n["location_lon"] == pytest.approx(fresh_lon)
    assert n["location_source"] == "heartbeat"


def test_map_nodes_no_position_for_never_seen_node(
    client: TestClient, registry_path: str
) -> None:
    """A node that has never polled in (no heartbeat, no events, no
    persisted location) reports None for location and 'none' for source."""
    resp = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    n = nodes[_NODE_ID]
    assert n["location_lat"] is None
    assert n["location_lon"] is None
    assert n["location_source"] == "none"


# ---------------------------------------------------------------------------
# /map/data emits a registry-fallback feature after restart simulation
# ---------------------------------------------------------------------------

def test_map_data_includes_registry_fallback_node(
    client: TestClient, registry_path: str
) -> None:
    """After persisting a location and clearing the in-memory heartbeats
    dict (server-restart simulation), /map/data should still emit a
    feature_type='node' feature for the node, with
    location_source='registry'."""
    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        headers=_node_hdrs(),
        json={"latitude_deg": _NODE_LAT, "longitude_deg": _NODE_LON},
        params={"wait": 0},
    )
    client.app.state.heartbeats.clear()
    client.app.state.map_geojson_cache.clear()

    resp = client.get("/map/data")
    assert resp.status_code == 200
    features = resp.json()["features"]
    node_features = [
        f for f in features
        if f["properties"]["feature_type"] == "node"
        and f["properties"]["node_id"] == _NODE_ID
    ]
    assert len(node_features) == 1
    f = node_features[0]
    assert f["geometry"]["coordinates"] == [
        pytest.approx(_NODE_LON),
        pytest.approx(_NODE_LAT),
    ]
    assert f["properties"]["location_source"] == "registry"


def test_map_data_skips_node_with_no_known_location(
    client: TestClient, registry_path: str
) -> None:
    """A node that has no events, no heartbeat, and no persisted
    location should NOT emit a feature."""
    client.app.state.map_geojson_cache.clear()
    resp = client.get("/map/data")
    features = resp.json()["features"]
    node_features = [
        f for f in features
        if f["properties"]["feature_type"] == "node"
    ]
    assert node_features == []
