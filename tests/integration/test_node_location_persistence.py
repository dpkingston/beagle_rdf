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


# ---------------------------------------------------------------------------
# Recency-wins precedence: a stale event row must NOT override a fresh
# heartbeat from a relocated node.  This is the dpk-tdoa2 regression
# from 2026-04-08: the node was physically moved, the new config carried
# the new location, but a yesterday-vintage event row in the database
# was being preferred over today's heartbeat by the old precedence chain.
# ---------------------------------------------------------------------------

def test_relocated_node_fresh_heartbeat_wins_over_stale_event(
    client: TestClient, registry_path: str
) -> None:
    """A node was at OLD_LOC yesterday, emitted events from there, then
    was moved to NEW_LOC and is now polling.  The current heartbeat is
    fresh; the event row in the DB is from yesterday and has OLD_LOC.
    The map must show NEW_LOC."""
    OLD_LAT, OLD_LON = 47.671928, -122.404209  # the colocated-with-tdoa1 spot
    NEW_LAT, NEW_LON = 47.721666, -122.359034  # the new physical location

    # Insert an old event row directly into the events DB carrying OLD_LOC.
    # received_at is yesterday.  This simulates the stale event-row state
    # in production for dpk-tdoa2 on 2026-04-08.
    yesterday = time.time() - 86400.0
    async def _insert_old_event():
        # The events DB is opened lazily by the API; insert via a fresh
        # connection at the same path.
        cfg = client.app.state.config
        db = await db_module.open_db(cfg.database.path)
        await db.execute(
            """
            INSERT INTO events
                (event_id, node_id, received_at, channel_hz, event_type,
                 sync_delta_ns, sync_tx_id, sync_tx_lat, sync_tx_lon,
                 node_lat, node_lon, onset_time_ns, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "stale-event-1", _NODE_ID, yesterday, 443475000.0, "onset",
                100_000, "KUOW_94.9", 47.61576, -122.30919,
                OLD_LAT, OLD_LON, int(yesterday * 1e9), "{}",
            ),
        )
        await db.commit()
        await db.close()
    asyncio.run(_insert_old_event())

    # Now drop a fresh heartbeat into the in-memory dict carrying NEW_LOC.
    fresh_ts = time.time()
    client.app.state.heartbeats[_NODE_ID] = {
        "node_id": _NODE_ID,
        "latitude_deg": NEW_LAT,
        "longitude_deg": NEW_LON,
        "received_at": fresh_ts,
    }

    # /map/nodes should pick the fresh heartbeat
    resp = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    n = nodes[_NODE_ID]
    assert n["location_lat"] == pytest.approx(NEW_LAT)
    assert n["location_lon"] == pytest.approx(NEW_LON)
    assert n["location_source"] == "heartbeat"

    # /map/data should also place the marker at the new location
    client.app.state.map_geojson_cache.clear()
    resp = client.get("/map/data")
    features = resp.json()["features"]
    nf = [
        f for f in features
        if f["properties"]["feature_type"] == "node"
        and f["properties"]["node_id"] == _NODE_ID
    ]
    assert len(nf) == 1
    assert nf[0]["geometry"]["coordinates"] == [
        pytest.approx(NEW_LON), pytest.approx(NEW_LAT),
    ]
    assert nf[0]["properties"]["location_source"] == "heartbeat"


def test_fresh_event_wins_over_stale_heartbeat(
    client: TestClient, registry_path: str
) -> None:
    """The reverse case: a fresh event arrives after the heartbeat goes
    stale.  The event must win because it's newer."""
    OLD_LAT, OLD_LON = 47.6, -122.3
    NEW_LAT, NEW_LON = 47.8, -122.4

    # Stale heartbeat in memory
    client.app.state.heartbeats[_NODE_ID] = {
        "node_id": _NODE_ID,
        "latitude_deg": OLD_LAT,
        "longitude_deg": OLD_LON,
        "received_at": time.time() - 3600.0,
    }

    # Fresh event in the DB
    fresh_ts = time.time()
    async def _insert_fresh_event():
        cfg = client.app.state.config
        db = await db_module.open_db(cfg.database.path)
        await db.execute(
            """
            INSERT INTO events
                (event_id, node_id, received_at, channel_hz, event_type,
                 sync_delta_ns, sync_tx_id, sync_tx_lat, sync_tx_lon,
                 node_lat, node_lon, onset_time_ns, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fresh-event-1", _NODE_ID, fresh_ts, 443475000.0, "onset",
                100_000, "KUOW_94.9", 47.61576, -122.30919,
                NEW_LAT, NEW_LON, int(fresh_ts * 1e9), "{}",
            ),
        )
        await db.commit()
        await db.close()
    asyncio.run(_insert_fresh_event())

    resp = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in resp.json()["nodes"]}
    n = nodes[_NODE_ID]
    assert n["location_lat"] == pytest.approx(NEW_LAT)
    assert n["location_lon"] == pytest.approx(NEW_LON)
    assert n["location_source"] == "event"


# ---------------------------------------------------------------------------
# Direct unit tests of the resolve_node_location helper
# ---------------------------------------------------------------------------

def test_resolve_helper_picks_newest_timestamp() -> None:
    from beagle_server.map_output import resolve_node_location

    t_old = 1000.0
    t_mid = 2000.0
    t_new = 3000.0

    event_row = {"node_lat": 1.0, "node_lon": 1.0, "last_seen_at": t_old}
    heartbeat = {"latitude_deg": 2.0, "longitude_deg": 2.0, "received_at": t_new}
    registry = {"location_lat": 3.0, "location_lon": 3.0, "last_seen_at": t_mid}

    lat, lon, source, ts = resolve_node_location(event_row, heartbeat, registry)
    assert (lat, lon) == (2.0, 2.0)
    assert source == "heartbeat"
    assert ts == t_new


def test_resolve_helper_event_wins_when_newest() -> None:
    from beagle_server.map_output import resolve_node_location
    event_row = {"node_lat": 1.0, "node_lon": 1.0, "last_seen_at": 3000.0}
    heartbeat = {"latitude_deg": 2.0, "longitude_deg": 2.0, "received_at": 1000.0}
    registry = {"location_lat": 3.0, "location_lon": 3.0, "last_seen_at": 2000.0}
    lat, lon, source, _ = resolve_node_location(event_row, heartbeat, registry)
    assert source == "event"
    assert (lat, lon) == (1.0, 1.0)


def test_resolve_helper_registry_wins_when_others_absent() -> None:
    from beagle_server.map_output import resolve_node_location
    registry = {"location_lat": 3.0, "location_lon": 3.0, "last_seen_at": 2000.0}
    lat, lon, source, _ = resolve_node_location(None, None, registry)
    assert source == "registry"
    assert (lat, lon) == (3.0, 3.0)


def test_resolve_helper_skips_candidates_with_partial_data() -> None:
    """A candidate that has a timestamp but no coordinates (or vice versa)
    is ignored, even if it's the newest."""
    from beagle_server.map_output import resolve_node_location
    # heartbeat has the newest timestamp but no coordinates
    event_row = {"node_lat": 1.0, "node_lon": 1.0, "last_seen_at": 1000.0}
    heartbeat = {"latitude_deg": None, "longitude_deg": None, "received_at": 9999.0}
    lat, lon, source, _ = resolve_node_location(event_row, heartbeat, None)
    assert source == "event"
    assert (lat, lon) == (1.0, 1.0)


def test_resolve_helper_returns_none_when_all_empty() -> None:
    from beagle_server.map_output import resolve_node_location
    lat, lon, source, ts = resolve_node_location(None, None, None)
    assert lat is None
    assert lon is None
    assert source == "none"
    assert ts is None
