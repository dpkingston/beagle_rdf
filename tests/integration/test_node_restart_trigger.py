# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration tests for the remote node-restart trigger and the uptime_s
field that lets the GUI confirm a restart took effect.

Server-side mechanism:
  1. POST /api/v1/nodes/{id}/restart  (admin)
       -> sets ``nodes.restart_requested = 1`` in the registry
  2. The next config long-poll from that node observes the flag, includes
     ``restart_requested: true`` in its response, and atomically clears
     the flag.  A waiting long-poll wakes immediately on the transition,
     not just on a config_version advance.
  3. The node's poll thread invokes its on_restart callback (in
     production: os._exit(75)) so systemd brings the service back up.

Uptime mechanism:
  - The node's heartbeat (POSTed as the body of the long-poll request)
    carries ``uptime_s``.  The server stores it on
    ``app.state.heartbeats[node_id]`` and surfaces it through
    ``GET /map/nodes`` so the Node panel can render the live value.
  - After a restart the value drops back to a small number, which is the
    operator's visual confirmation.

Coverage in this file:
  * db.request_node_restart and db.consume_restart_flag (unit-style)
  * The ``restart_requested`` migration is idempotent
  * POST /restart auth gating (admin-required, 404 unknown node)
  * The long-poll response carries ``restart_requested: true`` exactly
    once after a request, then false on subsequent polls
  * The long-poll wakes immediately when restart_requested toggles
  * Heartbeat ``uptime_s`` round-trips into ``GET /map/nodes``
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

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


_ADMIN_TOKEN = "admin-test-token-restart"
_NODE_ID = "test-restart-node"
_NODE_SECRET = "test-restart-secret-zzz"
_OTHER_NODE_ID = "other-restart-node"
_OTHER_NODE_SECRET = "other-restart-secret-zzz"

_VALID_CONFIG: dict = {
    "node_id": _NODE_ID,
    "location": {"latitude_deg": 47.6, "longitude_deg": -122.3},
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


def _node_hdrs(node_id: str = _NODE_ID, secret: str = _NODE_SECRET) -> dict:
    return {"Authorization": f"Bearer {secret}", "X-Node-ID": node_id}


def _admin_hdrs() -> dict:
    return {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


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
    """A registry DB seeded with two nodes (so admin/auth tests can verify
    the per-node scoping of the restart flag)."""
    db_path = str(tmp_path / "test_registry.db")
    config_json = json.dumps(_VALID_CONFIG)

    async def _seed() -> None:
        db = await db_module.open_registry_db(db_path)
        now = time.time()
        await db.execute(
            "INSERT INTO nodes (node_id, secret_hash, label, registered_at, "
            "enabled, config_version, config_json) "
            "VALUES (?, ?, ?, ?, 1, 1, ?)",
            (_NODE_ID, _sha256_hash(_NODE_SECRET), "Restart Test Node",
             now, config_json),
        )
        # Second node so we can verify per-node flag scoping
        other_cfg = dict(_VALID_CONFIG)
        other_cfg["node_id"] = _OTHER_NODE_ID
        await db.execute(
            "INSERT INTO nodes (node_id, secret_hash, label, registered_at, "
            "enabled, config_version, config_json) "
            "VALUES (?, ?, ?, ?, 1, 1, ?)",
            (_OTHER_NODE_ID, _sha256_hash(_OTHER_NODE_SECRET), "Other Node",
             now, json.dumps(other_cfg)),
        )
        await db.commit()
        await db.close()

    asyncio.run(_seed())
    return db_path


@pytest.fixture
def client(registry_path, tmp_path):
    db_path = str(tmp_path / "test_data.db")
    config = _make_config(db_path, registry_path)
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# DB layer: request_node_restart, consume_restart_flag
# ---------------------------------------------------------------------------

def test_open_registry_db_adds_restart_requested_column(registry_path: str) -> None:
    """The schema migration adds ``restart_requested`` with default 0."""
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        async with db.execute("PRAGMA table_info(nodes)") as cur:
            cols = {row[1]: dict(zip([d[0] for d in cur.description], row))
                    for row in await cur.fetchall()}
        assert "restart_requested" in cols
        # Existing rows take the default 0.
        async with db.execute(
            "SELECT restart_requested FROM nodes WHERE node_id = ?", (_NODE_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 0
        await db.close()
    asyncio.run(_go())


def test_request_node_restart_sets_flag(registry_path: str) -> None:
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        ok = await db_module.request_node_restart(db, _NODE_ID)
        assert ok is True
        async with db.execute(
            "SELECT restart_requested FROM nodes WHERE node_id = ?", (_NODE_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 1
        # Other node still has flag clear -- per-node scoping.
        async with db.execute(
            "SELECT restart_requested FROM nodes WHERE node_id = ?",
            (_OTHER_NODE_ID,),
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 0
        await db.close()
    asyncio.run(_go())


def test_request_node_restart_unknown_node_returns_false(registry_path: str) -> None:
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        ok = await db_module.request_node_restart(db, "no-such-node")
        assert ok is False
        await db.close()
    asyncio.run(_go())


def test_consume_restart_flag_returns_true_once_then_false(registry_path: str) -> None:
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        await db_module.request_node_restart(db, _NODE_ID)
        first = await db_module.consume_restart_flag(db, _NODE_ID)
        second = await db_module.consume_restart_flag(db, _NODE_ID)
        assert first is True
        assert second is False
        await db.close()
    asyncio.run(_go())


def test_consume_restart_flag_unknown_node_false(registry_path: str) -> None:
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        assert await db_module.consume_restart_flag(db, "no-such-node") is False
        await db.close()
    asyncio.run(_go())


def test_request_node_restart_idempotent(registry_path: str) -> None:
    """Setting the flag while it's already set is a no-op."""
    async def _go() -> None:
        db = await db_module.open_registry_db(registry_path)
        assert await db_module.request_node_restart(db, _NODE_ID) is True
        assert await db_module.request_node_restart(db, _NODE_ID) is True
        # One consume still clears it; not two.
        assert await db_module.consume_restart_flag(db, _NODE_ID) is True
        assert await db_module.consume_restart_flag(db, _NODE_ID) is False
        await db.close()
    asyncio.run(_go())


# ---------------------------------------------------------------------------
# POST /api/v1/nodes/{node_id}/restart
# ---------------------------------------------------------------------------

def test_post_restart_admin_succeeds_and_sets_flag(
    client: TestClient, registry_path: str,
) -> None:
    r = client.post(
        f"/api/v1/nodes/{_NODE_ID}/restart", headers=_admin_hdrs(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node_id"] == _NODE_ID
    assert body["restart_requested"] is True

    async def _check() -> None:
        db = await db_module.open_registry_db(registry_path)
        async with db.execute(
            "SELECT restart_requested FROM nodes WHERE node_id = ?", (_NODE_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 1
        await db.close()
    asyncio.run(_check())


def test_post_restart_unknown_node_404(client: TestClient) -> None:
    r = client.post(
        "/api/v1/nodes/no-such-node/restart", headers=_admin_hdrs(),
    )
    assert r.status_code == 404


def test_post_restart_unauthenticated_rejected(client: TestClient) -> None:
    """Without admin auth the endpoint must reject."""
    r = client.post(f"/api/v1/nodes/{_NODE_ID}/restart")
    # require_admin returns 401 when no auth header is present
    assert r.status_code in (401, 403)


def test_post_restart_with_node_secret_rejected(client: TestClient) -> None:
    """A node's own secret is NOT admin auth -- reject."""
    r = client.post(
        f"/api/v1/nodes/{_NODE_ID}/restart",
        headers=_node_hdrs(),
    )
    assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Long-poll delivery and atomic flag clear
# ---------------------------------------------------------------------------

def test_long_poll_returns_restart_requested_once(client: TestClient) -> None:
    """After POST /restart, the next config poll reports restart_requested=true,
    and a follow-up poll reports false (atomic clear-on-deliver)."""
    # Trigger restart
    r = client.post(f"/api/v1/nodes/{_NODE_ID}/restart", headers=_admin_hdrs())
    assert r.status_code == 200

    # First poll (no wait): should carry the flag
    r1 = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["restart_requested"] is True

    # Subsequent poll (no wait): flag must now be clear
    r2 = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["restart_requested"] is False


def test_long_poll_no_restart_requested_default_false(client: TestClient) -> None:
    """A normal poll with no pending restart returns restart_requested=false."""
    r = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
    )
    assert r.status_code == 200
    assert r.json()["restart_requested"] is False


def test_long_poll_wakes_on_restart_request(client: TestClient) -> None:
    """A waiting long-poll (since_version=current, wait>0) wakes immediately
    when restart_requested transitions to 1, even though config_version
    has not advanced.  Verifies the long-poll handler checks the flag
    alongside the version comparison.
    """
    import threading

    # Get current config_version so we can request a wait that would
    # otherwise time out.
    r0 = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
    )
    cur_ver = r0.json()["config_version"]

    box: dict = {}

    def _do_poll() -> None:
        # Wait up to 10 s for an update that won't come from version alone.
        r = client.post(
            f"/api/v1/nodes/{_NODE_ID}/config?wait=10&since_version={cur_ver}",
            json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
        )
        box["resp"] = r

    t = threading.Thread(target=_do_poll, daemon=True)
    t.start()

    # Give the server a moment to actually be inside the wait loop, then
    # trigger the restart.  The handler polls fetch_node every ~1 s, so
    # 0.2 s of buffer + the 1 s tick should still complete inside the
    # 10 s window with margin.
    time.sleep(0.3)
    r_admin = client.post(
        f"/api/v1/nodes/{_NODE_ID}/restart", headers=_admin_hdrs(),
    )
    assert r_admin.status_code == 200

    t.join(timeout=8.0)
    assert "resp" in box, "long-poll did not return within timeout"
    assert box["resp"].status_code == 200
    body = box["resp"].json()
    assert body["restart_requested"] is True
    # config_version must NOT have advanced (no spoofed bump)
    assert body["config_version"] == cur_ver


def test_restart_flag_per_node_isolation(client: TestClient) -> None:
    """A restart request for node A does not surface to node B's poll."""
    r = client.post(f"/api/v1/nodes/{_NODE_ID}/restart", headers=_admin_hdrs())
    assert r.status_code == 200

    # OTHER node polls -- must NOT see restart_requested
    r_other = client.post(
        f"/api/v1/nodes/{_OTHER_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"},
        headers=_node_hdrs(_OTHER_NODE_ID, _OTHER_NODE_SECRET),
    )
    assert r_other.status_code == 200
    assert r_other.json()["restart_requested"] is False


# ---------------------------------------------------------------------------
# Heartbeat uptime_s round-trip into /map/nodes
# ---------------------------------------------------------------------------

def test_heartbeat_uptime_round_trips_to_map_nodes(client: TestClient) -> None:
    """A node's heartbeat uptime_s is surfaced in /map/nodes."""
    body = {
        "sdr_mode": "freq_hop",
        "uptime_s": 12345.6,
        "noise_floor_db": -55.0,
        "onset_threshold_db": -42.0,
    }
    r = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config", json=body, headers=_node_hdrs(),
    )
    assert r.status_code == 200

    r_map = client.get("/map/nodes")
    assert r_map.status_code == 200
    nodes = {n["node_id"]: n for n in r_map.json()["nodes"]}
    assert _NODE_ID in nodes
    assert nodes[_NODE_ID]["uptime_s"] == 12345.6


def test_restart_request_and_delivery_logged(
    client: TestClient, caplog,
) -> None:
    """The admin endpoint logs the trigger; the long-poll logs the delivery.

    Together these give an audit trail in journalctl: who triggered the
    restart and when it actually landed at the node.
    """
    import logging as _logging

    with caplog.at_level(_logging.INFO, logger="beagle_server.api"):
        r = client.post(
            f"/api/v1/nodes/{_NODE_ID}/restart", headers=_admin_hdrs(),
        )
        assert r.status_code == 200
        # The request log line must include the node id.
        assert any(
            "Remote restart requested" in rec.message and _NODE_ID in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records]

        caplog.clear()
        r2 = client.post(
            f"/api/v1/nodes/{_NODE_ID}/config",
            json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
        )
        assert r2.status_code == 200
        assert r2.json()["restart_requested"] is True
        # The delivery log line must include the node id.
        assert any(
            "Delivering restart instruction" in rec.message
            and _NODE_ID in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records]


def test_heartbeat_without_uptime_yields_none(client: TestClient) -> None:
    """An older node that doesn't send uptime_s shows None in /map/nodes."""
    r = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop"}, headers=_node_hdrs(),
    )
    assert r.status_code == 200
    r_map = client.get("/map/nodes")
    nodes = {n["node_id"]: n for n in r_map.json()["nodes"]}
    assert nodes[_NODE_ID]["uptime_s"] is None


# ---------------------------------------------------------------------------
# Restart detection from heartbeat uptime drop
# ---------------------------------------------------------------------------

def test_uptime_drop_logged_as_restart_detection(
    client: TestClient, caplog,
) -> None:
    """When a heartbeat reports uptime_s lower than the previous heartbeat
    for the same node, the server logs an INFO line.  This is the
    "node came back" half of the audit trail (paired with the
    ``Delivering restart instruction`` line on the trigger side).
    """
    import logging as _logging

    # First heartbeat -- establishes a prior uptime to compare against.
    r1 = client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"sdr_mode": "freq_hop", "uptime_s": 5000.0},
        headers=_node_hdrs(),
    )
    assert r1.status_code == 200

    # Second heartbeat with lower uptime simulates a restart.  Capture
    # only AFTER the first heartbeat so we don't see noise from setup.
    with caplog.at_level(_logging.INFO, logger="beagle_server.api"):
        r2 = client.post(
            f"/api/v1/nodes/{_NODE_ID}/config",
            json={"sdr_mode": "freq_hop", "uptime_s": 3.2},
            headers=_node_hdrs(),
        )
        assert r2.status_code == 200

    detect_lines = [
        rec.message for rec in caplog.records
        if "restart detected" in rec.message and _NODE_ID in rec.message
    ]
    assert len(detect_lines) == 1, detect_lines
    # The line must include both old and new uptime values for forensics.
    assert "5000" in detect_lines[0]
    assert "3.2" in detect_lines[0]


def test_uptime_monotonic_increase_does_not_log_restart(
    client: TestClient, caplog,
) -> None:
    """Normal uptime growth between heartbeats must NOT trigger the
    restart-detection log line."""
    import logging as _logging

    client.post(
        f"/api/v1/nodes/{_NODE_ID}/config",
        json={"uptime_s": 100.0}, headers=_node_hdrs(),
    )
    with caplog.at_level(_logging.INFO, logger="beagle_server.api"):
        client.post(
            f"/api/v1/nodes/{_NODE_ID}/config",
            json={"uptime_s": 130.0}, headers=_node_hdrs(),
        )

    assert not any(
        "restart detected" in rec.message for rec in caplog.records
    )


def test_first_heartbeat_does_not_log_restart(
    client: TestClient, caplog,
) -> None:
    """The very first heartbeat from a node has no prior to compare to;
    don't spuriously log "restart detected" on initial connect."""
    import logging as _logging

    with caplog.at_level(_logging.INFO, logger="beagle_server.api"):
        client.post(
            f"/api/v1/nodes/{_NODE_ID}/config",
            json={"uptime_s": 1.0}, headers=_node_hdrs(),
        )
    assert not any(
        "restart detected" in rec.message for rec in caplog.records
    )
