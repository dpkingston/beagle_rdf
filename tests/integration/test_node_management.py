# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration tests for node management API endpoints.

Covers:
- GET  /api/v1/nodes                       list all nodes (admin token)
- GET  /api/v1/nodes/{node_id}             get node detail (admin token)
- POST /api/v1/nodes/register              node self-registration (node secret)
- GET  /api/v1/nodes/{node_id}/config      fetch config with optional long-poll
- PATCH /api/v1/nodes/{node_id}            update node enabled/config (admin token)
- POST /api/v1/events in nodedb auth mode  per-node auth + enabled check

Nodes are pre-seeded into a file-based SQLite DB before the app starts,
mirroring what manage_nodes.py does in production.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time

import numpy as np
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

# ---------------------------------------------------------------------------
# Test credentials
# ---------------------------------------------------------------------------

_ADMIN_TOKEN = "admin-test-token-xyz"

# Node 1: enabled, no config assigned
_NODE_1_ID = "test-node-one"
_NODE_1_SECRET = "test-secret-node-one-aaaa"

# Node 2: enabled, config assigned (config_version=1)
_NODE_2_ID = "test-node-two"
_NODE_2_SECRET = "test-secret-node-two-bbbb"
_NODE_2_CONFIG = {"arbitrary": "config", "version": 2}


def _sha256_hash(plaintext: str) -> str:
    return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(db_path: str, registry_path: str, node_auth: str = "nodedb") -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(
            host="127.0.0.1",
            port=8765,
            auth_token=_ADMIN_TOKEN,
            node_auth=node_auth,
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
def node_registry_path(tmp_path) -> str:
    """Create and seed a file-based registry DB with two test nodes."""
    registry_path = str(tmp_path / "test_registry.db")

    async def _seed() -> None:
        db = await db_module.open_registry_db(registry_path)
        now = time.time()
        await db.execute(
            """
            INSERT INTO nodes
                (node_id, secret_hash, label, registered_at, enabled, config_version, config_json)
            VALUES (?, ?, ?, ?, 1, 0, NULL)
            """,
            (_NODE_1_ID, _sha256_hash(_NODE_1_SECRET), "Test Node One", now),
        )
        await db.execute(
            """
            INSERT INTO nodes
                (node_id, secret_hash, label, registered_at, enabled, config_version, config_json)
            VALUES (?, ?, ?, ?, 1, 1, ?)
            """,
            (
                _NODE_2_ID,
                _sha256_hash(_NODE_2_SECRET),
                "Test Node Two",
                now,
                json.dumps(_NODE_2_CONFIG),
            ),
        )
        await db.commit()
        await db.close()

    asyncio.run(_seed())
    return registry_path


@pytest.fixture
def client(node_registry_path, tmp_path) -> TestClient:
    """TestClient running in nodedb auth mode with admin token configured."""
    db_path = str(tmp_path / "test_data.db")
    config = _make_config(db_path, node_registry_path)
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _admin_hdrs() -> dict:
    return {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


def _node_hdrs(node_id: str, secret: str) -> dict:
    return {"Authorization": f"Bearer {secret}", "X-Node-ID": node_id}


# ---------------------------------------------------------------------------
# GET /api/v1/nodes - list all nodes
# ---------------------------------------------------------------------------

def test_list_nodes_requires_admin_token(client: TestClient) -> None:
    assert client.get("/api/v1/nodes").status_code == 401


def test_list_nodes_wrong_token_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/nodes", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_list_nodes_returns_both_nodes(client: TestClient) -> None:
    resp = client.get("/api/v1/nodes", headers=_admin_hdrs())
    assert resp.status_code == 200
    ids = [n["node_id"] for n in resp.json()]
    assert _NODE_1_ID in ids
    assert _NODE_2_ID in ids


def test_list_nodes_hides_secret_hash(client: TestClient) -> None:
    resp = client.get("/api/v1/nodes", headers=_admin_hdrs())
    assert resp.status_code == 200
    for node in resp.json():
        assert "secret_hash" not in node


# ---------------------------------------------------------------------------
# GET /api/v1/nodes/{node_id} - single node detail
# ---------------------------------------------------------------------------

def test_get_node_ok(client: TestClient) -> None:
    resp = client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs())
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == _NODE_1_ID
    assert data["enabled"] == 1


def test_get_node_hides_secret_hash(client: TestClient) -> None:
    resp = client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs())
    assert "secret_hash" not in resp.json()


def test_get_node_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/nodes/nonexistent-node", headers=_admin_hdrs())
    assert resp.status_code == 404


def test_get_node_requires_admin_token(client: TestClient) -> None:
    assert client.get(f"/api/v1/nodes/{_NODE_1_ID}").status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/nodes/register - node self-registration
# ---------------------------------------------------------------------------

def test_register_returns_pending_when_no_config(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["node_id"] == _NODE_1_ID
    assert data["config"] is None


def test_register_returns_config_when_assigned(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers=_node_hdrs(_NODE_2_ID, _NODE_2_SECRET),
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["node_id"] == _NODE_2_ID
    assert data["config"] == _NODE_2_CONFIG
    assert data["config_version"] == 1


def test_register_updates_last_seen(client: TestClient) -> None:
    before = time.time()
    client.post(
        "/api/v1/nodes/register",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
        json={},
    )
    node = client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs()).json()
    assert node["last_seen_at"] is not None
    assert node["last_seen_at"] >= before


def test_register_wrong_secret_returns_401(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers=_node_hdrs(_NODE_1_ID, "wrong-secret"),
        json={},
    )
    assert resp.status_code == 401


def test_register_unknown_node_returns_403(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers=_node_hdrs("unknown-node-xyz", "any-secret"),
        json={},
    )
    assert resp.status_code == 403


def test_register_missing_node_id_header_returns_401(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers={"Authorization": f"Bearer {_NODE_1_SECRET}"},
        json={},
    )
    assert resp.status_code == 401


def test_register_missing_auth_header_returns_401(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/nodes/register",
        headers={"X-Node-ID": _NODE_1_ID},
        json={},
    )
    assert resp.status_code == 401


def test_register_no_headers_returns_401(client: TestClient) -> None:
    assert client.post("/api/v1/nodes/register", json={}).status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/nodes/{node_id}/config - fetch config (node auth)
# ---------------------------------------------------------------------------

def test_get_config_pending_when_no_config_assigned(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/nodes/{_NODE_1_ID}/config",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["config"] is None
    assert data["node_id"] == _NODE_1_ID


def test_get_config_returns_config_and_version(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}/config",
        headers=_node_hdrs(_NODE_2_ID, _NODE_2_SECRET),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["config"] == _NODE_2_CONFIG
    assert data["config_version"] == 1


def test_get_config_returns_304_on_long_poll_timeout(client: TestClient) -> None:
    """
    Long-polling with since_version equal to the current version returns
    HTTP 304 after the wait window expires.  wait=1 keeps the test quick.
    """
    resp = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}/config",
        headers=_node_hdrs(_NODE_2_ID, _NODE_2_SECRET),
        params={"wait": 1, "since_version": 1},  # current config_version is 1
    )
    assert resp.status_code == 304


def test_get_config_returns_immediately_when_ahead_of_since_version(client: TestClient) -> None:
    """since_version=0 with current version=1 -> returns immediately without waiting."""
    resp = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}/config",
        headers=_node_hdrs(_NODE_2_ID, _NODE_2_SECRET),
        params={"wait": 30, "since_version": 0},
    )
    assert resp.status_code == 200
    assert resp.json()["config_version"] == 1


def test_get_config_wrong_secret_returns_401(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/nodes/{_NODE_1_ID}/config",
        headers=_node_hdrs(_NODE_1_ID, "wrong-secret"),
    )
    assert resp.status_code == 401


def test_get_config_node_id_mismatch_returns_403(client: TestClient) -> None:
    """node_id in the URL path must match the X-Node-ID header."""
    resp = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}/config",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),  # auth as node-1, path says node-2
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /api/v1/nodes/{node_id} - update node (admin)
# ---------------------------------------------------------------------------

def test_patch_disable_node(client: TestClient) -> None:
    resp = client.patch(
        f"/api/v1/nodes/{_NODE_1_ID}",
        headers=_admin_hdrs(),
        json={"enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    # Verify persisted
    node = client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs()).json()
    assert node["enabled"] == 0


def test_patch_enable_after_disable(client: TestClient) -> None:
    client.patch(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs(), json={"enabled": False})
    resp = client.patch(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs(), json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs()).json()["enabled"] == 1


def test_patch_set_config_json_as_dict(client: TestClient) -> None:
    new_cfg = {"key": "value", "num": 42}
    resp = client.patch(
        f"/api/v1/nodes/{_NODE_1_ID}",
        headers=_admin_hdrs(),
        json={"config_json": new_cfg},
    )
    assert resp.status_code == 200
    # Fetch it back via node auth
    config_resp = client.get(
        f"/api/v1/nodes/{_NODE_1_ID}/config",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
    )
    assert config_resp.status_code == 200
    assert config_resp.json()["config"] == new_cfg


def test_patch_config_increments_version(client: TestClient) -> None:
    old_version = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}", headers=_admin_hdrs()
    ).json()["config_version"]

    resp = client.patch(
        f"/api/v1/nodes/{_NODE_2_ID}",
        headers=_admin_hdrs(),
        json={"config_json": {"updated": True}},
    )
    assert resp.status_code == 200
    assert resp.json()["config_version"] == old_version + 1


def test_patch_config_null_clears_config(client: TestClient) -> None:
    """Setting config_json to null should put the node back to pending state."""
    resp = client.patch(
        f"/api/v1/nodes/{_NODE_2_ID}",
        headers=_admin_hdrs(),
        json={"config_json": None},
    )
    assert resp.status_code == 200
    config_resp = client.get(
        f"/api/v1/nodes/{_NODE_2_ID}/config",
        headers=_node_hdrs(_NODE_2_ID, _NODE_2_SECRET),
    )
    assert config_resp.json()["status"] == "pending"


def test_patch_node_not_found_returns_404(client: TestClient) -> None:
    resp = client.patch(
        "/api/v1/nodes/nonexistent",
        headers=_admin_hdrs(),
        json={"enabled": False},
    )
    assert resp.status_code == 404


def test_patch_node_requires_admin_token(client: TestClient) -> None:
    assert client.patch(
        f"/api/v1/nodes/{_NODE_1_ID}", json={"enabled": False}
    ).status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/events - nodedb auth mode
# ---------------------------------------------------------------------------

def _make_test_snippet_b64() -> str:
    rng = np.random.default_rng(0)
    samples = rng.integers(-127, 127, size=640 * 2, dtype=np.int8)
    return base64.b64encode(samples.tobytes()).decode()


def _event_payload(node_id: str) -> dict:
    onset_ns = int(time.time() * 1e9)
    return {
        "schema_version": "1.4",
        "event_id": f"evt-nodedb-{node_id}-{onset_ns}",
        "node_id": node_id,
        "node_location": {
            "latitude_deg": 47.7,
            "longitude_deg": -122.4,
            "altitude_m": 50.0,
            "uncertainty_m": 5.0,
        },
        "channel_frequency_hz": 155_100_000.0,
        "sync_delta_ns": 500_000_000,
        "sync_transmitter": {
            "station_id": "KISW_99.9",
            "frequency_hz": 99_900_000.0,
            "latitude_deg": 47.625,
            "longitude_deg": -122.356,
        },
        "sdr_mode": "freq_hop",
        "pps_anchored": False,
        "event_type": "onset",
        "onset_time_ns": onset_ns,
        "peak_power_db": -30.0,
        "mean_power_db": -35.0,
        "noise_floor_db": -60.0,
        "snr_db": 25.0,
        "sync_corr_peak": 0.9,
        "clock_source": "ntp",
        "clock_uncertainty_ns": 5000,
        "node_software_version": "test",
        "iq_snippet_b64": _make_test_snippet_b64(),
        "channel_sample_rate_hz": 64_000.0,
    }


def test_ingest_event_nodedb_accepted(client: TestClient) -> None:
    payload = _event_payload(_NODE_1_ID)
    resp = client.post(
        "/api/v1/events",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
        json=payload,
    )
    assert resp.status_code == 201
    assert resp.json()["event_id"] == payload["event_id"]


def test_ingest_event_nodedb_disabled_node_rejected(client: TestClient) -> None:
    client.patch(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs(), json={"enabled": False})
    resp = client.post(
        "/api/v1/events",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
        json=_event_payload(_NODE_1_ID),
    )
    assert resp.status_code == 403


def test_ingest_event_nodedb_wrong_secret_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/events",
        headers=_node_hdrs(_NODE_1_ID, "wrong-secret"),
        json=_event_payload(_NODE_1_ID),
    )
    assert resp.status_code == 401


def test_ingest_event_nodedb_unknown_node_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/events",
        headers=_node_hdrs("unknown-node-zz", "any-secret"),
        json=_event_payload("unknown-node-zz"),
    )
    assert resp.status_code == 403


def test_ingest_event_nodedb_missing_headers_rejected(client: TestClient) -> None:
    assert client.post("/api/v1/events", json=_event_payload(_NODE_1_ID)).status_code == 401


def test_ingest_event_updates_last_seen(client: TestClient) -> None:
    before = time.time()
    client.post(
        "/api/v1/events",
        headers=_node_hdrs(_NODE_1_ID, _NODE_1_SECRET),
        json=_event_payload(_NODE_1_ID),
    )
    node = client.get(f"/api/v1/nodes/{_NODE_1_ID}", headers=_admin_hdrs()).json()
    assert node["last_seen_at"] is not None
    assert node["last_seen_at"] >= before
