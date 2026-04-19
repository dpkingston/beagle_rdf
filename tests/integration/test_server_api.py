# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration tests for the Beagle aggregation server.

Uses FastAPI's TestClient (synchronous) with a temporary in-memory SQLite DB.
The delivery_buffer_s is set to 50 ms - long enough to collect all sequential
POSTs before the timer fires, short enough for fast tests.

Test geometry (same triangle as test_solver.py - known to give < 500 m fix error)
-------------
Three nodes around Seattle, known target at (47.660, -122.310).

    Node A:  47.700, -122.400   (NW)
    Node B:  47.620, -122.220   (E)
    Node C:  47.540, -122.360   (S)

Target:      47.660, -122.310
Sync TX:     47.6253, -122.3563   (KISW)
"""

from __future__ import annotations

import base64
import json
import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from beagle_server.api import create_app
from beagle_server.config import (
    DatabaseConfig,
    MapConfig,
    PairingConfig,
    ServerConfig,
    ServerFullConfig,
    SolverConfig,
)
from beagle_server.tdoa import haversine_m

_C_M_S = 299_792_458.0

# ---------------------------------------------------------------------------
# Synthetic geometry
# ---------------------------------------------------------------------------

_SYNC_TX_LAT = 47.6253
_SYNC_TX_LON = -122.3563
_SYNC_TX_ID = "KISW_99.9"

_TARGET_LAT = 47.660
_TARGET_LON = -122.310
_CHANNEL_HZ = 155_100_000.0

# Same triangle as test_solver.py - known to give < 500 m fix error
_NODES = {
    "node-a": (47.700, -122.400),
    "node-b": (47.620, -122.220),
    "node-c": (47.540, -122.360),
}


# ---------------------------------------------------------------------------
# Synthetic IQ snippet helpers
# ---------------------------------------------------------------------------

_SNIPPET_RATE_HZ = 1_000_000   # 1 MHz encoding rate - +/-1 usec = ~+/-300 m timing precision
_SNIPPET_LEN     = 10_000      # 10 ms at 1 MHz - plenty of headroom for all onset offsets

# Amplitude-modulated carrier sequence used across all nodes.
# Each node receives the same transmission with its content starting at a
# different offset into _CARRIER_SEQ (encoding the inter-node TDOA).
# The carrier has a slowly-varying AM envelope so that the power envelope has
# texture for cross-correlation - pure QPSK (constant power) would give a flat
# power envelope that makes power-envelope xcorr unable to find any lag.
_BASE_ONSET    = _SNIPPET_LEN // 4   # onset at 1/4 - matches real snippet encoder
_RAMP_SAMPLES  = 32                  # PA rise time; same for all nodes
_CARRIER_TOTAL = _SNIPPET_LEN * 4
_rng_carrier = np.random.default_rng(0xCAFE)
_bits_i = _rng_carrier.integers(0, 2, _CARRIER_TOTAL) * 2 - 1
_bits_q = _rng_carrier.integers(0, 2, _CARRIER_TOTAL) * 2 - 1
_qpsk: np.ndarray = (_bits_i + 1j * _bits_q).astype(np.complex64) / np.sqrt(2)
# AM envelope with ~8-sample correlation length so that power-envelope xcorr
# can resolve a 4-sample carrier delay (= 4 usec TDOA at 1 MHz).
# Shorter smoothing -> AM decorrelates over 4 samples -> clear xcorr peak at correct lag.
_rng_am = np.random.default_rng(0xBEEF)
_am_raw = np.convolve(
    np.abs(_rng_am.standard_normal(_CARRIER_TOTAL + 16)),
    np.ones(16) / 16, mode="valid",
)[:_CARRIER_TOTAL]
_am_env = ((_am_raw - _am_raw.min()) / (_am_raw.max() - _am_raw.min()) * 0.6 + 0.4).astype(np.float32)
_CARRIER_SEQ: np.ndarray = _qpsk * _am_env

# Minimum sync_delta across all nodes: set after _make_sync_delta_ns is defined.
_MIN_SYNC_DELTA: int = 0


def _make_node_snippet_b64(sync_delta_ns: int) -> str:
    """
    Generate an IQ snippet with the PA transition anchored at a fixed position.

    All snippets have the same onset (_BASE_ONSET) and ramp (_RAMP_SAMPLES)
    at the same position, matching real derivative-peak anchored snippets.
    The carrier content is identical across nodes -- xcorr measures ~0
    refinement.  The TDOA is carried entirely by sync_delta_ns.
    """
    onset = _BASE_ONSET
    ramp_end = onset + _RAMP_SAMPLES
    n_carrier = _SNIPPET_LEN - ramp_end
    carrier_offset = 0  # same content for all nodes (anchored snippets)

    rng_noise = np.random.default_rng(42)  # fixed seed -> same noise for every node
    iq = np.zeros(_SNIPPET_LEN, dtype=np.complex64)
    iq[:onset] = (
        rng_noise.standard_normal(onset) + 1j * rng_noise.standard_normal(onset)
    ).astype(np.complex64) * 0.01

    # Linear amplitude ramp: same shape for all nodes
    ramp = np.linspace(0.0, 1.0, _RAMP_SAMPLES, dtype=np.float32)
    iq[onset:ramp_end] = _CARRIER_SEQ[carrier_offset : carrier_offset + _RAMP_SAMPLES] * ramp

    # Full-amplitude carrier after the ramp
    iq[ramp_end:] = _CARRIER_SEQ[carrier_offset + _RAMP_SAMPLES : carrier_offset + _RAMP_SAMPLES + n_carrier]

    scale = float(np.max(np.abs(iq))) + 1e-30
    normed = iq / scale
    int8_ri = np.empty(_SNIPPET_LEN * 2, dtype=np.int8)
    int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
    int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
    return base64.b64encode(int8_ri.tobytes()).decode()


def _make_sync_delta_ns(node_lat: float, node_lon: float) -> int:
    """
    Synthetic sync_delta_ns for a given node position.

    sync_delta_n = K + (dist(target, n) - dist(sync, n)) / c
    K is arbitrary; we choose K = 500_000_000 ns.
    """
    K = 500_000_000
    d_target = haversine_m(_TARGET_LAT, _TARGET_LON, node_lat, node_lon)
    d_sync = haversine_m(_SYNC_TX_LAT, _SYNC_TX_LON, node_lat, node_lon)
    return K + int((d_target - d_sync) / _C_M_S * 1e9)


# Set _MIN_SYNC_DELTA to the minimum across all nodes so carrier_delay >= 0.
_MIN_SYNC_DELTA = min(_make_sync_delta_ns(lat, lon) for lat, lon in _NODES.values())



def _make_event_payload(
    node_id: str,
    node_lat: float,
    node_lon: float,
    base_time_ns: int | None = None,
) -> dict:
    sync_delta = _make_sync_delta_ns(node_lat, node_lon)
    # onset_time_ns must reflect the true carrier arrival time at each node.
    #
    # When multiple nodes observe the same transmission, onset_time_ns should
    # differ between nodes only by the propagation delay from the target (usec
    # range).  Using time.time_ns() inside a loop gives different base times
    # for each node (HTTP POST overhead can be 1-5 ms per call), which causes
    # pilot disambiguation to incorrectly round raw_ns to the nearest T_sync.
    #
    # Pass base_time_ns (computed ONCE before the loop) as the common epoch;
    # onset_ns then correctly represents the physical arrival time at each node.
    t0 = base_time_ns if base_time_ns is not None else int(time.time() * 1e9)
    _d_target_m = haversine_m(_TARGET_LAT, _TARGET_LON, node_lat, node_lon)
    onset_ns = t0 + round(_d_target_m / _C_M_S * 1e9)
    return {
        "schema_version": "1.1",
        "event_id": f"evt-{node_id}-{onset_ns}",
        "node_id": node_id,
        "node_location": {
            "latitude_deg": node_lat,
            "longitude_deg": node_lon,
        },
        "channel_frequency_hz": _CHANNEL_HZ,
        "sync_delta_ns": sync_delta,
        "sync_transmitter": {
            "station_id": _SYNC_TX_ID,
            "frequency_hz": 99_900_000,
            "latitude_deg": _SYNC_TX_LAT,
            "longitude_deg": _SYNC_TX_LON,
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
        "node_software_version": "test",
        "iq_snippet_b64": _make_node_snippet_b64(sync_delta),
        "channel_sample_rate_hz": float(_SNIPPET_RATE_HZ),
        # Give the knee finder a narrow window around the known ramp position.
        "transition_start": _BASE_ONSET - 30,
        "transition_end": _BASE_ONSET + _RAMP_SAMPLES + 30,
    }


# ---------------------------------------------------------------------------
# Fixture: test app + client
# ---------------------------------------------------------------------------

def _test_config() -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(host="127.0.0.1", port=8765, auth_token=""),
        database=DatabaseConfig(path=":memory:", registry_path=":memory:"),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=0.05,  # 50 ms: all sequential POSTs arrive before timer fires
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
            # Lower SNR gate: synthetic test signal has modest d1 SNR
            # because the post-ramp carrier has deterministic structure
            # that contributes to the RMS-of-d1 "noise" metric.
            min_xcorr_snr=1.5,
        ),
        map=MapConfig(output_dir="/tmp/tdoa_test_maps", max_age_s=3600.0),
    )


@pytest.fixture()
def client():
    config = _test_config()
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["event_count"] == 0
    assert data["fix_count"] == 0
    assert data["last_fix_age_s"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/events - basic ingestion
# ---------------------------------------------------------------------------

def test_post_event_accepted(client: TestClient) -> None:
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    resp = client.post("/api/v1/events", json=payload)
    assert resp.status_code == 201
    assert resp.json()["event_id"] == payload["event_id"]


def test_post_event_low_corr_rejected(client: TestClient) -> None:
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    payload["sync_corr_peak"] = 0.01   # below min_corr_peak=0.05
    resp = client.post("/api/v1/events", json=payload)
    assert resp.status_code == 422


def test_post_event_appears_in_list(client: TestClient) -> None:
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    client.post("/api/v1/events", json=payload)
    resp = client.get("/api/v1/events")
    assert resp.status_code == 200
    events = resp.json()
    assert any(e["event_id"] == payload["event_id"] for e in events)


def test_post_event_amendment(client: TestClient) -> None:
    """Re-posting the same event_id updates the record."""
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    client.post("/api/v1/events", json=payload)
    payload["sync_delta_ns"] += 1000
    resp = client.post("/api/v1/events", json=payload)
    assert resp.status_code == 201
    resp2 = client.get("/api/v1/events")
    # Should still be only one event with this event_id
    matched = [e for e in resp2.json() if e["event_id"] == payload["event_id"]]
    assert len(matched) == 1
    assert matched[0]["sync_delta_ns"] == payload["sync_delta_ns"]


# ---------------------------------------------------------------------------
# Fix computation - three synthetic nodes
# ---------------------------------------------------------------------------

def test_three_nodes_produce_fix(client: TestClient) -> None:
    """
    POST events from 3 nodes; with delivery_buffer_s=0 the fix fires
    after the third event.  A brief sleep gives the background event loop
    thread time to run the async delivery task before we query.
    """
    for node_id, (lat, lon) in _NODES.items():
        payload = _make_event_payload(node_id, lat, lon)
        resp = client.post("/api/v1/events", json=payload)
        assert resp.status_code == 201

    # delivery_buffer_s=0 -> asyncio task fires immediately in the server thread;
    # give the server event loop time to complete the async work.
    time.sleep(0.2)
    resp = client.get("/api/v1/fixes")
    assert resp.status_code == 200
    fixes = resp.json()
    assert len(fixes) >= 1, "Expected at least one fix"


def test_fix_accuracy_within_500m(client: TestClient) -> None:
    """Fix should be within 500 m of the true target location."""
    t0 = int(time.time() * 1e9)
    for node_id, (lat, lon) in _NODES.items():
        payload = _make_event_payload(node_id, lat, lon, base_time_ns=t0)
        client.post("/api/v1/events", json=payload)

    time.sleep(0.2)
    fixes = client.get("/api/v1/fixes").json()
    assert len(fixes) >= 1

    fix = fixes[0]
    error_m = haversine_m(fix["latitude_deg"], fix["longitude_deg"], _TARGET_LAT, _TARGET_LON)
    assert error_m < 500.0, f"Fix error {error_m:.0f} m > 500 m"


def test_fix_node_count(client: TestClient) -> None:
    t0 = int(time.time() * 1e9)
    for node_id, (lat, lon) in _NODES.items():
        client.post("/api/v1/events", json=_make_event_payload(node_id, lat, lon, base_time_ns=t0))

    time.sleep(0.2)
    fixes = client.get("/api/v1/fixes").json()
    assert fixes[0]["node_count"] == 3


def test_get_fix_by_id(client: TestClient) -> None:
    for node_id, (lat, lon) in _NODES.items():
        client.post("/api/v1/events", json=_make_event_payload(node_id, lat, lon))

    time.sleep(0.2)
    fixes = client.get("/api/v1/fixes").json()
    fix_id = fixes[0]["id"]
    resp = client.get(f"/api/v1/fixes/{fix_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == fix_id


def test_get_fix_by_id_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/fixes/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /health after fix
# ---------------------------------------------------------------------------

def test_health_after_fix(client: TestClient) -> None:
    for node_id, (lat, lon) in _NODES.items():
        client.post("/api/v1/events", json=_make_event_payload(node_id, lat, lon))

    time.sleep(0.2)
    resp = client.get("/health")
    data = resp.json()
    assert data["event_count"] == 3
    assert data["fix_count"] >= 1
    assert data["last_fix_age_s"] is not None


# ---------------------------------------------------------------------------
# /map endpoint
# ---------------------------------------------------------------------------

def test_map_returns_html(client: TestClient) -> None:
    resp = client.get("/map")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Folium renders a full HTML document
    assert "<!doctype html>" in resp.text.lower()


def test_map_with_max_age_zero(client: TestClient) -> None:
    for node_id, (lat, lon) in _NODES.items():
        client.post("/api/v1/events", json=_make_event_payload(node_id, lat, lon))

    resp = client.get("/map?max_age_s=0")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_required_when_configured() -> None:
    config = _test_config()
    config.server.auth_token = "secret-token"
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        payload = _make_event_payload("node-a", *_NODES["node-a"])
        # No auth -> 401
        resp = c.post("/api/v1/events", json=payload)
        assert resp.status_code == 401
        # Wrong token -> 401
        resp = c.post("/api/v1/events", json=payload, headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        # Correct token -> 201
        resp = c.post("/api/v1/events", json=payload, headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 201


def test_get_routes_require_no_auth(client: TestClient) -> None:
    """GET routes should not require auth even when auth is configured."""
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/events").status_code == 200
    assert client.get("/api/v1/fixes").status_code == 200
    assert client.get("/map").status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v1/nodes/snr
# ---------------------------------------------------------------------------

def test_nodes_snr_empty(client: TestClient) -> None:
    """No events -> empty list."""
    resp = client.get("/api/v1/nodes/snr")
    assert resp.status_code == 200
    assert resp.json() == []


def test_nodes_snr_basic(client: TestClient) -> None:
    """Events from two nodes -> two entries with correct structure."""
    for node_id in ("node-a", "node-b"):
        payload = _make_event_payload(node_id, *_NODES[node_id])
        payload["peak_power_db"] = -28.0
        payload["noise_floor_db"] = -55.0
        client.post("/api/v1/events", json=payload)

    resp = client.get("/api/v1/nodes/snr")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    node_ids = {n["node_id"] for n in data}
    assert node_ids == {"node-a", "node-b"}

    for node in data:
        assert "status" in node
        assert node["status"] in ("ok", "marginal", "stale")
        assert "last_event_age_s" in node
        assert len(node["channels"]) == 1
        ch = node["channels"][0]
        assert ch["event_count"] == 1
        assert ch["corr_peak_mean"] == pytest.approx(0.9, abs=0.01)
        # SNR = -28 - (-55) = 27 dB
        assert ch["snr_db_mean"] == pytest.approx(27.0, abs=0.5)


def test_nodes_snr_marginal_status(client: TestClient) -> None:
    """Node with low corr_peak should be flagged marginal."""
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    payload["sync_corr_peak"] = 0.2   # well below marginal_corr_peak default 0.5
    client.post("/api/v1/events", json=payload)

    resp = client.get("/api/v1/nodes/snr")
    assert resp.status_code == 200
    node = resp.json()[0]
    assert node["node_id"] == "node-a"
    assert node["status"] == "marginal"


def test_nodes_snr_no_auth_required(client: TestClient) -> None:
    """SNR endpoint must be readable without authentication."""
    resp = client.get("/api/v1/nodes/snr")
    assert resp.status_code == 200


def test_nodes_snr_snr_null_for_zero_noise_floor(client: TestClient) -> None:
    """Events with noise_floor_db=0 (old schema default) yield null snr_db fields."""
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    payload["noise_floor_db"] = 0.0   # old default - not a real measurement
    client.post("/api/v1/events", json=payload)

    resp = client.get("/api/v1/nodes/snr")
    ch = resp.json()[0]["channels"][0]
    assert ch["snr_db_mean"] is None
    assert ch["snr_db_min"] is None
    assert ch["snr_db_p10"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_accepted(client: TestClient) -> None:
    resp = client.post("/api/v1/heartbeat", json={
        "node_id": "hb-node-1",
        "latitude_deg": 47.65,
        "longitude_deg": -122.35,
        "sdr_mode": "freq_hop",
    })
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_heartbeat_missing_node_id(client: TestClient) -> None:
    resp = client.post("/api/v1/heartbeat", json={
        "latitude_deg": 47.65,
    })
    assert resp.status_code == 422


def test_heartbeat_carries_software_version(client: TestClient) -> None:
    """software_version in heartbeat body is stored and visible in /map/nodes."""
    _seed_node(client, "hb-ver-node")
    client.post("/api/v1/heartbeat", json={
        "node_id": "hb-ver-node",
        "latitude_deg": 47.65,
        "longitude_deg": -122.35,
        "sdr_mode": "rspduo",
        "software_version": "0.2.0+deadbeef",
    })
    resp = client.get("/map/nodes")
    nodes = resp.json()["nodes"]
    matched = [n for n in nodes if n["node_id"] == "hb-ver-node"]
    assert len(matched) == 1
    assert matched[0]["software_version"] == "0.2.0+deadbeef"


def test_heartbeat_without_version_shows_none(client: TestClient) -> None:
    """Nodes that don't send software_version should show None (backward compat)."""
    _seed_node(client, "hb-nover-node")
    client.post("/api/v1/heartbeat", json={
        "node_id": "hb-nover-node",
        "latitude_deg": 47.65,
        "longitude_deg": -122.35,
    })
    resp = client.get("/map/nodes")
    nodes = resp.json()["nodes"]
    matched = [n for n in nodes if n["node_id"] == "hb-nover-node"]
    assert len(matched) == 1
    assert matched[0]["software_version"] is None


def test_heartbeat_no_auth_required() -> None:
    """Heartbeat should work even with auth_token configured."""
    config = _test_config()
    config.server.auth_token = "secret-token"
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        resp = c.post("/api/v1/heartbeat", json={
            "node_id": "hb-node-1",
            "latitude_deg": 47.65,
            "longitude_deg": -122.35,
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /map/nodes - merged node list with heartbeats
# ---------------------------------------------------------------------------

def test_map_nodes_empty(client: TestClient) -> None:
    resp = client.get("/map/nodes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert "server_time" in data


def test_map_nodes_heartbeat_only(client: TestClient) -> None:
    """A node known only via heartbeat appears in the node list."""
    client.post("/api/v1/heartbeat", json={
        "node_id": "hb-only",
        "latitude_deg": 47.65,
        "longitude_deg": -122.35,
        "sdr_mode": "single_sdr",
    })
    resp = client.get("/map/nodes")
    nodes = resp.json()["nodes"]
    assert len(nodes) == 1
    n = nodes[0]
    assert n["node_id"] == "hb-only"
    assert n["location_lat"] == pytest.approx(47.65)
    assert n["location_lon"] == pytest.approx(-122.35)
    assert n["sdr_mode"] == "single_sdr"
    assert n["registered"] is True  # shadow-registered via heartbeat
    assert n["enabled"] == 1
    assert n["heartbeat_age_s"] is not None
    assert n["heartbeat_age_s"] < 5.0  # just posted


def test_map_nodes_event_and_heartbeat(client: TestClient) -> None:
    """Node with both events and heartbeat: position from event, heartbeat_age_s present."""
    # Post an event
    payload = _make_event_payload("node-a", *_NODES["node-a"])
    client.post("/api/v1/events", json=payload)
    # Post a heartbeat from same node
    client.post("/api/v1/heartbeat", json={
        "node_id": "node-a",
        "latitude_deg": 47.70,
        "longitude_deg": -122.40,
    })
    resp = client.get("/map/nodes")
    nodes = resp.json()["nodes"]
    matched = [n for n in nodes if n["node_id"] == "node-a"]
    assert len(matched) == 1
    n = matched[0]
    # Position should come from event data (more authoritative)
    assert n["location_lat"] == pytest.approx(_NODES["node-a"][0])
    assert n["heartbeat_age_s"] is not None


# ---------------------------------------------------------------------------
# 2-node LOP fix (min_nodes=2)
# ---------------------------------------------------------------------------

def test_two_nodes_produce_fix(client: TestClient) -> None:
    """POST events from 2 nodes; with min_nodes=2 a fix (LOP) should be computed."""
    two_nodes = dict(list(_NODES.items())[:2])  # node-a, node-b
    for node_id, (lat, lon) in two_nodes.items():
        payload = _make_event_payload(node_id, lat, lon)
        resp = client.post("/api/v1/events", json=payload)
        assert resp.status_code == 201

    time.sleep(0.2)
    resp = client.get("/api/v1/fixes")
    assert resp.status_code == 200
    fixes = resp.json()
    assert len(fixes) >= 1, "Expected at least one fix from 2 nodes"
    assert fixes[0]["node_count"] == 2


# ---------------------------------------------------------------------------
# Frequency Group API
# ---------------------------------------------------------------------------

_GROUP_BODY = {
    "group_id": "seattle-fm",
    "label": "Seattle FM",
    "description": "Seattle area FM sync",
    "sync_freq_hz": 99_900_000.0,
    "sync_station_id": "KISW_99.9",
    "sync_station_lat": 47.6253,
    "sync_station_lon": -122.3563,
    "target_channels": [
        {"frequency_hz": 460_000_000.0, "label": "Target 460"},
        {"frequency_hz": 462_500_000.0, "label": "Target 462.5"},
    ],
}


def test_group_crud(client: TestClient) -> None:
    """Create, read, update, delete a frequency group."""
    # Create
    resp = client.post("/api/v1/groups", json=_GROUP_BODY)
    assert resp.status_code == 201
    grp = resp.json()
    assert grp["group_id"] == "seattle-fm"
    assert grp["label"] == "Seattle FM"
    assert len(grp["target_channels"]) == 2
    assert grp["member_count"] == 0

    # List
    resp = client.get("/api/v1/groups")
    assert resp.status_code == 200
    groups = resp.json()
    assert len(groups) == 1
    assert groups[0]["group_id"] == "seattle-fm"

    # Read
    resp = client.get("/api/v1/groups/seattle-fm")
    assert resp.status_code == 200
    assert resp.json()["label"] == "Seattle FM"

    # Update
    resp = client.patch("/api/v1/groups/seattle-fm",
                        json={"label": "Seattle FM v2"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "Seattle FM v2"

    # Delete
    resp = client.delete("/api/v1/groups/seattle-fm")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Verify gone
    resp = client.get("/api/v1/groups/seattle-fm")
    assert resp.status_code == 404


def test_group_duplicate(client: TestClient) -> None:
    """Creating a group with an existing ID returns 409."""
    resp = client.post("/api/v1/groups", json=_GROUP_BODY)
    assert resp.status_code == 201
    resp = client.post("/api/v1/groups", json=_GROUP_BODY)
    assert resp.status_code == 409


def _seed_node(client: TestClient, node_id: str) -> None:
    """Insert a minimal node record via the app's registry DB.

    TestClient runs an ASGI app in a background thread with its own event
    loop.  We schedule the async insert on that loop via the portal that
    Starlette's TestClient exposes.
    """
    async def _insert():
        db = client.app.state.registry_db
        await db.execute(
            "INSERT OR IGNORE INTO nodes"
            " (node_id, secret_hash, registered_at, enabled, config_version)"
            " VALUES (?, 'sha256:dummy', ?, 1, 0)",
            (node_id, time.time()),
        )
        await db.commit()

    portal = client.portal  # type: ignore[attr-defined]
    portal.call(_insert)


def test_group_node_assignment(client: TestClient) -> None:
    """Assigning a node to a group updates freq_group_id and bumps config_version."""
    _seed_node(client, "test-node-01")

    # Create a group
    resp = client.post("/api/v1/groups", json=_GROUP_BODY)
    assert resp.status_code == 201

    # Assign node to group
    resp = client.patch("/api/v1/nodes/test-node-01",
                        json={"freq_group_id": "seattle-fm"})
    assert resp.status_code == 200
    node = resp.json()
    assert node["freq_group_id"] == "seattle-fm"
    assert node["config_version"] > 0

    # Group should list the node as a member
    resp = client.get("/api/v1/groups/seattle-fm")
    assert resp.status_code == 200
    grp = resp.json()
    assert "test-node-01" in grp["member_node_ids"]
    assert grp["member_count"] == 1


def test_map_groups_endpoint(client: TestClient) -> None:
    """GET /map/groups returns groups without auth."""
    resp = client.post("/api/v1/groups", json=_GROUP_BODY)
    assert resp.status_code == 201

    resp = client.get("/map/groups")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["groups"]) == 1
    assert data["groups"][0]["group_id"] == "seattle-fm"


def test_map_nodes_includes_freq_group_id(client: TestClient) -> None:
    """GET /map/nodes includes freq_group_id for registered nodes."""
    _seed_node(client, "grp-node-01")
    client.post("/api/v1/groups", json=_GROUP_BODY)
    client.patch("/api/v1/nodes/grp-node-01",
                 json={"freq_group_id": "seattle-fm"})

    resp = client.get("/map/nodes")
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    matched = [n for n in nodes if n["node_id"] == "grp-node-01"]
    assert len(matched) == 1
    assert matched[0]["freq_group_id"] == "seattle-fm"


def test_group_update_bumps_member_config(client: TestClient) -> None:
    """Updating a group's frequency fields bumps config_version of members."""
    _seed_node(client, "bump-node")
    client.post("/api/v1/groups", json=_GROUP_BODY)
    client.patch("/api/v1/nodes/bump-node",
                 json={"freq_group_id": "seattle-fm"})

    # Get initial version
    resp = client.get("/api/v1/nodes/bump-node")
    ver_before = resp.json()["config_version"]

    # Update group frequency
    client.patch("/api/v1/groups/seattle-fm",
                 json={"sync_freq_hz": 100_100_000.0})

    # Node version should have bumped
    resp = client.get("/api/v1/nodes/bump-node")
    ver_after = resp.json()["config_version"]
    assert ver_after > ver_before


def test_get_node_merged_applies_freq_group_overlay(client: TestClient) -> None:
    """GET /api/v1/nodes/{id}?merged=1 returns the same view the long-poll
    handler would serve to the node."""
    # Create a node with a base config that has its own primary station and
    # target channels - the overlay should replace BOTH from the freq group.
    _seed_node(client, "merged-node")
    base_cfg = {
        "sync_signal": {
            "primary_station": {
                "station_id": "OLD_STATION",
                "frequency_hz": 88_500_000.0,
                "latitude_deg": 0.0,
                "longitude_deg": 0.0,
            },
            "min_corr_peak": 0.42,   # should be preserved (not in overlay)
        },
        "target_channels": [
            {"frequency_hz": 999_000_000.0, "label": "OLD"},
        ],
    }
    client.patch("/api/v1/nodes/merged-node",
                 json={"config_json": base_cfg})

    # Create the group and assign the node.
    client.post("/api/v1/groups", json=_GROUP_BODY)
    client.patch("/api/v1/nodes/merged-node",
                 json={"freq_group_id": "seattle-fm"})

    # Raw view: per-node config still has the old values.
    raw = client.get("/api/v1/nodes/merged-node").json()
    raw_cfg = json.loads(raw["config_json"])
    assert raw_cfg["sync_signal"]["primary_station"]["station_id"] == "OLD_STATION"
    assert raw_cfg["target_channels"][0]["frequency_hz"] == 999_000_000.0
    assert "config_merged" not in raw

    # Merged view: overlay applied.
    merged = client.get("/api/v1/nodes/merged-node?merged=1").json()
    assert merged["config_merged"] is True
    merged_cfg = json.loads(merged["config_json"])
    ps = merged_cfg["sync_signal"]["primary_station"]
    assert ps["station_id"] == "KISW_99.9"
    assert ps["frequency_hz"] == 99_900_000.0
    assert ps["latitude_deg"] == 47.6253
    assert ps["longitude_deg"] == -122.3563
    # Sibling fields under sync_signal must survive the overlay merge.
    assert merged_cfg["sync_signal"]["min_corr_peak"] == 0.42
    # target_channels replaced wholesale with the group's plan.
    freqs = sorted(c["frequency_hz"] for c in merged_cfg["target_channels"])
    assert freqs == [460_000_000.0, 462_500_000.0]


def test_get_node_merged_no_group_returns_raw(client: TestClient) -> None:
    """?merged=1 on an ungrouped node returns the raw config (not modified)."""
    _seed_node(client, "ungrouped-node")
    base_cfg = {"sync_signal": {"min_corr_peak": 0.5}}
    client.patch("/api/v1/nodes/ungrouped-node",
                 json={"config_json": base_cfg})

    merged = client.get("/api/v1/nodes/ungrouped-node?merged=1").json()
    assert merged["config_merged"] is True
    cfg = json.loads(merged["config_json"])
    assert cfg == base_cfg


def test_get_node_default_is_raw(client: TestClient) -> None:
    """Without merged=1, the endpoint returns the raw stored config_json."""
    _seed_node(client, "raw-node")
    client.post("/api/v1/groups", json=_GROUP_BODY)
    client.patch("/api/v1/nodes/raw-node",
                 json={"config_json": {"target_channels": [{"frequency_hz": 1.0}]}})
    client.patch("/api/v1/nodes/raw-node",
                 json={"freq_group_id": "seattle-fm"})

    resp = client.get("/api/v1/nodes/raw-node").json()
    assert "config_merged" not in resp
    cfg = json.loads(resp["config_json"])
    # Raw config has the original 1.0 Hz target, not the group's targets.
    assert cfg["target_channels"][0]["frequency_hz"] == 1.0


# ---------------------------------------------------------------------------
# Admin Node Management API
# ---------------------------------------------------------------------------

def test_admin_create_node(client: TestClient) -> None:
    """POST /api/v1/nodes creates a node and returns the secret."""
    resp = client.post("/api/v1/nodes", json={
        "node_id": "new-node-01",
        "label": "New Node 01",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["node_id"] == "new-node-01"
    assert data["label"] == "New Node 01"
    assert "secret" in data
    assert len(data["secret"]) == 64  # token_hex(32) -> 64 hex chars

    # Verify the node exists
    resp = client.get("/api/v1/nodes/new-node-01")
    assert resp.status_code == 200
    node = resp.json()
    assert node["node_id"] == "new-node-01"
    assert node["label"] == "New Node 01"
    assert node["enabled"] == 1
    assert "secret_hash" not in node  # must be stripped


def test_admin_create_node_duplicate(client: TestClient) -> None:
    """Creating a node with an existing ID returns 409."""
    resp = client.post("/api/v1/nodes", json={"node_id": "dup-node"})
    assert resp.status_code == 201
    resp = client.post("/api/v1/nodes", json={"node_id": "dup-node"})
    assert resp.status_code == 409


def test_admin_create_node_missing_id(client: TestClient) -> None:
    """Creating a node without node_id returns 422."""
    resp = client.post("/api/v1/nodes", json={"label": "no id"})
    assert resp.status_code == 422


def test_admin_regen_secret(client: TestClient) -> None:
    """POST regen-secret returns a new secret; old one is invalid."""
    import hashlib

    # Create node and get initial secret
    resp = client.post("/api/v1/nodes", json={"node_id": "regen-node"})
    assert resp.status_code == 201
    secret1 = resp.json()["secret"]

    # Regen secret
    resp = client.post("/api/v1/nodes/regen-node/regen-secret")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "regen-node"
    secret2 = data["secret"]
    assert secret2 != secret1
    assert len(secret2) == 64

    # Verify the new hash is in the DB (via portal)
    async def _check_hash():
        db = client.app.state.registry_db
        cursor = await db.execute(
            "SELECT secret_hash FROM nodes WHERE node_id = ?", ("regen-node",))
        row = await cursor.fetchone()
        return row[0] if row else None

    portal = client.portal  # type: ignore[attr-defined]
    stored_hash = portal.call(_check_hash)
    expected = "sha256:" + hashlib.sha256(secret2.encode()).hexdigest()
    assert stored_hash == expected


def test_regen_secret_not_found(client: TestClient) -> None:
    """Regen-secret for non-existent node returns 404."""
    resp = client.post("/api/v1/nodes/ghost-node/regen-secret")
    assert resp.status_code == 404


def test_patch_node_label(client: TestClient) -> None:
    """PATCH with label updates the node's label."""
    _seed_node(client, "label-node")

    resp = client.patch("/api/v1/nodes/label-node",
                        json={"label": "My Label"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "My Label"

    # Verify via GET
    resp = client.get("/api/v1/nodes/label-node")
    assert resp.status_code == 200
    assert resp.json()["label"] == "My Label"

    # Clear label
    resp = client.patch("/api/v1/nodes/label-node",
                        json={"label": None})
    assert resp.status_code == 200
    resp = client.get("/api/v1/nodes/label-node")
    assert resp.json()["label"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/nodes/{node_id}/config - heartbeat-in-config-poll
# ---------------------------------------------------------------------------

def test_config_poll_post_carries_heartbeat(client: TestClient) -> None:
    """POST to the config endpoint stores heartbeat telemetry on the server."""
    # Create a node via admin endpoint to get a valid secret
    resp = client.post("/api/v1/nodes", json={"node_id": "cfg-hb-node"})
    assert resp.status_code == 201
    secret = resp.json()["secret"]

    node_headers = {
        "Authorization": f"Bearer {secret}",
        "X-Node-ID": "cfg-hb-node",
    }

    # POST config poll with heartbeat telemetry
    resp = client.post(
        "/api/v1/nodes/cfg-hb-node/config",
        json={
            "latitude_deg": 47.65,
            "longitude_deg": -122.35,
            "sdr_mode": "freq_hop",
            "software_version": "0.2.0+abc1234",
            "noise_floor_db": -55.2,
            "onset_threshold_db": -30.0,
            "offset_threshold_db": -40.0,
        },
        headers=node_headers,
    )
    assert resp.status_code == 200

    # Verify heartbeat data appears in /map/nodes
    resp = client.get("/map/nodes")
    nodes = resp.json()["nodes"]
    matched = [n for n in nodes if n["node_id"] == "cfg-hb-node"]
    assert len(matched) == 1
    n = matched[0]
    assert n["noise_floor_db"] == pytest.approx(-55.2)
    assert n["onset_threshold_db"] == pytest.approx(-30.0)
    assert n["offset_threshold_db"] == pytest.approx(-40.0)
    assert n["sdr_mode"] == "freq_hop"
    assert n["software_version"] == "0.2.0+abc1234"
    assert n["heartbeat_age_s"] is not None
    assert n["heartbeat_age_s"] < 5.0


def test_config_poll_get_still_works(client: TestClient) -> None:
    """GET to config endpoint still works (backward compat) without heartbeat data."""
    resp = client.post("/api/v1/nodes", json={"node_id": "cfg-get-node"})
    assert resp.status_code == 201
    secret = resp.json()["secret"]

    node_headers = {
        "Authorization": f"Bearer {secret}",
        "X-Node-ID": "cfg-get-node",
    }

    resp = client.get(
        "/api/v1/nodes/cfg-get-node/config",
        headers=node_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "cfg-get-node"
    assert data["status"] == "pending"  # no config_json assigned yet
