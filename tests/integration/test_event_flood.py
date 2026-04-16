# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Stress test: verify the server handles event floods correctly.

Sends many groups of events rapidly and verifies:
1. All events are accepted (201)
2. The solver produces fixes for all groups
3. /health remains responsive during and after the flood
4. /map generation doesn't block /health

NOTE: TestClient runs the ASGI app in a background thread with its own
event loop, which prevents true reproduction of event-loop-blocking bugs.
The EventLoopWatchdog (watchdog.py) provides runtime detection of blocked
loops in production.  These tests validate correctness under load, not
the specific executor-vs-synchronous architecture.

Run standalone:
    env/bin/python -m pytest tests/integration/test_event_flood.py -v -s
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

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

_SYNC_TX_LAT = 47.6253
_SYNC_TX_LON = -122.3563
_SYNC_TX_ID = "KISW_99.9"
_TARGET_LAT = 47.660
_TARGET_LON = -122.310

_NODES = {
    "node-a": (47.700, -122.400),
    "node-b": (47.620, -122.220),
    "node-c": (47.540, -122.360),
}


def _make_sync_delta_ns(node_lat: float, node_lon: float) -> int:
    K = 500_000_000
    d_target = haversine_m(_TARGET_LAT, _TARGET_LON, node_lat, node_lon)
    d_sync = haversine_m(_SYNC_TX_LAT, _SYNC_TX_LON, node_lat, node_lon)
    return K + int((d_target - d_sync) / _C_M_S * 1e9)


def _make_event(node_id: str, node_lat: float, node_lon: float,
                channel_hz: float, onset_ns: int) -> dict:
    sync_delta = _make_sync_delta_ns(node_lat, node_lon)
    return {
        "schema_version": "1.4",
        "event_id": str(uuid.uuid4()),
        "node_id": node_id,
        "node_location": {
            "latitude_deg": node_lat,
            "longitude_deg": node_lon,
        },
        "channel_frequency_hz": channel_hz,
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
        "sync_corr_peak": 0.9,
        "node_software_version": "test-flood",
        "iq_snippet_b64": "AAAA",
        "channel_sample_rate_hz": 64000.0,
    }


def _flood_config(delivery_buffer_s: float = 0.15) -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(host="127.0.0.1", port=8765, auth_token="",
                            node_rate_limit_events=0),
        database=DatabaseConfig(path=":memory:", registry_path=":memory:"),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=delivery_buffer_s,
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
            min_xcorr_snr=0.0,  # load test — bypass xcorr SNR check
        ),
        map=MapConfig(output_dir="/tmp/tdoa_flood_test", max_age_s=3600.0),
    )


N_GROUPS = 20


@pytest.fixture()
def flood_client():
    config = _flood_config()
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_event_flood_all_groups_solved(flood_client: TestClient) -> None:
    """
    Send 20 groups of 3-node events on distinct channels.  Verify all
    events are accepted and every group produces a fix.
    """
    client = flood_client
    r = client.get("/health")
    assert r.status_code == 200

    base_channel = 150_000_000.0
    onset_ns = int(time.time() * 1e9)

    for g in range(N_GROUPS):
        channel_hz = base_channel + g * 10_000
        for node_id, (lat, lon) in _NODES.items():
            evt = _make_event(node_id, lat, lon, channel_hz, onset_ns)
            r = client.post("/api/v1/events", json=evt)
            assert r.status_code == 201, f"Event POST failed: {r.text}"

    print(f"\n  Sent {N_GROUPS * len(_NODES)} events across {N_GROUPS} groups")

    # Wait for all delivery buffers to fire and solvers to complete.
    time.sleep(2.0)

    r = client.get("/health")
    assert r.status_code == 200
    health = r.json()
    fix_count = health["fix_count"]
    print(f"  Fixes computed: {fix_count} / {N_GROUPS} groups")
    assert fix_count == N_GROUPS, (
        f"Expected {N_GROUPS} fixes but got {fix_count}"
    )


def test_health_responsive_during_flood(flood_client: TestClient) -> None:
    """
    Fire health probes concurrently with event processing.
    All probes should return 200 promptly.
    """
    client = flood_client

    onset_ns = int(time.time() * 1e9)
    for g in range(N_GROUPS):
        channel_hz = 150_000_000.0 + g * 10_000
        for node_id, (lat, lon) in _NODES.items():
            evt = _make_event(node_id, lat, lon, channel_hz, onset_ns)
            client.post("/api/v1/events", json=evt)

    # Probe health from multiple threads while events are being processed.
    time.sleep(0.2)  # let delivery buffers start firing

    health_times: list[float] = []

    def probe(i: int) -> float:
        t0 = time.monotonic()
        r = client.get("/health")
        elapsed = time.monotonic() - t0
        assert r.status_code == 200, f"probe {i}: status {r.status_code}"
        return elapsed

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(probe, i) for i in range(5)]
        for f in as_completed(futures):
            health_times.append(f.result())

    max_t = max(health_times)
    print(f"\n  Health probe times: {[f'{t*1000:.0f}ms' for t in health_times]}")
    print(f"  Max: {max_t*1000:.0f} ms")
    assert max_t < 2.0, f"Slowest health probe took {max_t:.1f}s"


def test_health_during_map_generation(flood_client: TestClient) -> None:
    """
    Verify /health responds promptly while /map is being generated.
    """
    client = flood_client

    onset_ns = int(time.time() * 1e9)
    for g in range(5):
        channel_hz = 155_000_000.0 + g * 10_000
        for node_id, (lat, lon) in _NODES.items():
            evt = _make_event(node_id, lat, lon, channel_hz, onset_ns)
            client.post("/api/v1/events", json=evt)

    time.sleep(1.0)

    results: dict[str, float] = {}

    def fetch_map():
        t0 = time.monotonic()
        r = client.get("/map")
        results["map_time"] = time.monotonic() - t0
        results["map_status"] = r.status_code

    def fetch_health():
        time.sleep(0.1)
        t0 = time.monotonic()
        r = client.get("/health")
        results["health_time"] = time.monotonic() - t0
        results["health_status"] = r.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(fetch_map)
        pool.submit(fetch_health)

    print(f"\n  /map took {results.get('map_time', 0)*1000:.0f} ms")
    print(f"  /health took {results.get('health_time', 0)*1000:.0f} ms")

    assert results.get("health_status") == 200
    assert results.get("map_status") == 200
    assert results.get("health_time", 999) < 2.0


def _rate_limit_config(max_events: int = 3, window_s: float = 10.0) -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(host="127.0.0.1", port=8765, auth_token="",
                            node_rate_limit_events=max_events,
                            node_rate_limit_window_s=window_s),
        database=DatabaseConfig(path=":memory:", registry_path=":memory:"),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=0.15,
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
            min_xcorr_snr=0.0,  # load test — bypass xcorr SNR check
        ),
        map=MapConfig(output_dir="/tmp/tdoa_flood_test", max_age_s=3600.0),
    )


@pytest.fixture()
def rate_limit_client():
    config = _rate_limit_config(max_events=3, window_s=10.0)
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_rate_limit_rejects_excess_events(rate_limit_client: TestClient) -> None:
    """Per-node rate limit rejects the 4th event within the window."""
    client = rate_limit_client
    node_id = "node-a"
    lat, lon = _NODES[node_id]
    onset_ns = int(time.time() * 1e9)

    # First 3 events should be accepted
    for i in range(3):
        evt = _make_event(node_id, lat, lon, 150_000_000.0 + i * 10_000, onset_ns)
        r = client.post("/api/v1/events", json=evt)
        assert r.status_code == 201, f"Event {i} rejected: {r.text}"

    # 4th event from the same node should be rate-limited
    evt = _make_event(node_id, lat, lon, 150_030_000.0, onset_ns)
    r = client.post("/api/v1/events", json=evt)
    assert r.status_code == 429
    assert "Rate limit exceeded" in r.json()["detail"]


def test_rate_limit_per_node_independent(rate_limit_client: TestClient) -> None:
    """Rate limit is tracked independently per node."""
    client = rate_limit_client
    onset_ns = int(time.time() * 1e9)

    # Send 3 events from node-a (fills its quota)
    for i in range(3):
        evt = _make_event("node-a", 47.700, -122.400, 150_000_000.0 + i * 10_000, onset_ns)
        r = client.post("/api/v1/events", json=evt)
        assert r.status_code == 201

    # node-b should still be able to send
    evt = _make_event("node-b", 47.620, -122.220, 150_000_000.0, onset_ns)
    r = client.post("/api/v1/events", json=evt)
    assert r.status_code == 201

    # node-a should be rate-limited
    evt = _make_event("node-a", 47.700, -122.400, 150_040_000.0, onset_ns)
    r = client.post("/api/v1/events", json=evt)
    assert r.status_code == 429
