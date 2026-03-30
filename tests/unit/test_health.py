# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for utils/health.py."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from beagle_node.utils.health import HealthServer, HealthState


@pytest.fixture
def state():
    return HealthState(node_id="test-node")


# ---------------------------------------------------------------------------
# HealthState
# ---------------------------------------------------------------------------

def test_initial_snapshot_keys(state):
    snap = state.snapshot()
    for key in ("status", "node_id", "uptime_s", "last_event_age_s",
                "events_submitted", "events_dropped", "queue_depth",
                "crystal_correction", "sdr_overflows"):
        assert key in snap, f"Missing key: {key}"


def test_initial_status_is_starting(state):
    assert state.snapshot()["status"] == "starting"


def test_node_id_in_snapshot(state):
    assert state.snapshot()["node_id"] == "test-node"


def test_no_event_age_before_first_event(state):
    assert state.snapshot()["last_event_age_s"] is None


def test_record_event_sets_age(state):
    state.record_event()
    age = state.snapshot()["last_event_age_s"]
    assert age is not None
    assert age >= 0.0


def test_update_fields(state):
    state.update(
        events_submitted=10,
        events_dropped=2,
        queue_depth=5,
        crystal_correction=1.000050,
        sdr_overflows=1,
    )
    snap = state.snapshot()
    assert snap["events_submitted"] == 10
    assert snap["events_dropped"] == 2
    assert snap["queue_depth"] == 5
    assert abs(snap["crystal_correction"] - 1.000050) < 1e-9
    assert snap["sdr_overflows"] == 1


def test_degraded_when_events_dropped(state):
    # Fake enough uptime to leave "starting"
    state.start_time -= 40.0
    state.update(events_dropped=1, sync_event_count=1)
    snap = state.snapshot()
    assert snap["status"] == "degraded"
    assert any("dropped" in r for r in snap["degraded_reasons"])


def test_degraded_when_no_sync_events(state):
    state.start_time -= 40.0
    state.record_event()
    state.update(events_dropped=0)
    snap = state.snapshot()
    assert snap["status"] == "degraded"
    assert any("sync" in r for r in snap["degraded_reasons"])


def test_degraded_when_sync_stale(state):
    state.start_time -= 40.0
    # Feed a sync event then make it stale
    state.update(sync_event_count=1)
    state.last_sync_time = time.monotonic() - 10.0  # 10s ago, threshold is 5s
    snap = state.snapshot()
    assert snap["status"] == "degraded"
    assert any("sync" in r for r in snap["degraded_reasons"])


def test_ok_when_recent_sync_no_drops(state):
    state.start_time -= 40.0   # past the 30 s "starting" window
    state.record_event()
    state.update(events_dropped=0, sync_event_count=1)
    assert state.snapshot()["status"] == "ok"


def test_set_config(state):
    state.set_config(sdr_mode="rspduo", sync_station="KISW_99.9")
    snap = state.snapshot()
    assert snap["sdr_mode"] == "rspduo"
    assert snap["sync_station"] == "KISW_99.9"


def test_set_config_partial(state):
    """set_config only updates provided keys, leaves others unchanged."""
    state.set_config(sdr_mode="rspduo")
    state.set_config(sync_station="KISW_99.9")
    snap = state.snapshot()
    assert snap["sdr_mode"] == "rspduo"
    assert snap["sync_station"] == "KISW_99.9"


# ---------------------------------------------------------------------------
# HealthServer HTTP endpoint
# ---------------------------------------------------------------------------

def _get_health(port: int) -> dict:
    url = f"http://127.0.0.1:{port}/health"
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read())


def test_health_endpoint_returns_200(state):
    srv = HealthServer(state, port=18080, host="127.0.0.1")
    srv.start()
    try:
        data = _get_health(18080)
        assert isinstance(data, dict)
        assert "status" in data
    finally:
        srv.stop()


def test_health_endpoint_json_fields(state):
    srv = HealthServer(state, port=18081, host="127.0.0.1")
    srv.start()
    try:
        data = _get_health(18081)
        assert data["node_id"] == "test-node"
        assert data["status"] == "starting"
    finally:
        srv.stop()


def test_health_404_on_wrong_path(state):
    srv = HealthServer(state, port=18082, host="127.0.0.1")
    srv.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen("http://127.0.0.1:18082/other", timeout=3)
        assert exc_info.value.code == 404
    finally:
        srv.stop()
