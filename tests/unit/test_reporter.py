# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for events/reporter.py."""

from __future__ import annotations

import time
import uuid
from unittest.mock import patch

import pytest
import httpx

from beagle_node.events.model import CarrierEvent, NodeLocation, SyncTransmitter
from beagle_node.events.reporter import EventReporter


def make_event(**kwargs) -> CarrierEvent:
    defaults = dict(
        node_id="test-node",
        node_location=NodeLocation(latitude_deg=47.6, longitude_deg=-122.3),
        channel_frequency_hz=155_100_000.0,
        sync_delta_ns=5_000_000,
        sync_transmitter=SyncTransmitter(
            station_id="KISW_99.9",
            frequency_hz=99_900_000.0,
            latitude_deg=47.652,
            longitude_deg=-122.130,
        ),
        sdr_mode="freq_hop",
        onset_time_ns=1_700_000_000_000_000_000,
        iq_snippet_b64="AAAA",
        channel_sample_rate_hz=64_000.0,
    )
    defaults.update(kwargs)
    return CarrierEvent(**defaults)


def make_reporter(**kwargs) -> EventReporter:
    defaults = dict(
        server_url="http://localhost:9999",
        auth_token="test-token",
        max_queue=10,
        timeout_s=1.0,
        max_retries=2,
        retry_base_s=0.01,   # fast retries in tests
    )
    defaults.update(kwargs)
    return EventReporter(**defaults)


# ---------------------------------------------------------------------------
# Queue and properties
# ---------------------------------------------------------------------------

def test_initial_state():
    r = make_reporter()
    assert r.queue_depth == 0
    assert r.events_submitted == 0
    assert r.events_dropped == 0


def test_submit_increments_queue_depth():
    r = make_reporter()
    r.submit(make_event())
    assert r.queue_depth == 1
    r.submit(make_event())
    assert r.queue_depth == 2


def test_queue_full_drops_oldest():
    """When queue is full, oldest event is dropped to make room for newest."""
    r = make_reporter(max_queue=3)
    for _ in range(4):
        r.submit(make_event())
    assert r.queue_depth <= 3
    assert r.events_dropped >= 1


# ---------------------------------------------------------------------------
# Successful delivery
# ---------------------------------------------------------------------------

def test_successful_delivery(httpx_mock):
    httpx_mock.add_response(status_code=201)
    r = make_reporter()
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_submitted == 1
    assert r.events_dropped == 0


def test_posts_to_correct_url(httpx_mock):
    httpx_mock.add_response(status_code=200)
    r = make_reporter(server_url="http://server.example.com")
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert str(requests[0].url) == "http://server.example.com/api/v1/events"


def test_auth_header_sent(httpx_mock):
    httpx_mock.add_response(status_code=200)
    r = make_reporter(auth_token="secret-xyz")
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    req = httpx_mock.get_requests()[0]
    assert req.headers["authorization"] == "Bearer secret-xyz"


def test_multiple_events_delivered(httpx_mock):
    httpx_mock.add_response(status_code=200)
    httpx_mock.add_response(status_code=200)
    httpx_mock.add_response(status_code=200)
    r = make_reporter()
    r.start()
    for _ in range(3):
        r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_submitted == 3


# ---------------------------------------------------------------------------
# Retry and failure
# ---------------------------------------------------------------------------

def test_server_error_retries_then_drops(httpx_mock):
    """500 response on all attempts -> event dropped after max_retries."""
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    r = make_reporter(max_retries=2, retry_base_s=0.01)
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_submitted == 0
    assert r.events_dropped == 1


def test_retry_succeeds_on_second_attempt(httpx_mock):
    """First attempt fails, second succeeds."""
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=200)
    r = make_reporter(max_retries=2, retry_base_s=0.01)
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_submitted == 1
    assert r.events_dropped == 0


def test_transport_error_retries(httpx_mock):
    """Connection refused -> retries -> drop."""
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    r = make_reporter(max_retries=2, retry_base_s=0.01)
    r.start()
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_dropped == 1


# ---------------------------------------------------------------------------
# Start / stop idempotency
# ---------------------------------------------------------------------------

def test_start_twice_is_safe(httpx_mock):
    httpx_mock.add_response(status_code=200)
    r = make_reporter()
    r.start()
    r.start()   # second call should be no-op
    r.submit(make_event())
    r.stop(timeout_s=3.0)
    assert r.events_submitted == 1


def test_stop_before_start_is_safe():
    r = make_reporter()
    r.stop()   # should not raise


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_posts_to_correct_url(httpx_mock):
    httpx_mock.add_response(status_code=200)
    r = make_reporter(server_url="http://server.example.com")
    r.post_heartbeat({"node_id": "n1", "latitude_deg": 47.6})
    # Give the daemon thread time to complete
    time.sleep(0.5)
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert str(requests[0].url) == "http://server.example.com/api/v1/heartbeat"
    body = requests[0].read()
    import json
    assert json.loads(body)["node_id"] == "n1"


def test_heartbeat_sends_auth_header(httpx_mock):
    httpx_mock.add_response(status_code=200)
    r = make_reporter(auth_token="secret-hb")
    r.post_heartbeat({"node_id": "n1"})
    time.sleep(0.5)
    req = httpx_mock.get_requests()[0]
    assert req.headers["authorization"] == "Bearer secret-hb"


def test_heartbeat_disabled_reporter_does_nothing():
    """Reporter with empty server_url should not attempt heartbeat."""
    r = make_reporter(server_url="")
    # Should return immediately without error
    r.post_heartbeat({"node_id": "n1"})
