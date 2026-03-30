# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for events/model.py - CarrierEvent."""

from __future__ import annotations

import json
import re

import pytest

from beagle_node.events.model import CarrierEvent, NodeLocation, SyncTransmitter


def make_event(**overrides) -> CarrierEvent:
    defaults = dict(
        node_id="seattle-north-01",
        node_location=NodeLocation(latitude_deg=47.71, longitude_deg=-122.33),
        channel_frequency_hz=155_100_000.0,
        sync_delta_ns=123_456_789,
        sync_transmitter=SyncTransmitter(
            station_id="KISW_99.9",
            frequency_hz=99_900_000.0,
            latitude_deg=47.6253,
            longitude_deg=-122.3563,
        ),
        sdr_mode="freq_hop",
        onset_time_ns=1_700_000_000_500_000_000,
        iq_snippet_b64="AAAA",
        channel_sample_rate_hz=64_000.0,
    )
    defaults.update(overrides)
    return CarrierEvent(**defaults)


def test_event_has_uuid_id():
    event = make_event()
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    assert uuid_pattern.match(event.event_id)


def test_event_ids_are_unique():
    e1 = make_event()
    e2 = make_event()
    assert e1.event_id != e2.event_id


def test_event_serializes_to_json():
    event = make_event()
    d = event.to_json_dict()
    # Must round-trip through JSON without error
    json_str = json.dumps(d)
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == "1.4"
    assert parsed["node_id"] == "seattle-north-01"
    assert parsed["sync_delta_ns"] == 123_456_789


def test_event_offset_time_optional():
    event = make_event()
    assert event.offset_time_ns is None
    d = event.to_json_dict()
    assert d["offset_time_ns"] is None


def test_event_pps_anchored_default_false():
    event = make_event()
    assert event.pps_anchored is False


def test_event_schema_version():
    event = make_event()
    assert event.schema_version == "1.4"


# ---------------------------------------------------------------------------
# event_type
# ---------------------------------------------------------------------------

def test_event_type_default_onset():
    event = make_event()
    assert event.event_type == "onset"


def test_event_type_offset():
    event = make_event(event_type="offset")
    assert event.event_type == "offset"


def test_event_type_serialized():
    d = make_event(event_type="offset").to_json_dict()
    assert d["event_type"] == "offset"


def test_event_type_invalid():
    with pytest.raises(Exception):
        make_event(event_type="unknown")


# ---------------------------------------------------------------------------
# sdr_mode valid values
# ---------------------------------------------------------------------------

def test_sdr_mode_valid_values():
    for mode in ("freq_hop", "two_sdr", "single_sdr", "rspduo"):
        event = make_event(sdr_mode=mode)
        assert event.sdr_mode == mode


def test_sdr_mode_invalid():
    with pytest.raises(Exception):
        make_event(sdr_mode="same_sdr")   # old incorrect value
