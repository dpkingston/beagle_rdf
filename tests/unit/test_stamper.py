# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for timing/stamper.py."""

from __future__ import annotations

import pytest

from beagle_node.timing.clock import MockClock
from beagle_node.timing.stamper import EventStamper


SAMPLE_RATE = 48_000.0
T0 = 1_700_000_000_000_000_000  # Fixed epoch


def make_stamper(calibration_offset_ns: int = 0) -> tuple[EventStamper, MockClock]:
    clock = MockClock(start_ns=T0)
    stamper = EventStamper(
        sample_rate_hz=SAMPLE_RATE,
        calibration_offset_ns=calibration_offset_ns,
        clock=clock,
    )
    return stamper, clock


def test_stamp_first_sample():
    stamper, _ = make_stamper()
    stamper.mark_buffer_start(cumulative_sample_index=0)
    result = stamper.stamp(0)
    assert result == T0


def test_stamp_offset_10ms():
    """Sample index 480 at 48 kHz should be 10 ms after buffer start."""
    stamper, _ = make_stamper()
    stamper.mark_buffer_start(0)
    result = stamper.stamp(480)
    expected = T0 + 10_000_000  # 10 ms in ns
    assert result == expected


def test_stamp_calibration_offset_applied():
    calibration_ns = 3_500_000  # 3.5 ms
    stamper, _ = make_stamper(calibration_offset_ns=calibration_ns)
    stamper.mark_buffer_start(0)
    result = stamper.stamp(0)
    assert result == T0 - calibration_ns


def test_stamp_across_buffer_boundary():
    """Second buffer starts with a later cumulative sample index."""
    stamper, clock = make_stamper()
    # First buffer: 1024 samples
    stamper.mark_buffer_start(0)
    t_at_1024 = stamper.stamp(1024)
    expected_1024 = T0 + int(1024 * 1_000_000_000 / SAMPLE_RATE)
    assert t_at_1024 == expected_1024

    # Second buffer starts 1024 samples later, clock advances by the same interval
    clock.advance(int(1024 * 1_000_000_000 / SAMPLE_RATE))
    stamper.mark_buffer_start(1024)
    t_at_2048 = stamper.stamp(2048)
    expected_2048 = T0 + int(1024 * 1_000_000_000 / SAMPLE_RATE) + int(
        1024 * 1_000_000_000 / SAMPLE_RATE
    )
    assert t_at_2048 == expected_2048


def test_sample_rate_property():
    stamper, _ = make_stamper()
    assert stamper.sample_rate_hz == SAMPLE_RATE


def test_calibration_offset_property():
    stamper, _ = make_stamper(calibration_offset_ns=12345)
    assert stamper.calibration_offset_ns == 12345
