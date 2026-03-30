# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for timing/clock.py."""

from __future__ import annotations

import time

from beagle_node.timing.clock import MockClock, SystemClock


def test_mock_clock_starts_at_given_value():
    clock = MockClock(start_ns=1_000_000_000)
    assert clock.time_ns() == 1_000_000_000


def test_mock_clock_step():
    clock = MockClock(start_ns=0, step_ns=1_000_000)
    assert clock.time_ns() == 0
    assert clock.time_ns() == 1_000_000
    assert clock.time_ns() == 2_000_000


def test_mock_clock_advance():
    clock = MockClock(start_ns=0)
    clock.advance(5_000_000)
    assert clock.time_ns() == 5_000_000


def test_mock_clock_set():
    clock = MockClock(start_ns=100)
    clock.set(999_999)
    assert clock.time_ns() == 999_999


def test_system_clock_returns_reasonable_value():
    clock = SystemClock()
    t = clock.time_ns()
    # Must be after 2020-01-01 and before 2100-01-01
    assert 1_577_836_800_000_000_000 < t < 4_102_444_800_000_000_000


def test_system_clock_advances():
    clock = SystemClock()
    t1 = clock.time_ns()
    time.sleep(0.001)
    t2 = clock.time_ns()
    assert t2 > t1
