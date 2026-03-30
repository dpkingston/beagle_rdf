# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Clock abstraction for wall-time timestamps.

SystemClock uses time.time_ns() which, on a GPS-1PPS-disciplined node
(chrony + gpsd), is accurate to ~1 us absolute. MockClock provides
a controllable clock for deterministic testing.
"""

from __future__ import annotations

import abc
import time


class ClockSource(abc.ABC):
    """Abstract wall-clock source returning nanoseconds since Unix epoch."""

    @abc.abstractmethod
    def time_ns(self) -> int:
        """Current time as integer nanoseconds since Unix epoch."""


class SystemClock(ClockSource):
    """
    Reads the kernel clock via time.time_ns().

    On a node with GPS 1PPS feeding chrony, this reflects the
    GPS-disciplined system clock, accurate to ~1 us absolute.
    No additional configuration is needed here - chrony handles discipline.
    """

    def time_ns(self) -> int:
        return time.time_ns()


class MockClock(ClockSource):
    """
    Controllable clock for testing.

    Starts at `start_ns` and advances by `step_ns` on each call to time_ns().
    Can also be advanced or set explicitly.
    """

    def __init__(self, start_ns: int = 1_700_000_000_000_000_000, step_ns: int = 0) -> None:
        self._current_ns = start_ns
        self._step_ns = step_ns

    def time_ns(self) -> int:
        value = self._current_ns
        self._current_ns += self._step_ns
        return value

    def advance(self, ns: int) -> None:
        """Advance the clock by `ns` nanoseconds."""
        self._current_ns += ns

    def set(self, ns: int) -> None:
        """Set the clock to an absolute nanosecond value."""
        self._current_ns = ns
