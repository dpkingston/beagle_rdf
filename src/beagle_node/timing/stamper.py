# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
Sample-domain to wall-clock timestamp conversion.

The EventStamper records the wall-clock time at the start of each SDR buffer,
then converts any sample index within that buffer to a nanosecond timestamp.

This is used ONLY for the rough `onset_time_ns` field in CarrierEvent, which
the server uses for event association across nodes. The precise TDOA measurement
uses sample-domain delta computation in pipeline/delta.py, not this module.
"""

from __future__ import annotations

from beagle_node.timing.clock import ClockSource, SystemClock


class EventStamper:
    """
    Converts sample stream positions to approximate wall-clock timestamps.

    Timing model
    ------------
    When a buffer is received from the SDR, record::

        buffer_start_time_ns = clock.time_ns()  (called immediately after read)
        buffer_start_sample  = cumulative sample count before this buffer

    For an event at sample index S::

        offset_samples = S - buffer_start_sample
        offset_ns      = int(offset_samples * 1_000_000_000 / sample_rate_hz)
        event_time_ns  = buffer_start_time_ns + offset_ns - calibration_offset_ns

    The `calibration_offset_ns` compensates for the constant latency between
    when samples were captured and when time_ns() is called (USB buffering,
    SoapySDR overhead -- typically 1-10 ms).

    Note: this produces rough absolute times (+/-1-10 us) useful for event
    association but NOT for TDOA precision. See pipeline/delta.py.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        calibration_offset_ns: int = 0,
        clock: ClockSource | None = None,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._calibration_offset_ns = calibration_offset_ns
        self._clock = clock if clock is not None else SystemClock()

        self._buffer_start_time_ns: int = 0
        self._buffer_start_sample: int = 0

    def mark_buffer_start(self, cumulative_sample_index: int) -> None:
        """
        Record the wall-clock time at the start of a new buffer.

        Call this immediately after receiving a buffer from the SDR,
        before any processing of its contents.
        """
        self._buffer_start_time_ns = self._clock.time_ns()
        self._buffer_start_sample = cumulative_sample_index

    def stamp(self, sample_index: int) -> int:
        """
        Return the approximate nanosecond timestamp for a given sample index.

        Parameters
        ----------
        sample_index:
            Cumulative sample index within the stream (not within the current buffer).

        Returns
        -------
        int
            Nanoseconds since Unix epoch, corrected by calibration_offset_ns.
        """
        offset_samples = sample_index - self._buffer_start_sample
        offset_ns = int(offset_samples * 1_000_000_000 / self._sample_rate_hz)
        return self._buffer_start_time_ns + offset_ns - self._calibration_offset_ns

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def calibration_offset_ns(self) -> int:
        return self._calibration_offset_ns
