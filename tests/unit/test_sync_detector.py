# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/sync_detector.py."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.sync_detector import (
    CrystalCalibrator,
    FMPilotSyncDetector,
    SyncEvent,
    PILOT_FREQ_HZ,
)

RATE = 256_000.0      # post-decimation sync channel rate
PERIOD_MS = 10.0      # 10 ms sync windows -> 2560 samples


def _pilot_audio(n: int, rate: float, freq_hz: float = PILOT_FREQ_HZ,
                 amplitude: float = 1.0) -> np.ndarray:
    """Sinusoidal audio at pilot frequency (simulates FM-demodulated pilot)."""
    t = np.arange(n) / rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _noise_audio(n: int, amplitude: float = 0.01,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# CrystalCalibrator
# ---------------------------------------------------------------------------

class TestCrystalCalibrator:

    def test_initial_correction_is_one(self):
        cal = CrystalCalibrator(sync_period_s=0.010)
        # First update sets baseline; no correction yet
        c = cal.update(0.0)
        assert c == 1.0

    def test_correction_near_one_for_exact_rate(self):
        """If measured phase advance exactly matches expected, correction -> 1.0."""
        cal = CrystalCalibrator(sync_period_s=0.010, window=5)
        expected_advance = 2.0 * np.pi * PILOT_FREQ_HZ * 0.010

        phase = 0.0
        for _ in range(10):
            cal.update(phase)
            phase += expected_advance

        c = cal.update(phase)
        assert abs(c - 1.0) < 1e-6

    def test_correction_for_fast_crystal(self):
        """Crystal running 100 ppm fast -> correction < 1.0."""
        cal = CrystalCalibrator(sync_period_s=0.010, window=5)
        fast_factor = 1 + 100e-6   # 100 ppm fast
        expected_advance = 2.0 * np.pi * PILOT_FREQ_HZ * 0.010

        phase = 0.0
        for _ in range(20):
            phase += expected_advance * fast_factor
            cal.update(phase)

        c = cal.update(phase)
        # correction should be close to fast_factor
        assert abs(c - fast_factor) < 1e-4

    def test_outlier_rejected(self):
        """Phase jump > 0.2% is rejected; correction stays near 1.0."""
        cal = CrystalCalibrator(sync_period_s=0.010, window=10)
        expected_advance = 2.0 * np.pi * PILOT_FREQ_HZ * 0.010

        phase = 0.0
        for _ in range(10):
            phase += expected_advance
            cal.update(phase)

        # Inject an outlier (10% off)
        cal.update(phase + expected_advance * 1.10)

        c = cal.update(phase + expected_advance * 2.10)
        # Median over 10 good + 1 bad; should still be close to 1.0
        assert abs(c - 1.0) < 0.01

    def test_reset_clears_history(self):
        cal = CrystalCalibrator(sync_period_s=0.010, window=5)
        expected_advance = 2.0 * np.pi * PILOT_FREQ_HZ * 0.010
        phase = 0.0
        for _ in range(10):
            phase += expected_advance
            cal.update(phase)
        cal.reset()
        assert cal.update(0.0) == 1.0


# ---------------------------------------------------------------------------
# FMPilotSyncDetector - event rate
# ---------------------------------------------------------------------------

class TestFMPilotSyncDetector:

    def _make_detector(self, **kwargs) -> FMPilotSyncDetector:
        defaults = dict(sample_rate_hz=RATE, sync_period_ms=PERIOD_MS)
        defaults.update(kwargs)
        return FMPilotSyncDetector(**defaults)

    def test_one_event_per_period(self):
        """N * period_samples input -> exactly N events."""
        det = self._make_detector()
        period = det.sync_period_samples
        n_periods = 5
        audio = _pilot_audio(period * n_periods, RATE)
        events = det.process(audio, start_sample=0)
        assert len(events) == n_periods

    def test_partial_buffer_no_event(self):
        """Input shorter than one period -> no events yet."""
        det = self._make_detector()
        audio = _pilot_audio(det.sync_period_samples - 1, RATE)
        events = det.process(audio, start_sample=0)
        assert events == []

    def test_event_accumulates_across_buffers(self):
        """Feeding half-period chunks still produces one event per period."""
        det = self._make_detector()
        period = det.sync_period_samples
        half = period // 2
        audio = _pilot_audio(period * 4, RATE)

        events = []
        for i in range(0, len(audio), half):
            chunk = audio[i : i + half]
            events += det.process(chunk, start_sample=i)

        assert len(events) == 4

    def test_event_is_syncEvent_instance(self):
        det = self._make_detector()
        period = det.sync_period_samples
        audio = _pilot_audio(period, RATE)
        events = det.process(audio, start_sample=0)
        assert len(events) == 1
        assert isinstance(events[0], SyncEvent)

    def test_sample_index_within_window(self):
        """Event sample_index must lie within [start, start + period)."""
        det = self._make_detector()
        period = det.sync_period_samples
        start = 50_000
        audio = _pilot_audio(period, RATE)
        events = det.process(audio, start_sample=start)
        assert len(events) == 1
        assert start <= events[0].sample_index < start + period

    # ---------------------------------------------------------------------------
    # Signal quality
    # ---------------------------------------------------------------------------

    def test_high_corr_peak_with_clean_pilot(self):
        """Clean pilot signal should produce corr_peak > 0.3."""
        det = self._make_detector()
        period = det.sync_period_samples
        audio = _pilot_audio(period * 3, RATE, amplitude=10.0)
        events = det.process(audio, start_sample=0)
        assert all(e.corr_peak > 0.3 for e in events), \
            f"corr_peaks = {[e.corr_peak for e in events]}"

    def test_low_corr_peak_with_pure_noise(self):
        """Pure noise (no pilot) should have low corr_peak."""
        rng = np.random.default_rng(42)
        det = self._make_detector()
        period = det.sync_period_samples
        audio = _noise_audio(period * 3, amplitude=1.0, rng=rng)
        events = det.process(audio, start_sample=0)
        assert all(e.corr_peak < 0.5 for e in events), \
            f"corr_peaks = {[e.corr_peak for e in events]}"

    # ---------------------------------------------------------------------------
    # Crystal calibration integration
    # ---------------------------------------------------------------------------

    def test_correction_near_one_for_correct_rate(self):
        """After several windows at the correct rate, correction ~= 1.0."""
        det = self._make_detector(calibration_window=20)
        period = det.sync_period_samples
        audio = _pilot_audio(period * 25, RATE, amplitude=10.0)
        events = det.process(audio, start_sample=0)
        # After warm-up, correction should be close to 1.0
        late_corrections = [e.sample_rate_correction for e in events[5:]]
        assert all(abs(c - 1.0) < 0.01 for c in late_corrections), \
            f"corrections = {late_corrections}"

    # ---------------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------------

    def test_reset_clears_buffer_and_state(self):
        det = self._make_detector()
        period = det.sync_period_samples
        # Feed partial window (stays in buffer)
        det.process(_pilot_audio(period // 2, RATE), start_sample=0)
        det.reset()
        # After reset, feeding a full period from sample 0 should work cleanly
        events = det.process(_pilot_audio(period, RATE), start_sample=0)
        assert len(events) == 1

    # ---------------------------------------------------------------------------
    # freq_hop gap handling
    # ---------------------------------------------------------------------------

    def test_gap_resets_sample_counter(self):
        """
        In freq_hop mode, consecutive sync blocks are separated by a gap equal
        to target_block / sync_decimation samples.  The second sync block must
        be treated as starting at its declared start_sample, not continuing
        from the end of the previous block's internal buffer.

        Without gap handling, the second event's sample_index would be near
        the tail of the first block's internal counter (~172,800), far below
        the actual start_sample of the second block (~209,920).  With gap
        handling, it must be within [second_start, second_start + period).
        """
        det = self._make_detector()
        period = det.sync_period_samples   # 2560 at 256 kHz / 10 ms
        usable = 3072                      # (65,536 - 40,960) / 8  (typical freq_hop)

        first_start = 168_960
        # target_block = 262,144 raw samples -> /8 = 32,768 sync-dec gap
        second_start = first_start + usable + 32_768   # = 204,800 + some offset

        audio1 = _pilot_audio(usable, RATE)
        audio2 = _pilot_audio(usable, RATE)

        ev1 = det.process(audio1, start_sample=first_start)
        ev2 = det.process(audio2, start_sample=second_start)

        assert len(ev1) >= 1, "First block should produce at least one sync event"
        assert len(ev2) >= 1, "Second block should produce at least one sync event"

        # Event from second block must be anchored to second_start, not first_start
        for e in ev2:
            assert second_start <= e.sample_index < second_start + period, (
                f"Second-block event sample_index {e.sample_index} not anchored "
                f"to second_start={second_start} (gap was not detected/handled)"
            )

    def test_no_gap_preserves_continuity(self):
        """
        When blocks are truly contiguous (no gap), the second block's event
        continues naturally from the first block's counter.
        """
        det = self._make_detector()
        period = det.sync_period_samples   # 2560

        start1 = 10_000
        audio1 = _pilot_audio(period, RATE)
        ev1 = det.process(audio1, start_sample=start1)
        assert len(ev1) == 1

        # Second block starts right where the first ended - no gap
        start2 = start1 + period
        audio2 = _pilot_audio(period, RATE)
        ev2 = det.process(audio2, start_sample=start2)
        assert len(ev2) == 1

        # Without a gap the event is within [start2, start2 + period)
        assert start2 <= ev2[0].sample_index < start2 + period
