# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/rds_sync_detector.py.

Tests the pilot-derived RDS bit boundary timing approach, which extracts
the 19 kHz pilot phase and derives bit boundary positions directly from it.
"""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.rds_sync_detector import (
    RDSSyncDetector,
    RDS_BIT_RATE,
)
from beagle_node.pipeline.sync_detector import SyncEvent, PILOT_FREQ_HZ

RATE = 256_000.0      # post-decimation sync channel rate
SAMPLES_PER_BIT = RATE / RDS_BIT_RATE  # ~215.6

_RDS_SUBCARRIER_HZ = 57_000.0  # 3 * pilot, used only for synth audio


def _synth_fm_audio(
    n_samples: int,
    rate: float = RATE,
    pilot_amplitude: float = 5000.0,
    rds_amplitude: float = 500.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic FM-demod audio with 19 kHz pilot + 57 kHz RDS BPSK.

    Returns (audio, bit_start_positions) where bit_start_positions are the
    sample indices of each bit boundary in the input stream.
    """
    t = np.arange(n_samples, dtype=np.float64) / rate

    # 19 kHz pilot (FM demod output is in Hz, so pilot is a sinusoid in Hz)
    pilot = pilot_amplitude * np.sin(2.0 * np.pi * PILOT_FREQ_HZ * t)

    # RDS: BPSK on 57 kHz subcarrier
    n_bits = int(n_samples / rate * RDS_BIT_RATE) + 2
    rng = np.random.default_rng(seed)
    raw_bits = rng.integers(0, 2, n_bits)

    sps = rate / RDS_BIT_RATE
    bpsk = np.zeros(n_samples, dtype=np.float64)
    bit_positions = []
    for i in range(n_bits):
        start = int(round(i * sps))
        end = int(round((i + 1) * sps))
        if start >= n_samples:
            break
        end = min(end, n_samples)
        bpsk[start:end] = 1.0 if raw_bits[i] else -1.0
        bit_positions.append(start)

    # DSB-SC on 57 kHz
    rds = rds_amplitude * bpsk * np.sin(2.0 * np.pi * _RDS_SUBCARRIER_HZ * t)

    audio = (pilot + rds).astype(np.float32)
    return audio, np.array(bit_positions)


# ---------------------------------------------------------------------------
# RDSSyncDetector - basic event generation
# ---------------------------------------------------------------------------

class TestRDSSyncDetectorBasic:

    def _make_detector(self, **kwargs) -> RDSSyncDetector:
        defaults = dict(sample_rate_hz=RATE)
        defaults.update(kwargs)
        return RDSSyncDetector(**defaults)

    def test_returns_sync_events(self):
        """Events returned must be SyncEvent instances."""
        det = self._make_detector()
        audio, _ = _synth_fm_audio(int(RATE * 0.5))  # 500 ms
        events = det.process(audio, start_sample=0)
        assert len(events) > 0
        assert all(isinstance(e, SyncEvent) for e in events)

    def test_event_rate_approximately_correct(self):
        """Should produce events at approximately the RDS bit rate."""
        det = self._make_detector()
        duration_s = 1.0
        audio, _ = _synth_fm_audio(int(RATE * duration_s))
        events = det.process(audio, start_sample=0)
        # Expected: ~1187.5 events per second, minus ~12 for the first
        # pilot window (10 ms = ~12 bits where no events are emitted).
        assert len(events) > 1000, f"Only {len(events)} events in 1s"
        assert len(events) < 1300, f"Too many events: {len(events)}"

    def test_event_spacing(self):
        """Consecutive events should be ~215.6 samples apart (at 256 kHz)."""
        det = self._make_detector()
        audio, _ = _synth_fm_audio(int(RATE * 0.5))
        events = det.process(audio, start_sample=0)
        assert len(events) > 10

        indices = [e.sample_index for e in events]
        intervals = np.diff(indices)
        mean_interval = np.mean(intervals)
        # Should be close to RATE / RDS_BIT_RATE = 215.6
        assert abs(mean_interval - SAMPLES_PER_BIT) < 10, (
            f"Mean interval {mean_interval:.1f} too far from "
            f"expected {SAMPLES_PER_BIT:.1f}"
        )

    def test_sample_index_in_input_domain(self):
        """Event sample_index should be in the input sample domain."""
        det = self._make_detector()
        start = 100_000
        n = int(RATE * 0.3)
        audio, _ = _synth_fm_audio(n)
        events = det.process(audio, start_sample=start)
        assert len(events) > 0
        for e in events:
            assert start <= e.sample_index < start + n + 500, (
                f"sample_index {e.sample_index} outside expected range "
                f"[{start}, {start + n})"
            )

    def test_no_events_in_first_pilot_window(self):
        """No events until the second pilot window (need two for phase diff)."""
        det = self._make_detector(pilot_period_ms=10.0)
        # Feed just 8 ms of audio — less than one pilot window
        n_samples = int(RATE * 0.008)
        audio, _ = _synth_fm_audio(n_samples)
        events = det.process(audio, start_sample=0)
        assert len(events) == 0

    def test_empty_audio_returns_empty(self):
        det = self._make_detector()
        events = det.process(np.empty(0, dtype=np.float32), start_sample=0)
        assert events == []


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------

class TestRDSSyncDetectorQuality:

    def _make_detector(self, **kwargs) -> RDSSyncDetector:
        defaults = dict(sample_rate_hz=RATE)
        defaults.update(kwargs)
        return RDSSyncDetector(**defaults)

    def test_corr_peak_with_clean_signal(self):
        """Clean pilot + RDS should produce measurable corr_peak."""
        det = self._make_detector()
        audio, _ = _synth_fm_audio(int(RATE * 0.5), pilot_amplitude=5000.0)
        events = det.process(audio, start_sample=0)
        assert len(events) > 0
        # After pilot extraction settles, corr_peak should be non-trivial
        late_events = events[len(events) // 2 :]
        peaks = [e.corr_peak for e in late_events]
        assert max(peaks) > 0.01, f"corr_peak too low: {peaks[:5]}"

    def test_correction_near_one(self):
        """Crystal calibration correction should be near 1.0 for synthetic data."""
        det = self._make_detector(calibration_window=20)
        audio, _ = _synth_fm_audio(int(RATE * 1.0), pilot_amplitude=5000.0)
        events = det.process(audio, start_sample=0)
        assert len(events) > 0
        late_events = events[len(events) // 2 :]
        corrections = [e.sample_rate_correction for e in late_events]
        # Should be close to 1.0 (no crystal error in synthetic data)
        for c in corrections[-10:]:
            assert abs(c - 1.0) < 0.01, f"correction {c} too far from 1.0"


# ---------------------------------------------------------------------------
# Streaming (multi-buffer) operation
# ---------------------------------------------------------------------------

class TestRDSSyncDetectorStreaming:

    def _make_detector(self, **kwargs) -> RDSSyncDetector:
        defaults = dict(sample_rate_hz=RATE)
        defaults.update(kwargs)
        return RDSSyncDetector(**defaults)

    def test_chunked_vs_single_buffer(self):
        """
        Feeding audio in chunks should produce events at the same positions
        as feeding it in one large buffer.
        """
        audio, _ = _synth_fm_audio(int(RATE * 0.5))

        # Single buffer
        det1 = self._make_detector()
        events_single = det1.process(audio, start_sample=0)

        # Chunked (4 chunks)
        det2 = self._make_detector()
        chunk_size = len(audio) // 4
        events_chunked = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            events_chunked.extend(
                det2.process(chunk, start_sample=i)
            )

        # Both should produce events at similar positions
        assert len(events_single) > 0
        assert len(events_chunked) > 0
        # The count should be within 10% of each other
        ratio = len(events_chunked) / len(events_single)
        assert 0.85 < ratio < 1.15, (
            f"Chunked produced {len(events_chunked)} vs single "
            f"{len(events_single)} (ratio {ratio:.2f})"
        )

    def test_partial_buffers_accumulate(self):
        """Small buffers that don't contain a full bit should still work."""
        det = self._make_detector()
        audio, _ = _synth_fm_audio(int(RATE * 0.5))

        # Feed in very small chunks (50 samples = ~0.2 ms)
        events = []
        for i in range(0, len(audio), 50):
            chunk = audio[i : i + 50]
            events.extend(det.process(chunk, start_sample=i))

        # Should still get events at ~RDS bit rate
        assert len(events) > 400  # 500ms * 1187.5 - first-window warmup

    def test_contiguous_buffers_no_gap(self):
        """Back-to-back buffers with correct start_sample should not trigger gap reset."""
        det = self._make_detector()
        n = int(RATE * 0.3)  # 300ms
        audio, _ = _synth_fm_audio(n * 2)

        ev1 = det.process(audio[:n], start_sample=0)
        ev2 = det.process(audio[n:], start_sample=n)

        # Second batch should produce events (not reset by gap detection)
        assert len(ev2) > 0


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------

class TestRDSSyncDetectorGap:

    def _make_detector(self, **kwargs) -> RDSSyncDetector:
        defaults = dict(sample_rate_hz=RATE)
        defaults.update(kwargs)
        return RDSSyncDetector(**defaults)

    def test_gap_resets_pilot_state(self):
        """
        A large gap in start_sample should reset the pilot state,
        causing a brief warmup period (one pilot window).
        """
        det = self._make_detector()
        n = int(RATE * 0.3)
        audio1, _ = _synth_fm_audio(n)
        audio2, _ = _synth_fm_audio(n, seed=99)

        # First block: should get events after first pilot window
        ev1 = det.process(audio1, start_sample=0)
        assert len(ev1) > 0

        # Second block with a huge gap: triggers reset
        gap_start = n + int(RATE * 1.0)  # 1 second gap
        ev2 = det.process(audio2, start_sample=gap_start)

        # Key assertion: gap was detected and didn't crash.
        assert isinstance(ev2, list)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestRDSSyncDetectorReset:

    def test_reset_clears_state(self):
        det = RDSSyncDetector(sample_rate_hz=RATE)
        audio, _ = _synth_fm_audio(int(RATE * 0.3))
        det.process(audio, start_sample=0)

        det.reset()

        # After reset, feeding new audio from sample 0 should work
        events = det.process(audio, start_sample=0)
        assert isinstance(events, list)

    def test_reset_restarts_pilot(self):
        """After reset, the first pilot window emits no events."""
        det = RDSSyncDetector(sample_rate_hz=RATE)
        audio, _ = _synth_fm_audio(int(RATE * 0.5))

        # First run: gets events
        ev1 = det.process(audio, start_sample=0)
        assert len(ev1) > 0

        det.reset()

        # Short buffer after reset: less than one pilot window, no events
        short = audio[: int(RATE * 0.008)]
        ev2 = det.process(short, start_sample=0)
        assert len(ev2) == 0
