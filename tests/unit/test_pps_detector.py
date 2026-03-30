# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for pipeline/pps_detector.py."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.pps_detector import PPSAnchor, PPSDetector

RATE = 2_048_000.0   # raw SDR rate (PPS detected before decimation)


def make_detector(**kwargs) -> PPSDetector:
    defaults = dict(
        sample_rate_hz=RATE,
        spike_threshold_db=10.0,
        window_samples=32,
        baseline_window=50,
        min_spacing_s=0.9,
    )
    defaults.update(kwargs)
    return PPSDetector(**defaults)


def _noise(n: int, power_db: float, rng: np.random.Generator) -> np.ndarray:
    amp = 10 ** (power_db / 20.0)
    return (amp / np.sqrt(2)) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)


def _spike(n: int, spike_pos: int, spike_width: int,
           noise_db: float, spike_db: float,
           rng: np.random.Generator) -> np.ndarray:
    """Noise buffer with a brief power spike at spike_pos."""
    iq = _noise(n, noise_db, rng)
    spike_amp = 10 ** (spike_db / 20.0)
    sl = slice(spike_pos, spike_pos + spike_width)
    iq[sl] = spike_amp * np.ones(spike_width, dtype=np.complex64)
    return iq


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_invalid_threshold():
    with pytest.raises(ValueError, match="spike_threshold_db"):
        PPSDetector(RATE, spike_threshold_db=0.0)


def test_invalid_window():
    with pytest.raises(ValueError, match="window_samples"):
        PPSDetector(RATE, window_samples=0)


# ---------------------------------------------------------------------------
# No spike -> no events
# ---------------------------------------------------------------------------

def test_no_events_on_noise():
    rng = np.random.default_rng(0)
    det = make_detector()
    iq = _noise(65536, power_db=-60.0, rng=rng)
    events = det.process(iq, start_sample=0)
    assert events == []


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def test_spike_detected():
    rng = np.random.default_rng(1)
    det = make_detector(baseline_window=20)
    # Give the detector some quiet background first
    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)

    spike_pos = 500
    iq = _spike(4096, spike_pos, spike_width=16,
                noise_db=-60.0, spike_db=-20.0, rng=rng)
    events = det.process(iq, start_sample=10_000)
    assert len(events) == 1
    assert isinstance(events[0], PPSAnchor)


def test_spike_sample_index_near_spike_position():
    """Detected sample index should be close to the actual spike position."""
    rng = np.random.default_rng(2)
    det = make_detector(window_samples=16, baseline_window=20)

    # Warm up baseline
    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)

    start = 10_000
    spike_pos = 512  # within the buffer
    iq = _spike(4096, spike_pos, spike_width=8,
                noise_db=-60.0, spike_db=-10.0, rng=rng)
    events = det.process(iq, start_sample=start)

    assert len(events) == 1
    detected = events[0].sample_index - start
    # Should be within 2 window widths of the true position
    assert abs(detected - spike_pos) < 64, \
        f"Detected at {detected}, expected near {spike_pos}"


def test_spike_power_exceeds_baseline():
    rng = np.random.default_rng(3)
    det = make_detector(baseline_window=20)
    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)

    iq = _spike(4096, 512, 16, noise_db=-60.0, spike_db=-20.0, rng=rng)
    events = det.process(iq, start_sample=10_000)
    assert len(events) == 1
    assert events[0].power_db > events[0].baseline_db + 5.0


# ---------------------------------------------------------------------------
# Min-spacing enforcement
# ---------------------------------------------------------------------------

def test_double_trigger_suppressed():
    """Two spikes < 0.9 s apart: only the first is reported."""
    rng = np.random.default_rng(4)
    det = make_detector(window_samples=16, baseline_window=20, min_spacing_s=0.1)

    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)

    # Two spikes 1000 samples apart (0.49 ms at 2 MSPS -- well within 0.1 s)
    iq = _noise(8192, power_db=-60.0, rng=rng)
    spike_amp = 10 ** (-10.0 / 20.0)
    iq[500:516]  = spike_amp
    iq[1500:1516] = spike_amp

    events = det.process(iq, start_sample=10_000)
    assert len(events) == 1


def test_two_spikes_one_second_apart():
    """Two spikes > min_spacing apart should both be detected."""
    rng = np.random.default_rng(5)
    det = make_detector(window_samples=16, baseline_window=20, min_spacing_s=0.001)

    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)

    # First spike
    iq1 = _spike(4096, 500, 16, noise_db=-60.0, spike_db=-10.0, rng=rng)
    events1 = det.process(iq1, start_sample=10_000)

    # Second spike -- well after min_spacing
    quiet2 = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet2, start_sample=14_096)

    iq2 = _spike(4096, 500, 16, noise_db=-60.0, spike_db=-10.0, rng=rng)
    events2 = det.process(iq2, start_sample=24_096)

    assert len(events1) == 1
    assert len(events2) == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_baseline_and_spacing():
    rng = np.random.default_rng(6)
    det = make_detector(baseline_window=20)
    quiet = _noise(10_000, power_db=-60.0, rng=rng)
    det.process(quiet, start_sample=0)
    det.reset()
    # After reset, detector should work cleanly from sample 0 again
    quiet2 = _noise(5_000, power_db=-60.0, rng=rng)
    events = det.process(quiet2, start_sample=0)
    assert events == []
