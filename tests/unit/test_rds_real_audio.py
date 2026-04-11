# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Regression test: real FM audio → RDSSyncDetector → stable sync events.

Uses tests/fixtures/kuow_sync_audio_30s.npz (KUOW 94.9 MHz captured from
dpk-tdoa1 via BEAGLE_CAPTURE_SYNC_AUDIO).  This validates the pilot-derived
RDS bit boundary timing on real-world signal conditions (multipath, fading,
noise, etc.) and guards against regressions in the sync detector.

Key metric: mod-RDS-bit-period std of sync event positions should be <1 sample
(was 0.055 samples after the pilot-derived timing fix).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from beagle_node.pipeline.rds_sync_detector import RDSSyncDetector

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "kuow_sync_audio_30s.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="kuow_sync_audio_30s.npz fixture not available",
)


@pytest.fixture(scope="module")
def sync_events():
    """Feed real FM audio through RDSSyncDetector and collect all events."""
    data = np.load(FIXTURE_PATH)
    audio = data["audio"]
    starts = data["start_samples"]
    rate = float(data["sample_rate_hz"])

    det = RDSSyncDetector(sample_rate_hz=rate)

    all_events = []
    offset = 0
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(audio)
        chunk = audio[offset:offset + (end - start)]
        events = det.process(chunk, start_sample=int(start))
        all_events.extend(events)
        offset += len(chunk)

    return all_events, rate


def test_produces_sync_events(sync_events):
    """30 seconds of real audio should produce many sync events."""
    events, rate = sync_events
    # At 1187.5 Hz, 30s should produce ~35,625 events.
    # Warmup takes ~5s (500 pilot windows), so expect ~29,700.
    assert len(events) > 25_000, f"Too few sync events: {len(events)}"


def test_post_warmup_spacing_stable(sync_events):
    """
    After BPF settling (~5s), inter-event spacings should be very stable.

    The pilot-derived timing produces sub-sample jitter on the RDS bit
    grid.  During warmup (first 500 pilot windows = 5s) the phase offset
    is not yet set, causing ~1.4% of spacings to be 26 samples short.
    After warmup, all spacings should be within ±1 sample of expected.

    Key metric: spacing residual std < 0.5 samples (was 0.055 in session).
    """
    events, rate = sync_events
    # Skip warmup: only use events after 6 seconds (conservative)
    warmup_samples = 6.0 * rate
    good = [e for e in events
            if e.corr_peak > 0.5 and e.sample_index > warmup_samples]
    assert len(good) > 20_000, f"Too few post-warmup events: {len(good)}"

    positions = np.array([e.sample_index for e in good])
    diffs = np.diff(positions)

    # Expected spacing using median crystal correction
    corrections = [e.sample_rate_correction for e in good]
    median_corr = float(np.median(corrections))
    expected = rate * median_corr / 1187.5

    residuals = diffs - expected
    std = float(np.std(residuals))
    max_err = float(np.max(np.abs(residuals)))

    assert std < 0.5, (
        f"Post-warmup spacing std = {std:.4f} samples — expected < 0.5 "
        f"(pilot-derived timing regression?)"
    )
    assert max_err < 2.0, (
        f"Post-warmup max spacing error = {max_err:.2f} samples — "
        f"expected < 2.0"
    )


def test_corr_peak_healthy(sync_events):
    """Pilot correlation peaks should be consistently strong on real FM audio."""
    events, rate = sync_events
    peaks = np.array([e.corr_peak for e in events])
    median_peak = float(np.median(peaks))
    # KUOW 94.9 typically shows corr_peak ~0.70
    assert median_peak > 0.4, (
        f"Median corr_peak = {median_peak:.4f} — expected > 0.4"
    )


def test_no_large_gaps(sync_events):
    """Sync events should arrive regularly — no gaps > 5 ms after warmup."""
    events, rate = sync_events
    if len(events) < 100:
        pytest.skip("Too few events for gap analysis")

    positions = np.array([e.sample_index for e in events])
    # Skip warmup: first event with corr_peak > 0.5
    good_mask = np.array([e.corr_peak > 0.5 for e in events])
    if not np.any(good_mask):
        pytest.skip("No events with good corr_peak")
    first_good = int(np.argmax(good_mask))
    positions = positions[first_good:]

    diffs = np.diff(positions)
    max_gap_samples = float(np.max(diffs))
    max_gap_ms = max_gap_samples / rate * 1000
    # RDS bit period is ~3.37 ms at 250 kHz; allow up to 5 ms
    assert max_gap_ms < 5.0, (
        f"Max inter-event gap = {max_gap_ms:.2f} ms — expected < 5.0 ms"
    )


def test_crystal_correction_reasonable(sync_events):
    """Crystal calibration factor should be close to 1.0 (< 50 ppm)."""
    events, rate = sync_events
    corrections = [e.sample_rate_correction for e in events if e.corr_peak > 0.5]
    if not corrections:
        pytest.skip("No events with good corr_peak")
    median_corr = float(np.median(corrections))
    ppm = abs(median_corr - 1.0) * 1e6
    assert ppm < 50.0, (
        f"Crystal correction = {median_corr:.8f} ({ppm:.1f} ppm) — expected < 50 ppm"
    )
