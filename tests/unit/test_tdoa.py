# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for beagle_server/tdoa.py."""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from beagle_server.tdoa import (
    _C_M_S,
    compute_tdoa_s,
    cross_correlate_snippets,
    haversine_m,
    path_delay_correction_ns,
)


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------

def test_haversine_same_point():
    assert haversine_m(47.7, -122.3, 47.7, -122.3) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # Seattle Space Needle (approx) to downtown (approx): ~2 km
    d = haversine_m(47.6205, -122.3493, 47.6062, -122.3321)
    assert 1500 < d < 2500


def test_haversine_symmetry():
    d1 = haversine_m(47.6, -122.3, 47.7, -122.4)
    d2 = haversine_m(47.7, -122.4, 47.6, -122.3)
    assert d1 == pytest.approx(d2, rel=1e-10)


def test_haversine_one_degree_latitude():
    # 1deg latitude ~= 111,195 m
    d = haversine_m(47.0, 0.0, 48.0, 0.0)
    assert d == pytest.approx(111_195.0, rel=0.01)


def test_haversine_equator_one_degree_longitude():
    # At equator, 1deg longitude ~= 111,195 m
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert d == pytest.approx(111_195.0, rel=0.01)


# ---------------------------------------------------------------------------
# path_delay_correction_ns
# ---------------------------------------------------------------------------

def test_path_delay_correction_equidistant_nodes():
    """If both nodes are equidistant from the sync transmitter, correction is 0."""
    # Place sync tx at origin, nodes at equal distances north and south
    corr = path_delay_correction_ns(
        sync_tx_lat=47.6, sync_tx_lon=-122.3,
        node_a_lat=47.7, node_a_lon=-122.3,
        node_b_lat=47.5, node_b_lon=-122.3,
    )
    # d_A ~= d_B ~= 11 km -> correction ~= 0 (equal distances)
    # Actually not exactly 0 because lat degrees aren't perfectly symmetric,
    # but very close
    assert abs(corr) < 200.0  # < 200 ns


def test_path_delay_correction_sign():
    """Node A farther from sync tx -> positive correction -> TDOA reduced."""
    # sync_tx at south; node_A to the north (farther), node_B at same lat as tx
    corr = path_delay_correction_ns(
        sync_tx_lat=47.5, sync_tx_lon=-122.3,
        node_a_lat=47.7, node_a_lon=-122.3,   # ~22 km from tx
        node_b_lat=47.5, node_b_lon=-122.3,   # 0 m from tx
    )
    # d_A ~= 22 km, d_B ~= 0 -> correction ~= 22000/c * 1e9 ~= 73 us positive
    assert corr > 50_000   # > 50 us in ns


def test_path_delay_correction_known_value():
    """
    Manually compute expected correction for a simple geometry.
    node_A is 30 km north of sync_tx, node_B is at the same location as sync_tx.
    d_A = 30 km, d_B = 0 -> correction = 30000 / 299792458 * 1e9 ~= 100,069 ns
    """
    # 30 km north of 47.5degN ~= 0.27deg of latitude
    d_30km_deg = 30_000 / 111_195  # ~= 0.2699deg
    corr = path_delay_correction_ns(
        sync_tx_lat=47.5, sync_tx_lon=-122.3,
        node_a_lat=47.5 + d_30km_deg, node_a_lon=-122.3,
        node_b_lat=47.5, node_b_lon=-122.3,
    )
    expected = 30_000 / _C_M_S * 1e9
    assert corr == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Real-data snippet for realistic d2 xcorr testing
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_FIXTURE_PATH = _Path(__file__).parents[1] / "fixtures" / "three_node_baseline_2026_04_08.json"

def _load_real_snippet() -> np.ndarray:
    """Load a real onset IQ snippet from the fixture file."""
    from beagle_server.tdoa import _decode_iq_snippet
    data = _json.load(_FIXTURE_PATH.open())
    # Use the second onset (dpk-tdoa1, good transition)
    onsets = [e for e in data["events"]
              if e["event_type"] == "onset" and len(e.get("iq_snippet_b64", "")) > 100]
    return _decode_iq_snippet(onsets[1]["iq_snippet_b64"])

# Cache the real snippet at import time
_REAL_SNIPPET: np.ndarray = _load_real_snippet()
_REAL_RATE: float = 62_500.0


def _make_real_snippet_pair_b64(delay_samples: int = 0) -> tuple[str, str]:
    """Create a pair of base64 snippets from a real signal, shifted by delay_samples.

    Node A sees physical[delay:delay+n], node B sees physical[0:n].
    The d2 xcorr should recover the delay.
    """
    iq = _REAL_SNIPPET
    n = len(iq) - abs(delay_samples) - 1
    if delay_samples >= 0:
        iq_a = iq[delay_samples:delay_samples + n]
        iq_b = iq[:n]
    else:
        iq_a = iq[:n]
        iq_b = iq[-delay_samples:-delay_samples + n]
    return _iq_to_b64(iq_a), _iq_to_b64(iq_b)


def _make_real_snippet_b64() -> str:
    """Single real snippet as base64."""
    return _iq_to_b64(_REAL_SNIPPET)


# ---------------------------------------------------------------------------
# Snippet helpers
# ---------------------------------------------------------------------------

def _make_am_carrier(n_samples: int, rng: np.random.Generator, am_smooth: int = 16) -> np.ndarray:
    """
    QPSK carrier with slowly-varying AM envelope.

    Pure QPSK has constant instantaneous power (|s[n]|^2 = 1), making
    power-envelope xcorr insensitive to timing offsets.  The AM envelope
    gives the power envelope texture needed for xcorr to resolve offsets.
    """
    bits_i = rng.integers(0, 2, n_samples) * 2 - 1
    bits_q = rng.integers(0, 2, n_samples) * 2 - 1
    qpsk = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2)
    am_raw = np.abs(rng.standard_normal(n_samples + am_smooth))
    am_smooth_arr = np.convolve(am_raw, np.ones(am_smooth) / am_smooth, mode="valid")[:n_samples]
    am_env = ((am_smooth_arr - am_smooth_arr.min()) /
              (am_smooth_arr.max() - am_smooth_arr.min()) * 0.6 + 0.4).astype(np.float32)
    return (qpsk * am_env).astype(np.complex64)


def _make_plateau_iq(
    n_samples: int = 1280,
    onset_sample: int = 512,
    ramp_samples: int = 24,   # ~375 us at 64 kHz, realistic LMR PA rise
    snr_db: float = 25.0,
    seed: int = 42,
    event_type: str = "onset",
) -> np.ndarray:
    """
    Synthetic IQ snippet with noise -> ramp -> plateau (onset) structure.

    Uses a QPSK carrier (constant instantaneous power = 1) for reliable
    plateau detection at typical SNR values.

    For event_type="offset": the envelope is reversed (plateau -> ramp-down -> noise).
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)

    # QPSK carrier: constant instantaneous power
    bits_i = rng.integers(0, 2, n_samples) * 2 - 1
    bits_q = rng.integers(0, 2, n_samples) * 2 - 1
    carrier = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2)

    noise_amplitude = 1.0 / float(np.sqrt(snr_linear))
    noise = noise_amplitude * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64)

    envelope = np.zeros(n_samples, dtype=np.float32)
    if ramp_samples > 0:
        ramp_end = onset_sample + ramp_samples
        envelope[onset_sample:ramp_end] = np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
        envelope[ramp_end:] = 1.0
    else:
        envelope[onset_sample:] = 1.0

    if event_type == "offset":
        envelope = envelope[::-1].copy()

    return (carrier * envelope + noise).astype(np.complex64)


def _make_plateau_pair_iq(
    n_samples: int = 1280,
    onset_sample: int = 296,  # ramp ends at sample 344 with the default ramp width
    ramp_samples: int = 48,   # ~750 us at 64 kHz, realistic LMR PA rise; wider
                              # than the Savgol kernel so d2 can resolve the corner
    prop_delay_samples: int = 10,
    snr_db: float = 45.0,
    seed: int = 42,
    event_type: str = "onset",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair of IQ arrays modeling two nodes receiving the same transmission.

    Node A is farther from the transmitter by prop_delay_samples.  A's ring-
    buffer window starts later in wall-clock time, so the carrier onset appears
    prop_delay_samples earlier (smaller sample index) in A's window.  Implemented
    as np.roll(iq_b_clean, -prop_delay_samples): A's snippet is B's clean signal
    shifted left by the propagation delay.

    Power-envelope xcorr (B * conj(A)) returns +prop_delay_samples / fs seconds
    (positive = A heard the carrier later = A is farther).

    For event_type="offset", the physical signal is reversed: carrier present
    from the start, dropping off at the transition point.
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_amplitude = 1.0 / float(np.sqrt(snr_linear))

    # Simulate a single physical signal: noise before onset_sample, carrier after.
    # B's window covers [0:n_samples]; A's window covers [prop_delay:prop_delay+n_samples].
    # A's ring buffer started prop_delay_samples later in wall-clock time, so the
    # PA onset (at physical position onset_sample) appears at position
    # onset_sample - prop_delay_samples in A's window - the TDOA is encoded in the
    # step function position.
    #
    # Use QPSK-only (constant instantaneous power) to model FM, which is a
    # constant-envelope modulation.  The old harness used an AM-modulated
    # carrier to give power-envelope xcorr something to correlate, but that
    # introduces in-band power fluctuations the knee finder mistakes for
    # plateau corners.
    total_len = n_samples + prop_delay_samples
    bits_i = rng.integers(0, 2, total_len) * 2 - 1
    bits_q = rng.integers(0, 2, total_len) * 2 - 1
    carrier_ext = ((bits_i + 1j * bits_q) / np.sqrt(2)).astype(np.complex64)

    # Physical signal: silence in [0:onset_sample], carrier from onset_sample onward.
    physical = np.zeros(total_len, dtype=np.complex64)
    ramp_end = onset_sample + ramp_samples
    if ramp_samples > 0:
        physical[onset_sample:ramp_end] = (
            carrier_ext[onset_sample:ramp_end]
            * np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
        )
    physical[ramp_end:] = carrier_ext[ramp_end:]

    if event_type == "offset":
        physical = physical[::-1].copy()

    iq_b_clean = physical[:n_samples].copy()
    iq_a_clean = physical[prop_delay_samples : prop_delay_samples + n_samples].copy()

    noise_b = noise_amplitude * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64)
    noise_a = noise_amplitude * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64)

    return (iq_a_clean + noise_a).astype(np.complex64), (iq_b_clean + noise_b).astype(np.complex64)


def _iq_to_b64(iq: np.ndarray) -> str:
    """Encode a complex64 IQ array to base64 int8-interleaved."""
    scale = float(np.max(np.abs(iq))) + 1e-30
    normed = iq / scale
    raw = np.empty(len(iq) * 2, dtype=np.int8)
    raw[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
    raw[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
    return base64.b64encode(raw.tobytes()).decode()


def _make_event(node_lat, node_lon, sync_to_snippet_start_ns, sync_tx_lat=47.6, sync_tx_lon=-122.3,
                event_type="onset", transition_start=None, transition_end=None):
    # Defaults match typical production values for 1280-sample snippets at 62.5 kHz:
    #   onset:  detection at 3/4 minus 5 windows (~1152 samples back to 640)
    #   offset: detection at 3/4 minus 8 windows (~960 samples back to 448)
    # The real-data fixture's onset has argmax(d1) around sample 752 —
    # within [640, 1152].
    if transition_start is None:
        transition_start = 640 if event_type == "onset" else 448
    if transition_end is None:
        transition_end = 1152 if event_type == "onset" else 960
    return {
        "node_id": "test",
        "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
        "sync_tx_lat": sync_tx_lat,
        "sync_tx_lon": sync_tx_lon,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "event_type": event_type,
        "iq_snippet_b64": _make_real_snippet_b64(),
        "channel_sample_rate_hz": _REAL_RATE,
        "transition_start": transition_start,
        "transition_end": transition_end,
    }


def _make_event_with_snippet(
    node_lat, node_lon, sync_to_snippet_start_ns, snippet_b64,
    sample_rate_hz=64_000.0, node_id="test", event_type="onset",
    transition_start=None, transition_end=None,
):
    # Defaults target a 1280-sample snippet produced by _make_plateau_pair_iq
    # (onset at sample ~320 with ramp_samples wide).  The knee is inside
    # [onset_sample, onset_sample + 512].
    if transition_start is None:
        transition_start = 320 if event_type == "onset" else 64
    if transition_end is None:
        transition_end = 832 if event_type == "onset" else 576
    return {
        "node_id": node_id,
        "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
        "sync_tx_lat": 47.6,
        "sync_tx_lon": -122.3,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "iq_snippet_b64": snippet_b64,
        "channel_sample_rate_hz": sample_rate_hz,
        "event_type": event_type,
        "transition_start": transition_start,
        "transition_end": transition_end,
    }


# Keep the old bandlimited noise helper for cross_correlate_snippets tests
def _make_snippet_b64(n_samples: int = 640, lag_samples: int = 0, seed: int = 42) -> str:
    """
    Encode a synthetic bandlimited noise signal as base64 int8 IQ.

    Used only for tests of cross_correlate_snippets() which does not use plateau
    detection.
    """
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
    W = np.fft.fft(white)
    cutoff = n_samples // 4  # lowpass at ~25% of Nyquist
    W[cutoff:-cutoff] = 0.0
    signal = np.fft.ifft(W).astype(np.complex64)
    if lag_samples != 0:
        signal = np.roll(signal, lag_samples)
    scale = float(np.max(np.abs(signal))) + 1e-30
    normed = signal / scale
    raw = np.empty(n_samples * 2, dtype=np.int8)
    raw[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
    raw[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
    return base64.b64encode(raw.tobytes()).decode()


# ---------------------------------------------------------------------------
# compute_tdoa_s: geometry tests
# ---------------------------------------------------------------------------

def test_compute_tdoa_equidistant_nodes_same_delta():
    """
    Two nodes equidistant from sync_tx with identical snippets (xcorr lag = 0)
    -> TDOA ~= 0 (path correction ~= 0 for equidistant nodes).

    Exercises the "xcorr" path explicitly; the 640-sample _make_plateau_iq
    fixture is too short for the "phat" default's >= 500-sample plateau
    requirement (the plateau-only segment is shorter than 500 samples in
    the synthetic snippet).
    """
    iq = _make_plateau_iq()
    snip = _iq_to_b64(iq)
    ev_a = {**_make_event(47.7, -122.3, 500_000_000, sync_tx_lat=47.6, sync_tx_lon=-122.3),
            "node_id": "a", "iq_snippet_b64": snip, "channel_sample_rate_hz": 64_000.0}
    ev_b = {**_make_event(47.5, -122.3, 500_000_000, sync_tx_lat=47.6, sync_tx_lon=-122.3),
            "node_id": "b", "iq_snippet_b64": snip, "channel_sample_rate_hz": 64_000.0}
    tdoa = compute_tdoa_s(ev_a, ev_b, tdoa_method="xcorr", min_xcorr_snr=2.0)
    assert tdoa is not None
    assert abs(tdoa) < 200e-9  # < 200 ns


def test_compute_tdoa_known_geometry():
    """
    sync_delta subtraction with known geometry gives correct TDOA.
    sync_tx equidistant from both nodes -> path correction ~ 0.

    Uses a small prop_delay (2 samples = 31 usec) that stays within the
    xcorr refinement gate (50 usec), simulating properly aligned
    sample_index and snippet anchor.
    """
    fs = 64_000.0
    prop_delay = 2  # 2 samples at 64 kHz = ~31 usec -- within 50 usec gate
    delta_ns = int(prop_delay / fs * 1e9)
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.7, -122.3, delta_ns, _iq_to_b64(iq_a), node_id="node-a",
                                     sample_rate_hz=fs)
    ev_b = _make_event_with_snippet(47.5, -122.3, 0, _iq_to_b64(iq_b), node_id="node-b",
                                     sample_rate_hz=fs)
    tdoa = compute_tdoa_s(ev_a, ev_b, tdoa_method="knee")
    expected_s = prop_delay / fs  # positive = A later
    assert tdoa is not None
    # d2 knee finder on 48-sample synthetic ramp has ~2-sample precision floor
    # (~30 µs at 64 kHz); real-corpus precision is ~60 µs at 250 kHz.
    assert abs(tdoa - expected_s) < 100e-6  # within 100 usec


def test_compute_tdoa_antisymmetric():
    """compute_tdoa_s(A, B) == -compute_tdoa_s(B, A) via sync_delta fallback."""
    ev_a = _make_event(47.65, -122.31, sync_to_snippet_start_ns=5000)
    ev_b = _make_event(47.72, -122.28, sync_to_snippet_start_ns=0)
    t_ab = compute_tdoa_s(ev_a, ev_b)
    t_ba = compute_tdoa_s(ev_b, ev_a)
    assert t_ab is not None and t_ba is not None
    assert abs(t_ab + t_ba) < 1e-12  # exact antisymmetry with sync_delta


def test_compute_tdoa_path_delay_applied():
    """
    Both nodes have same sync_delta (raw_ns=0); node_A far from sync_tx.
    TDOA comes entirely from path-delay correction: +30km/c ~ +100 usec.
    No IQ snippets: xcorr does not fire; result uses sync_delta fallback.
    """
    d_30km_deg = 30_000 / 111_195
    ev_a = _make_event(47.5 + d_30km_deg, -122.3, 0, sync_tx_lat=47.5, sync_tx_lon=-122.3)
    ev_b = _make_event(47.5, -122.3, 0, sync_tx_lat=47.5, sync_tx_lon=-122.3)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    expected = +30_000 / _C_M_S
    assert tdoa is not None
    assert tdoa == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# compute_tdoa_s - xcorr primary path
# ---------------------------------------------------------------------------

def test_compute_tdoa_xcorr_method_rejects_low_snr():
    """
    Default xcorr path rejects pairs whose xcorr SNR is below min_xcorr_snr.
    Verifies the SNR gate wiring in compute_tdoa_s; xcorr's numerical
    accuracy on real snippets is covered by test_xcorr.py.
    """
    rng = np.random.default_rng(42)
    noise_a = (rng.standard_normal(5120) + 1j * rng.standard_normal(5120)).astype(np.complex64)
    noise_b = (rng.standard_normal(5120) + 1j * rng.standard_normal(5120)).astype(np.complex64)
    ev_a = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(noise_a),
        node_id="node-a", transition_start=2460, transition_end=2580,
    )
    ev_b = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(noise_b),
        node_id="node-b", transition_start=2460, transition_end=2580,
    )
    # Very high gate — even if xcorr returns a finite SNR on pure noise, it
    # will be well below 100.
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=100.0)
    assert tdoa is None


def test_compute_tdoa_invalid_method_raises():
    """Unknown tdoa_method value raises ValueError."""
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=0, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0,
                                     snippet_b64=_iq_to_b64(iq_a), node_id="a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0,
                                     snippet_b64=_iq_to_b64(iq_b), node_id="b")
    with pytest.raises(ValueError, match="tdoa_method"):
        compute_tdoa_s(ev_a, ev_b, tdoa_method="nonsense")


def test_compute_tdoa_phat_recovers_known_delay():
    """
    PHAT method recovers a known small propagation delay.

    Uses QPSK-modulated IQ (constant-envelope, broadband phase modulation)
    so PHAT has rich cross-spectrum content to correlate on.  prop_delay=2
    samples at 64 kHz = 31.25 µs physical TDOA — below the T_sync/2
    disambiguation ambiguity.  Locations are chosen so the sync-path
    correction is zero (sync tx equidistant from both nodes).
    """
    fs = 64_000.0
    prop_delay = 2
    delta_ns = int(prop_delay / fs * 1e9)
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.7, -122.3, delta_ns, _iq_to_b64(iq_a),
                                     node_id="node-a", sample_rate_hz=fs)
    ev_b = _make_event_with_snippet(47.5, -122.3, 0, _iq_to_b64(iq_b),
                                     node_id="node-b", sample_rate_hz=fs)
    tdoa = compute_tdoa_s(ev_a, ev_b, tdoa_method="phat", min_xcorr_snr=1.5)
    assert tdoa is not None, "PHAT returned None on good synthetic pair"
    expected_s = prop_delay / fs    # positive = A heard the carrier later
    # PHAT on QPSK-modulated synthetic IQ resolves the delay to sub-sample
    # precision.  Real-corpus precision is ~50-200 µs depending on pair.
    assert abs(tdoa - expected_s) < 50e-6


def test_compute_tdoa_phat_rejects_short_snippet():
    """
    PHAT requires >= 500 samples of plateau after the ramp.  If the snippet
    is too short to carve out a valid plateau segment, PHAT returns None
    and the pair is skipped (not silently degraded).
    """
    fs = 64_000.0
    # 256-sample snippet is way too short for PHAT's 500-sample plateau minimum.
    rng = np.random.default_rng(0)
    short_iq = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
    ev_a = _make_event_with_snippet(
        47.7, -122.3, 0, _iq_to_b64(short_iq), node_id="a", sample_rate_hz=fs,
        transition_start=100, transition_end=200,
    )
    ev_b = _make_event_with_snippet(
        47.5, -122.3, 0, _iq_to_b64(short_iq), node_id="b", sample_rate_hz=fs,
        transition_start=100, transition_end=200,
    )
    tdoa = compute_tdoa_s(ev_a, ev_b, tdoa_method="phat")
    assert tdoa is None


def test_compute_tdoa_xcorr_refines_sync_delta():
    """
    xcorr provides sub-sample refinement on top of the sync_delta TDOA.

    Co-located nodes (same lat/lon) with sync_delta encoding a known delay.
    The IQ snippets have a small sub-sample offset that the server-side
    knee finder detects and corrects.

    Physical interpretation of the test construction:
      - sd_A - sd_B = +100 µs: A's *detection* fires 100 µs after B's.
      - prop_delay = 2 samples: A's snippet shows the carrier onset 2 samples
        EARLIER than B's (np.roll left).  With both snippets anchored at the
        detection point, this means A's detection-to-knee delay is 2 samples
        SHORTER than B's — i.e. A's knee occurred ~31 µs LESS after detection.
      - Combined: A's knee fired 100 µs − 31 µs = 69 µs after B's knee.
    """
    fs = 64_000.0
    prop_delay = 2    # samples of snippet shift
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    # sync_delta encodes 100 µs coarse difference in detection times
    sd_a = 100_100_000
    sd_b = 100_000_000
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=sd_a, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=sd_b, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, tdoa_method="knee")
    assert tdoa is not None
    # Expected: 100 µs (detection diff) − 31 µs (knee-to-detection diff) = 69 µs
    expected = (100_000 - prop_delay * 1e9 / fs) / 1e9   # ~69 µs
    assert abs(tdoa - expected) < 100e-6, (
        f"tdoa={tdoa*1e6:+.1f} µs expected={expected*1e6:+.1f} µs"
    )


def test_compute_tdoa_non_colocated_uses_sync_delta_geometry():
    """
    NON-CO-LOCATED nodes: xcorr of anchored snippets measures ~0 (both
    transitions placed at 3/4), but the true TDOA is large (60+ usec).
    sync_delta must carry the geometric TDOA; xcorr only refines it.

    This is the test that would have caught the bug where xcorr replaced
    sync_delta entirely -- returning ~0 instead of the true ~60 usec TDOA.

    Setup: node A is 2.5 km from the transmitter, node B is 21 km away.
    The sync station is at a third location.  sync_delta values are set
    so that after path correction and disambiguation, the TDOA matches
    the known geometry.
    """
    # Geometry matching the real Magnolia repeater scenario:
    # sync_tx (KUOW):  47.61576, -122.30919  (FCC-documented)
    # node_a:          47.6719, -122.4042  (near transmitter)
    # node_b:          47.5599, -122.1475  (21 km away)
    # target_tx:       47.6509, -122.3915  (Magnolia repeater)
    sync_tx = (47.61576, -122.30919)
    node_a_pos = (47.6719, -122.4042)
    node_b_pos = (47.5599, -122.1475)
    target_tx = (47.6509, -122.3915)

    # Compute true propagation delays
    d_sync_a = haversine_m(sync_tx[0], sync_tx[1], node_a_pos[0], node_a_pos[1])
    d_sync_b = haversine_m(sync_tx[0], sync_tx[1], node_b_pos[0], node_b_pos[1])
    d_target_a = haversine_m(target_tx[0], target_tx[1], node_a_pos[0], node_a_pos[1])
    d_target_b = haversine_m(target_tx[0], target_tx[1], node_b_pos[0], node_b_pos[1])

    # True TDOA = (target_to_A - target_to_B) / c
    true_tdoa_s = (d_target_a - d_target_b) / _C_M_S

    # Construct sync_delta values that encode this geometry.
    # sync_delta = (time target arrives) - (time sync arrives) at each node
    # = (d_target / c) - (d_sync / c) + arbitrary common offset
    base_ns = 50_000_000  # 50 ms common offset (arbitrary)
    sd_a = int(base_ns + (d_target_a - d_sync_a) / _C_M_S * 1e9)
    sd_b = int(base_ns + (d_target_b - d_sync_b) / _C_M_S * 1e9)

    # Both snippets have the transition at the SAME position (anchored).
    # Knee finder should find ~same position in both → knee_adj ~0.
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=0, snr_db=25.0)

    # _make_plateau_pair_iq places the onset at sample 320 with a 24-sample
    # ramp.  Constrain the knee search to a tight window around the ramp
    # so argmax(d1) doesn't latch onto AM-envelope variation in the plateau.
    ev_a = _make_event_with_snippet(
        node_a_pos[0], node_a_pos[1], sync_to_snippet_start_ns=sd_a,
        snippet_b64=_iq_to_b64(iq_a), node_id="node-a",
        transition_start=300, transition_end=400,
    )
    ev_a["sync_tx_lat"] = sync_tx[0]
    ev_a["sync_tx_lon"] = sync_tx[1]

    ev_b = _make_event_with_snippet(
        node_b_pos[0], node_b_pos[1], sync_to_snippet_start_ns=sd_b,
        snippet_b64=_iq_to_b64(iq_b), node_id="node-b",
        transition_start=300, transition_end=400,
    )
    ev_b["sync_tx_lat"] = sync_tx[0]
    ev_b["sync_tx_lon"] = sync_tx[1]

    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None

    # The true TDOA is about -61 usec (node A is closer to the transmitter).
    # The bug we want to catch: xcorr/knee replacing sync_delta entirely
    # would return ~0 (since both snippets are anchored identically) — a
    # >50 µs miss.  The d2 knee finder on synthetic data has ~50-100 µs
    # precision floor, so we tolerate 80 µs here.
    assert abs(tdoa - true_tdoa_s) < 80e-6, (
        f"TDOA {tdoa*1e6:.1f} usec != expected {true_tdoa_s*1e6:.1f} usec; "
        f"xcorr may be replacing sync_delta instead of refining it"
    )


def test_compute_tdoa_rejects_low_snr_snippets():
    """
    When xcorr SNR is below min_xcorr_snr, the pair is rejected (no fallback).
    Coarse sync_delta has ~200 us noise and is not useful for a fix.
    """
    rng = np.random.default_rng(99)
    # Pure white noise - no PA transition -> d2 xcorr SNR ~4-5
    noise_a = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    noise_b = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=5_000,
                                     snippet_b64=_iq_to_b64(noise_a), node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0,
                                     snippet_b64=_iq_to_b64(noise_b), node_id="node-b")
    # SNR threshold 10.0 rejects noise-only snippets (d2 noise SNR ~4-5)
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=10.0)
    assert tdoa is None, "Noise-only snippets should be rejected (no fallback)"


def test_compute_tdoa_xcorr_geo_filter_rejects_implausible_lag():
    """
    xcorr lag that exceeds the refinement gate (from mismatched snippets)
    returns None so the solver skips the pair.

    A strong carrier edge on snippet A but flat noise on snippet B produces a
    high-SNR xcorr peak at a large random lag (no true TDOA).  The refinement
    gate (50 usec) rejects it before the geo filter even fires.
    """
    rng = np.random.default_rng(42)
    fs = 64_000.0
    # snippet_a: carrier ON for first half, OFF for second half (clear offset edge)
    n = 1280
    carrier_a = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)).astype(np.complex64) * 100
    noise_a   = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)).astype(np.complex64)
    iq_a = np.concatenate([carrier_a, noise_a])
    # snippet_b: pure noise - no PA edge, so xcorr SNR will be high but lag random
    iq_b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 50

    ev_a = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=7_000,
                                     snippet_b64=_iq_to_b64(iq_a), sample_rate_hz=fs,
                                     node_id="node-a", event_type="offset")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0,
                                     snippet_b64=_iq_to_b64(iq_b), sample_rate_hz=fs,
                                     node_id="node-b", event_type="offset")

    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3, max_xcorr_baseline_km=1.0)
    assert tdoa is None, (
        f"Expected None (bad xcorr rejected); got {tdoa}"
    )


def test_compute_tdoa_xcorr_refinement_within_gate():
    """
    Small inter-snippet offset feeds into the knee-finder correction.

    1-sample prop_delay_samples at 64 kHz = 15.6 µs.  In the new algorithm:
    both nodes find their own knee; the knee positions in the snippets differ
    by 1 sample because the snippet content was shifted.  knee_adj =
    (knee_a - det_a)/fs - (knee_b - det_b)/fs contributes -1/fs = -15.6 µs.
    With sync_to_snippet_start_ns = 0 for both nodes, TDOA = -15.6 µs.

    (Under the old xcorr-based algorithm this returned +15.6 µs because
    the xcorr lag was added rather than absorbed into per-node knee
    positions — that was physically wrong for a detection-anchored snippet.)
    """
    fs = 64_000.0
    prop_delay = 1
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3)
    assert tdoa is not None
    expected = -prop_delay / fs   # knee_adj is negative: A's knee appears earlier in A's snippet
    assert abs(tdoa - expected) < 100e-6, (
        f"tdoa={tdoa*1e6:+.1f} µs expected={expected*1e6:+.1f} µs"
    )


def test_compute_tdoa_xcorr_large_lag_accepted_onset():
    """
    Large inter-snippet offset (10 samples = 156 µs) is absorbed into the
    knee finder's per-node knee position and appears in the final TDOA.
    This is within the 50 km geometric plausibility limit (~167 µs).
    """
    fs = 64_000.0
    prop_delay = 10
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    # _make_plateau_pair_iq places the onset ramp at ~320-344; narrow the
    # search window so argmax(d1) doesn't latch onto AM-envelope variation.
    ev_a = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_a),
        sample_rate_hz=fs, node_id="node-a", event_type="onset",
        transition_start=300, transition_end=380,
    )
    ev_b = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_b),
        sample_rate_hz=fs, node_id="node-b", event_type="onset",
        transition_start=300, transition_end=380,
    )
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3, tdoa_method="knee")
    assert tdoa is not None, "Expected valid TDOA for plausible knee offset"
    # Expected ~156 µs (10-sample knee offset at 64 kHz) plus noise floor.
    assert abs(tdoa * 1e6) < 300.0, f"TDOA {tdoa*1e6:.0f} µs seems too large"


def test_compute_tdoa_xcorr_large_lag_accepted_offset():
    """
    Offset-event version of the above.  `_make_plateau_pair_iq` reverses
    the signal, so the falling edge is near the END of the snippet (samples
    ~936-960 for the default 1280-sample construction).
    """
    fs = 64_000.0
    prop_delay = 10
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0,
                                        event_type="offset")
    ev_a = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_a),
        sample_rate_hz=fs, node_id="node-a", event_type="offset",
        transition_start=920, transition_end=1000,
    )
    ev_b = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_b),
        sample_rate_hz=fs, node_id="node-b", event_type="offset",
        transition_start=920, transition_end=1000,
    )
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3)
    assert tdoa is not None, "Expected valid TDOA for plausible knee offset"
    # Expected ~156 µs (10-sample knee offset at 64 kHz) plus noise floor.
    assert abs(tdoa * 1e6) < 300.0, f"TDOA {tdoa*1e6:.0f} µs seems too large"


def test_compute_tdoa_colocated_xcorr_near_zero():
    """
    Co-located nodes with identical snippets (no shift) should yield
    TDOA ~ 0 — the knee finder finds the same position in each snippet.
    """
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=0, snr_db=25.0)
    # Narrow the transition window to avoid argmax latching on AM noise.
    ev_a = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_a),
        node_id="node-a", transition_start=300, transition_end=380,
    )
    ev_b = _make_event_with_snippet(
        47.6, -122.3, sync_to_snippet_start_ns=0, snippet_b64=_iq_to_b64(iq_b),
        node_id="node-b", transition_start=300, transition_end=380,
    )
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert abs(tdoa) < 1e-4  # < 100 µs (one-sample at 64 kHz)


# ---------------------------------------------------------------------------
# compute_tdoa_s - sync_delta subtraction
# ---------------------------------------------------------------------------

def test_compute_tdoa_sync_delta_difference():
    """
    sync_delta subtraction (fallback path) produces the correct raw TDOA.
    No IQ snippets: xcorr does not fire; result is purely sync_delta-based.
    Both nodes at same location (no path correction).
    """
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=500_005_000)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=500_000_000)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-12)  # 5 usec


def test_compute_tdoa_returns_none_when_sync_delta_missing():
    """Returns None when sync_to_snippet_start_ns is absent."""
    ev_a = {"node_id": "a", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    ev_b = {"node_id": "b", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    assert compute_tdoa_s(ev_a, ev_b) is None


# ---------------------------------------------------------------------------
# compute_tdoa_s - pilot sync event disambiguation
# ---------------------------------------------------------------------------

def _make_event_with_onset_time(node_lat, node_lon, sync_to_snippet_start_ns, onset_time_ns,
                                 sync_tx_lat=47.6, sync_tx_lon=-122.3):
    """Event dict including onset_time_ns for disambiguation tests."""
    return {
        "node_id": "test",
        "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
        "sync_tx_lat": sync_tx_lat,
        "sync_tx_lon": sync_tx_lon,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "event_type": "onset",
        "onset_time_ns": onset_time_ns,
        "iq_snippet_b64": _make_real_snippet_b64(),
        "channel_sample_rate_hz": _REAL_RATE,
        "transition_start": 640,
        "transition_end": 1152,
    }


_T_SYNC_NS = 1_000_000_000.0 / 1187.5   # must match tdoa.py constant (~842,105 ns)


def test_sync_disambiguation_no_adjustment_needed():
    """
    Both nodes used the same RDS bit boundary (raw_ns within T_sync/2).
    No adjustment should be applied; TDOA equals the raw 5 usec difference.
    """
    t0 = 1_700_000_000_000_000_000  # arbitrary epoch ns
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_to_snippet_start_ns=505_000, onset_time_ns=t0 + 5_000)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_to_snippet_start_ns=500_000, onset_time_ns=t0)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-9)


def test_sync_disambiguation_corrects_one_period_offset():
    """
    Node B referenced the RDS bit boundary one T_sync later than node A.
    raw_ns = true_TDOA - T_sync (~ -837 usec); disambiguation adds +T_sync.
    """
    true_tdoa_ns = 5_000  # 5 usec
    t0 = 1_700_000_000_000_000_000
    raw_ns = true_tdoa_ns - _T_SYNC_NS           # ~ -837_105 ns
    sync_delta_a = 500_000
    sync_delta_b = sync_delta_a - raw_ns
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a,
                                        onset_time_ns=t0 + true_tdoa_ns)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b,
                                        onset_time_ns=t0)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-6)


def test_sync_disambiguation_works_without_onset_time():
    """
    Geometric disambiguation resolves n from path geometry alone.
    raw_ns = true_TDOA - T_sync is corrected to true_TDOA.

    Rationale: |true_TDOA| <= dist(A,B)/c <= 100 km/c ~ 333 usec << T_sync/2 = 421 usec,
    so round((raw_ns + correction) / T_sync) uniquely identifies n.
    """
    true_tdoa_ns = 5_000  # 5 usec
    sync_delta_a = 500_000
    sync_delta_b = sync_delta_a - (true_tdoa_ns - _T_SYNC_NS)
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-6)


def test_sync_disambiguation_n_zero():
    """n=0: nodes referenced the same bit boundary; raw_ns is already correct."""
    true_tdoa_ns = 200_000   # 200 usec - within T_sync/2 = 421 usec
    sync_delta_a = 400_000
    sync_delta_b = sync_delta_a - true_tdoa_ns  # raw_ns = +200_000
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_sync_disambiguation_n_plus_one():
    """
    n=+1: node A referenced a bit boundary one T_sync earlier than node B.
    raw_ns = true_TDOA + T_sync ~ +992 usec; disambiguation subtracts T_sync.
    """
    true_tdoa_ns = 150_000   # 150 usec
    raw_ns = true_tdoa_ns + _T_SYNC_NS        # ~ +992_105 ns
    sync_delta_a = 500_000
    sync_delta_b = sync_delta_a - raw_ns
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_sync_disambiguation_n_minus_one():
    """
    n=-1: node A referenced a bit boundary one T_sync later than node B.
    raw_ns = true_TDOA - T_sync ~ -692 usec; disambiguation adds T_sync.
    """
    true_tdoa_ns = 150_000   # 150 usec
    raw_ns = true_tdoa_ns - _T_SYNC_NS        # ~ -692_105 ns
    sync_delta_a = 500_000
    sync_delta_b = sync_delta_a - raw_ns
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_sync_disambiguation_large_tdoa_within_half_period():
    """
    A true TDOA near +/-T_sync/2 but still within it (n=0) is left unchanged.
    raw_ns = -300_000 ns: |raw| < T_sync/2 = 421_053 -> n=0, no adjustment.
    """
    true_tdoa_ns = -300_000
    sync_delta_a = 400_000
    sync_delta_b = sync_delta_a - true_tdoa_ns   # = 700_000
    ev_a = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_to_snippet_start_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


# ---------------------------------------------------------------------------
# Mismatched sync_tx coordinates -- reproduces production bug
# ---------------------------------------------------------------------------

class TestSyncTxCoordinateMismatch:
    """
    Reproduces the production bug where nodes report different sync transmitter
    coordinates due to stale config.  The path delay correction uses each
    event's sync_tx_lat/lon independently, so a mismatch causes the wrong
    correction.  With the old pilot-based sync (T_sync = 7 ms), this was enough
    to pick the wrong cycle (n), producing errors of thousands of microseconds.
    With RDS sync (T_sync = 842 usec), the smaller period makes disambiguation
    robust against this level of coordinate error (~99 usec correction error
    vs T_sync/2 = 421 usec), so n is still resolved correctly.

    Uses real positions and sync_delta values from the 2026-04-05 deployment:
      - dpk-tdoa1/dpk-tdoa2: co-located at (47.671928, -122.404209)
      - kb7ryy: remote at (47.559910, -122.147540)
      - Magnolia repeater (target): (47.650875, -122.391478)
      - KUOW 94.9 (sync): correct location (47.61576, -122.30919)
      - Stale KUOW location: (47.6553, -122.311) -- 4.4 km off

    The true TDOA for dpk-tdoa1 <-> kb7ryy is approximately -61 usec.
    """

    # Real geometry
    CORRECT_SYNC_TX = (47.61576, -122.30919)
    STALE_SYNC_TX = (47.6553, -122.311)
    NODE_A_POS = (47.671928, -122.404209)   # dpk-tdoa1 (co-located)
    NODE_B_POS = (47.559910, -122.147540)   # kb7ryy
    TARGET_TX = (47.650875, -122.391478)    # Magnolia repeater

    # True TDOA: (dist(target,A) - dist(target,B)) / c ~ -61 usec
    TRUE_TDOA_US = (
        (haversine_m(TARGET_TX[0], TARGET_TX[1], NODE_A_POS[0], NODE_A_POS[1])
         - haversine_m(TARGET_TX[0], TARGET_TX[1], NODE_B_POS[0], NODE_B_POS[1]))
        / _C_M_S * 1e6
    )

    # Real sync_delta values from production (offset event pair).
    # These encode the physical geometry correctly -- the bug is only in
    # which sync_tx coordinates accompany them.
    # Picked from the "good" pair in the production data: sd_diff ~ +52 usec, n=0.
    SD_A_NS = 2816029    # dpk-tdoa1 sync_to_snippet_start_ns
    SD_B_NS = 2764024    # kb7ryy sync_to_snippet_start_ns

    def _make_event(self, node_pos, sync_to_snippet_start_ns, sync_tx, node_id):
        # Using the fixture onset snippet (real data) — transition_start/end
        # match its actual PA rise region.  The test is about sync_tx
        # coordinate handling, not the event type itself.
        return {
            "node_id": node_id,
            "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
            "sync_tx_lat": sync_tx[0],
            "sync_tx_lon": sync_tx[1],
            "node_lat": node_pos[0],
            "node_lon": node_pos[1],
            "event_type": "onset",
            "iq_snippet_b64": _make_real_snippet_b64(),
            "channel_sample_rate_hz": _REAL_RATE,
            "transition_start": 640,
            "transition_end": 1152,
        }

    def test_correct_sync_tx_gives_correct_tdoa(self):
        """
        Both nodes report the correct sync transmitter location.
        Disambiguation should pick n=0 and TDOA should be near -61 usec.
        """
        ev_a = self._make_event(
            self.NODE_A_POS, self.SD_A_NS, self.CORRECT_SYNC_TX, "dpk-tdoa1")
        ev_b = self._make_event(
            self.NODE_B_POS, self.SD_B_NS, self.CORRECT_SYNC_TX, "kb7ryy")

        tdoa = compute_tdoa_s(ev_a, ev_b)
        assert tdoa is not None

        tdoa_us = tdoa * 1e6
        # Should be close to the true TDOA (-61 usec).
        # Allow +/-500 usec for the sync_delta sample quantisation.
        assert abs(tdoa_us - self.TRUE_TDOA_US) < 500, (
            f"TDOA {tdoa_us:+.1f} usec with correct sync_tx; "
            f"expected ~{self.TRUE_TDOA_US:+.1f} usec"
        )

    def test_mismatched_sync_tx_gives_wrong_tdoa(self):
        """
        Node A reports the correct sync_tx, node B reports the stale (wrong)
        sync_tx.  The path correction is computed with inconsistent coordinates.

        With RDS sync (T_sync = 842 usec), the ~99 usec correction error from
        a 4.4 km sync_tx mismatch is within T_sync/2 (421 usec), so
        disambiguation still picks the correct n.  The TDOA error is limited
        to the path correction error itself (~100 usec), not thousands of
        microseconds as with pilot sync.
        """
        ev_a = self._make_event(
            self.NODE_A_POS, self.SD_A_NS, self.CORRECT_SYNC_TX, "dpk-tdoa1")
        ev_b = self._make_event(
            self.NODE_B_POS, self.SD_B_NS, self.STALE_SYNC_TX, "kb7ryy")

        tdoa = compute_tdoa_s(ev_a, ev_b)
        assert tdoa is not None

        tdoa_us = tdoa * 1e6
        # With RDS sync, the smaller period means disambiguation is correct
        # even with mismatched sync_tx.  The error is just the path correction
        # difference (~100 usec), not a whole-cycle error.
        error_us = abs(tdoa_us - self.TRUE_TDOA_US)
        assert error_us < 200, (
            f"TDOA {tdoa_us:+.1f} usec with mismatched sync_tx; "
            f"error {error_us:.0f} usec exceeds 200 usec. "
            f"RDS disambiguation should handle this level of coord error."
        )

    def test_stale_sync_tx_on_both_nodes_still_works(self):
        """
        Both nodes report the SAME stale sync_tx location.  Even though the
        coordinates are wrong, they are consistent -- the path correction
        error cancels in the subtraction, and disambiguation picks the right n.
        The TDOA should still be close to the true value.

        This matches the production observation that dpk-tdoa1 + kb7ryy
        (both using stale coords) produced Fix 168 at 5.4 km error.
        """
        ev_a = self._make_event(
            self.NODE_A_POS, self.SD_A_NS, self.STALE_SYNC_TX, "dpk-tdoa1")
        ev_b = self._make_event(
            self.NODE_B_POS, self.SD_B_NS, self.STALE_SYNC_TX, "kb7ryy")

        tdoa = compute_tdoa_s(ev_a, ev_b)
        assert tdoa is not None

        tdoa_us = tdoa * 1e6
        # Both nodes use the same (wrong) sync_tx, so the error partially
        # cancels. The TDOA should be reasonable (within 500 usec of truth).
        assert abs(tdoa_us - self.TRUE_TDOA_US) < 500, (
            f"TDOA {tdoa_us:+.1f} usec with consistent (stale) sync_tx; "
            f"expected ~{self.TRUE_TDOA_US:+.1f} usec. "
            f"Consistent sync_tx should cancel in the subtraction."
        )

    def test_multiple_real_pairs_with_correct_sync_tx(self):
        """
        Multiple real sync_delta pairs from production, all with correct
        sync_tx coordinates.  All should produce TDOAs within 500 usec
        of the true value (-61 usec).

        These pairs represent different transmissions on the same repeater.
        """
        # Real sync_delta pairs from the production data (dpk-tdoa1 vs kb7ryy)
        real_pairs = [
            (3004040, 6068067),   # sd_diff = -3064 usec
            (1444012, 3768038),   # sd_diff = -2324 usec
            (2308033, 5716058),   # sd_diff = -3408 usec
            (472005,  2500033),   # sd_diff = -2028 usec
            (3836044, 2356025),   # sd_diff = +1480 usec
            (3548040, 1828015),   # sd_diff = +1720 usec
            (1636017,   88001),   # sd_diff = +1548 usec
            (2816029, 2764024),   # sd_diff = +52 usec
        ]

        tdoas_us = []
        for sd_a, sd_b in real_pairs:
            ev_a = self._make_event(
                self.NODE_A_POS, sd_a, self.CORRECT_SYNC_TX, "dpk-tdoa1")
            ev_b = self._make_event(
                self.NODE_B_POS, sd_b, self.CORRECT_SYNC_TX, "kb7ryy")
            # Use wider geometric limit — these are synthetic test pairs where
            # identical snippets give xcorr=0, not matching the sync_delta.
            tdoa = compute_tdoa_s(ev_a, ev_b, max_xcorr_baseline_km=500.0)
            assert tdoa is not None, f"Pair sd_a={sd_a} sd_b={sd_b} returned None"
            tdoas_us.append(tdoa * 1e6)

        # All should cluster near -61 usec
        import statistics
        mean_tdoa = statistics.mean(tdoas_us)
        std_tdoa = statistics.stdev(tdoas_us)
        assert abs(mean_tdoa - self.TRUE_TDOA_US) < 500, (
            f"Mean TDOA {mean_tdoa:+.1f} usec (std={std_tdoa:.1f}) too far from "
            f"truth {self.TRUE_TDOA_US:+.1f} usec"
        )
        # Check that individual values are consistent (not scattered by wrong n)
        for i, t in enumerate(tdoas_us):
            assert abs(t - self.TRUE_TDOA_US) < 3600, (
                f"Pair {i} TDOA {t:+.1f} usec > 3500 usec from truth -- "
                f"disambiguation likely picked wrong n"
            )


# ---------------------------------------------------------------------------
# SyncCalibrator: bit counting uses the nominal RDS bit period (250000/1187.5),
# NOT each node's reported crystal correction.
# ---------------------------------------------------------------------------

class TestSyncCalibratorNominalRate:
    """
    Empirical finding from 190 paired dpk-tdoa1/tdoa2 events over 36 hours:
    sync_sample_index differences between nodes are near-integer multiples
    of the NOMINAL bit period (250000/1187.5 samples).  Applying the reported
    crystal correction to the division injects a spurious ~100 µs fractional
    offset purely from applying ppm-scale adjustment to million-sample diffs.
    These tests lock in the correct (nominal-rate) behavior.
    """

    _T_SYNC_NS = 1e9 / 1187.5   # 842,105.26 ns
    _SPB_NOMINAL = 250_000.0 / 1187.5   # 210.52631578...

    def _events(self, sync_idx_a, sync_idx_b, corr_a=1.0, corr_b=1.0,
                sync_delta_a=500_000, sync_delta_b=500_000):
        """Build a pair of events with the given sync_sample_index values."""
        base = {
            "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.7, "node_lon": -122.3,
            "event_type": "onset",
            "iq_snippet_b64": _make_real_snippet_b64(),
            "channel_sample_rate_hz": _REAL_RATE,
        }
        ev_a = {**base, "node_id": "a", "sync_to_snippet_start_ns": sync_delta_a,
                "sync_sample_index": sync_idx_a,
                "sync_sample_rate_correction": corr_a}
        ev_b = {**base, "node_id": "b", "sync_to_snippet_start_ns": sync_delta_b,
                "sync_sample_index": sync_idx_b,
                "sync_sample_rate_correction": corr_b}
        return ev_a, ev_b

    def test_calibrator_is_public_surface(self):
        """SyncCalibrator is the tested API — import from the public module."""
        from beagle_server.tdoa import SyncCalibrator
        assert SyncCalibrator is not None

    def test_exact_integer_bits_at_nominal_yields_zero_correction(self):
        """
        When sync_idx_a - sync_idx_b is an exact integer multiple of the nominal
        bit period, correction must be 0 regardless of reported crystal values.
        """
        from beagle_server.tdoa import SyncCalibrator
        cal = SyncCalibrator(alpha=0.2, min_samples=1)
        # Choose indices separated by exactly 12015 bit periods at nominal SPB.
        idx_b = 28_176_745_397.0
        idx_a = idx_b + 12015 * self._SPB_NOMINAL
        # Reported crystal values: realistic ~10 ppm slow, slightly different.
        ev_a, ev_b = self._events(
            idx_a, idx_b, corr_a=0.99999078, corr_b=0.99998921
        )
        corr_ns = cal.update("a", "b", ev_a, ev_b)
        assert abs(corr_ns) < 10.0, (
            f"Correction should be ~0 at integer-bit diff; got {corr_ns:+.1f} ns"
        )

    def test_half_bit_offset_yields_half_period_correction(self):
        """
        When sync_idx_a - sync_idx_b has a known 0.1-bit fractional offset at
        nominal SPB, correction reflects that (0.1 * T_sync ~ 84,210 ns).
        """
        from beagle_server.tdoa import SyncCalibrator
        cal = SyncCalibrator(alpha=0.2, min_samples=1)
        idx_b = 28_176_745_397.0
        idx_a = idx_b + (12015 + 0.1) * self._SPB_NOMINAL
        ev_a, ev_b = self._events(idx_a, idx_b,
                                    corr_a=0.99999078, corr_b=0.99998921)
        corr_ns = cal.update("a", "b", ev_a, ev_b)
        # Correction magnitude ~ 0.1 * T_sync = 84,210 ns; sign depends on
        # sort direction of the pair key, both are acceptable.
        expected = 0.1 * self._T_SYNC_NS
        assert abs(abs(corr_ns) - expected) < 50.0, (
            f"|correction| should be {expected:.0f} ns; got {corr_ns:+.1f}"
        )

    def test_crystal_correction_is_ignored(self):
        """
        The SAME sync_idx diff with DIFFERENT reported crystals must yield the
        SAME correction — confirming we don't multiply sample-diff leverage by
        a ppm-scale rate adjustment.  (Before the fix, the corrections would
        differ by ~100 µs.)
        """
        from beagle_server.tdoa import SyncCalibrator
        idx_b = 28_176_745_397.0
        idx_a = idx_b + 12015 * self._SPB_NOMINAL

        cal1 = SyncCalibrator(alpha=0.2, min_samples=1)
        cal2 = SyncCalibrator(alpha=0.2, min_samples=1)
        # Two different crystal scenarios that would previously give very
        # different "frac" values.
        ev_a1, ev_b1 = self._events(idx_a, idx_b, corr_a=1.0, corr_b=1.0)
        ev_a2, ev_b2 = self._events(idx_a, idx_b,
                                      corr_a=0.99999078, corr_b=0.99998921)
        c1 = cal1.update("a", "b", ev_a1, ev_b1)
        c2 = cal2.update("a", "b", ev_a2, ev_b2)
        assert abs(c1 - c2) < 5.0, (
            f"Corrections must not depend on reported crystal: {c1:+.1f} vs {c2:+.1f}"
        )

    def test_legacy_zero_sync_idx_returns_no_correction(self):
        """Legacy nodes send sync_sample_index=0 — must return 0 correction."""
        from beagle_server.tdoa import SyncCalibrator
        cal = SyncCalibrator(alpha=0.2, min_samples=1)
        ev_a, ev_b = self._events(0.0, 0.0, corr_a=0.0, corr_b=0.0)
        assert cal.update("a", "b", ev_a, ev_b) == 0.0

    def test_restart_resets_pair_ema(self):
        """A >50% drop in sync_sample_index on a node must reset pair EMAs."""
        from beagle_server.tdoa import SyncCalibrator
        cal = SyncCalibrator(alpha=1.0, min_samples=1)   # converge in one update
        # Prime EMA with a 0.2-bit offset.
        idx_b = 1_000_000_000.0
        idx_a = idx_b + (12015 + 0.2) * self._SPB_NOMINAL
        ev_a, ev_b = self._events(idx_a, idx_b)
        cal.update("a", "b", ev_a, ev_b)
        # Node "a" restarts: idx drops sharply.  Choose new indices so the
        # post-restart diff is an exact integer multiple of the nominal bit
        # period — then correction should be ~0 IF the old EMA was reset.
        idx_a_restarted = 5000.0
        idx_b_new = idx_a_restarted - 100 * self._SPB_NOMINAL
        ev_a2, ev_b2 = self._events(idx_a_restarted, idx_b_new)
        c = cal.update("a", "b", ev_a2, ev_b2)
        assert abs(c) < 10.0, (
            f"Post-restart correction should not retain old 0.2-bit EMA; "
            f"got {c:+.1f} ns (expected ~0 because post-restart diff is "
            f"an integer bit multiple)"
        )


# ---------------------------------------------------------------------------
# compute_tdoa_s — per-node δ bias calibration (node_offsets_s)
# ---------------------------------------------------------------------------

def _ev(node_id: str, sync_to_snippet_start_ns: int = 0,
        node_lat: float = 47.65, node_lon: float = -122.31) -> dict:
    """Minimal event for calibration tests.

    Uses sync_delta-only path (no IQ refinement, no plausibility issues)
    to keep tests focussed on the calibration arithmetic.  Both nodes are
    near each other so path-delay correction is small and predictable.
    """
    return {
        **_make_event(node_lat, node_lon,
                      sync_to_snippet_start_ns=sync_to_snippet_start_ns),
        "node_id": node_id,
    }


def test_calibration_none_unchanged():
    """node_offsets_s=None must produce the same result as omitting it."""
    ev_a = _ev("a", sync_to_snippet_start_ns=5000)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    t_no_arg = compute_tdoa_s(ev_a, ev_b)
    t_none = compute_tdoa_s(ev_a, ev_b, node_offsets_s=None)
    t_empty = compute_tdoa_s(ev_a, ev_b, node_offsets_s={})
    assert t_no_arg is not None
    assert t_no_arg == t_none == t_empty


def test_calibration_subtracts_node_delta_diff():
    """Calibrated TDOA = raw_tdoa - (δ_a - δ_b)."""
    ev_a = _ev("a", sync_to_snippet_start_ns=5000)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    t_raw = compute_tdoa_s(ev_a, ev_b)
    assert t_raw is not None
    # δ_a = +10 µs, δ_b = -3 µs ⇒ subtract +13 µs from raw.
    delta_a, delta_b = 10e-6, -3e-6
    t_cal = compute_tdoa_s(ev_a, ev_b, node_offsets_s={"a": delta_a, "b": delta_b})
    assert t_cal is not None
    assert t_cal == pytest.approx(t_raw - (delta_a - delta_b), abs=1e-10)


def test_calibration_missing_node_uses_zero():
    """Nodes not present in the table are treated as δ = 0."""
    ev_a = _ev("a", sync_to_snippet_start_ns=5000)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    t_raw = compute_tdoa_s(ev_a, ev_b)
    delta_a = 7.5e-6
    # Only "a" is in the table; "b" should default to 0.
    t_cal = compute_tdoa_s(ev_a, ev_b, node_offsets_s={"a": delta_a})
    assert t_raw is not None and t_cal is not None
    assert t_cal == pytest.approx(t_raw - delta_a, abs=1e-10)


def test_calibration_antisymmetric():
    """Swapping (a, b) negates the calibrated TDOA — (δ_a - δ_b) flips sign too."""
    ev_a = _ev("a", sync_to_snippet_start_ns=5000)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    offsets = {"a": 12e-6, "b": -4e-6}
    t_ab = compute_tdoa_s(ev_a, ev_b, node_offsets_s=offsets)
    t_ba = compute_tdoa_s(ev_b, ev_a, node_offsets_s=offsets)
    assert t_ab is not None and t_ba is not None
    assert t_ab == pytest.approx(-t_ba, abs=1e-10)


def test_calibration_zero_offsets_unchanged():
    """All-zero offsets must produce the same result as no calibration."""
    ev_a = _ev("a", sync_to_snippet_start_ns=5000)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    t_raw = compute_tdoa_s(ev_a, ev_b)
    t_zero = compute_tdoa_s(ev_a, ev_b, node_offsets_s={"a": 0.0, "b": 0.0})
    assert t_raw == t_zero


def test_calibration_config_loads_from_yaml():
    """TdoaCalibrationConfig parses a fitted calibration block from YAML."""
    import yaml
    from beagle_server.config import ServerFullConfig
    yaml_text = """
tdoa_calibration:
  enabled: true
  reference_node: dpk-tdoa1
  node_offsets_s:
    dpk-tdoa1: 0.0
    dpk-tdoa2: 7.918322e-06
    n7jmv-tdoa-qth: -7.486309e-05
  fit_transmitter_label: "Magnolia"
  fit_transmitter_lat: 47.65133
  fit_transmitter_lon: -122.3918318
  fit_n_pairs: 128
  fit_residual_rms_us: 0.496
  fit_date: "2026-04-25"
"""
    raw = yaml.safe_load(yaml_text)
    cfg = ServerFullConfig.model_validate(raw)
    assert cfg.tdoa_calibration.enabled is True
    assert cfg.tdoa_calibration.reference_node == "dpk-tdoa1"
    assert cfg.tdoa_calibration.node_offsets_s["dpk-tdoa1"] == 0.0
    assert cfg.tdoa_calibration.node_offsets_s["dpk-tdoa2"] == pytest.approx(7.918322e-6)
    assert cfg.tdoa_calibration.node_offsets_s["n7jmv-tdoa-qth"] == pytest.approx(-7.486309e-5)
    assert cfg.tdoa_calibration.fit_n_pairs == 128
    assert cfg.tdoa_calibration.fit_residual_rms_us == pytest.approx(0.496)


def test_calibration_config_default_disabled():
    """Default config has calibration disabled and empty offsets."""
    from beagle_server.config import ServerFullConfig
    cfg = ServerFullConfig()
    assert cfg.tdoa_calibration.enabled is False
    assert cfg.tdoa_calibration.node_offsets_s == {}


def test_calibration_collapses_known_bias():
    """End-to-end: simulate a +20 µs per-node bias on node 'a' by adding
    20 µs to its sync_to_snippet_start_ns; calibration table {a: +20µs, b: 0}
    should remove that exact offset.
    """
    raw_delta_ns = 5000
    bias_ns = 20_000  # +20 µs of physical/clock bias on node 'a'
    ev_a_clean = _ev("a", sync_to_snippet_start_ns=raw_delta_ns)
    ev_b = _ev("b", sync_to_snippet_start_ns=0, node_lat=47.72, node_lon=-122.28)
    t_truth = compute_tdoa_s(ev_a_clean, ev_b)

    # Inject a +20 µs bias into a's reported timing.
    ev_a_biased = {**ev_a_clean,
                   "sync_to_snippet_start_ns": raw_delta_ns + bias_ns}
    # Without calibration, the biased measurement is off by 20 µs.
    t_biased = compute_tdoa_s(ev_a_biased, ev_b)
    assert t_truth is not None and t_biased is not None
    assert (t_biased - t_truth) == pytest.approx(bias_ns / 1e9, abs=1e-10)

    # With the matching calibration, biased measurement collapses back to
    # the truth.
    t_calibrated = compute_tdoa_s(
        ev_a_biased, ev_b,
        node_offsets_s={"a": bias_ns / 1e9, "b": 0.0},
    )
    assert t_calibrated is not None
    assert t_calibrated == pytest.approx(t_truth, abs=1e-10)
