# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
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
# Snippet helpers
# ---------------------------------------------------------------------------

def _make_am_carrier(n_samples: int, rng: np.random.Generator, am_smooth: int = 16) -> np.ndarray:
    """
    QPSK carrier with slowly-varying AM envelope.

    Pure QPSK has constant instantaneous power (|s[n]|² = 1), making
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
    ramp_samples: int = 8,
    snr_db: float = 25.0,
    seed: int = 42,
    event_type: str = "onset",
) -> np.ndarray:
    """
    Synthetic IQ snippet with noise → ramp → plateau (onset) structure.

    Uses a QPSK carrier (constant instantaneous power = 1) for reliable
    plateau detection at typical SNR values.

    For event_type="offset": the envelope is reversed (plateau → ramp-down → noise).
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
    onset_sample: int = 320,  # 1/4 of n_samples — matches real snippet encoder
    ramp_samples: int = 8,
    prop_delay_samples: int = 10,
    snr_db: float = 30.0,
    seed: int = 42,
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
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_amplitude = 1.0 / float(np.sqrt(snr_linear))

    # Simulate a single physical signal: noise before onset_sample, carrier after.
    # B's window covers [0:n_samples]; A's window covers [prop_delay:prop_delay+n_samples].
    # A's ring buffer started prop_delay_samples later in wall-clock time, so the
    # PA onset (at physical position onset_sample) appears at position
    # onset_sample - prop_delay_samples in A's window — the TDOA is encoded in the
    # step function position, not just in the AM texture.
    total_len = n_samples + prop_delay_samples
    carrier_ext = _make_am_carrier(total_len, rng)

    # Physical signal: silence in [0:onset_sample], carrier from onset_sample onward.
    physical = np.zeros(total_len, dtype=np.complex64)
    ramp_end = onset_sample + ramp_samples
    if ramp_samples > 0:
        physical[onset_sample:ramp_end] = (
            carrier_ext[onset_sample:ramp_end]
            * np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
        )
    physical[ramp_end:] = carrier_ext[ramp_end:]

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


def _make_event(node_lat, node_lon, sync_delta_ns, sync_tx_lat=47.6, sync_tx_lon=-122.3):
    return {
        "sync_delta_ns": sync_delta_ns,
        "sync_tx_lat": sync_tx_lat,
        "sync_tx_lon": sync_tx_lon,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "event_type": "onset",
    }


def _make_event_with_snippet(
    node_lat, node_lon, sync_delta_ns, snippet_b64,
    sample_rate_hz=64_000.0, node_id="test", event_type="onset",
):
    return {
        "node_id": node_id,
        "sync_delta_ns": sync_delta_ns,
        "sync_tx_lat": 47.6,
        "sync_tx_lon": -122.3,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "iq_snippet_b64": snippet_b64,
        "channel_sample_rate_hz": sample_rate_hz,
        "event_type": event_type,
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
    → TDOA ~= 0 (path correction ~= 0 for equidistant nodes).
    """
    iq = _make_plateau_iq()
    snip = _iq_to_b64(iq)
    ev_a = {**_make_event(47.7, -122.3, 500_000_000, sync_tx_lat=47.6, sync_tx_lon=-122.3),
            "node_id": "a", "iq_snippet_b64": snip, "channel_sample_rate_hz": 64_000.0}
    ev_b = {**_make_event(47.5, -122.3, 500_000_000, sync_tx_lat=47.6, sync_tx_lon=-122.3),
            "node_id": "b", "iq_snippet_b64": snip, "channel_sample_rate_hz": 64_000.0}
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=2.0)
    assert tdoa is not None
    assert abs(tdoa) < 200e-9  # < 200 ns


def test_compute_tdoa_known_geometry():
    """
    sync_delta subtraction with known geometry gives correct TDOA.
    sync_tx equidistant from both nodes → path correction ≈ 0.
    """
    fs = 64_000.0
    prop_delay = 10
    delta_ns = int(prop_delay / fs * 1e9)
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.7, -122.3, delta_ns, _iq_to_b64(iq_a), node_id="node-a",
                                     sample_rate_hz=fs)
    ev_b = _make_event_with_snippet(47.5, -122.3, 0, _iq_to_b64(iq_b), node_id="node-b",
                                     sample_rate_hz=fs)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    expected_s = prop_delay / fs  # positive = A later
    assert tdoa is not None
    assert abs(tdoa - expected_s) < 50e-6  # within 50 µs


def test_compute_tdoa_antisymmetric():
    """compute_tdoa_s(A, B) == -compute_tdoa_s(B, A) via sync_delta fallback."""
    ev_a = _make_event(47.65, -122.31, sync_delta_ns=5000)
    ev_b = _make_event(47.72, -122.28, sync_delta_ns=0)
    t_ab = compute_tdoa_s(ev_a, ev_b)
    t_ba = compute_tdoa_s(ev_b, ev_a)
    assert t_ab is not None and t_ba is not None
    assert abs(t_ab + t_ba) < 1e-12  # exact antisymmetry with sync_delta


def test_compute_tdoa_path_delay_applied():
    """
    Both nodes have same sync_delta (raw_ns=0); node_A far from sync_tx.
    TDOA comes entirely from path-delay correction: +30km/c ≈ +100 µs.
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
# compute_tdoa_s -- xcorr primary path
# ---------------------------------------------------------------------------

def test_compute_tdoa_xcorr_preferred_over_sync_delta():
    """
    When IQ snippets are present and xcorr SNR is sufficient, the xcorr lag
    is returned rather than sync_delta.

    Node pair with a known 10-sample propagation delay (~156 µs at 64 kHz).
    The sync_delta values are deliberately set to 0 for both nodes, so if
    sync_delta were used the result would be 0; xcorr must give the non-zero lag.
    """
    fs = 64_000.0
    prop_delay = 10
    expected_s = prop_delay / fs
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert abs(tdoa) > 1e-9, "Expected non-zero xcorr lag; sync_delta fallback may have fired"
    assert abs(tdoa - expected_s) < 50e-6  # within 50 µs of true propagation delay


def test_compute_tdoa_xcorr_falls_back_on_low_snr():
    """
    When xcorr SNR is below min_xcorr_snr, the sync_delta fallback is used.

    Both events carry a pure noise snippet (no carrier structure → very low
    xcorr SNR) and different sync_delta values.  The result should equal the
    sync_delta difference, not the xcorr lag.
    """
    rng = np.random.default_rng(99)
    # Pure white noise — no amplitude edge → power-envelope xcorr SNR ≈ 1
    noise_a = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    noise_b = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=5_000,
                                     snippet_b64=_iq_to_b64(noise_a), node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0,
                                     snippet_b64=_iq_to_b64(noise_b), node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.5)
    assert tdoa is not None
    # sync_delta fallback: 5000 ns − 0 + path_correction ≈ 5 µs (nodes co-located)
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-6)


def test_compute_tdoa_xcorr_geo_filter_rejects_implausible_lag():
    """
    xcorr lag that exceeds the geometric plausibility limit falls back to sync_delta.

    A strong carrier edge on snippet A but flat noise on snippet B produces a
    high-SNR xcorr peak at a large random lag (no true TDOA).  With a tight
    max_xcorr_baseline_km (1 km → max TDOA ≈ 3.3 µs), the lag is rejected and
    sync_delta is used instead.
    """
    rng = np.random.default_rng(42)
    fs = 64_000.0
    # snippet_a: carrier ON for first half, OFF for second half (clear offset edge)
    n = 1280
    carrier_a = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)).astype(np.complex64) * 100
    noise_a   = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)).astype(np.complex64)
    iq_a = np.concatenate([carrier_a, noise_a])
    # snippet_b: pure noise — no PA edge, so xcorr SNR will be high but lag random
    iq_b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 50

    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=7_000,
                                     snippet_b64=_iq_to_b64(iq_a), sample_rate_hz=fs,
                                     node_id="node-a", event_type="offset")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0,
                                     snippet_b64=_iq_to_b64(iq_b), sample_rate_hz=fs,
                                     node_id="node-b", event_type="offset")

    # Tight baseline: 1 km → max TDOA ≈ 3.3 µs.  Any genuine xcorr lag from
    # mismatched snippets will exceed this, so geo filter must reject it.
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3, max_xcorr_baseline_km=1.0)
    assert tdoa is not None
    # Must have fallen back to sync_delta (7000 ns − 0, nodes co-located)
    assert tdoa == pytest.approx(7_000 / 1e9, abs=1e-6), (
        "Expected sync_delta fallback; geo filter may not have fired"
    )


def test_compute_tdoa_xcorr_geo_filter_accepts_plausible_lag():
    """
    xcorr lag within the geometric plausibility limit is accepted.

    10-sample prop delay at 64 kHz ≈ 156 µs.  With max_xcorr_baseline_km=100
    (max TDOA ≈ 333 µs), the lag is within bounds and xcorr result is used.
    """
    fs = 64_000.0
    prop_delay = 10  # ≈ 156 µs
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3, max_xcorr_baseline_km=100.0)
    assert tdoa is not None
    assert abs(tdoa) > 1e-9, "Expected non-zero xcorr lag; sync_delta fallback may have fired"
    assert abs(tdoa - prop_delay / fs) < 50e-6  # within 50 µs of true delay


def test_compute_tdoa_colocated_xcorr_near_zero():
    """
    Co-located nodes receiving the same transmission produce xcorr TDOA ≈ 0.
    This mirrors the co-located calibration scenario and validates that xcorr
    does not introduce spurious bias.
    """
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=0, snr_db=25.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert abs(tdoa) < 1e-4  # < 100 µs (one-sample at 64 kHz)


# ---------------------------------------------------------------------------
# compute_tdoa_s -- sync_delta subtraction
# ---------------------------------------------------------------------------

def test_compute_tdoa_sync_delta_difference():
    """
    sync_delta subtraction (fallback path) produces the correct raw TDOA.
    No IQ snippets: xcorr does not fire; result is purely sync_delta-based.
    Both nodes at same location (no path correction).
    """
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=500_005_000)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=500_000_000)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-12)  # 5 µs


def test_compute_tdoa_returns_none_when_sync_delta_missing():
    """Returns None when sync_delta_ns is absent."""
    ev_a = {"node_id": "a", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    ev_b = {"node_id": "b", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    assert compute_tdoa_s(ev_a, ev_b) is None


# ---------------------------------------------------------------------------
# compute_tdoa_s -- pilot sync event disambiguation
# ---------------------------------------------------------------------------

def _make_event_with_onset_time(node_lat, node_lon, sync_delta_ns, onset_time_ns,
                                 sync_tx_lat=47.6, sync_tx_lon=-122.3):
    """Event dict including onset_time_ns for disambiguation tests."""
    return {
        "node_id": "test",
        "sync_delta_ns": sync_delta_ns,
        "sync_tx_lat": sync_tx_lat,
        "sync_tx_lon": sync_tx_lon,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "event_type": "onset",
        "onset_time_ns": onset_time_ns,
    }


_T_SYNC_NS = 7_000_000   # must match tdoa.py constant


def test_pilot_disambiguation_no_adjustment_needed():
    """
    Both nodes used the same pilot pulse (raw_ns within T_sync/2 of onset_diff).
    No adjustment should be applied; TDOA equals the raw 5 µs difference.
    """
    # onset_time_ns identical → onset_diff = 0; raw_ns = 5_000 → n = 0
    t0 = 1_700_000_000_000_000_000  # arbitrary epoch ns
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=5_005_000, onset_time_ns=t0 + 5_000)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=5_000_000, onset_time_ns=t0)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-9)


def test_pilot_disambiguation_corrects_one_period_offset():
    """
    Node B used the pilot pulse one T_sync later than node A.
    raw_ns = true_TDOA − T_sync (≈ −6.995 ms); disambiguation adds +T_sync.
    """
    true_tdoa_ns = 5_000  # 5 µs — the real propagation-delay difference
    t0 = 1_700_000_000_000_000_000
    # Node A: sync_delta = 6_000_000; Node B: used a pilot T_sync later,
    # so its sync_delta = 6_000_000 + true_tdoa − T_sync = 6_000_000 + 5_000 − 7_000_000
    raw_ns = true_tdoa_ns - _T_SYNC_NS           # = −6_995_000 ns (without fix)
    sync_delta_a = 6_000_000
    sync_delta_b = sync_delta_a - raw_ns         # = 13_000_000 (> T_sync, but only after wrap)
    # Onset times: nearly equal for co-located nodes, difference = true_tdoa_ns
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=sync_delta_a,
                                        onset_time_ns=t0 + true_tdoa_ns)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=sync_delta_b,
                                        onset_time_ns=t0)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-6)


def test_pilot_disambiguation_works_without_onset_time():
    """
    Geometric disambiguation resolves n from path geometry alone — onset_time_ns
    is not required.  raw_ns = true_TDOA − T_sync is corrected to true_TDOA.

    Rationale: |true_TDOA| ≤ dist(A,B)/c ≤ 100 km/c ≈ 333 µs << T_sync/2 = 3.5 ms,
    so round((raw_ns + correction) / T_sync) uniquely identifies n without any
    wall-clock comparison.
    """
    true_tdoa_ns = 5_000  # 5 µs
    sync_delta_a = 6_000_000
    sync_delta_b = sync_delta_a - (true_tdoa_ns - _T_SYNC_NS)  # raw_ns = -6_995_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    # Geometric disambiguation corrects the T_sync offset; result is near true_TDOA
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-6)


def test_pilot_disambiguation_n_zero():
    """n=0: nodes locked to the same pilot cycle; raw_ns is already correct."""
    true_tdoa_ns = 200_000   # 200 µs — within one pilot period
    sync_delta_a = 3_000_000
    sync_delta_b = sync_delta_a - true_tdoa_ns  # raw_ns = +200_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_pilot_disambiguation_n_plus_one():
    """
    n=+1: node A locked to a pilot cycle one T_sync *earlier* than node B.
    raw_ns = true_TDOA + T_sync ≈ +7 ms; disambiguation subtracts T_sync.
    """
    true_tdoa_ns = 150_000   # 150 µs
    sync_delta_a = 2_000_000
    sync_delta_b = sync_delta_a - (true_tdoa_ns + _T_SYNC_NS)  # raw = -6_850_000 → n=-1?
    # Construct n=+1 case: raw_ns = true_tdoa + T_sync
    raw_ns = true_tdoa_ns + _T_SYNC_NS        # = +7_150_000 ns
    sync_delta_a2 = 5_000_000
    sync_delta_b2 = sync_delta_a2 - raw_ns    # = -2_150_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a2)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b2)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_pilot_disambiguation_n_minus_one():
    """
    n=−1: node A locked to a pilot cycle one T_sync *later* than node B.
    raw_ns = true_TDOA − T_sync ≈ −7 ms; disambiguation adds T_sync.
    """
    true_tdoa_ns = 150_000   # 150 µs
    raw_ns = true_tdoa_ns - _T_SYNC_NS        # = -6_850_000 ns
    sync_delta_a = 5_000_000
    sync_delta_b = sync_delta_a - raw_ns      # = 11_850_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)


def test_pilot_disambiguation_large_tdoa_within_half_period():
    """
    A true TDOA near ±T_sync/2 but still within it (n=0) is left unchanged.
    raw_ns = −3_400_000 ns: |raw| < T_sync/2 = 3_500_000 → n=0, no adjustment.
    """
    true_tdoa_ns = -3_400_000
    sync_delta_a = 1_000_000
    sync_delta_b = sync_delta_a - true_tdoa_ns   # = 4_400_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-9)
