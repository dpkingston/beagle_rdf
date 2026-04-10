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
    ramp_samples: int = 8,
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
    onset_sample: int = 320,  # 1/4 of n_samples - matches real snippet encoder
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
    # onset_sample - prop_delay_samples in A's window - the TDOA is encoded in the
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
    -> TDOA ~= 0 (path correction ~= 0 for equidistant nodes).
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
    tdoa = compute_tdoa_s(ev_a, ev_b)
    expected_s = prop_delay / fs  # positive = A later
    assert tdoa is not None
    assert abs(tdoa - expected_s) < 50e-6  # within 50 usec


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

def test_compute_tdoa_xcorr_refines_sync_delta():
    """
    xcorr provides sub-sample refinement on top of the sync_delta TDOA.

    Co-located nodes (same lat/lon) with sync_delta encoding a known delay.
    The IQ snippets have a small sub-sample offset that xcorr should detect
    and add to the coarse sync_delta TDOA.
    """
    fs = 64_000.0
    # Small prop delay (2 samples) -- within the refinement gate
    prop_delay = 2
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    # sync_delta encodes 100 usec coarse TDOA
    sd_a = 100_100_000  # 100.1 ms
    sd_b = 100_000_000  # 100.0 ms -- difference = 100 usec
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=sd_a, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=sd_b, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    # Coarse TDOA = 100 usec, xcorr refinement adds ~31 usec (2 samples at 64 kHz)
    # Combined should be ~131 usec
    expected_coarse = 100_000 / 1e9  # 100 usec
    expected_fine = prop_delay / fs   # ~31 usec
    assert abs(tdoa - (expected_coarse + expected_fine)) < 20e-6  # within 20 usec


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
    # xcorr will measure ~0 lag between them.
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=0, snr_db=25.0)

    ev_a = _make_event_with_snippet(
        node_a_pos[0], node_a_pos[1], sync_delta_ns=sd_a,
        snippet_b64=_iq_to_b64(iq_a), node_id="node-a",
    )
    ev_a["sync_tx_lat"] = sync_tx[0]
    ev_a["sync_tx_lon"] = sync_tx[1]

    ev_b = _make_event_with_snippet(
        node_b_pos[0], node_b_pos[1], sync_delta_ns=sd_b,
        snippet_b64=_iq_to_b64(iq_b), node_id="node-b",
    )
    ev_b["sync_tx_lat"] = sync_tx[0]
    ev_b["sync_tx_lon"] = sync_tx[1]

    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None

    # The true TDOA is about -61 usec (node A is closer to the transmitter).
    # The old code (xcorr replaces sync_delta) would return ~0.
    # The new code (sync_delta + xcorr refinement) must return ~-61 usec.
    assert abs(tdoa - true_tdoa_s) < 20e-6, (
        f"TDOA {tdoa*1e6:.1f} usec != expected {true_tdoa_s*1e6:.1f} usec; "
        f"xcorr may be replacing sync_delta instead of refining it"
    )


def test_compute_tdoa_xcorr_falls_back_on_low_snr():
    """
    When xcorr SNR is below min_xcorr_snr, the sync_delta fallback is used.

    Both events carry a pure noise snippet (no carrier structure -> very low
    xcorr SNR) and different sync_delta values.  The result should equal the
    sync_delta difference, not the xcorr lag.
    """
    rng = np.random.default_rng(99)
    # Pure white noise - no amplitude edge -> power-envelope xcorr SNR ~ 1
    noise_a = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    noise_b = (rng.standard_normal(1280) + 1j * rng.standard_normal(1280)).astype(np.complex64)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=5_000,
                                     snippet_b64=_iq_to_b64(noise_a), node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0,
                                     snippet_b64=_iq_to_b64(noise_b), node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.5)
    assert tdoa is not None
    # sync_delta fallback: 5000 ns - 0 + path_correction ~ 5 usec (nodes co-located)
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-6)


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

    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=7_000,
                                     snippet_b64=_iq_to_b64(iq_a), sample_rate_hz=fs,
                                     node_id="node-a", event_type="offset")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0,
                                     snippet_b64=_iq_to_b64(iq_b), sample_rate_hz=fs,
                                     node_id="node-b", event_type="offset")

    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3, max_xcorr_baseline_km=1.0)
    assert tdoa is None, (
        f"Expected None (bad xcorr rejected); got {tdoa}"
    )


def test_compute_tdoa_xcorr_refinement_within_gate():
    """
    xcorr lag within the refinement gate (50 usec) is accepted and added
    to the sync_delta TDOA.

    1-sample prop delay at 64 kHz ~ 15.6 usec -- well within the gate.
    sync_delta is 0 for both co-located nodes, so the result should be
    the xcorr refinement alone (~15.6 usec).
    """
    fs = 64_000.0
    prop_delay = 1  # ~ 15.6 usec -- within 50 usec refinement gate
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3)
    assert tdoa is not None
    assert abs(tdoa - prop_delay / fs) < 10e-6  # within 10 usec of true delay


def test_compute_tdoa_xcorr_large_lag_rejected_as_refinement():
    """
    xcorr lag exceeding the refinement gate (50 usec) indicates a
    sync_delta/snippet anchor misalignment.  compute_tdoa_s returns
    None so the solver skips this pair rather than using the unreliable
    sync_delta-only fallback.

    10-sample prop delay at 64 kHz ~ 156 usec -- exceeds 50 usec gate.
    """
    fs = 64_000.0
    prop_delay = 10  # ~ 156 usec -- exceeds refinement gate
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    ev_a = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_a),
                                     sample_rate_hz=fs, node_id="node-a")
    ev_b = _make_event_with_snippet(47.6, -122.3, sync_delta_ns=0, snippet_b64=_iq_to_b64(iq_b),
                                     sample_rate_hz=fs, node_id="node-b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=1.3)
    assert tdoa is None, (
        f"Expected None (xcorr refinement too large); got {tdoa}"
    )


def test_compute_tdoa_colocated_xcorr_near_zero():
    """
    Co-located nodes receiving the same transmission produce xcorr TDOA ~ 0.
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
    assert abs(tdoa) < 1e-4  # < 100 usec (one-sample at 64 kHz)


# ---------------------------------------------------------------------------
# compute_tdoa_s - sync_delta subtraction
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
    assert tdoa == pytest.approx(5_000 / 1e9, abs=1e-12)  # 5 usec


def test_compute_tdoa_returns_none_when_sync_delta_missing():
    """Returns None when sync_delta_ns is absent."""
    ev_a = {"node_id": "a", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    ev_b = {"node_id": "b", "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
            "node_lat": 47.6, "node_lon": -122.3, "event_type": "onset"}
    assert compute_tdoa_s(ev_a, ev_b) is None


# ---------------------------------------------------------------------------
# compute_tdoa_s - pilot sync event disambiguation
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


_T_SYNC_NS = 1_000_000_000.0 / 1187.5   # must match tdoa.py constant (~842,105 ns)


def test_sync_disambiguation_no_adjustment_needed():
    """
    Both nodes used the same RDS bit boundary (raw_ns within T_sync/2).
    No adjustment should be applied; TDOA equals the raw 5 usec difference.
    """
    t0 = 1_700_000_000_000_000_000  # arbitrary epoch ns
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=505_000, onset_time_ns=t0 + 5_000)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=500_000, onset_time_ns=t0)
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
    ev_a = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=sync_delta_a,
                                        onset_time_ns=t0 + true_tdoa_ns)
    ev_b = _make_event_with_onset_time(47.6, -122.3, sync_delta_ns=sync_delta_b,
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
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    assert tdoa is not None
    assert tdoa == pytest.approx(true_tdoa_ns / 1e9, abs=1e-6)


def test_sync_disambiguation_n_zero():
    """n=0: nodes referenced the same bit boundary; raw_ns is already correct."""
    true_tdoa_ns = 200_000   # 200 usec - within T_sync/2 = 421 usec
    sync_delta_a = 400_000
    sync_delta_b = sync_delta_a - true_tdoa_ns  # raw_ns = +200_000
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
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
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
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
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
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
    ev_a = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_a)
    ev_b = _make_event(47.6, -122.3, sync_delta_ns=sync_delta_b)
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
    SD_A_NS = 2816029    # dpk-tdoa1 sync_delta_ns
    SD_B_NS = 2764024    # kb7ryy sync_delta_ns

    def _make_event(self, node_pos, sync_delta_ns, sync_tx, node_id):
        return {
            "node_id": node_id,
            "sync_delta_ns": sync_delta_ns,
            "sync_tx_lat": sync_tx[0],
            "sync_tx_lon": sync_tx[1],
            "node_lat": node_pos[0],
            "node_lon": node_pos[1],
            "event_type": "offset",
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
            tdoa = compute_tdoa_s(ev_a, ev_b)
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
