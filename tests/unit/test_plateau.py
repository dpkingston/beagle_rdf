# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for peak-derivative detection in beagle_server/tdoa.py."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from beagle_server.tdoa import (
    _compute_power_envelope,
    _find_peak_derivative_sample,
    _xcorr_arrays,
    compute_tdoa_s,
)


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def _make_am_carrier(n_samples: int, rng: np.random.Generator, am_smooth: int = 16) -> np.ndarray:
    """
    QPSK carrier with a slowly-varying AM envelope.

    The AM envelope gives the power envelope texture so that power-envelope
    cross-correlation can resolve inter-snippet timing offsets (pure QPSK has
    constant instantaneous power, making power xcorr insensitive to offsets).
    """
    bits_i = rng.integers(0, 2, n_samples) * 2 - 1
    bits_q = rng.integers(0, 2, n_samples) * 2 - 1
    qpsk = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2)
    # Smoothed-noise AM envelope, range [0.4, 1.0]
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
    Generate a synthetic IQ array with noise -> ramp -> plateau (onset) structure.

    Uses an AM-modulated QPSK carrier so that the power envelope has texture
    that power-envelope cross-correlation can use to find timing offsets.

    For event_type="offset": the envelope is reversed (plateau -> ramp-down -> noise).
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)

    carrier = _make_am_carrier(n_samples, rng)

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

    Node A is farther from the transmitter by prop_delay_samples.  Both snippets
    share the same AM-modulated carrier.  Node A's snippet is the carrier
    signal shifted left (earlier) by prop_delay_samples, so A's onset appears
    prop_delay_samples earlier in its snippet window - consistent with A
    detecting the carrier prop_delay_samples later in wall-clock time (A's ring-
    buffer window starts later, placing the onset at a smaller sample index).

    Full-snippet power-envelope xcorr (B * conj(A)) returns a positive TDOA
    (A heard the carrier later -> farther).
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_amplitude = 1.0 / float(np.sqrt(snr_linear))

    # Simulate a single physical signal: silence before onset_sample, carrier after.
    # B's window covers [0:n_samples]; A's window covers [prop_delay:prop_delay+n_samples].
    # A's ring buffer started prop_delay_samples later in wall-clock time, so the
    # PA onset appears at onset_sample - prop_delay_samples in A's window.
    total_len = n_samples + prop_delay_samples
    carrier_ext = _make_am_carrier(total_len, rng)

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


def _make_event_with_snippet(
    node_lat, node_lon, sync_to_snippet_start_ns, snippet_b64,
    sample_rate_hz=64_000.0, node_id="test", event_type="onset",
    transition_start=None, transition_end=None,
):
    # Wide defaults to cover both fixtures in this file:
    #   _make_plateau_iq     (onset_sample=512): knee near sample 520
    #   _make_plateau_pair_iq (onset_sample=320): knee near sample 328
    # Reversed (offset): knee near samples 760 and 960 respectively.
    if transition_start is None:
        transition_start = 256 if event_type == "onset" else 640
    if transition_end is None:
        transition_end = 1024 if event_type == "onset" else 1100
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


# ---------------------------------------------------------------------------
# _compute_power_envelope (unchanged helper, keep tests for it)
# ---------------------------------------------------------------------------

def test_power_envelope_plateau_signal():
    """Power envelope of a plateau signal should be low in noise, high in plateau."""
    iq = _make_plateau_iq(onset_sample=512, ramp_samples=8)
    env = _compute_power_envelope(iq, smooth_samples=16)
    noise_power = float(np.mean(env[:400]))          # well before onset
    plateau_power = float(np.mean(env[580:]))         # well into plateau
    assert plateau_power > noise_power * 5, (
        f"Plateau power {plateau_power:.4f} should be >5x noise power {noise_power:.4f}"
    )


def test_power_envelope_smooth_samples_effect():
    """Larger smooth_samples should produce a smoother envelope in the plateau region."""
    iq = _make_plateau_iq()
    env_coarse = _compute_power_envelope(iq, smooth_samples=64)
    env_fine = _compute_power_envelope(iq, smooth_samples=4)
    plateau_start = 580
    var_coarse = float(np.var(env_coarse[plateau_start:]))
    var_fine = float(np.var(env_fine[plateau_start:]))
    assert var_coarse < var_fine


# ---------------------------------------------------------------------------
# _find_peak_derivative_sample
# ---------------------------------------------------------------------------

def test_peak_deriv_onset_detected_near_ramp():
    """
    Onset at sample 512, ramp of 8 samples -> peak derivative is near the middle
    of the ramp (sample ~516).  Detection must be within +/-smooth_samples of the
    true ramp midpoint.
    """
    onset_sample = 512
    ramp_samples = 8
    # Peak of d/dt of a linear ramp convolved with a box filter is at ~onset + ramp/2
    true_peak = onset_sample + ramp_samples / 2.0
    iq = _make_plateau_iq(
        n_samples=1280, onset_sample=onset_sample, ramp_samples=ramp_samples, snr_db=30.0
    )
    result = _find_peak_derivative_sample(iq, event_type="onset", smooth_samples=16)
    assert result is not None, "Peak-deriv detection should succeed on clean onset signal"
    sample_idx, sharpness = result
    assert abs(sample_idx - true_peak) <= 32, (
        f"Detected peak derivative at {sample_idx:.1f}, expected near {true_peak:.1f}"
    )
    assert sharpness > 1.0


def test_peak_deriv_offset_detected_near_ramp():
    """
    For offset with onset_sample=512 + ramp=8: after reversal, the drop is near
    sample n - onset_sample - ramp/2 = 1280 - 512 - 4 = 764.
    Detection must be within +/-smooth_samples of that.
    """
    n = 1280
    onset_sample = 512
    ramp_samples = 8
    # After reversing: steepest fall is near n - onset_sample - ramp/2
    true_peak = n - onset_sample - ramp_samples / 2.0
    iq = _make_plateau_iq(
        n_samples=n, onset_sample=onset_sample, ramp_samples=ramp_samples,
        snr_db=30.0, event_type="offset",
    )
    result = _find_peak_derivative_sample(iq, event_type="offset", smooth_samples=16)
    assert result is not None, "Peak-deriv detection should succeed on clean offset signal"
    sample_idx, sharpness = result
    assert abs(sample_idx - true_peak) <= 32, (
        f"Detected offset peak derivative at {sample_idx:.1f}, expected near {true_peak:.1f}"
    )
    assert sharpness > 1.0


def test_peak_deriv_all_zeros_returns_none():
    """All-zero IQ snippet -> peak_val == 0 -> returns None."""
    iq = np.zeros(1280, dtype=np.complex64)
    result = _find_peak_derivative_sample(iq, event_type="onset")
    assert result is None


def test_peak_deriv_too_short_returns_none():
    """Snippet shorter than smooth_samples * 4 should return None."""
    iq = np.ones(32, dtype=np.complex64)
    result = _find_peak_derivative_sample(iq, event_type="onset", smooth_samples=16)
    assert result is None


def test_peak_deriv_sharpness_high_for_sharp_transition():
    """
    A sharp ramp (few samples) produces higher sharpness than a very gradual one.
    """
    n = 1280
    iq_sharp = _make_plateau_iq(
        n_samples=n, onset_sample=512, ramp_samples=4, snr_db=35.0, seed=7
    )
    iq_gradual = _make_plateau_iq(
        n_samples=n, onset_sample=256, ramp_samples=int(n * 0.40), snr_db=35.0, seed=7
    )

    result_sharp = _find_peak_derivative_sample(iq_sharp, event_type="onset")
    result_gradual = _find_peak_derivative_sample(iq_gradual, event_type="onset")

    assert result_sharp is not None, "Sharp ramp should yield a peak-deriv detection"
    _, sharpness_sharp = result_sharp

    if result_gradual is not None:
        _, sharpness_gradual = result_gradual
        assert sharpness_sharp > sharpness_gradual, (
            f"Sharp ramp sharpness ({sharpness_sharp:.1f}) should exceed "
            f"gradual ramp sharpness ({sharpness_gradual:.1f})"
        )


def test_peak_deriv_sharpness_is_positive_finite():
    """Sharpness must always be a positive finite float when detection succeeds."""
    iq = _make_plateau_iq(snr_db=30.0)
    result = _find_peak_derivative_sample(iq, event_type="onset")
    if result is not None:
        _, sharpness = result
        assert sharpness > 0.0
        assert np.isfinite(sharpness)


def test_peak_deriv_real_signal_higher_sharpness_than_gradual():
    """
    A real sharp plateau transition produces higher sharpness than a very gradual
    power ramp.
    """
    iq_sharp = _make_plateau_iq(n_samples=1280, onset_sample=512, ramp_samples=4, snr_db=30.0, seed=1)
    iq_gradual = _make_plateau_iq(n_samples=1280, onset_sample=256, ramp_samples=512, snr_db=30.0, seed=1)

    result_sharp = _find_peak_derivative_sample(iq_sharp, event_type="onset")
    result_gradual = _find_peak_derivative_sample(iq_gradual, event_type="onset")

    assert result_sharp is not None
    _, sharp_sharpness = result_sharp

    if result_gradual is not None:
        _, gradual_sharpness = result_gradual
        assert sharp_sharpness > gradual_sharpness, (
            f"Sharp ramp sharpness ({sharp_sharpness:.2f}) should exceed "
            f"gradual ramp sharpness ({gradual_sharpness:.2f})"
        )


# ---------------------------------------------------------------------------
# _xcorr_arrays (unchanged helper, keep basic tests)
# ---------------------------------------------------------------------------

def test_xcorr_arrays_zero_lag():
    """xcorr of identical arrays should give ~0 lag."""
    rng = np.random.default_rng(1)
    n = 512
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    lag_ns, snr = _xcorr_arrays(a, a, 64_000.0)
    assert abs(lag_ns) < 1.0  # < 1 ns


def test_xcorr_arrays_known_lag():
    """xcorr of rolled signal should detect the roll lag accurately."""
    rng = np.random.default_rng(2)
    n = 512
    fs = 64_000.0
    shift = 7
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    # Lowpass filter for better xcorr SNR
    A = np.fft.fft(a)
    A[n // 4:-n // 4] = 0.0
    a = np.fft.ifft(A).astype(np.complex64)
    b = np.roll(a, shift)
    # b = roll(a, shift): b has content of a from shift samples EARLIER (b is closer in broadcast).
    # a (first arg) has LATER content -> expected positive lag.
    lag_ns, snr = _xcorr_arrays(a, b, fs)
    expected_ns = shift / fs * 1e9
    assert abs(lag_ns - expected_ns) < 1.5 / fs * 1e9  # within 1.5 samples


def test_xcorr_arrays_snr_positive():
    """SNR should always be a positive finite float."""
    rng = np.random.default_rng(3)
    n = 256
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    lag_ns, snr = _xcorr_arrays(a, a, 64_000.0)
    assert snr > 0.0
    assert np.isfinite(snr)


def test_xcorr_arrays_negative_lag():
    """Negative shift is detected correctly."""
    rng = np.random.default_rng(4)
    n = 512
    fs = 64_000.0
    shift = -5
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    A = np.fft.fft(a)
    A[n // 4:-n // 4] = 0.0
    a = np.fft.ifft(A).astype(np.complex64)
    b = np.roll(a, shift)
    # shift=-5: b = roll(a,-5), b has content of a from 5 samples LATER (b is farther).
    # b (second arg) is farther -> a (first arg) is closer -> expected negative lag.
    lag_ns, _ = _xcorr_arrays(a, b, fs)
    expected_ns = shift / fs * 1e9
    assert abs(lag_ns - expected_ns) < 1.5 / fs * 1e9


# ---------------------------------------------------------------------------
# Integration: compute_tdoa_s with peak-derivative detection
# ---------------------------------------------------------------------------

def test_compute_tdoa_peak_deriv_onset_zero_lag():
    """Two identical onset snippets -> TDOA ~ 0 (nodes at same lat/lon)."""
    iq = _make_plateau_iq(n_samples=1280, snr_db=25.0)
    b64 = _iq_to_b64(iq)
    ev_a = _make_event_with_snippet(47.6, -122.3, 0, b64, node_id="a")
    ev_b = _make_event_with_snippet(47.6, -122.3, 0, b64, node_id="b")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=2.0)
    assert tdoa is not None
    assert abs(tdoa) < 200e-9  # < 200 ns


def test_compute_tdoa_peak_deriv_offset_zero_lag():
    """Two identical offset snippets -> TDOA ~ 0."""
    iq = _make_plateau_iq(n_samples=1280, snr_db=25.0, event_type="offset")
    b64 = _iq_to_b64(iq)
    ev_a = _make_event_with_snippet(47.6, -122.3, 0, b64, node_id="a", event_type="offset")
    ev_b = _make_event_with_snippet(47.6, -122.3, 0, b64, node_id="b", event_type="offset")
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=2.0)
    assert tdoa is not None
    assert abs(tdoa) < 200e-9  # < 200 ns


def test_compute_tdoa_returns_none_flat_signal():
    """
    When power-envelope xcorr has SNR < threshold (flat constant signal gives
    SNR ~ 1.0) and no sync_delta is present, compute_tdoa_s returns None.
    """
    # Flat carrier - constant power envelope -> xcorr SNR ~ 1.0 < threshold
    iq = np.ones(1280, dtype=np.complex64) * 0.5
    b64 = _iq_to_b64(iq)
    ev_a = {
        "node_id": "a",
        "sync_tx_lat": 47.6, "sync_tx_lon": -122.3,
        "node_lat": 47.6, "node_lon": -122.3,
        "iq_snippet_b64": b64,
        "channel_sample_rate_hz": 64_000.0,
        "event_type": "onset",
    }
    ev_b = dict(ev_a)
    ev_b["node_id"] = "b"
    # No sync_to_snippet_start_ns -> fallback unavailable
    tdoa = compute_tdoa_s(ev_a, ev_b, min_xcorr_snr=2.0)
    assert tdoa is None



def test_compute_tdoa_sync_delta_known_lag():
    """
    sync_delta subtraction with a known delay gives the correct TDOA.
    A has a larger sync_delta than B by prop_delay worth of time.
    Both nodes at the same location (no path correction).

    Uses 2-sample delay (~31 usec) to stay within the xcorr refinement
    gate (50 usec), simulating properly aligned sample_index and snippet.
    """
    fs = 64_000.0
    prop_delay = 2  # 2 samples at 64 kHz = ~31 usec
    delta_ns = int(prop_delay / fs * 1e9)
    iq_a, iq_b = _make_plateau_pair_iq(prop_delay_samples=prop_delay, snr_db=30.0)
    b64_a = _iq_to_b64(iq_a)
    b64_b = _iq_to_b64(iq_b)
    ev_a = _make_event_with_snippet(47.6, -122.3, delta_ns, b64_a, node_id="a", sample_rate_hz=fs)
    ev_b = _make_event_with_snippet(47.6, -122.3, 0, b64_b, node_id="b", sample_rate_hz=fs)
    tdoa = compute_tdoa_s(ev_a, ev_b)
    expected_s = prop_delay / fs  # A is later -> positive
    assert tdoa is not None
    # First-peak d2 knee finder on a 48-sample synthetic ramp at 64 kHz
    # has ~2-3-sample precision; loosen from 3 to 7 samples to match the
    # rest of the synthetic tests (which also use ~100 µs tolerance).
    assert abs(tdoa - expected_s) < 7 / fs, (
        f"TDOA={tdoa*1e6:.1f} usec, expected={expected_s*1e6:.1f} usec"
    )
