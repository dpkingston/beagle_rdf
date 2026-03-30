# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for IQ snippet cross-correlation (beagle_server.tdoa)."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from beagle_server.tdoa import cross_correlate_snippets, _decode_iq_snippet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_iq(samples: np.ndarray) -> str:
    """Encode complex64 array to base64 int8-interleaved string."""
    scale = float(np.max(np.abs(samples))) + 1e-30
    normed = samples / scale
    raw = np.empty(len(normed) * 2, dtype=np.int8)
    raw[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
    raw[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
    return base64.b64encode(raw.tobytes()).decode()


def _bandlimited_noise(n: int = 640, seed: int = 0) -> np.ndarray:
    """
    Generate bandlimited complex noise via FFT lowpass filtering.

    Bandlimited noise has a non-flat autocorrelation so the cross-correlation
    peak is sharp and SNR is high.  Pure tones have constant-magnitude
    autocorrelation (SNR ~ 1) regardless of lag and are unsuitable for
    cross-correlation tests.
    """
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    W = np.fft.fft(white)
    cutoff = n // 4   # lowpass at ~25% of Nyquist
    W[cutoff:-cutoff] = 0.0
    return np.fft.ifft(W).astype(np.complex64)


# ---------------------------------------------------------------------------
# _decode_iq_snippet
# ---------------------------------------------------------------------------

def test_decode_roundtrip():
    """Encode then decode should reproduce the original signal within int8 quantisation."""
    original = _bandlimited_noise(128, seed=7)
    b64 = _encode_iq(original)
    decoded = _decode_iq_snippet(b64)
    assert len(decoded) == 128
    # After normalise-to-+/-1 roundtrip, phases should be preserved; magnitudes ~1
    phase_orig = np.angle(original)
    phase_dec  = np.angle(decoded)
    phase_diff = np.abs(np.angle(np.exp(1j * (phase_dec - phase_orig))))
    assert float(np.mean(phase_diff)) < 0.05  # < ~3 degree mean phase error


# ---------------------------------------------------------------------------
# cross_correlate_snippets - lag recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lag_samples", [0, 1, 5, 10, -5, -10])
def test_known_lag_recovery(lag_samples: int):
    """
    Two snippets of the same bandlimited noise, one shifted by lag_samples.

    np.roll(base, D) gives b[n] = a[n-D], which physically models node A
    detecting the carrier D samples *later* than node B (A is later).
    The correlator should return a positive lag (A is later) equal to +D samples.
    Tolerance: +/-1.5 samples.
    """
    rate = 64_000.0
    n = 640
    base = _bandlimited_noise(n, seed=1)
    shifted = np.roll(base, lag_samples)   # b[n] = a[n - lag_samples]; A is later by lag_samples

    b64_a = _encode_iq(base)
    b64_b = _encode_iq(shifted)

    lag_ns, snr = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=rate)

    # b[n] = a[n - D] -> A started D samples later -> A is later -> positive lag
    expected_ns = lag_samples * 1e9 / rate
    tol_ns = 1.5 * 1e9 / rate
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"lag={lag_samples}: expected {expected_ns:.1f} ns, got {lag_ns:.1f} ns"
    )
    if lag_samples != 0:
        # Power-envelope xcorr SNR is inherently low (~1.6) for random noise snippets
        # because the DC component of |IQ|^2 dominates the sidelobe mean.  Real carrier
        # snippets with onset/offset transitions yield higher SNR.  The key property
        # tested here is lag accuracy, not SNR magnitude.
        assert snr > 1.1, f"Expected SNR > 1.1 for bandlimited noise, got {snr:.2f}"


def test_zero_lag_gives_near_zero_ns():
    """Identical snippets should give lag ~= 0 ns."""
    b64 = _encode_iq(_bandlimited_noise(640, seed=2))
    lag_ns, snr = cross_correlate_snippets(b64, b64, sample_rate_hz_a=64_000.0)
    assert abs(lag_ns) < 1.0  # sub-ns for identical signals
    # Power-envelope autocorrelation SNR ~ 1 + Var(|IQ|^2)/mean(|IQ|^2)^2 ~ 1.6 for noise
    assert snr > 1.1


def test_sample_rate_scales_lag():
    """Lag in ns scales inversely with sample rate."""
    base = _bandlimited_noise(640, seed=3)
    shifted = np.roll(base, 5)   # A is later by 5 samples
    b64_a = _encode_iq(base)
    b64_b = _encode_iq(shifted)

    lag_64k, _ = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=64_000.0)
    lag_62k, _ = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=62_500.0)

    # Both should recover ~+5 samples but at different ns/sample
    expected_64k = 5 * 1e9 / 64_000.0
    expected_62k = 5 * 1e9 / 62_500.0
    assert abs(lag_64k - expected_64k) < 1e9 / 64_000.0 * 1.5
    assert abs(lag_62k - expected_62k) < 1e9 / 62_500.0 * 1.5


# ---------------------------------------------------------------------------
# cross_correlate_snippets - robustness
# ---------------------------------------------------------------------------

def test_all_zero_input_does_not_crash():
    """All-zero IQ should not raise; SNR will be near zero."""
    zero = np.zeros(640, dtype=np.complex64)
    b64 = _encode_iq(zero + 1e-30)  # avoid /0 in scale
    lag_ns, snr = cross_correlate_snippets(b64, b64, sample_rate_hz_a=64_000.0)
    assert isinstance(lag_ns, (float, np.floating))
    assert isinstance(snr, (float, np.floating))


def test_mismatched_lengths():
    """Snippets of different lengths should still produce a result."""
    b64_a = _encode_iq(_bandlimited_noise(640, seed=4))
    b64_b = _encode_iq(_bandlimited_noise(320, seed=4))
    lag_ns, snr = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=64_000.0)
    assert isinstance(lag_ns, (float, np.floating))


def test_low_snr_produces_low_correlation_snr():
    """Pure noise snippets (independent) should give low xcorr SNR."""
    rng = np.random.default_rng(99)
    noise_a = (rng.standard_normal(640) + 1j * rng.standard_normal(640)).astype(np.complex64)
    noise_b = (rng.standard_normal(640) + 1j * rng.standard_normal(640)).astype(np.complex64)
    _, snr = cross_correlate_snippets(_encode_iq(noise_a), _encode_iq(noise_b))
    # Independent noise should give SNR close to 1 (peak ~ mean sidelobe)
    assert snr < 5.0


def test_mixed_sample_rate_resampling():
    """
    Snippets captured at different sample rates (RTL-SDR 64 kHz vs RSPduo
    62.5 kHz) must give the correct lag in nanoseconds after resampling.

    Two sub-tests:

    1. Zero TDOA: the same physical signal resampled to each node's rate must
       give lag ~ 0 ns.  Without rate-aware resampling this would give a large
       systematic error proportional to the rate difference (2.4%).

    2. Known TDOA: node A is delayed by D samples at 64 kHz (= D/64000 s).
       Node B has no extra delay but captures at 62.5 kHz.  The xcorr must
       recover the correct lag in nanoseconds regardless of which rate is the
       "reference".
    """
    from scipy.signal import resample_poly as _rsp

    # Physical bandlimited signal (complex64, 64 kHz reference rate).
    base_64k = _bandlimited_noise(1280, seed=20)

    # Same physical signal sampled at 62.5 kHz (ratio 125:128 of 64 kHz).
    base_62k = (
        _rsp(base_64k.real, 125, 128).astype(np.float32)
        + 1j * _rsp(base_64k.imag, 125, 128).astype(np.float32)
    ).astype(np.complex64)

    tolerance_ns = 1e9 / 62_500.0 * 2  # +/-2 samples at 62.5 kHz ~ +/-32 usec

    # --- Sub-test 1: zero TDOA ---
    lag_zero, _ = cross_correlate_snippets(
        _encode_iq(base_64k), _encode_iq(base_62k),
        sample_rate_hz_a=64_000.0,
        sample_rate_hz_b=62_500.0,
    )
    assert abs(lag_zero) < tolerance_ns, (
        f"Zero-TDOA mixed-rate lag should be ~0 ns, got {lag_zero:.0f} ns"
    )

    # --- Sub-test 2: B delayed by 10 samples at 64 kHz = 156.25 usec ---
    # Correlation convention: lag > 0 when B is delayed (B arrives later).
    # We delay B (base_62k shifted by ~10 samples worth in physical time) so
    # the sign of the expected result is unambiguous.
    delay_samples = 10
    true_lag_ns = delay_samples * 1e9 / 64_000.0  # 156 250 ns
    # Delay B by rolling the 62.5 kHz signal by the equivalent sample count.
    delay_62k = round(delay_samples * 62_500.0 / 64_000.0)  # 10 samples at 64k -> ~10 at 62.5k
    delayed_b = np.roll(base_62k, delay_62k)

    lag_known, _ = cross_correlate_snippets(
        _encode_iq(base_64k), _encode_iq(delayed_b),
        sample_rate_hz_a=64_000.0,
        sample_rate_hz_b=62_500.0,
    )
    assert abs(lag_known - true_lag_ns) < tolerance_ns, (
        f"Mixed-rate lag wrong: {lag_known:.0f} ns vs {true_lag_ns:.0f} ns "
        f"(tolerance +/-{tolerance_ns:.0f} ns)"
    )

    # Explicit target_rate_hz should give the same result as auto-lower.
    lag_forced, _ = cross_correlate_snippets(
        _encode_iq(base_64k), _encode_iq(delayed_b),
        sample_rate_hz_a=64_000.0,
        sample_rate_hz_b=62_500.0,
        target_rate_hz=62_500.0,
    )
    assert abs(lag_forced - lag_known) < 1.0, (
        "Explicit target_rate_hz=62500 should match auto-lower result"
    )


# ---------------------------------------------------------------------------
# Transition windowing (event_type-based half-snippet selection)
# ---------------------------------------------------------------------------

def _make_onset_snippet(n: int = 1280, transition_at: int = 320, seed: int = 10) -> np.ndarray:
    """
    Synthesise a 1280-sample onset snippet: noise before the transition,
    bandlimited carrier after.  PA rise is at *transition_at* (1/4 of n).
    """
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.02
    carrier = _bandlimited_noise(n, seed=seed + 1)
    # Normalise carrier to unit amplitude
    carrier = carrier / (np.max(np.abs(carrier)) + 1e-30)
    snippet = noise.copy()
    snippet[transition_at:] = carrier[transition_at:]
    return snippet


def test_onset_transition_windowing_recovers_correct_lag():
    """
    With event_type="onset" the xcorr uses the first 3/4 of the snippet.
    The PA rise at ~1/4 position falls within the trim window; the lag should
    be accurately recovered and SNR above the acceptance threshold.
    """
    n = 1280
    lag_samples = 5
    sig_a = _make_onset_snippet(n, transition_at=n // 4, seed=10)
    sig_b = np.roll(sig_a, lag_samples)

    b64_a = _encode_iq(sig_a)
    b64_b = _encode_iq(sig_b)
    rate = 62_500.0

    lag_ns, snr = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=rate, event_type="onset")

    expected_ns = lag_samples * 1e9 / rate
    tol_ns = 1.5 * 1e9 / rate
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"Onset windowing lag wrong: {lag_ns:.0f} ns vs {expected_ns:.0f} ns"
    )
    assert snr > 1.5, f"Onset windowing SNR should exceed threshold 1.5, got {snr:.2f}"


def test_misaligned_onset_snippet_returns_zero_snr():
    """
    If the PA transition is outside the first-3/4 window (misaligned snippet),
    the xcorr trim contains only silence.  The power guard must return SNR=0.0.

    The onset trim now uses the first 3/4 of the snippet.  A transition in the
    final 1/8 is well outside this window (no convolution bleed at 16 samples).
    """
    n = 1280
    # Carrier starts at 7/8 of snippet (sample 1120) - completely outside the
    # first-3/4 trim [0:960], with >= 160 samples of margin vs. the 16-sample
    # smooth kernel, ensuring no power bleed into the trim window.
    sig = np.zeros(n, dtype=np.complex64)
    carrier = _bandlimited_noise(n, seed=20)
    carrier = carrier / (np.max(np.abs(carrier)) + 1e-30)
    sig[7 * n // 8 :] = carrier[7 * n // 8 :]   # carrier only in last 1/8

    b64 = _encode_iq(sig)

    _, snr = cross_correlate_snippets(b64, b64, sample_rate_hz_a=62_500.0, event_type="onset")
    assert snr == 0.0, f"Misaligned onset snippet should return SNR=0, got {snr:.2f}"


def test_offset_transition_windowing_uses_second_half():
    """
    event_type="offset" should use the second half [N//2:] where the PA fall
    is anchored at ~3/4 of the snippet.  The lag should still be correct.
    """
    rng = np.random.default_rng(30)
    n = 1280
    # Carrier in first 960 samples, then noise (PA cutoff at sample 960 = 3/4).
    carrier = _bandlimited_noise(n, seed=30)
    carrier = carrier / (np.max(np.abs(carrier)) + 1e-30)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.02
    sig = carrier.copy()
    sig[3 * n // 4 :] = noise[3 * n // 4 :]

    lag_samples = 4
    sig_b = np.roll(sig, lag_samples)

    b64_a = _encode_iq(sig)
    b64_b = _encode_iq(sig_b)
    rate = 62_500.0

    lag_ns, snr = cross_correlate_snippets(b64_a, b64_b, sample_rate_hz_a=rate, event_type="offset")
    expected_ns = lag_samples * 1e9 / rate
    tol_ns = 1.5 * 1e9 / rate

    assert snr > 1.5, f"Offset windowing SNR should be > 1.5, got {snr:.2f}"
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"Offset lag wrong: {lag_ns:.0f} ns vs {expected_ns:.0f} ns"
    )


# ---------------------------------------------------------------------------
# _carrier_event_to_db_dict includes xcorr fields
# ---------------------------------------------------------------------------

def test_carrier_event_to_db_dict_includes_snippet():
    """
    _carrier_event_to_db_dict must include iq_snippet_b64 and
    channel_sample_rate_hz so the solver receives them without reparsing raw_json.
    """
    from beagle_server.api import _carrier_event_to_db_dict
    from beagle_node.events.model import CarrierEvent, NodeLocation, SyncTransmitter

    snippet = _encode_iq(_bandlimited_noise(640, seed=5))
    event = CarrierEvent(
        node_id="test-node",
        node_location=NodeLocation(latitude_deg=47.6, longitude_deg=-122.3, altitude_m=50.0),
        channel_frequency_hz=155_100_000.0,
        sync_delta_ns=3_000_000,
        sync_transmitter=SyncTransmitter(
            station_id="KISW_99.9",
            frequency_hz=99_900_000.0,
            latitude_deg=47.6253,
            longitude_deg=-122.3563,
        ),
        sdr_mode="freq_hop",
        event_type="onset",
        onset_time_ns=1_700_000_000_000_000_000,
        iq_snippet_b64=snippet,
        channel_sample_rate_hz=64_000.0,
    )

    d = _carrier_event_to_db_dict(event)
    assert d["iq_snippet_b64"] == snippet
    assert d["channel_sample_rate_hz"] == pytest.approx(64_000.0)


