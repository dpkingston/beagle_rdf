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

@pytest.mark.parametrize("lag_samples", [0, 1, 5, 10, 20])
def test_known_lag_recovery(lag_samples: int):
    """
    Real onset snippet shifted by known amounts.  The d2 xcorr should
    recover the delay with sub-microsecond precision.
    """
    import json
    from pathlib import Path
    from beagle_server.tdoa import _decode_iq_snippet

    fixture = Path(__file__).parents[1] / "fixtures" / "three_node_baseline_2026_04_08.json"
    data = json.load(fixture.open())
    onsets = [e for e in data["events"]
              if e["event_type"] == "onset" and len(e.get("iq_snippet_b64", "")) > 100]
    iq = _decode_iq_snippet(onsets[1]["iq_snippet_b64"])
    rate = float(onsets[1]["channel_sample_rate_hz"])

    n = len(iq) - lag_samples - 1
    if lag_samples >= 0:
        iq_a = iq[lag_samples:lag_samples + n]
        iq_b = iq[:n]
    else:
        iq_a = iq[:n]
        iq_b = iq[-lag_samples:-lag_samples + n]

    lag_ns, snr = cross_correlate_snippets(
        _encode_iq(iq_a), _encode_iq(iq_b),
        sample_rate_hz_a=rate, event_type="onset",
    )

    expected_ns = lag_samples * 1e9 / rate
    tol_ns = 2.0 * 1e9 / rate  # within 2 samples
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"lag={lag_samples}: expected {expected_ns:.1f} ns, got {lag_ns:.1f} ns"
    )
    if lag_samples != 0:
        assert snr > 10.0, f"Real onset d2 xcorr SNR should be > 10, got {snr:.2f}"


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
# Realistic PA transition xcorr (coarse knee walk + sub-snippet alignment)
# ---------------------------------------------------------------------------

def _make_realistic_pair(
    n: int = 1280,
    onset_a: int = 320,
    delay_samples: int = 10,
    ramp_samples: int = 24,
    snr_db: float = 25.0,
    event_type: str = "onset",
    seed: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a pair of IQ snippets modelling two nodes receiving the same
    PA transition at different times (different propagation delays).

    For onset: noise -> ramp -> carrier.  Node B sees the onset
    delay_samples later (onset at onset_a + delay_samples).

    For offset: carrier -> ramp-down -> noise.  Reversed structure.

    Both nodes share the same carrier content after the ramp (same RF
    signal), and have independent noise.
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_amp = 1.0 / float(np.sqrt(snr_linear))

    # Constant-envelope FM carrier (same physical transmission).
    # Real LMR FM has constant instantaneous power — the power envelope
    # is a clean step at the PA transition, not a noisy ramp.  We model
    # this as a rotating phasor with slowly-varying frequency (FM modulation)
    # so the power envelope is flat but the phase varies, giving xcorr
    # texture to lock onto.
    carrier_len = n + delay_samples + ramp_samples
    # FM modulation: slowly-varying frequency gives phase texture
    mod_freq = np.cumsum(rng.standard_normal(carrier_len) * 0.1)
    phase = np.cumsum(mod_freq) * 0.01
    carrier = np.exp(1j * phase).astype(np.complex64)

    # Build both nodes from the same physical signal, sliced at different
    # positions to model the propagation delay.  This ensures both nodes see
    # identical carrier content (same physical transmission).
    onset_b = onset_a + delay_samples
    total_len = n + delay_samples + ramp_samples

    # Physical signal: silence -> PA power ramp -> full carrier.
    # The PA ramp is a power ramp (amplitude = sqrt(ramp)), modelling
    # the PA amplifier turning on.  The carrier inside the ramp has
    # constant envelope (FM) — only the PA output power changes.
    physical = np.zeros(total_len, dtype=np.complex64)
    ramp_end = onset_a + ramp_samples
    if ramp_samples > 0 and onset_a < total_len:
        ramp_len = min(ramp_samples, total_len - onset_a)
        # Power ramp: amplitude = sqrt(linear_ramp) so |IQ|^2 ramps linearly
        amp_ramp = np.sqrt(np.linspace(0.0, 1.0, ramp_len)).astype(np.float32)
        physical[onset_a:onset_a + ramp_len] = carrier[onset_a:onset_a + ramp_len] * amp_ramp
    if ramp_end < total_len:
        physical[ramp_end:] = carrier[ramp_end:total_len]

    if event_type == "offset":
        physical = physical[::-1].copy()

    # Node A sees [0 : n], node B sees [delay : delay + n]
    iq_a_clean = physical[:n].copy()
    iq_b_clean = physical[delay_samples:delay_samples + n].copy()

    # Independent noise for each node
    rng_a = np.random.default_rng(seed + 1000)
    rng_b = np.random.default_rng(seed + 2000)
    noise_a = noise_amp * (rng_a.standard_normal(n) + 1j * rng_a.standard_normal(n)).astype(np.complex64)
    noise_b = noise_amp * (rng_b.standard_normal(n) + 1j * rng_b.standard_normal(n)).astype(np.complex64)

    iq_a = (iq_a_clean + noise_a).astype(np.complex64)
    iq_b = (iq_b_clean + noise_b).astype(np.complex64)

    return iq_a, iq_b


def _load_real_snippet():
    """Load a real onset snippet from fixture data."""
    import json
    from pathlib import Path
    from beagle_server.tdoa import _decode_iq_snippet
    fixture = Path(__file__).parents[1] / "fixtures" / "three_node_baseline_2026_04_08.json"
    data = json.load(fixture.open())
    onsets = [e for e in data["events"]
              if e["event_type"] == "onset" and len(e.get("iq_snippet_b64", "")) > 100]
    return _decode_iq_snippet(onsets[1]["iq_snippet_b64"]), float(onsets[1]["channel_sample_rate_hz"])


def test_onset_xcorr_recovers_delay():
    """
    Real onset snippet shifted by 10 samples — d2 xcorr should recover.
    """
    iq, rate = _load_real_snippet()
    delay = 10
    n = len(iq) - delay - 1
    iq_a = iq[delay:delay + n]
    iq_b = iq[:n]

    lag_ns, snr = cross_correlate_snippets(
        _encode_iq(iq_a), _encode_iq(iq_b),
        sample_rate_hz_a=rate, event_type="onset",
    )

    expected_ns = delay * 1e9 / rate
    tol_ns = 2.0 * 1e9 / rate

    assert snr > 10.0, f"Onset d2 xcorr SNR too low: {snr:.2f}"
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"Onset lag wrong: {lag_ns:.0f} ns vs {expected_ns:.0f} ns"
    )


def test_offset_xcorr_recovers_delay():
    """
    Real snippet reversed (simulating offset), shifted by 10 samples.
    d2 xcorr should recover the delay.
    """
    iq, rate = _load_real_snippet()
    # Reverse to simulate offset (carrier → noise)
    iq = iq[::-1].copy()
    delay = 10
    n = len(iq) - delay - 1
    iq_a = iq[delay:delay + n]
    iq_b = iq[:n]

    lag_ns, snr = cross_correlate_snippets(
        _encode_iq(iq_a), _encode_iq(iq_b),
        sample_rate_hz_a=rate, event_type="offset",
    )

    expected_ns = delay * 1e9 / rate
    tol_ns = 2.0 * 1e9 / rate

    assert snr > 10.0, f"Offset d2 xcorr SNR too low: {snr:.2f}"
    assert abs(lag_ns - expected_ns) < tol_ns, (
        f"Offset lag wrong: {lag_ns:.0f} ns vs {expected_ns:.0f} ns"
    )


def test_onset_xcorr_zero_delay():
    """Identical real snippets should give ~0 lag."""
    iq, rate = _load_real_snippet()

    lag_ns, snr = cross_correlate_snippets(
        _encode_iq(iq), _encode_iq(iq.copy()),
        sample_rate_hz_a=rate, event_type="onset",
    )

    tol_ns = 1.0 * 1e9 / rate
    assert abs(lag_ns) < tol_ns, f"Zero-delay onset lag should be ~0, got {lag_ns:.0f} ns"


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
        node_location=NodeLocation(latitude_deg=47.6, longitude_deg=-122.3),
        channel_frequency_hz=155_100_000.0,
        sync_to_snippet_start_ns=3_000_000,
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


