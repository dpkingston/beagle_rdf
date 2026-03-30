# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/decimator.py."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.decimator import Decimator

INPUT_RATE = 2_048_000.0
CUTOFF     = 128_000.0
DECIM      = 8          # -> 256 kHz output


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_output_rate():
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    assert dec.output_rate_hz == INPUT_RATE / DECIM


def test_invalid_decimation():
    with pytest.raises(ValueError, match="decimation"):
        Decimator(0, INPUT_RATE, CUTOFF)


def test_invalid_cutoff_too_high():
    with pytest.raises(ValueError, match="cutoff_hz"):
        Decimator(DECIM, INPUT_RATE, INPUT_RATE / 2)


def test_invalid_cutoff_zero():
    with pytest.raises(ValueError, match="cutoff_hz"):
        Decimator(DECIM, INPUT_RATE, 0.0)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

def test_output_length_exact_multiple():
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    iq = np.zeros(8192, dtype=np.complex64)
    out = dec.process(iq)
    assert len(out) == 8192 // DECIM


def test_output_length_non_multiple():
    """Input length not a multiple of decimation - extra samples consumed."""
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    iq = np.zeros(8195, dtype=np.complex64)   # 8195 // 8 = 1024
    out = dec.process(iq)
    assert len(out) == 8195 // DECIM


def test_output_dtype():
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
    out = dec.process(iq)
    assert out.dtype == np.complex64


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------

def _power_db(signal: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.abs(signal) ** 2) + 1e-30))


def _tone(freq_hz: float, n: int, rate: float) -> np.ndarray:
    t = np.arange(n) / rate
    return np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)


def test_passband_not_attenuated():
    """10 kHz tone (well inside passband) should pass with < 3 dB loss."""
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    n = 131_072
    iq = _tone(10_000, n, INPUT_RATE)
    out = dec.process(iq)
    skip = dec.group_delay_samples + 20  # skip transient
    atten = _power_db(out[skip:]) - _power_db(iq[skip * DECIM:])
    assert atten > -3.0, f"Passband attenuation {atten:.1f} dB > -3 dB"


def test_stopband_attenuated():
    """500 kHz tone (well above cutoff) should be attenuated > 40 dB."""
    dec = Decimator(DECIM, INPUT_RATE, CUTOFF)
    n = 131_072
    iq = _tone(500_000, n, INPUT_RATE)
    out = dec.process(iq)
    skip = dec.group_delay_samples + 20
    atten = _power_db(out[skip:]) - _power_db(iq[skip * DECIM:])
    assert atten < -40.0, f"Stopband attenuation {atten:.1f} dB < -40 dB"


# ---------------------------------------------------------------------------
# State continuity
# ---------------------------------------------------------------------------

def test_state_continuity_across_buffers():
    """
    Two consecutive half-length calls must produce the same result as one
    full-length call (verifies filter state is preserved across buffers).
    """
    rng = np.random.default_rng(42)
    n = 65_536
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

    dec_single = Decimator(DECIM, INPUT_RATE, CUTOFF)
    dec_split  = Decimator(DECIM, INPUT_RATE, CUTOFF)

    out_single = dec_single.process(iq)
    out_a = dec_split.process(iq[: n // 2])
    out_b = dec_split.process(iq[n // 2 :])
    out_split = np.concatenate([out_a, out_b])

    # Tolerance is 1e-3: on macOS the vDSP_desamp backend accumulates the
    # 127-tap FIR in a different order than the upfirdn fallback, giving
    # ~6e-4 error at the exact buffer boundary (1 sample out of 8192).
    # This is inherent float32 summation-order variance, not a state bug.
    np.testing.assert_allclose(
        out_single.real, out_split.real, atol=1e-3,
        err_msg="Real parts differ across buffer boundary"
    )
    np.testing.assert_allclose(
        out_single.imag, out_split.imag, atol=1e-3,
        err_msg="Imag parts differ across buffer boundary"
    )


def test_reset_clears_state():
    """After reset(), output matches a freshly constructed Decimator."""
    rng = np.random.default_rng(7)
    n = 4096
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

    dec_fresh = Decimator(DECIM, INPUT_RATE, CUTOFF)
    dec_reuse = Decimator(DECIM, INPUT_RATE, CUTOFF)

    # Warm up dec_reuse with some data, then reset
    dec_reuse.process(iq)
    dec_reuse.reset()

    out_fresh = dec_fresh.process(iq)
    out_reuse = dec_reuse.process(iq)

    np.testing.assert_allclose(out_fresh, out_reuse, atol=1e-4)


# ---------------------------------------------------------------------------
# Decimation-by-1 (passthrough)
# ---------------------------------------------------------------------------

def test_decimation_by_one():
    """Decimation=1 with cutoff below Nyquist should filter but not downsample."""
    dec = Decimator(1, INPUT_RATE, CUTOFF)
    iq = np.zeros(1024, dtype=np.complex64)
    out = dec.process(iq)
    assert len(out) == 1024
    assert dec.output_rate_hz == INPUT_RATE
