# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/demodulator.py."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.demodulator import FMDemodulator

RATE = 256_000.0   # typical post-decimation rate for the sync channel


def _fm_modulate(freq_dev_hz: float, n: int, rate: float) -> np.ndarray:
    """Generate an FM-modulated IQ signal with constant frequency deviation."""
    phase = 2.0 * np.pi * freq_dev_hz / rate * np.arange(n)
    return np.exp(1j * np.cumsum(np.full(n, 2.0 * np.pi * freq_dev_hz / rate))).astype(
        np.complex64
    )


# ---------------------------------------------------------------------------
# Basic shape / dtype
# ---------------------------------------------------------------------------

def test_output_length():
    dem = FMDemodulator(RATE)
    iq = np.zeros(1024, dtype=np.complex64)
    out = dem.process(iq)
    assert len(out) == 1024


def test_output_dtype():
    dem = FMDemodulator(RATE)
    iq = np.zeros(1024, dtype=np.complex64)
    out = dem.process(iq)
    assert out.dtype == np.float32


def test_empty_input():
    dem = FMDemodulator(RATE)
    out = dem.process(np.array([], dtype=np.complex64))
    assert len(out) == 0


def test_invalid_rate():
    with pytest.raises(ValueError, match="sample_rate_hz"):
        FMDemodulator(0.0)


# ---------------------------------------------------------------------------
# Frequency recovery
# ---------------------------------------------------------------------------

def test_demod_constant_tone():
    """
    A constant-deviation FM signal should demodulate to a constant frequency
    close to the modulating frequency.
    """
    rate = RATE
    dev_hz = 19_000.0   # 19 kHz - the FM pilot frequency
    n = 65_536

    # Build IQ: constant phase increment per sample
    delta_phi = 2.0 * np.pi * dev_hz / rate
    phases = np.arange(n) * delta_phi
    iq = np.exp(1j * phases).astype(np.complex64)

    dem = FMDemodulator(rate)
    out = dem.process(iq)

    # Skip first sample (uses arbitrary prev state)
    measured = np.mean(out[10:])
    assert abs(measured - dev_hz) < 5.0, f"Demodulated {measured:.1f} Hz, expected {dev_hz} Hz"


def test_demod_negative_deviation():
    """Negative frequency deviation should produce negative output."""
    rate = RATE
    dev_hz = -19_000.0
    n = 65_536
    delta_phi = 2.0 * np.pi * dev_hz / rate
    iq = np.exp(1j * np.arange(n) * delta_phi).astype(np.complex64)

    dem = FMDemodulator(rate)
    out = dem.process(iq)
    measured = np.mean(out[10:])
    assert abs(measured - dev_hz) < 5.0


def test_demod_zero_deviation():
    """Constant-phase IQ (DC carrier) should demodulate to ~0 Hz."""
    dem = FMDemodulator(RATE)
    iq = np.ones(4096, dtype=np.complex64)   # phase = 0, no deviation
    out = dem.process(iq)
    assert np.allclose(out[1:], 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# Cross-buffer continuity
# ---------------------------------------------------------------------------

def test_continuity_across_buffers():
    """
    Demodulating in two halves must produce the same result as one full call,
    except for the very first sample (which depends on initial state).
    """
    rate = RATE
    dev_hz = 19_000.0
    n = 65_536
    delta_phi = 2.0 * np.pi * dev_hz / rate
    iq = np.exp(1j * np.arange(n) * delta_phi).astype(np.complex64)

    dem_single = FMDemodulator(rate)
    dem_split  = FMDemodulator(rate)

    out_single = dem_single.process(iq)
    out_a = dem_split.process(iq[: n // 2])
    out_b = dem_split.process(iq[n // 2 :])
    out_split = np.concatenate([out_a, out_b])

    # All samples including the boundary should agree
    np.testing.assert_allclose(out_single, out_split, atol=1.0)


def test_reset_restores_initial_state():
    """After reset(), results match a freshly constructed demodulator."""
    rate = RATE
    n = 4096
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

    dem_fresh = FMDemodulator(rate)
    dem_reuse = FMDemodulator(rate)

    dem_reuse.process(iq)   # warm up
    dem_reuse.reset()

    np.testing.assert_allclose(dem_fresh.process(iq), dem_reuse.process(iq), atol=1e-3)
