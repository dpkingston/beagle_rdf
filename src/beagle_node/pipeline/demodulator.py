# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
FM discriminator demodulator.

Converts a complex IQ stream (bandlimited to one FM channel) into
instantaneous frequency deviation - i.e., demodulated audio.

Algorithm: angle(conj(x[n-1]) * x[n]) / (2pi * deltat)
This is the standard FM discriminator, numerically equivalent to
d/dt arg(x(t)) but computed sample-by-sample without unwrapping.

Output units: normalised frequency deviation in the range (-Fs/2, +Fs/2).
Multiply by sample_rate_hz / max_deviation_hz to get a +/-1-normalised
audio signal, but for pilot extraction we work directly in Hz.
"""

from __future__ import annotations

import numpy as np


class FMDemodulator:
    """
    Instantaneous-frequency FM discriminator.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the *input* IQ stream (after decimation).
    """

    def __init__(self, sample_rate_hz: float) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
        self._rate = float(sample_rate_hz)
        # Last sample of the previous buffer, for cross-buffer continuity
        self._prev: np.complex64 = np.complex64(1.0 + 0j)

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    def process(self, iq: np.ndarray) -> np.ndarray:
        """
        Demodulate an IQ buffer to instantaneous frequency.

        Parameters
        ----------
        iq : np.ndarray, complex64 or complex128
            Input IQ samples.

        Returns
        -------
        np.ndarray, float32
            Instantaneous frequency in Hz, same length as input.
            First sample uses the last sample of the previous call.
        """
        if len(iq) == 0:
            return np.empty(0, dtype=np.float32)

        iq = np.asarray(iq, dtype=np.complex64)

        # Prepend last sample from previous buffer for continuity
        extended = np.empty(len(iq) + 1, dtype=np.complex64)
        extended[0] = self._prev
        extended[1:] = iq

        # Discriminator: angle of conjugate product
        conj_prod = np.conj(extended[:-1]) * extended[1:]
        phase_diff = np.angle(conj_prod)  # radians per sample

        # Convert to Hz: phase_diff / (2pi) * sample_rate
        freq_hz = (phase_diff / (2.0 * np.pi) * self._rate).astype(np.float32)

        # Store last sample for next call
        self._prev = np.complex64(iq[-1])

        return freq_hz

    def reset(self) -> None:
        """Reset state (call when starting a new stream)."""
        self._prev = np.complex64(1.0 + 0j)
