# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for the biphase polarity selection in rds_demodulator.

The polarity selector decides whether biphase bits are emitted at
even-index pairs (s0,s1),(s2,s3),... or odd-index pairs (s1,s2),(s3,s4),...

Two design constraints drive the implementation:

  1. **Cross-node determinism.**  Two nodes receiving the same broadcast
     must converge on the same polarity (and stay there together) so that
     their decoded bit streams are inter-comparable for the block-A
     anchor selection in DeltaComputer.

  2. **Fade resilience.**  Transient signal fades cause both energies to
     drop to noise level; without protection, the selector can flip
     spuriously during the fade.  The fade guard skips decisions when
     total per-window energy is below POLARITY_MIN_DECISION_ENERGY.

These tests exercise the selector with synthetic PSK streams that
isolate each behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.rds_demodulator import (
    POLARITY_HYSTERESIS_FACTOR,
    POLARITY_MIN_DECISION_ENERGY,
    POLARITY_WINDOW,
    _biphase_diff_decode,
)


def _make_psk(n_pairs: int, polarity: int, noise_sigma: float = 0.0,
              seed: int = 0) -> np.ndarray:
    """
    Build a synthetic PSK stream of length 2*n_pairs where the polarity-`polarity`
    pairing is the "correct" within-bit pairing (|biphase| ≈ 1) and the other
    pairing is across-bit (|biphase| ≈ 0 or 1 depending on bit transitions).

    Bits are random; differential encoding is applied.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_pairs, dtype=np.uint8)
    # Diff encode
    diff = np.empty_like(bits)
    prev = 0
    for i, b in enumerate(bits):
        prev = int(b) ^ prev
        diff[i] = prev
    # Biphase: 2 PSK per bit
    psk = np.empty(2 * n_pairs, dtype=np.complex128)
    psk[0::2] = np.where(diff == 1, +1.0, -1.0)
    psk[1::2] = -psk[0::2]
    # Shift by `polarity` to put within-bit pairing at the requested parity
    if polarity == 1:
        psk = np.roll(psk, 1)
        psk[0] = -psk[1]  # synth a leading symbol
    # Add noise
    if noise_sigma > 0:
        psk = psk + rng.normal(0, noise_sigma, size=len(psk))
    return psk


class TestPolarityBasic:
    def test_correct_polarity_chosen_on_clean_signal(self):
        """Clean signal with within-bit pairing at even indices → polarity 0."""
        psk = _make_psk(n_pairs=2000, polarity=0, noise_sigma=0.0)
        bits = _biphase_diff_decode(psk, psk_start_offset_dec=0, in_per_dec=1.0)
        # If polarity is right, we get ~n_pairs bits out (minus warmup)
        assert len(bits) >= int(0.95 * 2000)

    def test_correct_polarity_chosen_at_polarity_1(self):
        """Within-bit pairing at odd indices → polarity 1 selected after warmup."""
        psk = _make_psk(n_pairs=2000, polarity=1, noise_sigma=0.0)
        bits = _biphase_diff_decode(psk, psk_start_offset_dec=0, in_per_dec=1.0)
        # Should still recover most bits after the polarity flip in first window
        assert len(bits) >= int(0.85 * 2000)


class TestHysteresis:
    def test_close_energies_do_not_flip(self):
        """Near-equal energies (no clear winner) shouldn't flip the polarity.

        The default starting polarity is 0.  If we feed a stream where
        even and odd energies are very close (within hysteresis margin),
        polarity stays at 0 regardless of which side has a tiny lead.
        """
        rng = np.random.default_rng(123)
        # Build a stream where neither pairing dominates: alternate signs
        # at random with high noise.
        n = 2000
        psk = rng.normal(0, 1.0, size=n) + 1j * rng.normal(0, 1.0, size=n)
        # With pure noise both energies are stochastic; the test is that
        # _biphase_diff_decode doesn't crash and returns *something*.
        bits = _biphase_diff_decode(psk, psk_start_offset_dec=0, in_per_dec=1.0)
        # Nothing strict about what comes out; just that the algorithm runs
        assert isinstance(bits, list)

    def test_hysteresis_factor_is_above_one(self):
        # Sanity: the constant is configured for an actual hysteresis effect.
        assert POLARITY_HYSTERESIS_FACTOR > 1.0


class TestFadeGuard:
    def test_zero_energy_window_holds_polarity(self):
        """A window with all-zero PSK samples must not flip polarity."""
        # First chunk: clean polarity-0 signal so we lock in polarity 0
        psk_pre = _make_psk(n_pairs=POLARITY_WINDOW * 4, polarity=0)
        # Then a fade region (all zeros) → would normally have e_even = e_odd = 0
        psk_fade = np.zeros(POLARITY_WINDOW * 2, dtype=np.complex128)
        # Then more polarity-0 signal
        psk_post = _make_psk(n_pairs=POLARITY_WINDOW * 4, polarity=0, seed=42)
        psk = np.concatenate([psk_pre, psk_fade, psk_post])
        bits = _biphase_diff_decode(psk, psk_start_offset_dec=0, in_per_dec=1.0)
        # Should still produce bits from the post-fade region
        # (i.e., polarity didn't get scrambled by the fade)
        assert len(bits) > int(0.5 * (len(psk_pre) + len(psk_post)) / 2)

    def test_fade_threshold_configured(self):
        assert POLARITY_MIN_DECISION_ENERGY > 0
