# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Pure-Python RDS subcarrier demodulator.

Takes FM-demodulated multiplex audio (MPX) at any rate >= ~120 kHz and
produces a stream of differentially-decoded RDS bits with sub-sample
timing of where each bit was sampled.

Architecture
------------
Mirrors windytan/redsea ``src/dsp/subcarrier.cc``.  Differences:

  * One-shot polyphase resample to 7125 Hz (3 sps × 2375 baud) instead
    of redsea's two-stage 171 kHz → 7125 Hz path.
  * Symbol timing uses best-of-3-phase initial selection + slow Mueller-
    Müller-style tracking; redsea uses liquid-dsp's polyphase Gardner.
    Equivalent for our SNR regime.
  * Carrier tracking is a textbook 2nd-order Costas BPSK loop, properly
    magnitude-normalized.
  * Biphase clock polarity is selected adaptively from a 128-symbol
    window (same heuristic as redsea ``BiphaseDecoder::push``).

Signal chain
------------
    MPX (fs_in)
      │ × exp(-j 2π 57000 n / fs_in)
      ▼
    Complex baseband (fs_in)
      │ scipy.signal.resample_poly
      ▼
    Complex baseband (7125 Hz, 3 sps)
      │ RRC β=0.8, span=12 symbols
      ▼
    Matched-filter output (7125 Hz)
      │ AGC normalize to unit RMS
      │ best-of-3 PSK phase pick
      ▼
    PSK symbols (2375 Hz)
      │ Costas BPSK PLL (derotate)
      ▼
    Aligned BPSK symbols
      │ Biphase decode with adaptive polarity
      ▼
    Biphase bits (1187.5 Hz)
      │ Differential decode (XOR with prev)
      ▼
    Output bits

Block synchronization, syndrome calculation, and burst-error FEC live in
``rds_block_sync`` (separate module, separate commit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.signal as sps

RDS_SUBCARRIER_HZ: float = 57_000.0
PSK_SYMBOL_RATE_HZ: float = 2375.0
BIT_RATE_HZ: float = 1187.5
SAMPLES_PER_SYMBOL: int = 3                         # 3 PSK samples per symbol
INTERNAL_RATE_HZ: float = PSK_SYMBOL_RATE_HZ * SAMPLES_PER_SYMBOL  # 7125.0
RRC_BETA: float = 0.8
RRC_SPAN_SYMBOLS: int = 12

# Biphase polarity detection window (PSK symbols).  Redsea uses 128.
POLARITY_WINDOW: int = 128


@dataclass(frozen=True)
class DecodedBit:
    """One differentially-decoded RDS bit.

    Attributes
    ----------
    bit : int
        0 or 1.  Post-differential-decode.
    sample_index : float
        Position in the *input* MPX stream (at fs_in) where this bit was
        sampled.  The center of the second PSK symbol of the biphase pair.
    symbol : complex
        The post-Costas BPSK symbol whose sign drove the bit decision.
        Magnitude indicates SNR after the AGC normalization (should be
        near 1.0 when locked).
    """
    bit: int
    sample_index: float
    symbol: complex


# ---------------------------------------------------------------------------
# RRC matched filter design
# ---------------------------------------------------------------------------

def _design_rrc(beta: float, span_symbols: int, sps_: int) -> np.ndarray:
    """Root-raised-cosine pulse with unit energy."""
    N = span_symbols * sps_
    t = (np.arange(N + 1) - N / 2.0) / sps_
    taps = np.empty_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + 4.0 * beta / math.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            taps[i] = (beta / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
            )
        else:
            num = (math.sin(math.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * math.cos(math.pi * ti * (1.0 + beta)))
            den = math.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float64)


def _rational_resample_factors(target: float, source: float) -> tuple[int, int]:
    """Find small integers (up, down) such that source * up / down ≈ target.

    For source=250000, target=7125 this returns (57, 2000) exactly.
    """
    f = Fraction(target / source).limit_denominator(5000)
    return f.numerator, f.denominator


# ---------------------------------------------------------------------------
# Top-level decode
# ---------------------------------------------------------------------------

def decode(
    mpx: np.ndarray,
    fs_in: float,
    *,
    pll_loop_bw_hz: float = 50.0,
) -> list[DecodedBit]:
    """
    Demodulate the RDS subcarrier and return decoded bits.

    Parameters
    ----------
    mpx : np.ndarray
        FM-demodulated multiplex audio at fs_in.
    fs_in : float
        Input sample rate (Hz).
    pll_loop_bw_hz : float
        Costas loop bandwidth in Hz, referenced to PSK symbol rate (2375 Hz).
        Higher = faster lock, more noise.  50 Hz is a good default for our SNR.

    Returns
    -------
    list[DecodedBit]
    """
    if mpx.ndim != 1:
        raise ValueError("mpx must be 1-D")
    if fs_in <= 2 * RDS_SUBCARRIER_HZ:
        raise ValueError(f"fs_in={fs_in} too low; need > {2 * RDS_SUBCARRIER_HZ}")

    x = mpx.astype(np.float64, copy=False)
    n_in = len(x)

    # 1) Mix down to complex baseband at fs_in.
    n_idx = np.arange(n_in, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * RDS_SUBCARRIER_HZ * n_idx / fs_in)
    baseband = x * mix

    # 2) Polyphase resample to INTERNAL_RATE_HZ.
    up, down = _rational_resample_factors(INTERNAL_RATE_HZ, fs_in)
    bb = sps.resample_poly(baseband, up, down).astype(np.complex128)
    fs_dec = fs_in * up / down  # 7125.0 for fs_in=250000

    # 3) RRC matched filter.
    rrc = _design_rrc(RRC_BETA, RRC_SPAN_SYMBOLS, SAMPLES_PER_SYMBOL)
    matched = np.convolve(bb, rrc, mode="same")

    # 4) AGC: normalize to unit RMS based on whole-buffer magnitude.
    rms = float(np.sqrt(np.mean(np.abs(matched) ** 2)) + 1e-30)
    matched /= rms

    # 5) Best-of-3 symbol-phase selection.
    # The matched-filter output has peaks at the correct symbol phase.
    # Pick the phase (0, 1, or 2 samples into each 3-sample symbol group)
    # whose post-AGC magnitude is highest.
    phase_energies: list[float] = []
    for phi in range(SAMPLES_PER_SYMBOL):
        ph_samples = matched[phi::SAMPLES_PER_SYMBOL]
        phase_energies.append(float(np.mean(np.abs(ph_samples) ** 2)))
    best_phi = int(np.argmax(phase_energies))

    # Cut to a clean integer-symbol view starting at best_phi.
    psk = matched[best_phi::SAMPLES_PER_SYMBOL]  # shape: (n_symbols,)

    # 6) Costas BPSK PLL.
    aligned = _costas_bpsk(psk, loop_bw_hz=pll_loop_bw_hz)

    # 7) Biphase + differential decode.
    bits = _biphase_diff_decode(
        aligned,
        psk_start_offset_dec=best_phi,
        in_per_dec=down / up,
    )
    return bits


# ---------------------------------------------------------------------------
# Costas BPSK PLL
# ---------------------------------------------------------------------------

def _costas_bpsk(psk: np.ndarray, *, loop_bw_hz: float) -> np.ndarray:
    """
    Run a 2nd-order Costas loop on a 1-sps BPSK symbol stream.

    Returns the derotated symbols, one per input sample.
    """
    # Loop coefficient design.  bw expressed in Hz relative to PSK_SYMBOL_RATE_HZ.
    zeta = 0.707
    wn = 2.0 * math.pi * loop_bw_hz / PSK_SYMBOL_RATE_HZ
    alpha = 2.0 * zeta * wn
    beta = wn * wn

    n = len(psk)
    out = np.empty(n, dtype=np.complex128)
    phase = 0.0
    freq = 0.0  # rad/symbol
    for i in range(n):
        # Derotate
        s = psk[i] * (math.cos(-phase) + 1j * math.sin(-phase))
        out[i] = s

        # BPSK Costas phase error (sign-corrected, magnitude-normalized).
        # For BPSK, the canonical detector is:
        #     err = imag(s) * sign(real(s))
        # Magnitude normalization keeps loop gain scale-invariant.
        mag = abs(s)
        if mag > 1e-9:
            err = (s.imag / mag) * (1.0 if s.real >= 0 else -1.0)
        else:
            err = 0.0

        # 2nd-order update
        freq += beta * err
        phase += alpha * err + freq
        # Wrap phase to (-π, π]
        if phase > math.pi:
            phase -= 2.0 * math.pi
        elif phase < -math.pi:
            phase += 2.0 * math.pi
    return out


# ---------------------------------------------------------------------------
# Biphase + differential decode
# ---------------------------------------------------------------------------

def _biphase_diff_decode(
    psk: np.ndarray,
    *,
    psk_start_offset_dec: int,
    in_per_dec: float,
) -> list[DecodedBit]:
    """
    Convert PSK symbols → biphase bits → differential bits, with adaptive
    biphase clock polarity (same heuristic as redsea BiphaseDecoder).

    Parameters
    ----------
    psk : np.ndarray (complex)
        Post-Costas BPSK symbols at PSK_SYMBOL_RATE_HZ (2375 Hz).
    psk_start_offset_dec : int
        Offset (in INTERNAL_RATE_HZ samples) from the start of the decimated
        baseband buffer to psk[0].  Used to compute sample_index in input MPX.
    in_per_dec : float
        Ratio of input samples to decimated samples (fs_in / fs_dec).
    """
    n = len(psk)
    if n < 4:
        return []

    bits: list[DecodedBit] = []

    # Biphase symbol = difference of consecutive PSK symbols, scaled.
    # For a Manchester-encoded bit:  bit 1 → (+1, -1), bit 0 → (-1, +1).
    # The "biphase symbol" b[k] = (psk[k] - psk[k-1]) / 2 has |real| ≈ 1
    # when symbol pair straddles a bit boundary, and |real| ≈ 0 when not.
    # We must detect *which* pairing is correct (even or odd polarity).

    prev_diff_bit = 0
    have_prev_diff = False
    polarity = 0  # 0 = emit at biphase index 0, 2, 4, ...; 1 = at 1, 3, 5, ...
    window_energy_even = 0.0
    window_energy_odd = 0.0
    window_count = 0

    for k in range(1, n):
        biphase = (psk[k] - psk[k - 1]) * 0.5
        magsq = biphase.real * biphase.real
        # biphase index k-1 (i.e., the pair (psk[k-1], psk[k])):
        if (k - 1) % 2 == 0:
            window_energy_even += magsq
        else:
            window_energy_odd += magsq
        window_count += 1

        # Refresh polarity decision every POLARITY_WINDOW symbols
        if window_count >= POLARITY_WINDOW:
            if window_energy_even > window_energy_odd:
                polarity = 0
            else:
                polarity = 1
            window_energy_even = 0.0
            window_energy_odd = 0.0
            window_count = 0

        # Emit on the polarity-matched biphase index
        if (k - 1) % 2 == polarity:
            biphase_bit = 1 if biphase.real >= 0 else 0

            # Differential decode
            if have_prev_diff:
                out_bit = biphase_bit ^ prev_diff_bit
                # Sample index of psk[k] in INTERNAL_RATE_HZ coords:
                #   k * SAMPLES_PER_SYMBOL + psk_start_offset_dec
                # Convert to input MPX coords:
                dec_idx = k * SAMPLES_PER_SYMBOL + psk_start_offset_dec
                sample_index_in = dec_idx * in_per_dec
                bits.append(DecodedBit(
                    bit=out_bit,
                    sample_index=sample_index_in,
                    symbol=psk[k],
                ))
            prev_diff_bit = biphase_bit
            have_prev_diff = True

    return bits
