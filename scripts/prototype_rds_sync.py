#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Phase 0 prototype: RDS bit transition detection from FM broadcast IQ.

Demonstrates that RDS BPSK bit boundaries can be reliably detected and
used as sync events at 1187.5 Hz (842 usec period).

Usage:
    # Capture IQ from KUOW on the node:
    python3 scripts/prototype_rds_sync.py --capture /tmp/kuow_3s.npy --duration 3

    # Process existing IQ file:
    python3 scripts/prototype_rds_sync.py --input /tmp/kuow_sync_3s.npy

    # Compare pilot sync events vs RDS sync events:
    python3 scripts/prototype_rds_sync.py --input /tmp/kuow_sync_3s.npy --compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project src to path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.sync_detector import FMPilotSyncDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PILOT_FREQ_HZ = 19_000.0
RDS_SUBCARRIER_HZ = 57_000.0       # 3 * pilot
RDS_BIT_RATE = 1187.5               # pilot / 16
RDS_BIT_PERIOD_US = 1e6 / RDS_BIT_RATE  # 842.105 usec
SDR_RATE = 2_000_000.0
SYNC_DEC = 8
SYNC_RATE = SDR_RATE / SYNC_DEC     # 250,000 Hz
SAMPLES_PER_BIT = SYNC_RATE / RDS_BIT_RATE  # ~210.526 at 250 kHz

# CRC-10 for RDS block sync (polynomial x^10+x^8+x^7+x^5+x^4+x^3+1)
_RDS_POLY = 0x1B9
_RDS_SYNDROMES = {
    0x3D8: "A",
    0x3D4: "B",
    0x25C: "C",
    0x3CC: "C'",
    0x258: "D",
}


# ---------------------------------------------------------------------------
# RDS signal processing
# ---------------------------------------------------------------------------

def recover_pilot_phase(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Recover the instantaneous pilot phase from demodulated FM audio.

    Uses a narrow BPF around 19 kHz followed by analytic signal extraction
    to get the instantaneous phase of the pilot tone.
    """
    from scipy.signal import firwin, lfilter

    # Narrow BPF around 19 kHz
    nyq = rate / 2.0
    bw = 200.0  # +/- 200 Hz
    lo = (PILOT_FREQ_HZ - bw) / nyq
    hi = (PILOT_FREQ_HZ + bw) / nyq
    taps = firwin(255, [lo, hi], pass_zero=False, window="hamming")

    pilot_filtered = lfilter(taps, 1.0, audio.astype(np.float64))

    # Analytic signal via Hilbert-like approach: correlate with complex template
    # in short windows to get instantaneous phase
    t = np.arange(len(audio)) / rate
    ref_i = np.cos(2 * np.pi * PILOT_FREQ_HZ * t)
    ref_q = -np.sin(2 * np.pi * PILOT_FREQ_HZ * t)

    # Smooth with a short window to get the pilot envelope and phase
    smooth_len = int(rate / PILOT_FREQ_HZ * 4)  # ~4 pilot cycles
    kernel = np.ones(smooth_len) / smooth_len

    i_comp = np.convolve(pilot_filtered * ref_i, kernel, mode='same')
    q_comp = np.convolve(pilot_filtered * ref_q, kernel, mode='same')

    pilot_phase = np.arctan2(q_comp, i_comp)
    return pilot_phase


def mix_rds_to_baseband(
    audio: np.ndarray,
    rate: float,
    pilot_locked: bool = True,
) -> np.ndarray:
    """
    Mix the RDS subcarrier to complex baseband.

    When pilot_locked=True, recovers the 19 kHz pilot via BPF and PLL,
    then triples to get a phase-locked 57 kHz reference. This is critical
    for low-jitter bit timing -- a free-running 57 kHz has phase drift
    relative to the actual broadcast signal.
    """
    from scipy.signal import firwin, lfilter

    if pilot_locked:
        # Step 1: Isolate the 19 kHz pilot with a narrow BPF
        nyq = rate / 2.0
        bw = 200.0
        lo = (PILOT_FREQ_HZ - bw) / nyq
        hi = (PILOT_FREQ_HZ + bw) / nyq
        taps = firwin(255, [lo, hi], pass_zero=False, window="hamming")
        pilot = lfilter(taps, 1.0, audio.astype(np.float64))

        # Step 2: Square the pilot to get a 38 kHz tone, then triple
        # the pilot directly by cubing (or squaring + mixing).
        # Simpler approach: the BPF output is a clean 19 kHz tone.
        # Hard-limit it to a square wave, then extract the 3rd harmonic (57 kHz).
        # The square wave has odd harmonics: 19, 57, 95, ...
        # So squaring gives us 57 kHz directly in the harmonics.
        pilot_sq = np.sign(pilot)  # hard limiter -> square wave

        # BPF the square wave at 57 kHz to isolate the 3rd harmonic
        bw_57 = 2500.0  # wider BW to track the RDS signal
        lo_57 = (RDS_SUBCARRIER_HZ - bw_57) / nyq
        hi_57 = min((RDS_SUBCARRIER_HZ + bw_57) / nyq, 0.999)
        taps_57 = firwin(127, [lo_57, hi_57], pass_zero=False, window="hamming")
        carrier_57 = lfilter(taps_57, 1.0, pilot_sq)

        # Use the recovered 57 kHz carrier and its Hilbert transform as I/Q ref.
        # The Hilbert transform gives us the quadrature component.
        from scipy.signal import hilbert
        carrier_analytic = hilbert(carrier_57)
        ref_i = np.real(carrier_analytic).astype(np.float32)
        ref_q = -np.imag(carrier_analytic).astype(np.float32)

        # Normalize
        env = np.abs(carrier_analytic).astype(np.float32) + 1e-10
        ref_i /= env
        ref_q /= env
    else:
        # Free-running 57 kHz (for comparison -- higher jitter expected)
        t = np.arange(len(audio)) / rate
        ref_i = np.cos(2 * np.pi * RDS_SUBCARRIER_HZ * t).astype(np.float32)
        ref_q = -np.sin(2 * np.pi * RDS_SUBCARRIER_HZ * t).astype(np.float32)

    audio_f32 = audio.astype(np.float32)
    bb_i = audio_f32 * ref_i
    bb_q = audio_f32 * ref_q

    return (bb_i + 1j * bb_q).astype(np.complex64)


def lowpass_and_decimate(
    bb: np.ndarray,
    input_rate: float,
    cutoff_hz: float = 2400.0,
    decimation: int = 8,
) -> tuple[np.ndarray, float]:
    """
    Lowpass filter and decimate the complex baseband RDS signal.

    Returns (decimated_signal, output_rate).
    """
    from scipy.signal import firwin, lfilter

    nyq = input_rate / 2.0
    taps = firwin(64, cutoff_hz / nyq, window="hamming").astype(np.float32)

    # Filter real and imag separately
    bb_i_filt = lfilter(taps, 1.0, bb.real)
    bb_q_filt = lfilter(taps, 1.0, bb.imag)

    # Decimate
    bb_dec = (bb_i_filt[::decimation] + 1j * bb_q_filt[::decimation]).astype(np.complex64)
    return bb_dec, input_rate / decimation


def gardner_timing_recovery(
    bb: np.ndarray,
    samples_per_symbol: float,
    loop_bw: float = 0.01,
) -> tuple[list[int], list[float], list[int]]:
    """
    Gardner timing error detector for BPSK timing recovery.

    Returns:
        bit_sample_indices: sample index of each recovered bit boundary
            (in the input bb array coordinates)
        bit_values: soft decision value at each bit boundary
        ted_errors: timing error at each bit (for diagnostics)
    """
    # Loop filter coefficients (proportional + integral)
    # Using a second-order loop with damping
    zeta = 1.0 / np.sqrt(2)
    Kp = 4 * zeta * loop_bw / (1 + 2 * zeta * loop_bw + loop_bw ** 2)
    Ki = 4 * loop_bw ** 2 / (1 + 2 * zeta * loop_bw + loop_bw ** 2)

    mu = 0.0  # fractional sample offset (0..1)
    period = samples_per_symbol
    integrator = 0.0

    bit_indices: list[int] = []
    bit_values: list[float] = []
    ted_errors: list[int] = []

    # State for Gardner TED: needs samples at k, k-T/2, k-T
    prev_sample = 0.0 + 0j
    prev_mid_sample = 0.0 + 0j

    i = float(period)  # start after one symbol period
    nominal_period = period
    while i < len(bb) - 2:
        # Current sample (at the optimal sampling point) - linear interpolation
        idx = int(i)
        frac = i - idx
        if idx < 0 or idx >= len(bb) - 1:
            break
        current = bb[idx] * (1 - frac) + bb[idx + 1] * frac

        # Mid-point sample (T/2 before current)
        mid_i = i - period / 2
        mid_idx = int(mid_i)
        mid_frac = mid_i - mid_idx
        if 0 <= mid_idx < len(bb) - 1:
            mid_sample = bb[mid_idx] * (1 - mid_frac) + bb[mid_idx + 1] * mid_frac
        elif 0 <= mid_idx < len(bb):
            mid_sample = bb[mid_idx]
        else:
            mid_sample = 0.0 + 0j

        # Gardner TED: e = Re{(current - prev_sample) * conj(mid_sample)}
        ted = np.real((current - prev_sample) * np.conj(mid_sample))

        # Loop filter
        integrator += Ki * ted
        # Clamp the period adjustment to prevent runaway
        period_adjust = np.clip(Kp * ted + integrator, -period * 0.1, period * 0.1)

        # Record this bit
        bit_indices.append(idx)
        bit_values.append(float(np.real(current)))
        ted_errors.append(int(ted * 1000))

        # Update state
        prev_sample = current

        # Advance by one symbol period (adjusted)
        i += nominal_period + period_adjust

    return bit_indices, bit_values, ted_errors


def compute_rds_crc_syndrome(bits_26: int) -> int:
    """Compute the RDS CRC-10 syndrome for a 26-bit block."""
    reg = 0
    for i in range(25, -1, -1):
        bit = (bits_26 >> i) & 1
        fb = ((reg >> 9) & 1) ^ bit
        reg = ((reg << 1) & 0x3FF)
        if fb:
            reg ^= _RDS_POLY
    return reg


def check_rds_block_sync(bits: list[int]) -> list[tuple[int, str]]:
    """
    Slide a 26-bit window over the bit stream and check CRC syndromes.

    Returns list of (bit_index, block_type) for each detected block boundary.
    """
    detections = []
    if len(bits) < 26:
        return detections

    # Build integer from first 26 bits
    window = 0
    for i in range(26):
        window = (window << 1) | (1 if bits[i] > 0 else 0)

    syndrome = compute_rds_crc_syndrome(window)
    if syndrome in _RDS_SYNDROMES:
        detections.append((0, _RDS_SYNDROMES[syndrome]))

    for i in range(1, len(bits) - 25):
        # Shift window left, add new bit
        window = ((window << 1) & 0x3FFFFFF) | (1 if bits[i + 25] > 0 else 0)
        syndrome = compute_rds_crc_syndrome(window)
        if syndrome in _RDS_SYNDROMES:
            detections.append((i, _RDS_SYNDROMES[syndrome]))

    return detections


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_iq(iq: np.ndarray, verbose: bool = False, compare: bool = False):
    """Process raw IQ and extract RDS bit transitions."""

    duration = len(iq) / SDR_RATE
    print(f"Input: {len(iq)} samples ({duration:.1f}s at {SDR_RATE/1e6:.1f} MHz)")

    # Step 1: Decimate and FM demodulate (same as existing pipeline)
    print("\nStep 1: Decimation and FM demodulation...")
    dec = Decimator(SYNC_DEC, SDR_RATE, 128_000.0)
    dem = FMDemodulator(SYNC_RATE)

    chunk_size = 65536
    audio_chunks = []
    for i in range(0, len(iq), chunk_size):
        chunk = iq[i:i + chunk_size]
        iq_dec = dec.process(chunk)
        audio = dem.process(iq_dec)
        audio_chunks.append(audio)
    audio_all = np.concatenate(audio_chunks)
    print(f"  Demodulated audio: {len(audio_all)} samples at {SYNC_RATE:.0f} Hz")

    # Step 2: Mix RDS subcarrier to baseband
    print("\nStep 2: Mix 57 kHz RDS subcarrier to baseband...")
    t0 = time.monotonic()
    bb = mix_rds_to_baseband(audio_all, SYNC_RATE)
    print(f"  Baseband RDS: {len(bb)} complex samples ({(time.monotonic()-t0)*1000:.0f} ms)")

    # Step 3: Lowpass filter and decimate
    print("\nStep 3: Lowpass filter (2.4 kHz) and decimate 8x...")
    t0 = time.monotonic()
    bb_dec, dec_rate = lowpass_and_decimate(bb, SYNC_RATE, cutoff_hz=2400.0, decimation=8)
    spb = dec_rate / RDS_BIT_RATE
    print(f"  Decimated: {len(bb_dec)} samples at {dec_rate:.0f} Hz ({spb:.2f} samples/bit)")
    print(f"  ({(time.monotonic()-t0)*1000:.0f} ms)")

    # Step 4: Gardner timing recovery
    print("\nStep 4: BPSK timing recovery (Gardner TED)...")
    t0 = time.monotonic()
    bit_indices, bit_values, ted_errors = gardner_timing_recovery(
        bb_dec, spb, loop_bw=0.005
    )
    elapsed = time.monotonic() - t0
    print(f"  Recovered {len(bit_indices)} bit boundaries ({elapsed*1000:.0f} ms)")
    actual_rate = len(bit_indices) / duration if duration > 0 else 0
    print(f"  Bit rate: {actual_rate:.1f} bps (expected {RDS_BIT_RATE:.1f} bps)")

    if len(bit_indices) < 10:
        print("\nERROR: Too few bits recovered. RDS may not be present on this station.")
        return

    # Step 5: Timing statistics
    print("\nStep 5: Bit transition timing statistics...")
    # Convert bit indices back to 250 kHz sync stream coordinates
    # Each decimated sample = 8 sync samples
    bit_sync_indices = [idx * 8 for idx in bit_indices]
    intervals = np.diff(bit_sync_indices)
    expected_interval = SYNC_RATE / RDS_BIT_RATE  # ~210.5 samples at 250 kHz
    interval_errors = intervals - expected_interval

    print(f"  Expected bit interval: {expected_interval:.1f} sync samples ({RDS_BIT_PERIOD_US:.1f} usec)")
    print(f"  Measured intervals: mean={np.mean(intervals):.1f} std={np.std(intervals):.1f} "
          f"min={np.min(intervals)} max={np.max(intervals)}")
    print(f"  Interval jitter (std): {np.std(interval_errors):.2f} samples "
          f"({np.std(interval_errors) / SYNC_RATE * 1e6:.2f} usec)")

    # Step 6: RDS block sync detection
    print("\nStep 6: CRC syndrome block detection...")
    # Slice bits to hard decisions
    hard_bits = [1 if v > 0 else 0 for v in bit_values]
    blocks = check_rds_block_sync(hard_bits)
    print(f"  Found {len(blocks)} valid CRC syndromes in {len(hard_bits)} bits")

    if blocks:
        # Check for consecutive A-B-C-D patterns
        block_types = [b[1] for b in blocks]
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(blocks)):
            expected_next = {"A": "B", "B": "C", "C": "D", "C'": "D", "D": "A"}
            if blocks[i][0] == blocks[i-1][0] + 26:
                if block_types[i] == expected_next.get(block_types[i-1]):
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            else:
                consecutive = 0

        type_counts = {}
        for _, t in blocks:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Block types: {type_counts}")
        print(f"  Max consecutive valid blocks: {max_consecutive + 1}")

        if max_consecutive >= 3:
            print("  RDS SYNC LOCKED: consistent A-B-C-D pattern detected")
        else:
            print("  RDS sync not fully locked (need 4+ consecutive blocks)")

        # Show first few block detections
        if verbose:
            print("\n  First 20 block detections:")
            for bit_idx, btype in blocks[:20]:
                sync_sample = bit_sync_indices[bit_idx] if bit_idx < len(bit_sync_indices) else -1
                print(f"    bit {bit_idx:5d}  type={btype}  sync_sample={sync_sample}")

    # Step 7: Compare with pilot sync events
    if compare:
        print("\nStep 7: Comparison with pilot sync events...")
        det = FMPilotSyncDetector(SYNC_RATE, sync_period_ms=7.0, calibration_window=100)
        pilot_events = []
        for i in range(0, len(audio_all), chunk_size // SYNC_DEC):
            chunk = audio_all[i:i + chunk_size // SYNC_DEC]
            events = det.process(chunk, start_sample=i)
            pilot_events.extend(events)

        print(f"  Pilot sync events: {len(pilot_events)} ({len(pilot_events)/duration:.0f}/sec)")
        print(f"  RDS bit transitions: {len(bit_sync_indices)} ({len(bit_sync_indices)/duration:.0f}/sec)")
        print(f"  Ratio: {len(bit_sync_indices)/max(len(pilot_events),1):.1f}x more RDS events")

        # For each pilot event, find the nearest RDS bit transition
        if pilot_events and bit_sync_indices:
            pilot_samples = np.array([e.sample_index for e in pilot_events])
            rds_samples = np.array(bit_sync_indices)
            offsets = []
            for ps in pilot_samples:
                idx = np.searchsorted(rds_samples, ps)
                candidates = []
                if idx > 0:
                    candidates.append(rds_samples[idx - 1])
                if idx < len(rds_samples):
                    candidates.append(rds_samples[idx])
                if candidates:
                    nearest = min(candidates, key=lambda x: abs(x - ps))
                    offsets.append((ps - nearest) / SYNC_RATE * 1e6)  # usec

            offsets = np.array(offsets)
            print(f"\n  Pilot-to-nearest-RDS offset:")
            print(f"    mean={np.mean(offsets):+.1f} usec  std={np.std(offsets):.1f} usec")
            print(f"    min={np.min(offsets):+.1f}  max={np.max(offsets):+.1f} usec")
            print(f"    (expected: uniformly distributed in +/-{RDS_BIT_PERIOD_US/2:.0f} usec)")

    print("\n--- SUMMARY ---")
    print(f"RDS bit rate:        {actual_rate:.1f} bps (expected {RDS_BIT_RATE:.1f})")
    print(f"Timing jitter:       {np.std(interval_errors)/SYNC_RATE*1e6:.2f} usec")
    print(f"CRC blocks found:    {len(blocks)}")
    if blocks:
        print(f"Consecutive valid:   {max_consecutive + 1}")
    print(f"Disambiguation:      RDS bit period = {RDS_BIT_PERIOD_US:.0f} usec "
          f"vs max TDOA ~333 usec -> {RDS_BIT_PERIOD_US/333:.1f}x margin")


# ---------------------------------------------------------------------------
# Working RDS decoder (PySDR approach)
# Reference: https://pysdr.org/content/rds.html
# ---------------------------------------------------------------------------

_RDS_SYNDROME_PYSDR = [383, 14, 303, 663, 748]
_RDS_OFFSET_POS = [0, 1, 2, 3, 2]
_RDS_OFFSET_WORD = [252, 408, 360, 436, 848]


def _calc_syndrome_pysdr(x_val: int, mlen: int) -> int:
    """CRC-10 syndrome per PySDR / NRSC-4-B (polynomial 0x5B9)."""
    reg = 0
    plen = 10
    for ii in range(mlen, 0, -1):
        reg = (reg << 1) | ((x_val >> (ii - 1)) & 0x01)
        if reg & (1 << plen):
            reg ^= 0x5B9
    for ii in range(plen, 0, -1):
        reg = reg << 1
        if reg & (1 << plen):
            reg ^= 0x5B9
    return reg & ((1 << plen) - 1)


def decode_rds_pysdr(iq: np.ndarray, verbose: bool = False) -> dict:
    """
    Full RDS decode using the PySDR signal chain.

    Input: raw complex64 IQ at 2 MHz (or 250 kHz if pre-decimated).
    Returns dict with groups, PI code, bit transition timestamps, etc.
    """
    from scipy.signal import resample_poly, firwin

    # Decimate to 250 kHz if needed
    if len(iq) > 1_000_000:
        dec = Decimator(8, 2_000_000.0, 128_000.0)
        chunks = []
        for i in range(0, len(iq), 65536):
            chunks.append(dec.process(iq[i:i + 65536]))
        iq_dec = np.concatenate(chunks)
        sample_rate = 250_000.0
    else:
        iq_dec = iq
        sample_rate = 250_000.0

    # FM demod (phase change, radians)
    x = 0.5 * np.angle(iq_dec[:-1] * np.conj(iq_dec[1:]))

    # Freq shift -57 kHz
    N = len(x)
    t = np.arange(N) / sample_rate
    x = x * np.exp(2j * np.pi * (-57e3) * t)

    # LPF
    taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
    x = np.convolve(x, taps, 'valid')

    # Decimate by 10
    x = x[::10]
    sample_rate = 25e3

    # Resample to 19 kHz (16 sps)
    x = resample_poly(x, 19, 25)
    sample_rate = 19e3

    # M&M timing recovery (32x pre-interpolation)
    samples_interpolated = resample_poly(x, 32, 1)
    sps = 16
    mu = 0.01
    out = np.zeros(len(x) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(x) + 10, dtype=np.complex64)
    i_in = 0
    i_out = 2
    # Track M&M sample indices for timing extraction
    mm_indices = [0, 0]  # pad for i_out starting at 2
    while i_out < len(x) and i_in + 32 < len(x):
        out[i_out] = samples_interpolated[i_in * 32 + int(mu * 32)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]) > 0)
        x_val = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
        y_val = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
        mm_val = np.real(y_val - x_val)
        mu += sps + 0.01 * mm_val
        # Record the input sample index for this symbol
        mm_indices.append(i_in + mu / sps)
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1
    x = out[2:i_out]
    mm_indices = mm_indices[2:i_out]

    # Costas loop
    N_s = len(x)
    phase = 0.0
    freq = 0.0
    out2 = np.zeros(N_s, dtype=np.complex64)
    for i in range(N_s):
        out2[i] = x[i] * np.exp(-1j * phase)
        error = np.real(out2[i]) * np.imag(out2[i])
        freq += 0.02 * error
        phase += freq + 8.0 * error
        while phase >= 2 * np.pi:
            phase -= 2 * np.pi
        while phase < 0:
            phase += 2 * np.pi
    x = out2

    # Bit decisions + differential decode
    bits = (np.real(x) > 0).astype(int)
    bits = (bits[1:] - bits[0:-1]) % 2
    bits = bits.astype(np.uint8)

    # RDS block sync + decode
    synced = False
    presync = False
    wrong_blocks_counter = 0
    blocks_counter = 0
    group_good_blocks_counter = 0
    reg = np.uint32(0)
    lastseen_offset_counter = 0
    lastseen_offset = 0
    groups = []
    sync_bit_index = None

    for i in range(len(bits)):
        reg = np.bitwise_or(np.left_shift(reg, 1), bits[i])
        if not synced:
            reg_syndrome = _calc_syndrome_pysdr(reg, 26)
            for j in range(5):
                if reg_syndrome == _RDS_SYNDROME_PYSDR[j]:
                    if not presync:
                        lastseen_offset = j
                        lastseen_offset_counter = i
                        presync = True
                    else:
                        if _RDS_OFFSET_POS[lastseen_offset] >= _RDS_OFFSET_POS[j]:
                            block_distance = _RDS_OFFSET_POS[j] + 4 - _RDS_OFFSET_POS[lastseen_offset]
                        else:
                            block_distance = _RDS_OFFSET_POS[j] - _RDS_OFFSET_POS[lastseen_offset]
                        if (block_distance * 26) != (i - lastseen_offset_counter):
                            presync = False
                        else:
                            if verbose:
                                print(f"  RDS sync acquired at bit {i}")
                            sync_bit_index = i
                            wrong_blocks_counter = 0
                            blocks_counter = 0
                            block_bit_counter = 0
                            block_number = (j + 1) % 4
                            group_assembly_started = False
                            synced = True
                break
        else:
            if block_bit_counter < 25:
                block_bit_counter += 1
            else:
                good_block = False
                dataword = (reg >> 10) & 0xFFFF
                block_calculated_crc = _calc_syndrome_pysdr(dataword, 16)
                checkword = reg & 0x3FF
                if block_number == 2:
                    block_received_crc = checkword ^ _RDS_OFFSET_WORD[block_number]
                    if block_received_crc == block_calculated_crc:
                        good_block = True
                    else:
                        block_received_crc = checkword ^ _RDS_OFFSET_WORD[4]
                        if block_received_crc == block_calculated_crc:
                            good_block = True
                        else:
                            wrong_blocks_counter += 1
                else:
                    block_received_crc = checkword ^ _RDS_OFFSET_WORD[block_number]
                    if block_received_crc == block_calculated_crc:
                        good_block = True
                    else:
                        wrong_blocks_counter += 1

                if block_number == 0 and good_block:
                    group_assembly_started = True
                    group_good_blocks_counter = 1
                    group = bytearray(8)
                if group_assembly_started:
                    if not good_block:
                        group_assembly_started = False
                    else:
                        group[block_number * 2] = (dataword >> 8) & 255
                        group[block_number * 2 + 1] = dataword & 255
                        group_good_blocks_counter += 1
                    if group_good_blocks_counter == 5:
                        groups.append(bytes(group))
                block_bit_counter = 0
                block_number = (block_number + 1) % 4
                blocks_counter += 1
                if blocks_counter == 50:
                    if wrong_blocks_counter > 35:
                        if verbose:
                            print(f"  RDS sync lost at bit {i}")
                        synced = False
                        presync = False
                    blocks_counter = 0
                    wrong_blocks_counter = 0

    # Extract PI code
    pi_code = None
    if groups:
        pi_code = (groups[0][0] << 8) | groups[0][1]

    # Map M&M symbol positions back to 250 kHz sync stream coordinates.
    # Chain: 250k ->/10-> 25k ->*19/25-> 19k -> M&M -> symbols
    # Reverse: pos_250k = pos_19k * (25/19) * 10 + lpf_group_delay
    # lpf_group_delay = (101-1)//2 = 50 samples at 250 kHz
    _LPF_DELAY = 50  # (numtaps-1)//2 for the 101-tap LPF
    symbol_positions_250k = np.array(mm_indices, dtype=np.float64) * (25.0 / 19.0) * 10.0 + _LPF_DELAY

    # Differential decode shifts by 1 symbol, so bit positions start at index 1
    bit_positions_250k = symbol_positions_250k[1:] if len(symbol_positions_250k) > 1 else np.array([])

    return {
        "groups": groups,
        "pi_code": pi_code,
        "n_bits": len(bits),
        "n_symbols": N_s,
        "sync_bit_index": sync_bit_index,
        "bit_positions_250k": bit_positions_250k,
        "synced": synced,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 prototype: RDS bit transition sync detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", metavar="PATH",
                        help="Path to .npy IQ file (complex64, 2 MHz)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Compare RDS sync with pilot sync events")
    parser.add_argument("--decode", action="store_true",
                        help="Run full RDS decode (PySDR chain)")
    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required (path to .npy IQ capture)")

    iq = np.load(args.input)
    if iq.dtype != np.complex64:
        iq = iq.astype(np.complex64)

    if args.decode:
        print("=" * 60)
        print("Full RDS Decode (PySDR chain)")
        print("=" * 60)
        result = decode_rds_pysdr(iq, verbose=args.verbose)
        duration = len(iq) / SDR_RATE
        print(f"\nGroups decoded: {len(result['groups'])}")
        if result['pi_code']:
            print(f"PI code: 0x{result['pi_code']:04X}")
        print(f"Bits: {result['n_bits']}")
        print(f"Still synced: {result['synced']}")

        bp = result['bit_positions_250k']
        if len(bp) > 2:
            intervals = np.diff(bp)
            expected = SYNC_RATE / RDS_BIT_RATE
            jitter_samples = np.std(intervals - expected)
            print(f"\nBit transition timing (250 kHz sync stream):")
            print(f"  Transitions: {len(bp)} ({len(bp)/duration:.0f}/sec)")
            print(f"  Interval: {np.mean(intervals):.2f} +/- {jitter_samples:.3f} samples")
            print(f"  Jitter: {jitter_samples/SYNC_RATE*1e6:.2f} usec")
            print(f"  (pilot sync: ~143/sec; RDS: {len(bp)/duration:.0f}/sec = "
                  f"{len(bp)/duration/143:.1f}x more)")

        if result['groups']:
            print("\nDecoded groups:")
            for g in result['groups']:
                pi = (g[0] << 8) | g[1]
                gt = (g[2] >> 4) & 0xF
                print(f"  PI=0x{pi:04X} type={gt}")
    else:
        process_iq(iq, verbose=args.verbose, compare=args.compare)


if __name__ == "__main__":
    main()
