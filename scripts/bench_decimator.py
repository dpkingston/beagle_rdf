#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Micro-benchmark for decimator alternatives.

Tests different approaches to FIR filtering + decimation to find the
fastest option for the Beagle pipeline.

Candidates:
1. scipy.signal.lfilter (current - split real/imag float32)
2. numpy.convolve (no state, but simpler)
3. scipy.signal.sosfilt (SOS form - potentially faster for long filters)
4. scipy.signal.upfirdn (was previous approach, replaced in commit 23bf09e)
5. Direct FFT overlap-save (batch convolution)
6. scipy.signal.fftconvolve
7. Polyphase decimation (downsample-aware - filters only needed outputs)

Also tests the impact of:
- Tap count reduction (127 -> 63 -> 31)
- Buffer size variation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, upfirdn, sosfilt, sosfilt_zi
from scipy.signal import dlti, tf2sos

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def bench(func, n_iters: int, label: str, warmup: int = 3) -> float:
    """Run func() n_iters times, return median usec/call."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    print(f"  {label:<45s}  median={1e6*median:8.0f} usec  mean={1e6*mean:8.0f} usec")
    return median


def main():
    sdr_rate = 2_048_000.0
    buffer_samples = 65536
    n_iters = 100

    # Generate test data
    np.random.seed(42)
    iq = (np.random.randn(buffer_samples) + 1j * np.random.randn(buffer_samples)).astype(np.complex64)

    for decimation, cutoff_hz, label in [
        (8, 128_000.0, "SYNC (8x, 127 taps)"),
        (32, 25_000.0, "TARGET (32x, 127 taps)"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {label}:  {buffer_samples} samples -> {buffer_samples // decimation}")
        print(f"{'='*70}")

        for num_taps in [127, 63, 31]:
            nyq = sdr_rate / 2.0
            taps = firwin(num_taps, cutoff_hz / nyq, window="hamming").astype(np.float32)
            n_out = buffer_samples // decimation
            d = decimation
            end = n_out * d

            print(f"\n  --- {num_taps} taps ---")

            # 1. Current: lfilter split real/imag + stride
            zi_shape = lfilter_zi(taps, 1.0).astype(np.float32)
            zi_re = zi_shape * 0
            zi_im = zi_shape * 0

            def run_lfilter_split():
                nonlocal zi_re, zi_im
                re_f, zi_re = lfilter(taps, 1.0, iq.real, zi=zi_re)
                im_f, zi_im = lfilter(taps, 1.0, iq.imag, zi=zi_im)
                return (re_f[:end:d] + 1j * im_f[:end:d]).astype(np.complex64)

            bench(run_lfilter_split, n_iters, f"lfilter split re/im (current)")

            # 1b. lfilter split - avoid intermediate complex128
            def run_lfilter_split_direct():
                nonlocal zi_re, zi_im
                re_f, zi_re = lfilter(taps, 1.0, iq.real, zi=zi_re)
                im_f, zi_im = lfilter(taps, 1.0, iq.imag, zi=zi_im)
                out = np.empty(n_out, dtype=np.complex64)
                out.real = re_f[:end:d]
                out.imag = im_f[:end:d]
                return out

            bench(run_lfilter_split_direct, n_iters, f"lfilter split - direct complex64")

            # 2. upfirdn (previous approach)
            def run_upfirdn():
                nonlocal zi_re, zi_im
                # upfirdn doesn't maintain state; we just measure raw speed
                re_d = upfirdn(taps, iq.real, down=d)[:n_out]
                im_d = upfirdn(taps, iq.imag, down=d)[:n_out]
                out = np.empty(n_out, dtype=np.complex64)
                out.real = re_d
                out.imag = im_d
                return out

            bench(run_upfirdn, n_iters, f"upfirdn split re/im")

            # 3. SOS form
            sos = tf2sos(taps, [1.0])
            sos_zi_shape = sosfilt_zi(sos)
            sos_zi_re = sos_zi_shape * 0
            sos_zi_im = sos_zi_shape * 0

            def run_sosfilt():
                nonlocal sos_zi_re, sos_zi_im
                re_f, sos_zi_re = sosfilt(sos, iq.real, zi=sos_zi_re)
                im_f, sos_zi_im = sosfilt(sos, iq.imag, zi=sos_zi_im)
                out = np.empty(n_out, dtype=np.complex64)
                out.real = re_f[:end:d]
                out.imag = im_f[:end:d]
                return out

            bench(run_sosfilt, n_iters, f"sosfilt (SOS) split re/im")

            # 4. FFT overlap-save (no state, but shows FFT potential)
            # Use FFT size = next power of 2 >= buffer + taps
            fft_n = 1
            while fft_n < buffer_samples + num_taps:
                fft_n *= 2
            taps_fft_re = np.fft.rfft(taps, n=fft_n)

            def run_fft_ola():
                re_fft = np.fft.rfft(iq.real, n=fft_n)
                im_fft = np.fft.rfft(iq.imag, n=fft_n)
                re_f = np.fft.irfft(re_fft * taps_fft_re, n=fft_n)[:buffer_samples]
                im_f = np.fft.irfft(im_fft * taps_fft_re, n=fft_n)[:buffer_samples]
                out = np.empty(n_out, dtype=np.complex64)
                out.real = re_f[:end:d].astype(np.float32)
                out.imag = im_f[:end:d].astype(np.float32)
                return out

            bench(run_fft_ola, n_iters, f"FFT overlap-save (no state)")

            # 5. Polyphase: only compute output samples (skip d-1 of every d)
            # This is what a proper decimating filter does - M times less work.
            # Build polyphase branches from the FIR taps.
            poly_len = (num_taps + d - 1) // d
            poly_taps = np.zeros((d, poly_len), dtype=np.float32)
            for k in range(num_taps):
                branch = k % d
                coeff_idx = k // d
                poly_taps[branch, coeff_idx] = taps[k]

            def run_polyphase():
                # Reshape input into d phases
                usable = n_out * d
                x_re = iq.real[:usable].reshape(n_out, d).T  # shape (d, n_out)
                x_im = iq.imag[:usable].reshape(n_out, d).T

                # Each branch: convolve with its polyphase taps, then sum branches
                out_re = np.zeros(n_out, dtype=np.float32)
                out_im = np.zeros(n_out, dtype=np.float32)
                for b in range(d):
                    out_re += np.convolve(x_re[b], poly_taps[b], mode='full')[:n_out]
                    out_im += np.convolve(x_im[b], poly_taps[b], mode='full')[:n_out]

                out = np.empty(n_out, dtype=np.complex64)
                out.real = out_re
                out.imag = out_im
                return out

            bench(run_polyphase, n_iters, f"polyphase (d branches x convolve)")

    # --- Sync BPF tap reduction ---
    print(f"\n\n{'='*70}")
    print(f"  Sync BPF (19 kHz pilot isolation): tap count comparison")
    print(f"{'='*70}")

    sync_rate = sdr_rate / 8  # 256 kHz
    pilot_freq = 19_000.0
    pilot_bw = 100.0
    nyq = sync_rate / 2.0
    lo = (pilot_freq - pilot_bw) / nyq
    hi = (pilot_freq + pilot_bw) / nyq

    # Simulate one sync window of audio
    sync_window = int(sync_rate * 7.0 / 1000.0)  # 7ms = 1792 samples
    audio = np.random.randn(sync_window).astype(np.float32)

    for ntaps in [255, 127, 63]:
        bpf = firwin(ntaps, [lo, hi], pass_zero=False, window="hamming").astype(np.float32)
        bpf_zi = np.zeros(ntaps - 1, dtype=np.float32)

        def run_bpf(taps=bpf, zi=bpf_zi):
            out, _ = lfilter(taps, 1.0, audio, zi=zi)
            return out

        bench(run_bpf, 500, f"BPF {ntaps} taps x {sync_window} samples")

    print()


if __name__ == "__main__":
    main()
