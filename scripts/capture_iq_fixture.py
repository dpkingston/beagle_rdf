#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Capture IQ samples from a real SDR to a .npy fixture file.

Saved fixtures can be used as test inputs for the signal processing pipeline
without requiring hardware.  Load them with MockReceiver.from_file().

Usage examples
--------------
# Capture 5 s of FM broadcast (for sync_detector tests):
python3 scripts/capture_iq_fixture.py \\
    --freq 99.9e6 --rate 2.048e6 --gain 0 --duration 5 \\
    --output tests/fixtures/iq_fm_kisw_99.9.npy

# Capture 3 s of LMR channel (for carrier_detect tests):
python3 scripts/capture_iq_fixture.py \\
    --freq 462.5625e6 --rate 2.048e6 --gain 30 --duration 3 \\
    --output tests/fixtures/iq_lmr_462.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SoapySDR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture SDR IQ to .npy fixture")
    p.add_argument("--freq",     type=float, required=True, metavar="HZ",
                   help="Center frequency in Hz (e.g. 99.9e6)")
    p.add_argument("--rate",     type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate in samples/sec (default 2.048e6)")
    p.add_argument("--gain",     type=float, default=0.0, metavar="DB",
                   help="Receiver gain in dB (default 0)")
    p.add_argument("--duration", type=float, default=5.0, metavar="SEC",
                   help="Capture duration in seconds (default 5)")
    p.add_argument("--device",   type=str, default="", metavar="ARGS",
                   help="SoapySDR device args (default: first available)")
    p.add_argument("--output",   type=str, required=True, metavar="PATH",
                   help="Output .npy file path")
    p.add_argument("--buffer-size", type=int, default=131_072, metavar="N",
                   help="Read buffer size in samples (default 131072)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples_total = int(args.rate * args.duration)
    buf_size = args.buffer_size

    print(f"Capturing {args.duration:.1f} s at {args.freq/1e6:.3f} MHz "
          f"({n_samples_total:,} samples) -> {out_path}")

    # Open device
    devs = SoapySDR.Device.enumerate(args.device)
    if not devs:
        print("ERROR: No SDR device found", file=sys.stderr)
        return 1

    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, args.rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.freq)
    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, args.gain)

    actual_rate = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
    actual_freq = sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
    actual_gain = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
    print(f"  Rate  {actual_rate/1e6:.3f} MSps")
    print(f"  Freq  {actual_freq/1e6:.3f} MHz")
    print(f"  Gain  {actual_gain:.1f} dB")

    rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    collected: list[np.ndarray] = []
    samples_collected = 0
    buf = np.zeros(buf_size, dtype=np.complex64)

    try:
        while samples_collected < n_samples_total:
            n_want = min(buf_size, n_samples_total - samples_collected)
            sr = sdr.readStream(rx, [buf], n_want, timeoutUs=2_000_000)
            if sr.ret < 0:
                print(f"ERROR: readStream returned {sr.ret}", file=sys.stderr)
                break
            collected.append(buf[:sr.ret].copy())
            samples_collected += sr.ret
            pct = 100 * samples_collected / n_samples_total
            print(f"\r  {samples_collected:,} / {n_samples_total:,} ({pct:.0f}%)",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nCapture interrupted")
    finally:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)

    print()  # newline after progress

    if not collected:
        print("ERROR: No samples collected", file=sys.stderr)
        return 1

    iq = np.concatenate(collected)
    np.save(str(out_path), iq)

    power_db = float(10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-30))
    print(f"Saved {len(iq):,} samples  RMS power {power_db:.1f} dBFS  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
