#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Quick SDR connectivity and IQ sanity check.
Run outside the virtualenv (needs system SoapySDR bindings):
    python3 scripts/check_sdr.py
"""

import sys
import numpy as np
import SoapySDR

FREQ_HZ   = 99.9e6    # KISW FM - strong signal, confirms reception
RATE_SPS  = 2.048e6
GAIN_DB   = 0
N_SAMPLES = 131_072   # ~64 ms

def main() -> int:
    # --- enumerate ---
    results = SoapySDR.Device.enumerate()
    print(f"Detected devices: {len(results)}")
    for r in results:
        d = dict(r)
        print(f"  driver={d.get('driver','?')}  label={d.get('label','?')}")

    if not results:
        print("ERROR: No SDR found - check USB connection and driver", file=sys.stderr)
        return 1

    # --- open ---
    sdr = SoapySDR.Device(results[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, RATE_SPS)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX,  0, FREQ_HZ)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX,       0, GAIN_DB)

    actual_rate = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
    actual_freq = sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
    actual_gain = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
    print(f"\nConfigured:")
    print(f"  Rate  {actual_rate/1e6:.3f} MSps")
    print(f"  Freq  {actual_freq/1e6:.3f} MHz")
    print(f"  Gain  {actual_gain:.1f} dB")

    # --- stream ---
    rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    buf = np.zeros(N_SAMPLES, dtype=np.complex64)
    sr  = sdr.readStream(rx, [buf], N_SAMPLES, timeoutUs=2_000_000)

    sdr.deactivateStream(rx)
    sdr.closeStream(rx)

    # --- report ---
    if sr.ret < 0:
        print(f"ERROR: readStream returned {sr.ret}", file=sys.stderr)
        return 1

    samples = buf[:sr.ret]
    power_db = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12)
    peak_db  = 10 * np.log10(np.max(np.abs(samples) ** 2)  + 1e-12)
    dc_i     = np.mean(samples.real)
    dc_q     = np.mean(samples.imag)

    print(f"\nStream result:")
    print(f"  Samples received : {sr.ret:,}")
    print(f"  RMS power        : {power_db:.1f} dBFS")
    print(f"  Peak power       : {peak_db:.1f} dBFS")
    print(f"  DC offset I/Q    : {dc_i:.4f} / {dc_q:.4f}")
    print(f"  Any non-zero     : {np.any(samples != 0)}")

    if power_db < -40:
        print("\nWARNING: Very low power - check antenna connection or increase gain")
    elif power_db > -3:
        print("\nWARNING: Very high power - may be clipping, reduce gain")
    else:
        print(f"\nOK: Signal level looks reasonable for FM reception")

    return 0


if __name__ == "__main__":
    sys.exit(main())
