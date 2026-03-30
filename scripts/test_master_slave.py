#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Test RSPduo Master/Slave mode for truly independent per-tuner frequency control.

Master = Tuner 1 at 99.9 MHz FM (high attenuation - sync/drain channel)
Slave  = Tuner 2 at 462.5625 MHz   (max sensitivity - target channel)

Run from the Beagle virtualenv:
  python3 scripts/test_master_slave.py

Key the HT a few times during the 20-second measurement window.
"""

import sys
import time

import numpy as np


SERIAL       = "240504F534"
MASTER_FREQ  = 99.9e6       # FM sync channel (high attenuation)
TARGET_FREQ  = 462.5625e6    # LMR target channel
RATE         = 2e6
MEASURE_BW   = 50_000.0     # FFT window around DC
WIN_SAMPS    = int(RATE * 0.5)


def main() -> int:
    try:
        import SoapySDR
    except ImportError:
        print("ERROR: SoapySDR not available", file=sys.stderr)
        return 1

    RX   = SoapySDR.SOAPY_SDR_RX
    CF32 = SoapySDR.SOAPY_SDR_CF32

    # ------------------------------------------------------------------ #
    # Open Master (Tuner 1 = FM sync)                                    #
    # ------------------------------------------------------------------ #
    print(f"Opening Master (Tuner 1) at {MASTER_FREQ/1e6:.1f} MHz ...")
    master = SoapySDR.Device({"driver": "sdrplay", "mode": "MA", "serial": SERIAL})
    master.setSampleRate(RX, 0, RATE)
    master.setFrequency(RX, 0, MASTER_FREQ)
    master.setGainMode(RX, 0, False)
    master.setGain(RX, 0, "IFGR", 59)   # max IF attenuation
    master.setGain(RX, 0, "RFGR", 9)    # max LNA attenuation
    print(f"  Master: freq={master.getFrequency(RX,0)/1e6:.3f} MHz  "
          f"IFGR={master.getGain(RX,0,'IFGR'):.0f}  "
          f"RFGR={master.getGain(RX,0,'RFGR'):.0f}")

    # ------------------------------------------------------------------ #
    # Start Master stream BEFORE opening Slave                           #
    # The sdrplay service requires the Master to be actively streaming   #
    # before SelectDevice() for the Slave will succeed.                  #
    # ------------------------------------------------------------------ #
    ms = master.setupStream(RX, CF32, [0])
    master.activateStream(ms)
    print("  Master stream active - waiting for MasterInitialised event ...")
    _drain = np.zeros(65536, dtype=np.complex64)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        master.readStream(ms, [_drain], len(_drain), timeoutUs=500_000)
        if master.readSetting("master_initialised") == "true":
            print(f"  MasterInitialised event received after {time.monotonic()-t0:.2f}s")
            break
    else:
        print("  WARNING: MasterInitialised event not received within 5s - proceeding anyway")

    # ------------------------------------------------------------------ #
    # Find Slave device (appears after Master is opened)                 #
    # ------------------------------------------------------------------ #
    devs = list(SoapySDR.Device.enumerate("driver=sdrplay"))
    modes = [dict(d).get("mode") for d in devs]
    print(f"Devices after master open: {modes}")

    slave_kwargs = next((d for d in devs if dict(d).get("mode") == "SL"), None)
    if slave_kwargs is None:
        print("ERROR: No Slave device found after opening Master.", file=sys.stderr)
        print(f"  Available modes: {modes}", file=sys.stderr)
        master.deactivateStream(ms); master.closeStream(ms)
        return 1
    print(f"Slave device: {dict(slave_kwargs)}")

    # ------------------------------------------------------------------ #
    # Open Slave (Tuner 2 = LMR target) - retry until success or timeout#
    # ------------------------------------------------------------------ #
    print(f"Opening Slave (Tuner 2) at {TARGET_FREQ/1e6:.3f} MHz ...")
    slave = None
    for attempt in range(20):
        try:
            slave = SoapySDR.Device(slave_kwargs)
            print(f"  Slave opened on attempt {attempt+1}")
            break
        except RuntimeError as e:
            print(f"  Attempt {attempt+1}: {e} - retrying in 0.5s")
            master.readStream(ms, [_drain], len(_drain), timeoutUs=500_000)
            time.sleep(0.5)
    if slave is None:
        print("ERROR: Could not open Slave after 20 attempts", file=sys.stderr)
        master.deactivateStream(ms); master.closeStream(ms)
        return 1
    slave.setSampleRate(RX, 0, RATE)
    slave.setFrequency(RX, 0, TARGET_FREQ)
    slave.setGainMode(RX, 0, False)
    slave.setGain(RX, 0, "IFGR", 49)    # gain=30 equivalent
    slave.setGain(RX, 0, "RFGR", 0)     # max LNA gain
    print(f"  Slave:  freq={slave.getFrequency(RX,0)/1e6:.3f} MHz  "
          f"IFGR={slave.getGain(RX,0,'IFGR'):.0f}  "
          f"RFGR={slave.getGain(RX,0,'RFGR'):.0f}")

    # ------------------------------------------------------------------ #
    # Slave stream (master stream already active)                        #
    # ------------------------------------------------------------------ #
    ss = slave.setupStream(RX, CF32, [0])
    slave.activateStream(ss)

    drain_chunk = np.zeros(65536, dtype=np.complex64)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        master.readStream(ms, [drain_chunk], len(drain_chunk), timeoutUs=1_000_000)
        slave.readStream(ss, [drain_chunk], len(drain_chunk), timeoutUs=1_000_000)

    # ------------------------------------------------------------------ #
    # Narrowband power measurement loop (50 kHz FFT window at DC)        #
    # ------------------------------------------------------------------ #
    print()
    print(f"Measuring Tuner 2 at {TARGET_FREQ/1e6:.3f} MHz - key HT now - 20 seconds")
    print(f"  {'Time':>6}  {'dBFS':>7}")
    print(f"  {'-'*20}")

    buf_s  = np.zeros(WIN_SAMPS, dtype=np.complex64)
    buf_m  = np.zeros(65536,     dtype=np.complex64)
    t_start = time.monotonic()

    while time.monotonic() - t_start < 20:
        collected = 0
        while collected < WIN_SAMPS:
            want = min(65536, WIN_SAMPS - collected)
            master.readStream(ms, [buf_m], want, timeoutUs=2_000_000)
            sr = slave.readStream(ss, [buf_s[collected:]], want, timeoutUs=2_000_000)
            if sr.ret > 0:
                collected += sr.ret
            elif sr.ret < 0:
                break

        N = collected
        fft   = np.fft.fft(buf_s[:N])
        freqs = np.fft.fftfreq(N, 1.0 / RATE)
        mask  = np.abs(freqs) <= MEASURE_BW / 2.0
        pwr   = float(np.sum(np.abs(fft[mask]) ** 2)) / (N ** 2)
        db    = 10.0 * np.log10(pwr + 1e-30)
        elapsed = time.monotonic() - t_start
        print(f"  {elapsed:5.1f}s  {db:7.1f} dBFS")

    # ------------------------------------------------------------------ #
    # Cleanup                                                             #
    # ------------------------------------------------------------------ #
    slave.deactivateStream(ss);  slave.closeStream(ss)
    master.deactivateStream(ms); master.closeStream(ms)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
