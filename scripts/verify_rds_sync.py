#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Live RDS sync detection verification.

Runs the RDSSyncDetector against a live SDR and reports event rate, pilot
correlation peak, crystal correction, and bit-interval jitter.

Healthy RDS sync indicators:
  - Event rate ~1187 events/sec (after warmup)
  - Pilot correlation peak > 0.3
  - Crystal correction stable, < +/-50 ppm
  - Bit interval jitter < 1 sample (sub-microsecond)

Primary usage (recommended) - uses the actual node receiver:
--------------------------------------------------------------------
python3 scripts/verify_rds_sync.py --config config/node.yaml --duration 30

Simple usage - bare SoapySDR device:
--------------------------------------------------------------------
python3 scripts/verify_rds_sync.py --device "driver=sdrplay" --freq 94.9e6 --gain auto --duration 30
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics as _stats
import sys
import time
from pathlib import Path

import numpy as np

# Add project src to path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.rds_sync_detector import (
    RDSSyncDetector,
    RDS_BIT_RATE,
)

# These match the pipeline defaults in pipeline.py / PipelineConfig
SYNC_DEC = 8
CUTOFF_HZ = 128_000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify RDS bit-transition sync detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", metavar="PATH",
                     help="Node YAML config file (recommended; uses the real receiver)")
    src.add_argument("--device", metavar="ARGS", default=None,
                     help="Simple SoapySDR device args (requires --freq)")

    p.add_argument("--freq", type=float, default=94.9e6, metavar="HZ",
                   help="FM station frequency (default 94.9e6 = KUOW); --device mode only")
    p.add_argument("--gain", default="0", metavar="DB",
                   help="Receiver gain in dB, or 'auto' (default 0); --device mode only")
    p.add_argument("--rate", type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate (default 2.048e6); --device mode only")
    p.add_argument("--duration", type=float, default=30.0, metavar="SEC",
                   help="Run for this many seconds then exit (default 30)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sync processing helpers (shared by both paths)
# ---------------------------------------------------------------------------

def _make_pipeline(sdr_rate: float) -> tuple:
    """Return (decimator, demodulator, detector, sync_rate)."""
    sync_rate = sdr_rate / SYNC_DEC
    dec = Decimator(SYNC_DEC, sdr_rate, CUTOFF_HZ)
    dem = FMDemodulator(sync_rate)
    det = RDSSyncDetector(sync_rate)
    return dec, dem, det, sync_rate


def _print_header(freq_hz: float, gain, sdr_rate: float, sync_rate: float,
                  duration: float, mode: str = "") -> None:
    mode_str = f"  [{mode}]" if mode else ""
    gain_str = str(gain) if isinstance(gain, str) else f"{gain:.0f} dB"
    print(f"Tuned to {freq_hz/1e6:.1f} MHz  gain={gain_str}  "
          f"rate={sdr_rate/1e6:.3f} MSps{mode_str}")
    print(f"Pipeline: {SYNC_DEC}x decimation -> {sync_rate/1e3:.0f} kHz "
          f"-> FM demod -> RDS bit-transition detector")
    print(f"Expected event rate: {RDS_BIT_RATE:.0f}/s after warmup (~50 ms)")
    print(f"Running for {duration:.0f} s  (Ctrl-C to stop early)\n")
    print(f"{'Time':>6}  {'Events':>8}  {'Rate/s':>7}  "
          f"{'PilotCor':>9}  {'Crystal':>10}  {'Power':>8}")
    print("-" * 65)


def _report_loop(dec, dem, det, buf: np.ndarray, n: int,
                 dec_start: int, state: dict) -> int:
    """Process one IQ buffer through the RDS sync pipeline; update state.

    Returns the new dec_start (for next call).
    """
    iq = buf[:n]

    # Raw IQ power in dBFS for sanity check.
    power_linear = float(np.mean(np.abs(iq) ** 2))
    state["last_power_dbfs"] = 10.0 * math.log10(max(power_linear, 1e-12))

    iq_dec = dec.process(iq)
    audio = dem.process(iq_dec)
    events = det.process(audio, start_sample=dec_start)

    for e in events:
        state["last_corr"] = e.corr_peak
        state["last_correction"] = e.sample_rate_correction
        state["last_phase"] = e.pilot_phase_rad
        # Track bit intervals: difference between consecutive sample_index values.
        # In healthy RDS, this should be ~RATE/1187.5 (~215.6 at 256 kHz)
        # with very low jitter (M&M timing recovery is sub-sample accurate).
        if state["last_sample_index"] is not None:
            interval = e.sample_index - state["last_sample_index"]
            state["bit_intervals"].append(interval)
        state["last_sample_index"] = e.sample_index

    state["window_events"] += len(events)
    state["total_events"] += len(events)
    state["sample_count"] += n

    return dec_start + n // SYNC_DEC


def _maybe_report(state: dict, t_report_ref: list, t_start: float) -> None:
    now = time.monotonic()
    if now >= t_report_ref[0]:
        elapsed = now - t_start
        dt = max(now - (t_report_ref[0] - 1.0), 0.001)
        rate = state["window_events"] / dt
        corr_ppm = (state["last_correction"] - 1.0) * 1e6
        state["last_corr_ppm"] = corr_ppm
        # Snapshot ppm at ~10 s for the drift check (after warmup + lock-in).
        if elapsed >= 10.0 and state["total_events"] >= 1000 and state["corr_ppm_at_10s"] is None:
            state["corr_ppm_at_10s"] = corr_ppm
        print(f"{elapsed:6.1f}  {state['total_events']:8d}  {rate:7.1f}  "
              f"{state['last_corr']:9.4f}  {corr_ppm:+8.1f} ppm  "
              f"{state['last_power_dbfs']:+7.1f} dBFS")
        state["window_events"] = 0
        t_report_ref[0] = now + 1.0


# ---------------------------------------------------------------------------
# Config-based path (recommended)
# ---------------------------------------------------------------------------

def run_with_config(args: argparse.Namespace) -> int:
    from beagle_node.config.schema import load_config
    from beagle_node.sdr.factory import create_receiver

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    receiver = create_receiver(config)
    sdr_rate = receiver.config.sample_rate_hz

    dec, dem, det, sync_rate = _make_pipeline(sdr_rate)
    _print_header(
        freq_hz=config.sync_signal.primary_station.frequency_hz,
        gain=float(receiver.config.gain_db) if receiver.config.gain_db != "auto" else 0,
        sdr_rate=sdr_rate,
        sync_rate=sync_rate,
        duration=args.duration,
        mode=config.sdr_mode,
    )

    state = _new_state()
    t_start = time.monotonic()
    t_report = [t_start + 1.0]
    dec_start = 0

    try:
        with receiver:
            if config.sdr_mode == "rspduo" and hasattr(receiver, "paired_stream"):
                # RSPduoReceiver.paired_stream() reads BOTH channels per iteration
                for sync_buf, target_buf, _buf_wall_ns in receiver.paired_stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    dec_start = _report_loop(dec, dem, det, sync_buf, len(sync_buf),
                                             dec_start, state)
                    _maybe_report(state, t_report, t_start)

            elif config.sdr_mode == "freq_hop" and hasattr(receiver, "labeled_stream"):
                # FreqHopReceiver alternates sync/target blocks
                for role, buf, _wall_ns in receiver.labeled_stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    if role == "sync":
                        dec_start = _report_loop(dec, dem, det, buf, len(buf),
                                                 dec_start, state)
                        _maybe_report(state, t_report, t_start)

            else:
                # single_sdr / two_sdr: single stream carries the sync channel
                for iq_buf in receiver.stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    dec_start = _report_loop(dec, dem, det, iq_buf, len(iq_buf),
                                             dec_start, state)
                    _maybe_report(state, t_report, t_start)

    except KeyboardInterrupt:
        print("\nStopped by user")

    return _print_summary(state, time.monotonic() - t_start, sync_rate)


# ---------------------------------------------------------------------------
# Simple device path (bare SoapySDR, no config file)
# ---------------------------------------------------------------------------

def run_simple(args: argparse.Namespace) -> int:
    import SoapySDR

    devs = SoapySDR.Device.enumerate(args.device)
    if not devs:
        print("ERROR: No SDR device found", file=sys.stderr)
        return 1

    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, args.rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.freq)
    if str(args.gain).lower() == "auto":
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
        gain_display = "auto"
    else:
        gain_val = float(args.gain)
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_val)
        gain_display = f"{gain_val:.0f} dB"

    rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
    sdr.activateStream(rx)

    dec, dem, det, sync_rate = _make_pipeline(args.rate)
    _print_header(args.freq, gain_display, args.rate, sync_rate, args.duration)

    buf_size = 131_072
    buf = np.zeros(buf_size, dtype=np.complex64)
    state = _new_state()
    t_start = time.monotonic()
    t_report = [t_start + 1.0]
    dec_start = 0

    try:
        while time.monotonic() - t_start < args.duration:
            sr = sdr.readStream(rx, [buf], buf_size, timeoutUs=2_000_000)
            if sr.ret < 0:
                print(f"\nERROR: readStream {sr.ret}", file=sys.stderr)
                break
            if sr.ret == 0:
                continue
            dec_start = _report_loop(dec, dem, det, buf, sr.ret, dec_start, state)
            _maybe_report(state, t_report, t_start)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)

    return _print_summary(state, time.monotonic() - t_start, sync_rate)


def _new_state() -> dict:
    return dict(
        last_corr=0.0,
        last_correction=1.0,
        last_phase=0.0,
        last_power_dbfs=-99.0,
        last_corr_ppm=0.0,
        window_events=0,
        total_events=0,
        sample_count=0,
        last_sample_index=None,
        bit_intervals=[],
        corr_ppm_at_10s=None,
    )


def _print_summary(state: dict, elapsed: float, sync_rate: float) -> int:
    n = state["total_events"]
    corr = state["last_corr"]
    print(f"\nTotal: {n} sync events in {elapsed:.1f} s ({n/max(elapsed,0.001):.1f}/s)")
    print(f"Expected: ~{RDS_BIT_RATE:.0f} events/s (one per RDS bit transition)")

    # Crystal drift check
    final_ppm = state.get("last_corr_ppm", 0.0)
    ppm_at_10 = state.get("corr_ppm_at_10s")
    if ppm_at_10 is not None and elapsed >= 15.0:
        drift = abs(final_ppm - ppm_at_10)
        drift_ok = drift < 10.0
        print(f"Crystal drift: {ppm_at_10:+.1f} ppm at t~10 s -> {final_ppm:+.1f} ppm at end  "
              f"(drift={drift:.1f} ppm  "
              f"{'OK' if drift_ok else 'WARN: >10 ppm - calibrator may not have converged'})")
    elif elapsed < 15.0:
        print(f"Crystal: {final_ppm:+.1f} ppm  (run >= 15 s for drift check)")

    # Bit interval jitter check
    intervals = state.get("bit_intervals", [])
    if len(intervals) >= 100:
        # Skip the first ~50 intervals (warmup transients) and any that are
        # large outliers from gap resets in freq_hop mode.
        clean = intervals[50:]
        expected_interval = sync_rate / RDS_BIT_RATE
        # Filter out gap-induced jumps (more than 10x expected)
        clean = [i for i in clean if abs(i - expected_interval) < expected_interval * 5]
        if len(clean) >= 100:
            mean_int = _stats.mean(clean)
            stdev_int = _stats.stdev(clean)
            jitter_us = stdev_int / sync_rate * 1e6
            print(f"Bit interval: mean={mean_int:.2f} samples "
                  f"(expected {expected_interval:.2f})  "
                  f"stdev={stdev_int:.3f} samples ({jitter_us:.2f} usec)")
            if jitter_us > 5.0:
                print(f"  WARN: high bit interval jitter (>{5.0} usec) - "
                      f"M&M timing recovery may be unstable")
        else:
            print(f"Bit interval: insufficient clean samples for stats")

    # Health summary
    expected_rate = RDS_BIT_RATE
    actual_rate = n / max(elapsed, 0.001)
    rate_ok = actual_rate > expected_rate * 0.7   # within 30% of expected
    pilot_ok = corr > 0.3

    if rate_ok and pilot_ok:
        print("OK: RDS sync detection looks good")
        return 0
    if not pilot_ok:
        print("WARNING: Low pilot correlation - check antenna, gain, or station selection")
    if not rate_ok:
        print(f"WARNING: Event rate ({actual_rate:.0f}/s) below expected "
              f"({expected_rate:.0f}/s) - station may not have RDS or signal too weak")
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.config:
        return run_with_config(args)
    return run_simple(args)


if __name__ == "__main__":
    sys.exit(main())
