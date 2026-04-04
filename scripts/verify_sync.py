#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Live FM pilot sync detection verification.

Runs the same SDR receiver and sync pipeline that the node uses in production,
reporting live statistics: events/sec, correlation peak, crystal correction.

A high correlation peak (> 0.3) and stable crystal correction (near 1.0)
confirm that the 19 kHz pilot is being reliably extracted.

Primary usage (recommended) - uses the actual node receiver + pipeline config:
--------------------------------------------------------------------
python3 scripts/verify_sync.py --config config/node.yaml --duration 30

Simple usage - bare SoapySDR device, no config file needed:
--------------------------------------------------------------------
python3 scripts/verify_sync.py --device "driver=rtlsdr" --freq 99.9e6 --gain 30 --duration 30
python3 scripts/verify_sync.py --device "driver=sdrplay" --freq 94.9e6 --gain auto --duration 30

The --config path uses create_receiver() and the same pipeline parameters as
the node, so any discrepancy between this script and the live node is eliminated.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

# Add project src to path when run as a script
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.sync_detector import FMPilotSyncDetector

# These match the pipeline defaults in pipeline.py / PipelineConfig
SYNC_DEC  = 8
CUTOFF_HZ = 128_000.0
PERIOD_MS = 10.0            # overridden by config when --config is used


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify FM pilot sync detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", metavar="PATH",
                     help="Node YAML config file (recommended; uses the real receiver)")
    src.add_argument("--device", metavar="ARGS", default=None,
                     help="Simple SoapySDR device args (RTL-SDR only; requires --freq)")

    p.add_argument("--freq",     type=float, default=99.9e6, metavar="HZ",
                   help="FM station frequency (default 99.9e6 = KISW); --device mode only")
    p.add_argument("--gain",     default="0", metavar="DB",
                   help="Receiver gain in dB, or 'auto' for AGC (default 0); --device mode only")
    p.add_argument("--rate",     type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate (default 2.048e6); --device mode only")
    p.add_argument("--duration", type=float, default=30.0, metavar="SEC",
                   help="Run for this many seconds then exit (default 30)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sync processing helpers (shared by both paths)
# ---------------------------------------------------------------------------

def _make_pipeline(sdr_rate: float, period_ms: float) -> tuple:
    """Return (decimator, demodulator, detector) for the given SDR rate."""
    sync_rate = sdr_rate / SYNC_DEC
    dec  = Decimator(SYNC_DEC, sdr_rate, CUTOFF_HZ)
    dem  = FMDemodulator(sync_rate)
    det  = FMPilotSyncDetector(sync_rate, sync_period_ms=period_ms,
                                calibration_window=100)
    return dec, dem, det, sync_rate


def _print_header(freq_hz: float, gain, sdr_rate: float, sync_rate: float,
                  duration: float, mode: str = "") -> None:
    mode_str = f"  [{mode}]" if mode else ""
    gain_str = str(gain) if isinstance(gain, str) else f"{gain:.0f} dB"
    print(f"Tuned to {freq_hz/1e6:.1f} MHz  gain={gain_str}  "
          f"rate={sdr_rate/1e6:.3f} MSps{mode_str}")
    print(f"Pipeline: {SYNC_DEC}* decimation -> {sync_rate/1e3:.0f} kHz "
          f"-> FM demod -> 19 kHz pilot detector")
    print(f"Running for {duration:.0f} s  (Ctrl-C to stop early)\n")
    print(f"{'Time':>6}  {'Events':>8}  {'Rate/s':>7}  "
          f"{'CorPeak':>8}  {'Crystal':>10}  {'Power':>8}")
    print(f"  (ideal FM power: -10 to -40 dBFS; crystal <+/-50 ppm when pilot is good)")
    print("-" * 65)


def _report_loop(dec, dem, det, buf: np.ndarray, n: int,
                 dec_start: int, state: dict) -> int:
    """Process one IQ buffer through the sync pipeline; update state dict.

    Returns the new dec_start (for next call).
    """
    import math
    iq = buf[:n]

    # Raw IQ power in dBFS (full scale = 1.0 for CF32 SoapySDR output).
    # Ideal FM signal: -10 to -40 dBFS.
    # Near 0 dBFS -> ADC saturation (reduce gain).
    # Below -50 dBFS -> signal too weak or wrong antenna (increase gain or check connections).
    power_linear = float(np.mean(np.abs(iq) ** 2))
    state["last_power_dbfs"] = 10.0 * math.log10(max(power_linear, 1e-12))

    iq_dec = dec.process(iq)
    audio  = dem.process(iq_dec)
    events = det.process(audio, start_sample=dec_start)

    pilot_period = det.sample_rate_hz / 19_000.0   # ~13.47 samples at 256 kHz
    # Expected stdev of (sample_index % pilot_period) if Phase 1 interpolation is active:
    # offsets are uniformly distributed in [-half_period, +half_period] -> sigma = pilot_period/sqrt12.
    # Without Phase 1 (window-centre only), the value is constant -> sigma ~ 0.
    state["pilot_period"] = pilot_period
    state["sync_period_samples"] = det.sync_period_samples

    for e in events:
        state["last_corr"]       = e.corr_peak
        state["last_correction"] = e.sample_rate_correction
        state["last_phase"]      = e.pilot_phase_rad
        # Record (sample_index % pilot_period) to diagnose Phase 1 interpolation.
        # After Phase 1: offsets span all zero-crossings -> sigma ~ pilot_period/sqrt12 ~ 3.8 samples.
        # Before Phase 1: constant window-centre -> sigma ~ 0.
        state["phase_mods"].append(e.sample_index % pilot_period)

    state["window_events"] += len(events)
    state["total_events"]  += len(events)
    state["sample_count"]  += n

    return dec_start + n // SYNC_DEC


def _maybe_report(state: dict, t_report_ref: list, t_start: float) -> None:
    now = time.monotonic()
    if now >= t_report_ref[0]:
        elapsed = now - t_start
        dt = max(now - (t_report_ref[0] - 1.0), 0.001)
        rate = state["window_events"] / dt
        corr_ppm = (state["last_correction"] - 1.0) * 1e6
        state["last_corr_ppm"] = corr_ppm
        # Snapshot ppm for the drift check in the summary.
        # Require >= 100 events so the calibrator has had real data to converge
        # before we lock in the baseline.  On hardware with a long initialisation
        # delay (e.g. RSPduo sdrplay service startup, ~15 s of timeouts) the
        # elapsed-time check alone fires too early, during the lock-in transient.
        if elapsed >= 10.0 and state["total_events"] >= 100 and state["corr_ppm_at_10s"] is None:
            state["corr_ppm_at_10s"] = corr_ppm
        print(f"{elapsed:6.1f}  {state['total_events']:8d}  {rate:7.1f}  "
              f"{state['last_corr']:8.4f}  {corr_ppm:+8.1f} ppm  "
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
    period_ms = config.sync_signal.sync_period_ms

    dec, dem, det, sync_rate = _make_pipeline(sdr_rate, period_ms)
    _print_header(
        freq_hz=config.sync_signal.primary_station.frequency_hz,
        gain=float(receiver.config.gain_db) if receiver.config.gain_db != "auto" else 0,
        sdr_rate=sdr_rate,
        sync_rate=sync_rate,
        duration=args.duration,
        mode=config.sdr_mode,
    )

    state = dict(last_corr=0.0, last_correction=1.0, last_phase=0.0,
                 last_power_dbfs=-99.0, last_corr_ppm=0.0,
                 window_events=0, total_events=0, sample_count=0,
                 phase_mods=[], corr_ppm_at_10s=None)
    t_start = time.monotonic()
    t_report = [t_start + 1.0]
    dec_start = 0

    buf_sync = None   # allocated on first buffer

    try:
        with receiver:
            if config.sdr_mode == "rspduo" and hasattr(receiver, "paired_stream"):
                # RSPduoReceiver.paired_stream() reads BOTH channels on every
                # iteration - the only correct way to drive DT mode.  We only
                # feed the sync buffer into the pilot pipeline.
                dummy_buf = None
                for sync_buf, target_buf, _buf_wall_ns in receiver.paired_stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    n = len(sync_buf)
                    dec_start = _report_loop(dec, dem, det, sync_buf, n,
                                             dec_start, state)
                    _maybe_report(state, t_report, t_start)

            elif config.sdr_mode == "freq_hop" and hasattr(receiver, "labeled_stream"):
                # FreqHopReceiver.labeled_stream() alternates sync/target blocks.
                # Only sync-role blocks are fed into the pilot pipeline.
                for role, buf, _wall_ns in receiver.labeled_stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    if role == "sync":
                        dec_start = _report_loop(dec, dem, det, buf, len(buf),
                                                 dec_start, state)
                        _maybe_report(state, t_report, t_start)

            else:
                # single_sdr / two_sdr: single stream carries the sync channel.
                buf_size = 131_072
                buf_sync = np.zeros(buf_size, dtype=np.complex64)
                for iq_buf in receiver.stream():
                    if time.monotonic() - t_start >= args.duration:
                        break
                    dec_start = _report_loop(dec, dem, det, iq_buf, len(iq_buf),
                                             dec_start, state)
                    _maybe_report(state, t_report, t_start)

    except KeyboardInterrupt:
        print("\nStopped by user")

    return _print_summary(state, time.monotonic() - t_start)


# ---------------------------------------------------------------------------
# Simple device path (RTL-SDR / bare SoapySDR, no config file)
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
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)   # AGC
        gain_display = "auto"
    else:
        gain_val = float(args.gain)
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_val)
        gain_display = f"{gain_val:.0f} dB"

    rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
    sdr.activateStream(rx)

    dec, dem, det, sync_rate = _make_pipeline(args.rate, PERIOD_MS)
    _print_header(args.freq, gain_display, args.rate, sync_rate, args.duration)

    buf_size = 131_072
    buf = np.zeros(buf_size, dtype=np.complex64)
    state = dict(last_corr=0.0, last_correction=1.0, last_phase=0.0,
                 last_power_dbfs=-99.0, last_corr_ppm=0.0,
                 window_events=0, total_events=0, sample_count=0,
                 phase_mods=[], corr_ppm_at_10s=None)
    t_start  = time.monotonic()
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

    return _print_summary(state, time.monotonic() - t_start)


def _print_summary(state: dict, elapsed: float) -> int:
    import statistics as _stats
    n = state["total_events"]
    corr = state["last_corr"]
    print(f"\nTotal: {n} sync events in {elapsed:.1f} s ({n/max(elapsed,0.001):.1f}/s)")

    # Crystal drift check (Step 3 of calibration procedure).
    # Report the ppm value at ~10 s, at end-of-run, and the drift between them.
    # A stable calibrator drifts < 10 ppm over 50 s; > 50 ppm suggests a
    # noisy crystal or a pilot detection problem.
    final_ppm = state.get("last_corr_ppm", 0.0)
    ppm_at_10 = state.get("corr_ppm_at_10s")
    if ppm_at_10 is not None and elapsed >= 15.0:
        drift = abs(final_ppm - ppm_at_10)
        drift_ok = drift < 10.0
        print(f"Crystal drift: {ppm_at_10:+.1f} ppm at t~10 s -> {final_ppm:+.1f} ppm at end  "
              f"(drift={drift:.1f} ppm  {'OK' if drift_ok else 'WARN: >10 ppm - calibrator may not have converged'})")
    elif elapsed < 15.0:
        print(f"Crystal: {final_ppm:+.1f} ppm  (run >= 60 s for drift check)")

    mods = state.get("phase_mods", [])
    pilot_period = state.get("pilot_period", 13.16)
    expected_stdev = pilot_period / (12 ** 0.5)   # sigma of uniform dist over [0, pilot_period)
    if len(mods) >= 10:
        stdev_mod = _stats.stdev(mods)
        ratio = stdev_mod / expected_stdev
        # Check whether the sync period contains an integer number of pilot cycles.
        # When sync_period / pilot_period ~ integer (e.g. 7 ms x 19 kHz = 133.0 exactly),
        # Phase 1 always snaps to the same zero-crossing, giving sigma~0 even though
        # interpolation is working correctly.  The uniform-distribution diagnostic only
        # applies when the ratio is non-integer.
        sync_period_samples = state.get("sync_period_samples", 0)
        cycles = sync_period_samples / pilot_period if pilot_period > 0 else 0.0
        near_integer = abs(cycles - round(cycles)) < 0.02
        print(f"Phase 1 pilot interpolation: phase_mod stdev={stdev_mod:.3f} samples  "
              f"(expected ~{expected_stdev:.2f} if active; sync={sync_period_samples} "
              f"samples = {cycles:.2f} pilot cycles)")
        if near_integer:
            print(f"  OK (integer-cycle sync period): sigma/sigma_expected={ratio:.2f} - "
                  f"near-integer pilot cycles per window means Phase 1 always picks the "
                  f"same zero-crossing.  sigma~0 is correct, not a sign of missing interpolation.")
        elif ratio > 0.7:
            print(f"  ACTIVE: sigma/sigma_expected={ratio:.2f} - interpolation is running "
                  f"(each event snapped to its nearest zero-crossing)")
        else:
            print(f"  INACTIVE or constant: sigma/sigma_expected={ratio:.2f} - "
                  f"events may be stuck at window-centre (sigma~0 means no interpolation)")

    if n > 0 and corr > 0.3:
        print("OK: Pilot detection looks good")
        return 0
    else:
        print("WARNING: Low correlation peak - check antenna, gain, or SDR setup")
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
