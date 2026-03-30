#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
End-to-end verification of RSPduo mode using SoapySDRPlay3.

Opens the RSPduo as a dual-tuner receiver (master=sync, slave=target),
feeds each buffer pair through the full NodePipeline, and prints every
TDOAMeasurement as it arrives.  Also shows a status line every 5 seconds
with sync event rate, measurement count, and overflow count.

Requires the SDRplay API and SoapySDRPlay3 plugin:
    See docs/setup-rspduo-debian.md for installation steps.
    (SoapySDRPlay3 is not in apt; it must be built from source.)

Usage examples
--------------
# Basic test: KISW 99.9 MHz sync, LMR target on 462.5625 MHz
python3 scripts/verify_rspduo.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6

# Custom gains and thresholds
python3 scripts/verify_rspduo.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 \\
    --sync-gain 30 --target-gain 40 \\
    --onset-db -15 --offset-db -25 --min-corr 0.2 --duration 120

# Specific RSPduo by serial number
python3 scripts/verify_rspduo.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 \\
    --device-args "driver=sdrplay,serial=XXXXXXXXX"

# Verbose mode: print per-buffer power and carrier detector state
python3 scripts/verify_rspduo.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 --verbose
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


class _DurationExpired(Exception):
    """Raised by SIGALRM handler when the run duration has elapsed."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end RSPduo pipeline verification (requires SoapySDRPlay3)"
    )
    p.add_argument("--sync-freq",   type=float, required=True, metavar="HZ",
                   help="FM broadcast sync frequency (e.g. 99.9e6 for KISW)")
    p.add_argument("--target-freq", type=float, required=True, metavar="HZ",
                   help="LMR target frequency (e.g. 462.5625e6)")
    p.add_argument("--rate",        type=float, default=2_000_000.0, metavar="SPS",
                   help="Sample rate in samples/sec (default 2.0e6; max 2 MHz for RSPduo dual-tuner)")
    p.add_argument("--sync-gain",   default="auto", metavar="DB",
                   help="Sync channel gain in dB, or 'auto' for AGC (default: auto)")
    p.add_argument("--target-gain", default="auto", metavar="DB",
                   help="Target channel gain in dB, or 'auto' for AGC (default: auto)")
    p.add_argument("--device-args", type=str, default="driver=sdrplay", metavar="ARGS",
                   help="SoapySDR device args (default: 'driver=sdrplay'). "
                        "Use 'driver=sdrplay,serial=XXXX' for a specific unit.")
    p.add_argument("--buffer",      type=int, default=65_536, metavar="N",
                   help="IQ samples per read per channel (default 65536 = ~32 ms at 2 MSPS)")
    p.add_argument("--duration",    type=float, default=60.0, metavar="SEC",
                   help="Run duration in seconds (default 60)")
    p.add_argument("--onset-db",    type=float, default=-15.0, metavar="DB",
                   help="Carrier onset threshold dBFS (default -15; field-calibrated)")
    p.add_argument("--offset-db",   type=float, default=-25.0, metavar="DB",
                   help="Carrier offset threshold dBFS (default -25; field-calibrated)")
    p.add_argument("--min-corr",    type=float, default=0.1,   metavar="FLOAT",
                   help="Minimum pilot cross-correlation peak to accept a sync event (default 0.1)")
    p.add_argument("--min-hold",    type=int,   default=4,     metavar="N",
                   help="Consecutive above-threshold carrier-detect windows required "
                        "before onset is declared (default 4 = ~4 ms at 64 kHz). "
                        "Increase to suppress transient noise spikes.")
    p.add_argument("--startup-buffers", type=int, default=2, metavar="N",
                   help="Buffer pairs to discard at startup for AGC and pilot lock stabilisation "
                        "(default 2 = ~64 ms at 2 MSPS). Increase if spurious onsets appear at t=0.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-buffer sync event count, decimated power, and detector state.")
    return p.parse_args()


def _parse_gain(raw: str) -> float | str:
    """Return numeric float for a dB string, or 'auto' unchanged."""
    if raw.lower() == "auto":
        return "auto"
    return float(raw)


def main() -> int:
    args = parse_args()

    from beagle_node.sdr.rspduo import RSPduoReceiver
    from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig

    sync_gain   = _parse_gain(args.sync_gain)
    target_gain = _parse_gain(args.target_gain)

    receiver = RSPduoReceiver(
        sync_frequency_hz=args.sync_freq,
        target_frequency_hz=args.target_freq,
        sample_rate_hz=args.rate,
        sync_gain_db=sync_gain,
        target_gain_db=target_gain,
        master_device_args=args.device_args,
        buffer_size=args.buffer,
    )

    # The RSPduo delivers continuous simultaneous buffers; no freq-hop
    # settling time and no coverage gaps.  The DeltaComputer must keep
    # sync events valid for at least one full buffer period (plus margin).
    #
    # Build PipelineConfig first so sync_decimation comes from the config
    # itself rather than being duplicated here.
    pipe_cfg = PipelineConfig(
        sdr_rate_hz=args.rate,
        carrier_onset_db=args.onset_db,
        carrier_offset_db=args.offset_db,
        min_corr_peak=args.min_corr,
        carrier_min_hold_windows=args.min_hold,
    )
    sync_dec_rate = args.rate / pipe_cfg.sync_decimation
    # Keep sync events valid for ~2.5 buffer lengths to handle pipeline latency.
    max_sync_age_ms = args.buffer / args.rate * 1000 * 2.5
    max_sync_age_samples = int(max_sync_age_ms * sync_dec_rate / 1000)
    pipe_cfg.max_sync_age_samples = max_sync_age_samples

    measurements: list = []
    sync_event_count = 0
    buffer_pair_count = 0
    sample_pos = 0   # ADC sample position shared by both channels

    def on_measurement(m) -> None:
        measurements.append(m)
        tag = "ONSET " if m.event_type == "onset" else "OFFSET"
        print(
            f"  {tag} #{len(measurements):<4d}"
            f"  sync_delta={m.sync_delta_ns:+12,d} ns"
            f"  ({m.sync_delta_ns / 1e6:+.3f} ms)"
            f"  corr={m.corr_peak:.3f}"
            f"  power={m.onset_power_db:.1f} dBFS"
        )

    pipeline = NodePipeline(config=pipe_cfg, on_measurement=on_measurement)

    # Count sync events by wrapping DeltaComputer.feed_sync
    _orig_feed = pipeline._delta.feed_sync

    def _counting_feed(se):
        nonlocal sync_event_count
        sync_event_count += 1
        _orig_feed(se)

    pipeline._delta.feed_sync = _counting_feed

    # Verbose mode: capture decimated target power each buffer
    import numpy as np
    _last_target_power_db: list[float] = [float("nan")]

    if args.verbose:
        _orig_carrier_process = pipeline._carrier_det.process

        def _patched_carrier_process(iq_dec, start_sample):
            power_lin = float(np.mean(np.abs(iq_dec) ** 2))
            _last_target_power_db[0] = 10.0 * np.log10(power_lin + 1e-30)
            return _orig_carrier_process(iq_dec, start_sample)

        pipeline._carrier_det.process = _patched_carrier_process

    # Expected sync rate: at 2 MSPS, a 65536-sample buffer = 32 ms;
    # sync events every 10 ms -> ~3 per buffer.
    buffer_ms = args.buffer / args.rate * 1000
    est_sync_rate = 1000.0 / 10.0   # FMPilotSyncDetector emits every 10 ms by default

    sync_gain_str   = f"{sync_gain} dB"   if sync_gain != "auto" else "auto (AGC)"
    target_gain_str = f"{target_gain} dB" if target_gain != "auto" else "auto (AGC)"

    print("RSPduo end-to-end verification")
    print(f"  Device args:    {args.device_args}")
    print(f"  Sync freq:      {args.sync_freq / 1e6:.3f} MHz   gain={sync_gain_str}")
    print(f"  Target freq:    {args.target_freq / 1e6:.3f} MHz   gain={target_gain_str}")
    print(f"  Sample rate:    {args.rate / 1e6:.3f} MSPS")
    print(f"  Buffer size:    {args.buffer:>7,} samples  ({buffer_ms:.1f} ms)")
    print(f"  Max sync age:   {max_sync_age_ms:.0f} ms  ({max_sync_age_samples:,} dec samples)")
    print(f"  Est. sync rate: {est_sync_rate:.0f}/s  (target ~100/s)")
    print(f"  Duration:       {args.duration:.0f} s")
    print(f"  Startup skip:   {args.startup_buffers} buffer pair(s)  ({args.startup_buffers * buffer_ms:.0f} ms drain)")
    print(f"  Onset thr:      {args.onset_db:.1f} dBFS")
    print(f"  Offset thr:     {args.offset_db:.1f} dBFS")
    print(f"  Min hold:       {args.min_hold} windows")
    print(f"  Min corr:       {args.min_corr:.2f}")
    print()

    # SIGALRM exits cleanly after the requested duration even if readStream is
    # blocked waiting for data.
    def _alarm_handler(signum, frame):
        raise _DurationExpired

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max(1, int(args.duration + 0.5)))

    t_start = time.monotonic()
    t_last_status = t_start
    startup_left = args.startup_buffers

    try:
        with receiver:
            for sync_buf, target_buf, _buf_wall_ns in receiver.paired_stream():
                now = time.monotonic()

                # Discard startup pairs to allow AGC and pilot phase-lock to settle
                if startup_left > 0:
                    startup_left -= 1
                    sample_pos += len(sync_buf)
                    continue

                # Both channels share the same ADC clock and the same sample
                # position.  Pass the same raw_start_sample to both stages.
                sync_before = sync_event_count
                pipeline.process_sync_buffer(sync_buf, raw_start_sample=sample_pos)
                pipeline.process_target_buffer(target_buf, raw_start_sample=sample_pos)
                sample_pos += len(sync_buf)
                buffer_pair_count += 1

                if args.verbose:
                    n_new_sync = sync_event_count - sync_before
                    print(
                        f"  [pair #{buffer_pair_count:4d}]"
                        f"  sync_events={n_new_sync:2d}  total={sync_event_count}"
                        f"  target_power={_last_target_power_db[0]:+.1f} dBFS"
                        f"  (onset={args.onset_db:.0f} / offset={args.offset_db:.0f})"
                        f"  detector={pipeline._carrier_det.state}"
                    )

                if now - t_last_status >= 5.0:
                    elapsed = now - t_start
                    sync_rate = sync_event_count / max(elapsed, 0.001)
                    overflows = receiver.overflow_count
                    print(
                        f"  [{elapsed:5.0f} s]"
                        f"  buffer_pairs={buffer_pair_count}"
                        f"  sync_rate={sync_rate:.1f}/s"
                        f"  measurements={len(measurements)}"
                        f"  overflows={overflows}"
                    )
                    t_last_status = now

    except _DurationExpired:
        pass
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        signal.alarm(0)

    elapsed = time.monotonic() - t_start

    print(f"\n--- Summary ({elapsed:.1f} s) ---")
    print(f"Buffer pairs processed: {buffer_pair_count}")
    print(f"Sync events:            {sync_event_count}  ({sync_event_count / max(elapsed, 1):.1f}/s"
          f"  target ~100/s)")
    print(f"Overflows:              {receiver.overflow_count}")
    print(f"Measurements:           {len(measurements)}")

    if len(measurements) >= 2:
        deltas = [m.sync_delta_ns for m in measurements]
        mean_ns = sum(deltas) // len(deltas)
        print(f"sync_delta_ns:          mean={mean_ns:+,}  min={min(deltas):+,}  max={max(deltas):+,}")

    actual_sync_rate = sync_event_count / max(elapsed, 1)
    if sync_event_count == 0:
        print("\nWARNING: No sync events produced.")
        print("  Check: SoapySDRPlay3 is installed, RSPduo is connected, sync frequency is correct.")
        print("  Run: SoapySDRUtil --find to confirm the device is visible.")
    elif len(measurements) == 0:
        print("\nNo measurements produced.")
        if actual_sync_rate < 5.0:
            print(f"  Low sync rate ({actual_sync_rate:.1f}/s): check sync frequency and antenna.")
        print("  If no LMR transmission was present during the run, that is expected.")
        print(f"  To detect weaker signals lower the threshold: --onset-db {args.onset_db - 10:.0f}")

    if receiver.overflow_count > 0:
        print(f"\nWARNING: {receiver.overflow_count} buffer overflow(s).")
        print("  Reduce --buffer size or improve USB bandwidth (use a USB 3 port).")
        print("  Overflows introduce gaps in the IQ stream and degrade timing accuracy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
