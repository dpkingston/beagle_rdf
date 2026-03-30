#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
End-to-end verification of freq_hop mode using pyrtlsdr.

Feeds each alternating sync/target block through the full NodePipeline and
prints every TDOAMeasurement as it arrives.  Also shows a status line every
5 seconds with sync event rate and measurement count.

Usage examples
--------------
# Basic test: KISW 99.9 MHz sync, LMR target on 462.5625 MHz
python3 scripts/verify_freq_hop.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 --gain 30 --duration 60

# Longer run with custom thresholds
python3 scripts/verify_freq_hop.py \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 \\
    --gain 30 --onset-db -25 --min-corr 0.2 --duration 300

# Select a specific RTL-SDR by USB serial number
python3 scripts/verify_freq_hop.py \\
    --device-serial 00000001 \\
    --sync-freq 99.9e6 --target-freq 462.5625e6 --gain 30
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
        description="End-to-end freq_hop pipeline verification (requires pyrtlsdr)"
    )
    p.add_argument("--sync-freq",   type=float, required=True, metavar="HZ",
                   help="FM broadcast sync frequency (e.g. 99.9e6 for KISW)")
    p.add_argument("--target-freq", type=float, required=True, metavar="HZ",
                   help="LMR target frequency (e.g. 462.5625e6)")
    p.add_argument("--rate",        type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate in samples/sec (default 2.048e6)")
    p.add_argument("--gain",        type=float, default=30.0, metavar="DB",
                   help="Receiver gain in dB (default 30)")
    p.add_argument("--block",        type=int,   default=65_536, metavar="N",
                   help="Samples per sync block (default 65536 = ~32 ms)")
    p.add_argument("--target-block", type=int,   default=0,      metavar="N",
                   help="Samples per target block (default 0 = same as --block). "
                        "Set larger than --block to increase target duty cycle, "
                        "e.g. --block 65536 --target-block 131072 -> ~66%% target time.")
    p.add_argument("--settling",     type=int,   default=49_152, metavar="N",
                   help="Settling samples to discard per block (default 49152 = ~24 ms, "
                        "measured with measure_settling.py)")
    p.add_argument("--duration",    type=float, default=60.0, metavar="SEC",
                   help="Run duration in seconds (default 60)")
    p.add_argument("--onset-db",    type=float, default=-30.0, metavar="DB",
                   help="Carrier onset threshold dBFS (default -30)")
    p.add_argument("--offset-db",   type=float, default=-40.0, metavar="DB",
                   help="Carrier offset threshold dBFS (default -40)")
    p.add_argument("--min-corr",       type=float, default=0.1,   metavar="FLOAT",
                   help="Minimum pilot cross-correlation peak (default 0.1)")
    p.add_argument("--min-hold",       type=int,   default=4,     metavar="N",
                   help="Consecutive above-threshold carrier-detect windows required "
                        "before onset is declared (default 4 = 4 ms at 64 kHz). "
                        "Increase to suppress transient noise spikes.")
    p.add_argument("--max-sync-age-ms", type=float, default=0.0,  metavar="MS",
                   help="How long a SyncEvent stays valid for pairing with a carrier "
                        "onset (ms). Default: auto-computed from block sizes so the "
                        "full target block is covered. Increase if you see sync events "
                        "but no measurements.")
    p.add_argument("--device-serial",  type=str,   default=None,      metavar="SERIAL",
                   help="USB serial number of the RTL-SDR to use (default: first found). "
                        "Run rtl_test to list serials.")
    p.add_argument("--startup-pairs",  type=int,   default=1,         metavar="N",
                   help="Complete sync+target pairs to discard at startup for USB pipeline "
                        "drain and ADC/AGC stabilisation (default: 1 = ~1 cycle). "
                        "Increase to 2 if spurious onset detections appear at t=0.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-block decimated power and carrier detector state. "
                        "Use to diagnose false onsets or missing detections.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from beagle_node.sdr.base import SDRConfig
    from beagle_node.sdr.freq_hop import FreqHopReceiver
    from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig

    target_block = args.target_block if args.target_block > 0 else args.block

    # max_sync_age: a SyncEvent must remain valid long enough to pair with a
    # carrier onset anywhere in the following target block.
    # Auto-compute = full cycle time (sync + target) * 1.5 safety margin.
    # The DeltaComputer uses sync-decimated samples (rate / sync_decimation = rate/8).
    sync_dec = 8
    sync_dec_rate = args.rate / sync_dec
    if args.max_sync_age_ms > 0:
        max_sync_age_ms = args.max_sync_age_ms
    else:
        cycle_ms = (args.block + target_block) / args.rate * 1000
        max_sync_age_ms = cycle_ms * 1.5
    max_sync_age_samples = int(max_sync_age_ms * sync_dec_rate / 1000)

    sdr_cfg = SDRConfig(
        center_frequency_hz=args.target_freq,
        sample_rate_hz=args.rate,
        gain_db=args.gain,
    )
    receiver = FreqHopReceiver(
        config=sdr_cfg,
        sync_frequency_hz=args.sync_freq,
        samples_per_block=args.block,
        target_samples_per_block=target_block,
        settling_samples=args.settling,
        device_serial=args.device_serial,
    )

    measurements: list = []
    sync_event_count = 0
    block_count = 0
    adc_pos = 0   # tracks start of current block in the continuous ADC sample stream

    def on_measurement(m) -> None:
        measurements.append(m)
        tag = "ONSET " if m.event_type == "onset" else "OFFSET"
        print(
            f"  {tag} #{len(measurements):<4d}"
            f"  sync_delta={m.sync_delta_ns:+12,d} ns"
            f"  ({m.sync_delta_ns / 1e6:+.3f} ms)"
            f"  corr={m.corr_peak:.3f}"
            f"  power={m.onset_power_db:.1f} dBFS"
            f"  sync_samp={m.sync_sample:,}"
            f"  tgt_samp={m.target_sample:,}"
        )

    pipe_cfg = PipelineConfig(
        sdr_rate_hz=args.rate,
        carrier_onset_db=args.onset_db,
        carrier_offset_db=args.offset_db,
        min_corr_peak=args.min_corr,
        max_sync_age_samples=max_sync_age_samples,
        carrier_min_hold_windows=args.min_hold,
    )
    pipeline = NodePipeline(config=pipe_cfg, on_measurement=on_measurement)

    # Patch DeltaComputer to count sync events without modifying library code
    _orig_feed = pipeline._delta.feed_sync

    def _counting_feed(se):
        nonlocal sync_event_count
        sync_event_count += 1
        _orig_feed(se)

    pipeline._delta.feed_sync = _counting_feed

    # ------------------------------------------------------------------
    # Verbose mode: patch carrier detector to record decimated power
    # ------------------------------------------------------------------
    import numpy as np
    _last_target_power_db: list[float] = [float("nan")]

    if args.verbose:
        _orig_carrier_process = pipeline._carrier_det.process

        def _patched_carrier_process(iq_dec, start_sample):
            power_lin = float(np.mean(np.abs(iq_dec) ** 2))
            _last_target_power_db[0] = 10.0 * np.log10(power_lin + 1e-30)
            return _orig_carrier_process(iq_dec, start_sample)

        pipeline._carrier_det.process = _patched_carrier_process

    usable_target = max(0, target_block - args.settling)
    usable_sync   = max(0, args.block   - args.settling)
    cycle_ms      = (args.block + target_block) / args.rate * 1000
    duty_pct      = 100.0 * usable_target / (args.block + target_block)
    # Estimated sync event rate: sync events per usable-sync-ms / cycle_ms
    sync_period_ms   = 10.0   # FMPilotSyncDetector default
    usable_sync_ms   = usable_sync / args.rate * 1000
    est_sync_rate    = (usable_sync_ms / sync_period_ms) / (cycle_ms / 1000)

    print("freq_hop end-to-end verification")
    print(f"  Sync freq:      {args.sync_freq / 1e6:.3f} MHz")
    print(f"  Target:         {args.target_freq / 1e6:.3f} MHz")
    print(f"  Sync block:     {args.block:>7,} samples  ({args.block / args.rate * 1000:.1f} ms)"
          f"  -> {usable_sync_ms:.1f} ms usable")
    print(f"  Target block:   {target_block:>7,} samples  ({target_block / args.rate * 1000:.1f} ms)"
          f"  -> {usable_target / args.rate * 1000:.1f} ms usable")
    print(f"  Settling:       {args.settling:>7,} samples  ({args.settling / args.rate * 1000:.1f} ms)")
    print(f"  Target duty:    {duty_pct:.0f}%  of wall time is usable target")
    print(f"  Est. sync rate: {est_sync_rate:.1f}/s  (target ~100/s; low = sync blocks too small)")
    print(f"  Max sync age:   {max_sync_age_ms:.0f} ms  ({max_sync_age_samples:,} dec samples)")
    print(f"  Duration:       {args.duration:.0f} s")
    print(f"  Startup skip:   {args.startup_pairs} pair(s)  ({args.startup_pairs * cycle_ms:.0f} ms drain)")
    print(f"  Onset thr:      {args.onset_db:.1f} dBFS")
    print(f"  Min hold:       {args.min_hold} windows  ({args.min_hold * pipe_cfg.carrier_window_samples / (args.rate / pipe_cfg.target_decimation) * 1000:.1f} ms)")
    print(f"  Min corr:       {args.min_corr:.2f}")
    if est_sync_rate < 5.0:
        print(f"\n  WARNING: estimated sync rate {est_sync_rate:.1f}/s is low.")
        print(f"    Increase --block (sync block size) to improve sync event rate.")
        print(f"    e.g. --block 131072 gives {usable_sync_ms * 2:.0f} ms usable sync -> ~{est_sync_rate * 2:.0f}/s")
    print()

    # Use SIGALRM so the run exits after exactly `duration` seconds even if
    # labeled_stream() is blocked waiting for data from the background thread.
    def _alarm_handler(signum, frame):
        raise _DurationExpired

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max(1, int(args.duration + 0.5)))

    t_start = time.monotonic()
    t_last_status = t_start

    # Startup drain counter: discard the first N complete sync+target pairs to
    # let the USB pipeline fill cleanly and the ADC/AGC stabilise.  Without
    # this, the very first target block often contains a high-power transient
    # that produces a spurious CarrierOnset and leaves the detector in "active"
    # state, masking real transmissions until the transient decays.
    startup_blocks_left = args.startup_pairs * 2

    try:
        with receiver:
            for role, iq_buf, _wall_ns in receiver.labeled_stream():
                now = time.monotonic()

                block_size = args.block if role == "sync" else target_block

                # Drain startup transient -- advance adc_pos but skip pipeline
                if startup_blocks_left > 0:
                    startup_blocks_left -= 1
                    adc_pos += block_size
                    continue

                # raw_start_sample = ADC index of the first *usable* sample
                # (after settling) in the continuous ADC stream.
                raw_start = adc_pos + args.settling
                if role == "sync":
                    sync_before = sync_event_count
                    pipeline.process_sync_buffer(iq_buf, raw_start_sample=raw_start)
                    if args.verbose:
                        n_new = sync_event_count - sync_before
                        print(
                            f"  [sync  #{block_count // 2 + 1:4d}]"
                            f"  sync_events_this_block={n_new:2d}"
                            f"  total={sync_event_count}"
                        )
                else:
                    pipeline.process_target_buffer(iq_buf, raw_start_sample=raw_start)
                    if args.verbose:
                        print(
                            f"  [target #{block_count // 2:4d}]"
                            f"  dec_power={_last_target_power_db[0]:+.1f} dBFS"
                            f"  (onset={args.onset_db:.0f} / offset={args.offset_db:.0f})"
                            f"  detector={pipeline._carrier_det.state}"
                        )
                block_count += 1
                adc_pos += block_size

                # Print a status line every 5 seconds
                if now - t_last_status >= 5.0:
                    elapsed = now - t_start
                    n_pairs = block_count // 2
                    sync_rate = sync_event_count / max(elapsed, 0.001)
                    print(
                        f"  [{elapsed:5.0f} s]"
                        f"  block pairs={n_pairs}"
                        f"  sync_rate={sync_rate:.1f}/s"
                        f"  measurements={len(measurements)}"
                    )
                    t_last_status = now

    except _DurationExpired:
        pass   # clean exit after timer fires
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        signal.alarm(0)   # cancel any pending alarm

    elapsed = time.monotonic() - t_start
    n_pairs = block_count // 2

    print(f"\n--- Summary ({elapsed:.1f} s) ---")
    print(f"Block pairs processed:  {n_pairs}")
    print(f"Sync events:            {sync_event_count}  ({sync_event_count / max(elapsed, 1):.1f}/s"
          f"  target ~100/s)")
    print(f"Measurements:           {len(measurements)}")

    if len(measurements) >= 2:
        deltas = [m.sync_delta_ns for m in measurements]
        mean_ns = sum(deltas) // len(deltas)
        print(f"sync_delta_ns:          mean={mean_ns:+,}  min={min(deltas):+,}  max={max(deltas):+,}")

    actual_sync_rate = sync_event_count / max(elapsed, 1)
    if sync_event_count == 0:
        print("\nWARNING: No sync events produced.")
        print("  Check: sync frequency, antenna, and that pyrtlsdr is installed (pip install pyrtlsdr).")
    elif len(measurements) == 0:
        print("\nNo measurements produced.")
        if actual_sync_rate < 5.0:
            print(f"  Low sync rate ({actual_sync_rate:.1f}/s): increase --block to get more sync events.")
            print(f"  Try: --block 131072 (64 ms sync block)")
        print("  If no LMR transmission was present during the run, that is expected.")
        print(f"  To detect weaker signals lower the threshold: --onset-db {args.onset_db - 10:.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
