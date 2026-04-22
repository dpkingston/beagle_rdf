#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Feed stored IQ .npy files through the signal processing pipeline offline.

Useful for:
  - Tuning carrier detect thresholds without live hardware
  - Verifying pipeline behaviour on captured real signals
  - Offline debugging / regression testing of detectors

Usage examples
--------------
# Count FM pilot sync events in a captured FM IQ file:
python3 scripts/replay_iq.py --sync tests/fixtures/iq_fm_kisw_99.9.npy

# Detect LMR carrier onsets (no TDOA measurement without sync):
python3 scripts/replay_iq.py --target tests/fixtures/iq_lmr_462.npy

# Full pipeline - FM sync + LMR target captured at the same time (single_sdr):
python3 scripts/replay_iq.py \\
    --sync   tests/fixtures/iq_fm_kisw_99.9.npy \\
    --target tests/fixtures/iq_lmr_462.npy

# Tune detection thresholds (lower onset-db detects weaker carriers):
python3 scripts/replay_iq.py \\
    --target tests/fixtures/iq_lmr_462.npy \\
    --onset-db -20 --offset-db -30

# Replay a freq_hop capture where sync and target alternate with known offset:
# (use --freq-hop-block to specify the block size used during capture)
python3 scripts/replay_iq.py \\
    --sync   capture_sync.npy \\
    --target capture_target.npy \\
    --freq-hop-block 65536
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay IQ .npy files through the Beagle pipeline"
    )
    p.add_argument("--sync",   type=str, metavar="PATH",
                   help="Sync channel IQ .npy file (FM broadcast at SDR rate)")
    p.add_argument("--target", type=str, metavar="PATH",
                   help="Target channel IQ .npy file (LMR at SDR rate)")
    p.add_argument("--rate",   type=float, default=2_048_000.0, metavar="SPS",
                   help="SDR sample rate of the input files (default 2.048e6)")
    p.add_argument("--buffer-size", type=int, default=65_536, metavar="N",
                   help="Buffer size for pipeline feeding in samples (default 65536)")

    # Carrier detect tuning
    p.add_argument("--onset-db",  type=float, default=-30.0, metavar="DB",
                   help="Carrier onset threshold dBFS (default -30)")
    p.add_argument("--offset-db", type=float, default=-40.0, metavar="DB",
                   help="Carrier offset threshold dBFS (default -40)")
    p.add_argument("--window",    type=int,   default=64,    metavar="N",
                   help="Power averaging window in target-decimated samples (default 64)")
    p.add_argument("--min-corr",  type=float, default=0.1,   metavar="FLOAT",
                   help="Minimum pilot corr_peak to accept (default 0.1)")

    # freq_hop replay option
    p.add_argument("--freq-hop-block", type=int, default=0, metavar="N",
                   help="If set, treat sync and target as freq_hop blocks with this "
                        "block size (samples).  Feeds them with correct raw_start_sample "
                        "offsets matching how FreqHopReceiver numbers them.")

    return p.parse_args()


def _load(path: str, label: str, rate: float) -> np.ndarray:
    iq = np.load(path).astype(np.complex64)
    duration = len(iq) / rate
    power_db = 10 * np.log10(float(np.mean(np.abs(iq) ** 2)) + 1e-30)
    print(f"{label}: {len(iq):,} samples  {duration:.2f} s  "
          f"{power_db:.1f} dBFS  <- {path}")
    return iq


def main() -> int:
    args = parse_args()

    if not args.sync and not args.target:
        print("ERROR: at least one of --sync or --target is required", file=sys.stderr)
        return 1

    from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig

    sync_iq: np.ndarray | None = None
    target_iq: np.ndarray | None = None

    if args.sync:
        sync_iq = _load(args.sync, "Sync IQ  ", args.rate)
    if args.target:
        target_iq = _load(args.target, "Target IQ", args.rate)
    print()

    sync_events: list = []
    measurements: list = []

    def on_measurement(m) -> None:
        measurements.append(m)
        print(
            f"  MEASUREMENT #{len(measurements):<4d}"
            f"  sync_delta={m.sync_to_snippet_start_ns:+12,d} ns"
            f"  ({m.sync_to_snippet_start_ns / 1e6:+.3f} ms)"
            f"  corr={m.corr_peak:.3f}"
            f"  power={m.onset_power_db:.1f} dBFS"
        )

    cfg = PipelineConfig(
        sdr_rate_hz=args.rate,
        carrier_onset_db=args.onset_db,
        carrier_offset_db=args.offset_db,
        carrier_window_samples=args.window,
        min_corr_peak=args.min_corr,
    )
    pipeline = NodePipeline(config=cfg, on_measurement=on_measurement)

    # Patch to count sync events
    _orig_feed = pipeline._delta.feed_sync

    def _counting_feed(se):
        sync_events.append(se)
        _orig_feed(se)

    pipeline._delta.feed_sync = _counting_feed

    B = args.buffer_size
    freq_hop_block = args.freq_hop_block

    n_total = max(
        len(sync_iq)   if sync_iq   is not None else 0,
        len(target_iq) if target_iq is not None else 0,
    )
    duration_s = n_total / args.rate

    print(f"Replaying {duration_s:.2f} s of IQ ...")
    print(f"  onset threshold: {args.onset_db:.1f} dBFS   "
          f"offset: {args.offset_db:.1f} dBFS   "
          f"min_corr: {args.min_corr:.2f}")
    print()

    t_start = time.monotonic()

    if freq_hop_block > 0:
        # -------------------------------------------------------------------
        # freq_hop mode: sync and target are separate files, fed with the
        # raw_start_sample offsets that FreqHopReceiver would assign.
        #
        # Block numbering (zero-indexed):
        #   even blocks -> sync  (raw ADC offsets 0, 2B, 4B, ...)
        #   odd  blocks -> target(raw ADC offsets  B, 3B, 5B, ...)
        # -------------------------------------------------------------------
        FH = freq_hop_block
        print(f"[freq_hop mode]  block_size={FH}  ({FH / args.rate * 1000:.1f} ms / block)")
        n_sync_blocks   = (len(sync_iq)   // FH) if sync_iq   is not None else 0
        n_target_blocks = (len(target_iq) // FH) if target_iq is not None else 0
        n_blocks = max(n_sync_blocks * 2, n_target_blocks * 2)

        si = ti = 0  # sync/target block counters
        for blk in range(n_blocks):
            raw_start = blk * FH   # no settling here - files already start at sample 0
            if blk % 2 == 0 and sync_iq is not None and si < n_sync_blocks:
                buf = sync_iq[si * FH:(si + 1) * FH]
                pipeline.process_sync_buffer(buf, raw_start_sample=raw_start)
                si += 1
            elif blk % 2 == 1 and target_iq is not None and ti < n_target_blocks:
                buf = target_iq[ti * FH:(ti + 1) * FH]
                pipeline.process_target_buffer(buf, raw_start_sample=raw_start)
                ti += 1
    else:
        # -------------------------------------------------------------------
        # Normal (same-clock) mode: sync and target are time-aligned IQ from
        # the same ADC stream (single_sdr) or recorded simultaneously.
        # -------------------------------------------------------------------
        for offset in range(0, n_total, B):
            sl = slice(offset, offset + B)
            if sync_iq is not None and offset < len(sync_iq):
                pipeline.process_sync_buffer(sync_iq[sl])
            if target_iq is not None and offset < len(target_iq):
                pipeline.process_target_buffer(target_iq[sl])

    elapsed = time.monotonic() - t_start

    print(f"\n--- Summary ---")
    print(f"Replay time:    {elapsed:.2f} s  ({duration_s / max(elapsed, 0.001):.0f}* realtime)")

    if sync_iq is not None:
        rate = len(sync_events) / max(duration_s, 0.001)
        ok = "[OK]" if rate >= 80 else "LOW - check station/antenna/gain"
        print(f"Sync events:    {len(sync_events)}  ({rate:.1f}/s  target ~100/s)  {ok}")

    if target_iq is not None:
        print(f"Measurements:   {len(measurements)}")
        if not measurements:
            if sync_iq is None:
                print("  (add --sync to pair detections with sync events and get measurements)")
            else:
                print(f"  Try lowering --onset-db (current: {args.onset_db})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
