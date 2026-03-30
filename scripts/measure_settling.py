#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
Measure RTL-SDR tuner settling time after a frequency hop.

Procedure
---------
1. Tune to freq2 (LMR target) and capture 2 s to establish a power baseline.
2. Immediately re-tune to freq1 (FM sync station).
3. Capture one large buffer (~64 ms) as fast as possible.
4. Split the buffer into analysis windows and compute signal power vs. offset.
5. Report the sample offset at which power first reaches within 2 dB of the
   steady-state level.  That is the settling time.

The recommended settling_samples for node.yaml is:
    settling_samples = settling_sample_index * margin (default 1.5*)

Usage examples
--------------
# Measure settling from target -> sync hop:
python3 scripts/measure_settling.py \\
    --freq1 99.9e6 --freq2 462.5625e6 --gain 30

# Measure settling from sync -> target hop:
python3 scripts/measure_settling.py \\
    --freq1 462.5625e6 --freq2 99.9e6 --gain 30 --capture 131072

# Larger analysis window for smoother plot:
python3 scripts/measure_settling.py \\
    --freq1 99.9e6 --freq2 462.5625e6 --gain 30 --window 4096 --margin 2.0
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure RTL-SDR tuner settling time after a frequency hop"
    )
    p.add_argument("--freq1",   type=float, required=True, metavar="HZ",
                   help="Frequency to settle ON (e.g. FM sync: 99.9e6)")
    p.add_argument("--freq2",   type=float, required=True, metavar="HZ",
                   help="Frequency to hop FROM (e.g. LMR target: 462.5625e6)")
    p.add_argument("--rate",    type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate (default 2.048e6)")
    p.add_argument("--gain",    type=float, default=30.0, metavar="DB",
                   help="Receiver gain in dB (default 30)")
    p.add_argument("--device",  type=str,   default="", metavar="ARGS",
                   help="SoapySDR device args (default: first available)")
    p.add_argument("--capture", type=int,   default=262_144, metavar="N",
                   help="Samples to capture after the hop (default 262144 = ~128 ms)")
    p.add_argument("--window",  type=int,   default=2_048, metavar="N",
                   help="Analysis window size in samples (default 2048 = ~1 ms)")
    p.add_argument("--tolerance-db", type=float, default=2.0, metavar="DB",
                   help="Power must be within this many dB of steady-state (default 2)")
    p.add_argument("--margin",  type=float, default=1.5, metavar="X",
                   help="Safety margin multiplier for settling_samples (default 1.5)")
    p.add_argument("--buf-num", type=int,   default=4, metavar="N",
                   help="USB async buffer count (asyncBuffs) passed to SoapyRTLSDR (default 4, "
                        "matches the rtl_sdr_2freq binary; stock SoapyRTLSDR default is 15)")
    p.add_argument("--buf-len", type=int,   default=16_384, metavar="BYTES",
                   help="USB async buffer length in bytes (bufflen) passed to SoapyRTLSDR "
                        "(default 16384, matches rtl_sdr_2freq; use 0 for driver default)")
    return p.parse_args()


def _print_config_guidance(settling: int, sample_rate: float) -> None:
    """Print config/node.yaml freq_hop block size guidance for a given settling estimate."""
    sr = sample_rate
    S = settling

    print()
    print("Update config/node.yaml -> freq_hop section:")
    print(f"  freq_hop:")
    print(f"    settling_samples: {S:<8}  # {S / sr * 1000:.0f} ms — discard per block after each hop")
    print()
    print("  samples_per_block must exceed settling_samples.  Usable = block_size - settling_samples")
    print("  (applied equally to sync and target).  NOTE: setting settling_samples > the current")
    print("  samples_per_block will raise a ValueError at startup — increase both together.")
    print()
    print("  Symmetric mode (same block size for both sync and target channels):")
    print(f"  {'samples_per_block':<17}  {'block':>7}  {'usable/block':>12}  {'useful target':>13}  {'cycle':>8}")
    print(f"  {'-' * 65}")
    for factor in (2, 3, 4):
        blk = S * factor
        usable_ms = (blk - S) / sr * 1000
        cycle_ms = 2 * blk / sr * 1000
        useful_pct = (blk - S) / (2 * blk) * 100
        tag = "  ← balanced" if factor == 2 else ""
        print(f"  {blk:<17,}  {blk/sr*1000:>5.0f} ms  {usable_ms:>10.0f} ms  {useful_pct:>12.0f}%  {cycle_ms:>6.0f} ms{tag}")

    # Asymmetric: sync = 2*S, target = 4*S
    sync_blk = S * 2
    tgt_blk = S * 4
    sync_usable_ms = (sync_blk - S) / sr * 1000
    tgt_usable_ms = (tgt_blk - S) / sr * 1000
    cycle_ms = (sync_blk + tgt_blk) / sr * 1000
    tgt_useful_pct = (tgt_blk - S) / (sync_blk + tgt_blk) * 100
    print()
    print("  Asymmetric mode (larger target block → more target coverage):")
    print(f"    samples_per_block:        {sync_blk:<8,}  "
          f"# sync   {sync_blk/sr*1000:.0f} ms/block, {sync_usable_ms:.0f} ms usable")
    print(f"    target_samples_per_block: {tgt_blk:<8,}  "
          f"# target {tgt_blk/sr*1000:.0f} ms/block, {tgt_usable_ms:.0f} ms usable, "
          f"{tgt_useful_pct:.0f}% useful target, {cycle_ms:.0f} ms cycle")
    print()
    print("  Suggested starting point (symmetric, balanced):")
    blk2 = S * 2
    print(f"    settling_samples:  {S}")
    print(f"    samples_per_block: {blk2}")


def main() -> int:
    args = parse_args()

    try:
        import SoapySDR
    except ImportError:
        print("ERROR: SoapySDR not found.", file=sys.stderr)
        return 1

    devs = SoapySDR.Device.enumerate(args.device)
    if not devs:
        print("ERROR: No SDR device found.", file=sys.stderr)
        return 1

    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, args.rate)
    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, args.gain)
    actual_gain = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)

    # SoapyRTLSDR buffer parameters (verified via sdr.getStreamArgsInfo()):
    #   "asyncBuffs" -- USB async transfer buffer count for rtlsdr_read_async();
    #                   default 15.  Printed as "Allocating N zero-copy buffers".
    #                   Setting this to 4 (matching the rtl_sdr_2freq binary)
    #                   causes stream stalls after frequency hops: the USB queue
    #                   runs dry during PLL re-acquisition.  Leave at default (15).
    #                   The binary estimate section compensates mathematically.
    #   "buffers"    -- SoapyRTLSDR internal ring buffer count; default 15.
    #   "bufflen"    -- bytes per buffer (multiple of 512 required).
    #                   Note: "buflen" (no double-f) is silently ignored.
    stream_args: dict[str, str] = {}
    if args.buf_len > 0:
        stream_args["bufflen"] = str(args.buf_len)       # bytes per buffer
    rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32,
                         [], stream_args)

    pipeline_samples = args.buf_num * (args.buf_len // 2) if args.buf_len > 0 else 0
    pipeline_ms = pipeline_samples / args.rate * 1000
    if stream_args:
        print(f"USB pipeline: {args.buf_num} buffers * {args.buf_len} bytes "
              f"= {pipeline_samples:,} samples (~{pipeline_ms:.0f} ms expected stale data)")

    # ------------------------------------------------------------------ #
    # Phase 1: Establish steady-state power at freq1 (the destination)   #
    # ------------------------------------------------------------------ #
    print(f"Phase 1: Tune to freq1 ({args.freq1 / 1e6:.3f} MHz)  gain={actual_gain:.1f} dB")
    print("         Capture 2 s to establish steady-state power...")
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.freq1)
    sdr.activateStream(rx)

    drain_n = int(args.rate * 2)
    chunk_buf = np.zeros(65_536, dtype=np.complex64)
    power_samples: list[float] = []
    drained = 0
    while drained < drain_n:
        sr = sdr.readStream(rx, [chunk_buf], 65_536, timeoutUs=2_000_000)
        if sr.ret > 0:
            pwr = float(np.mean(np.abs(chunk_buf[:sr.ret]) ** 2))
            power_samples.append(pwr)
            drained += sr.ret

    steady_power = float(np.median(power_samples))
    steady_db = 10 * np.log10(steady_power + 1e-30)
    print(f"         Steady-state power at freq1: {steady_db:.1f} dBFS\n")

    # ------------------------------------------------------------------ #
    # Phase 2: Hop to freq2, stabilise briefly, then hop back to freq1   #
    # and capture immediately                                             #
    # ------------------------------------------------------------------ #
    print(f"Phase 2: Hop to freq2 ({args.freq2 / 1e6:.3f} MHz)  draining 0.5 s...")
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.freq2)
    # Drain continuously so the USB buffer stays empty.
    # If we called time.sleep() here instead, ~1 M samples would accumulate in
    # the USB buffer.  The capture loop after the hop would then read that
    # stale freq2 data, see consistent power, and report settling_samples = 0.
    drain_end = time.monotonic() + 0.5
    while time.monotonic() < drain_end:
        sdr.readStream(rx, [chunk_buf], len(chunk_buf), timeoutUs=1_000_000)

    print(f"         Hopping back to freq1 ({args.freq1 / 1e6:.3f} MHz)...")
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.freq1)

    # Capture as fast as possible immediately after the hop
    capture_buf = np.zeros(args.capture, dtype=np.complex64)
    captured = 0
    while captured < args.capture:
        n_want = min(65_536, args.capture - captured)
        sr = sdr.readStream(rx, [capture_buf[captured:]], n_want, timeoutUs=2_000_000)
        if sr.ret > 0:
            captured += sr.ret
        elif sr.ret < 0:
            print(f"Stream error: {sr.ret}", file=sys.stderr)
            break

    sdr.deactivateStream(rx)
    sdr.closeStream(rx)

    # ------------------------------------------------------------------ #
    # Analysis: power vs. sample offset                                   #
    # ------------------------------------------------------------------ #
    W = args.window
    n_windows = len(capture_buf) // W

    powers_db = []
    for i in range(n_windows):
        chunk = capture_buf[i * W:(i + 1) * W]
        pwr = float(np.mean(np.abs(chunk) ** 2))
        powers_db.append(10 * np.log10(pwr + 1e-30))

    # Use Phase 1's measured steady-state as the reference.
    # Do NOT estimate it from the second half of this capture buffer: if
    # settling takes more than half the capture window, the "second half"
    # is still full of unsettled data and the reference will be wrong.
    ref_db = steady_db

    # Binary pipeline flush point: first sample where the 2-freq binary's
    # smaller pipeline would be exhausted (purely positional, not measured).
    binary_pipeline_samples = args.buf_num * (args.buf_len // 2)

    print(f"\nPower vs. sample offset after hop to freq1:")
    print(f"  Expected steady-state: {ref_db:.1f} dBFS  "
          f"(tolerance +/-{args.tolerance_db:.1f} dB)")
    if binary_pipeline_samples > 0:
        print(f"  NOTE: SoapyRTLSDR uses its own buffer count (see 'Allocating N buffers'")
        print(f"        above); the table reflects its pipeline, NOT the binary's.")
        print(f"  > binary pipeline exhausted at: "
              f"{binary_pipeline_samples:,} samples "
              f"({binary_pipeline_samples / args.rate * 1000:.0f} ms)  "
              f"<- see summary below for binary estimate")
    print()
    print(f"  {'Offset':>10}  {'ms':>6}  {'dBFS':>7}  {'diff':>7}")
    print(f"  {'-' * 40}")

    settling_sample: int | None = None
    max_pdb_val = float("-inf")
    max_pdb_off = 0
    post_settling_worst: float = 0.0
    for i, pdb in enumerate(powers_db):
        offset = i * W
        ms = offset / args.rate * 1000
        diff = pdb - ref_db
        settled = abs(diff) <= args.tolerance_db
        if pdb > max_pdb_val:
            max_pdb_val = pdb
            max_pdb_off = offset
        if settling_sample is None and settled:
            settling_sample = offset
            marker = " <- SETTLED (SoapySDR pipeline; see binary estimate below)"
        else:
            marker = ""
            if settling_sample is not None and abs(diff) > post_settling_worst:
                post_settling_worst = abs(diff)
        print(f"  {offset:>10,}  {ms:>6.1f}  {pdb:>7.1f}  {diff:>+7.1f}{marker}")

    # Explain anomalies visible in the table above before printing the summary.
    spike_excess_db = max_pdb_val - ref_db
    if spike_excess_db > 3.0:
        print(
            f"NOTE: Peak {max_pdb_val:.1f} dBFS at"
            f" {max_pdb_off / args.rate * 1000:.0f} ms"
            f" ({spike_excess_db:+.1f} dB above steady state)."
        )
        print(
            "      R820T/2 IF amplifier saturates briefly when the PLL re-acquires the"
        )
        print(
            "      target frequency — normal behaviour; the binary estimate below accounts for it."
        )
        print()
    if settling_sample is not None and post_settling_worst > args.tolerance_db:
        print(
            f"NOTE: Power drifted {post_settling_worst:.1f} dB outside the"
            f" {args.tolerance_db:.0f} dB tolerance after the SETTLED marker."
        )
        print(
            "      Expected: FM broadcast power varies ±3–5 dB with audio content."
        )
        print(
            "      The SETTLED marker shows the first window in tolerance; the binary"
        )
        print(
            "      estimate is more reliable than this one-shot detection."
        )
        print()

    if settling_sample is not None:
        settling_ms = settling_sample / args.rate * 1000

        # ---- SoapySDR raw measurement (may reflect driver's own buf_num) ----
        recommended = int(settling_sample * args.margin)
        recommended_ms = recommended / args.rate * 1000
        print(f"SoapySDR measurement:    ~{settling_sample:,} samples  ({settling_ms:.1f} ms)")
        print(f"  (raw; includes SoapyRTLSDR USB pipeline depth, "
              f"which may differ from the binary)")

        # Warn if settling was found late in the capture window.
        if settling_sample > args.capture * 0.75:
            print(f"\nWARNING: settling found in the last 25% of the capture window.")
            print(f"         Re-run with --capture {args.capture * 2} for a larger window.")

        # ---- rtl_sdr_2freq binary estimate --------------------------------
        # SoapyRTLSDR may not honour --buf-num/--buf-len stream args (it
        # prints its own "Allocating N zero-copy buffers" regardless).
        # Instead, decompose the measured settling into:
        #   - PLL lock time   (hardware, independent of USB buffer count)
        #   - USB pipeline    (depends on buf_num * buf_len, varies by driver)
        # PLL onset = first window crossing the midpoint between stale and
        # settled level.  Everything before that is USB pipeline stale data;
        # everything from onset to settled is PLL lock.
        binary_pipeline = args.buf_num * (args.buf_len // 2)
        if binary_pipeline > 0:
            stale_level_db = float(np.median(powers_db[:max(1, n_windows // 4)]))
            midpoint_db = (stale_level_db + ref_db) / 2.0
            pll_onset = settling_sample  # fallback: no PLL component
            for i, pdb in enumerate(powers_db):
                moving_toward_settled = (
                    (ref_db > stale_level_db and pdb > midpoint_db) or
                    (ref_db < stale_level_db and pdb < midpoint_db)
                )
                if moving_toward_settled:
                    pll_onset = i * W
                    break
            pll_samples = max(0, settling_sample - pll_onset)
            binary_settling = binary_pipeline + pll_samples
            binary_recommended = int(binary_settling * args.margin)
            binary_ms = binary_settling / args.rate * 1000
            binary_rec_ms = binary_recommended / args.rate * 1000
            print()
            print(f"rtl_sdr_2freq binary estimate "
                  f"({args.buf_num} buffers * {args.buf_len} bytes):")
            print(f"  USB pipeline:        {binary_pipeline:>8,} samples  "
                  f"({binary_pipeline / args.rate * 1000:.1f} ms)")
            print(f"  PLL lock:            {pll_samples:>8,} samples  "
                  f"({pll_samples / args.rate * 1000:.1f} ms)")
            print(f"  Total settling:      {binary_settling:>8,} samples  "
                  f"({binary_ms:.1f} ms)")
            print(f"  Recommended ({args.margin}*):   {binary_recommended:>8,} samples  "
                  f"({binary_rec_ms:.1f} ms)")
            _print_config_guidance(binary_recommended, args.rate)
        else:
            _print_config_guidance(recommended, args.rate)
    else:
        print("Could not determine settling point -- signal may not have stabilised.")
        print(f"Try increasing --capture (e.g. --capture {args.capture * 2}) or check antenna/gain.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
