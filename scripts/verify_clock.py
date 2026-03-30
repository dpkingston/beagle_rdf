#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Characterise system clock jitter using time.time_ns().

Measures the distribution of successive time.time_ns() call deltas to
quantify scheduling jitter on the current system.  Useful for understanding
how much uncertainty the kernel clock adds to onset_time_ns in CarrierEvents,
and for comparing NTP vs. GPS-disciplined chrony accuracy.

Usage examples
--------------
# Quick 10,000-sample run:
python3 scripts/verify_clock.py

# Longer run with chrony status:
python3 scripts/verify_clock.py --samples 100000 --show-chrony

# Tighter sampling interval (more sensitive to scheduling jitter):
python3 scripts/verify_clock.py --samples 50000 --interval-us 50
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure system clock jitter via time.time_ns()")
    p.add_argument("--samples",     type=int,   default=10_000,  metavar="N",
                   help="Number of time samples (default 10000)")
    p.add_argument("--interval-us", type=float, default=100.0,  metavar="US",
                   help="Target sampling interval in microseconds (default 100)")
    p.add_argument("--show-chrony", action="store_true",
                   help="Also print chronyc tracking output")
    return p.parse_args()


def _percentile(data: list[int], p: float) -> int:
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


def main() -> int:
    args = parse_args()
    interval_ns = int(args.interval_us * 1_000)

    print(f"Collecting {args.samples:,} time.time_ns() samples"
          f" at ~{args.interval_us:.0f} us intervals...")

    samples: list[int] = []
    t_next = time.time_ns()
    for _ in range(args.samples):
        t_next += interval_ns
        # Busy-wait for the target time - avoids sleep jitter
        while time.time_ns() < t_next:
            pass
        samples.append(time.time_ns())

    deltas = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]

    mean_ns = sum(deltas) // len(deltas)
    min_ns  = min(deltas)
    max_ns  = max(deltas)
    p50_ns  = _percentile(deltas, 50)
    p90_ns  = _percentile(deltas, 90)
    p95_ns  = _percentile(deltas, 95)
    p99_ns  = _percentile(deltas, 99)
    jitter  = p99_ns - p50_ns

    print(f"\ntime.time_ns() delta statistics  ({len(deltas):,} intervals):")
    print(f"  Mean:        {mean_ns / 1e3:9.2f} us")
    print(f"  Min:         {min_ns  / 1e3:9.2f} us")
    print(f"  Max:         {max_ns  / 1e3:9.2f} us")
    print(f"  P50:         {p50_ns  / 1e3:9.2f} us")
    print(f"  P90:         {p90_ns  / 1e3:9.2f} us")
    print(f"  P95:         {p95_ns  / 1e3:9.2f} us")
    print(f"  P99:         {p99_ns  / 1e3:9.2f} us")
    print(f"  P99 - P50:   {jitter  / 1e3:9.2f} us  <- scheduling jitter")

    # ------------------------------------------------------------------ #
    # Verdict                                                             #
    # ------------------------------------------------------------------ #
    if jitter < 10_000:
        verdict = "EXCELLENT (<10 us) - suitable for onset_time_ns timestamps"
    elif jitter < 100_000:
        verdict = "GOOD (<100 us) - NTP-class accuracy for event association"
    elif jitter < 1_000_000:
        verdict = "FAIR (<1 ms) - acceptable for event association, not for TDOA"
    else:
        verdict = "POOR (>1 ms) - check system load / consider real-time kernel"

    print(f"\nJitter verdict: {verdict}")

    # Practical implication for onset_time_ns uncertainty
    onset_budget_ns = jitter + 1_000_000  # scheduling jitter + ~1 ms USB buffer latency
    print(f"\nExpected onset_time_ns uncertainty: ~{onset_budget_ns / 1e6:.1f} ms "
          f"(jitter + USB latency estimate)")
    print("  Note: onset_time_ns is used only for event association, not TDOA.  "
          "sync_delta_ns is immune to this jitter.")

    # ------------------------------------------------------------------ #
    # Optional: chrony status                                             #
    # ------------------------------------------------------------------ #
    if args.show_chrony:
        print("\n--- chronyc tracking ---")
        import subprocess
        try:
            r = subprocess.run(
                ["chronyc", "tracking"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                print(r.stdout)
            else:
                print(f"chronyc returned {r.returncode}")
        except FileNotFoundError:
            print("chronyc not found - install chrony for GPS/NTP clock discipline")
        except subprocess.TimeoutExpired:
            print("chronyc timed out")

    return 0


if __name__ == "__main__":
    sys.exit(main())
