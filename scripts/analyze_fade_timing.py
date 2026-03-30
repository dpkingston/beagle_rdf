#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Analyze carrier fade timing from DB offset event snippets.

Answers three questions needed to tune offset detection and ring buffer sizing:

  1. How long does the PA power cutoff take to cross offset_db?
     (= how many samples back in the ring is the true shutoff at detection time)

  2. What is the carrier plateau level relative to offset_db?
     (= is offset_db well-placed or too close to the noise floor)

  3. Is the PA shutoff sharp enough for reliable xcorr?
     (= is the negative derivative peak narrow and dominant)

Works with snippets captured by either the old _encode_combined (centered on
threshold crossing) or the new _encode_offset_snippet (centered on PA shutoff).
For old-style snippets the PA shutoff is in the first half (pre-event samples).

Usage
-----
  python scripts/analyze_fade_timing.py \\
      --db data/tdoa_data.db \\
      --node node-discovery \\
      --sample-rate 62500
"""
from __future__ import annotations

import argparse
import base64
import sqlite3
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _decode_snippet(b64: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.int8)
    iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 127.0
    return iq


def _power_envelope(iq: np.ndarray, smooth: int = 16) -> np.ndarray:
    power = iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2
    return np.convolve(power, np.ones(smooth) / smooth, mode="same")


def _power_db(envelope: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(envelope, 1e-30))


def _analyze_snippet(
    iq: np.ndarray,
    sample_rate_hz: float,
    smooth: int = 16,
) -> dict:
    """
    Extract PA shutoff timing metrics from an offset event snippet.

    Returns a dict with:
      pa_cutoff_sample   : sample index of peak negative derivative
      pa_cutoff_ms       : time from start of snippet to PA cutoff
      cutoff_from_end_samples : samples from PA cutoff to end of snippet
      cutoff_from_end_ms      : same in ms
      plateau_db         : estimated carrier plateau power (dBFS)
      noise_db           : estimated noise floor power (dBFS)
      shutoff_sharpness  : peak |deriv| / RMS of remaining deriv (dimensionless)
      snippet_len        : total number of samples
    """
    n = len(iq)
    env = _power_envelope(iq, smooth)
    db = _power_db(env)
    deriv = np.diff(env)

    # PA cutoff = peak negative derivative (fastest power drop)
    cut_idx = int(np.argmin(deriv))
    cut_ms = cut_idx / sample_rate_hz * 1000.0

    # Sharpness: peak |deriv| vs RMS of the rest
    peak_val = abs(float(deriv[cut_idx]))
    noise_mask = np.ones(len(deriv), dtype=bool)
    noise_mask[max(0, cut_idx - 5): cut_idx + 6] = False
    noise_region = deriv[noise_mask]
    rms_noise = float(np.sqrt(np.mean(noise_region ** 2))) if len(noise_region) else 1e-30
    sharpness = peak_val / max(rms_noise, 1e-30)

    # Plateau: mean power in a 64-sample window well before the cutoff
    plateau_start = max(0, cut_idx - 256)
    plateau_end = max(0, cut_idx - 64)
    if plateau_end > plateau_start:
        plateau_db = float(np.mean(db[plateau_start:plateau_end]))
    else:
        plateau_db = float(np.mean(db[:max(1, cut_idx)]))

    # Noise floor: mean power in the last 128 samples (well after cutoff).
    # If the cutoff is too close to the end of the snippet there is no usable
    # post-cutoff region; return NaN so the caller can flag the snippet.
    noise_start = max(n - 128, cut_idx + 64)
    noise_db = float(np.mean(db[noise_start:])) if noise_start < n else float("nan")

    cutoff_from_end = n - 1 - cut_idx
    cutoff_from_end_ms = cutoff_from_end / sample_rate_hz * 1000.0

    return {
        "pa_cutoff_sample": cut_idx,
        "pa_cutoff_ms": cut_ms,
        "cutoff_from_end_samples": cutoff_from_end,
        "cutoff_from_end_ms": cutoff_from_end_ms,
        "plateau_db": plateau_db,
        "noise_db": noise_db,
        "shutoff_sharpness": sharpness,
        "snippet_len": n,
    }


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

def _load_offset_snippets(db_path: str, node_id: str, limit: int = 50) -> list[dict]:
    import json as _json
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT raw_json FROM events
        WHERE node_id = ? AND event_type = 'offset'
        ORDER BY onset_time_ns DESC
        LIMIT ?
        """,
        (node_id, limit),
    ).fetchall()
    con.close()
    results = []
    for (raw_json,) in rows:
        d = _json.loads(raw_json)
        if d.get("iq_snippet_b64"):
            results.append({
                "onset_time_ns": d.get("onset_time_ns"),
                "channel_hz": d.get("channel_frequency_hz"),
                "iq_snippet_b64": d["iq_snippet_b64"],
                "channel_sample_rate_hz": d.get("channel_sample_rate_hz"),
                "event_type": d.get("event_type"),
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, metavar="PATH", help="tdoa_data.db path")
    ap.add_argument("--node", required=True, metavar="NODE_ID",
                    help="Node ID to analyze (e.g. node-discovery)")
    ap.add_argument("--sample-rate", type=float, default=62500.0, metavar="HZ",
                    help="Target sample rate in Hz (default 62500)")
    ap.add_argument("--limit", type=int, default=20, metavar="N",
                    help="Max snippets to analyze (default 20)")
    ap.add_argument("--smooth", type=int, default=16, metavar="N",
                    help="Power envelope smoothing samples (default 16)")
    args = ap.parse_args()

    rows = _load_offset_snippets(args.db, args.node, args.limit)
    if not rows:
        print(f"No offset snippets found for node '{args.node}' in {args.db}")
        sys.exit(1)

    print(f"\nOffset snippet fade-timing analysis")
    print(f"  Node:        {args.node}")
    print(f"  DB:          {args.db}")
    print(f"  Sample rate: {args.sample_rate:.0f} Hz")
    print(f"  Snippets:    {len(rows)}")
    print()

    results = []
    header = (
        f"{'#':>3}  {'ch_MHz':>9}  {'cut_ms':>6}  {'from_end_ms':>11}  "
        f"{'plateau_dB':>10}  {'noise_dB':>8}  {'sharpness':>9}  {'len':>5}"
    )
    print(header)
    print("-" * len(header))

    for i, row in enumerate(rows):
        b64 = row["iq_snippet_b64"]
        ch_mhz = (row["channel_hz"] or 0) / 1e6
        rate = float(row["channel_sample_rate_hz"] or args.sample_rate)
        iq = _decode_snippet(b64)
        m = _analyze_snippet(iq, rate, args.smooth)
        results.append(m)
        print(
            f"{i+1:>3}  {ch_mhz:>9.4f}  {m['pa_cutoff_ms']:>6.2f}  "
            f"{m['cutoff_from_end_ms']:>11.2f}  "
            f"{m['plateau_db']:>10.1f}  {m['noise_db']:>8.1f}  "
            f"{m['shutoff_sharpness']:>9.1f}  {m['snippet_len']:>5}"
        )

    # Summary statistics
    cuts = [r["cutoff_from_end_ms"] for r in results]
    plateaus = [r["plateau_db"] for r in results]
    noises = [r["noise_db"] for r in results]
    sharpnesses = [r["shutoff_sharpness"] for r in results]

    print()
    print("Summary (PA cutoff distance from END of snippet):")
    print(f"  Mean:  {np.mean(cuts):.2f} ms  ({np.mean(cuts) * args.sample_rate / 1000:.0f} samples)")
    print(f"  P50:   {np.median(cuts):.2f} ms")
    print(f"  P95:   {np.percentile(cuts, 95):.2f} ms  ({np.percentile(cuts, 95) * args.sample_rate / 1000:.0f} samples)")
    print(f"  Max:   {np.max(cuts):.2f} ms  ({np.max(cuts) * args.sample_rate / 1000:.0f} samples)")

    print()
    print("Power levels:")
    print(f"  Plateau: mean {np.mean(plateaus):.1f} dBFS  (range {np.min(plateaus):.1f}-{np.max(plateaus):.1f})")
    print(f"  Noise:   mean {np.mean(noises):.1f} dBFS   (range {np.min(noises):.1f}-{np.max(noises):.1f})")
    print(f"  Margin plateau->noise: {np.mean(plateaus) - np.mean(noises):.1f} dB")

    print()
    print("PA shutoff sharpness (peak |deriv| / RMS rest):")
    print(f"  Mean: {np.mean(sharpnesses):.1f}   P50: {np.median(sharpnesses):.1f}   Min: {np.min(sharpnesses):.1f}")
    print(f"  Values >= 10 indicate a sharp, dominant power cutoff (ideal for xcorr).")

    # Ring buffer sizing recommendation
    p95_samples = np.percentile(cuts, 95) * args.sample_rate / 1000.0
    window = 64  # typical
    min_ring_windows = int(np.ceil(p95_samples / window)) + 5  # 5-window margin
    print()
    print("Ring buffer sizing recommendation:")
    print(f"  PA cutoff is <= {np.percentile(cuts, 95):.1f} ms ({p95_samples:.0f} samples) from end of snippet at P95.")
    print(f"  For old-style snippets (centered on threshold crossing) 'end of snippet'")
    print(f"  ~ detection time.  The ring must hold at least {p95_samples:.0f} samples.")
    print(f"  Recommended ring_lookback_windows >= {min_ring_windows} "
          f"(= {min_ring_windows * window} samples = {min_ring_windows * window / args.sample_rate * 1000:.1f} ms)")

    # Threshold advice
    print()
    print("Threshold advice:")
    mean_plateau = np.mean(plateaus)
    mean_noise = np.mean(noises)
    suggested_offset_db = mean_plateau - 6.0  # 6 dB below plateau
    print(f"  Current plateau ~ {mean_plateau:.1f} dBFS, noise floor ~ {mean_noise:.1f} dBFS")
    print(f"  Raising offset_db to {suggested_offset_db:.1f} dBFS (6 dB below plateau) would")
    print(f"  trigger detection much sooner after PA shutoff, reducing ring buffer requirements")
    print(f"  and keeping detection close to the actual power-cut event.")
    print(f"  Use 'derivative' mode (future feature) for SNR-independent detection.")


if __name__ == "__main__":
    main()
