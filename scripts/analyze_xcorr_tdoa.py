#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Analyze xcorr TDOA measurements from the beagle_node server database.

Groups events from the database by transmission (channel, event_type, sync_tx,
onset_time_ns proximity) and reports the cross-correlation TDOA between each
pair of nodes, in microseconds.

For co-located nodes the true TDOA is zero, so the reported values measure
the combined xcorr noise floor (quantization + FM sync jitter + clock phase
noise between receivers).

Usage
-----
    python scripts/analyze_xcorr_tdoa.py [--db data/tdoa_data.db]
                                         [--min-snr 1.5]
                                         [--max-baseline-km 50]
                                         [--window-ms 200]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Allow running from repo root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beagle_server.tdoa import cross_correlate_snippets, haversine_m

_C_M_S = 299_792_458.0


def _to_db_dict(row: sqlite3.Row) -> dict:
    rj = json.loads(row["raw_json"])
    loc = rj.get("node_location", {})
    tx  = rj.get("sync_transmitter", {})
    sr  = rj.get("channel_sample_rate_hz")
    return {
        "node_id":              row["node_id"],
        "node_lat":             loc.get("latitude_deg"),
        "node_lon":             loc.get("longitude_deg"),
        "sync_tx_id":           tx.get("station_id"),
        "sync_tx_lat":          tx.get("latitude_deg"),
        "sync_tx_lon":          tx.get("longitude_deg"),
        "sync_delta_ns":        row["sync_delta_ns"],
        "onset_time_ns":        row["onset_time_ns"],
        "channel_hz":           row["channel_hz"],
        "event_type":           row["event_type"],
        "corr_peak":            row["corr_peak"],
        "iq_snippet_b64":       rj.get("iq_snippet_b64"),
        "channel_sample_rate_hz": float(sr) if sr is not None else None,
    }


def _group_events(events: list[dict], window_ns: int) -> list[list[dict]]:
    """Group events by (channel_hz, event_type, sync_tx_id) + onset proximity."""
    base_groups: dict = defaultdict(list)
    for ev in events:
        bk = (ev["channel_hz"], ev["event_type"], ev["sync_tx_id"])
        placed = False
        for g in base_groups[bk]:
            if abs(ev["onset_time_ns"] - g[0]) < window_ns:
                g[1].append(ev)
                placed = True
                break
        if not placed:
            base_groups[bk].append([ev["onset_time_ns"], [ev]])

    groups = []
    for bk, group_list in base_groups.items():
        for _anchor, gevs in group_list:
            node_ids = {e["node_id"] for e in gevs}
            if len(node_ids) >= 2:
                groups.append(gevs)
    return groups


def _deduplicate(gevs: list[dict]) -> list[dict]:
    """One event per node - keep highest corr_peak."""
    best: dict = {}
    for ev in gevs:
        nid = ev["node_id"]
        if nid not in best or ev["corr_peak"] > best[nid]["corr_peak"]:
            best[nid] = ev
    return list(best.values())


def _xcorr_pair(
    a: dict,
    b: dict,
    min_snr: float,
    max_baseline_km: float,
) -> tuple[float | None, float | None, str]:
    """
    Compute xcorr TDOA between two events.

    Returns (xcorr_us, snr, status_string).
    xcorr_us is None when the pair is skipped.
    """
    sa = a.get("iq_snippet_b64")
    sb = b.get("iq_snippet_b64")
    if not sa or not sb:
        return None, None, "NO_SNIPPET"

    rate_a = float(a.get("channel_sample_rate_hz") or 64_000.0)
    rate_b = float(b.get("channel_sample_rate_hz") or 64_000.0)
    etype  = a.get("event_type", "")

    try:
        xcorr_ns, snr = cross_correlate_snippets(
            sa, sb,
            sample_rate_hz_a=rate_a,
            sample_rate_hz_b=rate_b,
            target_rate_hz=None,
            event_type=etype,
        )
    except Exception as exc:
        return None, None, f"XCORR_ERR({exc})"

    if snr < min_snr:
        return None, snr, f"LOW_SNR({snr:.2f})"

    max_lag_ns = max_baseline_km * 1e3 / _C_M_S * 1e9
    if max_baseline_km > 0.0 and abs(xcorr_ns) > max_lag_ns:
        return None, snr, f"BASELINE_REJ({xcorr_ns/1000:.1f} usec, {snr:.1f} SNR)"

    return xcorr_ns / 1_000.0, snr, "OK"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",              default="data/tdoa_data.db", help="Path to SQLite database")
    ap.add_argument("--min-snr",         type=float, default=1.5,  help="Minimum xcorr SNR (default 1.5)")
    ap.add_argument("--max-baseline-km", type=float, default=50.0, help="Max plausible lag in km-equivalent (default 50)")
    ap.add_argument("--window-ms",       type=float, default=200,  help="Grouping window in ms (default 200)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT node_id, event_type, channel_hz, sync_tx_id, "
        "       onset_time_ns, sync_delta_ns, corr_peak, raw_json "
        "FROM events ORDER BY onset_time_ns"
    ).fetchall()

    if not rows:
        print("No events in database.")
        return

    events = [_to_db_dict(r) for r in rows]
    window_ns = int(args.window_ms * 1_000_000)
    groups = _group_events(events, window_ns)

    node_ids_all = sorted({e["node_id"] for e in events})
    channels = sorted({e["channel_hz"] for e in events})

    print(f"Database : {db_path}")
    print(f"Events   : {len(events)}  (nodes: {', '.join(node_ids_all)})")
    print(f"Channels : {', '.join(f'{c/1e6:.4f} MHz' for c in channels)}")
    print(f"Groups   : {len(groups)} multi-node")
    print(f"Settings : min_snr={args.min_snr}  max_baseline={args.max_baseline_km} km  window={args.window_ms} ms")
    print()

    # Header
    w = 12
    print(f"{'Type':6}  {'Nodes':<35}  {'XCorr (usec)':>12}  {'SNR':>6}  Status")
    print("-" * 80)

    ok_us: dict[str, list[float]] = {"onset": [], "offset": []}
    ok_snr: dict[str, list[float]] = {"onset": [], "offset": []}
    counts = {"ok": 0, "low_snr": 0, "baseline": 0, "no_snippet": 0, "xcorr_err": 0}

    for gevs in sorted(groups, key=lambda g: g[0]["onset_time_ns"]):
        node_events = _deduplicate(gevs)
        # All pairs within the group
        for i in range(len(node_events)):
            for j in range(i + 1, len(node_events)):
                a, b = node_events[i], node_events[j]
                etype = a["event_type"]
                pair_label = f"{a['node_id']}<->{b['node_id']}"

                xcorr_us, snr, status = _xcorr_pair(a, b, args.min_snr, args.max_baseline_km)

                snr_str = f"{snr:.2f}" if snr is not None else "N/A"
                us_str  = f"{xcorr_us:+.3f}" if xcorr_us is not None else "--"

                print(f"{etype:6}  {pair_label:<35}  {us_str:>12}  {snr_str:>6}  {status}")

                if status == "OK":
                    counts["ok"] += 1
                    ok_us[etype].append(xcorr_us)
                    ok_snr[etype].append(snr)
                elif "LOW_SNR" in status:
                    counts["low_snr"] += 1
                elif "BASELINE" in status:
                    counts["baseline"] += 1
                elif "NO_SNIPPET" in status:
                    counts["no_snippet"] += 1
                else:
                    counts["xcorr_err"] += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  OK: {counts['ok']}  low_SNR: {counts['low_snr']}  "
          f"baseline_rej: {counts['baseline']}  no_snippet: {counts['no_snippet']}")

    sample_rate = None
    for e in events:
        sr = e.get("channel_sample_rate_hz")
        if sr:
            sample_rate = sr
            break
    if sample_rate:
        print(f"  IQ sample rate: {sample_rate:.0f} Hz  "
              f"(1 sample = {1e6/sample_rate:.2f} usec; sub-sample via parabolic interp)")

    print()
    all_us = []
    for etype in ("onset", "offset"):
        vals = ok_us[etype]
        snrs = ok_snr[etype]
        if not vals:
            print(f"  {etype}: no valid results")
            continue
        med = statistics.median(vals)
        mn  = statistics.mean(vals)
        sd  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {etype} (n={len(vals)}):  "
              f"mean={mn:+.3f} usec  median={med:+.3f} usec  stdev={sd:.3f} usec  "
              f"range=[{min(vals):+.3f}, {max(vals):+.3f}] usec")
        print(f"    SNR:  mean={statistics.mean(snrs):.2f}  "
              f"min={min(snrs):.2f}  max={max(snrs):.2f}")
        all_us.extend(vals)

    if len(all_us) > 1:
        print()
        print(f"  All combined (n={len(all_us)}):  "
              f"mean={statistics.mean(all_us):+.3f} usec  "
              f"median={statistics.median(all_us):+.3f} usec  "
              f"stdev={statistics.stdev(all_us):.3f} usec")


if __name__ == "__main__":
    main()
