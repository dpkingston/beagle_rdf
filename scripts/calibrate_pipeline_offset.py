#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Calibrate pipeline_offset_ns by comparing xcorr TDOA against sync_delta TDOA.

xcorr measures the physical carrier arrival time difference directly from IQ
snippets -- it is completely independent of pipeline_offset_ns.  sync_delta
includes whatever pipeline bias the node has.  The difference between the two
is the pipeline offset error:

    pipeline_offset_error = sync_delta_TDOA - xcorr_TDOA

For a node with the correct pipeline_offset_ns, this error should be zero on
average.  For an uncalibrated node, the mean error IS the offset to apply.

The script queries all event pairs where both xcorr and sync_delta can be
computed for a given target node, and reports the mean offset.

No co-location, no known transmitter location, and no special hardware are
required -- normal traffic is sufficient.

Usage
-----
  python3 scripts/calibrate_pipeline_offset.py \\
      --db data/tdoa_data.db \\
      --node node-discovery \\
      --since 60

  # Restrict to a specific partner node
  python3 scripts/calibrate_pipeline_offset.py \\
      --db data/tdoa_data.db \\
      --node node-discovery --partner node-mapleleaf \\
      --since 60

  # Filter by event type (onset transitions are noisier; offset preferred)
  python3 scripts/calibrate_pipeline_offset.py \\
      --db data/tdoa_data.db \\
      --node node-discovery --event-type offset \\
      --since 60
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from beagle_server.tdoa import (
    _T_SYNC_NS,
    cross_correlate_snippets,
    haversine_m,
    path_delay_correction_ns,
)


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

def _load_events(
    db_path: str,
    node_ids: list[str],
    channel_hz: float | None,
    event_type: str | None,
    since_minutes: float,
    limit: int = 10_000,
) -> dict[str, list[dict[str, Any]]]:
    """Load events with IQ snippets from the aggregation server DB."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(node_ids))
        query = (
            "SELECT node_id, sync_to_snippet_start_ns, sync_tx_lat, sync_tx_lon, "
            "node_lat, node_lon, event_type, onset_time_ns, channel_hz, raw_json "
            f"FROM events WHERE node_id IN ({placeholders})"
        )
        params: list[Any] = list(node_ids)
        if channel_hz is not None:
            query += " AND ABS(channel_hz - ?) < 5000"
            params.append(channel_hz)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if since_minutes > 0:
            cutoff_ns = int((time.time() - since_minutes * 60) * 1e9)
            query += " AND onset_time_ns >= ?"
            params.append(cutoff_ns)
        query += " ORDER BY onset_time_ns ASC LIMIT ?"
        params.append(limit)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    result: dict[str, list[dict[str, Any]]] = {nid: [] for nid in node_ids}
    for row in rows:
        nid = row["node_id"]
        if nid not in result:
            continue
        ev = dict(row)
        raw = ev.pop("raw_json", None)
        if raw:
            try:
                parsed = json.loads(raw)
                ev["iq_snippet_b64"] = parsed.get("iq_snippet_b64")
                ev["channel_sample_rate_hz"] = parsed.get("channel_sample_rate_hz")
            except (json.JSONDecodeError, AttributeError):
                pass
        result[nid].append(ev)
    return result


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------

def _match_events(
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
    window_ns: int = 200_000_000,  # 200 ms default
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match events by onset_time_ns within a window. Greedy nearest-neighbour."""
    if not events_a or not events_b:
        return []

    sorted_b = sorted(events_b, key=lambda e: e["onset_time_ns"])
    b_times = [e["onset_time_ns"] for e in sorted_b]
    used_b: set[int] = set()
    pairs = []

    for ev_a in events_a:
        t_a = ev_a["onset_time_ns"]
        idx = bisect.bisect_left(b_times, t_a)
        best_dist = window_ns + 1
        best_idx = -1
        for candidate in (idx - 1, idx, idx + 1):
            if candidate < 0 or candidate >= len(sorted_b):
                continue
            if candidate in used_b:
                continue
            dist = abs(b_times[candidate] - t_a)
            if dist <= window_ns and dist < best_dist:
                best_dist = dist
                best_idx = candidate
        if best_idx >= 0:
            used_b.add(best_idx)
            pairs.append((ev_a, sorted_b[best_idx]))

    return pairs


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------

def _compute_sync_delta_tdoa_ns(ev_a: dict, ev_b: dict) -> float | None:
    """Compute sync_delta TDOA with path correction and pilot disambiguation."""
    delta_a = ev_a.get("sync_to_snippet_start_ns")
    delta_b = ev_b.get("sync_to_snippet_start_ns")
    if delta_a is None or delta_b is None:
        return None

    raw_ns = float(delta_a) - float(delta_b)

    correction_ns = path_delay_correction_ns(
        sync_tx_lat=ev_a["sync_tx_lat"],
        sync_tx_lon=ev_a["sync_tx_lon"],
        node_a_lat=ev_a["node_lat"],
        node_a_lon=ev_a["node_lon"],
        node_b_lat=ev_b["node_lat"],
        node_b_lon=ev_b["node_lon"],
    )

    corrected_ns = raw_ns + correction_ns

    # Pilot disambiguation: reduce to +/- T_sync/2
    n = round(corrected_ns / _T_SYNC_NS)
    corrected_ns -= n * _T_SYNC_NS

    return corrected_ns


def calibrate(
    db_path: str,
    target_node: str,
    partner_nodes: list[str] | None = None,
    channel_hz: float | None = None,
    event_type: str | None = None,
    since_minutes: float = 60.0,
    min_xcorr_snr: float = 1.3,
    max_xcorr_baseline_km: float = 50.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Compute pipeline_offset_ns calibration for target_node.

    Returns a dict with:
      offset_ns: recommended pipeline_offset_ns (mean of sync_delta - xcorr)
      std_ns: standard deviation of the measurements
      n_pairs: number of usable pairs
      per_partner: dict of partner_node -> {offset_ns, std_ns, n_pairs}
      details: list of per-pair measurements (if verbose)
    """
    # Discover partner nodes if not specified
    if not partner_nodes:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT DISTINCT node_id FROM events WHERE node_id != ?",
                (target_node,),
            ).fetchall()
            partner_nodes = [r[0] for r in rows]
        finally:
            con.close()

    if not partner_nodes:
        return {"offset_ns": None, "std_ns": None, "n_pairs": 0,
                "per_partner": {}, "details": []}

    all_node_ids = [target_node] + partner_nodes
    events_by_node = _load_events(
        db_path, all_node_ids, channel_hz, event_type, since_minutes,
    )

    target_events = events_by_node.get(target_node, [])
    if not target_events:
        print(f"No events found for target node '{target_node}'", file=sys.stderr)
        return {"offset_ns": None, "std_ns": None, "n_pairs": 0,
                "per_partner": {}, "details": []}

    max_lag_ns = max_xcorr_baseline_km * 1_000.0 / 299_792_458.0 * 1e9

    all_offsets: list[float] = []
    per_partner: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    for partner in partner_nodes:
        partner_events = events_by_node.get(partner, [])
        if not partner_events:
            continue

        pairs = _match_events(target_events, partner_events)
        partner_offsets: list[float] = []

        for ev_target, ev_partner in pairs:
            etype = ev_target.get("event_type", "")

            # Both must have IQ snippets for xcorr
            iq_target = ev_target.get("iq_snippet_b64", "")
            iq_partner = ev_partner.get("iq_snippet_b64", "")
            if not iq_target or not iq_partner:
                continue

            # Compute xcorr TDOA
            rate_t = float(ev_target.get("channel_sample_rate_hz", 64_000.0))
            rate_p = float(ev_partner.get("channel_sample_rate_hz", 64_000.0))
            xcorr_lag_ns, xcorr_snr = cross_correlate_snippets(
                iq_target, iq_partner,
                sample_rate_hz_a=rate_t,
                sample_rate_hz_b=rate_p,
                event_type=etype,
            )

            if xcorr_snr < min_xcorr_snr:
                if verbose:
                    details.append({
                        "partner": partner, "type": etype,
                        "status": "xcorr_snr_low",
                        "xcorr_snr": xcorr_snr,
                    })
                continue

            if abs(xcorr_lag_ns) > max_lag_ns:
                if verbose:
                    details.append({
                        "partner": partner, "type": etype,
                        "status": "xcorr_lag_rejected",
                        "xcorr_lag_ns": xcorr_lag_ns,
                    })
                continue

            # Compute sync_delta TDOA
            sd_tdoa_ns = _compute_sync_delta_tdoa_ns(ev_target, ev_partner)
            if sd_tdoa_ns is None:
                continue

            # The offset error: what sync_delta says minus what xcorr says
            offset_err = sd_tdoa_ns - xcorr_lag_ns

            partner_offsets.append(offset_err)
            all_offsets.append(offset_err)

            if verbose:
                details.append({
                    "partner": partner, "type": etype,
                    "status": "ok",
                    "xcorr_ns": xcorr_lag_ns,
                    "xcorr_snr": xcorr_snr,
                    "sync_to_snippet_start_ns": sd_tdoa_ns,
                    "offset_err_ns": offset_err,
                })

        if partner_offsets:
            per_partner[partner] = {
                "offset_ns": statistics.mean(partner_offsets),
                "std_ns": statistics.stdev(partner_offsets) if len(partner_offsets) > 1 else 0.0,
                "n_pairs": len(partner_offsets),
            }

    result: dict[str, Any] = {
        "offset_ns": statistics.mean(all_offsets) if all_offsets else None,
        "std_ns": statistics.stdev(all_offsets) if len(all_offsets) > 1 else 0.0,
        "n_pairs": len(all_offsets),
        "per_partner": per_partner,
    }
    if verbose:
        result["details"] = details
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate pipeline_offset_ns using xcorr vs sync_delta comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", required=True,
                        help="Path to aggregation server SQLite database")
    parser.add_argument("--node", required=True,
                        help="Target node to calibrate")
    parser.add_argument("--partner", action="append", default=None,
                        help="Partner node(s) to pair with (default: all other nodes)")
    parser.add_argument("--channel-hz", type=float, default=None,
                        help="Filter to a specific channel frequency (Hz)")
    parser.add_argument("--event-type", choices=["onset", "offset"], default=None,
                        help="Filter to onset or offset events (default: both)")
    parser.add_argument("--since", type=float, default=60.0,
                        help="Only use events from the last N minutes (default: 60)")
    parser.add_argument("--min-xcorr-snr", type=float, default=1.3,
                        help="Minimum xcorr SNR to accept a pair (default: 1.3)")
    parser.add_argument("--max-baseline-km", type=float, default=50.0,
                        help="Maximum xcorr lag in km-equivalent (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-pair details")

    args = parser.parse_args()

    result = calibrate(
        db_path=args.db,
        target_node=args.node,
        partner_nodes=args.partner,
        channel_hz=args.channel_hz,
        event_type=args.event_type,
        since_minutes=args.since,
        min_xcorr_snr=args.min_xcorr_snr,
        max_xcorr_baseline_km=args.max_baseline_km,
        verbose=args.verbose,
    )

    print(f"\nPipeline offset calibration for '{args.node}'")
    print("=" * 60)

    if result["n_pairs"] == 0:
        print("\nNo usable pairs found. Possible causes:")
        print("  - No events with IQ snippets in the time window")
        print("  - xcorr SNR below threshold for all pairs")
        print("  - No partner nodes with matching events")
        print(f"\nTry: --since {args.since * 2:.0f} or --min-xcorr-snr 1.2")
        sys.exit(1)

    print(f"\n  Total usable pairs: {result['n_pairs']}")
    print(f"  Mean offset error:  {result['offset_ns']:+.0f} ns")
    print(f"  Std deviation:      {result['std_ns']:.0f} ns")

    if result["per_partner"]:
        print(f"\n  Per-partner breakdown:")
        for partner, pdata in sorted(result["per_partner"].items()):
            print(f"    {partner:20s}  N={pdata['n_pairs']:3d}  "
                  f"offset={pdata['offset_ns']:+.0f} ns  "
                  f"std={pdata['std_ns']:.0f} ns")

    print(f"\n  Recommended pipeline_offset_ns: {int(round(result['offset_ns']))}")
    print(f"  (set in node.yaml under the sdr_mode section)\n")

    if args.verbose and "details" in result:
        print("\nPer-pair details:")
        print(f"  {'Partner':20s} {'Type':8s} {'Status':18s} {'xcorr_ns':>10s} "
              f"{'SNR':>6s} {'sd_ns':>10s} {'offset':>10s}")
        print("  " + "-" * 90)
        for d in result["details"]:
            if d["status"] == "ok":
                print(f"  {d['partner']:20s} {d['type']:8s} {d['status']:18s} "
                      f"{d['xcorr_ns']:+10.0f} {d['xcorr_snr']:6.2f} "
                      f"{d['sync_to_snippet_start_ns']:+10.0f} {d['offset_err_ns']:+10.0f}")
            else:
                extra = ""
                if "xcorr_snr" in d:
                    extra = f"SNR={d['xcorr_snr']:.2f}"
                elif "xcorr_lag_ns" in d:
                    extra = f"lag={d['xcorr_lag_ns']:.0f}ns"
                print(f"  {d['partner']:20s} {d['type']:8s} {d['status']:18s} {extra}")


if __name__ == "__main__":
    main()
