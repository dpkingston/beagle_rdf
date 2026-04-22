#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Co-located pair test: characterise timing jitter and fix accuracy when two
nodes are placed at the same physical location.

Since both nodes receive the same LMR transmission from the same position,
their true TDOA is exactly 0 (after path-delay correction).  Any measured
deviation is pure timing noise.  This script measures that noise and shows
how it propagates into position error.

Two operating modes
-------------------

--simulate
    Monte Carlo: synthesise N transmission events, add Gaussian noise to each
    node's sync_to_snippet_start_ns, compute TDOA statistics and position error.
    No hardware required.  Useful for pre-deployment threshold selection
    and for comparing SDR types by their expected pilot-extraction sigma.

--db PATH
    Analysis: read real events from the aggregation server SQLite database.
    Matches events from two named co-located nodes by onset_time_ns, then
    computes the same statistics from live data.  Works with any node hardware
    type (RSPduo, RTL-SDR 2freq, SoapySDR dual, etc.) as long as both nodes
    report to the same aggregation server.

Usage examples
--------------
  # Simulation - compare two SDR types by expected sigma
  python3 scripts/colocated_pair_test.py --simulate \\
      --node-a-sigma-us 1.5 --node-b-sigma-us 2.5 \\
      --anchor-sigma-us 1.5 --n-trials 2000

  # Real-data analysis from the aggregation DB
  python3 scripts/colocated_pair_test.py \\
      --db data/tdoa_data.db \\
      --node-a seattle-north-01 --node-b seattle-north-02 \\
      --channel-hz 462562500 --event-type onset

  # With explicit anchors for position-fix error distribution
  python3 scripts/colocated_pair_test.py \\
      --db data/tdoa_data.db \\
      --node-a seattle-north-01 --node-b seattle-north-02 \\
      --anchors seattle-south-01 seattle-east-01 \\
      --channel-hz 462562500
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from beagle_server.solver import solve_fix
from beagle_server.tdoa import (
    _C_M_S,
    _T_SYNC_NS,
    compute_tdoa_s,
    cross_correlate_snippets,
    haversine_m,
    path_delay_correction_ns,
)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of data."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _print_stat_block(label: str, values: list[float], unit: str) -> None:
    if not values:
        print(f"  {label}: no data")
        return
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    p50 = _percentile(values, 50)
    p95 = _percentile(values, 95)
    p99 = _percentile(values, 99)
    print(f"  {label}:")
    print(f"    N={len(values)}  mean={mean:+.1f} {unit}  std={std:.1f} {unit}")
    print(f"    P50={p50:.1f}  P95={p95:.1f}  P99={p99:.1f}  {unit}")


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

_SEATTLE_GEOMETRY = {
    # Two co-located nodes (true TDOA between them is always 0)
    "A": (47.730, -122.300),
    "B": (47.730, -122.300),
    # Two anchor nodes at different positions
    "C": (47.520, -122.300),   # ~23 km south
    "D": (47.620, -122.100),   # ~18 km east
}
_SYNC_TX = (47.625, -122.356)   # approximate KISW 99.9 MHz location
_TRANSMITTER = (47.615, -122.350)  # synthetic LMR transmitter


def _true_sync_delta_ns(
    node_lat: float, node_lon: float,
    tx_lat: float, tx_lon: float,
    sync_lat: float, sync_lon: float,
) -> float:
    """
    Compute the true (noiseless) sync_to_snippet_start_ns for a node.

    sync_to_snippet_start_ns = (dist(tx, node) - dist(sync, node)) / c * 1e9
    """
    d_tx = haversine_m(tx_lat, tx_lon, node_lat, node_lon)
    d_sync = haversine_m(sync_lat, sync_lon, node_lat, node_lon)
    return (d_tx - d_sync) / _C_M_S * 1e9


def _make_event(
    node_id: str,
    node_lat: float,
    node_lon: float,
    sync_to_snippet_start_ns: float,
    sync_tx_lat: float,
    sync_tx_lon: float,
    channel_hz: float = 155_100_000.0,
    event_type: str = "onset",
    corr_peak: float = 0.9,
    onset_time_ns: int = 0,
) -> dict[str, Any]:
    """Build an event dict in the format expected by solve_fix()."""
    return {
        "node_id": node_id,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "sync_to_snippet_start_ns": int(sync_to_snippet_start_ns),
        "sync_tx_lat": sync_tx_lat,
        "sync_tx_lon": sync_tx_lon,
        "channel_hz": channel_hz,
        "event_type": event_type,
        "corr_peak": corr_peak,
        "onset_time_ns": onset_time_ns,
    }


def run_simulate(args: argparse.Namespace) -> None:
    """Monte Carlo simulation of co-located pair timing noise."""
    rng = np.random.default_rng(args.seed)

    sigma_a_ns = args.node_a_sigma_us * 1_000.0
    sigma_b_ns = args.node_b_sigma_us * 1_000.0
    sigma_anc_ns = args.anchor_sigma_us * 1_000.0

    geo = _SEATTLE_GEOMETRY
    sync_tx_lat, sync_tx_lon = _SYNC_TX
    tx_lat, tx_lon = _TRANSMITTER

    # Pre-compute true sync_to_snippet_start_ns for each node position
    true_deltas: dict[str, float] = {}
    sigmas: dict[str, float] = {
        "A": sigma_a_ns,
        "B": sigma_b_ns,
        "C": sigma_anc_ns,
        "D": sigma_anc_ns,
    }
    for nid, (nlat, nlon) in geo.items():
        true_deltas[nid] = _true_sync_delta_ns(nlat, nlon, tx_lat, tx_lon,
                                                sync_tx_lat, sync_tx_lon)

    # Solver search centre: geometric mean of node positions (NOT the true
    # transmitter - using the true answer as the starting point would let
    # the solver converge trivially without exercising the noisy cost surface).
    search_lat = sum(p[0] for p in geo.values()) / len(geo)
    search_lon = sum(p[1] for p in geo.values()) / len(geo)

    print("=" * 60)
    print("Co-located Pair Test - SIMULATION MODE")
    print("=" * 60)
    print(f"Geometry:")
    print(f"  Co-located nodes A,B: ({geo['A'][0]:.4f}, {geo['A'][1]:.4f})")
    print(f"  Anchor C:             ({geo['C'][0]:.4f}, {geo['C'][1]:.4f})")
    print(f"  Anchor D:             ({geo['D'][0]:.4f}, {geo['D'][1]:.4f})")
    print(f"  Sync TX:              ({sync_tx_lat:.4f}, {sync_tx_lon:.4f})")
    print(f"  Transmitter:          ({tx_lat:.4f}, {tx_lon:.4f})")
    baseline_AB = haversine_m(*geo["A"], *geo["C"])
    print(f"  A-to-C baseline:      {baseline_AB/1000:.1f} km")
    print(f"Per-node 1-sigma timing noise:")
    print(f"  Node A: {sigma_a_ns:.0f} ns ({args.node_a_sigma_us:.2f} usec)")
    print(f"  Node B: {sigma_b_ns:.0f} ns ({args.node_b_sigma_us:.2f} usec)")
    print(f"  Anchors C,D: {sigma_anc_ns:.0f} ns ({args.anchor_sigma_us:.2f} usec)")
    print(f"  Expected std(TDOA_AB) = sqrt(2) x sigma_A = "
          f"{math.sqrt(2)*sigma_a_ns:.0f} ns "
          f"(if sigma_A = sigma_B = {sigma_a_ns:.0f} ns)")
    print(f"Trials: {args.n_trials}")
    print()

    tdoa_ab_ns: list[float] = []
    fix_err_A: list[float] = []
    fix_err_B: list[float] = []
    solver_failures = 0

    for _ in range(args.n_trials):
        # Noisy measurements for each node
        noisy: dict[str, float] = {
            nid: true_deltas[nid] + rng.normal(0.0, sigmas[nid])
            for nid in ("A", "B", "C", "D")
        }

        # TDOA between co-located pair (true value = 0, only noise remains)
        ev_a = _make_event("A", *geo["A"], noisy["A"], sync_tx_lat, sync_tx_lon)
        ev_b = _make_event("B", *geo["B"], noisy["B"], sync_tx_lat, sync_tx_lon)
        tdoa_ab_s = compute_tdoa_s(ev_a, ev_b)
        tdoa_ab_ns.append(tdoa_ab_s * 1e9)

        # Position fix using A+C+D and B+C+D independently
        for pair_node, pair_sigma, err_list in [
            ("A", sigma_a_ns, fix_err_A),
            ("B", sigma_b_ns, fix_err_B),
        ]:
            ev_pair = _make_event(pair_node, *geo[pair_node],
                                  noisy[pair_node], sync_tx_lat, sync_tx_lon)
            ev_c = _make_event("C", *geo["C"], noisy["C"], sync_tx_lat, sync_tx_lon)
            ev_d = _make_event("D", *geo["D"], noisy["D"], sync_tx_lat, sync_tx_lon)
            fix = solve_fix(
                events=[ev_pair, ev_c, ev_d],
                search_center_lat=search_lat,
                search_center_lon=search_lon,
                search_radius_km=200.0,
            )
            if fix is not None:
                err_m = haversine_m(fix.latitude_deg, fix.longitude_deg,
                                    tx_lat, tx_lon)
                err_list.append(err_m)
            else:
                if pair_node == "A":
                    solver_failures += 1

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("TDOA_AB distribution (true value = 0):")
    _print_stat_block("TDOA_AB", tdoa_ab_ns, "ns")
    expected_std = math.sqrt(sigma_a_ns**2 + sigma_b_ns**2)
    actual_std = statistics.stdev(tdoa_ab_ns) if len(tdoa_ab_ns) > 1 else 0.0
    print(f"    Expected std (sqrt(sigma_A^2+sigma_B^2)): {expected_std:.1f} ns  "
          f"Actual: {actual_std:.1f} ns")

    print()
    print(f"Position fix error using Node A + anchors C,D  "
          f"(sigma={sigma_a_ns:.0f} ns):")
    _print_stat_block("error", fix_err_A, "m")

    print()
    print(f"Position fix error using Node B + anchors C,D  "
          f"(sigma={sigma_b_ns:.0f} ns):")
    _print_stat_block("error", fix_err_B, "m")

    if solver_failures:
        print(f"\n  Solver failures: {solver_failures} / {args.n_trials}")

    # Scale factor: ns of TDOA error -> metres of position error
    if fix_err_A and tdoa_ab_ns:
        p95_tdoa_ns = _percentile([abs(x) for x in tdoa_ab_ns], 95)
        p95_err_m = _percentile(fix_err_A, 95)
        if p95_tdoa_ns > 0:
            scale = p95_err_m / (p95_tdoa_ns / 1000.0)
            print(f"\n  Geometry scale factor (P95): "
                  f"~{scale:.0f} m per usec of TDOA noise")

    print()


# ---------------------------------------------------------------------------
# DB analysis mode
# ---------------------------------------------------------------------------

def _load_events_from_db(
    db_path: str,
    node_ids: list[str],
    channel_hz: float | None,
    event_type: str | None,
    limit: int = 10_000,
    since_ns: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load events from the aggregation server SQLite DB for the given node IDs.

    Returns a dict mapping node_id -> list of event dicts, ordered by
    onset_time_ns ascending.  IQ snippet fields (iq_snippet_b64 and
    channel_sample_rate_hz) are extracted from the raw_json column so that
    compute_tdoa_s() can use cross-correlation when snippets are present.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(node_ids))
        query = (
            "SELECT node_id, sync_to_snippet_start_ns, sync_tx_lat, sync_tx_lon, "
            "node_lat, node_lon, event_type, onset_time_ns, corr_peak, "
            "channel_hz, raw_json "
            f"FROM events WHERE node_id IN ({placeholders})"
        )
        params: list[Any] = list(node_ids)
        if channel_hz is not None:
            query += " AND ABS(channel_hz - ?) < 5000"
            params.append(channel_hz)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if since_ns is not None:
            query += " AND onset_time_ns >= ?"
            params.append(since_ns)
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
        # Extract IQ snippet fields from raw_json so compute_tdoa_s() can use
        # cross-correlation.  These are not stored as separate DB columns.
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


def _match_events(
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
    window_ns: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Match events from two nodes by onset_time_ns.

    Returns pairs (ev_a, ev_b) where |onset_a - onset_b| <= window_ns.
    Uses a greedy nearest-neighbour approach (events_b sorted by onset_time_ns).
    Each event is used at most once.
    """
    if not events_a or not events_b:
        return []

    sorted_b = sorted(events_b, key=lambda e: e["onset_time_ns"])
    used_b: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    b_idx = 0
    for ev_a in sorted(events_a, key=lambda e: e["onset_time_ns"]):
        t_a = ev_a["onset_time_ns"]

        # Advance b_idx to the first b event that could possibly match
        while b_idx < len(sorted_b) and sorted_b[b_idx]["onset_time_ns"] < t_a - window_ns:
            b_idx += 1

        best_b = None
        best_diff = window_ns + 1
        for k in range(b_idx, len(sorted_b)):
            ev_b = sorted_b[k]
            diff = abs(ev_b["onset_time_ns"] - t_a)
            if diff > window_ns:
                break
            if k not in used_b and diff < best_diff:
                best_diff = diff
                best_b = (k, ev_b)

        if best_b is not None:
            k, ev_b = best_b
            used_b.add(k)
            pairs.append((ev_a, ev_b))

    return pairs


def run_db_analysis(args: argparse.Namespace) -> None:
    """Analyse real events from the aggregation server DB."""
    db_path = args.db
    if not Path(db_path).exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    node_a = args.node_a
    node_b = args.node_b
    anchors: list[str] = args.anchors or []
    all_nodes = [node_a, node_b] + anchors

    channel_hz = args.channel_hz
    window_ns = int(args.window_ms * 1_000_000)

    print("=" * 60)
    print("Co-located Pair Test - DB ANALYSIS MODE")
    print("=" * 60)
    print(f"DB:          {db_path}")
    print(f"Co-located:  {node_a} (A)  {node_b} (B)")
    if anchors:
        print(f"Anchors:     {', '.join(anchors)}")
    if channel_hz:
        print(f"Channel:     {channel_hz / 1e6:.4f} MHz")
    if args.event_type:
        print(f"Event type:  {args.event_type} only")
    since_ns = None
    if args.since:
        since_minutes = float(args.since)
        since_ns = int((time.time() - since_minutes * 60) * 1e9)
        print(f"Since:       last {since_minutes:.0f} minutes")
    print(f"Match window: {args.window_ms:.0f} ms")
    if args.pipeline_offset_a or args.pipeline_offset_b:
        print(f"Pipeline offsets: A={args.pipeline_offset_a:+.0f} ns  B={args.pipeline_offset_b:+.0f} ns")
    print()

    # Load all event types; we split onset/offset ourselves so they are never
    # cross-matched (onset accuracy and offset accuracy can differ).
    events_by_node = _load_events_from_db(
        db_path, all_nodes, channel_hz, event_type=None, since_ns=since_ns
    )

    n_a = len(events_by_node[node_a])
    n_b = len(events_by_node[node_b])
    print(f"Events loaded: {node_a}={n_a}  {node_b}={n_b}", end="")
    for anc in anchors:
        print(f"  {anc}={len(events_by_node[anc])}", end="")
    print()

    if n_a == 0 or n_b == 0:
        print("ERROR: no events found for one or both co-located nodes.", file=sys.stderr)
        if not channel_hz:
            print("Hint: try --channel-hz to filter to a specific frequency.", file=sys.stderr)
        sys.exit(1)

    print()

    # Analyse onset and offset independently - detection accuracy differs
    # between the rising and falling edges of a carrier.
    types_to_analyse = [args.event_type] if args.event_type else ["onset", "offset"]
    all_pairs_ab: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for ev_type in types_to_analyse:
        a_typed = [e for e in events_by_node[node_a] if e["event_type"] == ev_type]
        b_typed = [e for e in events_by_node[node_b] if e["event_type"] == ev_type]
        print(f"--- {ev_type.upper()} ---  {node_a}={len(a_typed)}  {node_b}={len(b_typed)}")
        if not a_typed or not b_typed:
            print(f"  (no {ev_type} events for one or both nodes - skipping)")
            continue
        pairs = _match_events(a_typed, b_typed, window_ns)
        print(f"Matched pairs: {len(pairs)}")
        if not pairs:
            print(f"  No matched pairs - try a larger --window-ms")
            continue
        min_xcorr_snr: float = args.min_xcorr_snr
        tdoa_ab_ns: list[float] = []
        sync_tdoa_ab_ns: list[float] = []
        n_xcorr = 0
        n_skipped = 0
        n_sync_ok = 0
        n_sync_disambig = 0

        # Per-source accumulators: channel_hz -> {tdoa_ns, snr_all, n_xcorr, n_total}
        src_stats: dict[float, dict] = {}

        # Suppress the per-pair INFO log from compute_tdoa_s
        _tdoa_logger = logging.getLogger("beagle_server.tdoa")
        _tdoa_logger_level = _tdoa_logger.level
        _tdoa_logger.setLevel(logging.WARNING)

        for ev_a, ev_b in pairs:
            ch_hz = float(ev_a.get("channel_hz") or 0.0)
            if ch_hz not in src_stats:
                src_stats[ch_hz] = {
                    "tdoa_ns": [], "snr_all": [], "n_xcorr": 0, "n_total": 0,
                }
            src = src_stats[ch_hz]
            src["n_total"] += 1

            snip_a = ev_a.get("iq_snippet_b64")
            snip_b = ev_b.get("iq_snippet_b64")
            snr = float("nan")
            xcorr_lag_ns = float("nan")
            corr_ns = 0.0
            tdoa_ns: float | None = None
            method = "skipped"
            try:
                rate = float(ev_a.get("channel_sample_rate_hz") or 64_000.0)
                xcorr_lag_ns, snr = cross_correlate_snippets(snip_a, snip_b, rate)
                src["snr_all"].append(snr)
                corr_ns = path_delay_correction_ns(
                    ev_a["sync_tx_lat"], ev_a["sync_tx_lon"],
                    ev_a["node_lat"], ev_a["node_lon"],
                    ev_b["node_lat"], ev_b["node_lon"],
                )
                if snr >= min_xcorr_snr:
                    tdoa_ns = xcorr_lag_ns + corr_ns - args.pipeline_offset_a + args.pipeline_offset_b
                    method = "xcorr"
                    n_xcorr += 1
                    src["n_xcorr"] += 1
                else:
                    method = f"skipped (SNR={snr:.1f} < {min_xcorr_snr})"
                    n_skipped += 1
            except Exception as exc:
                method = f"skipped (xcorr error: {exc})"
                n_skipped += 1
            if tdoa_ns is not None:
                tdoa_ab_ns.append(float(tdoa_ns))
                src["tdoa_ns"].append(float(tdoa_ns))

            # --- sync_delta method ---
            # Compute TDOA via sync_delta subtraction + geometric disambiguation.
            # This matches compute_tdoa_s exactly: the disambiguation uses the
            # path delay correction (geometric), NOT the wall-clock onset_time
            # difference.  Wall-clock disambiguation was a hold-over from when
            # T_sync was 7 ms (pilot period); with RDS sync at 842 usec, NTP
            # jitter alone can push n far off and reapply the wall-clock offset.
            delta_a = ev_a.get("sync_to_snippet_start_ns")
            delta_b = ev_b.get("sync_to_snippet_start_ns")
            sync_method = "no sync_delta"
            if delta_a is not None and delta_b is not None:
                raw_ns = float(delta_a) - float(delta_b)
                path_corr_ns = path_delay_correction_ns(
                    ev_a["sync_tx_lat"], ev_a["sync_tx_lon"],
                    ev_a["node_lat"], ev_a["node_lon"],
                    ev_b["node_lat"], ev_b["node_lon"],
                )
                n_periods = round((raw_ns + path_corr_ns) / _T_SYNC_NS)
                if n_periods != 0:
                    raw_ns -= n_periods * _T_SYNC_NS
                    n_sync_disambig += 1
                sync_method = f"sync_delta (n={n_periods:+d})"
                sync_tdoa_ab_ns.append(raw_ns + path_corr_ns)
                n_sync_ok += 1
            # ---

            if args.verbose:
                t_a_ms = ev_a["onset_time_ns"] / 1e6
                snr_str = f" SNR={snr:.1f}" if not math.isnan(snr) else ""
                tdoa_str = f"TDOA={tdoa_ns:+.0f} ns" if tdoa_ns is not None else "TDOA=N/A"
                xcorr_str = (
                    f" xcorr_would_give={xcorr_lag_ns + corr_ns:+.0f} ns"
                    if method.startswith("skipped") and not math.isnan(xcorr_lag_ns)
                    else ""
                )
                sync_str = (
                    f" | sync_delta={sync_tdoa_ab_ns[-1]:+.0f} ns [{sync_method}]"
                    if delta_a is not None and delta_b is not None
                    else " | sync_delta=N/A"
                )
                ch_str = f" ch={ch_hz/1e6:.4f}MHz" if ch_hz else ""
                print(f"  pair t={t_a_ms:.0f} ms{ch_str}  {tdoa_str}  [{method}{snr_str}{xcorr_str}]{sync_str}")

        _tdoa_logger.setLevel(_tdoa_logger_level)

        print(f"  xcorr used: {n_xcorr}/{len(pairs)} pairs  skipped: {n_skipped}")
        print(f"TDOA_AB (true value = 0 for co-located nodes, xcorr pairs only):")
        _print_stat_block("TDOA_AB", tdoa_ab_ns, "ns")

        print(f"\n  sync_delta used: {n_sync_ok}/{len(pairs)} pairs"
              f"  disambiguated: {n_sync_disambig}")
        print(f"TDOA_AB via sync_delta (true value = 0; mean = residual pipeline offset error):")
        _print_stat_block("TDOA_AB", sync_tdoa_ab_ns, "ns")
        if sync_tdoa_ab_ns:
            mean_ns = statistics.mean(sync_tdoa_ab_ns)
            adj = mean_ns / 2
            print(f"  -> Suggested pipeline_offset adjustment: "
                  f"{node_a} {-adj:+.0f} ns  {node_b} {+adj:+.0f} ns")

        # Per-source quality breakdown (always shown when there are matched pairs)
        if src_stats:
            print(f"\nPer-source quality ({len(src_stats)} sources):")
            sorted_sources = sorted(
                src_stats.items(),
                key=lambda kv: kv[1]["n_xcorr"] / max(kv[1]["n_total"], 1),
                reverse=True,
            )
            for ch_hz, s in sorted_sources:
                n_tot = s["n_total"]
                n_ok = s["n_xcorr"]
                xcorr_rate = 100.0 * n_ok / n_tot if n_tot else 0.0
                snr_list = s["snr_all"]
                tdoa_list = s["tdoa_ns"]
                snr_str = ""
                if snr_list:
                    snr_mean = statistics.mean(snr_list)
                    snr_min = min(snr_list)
                    snr_max = max(snr_list)
                    snr_str = f"  SNR mean={snr_mean:.1f} min={snr_min:.1f} max={snr_max:.1f}"
                tdoa_str = ""
                if tdoa_list:
                    tdoa_mean = statistics.mean(tdoa_list)
                    tdoa_std = statistics.stdev(tdoa_list) if len(tdoa_list) > 1 else 0.0
                    tdoa_str = f"  TDOA mean={tdoa_mean:+.0f} ns std={tdoa_std:.0f} ns"
                print(f"  {ch_hz/1e6:.4f} MHz  xcorr={n_ok}/{n_tot} ({xcorr_rate:.0f}%)"
                      f"{snr_str}{tdoa_str}")

        all_pairs_ab.extend(pairs)

    print()
    if not all_pairs_ab:
        print("ERROR: no matched pairs found.  Try a larger --window-ms.", file=sys.stderr)
        sys.exit(1)

    pairs_ab = all_pairs_ab

    # Position fix error (if anchor events available)
    if anchors and len(pairs_ab) > 0:
        print()
        anchor_events = [events_by_node[anc] for anc in anchors]
        if any(len(ae) == 0 for ae in anchor_events):
            print("  (skipping position fix: no events for one or more anchors)")
        else:
            # For each matched A-B pair, try to find matching anchor events
            fix_err_A: list[float] = []
            fix_err_B: list[float] = []

            for ev_a, ev_b in pairs_ab:
                t_ref = (ev_a["onset_time_ns"] + ev_b["onset_time_ns"]) // 2

                # Find the closest anchor event for each anchor node
                anchor_ev_list: list[dict[str, Any]] = []
                ok = True
                for ae_list in anchor_events:
                    closest = min(ae_list,
                                  key=lambda e: abs(e["onset_time_ns"] - t_ref),
                                  default=None)
                    if closest is None or abs(closest["onset_time_ns"] - t_ref) > window_ns:
                        ok = False
                        break
                    anchor_ev_list.append(closest)

                if not ok:
                    continue

                # Solve fix with A + anchors and B + anchors
                sync_tx_lat = ev_a["sync_tx_lat"]
                sync_tx_lon = ev_a["sync_tx_lon"]
                center_lat = ev_a["node_lat"]
                center_lon = ev_a["node_lon"]

                for pair_ev, err_list in [(ev_a, fix_err_A), (ev_b, fix_err_B)]:
                    fix = solve_fix(
                        events=[pair_ev] + anchor_ev_list,
                        search_center_lat=center_lat,
                        search_center_lon=center_lon,
                        search_radius_km=200.0,
                    )
                    if fix is not None:
                        # We don't know the true transmitter position in DB mode,
                        # so report absolute fix position scatter instead
                        err_list.append((fix.latitude_deg, fix.longitude_deg,
                                         fix.residual_ns))

            if fix_err_A and fix_err_B:
                # Compute scatter: distance from median fix position
                def _median_fix(
                    fixes: list[tuple[float, float, float]],
                ) -> tuple[float, float]:
                    lats = [f[0] for f in fixes]
                    lons = [f[1] for f in fixes]
                    return statistics.median(lats), statistics.median(lons)

                med_lat_A, med_lon_A = _median_fix(fix_err_A)
                med_lat_B, med_lon_B = _median_fix(fix_err_B)
                print(f"Position fix scatter using Node A + anchors:")
                scatter_A = [haversine_m(f[0], f[1], med_lat_A, med_lon_A)
                             for f in fix_err_A]
                _print_stat_block("scatter from median", scatter_A, "m")
                print(f"  Median fix: ({med_lat_A:.5f}, {med_lon_A:.5f})")
                res_A = [f[2] for f in fix_err_A]
                _print_stat_block("solver residual", res_A, "ns")

                print()
                print(f"Position fix scatter using Node B + anchors:")
                scatter_B = [haversine_m(f[0], f[1], med_lat_B, med_lon_B)
                             for f in fix_err_B]
                _print_stat_block("scatter from median", scatter_B, "m")
                print(f"  Median fix: ({med_lat_B:.5f}, {med_lon_B:.5f})")
                res_B = [f[2] for f in fix_err_B]
                _print_stat_block("solver residual", res_B, "ns")

                sep_m = haversine_m(med_lat_A, med_lon_A, med_lat_B, med_lon_B)
                print(f"\n  Separation between median(A) and median(B) fixes: "
                      f"{sep_m:.1f} m")
                print(f"  (should be ~0 m for truly co-located nodes)")

    print()


# ---------------------------------------------------------------------------
# Snippet envelope analysis
# ---------------------------------------------------------------------------

def _analyze_snippet_envelope(
    snippet_b64: str,
    event_type: str,
    analysis_window: int = 32,
) -> dict[str, float] | None:
    """
    Decode one IQ snippet and measure its power envelope transition.

    Returns a dict with:
      pre_margin_w   - windows of noise before the transition begins
      transition_w   - windows spanning the carrier rise (onset) or fall (offset)
      post_margin_w  - windows of plateau after the transition completes
      total_w        - total windows in the snippet
      noise_db       - estimated noise floor (dBFS relative to peak)
      plateau_db     - estimated plateau level (dBFS relative to peak)
      clipped_pre    - True if the transition starts at window 0 (no pre context)
      clipped_post   - True if the transition reaches window total_w-1 (no post context)

    Returns None if the snippet is too short or ambiguous (no visible transition).
    """
    try:
        raw = base64.b64decode(snippet_b64)
    except Exception:
        return None
    if len(raw) < analysis_window * 2:
        return None

    # Decode interleaved int8 IQ -> complex float
    arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    n_samples = len(arr) // 2
    iq = arr[0::2] + 1j * arr[1::2]

    # Compute power per analysis window
    n_windows = n_samples // analysis_window
    if n_windows < 4:
        return None
    power_db = np.zeros(n_windows)
    for i in range(n_windows):
        sl = slice(i * analysis_window, (i + 1) * analysis_window)
        avg = float(np.mean(np.abs(iq[sl]) ** 2))
        power_db[i] = 10.0 * np.log10(avg + 1e-30)

    # Estimate noise floor and plateau from the quietest/loudest 20% of windows
    sorted_pw = np.sort(power_db)
    fifth = max(1, n_windows // 5)
    noise_db = float(np.median(sorted_pw[:fifth]))
    plateau_db = float(np.median(sorted_pw[-fifth:]))
    dynamic_range = plateau_db - noise_db
    if dynamic_range < 3.0:
        # No visible transition (flat snippet - all noise or all carrier)
        return None

    # Define transition thresholds at 20% and 80% of the dynamic range
    thresh_lo = noise_db + 0.20 * dynamic_range
    thresh_hi = noise_db + 0.80 * dynamic_range

    if event_type == "onset":
        # Find first window above thresh_lo (rise start) and thresh_hi (rise end)
        rise_start = next((i for i in range(n_windows) if power_db[i] >= thresh_lo), None)
        rise_end = next((i for i in range(n_windows) if power_db[i] >= thresh_hi), None)
        if rise_start is None:
            return None
        if rise_end is None:
            rise_end = n_windows - 1  # clipped post
        pre_margin = rise_start
        transition = max(1, rise_end - rise_start)
        post_margin = n_windows - rise_end - 1
    else:
        # offset: power falls from plateau to noise
        fall_start = next((i for i in range(n_windows) if power_db[i] <= thresh_hi), None)
        fall_end = next((i for i in range(n_windows) if power_db[i] <= thresh_lo), None)
        if fall_start is None:
            return None
        if fall_end is None:
            fall_end = n_windows - 1
        pre_margin = fall_start
        transition = max(1, fall_end - fall_start)
        post_margin = n_windows - fall_end - 1

    return {
        "pre_margin_w": float(pre_margin),
        "transition_w": float(transition),
        "post_margin_w": float(post_margin),
        "total_w": float(n_windows),
        "noise_db": noise_db,
        "plateau_db": plateau_db,
        "clipped_pre": pre_margin == 0,
        "clipped_post": post_margin == 0,
    }


def _load_events_single_node(
    db_path: str,
    node_id: str,
    channel_hz: float | None = None,
    event_type: str | None = None,
    limit: int = 5_000,
) -> list[dict[str, Any]]:
    """Load events from DB for a single node, including IQ snippet fields."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        query = (
            "SELECT node_id, event_type, onset_time_ns, channel_hz, raw_json "
            "FROM events WHERE node_id = ?"
        )
        params: list[Any] = [node_id]
        if channel_hz is not None:
            query += " AND ABS(channel_hz - ?) < 5000"
            params.append(channel_hz)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY onset_time_ns DESC LIMIT ?"
        params.append(limit)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    result = []
    for row in rows:
        ev = dict(row)
        raw = ev.pop("raw_json", None)
        if raw:
            try:
                parsed = json.loads(raw)
                ev["iq_snippet_b64"] = parsed.get("iq_snippet_b64")
                ev["channel_sample_rate_hz"] = parsed.get("channel_sample_rate_hz", 64_000.0)
            except (json.JSONDecodeError, AttributeError):
                pass
        result.append(ev)
    return result


def run_snippet_analysis(args: argparse.Namespace) -> None:
    """
    Analyse IQ snippet envelopes from a single node to determine whether
    snippet_samples and snippet_post_windows are set correctly.

    For each event type x channel combination, reports:
      - How many windows of "silence" precede the transition (pre-margin)
      - How many windows span the carrier rise/fall (transition time)
      - How many windows of "plateau" follow the transition (post-margin)

    Clipped pre: the snippet starts during the transition - the ring buffer is
    too short (pre-event context is missing).

    Clipped post: the snippet ends during the transition or immediately at the
    plateau - snippet_post_windows is too small to capture the full transient.

    Recommended settings are printed at the end.
    """
    db_path: str = args.db
    node_id: str = args.node_a
    analysis_window: int = args.analysis_window
    channel_hz: float | None = args.channel_hz
    event_type_filter: str | None = args.event_type

    print(f"Snippet envelope analysis: node={node_id}  db={db_path}")
    print(f"  Analysis window: {analysis_window} samples")
    print()

    events = _load_events_single_node(
        db_path, node_id, channel_hz=channel_hz, event_type=event_type_filter
    )
    if not events:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    n_with_snippets = sum(1 for e in events if e.get("iq_snippet_b64"))
    print(f"  Loaded {len(events)} events; {n_with_snippets} have IQ snippets.")
    if n_with_snippets == 0:
        print("  No snippets to analyse - ensure the node is running a recent firmware "
              "with iq_snippet_b64 reporting enabled.", file=sys.stderr)
        sys.exit(1)
    print()

    # Group by (event_type, channel_hz)
    groups: dict[tuple[str, float], list[dict]] = {}
    for ev in events:
        if not ev.get("iq_snippet_b64"):
            continue
        key = (ev["event_type"], float(ev["channel_hz"]))
        groups.setdefault(key, []).append(ev)

    # Per-group analysis
    all_recommendations: dict[str, int] = {}  # channel label -> recommended post_windows

    for (etype, ch_hz), evs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        label = f"{ch_hz / 1e6:.4f} MHz {etype}"
        results = []
        n_flat = 0
        for ev in evs:
            r = _analyze_snippet_envelope(
                ev["iq_snippet_b64"], etype, analysis_window=analysis_window
            )
            if r is None:
                n_flat += 1
            else:
                results.append(r)

        if not results:
            print(f"  {label}: no analysable snippets ({n_flat} flat/ambiguous)")
            continue

        n = len(results)
        pre_vals = [r["pre_margin_w"] for r in results]
        trans_vals = [r["transition_w"] for r in results]
        post_vals = [r["post_margin_w"] for r in results]
        total_w = statistics.median([r["total_w"] for r in results])
        n_clipped_pre = sum(1 for r in results if r["clipped_pre"])
        n_clipped_post = sum(1 for r in results if r["clipped_post"])

        pre_med = _percentile(pre_vals, 50)
        trans_med = _percentile(trans_vals, 50)
        trans_p95 = _percentile(trans_vals, 95)
        post_med = _percentile(post_vals, 50)

        # Each analysis window in ms
        sample_rate_hz = evs[0].get("channel_sample_rate_hz") or 64_000.0
        w_ms = 1000.0 * analysis_window / sample_rate_hz

        print(f"  {label}  N={n}  ({n_flat} flat/skipped)")
        print(f"    Snippet: {total_w:.0f} analysis windows x {w_ms:.2f} ms = "
              f"{total_w * w_ms:.1f} ms total")
        print(f"    Pre-margin:  median={pre_med:.1f}w ({pre_med * w_ms:.1f} ms)"
              + (f"  ** {n_clipped_pre}/{n} CLIPPED (ring too short)" if n_clipped_pre else ""))
        print(f"    Transition:  median={trans_med:.1f}w ({trans_med * w_ms:.1f} ms)"
              f"  P95={trans_p95:.1f}w ({trans_p95 * w_ms:.1f} ms)")
        print(f"    Post-margin: median={post_med:.1f}w ({post_med * w_ms:.1f} ms)"
              + (f"  ** {n_clipped_post}/{n} CLIPPED (need more post windows)" if n_clipped_post else ""))

        # Recommendation: post_windows needed so median post_margin >= 3 analysis windows
        # Convert analysis windows back to detector windows (analysis_window / window_samples).
        # We don't know window_samples, so report in analysis windows and ms.
        want_post_ms = max(0.0, (3.0 - post_med) * w_ms)
        # Assuming default detector window of 64 samples at 64 kHz = 1 ms:
        detector_w_ms = 64.0 / sample_rate_hz * 1000.0
        recommended_post = max(0, math.ceil(want_post_ms / detector_w_ms))
        if n_clipped_post > n // 4:
            # More than 25% of events are clipped post - concrete recommendation
            # Use P95 transition + 3 windows buffer, minus current snippet post portion.
            current_post_ms = post_med * w_ms
            need_ms = trans_p95 * w_ms + 3.0 * detector_w_ms - current_post_ms
            recommended_post = max(1, math.ceil(need_ms / detector_w_ms))
            print(f"    -> Recommend snippet_post_windows >= {recommended_post} "
                  f"({recommended_post * detector_w_ms:.1f} ms) to capture P95 transition")
        elif n_clipped_pre > n // 4:
            print(f"    -> Pre-clipping detected: consider increasing snippet_samples "
                  f"(current total={total_w * w_ms:.1f} ms; "
                  f"pre-context needed ~ {(3.0 + trans_p95) * w_ms:.1f} ms)")
        else:
            print(f"    -> Coverage looks adequate (median post-margin {post_med:.1f}w, "
                  f"pre-margin {pre_med:.1f}w)")

        all_recommendations[label] = recommended_post
        print()

    # Summary
    if all_recommendations:
        max_post = max(all_recommendations.values())
        if max_post > 0:
            print(f"Overall recommendation: set snippet_post_windows: {max_post}")
            sample_rate_hz_ref = 64_000.0
            detector_w_ms_ref = 64.0 / sample_rate_hz_ref * 1000.0
            current_snippet_ms = 640.0 / sample_rate_hz_ref * 1000.0
            needed_total_ms = current_snippet_ms + max_post * detector_w_ms_ref
            print(f"  If snippet_samples is still {640} ({current_snippet_ms:.0f} ms), "
                  f"the combined snippet will be ~{needed_total_ms:.0f} ms.")
            if needed_total_ms > 20.0:
                print(f"  Consider also increasing snippet_samples to "
                      f"{int(needed_total_ms / 1000 * sample_rate_hz_ref)} samples "
                      f"({needed_total_ms:.0f} ms) to avoid truncation.")
        else:
            print("Current snippet settings appear adequate for all channels.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Co-located pair test: measure timing jitter and position accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--simulate", action="store_true",
        help="Monte Carlo simulation (no hardware required)",
    )
    mode.add_argument(
        "--db", metavar="PATH",
        help="Aggregation server SQLite DB to analyse",
    )
    mode.add_argument(
        "--analyze-snippets", metavar="PATH",
        help="Analyse IQ snippet envelopes from a single node to calibrate "
             "snippet_post_windows and snippet_samples. Requires --node-a.",
    )

    # Simulation options
    sim = p.add_argument_group("Simulation options (--simulate)")
    sim.add_argument(
        "--node-a-sigma-us", type=float, default=1.5, metavar="US",
        help="Per-node 1-sigma timing noise for node A in usec (default 1.5)",
    )
    sim.add_argument(
        "--node-b-sigma-us", type=float, default=1.5, metavar="US",
        help="Per-node 1-sigma timing noise for node B in usec (default 1.5)",
    )
    sim.add_argument(
        "--anchor-sigma-us", type=float, default=1.5, metavar="US",
        help="Per-node 1-sigma timing noise for anchor nodes in usec (default 1.5)",
    )
    sim.add_argument(
        "--n-trials", type=int, default=1000, metavar="N",
        help="Number of Monte Carlo trials (default 1000)",
    )
    sim.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducibility (default 42)",
    )

    # DB analysis options
    db_grp = p.add_argument_group("DB analysis options (--db)")
    db_grp.add_argument(
        "--node-a", metavar="NODE_ID",
        help="node_id of the first co-located node",
    )
    db_grp.add_argument(
        "--node-b", metavar="NODE_ID",
        help="node_id of the second co-located node",
    )
    db_grp.add_argument(
        "--anchors", nargs="*", metavar="NODE_ID",
        help="Additional anchor node_ids for position-fix error analysis",
    )
    db_grp.add_argument(
        "--channel-hz", type=float, metavar="HZ",
        help="Target channel frequency in Hz (optional filter)",
    )
    db_grp.add_argument(
        "--event-type", choices=["onset", "offset"],
        help="Event type filter (default: both)",
    )
    db_grp.add_argument(
        "--window-ms", type=float, default=500.0, metavar="MS",
        help="Onset-time matching window in ms (default 500 ms for NTP nodes; "
             "use 10 ms for GPS-disciplined nodes)",
    )
    db_grp.add_argument(
        "--min-xcorr-snr", type=float, default=1.5, metavar="SNR",
        help="Minimum power-envelope xcorr peak-to-sidelobe ratio to accept (default 1.5). "
             "Set to 0 to always use xcorr, or higher to force more fallbacks.",
    )
    db_grp.add_argument(
        "--verbose", action="store_true",
        help="Print per-pair TDOA, xcorr SNR, and method used.",
    )
    db_grp.add_argument(
        "--since", type=str, metavar="MINUTES",
        help="Only include events from the last N minutes (e.g. '10').",
    )
    db_grp.add_argument(
        "--pipeline-offset-a", type=float, default=0.0, metavar="NS",
        help="Pipeline delay correction for node-A in ns (subtracted from TDOA_AB). "
             "Set to the mean TDOA_AB measured from a co-located run to zero it out.",
    )
    db_grp.add_argument(
        "--pipeline-offset-b", type=float, default=0.0, metavar="NS",
        help="Pipeline delay correction for node-B in ns (added to TDOA_AB). "
             "Set to the mean TDOA_AB measured from a co-located run to zero it out.",
    )

    # Snippet analysis options
    snip_grp = p.add_argument_group("Snippet analysis options (--analyze-snippets)")
    snip_grp.add_argument(
        "--analysis-window", type=int, default=32, metavar="N",
        help="Samples per analysis window when measuring the power envelope "
             "(default 32; use a divisor of the detector window_samples for "
             "clean alignment). Smaller = finer resolution.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.simulate:
        run_simulate(args)
    elif args.analyze_snippets:
        if not args.node_a:
            parser.error("--analyze-snippets requires --node-a")
        # Expose the DB path via args.db for _load_events_single_node
        args.db = args.analyze_snippets
        run_snippet_analysis(args)
    else:
        if not args.node_a or not args.node_b:
            parser.error("--db mode requires --node-a and --node-b")
        run_db_analysis(args)


if __name__ == "__main__":
    main()
