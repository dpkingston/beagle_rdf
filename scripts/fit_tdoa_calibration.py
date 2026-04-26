#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Fit per-node TDOA bias offsets (δ_n) against a known-position transmitter.

For each node-pair, computes the bias of the server's PHAT TDOA against the
geometric expected TDOA (from the known transmitter position).  Solves for
per-node δ values such that

    bias(a, b) = δ_a - δ_b

via least-squares over all observed pair biases.  Outputs a
``tdoa_calibration:`` YAML block ready to paste into the server config.

Plateau-only by design: per-event-type biases differ on real hardware
(different code paths through ``compute_tdoa_s``); plateau is the cleanest
event type for fine TDOA, so we calibrate from plateau measurements and
consume the calibration on plateau measurements at fix time.  Onset/offset
events are skipped.

Usage
-----
  # Pull a corpus from the server DB and fit:
  python3 scripts/fit_tdoa_calibration.py \\
      --corpus /path/to/corpus.json \\
      --tx-label "Magnolia" --tx-lat 47.65133 --tx-lon -122.3918318 \\
      --reference-node dpk-tdoa1 \\
      [--output config/tdoa_calibration.yaml]

The corpus is a JSON array of normalized event dicts, one per event, with
keys: node_id, channel_hz, event_type, sync_tx_id, sync_tx_lat, sync_tx_lon,
node_lat, node_lon, onset_time_ns, sync_to_snippet_start_ns, iq_snippet_b64,
channel_sample_rate_hz, transition_start, transition_end.

If the corpus is line-delimited JSON (raw rows from the server DB), pass
``--line-delimited`` and the script will normalize from the raw_json schema.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import numpy as np  # noqa: E402

from beagle_server.tdoa import (  # noqa: E402
    compute_tdoa_s,
    haversine_m,
    reset_sync_calibrator,
)

_C = 299_792_458.0


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a server-DB ``raw_json`` event into the flat dict that
    ``compute_tdoa_s`` expects."""
    out = dict(raw)
    out["channel_hz"] = raw.get("channel_hz") or raw["channel_frequency_hz"]
    if "node_lat" not in out:
        out["node_lat"] = raw["node_location"]["latitude_deg"]
        out["node_lon"] = raw["node_location"]["longitude_deg"]
    if "sync_tx_id" not in out:
        sx = raw["sync_transmitter"]
        out["sync_tx_id"] = sx["station_id"]
        out["sync_tx_lat"] = sx["latitude_deg"]
        out["sync_tx_lon"] = sx["longitude_deg"]
    return out


def load_corpus(path: Path, line_delimited: bool) -> list[dict[str, Any]]:
    if line_delimited:
        events: list[dict[str, Any]] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(_normalize_raw(json.loads(line)))
        return events
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of events in {path}; got {type(data)}")
    return [_normalize_raw(e) for e in data]


# ---------------------------------------------------------------------------
# Pair-grouping (same logic as analyze_3node.py)
# ---------------------------------------------------------------------------

def group_by_transmission(
    events: list[dict[str, Any]],
    onset_window_ns: int = 200_000_000,
) -> list[dict[str, dict[str, Any]]]:
    """Bucket events by (channel, type, sync_tx) within ±onset_window_ns
    of each other, then collapse to one event per node per group.
    Returns a list of dicts {node_id -> event} with len >= 2.
    """
    buckets: dict[tuple, list[list]] = defaultdict(list)
    for ev in events:
        bk = (ev["channel_hz"], ev["event_type"], ev["sync_tx_id"])
        placed = False
        for entry in buckets[bk]:
            if abs(ev["onset_time_ns"] - entry[0]) < onset_window_ns:
                entry[1].append(ev)
                placed = True
                break
        if not placed:
            buckets[bk].append([ev["onset_time_ns"], [ev]])
    multi = []
    for entries in buckets.values():
        for _, evs in entries:
            by: dict[str, dict[str, Any]] = {}
            for ev in evs:
                cur = by.get(ev["node_id"])
                if cur is None or ev["onset_time_ns"] > cur["onset_time_ns"]:
                    by[ev["node_id"]] = ev
            if len(by) >= 2:
                multi.append(by)
    return multi


# ---------------------------------------------------------------------------
# Bias measurement
# ---------------------------------------------------------------------------

def expected_tdoa_ns(
    a: dict[str, Any], b: dict[str, Any], tx_lat: float, tx_lon: float,
) -> float:
    """Geometric TDOA in ns for a known-position transmitter."""
    return (
        haversine_m(tx_lat, tx_lon, a["node_lat"], a["node_lon"])
        - haversine_m(tx_lat, tx_lon, b["node_lat"], b["node_lon"])
    ) / _C * 1e9


def measure_pair_biases(
    multi: list[dict[str, dict[str, Any]]],
    tx_lat: float,
    tx_lon: float,
    event_type: str = "plateau",
    min_xcorr_snr: float = 1.5,
    max_xcorr_baseline_km: float = 30.0,
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    """Returns (errs_per_pair, skip_counts) where ``errs_per_pair`` maps
    sorted-(node_a, node_b) -> list of (measured - expected) in ns.
    """
    reset_sync_calibrator()
    errs: dict[tuple[str, str], list[float]] = defaultdict(list)
    skips: dict[str, int] = defaultdict(int)
    for g in multi:
        ids = sorted(g)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                a, b = g[a_id], g[b_id]
                if a["event_type"] != event_type or b["event_type"] != event_type:
                    skips["wrong_type"] += 1
                    continue
                try:
                    t = compute_tdoa_s(
                        a, b,
                        tdoa_method="phat",
                        min_xcorr_snr=min_xcorr_snr,
                        max_xcorr_baseline_km=max_xcorr_baseline_km,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    skips[f"exc:{type(exc).__name__}"] += 1
                    continue
                if t is None:
                    skips["compute_returned_none"] += 1
                    continue
                err_ns = t * 1e9 - expected_tdoa_ns(a, b, tx_lat, tx_lon)
                errs[(a_id, b_id)].append(err_ns)
    return errs, dict(skips)


# ---------------------------------------------------------------------------
# Per-node δ fit
# ---------------------------------------------------------------------------

def fit_node_offsets(
    pair_biases_ns: dict[tuple[str, str], float],
    reference_node: str,
) -> tuple[dict[str, float], float]:
    """Solve bias(a, b) = δ_a - δ_b by least-squares with δ_reference = 0.

    Returns (offsets_s, residual_rms_us).
    """
    nodes = sorted({n for pair in pair_biases_ns for n in pair})
    if reference_node not in nodes:
        raise ValueError(
            f"reference_node {reference_node!r} not found in observed pairs "
            f"{nodes}"
        )
    free = [n for n in nodes if n != reference_node]
    free_idx = {n: i for i, n in enumerate(free)}

    rows = []
    rhs = []
    for (a, b), bias in pair_biases_ns.items():
        row = np.zeros(len(free))
        if a == reference_node:
            row[free_idx[b]] = -1.0
        elif b == reference_node:
            row[free_idx[a]] = 1.0
        else:
            row[free_idx[a]] = 1.0
            row[free_idx[b]] = -1.0
        rows.append(row)
        rhs.append(bias)

    A = np.array(rows)
    b_vec = np.array(rhs)
    delta_ns, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    pred = A @ delta_ns
    residuals = b_vec - pred
    residual_rms_us = float(np.sqrt(np.mean(residuals ** 2)) / 1e3)

    offsets = {reference_node: 0.0}
    for n in free:
        offsets[n] = float(delta_ns[free_idx[n]] / 1e9)  # convert ns -> s
    return offsets, residual_rms_us


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

def emit_yaml(
    offsets_s: dict[str, float],
    *,
    reference_node: str,
    tx_label: str,
    tx_lat: float,
    tx_lon: float,
    n_pairs: int,
    residual_rms_us: float,
    fit_date: str,
    enable: bool,
) -> str:
    lines = [
        "# Per-node TDOA bias calibration.  Fitted by",
        "# scripts/fit_tdoa_calibration.py against a known-position transmitter.",
        "# δ values are in seconds; positive δ means the node reports later than truth.",
        "tdoa_calibration:",
        f"  enabled: {'true' if enable else 'false'}",
        f"  fit_mode: \"per_node\"",
        f"  reference_node: {reference_node}",
        "  node_offsets_s:",
    ]
    # Sort with reference node first, then alphabetical.
    for n in [reference_node] + sorted(o for o in offsets_s if o != reference_node):
        v = offsets_s[n]
        # Express small offsets in human-friendly form:  +7.928e-06   # +7.928 µs
        if v == 0.0:
            lines.append(f"    {n}: 0.0           # reference")
        else:
            us = v * 1e6
            lines.append(f"    {n}: {v:+.6e}    # {us:+.3f} µs")
    lines.extend([
        f"  fit_transmitter_label: \"{tx_label}\"",
        f"  fit_transmitter_lat: {tx_lat}",
        f"  fit_transmitter_lon: {tx_lon}",
        f"  fit_n_pairs: {n_pairs}",
        f"  fit_residual_rms_us: {residual_rms_us:.3f}",
        f"  fit_date: \"{fit_date}\"",
    ])
    return "\n".join(lines) + "\n"


def emit_yaml_pair(
    pair_biases_ns: dict[tuple[str, str], float],
    pair_n: dict[tuple[str, str], int],
    *,
    tx_label: str,
    tx_lat: float,
    tx_lon: float,
    n_pairs: int,
    fit_date: str,
    enable: bool,
) -> str:
    """Emit per-pair calibration YAML.

    The stored value is the *signed* mean bias of compute_tdoa_s(a, b) -
    geometric_expected(a, b) in seconds, with the pair key in ascending
    sort order ("a,b" with a < b).  At apply time the server looks up
    pair_offsets_s[sorted(a,b)] and subtracts it from the measured TDOA
    (negating when queried in reverse order).  Captures the full
    observable bias structure (clock + cable + multipath-to-target).
    """
    lines = [
        "# Per-pair TDOA bias calibration.  Fitted by",
        "# scripts/fit_tdoa_calibration.py against a known-position transmitter.",
        "# Pair offsets are the empirically-measured biases of",
        "#   compute_tdoa_s(a,b) - geometric_expected(a,b)",
        "# in seconds, with the pair key in ascending sort order.  Apply by",
        "# subtracting from the measured TDOA (negating when queried as (b,a)).",
        "#",
        "# Per-pair calibration captures pair-specific multipath that the",
        "# per-node δ model cannot represent, but is target-specific:  the",
        "# values fitted against transmitter T1 may not generalise to T2 if",
        "# multipath geometry differs significantly between bearings.",
        "tdoa_calibration:",
        f"  enabled: {'true' if enable else 'false'}",
        f"  fit_mode: \"per_pair\"",
        "  pair_offsets_s:",
    ]
    for (a, b) in sorted(pair_biases_ns.keys()):
        # Ensure a < b in stored key (the inputs already come sorted, but
        # be defensive).
        key_a, key_b = (a, b) if a < b else (b, a)
        sign = 1.0 if a < b else -1.0
        value_s = sign * pair_biases_ns[(a, b)] / 1e9
        n_obs = pair_n[(a, b)]
        us = value_s * 1e6
        lines.append(f'    "{key_a},{key_b}": {value_s:+.6e}    # {us:+.3f} µs (N={n_obs})')
    lines.extend([
        f"  fit_transmitter_label: \"{tx_label}\"",
        f"  fit_transmitter_lat: {tx_lat}",
        f"  fit_transmitter_lon: {tx_lon}",
        f"  fit_n_pairs: {n_pairs}",
        f"  fit_residual_rms_us: 0.0   # per-pair fit; residual is per-event noise only",
        f"  fit_date: \"{fit_date}\"",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--corpus", required=True, type=Path,
                   help="JSON corpus (array) or line-delimited raw events.")
    p.add_argument("--line-delimited", action="store_true",
                   help="Corpus is one raw_json row per line (server-DB dump).")
    p.add_argument("--tx-label", required=True,
                   help="Human label for the ground-truth transmitter.")
    p.add_argument("--tx-lat", required=True, type=float,
                   help="Ground-truth transmitter latitude (deg).")
    p.add_argument("--tx-lon", required=True, type=float,
                   help="Ground-truth transmitter longitude (deg).")
    p.add_argument("--reference-node", default=None,
                   help="Node id to anchor at δ = 0 (per-node mode only). "
                        "Required when --mode=per_node.")
    p.add_argument("--mode", choices=("per_node", "per_pair"), default="per_node",
                   help=(
                       "Calibration model.  "
                       "per_node: fit one δ per node by least-squares "
                       "(generalises across transmitters; needs a reference node).  "
                       "per_pair: store the observed bias per pair directly "
                       "(maximally accurate against the fit transmitter; may "
                       "not generalise to other bearings)."
                   ))
    p.add_argument("--event-type", default="plateau",
                   choices=("plateau", "onset", "offset"),
                   help="Event type to fit against (default: plateau, "
                        "recommended).")
    p.add_argument("--min-xcorr-snr", type=float, default=1.5,
                   help="Minimum PHAT SNR for accepting a pair.")
    p.add_argument("--max-baseline-km", type=float, default=30.0,
                   help="Max plausible baseline (km).")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the YAML block to this file (default: stdout).")
    p.add_argument("--enable", action="store_true",
                   help="Set tdoa_calibration.enabled: true in the output.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("fit_tdoa_calibration")

    events = load_corpus(args.corpus, line_delimited=args.line_delimited)
    log.info("Loaded %d events from %s", len(events), args.corpus)

    multi = group_by_transmission(events)
    log.info("Found %d multi-node groups", len(multi))

    err_lists, skips = measure_pair_biases(
        multi,
        tx_lat=args.tx_lat,
        tx_lon=args.tx_lon,
        event_type=args.event_type,
        min_xcorr_snr=args.min_xcorr_snr,
        max_xcorr_baseline_km=args.max_baseline_km,
    )
    if not err_lists:
        print("No pair biases measured.  "
              "Check event-type filter, corpus contents, and SNR threshold.",
              file=sys.stderr)
        if skips:
            print(f"Skips: {skips}", file=sys.stderr)
        return 1

    print("Per-pair bias summary:", file=sys.stderr)
    print(f"  {'pair':50s} {'N':>4s} {'med|err|':>10s} {'std':>9s} "
          f"{'bias_median':>13s} {'bias_mean':>11s}", file=sys.stderr)
    bias_per_pair: dict[tuple[str, str], float] = {}
    n_per_pair: dict[tuple[str, str], int] = {}
    for pair in sorted(err_lists):
        vs = err_lists[pair]
        abs_med = statistics.median([abs(v) for v in vs])
        std = statistics.stdev(vs) if len(vs) > 1 else 0.0
        bias_median = statistics.median(vs)   # robust to PHAT mis-locks
        bias_mean = statistics.mean(vs)       # legacy / for comparison
        # MEDIAN is the calibration value: heavy-tailed pair distributions
        # (sync-period mis-disambiguation, occasional PHAT mis-locks)
        # contaminate the mean but not the median.  Fixes Maple Valley
        # attractor pattern observed 2026-04-25.
        bias_per_pair[pair] = bias_median
        n_per_pair[pair] = len(vs)
        print(f"  {pair[0] + ' <-> ' + pair[1]:50s} {len(vs):>4d} "
              f"{abs_med:>9.0f} {std:>8.0f} "
              f"{bias_median:>+12.0f} {bias_mean:>+10.0f} ns",
              file=sys.stderr)
    if skips:
        print(f"Skips during fit: {skips}", file=sys.stderr)

    n_pairs_total = sum(len(v) for v in err_lists.values())

    if args.mode == "per_pair":
        # Per-pair calibration: store the observed bias per pair directly.
        # Maximally accurate against the fit transmitter; target-specific.
        print("\nPer-pair calibration values (signed bias, sorted-key form):",
              file=sys.stderr)
        for pair in sorted(bias_per_pair):
            us = bias_per_pair[pair] / 1e3
            print(f"  {pair[0] + ',' + pair[1]:50s}  {us:+8.3f} µs  "
                  f"(N={n_per_pair[pair]})", file=sys.stderr)
        yaml_block = emit_yaml_pair(
            bias_per_pair,
            n_per_pair,
            tx_label=args.tx_label,
            tx_lat=args.tx_lat,
            tx_lon=args.tx_lon,
            n_pairs=n_pairs_total,
            fit_date=_dt.date.today().isoformat(),
            enable=args.enable,
        )
    else:
        # Per-node δ via least-squares.  Generalises across transmitters
        # if the bias is genuinely per-node (clock/cable, not multipath).
        if args.reference_node is None:
            print("ERROR: --reference-node is required when --mode=per_node.",
                  file=sys.stderr)
            return 2
        offsets_s, residual_rms_us = fit_node_offsets(
            bias_per_pair, reference_node=args.reference_node,
        )
        print(f"\nFitted per-node δ (relative to {args.reference_node}):",
              file=sys.stderr)
        for n, v in sorted(offsets_s.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {n:25s}  {v * 1e6:+8.3f} µs", file=sys.stderr)
        print(f"Residual RMS: {residual_rms_us:.3f} µs", file=sys.stderr)
        if residual_rms_us > 5.0:
            print("WARNING: residual RMS > 5 µs — per-node δ model may not be "
                  "capturing all bias structure.  Consider --mode=per_pair "
                  "for target-specific accuracy at the cost of generalisation.",
                  file=sys.stderr)

        yaml_block = emit_yaml(
            offsets_s,
            reference_node=args.reference_node,
            tx_label=args.tx_label,
            tx_lat=args.tx_lat,
            tx_lon=args.tx_lon,
            n_pairs=n_pairs_total,
            residual_rms_us=residual_rms_us,
            fit_date=_dt.date.today().isoformat(),
            enable=args.enable,
        )

    if args.output is None:
        print(yaml_block)
    else:
        args.output.write_text(yaml_block)
        print(f"Wrote calibration YAML to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
