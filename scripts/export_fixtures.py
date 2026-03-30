#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Export paired measurement fixtures from the aggregation server SQLite database.

For each matched pair of events (one per node, same transmission), the fixture
record contains both nodes' sync_delta_ns, onset_time_ns, location fields, and
IQ snippet.  Only pairs where both nodes report a snippet are included.

The output JSON is compatible with tests/fixtures/real_event_pairs.json and
can be loaded by tests/unit/test_real_data.py.

Usage examples
--------------
  # All available pairs for two nodes, appended to the default fixture file
  python3 scripts/export_fixtures.py \\
      --node-a node-mapleleaf --node-b node-greenlake \\
      --append

  # Specific time window, written to a new file
  python3 scripts/export_fixtures.py \\
      --node-a node-mapleleaf --node-b node-greenlake \\
      --start "2026-03-18T10:00:00" --stop "2026-03-18T11:00:00" \\
      --output tests/fixtures/march18_session.json

  # Filter to one channel and event type
  python3 scripts/export_fixtures.py \\
      --db data/tdoa_data.db \\
      --node-a node-mapleleaf --node-b node-greenlake \\
      --channel-hz 462562500 --event-type onset
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "tests" / "fixtures" / "real_event_pairs.json"
_DEFAULT_WINDOW_MS = 500.0  # onset matching window; 500 ms handles NTP-quality clocks


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

def _load_events(
    db_path: str,
    node_ids: list[str],
    channel_hz: float | None,
    event_type: str | None,
    start_ns: int | None,
    stop_ns: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {node_id: [event_dict, ...]} for the given filters."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(node_ids))
        query = (
            "SELECT node_id, sync_delta_ns, sync_tx_lat, sync_tx_lon, "
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
        if start_ns is not None:
            query += " AND onset_time_ns >= ?"
            params.append(start_ns)
        if stop_ns is not None:
            query += " AND onset_time_ns <= ?"
            params.append(stop_ns)
        query += " ORDER BY onset_time_ns ASC"
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
# Pairing
# ---------------------------------------------------------------------------

def _match_events(
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
    window_ns: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Greedy nearest-neighbour pairing by onset_time_ns."""
    if not events_a or not events_b:
        return []

    sorted_b = sorted(events_b, key=lambda e: e["onset_time_ns"])
    used_b: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    b_idx = 0

    for ev_a in sorted(events_a, key=lambda e: e["onset_time_ns"]):
        t_a = ev_a["onset_time_ns"]
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


def _to_fixture(
    ev_a: dict[str, Any],
    ev_b: dict[str, Any],
    node_a: str,
    node_b: str,
    description: str,
) -> dict[str, Any]:
    return {
        "description": description,
        "node_id_a": node_a,
        "node_id_b": node_b,
        "event_type": ev_a["event_type"],
        "onset_time_ns_a": ev_a["onset_time_ns"],
        "onset_time_ns_b": ev_b["onset_time_ns"],
        "sync_delta_ns_a": ev_a["sync_delta_ns"],
        "sync_delta_ns_b": ev_b["sync_delta_ns"],
        "node_lat_a": ev_a["node_lat"],
        "node_lon_a": ev_a["node_lon"],
        "node_lat_b": ev_b["node_lat"],
        "node_lon_b": ev_b["node_lon"],
        "sync_tx_lat": ev_a["sync_tx_lat"],
        "sync_tx_lon": ev_a["sync_tx_lon"],
        "iq_snippet_b64_a": ev_a["iq_snippet_b64"],
        "iq_snippet_b64_b": ev_b["iq_snippet_b64"],
        "channel_sample_rate_hz": ev_a.get("channel_sample_rate_hz"),
        "channel_hz": ev_a["channel_hz"],
    }


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def _parse_time(s: str) -> int:
    """Parse an ISO 8601 timestamp or a Unix float into nanoseconds since epoch."""
    try:
        # Try Unix timestamp (float seconds)
        return int(float(s) * 1_000_000_000)
    except ValueError:
        pass
    # ISO 8601 (no timezone -> assume local; with Z or +hh:mm -> use as given)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse time '{s}'. Use ISO 8601 (e.g. '2026-03-18T10:00:00') "
            "or a Unix timestamp in seconds."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export paired event fixtures from the aggregation server DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _DEFAULT_DB = Path(__file__).parent.parent / "data" / "tdoa_data.db"
    p.add_argument("--db", metavar="PATH", default=str(_DEFAULT_DB),
                   help=f"Path to the aggregation server SQLite database (default: {_DEFAULT_DB}).")
    p.add_argument("--node-a", required=True, metavar="NODE_ID")
    p.add_argument("--node-b", required=True, metavar="NODE_ID")
    p.add_argument("--start", metavar="TIME",
                   help="Start of time window. ISO 8601 or Unix timestamp in seconds.")
    p.add_argument("--stop", metavar="TIME",
                   help="End of time window. ISO 8601 or Unix timestamp in seconds.")
    p.add_argument("--channel-hz", type=float, metavar="HZ",
                   help="Filter to events within 5 kHz of this channel frequency.")
    p.add_argument("--event-type", choices=["onset", "offset"],
                   help="Export only this event type (default: both).")
    p.add_argument("--window-ms", type=float, default=_DEFAULT_WINDOW_MS, metavar="MS",
                   help=f"Onset-time matching window in ms (default {_DEFAULT_WINDOW_MS:.0f} ms).")
    p.add_argument("--output", metavar="PATH", default=str(_DEFAULT_OUTPUT),
                   help=f"Output JSON file (default: {_DEFAULT_OUTPUT}).")
    p.add_argument("--append", action="store_true",
                   help="Append to the output file instead of overwriting it.")
    args = p.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    start_ns = _parse_time(args.start) if args.start else None
    stop_ns  = _parse_time(args.stop)  if args.stop  else None

    # Human-readable window description for the 'description' field
    def _fmt_ns(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    window_desc = (
        f"{_fmt_ns(start_ns) if start_ns else 'epoch'}"
        f"-{_fmt_ns(stop_ns) if stop_ns else 'now'}"
    )

    print(f"DB:          {args.db}")
    print(f"Nodes:       {args.node_a} (A)  {args.node_b} (B)")
    print(f"Window:      {window_desc}")
    if args.channel_hz:
        print(f"Channel:     {args.channel_hz / 1e6:.4f} MHz")
    if args.event_type:
        print(f"Event type:  {args.event_type} only")

    # Load events
    by_node = _load_events(
        db_path=args.db,
        node_ids=[args.node_a, args.node_b],
        channel_hz=args.channel_hz,
        event_type=args.event_type,
        start_ns=start_ns,
        stop_ns=stop_ns,
    )
    events_a = by_node[args.node_a]
    events_b = by_node[args.node_b]
    print(f"\nEvents loaded:  {len(events_a)} from {args.node_a}, "
          f"{len(events_b)} from {args.node_b}")

    # Match pairs
    window_ns = int(args.window_ms * 1_000_000)
    all_pairs = _match_events(events_a, events_b, window_ns)
    print(f"Matched pairs:  {len(all_pairs)} (within {args.window_ms:.0f} ms window)")

    # Filter to snippet-complete pairs
    complete = [
        (a, b) for a, b in all_pairs
        if a.get("iq_snippet_b64") and b.get("iq_snippet_b64")
    ]
    skipped = len(all_pairs) - len(complete)
    if skipped:
        print(f"Skipped:        {skipped} pairs missing snippet on one or both nodes")
    print(f"Exportable:     {len(complete)} complete pairs")

    if not complete:
        print("Nothing to export.", file=sys.stderr)
        sys.exit(0)

    # Build fixture records
    desc = f"exported {window_desc} {args.node_a}/{args.node_b}"
    fixtures = [_to_fixture(a, b, args.node_a, args.node_b, desc) for a, b in complete]

    # Merge with existing if appending
    output_path = Path(args.output)
    existing: list[dict] = []
    if args.append and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            print(f"\nAppending to existing file ({len(existing)} records)")
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read existing output file: {e}", file=sys.stderr)

    combined = existing + fixtures
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2))

    # Summary by event type
    by_type: dict[str, int] = {}
    for fx in fixtures:
        by_type[fx["event_type"]] = by_type.get(fx["event_type"], 0) + 1
    type_summary = "  ".join(f"{t}: {n}" for t, n in sorted(by_type.items()))

    print(f"\nWrote {len(combined)} total records to {output_path}")
    print(f"  New: {len(fixtures)} ({type_summary})")


if __name__ == "__main__":
    main()
