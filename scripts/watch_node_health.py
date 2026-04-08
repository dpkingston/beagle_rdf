#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Live node health watcher.

Polls a beagle-node's /health endpoint every interval seconds and prints a
clean, easy-to-read summary.  Computes derived metrics (sync event rate,
event delta) from successive snapshots.

Usage
-----
  # Local node:
  python3 scripts/watch_node_health.py

  # Remote node:
  python3 scripts/watch_node_health.py --host dpk-tdoa1.local

  # Custom port + interval:
  python3 scripts/watch_node_health.py --host kb7ryy --port 9090 --interval 2

Press Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch a beagle-node's /health endpoint live",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="localhost",
                   help="Node hostname or IP (default: localhost)")
    p.add_argument("--port", type=int, default=8080,
                   help="Health server port (default: 8080)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Poll interval in seconds (default: 1.0)")
    p.add_argument("--no-clear", action="store_true",
                   help="Don't clear the screen between updates (append mode)")
    p.add_argument("--once", action="store_true",
                   help="Print one snapshot and exit")
    p.add_argument("--json", action="store_true",
                   help="Print raw JSON instead of formatted output")
    return p.parse_args()


def fetch_health(host: str, port: int, timeout: float = 3.0) -> dict | None:
    """Fetch /health JSON.  Returns None on connection failure."""
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def colour_status(status: str) -> str:
    if status == "ok":
        return f"{GREEN}{BOLD}{status}{RESET}"
    if status == "starting":
        return f"{YELLOW}{status}{RESET}"
    if status == "degraded":
        return f"{RED}{BOLD}{status}{RESET}"
    return status


def fmt_age(age_s: float | None) -> str:
    """Format an age in seconds with colour based on staleness."""
    if age_s is None:
        return f"{DIM}n/a{RESET}"
    if age_s < 0.5:
        return f"{GREEN}{age_s:.2f}s{RESET}"
    if age_s < 5.0:
        return f"{YELLOW}{age_s:.2f}s{RESET}"
    return f"{RED}{age_s:.1f}s{RESET}"


def fmt_uptime(uptime_s: float) -> str:
    if uptime_s < 60:
        return f"{uptime_s:.0f}s"
    if uptime_s < 3600:
        return f"{uptime_s/60:.1f}m"
    if uptime_s < 86400:
        return f"{uptime_s/3600:.1f}h"
    return f"{uptime_s/86400:.1f}d"


def fmt_freq(hz: float | None) -> str:
    if hz is None:
        return "?"
    return f"{hz/1e6:.4f} MHz"


def fmt_int_with_delta(curr: int, prev: int | None, suffix: str = "") -> str:
    """Format an integer with its delta-since-last-poll if known."""
    if prev is None or curr == prev:
        return f"{curr:>10d}{suffix}"
    delta = curr - prev
    sign = "+" if delta > 0 else ""
    return f"{curr:>10d}{suffix} {DIM}({sign}{delta}){RESET}"


def fmt_ppm(crystal_correction: float) -> str:
    ppm = (crystal_correction - 1.0) * 1e6
    if abs(ppm) < 1.0:
        return f"{GREEN}{ppm:+6.2f} ppm{RESET}"
    if abs(ppm) < 50.0:
        return f"{ppm:+6.1f} ppm"
    return f"{YELLOW}{ppm:+6.1f} ppm{RESET}"


def render(snapshot: dict, prev: dict | None, prev_time: float | None,
           now: float, host: str) -> str:
    """Render a snapshot to a multi-line string."""
    lines: list[str] = []

    # Header
    node_id = snapshot.get("node_id", "?")
    status = snapshot.get("status", "?")
    uptime = fmt_uptime(snapshot.get("uptime_s", 0))
    timestamp = time.strftime("%H:%M:%S")
    lines.append(f"{BOLD}{CYAN}beagle-node health{RESET}  "
                 f"node={BOLD}{node_id}{RESET}  host={host}  "
                 f"status={colour_status(status)}  uptime={uptime}  "
                 f"{DIM}{timestamp}{RESET}")

    if "degraded_reasons" in snapshot:
        for reason in snapshot["degraded_reasons"]:
            lines.append(f"  {RED}-> {reason}{RESET}")

    lines.append("")

    # Configuration block
    sdr_mode = snapshot.get("sdr_mode", "?")
    sample_rate = snapshot.get("sample_rate_hz")
    sync_station = snapshot.get("sync_station", "?")
    sync_freq = snapshot.get("sync_freq_hz")
    rate_str = f"{sample_rate/1e6:.3f} MSps" if sample_rate else "?"
    lines.append(f"{BOLD}Configuration{RESET}")
    lines.append(f"  sdr_mode      {sdr_mode}")
    lines.append(f"  sample_rate   {rate_str}")
    lines.append(f"  sync_station  {sync_station} @ {fmt_freq(sync_freq)}")
    targets = snapshot.get("target_channels") or []
    if targets:
        target_strs = [
            f"{t.get('label', '')} @ {fmt_freq(t.get('frequency_hz'))}"
            for t in targets
        ]
        lines.append(f"  targets       {', '.join(target_strs)}")

    lines.append("")

    # Sync block
    sync_events = snapshot.get("sync_events", 0)
    last_sync_age = snapshot.get("last_sync_age_s")
    crystal = snapshot.get("crystal_correction", 1.0)

    # Compute sync event rate from successive snapshots
    sync_rate_str = f"{DIM}(measuring...){RESET}"
    if prev is not None and prev_time is not None:
        dt = now - prev_time
        if dt > 0:
            d_events = sync_events - prev.get("sync_events", 0)
            rate = d_events / dt
            if rate > 1000:
                sync_rate_str = f"{GREEN}{rate:7.1f}/s{RESET}"
            elif rate > 100:
                sync_rate_str = f"{YELLOW}{rate:7.1f}/s{RESET}"
            elif rate > 0:
                sync_rate_str = f"{RED}{rate:7.1f}/s{RESET}"
            else:
                sync_rate_str = f"{RED}     0/s{RESET}"

    corr_peak = snapshot.get("sync_corr_peak")
    if corr_peak is None:
        corr_str = f"{DIM}n/a{RESET}"
    elif corr_peak >= 0.5:
        corr_str = f"{GREEN}{corr_peak:.4f}{RESET}"
    elif corr_peak >= 0.2:
        corr_str = f"{YELLOW}{corr_peak:.4f}{RESET}"
    else:
        corr_str = f"{RED}{corr_peak:.4f}{RESET}"

    lines.append(f"{BOLD}Sync{RESET}")
    lines.append(f"  events        {fmt_int_with_delta(sync_events, prev.get('sync_events') if prev else None)}")
    lines.append(f"  rate          {sync_rate_str}    {DIM}(RDS expected ~1188/s; pilot ~143/s){RESET}")
    lines.append(f"  last_sync     {fmt_age(last_sync_age)}")
    lines.append(f"  corr_peak     {corr_str}      {DIM}(>0.5 good, >0.2 ok, <0.2 weak){RESET}")
    lines.append(f"  crystal       {fmt_ppm(crystal)}    {DIM}(correction={crystal:.8f}){RESET}")

    lines.append("")

    # Carrier / target events
    last_event_age = snapshot.get("last_event_age_s")
    submitted = snapshot.get("events_submitted", 0)
    dropped = snapshot.get("events_dropped", 0)
    queue = snapshot.get("queue_depth", 0)
    noise_floor = snapshot.get("noise_floor_db")
    onset_db = snapshot.get("onset_threshold_db")
    offset_db = snapshot.get("offset_threshold_db")
    prev_submitted = prev.get("events_submitted") if prev else None
    prev_dropped = prev.get("events_dropped") if prev else None

    lines.append(f"{BOLD}Carrier / Target{RESET}")
    lines.append(f"  last_event    {fmt_age(last_event_age)}")
    lines.append(f"  submitted     {fmt_int_with_delta(submitted, prev_submitted)}")
    drop_str = fmt_int_with_delta(dropped, prev_dropped)
    if dropped > 0:
        drop_str = f"{RED}{drop_str}{RESET}"
    lines.append(f"  dropped       {drop_str}")
    lines.append(f"  queue_depth   {queue:>10d}")
    if noise_floor is not None:
        thr_str = ""
        if onset_db is not None and offset_db is not None:
            thr_str = f"  thresholds: onset={onset_db:.1f} offset={offset_db:.1f} dB"
        lines.append(f"  noise_floor   {noise_floor:>10.1f} dBFS{thr_str}")

    lines.append("")

    # Hardware
    sdr_overflows = snapshot.get("sdr_overflows", 0)
    backlog_drains = snapshot.get("backlog_drains", 0)
    prev_overflows = prev.get("sdr_overflows") if prev else None
    prev_backlog = prev.get("backlog_drains") if prev else None

    lines.append(f"{BOLD}Hardware{RESET}")
    of_str = fmt_int_with_delta(sdr_overflows, prev_overflows)
    if sdr_overflows > 0:
        of_str = f"{YELLOW}{of_str}{RESET}"
    lines.append(f"  sdr_overflows  {of_str}")
    bd_str = fmt_int_with_delta(backlog_drains, prev_backlog)
    if backlog_drains > 0:
        bd_str = f"{YELLOW}{bd_str}{RESET}"
    lines.append(f"  backlog_drains {bd_str}")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    prev_snap: dict | None = None
    prev_time: float | None = None

    try:
        while True:
            now = time.monotonic()
            snap = fetch_health(args.host, args.port)

            if snap is None:
                msg = (f"{RED}ERROR: cannot reach http://{args.host}:{args.port}/health"
                       f"{RESET}")
                if args.no_clear or args.once:
                    print(msg, file=sys.stderr)
                else:
                    sys.stdout.write(CLEAR)
                    sys.stdout.write(msg + "\n")
                    sys.stdout.flush()
                if args.once:
                    return 1
                time.sleep(args.interval)
                continue

            if args.json:
                print(json.dumps(snap, indent=2))
            else:
                output = render(snap, prev_snap, prev_time, now, args.host)
                if not args.no_clear and not args.once:
                    sys.stdout.write(CLEAR)
                sys.stdout.write(output + "\n")
                sys.stdout.flush()

            prev_snap = snap
            prev_time = now

            if args.once:
                return 0

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()  # newline so the prompt isn't on the same line
        return 0


if __name__ == "__main__":
    sys.exit(main())
