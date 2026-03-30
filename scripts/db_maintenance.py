#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Beagle Database Maintenance
============================
Periodic housekeeping for the operational and registry SQLite databases.
Intended to be called daily from cron (or systemd timer).

Actions performed
-----------------
Operational DB (tdoa_data.db):
  1. Delete events older than --events-days (default 14)
  2. Delete fixes older than --fixes-days (default 14)
  3. WAL checkpoint (TRUNCATE)

Registry DB (tdoa_registry.db):
  4. Purge expired user sessions
  5. Prune node_config_history, keeping --config-history-keep (default 50)
     most recent versions per node
  6. WAL checkpoint (TRUNCATE)

Both databases are safe to maintain while the server is running - all
operations use normal DELETE + WAL checkpoint, not VACUUM.  If you need
to reclaim disk space after a large prune, stop the server and run
``sqlite3 <db> "VACUUM;"`` manually.

Usage
-----
    # Read DB paths from server config:
    python scripts/db_maintenance.py --server-config config/server.yaml

    # Or specify paths directly:
    python scripts/db_maintenance.py --data-db data/tdoa_data.db \\
                                     --registry-db data/tdoa_registry.db

    # Dry run (report what would be deleted, don't actually delete):
    python scripts/db_maintenance.py --server-config config/server.yaml --dry-run

    # Custom retention:
    python scripts/db_maintenance.py --server-config config/server.yaml \\
        --events-days 7 --fixes-days 30 --config-history-keep 20

Cron example (daily at 03:00)::

    0 3 * * * cd /opt/beagle && /opt/beagle/env/bin/python scripts/db_maintenance.py \\
        --server-config config/server.yaml >> /var/log/tdoa/maintenance.log 2>&1
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> float:
    return time.time()


def _fmt_ts(epoch: float) -> str:
    """Format epoch timestamp as human-readable UTC string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _open_db(path: str) -> sqlite3.Connection:
    p = Path(path)
    if not p.exists():
        print(f"  [skip] database not found: {p}")
        return None  # type: ignore[return-value]
    con = sqlite3.connect(str(p))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row and row[0])


# ---------------------------------------------------------------------------
# Operational DB maintenance
# ---------------------------------------------------------------------------

def prune_events(con: sqlite3.Connection, max_age_days: int, dry_run: bool) -> int:
    """Delete events older than max_age_days. Returns rows deleted."""
    if not _table_exists(con, "events"):
        print("  [skip] events table does not exist")
        return 0

    cutoff = _now_epoch() - max_age_days * 86400
    count = con.execute(
        "SELECT COUNT(*) FROM events WHERE received_at < ?", (cutoff,)
    ).fetchone()[0]

    if count == 0:
        print(f"  events: 0 rows to prune (cutoff {_fmt_ts(cutoff)})")
        return 0

    if dry_run:
        print(f"  events: would delete {count:,} rows older than {_fmt_ts(cutoff)}")
        return 0

    con.execute("DELETE FROM events WHERE received_at < ?", (cutoff,))
    con.commit()
    print(f"  events: deleted {count:,} rows older than {_fmt_ts(cutoff)}")
    return count


def prune_fixes(con: sqlite3.Connection, max_age_days: int, dry_run: bool) -> int:
    """Delete fixes older than max_age_days. Returns rows deleted."""
    if not _table_exists(con, "fixes"):
        print("  [skip] fixes table does not exist")
        return 0

    cutoff = _now_epoch() - max_age_days * 86400
    count = con.execute(
        "SELECT COUNT(*) FROM fixes WHERE computed_at < ?", (cutoff,)
    ).fetchone()[0]

    if count == 0:
        print(f"  fixes: 0 rows to prune (cutoff {_fmt_ts(cutoff)})")
        return 0

    if dry_run:
        print(f"  fixes: would delete {count:,} rows older than {_fmt_ts(cutoff)}")
        return 0

    con.execute("DELETE FROM fixes WHERE computed_at < ?", (cutoff,))
    con.commit()
    print(f"  fixes: deleted {count:,} rows older than {_fmt_ts(cutoff)}")
    return count


def wal_checkpoint(con: sqlite3.Connection, db_label: str) -> None:
    """Run WAL checkpoint to reclaim WAL file space."""
    try:
        result = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result:
            busy, log_pages, checkpointed = result
            if busy:
                print(f"  {db_label} WAL checkpoint: partial (busy={busy}, "
                      f"log={log_pages}, checkpointed={checkpointed})")
            else:
                print(f"  {db_label} WAL checkpoint: ok "
                      f"(log={log_pages}, checkpointed={checkpointed})")
    except sqlite3.OperationalError as exc:
        print(f"  {db_label} WAL checkpoint failed: {exc}")


def report_db_size(path: str, label: str) -> None:
    """Print the size of the database file and its WAL."""
    p = Path(path)
    if not p.exists():
        return
    size_mb = p.stat().st_size / (1024 * 1024)
    wal = Path(str(p) + "-wal")
    wal_mb = wal.stat().st_size / (1024 * 1024) if wal.exists() else 0
    print(f"  {label}: {size_mb:.1f} MB (WAL: {wal_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Registry DB maintenance
# ---------------------------------------------------------------------------

def purge_expired_sessions(con: sqlite3.Connection, dry_run: bool) -> int:
    """Delete expired user sessions. Returns rows deleted."""
    if not _table_exists(con, "user_sessions"):
        print("  [skip] user_sessions table does not exist")
        return 0

    now = _now_epoch()
    count = con.execute(
        "SELECT COUNT(*) FROM user_sessions WHERE expires_at <= ?", (now,)
    ).fetchone()[0]

    if count == 0:
        print("  sessions: 0 expired sessions")
        return 0

    if dry_run:
        print(f"  sessions: would purge {count:,} expired sessions")
        return 0

    con.execute("DELETE FROM user_sessions WHERE expires_at <= ?", (now,))
    con.commit()
    print(f"  sessions: purged {count:,} expired sessions")
    return count


def prune_config_history(
    con: sqlite3.Connection, keep_per_node: int, dry_run: bool
) -> int:
    """Prune node_config_history, keeping the N most recent versions per node."""
    if not _table_exists(con, "node_config_history"):
        print("  [skip] node_config_history table does not exist")
        return 0

    # Find nodes with more than keep_per_node history rows
    rows = con.execute(
        "SELECT node_id, COUNT(*) AS cnt FROM node_config_history "
        "GROUP BY node_id HAVING cnt > ?",
        (keep_per_node,),
    ).fetchall()

    if not rows:
        total = con.execute("SELECT COUNT(*) FROM node_config_history").fetchone()[0]
        print(f"  config_history: {total:,} total rows, all within keep={keep_per_node} limit")
        return 0

    total_deleted = 0
    for node_id, cnt in rows:
        excess = cnt - keep_per_node
        if dry_run:
            print(f"  config_history: would prune {excess:,} old versions for {node_id} "
                  f"(has {cnt}, keeping {keep_per_node})")
        else:
            con.execute(
                "DELETE FROM node_config_history "
                "WHERE node_id = ? AND id NOT IN ("
                "  SELECT id FROM node_config_history "
                "  WHERE node_id = ? ORDER BY version DESC LIMIT ?"
                ")",
                (node_id, node_id, keep_per_node),
            )
            print(f"  config_history: pruned {excess:,} old versions for {node_id}")
        total_deleted += excess

    if not dry_run and total_deleted > 0:
        con.commit()

    return total_deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_paths(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Return (data_db_path, registry_db_path) from CLI args."""
    data_path = args.data_db
    registry_path = args.registry_db

    if args.server_config:
        cfg_path = Path(args.server_config)
        if not cfg_path.exists():
            sys.exit(f"error: server config not found: {cfg_path}")
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            sys.exit("error: PyYAML is required for --server-config (pip install pyyaml)")
        with cfg_path.open() as f:
            raw = yaml.safe_load(f) or {}
        db_cfg = raw.get("database", {})
        if not data_path:
            data_path = db_cfg.get("path", "data/tdoa_data.db")
        if not registry_path:
            registry_path = db_cfg.get("registry_path", "data/tdoa_registry.db")

    return data_path, registry_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beagle database maintenance - prune old data and expired sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --server-config config/server.yaml\n"
            "  %(prog)s --server-config config/server.yaml --dry-run\n"
            "  %(prog)s --data-db data/tdoa_data.db --registry-db data/tdoa_registry.db\n"
            "  %(prog)s --server-config config/server.yaml --events-days 7 --fixes-days 30\n"
        ),
    )

    # DB paths
    parser.add_argument(
        "--server-config", metavar="PATH",
        help="Path to server.yaml; database paths are read from it.",
    )
    parser.add_argument(
        "--data-db", metavar="PATH",
        help="Path to operational DB (overrides server config).",
    )
    parser.add_argument(
        "--registry-db", metavar="PATH",
        help="Path to registry DB (overrides server config).",
    )

    # Retention settings
    parser.add_argument(
        "--events-days", type=int, default=14,
        help="Delete events older than N days (default: 14).",
    )
    parser.add_argument(
        "--fixes-days", type=int, default=14,
        help="Delete fixes older than N days (default: 14).",
    )
    parser.add_argument(
        "--config-history-keep", type=int, default=50,
        help="Keep N most recent config versions per node (default: 50).",
    )

    # Flags
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without actually deleting.",
    )

    args = parser.parse_args()
    data_path, registry_path = _resolve_paths(args)

    if not data_path and not registry_path:
        parser.error("Specify --server-config, or --data-db and/or --registry-db.")

    dry_run = args.dry_run
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== Beagle DB Maintenance - {timestamp} ===")
    if dry_run:
        print("    (DRY RUN - no changes will be made)\n")
    else:
        print()

    # --- Operational DB ---
    if data_path:
        print(f"Operational DB: {data_path}")
        report_db_size(data_path, "size")
        data_con = _open_db(data_path)
        if data_con:
            prune_events(data_con, args.events_days, dry_run)
            prune_fixes(data_con, args.fixes_days, dry_run)
            if not dry_run:
                wal_checkpoint(data_con, "operational")
            data_con.close()
        print()

    # --- Registry DB ---
    if registry_path:
        print(f"Registry DB: {registry_path}")
        report_db_size(registry_path, "size")
        reg_con = _open_db(registry_path)
        if reg_con:
            purge_expired_sessions(reg_con, dry_run)
            prune_config_history(reg_con, args.config_history_keep, dry_run)
            if not dry_run:
                wal_checkpoint(reg_con, "registry")
            reg_con.close()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
