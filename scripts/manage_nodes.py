#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Beagle Node Config Manager
===========================
CLI tool for managing node records and their server-assigned configs in the
Beagle SQLite database.  Used to pre-register nodes before they can authenticate
with the server, and as a permanent low-level escape hatch for admins who prefer
a shell over the web UI or REST API.

Typical workflow
----------------
1. Register a new node and generate its secret::

       manage_nodes.py --db data/tdoa_registry.db add seattle-north-01 \\
           --label "Seattle North"

   Copy the printed secret to the node's bootstrap.yaml immediately.

2. Assign a full operating config to the node::

       manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \\
           --config-file configs/seattle-north-01.yaml

3. Start the node in bootstrap mode (on the node itself)::

       beagle-node --bootstrap /etc/beagle/bootstrap.yaml

   The node fetches its full config from the server using the secret.

Database tables
---------------
This script creates the ``nodes`` and ``node_config_history`` tables if they
do not already exist (idempotent).  The aggregation server (beagle-server) also
creates these tables at startup, so running this script against a live server
database is safe - they will not conflict.

Usage
-----
    # Point directly at the database:
    python scripts/manage_nodes.py --db data/tdoa_registry.db list
    python scripts/manage_nodes.py --db data/tdoa_registry.db add seattle-north-01 \\
        --label "Seattle North Roof"
    python scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \\
        --config-file configs/seattle-north-01.yaml
    python scripts/manage_nodes.py --db data/tdoa_registry.db show seattle-north-01
    python scripts/manage_nodes.py --db data/tdoa_registry.db enable seattle-north-01
    python scripts/manage_nodes.py --db data/tdoa_registry.db disable seattle-north-01
    python scripts/manage_nodes.py --db data/tdoa_registry.db remove seattle-north-01
    python scripts/manage_nodes.py --db data/tdoa_registry.db regen-secret seattle-north-01

    # Read DB path from the server config YAML:
    python scripts/manage_nodes.py --server-config config/server.yaml list

Commands
--------
    list            Print a summary table of all registered nodes.
    add NODE_ID     Register a new node.  Generates and prints a secret the
                    node operator must copy into bootstrap.yaml.
    show NODE_ID    Print full details and current config for a node.
                    Pass --merged to apply the node's frequency-group overlay
                    and print exactly what the server will hand to the node on
                    its next config fetch.
    set-config NODE_ID
                    Replace (or clear) the server-assigned config JSON for a node.
                    Accepts --config-file (YAML or JSON) or --config-json string.
                    Passing neither clears the config (node falls back to server defaults).
    enable NODE_ID  Mark a node as enabled (default).
    disable NODE_ID Mark a node as disabled; server will reject its events.
    remove NODE_ID  Delete the node record entirely.
    regen-secret NODE_ID
                    Generate and store a new secret.  The old secret stops working
                    immediately.  Prints the new plaintext secret for the operator.

    group-list      List all frequency groups.
    group-add GROUP_ID
                    Create a new frequency group with sync station and target channels.
    group-show GROUP_ID
                    Show full details for a group including member nodes.
    group-remove GROUP_ID
                    Delete a group.  Member nodes become ungrouped.
    group-set-node NODE_ID
                    Assign a node to a group (--group GID) or unassign it (no --group).

Node secrets
------------
Secrets are stored as SHA-256 hashes (hex-encoded, prefixed with "sha256:").
This is a placeholder until the full authentication system (which will use bcrypt
via passlib) is implemented.  When that migration happens, existing hashes will be
re-hashed on first successful login.

The plaintext secret is printed ONCE at add/regen-secret time and never stored.
The node operator must copy it into the node's bootstrap.yaml immediately.

Config files
-----------
--config-file accepts either YAML or JSON.  The file content is stored as-is
(after round-tripping through the parser to validate syntax) as the node's
config_json blob.  The server merges this with any assigned template config
when responding to a node's config fetch request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema (subset - only the tables this script manages)
# ---------------------------------------------------------------------------

_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id            TEXT    PRIMARY KEY,
    secret_hash        TEXT    NOT NULL,
    label              TEXT,
    registered_at      REAL    NOT NULL,
    last_seen_at       REAL,
    last_ip            TEXT,
    enabled            INTEGER NOT NULL DEFAULT 1,
    config_version     INTEGER NOT NULL DEFAULT 0,
    config_json        TEXT,
    config_template_id TEXT
);

CREATE TABLE IF NOT EXISTS node_config_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    config_json TEXT,
    changed_by  TEXT    NOT NULL,
    changed_at  REAL    NOT NULL,
    diff_note   TEXT
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.executescript(_NODES_SCHEMA)
    # Idempotent migrations for columns added after initial schema.
    cols = {row[1] for row in con.execute("PRAGMA table_info(nodes)").fetchall()}
    if "freq_group_id" not in cols:
        con.execute("ALTER TABLE nodes ADD COLUMN freq_group_id TEXT")
    if "config_file_path" not in cols:
        con.execute("ALTER TABLE nodes ADD COLUMN config_file_path TEXT")
    if "config_file_mtime" not in cols:
        con.execute("ALTER TABLE nodes ADD COLUMN config_file_mtime REAL")
    con.commit()
    return con


def _hash_secret(plaintext: str) -> str:
    """
    SHA-256 hash of the secret, stored as 'sha256:<hex>'.

    NOTE: This will be replaced with bcrypt (via passlib[bcrypt]) when the full
    authentication system is implemented.  The 'sha256:' prefix lets the server
    distinguish old-style hashes from bcrypt hashes during migration.
    """
    digest = hashlib.sha256(plaintext.encode()).hexdigest()
    return f"sha256:{digest}"


def _generate_secret() -> str:
    """Return a 256-bit (64 hex char) cryptographically random token."""
    return secrets.token_hex(32)


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_config_file(path: str) -> Any:
    """Load a YAML or JSON file and return the parsed object."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: config file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            sys.exit("error: PyYAML is required to load .yaml config files (pip install pyyaml)")
        return yaml.safe_load(text)
    # Treat everything else as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        sys.exit(f"error: failed to parse {p} as JSON: {exc}")


def _record_config_history(
    con: sqlite3.Connection,
    node_id: str,
    version: int,
    config_json: str | None,
    changed_by: str,
    diff_note: str,
) -> None:
    con.execute(
        """
        INSERT INTO node_config_history
            (node_id, version, config_json, changed_by, changed_at, diff_note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (node_id, version, config_json, changed_by, time.time(), diff_note),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(con: sqlite3.Connection, _args: argparse.Namespace) -> None:
    rows = con.execute(
        "SELECT node_id, label, enabled, "
        "       config_version, last_seen_at "
        "FROM nodes ORDER BY node_id"
    ).fetchall()

    if not rows:
        print("No nodes registered.")
        return

    # Column widths
    col_id  = max(len("NODE ID"),    max(len(r["node_id"]) for r in rows))
    col_lbl = max(len("LABEL"),      max(len(r["label"] or "") for r in rows))
    hdr = (
        f"{'NODE ID':<{col_id}}  {'LABEL':<{col_lbl}}  "
        f"{'EN':>2}  {'VER':>3}  LAST SEEN"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['node_id']:<{col_id}}  {(r['label'] or ''):<{col_lbl}}  "
            f"{'Y' if r['enabled'] else 'N':>2}  "
            f"{r['config_version']:>3}  {_fmt_ts(r['last_seen_at'])}"
        )


def cmd_add(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    node_id = args.node_id
    existing = con.execute(
        "SELECT node_id FROM nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if existing:
        sys.exit(f"error: node '{node_id}' already exists.  Use set-config or regen-secret.")

    plaintext = _generate_secret()
    secret_hash = _hash_secret(plaintext)
    now = time.time()

    con.execute(
        """
        INSERT INTO nodes
            (node_id, secret_hash, label,
             registered_at, enabled, config_version, config_json)
        VALUES (?, ?, ?, ?, 1, 0, NULL)
        """,
        (
            node_id,
            secret_hash,
            args.label,
            now,
        ),
    )
    con.commit()

    print(f"Node '{node_id}' registered.")
    print()
    print("Bootstrap config for the node operator")
    print("---------------------------------------")
    print(f"server_url:  <your server URL>")
    print(f"node_id:     {node_id}")
    print(f"node_secret: {plaintext}")
    print()
    print("WARNING: this secret will not be shown again.  Copy it to the node's")
    print("         bootstrap.yaml (or /etc/beagle/bootstrap.yaml) immediately.")


def cmd_show(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = con.execute(
        "SELECT * FROM nodes WHERE node_id = ?", (args.node_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: node '{args.node_id}' not found.")

    r = dict(row)
    print(f"node_id:          {r['node_id']}")
    print(f"label:            {r['label'] or '(none)'}")
    print(f"enabled:          {'yes' if r['enabled'] else 'NO (disabled)'}")
    print(f"registered_at:    {_fmt_ts(r['registered_at'])}")
    print(f"last_seen_at:     {_fmt_ts(r['last_seen_at'])}")
    print(f"last_ip:          {r['last_ip'] or '(never connected)'}")
    print(f"config_version:   {r['config_version']}")
    print(f"template_id:      {r['config_template_id'] or '(none)'}")
    print(f"freq_group_id:    {r.get('freq_group_id') or '(none)'}")
    print(f"config_file_path: {r.get('config_file_path') or '(none - inline / API)'}")
    print()

    config_json = r["config_json"]
    merged_note = ""

    if getattr(args, "merged", False) and config_json:
        # Apply the same overlay the long-poll handler uses, so the operator
        # sees exactly what the node will receive on its next config fetch.
        try:
            from beagle_server.db import apply_freq_group_overlay
        except ImportError as exc:
            sys.exit(
                "error: --merged requires the beagle_server package on "
                f"PYTHONPATH ({exc})"
            )
        try:
            parsed = json.loads(config_json)
        except json.JSONDecodeError as exc:
            sys.exit(f"error: stored config_json is not valid JSON: {exc}")

        group_id = r.get("freq_group_id")
        if group_id:
            grp_row = con.execute(
                "SELECT * FROM node_freq_groups WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            grp = dict(grp_row) if grp_row is not None else None
            if grp is None:
                merged_note = (
                    f"  (note: freq_group_id '{group_id}' not found - "
                    "no overlay applied)"
                )
            else:
                apply_freq_group_overlay(parsed, grp)
                merged_note = f"  (freq group '{group_id}' overlay applied)"
        else:
            merged_note = "  (no freq_group assigned - raw config shown)"

        print(f"config_json (merged):{merged_note}")
        print(json.dumps(parsed, indent=2))
        return

    if config_json:
        try:
            parsed = json.loads(config_json)
            print("config_json:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(f"config_json (raw): {config_json}")
    else:
        print("config_json:      (none - node uses server defaults)")


def cmd_set_config(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    node_id = args.node_id
    row = con.execute(
        "SELECT config_version FROM nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: node '{node_id}' not found.")

    if args.config_file and args.config_json:
        sys.exit("error: specify --config-file or --config-json, not both.")

    file_path: str | None = None
    file_mtime: float | None = None

    if args.config_file:
        obj = _load_config_file(args.config_file)
        new_json: str | None = json.dumps(obj)
        diff_note = f"loaded from {args.config_file}"
        # Remember absolute path and mtime for reload-configs
        file_path = str(Path(args.config_file).resolve())
        file_mtime = Path(args.config_file).stat().st_mtime
    elif args.config_json:
        # Validate it parses
        try:
            json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            sys.exit(f"error: invalid JSON in --config-json: {exc}")
        new_json = args.config_json
        diff_note = "set via --config-json"
    else:
        new_json = None
        diff_note = "config cleared"

    new_version = row["config_version"] + 1
    _record_config_history(
        con, node_id, new_version, new_json,
        changed_by="manage_nodes.py", diff_note=diff_note,
    )
    con.execute(
        "UPDATE nodes SET config_json = ?, config_version = ?, "
        "config_file_path = ?, config_file_mtime = ? WHERE node_id = ?",
        (new_json, new_version, file_path, file_mtime, node_id),
    )
    con.commit()

    if new_json is None:
        print(f"Config cleared for '{node_id}' (version -> {new_version}).")
    else:
        print(f"Config updated for '{node_id}' (version -> {new_version}).")
        if file_path:
            print(f"  file: {file_path} (mtime tracked for reload)")


def cmd_enable(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _set_enabled(con, args.node_id, enabled=True)


def cmd_disable(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _set_enabled(con, args.node_id, enabled=False)


def _set_enabled(con: sqlite3.Connection, node_id: str, enabled: bool) -> None:
    cur = con.execute(
        "UPDATE nodes SET enabled = ? WHERE node_id = ?",
        (1 if enabled else 0, node_id),
    )
    if cur.rowcount == 0:
        sys.exit(f"error: node '{node_id}' not found.")
    con.commit()
    state = "enabled" if enabled else "disabled"
    print(f"Node '{node_id}' {state}.")


def cmd_remove(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    node_id = args.node_id
    row = con.execute(
        "SELECT node_id FROM nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: node '{node_id}' not found.")

    if not args.yes:
        answer = input(
            f"Remove node '{node_id}' and all its config history? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    con.execute("DELETE FROM node_config_history WHERE node_id = ?", (node_id,))
    con.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
    con.commit()
    print(f"Node '{node_id}' removed.")


def cmd_regen_secret(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    node_id = args.node_id
    row = con.execute(
        "SELECT node_id FROM nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: node '{node_id}' not found.")

    plaintext = _generate_secret()
    secret_hash = _hash_secret(plaintext)
    con.execute(
        "UPDATE nodes SET secret_hash = ? WHERE node_id = ?",
        (secret_hash, node_id),
    )
    con.commit()

    print(f"New secret for '{node_id}':")
    print()
    print(f"node_secret: {plaintext}")
    print()
    print("WARNING: the old secret is immediately invalid.  Update bootstrap.yaml on")
    print("         the node before restarting it.  This secret will not be shown again.")


# ---------------------------------------------------------------------------
# Frequency group commands
# ---------------------------------------------------------------------------

_FREQ_GROUPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS node_freq_groups (
    group_id             TEXT    PRIMARY KEY,
    label                TEXT    NOT NULL,
    description          TEXT,
    sync_freq_hz         REAL    NOT NULL,
    sync_station_id      TEXT    NOT NULL,
    sync_station_lat     REAL    NOT NULL,
    sync_station_lon     REAL    NOT NULL,
    target_channels_json TEXT    NOT NULL,
    created_at           REAL    NOT NULL,
    updated_at           REAL    NOT NULL
);
"""


def _ensure_groups_schema(con: sqlite3.Connection) -> None:
    """Create node_freq_groups table and freq_group_id column if missing."""
    con.executescript(_FREQ_GROUPS_SCHEMA)
    cols = {row[1] for row in con.execute("PRAGMA table_info(nodes)").fetchall()}
    if "freq_group_id" not in cols:
        con.execute("ALTER TABLE nodes ADD COLUMN freq_group_id TEXT")
        con.commit()


def cmd_group_list(con: sqlite3.Connection, _args: argparse.Namespace) -> None:
    _ensure_groups_schema(con)
    rows = con.execute(
        "SELECT group_id, label, sync_freq_hz, sync_station_id "
        "FROM node_freq_groups ORDER BY group_id"
    ).fetchall()

    if not rows:
        print("No frequency groups defined.")
        return

    col_id  = max(len("GROUP ID"),   max(len(r["group_id"]) for r in rows))
    col_lbl = max(len("LABEL"),      max(len(r["label"] or "") for r in rows))
    col_sid = max(len("SYNC STATION"), max(len(r["sync_station_id"] or "") for r in rows))
    hdr = (
        f"{'GROUP ID':<{col_id}}  {'LABEL':<{col_lbl}}  "
        f"{'SYNC MHz':>9}  {'SYNC STATION':<{col_sid}}  MEMBERS"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        member_count = con.execute(
            "SELECT COUNT(*) FROM nodes WHERE freq_group_id = ?",
            (r["group_id"],)
        ).fetchone()[0]
        freq_mhz = r["sync_freq_hz"] / 1e6
        print(
            f"{r['group_id']:<{col_id}}  {(r['label'] or ''):<{col_lbl}}  "
            f"{freq_mhz:>9.3f}  {r['sync_station_id']:<{col_sid}}  {member_count}"
        )


def cmd_group_add(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _ensure_groups_schema(con)
    group_id = args.group_id

    existing = con.execute(
        "SELECT group_id FROM node_freq_groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if existing:
        sys.exit(f"error: group '{group_id}' already exists.")

    # Parse target channels from JSON string or file
    if args.channels_file:
        tc = _load_config_file(args.channels_file)
    elif args.channels_json:
        try:
            tc = json.loads(args.channels_json)
        except json.JSONDecodeError as exc:
            sys.exit(f"error: invalid JSON in --channels-json: {exc}")
    else:
        sys.exit("error: specify --channels-file or --channels-json")

    if not isinstance(tc, list) or not tc:
        sys.exit("error: target channels must be a non-empty list")

    now = time.time()
    con.execute(
        """
        INSERT INTO node_freq_groups
            (group_id, label, description, sync_freq_hz, sync_station_id,
             sync_station_lat, sync_station_lon, target_channels_json,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            group_id, args.label, args.description,
            args.sync_freq_hz, args.sync_station_id,
            args.sync_station_lat, args.sync_station_lon,
            json.dumps(tc), now, now,
        ),
    )
    con.commit()
    print(f"Group '{group_id}' created.")


def cmd_group_show(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _ensure_groups_schema(con)
    row = con.execute(
        "SELECT * FROM node_freq_groups WHERE group_id = ?", (args.group_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: group '{args.group_id}' not found.")

    r = dict(row)
    print(f"group_id:        {r['group_id']}")
    print(f"label:           {r['label']}")
    print(f"description:     {r['description'] or '(none)'}")
    print(f"sync_freq_hz:    {r['sync_freq_hz']}  ({r['sync_freq_hz']/1e6:.3f} MHz)")
    print(f"sync_station_id: {r['sync_station_id']}")
    print(f"sync_station:    {r['sync_station_lat']}, {r['sync_station_lon']}")
    print(f"created_at:      {_fmt_ts(r['created_at'])}")
    print(f"updated_at:      {_fmt_ts(r['updated_at'])}")
    print()

    try:
        channels = json.loads(r["target_channels_json"])
        print("target_channels:")
        for ch in channels:
            freq = ch.get("frequency_hz", "?")
            label = ch.get("label", "")
            if isinstance(freq, (int, float)):
                print(f"  {freq/1e6:.3f} MHz  {label}")
            else:
                print(f"  {freq}  {label}")
    except (json.JSONDecodeError, TypeError):
        print(f"target_channels_json (raw): {r['target_channels_json']}")

    print()
    members = con.execute(
        "SELECT node_id FROM nodes WHERE freq_group_id = ? ORDER BY node_id",
        (args.group_id,)
    ).fetchall()
    if members:
        print(f"members ({len(members)}):")
        for m in members:
            print(f"  {m['node_id']}")
    else:
        print("members: (none)")


def cmd_group_remove(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _ensure_groups_schema(con)
    group_id = args.group_id
    row = con.execute(
        "SELECT group_id FROM node_freq_groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: group '{group_id}' not found.")

    if not args.yes:
        answer = input(
            f"Remove group '{group_id}' and unassign all its members? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    con.execute(
        "UPDATE nodes SET freq_group_id = NULL WHERE freq_group_id = ?",
        (group_id,),
    )
    con.execute(
        "DELETE FROM node_freq_groups WHERE group_id = ?", (group_id,),
    )
    con.commit()
    print(f"Group '{group_id}' removed.")


def cmd_group_set_node(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    _ensure_groups_schema(con)
    node_id = args.node_id
    group_id = args.group_id  # None means unassign

    row = con.execute(
        "SELECT config_version FROM nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"error: node '{node_id}' not found.")

    if group_id is not None:
        grp = con.execute(
            "SELECT group_id FROM node_freq_groups WHERE group_id = ?", (group_id,)
        ).fetchone()
        if grp is None:
            sys.exit(f"error: group '{group_id}' not found.")

    new_version = row["config_version"] + 1
    _record_config_history(
        con, node_id, new_version, None,
        changed_by="manage_nodes.py",
        diff_note=f"freq_group_id -> {group_id or '(none)'}",
    )
    con.execute(
        "UPDATE nodes SET freq_group_id = ?, config_version = ? WHERE node_id = ?",
        (group_id, new_version, node_id),
    )
    con.commit()

    if group_id:
        print(f"Node '{node_id}' assigned to group '{group_id}' (version -> {new_version}).")
    else:
        print(f"Node '{node_id}' removed from its frequency group (version -> {new_version}).")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manage Beagle node records and configs in the server database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # DB / config source (mutually exclusive)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--db",
        metavar="PATH",
        help="Path to the SQLite database file.",
    )
    src.add_argument(
        "--server-config",
        metavar="PATH",
        help="Path to server.yaml; the database.path value is used.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered nodes.")

    # add
    add_p = sub.add_parser("add", help="Register a new node and generate its secret.")
    add_p.add_argument("node_id", help="Unique node identifier, e.g. seattle-north-01")
    add_p.add_argument("--label", default=None, help="Human-readable display name.")

    # show
    show_p = sub.add_parser("show", help="Show full details for a node.")
    show_p.add_argument("node_id")
    show_p.add_argument(
        "--merged",
        action="store_true",
        help=(
            "Print the config_json with the node's frequency group overlay "
            "applied - i.e. exactly what the long-poll endpoint would serve "
            "to the node on its next config fetch."
        ),
    )

    # set-config
    sc_p = sub.add_parser("set-config", help="Replace the server-assigned config for a node.")
    sc_p.add_argument("node_id")
    sc_p.add_argument("--config-file", metavar="PATH",
                      help="YAML or JSON file containing the node config.")
    sc_p.add_argument("--config-json", metavar="JSON",
                      help="Inline JSON string (alternative to --config-file).")

    # enable / disable
    en_p = sub.add_parser("enable", help="Enable a node.")
    en_p.add_argument("node_id")
    dis_p = sub.add_parser("disable", help="Disable a node (server rejects its events).")
    dis_p.add_argument("node_id")

    # remove
    rm_p = sub.add_parser("remove", help="Delete a node record.")
    rm_p.add_argument("node_id")
    rm_p.add_argument("-y", "--yes", action="store_true",
                      help="Skip confirmation prompt.")

    # regen-secret
    rg_p = sub.add_parser("regen-secret", help="Generate a new secret for a node.")
    rg_p.add_argument("node_id")

    # --- Frequency group commands ---

    # group-list
    sub.add_parser("group-list", help="List all frequency groups.")

    # group-add
    ga_p = sub.add_parser("group-add", help="Create a new frequency group.")
    ga_p.add_argument("group_id", help="Unique group identifier, e.g. seattle-fm")
    ga_p.add_argument("--label", required=True, help="Human-readable display name.")
    ga_p.add_argument("--description", default=None, help="Optional description.")
    ga_p.add_argument("--sync-freq-hz", type=float, required=True,
                      help="Sync station frequency in Hz.")
    ga_p.add_argument("--sync-station-id", required=True,
                      help="Sync station identifier (e.g. call sign).")
    ga_p.add_argument("--sync-station-lat", type=float, required=True,
                      help="Sync station latitude (decimal degrees).")
    ga_p.add_argument("--sync-station-lon", type=float, required=True,
                      help="Sync station longitude (decimal degrees).")
    ga_p.add_argument("--channels-file", metavar="PATH",
                      help="JSON or YAML file with target channels list.")
    ga_p.add_argument("--channels-json", metavar="JSON",
                      help="Inline JSON target channels list.")

    # group-show
    gs_p = sub.add_parser("group-show", help="Show details for a frequency group.")
    gs_p.add_argument("group_id")

    # group-remove
    gr_p = sub.add_parser("group-remove", help="Delete a frequency group.")
    gr_p.add_argument("group_id")
    gr_p.add_argument("-y", "--yes", action="store_true",
                      help="Skip confirmation prompt.")

    # group-set-node (assign or unassign a node)
    gsn_p = sub.add_parser("group-set-node",
                           help="Assign a node to a frequency group (or unassign).")
    gsn_p.add_argument("node_id", help="Node to assign.")
    gsn_p.add_argument("--group", dest="group_id", default=None,
                       help="Group ID to assign to.  Omit or pass empty to unassign.")

    return p


def _resolve_db_path(args: argparse.Namespace) -> str:
    if args.db:
        return args.db

    # Load from server config YAML
    cfg_path = Path(args.server_config)
    if not cfg_path.exists():
        sys.exit(f"error: server config not found: {cfg_path}")
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        sys.exit("error: PyYAML is required to read --server-config (pip install pyyaml)")
    with cfg_path.open() as f:
        raw = yaml.safe_load(f) or {}
    db_path = raw.get("database", {}).get("registry_path", "data/tdoa_registry.db")
    return str(db_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    "list":           cmd_list,
    "add":            cmd_add,
    "show":           cmd_show,
    "set-config":     cmd_set_config,
    "enable":         cmd_enable,
    "disable":        cmd_disable,
    "remove":         cmd_remove,
    "regen-secret":   cmd_regen_secret,
    "group-list":     cmd_group_list,
    "group-add":      cmd_group_add,
    "group-show":     cmd_group_show,
    "group-remove":   cmd_group_remove,
    "group-set-node": cmd_group_set_node,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    db_path = _resolve_db_path(args)
    con = _open_db(db_path)
    try:
        _COMMANDS[args.command](con, args)
    finally:
        con.close()


if __name__ == "__main__":
    main()
