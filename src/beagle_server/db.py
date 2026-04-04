# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
SQLite database layer for the aggregation server.

Uses aiosqlite for async access from FastAPI request handlers.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_OPERATIONAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT    UNIQUE NOT NULL,
    node_id       TEXT    NOT NULL,
    channel_hz    REAL    NOT NULL,
    sync_delta_ns INTEGER NOT NULL,
    sync_tx_id    TEXT    NOT NULL,
    sync_tx_lat   REAL    NOT NULL,
    sync_tx_lon   REAL    NOT NULL,
    node_lat      REAL    NOT NULL,
    node_lon      REAL    NOT NULL,
    event_type    TEXT    NOT NULL,
    onset_time_ns INTEGER NOT NULL,
    corr_peak     REAL    NOT NULL DEFAULT 0.0,
    received_at   REAL    NOT NULL,
    raw_json      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_group
    ON events (channel_hz, event_type, sync_tx_id, onset_time_ns);
CREATE INDEX IF NOT EXISTS idx_events_node_recency
    ON events (node_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_received_at
    ON events (received_at);

CREATE TABLE IF NOT EXISTS fixes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_hz    REAL    NOT NULL,
    event_type    TEXT    NOT NULL,
    computed_at   REAL    NOT NULL,
    latitude_deg  REAL    NOT NULL,
    longitude_deg REAL    NOT NULL,
    residual_ns   REAL,
    node_count    INTEGER NOT NULL,
    nodes_json    TEXT    NOT NULL,
    onset_time_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS heatmap_cells (
    cell_i      INTEGER NOT NULL,
    cell_j      INTEGER NOT NULL,
    cell_size_m REAL    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 0.0,
    updated_at  REAL    NOT NULL,
    PRIMARY KEY (cell_i, cell_j, cell_size_m)
);
"""

_REGISTRY_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'viewer',
    created_at    REAL NOT NULL,
    last_login_at REAL,
    totp_secret   TEXT,
    totp_enabled  INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

CREATE TABLE IF NOT EXISTS user_sessions (
    token        TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    last_used_at REAL
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions (user_id);

CREATE TABLE IF NOT EXISTS partial_sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS oauth_accounts (
    provider         TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    email            TEXT,
    linked_at        REAL NOT NULL,
    PRIMARY KEY (provider, provider_user_id)
);
CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_accounts (user_id);
"""


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

async def open_db(path: str) -> aiosqlite.Connection:
    """Open (or create) the operational SQLite database (events, fixes, heatmap)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(p))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.executescript(_OPERATIONAL_SCHEMA)
    await db.commit()
    return db


async def open_registry_db(path: str) -> aiosqlite.Connection:
    """Open (or create) the registry SQLite database (nodes, users, sessions)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(p))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.executescript(_REGISTRY_SCHEMA)
    # Idempotent migration: add freq_group_id column if missing.
    async with db.execute("PRAGMA table_info(nodes)") as cur:
        cols = {row[1] for row in await cur.fetchall()}
    if "freq_group_id" not in cols:
        await db.execute(
            "ALTER TABLE nodes ADD COLUMN freq_group_id TEXT"
        )
    if "config_file_path" not in cols:
        await db.execute(
            "ALTER TABLE nodes ADD COLUMN config_file_path TEXT"
        )
    if "config_file_mtime" not in cols:
        await db.execute(
            "ALTER TABLE nodes ADD COLUMN config_file_mtime REAL"
        )
    # Idempotent migration: add TOTP columns to users if missing.
    async with db.execute("PRAGMA table_info(users)") as cur:
        user_cols = {row[1] for row in await cur.fetchall()}
    if "totp_secret" not in user_cols:
        await db.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
    if "totp_enabled" not in user_cols:
        await db.execute("ALTER TABLE users ADD COLUMN totp_enabled INTEGER DEFAULT 0")
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

async def upsert_event(db: aiosqlite.Connection, event_data: dict[str, Any]) -> None:
    """
    Insert a new event or replace it if the event_id already exists (amendment).
    """
    await db.execute(
        """
        INSERT INTO events
            (event_id, node_id, channel_hz, sync_delta_ns,
             sync_tx_id, sync_tx_lat, sync_tx_lon,
             node_lat, node_lon, event_type, onset_time_ns,
             corr_peak, received_at, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            sync_delta_ns  = excluded.sync_delta_ns,
            corr_peak      = excluded.corr_peak,
            received_at    = excluded.received_at,
            raw_json       = excluded.raw_json
        """,
        (
            event_data["event_id"],
            event_data["node_id"],
            event_data["channel_hz"],
            event_data["sync_delta_ns"],
            event_data["sync_tx_id"],
            event_data["sync_tx_lat"],
            event_data["sync_tx_lon"],
            event_data["node_lat"],
            event_data["node_lon"],
            event_data["event_type"],
            event_data["onset_time_ns"],
            event_data["corr_peak"],
            event_data["received_at"],
            event_data["raw_json"],
        ),
    )
    await db.commit()


async def fetch_candidate_events(
    db: aiosqlite.Connection,
    channel_hz: float,
    event_type: str,
    sync_tx_id: str,
    onset_time_ns: int,
    correlation_window_ns: int,
    min_corr_peak: float,
) -> list[dict[str, Any]]:
    """
    Fetch all events that could belong to the same transmission group as a
    reference event (same channel, event_type, sync_tx, and onset within window).
    """
    low  = onset_time_ns - correlation_window_ns
    high = onset_time_ns + correlation_window_ns
    async with db.execute(
        """
        SELECT * FROM events
        WHERE channel_hz  = ?
          AND event_type  = ?
          AND sync_tx_id  = ?
          AND onset_time_ns BETWEEN ? AND ?
          AND corr_peak   >= ?
        ORDER BY onset_time_ns
        """,
        (channel_hz, event_type, sync_tx_id, low, high, min_corr_peak),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def fetch_recent_events(
    db: aiosqlite.Connection,
    limit: int = 100,
    node_id: str | None = None,
    channel_hz: float | None = None,
) -> list[dict[str, Any]]:
    """Fetch the most recent events, optionally filtered."""
    clauses: list[str] = []
    params: list[Any] = []
    if node_id is not None:
        clauses.append("node_id = ?")
        params.append(node_id)
    if channel_hz is not None:
        clauses.append("channel_hz = ?")
        params.append(channel_hz)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    async with db.execute(
        f"SELECT * FROM events {where} ORDER BY received_at DESC LIMIT ?",
        params,
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def count_events(db: aiosqlite.Connection) -> int:
    async with db.execute("SELECT COUNT(*) FROM events") as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Fixes
# ---------------------------------------------------------------------------

async def insert_fix(db: aiosqlite.Connection, fix: dict[str, Any]) -> int:
    """Insert a fix record and return the new row id."""
    async with db.execute(
        """
        INSERT INTO fixes
            (channel_hz, event_type, computed_at,
             latitude_deg, longitude_deg, residual_ns,
             node_count, nodes_json, onset_time_ns)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fix["channel_hz"],
            fix["event_type"],
            fix["computed_at"],
            fix["latitude_deg"],
            fix["longitude_deg"],
            fix.get("residual_ns"),
            fix["node_count"],
            json.dumps(fix["nodes"]),
            fix["onset_time_ns"],
        ),
    ) as cur:
        fix_id = cur.lastrowid
    await db.commit()
    return fix_id  # type: ignore[return-value]


async def fetch_fixes(
    db: aiosqlite.Connection,
    limit: int = 100,
    max_age_s: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Fetch recent fixes, optionally filtered by age.

    max_age_s=0 means return all fixes.
    """
    params: list[Any] = []
    where = ""
    if max_age_s > 0:
        cutoff = time.time() - max_age_s
        where = "WHERE computed_at >= ?"
        params.append(cutoff)
    params.append(limit)
    async with db.execute(
        f"SELECT * FROM fixes {where} ORDER BY computed_at DESC LIMIT ?",
        params,
    ) as cur:
        rows = await cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["nodes"] = json.loads(d.pop("nodes_json"))
        result.append(d)
    return result


async def fetch_fix_by_id(
    db: aiosqlite.Connection,
    fix_id: int,
) -> dict[str, Any] | None:
    async with db.execute("SELECT * FROM fixes WHERE id = ?", (fix_id,)) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["nodes"] = json.loads(d.pop("nodes_json"))
    return d


async def count_fixes(db: aiosqlite.Connection) -> int:
    async with db.execute("SELECT COUNT(*) FROM fixes") as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


async def delete_all_fixes(db: aiosqlite.Connection) -> int:
    """Delete every row from the fixes table. Returns the number deleted."""
    async with db.execute("DELETE FROM fixes") as cur:
        deleted = cur.rowcount
    await db.commit()
    return deleted if deleted is not None else 0


# ---------------------------------------------------------------------------
# Heatmap accumulator
# ---------------------------------------------------------------------------

def _lat_lon_for_cell(cell_i: int, cell_j: int, cell_size_m: float) -> tuple[float, float]:
    """
    Convert integer cell indices back to (lat, lon) of the cell centre.

    cell_i is the rounded(lat / lat_step) index, independent of longitude.
    cell_j is the rounded(lon / lon_step) index where lon_step is computed
    from the cell centre latitude (cell_i * lat_step), so every cell that
    maps to the same (cell_i, cell_j) pair is guaranteed to produce the
    same lat/lon regardless of where the contributing fix was located.
    """
    lat_step = cell_size_m / 111_195.0
    lat_c = cell_i * lat_step
    cos_lat = math.cos(math.radians(lat_c))
    lon_step = cell_size_m / (111_195.0 * max(cos_lat, 0.01))
    return lat_c, cell_j * lon_step


async def add_fix_to_heatmap(
    db: aiosqlite.Connection,
    lat: float,
    lon: float,
    cell_size_m: float = 200.0,
    sigma_cells: float = 1.5,
) -> None:
    """
    Accumulate Gaussian-weighted contributions from a fix into heatmap_cells.

    Each fix spreads weight across nearby cells:
        weight(di, dj) = exp(-(di^2 + dj^2) / (2 * sigma_cells^2))
    Cells beyond 3 * sigma_cells are skipped (weight < 0.011).
    """
    lat_step = cell_size_m / 111_195.0
    center_i = round(lat / lat_step)
    # Use cell centre latitude for lon_step so all fixes in the same cell
    # produce identical (cell_i, cell_j) pairs.
    center_lat = center_i * lat_step
    cos_lat = math.cos(math.radians(center_lat))
    lon_step = cell_size_m / (111_195.0 * max(cos_lat, 0.01))
    center_j = round(lon / lon_step)

    cutoff = math.ceil(3.0 * sigma_cells)
    now = time.time()
    two_sigma_sq = 2.0 * sigma_cells ** 2

    rows = []
    for di in range(-cutoff, cutoff + 1):
        for dj in range(-cutoff, cutoff + 1):
            w = math.exp(-(di * di + dj * dj) / two_sigma_sq)
            if w < 0.01:
                continue
            rows.append((center_i + di, center_j + dj, cell_size_m, w, now))

    await db.executemany(
        """
        INSERT INTO heatmap_cells (cell_i, cell_j, cell_size_m, weight, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(cell_i, cell_j, cell_size_m)
        DO UPDATE SET weight = weight + excluded.weight,
                      updated_at = excluded.updated_at
        """,
        rows,
    )
    await db.commit()


async def fetch_heatmap_cells(
    db: aiosqlite.Connection,
    cell_size_m: float = 200.0,
) -> list[list[float]]:
    """
    Return [[lat, lon, weight], ...] for all accumulated heatmap cells of the
    given cell size, suitable for passing directly to folium.plugins.HeatMap.
    """
    async with db.execute(
        "SELECT cell_i, cell_j, weight FROM heatmap_cells WHERE cell_size_m = ?",
        (cell_size_m,),
    ) as cur:
        rows = await cur.fetchall()
    result: list[list[float]] = []
    for row in rows:
        lat_c, lon_c = _lat_lon_for_cell(int(row[0]), int(row[1]), cell_size_m)
        result.append([lat_c, lon_c, float(row[2])])
    return result


async def delete_heatmap(db: aiosqlite.Connection) -> int:
    """Truncate heatmap_cells. Returns the number of rows deleted."""
    async with db.execute("DELETE FROM heatmap_cells") as cur:
        deleted = cur.rowcount
    await db.commit()
    return deleted if deleted is not None else 0


async def count_heatmap_cells(db: aiosqlite.Connection) -> int:
    async with db.execute("SELECT COUNT(*) FROM heatmap_cells") as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


async def fetch_last_fix_age_s(db: aiosqlite.Connection) -> float | None:
    """Returns seconds since the most recent fix, or None if no fixes exist."""
    async with db.execute("SELECT MAX(computed_at) FROM fixes") as cur:
        row = await cur.fetchone()
    if row is None or row[0] is None:
        return None
    return time.time() - row[0]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def ensure_node_exists(
    db: aiosqlite.Connection,
    node_id: str,
    ip: str | None = None,
) -> dict[str, Any]:
    """Return the node row, creating a shadow registration if it doesn't exist.

    Shadow-registered nodes have secret_hash='unregistered' and can be
    managed (enabled/disabled, labelled, configured) exactly like
    explicitly registered nodes.  They cannot authenticate in nodedb mode
    because no valid secret exists.
    """
    row = await fetch_node(db, node_id)
    if row is not None:
        return row
    now = time.time()
    await db.execute(
        "INSERT OR IGNORE INTO nodes "
        "(node_id, secret_hash, label, registered_at, last_seen_at, last_ip, enabled) "
        "VALUES (?, 'unregistered', NULL, ?, ?, ?, 1)",
        (node_id, now, now, ip),
    )
    await db.commit()
    return dict(await (await db.execute(
        "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
    )).fetchone())  # type: ignore[arg-type]


async def fetch_node(db: aiosqlite.Connection, node_id: str) -> dict[str, Any] | None:
    """Return a single node row as a dict, or None if not found."""
    async with db.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)) as cur:
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def fetch_all_nodes(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all node rows ordered by node_id."""
    async with db.execute("SELECT * FROM nodes ORDER BY node_id") as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def update_node_seen(
    db: aiosqlite.Connection,
    node_id: str,
    ip: str | None,
) -> None:
    """Update last_seen_at and last_ip for a node."""
    await db.execute(
        "UPDATE nodes SET last_seen_at = ?, last_ip = ? WHERE node_id = ?",
        (time.time(), ip, node_id),
    )
    await db.commit()


async def update_node_enabled(
    db: aiosqlite.Connection,
    node_id: str,
    enabled: bool,
) -> bool:
    """Set the enabled flag.  Returns True if the node was found."""
    cur = await db.execute(
        "UPDATE nodes SET enabled = ? WHERE node_id = ?",
        (1 if enabled else 0, node_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def update_node_config(
    db: aiosqlite.Connection,
    node_id: str,
    config_json: str | None,
    changed_by: str,
    diff_note: str = "",
) -> int:
    """
    Replace the node's config_json and increment config_version.

    Records the change in node_config_history.
    Returns the new config_version, or -1 if the node was not found.
    """
    row = await fetch_node(db, node_id)
    if row is None:
        return -1
    new_version = row["config_version"] + 1
    await db.execute(
        "UPDATE nodes SET config_json = ?, config_version = ? WHERE node_id = ?",
        (config_json, new_version, node_id),
    )
    await db.execute(
        """
        INSERT INTO node_config_history
            (node_id, version, config_json, changed_by, changed_at, diff_note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (node_id, new_version, config_json, changed_by, time.time(), diff_note),
    )
    await db.commit()
    return new_version


async def update_node_config_file_meta(
    db: aiosqlite.Connection,
    node_id: str,
    file_path: str | None,
    file_mtime: float | None,
) -> None:
    """Update the config file path and mtime for a node."""
    await db.execute(
        "UPDATE nodes SET config_file_path = ?, config_file_mtime = ? WHERE node_id = ?",
        (file_path, file_mtime, node_id),
    )
    await db.commit()


async def reload_node_configs(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Stat config files for all nodes and reload any that have changed.

    For each node with a config_file_path, stats the file and compares its
    mtime to config_file_mtime.  If the file is newer (or mtime was never
    recorded), re-reads the file, updates config_json, bumps config_version,
    and records the new mtime.

    Returns a list of dicts describing what changed:
      [{"node_id": ..., "new_version": ..., "status": "updated"|"unchanged"|"missing"|"error"}, ...]
    """
    import json as _json
    from pathlib import Path as _Path

    async with db.execute(
        "SELECT node_id, config_file_path, config_file_mtime, config_version "
        "FROM nodes WHERE config_file_path IS NOT NULL AND config_file_path != ''"
    ) as cur:
        nodes = [dict(r) for r in await cur.fetchall()]

    results: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["node_id"]
        fpath = _Path(node["config_file_path"])
        old_mtime = node["config_file_mtime"]

        if not fpath.exists():
            results.append({"node_id": node_id, "status": "missing",
                            "path": str(fpath)})
            continue

        try:
            current_mtime = fpath.stat().st_mtime
        except OSError as exc:
            results.append({"node_id": node_id, "status": "error",
                            "error": str(exc)})
            continue

        if old_mtime is not None and current_mtime <= old_mtime:
            results.append({"node_id": node_id, "status": "unchanged"})
            continue

        # File is newer -- reload it
        try:
            text = fpath.read_text()
            if fpath.suffix.lower() in (".yaml", ".yml"):
                import yaml  # type: ignore[import]
                obj = yaml.safe_load(text)
            else:
                obj = _json.loads(text)
            config_json = _json.dumps(obj)
        except Exception as exc:
            results.append({"node_id": node_id, "status": "error",
                            "error": f"parse error: {exc}"})
            continue

        new_version = node["config_version"] + 1
        await db.execute(
            "UPDATE nodes SET config_json = ?, config_version = ?, "
            "config_file_mtime = ? WHERE node_id = ?",
            (config_json, new_version, current_mtime, node_id),
        )
        await db.execute(
            """
            INSERT INTO node_config_history
                (node_id, version, config_json, changed_by, changed_at, diff_note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (node_id, new_version, config_json, "reload-configs",
             time.time(), f"reloaded from {fpath}"),
        )
        results.append({"node_id": node_id, "status": "updated",
                        "new_version": new_version})

    await db.commit()
    return results


async def delete_node(db: aiosqlite.Connection, node_id: str) -> bool:
    """Delete a node from the nodes table. Returns True if found and deleted."""
    cur = await db.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
    await db.commit()
    return (cur.rowcount or 0) > 0


async def create_node(
    db: aiosqlite.Connection,
    node_id: str,
    secret_hash: str,
    label: str | None,
) -> dict[str, Any]:
    """Insert a new node and return its row.  Raises sqlite3.IntegrityError on duplicate."""
    now = time.time()
    await db.execute(
        "INSERT INTO nodes (node_id, secret_hash, label, registered_at, enabled, config_version) "
        "VALUES (?, ?, ?, ?, 1, 0)",
        (node_id, secret_hash, label, now),
    )
    await db.commit()
    return dict(await (await db.execute(
        "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
    )).fetchone())  # type: ignore[arg-type]


async def update_node_secret(
    db: aiosqlite.Connection,
    node_id: str,
    secret_hash: str,
) -> bool:
    """Replace the node's secret_hash.  Returns True if the node was found."""
    cur = await db.execute(
        "UPDATE nodes SET secret_hash = ? WHERE node_id = ?",
        (secret_hash, node_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def update_node_label(
    db: aiosqlite.Connection,
    node_id: str,
    label: str | None,
) -> bool:
    """Update the node's display label.  Returns True if the node was found."""
    cur = await db.execute(
        "UPDATE nodes SET label = ? WHERE node_id = ?",
        (label, node_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


# ---------------------------------------------------------------------------
# Frequency groups
# ---------------------------------------------------------------------------

async def create_freq_group(
    db: aiosqlite.Connection,
    group_id: str,
    label: str,
    description: str | None,
    sync_freq_hz: float,
    sync_station_id: str,
    sync_station_lat: float,
    sync_station_lon: float,
    target_channels_json: str,
) -> dict[str, Any]:
    """Create a new frequency group. Returns the new row as a dict."""
    now = time.time()
    await db.execute(
        """
        INSERT INTO node_freq_groups
            (group_id, label, description,
             sync_freq_hz, sync_station_id, sync_station_lat, sync_station_lon,
             target_channels_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (group_id, label, description,
         sync_freq_hz, sync_station_id, sync_station_lat, sync_station_lon,
         target_channels_json, now, now),
    )
    await db.commit()
    return await fetch_freq_group(db, group_id)  # type: ignore[return-value]


async def fetch_freq_group(
    db: aiosqlite.Connection, group_id: str
) -> dict[str, Any] | None:
    """Return a single frequency group as a dict, or None."""
    async with db.execute(
        "SELECT * FROM node_freq_groups WHERE group_id = ?", (group_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def fetch_all_freq_groups(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all frequency groups ordered by group_id."""
    async with db.execute(
        "SELECT * FROM node_freq_groups ORDER BY group_id"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def update_freq_group(
    db: aiosqlite.Connection,
    group_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Update fields on a frequency group. Returns the updated row, or None.

    Accepted keys: label, description, sync_freq_hz, sync_station_id,
    sync_station_lat, sync_station_lon, target_channels_json.
    """
    allowed = {
        "label", "description", "sync_freq_hz", "sync_station_id",
        "sync_station_lat", "sync_station_lon", "target_channels_json",
    }
    sets = {k: v for k, v in updates.items() if k in allowed}
    if not sets:
        return await fetch_freq_group(db, group_id)
    sets["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in sets)
    params = list(sets.values()) + [group_id]
    await db.execute(
        f"UPDATE node_freq_groups SET {set_clause} WHERE group_id = ?",
        params,
    )
    await db.commit()
    return await fetch_freq_group(db, group_id)


async def delete_freq_group(db: aiosqlite.Connection, group_id: str) -> bool:
    """
    Delete a frequency group. Sets freq_group_id=NULL on all member nodes.
    Returns True if found and deleted.
    """
    await db.execute(
        "UPDATE nodes SET freq_group_id = NULL WHERE freq_group_id = ?",
        (group_id,),
    )
    cur = await db.execute(
        "DELETE FROM node_freq_groups WHERE group_id = ?", (group_id,)
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def fetch_group_member_ids(
    db: aiosqlite.Connection, group_id: str
) -> list[str]:
    """Return node_ids of all nodes assigned to a frequency group."""
    async with db.execute(
        "SELECT node_id FROM nodes WHERE freq_group_id = ? ORDER BY node_id",
        (group_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [row[0] for row in rows]


async def set_node_freq_group(
    db: aiosqlite.Connection,
    node_id: str,
    freq_group_id: str | None,
    changed_by: str,
) -> int:
    """
    Assign (or unassign) a node to a frequency group and bump config_version.

    This triggers the node's long-poll to wake up and fetch the new effective
    config with the group's frequency plan merged in.

    Returns new config_version, or -1 if the node was not found.
    """
    row = await fetch_node(db, node_id)
    if row is None:
        return -1
    new_version = row["config_version"] + 1
    await db.execute(
        "UPDATE nodes SET freq_group_id = ?, config_version = ? WHERE node_id = ?",
        (freq_group_id, new_version, node_id),
    )
    group_label = freq_group_id or "(none)"
    await db.execute(
        """
        INSERT INTO node_config_history
            (node_id, version, config_json, changed_by, changed_at, diff_note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (node_id, new_version, row["config_json"], changed_by, time.time(),
         f"freq_group_id -> {group_label}"),
    )
    await db.commit()
    return new_version


async def bump_group_members_version(
    db: aiosqlite.Connection,
    group_id: str,
    changed_by: str,
    diff_note: str = "",
) -> int:
    """
    Increment config_version on all nodes in a group so their long-poll
    connections wake up and fetch the updated frequency plan.

    Returns the number of nodes affected.
    """
    members = await fetch_group_member_ids(db, group_id)
    for node_id in members:
        row = await fetch_node(db, node_id)
        if row is None:
            continue
        new_version = row["config_version"] + 1
        await db.execute(
            "UPDATE nodes SET config_version = ? WHERE node_id = ?",
            (new_version, node_id),
        )
        await db.execute(
            """
            INSERT INTO node_config_history
                (node_id, version, config_json, changed_by, changed_at, diff_note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (node_id, new_version, row["config_json"], changed_by,
             time.time(), diff_note),
        )
    if members:
        await db.commit()
    return len(members)


async def fetch_event_node_summary(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """
    Return one summary row per distinct node_id seen in the events table.

    Selects the most recent event per node (by received_at) and extracts
    sdr_mode from the raw_json CarrierEvent payload.  Used by GET /map/nodes
    to surface nodes that have reported events but are not (or no longer)
    registered in the nodes table.
    """
    async with db.execute(
        """
        SELECT e.node_id, e.node_lat, e.node_lon,
               e.received_at AS last_seen_at, e.raw_json
        FROM events e
        INNER JOIN (
            SELECT node_id, MAX(id) AS max_id
            FROM events
            GROUP BY node_id
        ) latest ON e.id = latest.max_id
        ORDER BY e.node_id
        """
    ) as cur:
        rows = await cur.fetchall()

    result = []
    for row in rows:
        d = dict(row)
        sdr_mode = None
        try:
            raw = json.loads(d.pop("raw_json", "{}") or "{}")
            sdr_mode = raw.get("sdr_mode")
        except (json.JSONDecodeError, TypeError):
            pass
        d["sdr_mode"] = sdr_mode
        result.append(d)
    return result


async def fetch_node_snr_stats(
    db: aiosqlite.Connection,
    since_ts: float,
) -> list[dict[str, Any]]:
    """
    Return per-(node_id, channel_hz) signal quality statistics for events
    received since `since_ts` (Unix timestamp).

    Fetches individual event rows and aggregates in Python so we can compute
    percentiles (SQLite has no built-in percentile function).

    Each returned dict has:
      node_id, node_lat, node_lon, channel_hz, sync_tx_id,
      event_count, last_event_age_s,
      corr_peak_mean, corr_peak_min, corr_peak_p10,
      snr_db_mean, snr_db_min, snr_db_p10,   (peak_power_db - noise_floor_db)
      clock_source, clock_uncertainty_ns       (from most recent event)
    """
    async with db.execute(
        """
        SELECT node_id, node_lat, node_lon, channel_hz, sync_tx_id,
               corr_peak,
               CAST(json_extract(raw_json, '$.peak_power_db')   AS REAL)    AS peak_power_db,
               CAST(json_extract(raw_json, '$.noise_floor_db')  AS REAL)    AS noise_floor_db,
               json_extract(raw_json, '$.clock_source')                     AS clock_source,
               CAST(json_extract(raw_json, '$.clock_uncertainty_ns') AS INTEGER)
                                                                             AS clock_uncertainty_ns,
               received_at
        FROM events
        WHERE received_at >= ?
        ORDER BY node_id, channel_hz, received_at
        """,
        (since_ts,),
    ) as cur:
        rows = await cur.fetchall()

    # Aggregate per (node_id, channel_hz)
    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = dict(row)
        groups[(d["node_id"], d["channel_hz"])].append(d)

    now = time.time()
    result = []
    for (node_id, channel_hz), evs in sorted(groups.items()):
        latest = max(evs, key=lambda e: e["received_at"])

        corr_peaks = sorted(e["corr_peak"] for e in evs)
        n = len(corr_peaks)
        p10_idx = max(0, int(n * 0.10) - 1) if n > 1 else 0

        # SNR = peak_power_db - noise_floor_db.  Only include events where both
        # fields are present and noise_floor_db is non-zero (old events default 0).
        snr_vals = sorted(
            e["peak_power_db"] - e["noise_floor_db"]
            for e in evs
            if e["peak_power_db"] is not None
            and e["noise_floor_db"] is not None
            and e["noise_floor_db"] != 0.0
        )
        snr_n = len(snr_vals)
        snr_p10_idx = max(0, int(snr_n * 0.10) - 1) if snr_n > 1 else 0

        entry: dict[str, Any] = {
            "node_id": node_id,
            "node_lat": latest["node_lat"],
            "node_lon": latest["node_lon"],
            "channel_hz": channel_hz,
            "sync_tx_id": latest["sync_tx_id"],
            "event_count": n,
            "last_event_age_s": round(now - latest["received_at"], 1),
            "corr_peak_mean": round(sum(corr_peaks) / n, 3),
            "corr_peak_min": round(corr_peaks[0], 3),
            "corr_peak_p10": round(corr_peaks[p10_idx], 3),
            "clock_source": latest["clock_source"],
            "clock_uncertainty_ns": latest["clock_uncertainty_ns"],
        }
        if snr_vals:
            entry["snr_db_mean"] = round(sum(snr_vals) / snr_n, 1)
            entry["snr_db_min"] = round(snr_vals[0], 1)
            entry["snr_db_p10"] = round(snr_vals[snr_p10_idx], 1)
        else:
            entry["snr_db_mean"] = None
            entry["snr_db_min"] = None
            entry["snr_db_p10"] = None
        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# Users and sessions (userdb auth mode)
# ---------------------------------------------------------------------------

async def count_users(db: aiosqlite.Connection) -> int:
    """Return the total number of user accounts."""
    async with db.execute("SELECT COUNT(*) FROM users") as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


async def fetch_user_by_username(
    db: aiosqlite.Connection, username: str
) -> dict[str, Any] | None:
    """Return a user row by username, or None."""
    async with db.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def fetch_user_by_id(
    db: aiosqlite.Connection, user_id: str
) -> dict[str, Any] | None:
    """Return a user row by user_id, or None."""
    async with db.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def fetch_all_users(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all users ordered by username (password_hash excluded by caller)."""
    async with db.execute(
        "SELECT * FROM users ORDER BY username"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def create_user(
    db: aiosqlite.Connection,
    user_id: str,
    username: str,
    password_hash: str,
    role: str,
) -> None:
    """Insert a new user row."""
    await db.execute(
        """
        INSERT INTO users (user_id, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username, password_hash, role, time.time()),
    )
    await db.commit()


async def update_user_last_login(db: aiosqlite.Connection, user_id: str) -> None:
    await db.execute(
        "UPDATE users SET last_login_at = ? WHERE user_id = ?",
        (time.time(), user_id),
    )
    await db.commit()


async def update_user_role(
    db: aiosqlite.Connection, user_id: str, role: str
) -> bool:
    """Change a user's role. Returns True if the user was found."""
    cur = await db.execute(
        "UPDATE users SET role = ? WHERE user_id = ?", (role, user_id)
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def update_user_password(
    db: aiosqlite.Connection, user_id: str, password_hash: str
) -> bool:
    """Replace a user's password hash. Returns True if the user was found."""
    cur = await db.execute(
        "UPDATE users SET password_hash = ? WHERE user_id = ?",
        (password_hash, user_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def delete_user(db: aiosqlite.Connection, user_id: str) -> bool:
    """Delete a user and all their sessions. Returns True if found."""
    cur = await db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    await db.commit()
    return (cur.rowcount or 0) > 0


# - Sessions ---------------------------------------------------------------

async def create_session(
    db: aiosqlite.Connection,
    token: str,
    user_id: str,
    role: str,
    expires_at: float,
) -> None:
    """Insert a new session token."""
    now = time.time()
    await db.execute(
        """
        INSERT INTO user_sessions (token, user_id, role, created_at, expires_at, last_used_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (token, user_id, role, now, expires_at, now),
    )
    await db.commit()


async def fetch_session(
    db: aiosqlite.Connection, token: str
) -> dict[str, Any] | None:
    """Return a valid (non-expired) session row, or None."""
    now = time.time()
    async with db.execute(
        "SELECT * FROM user_sessions WHERE token = ? AND expires_at > ?",
        (token, now),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    # Touch last_used_at (fire-and-forget; don't block on commit)
    await db.execute(
        "UPDATE user_sessions SET last_used_at = ? WHERE token = ?",
        (now, token),
    )
    await db.commit()
    return dict(row)


async def delete_session(db: aiosqlite.Connection, token: str) -> bool:
    """Invalidate a session. Returns True if found."""
    cur = await db.execute(
        "DELETE FROM user_sessions WHERE token = ?", (token,)
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def delete_user_sessions(db: aiosqlite.Connection, user_id: str) -> int:
    """Invalidate all sessions for a user. Returns the count deleted."""
    cur = await db.execute(
        "DELETE FROM user_sessions WHERE user_id = ?", (user_id,)
    )
    await db.commit()
    return cur.rowcount or 0


async def purge_expired_sessions(db: aiosqlite.Connection) -> int:
    """Delete all expired sessions. Returns count deleted."""
    cur = await db.execute(
        "DELETE FROM user_sessions WHERE expires_at <= ?", (time.time(),)
    )
    await db.commit()
    return cur.rowcount or 0


# ---------------------------------------------------------------------------
# TOTP 2FA
# ---------------------------------------------------------------------------

async def update_user_totp(
    db: aiosqlite.Connection,
    user_id: str,
    totp_secret: str | None,
    totp_enabled: bool,
) -> bool:
    """Update TOTP secret and enabled flag. Returns True if user found."""
    cur = await db.execute(
        "UPDATE users SET totp_secret = ?, totp_enabled = ? WHERE user_id = ?",
        (totp_secret, 1 if totp_enabled else 0, user_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def create_partial_session(
    db: aiosqlite.Connection,
    token: str,
    user_id: str,
    expires_at: float,
) -> None:
    """Insert a partial session (password OK, awaiting TOTP)."""
    await db.execute(
        "INSERT INTO partial_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, time.time(), expires_at),
    )
    await db.commit()


async def fetch_partial_session(
    db: aiosqlite.Connection, token: str
) -> dict[str, Any] | None:
    """Fetch a partial session if it exists and is not expired."""
    async with db.execute(
        "SELECT token, user_id, created_at, expires_at FROM partial_sessions "
        "WHERE token = ? AND expires_at > ?",
        (token, time.time()),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return dict(row)


async def delete_partial_session(
    db: aiosqlite.Connection, token: str
) -> bool:
    """Delete a partial session. Returns True if found."""
    cur = await db.execute(
        "DELETE FROM partial_sessions WHERE token = ?", (token,)
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def purge_expired_partial_sessions(db: aiosqlite.Connection) -> int:
    """Delete all expired partial sessions. Returns count deleted."""
    cur = await db.execute(
        "DELETE FROM partial_sessions WHERE expires_at <= ?", (time.time(),)
    )
    await db.commit()
    return cur.rowcount or 0


# ---------------------------------------------------------------------------
# OAuth accounts
# ---------------------------------------------------------------------------

async def fetch_oauth_account(
    db: aiosqlite.Connection,
    provider: str,
    provider_user_id: str,
) -> dict[str, Any] | None:
    """Look up a linked OAuth account."""
    async with db.execute(
        "SELECT * FROM oauth_accounts WHERE provider = ? AND provider_user_id = ?",
        (provider, provider_user_id),
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def create_oauth_account(
    db: aiosqlite.Connection,
    provider: str,
    provider_user_id: str,
    user_id: str,
    email: str | None = None,
) -> None:
    """Link an OAuth provider account to a local user."""
    await db.execute(
        "INSERT INTO oauth_accounts (provider, provider_user_id, user_id, email, linked_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (provider, provider_user_id, user_id, email, time.time()),
    )
    await db.commit()


async def delete_oauth_account(
    db: aiosqlite.Connection,
    provider: str,
    provider_user_id: str,
) -> bool:
    """Unlink an OAuth account. Returns True if found."""
    cur = await db.execute(
        "DELETE FROM oauth_accounts WHERE provider = ? AND provider_user_id = ?",
        (provider, provider_user_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def fetch_oauth_accounts_for_user(
    db: aiosqlite.Connection, user_id: str
) -> list[dict[str, Any]]:
    """List all OAuth accounts linked to a user."""
    async with db.execute(
        "SELECT * FROM oauth_accounts WHERE user_id = ?", (user_id,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]
