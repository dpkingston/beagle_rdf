# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
FastAPI application for the TDOA aggregation server.

Routes
------
POST /api/v1/events               Ingest a CarrierEvent; returns 201 + event_id
GET  /api/v1/events               List recent events (query: limit, node_id, channel_hz)
GET  /api/v1/fixes                List recent fixes (query: limit, max_age_s)
GET  /api/v1/fixes/{fix_id}       Get a specific fix by ID
GET  /health                      Server health: uptime, event_count, fix_count, last_fix_age_s
GET  /map                         Folium HTML map (query: max_age_s)
GET  /map/heatmap                 JSON heatmap cells for live client-side update
GET  /api/v1/fixes/stream         SSE stream - emits a "new_fix" event on each computed fix
DELETE /api/v1/fixes              Delete all stored fixes (auth-gated if token configured)
POST /api/v1/heartbeat            Node heartbeat (position, health); no auth required
GET  /map/nodes                   Merged node list (registered + event + heartbeat); no auth

User authentication (userdb mode):
POST /auth/register               Create a user account (open if no users exist; admin-only otherwise)
POST /auth/login                  Authenticate and receive a session token
POST /auth/logout                 Invalidate the current session token
GET  /auth/me                     Return current user info
GET  /auth/users                  List all users (admin only)
PATCH /auth/users/{user_id}       Change role or password (admin or own password)
DELETE /auth/users/{user_id}      Delete a user account (admin only)

Node management (auth-gated):
POST /api/v1/nodes/register              Node self-registration (node-secret auth)
POST /api/v1/nodes                       Admin-create node; returns plaintext secret (admin)
GET  /api/v1/nodes                       List all registered nodes (admin)
GET  /api/v1/nodes/{node_id}             Get node details (admin)
PATCH /api/v1/nodes/{node_id}            Update node enabled/label/config_json (admin)
POST /api/v1/nodes/{node_id}/regen-secret  Regenerate node secret (admin)
GET  /api/v1/nodes/{node_id}/config      Fetch assigned config; supports long-poll
                                         via ?wait=N&since_version=V (node-secret auth)
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import secrets
import time
from typing import Any

_logger = logging.getLogger(__name__)

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse


class _SSEResponse(StreamingResponse):
    """StreamingResponse that silently absorbs CancelledError on server shutdown.

    Starlette's StreamingResponse runs a parallel ``listen_for_disconnect``
    task.  When uvicorn shuts down it cancels all running tasks; the
    CancelledError from that internal task propagates up through the ASGI
    stack and is logged as "Exception in ASGI application" even though the
    server exits cleanly.  Catching it here keeps the shutdown log noise-free.
    """

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            pass

from beagle_node.events.model import CarrierEvent
from beagle_server import db as db_module
from beagle_server import auth as auth_module
from beagle_server.config import ServerFullConfig
from beagle_server.map_output import build_fix_geojson, build_map
from beagle_server.pairing import EventPairer
from beagle_server.solver import FixResult, solve_fix


def _verify_secret_hash(plaintext: str, stored_hash: str) -> bool:
    """
    Verify a plaintext secret against a stored hash.

    Supports ``sha256:<hex>`` hashes produced by manage_nodes.py.

    Node secrets intentionally use a plain SHA-256 (no salt, no iterations) because
    they are server-generated random 256-bit values, not user-chosen passwords.
    The security property comes from the secret's entropy (not the hash strength).
    User passwords - which are client-chosen and potentially weak - use PBKDF2-HMAC-SHA256
    with 260,000 iterations and a random salt; see ``auth.py:hash_password()``.
    """
    if stored_hash.startswith("sha256:"):
        expected = "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()
        return stored_hash == expected
    return False

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(config: ServerFullConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Beagle Aggregation Server",
        description="Receives CarrierEvents from Beagle nodes and computes fixes.",
        version="1.0.0",
    )
    app.state.config = config
    app.state.start_time = time.time()
    app.state.pairer = None           # set in lifespan
    app.state.db = None               # set in lifespan
    app.state.sse_subscribers: dict[int, asyncio.Queue[str]] = {}
    # node_auth and user_auth are read directly from config; no runtime state needed
    app.state.heartbeats: dict[str, dict[str, Any]] = {}  # node_id -> heartbeat info
    app.state.known_nodes: dict[str, dict[str, Any]] = {}  # node_id -> cached registry row
    app.state.map_geojson_cache: dict[float, dict] = {}   # max_age_s -> GeoJSON; cleared on new fix
    # Per-node config-file reload status for UI surfacing.  Updated each
    # time the long-poll handler stats the file.  See
    # db.maybe_reload_node_config() for the result-dict shape; we also add
    # a "checked_at" timestamp.  Bounded in size by the number of nodes,
    # so no eviction policy needed.
    app.state.config_reload_status: dict[str, dict[str, Any]] = {}
    # Per-node sliding window rate limiter: node_id -> deque of timestamps
    app.state.node_event_times: dict[str, collections.deque] = {}
    # Per-node rate-limit log cooldown: node_id -> (last_log_monotonic, count_since_last).
    # When a node trips the rate limit we emit one WARNING that summarises the
    # burst; further 429s from the same node are counted but not logged until
    # the cooldown expires.  This prevents a misbehaving node from drowning
    # the server log with one WARNING per rejected POST.
    app.state.rate_limit_log_state: dict[str, tuple[float, int]] = {}

    # -------------------------------------------------------------------
    # Lifespan - open DB, wire pairer callback
    # -------------------------------------------------------------------
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        cfg: ServerFullConfig = app.state.config
        database = await db_module.open_db(cfg.database.path)
        registry_db = await db_module.open_registry_db(cfg.database.registry_path)
        app.state.db = database
        app.state.registry_db = registry_db

        async def on_group_ready(events: list[dict[str, Any]]) -> None:
            """Called by EventPairer when a delivery-buffer group is ready."""
            node_ids  = [e["node_id"]  for e in events]
            event_ids = [e["event_id"] for e in events]
            channel_hz = events[0]["channel_hz"] if events else 0.0
            onset_ns   = events[0]["onset_time_ns"] if events else 0

            _logger.debug(
                "group ready: channel=%.4f MHz  events=%d  nodes=%s  event_ids=%s  onset_ns=%d",
                channel_hz / 1e6, len(events), node_ids, event_ids, onset_ns,
            )

            loop = asyncio.get_event_loop()
            fix = await loop.run_in_executor(
                None,
                lambda: solve_fix(
                    events=events,
                    search_center_lat=cfg.solver.search_center_lat,
                    search_center_lon=cfg.solver.search_center_lon,
                    search_radius_km=cfg.solver.search_radius_km,
                    min_xcorr_snr=cfg.solver.min_xcorr_snr,
                    max_xcorr_baseline_km=cfg.solver.max_xcorr_baseline_km,
                    savgol_window_us=cfg.solver.savgol_window_us,
                ),
            )

            if fix is None:
                _logger.warning(
                    "fix FAILED (solver returned None): channel=%.4f MHz  "
                    "events=%d  nodes=%s  onset_ns=%d",
                    channel_hz / 1e6, len(events), node_ids, onset_ns,
                )
                return

            # Reject fixes whose residual exceeds the configured ceiling.
            max_res = cfg.solver.max_residual_ns
            if max_res > 0 and fix.residual_ns > max_res:
                _logger.warning(
                    "fix REJECTED (residual too large): residual=%.0f ns > max=%.0f ns  "
                    "channel=%.4f MHz  lat=%.5f  lon=%.5f  nodes=%s  onset_ns=%d",
                    fix.residual_ns, max_res, fix.channel_hz / 1e6,
                    fix.latitude_deg, fix.longitude_deg, node_ids, fix.onset_time_ns,
                )
                return

            fix_id = await db_module.insert_fix(database, _fix_to_dict(fix))
            fix_type = "LOP" if fix.node_count == 2 else "fix"
            _logger.info(
                "%s #%d: channel=%.4f MHz  lat=%.5f  lon=%.5f  "
                "residual=%.0f ns  nodes=%d %s  onset_ns=%d  event_ids=%s",
                fix_type, fix_id, fix.channel_hz / 1e6, fix.latitude_deg, fix.longitude_deg,
                fix.residual_ns, fix.node_count, node_ids, fix.onset_time_ns, event_ids,
            )

            # Only accumulate heatmap for 3+ node fixes (unique position).
            # 2-node LOPs are lines, not points - their position is arbitrary.
            if fix.node_count >= 3:
                await db_module.add_fix_to_heatmap(
                    database,
                    fix.latitude_deg,
                    fix.longitude_deg,
                    cfg.map.heatmap_cell_m,
                    cfg.map.heatmap_sigma_cells,
                )
            # Invalidate the /map/data GeoJSON cache so the next poll rebuilds.
            app.state.map_geojson_cache.clear()

            # Notify any open SSE map connections that a new fix is ready.
            msg = json.dumps({"fix_id": fix_id, "channel_hz": fix.channel_hz})
            dead: list[int] = []
            for sub_id, q in app.state.sse_subscribers.items():
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    dead.append(sub_id)
            for sub_id in dead:
                app.state.sse_subscribers.pop(sub_id, None)

        pairer = EventPairer(
            correlation_window_s=cfg.pairing.correlation_window_s,
            delivery_buffer_s=cfg.pairing.delivery_buffer_s,
            group_expiry_s=cfg.pairing.group_expiry_s,
            freq_tolerance_hz=cfg.pairing.freq_tolerance_hz,
            min_nodes=cfg.pairing.min_nodes,
            on_group_ready=on_group_ready,
        )
        app.state.pairer = pairer

        # Start event loop watchdog - logs diagnostics if the loop blocks.
        from beagle_server.watchdog import start_watchdog, stop_watchdog
        watchdog = start_watchdog(asyncio.get_event_loop(), threshold_s=2.0)

        yield

        stop_watchdog(watchdog)
        await database.close()
        await registry_db.close()

    app.router.lifespan_context = lifespan

    # -------------------------------------------------------------------
    # Auth helpers
    # -------------------------------------------------------------------
    def _check_auth(request: Request) -> None:
        """
        Verify the shared Bearer token.  Used by management endpoints when
        user_auth is 'token', and by event ingest when node_auth is 'token'.
        In userdb mode, callers should use ``require_admin`` instead.
        """
        cfg: ServerFullConfig = request.app.state.config
        if cfg.server.user_auth == "userdb":
            # Caller is responsible for using require_admin; this path is only
            # reached for endpoints that have not been updated to use it yet.
            return
        token = cfg.server.auth_token
        if not token:
            return
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token",
            )

    async def _check_node_auth(
        request: Request,
        registry_db: aiosqlite.Connection,
    ) -> dict[str, Any]:
        """
        Authenticate a node request using its individual secret.

        The node must send:
            Authorization: Bearer <node_secret>
            X-Node-ID: <node_id>

        Returns the node row dict on success.  Raises 401/403 on failure.
        """
        node_id = request.headers.get("X-Node-ID", "").strip()
        auth_header = request.headers.get("Authorization", "")
        if not node_id or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Node-ID or Authorization header",
            )
        secret = auth_header[len("Bearer "):]
        node = await db_module.fetch_node(registry_db, node_id)
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown node '{node_id}'. Register via manage_nodes.py first.",
            )
        if not _verify_secret_hash(secret, node["secret_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid node secret",
            )
        return node

    # -------------------------------------------------------------------
    # DB dependencies
    # -------------------------------------------------------------------
    def get_db(request: Request) -> aiosqlite.Connection:
        """Operational DB: events, fixes, heatmap."""
        return request.app.state.db  # type: ignore[no-any-return]

    def get_registry_db(request: Request) -> aiosqlite.Connection:
        """Registry DB: nodes, node_config_history, users, user_sessions."""
        return request.app.state.registry_db  # type: ignore[no-any-return]

    # -------------------------------------------------------------------
    # POST /auth/register
    # Create a user account.  Open while the users table is empty (bootstrap);
    # requires admin auth once any user exists.
    # -------------------------------------------------------------------
    @app.post("/auth/register", status_code=status.HTTP_201_CREATED)
    async def auth_register(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """
        Register a new user account.

        Request body (JSON):
          { "username": "...", "password": "...", "role": "admin"|"viewer" }

        The first registration is always open (bootstraps the system when no
        users exist yet).  Subsequent registrations require the caller to be
        authenticated as an admin.

        Role defaults to "viewer" if not specified.
        """
        body: dict[str, str] = await request.json()
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        role     = (body.get("role") or "viewer").strip()

        if not username:
            raise HTTPException(status_code=422, detail="username is required")
        if len(password) < 8:
            raise HTTPException(status_code=422, detail="password must be at least 8 characters")
        if role not in ("admin", "viewer"):
            raise HTTPException(status_code=422, detail="role must be 'admin' or 'viewer'")

        user_count = await db_module.count_users(registry_db)
        if user_count > 0:
            # Not the first user - must be authenticated as admin
            await auth_module.require_admin(request, registry_db)

        # Check for duplicate username
        existing = await db_module.fetch_user_by_username(registry_db, username)
        if existing is not None:
            raise HTTPException(status_code=409, detail=f"Username '{username}' already exists")

        import uuid
        user_id = str(uuid.uuid4())
        password_hash = auth_module.hash_password(password)
        await db_module.create_user(registry_db, user_id, username, password_hash, role)

        _logger.info("User registered: username=%s role=%s user_id=%s", username, role, user_id)
        return {"user_id": user_id, "username": username, "role": role}

    # -------------------------------------------------------------------
    # POST /auth/login
    # -------------------------------------------------------------------
    @app.post("/auth/login")
    async def auth_login(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """
        Authenticate with username + password and receive a session token.

        Request body (JSON):
          { "username": "...", "password": "..." }

        Response (normal):
          { "token": "...", "user_id": "...", "username": "...",
            "role": "...", "expires_at": <unix timestamp float> }

        Response (2FA required):
          { "requires_2fa": true, "partial_token": "..." }
        """
        body: dict[str, str] = await request.json()
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""

        user = await db_module.fetch_user_by_username(registry_db, username)
        if user is None or not auth_module.verify_password(password, user["password_hash"]):
            # Same error for missing user and wrong password (avoids user enumeration)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # If 2FA is enabled, return a partial token instead of a full session.
        if user.get("totp_enabled"):
            partial_token = auth_module.generate_token()
            partial_expires = time.time() + 300.0  # 5 minutes
            await db_module.create_partial_session(
                registry_db, partial_token, user["user_id"], partial_expires
            )
            return {
                "requires_2fa": True,
                "partial_token": partial_token,
            }

        cfg: ServerFullConfig = request.app.state.config
        lifetime_s = cfg.server.session_lifetime_hours * 3600.0
        expires_at = time.time() + lifetime_s
        token = auth_module.generate_token()
        await db_module.create_session(registry_db, token, user["user_id"], user["role"], expires_at)
        await db_module.update_user_last_login(registry_db, user["user_id"])

        _logger.info("User logged in: username=%s role=%s", username, user["role"])
        return {
            "token":      token,
            "user_id":    user["user_id"],
            "username":   user["username"],
            "role":       user["role"],
            "expires_at": expires_at,
        }

    # -------------------------------------------------------------------
    # POST /auth/logout
    # -------------------------------------------------------------------
    @app.post("/auth/logout")
    async def auth_logout(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Invalidate the current session token."""
        user = await auth_module.get_current_user(request, registry_db)
        await db_module.delete_session(registry_db, user["_token"])
        _logger.info("User logged out: username=%s", user["username"])
        return {"status": "logged_out"}

    # -------------------------------------------------------------------
    # GET /auth/me
    # -------------------------------------------------------------------
    @app.get("/auth/me")
    async def auth_me(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Return information about the authenticated user."""
        user = await auth_module.get_current_user(request, registry_db)
        return {
            "user_id":  user["user_id"],
            "username": user["username"],
            "role":     user["role"],
        }

    # -------------------------------------------------------------------
    # GET /auth/users  - list all users (admin only)
    # -------------------------------------------------------------------
    @app.get("/auth/users")
    async def auth_list_users(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> list[dict[str, str | float | None]]:
        """List all user accounts. Requires admin role."""
        await auth_module.require_admin(request, registry_db)
        users = await db_module.fetch_all_users(registry_db)
        _strip = {"password_hash", "totp_secret"}
        return [{k: v for k, v in u.items() if k not in _strip} for u in users]

    # -------------------------------------------------------------------
    # PATCH /auth/users/{user_id}  - change role or password
    # -------------------------------------------------------------------
    @app.patch("/auth/users/{user_id}")
    async def auth_patch_user(
        user_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """
        Change a user's role or password.

        Admins may change any user's role or password.
        Non-admins may only change their own password (not role).

        Request body (JSON, all fields optional):
          { "role": "admin"|"viewer", "password": "..." }
        """
        caller = await auth_module.get_current_user(request, registry_db)
        body: dict[str, str] = await request.json()

        target = await db_module.fetch_user_by_id(registry_db, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")

        result: dict[str, str] = {"user_id": user_id}

        if "role" in body:
            if caller["role"] != "admin":
                raise HTTPException(status_code=403, detail="Only admins may change roles")
            new_role = body["role"]
            if new_role not in ("admin", "viewer"):
                raise HTTPException(status_code=422, detail="role must be 'admin' or 'viewer'")
            await db_module.update_user_role(registry_db, user_id, new_role)
            result["role"] = new_role

        if "password" in body:
            if caller["role"] != "admin" and caller["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Cannot change another user's password")
            new_pw = body["password"]
            if len(new_pw) < 8:
                raise HTTPException(status_code=422, detail="password must be at least 8 characters")
            await db_module.update_user_password(registry_db, user_id, auth_module.hash_password(new_pw))
            # Invalidate all existing sessions for this user
            await db_module.delete_user_sessions(registry_db, user_id)
            result["sessions_revoked"] = "true"

        return result

    # -------------------------------------------------------------------
    # DELETE /auth/users/{user_id}  - delete a user (admin only)
    # -------------------------------------------------------------------
    @app.delete("/auth/users/{user_id}")
    async def auth_delete_user(
        user_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Delete a user account and all their sessions. Admin only."""
        await auth_module.require_admin(request, registry_db)
        deleted = await db_module.delete_user(registry_db, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")
        _logger.info("User deleted: user_id=%s", user_id)
        return {"user_id": user_id, "deleted": "true"}

    # -------------------------------------------------------------------
    # POST /auth/2fa/setup  - generate TOTP secret for current user
    # -------------------------------------------------------------------
    @app.post("/auth/2fa/setup")
    async def auth_2fa_setup(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Generate a TOTP secret. Does not enable 2FA until /auth/2fa/enable."""
        user = await auth_module.get_current_user(request, registry_db)
        secret = auth_module.generate_totp_secret()
        uri = auth_module.totp_provisioning_uri(secret, user["username"])
        # Store secret but leave totp_enabled=False until verified
        await db_module.update_user_totp(registry_db, user["user_id"], secret, False)
        return {"secret": secret, "otpauth_uri": uri}

    # -------------------------------------------------------------------
    # POST /auth/2fa/enable  - verify TOTP code and activate 2FA
    # -------------------------------------------------------------------
    @app.post("/auth/2fa/enable")
    async def auth_2fa_enable(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Verify a TOTP code against the stored secret and enable 2FA."""
        user = await auth_module.get_current_user(request, registry_db)
        body: dict[str, str] = await request.json()
        code = (body.get("code") or "").strip()
        if not code:
            raise HTTPException(status_code=422, detail="Code is required")

        # Re-fetch user to get totp_secret
        full_user = await db_module.fetch_user_by_id(registry_db, user["user_id"])
        if not full_user or not full_user.get("totp_secret"):
            raise HTTPException(status_code=400, detail="Call /auth/2fa/setup first")

        if not auth_module.verify_totp(full_user["totp_secret"], code):
            raise HTTPException(status_code=400, detail="Invalid code")

        await db_module.update_user_totp(
            registry_db, user["user_id"], full_user["totp_secret"], True
        )
        _logger.info("2FA enabled for user %s", user["username"])
        return {"status": "2fa_enabled"}

    # -------------------------------------------------------------------
    # POST /auth/2fa/verify  - complete login with TOTP code
    # -------------------------------------------------------------------
    @app.post("/auth/2fa/verify")
    async def auth_2fa_verify(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str | float]:
        """Exchange a partial token + TOTP code for a full session token."""
        body: dict[str, str] = await request.json()
        partial_token = (body.get("partial_token") or "").strip()
        code = (body.get("code") or "").strip()

        if not partial_token or not code:
            raise HTTPException(status_code=422, detail="partial_token and code are required")

        partial = await db_module.fetch_partial_session(registry_db, partial_token)
        if partial is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired partial token",
            )

        user = await db_module.fetch_user_by_id(registry_db, partial["user_id"])
        if user is None or not user.get("totp_secret"):
            await db_module.delete_partial_session(registry_db, partial_token)
            raise HTTPException(status_code=401, detail="User not found")

        if not auth_module.verify_totp(user["totp_secret"], code):
            raise HTTPException(status_code=401, detail="Invalid code")

        # Delete partial session and create full session
        await db_module.delete_partial_session(registry_db, partial_token)
        cfg: ServerFullConfig = request.app.state.config
        lifetime_s = cfg.server.session_lifetime_hours * 3600.0
        expires_at = time.time() + lifetime_s
        token = auth_module.generate_token()
        await db_module.create_session(
            registry_db, token, user["user_id"], user["role"], expires_at
        )
        await db_module.update_user_last_login(registry_db, user["user_id"])

        _logger.info("2FA login completed: username=%s", user["username"])
        return {
            "token":      token,
            "user_id":    user["user_id"],
            "username":   user["username"],
            "role":       user["role"],
            "expires_at": expires_at,
        }

    # -------------------------------------------------------------------
    # POST /auth/2fa/disable  - disable 2FA (admin for any, self with code)
    # -------------------------------------------------------------------
    @app.post("/auth/2fa/disable")
    async def auth_2fa_disable(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """
        Disable 2FA for a user.

        Admin can disable any user's 2FA by passing {"user_id": "..."}.
        Self-disable requires {"password": "...", "code": "..."}.
        """
        caller = await auth_module.get_current_user(request, registry_db)
        body: dict[str, str] = await request.json()
        target_user_id = body.get("user_id") or caller["user_id"]

        if target_user_id != caller["user_id"]:
            # Must be admin to disable another user's 2FA
            if caller["role"] != "admin":
                raise HTTPException(status_code=403, detail="Admin role required")
        else:
            # Self-disable: require password + TOTP code
            password = body.get("password") or ""
            code = (body.get("code") or "").strip()
            full_user = await db_module.fetch_user_by_id(registry_db, caller["user_id"])
            if not full_user:
                raise HTTPException(status_code=404, detail="User not found")
            if not auth_module.verify_password(password, full_user["password_hash"]):
                raise HTTPException(status_code=401, detail="Invalid password")
            if full_user.get("totp_secret") and not auth_module.verify_totp(
                full_user["totp_secret"], code
            ):
                raise HTTPException(status_code=401, detail="Invalid code")

        await db_module.update_user_totp(registry_db, target_user_id, None, False)
        _logger.info("2FA disabled for user_id=%s by %s", target_user_id, caller["username"])
        return {"user_id": target_user_id, "status": "2fa_disabled"}

    # -------------------------------------------------------------------
    # Google OAuth
    # -------------------------------------------------------------------

    # In-memory store for OAuth state tokens (short-lived, max 100 entries)
    _oauth_states: dict[str, float] = {}

    def _google_oauth_enabled() -> bool:
        return bool(config.server.google_client_id and config.server.google_client_secret)

    def _google_redirect_uri(request: Request) -> str:
        """Derive the OAuth callback URL from the current request.

        Includes the reverse-proxy root_path (e.g. /beagle) so the URL we
        register with Google matches the actual route inside our subpath
        deployment.  Without this, Google would redirect to
        https://example.com/auth/oauth/google/callback (no /beagle/ prefix)
        and Apache would route to nowhere.
        """
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
        root_path = request.scope.get("root_path", "") or ""
        return f"{scheme}://{host}{root_path}/auth/oauth/google/callback"

    @app.get("/auth/oauth/google")
    async def oauth_google_redirect(request: Request) -> RedirectResponse:
        """Redirect to Google's OAuth consent screen."""
        if not _google_oauth_enabled():
            raise HTTPException(status_code=404, detail="Google OAuth not configured")

        state = secrets.token_urlsafe(32)
        _oauth_states[state] = time.time() + 600  # 10 min expiry
        # Prune expired states
        now = time.time()
        expired = [k for k, v in _oauth_states.items() if v < now]
        for k in expired:
            del _oauth_states[k]

        from urllib.parse import urlencode
        params = urlencode({
            "client_id": config.server.google_client_id,
            "redirect_uri": _google_redirect_uri(request),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        })
        return RedirectResponse(
            url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}",
            status_code=302,
        )

    @app.get("/auth/oauth/google/callback")
    async def oauth_google_callback(
        request: Request,
        code: str = Query(default=""),
        state: str = Query(default=""),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> RedirectResponse:
        """Handle Google OAuth callback: exchange code, find/create user, redirect."""
        if not _google_oauth_enabled():
            raise HTTPException(status_code=404, detail="Google OAuth not configured")

        # Validate state
        if state not in _oauth_states or _oauth_states[state] < time.time():
            _oauth_states.pop(state, None)
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
        del _oauth_states[state]

        if not code:
            raise HTTPException(status_code=400, detail="Missing authorization code")

        import httpx
        # Exchange code for tokens
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": config.server.google_client_id,
                    "client_secret": config.server.google_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": _google_redirect_uri(request),
                },
            )
        if token_resp.status_code != 200:
            _logger.warning("Google token exchange failed: %s", token_resp.text)
            raise HTTPException(status_code=502, detail="Google token exchange failed")

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=502, detail="No access token from Google")

        # Fetch user info
        async with httpx.AsyncClient(timeout=10.0) as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch Google user info")

        ginfo = userinfo_resp.json()
        google_id = ginfo.get("id", "")
        email = ginfo.get("email", "")
        name = ginfo.get("name", "")

        if not google_id:
            raise HTTPException(status_code=502, detail="No user ID from Google")

        # Look up existing OAuth link
        oauth = await db_module.fetch_oauth_account(registry_db, "google", google_id)

        if oauth:
            # Existing linked user
            user = await db_module.fetch_user_by_id(registry_db, oauth["user_id"])
        else:
            # Try to match by email
            user = await db_module.fetch_user_by_username(registry_db, email) if email else None

            if not user:
                # Auto-create new user
                user_count = await db_module.count_users(registry_db)
                new_role = "admin" if user_count == 0 else "viewer"
                user_id = str(__import__("uuid").uuid4())
                username = email or name or f"google_{google_id}"
                await db_module.create_user(
                    registry_db, user_id, username, "oauth:nologin", new_role
                )
                user = await db_module.fetch_user_by_id(registry_db, user_id)
                _logger.info(
                    "OAuth auto-created user: username=%s role=%s", username, new_role
                )

            # Link the Google account
            assert user is not None
            await db_module.create_oauth_account(
                registry_db, "google", google_id, user["user_id"], email
            )

        if user is None:
            raise HTTPException(status_code=500, detail="User creation failed")

        # Reverse-proxy subpath prefix (empty when mounted at root).
        root_path = request.scope.get("root_path", "") or ""

        # Check if 2FA is required
        if user.get("totp_enabled"):
            partial_token = auth_module.generate_token()
            await db_module.create_partial_session(
                registry_db, partial_token, user["user_id"], time.time() + 300.0
            )
            return RedirectResponse(
                url=f"{root_path}/map?pending_2fa={partial_token}",
                status_code=302,
            )

        # Create full session
        lifetime_s = config.server.session_lifetime_hours * 3600.0
        expires_at = time.time() + lifetime_s
        session_token = auth_module.generate_token()
        await db_module.create_session(
            registry_db, session_token, user["user_id"], user["role"], expires_at
        )
        await db_module.update_user_last_login(registry_db, user["user_id"])

        _logger.info("OAuth login: username=%s provider=google", user["username"])
        # Redirect to /map with token - JS will pick it up from the URL fragment
        return RedirectResponse(
            url=f"{root_path}/map?oauth_token={session_token}",
            status_code=302,
        )

    @app.get("/auth/oauth/accounts")
    async def oauth_list_accounts(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> list[dict[str, Any]]:
        """List OAuth accounts linked to the current user."""
        user = await auth_module.get_current_user(request, registry_db)
        accounts = await db_module.fetch_oauth_accounts_for_user(registry_db, user["user_id"])
        return accounts

    @app.delete("/auth/oauth/link/google")
    async def oauth_unlink_google(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """Unlink Google account from the current user."""
        user = await auth_module.get_current_user(request, registry_db)
        accounts = await db_module.fetch_oauth_accounts_for_user(registry_db, user["user_id"])
        google_acct = [a for a in accounts if a["provider"] == "google"]
        if not google_acct:
            raise HTTPException(status_code=404, detail="No Google account linked")
        await db_module.delete_oauth_account(
            registry_db, "google", google_acct[0]["provider_user_id"]
        )
        return {"status": "unlinked"}

    # -------------------------------------------------------------------
    # POST /api/v1/events
    # -------------------------------------------------------------------
    @app.post("/api/v1/events", status_code=status.HTTP_201_CREATED)
    async def ingest_event(
        event: CarrierEvent,
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        cfg: ServerFullConfig = request.app.state.config
        client_ip = request.client.host if request.client else None
        # The event itself carries the node's lat/lon in node_location.
        # Persist it to the registry alongside last_seen_at so the map can
        # reflect a relocated node immediately, even if the node hasn't
        # polled for config since the move.  Mobile / movable nodes are
        # supported by always taking the latest message's coordinates as
        # the current truth.
        ev_lat = event.node_location.latitude_deg
        ev_lon = event.node_location.longitude_deg
        if cfg.server.node_auth == "nodedb":
            node = await _check_node_auth(request, registry_db)
            await db_module.update_node_seen(
                registry_db, node["node_id"], client_ip,
                location_lat=ev_lat, location_lon=ev_lon,
            )
        else:
            if cfg.server.node_auth == "token":
                _check_auth(request)
            # Use cached node row to avoid DB round-trip on every event.
            # Cache is invalidated when enable/disable changes via PATCH.
            cached = request.app.state.known_nodes.get(event.node_id)
            if cached is not None:
                node = cached
            else:
                node = await db_module.ensure_node_exists(
                    registry_db, event.node_id, ip=client_ip,
                )
                request.app.state.known_nodes[event.node_id] = dict(node)
            await db_module.update_node_seen(
                registry_db, event.node_id, client_ip,
                location_lat=ev_lat, location_lon=ev_lon,
            )

        if not node["enabled"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Node '{node['node_id']}' is disabled",
            )

        # Per-node sliding window rate limit
        max_events = cfg.server.node_rate_limit_events
        window_s = cfg.server.node_rate_limit_window_s
        if max_events > 0:
            now = time.monotonic()
            times = request.app.state.node_event_times
            dq = times.get(event.node_id)
            if dq is None:
                dq = collections.deque()
                times[event.node_id] = dq
            # Expire old timestamps outside the window
            cutoff = now - window_s
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= max_events:
                # De-duplicate rate-limit logging: emit one WARNING on the
                # first rejection, then a summary "N more suppressed"
                # WARNING when the cooldown expires.  Cooldown = one full
                # window, so a sustained flood produces at most one line
                # per window per node (vs one per rejected POST).
                log_state = request.app.state.rate_limit_log_state
                prev = log_state.get(event.node_id)
                cooldown_expired = (prev is None
                                    or (now - prev[0]) >= window_s)
                if cooldown_expired:
                    suppressed = prev[1] if prev is not None else 0
                    if suppressed > 0:
                        _logger.warning(
                            "rate limit: node %s exceeded again "
                            "(%d more rejections suppressed during last %ds); "
                            "now %d events in %.0fs (limit %d/%ds)",
                            event.node_id, suppressed, int(window_s),
                            len(dq) + 1, window_s, max_events, int(window_s),
                        )
                    else:
                        _logger.warning(
                            "rate limit: node %s sent %d events in %.0fs "
                            "(limit %d/%ds); further rejections suppressed "
                            "until cooldown",
                            event.node_id, len(dq) + 1, window_s,
                            max_events, int(window_s),
                        )
                    log_state[event.node_id] = (now, 0)
                else:
                    log_state[event.node_id] = (prev[0], prev[1] + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit exceeded: {max_events} events per "
                        f"{int(window_s)}s for node '{event.node_id}'"
                    ),
                )
            dq.append(now)

        # Filter low-quality events
        if event.sync_corr_peak < cfg.pairing.min_corr_peak:
            _logger.debug(
                "event rejected (low corr_peak): peak=%.3f < %.3f  node=%s  event=%s",
                event.sync_corr_peak, cfg.pairing.min_corr_peak,
                event.node_id, event.event_id,
            )
            raise HTTPException(
                status_code=422,
                detail=f"sync_corr_peak {event.sync_corr_peak:.3f} below threshold {cfg.pairing.min_corr_peak}",
            )

        event_data = _carrier_event_to_db_dict(event)
        await db_module.upsert_event(database, event_data)

        pairer: EventPairer = request.app.state.pairer
        await pairer.add_event(event_data)

        return {"event_id": event.event_id}

    # -------------------------------------------------------------------
    # POST /api/v1/heartbeat
    # -------------------------------------------------------------------
    @app.post("/api/v1/heartbeat", status_code=200)
    async def post_heartbeat(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, str]:
        """
        Accept a heartbeat from a node.  No auth required - nodes announce
        themselves on startup and periodically so the map can display their
        position and health status even before any carrier events arrive.

        Body: {"node_id": str, "latitude_deg": float, "longitude_deg": float,
               "sdr_mode": str, ...}
        """
        body = await request.json()
        node_id = body.get("node_id")
        if not node_id:
            raise HTTPException(status_code=422, detail="node_id is required")
        client_ip = request.client.host if request.client else None
        # Shadow-register so the node appears in the registry and can be
        # enabled/disabled from the UI before any events arrive.
        await db_module.ensure_node_exists(registry_db, node_id, ip=client_ip)
        request.app.state.heartbeats[node_id] = {
            "node_id": node_id,
            "latitude_deg": body.get("latitude_deg"),
            "longitude_deg": body.get("longitude_deg"),
            "sdr_mode": body.get("sdr_mode"),
            "software_version": body.get("software_version"),
            "noise_floor_db": body.get("noise_floor_db"),
            "onset_threshold_db": body.get("onset_threshold_db"),
            "offset_threshold_db": body.get("offset_threshold_db"),
            "received_at": time.time(),
            "ip": client_ip,
        }
        _logger.debug("Heartbeat from %s", node_id)
        return {"status": "ok"}

    # -------------------------------------------------------------------
    # GET /api/v1/events
    # -------------------------------------------------------------------
    @app.get("/api/v1/events")
    async def list_events(
        limit: int = Query(default=100, ge=1, le=1000),
        node_id: str | None = None,
        channel_hz: float | None = None,
        database: aiosqlite.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        return await db_module.fetch_recent_events(database, limit=limit, node_id=node_id, channel_hz=channel_hz)

    # -------------------------------------------------------------------
    # GET /api/v1/fixes
    # -------------------------------------------------------------------
    @app.get("/api/v1/fixes")
    async def list_fixes(
        limit: int = Query(default=100, ge=1, le=1000),
        max_age_s: float = Query(default=0.0, ge=0.0),
        database: aiosqlite.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        return await db_module.fetch_fixes(database, limit=limit, max_age_s=max_age_s)

    # -------------------------------------------------------------------
    # GET /api/v1/fixes/stream  - Server-Sent Events
    # IMPORTANT: must be registered BEFORE /api/v1/fixes/{fix_id} so that
    # FastAPI does not swallow the literal path segment "stream" as fix_id.
    # -------------------------------------------------------------------
    @app.get("/api/v1/fixes/stream")
    async def fixes_stream(request: Request) -> StreamingResponse:
        """
        SSE stream that pushes a "new_fix" event whenever a fix is computed.

        The map page subscribes to this endpoint and reloads when it receives
        an event, giving live updates without manual refresh.

        Each event payload is a JSON object: {"fix_id": int, "channel_hz": float}
        A ": keepalive" comment is sent every 25 s to prevent proxy timeouts.
        """
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
        sub_id = id(queue)
        request.app.state.sse_subscribers[sub_id] = queue

        async def generate() -> Any:
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=25.0)
                        yield f"event: new_fix\ndata: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                request.app.state.sse_subscribers.pop(sub_id, None)

        return _SSEResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    # -------------------------------------------------------------------
    # GET /api/v1/fixes/{fix_id}
    # -------------------------------------------------------------------
    @app.get("/api/v1/fixes/{fix_id}")
    async def get_fix(
        fix_id: int,
        database: aiosqlite.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        fix = await db_module.fetch_fix_by_id(database, fix_id)
        if fix is None:
            raise HTTPException(status_code=404, detail="Fix not found")
        return fix

    # -------------------------------------------------------------------
    # GET /health
    # -------------------------------------------------------------------
    @app.get("/health")
    async def health(
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        uptime_s = time.time() - request.app.state.start_time
        event_count = await db_module.count_events(database)
        fix_count = await db_module.count_fixes(database)
        last_fix_age = await db_module.fetch_last_fix_age_s(database)
        pairer: EventPairer = request.app.state.pairer
        return {
            "status": "ok",
            "uptime_s": round(uptime_s, 1),
            "event_count": event_count,
            "fix_count": fix_count,
            "last_fix_age_s": round(last_fix_age, 1) if last_fix_age is not None else None,
            "pending_groups": pairer.pending_group_count(),
        }

    # -------------------------------------------------------------------
    # DELETE /api/v1/fixes
    # -------------------------------------------------------------------
    @app.delete("/api/v1/fixes")
    async def delete_fixes(
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, int]:
        """Delete all stored fixes and the accumulated heatmap. Auth-gated if token set."""
        await auth_module.require_admin(request, registry_db)
        deleted = await db_module.delete_all_fixes(database)
        heatmap_deleted = await db_module.delete_heatmap(database)
        return {"deleted": deleted, "heatmap_deleted": heatmap_deleted}

    # -------------------------------------------------------------------
    # DELETE /api/v1/heatmap
    # -------------------------------------------------------------------
    @app.delete("/api/v1/heatmap")
    async def delete_heatmap(
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, int]:
        """Clear accumulated heatmap data. Fix history is preserved."""
        await auth_module.require_admin(request, registry_db)
        deleted = await db_module.delete_heatmap(database)
        return {"deleted": deleted}

    # -------------------------------------------------------------------
    # GET /map/heatmap - JSON heatmap data for live client-side update
    #
    # Called by loadHeatmap() in the map page JS on every SSE new_fix event.
    # Returns {"cells": [[lat, lon, weight], ...]} which is passed directly
    # to the Leaflet.heat layer's setLatLngs() method.
    # -------------------------------------------------------------------
    @app.get("/map/heatmap")
    async def map_heatmap(
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
    ) -> JSONResponse:
        cfg: ServerFullConfig = request.app.state.config
        cells = await db_module.fetch_heatmap_cells(
            database, cell_size_m=cfg.map.heatmap_cell_m
        )
        return JSONResponse(content={"cells": cells})

    # -------------------------------------------------------------------
    # GET /map/data - GeoJSON FeatureCollection for dynamic fix layer
    #
    # Served to the map page JS via fetch('/map/data?max_age_s=N').
    # Returns fix point features (color-faded by age) and hyperbola arc
    # features for the most recent fix.  The page re-fetches this on every
    # SSE new_fix event and on age-preset button clicks without a page reload.
    # -------------------------------------------------------------------
    @app.get("/map/data")
    async def map_data(
        request: Request,
        max_age_s: float = Query(default=-1.0, ge=-1.0),
        database: aiosqlite.Connection = Depends(get_db),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> JSONResponse:
        cfg: ServerFullConfig = request.app.state.config
        age = cfg.map.max_age_s if max_age_s < 0 else max_age_s
        cache: dict[float, dict] = request.app.state.map_geojson_cache
        if age not in cache:
            fixes = await db_module.fetch_fixes(database, limit=1000, max_age_s=age)
            events = await db_module.fetch_recent_events(database, limit=200)
            registered_nodes = await db_module.fetch_all_nodes(registry_db)
            cache[age] = build_fix_geojson(
                fixes=fixes,
                recent_events=events,
                max_age_s=age,
                hyperbola_points=cfg.map.hyperbola_points,
                heartbeats=request.app.state.heartbeats,
                registered_nodes=registered_nodes,
            )
        return JSONResponse(content=cache[age])

    # -------------------------------------------------------------------
    # GET /map
    # -------------------------------------------------------------------
    @app.get("/map", response_class=HTMLResponse)
    async def map_view(
        request: Request,
        max_age_s: float = Query(default=-1.0, ge=-1.0),
        database: aiosqlite.Connection = Depends(get_db),
    ) -> HTMLResponse:
        cfg: ServerFullConfig = request.app.state.config
        age = cfg.map.max_age_s if max_age_s < 0 else max_age_s
        fixes = await db_module.fetch_fixes(database, limit=1000, max_age_s=age)
        events = await db_module.fetch_recent_events(database, limit=200)
        heatmap_cells = await db_module.fetch_heatmap_cells(
            database, cell_size_m=cfg.map.heatmap_cell_m
        )
        # If launched with --root-path /beagle, all absolute URLs in the
        # rendered HTML/JS need that prefix.  Starlette stores it in the
        # request scope; "" means mounted at the document root.
        root_path = request.scope.get("root_path", "") or ""
        loop = asyncio.get_event_loop()
        html = await loop.run_in_executor(
            None,
            lambda: build_map(
                fixes=fixes,
                recent_events=events,
                max_age_s=age,
                center_lat=cfg.solver.search_center_lat,
                center_lon=cfg.solver.search_center_lon,
                server_label=request.headers.get("host", ""),
                heatmap_cells=heatmap_cells,
                auth_token=cfg.server.auth_token,
                user_auth=cfg.server.user_auth,
                google_oauth_enabled=bool(
                    config.server.google_client_id and config.server.google_client_secret
                ),
                root_path=root_path,
            ),
        )
        return HTMLResponse(content=html)

    # -------------------------------------------------------------------
    # POST /api/v1/nodes/register
    # Node self-registration (authenticated with per-node secret).
    # -------------------------------------------------------------------
    @app.post("/api/v1/nodes/register")
    async def register_node(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """
        Called by a node on first boot (or after bootstrap.yaml changes).
        The node authenticates with X-Node-ID + Bearer <secret>.

        Returns its current config, or {"status": "pending"} if no config
        has been assigned yet (admin must run: manage_nodes.py set-config).
        """
        node = await _check_node_auth(request, registry_db)
        node_id = node["node_id"]
        client_ip = request.client.host if request.client else None
        await db_module.update_node_seen(registry_db, node_id, client_ip)

        config_obj: Any = None
        if node["config_json"]:
            try:
                config_obj = json.loads(node["config_json"])
            except (json.JSONDecodeError, TypeError):
                config_obj = None

        return {
            "status": "pending" if config_obj is None else "ok",
            "node_id": node_id,
            "config_version": node["config_version"],
            "config": config_obj,
        }

    # -------------------------------------------------------------------
    # GET /api/v1/nodes  - list all nodes (admin)
    # -------------------------------------------------------------------
    @app.get("/api/v1/nodes")
    async def list_nodes(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> list[dict[str, Any]]:
        await auth_module.require_admin(request, registry_db)
        nodes = await db_module.fetch_all_nodes(registry_db)
        # Strip secret_hash from the response
        for n in nodes:
            n.pop("secret_hash", None)
        return nodes

    # -------------------------------------------------------------------
    # POST /api/v1/nodes  - admin creates a node and receives its secret
    #
    # IMPORTANT: must be registered BEFORE /api/v1/nodes/{node_id} so that
    # FastAPI does not swallow the literal path segment as node_id.
    # -------------------------------------------------------------------
    @app.post("/api/v1/nodes", status_code=201)
    async def admin_create_node(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Admin-creates a node and returns the plaintext secret (shown once)."""
        await auth_module.require_admin(request, registry_db)
        body: dict[str, Any] = await request.json()
        node_id = (body.get("node_id") or "").strip()
        if not node_id:
            raise HTTPException(status_code=422, detail="node_id is required")
        label = body.get("label") or None

        existing = await db_module.fetch_node(registry_db, node_id)
        if existing:
            raise HTTPException(status_code=409,
                                detail=f"Node '{node_id}' already exists")

        plaintext = secrets.token_hex(32)
        secret_hash = "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()
        await db_module.create_node(registry_db, node_id, secret_hash, label)

        return {"node_id": node_id, "label": label, "secret": plaintext}

    # -------------------------------------------------------------------
    # GET /api/v1/nodes/snr  - per-node signal quality stats (unauthenticated)
    #
    # IMPORTANT: must be registered BEFORE /api/v1/nodes/{node_id} so that
    # FastAPI does not swallow the literal path segment "snr" as node_id.
    # -------------------------------------------------------------------
    @app.get("/api/v1/nodes/snr")
    async def node_snr_stats(
        request: Request,
        window_s: float = Query(default=3600.0, ge=60.0, description="Lookback window in seconds"),
        database: aiosqlite.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        """
        Per-node, per-channel signal quality statistics computed from events
        received in the last `window_s` seconds (default 1 hour).

        Each node entry includes:
          - status: "ok" | "marginal" | "stale"
          - corr_peak stats (mean, min, p10) - FM pilot lock quality (0-1)
          - snr_db stats (mean, min, p10) - peak_power_db minus noise_floor_db
          - last_event_age_s, event_count, clock info

        "stale"    - no event received in the last 5 minutes.
        "marginal" - mean sync_corr_peak below pairing.marginal_corr_peak (default 0.5).
        "ok"       - recent events with good FM sync quality.

        snr_db fields are null for events captured before noise floor tracking
        was added (schema < 1.4 or noise_floor_db defaulted to 0).
        """
        cfg: ServerFullConfig = request.app.state.config
        since_ts = time.time() - window_s
        channel_rows = await db_module.fetch_node_snr_stats(database, since_ts)

        # Group channel rows by node_id and compute per-node status.
        from collections import defaultdict
        by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in channel_rows:
            by_node[row["node_id"]].append(row)

        _STALE_S = 300.0  # 5 minutes: node should be sending events continuously
        marginal_threshold = cfg.pairing.marginal_corr_peak

        result: list[dict[str, Any]] = []
        for node_id, channels in sorted(by_node.items()):
            last_event_age_s = min(c["last_event_age_s"] for c in channels)
            min_mean_corr = min(c["corr_peak_mean"] for c in channels)

            if last_event_age_s > _STALE_S:
                status = "stale"
            elif min_mean_corr < marginal_threshold:
                status = "marginal"
            else:
                status = "ok"

            result.append({
                "node_id": node_id,
                "node_lat": channels[0]["node_lat"],
                "node_lon": channels[0]["node_lon"],
                "status": status,
                "last_event_age_s": last_event_age_s,
                "channels": channels,
            })

        return result

    # -------------------------------------------------------------------
    # GET|POST /api/v1/nodes/{node_id}/config  - node config + heartbeat
    # -------------------------------------------------------------------
    @app.api_route("/api/v1/nodes/{node_id}/config", methods=["GET", "POST"])
    async def get_node_config(
        node_id: str,
        request: Request,
        wait: int = Query(default=0, ge=0, le=120),
        since_version: int = Query(default=0, ge=0),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> Any:
        """
        Return the node's assigned config JSON.

        With ?wait=N&since_version=V the server holds the connection for up
        to N seconds, returning immediately if the config version advances
        beyond V.  Returns HTTP 304 if the timeout expires with no update.

        POST requests carry heartbeat telemetry in the JSON body (noise_floor_db,
        onset_threshold_db, offset_threshold_db, sdr_mode, location, etc.).
        This merges the config poll and heartbeat into a single round trip.
        GET requests still work for backward compatibility or admin tools.

        The node authenticates with X-Node-ID + Bearer <secret>.
        """
        node = await _check_node_auth(request, registry_db)
        if node["node_id"] != node_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="node_id mismatch")

        client_ip = request.client.host if request.client else None

        # POST body carries heartbeat telemetry - update the in-memory
        # heartbeat store so /map/nodes reflects live carrier state, AND
        # persist the location to the registry so the map can place
        # markers immediately on page load even after a server restart.
        body_lat: float | None = None
        body_lon: float | None = None
        if request.method == "POST":
            try:
                body = await request.json()
            except Exception:
                body = {}
            body_lat = body.get("latitude_deg")
            body_lon = body.get("longitude_deg")
            request.app.state.heartbeats[node_id] = {
                "node_id": node_id,
                "latitude_deg": body_lat,
                "longitude_deg": body_lon,
                "sdr_mode": body.get("sdr_mode"),
                "software_version": body.get("software_version"),
                "noise_floor_db": body.get("noise_floor_db"),
                "onset_threshold_db": body.get("onset_threshold_db"),
                "offset_threshold_db": body.get("offset_threshold_db"),
                "received_at": time.time(),
                "ip": client_ip,
            }
        else:
            # GET: still refresh timestamp on existing heartbeat entry
            hb = request.app.state.heartbeats.get(node_id)
            if hb:
                hb["received_at"] = time.time()
                hb["ip"] = client_ip

        # Update last_seen_at + last_ip in the registry, AND persist the
        # location if the heartbeat body included one.  Always done after
        # the body parse so we have lat/lon to write.  Pass None for
        # missing coordinates so a partial heartbeat (e.g. node started
        # with location not yet configured) doesn't clobber a previously
        # good registry value.
        await db_module.update_node_seen(
            registry_db, node_id, client_ip,
            location_lat=body_lat, location_lon=body_lon,
        )

        from fastapi.responses import Response

        # Auto-reload from disk (once per request, before entering the
        # long-poll loop): if the operator has edited the node's config
        # file on disk since we last stat'd it, re-read, validate, and
        # update config_json + bump config_version BEFORE the version
        # comparison below decides whether to return immediately or hold
        # the long-poll.  This delivers config edits to nodes within one
        # poll cycle (~120 s worst case) without any manual reload step.
        # The file is the source of truth: if both the file and an API
        # PATCH have changed config_json, the file wins on the next poll.
        # On any reload error (missing file, parse failure, schema
        # validation failure) we leave the existing cached config in
        # place and surface the status in app.state.config_reload_status
        # for the Nodes panel UI to display.  See
        # db.maybe_reload_node_config() for details.
        node_for_reload = await db_module.fetch_node(registry_db, node_id)
        if node_for_reload is not None:
            try:
                reload_result = await db_module.maybe_reload_node_config(
                    registry_db, dict(node_for_reload),
                    changed_by="auto-reload",
                )
            except Exception as exc:
                # Defensive: never let a reload bug break the poll.
                _logger.exception(
                    "Unexpected error reloading config for %s: %s",
                    node_id, exc,
                )
                reload_result = {
                    "node_id": node_id,
                    "status": "error",
                    "message": f"internal: {exc}",
                }
            # Cache for /map/nodes UI surfacing.  Only log on state
            # transition (status changed since last poll) so a permanent
            # error doesn't spam the log on every poll.
            prev = request.app.state.config_reload_status.get(node_id, {})
            now_ts = time.time()
            new_state = {**reload_result, "checked_at": now_ts}
            if reload_result["status"] != prev.get("status"):
                if reload_result["status"] == "updated":
                    _logger.info(
                        "Config auto-reload: %s updated to v%d",
                        node_id, reload_result.get("new_version"),
                    )
                elif reload_result["status"] in ("missing", "parse_error",
                                                 "validation_error", "error"):
                    _logger.warning(
                        "Config auto-reload: %s -> %s: %s",
                        node_id, reload_result["status"],
                        reload_result.get("message", ""),
                    )
            request.app.state.config_reload_status[node_id] = new_state
            # Invalidate the event-ingest cache so the new enabled state
            # / config takes effect on the next event POST.
            if reload_result["status"] == "updated":
                request.app.state.known_nodes.pop(node_id, None)

        deadline = time.monotonic() + wait
        try:
            while True:
                node = await db_module.fetch_node(registry_db, node_id)
                if node is None:
                    raise HTTPException(status_code=404,
                                        detail="Node not found")

                if node["config_version"] > since_version or wait == 0:
                    config_obj = None
                    if node["config_json"]:
                        try:
                            config_obj = json.loads(node["config_json"])
                        except (json.JSONDecodeError, TypeError):
                            config_obj = None

                    # Merge frequency group plan (highest priority overlay).
                    # The same helper is used by the admin merged-config
                    # inspection endpoint so the two views never disagree.
                    if config_obj is not None and node.get("freq_group_id"):
                        grp = await db_module.fetch_freq_group(
                            registry_db, node["freq_group_id"])
                        config_obj = db_module.apply_freq_group_overlay(
                            config_obj, grp,
                        )

                    return {
                        "node_id": node_id,
                        "config_version": node["config_version"],
                        "config": config_obj,
                        "status": "pending" if config_obj is None else "ok",
                    }

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return Response(status_code=304)
                await asyncio.sleep(min(1.0, remaining))
        except asyncio.CancelledError:
            return Response(status_code=503)

    # -------------------------------------------------------------------
    # POST /api/v1/nodes/{node_id}/regen-secret  - regenerate node secret
    # -------------------------------------------------------------------
    @app.post("/api/v1/nodes/{node_id}/regen-secret")
    async def regen_node_secret(
        node_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Generate a new secret for an existing node.  Old secret stops working immediately."""
        await auth_module.require_admin(request, registry_db)
        node = await db_module.fetch_node(registry_db, node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        plaintext = secrets.token_hex(32)
        secret_hash = "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()
        await db_module.update_node_secret(registry_db, node_id, secret_hash)

        return {"node_id": node_id, "secret": plaintext}

    @app.get("/api/v1/nodes/{node_id}")
    async def get_node(
        node_id: str,
        request: Request,
        merged: bool = Query(
            default=False,
            description=(
                "If true, replace the raw config_json with the same "
                "merged config the long-poll endpoint would serve to "
                "the node, including the freq_group overlay.  Useful "
                "for confirming what a bootstrapped node will actually "
                "receive when its freq_group_id is set, since the raw "
                "config_json on its own does not show the overlay."
            ),
        ),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        await auth_module.require_admin(request, registry_db)
        node = await db_module.fetch_node(registry_db, node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        node.pop("secret_hash", None)

        if merged and node.get("config_json"):
            # Parse the raw config_json, apply the freq_group overlay,
            # and serialise back.  We replace the string field with the
            # merged JSON string so the response shape is unchanged for
            # callers that aren't expecting a dict.  A new
            # config_json_merged_from_raw flag tells the caller this is
            # the merged form, not the on-disk form.
            try:
                config_obj = json.loads(node["config_json"])
            except (json.JSONDecodeError, TypeError):
                config_obj = None
            if config_obj is not None and node.get("freq_group_id"):
                grp = await db_module.fetch_freq_group(
                    registry_db, node["freq_group_id"]
                )
                config_obj = db_module.apply_freq_group_overlay(config_obj, grp)
            if config_obj is not None:
                node["config_json"] = json.dumps(config_obj)
                node["config_merged"] = True

        return node

    # -------------------------------------------------------------------
    # PATCH /api/v1/nodes/{node_id}  - update node (admin)
    # Accepts: {"enabled": bool} and/or {"config_json": str|null}
    # -------------------------------------------------------------------
    @app.patch("/api/v1/nodes/{node_id}")
    async def patch_node(
        node_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        await auth_module.require_admin(request, registry_db)
        body: dict[str, Any] = await request.json()

        node = await db_module.fetch_node(registry_db, node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        result: dict[str, Any] = {"node_id": node_id}

        if "label" in body:
            await db_module.update_node_label(registry_db, node_id, body["label"] or None)
            result["label"] = body["label"]

        if "enabled" in body:
            await db_module.update_node_enabled(registry_db, node_id, bool(body["enabled"]))
            result["enabled"] = bool(body["enabled"])
            # Invalidate cached node so the event ingest path picks up the change.
            request.app.state.known_nodes.pop(node_id, None)

        if "config_json" in body:
            raw = body["config_json"]
            # Accept either a pre-serialised JSON string or a dict/object
            if isinstance(raw, dict):
                config_str: str | None = json.dumps(raw)
            elif isinstance(raw, str) or raw is None:
                config_str = raw
            else:
                raise HTTPException(status_code=422, detail="config_json must be a JSON object, string, or null")
            new_version = await db_module.update_node_config(
                registry_db, node_id, config_str,
                changed_by="api:PATCH",
                diff_note="updated via PATCH /api/v1/nodes",
            )
            result["config_version"] = new_version

        if "freq_group_id" in body:
            gid = body["freq_group_id"]
            if gid is not None:
                grp = await db_module.fetch_freq_group(registry_db, gid)
                if grp is None:
                    raise HTTPException(status_code=404,
                                        detail=f"Frequency group '{gid}' not found")
            new_version = await db_module.set_node_freq_group(
                registry_db, node_id, gid, changed_by="api:PATCH",
            )
            result["freq_group_id"] = gid
            result["config_version"] = new_version

        return result

    # -------------------------------------------------------------------
    # DELETE /api/v1/nodes/{node_id}  - remove node (admin)
    # -------------------------------------------------------------------
    @app.delete("/api/v1/nodes/{node_id}")
    async def delete_node_endpoint(
        node_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Delete a registered node. Auth-gated if token is set."""
        await auth_module.require_admin(request, registry_db)
        deleted = await db_module.delete_node(registry_db, node_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Node not found")
        request.app.state.known_nodes.pop(node_id, None)
        return {"node_id": node_id, "deleted": True}

    # -------------------------------------------------------------------
    # POST /api/v1/nodes/reload-configs
    # Stat config files for all nodes and reload any that have changed.
    # -------------------------------------------------------------------
    @app.post("/api/v1/nodes/reload-configs")
    async def reload_node_configs(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Reload node configs from disk for any files that have changed."""
        await auth_module.require_admin(request, registry_db)
        results = await db_module.reload_node_configs(registry_db)
        updated = [r for r in results if r["status"] == "updated"]
        # Invalidate cached nodes so event ingest picks up new configs
        for r in updated:
            request.app.state.known_nodes.pop(r["node_id"], None)
        return {"results": results, "updated": len(updated)}

    # -------------------------------------------------------------------
    # POST /api/v1/nodes/{node_id}/config/reload
    # Force-resync one node's config from its attached file, overwriting
    # any in-memory JSON edits made via the GUI editor or PATCH endpoint.
    # -------------------------------------------------------------------
    @app.post("/api/v1/nodes/{node_id}/config/reload")
    async def reload_node_config(
        node_id: str,
        request: Request,
        force: bool = False,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Reload one node's config from its config_file_path.

        ``force=true`` bypasses the mtime check so in-memory JSON edits
        are reverted to whatever the file currently specifies.  Without
        force the call behaves like the automatic poll (reloads only if
        mtime has advanced).
        """
        await auth_module.require_admin(request, registry_db)
        node_row = await db_module.fetch_node(registry_db, node_id)
        if node_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown node: {node_id}",
            )
        result = await db_module.maybe_reload_node_config(
            registry_db, dict(node_row),
            changed_by="admin-revert" if force else "admin-reload",
            force=force,
        )
        if result["status"] == "updated":
            request.app.state.known_nodes.pop(node_id, None)
        return result

    # -------------------------------------------------------------------
    # GET /api/v1/nodes/{node_id}/config/file
    # Read the raw content of the config file that would be loaded on
    # the next reload.  Used by the UI to detect divergence between the
    # in-memory JSON and the file on disk.
    # -------------------------------------------------------------------
    @app.get("/api/v1/nodes/{node_id}/config/file")
    async def read_node_config_file(
        node_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Return the current on-disk content of the node's config file.

        Response shape:
          {"node_id": str,
           "has_file": bool,
           "path": str | null,
           "exists": bool,
           "content": <parsed JSON/YAML as dict> | null,
           "raw": str | null,
           "error": str | null}

        ``content`` is the parsed object suitable for diffing against the
        stored ``config_json``.  On parse error the raw text is returned
        instead so the UI can show the operator what's broken.
        """
        await auth_module.require_admin(request, registry_db)
        node_row = await db_module.fetch_node(registry_db, node_id)
        if node_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown node: {node_id}",
            )
        file_path = node_row.get("config_file_path") if isinstance(
            node_row, dict
        ) else node_row["config_file_path"]
        if not file_path:
            return {
                "node_id": node_id,
                "has_file": False,
                "path": None,
                "exists": False,
                "content": None,
                "raw": None,
                "error": None,
            }
        import os as _os
        if not _os.path.exists(file_path):
            return {
                "node_id": node_id,
                "has_file": True,
                "path": file_path,
                "exists": False,
                "content": None,
                "raw": None,
                "error": "file not found",
            }
        try:
            with open(file_path) as _f:
                raw = _f.read()
        except OSError as exc:
            return {
                "node_id": node_id,
                "has_file": True,
                "path": file_path,
                "exists": True,
                "content": None,
                "raw": None,
                "error": f"read failed: {exc}",
            }
        try:
            if file_path.lower().endswith((".yaml", ".yml")):
                import yaml  # type: ignore[import]
                parsed = yaml.safe_load(raw)
            else:
                parsed = json.loads(raw)
        except Exception as exc:
            return {
                "node_id": node_id,
                "has_file": True,
                "path": file_path,
                "exists": True,
                "content": None,
                "raw": raw,
                "error": f"parse error: {exc}",
            }
        return {
            "node_id": node_id,
            "has_file": True,
            "path": file_path,
            "exists": True,
            "content": parsed,
            "raw": raw,
            "error": None,
        }

    # -------------------------------------------------------------------
    # Frequency groups CRUD (admin-gated)
    # -------------------------------------------------------------------

    @app.get("/api/v1/groups")
    async def list_groups(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> list[dict[str, Any]]:
        """List all frequency groups with member counts."""
        await auth_module.require_admin(request, registry_db)
        groups = await db_module.fetch_all_freq_groups(registry_db)
        for g in groups:
            g["target_channels"] = json.loads(g.pop("target_channels_json", "[]"))
            members = await db_module.fetch_group_member_ids(registry_db, g["group_id"])
            g["member_count"] = len(members)
            g["member_node_ids"] = members
        return groups

    @app.post("/api/v1/groups", status_code=201)
    async def create_group(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Create a new frequency group."""
        await auth_module.require_admin(request, registry_db)
        body: dict[str, Any] = await request.json()
        required = ["group_id", "label", "sync_freq_hz",
                     "sync_station_id", "sync_station_lat", "sync_station_lon",
                     "target_channels"]
        for field in required:
            if field not in body:
                raise HTTPException(status_code=422,
                                    detail=f"Missing required field: {field}")
        # Validate target_channels is a list of dicts with frequency_hz
        tc = body["target_channels"]
        if not isinstance(tc, list) or not tc:
            raise HTTPException(status_code=422,
                                detail="target_channels must be a non-empty list")
        for ch in tc:
            if not isinstance(ch, dict) or "frequency_hz" not in ch:
                raise HTTPException(status_code=422,
                                    detail="Each target_channel must have frequency_hz")

        existing = await db_module.fetch_freq_group(registry_db, body["group_id"])
        if existing:
            raise HTTPException(status_code=409,
                                detail=f"Group '{body['group_id']}' already exists")

        grp = await db_module.create_freq_group(
            registry_db,
            group_id=body["group_id"],
            label=body["label"],
            description=body.get("description"),
            sync_freq_hz=body["sync_freq_hz"],
            sync_station_id=body["sync_station_id"],
            sync_station_lat=body["sync_station_lat"],
            sync_station_lon=body["sync_station_lon"],
            target_channels_json=json.dumps(tc),
        )
        grp["target_channels"] = json.loads(grp.pop("target_channels_json", "[]"))
        grp["member_count"] = 0
        grp["member_node_ids"] = []
        return grp

    @app.get("/api/v1/groups/{group_id}")
    async def get_group(
        group_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Get a frequency group with its members."""
        await auth_module.require_admin(request, registry_db)
        grp = await db_module.fetch_freq_group(registry_db, group_id)
        if grp is None:
            raise HTTPException(status_code=404, detail="Group not found")
        grp["target_channels"] = json.loads(grp.pop("target_channels_json", "[]"))
        members = await db_module.fetch_group_member_ids(registry_db, group_id)
        grp["member_count"] = len(members)
        grp["member_node_ids"] = members
        return grp

    @app.patch("/api/v1/groups/{group_id}")
    async def patch_group(
        group_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Update a frequency group's label or frequency plan.
        Bumps config_version on all member nodes."""
        await auth_module.require_admin(request, registry_db)
        grp = await db_module.fetch_freq_group(registry_db, group_id)
        if grp is None:
            raise HTTPException(status_code=404, detail="Group not found")

        body: dict[str, Any] = await request.json()
        updates: dict[str, Any] = {}
        for key in ("label", "description", "sync_freq_hz", "sync_station_id",
                     "sync_station_lat", "sync_station_lon"):
            if key in body:
                updates[key] = body[key]
        if "target_channels" in body:
            tc = body["target_channels"]
            if not isinstance(tc, list) or not tc:
                raise HTTPException(status_code=422,
                                    detail="target_channels must be a non-empty list")
            updates["target_channels_json"] = json.dumps(tc)

        grp = await db_module.update_freq_group(registry_db, group_id, updates)
        if grp is None:
            raise HTTPException(status_code=404, detail="Group not found")

        # Bump all member nodes so they pick up the new frequency plan.
        freq_fields = {"sync_freq_hz", "sync_station_id", "sync_station_lat",
                       "sync_station_lon", "target_channels_json"}
        if updates.keys() & freq_fields:
            n = await db_module.bump_group_members_version(
                registry_db, group_id,
                changed_by="api:PATCH group",
                diff_note=f"freq group '{group_id}' updated",
            )
            _logger.info("Group %s updated; bumped %d member nodes", group_id, n)

        grp["target_channels"] = json.loads(grp.pop("target_channels_json", "[]"))
        members = await db_module.fetch_group_member_ids(registry_db, group_id)
        grp["member_count"] = len(members)
        grp["member_node_ids"] = members
        return grp

    @app.delete("/api/v1/groups/{group_id}")
    async def delete_group(
        group_id: str,
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Delete a frequency group. Member nodes become ungrouped."""
        await auth_module.require_admin(request, registry_db)
        deleted = await db_module.delete_freq_group(registry_db, group_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Group not found")
        return {"group_id": group_id, "deleted": True}

    # -------------------------------------------------------------------
    # GET /api/v1/settings  - read runtime-mutable server settings
    # PATCH /api/v1/settings - update runtime-mutable server settings (admin)
    # -------------------------------------------------------------------
    @app.get("/api/v1/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        """Return server auth settings (read-only)."""
        cfg: ServerFullConfig = request.app.state.config
        return {
            "node_auth": cfg.server.node_auth,
            "user_auth": cfg.server.user_auth,
        }

    @app.patch("/api/v1/settings")
    async def patch_settings(
        request: Request,
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> dict[str, Any]:
        """Reserved for future runtime-mutable settings. Auth-gated."""
        await auth_module.require_admin(request, registry_db)
        cfg: ServerFullConfig = request.app.state.config
        return {
            "node_auth": cfg.server.node_auth,
            "user_auth": cfg.server.user_auth,
        }

    # -------------------------------------------------------------------
    # GET /map/nodes - merged node status (no auth, read-only)
    #
    # Returns all nodes that have ever reported events or are registered.
    # Registered nodes have full controls (enable/disable/delete).
    # Event-only nodes are shown for awareness but have no controls.
    # -------------------------------------------------------------------
    @app.get("/map/nodes")
    async def map_nodes_data(
        request: Request,
        database: aiosqlite.Connection = Depends(get_db),
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> JSONResponse:
        """
        Merged node status from the nodes table (registered) and the events
        table (event-only).  No auth required - read-only.

        Response: {"nodes": [...], "server_time": float}
        Each node dict includes: node_id, label, location_lat, location_lon,
        enabled, last_seen_at, last_ip, sdr_mode, registered.
        Node positions are sourced from the most recent event, not the
        nodes table.
        """
        now = time.time()
        registered = await db_module.fetch_all_nodes(registry_db)
        event_summary = await db_module.fetch_event_node_summary(database)

        event_by_id: dict[str, dict[str, Any]] = {e["node_id"]: e for e in event_summary}
        registered_ids: set[str] = {n["node_id"] for n in registered}
        heartbeats: dict[str, dict[str, Any]] = request.app.state.heartbeats
        reload_status: dict[str, dict[str, Any]] = request.app.state.config_reload_status
        seen_ids: set[str] = set()

        def _config_status_summary(rs: dict[str, Any] | None) -> dict[str, Any] | None:
            """Trim the in-memory reload-status entry for client consumption."""
            if not rs:
                return None
            return {
                "status": rs.get("status"),
                "message": rs.get("message"),
                "checked_at": rs.get("checked_at"),
            }

        from beagle_server.map_output import resolve_node_location

        result: list[dict[str, Any]] = []
        for n in registered:
            n = dict(n)
            n.pop("secret_hash", None)
            n["registered"] = True
            # Location is the value from the most recent message we have
            # about this node, regardless of channel.  Each candidate
            # carries its own timestamp; the freshest one wins.  See
            # map_output.resolve_node_location() for the rule.  Nodes can
            # be physically moved, so a stale event row's coordinates
            # must NOT permanently override a fresher heartbeat or
            # registry-cached value.
            ev = event_by_id.get(n["node_id"])
            hb = heartbeats.get(n["node_id"])
            lat, lon, src, _ts = resolve_node_location(
                event_row=ev, heartbeat=hb, registry_row=n,
            )
            n["location_lat"] = lat
            n["location_lon"] = lon
            n["location_source"] = src
            n["sdr_mode"] = ev["sdr_mode"] if ev else (hb.get("sdr_mode") if hb else None)
            n["software_version"] = hb.get("software_version") if hb else None
            n["heartbeat_age_s"] = now - hb["received_at"] if hb else None
            n["noise_floor_db"] = hb.get("noise_floor_db") if hb else None
            n["onset_threshold_db"] = hb.get("onset_threshold_db") if hb else None
            n["offset_threshold_db"] = hb.get("offset_threshold_db") if hb else None
            n["config_reload"] = _config_status_summary(reload_status.get(n["node_id"]))
            seen_ids.add(n["node_id"])
            result.append(n)

        for e in event_summary:
            if e["node_id"] in seen_ids:
                continue  # already included above
            hb = heartbeats.get(e["node_id"])
            result.append({
                "node_id": e["node_id"],
                "label": None,
                "location_lat": e["node_lat"],
                "location_lon": e["node_lon"],
                "enabled": None,
                "last_seen_at": e["last_seen_at"],
                "last_ip": None,
                "sdr_mode": e["sdr_mode"],
                "registered": False,
                "config_version": None,
                "config_json": None,
                "config_template_id": None,
                "freq_group_id": None,
                "heartbeat_age_s": now - hb["received_at"] if hb else None,
                "noise_floor_db": hb.get("noise_floor_db") if hb else None,
                "onset_threshold_db": hb.get("onset_threshold_db") if hb else None,
                "offset_threshold_db": hb.get("offset_threshold_db") if hb else None,
            })
            seen_ids.add(e["node_id"])

        # Nodes known only from heartbeats (no events, not registered)
        for nid, hb in heartbeats.items():
            if nid in seen_ids:
                continue
            result.append({
                "node_id": nid,
                "label": None,
                "location_lat": hb.get("latitude_deg"),
                "location_lon": hb.get("longitude_deg"),
                "enabled": None,
                "last_seen_at": hb["received_at"],
                "last_ip": hb.get("ip"),
                "sdr_mode": hb.get("sdr_mode"),
                "registered": False,
                "config_version": None,
                "config_json": None,
                "config_template_id": None,
                "freq_group_id": None,
                "heartbeat_age_s": now - hb["received_at"],
                "noise_floor_db": hb.get("noise_floor_db"),
                "onset_threshold_db": hb.get("onset_threshold_db"),
                "offset_threshold_db": hb.get("offset_threshold_db"),
            })

        result.sort(key=lambda n: n["node_id"])
        return JSONResponse(content={"nodes": result, "server_time": now})

    # -------------------------------------------------------------------
    # GET /map/groups - frequency groups (no auth, read-only for UI)
    # -------------------------------------------------------------------
    @app.get("/map/groups")
    async def map_groups_data(
        registry_db: aiosqlite.Connection = Depends(get_registry_db),
    ) -> JSONResponse:
        """Return all frequency groups with member node IDs. No auth."""
        groups = await db_module.fetch_all_freq_groups(registry_db)
        for g in groups:
            g["target_channels"] = json.loads(g.pop("target_channels_json", "[]"))
            members = await db_module.fetch_group_member_ids(
                registry_db, g["group_id"]
            )
            g["member_count"] = len(members)
            g["member_node_ids"] = members
        return JSONResponse(content={"groups": groups})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carrier_event_to_db_dict(event: CarrierEvent) -> dict[str, Any]:
    """Flatten a CarrierEvent into the DB events table schema."""
    return {
        "event_id": event.event_id,
        "node_id": event.node_id,
        "channel_hz": event.channel_frequency_hz,
        "sync_delta_ns": event.sync_delta_ns,
        "sync_tx_id": event.sync_transmitter.station_id,
        "sync_tx_lat": event.sync_transmitter.latitude_deg,
        "sync_tx_lon": event.sync_transmitter.longitude_deg,
        "node_lat": event.node_location.latitude_deg,
        "node_lon": event.node_location.longitude_deg,
        "event_type": event.event_type,
        "onset_time_ns": event.onset_time_ns,
        "corr_peak": event.sync_corr_peak,
        "received_at": time.time(),
        "raw_json": json.dumps(event.model_dump(mode="json")),
        # IQ cross-correlation fields (schema 1.2+/1.3+).  Kept in the in-memory
        # event dict so the solver can use them without re-parsing raw_json.
        "iq_snippet_b64": event.iq_snippet_b64,
        "channel_sample_rate_hz": event.channel_sample_rate_hz,
        # Transition bounds for xcorr windowing
        "transition_start": event.transition_start,
        "transition_end": event.transition_end,
        # Sync diagnostics for cross-node verification
        "sync_pilot_phase_rad": event.sync_pilot_phase_rad,
        "sync_sample_index": event.sync_sample_index,
        "sync_delta_samples": event.sync_delta_samples,
        "sync_sample_rate_correction": event.sync_sample_rate_correction,
    }


def _fix_to_dict(fix: FixResult) -> dict[str, Any]:
    """Convert a FixResult dataclass to the DB fixes table schema."""
    return {
        "channel_hz": fix.channel_hz,
        "event_type": fix.event_type,
        "computed_at": time.time(),
        "latitude_deg": fix.latitude_deg,
        "longitude_deg": fix.longitude_deg,
        "residual_ns": fix.residual_ns,
        "node_count": fix.node_count,
        "nodes": fix.nodes,
        "onset_time_ns": fix.onset_time_ns,
    }
