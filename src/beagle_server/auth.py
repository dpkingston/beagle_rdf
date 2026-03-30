# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
User authentication helpers for the TDOA aggregation server.

Password hashing
----------------
Passwords are hashed with PBKDF2-HMAC-SHA256 using a 16-byte random salt,
260 000 iterations (OWASP 2023 recommendation), and a 32-byte derived key.
The stored value is a self-describing string:

    pbkdf2:sha256:<iterations>:<hex-salt>:<hex-dk>

This avoids any external dependency (uses Python stdlib ``hashlib`` and
``secrets`` only).

Session tokens
--------------
Each successful login generates a cryptographically random 32-byte token
(``secrets.token_urlsafe(32)``), stored in the ``user_sessions`` table with
a configurable expiry time.  Tokens are opaque to the client; the server
looks them up on every authenticated request.

Roles
-----
Two roles are supported:

  admin  -- full access: all management endpoints, user administration
  viewer -- read-only access: fixes, map, events, health (no writes)

The role is embedded in the session row so that role checks do not require a
separate users table lookup per request.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

import aiosqlite
from fastapi import HTTPException, Request, status

from beagle_server import db as db_module

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_ITERATIONS = 260_000
_DK_LEN = 32
_ALG = "sha256"


def hash_password(plaintext: str) -> str:
    """
    Return a stored hash string for *plaintext*.

    Format: ``pbkdf2:sha256:<iterations>:<hex-salt>:<hex-dk>``
    """
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(_ALG, plaintext.encode(), salt, _ITERATIONS, _DK_LEN)
    return f"pbkdf2:{_ALG}:{_ITERATIONS}:{salt.hex()}:{dk.hex()}"


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Return True if *plaintext* matches *stored_hash*.

    Handles both pbkdf2 hashes (userdb) and the legacy ``sha256:<hex>``
    format used by manage_nodes.py for node secrets.
    """
    # OAuth-only users have no password — always reject
    if stored_hash == "oauth:nologin":
        return False
    if stored_hash.startswith("pbkdf2:"):
        try:
            _, alg, iters_str, salt_hex, dk_hex = stored_hash.split(":")
            salt = bytes.fromhex(salt_hex)
            expected_dk = bytes.fromhex(dk_hex)
            iterations = int(iters_str)
        except (ValueError, AttributeError):
            return False
        candidate = hashlib.pbkdf2_hmac(alg, plaintext.encode(), salt, iterations, len(expected_dk))
        return secrets.compare_digest(candidate, expected_dk)
    # Legacy sha256 (node secrets)
    if stored_hash.startswith("sha256:"):
        expected = "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()
        return secrets.compare_digest(stored_hash, expected)
    return False


# ---------------------------------------------------------------------------
# TOTP 2FA
# ---------------------------------------------------------------------------

def generate_totp_secret() -> str:
    """Generate a random base32 TOTP secret."""
    import pyotp
    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, username: str) -> str:
    """Return an otpauth:// URI for QR code / manual entry."""
    import pyotp
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name="Beagle")


def verify_totp(secret: str, code: str) -> bool:
    """Verify a 6-digit TOTP code. Accepts current and +/-1 time window."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


# ---------------------------------------------------------------------------
# Session token
# ---------------------------------------------------------------------------

def generate_token() -> str:
    """Return a cryptographically random URL-safe token string (~43 chars)."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# FastAPI auth dependencies
# ---------------------------------------------------------------------------

def _extract_bearer(request: Request) -> str | None:
    """Return the Bearer token from the Authorization header, or None."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):]
    return None


async def get_current_user(
    request: Request,
    database: aiosqlite.Connection,
) -> dict[str, Any]:
    """
    Look up the current user from the session token in the Authorization header.

    Returns a dict with at least: ``user_id``, ``username``, ``role``.
    Raises HTTP 401 if the token is missing, invalid, or expired.
    """
    token = _extract_bearer(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session = await db_module.fetch_session(database, token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token invalid or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await db_module.fetch_user_by_id(database, session["user_id"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found",
        )
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "role": session["role"],  # use session role (may differ after role change)
        "_token": token,
    }


async def require_admin(
    request: Request,
    database: aiosqlite.Connection,
) -> dict[str, Any]:
    """
    Authenticate and require admin role.

    In ``userdb`` mode this validates the session token.
    In ``token`` mode this falls back to the shared auth_token check.
    In ``none`` mode this is a no-op and returns a synthetic admin principal.

    Raises 401/403 on auth failure.
    """
    from beagle_server.config import ServerFullConfig
    cfg: ServerFullConfig = request.app.state.config

    if cfg.server.auth_mode == "userdb":
        user = await get_current_user(request, database)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required",
            )
        return user

    if cfg.server.auth_mode == "none":
        return {"user_id": "system", "username": "system", "role": "admin"}

    # token / nodedb modes: fall back to shared auth_token
    token = cfg.server.auth_token
    if token:
        header = request.headers.get("Authorization", "")
        if header != f"Bearer {token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token",
            )
    return {"user_id": "system", "username": "system", "role": "admin"}


async def require_viewer(
    request: Request,
    database: aiosqlite.Connection,
) -> dict[str, Any]:
    """
    Authenticate and accept any authenticated user (admin or viewer).

    In ``userdb`` mode requires a valid session token.
    In other modes passes through without authentication.
    """
    from beagle_server.config import ServerFullConfig
    cfg: ServerFullConfig = request.app.state.config

    if cfg.server.auth_mode == "userdb":
        return await get_current_user(request, database)

    return {"user_id": "system", "username": "system", "role": "admin"}
