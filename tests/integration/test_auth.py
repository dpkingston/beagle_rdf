# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration tests for the /auth/* endpoints in userdb auth mode.

All tests use an in-memory SQLite DB and a config with auth_mode="userdb".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from beagle_server.api import create_app
from beagle_server.config import (
    DatabaseConfig,
    MapConfig,
    PairingConfig,
    ServerConfig,
    ServerFullConfig,
    SolverConfig,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _userdb_config() -> ServerFullConfig:
    return ServerFullConfig(
        server=ServerConfig(
            host="127.0.0.1",
            port=8765,
            auth_token="",
            auth_mode="userdb",
            session_lifetime_hours=24.0,
        ),
        database=DatabaseConfig(path=":memory:", registry_path=":memory:"),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=0.05,
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
        ),
        map=MapConfig(output_dir="/tmp/tdoa_test_auth_maps", max_age_s=3600.0),
    )


@pytest.fixture()
def client():
    config = _userdb_config()
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def admin_client():
    """Client with a pre-registered admin user; yields (client, token)."""
    config = _userdb_config()
    app = create_app(config)
    with TestClient(app, raise_server_exceptions=True) as c:
        resp = c.post("/auth/register", json={
            "username": "admin",
            "password": "adminpass",
            "role": "admin",
        })
        assert resp.status_code == 201
        resp2 = c.post("/auth/login", json={
            "username": "admin",
            "password": "adminpass",
        })
        assert resp2.status_code == 200
        token = resp2.json()["token"]
        yield c, token


# ---------------------------------------------------------------------------
# POST /auth/register - bootstrap
# ---------------------------------------------------------------------------

class TestRegisterBootstrap:
    def test_first_user_open_no_auth_required(self, client: TestClient) -> None:
        resp = client.post("/auth/register", json={
            "username": "alice",
            "password": "alicepass",
            "role": "admin",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "alice"
        assert data["role"] == "admin"
        assert "user_id" in data

    def test_first_user_default_role_viewer(self, client: TestClient) -> None:
        resp = client.post("/auth/register", json={
            "username": "alice",
            "password": "alicepass",
        })
        assert resp.status_code == 201
        assert resp.json()["role"] == "viewer"

    def test_second_user_requires_admin_auth(self, client: TestClient) -> None:
        # Register first user (bootstrap)
        client.post("/auth/register", json={
            "username": "admin",
            "password": "adminpass",
            "role": "admin",
        })
        # Second registration without auth -> 401
        resp = client.post("/auth/register", json={
            "username": "bob",
            "password": "bobspassword",
            "role": "viewer",
        })
        assert resp.status_code == 401

    def test_second_user_as_admin_succeeds(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        login = client.post("/auth/login", json={"username": "admin", "password": "adminpass"})
        token = login.json()["token"]

        resp = client.post(
            "/auth/register",
            json={"username": "bob", "password": "bobspassword", "role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 201
        assert resp.json()["username"] == "bob"

    def test_duplicate_username_rejected(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        resp = client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        # Second bootstrap attempt - users table not empty -> 401 (no auth), not 409
        assert resp.status_code in (401, 409)

    def test_duplicate_as_admin_returns_409(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        login = client.post("/auth/login", json={"username": "alice", "password": "alicepass"})
        token = login.json()["token"]

        resp = client.post(
            "/auth/register",
            json={"username": "alice", "password": "newpassword", "role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 409

    def test_short_password_rejected(self, client: TestClient) -> None:
        resp = client.post("/auth/register", json={
            "username": "alice",
            "password": "short",
            "role": "admin",
        })
        assert resp.status_code == 422

    def test_invalid_role_rejected(self, client: TestClient) -> None:
        resp = client.post("/auth/register", json={
            "username": "alice",
            "password": "alicepass",
            "role": "superuser",
        })
        assert resp.status_code == 422

    def test_missing_username_rejected(self, client: TestClient) -> None:
        resp = client.post("/auth/register", json={"password": "alicepass", "role": "admin"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_login_success(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        resp = client.post("/auth/login", json={"username": "alice", "password": "alicepass"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == "alice"
        assert data["role"] == "admin"
        assert "expires_at" in data
        assert isinstance(data["expires_at"], float)

    def test_login_wrong_password(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        resp = client.post("/auth/login", json={"username": "alice", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, client: TestClient) -> None:
        resp = client.post("/auth/login", json={"username": "nobody", "password": "password"})
        assert resp.status_code == 401

    def test_login_returns_unique_tokens(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        t1 = client.post("/auth/login", json={"username": "alice", "password": "alicepass"}).json()["token"]
        t2 = client.post("/auth/login", json={"username": "alice", "password": "alicepass"}).json()["token"]
        assert t1 != t2


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------

class TestLogout:
    def test_logout_invalidates_token(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        token = client.post("/auth/login", json={"username": "alice", "password": "alicepass"}).json()["token"]

        # Token works before logout
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

        # Logout
        resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged_out"

        # Token no longer works after logout
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_logout_without_token_rejected(self, client: TestClient) -> None:
        resp = client.post("/auth/logout")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

class TestMe:
    def test_me_returns_user_info(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "alice", "password": "alicepass", "role": "admin"})
        token = client.post("/auth/login", json={"username": "alice", "password": "alicepass"}).json()["token"]

        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["role"] == "admin"
        assert "user_id" in data
        assert "password_hash" not in data

    def test_me_without_auth_rejected(self, client: TestClient) -> None:
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_with_invalid_token_rejected(self, client: TestClient) -> None:
        resp = client.get("/auth/me", headers={"Authorization": "Bearer notavalidtoken"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/users
# ---------------------------------------------------------------------------

class TestListUsers:
    def test_admin_can_list_users(self, admin_client) -> None:
        c, token = admin_client
        resp = c.get("/auth/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        users = resp.json()
        assert isinstance(users, list)
        assert len(users) >= 1
        assert all("password_hash" not in u for u in users)

    def test_viewer_cannot_list_users(self, client: TestClient) -> None:
        # Bootstrap admin
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        # Register viewer
        client.post(
            "/auth/register",
            json={"username": "viewer", "password": "viewerpass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        viewer_token = client.post("/auth/login", json={"username": "viewer", "password": "viewerpass"}).json()["token"]

        resp = client.get("/auth/users", headers={"Authorization": f"Bearer {viewer_token}"})
        assert resp.status_code == 403

    def test_list_users_no_auth_rejected(self, admin_client) -> None:
        c, _ = admin_client
        resp = c.get("/auth/users")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PATCH /auth/users/{user_id}
# ---------------------------------------------------------------------------

class TestPatchUser:
    def test_admin_can_change_role(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r = client.post(
            "/auth/register",
            json={"username": "bob", "password": "bobspassword", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        bob_id = r.json()["user_id"]

        resp = client.patch(
            f"/auth/users/{bob_id}",
            json={"role": "admin"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_viewer_cannot_change_role(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r = client.post(
            "/auth/register",
            json={"username": "viewer", "password": "viewerpass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        viewer_id = r.json()["user_id"]
        viewer_token = client.post("/auth/login", json={"username": "viewer", "password": "viewerpass"}).json()["token"]

        resp = client.patch(
            f"/auth/users/{viewer_id}",
            json={"role": "admin"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    def test_user_can_change_own_password(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r = client.post(
            "/auth/register",
            json={"username": "bob", "password": "boboldpass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        bob_id = r.json()["user_id"]
        bob_token = client.post("/auth/login", json={"username": "bob", "password": "boboldpass"}).json()["token"]

        resp = client.patch(
            f"/auth/users/{bob_id}",
            json={"password": "bobnewpass"},
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        assert resp.status_code == 200
        assert resp.json().get("sessions_revoked") == "true"

        # Old token invalidated; new login works
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {bob_token}"}).status_code == 401
        new_token = client.post("/auth/login", json={"username": "bob", "password": "bobnewpass"}).json()["token"]
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200

    def test_user_cannot_change_another_password(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r_alice = client.post(
            "/auth/register",
            json={"username": "alice", "password": "alicepass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        alice_id = r_alice.json()["user_id"]
        r_bob = client.post(
            "/auth/register",
            json={"username": "bob", "password": "bobspassword", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        _ = r_bob.json()["user_id"]
        bob_token = client.post("/auth/login", json={"username": "bob", "password": "bobspassword"}).json()["token"]

        resp = client.patch(
            f"/auth/users/{alice_id}",
            json={"password": "newpassword"},
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        assert resp.status_code == 403

    def test_patch_nonexistent_user_returns_404(self, admin_client) -> None:
        c, token = admin_client
        resp = c.patch(
            "/auth/users/nonexistent-id",
            json={"role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_patch_invalid_role_rejected(self, admin_client) -> None:
        c, token = admin_client
        me = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        resp = c.patch(
            f"/auth/users/{me['user_id']}",
            json={"role": "superuser"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_patch_short_password_rejected(self, admin_client) -> None:
        c, token = admin_client
        me = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        resp = c.patch(
            f"/auth/users/{me['user_id']}",
            json={"password": "short"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /auth/users/{user_id}
# ---------------------------------------------------------------------------

class TestDeleteUser:
    def test_admin_can_delete_user(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r = client.post(
            "/auth/register",
            json={"username": "bob", "password": "bobspassword", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        bob_id = r.json()["user_id"]
        bob_token = client.post("/auth/login", json={"username": "bob", "password": "bobspassword"}).json()["token"]

        resp = client.delete(
            f"/auth/users/{bob_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "true"

        # Bob's session is invalidated (cascade delete)
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {bob_token}"}).status_code == 401

        # Bob no longer appears in user list
        users = client.get("/auth/users", headers={"Authorization": f"Bearer {admin_token}"}).json()
        assert all(u["user_id"] != bob_id for u in users)

    def test_viewer_cannot_delete_user(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        r = client.post(
            "/auth/register",
            json={"username": "viewer", "password": "viewerpass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        viewer_id = r.json()["user_id"]
        viewer_token = client.post("/auth/login", json={"username": "viewer", "password": "viewerpass"}).json()["token"]

        resp = client.delete(
            f"/auth/users/{viewer_id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    def test_delete_nonexistent_user_returns_404(self, admin_client) -> None:
        c, token = admin_client
        resp = c.delete("/auth/users/nonexistent-id", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Admin endpoints gated in userdb mode
# ---------------------------------------------------------------------------

class TestAdminEndpointsUserdbMode:
    """Existing management endpoints should require admin auth in userdb mode."""

    def test_delete_fixes_requires_admin(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/fixes")
        assert resp.status_code == 401

    def test_delete_fixes_viewer_denied(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "admin", "password": "adminpass", "role": "admin"})
        admin_token = client.post("/auth/login", json={"username": "admin", "password": "adminpass"}).json()["token"]
        client.post(
            "/auth/register",
            json={"username": "viewer", "password": "viewerpass", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        viewer_token = client.post("/auth/login", json={"username": "viewer", "password": "viewerpass"}).json()["token"]

        resp = client.delete("/api/v1/fixes", headers={"Authorization": f"Bearer {viewer_token}"})
        assert resp.status_code == 403

    def test_delete_fixes_admin_succeeds(self, admin_client) -> None:
        c, token = admin_client
        resp = c.delete("/api/v1/fixes", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TOTP 2FA tests
# ---------------------------------------------------------------------------

class TestTOTP:
    """Tests for TOTP 2FA setup, login, and disable."""

    def test_2fa_setup_returns_secret_and_uri(self, admin_client) -> None:
        c, token = admin_client
        resp = c.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "secret" in data
        assert "otpauth_uri" in data
        assert "otpauth://totp/" in data["otpauth_uri"]
        assert "Beagle" in data["otpauth_uri"]

    def test_2fa_enable_with_valid_code(self, admin_client) -> None:
        import pyotp

        c, token = admin_client
        # Setup
        setup = c.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"}).json()
        secret = setup["secret"]
        # Generate a valid TOTP code
        code = pyotp.TOTP(secret).now()
        resp = c.post(
            "/auth/2fa/enable",
            json={"code": code},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "2fa_enabled"

    def test_2fa_enable_with_invalid_code(self, admin_client) -> None:
        c, token = admin_client
        c.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
        resp = c.post(
            "/auth/2fa/enable",
            json={"code": "000000"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        assert "Invalid code" in resp.json()["detail"]

    def test_login_with_2fa_returns_partial_token(self, client) -> None:
        import pyotp

        # Bootstrap admin
        client.post("/auth/register", json={
            "username": "admin2fa", "password": "adminpass", "role": "admin",
        })
        login_resp = client.post("/auth/login", json={
            "username": "admin2fa", "password": "adminpass",
        })
        token = login_resp.json()["token"]

        # Setup and enable 2FA
        setup = client.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"}).json()
        code = pyotp.TOTP(setup["secret"]).now()
        client.post(
            "/auth/2fa/enable",
            json={"code": code},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Login now returns partial token
        resp = client.post("/auth/login", json={
            "username": "admin2fa", "password": "adminpass",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["requires_2fa"] is True
        assert "partial_token" in data
        assert "token" not in data

    def test_2fa_verify_valid_code_returns_full_session(self, client) -> None:
        import pyotp

        # Bootstrap admin with 2FA
        client.post("/auth/register", json={
            "username": "admin2fa2", "password": "adminpass", "role": "admin",
        })
        token = client.post("/auth/login", json={
            "username": "admin2fa2", "password": "adminpass",
        }).json()["token"]
        setup = client.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"}).json()
        secret = setup["secret"]
        code = pyotp.TOTP(secret).now()
        client.post(
            "/auth/2fa/enable",
            json={"code": code},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Login + 2FA verify
        login_resp = client.post("/auth/login", json={
            "username": "admin2fa2", "password": "adminpass",
        })
        partial = login_resp.json()["partial_token"]
        code2 = pyotp.TOTP(secret).now()
        resp = client.post("/auth/2fa/verify", json={
            "partial_token": partial, "code": code2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == "admin2fa2"
        assert data["role"] == "admin"
        assert "expires_at" in data

        # Verify the token works
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['token']}"})
        assert me.status_code == 200

    def test_2fa_verify_invalid_code(self, client) -> None:
        import pyotp

        client.post("/auth/register", json={
            "username": "admin2fa3", "password": "adminpass", "role": "admin",
        })
        token = client.post("/auth/login", json={
            "username": "admin2fa3", "password": "adminpass",
        }).json()["token"]
        setup = client.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"}).json()
        code = pyotp.TOTP(setup["secret"]).now()
        client.post(
            "/auth/2fa/enable",
            json={"code": code},
            headers={"Authorization": f"Bearer {token}"},
        )

        login_resp = client.post("/auth/login", json={
            "username": "admin2fa3", "password": "adminpass",
        })
        partial = login_resp.json()["partial_token"]
        resp = client.post("/auth/2fa/verify", json={
            "partial_token": partial, "code": "000000",
        })
        assert resp.status_code == 401

    def test_2fa_disable_admin_recovery(self, client) -> None:
        import pyotp

        # Create admin + viewer with 2FA
        client.post("/auth/register", json={
            "username": "adminrecov", "password": "adminpass", "role": "admin",
        })
        admin_token = client.post("/auth/login", json={
            "username": "adminrecov", "password": "adminpass",
        }).json()["token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

        client.post("/auth/register", json={
            "username": "viewerlock", "password": "viewerpass", "role": "viewer",
        }, headers=headers)
        viewer_token = client.post("/auth/login", json={
            "username": "viewerlock", "password": "viewerpass",
        }).json()["token"]

        # Enable 2FA on viewer
        setup = client.post("/auth/2fa/setup", headers={"Authorization": f"Bearer {viewer_token}"}).json()
        code = pyotp.TOTP(setup["secret"]).now()
        client.post(
            "/auth/2fa/enable",
            json={"code": code},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )

        # Verify login now requires 2FA
        resp = client.post("/auth/login", json={
            "username": "viewerlock", "password": "viewerpass",
        })
        assert resp.json()["requires_2fa"] is True

        # Get viewer's user_id
        users = client.get("/auth/users", headers=headers).json()
        viewer_uid = [u for u in users if u["username"] == "viewerlock"][0]["user_id"]

        # Admin disables viewer's 2FA
        resp = client.post("/auth/2fa/disable", json={
            "user_id": viewer_uid,
        }, headers=headers)
        assert resp.status_code == 200

        # Viewer can now login without 2FA
        resp = client.post("/auth/login", json={
            "username": "viewerlock", "password": "viewerpass",
        })
        assert "token" in resp.json()
        assert "requires_2fa" not in resp.json()

    def test_normal_login_unchanged_without_2fa(self, admin_client) -> None:
        """Regression: users without 2FA still get direct tokens."""
        c, _ = admin_client
        resp = c.post("/auth/login", json={
            "username": "admin", "password": "adminpass",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "requires_2fa" not in data

    def test_users_list_shows_totp_status(self, admin_client) -> None:
        c, token = admin_client
        users = c.get("/auth/users", headers={"Authorization": f"Bearer {token}"}).json()
        # totp_enabled should be present and false by default
        assert users[0]["totp_enabled"] == 0
        # totp_secret should NOT be exposed
        assert "totp_secret" not in users[0]


# ---------------------------------------------------------------------------
# Google OAuth tests
# ---------------------------------------------------------------------------


def _oauth_config() -> "ServerFullConfig":
    """Config with Google OAuth configured."""
    return ServerFullConfig(
        server=ServerConfig(
            host="127.0.0.1",
            port=8765,
            auth_token="",
            auth_mode="userdb",
            session_lifetime_hours=24.0,
            google_client_id="test-client-id",
            google_client_secret="test-client-secret",
        ),
        database=DatabaseConfig(path=":memory:", registry_path=":memory:"),
        pairing=PairingConfig(
            correlation_window_s=5.0,
            delivery_buffer_s=0.05,
            group_expiry_s=60.0,
            freq_tolerance_hz=1000.0,
            min_corr_peak=0.05,
        ),
        solver=SolverConfig(
            search_center_lat=47.6,
            search_center_lon=-122.3,
            search_radius_km=100.0,
        ),
        map=MapConfig(output_dir="/tmp/tdoa_test_oauth_maps", max_age_s=3600.0),
    )


class TestGoogleOAuth:
    """Tests for Google OAuth endpoints."""

    def test_oauth_redirect_when_configured(self) -> None:
        app = create_app(_oauth_config())
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/auth/oauth/google", follow_redirects=False)
            assert resp.status_code == 302
            location = resp.headers["location"]
            assert "accounts.google.com" in location
            assert "test-client-id" in location

    def test_oauth_redirect_when_not_configured(self, client) -> None:
        # Default config has no google_client_id
        resp = client.get("/auth/oauth/google")
        assert resp.status_code == 404

    def test_oauth_sentinel_password_blocks_login(self) -> None:
        from beagle_server.auth import verify_password
        assert verify_password("anything", "oauth:nologin") is False

    def test_oauth_callback_invalid_state(self) -> None:
        app = create_app(_oauth_config())
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get(
                "/auth/oauth/google/callback?code=test&state=invalid",
                follow_redirects=False,
            )
            assert resp.status_code == 400

    def test_map_includes_google_oauth_flag(self) -> None:
        app = create_app(_oauth_config())
        with TestClient(app) as c:
            resp = c.get("/map")
            assert resp.status_code == 200
            assert '"googleOAuthEnabled": true' in resp.text or '"googleOAuthEnabled":true' in resp.text

    def test_map_without_oauth_has_flag_false(self, client) -> None:
        resp = client.get("/map")
        assert resp.status_code == 200
        assert '"googleOAuthEnabled": false' in resp.text or '"googleOAuthEnabled":false' in resp.text
