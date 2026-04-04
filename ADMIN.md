# Beagle Server - Administration Guide

## Contents

- [Authentication Modes](#authentication-modes)
- [Web UI Login](#web-ui-login)
- [User Registration](#user-registration)
- [User Maintenance](#user-maintenance)
- [User Management (Web UI)](#user-management-web-ui)
- [Two-Factor Authentication (TOTP)](#two-factor-authentication-totp)
- [Google OAuth](#google-oauth)
- [Session Management](#session-management)
- [Node Registration](#node-registration)
- [Database Housekeeping](#database-housekeeping)
- [TLS with a Reverse Proxy](#tls-with-a-reverse-proxy)
- [Server Configuration Reference](#server-configuration-reference)

---

## Authentication

Authentication is configured with two independent settings in `config/server.yaml`:

| Setting | Controls | Allowed values |
|---------|----------|---------------|
| `server.node_auth` | How nodes authenticate event POSTs | `none`, `token` (default), `nodedb` |
| `server.user_auth` | How humans access the map UI and admin API | `none`, `token` (default), `userdb` |

These are fully independent. A production deployment typically uses `node_auth: nodedb`
(per-node secrets for events) combined with `user_auth: userdb` (per-user logins for the UI).

**node_auth values:**
- `none` -- event POSTs are accepted without authentication
- `token` -- nodes must include the shared `auth_token` as a Bearer token
- `nodedb` -- each node authenticates with its own secret (see `scripts/manage_nodes.py`)

**user_auth values:**
- `none` -- no authentication; all UI and admin endpoints are open
- `token` -- the shared `auth_token` is required for admin endpoints
- `userdb` -- per-user accounts with roles, sessions, optional 2FA and Google OAuth

---

## Web UI Login

When `user_auth: userdb`, the map page (`GET /map`) displays a login overlay automatically.
Users enter their username and password to obtain a session token, which is stored
in `sessionStorage` (cleared when the browser tab is closed).

- **Login:** Enter credentials in the overlay. On success the overlay disappears and
  the map loads with full functionality.
- **Logout:** Click the logout button in the panel header (top-right, next to the
  username display). This invalidates the server-side session and clears the browser token.
- **Session expiry:** If a session expires mid-use, any API call triggers a 401 response
  and the login overlay reappears automatically.

When `user_auth` is `token` or `none`, no login overlay is shown.

---

## User Registration

### First-time setup (bootstrap)

When no users exist the `POST /auth/register` endpoint is open - no authentication required.
Use this to create the initial admin account:

```bash
curl -s -X POST https://tdoa.example.com/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "your-strong-password", "role": "admin"}' \
  | python3 -m json.tool
```

Response:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "admin",
  "role": "admin"
}
```

**Important:** Once the first user exists, all subsequent registrations require an admin
session token. The bootstrap window is closed automatically.

### Adding additional users (admin required)

Log in to obtain a session token:

```bash
TOKEN=$(curl -s -X POST https://tdoa.example.com/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "your-strong-password"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
```

Register a new viewer account:

```bash
curl -s -X POST https://tdoa.example.com/auth/register \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"username": "operator1", "password": "operatorpass", "role": "viewer"}' \
  | python3 -m json.tool
```

### Roles

| Role     | Permissions                                                                 |
|----------|-----------------------------------------------------------------------------|
| `admin`  | Full access: all management endpoints, user administration, delete data.    |
| `viewer` | Read-only: fixes, map, events, health. Cannot modify server state.         |

### Password requirements

- Minimum 8 characters.
- Stored as PBKDF2-HMAC-SHA256 with a random 16-byte salt, 260,000 iterations (OWASP 2023).
- Plaintext passwords are never stored or logged.

---

## User Maintenance

### List all users

```bash
curl -s https://tdoa.example.com/auth/users \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Response includes `user_id`, `username`, `role`, `created_at`, `last_login_at`.
Password hashes are never returned by the API.

### Change a user's role

```bash
curl -s -X PATCH https://tdoa.example.com/auth/users/<user_id> \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"role": "admin"}'
```

Admin-only. Valid roles: `admin`, `viewer`.

### Change a password

An admin can reset any user's password:

```bash
curl -s -X PATCH https://tdoa.example.com/auth/users/<user_id> \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"password": "new-strong-password"}'
```

A non-admin user can change their **own** password only (using their own session token).

**Note:** Changing a password immediately invalidates **all** existing sessions for that
user. They must log in again with the new password.

### Delete a user

```bash
curl -s -X DELETE https://tdoa.example.com/auth/users/<user_id> \
  -H "Authorization: Bearer $TOKEN"
```

Admin-only. All sessions for the deleted user are cascade-deleted automatically.

---

## User Management (Web UI)

In `userdb` mode, admin users see a **Users** tab in the map control panel. This
provides full user administration without CLI or API access:

- **Create user:** Fill in username, password, and role (admin/viewer). Click "Create".
- **Change role:** Use the role dropdown on any user card. Changes take effect immediately.
- **Reset password:** Click "Reset Password" on a user card, enter the new password, and save.
  All of that user's sessions are invalidated.
- **Delete user:** Click "Delete", then confirm within 3 seconds (armed confirmation pattern).
  All sessions and linked OAuth accounts are cascade-deleted.
- **Change own password:** Any logged-in user (admin or viewer) can change their own password
  using the "Change Password" section at the bottom of the Users tab. All sessions are
  invalidated and the user must log in again.

The Users tab is hidden for `viewer` users and in non-`userdb` auth modes.

---

## Two-Factor Authentication (TOTP)

Beagle supports time-based one-time passwords (TOTP) compatible with Google
Authenticator, Authy, and other TOTP apps.

### Setting up 2FA

1. Log in to the web UI as any user.
2. In the **Users** tab, click **Setup 2FA**.
3. The server generates a secret key and displays it as a base32 string. Enter this
   key into your authenticator app manually (or copy it with the copy button).
4. Enter the 6-digit code from your authenticator app to verify.
5. Once verified, 2FA is active on your account.

### Logging in with 2FA

When 2FA is enabled, the login flow has two steps:

1. Enter username and password. The server returns a short-lived partial token
   (valid for 5 minutes) instead of a full session.
2. Enter the 6-digit TOTP code. The server verifies the code and issues a full
   session token.

### API flow (programmatic access)

```bash
# Step 1: Login returns requires_2fa
curl -s -X POST https://tdoa.example.com/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "password"}'
# Response: {"requires_2fa": true, "partial_token": "xxx..."}

# Step 2: Verify TOTP code
curl -s -X POST https://tdoa.example.com/auth/2fa/verify \
  -H 'Content-Type: application/json' \
  -d '{"partial_token": "xxx...", "code": "123456"}'
# Response: {"token": "yyy...", "user_id": "...", "username": "admin", ...}
```

### Disabling 2FA

- **Self-disable:** A user can disable their own 2FA from the Users tab. This
  requires entering their password and a valid TOTP code.
- **Admin recovery:** An admin can disable 2FA for any user from the Users tab
  (useful when a user loses their authenticator device). No TOTP code is required
  for admin recovery.

### API endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/2fa/setup` | POST | Any user | Generate TOTP secret and provisioning URI |
| `/auth/2fa/enable` | POST | Any user | Verify code and activate 2FA |
| `/auth/2fa/verify` | POST | None (partial token in body) | Exchange partial token + TOTP code for session |
| `/auth/2fa/disable` | POST | Admin or self | Disable 2FA (admin: `{user_id}`; self: `{password, code}`) |

---

## Google OAuth

Beagle supports "Sign in with Google" as an alternative to username/password login.

### Prerequisites

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Navigate to **APIs & Services -> Credentials -> Create Credentials -> OAuth client ID**.
3. Set the application type to **Web application**.
4. Add your server's callback URL as an authorized redirect URI:
   ```
   https://tdoa.example.com/auth/oauth/google/callback
   ```
5. Note the **Client ID** and **Client Secret**.

### Configuration

Add the credentials to `config/server.yaml`:

```yaml
server:
  user_auth: "userdb"
  google_client_id: "123456789-abcdefg.apps.googleusercontent.com"
  google_client_secret: "GOCSPX-xxxxxxxxxx"
```

Or set them as environment variables (takes precedence when the config field is empty):

```bash
export TDOA_GOOGLE_CLIENT_ID="123456789-abcdefg.apps.googleusercontent.com"
export TDOA_GOOGLE_CLIENT_SECRET="GOCSPX-xxxxxxxxxx"
```

When both `google_client_id` and `google_client_secret` are set, a **"Sign in with
Google"** button appears on the login overlay.

### How it works

1. User clicks "Sign in with Google" -> redirected to Google's consent screen.
2. After consent, Google redirects back to `/auth/oauth/google/callback` with an
   authorization code.
3. The server exchanges the code for user info (email, name).
4. Account matching:
   - If a linked OAuth account exists -> log in as that user.
   - If the Google email matches an existing user's username -> auto-link and log in.
   - If no match -> create a new `viewer` account (or `admin` if no users exist yet).
5. If the matched user has 2FA enabled -> redirect with a partial token for TOTP verification.
6. Otherwise -> redirect to `/map` with the session token.

### OAuth-only accounts

Users created via Google OAuth have no usable password (sentinel hash). They cannot
log in via the username/password form. An admin can set a password for them via
the Users tab if password-based login is needed.

### Linking and unlinking

Authenticated users can view their linked OAuth accounts via `GET /auth/oauth/accounts`
and unlink Google via `DELETE /auth/oauth/link/google`.

### API endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/oauth/google` | GET | None | Redirect to Google consent screen |
| `/auth/oauth/google/callback` | GET | None | Handle Google OAuth callback |
| `/auth/oauth/accounts` | GET | Any user | List linked OAuth providers |
| `/auth/oauth/link/google` | DELETE | Any user | Unlink Google account |

---

## Session Management

### Login

```bash
curl -s -X POST https://tdoa.example.com/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "your-strong-password"}' \
  | python3 -m json.tool
```

Response (without 2FA):
```json
{
  "token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "user_id": "550e8400-...",
  "username": "admin",
  "role": "admin",
  "expires_at": 1741999999.0
}
```

Response (with 2FA enabled):
```json
{
  "requires_2fa": true,
  "partial_token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

If `requires_2fa` is returned, complete the login with `POST /auth/2fa/verify`
(see [Two-Factor Authentication](#two-factor-authentication-totp)).

Store the `token` and pass it in the `Authorization: Bearer <token>` header on all
subsequent requests.

### Session lifetime

Default: 24 hours. Configurable via `server.session_lifetime_hours` in `server.yaml`:

```yaml
server:
  session_lifetime_hours: 48.0   # sessions last 2 days
```

### Logout

```bash
curl -s -X POST https://tdoa.example.com/auth/logout \
  -H "Authorization: Bearer $TOKEN"
```

Immediately invalidates the token. Subsequent requests with the same token return 401.

### Who am I?

```bash
curl -s https://tdoa.example.com/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### Expired session cleanup

Sessions are checked for expiry on every request (`fetch_session` rejects expired tokens).
To proactively purge stale session rows from the database, the `purge_expired_sessions()`
function is available in `db.py`. This can be called from a maintenance script or scheduled
task (see [Database Maintenance](TODO.md#database-maintenance-prevent-unbounded-growth)).

---

## Node Registration

Nodes can be managed from the **web UI** (Nodes tab in the map control panel) or
via the `scripts/manage_nodes.py` CLI tool.

The web UI supports registering nodes, regenerating secrets, editing labels,
editing config JSON, enabling/disabling, and deleting nodes.  It requires the
server to be running and an admin login.

The CLI tool operates directly on the registry SQLite database and does not
require the server to be running - useful for scripted provisioning or initial
setup.

You must specify the database location with one of two mutually exclusive flags:

```bash
# Point directly at the registry database:
python scripts/manage_nodes.py --db data/tdoa_registry.db <command> ...

# Or read the path from the server config (uses database.registry_path):
python scripts/manage_nodes.py --server-config config/server.yaml <command> ...
```

The examples below use `--db` for brevity.

### Add a node

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db add seattle-north-01 \
    --label "Seattle North Roof"
```

The optional `--label` flag sets a human-readable display name for the node.

This generates a random 256-bit node secret and prints it once. Copy the secret
into the node's bootstrap configuration (`/etc/beagle/bootstrap.yaml` or equivalent).

### Config cache directory

Bootstrap-mode nodes cache the fetched config to `/var/cache/beagle/node_config.json`
by default so they can start without network access.  This directory does not exist
by default and non-root users cannot create it.  Set it up once per node host:

```bash
sudo mkdir -p /var/cache/beagle
sudo chown tdoa:tdoa /var/cache/beagle   # or whichever user runs the node
```

If you run the node as a regular user without creating this directory, the node
will log a warning (`Permission denied`) but continue to operate normally - it
just won't have an offline config cache.

To use a different path, set `config_cache_path` in `bootstrap.yaml`:

```yaml
config_cache_path: "/home/tdoa/.cache/tdoa/node_config.json"
```

### Node position

The `nodes` table does not store position. Nodes supply their own location via
the `location` block in their config, and include their position in every event
they post. The server and solver obtain node positions from event data.

For **locally configured** nodes (standalone YAML), set `location` in the YAML file.

For **remotely managed** nodes (bootstrap mode), the config assigned via `set-config`
must include a valid `location` block. A typical workflow:

1. `add` the node
2. Create a config YAML that includes `location:` with the correct coordinates
3. `set-config` to assign it to the node

### List nodes

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db list
```

### Show node details

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db show seattle-north-01
```

### Enable / disable a node

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db disable seattle-north-01
python scripts/manage_nodes.py --db data/tdoa_registry.db enable seattle-north-01
```

Disabled nodes have their events rejected at ingest with HTTP 403.

### Update node config

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \
    --config-file configs/seattle-north-01.yaml
```

You can also pass inline JSON instead of a file:

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \
    --config-json '{"location": {"latitude_deg": 47.71, "longitude_deg": -122.34}, ...}'
```

Omitting both `--config-file` and `--config-json` clears the node's config.

Config changes are recorded in `node_config_history` for audit and rollback.

### Remove a node

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db remove seattle-north-01
```

Use `-y` / `--yes` to skip the confirmation prompt.

### Regenerate a node secret

If a node secret is compromised:

```bash
python scripts/manage_nodes.py --db data/tdoa_registry.db regen-secret seattle-north-01
```

Update the node's bootstrap file with the new secret immediately.

---

## Database Housekeeping

The server uses two SQLite databases:

| File | Contents | Growth rate | Maintenance |
|------|----------|-------------|-------------|
| `data/tdoa_data.db` | events, fixes, heatmap_cells | High - thousands of rows/hr | Regular pruning required |
| `data/tdoa_registry.db` | nodes, node_config_history, users, user_sessions, partial_sessions, oauth_accounts | Low - grows only on config/user changes | Periodic cleanup of expired sessions and old config history |

### Fresh start (wipe operational data)

To clear all accumulated detections and fixes while keeping node registrations and user accounts:

```bash
systemctl stop beagle-server   # or however you manage the process
rm data/tdoa_data.db
systemctl start beagle-server  # database is recreated automatically
```

The registry (`tdoa_registry.db`) is untouched - nodes remain registered, users remain active.

### Automated maintenance (recommended)

The `scripts/db_maintenance.py` script handles all routine pruning.  Run it
daily from cron:

```bash
# crontab -e
0 3 * * * cd /opt/beagle && /opt/beagle/env/bin/python scripts/db_maintenance.py \
    --server-config config/server.yaml >> /var/log/tdoa/maintenance.log 2>&1
```

What it does each run:

| Database | Action | Default retention |
|----------|--------|-------------------|
| Operational | Delete old events | 14 days |
| Operational | Delete old fixes | 14 days |
| Operational | WAL checkpoint | - |
| Registry | Purge expired user sessions | expired |
| Registry | Prune config history per node | keep last 50 |
| Registry | WAL checkpoint | - |

Override defaults with flags:

```bash
# Keep 7 days of events, 30 days of fixes, 20 config versions per node
python scripts/db_maintenance.py --server-config config/server.yaml \
    --events-days 7 --fixes-days 30 --config-history-keep 20

# Preview what would be deleted without changing anything
python scripts/db_maintenance.py --server-config config/server.yaml --dry-run

# Specify DB paths directly (no server config needed)
python scripts/db_maintenance.py --data-db data/tdoa_data.db \
    --registry-db data/tdoa_registry.db
```

The script is safe to run while the server is running.  It uses normal
`DELETE` + `PRAGMA wal_checkpoint(TRUNCATE)`, not `VACUUM`.

### Operational database growth (events and fixes)

The `events` and `fixes` tables accumulate continuously.  At 3 nodes x 30 events/hr:

- `events`: ~90 rows/hr -> ~200 MB after 6 months
- `fixes`: ~30 rows/hr -> manageable, but grows without bound

With `db_maintenance.py` running daily at the default 14-day retention, the
operational DB stays bounded at roughly 2 weeks of data.

### Manual housekeeping

For one-off operations or if you prefer not to use the maintenance script:

**Expired sessions:**

```bash
sqlite3 data/tdoa_registry.db \
  "DELETE FROM user_sessions WHERE expires_at < strftime('%s','now');"
```

**Config history audit** (review before pruning):

```bash
sqlite3 data/tdoa_registry.db \
  "SELECT node_id, COUNT(*), MIN(changed_at), MAX(changed_at) FROM node_config_history GROUP BY node_id;"
```

### VACUUM (reclaim disk space)

`DELETE` frees SQLite pages internally but does not shrink the file on disk.
After a large prune, run `VACUUM` to reclaim space.  This requires an exclusive
lock, so stop the server first:

```bash
systemctl stop beagle-server
sqlite3 data/tdoa_data.db "VACUUM;"
sqlite3 data/tdoa_registry.db "VACUUM;"
systemctl start beagle-server
```

WAL checkpoint (safe while server is running - just truncates the WAL file):

```bash
sqlite3 data/tdoa_data.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

---

## Remote Logging

Both the server and node write structured logs to stderr, which systemd's journal
captures automatically.  For centralized logging you can forward journal entries
to a remote syslog or log collector - no code changes required.

### Option 1: systemd journal remote forwarding

Forward the journal from each host to a central log server using
`systemd-journal-upload` (sender) and `systemd-journal-remote` (receiver).

**On the log server** (receiver):

```bash
apt install systemd-journal-remote
systemctl enable --now systemd-journal-remote.socket
```

This listens on port 19532 (HTTPS) by default.

**On each node/server host** (sender):

```bash
apt install systemd-journal-upload
```

Edit `/etc/systemd/journal-upload.conf`:

```ini
[Upload]
URL=https://logserver.example.com:19532
```

```bash
systemctl enable --now systemd-journal-upload
```

Logs from all hosts will appear on the log server under
`/var/log/journal/remote/`.  Query them with:

```bash
# All Beagle logs across all hosts
journalctl --directory=/var/log/journal/remote/ -u 'beagle-*'

# Specific node
journalctl --directory=/var/log/journal/remote/ -u beagle-node
```

See: https://www.freedesktop.org/software/systemd/man/latest/systemd-journal-remote.service.html

### Option 2: Forward to a syslog server

If you already run a syslog server (rsyslog, syslog-ng), configure journald to
forward to it.

Edit `/etc/systemd/journald.conf`:

```ini
[Journal]
ForwardToSyslog=yes
```

Then configure your syslog daemon to send to the remote server.  For rsyslog,
add to `/etc/rsyslog.d/50-tdoa-remote.conf`:

```
:programname, startswith, "beagle" @@logserver.example.com:514
```

(`@@` = TCP, `@` = UDP)

See: https://www.freedesktop.org/software/systemd/man/latest/journald.conf.html

### Option 3: Log shipper (Promtail, Vector, Filebeat)

For log aggregation platforms (Grafana Loki, Elasticsearch, Datadog), run a
shipper that reads from the journal:

```yaml
# Example: Promtail scraping the systemd journal
scrape_configs:
  - job_name: tdoa
    journal:
      labels:
        job: tdoa
      matches:
        - _SYSTEMD_UNIT=beagle-server.service
        - _SYSTEMD_UNIT=beagle-node.service
    relabel_configs:
      - source_labels: ['__journal__hostname']
        target_label: host
```

The node's structlog output is key=value formatted, which most aggregators parse
natively without custom extractors.

---

## TLS with a Reverse Proxy

The Beagle server (uvicorn) listens on plain HTTP.  For production deployments,
terminate TLS at a reverse proxy in front of the server.  The application already
respects the standard `X-Forwarded-Proto` and `X-Forwarded-Host` headers, so
OAuth redirect URIs, SSE streams, and all other functionality work correctly
behind a proxy with no code changes.

### Caddy (automatic Let's Encrypt)

```
tdoa.example.com {
    reverse_proxy localhost:8765
}
```

Caddy auto-provisions and renews TLS certificates from Let's Encrypt.

### nginx

```nginx
server {
    listen 443 ssl;
    server_name tdoa.example.com;
    ssl_certificate     /etc/letsencrypt/live/tdoa.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tdoa.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host  $host;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;

        # SSE (Server-Sent Events) support for /map/stream
        proxy_buffering off;
        proxy_cache     off;
        proxy_read_timeout 86400s;
    }
}
```

Use `certbot` to obtain Let's Encrypt certificates, or provide your own.

### Important notes

- **SSE streams** (`GET /map/stream`): Disable proxy buffering for this path
  (shown above for nginx; Caddy handles this automatically).  Without it, fix
  updates will be delayed or batched instead of streaming in real time.
- **Google OAuth callback**: The redirect URI is built dynamically from the
  request's `X-Forwarded-Proto` and `X-Forwarded-Host` headers.  Ensure your
  proxy sets these headers (shown in both examples above), and that the
  callback URL registered in Google Cloud Console matches the public hostname
  (`https://tdoa.example.com/auth/oauth/google/callback`).
- **WebSocket**: Not used - the map uses SSE, so no WebSocket upgrade
  configuration is needed.

---

## Server Configuration Reference

Relevant fields in `config/server.yaml` for authentication:

```yaml
server:
  host: "0.0.0.0"
  port: 8765

  # Node authentication: none | token | nodedb
  node_auth: "nodedb"

  # User authentication: none | token | userdb
  user_auth: "userdb"

  # Shared Bearer token (used when node_auth or user_auth is "token")
  auth_token: ""

  # Session lifetime for userdb mode (hours)
  session_lifetime_hours: 24.0

  # Google OAuth (optional; empty strings disable Google sign-in)
  google_client_id: ""       # or env TDOA_GOOGLE_CLIENT_ID
  google_client_secret: ""   # or env TDOA_GOOGLE_CLIENT_SECRET

database:
  path: "data/tdoa_data.db"              # Operational: events, fixes, heatmap
  registry_path: "data/tdoa_registry.db" # Permanent: nodes, users, sessions, oauth_accounts
```

### API endpoint auth summary (userdb mode)

| Endpoint                          | Auth required  | Notes                            |
|-----------------------------------|----------------|----------------------------------|
| `GET /health`                     | None           | Public                           |
| `GET /api/v1/events`              | None           | Public                           |
| `GET /api/v1/fixes`               | None           | Public                           |
| `GET /map`                        | None           | Public                           |
| `POST /api/v1/events`             | None*          | *nodedb mode requires node auth  |
| `POST /api/v1/heartbeat`          | None           | Node health/position announcement |
| `DELETE /api/v1/fixes`            | Admin          |                                  |
| `DELETE /api/v1/heatmap`          | Admin          |                                  |
| `GET /api/v1/nodes`               | Admin          |                                  |
| `GET /api/v1/nodes/{id}`          | Admin          |                                  |
| `PATCH /api/v1/nodes/{id}`        | Admin          |                                  |
| `DELETE /api/v1/nodes/{id}`       | Admin          |                                  |
| `PATCH /api/v1/settings`          | Admin          |                                  |
| `POST /auth/register`             | Open if no users; Admin otherwise |               |
| `POST /auth/login`                | None           | Returns session token (or partial token if 2FA) |
| `POST /auth/logout`               | Any user       |                                  |
| `GET /auth/me`                    | Any user       |                                  |
| `GET /auth/users`                 | Admin          | Includes `totp_status` per user  |
| `PATCH /auth/users/{id}`          | Admin or own   | Viewers may change own password  |
| `DELETE /auth/users/{id}`         | Admin          |                                  |
| `POST /auth/2fa/setup`            | Any user       | Generate TOTP secret             |
| `POST /auth/2fa/enable`           | Any user       | Activate 2FA with verification code |
| `POST /auth/2fa/verify`           | None           | Exchange partial token + TOTP code for session |
| `POST /auth/2fa/disable`          | Admin or self  | Disable 2FA (admin recovery or self) |
| `GET /auth/oauth/google`          | None           | Redirect to Google OAuth         |
| `GET /auth/oauth/google/callback` | None           | Google OAuth callback            |
| `GET /auth/oauth/accounts`        | Any user       | List linked OAuth providers      |
| `DELETE /auth/oauth/link/google`  | Any user       | Unlink Google account            |

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](LICENSE).
