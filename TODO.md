# Beagle - Feature TODO List

## Contents

**Outstanding**
- [Remote node restart trigger](#remote-node-restart-trigger)
- [Report node code version (git sha) in heartbeat / health](#report-node-code-version-git-sha-in-heartbeat--health)
- [Onset xcorr: investigate alternative detection methods](#onset-xcorr-investigate-alternative-detection-methods)
- [Refresh real-data test fixtures](#refresh-real-data-test-fixtures)
- [SoapySDR: long-term migration to direct SDRplay API](#soapysdr-long-term-migration-to-direct-sdrplay-api)
- [manage_nodes: add group-update command](#manage_nodes-add-group-update-command)
- [Review functioning of frequency groups](#review-functioning-of-frequency-groups)

**Completed**
- [✓ Web UI: User Management, Login, 2FA, Google OAuth](#web-ui-user-management-login-2fa-google-oauth)
- [✓ Code Cleanup: items from code review](#code-cleanup-items-from-code-review)
- [✓ Database Maintenance: automated pruning script](#database-maintenance-automated-pruning-script)
- [✓ Node CPU usage: decimator + sync BPF optimisation](#node-cpu-usage-decimator-sync-bpf-optimisation)
- [✓ Web UI: 2-Node Line-of-Position Display](#web-ui-2-node-line-of-position-display)
- [✓ Web UI: Per-Node Enable/Disable Control](#web-ui-per-node-enabledisable-control)
- [✓ Web UI: Group Management from GUI](#web-ui-group-management-from-gui)
- [✓ Web UI: Node Management from GUI](#web-ui-node-management-from-gui)
- [✓ Web UI: Node Frequency Groups](#web-ui-node-frequency-groups)
- [✓ Project Name Change](#project-name-change)
- [✓ Node: Python GC pauses causing RSPduo FIFO overflows](#node-python-gc-pauses-causing-rspduo-fifo-overflows)
- [✓ Server: map rendering efficiency (O(n^2) node pairs per poll)](#server-map-rendering-efficiency-on2-node-pairs-per-poll)
- [✓ Server: suppress HeatMap `max_val` UserWarning](#server-suppress-heatmap-max_val-userwarning)
- [✓ Node: onset_time_ns unit mismatch fix (3x drift)](#node-onset_time_ns-unit-mismatch-fix-3x-drift)
- [✓ Node: RSPduo TCXO hardware timestamps (Part A)](#node-rspduo-tcxo-hardware-timestamps-part-a)
- [✓ Node: pipeline_offset_ns calibration (node-mapleleaf / node-greenlake)](#node-pipeline_offset_ns-calibration-node-mapleleaf-node-greenlake)
- [✓ Node: GC pause monitoring (gc.callbacks)](#node-gc-pause-monitoring-gccallbacks)
- [✓ Server: --log-level CLI flag](#server-log-level-cli-flag)
- [✓ Server: heartbeat access log lines demoted to DEBUG](#server-heartbeat-access-log-lines-demoted-to-debug)
- [✓ Server: min_nodes=2 to enable 2-node LOP fixes](#server-min_nodes2-to-enable-2-node-lop-fixes)
- [✓ Server: LOP baseline-too-short log demoted to DEBUG](#server-lop-baseline-too-short-log-demoted-to-debug)
- [✓ Freq-hop: mid-transmission arrival suppression](#freq-hop-mid-transmission-arrival-suppression)
- [✓ Node SNR reporting: noise floor tracking + GET /api/v1/nodes/snr](#node-snr-reporting-noise-floor-tracking-get-apiv1nodessnr)
- [✓ SoapySDRPlay3: RSPduo DT-IR support via pothosware fork](#soapysdrplay3-rspduo-dt-ir-support-via-pothosware-fork)
- [✓ User Registration and Authentication](#user-registration-and-authentication)
- [✓ Hide Fixes (non-destructive reset)](#hide-fixes-non-destructive-reset)
- [✓ Fixed Time Window Display (start-time to end-time)](#fixed-time-window-display-start-time-to-end-time)
- [✓ Fix Popup: Human-Readable Timestamp on Hover](#fix-popup-human-readable-timestamp-on-hover)
- [✓ Remote Node Registration and Config Management](#remote-node-registration-and-config-management)
- [✓ Mock Generator - PTT Onset/Offset Pattern](#mock-generator-ptt-onsetoffset-pattern)
- [✓ Heat Map Mode](#heat-map-mode)
- [✓ Web Page Control - Reset Fix History](#web-page-control-reset-fix-history)
- [✓ Live Map Updates via SSE](#live-map-updates-via-sse)
- [✓ Web Page Control - Dynamic Aging Window](#web-page-control-dynamic-aging-window)
- [✓ Map Control Panel](#map-control-panel)
- [✓ Fix Layer Ordering + Hyperbola Generator](#fix-layer-ordering-hyperbola-generator)

---

### Remote node restart trigger

Add a mechanism to remotely trigger a node restart from the server UI or API.
When a node is running under systemd with `Restart=on-failure`, the server
can signal it to exit with code 75 (EX_TEMPFAIL), causing systemd to restart
it with the updated config.

**Motivation:** Config changes that require a full restart (sync station
coordinates, SDR hardware params, sample rate) cannot be hot-reloaded.
Currently the only way to restart a remote node is SSH access or asking
the operator to do it manually. A server-side "restart node" button would
allow the admin to trigger a clean restart remotely.

**Approach:** Add a `restart_requested` flag to the node's config response.
The node's config poll thread checks this flag; when set, it logs a message
and calls `sys.exit(75)`. The server UI gets a "Restart" button per node
(with armed confirmation). The flag is cleared after one delivery so the
node doesn't restart in a loop.

---

### Report node code version (git sha) in heartbeat / health

There is currently no way to tell which code revision a remote node is
running.  When debugging cross-node behavior we end up suspecting "maybe
node X just hasn't been restarted on the latest code" with no way to
confirm or refute it from the server.

**Approach:** at node startup, capture the git short sha (or tag, or
build version) of the running source tree and include it in:
- the `/health` JSON snapshot (`node_software_version` already exists
  but is hardcoded to `"0.1.0"` in main.py)
- the long-poll heartbeat body (so the server records it per-node)
- the GUI node details dropdown so the operator can see it at a glance

The git sha should be captured in a way that survives `pip install`
into a venv (e.g. via `git rev-parse --short HEAD` baked into the
package at install time, or via `importlib.metadata`).  Fall back to
`"unknown"` if the node was installed without git history.

**Motivation:** observed during the Magnolia repeater sync_delta
investigation 2026-04-08: kb7ryy was contributing significantly more
noise than the other two RSPduo nodes, and one plausible explanation
was that kb7ryy hadn't been restarted on the latest code.  We had no
way to confirm or refute this from the server.

---

### Onset xcorr: investigate alternative detection methods

Onset xcorr between nodes succeeds only ~20-30% of the time, while offset
xcorr succeeds ~85%. The root cause is that PA turn-on transitions are
broader and noisier than the sharp PA shutoff, producing wide/ambiguous
xcorr peaks that fail the lag or SNR gate.

Offset xcorr provides +/-7 usec TDOA accuracy when it works, so offset-only
fixes are viable. But onset detection would double the fix rate.

**Ideas to explore (post-deployment):**
- Alternative onset snippet anchoring (e.g. matched filter for known PA
  ramp profiles, or energy-onset detection instead of derivative-peak)
- Frequency-domain xcorr with phase weighting (GCC-PHAT) which may be
  more robust to amplitude ramp differences between nodes
- Onset-from-offset: use the offset TDOA to predict what the onset TDOA
  should be for the same transmission (same transmitter position)
- Longer onset snippets or adaptive snippet length based on transition
  sharpness
- Hardware-specific snippet processing: the RTL-SDR raw IQ shows a 5-10 dB
  power ramp at the start of each target block (R820T gain settling beyond
  the PLL settling period); this distorts onset snippets for weaker signals

**Current state (2026-03-26):** Offset xcorr is the reliable path.
node-discovery (RTL-SDR freq_hop) participates in ~50% of groups (limited by
70% target duty cycle). When offset xcorr succeeds for all 3 nodes, the
3-node fix residual is 5-8 usec. The server falls back to sync_delta
(+/-3.5 ms) when xcorr fails, which makes that node an outlier.

---

### Refresh real-data test fixtures

`tests/fixtures/real_event_pairs.json` contains 92 event pairs captured on
2026-03-23 from node-mapleleaf/node-greenlake. The snippet format is stale: onset
snippets were captured before the derivative-peak anchoring change
(2026-03-25, commit `a69a2f7`) which places the steepest rise at 25% of
the snippet. The old data has transitions at arbitrary positions.

**Action:** Collect a fresh set of event pairs from a live run with the
current codebase, including pairs from all three node types (node-mapleleaf,
node-greenlake, node-discovery) to cover both RSPduo and RTL-SDR snippet formats.
Delete the stale `real_event_pairs_20260323.json` (identical copy of the
tracked file).

Also consider updating `scripts/test_derivative_xcorr.py` to use the
refreshed fixture data.

---

### SoapySDR: long-term migration to direct SDRplay API

The `tdoa-hw-timestamps` branch in `dpkingston/SoapySDRPlay3` (permanent fork
of `pothosware/SoapySDRPlay3`) provides the TCXO hardware timestamps we need.
The pothosware maintainer has indicated they are unlikely to accept the
timestamp changes upstream.

**Short term:** Continue using the fork. Rebase `tdoa-hw-timestamps` onto
upstream `rspduo-dual-independent-tuners` (or main, once it merges) as needed.

**Long term:** Replace `rspduo.py`'s SoapySDR calls with direct SDRplay API
calls via `ctypes` or a small C extension. The SoapySDR abstraction layer
provides no benefit for the RSPduo dual-tuner path - `rspduo.py` is already
device-specific. Direct API access gives full control over `firstSampleNum`
timestamps without maintaining a C++ fork.

---

### manage_nodes: add group-update command

`scripts/manage_nodes.py` has `group-add`, `group-list`, `group-show`,
`group-remove`, and `group-set-node` commands but no `group-update`.  To
fix coordinates or other fields on an existing freq group, the only
options are direct SQL UPDATE on `tdoa_registry.db` or HTTP PATCH via
the web UI.

**Motivation:** Discovered when fixing the stale Magnolia KUOW coordinates
on 2026-04-06.  The freq-group sync_station_lat/lon overrides the
per-node `primary_station` in the API response (api.py:1407-1424), so
correcting the per-node config file alone has no effect for nodes
assigned to a freq group.  Hand-rolling the SQL UPDATE was awkward and
required also bumping `config_version` and writing to
`node_config_history` to trigger long-poll updates.

**Approach:** Add `cmd_group_update` mirroring the HTTP PATCH endpoint
(`api.py:patch_group`).  Should:
1. Accept partial updates (label, sync_freq_hz, sync_station_id,
   sync_station_lat, sync_station_lon, target_channels).
2. Bump `config_version` on all member nodes when any frequency-plan
   field changes (mirror `bump_group_members_version` from db.py).
3. Write a row to `node_config_history` for each bumped node.
4. Validate that the group exists; print a friendly error otherwise.

Usage example:
```
scripts/manage_nodes.py group-update Magnolia \
    --sync-station-lat 47.61576 --sync-station-lon -122.30919
```

---

### Review functioning of frequency groups

The freq-group overlay in `api.py:get_node_config` silently replaces a
node's per-node `sync_signal.primary_station` with values from the
`node_freq_groups` table whenever the node is assigned to a group.  Two
incidents to date have been confused by this:

1. **Stale Magnolia coords (2026-04-06)**.  Operator updated the per-node
   YAML in `remote_configs/<node>.yaml`; the node was still assigned to a
   freq group whose row carried older KUOW coordinates; the freq-group
   row silently overrode the per-node config in the API response.
   Effect: nodes kept reporting the stale coordinates and only the
   colocated_pair_test scatter alerted us.

2. **No `manage_nodes group-update` command**: see the entry above.  The
   only ways to fix freq-group coordinates today are direct SQL UPDATE
   on `tdoa_registry.db` or HTTP PATCH via the web UI; both require
   manually bumping `config_version` and writing a `node_config_history`
   row to make the long-poll wake up.

3. **No reload-from-file path**: per-node configs are now auto-reloaded
   from `config_file_path` on every poll (see commit history for the
   "config auto-reload" change).  Freq groups have no such mechanism --
   they only exist in the registry DB.

**Goals of the review:**

- Decide whether the freq-group concept is still pulling its weight now
  that per-node configs are file-driven and auto-reloaded.  If the same
  outcome can be achieved by editing N per-node files (and the
  auto-reload picks them up), the freq-group table may be redundant for
  the deployment scale we have.
- If we keep freq groups, decide whether they should also be
  file-backed (so the operator's source of truth is files everywhere
  instead of "files for nodes, DB for groups").
- If we keep them DB-only, build the `manage_nodes group-update`
  command and document the override semantics clearly so the next
  incident is less surprising.

This is a design review, not a single change.  Pick it up after the
auto-reload work has been in production for a few days and we can see
how the operational story actually feels.

---

## Completed

### ✓ Web UI: User Management, Login, 2FA, Google OAuth

Full browser-based authentication and user management for the web UI.

**Phase 1 - Browser Login UI**
- Login overlay in `userdb` mode with username/password form
- Session tokens stored in `sessionStorage` (cleared on tab close)
- `_fetch()` wrapper handles 401 -> auto-shows login overlay
- User info (username + role) and Logout button in panel header
- `authMode` injected into TDOA JS data object

**Phase 2 - Users Tab (admin-only)**
- 4th tab visible only to admins in `userdb` mode
- Full user CRUD: create, list, change role, reset password, delete
- Armed confirmation for destructive actions (delete)
- "Change own password" section for all authenticated users

**Phase 3 - TOTP 2FA**
- Google Authenticator-compatible TOTP via `pyotp`
- `POST /auth/2fa/setup` -> returns base32 secret + otpauth URI
- `POST /auth/2fa/enable` -> verify code and activate
- Modified login: returns `{requires_2fa, partial_token}` when 2FA active
- `POST /auth/2fa/verify` -> exchange partial token + code for full session
- `POST /auth/2fa/disable` -> admin recovery or self-disable with code
- DB: `totp_secret`/`totp_enabled` columns on users, `partial_sessions` table

**Phase 4 - Google OAuth**
- `google_client_id` / `google_client_secret` config fields (env var fallback)
- `GET /auth/oauth/google` -> redirect to Google consent screen
- `GET /auth/oauth/google/callback` -> exchange code, find/create user, redirect
- Auto-creates viewer account on first Google login (admin if no users exist)
- OAuth-only users get `oauth:nologin` sentinel password hash
- If user has 2FA, OAuth callback redirects with `?pending_2fa=` partial token
- "Sign in with Google" button on login overlay
- `oauth_accounts` table links Google IDs to local users

Files: `map_output.py`, `api.py`, `db.py`, `auth.py`, `config.py`, `pyproject.toml`
Tests: 15 new tests in `test_auth.py` (543 total, all passing)

---

### ✓ Code Cleanup: items from code review

All five items from the 2026-03-12 code review have been addressed:

1. **`map_output.py` - removed unused `hyperbola_points` parameter** from `build_map()`
   and its sole caller in `api.py`.  Hyperbola rendering is now client-side via
   `/map/data`; the parameter was dead code.

2. **`reporter.py` - circuit-breaker pattern implemented.**  Replaces the 3-retry
   WARNING spam with: one ERROR on disconnect, silent single-attempt retries while
   disconnected, ERROR on reconnect, hourly reminder if still disconnected.

3. **`api.py` - `GET /auth/users`** now uses an explicit comprehension to strip
   `password_hash` instead of mutable `dict.pop()`.

4. **`health.py` - `_make_handler()`** return type annotation added.

5. **`carrier_detect.py` - diagnostic log** moved from INFO to DEBUG.

6. **`logging.py` - httpx/httpcore silenced** at WARNING level (was TODO item 1
   in the reporter.py docstring).

---

### ✓ Database Maintenance: automated pruning script

Implemented as `scripts/db_maintenance.py` - a cron-friendly script that:
- Prunes events and fixes older than a configurable retention window (default 14 days)
- Purges expired user sessions from the registry
- Prunes node_config_history keeping the last N versions per node (default 50)
- Runs WAL checkpoint on both databases
- Supports `--dry-run` for safe preview
- Reads DB paths from `--server-config` or direct `--data-db`/`--registry-db` flags

See ADMIN.md "Automated maintenance" section for cron setup.

**Future enhancements (not yet implemented):**
- Strip `iq_snippet_b64` from `raw_json` for events older than the SNR window
  (saves ~85% of event storage without losing structured metadata)
- `POST /api/v1/maintenance/prune` admin endpoint for on-demand pruning from the UI
- `GET /api/v1/health` extended to report `db_size_bytes`

---

### ✓ Node CPU usage: decimator + sync BPF optimisation

**Observed (RSPduo hardware):** `python` consuming ~2/3 of one CPU core,
`sdrplay_api` consuming ~1/3.  Total: ~1 full core just to run the node pipeline.

**Root cause:** decimator FIR filter consumed 84% of Python CPU.
`scipy.signal.lfilter` computes all N input samples then strides by D,
wasting (D-1)/D of compute.

**Optimisations applied:**

1. **Decimator:** replaced `lfilter+stride` with `scipy.signal.upfirdn`
   (natively decimating) + optional `vDSP_desamp` on macOS.  Cross-buffer
   state via history prefix padded to a multiple of the decimation factor.

2. **Sync BPF:** reduced pilot bandpass filter from 255 to 127 taps.  The
   cross-correlation with the 19 kHz complex exponential template provides
   sufficient frequency selectivity; 127 taps still gives adequate rejection.

3. **Carrier detector idle fast-path:** investigated but rejected - numpy
   dispatch overhead on 32-element arrays exceeds the Python loop savings.
   The deque.append ring buffer is the unavoidable cost.

| Platform | Before | After | Improvement |
|----------|--------|-------|-------------|
| Linux i5-4570T | 1.9x realtime, 53% CPU | 6.8x realtime, 15% CPU | **3.6x faster** |
| Linux i7-4765T | 2.0x realtime, 50% CPU | 5.6x realtime, 18% CPU | **2.8x faster** |
| macOS (dev, vDSP) | 3.4x realtime, 29% CPU | 16.7x realtime, 6% CPU | **4.9x faster** |

**Remaining profile (post-optimisation):**
- Sync decimator: ~46% of pipeline CPU
- Sync detector (BPF + xcorr): ~27%
- Target decimator: ~14%
- Carrier detector: ~6%
- FM demod + DC removal: ~7%

**sdrplay_api CPU (~1/3 of one core) is intrinsic** - USB DMA, sample
unpacking, hardware CIC/FIR.  Not addressable without lowering sample rate.

Pipeline now runs at 5-7x realtime on target hardware (Intel i5/i7),
leaving substantial headroom for Raspberry Pi 5 deployment.

---

### ✓ Web UI: 2-Node Line-of-Position Display

When only 2 nodes have valid xcorr readings for a transmission, the solver
currently skips the fix entirely (a unique 2-D position requires >= 3 nodes).
However, the single TDOA measurement still defines a **hyperbolic line of
position (LOP)** that may be operationally useful - especially when combined
with other information such as:

- Prior full fixes on the same transmitter
- A second 2-node LOP from a different pair of nodes (different transmission
  or different event type on the same transmission)
- Operator knowledge of likely transmitter locations

**Proposed feature:**

Add a server config toggle (e.g. `pairing.min_nodes: 2` already exists as the
lower bound) and a separate display toggle to render LOPs on the map:

```yaml
pairing:
  min_nodes: 2          # Allow 2-node groups to proceed to the solver
  show_lop_only: true   # Render LOPs even when no fix is computed
```

Or via a web UI checkbox: **"Show 2-node lines of position"** (default off).

**Server-side changes:**
- `solve_fix()` with 2 nodes currently returns a result (somewhere on the
  hyperbola near the search centre) - the solver already runs, producing a
  bound-constrained point on the hyperbola arc with a large residual.
- A cleaner approach: when `node_count == 2`, skip the L-BFGS-B optimizer
  entirely and instead compute and return a `HyperbolaArc` object describing
  the TDOA hyperbola analytically (foci = the two nodes, deltad = TDOA x c).
  Include this in the fix GeoJSON as a `type: "lop"` feature rather than a
  `type: "fix"` marker.
- The existing hyperbola drawing code in `map_output.py` already generates
  hyperbola arcs for confirmed fixes - this would reuse that code path.

**Web UI changes:**
- LOP arcs displayed as dashed lines (vs. solid for fix-associated hyperbolas)
  in a distinct colour (e.g. amber) with a tooltip: "2-node LOP: nodeA <-> nodeB".
- Toggle in the map control panel to show/hide LOP-only results independently
  of full fixes.
- If multiple LOPs from different node pairs exist for the same transmission
  epoch, draw them all; their intersection visually approximates the fix.

**Relationship to outlier detection:**
The 2-node LOP display is also useful as a diagnostic: when the outlier
detector excludes a node from a 3-node fix, the server could optionally render
the 3 individual pair hyperbolas so the operator can see which arc was the
outlier.

**Open questions to resolve during implementation:**
- How to handle LOP display aging/hiding (same `max_age_s` window as fixes?).
- Whether to accumulate LOPs in the `fixes` DB table (with a `lop_only` flag)
  or keep them ephemeral (stream-only via SSE, not persisted).
- Folium/Leaflet layer ordering: LOPs should render below fix markers.
- How to present overlapping LOPs from the same approximate time clearly.

---

### ✓ Web UI: Per-Node Enable/Disable Control

Add a panel in the server web UI that lets an operator see all known nodes and
toggle whether each one contributes events to fix computation, without needing
shell access or the `manage_nodes.py` script.

**Use cases:**
- Temporarily exclude a node that is producing noisy or anomalous measurements
  (e.g. SDR overloads, bad antenna, known-bad clock) without removing it from
  the database or SSHing to the box.
- Commission a new node by watching it appear in the list before enabling it.
- Quickly isolate which node is causing poor fix residuals by toggling one at a time.

**Server-side (builds on Remote Node Registration item):**

The `nodes.enabled` column already carries this flag.  The server's event ingest
path (`POST /api/v1/events`) checks it:

```python
# In _check_auth / event ingest:
if not node_row["enabled"]:
    raise HTTPException(status_code=403, detail="Node is disabled")
```

New admin-only REST endpoints:

```
GET  /api/v1/nodes                        # list all nodes with status
PATCH /api/v1/nodes/{node_id}             # { "enabled": true/false }
```

These are auth-gated (admin role only once user auth is implemented; shared
token for now).

**Web UI panel - "Nodes" tab in the map control panel:**

- Table: one row per node - node_id, label, enabled toggle, last-seen age,
  event rate (events in last 60 s), clock source, config version.
- Enabled column is a live toggle (checkbox or ON/OFF button) that fires
  `PATCH /api/v1/nodes/{node_id}` immediately on click.
- Row colour: green = enabled + recently seen; amber = enabled but not seen
  in >60 s; grey = disabled; red = enabled but last_seen > 5 min (likely down).
- "Enable all" / "Disable all" buttons for quick fleet-wide changes.
- Table auto-refreshes every 10 s (or on SSE `new_fix` event).

**Interaction with fix computation:**

Disabled nodes' events are rejected at ingest time (403), so they never enter
the pairing pool.  No changes needed to the solver or pairing logic.

**Note:** Until the Remote Node Registration item is implemented, the `nodes`
table may not exist and the Nodes tab should be hidden or show a placeholder.
The `manage_nodes.py` script (`enable` / `disable` subcommands) serves as the
interim CLI alternative.

---

### ✓ Web UI: Group Management from GUI

Full group lifecycle management from the Groups tab - no CLI needed.

- **Create group**: "+ Create Group" button opens inline form with all required
  fields (group ID, label, sync station, frequency, lat/lon) and dynamic
  target channel list (add/remove rows).
- **Edit group**: "Edit" button on group detail panel opens the same form
  pre-populated with current values.
- **Unassign node**: "x" button on each member tag in the detail panel removes
  a node from the group.
- API: all endpoints already existed (`POST/PATCH/DELETE /api/v1/groups`).

---

### ✓ Web UI: Node Management from GUI

Full node lifecycle management from the Nodes tab - no CLI needed.

- **Register node**: "+ Register" button shows inline form (node ID + optional
  label).  Server generates secret; displayed once in a copyable modal overlay.
- **Regen secret**: Armed-confirmation button on each node card.  New secret
  shown in the same one-time modal.
- **Edit label**: Pencil icon next to label; inline input with Save/Cancel.
- **Edit config**: Expandable detail panel per node with `config_json` textarea
  and Save Config button.
- **Detail panel**: Shows config_version, registered_at, freq_group_id.
- API: added `POST /api/v1/nodes` (admin-create), `POST /api/v1/nodes/{id}/regen-secret`,
  extended `PATCH /api/v1/nodes/{id}` to accept `label`.
- DB: added `create_node()`, `update_node_secret()`, `update_node_label()`.

---

### ✓ Web UI: Node Frequency Groups

Allow nodes to be assigned to named **frequency groups**, where all nodes in a
group share the same sync signal and target channel list.  Groups are mutually
exclusive - a node belongs to at most one group.  Changing a group's frequency
plan immediately propagates to every member node via the existing config-push
mechanism (long-poll or next heartbeat cycle).

**Motivation:**

Different geographic clusters of nodes may monitor different LMR channels or use
different FM sync stations.  Without groups, each node's frequency plan must be
edited individually.  A group lets an operator say "all nodes in the south cluster
watch these three channels" and update all of them in one operation.

**Relationship to existing items:**

The `config_templates` table described in the Remote Node Registration item is a
general-purpose blob store.  Frequency groups are a more structured, first-class
concept built on top of it: a group is essentially a named template whose content
is restricted to the frequency-plan fields (`sync_signal` + `target_channels`),
with UI that makes membership and editing natural.

---

#### Data model

```sql
node_freq_groups (
    group_id    TEXT    PRIMARY KEY,   - e.g. "south-cluster"
    label       TEXT    NOT NULL,      - display name
    description TEXT,
    - Frequency plan (same field names as node config):
    sync_freq_hz        REAL NOT NULL,
    sync_station_id     TEXT NOT NULL,
    sync_station_lat    REAL NOT NULL,
    sync_station_lon    REAL NOT NULL,
    target_channels_json TEXT NOT NULL, - JSON array: [{frequency_hz, label}, ...]
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
)
```

`nodes` table gains one new nullable column:

```sql
ALTER TABLE nodes ADD COLUMN freq_group_id TEXT REFERENCES node_freq_groups(group_id);
```

A node with `freq_group_id = NULL` uses its own `config_json` frequency plan (or
server defaults) - no behaviour change for existing nodes.

When the server builds the effective config for a node (at registration or config
fetch), it merges in order:
1. Server defaults
2. Node's assigned `config_json` (per-node overrides)
3. **Group's frequency plan** (highest priority - overrides per-node freq settings)

This means an operator can still override non-frequency params per-node while the
group controls frequencies consistently.

---

#### REST API

```
GET    /api/v1/groups                         # list all groups
POST   /api/v1/groups                         # create group
GET    /api/v1/groups/{group_id}              # show group + member node_ids
PATCH  /api/v1/groups/{group_id}              # update label / freq plan
DELETE /api/v1/groups/{group_id}              # delete (nodes become ungrouped)

PATCH  /api/v1/nodes/{node_id}               # { "freq_group_id": "south-cluster" }
                                              # (also accepts null to unassign)
```

`PATCH /api/v1/groups/{group_id}` increments a `config_version` on every member
node and wakes any long-polling connections, causing the new frequency plan to
be delivered within seconds.

---

#### Web UI - "Groups" sub-tab within the Node Control tab

**Group list panel (left):**
- List of groups with member count and a "+ New Group" button.
- Selecting a group shows its details on the right.
- Delete button (with confirmation); disabled if group has members.

**Group detail panel (right):**
- Editable fields: label, description.
- Sync signal: frequency (Hz or MHz input), station ID, lat/lon.
- Target channels: editable list - add/remove rows of (frequency, label).
  Channel rows can be reordered; order is preserved in the config.
- "Save" button - updates the group and immediately pushes to all members.

**Node assignment:**
- In the Nodes table (from the Per-Node Enable/Disable item), add a "Group"
  column showing the assigned group name (or "--" if ungrouped).
- Clicking the group cell opens a dropdown of available groups + "None".
  Selecting one fires `PATCH /api/v1/nodes/{node_id}` immediately.
- Alternatively: the group detail panel has a "Members" list with an
  "Add node" picker and per-row remove buttons.

**Visual grouping on the map:**
- Node markers on the Leaflet map are colour-coded by group (ungrouped = grey).
- Group colour is auto-assigned from a small fixed palette; shown in the legend.

---

#### CLI support (`manage_nodes.py` extensions)

```
python scripts/manage_nodes.py --db ... group-list
python scripts/manage_nodes.py --db ... group-add south-cluster \
    --label "South Cluster" \
    --sync-freq 99900000 --sync-id KISW --sync-lat 47.65 --sync-lon -122.35 \
    --target 462562500:FRS_CH1 --target 462587500:FRS_CH2
python scripts/manage_nodes.py --db ... group-set-node seattle-south-01 south-cluster
python scripts/manage_nodes.py --db ... group-set-node seattle-south-01 --none
python scripts/manage_nodes.py --db ... group-remove south-cluster
```

---

#### Constraints and edge cases

- A node may belong to **at most one** group (enforced by the FK column, not a
  junction table).  Moving a node to a new group automatically removes it from
  the old one.
- Deleting a group sets `freq_group_id = NULL` on all member nodes (via `ON DELETE
  SET NULL` FK action).  Those nodes revert to their per-node frequency config.
- If a node has no per-node frequency config and no group, the server falls back
  to the server-wide default frequency plan (from `server.yaml`) if one is defined,
  or returns an incomplete config that the node will reject and log.
- Group changes are recorded in `node_config_history` for each affected node so
  the audit trail shows what changed and when.

---

### ✓ Project Name Change

Rename the project from `beagle_node` / `Beagle` to a new single-word name (TBD).
This is a comprehensive rename affecting package names, imports, documentation,
configuration, deployment artefacts, and the repository directory itself.

**Scope of changes:**

#### 1. Python packages (highest impact)

Two top-level packages must be renamed:
- `src/beagle_node/` -> `src/<newname>/` (~20 modules)
- `src/beagle_server/` -> `src/<newname>_server/` (~15 modules)

Every `import beagle_node` and `from beagle_node` statement must be updated:
- ~60+ imports of `beagle_node.*` across source and test files
- ~30+ imports of `beagle_server.*` across source and test files

**Strategy:** Rename the directories first, then use a global find-and-replace
for `beagle_server` (longer string first to avoid partial matches), then
`beagle_node`. Verify with `grep -r beagle_node` afterwards.

#### 2. Build and packaging (`pyproject.toml`)

- `[project] name = "beagle_node"` -> `"<newname>"`
- `[project.scripts]` entry point: `beagle-node = "beagle_node.main:main"` ->
  `<newname>-node = "<newname>.main:main"`
- `[tool.setuptools.packages.find] where = ["src"]` - no change needed
  (discovers packages by directory name)
- Delete generated metadata: `src/beagle_node.egg-info/`, any `dist/` artefacts

#### 3. Configuration files

- `config/server.example.yaml` - references to `beagle_node` in comments
- `config/node.example.yaml` - references in comments and example values
- Node bootstrap files (`/etc/beagle/bootstrap.yaml`) - no code reference but
  operators will need to know the new package/service names
- Any deployed `node.yaml` / `server.yaml` on live systems

#### 4. Systemd and deployment

- `etc/beagle-node.service` -> `etc/<newname>-node.service`
  - `Description=`, `ExecStart=`, unit filename all reference `beagle_node`
- Deployed systemd units on nodes must be updated (`systemctl disable` old,
  install new, `systemctl enable`)

#### 5. Scripts

- `scripts/manage_nodes.py` - docstring and argparse description mention `Beagle`
- `scripts/mock_generator.py` - docstring references `Beagle`
- `scripts/colocated_pair_test.py` - docstring
- `scripts/calibrate_sync.py` - docstring
- Other scripts in `scripts/` - grep for mentions

#### 6. Documentation

- `README.md` - title, all body references (~15+ occurrences)
- `ADMIN.md` - title and body references
- `TODO.md` - title and body references (this file)
- `CONTRIBUTING.md` (if it exists)
- `CLAUDE.md` files - project references
- Inline docstrings and module-level `"""` comments across the codebase

#### 7. Tests

- `tests/` - import paths (`from beagle_node...`, `from beagle_server...`)
- `conftest.py` files - same
- Test fixtures or mocks that reference package names

#### 8. Repository and directory

- Top-level directory: `Beagle/` -> `<NewName>/`
- Git remote / GitHub repo name (if applicable)
- Any CI/CD configuration (`.github/workflows/`, etc.)

#### 9. External references (manual, not automatable)

- Deployed node configurations on live Raspberry Pi systems
- Bookmarks, documentation links, or references shared with users
- Any external systems that reference the package name

**Execution checklist (when ready):**

1. Choose the new name
2. Create a branch: `git checkout -b rename-to-<newname>`
3. Rename directories: `src/beagle_node/` -> `src/<newname>/`, `src/beagle_server/` -> `src/<newname>_server/`
4. Global replace `beagle_server` -> `<newname>_server` (longer string first)
5. Global replace `beagle_node` -> `<newname>` (in code, configs, docs)
6. Global replace `Beagle` -> `<NewName>` (in prose, titles, comments)
7. Rename `etc/beagle-node.service` -> `etc/<newname>-node.service`
8. Update `pyproject.toml` (name, scripts, metadata)
9. Delete `src/beagle_node.egg-info/` and any stale build artefacts
10. Run full test suite to verify all imports resolve
11. `grep -ri beagle_node` across the entire repo to catch stragglers
12. Update deployed nodes (systemd units, bootstrap configs) - separate rollout

**Risk:** This is a single large commit that touches nearly every file. It will
conflict with any in-flight branches. Coordinate timing so no other feature
branches are open.

---

### ✓ Node: Python GC pauses causing RSPduo FIFO overflows

Periodic buffer backlog events occurred on both nodes every ~20-30 minutes.
Root cause was a memory leak causing the gen-2 heap to grow to ~890 MB, triggering
full collections that paused Python long enough to overflow the RSPduo FIFO.

**Resolved:** The underlying memory leak was fixed. After deploying the fix, no GC
pauses >50 ms have been observed (monitored via the `gc.callbacks` hook added
in `main.py`). The `gc.freeze()` mitigation is not needed - the heap no longer
grows unboundedly. Monitoring hook remains in place as an early-warning system.

---

### ✓ Server: map rendering efficiency (O(n^2) node pairs per poll)

`_collect_hyperbola_features()` in `map_output.py` iterates C(n,2) node pairs
for every fix on every `/map/data` poll. With many fixes this will not scale.

**Fixed 2026-03-19:** Added `app.state.map_geojson_cache: dict[float, dict]` in
`api.py`, keyed by `max_age_s`. The cache is cleared whenever a new fix is committed
(at the SSE emit point). All subsequent `/map/data` polls within that window return
the cached GeoJSON with zero DB queries or hyperbola recomputation. Age-preset button
clicks (different `max_age_s`) get their own cache slot, also invalidated on next fix.

---

### ✓ Server: suppress HeatMap `max_val` UserWarning

Folium deprecated the `max_val` parameter to `HeatMap`; passing it produced a
`UserWarning` on every map render.

**Fixed 2026-03-19:** Removed `max_val=10.0` from the `HeatMap(...)` call in
`map_output.py`. Folium now auto-scales intensity to the heaviest cell in the data.

---

### ✓ Node: onset_time_ns unit mismatch fix (3x drift)

`onset_time_ns` was growing ~3x faster than wall clock because `m.target_sample`
(in sync-decimated space, 256 kHz) was incorrectly treated as target-decimated
space (62.5 kHz), giving a 4x unit error in the sample->ns conversion.

**Fix** in `main.py`: convert sync-decimated sample to raw before computing offset:
```python
raw_event_sample = m.target_sample * _sync_dec_factor   # x8 -> raw domain
onset_offset_raw = raw_event_sample - sample_count
onset_offset_ns  = int(onset_offset_raw * 1e9 / receiver.config.sample_rate_hz)
onset_ns = buf_wall_ns + onset_offset_ns
```
This restored correct event matching and grouping on the server; 3-node groups
and LOP fixes appeared immediately after deployment.

---

### ✓ Node: RSPduo TCXO hardware timestamps (Part A)

Replaced per-buffer `time.time_ns()` calls (~400 usec NTP jitter) with TCXO-derived
timestamps anchored once at the first SDRplay callback.

**Implementation:** `dpkingston/SoapySDRPlay3`, branch `tdoa-hw-timestamps`
(permanent fork of `pothosware/SoapySDRPlay3`, based on
`rspduo-dual-independent-tuners`). Changes in `Streaming.cpp` / `SoapySDRPlay.hpp`:
- Capture `CLOCK_REALTIME` once at first `rx_callback` invocation (`anchor_wall_ns`)
- Extend 32-bit `firstSampleNum` TCXO counter to 64-bit with wraparound detection
- Record per-FIFO-slot first sample counter (`buffFirstSampleNums`)
- `acquireReadBuffer()` computes `timeNs` from TCXO counter and sets `SOAPY_SDR_HAS_TIME`

`rspduo.py` already checked `SOAPY_SDR_HAS_TIME` and used `sr_sync.timeNs`; no
Python changes required. Deployed to node-mapleleaf and node-greenlake.

pothosware declined to merge the timestamp changes; fork is now permanent.
Long-term plan: direct SDRplay API (see outstanding item).

---

### ✓ Node: pipeline_offset_ns calibration (node-mapleleaf / node-greenlake)

Calibrated using colocated_pair_test.py xcorr analysis (N=12 onset pairs,
SNR 1.8-2.2). Measured +86 usec mean TDOA_AB; split 50/50:
- `node-mapleleaf`: `pipeline_offset_ns: 43000`
- `node-greenlake`:  `pipeline_offset_ns: -43000`

---

### ✓ Node: GC pause monitoring (gc.callbacks)

Added `gc.callbacks` hook in `main.py` that logs any GC cycle lasting >50 ms
at WARNING level, to diagnose whether Python gen-2 collections are causing the
periodic RSPduo FIFO overflows (~20-30 min interval). If confirmed, `gc.freeze()`
will be applied after startup to eliminate gen-2 pauses on the stable heap.

---

### ✓ Server: --log-level CLI flag

Added `--log-level DEBUG|INFO|WARNING|ERROR` to `beagle-server` CLI.  Previously
uvicorn ran at INFO internally while the app logger had no CLI control, causing
DEBUG messages to appear on the console. The flag now controls both layers.

---

### ✓ Server: heartbeat access log lines demoted to DEBUG

`_HeartbeatAccessFilter` on `uvicorn.access` logger suppresses
`POST /api/v1/heartbeat` access log lines at INFO (returns `False` from filter
to drop the record outright). At `--log-level DEBUG` they pass through unchanged.

---

### ✓ Server: min_nodes=2 to enable 2-node LOP fixes

Server config `min_nodes` changed from 3 to 2. With only two RSPduo nodes
deployed, 2-node groups now proceed directly to the LOP solver rather than
requiring outlier rejection from a larger group.

---

### ✓ Server: LOP baseline-too-short log demoted to DEBUG

`_collect_hyperbola_features()` fires once per node pair per fix per map poll.
For co-located nodes (all pairs fail the baseline check) this produced repeated
INFO log noise. Demoted to DEBUG.

---

### ✓ Freq-hop: mid-transmission arrival suppression

Suppresses false onset events that fire when the RTL-SDR freq_hop node returns
to the target channel and finds a carrier already transmitting.  These events
had timing anchored to the block boundary (not the true carrier onset) and IQ
snippets with no noise->carrier transition (all carrier), producing unreliable
cross-correlation lags (observed as -325 usec and -410 usec outliers in
`colocated_pair_test.py`).

**Implemented - two-layer defence in `CarrierDetector`:**

1. **Idle window counting** (`_idle_window_count` / `_min_idle_for_onset = 2`):
   After `prime_state()`, at least 2 below-threshold windows must be observed
   before an onset is emitted.  Catches carriers already present at block start
   (count = 0) and single-window PLL settling artefacts that dip below threshold
   and immediately recover (count = 1).  Re-keys later in the same block are
   unaffected because the carrier drop adds genuine idle windows to the count.

2. **Snippet transition validation** (`_snippet_has_transition()`):  After
   `prime_state()`, each emitted event's IQ snippet is checked for sufficient
   dynamic range (>= 6 dB across per-window power).  Catches edge cases with high
   `min_hold_windows` where the ring fills with carrier before onset fires,
   pushing noise windows out of the snippet.

Structurally flawed events are dropped silently (logged at DEBUG).  The offset
transition from the same transmission is still captured if the carrier drops
within the target block.

Tests: `TestMidTransmissionSuppression` in `tests/unit/test_carrier_detect.py`
(8 tests covering both layers).  Full suite: 486 passed.

---

### ✓ Node SNR reporting: noise floor tracking + GET /api/v1/nodes/snr

Each node tracks a rolling noise floor estimate (dB) and computes onset/offset
detection thresholds relative to it.  Values are included in the heartbeat
payload and exposed via `GET /api/v1/nodes` and the map UI node tooltips.

---

### ✓ SoapySDRPlay3: RSPduo DT-IR support via pothosware fork

RSPduo dual-tuner independent-receiver (DT-IR) support was first developed as
a local fork with custom patches, then superseded when pothosware merged
equivalent functionality in the `rspduo-dual-independent-tuners` branch.

Current state: nodes run directly from `pothosware/rspduo-dual-independent-tuners`.
The `dpkingston/SoapySDRPlay3` fork (`tdoa-hw-timestamps` branch) adds TCXO
hardware timestamps on top of that branch. See the outstanding SoapySDR
long-term migration item and the hardware timestamps completed item above.

---

### ✓ User Registration and Authentication

Per-user accounts with role-based access control for the server API.

**Implemented (`user_auth = userdb`):**
- `POST /auth/register` - open for first user (bootstrap); admin-only thereafter
- `POST /auth/login` -> opaque session token (`secrets.token_urlsafe(32)`) stored in `user_sessions`
- `POST /auth/logout` - immediately invalidates token
- `GET /auth/me` - returns current user info
- `GET /auth/users` - admin: list all users
- `PATCH /auth/users/{id}` - admin: change role or reset password; viewer: change own password
- `DELETE /auth/users/{id}` - admin only; cascade-deletes sessions
- Roles: `admin` (full access) and `viewer` (read-only endpoints)
- Password storage: PBKDF2-HMAC-SHA256, 260 000 iterations, 16-byte random salt (OWASP 2023)
- Session lifetime: 24 h default, configurable via `server.session_lifetime_hours`
- Two independent auth settings: `server.node_auth` (none|token|nodedb) and `server.user_auth` (none|token|userdb) (configured in `server.node_auth` and `server.user_auth`)
- Users and sessions stored in `tdoa_registry.db` (permanent; survives operational DB wipe)
- All endpoints covered by integration tests in `tests/integration/test_auth.py` (34 tests)
- Administration documented in `ADMIN.md`

**Not yet implemented:** 2FA (TOTP / WebAuthn). The schema has no hooks for it yet; add when needed.

---

### ✓ Hide Fixes (non-destructive reset)

Replace the destructive "Reset Fix History" (DELETE) with a non-destructive
"Hide Fixes" mechanism. Hide state is **per browser session** (localStorage)
so multiple operators sharing the same server see independent views.

**Core mechanism - client-side only, no server or DB changes:**
- `localStorage.tdoa_hidden_before_t` stores a Unix timestamp (float seconds).
  Default: 0 (nothing hidden).
- "Hide Fixes" button: sets `localStorage.tdoa_hidden_before_t = Date.now() / 1000`.
- "Unhide All" button: sets `localStorage.tdoa_hidden_before_t = 0`.
- `loadFixes(userMaxAgeS)` computes the effective cutoff:
  ```js
  const hiddenBefore = parseFloat(localStorage.tdoa_hidden_before_t || 0);
  const cutoff = Math.max(hiddenBefore, Date.now()/1000 - userMaxAgeS);
  const effectiveMaxAgeS = Date.now()/1000 - cutoff;
  fetch(`/map/data?max_age_s=${effectiveMaxAgeS}`)
  ```
  This unifies the display-window control and the hide control into the single
  `max_age_s` parameter the server already accepts - no new endpoints needed.
  When display window is "ALL" (`userMaxAgeS = 0`), `cutoff = hiddenBefore`
  and `effectiveMaxAgeS = now - hiddenBefore`.

**Map control panel:**
- Rename "Reset Fix History" -> "Hide Fixes" (no confirmation needed; reversible).
- Add "Unhide All" button beside it.
- Show current hide state: "Hidden before HH:MM:SS" or "All visible".

**Keep DELETE endpoint:** retain `DELETE /api/v1/fixes` for genuine data
purges (disk space reclaim, fresh deployment). Document that Hide is preferred
for operational use.

**Note:** Hide state is lost on tab/browser close. This is intentional - it is
a personal view preference, not persistent server state.

---

### ✓ Fixed Time Window Display (start-time to end-time)

Allow the map to display fixes within an **absolute** time window rather than
only a rolling age from now. Complements the existing age-preset buttons.

**Use case:** reviewing a specific incident window (e.g. "show all fixes from
14:30 to 15:00") without new fixes entering the view as time passes.

**Implementation - client-side only:**
- Add two datetime-picker inputs to the control panel: "From" and "To"
  (or "From" + duration spinner: +5 min / +15 min / +1 h / custom).
- When a fixed window is active, `loadFixes()` passes absolute bounds:
  ```js
  // Convert to max_age_s (from now) for the existing API:
  const nowS = Date.now() / 1000;
  const fromS = windowStart.getTime() / 1000;
  const toS   = windowEnd.getTime()   / 1000;
  // Fetch with max_age_s = now - fromS, then filter client-side to toS
  fetch(`/map/data?max_age_s=${nowS - fromS}`)
    .then(geojson => filterFeaturesAfter(geojson, toS))
  ```
  Server-side filtering handles the start bound cheaply; client-side
  filtering drops fixes newer than `toS` (small set, negligible cost).
- "Clear window" button returns to rolling age-preset mode.
- Fixed window and rolling age-preset are mutually exclusive; activating
  one clears the other.

**Interaction with Hide Fixes:** `hiddenBefore` still applies inside the
fixed window - the effective lower bound is `max(hiddenBefore, fromS)`.

---

### ✓ Fix Popup: Human-Readable Timestamp on Hover

When the user mouses over (or clicks) a fix marker on the map, show the
time of the fix in a human-readable form alongside the existing fields.

**Current state:** the fix popup shows lat/lon, residual, node count, etc.
`onset_time_ns` is stored in the DB and available in the GeoJSON properties
but is displayed as a raw integer nanosecond value.

**Desired:** format `onset_time_ns` as local wall-clock time, e.g.
`"2025-11-14  14:32:07.412 local"` and/or UTC equivalent.

**Implementation:**
- In `build_fix_geojson()` (`map_output.py`), include `onset_time_ns` in
  the GeoJSON feature properties (already present as raw int).
- In the map JS popup template, convert to a readable string:
  ```js
  const t = new Date(props.onset_time_ns / 1e6);  // ns -> ms
  const label = t.toLocaleString() + ' local';
  ```
- Add a UTC line: `t.toUTCString()`.
- Position the timestamp at the top of the popup so it is the first thing
  the operator sees, since the most common question is "when did this happen?"

**Note:** `onset_time_ns` is the node's best wall-clock at carrier onset
(NTP-disciplined, +/-1-10 ms for most nodes). It is not the server
`computed_at` time. Both could be shown; `onset_time_ns` is more useful
operationally.

---

### ✓ Remote Node Registration and Config Management

Allow nodes to register with the server and receive their operating configuration
over the network, rather than requiring a local `node.yaml` on every box.
The server becomes the authoritative source for node config; a node's local file
is only a bootstrap/fallback.

**Background - current model:**
Nodes read a local `node.yaml` at startup. Authentication is a single shared Bearer
token (`reporter.auth_token`) sent on every `POST /api/v1/events`. There is no
per-node identity, no server awareness of which nodes exist, and no way to push
config changes without an SSH session to each box.

**Goal:** Each node has a stable identity, authenticates individually, can fetch its
full operating config from the server, and the server can push config updates at any
time. Nodes do not need a full local config beyond the minimum needed to find and
authenticate to the server.

---

#### Node Identity and Authentication

Each node is pre-provisioned with a **node secret** (a random 256-bit token, stored
in a minimal local bootstrap file, e.g. `/etc/beagle/bootstrap.yaml`):

```yaml
# bootstrap.yaml  - the only file a node needs locally
server_url: "https://tdoa.example.com"
node_id: "seattle-north-01"
node_secret: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

On all API calls the node sends:
```
Authorization: Bearer <node_secret>
X-Node-ID: seattle-north-01
```

The server stores a `nodes` table:

```sql
nodes (
  node_id       TEXT PRIMARY KEY,
  secret_hash   TEXT NOT NULL,      - bcrypt hash of node_secret
  label         TEXT,               - human-friendly display name
  location_lat  REAL,
  location_lon  REAL,
  location_alt_m REAL,
  registered_at REAL,
  last_seen_at  REAL,
  last_ip       TEXT,
  enabled       INTEGER DEFAULT 1,  - admin can disable a node
  config_json   TEXT                - current assigned config (JSON), NULL = use server defaults
)
```

Node secrets are **never** stored in plaintext on the server (bcrypt-hashed, same
as user passwords). An admin generates the secret and provisions the bootstrap file
out-of-band (e.g. via Ansible, printed QR, or the admin web UI).

Backward compatibility: the existing single shared `server.auth_token` continues to
work when `server.node_auth = token`. Per-node auth activates under `node_auth = nodedb`
(or `userdb` once that item is done).

---

#### Registration Flow

1. **Node calls `POST /api/v1/nodes/register`** on first boot (or whenever
   bootstrap.yaml changes). Body:

   ```json
   {
     "node_id": "seattle-north-01",
     "software_version": "3.2.1",
     "sdr_mode": "freq_hop",
     "location_lat": 47.71,
     "location_lon": -122.34,
     "location_alt_m": 85.0
   }
   ```

   Authentication via `Authorization: Bearer <node_secret>` as above.

2. **Server response (200):**

   ```json
   {
     "status": "registered",   // or "already_registered"
     "node_id": "seattle-north-01",
     "config_version": 7,
     "config": { /* full NodeConfig-equivalent JSON, see below */ }
   }
   ```

   If the node is unknown (first ever registration), status = `"pending"` and
   config = null; an admin must approve and assign a config via the web UI or
   `PATCH /api/v1/nodes/{node_id}`. The node polls (see below) until approved.

3. **On startup after initial registration**, a node that already has a local
   cached config (see SectionConfig Caching) can skip registration and go straight
   to the config-fetch poll, using a lower-cost `GET /api/v1/nodes/{node_id}/config`.

---

#### Config Fetch and Long-Poll Update

**Immediate fetch:**

```
GET /api/v1/nodes/{node_id}/config
```
Returns the current assigned config JSON plus a `config_version` integer and
`config_etag` string. The node caches this to local disk
(`/var/cache/beagle/node_config.json`) so it can start on next boot without network.

**Long-poll for updates (preferred push mechanism):**

```
GET /api/v1/nodes/{node_id}/config?wait=60&since_version=7
```

Server holds the connection for up to 60 s. Returns immediately if the server's
current version > `since_version`; otherwise returns 304 after the timeout.
The node runs this in a background thread and applies a new config without
restarting (where possible) or schedules a controlled restart.

This avoids the complexity of WebSocket or SSE for a low-update-rate control
channel while still achieving near-real-time config delivery.

**Fallback polling** (simpler alternative for nodes behind strict NAT/firewall):
Node polls `GET /api/v1/nodes/{node_id}/config` every `config_poll_interval_s`
(default 60 s, configurable in bootstrap). Uses `If-None-Match: <etag>` -> 304
when unchanged.

---

#### Remotely Controlled Parameters

The server-assigned config covers all parameters a node currently reads from
`node.yaml`. Grouped by sensitivity:

**Identification / location (set at registration, admin-editable):**
- `node_id`, `label`, `location_lat/lon/alt_m`

**SDR hardware (requires node restart to apply):**
- `sdr_mode` (freq_hop / rspduo / two_sdr / single_sdr)
- `sample_rate_hz`, `gain_db` (fixed or `"auto"`)
- `buffer_size`, `settling_samples`
- Mode-specific hardware args (e.g. `master_device_args`, `rtl_sdr_binary`)

**Frequency plan (hot-reloadable without restart):**
- `sync_signal.primary_station` (frequency, label, location)
- `target_channels[]` (list of `{frequency_hz, label}`)

**Signal processing thresholds (hot-reloadable):**
- `carrier.onset_db`, `carrier.offset_db`
- `carrier.window_samples`, `carrier.min_hold_windows`, `carrier.min_release_windows`
- `sync_signal.sync_period_ms`, `sync_signal.min_corr_peak`, `sync_signal.max_sync_age_ms`

**Reporting (hot-reloadable):**
- `reporter.server_url` (allows migrating server without touching bootstrap)
- `reporter.queue_size`, `reporter.timeout_s`, `reporter.retry_delays_s`

**Clock (hot-reloadable):**
- `clock.source`, `clock.calibration_offset_ns`

Parameters that require hardware re-init are flagged in the schema
(`requires_restart: true`) so the node can decide whether to restart immediately
or defer to the next maintenance window.

---

#### Server-Side: Pushing Config Updates

Admin UI / API for operators:

```
PATCH /api/v1/nodes/{node_id}
{
  "config": {
    "carrier": { "onset_db": -28.0 },
    "target_channels": [
      { "frequency_hz": 462562500, "label": "FRS_CH1" }
    ]
  },
  "restart_required": false   // server's assessment; node may override
}
```

- Merges the patch into the node's stored `config_json` (JSON Merge Patch, RFC 7396).
- Increments `config_version`.
- Any node currently long-polling for that `node_id` gets the response immediately.
- Change is logged in a `node_config_history` table (version, changed_by, changed_at, diff_json)
  for audit and rollback.

```
POST /api/v1/nodes/{node_id}/config/rollback?to_version=5
```

Rolls back to a previous stored version.

**Bulk / template configs:**
An admin can define a `config_template` (stored in `config_templates` table) and
assign it to multiple nodes. Per-node overrides are merged on top. Useful for
deploying a frequency plan change to all nodes at once.

---

#### Node Status Reporting

On the existing `POST /api/v1/events` path, each `CarrierEvent` already carries
`node_id`. Add a lightweight **heartbeat** endpoint to surface per-node health
without coupling it to the event stream:

```
POST /api/v1/nodes/{node_id}/heartbeat
{
  "config_version": 7,
  "software_version": "3.2.1",
  "uptime_s": 3600,
  "events_submitted": 1240,
  "events_dropped": 0,
  "queue_depth": 2,
  "sdr_overflows": 0
}
```

Server records in `nodes.last_seen_at` and a `node_heartbeats` table.
Map control panel gains a "Nodes" tab showing per-node status (online/offline,
config version, event rate).

---

#### Schema Summary (new tables)

```sql
nodes (
  node_id TEXT PRIMARY KEY, secret_hash TEXT, label TEXT,
  location_lat REAL, location_lon REAL, location_alt_m REAL,
  registered_at REAL, last_seen_at REAL, last_ip TEXT,
  enabled INTEGER DEFAULT 1,
  config_version INTEGER DEFAULT 0,
  config_json TEXT,                  - current effective config (merged)
  config_template_id TEXT REFERENCES config_templates(template_id)
)

config_templates (
  template_id TEXT PRIMARY KEY, label TEXT, config_json TEXT,
  created_at REAL, updated_at REAL
)

node_config_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT, version INTEGER, config_json TEXT,
  changed_by TEXT, changed_at REAL, diff_json TEXT
)

node_heartbeats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT, received_at REAL, payload_json TEXT
)
```

---

#### Admin CLI: `scripts/manage_nodes.py`

A standalone script is provided at `scripts/manage_nodes.py` for managing node
records directly in the SQLite database, without requiring the REST API or web UI
to be running.  It creates the `nodes` and `node_config_history` tables on first
use (idempotent - safe to run against an existing server database).

```
# Point at the registry DB directly:
python scripts/manage_nodes.py --db data/tdoa_registry.db list
python scripts/manage_nodes.py --db data/tdoa_registry.db add seattle-north-01 \
    --lat 47.71 --lon -122.34 --alt 85 --label "Seattle North Roof"
python scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \
    --config-file configs/seattle-north-01.yaml
python scripts/manage_nodes.py --db data/tdoa_registry.db show seattle-north-01
python scripts/manage_nodes.py --db data/tdoa_registry.db disable seattle-north-01
python scripts/manage_nodes.py --db data/tdoa_registry.db remove seattle-north-01
python scripts/manage_nodes.py --db data/tdoa_registry.db regen-secret seattle-north-01

# Or read the DB path from the server config:
python scripts/manage_nodes.py --server-config config/server.yaml list
```

Secrets are currently hashed with SHA-256 (prefixed `sha256:`).  When the full
auth system is implemented with bcrypt, existing hashes are migrated on first login.

---

#### Compatibility and Migration

- Nodes running the old code (no registration support) continue to work unchanged
  as long as `server.node_auth = token` and the shared token is set.
- New nodes start with `node_auth = nodedb`; the two modes can coexist during rollout.
- The server must never serve a config to an unauthenticated or unknown node.
- Bootstrap file (`bootstrap.yaml`) stays minimal; never embed full SDR config there.

---

### ✓ Mock Generator - PTT Onset/Offset Pattern

`synthesise_onset()` + `synthesise_offset()`: each transmission follows
onset -> [N s hold] -> offset -> [M s gap] -> next onset.
Offset events reuse each node's onset `event_id` (server upsert/amend path),
carry `event_type="offset"` and fresh `onset_time_ns` + `sync_delta_ns`.
Both onset and offset fixes are polled and printed separately.
New CLI: `--no-offset`, `--duration-mean-s`, `--gap-mean-s`.
Scenario YAML: `transmission_duration_mean/sigma_s`, `inter_transmission_gap_mean/sigma_s`.

---

### ✓ Heat Map Mode

Persistent Gaussian-weighted heat map accumulated in SQLite (`heatmap_cells` table).
Each fix spreads weight across nearby cells using `exp(-(di^2+dj^2)/(2sigma^2))`.
Rendered as a togglable Folium `HeatMap` layer with Leaflet `LayerControl`.
Config: `map.heatmap_cell_m`, `map.heatmap_sigma_cells`.
API: `DELETE /api/v1/heatmap` clears accumulated data.
Control panel: "Reset Heat Map" button.

---

### ✓ Web Page Control - Reset Fix History

`DELETE /api/v1/fixes` endpoint (auth-gated) truncates the fixes table.
"Reset Fix History" button in the map control panel with confirmation dialog.

---

### ✓ Live Map Updates via SSE

`GET /api/v1/fixes/stream` SSE endpoint; page auto-reloads on `new_fix` event.
LIVE / NEW FIX / OFFLINE status badge in control panel header.
25 s keepalive comments to prevent proxy timeouts.

---

### ✓ Web Page Control - Dynamic Aging Window

Age-preset buttons (1 m / 5 m / 15 m / 1 h / 6 h / 24 h / ALL) in the map
control panel.  Clicking a button calls `loadFixes(maxAgeS)` which fetches
`GET /map/data?max_age_s=N` (GeoJSON FeatureCollection) and re-renders only
the fix/hyperbola layer without a full page reload.  SSE `new_fix` events also
call `loadFixes()` instead of `window.location.reload()`.
`build_fix_geojson()` in `map_output.py` produces the GeoJSON; `build_map()`
produces the static shell (node markers, heatmap) with no embedded fix data.

---

### ✓ Map Control Panel

Dark translucent panel (top-right): server label, time, last fix age, age-preset
buttons, fix count (updated dynamically by `loadFixes()`). Extensible: new controls
require only a new key in `_render_control_panel`, a row/button in `_PANEL_HTML`,
and a JS handler reading `TDOA.<key>`.

---

### ✓ Fix Layer Ordering + Hyperbola Generator

Fixed newest-on-top ordering (Leaflet z-order requires oldest-first insertion).
Replaced broken grid-sampling hyperbola generator with analytic parametric form
using `x = sgn*a*cosh(t)`, `y = b_h*sinh(t)` in a local flat-earth frame.

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](LICENSE).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](LICENSE).
