# Beagle - Open Source RDF

Software for a distributed Time Difference of Arrival (TDOA) system that
locates land mobile radio (LMR) transmissions to neighbourhood-level accuracy
using low-cost RTL-SDR hardware.

The system has two components:

- **Nodes** - Raspberry Pi (or other Linux system) + SDR receivers deployed at fixed locations.  Each
  node listens for LMR carrier events and measures their timing relative to an
  FM broadcast pilot tone used as a shared clock reference.  Nodes POST their
  measurements to the central server.

- **Aggregation Server** - A central server (laptop, Pi, or cloud VM typically running Linux) that
  collects measurements from all nodes, computes hyperbolic position fixes, and
  serves a live map.  The server is where you see results.

You need **one server** and **two or more nodes** (three or more for a unique
position fix; two nodes produce a line of position).

---

## Getting Started

**Setting up a node?**  Follow [Installation (node)](#node-pi--sdr) then
[Node Quick Start](#node-quick-start).  A running server is needed to receive
measurements.

**Setting up the server?**  Follow [Installation (server)](#aggregation-server-host)
then [Aggregation Server](#aggregation-server).  You can test the server
immediately using the [Mock Event Generator](#mock-event-generator-demo-without-hardware)
-- no SDR hardware needed.

**Managing a multi-node deployment?**  See [Node Management](#node-management)
for registering nodes, pushing configs, and monitoring health.

---

## Contents

### Installation
- [Time synchronisation](#nodes-and-servers--time-synchronisation) - chrony setup (all machines)
- [Node (Pi/Linux + SDR)](#node-pi--sdr)
- [Server](#aggregation-server-host)

### Node
- [SDR Modes](#sdr-modes) - freq_hop, rspduo, two_sdr, single_sdr
- [Node Quick Start](#node-quick-start) - configure, verify, run
- [Calibration](#calibration) - threshold tuning, crystal check, accuracy verification
- [Production Deployment](#production-deployment-systemd)

### Server
- [Aggregation Server](#aggregation-server) - config, startup, live map, controls
- [Mock Event Generator](#mock-event-generator-demo-without-hardware) - test without hardware
- [Node Management](#node-management) - register nodes, push configs

### Reference
- [How It Works](#how-it-works) - timing model, signal processing, solver
- [Scripts](#scripts) - verification, analysis, provisioning tools
- [Tests](#tests)
- [Project Layout](#project-layout)

---

## How It Works

Time Difference of Arrival locates a radio transmitter by measuring how much
earlier or later its signal arrives at each receiver.  If node A hears a
transmission 3 usec before node B, the transmitter lies somewhere on a hyperbola
whose foci are A and B - with a 3 usec x c ~ 900 m path difference.  Two or
more such hyperbolas from different node pairs intersect at the transmitter's
location.  The challenge is measuring those sub-microsecond arrival-time
differences with inexpensive, unsynchronised hardware.

Beagle solves the synchronisation problem without GPS hardware at each node by
using the FM stereo pilot tone (19,000 Hz exactly) as a shared timing beacon.
Every FM broadcast station transmits this tone locked to a GPS-traceable
frequency standard.  Each node continuously cross-correlates the pilot against
a reference template, producing a `SyncEvent` timestamp every 10 ms with
sub-microsecond precision.  When a land mobile radio (LMR) carrier is detected,
the node records `sync_delta_ns = target_onset_sample - sync_event_sample`,
expressed in nanoseconds on the local sample clock.  Because both measurements
use the same unbroken ADC clock, the absolute clock offset cancels out entirely
-- only the interval between two events on that clock matters.

The aggregation server collects `sync_delta_ns` reports from all nodes,
corrects for the known FM transmitter-to-node propagation delay (computed from
FCC-documented station coordinates), and computes
`TDOA_AB = sync_delta_A - sync_delta_B`.  A scipy L-BFGS-B solver minimises
the squared residuals across all node pairs to produce a latitude/longitude fix,
which is logged to SQLite and displayed on a live Folium map.

```
   FM broadcast station (19 kHz pilot)       LMR transmitter
            |                                       |
   SDR (sync channel)                    SDR (target channel)
            |  decimated IQ                         |  decimated IQ
            v                                       v
   FMPilotSyncDetector                     CarrierDetector
            |                                       |
            |  SyncEvent (every 10 ms)              |  CarrierOnset
            |                                       |
            +-----------> DeltaComputer <-----------+
                                |
                                v
                    sync_delta_ns = target_onset - sync_event
                         (same ADC clock - offset cancels)
                                |
                                v
                         EventReporter  -->  HTTP POST /api/v1/events
                                                       |
                                                       v
                                           Aggregation Server
                                      pair events * correct path delay
                                      solve hyperbolic fix * update map
```

**Further reading:**

- Knapp & Carter (1976) - [The Generalized Correlation Method for Estimation of Time Delay](https://www.semanticscholar.org/paper/The-generalized-correlation-method-for-estimation-Knapp-Carter/29c74aad1986ff2e907e084820e990a0544e743a) - foundational cross-correlation technique underlying the FM pilot detector
- Chan & Ho (1994) - [A Simple and Efficient Estimator for Hyperbolic Location](https://www.semanticscholar.org/paper/A-simple-and-efficient-estimator-for-hyperbolic-Chan-Ho/fc51fb822024805533ff9eef4f7e486b38437109) - the closed-form TDOA solver this system is based on
- Howland, Maksimiuk & Reitsma (2005) - [FM Radio Based Bistatic Radar](https://www.theiet.org/media/11278/fm-radio-based-bistatic-radar.pdf) - demonstrates FM broadcasts as passive location reference signals
- Abramson (2020) - [Thesis: Locating Transmitters with TDOA and RTL-SDRs](https://www.rtl-sdr.com/thesis-on-locating-transmitters-with-tdoa-and-rtl-sdrs/) - end-to-end RTL-SDR TDOA system with source and tooling
- Lanius, MIT Lincoln Laboratory (GRCon 2023) - [Wideband TDOA Geolocation with GNU Radio](https://events.gnuradio.org/event/18/contributions/252/) ([video](https://www.youtube.com/watch?v=o9gbsxsLH9Q)) - practical SDR-based TDOA geolocation techniques

---

## SDR Modes

| Mode | Hardware | Clock alignment | Accuracy |
|------|----------|----------------|----------|
| `freq_hop` | 1 x RTL-SDR | Same unbroken ADC clock | 1-5 usec |
| `rspduo` | 1 x SDRplay RSPduo | Shared TCXO + single ADC | ~1-2 usec |
| `two_sdr` + GPS 1PPS | 2 x RTL-SDR + GPS | GPS 1PPS injected into both inputs (untested) | <1 usec |
| `single_sdr` | 1 x wideband SDR | Same ADC clock (untested) | <1 usec |

**`freq_hop`** is recommended for single-dongle development.  The RTL2832
ADC runs continuously while the R820T tuner alternates between the FM sync
frequency and the LMR target frequency using pyrtlsdr's synchronous
`read_bytes()` API in a background thread.

**`rspduo`** is recommended for production when GPS 1PPS injection hardware
is undesirable.  The RSPduo's two tuners share one ADC clock and one USB
connection - no inter-channel jitter, no coverage gaps, no settling time.
Requires the SDRplay API installer (from sdrplay.com) and the SoapySDRPlay3
plugin (built from source on Debian - see
[docs/setup-rspduo-debian.md](docs/setup-rspduo-debian.md)).

**`two_sdr`** and **`single_sdr`** are theoretically supported but have not
been tested due to lack of equipment.

---

## Installation

### Nodes and servers - time synchronisation

Beagle requires all nodes and the server to have good time synchronisation.
Use `chrony` with the **same NTP sources** on every machine to minimise
clock divergence and improve event pairing.  If you have access to more
accurate timing (e.g. GPS with pulse-per-second), so much the better.

```bash
sudo apt install chrony

# Disable systemd-timesyncd if active
sudo systemctl disable --now systemd-timesyncd 2>/dev/null || true

# Add Google NTP sources (or substitute your preferred high-accuracy sources)
sudo tee /etc/chrony/sources.d/google.sources > /dev/null << 'EOF'
server 216.239.35.0 iburst
server 216.239.35.4 iburst
server 216.239.35.8 iburst
server 216.239.35.12 iburst
EOF

sudo systemctl restart chrony
```

Verify after ~30 s:
```bash
chronyc sources      # Google servers should appear with * or +
chronyc tracking     # System time should be within a few ms of NTP
```

Events from nodes using different NTP sources (or `systemd-timesyncd` vs
`chrony`) can fail to pair because their timestamps may diverge beyond the
`max_sync_age_ms` window (200 ms by default).

### Node (Pi/Linux + SDR)

For **RSPduo nodes**: Follow the the instructions in
[docs/setup-rspduo-debian.md](docs/setup-rspduo-debian.md) instead.
They include setup of SDRplay API + SoapySDRPlay3.

```bash
# System packages
# python3-soapysdr puts the SoapySDR Python bindings on the system path;
# --system-site-packages below makes them visible inside the venv.
sudo apt install python3-pip python3-venv python3-soapysdr soapysdr-tools rtl-sdr chrony gpsd

# Clone the repository and create the Python environment
git clone https://github.com/dpkingston/beagle_rdf.git
cd beagle_rdf
python3 -m venv --system-site-packages env
source env/bin/activate
pip install -e .
```

### Aggregation server host

```bash
# System packages
sudo apt install python3-pip python3-venv chrony

# Clone the repository and create the Python environment
git clone https://github.com/dpkingston/beagle_rdf.git
cd beagle_rdf
python3 -m venv env
source env/bin/activate
pip install -e ".[server]"
```

---

## Node Quick Start

Follow the [Installation](#installation) steps first, then choose one of the
two configuration approaches:

- **Local config** - the node reads a full `node.yaml` on startup.  Suitable
  for development and single-node deployments.
- **Remote config** (recommended for multi-node deployments) - the node starts
  with a minimal `bootstrap.yaml` and fetches its operating config from the
  server.  The server can push config updates at any time without SSHing to
  each box.  See [Node Management](#node-management) for the setup workflow.

### 1 - Configure

#### Option A: local config

```bash
cp config/node.example.yaml config/node.yaml
$EDITOR config/node.yaml    # set node_id, location, frequencies, server_url
```

Key fields in `node.yaml`:

| Field | Description |
|-------|-------------|
| `node_id` | Unique node identifier (lowercase, hyphens OK) |
| `location` | GPS coordinates of the node antenna |
| `sdr_mode` | `freq_hop` / `rspduo` / `two_sdr` / `single_sdr` |
| `freq_hop.sync_frequency_hz` | FM station for sync (e.g. KISW 99.9 MHz) |
| `freq_hop.target_frequency_hz` | LMR channel to monitor |
| `sync_signal.primary_station` | FCC-documented transmitter coordinates |
| `reporter.server_url` | Aggregation server URL |

#### Option B: remote config (bootstrap)

In this mode, the full node configuration (frequencies, gain, thresholds, etc.)
lives on the server and is fetched automatically.  The node only needs a minimal
`bootstrap.yaml` with its identity and server URL.

**On the server:** create a node config file the same way as Option A, register
the node, and push the config.  See [Node Management](#node-management) for
the full workflow - step 2 (registration) prints the secret you'll need below.

**On the node:** create the bootstrap file using the secret from registration:

```bash
cp config/bootstrap.example.yaml /etc/beagle/bootstrap.yaml
$EDITOR /etc/beagle/bootstrap.yaml
```

```yaml
server_url: "https://beagle.example.com"
node_id: "seattle-north-01"
node_secret: "replace-with-secret-from-registration"
```

This is the **only** file the node needs locally.  The operating config is
fetched from the server on startup and updated automatically whenever the
server config changes.

### 2 - Verify hardware and pilot detection

The verification script depends on your SDR hardware:

**RTL-SDR (`freq_hop` mode)** - confirm FM pilot detection before a full run:
```bash
python3 scripts/verify_sync.py --config config/node.yaml --duration 10
```
Expected: steady sync event rate, mean `corr_peak` >= 0.5.

**RSPduo (`rspduo` mode)** - end-to-end dual-tuner pipeline test:
```bash
python3 scripts/verify_rspduo.py --sync-freq 99.9e6 --target-freq 462.5625e6
```
Expected: sync rate >= 95/s, 0 overflows, measurements produced when an LMR
transmission is present.  See [docs/setup-rspduo-debian.md](docs/setup-rspduo-debian.md)
for installation and setup of the SDRplay API and SoapySDRPlay3.

### 3 - Calibrate carrier detection thresholds

Tune the SDR to your target frequency and key your transmitter a few times.
`check_target.py` reports the noise floor and peak signal level and recommends
`onset_db` / `offset_db` values:

```bash
python3 scripts/check_target.py --freq 462.5625e6 --gain 30 --duration 30
```

Copy the recommended values into the `carrier:` section of `node.yaml`:

```yaml
carrier:
  onset_db:  -18   # from check_target.py output
  offset_db: -28
```

Skip this step for a first smoke-test with the default thresholds (-30/-40 dBFS),
but run it before production deployment.  See [Calibration](#calibration)
for the full calibration procedure.

### 4 - Run the node

```bash
# Local config
python -m beagle_node --config config/node.yaml

# Remote config (bootstrap)
python -m beagle_node --bootstrap /etc/beagle/bootstrap.yaml

# Synthetic IQ (no hardware needed)
python -m beagle_node --config config/node.yaml --mock --mock-duration 30
```

### 5 - Check health

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "node_id": "seattle-north-01",
  "uptime_s": 42.3,
  "events_submitted": 7,
  "clock_source": "gps_1pps",
  "clock_uncertainty_ns": 456,
  "crystal_correction": 1.0000432
}
```

---

## Aggregation Server

The aggregation server receives events from all nodes, pairs them, computes
hyperbolic fixes, and serves a live Folium map.  It can run on any machine
reachable by the nodes - a laptop, a Pi, or a cloud VM.

### 1 - Install server extras

The server dependencies (FastAPI, uvicorn, aiosqlite, folium) are not
installed by default.  With the venv active:

```bash
pip install -e ".[server]"
```

### 2 - Create a server config

```bash
cp config/server.example.yaml config/server.yaml
$EDITOR config/server.yaml
```

The only fields you must change before first use:

| Field | Default | What to set |
|-------|---------|-------------|
| `solver.search_center_lat` | 47.7 | Latitude of the centre of your deployment area |
| `solver.search_center_lon` | -122.3 | Longitude of the centre of your deployment area |
| `server.auth_token` | (empty) | A secret string, or leave empty during development |
| `server.node_auth` | `token` | How nodes authenticate event POSTs: `none`, `token` (default), `nodedb` (per-node secrets) |
| `server.user_auth` | `token` | How humans access the UI: `none`, `token` (default), `userdb` (per-user accounts) |

All other defaults are reasonable for a first run.

### 3 - Start the server

```bash
env/bin/beagle-server --config config/server.yaml
```

Two SQLite databases are created automatically on first run:
- `data/tdoa_data.db` - operational data (events, fixes, heatmap)
- `data/tdoa_registry.db` - permanent configuration (nodes, users, sessions)

The operational database can be deleted at any time for a clean start;
node registrations and user accounts in the registry are unaffected.

Expected startup output:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8765 (Press CTRL+C to quit)
```

### 4 - Create the first admin user (userdb mode)

If you set `server.user_auth: userdb` in your config, register the first admin
account while the server is running.  You can do this from the browser or the
command line.

**Browser:** Open `http://localhost:8765/map`.  The login overlay has a
"Create" link while no users exist - fill in a username, password, and the
`admin` role.

**Command line:**

```bash
curl -s -X POST http://localhost:8765/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "your-strong-password", "role": "admin"}'
```

Once the first user is created, all further registrations require an admin
session token.  See [ADMIN.md](ADMIN.md) for full user management documentation.

> **Skip this step** if using `user_auth: token` or `user_auth: none` - those
> modes use the shared `auth_token` or no authentication instead of per-user accounts.

### 5 - Verify it is running

```bash
curl http://localhost:8765/health
```

```json
{
  "status": "ok",
  "uptime_s": 4.1,
  "event_count": 0,
  "fix_count": 0,
  "last_fix_age_s": null,
  "pending_groups": 0
}
```

### 6 - Open the live map

```
http://localhost:8765/map
```

The map updates live via SSE - each new fix is rendered without a page reload.
Nodes send a periodic heartbeat (`POST /api/v1/heartbeat`) every 60 seconds,
so they appear on the map with a green (online) or red (offline) indicator
even before any carrier events arrive.

The control panel in the top-right corner provides the following controls.

#### Status bar

| Field | Description |
|-------|-------------|
| **LIVE / NEW FIX / OFFLINE** | SSE connection state. Green = connected; amber flash = new fix received; red = disconnected (auto-reconnects after 5 s). |
| **Time** | Browser's current local time (updates every second). |
| **Last fix** | Age of the most recent computed fix. |
| **Fixes** | Number of fix markers currently visible on the map. |
| **Visible** | Shows *All* normally; shows *since HH:MM:SS* when the Hide Fixes filter is active. |

#### Age-preset buttons - `1m  5m  15m  1h  6h  24h  ALL`

Rolling time window: show fixes from the last N minutes or hours.  The active
preset is highlighted blue.  `ALL` shows every fix in the database.  Clicking
any preset cancels an active fixed window and resumes live updates.  Any active
Hide Fixes filter continues to apply.

#### Fixed time window - `From` / `To` / `Set Window` / `Clear`

Pin the display to an absolute time range - useful for reviewing a specific
incident without new fixes entering the view.  Both inputs are pre-filled with
the last hour on page load.

- **Set Window** - activates the fixed window; age-preset buttons go inactive.
  The button highlights blue while a window is active.
- **Clear** - cancels the fixed window and returns to rolling age-preset mode.

Fixed-window mode and rolling age-preset mode are mutually exclusive.

#### Action buttons

| Button | Description |
|--------|-------------|
| **Hide Heat Map / Show Heat Map** | Toggle the heatmap layer visibility. |
| **Hide Fixes** | Hides all fixes currently visible.  Fixes computed *after* clicking continue to appear.  The hide cutoff is stored per browser session in `localStorage` and is cleared when the tab is closed. |
| **Unhide All** | Cancels the hide filter and restores all fixes within the current age window. |
| **Reset Heat Map** | Clears the cumulative heatmap from the server database.  Requires a second click within 3 s as confirmation. |

#### Map markers and popups

| Marker | Meaning |
|--------|---------|
| Coloured circle (red -> grey) | Fix location (3+ nodes). Red = newest, grey = oldest relative to the current age window. Click to open the popup. |
| Green circle | Receiving node - online (heartbeat received within 120 s). Click for node ID and last-seen time. |
| Red circle | Receiving node - offline (no recent heartbeat). |
| Grey antenna | FM sync transmitter used for timing reference. |
| Dashed amber line | Line of Position (LOP) - 2-node solution (see below). |

**Fix popup** (click a fix circle) shows:

- Signal event timestamp in the browser's local timezone and UTC
- Latitude / Longitude
- TDOA residual (RMS timing error after the solver converged)
- Contributing node IDs and count
- Channel frequency and event type (`onset` / `offset`)

**Hyperbola arcs** (red curves): drawn for the most recent visible fix only.
Each arc is the locus of transmitter positions consistent with the TDOA measured
by one node pair.  Where the arcs intersect is the computed fix.

**Lines of Position** (dashed amber arcs): when only 2 nodes observe an event,
the system cannot compute a unique fix - instead it displays the hyperbola arc
representing all positions consistent with the measured TDOA.  The transmitter
is somewhere along this line.  LOPs fade with age and are stored as fixes with
`node_count=2`.

#### Heatmap layer

Accumulates fix locations over the lifetime of the database.  Useful for
identifying frequently-used transmitter locations.  Updated live on every new
fix; cleared with **Reset Heat Map**.  Also toggleable via the Leaflet layer
control in the bottom-right corner of the map.

#### Signing in (userdb mode)

When the server runs with `user_auth: userdb`, the map page shows a login
overlay.  Enter your username and password to sign in.  Your session is stored
in the browser tab and cleared when you close it.

If your administrator has configured Google OAuth, a **Sign in with Google**
button is also available - click it and follow the Google consent flow to log in
without a password.

**Enabling two-factor authentication (2FA):**

1. Sign in and open the **Users** tab in the control panel.
2. Click **Setup 2FA**.  A secret key is displayed.
3. Add the key to your authenticator app (Google Authenticator, Authy, etc.).
4. Enter the 6-digit code from the app and click **Enable**.

From then on, logging in requires both your password and a 6-digit code from
your authenticator app.  To disable 2FA, use the **Disable 2FA** option in the
Users tab (you will need your password and a current code).

If you lose access to your authenticator device, ask an admin to disable 2FA
on your account from the Users tab.

For server administrators: see [ADMIN.md](ADMIN.md) for full details on
authentication configuration, user management, Google OAuth setup, and API
endpoints.

---

## Mock Event Generator (demo without hardware)

`scripts/mock_event_generator.py` synthesises realistic `CarrierEvent` POSTs
to a running server, simulating radio conversations observed by multiple nodes
with configurable measurement noise.  Use it to verify the full server pipeline
without any SDR hardware, and to explore the relationship between timing error
and position accuracy.

**Prerequisites:** the aggregation server must be running (see above), and
`delivery_buffer_s` in `server.yaml` should match `--delivery-buffer-s` below.

### Run with the Seattle scenario

```bash
# In one terminal: start the server
env/bin/beagle-server --config config/server.yaml

# In another terminal: send synthetic events
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10
```

`--delivery-buffer-s` tells the script how long to wait after posting events
before polling for the fix result.  It must match `pairing.delivery_buffer_s`
in `server.yaml` (default 10 s).

### Explore timing accuracy vs. hardware mode

The `--pilot-sigma-us` flag sets the 1-sigma FM pilot timing noise, which
directly determines TDOA and position accuracy:

```bash
# Uncalibrated RTL-SDR crystal (100 ppm * 200 ms window) - expect ~6 km error
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10 \
    --pilot-sigma-us 20.0

# RTL-SDR TCXO + FM pilot calibration - expect ~500-1500 m error (default)
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10 \
    --pilot-sigma-us 2.0

# two_sdr mode with GPS 1PPS injection - expect ~100-400 m error
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10 \
    --pilot-sigma-us 0.5
```

The `--ntp-sigma-ms` flag controls `onset_time_ns` accuracy.  This affects
only the T_sync grouping window (events from different nodes must land in the
same 200 ms bucket); it has no effect on TDOA or position accuracy.

### Rapid key-up demo

To verify that two transmissions in quick succession are placed in separate
fix groups, edit `mock_scenario_seattle.yaml` and set:

```yaml
rapid_keyup_interval_ms: 400   # 400 ms between transmissions
transmissions_per_target: 4
```

The server should produce one fix per transmission (two per target, four
total), not merge them.

---

## Node Management

`scripts/manage_nodes.py` manages node records directly in the server's SQLite
database.  It creates the required tables on first use (idempotent - safe to run
against an existing database).

Most node management tasks can also be performed from the **web UI** - see the
Nodes tab in the map control panel.  The web UI supports registering nodes,
regenerating secrets, editing labels, editing config JSON, enabling/disabling
nodes, and deleting nodes.  The CLI remains useful for scripted provisioning and
when the server is not running.

### Typical workflow

**1. Create a node config file** on the server, following the same format as
[Option A (local config)](#option-a-local-config) - set the node_id, location,
SDR mode, frequencies, carrier thresholds, and reporter URL:

```bash
cp config/node.example.yaml configs/seattle-north-01.yaml
$EDITOR configs/seattle-north-01.yaml
```

**2. Register the node** in the server database.  This prints a one-time secret
-- save it for the bootstrap file:

```bash
python3 scripts/manage_nodes.py --db data/tdoa_registry.db add seattle-north-01 \
    --label "Seattle North Roof"
```

The output includes the bootstrap config block with the `node_secret` value.
Copy this secret immediately - it is shown only once and cannot be retrieved
later.

**3. Push the config** to the server so the node can fetch it:

```bash
python3 scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \
    --config-file configs/seattle-north-01.yaml
```

**4. On the node**, create `/etc/beagle/bootstrap.yaml` with the `server_url`,
`node_id`, and the secret from step 2.  Then start the node:

```bash
python -m beagle_node --bootstrap /etc/beagle/bootstrap.yaml
```

The node fetches its full config from the server on startup.  Future config
changes pushed via `set-config` are picked up automatically (see below).

### Pushing config updates

Nodes in bootstrap mode long-poll the server for config changes.  To update a
running node's config:

```bash
# Edit the config file, then push it:
python3 scripts/manage_nodes.py --db data/tdoa_registry.db set-config seattle-north-01 \
    --config-file configs/seattle-north-01.yaml
```

The node picks up the new config within seconds (long-poll returns immediately
when the version increments).  The behaviour depends on what changed:

| Changed fields | Behaviour |
|----------------|-----------|
| Carrier thresholds, target channels, clock calibration, sync signal thresholds | **Hot-reloaded** - applied immediately, no downtime |
| SDR hardware params (gain, sample rate, mode, antennas), sync frequency | **Automatic restart** - node exits cleanly (code 75) and systemd restarts it with the new config (~2-3 s downtime) |

No manual intervention is needed in either case.  The systemd unit
(`etc/beagle-node.service`) has `Restart=on-failure` which covers exit code 75.

### Available commands

| Command | Description |
|---------|-------------|
| `list` | List all registered nodes with their status |
| `add <node_id>` | Register a new node and print its secret |
| `show <node_id>` | Show full details for a node |
| `set-config <node_id> --config-file <file>` | Assign a config (YAML or JSON) |
| `enable <node_id>` | Allow the node to submit events |
| `disable <node_id>` | Block the node from submitting events |
| `regen-secret <node_id>` | Generate a new secret (invalidates the old one) |
| `remove <node_id>` | Delete the node record |
| `group-list` | List all frequency groups |
| `group-add <group_id>` | Create a frequency group (requires `--label`, `--sync-freq-hz`, `--sync-station-id`, `--sync-station-lat`, `--sync-station-lon`, and `--channels-file` or `--channels-json`) |
| `group-show <group_id>` | Show group details and member nodes |
| `group-remove <group_id>` | Delete a group (members become ungrouped) |
| `group-set-node <node_id> --group <gid>` | Assign a node to a group (omit `--group` to unassign) |

Use `--server-config config/server.yaml` instead of `--db` to read the DB path
from the server config file.

### Frequency groups

Frequency groups let you manage shared frequency plans (sync signal + target
channels) for multiple nodes.  All nodes in a group share the same sync station
and target channel list.  A node can belong to at most one group.

```bash
# Create a group with a JSON channels file
python3 scripts/manage_nodes.py --db data/tdoa_registry.db group-add seattle-fm \
    --label "Seattle FM" \
    --sync-freq-hz 99900000 \
    --sync-station-id KISW_99.9 \
    --sync-station-lat 47.6253 \
    --sync-station-lon -122.3563 \
    --channels-json '[{"frequency_hz": 460000000, "label": "Target 460"}]'

# Assign a node to the group
python3 scripts/manage_nodes.py --db data/tdoa_registry.db group-set-node seattle-north-01 \
    --group seattle-fm
```

The group's frequency plan takes highest priority in the config merge order:
**server defaults < node config_json < frequency group plan**.  When a group's
frequency fields are updated (via API or CLI), all member nodes' config versions
are bumped, triggering them to pick up the new plan via long-poll.

The web UI (Groups tab in the map control panel) provides a full management
interface for groups: creating, editing, and deleting groups, managing target
channels, and assigning or unassigning member nodes - all without CLI access.

### Server config for nodedb mode

Set `node_auth: nodedb` in `server.yaml` to activate per-node authentication:

```yaml
server:
  node_auth: nodedb           # per-node secrets for event POSTs
  user_auth: token            # or userdb for per-user UI login
  auth_token: "admin-token"   # used by user_auth: token and admin endpoints
```

In `nodedb` mode:
- Nodes authenticate with their individual secret on every event POST.
- Disabled nodes' events are rejected at ingest time (HTTP 403).
- The `user_auth` setting controls UI/admin access independently.

---

## Scripts
All scripts accept `--help` for full option descriptions.

### Signal verification

#### `scripts/verify_sync.py`
Live FM pilot detection display.  Run this first to confirm the sync chain
is working before attempting a full freq_hop run.

```bash
python3 scripts/verify_sync.py --config config/node.yaml --duration 10
```

Without a config file (bare RTL-SDR, SoapySDR path):
```bash
python3 scripts/verify_sync.py --device "driver=rtlsdr" --freq 99.9e6 --gain 0 --duration 10
```

Pass criteria: steady sync event rate, corr_peak >= 0.5.

---

#### `scripts/verify_freq_hop.py`
End-to-end freq_hop pipeline test using pyrtlsdr.  Prints each
`TDOAMeasurement` as it arrives.

```bash
python3 scripts/verify_freq_hop.py \
    --sync-freq 99.9e6 --target-freq 462.5625e6 \
    --gain 30 --duration 60
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--sync-freq` | required | FM sync station Hz |
| `--target-freq` | required | LMR target Hz |
| `--gain` | 30 | Gain dB |
| `--block` | 65536 | Samples per freq block |
| `--settling` | 49152 | Settling samples to discard |
| `--onset-db` | -15 | Carrier onset threshold dBFS |
| `--offset-db` | -25 | Carrier offset threshold dBFS |
| `--min-corr` | 0.1 | Minimum pilot corr_peak |
| `--duration` | 60 | Run time seconds |
| `--device-serial` | *(first found)* | RTL-SDR USB serial number |

---

#### `scripts/verify_rspduo.py`
End-to-end RSPduo pipeline test.  Requires the SDRplay API and SoapySDRPlay3
plugin.  Opens the RSPduo as a dual-tuner receiver and prints each
`TDOAMeasurement` as it arrives.  Both channels run simultaneously with no
coverage gaps or settling time.

```bash
python3 scripts/verify_rspduo.py \
    --sync-freq 99.9e6 --target-freq 462.5625e6 \
    --sync-gain 30 --target-gain 40 --duration 60
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--sync-freq` | required | FM sync station Hz |
| `--target-freq` | required | LMR target Hz |
| `--sync-gain` | auto | Sync channel gain dB (or `auto` for AGC) |
| `--target-gain` | auto | Target channel gain dB (or `auto` for AGC) |
| `--device-args` | `driver=sdrplay` | SoapySDR device string |
| `--rate` | 2.0e6 | Sample rate (max 2 MHz in dual-tuner mode) |
| `--buffer` | 65536 | Samples per read per channel |
| `--onset-db` | -15 | Carrier onset threshold dBFS |
| `--offset-db` | -25 | Carrier offset threshold dBFS |
| `--min-corr` | 0.1 | Minimum pilot corr_peak |
| `--duration` | 60 | Run time seconds |

---

#### `scripts/verify_clock.py`
Measures `time.time_ns()` scheduling jitter and reports expected
`onset_time_ns` uncertainty.

```bash
python3 scripts/verify_clock.py --samples 50000 --show-chrony
```

| Jitter (P99-P50) | Verdict |
|-----------------|---------|
| < 10 usec | Excellent - GPS-disciplined kernel |
| < 100 usec | Good - NTP class |
| < 1 ms | Fair - event association still works |
| > 1 ms | Poor - check system load |

---

### Hardware calibration

#### `scripts/measure_settling.py`
Empirically measures RTL-SDR tuner settling time after a frequency hop and
outputs a recommended `settling_samples` value for `node.yaml`.

```bash
python3 scripts/measure_settling.py \
    --freq1 99.9e6 --freq2 462.5625e6 --gain 30
```

`--freq1` is the frequency to settle **on** (destination); `--freq2` is
where the tuner was before the hop.  The script tunes to `freq2`, then
immediately hops to `freq1`, captures a buffer, and plots power vs. offset.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--freq1` | required | Destination frequency Hz |
| `--freq2` | required | Source frequency Hz |
| `--gain` | 30 | Gain dB |
| `--capture` | 131072 | Samples to capture after hop |
| `--window` | 2048 | Analysis window samples (~1 ms) |
| `--tolerance-db` | 2.0 | Power must be within this of steady-state |
| `--margin` | 1.5 | Safety multiplier for recommendation |

---

### Offline analysis

#### `scripts/replay_iq.py`
Feed stored `.npy` IQ files through the pipeline offline.  Useful for
tuning carrier detect thresholds without live hardware.

```bash
# FM pilot only (count sync events):
python3 scripts/replay_iq.py --sync iq_fm_kisw.npy

# LMR carrier detect (no TDOA measurement without sync):
python3 scripts/replay_iq.py --target iq_lmr_155.npy --onset-db -20

# Full pipeline (sync + target aligned, single_sdr capture):
python3 scripts/replay_iq.py --sync iq_fm.npy --target iq_lmr.npy

# freq_hop capture (sync and target as separate files with block offsets):
python3 scripts/replay_iq.py \
    --sync iq_sync.npy --target iq_target.npy \
    --freq-hop-block 65536
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--sync` | - | Sync channel `.npy` |
| `--target` | - | Target channel `.npy` |
| `--rate` | 2.048e6 | SDR sample rate of the files |
| `--onset-db` | -30 | Carrier onset threshold dBFS |
| `--offset-db` | -40 | Carrier offset threshold dBFS |
| `--min-corr` | 0.1 | Minimum pilot corr_peak |
| `--freq-hop-block` | 0 | Enable freq_hop mode with this block size |

---

#### `scripts/capture_iq_fixture.py`
Capture live IQ from a real SDR to a `.npy` file for use with `replay_iq.py`
or as test fixtures.

```bash
# 5 s of FM broadcast at 99.9 MHz:
python3 scripts/capture_iq_fixture.py \
    --freq 99.9e6 --rate 2.048e6 --gain 0 --duration 5 \
    --output tests/fixtures/iq_fm_kisw_99.9.npy

# 3 s of LMR channel:
python3 scripts/capture_iq_fixture.py \
    --freq 462.5625e6 --rate 2.048e6 --gain 30 --duration 3 \
    --output tests/fixtures/iq_lmr_462.npy
```

---

#### `scripts/colocated_pair_test.py`
Measures timing jitter and position accuracy by comparing two nodes at the same
physical location.  Supports Monte Carlo simulation (no hardware) and live data
analysis from the aggregation server DB.

```bash
# Simulation - model RSPduo vs RTL-SDR 2freq timing noise
python3 scripts/colocated_pair_test.py --simulate \
    --node-a-sigma-us 1.0 --node-b-sigma-us 2.0 --n-trials 1000

# Analysis from server DB
python3 scripts/colocated_pair_test.py \
    --db data/tdoa_data.db \
    --node-a rspduo-node-01 --node-b rtlsdr-node-01 \
    --channel-hz 462562500
```

Works with any combination of node hardware types (RSPduo, RTL-SDR 2freq, etc.)
as long as both nodes report to the same aggregation server.

---

#### `scripts/analyze_xcorr_tdoa.py`

Analyses cross-correlation TDOA measurements from the server database and
reports the usec-level inter-node timing difference for each matched event pair.
Groups events by transmission (channel, event type, sync transmitter, onset
proximity), runs the power-envelope xcorr on each node pair, and summarises
results by event type with per-pair disposition (OK / LOW\_SNR / BASELINE\_REJ).

Most useful with co-located nodes to measure the combined xcorr noise floor,
or to sanity-check new node deployments before attempting a position fix.

```bash
# Analyse live database with defaults (min SNR 1.5, max baseline 50 km)
python3 scripts/analyze_xcorr_tdoa.py

# Specify a different database or relax filters
python3 scripts/analyze_xcorr_tdoa.py \
    --db data/tdoa_data.db \
    --min-snr 1.0 \
    --max-baseline-km 100
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `data/tdoa_data.db` | Path to SQLite database |
| `--min-snr` | 1.5 | Minimum xcorr peak-to-sidelobe ratio to accept |
| `--max-baseline-km` | 50.0 | Reject xcorr lags larger than this km-equivalent; 0 = disabled |
| `--window-ms` | 200 | Grouping window for pairing events from different nodes (ms) |

Example output (two co-located nodes, true TDOA = 0):

```
onset   rspduo-test<->node-discovery    -1.587   6.71  OK
onset   rspduo-test<->node-discovery    -4.827  28.59  OK
offset  rspduo-test<->node-discovery    -1.275   1.71  OK
...
  onset  (n=14): mean=+17.091 usec  median=+1.369 usec  stdev=41.600 usec
  offset (n=11): mean=-0.017 usec   median=+0.116 usec  stdev=0.657 usec
```

Offset events typically have lower SNR (1.5-2.5) but tighter timing (< 1 usec
stdev); onset events have higher SNR (6-28) with occasional false peaks from
ambiguous carrier patterns.

---

### Node provisioning

#### `scripts/manage_nodes.py`

Manage node records in the server database without the REST API or web UI.
Most of these operations are also available from the Nodes tab in the web UI.
See [Node Management](#node-management) for the full workflow.

```bash
python3 scripts/manage_nodes.py --db data/tdoa_registry.db list
python3 scripts/manage_nodes.py --db data/tdoa_registry.db add <node_id> \
    --label "Human-readable name"
python3 scripts/manage_nodes.py --db data/tdoa_registry.db set-config <node_id> \
    --config-file configs/<node_id>.yaml
```

---

## Calibration

Each step builds on the previous.  Steps 2-5 require no test transmitter;
only Step 6 requires a transmitter at a known location.

Steps marked **freq_hop only**, **rspduo only**, or **two_sdr only** apply to
that mode exclusively.  All other steps apply to every SDR mode.

### Prerequisites

- Node hardware installed and powered (Pi 5 + SDR + antenna)
- Python environment activated: `source env/bin/activate`
- Config file in place: `config/node.yaml` (or `/etc/beagle/node.yaml` for production)
- Chrony running and synchronised: `chronyc tracking` shows an NTP or GPS reference
- **`freq_hop` mode only:** pyrtlsdr installed (`pip install pyrtlsdr`) and
  librtlsdr system library present (`sudo apt install librtlsdr0`)
- **`rspduo` mode only:** SDRplay API and SoapySDRPlay3
  (`rspduo-dual-independent-tuners` branch) installed - see
  [docs/setup-rspduo-debian.md](docs/setup-rspduo-debian.md).
  Confirm the device is visible: `SoapySDRUtil --find`

---

### Step 1 - Measure Tuner Settling Time *(freq_hop mode only)*

**Goal:** Empirically determine how many samples the R820T tuner needs to
settle after a frequency hop, so `settling_samples` in `node.yaml` is correct
for your specific dongle.

The settling time varies across RTL-SDR dongles (typically 10-60 ms at
2.048 MSPS).  Setting `settling_samples` too low means pilot and carrier data
include settling artefacts; too high wastes usable signal.

```bash
python3 scripts/measure_settling.py \
    --freq1 99.9e6 --freq2 462.5625e6 --gain 30
```

`--freq1` is the destination frequency (what the tuner settles **on**);
`--freq2` is where the tuner was before the hop.  Swap them and re-run to
measure settling in both directions.

The script reports a recommended `settling_samples` value with a safety margin.

**Pass criteria:**
- Power stabilises within **< 65536 samples** (< 32 ms at 2.048 MSPS)
- If settling exceeds 65536 samples, increase `samples_per_block` accordingly

**Update `node.yaml`:**

```yaml
freq_hop:
  settling_samples: <value from script>   # e.g. 49152
  samples_per_block: 65536                # must be > settling_samples
```

**Record result:**
```
Settling to sync freq  (99.9 -> 462.5625 MHz): _____ samples (~___ ms)
Settling to target freq (462.5625 -> 99.9 MHz): _____ samples (~___ ms)
settling_samples set to: _____
```

---

### Step 2 - Verify FM Pilot Detection and Crystal Calibration

**Goal:** Confirm the sync chain produces clean SyncEvents and the
CrystalCalibrator converges to a stable correction factor.

```bash
python3 scripts/verify_sync.py --config config/node.yaml --duration 60
```

The same run covers both checks: pilot quality is visible from the first
few rows; crystal convergence requires ~60 s.

**Pilot detection - pass criteria (read from first 10 s of output):**
- Event rate: **>= 95 events / 10 s** (ideally ~120, one per ~8 ms window)
- Mean `corr_peak` >= **0.5** for a well-received station
- Values stable row-to-row (no sudden drops)

**Crystal calibration - pass criteria (read from full 60 s run):**
- `Crystal` converges within ~10 s and stays within **+/-100 ppm**
- Drift between t=10 s and t=60 s: **< 10 ppm** (shown in summary line)
- RSPduo values near 0 ppm (< +/-10 ppm) are normal and excellent

**Tuning:**
- If `corr_peak` is low, try a stronger FM station or increase gain.
  KISW 99.9 MHz works well in the Seattle metro area from a rooftop.
- If gain is too high the ADC will clip; reduce it until the IQ magnitude
  is <= -6 dBFS RMS.
- `min_corr_peak` in `node.yaml` -> `sync_signal.min_corr_peak` filters out
  weak sync events.  Set it to 0.3 for normal operation.

**Interpreting the Crystal column:**
- `Crystal = 0.0 ppm`: crystal is exact (or calibrator not yet converged)
- `Crystal = +50.0 ppm`: crystal runs 50 ppm fast -> corrected automatically
- Values outside +/-200 ppm suggest a low-quality crystal; use a TCXO
  RTL-SDR for best accuracy; RSPduo should always be well within +/-10 ppm

*RTL-SDR (standard crystal, ~50 ppm drift):*
```
  Time    Events   Rate/s   CorPeak     Crystal     Power
   5.0       500    100.0    0.7234   +43.2 ppm   -22.3 dBFS
  10.0      1000    100.0    0.7198   +43.1 ppm   -21.8 dBFS
  60.0      6000    100.0    0.7211   +43.0 ppm   -22.1 dBFS

Crystal drift: +43.1 ppm at t~10 s -> +43.0 ppm at end  (drift=0.1 ppm  OK)
```

*RSPduo (24 MHz TCXO, < 10 ppm):*
```
  Time    Events   Rate/s   CorPeak     Crystal     Power
   5.0       500    100.0    0.6978    -6.6 ppm   -28.5 dBFS
  10.0      1000    100.0    0.6994    -2.1 ppm   -28.3 dBFS
  60.0      6000    100.0    0.6983    -4.8 ppm   -28.4 dBFS

Crystal drift: -2.1 ppm at t~10 s -> -4.8 ppm at end  (drift=2.7 ppm  OK)
```

**Record result:**
```
Station: KISW 99.9 MHz
Gain: ___ dB
Event rate: ___ / s  Mean corr_peak: ___
Crystal (steady-state): ___ ppm   Drift over 50 s: ___ ppm
```

---

### Step 3 - Calibrate Carrier Detection Thresholds

**Goal:** Determine the correct `carrier_onset_db` and `carrier_offset_db`
values for your target frequency and local RF environment.

Tune to the target channel at the gain you plan to use for normal operation
and key your transmitter a few times during the run.

**Important:** always use `env/bin/python` (not `python3`) so the virtualenv's
SoapySDR bindings are used.  Running bare `python3` may pick up system audio
or UHD drivers before the SDR device and fail with a PulseAudio error.

*RTL-SDR (freq_hop node):*

```bash
env/bin/python scripts/check_target.py \
    --freq 462.5625e6 --gain 30 --duration 30
```

*RSPduo node - always specify device, channel, freq-offset, and rate:*

```bash
env/bin/python scripts/check_target.py \
    --freq 462.5625e6 --gain 30 --duration 30 \
    --device "driver=sdrplay" --channel 1 --freq-offset 0 --rate 2000000
```

- `--device "driver=sdrplay"` - selects the RSPduo instead of the first available device
- `--channel 1` - Tuner 2 / Antenna C (the LMR receive port)
- `--freq-offset 0` - RSPduo applies a hardware DC notch; no baseband offset needed
- `--rate 2000000` - RSPduo dual-tuner mode requires an exact supported rate (2.0 MHz, not the RTL-SDR default 2.048 MHz)

The script reports received power in dBFS once per 500 ms and marks windows
where a signal is detected.  At the end it recommends threshold values:

```
--- Results ---
Noise floor:     -30.3 dBFS  (10th percentile)
Peak signal:      -5.7 dBFS  (maximum observed)
Observed SNR:    24.6 dB

Recommended thresholds:
  onset_db  = -18 dBFS
  offset_db = -28 dBFS
```

**Pass criteria:**
- Noise floor settles at **-30 to -60 dBFS** with no signal - if it is above
  -6 dBFS the ADC is saturated; reduce gain (the script will warn you)
- Observed SNR >= **10 dB** with the transmitter keyed
- At least one window marked `*** SIGNAL ***` per transmission

**Update `node.yaml`:**

```yaml
carrier:            # top-level section, applies to all SDR modes
  onset_db:  -18   # from check_target.py output
  offset_db: -28
```

**Record result:**
```
Frequency: _____ MHz   Gain: ___ dB
Noise floor: _____ dBFS   Peak signal: _____ dBFS   SNR: _____ dB
carrier.onset_db set to: _____   carrier.offset_db set to: _____
```

---

### Step 4 - End-to-End Pipeline Test

**Goal:** Confirm that `TDOAMeasurement` objects are produced by the full
pipeline before attempting the quantitative path-delay check.

This step requires occasional LMR carrier activity on the target channel
(or a programmable radio you can key briefly).

#### freq_hop mode

```bash
python3 scripts/verify_freq_hop.py \
    --sync-freq 99.9e6 --target-freq 462.5625e6 \
    --gain 30 --duration 60
```

For asymmetric blocks (more target time), add `--target-block 262144`:

```bash
python3 scripts/verify_freq_hop.py \
    --sync-freq 99.9e6 --target-freq 462.5625e6 \
    --gain 30 --block 131072 --target-block 262144 --duration 60
```

The script prints each `TDOAMeasurement` as it arrives with columns:
`sync_delta_ns`, `corr_peak`, `onset_power_db`.

**Pass criteria (all modes):**
- SyncEvents appear continuously at **>= 10 events/s** (ideally ~100/s)
- At least one `TDOAMeasurement` printed while the target channel has a carrier
- `corr_peak` values match Step 2 results (consistent pilot quality)
- `onset_power_db` is above `carrier_onset_db` set in Step 3

**Common issues (freq_hop):**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: pyrtlsdr is required` | pyrtlsdr not installed | `pip install pyrtlsdr` |
| No measurements after carrier is present | `--onset-db` threshold wrong | Run Step 3 (`check_target.py`) to calibrate thresholds |
| Sync rate < 5/s | Sync block too small after settling | Use `--block 131072` |
| `corr_peak` much lower than Step 2 | Too many settling artefacts in sync blocks | Increase `--settling` |

#### rspduo mode

Requires the SDRplay API and SoapySDRPlay3 plugin (see Prerequisites).
Both channels run simultaneously with no settling time or coverage gaps.

Use `verify_sync.py` with the node config file (the same receiver and pipeline
the node uses in production):

```bash
env/bin/python scripts/verify_sync.py \
    --config config/node.yaml --duration 60
```

This confirms FM pilot lock on the sync channel (Tuner 1).  Watch for:
- `Rate/s` > 50/s - pilot lock confirmed (exact rate = 1000 / sync_period_ms;
  with the default 7 ms period at 2 MHz / 8x decimation -> ~143/s)
- `Crystal` within +/-10 ppm for RSPduo (TCXO)
- `Crystal drift` < 10 ppm in summary line

For the target channel, use `check_target.py` (Step 3 above).  A combined
end-to-end measurement script that drives the full RSPduo pipeline and prints
`TDOAMeasurement` objects is not yet implemented; use `main.py` with the node
config and watch the logs for `TDOAMeasurement` lines.

**Common issues (rspduo):**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RtAudio init error 'RtApiPulse...'` | SoapySDR picked an audio device | Use `env/bin/python`, not `python3`; add `--device "driver=sdrplay"` |
| "SoapySDR Python bindings not available" | Plugin not installed | See [docs/setup-rspduo-debian.md](docs/setup-rspduo-debian.md) |
| Device open fails in DT mode | SoapySDRPlay3 missing `rspduo-dual-independent-tuners` branch | Rebuild from source (see setup doc) |
| No device found | RSPduo not connected or driver conflict | `SoapySDRUtil --find`; check no other software is using the device |
| Target channel at noise floor only (no signal bump) | Pre-init frequency discarded | Ensure `--channel 1 --device "driver=sdrplay"` is specified |
| Overflows reported | USB bandwidth | Use a USB 3 port; reduce buffer_size in node.yaml |
| Sync rate low | Weak FM pilot | Check sync antenna (Antenna A); try a stronger FM station |

#### two_sdr and single_sdr modes

*[End-to-end test script for these modes TBD - use `main.py` with the full
config and watch the logs for `TDOAMeasurement` lines.]*

**Record result:**
```
Mode: freq_hop / rspduo / two_sdr / single_sdr
First measurement sync_delta_ns: _____ ns
corr_peak (typical): _____
onset_power_db: _____ dBFS
```

---

### Step 5 - Co-Located Pair Test

**Goal:** Measure the actual TDOA timing error between two nodes.  Since both
nodes are at the same location, the true TDOA is zero; any measured deviation
is pure noise in the timing pipeline.

**Why this matters:** The co-located pair test is the only single-site check
that directly measures what the TDOA system actually delivers.
`TDOA_AB = sync_delta_A - sync_delta_B` for a co-located pair should be 0 ns;
the standard deviation of that distribution tells you the timing noise floor,
which sets the position accuracy limit for the whole network.  Onset and offset
are measured separately because the rising and falling edges of a carrier have
different detection noise characteristics.

**Physical setup:**

- Two fully calibrated nodes (each through Steps 1-5), placed at the **same
  physical location** and connected to the **same aggregation server**.
- A transmitter at a different location - ideally > 1 km away so the signal
  does not arrive from the side-lobe of one node's antenna.
- Let both nodes run for at least 5 minutes before capturing data to allow
  crystal stabilisation on each.

**Simulation mode (no hardware - pre-deployment planning):**

```bash
python3 scripts/colocated_pair_test.py --simulate \
    --node-a-sigma-us 1.5 --node-b-sigma-us 1.5 --n-trials 2000
```

This shows TDOA noise and position error for given per-node timing sigmas.
Use it to decide whether the expected jitter is acceptable before deploying.

**Real-data mode (after capturing data with both nodes running):**

```bash
python3 scripts/colocated_pair_test.py \
    --db data/tdoa_data.db \
    --node-a node-a-id --node-b node-b-id \
    --channel-hz 462562500
```

Onset and offset are always analysed separately.  Example output:

```
--- ONSET ---  node-a=42  node-b=41
Matched pairs: 39
TDOA_AB (true value = 0 for co-located nodes):
  TDOA_AB:
    N=39  mean=+124 ns  std=1850 ns
    P50=87  P95=3210  P99=4100  ns

--- OFFSET ---  node-a=40  node-b=38
Matched pairs: 36
TDOA_AB (true value = 0 for co-located nodes):
  TDOA_AB:
    N=36  mean=-210 ns  std=2300 ns
    P50=-190  P95=4100  P99=5200  ns
```

**Pass criteria:**

| Metric | Target |
|--------|--------|
| TDOA std dev (onset) | <= 3 000 ns (3 usec) |
| TDOA std dev (offset) | <= 5 000 ns (5 usec) |
| TDOA mean bias | < +/-500 ns |
| P95 |TDOA\| | < 6 000 ns (6 usec) |

Offset jitter is typically higher than onset because the falling edge of a
carrier is less sharp than the rising edge.

If std dev is too high: verify settling calibration (Step 1), increase
`min_hold` to reduce false-early detections, or check for antenna coupling
between the two co-located nodes.

**Record result:**
```
ONSET  - N: _____  std: _____ ns  mean: _____ ns  P95: _____ ns
OFFSET - N: _____  std: _____ ns  mean: _____ ns  P95: _____ ns
Pass? [ ] Yes  [ ] No - action: ________________________________
```

---

### Step 6 - Known-Location Verification

**Goal:** Confirm end-to-end TDOA accuracy with a test transmission from a
GPS-surveyed location.

**Requirements:**
- Two or more calibrated nodes reporting to the aggregation server
- A handheld radio with a GPS receiver (to mark the true transmission point)
- A clear LMR channel in the configured frequency band

**Procedure:**

1. Survey the transmission point with your GPS receiver.  Note lat/lon to +/-1 m.
2. Key the handheld radio for 5 seconds.  Note the time.
3. Retrieve the server's computed TDOA fix for that event.
4. Compute fix error:
   ```
   error_m = haversine_m(fix_lat, fix_lon, true_lat, true_lon)
   ```

**Pass criteria:**

| Mode | Expected accuracy | Target error |
|------|-----------------|--------------|
| `freq_hop` (TCXO RTL-SDR) | 1-5 usec -> 300 m - 1.5 km | < 1 km |
| `freq_hop` (standard RTL-SDR) | 2-10 usec -> 600 m - 3 km | < 2 km |
| `rspduo` | ~1-2 usec -> ~300-600 m | < 1 km |
| `two_sdr` + GPS 1PPS | < 1 usec -> ~300 m | < 500 m |

If the error exceeds the target:
1. Re-run Steps 2-3 to check pilot quality, crystal correction, and thresholds.
2. Verify FCC coordinates for the sync station are correct.
3. Inspect `calibration_offset_ns` in `node.yaml` - set it to compensate for
   any measured systematic bias.

**Record result:**
```
True location: _______, _______
Server fix:    _______, _______
Error: _____ m
```

---

### Step 7 - GPS 1PPS Injection Verification *(two_sdr mode only)*

Skip this step for `freq_hop` and `single_sdr` modes.

**Goal:** Verify that the GPS 1PPS pulse is correctly detected in both SDR
streams and that inter-SDR alignment is < 1 usec.

```bash
python3 scripts/verify_pps.py --duration 10
```

**Pass criteria:**
- PPS detected in both streams every second
- Inter-SDR offset: **< 1 usec** (ideally < 500 ns)
- Jitter (std-dev of offset across 10 s): **< 500 ns**

---

### Calibration Record Template

Copy and fill in after each calibration run:

```
Date: ____________________
Node ID: _________________
Location: ________________
Hardware: SDR model, antenna, Pi model

Step 1 - Tuner Settling (freq_hop only; skip for rspduo/two_sdr/single_sdr)
  Settling to sync freq:   _____ samples (~___ ms)
  Settling to target freq: _____ samples (~___ ms)
  settling_samples in node.yaml: _____

Step 2 - FM Pilot + Crystal Calibration
  Station: _______________  Gain: ___ dB
  Event rate: ___ / s  (target >= 95)   Mean corr_peak: ___  (target >= 0.5)
  Crystal (steady): ___ ppm  (target +/-100 ppm)   Drift over 50 s: ___ ppm  (target < 10)

Step 3 - Carrier Detection Thresholds
  Frequency: _____ MHz   Gain: ___ dB
  Noise floor: _____ dBFS   Peak: _____ dBFS   SNR: _____ dB
  carrier.onset_db: _____   carrier.offset_db: _____

Step 4 - End-to-End Pipeline Test
  Script used: verify_freq_hop.py / verify_rspduo.py / main.py
  First measurement sync_delta_ns: _____ ns
  corr_peak: _____  onset_power_db: _____ dBFS
  Overflows (rspduo): _____

Step 5 - Co-Located Pair Test
  ONSET  std: _____ ns  mean: _____ ns  P95: _____ ns
  OFFSET std: _____ ns  mean: _____ ns  P95: _____ ns
  Pass? [ ] std(onset) <= 3 usec  [ ] std(offset) <= 5 usec

Step 6 - Known-Location Verification
  True: _______, _______  Fix: _______, _______  Error: ___ m
  Pass? [ ] within target

Clock
  chronyc source: ________  RMS offset: ___ ns
  calibration_offset_ns applied: ___
```

---

### Calibration Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| ADC saturated (power > -6 dBFS) | Gain too high | Reduce `--gain` by 20 dB; re-run `check_target.py` (Step 3) |
| Event rate < 50/s | FM pilot SNR too low | Move antenna, reduce gain to avoid clipping, try different station |
| `corr_peak` < 0.3 | Station too weak or wrong frequency | Confirm station, check antenna |
| `Crystal` column > +/-200 ppm | Noisy crystal | Use TCXO RTL-SDR; keep `max_sync_age_ms` <= 50 ms |
| No measurements after carrier present | Onset threshold wrong | Run Step 3 (`check_target.py`) to calibrate thresholds |
| Large path-delay residual | Wrong FCC coordinates or wrong node location | Verify GPS coords at deployment site |
| Fix error > 2 km | Clock error or SDR drift | Run Steps 2-3; adjust `calibration_offset_ns` |
| Health shows `clock_source: unknown` | Chrony not running | `sudo systemctl start chrony` |

---

## Production deployment (systemd)

```bash
# Install the service
sudo cp etc/beagle-node.service /etc/systemd/system/
sudo mkdir -p /etc/beagle

# Local config approach:
sudo cp config/node.yaml /etc/beagle/node.yaml
sudo $EDITOR /etc/beagle/node.yaml

# - OR - remote config (bootstrap) approach:
sudo cp config/bootstrap.example.yaml /etc/beagle/bootstrap.yaml
sudo $EDITOR /etc/beagle/bootstrap.yaml   # fill in server_url, node_id, node_secret

# Create service user
sudo useradd -r -s /usr/sbin/nologin tdoa
sudo usermod -aG plugdev tdoa   # USB SDR access

# Create config cache directory (used by bootstrap mode to survive reboots)
sudo mkdir -p /var/cache/beagle
sudo chown tdoa:tdoa /var/cache/beagle

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now beagle-node
sudo journalctl -u beagle-node -f
```

---

## Tests

```bash
source env/bin/activate
pytest tests/              # all tests (352 passing, 1 skipped without SoapySDR hardware)
pytest tests/unit/         # unit tests only (no hardware)
pytest tests/integration/  # integration tests (no hardware)
```

---

## Project layout

```
src/beagle_node/
|-- sdr/           SDR receiver backends (SoapySDR, freq_hop, mock)
|-- pipeline/      Signal processing (decimator, FM demod, pilot sync, carrier detect, delta)
|-- events/        CarrierEvent model + HTTP reporter
|-- config/        Pydantic config schema, YAML loader, remote config fetcher
|-- timing/        Clock sources + sample-index stamper
+-- utils/         Logging, /health endpoint, chrony parser

src/beagle_server/
|-- api.py         FastAPI routes (POST /api/v1/events, GET /map, GET /map/data, ...)
|-- pairing.py     T_sync-based event grouping and delivery buffer
|-- solver.py      Hyperbolic TDOA fix solver (scipy L-BFGS-B)
|-- map_output.py  Folium map shell, GeoJSON fix layer, and embedded control-panel JS
|-- db.py          SQLite schema and async CRUD (aiosqlite)
|-- config.py      Server config schema and YAML loader
+-- main.py        beagle-server entry point

scripts/           Verification, calibration, and demo tools
config/            node.example.yaml, server.example.yaml, bootstrap.example.yaml
etc/               beagle-node.service (systemd)
docs/design/       Architecture and timing model design notes
tests/             Unit and integration tests
```

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](LICENSE).
