# Beagle - Open Source RDF

Software for a distributed Time Difference of Arrival (TDOA) system that
locates land mobile radio (LMR) transmissions to neighbourhood-level accuracy
using low-cost RTL-SDR hardware.

The system has two components:

- **Nodes** - Raspberry Pi (or other Linux system) + SDR receivers deployed at fixed locations.  Each
  node listens for LMR carrier events and measures their timing relative to a
  reference signal extracted from an FM broadcast (RDS bit transitions on the
  57 kHz subcarrier).  All nodes receiving the same FM station see the same
  bit transitions at the same physical instant, providing a shared timing
  reference without GPS hardware at the node.  Nodes POST their
  measurements to the central server.

- **Aggregation Server** - A central server (laptop, Pi, or cloud VM typically running Linux) that
  collects measurements from all nodes, computes hyperbolic position fixes, and
  serves a live map.  The server is where you see results.

You need **one server** and **two or more nodes** (three or more for a unique
position fix; two nodes produce a line of position).

The primary SDR we have been testing with and the one we recommend is
the [SDRPlay RSPduo](https://www.hamradio.com/detail.cfm?pid=H0-016162).
The dual tunner feature is perfect for this application.
We believe others SDRs can be made to work but we have not worked with them yet.

The freq-hop mode with a single channel SDR should in theory function but needs more
work after recent sync algorithm changes.

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
- [TLS and reverse-proxy deployment](#tls-and-reverse-proxy-deployment) - Apache subdomain or subpath
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
extracting timing pulses from a shared FM broadcast.  Every FM stereo station
transmits a **19 kHz pilot tone** locked to a GPS-traceable frequency standard,
and modulated on the **57 kHz RDS subcarrier** (3 x pilot) is a 1187.5 bps BPSK
data stream whose **bit boundaries are phase-locked to the pilot by spec**.
Beagle's `RDSSyncDetector` recovers those bit-boundary timestamps and emits a
`SyncEvent` for each one.  Because the bit clock is a deterministic feature of
the broadcast (not an arbitrary zero-crossing), every node tuned to the same
station identifies the **same** bit transition as the same physical event --
the fundamental ambiguity of pilot zero-crossings (which are all identical and
indistinguishable across nodes) is eliminated.

When a land mobile radio (LMR) carrier is detected, the node records
`sync_delta_ns = target_onset_sample - sync_event_sample`, expressed in
nanoseconds on the local sample clock.  Because both measurements use the same
unbroken ADC clock, the absolute clock offset cancels out entirely -- only the
interval between two events on that clock matters.

The aggregation server collects `sync_delta_ns` reports from all nodes,
corrects for the known FM transmitter-to-node propagation delay (computed from
FCC-documented station coordinates), and computes
`TDOA_AB = sync_delta_A - sync_delta_B`.  A scipy L-BFGS-B solver minimises
the squared residuals across all node pairs to produce a latitude/longitude fix,
which is logged to SQLite and displayed on a live Folium map.

```
   FM broadcast station                           LMR transmitter
   (19 kHz pilot + 57 kHz RDS)                            |
            |                                             |
   SDR (sync channel)                            SDR (target channel)
            |  decimated IQ                              |  decimated IQ
            v                                             v
   RDSSyncDetector                                CarrierDetector
   (FM demod -> -57 kHz shift                            |
    -> LPF -> M&M timing                                  |
    -> Costas -> bit boundary)                            |
            |                                             |
            |  SyncEvent (every ~842 usec,                |  CarrierOnset
            |  one per RDS bit transition)                |
            +---------------> DeltaComputer <-------------+
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

**Why RDS instead of just the pilot tone?**  Earlier versions of Beagle used
the 19 kHz pilot zero-crossings as sync events.  This appeared to work for
co-located test pairs but failed for any real geometry: zero-crossings happen
every 52.6 microseconds and are *physically indistinguishable*, so different
nodes locked to different ones, producing an unresolvable `N x 52.6 usec`
ambiguity in the cross-node `sync_delta` subtraction.  RDS bit transitions
happen every 842 microseconds (1187.5 Hz = pilot/16) and are anchored to a
data signal, so all nodes agree on which transition is which.  The resulting
disambiguation period (842 usec) comfortably exceeds the maximum geometric
TDOA for a 100 km baseline (~333 usec), so disambiguation is unambiguous.

**Further reading:**

- Knapp & Carter (1976) - [The Generalized Correlation Method for Estimation of Time Delay](https://www.semanticscholar.org/paper/The-generalized-correlation-method-for-estimation-Knapp-Carter/29c74aad1986ff2e907e084820e990a0544e743a) - foundational cross-correlation technique
- NRSC-4-B / IEC 62106 - the RBDS / RDS standard (1187.5 bps BPSK, CRC-10, block sync)
- [PySDR: RDS chapter](https://pysdr.org/content/rds.html) - the Mueller-Muller + Costas chain Beagle's RDS decoder is built on
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

**Optional fields** in `bootstrap.yaml`:

| Field | Default | Notes |
|---|---|---|
| `config_cache_path` | `/var/cache/beagle/node_config.json` | Last fetched config is written here so the node can start offline using the cached copy if the server is unreachable |
| `config_poll_interval_s` | `60.0` | Long-poll interval in seconds.  Capped at 120 (server limit).  Larger = fewer connection cycles, lower NAT/firewall churn.  Lower = no benefit; the server only responds early when there's actually a config change |
| `register_on_start` | `true` | Whether to call `POST /api/v1/nodes/register` on startup to update `last_seen_at` |

**Failure handling**: when the server is unreachable or returns an error
(connection refused, HTTP 5xx, HTTP 4xx auth failure), the node retries with
**exponential backoff** -- 1 s, 2 s, 4 s, 8 s, ..., capped at 120 s, with
+/-25% jitter.  The backoff resets to 1 s on the first successful poll
(including a normal `304 Not Modified` response).  This means a brief server
restart causes one or two retry warnings in the node log, then a clean
recovery; a sustained outage spaces out retries gracefully without flooding
the log or hammering the server.

### 2 - Verify hardware and RDS sync detection

Before running the full pipeline, confirm your SDR can demodulate the chosen
FM station and the `RDSSyncDetector` can lock onto its RDS bit stream:

```bash
python3 scripts/verify_rds_sync.py --config config/node.yaml --duration 30
```

Expected after the ~50 ms warm-up:
- **Event rate ~1188/s** (one per RDS bit transition; the exact rate is
  pilot/16 = 1187.5 Hz)
- **Pilot `corr_peak` >= 0.5** (the RDS detector still extracts the pilot
  internally for crystal calibration, even though it's not the sync source)
- **Crystal correction** stable, drifting < 10 ppm over 30 s
- **Bit interval jitter < 5 usec** (sub-microsecond after the M&M timing loop
  has converged)

If the rate is much below 1188/s, the chosen station probably doesn't carry
RDS -- pick a different station.  In the US, almost all NPR affiliates and
most commercial stations carry RDS; verify with the
[FCC FM Query](https://www.fcc.gov/media/radio/fm-query) or the station's
website.

**RSPduo end-to-end pipeline test** (for the `rspduo` SDR mode, exercises both
tuners simultaneously):

```bash
python3 scripts/verify_rspduo.py --sync-freq 94.9e6 --target-freq 443.475e6
```

Expected: sync rate ~1188/s, 0 overflows, measurements produced when an LMR
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

The node exposes a JSON status endpoint on its health port (`8080` by default,
or whatever `health_port` is set to in `node.yaml`):

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "node_id": "seattle-north-01",
  "uptime_s": 42.3,
  "events_submitted": 7,
  "sync_events": 49832,
  "sync_corr_peak": 0.7042,
  "crystal_correction": 1.0000098
}
```

For a live, formatted view that polls the endpoint each second and shows
derived metrics (sync event rate, deltas), use `watch_node_health.py`:

```bash
# Local node:
python3 scripts/watch_node_health.py

# Remote node (any reachable host):
python3 scripts/watch_node_health.py --host dpk-tdoa1.local --port 8080
```

It groups the fields into Configuration / Sync / Carrier-Target / Hardware-Clock
sections, colour-codes status (green = healthy, yellow = marginal, red =
degraded), and computes the sync event rate from successive snapshots.
Press Ctrl-C to exit.  See the `scripts/watch_node_health.py` entry under
[Scripts](#scripts) for full options.

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

## TLS and reverse-proxy deployment

By default `beagle-server` listens on plain HTTP at the host/port from
`config/server.yaml` (typically `0.0.0.0:8765`).  This is fine for a private
LAN demo but **must not be used over the public internet**: admin session
cookies, login passwords, and node bootstrap secrets would all be sent in
the clear.

For internet-facing deployments, terminate TLS at a reverse proxy (Apache,
nginx, Caddy) that already has a cert -- typically the same vhost that's
already serving your main site.  Beagle's app supports this directly:

- `--root-path /<prefix>` makes FastAPI generate absolute URLs
  (OAuth callbacks, Location headers, internal redirects) with the subpath
  prefix the proxy is using.
- `proxy_headers=True` and `forwarded_allow_ips="127.0.0.1"` (already set in
  `beagle_server/main.py`) make uvicorn trust `X-Forwarded-Proto` and
  `X-Forwarded-For` from a localhost proxy, so `request.url.scheme` reports
  `https` even though the proxy connects to uvicorn over plain HTTP on the
  loopback interface.

Two common Apache deployment topologies are documented below.  Both assume
the proxy and Beagle run on the same host, with Beagle bound to
`127.0.0.1:8765` so it cannot be reached directly from the network.

**Required Apache modules** (enable once):
```bash
sudo a2enmod proxy proxy_http headers ssl rewrite
sudo systemctl reload apache2
```

### Option 1 - Subdomain (cleanest)

Put Beagle on its own subdomain like `https://beagle.example.com/`.  Requires
a DNS A record for the subdomain pointing at the same host, plus the cert
expanded to cover the new name (e.g. `certbot --expand -d example.com -d
beagle.example.com`).

Save as `/etc/apache2/sites-available/beagle.conf`:

```apache
# HTTP: redirect to HTTPS, preserving path and query
<VirtualHost *:80>
    ServerName beagle.example.com
    RewriteEngine On
    RewriteRule ^ https://%{HTTP_HOST}%{REQUEST_URI} [R=301,L]
    ErrorLog  ${APACHE_LOG_DIR}/beagle-http-error.log
    CustomLog ${APACHE_LOG_DIR}/beagle-http-access.log combined
</VirtualHost>

# HTTPS: terminate TLS, proxy to uvicorn on localhost
<VirtualHost *:443>
    ServerName beagle.example.com

    SSLEngine on
    SSLCertificateFile      /etc/letsencrypt/live/example.com/fullchain.pem
    SSLCertificateKeyFile   /etc/letsencrypt/live/example.com/privkey.pem
    Include                 /etc/letsencrypt/options-ssl-apache.conf

    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-Port  "443"
    ProxyPreserveHost On
    ProxyRequests     Off

    # Server-Sent Events stream: must NOT be buffered, must allow long timeouts
    <Location "/api/v1/fixes/stream">
        ProxyPass         "http://127.0.0.1:8765/api/v1/fixes/stream" flushpackets=on connectiontimeout=10 timeout=3600 keepalive=on
        ProxyPassReverse  "http://127.0.0.1:8765/api/v1/fixes/stream"
        Header always set X-Accel-Buffering "no"
        Header always set Cache-Control "no-cache"
    </Location>

    # General API, auth, map, health
    ProxyPass         "/api/"   "http://127.0.0.1:8765/api/"   keepalive=on
    ProxyPassReverse  "/api/"   "http://127.0.0.1:8765/api/"
    ProxyPass         "/auth/"  "http://127.0.0.1:8765/auth/"  keepalive=on
    ProxyPassReverse  "/auth/"  "http://127.0.0.1:8765/auth/"
    ProxyPass         "/map"    "http://127.0.0.1:8765/map"    keepalive=on
    ProxyPassReverse  "/map"    "http://127.0.0.1:8765/map"
    ProxyPass         "/health" "http://127.0.0.1:8765/health" keepalive=on
    ProxyPassReverse  "/health" "http://127.0.0.1:8765/health"

    RedirectMatch ^/$ /map

    ErrorLog  ${APACHE_LOG_DIR}/beagle-error.log
    CustomLog ${APACHE_LOG_DIR}/beagle-access.log combined
</VirtualHost>
```

Then enable and start:
```bash
sudo a2ensite beagle
sudo apache2ctl configtest
sudo systemctl reload apache2
```

Launch Beagle (no `--root-path` needed for subdomain):
```bash
env/bin/beagle-server --config config/server.yaml
```

Set `server.host: 127.0.0.1` in `config/server.yaml` so Beagle's port is
unreachable from outside the host.

### Option 2 - Subpath under an existing site

Mount Beagle under a path on your main vhost, e.g.
`https://example.com/beagle/`.  No DNS or cert changes needed -- you reuse
the existing main vhost.

**Important:** when using a subpath, you must launch Beagle with
`--root-path /beagle` so FastAPI generates absolute URLs with the prefix.
Without it, OAuth callbacks and the rendered map page will fetch
`/api/v1/...` (no prefix), which Apache will route to the main vhost and
return 404 or 405.

Add this snippet **inside** your existing `<VirtualHost *:443>` for
`example.com`:

```apache
# Beagle TDOA aggregation server - subpath under /beagle/
# Upstream: uvicorn on 127.0.0.1:8765 (launched with --root-path /beagle)

RequestHeader set X-Forwarded-Proto "https"
RequestHeader set X-Forwarded-Port  "443"

ProxyPreserveHost On
ProxyRequests     Off

# SSE stream first (most-specific match)
<Location "/beagle/api/v1/fixes/stream">
    ProxyPass         "http://127.0.0.1:8765/api/v1/fixes/stream" flushpackets=on connectiontimeout=10 timeout=3600 keepalive=on
    ProxyPassReverse  "http://127.0.0.1:8765/api/v1/fixes/stream"
    Header always set X-Accel-Buffering "no"
    Header always set Cache-Control "no-cache"
</Location>

ProxyPass         "/beagle/api/"   "http://127.0.0.1:8765/api/"   keepalive=on
ProxyPassReverse  "/beagle/api/"   "http://127.0.0.1:8765/api/"
ProxyPass         "/beagle/auth/"  "http://127.0.0.1:8765/auth/"  keepalive=on
ProxyPassReverse  "/beagle/auth/"  "http://127.0.0.1:8765/auth/"
ProxyPass         "/beagle/map"    "http://127.0.0.1:8765/map"    keepalive=on
ProxyPassReverse  "/beagle/map"    "http://127.0.0.1:8765/map"
ProxyPass         "/beagle/health" "http://127.0.0.1:8765/health" keepalive=on
ProxyPassReverse  "/beagle/health" "http://127.0.0.1:8765/health"

RedirectMatch ^/beagle/?$ /beagle/map
```

Reload Apache:
```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

Launch Beagle **with the matching root-path**:
```bash
env/bin/beagle-server --config config/server.yaml --root-path /beagle
```

Set `server.host: 127.0.0.1` in `config/server.yaml`.

### Verification

After deploying either option, test from a separate machine:

```bash
# Health endpoint (Option 1)
curl -s https://beagle.example.com/health

# Health endpoint (Option 2)
curl -s https://example.com/beagle/health

# HTTP redirect to HTTPS (Option 1 only)
curl -sI http://beagle.example.com/health    # expect 301

# SSE stream stays open and pushes events
curl -N https://beagle.example.com/api/v1/fixes/stream
# (or https://example.com/beagle/api/v1/fixes/stream for Option 2)

# Confirm the cleartext port is now closed
curl -m 5 http://example.com:8765/health      # expect connection refused
```

If the OAuth login flow fails after deployment (Google rejecting the
redirect URI), check that:
1. `proxy_headers=True` is in effect (it is by default in `main.py`)
2. The redirect URI registered in your Google OAuth client matches what
   the server is generating: `https://beagle.example.com/auth/oauth/google/callback`
   (Option 1) or `https://example.com/beagle/auth/oauth/google/callback`
   (Option 2 -- note the `/beagle/` prefix; this only works correctly if
   you launched Beagle with `--root-path /beagle`)

### Migrating nodes after enabling the proxy

Each node's `bootstrap.yaml` must be updated to use the new HTTPS URL:

```yaml
# Before:
server_url: "http://example.com:8765"

# After (Option 1):
server_url: "https://beagle.example.com"

# After (Option 2):
server_url: "https://example.com/beagle"
```

Restart `beagle-node` after editing.  The new HTTPS connection automatically
picks up the cert validation; if you used a self-signed cert during testing,
you'll need to either install a real cert or set `httpx`'s `verify=False`
(not currently exposed in `BootstrapConfig` -- file an issue if you need it).

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

The `--pilot-sigma-us` flag (named for historical reasons; it now models the
combined per-event sync timing noise from the RDS detector + carrier detector
quantisation) sets the 1-sigma sync timing noise, which directly determines
TDOA and position accuracy:

```bash
# Uncalibrated RTL-SDR crystal + carrier detector quantisation - expect ~6 km error
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10 \
    --pilot-sigma-us 20.0

# Typical RDS sync (RSPduo) - expect ~500-1500 m error (default)
python3 scripts/mock_event_generator.py \
    --scenario scripts/mock_scenario_seattle.yaml \
    --delivery-buffer-s 10 \
    --pilot-sigma-us 2.0

# Best case (GPS 1PPS) - expect ~100-400 m error
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

Nodes in bootstrap mode long-poll the server for config changes, and the
server **auto-reloads each node's config file from disk on every poll**.
To update a running node's config, just edit its file in
`remote_configs/<node_id>.yaml` (the path you registered with `set-config
--config-file`) and save it.  No `manage_nodes` command needed, no UI
button to click.

```bash
# Edit the file in place
$EDITOR remote_configs/seattle-north-01.yaml
# That's it -- the next config poll picks it up.
```

The new config propagates to the node within **one poll cycle**, which is
the long-poll wait time configured in the node's `bootstrap.yaml`.  The
default is 60 seconds; the cap is 120 seconds; in practice the node sees
the change within ~1 second of saving the file because the long-poll
returns immediately when `config_version` increments.

**Source-of-truth rule**: when both the file on disk and an API edit
(`PATCH /api/v1/nodes/{node_id}` or `manage_nodes set-config`) have
modified `config_json`, **the file wins on the next poll**.  API edits to
fields that come from the file will be reverted on the next config poll.
If you edit the file and want to verify the change took effect, watch the
Nodes panel in the UI: any reload error (file missing, YAML parse failure,
schema validation failure) is surfaced as a red badge on the node's row.

Schema validation: the file is validated against `NodeConfig.model_validate()`
before being stored.  YAML files with the wrong field names or missing
required fields are rejected at reload time, the previous good config is
left in place, and the error is shown in the Nodes panel.

The behaviour after the new config reaches the node depends on what
changed:

| Changed fields | Behaviour |
|----------------|-----------|
| Carrier thresholds, target channels, clock calibration, sync signal thresholds | **Hot-reloaded** - applied immediately, no downtime |
| SDR hardware params (gain, sample rate, mode, antennas), sync frequency | **Automatic restart** - node exits cleanly (code 75) and systemd restarts it with the new config (~2-3 s downtime) |

No manual intervention is needed in either case.  The systemd unit
(`etc/beagle-node.service`) has `Restart=on-failure` which covers exit code 75.

> **About the `POST /api/v1/nodes/reload-configs` endpoint**: this used
> to be wired to a "Reload Configs" button in the Nodes panel that
> walked all nodes' config files at once.  The button was removed when
> per-poll auto-reload landed -- it became a vestigial way of doing what
> the polls now do continuously.  The endpoint itself is preserved for
> tests and any external automation that depends on it.

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

#### `scripts/verify_rds_sync.py`
Live RDS sync detection display, using the same `RDSSyncDetector` class the
production node pipeline runs.  Reports event rate, internal pilot correlation
quality, crystal calibration drift, and bit-interval jitter.  Run this first
to confirm the sync chain is working before attempting a full pipeline run.

```bash
python3 scripts/verify_rds_sync.py --config config/node.yaml --duration 30
```

Without a config file (bare SoapySDR path -- works with any SoapySDR-supported
device):
```bash
python3 scripts/verify_rds_sync.py --device "driver=sdrplay" --freq 94.9e6 --gain auto --duration 30
```

Pass criteria after warm-up: event rate ~1188/s, `PilotCor` >= 0.5,
crystal drift < 10 ppm, bit interval jitter < 5 usec.  See
[Step 2 of Calibration](#step-2---verify-rds-sync-detection-bit-timing-and-crystal-calibration)
for full pass criteria and example output.

> **Note**: Earlier versions of Beagle used `scripts/verify_sync.py`, which
> exercises the standalone `FMPilotSyncDetector` class.  That script still
> works as a low-level diagnostic for debugging the pilot extraction subsystem,
> but it does **not** test the production sync path.  Use `verify_rds_sync.py`
> for all normal verification.

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

#### `scripts/watch_node_health.py`
Live, formatted view of a running node's `/health` endpoint.  Polls each
second by default and computes derived metrics (sync event rate, deltas)
from successive snapshots.  Works against any node by hostname; the script
itself has no SDR or pipeline dependencies, so you can run it from a laptop
to watch a remote field node over the network.

```bash
# Local node:
python3 scripts/watch_node_health.py

# Remote node on the LAN:
python3 scripts/watch_node_health.py --host dpk-tdoa1.local --port 8080

# One snapshot, then exit:
python3 scripts/watch_node_health.py --host dpk-tdoa1.local --once

# Raw JSON (useful for jq pipelines or logging):
python3 scripts/watch_node_health.py --host dpk-tdoa1.local --json
```

Output groups:

| Section | Fields shown |
|---------|--------------|
| Header | node_id, status (ok / starting / degraded), uptime, degraded reasons |
| Configuration | sdr_mode, sample_rate, sync_station + frequency, target channels |
| Sync | events count + delta, **rate/s** (derived), last_sync age, **corr_peak**, **crystal ppm** |
| Carrier / Target | last_event age, events submitted/dropped + delta, queue_depth, noise_floor, thresholds |
| Hardware | sdr_overflows, backlog_drains |

Colour coding: green = healthy, yellow = marginal, red = degraded or
significant deltas.  Healthy RDS sync shows `rate ~1188/s`, `corr_peak >= 0.5`,
and `crystal` within +/-50 ppm and stable.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Node hostname or IP |
| `--port` | `8080` | Health server port (`health_port` from `node.yaml`) |
| `--interval` | `1.0` | Poll interval in seconds |
| `--no-clear` | off | Append each snapshot instead of clearing the screen (good for piping to a log file) |
| `--once` | off | Print one snapshot and exit |
| `--json` | off | Print raw JSON instead of formatted output |

Press Ctrl-C to exit interactive mode.

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

### Step 2 - Verify RDS Sync Detection, Bit Timing, and Crystal Calibration

**Goal:** Confirm the sync chain produces clean SyncEvents at the RDS bit
rate, that the M&M timing loop has sub-microsecond jitter, and that the
CrystalCalibrator converges to a stable correction factor.

```bash
python3 scripts/verify_rds_sync.py --config config/node.yaml --duration 60
```

The same run covers all three checks: event rate and pilot quality are
visible from the first few rows; crystal convergence requires ~30 s; the
bit-interval jitter summary requires the full run.

**RDS sync detection - pass criteria (read from first ~5 s of output):**
- Event rate: **~1188 / s** (one per RDS bit transition; pilot/16 = 1187.5 Hz exactly)
- `PilotCor` (the internal pilot correlation used for crystal calibration)
  >= **0.5** for a well-received station
- Power in the **-25 to -40 dBFS** range -- avoid ADC clipping near 0 dBFS

**Crystal calibration - pass criteria (read from full 60 s run):**
- `Crystal` converges within ~10 s and stays within **+/-100 ppm**
- Drift between t=10 s and t=60 s: **< 10 ppm** (shown in summary line)
- RSPduo values near 0 ppm (< +/-10 ppm) are normal and excellent

**Bit interval jitter - pass criteria (read from summary line):**
- Mean interval: **~210.5 samples** at 250 kHz sync rate (= 250000 / 1187.5)
- Stdev: **< 0.1 samples** (~ 0.4 usec).  Larger values indicate the M&M
  timing loop is unstable -- usually caused by very weak RDS or strong
  multipath.

**Tuning:**
- If event rate is well below 1188/s, the chosen station probably doesn't
  carry RDS.  Pick a different station -- in the US, all NPR affiliates and
  most commercial FM stations carry RDS.  KUOW 94.9 (Seattle) is a known-good
  reference.
- If `PilotCor` is low, try a stronger station, increase gain, or improve the
  antenna position.
- If gain is too high the ADC will clip; reduce it until the IQ magnitude
  is between -25 and -40 dBFS.
- `min_corr_peak` in `node.yaml` -> `sync_signal.min_corr_peak` filters out
  weak sync events.  Set it to 0.3 for normal operation.

**Interpreting the Crystal column:**
- `Crystal = 0.0 ppm`: crystal is exact (or calibrator not yet converged)
- `Crystal = +50.0 ppm`: crystal runs 50 ppm fast -> corrected automatically
- Values outside +/-200 ppm suggest a low-quality crystal; use a TCXO
  RTL-SDR for best accuracy; RSPduo should always be well within +/-10 ppm

*RSPduo on KUOW 94.9 (24 MHz TCXO):*
```
  Time    Events   Rate/s   PilotCor    Crystal     Power
   5.1      5436   1186.1    0.7068    -10.4 ppm   -36.4 dBFS
  10.2     11465   1187.4    0.7063     -9.5 ppm   -36.4 dBFS
  30.3     34418   1189.7    0.7059     -9.4 ppm   -36.2 dBFS

Crystal drift: -10.1 ppm at t~10 s -> -9.4 ppm at end  (drift=0.7 ppm  OK)
Bit interval: mean=210.53 samples (expected 210.53)  stdev=0.017 samples (0.07 usec)
OK: RDS sync detection looks good
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

Use `verify_rds_sync.py` with the node config file (the same receiver and
pipeline the node uses in production):

```bash
env/bin/python scripts/verify_rds_sync.py \
    --config config/node.yaml --duration 60
```

This confirms RDS sync lock on the sync channel (Tuner 1).  Watch for:
- `Rate/s` ~ 1188/s after warmup -- one event per RDS bit transition
  (1187.5 Hz = pilot/16 exactly)
- `PilotCor` >= 0.5 (the RDS detector still extracts the pilot internally
  for crystal calibration)
- `Crystal` within +/-10 ppm for RSPduo (TCXO)
- `Crystal drift` < 10 ppm in the summary line
- `Bit interval` jitter < 5 usec in the summary line

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

---

## Production deployment (systemd)

```bash
sudo mkdir -p /etc/beagle

# Local config approach:
sudo cp config/node.yaml /etc/beagle/node.yaml
sudo $EDITOR /etc/beagle/node.yaml

# - OR - remote config (bootstrap) approach:
sudo cp config/bootstrap.example.yaml /etc/beagle/bootstrap.yaml
sudo $EDITOR /etc/beagle/bootstrap.yaml   # fill in server_url, node_id, node_secret

# Create service user
sudo useradd -m -r -s /usr/sbin/nologin beagle
sudo usermod -aG plugdev beagle   # USB SDR access

# Install the service
sudo cp etc/beagle-node.service /etc/systemd/system/
# if necessary, update location of beagle_rdf directory you want to use
sudo $EDITOR /etc/systemd/system/beagle-node.service

# Create config cache directory (used by bootstrap mode to survive reboots)
sudo mkdir -p /var/cache/beagle /var/lib/beagle
sudo chown beagle:beagle /var/cache/beagle /var/lib/beagle

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
|-- pipeline/      Signal processing (decimator, FM demod, RDS sync, carrier detect, delta)
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
