# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Aggregation server configuration schema and YAML loader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""
    """Bearer token required on write endpoints. Empty string disables token auth."""
    node_auth: Literal["none", "token", "nodedb"] = "token"
    """
    How nodes authenticate when POSTing events:
      none   - no authentication required on event POSTs
      token  - shared Bearer token (uses auth_token)
      nodedb - per-node secrets stored in the nodes table (see manage_nodes.py)

    Independent of user_auth.  For example, node_auth: nodedb + user_auth: userdb
    gives per-node secrets for events and per-user logins for the UI.
    """
    user_auth: Literal["none", "token", "userdb"] = "token"
    """
    How humans authenticate for the map UI and admin API:
      none   - no authentication required (open server)
      token  - shared Bearer token (uses auth_token)
      userdb - per-user accounts with roles stored in the users table.
               Admin users are required for management endpoints; viewer users
               can access read-only endpoints.  Use POST /auth/register to
               create the first admin account (open only while no users exist).

    Independent of node_auth.
    """
    session_lifetime_hours: float = 24.0
    """
    How long a user session token remains valid after creation (userdb mode).
    Tokens are invalidated on logout regardless of expiry.
    """
    node_rate_limit_events: int = 10
    """Maximum number of events accepted from a single node within the sliding
    window.  Excess events are rejected with HTTP 429.  Set to 0 to disable."""
    node_rate_limit_window_s: float = 20.0
    """Sliding window duration (seconds) for the per-node rate limit."""

    google_client_id: str = ""
    """Google OAuth2 client ID. Empty disables Google login."""
    google_client_secret: str = ""
    """Google OAuth2 client secret."""

    @model_validator(mode="after")
    def resolve_auth_token(self) -> "ServerConfig":
        if not self.auth_token:
            self.auth_token = os.environ.get("TDOA_SERVER_AUTH_TOKEN", "")
        if not self.google_client_id:
            self.google_client_id = os.environ.get("TDOA_GOOGLE_CLIENT_ID", "")
        if not self.google_client_secret:
            self.google_client_secret = os.environ.get("TDOA_GOOGLE_CLIENT_SECRET", "")
        return self


class DatabaseConfig(BaseModel):
    path: str = "data/tdoa_data.db"
    """
    Path to the operational SQLite database (events, fixes, heatmap).
    This file accumulates transient detection data and can be deleted freely
    to start with a clean slate - node registrations and user accounts are
    preserved in registry_path.  Created on first run.
    """
    registry_path: str = "data/tdoa_registry.db"
    """
    Path to the registry SQLite database (nodes, node_config_history,
    users, user_sessions).  This file contains permanent configuration data
    that must survive operational resets.  Created on first run.
    """


class PairingConfig(BaseModel):
    correlation_window_s: float = 0.2
    """
    Bucketing window (seconds) for T_sync - the estimated absolute time of the FM
    sync event at the transmitter, computed per-event as:

        T_sync = onset_time_ns - sync_delta_ns - dist(sync_tx, node) / c

    Events from different nodes whose T_sync values fall in the same bucket are
    treated as hearing the same LMR transmission.  Rapid key-ups produce distinct
    T_sync values and are correctly separated into different groups.

    Set to >= 2* your worst inter-node NTP offset.  Fixture data shows a maximum
    observed inter-node T_sync spread of ~39 ms, so 0.2 s (+/-100 ms half-window)
    provides a comfortable 2.5x safety margin for typical internet NTP:
      0.2 s   - standard internet NTP (observed max spread ~39 ms, +/-100 ms margin)
      0.005 s - GPS-disciplined nodes
    """
    delivery_buffer_s: float = 10.0
    """
    Seconds to wait after the first event in a group before running the fix solver.
    Set to the observed worst-case one-way delivery latency from the slowest node
    (time from LMR event occurrence to server receiving the POST).
    """
    group_expiry_s: float = 60.0
    """Discard unpaired event groups after this many seconds."""
    freq_tolerance_hz: float = 1000.0
    """Channel frequency tolerance for matching (Hz)."""
    min_corr_peak: float = 0.1
    """Minimum sync_corr_peak to accept an event. Lower quality events are discarded."""
    marginal_corr_peak: float = 0.5
    """
    Mean sync_corr_peak below which a node channel is flagged as 'marginal' by the
    GET /api/v1/nodes/snr endpoint.  A marginal node has a weak FM pilot lock and
    may produce noisier TDOA measurements than a healthy node.
    Must be > min_corr_peak (values below that are rejected at ingest).
    """
    min_nodes: int = 2
    """
    Minimum number of distinct receiver nodes required to attempt a solution.
    3+ nodes produce a unique 2-D position fix; 2 nodes produce a hyperbolic
    line-of-position (LOP) displayed as a dashed arc on the map.  Groups with
    fewer nodes are logged and discarded.
    """


class SolverConfig(BaseModel):
    search_center_lat: float = 47.7
    search_center_lon: float = -122.3
    search_radius_km: float = 100.0
    """Solver search area. Set to the approximate centre of your deployment region."""
    max_residual_ns: float = 0.0
    """
    Reject fixes whose RMS TDOA residual exceeds this threshold (nanoseconds).
    0 = accept all fixes regardless of residual (useful during initial setup).
    A good fix with 3 GPS-disciplined nodes will have residual < 500 ns.
    NTP-clocked nodes may see 5000-50000 ns residuals; tune empirically.
    """
    min_xcorr_snr: float = 1.5
    """
    Minimum SNR of the knee-finder's d1 peak (peak magnitude vs RMS of d1
    samples outside the peak region) to accept a TDOA measurement.  Pairs
    failing this gate are rejected (no fallback — coarse sync_delta has
    ~200 µs noise and is not useful).

    Counter-intuitive finding: wider Savgol windows give better per-event
    precision but LOWER SNR (the d1 peak flattens as the smoothing spreads
    it across more samples, while AM-modulation noise on the plateau only
    partially attenuates).  At 240 µs window and 250 kHz snippet rate,
    real PA transitions typically give SNR 2-4 (vs 5-10 at 62.5 kHz with
    the same window in samples).  A 1.5 gate accepts those while still
    rejecting noise-only snippets (SNR < 1.2).

    Recommended: 10.0 (cleanly separates real transitions from noise).
    0.0 = always accept (use only for debugging).
    """
    max_xcorr_baseline_km: float = 50.0
    """
    Maximum physically plausible TDOA, expressed as the equivalent one-way
    propagation distance in kilometres.  Any xcorr lag exceeding this limit is
    treated as a noise peak and the pair falls back to sync_delta subtraction.

    The check is: |lag_ns| <= max_xcorr_baseline_km * 1000 / c * 1e9

    Set to the maximum node-to-node baseline distance in your deployment area.
    For a 50 km deployment: |lag| <= ~167 usec.  Out-of-range results are almost
    always side-lobe noise artefacts from the power-envelope correlation.
    0.0 = disable the check (accept all xcorr lags regardless of magnitude).
    """
    savgol_window_us: float = 240.0
    """
    Savitzky-Golay smoothing window for knee finding, in microseconds.
    The server converts this to an odd number of samples at each snippet's
    rate, so the same value works across mixed-hardware deployments.

    240 µs is the empirical sweet spot:
      - Narrower → more AM noise admitted, SNR drops below the gate.
      - Wider → smears the knee position, reduces per-event precision.

    At 62.5 kHz this gives a 15-sample Savgol kernel (historical default).
    At 250 kHz it gives a 61-sample kernel — same effective bandwidth.
    """
    sync_diag: bool = False
    """
    Enable detailed sync-timing diagnostic logging.  When True, each TDOA
    computation logs pilot-phase comparison (sync_diag) and grid-calibration
    (sync_cal) lines.  Useful for verifying RDS bit-boundary alignment across
    nodes.  Can also be enabled via the BEAGLE_SYNC_DIAG=1 environment variable.
    """
    xcorr_resample_rate_hz: float | None = None
    """
    Target sample rate (Hz) to resample IQ snippets to before cross-correlation
    when two nodes captured at different rates.

    Mixed-hardware deployments produce snippets at different rates.
    With target_decimation=8 (post-2026-04-19):
      RTL-SDR:  2,048,000 / 8 = 256,000 Hz
      RSPduo:   2,000,000 / 8 = 250,000 Hz
    Correlating them at different rates introduces a ~2.4% lag error, which
    is eliminated by resampling to a common rate first.

    None (default): automatically use the lower of the two node rates for each
    pair (prefer downsampling; no interpolated data introduced; PA transition
    bandwidth is ~few kHz, well below either node's Nyquist).

    Set explicitly to force all pairs to a specific rate regardless of hardware:
      250000.0 - RSPduo native rate (all nodes downsampled if needed)
      256000.0 - RTL-SDR native rate (RSPduo nodes upsampled if needed)

    NOTE: this setting only applied to the legacy inter-node xcorr pipeline
    (cross_correlate_snippets).  The current per-node knee finder works in
    each snippet's native rate independently and does not resample.
    """


class MapConfig(BaseModel):
    output_dir: str = "data/maps"
    """Directory where Folium HTML map files are written."""
    max_age_s: float = 3600.0
    """
    Default age-out window (seconds) for displaying fixes on the map.
    0 = show all fixes ever recorded.
    Overridable per request via ?max_age_s= query param.
    """
    hyperbola_points: int = 500
    """Number of points used to render each hyperbola arc."""
    heatmap_cell_m: float = 200.0
    """
    Heat map grid cell size in metres.  Each fix contributes Gaussian-weighted
    counts to all cells within ~3 * heatmap_sigma_cells of its position.
    Smaller values give finer spatial resolution at higher DB cost.
    Changing this value invalidates accumulated data; reset the heat map.
    """
    heatmap_sigma_cells: float = 1.5
    """
    Standard deviation of the Gaussian kernel in cell units.
    1.5 means a fix at a cell centre gives its neighbours (1 cell away) weight
    exp(-0.5) = 0.61, neighbours 2 cells away weight exp(-2) = 0.14, etc.
    """


class ServerFullConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    map: MapConfig = Field(default_factory=MapConfig)


def load_config(path: str | Path) -> ServerFullConfig:
    """Load and validate server config from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    return ServerFullConfig.model_validate(raw)
