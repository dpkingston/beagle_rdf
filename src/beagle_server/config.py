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

        T_sync = onset_time_ns - sync_to_snippet_start_ns - dist(sync_tx, node) / c

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
    min_xcorr_snr: float = 0.5
    """
    Minimum SNR of the knee-finder's d2 peak (|min(d2)| in the transition
    region vs RMS of d2 outside it) to accept a TDOA measurement.  Pairs
    failing this gate are rejected (no fallback — coarse sync_delta has
    ~200 µs noise and is not useful).

    Empirical finding on the Magnolia corpus (2026-04-19, 250 kHz, d2
    knee finder, 360 µs Savgol): tightening the gate kicks out good
    pairs.  d2-SNR is not a clean predictor of knee accuracy because d2
    RMS depends on the ramp slope itself, not just on noise.
      SNR >= 0.5  -> 19/23 pairs,  59 µs median error   <-- default
      SNR >= 1.0  -> 17/23 pairs,  82 µs median error
      SNR >= 1.5  -> 12/23 pairs,  91 µs median error
      SNR >= 2.0  ->  8/23 pairs, 114 µs median error

    Name retained for config compatibility; the gate now applies to d2
    rather than xcorr peak.  0.0 = always accept (debugging only).
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
    savgol_window_us: float = 360.0
    """
    Savitzky-Golay smoothing window for knee finding, in microseconds.
    The server converts this to an odd number of samples at each snippet's
    rate, so the same value works across mixed-hardware deployments.

    360 µs is empirically the best for d2-based knee finding (argmin of
    second derivative locates the ramp/plateau corner):
      - Narrower → d2 dominated by thermal-noise-on-plateau fluctuations.
      - Wider    → the corner shape is smeared beyond the Savgol support,
        d2 minimum flattens and localisation degrades.

    At 62.5 kHz this gives a 23-sample Savgol kernel; at 250 kHz it gives
    a 91-sample kernel — same effective smoothing bandwidth in time.
    Real-corpus performance (Magnolia, 2026-04-19):
      240 µs  -> 181 µs median |err|
      360 µs  ->  59 µs median |err|   <-- default
      720 µs  -> 105 µs median |err|
    """
    sync_diag: bool = False
    """
    Enable detailed sync-timing diagnostic logging.  When True, each TDOA
    computation logs pilot-phase comparison (sync_diag) and grid-calibration
    (sync_cal) lines.  Useful for verifying RDS bit-boundary alignment across
    nodes.  Can also be enabled via the BEAGLE_SYNC_DIAG=1 environment variable.
    """
    tdoa_method: str = "xcorr"
    """
    Pair-level TDOA refinement method:
      "xcorr" (config default, safe for any snippet size):
        cross-correlate d²(power envelope) between the two snippets.
      "phat" (**recommended for production** with large snippets):
        coherent complex-IQ GCC-PHAT on the plateau segment of each
        snippet after per-node LO-offset removal.  Robust to receiver-
        channel mismatches (multipath, AGC).  Requires snippets sized
        for ~30 ms of post-knee plateau, i.e.
        ``carrier.snippet_samples >= 16384`` at 250 kHz (production
        post-2026-04-24).  Empirically improves pooled median |err|
        by ~17 % over "xcorr" at 3× yield.  Enable by setting
        ``solver.tdoa_method: phat`` in the server config.
      "knee": per-node Savgol-smoothed second-derivative knee finder.
        Uses ``savgol_window_us``.  Retained for comparison.
    """
    xcorr_resample_rate_hz: float | None = None
    """
    Target sample rate (Hz) to resample IQ snippets to before cross-correlation
    when two nodes captured at different rates.  Applied when
    ``tdoa_method="xcorr"``.

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
    """
    boundary_clamp_km: float = 2.0
    """
    Suppress fixes whose converged position lies within this many kilometres
    of the search-area bounds.  Such fixes are typically the L-BFGS-B
    optimizer being trapped at the constraint boundary because the true
    minimum lies outside the search area, OR the cost surface drives
    iteration against the bound — neither produces a valid transmitter
    estimate.  Set to 0 to disable.
    """
    multistart_disagreement_km: float = 5.0
    """
    Suppress fixes whose multistart-converged positions span more than this
    distance among results within 2x the best cost.  Such fixes have a
    rough cost surface with multiple comparable local minima; the choice
    of "best" is noise-dependent and unreliable.  Set to 0 to disable.

    Symptom in production: clusters of fixes piling up at multistart
    convergence attractors that are not the true transmitter location.
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


class TdoaCalibrationConfig(BaseModel):
    """
    Pair / per-node TDOA bias calibration.

    Each node carries an intrinsic timing offset (cable delay, processing
    pipeline offset, crystal-clock idiosyncrasies) and each pair may carry
    additional bias from multipath geometry to the calibration target.
    Two correction models are supported, with per-pair preferred when both
    are populated:

      Per-pair (most accurate, target-specific):
          calibrated_tdoa(a, b) = compute_tdoa_s(a, b) - pair_offset(a, b)
        Captures the full observable bias structure for a known-position
        transmitter, including any multipath-driven pair-specific terms.
        Stored as ``pair_offsets_s`` keyed by ``"a,b"`` with a < b.

      Per-node (more general, weaker fit):
          calibrated_tdoa(a, b) = compute_tdoa_s(a, b) - (δ_a - δ_b)
        Stored as ``node_offsets_s``.  Uses 1 free parameter per node
        instead of 1 per pair, so it cannot capture pair-specific biases
        (multipath, geometry) — those land as residuals.  Generalises
        across transmitter bearings IF the bias is genuinely per-node
        (clock/cable, not multipath).

    Both are fitted from observed pair biases against a known-position
    transmitter (see ``scripts/fit_tdoa_calibration.py``).  Sign
    convention: bias > 0 means the calibrated TDOA is greater than the
    geometric-expected TDOA before correction.

    The model is plateau-only: per-event-type biases differ by ~7-17 µs
    on real hardware (different code paths in compute_tdoa_s for onset/
    offset/plateau), so calibration is fitted from plateau measurements
    and consumed by plateau measurements.  Onset/offset measurements
    are uncalibrated and should not be used for fine fixes.
    """

    enabled: bool = False
    """Apply the calibration corrections.  Default False so that an
    unfitted calibration table doesn't silently bias output."""

    node_offsets_s: dict[str, float] = Field(default_factory=dict)
    """
    Per-node δ in **seconds**, relative to the chosen reference node.
    Empty dict (default) means "no per-node calibration."  Nodes not
    present are treated as δ = 0 (reference-aligned).  Used only when
    ``pair_offsets_s`` is empty — pair calibration is preferred when
    both are populated.

    Example::

        node_offsets_s:
          dpk-tdoa1:        0.0          # reference
          dpk-tdoa2:        7.928e-6     # +7.928 µs
          n7jmv-tdoa-qth:  -74.861e-6    # -74.861 µs
    """

    pair_offsets_s: dict[str, float] = Field(default_factory=dict)
    """
    Per-pair calibration in **seconds**, keyed by ``"<node_a>,<node_b>"``
    with the two node IDs in **ascending sort order**.  When non-empty,
    overrides ``node_offsets_s`` — the per-pair table is more specific.

    The stored value is the empirically-observed bias of
    ``compute_tdoa_s(a, b) - geometric_expected(a, b)`` (in seconds), so
    that ``calibrated = measured - pair_offsets_s[sorted(a,b)]``.  When a
    pair is queried in reverse order (b, a), the caller MUST negate the
    looked-up value: ``bias(a,b) = -bias(b,a)``.

    Example::

        pair_offsets_s:
          "dpk-tdoa1,dpk-tdoa2":      9.7136e-05    # +97.136 µs
          "dpk-tdoa1,kb7ryy":         1.5525e-04    # +155.252 µs
          "dpk-tdoa1,n7jmv-tdoa-qth": 1.3539e-04    # +135.395 µs
          "dpk-tdoa2,kb7ryy":         9.7268e-05    # +97.268 µs
          "dpk-tdoa2,n7jmv-tdoa-qth": 1.1987e-04    # +119.869 µs
          "kb7ryy,n7jmv-tdoa-qth":    1.69e-07      # +0.169 µs
    """

    reference_node: str = ""
    """Documentation only: which node has δ = 0 by construction in
    per-node mode.  Unused for per-pair mode."""

    fit_mode: str = "per_node"
    """Which calibration mode is in effect.  Documentation only — the
    code uses pair_offsets_s if populated, else node_offsets_s.  Values:
    "per_node", "per_pair"."""

    fit_transmitter_label: str = ""
    """Documentation: label of the transmitter the calibration was fitted against."""
    fit_transmitter_lat: float = 0.0
    fit_transmitter_lon: float = 0.0
    fit_n_pairs: int = 0
    """Number of pair-event observations used in the fit (provenance)."""
    fit_residual_rms_us: float = 0.0
    """Residual RMS of the per-node δ model fit, in µs.  A value > 5-10 µs
    suggests the per-node model is not capturing the true bias structure
    (e.g. multipath-driven pair-specific terms) and per-pair mode would
    track Magnolia much more accurately, at the cost of being target-
    specific.  Zero when fit_mode is per-pair."""
    fit_date: str = ""
    """ISO date string YYYY-MM-DD."""


class ServerFullConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    tdoa_calibration: TdoaCalibrationConfig = Field(default_factory=TdoaCalibrationConfig)


def load_config(path: str | Path) -> ServerFullConfig:
    """Load and validate server config from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    return ServerFullConfig.model_validate(raw)
