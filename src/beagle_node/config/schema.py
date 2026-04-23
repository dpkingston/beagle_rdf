# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Node configuration schema and YAML loader.

All configuration is validated by Pydantic v2 on load. Invalid configs
raise a descriptive ValidationError rather than failing silently at runtime.
"""

from __future__ import annotations

import os
import re
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class NodeLocation(BaseModel):
    """Geographic location of a Beagle node.

    Beagle's TDOA solver is two-dimensional (lat/lon only); altitude
    contributes negligibly compared to the carrier-detector quantisation
    floor and was never read by any code path.  Older configs may still
    include `altitude_m` and `uncertainty_m` fields -- Pydantic v2's
    default `extra="ignore"` silently drops them, so legacy configs
    continue to load without errors.
    """
    latitude_deg: float
    longitude_deg: float


class FreqHopConfig(BaseModel):
    """Configuration for freq_hop mode (pyrtlsdr callback-based continuous hopping)."""

    device_serial: str | None = None
    """USB serial number of the RTL-SDR device to open (e.g. '00000001').
    Run `rtl_test` or `lsusb -v` to find the serial of your dongle.
    If None, the first RTL-SDR found is used (suitable for single-dongle setups)."""
    samples_per_block: int = 65_536
    """Samples per block for the sync (FM) channel, and the default for the
    target channel when target_samples_per_block is not set.  ~32 ms at 2.048 MSPS."""
    target_samples_per_block: int | None = None
    """Samples per block for the target (LMR) channel.
    When set, enables asymmetric mode: the sync channel uses samples_per_block
    and the target channel uses this value.  Larger target blocks increase the
    fraction of time spent on the target channel.  Must be a multiple of
    samples_per_block, or vice versa (GCD determines the USB callback buffer size).
    Leave unset (None) for symmetric mode (both channels equal)."""
    settling_samples: int = 40_960
    """Samples to discard at the start of each block for R820T tuner settling (~20 ms).
    Applied to both channels.  Set to the maximum measured settling time
    across both hop directions (see scripts/measure_settling.py)."""
    sample_rate_hz: float = 2_048_000.0
    gain_db: float | str = 40
    pipeline_offset_ns: int = 0
    """Freq_hop pipeline timing correction (ns) relative to the theoretical
    zero-delay reference (physical onset-at-antenna minus physical pilot
    zero-crossing-at-antenna, with no pipeline delay).

    Because freq_hop processes channels sequentially (sync block then target
    block), the sync event matched to an onset occurred in a preceding sync
    block.  sync_to_snippet_start_ns therefore includes the remaining time in
    the sync block after the last sync event, plus settling_samples/sample_rate_hz,
    plus the time to onset detection within the target block.  This structural
    delay (typically 20-40 ms depending on block sizes and settling) is absent
    in RSPduo (simultaneous channels) and single_sdr nodes.

    Calibration: run scripts/colocated_pair_test.py --db against a co-located
    RSPduo or single_sdr reference node.  Set pipeline_offset_ns to the mean
    TDOA_AB for onset events with the sign that drives TDOA_AB toward zero
    (typically a large positive value, e.g. +26_500_000 for a 26.5 ms offset)."""


class RSPduoConfig(BaseModel):
    """Configuration for rspduo mode (SDRplay RSPduo dual-tuner)."""

    sample_rate_hz: float = 2_000_000.0
    """Sample rate for BOTH channels. Maximum 2 MHz in dual-tuner mode."""
    sync_gain_db: float | str = "auto"
    """Gain for the sync (FM) channel in dB, or 'auto' for AGC."""
    sync_lna_state: int = 9
    """LNA state for the sync channel.  Higher = more LNA attenuation.
    For RSPduo at VHF/FM: 0 = max LNA gain, 6 = min LNA gain.
    Default 9 (or driver-clamped maximum) minimises ADC saturation from
    strong FM broadcasts that would contaminate ch1 via the TDM ADC."""
    target_gain_db: float | str = "auto"
    """Gain for the target (LMR) channel in dB, or 'auto' for AGC."""
    target_lna_state: int = 0
    """LNA state for the target channel.  0 = maximum LNA gain (best
    sensitivity).  Increase only if the target signal overloads the ADC."""
    sync_antenna: str | None = None
    """Antenna port for Tuner 1 (sync/FM channel).
    If None, uses the SoapySDRPlay3 default. Set explicitly if the wrong
    physical port is selected by default. Check startup log output for
    'available antennas' to see the exact names your driver uses.
    Typical values: 'Antenna B' (50-ohm SMA, FM/VHF/UHF),
                    'Antenna A' (Hi-Z SMA, HF/MW only, < 30 MHz)."""
    target_antenna: str | None = None
    """Antenna port for Tuner 2 (target/LMR channel).
    If None, uses the SoapySDRPlay3 default (typically 'Antenna C')."""
    master_device_args: str = "driver=sdrplay"
    """SoapySDR device args for the master device (first tuner opened)."""
    slave_device_args: str | None = None
    """SoapySDR device args for the slave device.
    If None, uses master_device_args - correct for a single RSPduo."""
    buffer_size: int = 65_536
    """IQ samples per read call per channel (~33 ms at 2 MSPS)."""
    pipeline_offset_ns: int = 0
    """RSPduo pipeline timing correction (ns) relative to the theoretical
    zero-delay reference (physical onset-at-antenna minus physical pilot
    zero-crossing-at-antenna, with no pipeline delay).

    For the RSPduo in DT mode the dominant hardware effect is the TDM ADC
    interleave: ch1 (target) lags ch0 (sync) by 0.5 ADC periods.  At 2 MSPS
    one ADC period = 500 ns -> interleave ~ 250 ns.  Default 0 is adequate for
    most deployments; set to 250 to correct when sub-usec accuracy is needed and
    the RSPduo is paired with another RSPduo reference.

    Calibration: run scripts/colocated_pair_test.py --db with this RSPduo node
    and a co-located RSPduo reference node.  Set pipeline_offset_ns to the mean
    TDOA_AB for onset events with the sign that drives TDOA_AB toward zero.
    When paired with a freq_hop reference, calibrate the freq_hop node's
    pipeline_offset_ns instead (the large inter-block structural delay belongs
    to the freq_hop implementation, not to the RSPduo)."""


class SDRChannelConfig(BaseModel):
    """Configuration for one SDR in two_sdr or single_sdr mode."""

    backend: Literal["soapy"] = "soapy"
    device_args: str = ""
    sample_rate_hz: float = 2_048_000.0
    center_frequency_hz: float
    gain_db: float | str = "auto"
    buffer_size: int = 131_072


class FMStation(BaseModel):
    station_id: str
    """Human-readable ID, e.g. 'KISW_99.9'."""
    frequency_hz: float
    latitude_deg: float
    longitude_deg: float


class SyncSignalConfig(BaseModel):
    type: Literal["fm_pilot"] = "fm_pilot"
    primary_station: FMStation
    secondary_station: FMStation | None = None
    sync_period_ms: float = 10.0
    """Emit one SyncEvent every this many milliseconds."""
    min_corr_peak: float = 0.3
    """Minimum cross-correlation peak to accept a SyncEvent (0-1)."""
    max_sync_age_ms: float = 200.0
    """Maximum age of a SyncEvent that can be paired with a carrier onset/offset.
    Increase if you see 'Dropping onset/offset: no sync within N samples' warnings.
    At 256 kHz sync-dec rate, 200 ms = 51200 samples (~28x sync_period at 7 ms).
    For RSPduo (250 kHz sync-dec), 200 ms = 50000 samples."""


class TargetChannelConfig(BaseModel):
    frequency_hz: float
    label: str = ""


class CarrierDetectConfig(BaseModel):
    """Carrier detector tuning parameters.

    All fields are optional - omit the entire ``carrier:`` block to keep
    the pipeline defaults.  Tune these when you see false detections
    (lower thresholds / increase min_hold) or missed signals (raise
    thresholds / reduce min_hold).

    onset_db / offset_db calibration
    ---------------------------------
    Real LMR signals at the ADC are typically -40 to -60 dBFS.  If the
    node never detects a carrier, lower ``onset_db`` (e.g. -50).  The
    default -30 dBFS works well when gain is set correctly.

    min_release_windows
    -------------------
    Set to 4-8 (~ 4-8 ms at the default 64-sample / 62.5 kHz window) to
    suppress chattering caused by brief power fades mid-transmission.
    Chattering produces bursts of measurements spaced by exactly one
    power window (~1 ms); if you see that pattern, increase this value.
    """

    onset_db: float = -30.0
    """Power level (dBFS) that triggers CarrierOnset.

    When ``auto_threshold_margins`` is True this value is used only as the
    initial threshold during the noise-floor warmup period; after warmup the
    threshold tracks the noise floor at ``onset_margin_db`` above it."""
    offset_db: float = -40.0
    """Power level (dBFS) below which CarrierOffset fires.
    Must be lower (more negative) than onset_db.  Same warmup caveat as
    ``onset_db`` when ``auto_threshold_margins`` is True."""
    auto_threshold_margins: bool = True
    """If True, onset/offset thresholds track the tracked noise floor at the
    margins below; static onset_db/offset_db apply only during warmup (before
    the noise-floor EMA has stabilised).  If False, thresholds stay fixed at
    the static values regardless of noise floor changes.

    Matches the GUI "Auto-Calibrate" button (onset = floor + 12,
    offset = floor + 6) but applied continuously at runtime so the detector
    follows changing noise conditions without operator intervention."""
    onset_margin_db: float = 12.0
    """dB above the tracked noise floor at which onset fires when
    ``auto_threshold_margins`` is True.  Ignored otherwise."""
    offset_margin_db: float = 6.0
    """dB above the tracked noise floor at which offset fires when
    ``auto_threshold_margins`` is True.  Must be less than
    ``onset_margin_db`` to preserve hysteresis.  Ignored otherwise."""
    auto_threshold_update_interval_s: float = 2.0
    """How often (seconds) to re-evaluate and apply auto-tracked thresholds.
    Smaller values track noise changes faster at the cost of slightly more
    log noise; the noise-floor EMA's ~100-window time constant (~0.1 s at
    64-sample windows / 64 kHz) sets the practical lower bound."""
    window_samples: int = 64
    """IQ samples averaged per power measurement window.
    At 62.5 kHz target rate: 64 samples ~ 1 ms."""
    min_hold_windows: int = 1
    """Consecutive above-threshold windows required before onset fires.
    1 = fire immediately; 4 = require ~4 ms of sustained carrier."""
    min_release_windows: int = 1
    """Consecutive below-threshold windows required before offset fires.
    1 = fire immediately; 4-8 recommended for real-world RF signals."""
    snippet_samples: int = 640
    """IQ samples captured per event snippet.

    The ring buffer holds the most recent ``snippet_samples`` of IQ before the
    threshold crossing, and (when ``snippet_post_windows > 0``) additional
    post-event samples are appended up to this total.

    Each sample is 2 int8 bytes on the wire (interleaved real/imag), so the
    payload per event ~ snippet_samples x 2 x 4/3 bytes base64-encoded:

      640  samples -> ~1.7 KB   (default; 10 ms at 64 kHz)
      1280 samples -> ~3.4 KB   (20 ms)
      2560 samples -> ~6.8 KB   (40 ms - calibration/diagnostic mode)

    Use ``--analyze-snippets`` in colocated_pair_test.py to determine the
    minimum value needed to cover your transmitters' full rise/fall time, then
    scale back to avoid unnecessary bandwidth and storage."""
    ring_lookback_windows: int | None = None
    """Depth of the IQ ring buffer in detector windows (independent of snippet size).

    For offset events the PA shutoff must lie within this buffer at detection
    time.  Detection fires min_release_windows after offset_db is first crossed,
    but a gradual fade can push the shutoff further back.

    None (recommended) -> auto-sized as max(3 x snippet_windows, min_for_full_snippet).
    This covers both offset fade headroom and the minimum needed to fully populate
    snippet_samples after concatenating with snippet_post_windows of post-event IQ.

    Set explicitly only if your transmitter has fades longer than ~3 x snippet
    length.  If set too small to fill snippet_samples (i.e. below
    ceil(snippet_samples/window_samples) - snippet_post_windows), a warning is
    logged at startup and emitted snippets are silently truncated.

    Memory cost: ring_lookback_windows x window_samples x 8 bytes."""
    min_active_windows_for_offset: int = 0
    """Minimum above-threshold windows required before a CarrierOffset is emitted.

    Parallel to the onset-side ``_min_idle_for_onset`` guard.  When
    ``prime_state()`` sets state to ``active`` (carrier already present at block
    start), this many windows of power above ``offset_db`` must accumulate before
    a CarrierOffset is allowed.  Offsets that fire before this threshold are
    suppressed - they represent carrier tails from a previous transmission that
    was ongoing during the sync block and drops within the first few windows of
    the target block.

    0 (default) = disabled.  With prime_state() using onset_db as its
    threshold, PLL settling transients no longer trigger false active states,
    so this guard is rarely needed.  Set to 4-10 as defence-in-depth if
    spurious block-start offsets persist.

    Note: offsets that follow a genuine onset in the same block (i.e., the
    carrier started from idle) are never suppressed by this guard, regardless
    of how quickly the carrier dropped."""
    snippet_post_windows: int = 5
    """Post-event windows to append to each IQ snippet (for center-anchored xcorr).

    When > 0, the detector defers emitting a CarrierOnset/CarrierOffset until it has
    collected this many additional windows of IQ *after* the threshold crossing.
    _encode_combined() / _encode_offset_snippet() then trim the concatenated
    pre+post data so that the detection point lands at the midpoint of the
    shipped snippet (``snippet_samples // 2``), giving the server symmetric
    context around the PA knee.

    To achieve the midpoint placement, the collected post-event IQ must be at
    least ``snippet_samples // 2`` samples, i.e.

        snippet_post_windows * window_samples >= snippet_samples // 2

    With fewer post samples, the trim clamps at the tail of the available data
    and the detection point slides toward the end of the snippet, pushing the
    knee into the Savgol-filter edge zone of the server's knee finder.

    Trade-offs:
    - Latency: each event is delayed by snippet_post_windows x window_samples /
      sample_rate_hz (e.g. 45 windows x 64 samples / 250000 Hz ~ 11.5 ms).
    - If an offset occurs during post-collection for an onset (or vice versa), the
      pending event is emitted immediately with only the pre-event snippet."""

    @model_validator(mode="after")
    def check_thresholds(self) -> "CarrierDetectConfig":
        if self.offset_db >= self.onset_db:
            raise ValueError(
                f"carrier.offset_db ({self.offset_db}) must be less than "
                f"carrier.onset_db ({self.onset_db})"
            )
        if self.offset_margin_db >= self.onset_margin_db:
            raise ValueError(
                f"carrier.offset_margin_db ({self.offset_margin_db}) must be less "
                f"than carrier.onset_margin_db ({self.onset_margin_db})"
            )
        if self.auto_threshold_update_interval_s <= 0.0:
            raise ValueError(
                f"carrier.auto_threshold_update_interval_s must be > 0, "
                f"got {self.auto_threshold_update_interval_s}"
            )
        return self


class ReporterConfig(BaseModel):
    server_url: str = ""
    auth_token: str = ""
    """Bearer token. If empty, falls back to TDOA_AUTH_TOKEN env var."""
    batch_size: int = 10
    flush_interval_ms: float = 500.0
    max_queue_size: int = 1_000
    timeout_s: float = 10.0
    max_events_per_window: int = 5
    """
    Maximum number of events the reporter will submit in any rolling
    ``events_rate_window_s`` window.  Events beyond this cap are dropped at
    ``submit()`` before reaching the server (which has its own, stricter
    per-node rate limit).  Prevents a chattering carrier detector from
    flooding the upstream.  Set to 0 to disable.  Default 5 / 5s.
    """
    events_rate_window_s: float = 5.0
    """Sliding-window duration (seconds) for ``max_events_per_window``."""

    @model_validator(mode="after")
    def resolve_auth_token(self) -> "ReporterConfig":
        if not self.auth_token:
            self.auth_token = os.environ.get("TDOA_AUTH_TOKEN", "")
        return self

    @model_validator(mode="after")
    def check_rate_limit(self) -> "ReporterConfig":
        if self.max_events_per_window < 0:
            raise ValueError(
                f"reporter.max_events_per_window must be >= 0, "
                f"got {self.max_events_per_window}"
            )
        if self.events_rate_window_s <= 0.0:
            raise ValueError(
                f"reporter.events_rate_window_s must be > 0, "
                f"got {self.events_rate_window_s}"
            )
        return self


class ClockConfig(BaseModel):
    source: Literal["gps_1pps", "ntp", "system"] = "gps_1pps"
    calibration_offset_ns: int = 0
    """Subtract this from all wall-clock timestamps (compensates for buffer latency)."""


# ---------------------------------------------------------------------------
# Top-level NodeConfig
# ---------------------------------------------------------------------------


class NodeConfig(BaseModel):
    node_id: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    health_port: int = 8080

    location: NodeLocation

    sdr_mode: Literal["freq_hop", "two_sdr", "single_sdr", "rspduo"]

    # Mode-specific SDR config (only the relevant one is required)
    freq_hop: FreqHopConfig | None = None
    rspduo: RSPduoConfig | None = None
    sync_sdr: SDRChannelConfig | None = None
    target_sdr: SDRChannelConfig | None = None
    pps_injection: bool = False
    """True if GPS 1PPS is wired to both SDR inputs (required for two_sdr precision)."""

    sync_signal: SyncSignalConfig
    target_channels: list[TargetChannelConfig]

    carrier: CarrierDetectConfig = CarrierDetectConfig()
    reporter: ReporterConfig = ReporterConfig()
    clock: ClockConfig = ClockConfig()

    @field_validator("node_id")
    @classmethod
    def valid_node_id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", v):
            raise ValueError(
                "node_id must contain only lowercase letters, digits, and hyphens"
            )
        return v

    @field_validator("target_channels")
    @classmethod
    def at_least_one_channel(cls, v: list[TargetChannelConfig]) -> list[TargetChannelConfig]:
        if not v:
            raise ValueError("At least one target_channel must be configured")
        return v

    @model_validator(mode="after")
    def check_mode_config(self) -> "NodeConfig":
        if self.sdr_mode == "freq_hop" and self.freq_hop is None:
            raise ValueError("sdr_mode='freq_hop' requires a 'freq_hop' config block")
        if self.sdr_mode == "rspduo" and self.rspduo is None:
            raise ValueError("sdr_mode='rspduo' requires an 'rspduo' config block")
        if self.sdr_mode in ("two_sdr", "single_sdr"):
            if self.target_sdr is None:
                raise ValueError(
                    f"sdr_mode='{self.sdr_mode}' requires a 'target_sdr' config block"
                )
        if self.sdr_mode == "two_sdr" and self.sync_sdr is None:
            raise ValueError("sdr_mode='two_sdr' requires a 'sync_sdr' config block")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str) -> NodeConfig:
    """
    Load and validate a node configuration from a YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    pydantic.ValidationError
        If the YAML content fails schema validation.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return NodeConfig.model_validate(raw)
