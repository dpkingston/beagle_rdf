# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
CarrierEvent - the primary data unit produced by a Beagle node.

One CarrierEvent is produced per carrier onset detection. The primary timing
value is ``sync_to_snippet_start_ns``: nanoseconds from the matched sync event
to the first sample of the shipped IQ snippet. ``onset_time_ns`` is a rough
absolute wall-clock timestamp used only for event association across nodes.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NodeLocation(BaseModel):
    """Geographic location of a Beagle node, embedded in every CarrierEvent.

    Two-dimensional only -- the TDOA solver does not use altitude.
    Pydantic v2's default `extra="ignore"` lets older event payloads
    that still carry `altitude_m` / `uncertainty_m` deserialize cleanly.
    """
    latitude_deg: float
    longitude_deg: float


class SyncTransmitter(BaseModel):
    """Identifies the FM broadcast station used as the sync reference."""

    station_id: str
    """Human-readable ID, e.g. 'KISW_99.9'."""
    frequency_hz: float
    latitude_deg: float
    """FCC-documented transmitter latitude."""
    longitude_deg: float
    """FCC-documented transmitter longitude."""


class CarrierEvent(BaseModel):
    """
    A detected LMR carrier edge event, with timing relative to an FM sync event.

    Each event represents either a carrier onset (rising edge) or offset
    (falling edge). The server must pair like event_types across nodes --
    onset-with-onset and offset-with-offset - to compute valid TDOA.

    The server uses ``sync_to_snippet_start_ns`` plus the server-side knee
    position within the IQ snippet to compute sync -> knee time per node,
    then differences across nodes to compute TDOA (with sync-path-geometry
    correction).

    Schema version '1.5': renamed ``sync_delta_ns`` -> ``sync_to_snippet_start_ns``.
    The timing reference in the stream is now the first sample of the shipped
    IQ snippet (a stable sample boundary) instead of the transient detection
    point.  Detection point is still carried via ``transition_start`` /
    ``transition_end`` as the knee-search hint.

    Schema version '1.6': added ``"plateau"`` event_type for periodic
    snippets emitted while a carrier is sustained.  A plateau event is
    anchored to a sync-pilot bit boundary (sync_to_snippet_start_ns ≈ 0)
    so independent nodes' plateau snippets cover the same physical time
    window.  Plateau events flow through the same TDOA pipeline as
    onset/offset; they just give the server many more pair-samples per
    transmission for averaging.
    """

    schema_version: str = "1.6"

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Node-local unique identifier. Stable across amendment POSTs."""

    node_id: str
    node_location: NodeLocation

    # Target signal
    channel_frequency_hz: float
    """Nominal center frequency of the LMR channel (Hz)."""

    # THE primary TDOA measurement -----------------------------------------
    sync_to_snippet_start_ns: int
    """
    Time difference: first sample of the shipped IQ snippet minus the
    preceding sync event, measured on the same local sample clock
    (nanoseconds).  Node-side precision target: < 2 µs.

    Server computes the full sync-to-knee time as
        sync_to_knee_ns = sync_to_snippet_start_ns
                        + knee_position_in_snippet * 1e9 / corrected_rate
    where ``knee_position_in_snippet`` is found by the server's knee-finder
    using ``transition_start``/``transition_end`` as a search hint.
    Cross-node TDOA = (sync_to_knee_A - sync_to_knee_B) + sync-path correction.
    """
    sync_transmitter: SyncTransmitter
    sdr_mode: Literal["freq_hop", "two_sdr", "single_sdr", "rspduo"]
    pps_anchored: bool = False
    """True if two_sdr mode aligned streams via GPS 1PPS injection."""
    event_type: Literal["onset", "offset", "plateau"] = "onset"
    """
    Which carrier edge or sustained-carrier capture triggered this measurement.
    'onset'   - rising edge (carrier appeared).
    'offset'  - falling edge (carrier disappeared).
    'plateau' - periodic snapshot during a sustained carrier, anchored to a
                sync-pilot bit boundary so independent nodes cover the same
                physical time window.
    The server must pair onset-with-onset, offset-with-offset, and plateau-
    with-plateau across nodes.
    """

    # Rough absolute time (event association only) -------------------------
    onset_time_ns: int
    """
    GPS-disciplined wall-clock nanoseconds at the carrier edge.
    Accuracy: +/-1-10 us (kernel scheduling, not suitable for TDOA).
    Used by server to correlate events from different nodes.
    """

    # Carrier end (may arrive in a subsequent amendment POST) --------------
    offset_time_ns: int | None = None
    duration_ms: float | None = None

    # Signal quality -------------------------------------------------------
    peak_power_db: float = 0.0
    mean_power_db: float = 0.0
    noise_floor_db: float = 0.0
    snr_db: float = 0.0
    sync_corr_peak: float = 0.0
    """FM pilot cross-correlation peak at the paired SyncEvent (0-1)."""

    node_software_version: str = ""

    iq_snippet_b64: str
    """
    Base64-encoded int8-interleaved IQ snippet captured at the carrier edge.
    Used by the aggregation server to cross-correlate matched event pairs for
    usec-level TDOA.

    Encoding: bytes are interleaved real/imag int8, so for N complex samples
    the field decodes to 2N bytes.  Scale is arbitrary (normalised to +/-127);
    only relative timing is used by the cross-correlator.
    """

    transition_start: int = 0
    """Sample index within the snippet where the PA transition zone begins."""

    transition_end: int = 0
    """Sample index within the snippet where the PA transition zone ends."""

    # Sync event diagnostics — sent so the server can verify all nodes
    # are using the same RDS bit boundary for a given transmission event.
    sync_pilot_phase_rad: float = 0.0
    """Pilot phase (radians) at the matched SyncEvent."""

    sync_sample_index: float = 0.0
    """Absolute sample index of the matched SyncEvent (sync-decimated space)."""

    sync_delta_samples: float = 0.0
    """Raw sample difference (snippet_start_sample - sync_sample) before ns conversion."""

    sync_sample_rate_correction: float = 1.0
    """Crystal calibration factor applied to sample rate for ns conversion."""

    channel_sample_rate_hz: float
    """
    Sample rate of the IQ snippet in Hz.

    Equal to the target channel decimated rate: sdr_rate_hz / target_decimation.
    The server cross-correlator uses this to convert the correlation peak lag
    (in samples) to nanoseconds.
    """

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (for HTTP POST body)."""
        return self.model_dump(mode="json")
