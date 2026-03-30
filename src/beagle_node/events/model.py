# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
CarrierEvent - the primary data unit produced by a Beagle node.

One CarrierEvent is produced per carrier onset detection. The `sync_delta_ns`
field is the precise TDOA measurement. The `onset_time_ns` is a rough absolute
timestamp used by the aggregation server only for event association across nodes.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NodeLocation(BaseModel):
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    uncertainty_m: float = 5.0


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

    The server uses `sync_delta_ns` (together with the sync transmitter
    location) to compute TDOA between nodes. It uses `onset_time_ns` only
    to match events from different nodes that heard the same transmission.

    Schema version '1.1': added event_type field; fixed sdr_mode values.
    """

    schema_version: str = "1.4"

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Node-local unique identifier. Stable across amendment POSTs."""

    node_id: str
    node_location: NodeLocation

    # Target signal
    channel_frequency_hz: float
    """Nominal center frequency of the LMR channel (Hz)."""

    # THE primary TDOA measurement -----------------------------------------
    sync_delta_ns: int
    """
    Time difference: carrier edge minus the preceding sync event,
    measured on the same local sample clock (nanoseconds).

    The aggregation server computes TDOA between nodes as:
        TDOA_AB = sync_delta_A - sync_delta_B
    then applies a path-delay correction using sync_transmitter coordinates.
    The server MUST pair events of the same event_type across nodes.
    """
    sync_transmitter: SyncTransmitter
    sdr_mode: Literal["freq_hop", "two_sdr", "single_sdr", "rspduo"]
    pps_anchored: bool = False
    """True if two_sdr mode aligned streams via GPS 1PPS injection."""
    event_type: Literal["onset", "offset"] = "onset"
    """
    Which carrier edge triggered this measurement.
    'onset'  - rising edge (carrier appeared).
    'offset' - falling edge (carrier disappeared).
    The server must pair onset-with-onset and offset-with-offset across nodes.
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

    # Clock metadata -------------------------------------------------------
    clock_source: Literal["gps_1pps", "ntp", "unknown"] = "unknown"
    clock_uncertainty_ns: int = 0
    """Chrony-reported RMS offset uncertainty (nanoseconds)."""

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
