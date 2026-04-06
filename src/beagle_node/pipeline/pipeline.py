# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
NodePipeline - wires all signal processing stages together.

Data flow (freq_hop / same_sdr mode)
-------------------------------------

  SDRReceiver (IQ buffers at SDR rate, e.g. 2.048 MSPS)
      |
      |-> sync_decimator  (-> ~256 kHz)
      |       +-> FMDemodulator
      |               +-> RDSSyncDetector  ----------> SyncEvent
      |                                                    |
      +-> target_decimator (-> ~48 kHz)                    |
              +-> CarrierDetector                          |
                      +-> CarrierOnset  --> DeltaComputer -> TDOAMeasurement
                                                            |
                                                     on_measurement(m)

two_sdr mode adds a PPSDetector on the raw SDR stream before decimation.

The pipeline is intentionally synchronous (no threads).  The caller drives
it by calling process_buffer() for each buffer from the SDR receiver, and
provides an on_measurement callback to handle completed measurements.

For freq_hop mode the single SDR alternates frequencies; the caller is
responsible for passing sync-channel buffers to process_buffer(role='sync')
and target-channel buffers to process_buffer(role='target').
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from beagle_node.pipeline.carrier_detect import CarrierDetector, CarrierOnset, CarrierOffset
from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.delta import DeltaComputer, TDOAMeasurement
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.pps_detector import PPSDetector
from beagle_node.pipeline.rds_sync_detector import RDSSyncDetector

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable parameters for NodePipeline."""
    # SDR input
    sdr_rate_hz: float = 2_048_000.0

    # Sync channel (FM broadcast)
    sync_decimation: int = 8            # 2.048 MHz -> 256 kHz
    sync_cutoff_hz: float = 128_000.0
    sync_mode: str = "rds"              # sync detector type

    # Target channel (LMR)
    target_decimation: int = 32         # 2.048 MHz -> 64 kHz
    target_cutoff_hz: float = 25_000.0
    carrier_onset_db: float = -30.0
    carrier_offset_db: float = -40.0
    carrier_window_samples: int = 64
    # Require this many consecutive above-threshold windows before onset is
    # declared.  min_hold=1 (default) matches old behaviour; min_hold=4 means
    # the carrier must be present for >=4 * window_samples before triggering,
    # which strongly suppresses single-window noise spikes.
    carrier_min_hold_windows: int = 1
    # Require this many consecutive below-threshold windows before offset is
    # declared.  Default 1 preserves existing behaviour.  Set to 4-8 for
    # real-world signals to prevent chattering when power briefly dips below
    # the offset threshold mid-transmission.
    carrier_min_release_windows: int = 1
    carrier_snippet_samples: int = 640
    carrier_snippet_post_windows: int = 5     # Collect 5 windows after onset/offset detection.
    # _encode_combined() places the transition at the snippet midpoint,
    # independent of min_hold_windows.  This ensures consistent snippet
    # anchoring across nodes with different carrier detector settings,
    # which is essential for mixed-hardware xcorr TDOA.
    # Ring buffer depth for offset lookback.  The ring must be deep enough to
    # contain the PA shutoff at the moment offset is detected.  Detection fires
    # min_release_windows after the signal first crosses offset_db; if the fade
    # is gradual the shutoff may be many windows back.  Setting this well above
    # snippet_samples / window_samples guarantees the shutoff is captured for
    # any realistic fade.  Default: 3 x snippet windows (at 62.5 kHz with
    # snippet=1280 and window=64: 3 x 20 = 60 windows = ~61 ms lookback).
    # None -> use 3x the snippet window count (same as the default).
    carrier_ring_lookback_windows: int | None = None
    # Minimum above-threshold windows since prime_state() before a CarrierOffset
    # is allowed.  0 = disabled (default, backward-compatible).  Set to 4+ in
    # freq_hop mode to suppress carrier-tail offsets anchored to the block boundary.
    carrier_min_active_windows_for_offset: int = 0

    # Delta computer
    max_sync_age_samples: int = 20_480  # ~80 ms at 256 kHz (8x sync period)
    min_corr_peak: float = 0.1

    # PPS (two_sdr mode only)
    pps_spike_threshold_db: float = 10.0
    pps_window_samples: int = 32


class NodePipeline:
    """
    End-to-end node signal processing pipeline.

    Parameters
    ----------
    config : PipelineConfig
    on_measurement : callable
        Called with each TDOAMeasurement as it is produced.
    pps_anchored : bool
        Set True for two_sdr mode after GPS 1PPS has aligned both streams.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        on_measurement: Callable[[TDOAMeasurement], None] | None = None,
        pps_anchored: bool = False,
    ) -> None:
        self._cfg = config or PipelineConfig()
        self._on_measurement = on_measurement or (lambda m: None)

        c = self._cfg

        # Sync chain
        self._sync_dec = Decimator(c.sync_decimation, c.sdr_rate_hz, c.sync_cutoff_hz)
        self._sync_demod = FMDemodulator(c.sdr_rate_hz / c.sync_decimation)
        if c.sync_mode == "rds":
            self._sync_det = RDSSyncDetector(
                sample_rate_hz=c.sdr_rate_hz / c.sync_decimation,
            )
        else:
            raise ValueError(f"Unknown sync_mode: {c.sync_mode!r}")

        # Target chain
        self._target_dec = Decimator(c.target_decimation, c.sdr_rate_hz, c.target_cutoff_hz)
        self._carrier_det = CarrierDetector(
            sample_rate_hz=c.sdr_rate_hz / c.target_decimation,
            onset_threshold_db=c.carrier_onset_db,
            offset_threshold_db=c.carrier_offset_db,
            window_samples=c.carrier_window_samples,
            min_hold_windows=c.carrier_min_hold_windows,
            min_release_windows=c.carrier_min_release_windows,
            snippet_samples=c.carrier_snippet_samples,
            snippet_post_windows=c.carrier_snippet_post_windows,
            ring_lookback_windows=c.carrier_ring_lookback_windows,
            min_active_windows_for_offset=c.carrier_min_active_windows_for_offset,
        )

        # Delta computer
        self._delta = DeltaComputer(
            sample_rate_hz=c.sdr_rate_hz / c.sync_decimation,
            max_sync_age_samples=c.max_sync_age_samples,
            pps_anchored=pps_anchored,
            min_corr_peak=c.min_corr_peak,
        )

        # PPS detector (only used in two_sdr mode)
        self._pps_det = PPSDetector(
            sample_rate_hz=c.sdr_rate_hz,
            spike_threshold_db=c.pps_spike_threshold_db,
            window_samples=c.pps_window_samples,
        )

        # Cumulative sample counters (separate per role)
        self._sync_sample_count: int = 0
        self._target_sample_count: int = 0
        self.sync_event_count: int = 0   # total SyncEvents detected

    @property
    def carrier_detector(self) -> CarrierDetector:
        """Access the live carrier detector (for health reporting and threshold updates)."""
        return self._carrier_det

    # ------------------------------------------------------------------
    # Buffer processing
    # ------------------------------------------------------------------

    def process_sync_buffer(
        self, iq, raw_start_sample: int | None = None, time_ns: int = 0
    ) -> list[TDOAMeasurement]:
        """
        Process one buffer from the sync (FM) channel.

        Parameters
        ----------
        iq : array-like, complex64
            Raw IQ at sdr_rate_hz.
        raw_start_sample : int | None
            Absolute raw-sample index of iq[0] in the continuous ADC stream.
            If None, uses the internal running counter (correct for single_sdr
            and two_sdr modes where every buffer is processed in sequence).
            Pass explicitly in freq_hop mode so the block offset is correct.
        time_ns : int
            Rough wall-clock time of the first sample (for event association).

        Returns
        -------
        list[TDOAMeasurement]
            Any measurements produced by onsets that were waiting for sync.
        """
        import numpy as np
        iq = np.asarray(iq, dtype=np.complex64)
        raw_start = self._sync_sample_count if raw_start_sample is None else raw_start_sample

        if len(iq) == 0:
            self._sync_sample_count = raw_start
            return []

        iq_dec = self._sync_dec.process(iq)
        dec_start = raw_start // self._cfg.sync_decimation

        audio = self._sync_demod.process(iq_dec)
        sync_events = self._sync_det.process(audio, start_sample=dec_start, time_ns=time_ns)
        self.sync_event_count += len(sync_events)
        for se in sync_events:
            logger.debug("SyncEvent sample=%d corr=%.3f", se.sample_index, se.corr_peak)
            self._delta.feed_sync(se)

        self._sync_sample_count = raw_start + len(iq)
        return []   # measurements arrive via process_target_buffer / on_measurement

    def process_target_buffer(
        self, iq, raw_start_sample: int | None = None, time_ns: int = 0,
        new_target_block: bool = False,
    ) -> list[TDOAMeasurement]:
        """
        Process one buffer from the target (LMR) channel.

        Parameters
        ----------
        raw_start_sample : int | None
            Absolute raw-sample index of iq[0].  Must be set correctly in
            freq_hop mode (the block starts at a different ADC offset than
            the sync block).  If None, uses the internal running counter.
        new_target_block : bool
            Set True for freq_hop mode at the start of each new target block
            (after a sync block).  This calls ``CarrierDetector.prime_state()``
            on the decimated IQ before detection runs, so the detector's state
            matches the actual carrier state at block start without emitting a
            spurious onset or offset for a transition that occurred while the
            SDR was on the sync channel.

        Returns
        -------
        list[TDOAMeasurement]
            Any new measurements produced.
        """
        import numpy as np
        iq = np.asarray(iq, dtype=np.complex64)
        raw_start = self._target_sample_count if raw_start_sample is None else raw_start_sample

        if len(iq) == 0:
            self._target_sample_count = raw_start
            return []

        # Remove DC offset before decimation.  RTL-SDR (and other direct-conversion
        # SDRs) have a strong LO leakage component at 0 Hz that would otherwise
        # dominate the narrowband power measurement after the 32* LPF+decimate step,
        # keeping the carrier detector permanently triggered.  Subtracting the
        # per-block mean is safe: FM-modulated carriers produce a block mean close
        # to zero over >100 ms windows, so the real signal is unaffected.

        iq = iq - np.mean(iq)

        # Prime the decimation filter with a replica of the first usable
        # samples.  This eliminates the power ramp caused by stale filter
        # history from the previous target block (~200 ms ago) or settling
        # data that may have different carrier state than the usable block.
        if new_target_block:
            self._target_dec.prime_with_replica(iq)

        iq_dec = self._target_dec.process(iq)
        dec_start = raw_start // self._cfg.target_decimation

        if new_target_block:
            self._carrier_det.prime_state(iq_dec)

        carrier_events = self._carrier_det.process(iq_dec, start_sample=dec_start)

        measurements: list[TDOAMeasurement] = []

        for event in carrier_events:
            if isinstance(event, (CarrierOnset, CarrierOffset)):
                # Convert target-dec sample index -> sync-dec sample index via raw:
                #   raw_sample  = event.sample_index * target_decimation
                #   sync_sample = raw_sample         // sync_decimation
                event_in_sync_space = (
                    event.sample_index * self._cfg.target_decimation
                    // self._cfg.sync_decimation
                )
                if isinstance(event, CarrierOnset):
                    mapped = CarrierOnset(
                        sample_index=event_in_sync_space,
                        power_db=event.power_db,
                        noise_floor_db=event.noise_floor_db,
                        iq_snippet=event.iq_snippet,
                    )
                    new = self._delta.feed_onset(mapped)
                else:
                    mapped = CarrierOffset(
                        sample_index=event_in_sync_space,
                        power_db=event.power_db,
                        iq_snippet=event.iq_snippet,
                    )
                    new = self._delta.feed_offset(mapped)
                for m in new:
                    self._on_measurement(m)
                measurements.extend(new)

        self._target_sample_count = raw_start + len(iq)
        return measurements

    # ------------------------------------------------------------------
    # PPS (two_sdr mode)
    # ------------------------------------------------------------------

    def process_pps_buffer(self, iq, start_sample: int = 0) -> list:
        """
        Scan a raw IQ buffer for GPS 1PPS spikes (two_sdr mode).

        Returns list of PPSAnchor events.
        """
        import numpy as np
        iq = np.asarray(iq, dtype=np.complex64)
        return self._pps_det.process(iq, start_sample=start_sample)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all pipeline state."""
        self._sync_dec.reset()
        self._sync_demod.reset()
        self._sync_det.reset()
        self._target_dec.reset()
        self._carrier_det.reset()
        self._delta.reset()
        self._pps_det.reset()
        self._sync_sample_count = 0
        self._target_sample_count = 0
