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
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from beagle_node.pipeline.carrier_detect import (
    CarrierDetector, CarrierOnset, CarrierOffset, CarrierPlateau,
)
from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.delta import DeltaComputer, TDOAMeasurement
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.pps_detector import PPSDetector
from beagle_node.pipeline.rds_sync_detector import RDSSyncDetector

logger = logging.getLogger(__name__)

# Set BEAGLE_CAPTURE_SYNC_AUDIO=/path/to/file.npz to capture the first N
# seconds of demodulated FM audio (the input to RDSSyncDetector.process()).
# The capture includes start_sample indices so it can be replayed exactly.
# Use BEAGLE_CAPTURE_SYNC_SECONDS to set duration (default 30).
_CAPTURE_PATH = os.environ.get("BEAGLE_CAPTURE_SYNC_AUDIO")
_CAPTURE_SECONDS = float(os.environ.get("BEAGLE_CAPTURE_SYNC_SECONDS", "30"))


@dataclass
class PipelineConfig:
    """Tunable parameters for NodePipeline."""
    # SDR input
    sdr_rate_hz: float = 2_048_000.0

    # Sync channel (FM broadcast)
    sync_decimation: int = 8            # 2.048 MHz -> 256 kHz
    sync_cutoff_hz: float = 128_000.0
    sync_mode: str = "rds"              # sync detector type

    # Target channel (LMR).  Decimation 8 gives 250 kHz at 2.0 MHz SDR
    # (or 256 kHz at 2.048 MHz), matching the sync-channel rate.  The
    # higher sample rate improves per-event knee-timing precision: at
    # 62.5 kHz the PA transition resolves to ~200 µs per-event std;
    # scaling simulations predict ~50 µs at 250 kHz.
    target_decimation: int = 8          # 2.0 MHz -> 250 kHz
    target_cutoff_hz: float = 25_000.0
    carrier_onset_db: float = -30.0
    carrier_offset_db: float = -40.0
    # Detector window: 256 samples = ~1 ms at 250 kHz (same duration as
    # the previous 64 samples at 62.5 kHz).  Detection thresholds are
    # in the same dB range; timing semantics preserved.
    carrier_window_samples: int = 256
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
    # Snippet size: 16384 samples = ~65.5 ms at 250 kHz.  Sized to include
    # ~30 ms of post-knee plateau on onsets so the server's coherent
    # complex-IQ cross-correlation has real modulation bandwidth to lock
    # onto (CTCSS tones, audio, data); the pure-carrier portion right at
    # the knee carries no coherent timing information.  On-wire payload
    # ~33 KB raw, ~44 KB base64-encoded per event.
    carrier_snippet_samples: int = 16384
    carrier_snippet_post_windows: int = 140   # Enough to centre detection
                                              # when snippet_samples=16384
                                              # and window_samples=64.
    # _encode_combined() places the transition at the snippet midpoint,
    # independent of min_hold_windows.  This ensures consistent snippet
    # anchoring across nodes with different carrier detector settings,
    # which is essential for mixed-hardware xcorr TDOA.
    # Ring buffer depth for offset lookback.  The ring must be deep enough to
    # contain the PA shutoff at the moment offset is detected.  Detection fires
    # min_release_windows after the signal first crosses offset_db; if the fade
    # is gradual the shutoff may be many windows back.  Setting this well above
    # snippet_samples / window_samples guarantees the shutoff is captured for
    # any realistic fade.  Default: 3 x snippet windows (at 250 kHz with
    # snippet=16384 and window=64: 3 x 256 = 768 windows = ~197 ms lookback).
    # None -> use 3x the snippet window count (same as the default).
    carrier_ring_lookback_windows: int | None = None
    # Minimum above-threshold windows since prime_state() before a CarrierOffset
    # is allowed.  0 = disabled (default, backward-compatible).  Set to 4+ in
    # freq_hop mode to suppress carrier-tail offsets anchored to the block boundary.
    carrier_min_active_windows_for_offset: int = 0
    # Plateau-event interval (seconds).  When > 0, the carrier detector emits
    # a CarrierPlateau every N seconds while the carrier is sustained, giving
    # the server many additional pair-TDOA samples per transmission for
    # averaging.  0 (default) = disabled.  Recommended: 1.0-2.0 s for typical
    # PTT-radio transmissions of 5-30 s duration.
    carrier_plateau_event_interval_s: float = 0.0
    # Maximum plateau emissions allowed in a single active period before the
    # emitter mutes itself until the next idle->active transition.  Safety net
    # for "stuck active" failures; 0 = no cap (legacy).  See carrier.
    # plateau_max_per_active in the node config schema for full discussion.
    carrier_plateau_max_per_active: int = 30

    # Auto-threshold tracking (matches GUI "Auto-Calibrate" button, applied
    # continuously so thresholds follow changing noise conditions without
    # operator intervention).  When enabled, static carrier_onset_db /
    # carrier_offset_db are used only during noise-floor warmup.
    carrier_auto_threshold_margins: bool = True
    carrier_onset_margin_db: float = 12.0
    carrier_offset_margin_db: float = 6.0
    carrier_auto_threshold_update_interval_s: float = 2.0

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
            auto_threshold_margins=c.carrier_auto_threshold_margins,
            onset_margin_db=c.carrier_onset_margin_db,
            offset_margin_db=c.carrier_offset_margin_db,
            auto_threshold_update_interval_s=c.carrier_auto_threshold_update_interval_s,
            plateau_event_interval_s=c.carrier_plateau_event_interval_s,
            plateau_max_per_active=c.carrier_plateau_max_per_active,
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

        # Latest sync detector telemetry (updated each time process_sync_buffer
        # produces an event).  Exposed for health reporting.
        self._latest_corr_peak: float = 0.0
        self._latest_sample_rate_correction: float = 1.0

        # Sync audio capture for test fixture generation
        self._capture_audio: list[tuple[int, 'np.ndarray']] | None = None
        self._capture_samples_remaining: int = 0
        if _CAPTURE_PATH:
            sync_rate = c.sdr_rate_hz / c.sync_decimation
            self._capture_audio = []
            self._capture_samples_remaining = int(sync_rate * _CAPTURE_SECONDS)
            logger.info(
                "Sync audio capture enabled: %s (%.0f s, %d samples at %.0f Hz)",
                _CAPTURE_PATH, _CAPTURE_SECONDS,
                self._capture_samples_remaining, sync_rate,
            )

    @property
    def carrier_detector(self) -> CarrierDetector:
        """Access the live carrier detector (for health reporting and threshold updates)."""
        return self._carrier_det

    @property
    def latest_corr_peak(self) -> float:
        """Most recent SyncEvent.corr_peak (signal quality, 0-1)."""
        return self._latest_corr_peak

    @property
    def latest_sample_rate_correction(self) -> float:
        """Most recent SyncEvent.sample_rate_correction (crystal calibration factor)."""
        return self._latest_sample_rate_correction

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

        # Capture demodulated FM audio for test fixture generation
        if self._capture_audio is not None and self._capture_samples_remaining > 0:
            import numpy as np
            n_take = min(len(audio), self._capture_samples_remaining)
            self._capture_audio.append((dec_start, audio[:n_take].copy()))
            self._capture_samples_remaining -= n_take
            if self._capture_samples_remaining <= 0:
                self._save_capture()

        sync_events = self._sync_det.process(audio, start_sample=dec_start, time_ns=time_ns)
        self.sync_event_count += len(sync_events)
        for se in sync_events:
            logger.debug("SyncEvent sample=%d corr=%.3f", se.sample_index, se.corr_peak)
            self._delta.feed_sync(se)
        if sync_events:
            last_se = sync_events[-1]
            self._latest_corr_peak = last_se.corr_peak
            self._latest_sample_rate_correction = last_se.sample_rate_correction

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
            if isinstance(event, (CarrierOnset, CarrierOffset, CarrierPlateau)):
                # Convert target-dec sample index -> sync-dec sample index via raw:
                #   raw_sample  = event.sample_index * target_decimation
                #   sync_sample = raw_sample         // sync_decimation
                # Integer division is exact for detection points (integer window
                # boundaries).  The server's xcorr finds the sub-sample knee.
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
                        transition_start=event.transition_start,
                        transition_end=event.transition_end,
                    )
                    new = self._delta.feed_onset(mapped)
                elif isinstance(event, CarrierOffset):
                    mapped = CarrierOffset(
                        sample_index=event_in_sync_space,
                        power_db=event.power_db,
                        iq_snippet=event.iq_snippet,
                        transition_start=event.transition_start,
                        transition_end=event.transition_end,
                    )
                    new = self._delta.feed_offset(mapped)
                else:  # CarrierPlateau
                    mapped = CarrierPlateau(
                        sample_index=event_in_sync_space,
                        power_db=event.power_db,
                        iq_snippet=event.iq_snippet,
                        transition_start=event.transition_start,
                        transition_end=event.transition_end,
                    )
                    new = self._delta.feed_plateau(mapped)
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
    # Sync audio capture
    # ------------------------------------------------------------------

    def _save_capture(self) -> None:
        """Save captured sync audio to disk as a .npz file."""
        import numpy as np
        assert self._capture_audio is not None
        starts = np.array([s for s, _ in self._capture_audio], dtype=np.int64)
        audio = np.concatenate([a for _, a in self._capture_audio])
        sync_rate = self._cfg.sdr_rate_hz / self._cfg.sync_decimation
        np.savez_compressed(
            _CAPTURE_PATH,
            audio=audio,
            start_samples=starts,
            sample_rate_hz=sync_rate,
        )
        logger.info(
            "Sync audio capture saved: %s (%d samples, %.1f s)",
            _CAPTURE_PATH, len(audio), len(audio) / sync_rate,
        )
        self._capture_audio = None  # disable further capture

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def mark_discontinuity(self) -> None:
        """Signal that samples were lost (overflow, drain, stream restart).

        Discards stale pipeline state and forces the sync detector to
        re-lock.  Small gaps in detection readiness are acceptable —
        target events are infrequent and sync events arrive every ~842 µs.
        """
        logger.warning("Pipeline discontinuity: resetting sync and cancelling pending carrier events")
        self._sync_det.reset()
        self._carrier_det.cancel_pending()
        self._delta.reset()

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
