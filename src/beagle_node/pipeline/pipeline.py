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
      |               |                                    |
      |               +-> RDSDecoderService (visibility — Group records via
      |                                       rolling-window re-decode, used
      |                                       by DeltaComputer for block-A
      |                                       anchor selection)
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
import math
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
from beagle_node.pipeline.rds_decoder import RDSDecoderService
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
    # Burst-then-slow plateau emission.  When both knobs are > 0, the emitter
    # uses ``carrier_plateau_event_interval_s`` for the first ``carrier_plateau_burst_count``
    # emissions in an active period and then transitions to the slower
    # ``carrier_plateau_slow_interval_s`` cadence for the remainder.  When
    # either is 0, single-cadence (legacy) behaviour is preserved.  See
    # carrier.plateau_burst_count / carrier.plateau_slow_interval_s in the
    # node config schema.
    carrier_plateau_burst_count: int = 0
    carrier_plateau_slow_interval_s: float = 0.0

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

    # RDS block decoder service (visibility only in Commit 3; consumed by
    # DeltaComputer in Commit 4).  Window must be ≥ 0.5 s for the decoder
    # to lock; longer = more CPU per decode but better noise immunity.
    rds_decoder_window_seconds: float = 2.0
    rds_decoder_interval_ms: float = 1000.0   # min wallclock between decodes

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
            # RDS block decoder service: runs in parallel with the pilot-
            # derived sync detector to recover RDS group structure (block
            # letters, PI, group types).  Used for visibility / telemetry
            # now; will be consumed by DeltaComputer in a follow-up commit
            # for block-A anchor selection.
            self._rds_decoder: RDSDecoderService | None = RDSDecoderService(
                fs_in=c.sdr_rate_hz / c.sync_decimation,
                window_seconds=c.rds_decoder_window_seconds,
                decode_interval_ms=c.rds_decoder_interval_ms,
                use_fec=True,
            )
        else:
            raise ValueError(f"Unknown sync_mode: {c.sync_mode!r}")

        # Target chain
        self._target_dec = Decimator(c.target_decimation, c.sdr_rate_hz, c.target_cutoff_hz)

        # IQ-ring depth.  With an RDS decoder the anchor-triggered plateau
        # emitter (see ``_maybe_emit_anchor_plateau``) may pick a block-A
        # bit-0 anchor that is up to ~one decode-interval stale — the
        # decoder refreshes ``_latest_groups`` only every
        # ``rds_decoder_interval_ms`` over a ``rds_decoder_window_seconds``
        # rolling buffer.  For that older anchor's snippet to still be
        # extractable, the ring must hold at least the decode window plus a
        # margin (Step 2 of the plateau cross-node-sync fix).  Without the
        # RDS decoder the legacy snippet-sized auto-ring is fine.
        #
        # Pre-Step-2 the ring auto-sized to ~3× snippet (~196 ms), which is
        # why production showed skip_no_anchor=81% but skip_try_emit=0: the
        # 2-group lookback only ever returned anchors already inside that
        # small ring.  Widening the lookback (below) without enlarging the
        # ring would convert those misses into skip_try_emit; we enlarge the
        # ring here so the wider lookback's older anchors stay emittable.
        _ring_windows = c.carrier_ring_lookback_windows
        if self._rds_decoder is not None:
            _target_rate = c.sdr_rate_hz / c.target_decimation
            _ring_seconds_needed = c.rds_decoder_window_seconds + 1.0
            _needed_windows = math.ceil(
                _ring_seconds_needed * _target_rate / c.carrier_window_samples
            )
            _ring_windows = max(int(_ring_windows or 0), _needed_windows)

        self._carrier_det = CarrierDetector(
            sample_rate_hz=c.sdr_rate_hz / c.target_decimation,
            onset_threshold_db=c.carrier_onset_db,
            offset_threshold_db=c.carrier_offset_db,
            window_samples=c.carrier_window_samples,
            min_hold_windows=c.carrier_min_hold_windows,
            min_release_windows=c.carrier_min_release_windows,
            snippet_samples=c.carrier_snippet_samples,
            snippet_post_windows=c.carrier_snippet_post_windows,
            ring_lookback_windows=_ring_windows,
            min_active_windows_for_offset=c.carrier_min_active_windows_for_offset,
            auto_threshold_margins=c.carrier_auto_threshold_margins,
            onset_margin_db=c.carrier_onset_margin_db,
            offset_margin_db=c.carrier_offset_margin_db,
            auto_threshold_update_interval_s=c.carrier_auto_threshold_update_interval_s,
            plateau_event_interval_s=c.carrier_plateau_event_interval_s,
            plateau_max_per_active=c.carrier_plateau_max_per_active,
            plateau_burst_count=c.carrier_plateau_burst_count,
            plateau_slow_interval_s=c.carrier_plateau_slow_interval_s,
            # When an RDS decoder is configured, plateau emission is
            # driven by the pipeline's anchor-triggered scheduler (see
            # ``_maybe_emit_anchor_plateau``).  Disable the legacy
            # wall-clock-driven path inside carrier_detect so we don't
            # double-emit (or emit at the wrong, group-misaligned
            # position).
            enable_legacy_plateau_emission=(self._rds_decoder is None),
        )

        # Delta computer.  When the RDS decoder service is active, pass
        # its ``find_a_bit0_anchor`` callable so DeltaComputer can match
        # carrier events to block-A bit-0 SyncEvents.  See delta.py
        # ``BlockAAnchorLookup`` docstring.
        anchor_lookup = (
            self._rds_decoder.find_a_bit0_anchor
            if self._rds_decoder is not None
            else None
        )
        self._delta = DeltaComputer(
            sample_rate_hz=c.sdr_rate_hz / c.sync_decimation,
            max_sync_age_samples=c.max_sync_age_samples,
            pps_anchored=pps_anchored,
            min_corr_peak=c.min_corr_peak,
            block_a_anchor_lookup=anchor_lookup,
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

        # Anchor-triggered plateau scheduler state.  Plateau emission no
        # longer fires on a wall-clock interval grid inside carrier_detect;
        # the pipeline schedules emissions at RDS block-A bit-0 anchors
        # so paired nodes' snippets cover the same physical time window
        # (sync_to_snippet_start_ns ≈ 0).  See ``try_emit_plateau_at`` in
        # carrier_detect.py.
        #
        # ``_last_plateau_target_anchor`` is the target-domain sample of
        # the most recent plateau we emitted in the current active
        # period.  Cleared back to None on idle→active so the first
        # plateau of each active period fires at the soonest in-ring
        # anchor.
        self._last_plateau_target_anchor: int | None = None
        self._prev_carrier_state: str = "idle"
        # Group period at the target rate (RDS group = 104 bits / 1187.5 Hz).
        target_rate = c.sdr_rate_hz / c.target_decimation
        self._target_rate_hz: float = target_rate
        self._plateau_group_period_target_samples: int = max(
            1, round(target_rate / (1187.5 / 104.0)),
        )
        # Group period in nanoseconds, used for cross-node phase-locked
        # plateau emission (see _maybe_emit_anchor_plateau).
        self._group_period_ns: int = int(round(1e9 * 104.0 / 1187.5))  # 87_578_947 ns

        # Cross-node plateau phase-lock state.  ``_buf_anchor_wall_ns`` is
        # ``time.time_ns()`` captured at the start of the most recent
        # ``process_target_buffer`` call, and ``_buf_anchor_target_sample``
        # is the target-domain sample index corresponding to it.  Used to
        # convert the demod-derived block-A bit-0 anchor sample to a
        # wall-clock time so we can compute a globally-shared
        # ``global_group_epoch = floor(anchor_wall_ns / GROUP_PERIOD_NS)``.
        # All NTP-synced nodes observing the same broadcast group will
        # compute the same epoch integer (NTP error << 88 ms group period),
        # so emitting only when ``epoch % K == 0`` makes all nodes fire
        # plateaus on the SAME RDS group.  Snippets across nodes then
        # cover the same physical time window, and the server pairs them.
        self._buf_anchor_wall_ns: int | None = None
        self._buf_anchor_target_sample: int = 0
        # Most recent emitted global epoch (de-dupes within a single active
        # period).  Reset on idle→active.
        self._last_emitted_global_epoch: int | None = None

        # Per-attempt plateau-emit telemetry (Step 1 of the plateau
        # cross-node-sync fix).  Surfaced via rds_health_snapshot so the
        # aliasing between the 1 s RDS decode and the ~196 ms IQ ring is
        # directly observable in production.  Cumulative for the process
        # lifetime (like the anchor_* counters), NOT reset on idle→active.
        #   attempts             - calls that got past the disabled/idle/K=0
        #                          gates (i.e. a real chance to emit)
        #   ok                   - plateaus actually emitted
        #   skip_no_anchor       - find_a_bit0_anchor returned None (no
        #                          decoded block-A bit-0 within the lookback
        #                          of the snippet horizon — the decode-
        #                          staleness symptom)
        #   skip_not_kslot       - anchor epoch not a K-multiple
        #   skip_already_emitted - anchor epoch <= last emitted (same slot)
        #   skip_try_emit        - carrier_det.try_emit_plateau_at failed
        #                          (ring shortfall / edge clearance / cap —
        #                          the ring-too-small symptom)
        self._plateau_emit_counters: dict[str, int] = {
            "attempts": 0,
            "ok": 0,
            "skip_no_anchor": 0,
            "skip_not_kslot": 0,
            "skip_already_emitted": 0,
            "skip_try_emit": 0,
        }

        # K (number of groups between successive plateau emissions) is
        # derived at use time from the carrier_detect's current
        # ``_plateau_interval_s`` so live config reloads of
        # ``carrier.plateau_event_interval_s`` (via
        # ``CarrierDetector.update_thresholds``) take effect immediately
        # without needing a parallel pipeline-level update.  See
        # ``_plateau_K_groups``.

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

    @property
    def rds_decoder(self) -> RDSDecoderService | None:
        """The RDS block decoder service (visibility / future block-A anchor lookup)."""
        return self._rds_decoder

    @property
    def _plateau_K_groups(self) -> int:
        """Number of RDS groups between successive anchor-triggered plateau
        emissions (≥ 1).  0 means plateau emission is disabled.

        Derived dynamically from the carrier_detect's current
        ``_plateau_interval_s`` so live config reloads of
        ``carrier.plateau_event_interval_s`` (via
        ``CarrierDetector.update_thresholds``) take effect on the next
        ``process_target_buffer`` call without needing the pipeline to
        be reconstructed.
        """
        interval_s = self._carrier_det._plateau_interval_s
        if interval_s <= 0.0:
            return 0
        # 104.0 / 1187.5 = one RDS group period in seconds (~87.6 ms).
        return max(1, round(interval_s / (104.0 / 1187.5)))

    def rds_health_snapshot(self) -> dict | None:
        """
        Compact summary of RDS decoder + anchor-selection health for the
        ``/health`` endpoint and the server's heartbeat consumer.  Returns
        None when RDS sync mode is disabled.

        Keys:
          group_count            - most recent decode's group count
          group_period_hz        - 11.4 (constant; for the server's reference)
          bler_mean              - 0..1, mean over the rolling window
          decode_ms              - most recent decode CPU time
          anchor_emitted         - **per-event** count of measurements emitted
                                   with a block-A anchor
          anchor_aged_out        - **per-event** count of carrier events that
                                   never matched and aged out
          anchor_emit_fraction   - emitted / (emitted + aged_out) — the real
                                   per-event success rate.  None until any
                                   carrier event has been seen.
          anchor_match_attempts  - **per-attempt** sum of in-_match failures
                                   (no_lookup + no_a + no_sync_near_a). Useful
                                   for diagnosing matcher inefficiency; not a
                                   success metric.
        """
        if self._rds_decoder is None:
            return None
        stats = self._rds_decoder.stats
        emitted = self._delta._anchor_chose_block_a
        aged_out = self._delta._anchor_aged_out
        match_attempts = (
            self._delta._anchor_no_lookup_dropped
            + self._delta._anchor_no_a_in_window_dropped
            + self._delta._anchor_no_sync_near_a_dropped
        )
        total = emitted + aged_out
        emit_frac = (emitted / total) if total > 0 else None
        return {
            "group_count": stats.last_group_count,
            "group_period_hz": 1187.5 / 104.0,
            "bler_mean": (
                round(stats.last_bler_mean, 3)
                if stats.last_bler_mean == stats.last_bler_mean  # not NaN
                else None
            ),
            "decode_ms": round(stats.last_decode_duration_ms, 1),
            "anchor_emitted": emitted,
            "anchor_aged_out": aged_out,
            "anchor_emit_fraction": (
                round(emit_frac, 3) if emit_frac is not None else None
            ),
            "anchor_match_attempts_failed": match_attempts,
            # Per-attempt plateau-emit telemetry (cross-node-sync diagnostic).
            # A healthy emitter shows ``ok`` ≈ one per K-slot of active time
            # with small skip_no_anchor / skip_try_emit.  The aliasing bug
            # shows large skip_no_anchor (decode staleness) + skip_try_emit
            # (ring too small) relative to ok.
            "plateau_emit": dict(self._plateau_emit_counters),
        }

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

        # Feed the same FM-demodulated audio into the RDS block decoder
        # service.  It buffers internally and re-decodes on a configurable
        # interval; emits Group records consumed by DeltaComputer.lookup
        # for anchor selection.  Health-summary plumbing (Commit 7) surfaces
        # the per-decode stats to the server.
        if self._rds_decoder is not None:
            new_groups = self._rds_decoder.push_audio(audio, start_sample=dec_start)
            if new_groups:
                stats = self._rds_decoder.stats
                # DEBUG: detailed per-decode stats and per-group lines.  The
                # production health snapshot carries the rolling summary
                # numbers to the server; per-second console logs would just
                # be noise.
                logger.debug(
                    "RDS decode: %d groups in %.1f s window "
                    "(BLER mean %.2f, decode %.0f ms)",
                    stats.last_group_count,
                    stats.last_decode_input_seconds,
                    stats.last_bler_mean if stats.last_bler_mean == stats.last_bler_mean else -1,
                    stats.last_decode_duration_ms,
                )
                for g in new_groups[:10]:
                    if g.pi is not None:
                        logger.debug(
                            "RDS group: pi=%s type=%s bler=%.2f anchor_sample=%.1f",
                            f"0x{g.pi:04X}",
                            g.group_type or "?",
                            g.bler,
                            g.sample_index_first_bit,
                        )

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
        import time as _time
        iq = np.asarray(iq, dtype=np.complex64)
        raw_start = self._target_sample_count if raw_start_sample is None else raw_start_sample

        if len(iq) == 0:
            self._target_sample_count = raw_start
            return []

        # Record wall-clock anchor for sample→wall conversion in the
        # phase-locked plateau emitter.
        #
        # Strongly prefer the caller's ``time_ns`` (RSPduo HAS_TIME
        # provides the hardware timestamp of the buffer's first sample;
        # freq_hop provides ``time.time_ns()`` captured immediately
        # after ``read_bytes()`` returns).  Those are tied to the SDR
        # capture moment, NOT to pipeline-entry latency, which is what
        # cross-node phase-locking actually requires.
        #
        # Falling back to ``time.time_ns()`` here would re-introduce
        # per-node buffering / processing latency jitter into the
        # global-epoch derivation — observed in production 2026-05-31
        # as plateau emissions scattered across the second with no
        # cross-node clustering, even though each node's internal
        # cadence was correct.
        if time_ns:
            self._buf_anchor_wall_ns = time_ns
        else:
            self._buf_anchor_wall_ns = _time.time_ns()
        self._buf_anchor_target_sample = raw_start // self._cfg.target_decimation

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

        # Anchor-triggered plateau emission.  Replaces the legacy
        # wall-clock-driven emitter in carrier_detect (which was
        # producing snippets at arbitrary positions within the RDS group
        # cycle — independent per node — so paired snippets covered
        # different physical times and TDOAs were dominated by the
        # mis-alignment, not by signal arrival differences).
        plateau_ev = self._maybe_emit_anchor_plateau()
        if plateau_ev is not None:
            mapped = CarrierPlateau(
                sample_index=(
                    plateau_ev.sample_index * self._cfg.target_decimation
                    // self._cfg.sync_decimation
                ),
                power_db=plateau_ev.power_db,
                iq_snippet=plateau_ev.iq_snippet,
                transition_start=plateau_ev.transition_start,
                transition_end=plateau_ev.transition_end,
            )
            new = self._delta.feed_plateau(mapped)
            for m in new:
                self._on_measurement(m)
            measurements.extend(new)

        # Track state for idle→active reset of the plateau-anchor pointer.
        new_state = self._carrier_det.state
        if self._prev_carrier_state != "active" and new_state == "active":
            # Active period just started — clear the last-anchor pointer
            # AND the global-epoch tracker so the first plateau of this
            # period can fire at the soonest valid global epoch (rather
            # than wait K groups after the previous active period's last
            # anchor or epoch).
            self._last_plateau_target_anchor = None
            self._last_emitted_global_epoch = None
        self._prev_carrier_state = new_state

        self._target_sample_count = raw_start + len(iq)
        return measurements

    # ------------------------------------------------------------------
    # Anchor-triggered plateau scheduler
    # ------------------------------------------------------------------

    def _maybe_emit_anchor_plateau(self):
        """Return a CarrierPlateau on a GLOBAL K-slot epoch, or None.

        Step 3 of the plateau cross-node-sync fix.  Instead of reacting
        to "the most recent anchor near the horizon, if its epoch happens
        to be a K-multiple" (which is phase-dependent and unsynchronized
        across nodes — production showed ~90 % skip_not_kslot and zero
        cross-node coincidence), this **marches the global K-slot grid**:

          - enumerate every decoded block-A bit-0 anchor in the decode
            window (``block_a_bit0_anchors``);
          - compute each one's GLOBAL epoch
            ``round(anchor_wall_ns / GROUP_PERIOD_NS)`` from the SDR
            hardware timestamp, which is identical (sub-ms) on every node
            for the same broadcast group;
          - keep the K-multiple epochs that are not yet emitted and whose
            snippet is still in the ring;
          - emit *that epoch's* real decoded anchor.

        Because the epoch is a pure function of wall-clock and the
        hardware timestamps coincide across nodes, every active node
        targets the IDENTICAL global slots ``{…, 0, K, 2K, …}`` and so
        emits snippets covering the same physical window → the server
        pairs them.

        On idle→active (no ``_last_emitted_global_epoch``) we start at the
        most-recent available slot (fresh, low-latency); thereafter we
        emit the OLDEST unemitted in-ring slot, which marches the grid in
        order and auto-skips any slot whose audio has aged out.
        """
        # Disabled when there's no RDS decoder, when the interval knob is
        # 0, or when the detector is idle.
        if self._rds_decoder is None:
            return None
        K = self._plateau_K_groups
        if K == 0:
            return None
        if self._carrier_det.state != "active":
            return None

        target_now = self._carrier_det.cumulative_sample
        snippet_samples = self._carrier_det.snippet_samples
        # The latest legal anchor sample is one snippet behind the newest
        # sample we've processed — the snippet runs forward from the
        # anchor, so it must fit entirely within already-received data.
        latest_target_anchor = target_now - snippet_samples
        if latest_target_anchor < 0:
            return None

        # Past the disabled/idle/warmup gates: this is a real emit attempt.
        self._plateau_emit_counters["attempts"] += 1

        td = self._cfg.target_decimation
        sd = self._cfg.sync_decimation

        # Non-HAS_TIME callers / tests don't supply a hardware timestamp,
        # so the global-epoch grid can't be computed.  Fall back to the
        # legacy per-node "K groups since previous emission" scheduling.
        if self._buf_anchor_wall_ns is None:
            return self._emit_plateau_legacy_fallback(K, latest_target_anchor, td, sd)

        # Enumerate every decoded block-A bit-0 anchor in the window and
        # find the emittable global K-slots among them.
        sync_anchors = self._rds_decoder.block_a_bit0_anchors()
        if not sync_anchors:
            self._plateau_emit_counters["skip_no_anchor"] += 1
            return None

        last_emitted = self._last_emitted_global_epoch
        oldest_fresh: tuple[int, int] | None = None   # (epoch, target_anchor)
        newest_fresh: tuple[int, int] | None = None
        saw_fresh_kslot = False  # a K-slot > last_emitted existed but wasn't in-ring
        for s in sync_anchors:
            # ``math.ceil`` (not int) so target_anchor >= the sub-sample
            # float anchor; otherwise DeltaComputer's re-match regresses
            # one full RDS group (the +87.6 ms bug).  See git 5f8e310.
            ta = math.ceil(s) * sd // td
            if ta > latest_target_anchor:
                continue  # snippet would run past received data
            anchor_wall_ns = self._buf_anchor_wall_ns + int(
                (ta - self._buf_anchor_target_sample) * 1e9 / self._target_rate_hz
            )
            # ``round`` (not floor) → symmetric ±half-group (44 ms) margin
            # against cross-node clock skew; two nodes within 44 ms round
            # to the SAME epoch.
            epoch = int(round(anchor_wall_ns / self._group_period_ns))
            if epoch % K != 0:
                continue
            if last_emitted is not None and epoch <= last_emitted:
                continue
            saw_fresh_kslot = True
            if not self._carrier_det.plateau_snippet_available(ta):
                continue
            if oldest_fresh is None or epoch < oldest_fresh[0]:
                oldest_fresh = (epoch, ta)
            if newest_fresh is None or epoch > newest_fresh[0]:
                newest_fresh = (epoch, ta)

        if oldest_fresh is None:
            # No emittable fresh K-slot this call.
            if saw_fresh_kslot:
                # A fresh K-slot existed but its snippet had aged out of
                # the ring — a genuine ring-capacity miss.
                self._plateau_emit_counters["skip_try_emit"] += 1
            else:
                # All in-window K-slots already emitted: the normal
                # "waiting for the horizon to reveal the next slot" state.
                self._plateau_emit_counters["skip_already_emitted"] += 1
            return None

        # Fresh start → newest slot (low latency); catch-up → oldest
        # unemitted slot (marches the global grid in order).
        target_epoch, target_anchor = (
            newest_fresh if last_emitted is None else oldest_fresh
        )
        plateau = self._carrier_det.try_emit_plateau_at(target_anchor)
        if plateau is None:
            self._plateau_emit_counters["skip_try_emit"] += 1
            return None

        self._plateau_emit_counters["ok"] += 1
        self._last_plateau_target_anchor = target_anchor
        self._last_emitted_global_epoch = target_epoch
        return plateau

    def _emit_plateau_legacy_fallback(self, K, latest_target_anchor, td, sd):
        """Legacy plateau scheduling for callers without a hardware
        timestamp (single_sdr / mock / tests that don't pass ``time_ns``).

        Reacts to the most recent decoded block-A anchor and emits once
        per K-groups since the previous emission.  No cross-node global
        epoch (none is computable without a shared wall-clock), so this
        path does NOT provide cross-node sync — it only preserves the
        original behavior for non-RSPduo/test paths.
        """
        sync_rate_hz = self._cfg.sdr_rate_hz / sd
        sync_at_latest_anchor = latest_target_anchor * td // sd
        sync_lookback = int(self._cfg.rds_decoder_window_seconds * sync_rate_hz)
        best_ctx = self._rds_decoder.find_a_bit0_anchor(
            float(sync_at_latest_anchor), float(sync_lookback),
        )
        if best_ctx is None:
            self._plateau_emit_counters["skip_no_anchor"] += 1
            return None
        target_anchor = math.ceil(best_ctx.group_anchor_sample) * sd // td
        if (
            self._last_plateau_target_anchor is not None
            and target_anchor - self._last_plateau_target_anchor
                < K * self._plateau_group_period_target_samples
        ):
            self._plateau_emit_counters["skip_already_emitted"] += 1
            return None
        plateau = self._carrier_det.try_emit_plateau_at(target_anchor)
        if plateau is None:
            self._plateau_emit_counters["skip_try_emit"] += 1
            return None
        self._plateau_emit_counters["ok"] += 1
        self._last_plateau_target_anchor = target_anchor
        return plateau

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
        # Drop the RDS decoder's rolling buffer too — its 2-second window
        # would otherwise carry stale pre-discontinuity audio for the next
        # two seconds, producing garbled decodes during recovery.
        if self._rds_decoder is not None:
            self._rds_decoder.reset()

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
        # Reset anchor-plateau scheduler state too (including the
        # cross-node phase-locked global-epoch tracker).
        self._last_plateau_target_anchor = None
        self._last_emitted_global_epoch = None
        self._buf_anchor_wall_ns = None
        self._buf_anchor_target_sample = 0
        self._prev_carrier_state = "idle"
