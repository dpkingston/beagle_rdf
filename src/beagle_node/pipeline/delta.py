# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
DeltaComputer - converts sample indices into ``sync_to_snippet_start_ns``
measurements ready to ship to the server.

This is the measurement core.  All TDOA precision depends on what happens here.

Measurement model
-----------------
  sync_to_snippet_start_ns = (snippet_start_sample - sync_sample) * 1e9 / sample_rate_hz

Where:
  - snippet_start_sample = absolute stream sample index of the FIRST sample of
                           the shipped IQ snippet (CarrierOnset/Offset.sample_index).
                           This is the stable timing reference: the node's
                           detection point is used only to decide *when* to
                           package up a snippet; the time-accounted reference
                           is the snippet's first sample (on a sample boundary).
  - sync_sample          = SyncEvent.sample_index (sub-sample precision from
                           the pilot-phase sync detector)
  - sample_rate_hz       = nominal rate, corrected by CrystalCalibrator

The sign convention is: positive ``sync_to_snippet_start_ns`` means the snippet
begins *after* the most recent sync event (typical).

Triggering events
-----------------
Both carrier onset (rising edge) and carrier offset (falling edge) can
produce measurements via feed_onset() and feed_offset() respectively.
TDOAMeasurement.event_type records which edge triggered each measurement.

Using both edges doubles the measurement rate per transmission and allows
a measurement even when a target block begins mid-transmission (the offset
fires at carrier drop even if no onset was seen in that block).

The server must pair onset-with-onset and offset-with-offset across nodes;
mixing edge types would produce meaningless TDOA values.

Operating modes
---------------
freq_hop / same_sdr
    Both signals share one sample clock.  sample_rate_hz is the working
    rate after decimation (same for both channels).  No PPS anchoring.

two_sdr
    Two separate SDRs, each anchored to the GPS 1PPS in the sample domain.
    After PPS alignment, both streams are treated as a single shared clock.
    pps_anchored=True is set on the resulting CarrierEvent.

Sync matching
-------------
For each carrier event, DeltaComputer finds the best SyncEvent within
max_sync_age_samples.  The selection prefers **block-A bit-0** anchors
(the start of each RDS group) when an RDS block decoder lookup is
available, falling back to the most recent SyncEvent before the carrier
event when no block-A anchor is in range.

Block-A bit-0 anchoring matters cross-node: every receiver of the same
FM station sees the same broadcast bits, so when both nodes pick the
nearest block-A-bit-0 they end up locked to the same physical RDS
group boundary.  This converts a fixed-but-arbitrary per-pair offset
(which the server must calibrate out) into a shared zero reference.
"""

from __future__ import annotations

import json as _json
import logging
import os
from dataclasses import dataclass
from typing import Union

from typing import Callable, Optional

from beagle_node.pipeline.carrier_detect import CarrierOnset, CarrierOffset, CarrierPlateau
from beagle_node.pipeline.rds_decoder import BlockContext
from beagle_node.pipeline.sync_detector import SyncEvent

# Lookup signature: given a (sub-sample-precision) MPX sample index,
# return BlockContext or None.  Passed as a Callable rather than a direct
# RDSDecoderService dependency so DeltaComputer is testable with simple
# mock lookups (and so that nothing on the carrier-event hot path
# materializes the whole decoder).
BlockContextLookup = Callable[[float], Optional[BlockContext]]

logger = logging.getLogger(__name__)

_TIMING_DIAG = os.environ.get("BEAGLE_TIMING_DIAG") == "1"

# Either edge of a carrier transition can produce a measurement.
_CarrierEvent = Union[CarrierOnset, CarrierOffset, CarrierPlateau]


@dataclass(frozen=True)
class TDOAMeasurement:
    """
    One TDOA measurement ready for reporting.

    ``sync_to_snippet_start_ns`` is the primary timing value sent to the
    server: nanoseconds from the matched sync event to the first sample of
    the shipped IQ snippet.

    event_type indicates which carrier edge triggered this measurement:
      "onset"  - rising edge (carrier appeared)
      "offset" - falling edge (carrier disappeared)
    The server must pair like event_types across nodes.
    """
    sync_to_snippet_start_ns: int       # THE measurement (sync -> first snippet sample, ns)
    snippet_start_sample: int           # Absolute stream sample of snippet[0]
    sync_sample: float                  # SyncEvent sample index used (sub-sample precision)
    sample_rate_hz: float               # Corrected sample rate
    sample_rate_correction: float       # CrystalCalibrator factor
    pps_anchored: bool                  # True if GPS 1PPS was used
    corr_peak: float                    # SyncEvent correlation quality
    onset_power_db: float               # power at the triggering event
    noise_floor_db: float               # EMA of idle-state power before the event
    event_type: str                     # "onset" or "offset"
    iq_snippet: bytes = b""             # int8-interleaved IQ for server knee-finder
    transition_start: int = 0           # Knee-search hint: zone start, samples into snippet
    transition_end: int = 0             # Knee-search hint: zone end, samples into snippet
    # Sync event diagnostics for server-side verification
    sync_pilot_phase_rad: float = 0.0   # pilot_phase_rad from the matched SyncEvent
    sync_sample_index: float = 0.0      # absolute sample index of the matched SyncEvent
    sync_delta_samples: float = 0.0     # raw sample delta (snippet_start - sync_sample)
    # RDS block anchor context (Commit 4): when an RDS block context was
    # available for the matched SyncEvent, these record which group
    # boundary the measurement is anchored to.  ``anchor_block_letter``
    # is "A" with ``anchor_bit_in_block == 0`` for the preferred
    # block-A bit-0 anchor; otherwise None for legacy nearest matches.
    anchor_block_letter: str | None = None
    anchor_bit_in_block: int | None = None
    anchor_group_pi: int | None = None
    anchor_group_type: str | None = None


class DeltaComputer:
    """
    Matches CarrierOnset events to SyncEvents and computes sync_to_snippet_start_ns.

    Parameters
    ----------
    sample_rate_hz : float
        Nominal sample rate of the target IQ stream.
    max_sync_age_samples : int
        Maximum age (in samples) of a SyncEvent relative to a CarrierOnset.
        Onsets older than this without a matching sync are dropped.
        Default: 3 * typical sync period (30 ms at 256 kHz = 7680 samples).
    pps_anchored : bool
        Set to True in two_sdr mode after GPS 1PPS alignment is confirmed.
    min_corr_peak : float
        Minimum SyncEvent.corr_peak to use for measurement.  Events below
        this threshold are discarded (pilot too weak).
    """

    def __init__(
        self,
        sample_rate_hz: float,
        max_sync_age_samples: int = 7_680,
        pps_anchored: bool = False,
        min_corr_peak: float = 0.1,
        block_context_lookup: BlockContextLookup | None = None,
    ) -> None:
        self._rate = float(sample_rate_hz)
        self._max_age = int(max_sync_age_samples)
        self._pps_anchored = bool(pps_anchored)
        self._min_corr = float(min_corr_peak)
        # When provided, _match() prefers SyncEvents that land on the
        # first bit of an RDS block A (the natural group boundary).
        # When None, falls back to the legacy "most recent sync before
        # carrier event" behavior.
        self._block_lookup = block_context_lookup
        # Telemetry: counts of anchor-selection outcomes per match.
        self._anchor_chose_block_a: int = 0    # matches that found a block-A anchor
        self._anchor_fallback_legacy: int = 0  # matches that used legacy nearest

        # Recent sync events (kept until too old)
        self._sync_events: list[SyncEvent] = []
        # Pending carrier events (onset or offset) waiting for a sync match.
        # Each entry is (event, event_type) where event_type is "onset"|"offset".
        self._pending_events: list[tuple[_CarrierEvent, str]] = []

        # Low-quality sync tracking: log a warning when the pilot is
        # consistently weak so the user knows why the node went silent.
        #
        # Hysteresis prevents log spam when quality oscillates near the threshold:
        #   _WARN_AFTER  - warn only after this many consecutive bad events
        #   _RECOVER_AFTER - recover only after this many consecutive good events
        # At a 7 ms sync period, 5 events = 35 ms of sustained degradation before
        # the first warning, and 35 ms of sustained recovery before the info log.
        self._rejected_sync_count: int = 0   # consecutive bad events
        self._consecutive_good: int = 0       # consecutive good events (for recovery)
        self._pilot_warned: bool = False      # True once we have issued a warning
        self._WARN_AFTER: int = 5
        self._RECOVER_AFTER: int = 5
        self._WARN_EVERY: int = 500   # repeat warning roughly every few seconds

    # ------------------------------------------------------------------
    # Feed events
    # ------------------------------------------------------------------

    def feed_sync(self, event: SyncEvent) -> None:
        """Record a new SyncEvent."""
        if event.corr_peak < self._min_corr:
            self._consecutive_good = 0
            self._rejected_sync_count += 1
            if self._rejected_sync_count == self._WARN_AFTER:
                self._pilot_warned = True
                logger.warning(
                    "FM pilot quality below threshold: corr_peak %.3f < %.3f "
                    "(sync events will be dropped until pilot recovers)",
                    event.corr_peak, self._min_corr,
                )
            elif self._pilot_warned and self._rejected_sync_count % self._WARN_EVERY == 0:
                logger.warning(
                    "FM pilot still weak: %d consecutive sync events rejected "
                    "(corr_peak %.3f < %.3f) - no measurements possible",
                    self._rejected_sync_count, event.corr_peak, self._min_corr,
                )
            else:
                logger.debug("Dropping sync event: corr_peak %.3f < %.3f",
                             event.corr_peak, self._min_corr)
            return
        # Good event
        self._consecutive_good += 1
        if self._pilot_warned:
            if self._consecutive_good >= self._RECOVER_AFTER:
                logger.info(
                    "FM pilot recovered after %d rejected events (corr_peak %.3f)",
                    self._rejected_sync_count, event.corr_peak,
                )
                self._rejected_sync_count = 0
                self._consecutive_good = 0
                self._pilot_warned = False
        else:
            # Never warned - silently reset the bad count so brief dips don't
            # accumulate toward the warn threshold across unrelated good stretches.
            self._rejected_sync_count = 0
            self._consecutive_good = 0
        self._sync_events.append(event)
        # Prune sync events that are too old to match any future carrier event.
        # This must happen here (not only in _flush) because _flush is only called
        # by feed_onset/feed_offset.  During quiet periods with no carrier activity,
        # sync events accumulate at ~100 Hz indefinitely without this pruning.
        # A sync can only match a carrier event with sample_index >= sync.sample_index,
        # so future carrier events (arriving after this sync) need syncs no older
        # than max_sync_age_samples behind them.  Pruning to the current event's
        # sample_index - max_age is safe: any carrier event at or after this sync
        # can use syncs >= its own sample_index - max_age >= this cutoff.
        cutoff = event.sample_index - self._max_age
        self._sync_events = [s for s in self._sync_events if s.sample_index >= cutoff]

    def feed_onset(self, onset: CarrierOnset) -> list[TDOAMeasurement]:
        """
        Record a CarrierOnset (rising edge) and attempt to match it to a SyncEvent.

        Returns a list of TDOAMeasurement (0 or 1 element).
        """
        self._pending_events.append((onset, "onset"))
        return self._flush()

    def feed_offset(self, offset: CarrierOffset) -> list[TDOAMeasurement]:
        """
        Record a CarrierOffset (falling edge) and attempt to match it to a SyncEvent.

        Returns a list of TDOAMeasurement (0 or 1 element).
        The server must pair offset measurements with offset measurements from
        other nodes (not with onset measurements).
        """
        self._pending_events.append((offset, "offset"))
        return self._flush()

    def feed_plateau(self, plateau) -> list[TDOAMeasurement]:
        """
        Record a CarrierPlateau (periodic capture during sustained carrier)
        and attempt to match it to a SyncEvent for sub-microsecond timing.

        Returns a list of TDOAMeasurement (0 or 1 element).
        Plateau measurements pair across nodes only with other plateau
        measurements that fall within the same wall-clock window.
        """
        self._pending_events.append((plateau, "plateau"))
        return self._flush()

    def _flush(self) -> list[TDOAMeasurement]:
        """Try to resolve all pending carrier events."""
        resolved: list[TDOAMeasurement] = []
        still_pending: list[tuple[_CarrierEvent, str]] = []

        # Frontier sample: how far the stream has advanced.  Used to age out
        # pending events when no sync is available.
        #
        # The original code used only newest_sync, which fails when sync_events
        # is empty (returns 0 -> age condition 0-event > max_age is never true
        # for positive sample indices -> pending list grows forever).
        #
        # Fix: also consider the newest *carrier* sample.  If sync is dead but
        # carriers keep arriving, the carrier frontier advances and old pending
        # events age out correctly.  If both are absent, nothing new is added to
        # pending so _flush() is never called anyway.
        newest_sync = self._sync_events[-1].sample_index if self._sync_events else 0
        newest_carrier = (
            max(e.sample_index for e, _ in self._pending_events)
            if self._pending_events else 0
        )
        frontier = max(newest_sync, newest_carrier)

        for event, etype in self._pending_events:
            result = self._match(event, etype)
            if result is not None:
                resolved.append(result)
            else:
                # Check if the event has aged out
                if frontier - event.sample_index > self._max_age:
                    logger.warning(
                        "Dropping %s at sample %d: no sync within %d samples",
                        etype, event.sample_index, self._max_age,
                    )
                else:
                    still_pending.append((event, etype))

        self._pending_events = still_pending

        # Prune old sync events.  Always run (not only when pending_events is
        # non-empty) so that a carrier event during recovery after a quiet
        # period also trims stale syncs.  Anchor the cutoff to the oldest
        # still-pending carrier event when one exists so its match candidates
        # are preserved; otherwise use the current frontier.
        if self._pending_events:
            oldest_pending = min(e.sample_index for e, _ in self._pending_events)
            cutoff = oldest_pending - self._max_age
        else:
            cutoff = frontier - self._max_age
        self._sync_events = [s for s in self._sync_events if s.sample_index >= cutoff]

        return resolved

    def _match(self, event: _CarrierEvent, event_type: str) -> TDOAMeasurement | None:
        """
        Find the best SyncEvent for this carrier event.

        Selection (in priority order):
          1. **Block-A bit-0** anchor: if a block context lookup is configured
             and any SyncEvent within max_sync_age_samples is the first bit
             of an RDS group's block A (block_letter="A", bit_in_block=0),
             choose the one closest in time to the carrier event.
          2. **Legacy nearest**: most recent SyncEvent with
             sample_index <= event.sample_index and within max_sync_age_samples.

        Preference (1) is enabled whenever ``block_context_lookup`` was
        passed to __init__; it requires the RDS decoder to have actually
        decoded the group containing the candidate sync event.  During the
        decoder's warmup phase (or under poor signal conditions) no block
        context is available and we fall through to (2) — which keeps the
        legacy behavior unchanged, so this is a strict improvement.
        """
        in_range = [
            s for s in self._sync_events
            if abs(s.sample_index - event.sample_index) <= self._max_age
        ]
        if not in_range:
            logger.debug(
                "No sync event within %d samples of %s at %d "
                "(newest sync: %.1f, total syncs: %d)",
                self._max_age,
                event_type,
                event.sample_index,
                self._sync_events[-1].sample_index if self._sync_events else -1,
                len(self._sync_events),
            )
            return None

        # Tier 1: prefer block-A bit-0 anchors when the decoder lookup
        # has identified any in the search window.
        anchor_letter: str | None = None
        anchor_bit: int | None = None
        anchor_group_pi: int | None = None
        anchor_group_type: str | None = None
        best: SyncEvent | None = None

        if self._block_lookup is not None:
            a_anchors: list[tuple[SyncEvent, BlockContext]] = []
            for s in in_range:
                ctx = self._block_lookup(s.sample_index)
                if ctx is None:
                    continue
                if ctx.block_letter == "A" and ctx.bit_in_block == 0:
                    a_anchors.append((s, ctx))
            if a_anchors:
                best, best_ctx = min(
                    a_anchors,
                    key=lambda sc: abs(sc[0].sample_index - event.sample_index),
                )
                anchor_letter = best_ctx.block_letter
                anchor_bit = best_ctx.bit_in_block
                anchor_group_pi = best_ctx.group_pi
                anchor_group_type = best_ctx.group_type
                self._anchor_chose_block_a += 1
                logger.debug(
                    "block-A anchor at sample %.1f for %s at %d "
                    "(pi=0x%04X type=%s)",
                    best.sample_index, event_type, event.sample_index,
                    anchor_group_pi or 0, anchor_group_type or "?",
                )

        # Tier 2: legacy "most recent sync before event" fallback.
        if best is None:
            pre_event = [s for s in in_range if s.sample_index <= event.sample_index]
            if not pre_event:
                logger.debug(
                    "No pre-event sync within %d samples of %s at %d",
                    self._max_age, event_type, event.sample_index,
                )
                return None
            best = max(pre_event, key=lambda s: s.sample_index)
            self._anchor_fallback_legacy += 1

        # Apply crystal calibration to the sample rate
        corrected_rate = self._rate * best.sample_rate_correction

        # event.sample_index is the absolute stream sample index of the
        # snippet's FIRST sample (carrier_detect encodes it this way).
        delta_samples = event.sample_index - best.sample_index
        sync_to_snippet_start_ns = int(round(delta_samples * 1_000_000_000.0 / corrected_rate))

        noise_floor = getattr(event, "noise_floor_db", -100.0)

        if _TIMING_DIAG:
            # Plateau events fire at the configured cadence (default 1/s)
            # while a carrier is sustained.  In production this is the
            # dominant journal source on resource-constrained hosts; demote
            # to DEBUG so it can be enabled selectively.  Onset/offset stay
            # at INFO because they're rare (per-key-down) and high-signal.
            _level = logging.DEBUG if event_type == "plateau" else logging.INFO
            logger.log(
                _level,
                "TIMING_DIAG %s",
                _json.dumps({
                    "stage": "delta",
                    "event_type": event_type,
                    "snippet_start_sample": event.sample_index,
                    "sync_sample_float": round(best.sample_index, 3),
                    "delta_samples": round(delta_samples, 3),
                    "corrected_rate_hz": round(corrected_rate, 3),
                    "sample_rate_correction": round(best.sample_rate_correction, 8),
                    "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
                    "corr_peak": round(best.corr_peak, 4),
                    "n_sync_candidates": len(candidates),
                }),
            )

        return TDOAMeasurement(
            sync_to_snippet_start_ns=sync_to_snippet_start_ns,
            snippet_start_sample=event.sample_index,
            sync_sample=best.sample_index,
            sample_rate_hz=corrected_rate,
            sample_rate_correction=best.sample_rate_correction,
            pps_anchored=self._pps_anchored,
            corr_peak=best.corr_peak,
            onset_power_db=event.power_db,
            noise_floor_db=noise_floor,
            event_type=event_type,
            iq_snippet=event.iq_snippet,
            transition_start=getattr(event, 'transition_start', 0),
            transition_end=getattr(event, 'transition_end', 0),
            sync_pilot_phase_rad=best.pilot_phase_rad,
            sync_sample_index=best.sample_index,
            sync_delta_samples=delta_samples,
            anchor_block_letter=anchor_letter,
            anchor_bit_in_block=anchor_bit,
            anchor_group_pi=anchor_group_pi,
            anchor_group_type=anchor_group_type,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all buffered events."""
        self._sync_events.clear()
        self._pending_events.clear()
        self._rejected_sync_count = 0
        self._consecutive_good = 0
        self._pilot_warned = False
