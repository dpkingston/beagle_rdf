# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
DeltaComputer - converts raw sample indices into sync_delta_ns measurements.

This is the measurement core.  All TDOA precision depends on what happens here.

Measurement model
-----------------
  sync_delta_ns = (target_sample - sync_sample) * 1_000_000_000 / sample_rate_hz

Where:
  - target_sample  = CarrierOnset or CarrierOffset sample_index (in target IQ)
  - sync_sample    = SyncEvent.sample_index      (in the *sync* IQ stream)
  - sample_rate_hz = nominal rate, corrected by CrystalCalibrator

The sign convention is: positive sync_delta_ns means the carrier edge
appeared *after* the most recent sync event.

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
For each carrier event, DeltaComputer finds the most recent SyncEvent whose
sample_index <= event sample_index and is within max_sync_age_samples.
If no sync event is available within the search window, the event is dropped
and a warning is logged.
"""

from __future__ import annotations

import json as _json
import logging
import os
from dataclasses import dataclass
from typing import Union

from beagle_node.pipeline.carrier_detect import CarrierOnset, CarrierOffset
from beagle_node.pipeline.sync_detector import SyncEvent

logger = logging.getLogger(__name__)

_TIMING_DIAG = os.environ.get("BEAGLE_TIMING_DIAG") == "1"

# Either edge of a carrier transition can produce a measurement.
_CarrierEvent = Union[CarrierOnset, CarrierOffset]


@dataclass(frozen=True)
class TDOAMeasurement:
    """
    One TDOA measurement ready for reporting.

    sync_delta_ns is the primary value sent to the server.
    event_type indicates which carrier edge triggered this measurement:
      "onset"  - rising edge (carrier appeared)
      "offset" - falling edge (carrier disappeared)
    The server must pair like event_types across nodes.
    """
    sync_delta_ns: int                  # THE measurement
    target_sample: int                  # triggering event sample index
    sync_sample: float                  # SyncEvent sample index used (sub-sample precision)
    sample_rate_hz: float               # Corrected sample rate
    sample_rate_correction: float       # CrystalCalibrator factor
    pps_anchored: bool                  # True if GPS 1PPS was used
    corr_peak: float                    # SyncEvent correlation quality
    onset_power_db: float               # power at the triggering event
    noise_floor_db: float               # EMA of idle-state power before the event
    event_type: str                     # "onset" or "offset"
    iq_snippet: bytes = b""             # int8-interleaved IQ for server cross-correlation
    transition_start: int = 0           # Transition zone start within snippet (samples)
    transition_end: int = 0             # Transition zone end within snippet (samples)
    # Sync event diagnostics for server-side verification
    sync_pilot_phase_rad: float = 0.0   # pilot_phase_rad from the matched SyncEvent
    sync_sample_index: float = 0.0      # absolute sample index of the matched SyncEvent
    sync_delta_samples: float = 0.0     # raw sample delta before ns conversion


class DeltaComputer:
    """
    Matches CarrierOnset events to SyncEvents and computes sync_delta_ns.

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
    ) -> None:
        self._rate = float(sample_rate_hz)
        self._max_age = int(max_sync_age_samples)
        self._pps_anchored = bool(pps_anchored)
        self._min_corr = float(min_corr_peak)

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

        Strategy: use the most recent SyncEvent whose sample_index <= event.sample_index
        AND is within max_sync_age_samples of the event.
        """
        candidates = [
            s for s in self._sync_events
            if s.sample_index <= event.sample_index
            and event.sample_index - s.sample_index <= self._max_age
        ]
        if not candidates:
            logger.debug(
                "No sync event within %d samples of %s at %d "
                "(newest sync: %d, total syncs: %d)",
                self._max_age,
                event_type,
                event.sample_index,
                self._sync_events[-1].sample_index if self._sync_events else -1,
                len(self._sync_events),
            )
            return None

        # Most recent sync before the event within the age window
        best = max(candidates, key=lambda s: s.sample_index)

        # Apply crystal calibration to the sample rate
        corrected_rate = self._rate * best.sample_rate_correction

        delta_samples = event.sample_index - best.sample_index
        sync_delta_ns = int(round(delta_samples * 1_000_000_000.0 / corrected_rate))

        noise_floor = getattr(event, "noise_floor_db", -100.0)

        if _TIMING_DIAG:
            # Log in sync-decimated sample space (256 kHz).
            # event.sample_index is int (carrier side, mapped to sync space).
            # best.sample_index is float (M&M sub-sample in sync space).
            logger.info(
                "TIMING_DIAG %s",
                _json.dumps({
                    "stage": "delta",
                    "event_type": event_type,
                    "target_sample_sync": event.sample_index,
                    "sync_sample_float": round(best.sample_index, 3),
                    "delta_samples": round(delta_samples, 3),
                    "corrected_rate_hz": round(corrected_rate, 3),
                    "sample_rate_correction": round(best.sample_rate_correction, 8),
                    "sync_delta_ns": sync_delta_ns,
                    "corr_peak": round(best.corr_peak, 4),
                    "n_sync_candidates": len(candidates),
                }),
            )

        return TDOAMeasurement(
            sync_delta_ns=sync_delta_ns,
            target_sample=event.sample_index,
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
