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

Sync matching (Commit 6 — fail-closed block-A anchor)
-----------------------------------------------------
For each carrier event, DeltaComputer searches a window of
**±half an RDS group period** (≈ ±44 ms) around the carrier event's
sample index for a SyncEvent that the block-context lookup identifies
as the first bit (bit 0) of an RDS block A.

A complete RDS group spans 104 bits ≈ 87.6 ms; bit-0 of block A occurs
exactly once per group.  So a ±44 ms window contains **at most one**
broadcast A-bit-0.  When both receivers have decoded that group, both
independently pick the same physical anchor (deterministic agreement).
When either node has lost the group to a BLER gap, that node emits
**nothing** — there is no fallback.  The server-side pair-matcher will
drop the unpaired measurement on the other node, preserving cross-pair
consistency at the cost of an occasional dropped event.

This fail-closed design is appropriate because:

  - CarrierPlateau events fire periodically during sustained carriers,
    so a single sustained transmission yields multiple measurement
    opportunities.  Losing one to a BLER gap rarely loses the whole
    transmission.
  - The legacy "nearest sync before event" fallback is independently
    bad cross-node: each node has its own sub-bit alignment offset, so
    even when both fall back to legacy they pick anchors that differ
    by tens to hundreds of microseconds.  Emitting nothing is no worse
    than emitting an unaligned legacy measurement.

A block_context_lookup callable is required.  Without it, _match
returns None for every event.
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

# Anchor-lookup signature: given a carrier event's sample index and a
# lookback window, return the BlockContext of the most recent block-A
# bit-0 that occurred at or before the carrier event within that window
# (or None if the decoder has no such anchor).
#
# This replaces the older 1-arg lookup signature.  The matcher uses the
# returned context's group_anchor_sample only as a rough indicator to
# find the SyncEvent closest to it; that SyncEvent then becomes the
# actual TDOA anchor.  This decouples the matcher from the sync-event-
# vs-block-sample alignment offset between the pilot and demodulator
# timing paths, which the previous ±½ bit-width tolerance failed to
# absorb on production hardware.
BlockAAnchorLookup = Callable[[float, float], Optional[BlockContext]]

# Kept for backward compat with any older callers / tests that supplied
# a per-sample lookup.  No longer used by _match.
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
        block_a_anchor_lookup: BlockAAnchorLookup | None = None,
        # Legacy 1-arg lookup, ignored — present for backward compat with
        # tests that still pass it.
        block_context_lookup: BlockContextLookup | None = None,
    ) -> None:
        self._rate = float(sample_rate_hz)
        # RDS group period in samples (≈ 87.6 ms at 1187.5 / 104 bits/group)
        # and bit period in samples (≈ 842 µs).
        self._group_period_samples = self._rate / (1187.5 / 104.0)
        self._bit_period_samples = self._rate / 1187.5
        # max_sync_age_samples bounds how long a pending carrier event can
        # wait for a match before it ages out.  Used as-is for the SyncEvent
        # pre-filter / aging cutoff.
        self._max_age = int(max_sync_age_samples)
        # Anchor-lookup lookback: how far back to search for the most recent
        # block-A bit-0.  One group period is always enough (the next one
        # back is two group periods away, which we never want).
        self._anchor_lookback_samples = int(round(self._group_period_samples))
        # Closest-SyncEvent-to-Block-A tolerance.  SyncEvents are at exactly
        # bit-rate intervals, so the closest SyncEvent to any sample is at
        # most ½ bit_period away.  We accept up to a full bit_period to
        # leave margin for sync-buffer jitter / transient gaps.
        self._sync_to_anchor_tolerance_samples = int(round(self._bit_period_samples))
        self._pps_anchored = bool(pps_anchored)
        self._min_corr = float(min_corr_peak)
        # block_a_anchor_lookup is REQUIRED for the fail-closed matcher to
        # emit anything.  When None, every _match() call returns None and
        # no measurements are produced.  See module docstring "Sync matching".
        self._anchor_lookup = block_a_anchor_lookup
        # Legacy lookup parameter — accepted but not used.  Kept so that
        # existing test setups that pass it don't break.
        _ = block_context_lookup
        # Telemetry: outcomes of anchor selection per _match() invocation.
        self._anchor_chose_block_a: int = 0     # block-A anchor picked → measurement emitted
        self._anchor_no_lookup_dropped: int = 0  # no anchor lookup configured → dropped
        self._anchor_no_a_in_window_dropped: int = 0  # no block-A in lookback → dropped
        self._anchor_no_sync_near_a_dropped: int = 0  # no SyncEvent close enough to anchor → dropped

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
        #
        # Cutoff must be the older of:
        #   (a) event.sample_index - max_age — any future carrier event needs
        #       syncs no older than max_age before its own sample_index
        #   (b) oldest_pending.sample_index - max_age — the symmetric window
        #       used by _match means a *pending* carrier event still needs
        #       syncs UP TO max_age *after* its sample_index too; conversely
        #       it needs syncs at its sample_index - max_age and later
        #
        # Pre-Commit-6 this only used (a) because the legacy matcher accepted
        # only pre-event syncs.  The Commit 6 ±half-group symmetric search
        # window means pending events still need post-event syncs that haven't
        # arrived yet — so we must NOT prune syncs that fall in a pending
        # event's search window just because they're older than the latest
        # sync's max_age cutoff.
        if self._pending_events:
            oldest_pending = min(e.sample_index for e, _ in self._pending_events)
            cutoff = min(oldest_pending, event.sample_index) - self._max_age
        else:
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
                # _match logged the specific reason it failed.  Here we just
                # decide whether to keep the event pending (waiting for more
                # syncs / a future re-decode) or drop it for being too old.
                if frontier - event.sample_index > self._max_age:
                    logger.debug(
                        "Aging out %s at sample %d (frontier %d, max_age %d) "
                        "after unsuccessful match",
                        etype, event.sample_index, frontier, self._max_age,
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
        Find the SyncEvent that anchors this carrier event to a block-A bit-0.

        Algorithm (Commit 8):

          1. Ask the decoder: "what's the most recent block-A bit-0 you
             decoded at or before this carrier sample, within one group
             period back?"  → returns the demodulator-derived
             ``group_anchor_sample`` (in MPX coords), or None.
          2. If None, fail-closed (no measurement emitted).
          3. Otherwise, find the SyncEvent in our buffer whose
             ``sample_index`` is closest to ``group_anchor_sample``.
             That SyncEvent's sample_index is the actual TDOA anchor
             (pilot path, sub-µs precision).
          4. Sanity-check: closest SyncEvent must be within one bit
             period of the anchor; otherwise the SyncEvent stream has a
             gap and we drop the event rather than anchor to a wrong bit.

        Why this works cross-node: both nodes' decoders identify the
        same physical broadcast block A.  Both nodes' pilot-derived
        SyncEvent streams have a SyncEvent at the corresponding wall-
        clock instant.  The "closest SyncEvent to block A" operation
        deterministically picks the same physical bit on both nodes,
        even when the demodulator vs pilot timing paths have different
        group-delay offsets.

        Returns the TDOAMeasurement, or None for any of:
          - no anchor lookup configured (sync mode not RDS)
          - no block-A bit-0 decoded in the lookback (BLER gap)
          - no SyncEvent within one bit period of the anchor (sync gap)
        """
        if self._anchor_lookup is None:
            self._anchor_no_lookup_dropped += 1
            logger.debug(
                "No block-A anchor lookup configured; dropping %s at %d",
                event_type, event.sample_index,
            )
            return None

        # Step 1: ask decoder for the most recent block-A bit-0 in lookback range.
        best_ctx = self._anchor_lookup(
            float(event.sample_index), float(self._anchor_lookback_samples)
        )
        if best_ctx is None:
            self._anchor_no_a_in_window_dropped += 1
            logger.warning(
                "Dropping %s at sample %d: no decoded block-A bit-0 within "
                "%d samples (≈1 group period) before carrier — RDS BLER gap "
                "or decoder warmup",
                event_type, event.sample_index, self._anchor_lookback_samples,
            )
            return None

        # Step 2: find the SyncEvent closest to the block-A anchor's sample
        # position.  This SyncEvent becomes the actual TDOA reference.
        if not self._sync_events:
            self._anchor_no_sync_near_a_dropped += 1
            logger.warning(
                "Dropping %s at sample %d: no SyncEvents in buffer (sync "
                "detector still warming up or producing no events)",
                event_type, event.sample_index,
            )
            return None

        anchor_sample_demod = best_ctx.group_anchor_sample
        best = min(
            self._sync_events,
            key=lambda s: abs(s.sample_index - anchor_sample_demod),
        )
        sync_to_anchor_distance = abs(best.sample_index - anchor_sample_demod)

        # Step 3: sanity-check the alignment.  SyncEvents come at bit-rate
        # intervals, so the closest SyncEvent to any sample should be within
        # ½ bit period.  We tolerate up to one full bit period to absorb
        # transient sync-stream gaps.
        if sync_to_anchor_distance > self._sync_to_anchor_tolerance_samples:
            self._anchor_no_sync_near_a_dropped += 1
            logger.warning(
                "Dropping %s at sample %d: closest SyncEvent (sample %.1f) "
                "is %d samples from decoded block-A bit-0 (sample %.1f) — "
                "tolerance %d.  Sync stream gap?",
                event_type, event.sample_index,
                best.sample_index, int(sync_to_anchor_distance),
                anchor_sample_demod, self._sync_to_anchor_tolerance_samples,
            )
            return None

        anchor_letter = best_ctx.block_letter
        anchor_bit = best_ctx.bit_in_block
        anchor_group_pi = best_ctx.group_pi
        anchor_group_type = best_ctx.group_type
        self._anchor_chose_block_a += 1
        logger.debug(
            "block-A anchor: SyncEvent sample %.1f (closest to decoded "
            "anchor at %.1f, %d samples away) for %s at %d "
            "(pi=0x%04X type=%s)",
            best.sample_index, anchor_sample_demod,
            int(sync_to_anchor_distance),
            event_type, event.sample_index,
            anchor_group_pi or 0, anchor_group_type or "?",
        )

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
