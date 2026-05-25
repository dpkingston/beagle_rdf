# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Tests for DeltaComputer's RDS block-A anchor selection (Commit 4).

The default behavior of DeltaComputer is the legacy "most recent
SyncEvent before the carrier event" matcher.  When a block-context
lookup is provided, _match prefers SyncEvents that land on the first
bit of an RDS block A (the natural group boundary), falling back to
the legacy behavior when no block-A anchor is available in range.

This file covers:

  - Legacy behavior unchanged when no lookup is provided.
  - Lookup that always returns None → legacy behavior preserved.
  - Lookup that returns block-A bit-0 for one SyncEvent → that event
    is chosen even when it's not the most recent before the carrier.
  - Block-A anchor *after* the carrier event is acceptable (relaxes
    the pre-event constraint that legacy applies).
  - TDOAMeasurement.anchor_* fields are populated when the block-A
    path is used, and remain None when the legacy fallback is used.
  - Telemetry counters increment correctly.
"""

from __future__ import annotations

from typing import Optional

import pytest

from beagle_node.pipeline.carrier_detect import CarrierOnset
from beagle_node.pipeline.delta import DeltaComputer
from beagle_node.pipeline.rds_decoder import BlockContext
from beagle_node.pipeline.sync_detector import SyncEvent


def _sync(sample_index: float, corr: float = 1.0) -> SyncEvent:
    return SyncEvent(
        sample_index=sample_index,
        time_ns=0,
        corr_peak=corr,
        pilot_phase_rad=0.0,
        sample_rate_correction=1.0,
    )


def _onset(sample_index: int, power_db: float = 20.0) -> CarrierOnset:
    return CarrierOnset(
        sample_index=sample_index,
        power_db=power_db,
        noise_floor_db=-100.0,
        iq_snippet=b"",
        transition_start=0,
        transition_end=0,
    )


# Block context helpers
def _ctx_a0(anchor_sample: float, pi: int = 0x4652,
            group_type: str = "0A") -> BlockContext:
    """A block-A bit-0 BlockContext where ``group_anchor_sample`` is the
    demod-derived sample position of bit 0 (used by the matcher to find
    the closest SyncEvent)."""
    return BlockContext(
        block_letter="A",
        bit_in_block=0,
        group_pi=pi,
        group_type=group_type,
        group_anchor_sample=float(anchor_sample),
        bler=0.0,
    )


def _ctx_other(letter: str, bit: int) -> BlockContext:
    return BlockContext(
        block_letter=letter,
        bit_in_block=bit,
        group_pi=0x4652,
        group_type="0A",
        group_anchor_sample=0.0,
        bler=0.0,
    )


def _anchor_lookup_at(anchor_samples: list[float]):
    """Build a block_a_anchor_lookup that returns the most recent anchor
    from ``anchor_samples`` at or before the queried carrier sample, within
    the lookback window.  Used to simulate a decoder's view in tests."""
    def lookup(carrier: float, lookback: float):
        valid = [a for a in anchor_samples
                 if a <= carrier and carrier - a <= lookback]
        if not valid:
            return None
        return _ctx_a0(anchor_sample=max(valid))
    return lookup


# ---------------------------------------------------------------------------
# Fail-closed: no block-A bit-0 anchor in window → no measurement
# ---------------------------------------------------------------------------

class TestFailClosed:
    """Commits 6 and 8 made the matcher fail-closed.  A measurement is
    emitted only when a block_a_anchor_lookup returns a block-A bit-0
    context AND there's a SyncEvent close enough to that anchor's
    sample position.  All other configurations return no measurement.
    """

    def test_no_lookup_returns_no_measurement(self):
        dc = DeltaComputer(sample_rate_hz=256_000.0)   # no anchor lookup
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert results == []
        assert dc._anchor_no_lookup_dropped >= 1

    def test_lookup_returning_none_returns_no_measurement(self):
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            block_a_anchor_lookup=lambda c, lb: None,
        )
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert results == []
        assert dc._anchor_no_a_in_window_dropped >= 1

    def test_lookup_finds_a_but_no_sync_close_enough(self):
        """Lookup returns an A-anchor at sample 5000, but no SyncEvent
        is in the buffer close enough to that anchor → drop."""
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            # Anchor at 5000, but the syncs we feed are 1000-3000 (far from anchor)
            block_a_anchor_lookup=lambda c, lb: (
                _ctx_a0(5000.0) if c >= 5000 and c - 5000 <= lb else None
            ),
        )
        for s in [1000, 2000, 3000]:
            dc.feed_sync(_sync(s))
        results = dc.feed_onset(_onset(5500))
        assert results == []
        assert dc._anchor_no_sync_near_a_dropped >= 1


# ---------------------------------------------------------------------------
# Block-A anchor chosen
# ---------------------------------------------------------------------------

class TestBlockAAnchor:
    def test_block_a_anchor_picked_as_sync_sample(self):
        # Anchor available at sample 3000; matching SyncEvents at 1000, 2000, 3000.
        # The lookup returns "most recent A-anchor before carrier at sample 3000".
        # The closest SyncEvent to 3000 is the one at 3000.
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            block_a_anchor_lookup=_anchor_lookup_at([3000.0]),
        )
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert len(results) == 1
        m = results[0]
        assert m.sync_sample == 3000   # SyncEvent closest to the A anchor
        assert m.anchor_block_letter == "A"
        assert m.anchor_bit_in_block == 0
        assert m.anchor_group_pi == 0x4652
        assert m.anchor_group_type == "0A"

    def test_anchor_after_event_not_picked(self):
        """Commit 8 uses 'most recent A-anchor at or before the carrier
        event'.  A future anchor (sample > carrier) is never used.
        """
        # Anchor only exists at sample 5000; carrier is at 4000.
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            block_a_anchor_lookup=_anchor_lookup_at([5000.0]),
        )
        dc.feed_sync(_sync(3000))
        dc.feed_sync(_sync(5000))
        results = dc.feed_onset(_onset(4000))
        # No A-anchor at or before sample 4000 → drop
        assert results == []
        assert dc._anchor_no_a_in_window_dropped >= 1

    def test_chooses_most_recent_block_a_anchor_before_event(self):
        # Two A-anchors before the carrier (5000, 9000); carrier at 9500.
        # 'Most recent before' picks 9000.
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            max_sync_age_samples=20_000,
            block_a_anchor_lookup=_anchor_lookup_at([1000.0, 5000.0, 9000.0]),
        )
        for s in [1000, 5000, 9000]:
            dc.feed_sync(_sync(s))
        results = dc.feed_onset(_onset(9500))
        assert len(results) == 1
        # Closest SyncEvent to A-anchor 9000 is the SyncEvent at sample 9000
        assert results[0].sync_sample == 9000

    def test_anchor_outside_lookback_dropped(self):
        """An A-anchor too far back (beyond one group period) doesn't qualify."""
        # Anchor at sample 1000, but carrier at sample 100_000 — far beyond
        # one group period (~22000 samples at 256 kHz).
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            block_a_anchor_lookup=_anchor_lookup_at([1000.0]),
        )
        dc.feed_sync(_sync(99000))
        results = dc.feed_onset(_onset(100_000))
        # Lookup returns None because 100_000 - 1000 > 1 group period
        assert results == []
        assert dc._anchor_no_a_in_window_dropped >= 1


# ---------------------------------------------------------------------------
# Telemetry counters
# ---------------------------------------------------------------------------

class TestEndToEndRealFixture:
    """
    End-to-end: KUOW fixture → demod → block sync → DeltaComputer with
    block-context lookup → verify that ≥90% of resolved carrier onsets
    pick the block-A bit-0 anchor and carry PI=0x4652 metadata.
    """

    def test_block_a_anchor_dominates_on_kuow_fixture(self):
        import numpy as np
        from pathlib import Path
        from beagle_node.pipeline.rds_decoder import RDSDecoderService
        from beagle_node.pipeline.rds_sync_detector import RDSSyncDetector

        fixture = (
            Path(__file__).parents[1]
            / "fixtures"
            / "kuow_sync_audio_dpk_tdoa1_10s_20260524.npz"
        )
        if not fixture.exists():
            pytest.skip(f"fixture {fixture.name} not available")

        data = np.load(fixture)
        audio = data["audio"]
        fs = float(data["sample_rate_hz"])
        starts = data["start_samples"]

        # Decode all groups from the whole fixture
        svc = RDSDecoderService(
            fs_in=fs, window_seconds=10.0, decode_interval_ms=0.0
        )
        svc.push_audio(audio, start_sample=0)

        # Produce sync events at the natural per-chunk cadence
        det = RDSSyncDetector(sample_rate_hz=fs)
        all_events = []
        offset = 0
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(audio)
            chunk = audio[offset : offset + (end - start)]
            all_events.extend(det.process(chunk, start_sample=int(start)))
            offset += len(chunk)
        assert len(all_events) > 1000

        # Simulate carrier onsets every 0.5 s after decoder warmup (1.5 s in)
        onset_samples = [int(fs * (1.5 + i * 0.5)) for i in range(18)]
        # Build a time-sorted queue so feed_sync and feed_onset interleave
        # the way they would in the live pipeline.
        queue: list[tuple[float, str, object]] = []
        for ev in all_events:
            queue.append((ev.sample_index, "sync", ev))
        for s in onset_samples:
            if s < len(audio):
                queue.append((s, "onset", _onset(s)))
        queue.sort(key=lambda x: x[0])

        dc = DeltaComputer(
            sample_rate_hz=fs,
            block_a_anchor_lookup=svc.find_a_bit0_anchor,
            max_sync_age_samples=int(0.2 * fs),  # 200 ms pending-event aging
        )
        results = []
        for _, kind, ev in queue:
            if kind == "sync":
                dc.feed_sync(ev)
            else:
                results.extend(dc.feed_onset(ev))

        # Commit 6 fail-closed: emitted measurements may be fewer than the
        # number of onsets (some are dropped when no A-anchor is in window).
        # But every emitted measurement must be a block-A bit-0 anchor with
        # the correct KUOW PI.
        assert len(results) >= 10, (
            f"Too few resolved onsets: {len(results)} (expected ≥10 of 18; "
            f"some may be dropped due to BLER gaps)"
        )
        for m in results:
            assert m.anchor_block_letter == "A", (
                f"Fail-closed broken: emitted non-A measurement {m!r}"
            )
            assert m.anchor_bit_in_block == 0
            assert m.anchor_group_pi == 0x4652


class TestAnchorTelemetry:
    def test_counters_reflect_selection_outcomes(self):
        """Three carrier events, each preceded by an A-anchor:
          #1: no A-anchor yet → pending, eventually ages out (no_a counter)
          #2: A-anchor 100 samples back exists → emit (chose_block_a)
          #3: A-anchor still in lookback → emit (chose_block_a)
        """
        anchor_samples: list[float] = []

        def lookup(carrier: float, lookback: float):
            valid = [a for a in anchor_samples
                     if a <= carrier and carrier - a <= lookback]
            if not valid:
                return None
            return _ctx_a0(anchor_sample=max(valid))

        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            max_sync_age_samples=3_000,    # small so #1 ages out by #2
            block_a_anchor_lookup=lookup,
        )
        # Event #1: no anchor yet → pending
        dc.feed_sync(_sync(1500))
        r1 = dc.feed_onset(_onset(2000))
        assert r1 == []
        # Event #2: add an anchor before the carrier; #1 still has no anchor
        # before its time (anchor 5000 > 2000), so #1 ages out.  #2 resolves.
        anchor_samples.append(5000.0)
        dc.feed_sync(_sync(5000))
        r2 = dc.feed_onset(_onset(5500))
        assert len(r2) == 1
        # Event #3: still has anchor 5000 in lookback
        dc.feed_sync(_sync(8000))
        r3 = dc.feed_onset(_onset(8500))
        assert len(r3) == 1
        assert dc._anchor_chose_block_a == 2
        # #1 was checked many times and dropped each → counter increments
        assert dc._anchor_no_a_in_window_dropped >= 1
        assert dc._anchor_no_lookup_dropped == 0
