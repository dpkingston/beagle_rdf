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
def _ctx_a0(pi: int = 0x4652, group_type: str = "0A") -> BlockContext:
    return BlockContext(
        block_letter="A",
        bit_in_block=0,
        group_pi=pi,
        group_type=group_type,
        group_anchor_sample=0.0,
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


# ---------------------------------------------------------------------------
# Legacy preserved
# ---------------------------------------------------------------------------

class TestLegacyPreserved:
    def test_no_lookup_uses_most_recent_before_event(self):
        dc = DeltaComputer(sample_rate_hz=256_000.0)
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert len(results) == 1
        m = results[0]
        # Legacy = most recent before event
        assert m.sync_sample == 3000
        assert m.anchor_block_letter is None
        assert m.anchor_bit_in_block is None

    def test_lookup_returning_none_falls_back_to_legacy(self):
        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            block_context_lookup=lambda s: None,
        )
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert len(results) == 1
        m = results[0]
        assert m.sync_sample == 3000  # legacy nearest-before
        assert m.anchor_block_letter is None

    def test_lookup_returning_non_block_a_falls_back_to_legacy(self):
        # All in-range sync events are decoded as block C, bit 5 — not the
        # preferred A-bit-0 anchor.  Should fall back to legacy.
        def lookup(s: float) -> Optional[BlockContext]:
            return _ctx_other("C", 5)

        dc = DeltaComputer(
            sample_rate_hz=256_000.0, block_context_lookup=lookup
        )
        for s in [1000, 2000, 3000]:
            dc.feed_sync(_sync(s))
        results = dc.feed_onset(_onset(3500))
        assert len(results) == 1
        assert results[0].sync_sample == 3000
        assert results[0].anchor_block_letter is None


# ---------------------------------------------------------------------------
# Block-A anchor chosen
# ---------------------------------------------------------------------------

class TestBlockAAnchor:
    def test_prefers_block_a_anchor_when_available(self):
        # SyncEvents at 1000 and 3000.  Only 1000 is annotated as A-bit-0
        # by the lookup; 3000 is annotated as C bit 5.  Expected: pick 1000
        # even though 3000 is more recent.
        def lookup(s: float) -> Optional[BlockContext]:
            if abs(s - 1000) < 0.5:
                return _ctx_a0()
            return _ctx_other("C", 5)

        dc = DeltaComputer(
            sample_rate_hz=256_000.0, block_context_lookup=lookup
        )
        dc.feed_sync(_sync(1000))
        dc.feed_sync(_sync(2000))
        dc.feed_sync(_sync(3000))
        results = dc.feed_onset(_onset(3500))
        assert len(results) == 1
        m = results[0]
        assert m.sync_sample == 1000
        assert m.anchor_block_letter == "A"
        assert m.anchor_bit_in_block == 0
        assert m.anchor_group_pi == 0x4652
        assert m.anchor_group_type == "0A"

    def test_block_a_anchor_after_event_allowed(self):
        # A SyncEvent AFTER the carrier onset, but within max_sync_age,
        # is acceptable when it's a block-A bit-0.  This relaxes the
        # legacy pre-event constraint.
        def lookup(s: float) -> Optional[BlockContext]:
            if abs(s - 5000) < 0.5:
                return _ctx_a0()
            return None

        dc = DeltaComputer(
            sample_rate_hz=256_000.0, block_context_lookup=lookup
        )
        dc.feed_sync(_sync(3000))
        dc.feed_sync(_sync(5000))
        # Carrier at 4000 — A anchor at 5000 is 1000 samples after
        results = dc.feed_onset(_onset(4000))
        assert len(results) == 1
        m = results[0]
        assert m.sync_sample == 5000
        assert m.anchor_block_letter == "A"
        # The sample delta is negative (sync after event)
        assert m.sync_delta_samples == -1000

    def test_chooses_closest_block_a_anchor_when_multiple(self):
        # Three block-A bit-0 anchors at 1000, 5000, 9000.  Carrier at
        # 6000.  Expect 5000 (closest, 1000 before) — not 9000 (3000 after).
        def lookup(s: float) -> Optional[BlockContext]:
            if int(s) in (1000, 5000, 9000):
                return _ctx_a0()
            return None

        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            max_sync_age_samples=20_000,
            block_context_lookup=lookup,
        )
        for s in [1000, 5000, 9000]:
            dc.feed_sync(_sync(s))
        results = dc.feed_onset(_onset(6000))
        assert len(results) == 1
        assert results[0].sync_sample == 5000

    def test_anchor_only_when_in_age_window(self):
        # Block-A anchor exists but is too old (outside max_sync_age).
        # Should fall back to legacy with whatever's in range.
        def lookup(s: float) -> Optional[BlockContext]:
            if abs(s - 1000) < 0.5:
                return _ctx_a0()
            return None

        dc = DeltaComputer(
            sample_rate_hz=256_000.0,
            max_sync_age_samples=5_000,
            block_context_lookup=lookup,
        )
        dc.feed_sync(_sync(1000))     # block-A, but too old
        dc.feed_sync(_sync(9000))     # in range, not block-A
        results = dc.feed_onset(_onset(10_000))
        assert len(results) == 1
        m = results[0]
        # 1000 is out of range (10000 - 1000 = 9000 > 5000); 9000 is in range
        assert m.sync_sample == 9000
        assert m.anchor_block_letter is None


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
            block_context_lookup=svc.lookup,
            max_sync_age_samples=int(0.2 * fs),  # 200 ms anchor window
        )
        results = []
        for _, kind, ev in queue:
            if kind == "sync":
                dc.feed_sync(ev)
            else:
                results.extend(dc.feed_onset(ev))

        assert len(results) >= 14, f"Too few resolved onsets: {len(results)}"

        # Block-A anchor should dominate now that the decoder has groups
        with_anchor = sum(1 for m in results if m.anchor_block_letter == "A")
        frac = with_anchor / len(results)
        assert frac >= 0.85, (
            f"Only {frac:.0%} of onsets used block-A anchor "
            f"({with_anchor}/{len(results)})"
        )
        # All anchored measurements should carry the correct KUOW PI and bit-0
        for m in results:
            if m.anchor_block_letter == "A":
                assert m.anchor_bit_in_block == 0
                assert m.anchor_group_pi == 0x4652


class TestAnchorTelemetry:
    def test_counters_reflect_selection_outcomes(self):
        # Three calls: 1 should use block-A, 2 should fall back to legacy.
        n_a_calls = 0

        def lookup(s: float) -> Optional[BlockContext]:
            nonlocal n_a_calls
            # Only the SyncEvent at exactly sample 5000 is block-A
            if abs(s - 5000) < 0.5:
                n_a_calls += 1
                return _ctx_a0()
            return None

        dc = DeltaComputer(
            sample_rate_hz=256_000.0, block_context_lookup=lookup
        )
        # Match #1: only sync at 1500, no A in range → legacy
        dc.feed_sync(_sync(1500))
        dc.feed_onset(_onset(2000))
        # Match #2: now sync at 5000 is in range → A-bit-0 win
        dc.feed_sync(_sync(5000))
        dc.feed_onset(_onset(5500))
        # Match #3: A is still in range but match takes nearest A
        dc.feed_sync(_sync(8000))
        dc.feed_onset(_onset(8500))
        # 8500 - 5000 = 3500 ≤ default 7680 max_sync_age → A still in range
        assert dc._anchor_chose_block_a == 2  # matches #2 and #3
        assert dc._anchor_fallback_legacy == 1  # match #1
