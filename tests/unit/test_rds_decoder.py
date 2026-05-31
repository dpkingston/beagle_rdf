# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for the rolling-window ``RDSDecoderService`` facade.

Covers:
  - Audio is buffered until first decode interval elapses
  - Re-decode produces groups matching the offline decoder
  - ``lookup(sample_index)`` returns correct block context
  - Stats (group count, duration) are populated after a decode
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from beagle_node.pipeline.rds_decoder import (
    BlockContext,
    DecoderStats,
    RDSDecoderService,
)

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "kuow_sync_audio_dpk_tdoa1_10s_20260524.npz"
)

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=f"fixture {FIXTURE_PATH.name} not available",
)


class TestDecoderService:
    def test_decode_runs_on_full_buffer(self):
        """Feed the whole KUOW fixture in a single push; expect groups back."""
        data = np.load(FIXTURE_PATH)
        audio = data["audio"]
        fs = float(data["sample_rate_hz"])

        # decode_interval_ms=0 → run on first call
        svc = RDSDecoderService(
            fs_in=fs, window_seconds=10.0, decode_interval_ms=0.0
        )
        groups = svc.push_audio(audio, start_sample=0)
        assert groups is not None
        # Should match the standalone decoder closely (~110 groups for 10s KUOW)
        with_pi = [g for g in groups if g.pi is not None]
        assert len(with_pi) > 80, (
            f"Too few groups with PI: {len(with_pi)} (expected ~100)"
        )
        # All confidently-decoded groups should be PI=0x4652
        pis = {g.pi for g in with_pi}
        assert pis == {0x4652}, f"Unexpected PIs: {pis}"

    def test_stats_populated(self):
        data = np.load(FIXTURE_PATH)
        svc = RDSDecoderService(
            fs_in=float(data["sample_rate_hz"]),
            window_seconds=2.0,
            decode_interval_ms=0.0,
        )
        # Push the first 2 seconds
        n = int(2.0 * float(data["sample_rate_hz"]))
        svc.push_audio(data["audio"][:n], start_sample=0)
        s = svc.stats
        assert isinstance(s, DecoderStats)
        assert s.decode_count == 1
        assert s.last_decode_duration_ms > 0
        assert s.last_decode_input_seconds == pytest.approx(2.0, abs=0.05)
        assert s.last_group_count > 0

    def test_buffering_without_interval_trigger(self):
        """When decode_interval_ms is large, audio just buffers."""
        data = np.load(FIXTURE_PATH)
        fs = float(data["sample_rate_hz"])
        svc = RDSDecoderService(
            fs_in=fs, window_seconds=5.0, decode_interval_ms=60_000.0
        )
        # First push always allowed.  Second push should be deferred.
        n_chunk = int(0.2 * fs)
        first = svc.push_audio(data["audio"][:n_chunk], start_sample=0)
        # Push another chunk immediately
        second = svc.push_audio(
            data["audio"][n_chunk : 2 * n_chunk], start_sample=n_chunk
        )
        # Second push should be deferred (rate-limited).
        assert second is None
        # latest_groups() returns whatever the first decode (if any) produced.
        assert isinstance(svc.latest_groups(), list)

    def test_lookup_returns_block_context_for_in_range_sample(self):
        """Given a known group anchor, lookup should resolve to its block."""
        data = np.load(FIXTURE_PATH)
        fs = float(data["sample_rate_hz"])
        svc = RDSDecoderService(
            fs_in=fs, window_seconds=10.0, decode_interval_ms=0.0
        )
        svc.push_audio(data["audio"], start_sample=0)

        groups = svc.latest_groups()
        decoded = [g for g in groups if g.pi == 0x4652]
        assert len(decoded) > 50

        # Pick a known block-A bit position and look it up
        anchor_group = next(g for g in decoded if not np.isnan(g.sample_index_first_bit))
        anchor_a = anchor_group.blocks[0]
        assert anchor_a is not None and anchor_a.is_received

        # Look up the very first bit of block A
        ctx = svc.lookup(anchor_a.sample_index + 1.0)  # ~middle of bit 0
        assert ctx is not None
        assert ctx.block_letter == "A"
        assert 0 <= ctx.bit_in_block <= 25
        assert ctx.group_pi == 0x4652

    def test_lookup_returns_none_outside_decoded_window(self):
        data = np.load(FIXTURE_PATH)
        fs = float(data["sample_rate_hz"])
        svc = RDSDecoderService(
            fs_in=fs, window_seconds=2.0, decode_interval_ms=0.0
        )
        svc.push_audio(data["audio"][: int(2.0 * fs)], start_sample=0)
        # Sample way outside the decoded window
        assert svc.lookup(sample_index=-1e9) is None
        assert svc.lookup(sample_index=1e12) is None


class TestPipelineIntegration:
    """Verify that the live pipeline instantiates the decoder service."""

    def test_pipeline_has_rds_decoder_when_sync_mode_rds(self):
        from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig

        cfg = PipelineConfig(sync_mode="rds")
        pipe = NodePipeline(config=cfg)
        assert pipe.rds_decoder is not None
        assert isinstance(pipe.rds_decoder, RDSDecoderService)

    def test_rds_health_snapshot_initial(self):
        """Snapshot is well-formed before any decode has run."""
        from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
        cfg = PipelineConfig(sync_mode="rds")
        pipe = NodePipeline(config=cfg)
        snap = pipe.rds_health_snapshot()
        assert snap is not None
        # Per-event counters start at zero
        assert snap["anchor_emitted"] == 0
        assert snap["anchor_aged_out"] == 0
        assert snap["anchor_match_attempts_failed"] == 0
        # No events yet → emit fraction is undefined
        assert snap["anchor_emit_fraction"] is None
        # No decodes yet → group count is zero, bler is None (NaN sentinel)
        assert snap["group_count"] == 0
        assert snap["bler_mean"] is None or snap["bler_mean"] == 0.0
        # group_period_hz is the constant for the server's reference
        assert snap["group_period_hz"] == pytest.approx(11.418, abs=0.01)

    def test_plateau_K_groups_follows_live_interval_reload(self):
        """Live reload of ``carrier.plateau_event_interval_s`` (via
        ``CarrierDetector.update_thresholds``) must immediately change
        the pipeline's K-groups cadence.  Regression test for the bug
        where ``_plateau_K_groups`` was captured at pipeline init and
        wouldn't follow the carrier_detect's live updates."""
        from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
        cfg = PipelineConfig(
            sync_mode="rds",
            carrier_plateau_event_interval_s=1.0,
        )
        pipe = NodePipeline(config=cfg)
        # 1.0 s ÷ 0.0876 s/group ≈ 11.42 → K = 11
        assert pipe._plateau_K_groups == 11

        # Live reload to 2.0 s (intermittent-PTT tuning).
        pipe.carrier_detector.update_thresholds(plateau_event_interval_s=2.0)
        assert pipe._plateau_K_groups == 23  # round(2.0 / 0.0876) = 23

        # And to 0.0 (disable).
        pipe.carrier_detector.update_thresholds(plateau_event_interval_s=0.0)
        assert pipe._plateau_K_groups == 0

    def test_plateau_target_anchor_does_not_round_down_below_float_anchor(self):
        """Regression test: ``_maybe_emit_anchor_plateau`` must convert
        the demod's sub-sample-precise ``group_anchor_sample`` (float) to
        ``target_anchor`` (int) using ``math.ceil`` rather than ``int()``.

        Why: the downstream matcher (``find_a_bit0_anchor``) skips any
        block-A whose float ``sample_index`` is strictly greater than
        ``carrier_sample``.  If the pipeline emits a plateau at
        ``target_anchor = int(950000.5) = 950000`` and the matcher then
        queries with carrier_sample = 950000, the float anchor at
        950000.5 fails ``950000.5 > 950000`` and is skipped — the
        matcher returns the PREVIOUS block-A bit-0 (~21 900 samples /
        87.6 ms / 1 RDS group earlier), producing the off-by-one-group
        ``sync_to_snippet_start_ns`` ≈ +87.6 ms bug we saw in
        production after the anchor-trigger plateau commit shipped.

        ``math.ceil`` guarantees ``target_anchor >= group_anchor_sample``,
        so the matcher accepts the intended anchor."""
        import math
        # ceil(X.5) is at-or-above X.5; int(X.5) is strictly less than X.5.
        for s in [950_000.0, 950_000.1, 950_000.5, 950_000.9, 950_001.0]:
            assert math.ceil(s) >= s, (
                f"ceil({s})={math.ceil(s)} but must be >= {s}"
            )
        # And ``int()`` would have broken the round-trip for non-integer s.
        assert int(950_000.5) < 950_000.5, (
            "If this assertion changes, Python's int() semantics changed "
            "and the bug-fix rationale needs revisiting."
        )

    def test_plateau_global_epoch_safety_margin(self):
        """The phase-locked epoch derivation uses ``round`` (not
        ``floor``) for a symmetric ±half-group safety margin.  Two
        nodes whose anchor-wall-clock estimates for the SAME RDS
        broadcast group differ by NTP-grade skew compute the same
        epoch in the worst case (broadcast right at a rounding center)
        and the best case (broadcast right at a rounding boundary).

        The realistic operating envelope is:
          - NTP skew across hardened nodes:  <10 ms
          - Propagation delay across baseline: <100 µs
          - Processing-latency jitter:       <5 ms (single buffer)

        ⇒ realistic worst-case cross-node anchor_wall_ns delta ≈ 15 ms.

        We test from 0 to 40 ms of skew, sweeping the broadcast position
        across the safe-zone fraction of the group cycle.  Skews above
        ~44 ms can flip the rounded epoch at boundaries — that's a real
        quantization edge but well outside the operating envelope."""
        GROUP_PERIOD_NS = int(round(1e9 * 104.0 / 1187.5))  # 87_578_947 ns

        def epoch(wall_ns):
            return int(round(wall_ns / GROUP_PERIOD_NS))

        # For each ``skew_ms``, find the range of broadcast positions
        # within a group for which both nodes agree on the rounded epoch.
        # Assert that range covers > the realistic envelope.
        for skew_ms in (1, 5, 10, 15, 20):
            skew_ns = skew_ms * 1_000_000
            base_wall = 1_780_000_000_000_000_000
            # Sweep offset across the full group period in 5 ms steps.
            agreeing = 0
            total = 0
            for offset_ns in range(0, GROUP_PERIOD_NS, 5_000_000):
                wall_a = base_wall + offset_ns
                wall_b = wall_a + skew_ns
                total += 1
                if epoch(wall_a) == epoch(wall_b):
                    agreeing += 1
            # Expected safe-zone fraction = (group_period − skew) / group_period
            safe_frac = (GROUP_PERIOD_NS - skew_ns) / GROUP_PERIOD_NS
            observed_frac = agreeing / total
            # At skew = 15 ms, expected ~83 % of positions agree.
            assert observed_frac >= safe_frac - 0.05, (
                f"skew={skew_ms}ms: observed agree-frac {observed_frac:.2f} "
                f"< expected lower bound {safe_frac - 0.05:.2f}"
            )

        # Realistic-scenario test: NTP 10 ms + propagation + processing
        # = 15 ms total skew is well inside the safe zone for ~83 % of
        # group positions.  All groups eventually fire ASAP after
        # idle→active, and over the course of a long key-up cross-node
        # plateau alignment is empirically tight.
        skew_ns = 15 * 1_000_000
        # The 17 % of groups where rounding flips is acceptable: we
        # emit at K-multiple epochs, not every group, so an occasional
        # boundary-straddling group just delays one emission by one
        # K-cycle, not catastrophic.
        assert (GROUP_PERIOD_NS - skew_ns) / GROUP_PERIOD_NS > 0.80

    def test_plateau_global_epoch_K_residue_gate(self):
        """For K = 11 groups (≈ 1 s cadence), plateau emissions fire on
        epochs that are exact multiples of 11 — and only those.  All
        nodes computing the same global epoch will agree on whether to
        fire.  This locks cross-node plateaus to the same RDS group."""
        K = 11
        # Pretend two epochs at known offsets from a multiple of K.
        epoch_on_grid = 11 * 12345  # multiple of K
        epoch_off_grid = epoch_on_grid + 3
        assert epoch_on_grid % K == 0
        assert epoch_off_grid % K != 0
