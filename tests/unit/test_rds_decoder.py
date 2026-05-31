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

    def test_ring_enlarged_to_cover_decode_window_with_rds(self):
        """Step 2 of the plateau cross-node-sync fix: when an RDS decoder
        is active, the carrier-detector IQ ring must hold at least the RDS
        decode window so a (possibly ~1 decode-interval stale) block-A
        anchor returned by the widened ``find_a_bit0_anchor`` lookback is
        still snippet-extractable.

        Production telemetry (2026-05-31) had skip_try_emit=0 with the old
        ~196 ms ring only because the 2-group lookback never returned an
        anchor older than the ring.  Widening the lookback (same commit)
        without this ring change would convert skip_no_anchor misses into
        skip_try_emit misses; the enlarged ring prevents that."""
        from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
        import math

        cfg = PipelineConfig(sync_mode="rds")
        pipe = NodePipeline(config=cfg)
        ring = pipe.carrier_detector._iq_ring
        window = pipe.carrier_detector._window
        target_rate = cfg.sdr_rate_hz / cfg.target_decimation

        # Ring must cover decode window + 1 s margin.
        ring_seconds = ring.maxlen * window / target_rate
        assert ring_seconds >= cfg.rds_decoder_window_seconds + 1.0 - 1e-6, (
            f"ring holds {ring_seconds:.2f}s, need "
            f">= {cfg.rds_decoder_window_seconds + 1.0:.2f}s"
        )
        # And it must comfortably exceed the old ~3x-snippet auto-size,
        # which was the pre-fix default.
        snippet_windows = max(1, math.ceil(cfg.carrier_snippet_samples / window))
        assert ring.maxlen > snippet_windows * 3

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
        # Per-attempt plateau-emit telemetry present and zeroed.
        assert "plateau_emit" in snap
        pe = snap["plateau_emit"]
        assert pe == {
            "attempts": 0, "ok": 0, "skip_no_anchor": 0,
            "skip_not_kslot": 0, "skip_already_emitted": 0, "skip_try_emit": 0,
        }
        # Snapshot returns a COPY — mutating it must not corrupt the live
        # counters (dict() defensive copy in rds_health_snapshot).
        pe["ok"] = 999
        assert pipe.rds_health_snapshot()["plateau_emit"]["ok"] == 0

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


class _FakeCarrierDet:
    """Minimal carrier detector for driving ``_maybe_emit_anchor_plateau``
    in isolation (Step 3 tests)."""

    def __init__(self, interval_s, snippet_samples, cumulative_sample):
        self._plateau_interval_s = interval_s
        self.state = "active"
        self.snippet_samples = snippet_samples
        self.cumulative_sample = cumulative_sample
        self.emitted_at: list[int] = []

    def plateau_snippet_available(self, ta):  # big ring → always available
        return True

    def try_emit_plateau_at(self, ta):
        self.emitted_at.append(int(ta))
        # Return a sentinel truthy object with the attributes the caller
        # reads downstream (it only forwards them).
        from beagle_node.pipeline.carrier_detect import CarrierPlateau
        return CarrierPlateau(
            sample_index=int(ta), power_db=0.0, iq_snippet=b"",
            transition_start=0, transition_end=0,
        )


class _FakeDecoder:
    def __init__(self, anchors):
        self._anchors = list(anchors)

    def block_a_bit0_anchors(self):
        return list(self._anchors)


class TestStep3GlobalSlotMarch:
    """Step 3: the emitter marches the global K-slot grid so all nodes
    emit on the identical epochs."""

    def _make_pipe(self, *, interval_s=1.0):
        from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
        cfg = PipelineConfig(
            sync_mode="rds", carrier_plateau_event_interval_s=interval_s,
        )
        return NodePipeline(config=cfg), cfg

    def _sync_anchor_for_epoch(self, pipe, cfg, base_wall_ns, base_sample, epoch):
        """Return the sync-domain block-A bit-0 sample whose global epoch
        is ``epoch`` for a node anchored at (base_wall_ns, base_sample)."""
        # Place the anchor at the centre of epoch E's window so round()
        # recovers exactly E.
        anchor_wall = epoch * pipe._group_period_ns
        ta = base_sample + round(
            (anchor_wall - base_wall_ns) * pipe._target_rate_hz / 1e9
        )
        # Invert ta = ceil(s) * sd // td  →  s ≈ ta * td / sd.
        return ta * cfg.target_decimation / cfg.sync_decimation

    def test_first_emission_picks_newest_kslot_and_it_is_a_multiple_of_K(self):
        pipe, cfg = self._make_pipe()
        K = pipe._plateau_K_groups
        assert K == 11
        base_wall = 1_780_000_000_000_000_000
        base_sample = 5_000_000
        # A contiguous run of 30 groups around two K-multiples.
        e0 = (base_wall // pipe._group_period_ns) + 5
        epochs = list(range(e0, e0 + 30))
        anchors = [
            self._sync_anchor_for_epoch(pipe, cfg, base_wall, base_sample, e)
            for e in epochs
        ]
        # Cumulative sample far ahead so every anchor's snippet fits.
        max_ta = base_sample + round(
            (max(epochs) * pipe._group_period_ns - base_wall)
            * pipe._target_rate_hz / 1e9
        )
        pipe._carrier_det = _FakeCarrierDet(
            interval_s=1.0, snippet_samples=16384,
            cumulative_sample=max_ta + 16384 + 1000,
        )
        pipe._rds_decoder = _FakeDecoder(anchors)
        pipe._buf_anchor_wall_ns = base_wall
        pipe._buf_anchor_target_sample = base_sample

        plateau = pipe._maybe_emit_anchor_plateau()
        assert plateau is not None
        chosen = pipe._last_emitted_global_epoch
        assert chosen % K == 0, f"chosen epoch {chosen} not a K-multiple"
        # Fresh start → newest available K-slot in the run.
        kslots = [e for e in epochs if e % K == 0]
        assert chosen == max(kslots)

    def test_two_nodes_with_clock_skew_pick_the_same_epoch(self):
        """The cross-node invariant: two nodes seeing the same broadcast
        groups, whose hardware wall-clocks differ by NTP-grade skew, must
        select the IDENTICAL global K-slot epoch — so their plateaus pair."""
        K = 11
        base_wall = 1_780_000_000_000_000_000
        chosen = []
        for skew_ms in (0, +20, -20, +30):
            pipe, cfg = self._make_pipe()
            base_sample = 5_000_000
            e0 = (base_wall // pipe._group_period_ns) + 5
            epochs = list(range(e0, e0 + 30))
            # Node sees the SAME physical groups (same true wall-clocks),
            # but its own hardware-timestamp anchor is skewed by skew_ms.
            node_base_wall = base_wall + skew_ms * 1_000_000
            anchors = [
                self._sync_anchor_for_epoch(
                    pipe, cfg, node_base_wall, base_sample, e)
                for e in epochs
            ]
            max_ta = base_sample + round(
                (max(epochs) * pipe._group_period_ns - node_base_wall)
                * pipe._target_rate_hz / 1e9
            )
            pipe._carrier_det = _FakeCarrierDet(
                interval_s=1.0, snippet_samples=16384,
                cumulative_sample=max_ta + 16384 + 1000,
            )
            pipe._rds_decoder = _FakeDecoder(anchors)
            pipe._buf_anchor_wall_ns = node_base_wall
            pipe._buf_anchor_target_sample = base_sample
            assert pipe._maybe_emit_anchor_plateau() is not None
            chosen.append(pipe._last_emitted_global_epoch)

        assert len(set(chosen)) == 1, (
            f"nodes with skew picked different epochs: {chosen} — "
            f"cross-node plateau sync would be broken"
        )
        assert chosen[0] % K == 0

    def test_march_emits_next_kslot_in_order(self):
        """After emitting slot E, the next emission is E+K (the grid
        marches in order, no skips)."""
        pipe, cfg = self._make_pipe()
        K = pipe._plateau_K_groups
        base_wall = 1_780_000_000_000_000_000
        base_sample = 5_000_000
        e0 = (base_wall // pipe._group_period_ns) + 5
        epochs = list(range(e0, e0 + 40))
        anchors = [
            self._sync_anchor_for_epoch(pipe, cfg, base_wall, base_sample, e)
            for e in epochs
        ]
        max_ta = base_sample + round(
            (max(epochs) * pipe._group_period_ns - base_wall)
            * pipe._target_rate_hz / 1e9
        )
        pipe._carrier_det = _FakeCarrierDet(
            interval_s=1.0, snippet_samples=16384,
            cumulative_sample=max_ta + 16384 + 1000,
        )
        pipe._rds_decoder = _FakeDecoder(anchors)
        pipe._buf_anchor_wall_ns = base_wall
        pipe._buf_anchor_target_sample = base_sample

        # Pre-seed last_emitted to the OLDEST K-slot so the march has room.
        kslots = sorted(e for e in epochs if e % K == 0)
        assert len(kslots) >= 2
        pipe._last_emitted_global_epoch = kslots[0]
        assert pipe._maybe_emit_anchor_plateau() is not None
        # Catch-up branch picks the oldest unemitted K-slot → kslots[1].
        assert pipe._last_emitted_global_epoch == kslots[1]
