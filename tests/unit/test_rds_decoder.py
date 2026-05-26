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
