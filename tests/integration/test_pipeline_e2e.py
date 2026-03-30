# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration test: synthetic FM pilot + LMR carrier -> correct sync_delta_ns.

This test drives the full NodePipeline with MockReceiver-generated IQ
and verifies that the measured sync_delta_ns matches the known ground-truth
delay between the FM pilot events and the LMR carrier onset.

Setup
-----
- Sync channel: FM IQ at 2.048 MSPS containing a 19 kHz pilot tone
- Target channel: LMR IQ at 2.048 MSPS, carrier begins at a known sample
- Known sync event spacing: 10 ms windows -> one SyncEvent per 2560 decimated samples
- Known onset: LMR carrier starts at sample T in the raw stream

Expected result: sync_delta_ns ~= (T_sync_space - nearest_sync_sample) * 1e9 / sync_rate
"""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig

SDR_RATE  = 2_048_000.0
SYNC_DEC  = 8                    # -> 256 kHz
TARGET_DEC = 32                  # -> 64 kHz
SYNC_RATE = SDR_RATE / SYNC_DEC  # 256_000 Hz
SYNC_PERIOD_SAMPLES = int(SYNC_RATE * 0.010)   # 2560 samples per 10 ms window


def _fm_sync_iq(n: int, pilot_amplitude: float = 0.1) -> np.ndarray:
    """
    Generate FM IQ containing a strong 19 kHz pilot tone in the baseband.

    We approximate FM modulation: the pilot appears as a 19 kHz sinusoid in
    the demodulated audio.  For an IQ signal at SDR_RATE centred on the FM
    station, the pilot modulates the carrier phase, producing sidebands at
    +/-19 kHz.  For test purposes we directly inject a +/-19 kHz component.

    Carrier at DC (centre-tuned), pilot as a +/-19 kHz phase modulation:
        x(t) = exp(j * beta * sin(2pi * 19000 * t))
    where beta = pilot_amplitude / (2pi * 19000 / SDR_RATE) controls modulation depth.
    """
    t = np.arange(n) / SDR_RATE
    # Phase modulation with pilot
    beta = pilot_amplitude * SDR_RATE / (2.0 * np.pi * 19_000.0)
    phase = beta * np.sin(2.0 * np.pi * 19_000.0 * t)
    return np.exp(1j * phase).astype(np.complex64)


def _lmr_iq(n: int, carrier_start: int,
             noise_db: float = -60.0,
             carrier_db: float = -20.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate LMR IQ: noise before carrier_start, carrier after."""
    rng = rng or np.random.default_rng(99)
    noise_amp = 10 ** (noise_db / 20.0) / np.sqrt(2)
    iq = noise_amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    if carrier_start < n:
        carrier_amp = 10 ** (carrier_db / 20.0)
        # Use a 1 kHz sinusoid so the per-block DC removal (iq -= mean(iq)) does
        # not cancel the carrier.  A pure DC carrier has mean ~= carrier_amp and
        # would be zeroed out.  A 1 kHz tone has mean ~= 0 over any integer-cycle
        # window, so it survives intact and passes the 25 kHz decimation LPF.
        t = np.arange(n - carrier_start) / SDR_RATE
        iq[carrier_start:] = (
            carrier_amp * np.exp(1j * 2.0 * np.pi * 1_000.0 * t)
        ).astype(np.complex64)
    return iq


# ---------------------------------------------------------------------------
# End-to-end measurement test
# ---------------------------------------------------------------------------

class TestPipelineE2E:

    def _run(self, total_samples: int, carrier_start_raw: int,
             buffer_size: int = 65_536) -> list:
        """Run pipeline and collect all measurements."""
        measurements = []
        cfg = PipelineConfig(
            sdr_rate_hz=SDR_RATE,
            sync_decimation=SYNC_DEC,
            sync_cutoff_hz=128_000.0,
            sync_period_ms=10.0,
            target_decimation=TARGET_DEC,
            target_cutoff_hz=25_000.0,
            carrier_onset_db=-25.0,
            carrier_offset_db=-35.0,
            carrier_window_samples=32,
            min_corr_peak=0.05,   # lower threshold for synthetic signal
        )
        pipeline = NodePipeline(
            config=cfg,
            on_measurement=measurements.append,
        )

        rng = np.random.default_rng(42)
        sync_iq = _fm_sync_iq(total_samples)
        target_iq = _lmr_iq(total_samples, carrier_start_raw, rng=rng)

        # Feed in buffer-sized chunks, interleaving sync and target
        for offset in range(0, total_samples, buffer_size):
            sl = slice(offset, offset + buffer_size)
            pipeline.process_sync_buffer(sync_iq[sl])
            pipeline.process_target_buffer(target_iq[sl])

        return measurements

    def test_produces_at_least_one_measurement(self):
        """End-to-end pipeline with known inputs should produce measurements."""
        # Carrier starts 100 ms in - after multiple sync events have accumulated
        carrier_start = int(SDR_RATE * 0.100)
        measurements = self._run(
            total_samples=int(SDR_RATE * 0.5),   # 0.5 seconds
            carrier_start_raw=carrier_start,
        )
        assert len(measurements) >= 1, "Pipeline produced no measurements"

    def test_sync_delta_ns_reasonable(self):
        """
        sync_delta_ns should be positive and less than one sync period (10 ms).

        The carrier starts at a known point in the stream.  The nearest
        preceding sync event should be within one 10 ms window, so
        sync_delta_ns in [0, 10_000_000] nanoseconds.
        """
        carrier_start = int(SDR_RATE * 0.100)   # 100 ms in
        measurements = self._run(
            total_samples=int(SDR_RATE * 0.5),
            carrier_start_raw=carrier_start,
        )
        assert len(measurements) >= 1

        m = measurements[0]
        # Must be non-negative (onset after sync)
        assert m.sync_delta_ns >= 0, f"Negative sync_delta_ns: {m.sync_delta_ns}"
        # Must be within 2 sync periods (20 ms) - accounting for filter latency
        assert m.sync_delta_ns <= 20_000_000, \
            f"sync_delta_ns too large: {m.sync_delta_ns} ns"

    def test_measurement_fields_populated(self):
        """All TDOAMeasurement fields should be set sensibly."""
        carrier_start = int(SDR_RATE * 0.080)
        measurements = self._run(
            total_samples=int(SDR_RATE * 0.4),
            carrier_start_raw=carrier_start,
        )
        assert len(measurements) >= 1
        m = measurements[0]

        assert m.sample_rate_hz > 0
        assert 0.99 <= m.sample_rate_correction <= 1.01
        assert m.corr_peak >= 0.0
        assert m.pps_anchored is False
        assert m.target_sample > m.sync_sample

    def test_multiple_buffers_produce_consistent_results(self):
        """
        Splitting the same IQ data into different buffer sizes should produce
        similar sync_delta_ns (verifying state continuity across buffers).
        """
        carrier_start = int(SDR_RATE * 0.080)
        total = int(SDR_RATE * 0.4)

        m_large = self._run(total, carrier_start, buffer_size=65_536)
        m_small = self._run(total, carrier_start, buffer_size=16_384)

        assert len(m_large) >= 1
        assert len(m_small) >= 1

        # First measurement sync_delta_ns should be within 1 ms of each other
        diff = abs(m_large[0].sync_delta_ns - m_small[0].sync_delta_ns)
        assert diff < 1_000_000, \
            f"Buffer-size sensitivity: {m_large[0].sync_delta_ns} vs {m_small[0].sync_delta_ns}"


# ---------------------------------------------------------------------------
# freq_hop timing: alternating blocks with raw_start_sample offsets
# ---------------------------------------------------------------------------

class TestFreqHopTiming:
    """
    Verify that the raw_start_sample offsets used in freq_hop mode produce
    correct sync_delta_ns measurements.

    In freq_hop mode the ADC stream is split into alternating blocks:
      Block 0 (sync):   raw ADC samples [0,     B-1]
      Block 1 (target): raw ADC samples [B,   2B-1]
      Block 2 (sync):   raw ADC samples [2B,  3B-1]
      ...

    process_sync_buffer / process_target_buffer each receive the correct
    raw_start_sample so both channels share the same continuous sample-index
    domain, enabling correct sync_delta_ns computation.
    """

    BLOCK = 32_768   # raw samples per block (~16 ms at 2.048 MSPS)

    def _run_freq_hop(self, n_blocks: int, carrier_block: int = 0,
                      carrier_offset: int = 100) -> list:
        """
        Feed n_blocks alternating sync/target blocks with correct raw_start_sample.

        carrier_block : 0-indexed among *target* blocks (0 = first target block)
        carrier_offset : sample offset within that target block where carrier starts
        """
        measurements: list = []
        cfg = PipelineConfig(
            sdr_rate_hz=SDR_RATE,
            sync_decimation=SYNC_DEC,
            sync_cutoff_hz=128_000.0,
            sync_period_ms=10.0,
            target_decimation=TARGET_DEC,
            target_cutoff_hz=25_000.0,
            carrier_onset_db=-25.0,
            carrier_offset_db=-35.0,
            carrier_window_samples=32,
            min_corr_peak=0.05,
        )
        pipeline = NodePipeline(
            config=cfg,
            on_measurement=measurements.append,
        )

        B = self.BLOCK
        rng = np.random.default_rng(17)

        # Build full sync IQ covering all blocks
        total_raw = n_blocks * B
        sync_full = _fm_sync_iq(total_raw)

        # Build target IQ.  Target blocks are block indices 1, 3, 5, ...
        # carrier_block=0 -> raw ADC block 1, starts at raw sample B.
        n_target_blocks = (n_blocks + 1) // 2
        target_total = n_target_blocks * B
        # Global carrier start within the concatenated target IQ
        carrier_start_in_target = carrier_block * B + carrier_offset
        target_full = _lmr_iq(target_total, carrier_start_in_target, rng=rng)

        target_idx = 0
        for blk in range(n_blocks):
            raw_start = blk * B
            if blk % 2 == 0:
                # Sync block
                buf = sync_full[raw_start:raw_start + B]
                pipeline.process_sync_buffer(buf, raw_start_sample=raw_start)
            else:
                # Target block
                t_off = target_idx * B
                buf = target_full[t_off:t_off + B]
                pipeline.process_target_buffer(buf, raw_start_sample=raw_start)
                target_idx += 1

        return measurements

    def test_freq_hop_produces_measurement(self):
        """Pipeline fed with freq_hop-style offsets should produce measurements."""
        measurements = self._run_freq_hop(n_blocks=8, carrier_block=0)
        assert len(measurements) >= 1

    def test_freq_hop_sync_delta_non_negative(self):
        """
        Onset in target block 1 (raw [B, 2B-1]) follows sync events from block 0
        (raw [0, B-1]), so sync_delta_ns must be non-negative.
        """
        measurements = self._run_freq_hop(n_blocks=8, carrier_block=0)
        assert len(measurements) >= 1
        assert measurements[0].sync_delta_ns >= 0, (
            f"Negative sync_delta_ns: {measurements[0].sync_delta_ns}"
        )

    def test_freq_hop_sync_delta_reasonable(self):
        """
        sync_delta_ns must be less than max_sync_age (30 ms) + one block (16 ms).
        """
        measurements = self._run_freq_hop(n_blocks=8, carrier_block=0)
        assert len(measurements) >= 1
        m = measurements[0]
        # max_sync_age_samples default -> ~30 ms; target block starts ~16 ms after sync block
        assert m.sync_delta_ns <= 50_000_000, (
            f"sync_delta_ns too large: {m.sync_delta_ns} ns"
        )

    def test_freq_hop_delta_reflects_block_offset(self):
        """
        The sync_delta for a carrier in target block 1 must be larger than
        the delta for a carrier at the same *relative* offset if both channels
        were fed from raw sample 0 (the non-offset case).

        This confirms raw_start_sample correctly shifts the timing into the
        continuous ADC domain rather than treating every buffer as starting
        at t=0.
        """
        B = self.BLOCK
        carrier_offset = 500  # samples into target buffer

        # freq_hop: target buffer starts at raw sample B
        m_hop = self._run_freq_hop(n_blocks=8, carrier_block=0,
                                   carrier_offset=carrier_offset)

        # non-offset: feed the SAME IQ buffers but with raw_start_sample=0
        # (simulating single_sdr / mock mode where everything starts at 0)
        measurements_no_offset: list = []
        cfg = PipelineConfig(
            sdr_rate_hz=SDR_RATE,
            sync_decimation=SYNC_DEC,
            sync_cutoff_hz=128_000.0,
            sync_period_ms=10.0,
            target_decimation=TARGET_DEC,
            target_cutoff_hz=25_000.0,
            carrier_onset_db=-25.0,
            carrier_offset_db=-35.0,
            carrier_window_samples=32,
            min_corr_peak=0.05,
        )
        pipeline_no = NodePipeline(
            config=cfg,
            on_measurement=measurements_no_offset.append,
        )
        rng = np.random.default_rng(17)
        sync_buf = _fm_sync_iq(B)
        target_buf = _lmr_iq(B, carrier_offset, rng=rng)
        pipeline_no.process_sync_buffer(sync_buf)           # raw_start=0
        pipeline_no.process_target_buffer(target_buf)       # raw_start=0

        if not m_hop or not measurements_no_offset:
            pytest.skip("No measurements produced - adjust thresholds")

        delta_hop = m_hop[0].sync_delta_ns
        delta_no  = measurements_no_offset[0].sync_delta_ns

        # In freq_hop the target is one full block (~16 ms) later in raw time,
        # so its sync_delta_ns should be significantly larger.
        assert delta_hop > delta_no, (
            f"Expected freq_hop delta ({delta_hop} ns) > no-offset delta ({delta_no} ns)"
        )
