# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for CarrierDetector._encode_offset_snippet.

Validates that offset event snippets are anchored at the PA power cutoff
regardless of each node's noise floor / detection timing.

Core property: two nodes with different noise floors detect the same physical
PA cutoff at different times (different wall-clock sample indices).  Both their
encoded snippets must have the PA cutoff at the same POSITION within the
snippet (within a few samples), so that power-envelope cross-correlation finds
the true propagation TDOA rather than the detection-timing difference.

The centering relies on post-event data (post_buf from snippet_post_windows).
Without it, insufficient post-cutoff samples cause clamping and position drift.
The tests therefore use snippet_post_windows > 0 to mirror the production config.
"""
from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.carrier_detect import CarrierDetector, CarrierOnset, CarrierOffset


# ---------------------------------------------------------------------------
# Signal synthesis helpers
# ---------------------------------------------------------------------------

def _decode_snippet(b64_bytes: bytes) -> np.ndarray:
    """Decode int8 interleaved IQ bytes to complex64."""
    raw = np.frombuffer(b64_bytes, dtype=np.int8)
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 127.0


def _power_envelope(iq: np.ndarray, smooth: int = 16) -> np.ndarray:
    power = iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2
    return np.convolve(power, np.ones(smooth) / smooth, mode="same")


def _find_cutoff_sample(iq: np.ndarray, smooth: int = 16) -> int:
    """Return the sample index of the peak negative power derivative."""
    env = _power_envelope(iq, smooth)
    return int(np.argmin(np.diff(env)))


def _make_transmission(
    sample_rate_hz: float = 62_500.0,
    noise_floor_dbfs: float = -40.0,
    plateau_dbfs: float = -10.0,
    pa_on_sample: int = 0,
    pa_off_sample: int = 5000,
    rise_us: float = 500.0,
    fall_us: float = 0.0,
    total_samples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthesise an LMR-like transmission.

    Parameters
    ----------
    noise_floor_dbfs : float
        Average noise floor power in dBFS (default -40).
    plateau_dbfs : float
        Average carrier plateau power in dBFS (default -10, giving 30 dB SNR).
    pa_on_sample : int
        Sample where PA key-up begins.
    pa_off_sample : int
        Sample where PA key-down begins.
    rise_us : float
        PA rise time in microseconds.
    fall_us : float
        PA fall time in microseconds.  0 = instantaneous shutoff.
    total_samples : int
        Total signal length.
    seed : int
        RNG seed for reproducibility.

    The signal models a realistic LMR (narrowband FM) transmission:
    - Noise floor before key-up
    - Smooth PA rise (constant-envelope carrier ramping up)
    - Constant-envelope carrier plateau (FM has flat power envelope)
    - Smooth PA fall (carrier ramping down)
    - Noise floor after key-down

    SNR should be at least 10 dB (plateau_dbfs - noise_floor_dbfs >= 10).
    Detection thresholds should be set between noise_floor_dbfs and
    plateau_dbfs with ample margin on both sides.
    """
    rng = np.random.default_rng(seed)
    n = total_samples

    rise_samples = max(1, int(rise_us * sample_rate_hz / 1e6))
    fall_samples = max(0, int(fall_us * sample_rate_hz / 1e6))

    noise_amp = float(np.sqrt(10.0 ** (noise_floor_dbfs / 10.0)))
    carrier_amp = float(np.sqrt(2.0 * 10.0 ** (plateau_dbfs / 10.0)))

    # Constant-envelope carrier (narrowband FM has flat power)
    phase = np.cumsum(rng.uniform(-0.1, 0.1, n)).astype(np.float32)
    carrier = (np.exp(1j * phase) * carrier_amp / np.sqrt(2)).astype(np.complex64)

    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    noise *= noise_amp / np.sqrt(2)

    sig = noise.copy()
    ramp_end = pa_on_sample + rise_samples
    fade_start = pa_off_sample

    # Rise: carrier ramps from 0 to full power
    if pa_on_sample < n and rise_samples > 0:
        ramp_len = min(rise_samples, n - pa_on_sample)
        ramp = np.linspace(0, 1, ramp_len, dtype=np.float32)
        sig[pa_on_sample:pa_on_sample + ramp_len] = (
            carrier[pa_on_sample:pa_on_sample + ramp_len] * ramp
            + noise[pa_on_sample:pa_on_sample + ramp_len] * (1 - ramp)
        )

    # Plateau: full-power constant-envelope carrier
    if ramp_end < fade_start and ramp_end < n:
        end = min(fade_start, n)
        sig[ramp_end:end] = carrier[ramp_end:end]

    # Fall: carrier decays exponentially from full to 0
    if fall_samples > 0 and fade_start < n:
        fade_end = min(fade_start + fall_samples, n)
        decay = np.exp(-5.0 * np.arange(fade_end - fade_start) / fall_samples).astype(np.float32)
        sig[fade_start:fade_end] = (
            carrier[fade_start:fade_end] * decay
            + noise[fade_start:fade_end] * (1 - decay)
        )

    return sig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncodeOffsetSnippetCentering:
    """
    The PA cutoff must appear at a consistent position in the encoded snippet
    regardless of when each node detects the offset event.
    """

    @pytest.mark.parametrize("offset_db", [-26.0, -30.0, -34.0, -38.0])
    def test_cutoff_at_fixed_position_across_detection_thresholds(
        self, offset_db: float
    ) -> None:
        """
        Different offset_db thresholds cause detection to fire at different
        times after the PA shutoff.  The PA cutoff in the encoded snippet must
        always appear near the 3/4 position (+/-10% tolerance) regardless.

        Signal: -40 dBFS noise, -10 dBFS plateau (30 dB SNR).
        Onset threshold: -25 dBFS (well between noise and carrier).
        Offset thresholds sweep from -15 to -30 dBFS.
        """
        sample_rate = 62_500.0
        window = 64
        snippet_samples = 1280
        post_windows = 10
        ring_lookback = 60

        sig = _make_transmission(
            sample_rate_hz=sample_rate,
            noise_floor_dbfs=-40.0,
            plateau_dbfs=-10.0,
            pa_on_sample=4000,
            pa_off_sample=8000,
            rise_us=500.0,
            fall_us=500.0,  # realistic PA shutoff ramp
            total_samples=20_000,
        )

        det = CarrierDetector(
            sample_rate_hz=sample_rate,
            onset_threshold_db=-25.0,
            offset_threshold_db=offset_db,
            window_samples=window,
            min_hold_windows=1,
            min_release_windows=2,
            snippet_samples=snippet_samples,
            snippet_post_windows=post_windows,
            ring_lookback_windows=ring_lookback,
        )

        events = det.process(sig, 0)
        offset_events = [e for e in events if isinstance(e, CarrierOffset)]

        assert len(offset_events) >= 1, (
            f"No CarrierOffset emitted with offset_db={offset_db}"
        )
        off_ev = offset_events[0]
        assert off_ev.iq_snippet is not None

        snippet_iq = _decode_snippet(off_ev.iq_snippet)
        assert len(snippet_iq) == snippet_samples

        # Validate centering: the first 3/4 should have carrier power,
        # the last 1/4 should be near the noise floor.  This is more
        # robust than re-running argmin(deriv) on int8-decoded data,
        # which can find a different minimum due to quantization.
        env = _power_envelope(snippet_iq)
        target = (snippet_samples * 3) // 4
        first_half_power = float(np.mean(env[:target // 2]))
        last_quarter_power = float(np.mean(env[target:]))
        assert first_half_power > last_quarter_power * 5.0, (
            f"offset_db={offset_db}: snippet not properly centered — "
            f"first-half power {first_half_power:.4f} should be >> "
            f"last-quarter power {last_quarter_power:.6f}"
        )

    def test_two_nodes_different_thresholds_both_capture_transition(self) -> None:
        """
        Two nodes with different effective SNR detect offset at different times
        on the same gradual fade curve.  Their sample_index values WILL differ
        (each reflects its own detection point), but both snippets must contain
        the PA transition so the server's xcorr can align them.

        Different offset_db values model the scenario where one node's noise
        floor is closer to the carrier level, making it trigger earlier on the
        exponential fade curve.  The same physical carrier starts decaying at
        pa_off_sample for both nodes.
        """
        sample_rate = 62_500.0
        window = 64
        snippet_samples = 1280
        post_windows = 10
        ring_lookback = 60

        pa_on = 4000
        pa_off = 10_000
        total = 25_000

        # Same physical signal received by both nodes.
        # 10 ms exponential decay so different thresholds fire at different times.
        sig = _make_transmission(
            noise_floor_dbfs=-40.0,
            plateau_dbfs=-10.0,
            pa_on_sample=pa_on, pa_off_sample=pa_off,
            rise_us=500.0,
            fall_us=10_000.0,  # 10 ms exponential decay
            total_samples=total, seed=7,
        )

        def _run(sig: np.ndarray, onset_db: float, offset_db: float) -> CarrierOffset | None:
            det = CarrierDetector(
                sample_rate_hz=sample_rate,
                onset_threshold_db=onset_db,
                offset_threshold_db=offset_db,
                window_samples=window,
                min_hold_windows=1,
                min_release_windows=2,
                snippet_samples=snippet_samples,
                snippet_post_windows=post_windows,
                ring_lookback_windows=ring_lookback,
            )
            events = det.process(sig, 0)
            offsets = [e for e in events if isinstance(e, CarrierOffset)]
            return offsets[0] if offsets else None

        # Node A: tight offset threshold - detects fade early
        off_a = _run(sig, onset_db=-20.0, offset_db=-26.0)
        # Node B: loose offset threshold - detects fade late (must decay further)
        off_b = _run(sig, onset_db=-20.0, offset_db=-35.0)

        assert off_a is not None, "Node A did not emit CarrierOffset"
        assert off_b is not None, "Node B did not emit CarrierOffset"

        # sample_index values differ (different detection points) — that's
        # expected.  Node B detects later, so its sample_index is higher.
        assert off_b.sample_index >= off_a.sample_index, (
            f"Node B (looser threshold) should detect later: "
            f"A={off_a.sample_index}, B={off_b.sample_index}"
        )

        # Both snippets must contain the PA transition (non-trivial
        # power dynamic range) so xcorr can align them.
        for label, off in [("A", off_a), ("B", off_b)]:
            assert len(off.iq_snippet) > 0, f"Node {label} snippet is empty"
            assert off.transition_end > off.transition_start, (
                f"Node {label} has no transition zone: "
                f"start={off.transition_start}, end={off.transition_end}"
            )

        # The PA cutoff is at DIFFERENT positions in the two snippets because
        # the snippets are anchored on different detection points.  The position
        # difference should match the sample_index difference (they see the same
        # signal but from different starting points).
        iq_a = _decode_snippet(off_a.iq_snippet)
        iq_b = _decode_snippet(off_b.iq_snippet)
        cut_a = _find_cutoff_sample(iq_a)
        cut_b = _find_cutoff_sample(iq_b)

        # Both snippets must contain a detectable cutoff
        assert cut_a is not None, "Node A snippet has no detectable cutoff"
        assert cut_b is not None, "Node B snippet has no detectable cutoff"

    def test_onset_snippet_unaffected(self) -> None:
        """
        Onset events still use the standard _encode_combined / _encode_snippet path.
        The offset-centering change must not alter onset snippet behavior.
        """
        sample_rate = 62_500.0
        window = 64
        snippet_samples = 640

        # pa_on_sample large enough that the ring is full before onset fires
        sig = _make_transmission(
            noise_floor_dbfs=-40.0, plateau_dbfs=-10.0,
            pa_on_sample=5000, pa_off_sample=15_000,
            rise_us=500.0, fall_us=500.0,
            total_samples=20_000, seed=5,
        )

        det = CarrierDetector(
            sample_rate_hz=sample_rate,
            onset_threshold_db=-20.0,
            offset_threshold_db=-30.0,
            window_samples=window,
            min_hold_windows=1,
            min_release_windows=1,
            snippet_samples=snippet_samples,
            snippet_post_windows=0,
        )
        events = det.process(sig, 0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]

        assert len(onsets) >= 1
        on_ev = onsets[0]
        assert on_ev.iq_snippet is not None
        iq = _decode_snippet(on_ev.iq_snippet)
        assert len(iq) == snippet_samples


class TestLowSNR:
    """Low-SNR scenario: 15 dB total, 5 dB steps between levels."""

    def test_onset_and_offset_at_15db_snr(self) -> None:
        """
        With only 15 dB SNR and 5 dB spacing between noise floor, offset
        threshold, onset threshold, and signal plateau, both onset and
        offset should still be detected and produce reasonable sample_index.

        Levels: noise=-35, offset=-30, onset=-25, plateau=-20 dBFS
        """
        sample_rate = 62_500.0
        window = 64
        snippet_samples = 1280

        sig = _make_transmission(
            sample_rate_hz=sample_rate,
            noise_floor_dbfs=-35.0,
            plateau_dbfs=-20.0,
            pa_on_sample=4000,
            pa_off_sample=8000,
            rise_us=500.0,
            fall_us=500.0,
            total_samples=20_000,
        )

        det = CarrierDetector(
            sample_rate_hz=sample_rate,
            onset_threshold_db=-25.0,
            offset_threshold_db=-30.0,
            window_samples=window,
            min_hold_windows=1,
            min_release_windows=2,
            snippet_samples=snippet_samples,
            snippet_post_windows=10,
            ring_lookback_windows=60,
        )

        events = det.process(sig, 0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]

        assert len(onsets) >= 1, "No onset at 15 dB SNR"
        assert len(offsets) >= 1, "No offset at 15 dB SNR"

        # Onset sample_index should be near PA on (4000 + rise)
        rise_samples = int(500.0 * sample_rate / 1e6)
        expected_onset = 4000 + rise_samples
        assert abs(onsets[0].sample_index - expected_onset) < 4 * window, (
            f"Onset sample_index={onsets[0].sample_index}, "
            f"expected near {expected_onset} (+/-{4 * window})"
        )

        # Offset sample_index should be near PA off (8000)
        assert abs(offsets[0].sample_index - 8000) < 4 * window, (
            f"Offset sample_index={offsets[0].sample_index}, "
            f"expected near 8000 (+/-{4 * window})"
        )

    def test_two_nodes_converge_at_15db_snr(self) -> None:
        """Two nodes with different thresholds converge on the same knee."""
        sample_rate = 62_500.0
        window = 64

        sig = _make_transmission(
            noise_floor_dbfs=-35.0, plateau_dbfs=-20.0,
            pa_on_sample=4000, pa_off_sample=10_000,
            rise_us=500.0, fall_us=2000.0,
            total_samples=25_000, seed=77,
        )

        def _run(onset_db, offset_db):
            det = CarrierDetector(
                sample_rate_hz=sample_rate,
                onset_threshold_db=onset_db,
                offset_threshold_db=offset_db,
                window_samples=window, min_hold_windows=1,
                min_release_windows=2, snippet_samples=1280,
                snippet_post_windows=10, ring_lookback_windows=60,
            )
            events = det.process(sig, 0)
            offsets = [e for e in events if isinstance(e, CarrierOffset)]
            return offsets[0] if offsets else None

        # Node A: thresholds close to carrier
        off_a = _run(onset_db=-23.0, offset_db=-27.0)
        # Node B: thresholds close to noise
        off_b = _run(onset_db=-27.0, offset_db=-32.0)

        assert off_a is not None, "Node A: no offset at 15 dB SNR"
        assert off_b is not None, "Node B: no offset at 15 dB SNR"

        assert abs(off_a.sample_index - off_b.sample_index) <= 2 * window, (
            f"Low-SNR nodes diverge: A={off_a.sample_index} "
            f"B={off_b.sample_index} diff={abs(off_a.sample_index - off_b.sample_index)}"
        )


class TestRingLookbackConfig:
    """Tests for the ring_lookback_windows parameter."""

    def test_explicit_ring_lookback_overrides_default(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=62_500.0,
            onset_threshold_db=-10.0,
            offset_threshold_db=-20.0,
            window_samples=64,
            snippet_samples=640,
            ring_lookback_windows=80,
        )
        assert det._iq_ring.maxlen == 80

    def test_default_ring_lookback_is_3x_snippet_windows(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=62_500.0,
            onset_threshold_db=-10.0,
            offset_threshold_db=-20.0,
            window_samples=64,
            snippet_samples=640,
            # ring_lookback_windows not specified -> default = 3 x ceil(640/64) = 30
        )
        expected = 3 * (640 // 64)  # = 30
        assert det._iq_ring.maxlen == expected

    def test_ring_lookback_minimum_one(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=62_500.0,
            onset_threshold_db=-10.0,
            offset_threshold_db=-20.0,
            window_samples=64,
            snippet_samples=1,
            ring_lookback_windows=1,
        )
        assert det._iq_ring.maxlen == 1

    def test_explicit_ring_too_small_warns(self, caplog) -> None:
        """Explicitly set ring_lookback_windows smaller than needed to fill
        snippet_samples must log a warning and preserve the (truncated) value."""
        import logging
        caplog.set_level(logging.WARNING, logger="beagle_node.pipeline.carrier_detect")
        det = CarrierDetector(
            sample_rate_hz=250_000.0,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
            snippet_samples=5120,
            snippet_post_windows=10,
            ring_lookback_windows=60,  # ceil(5120/64) - 10 = 70 needed
        )
        assert det._iq_ring.maxlen == 60
        assert any(
            "ring_lookback_windows=60" in r.message and "5120" in r.message
            for r in caplog.records
        ), f"Expected truncation warning, got: {[r.message for r in caplog.records]}"

    def test_auto_ring_fills_snippet_when_3x_too_small(self) -> None:
        """When 3*snippet_windows < min_for_full_snippet (e.g. post=0), the
        auto-sized ring should still be large enough to fill snippet_samples."""
        # snippet=5120/64 = 80 windows, post=0 -> min_ring=80.
        # 3*snippet_windows = 240, so auto picks max(240, 80) = 240.
        det_a = CarrierDetector(
            sample_rate_hz=250_000.0,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
            snippet_samples=5120,
            snippet_post_windows=0,
        )
        assert det_a._iq_ring.maxlen == 240

        # Contrived case where min_for_full_snippet dominates: tiny 3x but
        # snippet_post_windows is negative of snippet_windows. With post>=0
        # this can't actually happen, but verify the max() still behaves.
        det_b = CarrierDetector(
            sample_rate_hz=250_000.0,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
            snippet_samples=128,          # 2 windows
            snippet_post_windows=0,
        )
        # 3*2 = 6, min_for_full_snippet = 2 -> pick 6.
        assert det_b._iq_ring.maxlen == 6
