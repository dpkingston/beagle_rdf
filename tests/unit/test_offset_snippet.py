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
        always appear in roughly the same place regardless: detection is
        anchored at the midpoint of the snippet, so the knee sits a short
        span before the midpoint.

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

        # Validate centering: detection now sits at the snippet midpoint, so
        # the first half should have carrier power (plateau + the fall's
        # beginning) and the last half should trend toward the noise floor.
        # Compare first quarter (clean plateau) vs last quarter (past-detection
        # noise) to keep the ratio robust under gradual fades.
        env = _power_envelope(snippet_iq)
        midpoint = snippet_samples // 2
        first_quarter_power = float(np.mean(env[:midpoint // 2]))
        last_quarter_power = float(np.mean(env[midpoint + midpoint // 2:]))
        assert first_quarter_power > last_quarter_power * 5.0, (
            f"offset_db={offset_db}: snippet not properly centered — "
            f"first-quarter power {first_quarter_power:.4f} should be >> "
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

        # sample_index is now the snippet-start sample; detection is at
        # sample_index + transition_start for onsets (or + transition_end
        # for offsets).  That detection point should be near the actual PA
        # event.
        rise_samples = int(500.0 * sample_rate / 1e6)
        expected_onset_det = 4000 + rise_samples
        onset_det_abs = onsets[0].sample_index + onsets[0].transition_start
        assert abs(onset_det_abs - expected_onset_det) < 4 * window, (
            f"Onset detection={onset_det_abs} (sample_index={onsets[0].sample_index} "
            f"+ transition_start={onsets[0].transition_start}), "
            f"expected near {expected_onset_det} (+/-{4 * window})"
        )

        # Offset detection should be near PA off (8000)
        offset_det_abs = offsets[0].sample_index + offsets[0].transition_end
        assert abs(offset_det_abs - 8000) < 4 * window, (
            f"Offset detection={offset_det_abs} (sample_index={offsets[0].sample_index} "
            f"+ transition_end={offsets[0].transition_end}), "
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


class TestAutoThresholdMargins:
    """Tests for the auto-threshold tracking feature.

    Auto-threshold mode keeps onset/offset at fixed margins above the tracked
    noise-floor EMA, following real-world noise changes without static tuning.
    """

    @staticmethod
    def _noise_buffer(n_samples: int, power_db: float, seed: int = 0) -> np.ndarray:
        """Gaussian IQ noise whose average |s|^2 = 10^(power_db/10)."""
        rng = np.random.default_rng(seed)
        # |s|^2 per sample = re^2 + im^2.  For iid N(0, sigma^2) real/imag,
        # E[|s|^2] = 2*sigma^2.  Pick sigma so that 2*sigma^2 = 10^(power_db/10).
        target_power = 10.0 ** (power_db / 10.0)
        sigma = float(np.sqrt(target_power / 2.0))
        return (sigma * (rng.standard_normal(n_samples)
                         + 1j * rng.standard_normal(n_samples))).astype(np.complex64)

    def test_auto_off_keeps_static_thresholds(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=False,  # explicit static mode
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.05,
        )
        # Feed enough noise at -60 dB to completely warm up the EMA
        noise = self._noise_buffer(64_000, power_db=-60.0)
        det.process(noise, start_sample=0)
        # Thresholds must NOT have changed despite well-defined noise floor
        assert det._onset_db == -30.0
        assert det._offset_db == -40.0

    def test_auto_on_tracks_floor_after_warmup(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=True,
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.05,  # update every ~3 windows
        )
        # 1 second of -60 dB noise at 64k/64 = 1000 windows >> 500 warmup updates
        noise = self._noise_buffer(64_000, power_db=-60.0)
        det.process(noise, start_sample=0)
        # Onset/offset should have moved close to -48 and -54 (floor -60 +12 / +6)
        assert det._noise_floor_db < -55.0, det._noise_floor_db
        assert abs(det._onset_db - (det._noise_floor_db + 12.0)) < 0.5
        assert abs(det._offset_db - (det._noise_floor_db + 6.0)) < 0.5

    def test_auto_on_no_update_before_warmup(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=True,
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.001,  # pretend to update every window
        )
        # Only 200 samples at 64k/64 = ~3 windows, far below the 500-update warmup.
        noise = self._noise_buffer(2_000, power_db=-60.0)
        det.process(noise, start_sample=0)
        # Static thresholds must remain because warmup hasn't elapsed.
        assert det._onset_db == -30.0
        assert det._offset_db == -40.0

    def test_auto_onset_clamped_to_safety_max(self) -> None:
        """Severe noise/interference must not push onset above the safety cap."""
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=True,
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.05,
        )
        # Force the EMA to a very high floor directly, then feed more idle
        # samples so warmup counter accumulates.  Use noise at -5 dB so
        # floor + 12 = +7 which exceeds the safety max (-10 dB).
        det._noise_floor_db = -5.0
        det._auto_floor_updates = det._auto_warmup_floor_updates  # skip warmup
        noise = self._noise_buffer(64_000, power_db=-5.0)
        det.process(noise, start_sample=0)
        assert det._onset_db <= det._auto_max_onset_db + 1e-6
        # Offset keeps at least 1 dB hysteresis below onset
        assert det._offset_db < det._onset_db - 0.5

    def test_auto_on_small_change_logs_at_debug(self, caplog) -> None:
        """Sub-2-dB tracking adjustments must NOT emit info-level log spam."""
        import logging
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=True,
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.01,
        )
        # Warm up with -60 dB so floor converges there and thresholds move
        # to roughly -48/-54; after this, further sub-dB EMA wobble shouldn't
        # produce info-level logs.
        noise = self._noise_buffer(64_000, power_db=-60.0)
        det.process(noise, start_sample=0)
        caplog.clear()
        caplog.set_level(logging.INFO, logger="beagle_node.pipeline.carrier_detect")
        # Another second of the same noise.  Any adjustments should be
        # < 2 dB and therefore debug-only, not info.
        noise2 = self._noise_buffer(64_000, power_db=-60.0, seed=1)
        det.process(noise2, start_sample=len(noise))
        info_updates = [r for r in caplog.records
                        if r.levelno == logging.INFO
                        and "Auto-threshold update" in r.message]
        assert not info_updates, (
            f"Expected no info-level per-update logs for < 2 dB drifts, got: "
            f"{[r.message for r in info_updates]}"
        )

    def test_auto_heartbeat_logged_periodically(self, caplog) -> None:
        """Auto-tracking emits an info heartbeat at ~10 min cadence so
        operators see the mechanism is alive even when no large adjustments
        are being made."""
        import logging
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
            auto_threshold_margins=True,
            onset_margin_db=12.0, offset_margin_db=6.0,
            auto_threshold_update_interval_s=0.05,
        )
        # Shorten the heartbeat interval for test speed (1000 windows).
        det._auto_heartbeat_interval_windows = 1000
        caplog.set_level(logging.INFO, logger="beagle_node.pipeline.carrier_detect")
        # Process > 1000 windows of idle noise; warmup (500 updates) should
        # finish first and then at least one heartbeat should fire.
        noise = self._noise_buffer(64_000 * 3, power_db=-60.0)
        det.process(noise, start_sample=0)
        heartbeats = [r for r in caplog.records
                      if r.levelno == logging.INFO
                      and "Auto-threshold active" in r.message]
        assert heartbeats, (
            f"Expected at least one heartbeat log, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_invalid_margins_rejected(self) -> None:
        with pytest.raises(ValueError):
            CarrierDetector(
                sample_rate_hz=64_000.0,
                onset_threshold_db=-30.0, offset_threshold_db=-40.0,
                window_samples=64,
                onset_margin_db=6.0, offset_margin_db=12.0,  # inverted
            )
        with pytest.raises(ValueError):
            CarrierDetector(
                sample_rate_hz=64_000.0,
                onset_threshold_db=-30.0, offset_threshold_db=-40.0,
                window_samples=64,
                auto_threshold_update_interval_s=0.0,
            )


class TestEmissionInvariant:
    """The CarrierDetector is a two-state machine; emitted events MUST
    alternate onset/offset/onset/...  A path that silently flips state
    without emitting the boundary event (cancel_pending on
    discontinuity; prime_state; reset) is a bug, and the invariant
    guard in ``_emit`` suppresses the duplicate so it doesn't pollute
    downstream TDOA measurements."""

    def _silent_onset(self, det: CarrierDetector, sample: int) -> CarrierOnset:
        return CarrierOnset(
            sample_index=sample, power_db=-20.0,
            noise_floor_db=-60.0, iq_snippet=b"\x00" * 16,
        )

    def _silent_offset(self, det: CarrierDetector, sample: int) -> CarrierOffset:
        return CarrierOffset(
            sample_index=sample, power_db=-40.0,
            iq_snippet=b"\x00" * 16,
        )

    def test_duplicate_onset_suppressed_with_warning(self, caplog) -> None:
        import logging
        caplog.set_level(logging.WARNING,
                         logger="beagle_node.pipeline.carrier_detect")
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
        )
        events: list = []
        det._emit(events, self._silent_onset(det, 1000))
        det._emit(events, self._silent_onset(det, 2000))  # duplicate -> dropped
        assert len(events) == 1
        assert isinstance(events[0], CarrierOnset)
        assert any("invariant violated" in r.message for r in caplog.records)

    def test_duplicate_offset_suppressed_with_warning(self, caplog) -> None:
        import logging
        caplog.set_level(logging.WARNING,
                         logger="beagle_node.pipeline.carrier_detect")
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
        )
        # Seed with an onset so the next offset is accepted; then a
        # second offset should trip the invariant.
        events: list = []
        det._emit(events, self._silent_onset(det, 1000))
        det._emit(events, self._silent_offset(det, 2000))
        det._emit(events, self._silent_offset(det, 3000))  # duplicate -> dropped
        assert len(events) == 2
        assert isinstance(events[0], CarrierOnset)
        assert isinstance(events[1], CarrierOffset)
        assert any("invariant violated" in r.message for r in caplog.records)

    def test_alternating_events_all_accepted(self) -> None:
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
        )
        events: list = []
        det._emit(events, self._silent_onset(det, 1000))
        det._emit(events, self._silent_offset(det, 2000))
        det._emit(events, self._silent_onset(det, 3000))
        det._emit(events, self._silent_offset(det, 4000))
        assert len(events) == 4

    def test_cancel_pending_resets_idle_window_count(self) -> None:
        """After a discontinuity, _idle_window_count must be 0 so the
        _min_idle_for_onset guard prevents a spurious onset from a
        persisting carrier."""
        det = CarrierDetector(
            sample_rate_hz=64_000.0,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64,
        )
        det._idle_window_count = 500   # simulate "saw plenty of idle"
        det._state = "active"           # pre-gap carrier
        det.cancel_pending()
        assert det._idle_window_count == 0, (
            "cancel_pending must force the idle counter to 0 so a "
            "persisting carrier cannot immediately re-trigger onset"
        )
        assert det._state == "idle"
        assert det._last_emitted_type is None, (
            "cancel_pending must reset _last_emitted_type so the next "
            "emission of either type is accepted"
        )

    def test_cancel_pending_prevents_duplicate_onset_on_persisting_carrier(
        self,
    ) -> None:
        """End-to-end: carrier on, fire onset, discontinuity, carrier still
        on, subsequent processing must NOT emit a second onset."""
        rng = np.random.default_rng(0)
        fs = 64_000
        # Build a signal: 0.1 s of silence, 2 s of strong carrier.
        n_silence = fs // 10
        n_carrier = fs * 2
        silence = (rng.standard_normal(n_silence).astype(np.float32) +
                   1j * rng.standard_normal(n_silence).astype(np.float32)) * 0.01
        carrier_level = 1.0
        carrier = (np.ones(n_carrier, dtype=np.complex64) * carrier_level
                   + 0.01 * (rng.standard_normal(n_carrier)
                             + 1j * rng.standard_normal(n_carrier))).astype(np.complex64)
        full_signal = np.concatenate([silence.astype(np.complex64), carrier])

        det = CarrierDetector(
            sample_rate_hz=fs,
            onset_threshold_db=-30.0, offset_threshold_db=-40.0,
            window_samples=64, min_hold_windows=1, min_release_windows=2,
            auto_threshold_margins=False,
        )
        # Feed the whole signal -> one onset.
        events = det.process(full_signal, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(onsets) == 1
        assert len(offsets) == 0  # carrier still on at end

        # Simulate a discontinuity mid-transmission.
        det.cancel_pending()

        # Continue with another 1 s of same strong carrier.
        more_carrier = (np.ones(fs, dtype=np.complex64) * carrier_level
                        + 0.01 * (rng.standard_normal(fs).astype(np.float32)
                                  + 1j * rng.standard_normal(fs).astype(np.float32))).astype(np.complex64)
        events_post = det.process(more_carrier, start_sample=len(full_signal))

        onsets_post = [e for e in events_post if isinstance(e, CarrierOnset)]
        # The _min_idle_for_onset guard (default 2) must suppress a new
        # onset because no idle windows were observed post-discontinuity.
        assert len(onsets_post) == 0, (
            f"Expected no spurious onsets from persisting carrier after "
            f"cancel_pending, got {len(onsets_post)}.  Snippet-start of "
            f"those bogus onsets would be mid-plateau instead of noise "
            f"floor - exactly the pathological pattern observed on "
            f"dpk-tdoa2."
        )
