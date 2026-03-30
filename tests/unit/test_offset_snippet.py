# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
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
    pa_on_sample: int = 0,
    pa_off_sample: int = 5000,
    noise_amplitude: float = 0.01,
    carrier_amplitude: float = 1.0,
    ramp_samples: int = 32,
    fade_samples: int = 0,
    total_samples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthesise a transmission with a configurable PA shutoff shape.

    The signal has:
      - Noise before pa_on_sample
      - Ramped-up carrier from pa_on_sample to pa_on_sample + ramp_samples
      - Full AM-QPSK carrier plateau from ramp_end to pa_off_sample
      - PA power cutoff at pa_off_sample: instantaneous if fade_samples=0,
        otherwise an exponential decay over fade_samples (simulates realistic
        LMR fade curves where the power drops gradually to the noise floor)
      - Noise after the fade

    fade_samples > 0 is needed to produce different detection times between
    nodes with different noise floors / SNR: nodes with higher effective SNR
    (carrier amplitude relative to noise) cross the offset threshold later on
    the decay curve than nodes with lower SNR.

    The carrier uses amplitude-modulated QPSK so the power envelope has
    texture for cross-correlation.
    """
    rng = np.random.default_rng(seed)
    n = total_samples

    # AM-QPSK carrier
    bits_i = rng.integers(0, 2, n) * 2 - 1
    bits_q = rng.integers(0, 2, n) * 2 - 1
    qpsk = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2)
    am_raw = np.abs(rng.standard_normal(n + 16))
    am_smooth = np.convolve(am_raw, np.ones(16) / 16, mode="valid")[:n]
    am_env = ((am_smooth - am_smooth.min()) / (am_smooth.max() - am_smooth.min()) * 0.6 + 0.4)
    carrier = (qpsk * am_env.astype(np.float32)).astype(np.complex64) * carrier_amplitude

    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    noise *= noise_amplitude / np.sqrt(2)

    sig = noise.copy()
    ramp_end = pa_on_sample + ramp_samples
    if pa_on_sample < n:
        ramp_len = min(ramp_samples, n - pa_on_sample)
        ramp = np.linspace(0, 1, ramp_len, dtype=np.float32)
        sig[pa_on_sample:pa_on_sample + ramp_len] = carrier[pa_on_sample:pa_on_sample + ramp_len] * ramp
    if ramp_end < pa_off_sample and ramp_end < n:
        end = min(pa_off_sample, n)
        sig[ramp_end:end] = carrier[ramp_end:end]

    # PA cutoff: instantaneous or gradual exponential decay
    if fade_samples > 0 and pa_off_sample < n:
        fade_end = min(pa_off_sample + fade_samples, n)
        decay = np.exp(-5.0 * np.arange(fade_end - pa_off_sample) / fade_samples).astype(np.float32)
        sig[pa_off_sample:fade_end] = carrier[pa_off_sample:fade_end] * decay + noise[pa_off_sample:fade_end]

    return sig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncodeOffsetSnippetCentering:
    """
    The PA cutoff must appear at a consistent position in the encoded snippet
    regardless of when each node detects the offset event.
    """

    @pytest.mark.parametrize("offset_db", [-8.0, -12.0, -18.0, -25.0])
    def test_cutoff_at_fixed_position_across_detection_thresholds(
        self, offset_db: float
    ) -> None:
        """
        Different offset_db thresholds cause detection to fire at different
        times after the PA shutoff.  The PA cutoff in the encoded snippet must
        always appear near the 3/4 position (±10% tolerance) regardless.

        Requires snippet_post_windows > 0 so the deferred emission path
        provides post-cutoff IQ for reliable centering.
        """
        sample_rate = 62_500.0
        window = 64
        snippet_samples = 1280
        post_windows = 10  # critical: provides post-cutoff IQ for centering
        ring_lookback = 60

        # Enough signal for any threshold to fire: long post-cutoff tail
        sig = _make_transmission(
            sample_rate_hz=sample_rate,
            pa_on_sample=4000,  # ring fills before PA fires
            pa_off_sample=8000,
            noise_amplitude=0.005,
            carrier_amplitude=1.0,
            total_samples=20_000,
        )

        det = CarrierDetector(
            sample_rate_hz=sample_rate,
            onset_threshold_db=-6.0,
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

        cutoff_pos = _find_cutoff_sample(snippet_iq)
        target = (snippet_samples * 3) // 4  # expected: cutoff at 3/4 from start

        # Tolerance: ±10% of snippet_samples (= ±128 samples)
        tolerance = snippet_samples * 0.10
        assert abs(cutoff_pos - target) <= tolerance, (
            f"offset_db={offset_db}: PA cutoff at {cutoff_pos}, "
            f"expected near {target} (±{tolerance:.0f})"
        )

    def test_two_nodes_different_thresholds_same_cutoff_position(self) -> None:
        """
        Two nodes with different effective SNR detect offset at different times
        on the same gradual fade curve.  Both their snippets must have the PA
        cutoff at the same position (within 8 samples).

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
        fade = 640   # 10 ms exponential decay (exp(-5) ≈ 0.007 at end)
        total = 25_000

        # Same physical signal received by both nodes
        sig = _make_transmission(
            pa_on_sample=pa_on, pa_off_sample=pa_off,
            fade_samples=fade,
            noise_amplitude=0.005,
            carrier_amplitude=1.0,
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

        # Node A: tight threshold — detects fade early (crosses -8 dBFS quickly)
        off_a = _run(sig, onset_db=-3.0, offset_db=-8.0)
        # Node B: loose threshold — detects fade late (must decay to -25 dBFS)
        off_b = _run(sig, onset_db=-3.0, offset_db=-25.0)

        assert off_a is not None, "Node A did not emit CarrierOffset"
        assert off_b is not None, "Node B did not emit CarrierOffset"

        # With derivative-peak refinement, sample_index IS the power-derivative peak
        # for both nodes.  Both see the same physical signal, so the same peak must
        # be identified — they should agree within a few samples.
        assert abs(off_a.sample_index - off_b.sample_index) <= window, (
            f"Refined sample_index differs by "
            f"{abs(off_a.sample_index - off_b.sample_index)} samples: "
            f"node_A={off_a.sample_index}, node_B={off_b.sample_index}; "
            f"derivative peak should converge on the same physical PA shutoff"
        )

        iq_a = _decode_snippet(off_a.iq_snippet)
        iq_b = _decode_snippet(off_b.iq_snippet)
        cut_a = _find_cutoff_sample(iq_a)
        cut_b = _find_cutoff_sample(iq_b)

        assert abs(cut_a - cut_b) <= 8, (
            f"PA cutoff positions differ by {abs(cut_a - cut_b)} samples: "
            f"node_A={cut_a}, node_B={cut_b} "
            f"(detection: A={off_a.sample_index}, B={off_b.sample_index})"
        )

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
            pa_on_sample=5000, pa_off_sample=15_000,
            noise_amplitude=0.005, carrier_amplitude=1.0,
            total_samples=20_000, seed=5,
        )

        det = CarrierDetector(
            sample_rate_hz=sample_rate,
            onset_threshold_db=-6.0,
            offset_threshold_db=-20.0,
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
            # ring_lookback_windows not specified → default = 3 × ceil(640/64) = 30
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
