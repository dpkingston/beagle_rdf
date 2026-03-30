# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/carrier_detect.py."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.pipeline.carrier_detect import CarrierDetector, CarrierOnset, CarrierOffset

RATE = 48_000.0
ONSET  = -20.0   # dBFS
OFFSET = -30.0   # dBFS


def make_detector(**kwargs) -> CarrierDetector:
    defaults = dict(
        sample_rate_hz=RATE,
        onset_threshold_db=ONSET,
        offset_threshold_db=OFFSET,
        window_samples=64,
        snippet_post_windows=0,   # FSM tests expect immediate (non-deferred) emission
    )
    defaults.update(kwargs)
    return CarrierDetector(**defaults)


def _noise(n: int, power_db: float, rng: np.random.Generator) -> np.ndarray:
    """Complex noise at given power level (dBFS)."""
    amp = 10 ** (power_db / 20.0)
    return (amp / np.sqrt(2)) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)


def _carrier(n: int, power_db: float) -> np.ndarray:
    """Constant-power carrier at given level."""
    amp = 10 ** (power_db / 20.0)
    return (amp * np.ones(n, dtype=np.complex64))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_invalid_thresholds():
    with pytest.raises(ValueError, match="offset_threshold_db"):
        CarrierDetector(RATE, onset_threshold_db=-20, offset_threshold_db=-15)


def test_invalid_window():
    with pytest.raises(ValueError, match="window_samples"):
        CarrierDetector(RATE, window_samples=0)


def test_invalid_min_hold():
    with pytest.raises(ValueError, match="min_hold_windows"):
        CarrierDetector(RATE, min_hold_windows=0)


def test_invalid_min_release():
    with pytest.raises(ValueError, match="min_release_windows"):
        CarrierDetector(RATE, min_release_windows=0)


def test_initial_state_idle():
    det = make_detector()
    assert det.state == "idle"


# ---------------------------------------------------------------------------
# No-signal: no events
# ---------------------------------------------------------------------------

def test_no_events_on_noise():
    """Low-power noise should produce no events."""
    rng = np.random.default_rng(0)
    det = make_detector()
    iq = _noise(4096, power_db=-60.0, rng=rng)
    events = det.process(iq, start_sample=0)
    assert events == []
    assert det.state == "idle"


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def test_onset_detected():
    """High-power carrier should trigger a CarrierOnset."""
    det = make_detector()
    iq = _carrier(4096, power_db=-10.0)
    events = det.process(iq, start_sample=0)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) == 1
    assert det.state == "active"


def test_onset_sample_index_reasonable():
    """Onset sample index must fall within the buffer."""
    det = make_detector()
    start = 100_000
    iq = _carrier(4096, power_db=-10.0)
    events = det.process(iq, start_sample=start)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) == 1
    assert start <= onsets[0].sample_index < start + len(iq)


# ---------------------------------------------------------------------------
# Offset detection
# ---------------------------------------------------------------------------

def test_offset_detected_after_onset():
    """Carrier then noise should produce onset + offset."""
    rng = np.random.default_rng(1)
    det = make_detector()

    iq_on  = _carrier(4096, power_db=-10.0)
    iq_off = _noise(4096, power_db=-60.0, rng=rng)

    ev1 = det.process(iq_on,  start_sample=0)
    ev2 = det.process(iq_off, start_sample=4096)
    events = ev1 + ev2

    onsets  = [e for e in events if isinstance(e, CarrierOnset)]
    offsets = [e for e in events if isinstance(e, CarrierOffset)]
    assert len(onsets)  == 1
    assert len(offsets) == 1
    assert offsets[0].sample_index > onsets[0].sample_index
    assert det.state == "idle"


# ---------------------------------------------------------------------------
# Hysteresis - no chatter between thresholds
# ---------------------------------------------------------------------------

def test_hysteresis_no_chatter():
    """
    Signal between onset and offset thresholds while active should NOT
    trigger an offset event.
    """
    rng = np.random.default_rng(2)
    det = make_detector(onset_threshold_db=-20, offset_threshold_db=-40)

    # Get into active state
    iq_on = _carrier(512, power_db=-10.0)
    det.process(iq_on, start_sample=0)
    assert det.state == "active"

    # Signal at -30 dB - between -20 (onset) and -40 (offset) thresholds
    iq_mid = _carrier(4096, power_db=-30.0)
    events = det.process(iq_mid, start_sample=512)
    offsets = [e for e in events if isinstance(e, CarrierOffset)]
    assert offsets == [], "Should not deactivate between thresholds"
    assert det.state == "active"


# ---------------------------------------------------------------------------
# Multiple events in one buffer
# ---------------------------------------------------------------------------

def test_multiple_on_off_cycles():
    """Two on/off cycles in one buffer produce two onsets and two offsets."""
    rng = np.random.default_rng(3)
    det = make_detector(window_samples=32)

    W = 32
    seg_on  = _carrier(W * 4, power_db=-10.0)
    seg_off = _noise(W * 4,   power_db=-60.0, rng=rng)
    iq = np.concatenate([seg_on, seg_off, seg_on, seg_off])

    events = det.process(iq, start_sample=0)
    onsets  = [e for e in events if isinstance(e, CarrierOnset)]
    offsets = [e for e in events if isinstance(e, CarrierOffset)]
    assert len(onsets)  == 2
    assert len(offsets) == 2


# ---------------------------------------------------------------------------
# State persists across buffer calls
# ---------------------------------------------------------------------------

def test_active_state_persists_across_buffers():
    det = make_detector()
    iq_on = _carrier(4096, power_db=-10.0)
    det.process(iq_on, start_sample=0)
    assert det.state == "active"

    # Second buffer - still carrier, no new onset
    events = det.process(iq_on, start_sample=4096)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert onsets == []
    assert det.state == "active"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_returns_to_idle():
    det = make_detector()
    iq_on = _carrier(4096, power_db=-10.0)
    det.process(iq_on, start_sample=0)
    assert det.state == "active"

    det.reset()
    assert det.state == "idle"


# ---------------------------------------------------------------------------
# min_hold_windows - transient-spike suppression
# ---------------------------------------------------------------------------

def test_min_hold_1_fires_on_single_window():
    """min_hold=1 (default) fires onset on the first above-threshold window."""
    det = make_detector(window_samples=64, min_hold_windows=1)
    W = 64
    # 1 window of carrier then silence
    iq = np.concatenate([_carrier(W, power_db=-10.0),
                         _noise(W * 8, power_db=-60.0, rng=np.random.default_rng(0))])
    events = det.process(iq, start_sample=0)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) >= 1, "min_hold=1 should fire on single above-threshold window"


def test_min_hold_2_suppresses_single_window_spike():
    """
    A single-window spike above the onset threshold must NOT trigger an onset
    when min_hold_windows=2.  The transient drops back to noise before the
    required second consecutive above-threshold window.
    """
    rng = np.random.default_rng(7)
    det = make_detector(window_samples=64, min_hold_windows=2)
    W = 64
    # noise -> 1 spike window -> noise  (only 1 consecutive above-threshold window)
    iq = np.concatenate([
        _noise(W * 4, power_db=-60.0, rng=rng),   # below threshold
        _carrier(W,   power_db=-10.0),             # 1 window spike
        _noise(W * 4, power_db=-60.0, rng=rng),   # back to noise
    ])
    events = det.process(iq, start_sample=0)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert onsets == [], (
        "Single-window spike should not trigger onset with min_hold_windows=2"
    )


def test_min_hold_2_triggers_on_two_consecutive_windows():
    """
    Two consecutive windows above the onset threshold must trigger an onset
    when min_hold_windows=2.
    """
    rng = np.random.default_rng(8)
    det = make_detector(window_samples=64, min_hold_windows=2)
    W = 64
    # noise -> 2 consecutive carrier windows -> noise
    iq = np.concatenate([
        _noise(W * 4, power_db=-60.0, rng=rng),
        _carrier(W * 2, power_db=-10.0),           # 2 consecutive above-threshold
        _noise(W * 4, power_db=-60.0, rng=rng),
    ])
    events = det.process(iq, start_sample=0)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) == 1, (
        "Two consecutive windows above threshold must fire exactly one onset "
        "with min_hold_windows=2"
    )


def test_min_hold_reset_on_gap():
    """
    If the signal drops below threshold before min_hold is reached, the counter
    must reset.  A subsequent spike must again accumulate min_hold windows.
    """
    rng = np.random.default_rng(9)
    det = make_detector(window_samples=64, min_hold_windows=3)
    W = 64
    # spike, drop, spike, drop - each spike is only 1 window, never reaches 3
    chunk = np.concatenate([
        _carrier(W,   power_db=-10.0),
        _noise(W * 2, power_db=-60.0, rng=rng),
        _carrier(W,   power_db=-10.0),
        _noise(W * 2, power_db=-60.0, rng=rng),
    ])
    events = det.process(chunk, start_sample=0)
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert onsets == [], "Interrupted spikes must not accumulate toward min_hold"


# ---------------------------------------------------------------------------
# Detection through the target decimation chain (Decimator -> CarrierDetector)
#
# In the full pipeline, target IQ at 2.048 MSPS is decimated by 32 before
# reaching CarrierDetector (output rate: 64 kHz).  These tests verify:
#   1. A carrier at the default onset threshold is detected after decimation.
#   2. The power level of a CW carrier is preserved through decimation
#      (decimation does NOT increase carrier power).
#   3. Signals below the offset threshold are not detected.
#
# Threshold calibration note
# --------------------------
# The default onset_threshold_db in PipelineConfig is -30 dBFS.  Real LMR
# signals received over-the-air are typically -40 to -60 dBFS at the ADC
# input.  If no detections occur, lower carrier_onset_db (e.g. -50 dBFS)
# in the freq_hop section of node.yaml and re-run.
# ---------------------------------------------------------------------------

SDR_RATE = 2_048_000.0
TARGET_DEC = 32          # -> 64 kHz
TARGET_CUTOFF = 25_000.0


def _target_decimator():
    from beagle_node.pipeline.decimator import Decimator
    return Decimator(TARGET_DEC, SDR_RATE, TARGET_CUTOFF)


def _decimated_carrier(n_raw: int, power_db: float) -> np.ndarray:
    """Generate raw-rate CW carrier and decimate to target rate."""
    dec = _target_decimator()
    raw = _carrier(n_raw, power_db)
    return dec.process(raw)


def _decimated_noise(n_raw: int, power_db: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate raw-rate noise and decimate to target rate."""
    dec = _target_decimator()
    raw = _noise(n_raw, power_db, rng)
    return dec.process(raw)


class TestCarrierDetectAfterDecimation:
    """Carrier detection at the decimated target rate (64 kHz)."""

    def test_carrier_power_preserved_through_decimation(self):
        """
        A CW carrier's power level is preserved by decimation - the decimator
        is a low-pass filter, not an averager that reduces carrier amplitude.
        A carrier at -20 dBFS before decimation should measure ~-20 dBFS after.
        """
        iq_dec = _decimated_carrier(n_raw=131_072, power_db=-20.0)
        # Discard filter transient (first ~0.1% of output)
        iq_steady = iq_dec[len(iq_dec) // 10:]
        measured_db = 10.0 * np.log10(float(np.mean(np.abs(iq_steady) ** 2)) + 1e-30)
        assert abs(measured_db - (-20.0)) < 3.0, (
            f"Carrier power after decimation: {measured_db:.1f} dBFS, expected ~-20 dBFS"
        )

    def test_carrier_detected_at_threshold(self):
        """
        A carrier at -25 dBFS with onset threshold -30 dBFS should be detected.
        This represents a moderately strong real signal.
        """
        ONSET = -30.0
        det = CarrierDetector(
            sample_rate_hz=SDR_RATE / TARGET_DEC,
            onset_threshold_db=ONSET,
            offset_threshold_db=ONSET - 10.0,
            window_samples=64,
        )
        iq_dec = _decimated_carrier(n_raw=131_072, power_db=-25.0)
        events = det.process(iq_dec, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) >= 1, (
            f"No onset at -25 dBFS with threshold {ONSET} dBFS - "
            "check carrier_onset_db in PipelineConfig"
        )

    def test_weak_carrier_missed_at_default_threshold(self):
        """
        A -50 dBFS carrier is below the default -30 dBFS onset threshold and
        must NOT trigger.  This documents the calibration requirement: if real
        signals are this weak, lower carrier_onset_db in node.yaml.
        """
        det = CarrierDetector(
            sample_rate_hz=SDR_RATE / TARGET_DEC,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
        )
        iq_dec = _decimated_carrier(n_raw=131_072, power_db=-50.0)
        events = det.process(iq_dec, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert onsets == [], (
            "Carrier at -50 dBFS should not trigger -30 dBFS threshold; "
            "if this fires, something is wrong with power measurement"
        )

    def test_weak_carrier_detected_with_lowered_threshold(self):
        """
        The same -50 dBFS carrier IS detected when onset_threshold_db is lowered
        to -55 dBFS.  Use this threshold in node.yaml for weak signals.
        """
        det = CarrierDetector(
            sample_rate_hz=SDR_RATE / TARGET_DEC,
            onset_threshold_db=-55.0,
            offset_threshold_db=-65.0,
            window_samples=64,
        )
        iq_dec = _decimated_carrier(n_raw=131_072, power_db=-50.0)
        events = det.process(iq_dec, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) >= 1, (
            "Carrier at -50 dBFS should be detected with threshold -55 dBFS"
        )

    def test_min_release_windows_property_readable(self):
        det = CarrierDetector(RATE, min_release_windows=4)
        assert det.min_release_windows == 4

    def test_noise_does_not_trigger_detection(self):
        """
        Band-limited noise below the onset threshold must not produce any events.
        """
        rng = np.random.default_rng(42)
        det = CarrierDetector(
            sample_rate_hz=SDR_RATE / TARGET_DEC,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
        )
        iq_dec = _decimated_noise(n_raw=131_072, power_db=-60.0, rng=rng)
        events = det.process(iq_dec, start_sample=0)
        assert events == [], "Noise below threshold should not trigger detection"

    def test_onset_then_offset_across_decimated_buffers(self):
        """
        Carrier then silence, split into two raw buffers, must produce exactly
        one onset and one offset after decimation.
        """
        N_RAW = 131_072
        dec_on  = _target_decimator()
        dec_off = _target_decimator()

        rng = np.random.default_rng(7)
        raw_on  = _carrier(N_RAW, power_db=-20.0)
        raw_off = _noise(N_RAW, power_db=-60.0, rng=rng)

        iq_on  = dec_on.process(raw_on)
        iq_off = dec_off.process(raw_off)

        det = CarrierDetector(
            sample_rate_hz=SDR_RATE / TARGET_DEC,
            onset_threshold_db=-30.0,
            offset_threshold_db=-40.0,
            window_samples=64,
        )
        ev1 = det.process(iq_on,  start_sample=0)
        ev2 = det.process(iq_off, start_sample=len(iq_on))
        events = ev1 + ev2

        onsets  = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(onsets)  == 1
        assert len(offsets) == 1
        assert offsets[0].sample_index > onsets[0].sample_index


# ---------------------------------------------------------------------------
# min_release_windows - transient fade suppression
# ---------------------------------------------------------------------------


def test_min_release_1_fires_on_single_below_window():
    """min_release=1 (default) fires offset on the first below-threshold window."""
    rng = np.random.default_rng(10)
    det = make_detector(window_samples=64, min_release_windows=1)
    W = 64
    iq = np.concatenate([
        _carrier(W * 8, power_db=-10.0),
        _noise(W,       power_db=-60.0, rng=rng),
    ])
    ev = det.process(iq, start_sample=0)
    offsets = [e for e in ev if isinstance(e, CarrierOffset)]
    assert len(offsets) == 1, "min_release=1 should fire offset on first below-threshold window"


def test_min_release_4_suppresses_single_window_fade():
    """
    A single below-threshold window that immediately recovers must NOT trigger
    a CarrierOffset when min_release_windows=4.
    """
    rng = np.random.default_rng(11)
    det = make_detector(window_samples=64, min_release_windows=4)
    W = 64
    # carrier -> 1 window of noise -> carrier -> sustained noise (real end)
    iq = np.concatenate([
        _carrier(W * 8, power_db=-10.0),   # onset
        _noise(W,       power_db=-60.0, rng=rng),  # single-window fade
        _carrier(W * 4, power_db=-10.0),   # signal recovers
        _noise(W * 8,   power_db=-60.0, rng=rng),  # true end (8 windows, >= 4)
    ])
    ev = det.process(iq, start_sample=0)
    onsets  = [e for e in ev if isinstance(e, CarrierOnset)]
    offsets = [e for e in ev if isinstance(e, CarrierOffset)]
    # The single-window fade must NOT produce an offset (and therefore no extra onset).
    # Only one onset (at the start) and one offset (at the true end) should be seen.
    assert len(onsets)  == 1, (
        "Single-window fade with min_release=4 must not cause a spurious re-onset"
    )
    assert len(offsets) == 1, (
        "Single-window fade with min_release=4 must not trigger a premature offset"
    )


def test_min_release_4_fires_after_four_consecutive_below_windows():
    """
    Four consecutive below-threshold windows must trigger a CarrierOffset when
    min_release_windows=4.
    """
    rng = np.random.default_rng(12)
    det = make_detector(window_samples=64, min_release_windows=4)
    W = 64
    iq = np.concatenate([
        _carrier(W * 8, power_db=-10.0),
        _noise(W * 4,   power_db=-60.0, rng=rng),  # exactly 4 below-threshold windows
    ])
    ev = det.process(iq, start_sample=0)
    offsets = [e for e in ev if isinstance(e, CarrierOffset)]
    assert len(offsets) == 1, (
        "Four consecutive below-threshold windows must fire offset with min_release_windows=4"
    )


def test_min_release_counter_resets_on_recovery():
    """
    Interrupted below-threshold windows (fade, recover, fade, ...) must not
    accumulate toward min_release_windows.
    """
    rng = np.random.default_rng(13)
    det = make_detector(window_samples=64, min_release_windows=4)
    W = 64
    # carrier -> 3x(1 fade + 1 carrier) -> sustained carrier (never reaches 4)
    fade_recover = np.concatenate([
        _noise(W,   power_db=-60.0, rng=rng),   # 1 below threshold
        _carrier(W, power_db=-10.0),             # recovers
    ])
    iq = np.concatenate([
        _carrier(W * 4, power_db=-10.0),
        fade_recover,
        fade_recover,
        fade_recover,
        _carrier(W * 4, power_db=-10.0),   # still active
    ])
    ev = det.process(iq, start_sample=0)
    offsets = [e for e in ev if isinstance(e, CarrierOffset)]
    assert offsets == [], (
        "Interrupted fades must not accumulate toward min_release_windows"
    )
    assert det.state == "active"


def test_min_release_no_chattering_at_offset_threshold():
    """
    Signal hovering at the offset threshold (alternating just above and just
    below) must produce at most one onset and one offset, not a burst of
    repeated transitions.

    This is the real-world scenario that triggered the min_release_windows
    feature: carrier power hovering near the -40 dBFS offset threshold caused
    every window to toggle the state, producing measurements spaced exactly
    one power window apart (~1 ms).
    """
    rng = np.random.default_rng(14)
    det = make_detector(
        onset_threshold_db=-20.0,
        offset_threshold_db=-40.0,
        window_samples=64,
        min_release_windows=8,
    )
    W = 64
    # Start with a clear carrier onset
    iq_onset = _carrier(W * 4, power_db=-10.0)
    # Signal hovers at the offset threshold: 20 windows alternating just above
    # and just below -40 dBFS (but all below the -20 dBFS onset threshold).
    hover = []
    for i in range(20):
        # Alternate -39 dBFS (above offset) and -41 dBFS (below offset)
        level = -39.0 if i % 2 == 0 else -41.0
        hover.append(_carrier(W, power_db=level))
    iq_hover = np.concatenate(hover)
    # Finally drop well below to ensure a true offset fires
    iq_end = _noise(W * 10, power_db=-60.0, rng=rng)

    iq = np.concatenate([iq_onset, iq_hover, iq_end])
    ev = det.process(iq, start_sample=0)

    onsets  = [e for e in ev if isinstance(e, CarrierOnset)]
    offsets = [e for e in ev if isinstance(e, CarrierOffset)]
    assert len(onsets)  == 1, (
        f"Expected 1 onset; got {len(onsets)}.  "
        "Hovering signal should not produce re-onsets."
    )
    assert len(offsets) == 1, (
        f"Expected 1 offset; got {len(offsets)}.  "
        "Hovering at the offset threshold should not chatter."
    )


# ---------------------------------------------------------------------------
# Noise floor tracking
# ---------------------------------------------------------------------------

def test_onset_carries_noise_floor():
    """noise_floor_db on a CarrierOnset should converge toward idle power."""
    rng = np.random.default_rng(42)
    det = make_detector()
    # Feed ~5000 windows of noise at -50 dBFS so the EMA can warm up.
    idle_power_db = -50.0
    iq_idle = _noise(5000 * 64, idle_power_db, rng)
    det.process(iq_idle, start_sample=0)

    # Now trigger an onset.
    iq_carrier = _carrier(4096, power_db=-10.0)
    events = det.process(iq_carrier, start_sample=len(iq_idle))
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) == 1
    # After 5000 * 0.01 = 50 time-constants of EMA the floor should be close
    # to idle_power_db.  Allow +/-5 dB tolerance.
    assert abs(onsets[0].noise_floor_db - idle_power_db) < 5.0


def test_noise_floor_not_updated_during_active():
    """noise_floor_db should not track carrier power - only idle power."""
    rng = np.random.default_rng(7)
    det = make_detector()
    # Warm up at -50 dBFS.
    iq_idle = _noise(5000 * 64, -50.0, rng)
    det.process(iq_idle, start_sample=0)

    # Now push a carrier at -10 dBFS.  Floor should stay near -50 dBFS.
    iq_carrier = _carrier(10_000, power_db=-10.0)
    events = det.process(iq_carrier, start_sample=len(iq_idle))
    onsets = [e for e in events if isinstance(e, CarrierOnset)]
    assert len(onsets) == 1
    # Floor should still be near -50 dBFS, not dragged toward -10 dBFS.
    assert onsets[0].noise_floor_db < -30.0


# ---------------------------------------------------------------------------
# freq_hop block-start onset suppression (prime_state + _idle_window_count)
# ---------------------------------------------------------------------------

class TestPrimeStateFreqHop:
    """
    Tests for the freq_hop mid-transmission-arrival suppression.

    In freq_hop mode prime_state() is called at the start of each target
    block.  If the carrier was already active during the sync block, the
    detector must not fire a spurious onset.
    """

    def test_prime_active_no_onset(self):
        """
        Carrier already at full power when prime_state is called -> no onset.
        """
        det = make_detector()
        iq_carrier = _carrier(4096, power_db=-10.0)
        det.prime_state(iq_carrier)
        assert det.state == "active"
        # Running process() on more carrier must NOT produce an onset.
        events = det.process(iq_carrier, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert onsets == [], (
            "prime_state with carrier already active should suppress onset"
        )

    def test_prime_hysteresis_zone_onset_allowed(self):
        """
        Carrier in the hysteresis zone (above offset_db, below onset_db) at
        block start -> state should be 'idle' (prime uses onset_db threshold).

        A signal in the hysteresis zone is not strong enough to be a genuine
        carrier - it's more likely a PLL settling transient.  prime_state
        classifies it as idle, so if the signal later rises above onset_db,
        a genuine onset fires (after _min_idle_for_onset windows).
        """
        det = make_detector(onset_threshold_db=-20.0, offset_threshold_db=-40.0)
        # Build IQ at -30 dBFS - between offset (-40) and onset (-20) thresholds.
        iq_hysteresis = _carrier(4096, power_db=-30.0)
        det.prime_state(iq_hysteresis)
        assert det.state == "idle", (
            "prime_state should classify hysteresis-zone power as idle "
            "(uses onset_db, not offset_db)"
        )

    def test_prime_idle_then_carrier_fires_onset(self):
        """
        Carrier absent at block start (prime -> idle), then carrier rises in
        the block -> onset fires normally.  The idle window before the carrier
        is the evidence that this is a genuine onset (not a mid-tx arrival).
        """
        rng = np.random.default_rng(42)
        det = make_detector()
        # prime on noise (below offset threshold -> idle)
        iq_noise = _noise(4096, power_db=-60.0, rng=rng)
        det.prime_state(iq_noise)
        assert det.state == "idle"

        # Feed idle noise (builds idle_window_count >= 2) then carrier.
        iq_block = np.concatenate([
            _noise(64 * 4, power_db=-60.0, rng=rng),   # idle windows
            _carrier(4096, power_db=-10.0),              # genuine onset
        ])
        events = det.process(iq_block, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "Genuine onset (carrier absent at block start, then rises) must fire"
        )

    def test_prime_idle_immediate_carrier_suppressed(self):
        """
        After prime_state(idle) with no prior idle windows, an onset that fires
        immediately (carrier already at full power from the first window) is
        suppressed because _idle_window_count is 0 (< min_idle_for_onset=2).

        This is the PLL-settling edge case: settling discarded the transient,
        but the carrier power hasn't ramped up to onset_threshold yet at the
        first window check.  prime_state -> idle, then process() sees full power
        immediately -> would be a block-start false onset.
        """
        det = make_detector()
        # prime_state on noise (below threshold -> idle), _idle_window_count=0
        rng = np.random.default_rng(0)
        det.prime_state(_noise(64, power_db=-60.0, rng=rng))
        assert det.state == "idle"

        # Immediately feed carrier with NO preceding idle window in process().
        iq_carrier = _carrier(4096, power_db=-10.0)
        events = det.process(iq_carrier, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert onsets == [], (
            "Block-start onset (no idle window after prime_state) must be suppressed"
        )
        # State must still be 'active' so subsequent offset is detected.
        assert det.state == "active"

    def test_prime_suppressed_onset_then_offset_detected(self):
        """
        After a block-start onset is suppressed, the carrier eventually drops.
        The offset event must still be detected (we're in 'active' state).
        """
        rng = np.random.default_rng(1)
        det = make_detector()
        det.prime_state(_noise(64, power_db=-60.0, rng=rng))

        # Carrier from the first window (suppressed onset), then drops.
        iq_carrier = _carrier(4096, power_db=-10.0)
        iq_noise   = _noise(4096, power_db=-60.0, rng=rng)
        ev1 = det.process(iq_carrier, start_sample=0)
        ev2 = det.process(iq_noise,   start_sample=4096)
        events = ev1 + ev2
        onsets  = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert onsets == [], "Block-start onset must still be suppressed"
        assert len(offsets) == 1, "Offset must be detected after suppressed onset"

    def test_prime_suppressed_onset_then_rekey_detected(self):
        """
        After a block-start onset is suppressed and the carrier drops, a
        subsequent re-key in the same block IS detected (because idle windows
        during the drop push _idle_window_count past min_idle_for_onset).
        """
        rng = np.random.default_rng(2)
        det = make_detector()
        det.prime_state(_noise(64, power_db=-60.0, rng=rng))

        # Block-start carrier (suppressed), drop, re-key.
        iq = np.concatenate([
            _carrier(64 * 4, power_db=-10.0),    # suppressed block-start
            _noise(64 * 8,   power_db=-60.0, rng=rng),  # drop (idle windows)
            _carrier(64 * 4, power_db=-10.0),    # genuine re-key
        ])
        events = det.process(iq, start_sample=0)
        onsets  = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        # Block-start suppressed -> no onset from that. Re-key -> 1 onset.
        # Between block-start and re-key there is one offset.
        assert len(onsets)  == 1, "Re-key after block-start suppression must produce onset"
        assert len(offsets) >= 1, "Drop between block-start and re-key must produce offset"

    def test_normal_reset_no_suppression(self):
        """
        After a full reset() (not prime_state), _idle_window_count starts
        high, so the very first onset in the next block fires normally.
        """
        det = make_detector()
        det.reset()
        iq_carrier = _carrier(4096, power_db=-10.0)
        events = det.process(iq_carrier, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "After reset() (not prime_state), onset must fire normally"
        )

    def test_mid_block_offset_then_immediate_rekey_fires_onset(self):
        """
        Regression: after prime_state(active), if the carrier drops mid-block
        (firing an offset) and immediately comes back, the re-onset must fire.

        Bug: release windows were processed in 'active' state, where
        _idle_window_count does not increment.  After offset fires and state
        becomes 'idle', _idle_window_count was still 0 (reset by prime_state),
        causing the re-onset to be suppressed by _min_idle_for_onset=2.

        Fix: release windows (power <= offset_db while active) now also
        increment _idle_window_count, so by the time offset fires,
        _idle_window_count >= min_release_windows >= min_idle_for_onset.
        """
        rng = np.random.default_rng(7)
        W = 64
        min_release = 4
        det = make_detector(min_hold_windows=1, min_release_windows=min_release)

        # prime_state with carrier present -> active, _idle_window_count=0
        det.prime_state(_carrier(W, power_db=-10.0))
        assert det.state == "active"

        # Block: carrier on for a while, drops for exactly min_release windows,
        # then immediately comes back.
        iq = np.concatenate([
            _carrier(W * 10,        power_db=-10.0),   # carrier on
            _noise(W * min_release, power_db=-60.0, rng=rng),  # drop (release)
            _carrier(W * 10,        power_db=-10.0),   # re-key
        ])
        events = det.process(iq, start_sample=0)
        onsets  = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]

        assert len(offsets) == 1, "Carrier drop must produce exactly one offset"
        assert len(onsets)  == 1, (
            "Re-onset after mid-block offset must fire even with no idle windows "
            "in idle state (release windows must count toward _idle_window_count)"
        )

    def test_mid_block_offset_rekey_produces_balanced_counts(self):
        """
        Multiple mid-block carrier pauses (each long enough to trigger an
        offset) must each also produce a re-onset: onset and offset counts
        must be equal across a block with several pause-rekey cycles.

        This is the freq_hop 7:1 offset:onset imbalance regression test.
        """
        rng = np.random.default_rng(11)
        W = 64
        det = make_detector(min_hold_windows=1, min_release_windows=4)

        # prime with carrier present
        det.prime_state(_carrier(W, power_db=-10.0))

        # Three pause-rekey cycles, each pause = 4 release windows.
        segment = np.concatenate([
            _carrier(W * 10,  power_db=-10.0),
            _noise(W * 4,     power_db=-60.0, rng=rng),
        ])
        iq = np.concatenate([segment, segment, segment, _carrier(W * 5, power_db=-10.0)])
        events = det.process(iq, start_sample=0)
        onsets  = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]

        assert len(offsets) == 3, f"Expected 3 offsets, got {len(offsets)}"
        assert len(onsets)  == 3, (
            f"Expected 3 re-onsets (one per pause), got {len(onsets)}; "
            "onset:offset imbalance indicates release windows are not being "
            "counted toward _idle_window_count"
        )


# ---------------------------------------------------------------------------
# Snippet transition validation (freq_hop mid-transmission defence-in-depth)
# ---------------------------------------------------------------------------

class TestMidTransmissionSuppression:
    """
    Tests for the two-layer freq_hop mid-transmission arrival defence:

    Layer 1 - idle window counting: after prime_state(), at least
    _min_idle_for_onset (default 2) below-threshold windows must be
    observed before an onset is emitted.  Catches carriers already
    present at block start AND single-window PLL settling artefacts.

    Layer 2 - snippet transition validation: after prime_state(), events
    whose IQ snippets lack a genuine power transition (dynamic range
    < _min_transition_db) are dropped.  Catches cases where the ring
    fills with carrier due to high min_hold_windows.
    """

    # --- Layer 1: idle window counting ---

    def test_single_idle_window_suppressed(self):
        """
        A single PLL-settling artefact window (below threshold) followed
        by sustained carrier should NOT produce an onset.  The single
        window is insufficient evidence of genuine noise.
        """
        rng = np.random.default_rng(50)
        W = 64
        det = make_detector(window_samples=W)

        det.prime_state(_noise(W, power_db=-60.0, rng=rng))

        # 1 idle window (PLL transient) then carrier.
        iq = np.concatenate([
            _noise(W, power_db=-60.0, rng=rng),      # 1 idle window
            _carrier(W * 10, power_db=-10.0),         # sustained carrier
        ])
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert onsets == [], (
            "Single idle window after prime_state must not produce onset "
            "(likely PLL settling artefact)"
        )
        assert det.state == "active"

    def test_two_idle_windows_sufficient(self):
        """
        Two idle windows after prime_state() satisfy the minimum for a
        genuine noise->carrier transition.
        """
        rng = np.random.default_rng(51)
        W = 64
        det = make_detector(window_samples=W)

        det.prime_state(_noise(W, power_db=-60.0, rng=rng))

        iq = np.concatenate([
            _noise(W * 2, power_db=-60.0, rng=rng),  # 2 idle windows
            _carrier(W * 4, power_db=-10.0),          # carrier
        ])
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "Two idle windows after prime_state must allow onset"
        )

    def test_rekey_after_suppressed_onset(self):
        """
        After a block-start onset is suppressed (0 idle windows), the
        carrier drops (adding idle windows to the count) and re-keys.
        The re-key onset must fire because the drop provided genuine
        idle windows.
        """
        rng = np.random.default_rng(52)
        W = 64
        det = make_detector(window_samples=W)

        det.prime_state(_carrier(W, power_db=-10.0))
        assert det.state == "active"

        # Carrier continues (no onset - already active), drops, re-keys.
        iq = np.concatenate([
            _carrier(W * 4, power_db=-10.0),          # carrier continues
            _noise(W * 4, power_db=-60.0, rng=rng),   # drop (4 idle windows)
            _carrier(W * 4, power_db=-10.0),           # re-key
        ])
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, "Drop must produce offset"
        assert len(onsets) == 1, (
            "Re-key after genuine drop must produce onset "
            "(idle windows from drop satisfy min_idle_for_onset)"
        )

    # --- Layer 2: snippet transition validation ---

    def test_all_carrier_snippet_dropped(self):
        """
        With high min_hold, the ring can fill with carrier before onset
        fires, pushing noise windows out.  The snippet is all-carrier
        and should be dropped by the dynamic range check.
        """
        W = 64
        # snippet=2 windows, ring_capacity=6, min_hold=4.
        # After 2 idle + 4 carrier windows, ring has 6 windows.
        # Snippet = last 2 windows = all carrier -> dynamic range < 6 dB.
        det = make_detector(
            window_samples=W,
            snippet_samples=W * 2,   # 2-window snippet (minimum for validation)
            min_hold_windows=4,
        )
        rng = np.random.default_rng(53)
        det.prime_state(_noise(W, power_db=-60.0, rng=rng))

        iq = np.concatenate([
            _noise(W * 2, power_db=-60.0, rng=rng),  # 2 idle (pass count)
            _carrier(W * 8, power_db=-10.0),          # onset at window 6
        ])
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert onsets == [], (
            "All-carrier snippet (ring filled with carrier due to high "
            "min_hold) should be dropped by snippet validation"
        )
        assert det.state == "active"

    def test_transition_snippet_kept(self):
        """
        A genuine onset where the snippet captures both noise and carrier
        passes the dynamic range check.
        """
        W = 64
        det = make_detector(
            window_samples=W,
            snippet_samples=W * 10,  # large snippet captures noise too
        )
        rng = np.random.default_rng(54)
        det.prime_state(_noise(W, power_db=-60.0, rng=rng))

        iq = np.concatenate([
            _noise(W * 8, power_db=-60.0, rng=rng),  # 8 idle windows
            _carrier(W * 4, power_db=-10.0),          # carrier
        ])
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "Onset with noise->carrier in snippet must pass validation"
        )

    def test_snippet_validation_not_armed_without_prime(self):
        """
        Without prime_state() (non-freq-hop modes), snippet validation
        is NOT armed and events pass through unconditionally.
        """
        det = make_detector(window_samples=64, snippet_samples=64 * 5)
        iq = _carrier(64 * 20, power_db=-10.0)
        events = det.process(iq, start_sample=0)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "Without prime_state, onset must not be filtered "
            "(non-freq-hop mode)"
        )

    def test_snippet_validation_disarmed_after_process(self):
        """
        Snippet validation is one-shot: arms on prime_state(), disarms
        after the next process() call.
        """
        rng = np.random.default_rng(55)
        W = 64
        det = make_detector(window_samples=W, snippet_samples=W * 5)

        det.prime_state(_noise(W, power_db=-60.0, rng=rng))
        det.process(_noise(W * 8, power_db=-60.0, rng=rng), start_sample=0)

        # Second call without prime_state: validation must be off.
        events = det.process(_carrier(W * 20, power_db=-10.0), start_sample=W * 8)
        onsets = [e for e in events if isinstance(e, CarrierOnset)]
        assert len(onsets) == 1, (
            "Second process() without prime_state must not filter events"
        )

    def test_offset_with_transition_kept(self):
        """
        A normal offset (carrier -> noise in the snippet) passes
        the dynamic range check after prime_state.
        """
        rng = np.random.default_rng(56)
        W = 64
        det = make_detector(window_samples=W, snippet_samples=W * 5)

        det.prime_state(_carrier(W, power_db=-10.0))

        iq = np.concatenate([
            _carrier(W * 30, power_db=-10.0),
            _noise(W, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, (
            "Offset with carrier->noise transition must be kept"
        )

    # --- Direct method tests ---

    def test_snippet_has_transition_method(self):
        """Direct test of _snippet_has_transition with crafted snippets."""
        det = make_detector(window_samples=64)

        # All-carrier snippet: uniform power -> no transition
        carrier_iq = _carrier(640, power_db=-10.0)
        scale = float(np.max(np.abs(carrier_iq))) + 1e-30
        normed = carrier_iq / scale
        int8_ri = np.empty(640 * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
        assert not det._snippet_has_transition(int8_ri.tobytes()), (
            "All-carrier snippet should fail transition check"
        )

        # Noise->carrier snippet: clear transition
        rng = np.random.default_rng(99)
        mixed = np.concatenate([_noise(320, -60.0, rng), _carrier(320, -10.0)])
        scale = float(np.max(np.abs(mixed))) + 1e-30
        normed = mixed / scale
        int8_ri = np.empty(640 * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
        assert det._snippet_has_transition(int8_ri.tobytes()), (
            "Noise->carrier snippet should pass transition check"
        )

        # Very short -> pass (can't validate)
        assert det._snippet_has_transition(b"\x10\x20")


# ---------------------------------------------------------------------------
# Block-start offset suppression (prime_state + _min_active_for_offset)
# ---------------------------------------------------------------------------

class TestBlockStartOffsetSuppression:
    """
    Tests for the freq_hop block-start carrier-tail offset suppression.

    When prime_state() sets state to 'active' (carrier already present at block
    start), a subsequent offset that fires before min_active_windows_for_offset
    above-threshold windows have accumulated is suppressed.  This prevents
    carrier-tail events from block boundaries being reported as measurements,
    which would produce timing anchored to the block boundary rather than the
    true PA shutoff.
    """

    W = 64  # window_samples used in all tests

    def _det(self, min_active=4, **kwargs):
        return make_detector(
            window_samples=self.W,
            min_hold_windows=1,
            min_release_windows=2,
            min_active_windows_for_offset=min_active,
            **kwargs,
        )

    def test_block_start_tail_suppressed(self):
        """
        Carrier present at block start, drops in the first few windows -> suppressed.

        This is the primary case seen on node-discovery: the transmitter was still
        keyed during the sync block and drops within 1-3 windows of the target
        block.  With min_active_windows_for_offset=4, only 1 active window is
        seen, so the offset is suppressed.
        """
        rng = np.random.default_rng(10)
        det = self._det()

        # prime_state with carrier -> state=active, _primed_active=True, _active_window_count=0
        det.prime_state(_carrier(self.W, power_db=-10.0))
        assert det.state == "active"

        # 1 window of carrier then noise; offset fires after 2 release windows
        iq = np.concatenate([
            _carrier(self.W, power_db=-10.0),         # 1 active window
            _noise(self.W * 4, power_db=-60.0, rng=rng),  # noise (triggers release)
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert offsets == [], (
            "Block-start carrier tail (1 active window) must be suppressed "
            "with min_active_windows_for_offset=4"
        )

    def test_block_start_long_carrier_emitted(self):
        """
        Carrier present at block start, remains active for many windows -> emitted.

        A carrier that was transmitting at block start but stays up for 30 windows
        before dropping has meaningful timing information; it should not be suppressed.
        """
        rng = np.random.default_rng(11)
        det = self._det()

        det.prime_state(_carrier(self.W, power_db=-10.0))

        # 30 windows of carrier then noise
        iq = np.concatenate([
            _carrier(self.W * 30, power_db=-10.0),
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, (
            "Block-start carrier with 30 active windows must be emitted "
            "(30 >= min_active_windows_for_offset=4)"
        )

    def test_exactly_at_threshold_emitted(self):
        """
        Carrier active for exactly min_active_windows_for_offset windows -> emitted.
        """
        rng = np.random.default_rng(12)
        min_active = 4
        det = self._det(min_active=min_active)

        det.prime_state(_carrier(self.W, power_db=-10.0))

        # Exactly min_active windows of carrier, then noise
        iq = np.concatenate([
            _carrier(self.W * min_active, power_db=-10.0),
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, (
            f"Carrier with exactly {min_active} active windows must be emitted"
        )

    def test_one_below_threshold_suppressed(self):
        """
        Carrier active for min_active_windows_for_offset - 1 windows -> suppressed.
        """
        rng = np.random.default_rng(13)
        min_active = 4
        det = self._det(min_active=min_active)

        det.prime_state(_carrier(self.W, power_db=-10.0))

        # One fewer than min_active windows of carrier, then noise
        iq = np.concatenate([
            _carrier(self.W * (min_active - 1), power_db=-10.0),
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert offsets == [], (
            f"Carrier with {min_active - 1} active windows must be suppressed"
        )

    def test_genuine_onset_then_offset_not_suppressed(self):
        """
        After prime_state(idle), a genuine onset then offset must not be suppressed.

        The block-start guard only applies when prime_state() set state to active.
        If the block started idle and a real carrier rose and fell, the offset
        must be emitted regardless of how few active windows it lasted.
        """
        rng = np.random.default_rng(14)
        det = self._det()

        # prime_state with noise -> state=idle, _primed_active=False
        det.prime_state(_noise(self.W, power_db=-60.0, rng=rng))
        assert det.state == "idle"

        # idle -> carrier -> noise
        iq = np.concatenate([
            _noise(self.W * 4, power_db=-60.0, rng=rng),   # idle
            _carrier(self.W * 2, power_db=-10.0),           # onset then immediate drop
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) >= 1, (
            "Offset after genuine onset (primed idle) must not be suppressed"
        )

    def test_block_start_drop_then_rekey_offset_not_suppressed(self):
        """
        Block-start tail suppressed, carrier re-keys later -> re-key offset emitted.

        After the block-start carrier tail is suppressed, the state goes idle.
        When the carrier re-keys (genuine onset, passes idle window count check),
        its subsequent offset must not be suppressed - _primed_active is cleared
        by the genuine onset.
        """
        rng = np.random.default_rng(15)
        det = self._det()

        det.prime_state(_carrier(self.W, power_db=-10.0))

        # 1 window carrier (tail, suppressed), noise, then re-key for 30 windows
        iq = np.concatenate([
            _carrier(self.W, power_db=-10.0),             # tail (suppressed offset)
            _noise(self.W * 4, power_db=-60.0, rng=rng), # idle (builds idle count)
            _carrier(self.W * 30, power_db=-10.0),        # genuine re-key
            _noise(self.W * 4, power_db=-60.0, rng=rng), # offset
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) >= 1, (
            "Offset from re-key after suppressed block-start tail must be emitted"
        )

    def test_disabled_when_zero(self):
        """
        min_active_windows_for_offset=0 disables the guard entirely (default).
        """
        rng = np.random.default_rng(16)
        det = self._det(min_active=0)

        det.prime_state(_carrier(self.W, power_db=-10.0))

        # 1 window carrier (would be suppressed if guard were active), then noise
        iq = np.concatenate([
            _carrier(self.W, power_db=-10.0),
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, (
            "min_active_windows_for_offset=0 must disable guard (backward-compatible)"
        )

    def test_property_readable(self):
        det = self._det(min_active=7)
        assert det.min_active_windows_for_offset == 7

    def test_update_thresholds_changes_guard(self):
        """min_active_windows_for_offset is hot-reloadable via update_thresholds."""
        rng = np.random.default_rng(17)
        det = self._det(min_active=10)

        # With guard=10, a 1-window tail is suppressed
        det.prime_state(_carrier(self.W, power_db=-10.0))
        iq = np.concatenate([
            _carrier(self.W, power_db=-10.0),
            _noise(self.W * 4, power_db=-60.0, rng=rng),
        ])
        events = det.process(iq, start_sample=0)
        assert [e for e in events if isinstance(e, CarrierOffset)] == []

        # After updating guard to 0, same scenario produces an offset
        det.update_thresholds(min_active_windows_for_offset=0)
        assert det.min_active_windows_for_offset == 0
        det.prime_state(_carrier(self.W, power_db=-10.0))
        events = det.process(iq, start_sample=0)
        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert len(offsets) == 1, (
            "After disabling guard via update_thresholds, offset must fire"
        )


# ---------------------------------------------------------------------------
# Live threshold updates (update_thresholds + noise_floor_db property)
# ---------------------------------------------------------------------------

class TestUpdateThresholds:
    def test_noise_floor_property(self):
        det = make_detector()
        # Initial noise floor is offset_threshold_db
        assert det.noise_floor_db == OFFSET

    def test_update_thresholds_changes_values(self):
        det = make_detector()
        det.update_thresholds(onset_threshold_db=-15.0, offset_threshold_db=-25.0)
        assert det.onset_threshold_db == -15.0
        assert det.offset_threshold_db == -25.0

    def test_update_thresholds_partial(self):
        det = make_detector()
        det.update_thresholds(onset_threshold_db=-15.0)
        assert det.onset_threshold_db == -15.0
        assert det.offset_threshold_db == OFFSET  # unchanged

    def test_update_thresholds_hold_release(self):
        det = make_detector()
        det.update_thresholds(min_hold_windows=4, min_release_windows=8)
        assert det.min_hold_windows == 4
        assert det.min_release_windows == 8

    def test_update_thresholds_rejects_invalid(self):
        det = make_detector()
        with pytest.raises(ValueError, match="offset.*must be <"):
            det.update_thresholds(onset_threshold_db=-30.0, offset_threshold_db=-20.0)

    def test_update_thresholds_preserves_state(self):
        """Updating thresholds should not reset the state machine."""
        rng = np.random.default_rng(42)
        det = make_detector()
        # Push into active state
        iq_carrier = _carrier(640, power_db=-10.0)
        det.process(iq_carrier, start_sample=0)
        assert det.state == "active"
        # Update thresholds - state should remain active
        det.update_thresholds(onset_threshold_db=-15.0, offset_threshold_db=-25.0)
        assert det.state == "active"


# ---------------------------------------------------------------------------
# Derivative-peak sample_index refinement for CarrierOffset
# ---------------------------------------------------------------------------

class TestOffsetSampleIndexRefinement:
    """
    Verify that CarrierOffset.sample_index is set to the PA shutoff sample
    (peak negative power derivative) rather than the coarse threshold-crossing
    window centre, which is delayed by min_release_windows from the true cutoff.
    """

    _WINDOW = 64
    _MIN_RELEASE = 4
    # Ring fills at snippet_windows*3; 1280-sample snippet / 64 = 20 windows -> 60 ring slots.
    # Run 65 carrier windows so the ring is guaranteed full at offset detection.
    _N_ON = 65

    def _make_det(self, post_windows: int = 5) -> CarrierDetector:
        return make_detector(
            window_samples=self._WINDOW,
            min_release_windows=self._MIN_RELEASE,
            snippet_post_windows=post_windows,
        )

    def _run_carrier_then_noise(
        self, det: CarrierDetector, rng: np.random.Generator
    ) -> tuple[list, int]:
        """Run N_ON carrier windows then noise; return (events, shutoff_sample)."""
        carrier = _carrier(self._N_ON * self._WINDOW, power_db=-10.0)
        det.process(carrier, start_sample=0)
        assert det.state == "active"

        shutoff_sample = self._N_ON * self._WINDOW
        noise = _noise(20 * self._WINDOW, power_db=-60.0, rng=rng)
        events = det.process(noise, start_sample=shutoff_sample)
        return events, shutoff_sample

    def test_deferred_path_sample_index_near_shutoff(self):
        """
        With post_windows > 0 (deferred emission path), sample_index must be
        within +/-2 windows of the actual PA shutoff sample.
        """
        rng = np.random.default_rng(7)
        det = self._make_det(post_windows=5)
        events, shutoff = self._run_carrier_then_noise(det, rng)

        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert offsets, "No CarrierOffset emitted on deferred path"

        idx = offsets[0].sample_index
        assert abs(idx - shutoff) < 2 * self._WINDOW, (
            f"Deferred path: sample_index={idx} is {idx - shutoff:+d} samples "
            f"from shutoff at {shutoff}; expected within +/-{2 * self._WINDOW}"
        )

    def test_immediate_path_sample_index_near_shutoff(self):
        """
        With post_windows=0 (immediate emission path), sample_index must be
        within +/-2 windows of the actual PA shutoff sample.
        """
        rng = np.random.default_rng(13)
        det = self._make_det(post_windows=0)
        events, shutoff = self._run_carrier_then_noise(det, rng)

        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert offsets, "No CarrierOffset emitted on immediate path"

        idx = offsets[0].sample_index
        assert abs(idx - shutoff) < 2 * self._WINDOW, (
            f"Immediate path: sample_index={idx} is {idx - shutoff:+d} samples "
            f"from shutoff at {shutoff}; expected within +/-{2 * self._WINDOW}"
        )

    def test_deferred_path_better_than_old_window_centre(self):
        """
        Refined sample_index is closer to the true shutoff than the old
        threshold-crossing window centre (which was delayed by min_release_windows).
        """
        rng = np.random.default_rng(21)
        det = self._make_det(post_windows=5)
        events, shutoff = self._run_carrier_then_noise(det, rng)

        offsets = [e for e in events if isinstance(e, CarrierOffset)]
        assert offsets

        idx = offsets[0].sample_index
        # Old code would set sample_index = window_sample at detection:
        #   detection fires min_release windows after first below-threshold window
        old_idx = shutoff + self._MIN_RELEASE * self._WINDOW + self._WINDOW // 2
        error_refined = abs(idx - shutoff)
        error_old = abs(old_idx - shutoff)
        assert error_refined < error_old, (
            f"Refined error {error_refined} >= old window-centre error {error_old}; "
            f"derivative peak did not improve accuracy over threshold-crossing estimate"
        )
