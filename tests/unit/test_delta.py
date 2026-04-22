# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for pipeline/delta.py."""

from __future__ import annotations

import pytest

from beagle_node.pipeline.carrier_detect import CarrierOnset, CarrierOffset
from beagle_node.pipeline.delta import DeltaComputer, TDOAMeasurement
from beagle_node.pipeline.sync_detector import SyncEvent

RATE = 256_000.0   # Hz


def make_sync(sample_index: int, corr_peak: float = 0.9,
              correction: float = 1.0) -> SyncEvent:
    return SyncEvent(
        sample_index=sample_index,
        time_ns=0,
        corr_peak=corr_peak,
        pilot_phase_rad=0.0,
        sample_rate_correction=correction,
    )


def make_onset(sample_index: int, power_db: float = -20.0) -> CarrierOnset:
    return CarrierOnset(sample_index=sample_index, power_db=power_db)


def make_offset(sample_index: int, power_db: float = -35.0) -> CarrierOffset:
    return CarrierOffset(sample_index=sample_index, power_db=power_db)


def make_dc(**kwargs) -> DeltaComputer:
    defaults = dict(sample_rate_hz=RATE, max_sync_age_samples=10_000,
                    pps_anchored=False, min_corr_peak=0.1)
    defaults.update(kwargs)
    return DeltaComputer(**defaults)


# ---------------------------------------------------------------------------
# Basic measurement
# ---------------------------------------------------------------------------

def test_basic_measurement():
    dc = make_dc()
    dc.feed_sync(make_sync(1000))
    results = dc.feed_onset(make_onset(3560))
    assert len(results) == 1
    m = results[0]
    assert isinstance(m, TDOAMeasurement)
    assert m.sync_sample == 1000
    assert m.snippet_start_sample == 3560


def test_sync_delta_ns_value():
    """sync_to_snippet_start_ns = (target - sync) * 1e9 / rate."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_onset(make_onset(2560))   # 10 ms at 256 kHz
    assert len(results) == 1
    expected_ns = int(round(2560 * 1e9 / RATE))   # 10_000_000 ns
    assert results[0].sync_to_snippet_start_ns == expected_ns


def test_zero_delta():
    """Onset at the same sample as sync -> sync_to_snippet_start_ns = 0."""
    dc = make_dc()
    dc.feed_sync(make_sync(5000))
    results = dc.feed_onset(make_onset(5000))
    assert len(results) == 1
    assert results[0].sync_to_snippet_start_ns == 0


def test_uses_most_recent_sync():
    """With multiple sync events, the most recent before the onset is used."""
    dc = make_dc()
    dc.feed_sync(make_sync(100))
    dc.feed_sync(make_sync(500))   # more recent
    dc.feed_sync(make_sync(900))   # most recent
    results = dc.feed_onset(make_onset(1000))
    assert len(results) == 1
    assert results[0].sync_sample == 900


def test_sync_after_onset_not_used():
    """A SyncEvent whose sample_index > onset is NOT used."""
    dc = make_dc()
    dc.feed_sync(make_sync(2000))   # after onset
    results = dc.feed_onset(make_onset(1000))
    # No valid sync -> no measurement yet
    assert results == []


# ---------------------------------------------------------------------------
# Crystal calibration applied
# ---------------------------------------------------------------------------

def test_crystal_correction_applied():
    """If correction = 1.0001, corrected_rate differs and delta_ns changes."""
    correction = 1.0001
    dc_corr   = make_dc()
    dc_nocorr = make_dc()

    dc_corr.feed_sync(make_sync(0, correction=correction))
    dc_nocorr.feed_sync(make_sync(0, correction=1.0))

    onset = make_onset(2_560)   # 10 ms at nominal rate (must be < max_sync_age_samples=10_000)

    r_corr   = dc_corr.feed_onset(onset)
    r_nocorr = dc_nocorr.feed_onset(onset)

    assert len(r_corr) == 1 and len(r_nocorr) == 1
    # With faster crystal, corrected_rate is higher -> delta_ns is smaller
    assert r_corr[0].sync_to_snippet_start_ns != r_nocorr[0].sync_to_snippet_start_ns
    assert r_corr[0].sample_rate_correction == correction


# ---------------------------------------------------------------------------
# Low corr_peak filtering
# ---------------------------------------------------------------------------

def test_low_corr_peak_sync_dropped():
    """Sync events below min_corr_peak are silently dropped."""
    dc = make_dc(min_corr_peak=0.5)
    dc.feed_sync(make_sync(0, corr_peak=0.3))   # below threshold
    results = dc.feed_onset(make_onset(1000))
    assert results == []


def test_high_corr_peak_sync_used():
    dc = make_dc(min_corr_peak=0.5)
    dc.feed_sync(make_sync(0, corr_peak=0.8))
    results = dc.feed_onset(make_onset(1000))
    assert len(results) == 1


# ---------------------------------------------------------------------------
# pps_anchored flag
# ---------------------------------------------------------------------------

def test_pps_anchored_false_by_default():
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_onset(make_onset(256))
    assert results[0].pps_anchored is False


def test_pps_anchored_propagates():
    dc = make_dc(pps_anchored=True)
    dc.feed_sync(make_sync(0))
    results = dc.feed_onset(make_onset(256))
    assert results[0].pps_anchored is True


# ---------------------------------------------------------------------------
# Onset arrives before any sync - buffered
# ---------------------------------------------------------------------------

def test_onset_before_sync_buffered_then_resolved():
    dc = make_dc()
    # Onset arrives first - no measurement yet
    r1 = dc.feed_onset(make_onset(500))
    assert r1 == []

    # Sync arrives after onset - onset is resolved
    dc.feed_sync(make_sync(200))
    # Trigger flush by feeding another onset
    r2 = dc.feed_onset(make_onset(2000))

    # Both onsets should now be resolved
    assert any(m.snippet_start_sample == 500  for m in r2)
    assert any(m.snippet_start_sample == 2000 for m in r2)


# ---------------------------------------------------------------------------
# Aged-out onset is dropped
# ---------------------------------------------------------------------------

def test_onset_aged_out_dropped(caplog):
    """Onset that never finds a sync within max_sync_age is dropped."""
    import logging
    dc = make_dc(max_sync_age_samples=100)

    # Onset at sample 0, but sync arrives very late
    r1 = dc.feed_onset(make_onset(0))
    assert r1 == []

    # Feed a sync far in the future - onset should be dropped
    dc.feed_sync(make_sync(200))
    with caplog.at_level(logging.WARNING):
        r2 = dc.feed_onset(make_onset(300))

    # The original onset at 0 is too old; only the new one at 300 resolves
    assert all(m.snippet_start_sample != 0 for m in r2)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    dc.feed_onset(make_onset(256))
    dc.reset()

    # After reset, a new onset should find no sync
    results = dc.feed_onset(make_onset(512))
    assert results == []


# ---------------------------------------------------------------------------
# Offset-triggered measurements
# ---------------------------------------------------------------------------

def test_offset_produces_measurement():
    """feed_offset() produces a TDOAMeasurement just like feed_onset()."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_offset(make_offset(2560))
    assert len(results) == 1
    m = results[0]
    assert isinstance(m, TDOAMeasurement)
    assert m.snippet_start_sample == 2560
    assert m.sync_sample == 0


def test_onset_event_type():
    """Onset-triggered measurements have event_type == 'onset'."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_onset(make_onset(1000))
    assert results[0].event_type == "onset"


def test_offset_event_type():
    """Offset-triggered measurements have event_type == 'offset'."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_offset(make_offset(1000))
    assert results[0].event_type == "offset"


def test_offset_delta_ns_value():
    """sync_to_snippet_start_ns is computed the same way for offset as for onset."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    results = dc.feed_offset(make_offset(2560))  # 10 ms at 256 kHz
    expected_ns = int(round(2560 * 1e9 / RATE))
    assert results[0].sync_to_snippet_start_ns == expected_ns


def test_onset_and_offset_same_sync():
    """Both onset and offset match the same preceding sync event."""
    dc = make_dc()
    dc.feed_sync(make_sync(0))
    r_on  = dc.feed_onset(make_onset(1000))
    r_off = dc.feed_offset(make_offset(3000))
    assert len(r_on) == 1 and len(r_off) == 1
    assert r_on[0].sync_sample == r_off[0].sync_sample == 0
    assert r_off[0].sync_to_snippet_start_ns > r_on[0].sync_to_snippet_start_ns


def test_offset_before_sync_buffered():
    """Offset arriving before any sync is buffered and resolved when sync arrives."""
    dc = make_dc()
    r1 = dc.feed_offset(make_offset(500))
    assert r1 == []

    dc.feed_sync(make_sync(200))
    r2 = dc.feed_offset(make_offset(2000))
    assert any(m.snippet_start_sample == 500  for m in r2)
    assert any(m.snippet_start_sample == 2000 for m in r2)


def test_offset_aged_out_dropped(caplog):
    """Offset that cannot find a sync within max_sync_age is dropped."""
    import logging
    dc = make_dc(max_sync_age_samples=100)

    r1 = dc.feed_offset(make_offset(0))
    assert r1 == []

    dc.feed_sync(make_sync(200))
    with caplog.at_level(logging.WARNING):
        r2 = dc.feed_offset(make_offset(300))

    assert all(m.snippet_start_sample != 0 for m in r2)


def test_reset_clears_pending_offset():
    """reset() also clears pending offset events."""
    dc = make_dc()
    dc.feed_offset(make_offset(500))   # buffered (no sync yet)
    dc.reset()

    dc.feed_sync(make_sync(200))
    results = dc.feed_offset(make_offset(1000))
    # Only the new offset resolves; the pre-reset one was cleared
    assert len(results) == 1
    assert results[0].snippet_start_sample == 1000


# ---------------------------------------------------------------------------
# Pilot quality warning hysteresis
# ---------------------------------------------------------------------------

def test_pilot_single_bad_event_no_warning(caplog):
    """A single bad sync event must NOT produce a warning (hysteresis requires 5)."""
    import logging
    dc = make_dc(min_corr_peak=0.5)
    with caplog.at_level(logging.WARNING):
        dc.feed_sync(make_sync(0, corr_peak=0.1))   # one bad event
    assert not any("below threshold" in r.message for r in caplog.records)


def test_pilot_warning_fires_after_warn_after_consecutive_bad(caplog):
    """Warning is logged after _WARN_AFTER (5) consecutive bad events."""
    import logging
    dc = make_dc(min_corr_peak=0.5)
    with caplog.at_level(logging.WARNING):
        for i in range(dc._WARN_AFTER):
            dc.feed_sync(make_sync(i * 100, corr_peak=0.1))
    assert any("below threshold" in r.message for r in caplog.records)


def test_pilot_brief_dip_no_warning(caplog):
    """Alternating good/bad events never accumulate to the warn threshold."""
    import logging
    dc = make_dc(min_corr_peak=0.5)
    with caplog.at_level(logging.WARNING):
        for i in range(20):
            # alternating bad / good - count resets on each good event
            dc.feed_sync(make_sync(i * 200,       corr_peak=0.1))   # bad
            dc.feed_sync(make_sync(i * 200 + 100, corr_peak=0.9))   # good
    assert not any("below threshold" in r.message for r in caplog.records)


def test_pilot_recovery_requires_n_consecutive_good(caplog):
    """Recovery info log requires _RECOVER_AFTER (5) consecutive good events."""
    import logging
    dc = make_dc(min_corr_peak=0.5)
    # Trigger the warning
    for i in range(dc._WARN_AFTER):
        dc.feed_sync(make_sync(i * 100, corr_peak=0.1))
    caplog.clear()

    # Feed fewer than _RECOVER_AFTER good events - no recovery log yet
    with caplog.at_level(logging.INFO):
        for i in range(dc._RECOVER_AFTER - 1):
            dc.feed_sync(make_sync(1000 + i * 100, corr_peak=0.9))
    assert not any("recovered" in r.message for r in caplog.records)

    # One more good event tips us over - recovery is logged
    with caplog.at_level(logging.INFO):
        dc.feed_sync(make_sync(2000, corr_peak=0.9))
    assert any("recovered" in r.message for r in caplog.records)


def test_pilot_recovery_resets_state(caplog):
    """After recovery, a fresh bad streak can warn again."""
    import logging
    dc = make_dc(min_corr_peak=0.5)

    # First bad streak -> warning
    for i in range(dc._WARN_AFTER):
        dc.feed_sync(make_sync(i * 100, corr_peak=0.1))
    # Recovery
    for i in range(dc._RECOVER_AFTER):
        dc.feed_sync(make_sync(1000 + i * 100, corr_peak=0.9))
    caplog.clear()

    # Second bad streak should warn again after _WARN_AFTER events
    with caplog.at_level(logging.WARNING):
        for i in range(dc._WARN_AFTER):
            dc.feed_sync(make_sync(2000 + i * 100, corr_peak=0.1))
    assert any("below threshold" in r.message for r in caplog.records)


def test_reset_clears_pilot_warning_state():
    """reset() clears _consecutive_good and _pilot_warned so fresh state starts clean."""
    dc = make_dc(min_corr_peak=0.5)
    # Put into warned state
    for i in range(dc._WARN_AFTER):
        dc.feed_sync(make_sync(i * 100, corr_peak=0.1))
    assert dc._pilot_warned is True

    dc.reset()
    assert dc._pilot_warned is False
    assert dc._consecutive_good == 0
    assert dc._rejected_sync_count == 0


# ---------------------------------------------------------------------------
# Sync event pruning in feed_sync() - memory-leak regression tests
# ---------------------------------------------------------------------------

class TestSyncEventPruning:
    """
    _sync_events must not grow without bound during quiet periods.

    Root cause of bug: _sync_events was only pruned inside _flush(), which is
    only called by feed_onset/feed_offset.  During carrier-free periods the FM
    pilot fires at 100 Hz, accumulating millions of SyncEvent objects over hours
    (all reachable -> 0 objects collected by GC -> escalating gen2 pause times).

    Fix: feed_sync() also prunes after each append, keeping only events within
    max_sync_age_samples of the newest sync.
    """

    def test_sync_events_bounded_during_quiet_period(self):
        """
        Feeding many sync events with no carrier activity must not accumulate
        all of them in _sync_events.
        """
        max_age = 1000
        dc = make_dc(max_sync_age_samples=max_age)
        sync_period = 100  # samples between consecutive sync events

        # Simulate 8 hours of sync events with no carrier activity.
        # At 100 Hz, 8 h = 2_880_000 events.  Use a smaller count here but
        # enough to verify the list stays bounded, not growing linearly.
        n_syncs = 5_000
        for i in range(n_syncs):
            dc.feed_sync(make_sync(i * sync_period))

        # The list must be bounded by max_age/sync_period + 1 (approx), not by n_syncs.
        assert len(dc._sync_events) <= max_age // sync_period + 2, (
            f"_sync_events has {len(dc._sync_events)} entries after {n_syncs} syncs "
            f"with max_age={max_age} - expected O({max_age // sync_period}), not O(n_syncs)"
        )

    def test_pruned_syncs_do_not_prevent_matching(self):
        """
        A carrier event arriving after many quiet syncs must still find a recent
        matching sync event.
        """
        max_age = 1000
        dc = make_dc(max_sync_age_samples=max_age)

        # Feed many syncs during quiet period
        n_syncs = 200
        sync_period = 100
        for i in range(n_syncs):
            dc.feed_sync(make_sync(i * sync_period))

        # Carrier event close to the last sync: should match
        last_sync_sample = (n_syncs - 1) * sync_period
        onset_sample = last_sync_sample + 50  # 50 samples after last sync
        results = dc.feed_onset(make_onset(onset_sample))

        assert len(results) == 1, "Expected one match after quiet period"
        assert results[0].sync_sample == last_sync_sample

    def test_old_syncs_not_usable_after_pruning(self):
        """
        After quiet-period pruning, syncs older than max_age are gone and
        cannot be used to match a carrier event that arrives much later.
        This is correct behaviour: such a match would also be rejected by
        _match()'s max_sync_age_samples guard anyway.
        """
        max_age = 500
        dc = make_dc(max_sync_age_samples=max_age)

        # One early sync; then a long quiet period pushes it out of the window
        dc.feed_sync(make_sync(0))
        for i in range(20):
            dc.feed_sync(make_sync((i + 1) * 100))  # up to sample 2000

        # Carrier event far in the future - the sync at 0 is pruned AND too old
        results = dc.feed_onset(make_onset(3000))
        # No sync within max_age=500 of sample 3000 (last sync is at 2000, gap=1000)
        assert results == []

    def test_flush_pruning_runs_when_pending_empty(self):
        """
        When _pending_events is empty and a carrier event arrives, _flush()
        pruning must run on _sync_events (previously skipped due to the
        'if self._pending_events:' guard).
        """
        max_age = 500
        dc = make_dc(max_sync_age_samples=max_age)

        # Feed some syncs then a carrier that resolves immediately
        dc.feed_sync(make_sync(0))
        dc.feed_sync(make_sync(100))
        dc.feed_sync(make_sync(200))
        dc.feed_onset(make_onset(250))  # resolves -> pending empty after flush

        # Feed more syncs (advancing frontier) then another carrier.
        # The _flush() pruning with the fixed 'else' branch should trim old syncs.
        for i in range(5):
            dc.feed_sync(make_sync(300 + i * max_age))  # up to sample 300 + 4*500 = 2300
        # Carrier at end
        dc.feed_onset(make_onset(2350))

        # Sync at 0 is far behind the current frontier (2350); it should be pruned
        remaining = [s.sample_index for s in dc._sync_events]
        assert 0 not in remaining, (
            f"Stale sync at sample 0 still present; _flush() else-branch pruning failed. "
            f"Remaining: {remaining}"
        )
