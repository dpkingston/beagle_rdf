# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for the live per-pair TDOA auto-calibrator."""

from __future__ import annotations

import pytest

from beagle_server.target_calibration import TargetCalibrator, _canonical
from beagle_server.tdoa import _C_M_S, haversine_m

# Capitol Hill 2 m calibration target + the two synced test nodes.
TGT_LAT, TGT_LON, TGT_HZ = 47.62389, -122.31517, 146_960_000.0
NODE = {
    "dpk-tdoa1": (47.671928, -122.404209),
    "dpk-tdoa2": (47.721666, -122.359034),
}


def _ev(node_id, channel_hz=TGT_HZ, event_type="plateau"):
    lat, lon = NODE[node_id]
    return {
        "node_id": node_id, "node_lat": lat, "node_lon": lon,
        "channel_hz": channel_hz, "event_type": event_type,
    }


def _expected_tdoa_s(lo_id, hi_id):
    la, lo = NODE[lo_id]; lb, lob = NODE[hi_id]
    return (haversine_m(TGT_LAT, TGT_LON, la, lo)
            - haversine_m(TGT_LAT, TGT_LON, lb, lob)) / _C_M_S


def _cal(**kw):
    return TargetCalibrator(TGT_LAT, TGT_LON, TGT_HZ, min_samples=5, **kw)


def test_canonical_orientation():
    assert _canonical("a", "b") == ("a", "b", 1.0)
    assert _canonical("b", "a") == ("a", "b", -1.0)


def test_recovers_known_bias():
    """If every raw measurement is expected + a fixed bias, the published
    offset equals that bias."""
    cal = _cal()
    bias_s = 220e-6  # +220 µs fixed per-pair bias
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    for _ in range(10):
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + bias_s,
                    _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
    off = cal.pair_offsets_s()
    assert "dpk-tdoa1,dpk-tdoa2" in off
    assert off["dpk-tdoa1,dpk-tdoa2"] == pytest.approx(bias_s, abs=1e-9)


def test_orientation_independent():
    """Observing the pair in reverse node order records the SAME canonical
    bias (so the offset is consistent regardless of which node the solver
    iterated first)."""
    cal_fwd = _cal()
    cal_rev = _cal()
    bias_s = 100e-6
    exp_fwd = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    exp_rev = _expected_tdoa_s("dpk-tdoa2", "dpk-tdoa1")
    for _ in range(8):
        cal_fwd.observe("dpk-tdoa1", "dpk-tdoa2", exp_fwd + bias_s,
                        _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
        # reverse orientation: raw is negated, expected is negated
        cal_rev.observe("dpk-tdoa2", "dpk-tdoa1", -(exp_fwd + bias_s),
                        _ev("dpk-tdoa2"), _ev("dpk-tdoa1"))
    assert cal_fwd.pair_offsets_s()["dpk-tdoa1,dpk-tdoa2"] == pytest.approx(
        cal_rev.pair_offsets_s()["dpk-tdoa1,dpk-tdoa2"], abs=1e-12)
    assert cal_fwd.pair_offsets_s()["dpk-tdoa1,dpk-tdoa2"] == pytest.approx(bias_s, abs=1e-9)


def test_median_rejects_outliers():
    """The median ignores a minority of wild (tone-only-style) outliers a
    mean would be dragged by."""
    cal = _cal()
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    true_bias = 50e-6
    # 7 clean + 3 wild outliers (±10 ms tonal-lock style)
    obs = [true_bias] * 7 + [10e-3, -9e-3, 11e-3]
    for b in obs:
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + b, _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
    off = cal.pair_offsets_s()["dpk-tdoa1,dpk-tdoa2"]
    assert off == pytest.approx(true_bias, abs=1e-9)


def test_min_samples_gate():
    """No offset is published until min_samples observations accrue."""
    cal = _cal()  # min_samples=5
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    for _ in range(4):
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + 1e-4, _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
    assert cal.pair_offsets_s() == {}
    cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + 1e-4, _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
    assert "dpk-tdoa1,dpk-tdoa2" in cal.pair_offsets_s()


def test_skips_wrong_channel():
    """Off-target-channel events are not used for calibration."""
    cal = _cal()
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    for _ in range(10):
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + 1e-4,
                    _ev("dpk-tdoa1", channel_hz=145_000_000.0),
                    _ev("dpk-tdoa2", channel_hz=145_000_000.0))
    assert cal.pair_offsets_s() == {}
    assert cal.health_snapshot()["total_skipped"] == 10
    assert cal.health_snapshot()["total_observed"] == 0


def test_skips_non_plateau():
    cal = _cal()
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    for _ in range(10):
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + 1e-4,
                    _ev("dpk-tdoa1", event_type="onset"),
                    _ev("dpk-tdoa2", event_type="onset"))
    assert cal.pair_offsets_s() == {}


def test_window_bounds_memory():
    """Only the last `window` observations are retained per pair."""
    cal = TargetCalibrator(TGT_LAT, TGT_LON, TGT_HZ, window=10, min_samples=5)
    exp = _expected_tdoa_s("dpk-tdoa1", "dpk-tdoa2")
    for i in range(100):
        cal.observe("dpk-tdoa1", "dpk-tdoa2", exp + 1e-4, _ev("dpk-tdoa1"), _ev("dpk-tdoa2"))
    assert cal.health_snapshot()["pairs"]["dpk-tdoa1,dpk-tdoa2"]["n"] == 10


def test_health_snapshot_shape():
    cal = _cal()
    snap = cal.health_snapshot()
    assert snap["target_channel_mhz"] == pytest.approx(146.96, abs=0.001)
    assert snap["total_observed"] == 0
    assert snap["pairs"] == {}
