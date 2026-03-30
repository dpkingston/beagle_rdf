# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for beagle_server/solver.py."""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from beagle_server.solver import FixResult, solve_fix
from beagle_server.tdoa import haversine_m

_C_M_S = 299_792_458.0

# ---------------------------------------------------------------------------
# Synthetic geometry helpers
# ---------------------------------------------------------------------------

# Three nodes in a triangle around a central search area (Seattle-like).
# We place nodes and a known transmitter, then compute the "true" sync_delta_ns
# for each node by computing the time the signal would arrive from the
# transmitter, scaled by the sample clock.
#
# The FM sync transmitter is at KISW (47.6253, -122.3563).
# We use a simplified model: all nodes have sync_delta_ns = constant + arrival_delay_ns
# where arrival_delay_ns = dist(tx, node) / c * 1e9.
#
# This produces exact TDOA = 0 when corrected -- which would place the fix at
# the transmitter.  To test a real fix scenario, we place a target transmitter
# at a different location and derive sync_deltas from there.

SYNC_TX_LAT = 47.6253
SYNC_TX_LON = -122.3563

# Three nodes forming a ~10 km triangle
NODES = [
    ("node-A", 47.700, -122.400),
    ("node-B", 47.620, -122.220),
    ("node-C", 47.540, -122.360),
]

# True target transmitter location
TARGET_LAT = 47.660
TARGET_LON = -122.310


def _dist(lat1, lon1, lat2, lon2):
    return haversine_m(lat1, lon1, lat2, lon2)


_SNIPPET_RATE_HZ = 1_000_000   # 1 MHz — gives ±1 µs = ~±300 m timing precision
_SNIPPET_LEN    = 10_000       # 10 ms @ 1 MHz — enough headroom for all onset offsets
_SNIPPET_BASE   = _SNIPPET_LEN // 4  # onset at 1/4 — matches real snippet encoder

# Signal model: all nodes share the same AM-modulated base snippet, shifted in
# time by their propagation delay relative to the closest node.  The TDOA is
# encoded as np.roll(base, -carrier_delay_samples): A's ring-buffer window starts
# carrier_delay_samples later in wall-clock time, so the onset appears that many
# samples earlier (smaller index) in A's snippet.  Power-envelope xcorr
# (B * conj(A)) then recovers the correct positive lag for A farther than B.
_RAMP_SAMPLES = 32       # PA rise time; same for all nodes


def _make_am_base_snippet() -> np.ndarray:
    """Create the shared AM-modulated QPSK base snippet (onset at _SNIPPET_BASE)."""
    rng = np.random.default_rng(0xDEAD)
    bits_i = rng.integers(0, 2, _SNIPPET_LEN) * 2 - 1
    bits_q = rng.integers(0, 2, _SNIPPET_LEN) * 2 - 1
    qpsk = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2)
    # Slowly-varying AM envelope so power-envelope xcorr can resolve timing offsets
    am_raw = np.abs(rng.standard_normal(_SNIPPET_LEN + 16))
    am_smooth = np.convolve(am_raw, np.ones(16) / 16, mode="valid")[:_SNIPPET_LEN]
    am_env = ((am_smooth - am_smooth.min()) /
              (am_smooth.max() - am_smooth.min()) * 0.6 + 0.4).astype(np.float32)
    carrier = (qpsk * am_env).astype(np.complex64)

    onset = _SNIPPET_BASE
    ramp_end = onset + _RAMP_SAMPLES
    env = np.zeros(_SNIPPET_LEN, dtype=np.float32)
    env[onset:ramp_end] = np.linspace(0.0, 1.0, _RAMP_SAMPLES, dtype=np.float32)
    env[ramp_end:] = 1.0
    return (carrier * env).astype(np.complex64)


_BASE_SNIPPET: np.ndarray = _make_am_base_snippet()


def _make_snippet_b64(carrier_delay_samples: int) -> str:
    """
    Create a base64-encoded int8 IQ snippet for a node with carrier_delay_samples
    additional propagation delay relative to the closest node.

    The base AM snippet is rolled left by carrier_delay_samples: the carrier
    onset shifts from _SNIPPET_BASE to _SNIPPET_BASE - carrier_delay_samples,
    reflecting that this node's ring-buffer window starts later in wall-clock
    time.  Power-envelope xcorr of two such snippets (B * conj(A)) recovers
    the correct positive lag = carrier_delay_A - carrier_delay_B.
    """
    iq = np.roll(_BASE_SNIPPET, -carrier_delay_samples)
    scale = float(np.max(np.abs(iq))) + 1e-30
    normed = iq / scale
    int8_ri = np.empty(_SNIPPET_LEN * 2, dtype=np.int8)
    int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
    int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
    return base64.b64encode(int8_ri.tobytes()).decode()


def make_synthetic_events(
    target_lat: float = TARGET_LAT,
    target_lon: float = TARGET_LON,
    base_delta_ns: int = 500_000_000,
) -> list[dict]:
    """
    Produce synthetic events such that the corrected TDOA between any two nodes
    equals (dist(target, A) - dist(target, B)) / c.

    sync_delta_ns_i = base + (dist(target, node_i) - dist(sync_tx, node_i)) / c * 1e9
    """
    # Compute sync_delta_ns for each node.
    sync_deltas: list[int] = []
    for _, nlat, nlon in NODES:
        target_delay_ns = _dist(target_lat, target_lon, nlat, nlon) / _C_M_S * 1e9
        sync_delay_ns   = _dist(SYNC_TX_LAT, SYNC_TX_LON, nlat, nlon) / _C_M_S * 1e9
        sync_deltas.append(int(base_delta_ns + target_delay_ns - sync_delay_ns))

    # Use the node with the smallest sync_delta as the carrier reference (delay=0).
    # All other nodes have a positive carrier delay proportional to how much later
    # they received the target carrier.  Power-envelope xcorr(A, B) of the rolled
    # snippets gives lag_ns ≈ (carrier_delay_A - carrier_delay_B) / rate * 1e9
    # = sync_delta_A - sync_delta_B  (within ±1 sample rounding from integer-
    # sample quantisation of the carrier delay at 1 MHz).
    min_delta = min(sync_deltas)

    events = []
    for (node_id, nlat, nlon), sync_delta_ns in zip(NODES, sync_deltas):
        carrier_delay = round((sync_delta_ns - min_delta) * _SNIPPET_RATE_HZ / 1e9)
        events.append({
            "event_id":               f"uuid-{node_id}",
            "node_id":                node_id,
            "channel_hz":             155_100_000.0,
            "event_type":             "onset",
            "sync_tx_id":             "KISW_99.9",
            "sync_tx_lat":            SYNC_TX_LAT,
            "sync_tx_lon":            SYNC_TX_LON,
            "node_lat":               nlat,
            "node_lon":               nlon,
            "sync_delta_ns":          sync_delta_ns,
            "corr_peak":              0.85,
            "onset_time_ns":          1_700_000_000_000_000_000,
            "iq_snippet_b64":         _make_snippet_b64(carrier_delay),
            "channel_sample_rate_hz": float(_SNIPPET_RATE_HZ),
        })
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_solve_fix_returns_none_for_single_node():
    events = make_synthetic_events()[:1]
    assert solve_fix(events, TARGET_LAT, TARGET_LON) is None


def test_solve_fix_returns_fix_result():
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert isinstance(result, FixResult)


@pytest.mark.xfail(
    reason=(
        "Synthetic snippets encode sync_delta differences, not physical propagation "
        "delays, so xcorr-as-primary returns the uncorrected lag (missing path "
        "correction), degrading fix accuracy.  Will pass once replaced by a real "
        "distributed node fixture."
    ),
    strict=False,
)
def test_solve_fix_three_nodes_accuracy():
    """
    With three nodes and exact synthetic TDOAs the solver should find the
    true transmitter location to within 500 m.
    """
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None

    error_m = haversine_m(result.latitude_deg, result.longitude_deg,
                          TARGET_LAT, TARGET_LON)
    assert error_m < 500.0, f"Fix error {error_m:.0f} m > 500 m"


def test_solve_fix_returns_none_when_no_tdoa_data():
    """
    If both sync_delta_ns and iq_snippet_b64 are absent, no TDOA method can
    fire and solve_fix returns None.
    """
    events = make_synthetic_events()
    for ev in events:
        del ev["sync_delta_ns"]
        ev.pop("iq_snippet_b64", None)
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is None


def test_solve_fix_node_count():
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3)
    assert result is not None
    assert result.node_count == 3


def test_solve_fix_node_ids():
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3)
    assert result is not None
    assert set(result.nodes) == {"node-A", "node-B", "node-C"}


def test_solve_fix_two_nodes_returns_result():
    """
    With only 2 nodes the problem is under-determined but the solver should
    still return a result (somewhere on the hyperbola near the search centre).
    """
    events = make_synthetic_events()[:2]
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.node_count == 2


def test_solve_fix_deduplicates_same_node():
    """Two events from the same node should count as one."""
    events = make_synthetic_events()
    # Duplicate node-A with slightly different delta (amendment)
    dup = dict(events[0])
    dup["event_id"] = "duplicate-id"
    dup["sync_delta_ns"] += 50
    events_with_dup = events + [dup]
    result = solve_fix(events_with_dup, 47.6, -122.3)
    assert result is not None
    assert result.node_count == 3  # still 3 unique nodes


def test_solve_fix_onset_time_set():
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3)
    assert result is not None
    assert result.onset_time_ns == 1_700_000_000_000_000_000


def test_solve_fix_channel_and_type():
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3)
    assert result is not None
    assert result.channel_hz == 155_100_000.0
    assert result.event_type == "onset"


def test_solve_fix_excluded_nodes_empty_by_default():
    """Normal fix with no outlier should have an empty excluded_nodes list."""
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.excluded_nodes == []


def test_solve_fix_outlier_node_detected_and_excluded():
    """
    When one node's sync_delta_ns is wildly wrong, the outlier detector should
    flag it and re-run the fix without it — provided at least 2 other nodes remain.

    We use 4 nodes so that after exclusion of the outlier, 3 remain, keeping
    the fix geometrically determined.  No IQ snippets: the sync_delta fallback
    path is exercised, so the corrupted sync_delta propagates directly to the
    TDOA residuals where the outlier detector identifies it.
    """
    four_nodes = [
        ("node-A", 47.700, -122.400),
        ("node-B", 47.620, -122.220),
        ("node-C", 47.540, -122.360),
        ("node-D", 47.590, -122.290),
    ]
    sync_deltas = []
    for _, nlat, nlon in four_nodes:
        target_delay_ns = _dist(TARGET_LAT, TARGET_LON, nlat, nlon) / _C_M_S * 1e9
        sync_delay_ns   = _dist(SYNC_TX_LAT, SYNC_TX_LON, nlat, nlon) / _C_M_S * 1e9
        sync_deltas.append(int(500_000_000 + target_delay_ns - sync_delay_ns))

    events = []
    for (node_id, nlat, nlon), sync_delta_ns in zip(four_nodes, sync_deltas):
        events.append({
            "event_id":               f"uuid-{node_id}",
            "node_id":                node_id,
            "channel_hz":             155_100_000.0,
            "event_type":             "onset",
            "sync_tx_id":             "KISW_99.9",
            "sync_tx_lat":            SYNC_TX_LAT,
            "sync_tx_lon":            SYNC_TX_LON,
            "node_lat":               nlat,
            "node_lon":               nlon,
            "sync_delta_ns":          sync_delta_ns,
            "corr_peak":              0.85,
            "onset_time_ns":          1_700_000_000_000_000_000,
            # No iq_snippet_b64: xcorr doesn't fire; sync_delta fallback is used.
            # This ensures the corrupt sync_delta reaches the residual stage.
        })

    # Corrupt node-D's sync_delta_ns: add 1 ms (~300 km equivalent) of error.
    # The 3 good nodes produce small TDOA residuals; all pairs involving
    # node-D have ~1 ms residual, so rms_excluding_D << rms_all / 3.
    events[3]["sync_delta_ns"] += 1_000_000  # +1 ms

    result = solve_fix(
        events, 47.6, -122.3,
        search_radius_km=80.0,
    )
    assert result is not None
    assert "node-D" in result.excluded_nodes, (
        f"Expected node-D flagged as outlier; excluded={result.excluded_nodes}"
    )
    assert "node-D" not in result.nodes
    # Fix should be close to the true target after outlier exclusion
    error_m = haversine_m(result.latitude_deg, result.longitude_deg, TARGET_LAT, TARGET_LON)
    assert error_m < 2_000.0, f"Fix error after outlier exclusion: {error_m:.0f} m"
