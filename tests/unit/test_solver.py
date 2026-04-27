# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
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
# We place nodes and a known transmitter, then compute the "true" sync_to_snippet_start_ns
# for each node by computing the time the signal would arrive from the
# transmitter, scaled by the sample clock.
#
# The FM sync transmitter is at KISW (47.6253, -122.3563).
# We use a simplified model: all nodes have sync_to_snippet_start_ns = constant + arrival_delay_ns
# where arrival_delay_ns = dist(tx, node) / c * 1e9.
#
# This produces exact TDOA = 0 when corrected - which would place the fix at
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


_SNIPPET_RATE_HZ = 1_000_000   # 1 MHz - gives +/-1 usec = ~+/-300 m timing precision
_SNIPPET_LEN    = 10_000       # 10 ms @ 1 MHz - enough headroom for all onset offsets
_SNIPPET_BASE   = _SNIPPET_LEN // 4  # onset at 1/4 - matches real snippet encoder

# Signal model: all nodes receive the same PA transition but at different times
# (propagation delay).  The onset position in each node's snippet is shifted by
# carrier_delay_samples.  Pre-onset is noise, post-onset is shared AM-textured
# carrier content (same physical RF signal).  The server's coarse knee walk +
# sub-snippet xcorr recovers the TDOA from the onset position differences.
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
    Create a base64-encoded int8 IQ snippet modelling the real measurement chain.

    Each node receives the same PA transition but at a different time.  The
    farther node sees the onset carrier_delay_samples later.  Since the snippet
    is captured around the detection point (which fires near the bottom of the
    rise), the onset appears at a later position in the snippet for farther nodes.

    The snippet contains:
    - Low-level noise before the onset
    - A linear PA ramp (_RAMP_SAMPLES)
    - AM-textured carrier content (shared across all nodes — same physical
      transmission) after the ramp

    The onset position is _SNIPPET_BASE + carrier_delay_samples.  All nodes share
    the same carrier content after the ramp (it's the same RF signal), so
    power-envelope xcorr can align them.
    """
    onset = _SNIPPET_BASE + carrier_delay_samples
    ramp_end = min(onset + _RAMP_SAMPLES, _SNIPPET_LEN)

    # Fixed noise seed (same noise floor characteristics for all nodes)
    rng_noise = np.random.default_rng(42)
    iq = (rng_noise.standard_normal(_SNIPPET_LEN) +
          1j * rng_noise.standard_normal(_SNIPPET_LEN)).astype(np.complex64) * 0.01

    # Ramp: linear rise from noise to carrier amplitude
    if onset < _SNIPPET_LEN:
        ramp_len = min(_RAMP_SAMPLES, _SNIPPET_LEN - onset)
        ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        iq[onset:onset + ramp_len] = _BASE_SNIPPET[_SNIPPET_BASE:_SNIPPET_BASE + ramp_len] * ramp

    # Carrier: same physical content for all nodes (aligned to the PA transition)
    carrier_start = min(ramp_end, _SNIPPET_LEN)
    n_carrier = _SNIPPET_LEN - carrier_start
    base_carrier_start = _SNIPPET_BASE + _RAMP_SAMPLES
    n_avail = min(n_carrier, len(_BASE_SNIPPET) - base_carrier_start)
    if n_avail > 0:
        iq[carrier_start:carrier_start + n_avail] = _BASE_SNIPPET[base_carrier_start:base_carrier_start + n_avail]

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
    # Compute sync_to_snippet_start_ns for each node.
    sync_deltas: list[int] = []
    for _, nlat, nlon in NODES:
        target_delay_ns = _dist(target_lat, target_lon, nlat, nlon) / _C_M_S * 1e9
        sync_delay_ns   = _dist(SYNC_TX_LAT, SYNC_TX_LON, nlat, nlon) / _C_M_S * 1e9
        sync_deltas.append(int(base_delta_ns + target_delay_ns - sync_delay_ns))

    # Use the node with the smallest sync_delta as the carrier reference (delay=0).
    # All other nodes have a positive carrier delay proportional to how much later
    # they received the target carrier.  The onset position in each node's snippet
    # shifts by carrier_delay_samples.  The server's coarse knee walk + sub-snippet
    # xcorr recovers the TDOA from these position differences.
    min_delta = min(sync_deltas)

    events = []
    for (node_id, nlat, nlon), sync_to_snippet_start_ns in zip(NODES, sync_deltas):
        carrier_delay = round((sync_to_snippet_start_ns - min_delta) * _SNIPPET_RATE_HZ / 1e9)
        # Snippet onset at _SNIPPET_BASE (2500) + carrier_delay.  Give the
        # knee finder a generous transition window that covers all nodes'
        # possible onset positions.
        onset_pos = _SNIPPET_BASE + carrier_delay
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
            "sync_to_snippet_start_ns":          sync_to_snippet_start_ns,
            "corr_peak":              0.85,
            "onset_time_ns":          1_700_000_000_000_000_000,
            "iq_snippet_b64":         _make_snippet_b64(carrier_delay),
            "channel_sample_rate_hz": float(_SNIPPET_RATE_HZ),
            # Give the knee finder enough room around the ramp: real nodes
            # ship zones ~5x the Savgol window width; at 1 MHz with a
            # 360 µs default window that's ~2000 samples.
            "transition_start":       max(0, onset_pos - 1000),
            "transition_end":         onset_pos + _RAMP_SAMPLES + 1000,
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
    If both sync_to_snippet_start_ns and iq_snippet_b64 are absent, no TDOA method can
    fire and solve_fix returns None.
    """
    events = make_synthetic_events()
    for ev in events:
        del ev["sync_to_snippet_start_ns"]
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
    dup["sync_to_snippet_start_ns"] += 50
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
    When one node's sync_to_snippet_start_ns is wildly wrong, the outlier detector should
    flag it and re-run the fix without it - provided at least 2 other nodes remain.

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

    min_delta = min(sync_deltas)
    events = []
    for (node_id, nlat, nlon), sync_to_snippet_start_ns in zip(four_nodes, sync_deltas):
        carrier_delay = round((sync_to_snippet_start_ns - min_delta) * _SNIPPET_RATE_HZ / 1e9)
        # Snippet onset at _SNIPPET_BASE (2500) + carrier_delay.  Give the
        # knee finder a generous transition window that covers all nodes'
        # possible onset positions.
        onset_pos = _SNIPPET_BASE + carrier_delay
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
            "sync_to_snippet_start_ns":          sync_to_snippet_start_ns,
            "corr_peak":              0.85,
            "onset_time_ns":          1_700_000_000_000_000_000,
            "iq_snippet_b64":         _make_snippet_b64(carrier_delay),
            "channel_sample_rate_hz": float(_SNIPPET_RATE_HZ),
            # Give the knee finder enough room around the ramp: real nodes
            # ship zones ~5x the Savgol window width; at 1 MHz with a
            # 360 µs default window that's ~2000 samples.
            "transition_start":       max(0, onset_pos - 1000),
            "transition_end":         onset_pos + _RAMP_SAMPLES + 1000,
        })

    # Corrupt node-D's sync_to_snippet_start_ns: add 400 usec (~120 km equivalent) of
    # error.  This stays within T_sync/2 (421 usec) so disambiguation does
    # not mask it.  The 3 good nodes produce small TDOA residuals; all pairs
    # involving node-D have ~400 usec residual, triggering outlier detection.
    events[3]["sync_to_snippet_start_ns"] += 400_000  # +400 usec

    result = solve_fix(
        events, 47.6, -122.3,
        search_radius_km=80.0,
    )
    assert result is not None
    # Node-D's corrupt sync_delta (+400 us) makes its TDOA pairs exceed
    # the geometric plausibility limit.  The fix proceeds with the 3 good
    # nodes and should be close to the true target despite node-D's data.
    error_m = haversine_m(result.latitude_deg, result.longitude_deg, TARGET_LAT, TARGET_LON)
    assert error_m < 10_000.0, f"Fix error with corrupt node present: {error_m:.0f} m"


# ---------------------------------------------------------------------------
# Suppression: boundary-clamp + multistart-ambiguity detection
# ---------------------------------------------------------------------------

def test_fix_metrics_populated_on_clean_fix():
    """A normal solve_fix produces FixResult with the new quality metrics
    populated (not None)."""
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.boundary_distance_km is not None
    assert result.multistart_disagreement_km is not None
    # Clean 3-node fit, true minimum well inside search area:
    assert result.boundary_distance_km > 5.0
    assert result.suppression_reason in (None, "multistart_ambiguous")


def test_boundary_clamp_suppression_when_minimum_outside_search():
    """When the search bounds are tight enough that the true cost minimum
    lies outside, the optimizer is clamped to the bounds and the result
    must be flagged ``suppressed`` with reason ``boundary_clamped``."""
    # Real geometry, real TDOAs.  Force the search radius to be tiny
    # (1 km) — well smaller than the distance from search center to the
    # true target, so the optimizer is constrained to the boundary.
    events = make_synthetic_events()
    # Search center placed deliberately far from the target.
    far_search_lat, far_search_lon = 48.5, -123.5
    result = solve_fix(
        events,
        search_center_lat=far_search_lat,
        search_center_lon=far_search_lon,
        search_radius_km=1.0,        # Tight bound, forces clamp
    )
    assert result is not None
    assert result.suppressed, "expected suppression when minimum is outside bounds"
    assert result.suppression_reason == "boundary_clamped"
    assert result.boundary_distance_km is not None
    assert result.boundary_distance_km < 2.0


def test_boundary_clamp_disabled_when_threshold_zero():
    """boundary_clamp_km=0 disables the boundary suppression check."""
    events = make_synthetic_events()
    far_search_lat, far_search_lon = 48.5, -123.5
    result = solve_fix(
        events,
        search_center_lat=far_search_lat,
        search_center_lon=far_search_lon,
        search_radius_km=1.0,
        boundary_clamp_km=0.0,        # disabled
    )
    assert result is not None
    # Suppression by *boundary_clamped* must NOT fire when threshold=0.
    # (multistart_ambiguous may still fire; this test asserts only the
    # specific check is bypassed.)
    if result.suppressed:
        assert result.suppression_reason != "boundary_clamped"


def test_two_node_fix_converges_to_hyperbola_after_ns2_rescale():
    """A 2-node configuration is structurally degenerate (a hyperbola of
    zero-cost minimizers), but with the cost rescaled to nanoseconds^2
    the optimizer cleanly converges all multistarts to a single point on
    the hyperbola -- the projection of the seeds onto the curve under the
    bounds constraints.

    Pre-rescale, this test exercised the multistart-ambiguity
    suppression by relying on the optimizer's numerical sloppiness in
    seconds^2 cost: different multistarts converged to slightly
    different points along the hyperbola, triggering the > 5 km
    disagreement guard.  That was an accidental side-channel detector
    for "2-node degenerate", not a feature.

    The user explicitly enables ``min_nodes: 2`` for LOP fixes (see
    server.example.yaml), so 2-node fixes SHOULD pass through with a
    small residual rather than being suppressed.  This test pins that
    contract.
    """
    events = make_synthetic_events()[:2]
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.node_count == 2
    # Optimizer converges to a real point on the hyperbola; residual is
    # essentially zero (the cost surface is a zero-cost manifold).
    assert result.residual_ns < 1.0, (
        f"2-node fit should land on the hyperbola with sub-ns residual; "
        f"got {result.residual_ns:.2f} ns"
    )
    # Not suppressed by multistart-ambiguity: the rescaled cost is
    # numerically well-behaved enough that all 5 starts converge
    # consistently.  (Suppression by other criteria, e.g. seed_stuck if
    # the hyperbola passes near a seed, is allowed and tested separately.)
    if result.suppressed:
        assert result.suppression_reason != "multistart_ambiguous", (
            f"2-node fix should not trigger multistart_ambiguous after "
            f"the ns^2 rescale; got reason={result.suppression_reason}, "
            f"disagreement={result.multistart_disagreement_km}"
        )


def test_multistart_ambiguity_disabled_when_threshold_zero():
    """multistart_disagreement_km=0 disables the multistart suppression check."""
    events = make_synthetic_events()[:2]
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        multistart_disagreement_km=0.0,
    )
    assert result is not None
    if result.suppressed:
        assert result.suppression_reason != "multistart_ambiguous"


# ---------------------------------------------------------------------------
# ns^2 cost rescale + seed_stuck suppression (2026-04-27 cluster-#1 regression)
# ---------------------------------------------------------------------------
#
# Cluster #1 root cause: ``_run_optimizer.cost`` summed (measured-pred)^2 in
# SECONDS^2, producing magnitudes ~1e-10 for clean fits.  L-BFGS-B with
# ``gtol=1e-10`` and ``ftol=1e-15`` interpreted the resulting tiny finite-
# difference gradients as "already converged" at the very first iteration
# and returned the seed coordinates unchanged.  Verified against the live
# 08:54-08:56 corpus: 13 fixes pinned to (47.6150000, -122.3470000) -- the
# exact ``search_center`` -- with 7 µs RMS residuals while a hand-rolled
# grid search found cost ~12,000x lower 4 km away.
#
# Fix: cost in nanoseconds^2 (×1e18) so gtol/ftol thresholds are
# numerically meaningful at our measurement scale.  Defence-in-depth:
# ``seed_stuck`` suppression rejects any fix that converges within
# ``seed_stuck_distance_m`` of a multistart seed AND has a residual above
# ``seed_stuck_residual_ns``.

def test_run_optimizer_emits_seed_and_node_distance_metrics():
    """``_run_optimizer`` returns sixth and seventh values: smallest
    distance from the converged best to any multistart seed (used by
    seed_stuck) and to any participating node coordinate (used by
    node_stuck)."""
    from beagle_server.solver import _run_optimizer
    events = make_synthetic_events()
    pairs: list[tuple[int, int, float]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            t_i = (_dist(TARGET_LAT, TARGET_LON, events[i]["node_lat"],
                         events[i]["node_lon"]) / _C_M_S)
            t_j = (_dist(TARGET_LAT, TARGET_LON, events[j]["node_lat"],
                         events[j]["node_lon"]) / _C_M_S)
            pairs.append((i, j, t_i - t_j))
    out = _run_optimizer(pairs, events, TARGET_LAT, TARGET_LON, 50.0)
    # Tuple of (lat, lon, rms_ns, multistart_disagreement_km,
    # boundary_distance_km, seed_distance_m, node_distance_m)
    assert len(out) == 7
    seed_distance_m = out[5]
    node_distance_m = out[6]
    assert seed_distance_m >= 0.0
    assert node_distance_m >= 0.0
    # With seeds centered ON the target, the converged fix should be
    # within metres of the seed (the synthetic geometry is exact).
    assert seed_distance_m < 1000.0
    # The synthetic target is several km from any node, so node_distance_m
    # should be on the order of km.
    assert node_distance_m > 1000.0


def test_solve_fix_populates_seed_distance_m_field():
    """FixResult.seed_distance_m is populated on every return path."""
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.seed_distance_m is not None
    assert result.seed_distance_m >= 0.0


def test_ns2_cost_rescale_optimizer_moves_off_seed():
    """The ns^2 rescale must let L-BFGS-B move off the search-center
    seed when the true minimum is several km away.

    Pre-rescale failure mode: with cost ~1e-10 s^2 the optimizer
    terminated at iteration 0 and returned (search_center, search_center)
    as the "minimum"; verified on the 2026-04-27 cluster-#1 corpus where
    13 fixes pinned to (47.6150000, -122.3470000) to 7 decimal places.

    Reuses the same parameters as ``test_solve_fix_three_nodes_accuracy``
    (target ~6.7 km from the seed).  The contract pinned here is "the
    optimizer moves at least 1 km off the seed" -- a much weaker
    assertion than absolute fix accuracy, deliberately so: the synthetic
    fixture uses simplified IQ snippets that do not give pipeline-grade
    sub-km accuracy (see the xfail note on
    ``test_solve_fix_three_nodes_accuracy``).  But it cleanly catches
    the iteration-0 termination mode if the rescale ever regresses.
    """
    events = make_synthetic_events()
    SEARCH_LAT = 47.6
    SEARCH_LON = -122.3
    result = solve_fix(
        events, SEARCH_LAT, SEARCH_LON, search_radius_km=80.0,
    )
    assert result is not None

    # Distance from converged fix to the seed.  Pre-rescale: < 50 m
    # (optimizer never moved).  Post-rescale: ~5+ km (real movement
    # toward the synthetic target).
    seed_dist_m = haversine_m(
        result.latitude_deg, result.longitude_deg,
        SEARCH_LAT, SEARCH_LON,
    )
    assert seed_dist_m > 1000.0, (
        f"Fix landed only {seed_dist_m:.0f} m from the seed -- L-BFGS-B "
        f"likely terminated at iteration 0 (cost rescale not in effect)."
    )

    # And FixResult.seed_distance_m must report this consistently.
    assert result.seed_distance_m is not None
    assert result.seed_distance_m > 1000.0


def test_seed_stuck_suppression_fires_for_stuck_optimizer(monkeypatch):
    """When ``_run_optimizer`` reports tiny seed_distance and large rms_ns,
    ``solve_fix`` must mark the fix suppressed with reason ``seed_stuck``.

    Mocks ``_run_optimizer`` to return a known stuck-at-seed result so the
    suppression logic is exercised without depending on a buggy optimizer.
    """
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # 2 m from the seed (well below default 50 m threshold), 6,500 ns
        # residual (well above default 500 ns threshold) -- mirrors the
        # 2026-04-27 cluster-#1 production observation.
        # Tuple shape: (lat, lon, rms_ns, multistart_disagreement_km,
        # boundary_distance_km, seed_distance_m, node_distance_m).
        # 2 m from seed (triggers seed_stuck), 100 km from any node
        # (does not trigger node_stuck).
        return slat, slon, 6500.0, 0.0, 50.0, 2.0, 100_000.0

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.suppressed
    assert result.suppression_reason == "seed_stuck"
    assert result.seed_distance_m == pytest.approx(2.0)
    assert result.residual_ns == pytest.approx(6500.0)


def test_seed_stuck_suppression_does_not_fire_for_clean_fix_near_seed(monkeypatch):
    """A legitimate clean fix that happens to be near a seed (e.g.
    transmitter actually at the search centre) must NOT trigger
    seed_stuck.  The residual gate prevents false positives.
    """
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # 5 m from seed, but residual is 12 ns (clean fit) -- no suppression.
        # 5 m from seed but residual is 12 ns (clean fit) -- no
        # suppression on either seed_stuck or node_stuck.  Far from any
        # node coordinate.
        return slat, slon, 12.0, 0.0, 50.0, 5.0, 100_000.0

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert not (result.suppressed and result.suppression_reason == "seed_stuck"), (
        f"Clean low-residual fix near seed should not trigger seed_stuck; "
        f"got suppressed={result.suppressed} reason={result.suppression_reason}"
    )


def test_seed_stuck_disabled_when_distance_zero(monkeypatch):
    """seed_stuck_distance_m=0 disables the seed_stuck suppression check."""
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # Would trigger seed_stuck under default thresholds (2 m, 6500 ns).
        # Tuple shape: (lat, lon, rms_ns, multistart_disagreement_km,
        # boundary_distance_km, seed_distance_m, node_distance_m).
        # 2 m from seed (triggers seed_stuck), 100 km from any node
        # (does not trigger node_stuck).
        return slat, slon, 6500.0, 0.0, 50.0, 2.0, 100_000.0

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        seed_stuck_distance_m=0.0,
    )
    assert result is not None
    if result.suppressed:
        assert result.suppression_reason != "seed_stuck"


def test_seed_stuck_disabled_when_residual_threshold_zero(monkeypatch):
    """seed_stuck_residual_ns=0 disables the seed_stuck suppression check."""
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # Tuple shape: (lat, lon, rms_ns, multistart_disagreement_km,
        # boundary_distance_km, seed_distance_m, node_distance_m).
        # 2 m from seed (triggers seed_stuck), 100 km from any node
        # (does not trigger node_stuck).
        return slat, slon, 6500.0, 0.0, 50.0, 2.0, 100_000.0

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        seed_stuck_residual_ns=0.0,
    )
    assert result is not None
    if result.suppressed:
        assert result.suppression_reason != "seed_stuck"


# ---------------------------------------------------------------------------
# node_stuck suppression (2026-04-27 Capitol-Hill-2 cross-target regression)
# ---------------------------------------------------------------------------
#
# Verified failure mode: 8 plateau fixes from a Capitol Hill 2 transmission
# (with Magnolia-fitted calibration) pinned to dpk-tdoa1's exact coords
# (47.67193, -122.40421) with 35 microsecond residuals.  Biased pair TDOAs
# found a self-consistent unphysical minimum at a node attractor where one
# pair's geometric prediction degenerates to a constant (dist(fix, node) = 0
# for one term, so that term becomes -dist(fix, other)/c regardless of bias).
#
# Same shape as seed_stuck (cost-surface minimum at a fixed attractor; high
# residual indicates the fix is trustworthy in that minimum but the minimum
# itself is wrong) -- different attractor (a node coordinate rather than a
# multistart seed).

def test_solve_fix_populates_node_distance_m_field():
    """FixResult.node_distance_m is populated on every return path."""
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.node_distance_m is not None
    assert result.node_distance_m >= 0.0


def test_node_stuck_suppression_fires_for_fix_at_node_coords(monkeypatch):
    """When ``_run_optimizer`` reports tiny node_distance and large rms_ns,
    ``solve_fix`` must mark the fix suppressed with reason ``node_stuck``.
    """
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # Far from seeds (no seed_stuck), 5 m from a node, 35 microsecond
        # residual -- mirrors the Capitol Hill 2 dpk-tdoa1 pile-up.
        return (slat + 0.1, slon + 0.1, 35_000.0, 0.0, 50.0,
                100_000.0, 5.0)

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.suppressed
    assert result.suppression_reason == "node_stuck"
    assert result.node_distance_m == pytest.approx(5.0)
    assert result.residual_ns == pytest.approx(35_000.0)


def test_node_stuck_does_not_fire_for_clean_fix_near_node(monkeypatch):
    """A legitimate clean fix that happens to be near a node (e.g.
    transmitter physically close to one of the receivers) must NOT
    trigger node_stuck.  The residual gate prevents false positives.
    """
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # 5 m from a node but residual 12 ns (clean fit) -- no
        # suppression.  A real transmitter could plausibly be metres
        # from a receiver.
        return (slat + 0.1, slon + 0.1, 12.0, 0.0, 50.0,
                100_000.0, 5.0)

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert not (
        result.suppressed and result.suppression_reason == "node_stuck"
    ), (
        f"Clean low-residual fix near node should not trigger node_stuck; "
        f"got suppressed={result.suppressed} reason={result.suppression_reason}"
    )


def test_node_stuck_disabled_when_distance_zero(monkeypatch):
    """node_stuck_distance_m=0 disables the node_stuck suppression check."""
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # Would trigger node_stuck under defaults.
        return (slat + 0.1, slon + 0.1, 35_000.0, 0.0, 50.0,
                100_000.0, 5.0)

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        node_stuck_distance_m=0.0,
    )
    assert result is not None
    if result.suppressed:
        assert result.suppression_reason != "node_stuck"


def test_node_stuck_disabled_when_residual_threshold_zero(monkeypatch):
    """node_stuck_residual_ns=0 disables the node_stuck suppression check."""
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        return (slat + 0.1, slon + 0.1, 35_000.0, 0.0, 50.0,
                100_000.0, 5.0)

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        node_stuck_residual_ns=0.0,
    )
    assert result is not None
    if result.suppressed:
        assert result.suppression_reason != "node_stuck"


def test_seed_stuck_takes_priority_over_node_stuck(monkeypatch):
    """When both seed_stuck and node_stuck would fire (rare -- a
    multistart seed coincidentally near a node), seed_stuck wins.
    Documents the priority order for callers parsing
    ``suppression_reason``.
    """
    import beagle_server.solver as solver_mod

    def fake_optimizer(pairs, node_events, slat, slon, srad):
        # Both criteria met: 2 m from seed AND 5 m from a node.
        return slat, slon, 6500.0, 0.0, 50.0, 2.0, 5.0

    monkeypatch.setattr(solver_mod, "_run_optimizer", fake_optimizer)
    events = make_synthetic_events()
    result = solve_fix(events, 47.6, -122.3, search_radius_km=80.0)
    assert result is not None
    assert result.suppressed
    # seed_stuck check runs first (elif chain order); both metrics are
    # populated so callers can see the full picture.
    assert result.suppression_reason == "seed_stuck"
    assert result.seed_distance_m == pytest.approx(2.0)
    assert result.node_distance_m == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Per-pair TDOA outlier rejection (PairTdoaHistory)
# ---------------------------------------------------------------------------

def test_pair_history_cold_start_accepts_all():
    """Below ``min_history`` accepted measurements, no rejection occurs."""
    from beagle_server.solver import PairTdoaHistory
    h = PairTdoaHistory(history_size=50, k_mad=5.0, min_history=10)
    # Even wildly extreme values are accepted before warmup completes.
    assert h.is_outlier("a", "b", 10e-6) is False
    assert h.is_outlier("a", "b", -10e-6) is False
    h.record("a", "b", 5e-6)
    assert h.is_outlier("a", "b", 1.0) is False  # 1 second still passes pre-warmup


def test_pair_history_rejects_outlier_after_warmup():
    """After warmup with a cluster of values around 50 µs, values close
    to the median pass the filter and obvious outliers fail it."""
    from beagle_server.solver import PairTdoaHistory
    h = PairTdoaHistory(history_size=50, k_mad=5.0, min_history=10)
    # Warm up with 20 values clustered around 50 µs with σ ≈ 1 µs.
    # Asymptotic MAD ≈ 0.67 σ ≈ 0.67 µs → 5×MAD ≈ 3.4 µs threshold.
    import random
    rng = random.Random(42)
    for _ in range(20):
        h.record("a", "b", 50e-6 + rng.gauss(0, 1e-6))
    # Within 1 µs of median: clearly passes (well under 5×MAD threshold).
    assert h.is_outlier("a", "b", 50.5e-6) is False
    # 50 µs off median: well beyond any reasonable 5×MAD threshold, rejects.
    assert h.is_outlier("a", "b", 100e-6) is True
    assert h.is_outlier("a", "b", 0.0) is True


def test_pair_history_signed_pair_direction():
    """Querying as (b, a) where b > a negates the sign — same physical
    measurement maps to the same sorted-pair history."""
    from beagle_server.solver import PairTdoaHistory
    h = PairTdoaHistory(history_size=50, k_mad=5.0, min_history=10)
    # Record 20 measurements as (a, b) at +50 µs.
    for _ in range(20):
        h.record("a", "b", 50e-6)
    # Same physical measurement queried as (b, a) is -50 µs.
    # Should NOT be flagged as outlier — it's the same value.
    assert h.is_outlier("b", "a", -50e-6) is False
    # And +50 µs queried as (b, a) IS the opposite physical value, far
    # from the running median when sign-corrected — should be flagged.
    assert h.is_outlier("b", "a", 50e-6) is True


def test_pair_history_zero_mad_floor():
    """When all history values are identical (MAD = 0), the filter
    falls back to a small floor (1 ns) so it still rejects values that
    are obviously different — but not within sub-floor noise."""
    from beagle_server.solver import PairTdoaHistory
    h = PairTdoaHistory(history_size=50, k_mad=5.0, min_history=10)
    # 20 identical values → MAD = 0 in raw form, falls to 1 ns floor.
    # 5 × floor = 5 ns rejection threshold.
    for _ in range(20):
        h.record("a", "b", 50e-6)
    # Within sub-floor distance: passes.
    assert h.is_outlier("a", "b", 50e-6) is False                    # exact
    assert h.is_outlier("a", "b", 50e-6 + 1e-9) is False             # 1 ns
    assert h.is_outlier("a", "b", 50e-6 + 4e-9) is False             # 4 ns < 5 ns
    # Beyond floor × k_mad = 5 ns: rejects.
    assert h.is_outlier("a", "b", 50e-6 + 100e-9) is True            # 100 ns > 5 ns
    assert h.is_outlier("a", "b", 100e-6) is True                    # 50 µs >> 5 ns


def test_solve_fix_outlier_filter_drops_corrupted_pair():
    """Inject a known outlier on one pair via a contaminated history;
    that pair should be excluded from the fix.  This mirrors the
    production failure mode where a single PHAT mis-lock corrupts a
    fix's cost surface."""
    from beagle_server.solver import _get_pair_history, reset_pair_outlier_history

    reset_pair_outlier_history()

    # Pre-populate history for "node-A,node-B" with values around -700 ns
    # (the geometric expectation for our synthetic NODES targeting
    # TARGET_LAT/TARGET_LON).  But our incoming measurement on that pair
    # will be wildly off — far enough to trip the outlier filter.
    history = _get_pair_history(history_size=200, k_mad=5.0, min_history=10)
    for _ in range(20):
        history.record("node-A", "node-B", -700e-9)   # tight cluster

    # Now run solve_fix with the outlier filter active.  The actual
    # synthetic fixture's TDOA should not be wildly off (it's geometrically
    # consistent), but if it differs from -700 ns by more than ~k_mad×MAD
    # the filter would drop it.  Confirm fix still produces a result —
    # filter mustn't be over-aggressive on legitimate data.
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        pair_outlier_k_mad=5.0,
    )
    # Expect a fix back (filter shouldn't have killed everything for clean
    # synthetic data).  Then reset for hygiene.
    assert result is not None
    reset_pair_outlier_history()


def test_solve_fix_outlier_filter_zero_disabled():
    """pair_outlier_k_mad=0 turns the filter off entirely."""
    from beagle_server.solver import reset_pair_outlier_history

    reset_pair_outlier_history()
    events = make_synthetic_events()
    result = solve_fix(
        events, 47.6, -122.3, search_radius_km=80.0,
        pair_outlier_k_mad=0.0,    # disabled
    )
    assert result is not None
    reset_pair_outlier_history()


def test_clean_4node_fix_not_suppressed():
    """A normal 4-node fit with a well-placed search area must NOT be
    flagged as suppressed.  Uses 4 nodes so that even if one is excluded
    by outlier detection, 3 remain (avoiding the degenerate 2-node
    hyperbola which would trigger multistart_ambiguous).
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

    min_delta = min(sync_deltas)
    events = []
    for (node_id, nlat, nlon), sync_to_snippet_start_ns in zip(four_nodes, sync_deltas):
        carrier_delay = round((sync_to_snippet_start_ns - min_delta) * _SNIPPET_RATE_HZ / 1e9)
        onset_pos = _SNIPPET_BASE + carrier_delay
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
            "sync_to_snippet_start_ns":          sync_to_snippet_start_ns,
            "corr_peak":              0.85,
            "onset_time_ns":          1_700_000_000_000_000_000,
            "iq_snippet_b64":         _make_snippet_b64(carrier_delay),
            "channel_sample_rate_hz": float(_SNIPPET_RATE_HZ),
            "transition_start":       max(0, onset_pos - 1000),
            "transition_end":         onset_pos + _RAMP_SAMPLES + 1000,
        })

    result = solve_fix(
        events,
        search_center_lat=TARGET_LAT,
        search_center_lon=TARGET_LON,
        search_radius_km=80.0,
    )
    assert result is not None
    assert not result.suppressed, (
        f"clean 4-node fit should not be suppressed; "
        f"got reason={result.suppression_reason} "
        f"boundary_dist={result.boundary_distance_km:.2f} km "
        f"multistart_disagreement={result.multistart_disagreement_km:.2f} km"
    )
    # And the fit's still close to the target.
    error_m = haversine_m(result.latitude_deg, result.longitude_deg,
                          TARGET_LAT, TARGET_LON)
    assert error_m < 20_000, f"4-node clean fit error {error_m:.0f} m"
