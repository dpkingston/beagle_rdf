# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Hyperbolic TDOA fix solver.

Given a set of TDOA measurements between node pairs, finds the transmitter
location that minimises the sum of squared TDOA residuals using
scipy.optimize.minimize (L-BFGS-B).

The objective is:
    cost = sum_ij ( measured_tdoa_ij - predicted_tdoa_ij(lat, lon) )^2

where predicted_tdoa_ij = ( dist(P, node_i) - dist(P, node_j) ) / c.

With 2 nodes (1 pair) the problem is under-determined - the solver will find
a point on the hyperbola rather than a unique location, with a large residual
signalling the degeneracy.  3+ nodes are needed for a reliable fix.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import minimize

from beagle_server.tdoa import compute_tdoa_s, haversine_m

_C_M_S = 299_792_458.0  # m/s


@dataclass
class FixResult:
    latitude_deg: float
    longitude_deg: float
    residual_ns: float
    """RMS TDOA residual in nanoseconds.  Lower is better; < 1000 ns is good."""
    node_count: int
    nodes: list[str]
    """node_ids that contributed to this fix."""
    excluded_nodes: list[str]
    """node_ids excluded as outliers during this fix (empty if none)."""
    onset_time_ns: int
    """Representative onset_time_ns (median of contributing events)."""
    channel_hz: float
    event_type: str


def _predicted_tdoa_s(
    lat: float, lon: float,
    node_i_lat: float, node_i_lon: float,
    node_j_lat: float, node_j_lon: float,
) -> float:
    d_i = haversine_m(lat, lon, node_i_lat, node_i_lon)
    d_j = haversine_m(lat, lon, node_j_lat, node_j_lon)
    return (d_i - d_j) / _C_M_S


def _run_optimizer(
    pairs: list[tuple[int, int, float]],
    node_events: list[dict[str, Any]],
    search_center_lat: float,
    search_center_lon: float,
    search_radius_km: float,
) -> tuple[float, float, float]:
    """
    Run the L-BFGS-B multi-start optimizer for the given TDOA pairs.

    Returns (fix_lat, fix_lon, rms_residual_ns).
    """
    def cost(xy: np.ndarray) -> float:
        lat, lon = float(xy[0]), float(xy[1])
        total = 0.0
        for i, j, measured in pairs:
            pred = _predicted_tdoa_s(
                lat, lon,
                node_events[i]["node_lat"], node_events[i]["node_lon"],
                node_events[j]["node_lat"], node_events[j]["node_lon"],
            )
            total += (measured - pred) ** 2
        return total

    deg_lat = search_radius_km / 111.0
    deg_lon = search_radius_km / (111.0 * math.cos(math.radians(search_center_lat)))
    bounds = [
        (search_center_lat - deg_lat, search_center_lat + deg_lat),
        (search_center_lon - deg_lon, search_center_lon + deg_lon),
    ]
    _offset = deg_lat * 0.3
    start_points = [
        np.array([search_center_lat, search_center_lon]),
        np.array([search_center_lat + _offset, search_center_lon + _offset]),
        np.array([search_center_lat - _offset, search_center_lon + _offset]),
        np.array([search_center_lat + _offset, search_center_lon - _offset]),
        np.array([search_center_lat - _offset, search_center_lon - _offset]),
    ]
    best_result = None
    for x0 in start_points:
        r = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10})
        if best_result is None or r.fun < best_result.fun:
            best_result = r

    fix_lat, fix_lon = float(best_result.x[0]), float(best_result.x[1])  # type: ignore[union-attr]
    residuals_ns = [
        (measured - _predicted_tdoa_s(
            fix_lat, fix_lon,
            node_events[i]["node_lat"], node_events[i]["node_lon"],
            node_events[j]["node_lat"], node_events[j]["node_lon"],
        )) * 1e9
        for i, j, measured in pairs
    ]
    rms_ns = float(np.sqrt(np.mean(np.array(residuals_ns) ** 2)))
    return fix_lat, fix_lon, rms_ns


def _pair_residuals_ns(
    node_events: list[dict[str, Any]],
    pairs: list[tuple[int, int, float]],
    fix_lat: float,
    fix_lon: float,
) -> list[float]:
    """Return absolute pair residuals (ns) at the given fix position."""
    result = []
    for i, j, measured in pairs:
        pred = _predicted_tdoa_s(
            fix_lat, fix_lon,
            node_events[i]["node_lat"], node_events[i]["node_lon"],
            node_events[j]["node_lat"], node_events[j]["node_lon"],
        )
        result.append(abs((measured - pred) * 1e9))
    return result


def _identify_outlier_node(
    node_events: list[dict[str, Any]],
    pairs: list[tuple[int, int, float]],
    fix_lat: float,
    fix_lon: float,
    improvement_factor: float = 3.0,
) -> str | None:
    """
    Identify the single node whose removal most reduces the RMS TDOA residual.

    For each node, compute the RMS of all pairs that do NOT involve it.  The
    node whose exclusion gives the largest fractional improvement is the
    outlier.  Requires at least 3 nodes and at least 1 surviving pair after
    exclusion.

    The criterion is: rms_excluding_suspect < overall_rms / improvement_factor.
    This correctly handles the structural dilution effect (bad pairs are shared
    with good nodes) and scales with the number of nodes.
    """
    if len(node_events) < 3 or len(pairs) < 2:
        return None

    node_ids = [e["node_id"] for e in node_events]
    pair_resids = _pair_residuals_ns(node_events, pairs, fix_lat, fix_lon)
    overall_rms = float(np.sqrt(np.mean(np.array(pair_resids) ** 2)))
    if overall_rms == 0.0:
        return None

    best_nid: str | None = None
    best_rms = overall_rms

    for suspect_idx, nid in enumerate(node_ids):
        excl_resids = [
            r for (i, j, _), r in zip(pairs, pair_resids)
            if i != suspect_idx and j != suspect_idx
        ]
        if not excl_resids:
            continue
        excl_rms = float(np.sqrt(np.mean(np.array(excl_resids) ** 2)))
        if excl_rms < best_rms:
            best_rms = excl_rms
            best_nid = nid

    if best_nid is not None and best_rms < overall_rms / improvement_factor:
        return best_nid
    return None


def solve_fix(
    events: list[dict[str, Any]],
    search_center_lat: float,
    search_center_lon: float,
    search_radius_km: float = 100.0,
    min_xcorr_snr: float = 1.3,
    max_xcorr_baseline_km: float = 50.0,
    savgol_window_us: float = 360.0,
) -> FixResult | None:
    """
    Compute a transmitter fix from a list of events from the same transmission.

    Parameters
    ----------
    events :
        List of event dicts from at least 2 distinct nodes.
        Each must have: node_id, node_lat, node_lon,
        sync_tx_lat, sync_tx_lon, onset_time_ns, channel_hz, event_type,
        sync_delta_ns.
    search_center_lat, search_center_lon :
        Initial search point and centre of the bounding box.
    search_radius_km :
        Solver search radius.

    Returns
    -------
    FixResult or None if no valid TDOA pairs can be formed.
    """
    # Deduplicate: one event per node (latest received, highest corr_peak)
    best: dict[str, dict[str, Any]] = {}
    for ev in events:
        nid = ev["node_id"]
        if nid not in best or ev.get("corr_peak", 0) > best[nid].get("corr_peak", 0):
            best[nid] = ev

    node_events = list(best.values())
    if len(node_events) < 2:
        return None

    node_ids = [e["node_id"] for e in node_events]

    # Check for degenerate geometry: all nodes effectively co-located.
    # With no meaningful baseline, the solver returns the search center
    # which is useless.  Log and skip.
    max_baseline_m = 0.0
    for i in range(len(node_events)):
        for j in range(i + 1, len(node_events)):
            d = haversine_m(
                node_events[i]["node_lat"], node_events[i]["node_lon"],
                node_events[j]["node_lat"], node_events[j]["node_lon"],
            )
            if d > max_baseline_m:
                max_baseline_m = d
    if max_baseline_m < 100.0:
        logger.info(
            "Degenerate geometry: max baseline %.0f m < 100 m "
            "(all nodes co-located); fix skipped for %s",
            max_baseline_m, node_ids,
        )
        return None

    # Build TDOA pairs from sync_delta subtraction + path-delay correction.
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(node_events)):
        for j in range(i + 1, len(node_events)):
            tdoa = compute_tdoa_s(
                node_events[i], node_events[j],
                min_xcorr_snr=min_xcorr_snr,
                max_xcorr_baseline_km=max_xcorr_baseline_km,
                savgol_window_us=savgol_window_us,
            )
            if tdoa is not None:
                pairs.append((i, j, tdoa))

    if not pairs:
        logger.warning(
            "No valid TDOA pairs for group (sync_delta_ns missing); fix skipped."
        )
        return None

    fix_lat, fix_lon, rms_ns = _run_optimizer(
        pairs, node_events, search_center_lat, search_center_lon, search_radius_km,
    )

    # Outlier detection: identify any node whose pairs have anomalously high
    # residuals compared to the rest of the network.
    excluded_nodes: list[str] = []
    outlier_id = _identify_outlier_node(node_events, pairs, fix_lat, fix_lon)

    if outlier_id:
        pair_resids = _pair_residuals_ns(node_events, pairs, fix_lat, fix_lon)
        node_ids_list = [e["node_id"] for e in node_events]
        resid_summary = "  ".join(
            f"{node_ids_list[i]}<->{node_ids_list[j]}={r:.0f} ns"
            for (i, j, _), r in sorted(zip(pairs, pair_resids), key=lambda x: -x[1])
        )
        logger.warning(
            "Outlier node detected: %s  pair residuals: [%s]",
            outlier_id, resid_summary,
        )
        # Re-run without the outlier if enough nodes remain for a 2-D fix.
        outlier_idx = node_ids.index(outlier_id)
        clean_pairs = [(i, j, t) for i, j, t in pairs if i != outlier_idx and j != outlier_idx]
        clean_node_idxs = {i for i, j, _ in clean_pairs} | {j for i, j, _ in clean_pairs}
        if len(clean_node_idxs) >= 2 and clean_pairs:
            fix_lat, fix_lon, rms_ns = _run_optimizer(
                clean_pairs, node_events,
                search_center_lat, search_center_lon, search_radius_km,
            )
            excluded_nodes = [outlier_id]
            logger.warning(
                "Re-solved excluding %s: residual %.0f ns  lat=%.5f  lon=%.5f",
                outlier_id, rms_ns, fix_lat, fix_lon,
            )
        else:
            logger.warning(
                "Cannot exclude %s - too few remaining nodes for a 2-D fix",
                outlier_id,
            )

    onset_times = sorted(e["onset_time_ns"] for e in node_events)
    median_onset = onset_times[len(onset_times) // 2]

    return FixResult(
        latitude_deg=fix_lat,
        longitude_deg=fix_lon,
        residual_ns=rms_ns,
        node_count=len(node_events) - len(excluded_nodes),
        nodes=[nid for nid in node_ids if nid not in excluded_nodes],
        excluded_nodes=excluded_nodes,
        onset_time_ns=median_onset,
        channel_hz=node_events[0]["channel_hz"],
        event_type=node_events[0]["event_type"],
    )
