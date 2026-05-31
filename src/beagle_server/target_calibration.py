# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Live per-pair TDOA bias auto-calibration against a known-position target.

When a known-position transmitter (the *calibration target*) is the only
source on a given channel, every paired plateau measurement on that channel
yields a per-pair bias:

    bias(a, b) = compute_tdoa_s(a, b) - geometric_expected_tdoa(a, b)

where ``geometric_expected_tdoa`` is the TDOA implied purely by the known
target geometry, ``(dist(tgt, a) - dist(tgt, b)) / c``.  Accumulating a
robust rolling estimate of that bias per node-pair gives the
``pair_offsets_s`` the solver subtracts — self-maintaining, with no manual
offline fit-and-paste loop (the offline equivalent is
``scripts/fit_tdoa_calibration.py``).

Design
------
* **Closed-loop, stateless in the offset.**  The caller observes the *raw*
  (pre-calibration) TDOA, reconstructed as ``calibrated + applied_offset``.
  The published offset is just ``median(raw_bias)`` over the window, so at
  convergence ``calibrated == expected`` and ``offset == true_bias`` — a
  stable fixed point with no integrator wind-up.
* **Median, not mean.**  In production ~40 % of plateau snippets are
  tone-only (no usable voice timing) and produce confident-but-wrong PHAT
  lags; the median over a window rejects those outliers where a mean would
  be dragged by them.
* **min_samples gate.**  A pair's offset is published only once it has
  accumulated ``min_samples`` observations, so a noisy single sample never
  biases the solver.
* **Canonical orientation.**  Biases are always recorded in sorted-pair
  (a < b) orientation, matching the ``pair_offsets_s`` key convention that
  ``compute_tdoa_s`` consumes (``"<node_a>,<node_b>"``, a < b).

Off by default: the server only feeds observations and publishes offsets
when ``tdoa_calibration.auto_calibrate`` is enabled in the config.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any

from beagle_server.tdoa import _C_M_S, haversine_m


def _canonical(node_a: str, node_b: str) -> tuple[str, str, float]:
    """Return (lo, hi, sign) so the pair is in ascending order.

    ``sign`` is +1 when (node_a, node_b) is already ascending, else -1;
    multiply a directional value measured as a→b by ``sign`` to express it
    in the canonical lo→hi orientation.
    """
    if node_a <= node_b:
        return node_a, node_b, 1.0
    return node_b, node_a, -1.0


class TargetCalibrator:
    """Rolling robust per-pair TDOA bias estimator for a known target.

    Parameters
    ----------
    target_lat, target_lon :
        WGS-84 position of the known calibration transmitter.
    target_channel_hz :
        Channel the target transmits on.  Only events whose
        ``channel_hz`` is within ``channel_tol_hz`` are used for
        calibration (so other-channel traffic can't pollute the fit).
    channel_tol_hz :
        Half-width of the channel match window (Hz).
    window :
        Number of most-recent bias observations retained per pair.
    min_samples :
        Minimum observations before a pair's offset is published.
    event_type :
        Only this event type is calibrated (``"plateau"`` — per-event-type
        biases differ on real hardware; plateau is the fine-fix path).
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        target_channel_hz: float,
        channel_tol_hz: float = 1000.0,
        window: int = 200,
        min_samples: int = 20,
        event_type: str = "plateau",
    ) -> None:
        self._tgt_lat = target_lat
        self._tgt_lon = target_lon
        self._tgt_hz = target_channel_hz
        self._chan_tol = channel_tol_hz
        self._window = window
        self._min_samples = min_samples
        self._event_type = event_type
        # canonical (lo, hi) -> deque of raw-bias observations (ns)
        self._biases: dict[tuple[str, str], deque[float]] = {}
        self._total_observed = 0
        self._total_skipped = 0

    # ------------------------------------------------------------------
    def _on_target_channel(self, event: dict[str, Any]) -> bool:
        ch = event.get("channel_hz")
        if ch is None:
            return False
        return abs(float(ch) - self._tgt_hz) <= self._chan_tol

    def expected_tdoa_ns(self, lo_event: dict[str, Any], hi_event: dict[str, Any]) -> float:
        """Geometric TDOA (lo - hi) in ns for the known target position."""
        d_lo = haversine_m(self._tgt_lat, self._tgt_lon,
                           lo_event["node_lat"], lo_event["node_lon"])
        d_hi = haversine_m(self._tgt_lat, self._tgt_lon,
                           hi_event["node_lat"], hi_event["node_lon"])
        return (d_lo - d_hi) / _C_M_S * 1e9

    # ------------------------------------------------------------------
    def observe(
        self,
        node_a: str,
        node_b: str,
        raw_tdoa_s: float,
        event_a: dict[str, Any],
        event_b: dict[str, Any],
    ) -> None:
        """Record one paired observation.

        ``raw_tdoa_s`` is the PRE-calibration TDOA of (node_a, node_b) in
        seconds (the caller reconstructs it as ``calibrated + applied_offset``
        so this stays a stable fixed point).  Both events must be on the
        target channel and of the calibrated event type, else the
        observation is skipped.
        """
        if event_a.get("event_type") != self._event_type or \
           event_b.get("event_type") != self._event_type:
            self._total_skipped += 1
            return
        if not (self._on_target_channel(event_a) and self._on_target_channel(event_b)):
            self._total_skipped += 1
            return

        lo, hi, sign = _canonical(node_a, node_b)
        lo_event, hi_event = (event_a, event_b) if sign > 0 else (event_b, event_a)
        # raw_tdoa_s is in (node_a, node_b) orientation; express lo->hi.
        raw_canonical_ns = sign * raw_tdoa_s * 1e9
        bias_ns = raw_canonical_ns - self.expected_tdoa_ns(lo_event, hi_event)

        dq = self._biases.get((lo, hi))
        if dq is None:
            dq = deque(maxlen=self._window)
            self._biases[(lo, hi)] = dq
        dq.append(bias_ns)
        self._total_observed += 1

    # ------------------------------------------------------------------
    def pair_offsets_s(self) -> dict[str, float]:
        """Published per-pair offsets (seconds), keyed ``"<lo>,<hi>"``.

        Only pairs with at least ``min_samples`` observations are included;
        each value is the median of that pair's retained bias window.
        """
        out: dict[str, float] = {}
        for (lo, hi), dq in self._biases.items():
            if len(dq) >= self._min_samples:
                out[f"{lo},{hi}"] = statistics.median(dq) / 1e9
        return out

    def health_snapshot(self) -> dict[str, Any]:
        """Per-pair calibration state for /health (counts, median, MAD)."""
        pairs: dict[str, Any] = {}
        for (lo, hi), dq in sorted(self._biases.items()):
            vals = list(dq)
            med = statistics.median(vals) if vals else 0.0
            mad = (statistics.median([abs(v - med) for v in vals])
                   if len(vals) > 1 else 0.0)
            pairs[f"{lo},{hi}"] = {
                "n": len(vals),
                "median_us": round(med / 1e3, 2),
                "mad_us": round(mad / 1e3, 2),
                "published": len(vals) >= self._min_samples,
            }
        return {
            "target_channel_mhz": round(self._tgt_hz / 1e6, 4),
            "window": self._window,
            "min_samples": self._min_samples,
            "total_observed": self._total_observed,
            "total_skipped": self._total_skipped,
            "pairs": pairs,
        }

    def reset(self) -> None:
        self._biases.clear()
        self._total_observed = 0
        self._total_skipped = 0
