# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
GPS 1PPS spike detector for the two_sdr operating mode.

In two_sdr mode a 3.3 V GPS 1PPS pulse is injected into both SDR antenna
inputs via a 10 MOhm series resistor.  This produces a brief broadband power
spike (~-82 dBm, well above the noise floor) that appears in both IQ streams
at the same moment, establishing a common sample-domain time reference and
eliminating USB-transfer jitter between the two devices.

Detection algorithm
-------------------
1. Compute the short-term RMS power over overlapping windows.
2. Maintain a rolling baseline (median of recent quiet windows).
3. Declare a PPSAnchor when power exceeds baseline by spike_threshold_db.
4. Sub-sample interpolation (parabolic peak) refines the sample index.
5. Enforce a minimum spacing of ~0.9 seconds between consecutive anchors to
   prevent double-triggering on the same pulse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PPSAnchor:
    """One GPS 1PPS event detected in the IQ stream."""
    sample_index: int       # Sub-sample interpolated position in continuous stream
    power_db: float         # Peak power at the spike (dBFS)
    baseline_db: float      # Estimated noise floor just before the spike


class PPSDetector:
    """
    GPS 1PPS spike detector.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the input IQ stream.
    spike_threshold_db : float
        Required excess above baseline to declare a PPS event.
        Default 10 dB (well above noise, conservative against false triggers).
    window_samples : int
        Number of samples per detection window.
    baseline_window : int
        Number of recent quiet windows used to estimate the noise baseline.
    min_spacing_s : float
        Minimum time between consecutive PPS events (default 0.9 s).
        Prevents double-triggering.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        spike_threshold_db: float = 10.0,
        window_samples: int = 32,
        baseline_window: int = 200,
        min_spacing_s: float = 0.9,
    ) -> None:
        if spike_threshold_db <= 0:
            raise ValueError(f"spike_threshold_db must be > 0, got {spike_threshold_db}")
        if window_samples < 1:
            raise ValueError(f"window_samples must be >= 1, got {window_samples}")

        self._rate = float(sample_rate_hz)
        self._threshold_db = float(spike_threshold_db)
        self._win = int(window_samples)
        self._baseline_n = int(baseline_window)
        self._min_spacing = int(round(min_spacing_s * sample_rate_hz))

        # Rolling baseline (power values of recent quiet windows)
        self._baseline_buf: list[float] = []
        self._last_anchor_sample: int = -self._min_spacing  # allow first event immediately

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self, iq: np.ndarray, start_sample: int
    ) -> list[PPSAnchor]:
        """
        Scan an IQ buffer for 1PPS spikes.

        Parameters
        ----------
        iq : np.ndarray, complex64
            IQ samples at sample_rate_hz.
        start_sample : int
            Cumulative sample index of iq[0].

        Returns
        -------
        list[PPSAnchor]
            Zero or more PPS events (typically at most one per call).
        """
        events: list[PPSAnchor] = []
        n = len(iq)
        if n == 0:
            return events

        power_lin = np.abs(iq) ** 2
        n_windows = n // self._win

        for i in range(n_windows):
            sl = slice(i * self._win, (i + 1) * self._win)
            avg_power = float(np.mean(power_lin[sl]))
            power_db = 10.0 * np.log10(avg_power + 1e-30)
            window_center = start_sample + i * self._win + self._win // 2

            # Compute current baseline
            if self._baseline_buf:
                baseline_db = float(np.median(self._baseline_buf))
            else:
                baseline_db = power_db  # bootstrap

            excess_db = power_db - baseline_db

            if excess_db >= self._threshold_db:
                # Enforce minimum spacing
                if window_center - self._last_anchor_sample >= self._min_spacing:
                    # Sub-sample refinement using parabolic interpolation on
                    # three consecutive window powers (previous, current, next)
                    refined = self._refine(
                        power_lin, i, start_sample
                    )
                    events.append(PPSAnchor(
                        sample_index=refined,
                        power_db=power_db,
                        baseline_db=baseline_db,
                    ))
                    self._last_anchor_sample = window_center
                # Don't add spike windows to baseline
            else:
                # Quiet window - update baseline
                self._baseline_buf.append(power_db)
                if len(self._baseline_buf) > self._baseline_n:
                    self._baseline_buf.pop(0)

        return events

    def _refine(
        self, power_lin: np.ndarray, peak_window: int, start_sample: int
    ) -> int:
        """
        Parabolic interpolation over sample-level power to sub-sample refine
        the peak position.

        Falls back to window-centre if neighbours are unavailable.
        """
        win = self._win
        i0 = peak_window * win
        i1 = i0 + win

        # Find sample-level peak within this window
        window_power = power_lin[i0:i1]
        if len(window_power) == 0:
            return start_sample + i0 + win // 2

        local_peak = int(np.argmax(window_power))
        peak_abs = i0 + local_peak

        # Parabolic fit using immediate neighbours
        if 0 < peak_abs < len(power_lin) - 1:
            y0 = float(power_lin[peak_abs - 1])
            y1 = float(power_lin[peak_abs])
            y2 = float(power_lin[peak_abs + 1])
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            if abs(denom) > 1e-30:
                offset = (y0 - y2) / denom   # fractional sample offset
                # Round to nearest integer sample (sub-sample accuracy limited
                # by integer indexing here; full sub-sample requires float index)
                refined = int(round(peak_abs + offset))
                return start_sample + max(0, min(refined, len(power_lin) - 1))

        return start_sample + peak_abs

    def reset(self) -> None:
        """Reset detector state."""
        self._baseline_buf.clear()
        self._last_anchor_sample = -self._min_spacing
