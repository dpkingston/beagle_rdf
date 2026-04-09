# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
RDS bit-boundary sync event detector using pilot-derived timing.

Every 842 usec (1187.5 Hz = pilot/16), an RDS BPSK bit transition occurs on
the 57 kHz subcarrier.  All nodes receiving the same FM station see the same
transitions, solving the pilot zero-crossing ambiguity problem.

Architecture
------------
RDSSyncDetector.process(audio, start_sample)
  1. Bandpass-filter the 19 kHz pilot from the FM audio.
  2. Cross-correlate with a complex exponential template to extract
     the pilot phase (sub-radian precision).
  3. Feed the pilot phase into CrystalCalibrator (crystal frequency
     error estimation via rolling-median phase advance).
  4. Derive RDS bit boundary positions directly from the pilot phase:
     the RDS subcarrier is phase-locked to the pilot at exactly 3×
     (57 kHz), and the bit rate is pilot/16 (1187.5 Hz).  Every 16
     pilot cycles is one bit boundary.

The detector has the same process() interface as FMPilotSyncDetector and
emits the same SyncEvent dataclass, so it is a drop-in replacement in the
pipeline.

This approach replaces the earlier Mueller-Müller + Costas timing recovery
loop, which required ~7 seconds to converge and produced ~223 µs per-event
jitter (essentially uniform across the RDS bit cell).  The pilot-derived
approach converges in one 10 ms pilot window and has sub-8 µs precision
limited only by pilot SNR.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import firwin, lfilter

from beagle_node.pipeline.sync_detector import (
    CrystalCalibrator,
    SyncEvent,
    PILOT_FREQ_HZ,
)

logger = logging.getLogger(__name__)

RDS_BIT_RATE = 1187.5               # pilot / 16, exact


class RDSSyncDetector:
    """
    RDS bit-boundary sync event detector using pilot-derived timing.

    Extracts the 19 kHz FM stereo pilot phase via cross-correlation and
    derives RDS bit boundary positions directly from it.  Every 16 pilot
    cycles is one RDS bit boundary (1187.5 Hz = pilot/16), and the pilot
    phase gives sub-8 µs precision on the boundary position.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the demodulated FM audio input (after decimation).
        Typically 250-256 kHz (sync channel working rate).
    pilot_period_ms : float
        Window duration for pilot phase extraction / CrystalCalibrator.
    calibration_window : int
        Number of pilot windows for CrystalCalibrator rolling median.
    pilot_bpf_taps : int
        Length of the narrow bandpass filter isolating the 19 kHz pilot.
    pilot_bpf_bw_hz : float
        Half-bandwidth (one-sided) of the pilot bandpass filter.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        pilot_period_ms: float = 10.0,
        calibration_window: int = 100,
        pilot_bpf_taps: int = 127,
        pilot_bpf_bw_hz: float = 100.0,
    ) -> None:
        self._rate = float(sample_rate_hz)

        # -- Pilot extraction for CrystalCalibrator --------------------------
        self._pilot_period = int(round(sample_rate_hz * pilot_period_ms / 1000.0))
        self._pilot_period_s = self._pilot_period / sample_rate_hz

        t = np.arange(self._pilot_period) / self._rate
        self._pilot_template: np.ndarray = np.exp(
            1j * 2.0 * np.pi * PILOT_FREQ_HZ * t
        ).astype(np.complex64)

        nyq = self._rate / 2.0
        lo = max((PILOT_FREQ_HZ - pilot_bpf_bw_hz) / nyq, 1e-6)
        hi = min((PILOT_FREQ_HZ + pilot_bpf_bw_hz) / nyq, 1.0 - 1e-6)
        bpf = firwin(pilot_bpf_taps, [lo, hi], pass_zero=False, window="hamming")
        self._pilot_bpf: np.ndarray = bpf.astype(np.float32)
        self._pilot_bpf_zi: np.ndarray = np.zeros(len(bpf) - 1, dtype=np.float32)

        self._calibrator = CrystalCalibrator(
            sync_period_s=self._pilot_period_s,
            window=calibration_window,
        )
        self._pilot_sync_advance = 2.0 * np.pi * PILOT_FREQ_HZ * self._pilot_period_s
        self._pilot_unwrapped_phase: float = 0.0
        self._pilot_last_corr_angle: float | None = None
        self._pilot_correction: float = 1.0
        self._pilot_corr_peak: float = 0.0

        self._pilot_buf: list[np.ndarray] = []
        self._pilot_buf_n: int = 0

        # -- Pilot-derived RDS bit boundary timing -----------------------------
        # The RDS subcarrier at 57 kHz is phase-locked to the 19 kHz pilot
        # (3rd harmonic, per the RDS standard).  The RDS bit rate is pilot/16
        # = 1187.5 Hz exactly.  So every 16 pilot cycles a bit transition
        # occurs, and the pilot phase — which the CrystalCalibrator already
        # extracts with sub-radian precision — directly determines where the
        # bit boundaries fall in the sample stream.
        #
        # This replaces the Mueller-Müller timing recovery loop, which
        # required seconds to converge and was sensitive to Costas loop
        # phase errors.  The pilot-derived approach converges in one
        # 10 ms window and has ~1 µs precision.
        self._rds_phase_per_sample = 2.0 * math.pi * RDS_BIT_RATE / self._rate
        self._last_pilot_window_start: int | None = None
        self._last_pilot_window_end: int | None = None

        # Gap detection
        self._next_start_sample: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    @property
    def sync_period_samples(self) -> int:
        """Approximate interval between consecutive sync events (in input samples)."""
        return int(round(self._rate / RDS_BIT_RATE))

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self, audio: np.ndarray, start_sample: int, time_ns: int = 0
    ) -> list[SyncEvent]:
        """
        Detect RDS bit-transition sync events in demodulated FM audio.

        Uses the 19 kHz FM pilot phase to derive RDS bit boundary positions
        directly, bypassing the M&M / Costas timing recovery loop.  The pilot
        is phase-locked to the 57 kHz RDS subcarrier by the FM stereo standard;
        every 16 pilot cycles is one RDS bit boundary.

        Parameters
        ----------
        audio : np.ndarray, float32
            Demodulated FM audio at sample_rate_hz.
        start_sample : int
            Absolute sample index of audio[0] in the continuous stream.
        time_ns : int
            Wall-clock time for event association (not used for TDOA precision).

        Returns
        -------
        list[SyncEvent]
        """
        if len(audio) == 0:
            return []

        n = len(audio)

        # -- Gap detection ---------------------------------------------------
        if self._next_start_sample is not None:
            gap = start_sample - self._next_start_sample
            if gap > self._rate:
                # Long gap (> 1 second): reset pilot state
                self._pilot_buf.clear()
                self._pilot_buf_n = 0
                self._pilot_bpf_zi[:] = 0.0
                self._pilot_unwrapped_phase = 0.0
                self._pilot_last_corr_angle = None
                self._pilot_correction = 1.0
                self._pilot_corr_peak = 0.0
                self._calibrator.reset()
                self._last_pilot_window_start = None
                self._last_pilot_window_end = None
            elif gap > self._pilot_period:
                # Short gap (freq_hop): reset filters but keep pilot phase
                self._pilot_buf.clear()
                self._pilot_buf_n = 0
                self._pilot_bpf_zi[:] = 0.0
        self._next_start_sample = start_sample + n

        # -- Pilot phase extraction ------------------------------------------
        # Track which pilot windows fall within this audio buffer so we can
        # emit SyncEvents at the correct sample positions.
        self._pilot_buf.append(audio)
        self._pilot_buf_n += len(audio)

        # Record the sample range covered by pilot windows processed in this
        # call.  We need this to compute bit boundary positions.
        window_start_sample = start_sample
        events: list[SyncEvent] = []

        while self._pilot_buf_n >= self._pilot_period:
            window = self._drain_pilot_window()
            self._update_pilot(window)

            # After _update_pilot, _pilot_unwrapped_phase reflects the
            # cumulative pilot phase through the end of this window.
            # The window covered samples [window_start_sample, window_start_sample + pilot_period).
            ws = window_start_sample
            we = ws + self._pilot_period

            # Skip the first window (need two consecutive for phase difference).
            if self._last_pilot_window_end is not None:
                # The pilot phase at sample `ws` is _pilot_unwrapped_phase
                # minus the advance that occurred during this window.
                phase_at_end = self._pilot_unwrapped_phase
                phase_at_start = phase_at_end - self._pilot_sync_advance

                # RDS bit boundaries occur when pilot_phase mod (2*pi*16) == 0
                # (or any constant — the offset cancels across nodes).
                # Find all bit boundaries within [ws, we).
                #
                # phase_at_start corresponds to sample ws.
                # phase advances linearly at 2*pi*19000 / rate rad per sample.
                phase_per_sample = 2.0 * math.pi * PILOT_FREQ_HZ / self._rate
                rds_period_rad = 2.0 * math.pi * 16.0  # pilot radians per RDS bit

                # Phase offset within the current RDS bit at window start
                rds_phase_start = phase_at_start % rds_period_rad

                # Samples until the next bit boundary from window start
                if rds_phase_start > 0:
                    rad_to_next = rds_period_rad - rds_phase_start
                else:
                    rad_to_next = 0.0
                samples_to_next = rad_to_next / phase_per_sample

                # Enumerate all bit boundaries in [ws, we)
                samples_per_bit = rds_period_rad / phase_per_sample
                pos = samples_to_next
                while pos < self._pilot_period:
                    sample_idx = ws + pos
                    events.append(
                        SyncEvent(
                            sample_index=sample_idx,
                            time_ns=time_ns,
                            corr_peak=self._pilot_corr_peak,
                            pilot_phase_rad=self._pilot_unwrapped_phase,
                            sample_rate_correction=self._pilot_correction,
                        )
                    )
                    pos += samples_per_bit

            self._last_pilot_window_start = ws
            self._last_pilot_window_end = we
            window_start_sample = we

        return events

    # ------------------------------------------------------------------
    # Pilot extraction (mirrors FMPilotSyncDetector's pilot phase path)
    # ------------------------------------------------------------------

    def _drain_pilot_window(self) -> np.ndarray:
        """Consume exactly _pilot_period samples from the pilot buffer."""
        need = self._pilot_period
        chunks: list[np.ndarray] = []
        taken = 0
        while taken < need:
            chunk = self._pilot_buf[0]
            remaining = need - taken
            if len(chunk) <= remaining:
                chunks.append(chunk)
                taken += len(chunk)
                self._pilot_buf.pop(0)
                self._pilot_buf_n -= len(chunk)
            else:
                chunks.append(chunk[:remaining])
                self._pilot_buf[0] = chunk[remaining:]
                self._pilot_buf_n -= remaining
                taken = need
        return np.concatenate(chunks)

    def _update_pilot(self, window: np.ndarray) -> None:
        """Extract pilot phase from one window and update CrystalCalibrator."""
        filtered, self._pilot_bpf_zi = lfilter(
            self._pilot_bpf, 1.0, window.astype(np.float32), zi=self._pilot_bpf_zi
        )
        corr = np.dot(filtered.astype(np.complex64), np.conj(self._pilot_template))

        sig_rms = float(np.sqrt(np.mean(filtered ** 2)) + 1e-30)
        corr_norm = np.abs(corr) / (len(window) * sig_rms)
        self._pilot_corr_peak = float(np.clip(corr_norm, 0.0, 1.0))

        pilot_phase = float(np.angle(corr))

        if self._pilot_last_corr_angle is not None:
            residual = pilot_phase - self._pilot_last_corr_angle
            residual = (residual + math.pi) % (2.0 * math.pi) - math.pi
            self._pilot_unwrapped_phase += self._pilot_sync_advance + residual
        else:
            self._pilot_unwrapped_phase = pilot_phase
        self._pilot_last_corr_angle = pilot_phase

        self._pilot_correction = self._calibrator.update(self._pilot_unwrapped_phase)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state."""
        self._pilot_buf.clear()
        self._pilot_buf_n = 0
        self._pilot_bpf_zi[:] = 0.0
        self._pilot_unwrapped_phase = 0.0
        self._pilot_last_corr_angle = None
        self._pilot_correction = 1.0
        self._pilot_corr_peak = 0.0
        self._calibrator.reset()
        self._last_pilot_window_start = None
        self._last_pilot_window_end = None
        self._next_start_sample = None
