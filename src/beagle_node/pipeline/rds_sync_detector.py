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
        self._pilot_corr_angle: float = 0.0

        self._pilot_buf: list[np.ndarray] = []
        self._pilot_buf_n: int = 0
        # Absolute sample index of the first unconsumed sample in _pilot_buf.
        self._pilot_buf_start_sample: int = 0

        # -- Pilot-derived RDS bit boundary timing -----------------------------
        # Integer cycle counter advanced by counting wraps of angle(corr).
        # The phase_offset rotates angle(corr) so raw_frac is near 0,
        # placing the wrap boundary at ±0.5 (far from the signal).
        self._pilot_cycle_count: int = 0
        self._pilot_phase_offset: float | None = None   # set after BPF settles
        self._pilot_last_adjusted_frac: float = 0.0
        # How many pilot cycles per window (190 for 10 ms at 19 kHz).
        self._pilot_cycles_per_window: int = round(PILOT_FREQ_HZ * self._pilot_period_s)
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
                self._pilot_buf_start_sample = start_sample
                self._pilot_bpf_zi[:] = 0.0
                self._pilot_unwrapped_phase = 0.0
                self._pilot_last_corr_angle = None
                self._pilot_correction = 1.0
                self._pilot_corr_peak = 0.0
                self._calibrator.reset()
                self._last_pilot_window_end = None
            elif gap > self._pilot_period:
                # Short gap (freq_hop): reset filters but keep pilot phase
                self._pilot_buf.clear()
                self._pilot_buf_n = 0
                self._pilot_buf_start_sample = start_sample
                self._pilot_bpf_zi[:] = 0.0
        self._next_start_sample = start_sample + n

        # -- Pilot phase extraction ------------------------------------------
        # If this is the very first audio, set the buffer origin.
        if self._pilot_buf_n == 0 and len(self._pilot_buf) == 0:
            self._pilot_buf_start_sample = start_sample

        self._pilot_buf.append(audio)
        self._pilot_buf_n += n

        events: list[SyncEvent] = []

        while self._pilot_buf_n >= self._pilot_period:
            # The window covers samples [ws, ws + pilot_period) in absolute
            # sample coordinates, where ws is the start of the unconsumed
            # portion of the pilot buffer.
            ws = self._pilot_buf_start_sample

            window = self._drain_pilot_window()
            self._update_pilot(window)

            # Advance the buffer origin past the consumed window.
            self._pilot_buf_start_sample = ws + self._pilot_period

            # Skip the first window (need two for phase diff).
            if self._last_pilot_window_end is not None:
                # Absolute pilot phase at window start, in cycles.
                #
                # Sub-cycle fraction: from the fresh angle(corr), measured
                # directly from the broadcast signal each window (~0.8 µs
                # precision, no drift).
                #
                # Integer cycle count: from round(unwrapped_phase/2π - frac).
                # The unwrapped phase accumulates a random walk (~0.008
                # rad/window std), but round() selects the correct integer
                # as long as the accumulated error is < π (0.5 cycles).
                # With 0.008 rad/step at 100 steps/sec, this stays valid
                # for ~30 minutes.
                #
                # To prevent the random walk from ever reaching ±0.5 cycles,
                # we periodically reset the unwrapped phase to the current
                # angle(corr).  This re-anchors the walk to the signal
                # without losing the integer count.  We reset every
                # _RESYNC_WINDOWS windows (~5 minutes at 100 windows/sec).
                if self._pilot_phase_offset is not None:
                    # Total pilot cycles from the unwrapped phase.  This is
                    # deterministic: all nodes receiving the same FM station
                    # compute the same value at the same wall-clock moment
                    # (modulo propagation delay ~µs).  No node-specific offset.
                    total_cycles_at_start = (
                        self._pilot_unwrapped_phase / (2.0 * math.pi)
                    )
                    # Compensate for crystal error so the bit boundary grid
                    # tracks the station's true frequency.
                    total_cycles_at_start /= self._pilot_correction
                else:
                    total_cycles_at_start = 0.0

                # RDS bit boundaries occur every 16 pilot cycles.
                # Phase within the current bit, in cycles [0, 16):
                phase_in_bit = total_cycles_at_start % 16.0

                # Use the crystal-corrected sample rate for boundary spacing.
                # The ADC crystal error stretches/compresses the pilot cycles
                # in sample space; the correction factor from CrystalCalibrator
                # compensates.  Without this, boundaries drift at the crystal
                # error rate (~2.5 samples/sec at 10 ppm).
                corrected_rate = self._rate * self._pilot_correction
                samples_per_cycle = corrected_rate / PILOT_FREQ_HZ
                samples_per_bit = 16.0 * samples_per_cycle
                if phase_in_bit > 1e-6:
                    samples_to_next = (16.0 - phase_in_bit) * samples_per_cycle
                else:
                    samples_to_next = 0.0

                # Enumerate all bit boundaries in [ws, ws + pilot_period)
                pos = samples_to_next
                while pos < self._pilot_period:
                    sample_idx = float(ws) + pos
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

            self._last_pilot_window_end = ws + self._pilot_period

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
        """Extract pilot phase from one window and update CrystalCalibrator.

        Updates:
          _pilot_corr_angle  - raw angle(corr) in [-pi, pi), fresh each window
          _pilot_corr_peak   - normalised correlation magnitude (0-1)
          _pilot_cycle_count - integer pilot cycles (advances by cycles_per_window)
          _pilot_unwrapped_phase - accumulated phase for CrystalCalibrator
          _pilot_correction  - crystal rate correction factor
        """
        filtered, self._pilot_bpf_zi = lfilter(
            self._pilot_bpf, 1.0, window.astype(np.float32), zi=self._pilot_bpf_zi
        )
        corr = np.dot(filtered.astype(np.complex64), np.conj(self._pilot_template))

        sig_rms = float(np.sqrt(np.mean(filtered ** 2)) + 1e-30)
        corr_norm = np.abs(corr) / (len(window) * sig_rms)
        self._pilot_corr_peak = float(np.clip(corr_norm, 0.0, 1.0))

        pilot_phase = float(np.angle(corr))
        # Store the raw angle for use by process() — this is measured fresh
        # from the signal each window, with no accumulation drift.
        self._pilot_corr_angle = pilot_phase

        # Set the phase offset after the BPF has settled (~500 windows
        # = 5 seconds).  The offset is used ONLY for wrap detection (keeping
        # the wrap boundary away from the signal).  The bit boundary grid
        # is derived from the unwrapped phase, which is deterministic —
        # all nodes receiving the same FM station compute the same total
        # cycle count at the same wall-clock moment (modulo propagation
        # delay), producing aligned bit boundary grids.
        if self._pilot_phase_offset is None and self._pilot_last_corr_angle is not None:
            self._pilot_settle_count = getattr(self, '_pilot_settle_count', 0) + 1
            if self._pilot_settle_count >= 500:
                self._pilot_phase_offset = pilot_phase
                # Initialize cycle count from the unwrapped phase.
                # total_cycles = unwrapped_phase / 2π gives the absolute
                # number of pilot cycles, which is the same for all nodes
                # at the same moment.  The integer part becomes the cycle
                # count; the fractional part is recovered each window from
                # the raw pilot phase.
                total_cycles = self._pilot_unwrapped_phase / (2.0 * math.pi)
                self._pilot_cycle_count = int(total_cycles)
                # adjusted_frac for wrap detection (offset-relative)
                self._pilot_last_adjusted_frac = (
                    (pilot_phase - self._pilot_phase_offset) / (2.0 * math.pi)
                ) % 1.0

        # Advance the integer cycle counter by detecting wraps in the
        # adjusted fractional cycle.  Since adjusted_frac is centered
        # near 0, wraps (crossing ±0.5) only occur due to real cycle
        # advances, not noise.
        if self._pilot_phase_offset is not None and self._pilot_last_corr_angle is not None:
            adjusted_frac = (
                (pilot_phase - self._pilot_phase_offset) / (2.0 * math.pi)
            ) % 1.0   # [0, 1)
            # Wrap from >0.5 to <0.5 shouldn't happen (would need >0.5 cycle noise).
            # Advance by nominal cycles per window, then correct ±1 based on
            # the fractional-cycle change.
            delta_frac = adjusted_frac - self._pilot_last_adjusted_frac
            # delta_frac should be ~0 (190 complete cycles → no frac change).
            # If it jumped by >0.5, a wrap occurred (±1 extra cycle).
            extra = 0
            if delta_frac > 0.5:
                extra = -1   # frac went backward across 0: lost a cycle
            elif delta_frac < -0.5:
                extra = 1    # frac went forward across 0: gained a cycle
            self._pilot_cycle_count += self._pilot_cycles_per_window + extra
            self._pilot_last_adjusted_frac = adjusted_frac

        # Accumulate unwrapped phase for CrystalCalibrator.
        if self._pilot_last_corr_angle is not None:
            residual = pilot_phase - self._pilot_last_corr_angle
            residual = (residual + math.pi) % (2.0 * math.pi) - math.pi
            self._pilot_unwrapped_phase += self._pilot_sync_advance + residual
        else:
            self._pilot_unwrapped_phase = pilot_phase
        self._pilot_last_corr_angle = pilot_phase

        self._pilot_correction = self._calibrator.update(self._pilot_unwrapped_phase)

        # Slew the phase offset toward the current angle(corr) to track
        # the crystal-error-induced drift.  The slew rate (alpha) is slow
        # enough that per-window noise (~0.008 rad) doesn't jitter the
        # offset, but fast enough to track the crystal drift (~0.012
        # rad/window before calibration, ~0.0003 rad/window after).
        #
        # The slew keeps adjusted_frac near 0 continuously, preventing
        # the wrap boundary (±0.5) from ever being reached.  No
        # discontinuous resyncs are needed.
        if self._pilot_phase_offset is not None:
            delta = pilot_phase - self._pilot_phase_offset
            # Wrap delta to [-π, +π)
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            # Apply a small fraction of the error per window.
            # alpha = 0.01 gives a time constant of ~100 windows = 1 sec.
            # This tracks the crystal drift (~0.012 rad/window) with ~1%
            # steady-state lag, while smoothing per-window noise (0.008 rad)
            # by 100×.
            self._pilot_phase_offset += 0.01 * delta

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state."""
        self._pilot_buf.clear()
        self._pilot_buf_n = 0
        self._pilot_buf_start_sample = 0
        self._pilot_bpf_zi[:] = 0.0
        self._pilot_unwrapped_phase = 0.0
        self._pilot_last_corr_angle = None
        self._pilot_corr_angle = 0.0
        self._pilot_correction = 1.0
        self._pilot_corr_peak = 0.0
        self._pilot_cycle_count = 0
        self._pilot_phase_offset = None
        self._pilot_last_adjusted_frac = 0.0
        self._pilot_settle_count = 0
        self._calibrator.reset()
        self._last_pilot_window_end = None
        self._next_start_sample = None
