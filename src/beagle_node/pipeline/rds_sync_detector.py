# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
RDS BPSK bit-transition sync event detector.

Replaces the FM pilot zero-crossing sync with an RDS bit boundary sync.
Every 842 usec (1187.5 Hz = pilot/16), a BPSK bit transition occurs on the
57 kHz subcarrier.  All nodes receiving the same FM station see the same
transitions, solving the pilot zero-crossing ambiguity problem.

Architecture
------------
RDSSyncDetector.process(audio, start_sample)
  1. Pilot phase extraction (BPF + correlation) for CrystalCalibrator.
  2. Frequency shift -57 kHz -> complex baseband RDS signal.
  3. Lowpass filter (7.5 kHz cutoff, 101 taps).
  4. Decimate by 10.
  5. Mueller-Muller timing recovery (streaming, cubic interpolation).
  6. Costas loop carrier phase/frequency correction.
  7. Map symbol positions to input sample coordinates -> emit SyncEvent.

The detector has the same process() interface as FMPilotSyncDetector and
emits the same SyncEvent dataclass, so it is a drop-in replacement in the
pipeline.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
from scipy.signal import firwin, lfilter

from beagle_node.pipeline.sync_detector import (
    CrystalCalibrator,
    SyncEvent,
    PILOT_FREQ_HZ,
)

logger = logging.getLogger(__name__)

RDS_SUBCARRIER_HZ = 57_000.0       # 3 * pilot, phase-locked by spec
RDS_BIT_RATE = 1187.5               # pilot / 16, exact


def _cubic_interp(buf: np.ndarray, idx: int, mu: float) -> complex:
    """Catmull-Rom cubic interpolation at buf[idx + mu]."""
    if idx < 1 or idx + 2 >= len(buf):
        # Linear fallback at boundaries
        if 0 <= idx and idx + 1 < len(buf):
            return buf[idx] * (1.0 - mu) + buf[idx + 1] * mu
        return complex(buf[max(0, min(idx, len(buf) - 1))])
    y0, y1, y2, y3 = buf[idx - 1], buf[idx], buf[idx + 1], buf[idx + 2]
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    d = y1
    return d + mu * (c + mu * (b + mu * a))


class RDSSyncDetector:
    """
    RDS bit-transition sync event detector.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the demodulated FM audio input (after decimation).
        Typically 256 kHz (sync channel working rate).
    pilot_period_ms : float
        Window duration for pilot phase extraction / CrystalCalibrator.
    calibration_window : int
        Number of pilot windows for CrystalCalibrator rolling median.
    pilot_bpf_taps : int
        Length of the narrow bandpass filter isolating the 19 kHz pilot.
    pilot_bpf_bw_hz : float
        Half-bandwidth (one-sided) of the pilot bandpass filter.
    rds_lpf_taps : int
        Number of taps for the RDS baseband lowpass filter.
    rds_lpf_cutoff_hz : float
        Cutoff frequency for the RDS baseband LPF.
    rds_decimation : int
        Decimation factor after RDS LPF.
    mm_gain : float
        Mueller-Muller timing loop gain.
    costas_alpha : float
        Costas loop proportional gain.
    costas_beta : float
        Costas loop integral gain.
    warmup_symbols : int
        Number of symbols to skip before emitting events (M&M convergence).
    """

    def __init__(
        self,
        sample_rate_hz: float,
        pilot_period_ms: float = 10.0,
        calibration_window: int = 100,
        pilot_bpf_taps: int = 127,
        pilot_bpf_bw_hz: float = 100.0,
        rds_lpf_taps: int = 101,
        rds_lpf_cutoff_hz: float = 7500.0,
        rds_decimation: int = 10,
        mm_gain: float = 0.01,
        costas_alpha: float = 8.0,
        costas_beta: float = 0.02,
        warmup_symbols: int = 50,
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

        # -- RDS processing chain --------------------------------------------
        self._dec_factor = int(rds_decimation)
        dec_rate = self._rate / self._dec_factor
        self._mm_sps = dec_rate / RDS_BIT_RATE  # samples per symbol after decim
        self._mm_gain = float(mm_gain)
        self._costas_alpha = float(costas_alpha)
        self._costas_beta = float(costas_beta)
        self._warmup = int(warmup_symbols)

        # Oscillator for -57 kHz frequency shift
        self._osc_phase: float = 0.0
        self._osc_dphi: float = 2.0 * np.pi * (-RDS_SUBCARRIER_HZ) / self._rate

        # RDS baseband LPF
        rds_lpf = firwin(rds_lpf_taps, rds_lpf_cutoff_hz, fs=self._rate)
        self._rds_lpf_b: np.ndarray = rds_lpf.astype(np.complex64)
        self._rds_lpf_zi: np.ndarray = np.zeros(
            rds_lpf_taps - 1, dtype=np.complex64
        )

        # Decimation phase counter
        self._dec_phase: int = 0

        # M&M state
        self._mm_buf: np.ndarray = np.empty(0, dtype=np.complex64)
        self._mm_buf_origin: float = 0.0   # input-stream pos of mm_buf[0]
        self._mm_i_in: int = 1              # current index in mm_buf (>=1 for cubic)
        self._mm_mu: float = 0.01           # fractional timing offset
        self._mm_out: list[complex] = [0j, 0j]   # last 2 symbols
        self._mm_rail: list[complex] = [0j, 0j]  # last 2 railed symbols

        # Costas loop state
        self._costas_phase: float = 0.0
        self._costas_freq: float = 0.0

        # Symbol counting (for warmup gate)
        self._total_symbols: int = 0

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

        # -- Gap detection ---------------------------------------------------
        if self._next_start_sample is not None:
            gap = start_sample - self._next_start_sample
            if gap > self._rate:
                # Long gap (> 1 second): full reset including warmup
                self._reset_rds_state()
            elif gap > self._pilot_period:
                # Short gap (freq_hop): reset signal chain but preserve
                # M&M timing, Costas lock, and warmup progress.
                self._osc_phase = (
                    self._osc_phase + self._osc_dphi * gap
                ) % (2.0 * math.pi)
                self._rds_lpf_zi[:] = 0.0
                self._dec_phase = 0
                self._mm_buf = np.empty(0, dtype=np.complex64)
                self._mm_i_in = 1
                self._pilot_buf.clear()
                self._pilot_buf_n = 0
                self._pilot_bpf_zi[:] = 0.0
        self._next_start_sample = start_sample + len(audio)

        # -- Pilot phase extraction (for CrystalCalibrator) ------------------
        self._pilot_buf.append(audio)
        self._pilot_buf_n += len(audio)
        while self._pilot_buf_n >= self._pilot_period:
            window = self._drain_pilot_window()
            self._update_pilot(window)

        # -- RDS chain: freq shift -> LPF -> decimate -> M&M -> Costas ------

        # Normalise audio amplitude before the RDS chain.  The M&M loop gain
        # is tuned for unit-RMS BPSK symbols; without normalisation, the loop
        # corrections scale with the FM demod amplitude (Hz), producing
        # large jumps in the timing estimate and bimodal interval jitter.
        n = len(audio)
        audio_f32 = audio.astype(np.float32)
        audio_rms = float(np.sqrt(np.mean(audio_f32 * audio_f32)) + 1e-30)
        audio_norm = audio_f32 / audio_rms

        # 1. Frequency shift -57 kHz
        phases = self._osc_phase + self._osc_dphi * np.arange(n, dtype=np.float64)
        osc = np.exp(1j * phases).astype(np.complex64)
        shifted = audio_norm.astype(np.complex64) * osc
        self._osc_phase = (self._osc_phase + self._osc_dphi * n) % (2.0 * np.pi)

        # 2. LPF
        filtered, self._rds_lpf_zi = lfilter(
            self._rds_lpf_b, 1.0, shifted, zi=self._rds_lpf_zi
        )

        # 3. Decimate
        first = (self._dec_factor - self._dec_phase) % self._dec_factor
        decimated = np.asarray(
            filtered[first :: self._dec_factor], dtype=np.complex64
        )
        self._dec_phase = (self._dec_phase + n) % self._dec_factor

        if len(decimated) == 0:
            return []

        # Track position: decimated[0] came from audio[first] = input pos
        # start_sample + first.
        dec_origin = start_sample + first
        if len(self._mm_buf) == 0:
            self._mm_buf_origin = float(dec_origin)
        self._mm_buf = np.concatenate([self._mm_buf, decimated])

        # 4-5. M&M timing recovery + Costas loop
        symbols, positions = self._run_mm_costas()

        # 6. Emit SyncEvents (one per symbol after warmup).
        # Emit the float position from the M&M loop (sub-sample precision);
        # rounding to int would discard the timing information that makes RDS
        # sync valuable.
        events: list[SyncEvent] = []
        for _sym, pos in zip(symbols, positions):
            self._total_symbols += 1
            if self._total_symbols > self._warmup:
                events.append(
                    SyncEvent(
                        sample_index=float(pos),
                        time_ns=time_ns,
                        corr_peak=self._pilot_corr_peak,
                        pilot_phase_rad=self._pilot_unwrapped_phase,
                        sample_rate_correction=self._pilot_correction,
                    )
                )

        # Trim consumed portion of M&M buffer
        self._trim_mm_buf()

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
    # M&M timing recovery + Costas loop (streaming)
    # ------------------------------------------------------------------

    def _run_mm_costas(self) -> tuple[list[complex], list[float]]:
        """
        Run Mueller-Muller TED + Costas loop on the M&M buffer.

        Returns (symbols, input_positions) where input_positions are in the
        input audio sample coordinate system.
        """
        symbols: list[complex] = []
        positions: list[float] = []
        buf = self._mm_buf
        sps = self._mm_sps
        gain = self._mm_gain

        # Need indices i_in-1 .. i_in+2 for cubic interpolation
        while self._mm_i_in >= 1 and self._mm_i_in + 2 < len(buf):
            # Cubic interpolation at i_in + mu
            sample = _cubic_interp(buf, self._mm_i_in, self._mm_mu)

            # Rail (hard decision on raw, pre-Costas symbol)
            rail = complex(
                int(np.real(sample) > 0), int(np.imag(sample) > 0)
            )

            # M&M timing error detector
            if self._mm_out[0] != 0j:
                x_val = (rail - self._mm_rail[0]) * np.conj(self._mm_out[1])
                y_val = (sample - self._mm_out[0]) * np.conj(self._mm_rail[1])
                mm_val = float(np.real(y_val - x_val))
            else:
                mm_val = 0.0

            # Record input-stream position of this symbol
            frac_pos = self._mm_i_in + self._mm_mu
            input_pos = self._mm_buf_origin + frac_pos * self._dec_factor

            # Costas loop (phase/freq correction)
            corrected = sample * np.exp(-1j * self._costas_phase)
            error = float(np.real(corrected) * np.imag(corrected))
            self._costas_freq += self._costas_beta * error
            self._costas_phase += self._costas_freq + self._costas_alpha * error
            self._costas_phase %= (2.0 * math.pi)

            symbols.append(corrected)
            positions.append(input_pos)

            # Update M&M history (keep exactly 2: [i_out-2, i_out-1])
            self._mm_out = [self._mm_out[1], sample]
            self._mm_rail = [self._mm_rail[1], rail]

            # Advance
            self._mm_mu += sps + gain * mm_val
            advance = int(math.floor(self._mm_mu))
            self._mm_i_in += advance
            self._mm_mu -= advance

        return symbols, positions

    def _trim_mm_buf(self) -> None:
        """Remove consumed samples from the M&M buffer to bound memory.

        Two invariants must hold across calls:
          1. Keep at least 2 samples before mm_i_in for cubic interp context
          2. Never let mm_buf become empty in steady state -- if it did,
             process() would reset mm_buf_origin to dec_origin on the next
             call, discarding any "outstanding advance" of mm_i_in past the
             buffer end and causing the M&M's logical position to jump
             backward by (overshoot - 2) * dec_factor samples.
        """
        if len(self._mm_buf) == 0:
            return
        # Cap trim so the buffer always retains at least 2 samples.
        max_trim = max(0, len(self._mm_buf) - 2)
        desired_trim = max(0, self._mm_i_in - 2)
        trim = min(desired_trim, max_trim)
        if trim > 0:
            self._mm_buf = self._mm_buf[trim:]
            self._mm_buf_origin += trim * self._dec_factor
            self._mm_i_in -= trim

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_rds_state(self) -> None:
        """Reset RDS chain state (called on gap detection)."""
        self._osc_phase = 0.0
        self._rds_lpf_zi[:] = 0.0
        self._dec_phase = 0
        self._mm_buf = np.empty(0, dtype=np.complex64)
        self._mm_buf_origin = 0.0
        self._mm_i_in = 1
        self._mm_mu = 0.01
        self._mm_out = [0j, 0j]
        self._mm_rail = [0j, 0j]
        self._costas_phase = 0.0
        self._costas_freq = 0.0
        self._total_symbols = 0

    def reset(self) -> None:
        """Reset all state."""
        # Pilot
        self._pilot_buf.clear()
        self._pilot_buf_n = 0
        self._pilot_bpf_zi[:] = 0.0
        self._pilot_unwrapped_phase = 0.0
        self._pilot_last_corr_angle = None
        self._pilot_correction = 1.0
        self._pilot_corr_peak = 0.0
        self._calibrator.reset()
        # RDS chain
        self._reset_rds_state()
        # Gap detection
        self._next_start_sample = None
