# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
FM stereo pilot (19 kHz) sync event detector with crystal calibration.

Architecture
------------
FMPilotSyncDetector.process(audio, start_sample)
  1. Buffer incoming demodulated FM audio until sync_period_samples accumulate.
  2. Extract pilot via cross-correlation with complex exponential template.
  3. Sub-sample timing: use np.angle(corr) (pilot phase at window start) to
     locate the nearest 19 kHz zero-crossing within the window, replacing the
     coarse window-centre estimate (±3.5 ms) with ~1–2 µs precision.
  4. Feed measured pilot phase into CrystalCalibrator.
  5. Emit SyncEvent with corr_peak and sample_rate_correction.

CrystalCalibrator
-----------------
Tracks the phase of the 19 kHz pilot across consecutive sync windows.
The expected phase advance per window is:
    deltaphi_expected = 2pi * 19000 * sync_period_s
The measured phase advance from the cross-correlation angle gives the
actual elapsed time, and thus the true sample rate:
    correction = deltaphi_measured / deltaphi_expected
A rolling median over the last N windows (default 100 ~= 1 second) provides
a stable, noise-resistant correction factor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, lfilter


PILOT_FREQ_HZ = 19_000.0   # FM stereo pilot -- locked to station frequency standard


@dataclass(frozen=True)
class SyncEvent:
    """One FM pilot sync event."""
    sample_index: int               # In the sync IQ stream
    time_ns: int                    # Rough wall-clock (from EventStamper), event-association only
    corr_peak: float                # Cross-correlation peak magnitude (0-1)
    pilot_phase_rad: float          # Measured pilot phase (for CrystalCalibrator)
    sample_rate_correction: float   # Crystal calibration factor (multiply deltas by this)


# ---------------------------------------------------------------------------
# CrystalCalibrator
# ---------------------------------------------------------------------------

class CrystalCalibrator:
    """
    Estimates SDR crystal frequency error from the FM pilot phase.

    Parameters
    ----------
    sync_period_s : float
        Duration of one sync window in seconds.
    window : int
        Number of recent phase measurements to include in the rolling median.
    """

    def __init__(self, sync_period_s: float, window: int = 100) -> None:
        self._expected_advance = 2.0 * np.pi * PILOT_FREQ_HZ * sync_period_s
        self._window = window
        self._prev_phase: float | None = None
        self._corrections: deque[float] = deque(maxlen=window)

    def update(self, pilot_phase_rad: float) -> float:
        """
        Record one pilot phase measurement and return the current correction.

        Parameters
        ----------
        pilot_phase_rad : float
            Unwrapped pilot phase from cross-correlation.

        Returns
        -------
        float
            Current best-estimate correction factor (1.0 = no correction needed).
        """
        if self._prev_phase is not None:
            measured = pilot_phase_rad - self._prev_phase
            # Wrap to (-pi, +3pi) to handle one expected advance of ~380pi per 10 ms
            # -- actually the advance is large; just take the raw difference
            correction = measured / self._expected_advance
            # Sanity-clamp: RTL-SDR crystals are < 200 ppm off, so correction
            # should be between 0.9999 and 1.0001
            if 0.998 < correction < 1.002:
                self._corrections.append(correction)

        self._prev_phase = pilot_phase_rad

        if self._corrections:
            return float(np.median(np.fromiter(self._corrections, dtype=np.float64, count=len(self._corrections))))
        return 1.0

    def reset(self) -> None:
        self._prev_phase = None
        self._corrections.clear()


# ---------------------------------------------------------------------------
# FMPilotSyncDetector
# ---------------------------------------------------------------------------

class FMPilotSyncDetector:
    """
    FM stereo pilot tone (19 kHz) sync event detector.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the *demodulated FM audio* input (after decimation).
        Typically 256 kHz (sync channel working rate).
    sync_period_ms : float
        Duration of each sync window.  One SyncEvent is emitted per window.
        Default 10 ms -> 100 events/second.
    calibration_window : int
        Number of windows for CrystalCalibrator rolling median.
    pilot_bpf_taps : int
        Length of the narrow bandpass filter isolating the 19 kHz pilot.
    pilot_bpf_bw_hz : float
        Half-bandwidth (one-sided) of the pilot bandpass filter.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        sync_period_ms: float = 10.0,
        calibration_window: int = 100,
        pilot_bpf_taps: int = 127,
        pilot_bpf_bw_hz: float = 100.0,
    ) -> None:
        self._rate = float(sample_rate_hz)
        self._period_samples = int(round(sample_rate_hz * sync_period_ms / 1000.0))
        self._sync_period_s = self._period_samples / sample_rate_hz

        # Pre-build complex exponential template for cross-correlation
        t = np.arange(self._period_samples) / self._rate
        self._template: np.ndarray = np.exp(
            1j * 2.0 * np.pi * PILOT_FREQ_HZ * t
        ).astype(np.complex64)

        # Narrow BPF to isolate pilot before cross-correlation
        nyq = self._rate / 2.0
        lo = max((PILOT_FREQ_HZ - pilot_bpf_bw_hz) / nyq, 1e-6)
        hi = min((PILOT_FREQ_HZ + pilot_bpf_bw_hz) / nyq, 1.0 - 1e-6)
        taps = firwin(pilot_bpf_taps, [lo, hi], pass_zero=False, window="hamming")
        self._bpf_taps: np.ndarray = taps.astype(np.float32)   # was float64
        self._bpf_zi: np.ndarray = np.zeros(len(taps) - 1, dtype=np.float32)  # was float64

        # template is exp(j*...) so |template[k]| = 1 for all k; rms = 1.0 always.
        self._template_rms: float = 1.0

        # Rolling sample buffer for incomplete windows
        self._buf: list[np.ndarray] = []
        self._buf_samples: int = 0
        self._next_start_sample: int | None = None  # set on first process() call

        self._calibrator = CrystalCalibrator(
            sync_period_s=self._sync_period_s,
            window=calibration_window,
        )
        # Expected phase advance per window: 2pi * 19000 Hz * period_s.
        # Because 19000 * period_s = 190 exact cycles, this is also 0 mod 2pi,
        # so np.angle(corr) only shows the crystal-error residual.
        # To give CrystalCalibrator the full phase it expects we add this
        # back in explicitly every window.
        self._sync_period_advance: float = (
            2.0 * np.pi * PILOT_FREQ_HZ * self._sync_period_s
        )
        # Accumulated phase (full advance + crystal residual) for CrystalCalibrator
        self._unwrapped_phase: float = 0.0
        self._last_corr_angle: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    @property
    def sync_period_samples(self) -> int:
        return self._period_samples

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self, audio: np.ndarray, start_sample: int, time_ns: int = 0
    ) -> list[SyncEvent]:
        """
        Detect pilot sync events in a buffer of demodulated FM audio.

        Parameters
        ----------
        audio : np.ndarray, float32
            Demodulated FM audio (instantaneous frequency in Hz) at sample_rate_hz.
        start_sample : int
            Absolute sample index of audio[0] in the continuous stream.
        time_ns : int
            Wall-clock time of start_sample (from EventStamper), for rough
            onset_time_ns in events.  Not used for TDOA precision.

        Returns
        -------
        list[SyncEvent]
            One event per complete sync_period_samples block consumed.
        """
        if len(audio) == 0:
            return []

        events: list[SyncEvent] = []

        # Seed absolute sample counter on first call; detect and handle gaps
        # (e.g. the target-frequency block between consecutive sync blocks in
        # freq_hop mode).  If the incoming start_sample is more than one sync
        # period ahead of where the internal counter expects the next sample to
        # arrive, the buffered remnant from the previous block belongs to a
        # different (non-contiguous) region of the ADC stream.  Mixing it with
        # the new block would produce a corrupted window with a wrong sample
        # index and garbage signal content, so we discard it and resync.
        if self._next_start_sample is None:
            self._next_start_sample = start_sample
        else:
            expected_next = self._next_start_sample + self._buf_samples
            if start_sample > expected_next + self._period_samples:
                # Gap detected -- discard stale buffer and reset filter state
                self._buf.clear()
                self._buf_samples = 0
                self._bpf_zi[:] = 0.0
                self._next_start_sample = start_sample

        # Append to rolling buffer
        self._buf.append(audio)
        self._buf_samples += len(audio)

        # Process complete windows
        while self._buf_samples >= self._period_samples:
            window = self._drain_window()
            event = self._process_window(
                window,
                window_start_sample=self._next_start_sample,
                time_ns=time_ns,
            )
            if event is not None:
                events.append(event)
            self._next_start_sample += self._period_samples

        return events

    def _drain_window(self) -> np.ndarray:
        """Consume exactly period_samples from the front of the buffer."""
        need = self._period_samples
        chunks = []
        taken = 0
        while taken < need:
            chunk = self._buf[0]
            remaining = need - taken
            if len(chunk) <= remaining:
                chunks.append(chunk)
                taken += len(chunk)
                self._buf.pop(0)
                self._buf_samples -= len(chunk)
            else:
                chunks.append(chunk[:remaining])
                self._buf[0] = chunk[remaining:]
                self._buf_samples -= remaining
                taken = need
        return np.concatenate(chunks)

    def _process_window(
        self, audio_window: np.ndarray, window_start_sample: int, time_ns: int
    ) -> SyncEvent | None:
        """Correlate one sync window against the 19 kHz template."""
        n = len(audio_window)
        if n < self._period_samples:
            return None

        # Narrow BPF (real-valued filter applied to audio)
        audio_f32 = audio_window.astype(np.float32)
        filtered, self._bpf_zi = lfilter(self._bpf_taps, 1.0, audio_f32, zi=self._bpf_zi)

        # Cross-correlate with complex template: sum(filtered * conj(template))
        # This gives a complex number whose angle is the pilot phase at this window.
        corr = np.dot(filtered.astype(np.complex64), np.conj(self._template))

        # Normalise by window energy
        sig_rms = float(np.sqrt(np.mean(filtered ** 2)) + 1e-30)
        # Was: template_rms = float(np.sqrt(np.mean(np.abs(self._template) ** 2)))
        template_rms = self._template_rms
        corr_norm = np.abs(corr) / (n * sig_rms * template_rms)
        corr_peak = float(np.clip(corr_norm, 0.0, 1.0))

        # Pilot phase (angle of cross-correlation)
        pilot_phase = float(np.angle(corr))

        # Accumulate phase for CrystalCalibrator.
        # np.angle(corr) is in [-pi, pi] and only carries the crystal-drift
        # residual (since the template is exactly 190 cycles per window ->
        # ideal advance is 0 mod 2pi).  Re-add the full expected advance so
        # CrystalCalibrator receives the total phase it needs to compute the
        # correction ratio correctly.
        if self._last_corr_angle is not None:
            residual = pilot_phase - self._last_corr_angle
            # Wrap residual to [-pi, pi]
            residual = (residual + np.pi) % (2 * np.pi) - np.pi
            self._unwrapped_phase += self._sync_period_advance + residual
        else:
            self._unwrapped_phase = pilot_phase
        self._last_corr_angle = pilot_phase

        correction = self._calibrator.update(self._unwrapped_phase)

        # Sub-sample timing: use the pilot phase at the window start (φ₀ = np.angle(corr))
        # to locate the nearest 19 kHz zero-crossing, replacing the coarse window-centre
        # estimate (±period/2 = ±3.5 ms) with ~1–2 µs precision.
        #
        # The template is exp(j·2π·19000·k/rate) starting at k=0, so np.angle(corr) gives
        # the pilot phase at window_start_sample.  We propagate it to the window centre,
        # then solve for the nearest zero-crossing offset.
        import math as _math
        _phase_per_sample = 2.0 * _math.pi * PILOT_FREQ_HZ / self._rate
        _center = window_start_sample + self._period_samples // 2
        _phase_at_center = pilot_phase + _phase_per_sample * (self._period_samples // 2)
        # Wrap to [-π, π]
        _phase_at_center = (_phase_at_center + _math.pi) % (2.0 * _math.pi) - _math.pi
        # Offset (in samples) from centre to nearest zero-crossing
        _pilot_period = self._rate / PILOT_FREQ_HZ          # ≈13.47 samples at 256 kHz
        _half_period  = _pilot_period / 2.0
        _offset = -_phase_at_center / _phase_per_sample
        _offset = (_offset + _half_period) % _pilot_period - _half_period
        sample_index = _center + round(_offset)

        return SyncEvent(
            sample_index=sample_index,
            time_ns=time_ns,
            corr_peak=corr_peak,
            pilot_phase_rad=self._unwrapped_phase,
            sample_rate_correction=correction,
        )

    def reset(self) -> None:
        """Reset all state."""
        self._buf.clear()
        self._buf_samples = 0
        self._next_start_sample = None
        self._bpf_zi[:] = 0.0
        self._unwrapped_phase = 0.0
        self._last_corr_angle = None
        self._calibrator.reset()
