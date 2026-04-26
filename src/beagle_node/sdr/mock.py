# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Mock SDR receiver for testing and development without hardware.

Provides two factory methods:
- MockReceiver.synthetic()  - generates configurable synthetic IQ signals
- MockReceiver.from_file()  - replays a .npy file of complex64 samples
"""

from __future__ import annotations

import threading
from typing import Generator

import numpy as np

from beagle_node.sdr.base import SDRConfig, SDRReceiver


class MockReceiver(SDRReceiver):
    """
    In-memory SDR receiver for testing.

    Yields pre-computed or synthetically generated IQ buffers at the
    configured sample rate (in wall time - buffers are delivered as fast
    as possible in tests, or with optional real-time pacing).
    """

    def __init__(
        self,
        config: SDRConfig,
        samples: np.ndarray,
        loop: bool = False,
        realtime: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        config:
            SDR configuration (used by the pipeline for sample rate, etc.)
        samples:
            complex64 array of IQ samples to yield.
        loop:
            If True, repeat the samples indefinitely.
        realtime:
            If True, pace delivery to match the configured sample rate.
            If False (default), deliver buffers as fast as possible (for tests).
        """
        self._config = config
        self._samples = samples.astype(np.complex64)
        self._loop = loop
        self._realtime = realtime
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        config: SDRConfig,
        duration_s: float = 5.0,
        carrier_intervals: list[tuple[float, float]] | None = None,
        pilot_present: bool = True,
        pilot_freq_hz: float = 19_000.0,
        pilot_fm_carrier_hz: float = 0.0,
        carrier_bw_hz: float = 12_500.0,
        snr_db: float = 20.0,
        pps_interval_samples: int | None = None,
        pps_amplitude: float = 0.1,
        loop: bool = False,
        realtime: bool = False,
        rng: np.random.Generator | None = None,
    ) -> "MockReceiver":
        """
        Generate a synthetic IQ signal.

        Parameters
        ----------
        config:
            SDR configuration; `sample_rate_hz` and `center_frequency_hz` are used.
        duration_s:
            Total duration of the synthetic signal in seconds.
        carrier_intervals:
            List of (start_sec, stop_sec) tuples for LMR carrier presence.
            Each interval adds a narrowband FM carrier to the signal.
        pilot_present:
            If True, inject a 19 kHz FM stereo pilot tone.
            The pilot is modulated onto a carrier at `pilot_fm_carrier_hz`
            offset from the SDR center frequency (0.0 = same channel).
        pilot_freq_hz:
            Pilot tone frequency in Hz (default 19 kHz).
        pilot_fm_carrier_hz:
            Offset of the FM station carrier from the SDR center (Hz).
        carrier_bw_hz:
            Bandwidth of each synthetic LMR carrier (Hz).
        snr_db:
            Signal-to-noise ratio of injected signals (dB above noise floor).
        pps_interval_samples:
            If set, inject a 1PPS-like amplitude spike every this many samples.
            Typically sample_rate_hz (i.e., once per second).
        pps_amplitude:
            Amplitude of the 1PPS spike relative to the signal level.
        loop:
            If True, the receiver loops the generated samples indefinitely.
        realtime:
            If True, pace delivery at the configured sample rate.
        rng:
            Optional numpy random generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        fs = config.sample_rate_hz
        n_total = int(duration_s * fs)
        t = np.arange(n_total) / fs

        # White noise floor
        noise_amplitude = 1.0
        signal_amplitude = noise_amplitude * (10.0 ** (snr_db / 20.0))
        iq = (rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)).astype(
            np.complex64
        ) * (noise_amplitude / np.sqrt(2))

        # FM stereo pilot tone (injected as a baseband tone on the sync channel)
        if pilot_present:
            pilot_phase = 2 * np.pi * pilot_freq_hz * t
            # Modulate onto FM carrier offset
            if pilot_fm_carrier_hz != 0.0:
                carrier_phase = 2 * np.pi * pilot_fm_carrier_hz * t
                # FM: pilot modulates the FM carrier's instantaneous frequency
                fm_signal = np.exp(1j * (carrier_phase + 0.1 * np.sin(pilot_phase)))
            else:
                fm_signal = np.exp(1j * pilot_phase)
            iq += (signal_amplitude * fm_signal).astype(np.complex64)

        # LMR carrier intervals
        if carrier_intervals:
            for start_s, stop_s in carrier_intervals:
                start_idx = int(start_s * fs)
                stop_idx = int(stop_s * fs)
                start_idx = max(0, min(start_idx, n_total))
                stop_idx = max(0, min(stop_idx, n_total))
                if start_idx >= stop_idx:
                    continue
                carrier_t = t[start_idx:stop_idx]
                # Narrowband FM carrier with slight frequency modulation
                fm_dev = carrier_bw_hz * 0.4
                audio_tone = np.sin(2 * np.pi * 1000.0 * carrier_t)
                carrier_signal = np.exp(
                    1j * np.cumsum(
                        2 * np.pi * fm_dev * audio_tone / fs
                    )
                )
                iq[start_idx:stop_idx] += (signal_amplitude * carrier_signal).astype(
                    np.complex64
                )

        # GPS 1PPS spikes
        if pps_interval_samples is not None and pps_interval_samples > 0:
            spike_indices = np.arange(0, n_total, pps_interval_samples)
            for idx in spike_indices:
                if idx < n_total:
                    iq[idx] += pps_amplitude * signal_amplitude * (1 + 1j)

        return cls(config, iq.astype(np.complex64), loop=loop, realtime=realtime)

    @classmethod
    def from_file(
        cls,
        path: str,
        config: SDRConfig,
        loop: bool = False,
        realtime: bool = False,
    ) -> "MockReceiver":
        """
        Load IQ samples from a .npy file (complex64).

        The file must contain a 1-D complex64 numpy array as saved by
        ``numpy.save(path, iq_array)``.
        """
        samples = np.load(path)
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)
        return cls(config, samples, loop=loop, realtime=realtime)

    # ------------------------------------------------------------------
    # SDRReceiver interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> SDRConfig:
        return self._config

    def open(self) -> None:
        self._stop_event.clear()

    def close(self) -> None:
        self._stop_event.set()

    def set_target_frequency(self, frequency_hz: float) -> None:
        """Update the recorded target frequency.

        The mock receiver doesn't actually generate frequency-modulated IQ
        samples — it replays whatever was loaded — so this only updates
        the bookkeeping ``config.center_frequency_hz``.  Useful for tests
        that exercise the hot-reload retune path.
        """
        from dataclasses import replace
        self._config = replace(self._config, center_frequency_hz=float(frequency_hz))

    def stream(self) -> Generator[tuple[np.ndarray, bool], None, None]:
        buf_size = self._config.buffer_size
        samples = self._samples

        while True:
            offset = 0
            while offset < len(samples):
                if self._stop_event.is_set():
                    return
                end = min(offset + buf_size, len(samples))
                yield samples[offset:end].copy(), False
                if self._realtime:
                    import time
                    time.sleep((end - offset) / self._config.sample_rate_hz)
                offset = end

            if not self._loop:
                return
