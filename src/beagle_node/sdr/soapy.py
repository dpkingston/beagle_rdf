# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
SoapySDR-based SDRReceiver implementation.

Used for two_sdr and single_sdr operating modes.  Wraps the SoapySDR
Python bindings, which must be installed system-wide (not via pip).

Overflow detection
------------------
SoapySDR signals an overflow via the SOAPY_SDR_OVERFLOW flag bit in
readStream's return flags field.  We count these and expose them via
overflow_count so the health endpoint can report them.

Gain handling
-------------
If gain_db is the string "auto", AGC is enabled.
Otherwise gain_db is a numeric dB value applied to the TUNER gain element.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np

from beagle_node.sdr.base import SDRConfig, SDRReceiver

logger = logging.getLogger(__name__)

# Lazy import - SoapySDR is a system library, not in pyproject.toml deps
try:
    import SoapySDR as _SoapySDR  # noqa: N813
    _SOAPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SOAPY_AVAILABLE = False
    _SoapySDR = None  # type: ignore[assignment]


class SoapyReceiver(SDRReceiver):
    """
    SDRReceiver backed by SoapySDR.

    Parameters
    ----------
    config : SDRConfig
        Hardware-independent SDR configuration.
    """

    def __init__(self, config: SDRConfig) -> None:
        if not _SOAPY_AVAILABLE:
            raise RuntimeError(
                "SoapySDR Python bindings not available.  "
                "Install via: brew install soapysdr soapyrtlsdr"
            )
        self._config = config
        self._device = None
        self._stream = None
        self._overflow_count: int = 0
        self._discontinuity_pending: bool = False
        self._discontinuity_count: int = 0
        self._is_open: bool = False

    # ------------------------------------------------------------------
    # SDRReceiver interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> SDRConfig:
        return self._config

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def discontinuity_count(self) -> int:
        return self._discontinuity_count

    def open(self) -> None:
        if self._is_open:
            return

        args = self._config.device_args or ""
        logger.info("Opening SoapySDR device: %r", args or "(first available)")
        self._device = _SoapySDR.Device(args)

        ch = 0  # channel index (RTL-SDR has one)

        self._device.setSampleRate(_SoapySDR.SOAPY_SDR_RX, ch,
                                   self._config.sample_rate_hz)
        self._device.setFrequency(_SoapySDR.SOAPY_SDR_RX, ch,
                                  self._config.center_frequency_hz)

        if self._config.gain_db == "auto":
            self._device.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, True)
            logger.debug("AGC enabled")
        else:
            self._device.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, False)
            self._device.setGain(_SoapySDR.SOAPY_SDR_RX, ch,
                                 float(self._config.gain_db))
            logger.debug("Gain set to %.1f dB", float(self._config.gain_db))

        actual_rate = self._device.getSampleRate(_SoapySDR.SOAPY_SDR_RX, ch)
        actual_freq = self._device.getFrequency(_SoapySDR.SOAPY_SDR_RX, ch)
        actual_gain = self._device.getGain(_SoapySDR.SOAPY_SDR_RX, ch)
        logger.info(
            "SDR configured: %.3f MHz  %.3f MSps  %.1f dB",
            actual_freq / 1e6, actual_rate / 1e6, actual_gain,
        )

        self._stream = self._device.setupStream(
            _SoapySDR.SOAPY_SDR_RX,
            _SoapySDR.SOAPY_SDR_CF32,
        )
        self._device.activateStream(self._stream)
        self._is_open = True

    def set_target_frequency(self, frequency_hz: float) -> None:
        """Retune to a new center frequency at runtime.

        For single-tuner SDRs used as the *target* receiver (i.e. in
        two_sdr mode where a separate SDR handles the sync channel),
        this directly retunes the only channel.  Flags a sample-stream
        discontinuity so the pipeline resets across the brief retune
        transient.
        """
        if not self._is_open or self._device is None:
            raise RuntimeError(
                "SoapyReceiver.set_target_frequency: device not open"
            )
        old_freq = self._config.center_frequency_hz
        try:
            self._device.setFrequency(_SoapySDR.SOAPY_SDR_RX, 0, float(frequency_hz))
            actual = self._device.getFrequency(_SoapySDR.SOAPY_SDR_RX, 0)
        except Exception as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                f"SoapyReceiver retune {old_freq/1e6:.4f}->"
                f"{frequency_hz/1e6:.4f} MHz failed: {exc}"
            ) from exc
        # Update stored config to reflect the new tuning.
        from dataclasses import replace
        self._config = replace(self._config, center_frequency_hz=float(actual))
        logger.info(
            "SoapyReceiver retuned: %.4f MHz -> %.4f MHz",
            old_freq / 1e6, actual / 1e6,
        )
        self._discontinuity_pending = True

    def close(self) -> None:
        if not self._is_open:
            return
        if self._stream is not None:
            self._device.deactivateStream(self._stream)
            self._device.closeStream(self._stream)
            self._stream = None
        self._device = None
        self._is_open = False
        logger.info("SoapySDR device closed")

    def stream(self) -> Generator[tuple[np.ndarray, bool], None, None]:
        """Yield ``(iq_buffer, discontinuity)`` tuples.

        ``discontinuity`` is ``True`` on the first buffer after an overflow,
        signaling that samples were lost and pipeline state should be reset.
        """
        if not self._is_open:
            self.open()

        buf = np.zeros(self._config.buffer_size, dtype=np.complex64)
        OVERFLOW_FLAG = _SoapySDR.SOAPY_SDR_OVERFLOW

        while True:
            sr = self._device.readStream(
                self._stream, [buf], len(buf), timeoutUs=1_000_000
            )

            if sr.ret < 0:
                # Negative return is an error code
                logger.error("readStream error: %d", sr.ret)
                self._discontinuity_pending = True
                break

            if sr.flags & OVERFLOW_FLAG:
                self._overflow_count += 1
                logger.warning("SDR overflow #%d", self._overflow_count)
                self._discontinuity_pending = True

            if sr.ret > 0:
                _disc = self._discontinuity_pending
                if _disc:
                    self._discontinuity_count += 1
                    self._discontinuity_pending = False
                yield buf[: sr.ret].copy(), _disc
