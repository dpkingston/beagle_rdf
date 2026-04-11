# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
SDR abstraction layer - base classes.

All hardware-specific code implements SDRReceiver. The rest of the codebase
never imports SoapySDR or librtlsdr directly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SDRConfig:
    """Hardware-independent SDR configuration."""

    center_frequency_hz: float
    """Center tuning frequency in Hz."""

    sample_rate_hz: float
    """ADC sample rate in samples/second."""

    gain_db: float | str
    """Receiver gain in dB, or 'auto' for AGC."""

    device_args: str = ""
    """SoapySDR device string, e.g. 'driver=rtlsdr,serial=00000001'.
    Empty string uses the first available device."""

    buffer_size: int = 131_072
    """Number of IQ samples per read call."""


class SDRReceiver(abc.ABC):
    """
    Abstract base class for an SDR receiver.

    Yields a continuous stream of complex64 IQ sample buffers.
    Each buffer contains `buffer_size` samples (or fewer at end-of-stream).

    Usage::

        with SomeReceiver(config) as rx:
            for iq_buffer in rx.stream():
                process(iq_buffer)
    """

    @property
    @abc.abstractmethod
    def config(self) -> SDRConfig:
        """The configuration this receiver was created with."""

    @abc.abstractmethod
    def open(self) -> None:
        """Initialise hardware and start the sample stream. Idempotent."""

    @abc.abstractmethod
    def close(self) -> None:
        """Stop the sample stream and release hardware resources."""

    @abc.abstractmethod
    def stream(self) -> Generator[tuple[np.ndarray, bool], None, None]:
        """
        Yield ``(iq_buffer, discontinuity)`` tuples.

        Each call to next() blocks until a full buffer is available.
        Raises StopIteration (or GeneratorExit) when close() is called.
        ``discontinuity`` is True on the first buffer after lost samples
        (overflow, timeout, stream restart).  The caller should reset
        pipeline state when this flag is set.
        """

    @property
    def overflow_count(self) -> int:
        """Number of sample overflow events since open(). Override if supported."""
        return 0

    @property
    def backlog_drain_count(self) -> int:
        """Number of stale buffers discarded by backlog drain logic. Override if supported."""
        return 0

    @property
    def discontinuity_count(self) -> int:
        """Number of sample discontinuities (overflows, restarts) since open()."""
        return 0

    def __enter__(self) -> "SDRReceiver":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
