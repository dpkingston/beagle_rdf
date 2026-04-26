# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for sdr/base.py - SDRConfig and SDRReceiver interface."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.sdr.base import SDRConfig, SDRReceiver


def test_sdr_config_frozen():
    cfg = SDRConfig(center_frequency_hz=155e6, sample_rate_hz=2.048e6, gain_db=40)
    with pytest.raises((AttributeError, TypeError)):
        cfg.gain_db = 30  # type: ignore[misc]


def test_sdr_config_defaults():
    cfg = SDRConfig(center_frequency_hz=100e6, sample_rate_hz=2e6, gain_db="auto")
    assert cfg.device_args == ""
    assert cfg.buffer_size == 131_072


def test_sdr_receiver_is_abstract():
    with pytest.raises(TypeError):
        SDRReceiver()  # type: ignore[abstract]


def test_sdr_receiver_context_manager(sdr_config):
    """MockReceiver works as a context manager and stream() yields complex64."""
    from beagle_node.sdr.mock import MockReceiver

    rx = MockReceiver.synthetic(config=sdr_config, duration_s=0.1)
    with rx as r:
        buf, disc = next(r.stream())
    assert buf.dtype == np.complex64
    assert buf.ndim == 1
    assert len(buf) > 0
    assert disc is False


def test_sdr_base_set_target_frequency_default_raises():
    """The abstract base class default raises NotImplementedError, signalling
    callers to fall back to a process restart for the retune."""
    # Construct a minimal concrete subclass that implements the OTHER
    # abstract methods but inherits the default set_target_frequency.
    class _StubReceiver(SDRReceiver):
        def __init__(self):
            self._cfg = SDRConfig(
                center_frequency_hz=100e6, sample_rate_hz=2e6, gain_db=20,
            )
        @property
        def config(self): return self._cfg
        def open(self): pass
        def close(self): pass
        def stream(self): yield (np.zeros(0, dtype=np.complex64), False)

    rx = _StubReceiver()
    with pytest.raises(NotImplementedError, match="restart"):
        rx.set_target_frequency(155e6)


def test_mock_receiver_set_target_frequency_updates_config(sdr_config):
    """MockReceiver.set_target_frequency updates the recorded center_frequency_hz."""
    from beagle_node.sdr.mock import MockReceiver
    rx = MockReceiver.synthetic(config=sdr_config, duration_s=0.1)
    old_freq = rx.config.center_frequency_hz
    new_freq = old_freq + 5e6
    rx.set_target_frequency(new_freq)
    assert rx.config.center_frequency_hz == new_freq
    # And original freq is no longer there.
    assert rx.config.center_frequency_hz != old_freq
