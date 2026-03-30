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
        buf = next(r.stream())
    assert buf.dtype == np.complex64
    assert buf.ndim == 1
    assert len(buf) > 0
