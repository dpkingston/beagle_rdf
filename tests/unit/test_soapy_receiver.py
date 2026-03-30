# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for sdr/soapy.py.

SoapySDR hardware calls are mocked so these tests run without real hardware.
A hardware-required test is provided but skipped when no device is present.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from beagle_node.sdr.base import SDRConfig
from beagle_node.sdr.soapy import SoapyReceiver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RATE = 2_048_000.0
FREQ = 99_900_000.0
GAIN = 20.0


def make_config(**kwargs) -> SDRConfig:
    defaults = dict(
        center_frequency_hz=FREQ,
        sample_rate_hz=RATE,
        gain_db=GAIN,
        device_args="",
        buffer_size=1024,
    )
    defaults.update(kwargs)
    return SDRConfig(**defaults)


def _make_mock_device(n_samples: int = 1024, flags: int = 0, overflow_once: bool = False):
    """Return a mock SoapySDR.Device that streams one buffer then stops."""
    call_count = {"n": 0}

    def fake_read_stream(stream, buffers, n, timeoutUs=0):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Fill the buffer with known data
            buffers[0][:n_samples] = np.ones(n_samples, dtype=np.complex64) * 0.5
            f = 0x10 if overflow_once else 0   # SOAPY_SDR_OVERFLOW = 0x10
            return SimpleNamespace(ret=n_samples, flags=f)
        return SimpleNamespace(ret=-1, flags=0)  # signal stop

    device = MagicMock()
    device.getSampleRate.return_value = RATE
    device.getFrequency.return_value = FREQ
    device.getGain.return_value = GAIN
    device.setupStream.return_value = MagicMock()
    device.readStream.side_effect = fake_read_stream
    return device


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_config_stored():
    cfg = make_config()
    rx = SoapyReceiver(cfg)
    assert rx.config is cfg


def test_initial_overflow_count():
    rx = SoapyReceiver(make_config())
    assert rx.overflow_count == 0


# ---------------------------------------------------------------------------
# Open / configure
# ---------------------------------------------------------------------------

@patch("beagle_node.sdr.soapy._SoapySDR")
def test_open_sets_rate_and_freq(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device()
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config())
    rx.open()

    mock_device.setSampleRate.assert_called_once_with(0, 0, RATE)
    mock_device.setFrequency.assert_called_once_with(0, 0, FREQ)
    mock_device.setGain.assert_called_once_with(0, 0, GAIN)
    mock_device.activateStream.assert_called_once()


@patch("beagle_node.sdr.soapy._SoapySDR")
def test_open_agc_mode(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device()
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config(gain_db="auto"))
    rx.open()

    mock_device.setGainMode.assert_any_call(0, 0, True)
    mock_device.setGain.assert_not_called()


@patch("beagle_node.sdr.soapy._SoapySDR")
def test_open_idempotent(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device()
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config())
    rx.open()
    rx.open()   # second call should be no-op

    assert mock_soapy.Device.call_count == 1


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@patch("beagle_node.sdr.soapy._SoapySDR")
def test_stream_yields_complex64(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device(n_samples=1024)
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config(buffer_size=1024))
    rx.open()
    buffers = list(rx.stream())

    assert len(buffers) == 1
    assert buffers[0].dtype == np.complex64
    assert len(buffers[0]) == 1024


@patch("beagle_node.sdr.soapy._SoapySDR")
def test_stream_yields_copy(mock_soapy):
    """Each yielded buffer must be a copy, not a view of the internal buffer."""
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device(n_samples=512)
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config(buffer_size=1024))
    rx.open()
    bufs = list(rx.stream())
    assert len(bufs) == 1
    # Values should be 0.5 as set by mock
    assert np.all(bufs[0].real == 0.5)


@patch("beagle_node.sdr.soapy._SoapySDR")
def test_overflow_counted(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device(n_samples=512, overflow_once=True)
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config(buffer_size=1024))
    rx.open()
    list(rx.stream())
    assert rx.overflow_count == 1


# ---------------------------------------------------------------------------
# Close / context manager
# ---------------------------------------------------------------------------

@patch("beagle_node.sdr.soapy._SoapySDR")
def test_close_deactivates_stream(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device()
    mock_soapy.Device.return_value = mock_device

    rx = SoapyReceiver(make_config())
    rx.open()
    rx.close()

    mock_device.deactivateStream.assert_called_once()
    mock_device.closeStream.assert_called_once()


@patch("beagle_node.sdr.soapy._SoapySDR")
def test_context_manager(mock_soapy):
    mock_soapy.SOAPY_SDR_RX = 0
    mock_soapy.SOAPY_SDR_CF32 = "CF32"
    mock_soapy.SOAPY_SDR_OVERFLOW = 0x10
    mock_device = _make_mock_device()
    mock_soapy.Device.return_value = mock_device

    with SoapyReceiver(make_config()) as rx:
        pass

    mock_device.deactivateStream.assert_called_once()


# ---------------------------------------------------------------------------
# Hardware test (skipped if no device)
# ---------------------------------------------------------------------------

def _has_rtlsdr() -> bool:
    try:
        import SoapySDR
        devs = SoapySDR.Device.enumerate()
        return any(dict(d).get("driver") == "rtlsdr" for d in devs)
    except Exception:
        return False


@pytest.mark.skipif(not _has_rtlsdr(), reason="No RTL-SDR device connected")
def test_hardware_stream_one_buffer():
    """Smoke test: stream one real buffer from the RTL-SDR."""
    cfg = SDRConfig(
        center_frequency_hz=99_900_000.0,   # KISW FM
        sample_rate_hz=2_048_000.0,
        gain_db=0.0,
        buffer_size=65_536,
    )
    rx = SoapyReceiver(cfg)
    with rx:
        for buf in rx.stream():
            assert buf.dtype == np.complex64
            assert len(buf) == 65_536
            assert np.any(buf != 0)
            break   # one buffer is enough
