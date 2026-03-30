# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Shared pytest fixtures for Beagle tests."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.sdr.base import SDRConfig
from beagle_node.sdr.mock import MockReceiver
from beagle_node.timing.clock import MockClock


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 2_048_000.0
CENTER_FREQ = 155_100_000.0
FIXED_EPOCH_NS = 1_700_000_000_000_000_000  # 2023-11-14 ~22:13 UTC


# ---------------------------------------------------------------------------
# SDR fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sdr_config() -> SDRConfig:
    return SDRConfig(
        center_frequency_hz=CENTER_FREQ,
        sample_rate_hz=SAMPLE_RATE,
        gain_db=40,
        device_args="",
        buffer_size=16_384,
    )


@pytest.fixture
def sdr_config_fm() -> SDRConfig:
    """Config for the FM sync SDR (tuned to KISW 99.9 MHz)."""
    return SDRConfig(
        center_frequency_hz=99_900_000.0,
        sample_rate_hz=SAMPLE_RATE,
        gain_db=30,
        device_args="",
        buffer_size=16_384,
    )


@pytest.fixture
def mock_clock() -> MockClock:
    return MockClock(start_ns=FIXED_EPOCH_NS, step_ns=0)


@pytest.fixture
def carrier_receiver(sdr_config: SDRConfig) -> MockReceiver:
    """MockReceiver with a 1-second carrier starting at t=0.5s."""
    return MockReceiver.synthetic(
        config=sdr_config,
        duration_s=3.0,
        carrier_intervals=[(0.5, 1.5)],
        pilot_present=False,
        snr_db=20.0,
    )


@pytest.fixture
def noise_receiver(sdr_config: SDRConfig) -> MockReceiver:
    """MockReceiver with noise only (no carrier, no pilot)."""
    return MockReceiver.synthetic(
        config=sdr_config,
        duration_s=3.0,
        carrier_intervals=[],
        pilot_present=False,
        snr_db=0.0,
    )


@pytest.fixture
def pilot_receiver(sdr_config_fm: SDRConfig) -> MockReceiver:
    """MockReceiver with FM pilot tone only."""
    return MockReceiver.synthetic(
        config=sdr_config_fm,
        duration_s=3.0,
        pilot_present=True,
        snr_db=20.0,
    )
