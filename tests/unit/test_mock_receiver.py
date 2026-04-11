# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for sdr/mock.py - MockReceiver."""

from __future__ import annotations

import numpy as np
import pytest

from beagle_node.sdr.mock import MockReceiver


def test_synthetic_yields_correct_total_samples(sdr_config):
    rx = MockReceiver.synthetic(config=sdr_config, duration_s=1.0)
    total = sum(len(buf) for buf, _disc in rx.stream())
    expected = int(1.0 * sdr_config.sample_rate_hz)
    assert total == expected


def test_synthetic_dtype_is_complex64(sdr_config):
    rx = MockReceiver.synthetic(config=sdr_config, duration_s=0.01)
    for buf, _disc in rx.stream():
        assert buf.dtype == np.complex64


def test_synthetic_carrier_raises_power(sdr_config):
    """Carrier interval should produce higher IQ power than noise-only."""
    rx_carrier = MockReceiver.synthetic(
        config=sdr_config,
        duration_s=1.0,
        carrier_intervals=[(0.0, 1.0)],
        snr_db=20,
        rng=np.random.default_rng(0),
    )
    rx_noise = MockReceiver.synthetic(
        config=sdr_config,
        duration_s=1.0,
        carrier_intervals=[],
        snr_db=0,
        rng=np.random.default_rng(0),
    )
    power_carrier = np.mean([np.mean(np.abs(b) ** 2) for b, _ in rx_carrier.stream()])
    power_noise = np.mean([np.mean(np.abs(b) ** 2) for b, _ in rx_noise.stream()])
    assert power_carrier > power_noise * 2


def test_synthetic_pps_spike_visible(sdr_config):
    """1PPS spikes should produce distinct amplitude peaks."""
    pps_interval = int(sdr_config.sample_rate_hz)  # once per second
    rx = MockReceiver.synthetic(
        config=sdr_config,
        duration_s=3.0,
        pps_interval_samples=pps_interval,
        pps_amplitude=10.0,
        carrier_intervals=[],
        pilot_present=False,
        snr_db=0,
    )
    all_samples = np.concatenate([buf for buf, _ in rx.stream()])
    magnitudes = np.abs(all_samples)
    # The maximum should be at a 1PPS spike location
    max_idx = int(np.argmax(magnitudes))
    expected_spike_indices = [pps_interval * k for k in range(3)]
    assert any(abs(max_idx - s) < 5 for s in expected_spike_indices)


def test_from_file_roundtrip(tmp_path, sdr_config):
    """Samples saved and reloaded via from_file match the originals."""
    rng = np.random.default_rng(123)
    original = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(
        np.complex64
    )
    path = str(tmp_path / "test.npy")
    np.save(path, original)

    rx = MockReceiver.from_file(path, sdr_config)
    recovered = np.concatenate([buf for buf, _ in rx.stream()])
    np.testing.assert_array_equal(original, recovered)


def test_loop_mode_yields_more_than_one_pass(sdr_config):
    """In loop=True mode, stream() should yield more than one pass of samples."""
    rx = MockReceiver.synthetic(
        config=sdr_config, duration_s=0.01, loop=True
    )
    count = 0
    for buf, _disc in rx.stream():
        count += len(buf)
        if count > int(0.025 * sdr_config.sample_rate_hz):
            rx.close()
            break
    assert count > int(0.015 * sdr_config.sample_rate_hz)


def test_discontinuity_always_false(sdr_config):
    """MockReceiver never signals discontinuity."""
    rx = MockReceiver.synthetic(config=sdr_config, duration_s=0.01)
    for _buf, disc in rx.stream():
        assert disc is False
