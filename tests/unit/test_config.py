# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for config/schema.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from beagle_node.config.schema import NodeConfig, load_config

_EXAMPLE_CONFIG = str(Path(__file__).parent.parent.parent / "config" / "node.example.yaml")


def test_load_example_config():
    """The example config file must parse without errors."""
    config = load_config(_EXAMPLE_CONFIG)
    assert config.node_id == "seattle-north-01"
    assert config.sdr_mode == "freq_hop"
    assert len(config.target_channels) == 2


def test_invalid_node_id():
    with pytest.raises(Exception, match="node_id"):
        NodeConfig.model_validate({
            "node_id": "Seattle Node 01",  # spaces and uppercase not allowed
            "sdr_mode": "freq_hop",
            "location": {"latitude_deg": 47.0, "longitude_deg": -122.0},
            "freq_hop": {
                "target_frequency_hz": 155e6,
                "sync_frequency_hz": 99.9e6,
            },
            "sync_signal": {
                "primary_station": {
                    "station_id": "KISW_99.9",
                    "frequency_hz": 99.9e6,
                    "latitude_deg": 47.6,
                    "longitude_deg": -122.3,
                }
            },
            "target_channels": [{"frequency_hz": 155e6}],
        })


def test_missing_freq_hop_config():
    with pytest.raises(Exception):
        NodeConfig.model_validate({
            "node_id": "test-node",
            "sdr_mode": "freq_hop",  # no freq_hop block
            "location": {"latitude_deg": 47.0, "longitude_deg": -122.0},
            "sync_signal": {
                "primary_station": {
                    "station_id": "KISW_99.9",
                    "frequency_hz": 99.9e6,
                    "latitude_deg": 47.6,
                    "longitude_deg": -122.3,
                }
            },
            "target_channels": [{"frequency_hz": 155e6}],
        })


def test_auth_token_env_var(monkeypatch):
    monkeypatch.setenv("TDOA_AUTH_TOKEN", "test-secret-token")
    config = load_config(_EXAMPLE_CONFIG)
    assert config.reporter.auth_token == "test-secret-token"


def test_empty_target_channels():
    with pytest.raises(Exception, match="target_channel"):
        NodeConfig.model_validate({
            "node_id": "test-node",
            "sdr_mode": "freq_hop",
            "location": {"latitude_deg": 47.0, "longitude_deg": -122.0},
            "freq_hop": {
                "target_frequency_hz": 155e6,
                "sync_frequency_hz": 99.9e6,
            },
            "sync_signal": {
                "primary_station": {
                    "station_id": "KISW_99.9",
                    "frequency_hz": 99.9e6,
                    "latitude_deg": 47.6,
                    "longitude_deg": -122.3,
                }
            },
            "target_channels": [],
        })
