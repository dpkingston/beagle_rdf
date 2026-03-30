# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
SDR receiver factory.

Reads the `sdr_mode` from node config and instantiates the appropriate
SDRReceiver implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beagle_node.sdr.base import SDRReceiver

if TYPE_CHECKING:
    from beagle_node.config.schema import NodeConfig


def create_receiver(node_config: "NodeConfig", role: str = "target") -> SDRReceiver:
    """
    Instantiate an SDRReceiver based on node configuration.

    Parameters
    ----------
    node_config:
        Loaded and validated node configuration.
    role:
        'target' or 'sync' - relevant for two_sdr mode where each role
        has its own SDR config.

    Returns
    -------
    SDRReceiver
        An unopened receiver. Call open() or use as a context manager.
    """
    mode = node_config.sdr_mode

    if mode == "freq_hop":
        from beagle_node.sdr.freq_hop import FreqHopReceiver
        if node_config.freq_hop is None:
            raise ValueError("sdr_mode='freq_hop' requires a 'freq_hop' config block")
        return FreqHopReceiver.from_config(node_config)

    if mode in ("two_sdr", "single_sdr"):
        from beagle_node.sdr.soapy import SoapyReceiver
        sdr_cfg = node_config.sync_sdr if role == "sync" else node_config.target_sdr
        if sdr_cfg is None:
            raise ValueError(
                f"sdr_mode='{mode}' requires '{role}_sdr' config block"
            )
        return SoapyReceiver(sdr_cfg)

    if mode == "rspduo":
        from beagle_node.sdr.rspduo import RSPduoReceiver
        if node_config.rspduo is None:
            raise ValueError("sdr_mode='rspduo' requires an 'rspduo' config block")
        return RSPduoReceiver.from_config(node_config)

    if mode in ("mock_synthetic", "mock_file"):
        # Used in tests; callers typically construct MockReceiver directly.
        raise ValueError(
            f"sdr_mode='{mode}' is for testing only; construct MockReceiver directly."
        )

    raise ValueError(f"Unknown sdr_mode: {mode!r}")
