# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for scripts/fit_tdoa_calibration.py — the per-node δ
least-squares fitter.

We test the fitting math directly (no end-to-end PHAT pipeline) by
synthesising pair biases that exactly match a chosen per-node δ vector
and then verifying the fitter recovers them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script as a module via its file path (the scripts/ directory
# is not on the package path).
_REPO = Path(__file__).resolve().parent.parent.parent
_FITTER = _REPO / "scripts" / "fit_tdoa_calibration.py"
spec = importlib.util.spec_from_file_location("fit_tdoa_calibration", _FITTER)
assert spec is not None and spec.loader is not None
fit_mod = importlib.util.module_from_spec(spec)
sys.modules["fit_tdoa_calibration"] = fit_mod
spec.loader.exec_module(fit_mod)


def test_fit_recovers_known_offsets_3_nodes():
    """Three nodes with known δ values; fitter recovers them exactly."""
    # δ values in ns
    deltas_ns = {"a": 0.0, "b": 5_000.0, "c": -8_000.0}
    pair_biases_ns = {
        ("a", "b"): deltas_ns["a"] - deltas_ns["b"],
        ("a", "c"): deltas_ns["a"] - deltas_ns["c"],
        ("b", "c"): deltas_ns["b"] - deltas_ns["c"],
    }
    offsets_s, rms_us = fit_mod.fit_node_offsets(pair_biases_ns, reference_node="a")
    assert offsets_s["a"] == 0.0
    assert offsets_s["b"] == pytest.approx(deltas_ns["b"] / 1e9, abs=1e-15)
    assert offsets_s["c"] == pytest.approx(deltas_ns["c"] / 1e9, abs=1e-15)
    assert rms_us == pytest.approx(0.0, abs=1e-9)


def test_fit_handles_overdetermined_consistent_system():
    """Three independent pair biases vs two free parameters (overdetermined)
    must still recover the true δ at zero residual when the inputs are
    self-consistent.  Mirrors the production 3-node case (3 pairs, 2 unknowns).
    """
    delta_b_ns, delta_c_ns = 7_918.322, -74_863.09  # ns
    pair_biases_ns = {
        ("a", "b"): -delta_b_ns,
        ("a", "c"): -delta_c_ns,
        ("b", "c"): delta_b_ns - delta_c_ns,
    }
    offsets_s, rms_us = fit_mod.fit_node_offsets(pair_biases_ns, reference_node="a")
    assert offsets_s["b"] == pytest.approx(delta_b_ns / 1e9, abs=1e-12)
    assert offsets_s["c"] == pytest.approx(delta_c_ns / 1e9, abs=1e-12)
    assert rms_us == pytest.approx(0.0, abs=1e-6)


def test_fit_residual_signals_inconsistent_data():
    """If pair biases are inconsistent with any per-node-δ model
    (e.g. multipath-driven per-pair-per-target effects), the LSQ fit
    distributes residual across pairs and reports nonzero RMS — which is
    the operator's signal that the per-node model isn't sufficient.
    """
    # Biases that violate the triangle equality: a-b + b-c != a-c
    pair_biases_ns = {
        ("a", "b"): -5_000.0,
        ("a", "c"): +5_000.0,
        ("b", "c"): +0.0,        # consistent value would be +10_000
    }
    offsets_s, rms_us = fit_mod.fit_node_offsets(pair_biases_ns, reference_node="a")
    # Sanity: fitter still returns offsets for all nodes
    assert set(offsets_s) == {"a", "b", "c"}
    # 10 µs of inconsistency distributed across 3 pairs → ~3-4 µs RMS
    assert rms_us > 1.0, (
        f"residual RMS should flag inconsistent input; got {rms_us:.3f} µs"
    )


def test_fit_handles_pair_subset():
    """4-node system with only 3 pairs observed — those 3 pairs span all
    4 nodes through the reference, so the fit is still uniquely determined.
    """
    pair_biases_ns = {
        ("ref", "x"): -3_000.0,
        ("ref", "y"): +6_000.0,
        ("ref", "z"): +1_500.0,
    }
    offsets_s, rms_us = fit_mod.fit_node_offsets(pair_biases_ns, reference_node="ref")
    assert offsets_s["ref"] == 0.0
    assert offsets_s["x"] == pytest.approx(3_000.0 / 1e9, abs=1e-12)
    assert offsets_s["y"] == pytest.approx(-6_000.0 / 1e9, abs=1e-12)
    assert offsets_s["z"] == pytest.approx(-1_500.0 / 1e9, abs=1e-12)
    assert rms_us < 1e-6


def test_fit_rejects_unknown_reference():
    """Reference node not present in observed pairs is a hard error."""
    pair_biases_ns = {("a", "b"): 1_000.0}
    with pytest.raises(ValueError, match="reference_node"):
        fit_mod.fit_node_offsets(pair_biases_ns, reference_node="not-in-data")


def test_emit_yaml_format():
    """YAML output is parseable and round-trips back into TdoaCalibrationConfig."""
    import yaml
    from beagle_server.config import ServerFullConfig

    yaml_block = fit_mod.emit_yaml(
        offsets_s={"a": 0.0, "b": 5e-6, "c": -7.5e-6},
        reference_node="a",
        tx_label="TestTx",
        tx_lat=47.5,
        tx_lon=-122.3,
        n_pairs=42,
        residual_rms_us=0.123,
        fit_date="2026-04-25",
        enable=True,
    )
    parsed = yaml.safe_load(yaml_block)
    assert parsed is not None
    assert "tdoa_calibration" in parsed
    cfg = ServerFullConfig.model_validate(parsed)
    assert cfg.tdoa_calibration.enabled is True
    assert cfg.tdoa_calibration.reference_node == "a"
    assert cfg.tdoa_calibration.node_offsets_s == {"a": 0.0, "b": 5e-6, "c": -7.5e-6}
    assert cfg.tdoa_calibration.fit_transmitter_label == "TestTx"
    assert cfg.tdoa_calibration.fit_n_pairs == 42
    assert cfg.tdoa_calibration.fit_residual_rms_us == pytest.approx(0.123)


def test_emit_yaml_disabled():
    """enable=False produces enabled: false in the output."""
    import yaml
    yaml_block = fit_mod.emit_yaml(
        offsets_s={"a": 0.0, "b": 5e-6},
        reference_node="a",
        tx_label="TestTx", tx_lat=47.5, tx_lon=-122.3,
        n_pairs=10, residual_rms_us=0.5, fit_date="2026-04-25",
        enable=False,
    )
    parsed = yaml.safe_load(yaml_block)
    assert parsed["tdoa_calibration"]["enabled"] is False
