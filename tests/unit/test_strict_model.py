# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Tests for ``WarnOnUnknownFieldsBase`` — the runtime warn-on-unknown-field
base class used by every config / event pydantic model.

Verifies:
  * A subclass still constructs successfully when given an unknown field
    (legacy/fleet-safety behavior preserved).
  * The unknown field is **dropped** (not stored on the instance).
  * A WARNING is logged the first time each (class, field) pair is seen.
  * The same (class, field) pair only warns once per process (rate-limiting).
"""

from __future__ import annotations

import logging

import pytest

from beagle_node.utils.strict_model import (
    WarnOnUnknownFieldsBase,
    _reset_warned_keys,
)


class _Sample(WarnOnUnknownFieldsBase):
    name: str
    count: int = 0


@pytest.fixture(autouse=True)
def reset_warn_cache():
    """Each test starts with a clean warned-keys cache."""
    _reset_warned_keys()
    yield
    _reset_warned_keys()


def test_known_fields_accepted_normally():
    obj = _Sample(name="foo", count=5)
    assert obj.name == "foo"
    assert obj.count == 5


def test_unknown_field_dropped_not_stored(caplog):
    with caplog.at_level(logging.WARNING):
        obj = _Sample(name="foo", count=5, mystery_field=42)
    # The unknown field is not an attribute on the model
    assert not hasattr(obj, "mystery_field")
    # The known fields still work
    assert obj.name == "foo"
    # And a warning was emitted
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mystery_field" in r.message and "_Sample" in r.message
               for r in warn_records), \
        f"expected warning for mystery_field on _Sample; got: {[r.message for r in warn_records]}"


def test_warning_rate_limited_per_class_field(caplog):
    """The same (class, field) only logs once even if the model is
    instantiated many times with the same unknown field."""
    with caplog.at_level(logging.WARNING):
        for i in range(10):
            _Sample(name=f"n{i}", mystery_field=i)
    warns_for_mystery = [r for r in caplog.records
                         if r.levelno == logging.WARNING and "mystery_field" in r.message]
    assert len(warns_for_mystery) == 1, \
        f"expected exactly 1 warning across 10 instances; got {len(warns_for_mystery)}"


def test_different_fields_warn_separately(caplog):
    """A second unknown field on the same class gets its own warning."""
    with caplog.at_level(logging.WARNING):
        _Sample(name="x", field_a=1)
        _Sample(name="x", field_b=2)
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("field_a" in m for m in msgs)
    assert any("field_b" in m for m in msgs)


def test_no_warning_for_known_fields(caplog):
    with caplog.at_level(logging.WARNING):
        _Sample(name="a", count=3)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == [], f"expected no warnings; got: {[r.message for r in warns]}"


def test_nodelocation_warns_on_legacy_fields(caplog):
    """The real-world case: a config with the retired ``uncertainty_m``
    field still parses, but the operator gets a warning."""
    from beagle_node.config.schema import NodeLocation
    with caplog.at_level(logging.WARNING):
        loc = NodeLocation(latitude_deg=47.5, longitude_deg=-122.1,
                           uncertainty_m=5.0, altitude_m=100.0)
    # Both legacy fields are dropped
    assert not hasattr(loc, "uncertainty_m")
    assert not hasattr(loc, "altitude_m")
    # Known fields still work
    assert loc.latitude_deg == 47.5
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("uncertainty_m" in m and "NodeLocation" in m for m in msgs)
    assert any("altitude_m" in m and "NodeLocation" in m for m in msgs)
