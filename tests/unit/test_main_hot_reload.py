# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Regression tests for the node main.run() hot-reload event-stamping path.

Background
----------
The on_measurement closure inside ``beagle_node.main.run`` builds a
``CarrierEvent`` for every detection.  Several event fields come from
``config.target_channels[0]`` -- in particular ``channel_frequency_hz``,
which the server uses as a primary grouping key when pairing events
from co-tuned peer nodes.

When the hot-reload retune path was added (commit d2fe7de), it correctly
called ``receiver.set_target_frequency(...)`` and rebound
``config.target_channels`` to a fresh list of TargetChannel objects, but
``on_measurement`` was reading from a closure-captured local
(``target_ch = config.target_channels[0]`` evaluated once at startup).
That local kept pointing at the OLD TargetChannel, so post-retune events
were stamped with the pre-retune ``frequency_hz`` -- silently splitting
the fleet into stale-stamped and fresh-stamped sub-groups in the
server's pairing logic, killing 4-node fixes even though every node was
actually tuned to the same frequency.

These tests pin the contract that ``channel_frequency_hz`` is read live
from ``config.target_channels`` on every event, not from a captured
local.
"""
from __future__ import annotations

from pathlib import Path


_MAIN_PY = Path(__file__).resolve().parents[2] / "src" / "beagle_node" / "main.py"


def _read_on_measurement_body() -> str:
    """Return the source body of the on_measurement closure inside run()."""
    src = _MAIN_PY.read_text()
    start_marker = "def on_measurement(m)"
    start = src.index(start_marker)
    # The closure ends at the next definition at the same indentation level
    # (4 spaces) or at the end of run().  Look for "\n    def " or "\n    "
    # followed by a non-space at the matching column.
    after = src[start + len(start_marker):]
    next_def = after.find("\n    def ")
    if next_def < 0:
        body_end = len(src)
    else:
        body_end = start + len(start_marker) + next_def
    return src[start:body_end]


def test_on_measurement_does_not_read_stale_captured_target_ch() -> None:
    """The closure must NOT reference ``target_ch.frequency_hz``.

    ``target_ch`` is the closure-captured local set once at startup
    (``target_ch = config.target_channels[0] if ...``).  Reading from it
    in ``on_measurement`` causes post-retune events to be stamped with
    the pre-retune frequency.  The fix is to read
    ``config.target_channels[0]`` live on each event.
    """
    body = _read_on_measurement_body()
    assert "target_ch.frequency_hz" not in body, (
        "on_measurement still reads channel_frequency_hz from the closure-"
        "captured target_ch local.  This is the hot-reload retune bug: post-"
        "retune events get stamped with the pre-retune frequency, splitting "
        "the fleet's events across two channel_hz buckets in server-side "
        "pairing.  Read config.target_channels[0] live inside the closure."
    )


def test_on_measurement_reads_target_channel_from_config_live() -> None:
    """The closure must read its current target from ``config.target_channels``.

    Positive form of the negative assertion above: at least one read of
    ``config.target_channels`` must appear inside the closure body.
    """
    body = _read_on_measurement_body()
    assert "config.target_channels" in body, (
        "on_measurement must read config.target_channels live (e.g. "
        "current_target = config.target_channels[0] ...) so that hot-reload "
        "retunes are reflected in the channel_frequency_hz of emitted events."
    )


def test_target_ch_local_kept_only_for_startup_validation() -> None:
    """Document the intent of the lingering ``target_ch`` local.

    The ``target_ch`` local at run() scope is preserved for startup-time
    validation/logging context only.  This test ensures any future
    refactor that drops it does not also drop the regression comments
    explaining why the closure must read live state.
    """
    src = _MAIN_PY.read_text()
    assert "target_ch = config.target_channels[0]" in src
    # And the rationale comment lives next to the definition.
    assert "hot-reload" in src.lower(), (
        "Lost the hot-reload rationale comment near target_ch / on_measurement."
    )
