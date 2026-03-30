# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for beagle_server/pairing.py."""

from __future__ import annotations

import asyncio
import time

import pytest

from beagle_server.pairing import EventPairer


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_NOW_NS = int(time.time() * 1e9)


def make_event(
    node_id: str = "node-A",
    channel_hz: float = 155_100_000.0,
    event_type: str = "onset",
    sync_tx_id: str = "KISW_99.9",
    onset_time_ns: int = _NOW_NS,
    sync_delta_ns: int = 500_000_000,
    corr_peak: float = 0.8,
    event_id: str | None = None,
) -> dict:
    return {
        "event_id":      event_id or f"uuid-{node_id}",
        "node_id":       node_id,
        "channel_hz":    channel_hz,
        "event_type":    event_type,
        "sync_tx_id":    sync_tx_id,
        "onset_time_ns": onset_time_ns,
        "sync_delta_ns": sync_delta_ns,
        "corr_peak":     corr_peak,
        "node_lat":      47.6,
        "node_lon":      -122.3,
        "sync_tx_lat":   47.7,
        "sync_tx_lon":   -122.4,
    }


def _null_cb():
    async def cb(events): pass
    return cb


# ---------------------------------------------------------------------------
# Base key bucketing (synchronous - test _base_key directly)
# ---------------------------------------------------------------------------

def test_same_channel_same_bucket():
    pairer = EventPairer(_null_cb(), freq_tolerance_hz=1000.0)
    e1 = make_event("A", channel_hz=155_100_000.0)
    e2 = make_event("B", channel_hz=155_100_200.0)  # within 1 kHz
    assert pairer._base_key(e1) == pairer._base_key(e2)


def test_different_channels_different_buckets():
    pairer = EventPairer(_null_cb(), freq_tolerance_hz=1000.0)
    e1 = make_event("A", channel_hz=155_100_000.0)
    e2 = make_event("B", channel_hz=462_000_000.0)
    assert pairer._base_key(e1) != pairer._base_key(e2)


def test_different_event_types_different_keys():
    pairer = EventPairer(_null_cb())
    e1 = make_event("A", event_type="onset")
    e2 = make_event("B", event_type="offset")
    assert pairer._base_key(e1) != pairer._base_key(e2)


def test_different_sync_stations_different_keys():
    pairer = EventPairer(_null_cb())
    e1 = make_event("A", sync_tx_id="KISW_99.9")
    e2 = make_event("B", sync_tx_id="KUOW_94.9")
    assert pairer._base_key(e1) != pairer._base_key(e2)


# ---------------------------------------------------------------------------
# Delivery buffer and fix callback (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delivery_buffer_fires_with_two_nodes():
    """Two events from different nodes -> callback fires after buffer."""
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01, group_expiry_s=60.0, min_nodes=2)
    await pairer.add_event(make_event("A", event_id="e1"))
    await pairer.add_event(make_event("B", event_id="e2"))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1
    node_ids = {e["node_id"] for e in received_groups[0]}
    assert node_ids == {"A", "B"}


@pytest.mark.asyncio
async def test_delivery_buffer_skips_single_node():
    """Single node group -> callback not called (cannot compute TDOA)."""
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01)
    await pairer.add_event(make_event("A", event_id="e1"))

    await asyncio.sleep(0.05)

    assert received_groups == []


@pytest.mark.asyncio
async def test_events_from_three_nodes_all_delivered():
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01)
    for node in ("A", "B", "C"):
        await pairer.add_event(make_event(node, event_id=f"e-{node}"))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1
    assert len(received_groups[0]) == 3


@pytest.mark.asyncio
async def test_duplicate_event_id_not_double_counted():
    """
    Submitting the same event_id twice (amendment) must not double the event.

    Realistic scenario: the amendment corrects sync_delta_ns by a few nanoseconds.
    Both versions must land in the same T_sync bucket (same transmission).
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01, min_nodes=2)
    # Node A reports 500 ms + 100 ns after its FM sync event, then amends to +200 ns.
    # Node B independently heard the same LMR onset at 500 ms + 0 ns.
    # All three events have T_sync within ~200 ns of each other - same group.
    await pairer.add_event(make_event("A", event_id="same-id", sync_delta_ns=500_000_100))
    await pairer.add_event(make_event("A", event_id="same-id", sync_delta_ns=500_000_200))  # amendment
    await pairer.add_event(make_event("B", event_id="b-id", sync_delta_ns=500_000_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1
    node_a_events = [e for e in received_groups[0] if e["node_id"] == "A"]
    assert len(node_a_events) == 1
    assert node_a_events[0]["sync_delta_ns"] == 500_000_200  # amendment overwrote


@pytest.mark.asyncio
async def test_different_channels_fire_separately():
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01, min_nodes=2)
    await pairer.add_event(make_event("A", channel_hz=155_100_000.0, event_id="e1"))
    await pairer.add_event(make_event("B", channel_hz=155_100_000.0, event_id="e2"))
    await pairer.add_event(make_event("A", channel_hz=462_000_000.0, event_id="e3"))
    await pairer.add_event(make_event("B", channel_hz=462_000_000.0, event_id="e4"))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 2
    channels = {e["channel_hz"] for grp in received_groups for e in grp}
    assert 155_100_000.0 in channels
    assert 462_000_000.0 in channels


@pytest.mark.asyncio
async def test_pending_group_count():
    pairer = EventPairer(_null_cb(), delivery_buffer_s=60.0)
    assert pairer.pending_group_count() == 0
    await pairer.add_event(make_event("A", event_id="e1"))
    assert pairer.pending_group_count() == 1
    await pairer.add_event(make_event("A", channel_hz=462e6, event_id="e2"))
    assert pairer.pending_group_count() == 2


@pytest.mark.asyncio
async def test_rapid_keyups_produce_separate_groups():
    """
    Two quick transmissions on the same channel, 500 ms apart, must be
    treated as separate events and produce two independent fix groups.

    With the T_sync grouping, the two transmissions have T_sync values 500 ms
    apart - well outside the 200 ms correlation window - so they go into
    different buckets and fire separately.
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    # correlation_window_s=0.2 -> half-window 100ms; 500ms separation -> separate groups
    pairer = EventPairer(cb, delivery_buffer_s=0.01, correlation_window_s=0.2, min_nodes=2)

    # Transmission 1: onset_time_ns = _NOW_NS, sync_delta_ns = 500_000_000
    await pairer.add_event(make_event("A", event_id="t1-a",
                                     onset_time_ns=_NOW_NS, sync_delta_ns=500_000_000))
    await pairer.add_event(make_event("B", event_id="t1-b",
                                     onset_time_ns=_NOW_NS, sync_delta_ns=500_000_000))

    # Transmission 2: 500 ms later - different T_sync, goes to a different bucket
    onset2 = _NOW_NS + 500_000_000  # 500 ms later
    await pairer.add_event(make_event("A", event_id="t2-a",
                                     onset_time_ns=onset2, sync_delta_ns=500_000_000))
    await pairer.add_event(make_event("B", event_id="t2-b",
                                     onset_time_ns=onset2, sync_delta_ns=500_000_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 2, (
        f"Expected 2 separate fix groups for 2 transmissions, got {len(received_groups)}"
    )


@pytest.mark.asyncio
async def test_freq_hop_sync_delta_groups_with_rspduo():
    """
    A freq_hop node (sync_delta spanning multiple 7 ms sync periods) must
    group with an RSPduo node (sync_delta 0-7 ms) for the same transmission.

    freq_hop sync_delta = 32_600_000 ns (32.6 ms = 4 full periods + 4.6 ms)
    RSPduo  sync_delta =  4_600_000 ns (4.6 ms = within one period)

    After mod-7ms reduction both become ~4.6 ms, so T_sync values match.
    Without the reduction, T_sync would differ by 28 ms (4 x 7 ms) and
    might or might not pair depending on the correlation window.
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    # Use a tight window (20 ms half-window) to prove the mod reduction
    # is working - without it, the 28 ms T_sync offset would exceed
    # the half-window and the events would land in separate groups.
    pairer = EventPairer(cb, delivery_buffer_s=0.01, correlation_window_s=0.04,
                         min_nodes=2)

    await pairer.add_event(make_event("rtlsdr-north", event_id="fh-1",
                                      onset_time_ns=_NOW_NS,
                                      sync_delta_ns=32_600_000))
    await pairer.add_event(make_event("rspduo-north", event_id="rsp-1",
                                      onset_time_ns=_NOW_NS,
                                      sync_delta_ns=4_600_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1, (
        f"Expected 1 merged group (freq_hop + RSPduo with mod-7ms reduction); "
        f"got {len(received_groups)}"
    )
    node_ids = {e["node_id"] for e in received_groups[0]}
    assert node_ids == {"rtlsdr-north", "rspduo-north"}


@pytest.mark.asyncio
async def test_cross_node_clock_offset_paired_via_pilot_disambiguation():
    """
    A freq_hop node (accurate NTP) and an RSPduo node whose HAS_TIME anchor
    was captured when NTP was ~315 ms ahead (45 x 7 ms) must be grouped for
    the same transmission via pilot-period disambiguation.

    After sync_delta mod-7ms reduction the T_sync values differ by ~315 ms --
    outside the 250 ms half-window for direct matching - but are within 3.5 ms
    of 45 x 7 ms, so disambiguation succeeds.

    freq_hop  sync_delta = 32_600_000 ns  ->  reduced = 4_600_000 ns (4.6 ms)
    RSPduo    sync_delta =  5_000_000 ns  ->  reduced = 5_000_000 ns (5.0 ms)
    RSPduo clock offset  = +315_000_000 ns  (= 45 x 7_000_000 ns exactly)

    Expected T_sync delta = 315_000_000 - (5_000_000 - 4_600_000)
                           = 314_600_000 ns  ~ 314.6 ms
    n = round(314.6 / 7) = 45,  residual = |314.6 - 315| = 0.4 ms  OK
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01, correlation_window_s=0.5,
                         min_nodes=2)

    # freq_hop node: accurate NTP
    await pairer.add_event(make_event("rtlsdr-server", event_id="fh-1",
                                      onset_time_ns=_NOW_NS,
                                      sync_delta_ns=32_600_000))
    # RSPduo node: HAS_TIME anchor captured when NTP was 315 ms ahead
    await pairer.add_event(make_event("rspduo-node", event_id="rsp-1",
                                      onset_time_ns=_NOW_NS + 315_000_000,
                                      sync_delta_ns=5_000_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1, (
        f"Expected 1 merged group (freq_hop + RSPduo with ~315 ms clock offset "
        f"= 45x7ms, should pair via pilot disambiguation); got {len(received_groups)}"
    )
    node_ids = {e["node_id"] for e in received_groups[0]}
    assert node_ids == {"rtlsdr-server", "rspduo-node"}


@pytest.mark.asyncio
async def test_same_node_rapid_keyups_not_merged_by_disambiguation():
    """
    Two transmissions 497 ms apart (~ 71 x 7 ms) on the same node must NOT
    be falsely merged by pilot-period disambiguation.

    Without the per-node guard, round(497/7) = 71, residual = 0 ms < half-window,
    so disambiguation would wrongly accept the second event into the first group.
    The guard skips disambiguation when the incoming node is already in the group.
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    pairer = EventPairer(cb, delivery_buffer_s=0.01, correlation_window_s=0.2,
                         min_nodes=2)

    # Transmission 1
    await pairer.add_event(make_event("A", event_id="t1-A",
                                      onset_time_ns=_NOW_NS,
                                      sync_delta_ns=500_000_000))
    await pairer.add_event(make_event("B", event_id="t1-B",
                                      onset_time_ns=_NOW_NS,
                                      sync_delta_ns=500_000_000))

    # Transmission 2: 497 ms later - exactly 71 x 7 ms, so disambiguation
    # residual = 0 ms, but same-node guard must block the merge.
    await pairer.add_event(make_event("A", event_id="t2-A",
                                      onset_time_ns=_NOW_NS + 497_000_000,
                                      sync_delta_ns=500_000_000))
    await pairer.add_event(make_event("B", event_id="t2-B",
                                      onset_time_ns=_NOW_NS + 497_000_000,
                                      sync_delta_ns=500_000_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 2, (
        f"Expected 2 separate groups for 2 transmissions 497 ms apart "
        f"(same-node guard must prevent pilot-disambiguation false merge); "
        f"got {len(received_groups)}"
    )


@pytest.mark.asyncio
async def test_tsync_near_boundary_merged_into_one_group():
    """
    Regression test for the fixed-bucket boundary-split bug.

    With the old round(T_sync / window) bucketing, two events whose T_sync
    values straddle a bucket boundary land in different groups even though
    they heard the same transmission.  Anchor-based proximity matching fixes
    this: the second event joins the first group as long as its T_sync is
    within half the correlation window of the anchor.

    Setup: 80 ms T_sync spread (within the 100 ms half-window of a 200 ms
    window), so both events must land in the same group.
    """
    received_groups: list[list] = []

    async def cb(events):
        received_groups.append(events)

    # Half-window = 100 ms.  T_sync spread = 80 ms -> must merge.
    pairer = EventPairer(cb, delivery_buffer_s=0.01, correlation_window_s=0.2, min_nodes=2)

    # All make_event() calls use identical node/sync_tx positions, so
    # T_sync difference == onset_time_ns difference.
    await pairer.add_event(make_event("A", event_id="e-A",
                                     onset_time_ns=_NOW_NS,
                                     sync_delta_ns=500_000_000))
    await pairer.add_event(make_event("B", event_id="e-B",
                                     onset_time_ns=_NOW_NS + 80_000_000,  # 80 ms later
                                     sync_delta_ns=500_000_000))

    await asyncio.sleep(0.05)

    assert len(received_groups) == 1, (
        f"Expected 1 merged group (T_sync spread 80ms < 100ms half-window); "
        f"got {len(received_groups)}.  Fixed-bucket approach would have split these."
    )
    node_ids = {e["node_id"] for e in received_groups[0]}
    assert node_ids == {"A", "B"}
