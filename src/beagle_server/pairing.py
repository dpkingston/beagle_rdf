# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Event pairing and delivery-buffer logic.

Design
------
Events are grouped by:

    base_key  = (channel_hz_bucket, event_type, sync_tx_id)

    t_sync_ns = onset_time_ns - sync_delta_ns - dist(sync_tx, node) / c * 1e9

T_sync_ns is the estimated absolute wall-clock time of the FM sync event
**at the transmitter site**.  Because every node hears the same FM pilot
broadcast at the same physical instant, T_sync_ns should agree across all
nodes to within the inter-node NTP/GPS clock error (typically < 50 ms for
internet NTP).

Group membership
----------------
Rather than assigning each event to a fixed-size bucket (which causes
boundary-split failures when the true T_sync falls within NTP-noise distance
of a bucket edge), we use anchor-based proximity matching:

1. When the first event with a given base_key arrives, a new group is created
   and its T_sync value becomes the *anchor*.
2. Subsequent events with the same base_key are added to an existing group
   if their T_sync is within `correlation_window_s / 2` of that group's anchor.
3. If no group is close enough, a new group is created.

This eliminates boundary-split failures for realistic NTP noise (< 50 ms),
while still separating rapid key-ups whose T_sync values differ by more than
half the window.

Delivery buffer
---------------
When the first event in a new group arrives, a delivery timer is set for
`delivery_buffer_s` seconds.  When the timer fires, all events accumulated
in the group are passed to the fix solver.  Set `delivery_buffer_s` to the
observed worst-case one-way delivery latency from the slowest node (the
time between the LMR event occurring and the server receiving the POST).

Groups older than group_expiry_s are evicted from the in-memory store.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from beagle_server.tdoa import _C_M_S, _T_SYNC_NS, haversine_m

logger = logging.getLogger(__name__)

# Type alias for the fix callback
FixCallback = Callable[[list[dict[str, Any]]], Coroutine[Any, Any, None]]


@dataclass
class _Group:
    """In-memory state for one pending event group."""
    key: tuple                        # (channel_bucket, event_type, sync_tx_id, t_sync_anchor_ns)
    t_sync_anchor_ns: int             # T_sync of the first event; nearby events join this group
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    # event_id -> event row; dict prevents duplicates / implements amendment
    first_received_at: float = field(default_factory=time.time)
    fix_scheduled: bool = False


class EventPairer:
    """
    In-memory delivery-buffer and pairing engine.

    Call :meth:`add_event` from the FastAPI POST handler.  When the
    delivery buffer expires the provided ``on_group_ready`` coroutine is
    awaited with the list of event dicts that belong to the group.

    Parameters
    ----------
    on_group_ready :
        Async callback receiving a list of event dicts when a group is ready.
    correlation_window_s :
        Grouping window for T_sync.  A new event joins an existing group if
        its T_sync is within ``correlation_window_s / 2`` of that group's
        anchor.  Default 0.2 s works for standard internet NTP (observed max
        inter-node spread ~39 ms, giving a 2.5x safety margin).
        Use 0.005 s for GPS-disciplined nodes.
    delivery_buffer_s :
        Seconds to wait after the first event in a group before calling
        on_group_ready.  Set to the observed worst-case one-way delivery
        latency from the slowest node.
    group_expiry_s :
        Seconds after which an unfired group is evicted.
    freq_tolerance_hz :
        Channel frequency bucketing width.
    min_nodes :
        Minimum number of distinct nodes required to attempt a solution.
        3+ nodes yield a unique 2-D position fix; 2 nodes yield a hyperbolic
        line-of-position (LOP).  Default 2.
    """

    def __init__(
        self,
        on_group_ready: FixCallback,
        correlation_window_s: float = 0.2,
        delivery_buffer_s: float = 10.0,
        group_expiry_s: float = 60.0,
        freq_tolerance_hz: float = 1000.0,
        min_nodes: int = 3,
    ) -> None:
        self._on_ready = on_group_ready
        self._half_window_ns = int(correlation_window_s * 1e9) // 2
        self._delivery_s = delivery_buffer_s
        self._expiry_s = group_expiry_s
        self._freq_tol = freq_tolerance_hz
        self._min_nodes = min_nodes
        self._groups: dict[tuple, _Group] = {}

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _fmt_key(self, key: tuple) -> str:
        """Render a group key as a human-readable string for log messages.

        key = (channel_bucket, event_type, sync_tx_id, t_sync_anchor_ns)
        Returns e.g.  "443.4750MHz onset KUOW_94.9 T_sync=1775510456.424"
        """
        if len(key) < 4:
            return repr(key)
        channel_bucket, event_type, sync_tx_id, t_sync_anchor_ns = key
        channel_mhz = (channel_bucket * self._freq_tol) / 1e6
        # T_sync as fractional seconds since epoch (truncated to ms for brevity)
        t_sync_s = t_sync_anchor_ns / 1e9
        return (
            f"{channel_mhz:.4f}MHz {event_type} {sync_tx_id} "
            f"T_sync={t_sync_s:.3f}"
        )

    @staticmethod
    def _fmt_nodes(events: dict[str, dict[str, Any]] | list[dict[str, Any]]) -> str:
        """Render the unique node IDs in a group for log messages."""
        items = events.values() if isinstance(events, dict) else events
        node_ids = sorted({e.get("node_id", "?") for e in items})
        return ",".join(node_ids) if node_ids else "(none)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_event(self, event: dict[str, Any]) -> None:
        """
        Register a new event.  Schedules a delivery-buffer task if this is
        the first event in its group.

        Must be awaited from within a running asyncio event loop (i.e. from
        an async FastAPI handler or background task).
        """
        self._evict_expired()
        base_key = self._base_key(event)
        t_sync_ns = self._t_sync_ns(event)

        # Find an existing group whose T_sync anchor is within half the window.
        grp = self._find_group(base_key, t_sync_ns, event["node_id"])

        # Prevent different transmissions from the same node being merged
        # into one group.  Event amendments (same event_id) are fine — they
        # overwrite the existing entry.  But a NEW event_id from a node that
        # already has an event in the group means a different transmission
        # landed within the correlation window; force it into its own group.
        if grp is not None:
            eid = event["event_id"]
            nid = event["node_id"]
            has_different_event = any(
                e["node_id"] == nid and e["event_id"] != eid
                for e in grp.events.values()
            )
            if has_different_event:
                logger.info(
                    "Rejecting group merge: node %s already has a different "
                    "event in group [%s]; creating new group",
                    nid, self._fmt_key(grp.key),
                )
                grp = None

        if grp is None:
            # No matching group - create one anchored to this event's T_sync.
            full_key = (*base_key, t_sync_ns)
            grp = _Group(key=full_key, t_sync_anchor_ns=t_sync_ns)
            self._groups[full_key] = grp
            logger.debug(
                "New event group [%s] first node=%s",
                self._fmt_key(full_key), event.get("node_id", "?"),
            )

        grp.events[event["event_id"]] = event

        if not grp.fix_scheduled:
            grp.fix_scheduled = True
            asyncio.ensure_future(self._schedule_fix(grp.key))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _t_sync_ns(self, event: dict[str, Any]) -> int:
        """
        Estimate the absolute wall-clock time of the FM sync event at the
        transmitter, in nanoseconds since the Unix epoch.

            T_sync = onset_time_ns - (sync_delta_ns mod T_sync) - dist(sync_tx, node) / c

        The modular reduction ensures that nodes with different observation
        windows (e.g. freq_hop seeing sync pulses only during the sync block)
        produce the same T_sync as nodes with continuous observation.  Without
        the reduction, T_sync would reference a different (older) sync event.
        """
        dist_m = haversine_m(
            event["sync_tx_lat"], event["sync_tx_lon"],
            event["node_lat"],    event["node_lon"],
        )
        propagation_ns = int(dist_m / _C_M_S * 1e9)
        sync_delta_reduced = event["sync_delta_ns"] % int(_T_SYNC_NS)
        return event["onset_time_ns"] - sync_delta_reduced - propagation_ns

    def _base_key(self, event: dict[str, Any]) -> tuple:
        """
        Return the channel/type/station prefix of the grouping key.

        Events with the same base_key but different T_sync values may still
        belong to different transmissions (rapid key-ups); T_sync proximity
        is checked separately in :meth:`_find_group`.
        """
        channel_bucket = round(event["channel_hz"] / self._freq_tol)
        return (channel_bucket, event["event_type"], event["sync_tx_id"])

    def _find_group(self, base_key: tuple, t_sync_ns: int, node_id: str) -> _Group | None:
        """
        Return the first group whose base_key matches and whose T_sync anchor
        is within half the correlation window of ``t_sync_ns``.

        Also tries pilot-period disambiguation: if the T_sync difference
        exceeds the direct-match window but is approximately an integer
        multiple of _T_SYNC_NS (RDS bit period ~842 usec), the event joins the
        group - provided the incoming node is not already in the group.

        The per-node guard is essential: because any time difference is within
        3.5 ms of *some* multiple of 7 ms, omitting the guard would cause
        different-transmission events from the same node to be falsely merged.
        The guard is safe for event amendments (same event_id, tiny T_sync
        delta) because amendments always pass the direct-match check first.

        Returns None if no such group exists.
        """
        for grp in self._groups.values():
            if grp.key[:3] != base_key:
                continue
            delta = t_sync_ns - grp.t_sync_anchor_ns
            if abs(delta) <= self._half_window_ns:
                return grp
            # Sync-period disambiguation for cross-node clock offsets.
            # Only attempt when this node has no existing event in the group,
            # so that different transmissions from the same node are not merged.
            node_in_group = any(
                e["node_id"] == node_id for e in grp.events.values()
            )
            if not node_in_group:
                n = round(delta / _T_SYNC_NS)
                if n != 0 and abs(delta - n * _T_SYNC_NS) <= self._half_window_ns:
                    logger.debug(
                        "Sync-period disambiguation: node=%s joining "
                        "group [%s] (existing nodes=%s) with n=%+d "
                        "(delta=%.3f ms)",
                        node_id, self._fmt_key(grp.key),
                        self._fmt_nodes(grp.events), n, delta / 1e6,
                    )
                    return grp
        return None

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [
            (k, g) for k, g in self._groups.items()
            if now - g.first_received_at > self._expiry_s
        ]
        for k, g in stale:
            logger.debug(
                "Evicting expired group [%s] nodes=%s",
                self._fmt_key(k), self._fmt_nodes(g.events),
            )
            del self._groups[k]

    async def _schedule_fix(self, key: tuple) -> None:
        """Sleep for delivery_buffer_s then fire the group."""
        await asyncio.sleep(self._delivery_s)
        grp = self._groups.pop(key, None)
        if grp is None:
            return  # evicted while we slept

        events = list(grp.events.values())
        node_ids = {e["node_id"] for e in events}
        if len(node_ids) < self._min_nodes:
            logger.info(
                "Group [%s] has only %d node(s) after delivery buffer "
                "(need %d for a fix) - skipping; nodes=%s",
                self._fmt_key(key), len(node_ids), self._min_nodes,
                self._fmt_nodes(events),
            )
            return

        logger.info(
            "Group [%s] ready: %d events from %d nodes - running fix; nodes=%s",
            self._fmt_key(key), len(events), len(node_ids),
            self._fmt_nodes(events),
        )
        try:
            await self._on_ready(events)
        except Exception:
            logger.exception(
                "Fix computation failed for group [%s] nodes=%s",
                self._fmt_key(key), self._fmt_nodes(events),
            )

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def pending_group_count(self) -> int:
        """Number of groups currently waiting for their delivery buffer."""
        return len(self._groups)
