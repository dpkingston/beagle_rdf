# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
EventReporter - non-blocking HTTP event submission.

Accepts CarrierEvent objects via submit(), queues them, and POSTs them to
the central server from a background worker thread.  The pipeline is never
blocked by network I/O.

Retry policy
------------
When connected: exponential backoff (1 s, 2 s, 4 s) then drop.
When disconnected (circuit-breaker open): one attempt per event, no backoff,
no per-failure logging.  Reconnection is logged at ERROR level so it stands
out in production logs.  An hourly reminder is logged if still disconnected.

If the queue is full (default 1000 events), the oldest event is dropped and
a warning is logged - this prevents memory growth during extended outages.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Any

import httpx

from beagle_node.events.model import CarrierEvent

logger = logging.getLogger(__name__)

_SENTINEL = object()   # signals worker to stop

_REMINDER_INTERVAL_S = 3600.0  # log a reminder once per hour while disconnected


class EventReporter:
    """
    Background-threaded HTTP reporter.

    Parameters
    ----------
    server_url : str
        Base URL of the reporting server, e.g. ``https://tdoa.example.com``.
        Events are POSTed to ``{server_url}/api/v1/events``.
    auth_token : str
        Bearer token sent in the ``Authorization`` header.
    max_queue : int
        Maximum number of events to buffer.  Oldest are dropped when full.
    timeout_s : float
        HTTP request timeout in seconds.
    max_retries : int
        Retry attempts before dropping a single event (used only while connected).
    retry_base_s : float
        Base sleep time for exponential backoff between retries.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str,
        max_queue: int = 1_000,
        timeout_s: float = 5.0,
        max_retries: int = 3,
        retry_base_s: float = 1.0,
        max_events_per_window: int = 5,
        events_rate_window_s: float = 5.0,
    ) -> None:
        if max_events_per_window < 0:
            raise ValueError(
                f"max_events_per_window must be >= 0, got {max_events_per_window}"
            )
        if events_rate_window_s <= 0.0:
            raise ValueError(
                f"events_rate_window_s must be > 0, got {events_rate_window_s}"
            )
        self._disabled = not server_url.strip()
        self._url = server_url.rstrip("/") + "/api/v1/events"
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"
        self._max_queue = max_queue
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._retry_base = retry_base_s

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._events_submitted: int = 0
        self._events_dropped: int = 0
        self._lock = threading.Lock()

        # Circuit-breaker state
        self._disconnected: bool = False
        self._fail_count: int = 0
        self._last_reminder_ts: float = 0.0

        # Local sliding-window rate limit on submit().  Prevents a chattering
        # detector from flooding the server; the server has its own stricter
        # rate limit but we should stop feeding the pipe far upstream of that.
        self._max_events_per_window = int(max_events_per_window)
        self._rate_window_s = float(events_rate_window_s)
        self._submit_times: collections.deque[float] = collections.deque()
        # Dedup state for drop logging: (last_log_monotonic, count_since_last).
        self._drop_log_state: tuple[float, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def events_submitted(self) -> int:
        with self._lock:
            return self._events_submitted

    @property
    def events_dropped(self) -> int:
        with self._lock:
            return self._events_dropped

    def post_heartbeat(self, payload: dict[str, Any]) -> None:
        """
        Send a heartbeat POST to the server.  Non-blocking: fires a one-shot
        thread so the pipeline is never stalled by network I/O.
        """
        if self._disabled:
            return
        url = self._url.rsplit("/", 3)[0] + "/api/v1/heartbeat"

        def _send() -> None:
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        url,
                        json=payload,
                        headers=self._headers,
                    )
                    if resp.status_code in (200, 201, 202, 204):
                        logger.debug("Heartbeat delivered (HTTP %d)", resp.status_code)
                    else:
                        logger.warning("Heartbeat HTTP %d", resp.status_code)
            except httpx.TransportError as exc:
                logger.debug("Heartbeat failed: %s", exc)

        threading.Thread(target=_send, name="heartbeat", daemon=True).start()

    def submit(self, event: CarrierEvent) -> None:
        """
        Queue an event for delivery.  Non-blocking.

        Applies a sliding-window rate limit first (max_events_per_window
        events per events_rate_window_s seconds); excess events are dropped
        with deduplicated logging.

        If the queue is full the oldest pending event is discarded to make
        room, rather than blocking the caller.
        """
        # Sliding-window rate check.  0 = disabled.
        if self._max_events_per_window > 0:
            now = time.monotonic()
            dq = self._submit_times
            cutoff = now - self._rate_window_s
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max_events_per_window:
                with self._lock:
                    self._events_dropped += 1
                self._log_rate_drop(now)
                return
            dq.append(now)

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to make room
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._events_dropped += 1
            logger.warning("Reporter queue full - dropping oldest event")
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass

    def _log_rate_drop(self, now: float) -> None:
        """Deduped WARNING for rate-limit drops.

        On the first drop after a cooldown, log a WARNING.  Subsequent
        drops within the same window are counted but not logged.  When the
        next WARNING fires (at least ``events_rate_window_s`` later), it
        reports how many were silently suppressed.
        """
        prev = self._drop_log_state
        cooldown_expired = prev is None or (now - prev[0]) >= self._rate_window_s
        if cooldown_expired:
            suppressed = prev[1] if prev is not None else 0
            if suppressed > 0:
                logger.warning(
                    "Reporter rate limit still active: %d more events dropped "
                    "in the last %.1fs (limit %d/%.1fs)",
                    suppressed, self._rate_window_s,
                    self._max_events_per_window, self._rate_window_s,
                )
            else:
                logger.warning(
                    "Reporter rate limit: dropping event (exceeded %d events/%.1fs); "
                    "further drops suppressed until cooldown",
                    self._max_events_per_window, self._rate_window_s,
                )
            self._drop_log_state = (now, 0)
        else:
            self._drop_log_state = (prev[0], prev[1] + 1)

    def start(self) -> None:
        """Start the background delivery thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._worker, name="event-reporter", daemon=True
        )
        self._thread.start()
        if self._disabled:
            logger.warning(
                "EventReporter: no server_url configured - events will be logged "
                "locally only (set reporter.server_url in node.yaml to enable upload)"
            )
        else:
            logger.info("EventReporter started, posting to %s", self._url)

    def stop(self, timeout_s: float = 10.0) -> None:
        """
        Signal the worker to stop and wait for it to drain the queue.

        Events already in the queue when stop() is called are delivered
        before the thread exits (up to timeout_s).
        """
        if self._thread is None:
            return
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            logger.warning("EventReporter worker did not stop within %.0f s", timeout_s)
        self._thread = None

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                self._deliver(client, item)

    def _deliver(self, client: httpx.Client, event: CarrierEvent) -> None:
        if self._disabled:
            logger.info(
                "Event (local-only): sync_delta_ns=%d corr=%.3f channel=%.3f MHz",
                event.sync_delta_ns, event.sync_corr_peak,
                event.channel_frequency_hz / 1e6,
            )
            with self._lock:
                self._events_submitted += 1
            return

        payload = event.model_dump_json()

        if self._disconnected:
            # Circuit-breaker open: one attempt, no backoff, no per-failure log.
            if self._try_post(client, payload, event.event_id):
                self._on_reconnect()
                return
            self._on_fail_disconnected()
            return

        # Circuit-breaker closed: normal retry with backoff.
        for attempt in range(self._max_retries):
            if self._try_post(client, payload, event.event_id):
                return
            if attempt < self._max_retries - 1:
                sleep_s = self._retry_base * (2 ** attempt)
                time.sleep(sleep_s)

        # All retries exhausted - enter disconnected state.
        self._on_fail_connected()

    def _try_post(self, client: httpx.Client, payload: str, event_id: str) -> bool:
        """Attempt one POST. Returns True on success, False on failure."""
        try:
            resp = client.post(self._url, content=payload, headers=self._headers)
            if resp.status_code in (200, 201, 202, 204):
                with self._lock:
                    self._events_submitted += 1
                logger.debug("Event %s delivered (HTTP %d)", event_id, resp.status_code)
                return True
            if not self._disconnected:
                logger.warning(
                    "Server returned HTTP %d for event %s",
                    resp.status_code, event_id,
                )
        except httpx.TransportError as exc:
            if not self._disconnected:
                logger.warning("Transport error for event %s: %s", event_id, exc)
        return False

    def _on_fail_connected(self) -> None:
        """Transition from connected to disconnected (circuit-breaker opens)."""
        self._disconnected = True
        self._fail_count = 1
        self._last_reminder_ts = time.monotonic()
        with self._lock:
            self._events_dropped += 1
        logger.error(
            "Server unreachable at %s - entering disconnected mode; "
            "will retry once per event without backoff",
            self._url,
        )

    def _on_fail_disconnected(self) -> None:
        """Record another failure while already disconnected."""
        self._fail_count += 1
        with self._lock:
            self._events_dropped += 1
        now = time.monotonic()
        if now - self._last_reminder_ts >= _REMINDER_INTERVAL_S:
            logger.warning(
                "Still disconnected from server (%d events dropped since last reminder)",
                self._fail_count,
            )
            self._fail_count = 0
            self._last_reminder_ts = now

    def _on_reconnect(self) -> None:
        """Transition from disconnected back to connected."""
        logger.error(
            "Reconnected to server after %d failed events", self._fail_count
        )
        self._disconnected = False
        self._fail_count = 0
