# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Event loop watchdog - detects when the asyncio event loop is blocked.

A daemon thread periodically schedules a no-op callback on the event loop.
If the callback doesn't execute within ``threshold_s``, the watchdog logs
a WARNING with full diagnostics: all thread stack traces and pending
asyncio tasks.  This makes it easy to identify what is blocking the loop
(e.g. a synchronous solver call, a blocking DB query, etc.).

Usage (in your FastAPI lifespan)::

    from beagle_server.watchdog import start_watchdog, stop_watchdog

    watchdog = start_watchdog(asyncio.get_event_loop(), threshold_s=2.0)
    yield
    stop_watchdog(watchdog)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import traceback
import time

logger = logging.getLogger(__name__)


class EventLoopWatchdog:
    """Monitors the asyncio event loop from a daemon thread."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        threshold_s: float = 2.0,
        check_interval_s: float = 1.0,
    ) -> None:
        self._loop = loop
        self._threshold_s = threshold_s
        self._check_interval_s = check_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="event-loop-watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Event loop watchdog started (threshold=%.1fs, interval=%.1fs)",
            self._threshold_s, self._check_interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Event loop watchdog stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._check_interval_s)
            if self._stop_event.is_set():
                break
            self._check_loop()

    def _check_loop(self) -> None:
        """Schedule a callback on the event loop and wait for it to fire."""
        responded = threading.Event()

        def _pong() -> None:
            responded.set()

        try:
            self._loop.call_soon_threadsafe(_pong)
        except RuntimeError:
            # Event loop is closed - server is shutting down.
            return

        if responded.wait(timeout=self._threshold_s):
            return  # event loop is healthy

        # Event loop did not respond in time - dump diagnostics.
        self._dump_diagnostics()

    def _dump_diagnostics(self) -> None:
        logger.warning(
            "EVENT LOOP BLOCKED - no response in %.1fs. Dumping diagnostics.",
            self._threshold_s,
        )

        # 1. All thread stack traces
        lines = ["\n=== Thread stack traces ==="]
        for thread_id, frame in sys._current_frames().items():
            thread_name = _thread_name(thread_id)
            lines.append(f"\n--- Thread {thread_id} ({thread_name}) ---")
            lines.extend(traceback.format_stack(frame))
        logger.warning("".join(lines))

        # 2. Pending asyncio tasks
        try:
            tasks = asyncio.all_tasks(self._loop)
            lines = [f"\n=== Pending asyncio tasks ({len(tasks)}) ==="]
            for task in tasks:
                coro = task.get_coro()
                state = task.get_name()
                lines.append(f"  {state}: {coro}")
                if not task.done():
                    # Get the coroutine's current stack frame
                    frames = task.get_stack(limit=5)
                    for frame in frames:
                        lines.append(
                            f"    File \"{frame.f_code.co_filename}\", "
                            f"line {frame.f_lineno}, in {frame.f_code.co_name}"
                        )
            logger.warning("\n".join(lines))
        except RuntimeError:
            logger.warning("Could not enumerate asyncio tasks (loop closed?)")


def _thread_name(thread_id: int) -> str:
    """Look up a human-readable name for a thread ID."""
    for t in threading.enumerate():
        if t.ident == thread_id:
            return t.name
    return "unknown"


def start_watchdog(
    loop: asyncio.AbstractEventLoop,
    threshold_s: float = 2.0,
    check_interval_s: float = 1.0,
) -> EventLoopWatchdog:
    """Create and start a watchdog for the given event loop."""
    wd = EventLoopWatchdog(loop, threshold_s, check_interval_s)
    wd.start()
    return wd


def stop_watchdog(wd: EventLoopWatchdog) -> None:
    """Stop a running watchdog."""
    wd.stop()
