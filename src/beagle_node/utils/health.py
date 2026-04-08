# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
HTTP health endpoint for the TDOA node.

Exposes GET /health on a configurable port (default 8080), returning JSON
with node status information.  Runs in a daemon thread so it does not
prevent clean shutdown.

Response schema
---------------
{
  "status":               "ok" | "degraded" | "starting",
  "degraded_reasons":     ["no sync for 8.2s (threshold: 5s)"],  # only if degraded
  "node_id":              "seattle-north-01",
  "uptime_s":             123.4,
  "last_event_age_s":     0.5,      # seconds since last target event; null if none yet
  "last_sync_age_s":      0.01,     # seconds since last sync event; null if none yet
  "sync_events":          4200,
  "events_submitted":     42,
  "events_dropped":       0,
  "queue_depth":          0,
  "crystal_correction":   1.000012,
  "sync_corr_peak":       0.7042,    # latest SyncEvent quality (0-1); RDS pilot ~0.7
  "sdr_overflows":        0,
  "backlog_drains":       0,
  "sdr_mode":             "rspduo",           # present if configured
  "sample_rate_hz":       2048000.0,          # present if configured
  "sync_station":         "KISW_99.9",        # present if configured
  "sync_freq_hz":         99900000.0,         # present if configured
  "target_channels":      [{"frequency_hz": 460000000, "label": "Target 460"}]
}

"degraded" is reported when:
  - No sync events received yet (after 30 s uptime)
  - No sync event in the last 5 seconds (sync pulses arrive every ~7 ms)
  - events_dropped > 0
Target event silence is NOT a degraded condition (targets are intermittent
push-to-talk radios).
The "degraded_reasons" array explains which conditions triggered the status.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


class HealthState:
    """
    Shared mutable state updated by the pipeline and read by the health handler.
    All writes are protected by a lock.
    """

    def __init__(self, node_id: str) -> None:
        self._lock = threading.Lock()
        self.node_id: str = node_id
        self.start_time: float = time.monotonic()
        self.last_event_time: float | None = None
        self.last_sync_time: float | None = None
        self.sync_events: int = 0
        self.events_submitted: int = 0
        self.events_dropped: int = 0
        self.queue_depth: int = 0
        self.crystal_correction: float = 1.0
        self.sdr_overflows: int = 0
        self.backlog_drains: int = 0
        self.sync_station: str | None = None
        self.sync_freq_hz: float | None = None
        self.target_channels: list[dict[str, Any]] | None = None
        self.sdr_mode: str | None = None
        self.sample_rate_hz: float | None = None
        # Carrier detector state (updated from pipeline each health cycle)
        self.noise_floor_db: float | None = None
        self.onset_threshold_db: float | None = None
        self.offset_threshold_db: float | None = None
        # Sync detector quality (latest SyncEvent.corr_peak; 0-1).
        self.sync_corr_peak: float | None = None

    def record_event(self) -> None:
        with self._lock:
            self.last_event_time = time.monotonic()

    def set_config(self, **kwargs: Any) -> None:
        """Update static config info (only provided keys are changed)."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def update(
        self,
        events_submitted: int = 0,
        events_dropped: int = 0,
        queue_depth: int = 0,
        crystal_correction: float = 1.0,
        sdr_overflows: int = 0,
        backlog_drains: int = 0,
        sync_event_count: int = 0,
        noise_floor_db: float | None = None,
        onset_threshold_db: float | None = None,
        offset_threshold_db: float | None = None,
        sync_corr_peak: float | None = None,
    ) -> None:
        with self._lock:
            # Update last_sync_time if new sync events have arrived
            if sync_event_count > self.sync_events:
                self.last_sync_time = time.monotonic()
                self.sync_events = sync_event_count
            self.events_submitted = events_submitted
            self.events_dropped = events_dropped
            self.queue_depth = queue_depth
            self.crystal_correction = crystal_correction
            self.sdr_overflows = sdr_overflows
            self.backlog_drains = backlog_drains
            if noise_floor_db is not None:
                self.noise_floor_db = noise_floor_db
            if onset_threshold_db is not None:
                self.onset_threshold_db = onset_threshold_db
            if offset_threshold_db is not None:
                self.offset_threshold_db = offset_threshold_db
            if sync_corr_peak is not None:
                self.sync_corr_peak = sync_corr_peak

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            uptime = now - self.start_time
            last_event_age = (now - self.last_event_time) if self.last_event_time is not None else None
            last_sync_age = (now - self.last_sync_time) if self.last_sync_time is not None else None

            reasons: list[str] = []
            if uptime < 30.0:
                status = "starting"
            else:
                # Sync loss is the real health indicator - target activity
                # is intermittent by nature (push-to-talk, etc.)
                if last_sync_age is None:
                    reasons.append("no sync events received yet")
                elif last_sync_age > 5.0:
                    reasons.append(f"no sync for {last_sync_age:.1f}s (threshold: 5s)")
                if self.events_dropped > 0:
                    reasons.append(f"{self.events_dropped} events dropped")
                status = "degraded" if reasons else "ok"

            result: dict[str, Any] = {
                "status": status,
                "node_id": self.node_id,
                "uptime_s": round(uptime, 1),
                "last_event_age_s": round(last_event_age, 2) if last_event_age is not None else None,
                "last_sync_age_s": round(last_sync_age, 2) if last_sync_age is not None else None,
                "sync_events": self.sync_events,
                "events_submitted": self.events_submitted,
                "events_dropped": self.events_dropped,
                "queue_depth": self.queue_depth,
                "crystal_correction": round(self.crystal_correction, 8),
                "sdr_overflows": self.sdr_overflows,
                "backlog_drains": self.backlog_drains,
            }
            if reasons:
                result["degraded_reasons"] = reasons
            if self.sdr_mode is not None:
                result["sdr_mode"] = self.sdr_mode
            if self.sample_rate_hz is not None:
                result["sample_rate_hz"] = self.sample_rate_hz
            if self.sync_station is not None:
                result["sync_station"] = self.sync_station
            if self.sync_freq_hz is not None:
                result["sync_freq_hz"] = self.sync_freq_hz
            if self.target_channels is not None:
                result["target_channels"] = self.target_channels
            if self.noise_floor_db is not None:
                result["noise_floor_db"] = round(self.noise_floor_db, 1)
            if self.onset_threshold_db is not None:
                result["onset_threshold_db"] = round(self.onset_threshold_db, 1)
            if self.offset_threshold_db is not None:
                result["offset_threshold_db"] = round(self.offset_threshold_db, 1)
            if self.sync_corr_peak is not None:
                result["sync_corr_peak"] = round(self.sync_corr_peak, 4)
            return result


def _make_handler(state: HealthState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(state.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            # Suppress default request logging; use our logger at DEBUG
            logger.debug("health request: " + fmt, *args)

    return _Handler


class HealthServer:
    """
    Tiny HTTP server exposing /health.

    Parameters
    ----------
    state : HealthState
        Shared state object updated by the main pipeline loop.
    port : int
        TCP port to listen on (default 8080).
    host : str
        Bind address (default "0.0.0.0").
    """

    def __init__(self, state: HealthState, port: int = 8080, host: str = "0.0.0.0") -> None:
        self._state = state
        self._port = port
        self._host = host
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self._state)
        try:
            self._server = HTTPServer((self._host, self._port), handler)
        except OSError as exc:
            raise OSError(
                f"Health server could not bind to port {self._port}: {exc}. "
                f"Is another node instance already running? "
                f"Kill it first (kill $(lsof -ti :{self._port})) "
                f"or change health_port in your config."
            ) from exc
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="health-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Health server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
