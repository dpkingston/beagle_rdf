# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Integration test for main.py - runs the node in --mock mode.

Uses a short mock duration and a mock HTTP server so no real SDR or
network is needed.
"""

from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from beagle_node.main import run

_EXAMPLE_CONFIG = str(
    Path(__file__).parent.parent.parent / "config" / "node.example.yaml"
)

# ---------------------------------------------------------------------------
# Minimal mock HTTP server to absorb reporter POSTs
# ---------------------------------------------------------------------------

class _RecordingHandler(BaseHTTPRequestHandler):
    received: list[bytes] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _RecordingHandler.received.append(body)
        self.send_response(201)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass   # silence


@pytest.fixture(scope="module")
def mock_server():
    """Start a local HTTP server that accepts reporter POSTs."""
    _RecordingHandler.received.clear()
    srv = HTTPServer(("127.0.0.1", 19100), _RecordingHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        config=_EXAMPLE_CONFIG,
        mock=True,
        mock_duration=1.0,       # 1 second of synthetic IQ
        log_level="WARNING",     # quiet in tests
        json_logs=False,
        no_health=True,          # don't bind a port in tests
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_mock_run_exits_cleanly(mock_server):
    """Node should start, process mock IQ, and return 0."""
    args = _make_args()
    # Point reporter at our mock server
    import beagle_node.events.reporter as rep_mod
    original_init = rep_mod.EventReporter.__init__

    exit_code = run(args)
    assert exit_code == 0


def test_mock_run_with_missing_config():
    """Missing config file should return exit code 1."""
    args = _make_args(config="/nonexistent/path/node.yaml")
    exit_code = run(args)
    assert exit_code == 1


def test_main_imports():
    """All Sprint 3 modules should import without error."""
    import beagle_node.main
    import beagle_node.events.reporter
    import beagle_node.utils.logging
    import beagle_node.utils.health
