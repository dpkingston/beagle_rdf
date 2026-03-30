# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Beagle aggregation server entry point.

Usage
-----
    beagle-server --config config/server.yaml
    python -m beagle_server --config config/server.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from beagle_server.api import create_app
from beagle_server.config import ServerFullConfig, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="beagle-server",
        description="Beagle aggregation server - receives events, computes fixes.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        required=True,
        help="Path to YAML configuration file (see config/server.example.yaml)",
    )
    parser.add_argument(
        "--host",
        metavar="HOST",
        default=None,
        help="Override server.host from config",
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=None,
        help="Override server.port from config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default INFO)",
    )
    args = parser.parse_args(argv)

    try:
        config: ServerFullConfig = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    host = args.host or config.server.host
    port = args.port or config.server.port

    # Configure the application logger before uvicorn starts.
    # Uvicorn's dictConfig leaves the root logger at WARNING, which would
    # silence our _logger.info() calls in api.py.  Configuring the package
    # logger explicitly (propagate=False) avoids duplicate output.
    _app_log = logging.getLogger("beagle_server")
    _app_log.setLevel(args.log_level)
    if not _app_log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s"))
        _app_log.addHandler(_h)
        _app_log.propagate = False

    # Suppress high-frequency heartbeat endpoints from uvicorn's access log at
    # INFO level; still visible when --log-level DEBUG is passed.
    # Note: uvicorn's access handler has level=NOTSET so we cannot simply demote
    # the record - we must suppress it outright at INFO and allow it only at DEBUG.
    _show_debug_access = args.log_level.upper() == "DEBUG"

    class _HeartbeatAccessFilter(logging.Filter):
        _SUPPRESS = ("/api/v1/heartbeat",)

        def filter(self, record: logging.LogRecord) -> bool:
            if not _show_debug_access:
                msg = record.getMessage()
                if any(path in msg for path in self._SUPPRESS):
                    return False
            return True

    logging.getLogger("uvicorn.access").addFilter(_HeartbeatAccessFilter())

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level.lower(),
        # Allow 2 s for open connections (SSE streams) to drain on first
        # Ctrl+C before force-closing them.  Without this, a second Ctrl+C
        # is required and produces a noisy CancelledError traceback.
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
