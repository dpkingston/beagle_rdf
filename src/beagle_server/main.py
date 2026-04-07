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

    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    # Goal: every log line - ours, uvicorn's, and any third-party library's
    # - is prefixed with a timestamp and the logger name.  We achieve this
    # by:
    #   1. Configuring the root logger with a StreamHandler + formatter so
    #      anything that propagates (third-party libs) inherits the format.
    #   2. Passing the same formatter to uvicorn via its dictConfig log_config
    #      argument so uvicorn.error and uvicorn.access also get it.
    #
    # Without (2), uvicorn replaces the root config when it starts and its
    # own handlers use a format with no timestamp -- the symptom users see
    # as "some log lines lack timestamps".
    _LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    _LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

    # Root logger: catches third-party libraries (httpx, sqlite, etc.).
    _root_handler = logging.StreamHandler()
    _root_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
    _root = logging.getLogger()
    # Replace any existing handlers so we don't double-print after a reload.
    for _h in list(_root.handlers):
        _root.removeHandler(_h)
    _root.addHandler(_root_handler)
    _root.setLevel(args.log_level)

    # Our package logger inherits the root handler via propagation.  We only
    # need to set its level explicitly so DEBUG calls in beagle_server.* are
    # visible when --log-level DEBUG is passed.
    logging.getLogger("beagle_server").setLevel(args.log_level)

    # Suppress high-frequency heartbeat endpoints from uvicorn's access log at
    # INFO level; still visible when --log-level DEBUG is passed.
    _show_debug_access = args.log_level.upper() == "DEBUG"

    class _HeartbeatAccessFilter(logging.Filter):
        _SUPPRESS = ("/api/v1/heartbeat",)

        def filter(self, record: logging.LogRecord) -> bool:
            if not _show_debug_access:
                msg = record.getMessage()
                if any(path in msg for path in self._SUPPRESS):
                    return False
            return True

    # uvicorn's dictConfig: route all uvicorn loggers through our formatter.
    # disable_existing_loggers=False so the root config above survives the
    # uvicorn import.
    _uvicorn_log_config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "beagle": {
                "format": _LOG_FORMAT,
                "datefmt": _LOG_DATEFMT,
            },
        },
        "filters": {
            "heartbeat_filter": {
                "()": _HeartbeatAccessFilter,
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "beagle",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "beagle",
                "stream": "ext://sys.stdout",
                "filters": ["heartbeat_filter"],
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": args.log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": args.log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": args.log_level,
                "propagate": False,
            },
        },
    }

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level.lower(),
        log_config=_uvicorn_log_config,
        # When fronted by a reverse proxy on localhost (e.g. Apache or
        # nginx terminating TLS), trust the X-Forwarded-Proto / -For
        # headers so absolute URLs (OAuth callbacks, redirects) are
        # generated with the original scheme and client IP rather than
        # the plain HTTP localhost connection from the proxy.
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        # Allow 2 s for open connections (SSE streams) to drain on first
        # Ctrl+C before force-closing them.  Without this, a second Ctrl+C
        # is required and produces a noisy CancelledError traceback.
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
