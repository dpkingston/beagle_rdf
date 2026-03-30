# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Structured logging configuration using structlog.

Two output modes:
  - dev  : human-readable coloured console output (default when stderr is a tty)
  - json : newline-delimited JSON (for log aggregation in production)

Usage
-----
Call configure_logging() once at startup before any log messages are emitted.
The node_id is bound into every log record automatically.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    node_id: str,
    level: str = "INFO",
    json_output: bool | None = None,
) -> None:
    """
    Configure structlog and the stdlib root logger.

    Parameters
    ----------
    node_id : str
        Bound into every log record as the ``node_id`` key.
    level : str
        Log level name: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    json_output : bool | None
        If None, auto-detect: JSON when stderr is not a tty,
        human-readable when it is.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_output is None:
        json_output = not sys.stderr.isatty()

    # Shared processors applied to every log record
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # Production: newline-delimited JSON
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        # Development: coloured console
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            foreign_pre_chain=shared_processors,
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so that third-party libraries (httpx, etc.) are
    # captured and formatted consistently.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Silence httpx/httpcore request-level INFO lines (e.g. "HTTP Request:
    # POST ... 201 Created") - they add no operational value and clutter logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Bind node_id into every structlog record for this process
    structlog.contextvars.bind_contextvars(node_id=node_id)
