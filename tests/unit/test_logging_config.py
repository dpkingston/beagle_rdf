# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for utils/logging.py."""

from __future__ import annotations

import logging
import sys

import structlog

from beagle_node.utils.logging import configure_logging


def test_configure_logging_does_not_raise():
    configure_logging("test-node", level="DEBUG", json_output=False)


def test_configure_logging_json_does_not_raise():
    configure_logging("test-node", level="INFO", json_output=True)


def test_log_level_respected(capsys):
    """DEBUG messages should not appear when level=INFO."""
    configure_logging("test-node", level="INFO", json_output=False)
    log = structlog.get_logger("test")
    log.debug("this should not appear")
    log.info("this should appear")
    captured = capsys.readouterr()
    assert "this should not appear" not in captured.err
    assert "this should appear" in captured.err


def test_json_output_emits_to_stderr(capsys):
    """With json_output=True, log output should appear on stderr."""
    configure_logging("test-node", level="DEBUG", json_output=True)
    log = structlog.get_logger("test")
    log.info("hello json")
    captured = capsys.readouterr()
    assert "hello json" in captured.err


def test_node_id_bound(capsys):
    """node_id should appear in the log output."""
    configure_logging("seattle-north-01", level="INFO", json_output=False)
    log = structlog.get_logger("test")
    log.info("check node id")
    captured = capsys.readouterr()
    assert "seattle-north-01" in captured.err
