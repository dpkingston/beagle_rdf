# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Node software version, captured at import time.

Reads the git short SHA from the source tree so the server can tell
which code revision each node is running.  Falls back to "unknown"
if git is unavailable (e.g. installed via pip without git history).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_VERSION_BASE = "0.2.0"


def _git_sha() -> str:
    """Return the short git SHA of the source tree, or empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _build_version() -> str:
    sha = _git_sha()
    if sha:
        return f"{_VERSION_BASE}+{sha}"
    return _VERSION_BASE


VERSION: str = _build_version()
