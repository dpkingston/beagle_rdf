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
    """Return the short git SHA of the source tree, or empty string.

    Reads .git/HEAD directly to avoid the "dubious ownership" error that
    occurs when the service user (e.g. beagle) differs from the repo
    owner (e.g. dpk).  Falls back to subprocess if the file isn't found.
    """
    # Walk up from this file to find the .git directory
    d = Path(__file__).resolve().parent
    for _ in range(10):
        git_dir = d / ".git"
        if git_dir.is_dir():
            try:
                head = (git_dir / "HEAD").read_text().strip()
                if head.startswith("ref: "):
                    ref_path = git_dir / head[5:]
                    if ref_path.exists():
                        return ref_path.read_text().strip()[:7]
                elif len(head) >= 7:
                    # Detached HEAD — raw sha
                    return head[:7]
            except OSError:
                break
        if d.parent == d:
            break
        d = d.parent

    # Fallback: try git subprocess (works if same user owns the repo)
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
