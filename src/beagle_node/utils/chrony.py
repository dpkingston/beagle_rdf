# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Chrony (NTP daemon) status reader.

Parses `chronyc tracking` output to extract:
  - clock reference source (GPS / NTP / unknown)
  - RMS offset uncertainty in nanoseconds

Used to populate `clock_source` and `clock_uncertainty_ns` in CarrierEvents,
giving the server a measure of absolute timestamp quality.

If chronyc is not installed or fails (e.g. no chrony daemon running) the
function returns a safe fallback with source="unknown" and uncertainty 0 so
the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

ClockSource = Literal["gps_1pps", "ntp", "unknown"]

# Regex patterns for chronyc tracking fields we care about
_RE_REFERENCE = re.compile(r"^Reference ID\s*:\s*\S+\s+\((.+?)\)", re.MULTILINE)
_RE_RMS       = re.compile(r"^RMS offset\s*:\s*([\d.e+\-]+)\s+seconds", re.MULTILINE)
_RE_STRATUM   = re.compile(r"^Stratum\s*:\s*(\d+)", re.MULTILINE)


@dataclass
class ChronyStatus:
    """Parsed output of `chronyc tracking`."""

    source: ClockSource = "unknown"
    """Classified clock reference: 'gps_1pps', 'ntp', or 'unknown'."""

    rms_offset_ns: int = 0
    """RMS offset from the reference, in nanoseconds."""

    stratum: int = 16
    """NTP stratum (1 = GPS/hardware clock, 2+ = NTP, 16 = unsynchronised)."""

    ref_label: str = ""
    """Raw reference label string, e.g. 'GPS' or '169.254.169.254'."""


def read_chrony_status(timeout_s: float = 3.0) -> ChronyStatus:
    """
    Run `chronyc tracking` and parse the result.

    Returns a ChronyStatus with source="unknown" and rms_offset_ns=0 if
    chronyc is unavailable, the daemon is not running, or parsing fails.
    """
    try:
        result = subprocess.run(
            ["chronyc", "tracking"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        logger.debug("chronyc not found; clock source unknown")
        return ChronyStatus()
    except subprocess.TimeoutExpired:
        logger.warning("chronyc tracking timed out after %.1f s", timeout_s)
        return ChronyStatus()
    except Exception as exc:
        logger.warning("chronyc tracking failed: %s", exc)
        return ChronyStatus()

    if result.returncode != 0:
        logger.debug("chronyc tracking returned %d; clock source unknown", result.returncode)
        return ChronyStatus()

    return _parse_tracking(result.stdout)


def _parse_tracking(text: str) -> ChronyStatus:
    """Parse the text output of `chronyc tracking`."""
    status = ChronyStatus()

    # Stratum
    m = _RE_STRATUM.search(text)
    if m:
        status.stratum = int(m.group(1))

    # Reference label
    m = _RE_REFERENCE.search(text)
    if m:
        status.ref_label = m.group(1).strip()
        status.source = _classify_source(status.ref_label, status.stratum)

    # RMS offset -> nanoseconds
    m = _RE_RMS.search(text)
    if m:
        rms_s = float(m.group(1))
        status.rms_offset_ns = int(rms_s * 1_000_000_000)

    return status


def _classify_source(ref_label: str, stratum: int) -> ClockSource:
    """
    Classify the clock reference from the label string.

    GPS-disciplined sources (stratum 1) with 'GPS', 'PPS', or 'GNSS' in
    the label are classified as 'gps_1pps'.  All other synchronised clocks
    are 'ntp'.  Stratum 16 (unsynchronised) is 'unknown'.
    """
    if stratum == 16:
        return "unknown"
    upper = ref_label.upper()
    if stratum == 1 and any(kw in upper for kw in ("GPS", "PPS", "GNSS", "NMEA")):
        return "gps_1pps"
    return "ntp"
