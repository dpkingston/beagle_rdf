# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for utils/chrony.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import subprocess

import pytest

from beagle_node.utils.chrony import ChronyStatus, _parse_tracking, read_chrony_status


# ---------------------------------------------------------------------------
# Sample chronyc tracking outputs
# ---------------------------------------------------------------------------

_GPS_OUTPUT = """\
Reference ID    : 47505300 (GPS)
Stratum         : 1
Ref time (UTC)  : Mon Feb 24 02:00:00 2025
System time     : 0.000000123 seconds fast of NTP time
Last offset     : +0.000000045 seconds
RMS offset      : 0.000000456 seconds
Frequency       : 0.123 ppm slow
Residual freq   : +0.001 ppm
Skew            : 0.012 ppm
Root delay      : 0.000010000 seconds
Root dispersion : 0.000001234 seconds
Update interval : 16.0 seconds
Leap status     : Normal
"""

_NTP_OUTPUT = """\
Reference ID    : C0A80101 (192.168.1.1)
Stratum         : 3
Ref time (UTC)  : Mon Feb 24 02:00:00 2025
System time     : 0.000012345 seconds slow of NTP time
Last offset     : +0.000011234 seconds
RMS offset      : 0.000023456 seconds
Frequency       : 1.234 ppm slow
Residual freq   : +0.123 ppm
Skew            : 0.456 ppm
Root delay      : 0.012345678 seconds
Root dispersion : 0.000123456 seconds
Update interval : 64.4 seconds
Leap status     : Normal
"""

_PPS_OUTPUT = """\
Reference ID    : 50505300 (PPS0)
Stratum         : 1
Ref time (UTC)  : Mon Feb 24 02:00:00 2025
System time     : 0.000000050 seconds fast of NTP time
Last offset     : +0.000000020 seconds
RMS offset      : 0.000000100 seconds
Frequency       : 0.010 ppm slow
Residual freq   : +0.000 ppm
Skew            : 0.001 ppm
Root delay      : 0.000001000 seconds
Root dispersion : 0.000000500 seconds
Update interval : 16.0 seconds
Leap status     : Normal
"""

_UNSYNC_OUTPUT = """\
Reference ID    : 00000000 ()
Stratum         : 16
Ref time (UTC)  : Thu Jan  1 00:00:00 1970
System time     : 0.000000000 seconds slow of NTP time
Last offset     : +0.000000000 seconds
RMS offset      : 0.000000000 seconds
Frequency       : 0.000 ppm slow
Residual freq   : +0.000 ppm
Skew            : 0.000 ppm
Root delay      : 0.000000000 seconds
Root dispersion : 0.000000000 seconds
Update interval : 0.0 seconds
Leap status     : Not synchronised
"""


# ---------------------------------------------------------------------------
# _parse_tracking
# ---------------------------------------------------------------------------

class TestParseTracking:

    def test_gps_source_classified(self):
        s = _parse_tracking(_GPS_OUTPUT)
        assert s.source == "gps_1pps"

    def test_pps_source_classified(self):
        s = _parse_tracking(_PPS_OUTPUT)
        assert s.source == "gps_1pps"

    def test_ntp_source_classified(self):
        s = _parse_tracking(_NTP_OUTPUT)
        assert s.source == "ntp"

    def test_unsync_source_classified(self):
        s = _parse_tracking(_UNSYNC_OUTPUT)
        assert s.source == "unknown"

    def test_gps_stratum(self):
        s = _parse_tracking(_GPS_OUTPUT)
        assert s.stratum == 1

    def test_ntp_stratum(self):
        s = _parse_tracking(_NTP_OUTPUT)
        assert s.stratum == 3

    def test_gps_rms_offset_ns(self):
        s = _parse_tracking(_GPS_OUTPUT)
        # 0.000000456 seconds -> 456 ns
        assert s.rms_offset_ns == 456

    def test_ntp_rms_offset_ns(self):
        s = _parse_tracking(_NTP_OUTPUT)
        # 0.000023456 seconds -> 23456 ns
        assert s.rms_offset_ns == 23456

    def test_pps_rms_offset_ns(self):
        s = _parse_tracking(_PPS_OUTPUT)
        # 0.000000100 seconds -> 100 ns
        assert s.rms_offset_ns == 100

    def test_gps_ref_label(self):
        s = _parse_tracking(_GPS_OUTPUT)
        assert s.ref_label == "GPS"

    def test_ntp_ref_label(self):
        s = _parse_tracking(_NTP_OUTPUT)
        assert "192.168.1.1" in s.ref_label

    def test_unsync_rms_zero(self):
        # RMS offset line exists but is 0.000000000
        s = _parse_tracking(_UNSYNC_OUTPUT)
        assert s.rms_offset_ns == 0
        assert s.source == "unknown"


# ---------------------------------------------------------------------------
# read_chrony_status
# ---------------------------------------------------------------------------

class TestReadChronyStatus:

    @patch("beagle_node.utils.chrony.subprocess.run")
    def test_returns_gps_status(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_GPS_OUTPUT
        )
        s = read_chrony_status()
        assert s.source == "gps_1pps"
        assert s.rms_offset_ns == 456

    @patch("beagle_node.utils.chrony.subprocess.run")
    def test_nonzero_returncode_gives_unknown(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        s = read_chrony_status()
        assert s.source == "unknown"
        assert s.rms_offset_ns == 0

    @patch("beagle_node.utils.chrony.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_chronyc_gives_unknown(self, mock_run):
        s = read_chrony_status()
        assert s.source == "unknown"
        assert s.rms_offset_ns == 0

    @patch("beagle_node.utils.chrony.subprocess.run",
           side_effect=subprocess.TimeoutExpired(cmd="chronyc", timeout=3))
    def test_timeout_gives_unknown(self, mock_run):
        s = read_chrony_status()
        assert s.source == "unknown"

    @patch("beagle_node.utils.chrony.subprocess.run")
    def test_ntp_status(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_NTP_OUTPUT
        )
        s = read_chrony_status()
        assert s.source == "ntp"
        assert s.rms_offset_ns == 23_456
