# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""Unit tests for sdr/rspduo.py."""

from __future__ import annotations

import itertools
import sys
import time as _time_module
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal SoapySDR stub - injected before importing rspduo
# ---------------------------------------------------------------------------

def _make_soapy_stub():
    """Return a minimal SoapySDR module stub."""
    stub = ModuleType("SoapySDR")
    stub.SOAPY_SDR_RX = 0
    stub.SOAPY_SDR_CF32 = "CF32"
    stub.SOAPY_SDR_OVERFLOW = -4   # return code, NOT a flag bit
    stub.SOAPY_SDR_TIMEOUT = -1
    stub.SOAPY_SDR_HAS_TIME = 4
    # Log level constants (values match the real SoapySDR)
    stub.SOAPY_SDR_FATAL    = 1
    stub.SOAPY_SDR_CRITICAL = 2
    stub.SOAPY_SDR_ERROR    = 3
    stub.SOAPY_SDR_WARNING  = 4
    stub.SOAPY_SDR_NOTICE   = 5
    stub.SOAPY_SDR_INFO     = 6
    stub.SOAPY_SDR_DEBUG    = 7
    stub.SOAPY_SDR_TRACE    = 8
    stub.SOAPY_SDR_SSI      = 9
    stub.registerLogHandler = lambda handler: None

    class _StreamResult:
        def __init__(self, ret, flags=0, timeNs=0):
            self.ret = ret
            self.flags = flags
            self.timeNs = timeNs

    stub._StreamResult = _StreamResult

    def _make_mock_dev(args):
        dev = MagicMock()
        dev.getSampleRate.return_value = 2_000_000.0
        dev.getFrequency.return_value = 99_900_000.0
        dev.getGain.return_value = 30.0
        dev.setupStream.return_value = object()  # opaque stream handle

        # readStream fills all buffers (one buffer per single-channel stream)
        def read_stream(stream, buffers, num_samples, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
            return _StreamResult(num_samples)

        dev.readStream.side_effect = read_stream
        return dev

    # Device must be a class (not a bare function) so Device(args) works.
    class _FakeDevice:
        def __new__(cls, args):  # Device(kwargs) -> mock device
            return _make_mock_dev(args)

    stub.Device = _FakeDevice
    return stub


@pytest.fixture(autouse=True)
def inject_soapy_stub(monkeypatch):
    """Inject the SoapySDR stub so rspduo can be imported without the real lib."""
    stub = _make_soapy_stub()
    monkeypatch.setitem(sys.modules, "SoapySDR", stub)
    # Force re-import so _SOAPY_AVAILABLE is recalculated with stub present
    if "beagle_node.sdr.rspduo" in sys.modules:
        del sys.modules["beagle_node.sdr.rspduo"]

    # The mock readStream returns instantly (microseconds).  Backlog detection
    # compares the sync readStream duration against drain_thresh_ns
    # (buffer_size/rate/2).  Without this patch every test would trigger the
    # drain loop because the mock completes in << any realistic threshold.
    # Patch monotonic_ns to return a large delta (100 ms per call) so the
    # drain logic sees "real-time" buffers.  TestBacklogDrain overrides this.
    _calls = [0]

    def _big_delta_monotonic():
        c = _calls[0]
        _calls[0] += 1
        return c * 100_000_000  # 100 ms per call -> always > threshold

    monkeypatch.setattr(_time_module, "monotonic_ns", _big_delta_monotonic)

    yield stub


def _make_receiver(**kwargs):
    from beagle_node.sdr.rspduo import RSPduoReceiver
    defaults = dict(
        sync_frequency_hz=99_900_000.0,
        target_frequency_hz=155_100_000.0,
        sample_rate_hz=2_000_000.0,
        sync_gain_db="auto",
        target_gain_db="auto",
        master_device_args="driver=sdrplay",
        slave_device_args=None,
        buffer_size=256,
    )
    defaults.update(kwargs)
    return RSPduoReceiver(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_succeeds():
    rx = _make_receiver()
    assert rx.config.sample_rate_hz == 2_000_000.0
    assert not rx._is_open


def test_master_args_stored():
    rx = _make_receiver(master_device_args="driver=sdrplay,serial=X")
    assert rx._master_args == "driver=sdrplay,serial=X"


def test_overflow_count_starts_zero():
    rx = _make_receiver()
    assert rx.overflow_count == 0


# ---------------------------------------------------------------------------
# open / close
# ---------------------------------------------------------------------------

def test_open_sets_is_open():
    rx = _make_receiver()
    rx.open()
    assert rx._is_open


def test_double_open_is_idempotent():
    rx = _make_receiver()
    rx.open()
    rx.open()   # second call should be a no-op
    assert rx._is_open


def test_close_clears_is_open():
    rx = _make_receiver()
    rx.open()
    rx.close()
    assert not rx._is_open


def test_double_close_is_idempotent():
    rx = _make_receiver()
    rx.open()
    rx.close()
    rx.close()  # should not raise
    assert not rx._is_open


def test_context_manager():
    rx = _make_receiver()
    with rx:
        assert rx._is_open
    assert not rx._is_open


# ---------------------------------------------------------------------------
# paired_stream
# ---------------------------------------------------------------------------

def test_paired_stream_yields_tuples():
    rx = _make_receiver(buffer_size=64)
    it = rx.paired_stream()
    sync_buf, target_buf, buf_wall_ns = next(it)
    assert isinstance(sync_buf, np.ndarray)
    assert isinstance(target_buf, np.ndarray)
    assert sync_buf.dtype == np.complex64
    assert target_buf.dtype == np.complex64
    assert isinstance(buf_wall_ns, int)


def test_paired_stream_yields_correct_size():
    rx = _make_receiver(buffer_size=128)
    it = rx.paired_stream()
    sync_buf, target_buf, _ = next(it)
    assert len(sync_buf) == 128
    assert len(target_buf) == 128


def test_paired_stream_opens_device_if_not_open():
    rx = _make_receiver(buffer_size=64)
    assert not rx._is_open
    it = rx.paired_stream()
    next(it)
    assert rx._is_open


def test_paired_stream_yields_copies(inject_soapy_stub):
    """Buffers yielded must be copies so internal arrays can be reused."""
    rx = _make_receiver(buffer_size=64)
    it = rx.paired_stream()
    a, b, _ = next(it)
    c, d, _ = next(it)
    # Different objects each iteration
    assert a is not c
    assert b is not d


def test_paired_stream_stops_on_read_error(inject_soapy_stub):
    """If readStream returns a non-timeout negative ret, iteration ends."""
    stub = inject_soapy_stub
    sync_calls = [0]

    def read_sync_fail(stream, buffers, num_samples, timeoutUs=1_000_000):
        sync_calls[0] += 1
        if sync_calls[0] > 1:
            return stub._StreamResult(-2)  # -2 = stream error, not timeout
        buffers[0][:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    def read_tgt_ok(stream, buffers, num_samples, timeoutUs=1_000_000):
        buffers[0][:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx.open()
    sync_h = rx._sync_stream

    def dispatch(stream, buffers, num_samples, timeoutUs=1_000_000):
        if stream is sync_h:
            return read_sync_fail(stream, buffers, num_samples, timeoutUs)
        return read_tgt_ok(stream, buffers, num_samples, timeoutUs)

    rx._dev.readStream.side_effect = dispatch

    results = list(rx.paired_stream())
    assert len(results) <= 1   # at most one good pair before error stops iteration


def test_paired_stream_retries_on_timeout(inject_soapy_stub):
    """A SOAPY_SDR_TIMEOUT (-1) on one stream is retried, not fatal."""
    stub = inject_soapy_stub
    sync_calls = [0]

    def read_sync_timeout_once(stream, buffers, num_samples, timeoutUs=1_000_000):
        sync_calls[0] += 1
        if sync_calls[0] == 2:
            return stub._StreamResult(-1)  # timeout on second call only
        buffers[0][:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    def read_tgt_ok(stream, buffers, num_samples, timeoutUs=1_000_000):
        buffers[0][:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx.open()
    sync_h = rx._sync_stream

    def dispatch(stream, buffers, num_samples, timeoutUs=1_000_000):
        if stream is sync_h:
            return read_sync_timeout_once(stream, buffers, num_samples, timeoutUs)
        return read_tgt_ok(stream, buffers, num_samples, timeoutUs)

    rx._dev.readStream.side_effect = dispatch

    # Collect 3 pairs: call 1 OK, call 2 timeout (retry), call 3 OK, call 4 OK
    it = rx.paired_stream()
    results = [next(it), next(it), next(it)]
    assert len(results) == 3   # timeout was skipped, stream continued


def test_timeout_rolling_window_reopens_on_storm(inject_soapy_stub, monkeypatch):
    """_TIMEOUT_WINDOW_MAX timeouts within _TIMEOUT_WINDOW_SECS triggers close+reopen.

    After reopen the stream recovers: the new device (created by open()) uses
    the default OK side_effect, so paired_stream() continues yielding.
    The previous consecutive counter would reset on every intermittent success;
    the rolling window fires regardless.
    """
    stub = inject_soapy_stub
    call_count = [0]

    def alternating(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        # Alternate: timeout, ok - simulates sync/target-swap storm.
        if call_count[0] % 2 == 1:
            return stub._StreamResult(-1)  # timeout
        for buf in buffers:
            buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5   # lower threshold for fast test
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx.open()
    rx._dev.readStream.side_effect = alternating
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)

    it = rx.paired_stream()
    # The storm triggers reopen; the new device (from stub.Device()) has default
    # OK reads, so the generator recovers and yields normally after restart.
    results = [next(it), next(it)]
    assert len(results) == 2


def test_timeout_rolling_window_retries_reopen_until_success(inject_soapy_stub, monkeypatch):
    """After a timeout storm, reopen is retried until it succeeds - not fatal."""
    stub = inject_soapy_stub

    def all_timeout(stream, buffers, num_samples, timeoutUs=1_000_000):
        return stub._StreamResult(-1)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx.open()
    rx._dev.readStream.side_effect = all_timeout
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)

    open_calls = [0]
    original_open = rx.open.__func__

    def open_fail_once(self=rx):
        open_calls[0] += 1
        if open_calls[0] < 3:
            raise RuntimeError("device unavailable")
        original_open(self)

    monkeypatch.setattr(rx, "open", open_fail_once)
    monkeypatch.setattr(_time_module, "sleep", lambda s: None)

    it = rx.paired_stream()
    result = next(it)
    assert result is not None
    assert open_calls[0] == 3


def test_timeout_single_does_not_exit(inject_soapy_stub, monkeypatch):
    """A single timeout within the window retries and continues - not fatal."""
    stub = inject_soapy_stub
    call_count = [0]

    def timeout_once(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        if call_count[0] == 1:
            return stub._StreamResult(-1)
        for buf in buffers:
            buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx.open()
    rx._dev.readStream.side_effect = timeout_once
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)

    it = rx.paired_stream()
    results = [next(it), next(it)]   # retry after timeout, then two successes
    assert len(results) == 2


def test_timeout_window_expiry_resets_count(inject_soapy_stub, monkeypatch):
    """Timeouts older than _TIMEOUT_WINDOW_SECS are discarded; old entries don't cause exit."""
    stub = inject_soapy_stub
    mono_time = [0.0]

    call_count = [0]

    def timeout_then_ok(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        if call_count[0] <= 4:
            # 4 timeouts at t=0 - just under the max of 5
            return stub._StreamResult(-1)
        # Advance time past the window before the 5th call
        mono_time[0] += 61.0
        for buf in buffers:
            buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx.open()
    rx._dev.readStream.side_effect = timeout_then_ok
    monkeypatch.setattr(_time_module, "monotonic", lambda: mono_time[0])

    it = rx.paired_stream()
    # Should survive: 4 timeouts at t=0, then time jumps past window, then 2 successes
    results = [next(it), next(it)]
    assert len(results) == 2


def test_timeout_first_logs_info_second_logs_warning(
    inject_soapy_stub, monkeypatch, caplog
):
    """First TIMEOUT in the rolling window logs at INFO; second logs at WARNING.

    Each TIMEOUT event is one trip through the handler (sync=-1, tgt=ok, or
    vice versa).  We produce two separate events so the level escalates.
    """
    stub = inject_soapy_stub
    call_count = [0]

    def sync_timeouts_twice(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        c = call_count[0]
        # Calls 1 and 3 are sync reads (the blocking calls) -> timeout.
        # Calls 2 and 4 are tgt reads (always ok) -> success.
        # Calls 5+ are both ok -> yield.
        if c in (1, 3):
            return stub._StreamResult(-1)
        for buf in buffers:
            buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 10
    rx._TIMEOUT_WINDOW_SECS = 120.0
    rx.open()
    rx._dev.readStream.side_effect = sync_timeouts_twice
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)

    import logging
    with caplog.at_level(logging.DEBUG, logger="beagle_node.sdr.rspduo"):
        it = rx.paired_stream()
        next(it)   # two sync timeouts, then success

    timeout_records = [
        r for r in caplog.records
        if "TIMEOUT reached error handler" in r.message
    ]
    assert len(timeout_records) == 2, f"Expected 2 TIMEOUT records, got {len(timeout_records)}"
    assert timeout_records[0].levelno == logging.INFO, "First TIMEOUT should be INFO"
    assert timeout_records[1].levelno == logging.WARNING, "Second TIMEOUT should be WARNING"


def test_instant_timeout_after_reopen_does_not_spinloop(inject_soapy_stub, monkeypatch):
    """Regression: instant TIMEOUT (ret=-1, read_ms~0, HAS_TIME=False) after reopen
    must not bypass the error handler via the stale-buffer drain path.

    Scenario: sdrplay_api service is redeployed; after paired_stream() triggers a
    close/reopen the new driver returns TIMEOUT in ~0 ms (no HAS_TIME flag).
    The old code classified this as stale (fast return < threshold) and looped via
    'continue', never reaching the timeout counter, causing a 300% CPU spinloop.
    The fix guards the stale check with 'sr_sync.ret > 0'.
    """
    stub = inject_soapy_stub
    call_count = [0]

    def always_timeout(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        return stub._StreamResult(-1)  # always timeout, instant return

    # Replace the Device factory so ALL devices (before and after reopen) always
    # return timeout - simulating a persistently broken sdrplay_api service.
    class _AlwaysTimeoutDevice:
        def __new__(cls, args):
            dev = MagicMock()
            dev.getSampleRate.return_value = 2_000_000.0
            dev.getFrequency.return_value = 99_900_000.0
            dev.getGain.return_value = 30.0
            dev.setupStream.return_value = object()
            dev.readStream.side_effect = always_timeout
            return dev

    monkeypatch.setattr(stub, "Device", _AlwaysTimeoutDevice)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx._MAX_RESTARTS_WITHOUT_RECOVERY = 1   # bail after 1 unproductive reopen
    rx._REOPEN_DELAY_S = 0.0
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    rx.open()

    it = rx.paired_stream()
    # Must terminate (StopIteration) rather than spin forever.
    with pytest.raises(StopIteration):
        for _ in range(500):
            next(it)

    # Verify the error handler was reached: call count should be ~20 (5 timeouts
    # per storm x 2 pairs per timeout x 2 storms before bail), not 500.
    assert call_count[0] < 500, (
        "Generator should have exited early via error handler, not iterated 500 times"
    )


def test_restarts_without_recovery_counter_resets_on_yield(inject_soapy_stub, monkeypatch):
    """_restarts_without_recovery resets to 0 when a buffer is successfully yielded.

    After a storm+reopen, if the stream recovers and yields buffers, a subsequent
    storm should get the full _MAX_RESTARTS_WITHOUT_RECOVERY budget rather than
    immediately hitting the limit from the previous storm.

    With _MAX_RESTARTS_WITHOUT_RECOVERY=1: two separate storms each followed by
    recovery should both trigger exactly one reopen (not bail after the second
    storm because the first storm's counter was not cleared).
    """
    stub = inject_soapy_stub
    # Shared counter across all device instances so reopens don't reset it.
    call_count = [0]

    def always_timeout_or_ok(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        c = call_count[0]
        # Pattern: 5 timeouts, 4 OK (2 pairs), 5 timeouts, 4 OK (2 pairs).
        # Each group of 5 timeouts triggers one storm+reopen.
        # Each group of 4 OKs yields 2 buffer pairs (sync+target per pair).
        phase = (c - 1) % 18   # 18-call cycle: 5 timeout + 4 ok + 5 timeout + 4 ok
        if phase < 5 or (9 <= phase < 14):
            return stub._StreamResult(-1)
        for buf in buffers:
            buf[:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    class _PatchedDevice:
        def __new__(cls, args):
            dev = MagicMock()
            dev.getSampleRate.return_value = 2_000_000.0
            dev.getFrequency.return_value = 99_900_000.0
            dev.getGain.return_value = 30.0
            dev.setupStream.return_value = object()
            dev.readStream.side_effect = always_timeout_or_ok
            return dev

    monkeypatch.setattr(stub, "Device", _PatchedDevice)

    rx = _make_receiver(buffer_size=64)
    rx._TIMEOUT_WINDOW_MAX = 5
    rx._TIMEOUT_WINDOW_SECS = 60.0
    rx._MAX_RESTARTS_WITHOUT_RECOVERY = 1   # only 1 unproductive reopen allowed
    rx._REOPEN_DELAY_S = 0.0
    monkeypatch.setattr(_time_module, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    rx.open()

    it = rx.paired_stream()
    # Storm 1 -> reopen -> 2 yields -> storm 2 -> reopen -> 2 more yields.
    # If the counter did not reset on yield, storm 2 would bail immediately
    # (without_recovery=2 > _MAX=1) instead of reopening and recovering.
    results = [next(it), next(it), next(it), next(it)]
    assert len(results) == 4, (
        "_restarts_without_recovery must reset after each yield so the second "
        "storm gets a fresh budget (expected 4 yields across 2 recover cycles)"
    )


def test_paired_stream_retries_on_overflow_ret(inject_soapy_stub):
    """A readStream return code of -4 (SOAPY_SDR_OVERFLOW) is retried, not fatal."""
    stub = inject_soapy_stub
    call_count = [0]

    def read_overflow_ret_once(stream, buffers, num_samples, timeoutUs=1_000_000):
        call_count[0] += 1
        if call_count[0] == 2:
            return stub._StreamResult(-4)  # overflow ret code, not timeout
        buffers[0][:num_samples] = np.zeros(num_samples, dtype=np.complex64)
        return stub._StreamResult(num_samples)

    rx = _make_receiver(buffer_size=64)
    rx.open()
    rx._dev.readStream.side_effect = read_overflow_ret_once

    # Collect 3 pairs: call 1 OK, call 2 overflow-ret (retry), call 3 OK, call 4 OK
    it = rx.paired_stream()
    results = [next(it), next(it), next(it)]
    assert len(results) == 3   # overflow was skipped, stream continued
    assert rx.overflow_count == 1  # overflow counter incremented



# ---------------------------------------------------------------------------
# stream() shim
# ---------------------------------------------------------------------------

def test_stream_yields_target_buffers():
    rx = _make_receiver(buffer_size=64)
    buf = next(iter(rx.stream()))
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.complex64
    assert len(buf) == 64


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------

def test_from_config():
    from beagle_node.config.schema import (
        FMStation, NodeConfig, NodeLocation, RSPduoConfig,
        SyncSignalConfig, TargetChannelConfig,
    )
    from beagle_node.sdr.rspduo import RSPduoReceiver

    node_cfg = NodeConfig(
        node_id="test-node",
        location=NodeLocation(latitude_deg=47.0, longitude_deg=-122.0, altitude_m=10.0),
        sdr_mode="rspduo",
        rspduo=RSPduoConfig(sample_rate_hz=2_000_000.0, buffer_size=1024, pipeline_offset_ns=250),
        sync_signal=SyncSignalConfig(
            primary_station=FMStation(
                station_id="TEST_FM", frequency_hz=99_900_000.0,
                latitude_deg=47.0, longitude_deg=-122.0,
            ),
        ),
        target_channels=[TargetChannelConfig(frequency_hz=155_100_000.0, label="TEST")],
    )
    rx = RSPduoReceiver.from_config(node_cfg)
    assert rx._sync_freq == 99_900_000.0
    assert rx._target_freq == 155_100_000.0
    assert rx._buffer_size == 1024


# ---------------------------------------------------------------------------
# Critical API ordering: post-init ch1 frequency re-apply
# ---------------------------------------------------------------------------

def test_open_reapplies_ch1_frequency_after_activate():
    """
    setFrequency(ch1) must be called AFTER both activateStream() calls in open().

    sdrplay_api_Init() fires during the first activateStream() and may
    program Rf_B = Rf_A.  The upstream driver's activateStream() re-applies
    pending Tuner B params, but our post-init setFrequency is a safety net.

    Regression guard: moving the setFrequency call before activateStream
    risks Tuner 2 locking to Tuner 1's frequency.
    """
    rx = _make_receiver(target_frequency_hz=443_475_000.0)
    rx.open()

    RX = 0  # SOAPY_SDR_RX stub value
    dev = rx._dev
    calls = dev.method_calls

    activate_indices = [i for i, c in enumerate(calls) if c[0] == "activateStream"]
    # All setFrequency(RX, 1, ...) calls on ch1 - there are two: one pre-init
    # (writes struct) and one post-init (sends sdrplay_api_Update).
    freq_ch1_indices = [
        i for i, c in enumerate(calls)
        if c[0] == "setFrequency" and len(c[1]) >= 3 and c[1][0] == RX and c[1][1] == 1
    ]

    assert activate_indices, "activateStream must be called"
    assert len(freq_ch1_indices) >= 2, (
        "setFrequency(ch1) must be called at least twice: once before activateStream "
        "(struct init) and once after (sdrplay_api_Update)"
    )

    last_activate = max(activate_indices)
    # The post-init call is the last setFrequency on ch1.
    last_freq_ch1 = max(freq_ch1_indices)
    assert last_freq_ch1 > last_activate, (
        "The final setFrequency(ch1) must come AFTER the last activateStream() call. "
        "sdrplay_api_Update(Tuner_B, Frf) is only sent when streamActive=True."
    )

    # Verify the post-init call uses the correct target frequency.
    post_init_call = calls[last_freq_ch1]
    assert post_init_call[1][2] == pytest.approx(443_475_000.0)


def test_open_reapplies_gains_after_activate():
    """
    _apply_gains() must be called after both activateStream() calls.

    sdrplay_api_Init() fires on the first activateStream(); at that point
    _streams[1] is NULL so Tuner_B gain updates are not acknowledged and Init
    may reset ch1's IFGR/RFGR to ch0's values.  Re-applying after both
    streams are active ensures sdrplay_api_Update confirms the settings.
    """
    rx = _make_receiver()
    rx.open()

    RX = 0  # SOAPY_SDR_RX stub value
    dev = rx._dev
    calls = dev.method_calls

    activate_indices = [i for i, c in enumerate(calls) if c[0] == "activateStream"]
    # Any setGain call on ch1 after the last activateStream
    gain_ch1_post_indices = [
        i for i, c in enumerate(calls)
        if c[0] == "setGain" and len(c[1]) >= 2 and c[1][0] == RX and c[1][1] == 1
        and i > max(activate_indices)
    ]

    assert activate_indices, "activateStream must be called"
    assert gain_ch1_post_indices, (
        "setGain on ch1 must be called after activateStream() to push "
        "sdrplay_api_Update(Tuner_B, Gain) through the service"
    )


def test_gain_uses_named_ifgr_not_distribution():
    """
    setGain must use named 'IFGR' (not the unnamed float distribution).

    SoapySDRPlay3's unnamed setGain(float) distributes gain across RFGR and
    IFGR based on its own table, which for the sync channel (ch0) produced
    IFGR~25 (high IF gain) instead of the intended ~59 (max attenuation).
    IFGR=25 on ch0 risks saturating the shared TDM ADC with a strong FM
    signal, corrupting ch1's interleaved samples.

    Expected IFGR = max(20, min(59, 79 - int(gain_db))):
      sync_gain_db=5   -> IFGR = max(20, min(59, 74)) = 59
      target_gain_db=30 -> IFGR = max(20, min(59, 49)) = 49
    """
    rx = _make_receiver(sync_gain_db=5, target_gain_db=30,
                        sync_lna_state=9, target_lna_state=0)
    rx.open()

    RX = 0  # SOAPY_SDR_RX stub value
    dev = rx._dev
    calls = dev.method_calls

    # Collect all named-IFGR setGain calls: setGain(RX, ch, "IFGR", value)
    ifgr_calls = [
        c for c in calls
        if c[0] == "setGain" and len(c[1]) == 4
        and c[1][0] == RX and c[1][2] == "IFGR"
    ]
    assert ifgr_calls, (
        "setGain must be called with named 'IFGR' argument; "
        "unnamed float distribution leaves IFGR uncontrolled"
    )

    # Confirm no unnamed float-only setGain calls for gain_db != "auto" channels
    # (unnamed calls have 3 positional args: RX, ch, float_value)
    unnamed_gain_calls = [
        c for c in calls
        if c[0] == "setGain" and len(c[1]) == 3
        and c[1][0] == RX and isinstance(c[1][2], float)
    ]
    assert not unnamed_gain_calls, (
        f"Unnamed setGain(float) must not be used - found: {unnamed_gain_calls}"
    )

    # Verify the computed IFGR values are correct for both channels
    ifgr_by_ch = {c[1][1]: c[1][3] for c in ifgr_calls}
    assert ifgr_by_ch.get(0) == pytest.approx(59), (
        f"sync ch0 IFGR should be 59 (max(20,min(59,79-5))), got {ifgr_by_ch.get(0)}"
    )
    assert ifgr_by_ch.get(1) == pytest.approx(49), (
        f"target ch1 IFGR should be 49 (max(20,min(59,79-30))), got {ifgr_by_ch.get(1)}"
    )


# ---------------------------------------------------------------------------
# No SoapySDR available
# ---------------------------------------------------------------------------

def test_raises_when_soapy_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "SoapySDR", None)
    if "beagle_node.sdr.rspduo" in sys.modules:
        del sys.modules["beagle_node.sdr.rspduo"]

    # Patch _SOAPY_AVAILABLE directly after import
    import importlib
    rspduo_mod = importlib.import_module("beagle_node.sdr.rspduo")
    monkeypatch.setattr(rspduo_mod, "_SOAPY_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="SoapySDR"):
        rspduo_mod.RSPduoReceiver(
            sync_frequency_hz=99_900_000.0,
            target_frequency_hz=155_100_000.0,
        )


# ---------------------------------------------------------------------------
# Backlog detection and draining
# ---------------------------------------------------------------------------

import logging  # noqa: E402 (needed for caplog.at_level)


class TestBacklogDrain:
    """Tests for the stale-buffer detection and drain logic in paired_stream().

    Two detection paths:
      - Fallback (no HAS_TIME): stale when sync readStream returns faster
        than half a buffer period (driver FIFO held pre-buffered data).
      - HAS_TIME: stale when wall-clock age of the buffer exceeds threshold.
    """

    HAS_TIME = 4  # SOAPY_SDR_HAS_TIME

    # buffer_size=256, rate=2 MSPS:
    #   _buf_dur_ns       = 256 / 2_000_000 * 1e9 = 128_000 ns
    #   _drain_thresh_ns  = 64_000 ns  (HAS_TIME age threshold)
    #   _fallback_thresh_ns = 1_000_000 ns (1 ms, fixed - not buffer-size-based)
    _BUF_SIZE = 256
    _THRESH_NS = 64_000
    _FALLBACK_THRESH_NS = 1_000_000  # 1 ms
    _NORMAL_READ_NS = 2_000_000      # 2 ms > fallback threshold -> not stale

    def test_backlog_drain_count_starts_zero(self):
        rx = _make_receiver()
        assert rx.backlog_drain_count == 0

    # ------ fallback path (flags=0, no HAS_TIME) ------

    def test_fallback_normal_read_yields_triple(self, monkeypatch, inject_soapy_stub):
        """Slow readStream (delta >= fallback threshold) -> not stale -> yields 3-tuple."""
        # monotonic_ns: pairs of (start, start+2_000_000) -> delta=2 ms > 1 ms threshold
        values = itertools.chain(
            [0, self._NORMAL_READ_NS], itertools.cycle([0, self._NORMAL_READ_NS])
        )
        monkeypatch.setattr(_time_module, "monotonic_ns", lambda: next(values))

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        s, t, w = next(rx.paired_stream())
        assert isinstance(w, int)
        assert rx.backlog_drain_count == 0

    def test_fallback_fast_read_discards(self, monkeypatch, inject_soapy_stub):
        """Fast readStream (delta < 1 ms fallback threshold) -> stale -> discards, drain_count += 1."""
        # Iter 1: delta=1_000 ns < 1_000_000 ns -> stale -> discard
        # Iter 2: delta=2_000_000 ns > 1_000_000 ns -> not stale -> yield
        values = itertools.chain(
            [0, 1_000, 0, self._NORMAL_READ_NS],
            itertools.cycle([0, self._NORMAL_READ_NS]),
        )
        monkeypatch.setattr(_time_module, "monotonic_ns", lambda: next(values))

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        next(rx.paired_stream())
        assert rx.backlog_drain_count == 1

    def test_drain_count_accumulates(self, monkeypatch, inject_soapy_stub):
        """Multiple stale reads accumulate drain_count correctly."""
        # 3 fast iterations, then a slow one (yield)
        values = itertools.chain(
            [0, 1_000, 0, 1_000, 0, 1_000],       # 3 stale
            itertools.cycle([0, self._NORMAL_READ_NS]),  # then normal
        )
        monkeypatch.setattr(_time_module, "monotonic_ns", lambda: next(values))

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        next(rx.paired_stream())
        assert rx.backlog_drain_count == 3

    def test_drain_clears_after_recovery_logs_info(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """After draining, first good buffer logs INFO 'backlog cleared'."""
        # 2 stale, then normal
        values = itertools.chain(
            [0, 1_000, 0, 1_000],                       # 2 stale
            itertools.cycle([0, self._NORMAL_READ_NS]),  # then normal
        )
        monkeypatch.setattr(_time_module, "monotonic_ns", lambda: next(values))

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        with caplog.at_level(logging.INFO, logger="beagle_node.sdr.rspduo"):
            next(rx.paired_stream())

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("backlog cleared" in m for m in info_msgs), (
            f"Expected 'backlog cleared' INFO log; got: {info_msgs}"
        )
        assert rx.backlog_drain_count == 2

    # ------ HAS_TIME path ------

    def test_has_time_not_stale_yields_triple(self, monkeypatch, inject_soapy_stub):
        """HAS_TIME set, buffer age < threshold -> not stale -> yields triple."""
        stub = inject_soapy_stub
        hw_time_ns = 1_700_000_000_000_000_000

        def read_has_time(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=hw_time_ns)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_has_time

        # age = time.time_ns() - hw_time_ns = 5_000 < 64_000 -> not stale
        monkeypatch.setattr(_time_module, "time_ns", lambda: hw_time_ns + 5_000)

        s, t, w = next(rx.paired_stream())
        assert w == hw_time_ns
        assert rx.backlog_drain_count == 0

    def test_has_time_stale_discards(self, monkeypatch, inject_soapy_stub):
        """HAS_TIME set, buffer with old timestamp -> yielded (not drained).

        Hardware timestamps are always accurate regardless of how old the buffer
        is in the FIFO - a slow host may queue many buffer periods.  No drain
        is performed when HAS_TIME is set; the buffer is passed through normally.
        """
        stub = inject_soapy_stub
        old_time_ns = 1_700_000_000_000_000_000
        now_ns = old_time_ns + 60_000_000_000  # 60 s later

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()

        def read_old_timestamp(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=old_time_ns)

        rx._dev.readStream.side_effect = read_old_timestamp
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        # Buffer should be yielded despite being 60 s old - timestamps are correct.
        sync_buf, target_buf, buf_wall_ns = next(rx.paired_stream())
        assert buf_wall_ns == old_time_ns
        assert rx.backlog_drain_count == 0

    def test_buf_wall_ns_is_timens_when_has_time(self, monkeypatch, inject_soapy_stub):
        """Yielded buf_wall_ns equals sr_sync.timeNs when SOAPY_SDR_HAS_TIME is set."""
        stub = inject_soapy_stub
        hw_time_ns = 1_234_567_890_000_000_000

        def read_has_time(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=hw_time_ns)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_has_time

        # age = 1_000 ns < 64_000 ns threshold -> not stale
        monkeypatch.setattr(_time_module, "time_ns", lambda: hw_time_ns + 1_000)

        _, _, w = next(rx.paired_stream())
        assert w == hw_time_ns

    def test_has_time_future_sanity_check_fallback(self, monkeypatch, inject_soapy_stub,
                                                    caplog):
        """HAS_TIME timestamp > 5 s in the future -> logs warning once, applies
        offset correction; first bad buffer yields buf_wall_ns == now."""
        import logging
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000     # reasonable 2025 wall clock
        # Simulate a driver bug: timestamp 3.75 years in the future
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000

        def read_future(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=future_ns)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_future
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        with caplog.at_level(logging.WARNING, logger="beagle_node.sdr.rspduo"):
            _, _, w = next(rx.paired_stream())

        # First bad buffer: correction = now - future_ns -> corrected = future_ns + (now - future_ns) = now
        assert w == now_ns, "first bad buffer: corrected timestamp should equal time.time_ns()"
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("HAS_TIME timestamp offset" in m for m in warn_msgs), (
            f"Expected one-time offset warning; got: {warn_msgs}"
        )

    def test_has_time_warn_logged_only_once(self, monkeypatch, inject_soapy_stub, caplog):
        """HAS_TIME future-offset warning is logged exactly once per stream session,
        not on every buffer."""
        import logging
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000

        def read_future(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=future_ns)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_future
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        with caplog.at_level(logging.WARNING, logger="beagle_node.sdr.rspduo"):
            next(it)   # first bad buffer -> warn
            next(it)   # second bad buffer -> no new warn
            next(it)   # third bad buffer -> no new warn

        warn_msgs = [r for r in caplog.records
                     if r.levelno == logging.WARNING and "HAS_TIME timestamp offset" in r.message]
        assert len(warn_msgs) == 1, (
            f"Expected exactly 1 HAS_TIME offset warning; got {len(warn_msgs)}"
        )

    def test_has_time_correction_applied_second_buffer(self, monkeypatch, inject_soapy_stub):
        """After the first bad buffer computes the correction, subsequent buffers
        use corrected timeNs (TCXO-based relative accuracy, not raw time.time_ns())."""
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        buf_dur_ns = 32_768_000   # ~32 ms at 2 MSPS, 65536 samples

        # Two consecutive hardware timestamps offset by one buffer duration
        future_ns_1 = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000
        future_ns_2 = future_ns_1 + buf_dur_ns

        timestamps = iter([future_ns_1, future_ns_2])

        def read_future(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            # Both streams (sync and target) return the same timestamp sequence;
            # only the sync stream's timeNs is used for buf_wall_ns.
            try:
                t = next(timestamps)
            except StopIteration:
                t = future_ns_2
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=t)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_future
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        _, _, w1 = next(it)   # first bad buffer: corrected = now_ns
        _, _, w2 = next(it)   # second bad buffer: corrected = now_ns + buf_dur_ns

        assert w1 == now_ns, f"first corrected buf_wall_ns should be now_ns; got {w1}"
        # Second buffer uses TCXO increment: corrected = future_ns_2 + (now - future_ns_1)
        #   = future_ns_1 + buf_dur_ns + now - future_ns_1 = now + buf_dur_ns
        expected_w2 = now_ns + buf_dur_ns
        assert w2 == expected_w2, (
            f"second corrected buf_wall_ns should be now_ns + buf_dur_ns = {expected_w2}; got {w2}"
        )

    def test_has_time_slightly_future_accepted(self, monkeypatch, inject_soapy_stub):
        """HAS_TIME timestamp <= 5 s in the future -> accepted (NTP clock skew)."""
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        slightly_future = now_ns + 3_000_000_000  # 3 seconds ahead (within tolerance)

        def read_slightly_future(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=slightly_future)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_slightly_future
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        _, _, w = next(rx.paired_stream())
        assert w == slightly_future, "timestamp <=5 s in the future should be accepted"

    # ------ Post-reinit recovery (backlog storm suppression) ------
    # These tests route timestamps per-stream: sync_stream gets the driving
    # pattern, target_stream always returns now_ns (its timestamp is unused).

    def _make_sync_routed_reader(self, stub, rx, sync_pattern, now_ns):
        """Return a readStream side_effect that feeds sync_pattern to sync calls only.

        paired_stream() always calls readStream for sync before target in each
        loop iteration, so even-numbered calls (0, 2, 4, ...) are sync and odd
        calls (1, 3, 5, ...) are target.  The mock returns the same stream handle
        for both channels, so stream-identity comparison is unreliable; using
        call order is the correct approach.
        """
        sync_ts_iter = iter(sync_pattern)
        call_idx = [0]

        def read_fn(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            idx = call_idx[0]
            call_idx[0] += 1
            # Even calls = sync stream (first per loop iteration)
            t = next(sync_ts_iter, now_ns) if idx % 2 == 0 else now_ns
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=t)

        return read_fn

    def test_reinit_recovery_suppresses_warning_spam(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """After HAS_TIME correction fires, individual backlog WARNINGs are suppressed.

        Simulates the FIFO oscillation pattern seen in node-greenlake logs:
          bad_timestamp -> stale(12) fresh(1) stale(1) fresh(1) stale(1) stale(14)
        followed by stable fresh buffers.  Expected: zero backlog WARNINGs,
        one INFO summary at the end.
        """
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000
        old_ns = now_ns - 60_000_000_000  # stale: 60 s ago

        # Sync stream sees: correction, then oscillating stale/fresh, then stable fresh.
        sync_pattern = (
            [future_ns]        # triggers correction + enters recovery
            + [old_ns] * 12    # stale episode 1
            + [now_ns] * 1     # fresh (consecutive=1, not yet 5)
            + [old_ns] * 1     # stale -> resets consecutive
            + [now_ns] * 1     # fresh (consecutive=1)
            + [old_ns] * 14    # stale -> resets consecutive
            + [now_ns] * 10    # 10 fresh -> exit recovery after 5
        )

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = self._make_sync_routed_reader(
            stub, rx, sync_pattern, now_ns
        )
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        with caplog.at_level(logging.WARNING, logger="beagle_node.sdr.rspduo"):
            for _ in range(8):
                next(it)

        # No backlog WARNING should have been emitted during recovery
        backlog_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "backlog detected" in r.message
        ]
        assert backlog_warns == [], (
            f"Expected 0 backlog WARNINGs during reinit recovery; "
            f"got {len(backlog_warns)}: {[r.message for r in backlog_warns]}"
        )

    def test_reinit_recovery_logs_summary_info(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """After recovery stabilises (N consecutive fresh), a single INFO summary is logged."""
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000

        # Correction fires (1 buffer), then 6 fresh.  With _REINIT_FRESH_NEEDED=5,
        # recovery completes after the 5th fresh buffer.  No stale buffers are
        # drained during recovery because HAS_TIME timestamps are never drained.
        sync_pattern = [future_ns] + [now_ns] * 6

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx._REINIT_FRESH_NEEDED = 5  # override class default for fast test
        rx.open()
        rx._dev.readStream.side_effect = self._make_sync_routed_reader(
            stub, rx, sync_pattern, now_ns
        )
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        with caplog.at_level(logging.INFO, logger="beagle_node.sdr.rspduo"):
            for _ in range(7):   # 1 (correction) + 6 fresh
                try:
                    next(it)
                except StopIteration:
                    break

        recovery_infos = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "reinit recovery complete" in r.message
        ]
        assert len(recovery_infos) == 1, (
            f"Expected exactly 1 reinit recovery INFO; got {len(recovery_infos)}"
        )
        assert "0" in recovery_infos[0].message, (
            f"Recovery INFO should mention 0 drained buffers; got: {recovery_infos[0].message}"
        )

    def test_reinit_recovery_no_cleared_info_during_oscillation(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """Individual 'backlog cleared' INFO messages are suppressed during recovery."""
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000
        old_ns = now_ns - 60_000_000_000

        # Alternating stale/fresh/stale/fresh - each "clear" should be suppressed.
        sync_pattern = [future_ns, old_ns, now_ns, old_ns, now_ns] + [now_ns] * 6

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = self._make_sync_routed_reader(
            stub, rx, sync_pattern, now_ns
        )
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        with caplog.at_level(logging.INFO, logger="beagle_node.sdr.rspduo"):
            for _ in range(6):
                next(it)

        cleared_infos = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "backlog cleared" in r.message
        ]
        assert cleared_infos == [], (
            f"Expected 0 'backlog cleared' INFO during recovery; "
            f"got {len(cleared_infos)}: {[r.message for r in cleared_infos]}"
        )

    def test_correction_persists_after_recovery(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """After reinit recovery completes, _has_time_correction is NOT cleared.

        A subsequent future timestamp must NOT trigger a second warning or a new
        recovery cycle.  The correction is applied silently until the next
        rollover epoch (drift > 5 s), which logs an INFO update instead.
        """
        import logging
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        future_ns = now_ns + 4 * 365 * 24 * 3600 * 1_000_000_000

        # Phase 1: correction fires, 5 consecutive fresh -> recovery ends.
        # Phase 2: another future timestamp -> correction applied silently (no WARNING).
        sync_pattern = (
            [future_ns]        # correction fires, recovery starts
            + [now_ns] * 5     # 5 consecutive fresh -> recovery ends
            + [future_ns]      # post-recovery future: correction applies silently
            + [now_ns] * 3     # stable fresh
        )

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx._REINIT_FRESH_NEEDED = 5
        rx.open()
        rx._dev.readStream.side_effect = self._make_sync_routed_reader(
            stub, rx, sync_pattern, now_ns
        )
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        it = rx.paired_stream()
        with caplog.at_level(logging.WARNING, logger="beagle_node.sdr.rspduo"):
            for _ in range(10):
                try:
                    next(it)
                except StopIteration:
                    break

        # Expect exactly ONE "HAS_TIME timestamp offset" WARNING (initial detection).
        # The post-recovery future timestamp must NOT re-trigger the warning.
        warn_msgs = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "HAS_TIME timestamp offset" in r.message
        ]
        assert len(warn_msgs) == 1, (
            f"Expected 1 offset warning (correction persists after recovery); "
            f"got {len(warn_msgs)}: {[r.message for r in warn_msgs]}"
        )

    def test_implausible_buf_wall_ns_falls_back_to_system_clock(
        self, monkeypatch, inject_soapy_stub, caplog
    ):
        """A negative or impossibly old HAS_TIME timestamp falls back to time.time_ns().

        This guards against C driver int64 overflow (observed as buf_wall_ns ~
        -7.45e18 producing 'buf age 9.22e12 ms' warnings and infinite stale loop).
        """
        import logging
        stub = inject_soapy_stub
        now_ns = 1_742_000_000_000_000_000
        # Simulate C driver int64 overflow -> large negative timeNs
        bad_timeNs = -7_450_000_000_000_000_000

        call_count = [0]

        def read_fn(stream, buffers, n, timeoutUs=1_000_000):
            for buf in buffers:
                buf[:n] = np.zeros(n, dtype=np.complex64)
            call_count[0] += 1
            # First sync call returns bad timestamp; subsequent calls return now.
            t = bad_timeNs if call_count[0] == 1 else now_ns
            return stub._StreamResult(n, flags=self.HAS_TIME, timeNs=t)

        rx = _make_receiver(buffer_size=self._BUF_SIZE)
        rx.open()
        rx._dev.readStream.side_effect = read_fn
        monkeypatch.setattr(_time_module, "time_ns", lambda: now_ns)

        with caplog.at_level(logging.ERROR, logger="beagle_node.sdr.rspduo"):
            _, _, w = next(rx.paired_stream())

        # Should fall back to system clock, not use the bad timestamp
        assert w == now_ns, (
            f"Implausible buf_wall_ns should fall back to time.time_ns()={now_ns}; "
            f"got {w}"
        )
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("implausible buf_wall_ns" in m for m in error_msgs), (
            f"Expected ERROR log for implausible timestamp; got: {error_msgs}"
        )
        # Must NOT be stuck in an infinite stale loop: the yielded buffer
        # shows processing continued normally after the fallback.
        assert rx.backlog_drain_count == 0
