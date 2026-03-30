# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""Unit tests for sdr/freq_hop.py (synchronous read-based continuous freq-hopping)."""

from __future__ import annotations

import itertools
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from beagle_node.sdr.base import SDRConfig
from beagle_node.sdr.freq_hop import FreqHopReceiver

# Arbitrary fixed wall-clock timestamp used where timing doesn't matter.
_NOW_NS: int = 1_700_000_000_000_000_000


def _queue_item(role: str, n_samples: int, wall_ns: int | None = None) -> tuple:
    """Build a 3-tuple queue item as produced by _run_loop.

    wall_ns defaults to time.time_ns() so items are 'fresh' by default.
    Pass an explicit wall_ns (e.g. _NOW_NS with monkeypatched time) for drain tests.
    """
    return (role, _raw_block(n_samples), wall_ns if wall_ns is not None else time.time_ns())

RATE   = 2_048_000.0
TARGET = 155_100_000.0
SYNC   = 99_900_000.0
BLOCK  = 512
SETTLE = 128

# Asymmetric: sync smaller than target, ratio 1:2
SYNC_BLOCK   = 256
TARGET_BLOCK = 512


def make_receiver(**kwargs) -> FreqHopReceiver:
    cfg = SDRConfig(center_frequency_hz=TARGET, sample_rate_hz=RATE, gain_db=40.0)
    defaults = dict(
        config=cfg,
        sync_frequency_hz=SYNC,
        samples_per_block=BLOCK,
        settling_samples=SETTLE,
    )
    defaults.update(kwargs)
    return FreqHopReceiver(**defaults)


def make_asymmetric_receiver(**kwargs) -> FreqHopReceiver:
    """Receiver with SYNC_BLOCK sync and TARGET_BLOCK target (ratio 1:2)."""
    cfg = SDRConfig(center_frequency_hz=TARGET, sample_rate_hz=RATE, gain_db=40.0)
    defaults = dict(
        config=cfg,
        sync_frequency_hz=SYNC,
        samples_per_block=SYNC_BLOCK,
        target_samples_per_block=TARGET_BLOCK,
        settling_samples=SETTLE,
    )
    defaults.update(kwargs)
    return FreqHopReceiver(**defaults)


def _raw_block(n_samples: int, value: int = 127) -> bytes:
    """Raw uint8 IQ block with all samples at `value`."""
    return bytes([value, value] * n_samples)


# ===========================================================================
# Layer A — _run_loop unit tests (mock read_bytes, no real threading)
# ===========================================================================

def _make_looping_sdr(rx: FreqHopReceiver, n_cycles: int) -> tuple[MagicMock, list[int]]:
    """
    Return a (mock_sdr, read_sizes) pair.

    mock_sdr.read_bytes() returns 127-filled bytes for `n_cycles` complete
    sync+target cycles (2*n_cycles calls total), then on the next call sets
    rx._stop_event and raises OSError so _run_loop exits cleanly.
    read_sizes accumulates the byte-count argument of each successful call.
    """
    read_sizes: list[int] = []
    call_n = [0]
    max_calls = n_cycles * 2

    def fake_read_bytes(n: int) -> bytes:
        if call_n[0] >= max_calls:
            rx._stop_event.set()
            raise OSError("test done")
        call_n[0] += 1
        read_sizes.append(n)
        return bytes([127] * n)

    mock_sdr = MagicMock()
    mock_sdr.read_bytes.side_effect = fake_read_bytes
    rx._sdr = mock_sdr
    rx._stop_event.clear()
    return mock_sdr, read_sizes


class TestRunLoop:
    """_run_loop(): sequential read_bytes + freq-switch pattern."""

    def test_sync_read_correct_byte_count(self):
        rx = make_receiver()
        _, read_sizes = _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        assert read_sizes[0] == BLOCK * 2

    def test_target_read_correct_byte_count(self):
        rx = make_receiver()
        _, read_sizes = _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        assert read_sizes[1] == BLOCK * 2

    def test_asymmetric_sync_read_byte_count(self):
        rx = make_asymmetric_receiver()
        _, read_sizes = _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        assert read_sizes[0] == SYNC_BLOCK * 2

    def test_asymmetric_target_read_byte_count(self):
        rx = make_asymmetric_receiver()
        _, read_sizes = _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        assert read_sizes[1] == TARGET_BLOCK * 2

    def test_queues_sync_first(self):
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        role, _, _w = rx._queue.get_nowait()
        assert role == "sync"

    def test_queues_target_second(self):
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        rx._queue.get_nowait()  # discard sync
        role, _, _w = rx._queue.get_nowait()
        assert role == "target"

    def test_queue_raw_bytes_length_sync(self):
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        _, raw, _w = rx._queue.get_nowait()
        assert len(raw) == BLOCK * 2

    def test_queue_raw_bytes_length_target(self):
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        rx._queue.get_nowait()  # discard sync
        _, raw, _w = rx._queue.get_nowait()
        assert len(raw) == BLOCK * 2

    def test_asymmetric_queue_bytes_length_sync(self):
        rx = make_asymmetric_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        _, raw, _w = rx._queue.get_nowait()
        assert len(raw) == SYNC_BLOCK * 2

    def test_asymmetric_queue_bytes_length_target(self):
        rx = make_asymmetric_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        rx._queue.get_nowait()  # discard sync
        _, raw, _w = rx._queue.get_nowait()
        assert len(raw) == TARGET_BLOCK * 2

    def test_queue_wall_ns_is_int(self):
        """_run_loop must record a wall_ns int immediately after each read_bytes()."""
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=1)
        rx._run_loop()
        _, _raw, wall_ns = rx._queue.get_nowait()
        assert isinstance(wall_ns, int) and wall_ns > 0

    def test_switches_to_target_freq_after_sync(self):
        rx = make_receiver()
        freq_sets: list[int] = []
        mock_sdr, _ = _make_looping_sdr(rx, n_cycles=1)
        type(mock_sdr).center_freq = property(
            fget=lambda self: None,
            fset=lambda self, v: freq_sets.append(v),
        )
        rx._run_loop()
        assert freq_sets[0] == int(TARGET)

    def test_switches_back_to_sync_freq_after_target(self):
        rx = make_receiver()
        freq_sets: list[int] = []
        mock_sdr, _ = _make_looping_sdr(rx, n_cycles=1)
        type(mock_sdr).center_freq = property(
            fget=lambda self: None,
            fset=lambda self, v: freq_sets.append(v),
        )
        rx._run_loop()
        assert freq_sets[1] == int(SYNC)

    def test_alternates_sync_target_over_multiple_cycles(self):
        rx = make_receiver()
        _make_looping_sdr(rx, n_cycles=3)
        rx._run_loop()
        roles = []
        while not rx._queue.empty():
            role, _, _w = rx._queue.get_nowait()
            roles.append(role)
        assert roles == ["sync", "target", "sync", "target", "sync", "target"]

    def test_stops_immediately_on_stop_event(self):
        rx = make_receiver()
        mock_sdr = MagicMock()
        rx._sdr = mock_sdr
        rx._stop_event.set()   # set before loop starts
        rx._run_loop()
        mock_sdr.read_bytes.assert_not_called()


# ===========================================================================
# Layer B — consumer unit tests (queue injection, no threading)
# ===========================================================================

class TestLabeledStreamConsumer:
    """Pre-populate the queue and verify labeled_stream output without hardware."""

    def _make_rx_with_queue(self, **kwargs) -> FreqHopReceiver:
        # Use a large block so _drain_thresh_ns (~16 ms) is comfortably above
        # Python test-execution overhead (IQ conversion, queue get/put, etc.).
        # BLOCK=512 gives only ~125 µs, which the second queue item routinely exceeds.
        kwargs.setdefault("samples_per_block", 65_536)
        kwargs.setdefault("settling_samples", SETTLE)
        rx = make_receiver(**kwargs)
        # Mark as open by setting a non-None _sdr sentinel so open() is skipped
        rx._sdr = object()
        return rx

    def test_yields_correct_roles_from_queue(self):
        rx = self._make_rx_with_queue()
        rx._queue.put(_queue_item("sync",   BLOCK))
        rx._queue.put(_queue_item("target", BLOCK))
        rx._queue.put(_queue_item("sync",   BLOCK))

        results = list(itertools.islice(rx.labeled_stream(), 3))
        roles = [r for r, _, _w in results]
        assert roles == ["sync", "target", "sync"]

    def test_yields_wall_ns(self):
        """labeled_stream yields the wall_ns from _run_loop as the third element."""
        rx = self._make_rx_with_queue()
        wall = time.time_ns()
        rx._queue.put(("sync", _raw_block(BLOCK), wall))
        _, _buf, got_wall = next(rx.labeled_stream())
        assert got_wall == wall

    def test_output_length_after_settling(self):
        rx = self._make_rx_with_queue()
        rx._queue.put(_queue_item("sync", BLOCK))
        _, buf, _ = next(rx.labeled_stream())
        assert len(buf) == BLOCK - SETTLE

    def test_output_dtype_complex64(self):
        rx = self._make_rx_with_queue()
        rx._queue.put(_queue_item("sync", BLOCK))
        _, buf, _ = next(rx.labeled_stream())
        assert buf.dtype == np.complex64

    def test_settling_bytes_discarded(self):
        """Settling region (value=200) is discarded; usable region (value=50) remains."""
        rx = self._make_rx_with_queue()
        settling_bytes = bytes([200, 200] * SETTLE)
        usable_bytes   = bytes([50, 50] * (BLOCK - SETTLE))
        rx._queue.put(("sync", settling_bytes + usable_bytes, time.time_ns()))
        _, buf, _ = next(rx.labeled_stream())
        expected = (50.0 - 127.5) / 128.0
        assert np.allclose(buf.real, expected, atol=0.01)

    def test_uint8_conversion(self):
        """uint8 127 → ~0+0j, 255 → ~+1+1j, 0 → ~-1-1j."""
        rx = self._make_rx_with_queue()
        rx._queue.put(_queue_item("sync", BLOCK))
        _, buf, _ = next(rx.labeled_stream())
        assert np.allclose(buf.real, (127.0 - 127.5) / 128.0, atol=0.01)

    def test_stall_exits_generator(self):
        """Queue.Empty (stall) breaks labeled_stream without error."""
        rx = self._make_rx_with_queue()
        rx._STALL_TIMEOUT_S = 0.01  # very short for test speed
        results = list(rx.labeled_stream())
        assert results == []

    def test_asymmetric_correct_usable_lengths(self):
        rx = make_asymmetric_receiver()
        rx._sdr = object()
        rx._queue.put(_queue_item("sync",   SYNC_BLOCK))
        rx._queue.put(_queue_item("target", TARGET_BLOCK))

        results = list(itertools.islice(rx.labeled_stream(), 2))
        (r0, b0, _), (r1, b1, _) = results
        assert r0 == "sync"   and len(b0) == SYNC_BLOCK   - SETTLE
        assert r1 == "target" and len(b1) == TARGET_BLOCK - SETTLE


class TestStreamWrapper:
    """stream() yields raw arrays without roles."""

    def test_stream_yields_all_buffers(self):
        rx = make_receiver()
        rx._sdr = object()
        for _ in range(4):
            rx._queue.put(_queue_item("sync", BLOCK))
        bufs = list(itertools.islice(rx.stream(), 4))
        assert len(bufs) == 4
        for buf in bufs:
            assert buf.dtype == np.complex64


# ===========================================================================
# Construction and validation
# ===========================================================================

class TestConstruction:
    def test_config_stored(self):
        rx = make_receiver()
        assert rx.config.center_frequency_hz == TARGET
        assert rx.config.sample_rate_hz == RATE

    def test_settling_must_be_less_than_block(self):
        with pytest.raises(ValueError, match="settling_samples"):
            make_receiver(samples_per_block=100, settling_samples=100)

    def test_symmetric_block_sizes(self):
        rx = make_receiver()
        assert rx._sync_block == BLOCK
        assert rx._target_block == BLOCK

    def test_usable_samples_symmetric(self):
        rx = make_receiver()
        assert rx.usable_samples_per_block == BLOCK - SETTLE
        assert rx.usable_samples("sync")   == BLOCK - SETTLE
        assert rx.usable_samples("target") == BLOCK - SETTLE

    def test_asymmetric_block_sizes(self):
        rx = make_asymmetric_receiver()
        assert rx._sync_block == SYNC_BLOCK
        assert rx._target_block == TARGET_BLOCK

    def test_asymmetric_usable_samples(self):
        rx = make_asymmetric_receiver()
        assert rx.usable_samples("sync")   == SYNC_BLOCK   - SETTLE
        assert rx.usable_samples("target") == TARGET_BLOCK - SETTLE
        assert rx.usable_samples_per_block == SYNC_BLOCK   - SETTLE

    def test_asymmetric_settling_must_be_less_than_smaller_block(self):
        with pytest.raises(ValueError, match="settling_samples"):
            make_asymmetric_receiver(settling_samples=SYNC_BLOCK)

    def test_device_serial_stored(self):
        rx = make_receiver(device_serial="TESTSERIAL")
        assert rx._device_serial == "TESTSERIAL"

    def test_device_serial_none_by_default(self):
        rx = make_receiver()
        assert rx._device_serial is None


# ===========================================================================
# from_config
# ===========================================================================

def _make_node_config_freq_hop(**freq_hop_overrides):
    """Build a minimal NodeConfig for freq_hop tests."""
    from beagle_node.config.schema import (
        FMStation, FreqHopConfig, NodeConfig, NodeLocation,
        SyncSignalConfig, TargetChannelConfig,
    )
    freq_hop_defaults = dict(
        sample_rate_hz=RATE,
        gain_db=40,
        samples_per_block=BLOCK,
        settling_samples=SETTLE,
    )
    freq_hop_defaults.update(freq_hop_overrides)
    return NodeConfig(
        node_id="test-node",
        location=NodeLocation(latitude_deg=47.0, longitude_deg=-122.0, altitude_m=10.0),
        sdr_mode="freq_hop",
        freq_hop=FreqHopConfig(**freq_hop_defaults),
        sync_signal=SyncSignalConfig(
            primary_station=FMStation(
                station_id="TEST_FM", frequency_hz=SYNC,
                latitude_deg=47.0, longitude_deg=-122.0,
            ),
        ),
        target_channels=[TargetChannelConfig(frequency_hz=TARGET, label="TEST")],
    )


def test_from_config_symmetric():
    node_cfg = _make_node_config_freq_hop()
    rx = FreqHopReceiver.from_config(node_cfg)
    assert rx.config.center_frequency_hz == TARGET
    assert rx._sync_freq == SYNC
    assert rx._sync_block == BLOCK
    assert rx._target_block == BLOCK
    assert rx._settling == SETTLE
    assert rx._device_serial is None


def test_from_config_asymmetric():
    node_cfg = _make_node_config_freq_hop(
        samples_per_block=SYNC_BLOCK,
        target_samples_per_block=TARGET_BLOCK,
    )
    rx = FreqHopReceiver.from_config(node_cfg)
    assert rx._sync_block == SYNC_BLOCK
    assert rx._target_block == TARGET_BLOCK


def test_from_config_device_serial():
    node_cfg = _make_node_config_freq_hop(device_serial="DEADBEEF")
    rx = FreqHopReceiver.from_config(node_cfg)
    assert rx._device_serial == "DEADBEEF"


# ===========================================================================
# Layer C — integration tests (mocked RtlSdr, real threading)
# ===========================================================================

def _make_mock_sdr_loop(rx: FreqHopReceiver, cycles: int) -> MagicMock:
    """
    Return a mock RtlSdr for _run_loop integration tests.

    read_bytes() returns 127-filled data for `cycles` complete sync+target
    pairs (2*cycles calls), then blocks on rx._stop_event (simulating a USB
    read stalled until close() sets the stop event) and raises OSError.

    This mirrors the real close() path: close() sets _stop_event and then
    joins the thread.  _run_loop sees the stop event and exits after the
    current read unblocks.
    """
    max_reads = cycles * 2
    call_count = [0]

    def fake_read_bytes(n: int) -> bytes:
        if call_count[0] >= max_reads:
            rx._stop_event.wait()   # unblocked when close() sets _stop_event
            raise OSError("stopped")
        call_count[0] += 1
        return bytes([127] * n)

    mock_sdr = MagicMock()
    mock_sdr.read_bytes.side_effect = fake_read_bytes
    return mock_sdr


@patch("beagle_node.sdr.freq_hop.FreqHopReceiver.open")
def test_integration_symmetric(mock_open):
    """Integration: _run_loop thread delivers correct blocks via queue."""
    # Use a large block (65 536 samples = ~32 ms) so the drain threshold is
    # ~16 ms — far more than any threading overhead in a test environment.
    rx = make_receiver(samples_per_block=65_536, settling_samples=1024)

    # 3 cycles → 6 queue items; we only consume 4 before calling close()
    mock_sdr = _make_mock_sdr_loop(rx, cycles=3)

    def real_open():
        rx._sdr = mock_sdr
        rx._stop_event.clear()
        while not rx._queue.empty():
            try:
                rx._queue.get_nowait()
            except queue.Empty:
                break
        rx._stream_thread = threading.Thread(
            target=rx._run_loop, daemon=True, name="FreqHopLoop",
        )
        rx._stream_thread.start()

    mock_open.side_effect = real_open

    results = list(itertools.islice(rx.labeled_stream(), 4))
    rx.close()

    assert len(results) == 4
    roles = [r for r, _, _w in results]
    assert roles == ["sync", "target", "sync", "target"]
    for role, buf, wall_ns in results:
        assert buf.dtype == np.complex64
        assert len(buf) == rx.usable_samples(role)
        assert isinstance(wall_ns, int)


@patch("beagle_node.sdr.freq_hop.FreqHopReceiver.open")
def test_integration_asymmetric(mock_open):
    """Integration: asymmetric mode delivers correct block sizes."""
    # Large blocks so the drain threshold (half-block duration) is well above
    # any threading overhead in the test environment.
    rx = make_asymmetric_receiver(
        samples_per_block=32_768, target_samples_per_block=65_536,
        settling_samples=1024,
    )

    # 2 cycles → 4 queue items
    mock_sdr = _make_mock_sdr_loop(rx, cycles=2)

    def real_open():
        rx._sdr = mock_sdr
        rx._stop_event.clear()
        while not rx._queue.empty():
            try:
                rx._queue.get_nowait()
            except queue.Empty:
                break
        rx._stream_thread = threading.Thread(
            target=rx._run_loop, daemon=True, name="FreqHopLoop",
        )
        rx._stream_thread.start()

    mock_open.side_effect = real_open

    results = list(itertools.islice(rx.labeled_stream(), 4))
    rx.close()

    assert len(results) == 4
    roles = [r for r, _, _w in results]
    assert roles == ["sync", "target", "sync", "target"]
    for role, buf, wall_ns in results:
        assert buf.dtype == np.complex64
        assert len(buf) == rx.usable_samples(role)
        assert isinstance(wall_ns, int)


# ===========================================================================
# close()
# ===========================================================================

def test_close_calls_sdr_close():
    rx = make_receiver()
    mock_sdr = MagicMock()
    rx._sdr = mock_sdr
    rx._stream_thread = None
    rx.close()
    mock_sdr.cancel_read_async.assert_not_called()
    mock_sdr.close.assert_called_once()
    assert rx._sdr is None


def test_close_is_idempotent():
    rx = make_receiver()
    rx.close()   # sdr is None — should not raise
    rx.close()


# ===========================================================================
# Backlog drain (buf_wall_ns age check in labeled_stream)
# ===========================================================================

class TestBacklogDrain:
    """
    labeled_stream() must silently discard buffers whose wall_ns age exceeds
    half a block duration (backlog drain), and expose the count via
    backlog_drain_count.

    Control time.time_ns() via monkeypatch so tests run deterministically.
    """

    def _make_rx(self) -> FreqHopReceiver:
        rx = make_receiver()
        rx._sdr = object()  # skip open()
        return rx

    def _drain_thresh_ns(self, rx: FreqHopReceiver) -> int:
        return int(rx._block * 1_000_000_000 / rx._config.sample_rate_hz / 2)

    def test_fresh_buffer_yields_and_not_counted(self, monkeypatch):
        """A buffer timestamped 'now' is fresh and must be yielded."""
        rx = self._make_rx()
        now = _NOW_NS
        monkeypatch.setattr("beagle_node.sdr.freq_hop.time.time_ns", lambda: now)
        rx._queue.put(("sync", _raw_block(BLOCK), now))  # age = 0
        _, _buf, wall = next(rx.labeled_stream())
        assert wall == now
        assert rx.backlog_drain_count == 0

    def test_stale_buffer_discarded_and_counted(self, monkeypatch):
        """A buffer older than the drain threshold is discarded."""
        rx = self._make_rx()
        thresh = self._drain_thresh_ns(rx)
        stale_wall = _NOW_NS
        now = _NOW_NS + thresh + 1  # 1 ns past the threshold
        monkeypatch.setattr("beagle_node.sdr.freq_hop.time.time_ns", lambda: now)
        # One stale buffer then a fresh one so the generator has something to yield
        rx._queue.put(("sync", _raw_block(BLOCK), stale_wall))
        rx._queue.put(("sync", _raw_block(BLOCK), now))
        results = list(itertools.islice(rx.labeled_stream(), 1))
        assert len(results) == 1  # only the fresh buffer
        assert rx.backlog_drain_count == 1

    def test_drain_count_accumulates(self, monkeypatch):
        """Multiple stale buffers each increment backlog_drain_count."""
        rx = self._make_rx()
        thresh = self._drain_thresh_ns(rx)
        stale_wall = _NOW_NS
        now = _NOW_NS + thresh + 1
        monkeypatch.setattr("beagle_node.sdr.freq_hop.time.time_ns", lambda: now)
        for _ in range(3):
            rx._queue.put(("sync", _raw_block(BLOCK), stale_wall))
        rx._queue.put(("sync", _raw_block(BLOCK), now))
        list(itertools.islice(rx.labeled_stream(), 1))
        assert rx.backlog_drain_count == 3

    def test_recovery_clears_draining_state(self, monkeypatch):
        """After draining, a fresh buffer is yielded normally."""
        rx = self._make_rx()
        thresh = self._drain_thresh_ns(rx)
        stale_wall = _NOW_NS
        now = _NOW_NS + thresh + 1
        monkeypatch.setattr("beagle_node.sdr.freq_hop.time.time_ns", lambda: now)
        rx._queue.put(("sync", _raw_block(BLOCK), stale_wall))  # stale → drain
        rx._queue.put(("sync", _raw_block(BLOCK), now))          # fresh → yield
        rx._queue.put(("sync", _raw_block(BLOCK), now))          # fresh → yield
        results = list(itertools.islice(rx.labeled_stream(), 2))
        assert len(results) == 2
        assert rx.backlog_drain_count == 1

    def test_wall_ns_passed_through_to_caller(self, monkeypatch):
        """The wall_ns yielded is the one from the queue item, not time.time_ns()."""
        rx = self._make_rx()
        now = _NOW_NS
        monkeypatch.setattr("beagle_node.sdr.freq_hop.time.time_ns", lambda: now)
        expected_wall = now - 100  # slightly old but within threshold
        rx._queue.put(("sync", _raw_block(BLOCK), expected_wall))
        _, _buf, got_wall = next(rx.labeled_stream())
        assert got_wall == expected_wall

    def test_backlog_drain_count_zero_initially(self):
        rx = make_receiver()
        assert rx.backlog_drain_count == 0
