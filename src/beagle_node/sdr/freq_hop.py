# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Continuous frequency-hopping SDR receiver using pyrtlsdr.

Uses pyrtlsdr's synchronous `read_bytes(n)` API in a background thread to
alternate a single RTL-SDR between two frequencies continuously:

  sync block -> switch to target -> target block -> switch to sync -> repeat

Why not do the frequency switch inside the async callback?
----------------------------------------------------------
Calling `rtlsdr_set_center_freq` (a USB control transfer) from within the
libusb async bulk completion callback causes LIBUSB_ERROR_BUSY (-6): the
event loop is already dispatching the callback and cannot process the
synchronous control transfer.  The DC9ST C binary avoids this because it
calls directly into libusb with a different locking strategy; Python via
pyrtlsdr cannot replicate that.

Instead, we use pyrtlsdr's synchronous `read_bytes(n)`:
  - `read_bytes(sync_bytes)` blocks until exactly `sync_block x 2` bytes
    have been read from the device, then returns.
  - At that point no async loop is running, so `sdr.center_freq = freq`
    (a synchronous control transfer) is completely safe.
  - `read_bytes(target_bytes)` then runs and returns.
  - `sdr.center_freq = sync_freq` switches back.
  - Repeat indefinitely.

The Python overhead between a `read_bytes` return and the next `center_freq`
assignment is sub-millisecond - negligible compared to the 24+ ms discarded
by `settling_samples`.

Settling
--------
The R820T PLL needs ~10-40 ms to lock after each hop.  The first
`settling_samples` of every block are discarded in the consumer.
Use `scripts/measure_settling.py` to calibrate the value.

labeled_stream()
----------------
Yields (role, iq_buf, wall_ns) tuples where role is "sync" or "target" and
wall_ns is the time.time_ns() value captured immediately after read_bytes()
returned in the background thread.  Buffers whose age exceeds half a block
duration are silently discarded (backlog drain); wall_ns can be used directly
as buf_wall_ns for onset_time_ns computation, analogous to the RSPduo path.

stream()
--------
Yields all buffers sequentially (sync, target, sync, target, ...).
For compatibility with the SDRReceiver interface.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Literal

import numpy as np

from beagle_node.sdr.base import SDRConfig, SDRReceiver

if TYPE_CHECKING:
    from beagle_node.config.schema import NodeConfig

logger = logging.getLogger(__name__)

Role = Literal["sync", "target"]


class FreqHopReceiver(SDRReceiver):
    """
    Frequency-hopping RTL-SDR receiver using pyrtlsdr.

    Alternates between sync (FM) and target (LMR) frequencies by issuing
    synchronous block reads separated by explicit frequency changes.  The
    frequency switch happens immediately after each `read_bytes()` call
    returns, outside any async context.

    Parameters
    ----------
    config : SDRConfig
        Uses center_frequency_hz as the **target** frequency.
    sync_frequency_hz : float
        FM broadcast station frequency (e.g. 99.9 MHz for KISW).
    samples_per_block : int
        IQ samples per sync block (~32 ms at 2.048 MSPS).  Also the default
        target block size in symmetric mode.
    target_samples_per_block : int or None
        Override for the target block size (asymmetric mode).  If None,
        symmetric mode (both channels equal samples_per_block).
    settling_samples : int
        Samples discarded at the start of each block for R820T PLL settling.
        Must be < the smaller of the two block sizes.
    device_serial : str or None
        USB serial number of the RTL-SDR to open.  None = first found.
    """

    # Seconds to wait for any block before declaring a USB stall.
    _STALL_TIMEOUT_S: float = 10.0

    def __init__(
        self,
        config: SDRConfig,
        sync_frequency_hz: float,
        samples_per_block: int = 65_536,
        target_samples_per_block: int | None = None,
        settling_samples: int = 40_960,
        device_serial: str | None = None,
    ) -> None:
        sync_block = int(samples_per_block)
        target_block = int(target_samples_per_block) if target_samples_per_block is not None \
            else sync_block
        min_block = min(sync_block, target_block)
        if settling_samples >= min_block:
            raise ValueError(
                f"settling_samples ({settling_samples}) must be < "
                f"min(sync_block={sync_block}, target_block={target_block}) = {min_block}"
            )
        self._config = config
        self._sync_freq = float(sync_frequency_hz)
        self._sync_block = sync_block
        self._target_block = target_block
        self._block = max(sync_block, target_block)
        self._settling = int(settling_samples)
        self._device_serial = device_serial

        # Runtime state - all None/unset until open()
        self._sdr: object | None = None
        self._stream_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._queue: queue.Queue[tuple[Role, bytes, int]] = queue.Queue()
        self._backlog_drain_count: int = 0

    # ------------------------------------------------------------------
    # Factory from NodeConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, node_config: "NodeConfig") -> "FreqHopReceiver":
        """Construct from a NodeConfig."""
        cfg = node_config.freq_hop
        if cfg is None:
            raise ValueError("from_config requires node_config.freq_hop to be set")

        target_channels = node_config.target_channels
        if len(target_channels) > 1:
            logger.warning(
                "freq_hop mode supports one target channel; "
                "using target_channels[0] (%.3f MHz), ignoring %d others",
                target_channels[0].frequency_hz / 1e6,
                len(target_channels) - 1,
            )

        sdr_config = SDRConfig(
            center_frequency_hz=target_channels[0].frequency_hz,
            sample_rate_hz=cfg.sample_rate_hz,
            gain_db=cfg.gain_db,
        )
        return cls(
            config=sdr_config,
            sync_frequency_hz=node_config.sync_signal.primary_station.frequency_hz,
            samples_per_block=cfg.samples_per_block,
            target_samples_per_block=cfg.target_samples_per_block,
            settling_samples=cfg.settling_samples,
            device_serial=cfg.device_serial,
        )

    # ------------------------------------------------------------------
    # SDRReceiver interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> SDRConfig:
        return self._config

    @property
    def overflow_count(self) -> int:
        return 0

    @property
    def backlog_drain_count(self) -> int:
        """Number of stale buffers discarded by the backlog drain logic."""
        return self._backlog_drain_count

    @property
    def sync_block_samples(self) -> int:
        """Total samples per sync block (including settling transient)."""
        return self._sync_block

    @property
    def target_block_samples(self) -> int:
        """Total samples per target block (including settling transient)."""
        return self._target_block

    @property
    def settling_samples(self) -> int:
        """Samples discarded at the start of each block for R820T PLL settling."""
        return self._settling

    @property
    def usable_samples_per_block(self) -> int:
        """Usable samples per sync block after discarding settling transient."""
        return self._sync_block - self._settling

    def usable_samples(self, role: Role) -> int:
        """Usable samples for the given role after discarding settling."""
        block = self._sync_block if role == "sync" else self._target_block
        return block - self._settling

    def open(self) -> None:
        if self._sdr is not None:
            return

        try:
            from rtlsdr import RtlSdr  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pyrtlsdr is required for freq_hop mode.  "
                "Install with: pip install pyrtlsdr"
            ) from exc

        if self._device_serial is not None:
            sdr = RtlSdr(serial_number=self._device_serial)
        else:
            sdr = RtlSdr()

        sdr.sample_rate = self._config.sample_rate_hz
        sdr.center_freq = int(self._sync_freq)   # start on sync frequency
        sdr.gain = self._config.gain_db
        if hasattr(sdr, "reset_buffer"):
            sdr.reset_buffer()

        self._sdr = sdr
        self._stop_event.clear()
        # Drain any stale items from a previous run
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if self._sync_block == self._target_block:
            logger.info(
                "Starting freq_hop (symmetric): sync=%.3f MHz  target=%.3f MHz  block=%d samp",
                self._sync_freq / 1e6, self._config.center_frequency_hz / 1e6,
                self._sync_block,
            )
        else:
            logger.info(
                "Starting freq_hop (asymmetric): sync=%.3f MHz [%d samp]  "
                "target=%.3f MHz [%d samp]",
                self._sync_freq / 1e6, self._sync_block,
                self._config.center_frequency_hz / 1e6, self._target_block,
            )

        self._stream_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="FreqHopLoop",
        )
        self._stream_thread.start()

    def close(self) -> None:
        if self._sdr is None:
            return
        self._stop_event.set()
        sdr = self._sdr
        self._sdr = None
        # We use synchronous read_bytes(), NOT read_bytes_async().
        # cancel_read_async() is the API for the async callback mode; calling it
        # during a synchronous read sets librtlsdr's async_cancel flag, which
        # corrupts internal state and causes rtlsdr_close() to crash with
        # "rtlsdr_demod_write_reg failed with -1" -> segfault on some librtlsdr
        # versions.  Instead, we simply signal the stop event and let the
        # background thread finish its current read_bytes() call naturally
        # (at most one sync + one target block, ~64 ms at 2.048 MSPS).
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None
        try:
            sdr.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        logger.info("FreqHopReceiver closed")

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield all blocks sequentially (sync, target, sync, target, ...)."""
        for _role, buf, _wall_ns in self.labeled_stream():
            yield buf

    def labeled_stream(self) -> Generator[tuple[Role, np.ndarray, int], None, None]:
        """
        Yield (role, iq_buf, wall_ns) tuples, alternating sync and target blocks.

        Blocks are placed on an internal queue by the background read loop.
        Each iq_buf has usable_samples(role) complex64 samples (settling
        transient already removed).  wall_ns is time.time_ns() captured
        immediately after read_bytes() returned in the background thread --
        use it directly as buf_wall_ns for onset_time_ns computation.

        Buffers whose age exceeds half a block duration are silently discarded
        (backlog drain).  A WARNING is logged when draining begins and an INFO
        when it clears.  backlog_drain_count accumulates discarded buffers.
        """
        if self._sdr is None:
            self.open()

        skip_bytes = self._settling * 2
        # Stale-buffer threshold: half the duration of the largest block.
        _drain_thresh_ns = int(self._block * 1_000_000_000 / self._config.sample_rate_hz / 2)
        _draining = False
        _drain_episode_count = 0

        while True:
            try:
                role, raw, wall_ns = self._queue.get(timeout=self._STALL_TIMEOUT_S)
            except queue.Empty:
                logger.error(
                    "FreqHop stalled - no data for %.0f s (USB error or device hang?); "
                    "exiting stream",
                    self._STALL_TIMEOUT_S,
                )
                break

            buf_age_ns = time.time_ns() - wall_ns
            if buf_age_ns > _drain_thresh_ns:
                if not _draining:
                    logger.warning(
                        "FreqHop buffer backlog detected "
                        "(buf age %d ms > %d ms threshold); "
                        "discarding stale samples until real-time",
                        buf_age_ns // 1_000_000,
                        _drain_thresh_ns // 1_000_000,
                    )
                    _draining = True
                _drain_episode_count += 1
                self._backlog_drain_count += 1
                continue

            if _draining:
                logger.info(
                    "FreqHop backlog cleared after discarding %d stale buffers",
                    _drain_episode_count,
                )
                _draining = False
                _drain_episode_count = 0

            usable = raw[skip_bytes:]
            raw_arr = np.frombuffer(usable, dtype=np.uint8).astype(np.float32)
            raw_arr = (raw_arr - 127.5) / 128.0
            iq = (raw_arr[0::2] + 1j * raw_arr[1::2]).astype(np.complex64)
            yield role, iq, wall_ns

    # ------------------------------------------------------------------
    # Internal: sequential read loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        Background thread: sequential sync/target reads with explicit freq switches.

        read_bytes(n) blocks until n bytes arrive, then returns.  At that point
        no async loop is running, so sdr.center_freq assignment (a USB control
        transfer) is safe - no LIBUSB_ERROR_BUSY re-entrancy issue.
        """
        sdr = self._sdr
        if sdr is None:
            return

        sync_bytes = self._sync_block * 2
        target_bytes = self._target_block * 2

        while not self._stop_event.is_set():
            # --- Sync block ---
            try:
                raw = sdr.read_bytes(sync_bytes)  # type: ignore[union-attr]
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.error("FreqHop sync read error: %s", exc)
                return
            self._queue.put_nowait(("sync", bytes(raw), time.time_ns()))

            if self._stop_event.is_set():
                return

            # Switch to target frequency - safe here (outside async callback)
            sdr.center_freq = int(self._config.center_frequency_hz)  # type: ignore[union-attr]

            # --- Target block ---
            try:
                raw = sdr.read_bytes(target_bytes)  # type: ignore[union-attr]
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.error("FreqHop target read error: %s", exc)
                return
            self._queue.put_nowait(("target", bytes(raw), time.time_ns()))

            if self._stop_event.is_set():
                return

            # Switch back to sync frequency
            sdr.center_freq = int(self._sync_freq)  # type: ignore[union-attr]
