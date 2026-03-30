# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
RSPduo dual-tuner receiver using SoapySDR Dual Tuner (DT) mode.

Hardware overview
-----------------
The SDRplay RSPduo contains two independent RF tuners (MSi001) feeding a
single shared ADC (MSi2500) via time-division multiplexing.  Both tuners
are clocked from the same 24 MHz TCXO.  All sample data travels through
one USB 2.0 connection as a single interleaved stream.

Because both channels share one ADC clock there is no inter-channel USB
jitter and no GPS 1PPS injection hardware is needed.  This makes the RSPduo
the simplest path to a production-quality two-channel TDOA node.

SoapySDR driver requirement
---------------------------
Independent per-channel tuning (separate setFrequency / setGain per tuner)
requires the ``rspduo-dual-independent-tuners`` branch of SoapySDRPlay3:

  git clone -b rspduo-dual-independent-tuners \\
      https://github.com/pothosware/SoapySDRPlay3.git
  cd SoapySDRPlay3 && mkdir build && cd build
  cmake .. && make -j$(nproc) && sudo make install && sudo ldconfig

This build adds independent per-channel tuning to the existing ``mode=DT``
(Dual Tuner) entry.  Both tuners start mirrored (same frequency/gain); the
driver lazily splits them into independent operation the first time a
parameter is set on channel 1.

RSPduoReceiver.open() opens the device directly with ``mode=DT``:

  dev = SoapySDR.Device('driver=sdrplay,mode=DT')

The device exposes two channels with independent frequency and gain:
  - Channel 0: Tuner 1 (sync channel  - FM broadcast)
  - Channel 1: Tuner 2 (target channel - LMR)

Each channel has its own setSampleRate / setFrequency / setGain.
SoapySDRPlay3 does not support a multi-channel [0, 1] setupStream on this
device, so two separate single-channel streams are used:

  sync_stream   = setupStream(RX, CF32, [0])
  target_stream = setupStream(RX, CF32, [1])
  readStream(sync_stream,   [sync_buf],   N)
  readStream(target_stream, [target_buf], N)

Note: older SoapySDRPlay3 docs describe opening two separate devices in
mode=MA (master) and mode=SL (slave) sequence.  That approach is unreliable
on current SoapySDRPlay3 builds because SelectDevice() for the slave fails
intermittently.

Timing model
------------
In dual-tuner mode the ADC interleaves samples from both tuners:
  ADC samples:  [T1, T2, T1, T2, ...]
  Ch 0 yields:  [T1, T1, T1, ...]   (even ADC samples)
  Ch 1 yields:  [T2, T2, T2, ...]   (odd  ADC samples)

After the MSi2500 decimation filters this produces two independent streams
at the requested sample rate, offset by 0.5 ADC sample periods relative to
each other.  At 2 MSPS this is ~= 250 ns.

The RSPduoConfig.pipeline_offset_ns field holds the correction value.
Set it to 0 initially; calibrate empirically using scripts/colocated_pair_test.py
against a co-located reference node (ideally another RSPduo).
"""

from __future__ import annotations

import collections
import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

from beagle_node.sdr.base import SDRConfig, SDRReceiver

if TYPE_CHECKING:
    from beagle_node.config.schema import NodeConfig

logger = logging.getLogger(__name__)


# Lazy import - SoapySDR is a system library, not in pyproject.toml deps
try:
    import SoapySDR as _SoapySDR  # noqa: N813
    _SOAPY_AVAILABLE = True

    # Route SoapySDR C-level log messages through Python logging so they
    # pick up structlog formatting (timestamps, node_id, etc.) automatically.
    _soapy_logger = logging.getLogger("soapy")
    _SOAPY_LEVEL_MAP = {
        _SoapySDR.SOAPY_SDR_FATAL:    logging.CRITICAL,
        _SoapySDR.SOAPY_SDR_CRITICAL: logging.CRITICAL,
        _SoapySDR.SOAPY_SDR_ERROR:    logging.ERROR,
        _SoapySDR.SOAPY_SDR_WARNING:  logging.WARNING,
        _SoapySDR.SOAPY_SDR_NOTICE:   logging.INFO,
        _SoapySDR.SOAPY_SDR_INFO:     logging.INFO,
        _SoapySDR.SOAPY_SDR_DEBUG:    logging.DEBUG,
        _SoapySDR.SOAPY_SDR_TRACE:    logging.DEBUG,
        _SoapySDR.SOAPY_SDR_SSI:      logging.DEBUG,
    }
    def _soapy_log_handler(level: int, msg: str) -> None:
        _soapy_logger.log(_SOAPY_LEVEL_MAP.get(level, logging.INFO), "%s", msg)
    _SoapySDR.registerLogHandler(_soapy_log_handler)

except ImportError:  # pragma: no cover
    _SOAPY_AVAILABLE = False
    _SoapySDR = None  # type: ignore[assignment]


class RSPduoReceiver(SDRReceiver):
    """
    SDRplay RSPduo dual-tuner receiver.

    Opens the RSPduo in Dual Tuner (DT) mode as a single SoapySDR device
    with two independent channels, and yields simultaneous IQ buffers from
    both tuners via a multi-channel stream.

    Parameters
    ----------
    sync_frequency_hz : float
        FM broadcast frequency for the sync (pilot) channel.
    target_frequency_hz : float
        LMR target frequency.
    sample_rate_hz : float
        Sample rate for BOTH channels.  Maximum 2 MHz in dual-tuner mode.
    sync_gain_db : float | str
        Gain for the sync channel in dB, or "auto" for AGC.
    sync_lna_state : int
        LNA state for the sync channel.  Higher = more LNA attenuation.
        Default 9 (maximum) prevents strong FM from saturating the TDM ADC.
    target_gain_db : float | str
        Gain for the target channel in dB, or "auto" for AGC.
    target_lna_state : int
        LNA state for the target channel.  0 = maximum LNA gain (best
        sensitivity for weak LMR signals).
    master_device_args : str
        SoapySDR device args.  Default: "driver=sdrplay".
        Use "driver=sdrplay,serial=XXXX" to target a specific RSPduo.
        Requires the rspduo-dual-independent-tuners branch of
        SoapySDRPlay3 (see module doc).
    slave_device_args : str | None
        Unused; kept for config compatibility.  Ignored.
    buffer_size : int
        IQ samples per read call per channel.
    """

    # Consecutive fresh buffers required after a HAS_TIME correction before
    # recovery is declared complete and normal backlog logging resumes.
    # Exposed as a class attribute so tests can override it.
    _REINIT_FRESH_NEEDED: int = 150

    # Rolling-window timeout exit: if more than _TIMEOUT_WINDOW_MAX timeouts
    # occur within _TIMEOUT_WINDOW_SECS seconds, the sdrplay_api stream is
    # considered unrecoverably unstable and the generator exits so systemd can
    # restart the node.  A simple consecutive counter fails here because the
    # alternating-FIFO storm (sync and target swap roles on successive reinits)
    # inserts intermittent successes that keep resetting a consecutive counter
    # indefinitely.  Normal operation sees at most 1 timeout per ~12 minutes;
    # a storm sees ~1 per second - the window easily distinguishes them.
    # Exposed as class attributes so tests can override them.
    _TIMEOUT_WINDOW_SECS: float = 60.0    # rolling window length
    _TIMEOUT_WINDOW_MAX:  int   = 6       # max timeouts allowed in that window
    _MAX_RESTARTS_WITHOUT_RECOVERY: int   = 3    # bail after this many consecutive unproductive reopens
    _REOPEN_DELAY_S:                float = 2.0  # sleep before each reopen attempt

    def __init__(
        self,
        sync_frequency_hz: float,
        target_frequency_hz: float,
        sample_rate_hz: float = 2_000_000.0,
        sync_gain_db: float | str = "auto",
        sync_lna_state: int = 9,
        target_gain_db: float | str = "auto",
        target_lna_state: int = 0,
        sync_antenna: str | None = None,
        target_antenna: str | None = None,
        master_device_args: str = "driver=sdrplay",
        slave_device_args: str | None = None,
        buffer_size: int = 65_536,
    ) -> None:
        if not _SOAPY_AVAILABLE:
            raise RuntimeError(
                "SoapySDR Python bindings not available.  "
                "Install via your system package manager (e.g. brew install soapysdr soapysdrsdr)"
            )
        self._sync_freq = float(sync_frequency_hz)
        self._target_freq = float(target_frequency_hz)
        self._rate = float(sample_rate_hz)
        self._sync_gain = sync_gain_db
        self._sync_lna_state = int(sync_lna_state)
        self._target_gain = target_gain_db
        self._target_lna_state = int(target_lna_state)
        self._sync_antenna = sync_antenna      # None = use driver default
        self._target_antenna = target_antenna  # None = use driver default
        self._master_args = master_device_args
        self._buffer_size = int(buffer_size)

        # SDRConfig represents the target channel (used by health reporting
        # and pipeline for sample_rate_hz).
        self._config = SDRConfig(
            center_frequency_hz=target_frequency_hz,
            sample_rate_hz=sample_rate_hz,
            gain_db=target_gain_db,
            device_args=master_device_args,
            buffer_size=buffer_size,
        )

        self._dev = None
        self._sync_stream = None
        self._target_stream = None
        self._overflow_count: int = 0
        self._backlog_drain_count: int = 0
        self._is_open: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, node_config: "NodeConfig") -> "RSPduoReceiver":
        """
        Construct from a NodeConfig.

        Frequencies come from node_config.sync_signal and node_config.target_channels
        so they are defined in exactly one place in the YAML.  All other RSPduo
        parameters come from node_config.rspduo.
        """
        cfg = node_config.rspduo
        if cfg is None:
            raise ValueError("from_config requires node_config.rspduo to be set")

        target_channels = node_config.target_channels
        if len(target_channels) > 1:
            logger.warning(
                "rspduo mode supports one target channel; "
                "using target_channels[0] (%.3f MHz), ignoring %d others",
                target_channels[0].frequency_hz / 1e6,
                len(target_channels) - 1,
            )

        return cls(
            sync_frequency_hz=node_config.sync_signal.primary_station.frequency_hz,
            target_frequency_hz=target_channels[0].frequency_hz,
            sample_rate_hz=cfg.sample_rate_hz,
            sync_gain_db=cfg.sync_gain_db,
            sync_lna_state=cfg.sync_lna_state,
            target_gain_db=cfg.target_gain_db,
            target_lna_state=cfg.target_lna_state,
            sync_antenna=cfg.sync_antenna,
            target_antenna=cfg.target_antenna,
            master_device_args=cfg.master_device_args,
            slave_device_args=cfg.slave_device_args,
            buffer_size=cfg.buffer_size,
        )

    # ------------------------------------------------------------------
    # SDRReceiver interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> SDRConfig:
        return self._config

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def backlog_drain_count(self) -> int:
        return self._backlog_drain_count

    def open(self) -> None:
        if self._is_open:
            return

        # Build device args for DT (Dual Tuner) mode.  The
        # rspduo-dual-independent-tuners branch of SoapySDRPlay3 adds
        # independent per-channel tuning to the existing mode=DT entry:
        # both tuners start mirrored, and the driver lazily splits them
        # into independent operation when a parameter is set on channel 1.
        # Build device args as a comma-separated string rather than a dict.
        # SoapySDR 0.8.1-5 (Debian 13/trixie) has a bug in Device.make()
        # where dict kwargs trigger a hash collision in the module registry.
        # The string form uses a different code path and works on all versions.
        device_parts = ["driver=sdrplay", "mode=DT"]
        for part in self._master_args.split(","):
            if part.startswith("serial="):
                device_parts.append(part.strip())
                break
        device_args = ",".join(device_parts)

        logger.info(
            "Opening RSPduo (DT mode): args=%r  sync=%.3f MHz  target=%.3f MHz",
            device_args, self._sync_freq / 1e6, self._target_freq / 1e6,
        )
        self._dev = _SoapySDR.Device(device_args)

        # Channel 0 = Tuner 1 (sync/FM), Channel 1 = Tuner 2 (target/LMR)
        # Setting ch1 params triggers the driver's lazy split from mirrored
        # to independent tuners.  Each channel's setFrequency / setGain then
        # routes to the correct tuner (Tuner_A or Tuner_B) in the SDRplay API.
        self._setup_channel(self._dev, 0, self._sync_freq, self._sync_gain,
                            self._sync_antenna, self._sync_lna_state)
        self._setup_channel(self._dev, 1, self._target_freq, self._target_gain,
                            self._target_antenna, self._target_lna_state)

        # SoapySDRPlay3 does not support a multi-channel [0, 1] setupStream.
        # Use two separate single-channel streams on the same device instead.
        self._sync_stream = self._dev.setupStream(
            _SoapySDR.SOAPY_SDR_RX, _SoapySDR.SOAPY_SDR_CF32, [0],
        )
        self._target_stream = self._dev.setupStream(
            _SoapySDR.SOAPY_SDR_RX, _SoapySDR.SOAPY_SDR_CF32, [1],
        )

        for ch, label in [(0, "sync  (ch0)"), (1, "target (ch1)")]:
            actual_rate = self._dev.getSampleRate(_SoapySDR.SOAPY_SDR_RX, ch)
            actual_freq = self._dev.getFrequency(_SoapySDR.SOAPY_SDR_RX, ch)
            actual_gain = self._dev.getGain(_SoapySDR.SOAPY_SDR_RX, ch)
            actual_ant  = self._dev.getAntenna(_SoapySDR.SOAPY_SDR_RX, ch)
            avail_ants  = self._dev.listAntennas(_SoapySDR.SOAPY_SDR_RX, ch)
            logger.info(
                "RSPduo %-15s  %.3f MHz  %.3f MSps  %.1f dB  antenna=%r  (available: %s)",
                label, actual_freq / 1e6, actual_rate / 1e6, actual_gain,
                actual_ant, ", ".join(repr(a) for a in avail_ants),
            )

        self._dev.activateStream(self._sync_stream)
        self._dev.activateStream(self._target_stream)

        # sdrplay_api_Init() fires during the first activateStream() (sync stream).
        # The upstream driver's activateStream() re-applies ch1 params that
        # differ from the Init defaults, but we re-apply here as a safety net
        # in case the driver's post-Init fixup doesn't cover all cases.
        self._dev.setFrequency(_SoapySDR.SOAPY_SDR_RX, 1, self._target_freq)
        actual_ch1_freq = self._dev.getFrequency(_SoapySDR.SOAPY_SDR_RX, 1)
        if abs(actual_ch1_freq - self._target_freq) > 1000.0:
            logger.error(
                "RSPduo ch1 frequency mismatch after post-init setFrequency: "
                "wanted %.3f MHz, got %.3f MHz",
                self._target_freq / 1e6, actual_ch1_freq / 1e6,
            )
        else:
            logger.info(
                "RSPduo ch1 post-init frequency: %.3f MHz (confirmed)",
                actual_ch1_freq / 1e6,
            )
        # Also re-apply gains: at Init time _streams[1] is NULL so Tuner_B
        # gain updates cannot be acknowledged; Init may reset ch1 to ch0 values.
        self._apply_gains(self._dev)

        self._is_open = True
        logger.info("RSPduo open - both tuners active (DT mode, independent tuning)")

    def close(self) -> None:
        if not self._is_open:
            return
        # Signal the streaming loop to stop.  Do NOT deactivate/close streams
        # here - readStream may be blocked in the C library on another thread,
        # and closing the stream underneath it causes a use-after-free crash.
        # The streaming loop checks _is_open between reads and will call
        # _close_streams() after breaking out.
        self._is_open = False
        logger.info("RSPduo close requested")

    def _close_streams(self) -> None:
        """Actually release SoapySDR streams + device (called from streaming thread)."""
        for stream in [self._sync_stream, self._target_stream]:
            if self._dev is not None and stream is not None:
                try:
                    self._dev.deactivateStream(stream)
                    self._dev.closeStream(stream)
                except Exception as exc:  # pragma: no cover
                    logger.warning("RSPduo stream close error: %s", exc)
        self._sync_stream = None
        self._target_stream = None
        self._dev = None
        logger.info("RSPduo closed")

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield target-channel IQ buffers (SDRReceiver compatibility shim)."""
        for _sync_buf, target_buf, _buf_wall_ns in self.paired_stream():
            yield target_buf

    # ------------------------------------------------------------------
    # Dual-channel stream
    # ------------------------------------------------------------------

    def paired_stream(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (sync_buf, target_buf, buf_wall_ns) triples captured simultaneously.

        Both buffers in each triple cover the same time window on the same
        shared ADC clock.  ``buf_wall_ns`` is the wall-clock time (``time.time_ns()``
        epoch) at which the buffer pair was received from the driver FIFO - use it
        as the ``onset_time_ns`` reference rather than calling ``time.time_ns()``
        inside the callback, to avoid GIL scheduling latency.  If the driver sets
        ``SOAPY_SDR_HAS_TIME``, the hardware sample-counter timestamp is used
        instead (eliminates per-event NTP jitter).

        Stale buffers (backlog) are silently discarded before yielding; a WARNING
        is logged when a backlog episode begins and an INFO when it clears.

            pos = 0
            for sync_buf, target_buf, buf_wall_ns in receiver.paired_stream():
                pipeline.process_sync_buffer(sync_buf,   raw_start_sample=pos)
                pipeline.process_target_buffer(target_buf, raw_start_sample=pos)
                pos += len(sync_buf)

        The caller is responsible for stopping iteration (e.g. via a stop
        flag) and calling close() afterward.
        """
        if not self._is_open:
            self.open()

        sync_buf   = np.zeros(self._buffer_size, dtype=np.complex64)
        target_buf = np.zeros(self._buffer_size, dtype=np.complex64)
        HAS_TIME = _SoapySDR.SOAPY_SDR_HAS_TIME

        # Backlog detection: only applies when HAS_TIME is NOT set.
        #
        # When HAS_TIME is set, buf_wall_ns = timeNs (hardware timestamp of the
        # buffer's first sample). This timestamp is always accurate regardless
        # of how many buffers are queued in the FIFO - a slow Pi may run 5+
        # buffer periods behind real-time in steady state, but the timestamps
        # are still correct. The C driver flushes the FIFO on sdrplay_api
        # re-anchor, so post-reinit stale data never reaches here. Genuine
        # overflow (FIFO full) produces SOAPY_SDR_OVERFLOW which is handled
        # separately.  No drain is needed when HAS_TIME is set.
        #
        # When HAS_TIME is not set, buf_wall_ns = time.time_ns() at readStream
        # completion time. A backlogged FIFO returns immediately (< 1 ms) with
        # stale data whose timestamp is wrong (it says "now" but the signal
        # arrived earlier). Drain to restore real-time accuracy.
        # 1 ms is safely below any real slot delivery time but above a truly
        # immediate (pre-filled, i.e. stale) return.
        _fallback_thresh_ns = 1_000_000  # 1 ms
        _draining            = False
        _drain_episode_count = 0

        # HAS_TIME correction: some SoapySDRPlay3 builds produce hardware
        # timestamps with a constant anchor offset (e.g. +151 days) due to a
        # driver bug.  We detect this on the first bad buffer, compute a one-time
        # additive correction (corrected_ns = timeNs + _has_time_correction), and
        # re-sync it whenever the corrected value drifts > 5 s.  This preserves
        # the TCXO counter's per-event relative accuracy (~usec) while keeping the
        # absolute timestamp within NTP error of true wall-clock time.
        _has_time_correction: int | None = None  # None = not yet determined
        _has_time_warn_logged: bool = False

        # Post-reinit recovery: when a HAS_TIME correction fires, the sdrplay_api
        # FIFO goes through an extended period of oscillating stale/fresh cycles
        # as the driver recovers from the internal stream reinit that reset
        # firstSampleNum.  These would generate many WARNING/INFO pairs (the
        # "backlog storm").  Instead, while in recovery mode we drain silently and
        # emit a single INFO summary once enough consecutive fresh buffers confirm
        # stability.  Threshold is self._REINIT_FRESH_NEEDED (class constant).
        _in_reinit_recovery: bool = False
        _reinit_total_drained: int = 0
        _reinit_consecutive_fresh: int = 0

        # Rolling timeout window: timestamps (monotonic seconds) of recent
        # readStream timeouts.  Old entries (> _TIMEOUT_WINDOW_SECS) are
        # trimmed before each check.  When the deque reaches _TIMEOUT_WINDOW_MAX
        # entries the sdrplay_api stream is in a burst-reinit storm; we close
        # and reopen the device to recover rather than exiting the process.
        # Only exit if the reopen itself fails (device truly gone).
        _timeout_times: collections.deque[float] = collections.deque()
        _stream_restart_count: int = 0
        # Counts consecutive close/reopen cycles that yielded no successful
        # buffers.  Resets to 0 each time a buffer is yielded.  If this reaches
        # _MAX_RESTARTS_WITHOUT_RECOVERY the driver is in a persistent broken
        # state (e.g. sdrplay_api service flapping after a redeploy) and we exit
        # so systemd can restart the node cleanly rather than spin indefinitely.
        # A sleep of _REOPEN_DELAY_S before each reopen gives the daemon time to
        # stabilise between attempts.
        _restarts_without_recovery: int = 0

        try:
            while True:
                if not self._is_open:
                    break

                # Time only the sync readStream - it is the blocking call.
                # The target read always returns immediately because target data
                # accumulates in the driver FIFO while the sync read waits.
                _t0 = time.monotonic_ns()
                sr_sync = self._dev.readStream(
                    self._sync_stream, [sync_buf],
                    self._buffer_size, timeoutUs=1_000_000,
                )
                _sync_read_ns = time.monotonic_ns() - _t0

                # Wall-clock time at which this buffer pair was received from
                # the driver FIFO.  If the driver provides a hardware timestamp
                # (SOAPY_SDR_HAS_TIME), use it - it was captured by the C
                # callback thread at sample-arrival time, avoiding GIL latency
                # and per-call NTP jitter.  Otherwise fall back to time.time_ns()
                # recorded here (earlier than the callback, later than capture).
                if sr_sync.flags & HAS_TIME:
                    # Use the driver's hardware timestamp when plausible.
                    # Stale (past) timestamps are valid - the backlog drain
                    # handles them.  A timestamp > 5 s in the future indicates
                    # a driver anchor bug; in that case apply a stored one-time
                    # correction so we still get TCXO relative accuracy rather
                    # than losing all hardware timing.  The correction is
                    # refreshed automatically if it drifts > 5 s (NTP step or
                    # repeated corruption), providing automatic re-sync.
                    _now = time.time_ns()
                    if sr_sync.timeNs <= _now + 5_000_000_000:
                        # Plausible timestamp (present or past) - use directly.
                        buf_wall_ns = sr_sync.timeNs
                        _has_time_correction = None   # driver healthy; reset
                    else:
                        # Future timestamp - apply or (re-)compute correction.
                        if not _has_time_warn_logged:
                            logger.warning(
                                "RSPduo HAS_TIME timestamp offset +%.3f s "
                                "(timeNs=%d, now=%d); applying one-time correction "
                                "for TCXO-based relative accuracy. "
                                "Check SoapySDRPlay3 driver build.",
                                (sr_sync.timeNs - _now) / 1e9,
                                sr_sync.timeNs, _now,
                            )
                            _has_time_warn_logged = True
                        if _has_time_correction is None:
                            _has_time_correction = _now - sr_sync.timeNs
                            # Enter recovery mode: suppress individual episode
                            # logs during the FIFO oscillation that follows.
                            _in_reinit_recovery = True
                            _reinit_total_drained = 0
                            _reinit_consecutive_fresh = 0
                        corrected = sr_sync.timeNs + _has_time_correction
                        # Re-sync if the correction has drifted > 5 s (this
                        # happens every ~12 minutes when the C driver's TCXO
                        # counter epoch increments: each rollover adds a fixed
                        # offset to timeNs that the correction no longer covers).
                        if corrected > _now + 5_000_000_000:
                            _old_offset = (sr_sync.timeNs - _now) / 1e9
                            _has_time_correction = _now - sr_sync.timeNs
                            corrected = _now
                            logger.info(
                                "RSPduo HAS_TIME correction updated "
                                "(offset now +%.1f s, up from previous value); "
                                "TCXO epoch rollover in C driver",
                                _old_offset,
                            )
                            if not _in_reinit_recovery:
                                _in_reinit_recovery = True
                                _reinit_total_drained = 0
                                _reinit_consecutive_fresh = 0
                        buf_wall_ns = corrected
                else:
                    buf_wall_ns = time.time_ns()

                # Safety check: a negative or impossibly old buf_wall_ns
                # means the driver produced a bad timeNs (e.g. int64 overflow
                # from a stale FIFO slot after an unhandled re-anchor).  Fall
                # back to system clock and reset the correction state so the
                # next valid HAS_TIME reading starts a fresh detection cycle.
                _now_check = time.time_ns()
                if sr_sync.flags & HAS_TIME and (
                    buf_wall_ns <= 0 or buf_wall_ns < _now_check - 300_000_000_000
                ):
                    logger.error(
                        "RSPduo HAS_TIME: implausible buf_wall_ns=%d "
                        "(age %.0f s); driver produced bad timestamp - "
                        "falling back to system clock and resetting correction",
                        buf_wall_ns,
                        (_now_check - buf_wall_ns) / 1e9,
                    )
                    buf_wall_ns = _now_check
                    _has_time_correction = None
                    _has_time_warn_logged = False
                    _in_reinit_recovery = False

                if not self._is_open:
                    break
                sr_tgt = self._dev.readStream(
                    self._target_stream, [target_buf],
                    self._buffer_size, timeoutUs=1_000_000,
                )

                logger.debug(
                    "RSPduo readStream: sync_ret=%d flags=0x%x timeNs=%d "
                    "sync_read_ms=%.2f  HAS_TIME=%s",
                    sr_sync.ret, sr_sync.flags, sr_sync.timeNs,
                    _sync_read_ns / 1e6,
                    bool(sr_sync.flags & HAS_TIME),
                )

                # Stale-buffer detection: not needed when HAS_TIME is set
                # (hardware timestamps are always accurate regardless of FIFO
                # depth). Only drain when falling back to time.time_ns(), where
                # a fast readStream return signals a stale pre-filled FIFO slot.
                # IMPORTANT: only classify as stale when sr_sync.ret > 0 (valid
                # data returned).  A fast return with ret < 0 is a driver error
                # (e.g. TIMEOUT from a broken stream after reopen) - it must fall
                # through to the error handler below, not be silently drained.
                # Without this guard, an instant-TIMEOUT loop after a reopen
                # bypasses the error handler and spins at 100%+ CPU indefinitely.
                if sr_sync.flags & HAS_TIME:
                    stale = False
                else:
                    stale = sr_sync.ret > 0 and _sync_read_ns < _fallback_thresh_ns

                if stale:
                    if not _draining:
                        # During post-reinit recovery the FIFO oscillates
                        # stale/fresh many times in quick succession; suppress
                        # individual episode WARNINGs and emit a single INFO
                        # summary once the driver stabilises.
                        if not _in_reinit_recovery:
                            logger.warning(
                                "RSPduo buffer backlog detected "
                                "(sync readStream returned in %d ms < threshold %d ms); "
                                "discarding stale samples until real-time",
                                _sync_read_ns // 1_000_000,
                                _fallback_thresh_ns // 1_000_000,
                            )
                        _draining = True
                    _drain_episode_count += 1
                    if _in_reinit_recovery:
                        _reinit_total_drained += 1
                        _reinit_consecutive_fresh = 0
                    self._backlog_drain_count += 1
                    # Diagnostic: log a summary every 100 drains so we can see
                    # what readStream is actually returning during a long drain.
                    # This reveals whether the driver is returning errors vs
                    # real (stale) buffers, and how fast it is returning.
                    if _drain_episode_count % 100 == 0:
                        logger.warning(
                            "RSPduo still draining: episode=%d  "
                            "sync_ret=%d  sync_flags=0x%x  sync_read_ms=%.1f  "
                            "HAS_TIME=%s  buf_wall_age_ms=%.1f  "
                            "in_reinit_recovery=%s",
                            _drain_episode_count,
                            sr_sync.ret, sr_sync.flags,
                            _sync_read_ns / 1e6,
                            bool(sr_sync.flags & HAS_TIME),
                            (time.time_ns() - buf_wall_ns) / 1e6,
                            _in_reinit_recovery,
                        )
                    continue

                if _draining:
                    if not _in_reinit_recovery:
                        logger.info(
                            "RSPduo backlog cleared after discarding %d stale buffers",
                            _drain_episode_count,
                        )
                    _draining = False
                    _drain_episode_count = 0

                if _in_reinit_recovery:
                    _reinit_consecutive_fresh += 1
                    if _reinit_consecutive_fresh >= self._REINIT_FRESH_NEEDED:
                        logger.info(
                            "RSPduo HAS_TIME reinit recovery complete: discarded "
                            "%d stale buffers during post-correction FIFO flush",
                            _reinit_total_drained,
                        )
                        _in_reinit_recovery = False
                        # Do NOT reset _has_time_correction here: the driver bug
                        # that produces future timestamps is persistent.  Clearing
                        # the correction after recovery would cause the warning and
                        # recovery cycle to repeat every ~5 s (150 buffers).  The
                        # correction stays in effect until the driver naturally
                        # produces a plausible timestamp (line ~488 above resets it
                        # then) or until the drift check above detects the next
                        # rollover epoch jump (~12 min) and recomputes it.

                if sr_sync.ret < 0 or sr_tgt.ret < 0:
                    TIMEOUT  = _SoapySDR.SOAPY_SDR_TIMEOUT  # -1
                    OVERFLOW = -4  # SOAPY_SDR_OVERFLOW as readStream return code
                    if sr_sync.ret == TIMEOUT or sr_tgt.ret == TIMEOUT:
                        if not self._is_open:
                            break
                        # Log the readStream duration so we can tell whether
                        # the driver is blocking for the full timeout period
                        # or returning immediately (which would mean the stale
                        # check above should have caught this and we have a
                        # logic gap).
                        _now_mono = time.monotonic()
                        _timeout_times.append(_now_mono)
                        _cutoff = _now_mono - self._TIMEOUT_WINDOW_SECS
                        while _timeout_times and _timeout_times[0] < _cutoff:
                            _timeout_times.popleft()
                        # First timeout in the rolling window is the expected
                        # periodic sdrplay_api reinit (~12 min); log at INFO.
                        # Subsequent timeouts in the same window signal a storm.
                        _timeout_log = (
                            logger.info
                            if len(_timeout_times) == 1
                            else logger.warning
                        )
                        _timeout_log(
                            "RSPduo readStream TIMEOUT reached error handler: "
                            "sync_ret=%d tgt_ret=%d  sync_read_ms=%.1f  "
                            "sync_flags=0x%x  HAS_TIME=%s  "
                            "fallback_thresh_ms=%.1f",
                            sr_sync.ret, sr_tgt.ret,
                            _sync_read_ns / 1e6,
                            sr_sync.flags, bool(sr_sync.flags & HAS_TIME),
                            _fallback_thresh_ns / 1e6,
                        )
                        if len(_timeout_times) >= self._TIMEOUT_WINDOW_MAX:
                            _stream_restart_count += 1
                            _restarts_without_recovery += 1
                            if _restarts_without_recovery > self._MAX_RESTARTS_WITHOUT_RECOVERY:
                                logger.error(
                                    "RSPduo readStream: %d consecutive reopens "
                                    "with no successful buffers - sdrplay_api "
                                    "service may be down or broken after redeploy; "
                                    "exiting so systemd can restart the node",
                                    _restarts_without_recovery - 1,
                                )
                                break
                            logger.warning(
                                "RSPduo readStream: %d timeouts in %.0fs "
                                "-- closing and reopening streams (restart #%d, "
                                "%d without recovery)",
                                len(_timeout_times), self._TIMEOUT_WINDOW_SECS,
                                _stream_restart_count,
                                _restarts_without_recovery,
                            )
                            self._is_open = False
                            self._close_streams()
                            while True:
                                time.sleep(self._REOPEN_DELAY_S)
                                try:
                                    self.open()
                                    break
                                except Exception as exc:
                                    logger.warning(
                                        "RSPduo reopen failed (restart #%d): %s "
                                        "-- retrying in %.0fs",
                                        _stream_restart_count, exc,
                                        self._REOPEN_DELAY_S,
                                    )
                            # Reset all loop-local state for the fresh stream.
                            _timeout_times.clear()
                            _draining = False
                            _drain_episode_count = 0
                            _in_reinit_recovery = False
                            _reinit_total_drained = 0
                            _reinit_consecutive_fresh = 0
                            _has_time_correction = None
                            _has_time_warn_logged = False
                            continue
                        _timeout_log(
                            "RSPduo readStream timeout: sync=%d  target=%d "
                            "(%d in %.0fs window) - retrying",
                            sr_sync.ret, sr_tgt.ret,
                            len(_timeout_times), self._TIMEOUT_WINDOW_SECS,
                        )
                        continue
                    if sr_sync.ret == OVERFLOW or sr_tgt.ret == OVERFLOW:
                        self._overflow_count += 1
                        logger.warning(
                            "RSPduo readStream overflow #%d: sync=%d  target=%d - retrying",
                            self._overflow_count, sr_sync.ret, sr_tgt.ret,
                        )
                        continue
                    logger.error(
                        "RSPduo readStream error: sync=%d  target=%d",
                        sr_sync.ret, sr_tgt.ret,
                    )
                    break

                n = min(sr_sync.ret, sr_tgt.ret)
                if n <= 0:
                    # ret=0 is technically valid (no samples ready) but should
                    # not happen with a 1-second timeout.  Guard against an
                    # immediate-return loop: the error handler above catches
                    # ret<0, but ret=0 would fall through and spin here.
                    continue
                _restarts_without_recovery = 0
                yield sync_buf[:n].copy(), target_buf[:n].copy(), buf_wall_ns
        finally:
            self._close_streams()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_channel(
        self,
        dev,
        ch: int,
        freq_hz: float,
        gain_db: float | str,
        antenna: str | None = None,
        lna_state: int = 0,
    ) -> None:
        """Apply sample rate, frequency, gain, LNA state, and optional antenna."""
        dev.setSampleRate(_SoapySDR.SOAPY_SDR_RX, ch, self._rate)
        dev.setFrequency(_SoapySDR.SOAPY_SDR_RX, ch, freq_hz)
        if antenna is not None:
            dev.setAntenna(_SoapySDR.SOAPY_SDR_RX, ch, antenna)
            logger.debug("RSPduo ch%d: antenna set to %r", ch, antenna)
        # Always set LNA state explicitly; the unnamed setGain() only distributes
        # across IFGR (IF gain reduction) and leaves RFGR (LNA state) at the
        # hardware default (0 = maximum LNA gain).  For the sync channel a high
        # LNA state prevents strong FM from saturating the shared TDM ADC.
        if gain_db == "auto":
            dev.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, True)
        else:
            dev.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, False)
            # Use explicit named IFGR (same formula as check_target.py) to avoid
            # SoapySDRPlay3's gain distribution, which can leave IFGR lower than
            # intended (more gain) when the unnamed setGain() is used.  For the
            # sync channel (ch0) this is critical: unnamed distribution resulted
            # in IFGR~25 (high IF gain) instead of the intended ~59 (max attenuation),
            # risking TDM ADC saturation from the local FM station and corruption
            # of the interleaved ch1 (target) samples.
            ifgr = max(20, min(59, 79 - int(float(gain_db))))
            dev.setGain(_SoapySDR.SOAPY_SDR_RX, ch, "IFGR", ifgr)
        # Set RFGR explicitly (named) so it isn't reset by any prior distribution.
        dev.setGain(_SoapySDR.SOAPY_SDR_RX, ch, "RFGR", lna_state)
        if gain_db == "auto":
            logger.debug("RSPduo ch%d: AGC enabled  LNA state=%d", ch, lna_state)
        else:
            logger.debug("RSPduo ch%d: IFGR=%d (gain_db=%.1f)  LNA state=%d",
                         ch, ifgr, float(gain_db), lna_state)

    def _apply_gains(self, dev) -> None:
        """Re-apply gain settings for both channels via sdrplay_api_Update.

        Must be called after both streams are active.  sdrplay_api_Init() fires
        during the first activateStream(); at that point the second stream's
        slot is NULL so Tuner_B gain updates are not acknowledged, and Init
        may reset ch1's gain to ch0's values.  Calling this after both
        activateStream() calls pushes the correct settings via Update.
        """
        for ch, gain_db, lna_state in [
            (0, self._sync_gain,   self._sync_lna_state),
            (1, self._target_gain, self._target_lna_state),
        ]:
            if gain_db == "auto":
                dev.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, True)
            else:
                dev.setGainMode(_SoapySDR.SOAPY_SDR_RX, ch, False)
                ifgr_set = max(20, min(59, 79 - int(float(gain_db))))
                dev.setGain(_SoapySDR.SOAPY_SDR_RX, ch, "IFGR", ifgr_set)
            dev.setGain(_SoapySDR.SOAPY_SDR_RX, ch, "RFGR", lna_state)
            ifgr = dev.getGain(_SoapySDR.SOAPY_SDR_RX, ch, "IFGR")
            rfgr = dev.getGain(_SoapySDR.SOAPY_SDR_RX, ch, "RFGR")
            logger.info("RSPduo ch%d post-init gain: IFGR=%d  RFGR(LNA)=%d", ch, ifgr, rfgr)
