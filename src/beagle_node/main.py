# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Beagle node entry point.

Usage
-----
    # Classic mode: full config in a local YAML file
    python -m beagle_node --config config/node.yaml

    # Remote-config mode: fetch config from server at startup
    python -m beagle_node --bootstrap /etc/beagle/bootstrap.yaml

    beagle-node --config config/node.yaml          # if installed via pip

The node:
  1. Loads and validates the YAML config (or fetches it from the server).
  2. Configures structured logging.
  3. Creates the SDR receiver (or MockReceiver in --mock mode).
  4. Starts the EventReporter background thread.
  5. Starts the HealthServer background thread.
  6. Runs the pipeline loop: stream IQ -> pipeline -> reporter.
  7. Handles SIGTERM / SIGINT for clean shutdown.

Remote config mode (--bootstrap)
---------------------------------
When --bootstrap is supplied the node reads only a minimal bootstrap.yaml
(server_url, node_id, node_secret) and fetches its full operating config
from the server.  The fetched config is cached locally so the node can
start even if the server is temporarily unreachable.

A background thread long-polls the server for config updates.  When a new
version is available, hot-reloadable fields (carrier thresholds, target
channels, clock calibration) are applied immediately.  Fields that require
an SDR restart are logged as warnings; restart the node to apply them.
"""

from __future__ import annotations

import argparse
import base64
import gc
import logging
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GC pause monitoring
# ---------------------------------------------------------------------------
# Full (gen-2) collections on a large heap can pause Python for hundreds of ms,
# long enough to overflow the RSPduo sample FIFO.  Log any cycle >50 ms so we
# can correlate with buffer backlog events and decide whether gc.freeze() is
# warranted.
_gc_t0: float = 0.0

def _gc_callback(phase: str, info: dict) -> None:
    global _gc_t0
    if phase == "start":
        _gc_t0 = time.monotonic()
    else:
        elapsed_ms = (time.monotonic() - _gc_t0) * 1000
        if elapsed_ms > 50:
            logger.warning(
                "GC gen%d pause %.0f ms (collected %d objects)",
                info.get("generation", -1),
                elapsed_ms,
                info.get("collected", 0),
            )

gc.callbacks.append(_gc_callback)

# Exit code that signals "restart me with new config".
# 75 = EX_TEMPFAIL from sysexits.h - conventional for "try again later".
# The systemd unit uses Restart=always to honour this.
EXIT_RESTART = 75


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beagle-node",
        description="Beagle node - SDR capture and event reporting",
    )
    config_src = p.add_mutually_exclusive_group(required=True)
    config_src.add_argument("--config", metavar="PATH",
                            help="Path to full node YAML config file")
    config_src.add_argument("--bootstrap", metavar="PATH",
                            help="Path to minimal bootstrap.yaml; full config is "
                                 "fetched from the server at startup")
    p.add_argument("--mock", action="store_true",
                   help="Use MockReceiver instead of real SDR (for testing)")
    p.add_argument("--mock-duration", type=float, default=30.0, metavar="SEC",
                   help="Duration of synthetic IQ when using --mock (default 30 s)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Log verbosity (default INFO)")
    p.add_argument("--json-logs", action="store_true",
                   help="Force JSON log output (default: auto-detect from tty)")
    p.add_argument("--no-health", action="store_true",
                   help="Disable the health HTTP endpoint")
    return p


def run(args: argparse.Namespace | None = None) -> int:
    """
    Main entry point.  Returns exit code.

    Separated from main() to allow testing without sys.exit().
    """
    # ------------------------------------------------------------------
    # Parse args
    # ------------------------------------------------------------------
    if args is None:
        args = _build_argparser().parse_args()

    # ------------------------------------------------------------------
    # Config  (local file or remote fetch)
    # ------------------------------------------------------------------
    from beagle_node.config.schema import NodeConfig, load_config

    _remote_fetcher = None  # set below in bootstrap mode

    if args.config:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR: invalid config: {exc}", file=sys.stderr)
            return 1
    else:
        # --bootstrap mode: fetch full config from server
        from beagle_node.config.remote import RemoteConfigFetcher, load_bootstrap
        try:
            bootstrap = load_bootstrap(args.bootstrap)
        except FileNotFoundError:
            print(f"ERROR: bootstrap file not found: {args.bootstrap}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR: invalid bootstrap config: {exc}", file=sys.stderr)
            return 1

        _remote_fetcher = RemoteConfigFetcher(bootstrap)
        try:
            config = _remote_fetcher.fetch_initial_config()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR: unexpected error fetching remote config: {exc}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    from beagle_node.utils.logging import configure_logging
    configure_logging(
        node_id=config.node_id,
        level=args.log_level,
        json_output=args.json_logs if args.json_logs else None,
    )
    logger.info("Beagle node starting", extra={"node_id": config.node_id,
                                                 "sdr_mode": config.sdr_mode})

    # ------------------------------------------------------------------
    # Health state (shared with reporter and pipeline loop)
    # ------------------------------------------------------------------
    from beagle_node.utils.health import HealthServer, HealthState
    health = HealthState(node_id=config.node_id)
    health.set_config(
        sync_station=config.sync_signal.primary_station.station_id,
        sync_freq_hz=config.sync_signal.primary_station.frequency_hz,
        target_channels=[
            {"frequency_hz": ch.frequency_hz, "label": ch.label}
            for ch in config.target_channels
        ],
        sdr_mode=config.sdr_mode,
    )

    if not args.no_health:
        health_srv = HealthServer(health, port=config.health_port)
        health_srv.start()
    else:
        health_srv = None

    # ------------------------------------------------------------------
    # Reporter
    # ------------------------------------------------------------------
    from beagle_node.events.reporter import EventReporter
    reporter = EventReporter(
        server_url=config.reporter.server_url,
        auth_token=config.reporter.auth_token,
        max_queue=config.reporter.max_queue_size,
        timeout_s=config.reporter.timeout_s,
    )
    reporter.start()

    # Heartbeat payload - updated in the SDR loop with live telemetry.
    # In bootstrap mode the config poll carries this data; in classic mode
    # it's sent via reporter.post_heartbeat().
    _heartbeat_payload: dict[str, object] = {
        "node_id": config.node_id,
        "latitude_deg": config.location.latitude_deg,
        "longitude_deg": config.location.longitude_deg,
        "sdr_mode": config.sdr_mode,
    }
    if _remote_fetcher is not None:
        # Seed the fetcher so the first config poll carries location/mode
        _remote_fetcher.set_heartbeat_data(_heartbeat_payload)
    else:
        # Classic mode: send initial heartbeat via reporter
        reporter.post_heartbeat(_heartbeat_payload)

    # In bootstrap mode, start background config poll after reporter is up.
    if _remote_fetcher is not None:
        def _on_config_update(new_config: NodeConfig) -> None:
            """
            Called by the remote config poll thread when a new version arrives.

            Hot-reloadable fields are applied in-place on the shared config
            object.  Fields that require an SDR restart trigger a clean
            self-restart (exit code 75) so systemd brings us back up with
            the new config.
            """
            need_restart = False

            # --- Restart-required: SDR hardware params ---
            # Compare the mode-specific SDR config blocks and sdr_mode itself.
            _restart_checks: list[tuple[str, object, object]] = [
                ("sdr_mode", config.sdr_mode, new_config.sdr_mode),
                ("freq_hop", config.freq_hop, new_config.freq_hop),
                ("rspduo", config.rspduo, new_config.rspduo),
                ("sync_sdr", config.sync_sdr, new_config.sync_sdr),
                ("target_sdr", config.target_sdr, new_config.target_sdr),
            ]
            # Also check sync frequency change (requires re-tuning the SDR)
            if (new_config.sync_signal.primary_station.frequency_hz
                    != config.sync_signal.primary_station.frequency_hz):
                _restart_checks.append((
                    "sync_signal.primary_station.frequency_hz",
                    config.sync_signal.primary_station.frequency_hz,
                    new_config.sync_signal.primary_station.frequency_hz,
                ))

            changed_fields: list[str] = []
            for field, old_val, new_val in _restart_checks:
                if old_val != new_val:
                    changed_fields.append(field)

            if changed_fields:
                need_restart = True
                logger.info(
                    "Remote config update: SDR config changed (%s) - "
                    "initiating automatic restart to apply",
                    ", ".join(changed_fields),
                )

            # --- Carrier config: split into hot-reload vs restart ---
            if new_config.carrier != config.carrier:
                # Fields that change ring buffer geometry require restart
                _carrier_restart_fields = (
                    "window_samples", "snippet_samples",
                    "snippet_post_windows", "ring_lookback_windows",
                )
                carrier_needs_restart = any(
                    getattr(new_config.carrier, f) != getattr(config.carrier, f)
                    for f in _carrier_restart_fields
                )
                if carrier_needs_restart:
                    need_restart = True
                    logger.info(
                        "Remote config update: carrier geometry changed - "
                        "initiating restart"
                    )
                # Threshold/debounce fields are hot-reloadable
                config.carrier = new_config.carrier
                try:
                    pipeline.carrier_detector.update_thresholds(
                        onset_threshold_db=new_config.carrier.onset_db,
                        offset_threshold_db=new_config.carrier.offset_db,
                        min_hold_windows=new_config.carrier.min_hold_windows,
                        min_release_windows=new_config.carrier.min_release_windows,
                        min_active_windows_for_offset=new_config.carrier.min_active_windows_for_offset,
                    )
                    logger.info(
                        "Remote config update: carrier thresholds applied "
                        "(onset=%.1f offset=%.1f hold=%d release=%d "
                        "min_active_for_offset=%d)",
                        new_config.carrier.onset_db, new_config.carrier.offset_db,
                        new_config.carrier.min_hold_windows,
                        new_config.carrier.min_release_windows,
                        new_config.carrier.min_active_windows_for_offset,
                    )
                except Exception as exc:
                    logger.error("Failed to update carrier thresholds: %s", exc)

            # --- Hot-reloadable: target channels ---
            if new_config.target_channels != config.target_channels:
                config.target_channels = new_config.target_channels
                logger.info(
                    "Remote config update: target_channels updated (%d channels)",
                    len(new_config.target_channels),
                )

            # --- Hot-reloadable: clock calibration ---
            if new_config.clock != config.clock:
                config.clock = new_config.clock
                logger.info(
                    "Remote config update: clock config applied "
                    "(source=%s offset_ns=%d)",
                    new_config.clock.source, new_config.clock.calibration_offset_ns,
                )

            # --- Hot-reloadable: sync signal thresholds (frequency handled above) ---
            if (new_config.sync_signal != config.sync_signal
                    and not need_restart):
                config.sync_signal = new_config.sync_signal
                logger.info("Remote config update: sync signal thresholds applied")

            # --- Hot-reloadable: node location ---
            # Mobile and relocated nodes need their broadcast position to
            # follow the config change immediately, without a process
            # restart.  Update both the in-memory config object AND the
            # _heartbeat_payload dict that the next config poll will send
            # back to the server.  Then push the updated payload to the
            # fetcher's snapshot so the very next poll carries it,
            # without waiting for the heartbeat-interval cycle in the
            # main SDR loop to do another set_heartbeat_data call.
            if new_config.location != config.location:
                old_loc = config.location
                config.location = new_config.location
                _heartbeat_payload["latitude_deg"] = new_config.location.latitude_deg
                _heartbeat_payload["longitude_deg"] = new_config.location.longitude_deg
                if _remote_fetcher is not None:
                    _remote_fetcher.set_heartbeat_data(_heartbeat_payload)
                logger.info(
                    "Remote config update: location applied "
                    "(%.6f, %.6f) -> (%.6f, %.6f)",
                    old_loc.latitude_deg, old_loc.longitude_deg,
                    new_config.location.latitude_deg,
                    new_config.location.longitude_deg,
                )

            # Update health endpoint with any config changes
            health.set_config(
                sync_station=config.sync_signal.primary_station.station_id,
                sync_freq_hz=config.sync_signal.primary_station.frequency_hz,
                target_channels=[
                    {"frequency_hz": ch.frequency_hz, "label": ch.label}
                    for ch in config.target_channels
                ],
                sdr_mode=config.sdr_mode,
            )

            # --- Trigger restart if needed ---
            if need_restart:
                logger.info(
                    "Exiting with code %d for automatic restart with new config",
                    EXIT_RESTART,
                )
                # Signal the main loop to stop, then the process exits
                # with EXIT_RESTART so systemd restarts us.
                _stop["flag"] = True
                _stop["restart"] = True
                # Stop the poll thread immediately so it doesn't fire
                # another 60-second long-poll while the main loop drains.
                _remote_fetcher.stop()
                try:
                    receiver.close()
                except Exception:
                    pass

        _remote_fetcher.start_poll(_on_config_update)

    # ------------------------------------------------------------------
    # SDR receiver
    # ------------------------------------------------------------------
    from beagle_node.sdr.factory import create_receiver
    from beagle_node.sdr.mock import MockReceiver

    if args.mock:
        from beagle_node.config.schema import SDRChannelConfig
        import numpy as np
        mock_config = config.target_channels[0] if config.target_channels else None
        from beagle_node.sdr.base import SDRConfig
        sdr_cfg = SDRConfig(
            center_frequency_hz=mock_config.frequency_hz if mock_config else 155_100_000.0,
            sample_rate_hz=2_048_000.0,
            gain_db=30.0,
        )
        receiver = MockReceiver.synthetic(
            config=sdr_cfg,
            duration_s=args.mock_duration,
            pilot_present=True,
            carrier_intervals=[(args.mock_duration * 0.3, args.mock_duration * 0.8)],
            snr_db=25.0,
            loop=False,
        )
        logger.info("Using MockReceiver (duration=%.1f s)", args.mock_duration)
    else:
        receiver = create_receiver(config)
        logger.info("Using real SDR receiver (mode=%s)", config.sdr_mode)
    health.set_config(sample_rate_hz=receiver.config.sample_rate_hz)

    # ------------------------------------------------------------------
    # Clock status (chrony)
    # ------------------------------------------------------------------
    from beagle_node.utils.chrony import ChronyStatus, read_chrony_status
    _clock: ChronyStatus = read_chrony_status()
    logger.info(
        "Initial clock status: source=%s uncertainty_ns=%d stratum=%d",
        _clock.source, _clock.rms_offset_ns, _clock.stratum,
    )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
    from beagle_node.events.model import CarrierEvent, NodeLocation, SyncTransmitter

    # Build CarrierEvent template from config for each measurement
    node_loc = NodeLocation(
        latitude_deg=config.location.latitude_deg,
        longitude_deg=config.location.longitude_deg,
        altitude_m=config.location.altitude_m,
    )
    sync_station = config.sync_signal.primary_station
    sync_tx = SyncTransmitter(
        station_id=sync_station.station_id,
        frequency_hz=sync_station.frequency_hz,
        latitude_deg=sync_station.latitude_deg,
        longitude_deg=sync_station.longitude_deg,
    )

    target_ch = config.target_channels[0] if config.target_channels else None

    # Per-mode pipeline offset (subtracted from every sync_delta_ns).
    # Each SDR mode corrects its own systematic delay relative to the
    # theoretical zero-delay reference (physical onset at antenna minus
    # physical pilot zero-crossing at antenna, no pipeline delay).
    _pipeline_offset_ns: int = 0
    if config.sdr_mode == "rspduo" and config.rspduo is not None:
        _pipeline_offset_ns = config.rspduo.pipeline_offset_ns
    elif config.sdr_mode == "freq_hop" and config.freq_hop is not None:
        _pipeline_offset_ns = config.freq_hop.pipeline_offset_ns

    # Wall-clock time (time.time_ns() epoch) of the most recent buffer pair
    # received from the SDR driver.  Updated by the rspduo/freq_hop loops before
    # each pipeline call; on_measurement() uses it instead of calling time.time_ns()
    # in the callback, which avoids GIL scheduling latency.  None until the
    # first buffer arrives; other SDR modes leave it None and fall back
    # to time.time_ns() inside the callback.
    buf_wall_ns: int | None = None

    # Raw ADC sample index corresponding to buf_wall_ns.
    #
    # For RSPduo (HAS_TIME): buf_wall_ns is the hardware timestamp of the first
    # sample of the buffer pair -> _buf_ref_sample = sample_count (buffer start).
    #
    # For RSPduo (no HAS_TIME): buf_wall_ns = time.time_ns() after readStream()
    # returns ~ time of last sample of the buffer.  We still use sample_count
    # (buffer start) here; this introduces a ~0.5 buffer systematic offset
    # (~16 ms) that cancels between co-located nodes.
    #
    # For freq_hop: buf_wall_ns = time.time_ns() immediately after read_bytes()
    # returns in the background thread ~ time of the LAST raw sample of the
    # block (including settling region).  The block ends at raw ADC position
    # adc_pos + block_n, so _buf_ref_sample is set to that value.
    _buf_ref_sample: int = 0

    def on_measurement(m) -> None:
        if target_ch is None:
            return
        if buf_wall_ns is not None:
            # Compute precise within-buffer onset position.
            # m.target_sample is in *sync-decimated* space (sample_rate/8 = 256 kHz):
            # process_target_buffer maps carrier events from target space (/32) back
            # to raw (x32) then to sync space (/8) before passing to DeltaComputer.
            # Convert to raw (x8) and compute the signed offset from _buf_ref_sample
            # (the raw ADC position corresponding to buf_wall_ns).  Negative means
            # the event occurred before the reference point (freq_hop case); positive
            # means after (RSPduo HAS_TIME case where buf_wall_ns is buffer start).
            raw_event_sample = m.target_sample * _sync_dec_factor
            onset_offset_raw = raw_event_sample - _buf_ref_sample
            onset_offset_ns = int(onset_offset_raw * 1e9 / receiver.config.sample_rate_hz)
            onset_ns = buf_wall_ns + onset_offset_ns
            logger.debug(
                "onset timing: mode=%s target_sample=%d raw_event=%d ref=%d "
                "offset_raw=%d offset_ns=%d buf_wall_age_ms=%.1f onset_ns=%d",
                config.sdr_mode, m.target_sample, raw_event_sample, _buf_ref_sample,
                onset_offset_raw, onset_offset_ns,
                (time.time_ns() - buf_wall_ns) / 1e6, onset_ns,
            )
        else:
            onset_ns = int(time.time_ns())
        onset_ns -= config.clock.calibration_offset_ns
        corrected_delta = m.sync_delta_ns - _pipeline_offset_ns
        event = CarrierEvent(
            node_id=config.node_id,
            node_location=node_loc,
            channel_frequency_hz=target_ch.frequency_hz,
            sync_delta_ns=corrected_delta,
            sync_transmitter=sync_tx,
            sdr_mode=config.sdr_mode,
            pps_anchored=m.pps_anchored,
            event_type=m.event_type,
            onset_time_ns=onset_ns,
            peak_power_db=m.onset_power_db,
            noise_floor_db=m.noise_floor_db,
            sync_corr_peak=m.corr_peak,
            clock_source=_clock.source,
            clock_uncertainty_ns=_clock.rms_offset_ns,
            node_software_version="0.1.0",
            iq_snippet_b64=base64.b64encode(m.iq_snippet).decode(),
            channel_sample_rate_hz=_target_sample_rate_hz,
        )
        reporter.submit(event)
        health.record_event()
        logger.info(
            "Measurement: %s sync_delta_ns=%d corr=%.3f",
            m.event_type, corrected_delta, m.corr_peak,
        )

    # Convert max_sync_age_ms -> samples in the sync-decimated domain.
    # sync_decimation is fixed at 8x in PipelineConfig (2 MHz -> 256/250 kHz).
    _sync_dec_factor = 8
    _max_sync_age_samples = int(
        config.sync_signal.max_sync_age_ms / 1000.0
        * receiver.config.sample_rate_hz
        / _sync_dec_factor
    )

    # Target channel decimated sample rate - used to annotate CarrierEvents so
    # the server cross-correlator can convert lag samples to nanoseconds.
    # PipelineConfig.target_decimation is fixed at 32 (2 MSPS -> 64 kHz / 62.5 kHz).
    _target_sample_rate_hz: float = receiver.config.sample_rate_hz / 32

    pipeline = NodePipeline(
        config=PipelineConfig(
            sdr_rate_hz=receiver.config.sample_rate_hz,
            min_corr_peak=config.sync_signal.min_corr_peak,
            max_sync_age_samples=_max_sync_age_samples,
            carrier_onset_db=config.carrier.onset_db,
            carrier_offset_db=config.carrier.offset_db,
            carrier_window_samples=config.carrier.window_samples,
            carrier_min_hold_windows=config.carrier.min_hold_windows,
            carrier_min_release_windows=config.carrier.min_release_windows,
            carrier_snippet_samples=config.carrier.snippet_samples,
            carrier_snippet_post_windows=config.carrier.snippet_post_windows,
            carrier_ring_lookback_windows=config.carrier.ring_lookback_windows,
            carrier_min_active_windows_for_offset=config.carrier.min_active_windows_for_offset,
        ),
        on_measurement=on_measurement,
    )

    # ------------------------------------------------------------------
    # Shutdown handling
    # ------------------------------------------------------------------
    _stop: dict[str, bool] = {"flag": False, "restart": False}

    def _handle_signal(sig, frame):
        logger.info("Shutdown signal received (%s)", signal.Signals(sig).name)
        _stop["flag"] = True
        # Close the receiver so any blocking stdout.read() in FreqHopReceiver
        # (or similar) returns immediately, allowing the main loop to exit.
        try:
            receiver.close()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    logger.info("Entering main SDR loop")
    exit_code = 0
    sample_count = 0
    _last_chrony_refresh: float = 0.0
    _CHRONY_INTERVAL_S: float = 30.0
    _last_heartbeat: float = 0.0
    _HEARTBEAT_INTERVAL_S: float = 30.0

    try:
        with receiver:
            if config.sdr_mode == "freq_hop" and hasattr(receiver, "labeled_stream"):
                # Single ADC alternates sync/target blocks.  Track the running
                # ADC sample position so both channels share the same continuous
                # sample-index space, which DeltaComputer needs for sync_delta_ns.
                # With asymmetric blocks the two channels have different sizes, so
                # a simple block_count * fixed_block formula is incorrect; we must
                # accumulate the actual block sizes seen.
                adc_pos = 0  # ADC sample index of the start of the current block
                for role, iq_buf, buf_wall_ns in receiver.labeled_stream():
                    if _stop["flag"]:
                        break
                    block_n = receiver.sync_block_samples if role == "sync" else receiver.target_block_samples
                    raw_start = adc_pos + receiver.settling_samples
                    if role == "sync":
                        pipeline.process_sync_buffer(iq_buf, raw_start_sample=raw_start)
                    else:
                        # buf_wall_ns is captured after read_bytes() returns, i.e.
                        # approximately when the last raw sample of this block
                        # (including settling region) was received.  The last raw
                        # sample sits at raw ADC position adc_pos + block_n.
                        _buf_ref_sample = adc_pos + block_n
                        logger.debug(
                            "freq_hop target block: adc_pos=%d block_n=%d "
                            "ref=%d buf_wall_age_ms=%.1f",
                            adc_pos, block_n, _buf_ref_sample,
                            (time.time_ns() - buf_wall_ns) / 1e6,
                        )
                        pipeline.process_target_buffer(
                            iq_buf, raw_start_sample=raw_start, new_target_block=True,
                        )
                    adc_pos += block_n
                    sample_count += len(iq_buf)

                    # Refresh chrony + update health every ~1 second worth of samples
                    if sample_count % (int(receiver.config.sample_rate_hz) or 2_048_000) < len(iq_buf):
                        now = time.monotonic()
                        if now - _last_chrony_refresh >= _CHRONY_INTERVAL_S:
                            _clock = read_chrony_status()
                            _last_chrony_refresh = now
                        if now - _last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                            _heartbeat_payload["clock_source"] = _clock.source
                            _heartbeat_payload["noise_floor_db"] = pipeline.carrier_detector.noise_floor_db
                            _heartbeat_payload["onset_threshold_db"] = pipeline.carrier_detector.onset_threshold_db
                            _heartbeat_payload["offset_threshold_db"] = pipeline.carrier_detector.offset_threshold_db
                            if _remote_fetcher is not None:
                                _remote_fetcher.set_heartbeat_data(_heartbeat_payload)
                            else:
                                reporter.post_heartbeat(_heartbeat_payload)
                            _last_heartbeat = now
                            gc.collect(0)
                        health.update(
                            events_submitted=reporter.events_submitted,
                            events_dropped=reporter.events_dropped,
                            queue_depth=reporter.queue_depth,
                            crystal_correction=pipeline.latest_sample_rate_correction,
                            sdr_overflows=receiver.overflow_count,
                            backlog_drains=receiver.backlog_drain_count,
                            clock_source=_clock.source,
                            clock_uncertainty_ns=_clock.rms_offset_ns,
                            sync_event_count=pipeline.sync_event_count,
                            noise_floor_db=pipeline.carrier_detector.noise_floor_db,
                            onset_threshold_db=pipeline.carrier_detector.onset_threshold_db,
                            offset_threshold_db=pipeline.carrier_detector.offset_threshold_db,
                            sync_corr_peak=pipeline.latest_corr_peak,
                        )
            elif config.sdr_mode == "rspduo" and hasattr(receiver, "paired_stream"):
                # RSPduo dual-tuner: master=sync, slave=target.  Both buffers
                # share the same ADC clock, so we pass the same sample position
                # as raw_start_sample for both channels.
                for sync_buf, target_buf, buf_wall_ns in receiver.paired_stream():
                    if _stop["flag"]:
                        break
                    # _buf_ref_sample = buffer start (raw ADC position).
                    # buf_wall_ns (HAS_TIME) is the hardware timestamp of the
                    # first sample of this buffer pair, which is sample_count.
                    _buf_ref_sample = sample_count
                    pipeline.process_sync_buffer(sync_buf, raw_start_sample=sample_count)
                    pipeline.process_target_buffer(target_buf, raw_start_sample=sample_count)
                    sample_count += len(sync_buf)

                    if sample_count % (int(receiver.config.sample_rate_hz) or 2_000_000) < len(sync_buf):
                        now = time.monotonic()
                        if now - _last_chrony_refresh >= _CHRONY_INTERVAL_S:
                            _clock = read_chrony_status()
                            _last_chrony_refresh = now
                        if now - _last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                            _heartbeat_payload["clock_source"] = _clock.source
                            _heartbeat_payload["noise_floor_db"] = pipeline.carrier_detector.noise_floor_db
                            _heartbeat_payload["onset_threshold_db"] = pipeline.carrier_detector.onset_threshold_db
                            _heartbeat_payload["offset_threshold_db"] = pipeline.carrier_detector.offset_threshold_db
                            if _remote_fetcher is not None:
                                _remote_fetcher.set_heartbeat_data(_heartbeat_payload)
                            else:
                                reporter.post_heartbeat(_heartbeat_payload)
                            _last_heartbeat = now
                            # Proactively collect gen-0 cycles every heartbeat
                            # interval (~30 s) so Pydantic model cycles are
                            # broken in small batches rather than accumulating
                            # into one large pause that can trigger backlog drain.
                            gc.collect(0)
                        health.update(
                            events_submitted=reporter.events_submitted,
                            events_dropped=reporter.events_dropped,
                            queue_depth=reporter.queue_depth,
                            crystal_correction=pipeline.latest_sample_rate_correction,
                            sdr_overflows=receiver.overflow_count,
                            backlog_drains=receiver.backlog_drain_count,
                            clock_source=_clock.source,
                            clock_uncertainty_ns=_clock.rms_offset_ns,
                            sync_event_count=pipeline.sync_event_count,
                            noise_floor_db=pipeline.carrier_detector.noise_floor_db,
                            onset_threshold_db=pipeline.carrier_detector.onset_threshold_db,
                            offset_threshold_db=pipeline.carrier_detector.offset_threshold_db,
                            sync_corr_peak=pipeline.latest_corr_peak,
                        )

            else:
                # single_sdr / two_sdr / mock: single stream covers both roles.
                for iq_buf in receiver.stream():
                    if _stop["flag"]:
                        break
                    pipeline.process_sync_buffer(iq_buf)
                    pipeline.process_target_buffer(iq_buf)
                    sample_count += len(iq_buf)

                    # Refresh chrony + update health every ~1 second worth of samples
                    if sample_count % (int(receiver.config.sample_rate_hz) or 2_048_000) < len(iq_buf):
                        now = time.monotonic()
                        if now - _last_chrony_refresh >= _CHRONY_INTERVAL_S:
                            _clock = read_chrony_status()
                            _last_chrony_refresh = now
                        if now - _last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                            _heartbeat_payload["clock_source"] = _clock.source
                            _heartbeat_payload["noise_floor_db"] = pipeline.carrier_detector.noise_floor_db
                            _heartbeat_payload["onset_threshold_db"] = pipeline.carrier_detector.onset_threshold_db
                            _heartbeat_payload["offset_threshold_db"] = pipeline.carrier_detector.offset_threshold_db
                            if _remote_fetcher is not None:
                                _remote_fetcher.set_heartbeat_data(_heartbeat_payload)
                            else:
                                reporter.post_heartbeat(_heartbeat_payload)
                            _last_heartbeat = now
                        health.update(
                            events_submitted=reporter.events_submitted,
                            events_dropped=reporter.events_dropped,
                            queue_depth=reporter.queue_depth,
                            crystal_correction=pipeline.latest_sample_rate_correction,
                            sdr_overflows=receiver.overflow_count,
                            clock_source=_clock.source,
                            clock_uncertainty_ns=_clock.rms_offset_ns,
                            sync_event_count=pipeline.sync_event_count,
                            noise_floor_db=pipeline.carrier_detector.noise_floor_db,
                            onset_threshold_db=pipeline.carrier_detector.onset_threshold_db,
                            offset_threshold_db=pipeline.carrier_detector.offset_threshold_db,
                            sync_corr_peak=pipeline.latest_corr_peak,
                        )

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt - shutting down")
    except Exception as exc:
        logger.exception("Fatal error in main loop: %s", exc)
        exit_code = 1

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    if _stop.get("restart"):
        exit_code = EXIT_RESTART
        logger.info("Restarting node to apply new SDR config (exit code %d)", exit_code)
    logger.info("Stopping reporter and health server")
    reporter.stop(timeout_s=5.0)
    if _remote_fetcher is not None:
        _remote_fetcher.stop()
    if health_srv:
        health_srv.stop()

    logger.info("Beagle node stopped (exit_code=%d)", exit_code)
    return exit_code


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
