#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Profile the Beagle node signal-processing pipeline.

Generates synthetic IQ data (noise + injected FM pilot + carrier bursts)
and feeds it through the full NodePipeline, measuring wall-clock time
per stage.  No SDR hardware required.

Usage
-----
    # Quick 5-second run with per-stage timing:
    python scripts/profile_pipeline.py

    # Longer run for stable numbers:
    python scripts/profile_pipeline.py --duration 30

    # Full cProfile dump (sortable flamegraph-ready):
    python scripts/profile_pipeline.py --duration 10 --cprofile profile_out.prof

    # With py-spy (run separately):
    #   py-spy record -o flame.svg - python scripts/profile_pipeline.py --duration 30

The script prints a per-stage breakdown showing where wall-clock time is
spent, so you can identify the dominant hotspots without external tooling.
"""

from __future__ import annotations

import argparse
import cProfile
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beagle_node.pipeline.pipeline import NodePipeline, PipelineConfig
from beagle_node.pipeline.decimator import Decimator
from beagle_node.pipeline.demodulator import FMDemodulator
from beagle_node.pipeline.sync_detector import FMPilotSyncDetector
from beagle_node.pipeline.carrier_detect import CarrierDetector


# ---------------------------------------------------------------------------
# Synthetic IQ generation
# ---------------------------------------------------------------------------

def make_fm_pilot_iq(
    n_samples: int,
    sdr_rate: float = 2_048_000.0,
    pilot_freq: float = 19_000.0,
    deviation_hz: float = 7500.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generate IQ with an FM-modulated 19 kHz pilot tone + noise.

    This simulates what the sync channel sees: a strong FM broadcast
    station with a stereo pilot subcarrier.
    """
    t = np.arange(n_samples, dtype=np.float64) / sdr_rate
    # FM modulation: carrier + pilot tone deviation
    phase = 2.0 * np.pi * (deviation_hz / pilot_freq) * np.sin(2.0 * np.pi * pilot_freq * t)
    iq = np.exp(1j * phase).astype(np.complex64)
    # Add noise
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    iq += noise_level * noise
    return iq


def make_target_iq(
    n_samples: int,
    sdr_rate: float = 2_048_000.0,
    carrier_freq_offset: float = 5_000.0,
    carrier_power: float = 0.5,
    burst_on_frac: float = 0.3,
    noise_level: float = 0.001,
) -> np.ndarray:
    """Generate IQ for the target channel: noise with periodic carrier bursts.

    Simulates an LMR transmitter keying up for burst_on_frac of the time.
    """
    t = np.arange(n_samples, dtype=np.float64) / sdr_rate
    iq = noise_level * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)

    # Add carrier bursts
    burst_len = int(n_samples * burst_on_frac)
    burst_start = n_samples // 4  # start 25% in
    burst_end = min(burst_start + burst_len, n_samples)
    carrier = carrier_power * np.exp(1j * 2.0 * np.pi * carrier_freq_offset * t[burst_start:burst_end]).astype(np.complex64)
    iq[burst_start:burst_end] += carrier

    return iq


# ---------------------------------------------------------------------------
# Isolated stage benchmarks
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates wall-clock time for named stages."""

    def __init__(self):
        self._times: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._start: float = 0.0

    def start(self, name: str):
        self._start = time.perf_counter()

    def stop(self, name: str):
        elapsed = time.perf_counter() - self._start
        self._times[name] = self._times.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def report(self, total_iq_seconds: float):
        total = sum(self._times.values())
        print(f"\n{'Stage':<35s} {'Time (s)':>10s} {'%':>6s} {'Calls':>8s} {'usec/call':>10s}")
        print("-" * 75)
        for name in sorted(self._times, key=lambda k: -self._times[k]):
            t = self._times[name]
            c = self._counts[name]
            pct = 100.0 * t / total if total > 0 else 0
            us_per = 1e6 * t / c if c > 0 else 0
            print(f"  {name:<33s} {t:10.4f} {pct:6.1f} {c:8d} {us_per:10.1f}")
        print("-" * 75)
        print(f"  {'TOTAL':<33s} {total:10.4f}")
        print(f"\n  IQ duration processed: {total_iq_seconds:.1f} s")
        print(f"  Realtime ratio: {total_iq_seconds / total:.1f}x  "
              f"({'OK' if total_iq_seconds / total > 1.0 else 'TOO SLOW'})")
        print(f"  Pipeline CPU load estimate: {100.0 * total / total_iq_seconds:.0f}% of one core")


def run_profiled(duration_s: float = 5.0, buffer_samples: int = 65536):
    """Run the full pipeline with per-stage timing."""
    sdr_rate = 2_048_000.0
    total_samples = int(duration_s * sdr_rate)
    n_buffers = total_samples // buffer_samples

    print(f"Generating {duration_s:.0f}s of synthetic IQ ({n_buffers} buffers x "
          f"{buffer_samples} samples @ {sdr_rate/1e6:.3f} MSPS)...")

    # Pre-generate all buffers to exclude generation time from profiling
    sync_bufs = []
    target_bufs = []
    for i in range(n_buffers):
        sync_bufs.append(make_fm_pilot_iq(buffer_samples, sdr_rate))
        target_bufs.append(make_target_iq(buffer_samples, sdr_rate))

    print(f"Generated {n_buffers * 2} buffers. Starting pipeline benchmark...\n")

    # ---- Full pipeline benchmark ----
    measurements = []
    config = PipelineConfig(sdr_rate_hz=sdr_rate)
    pipeline = NodePipeline(config=config, on_measurement=measurements.append)

    timer = StageTimer()
    sync_sample = 0
    target_sample = 0

    t0 = time.perf_counter()
    for i in range(n_buffers):
        timer.start("sync_total")
        pipeline.process_sync_buffer(sync_bufs[i], raw_start_sample=sync_sample)
        timer.stop("sync_total")
        sync_sample += buffer_samples

        timer.start("target_total")
        pipeline.process_target_buffer(target_bufs[i], raw_start_sample=target_sample)
        timer.stop("target_total")
        target_sample += buffer_samples
    t_full = time.perf_counter() - t0

    iq_seconds = n_buffers * buffer_samples / sdr_rate
    print(f"=== Full Pipeline ===")
    print(f"  Wall time: {t_full:.3f}s for {iq_seconds:.1f}s of IQ")
    print(f"  Realtime ratio: {iq_seconds / t_full:.1f}x")
    print(f"  Measurements produced: {len(measurements)}")
    timer.report(iq_seconds)

    # ---- Isolated stage benchmarks ----
    print(f"\n\n=== Isolated Stage Benchmarks ({iq_seconds:.0f}s of IQ) ===")
    iso_timer = StageTimer()

    # 1. Sync decimator
    sync_dec = Decimator(8, sdr_rate, 128_000.0)
    for buf in sync_bufs:
        iso_timer.start("1. sync_decimator")
        sync_dec.process(buf)
        iso_timer.stop("1. sync_decimator")
    sync_dec.reset()

    # 2. Target decimator
    target_dec = Decimator(32, sdr_rate, 25_000.0)
    for buf in target_bufs:
        iso_timer.start("2. target_decimator")
        target_dec.process(buf)
        iso_timer.stop("2. target_decimator")
    target_dec.reset()

    # 3. DC removal
    for buf in target_bufs:
        iso_timer.start("3. dc_removal")
        _ = buf - np.mean(buf)
        iso_timer.stop("3. dc_removal")

    # 4. FM demodulator
    demod = FMDemodulator(sdr_rate / 8)
    dec_bufs_sync = [Decimator(8, sdr_rate, 128_000.0).process(b) for b in sync_bufs[:1]]
    sync_dec2 = Decimator(8, sdr_rate, 128_000.0)
    dec_sync_all = [sync_dec2.process(b) for b in sync_bufs]
    for dbuf in dec_sync_all:
        iso_timer.start("4. fm_demod")
        demod.process(dbuf)
        iso_timer.stop("4. fm_demod")

    # 5. Sync detector (BPF + xcorr)
    sync_det = FMPilotSyncDetector(sdr_rate / 8, sync_period_ms=7.0)
    demod2 = FMDemodulator(sdr_rate / 8)
    audio_bufs = [demod2.process(d) for d in dec_sync_all]
    sample_idx = 0
    for abuf in audio_bufs:
        iso_timer.start("5. sync_detector")
        sync_det.process(abuf, start_sample=sample_idx)
        iso_timer.stop("5. sync_detector")
        sample_idx += len(abuf)

    # 6. Carrier detector
    target_dec2 = Decimator(32, sdr_rate, 25_000.0)
    dec_target_all = [target_dec2.process(b - np.mean(b)) for b in target_bufs]
    carrier_det = CarrierDetector(
        sample_rate_hz=sdr_rate / 32,
        onset_threshold_db=-30.0,
        offset_threshold_db=-40.0,
        window_samples=64,
    )
    target_idx = 0
    for dbuf in dec_target_all:
        iso_timer.start("6. carrier_detector")
        carrier_det.process(dbuf, start_sample=target_idx)
        iso_timer.stop("6. carrier_detector")
        target_idx += len(dbuf)

    # 7. np.asarray overhead (simulate the copy)
    for buf in sync_bufs:
        iso_timer.start("7. asarray_copy")
        np.asarray(buf, dtype=np.complex64)
        iso_timer.stop("7. asarray_copy")

    iso_timer.report(iq_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Profile the Beagle signal-processing pipeline.",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration of synthetic IQ to process (seconds, default: 5).",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=65536,
        help="Samples per buffer (default: 65536, ~32ms at 2.048 MSPS).",
    )
    parser.add_argument(
        "--cprofile", metavar="FILE",
        help="Save cProfile output to FILE for external analysis "
             "(e.g. snakeviz, flameprof).",
    )
    args = parser.parse_args()

    if args.cprofile:
        print(f"Running with cProfile -> {args.cprofile}")
        profiler = cProfile.Profile()
        profiler.enable()
        run_profiled(duration_s=args.duration, buffer_samples=args.buffer_size)
        profiler.disable()
        profiler.dump_stats(args.cprofile)
        print(f"\ncProfile data saved to {args.cprofile}")
        print(f"View with: python -m snakeviz {args.cprofile}")
        print(f"  or:      python -m flameprof {args.cprofile} > flame.svg")
    else:
        run_profiled(duration_s=args.duration, buffer_samples=args.buffer_size)


if __name__ == "__main__":
    main()
