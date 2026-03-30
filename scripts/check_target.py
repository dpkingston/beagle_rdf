#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
Measure received signal power at a target frequency to calibrate carrier
detection thresholds.

Tunes the SDR to the target frequency and reports received power in dBFS
once per second.  Key your transmitter a few times during the run.  At the
end the script reports the observed noise floor and peak signal level and
recommends --onset-db / --offset-db values for verify_freq_hop.py /
verify_rspduo.py and carrier_onset_db / carrier_offset_db values for node.yaml.

Threshold philosophy
--------------------
Thresholds are set 10 dB (onset) and 6 dB (offset) above the measured noise
floor, not at the midpoint between noise and peak.  This keeps them low enough
to detect the carrier in freq_hop mode, where the effective signal level during
the short usable window after each frequency hop can be 15-25 dB below the
steady-state level measured here (due to R820T gain settling after each hop).
A midpoint formula fails at high SNR because it places the threshold too far
into the dynamic range, above where the freq_hop pipeline can observe.

No frequency hopping or FM sync is involved -- this is a pure receive test.

Offset tuning (--freq-offset)
------------------------------
RTL-SDR and other direct-conversion SDRs have a strong LO leakage spike at
exactly DC (0 Hz in baseband = the tuned frequency).  When the SDR is tuned
to exactly the target frequency the LO spike contaminates the measurement.

The default --freq-offset 200000 tunes the hardware 200 kHz ABOVE the
requested frequency.  The target signal then appears at -200 kHz in baseband
(well away from DC), and power is measured in a 50 kHz band around that
offset using an FFT.  The LO spike stays at DC and is excluded.

Set --freq-offset 0 to disable (useful for SDRs with good DC rejection, e.g.
RSPduo via SoapySDRPlay3 which applies hardware DC notch).

RSPduo note
-----------
The RSPduo has two independent tuners on separate antenna ports:
  Tuner 1 / Antenna A (channel 0) -- used for FM sync signal
  Tuner 2 / Antenna C (channel 1) -- used for LMR target signal

To measure the target channel through the correct antenna port (Antenna C),
use --channel 1 with --device "driver=sdrplay".  The script selects the
"Dual Tuner (independent RX)" mode automatically, configures both channels,
and reads from channel 1.

Do NOT use --channel 0 (the default) with --device "driver=sdrplay".
Channel 0 is the FM sync tuner; tuning it to a UHF target frequency reads
from Antenna A (the FM antenna) and you will see poor SNR or noise
regardless of the signal level.

Usage examples
--------------
# RTL-SDR / freq_hop node (uses 200 kHz offset by default to avoid LO spike)
python3 scripts/check_target.py --freq 462.5625e6 --gain 40

# RSPduo -- read from Tuner 2 / Antenna C (the LMR target port)
# Use --freq-offset 100000 (100 kHz) to move the signal off DC;
# the RSPduo DC notch may not be active in independent-RX mode.
python3 scripts/check_target.py --freq 462.5625e6 --gain 30 \\
    --device "driver=sdrplay" --channel 1 --freq-offset 100000 --rate 2000000

# Longer run
python3 scripts/check_target.py --freq 462.5625e6 --gain 40 --duration 60
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np


BAR_WIDTH = 30    # characters for the ASCII power bar
BAR_RANGE = 60.0  # dBFS span shown in bar (e.g. -80 to -20)
BAR_MAX   = -20.0 # right edge of bar

# Bandwidth of the frequency-domain window used to measure signal power.
# 50 kHz covers a 25 kHz LMR channel with margin; wide enough to capture
# a slow-settling signal while rejecting most out-of-band noise.
MEASURE_BW_HZ = 50_000.0

# Placeholder frequency for the unused DT channel (channel 0 when measuring
# channel 1).  Must be a valid tunable frequency; value does not matter for
# the measurement.
_DT_UNUSED_CHANNEL_FREQ_HZ = 99_900_000.0  # 99.9 MHz (FM band)


class _Done(Exception):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure received signal power to calibrate carrier detection thresholds"
    )
    p.add_argument("--freq",        type=float, required=True, metavar="HZ",
                   help="Target receive frequency (e.g. 462.5625e6)")
    p.add_argument("--rate",        type=float, default=2_048_000.0, metavar="SPS",
                   help="Sample rate (default 2.048e6)")
    p.add_argument("--gain",        type=float, default=30.0, metavar="DB",
                   help="Receiver gain in dB (default 30)")
    p.add_argument("--device",      type=str,   default="",  metavar="ARGS",
                   help="SoapySDR device args (default: first available). "
                        "For RSPduo use \"driver=sdrplay\" with --channel 1.")
    p.add_argument("--channel",     type=int,   default=0,   metavar="N",
                   help="SoapySDR RX channel to read from (default 0). "
                        "Use 1 with --device \"driver=sdrplay\" for RSPduo Tuner 2 / Antenna C.")
    p.add_argument("--duration",    type=float, default=30.0, metavar="SEC",
                   help="Run duration in seconds (default 30)")
    p.add_argument("--window-ms",   type=float, default=500.0, metavar="MS",
                   help="Power measurement window in ms (default 500)")
    p.add_argument("--freq-offset", type=float, default=200_000.0, metavar="HZ",
                   help="Tune this many Hz above --freq; signal appears at -OFFSET in "
                        "baseband; power measured via FFT excluding DC. "
                        "Default 200000 avoids LO leakage on RTL-SDR. "
                        "Use 0 for SDRs with hardware DC rejection (e.g. RSPduo).")
    p.add_argument("--ch0-freq", type=float, default=None, metavar="HZ",
                   help="RSPduo only: frequency for the drain channel (ch0). "
                        "Default: 99.9 MHz (FM band). "
                        "Set to --freq to test with both channels at the same quiet "
                        "frequency (useful for isolating cross-channel contamination).")
    return p.parse_args()


def _power_bar(db: float) -> str:
    """Return an ASCII bar representing `db` dBFS."""
    bar_min = BAR_MAX - BAR_RANGE
    filled = int(BAR_WIDTH * max(0.0, min(1.0, (db - bar_min) / BAR_RANGE)))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _measure_power_db(window: np.ndarray, sample_rate: float, freq_offset: float) -> float:
    """
    Return channel power in dBFS.

    If freq_offset != 0 (offset-tuned mode):
      The target signal is at -freq_offset Hz in baseband.
      Use FFT to measure power in MEASURE_BW_HZ around that offset,
      completely avoiding the LO leakage spike at DC.

    If freq_offset == 0 (direct-tune mode):
      The target is at DC.  Subtract the window mean to remove the
      LO spike, then measure total band power.  Less accurate on RTL-SDR
      but works on hardware with good DC rejection.
    """
    # Always use narrowband FFT so out-of-band signals don't contaminate the
    # measurement.  When freq_offset != 0 the signal sits at -freq_offset Hz;
    # when freq_offset == 0 the signal sits at DC (0 Hz) and hardware DC notch
    # (RSPduo) removes LO leakage.  Total-power measurement is intentionally
    # avoided: at UHF a 2 MHz passband can contain many interferers that would
    # make the reading look saturated even when the target signal is absent.
    N = len(window)
    fft = np.fft.fft(window)
    freqs = np.fft.fftfreq(N, 1.0 / sample_rate)
    signal_hz = -freq_offset  # 0.0 when direct-tuned
    mask = np.abs(freqs - signal_hz) <= MEASURE_BW_HZ / 2.0
    # Power = sum(|X[k]|^2) / N^2  (Parseval, normalised by N^2 for dBFS)
    pwr = float(np.sum(np.abs(fft[mask]) ** 2)) / (N ** 2)

    return 10.0 * np.log10(pwr + 1e-30)


def _setup_channel(sdr, SoapySDR, ch: int, freq_hz: float, rate: float, gain: float) -> None:
    """Apply sample rate, frequency, and manual gain to one SoapySDR channel."""
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, ch, rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, ch, freq_hz)
    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, ch, False)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, ch, gain)


def main() -> int:
    args = parse_args()

    try:
        import SoapySDR
    except ImportError:
        print("ERROR: SoapySDR not found.", file=sys.stderr)
        return 1

    is_rspduo = "driver=sdrplay" in args.device

    # Sanity checks for common mistakes
    if not is_rspduo and args.channel != 0:
        print(
            f"ERROR: --channel {args.channel} requires an RSPduo (driver=sdrplay).\n"
            f"  In single-tuner mode only channel 0 exists.\n"
            f"  For RSPduo Tuner 2 / Antenna C: --device \"driver=sdrplay\" --channel 1",
            file=sys.stderr,
        )
        return 1

    if is_rspduo and args.channel == 0:
        print(
            "ERROR: --channel 0 on the RSPduo is the FM/sync tuner (Tuner 1 / Antenna A).\n"
            "  To measure the LMR target signal through Tuner 2 / Antenna C use:\n"
            "    --channel 1 --freq-offset 0",
            file=sys.stderr,
        )
        return 1

    tune_freq = args.freq + args.freq_offset

    if is_rspduo:
        ch0_freq = args.ch0_freq if args.ch0_freq is not None else _DT_UNUSED_CHANNEL_FREQ_HZ

        # Open in DT (Dual Tuner) mode.  The rspduo-dual-independent-tuners
        # branch of SoapySDRPlay3 adds independent per-channel tuning to
        # mode=DT.  Use two single-channel streams -- same as RSPduoReceiver.
        # Use string form for device args -- SoapySDR 0.8.1-5 (Debian 13)
        # has a bug in Device.make() with dict kwargs (hash collision).
        device_parts = ["driver=sdrplay", "mode=DT"]
        for part in args.device.split(","):
            if part.startswith("serial="):
                device_parts.append(part.strip())
                break
        sdr = SoapySDR.Device(",".join(device_parts))
        # Channel 0 = sync/drain (FM).  Both channels must stream simultaneously
        # (TDM ADC).  A strong local FM broadcast at ch0's frequency can drive
        # the shared ADC near full scale and contaminate ch1's interleaved sample
        # slots.  Set ch0 to maximum IF attenuation (IFGR=59 gRdB) and minimum
        # LNA gain (RFGR=0) to prevent this.
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, args.rate)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, ch0_freq)
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
        # gRdB is IF Gain Reduction: gRdB=59 → maximum attenuation → minimum IF gain.
        # The unnamed setGain(0) clips to the IFGR range minimum (20), which is
        # near-maximum IF gain and saturates the shared TDM ADC.  Use named elements.
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR", 59)  # max IF attenuation (gRdB=59)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR", 9)   # max LNA attenuation (state 9)
        # Channel 1 = measurement channel.  Avoid unnamed setGain — its
        # distribution across RFGR and IFGR is unreliable for RSPduo in
        # DT mode.  Compute IFGR directly: IFGR = 79 - gain (clipped to
        # [20, 59]) so that higher --gain values mean more IF sensitivity.
        ch1_ifgr = max(20, min(59, 79 - int(args.gain)))
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 1, args.rate)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 1, tune_freq)
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 1, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 1, "IFGR", ch1_ifgr)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 1, "RFGR", 0)

        # Readback: verify what the driver actually programmed into each channel.
        # Note: getGain() with no name always returns 0 in SoapySDRPlay3 (driver
        # falls through to return 0 for unknown names).  Use named elements instead.
        for ch in (0, 1):
            actual_f = sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, ch)
            actual_r = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, ch)
            ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, ch, "IFGR")  # gRdB (reduction)
            rfgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, ch, "RFGR")  # LNA state
            actual_ant = sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, ch)
            avail_ants = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, ch)
            print(f"  [diag] ch{ch}: freq={actual_f/1e6:.4f} MHz  rate={actual_r/1e6:.3f} Msps  "
                  f"IFGR={ifgr:.0f} dB  RFGR(LNA)={rfgr:.0f}  "
                  f"antenna={actual_ant!r}  available={avail_ants}")
        print(f"  [diag] ch0 requested: {ch0_freq/1e6:.4f} MHz")
        print(f"  [diag] ch1 requested: {tune_freq/1e6:.4f} MHz")

        rx_unused = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        rx        = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [1])
        # Enable FM notch on Tuner B (ch1) BEFORE activateStream so that
        # sdrplay_api_Init() picks up rxChannelB->rfNotchEnable=1 directly.
        # Calling writeSetting while streamActive=False sets the param in the
        # struct without sending sdrplay_api_Update; Init then applies it.
        # ch0 (sync tuner) deliberately has no notch — it receives FM as its
        # timing reference.
        sdr.writeSetting("rfnotch_ctrl_ch1", "true")
        sdr.activateStream(rx_unused)
        sdr.activateStream(rx)
        # sdrplay_api_Init() fires during the first activateStream() (for ch0).
        # The upstream driver's activateStream() re-applies Tuner B params,
        # but we re-apply here as a safety net.
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 1, tune_freq)
        # Re-apply gains as a safety net — Init may reset ch1 gain to ch0 values.
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR", 59)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR", 9)
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 1, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 1, "IFGR", ch1_ifgr)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 1, "RFGR", 0)
        # Re-apply notch after activation in case Init reset it.
        sdr.writeSetting("rfnotch_ctrl_ch1", "true")
        # Diagnostic: verify frequency, gain and notch settings after post-init re-apply.
        for _ch in (0, 1):
            _freq = sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, _ch)
            _ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, _ch, "IFGR")
            _rfgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, _ch, "RFGR")
            print(f"  [post-init] ch{_ch}: freq={_freq/1e6:.4f} MHz  IFGR={_ifgr:.0f} dB  RFGR(LNA)={_rfgr:.0f}")
        _notch_ch1 = sdr.readSetting("rfnotch_ctrl_ch1")
        print(f"  [post-init] rfnotch_ctrl_ch1={_notch_ch1}")
    else:
        devs = SoapySDR.Device.enumerate(args.device)
        if not devs:
            print("ERROR: No SDR device found.", file=sys.stderr)
            return 1
        sdr = SoapySDR.Device(devs[0])
        _setup_channel(sdr, SoapySDR, 0, tune_freq, args.rate, args.gain)
        rx_unused = None
        rx = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        sdr.activateStream(rx)

    if is_rspduo:
        # getGain() with no name always returns 0 in SoapySDRPlay3; use named elements.
        actual_gain_ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, args.channel, "IFGR")
        actual_gain_rfgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, args.channel, "RFGR")
    else:
        actual_gain_ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, args.channel)
        actual_gain_rfgr = None
    actual_rate = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, args.channel)
    if abs(actual_rate - args.rate) > 1000:
        print(
            f"  [WARNING] Requested sample rate {args.rate/1e6:.6g} MHz not accepted by device; "
            f"using actual rate {actual_rate/1e6:.6g} MHz for calculations.\n"
            f"  (RSPduo supports: 0.0625, 0.125, 0.25, 0.5, 1, 2 MHz -- pass --rate 2000000)",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Drain stale USB pipeline samples (0.3 s)                           #
    # ------------------------------------------------------------------ #
    drain_end = time.monotonic() + 0.3
    chunk = np.zeros(65_536, dtype=np.complex64)
    while time.monotonic() < drain_end:
        sdr.readStream(rx, [chunk], len(chunk), timeoutUs=1_000_000)
        if rx_unused is not None:
            sdr.readStream(rx_unused, [chunk], len(chunk), timeoutUs=1_000_000)

    window_samples = max(1024, int(actual_rate * args.window_ms / 1000))

    ch_desc = f"ch{args.channel} (Tuner {args.channel + 1})" if is_rspduo else f"ch{args.channel}"
    print(f"Target signal check")
    print(f"  Channel:    {ch_desc}")
    print(f"  Frequency:  {args.freq / 1e6:.3f} MHz", end="")
    if args.freq_offset != 0.0:
        print(f"  (tuned to {tune_freq / 1e6:.3f} MHz, {args.freq_offset/1e3:.0f} kHz offset)")
    else:
        print()
    if is_rspduo:
        print(f"  Gain:       IFGR={actual_gain_ifgr:.0f} dB (IF reduction)  LNA state={actual_gain_rfgr:.0f}")
    else:
        print(f"  Gain:       {actual_gain_ifgr:.1f} dB")
    print(f"  Window:     {window_samples / actual_rate * 1000:.0f} ms")
    print(f"  Duration:   {args.duration:.0f} s")
    if args.freq_offset != 0.0:
        print(f"  Bandwidth:  {MEASURE_BW_HZ/1e3:.0f} kHz FFT window around signal")
    print(f"  Bar range:  {BAR_MAX - BAR_RANGE:.0f} dBFS (left) ... {BAR_MAX:.0f} dBFS (right)")
    print()
    print(f"  Key your transmitter a few times now.")
    print()
    print(f"  {'Time':>6}  {'dBFS':>7}  {'':30}  Signal?")
    print(f"  {'-' * 60}")

    all_db: list[float] = []

    def _alarm(signum, frame):
        raise _Done

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, int(args.duration + 0.5)))

    buf = np.zeros(window_samples, dtype=np.complex64)
    t_start = time.monotonic()

    unused_chunk = np.zeros(65_536, dtype=np.complex64)

    try:
        while True:
            # Fill one measurement window.  On RSPduo, drain rx_unused in lockstep
            # so the TDM ADC buffer doesn't back up while we fill the target window.
            collected = 0
            while collected < window_samples:
                want = min(65_536, window_samples - collected)
                if rx_unused is not None:
                    sdr.readStream(rx_unused, [unused_chunk], want, timeoutUs=2_000_000)
                sr = sdr.readStream(rx, [buf[collected:]], want, timeoutUs=2_000_000)
                if sr.ret > 0:
                    collected += sr.ret
                elif sr.ret < 0:
                    break

            db = _measure_power_db(buf[:collected], actual_rate, args.freq_offset)
            all_db.append(db)

            # Running noise floor = 10th percentile of all measurements so far
            noise_db = float(np.percentile(all_db, 10))

            # Simple signal/noise heuristic: >10 dB above current noise floor
            is_signal = (db - noise_db) > 10.0
            signal_tag = "*** SIGNAL ***" if is_signal else ""

            elapsed = time.monotonic() - t_start
            print(f"  {elapsed:5.1f} s  {db:7.1f}  {_power_bar(db)}  {signal_tag}")

    except _Done:
        pass
    finally:
        signal.alarm(0)
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        if rx_unused is not None:
            sdr.deactivateStream(rx_unused)
            sdr.closeStream(rx_unused)

    if not all_db:
        print("\nNo data collected.")
        return 1

    # ------------------------------------------------------------------ #
    # Analysis and threshold recommendation                               #
    # ------------------------------------------------------------------ #
    all_db_arr = np.array(all_db)
    noise_floor  = float(np.percentile(all_db_arr, 10))
    peak_signal  = float(np.max(all_db_arr))
    snr_db       = peak_signal - noise_floor

    print()
    print("--- Results ---")
    print(f"Measurements:    {len(all_db)} windows * {window_samples / actual_rate * 1000:.0f} ms")
    print(f"Noise floor:     {noise_floor:.1f} dBFS  (10th percentile)")
    print(f"Peak signal:     {peak_signal:.1f} dBFS  (maximum observed)")
    print(f"Observed SNR:    {snr_db:.1f} dB")

    if noise_floor > -6.0:
        print()
        print(f"WARNING: ADC is saturated -- noise floor is {noise_floor:.1f} dBFS (near full scale).")
        print("  The gain is far too high; the ADC has no headroom to distinguish signal from noise.")
        print(f"  Reduce gain significantly: --gain {max(0, int(args.gain - 20))}")
        print("  Target noise floor: -40 to -60 dBFS with no signal present.")
        return 0

    if snr_db < 6.0:
        print()
        print("WARNING: Observed SNR is low (<6 dB).")
        print("  Either no transmission occurred, or gain needs adjustment.")
        print(f"  Try reducing gain if noise floor is high: --gain {max(0, int(args.gain - 10))}")
        print(f"  Or try increasing gain if noise floor is very low: --gain {args.gain + 10:.0f}")
        print("  Re-run and key your transmitter several times during the run.")
        return 0

    # Onset: 10 dB above the noise floor.  Anchoring to the noise floor (rather
    # than the midpoint between noise and peak) keeps the threshold low enough to
    # detect the carrier reliably even when the effective signal level is reduced
    # by R820T gain settling after each frequency hop in freq_hop mode.  The
    # midpoint rule fails at high SNR because it places the threshold 20+ dB above
    # noise, where a still-settling receiver may never reach.
    # Cap 6 dB below peak so the threshold is always below the observable signal.
    ONSET_MARGIN_DB  = 10.0
    OFFSET_MARGIN_DB =  6.0   # 4 dB hysteresis below onset
    onset_db  = round(min(noise_floor + ONSET_MARGIN_DB, peak_signal - 6.0))
    offset_db = round(max(onset_db - 4.0, noise_floor + OFFSET_MARGIN_DB))

    noise_margin  = offset_db - noise_floor        # dB above noise (for display)
    actual_hyst   = onset_db  - offset_db

    print()
    print("Recommended thresholds:")
    print(f"  onset_db  = {onset_db:.0f} dBFS   ({onset_db - noise_floor:.1f} dB above noise floor)")
    print(f"  offset_db = {offset_db:.0f} dBFS   ({actual_hyst:.0f} dB hysteresis, "
          f"{noise_margin:.1f} dB above noise floor)")

    if actual_hyst < 4:
        print()
        print(f"  NOTE: SNR is only {snr_db:.1f} dB -- hysteresis reduced to {actual_hyst:.0f} dB.")
        print(f"  Set min_release_windows: 4-6 in node.yaml to suppress chattering.")
    print()
    print("For verify_freq_hop.py / verify_rspduo.py:")
    print(f"  --onset-db {onset_db:.0f} --offset-db {offset_db:.0f}")
    print()
    print("For config/node.yaml (carrier: section, all modes):")
    print(f"  carrier:")
    print(f"    onset_db:  {onset_db:.0f}")
    print(f"    offset_db: {offset_db:.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
