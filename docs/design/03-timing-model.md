# Beagle Timing Model

## The Core Insight

TDOA geolocation requires knowing, for each node, *when* the target signal arrived relative to a common reference. The naive approach - compare absolute GPS-disciplined timestamps across nodes - requires microsecond-level absolute clock synchronization. GPS-disciplined chrony can achieve ~1 usec, but USB jitter and kernel scheduling can degrade this significantly for SDR captures.

Beagle uses a better approach: **measure the time difference between the sync signal event and the target carrier onset locally on each node, using the same sample clock**. This converts the problem from absolute time synchronization to local sample-rate precision. The local ADC clock is stable to <<1 ppm over the <200 ms interval between sync event and target onset - far better than any NTP or chrony-based absolute timestamp.

## Measurement Formula

```
sync_delta_ns = (N_target - N_sync) x 1_000_000_000 / sample_rate_hz
```

Where:
- `N_target` = sample index of the LMR carrier onset detection
- `N_sync`   = sample index of the most recent FM pilot sync event before the onset
- `sample_rate_hz` = the SDR's nominal sample rate, corrected by crystal calibration

Both sample indices must be from **the same continuous ADC clock** (see SDR Modes below).

## Sample Rate Crystal Calibration

RTL-SDR crystal accuracy is typically 50-100 ppm uncalibrated. Over a 200 ms sync window, this introduces 10-20 usec of timing error. The FM stereo pilot eliminates this error.

The FM pilot at exactly 19,000.000 Hz is locked to the broadcast station's precision frequency standard (traceable to GPS/UTC). We track the unwrapped phase of the pilot across consecutive 10 ms sync windows:

```
expected_phase_advance = 2pi x 19000 x 0.010 = 380pi rad per 10 ms window
measured_phase_advance = cross_correlation_angle(window_N+1) - cross_correlation_angle(window_N)
correction_factor      = measured_phase_advance / expected_phase_advance
```

A rolling median over the last 100 windows (~1 second) gives a stable correction that is applied to all `sync_delta_ns` calculations:

```
sync_delta_ns_corrected = sync_delta_ns x correction_factor
```

After calibration, crystal error is reduced to <1 ppm, cutting timing error to <0.2 usec over 200 ms.

## SDR Operating Modes

### Mode 1: `freq_hop` (single RTL-SDR via librtlsdr-2freq)

The RTL2832 ADC runs continuously while the tuner alternates between sync frequency (FM) and target frequency (LMR):

```
Time:  |<-- sync block -->|<-- target block -->|<-- sync block -->|...
Freq:  |    99.9 MHz      |    462.5625 MHz        |    99.9 MHz      |
ADC:   |--------------------------------------------------------------- continuous
Index: |0         N        |N         2N         |2N        3N      |
```

Sample indices are from one unbroken stream - `sync_delta_ns` is computed directly from sample index arithmetic. No inter-device synchronization needed.

**Settling time handling:** After each frequency switch, the tuner takes 10-100 ms to stabilize. The first `settling_samples` (default: 40,960 ~ 20 ms at 2.048 MSPS) of each block are discarded in `FreqHopSDRReceiver`. The remaining samples are valid.

**Coverage gap:** During the sync frequency block (~32 ms), the target frequency is not monitored. LMR transmissions shorter than ~32 ms may be missed. This is acceptable since typical LMR PTT duration is >1 second.

### Mode 2: `two_sdr` with GPS 1PPS Injection

Two separate RTL-SDRs run simultaneously - one on the FM sync band, one on the LMR target band. They have **independent ADC clocks** and **independent USB transfer scheduling**. Naive kernel-clock timestamps would introduce ~1-10 ms jitter between devices - far too coarse for TDOA.

**Solution: GPS 1PPS RF injection.** The GPS module's 1PPS output is attenuated (via 10 Mohm resistors) and injected into both SDR antenna inputs. The pulse appears as a broadband amplitude spike in both IQ streams at the same physical instant.

In software:
1. `PPSDetector` finds the 1PPS spike sample index in each stream: `N_pps_sync`, `N_pps_target`
2. Events are expressed relative to the most recent 1PPS:

```
t_sync_event   = (N_sync_event  - N_pps_sync)   / rate_sync
t_target_onset = (N_target_onset - N_pps_target) / rate_target
sync_delta_ns  = (t_target_onset - t_sync_event) x 1e9
```

This eliminates USB jitter from the measurement. The only sources of error are GPS 1PPS jitter (<100 ns) and spike detection accuracy (~0.5 usec at 2 MSPS).

**Hardware for 1PPS injection:**

```
GPS 1PPS (3.3V TTL) ---- 10 Mohm resistor ---- SMA tee ---- SDR antenna input
                                                    +------- antenna (coax)
```

One 10 Mohm resistor per SDR. Signal level at SDR input: ~ -82 dBm (38 dB above noise floor, well below saturation). Effect on receiver sensitivity: negligible. Total cost: ~$5.

### Mode 3: `rspduo` (SDRplay RSPduo dual-tuner)

The RSPduo contains two independent RF tuners feeding a **single shared ADC**
via time-division multiplexing, all clocked from the same 24 MHz TCXO:

```
Tuner 1 (sync, e.g. 99.9 MHz FM)  -+
                                    +-> single ADC (TDM) -> USB -> host
Tuner 2 (target, e.g. 155 MHz LMR) -+
```

Because both channels share one ADC clock and one USB transfer stream there
is no inter-channel USB jitter. The timing model is nearly identical to
`freq_hop`:

```
sync_delta_ns = (N_target - N_sync) x 1_000_000_000 / sample_rate_hz
              - pipeline_offset_ns
```

The `pipeline_offset_ns` correction accounts for the deterministic 0.5-sample
ADC interleave offset between the two channels (~ 250 ns at 2 MSPS). Start
at 0 and calibrate empirically using `scripts/colocated_pair_test.py --db`
against a co-located RSPduo reference.

SoapySDR access uses a master/slave open pattern: the first `Device()` open
becomes the master (configures the ADC clock); the second open on the same
device string becomes the slave (second tuner). Both produce separate,
simultaneous `complex64` IQ streams.

**No GPS 1PPS hardware required** - the shared ADC clock eliminates the
inter-device jitter problem entirely.

### Mode 4: `single_sdr` (wideband SDR)

A single wideband SDR (e.g. HydraSDR RFOne) captures both FM and LMR simultaneously in one wide IQ stream. Sync and target events are extracted from the same sample stream - perfect sample-level alignment, no injection hardware needed.

**Limitation:** Only valid when both signals fit within the SDR's instantaneous bandwidth. FM sync (99 MHz) + VHF LMR (155 MHz) = 56 MHz gap. FM sync + UHF LMR (440 MHz) = 341 MHz gap. A single consumer SDR typically cannot cover both FM and UHF LMR simultaneously.

## Why Absolute GPS Timestamps Are Still Useful

Even though `sync_delta_ns` is the precise TDOA measurement, we still include `onset_time_ns` (from the GPS-disciplined kernel clock) in `CarrierEvent`. The aggregation server uses this for **event association**: it needs to identify which events from different nodes correspond to the same LMR transmission. With `onset_time_ns` accurate to +/-10 ms and typical LMR PTT durations of >1 second, the server can reliably match events across nodes using a +/-200 ms time window.

## Error Budget

For `freq_hop` mode (most common initial deployment):

| Error source | Magnitude | Corrected by |
|-------------|-----------|-------------|
| Crystal rate (100 ppm, 200 ms) | 20 usec raw | FM pilot crystal calibration -> <0.2 usec |
| Pilot phase measurement | <1 usec | - |
| Carrier onset detection | ~window/2 = 5 ms | Dual-threshold hold reduces onset jitter; acceptable since TDOA uses first-onset |
| Filter group delay | Deterministic | Subtracted in calibration_offset_ns |
| Total after calibration | ~1-5 usec | - |

At 5 usec: position uncertainty ~ 1.5 km, meets the neighborhood-level goal.

For `rspduo` mode:

| Error source | Magnitude |
|-------------|-----------|
| Interleave offset correction | Deterministic; subtracted by `pipeline_offset_ns` |
| Pilot phase measurement | <1 usec |
| Crystal calibration residual | <0.1 usec |
| **Total** | **~1-2 usec -> ~300-600 m position** |

For `two_sdr` + 1PPS injection:

| Error source | Magnitude |
|-------------|-----------|
| GPS 1PPS jitter | <100 ns |
| 1PPS spike detection | ~0.5 usec at 2 MSPS |
| Pilot phase measurement | <1 usec |
| Crystal calibration residual | <0.1 usec |
| **Total** | **~1 usec -> ~300 m position** |

## Node-Server Measurement Contract

The aggregation server must be able to pair and correlate events from any
combination of node hardware (RTL-SDR, RSPduo, future SDRs) without
knowledge of each node's internal configuration.  This requires that nodes
produce **uniformly constructed measurements**.

### Principles

1. **Measurements should be hardware-agnostic.**  The `sync_delta_ns`,
   `onset_time_ns`, and IQ snippet delivered by every node must have the
   same semantic meaning regardless of the SDR hardware, carrier detector
   thresholds, or pipeline parameters used to produce them.  Two nodes
   observing the same carrier edge should produce measurements that the
   server can directly compare without per-node correction factors.

2. **Compensation belongs on the node.**  If a node's pipeline introduces a
   structural delay or anchoring bias (e.g. decimation filter group delay,
   frequency-hop inter-block gap, ring buffer fill asymmetry), the node
   must correct for it before reporting.  The server should not need to
   know `min_hold_windows`, `settling_samples`, or any other node-local
   parameter.

3. **When the server must know a parameter, send it with the event.**
   Some differences are unavoidable: nodes capture IQ at different sample
   rates (64 kHz vs 62.5 kHz).  Rather than requiring the server to
   maintain a node configuration registry, the sample rate is included in
   every `CarrierEvent` (`channel_sample_rate_hz`).  The server resamples
   as needed for cross-correlation.

4. **IQ snippets must be consistently anchored.**  The carrier transition
   (onset rise or offset fall) must sit at the **same relative position**
   in every snippet, regardless of the carrier detector's
   `min_hold_windows` or `min_release_windows` setting.  This is achieved
   by collecting post-event IQ windows (`carrier_snippet_post_windows`)
   and using center-anchored encoding (`_encode_combined`), which places
   the transition at the snippet midpoint.  Without this, cross-correlation
   between nodes with different detector settings produces a systematic lag
   that the server cannot distinguish from a real TDOA.

### What the server may assume

- `sync_delta_ns` is the time from the most recent FM pilot zero-crossing
  to the carrier edge, measured on a single continuous sample clock,
  corrected for crystal rate error.  It may span multiple pilot periods
  (freq_hop mode); the server reduces it modulo the pilot period (7 ms)
  for event grouping.
- `onset_time_ns` is the wall-clock time of the carrier edge, suitable for
  coarse event association (+/-200 ms window).
- `iq_snippet_b64` contains a center-anchored IQ capture of the carrier
  transition at `channel_sample_rate_hz`, suitable for sub-sample
  cross-correlation without knowledge of the node's detector parameters.

### What the server must not assume

- Any specific value of `min_hold_windows`, `min_release_windows`,
  `settling_samples`, or `pipeline_offset_ns`.
- That all nodes use the same sample rate (resampling is the server's
  responsibility for cross-correlation).
- That `sync_delta_ns` is within one pilot period (it may span many;
  modular reduction is applied at pairing time).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
