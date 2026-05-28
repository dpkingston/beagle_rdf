# Beagle Timing Model

## The Core Insight

TDOA geolocation requires knowing, for each node, *when* the target signal arrived relative to a common reference. The naive approach - compare absolute GPS-disciplined timestamps across nodes - requires microsecond-level absolute clock synchronization. GPS-disciplined chrony can achieve ~1 usec, but USB jitter and kernel scheduling can degrade this significantly for SDR captures.

Beagle uses a better approach: **measure the time difference between the sync signal event and the target carrier onset locally on each node, using the same sample clock**. This converts the problem from absolute time synchronization to local sample-rate precision. The local ADC clock is stable to <<1 ppm over the <200 ms interval between sync event and target onset - far better than any NTP or chrony-based absolute timestamp.

## Measurement Formula

```
sync_to_snippet_start_ns = (N_target - N_sync) x 1_000_000_000 / sample_rate_hz
```

Where:
- `N_target` = sample index of the LMR carrier onset detection
- `N_sync`   = sample index of the most recent **RDS bit-transition sync
  event** before the onset (one event per ~842 usec; emitted by
  `RDSSyncDetector` -- see [02-signal-processing.md](02-signal-processing.md))
- `sample_rate_hz` = the SDR's nominal sample rate, corrected by crystal calibration

Both sample indices must be from **the same continuous ADC clock** (see SDR Modes below).

## Sync events: RDS bit transitions, not pilot zero-crossings

Beagle uses **RDS bit transitions** as sync events, not raw pilot zero-crossings.
The reason is the only one that matters for TDOA: pilot zero-crossings are
physically indistinguishable -- every node tuned to the same FM station sees
the same pilot waveform but locks to a *different* zero-crossing chosen by
its arbitrary buffer alignment.  The result is an unresolvable
`N x 52.6 usec` cross-node ambiguity in `sync_delta` subtraction that no
amount of calibration can fix.

The RDS data signal modulated on the 57 kHz subcarrier is **phase-locked to
the pilot by spec** (NRSC-4-B / IEC 62106) at exactly `pilot/16 = 1187.5 Hz`.
Bit boundaries are deterministic features of the broadcast signal, so every
node identifies the **same** bit transition as the same physical event.  The
RDS bit period of 842 usec is comfortably larger than the maximum geometric
TDOA for a 100 km baseline (333 usec), so cross-node disambiguation is
unambiguous.

## Sample Rate Crystal Calibration

RTL-SDR crystal accuracy is typically 50-100 ppm uncalibrated. Over a 200 ms
sync window, this introduces 10-20 usec of timing error. The FM stereo pilot
(still extracted internally by `RDSSyncDetector`) eliminates this error.

The FM pilot at exactly 19,000.000 Hz is locked to the broadcast station's
precision frequency standard (traceable to GPS/UTC).  Even though the *sync
events themselves* now come from the RDS bit clock, the detector also runs a
pilot extraction path in parallel (19 kHz BPF + complex correlation in 10 ms
windows) and tracks the unwrapped pilot phase across windows:

```
expected_phase_advance = 2pi x 19000 x 0.010 = 380pi rad per 10 ms window
measured_phase_advance = cross_correlation_angle(window_N+1) - cross_correlation_angle(window_N)
correction_factor      = measured_phase_advance / expected_phase_advance
```

A rolling median over the last 100 windows (~1 second) gives a stable
correction.  The current correction factor is attached to every `SyncEvent`
emitted by `RDSSyncDetector` (in the `sample_rate_correction` field) and
applied by `DeltaComputer` when converting sample indices to nanoseconds:

```
corrected_rate = nominal_rate * sample_rate_correction
sync_to_snippet_start_ns  = round(delta_samples * 1e9 / corrected_rate)
```

After calibration, crystal error is reduced to <1 ppm, cutting timing error
to <0.2 usec over 200 ms.

## SDR Operating Modes

### Mode 1: `freq_hop` (single RTL-SDR via librtlsdr-2freq)

The RTL2832 ADC runs continuously while the tuner alternates between sync frequency (FM) and target frequency (LMR):

```
Time:  |<-- sync block -->|<-- target block -->|<-- sync block -->|...
Freq:  |    99.9 MHz      |    462.5625 MHz        |    99.9 MHz      |
ADC:   |--------------------------------------------------------------- continuous
Index: |0         N        |N         2N         |2N        3N      |
```

Sample indices are from one unbroken stream - `sync_to_snippet_start_ns` is computed directly from sample index arithmetic. No inter-device synchronization needed.

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
sync_to_snippet_start_ns  = (t_target_onset - t_sync_event) x 1e9
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
sync_to_snippet_start_ns = (N_target - N_sync) x 1_000_000_000 / sample_rate_hz
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

Even though `sync_to_snippet_start_ns` is the precise TDOA measurement, we still include `onset_time_ns` (from the GPS-disciplined kernel clock) in `CarrierEvent`. The aggregation server uses this for **event association**: it needs to identify which events from different nodes correspond to the same LMR transmission. With `onset_time_ns` accurate to +/-10 ms and typical LMR PTT durations of >1 second, the server can reliably match events across nodes using a +/-200 ms time window.

## Error Budget

For `rspduo` mode (the production deployment as of 2026-05):

| Error source | Magnitude | Notes |
|-------------|-----------|-------|
| RDS bit-transition timing (pilot-phase derived) | < 0.1 usec | Measured ~0.06 usec on KUOW 94.9 |
| Crystal calibration residual | < 0.1 usec | RSPduo TCXO at <10 ppm before correction |
| Inter-node knee position (coherent IQ GCC-PHAT) | ~50 µs typical, ~190 µs pooled median on Magnolia corpus | Replaced the 290 µs window-quantisation floor; primary error source in current production |
| RDS group-anchor mismatch (±1 group) | 0 after server-side snap; was ~30 % of pairs on Capitol Hill 2 m corpus | Snap correction in `compute_tdoa_s` removes ±87.6 ms anchor mismatches before refinement |
| Per-pair calibration residual | ~1–3 µs after fit | Per-pair `pair_offsets_s` removes the dominant per-pair bias on a known target |
| RSPduo interleave offset | Deterministic; subtracted by `pipeline_offset_ns` | ~250 ns at 2 MSPS, calibrated empirically |
| **Position uncertainty (Capitol Hill 2 m corpus, post-snap pre-calibration)** | **p50 ~11 km, p75 ~15 km, p90 ~54 km** | Simulation; Commit 14 auto-calibration is the next reduction step |

The headline result from the offline replay of the Capitol Hill 2 m
corpus (dpk-tdoa1 / dpk-tdoa2 / kb7ryy / n7jmv-tdoa-qth, KUOW 94.9 sync,
2026-05): without group-period snap, p75 = 187 km and p90 = 992 km
(diagonal SSW–NNE outlier smear caused by 26–30 % of pairs anchored to
different RDS groups).  Adding the snap collapses outliers to
p75 ≈ 15 km / p90 ≈ 54 km.  Auto-calibration (Commit 14) is expected
to drag the median into the ~1 km regime.

For `freq_hop` mode (RTL-SDR with `librtlsdr-2freq`):

> Status as of 2026-05: the RDS sync subsystem (pilot-phase
> derivation + block sync + block-A anchor matcher) is in principle
> compatible with freq_hop's alternating sync/target blocks, because the
> short-gap partial reset preserves pilot-phase state across retunes
> (`02-signal-processing.md` Stage 3).  Block sync and anchor matching
> have not yet been exercised in freq_hop mode end-to-end, so freq_hop
> remains in the codebase but is not yet a supported production
> configuration; expect `anchor_emit_fraction` to be lower than in
> rspduo mode until block sync re-locks after each sync-block.

For `two_sdr` + 1PPS injection (legacy mode, not currently exercised):

| Error source | Magnitude |
|-------------|-----------|
| GPS 1PPS jitter | <100 ns |
| 1PPS spike detection | ~0.5 µs at 2 MSPS |
| RDS bit-transition timing | < 0.1 µs |
| Crystal calibration residual | < 0.1 µs |
| Inter-node knee position (coherent IQ GCC-PHAT) | ~50 µs typical |
| **Total** | dominated by knee position (~50 µs) |

## Group-period anchor-snap correction

RDS-group-anchor mismatches are the dominant source of large outliers in
production fixes.  The RDS bit period (842 µs) is exactly 1/104 of the
RDS group period (87.6 ms), and the node-side `DeltaComputer`
matcher anchors to a specific block-A bit-0 transition within the
±half-group window of the carrier edge.  Under fades or boot-time
misalignment, two paired nodes can pick block-A bit-0 transitions from
adjacent RDS groups — their `sync_to_snippet_start_ns` then differ by
exactly one group period (±87.6 ms) more than the true TDOA.

### Why bit-period disambiguation alone isn't enough

`compute_tdoa_s` historically disambiguated by rounding the combined
TDOA to the nearest bit period.  Because the group period is an exact
integer multiple of the bit period, a ±1-group offset rounds *near* the
correct bit count.  But the disambiguation operates on
`combined_ns = raw_ns + refinement_ns + correction_ns`, and the
refinement (inter-node knee position) can itself reach tens of ms in a
16 384-sample snippet.  A multi-bit knee shift combined with an 87.6 ms
raw offset can push the bit-period rounder to pick the wrong `n` and
corrupt the residual by hundreds of µs.

### The snap

`_apply_group_period_snap(raw_ns)` in `beagle_server/tdoa.py` snaps
`raw_ns` *before* refinement:

```
k = round(raw_ns / GROUP_PERIOD_NS)
|k| > 1            → drop pair (cross-transmission noise)
|k| == 1 + |residual| ≤ 20 ms → raw_ns -= k * GROUP_PERIOD_NS  (snap)
|k| == 1 + |residual|  > 20 ms → leave raw_ns alone, count as 'noise'
k == 0              → pass through unchanged
```

Counters `k_zero` / `k_plus_1` / `k_minus_1` / `noise` / `out_of_range`
are surfaced on the server's `/health`.

After the snap, the existing bit-period disambiguation (step 6 below)
only ever sees clean sub-millisecond inputs, so its `round()` choice is
unambiguous.

### Disambiguation cascade

The full disambiguation order in `compute_tdoa_s` is now:

1. Compute `raw_ns = sync_to_snippet_start_A − sync_to_snippet_start_B`.
2. Apply `SyncCalibrator` per-pair grid correction (sub-bit).
3. **Group-period snap** (this section): collapse ±87.6 ms mismatches.
4. Apply sync-path geometry correction.
5. Refine via `tdoa_method` (`xcorr` / `phat` / `audio_phat` / `knee`).
6. **Bit-period disambiguation**: `n = round(combined_ns / 842 µs)`.
7. Apply per-pair / per-node TDOA calibration.
8. Geometric plausibility check vs `max_xcorr_baseline_km`.

## Node-Server Measurement Contract

The aggregation server must be able to pair and correlate events from any
combination of node hardware (RTL-SDR, RSPduo, future SDRs) without
knowledge of each node's internal configuration.  This requires that nodes
produce **uniformly constructed measurements**.

### Principles

1. **Measurements should be hardware-agnostic.**  The `sync_to_snippet_start_ns`,
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

- `sync_to_snippet_start_ns` is the time from the most recent **RDS bit-transition sync
  event** to the carrier edge, measured on a single continuous sample clock,
  corrected for crystal rate error.  It may span multiple RDS bit periods;
  the server reduces it modulo the bit period (`1e9 / 1187.5 = 842,105 ns`)
  for event grouping and disambiguates cross-node sample counts via
  `n = round((raw_ns + path_correction_ns) / 842,105)`.  The disambiguation
  is unambiguous because `T_sync / 2 = 421 usec >> max_TDOA ~= 333 usec` for a
  100 km baseline.
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
- That `sync_to_snippet_start_ns` is within one RDS bit period (it may span many;
  modular reduction is applied at pairing time, and geometric disambiguation
  is applied during TDOA computation).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
