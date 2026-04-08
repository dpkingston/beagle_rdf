# Beagle Sync Signal: FM RDS Bit Transitions

## Why FM Broadcast?

Beagle needs each node to produce a timing reference that is:

1. **Simultaneous across all nodes** - every node must reference the same physical
   event at the same instant.
2. **Precisely known** - the reference event must be localizable in sample space
   to sub-microsecond accuracy.
3. **Passively receivable** - no special infrastructure, no active transmitter.
4. **Location-known** - the reference transmitter's geographic position must be
   known accurately for path-delay correction.
5. **Cross-node identifiable** - every node must agree on *which* event is the
   "same" event.

FM broadcast fits all five requirements.  Every radio in the coverage area
receives the same broadcast simultaneously (modulo propagation delay, which is
correctable).  The stereo pilot at exactly **19,000.000 Hz** is locked to the
station's frequency standard, which is in turn traceable to GPS/UTC.  And the
**RDS data signal** modulated on the 57 kHz subcarrier provides
unambiguously-identifiable bit transitions at 1187.5 Hz that all nodes can
agree on.

The fifth requirement is what changed Beagle's design from pilot
zero-crossings to RDS bit transitions -- see the next section.

---

## Why RDS bit transitions instead of pilot zero-crossings?

Beagle previously used the FM stereo pilot tone (19 kHz) directly as the sync
event source.  Each node located the nearest pilot zero-crossing within an
arbitrary 7 ms window and emitted that as a `SyncEvent`.

This approach **passes all the standard sub-microsecond tests** for a single
node: the pilot is at exactly 19 kHz, locked to a precision frequency
standard, and easy to extract via narrow bandpass + complex correlation.
Within a single node's clock domain, the pilot phase converges to better
than 1 usec.

But when you put two nodes side-by-side and compute cross-node
`sync_delta_A - sync_delta_B`, the result is wrong by an unbounded multiple
of the pilot period (52.6 usec per cycle).  The reason is brutal in
hindsight: **pilot zero-crossings are pairwise indistinguishable**.  Every
zero-crossing is the same waveform feature.  Two nodes whose buffer windows
are not aligned (which is always the case in practice -- the buffer grid
starts at SDR startup, and SDRs start at different times) lock to different
zero-crossings, separated by some integer number of pilot cycles, with no
mechanism to figure out which is "the same one".

The geometric correction the server applies (path-delay differences from
FCC station coordinates) is much smaller than 52.6 usec, so it cannot
disambiguate.  The result for non-co-located node pairs in production was
fixes scattered across thousands of microseconds, often landing at the
search boundary.

**RDS solves this**.  The RDS data stream is BPSK modulation on the 57 kHz
subcarrier (third harmonic of the pilot, phase-locked to it by the
NRSC-4-B / IEC 62106 standard) at exactly **1187.5 bps = pilot/16**.  Bit
boundaries are deterministic features of the broadcast signal, anchored to
the data content -- when nodes A and B identify "the bit transition at
T_sync", they are *unambiguously* referring to the same physical event.

The bit period is 842 usec, comfortably larger than the maximum geometric
TDOA for a 100 km baseline (333 usec), so cross-node disambiguation via
`n = round((raw_ns + path_correction_ns) / 842,105)` is unique.

> **The pilot is not abandoned**, just demoted.  The `RDSSyncDetector` still
> extracts the 19 kHz pilot internally on a parallel path, feeds the
> unwrapped pilot phase into `CrystalCalibrator`, and uses the resulting
> sample-rate correction when converting sample indices to nanoseconds.
> The pilot is now a *frequency reference* (used to measure the SDR's true
> sample rate) rather than a *timing reference*.

---

## RDS signal characteristics

| Property | Value |
|----------|-------|
| Subcarrier | 57,000 Hz (= 3 x pilot, phase-locked by spec) |
| Modulation | BPSK (DSB-SC, 100% modulation depth) |
| Bit rate | **1187.5 bps = 19000 / 16** (exact, locked to pilot) |
| Bit period | **842.105 microseconds** |
| Bandwidth | +/- 2.4 kHz around 57 kHz |
| Group structure | 4 x 26-bit blocks (16 data + 10 CRC) per group; ~11 groups/sec |
| Injection level | typically -11 dB relative to pilot |
| Coverage | city-wide; same as pilot |
| Availability | nearly all NPR affiliates and commercial FM stations carry RDS |
| Standards | NRSC-4-B (US, "RBDS"), IEC 62106 (international) |

The bit clock is **exactly** pilot/16 and phase-coherent by spec.  All nodes
receiving the same station see the same bit transitions at the same physical
time (plus propagation delay).

### Timing precision

The recovered bit timing comes from a Mueller-Muller timing-recovery loop with
sub-sample interpolation.  On synthetic FM IQ with a clean RDS signal, the
inter-bit interval jitter is < 0.01 usec.  On a live RSPduo capture of KUOW
94.9 (Seattle), the measured jitter is ~0.06 usec.  The SyncEvent
`sample_index` is a **float** carrying full sub-sample precision, propagated
end-to-end through the server's TDOA calculation.

Per-event timing precision is therefore not the limiting factor in Beagle's
overall accuracy budget; the carrier detector's 1 ms power window
(~290 usec uniform-noise sigma) dominates.  See
[03-timing-model.md](03-timing-model.md) for the full error budget.

---

## Seattle FM Stations

The sync station is configured per-node in `node.yaml` under `sync_signal:`.
The following stations provide good coverage across the Seattle/Eastside area:

| Station | Frequency | Transmitter Location | Notes |
|---------|-----------|---------------------|-------|
| KISW | 99.9 MHz | 47.6253 deg N, 122.3563 deg W (Queen Anne Hill) | Primary; used in TDOAv1/v2 |
| KUOW | 94.9 MHz | 47.61576 deg N, 122.30919 deg W | Primary; FCC-documented |
| KEXP | 90.3 MHz | 47.6619 deg N, 122.3487 deg W (Capitol Hill) | Tertiary option |
| KING-FM | 98.1 MHz | 47.5235 deg N, 122.1306 deg W (Cougar Mountain) | Useful for eastern nodes |

**Always verify FCC coordinates** before deployment.  Transmitter locations are
available in the FCC AM/FM Query database (https://www.fcc.gov/media/radio/fm-query).
Coordinates accurate to ~100 m are achievable; 100 m translates to <0.33 usec path-delay
correction error - negligible for the system's overall accuracy target.

### Selecting a station

- **Verify the station carries RDS.**  Run `verify_rds_sync.py` against a
  candidate station and confirm the event rate reaches ~1188/s after the
  M&M warmup.  Nearly all NPR affiliates and most commercial FM stations
  in the US carry RDS, but check before committing.  Stations that broadcast
  in mono only (no stereo, no pilot, no RDS) are unusable.
- Prefer stations with transmitters **distant from the node by different
  amounts across your node network** - this maximises the path-delay
  correction variation and makes calibration cross-checks more sensitive.
- Prefer stations with **strong local signal** (`PilotCor` consistently
  >0.6).  Set `min_corr_peak: 0.3` in `sync_signal` as a minimum quality
  threshold.
- For urban nodes with strong multipath, the narrow 100 Hz BPF in the pilot
  extractor significantly reduces multipath effects.  RDS itself is fairly
  robust to multipath thanks to the BPSK + CRC structure (the M&M loop will
  drop out if multipath is severe, but most urban environments are fine).
- All nodes in a deployment **must use the same primary station** for their
  `sync_delta_ns` measurements to be comparable.

---

## Path-Delay Correction

Because the FM broadcast signal travels at the speed of light from a fixed
transmitter, it arrives at nodes at different times based on their distances.
This introduces a deterministic offset into each node's `sync_delta_ns` that
must be removed before computing TDOA.

### The problem

Node A is 15 km from the FM transmitter; Node B is 10 km away.  Node A's sync
event fires 16.7 usec later than Node B's (for the same broadcast sample), not
because of any target-signal timing difference, but purely due to propagation.
If uncorrected, this appears as a 16.7 usec TDOA bias.

### The correction

```
TDOA_AB_raw       = sync_delta_A - sync_delta_B

propagation_A     = dist(FM_transmitter, node_A) / c
propagation_B     = dist(FM_transmitter, node_B) / c

TDOA_AB_corrected = TDOA_AB_raw - (propagation_A - propagation_B)
                  = TDOA_AB_raw - (d_A - d_B) / c
```

The correction term `(d_A - d_B) / c` is computed by the aggregation server from
the `sync_transmitter` coordinates (carried in every `CarrierEvent`) and each
node's `node_location`.  Distances are computed using the haversine formula.

### Magnitude

For two nodes separated by 30 km with a transmitter 20 km from one and 40 km from
the other:

```
correction = (40 km - 20 km) / 299,792 km/s ~ 66.7 usec
```

Without this correction, the computed TDOA would have a 66.7 usec error, placing
the hyperbola ~20 km from the true answer - useless for geolocation.

### Accuracy

FCC-documented transmitter coordinates are typically accurate to ~100 m.  The error
in path-delay correction from a 100 m coordinate error is at most:

```
deltacorrection <= 100 m / 299,792,458 m/s ~ 0.33 usec
```

This is well below the system's ~1-5 usec total timing budget.

---

## Dual-Station Self-Calibration

Configuring a `secondary_station` in `sync_signal` enables a powerful passive
calibration check that requires no test transmitter.

### Principle

When a node tracks two FM stations simultaneously, it computes `sync_delta_ns`
relative to each:

```
sync_delta_FM1 = target_onset_sample - FM1_sync_sample  (on same clock)
sync_delta_FM2 = target_onset_sample - FM2_sync_sample  (on same clock)
```

The difference:

```
delta_FM1 - delta_FM2 = FM2_sync_sample - FM1_sync_sample
```

should equal the propagation delay difference between the two FM stations:

```
expected = (dist(node, FM1) - dist(node, FM2)) / c  +  (FM1_phase - FM2_phase)
```

Both distances are computed from FCC coordinates; the pilot phase difference is
measured.  The residual after subtracting the expected value reveals any
systematic error in pilot extraction (BPF phase response, cross-correlation
bias, etc.).

### What dual-station validates

1. **FCC coordinate accuracy** - large residual implies a coordinate error in one station.
2. **Pilot extraction quality** - residual jitter reveals the per-event timing noise floor.
3. **Per-node calibration** - `calibration_offset_ns` in `clock:` can be tuned until
   the dual-station residual is minimised.
4. **Inter-node consistency** - if two nodes both show the same dual-station residual,
   their `sync_delta_ns` values are consistent even without a test transmitter.

### Current implementation status

`SyncSignalConfig` accepts a `secondary_station` field and the config validation
supports it, but active dual-station cross-checking is not yet implemented in the
pipeline (the second station would require either a second sync decimation chain
or time-multiplexed processing).  It is planned for a future sprint.

---

## Minimum Correlation Peak Threshold

The `min_corr_peak` parameter in `sync_signal` sets the minimum acceptable
**internal pilot correlation quality** for a `SyncEvent` to be accepted by
`DeltaComputer.feed_sync()`.

> Note: with the RDS sync detector, `corr_peak` is the quality of the
> *parallel pilot extraction path* used for crystal calibration, not the
> RDS bit detection itself.  Both run on the same audio stream.  If the
> pilot is too weak to extract reliably, the RDS bit recovery is also
> almost certainly compromised, so this single threshold serves both
> functions.

| corr_peak range | Condition | Action |
|-----------------|-----------|--------|
| 0.7 - 1.0 | Strong pilot, good SNR | Used |
| 0.3 - 0.7 | Usable pilot, some noise | Used (default threshold is 0.1) |
| 0.1 - 0.3 | Weak pilot | Filtered if min_corr_peak >= 0.3 |
| < 0.1 | Very poor pilot or interference | Always filtered |

**Recommended settings:**

- Production nodes with a clear line-of-sight FM signal: `min_corr_peak: 0.3`
- Urban nodes with multipath or adjacent-channel interference: `min_corr_peak: 0.2`
- Development/testing: `min_corr_peak: 0.1` (accept almost anything)

Setting `min_corr_peak` too high increases the risk of gaps in the sync event
stream (see [02-signal-processing.md](02-signal-processing.md), Stage 5).  The
`max_sync_age_samples` of 80 ms (~ 95 RDS bit periods) provides headroom for
up to ~95 consecutive filtered sync events before a carrier measurement is
dropped.

---

## Practical Notes

### AGC and gain

For `rspduo` mode with `sync_gain_db: "auto"`, the SDRplay AGC typically settles
on a gain that keeps the FM station 10-20 dB below full scale - ideal.  Manual
gain (`sync_gain_db: 30`) may be preferable in environments with a nearby
very strong FM station (to prevent ADC clipping) or very weak FM signal (to
maximise pilot SNR).

Check `corr_peak` values in the log at startup.  Values consistently above 0.5
indicate a healthy pilot.  Values below 0.3 suggest gain adjustment or station
selection is needed.

### FM stereo + RDS requirement

The pilot tone and RDS subcarrier are both part of FM **stereo** broadcasts.
Nearly all commercial music stations broadcast in stereo with RDS.  News/talk
stations and very small low-power stations occasionally broadcast in mono (no
pilot, no RDS) or in stereo without RDS.  Symptoms:

- `corr_peak` consistently near zero despite a strong FM signal -> the
  station may be mono.
- `corr_peak` good (~0.7) but the RDS event rate is far below 1188/s ->
  the station has stereo but no RDS.

In either case, select a different station.

### Startup latency

On startup, the RDS sync detector takes ~50 ms (50 symbols * 842 usec) for
the M&M timing loop to converge before it begins emitting SyncEvents.  This
warmup is configurable via the `RDSSyncDetector(warmup_symbols=...)`
constructor argument; the default of 50 is generous for typical signal
quality.

In parallel, the internal pilot extraction path runs from the very first
audio buffer.  However, the first sync event has no prior phase
measurement for CrystalCalibrator to compute a correction from, so
`sample_rate_correction = 1.0` for the first event.  The calibrator converges
to a stable estimate within ~1 second (100 pilot windows).  Carrier events in
the first ~1 second may have slightly larger crystal error contributions
(~1-2 usec additional uncertainty).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
