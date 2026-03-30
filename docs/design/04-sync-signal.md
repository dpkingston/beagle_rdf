# Beagle Sync Signal: FM Stereo Pilot

## Why FM Broadcast?

Beagle needs each node to produce a timing reference that is:

1. **Simultaneous across all nodes** — every node must reference the same physical
   event at the same instant.
2. **Precisely known** — the reference event must be localizable in sample space
   to sub-microsecond accuracy.
3. **Passively receivable** — no special infrastructure, no active transmitter.
4. **Location-known** — the reference transmitter's geographic position must be
   known accurately for path-delay correction.

FM broadcast fits all four requirements.  Every radio in the coverage area receives
the same broadcast simultaneously (modulo propagation delay, which is correctable).
The stereo pilot tone at exactly **19,000.000 Hz** is locked to the station's
frequency standard, which is in turn traceable to GPS/UTC.  FCC license data
documents each transmitter's location to ~100 m.

---

## The 19 kHz FM Stereo Pilot Tone

FM stereo broadcasts contain a **pilot tone at exactly 19 kHz** that receivers use
to detect and decode the stereo subcarrier.  Properties relevant to Beagle:

| Property | Value |
|----------|-------|
| Frequency | 19,000.000 Hz (±<0.1 Hz, locked to FCC-licensed frequency standard) |
| Level | typically −20 to −15 dBc relative to mono audio |
| Phase continuity | continuous sinusoid for the life of the broadcast |
| Coverage | city-wide (typically 20–60 km radius from a mountaintop transmitter) |
| Availability | 24/7 for major commercial stations |

The pilot is present in the demodulated FM audio as a clean sinusoid.  A 10 ms
cross-correlation window at 256 kHz contains 2560 samples with 190 complete cycles
of the 19 kHz sinusoid, giving excellent signal energy even at moderate FM SNR.

### Timing precision

The phase of a 19 kHz sinusoid changes by 2π × 19,000 ≈ 119,380 rad/s.  A 1 µs
timing error corresponds to a phase error of 0.12 rad (≈ 7°), which is resolvable
from a cross-correlation over even a few cycles.  In practice, a 10 ms window
with good SNR achieves sub-microsecond timing repeatability.  The CrystalCalibrator
(see [03-timing-model.md](03-timing-model.md)) removes the dominant RTL-SDR crystal
error, leaving pilot phase measurement as the primary uncertainty at <1 µs.

---

## Seattle FM Stations

The sync station is configured per-node in `node.yaml` under `sync_signal:`.
The following stations provide good coverage across the Seattle/Eastside area:

| Station | Frequency | Transmitter Location | Notes |
|---------|-----------|---------------------|-------|
| KISW | 99.9 MHz | 47.6253° N, 122.3563° W (Queen Anne Hill) | Primary; used in TDOAv1/v2 |
| KUOW | 94.9 MHz | 47.6553° N, 122.3110° W (Capitol Hill) | Secondary; good cross-check |
| KEXP | 90.3 MHz | 47.6619° N, 122.3487° W (Capitol Hill) | Tertiary option |
| KING-FM | 98.1 MHz | 47.5235° N, 122.1306° W (Cougar Mountain) | Useful for eastern nodes |

**Always verify FCC coordinates** before deployment.  Transmitter locations are
available in the FCC AM/FM Query database (https://www.fcc.gov/media/radio/fm-query).
Coordinates accurate to ~100 m are achievable; 100 m translates to <0.33 µs path-delay
correction error — negligible for the system's overall accuracy target.

### Selecting a station

- Prefer stations with transmitters **distant from the node by different amounts
  across your node network** — this maximises the path-delay correction variation
  and makes calibration cross-checks more sensitive.
- Prefer stations with **strong local signal** (corr_peak consistently >0.6).
  Set `min_corr_peak: 0.3` in `sync_signal` as a minimum quality threshold.
- For urban nodes with strong multipath, the narrow 100 Hz BPF in the pilot
  extractor significantly reduces multipath effects.
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
event fires 16.7 µs later than Node B's (for the same broadcast sample), not
because of any target-signal timing difference, but purely due to propagation.
If uncorrected, this appears as a 16.7 µs TDOA bias.

### The correction

```
TDOA_AB_raw       = sync_delta_A − sync_delta_B

propagation_A     = dist(FM_transmitter, node_A) / c
propagation_B     = dist(FM_transmitter, node_B) / c

TDOA_AB_corrected = TDOA_AB_raw − (propagation_A − propagation_B)
                  = TDOA_AB_raw − (d_A − d_B) / c
```

The correction term `(d_A − d_B) / c` is computed by the aggregation server from
the `sync_transmitter` coordinates (carried in every `CarrierEvent`) and each
node's `node_location`.  Distances are computed using the haversine formula.

### Magnitude

For two nodes separated by 30 km with a transmitter 20 km from one and 40 km from
the other:

```
correction = (40 km − 20 km) / 299,792 km/s ≈ 66.7 µs
```

Without this correction, the computed TDOA would have a 66.7 µs error, placing
the hyperbola ~20 km from the true answer — useless for geolocation.

### Accuracy

FCC-documented transmitter coordinates are typically accurate to ~100 m.  The error
in path-delay correction from a 100 m coordinate error is at most:

```
Δcorrection ≤ 100 m / 299,792,458 m/s ≈ 0.33 µs
```

This is well below the system's ~1–5 µs total timing budget.

---

## Dual-Station Self-Calibration

Configuring a `secondary_station` in `sync_signal` enables a powerful passive
calibration check that requires no test transmitter.

### Principle

When a node tracks two FM stations simultaneously, it computes `sync_delta_ns`
relative to each:

```
sync_delta_FM1 = target_onset_sample − FM1_sync_sample  (on same clock)
sync_delta_FM2 = target_onset_sample − FM2_sync_sample  (on same clock)
```

The difference:

```
delta_FM1 − delta_FM2 = FM2_sync_sample − FM1_sync_sample
```

should equal the propagation delay difference between the two FM stations:

```
expected = (dist(node, FM1) − dist(node, FM2)) / c  +  (FM1_phase − FM2_phase)
```

Both distances are computed from FCC coordinates; the pilot phase difference is
measured.  The residual after subtracting the expected value reveals any
systematic error in pilot extraction (BPF phase response, cross-correlation
bias, etc.).

### What dual-station validates

1. **FCC coordinate accuracy** — large residual implies a coordinate error in one station.
2. **Pilot extraction quality** — residual jitter reveals the per-event timing noise floor.
3. **Per-node calibration** — `calibration_offset_ns` in `clock:` can be tuned until
   the dual-station residual is minimised.
4. **Inter-node consistency** — if two nodes both show the same dual-station residual,
   their `sync_delta_ns` values are consistent even without a test transmitter.

### Current implementation status

`SyncSignalConfig` accepts a `secondary_station` field and the config validation
supports it, but active dual-station cross-checking is not yet implemented in the
pipeline (the second station would require either a second sync decimation chain
or time-multiplexed processing).  It is planned for a future sprint.

---

## Minimum Correlation Peak Threshold

The `min_corr_peak` parameter in `sync_signal` sets the minimum acceptable
cross-correlation quality for a `SyncEvent` to be used in a measurement.  The
filtering happens in `DeltaComputer.feed_sync()`.

| corr_peak range | Condition | Action |
|-----------------|-----------|--------|
| 0.7 – 1.0 | Strong pilot, good SNR | Used |
| 0.3 – 0.7 | Usable pilot, some noise | Used (default threshold is 0.1) |
| 0.1 – 0.3 | Weak pilot | Filtered if min_corr_peak ≥ 0.3 |
| < 0.1 | Very poor pilot or interference | Always filtered |

**Recommended settings:**

- Production nodes with a clear line-of-sight FM signal: `min_corr_peak: 0.3`
- Urban nodes with multipath or adjacent-channel interference: `min_corr_peak: 0.2`
- Development/testing: `min_corr_peak: 0.1` (accept almost anything)

Setting `min_corr_peak` too high increases the risk of gaps in the sync event
stream (see [02-signal-processing.md](02-signal-processing.md), Stage 5).  The
`max_sync_age_samples` of 80 ms (≈ 8 sync periods) provides headroom for up to
8 consecutive filtered sync events before a carrier measurement is dropped.

---

## Practical Notes

### AGC and gain

For `rspduo` mode with `sync_gain_db: "auto"`, the SDRplay AGC typically settles
on a gain that keeps the FM station 10–20 dB below full scale — ideal.  Manual
gain (`sync_gain_db: 30`) may be preferable in environments with a nearby
very strong FM station (to prevent ADC clipping) or very weak FM signal (to
maximise pilot SNR).

Check `corr_peak` values in the log at startup.  Values consistently above 0.5
indicate a healthy pilot.  Values below 0.3 suggest gain adjustment or station
selection is needed.

### FM stereo requirement

The pilot tone is part of FM **stereo** broadcasts.  Nearly all commercial music
stations broadcast in stereo.  News/talk stations occasionally broadcast in mono
(no pilot tone).  If `corr_peak` is consistently near zero despite a strong FM
signal, the station may be mono — select a different station.

### Propagation delay at node startup

On startup, the pipeline begins emitting SyncEvents within the first 10 ms of SDR
data.  However, the very first sync event has no prior SyncEvent for
CrystalCalibrator to compute a correction, so `sample_rate_correction = 1.0` for
the first event.  The calibrator converges to a stable estimate within ~1 second
(100 sync windows).  Carrier events in the first second may have slightly larger
crystal error contributions (~1–2 µs additional uncertainty).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License — see [LICENSE](../../LICENSE).
