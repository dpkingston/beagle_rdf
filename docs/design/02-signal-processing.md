# Beagle Signal Processing Pipeline

## Overview

The signal processing pipeline converts raw IQ samples from the SDR into two kinds
of events: **SyncEvents** (FM pilot timing pulses) and **CarrierOnset/Offset events**
(LMR carrier edges).  Both are expressed as sample indices in the same continuous
ADC clock domain so that `DeltaComputer` can compute `sync_delta_ns` by pure
sample-index arithmetic.

```
SDR IQ (2 MSPS complex64)
     │
     ├──────────────────────────────► sync_decimator  (8×, LPF 128 kHz)
     │                                      │   → 256 kHz complex IQ
     │                                FMDemodulator
     │                                      │   → 256 kHz float32 audio (Hz)
     │                               FMPilotSyncDetector
     │                                      │
     │                                SyncEvent (sample_index, corr_peak,
     │                                           sample_rate_correction)
     │
     └──────────────────────────────► target_decimator (32×, LPF 25 kHz)
                                            │   → 64 kHz complex IQ
                                      [DC removal: iq − mean(iq)]
                                       CarrierDetector
                                            │
                               CarrierOnset / CarrierOffset
                               (sample_index in sync-dec space)
                                            │
                                      DeltaComputer
                                            │
                                   TDOAMeasurement (sync_delta_ns)
```

For `freq_hop` mode the single ADC alternates frequencies, so the caller drives
separate `process_sync_buffer()` / `process_target_buffer()` calls with the
correct ADC sample offsets.  For `rspduo` mode both channels share one ADC
clock and are processed with the same `raw_start_sample`.

---

## Stage 1: Decimation (`pipeline/decimator.py`)

### Purpose

Reduce the SDR sample rate to a narrower working rate, rejecting out-of-band
energy and reducing CPU load for downstream stages.

### Implementation

`Decimator` applies a single-stage **FIR low-pass filter** followed by integer
downsampling.  The filter is designed with `scipy.signal.firwin` using a Hamming
window (127 taps by default, giving ~80 dB stopband attenuation), then applied
via `scipy.signal.lfilter` with persistent state — consecutive buffers produce a
continuous output stream with no boundary artifacts.

The I and Q channels are filtered separately (both are real-valued within
`lfilter`) and recombined into `complex64` output.

### Parameters in use

| Chain | Input rate | Factor | Cutoff | Output rate |
|-------|-----------|--------|--------|-------------|
| Sync (RTL-SDR) | 2.048 MSPS | ÷8 | 128 kHz | **256 kHz** |
| Sync (RSPduo)  | 2.000 MSPS | ÷8 | 128 kHz | **250 kHz** |
| Target (both)  | 2.048/2.000 MSPS | ÷32 | 25 kHz | **64/62.5 kHz** |

The 128 kHz sync cutoff passes the full ±75 kHz FM deviation including the
stereo pilot at 19 kHz and pilot sidebands at 23–53 kHz.  The 25 kHz target
cutoff matches the 25 kHz channel bandwidth of narrowband LMR.

### Sample-index arithmetic

`Decimator.process()` always outputs exactly `len(input) // decimation` samples.
The output sample at index `k` corresponds to input sample at index `k × decimation`
(after filtering).  The caller tracks the raw ADC sample count and computes:

```
dec_start = raw_start_sample // decimation
```

This integer division gives the correct output-domain start index, and the same
calculation is applied consistently to both channels so their sample-domain indices
can be compared directly.

---

## Stage 2: FM Demodulation (`pipeline/demodulator.py`)

### Purpose

Convert the complex FM IQ stream into a real-valued instantaneous-frequency
signal (in Hz) that contains the baseband audio, including the 19 kHz stereo pilot.

### Algorithm

The standard FM discriminator:

```
audio[n] = angle(conj(iq[n-1]) × iq[n]) / (2π) × sample_rate_hz
```

This is the discrete-time derivative of the instantaneous phase, expressed in Hz.
For a pure FM signal with modulation deviation `Δf`, the output is a clean
`Δf`-amplitude real sinusoid at the audio frequency.

`FMDemodulator` maintains `self._prev` — the last IQ sample from the previous
buffer — so the first output sample of each call is computed correctly across
buffer boundaries.

**Output units:** instantaneous frequency in Hz, centred at 0 Hz (corresponding
to the carrier frequency).  The FM stereo pilot at exactly 19,000 Hz appears as
a 19 kHz sinusoid in the output.

---

## Stage 3: FM Pilot Sync Detection (`pipeline/sync_detector.py`)

### Purpose

Extract precise timing pulses from the 19 kHz FM stereo pilot tone.  One
`SyncEvent` is emitted every `sync_period_ms` (default 10 ms), carrying a
sample index accurate to <1 µs via sub-sample phase interpolation.

### Processing steps (per 10 ms window)

1. **Accumulate audio** into a rolling buffer until `sync_period_samples`
   (= `round(sample_rate × 0.010)`) samples are available.

2. **19 kHz narrow BPF** — a 255-tap FIR bandpass filter with ±100 Hz
   half-bandwidth (designed with `firwin`, Hamming window) isolates the pilot
   sinusoid from voice/data content.  Filter state is preserved across calls.

3. **Complex cross-correlation** with the template
   `exp(j 2π × 19000 × t)` over the window:

   ```
   corr = Σ  filtered[n] × conj(template[n])
   ```

   The result is a single complex number.  Its magnitude (normalised by window
   energy) gives `corr_peak` ∈ [0, 1]; its angle is the pilot phase.

4. **Normalisation:**

   ```
   corr_peak = |corr| / (N × RMS(filtered) × RMS(template))
   ```

   A `corr_peak` of 1.0 means a perfect pilot sinusoid.  Typical values for a
   good FM signal are 0.7–0.95.  `DeltaComputer` discards sync events below
   `min_corr_peak` (default 0.1).

5. **Sample index** — assigned to the centre of the window:
   `sample_index = window_start_sample + period_samples // 2`.

6. **CrystalCalibrator** (described in [03-timing-model.md](03-timing-model.md))
   accumulates the pilot phase across successive windows to measure the SDR's
   actual sample rate and produces a `sample_rate_correction` factor attached
   to each `SyncEvent`.

### Gap detection

If `start_sample` arrives more than one sync period ahead of the expected next
sample (indicating a non-contiguous ADC stream, as occurs between alternating
blocks in `freq_hop` mode), the internal buffer and BPF state are reset so that
a corrupted cross-correlation window is never emitted.

---

## Stage 4: Target Channel — DC Removal and Carrier Detection

### DC offset removal (`pipeline/pipeline.py`)

Before the target IQ stream is passed to the decimator, the per-buffer DC offset
is subtracted:

```python
iq = iq - np.mean(iq)
```

Direct-conversion SDRs (RTL-SDR, RSPduo) produce a DC spike at 0 Hz due to
LO leakage.  This spike would otherwise appear as a large bias in the decimated
IQ, keeping the power detector permanently above the onset threshold.  Subtracting
the block mean is safe: FM-modulated narrowband LMR has approximately zero mean
over any window >10 ms.

### Power measurement (`pipeline/carrier_detect.py`)

After decimation the IQ stream is divided into non-overlapping windows of
`window_samples` (default 64 samples ≈ 1 ms at 64 kHz).  For each window:

```
power_lin = mean(|iq|²)
power_db  = 10 × log10(power_lin + ε)
```

The `ε = 1e-30` floor prevents log(0).  The result is in **dBFS** (0 dBFS = full
scale, all real-world signals are negative).

### Dual-threshold state machine

`CarrierDetector` implements a hysteresis comparator to avoid chattering around
the threshold:

```
State IDLE  →  ACTIVE  when power_db ≥ onset_threshold_db  (default −30 dBFS)
State ACTIVE → IDLE    when power_db ≤ offset_threshold_db  (default −40 dBFS)
```

`offset_threshold_db < onset_threshold_db` is enforced at construction time.
A 10 dB gap between the two thresholds is the current default.

### Minimum hold (`carrier_min_hold_windows`)

The `min_hold_windows` parameter (default 1) requires a carrier to be detected
in `N` consecutive power windows before an onset is declared.  This suppresses
single-window noise spikes.  Set to 4 (≈ 4 ms) to further reduce false positives
in noisy RF environments.

### Event sample indices

`CarrierOnset` and `CarrierOffset` carry a `sample_index` in the
**target-decimated** sample domain.  `NodePipeline.process_target_buffer()`
converts this to the **sync-decimated** domain before passing to `DeltaComputer`:

```python
event_in_sync_space = event.sample_index * target_decimation // sync_decimation
                    = event.sample_index × 32 // 8
                    = event.sample_index × 4
```

This linear scaling preserves relative timing across the two decimated streams.

---

## Stage 5: Delta Computation (`pipeline/delta.py`)

### Purpose

Match each `CarrierOnset` or `CarrierOffset` to the most recent preceding
`SyncEvent` and compute `sync_delta_ns`.

### Matching rule

For a carrier event at sync-space sample index `T`, the best sync event is:

```
best_sync = max({ s ∈ sync_events :  s.sample_index ≤ T
                                 and T − s.sample_index ≤ max_sync_age_samples })
```

`max_sync_age_samples` (default 20,480 ≈ 80 ms at 256 kHz) is the maximum
tolerated age of a sync event.  This window spans approximately 8 sync periods,
providing robust matching even if several consecutive FM pilot windows are
corrupted by RF noise.

### Measurement

```
delta_samples = T − best_sync.sample_index
corrected_rate = nominal_rate × best_sync.sample_rate_correction
sync_delta_ns = round(delta_samples × 1_000_000_000 / corrected_rate)
```

### Offset measurements

Both `feed_onset()` and `feed_offset()` produce `TDOAMeasurement` objects.
The `event_type` field (`"onset"` or `"offset"`) is carried through to
`CarrierEvent` and on to the aggregation server.  **The server must pair
onset-with-onset and offset-with-offset across nodes** — mixing edge types
produces meaningless TDOA values.

Using both edges doubles the measurement rate per transmission and ensures
a measurement is produced for a carrier that was already active when the node
started (the first detected event will be an offset, not an onset).

### Pending events and age-out

If no matching sync event exists when a carrier event arrives (e.g. at node
startup before any sync events have been received), the carrier event is held
in a `_pending_events` list.  Each subsequent call to `feed_sync()` / `feed_onset()`
/ `feed_offset()` triggers `_flush()`, which reattempts resolution.

A pending event is **dropped** when the newest available sync event has moved
more than `max_sync_age_samples` ahead of it — this means no preceding sync will
ever become available.  A `WARNING` log is emitted on each drop.

---

## End-to-End Sample Rate Summary

For `freq_hop` mode (RTL-SDR @ 2.048 MSPS):

| Stage | Rate | Period |
|-------|------|--------|
| ADC input | 2,048,000 sps | 0.488 µs/sample |
| Sync after ÷8 | 256,000 sps | 3.9 µs/sample |
| Target after ÷32 | 64,000 sps | 15.6 µs/sample |
| Sync event period | — | 10 ms (2,560 sync samples) |
| Target power window | — | 1 ms (64 target samples) |
| max_sync_age | — | 80 ms (20,480 sync samples) |

For `rspduo` mode (RSPduo @ 2.000 MSPS):

| Stage | Rate | Period |
|-------|------|--------|
| ADC input | 2,000,000 sps | 0.500 µs/sample |
| Sync after ÷8 | 250,000 sps | 4.0 µs/sample |
| Target after ÷32 | 62,500 sps | 16.0 µs/sample |
| Sync event period | — | 10 ms (2,500 sync samples) |
| RSPduo buffer | — | 32.8 ms (65,536 raw samples) |
| max_sync_age | — | 82 ms (20,480 sync samples) |

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License — see [LICENSE](../../LICENSE).
