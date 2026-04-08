# Beagle Signal Processing Pipeline

## Overview

The signal processing pipeline converts raw IQ samples from the SDR into two kinds
of events: **SyncEvents** (RDS bit-transition timing pulses) and **CarrierOnset/Offset
events** (LMR carrier edges).  Both are expressed as sample indices in the same
continuous ADC clock domain so that `DeltaComputer` can compute `sync_delta_ns`
by pure sample-index arithmetic.

```
SDR IQ (2 MSPS complex64)
     |
     +------------------------------> sync_decimator  (8x, LPF 128 kHz)
     |                                      |   -> 256 kHz complex IQ
     |                                FMDemodulator
     |                                      |   -> 256 kHz float32 audio (Hz)
     |                                RDSSyncDetector
     |                                  (-57 kHz freq shift, LPF, /10,
     |                                   M&M timing recovery + Costas loop)
     |                                      |
     |                                SyncEvent (sample_index, corr_peak,
     |                                           sample_rate_correction)
     |                                  one event per RDS bit transition
     |                                  (~1188/sec, ~842 usec apart)
     |
     +------------------------------> target_decimator (32x, LPF 25 kHz)
                                            |   -> 64 kHz complex IQ
                                      [DC removal: iq - mean(iq)]
                                       CarrierDetector
                                            |
                               CarrierOnset / CarrierOffset
                               (sample_index in sync-dec space)
                                            |
                                      DeltaComputer
                                            |
                                   TDOAMeasurement (sync_delta_ns)
```

> **Historical note**: Beagle previously used `FMPilotSyncDetector`, which
> emitted a SyncEvent at every 19 kHz pilot zero-crossing window.  That
> approach was discovered to have a fundamental flaw: pilot zero-crossings are
> physically indistinguishable, so different nodes locked to different
> crossings, producing an unresolvable `N x 52.6 usec` ambiguity in cross-node
> sync_delta subtraction.  RDS bit transitions are anchored to a data signal
> that all nodes can identify uniquely, eliminating the ambiguity.  The
> `FMPilotSyncDetector` class still exists in the codebase as a diagnostic
> tool but is not in the production pipeline.

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
via `scipy.signal.lfilter` with persistent state - consecutive buffers produce a
continuous output stream with no boundary artifacts.

The I and Q channels are filtered separately (both are real-valued within
`lfilter`) and recombined into `complex64` output.

### Parameters in use

| Chain | Input rate | Factor | Cutoff | Output rate |
|-------|-----------|--------|--------|-------------|
| Sync (RTL-SDR) | 2.048 MSPS | /8 | 128 kHz | **256 kHz** |
| Sync (RSPduo)  | 2.000 MSPS | /8 | 128 kHz | **250 kHz** |
| Target (both)  | 2.048/2.000 MSPS | /32 | 25 kHz | **64/62.5 kHz** |

The 128 kHz sync cutoff passes the full +/-75 kHz FM deviation including the
stereo pilot at 19 kHz and pilot sidebands at 23-53 kHz.  The 25 kHz target
cutoff matches the 25 kHz channel bandwidth of narrowband LMR.

### Sample-index arithmetic

`Decimator.process()` always outputs exactly `len(input) // decimation` samples.
The output sample at index `k` corresponds to input sample at index `k x decimation`
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
audio[n] = angle(conj(iq[n-1]) x iq[n]) / (2pi) x sample_rate_hz
```

This is the discrete-time derivative of the instantaneous phase, expressed in Hz.
For a pure FM signal with modulation deviation `deltaf`, the output is a clean
`deltaf`-amplitude real sinusoid at the audio frequency.

`FMDemodulator` maintains `self._prev` - the last IQ sample from the previous
buffer - so the first output sample of each call is computed correctly across
buffer boundaries.

**Output units:** instantaneous frequency in Hz, centred at 0 Hz (corresponding
to the carrier frequency).  The FM stereo pilot at exactly 19,000 Hz appears as
a 19 kHz sinusoid in the output.

---

## Stage 3: RDS Sync Detection (`pipeline/rds_sync_detector.py`)

### Purpose

Extract precise timing pulses from the **RDS** (Radio Data System) BPSK data
stream modulated on the FM stereo subcarrier at 57 kHz.  One `SyncEvent` is
emitted per recovered RDS bit transition (~1188 per second, exactly
**pilot/16 = 1187.5 Hz**), carrying a sample index accurate to better than
0.1 usec via the M&M timing loop's sub-sample interpolation.

### Why the RDS bit clock?

The RDS subcarrier is the third harmonic of the 19 kHz pilot, **phase-locked
to the pilot by the IEC 62106 / NRSC-4-B standard**.  RDS data is BPSK
modulated at exactly **1187.5 bps = pilot/16** -- a frequency-division of the
pilot.  Bit boundaries are therefore deterministic features of the broadcast
signal: every node receiving the same FM station identifies the **same** bit
transition as the same physical event.  Contrast with raw pilot
zero-crossings, which happen every 52.6 microseconds and are pairwise
indistinguishable, so different nodes lock to different crossings and produce
an unresolvable cross-node ambiguity.

### Processing chain

```
audio (256 kHz, real, instantaneous freq in Hz)
  |
  v
 [normalise to unit RMS]               -- so M&M loop gain is signal-independent
  |
  v
 [freq shift -57 kHz]                  -- multiply by exp(-j 2pi 57000 t)
  |
  v
 [LPF: 101-tap firwin, 7.5 kHz cutoff]
  |
  v
 [decimate by 10]                      -- working rate now 25 kHz
  |
  v
 [Mueller-Muller timing recovery]      -- cubic interpolation, 16 sps nominal
  |                                       (RDS rate: 25000 / 1187.5 = ~21.05 sps)
  |  one symbol per ~21 input samples (~842 usec)
  v
 [Costas loop]                         -- residual phase/frequency correction
  |
  v
 [bit decisions + differential decode]
  |
  v
 SyncEvent at the input-stream sample index of each bit boundary
```

The detector emits one `SyncEvent` per recovered bit (after a configurable
warmup period during which the M&M timing loop converges; default 50 symbols
= ~42 ms).  Each event carries:

- `sample_index` (float, sub-sample precision in the 256 kHz sync-dec stream)
- `corr_peak` -- the most recent **pilot** correlation value (the detector
  still extracts the pilot internally for crystal calibration; see below)
- `sample_rate_correction` -- the current `CrystalCalibrator` factor
- `pilot_phase_rad` -- accumulated unwrapped pilot phase

### Pilot extraction (for crystal calibration)

The RDS detector contains a parallel pilot extraction path that runs alongside
the RDS chain.  Every 10 ms it does a 19 kHz BPF + complex correlation against
a `exp(j 2pi 19000 t)` template (the same algorithm `FMPilotSyncDetector`
uses), feeds the unwrapped phase into `CrystalCalibrator`, and stores the
resulting correction factor on the next emitted `SyncEvent`.

This is **only** used for crystal calibration -- the SyncEvent sample index
itself comes from the M&M timing loop, not the pilot.  The pilot's role is
reduced to "report the SDR's true sample rate so we can convert sample indices
to nanoseconds accurately."

### M&M timing loop and sub-sample precision

Mueller-Muller is a feedback timing-recovery algorithm for symbol-synchronous
demodulation: it adjusts a fractional sample-position estimate (`mu`) at each
symbol so that successive samples land on the symbol centre.  Beagle's
implementation interpolates the input stream with a Catmull-Rom cubic so the
sample position can move in continuous fractional steps.

The output `sample_index` is the float-valued input-stream position at which
the bit transition was recovered, with full sub-sample precision (the float
value is propagated all the way through to the server's TDOA calculation).
On synthetic data the per-bit interval jitter is < 0.01 usec; on a healthy
RSPduo capture of KUOW 94.9 it's measured at ~0.06 usec.

### Costas loop

The recovered symbols still carry residual phase/frequency offset (the FM
demod centre frequency vs. the RDS subcarrier are not exactly 57 kHz apart in
the receiver's view because of LO offset and crystal error).  A standard
2nd-order Costas loop in the M&M output stream corrects this and produces
real-valued bit values at the slicer.

### Buffer management and gap handling

The `RDSSyncDetector` keeps an internal `mm_buf` of decimated samples between
calls so that the M&M loop can reach across `process()` boundaries without
restarting.  Two invariants:

1. The buffer always retains at least 2 samples after each trim, so the M&M
   loop's "outstanding advance" past the current buffer end is preserved for
   the next call.  Violating this invariant caused a periodic 20-sample
   backward jump in early development; see commit `54b2f5f` for the fix.

2. **Long gaps** (`start_sample` jumps more than 1 second ahead of the
   expected next sample) trigger a full state reset including warmup.

3. **Short gaps** (typical of `freq_hop` mode where the SDR retunes between
   sync blocks) trigger only a partial reset: the LPF / decimation / M&M
   buffers are flushed (because the audio between blocks is from a different
   frequency and is meaningless) but M&M timing state, Costas lock, warmup
   counter, and crystal calibration are **preserved**.  The `-57 kHz`
   oscillator phase is also advanced across the gap so it stays coherent.

### Disambiguation period

The bit period is `1 / 1187.5 Hz = 842.105 usec`.  The maximum geometric TDOA
for a 100 km baseline is `100 km / c = 333 usec`, comfortably less than
`T_sync / 2 = 421 usec`, so the server can disambiguate cross-node sample
counts via `n = round((raw_ns + path_correction_ns) / 842,105)` without
ambiguity.  This is implemented in `beagle_server/tdoa.py` -- see
[03-timing-model.md](03-timing-model.md) for the full derivation.

---

## Stage 4: Target Channel - DC Removal and Carrier Detection

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
`window_samples` (default 64 samples ~ 1 ms at 64 kHz).  For each window:

```
power_lin = mean(|iq|^2)
power_db  = 10 x log10(power_lin + epsilon)
```

The `epsilon = 1e-30` floor prevents log(0).  The result is in **dBFS** (0 dBFS = full
scale, all real-world signals are negative).

### Dual-threshold state machine

`CarrierDetector` implements a hysteresis comparator to avoid chattering around
the threshold:

```
State IDLE  ->  ACTIVE  when power_db >= onset_threshold_db  (default -30 dBFS)
State ACTIVE -> IDLE    when power_db <= offset_threshold_db  (default -40 dBFS)
```

`offset_threshold_db < onset_threshold_db` is enforced at construction time.
A 10 dB gap between the two thresholds is the current default.

### Minimum hold (`carrier_min_hold_windows`)

The `min_hold_windows` parameter (default 1) requires a carrier to be detected
in `N` consecutive power windows before an onset is declared.  This suppresses
single-window noise spikes.  Set to 4 (~ 4 ms) to further reduce false positives
in noisy RF environments.

### Event sample indices

`CarrierOnset` and `CarrierOffset` carry a `sample_index` in the
**target-decimated** sample domain.  `NodePipeline.process_target_buffer()`
converts this to the **sync-decimated** domain before passing to `DeltaComputer`:

```python
event_in_sync_space = event.sample_index * target_decimation // sync_decimation
                    = event.sample_index x 32 // 8
                    = event.sample_index x 4
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
best_sync = max({ s in sync_events :  s.sample_index <= T
                                 and T - s.sample_index <= max_sync_age_samples })
```

`max_sync_age_samples` (default 20,480 ~ 80 ms at 256 kHz) is the maximum
tolerated age of a sync event.  This window spans approximately 8 sync periods,
providing robust matching even if several consecutive FM pilot windows are
corrupted by RF noise.

### Measurement

```
delta_samples = T - best_sync.sample_index
corrected_rate = nominal_rate x best_sync.sample_rate_correction
sync_delta_ns = round(delta_samples x 1_000_000_000 / corrected_rate)
```

### Offset measurements

Both `feed_onset()` and `feed_offset()` produce `TDOAMeasurement` objects.
The `event_type` field (`"onset"` or `"offset"`) is carried through to
`CarrierEvent` and on to the aggregation server.  **The server must pair
onset-with-onset and offset-with-offset across nodes** - mixing edge types
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
more than `max_sync_age_samples` ahead of it - this means no preceding sync will
ever become available.  A `WARNING` log is emitted on each drop.

---

## End-to-End Sample Rate Summary

For `freq_hop` mode (RTL-SDR @ 2.048 MSPS):

| Stage | Rate | Period |
|-------|------|--------|
| ADC input | 2,048,000 sps | 0.488 usec/sample |
| Sync after /8 | 256,000 sps | 3.9 usec/sample |
| RDS after additional /10 | 25,600 sps | 39 usec/sample |
| Target after /32 | 64,000 sps | 15.6 usec/sample |
| RDS bit period | - | **842.1 usec** (one SyncEvent per bit) |
| RDS sync event rate | 1187.5 / sec | (= pilot/16, exact) |
| Target power window | - | 1 ms (64 target samples) |
| max_sync_age | - | 80 ms (20,480 sync samples = ~95 RDS bits) |

For `rspduo` mode (RSPduo @ 2.000 MSPS):

| Stage | Rate | Period |
|-------|------|--------|
| ADC input | 2,000,000 sps | 0.500 usec/sample |
| Sync after /8 | 250,000 sps | 4.0 usec/sample |
| RDS after additional /10 | 25,000 sps | 40 usec/sample |
| Target after /32 | 62,500 sps | 16.0 usec/sample |
| RDS bit period | - | **842.1 usec** (one SyncEvent per bit) |
| RDS sync event rate | 1187.5 / sec | (= pilot/16, exact) |
| RSPduo buffer | - | 32.8 ms (65,536 raw samples = ~39 RDS bits) |
| max_sync_age | - | 82 ms (20,480 sync samples = ~97 RDS bits) |

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
