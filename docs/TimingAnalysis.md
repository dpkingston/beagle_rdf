# TDOA Timing Accuracy Analysis

_Last substantive pipeline updates reflected here: 2026-04-19 (rev 3)._
_Earlier sections retained for archaeological context; see the Status banner below._

## Status as of 2026-04-19

Several of the stages described below were replaced or substantially rewritten
after the 2026-03-19 baseline:

- **RDS sync extraction** -- replaced Mueller-Muller timing recovery + Costas
  with **pilot-phase-derived bit boundaries**.  Cross-node onset spread
  dropped from ~250 µs (M&M era) to ~105 ns (pilot-derived era) over the same
  fixture.  See [docs/design/04-sync-signal.md](design/04-sync-signal.md).
- **Target channel rate** -- raised from 62.5 kHz (/32 decimation) to ~250 kHz
  (/8), so per-sample resolution is 4 µs instead of 16 µs.  Snippet size
  raised from 1280 samples (20 ms) to 5120 samples (20.5 ms).
- **Carrier timing** -- replaced envelope-xcorr / derivative-peak
  `sample_index` with **server-side argmin(d2) knee finding** (Savgol second
  derivative of the power envelope, default 360 µs window).  Nodes emit a
  larger IQ snippet + `transition_start` / `transition_end` zone bounds; the
  server does the Savgol work.  Real-corpus median |err| is ~59 µs on
  onset pairs.
- **Auto-tracked carrier thresholds** -- onset/offset now track the
  noise-floor EMA continuously (`onset = floor + 12 dB`, `offset = floor + 6 dB`).
  Static `onset_db`/`offset_db` only apply during noise-floor warmup.
- **SyncCalibrator** -- new server-side per-pair grid calibration that
  removes the inter-node pilot-phase grid offset before pilot-period
  disambiguation.

Stages 7-12 (delta computation, onset_time_ns, network, grouping,
disambiguation, final TDOA) are substantively unchanged.  The improvement
table at the end still captures the arc correctly, but the specific
numbers in Stage 6 and the Measured-baseline section reflect the
pre-2026-04 pipeline.

This document traces every stage of the TDOA measurement pipeline, identifies variance
contributions at each step, and summarises the current noise floor and improvement paths.

---

## Pipeline overview

```
RF signal arrives at antenna
        v
RSPduo ADC (2 MSPS, dual-channel interleaved)
        v
buf_wall_ns hardware timestamp (TCXO anchor)
        v
Sync-domain decimation /8 -> 256 kHz
        v
FM demodulation + 19 kHz pilot phase lock (sub-sample)
        v
RDS bit boundaries derived from pilot phase
        v
Target-domain decimation /8 -> ~250 kHz
        v
Carrier detection (auto-tracked onset/offset from noise floor;
                   IQ snippet + transition bounds emitted)
        v
sync_delta_ns = (target_sample - sync_sample) / rate
        v
onset_time_ns = buf_wall_ns + within-buffer offset
        v
HTTP POST to server
        v
Server-side Savgol d2 knee finding + per-pair SyncCalibrator
        v
Event grouping by T_sync
        v
Bit-period disambiguation (n = round((raw_ns + path_correction) / 842 us))
        v
TDOA = sync_delta_A - sync_delta_B - n * 842 us + path_correction
```

---

## Stage-by-stage analysis

### Stage 1: RF propagation -> antenna

FM pilot is broadcast from a regulated station. Propagation delay is deterministic
(c x distance), stable to better than 1 ns over any measurement interval.
`path_delay_correction_ns()` in `tdoa.py` removes the systematic offset.

**Variance: 0**

---

### Stage 2: ADC capture - RSPduo @ 2 MSPS

The RSPduo interleaves two tuner channels in the ADC stream: `[T1, T2, T1, T2, ...]`.
Channel 0 gets even samples, Channel 1 gets odd samples. The fixed inter-channel
timing offset is exactly **0.5 ADC samples = 250 ns** at 2 MSPS. This is a systematic
bias, not random noise. It is corrected by `pipeline_offset_ns` (configured per node).

Buffer duration: 65 536 / 2 000 000 = **32.8 ms**.
FIFO depth: 8 buffers = **524 ms** before overflow.

**Variance: 0** (integer sample domain)
**Systematic bias: 250 ns inter-channel offset - corrected by `pipeline_offset_ns`**

---

### Stage 3: `buf_wall_ns` - hardware timestamp (Part A)

Implemented in `SoapySDRPlay3/Streaming.cpp` (`tdoa-hw-timestamps` branch,
`dpkingston/SoapySDRPlay3`).

At the first stream callback, the C thread (no GIL, no Python scheduler) captures
`CLOCK_REALTIME` and stores it as `anchor_wall_ns` alongside `anchor_sample_num`.
All subsequent buffer timestamps are derived from the TCXO sample counter:

```
timeNs = anchor_wall_ns + (extended_first - anchor_sample_num) x 1e9 / outputSampleRate
```

| Source | Magnitude | Character |
|--------|-----------|-----------|
| NTP quality at anchor time | +/-50-200 usec | One-time systematic per process run |
| TCXO drift (5 ppm) | 0.15 usec/30 s, 9 usec/30 min | Accumulates monotonically since anchor |
| Two-node onset_diff error | ~100-400 usec combined | What matters for disambiguation |

**For disambiguation:** requires `|error| < 3.5 ms`. Hardware timestamps deliver
~100-400 usec - correct disambiguation essentially 100% of the time.

**For TDOA precision:** `onset_time_ns` is used only to determine `n` (which pilot
period). It does not enter the TDOA formula directly. **Impact: 0** when disambiguation
is correct.

**Without Part A (fallback `time.time_ns()` in Python):** +/-50-500 usec RMS per buffer,
potentially seconds of error during buffer backlog. This caused occasional wrong-`n`
selections and +/-7 ms TDOA catastrophes - the dominant source of the pre-Part A ~230 usec
observed std in the co-located pair test.

---

### Stage 4: Sync-domain decimation (/8 -> 256 kHz)

Polyphase decimation filter. Integer sample domain. No jitter introduced.

**Variance: 0**

---

### Stage 5: FM demodulation + 19 kHz pilot detection

(Post-2026-04: the pilot detection below is still used; however bit
boundaries are now derived from the unwrapped pilot phase at pilot/16 =
1187.5 Hz rather than from a separate Mueller-Muller timing loop on the
57 kHz RDS subcarrier.  `FMPilotSyncDetector` has been superseded by
`RDSSyncDetector`.)

`RDSSyncDetector` (`pipeline/rds_sync_detector.py`) processes **10 ms
windows** (2 560 samples at 256 kHz):

1. Narrow bandpass filter (+/-100 Hz around 19 kHz) isolates the pilot tone
2. Cross-correlate with complex exponential template:
   `corr = sum(audio x conj(template))` - result is a complex number
3. **Sub-sample timing** via `np.angle(corr)`: the correlation angle gives the pilot
   phase at the window start, from which the nearest 19 kHz zero-crossing offset is
   solved analytically (`sync_detector.py:322-340`). This locates the sync event to
   sub-sample precision - far better than window-centre resolution.
4. `CrystalCalibrator` tracks the pilot phase across consecutive windows and maintains
   a rolling-median correction factor over the last 100 windows (~1 s).

**Achievable sync_sample precision:**

| FM pilot SNR | Phase uncertainty | Timing sigma |
|---|---|---|
| 20 dB (good) | ~0.01 rad | ~84 ns |
| 10 dB (marginal) | ~0.10 rad | ~840 ns |

The `corr_peak` quality gate (default `min_corr_peak = 0.1`) rejects pilots below
usable SNR.

**Crystal calibration residual:** <1 ppm after convergence -> for a 7 ms sync_delta
interval, timing residual < 7 ns.

**Variance of `sync_sample`: ~0.1-1 usec RMS - not the bottleneck.**

---

### Stage 6: Carrier detection - onset and offset thresholds

(Originally the **dominant noise source**.  The post-2026-04 pipeline
addresses this at two layers: 1) the target rate was raised from 62.5 kHz
to ~250 kHz, shrinking the per-window quantum from 1 ms to ~1 ms at the
new rate but with 4x finer snippet resolution for the server-side knee
finder; 2) `sample_index` is no longer used for per-event timing -- the
server finds the PA ramp-to-plateau knee via `argmin` of the Savgol
second derivative of the power envelope inside a reported transition
zone.  The text below describes the original state machine; the
quantisation it discusses still applies to the coarse sync_delta but is
no longer the TDOA bottleneck.)

#### Detection architecture

`CarrierDetector` (`pipeline/carrier_detect.py`) operates on the target channel at
**~250 kHz** (/8 from 2 MSPS, post-2026-04; was /32 -> 62.5 kHz previously),
using **256-sample averaging windows** (= 1 ms per window).

A **hysteresis state machine** with two thresholds prevents false triggers:

```
IDLE --(power >= onset_threshold_db for min_hold_windows)---> ACTIVE  (CarrierOnset)
ACTIVE --(power <= offset_threshold_db for min_release_windows)---> IDLE  (CarrierOffset)
```

- `onset_threshold_db` (default -30 dBFS): the power level that must be sustained to
  declare carrier present. Set above the noise floor with enough headroom to reject
  interference. Raising it increases confidence but adds detection latency.
- `offset_threshold_db` (default -40 dBFS): the lower level required to declare carrier
  absent. The gap between the two thresholds (hysteresis = 10 dB by default) prevents
  chattering during power fades on active carriers.
- `min_hold_windows` (default 1): number of consecutive above-threshold windows before
  onset fires. At 1 ms/window, `min_hold_windows = 4` adds 4 ms debounce latency but
  suppresses transient noise spikes.
- `min_release_windows` (default 1, recommended 4-8): same for offset direction.
  Real-world LMR signals have brief power dips that should not trigger false offsets.

#### Sample index assignment

The `sample_index` reported by the detector is:

```python
window_sample = start_sample + i * self._window + self._window // 2
```

This is the **centre of the first window** that satisfied the threshold (and debounce).
It is quantised to window-centre steps.

**At 64 kHz with 64-sample windows:** one step = 64/64 000 = **1 ms**.

The pipeline converts from target-domain (64 kHz) -> raw ADC (x32) -> sync-domain (/8)
before passing to `DeltaComputer`, but this conversion preserves the 1 ms granularity.
In sync-domain units, 1 window = 256 samples at 256 kHz = **1 ms**.

#### Quantisation impact on TDOA

For a **co-located pair** (both nodes hear the same carrier at essentially the same
sync_delta), the detection windows on both nodes correspond to the same signal feature.
If both nodes fire in the same window (which they often do for fast, clean keying), the
quantisation error cancels in the difference:

```
TDOA = sync_delta_A - sync_delta_B ~ 0 (correct)
```

If the signal onset is gradual, or if minor SNR differences shift one node by one window
relative to the other, the result differs by +/-1 ms.

**Effective TDOA noise from carrier detection:**
- Fast, clean onset: ~0-100 usec (correlated quantisation)
- Gradual or borderline onset: +/-1 ms (one-window shift between nodes)
- Mean in practice: likely 100-500 usec for typical LMR

#### Derivative-based snippet alignment (offset events)

For **carrier offset**, the `_encode_offset_snippet()` method (lines 584-646) uses a
fundamentally different approach for locating the PA shutoff within the IQ snippet --
**not** for the `sample_index` timing value, but for aligning the xcorr snippet:

```python
# Smoothed power envelope and its derivative
smooth = 16
power  = |IQ|^2
kernel = np.ones(smooth) / smooth
envelope = np.convolve(power, kernel, mode='same')
deriv  = np.diff(envelope)

# Peak negative derivative = fastest power drop = PA shutoff moment
cut_idx = int(np.argmin(deriv))
```

The snippet is then centred on `cut_idx` so that both nodes' snippets have the PA shutoff
at the same relative position (3/4 from start), enabling the xcorr correlator to lock
onto the true shutoff regardless of when the threshold-crossing state machine fired.

**As of 2026-03-19:** `_encode_offset_snippet()` now returns `(bytes, cut_idx)` and
all three `CarrierOffset` emission sites set:

```python
sample_index = pre_snap_start + cut_idx
```

where `pre_snap_start` is the absolute stream sample of `iq_cat[0]`.  The derivative
peak is now used for **both** xcorr snippet alignment **and** the `sample_index` value
that feeds `onset_time_ns`.  For an abrupt PA cutoff, this places `sample_index` within
one smoothing window (+/-125 usec at 64 kHz / 16-sample kernel) of the true PA shutoff --
approximately **8x better** than the previous threshold-crossing window centre
(+/-500 usec).

#### Why offset events are generally more precise than onset

- **PA shutoff** (offset): the power amplifier bias is cut electronically - a nearly
  instantaneous event (sub-usec). The derivative of the received power envelope has a
  sharp, clean minimum that is unambiguous.
- **Carrier onset**: involves PLL lock acquisition, PA ramp-up, and CTCSS tone
  establishment. These are all physical settling processes that take 1-20 ms. The
  power rise is gradual and the "true" onset moment is not physically well-defined.
- For TDOA purposes, the offset event is therefore the more reliable measurement type,
  and results from offset-triggered events should in principle show lower noise.

---

### Stage 7: `sync_delta_ns` computation

```python
sync_delta_ns = (target_sample - sync_sample) x 1e9 / corrected_rate
```

Both indices are in the same 256 kHz crystal-clocked sample domain - no NTP, no GIL,
no network is involved. This is a **pure sample-count measurement**.

| Source | Magnitude | Notes |
|--------|-----------|-------|
| Carrier detection quantisation | 0-1 ms (correlated for co-located nodes) | **Dominant** |
| Pilot sync timing | ~0.1-1 usec | Negligible |
| Crystal calibration residual | <7 ns for 7 ms interval | Negligible |

**Noise floor at this stage: 0-1 ms depending on onset quality.**

Maximum sync_delta value is one pilot period ~ 7 ms (carrier must be within
`max_sync_age_samples` of a sync event, default 30 ms / 3 pilot periods).

---

### Stage 8: `onset_time_ns`

```python
raw_event_sample = m.target_sample x _sync_dec_factor          # -> raw ADC domain
onset_offset_ns  = raw_event_sample x 1e9 / sample_rate_hz     # within-buffer offset
onset_ns         = buf_wall_ns + onset_offset_ns - calibration_offset_ns
```

The within-buffer offset converts a sample index to nanoseconds at 2 MSPS: resolution
is 0.5 ns - effectively zero error. `calibration_offset_ns` is a single node-specific
constant, not a per-event source of jitter.

`onset_time_ns` is **not used in the TDOA formula**. It is shipped in the event and
used only for pilot disambiguation at the server.

---

### Stage 9: Network transmission

HTTP POST from node to server. Delivery buffer waits 10 s (`delivery_buffer_s = 10.0`
in `EventPairer`). Network latency does not affect `sync_delta_ns` or `onset_time_ns`
values (both computed at the node before transmission).

---

### Stage 10: Server grouping by `T_sync`

```python
T_sync = onset_time_ns - sync_delta_ns - dist(sync_tx, node) / c x 1e9
```

Events from different nodes are grouped if their `T_sync` values agree within
`+/-correlation_window_s / 2` (default +/-100 ms). `T_sync` estimates the absolute
wall-clock time of the FM pilot sync event at the transmitter - it should be the same
for all nodes that heard the same sync event.

With hardware timestamps, `T_sync` accuracy is ~100-400 usec per node - well within the
+/-100 ms window. The window could safely be tightened to +/-5 ms with hardware timestamps
on all nodes, which would reduce the risk of false-grouping events from different rapid
keying cycles.

---

### Stage 11: Pilot disambiguation

The FM stereo pilot sync window is 7 ms (`T_sync`). `sync_delta` is measured as the
offset of the carrier edge within the most recent 7 ms window, so two nodes that locked
to *different* pilot windows produce:

```
raw_ns = sync_delta_A - sync_delta_B = true_TDOA + n x T_sync
```

for some unknown integer `n`.  The goal is to determine `n` and remove the offset.

#### Geometric disambiguation (current implementation)

**Key insight:** within a 100 km operational radius, the maximum physical carrier TDOA
between any two nodes is bounded by geometry:

```
|true_TDOA| <= dist(A, B) / c <= 100 km / (3x10^8 m/s) ~ 333 usec
```

After applying the known sync-tx path correction (which itself is <= 333 usec for 100 km
sync-tx range), the corrected value lies within:

```
|raw_ns + correction_ns| = |true_TDOA + n x T_sync|
```

Since `|true_TDOA| <= 333 usec << T_sync/2 = 3500 usec`, rounding to the nearest multiple
of `T_sync` uniquely identifies `n` with zero ambiguity:

```python
correction_ns = path_delay_correction_ns(sync_tx, node_A, node_B)
n = round((raw_ns + correction_ns) / T_sync)   # always correct within 100 km
raw_ns -= n x T_sync
tdoa_ns = raw_ns + correction_ns
```

There are at most **three candidate values of n** (-1, 0, +1) and the geometry selects
the unique one that leaves `|true_TDOA| < T_sync/2`.  **No wall-clock timestamps
(NTP, hardware, or otherwise) are required** - node and sync-tx positions are sufficient.

The 100 km constraint is a design parameter.  At 200 km the bound becomes 667 usec,
still well within T_sync/2 = 3500 usec - the method is valid to at least 1000 km
(3333 usec < 3500 usec).  The bound fails only beyond ~1050 km, far outside any realistic
single-network deployment.

| Source | Disambiguation result |
|---|---|
| Geometric (node positions + sync-tx, any NTP quality) | Correct for all deployments <= 1000 km |
| Old NTP/onset_time_ns method | Correct only when inter-node NTP error < 3.5 ms |

The real 68-pair fixture contains examples of all three `n` values (n=-1: 6 pairs,
n=0: majority, n=+1: 7 pairs) - all correctly resolved by the geometric method.
`TestRealDataPilotDisambiguation` in `tests/unit/test_real_data.py` covers each case.

---

### Stage 12: Final TDOA

```python
tdoa_ns = (sync_delta_A - sync_delta_B) - n x 7_000_000 + path_correction_ns
```

Full error budget:

| Source | Magnitude | Status |
|--------|-----------|--------|
| Carrier detection quantisation (~1 ms/window) | 0-1 ms, correlated | **Current noise floor (sync_delta path)** |
| Wrong pilot disambiguation (wrong `n`) | +/-7 ms | **Fixed - geometric method, no NTP required** |
| RSPduo inter-channel 250 ns offset | 250 ns systematic | Corrected by `pipeline_offset_ns` |
| Pilot sync timing | ~0.1-1 usec | Negligible |
| Path correction (node/sync-tx location) | <100 ns (GPS locations) | Negligible |

---

## Current state summary

| What was fixed | Method | Expected improvement |
|---|---|---|
| Buffer backlog -> stale `onset_time_ns` -> wrong `n` | Part B: backlog drain logic in `rspduo.py` | Eliminates +/-7 ms catastrophic outliers |
| Per-buffer NTP jitter on `onset_time_ns` | Part A: TCXO hardware timestamps in SoapySDRPlay3 | onset_diff error 50-200 usec -> correct disambiguation |

**Remaining noise floor: 0-1 ms per onset event, depending on signal onset sharpness.**
For fast keying events both nodes quantise to the same window -> TDOA residual near 0.
For gradual onsets -> +/-1 ms jitter.

Offset events should be systematically better than onset events because the PA shutoff
is sharper. Mixing event types at the server would produce garbage - pairing enforces
onset-with-onset and offset-with-offset.

---

## Improvement paths

### 1. ~~Use derivative peak for `sample_index` on offset events~~ [x] Done

**Implemented 2026-03-19** (`carrier_detect.py`, commit `abcdf29`).

`_encode_offset_snippet()` now returns `(bytes, cut_idx)`.  All three `CarrierOffset`
emission sites (deferred path, opposite-fired immediate, main immediate) compute
`sample_index = pre_snap_start + cut_idx`, where `pre_snap_start` is the absolute
stream sample of `iq_cat[0]`.

Measured impact: **xcorr TDOA numbers unchanged** (xcorr uses IQ snippet bytes, not
`sample_index`).  The improvement shows in `onset_time_ns` precision for disambiguation:
offset `sample_index` is now within +/-125 usec of the true PA shutoff (vs. +/-500 usec
threshold window), an ~8x improvement.  This will matter if disambiguation via
`onset_time_ns` is re-enabled (requires PPS-disciplined NTP for inter-node accuracy).

### 2. Cross-correlation TDOA (already implemented)

The xcorr path in `tdoa.py:cross_correlate_snippets()` cross-correlates the IQ power
envelopes from two nodes with parabolic sub-sample interpolation. This bypasses the
threshold-crossing sample index entirely - the correlator finds the lag that maximises
the envelope overlap, which is a high-SNR measurement.

- Precision: typically +/-5-50 usec depending on bandwidth, SNR, and snippet length
- Limitation: requires IQ snippets to be stored, transmitted, and aligned
- Already returns `(lag_ns, snr)` - just needs to be plumbed into the TDOA report

**Estimated improvement: +/-500 usec -> +/-5-50 usec.**
**Complexity: low - xcorr already works; needs server integration with sync_delta result.**

### 3. ~~Smaller detection windows~~ - Deferred

Halving `window_samples` from 64 to 32 (at 64 kHz) reduces quantisation from +/-500 usec
to +/-250 usec, at the cost of increased susceptibility to brief noise spikes.
Requires increasing `min_hold_windows` to compensate.
`window_samples = 16` at 64 kHz -> +/-125 usec, but needs `min_hold_windows >= 4` for
debounce - same net latency, but finer resolution.

**Estimated improvement: +/-500 usec -> +/-62-125 usec (with compensating debounce).**
**Complexity: low - config parameter changes only, but requires fresh captures to evaluate.**

**Why deferred:** this improvement only affects `sync_delta_ns` quantisation, which is
relevant only for the `sync_delta` fallback path (used when xcorr SNR is too low).
Xcorr already achieves sub-usec offset precision and handles 27/34 offset pairs in
the current fixture.  The 7 fallback pairs have low SNR due to weak signal, not window
size - smaller windows would not fix them.

Testing requires re-capturing raw SDR streams with a different `window_samples` config;
the existing fixture's `sync_delta_ns` values are baked at capture time and cannot be
re-derived from the 20 ms IQ snippets alone.  **Revisit if the environment changes
(e.g., SNR degrades such that xcorr fallback becomes frequent).**

### 4. Run carrier detection at higher sample rate

At 256 kHz (sync-domain rate, /8 instead of /32), with 64-sample windows:
1 window = 64/256 000 = **250 usec** -> halved quantisation.
With 16-sample windows: **62.5 usec**. Trade-off: 4x CPU.

### 5. ~~Tight `correlation_window_s` for grouped nodes~~ - Not planned

`correlation_window_s` controls how closely two nodes' `T_sync` estimates must agree
to be grouped as the same transmission.  The default was reduced from 0.5 s to **0.2 s**
(2026-03-19) based on fixture analysis: the observed maximum inter-node T_sync spread
across 68 real pairs is ~39 ms, so 0.2 s (+/-100 ms half-window) provides a 2.5x safety
margin for typical internet NTP.

Further tightening to ~0.005 s (+/-2.5 ms) would only be practical with PPS-disciplined
NTP (GPS 1PPS), which reduces absolute NTP error to <10 usec.  No nodes in the currently
planned deployment have PPS hardware, and the grouping accuracy at 0.2 s is already
more than adequate - false grouping of successive transmissions requires two key-ups
within the same 200 ms window on the same channel, which is rare in practice.

**Conclusion: not a useful improvement for any currently planned environment.**

---

## Measured baseline (2026-03-19)

Fixture: `tests/fixtures/real_event_pairs.json` - 68 co-located pairs
(34 onset, 34 offset) captured from node-mapleleaf / node-greenlake.
Snippet: 1280 samples @ 62.5 kHz = 20.5 ms.
`onset_time_ns` in this fixture is **pre-hardware-timestamp** (time.time_ns()
at processing time); disambiguation was therefore disabled for these measurements.

### sync_delta difference (no disambiguation)

| Event type | Mean (usec) | Std (usec) | Min (usec) | Max (usec) | Notes |
|------------|-----------|----------|----------|----------|-------|
| Onset | +188 | 1989 | -3394 | +3442 | Uniform across +/-3.5 ms pilot period |
| Offset | +719 | 2211 | -3238 | +3270 | Uniform across +/-3.5 ms pilot period |

Std ~ 2020 usec matches the theoretical std of a uniform distribution on [-3.5, 3.5] ms
(7 ms / sqrt12 ~ 2020 usec). The nodes almost always used different pilot periods as their
sync reference. Without disambiguation, the raw sync_delta difference is no better than
random within the pilot period - **useless for TDOA**.

The expanded 68-pair fixture (2026-03-19) was captured WITH hardware timestamps deployed
(SoapySDRPlay3 TCXO anchor, Part A + Part B).  However, per-buffer jitter and absolute
NTP offset are distinct problems:

- **Per-buffer NTP jitter**: eliminated by TCXO anchor - each buffer's `buf_wall_ns`
  is derived from `firstSampleNum`, so the within-session jitter is ~ppm-level.
- **Absolute inter-node clock offset**: still NTP-limited.  The two nodes'
  `CLOCK_REALTIME` values at driver start diverge by 10-40 ms over the capture session
  (chrony without PPS discipline, std ~ 14 ms across 68 pairs).

Disambiguation requires inter-node `onset_time_ns` accuracy << T_sync/2 = 3.5 ms.
With 14 ms std, disambiguation is unreliable on this fixture.  GPS-disciplined or
PPS-disciplined NTP (sub-ms accuracy) would make it reliable.

Since xcorr is now the primary TDOA method and does not use `onset_time_ns`, this
limitation is moot for the offset measurement path (std = 0.93 usec confirmed above).

### xcorr lag (cross_correlate_snippets on co-located pairs)

| Event type | n pairs (good SNR) | Mean (usec) | Std (usec) | Min (usec) | Max (usec) |
|------------|-------------------|-----------|----------|----------|----------|
| Onset | 34 | -4.3 | 299 | -854 | +603 |
| Offset | 27 | -0.5 | **0.9** | -3.0 | +1.3 |

- **Offset xcorr: std = 0.9 usec** - sub-usec TDOA precision confirmed on real hardware data.
  The PA shutoff is a hard electronic edge; the xcorr finds it to within a few usec.
- Onset xcorr: std = 299 usec - wide scatter due to gradual carrier ramp-up.
  Onset is inherently imprecise; offset events are the reliable TDOA measurement.
- 7/34 offset pairs excluded (SNR 1.29-1.31): false offset detections from node-greenlake
  where the carrier was still on in node B's snippet (brief power dip triggered
  the state machine without a true PA cutoff).

### xcorr SNR distribution

| Event type | Min | Median | Max | Below 1.5 |
|------------|-----|--------|-----|-----------|
| Onset | >1.5 | ~1.7 | ~2.x | 0/34 |
| Offset | 1.29 | 1.67 | 1.76 | 7/34 (false detections) |

### Improvement potential

| Method | Current | Target | Gain |
|--------|---------|--------|------|
| xcorr plumbing (offset) | **Done - 0.9 usec** | std ~ 0.9 usec | ~2400x vs. sync_delta |
| Derivative peak -> sample_index | **Done - +/-125 usec (onset_time_ns)** | ~200-500 ns | ~8x vs. window centre |
| Geometric pilot disambiguation | **Done** | removes pilot period ambiguity | no NTP required |
| Lower correlation_window_s default | **Done - 0.2 s** (was 0.5 s) | 2.5x safety margin over observed max | operational hygiene |
| Smaller windows (/2) | ~1 ms | ~500 usec | 2x |
| Tight correlation_window_s (GPS PPS only) | - | ~0.005 s | Not planned |

---

## Post-improvement results (2026-03-19) - xcorr as primary TDOA method

`compute_tdoa_s()` now uses `cross_correlate_snippets()` as its primary method when both
events carry `iq_snippet_b64`. If xcorr SNR >= 1.5 the lag is returned directly (no
path-delay correction; xcorr measures physical arrival-time difference directly). The
sync_delta subtraction path is retained as fallback.

Measured on the same fixture (68 co-located pairs), nodes node-mapleleaf / node-greenlake.

### compute_tdoa_s results

| Event type | n pairs (xcorr) | n pairs (sync_delta fallback) | xcorr mean (usec) | xcorr std (usec) | xcorr range (usec) |
|------------|----------------|-------------------------------|-----------------|----------------|------------------|
| Onset | 34 | 0 | -4.3 | 299 | [-854, +603] |
| Offset | 27 | 7 | **-0.6** | **0.93** | [-3.0, +1.3] |

The 7 offset pairs that fall back to sync_delta are the false-detection pairs (SNR 1.29-1.31);
their sync_delta fallback gives mean=-1511 usec, std=2339 usec (random within +/-3.5 ms pilot period).

### Before-vs-after comparison

| Event type | Method | Std (usec) | Improvement |
|------------|--------|----------|-------------|
| Onset | sync_delta (before) | 1989 | - |
| Onset | xcorr (after) | 299 | 6.6x |
| Offset | sync_delta (before) | 2211 | - |
| Offset | xcorr - good pairs (after) | **0.93** | **2376x** |

Offset xcorr precision is now **sub-usec** on real co-located hardware data.  For distributed
nodes, this translates directly to sub-usec TDOA precision - approximately +/-280 m position
uncertainty per pair (c x 0.93 usec ~ 279 m).

Onset xcorr (std=299 usec) is better than sync_delta but still wide because gradual carrier
ramp-up limits xcorr discrimination.  Offset events are the reliable measurement.

---

## Post-improvement results (2026-03-19) - derivative peak sample_index refinement

`CarrierOffset.sample_index` now tracks the PA shutoff via peak negative derivative
rather than the threshold-crossing window centre.  Measured on the same 68-pair fixture.

### TDOA impact

**Xcorr TDOA numbers unchanged** - xcorr uses the IQ snippet bytes (centred by the
same `cut_idx` logic, unchanged since xcorr was implemented).  `sample_index` is not
used by the xcorr path.

| Event type | n good pairs | xcorr std (usec) | vs. xcorr baseline |
|------------|-------------|----------------|-------------------|
| Onset | 34 | 299 | identical |
| Offset | 27 | **0.93** | identical |

### onset_time_ns precision improvement (sample_index)

| | Before | After |
|---|---|---|
| Offset `sample_index` error vs. true PA shutoff | ~+/-500 usec (threshold window) | ~+/-125 usec (16-sample derivative kernel) |
| Improvement factor | - | **~8x** |

This matters for pilot disambiguation accuracy when `onset_time_ns` is used.  With
the current NTP-only inter-node accuracy (std ~ 14 ms), disambiguation remains
unreliable regardless - the benefit is latent until PPS-disciplined NTP is deployed.

### Summary

Two improvements are now done and measurably reduce the noise floor:

| Improvement | TDOA std (offset, co-located) | TDOA std (onset, co-located) |
|---|---|---|
| Baseline (sync_delta, no disambiguation) | 2211 usec | 1989 usec |
| + xcorr as primary TDOA method | **0.93 usec** | 299 usec |
| + derivative peak sample_index | **0.93 usec** (unchanged) | 299 usec (unchanged) |

The derivative peak improvement prepares `onset_time_ns` for future disambiguation use;
it does not further reduce xcorr TDOA noise, which is already at the sub-usec floor set
by the PA shutoff physics and the 62.5 kHz sample rate.

---

## Measurement validation

The co-located pair test (`scripts/colocated_pair_test.py`) is the primary tool for
measuring the effective TDOA noise floor. Both nodes at the same location should produce
TDOA ~ 0 (after path correction). The std of the distribution is the combined noise.

- **Pre-Part A observed std:** ~230 usec (dominated by wrong-`n` disambiguation outliers)
- **Expected post-Part A std:** limited by carrier detection quantisation, expected
  0-500 usec depending on signal quality - requires clean data (no backlogs) to measure
- **Target:** +/-50 usec or better (achievable with derivative-based timing or xcorr)

---

## LMR signal features for timing - research notes

_What else in the LMR FM signal could serve as a high-precision timing reference?_

The existing system uses the **19 kHz FM broadcast stereo pilot** as its reference clock
(detected on the sync channel with sub-sample phase precision, ~100 ns). The question is
whether there are equivalent features _within the LMR target signal itself_ that could
improve onset/offset timing beyond the current 1 ms window limit.

---

### Sub-audible tones: CTCSS (67-254.1 Hz)

CTCSS tones start at PTT press and are transmitted continuously for the duration.
The tone is phase-coherent during a transmission (EIA/TIA-603 defines a 180 deg "reverse
tone burst" at PTT release that presupposes phase coherence). In principle, the tone
phase could act as a sub-sample timing reference - at 254.1 Hz, one period = 3.9 ms,
and if phase can be measured to 1-2 deg, that would give +/-11-22 usec timing within the period.

**However, CTCSS is not usable for carrier onset timing in practice:**

1. **Unknown frequency** - the target's CTCSS code is not known in advance for general
   LMR monitoring. A frequency scan would be needed.
2. **Random start phase** - many transmitters gate a free-running oscillator on at PTT.
   The phase at keyup is arbitrary and different each time.
3. **Settling delay** - the tone passes through the sub-audible filter chain (3rd/4th-order
   LPF below 300 Hz) and takes several cycles to reach full amplitude. At 67 Hz that is
   ~45 ms; at 254.1 Hz ~12 ms. The onset region is unusable.
4. **Frequency vs. RF timing** - CTCSS is an audio-domain signal; it measures the FM
   deviation, not the RF carrier itself.

CTCSS is useful for _target identification_ (confirming you are measuring the intended
radio) but is **not a viable timing reference** for carrier onset.

---

### DCS (Digital Coded Squelch, 134.4 baud FSK)

DCS starts at PTT simultaneously with CTCSS. The 23-bit Golay codeword is repeated
continuously at 134.4 bps, giving one bit period = 7.44 ms. Even ignoring the arbitrary
start phase and settling problems, 7.44 ms/bit is far coarser than the desired timing
precision.

**Verdict: not useful for TDOA timing.**

---

### Carrier phase at onset

An FM transmitter uses a PLL synthesizer with a VCO that runs continuously (gated PA,
not gated oscillator). At PTT, the PA bias is switched on into an already-running,
continuously-deviating carrier. The carrier phase at the moment of onset is:
- completely arbitrary (VCO phase has been running free),
- immediately perturbed by the PTT switching transient and CTCSS onset modulation.

At 155 MHz the carrier period is 6.5 ns - in principle sub-nanosecond precision --
but the phase is random at every keyup.

**Verdict: carrier phase at onset provides no timing information.**

---

### PA shutoff at carrier offset - the best edge in the signal

The PA _cutoff_ (PTT release) is a fundamentally different event from onset:

- The PA bias is cut by a digital gate signal - a hard electronic switch.
- No PLL settling, no modulator startup, no audio ramp-up.
- Production LMR PA fall times (VHF/UHF LDMOS/BJT): **1-10 usec** for the RF envelope
  to drop 20-30 dB, limited by bias network RC discharge.
- The envelope derivative has a single sharp minimum - the PA cutoff moment - that is
  repeatable to +/-200-500 ns across transmissions from the same radio.
- Research on GaN PAs shows sub-200 ns fall times are achievable; production LMR is
  typically 1-5 usec but still a hard, consistent edge.

This is **10-100x sharper than onset**, and is already partially exploited:
`_encode_offset_snippet()` uses `np.argmin(deriv)` to find the PA shutoff for xcorr
snippet alignment. The sample_index reported for sync_delta is still the threshold-
crossing window centre; replacing it with the derivative-peak position (Improvement
path #1 above) would directly realise this precision.

With 2-10 MS/s sampling and sub-sample interpolation, offset timing precision of
**+/-100-500 ns** is achievable. At the current 64 kHz / 1 ms window: +/-500 usec.

---

### Digital LMR modulations (P25, DMR, NXDN, TETRA)

Digital LMR signals have known, fixed sync words (preambles) that can be detected via
matched filter (cross-correlation against the known pattern). This is fundamentally more
precise than power-envelope threshold detection because:

- The sync word spans many symbols (5-10 ms) with deterministic structure.
- The cross-correlation peak width ~ 1/bandwidth.
- Sub-sample centroiding of the correlation peak gives timing far below 1 sample.

| Protocol | Symbol rate | Channel BW | Sync word | Est. timing precision |
|----------|-------------|-----------|-----------|----------------------|
| P25 Ph1 (C4FM) | 4800 sym/s | 12.5 kHz | 48 sym / 10 ms | ~10-20 usec |
| P25 Ph2 (H-DQPSK) | 6000 sym/s | 12.5 kHz | 24 sym / 4 ms | ~10-20 usec |
| DMR (4-FSK TDMA) | 4800 sym/s | 12.5 kHz | 24 sym / 5 ms | ~10-20 usec |
| NXDN 6.25 kHz | 2400 sym/s | 6.25 kHz | 18 sym / 7.5 ms | ~20-40 usec |
| TETRA (Pi/4-DQPSK) | 18 000 sym/s | 25 kHz | - | ~5-10 usec |

For P25 and DMR, which are common in North American public safety LMR, matched-filter
detection of the frame sync word would provide **~10-20 usec** onset timing - 50-100x
better than the current 1 ms window, and achievable at the existing 64 kHz sample rate
(the 12.5 kHz symbol waveform is well-captured at 64 kHz).

Note that DMR is TDMA (30 ms slots), so individual transmitters have slot-quantised
onset times. The sync word within the first burst is still the best timing anchor.

The existing xcorr IQ snippet infrastructure could be extended to include matched-filter
detection of P25/DMR sync words as an additional TDOA refinement path.

---

### NFM bandwidth vs. WBFM: the fundamental constraint

The CRFS rule of thumb: timing accuracy ~ c / (4 x bandwidth). For 12.5 kHz NFM:
timing resolution floor ~ 6 km - meaningless without structure. For 200 kHz WBFM:
~ 375 m. For the 19 kHz pilot tone itself (near-CW): the pilot period (52.6 usec) is the
ambiguity, and phase measurement precision of 1-2 deg at 20 dB SNR gives ~150-300 ns.

NFM has no equivalent continuous subcarrier, so we rely entirely on the transient
features (PA shutoff, digital sync words) for precision. The 15x narrower bandwidth
means the envelope transition is the only timing information available, and it requires
high sample rates (>=2 MS/s) to resolve sub-usec features.

---

### Recommended priority for timing improvement

| Method | Achievable precision | Works on | Complexity | Status |
|--------|---------------------|----------|------------|--------|
| Xcorr (already implemented) | ~1 usec (offset), ~300 usec (onset) | All LMR | Low | **Done** |
| Offset-event derivative peak -> `sample_index` | ~125 usec (onset_time_ns) | All LMR | Medium | **Done** |
| Geometric pilot disambiguation | removes +/-3.5 ms alias | All LMR | Medium | **Done** |
| Lower `correlation_window_s` default (0.2 s) | 2.5x grouping safety margin | All LMR | Trivial | **Done** |
| Smaller detection windows | ~250 usec (sync_delta fallback only) | All LMR | Low (config + recapture) | **Deferred** |
| P25 / DMR sync word matched filter | ~10-20 usec | Digital LMR | High (new detector) | Not started |
| Tight `correlation_window_s` (GPS PPS only) | grouping to +/-2.5 ms | PPS nodes only | Requires PPS hw | Not planned |
| CTCSS phase | Not viable | - | - | Ruled out |

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../LICENSE).
