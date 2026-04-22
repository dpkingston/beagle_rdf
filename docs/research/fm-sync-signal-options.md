# FM Broadcast Sync Signal Options for TDOA Timing

Research into alternatives to the 19 kHz stereo pilot for timing synchronization.

**Current baseline:** 19 kHz FM stereo pilot tone extracted every 7 ms
(133 pilot cycles), correlation peak ~0.7, achieving ~1-5 usec timing accuracy
on the sync_delta path.

**Motivation:** The pilot is a single-frequency sinusoid -- its correlation peak
is inherently wide (~52.6 usec per cycle). Any wideband signal would produce a
sharper correlation peak and better timing precision.

---

## Signals Present on a Typical Modern US FM Broadcast

### 1. 19 kHz Stereo Pilot (current)

- **Location:** 19.000 kHz in FM baseband
- **Bandwidth:** ~0 Hz (single tone)
- **Power:** Fixed at +/- 7.5 kHz deviation (10% of total)
- **Always present:** Yes, on all stereo FM stations
- **Timing rate:** 19,000 zero-crossings per second
- **Processing:** Low complexity (bandpass + correlation with complex exponential)

This is our current approach. The fundamental limitation is that a single tone
has zero bandwidth -- timing precision depends entirely on SNR and number of
cycles integrated. The Cramer-Rao bound for a pure tone at SNR=20 is ~0.4 usec
per cycle, which is roughly what we achieve.

### 2. RDS/RBDS (Radio Data System)

- **Location:** 57 kHz (3x pilot), +/- 2.4 kHz bandwidth
- **Modulation:** DSBSC-BPSK, 1187.5 bps (= 19000/16)
- **Power:** 2-4% of total modulation (6-10 dB weaker than pilot)
- **Always present:** Yes, on most commercial and NPR stations
- **Timing structure:** Bit transitions at 1187.5 Hz (every ~842 usec)

The 57 kHz carrier provides 3x the zero-crossing rate of the pilot, but the
subcarrier is suppressed (DSBSC) so there's no continuous carrier to track.
The bit pattern varies (station ID, program type, radiotext), so correlation
requires either short windows or data decoding.

**Verdict: Marginal.** The 3x frequency advantage is largely negated by the
6-10 dB lower SNR. The data-varying nature adds complexity. Probably <2x
improvement over the pilot at significant added complexity.

### 3. HD Radio (IBOC / NRSC-5) -- MOST PROMISING

- **Location:** OFDM digital sidebands at +/- 129 to +/- 198 kHz from center
- **Bandwidth:** ~69 kHz per sideband, ~138 kHz total (both sidebands)
- **Power:** -10 to -20 dBc (1-10% of analog power; many stations now at -10 dBc)
- **Always present:** Most major-market US stations; KUOW broadcasts HD1 + HD2
- **OFDM parameters:**
  - Subcarrier spacing: 363.373 Hz
  - Symbol duration: ~2.752 ms + ~364 usec cyclic prefix = ~2.9 ms total
  - Symbol rate: ~344 symbols/second
  - Active subcarriers: ~191 per sideband (hybrid mode)

**Key insight:** Bandwidth is the dominant factor in timing precision. The
Cramer-Rao lower bound on time-of-arrival estimation scales as 1/(BW * sqrt(SNR)).
HD Radio's 138 kHz combined bandwidth vs the pilot's effectively zero bandwidth
provides fundamentally superior timing resolution:

- Pilot correlation peak width: ~52.6 usec (one cycle)
- HD Radio correlation peak width: ~7 usec (1/138 kHz)
- **Improvement: 7-15x sharper correlation peaks**

Even at -10 dBc, the wide bandwidth more than compensates for the lower power.

**Capture compatibility:** HD Radio sidebands at +/- 129 to +/- 198 kHz are
well within our +/- 1 MHz capture bandwidth at 2 MHz sample rate. **No hardware
changes needed.**

**Processing approach:** No need to decode HD Radio. For timing:
1. Bandpass filter to isolate one or both digital sidebands
2. Cross-correlate the wideband sideband signal between nodes
3. The sharp correlation peak gives sub-microsecond timing precision

Alternatively, detect OFDM symbol boundaries via cyclic-prefix autocorrelation
(requires knowing CP length but not data content).

**Challenges:**
- Lower power than analog signal requires careful bandpass filtering
- OFDM signal is noise-like -- works well for cross-correlation between nodes
  but harder for single-node autocorrelation-based timing
- HD Radio is not on every FM station (but is on most major-market stations)
- Need to handle the frequency offset between upper and lower sidebands

### 4. L-R Stereo Subcarrier (38 kHz DSBSC)

- **Location:** 38 kHz (2x pilot), +/- 15 kHz bandwidth
- **Bandwidth:** ~30 kHz
- **Power:** Variable -- depends on stereo content (near zero during mono passages)
- **Always present:** Yes on stereo stations, but amplitude varies with programming

Cross-correlating the L-R subcarrier between nodes would give a correlation peak
width of ~33 usec (~1.6x better than pilot alone). However, the signal content
depends on programming -- during silent passages or mono-compatible content,
there is little or no L-R energy.

**Verdict: Unreliable standalone, but useful as supplement.** Could be combined
with pilot correlation to modestly improve peak sharpness when stereo content
is present.

### 5. Full Composite Baseband Correlation

- **Bandwidth:** 0-53 kHz (mono + pilot + L-R + RDS)
- **Correlation peak width:** ~17 usec (~3x better than pilot alone)

Instead of extracting just the pilot, correlate the entire demodulated FM
baseband between nodes. The pilot provides the dominant correlation peak, but
the L-R subcarrier and audio content contribute additional bandwidth that
narrows the peak.

**Verdict: Simple near-term improvement.** Requires minimal code changes --
correlate the full demodulated audio window instead of just the extracted pilot.
Roughly 2-3x improvement in peak sharpness. The audio content changes
continuously, but our existing 7 ms correlation windows handle this.

### 6. SCA Subcarriers (67 / 92 kHz)

- **Location:** 67 kHz and/or 92 kHz
- **Modulation:** Narrowband FM, +/- 7.5 kHz deviation
- **Historical use:** Background music (Muzak), reading services for the blind
- **Current status:** Rare and declining; most stations have replaced SCA
  allocation with HD Radio guard band

**Verdict: Not useful.** Unreliable presence, narrowband, no timing structure.

---

## Summary Comparison

| Signal | Bandwidth | Peak Width | Improvement | Reliability | Complexity |
|--------|-----------|------------|-------------|-------------|------------|
| 19 kHz Pilot (current) | ~0 Hz | ~52.6 usec | 1x (baseline) | Excellent | Low |
| RDS/RBDS | ~4.8 kHz | ~35 usec | ~1.5x | Good | Medium |
| **HD Radio sidebands** | **~138 kHz** | **~7 usec** | **7-15x** | **Good** | **Medium** |
| L-R stereo subcarrier | ~30 kHz | ~33 usec | ~1.6x | Variable | Low |
| Full composite baseband | ~53 kHz | ~17 usec | ~3x | Good | Low |
| SCA subcarriers | ~15 kHz | N/A | N/A | Poor | Low |

---

## Recommendations

### Primary: HD Radio sideband correlation

The clear winner for a future improvement. Implementation steps:

1. Bandpass filter existing IQ to isolate HD Radio sidebands (+129 to +198 kHz)
2. Cross-correlate the filtered wideband signal between nodes
3. Use the pilot for coarse timing / ambiguity resolution
4. Use HD Radio for fine timing (sub-microsecond precision)

No hardware changes. The signal is already in our 2 MHz capture. The main work
is bandpass filtering and adapting the correlator for wideband noise-like signals.

**Risk:** Station must broadcast HD Radio. KUOW does; verify any alternative
sync stations before depending on this.

### Secondary: Full composite baseband correlation

Simpler near-term improvement with minimal code changes. Correlate the full
demodulated FM baseband (0-53 kHz) instead of just the extracted pilot.
Roughly 2-3x improvement. Could be done as an intermediate step before the
HD Radio work.

### Not recommended

RDS (marginal gain, significant complexity) and SCA (unreliable, no timing
structure).

---

## Implementation Plans

### Option A: Full Composite Baseband Correlation (simpler, ~3x improvement)

**Concept:** Instead of extracting the 19 kHz pilot and correlating it in
isolation, correlate the full demodulated FM baseband (0-53 kHz) between nodes.
The pilot still dominates the correlation, but the L-R stereo subcarrier
(23-53 kHz) and mono audio (0-15 kHz) contribute additional bandwidth that
narrows the correlation peak.

**Effort:** Small. Mostly changes to `sync_detector.py` and `pipeline.py`.
No changes to the server, event model, or config schema.

#### Node-side changes

**`sync_detector.py` -- new `WidebandSyncDetector` class (or mode flag):**

The current `FMPilotSyncDetector` works on demodulated FM audio but immediately
bandpass-filters to isolate the 19 kHz pilot, then cross-correlates with a
complex exponential template. The new approach:

1. Skip the 19 kHz bandpass filter.
2. Use the full demodulated audio window (0 to Nyquist of the 256 kHz sync
   decimated rate = 0-128 kHz, though FM content is concentrated 0-53 kHz).
3. Cross-correlate the full audio window between consecutive windows (or between
   nodes, but that happens on the server). For the node-side sync event, we
   still need a periodic timing reference -- the pilot provides this.
4. **Hybrid approach (recommended):** Keep the pilot extraction for SyncEvent
   timing (it gives us the zero-crossing position and crystal calibration), but
   also include a short wideband audio snippet in the SyncEvent for server-side
   fine timing.

**Detailed changes:**

| File | Change |
|------|--------|
| `sync_detector.py` | Add optional `wideband_snippet` field to `SyncEvent` dataclass. When enabled, include the raw demodulated audio window (before BPF) alongside the existing pilot-based timing. |
| `pipeline.py` | Add `sync_wideband_snippets: bool = False` to `PipelineConfig`. Pass it through to `FMPilotSyncDetector`. |
| `config/schema.py` | Add `sync_wideband_snippets` to the sync_signal config section. |
| `events/model.py` | Add optional `sync_snippet_b64` field to `CarrierEvent` (the demodulated FM audio window around the matched sync event). |
| `delta.py` | When building the `TDOAMeasurement`, include the sync snippet from the matched `SyncEvent` if available. |
| `main.py` | Wire the new config field through to the pipeline. |

**Server-side changes:**

| File | Change |
|------|--------|
| `tdoa.py` | In `compute_tdoa_s()`, when both events carry `sync_snippet_b64`, cross-correlate the wideband sync snippets for a fine timing offset to apply on top of the coarse `sync_to_snippet_start_ns` difference. This is analogous to how carrier IQ snippet xcorr works, but applied to the sync path. |
| `api.py` | Accept the new field in event ingest (already permissive via `raw_json`). |

**Data volume impact:** Each sync snippet would be ~1792 samples at 256 kHz
(7 ms window) * 4 bytes (float32) = ~7 KB per event. This is significant --
roughly 5x larger than the current carrier IQ snippet (~1280 bytes). Could be
mitigated by decimating the sync snippet further before transmission (e.g.,
4x decimate to 64 kHz = ~1.7 KB), since the useful bandwidth is only 53 kHz.

**Alternative -- server-side only (no data volume increase):**

Instead of sending sync snippets to the server, each node could compute a
wideband sync correlation peak position locally and include it as a refined
`sync_sample_index` with sub-sample precision. This is a single float64 per
event instead of a 7 KB snippet. The server would use the refined sample
indices for sync_delta subtraction without needing to cross-correlate itself.

This is simpler but loses the ability to do server-side cross-correlation of
the sync signal between nodes (which could eliminate the sync_delta path
entirely -- see Option B below).

#### Testing

- Unit test: generate synthetic FM baseband with pilot + stereo + audio,
  verify that wideband correlation gives a sharper peak than pilot-only.
- Integration test: use existing `verify_sync.py` with a `--wideband` flag
  to compare timing jitter between pilot-only and wideband modes on real data.

#### Risks

- Audio content varies between stations and over time. Some content (talk
  radio, silence) has less wideband energy. The pilot still provides the
  baseline; wideband is an enhancement.
- Larger event payload increases network and database usage unless we use the
  refined-sample-index approach.

---

### Option B: HD Radio Sideband Correlation (larger effort, 7-15x improvement)

**Concept:** Bandpass filter the raw IQ (pre-demodulation) to isolate the HD
Radio OFDM sidebands at +/- 129-198 kHz, then cross-correlate between nodes
for precise timing. This operates on the raw IQ before FM demodulation, which
is a fundamentally different signal path from the current pilot extraction.

**Effort:** Medium-large. New processing stage in the pipeline, new sync event
type, server-side changes for a second correlation dimension.

#### Node-side changes

**New module: `hd_sync_detector.py`**

A completely new sync detector that operates on raw IQ (not demodulated audio):

1. **Bandpass filter** to isolate one or both HD Radio sidebands from the raw
   IQ stream. At 2.048 MHz sample rate:
   - Upper sideband: +129 to +198 kHz from center = passband at 129-198 kHz
     in the baseband IQ. This is a complex bandpass filter (frequency-shift
     the IQ by -163.5 kHz to center the sideband, then lowpass at 34.5 kHz).
   - Or: filter both sidebands separately and process independently.
   - Filter design: ~64-128 tap FIR at 2.048 MHz, passband 129-198 kHz.

2. **Windowing:** Divide the filtered sideband IQ into windows (e.g., 7 ms =
   14,336 samples at 2.048 MHz, or ~3 ms = one OFDM symbol period for
   symbol-boundary detection).

3. **Snippet extraction:** For each window, extract a short HD Radio IQ
   snippet and include it in the SyncEvent. The server cross-correlates
   snippets between nodes for fine timing.

4. **Optional: OFDM symbol boundary detection** via cyclic prefix
   autocorrelation. The CP is ~364 usec = ~746 samples at 2.048 MHz. The
   autocorrelation of the IQ at lag = symbol_duration - CP_duration produces
   a peak at each symbol boundary. This gives a periodic timing reference at
   ~344 Hz without needing to know the data content. But this requires knowing
   the exact OFDM parameters (which are standardized in NRSC-5).

**Detailed changes:**

| File | Change |
|------|--------|
| `hd_sync_detector.py` (new) | HD Radio sideband bandpass filter + snippet extractor. Operates on raw IQ at SDR rate (2.048 MHz). Emits `HDSyncEvent` with sideband IQ snippet. Optionally detects OFDM symbol boundaries. |
| `pipeline.py` | Add HD Radio sync path branching off the raw IQ before the sync decimator. New config fields: `hd_sync_enabled`, `hd_sideband` (upper/lower/both), `hd_snippet_samples`. The HD path runs in parallel with the pilot path. |
| `sync_detector.py` | No changes -- pilot path continues to provide coarse timing and crystal calibration. |
| `delta.py` | Accept `HDSyncEvent` as an alternative/supplementary sync source. When both pilot and HD sync events are available, prefer HD for fine timing. |
| `events/model.py` | Add `hd_snippet_b64` and `hd_sample_rate_hz` fields to `CarrierEvent`. |
| `config/schema.py` | Add `hd_sync` section: `enabled`, `sideband`, `snippet_samples`, `filter_taps`. |

**Server-side changes:**

| File | Change |
|------|--------|
| `tdoa.py` | New function `cross_correlate_hd_snippets()` that cross-correlates HD Radio sideband IQ between two nodes. Unlike carrier power envelope xcorr, this can use complex cross-correlation because both nodes receive the same HD Radio signal with the same phase relationship (the HD Radio carrier phase is locked to the FM carrier, which is common). However, LO frequency offsets between nodes (+/- a few ppm) will still rotate the phase -- use magnitude of the complex xcorr, or power envelope xcorr as with carrier snippets. |
| `tdoa.py` | In `compute_tdoa_s()`, add a third timing method: HD Radio sideband xcorr. Priority order: (1) HD sideband xcorr if available (highest precision), (2) carrier IQ snippet xcorr (current primary), (3) sync_delta subtraction (fallback). |
| `api.py` | Accept new fields in event ingest. |

**Data volume impact:** HD Radio sideband snippets at 2.048 MHz for 7 ms =
14,336 complex samples * 8 bytes = ~115 KB per event. This is too large.

**Mitigation -- decimate before transmission:** The HD Radio sideband is only
69 kHz wide. Decimate the filtered sideband IQ by 16x (2.048 MHz -> 128 kHz)
before extracting the snippet. 7 ms at 128 kHz = 896 complex samples * 8 bytes
= ~7 KB. Comparable to Option A.

Or: shorter snippets. 2 ms (roughly one OFDM symbol) at 128 kHz = 256 complex
samples = ~2 KB. Still enough bandwidth for sub-microsecond xcorr.

#### Pipeline data flow (with HD Radio)

```
SDRReceiver (IQ at 2.048 MSPS)
    |
    +-> sync_decimator (-> 256 kHz)
    |       +-> FMDemodulator
    |               +-> FMPilotSyncDetector ------> SyncEvent (coarse timing)
    |
    +-> hd_bandpass (129-198 kHz) -> hd_decimator (-> 128 kHz)
    |       +-> HDSyncDetector ------> HDSyncEvent (fine timing snippet)
    |                                      |
    +-> target_decimator (-> 64 kHz)       |
            +-> CarrierDetector            |
                    +-> CarrierOnset --> DeltaComputer -> TDOAMeasurement
                                           (uses SyncEvent + HDSyncEvent)
```

#### Testing

- Prototype: capture raw IQ from KUOW at 2.048 MHz, bandpass filter to
  isolate HD Radio sidebands, verify wideband signal is present and has
  sufficient SNR for xcorr.
- Unit test: synthetic OFDM-like signal, verify xcorr peak width matches
  expected 1/BW.
- Field test: two co-located RSPduo nodes, compare HD sideband xcorr TDOA
  jitter against pilot-only sync_delta jitter.

#### Risks

- Not all FM stations broadcast HD Radio. Must verify before selecting a sync
  station. KUOW (94.9 MHz, Seattle) does broadcast HD.
- HD Radio power is low (-10 to -20 dBc). At our antenna/location, the sideband
  SNR may be marginal. Need to measure empirically.
- OFDM symbol parameters (CP length, symbol duration) must match the NRSC-5
  standard exactly for CP-based symbol boundary detection. Getting these wrong
  means no correlation peak.
- More complex pipeline with two parallel sync paths. Increased CPU usage
  (additional bandpass filter + decimator at full SDR rate).
- The HD Radio sideband is noise-like (encrypted OFDM data). Cross-correlation
  between two nodes works because they receive the same signal, but the
  correlation peak position is only meaningful within one OFDM symbol period
  (~2.9 ms). Longer-range timing ambiguity must still be resolved by the pilot.

#### Phase 0: Feasibility check (before committing to full implementation)

Before building the full pipeline, validate the approach with a standalone
script:

1. Capture 10 seconds of raw IQ from two co-located nodes receiving KUOW.
2. Bandpass filter to isolate upper HD Radio sideband.
3. Window into 7 ms chunks and cross-correlate between nodes.
4. Measure the xcorr peak width and SNR.
5. Compare TDOA jitter (std dev over many windows) against pilot-only.

If the sideband SNR is insufficient (<3 dB after filtering), the approach
may not work at this antenna/distance. If the jitter is not significantly
better than the pilot, the added complexity is not justified.

Script: `scripts/prototype_hd_sync.py` (untracked, for experimentation).

---

### Option C: Hybrid Pilot + HD Radio (recommended long-term architecture)

Use both signals in a two-tier timing architecture:

1. **Pilot (coarse):** Provides the periodic timing grid (every 7 ms),
   crystal calibration, and long-range timing continuity. This is the
   "always works" baseline -- every stereo FM station has a pilot.

2. **HD Radio (fine):** When available, provides sub-microsecond refinement
   within each pilot sync window. The server cross-correlates HD sideband
   snippets to sharpen the TDOA measurement by 7-15x.

The pilot resolves timing ambiguity (which OFDM symbol are we in?) while
HD Radio provides the precision. This is analogous to how GPS uses coarse
acquisition (C/A) code for ambiguity resolution and carrier phase for
precision.

**Fallback:** If HD Radio is not available (station doesn't broadcast it, or
SNR is too low), the system falls back gracefully to pilot-only timing with
no degradation from current performance.

---

*Research date: 2026-04-05*
