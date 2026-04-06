# RDS-Based Sync Event Implementation Plan

## The Problem

The current sync event system is fundamentally broken for **all** node pairs,
including co-located ones. Each node's `FMPilotSyncDetector` picks the 19 kHz
pilot zero-crossing nearest to the center of an arbitrary 7 ms window. Two nodes
with different window grids (which is always the case -- the grid starts at SDR
startup time) pick **different** zero-crossings, offset by an unknown number of
pilot cycles (N * 52.6 usec). The sync_delta subtraction between nodes therefore
contains an unknown N * 52.6 usec error.

For **non-co-located nodes**, this produces wildly wrong fixes: the sync_delta
differences scatter across the full +/-3500 usec pilot disambiguation window,
and the geometric correction (~14 usec) is far too small to resolve the
ambiguity. Fixes land at the search boundary.

For **co-located nodes**, the error is masked: the true TDOA is ~0, so the
7 ms disambiguation window absorbs the N * 52.6 usec offset, producing results
within +/-3500 usec of zero. This appeared to "work" because small errors near
zero were attributed to noise, but the hundreds-of-microsecond scatter seen in
co-located pair tests was actually the pilot cycle ambiguity, not measurement
noise.

**The fix:** Replace the arbitrary zero-crossing selection with a sync event
anchored to a **physically identifiable feature** of the broadcast signal that
all nodes can recognize unambiguously. The RDS data stream provides this.

---

## Background: RDS Signal Characteristics

RDS is a 1187.5 bps BPSK data stream modulated on a 57 kHz subcarrier (3x the
19 kHz pilot). Key properties for timing:

| Parameter | Value |
|-----------|-------|
| Bit rate | 1187.5 bps (= 19000 / 16, exact) |
| Bit period | 842.1 usec |
| Subcarrier | 57 kHz (3x pilot, phase-locked) |
| Bandwidth | +/- 2.4 kHz around 57 kHz |
| Group length | 104 bits = 87.58 ms |
| Block length | 26 bits (16 data + 10 CRC) = 21.9 ms |
| Blocks per group | 4 (A, B, C/C', D) |
| Groups per second | 11.4 |
| Injection level | ~-11 dB relative to pilot |
| Presence | >95% of US FM stations, all NPR affiliates |

The bit clock is **exactly** pilot/16, phase-coherent by specification. All
nodes receiving the same station see the same bit transitions at the same
physical time (plus propagation delay).

**CRC syndrome detection:** Each 26-bit block has a 10-bit CRC with a
position-dependent offset word. Computing the syndrome on a sliding 26-bit
window reveals which block position (A/B/C/D) is present, without needing to
decode the data content. Block A contains the PI (station ID) code -- the same
16 bits in every group.

---

## Option 1: RDS Bit Transition Sync Events

### Concept

Replace the pilot zero-crossing sync event with an RDS **bit boundary** sync
event. Every 842 usec, a BPSK bit transition occurs on the 57 kHz subcarrier.
All nodes see the same transitions at the same time (plus propagation). The bit
period (842 usec) is much larger than the maximum geometric TDOA (~333 usec for
100 km baseline), so disambiguation is straightforward:

```
|true_TDOA| < 333 usec << 842/2 = 421 usec
```

One-to-one correspondence: `round((raw + correction) / 842000)` is unambiguous.

### Signal Processing Chain

Current sync chain (to be extended, not replaced):

```
SDR IQ (2.048 MHz)
  -> Decimator (8x -> 256 kHz)
  -> FM Demodulator (-> baseband audio at 256 kHz)
  -> FMPilotSyncDetector (19 kHz BPF + correlation -> SyncEvent every 7ms)
```

New RDS chain (branches from FM demodulator output):

```
FM demodulated audio (256 kHz)
  -> Regenerate 57 kHz carrier (3x pilot, using recovered pilot phase)
  -> Mix to baseband (multiply by cos/sin of 57 kHz -> I/Q at DC)
  -> Lowpass filter (2.4 kHz cutoff, ~32 taps at 256 kHz)
  -> Decimate (16x -> 16 kHz, 13.47 samples/bit)
  -> BPSK timing recovery (Gardner or Mueller-Muller TED)
  -> Bit slicer
  -> RDS bit clock output -> RDSSyncEvent (one per bit period, 842 usec)
```

### New Module: `rds_sync_detector.py`

**Input:** Demodulated FM audio at 256 kHz (same signal that feeds
`FMPilotSyncDetector`).

**Output:** `RDSSyncEvent` with:
- `sample_index: int` -- sample position of the bit transition (in sync IQ
  stream coordinates, same domain as pilot SyncEvent)
- `bit_phase: float` -- phase of the BPSK symbol for crystal calibration
- `corr_peak: float` -- quality metric (bit transition sharpness)
- `block_position: int | None` -- A/B/C/D if CRC syndrome matched (for
  group-level sync), None if not yet locked

**Processing steps:**

1. **57 kHz carrier regeneration:** The pilot BPF output from
   `FMPilotSyncDetector` provides the pilot phase. Multiply by 3 to get the 57
   kHz phase. Generate cos(3*phi) and sin(3*phi) reference signals.

2. **Mixing to baseband:** Multiply the FM audio by the I/Q reference.
   Output: complex baseband RDS signal centered at DC.

3. **Lowpass filter:** 2.4 kHz cutoff FIR at 256 kHz. With transition band of
   ~1 kHz, need ~32-64 taps. Cheap because the signal is narrowband.

4. **Decimation:** 16x to 16 kHz. At this rate, 13.47 samples per bit.

5. **BPSK timing recovery:** Gardner timing error detector (TED) or
   Mueller-Muller TED. These are standard digital comm algorithms that track
   the bit clock from the baseband BPSK waveform. The output is the optimal
   sampling instant for each bit.

6. **Bit slicer + sync event generation:** At each recovered bit boundary,
   emit an `RDSSyncEvent` with the `sample_index` mapped back to the 256 kHz
   sync stream coordinates.

### Integration with Pipeline

**`pipeline.py` changes:**

```
# In __init__:
self._rds_det = RDSSyncDetector(
    sample_rate_hz=c.sdr_rate_hz / c.sync_decimation,  # 256 kHz
    pilot_detector=self._sync_det,  # shares pilot phase
)

# In process_sync_buffer():
# After FM demod:
audio = self._sync_demod.process(iq_dec)
pilot_events = self._sync_det.process(audio, ...)
rds_events = self._rds_det.process(audio, ...)
for rds_ev in rds_events:
    self._delta.feed_sync(rds_ev)  # RDSSyncEvent is a SyncEvent
```

**`delta.py` changes:**

The `DeltaComputer._match()` method finds the most recent sync event before the
carrier event. Currently it searches `_sync_events` for the nearest pilot sync.
With RDS sync events at 842 usec intervals (vs 7 ms for pilot windows), there
are ~8x more sync events, but the nearest one is always within 842 usec of the
carrier event -- much tighter than the current 7 ms.

The `SyncEvent` dataclass needs to be compatible with `RDSSyncEvent`, or both
should implement a common protocol. The simplest approach: `RDSSyncEvent`
inherits from or replaces `SyncEvent`.

**`tdoa.py` (server) changes:**

The disambiguation formula changes from:
```python
n = round((raw_ns + correction_ns) / 7_000_000)  # pilot window period
```
to:
```python
n = round((raw_ns + correction_ns) / 842_105)    # RDS bit period
```

With `|true_TDOA| < 333 usec` and `T_rds/2 = 421 usec`, this is unambiguous.

### Resource Costs

**CPU (on Pi 5):**

| Component | Estimated cost | Notes |
|-----------|---------------|-------|
| 57 kHz mixing | ~2% of pipeline | Two multiplies per sample at 256 kHz |
| LPF (32-tap at 256 kHz) | ~5% of pipeline | Similar to existing pilot BPF |
| Decimate 16x | ~1% | Trivial (stride) |
| Timing recovery at 16 kHz | ~1% | Simple feedback loop at low rate |
| Total new | ~9% of pipeline | Pipeline currently uses ~15% of one core |

Estimated total CPU increase: from 15% to ~18% of one core. Well within the
5-7x realtime headroom on Pi 5.

**RAM:**

- Filter state: ~2 KB (32-tap FIR history + timing recovery state)
- Additional sync events: 1187 events/sec * ~64 bytes/event = ~76 KB/sec
  throughput, but only the last few are kept (pruned by `max_sync_age_samples`)
- Net additional RAM: < 100 KB

### Testing Strategy

**Phase 1: Standalone prototype (no pipeline changes)**

`scripts/prototype_rds_sync.py`:
1. Capture 10 seconds of raw IQ from a real station (KUOW)
2. Process through decimator + FM demod (existing code)
3. Implement the RDS chain in isolation
4. Verify bit transitions are detected at 1187.5 bps
5. Verify CRC syndromes match known block structure
6. Output bit transition sample indices

**Phase 2: Dual-node comparison**

Capture simultaneous IQ from two co-located nodes. Run the RDS sync detector
on both. Verify that the bit transition sample indices, after subtracting
propagation delay, agree to within one sample (~4 usec at 256 kHz).

**Phase 3: Unit tests**

- Synthetic FM baseband with known RDS bitstream -> verify bit recovery
- Two synthetic streams with known TDOA -> verify sync_delta gives correct TDOA
- Noise robustness: verify timing recovery works at 10 dB SNR

**Phase 4: Integration test on branch**

Deploy to two nodes on a feature branch. Run `colocated_pair_test.py` and
verify the sync_delta scatter drops from +/-3500 usec to +/-50 usec.

### Coding Complexity

**New code:**
- `rds_sync_detector.py`: ~200-300 lines (carrier regen, mixer, LPF, timing
  recovery, bit slicer, event generation)
- Pipeline integration: ~20 lines in `pipeline.py`
- Server disambiguation constant change: ~5 lines in `tdoa.py`

**Libraries needed:** None beyond numpy/scipy (already used). The timing
recovery is a simple feedback loop, not a complex library call.

**Risk areas:**
- BPSK timing recovery convergence time: the Gardner TED needs ~50-100 bit
  periods (~50-80 ms) to lock. During this time, sync events are unreliable.
  Mitigation: don't emit sync events until the timing loop has converged
  (monitor loop error variance).
- 57 kHz carrier phase ambiguity: the regenerated 57 kHz could be 0 or 180
  degrees off (pilot tripling gives 3x phase, but the RDS subcarrier is at
  90 degrees relative to the stereo subcarrier). This affects bit polarity but
  not transition timing.
- Filter group delay: the LPF adds a constant delay that must be accounted for
  in the sample_index mapping. This is deterministic and the same on all nodes,
  so it cancels in the TDOA subtraction.

### Limitations

- RDS bit period is 842 usec. For baselines > 126 km (TDOA > 421 usec),
  disambiguation becomes ambiguous. Our deployment is well within this limit.
- Not all FM stations have RDS. >95% in the US do, and all NPR affiliates do.
  The sync station selection must verify RDS presence.
- RDS is ~11 dB weaker than the pilot. At marginal FM reception, RDS timing
  recovery may fail while pilot detection still works. The pilot-based sync
  should remain as a fallback.

---

## Option 2: RDS Block/Group Sync Events

### Concept

Instead of using every RDS bit transition, use only the **Block A boundary**
as the sync event. Block A repeats every 104 bits (87.58 ms) and contains the
PI code (station identifier) -- the same 16 bits every time. This provides:

1. **Unambiguous identification:** The CRC syndrome uniquely identifies Block A.
   All nodes detecting the same Block A boundary are guaranteed to be
   referencing the same physical event.

2. **Very large disambiguation window:** The group period is 87.58 ms. The
   maximum geometric TDOA is ~333 usec. `round(raw / 87580000)` is unambiguous
   by a factor of >100.

3. **Known data content:** The PI code in Block A is fixed and predictable.
   This enables **correlation-based detection** -- correlate the received
   bitstream against the known Block A pattern for maximum SNR detection.

### Signal Processing Chain

Same as Option 1 through the bit slicer, plus:

```
Bit slicer output (1187.5 bps)
  -> Sliding 26-bit CRC syndrome check
  -> Block A detection (syndrome = 0x3D8)
  -> Verify consecutive A-B-C-D pattern (sync lock)
  -> On Block A boundary: emit RDSGroupSyncEvent
```

### New Module: `rds_sync_detector.py`

Same as Option 1 but with an additional layer:

**Output:** `RDSGroupSyncEvent` with:
- `sample_index: int` -- sample position of the Block A start
- `group_number: int` -- incrementing group counter (for tracking)
- `pi_code: int` -- decoded PI code (station verification)
- `corr_peak: float` -- quality metric
- `sample_rate_correction: float` -- from pilot crystal calibrator

**Additional processing (beyond Option 1):**

6. **CRC syndrome computation:** For each bit position, maintain a running
   26-bit shift register. Compute the syndrome at each position. This is a
   simple XOR/shift operation -- negligible CPU.

7. **Block sync state machine:**
   - SEARCHING: slide through bits, compute syndrome at each position
   - CANDIDATE: found a valid syndrome; check if the next 26 bits also match
     the expected next block
   - LOCKED: 2-3 consecutive valid blocks confirm sync; emit events at
     Block A boundaries
   - LOST: if 2+ consecutive blocks fail CRC, revert to SEARCHING

8. **Group-level event emission:** Emit one sync event per Block A start
   (11.4 per second).

### Integration with Pipeline

Same as Option 1, but the sync event rate is 11.4/sec instead of 1187.5/sec.
This means:

- Much fewer sync events stored in `DeltaComputer` (11 per second vs 1187)
- The nearest sync event is up to 87 ms before the carrier event (vs 842 usec
  for Option 1 or 7 ms for current pilot)
- The sync_delta values are larger (up to 87 ms vs up to 7 ms)
- Sample counting over 87 ms at 256 kHz = ~22,000 samples. Crystal error at
  10 ppm over 22,000 samples = 0.22 samples = ~0.9 usec. Still acceptable,
  especially with crystal calibration applied.

**`delta.py` changes:**

The `max_sync_age_samples` needs to increase to cover at least one full group
period: 87.58 ms * 256,000 Hz = ~22,400 samples. Current default is 7,680
(~30 ms). Change to ~30,000.

**`tdoa.py` (server) changes:**

```python
_T_SYNC_NS = 87_578_947.0  # RDS group period (104 bits at 1187.5 bps)
```

Disambiguation: `round((raw_ns + correction_ns) / 87578947)` with
`|true_TDOA| < 333 usec << 87579/2 = 43789 usec`. Completely unambiguous.

### Resource Costs

**CPU:** Same as Option 1 (the CRC syndrome check and block state machine add
negligible cost -- a few XOR operations per bit, 1187 times per second).

**RAM:** Less than Option 1 -- only 11 sync events per second instead of 1187.
Approximately 1 KB for the CRC state and shift register.

### Testing Strategy

Same Phase 1-4 as Option 1, plus:

**Phase 1 addition:** Verify CRC syndrome detection on real KUOW data. Confirm
Block A is detected at 11.4/sec with the correct PI code.

**Phase 2 addition:** Verify group-level sync events are identically numbered
on two co-located nodes (same group = same Block A detected).

### Coding Complexity

**Additional code beyond Option 1:**
- CRC syndrome computation: ~30 lines
- Block sync state machine: ~50 lines
- Total: Option 1 + ~80 lines

**Risk areas (in addition to Option 1 risks):**
- **Sync acquisition time:** Need 2-3 consecutive valid groups to lock.
  At 87.58 ms/group, this is 175-263 ms. During this time, no sync events
  are emitted. After a signal dropout, recovery takes ~250 ms.
- **Missed groups:** If the RDS signal drops below decodable SNR for one group,
  the state machine may lose lock and need to re-acquire. The pilot-based sync
  should serve as a fallback during re-acquisition.
- **Crystal error over 87 ms:** At 10 ppm, the accumulated error over one group
  period is 0.87 usec. This is acceptable but larger than for Option 1 (where
  the max interval is 0.84 ms -> 0.008 usec at 10 ppm). Crystal calibration
  (already implemented) corrects this.

### Limitations

- Sync event rate is only 11.4/sec. The carrier event must wait up to 87 ms
  for the most recent sync event. For freq_hop mode with short target blocks,
  a sync event might not fall within the current block. The pilot-based sync
  (at 143/sec) would still be needed for freq_hop mode.
- Higher latency to initial sync acquisition (250 ms vs ~50 ms for Option 1).

---

## Comparison

| Aspect | Option 1 (bit transitions) | Option 2 (group boundaries) |
|--------|---------------------------|----------------------------|
| Sync event rate | 1187.5 / sec | 11.4 / sec |
| Disambiguation period | 842 usec | 87,579 usec |
| Max TDOA / period | 333/421 = 79% | 333/43789 = 0.8% |
| Disambiguation safety | Adequate (1.3x margin) | Very safe (130x margin) |
| Max sync_delta | 842 usec | 87,579 usec |
| Crystal error at max | 0.008 usec @ 10ppm | 0.88 usec @ 10ppm |
| Sync acquisition time | ~50 ms | ~250 ms |
| Sync events in DeltaComputer | ~100 (at max_age=80ms) | ~1 |
| Coding complexity | Medium | Medium + 80 lines |
| CPU overhead | ~9% of pipeline | ~9% of pipeline |
| RAM overhead | ~100 KB | ~10 KB |
| freq_hop compatible | Yes | Marginal (87ms > target block) |
| CRC validation | Not needed | Provides station verification |
| Fallback to pilot | Easy (pilot still runs) | Easy (pilot still runs) |

## Recommendation

**Option 1 (RDS bit transitions)** is the better choice for several reasons:

1. **freq_hop compatibility:** The 842 usec sync period ensures a sync event
   is available within every target block, even short ones. Option 2's 87 ms
   period may exceed the target block length.

2. **Lower sync_delta magnitude:** 842 usec max means crystal error contributes
   at most 0.008 usec (at 10 ppm), vs 0.88 usec for Option 2. This matters
   for sub-microsecond TDOA accuracy.

3. **Faster sync acquisition:** ~50 ms vs ~250 ms means less downtime after
   signal dropouts.

4. **Disambiguation is adequate:** The 1.3x margin (333 usec / 421 usec) is
   sufficient given that the geometric correction is computed from known node
   positions with meter-level accuracy.

5. **Simpler implementation:** No CRC state machine needed. The bit-level
   timing recovery gives us what we need directly.

Option 2's CRC validation is valuable as a **diagnostic/verification tool**
(confirming all nodes are receiving the same station), but it's not needed for
the core timing function. It could be added later as an overlay on Option 1.

## Implementation Phases

### Phase 0: Feasibility prototype (untracked script, ~2 days)

Write `scripts/prototype_rds_sync.py` that:
1. Captures or loads raw IQ from KUOW
2. Runs existing decimator + FM demod
3. Implements the RDS chain (carrier regen, mix, LPF, timing recovery, slicer)
4. Outputs bit transition timestamps
5. Optionally: CRC syndrome check to verify block detection

Success criteria: bit transitions detected at the correct rate (1187.5/sec)
with stable timing recovery.

### Phase 1: Core module + unit tests (~3 days)

1. Write `rds_sync_detector.py` with the full signal chain
2. Unit tests with synthetic FM baseband
3. Unit tests verifying TDOA accuracy with known geometry

### Phase 2: Pipeline integration on feature branch (~2 days)

1. Branch: `feature/rds-sync`
2. Integrate `RDSSyncDetector` into `pipeline.py`
3. Update `delta.py` to accept RDS sync events
4. Update `tdoa.py` disambiguation constant
5. Config: `sync_mode: "rds"` (default) vs `"pilot"` (fallback)

### Phase 3: Field test with co-located nodes (~1 day)

1. Deploy to dpk-tdoa1 + dpk-tdoa2
2. Run `colocated_pair_test.py`
3. Verify sync_delta scatter < 50 usec (vs current thousands of usec)

### Phase 4: Non-co-located field test (~1 day)

1. Deploy to all nodes including kb7ryy
2. Verify fixes land near the Magnolia repeater (47.651, -122.391)
3. Measure fix accuracy with known transmitter location

### Phase 5: Merge to main

After successful field testing, merge `feature/rds-sync` to main.

---

*Research date: 2026-04-05*
*Related: docs/research/fm-sync-signal-options.md (HD Radio options)*
