# Test Fixtures

## `real_event_pairs.json`

Paired IQ snippet measurements captured from real hardware, used by
`tests/unit/test_real_data.py` for regression testing of the xcorr pipeline.

This file ships empty. To populate it with data from your own deployment:

```bash
python3 scripts/export_fixtures.py \
    --db data/tdoa_data.db \
    --node-a <node-id-1> --node-b <node-id-2> \
    --output tests/fixtures/real_event_pairs.json
```

Each record contains both nodes' `sync_to_snippet_start_ns`, `onset_time_ns`, location
fields, and base64-encoded IQ snippets for one matched event pair. Only pairs
where both nodes report a snippet are included.

The tests in `test_real_data.py` are automatically skipped when this file
is empty.

Note: a few tests in `TestRealDataPilotDisambiguation` hardcode specific pair
indices (e.g. "pair 3 must be in n=+1 range"), so they will not pass against
arbitrary populated data — those preconditions encode the original test
author's private capture.

## `three_node_baseline_2026_04_08.json`

Full three-node event capture from 2026-04-08 18:20-18:35 PDT, Magnolia
443.475 MHz repeater, KUOW 94.9 sync transmitter. 116 raw events with IQ
snippets from `dpk-tdoa1`, `dpk-tdoa2`, and `kb7ryy`, suitable for end-to-end
pipeline regression tests that need to see the same physical transmission
across all three nodes simultaneously.

**Status (2026-04-19): needs regeneration.**  This capture predates:

- the Mueller-Muller → pilot-phase-derived sync fix (commits `4041b9d`,
  `1880637`, `7a0b6db`) that reduced cross-node onset spread from ~250 µs
  to ~105 ns
- the target channel rate bump from 62.5 kHz to ~250 kHz (commit `46a43c8`)
- the switch to server-side argmin(d2) knee finding with a 5120-sample
  snippet (commits `08eb316`, `5c2740d`, `e33e100`)
- auto-tracked carrier thresholds (commits `abda5ef`, `4d0617b`)

The pairwise `sync_to_snippet_start_ns` differences in this capture have ~245 µs
standard deviation -- not because of an unidentified bug, but because the
nodes were running code that has since been fixed.  The IQ snippets are
also captured at 62.5 kHz / 1280 samples, so server-side knee finding
runs on much coarser data than the current pipeline produces.

A replacement capture from the current deployment is needed before the
fixture is useful for regression testing.  When regenerated, target a
baseline that can assert "cross-pair sync_delta std under 10 µs, and
median per-event knee TDOA error under 100 µs".

## `kuow_sync_audio_30s.npz`

30 s of FM-demodulated audio from KUOW 94.9 MHz captured 2026-04-09 via
`BEAGLE_CAPTURE_SYNC_AUDIO`.  250 kHz sample rate, float32.  Contains
the 19 kHz pilot at ~10 dB above noise floor and the 57 kHz RDS
subcarrier at ~17 dB SNR.

Used by `tests/unit/test_rds_real_audio.py` to regression-test the
pilot-derived RDS bit-boundary timing on real signal conditions
(multipath, fading, noise, etc.).

## `kuow_sync_audio_dpk_tdoa1_10s_20260524.npz`

10 s capture from `dpk-tdoa1` on 2026-05-24, used for RDS-block-decoder
development (Layer 3a of the sync-alignment work).  Same format as
`kuow_sync_audio_30s.npz`.

Spectral characteristics (matches the April fixture, ruling out
fixture-degradation as a cause):

- Pilot peak at 19012.45 Hz (≈ +632 ppm SDR clock vs nominal 19000)
- RDS subcarrier lobes at 55.8 kHz / 58.2 kHz around suppressed 57 kHz
  carrier (suppressed-carrier null power 1.7e+3 vs lobe peaks ~6e+4)
- Upper-lobe / lower-lobe asymmetry of ≈ 1.5× (consistent across
  fixtures so a real KUOW broadcast characteristic, not an SDR or
  demodulator artefact)
- Pilot / RDS total-power ratio ≈ 1.7×, lower than typical broadcast
  (6-25×) -- KUOW transmits RDS at relatively high modulation depth
- Per-bit matched-filter BER on this fixture is approximately 2-3%
  with naive BPSK demodulation; A/B/C/D block-letter syndromes appear
  at ~5× the random baseline; C' correctly stays near the random
  baseline (type-B groups are rare on this station).

The high BER means simple "two-consecutive-blocks 26 bits apart"
lock-acquisition is not reliable on this signal.  A working decoder
on this fixture needs either (a) error correction via the (26,16)
single-bit-correctable Hamming code, (b) long-integration block-
position detection (statistical accumulation across many groups), or
(c) substantially better carrier tracking (a true Costas PLL rather
than windowed `arg(mean(z²))/2` averaging).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
