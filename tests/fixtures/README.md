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

Each record contains both nodes' `sync_delta_ns`, `onset_time_ns`, location
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

The pairwise `sync_delta_ns` differences in this capture have ~245 µs
standard deviation -- not because of an unidentified bug, but because the
nodes were running code that has since been fixed.  The IQ snippets are
also captured at 62.5 kHz / 1280 samples, so server-side knee finding
runs on much coarser data than the current pipeline produces.

A replacement capture from the current deployment is needed before the
fixture is useful for regression testing.  When regenerated, target a
baseline that can assert "cross-pair sync_delta std under 10 µs, and
median per-event knee TDOA error under 100 µs".

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
