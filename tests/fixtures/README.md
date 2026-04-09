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

**Status:** pre-bugfix baseline. The pairwise `sync_delta_ns` differences in
this capture have ~245 us standard deviation (vs ~10 us expected for healthy
nodes), reflecting an unidentified noise source in the timing pipeline.
Once the bug is fixed, this fixture should be re-used as a regression test:
"replay these events; expect cross-pair sync_delta std under 30 us".

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
