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

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
