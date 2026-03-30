# Beagle Event Reporting

## Overview

Each node produces `CarrierEvent` objects when a carrier onset or offset is
detected.  The `EventReporter` delivers these to the aggregation server via
HTTP POST without blocking the SDR processing pipeline.

---

## CarrierEvent JSON Schema

`CarrierEvent` is a Pydantic v2 model defined in `src/beagle_node/events/model.py`.
It is serialised with `model_dump_json()` and sent as a JSON object.

### Full schema

```json
{
  "schema_version": "1.4",
  "event_id":       "550e8400-e29b-41d4-a716-446655440000",

  "node_id":        "seattle-north-01",
  "node_location": {
    "latitude_deg":   47.7112,
    "longitude_deg": -122.3312,
    "altitude_m":     85.0,
    "uncertainty_m":   5.0
  },

  "channel_frequency_hz": 462562500.0,

  "sync_delta_ns":  12345678,
  "sync_transmitter": {
    "station_id":     "KISW_99.9",
    "frequency_hz":   99900000.0,
    "latitude_deg":   47.6253,
    "longitude_deg": -122.3563
  },
  "sdr_mode":       "rspduo",
  "pps_anchored":   false,
  "event_type":     "onset",

  "onset_time_ns":  1740000000123456789,

  "offset_time_ns": null,
  "duration_ms":    null,

  "peak_power_db":  -28.4,
  "mean_power_db":  -31.2,
  "noise_floor_db": -55.0,
  "snr_db":          23.8,
  "sync_corr_peak":   0.84,

  "clock_source":   "gps_1pps",
  "clock_uncertainty_ns": 800,

  "node_software_version": "0.1.0",

  "iq_snippet_b64":        "<base64-encoded int8 IQ>",
  "channel_sample_rate_hz": 62500.0
}
```

### Field reference

#### Identification

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version; currently `"1.4"` |
| `event_id` | UUID string | Node-local unique ID.  Stable if the same event is re-POSTed for amendment. |
| `node_id` | string | Node identifier from config (`[a-z0-9][a-z0-9-]*`) |

#### Location

| Field | Type | Description |
|-------|------|-------------|
| `node_location.latitude_deg` | float | Node WGS-84 latitude |
| `node_location.longitude_deg` | float | Node WGS-84 longitude |
| `node_location.altitude_m` | float | Node altitude above MSL (default 0) |
| `node_location.uncertainty_m` | float | Node location uncertainty radius (default 5 m) |

#### The TDOA measurement - primary fields

| Field | Type | Description |
|-------|------|-------------|
| `sync_delta_ns` | int | **Primary TDOA measurement.** Time from the preceding FM pilot sync event to this carrier edge, in nanoseconds, on the same local ADC sample clock. |
| `sync_transmitter.station_id` | string | Human-readable FM station ID |
| `sync_transmitter.frequency_hz` | float | FM carrier frequency (Hz) |
| `sync_transmitter.latitude_deg` | float | FCC-documented transmitter latitude |
| `sync_transmitter.longitude_deg` | float | FCC-documented transmitter longitude |
| `sdr_mode` | enum | `"freq_hop"`, `"rspduo"`, `"two_sdr"`, `"single_sdr"` |
| `pps_anchored` | bool | `true` if GPS 1PPS injection was used to align two SDR streams |
| `event_type` | enum | `"onset"` (rising edge) or `"offset"` (falling edge) |
| `channel_frequency_hz` | float | Nominal LMR channel center frequency (Hz) |

#### Absolute timing (event association only)

| Field | Type | Description |
|-------|------|-------------|
| `onset_time_ns` | int | GPS-disciplined wall-clock nanoseconds at the carrier edge.  Accuracy +/-1-10 usec (kernel scheduling); used by the server only to match events across nodes, not for TDOA computation. |
| `offset_time_ns` | int or null | Wall-clock time of the carrier offset (falling edge).  May be null if the offset has not yet been observed. |
| `duration_ms` | float or null | Transmission duration (ms).  Null until offset is known. |

#### Signal quality

| Field | Type | Description |
|-------|------|-------------|
| `peak_power_db` | float | Peak instantaneous power during this event (dBFS) |
| `mean_power_db` | float | Mean power over the carrier-active period (dBFS) |
| `noise_floor_db` | float | Estimated noise floor (dBFS) |
| `snr_db` | float | Signal-to-noise ratio (dB) |
| `sync_corr_peak` | float | FM pilot cross-correlation quality at the matched SyncEvent (0-1).  Values below ~0.3 indicate a weak or noisy pilot. |

#### Clock quality

| Field | Type | Description |
|-------|------|-------------|
| `clock_source` | enum | `"gps_1pps"`, `"ntp"`, `"unknown"` - source of the kernel clock used for `onset_time_ns` |
| `clock_uncertainty_ns` | int | Chrony-reported RMS offset uncertainty (nanoseconds) |

#### IQ cross-correlation (required)

| Field | Type | Description |
|-------|------|-------------|
| `iq_snippet_b64` | string | **Required.** Base64-encoded int8-interleaved IQ samples captured at the carrier edge.  Used by the server to cross-correlate matched events from different nodes for usec-level TDOA refinement.  Encoding: interleaved real/imag int8, so N complex samples -> 2N bytes.  The node captures `carrier.snippet_samples` (default 1280) complex samples centred on the transition edge. |
| `channel_sample_rate_hz` | float | **Required.** Sample rate of the IQ snippet in Hz (`sdr_rate_hz / target_decimation`).  The server uses this to convert the xcorr peak lag (samples) to nanoseconds. |

The server rejects any event missing either field with HTTP 422.

---

## EventReporter Design

### Non-blocking architecture

The SDR pipeline runs in the main thread at hard real-time: each 32 ms buffer
must be fully processed before the next one arrives.  Any latency in event
delivery (network I/O, retries, server unavailability) must not stall the
pipeline.

`EventReporter` achieves this with a **bounded FIFO queue + background worker
thread**:

```
pipeline thread                    worker thread
    |                                  |
    +-- reporter.submit(event) -->     |-- _deliver(event) --> HTTP POST
    |   (non-blocking, <1 usec)         |
    |                                  |-- retry if HTTP error
    |                                  |
    |   (if queue full:                |-- sleep (backoff)
    |    drop oldest event)            |
```

The worker thread uses a persistent `httpx.Client` connection (keep-alive) to
the server, avoiding per-event TCP and TLS handshake overhead.

### Queue behaviour

- **Capacity:** 1000 events (configurable via `reporter.max_queue_size` in `node.yaml`).
- **Overflow policy:** when full, the **oldest** pending event is dropped to make
  room for the new one.  This keeps the queue current during extended server
  outages rather than building up stale historical data.
- **On shutdown:** `reporter.stop()` signals the worker to drain the queue and
  exit cleanly, then waits up to 10 seconds.  Events queued before `stop()` is
  called are delivered (subject to the timeout).

### Retry policy

Each event gets up to 3 delivery attempts with **exponential backoff**:

| Attempt | Delay before retry |
|---------|-------------------|
| 1 -> 2 | 1 second |
| 2 -> 3 | 2 seconds |
| 3 (final) | - |

On HTTP 2xx: success, increment `events_submitted` counter.
On HTTP 4xx/5xx or network error: retry.
After 3 failures: drop event, increment `events_dropped` counter, log `ERROR`.

The `timeout_s` parameter (default 5 s) is the HTTP request timeout per attempt.

### Authentication

If `reporter.auth_token` is non-empty (or set via the `TDOA_AUTH_TOKEN` environment
variable), the worker sends an `Authorization: Bearer <token>` header with every
request.  If the token is empty, no Authorization header is sent.

---

## Server API

Events are delivered to the aggregation server's REST API.

### Endpoint

```
POST {reporter.server_url}/api/v1/events
Content-Type: application/json
Authorization: Bearer <token>    (if configured)

<CarrierEvent JSON object>
```

**Response codes accepted as success:** 200, 201, 202, 204.

### Other available routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/events` | List recent events (query: `limit`, `node_id`, `channel_hz`) |
| `GET` | `/api/v1/fixes` | List computed position fixes (query: `limit`, `max_age_s`) |
| `GET` | `/api/v1/fixes/{id}` | Get a specific fix with full detail |
| `GET` | `/health` | Server health: uptime, event count, fix count, last fix age |
| `GET` | `/map` | Folium interactive HTML map (query: `max_age_s=3600`) |
| `GET` | `/maps/{filename}` | Serve static per-fix Folium HTML snapshots |

Write endpoints (`POST /api/v1/events`) require authentication if
`server.auth_token` is set in the server config.  Read/map endpoints are
unauthenticated.

---

## Operational Monitoring

### Node health endpoint

The node exposes a `/health` HTTP endpoint (default port 8080) returning a JSON
summary of pipeline state.  Relevant fields:

```json
{
  "node_id":             "seattle-north-01",
  "uptime_s":            3600,
  "events_submitted":    148,
  "events_dropped":      0,
  "queue_depth":         0,
  "sdr_overflows":       0,
  "clock_source":        "gps_1pps",
  "clock_uncertainty_ns": 850
}
```

Monitor `events_dropped` (persistent queue overflow -> server unreachable) and
`sdr_overflows` (USB bandwidth exhaustion) in production.

### Log fields

Each successfully queued event logs at `INFO` level:

```
Measurement: sync_delta_ns=12345678 corr=0.840
```

A `corr` below 0.3 warrants investigation (weak FM pilot, gain too low, wrong station).

Delivery failures log at `WARNING` (per retry) and `ERROR` (final drop).
Set `log_level: DEBUG` in `node.yaml` to see per-sync-event corr_peak values,
which helps diagnose sync dropouts.

---

## Config Reference

Relevant section of `node.yaml`:

```yaml
reporter:
  server_url: "https://tdoa.example.com"  # empty = log locally only
  auth_token: ""                           # or set TDOA_AUTH_TOKEN env var
  max_queue_size: 1000
  timeout_s: 5.0
  # (batch_size and flush_interval_ms are in the config schema but not
  #  currently used by EventReporter; each event is delivered individually)

clock:
  source: "gps_1pps"           # gps_1pps | ntp | system
  calibration_offset_ns: 0     # subtract from all wall-clock timestamps
                                # (compensates for known buffer latency)
```

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
