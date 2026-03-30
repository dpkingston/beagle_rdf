# Beagle Design Overview

## Purpose

Beagle is a Time Difference of Arrival (TDOA) radio direction finding system using Software Defined Radios (SDRs) distributed around the Seattle area. The goal is to locate the source of land mobile radio (LMR) transmissions to neighborhood-level accuracy (a few km).

## System Architecture

```
+-------------------------------------------------------------+
|  Node (Raspberry Pi 5 + SDR + GPS)                          |
|                                                             |
|  +----------+    +---------------------------------------+  |
|  | SDR sync |--->| FM pilot (19 kHz) sync event detector |  |
|  | (FM band)|    +-----------------+---------------------+  |
|  +----------+                      | SyncEvent               |
|                                    v                         |
|  +----------+    +-------------------------------------+    |
|  |SDR target|--->| LMR carrier detect + DeltaComputer  |    |
|  |(LMR band)|    +-------------+-----------------------+    |
|  +----------+                  | CarrierEvent                |
|  +----------+                  v                             |
|  | GPS      |    +-------------------------------------+    |
|  | receiver |--->| EventReporter (HTTP POST to server) |    |
|  +----------+    +-------------------------------------+    |
+-------------------------------------------------------------+
                              |
                    (internet / LAN)
                              |
              +---------------v--------------+
              |  Aggregation Server (future)  |
              |  - TDOA computation           |
              |  - Hyperbolic geolocation     |
              |  - Web dashboard              |
              +------------------------------+
```

## Node Data Flow

1. **SDR capture** - Two SDRs (or one freq-hopping) capture IQ samples continuously
2. **Decimation** - Band-pass filter + downsample to working rate (~48 kHz)
3. **FM demodulation** - Discriminator demod on the sync channel (FM station)
4. **Pilot extraction** - 19 kHz bandpass + cross-correlation -> `SyncEvent` every 10 ms
5. **Carrier detection** - Power-threshold state machine on the target channel -> `CarrierOnset`
6. **Delta computation** - `sync_delta_ns = target_onset_sample - sync_event_sample) / sample_rate`
7. **Event reporting** - Serialize `CarrierEvent` -> HTTP POST to aggregation server

## TDOA Measurement Model

### What each node measures

```
sync_delta_ns = (sample index of LMR carrier onset
               - sample index of preceding FM pilot sync event)
               x (1,000,000,000 / sample_rate_hz)
```

Both sample indices are from **the same continuous ADC clock**. This eliminates absolute clock synchronization as a precision requirement - only the local sample rate stability (<<1 ppm over <200 ms) matters.

### What the server computes

```
TDOA_AB = sync_delta_A - sync_delta_B

TDOA_AB_corrected = TDOA_AB - (dist(sync_tx, node_A) - dist(sync_tx, node_B)) / c
```

The path-delay correction accounts for the FM station's signal arriving at different times at each node (due to different distances). With FCC-documented transmitter coordinates (accuracy ~100 m), this correction is accurate to <1 usec across Seattle metro.

The corrected TDOA constrains the target to a hyperbola. With 3+ nodes, the intersection of hyperbolas locates the target.

## Node Hardware

Each deployed node consists of:

| Component | Role |
|-----------|------|
| Raspberry Pi 5 (8 GB RAM) | Local compute |
| RTL-SDR TCXO dongle (sync) | FM sync signal reception (88-108 MHz) |
| RTL-SDR TCXO dongle (target) | LMR target reception (VHF/UHF) |
| GPS module with 1PPS output | Clock discipline + inter-SDR alignment |
| SMA tees + 10 Mohm resistors | GPS 1PPS injection into SDR inputs |
| Suitable antenna(s) | Band-appropriate reception |
| PoE power or AC adapter | Power supply |

**Alternative: SDRplay RSPduo** - a single device with two independent RF
tuners sharing one ADC clock.  Eliminates inter-channel USB jitter and GPS
1PPS injection hardware without any coverage gaps.  Requires the `rspduo`
SDR mode and the SoapySDRPlay3 plugin.  See
[06-hardware-options.md](06-hardware-options.md) for evaluation details.

For development: a single RTL-SDR dongle using `librtlsdr-2freq` frequency hopping is sufficient.

## Sync Signal: FM Stereo Pilot Tone

Each node uses a Seattle FM broadcast station as a timing reference:

- **KISW 99.9 MHz** (primary, used in earlier experiments)
- **KUOW 94.9 MHz** (secondary, for calibration cross-check)

The FM stereo pilot at exactly 19,000 Hz (locked to the station's frequency standard) provides 38,000 timing events per second with sub-microsecond precision via phase interpolation.

See [04-sync-signal.md](04-sync-signal.md) for details.

## Aggregation Server Interface

Nodes POST JSON arrays of `CarrierEvent` objects to `POST /api/v1/events`.

**Required fields for TDOA computation:**
- `sync_delta_ns` - the precise timing measurement
- `sync_transmitter` - FM station ID and FCC-documented lat/lon
- `onset_time_ns` - rough absolute time (for event association across nodes)
- `node_location` - node lat/lon for path-delay correction
- `channel_frequency_hz` - which LMR channel

See [05-event-reporting.md](05-event-reporting.md) for the full JSON schema.

## Accuracy

| SDR mode | Timing precision | Position uncertainty |
|----------|-----------------|---------------------|
| `freq_hop` (single RTL-SDR) | ~1-5 usec | ~300 m-1.5 km |
| `rspduo` (SDRplay RSPduo) | ~1-2 usec | ~300-600 m |
| `two_sdr` + GPS 1PPS injection | <1 usec | ~300 m |
| `single_sdr` (wideband) | <1 usec | ~300 m |

The neighborhood-level goal (few km) is met by all modes.  The `rspduo`
mode combines the shared-clock precision of `two_sdr` with the simplicity
of `freq_hop` - no GPS 1PPS injection hardware required.

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
