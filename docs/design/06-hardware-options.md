# Hardware Options for Beagle

## SDRplay RSPduo

*Evaluated 2026-02-25*

### Summary

The RSPduo is a strong candidate for Beagle `two_sdr` mode. Its single-ADC
dual-tuner architecture eliminates the inter-channel USB jitter problem that
makes naive two-USB-device operation unsuitable for TDOA, without requiring
GPS 1PPS injection hardware.

### Architecture

The RSPduo contains two independent RF tuners (MSi001) feeding a **single
shared ADC** (MSi2500) via time-division multiplexing. The ADC interleaves
samples: odd IQ pairs come from Tuner 1, even pairs from Tuner 2 (or vice
versa). Both tuners are clocked from the **same 24 MHz TCXO**. All data
flows through one USB 2.0 connection as a single interleaved stream.

```
Tuner 1 (sync, e.g. 99.9 MHz FM)  ─┐
                                    ├─► single ADC (TDM) ─► USB ─► host
Tuner 2 (target, e.g. 155 MHz LMR) ─┘
```

### Why this matters for Beagle

The design document's `two_sdr` mode complexity (GPS 1PPS injection circuit,
attenuator hardware, `PPSDetector`, `PPSAnchor` pipeline) exists entirely to
solve one problem: two separate USB SDRs have independent sample clocks and
independent USB transfer scheduling, so inter-channel timing has ~1–10 ms
USB jitter — far too coarse for TDOA.

The RSPduo eliminates this problem by design:

| Property | Two RTL-SDRs | RSPduo |
|----------|-------------|--------|
| ADC clock | Independent per device | **Shared TCXO** |
| USB transfers | Separate (jitter source) | **One shared stream** |
| GPS 1PPS hardware | Required | **Not needed** |
| Coverage gaps | `freq_hop` only | **None** |
| Tuner settling time | `freq_hop` only | **None** |
| ADC resolution | 8-bit | **14-bit** |
| Frequency range | 24 MHz – 1.8 GHz | **1 kHz – 2 GHz** |

### Timing model

The single-sample ADC interleave introduces a **deterministic 0.5-sample
offset** between the two channels. At 2 MHz sample rate, this is 0.25 µs.
It is constant and correctable with a single fixed term:

```
sync_delta_corrected_ns = sync_delta_raw_ns − (0.5 / sample_rate_hz) × 1e9
```

This makes the RSPduo timing model almost identical to `freq_hop`: both
channels share one continuous sample clock, so `sync_delta_ns` is simply
`(target_sample − sync_sample) × 1e9 / rate − pipeline_offset_ns`.
No `PPSAnchor` machinery is needed.

### Comparison to existing modes

```
freq_hop:   one ADC clock, no extra hardware, but has coverage gaps
            and ~20 ms settling time per frequency hop
rspduo:     one ADC clock, no extra hardware, no gaps, no settling  ← sweet spot
two_sdr:    two ADC clocks, requires GPS 1PPS injection hardware,
            no gaps, no settling
```

The RSPduo sits between `freq_hop` and full `two_sdr` — it gets the
continuous-coverage benefit of `two_sdr` without the GPS 1PPS hardware
requirement.

### Limitations

- **2 MHz max bandwidth per tuner in dual mode.** In single-tuner mode each
  tuner supports up to 10 MHz, but both tuners active simultaneously is
  limited to 2 MHz each. This is adequate for narrowband LMR (25 kHz
  channels) and matches the current pipeline's 2.048 MSPS working rate.
- **SoapySDRPlay3 must be built from the `rspduo-dual-independent-tuners`
  branch.** Independent per-channel tuning in `mode=DT` is not present in
  the upstream master or pre-built packages.  See `docs/setup-rspduo-debian.md`.
- **Single USB connection = single point of failure** for both channels.
  Acceptable for a node; loss of the USB connection loses both channels.
- **~$200–250** vs. ~$35 per RTL-SDR dongle.

### API note: Init and Tuner B parameters

`sdrplay_api_Init()` fires during the first `activateStream()` call.  In
older SoapySDRPlay3 builds it would program **Rf_B = Rf_A**, ignoring ch1's
frequency.  The `rspduo-dual-independent-tuners` branch handles this
internally in `activateStream()` by re-applying any Tuner B parameters that
differ from the Init defaults.  Our code also re-applies ch1 frequency and
gains after both `activateStream()` calls as a safety net.

See `docs/setup-rspduo-debian.md` section 6 for details.

### Why Master/Slave mode is not used

The SDRplay API library enforces one selected device per `sdrplay_api_Open()`
handle.  SoapySDRPlay3 uses a process-wide singleton for this handle.
Opening a "Slave" device object calls `sdrplay_api_SelectDevice()` on the
same handle that the "Master" already holds, which returns `sdrplay_api_Fail`
inside the library — the request never reaches the sdrplay service.  The
symptom is the Slave `SoapySDR.Device()` constructor always throws
`RuntimeError("sdrplay_api_Fail")` regardless of retry count.

Use DT mode (single device object, two streams) instead.

### Beagle integration (implemented 2026-02-26, DT fix 2026-03-08)

The `rspduo` mode is implemented as a first-class SDR mode alongside
`freq_hop`, `two_sdr`, and `single_sdr`:

- **`sdr/rspduo.py`** — `RSPduoReceiver` opens the RSPduo in DT mode
  (Dual Tuner) as a single SoapySDR device with two separate single-channel
  streams.  The `rspduo-dual-independent-tuners` branch adds independent
  per-channel tuning to DT mode via a lazy-split mechanism.  The key method
  is `paired_stream()` which yields simultaneous `(sync_buf, target_buf)`
  pairs.  The `stream()` method is a compatibility shim for the `SDRReceiver`
  ABC.  Post-init frequency and gain re-apply (after both `activateStream`
  calls) are handled in `open()` as a safety net.
- **`config/schema.py`** — `RSPduoConfig` with `sync_frequency_hz`,
  `target_frequency_hz`, `sample_rate_hz`, `sync_gain_db`, `sync_lna_state`,
  `target_gain_db`, `target_lna_state`, `master_device_args`,
  `slave_device_args` (ignored; kept for config compat), `buffer_size`, and
  `pipeline_offset_ns` (default 0; calibrate empirically).
- **`sdr/factory.py`** — `create_receiver()` handles `sdr_mode="rspduo"`.
- **`main.py`** — dedicated `rspduo` loop branch using `paired_stream()`;
  `pipeline_offset_ns` is subtracted from every `sync_delta_ns` before
  the `CarrierEvent` is submitted to the reporter.
- **No changes** to `sync_detector.py`, `carrier_detect.py`, or `delta.py` —
  the timing model is identical to `freq_hop`.
- **`PPSDetector` not used** — shared ADC clock eliminates GPS 1PPS need.

To enable: set `sdr_mode: rspduo` and add an `rspduo:` block in `node.yaml`.
See `config/node.example.yaml` for a commented template and
`docs/setup-rspduo-debian.md` for the full setup procedure.

### References

- [SDRplay RSPduo product page](https://www.sdrplay.com/rspduo/)
- [RTL-SDR.com review — dual-tuner architecture details](https://www.rtl-sdr.com/sdrplay-release-a-dual-tuner-sdr-called-rspduo/)
- [DuoTools — dual-channel sample utilities](https://github.com/msiner/DuoTools)
- [SoapySDRPlay3 — Dual Tuner (independent RX) mode](https://github.com/pothosware/SoapySDRPlay3)

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License — see [LICENSE](../../LICENSE).
