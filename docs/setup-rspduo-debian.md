# RSPduo Setup on Debian Linux

Primary tested platform: **Raspberry Pi 5, Debian 13 (trixie), 64-bit**.  The
same steps work on Debian 12/13 amd64 (e.g. a server node). This may be similarly
applicable for other Linux systems.

The SDRplay SoapySDR plugin (`SoapySDRPlay3`) is not in the standard Debian
apt repos.  Installation requires four steps: SoapySDR base from apt, the
SDRplay API binary from sdrplay.com, the SoapySDRPlay3 plugin built from
source, and the Beagle Python environment with a SoapySDR binding shim.

---

## 1. SoapySDR base (apt)

```bash
sudo apt install \
    soapysdr-tools libsoapysdr-dev \
    python3-soapysdr \
    cmake build-essential git \
    python3-venv python3-pip
```

`python3-soapysdr` provides the SoapySDR Python bindings.  They are packaged
separately from the C++ library and are not pulled in by the other packages.

---

## 2. SDRplay API (binary installer from sdrplay.com)

Go to **https://www.sdrplay.com/downloads/** and download the Linux API
installer.  A single `.run` file supports x86-64, aarch64, and armhf - the
installer auto-detects the host architecture.

> **Important:** The installer prompts interactively for license acceptance and
> cannot be run non-interactively via a piped script.  Run it in a real
> terminal session (local console, `screen`/`tmux`, or `ssh -t`):

```bash
# Download and run - requires an interactive terminal
chmod +x SDRplay_RSP_API-Linux-3.x.x.run
sudo ./SDRplay_RSP_API-Linux-3.x.x.run
# Press RETURN to page through the license, then y to accept.
# Installs to /usr/local/lib/libsdrplay_api.so and starts the sdrplay service.
```

Enable the service to start on boot:
```bash
sudo systemctl enable sdrplay
sudo systemctl status sdrplay   # confirm: active (running)
```

---

## 3. SoapySDRPlay3 plugin (build from upstream branch)

The upstream `pothosware/SoapySDRPlay3` `master` branch does not support
independent per-channel tuning in Dual Tuner mode.  The
`rspduo-dual-independent-tuners` branch adds this capability to the existing
`mode=DT` entry - both tuners start mirrored, and the driver lazily splits
them into independent operation when a parameter is set on channel 1.
However event with this change they still don't support hardware timestamps
which are available from the rspduo, so we have a fork in github.com
with that support.

```bash
# Clone the branch with independent DT tuning support
git clone -b tdoa-hw-timestamps \
    https://github.com/dpkingston/SoapySDRPlay3.git
cd SoapySDRPlay3

# Build and install
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

# Restart the sdrplay service so the new .so is picked up
sudo systemctl restart sdrplay
```

> **Note:** `sudo ldconfig` is only needed on distros where
> `/usr/local/lib/SoapySDR/modules0.8/` is not already in the dynamic
> linker cache.  On Debian/Ubuntu, restarting the sdrplay service is
> sufficient since SoapySDR loads plugins via `dlopen` at runtime.

---

## 4. Verify device detection

```bash
SoapySDRUtil --find="driver=sdrplay"
```

For an RSPduo not currently in use by another process, this shows **four
entries** - one per operating mode:

```
Found device 0
  driver   = sdrplay
  label    = SDRplay Dev0 RSPduo XXXXXXXX - Single Tuner
  mode     = ST
  serial   = XXXXXXXX

Found device 1
  driver   = sdrplay
  label    = SDRplay Dev1 RSPduo XXXXXXXX - Dual Tuner
  mode     = DT
  serial   = XXXXXXXX

Found device 2
  driver   = sdrplay
  label    = SDRplay Dev2 RSPduo XXXXXXXX - Master
  mode     = MA
  serial   = XXXXXXXX

Found device 3
  driver   = sdrplay
  label    = SDRplay Dev3 RSPduo XXXXXXXX - Master (RSPduo sample rate=8Mhz)
  mode     = MA8
  serial   = XXXXXXXX
```

All four entries refer to the same physical device.  `RSPduoReceiver` opens
**mode=DT** (Dual Tuner) directly.  The `rspduo-dual-independent-tuners`
branch adds independent per-channel tuning to this mode - the driver
lazily splits both tuners when a parameter is set on channel 1.

If the `mode=DT` entry is missing, the sdrplay service may not be running
or the SoapySDRPlay3 plugin is not installed.  Rebuild from source (step 3).

---

## 5. Time synchronisation

See [Nodes and servers - time synchronisation](../README.md#nodes-and-servers--time-synchronisation)
in the main README.  This step is important for RSPduo setups.

---

## 6. Beagle installation

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/dpkingston/beagle_rdf.git
cd beagle_rdf
python3 -m venv env
env/bin/pip install -e .
```

### Expose system SoapySDR bindings to the venv

The `python3-soapysdr` package installs into the system Python
(`/usr/lib/python3/dist-packages/`), which the venv does not see by default.
Add a `.pth` file to make it visible:

```bash
SITE=$(env/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "/usr/lib/python3/dist-packages" > "$SITE/soapysdr-system.pth"
```

Verify:
```bash
env/bin/python -c "import SoapySDR; print(SoapySDR.__file__)"
# Should print: /usr/lib/python3/dist-packages/SoapySDR.py
```

---

## 7. End-to-end test

### Verify FM sync detection

```bash
cd ~/src/beagle_rdf
# Using the node config (recommended -- uses the same receiver and settings as production):
env/bin/python scripts/verify_sync.py \
    --config config/node.yaml --duration 30

# Or without a config file (quick check with a specific frequency and gain):
env/bin/python scripts/verify_sync.py \
    --device "driver=sdrplay" --freq 94.9e6 --gain auto --duration 30
```

Expected output (Pi 5, outdoor discone, sync_period_ms=7.0, sync_lna_state=6):
```
Tuned to 99.9 MHz  gain=30 dB  rate=2.000 MSps  [rspduo]
Pipeline: 8* decimation -> 250 kHz -> FM demod -> 19 kHz pilot detector
Running for 30 s  (Ctrl-C to stop early)

  Time    Events   Rate/s   CorPeak     Crystal     Power
   1.0        79     78.4    0.7010      -5.6 ppm    -33.7 dBFS
   2.0       224    142.7    0.7037      -5.3 ppm    -33.8 dBFS
  ...
  30.0      4221    142.7    0.7035      -4.5 ppm    -33.8 dBFS

Total: 4221 sync events in 31.1 s (135.8/s)
Crystal drift: -5.2 ppm at t~10 s -> -6.1 ppm at end  (drift=0.9 ppm  OK)
OK: Pilot detection looks good
```

- **Rate ~143/s** at `sync_period_ms: 7.0` (one SyncEvent per 7 ms)
- **CorPeak > 0.65** - FM pilot lock confirmed; drop below 0.4 indicates antenna or LNA issue
- **Crystal < +/-50 ppm** - normal for RSPduo TCXO
- **Power -10 to -40 dBFS** - adjust `sync_lna_state` if outside this range:
  - Too high (> -10 dBFS): increase `sync_lna_state` (more attenuation)
  - Too low (< -40 dBFS): decrease `sync_lna_state` (less attenuation)

### Check target channel

```bash
env/bin/python scripts/check_target.py \
    --freq 462.5625e6 --gain 30 \
    --device "driver=sdrplay" --channel 1 --freq-offset 0 --rate 2000000 \
    --duration 30
```

Key your radio a few times during the run.  The script reports recommended
`onset_db` / `offset_db` values for `config/node.yaml`.

---

## 8. node.yaml configuration

```yaml
sdr_mode: "rspduo"
rspduo:
  sample_rate_hz: 2000000          # 2 MHz max in dual-tuner mode
  sync_gain_db: "auto"             # AGC works well for FM sync
  sync_lna_state: 6                # tune based on FM power in verify_sync.py
  target_gain_db: 30               # adjust based on check_target.py calibration
  target_lna_state: 0              # max LNA gain for weak LMR signals
  master_device_args: "driver=sdrplay"
  slave_device_args: null          # null = same device as master (correct for single RSPduo)
  buffer_size: 65536               # ~33 ms per buffer pair at 2 MSPS
  pipeline_offset_ns: 0            # calibrate if mixing RSPduo + freq_hop nodes (see below)
```

Use the serial number to select a specific RSPduo if multiple SDRplay devices
are present:
```yaml
  master_device_args: "driver=sdrplay,serial=240504F534"
```

**Optional: antenna port selection.** The driver auto-selects antenna ports
(Antenna B for sync/Tuner 1, Antenna C for target/Tuner 2).  Override only if
the wrong physical port is selected:
```yaml
  sync_antenna: "Antenna B"        # null = driver default (Antenna B at VHF/UHF)
  target_antenna: "Antenna C"      # null = driver default (Antenna C)
```
Run `SoapySDRUtil --probe="driver=sdrplay,mode=DT"` to see the exact antenna
names your driver version uses.

**`sync_lna_state` starting points:**
- Outdoor directional or discone antenna: start at 6, adjust based on FM power
- Indoor mag-mount or short whip: start at 3-4
- State 9 = maximum attenuation; state 0 = minimum attenuation (most gain)

**Remote configuration (optional):** If you plan to manage this node's config
from the server instead of editing `node.yaml` locally, set up a
`bootstrap.yaml` file.  See
[Node Quick Start -- Option B](../README.md#option-b-remote-config-bootstrap)
in the README for instructions.

---

## 9. Run the node

With the config in place, start the node and verify it connects to the server
and begins reporting events.

**Using a local config file:**

```bash
cd ~/src/beagle_rdf
env/bin/python -m beagle_node --config config/node.yaml
```

**Using remote config (bootstrap):**

```bash
env/bin/python -m beagle_node --bootstrap config/bootstrap.yaml
```

You should see log output showing:
- `Beagle node starting` with the correct `node_id`
- `Opening RSPduo (DT mode)` with the expected sync and target frequencies
- Periodic sync events (no repeated `FM pilot quality below threshold` warnings)
- `Measurement: onset ...` / `Measurement: offset ...` lines when a carrier is
  detected on the target channel

Press Ctrl-C to stop.  For production deployment as a systemd service, see
[Production deployment](../README.md#production-deployment-systemd) in the
README.

---

## 10. `pipeline_offset_ns` calibration (optional)

This step is **not required** to get the system running.  Leave
`pipeline_offset_ns: 0` for initial deployment and return to this once the
node is operational and producing fixes.  The system works well without it --
calibration only improves accuracy on the sync_delta fallback path.

`pipeline_offset_ns` corrects a systematic bias in the **sync_delta** timing
path -- it is subtracted from `sync_delta_ns` before each event is submitted.

**Why this is optional:** the xcorr primary TDOA path (~80% of pairs) is
completely independent of `pipeline_offset_ns`.  Cross-correlation measures the
physical carrier arrival time difference directly from the IQ snippets, with no
reference to `sync_delta_ns` or wall-clock timestamps.  `pipeline_offset_ns`
only affects the sync_delta fallback path (~20% of pairs, used when xcorr SNR
is too low or snippets are absent).

**RSPduo** (`rspduo` mode): the dominant hardware effect is the TDM ADC
interleave - ch1 (target) lags ch0 (sync) by ~250 ns (0.5 ADC samples at 2
MSPS).  Since all RSPduo nodes have the same bias, it cancels in the
TDOA subtraction for RSPduo-vs-RSPduo pairs.  Leave `pipeline_offset_ns: 0`
for all-RSPduo deployments; the residual 250 ns contributes ~75 m of position
error on the fallback path, negligible compared to NTP uncertainty.

**RTL-SDR freq_hop**: because sync and target blocks are captured sequentially,
`sync_delta_ns` includes a structural inter-block delay (settling + frequency-
switch time, typically 20-40 ms) that RSPduo nodes do not have.  This causes
a large systematic error in the sync_delta fallback path when pairing a
freq_hop node against an RSPduo node.  Calibrate `pipeline_offset_ns` in the
`freq_hop` config block using a co-located RSPduo reference node.

**To calibrate a freq_hop node (mixed RSPduo + freq_hop deployment):**
1. Co-locate the freq_hop node with a calibrated RSPduo reference node.
2. Run `scripts/colocated_pair_test.py --db --node-a <freq_hop> --node-b <rspduo> --since 20`.
3. Note the mean **sync_delta TDOA_AB** (labelled "sync_delta method" in the output).
   For co-located nodes the true TDOA is 0; the mean offset is the uncorrected
   structural inter-block delay.
4. Set `pipeline_offset_ns` in the `freq_hop` block to that value (positive = freq_hop
   node sync_delta is later than reference by that amount).

**To calibrate an RSPduo node empirically (sub-usec accuracy, optional):**
1. Co-locate the new node with a calibrated reference RSPduo node.
2. Run `scripts/colocated_pair_test.py --db --node-a <new> --node-b <ref> --since 20`.
3. Note the mean **sync_delta TDOA_AB** for onset events; xcorr TDOA is unaffected by
   `pipeline_offset_ns` and should already be near 0 for co-located RSPduo nodes.
4. Set `pipeline_offset_ns` to that value only if > +/-1 usec (otherwise leave at 0).

See `docs/test-results/colocated-pair-log.md` for example calibration results.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `--find` shows nothing | sdrplay service not running | `sudo systemctl start sdrplay` |
| `--find` shows no DT entry | SoapySDRPlay3 plugin not installed | Build from source (step 3) and restart sdrplay |
| Tuner 2 locked to Tuner 1's frequency | Driver not from `rspduo-dual-independent-tuners` branch | Rebuild from the correct branch (step 3) |
| Both dBFS readings identical | Same root cause | See above |
| sync_rate = 0 in verify_sync.py | Wrong sync frequency or antenna disconnected | Check `sync_signal` in node.yaml; check Antenna A connection |
| target channel noise floor only (-60 dBFS, no signal bump) | Ch1 frequency not applied | Ensure using `rspduo-dual-independent-tuners` branch |
| overflow errors | USB bandwidth insufficient | Use a USB 3.0 port; reduce `buffer_size` to 32768 |
| `sdrplay_api_Fail` opening second device | Attempted Master/Slave mode | Use DT mode (single device, two streams) - see section 8 |
| Only Single Tuner mode available | Firmware / API version mismatch | Check API version matches sdrplay service version |
| `Device::make() no match` or `Hash collision` on Debian 13 | SoapySDR 0.8.1-5 bug with dict kwargs in `Device.make()` | Use string-form device args (`'driver=sdrplay,mode=DT'`); Beagle already does this |
| `ModuleNotFoundError: No module named 'SoapySDR'` in venv | System `python3-soapysdr` not visible to venv | Install `python3-soapysdr` and add `.pth` file (step 5) |
| SDRplay installer hangs waiting for input over SSH | `more` pager needs a tty | Run installer in an interactive session: `ssh -t user@host`, or `screen`/`tmux` |

---

## Operational Notes

### DT mode with independent tuning

`RSPduoReceiver` opens the device in Dual Tuner (DT) mode with **two separate
single-channel streams** on the same SoapySDR device object.  The
`rspduo-dual-independent-tuners` branch adds a lazy-split mechanism:
both tuners start mirrored (same frequency/gain), and the driver splits
them into independent operation the first time a parameter is set on
channel 1.

```python
# Use string form - SoapySDR 0.8.1-5 (Debian 13) has a Device.make()
# bug with dict kwargs.  String form works on all versions.
dev = SoapySDR.Device('driver=sdrplay,mode=DT')
# Channel 0 = Tuner 1 (sync/FM)   - Antenna port A
# Channel 1 = Tuner 2 (target/LMR) - Antenna port C
dev.setFrequency(RX, 0, sync_freq)
dev.setFrequency(RX, 1, target_freq)   # triggers lazy split to independent tuners
sync_stream   = dev.setupStream(RX, CF32, [0])
target_stream = dev.setupStream(RX, CF32, [1])
dev.activateStream(sync_stream)
dev.activateStream(target_stream)
# Safety net: re-apply ch1 frequency after both streams are active.
dev.setFrequency(RX, 1, target_freq)
```

**Why two streams instead of one?** SoapySDRPlay3 does not support a
multi-channel `setupStream(RX, CF32, [0, 1])` call in DT mode.  Each
channel gets its own independent stream handle.

### API notes: Init and Tuner B parameters

`sdrplay_api_Init()` fires during the **first** `activateStream()` call
(the sync stream, ch0).  In older SoapySDRPlay3 builds, Init would
program Rf_B = Rf_A regardless of what was set on ch1 beforehand.

The `rspduo-dual-independent-tuners` branch handles this internally:
`activateStream()` detects any Tuner B parameters that differ from the
Init defaults and re-applies them via `sdrplay_api_Update()`.  Our code
also re-applies ch1 frequency and gains after both `activateStream()`
calls as a safety net.

**Gain reset at Init:** At Init time `_streams[1]` may be `NULL`, so
Tuner B gain updates issued before `activateStream(target)` may not be
acknowledged.  Re-apply gains for both channels after both
`activateStream` calls.  `RSPduoReceiver._apply_gains()` handles this.

**Why Master/Slave mode does not work:** The SDRplay API library enforces
one selected device per `sdrplay_api_Open()` handle.  SoapySDRPlay3 uses
a singleton for the API handle, so `sdrplay_api_SelectDevice()` for the
Slave returns `sdrplay_api_Fail` immediately.  Use DT mode (single device
object, two streams) instead.

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../LICENSE).
