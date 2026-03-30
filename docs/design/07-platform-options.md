# Beagle Platform Options

## Requirements Summary

Before evaluating specific hardware, here are the key requirements for each role:

### Node requirements

| Requirement | Detail |
|-------------|--------|
| USB 2.0 (stable) | RTL-SDR dongles need reliable, low-jitter USB 2.0 transfer |
| USB 3.0 (for RSPduo) | RSPduo uses USB 2.0 protocol but a USB 3.0 port provides better power and controller isolation |
| CPU - sustained DSP | scipy FIR filter + FM demod + cross-correlation at 2 MSPS; roughly 100-200 million FP ops/s sustained |
| RAM | OS + Python runtime + pipeline state: 256 MB typical, 512 MB peak; 1 GB minimum |
| GPIO or serial UART | GPS module connection (NMEA over UART, or USB-serial adapter) |
| Thermal stability | Pipeline runs continuously; thermal throttling degrades timing accuracy |
| Network | Ethernet strongly preferred over Wi-Fi for reliability |
| OS | Debian or Ubuntu arm64/armhf with kernel >= 5.15; SoapySDR ecosystem support required |

### Server requirements

| Requirement | Detail |
|-------------|--------|
| CPU - bursty | Event ingestion and Folium map rendering; very low average load |
| RAM | FastAPI + SQLite + Folium map data; 512 MB typical, 2 GB comfortable |
| Storage | SQLite database; grows ~1 MB/day at typical event rates; SSD preferred |
| Network | Reachable by all nodes over the internet or LAN |
| OS | Linux x86-64 or arm64; standard Python 3.11+ |

---

## Node Platform Options

### Raspberry Pi 5 * Recommended

- **SoC:** Broadcom BCM2712 - 4x ARM Cortex-A76 @ 2.4 GHz
- **RAM:** 4 GB or 8 GB LPDDR4X (get 4 GB minimum)
- **USB:** 2x USB 3.0 + 2x USB 2.0 (separate controllers per pair)
- **Connectivity:** Gigabit Ethernet, Wi-Fi 5, Bluetooth 5.0
- **GPIO:** 40-pin header with UART0 for GPS
- **Price:** ~$60 (4 GB), ~$80 (8 GB)

**Why it works well:**  The A76 cores are roughly 2-3x faster than the Pi 4's
A72 for the scipy-heavy DSP pipeline.  Comfortable CPU headroom leaves room for
OS overhead, logging, and Python's GC without risk of buffer underruns.  The USB
3.0 controller is stable and well-tested with RTL-SDR and RSPduo.

**Thermal:**  The Pi 5 runs warm under sustained load.  The Raspberry Pi Active
Cooler (official accessory, ~$5) or a quality third-party heatsink+fan keeps
it well within thermal limits indefinitely.  The case fan is strongly recommended
for 24/7 node operation.  Without cooling, the BCM2712 will throttle to 1.5 GHz
at ~80 degC - still adequate but leaves less headroom.

**GPS:**  Use UART0 (`/dev/ttyAMA0`) directly on the 40-pin header.  Enable
`enable_uart=1` and `dtoverlay=miniuart-bt` in `/boot/firmware/config.txt` if
Bluetooth is not needed, freeing the PL011 UART for GPS.

**RSPduo note:**  Plug the RSPduo into a USB 3.0 port.  Avoid USB hubs - the
RSPduo's sustained 16 MB/s (2x 2 MSPS x 4 bytes x 2 channels) benefits from a
direct port connection.

---

### Raspberry Pi 4 Model B

- **SoC:** Broadcom BCM2711 - 4x ARM Cortex-A72 @ 1.5-1.8 GHz
- **RAM:** 2 GB, 4 GB, or 8 GB LPDDR4
- **USB:** 2x USB 3.0 + 2x USB 2.0 (via VL805 hub chip on shared bus)
- **Price:** ~$35 (2 GB) to $75 (8 GB); availability has been variable

**Works, but tighter margins:**  The A72 cores are adequate for the pipeline but
the CPU will run at 60-80% utilisation vs. 30-40% on the Pi 5.  The USB
architecture is a notable weakness: both USB 3.0 ports and both USB 2.0 ports
share one VL805 USB controller via PCIe x 1, so the RSPduo and a GPS USB adapter
compete for the same controller bandwidth.  For RSPduo deployments, the Pi 5 is
strongly preferred.

**Thermal:**  The A72 cores throttle more aggressively than the A76.  A heatsink
is mandatory for 24/7 operation; a fan is strongly recommended.

**Good for:**  `freq_hop` mode with a single RTL-SDR (low USB bandwidth, lower CPU).

---

### Orange Pi 5

- **SoC:** Rockchip RK3588S - 4x Cortex-A76 @ 2.4 GHz + 4x Cortex-A55 @ 1.8 GHz
- **RAM:** 4, 8, or 16 GB LPDDR5
- **USB:** 1x USB 3.0 (A or C depending on variant) + 2x USB 2.0
- **Connectivity:** Gigabit Ethernet
- **Price:** ~$60-100 depending on RAM and retailer

**Excellent compute, more setup:**  The RK3588S big.LITTLE arrangement (4 fast
A76 + 4 efficient A55) gives substantially more compute than the Pi 5 for
CPU-intensive tasks.  However, as of early 2026 the software ecosystem requires
more manual work:

- SoapySDR, SoapySDRPlay3, and their dependencies typically need to be compiled
  from source on Debian Bookworm for arm64.
- The Orange Pi 5 USB controller (Synopsys DesignWare) is generally solid but
  has seen occasional driver quirks with high-throughput USB audio/SDR devices;
  validate before deployment.
- The GPIO UART pinout differs from the Pi 40-pin standard; check the Orange Pi 5
  schematic for the correct pins.

**Good for:**  Deployments where DSP compute is the bottleneck (e.g. multiple
target channels, dual-station cross-checking, high sync event rates).

---

### Orange Pi 5 Plus / Orange Pi 5 Pro

- **SoC:** Rockchip RK3588 (full, not S variant) - same core arrangement as Orange Pi 5
- **RAM:** Up to 32 GB (Plus); up to 16 GB (Pro)
- **USB:** More ports (4x USB 3.0 on the Plus via hub)
- **Extra connectivity:** dual Ethernet on the Plus; M.2 for NVMe SSD

**Same caveats as Orange Pi 5** regarding software ecosystem.  The extra USB
ports on the Plus (via a hub chip) are useful if running a GPS USB adapter
alongside an RSPduo, though the hub adds a small latency overhead.

---

### Radxa Rock 5B

- **SoC:** Rockchip RK3588
- **RAM:** 4-16 GB LPDDR5
- **USB:** Multiple USB 3.0 via separate controllers; better USB topology than Orange Pi
- **Price:** ~$80-140

**Better USB topology than Orange Pi 5:**  The Rock 5B provides separate USB 3.0
controllers for different ports, which reduces interference between the RSPduo and
other USB devices.  Debian/Ubuntu images are maintained by Radxa.  SoapySDR
compilation required.

---

### ODROID N2+

- **SoC:** Amlogic S922X - 4x Cortex-A73 @ 2.4 GHz + 2x Cortex-A53 @ 2.0 GHz
- **RAM:** 4 GB LPDDR4
- **USB:** 4x USB 3.0 (separate controller from USB 2.0)
- **Price:** ~$80

**Solid choice:** Hardkernel (the manufacturer) provides well-maintained
Debian/Ubuntu builds and excellent long-term kernel support.  The A73 cores are
faster than the Pi 4's A72 and comparable to A76 for the scipy workload.  Good
USB 3.0 implementation.  The major drawback is that the Amlogic ecosystem is less
common in the SDR community - verify SoapySDR compatibility before committing.

---

### What to avoid

| Device | Issue |
|--------|-------|
| Raspberry Pi Zero 2 W | Single-core equivalent for most workloads; USB via dongle only; insufficient for RSPduo |
| Raspberry Pi 3 B/B+ | A53 cores (too slow for sustained DSP at 2 MSPS); USB 2.0 only |
| Orange Pi Zero / Zero 2 | Limited USB, insufficient RAM |
| Any device with USB 2.0 only | Works for RTL-SDR (`freq_hop`); RSPduo strongly prefers USB 3.0 port |
| Devices without active community | Kernel support gaps make SoapySDR compilation difficult |

---

### Node platform comparison

| Platform | CPU perf | USB for RSPduo | GPS UART | Ecosystem | Price |
|----------|----------|----------------|----------|-----------|-------|
| **Pi 5 (4GB)** | ***** | ***** | ***** | ***** | $60 |
| Pi 4 (4GB) | *** | *** | ***** | ***** | $55 |
| Orange Pi 5 (8GB) | ***** | **** | *** | *** | $75 |
| Rock 5B (8GB) | ***** | ***** | *** | **** | $100 |
| ODROID N2+ | **** | ***** | **** | **** | $80 |

**Bottom line for nodes:** Raspberry Pi 5 (4 GB) is the clear recommendation --
it provides ample compute, excellent USB, the most tested SDR/SoapySDR ecosystem,
and straightforward GPS UART integration.  Use Orange Pi 5 or Rock 5B if you need
more compute headroom or extra RAM for future features.

---

## Server Platform Options

The aggregation server is **not compute-intensive**.  At typical event rates (a few
events per second across all nodes), the bottleneck is network I/O and SQLite writes,
not CPU.  Folium map generation (Python-side HTML) is the most CPU-heavy operation but
runs only when a fix is computed.  A $5/month VPS handles a small network comfortably.

### Self-hosted: Raspberry Pi 5 (8 GB)

- Runs the server alongside other services
- FastAPI + uvicorn + SQLite uses ~150 MB RAM at rest
- Adequate for networks of up to ~10 nodes and hundreds of fixes
- Requires a static IP or dynamic DNS for nodes to reach it
- Advantage: no ongoing hosting cost; server stays local

### Self-hosted: Intel/AMD mini PC

Examples: Intel N100 mini PCs (Beelink Mini S12, Minisforum UN100, etc.) at ~$150-200.

- 4x Efficient Core Intel N100 @ up to 3.4 GHz - far more than needed
- 8-16 GB DDR5, 256 GB NVMe SSD
- x86-64 means pre-built packages for everything (no compilation from source)
- Runs standard Debian/Ubuntu
- Fanless or near-silent options available

**Good choice for a permanent self-hosted server** - runs alongside other home
services, easy to maintain.

### VPS / Cloud

The server is stateless except for SQLite (a single file on disk) and is trivial
to migrate between providers.  Requirements: 1 vCPU, 1 GB RAM, 10 GB disk, public
IPv4.  Recommended options as of early 2026:

| Provider | Instance | Spec | Price/mo | Notes |
|----------|----------|------|----------|-------|
| **Hetzner** | CAX21 | 4 vCPU arm64, 8 GB RAM, 80 GB SSD | ~EUR4 | Best value; arm64 packages available |
| **Hetzner** | CX22 | 2 vCPU x86, 4 GB RAM, 40 GB SSD | ~EUR4 | Good value; x86 standard packages |
| **DigitalOcean** | Basic | 1 vCPU, 1 GB RAM, 25 GB SSD | $6 | Easy management UI |
| **Linode (Akamai)** | Nanode | 1 vCPU, 1 GB RAM, 25 GB SSD | $5 | Good network |
| **Vultr** | Regular | 1 vCPU, 1 GB RAM, 25 GB SSD | $6 | Multiple locations |

A $4-6/month Hetzner or DigitalOcean instance is more than sufficient for a
city-scale deployment with 5-20 nodes.

**Networking:** Nodes push events outbound to the server - they do not receive
inbound connections.  Nodes behind NAT or firewalled networks work fine.  Only the
server needs a public IP.

**TLS:** Use Caddy or nginx as a reverse proxy in front of uvicorn to handle TLS
termination via Let's Encrypt.  The node's `reporter.server_url` should be
`https://your-domain.example.com`.

---

## GPS Module Options (for Nodes)

GPS is used on the node for `onset_time_ns` (rough absolute timestamp for event
association across nodes) and, in `two_sdr` mode, for GPS 1PPS injection.
In `freq_hop` and `rspduo` modes, GPS discipline is optional - NTP with
`clock_source: ntp` works for event association (nodes only need to agree on
absolute time to within +/-200 ms).

| Module | Interface | 1PPS | Price | Notes |
|--------|-----------|------|-------|-------|
| u-blox NEO-M8N | UART + USB | Yes | ~$20-30 | Widely used; well-supported by chrony and gpsd |
| u-blox NEO-M9N | UART + USB | Yes | ~$30-40 | Newer; better multipath rejection |
| GlobalSat BU-353S4 | USB-serial | No | ~$40 | USB dongle; no 1PPS (NTP discipline only) |
| Adafruit Ultimate GPS | UART | Yes | ~$30 | Easy to work with; known good |
| Any bare u-blox module | UART | Yes | ~$10-15 | Requires soldering or breakout board |

For `rspduo` and `freq_hop` modes, a **USB GPS dongle without 1PPS** (like the
BU-353S4) is sufficient for NTP discipline.  For `two_sdr` mode requiring GPS
1PPS injection into the SDR antenna inputs, a module with a 1PPS TTL output is
required.

Chrony configuration for GPS disciplining is covered in
[setup-rspduo-debian.md](../setup-rspduo-debian.md).

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../../LICENSE).
