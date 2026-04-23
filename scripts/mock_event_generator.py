#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Beagle Mock Event Generator
============================
Synthesises realistic CarrierEvent HTTP POSTs to a running beagle-server,
simulating radio conversations observed by multiple receiver nodes.

Each simulated transmission follows the realistic PTT pattern:

    [onset events] -> [carrier in progress N seconds] -> [offset events]
    -> [gap M seconds] -> [next onset events] -> ...

Both onset and offset events are computed from the same underlying geometry.
The offset events reuse each node's onset event_id (server upsert/amend path)
but carry a fresh sync_to_snippet_start_ns measurement and event_type="offset".

After the server's delivery buffer fires for each phase, the script polls for
the computed fix and reports position accuracy vs. the known true location.

Usage
-----
    python scripts/mock_event_generator.py \\
        --server http://localhost:8765 \\
        --scenario scripts/mock_scenario_seattle.yaml

    # Override noise model from the command line:
    python scripts/mock_event_generator.py \\
        --scenario scripts/mock_scenario_seattle.yaml \\
        --pilot-sigma-us 0.5     # GPS two_sdr mode
        --ntp-sigma-ms 1.0       # Stratum-1 reference

Error model physics
-------------------
sync_to_snippet_start_ns has two independent error sources:

1. FM pilot timing noise (pilot_sigma):
   Jitter in the FM stereo pilot cross-correlation peak that anchors each
   measurement to the shared clock reference.

2. Carrier edge detection jitter (edge_sigma):
   Uncertainty in identifying the exact carrier onset/offset sample within
   the IQ snippet.

Combined sync_delta error = sqrt(pilot_sigma^2 + edge_sigma^2).
TDOA error per pair = sqrt(2) * combined_sigma.
Position error (rough) ~= TDOA_error * c / sin(convergence_angle).

onset_time_ns error (NTP clock drift):
  Affects ONLY the T_sync grouping window, not TDOA.
  T_sync = onset_time_ns - sync_to_snippet_start_ns - dist(sync_tx, node) / c
  NTP noise appears equally on both terms, leaving TDOA unaffected.
  The server's correlation_window_s (default 0.2 s) must exceed 2* ntp_sigma.
"""

from __future__ import annotations

import argparse
import base64
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import yaml

_C_M_S = 299_792_458.0  # speed of light m/s
_POLL_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeDef:
    node_id: str
    latitude_deg: float
    longitude_deg: float


@dataclass
class SyncTxDef:
    station_id: str
    frequency_hz: float
    latitude_deg: float
    longitude_deg: float


@dataclass
class TargetDef:
    label: str
    latitude_deg: float
    longitude_deg: float
    channel_hz: float


@dataclass
class ErrorModel:
    """
    Noise parameters for the mock event generator.

    pilot_timing_sigma_ns : float
        1-sigma FM pilot cross-correlation timing noise (nanoseconds).
        Added to sync_to_snippet_start_ns.  See module docstring for presets.

    edge_timing_sigma_ns : float
        1-sigma carrier edge detection jitter (nanoseconds).
        Independent noise on sync_to_snippet_start_ns representing uncertainty in
        identifying the exact onset/offset sample.  Combined with
        pilot_timing_sigma_ns: total sync_delta sigma = sqrt(pilot^2 + edge^2).
        Defaults calibrated so pair TDOA sigma ~ 2 usec (colocated test).

    ntp_sigma_ns : float
        1-sigma NTP clock error on onset_time_ns (nanoseconds).
        Affects T_sync grouping ONLY. Does not affect TDOA accuracy
        because sync_to_snippet_start_ns is a within-node sample-clock measurement.

    corr_peak_mean / corr_peak_sigma : float
        FM pilot cross-correlation quality (0-1). Events below the
        server's min_corr_peak threshold (default 0.1) are rejected.

    delivery_latency_mean_ms / delivery_latency_sigma_ms : float
        Simulated one-way HTTP delivery latency per node.
        All events must arrive within the server's delivery_buffer_s window.
    """
    pilot_timing_sigma_ns: float = 1_000.0
    edge_timing_sigma_ns: float = 1_000.0
    ntp_sigma_ns: float = 20_000_000.0
    corr_peak_mean: float = 0.85
    corr_peak_sigma: float = 0.05
    delivery_latency_mean_ms: float = 150.0
    delivery_latency_sigma_ms: float = 50.0


@dataclass
class ScenarioDef:
    sync_transmitter: SyncTxDef
    nodes: list[NodeDef]
    targets: list[TargetDef]
    error_model: ErrorModel = field(default_factory=ErrorModel)
    transmissions_per_target: int = 3
    # PTT timing: each transmission = onset -> [duration] -> offset -> [gap] -> next onset
    transmission_duration_mean_s: float = 15.0
    transmission_duration_sigma_s: float = 3.0
    inter_transmission_gap_mean_s: float = 1.5
    inter_transmission_gap_sigma_s: float = 0.3
    rapid_keyup_interval_ms: float = 0.0  # set > 0 to test rapid re-key (overrides gap)


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> ScenarioDef:
    with open(path) as f:
        raw = yaml.safe_load(f)

    sync_tx = SyncTxDef(**raw["sync_transmitter"])
    nodes = [NodeDef(**n) for n in raw["nodes"]]

    # Strip legacy event_type field if present in old scenario files.
    targets = []
    for t in raw["targets"]:
        t_clean = {k: v for k, v in t.items() if k != "event_type"}
        targets.append(TargetDef(**t_clean))

    em_raw = raw.get("error_model", {})
    em = ErrorModel(**em_raw)

    # Backward compat: old transmission_interval_s maps to inter_transmission_gap_mean_s
    # when the new field is absent.
    gap_mean = raw.get("inter_transmission_gap_mean_s",
                        raw.get("transmission_interval_s", 1.5))

    return ScenarioDef(
        sync_transmitter=sync_tx,
        nodes=nodes,
        targets=targets,
        error_model=em,
        transmissions_per_target=raw.get("transmissions_per_target", 3),
        transmission_duration_mean_s=raw.get("transmission_duration_mean_s", 15.0),
        transmission_duration_sigma_s=raw.get("transmission_duration_sigma_s", 3.0),
        inter_transmission_gap_mean_s=gap_mean,
        inter_transmission_gap_sigma_s=raw.get("inter_transmission_gap_sigma_s", 0.3),
        rapid_keyup_interval_ms=raw.get("rapid_keyup_interval_ms", 0.0),
    )


# ---------------------------------------------------------------------------
# Event synthesis
# ---------------------------------------------------------------------------

@dataclass
class NodeEvent:
    """Synthesised event for one node, with diagnostics."""
    node: NodeDef
    event_payload: dict[str, Any]   # JSON-ready dict for HTTP POST
    true_sync_delta_ns: int
    true_onset_time_ns: int
    ntp_noise_ns: int
    pilot_noise_ns: int
    d_target_m: float
    d_sync_m: float
    delivery_delay_ms: float


def _make_synthetic_iq_snippet(
    n_samples: int = 640,
    event_type: str = "onset",
) -> str:
    """Generate a base64-encoded synthetic IQ snippet with a PA transition.

    The server uses sync_delta subtraction (not IQ cross-correlation) for
    TDOA computation, so the snippet content does not affect positioning
    accuracy.  The snippet just needs a valid PA transition so the server
    can verify the event is genuine.

    Transition placement matches the real node's carrier detector (detection
    anchored at the midpoint of the snippet for both event types).

    Encoded as interleaved int8 real/imag pairs.
    """
    buf = bytearray(n_samples * 2)  # 2 bytes per complex sample (I, Q)

    mid = n_samples // 2

    for i in range(n_samples):
        if event_type == "offset":
            is_carrier = i < mid
        else:
            is_carrier = i >= mid

        if is_carrier:
            phase = 2.0 * math.pi * 0.05 * i
            buf[2 * i] = int(100 * math.cos(phase)) & 0xFF
            buf[2 * i + 1] = int(100 * math.sin(phase)) & 0xFF
        else:
            buf[2 * i] = random.randint(0, 10)
            buf[2 * i + 1] = random.randint(0, 10)
    return base64.b64encode(bytes(buf)).decode("ascii")


_MOCK_SAMPLE_RATE_HZ = 64_000.0


def _make_event_payload(
    node: NodeDef,
    sync_tx: SyncTxDef,
    target: TargetDef,
    event_id: str,
    event_type: str,
    onset_time_ns: int,
    sync_to_snippet_start_ns: int,
    corr_peak: float,
) -> dict[str, Any]:
    return {
        "schema_version": "1.4",
        "event_id": event_id,
        "node_id": node.node_id,
        "node_location": {
            "latitude_deg": node.latitude_deg,
            "longitude_deg": node.longitude_deg,
        },
        "channel_frequency_hz": target.channel_hz,
        "sync_to_snippet_start_ns": sync_to_snippet_start_ns,
        "sync_transmitter": {
            "station_id": sync_tx.station_id,
            "frequency_hz": sync_tx.frequency_hz,
            "latitude_deg": sync_tx.latitude_deg,
            "longitude_deg": sync_tx.longitude_deg,
        },
        "sdr_mode": "freq_hop",
        "pps_anchored": False,
        "event_type": event_type,
        "onset_time_ns": onset_time_ns,
        "sync_corr_peak": round(corr_peak, 4),
        "node_software_version": "mock-1.0",
        "iq_snippet_b64": _make_synthetic_iq_snippet(event_type=event_type),
        "channel_sample_rate_hz": _MOCK_SAMPLE_RATE_HZ,
    }


def synthesise_onset(
    target: TargetDef,
    nodes: list[NodeDef],
    sync_tx: SyncTxDef,
    error_model: ErrorModel,
    t_tx_ns: int,
    rng: random.Random,
) -> list[NodeEvent]:
    """
    Synthesise one onset CarrierEvent per node for a single PTT press.

    Physics:
        T_sync_ns  = t_tx_ns - 500_000_000  (sync event 500 ms before TX)
        true_onset = t_tx_ns + dist(target, node) / c
        true_delta = true_onset - (T_sync_ns + dist(sync_tx, node) / c)
                   = 500_000_000 + (d_target - d_sync) / c

        Observed onset  = true_onset + NTP_noise        (clock error)
        Observed delta  = true_delta + pilot_noise       (FM timing error)
    """
    T_sync_ns = t_tx_ns - 500_000_000
    results: list[NodeEvent] = []

    for node in nodes:
        d_target_m = haversine_m(
            target.latitude_deg, target.longitude_deg,
            node.latitude_deg, node.longitude_deg,
        )
        d_sync_m = haversine_m(
            sync_tx.latitude_deg, sync_tx.longitude_deg,
            node.latitude_deg, node.longitude_deg,
        )

        prop_target_ns = d_target_m / _C_M_S * 1e9
        prop_sync_ns   = d_sync_m   / _C_M_S * 1e9

        true_onset_time_ns = int(t_tx_ns + prop_target_ns)
        true_sync_delta_ns = int(
            (t_tx_ns + prop_target_ns) - (T_sync_ns + prop_sync_ns)
        )

        ntp_noise_ns   = int(rng.gauss(0.0, error_model.ntp_sigma_ns))
        pilot_noise_ns = int(rng.gauss(0.0, error_model.pilot_timing_sigma_ns))
        edge_noise_ns  = int(rng.gauss(0.0, error_model.edge_timing_sigma_ns))
        onset_time_ns  = true_onset_time_ns + ntp_noise_ns
        sync_to_snippet_start_ns  = true_sync_delta_ns + pilot_noise_ns + edge_noise_ns

        corr_peak = max(0.15, min(1.0, rng.gauss(
            error_model.corr_peak_mean, error_model.corr_peak_sigma,
        )))
        delivery_ms = max(10.0, rng.gauss(
            error_model.delivery_latency_mean_ms,
            error_model.delivery_latency_sigma_ms,
        ))

        results.append(NodeEvent(
            node=node,
            event_payload=_make_event_payload(
                node, sync_tx, target,
                event_id=str(uuid.uuid4()),
                event_type="onset",
                onset_time_ns=onset_time_ns,
                sync_to_snippet_start_ns=sync_to_snippet_start_ns,
                corr_peak=corr_peak,
            ),
            true_sync_delta_ns=true_sync_delta_ns,
            true_onset_time_ns=true_onset_time_ns,
            ntp_noise_ns=ntp_noise_ns,
            pilot_noise_ns=pilot_noise_ns + edge_noise_ns,
            d_target_m=d_target_m,
            d_sync_m=d_sync_m,
            delivery_delay_ms=delivery_ms,
        ))

    return results


def synthesise_offset(
    target: TargetDef,
    onset_events: list[NodeEvent],
    sync_tx: SyncTxDef,
    error_model: ErrorModel,
    t_offset_ns: int,
    rng: random.Random,
) -> list[NodeEvent]:
    """
    Synthesise offset CarrierEvents for each node.

    The offset event for each node reuses its onset event_id, triggering the
    server's upsert/amend path.  event_type is "offset" and onset_time_ns is
    set to the offset wall-clock time so the server can compute duration.

    sync_to_snippet_start_ns is a fresh FM pilot measurement at the offset edge; it has
    the same geometric value as the onset (same positions) plus independent
    pilot timing noise.
    """
    T_sync_ns = t_offset_ns - 500_000_000
    results: list[NodeEvent] = []

    for onset_ne in onset_events:
        node = onset_ne.node
        # Geometry is identical - target, sync_tx, and nodes are stationary.
        d_target_m = onset_ne.d_target_m
        d_sync_m   = onset_ne.d_sync_m

        prop_target_ns = d_target_m / _C_M_S * 1e9
        prop_sync_ns   = d_sync_m   / _C_M_S * 1e9

        # The offset carrier edge arrives at the node at t_offset_ns + prop_target_ns.
        true_offset_time_ns = int(t_offset_ns + prop_target_ns)
        # sync_delta: time from FM sync event (500 ms before offset) to offset edge.
        # Simplifies to 500_000_000 + (d_target - d_sync) / c - same as onset.
        true_sync_delta_ns = int(
            (t_offset_ns + prop_target_ns) - (T_sync_ns + prop_sync_ns)
        )

        ntp_noise_ns   = int(rng.gauss(0.0, error_model.ntp_sigma_ns))
        pilot_noise_ns = int(rng.gauss(0.0, error_model.pilot_timing_sigma_ns))
        edge_noise_ns  = int(rng.gauss(0.0, error_model.edge_timing_sigma_ns))
        onset_time_ns  = true_offset_time_ns + ntp_noise_ns  # offset time reported as onset_time_ns
        sync_to_snippet_start_ns  = true_sync_delta_ns + pilot_noise_ns + edge_noise_ns

        corr_peak = max(0.15, min(1.0, rng.gauss(
            error_model.corr_peak_mean, error_model.corr_peak_sigma,
        )))
        delivery_ms = max(10.0, rng.gauss(
            error_model.delivery_latency_mean_ms,
            error_model.delivery_latency_sigma_ms,
        ))

        results.append(NodeEvent(
            node=node,
            event_payload=_make_event_payload(
                node, sync_tx, target,
                event_id=onset_ne.event_payload["event_id"],  # REUSE onset event_id
                event_type="offset",
                onset_time_ns=onset_time_ns,
                sync_to_snippet_start_ns=sync_to_snippet_start_ns,
                corr_peak=corr_peak,
            ),
            true_sync_delta_ns=true_sync_delta_ns,
            true_onset_time_ns=true_offset_time_ns,
            ntp_noise_ns=ntp_noise_ns,
            pilot_noise_ns=pilot_noise_ns + edge_noise_ns,
            d_target_m=d_target_m,
            d_sync_m=d_sync_m,
            delivery_delay_ms=delivery_ms,
        ))

    return results


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 72

def _hr() -> None:
    print(_SEP)

def _ns_to_us(ns: int | float) -> str:
    return f"{ns / 1_000:+.1f} us"

def _m_to_km(m: float) -> str:
    return f"{m / 1_000:.1f} km"

def _sign_str(ns: int) -> str:
    return f"+{ns / 1_000:.2f}" if ns >= 0 else f"{ns / 1_000:.2f}"


def print_scenario_summary(scenario: ScenarioDef) -> None:
    _hr()
    print("Beagle Mock Event Generator")
    _hr()
    em = scenario.error_model
    sigma_pilot = em.pilot_timing_sigma_ns
    sigma_edge  = em.edge_timing_sigma_ns
    sigma_combined = math.sqrt(sigma_pilot ** 2 + sigma_edge ** 2)
    tdoa_sigma  = sigma_combined * math.sqrt(2)
    pos_rough_m = tdoa_sigma * _C_M_S / 1e9

    print(f"  Sync TX:   {scenario.sync_transmitter.station_id} "
          f"({scenario.sync_transmitter.latitude_deg:.4f}degN, "
          f"{scenario.sync_transmitter.longitude_deg:.4f}degW)")
    print(f"  Nodes:     {len(scenario.nodes)}  "
          f"({', '.join(n.node_id for n in scenario.nodes)})")
    print(f"  Targets:   {len(scenario.targets)}")
    print()
    print("  Error model (1-sigma):")
    print(f"    FM pilot timing:   +/-{sigma_pilot:,.0f} ns")
    print(f"    Edge detection:    +/-{sigma_edge:,.0f} ns")
    print(f"    Combined delta:    +/-{sigma_combined:,.0f} ns  -> TDOA accuracy +/-{tdoa_sigma:,.0f} ns per pair")
    print(f"                         rough position uncertainty ~{pos_rough_m:,.0f} m")
    print(f"    NTP clock drift:   +/-{em.ntp_sigma_ns / 1e6:.1f} ms  -> onset_time_ns (grouping only; no TDOA impact)")
    print(f"    Delivery latency:  {em.delivery_latency_mean_ms:.0f} +/- {em.delivery_latency_sigma_ms:.0f} ms  (simulated HTTP)")
    print()
    print("  PTT transmission pattern:")
    print(f"    Transmissions per target:  {scenario.transmissions_per_target}")
    print(f"    Carrier hold time:         {scenario.transmission_duration_mean_s:.1f} +/- "
          f"{scenario.transmission_duration_sigma_s:.1f} s  (onset -> offset)")
    if scenario.rapid_keyup_interval_ms > 0:
        print(f"    Inter-TX gap:              {scenario.rapid_keyup_interval_ms:.0f} ms  "
              f"(rapid key-up - should produce separate T_sync buckets)")
    else:
        print(f"    Inter-TX gap:              {scenario.inter_transmission_gap_mean_s:.1f} +/- "
              f"{scenario.inter_transmission_gap_sigma_s:.1f} s  (offset -> next onset)")
    _hr()


def print_transmission_header(
    tx_num: int,
    total: int,
    target: TargetDef,
    t_tx_ns: int,
    event_type: str,
    duration_s: float | None = None,
) -> None:
    t_wall = t_tx_ns / 1e9
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t_wall))
    phase = event_type.upper()
    print()
    _hr()
    print(f"  TRANSMISSION {tx_num}/{total}  [{phase}]  -  {target.label}")
    print(f"    True position:  {target.latitude_deg:.6f}degN, {target.longitude_deg:.6f}degW")
    print(f"    Channel:        {target.channel_hz / 1e6:.3f} MHz")
    print(f"    Wall clock:     {ts}")
    if duration_s is not None:
        print(f"    Duration:       ~{duration_s:.1f} s")
    _hr()


def print_node_table(node_events: list[NodeEvent]) -> None:
    col = [22, 13, 13, 17, 13, 12]
    hdr = ["Node", "d_target", "d_sync", "true_delta", "NTP noise", "delta noise"]
    sep = "+".join("-" * (c + 2) for c in col)

    def row(vals: list[str]) -> str:
        return "|" + "|".join(f" {v:<{col[i]}} " for i, v in enumerate(vals)) + "|"

    print(row(hdr))
    print(sep)
    for ne in node_events:
        print(row([
            ne.node.node_id,
            f"{ne.d_target_m:,.0f} m",
            f"{ne.d_sync_m:,.0f} m",
            f"{ne.true_sync_delta_ns / 1_000:,.1f} us",
            _ns_to_us(ne.ntp_noise_ns),
            _ns_to_us(ne.pilot_noise_ns),
        ]))
    print()


def print_tdoa_pairs(node_events: list[NodeEvent]) -> None:
    """Print true geometric TDOA between each node pair."""
    print("  True geometric TDOA (before noise, after path correction):")
    for i in range(len(node_events)):
        for j in range(i + 1, len(node_events)):
            ne_i, ne_j = node_events[i], node_events[j]
            true_tdoa_ns = (ne_i.d_target_m - ne_j.d_target_m) / _C_M_S * 1e9
            obs_tdoa_ns = true_tdoa_ns + ne_i.pilot_noise_ns - ne_j.pilot_noise_ns
            print(f"    {ne_i.node.node_id} <-> {ne_j.node.node_id}:  "
                  f"true {_sign_str(int(true_tdoa_ns))} us  |  "
                  f"noisy {_sign_str(int(obs_tdoa_ns))} us  "
                  f"(delta {_ns_to_us(int(obs_tdoa_ns - true_tdoa_ns))})")
    print()


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def send_heartbeats(
    client: httpx.Client,
    server_url: str,
    nodes: list[NodeDef],
    label: str = "heartbeats",
) -> None:
    """Send a heartbeat for each node so they appear on the map."""
    print(f"  Sending {label}:")
    for node in nodes:
        body = {
            "node_id": node.node_id,
            "latitude_deg": node.latitude_deg,
            "longitude_deg": node.longitude_deg,
            "sdr_mode": "freq_hop",
            "software_version": "mock-1.0",
        }
        try:
            resp = client.post(
                f"{server_url}/api/v1/heartbeat",
                json=body,
                timeout=5.0,
            )
            ok = resp.status_code == 200
            mark = "OK  " if ok else "FAIL"
            print(f"    [{mark}] {node.node_id:<20}  {resp.status_code}")
        except Exception as exc:
            print(f"    [FAIL] {node.node_id:<20}  ERROR: {exc}")


def send_events(
    client: httpx.Client,
    server_url: str,
    auth_token: str,
    node_events: list[NodeEvent],
    label: str = "events",
) -> bool:
    """POST each event to the server. Returns True if all succeeded."""
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    print(f"  Sending {label}:")
    all_ok = True
    for ne in node_events:
        t0 = time.monotonic()
        try:
            resp = client.post(
                f"{server_url}/api/v1/events",
                json=ne.event_payload,
                headers=headers,
                timeout=10.0,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            status_str = f"{resp.status_code} {'Created' if resp.status_code == 201 else resp.text[:40]}"
            ok = resp.status_code == 201
            if not ok:
                all_ok = False
            mark = "OK  " if ok else "FAIL"
            print(f"    [{mark}] {ne.node.node_id:<20}  {status_str:<20}  ({elapsed_ms:.0f} ms)")
        except httpx.ConnectError:
            print(f"    [FAIL] {ne.node.node_id:<20}  CONNECTION REFUSED - is the server running at {server_url}?")
            all_ok = False
        except Exception as exc:
            print(f"    [FAIL] {ne.node.node_id:<20}  ERROR: {exc}")
            all_ok = False

    return all_ok


def poll_for_fix(
    client: httpx.Client,
    server_url: str,
    t_tx_ns: int,
    target: TargetDef,
    event_type: str,
    delivery_buffer_s: float,
    max_wait_s: float = 20.0,
) -> dict[str, Any] | None:
    """
    Wait for the server to compute a fix for this transmission, then return it.

    Matching strategy: find the fix whose onset_time_ns is closest to t_tx_ns
    within a +/-5 s window and whose event_type matches.
    """
    deadline = time.monotonic() + delivery_buffer_s + max_wait_s
    label = f"{event_type} fix (buffer: {delivery_buffer_s:.1f}s)"
    print(f"\n  Waiting for {label} ...", end="", flush=True)

    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        print(".", end="", flush=True)
        try:
            resp = client.get(f"{server_url}/api/v1/fixes?limit=20&max_age_s=300", timeout=5.0)
            if resp.status_code != 200:
                continue
            fixes = resp.json()
        except Exception:
            continue

        best = None
        best_dt = 5_000_000_000  # 5 s in nanoseconds
        for fix in fixes:
            if fix.get("channel_hz") != target.channel_hz:
                continue
            if fix.get("event_type") != event_type:
                continue
            dt = abs(fix.get("onset_time_ns", 0) - t_tx_ns)
            if dt < best_dt:
                best_dt = dt
                best = fix

        if best is not None:
            print()
            return best

    print()
    return None


def print_fix_result(
    fix: dict[str, Any] | None,
    target: TargetDef,
    event_type: str,
) -> None:
    label = f"{event_type.upper()} FIX"
    if fix is None:
        print(f"  [TIMEOUT] {label}: no fix received within wait window.")
        print("    Check: server delivery_buffer_s, node count (need >=2), server logs.")
        return

    fix_lat = fix["latitude_deg"]
    fix_lon = fix["longitude_deg"]
    error_m = haversine_m(target.latitude_deg, target.longitude_deg, fix_lat, fix_lon)
    residual_ns = fix.get("residual_ns", float("nan"))
    node_count = fix.get("node_count", "?")
    nodes_list = fix.get("nodes", [])
    fix_id = fix.get("id", "?")

    quality = "good" if residual_ns < 5_000 else ("marginal" if residual_ns < 20_000 else "poor")

    print(f"  [OK] {label}  (fix #{fix_id})")
    print(f"    Position:   {fix_lat:.6f}degN, {fix_lon:.6f}degW")
    print(f"    Error:      {error_m:,.0f} m from true target")
    print(f"    Residual:   {residual_ns:,.0f} ns RMS  ({quality})")
    print(f"    Nodes:      {node_count}  ({', '.join(nodes_list)})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Beagle mock event generator - synthesises PTT onset+offset events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Error model presets (--pilot-sigma-us):
  20.0   Uncalibrated RTL-SDR crystal (100 ppm * 200 ms window)
   1.0   RTL-SDR TCXO + FM pilot phase calibration  [default]
   0.5   two_sdr mode with GPS 1PPS injection
   0.1   Theoretical lower bound (GPS 1PPS + TCXO, high SNR)

Carrier edge detection jitter (--edge-sigma-us):
   1.0   Default (calibrated from colocated pair TDOA ~ +/-2 usec)
   0.5   High-SNR carrier with sharp onset
""",
    )
    p.add_argument("--server", default="http://localhost:8765",
                   help="Aggregation server base URL (default: http://localhost:8765)")
    p.add_argument("--scenario", required=True,
                   help="Path to scenario YAML file")
    p.add_argument("--auth-token", default="",
                   help="Bearer token if server auth is enabled")
    p.add_argument("--delivery-buffer-s", type=float, default=10.0,
                   help="Server delivery_buffer_s - wait before polling for fix (default: 10.0)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible noise (default: random)")
    p.add_argument("--no-offset", action="store_true",
                   help="Send onset events only (skip offset phase)")

    noise = p.add_argument_group("noise overrides (override scenario YAML)")
    noise.add_argument("--pilot-sigma-us", type=float, default=None,
                       help="FM pilot timing 1-sigma (us). Controls TDOA and position accuracy.")
    noise.add_argument("--edge-sigma-us", type=float, default=None,
                       help="Carrier edge detection jitter 1-sigma (us). Default 2.0 (colocated pair test).")
    noise.add_argument("--ntp-sigma-ms", type=float, default=None,
                       help="NTP clock 1-sigma (ms). Controls onset_time_ns (grouping only).")

    tx = p.add_argument_group("transmission overrides (override scenario YAML)")
    tx.add_argument("--transmissions", type=int, default=None,
                    help="Override transmissions_per_target from scenario")
    tx.add_argument("--duration-mean-s", type=float, default=None,
                    help="Override transmission_duration_mean_s")
    tx.add_argument("--gap-mean-s", type=float, default=None,
                    help="Override inter_transmission_gap_mean_s")
    tx.add_argument("--conversation", action="store_true",
                    help="Interleave transmissions between targets (A->B->A->B ...) "
                         "to simulate a two-radio conversation.  Requires >=2 targets "
                         "in the scenario.  Default: send all of target A, then all of B.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rng  = random.Random(args.seed)

    scenario = load_scenario(args.scenario)

    # Apply CLI overrides
    if args.pilot_sigma_us is not None:
        scenario.error_model.pilot_timing_sigma_ns = args.pilot_sigma_us * 1_000
    if args.edge_sigma_us is not None:
        scenario.error_model.edge_timing_sigma_ns = args.edge_sigma_us * 1_000
    if args.ntp_sigma_ms is not None:
        scenario.error_model.ntp_sigma_ns = args.ntp_sigma_ms * 1_000_000
    if args.transmissions is not None:
        scenario.transmissions_per_target = args.transmissions
    if args.duration_mean_s is not None:
        scenario.transmission_duration_mean_s = args.duration_mean_s
    if args.gap_mean_s is not None:
        scenario.inter_transmission_gap_mean_s = args.gap_mean_s

    print_scenario_summary(scenario)

    # Check server connectivity
    with httpx.Client() as probe:
        try:
            r = probe.get(f"{args.server}/health", timeout=3.0)
            health = r.json()
            print(f"  Server at {args.server} is UP  "
                  f"(events: {health.get('event_count', '?')}, "
                  f"fixes: {health.get('fix_count', '?')})")
        except Exception as exc:
            print(f"  [FAIL] Cannot reach server at {args.server}: {exc}")
            print("    Start the server first:  beagle-server --config config/server.yaml")
            sys.exit(1)
    print()

    # Send initial heartbeats so nodes appear on the map immediately.
    with httpx.Client() as hb_client:
        send_heartbeats(hb_client, args.server, scenario.nodes)
    print()

    # Build the ordered transmission sequence.
    # --conversation interleaves targets round-robin (A B A B ...),
    # mimicking two radios in conversation.  Default is sequential (all A, then all B).
    if args.conversation and len(scenario.targets) >= 2:
        tx_sequence = [
            (target, tx_i)
            for tx_i in range(scenario.transmissions_per_target)
            for target in scenario.targets
        ]
        print(f"  Conversation mode: interleaving {len(scenario.targets)} targets "
              f"({', '.join(t.label for t in scenario.targets)})")
        print()
    else:
        tx_sequence = [
            (target, tx_i)
            for target in scenario.targets
            for tx_i in range(scenario.transmissions_per_target)
        ]
        if args.conversation and len(scenario.targets) < 2:
            print("  [WARN] --conversation requires >=2 targets in the scenario; "
                  "running in sequential mode.\n")

    total_tx = len(tx_sequence)
    tx_num = 0

    with httpx.Client() as client:
        for seq_idx, (target, tx_i) in enumerate(tx_sequence):
            tx_num += 1

            # Draw random duration and inter-TX gap for this transmission.
            duration_s = max(1.0, rng.gauss(
                scenario.transmission_duration_mean_s,
                scenario.transmission_duration_sigma_s,
            ))
            if scenario.rapid_keyup_interval_ms > 0:
                gap_s = scenario.rapid_keyup_interval_ms / 1000.0
            else:
                gap_s = max(0.2, rng.gauss(
                    scenario.inter_transmission_gap_mean_s,
                    scenario.inter_transmission_gap_sigma_s,
                ))

            # -------------------------------------------------------
            # ONSET phase
            # -------------------------------------------------------
            t_onset_ns = int(time.time_ns())
            print_transmission_header(
                tx_num, total_tx, target, t_onset_ns,
                event_type="onset", duration_s=(None if args.no_offset else duration_s),
            )

            onset_events = synthesise_onset(
                target=target,
                nodes=scenario.nodes,
                sync_tx=scenario.sync_transmitter,
                error_model=scenario.error_model,
                t_tx_ns=t_onset_ns,
                rng=rng,
            )
            print_node_table(onset_events)
            print_tdoa_pairs(onset_events)

            ok = send_events(client, args.server, args.auth_token,
                             onset_events, label="onset events")

            if not ok:
                print("\n  [WARN] Some onset events failed - skipping remainder of this TX.\n")
                continue

            if args.no_offset:
                # Onset-only mode: wait for fix then move on.
                onset_fix = poll_for_fix(
                    client=client,
                    server_url=args.server,
                    t_tx_ns=t_onset_ns,
                    target=target,
                    event_type="onset",
                    delivery_buffer_s=args.delivery_buffer_s,
                )
                print_fix_result(onset_fix, target, "onset")
            else:
                # -------------------------------------------------------
                # Carrier in progress - wait for duration_s
                # -------------------------------------------------------
                print(f"\n  [CARRIER ON]  Holding for {duration_s:.1f} s ...")
                # The server's delivery_buffer fires during this wait,
                # computing the onset fix in the background.
                t_wait_start = time.monotonic()
                last_hb = t_wait_start
                elapsed = 0.0
                while elapsed < duration_s:
                    remaining = duration_s - elapsed
                    dot_interval = min(5.0, remaining)
                    time.sleep(dot_interval)
                    elapsed = time.monotonic() - t_wait_start
                    # Send periodic heartbeats every 30s to keep nodes "online"
                    if time.monotonic() - last_hb >= 30.0:
                        send_heartbeats(client, args.server, scenario.nodes,
                                        label="periodic heartbeats")
                        last_hb = time.monotonic()
                    if elapsed < duration_s:
                        print(f"  ... {duration_s - elapsed:.0f}s remaining", end="\r", flush=True)
                print(f"  [CARRIER OFF] ({duration_s:.1f} s elapsed)          ")

                # -------------------------------------------------------
                # OFFSET phase
                # -------------------------------------------------------
                t_offset_ns = int(time.time_ns())
                print()
                _hr()
                print(f"  TRANSMISSION {tx_num}/{total_tx}  [OFFSET]  -  {target.label}")
                t_wall = t_offset_ns / 1e9
                ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t_wall))
                print(f"    Offset time:    {ts}")
                print(f"    event_id reuse: server amends stored onset events")
                _hr()

                offset_events = synthesise_offset(
                    target=target,
                    onset_events=onset_events,
                    sync_tx=scenario.sync_transmitter,
                    error_model=scenario.error_model,
                    t_offset_ns=t_offset_ns,
                    rng=rng,
                )
                print_node_table(offset_events)

                ok_off = send_events(client, args.server, args.auth_token,
                                     offset_events, label="offset events")

                # -------------------------------------------------------
                # Poll for both fixes
                # -------------------------------------------------------
                # Onset fix: delivery_buffer fired during the hold time.
                # Remaining wait = max(0, delivery_buffer_s - duration_s).
                onset_remaining = max(0.0, args.delivery_buffer_s - duration_s)
                onset_fix = poll_for_fix(
                    client=client,
                    server_url=args.server,
                    t_tx_ns=t_onset_ns,
                    target=target,
                    event_type="onset",
                    delivery_buffer_s=onset_remaining,
                    max_wait_s=10.0,
                )
                print_fix_result(onset_fix, target, "onset")

                if ok_off:
                    offset_fix = poll_for_fix(
                        client=client,
                        server_url=args.server,
                        t_tx_ns=t_offset_ns,
                        target=target,
                        event_type="offset",
                        delivery_buffer_s=args.delivery_buffer_s,
                        max_wait_s=10.0,
                    )
                    print_fix_result(offset_fix, target, "offset")

            # -------------------------------------------------------
            # Inter-transmission gap
            # -------------------------------------------------------
            is_last = (seq_idx == total_tx - 1)
            if not is_last:
                print(f"  Next transmission in {gap_s:.1f} s ...")
                time.sleep(gap_s)

    _hr()
    print(f"  Done. {tx_num} transmission(s) sent.")
    print(f"  View live map: {args.server}/map")
    _hr()


if __name__ == "__main__":
    main()
