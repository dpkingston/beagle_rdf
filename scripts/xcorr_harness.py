# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
xcorr_harness.py — evaluate xcorr options against a corpus of real paired
snippets with geometric ground truth.

Matrix: {edge handling} × {differentiation basis} × {xcorr sign}.

Inputs:
  - /tmp/corpus_events.tsv  — fix_id|event_type|fix_onset_ns|node_id|raw_json
  - /tmp/recent_magnolia_fixes.tsv — for the fix metadata
  - TARGET coordinates (Magnolia repeater) hard-coded.

Run: env/bin/python scripts/xcorr_harness.py

Prints a summary table and writes detailed per-pair results to
/tmp/xcorr_harness_results.csv.
"""
from __future__ import annotations

import base64
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal.windows import hann, tukey

# --------------------------------------------------------------------------
# Geometry (ground truth)
# --------------------------------------------------------------------------

TARGET = (47.65133, -122.3918318)       # Magnolia repeater
C_M_S = 299_792_458.0
T_SYNC_NS = 1e9 / 1187.5                # 842,105.26 ns


def haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    phi1 = math.radians(la1)
    phi2 = math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def true_tdoa_ns(ev_a, ev_b):
    """True (target_arrival_A - target_arrival_B) in ns."""
    la = ev_a["node_location"]["latitude_deg"]
    lo = ev_a["node_location"]["longitude_deg"]
    lb = ev_b["node_location"]["latitude_deg"]
    lo2 = ev_b["node_location"]["longitude_deg"]
    d_a = haversine_m(TARGET[0], TARGET[1], la, lo)
    d_b = haversine_m(TARGET[0], TARGET[1], lb, lo2)
    return (d_a - d_b) / C_M_S * 1e9


def sync_path_corr_ns(ev_a, ev_b):
    """(d_sync_a - d_sync_b)/c in ns — added to raw_ns in current code."""
    sa = ev_a["sync_transmitter"]
    sb = ev_b["sync_transmitter"]
    la = ev_a["node_location"]["latitude_deg"]
    lo = ev_a["node_location"]["longitude_deg"]
    lb = ev_b["node_location"]["latitude_deg"]
    lo2 = ev_b["node_location"]["longitude_deg"]
    d_a = haversine_m(sa["latitude_deg"], sa["longitude_deg"], la, lo)
    d_b = haversine_m(sb["latitude_deg"], sb["longitude_deg"], lb, lo2)
    return (d_a - d_b) / C_M_S * 1e9


# --------------------------------------------------------------------------
# Snippet decoding and envelope variants
# --------------------------------------------------------------------------

def decode_iq(b64):
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.int8)
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 127.0


def env_zero_pad(iq, smooth=16):
    """E0: current production — np.convolve mode='same' with zero boundary."""
    p = iq.real.astype(np.float64)**2 + iq.imag.astype(np.float64)**2
    k = np.ones(smooth) / smooth
    return np.convolve(p, k, mode="same").astype(np.float32)


def env_edge_pad(iq, smooth=16):
    """E1: edge-pad the input before convolution, then mode='valid'."""
    p = iq.real.astype(np.float64)**2 + iq.imag.astype(np.float64)**2
    pad = smooth // 2
    p_padded = np.concatenate([np.full(pad, p[0]),
                                p,
                                np.full(smooth - 1 - pad, p[-1])])
    k = np.ones(smooth) / smooth
    return np.convolve(p_padded, k, mode="valid").astype(np.float32)  # len == len(p)


def env_valid_only(iq, smooth=16):
    """E5: mode='valid' — output shorter than input by smooth-1 samples."""
    p = iq.real.astype(np.float64)**2 + iq.imag.astype(np.float64)**2
    k = np.ones(smooth) / smooth
    return np.convolve(p, k, mode="valid").astype(np.float32)


# --------------------------------------------------------------------------
# xcorr
# --------------------------------------------------------------------------

def xcorr_lag_ns(a, b, rate_hz):
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()
    A = np.fft.fft(a, n=n_fft)
    B = np.fft.fft(b, n=n_fft)
    cc = np.fft.ifft(B * np.conj(A))
    cc_abs = np.abs(cc)
    max_lag = min(len(a), len(b)) // 2
    lags = np.concatenate([cc_abs[n_fft - max_lag:], cc_abs[:max_lag + 1]])
    peak = int(np.argmax(lags))
    il = peak - max_lag
    pl = lags[peak - 1] if peak > 0 else lags[peak]
    pr = lags[peak + 1] if peak < len(lags) - 1 else lags[peak]
    denom = pl - 2 * lags[peak] + pr
    sub = 0.0 if denom == 0 else 0.5 * (pl - pr) / denom
    lag_samples = il + sub
    pk = float(lags[peak])
    sl = float(np.mean(lags[lags < pk])) if pk > 0 else 1.0
    snr = pk / max(sl, 1e-30)
    return lag_samples * 1e9 / rate_hz, snr


# --------------------------------------------------------------------------
# Prep: strategy applies a particular (edge, basis, post-process) combo
# --------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    env_fn: Callable[[np.ndarray, int], np.ndarray]
    basis: str              # "env" | "d1" | "d2"
    trim: int = 0
    window_fn: Callable[[int], np.ndarray] | None = None
    # If set, use the reported transition_start/end to extract only the
    # transition region of the signal before xcorr.  `pad` samples are
    # included on each side of the transition.
    transition_window: int | None = None
    zero_mean: bool = False

    def prepare(self, iq, trans_start=None, trans_end=None):
        env = self.env_fn(iq, 16)
        if self.basis == "env":
            x = env
        elif self.basis == "d1":
            x = np.diff(env)
        else:
            x = np.diff(np.diff(env))
        if self.trim > 0:
            x = x[self.trim:len(x) - self.trim]
        if (self.transition_window is not None and trans_start is not None
                and trans_end is not None and trans_end > trans_start):
            pad = self.transition_window
            a = max(0, trans_start - pad)
            b = min(len(x), trans_end + pad)
            x = x[a:b]
        if self.zero_mean:
            x = x - x.mean()
        if self.window_fn is not None:
            w = self.window_fn(len(x)).astype(np.float32)
            x = x * w
        return x


STRATEGIES: list[Strategy] = [
    # --- Baseline: current production code ---
    Strategy("E0.d2 (current)", env_zero_pad, "d2", trim=0),

    # --- Edge-pad variants ---
    Strategy("E1.d1",            env_edge_pad, "d1", trim=0),
    Strategy("E1.d2",            env_edge_pad, "d2", trim=0),
    Strategy("E1.env",           env_edge_pad, "env", trim=0),

    # --- Post-smooth trim on zero-pad envelope ---
    Strategy("E2-16.d1",         env_zero_pad, "d1", trim=16),
    Strategy("E2-16.d2",         env_zero_pad, "d2", trim=16),
    Strategy("E2-32.d1",         env_zero_pad, "d1", trim=32),
    Strategy("E2-32.d2",         env_zero_pad, "d2", trim=32),
    Strategy("E2-64.d1",         env_zero_pad, "d1", trim=64),
    Strategy("E2-64.d2",         env_zero_pad, "d2", trim=64),

    # --- Tukey window on zero-pad d1/d2 (α=0.25) ---
    Strategy("E3a.d1",           env_zero_pad, "d1",
             window_fn=lambda n: tukey(n, 0.25)),
    Strategy("E3a.d2",           env_zero_pad, "d2",
             window_fn=lambda n: tukey(n, 0.25)),

    # --- Hann window ---
    Strategy("E4.d1",            env_zero_pad, "d1", window_fn=hann),
    Strategy("E4.d2",            env_zero_pad, "d2", window_fn=hann),

    # --- mode='valid' envelope ---
    Strategy("E5.d1",            env_valid_only, "d1", trim=0),
    Strategy("E5.d2",            env_valid_only, "d2", trim=0),

    # --- Transition-region only: use reported transition_start/end to
    #     isolate the PA transition before xcorr.  Both nodes report the
    #     same transition_start/end (from the snippet encoder anchoring),
    #     so this selects consistent regions across nodes.
    Strategy("TW-32.d1",         env_edge_pad, "d1", transition_window=32),
    Strategy("TW-32.d2",         env_edge_pad, "d2", transition_window=32),
    Strategy("TW-64.d1",         env_edge_pad, "d1", transition_window=64),
    Strategy("TW-64.d2",         env_edge_pad, "d2", transition_window=64),
    Strategy("TW-128.d1",        env_edge_pad, "d1", transition_window=128),
    Strategy("TW-128.d2",        env_edge_pad, "d2", transition_window=128),
    Strategy("TW-256.d1",        env_edge_pad, "d1", transition_window=256),
    Strategy("TW-256.d2",        env_edge_pad, "d2", transition_window=256),

    # --- Transition-region + Tukey taper to suppress edges of that window ---
    Strategy("TW-64+Tukey.d1",   env_edge_pad, "d1", transition_window=64,
             window_fn=lambda n: tukey(n, 0.3)),
    Strategy("TW-128+Tukey.d1",  env_edge_pad, "d1", transition_window=128,
             window_fn=lambda n: tukey(n, 0.3)),

    # --- Zero-mean + Tukey (remove DC bias from plateau region before xcorr) ---
    Strategy("ZM+Tukey.d1",      env_edge_pad, "d1",
             zero_mean=True, window_fn=lambda n: tukey(n, 0.25)),
    Strategy("ZM+Tukey.d2",      env_edge_pad, "d2",
             zero_mean=True, window_fn=lambda n: tukey(n, 0.25)),
]


# --------------------------------------------------------------------------
# Load corpus
# --------------------------------------------------------------------------

def load_corpus():
    """Return {fix_id: {'event_type': ..., 'pairs': {(node_a, node_b): (ev_a, ev_b)}}}."""
    by_fix: dict[int, dict[str, dict]] = defaultdict(dict)
    fix_meta: dict[int, dict] = {}
    with open("/tmp/corpus_events.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 4)
            if len(parts) != 5:
                continue
            fid, etype, fix_onset_ns, node_id, raw = parts
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            fid = int(fid)
            by_fix[fid][node_id] = d
            fix_meta[fid] = {"event_type": etype, "fix_onset_ns": int(fix_onset_ns)}
    out = []
    for fid, nodes in by_fix.items():
        if "dpk-tdoa1" in nodes and "dpk-tdoa2" in nodes:
            out.append({
                "fix_id": fid,
                "event_type": fix_meta[fid]["event_type"],
                "ev_a": nodes["dpk-tdoa1"],  # A=tdoa1, B=tdoa2
                "ev_b": nodes["dpk-tdoa2"],
            })
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def evaluate_pair(pair, strategy: Strategy, sign: int):
    """
    sign = +1 → combined = raw + xcorr + path (current code)
    sign = -1 → combined = raw - xcorr + path
    Returns (computed_tdoa_ns, xcorr_lag_ns, xcorr_snr, integer_n, err_vs_truth_ns).
    """
    ea = pair["ev_a"]
    eb = pair["ev_b"]
    iq_a = decode_iq(ea["iq_snippet_b64"])
    iq_b = decode_iq(eb["iq_snippet_b64"])
    rate = ea["channel_sample_rate_hz"]
    if rate != eb["channel_sample_rate_hz"]:
        return None

    # Pass transition_start/end so TW-* strategies can window to it.
    # The basis (d1/d2) reduces length by 1 or 2, and smoothing/trim may
    # shift positions slightly, but transition_start/end are approximate
    # indices into the full snippet — fine for coarse windowing.
    ts_a = int(ea.get("transition_start", 0))
    te_a = int(ea.get("transition_end", 0))
    ts_b = int(eb.get("transition_start", 0))
    te_b = int(eb.get("transition_end", 0))
    a = strategy.prepare(iq_a, ts_a, te_a)
    b = strategy.prepare(iq_b, ts_b, te_b)
    L = min(len(a), len(b))
    a, b = a[:L], b[:L]
    lag_ns, snr = xcorr_lag_ns(a, b, rate)

    raw_ns = float(ea["sync_delta_ns"]) - float(eb["sync_delta_ns"])
    path_ns = sync_path_corr_ns(ea, eb)
    combined = raw_ns + sign * lag_ns + path_ns
    n = round(combined / T_SYNC_NS)
    tdoa = combined - n * T_SYNC_NS

    truth = true_tdoa_ns(ea, eb)
    err = tdoa - truth
    return tdoa, lag_ns, snr, n, err


def main():
    pairs = load_corpus()
    print(f"Evaluating {len(pairs)} paired (dpk-tdoa1, dpk-tdoa2) fixes "
          f"against Magnolia ground truth.")
    print()

    rows = []
    for p in pairs:
        for strat in STRATEGIES:
            for sign in (+1, -1):
                r = evaluate_pair(p, strat, sign)
                if r is None:
                    continue
                tdoa, lag_ns, snr, n, err = r
                rows.append({
                    "fix_id": p["fix_id"],
                    "event_type": p["event_type"],
                    "strategy": strat.name,
                    "sign": sign,
                    "xcorr_lag_ns": lag_ns,
                    "xcorr_snr": snr,
                    "n_disambig": n,
                    "tdoa_ns": tdoa,
                    "err_ns": err,
                    "true_tdoa_ns": tdoa - err,
                })

    # Summary: per (strategy, sign), median |err|, p75 |err|, max |err|,
    # fraction with |err| < 50 us, median SNR
    summary = defaultdict(list)
    for r in rows:
        key = (r["strategy"], r["sign"])
        summary[key].append(r)

    print(f"{'strategy':20s} {'sign':4s}  {'med|err|(µs)':>14s} "
          f"{'p75|err|':>9s} {'max|err|':>9s} {'<50µs %':>8s} "
          f"{'med_SNR':>7s}")
    print("-" * 80)

    sorted_keys = sorted(summary.keys(),
                         key=lambda k: np.median([abs(r["err_ns"]) for r in summary[k]]))
    for key in sorted_keys:
        rs = summary[key]
        errs = np.array([abs(r["err_ns"]) for r in rs])
        snrs = np.array([r["xcorr_snr"] for r in rs])
        frac_good = 100 * np.mean(errs < 50_000)
        strat, sign = key
        sign_str = "+" if sign == +1 else "-"
        print(f"{strat:20s}  {sign_str:3s}   "
              f"{np.median(errs)/1000:>12.1f}   "
              f"{np.percentile(errs, 75)/1000:>8.1f}  "
              f"{np.max(errs)/1000:>8.1f}  "
              f"{frac_good:>7.1f}   "
              f"{np.median(snrs):>6.1f}")

    # Detail CSV
    with open("/tmp/xcorr_harness_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"Wrote {len(rows)} rows to /tmp/xcorr_harness_results.csv")


if __name__ == "__main__":
    main()
