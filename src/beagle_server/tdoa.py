# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
TDOA arithmetic for the aggregation server.

Core operations:
  - haversine_m: great-circle distance between two lat/lon points
  - compute_tdoa: raw sync_delta subtraction + path-delay correction

Derivation of the path-delay correction
----------------------------------------
Each node's sync_delta_ns is measured on the local sample clock:

    sync_delta_n = (target_onset_at_n) - (sync_event_at_n)
                 = [T_t + dist(target, n)/c] - [T_s + dist(sync, n)/c]
                 = K + [dist(target, n) - dist(sync, n)] / c

where K = T_t - T_s is a common constant (same for all nodes for the same
transmission event).

Raw TDOA (from two nodes A and B):

    raw_ns = sync_delta_A - sync_delta_B
           = [(dist(target,A) - dist(target,B)) - (dist(sync,A) - dist(sync,B))] / c * 1e9

True TDOA = (dist(target,A) - dist(target,B)) / c

Correction = (dist(sync,A) - dist(sync,B)) / c * 1e9   (ADD to raw)

    TDOA_ns = raw_ns + correction_ns

where c = 299,792,458 m/s.

A positive result means node A heard the target carrier *later* than node B
(i.e. the target is farther from A than from B).
"""

from __future__ import annotations

import base64
import logging
import math
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

import os
_SYNC_DIAG = os.environ.get("BEAGLE_SYNC_DIAG", "") == "1"


def set_sync_diag(enabled: bool) -> None:
    """Enable or disable sync diagnostic logging (called from server startup)."""
    global _SYNC_DIAG
    _SYNC_DIAG = enabled

_C_M_S = 299_792_458.0  # speed of light, m/s


class SyncCalibrator:
    """Rolling calibration of the per-pair sync bit-grid offset.

    Each node initializes its RDS bit boundary grid at an arbitrary pilot
    phase, creating a fixed fractional-bit offset between any two nodes.
    This offset is stable but node-pair-specific and resets on node restart.

    Measurement method (sample-based, no wall-clock contamination):
    Each carrier event includes sync_sample_index (the pilot-derived
    sub-sample position of the matched RDS bit boundary) and
    sync_sample_rate_correction (crystal calibration factor).

    The difference in sync_sample_index between two nodes, expressed in
    RDS bit periods, should be near an integer.  The fractional part is
    the grid offset.  We track this with an EMA and convert to ns for
    correction.
    """

    def __init__(self, alpha: float = 0.02, min_samples: int = 5) -> None:
        self._alpha = alpha
        self._min_samples = min_samples
        # Key: tuple(sorted([node_a, node_b])), value: (correction_frac, count)
        # correction_frac is in fractional bit periods, canonical direction.
        self._pairs: dict[tuple[str, str], tuple[float, int]] = {}
        # Track last sync_sample_index per node to detect restarts.
        self._last_idx: dict[str, float] = {}

    def update(
        self,
        node_a: str,
        node_b: str,
        event_a: dict,
        event_b: dict,
    ) -> float:
        """Update calibration from sync sample indices and return correction in ns.

        Uses sync_sample_index and sync_sample_rate_correction (sent by
        every node in every carrier event) to compute the fractional-bit
        grid offset purely from sample counting — no wall-clock involved.

        Detects node restarts (sync_sample_index drops) and resets the
        affected pair's EMA so stale calibration doesn't persist.
        """
        key = tuple(sorted([node_a, node_b]))
        sign = 1.0 if key[0] == node_a else -1.0

        sync_idx_a = event_a.get("sync_sample_index", 0.0)
        sync_idx_b = event_b.get("sync_sample_index", 0.0)
        corr_a = event_a.get("sync_sample_rate_correction", 0.0)
        corr_b = event_b.get("sync_sample_rate_correction", 0.0)

        # Need valid sample indices from both nodes.  Legacy nodes send 0.
        if sync_idx_a == 0.0 or sync_idx_b == 0.0 or corr_a == 0.0 or corr_b == 0.0:
            est, count = self._pairs.get(key, (0.0, 0))
            return est * sign * _T_SYNC_NS if count >= self._min_samples else 0.0

        # Detect node restarts: sync_sample_index drops significantly.
        # A restart resets the pilot phase tracker, changing the grid offset,
        # so all affected pair EMA states must be reset.
        for nid, idx in [(node_a, sync_idx_a), (node_b, sync_idx_b)]:
            prev = self._last_idx.get(nid)
            if prev is not None and idx < prev * 0.5:
                # Node restarted — reset all pairs involving this node.
                stale = [k for k in self._pairs if nid in k]
                for k in stale:
                    del self._pairs[k]
                if stale:
                    logger.info(
                        "sync_cal: node %s restarted (idx %.0f -> %.0f); "
                        "reset %d pair(s)", nid, prev, idx, len(stale),
                    )
            self._last_idx[nid] = idx

        # Compute fractional-bit offset from sample indices.
        # sync_sample_index is in sync-decimated sample space (~250 kHz).
        sync_diff_samples = sync_idx_a - sync_idx_b
        rate_avg = 250_000.0 * (corr_a + corr_b) / 2.0
        rds_bit_samples = rate_avg / 1187.5
        sync_diff_bits = sync_diff_samples / rds_bit_samples
        frac = sync_diff_bits - round(sync_diff_bits)

        # Sanity check: frac should be in [-0.5, 0.5].  If the existing EMA
        # and the new measurement disagree by more than 0.25 (a quarter bit
        # period = 210 us), the EMA is likely stale (e.g. from a restart we
        # didn't catch, or initial convergence).  Reset.
        signed_frac = frac * sign
        est, count = self._pairs.get(key, (0.0, 0))
        if count >= self._min_samples and abs(signed_frac - est) > 0.25:
            logger.info(
                "sync_cal: %s<->%s frac jumped %.4f -> %.4f; resetting EMA",
                key[0], key[1], est, signed_frac,
            )
            est = signed_frac
            count = 0

        if count == 0:
            est = signed_frac
        else:
            est = est + self._alpha * (signed_frac - est)
        self._pairs[key] = (est, count + 1)

        correction_frac = est * sign
        if count + 1 < self._min_samples:
            return 0.0
        return correction_frac * _T_SYNC_NS


# Global calibrator instance (persists across requests within a server process)
_sync_calibrator = SyncCalibrator()


def reset_sync_calibrator() -> None:
    """Reset the global sync calibrator.  For testing only."""
    global _sync_calibrator
    _sync_calibrator = SyncCalibrator()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS-84 points in metres.

    Uses the Haversine formula.  Accurate to within ~0.3% for distances up
    to a few thousand kilometres (more than sufficient for Seattle metro).
    """
    R = 6_371_000.0  # Earth mean radius, metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(a))


def path_delay_correction_ns(
    sync_tx_lat: float,
    sync_tx_lon: float,
    node_a_lat: float,
    node_a_lon: float,
    node_b_lat: float,
    node_b_lon: float,
) -> float:
    """
    Return the path-delay correction in nanoseconds:

        (dist(sync_tx, A) - dist(sync_tx, B)) / c * 1e9

    **Add** this to the raw TDOA (sync_delta_A - sync_delta_B) to obtain
    the true TDOA.  See module docstring for the derivation.
    """
    d_a = haversine_m(sync_tx_lat, sync_tx_lon, node_a_lat, node_a_lon)
    d_b = haversine_m(sync_tx_lat, sync_tx_lon, node_b_lat, node_b_lon)
    return (d_a - d_b) / _C_M_S * 1e9


def _decode_iq_snippet(b64: str) -> np.ndarray:
    """Decode a base64 int8-interleaved snippet to complex64."""
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.int8)
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 127.0


def _compute_power_envelope(iq: np.ndarray, smooth_samples: int = 16) -> np.ndarray:
    """Compute smoothed instantaneous power envelope from IQ samples."""
    power = iq.real.astype(np.float64)**2 + iq.imag.astype(np.float64)**2
    kernel = np.ones(smooth_samples) / smooth_samples
    return np.convolve(power, kernel, mode='same')


def _find_peak_derivative_sample(
    iq: np.ndarray,
    event_type: str,
    smooth_samples: int = 16,
) -> tuple[float, float] | None:
    """
    Find the sample of maximum power-change rate in an IQ snippet.

    For onset: argmax of d(power)/dt - the steepest point of the carrier rise.
    For offset: argmin of d(power)/dt - the steepest point of the carrier fall.

    Both are transmitter properties (set by PA electronics) and are observed at
    the same physical instant by all receivers, regardless of their gain or noise
    floor.  This makes them robust anchors for inter-node cross-correlation even
    when nodes have 8-10 dB noise-floor differences.

    Replaces the former plateau-start/end approach, which required finding a
    "stable" derivative region whose detection threshold was SNR-dependent.

    Parameters
    ----------
    iq : np.ndarray, complex64
        IQ snippet.
    event_type : str
        "onset" (find rising peak) or "offset" (find falling peak).
    smooth_samples : int
        Box-filter half-width for the power envelope (samples).

    Returns
    -------
    (float_sample_index, sharpness) or None
        float_sample_index : sub-sample position of the peak derivative.
        sharpness : peak |derivative| / RMS of the remaining derivative
            values.  Low sharpness (< min_plateau_sharpness) means the
            transition is gradual or noise-dominated and the pair should
            be skipped.
    Returns None if the snippet is too short or has no clear transition.
    """
    if len(iq) < smooth_samples * 4:
        return None

    envelope = _compute_power_envelope(iq, smooth_samples)
    deriv = np.diff(envelope)
    if len(deriv) < 4:
        return None

    if event_type == "onset":
        peak_idx = int(np.argmax(deriv))
        peak_val = float(deriv[peak_idx])
        if peak_val <= 0:
            return None
    else:  # offset
        peak_idx = int(np.argmin(deriv))
        peak_val = float(deriv[peak_idx])
        if peak_val >= 0:
            return None

    # Parabolic sub-sample interpolation on the derivative peak
    p_left  = float(deriv[peak_idx - 1]) if peak_idx > 0            else peak_val
    p_right = float(deriv[peak_idx + 1]) if peak_idx < len(deriv) - 1 else peak_val
    denom = p_left - 2.0 * peak_val + p_right
    sub_offset = 0.0 if denom == 0.0 else 0.5 * (p_left - p_right) / denom
    float_idx = float(peak_idx) + float(np.clip(sub_offset, -0.5, 0.5))

    # Sharpness: peak |derivative| vs RMS of the non-peak region
    noise_mask = np.ones(len(deriv), dtype=bool)
    noise_mask[max(0, peak_idx - 3) : peak_idx + 4] = False
    noise_region = deriv[noise_mask]
    rms_noise = float(np.sqrt(np.mean(noise_region ** 2))) if len(noise_region) > 0 else 1e-30
    sharpness = abs(peak_val) / max(rms_noise, 1e-30)

    return float_idx, sharpness


def _resample_to_rate(signal: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    """
    Resample *signal* from *src_rate* to *dst_rate* using exact rational
    polyphase resampling (scipy.signal.resample_poly).

    The up/down integers are derived from the integer representation of the
    two rates, so common SDR rates (62500, 64000, ...) map to small exact
    fractions:
        62500 -> 64000 : up=128, down=125
        64000 -> 62500 : up=125, down=128

    Returns the input array unchanged if src_rate == dst_rate (within 1 ppm).
    """
    if abs(src_rate - dst_rate) / max(dst_rate, 1.0) < 1e-6:
        return signal
    src_int = round(src_rate)
    dst_int = round(dst_rate)
    g = gcd(src_int, dst_int)
    up, down = dst_int // g, src_int // g
    return resample_poly(signal, up, down).astype(signal.dtype)


def _xcorr_arrays(
    a: np.ndarray, b: np.ndarray, sample_rate_hz: float
) -> tuple[float, float]:
    """FFT cross-correlation of two complex arrays. Returns (lag_ns, snr)."""
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()
    A = np.fft.fft(a, n=n_fft)
    B = np.fft.fft(b, n=n_fft)
    cc = np.fft.ifft(B * np.conj(A))
    cc_abs = np.abs(cc)
    max_lag = min(len(a), len(b)) // 2
    lags = np.concatenate([cc_abs[n_fft - max_lag:], cc_abs[:max_lag + 1]])
    peak_idx = int(np.argmax(lags))
    integer_lag = peak_idx - max_lag
    p_left = lags[peak_idx - 1] if peak_idx > 0 else lags[peak_idx]
    p_right = lags[peak_idx + 1] if peak_idx < len(lags) - 1 else lags[peak_idx]
    denom = p_left - 2.0 * lags[peak_idx] + p_right
    sub_offset = 0.0 if denom == 0.0 else 0.5 * (p_left - p_right) / denom
    lag_ns = (integer_lag + sub_offset) * 1e9 / sample_rate_hz
    peak_val = float(lags[peak_idx])
    sidelobe_mean = float(np.mean(lags[lags < peak_val])) if peak_val > 0 else 1.0
    snr = peak_val / max(sidelobe_mean, 1e-30)
    return lag_ns, snr


def cross_correlate_snippets(
    a_b64: str,
    b_b64: str,
    sample_rate_hz_a: float = 64_000.0,
    sample_rate_hz_b: float | None = None,
    target_rate_hz: float | None = None,
    event_type: str = "",
    transition_a: tuple[int, int] | None = None,
    transition_b: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """
    Cross-correlate two int8 IQ snippets to find the inter-node TDOA.

    Cross-correlates the **power envelopes** (not the raw IQ) so that LO phase
    differences between independent receivers do not produce false correlation
    peaks.  Uses FFT-based correlation with parabolic peak interpolation for
    sub-sample precision.

    If ``transition_a`` and ``transition_b`` are provided (from the node's
    knee-finding algorithm), the correlation is restricted to those sample
    ranges within each snippet.  This focuses xcorr on the PA transition and
    excludes noise/plateau regions that could produce false peaks.

    If transition bounds are not available (legacy nodes), falls back to
    fixed-fraction trimming based on event_type.

    Parameters
    ----------
    a_b64, b_b64 : str
        Base64-encoded int8-interleaved IQ snippets.
    sample_rate_hz_a : float
        Sample rate of snippet A (default 64 kHz).
    sample_rate_hz_b : float or None
        Sample rate of snippet B.  None means same as A.
    target_rate_hz : float or None
        Rate to resample both to before correlation.  None = lower of the two.
    event_type : str
        "onset" or "offset" for legacy trimming fallback.
    transition_a, transition_b : tuple[int, int] or None
        (start, end) sample indices within each snippet bounding the PA
        transition zone.  From the node's d2 knee-finding walk.

    Returns
    -------
    (lag_ns, corr_snr)
        lag_ns : float
            Estimated TDOA in nanoseconds.  Positive = A arrives *later* than B.
        corr_snr : float
            Peak-to-sidelobe ratio of the cross-correlation (dimensionless).
            Returns 0.0 if the transition window contains no signal (misaligned
            snippet), triggering sync_delta fallback in compute_tdoa_s.
    """
    rate_b = sample_rate_hz_b if sample_rate_hz_b is not None else sample_rate_hz_a
    effective_rate = target_rate_hz if target_rate_hz is not None else min(sample_rate_hz_a, rate_b)

    a = _decode_iq_snippet(a_b64)
    b = _decode_iq_snippet(b_b64)

    # Cross-correlate power envelopes to remove LO phase dependence.
    # Independent receivers have unrelated LO phases and small frequency
    # offsets (+/-30 ppm for RTL-SDR at 443 MHz = +/-13 kHz) that cause the
    # complex IQ to rotate at different rates, destroying the complex
    # cross-correlation peak.  The power envelope |IQ|^2 is phase-free.
    env_a = _compute_power_envelope(a).astype(np.float32)
    env_b = _compute_power_envelope(b).astype(np.float32)

    # Resample to a common rate if the two nodes captured at different rates.
    if abs(sample_rate_hz_a - effective_rate) / effective_rate > 1e-6:
        env_a = _resample_to_rate(env_a, sample_rate_hz_a, effective_rate)
    if abs(rate_b - effective_rate) / effective_rate > 1e-6:
        env_b = _resample_to_rate(env_b, rate_b, effective_rate)

    # Trim to the transition region.  When the node provides transition
    # bounds (from the d2 knee-finding walk), use them directly — they
    # tightly bracket the PA edge.  Otherwise fall back to fixed-fraction
    # trimming based on event_type.
    if transition_a and transition_a[1] > transition_a[0]:
        env_a = env_a[transition_a[0]:transition_a[1]]
    elif event_type == "onset":
        env_a = env_a[: 3 * len(env_a) // 4]
    elif event_type == "offset":
        env_a = env_a[len(env_a) // 2 :]

    if transition_b and transition_b[1] > transition_b[0]:
        env_b = env_b[transition_b[0]:transition_b[1]]
    elif event_type == "onset":
        env_b = env_b[: 3 * len(env_b) // 4]
    elif event_type == "offset":
        env_b = env_b[len(env_b) // 2 :]

    # Equalise lengths after resampling + trimming.  Different sample rates
    # (e.g. 64 kHz RTL-SDR vs 62.5 kHz RSPduo) produce different-length
    # envelopes after resampling and proportional trimming.  Without this,
    # _xcorr_arrays' max_lag = min(len_a, len_b)//2 creates an edge artifact:
    # a false correlation peak at the search boundary that masquerades as a
    # large TDOA (e.g. 3.752 ms when the true lag is ~0).
    min_len = min(len(env_a), len(env_b))
    env_a = env_a[:min_len]
    env_b = env_b[:min_len]

    # Guard: if the xcorr window has no signal (misaligned or missing snippet),
    # return zero SNR so the caller falls back to sync_delta.
    if float(np.max(env_a)) < 1e-4 or float(np.max(env_b)) < 1e-4:
        return 0.0, 0.0

    return _xcorr_arrays(env_a, env_b, effective_rate)


# RDS bit period in nanoseconds (1/1187.5 Hz = 842.1 usec).
# All nodes receiving the same FM station see the same RDS bit transitions,
# so disambiguation should rarely need n != 0.  The margin is:
#   max_TDOA (~333 usec) < T_sync/2 (421 usec) -> unambiguous.
_T_SYNC_NS: float = 1_000_000_000.0 / 1187.5  # 842,105.26 ns

# Speed of light in m/s - used to convert baseline distance to max TDOA.
_C_M_PER_S: float = 299_792_458.0


def compute_tdoa_s(
    event_a: dict[str, Any],
    event_b: dict[str, Any],
    min_xcorr_snr: float = 1.3,
    xcorr_target_rate_hz: float | None = None,
    max_xcorr_baseline_km: float = 100.0,
) -> float | None:
    """
    Compute the corrected TDOA between two events in **seconds**.

    Both events must be from the same transmission (same channel, event_type,
    and sync transmitter).  The path-delay correction is applied using the
    sync transmitter coordinates and node locations carried in each event.

    Method selection
    ----------------
    1. **xcorr** (primary): when both events carry ``iq_snippet_b64``, the
       power-envelope cross-correlation is computed.  If SNR >= min_xcorr_snr
       AND |lag| <= max physical TDOA for max_xcorr_baseline_km, the xcorr lag
       is returned directly as the TDOA - no path correction is applied because
       xcorr measures the physical carrier arrival time difference between nodes
       directly, bypassing the sync-event reference.  The geometric plausibility
       filter rejects false detections caused by snippets where the PA transition
       is absent (e.g. false offset triggers), which produce random large lags.

    2. **sync_delta** (fallback): when snippets are absent, xcorr SNR is below
       threshold, or the xcorr lag fails the geometric plausibility check.
       Computes sync_delta_A - sync_delta_B with pilot disambiguation (requires
       onset_time_ns) and adds the sync-tx path-delay correction.

    Parameters
    ----------
    event_a, event_b : dicts with keys:
        sync_delta_ns, sync_tx_lat, sync_tx_lon, node_lat, node_lon,
        event_type, node_id.
        onset_time_ns (optional): wall-clock time of the carrier edge in ns;
        used by the sync_delta fallback for pilot disambiguation.
        iq_snippet_b64 (optional): base64 IQ snippet; enables xcorr method.
        channel_sample_rate_hz (optional): snippet sample rate (default 64 kHz).
    min_xcorr_snr : float
        Minimum xcorr peak-to-sidelobe ratio to accept the xcorr result.
        Below this threshold the function falls back to sync_delta.
    xcorr_target_rate_hz : float or None
        Resample both envelopes to this rate before xcorr.
        None = use the lower of the two snippet rates.
    max_xcorr_baseline_km : float
        Maximum node-pair separation in km.  Used as a geometric plausibility
        filter: any xcorr lag whose magnitude exceeds the corresponding maximum
        physical TDOA (baseline / c) is treated as a false detection and falls
        back to sync_delta.  Defaults to 100 km (~ 333 usec max TDOA).

    Returns
    -------
    float or None
        TDOA in seconds, or None if sync_delta_ns is missing and xcorr
        is unavailable.  Positive -> A heard the carrier *later* than B.
    """
    node_a = event_a.get("node_id", "?")
    node_b = event_b.get("node_id", "?")
    event_type = event_a.get("event_type", "")

    # --- sync_delta: coarse TDOA from sample counting ---
    # sync_delta_ns = (carrier_sample - sync_sample) / sample_rate.
    # This is precise to the sample boundary (16 usec at 62.5 kHz).
    # The sync path correction removes sync signal propagation geometry,
    # leaving the target signal arrival time difference.
    delta_a = event_a.get("sync_delta_ns")
    delta_b = event_b.get("sync_delta_ns")

    # Sync diagnostics: log pilot phase comparison to verify bit-boundary
    # alignment across nodes.  Gated by BEAGLE_SYNC_DIAG=1 env var.
    if _SYNC_DIAG:
        dsamp_a = event_a.get("sync_delta_samples", 0.0)
        dsamp_b = event_b.get("sync_delta_samples", 0.0)
        sync_idx_a = event_a.get("sync_sample_index", 0.0)
        sync_idx_b = event_b.get("sync_sample_index", 0.0)
        corr_a = event_a.get("sync_sample_rate_correction", 1.0)
        corr_b = event_b.get("sync_sample_rate_correction", 1.0)
        if dsamp_a != 0.0 or dsamp_b != 0.0:
            # Sync sample difference in RDS bit periods — should be near an
            # integer if both nodes are counting bit boundaries consistently.
            sync_diff_samples = sync_idx_a - sync_idx_b
            rate_avg = 250_000.0 * (corr_a + corr_b) / 2.0
            rds_bit_samples = rate_avg / 1187.5
            sync_diff_bits = sync_diff_samples / rds_bit_samples if rds_bit_samples > 0 else 0.0
            sync_diff_frac = sync_diff_bits - round(sync_diff_bits)
            logger.info(
                "sync_diag %s<->%s (%s): delta_ns=[%s, %s]  delta_samp=[%.1f, %.1f]  "
                "sync_idx=[%.1f, %.1f]  sync_diff=%.1f samp (%.2f bits, frac=%.4f)  "
                "crystal=[%.8f, %.8f]",
                node_a, node_b, event_type,
                delta_a, delta_b, dsamp_a, dsamp_b,
                sync_idx_a, sync_idx_b,
                sync_diff_samples, sync_diff_bits, sync_diff_frac,
                corr_a, corr_b,
            )

    if delta_a is None or delta_b is None:
        logger.warning(
            "Missing sync_delta_ns for %s<->%s (%s) - pair skipped",
            node_a, node_b, event_type,
        )
        return None

    raw_ns = float(delta_a) - float(delta_b)

    # Apply sync grid calibration: remove the per-pair fractional-bit
    # offset from the coarse TDOA.  Uses sync-only timing (onset_time_ns
    # minus sync_delta_ns = sync event wall-clock time) to measure the
    # offset without carrier-side contamination.  Applied BEFORE
    # disambiguation so rounding picks the correct bit period.
    grid_correction_ns = _sync_calibrator.update(
        node_a, node_b, event_a, event_b,
    )
    raw_ns -= grid_correction_ns

    if _SYNC_DIAG and grid_correction_ns != 0.0:
        logger.info(
            "sync_cal %s<->%s (%s): grid_correction=%.0f ns  raw_ns_before=%.0f  after=%.0f",
            node_a, node_b, event_type,
            grid_correction_ns, raw_ns + grid_correction_ns, raw_ns,
        )

    correction_ns = path_delay_correction_ns(
        sync_tx_lat=event_a["sync_tx_lat"],
        sync_tx_lon=event_a["sync_tx_lon"],
        node_a_lat=event_a["node_lat"],
        node_a_lon=event_a["node_lon"],
        node_b_lat=event_b["node_lat"],
        node_b_lon=event_b["node_lon"],
    )

    # Sync period disambiguation: resolve which RDS bit boundary each node
    # referenced.  |true_TDOA| <= dist(A,B)/c << T_sync/2 = 421 usec.
    n = round((raw_ns + correction_ns) / _T_SYNC_NS)
    if n != 0:
        logger.debug(
            "Sync disambiguation %s<->%s (%s): n=%+d raw_ns=%+.0f->%+.0f ns",
            node_a, node_b, event_type, n, raw_ns, raw_ns - n * _T_SYNC_NS,
        )
        raw_ns -= n * _T_SYNC_NS

    coarse_tdoa_ns = raw_ns + correction_ns

    # --- xcorr: PA transition alignment across nodes ---
    #
    # Snippets are anchored at the detection point (threshold crossing),
    # NOT the PA transition knee.  Each node's detection fires at a
    # different point on the PA curve (different thresholds, SNR).  The
    # xcorr aligns the PA transition features across nodes, measuring
    # the timing difference between where each node's detection landed
    # relative to the true PA edge.
    #
    # The coarse TDOA (sync_delta) measures sync-to-detection-point; the
    # xcorr correction adjusts for the detection-to-feature offset
    # between nodes.  The sum gives the true TDOA.
    #
    # Geometric plausibility: the total TDOA (coarse + xcorr) must be
    # within max_xcorr_baseline_km.  The xcorr lag itself can be large
    # (hundreds of µs if detection points differ significantly) — this
    # is expected and correct.

    xcorr_refinement_ns = 0.0
    xcorr_used = False
    iq_a = event_a.get("iq_snippet_b64", "")
    iq_b = event_b.get("iq_snippet_b64", "")
    if iq_a and iq_b:
        rate_a = float(event_a.get("channel_sample_rate_hz", 64_000.0))
        rate_b = float(event_b.get("channel_sample_rate_hz", 64_000.0))

        # Extract transition bounds if the node provided them
        t_a_start = int(event_a.get("transition_start", 0))
        t_a_end = int(event_a.get("transition_end", 0))
        t_b_start = int(event_b.get("transition_start", 0))
        t_b_end = int(event_b.get("transition_end", 0))
        trans_a = (t_a_start, t_a_end) if t_a_end > t_a_start else None
        trans_b = (t_b_start, t_b_end) if t_b_end > t_b_start else None

        xcorr_lag_ns, xcorr_snr = cross_correlate_snippets(
            iq_a, iq_b,
            sample_rate_hz_a=rate_a,
            sample_rate_hz_b=rate_b,
            target_rate_hz=xcorr_target_rate_hz,
            event_type=event_type,
            transition_a=trans_a,
            transition_b=trans_b,
        )
        if xcorr_snr >= min_xcorr_snr:
            xcorr_refinement_ns = xcorr_lag_ns
            xcorr_used = True
        else:
            logger.debug(
                "xcorr SNR too low: %.2f < %.2f for %s<->%s (%s); "
                "using sync_delta only",
                xcorr_snr, min_xcorr_snr, node_a, node_b, event_type,
            )

    tdoa_ns = coarse_tdoa_ns + xcorr_refinement_ns

    # Geometric plausibility: when xcorr is used, the total TDOA must not
    # exceed the max physical TDOA for the node baseline.  This rejects false
    # xcorr peaks.  Without xcorr, the coarse TDOA passes through to the
    # solver, which has its own outlier detection.
    if xcorr_used and max_xcorr_baseline_km > 0:
        max_tdoa_ns = max_xcorr_baseline_km * 1000.0 / _C_M_PER_S * 1e9
        if abs(tdoa_ns) > max_tdoa_ns:
            logger.warning(
                "TDOA implausible: %.1f ns > %.0f ns max for %s<->%s "
                "(%s, xcorr=%.1f, SNR=%.2f); pair skipped",
                tdoa_ns, max_tdoa_ns, node_a, node_b, event_type,
                xcorr_refinement_ns, xcorr_snr,
            )
            return None

    if xcorr_used:
        logger.info(
            "TDOA (sync_delta+xcorr): %.1f ns (coarse=%.1f + xcorr=%.1f, SNR=%.2f, type=%s) %s<->%s",
            tdoa_ns, coarse_tdoa_ns, xcorr_refinement_ns, xcorr_snr, event_type, node_a, node_b,
        )
    else:
        logger.info(
            "TDOA (sync_delta): %.1f ns (raw=%.1f + corr=%.1f, type=%s) %s<->%s",
            tdoa_ns, raw_ns, correction_ns, event_type, node_a, node_b,
        )
    return float(tdoa_ns / 1e9)
