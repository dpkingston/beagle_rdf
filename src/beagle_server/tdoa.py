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
from scipy.signal import resample_poly, savgol_filter

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

    def __init__(self, alpha: float = 0.2, min_samples: int = 3) -> None:
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

        # Need valid sample indices from both nodes.  Legacy nodes send 0.
        if sync_idx_a == 0.0 or sync_idx_b == 0.0:
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

        # Compute fractional-bit offset from sample indices at the NOMINAL
        # bit period (250000/1187.5 samples).  Do NOT apply the per-node
        # crystal correction here: empirically (190 paired dpk-tdoa1/tdoa2
        # events over 36 h) diff/SPB_nominal is within 1 µs of an integer,
        # while diff/SPB_crystal injects a spurious ~100 µs offset purely
        # as a numerical artifact of applying ppm-scale correction to
        # sample differences in the millions.  The real crystal rates of
        # the paired nodes are near-identical (std < 1.2 ppm over 36 h).
        sync_diff_samples = sync_idx_a - sync_idx_b
        rds_bit_samples = 250_000.0 / 1187.5
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


def _find_knee_sub_sample(
    iq: np.ndarray,
    event_type: str,
    transition_start: int,
    transition_end: int,
    savgol_window: int = 15,
    savgol_order: int = 3,
) -> tuple[float, float] | None:
    """
    Find the sub-sample position of the PA transition knee in a snippet.

    Uses Savitzky-Golay filter to compute the first derivative of the power
    envelope, then finds the argmin (offset) or argmax (onset) within the
    reported transition region.  Parabolic interpolation on the d1 peak
    gives sub-sample precision.

    Savgol is preferred over box-smooth-then-np.diff because it fits a
    polynomial across the window, preserving the *position* of sharp
    features.  Empirically (harness across 30 paired real snippets, Magnolia
    ground truth) this gets ~116 µs median TDOA error at 62.5 kHz vs ~198 µs
    with box-16 smoothing + np.diff.

    Parameters
    ----------
    iq : complex64 array, the snippet.
    event_type : "onset" or "offset".
    transition_start, transition_end : int
        Reported snippet positions bracketing the PA transition (both nodes
        report these based on their detector anchoring).
    savgol_window, savgol_order : Savgol filter parameters.

    Returns
    -------
    (knee_position_samples, snr) or None.
      knee_position_samples : float sub-sample position in the snippet.
      snr : peak |d1| within the transition region vs RMS of d1 outside it —
            analogous to xcorr SNR, usable as a confidence gate.
    Returns None if the snippet is too short or the transition window is empty.
    """
    n = len(iq)
    if n < savgol_window + 4:
        return None
    power = iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2
    d1 = savgol_filter(power, savgol_window, savgol_order, deriv=1, mode="nearest")

    lo = max(2, int(transition_start))
    hi = min(len(d1) - 2, int(transition_end))
    if hi <= lo + 2:
        return None

    d1_region = d1[lo:hi]
    if event_type == "onset":
        peak_rel = int(np.argmax(d1_region))
    else:
        peak_rel = int(np.argmin(d1_region))
    peak_idx = peak_rel + lo
    peak_val = float(d1[peak_idx])

    # Sub-sample parabolic interpolation on the d1 peak.
    if 0 < peak_idx < len(d1) - 1:
        y0 = float(d1[peak_idx - 1])
        y2 = float(d1[peak_idx + 1])
        denom = y0 - 2.0 * peak_val + y2
        sub = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
        sub = float(np.clip(sub, -0.5, 0.5))
        knee_pos = float(peak_idx) + sub
    else:
        knee_pos = float(peak_idx)

    # SNR: peak |d1| vs RMS of d1 samples OUTSIDE the transition region.
    mask = np.ones(len(d1), dtype=bool)
    mask[max(0, peak_idx - 3) : peak_idx + 4] = False
    noise = d1[mask]
    if len(noise) > 0:
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    else:
        noise_rms = 1e-30
    snr = abs(peak_val) / max(noise_rms, 1e-30)

    return knee_pos, snr


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
) -> tuple[float, float]:
    """
    Cross-correlate two int8 IQ snippets to find the inter-node TDOA.

    Method:
    1. Decode snippets, compute smoothed power envelopes.
    2. Walk the power curve from each node's detection point to coarsely
       locate the PA knee (to the nearest window).
    3. Cut equal-length sub-snippets centered on each coarse knee.
    4. Hold node A's sub-snippet fixed, xcorr node B's against it.
    5. The xcorr lag is the sub-sample shift needed to align the knees.
    6. Return the timing offset between the two knee positions:
       (knee_position_B - knee_position_A) in nanoseconds, where each
       knee_position = coarse_knee_sample + xcorr_refinement.

    Cross-correlates **power envelopes** (not raw IQ) so that LO phase
    differences between independent receivers do not affect the result.

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
        "onset" or "offset" — determines the direction of the knee walk.

    Returns
    -------
    (lag_ns, corr_snr)
        lag_ns : float
            Timing offset in nanoseconds.  This is the difference between
            the two nodes' knee positions within their respective snippets.
            Add to (sync_delta_A - sync_delta_B) to get the full TDOA.
        corr_snr : float
            Peak-to-sidelobe ratio of the cross-correlation (dimensionless).
    """
    rate_b = sample_rate_hz_b if sample_rate_hz_b is not None else sample_rate_hz_a
    effective_rate = target_rate_hz if target_rate_hz is not None else min(sample_rate_hz_a, rate_b)

    a = _decode_iq_snippet(a_b64)
    b = _decode_iq_snippet(b_b64)

    # Power envelopes (phase-free for independent LO receivers).
    env_a = _compute_power_envelope(a).astype(np.float32)
    env_b = _compute_power_envelope(b).astype(np.float32)

    # Resample to a common rate if the two nodes captured at different rates.
    if abs(sample_rate_hz_a - effective_rate) / effective_rate > 1e-6:
        env_a = _resample_to_rate(env_a, sample_rate_hz_a, effective_rate)
    if abs(rate_b - effective_rate) / effective_rate > 1e-6:
        env_b = _resample_to_rate(env_b, rate_b, effective_rate)

    if float(np.max(env_a)) < 1e-4 or float(np.max(env_b)) < 1e-4:
        return 0.0, 0.0

    # Second derivative of the power envelope.
    #
    # The PA transition is a smooth monotonic ramp (constant-envelope FM
    # carrier with PA power ramping up or down).  The power envelope and
    # its first derivative are broad, featureless curves that xcorr cannot
    # resolve at sub-sample precision.
    #
    # The second derivative has sharp, distinctive features at the
    # inflection points of the transition — zero-crossings where the
    # ramp curvature changes sign.  These are physical properties of the
    # PA electronics, identical across all receivers.  xcorr on the second
    # derivative achieves sub-microsecond precision (SNR ~28 on real data).
    d2_a = np.diff(np.diff(env_a))
    d2_b = np.diff(np.diff(env_b))

    min_len = min(len(d2_a), len(d2_b))
    if min_len < 16:
        return 0.0, 0.0

    return _xcorr_arrays(d2_a[:min_len], d2_b[:min_len], effective_rate)


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

    Method
    ------
    1. Compute raw_ns = sync_delta_a - sync_delta_b (sync-to-detection diff)
    2. Apply per-pair grid calibration (removes node-pair pilot phase offset)
    3. Find the PA transition *knee* in each snippet using Savgol-smoothed
       first derivative (argmin for offset, argmax for onset) within the
       reported transition region.  Gives sub-sample position.
    4. Compute knee_adj_ns = (knee_a - det_a)/rate_a - (knee_b - det_b)/rate_b,
       which shifts each node's reference from detection to the physical PA
       edge.  Detection position comes from ``transition_start``/``transition_end``.
    5. Apply path correction (sync-transmitter geometry)
    6. Disambiguate modulo RDS bit period.
    7. Geometric plausibility check against max_xcorr_baseline_km.

    Previously used inter-node cross-correlation on d2-of-envelope (see
    ``cross_correlate_snippets``), but empirically that pipeline was dominated
    by boundary artifacts of the zero-pad convolution, locking xcorr on the
    wrong feature.  Per-node knee finding with Savgol avoids this and gives
    ~40% lower median per-event error on a corpus of real Magnolia fixes.

    Parameters
    ----------
    event_a, event_b : dicts with keys:
        sync_delta_ns, sync_tx_lat, sync_tx_lon, node_lat, node_lon,
        event_type, node_id, iq_snippet_b64, channel_sample_rate_hz,
        transition_start, transition_end.
        onset_time_ns (optional): wall-clock time of the carrier edge in ns.
    min_xcorr_snr : float
        Minimum knee-finder SNR (peak |d1| vs out-of-region RMS of d1) to
        accept the result.  Pairs failing this gate are dropped.
        Retains the old parameter name for config compatibility.
    xcorr_target_rate_hz : float or None
        Currently unused (retained for API compatibility).  Previously used
        to resample envelopes before inter-node xcorr.
    max_xcorr_baseline_km : float
        Maximum node-pair separation in km.  Used as a geometric plausibility
        filter: any TDOA whose magnitude exceeds (baseline / c) after
        disambiguation is treated as a false detection and rejected.

    Returns
    -------
    float or None
        TDOA in seconds, or None on missing data, failed knee finding,
        low SNR, or geometric implausibility.  Positive -> A heard the
        carrier *later* than B.
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
            # Sync sample difference in RDS bit periods, at nominal rate
            # (no crystal correction — see SyncCalibrator for rationale).
            # Should be near an integer if both nodes are counting bit
            # boundaries consistently.
            sync_diff_samples = sync_idx_a - sync_idx_b
            rds_bit_samples = 250_000.0 / 1187.5
            sync_diff_bits = sync_diff_samples / rds_bit_samples
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

    # --- Server-side knee finding ---
    #
    # Each node's `sync_delta_ns` measures from a pilot bit boundary to
    # its *detection point* (threshold crossing), which varies per event
    # by hundreds of µs due to noise floor and instantaneous signal level.
    #
    # We refine to the *knee* — the steepest point of the PA transition
    # — using Savitzky-Golay first-derivative on each snippet's power
    # envelope.  The knee is a physical property of the transmitter PA,
    # observed at essentially the same instant by all receivers (minus
    # propagation delay), so (knee_A - knee_B) gives the true inter-node
    # TDOA modulo bit periods.
    #
    # Empirically this gives ~116 µs per-event median error at 62.5 kHz
    # (vs ~172 µs with the older inter-node xcorr on d2-of-envelope).
    # Quick scaling check suggests this drops to ~30-60 µs at 250 kHz.
    iq_a_b64 = event_a.get("iq_snippet_b64", "")
    iq_b_b64 = event_b.get("iq_snippet_b64", "")
    if not iq_a_b64 or not iq_b_b64:
        logger.warning(
            "Missing IQ snippet for %s<->%s (%s); pair skipped",
            node_a, node_b, event_type,
        )
        return None

    rate_a = float(event_a.get("channel_sample_rate_hz", 64_000.0))
    rate_b = float(event_b.get("channel_sample_rate_hz", 64_000.0))

    # Detection position within each node's snippet — needed to convert
    # the knee position into a "sync_to_knee" time.
    #   For onset: detection fires at rising-edge threshold, which sits at
    #     ``transition_start`` in the encoded snippet.
    #   For offset: detection fires at falling-edge threshold, at
    #     ``transition_end``.
    det_a = int(event_a.get(
        "transition_start" if event_type == "onset" else "transition_end", 0
    ))
    det_b = int(event_b.get(
        "transition_start" if event_type == "onset" else "transition_end", 0
    ))
    ts_a = int(event_a.get("transition_start", 0))
    te_a = int(event_a.get("transition_end", 0))
    ts_b = int(event_b.get("transition_start", 0))
    te_b = int(event_b.get("transition_end", 0))

    iq_a = _decode_iq_snippet(iq_a_b64)
    iq_b = _decode_iq_snippet(iq_b_b64)

    knee_a_result = _find_knee_sub_sample(iq_a, event_type, ts_a, te_a)
    knee_b_result = _find_knee_sub_sample(iq_b, event_type, ts_b, te_b)
    if knee_a_result is None or knee_b_result is None:
        logger.warning(
            "Knee finding failed for %s<->%s (%s); pair skipped",
            node_a, node_b, event_type,
        )
        return None
    knee_a, snr_a = knee_a_result
    knee_b, snr_b = knee_b_result

    # SNR gate: reject when either node's knee is not clearly distinguishable
    # from envelope noise.  Reuses the min_xcorr_snr config so the existing
    # threshold continues to apply (semantics are similar — peak |d1| vs
    # out-of-region RMS).
    min_snr = min(snr_a, snr_b)
    if min_snr < min_xcorr_snr:
        logger.warning(
            "Knee SNR too low: min(%.2f, %.2f) < %.2f for %s<->%s (%s); pair skipped",
            snr_a, snr_b, min_xcorr_snr, node_a, node_b, event_type,
        )
        return None

    # Convert each node's knee position to a (knee − detection) offset in ns,
    # then take the inter-node difference.  This is what we add to raw_ns to
    # shift the measurement from "sync-to-detection" to "sync-to-knee".
    knee_adj_a_ns = (knee_a - det_a) * 1e9 / rate_a
    knee_adj_b_ns = (knee_b - det_b) * 1e9 / rate_b
    knee_adj_ns = knee_adj_a_ns - knee_adj_b_ns

    # Combine: raw sync_delta (sync-to-detection) + knee adjustment +
    # FM path correction.  The knee adjustment moves each node's reference
    # from its noisy detection point to the physical PA edge.
    combined_ns = raw_ns + knee_adj_ns + correction_ns

    # Sync period disambiguation: resolve which RDS bit boundary each node
    # referenced.  Applied AFTER the knee adjustment so the round() decision
    # is based on the refined TDOA, not the detection-jitter-dominated
    # coarse value.
    n = round(combined_ns / _T_SYNC_NS)
    if n != 0:
        logger.debug(
            "Sync disambiguation %s<->%s (%s): n=%+d combined=%+.0f->%+.0f ns",
            node_a, node_b, event_type, n, combined_ns, combined_ns - n * _T_SYNC_NS,
        )
        combined_ns -= n * _T_SYNC_NS

    tdoa_ns = combined_ns

    # Geometric plausibility: total TDOA must not exceed max physical TDOA
    # for the node baseline.
    if max_xcorr_baseline_km > 0:
        max_tdoa_ns = max_xcorr_baseline_km * 1000.0 / _C_M_PER_S * 1e9
        if abs(tdoa_ns) > max_tdoa_ns:
            logger.warning(
                "TDOA implausible: %.1f ns > %.0f ns max for %s<->%s "
                "(%s, knee_adj=%.1f, SNR=[%.2f,%.2f]); pair skipped",
                tdoa_ns, max_tdoa_ns, node_a, node_b, event_type,
                knee_adj_ns, snr_a, snr_b,
            )
            return None

    coarse_tdoa_ns = raw_ns + correction_ns
    logger.info(
        "TDOA (savgol_knee): %.1f ns (coarse=%.1f + knee_adj=%.1f, "
        "SNR=[%.2f,%.2f], type=%s) %s<->%s",
        tdoa_ns, coarse_tdoa_ns, knee_adj_ns, snr_a, snr_b, event_type,
        node_a, node_b,
    )
    return float(tdoa_ns / 1e9)
