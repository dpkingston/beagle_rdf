# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
TDOA arithmetic for the aggregation server.

Core operations:
  - haversine_m: great-circle distance between two lat/lon points
  - compute_tdoa_s: sync_to_snippet_start subtraction + server-side knee offset
    + sync-path-geometry correction

Timing model (schema v1.5+)
---------------------------
Each node ships ``sync_to_snippet_start_ns`` — the time on its local sample
clock from the matched sync event to the FIRST sample of the shipped IQ
snippet.  The node's detection threshold is merely a trigger for packaging;
the reference the server consumes is the snippet's first sample (an exact
sample boundary on the node's clock).

The full sync-to-knee time per node is:

    sync_to_knee_n = sync_to_snippet_start_n + knee_position_in_snippet_n / rate

where ``knee_position_in_snippet_n`` is located by the server's knee-finder,
using ``transition_start`` / ``transition_end`` as a search hint.

Derivation of the sync-path correction
--------------------------------------
Per-node sync-to-knee:

    sync_to_knee_n = [T_t + dist(target, n)/c] - [T_s + dist(sync, n)/c]
                   = K + [dist(target, n) - dist(sync, n)] / c

where K = T_t - T_s is a common constant across nodes for the same event.

Raw inter-node difference (A minus B):

    raw_ns = sync_to_knee_A - sync_to_knee_B
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
    sample_rate_hz: float = 62_500.0,
    savgol_window_us: float = 360.0,
    savgol_order: int = 3,
    first_peak_sig_ratio: float = 0.50,
) -> tuple[float, float] | None:
    """
    Find the sub-sample position of the PA transition knee in a snippet.

    The knee is the corner where the ramp meets the plateau:
      - ONSET  = top-of-rise   (ramp up  -> flat high plateau)
      - OFFSET = start-of-fall (flat high plateau -> ramp down)

    At either corner the power envelope has its most-negative second
    derivative (strong concave-down curvature as the curve bends from
    a non-zero slope back to approximately zero slope).

    Algorithm (first-peak-after-ramp):

    1. Seed with the ramp's middle: argmax(d1) for onset (steepest rise)
       or argmin(d1) for offset (steepest fall).  This is a robust
       locator of "where the ramp is happening" even under noise.
    2. Walk into the plateau side of the seed (forward for onset,
       backward for offset) and return the **first** local minimum of
       d2 whose magnitude exceeds ``first_peak_sig_ratio`` times the
       region's largest |min(d2)|.

    Why this avoids the bias the earlier global-argmin algorithm had:
    d2 has a strong negative peak at the real knee, but same-sign noise
    in the plateau can produce smaller negative peaks further out along
    the flat.  A pure argmin(d2) over the whole reported transition
    region can latch onto one of those plateau-noise peaks, biasing the
    TDOA and producing per-node systematic offsets because each node's
    RF chain shapes the leading/trailing edge differently.  By stopping
    at the first significant valley beyond the ramp's middle we lock to
    the structural corner — the knee — rather than whichever plateau
    noise later happens to dip deepest.

    If no qualifying local minimum exists on the plateau side (e.g.
    the corner fell outside the reported transition region, or the
    signal has no structural d2 peak at all) we return None so the pair
    is rejected.  Falling back to global argmin(d2) here would
    reintroduce exactly the plateau-noise bias this algorithm exists
    to avoid.

    Parabolic interpolation on the d2 minimum gives sub-sample precision.

    The Savgol window is specified in TIME (µs) so it auto-adapts to the
    snippet sample rate — 360 µs is ~23 samples at 62.5 kHz and ~90
    samples at 250 kHz, preserving the same smoothing bandwidth across
    rates.

    Parameters
    ----------
    iq : complex64 array, the snippet.
    event_type : "onset" or "offset".
    transition_start, transition_end : int
        Reported snippet positions bracketing the PA transition (both nodes
        report these based on their detector anchoring).
    sample_rate_hz : float
        Snippet sample rate.  Used to convert savgol_window_us to samples.
    savgol_window_us : float
        Savgol window width in microseconds.
    savgol_order : int
        Savgol polynomial order.  Must be >= 2 for a d2 output; 3 is a
        good default.
    first_peak_sig_ratio : float
        Magnitude threshold for a d2 local minimum to count as "the knee",
        expressed as a fraction of the region's largest |min(d2)|.

        Empirically scanned on two Magnolia corpora (2026-04-20 16:30 and
        2026-04-21 09:10): on both, increasing the ratio from 0.25 to
        0.50 cut algorithm-addressable per-pair bias roughly in half
        (~50 µs -> ~25 µs on clean pairs) with no yield loss on the
        newer corpus and a modest drop from 52% -> 42% on the older one.
        0.50 is now the default.

        Lower = admit smaller d2 dips as knee candidates, which keeps
        more events but risks latching onto plateau noise (the failure
        mode the first-peak algorithm was introduced to avoid).
        Higher = reject more events but retain only strong-structure
        knees.  0.0 disables the gate entirely (any d2 local minimum
        qualifies).

    Returns
    -------
    (knee_position_samples, snr) or None.
      knee_position_samples : float sub-sample position in the snippet.
      snr : |d2 at knee| vs RMS of d2 samples outside the peak region,
            analogous to xcorr SNR and usable as a confidence gate.
    Returns None if the snippet is too short or the transition window is empty.
    """
    # Convert the desired smoothing duration (µs) to an odd sample count.
    window = max(savgol_order + 2,
                 int(round(savgol_window_us * sample_rate_hz / 1e6)))
    if window % 2 == 0:
        window += 1
    n = len(iq)
    if n < window + 4:
        return None
    power = iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2
    d1 = savgol_filter(power, window, savgol_order, deriv=1, mode="nearest")
    d2 = savgol_filter(power, window, savgol_order, deriv=2, mode="nearest")

    lo = max(2, int(transition_start))
    hi = min(len(d2) - 2, int(transition_end))
    if hi <= lo + 4:
        return None

    d1_region = d1[lo:hi]
    d2_region = d2[lo:hi]

    # Significance threshold relative to the strongest d2 negative peak
    # anywhere in the region (not just the plateau side).  If the region
    # has no negative d2 at all there is no corner to find.
    d2_min_mag = float(-d2_region.min())
    if d2_min_mag <= 0.0:
        return None
    threshold = first_peak_sig_ratio * d2_min_mag

    if event_type == "onset":
        ramp_mid = int(np.argmax(d1_region))
        # Walk forward from ramp_mid into the plateau side; take the
        # first local d2 minimum past the significance threshold.
        knee_rel: int | None = None
        for k in range(ramp_mid + 1, len(d2_region) - 1):
            if (d2_region[k] < -threshold
                    and d2_region[k] <= d2_region[k - 1]
                    and d2_region[k] <= d2_region[k + 1]):
                knee_rel = k
                break
    else:
        ramp_mid = int(np.argmin(d1_region))
        knee_rel = None
        for k in range(ramp_mid - 1, 0, -1):
            if (d2_region[k] < -threshold
                    and d2_region[k] <= d2_region[k - 1]
                    and d2_region[k] <= d2_region[k + 1]):
                knee_rel = k
                break

    # No qualifying local minimum → reject.  A global-argmin fallback
    # would reintroduce the plateau-noise bias this algorithm exists to
    # avoid.
    if knee_rel is None:
        return None

    peak_idx = knee_rel + lo
    peak_val = float(d2[peak_idx])

    # Sub-sample parabolic interpolation on the d2 minimum.
    if 0 < peak_idx < len(d2) - 1:
        y0 = float(d2[peak_idx - 1])
        y2 = float(d2[peak_idx + 1])
        denom = y0 - 2.0 * peak_val + y2
        sub = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
        sub = float(np.clip(sub, -0.5, 0.5))
        knee_pos = float(peak_idx) + sub
    else:
        knee_pos = float(peak_idx)

    # SNR: |peak d2| vs RMS of d2 samples OUTSIDE the peak neighbourhood.
    mask = np.ones(len(d2), dtype=bool)
    mask[max(0, peak_idx - 3): peak_idx + 4] = False
    noise = d2[mask]
    noise_rms = float(np.sqrt(np.mean(noise ** 2))) if len(noise) else 1e-30
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
    min_xcorr_snr: float = 0.5,
    xcorr_target_rate_hz: float | None = None,
    max_xcorr_baseline_km: float = 100.0,
    savgol_window_us: float = 360.0,
    tdoa_method: str = "xcorr",
) -> float | None:
    """
    Compute the corrected TDOA between two events in **seconds**.

    Both events must be from the same transmission (same channel, event_type,
    and sync transmitter).  The path-delay correction is applied using the
    sync transmitter coordinates and node locations carried in each event.

    Method
    ------
    1. Compute raw_ns = sync_to_snippet_start_a - sync_to_snippet_start_b
       (difference in the node-side sync -> snippet-start time).
    2. Apply per-pair grid calibration (removes node-pair pilot phase offset).
    3. Find the inter-node knee offset (per ``tdoa_method``):
         "xcorr" (default): inter-node cross-correlation on the second
           derivative of each snippet's power envelope.  Returns the inter-
           node knee time difference as a single lag.  Works even when each
           node's individual knee SNR is low.  Empirically best for offsets
           (knee SNR typically << 1) and matches per-node knee finding for
           onsets with full event yield.
         "knee": per-node Savgol-smoothed second-derivative knee finder.
           Knee positions (samples from snippet-start) are converted to ns
           and differenced.  Retained for comparison / tests.
    4. Apply sync-path correction (sync-transmitter geometry).
    5. Disambiguate modulo RDS bit period.
    6. Geometric plausibility check against max_xcorr_baseline_km.

    Parameters
    ----------
    event_a, event_b : dicts with keys:
        sync_to_snippet_start_ns, sync_tx_lat, sync_tx_lon, node_lat, node_lon,
        event_type, node_id, iq_snippet_b64, channel_sample_rate_hz,
        transition_start, transition_end.
        onset_time_ns (optional): wall-clock time of the carrier edge in ns.
    min_xcorr_snr : float
        Minimum refinement-SNR required to accept the pair.  For "xcorr" this
        is the peak-to-sidelobe ratio returned by ``cross_correlate_snippets``;
        for "knee" it is ``|peak(d1)|`` vs out-of-region RMS.  Retained as a
        single parameter name for config compatibility.
    xcorr_target_rate_hz : float or None
        Target rate for envelope resampling inside ``cross_correlate_snippets``.
        None (default) uses the lower of the two snippet rates.  Only applies
        when ``tdoa_method="xcorr"``.
    max_xcorr_baseline_km : float
        Maximum node-pair separation in km.  Used as a geometric plausibility
        filter: any TDOA whose magnitude exceeds (baseline / c) after
        disambiguation is treated as a false detection and rejected.
    savgol_window_us : float
        Only used when ``tdoa_method="knee"``.  Savgol smoothing window in
        microseconds (auto-converted to an odd number of samples at each
        snippet's rate).
    tdoa_method : "xcorr" or "knee"
        Pair-level refinement method.  Defaults to "xcorr" based on empirical
        comparison on the 2026-04-21 Magnolia corpus (xcorr has equal-or-lower
        per-pair std, full event yield, and works for offsets where the per-
        snippet knee SNR is < 1).

    Returns
    -------
    float or None
        TDOA in seconds, or None on missing data, failed refinement,
        low SNR, or geometric implausibility.  Positive -> A heard the
        carrier *later* than B.
    """
    node_a = event_a.get("node_id", "?")
    node_b = event_b.get("node_id", "?")
    event_type = event_a.get("event_type", "")

    # --- sync_to_snippet_start: coarse TDOA from the shipped node timing ---
    # Each node reports time from the matched sync event to the first sample
    # of its shipped IQ snippet, on the node's local (crystal-corrected)
    # sample clock.  The detection point is NOT in this value — it's only a
    # hint that the node uses to decide when to package up the snippet.
    delta_a = event_a.get("sync_to_snippet_start_ns")
    delta_b = event_b.get("sync_to_snippet_start_ns")

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
                "sync_diag %s<->%s (%s): sync_to_snippet_start_ns=[%s, %s]  "
                "delta_samp=[%.1f, %.1f]  sync_idx=[%.1f, %.1f]  "
                "sync_diff=%.1f samp (%.2f bits, frac=%.4f)  crystal=[%.8f, %.8f]",
                node_a, node_b, event_type,
                delta_a, delta_b, dsamp_a, dsamp_b,
                sync_idx_a, sync_idx_b,
                sync_diff_samples, sync_diff_bits, sync_diff_frac,
                corr_a, corr_b,
            )

    if delta_a is None or delta_b is None:
        logger.warning(
            "Missing sync_to_snippet_start_ns for %s<->%s (%s) - pair skipped",
            node_a, node_b, event_type,
        )
        return None

    raw_ns = float(delta_a) - float(delta_b)

    # Apply sync grid calibration: remove the per-pair fractional-bit
    # offset from the coarse TDOA.  Uses sync_sample_index (the pilot-
    # bit-grid position on each node) to measure the offset without
    # carrier-side contamination.  Applied BEFORE disambiguation so
    # rounding picks the correct bit period.
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

    # --- Knee locatator within each snippet ---
    #
    # The node's ``sync_to_snippet_start_ns`` anchors sync to the first sample
    # of the shipped snippet.  The KNEE — the corner where the PA ramp meets
    # the plateau (top-of-rise for onset, start-of-fall for offset) — is a
    # physical property of the transmission observed at essentially the same
    # instant by all receivers (minus propagation delay).  The server locates
    # the knee in each snippet independently and uses its position (in samples
    # from the snippet start) to compute sync -> knee per node:
    #
    #     sync_to_knee_n = sync_to_snippet_start_n + knee_position_n / rate_n
    #
    # Inter-node:
    #     TDOA_AB = (sync_to_knee_A - sync_to_knee_B) + sync-path correction
    #             = raw_ns + (knee_A / rate_A - knee_B / rate_B) + correction
    #
    # where raw_ns = sync_to_snippet_start_A - sync_to_snippet_start_B.
    # ``transition_start`` / ``transition_end`` are knee-search hints only.
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
    ts_a = int(event_a.get("transition_start", 0))
    te_a = int(event_a.get("transition_end", 0))
    ts_b = int(event_b.get("transition_start", 0))
    te_b = int(event_b.get("transition_end", 0))

    if tdoa_method == "xcorr":
        # Inter-node cross-correlation on d²(power envelope).  Returns a
        # lag in ns equal to (knee_A_in_snippet - knee_B_in_snippet) expressed
        # as time.  See ``cross_correlate_snippets``.
        lag_ns, xcorr_snr = cross_correlate_snippets(
            iq_a_b64, iq_b_b64,
            sample_rate_hz_a=rate_a, sample_rate_hz_b=rate_b,
            target_rate_hz=xcorr_target_rate_hz, event_type=event_type,
        )
        if xcorr_snr < min_xcorr_snr:
            logger.warning(
                "Xcorr SNR too low: %.2f < %.2f for %s<->%s (%s); pair skipped",
                xcorr_snr, min_xcorr_snr, node_a, node_b, event_type,
            )
            return None
        refinement_ns = lag_ns
        snr_a = snr_b = xcorr_snr
        refinement_desc = "xcorr_lag"
    elif tdoa_method == "knee":
        iq_a = _decode_iq_snippet(iq_a_b64)
        iq_b = _decode_iq_snippet(iq_b_b64)

        knee_a_result = _find_knee_sub_sample(
            iq_a, event_type, ts_a, te_a,
            sample_rate_hz=rate_a, savgol_window_us=savgol_window_us,
        )
        knee_b_result = _find_knee_sub_sample(
            iq_b, event_type, ts_b, te_b,
            sample_rate_hz=rate_b, savgol_window_us=savgol_window_us,
        )
        if knee_a_result is None or knee_b_result is None:
            logger.warning(
                "Knee finding failed for %s<->%s (%s); pair skipped",
                node_a, node_b, event_type,
            )
            return None
        knee_a, snr_a = knee_a_result
        knee_b, snr_b = knee_b_result

        min_snr = min(snr_a, snr_b)
        if min_snr < min_xcorr_snr:
            logger.warning(
                "Knee SNR too low: min(%.2f, %.2f) < %.2f for %s<->%s (%s); pair skipped",
                snr_a, snr_b, min_xcorr_snr, node_a, node_b, event_type,
            )
            return None

        # knee_{a,b} are positions (samples) within each snippet, measured
        # from the snippet's first sample.  Converting each to ns and taking
        # the difference gives the inter-node knee offset directly — no
        # detection-point subtraction needed.
        refinement_ns = knee_a * 1e9 / rate_a - knee_b * 1e9 / rate_b
        refinement_desc = "knee_diff"
    else:
        raise ValueError(f"tdoa_method must be 'xcorr' or 'knee', got {tdoa_method!r}")

    # Combine: sync_to_snippet_start diff + inter-node knee offset +
    # sync-transmitter path-geometry correction.
    combined_ns = raw_ns + refinement_ns + correction_ns

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
                "(%s, %s=%.1f, SNR=[%.2f,%.2f]); pair skipped",
                tdoa_ns, max_tdoa_ns, node_a, node_b, event_type,
                refinement_desc, refinement_ns, snr_a, snr_b,
            )
            return None

    coarse_tdoa_ns = raw_ns + correction_ns
    logger.info(
        "TDOA (%s): %.1f ns (coarse=%.1f + %s=%.1f, "
        "SNR=[%.2f,%.2f], type=%s) %s<->%s",
        tdoa_method, tdoa_ns, coarse_tdoa_ns, refinement_desc, refinement_ns,
        snr_a, snr_b, event_type, node_a, node_b,
    )
    return float(tdoa_ns / 1e9)
