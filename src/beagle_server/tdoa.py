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

_C_M_S = 299_792_458.0  # speed of light, m/s


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
) -> tuple[float, float]:
    """
    Cross-correlate two int8 IQ snippets to find the inter-node TDOA.

    Cross-correlates the **power envelopes** (not the raw IQ) so that LO phase
    differences between independent receivers do not produce false correlation
    peaks.  Uses FFT-based correlation with parabolic peak interpolation for
    sub-sample precision.

    If the two snippets were captured at different sample rates (e.g. RTL-SDR
    at 64 kHz and RSPduo at 62.5 kHz), one envelope is resampled to the target
    rate before correlation so the correlation peak position is correctly
    interpreted in time.  The target rate defaults to the lower of the two
    input rates (prefer downsampling; no interpolated data introduced).

    The correlation is restricted to the half of the snippet that contains the
    PA transition, based on *event_type*:

    - "onset":  PA rise sits near the centre of the snippet.
                The first 3/4 is used (noise + transition + carrier prefix).
                Avoids false peaks from voice-modulated carrier in final quarter.
    - "offset": PA fall is anchored at ~3/4 of the snippet length.
                Only the second half is used (brief carrier + transition + noise).
    - "":       No trimming (full snippet).  For synthetic test signals.

    Parameters
    ----------
    a_b64, b_b64 : str
        Base64-encoded int8-interleaved IQ snippets (see CarrierEvent.iq_snippet_b64).
    sample_rate_hz_a : float
        Sample rate of snippet A (default 64 kHz = RTL-SDR target decimation rate).
    sample_rate_hz_b : float or None
        Sample rate of snippet B.  None means same as A (no resampling).
    target_rate_hz : float or None
        Rate to resample both snippets to before correlation.  None = use the
        lower of the two input rates (recommended).
    event_type : str
        "onset" or "offset" to enable transition windowing (recommended).
        Empty string uses the full snippet (for synthetic tests).

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

    # Trim to the transition region to avoid false peaks from voice-modulated
    # carrier content.  The PA transition sits near the centre (~1/2) of the
    # onset snippet and at ~3/4 of the offset snippet.
    #
    # Onset: use the first 3/4 of the snippet.  This brackets the transition
    # at ~1/2 with noise before and carrier after, while excluding the
    # voice-modulated carrier in the final quarter.
    # Offset: use the second half, which contains the carrier-to-noise drop.
    if event_type == "onset":
        env_a = env_a[: 3 * len(env_a) // 4]
        env_b = env_b[: 3 * len(env_b) // 4]
    elif event_type == "offset":
        env_a = env_a[len(env_a) // 2 :]
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
    if delta_a is None or delta_b is None:
        logger.warning(
            "Missing sync_delta_ns for %s<->%s (%s) - pair skipped",
            node_a, node_b, event_type,
        )
        return None

    raw_ns = float(delta_a) - float(delta_b)

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

    # --- xcorr: sub-sample refinement when IQ snippets are available ---
    #
    # The xcorr lag measures the time offset between the PA transition
    # positions in the two nodes' snippets.  Both snippets are anchored
    # to the detected transition (onset at 25%, offset at 75%), so xcorr
    # should produce a small lag (~few µs) that refines the coarse
    # sync_delta-based TDOA.
    #
    # If the xcorr lag is large (> max_refinement), it means the
    # sync_delta_ns and the snippet anchor are misaligned — the coarse
    # TDOA is unreliable for this pair.  Rather than falling back to
    # the noisy sync_delta value (which can scatter fixes by 100+ km),
    # we return None so the solver works with fewer pairs or skips the
    # event entirely.
    _MAX_XCORR_REFINEMENT_NS = 50_000.0  # 50 usec ~ 3 samples at 62.5 kHz

    xcorr_refinement_ns = 0.0
    xcorr_used = False
    iq_a = event_a.get("iq_snippet_b64", "")
    iq_b = event_b.get("iq_snippet_b64", "")
    if iq_a and iq_b:
        rate_a = float(event_a.get("channel_sample_rate_hz", 64_000.0))
        rate_b = float(event_b.get("channel_sample_rate_hz", 64_000.0))
        xcorr_lag_ns, xcorr_snr = cross_correlate_snippets(
            iq_a, iq_b,
            sample_rate_hz_a=rate_a,
            sample_rate_hz_b=rate_b,
            target_rate_hz=xcorr_target_rate_hz,
            event_type=event_type,
        )
        if xcorr_snr >= min_xcorr_snr:
            if abs(xcorr_lag_ns) <= _MAX_XCORR_REFINEMENT_NS:
                xcorr_refinement_ns = xcorr_lag_ns
                xcorr_used = True
            else:
                logger.warning(
                    "xcorr refinement too large: %.1f ns > %.0f ns limit "
                    "for %s<->%s (%s, SNR=%.2f); pair skipped "
                    "(sync_delta/snippet anchor misalignment)",
                    xcorr_lag_ns, _MAX_XCORR_REFINEMENT_NS,
                    node_a, node_b, event_type, xcorr_snr,
                )
                return None
        else:
            logger.debug(
                "xcorr SNR too low: %.2f < %.2f for %s<->%s (%s); "
                "using sync_delta only",
                xcorr_snr, min_xcorr_snr, node_a, node_b, event_type,
            )

    tdoa_ns = coarse_tdoa_ns + xcorr_refinement_ns

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
