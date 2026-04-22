# Per-Event Timing Refinement: Tuning Guide

This doc describes the server-side timing-refinement step (run inside
`compute_tdoa_s()` in `src/beagle_server/tdoa.py`) and the two parameters
that matter for field tuning.

> **Note (2026-04-21):** The server default is now `tdoa_method="xcorr"`
> (inter-node cross-correlation on `d²(power envelope)` — see
> `cross_correlate_snippets`). On the 2026-04-21 Magnolia corpus xcorr
> matches or beats the Savgol knee finder on per-pair std, has full event
> yield, and works for offsets (where per-snippet knee SNR is typically
> < 1). The Savgol-knee algorithm documented below is the `tdoa_method="knee"`
> path — still supported and test-covered, but no longer the default.

## Algorithm

For each carrier event the node ships an IQ snippet plus the `transition_start`
and `transition_end` indices bracketing where the detector fired.  The server
locates the **knee** of the PA transition -- the corner where the ramp meets
the plateau -- inside that bracket:

1. Compute the real-valued power envelope `|iq|^2`.
2. Apply a Savitzky-Golay filter to get the second derivative `d2` of the
   envelope (order 3 polynomial fit, time-specified window).
3. Within `[transition_start, transition_end]`, find `argmin(d2)`.  At a
   ramp-to-plateau corner the second derivative is strongly negative (the
   curve bends from a non-zero slope back to ~zero slope); this operator
   locates onset top-of-rise and offset start-of-fall with the same code
   path.
4. Parabolic interpolation on the three samples around the minimum gives
   sub-sample precision.
5. Report an SNR computed as `|min(d2)|` inside the region vs. RMS of `d2`
   outside it; use as a quality gate.

This replaced an earlier "plateau-anchored cross-correlation" method that
walked the power envelope forward from the detection point until the
derivative stabilised, as well as a `argmax(d1)` steepest-slope heuristic.
Both of those located the *middle* of the PA ramp, not the knee, so their
position slid with per-keyup ramp shape.  `argmin(d2)` is shape-invariant.

## Parameters

### `savgol_window_us = 360`

Savgol smoothing window in **microseconds** (auto-converted to an odd number
of samples at the snippet's rate).  Picking a time-domain value means the
same setting works across sample-rate changes (62.5 kHz vs 250 kHz).

Real-corpus results on the Magnolia repeater fixture (2026-04-19, 250 kHz):

| `savgol_window_us` | median |err| |
|-------------------:|--------------|
| 240                | 181 µs       |
| **360**            | **59 µs**    |
| 720                | 105 µs       |

- **Narrower** admits more plateau-noise fluctuation into d2, and the argmin
  starts chasing noise instead of the corner.
- **Wider** smears the corner feature across more samples; d2 flattens and
  the minimum loses its locator.

### `min_xcorr_snr = 0.5`

Minimum `|min(d2)| / rms(d2 outside region)` to accept a pair.  Name
retained for config compatibility with the old xcorr-based pipeline.

Empirical sweep on the same corpus:

| `min_xcorr_snr` | pairs kept | median |err| |
|----------------:|------------|--------------|
| **0.5**         | 19/23      | **59 µs**    |
| 1.0             | 17/23      | 82 µs        |
| 1.5             | 12/23      | 91 µs        |
| 2.0             |  8/23      | 114 µs       |

d2-SNR is a surprisingly poor predictor of knee accuracy -- tightening the
gate discards good pairs without improving the median.  The current floor
of 0.5 accepts essentially anything with a negative d2 peak while still
rejecting noise-only snippets.

## Tuning workflow

1. Run the server long enough to collect ≥30 paired events from a known
   geometry (e.g., a co-located pair or a known-position repeater like
   Magnolia).
2. Run `scripts/analyze_xcorr_tdoa.py` (or the ad-hoc analyser in
   `/private/tmp/analyze_new_corpus.py`) on the saved events; it pairs by
   `onset_time_ns` and reports median/p75 per window size.
3. If median |err| is > 100 µs, try 240 µs and 720 µs Savgol windows to
   confirm 360 µs is still the sweet spot for your signals.  A different
   transmitter class (much slower PA ramps, or pre-emphasis rollover) may
   prefer a wider window.
4. Drop the SNR gate if events are being rejected that show a clean knee
   in the raw envelope; raise it only if obvious noise-only events are
   being accepted.

## Reference values

| Parameter              | Default | Plausible range | Notes                    |
|------------------------|--------:|-----------------|--------------------------|
| `savgol_window_us`     |     360 | 240 - 720       | field-validated          |
| `min_xcorr_snr`        |     0.5 | 0.3 - 1.0       | field-validated          |
| `max_xcorr_baseline_km`|   100.0 | actual max km   | geometric plausibility   |

The `max_xcorr_baseline_km` guard rejects any TDOA whose magnitude exceeds
`baseline_km / c`.  Its behaviour is unchanged from the earlier xcorr
pipeline; set to your deployment's maximum inter-node distance.

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../LICENSE).
