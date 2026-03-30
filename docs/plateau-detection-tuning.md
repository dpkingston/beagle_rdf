# Plateau Detection Tuning Guide

Plateau-anchored cross-correlation finds the moment a carrier rises to (or falls
from) its steady-state level - a property of the transmitter, not the receiver.
This makes it independent of receiver gain differences between nodes, which is the
primary source of error in threshold-crossing TDOA.

Four parameters control the algorithm (`src/beagle_server/tdoa.py`,
`compute_tdoa_s()`). The defaults were set heuristically on synthetic signals.
Once field deployments accumulate data, use the diagnostics below to tune them.

---

## Parameters and what to watch for

### `smooth_samples = 16`

Box-filter half-width applied to the power envelope before derivative computation.
At the 64 kHz target rate, 16 samples = 250 usec of smoothing.

**Increase if** you see `_find_plateau_sample` returning spurious plateau indices
on signals that look stable in the envelope (noise spikes triggering a false
"peak derivative"). In logs: plateau sharpness is high but the resulting TDOA
has high variance across pairs for the same transmitter.

**Decrease if** fast-ramp transmitters (< 1 ms rise time) are not being detected
(`_find_plateau_sample` returns None more than ~20% of the time for clean signals).
Excessive smoothing blurs sharp transitions below the `min_consecutive` threshold.

**Data to collect:** For each transmitter type in the deployment, run
`colocated_pair_test.py --analyze-snippets` and note the `transition_w` values
(in windows, each window = `window_samples / sample_rate_hz`). If median
transition time < 4 x smooth_samples / sample_rate_hz, reduce smooth_samples.

---

### `stable_fraction = 0.15`

A sample is considered "in the plateau" when its derivative magnitude is less than
15% of the peak derivative in the snippet. The peak derivative is assumed to
correspond to the steepest point of the carrier ramp.

**Increase if** you see plateau indices detected mid-ramp rather than at the top
(TDOA histograms for co-located nodes show a consistent non-zero bias that varies
by transmitter, not by node pair). The plateau is being declared too early.

**Decrease if** plateau detection fails on signals with clean, fast transitions
because the stable plateau region never falls below the current fraction threshold.
Symptom: `_find_plateau_sample` returns None for signals where the carrier rise is
clearly visible in the snippet power envelope.

**Data to collect:** After a field deployment accumulates > 100 events per
transmitter, plot the distribution of `plt_sample` values (plateau index within
snippet). A healthy distribution clusters tightly around the true plateau start.
If the distribution is bimodal or has a long tail toward snippet center, the
fraction threshold may be too high.

---

### `min_consecutive = 3`

Number of consecutive windows that must all satisfy `stable_fraction` before the
plateau is confirmed. At 64 kHz and `window_samples=64`, 3 windows = 3 ms.

**Increase if** brief mid-ramp amplitude fluctuations (e.g. from multipath) are
triggering false plateau detections before the carrier actually stabilizes. Symptom:
TDOA variance for co-located nodes is reduced relative to threshold-crossing but
still > 50 usec std for clean LOS paths.

**Decrease if** short transmissions (PTT < 200 ms) frequently yield no plateau
detection because the carrier never stays stable long enough. Symptom: high
`_find_plateau_sample` None rate for short transmissions only.

**Data to collect:** Filter the event database by transmission duration
(offset_sample - onset_sample). Separately compute plateau detection success rate
for short (< 100 ms) vs. long (> 500 ms) transmissions. If short transmissions fail
disproportionately, reduce min_consecutive.

---

### `min_plateau_sharpness = 3.0`

Minimum ratio of peak derivative to median derivative in the snippet. Rejects
snippets where the "ramp" is not clearly sharper than background noise - i.e.,
gradual-onset transmitters or noise-only captures.

**Increase if** you see xcorr results with high SNR (> min_xcorr_snr) but wildly
scattered TDOA values across pairs. This means plateau was detected on a gradual
ramp where the "peak" is only marginally above the noise floor, so small noise
fluctuations shift the detected plateau index.

**Decrease if** legitimate transmitters are being rejected because their modulation
index at onset produces a power envelope with moderate (not sharp) derivative. Symptom:
plateau detection success rate < 50% for a specific transmitter that is clearly
audible in the IQ record.

**Data to collect:** Log `sharpness` values from `_find_plateau_sample` (requires
adding debug logging). Plot sharpness distribution per transmitter. Set threshold at
the 10th percentile of the "good" transmitters (ones that produce consistent TDOA)
minus a 20% safety margin.

---

### `min_xcorr_snr = 1.3`

Minimum xcorr SNR (peak / mean of the power-envelope cross-correlation) for the xcorr
result to be used as the primary TDOA.  If SNR is below this threshold, or if the geo
filter (see below) rejects the lag, `compute_tdoa_s()` falls back to the sync_delta
method.

The xcorr SNR has a structural ceiling: the PA cutoff spans only ~16 of the 640 xcorr
output samples (2.5%).  Even for strong signals the peak/mean ratio tends toward
1.0-2.0 rather than 10+ as it might for a pure tone.  A threshold of 1.3 accepts the
large majority of real transmissions while rejecting noise-only correlations, which
cluster near 1.0-1.05.  Field-validated on 92 co-located offset pairs:
threshold=1.3 accepted 36/46 (up from 32/46 at 1.5) with no increase in false detections
(geo filter handles those - see below).

**Increase if** xcorr is accepting pairs with obviously scattered TDOA values (cross-
correlation peak is on noise, not signal).  Start at 1.5.

**Decrease below 1.3** only if the xcorr lag clearly tracks a physical signal but the
SNR is being computed differently.

---

### `max_xcorr_baseline_km = 100.0`

Maximum physical baseline distance between any two nodes in the deployment (km).  Used
to compute the maximum plausible xcorr lag:

```
max_lag_ns = max_xcorr_baseline_km x 1000 / c x 1e9
```

At 100 km this gives ~ 333 usec.  Any xcorr lag exceeding this value is geometrically
impossible (light travel time) and is rejected regardless of SNR, falling back to
sync_delta.

**Set to your actual maximum inter-node distance** (or slightly larger to add margin for
measurement noise).  Setting too large allows false xcorr peaks at unrealistic lags to
slip through; setting too small rejects valid measurements from widely-separated node
pairs.

Field validation: with a 100 km limit, 7 onset pairs that previously passed the
SNR>=1.5 threshold (lags 386-782 usec) were correctly rejected and routed to sync_delta.
The 36 accepted xcorr pairs all had lags <= 7 usec (consistent with co-located geometry).

---

## Calibration process once field data is available

1. Run `colocated_pair_test.py --db <path>` with two co-located nodes for at least
   2 hours covering the transmitters of interest.
2. Check the TDOA distribution for co-located pairs. With plateau-anchored xcorr,
   a co-located pair should show mean ~ 0 usec and std < 20 usec for signals with
   clean transitions. If std is larger, the plateau parameter is the likely cause
   (see per-parameter guidance above).
3. Check the plateau detection success rate: count events where `compute_tdoa_s()`
   returns None vs. a value. If > 30% return None on good signals, loosen
   `min_plateau_sharpness` or `min_consecutive`.
4. After any parameter change, re-run the co-located test to confirm std improves
   without introducing new bias (non-zero mean for co-located nodes = systematic
   error).

## Reference values

Plateau parameters (smooth_samples, stable_fraction, min_consecutive, min_plateau_sharpness)
are heuristic baselines with no field validation yet.  xcorr parameters (min_xcorr_snr,
max_xcorr_baseline_km) are field-validated on 92 co-located offset pairs (2026-03-23).

| Parameter               | Default  | Likely range   | Notes                         |
|-------------------------|----------|----------------|-------------------------------|
| smooth_samples          | 16       | 8-32           | heuristic                     |
| stable_fraction         | 0.15     | 0.05-0.30      | heuristic                     |
| min_consecutive         | 3        | 2-8            | heuristic                     |
| min_plateau_sharpness   | 3.0      | 2.0-6.0        | heuristic                     |
| min_xcorr_snr           | 1.3      | 1.1-1.5        | field-validated                |
| max_xcorr_baseline_km   | 100.0    | actual max km  | set to deployment max baseline |

---

Copyright (c) 2026 Douglas P. Kingston III. MIT License - see [LICENSE](../LICENSE).
