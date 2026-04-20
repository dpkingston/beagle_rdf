# xcorr Improvement Plan — PA knee alignment

> **Status (2026-04-19): archived.** The work this plan describes was
> carried out across commits `311f545`, `5b932c7`, `6661230`, `08eb316`,
> `46a43c8`, `aa72744`, `e33e100`, and `5c2740d`.  The landing point
> diverged from the plan in one important way: we ended up replacing the
> inter-node envelope cross-correlation entirely with **per-node
> argmin(d2) knee finding** on the Savgol-smoothed power envelope, rather
> than fixing the xcorr boundary artifacts.  Current median |err| on the
> Magnolia corpus is ~59 µs; see
> [plateau-detection-tuning.md](plateau-detection-tuning.md) for
> parameters and [TimingAnalysis.md](TimingAnalysis.md) for the full
> history.  The rest of this document is preserved for archaeological
> context.

Driving observation (fix 1762, dpk-tdoa1 vs dpk-tdoa2 offset, Magnolia target):
d2 xcorr returned only +199 ns when the true detection-to-PA-knee offset
between the two nodes is ~4 ms. Root cause: `np.convolve(mode='same')` on
the power envelope zero-pads at the boundary, creating a 0→full ramp over
the first/last 8 samples. The d2 of that ramp dominates xcorr and aligns
the two nodes' identical boundary artifacts at lag ≈ 0, masking the real
PA knee ~253 samples deeper in the snippet.

## Phase 1 — Evaluation harness and test corpus (no code changes)

1a. Assemble ~20-30 paired snippets from DB: recent 2- and 3-node fixes on
    the Magnolia repeater (47.65133, -122.3918318), dpk-tdoa1/dpk-tdoa2
    (and kb7ryy once that's fixed). Mix of onsets/offsets, mix of SNR.

1b. Build `tools/xcorr_harness.py`. For each pair, exercise a matrix of:

    Edge handling:
      E0: current (`mode='same'`, zero boundary)
      E1: edge-pad input, `mode='valid'`
      E2: post-smooth trim (N=16,32,64)
      E3: Tukey window on smoothed envelope (α=0.1,0.25)
      E4: Hann window
      E5: explicit `mode='valid'` with anchor correction

    Differentiation basis: env, d1, d2
    Sign convention: +xcorr_lag, -xcorr_lag

    Record per run: xcorr_lag, xcorr_snr, disambig `n`, computed TDOA,
    error vs geometric truth.

## Phase 2 — Derive correct sign convention from first principles

Formalize the derivation: given raw_ns = sd_A - sd_B, snippets anchored at
each node's detection point, detection fires Δ after the PA knee (Δ varies
per node due to SNR/threshold), derive true_tdoa = raw + path ± xcorr_lag.
Settle whether it's `+` or `-`.

Reconcile with [`test_compute_tdoa_xcorr_refines_sync_delta`](../tests/unit/test_tdoa.py#L410).
That test uses `_make_plateau_pair_iq` + `np.roll` which may not model
detection-anchored snippets. Determine which is wrong: the test, the code,
or my derivation.

## Phase 3 — Select winning combination from harness output

Score by: median |TDOA error|, fraction picking correct disambig `n`,
stability of xcorr SNR (so the `min_xcorr_snr` gate still works), behavior
under SNR disparity.

Cross-check that the empirical winning sign matches the analytical
derivation. If not, stop and understand.

## Phase 4 — Incremental implementation, one commit per change

1. New physically-grounded regression test proving sign + accuracy against
   a geometric scenario. Should FAIL under current code if sign is wrong.
2. Fix sign if needed.
3. Implement winning edge-handling. Add regression test for the artifact.
4. Switch d2→d1 (if harness shows d1 wins).
5. Update `min_xcorr_snr` default if needed.

After each commit: full `pytest`, spot-check harness.

## Phase 5 — Deploy and validate

Deploy to server. Watch 30-60 min of live fixes. Compare residuals and
distance-from-truth before/after. Roll back if worse.

## Phase 6 — kb7ryy pairing (parallel, independent)

Investigate why kb7ryy events don't land in 3-node fixes despite being
online and reporting on the right channel.

## Prior context

- Fix 1762: true TDOA (tdoa1-tdoa2) = -19,101 ns. Server computed +90,187
  (error +71 µs). The ~100 µs grid correction was a crystal-correction
  arithmetic artifact (already fixed; see SyncCalibrator).
- With boundary-trim + d1 + sign-flipped: -22,864 ns (error -4 µs).
  That's the target to validate across the full corpus.
