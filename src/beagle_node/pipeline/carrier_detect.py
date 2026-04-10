# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Dual-threshold power state machine for LMR carrier detection.

Detects when an LMR transmitter keys up (CarrierOnset) and releases
(CarrierOffset) using a hysteresis comparator on the short-term average
power of the IQ stream.

Hysteresis prevents chattering around the threshold:
  - State IDLE  -> ACTIVE  when power rises above  onset_threshold_db
  - State ACTIVE -> IDLE   when power falls below  offset_threshold_db
  - offset_threshold_db < onset_threshold_db

min_hold_windows (onset) and min_release_windows (offset) provide
additional debounce:

  onset:  power must stay >= onset_threshold_db for min_hold_windows
          consecutive windows before CarrierOnset fires.

  offset: power must stay <= offset_threshold_db for min_release_windows
          consecutive windows before CarrierOffset fires.  A single
          power dip below the offset threshold that recovers before
          min_release_windows accumulate does NOT trigger an offset.
          Use min_release_windows >= 4 for real-world signals where
          brief power fades are common.

In freq_hop mode, prime_state() is called at the start of each target
block.  Two symmetrical guard conditions prevent spurious events anchored
to the block boundary rather than a genuine carrier edge:

  onset guard  (min_idle_for_onset):  at least 2 idle windows must be
      observed before a CarrierOnset is emitted.  This suppresses carriers
      already present at block start and single-window PLL settling artefacts.

  offset guard (min_active_windows_for_offset):  when prime_state() sets
      state to ``active`` (carrier already present at block start), at least
      this many windows of above-threshold signal must accumulate before a
      CarrierOffset is allowed.  This suppresses carrier-tail events where
      the transmitter was still keyed during the sync block and drops within
      the first few windows of the target block - events whose timing is
      anchored to the block boundary, not the true PA shutoff.

Both events carry the *sample index* in the continuous sample stream
(not relative to any buffer), which is what DeltaComputer needs.
"""

from __future__ import annotations

import json as _json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Set BEAGLE_TIMING_DIAG=1 to enable per-event structured diagnostic logging.
# See docs/dev/timing-diagnostics.md for the field reference and analysis scripts.
_TIMING_DIAG = os.environ.get("BEAGLE_TIMING_DIAG") == "1"


@dataclass(frozen=True)
class CarrierOnset:
    """LMR carrier has appeared."""
    sample_index: int           # In the continuous IQ stream
    power_db: float             # Instantaneous power at detection
    noise_floor_db: float = -100.0  # EMA of idle-state power; used to compute SNR
    iq_snippet: bytes = b""    # int8-interleaved IQ bytes for cross-correlation


@dataclass(frozen=True)
class CarrierOffset:
    """LMR carrier has disappeared."""
    sample_index: int
    power_db: float
    iq_snippet: bytes = b""    # int8-interleaved IQ bytes for cross-correlation


_State = Literal["idle", "active"]


class CarrierDetector:
    """
    Dual-threshold (hysteresis) power detector.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the input IQ stream.
    onset_threshold_db : float
        Power level (dBFS) that triggers a CarrierOnset.
    offset_threshold_db : float
        Power level (dBFS) below which a CarrierOffset is triggered.
        Must be < onset_threshold_db.
    window_samples : int
        Number of samples over which to average power before comparing
        to thresholds.  Larger values reduce false triggers but add latency.
    min_hold_windows : int
        Number of consecutive above-threshold windows required before
        a CarrierOnset is declared.  Default 1 (fire on first window).
        Set to 4 to suppress transient spikes (~4 ms at 64 kHz).
    min_release_windows : int
        Number of consecutive below-threshold windows required before
        a CarrierOffset is declared.  Default 1 (fire on first window).
        Set to 4-8 to suppress brief power fades in real-world signals
        and prevent spurious re-onset/offset chattering.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        onset_threshold_db: float = -30.0,
        offset_threshold_db: float = -40.0,
        window_samples: int = 64,
        min_hold_windows: int = 1,
        min_release_windows: int = 1,
        snippet_samples: int = 640,
        snippet_post_windows: int = 5,
        ring_lookback_windows: int | None = None,
        min_active_windows_for_offset: int = 0,
    ) -> None:
        if offset_threshold_db >= onset_threshold_db:
            raise ValueError(
                f"offset_threshold_db ({offset_threshold_db}) must be < "
                f"onset_threshold_db ({onset_threshold_db})"
            )
        if window_samples < 1:
            raise ValueError(f"window_samples must be >= 1, got {window_samples}")
        if min_hold_windows < 1:
            raise ValueError(f"min_hold_windows must be >= 1, got {min_hold_windows}")
        if min_release_windows < 1:
            raise ValueError(f"min_release_windows must be >= 1, got {min_release_windows}")
        if snippet_post_windows < 0:
            raise ValueError(f"snippet_post_windows must be >= 0, got {snippet_post_windows}")
        if min_active_windows_for_offset < 0:
            raise ValueError(
                f"min_active_windows_for_offset must be >= 0, got {min_active_windows_for_offset}"
            )

        self._rate = float(sample_rate_hz)
        self._onset_db = float(onset_threshold_db)
        self._offset_db = float(offset_threshold_db)
        self._window = int(window_samples)
        self._min_hold = int(min_hold_windows)
        self._min_release = int(min_release_windows)
        self._snippet_samples = int(snippet_samples)
        self._post_windows = int(snippet_post_windows)
        self._min_active_for_offset = int(min_active_windows_for_offset)

        self._state: _State = "idle"
        self._cumulative_sample: int = 0   # total samples seen so far
        # Exponential moving average of power during idle (no-carrier) windows.
        # Initialised to offset_threshold_db (a reasonable upper bound on noise).
        # Alpha = 0.01 -> time constant ~ 100 windows (~ 100 ms at 64 kHz / 64-sample window).
        self._noise_floor_db: float = float(offset_threshold_db)
        self._noise_floor_alpha: float = 0.01
        # freq_hop block-start onset suppression.
        # Counts below-threshold (idle) windows seen since the last
        # prime_state() call.  An onset is only emitted once this count
        # reaches _min_idle_for_onset, ensuring the detector has observed
        # a meaningful period of noise before declaring a carrier appeared.
        # This prevents false onsets from:
        #   (a) carrier already present at block start (mid-transmission
        #       arrival) - no idle windows at all (count = 0).
        #   (b) a single PLL-settling transient window that dips below
        #       threshold and immediately recovers - only count = 1.
        # Starts high so the very first block (no prime_state) works
        # normally.
        self._idle_window_count: int = 1000
        self._min_idle_for_onset: int = 2
        # Block-start offset suppression (parallels _idle_window_count / _min_idle_for_onset).
        # When prime_state() sets state to ``active`` (carrier already present at block
        # start), _primed_active is True and _active_window_count is reset to 0.
        # Each window where state is ``active`` and power is above offset_db increments
        # _active_window_count.  If min_active_windows_for_offset > 0 and
        # _active_window_count < _min_active_for_offset when an offset fires,
        # the event is suppressed - the carrier was only a block-start tail, not a
        # transmission we can meaningfully timestamp.
        # _primed_active is cleared (False) when a genuine onset fires, so that
        # offsets following a genuine onset in the same block are never suppressed.
        self._primed_active: bool = False
        self._active_window_count: int = 0
        # Count of windows processed since the last prime_state() call.
        # Used to diagnose whether offset/onset events fire early (block-start
        # artefacts) or mid-block (genuine transitions).  Starts high so the
        # very first block (no prime_state) is not mistakenly logged as block-start.
        self._windows_since_prime: int = 1000
        # Count of consecutive above-threshold windows seen while in pre-onset
        # state; onset is only declared once this reaches min_hold_windows.
        self._pre_onset_count: int = 0
        # Count of consecutive below-threshold windows seen while in pre-offset
        # state; offset is only declared once this reaches min_release_windows.
        self._pre_offset_count: int = 0
        # Rolling buffer of recent IQ windows for snippet capture.
        # For offset events the PA shutoff must lie within this buffer at
        # detection time: size = max(fade_windows + min_release, snippet_windows).
        # ring_lookback_windows lets operators decouple ring depth from snippet
        # size so gradual fades (10-50 ms) are always captured regardless of the
        # shipped snippet length.  Defaults to 3x the snippet window count so
        # that even a fade spanning the full snippet still has the shutoff inside.
        _snippet_windows = max(1, -(-self._snippet_samples // self._window))
        _ring_capacity = int(ring_lookback_windows) if ring_lookback_windows is not None else _snippet_windows * 3
        self._iq_ring: deque[np.ndarray] = deque(maxlen=_ring_capacity)

        # Deferred-emission state (used when snippet_post_windows > 0).
        # When a threshold crossing is detected, rather than emitting immediately,
        # we snapshot the pre-event ring and collect _post_windows more windows of
        # post-event IQ before emitting.  If the opposite transition fires while
        # collecting post-event IQ, we flush the pending event with a partial snippet.
        self._pending_event_type: str | None = None   # "onset" or "offset"
        self._pending_sample_index: int = 0
        self._pending_power_db: float = 0.0
        self._pending_noise_floor_db: float = -100.0   # noise floor snapshot at detection
        self._pending_pre_snap: list[np.ndarray] = []   # copy of ring at detection time
        self._pending_post_buf: list[np.ndarray] = []   # post-event windows collected so far
        self._pending_post_remaining: int = 0           # windows still to collect
        # Absolute sample index of pre_snap[0][0] for the pending offset event.
        # Used to convert cut_idx (index into iq_cat) back to a stream sample index.
        self._pending_pre_snap_start: int = 0

        # Snippet transition validation (freq_hop defence-in-depth).
        # When prime_state() is called, this flag arms validation for the next
        # process() call.  Events whose IQ snippets lack a genuine power
        # transition (all-carrier or all-noise) are dropped - they indicate a
        # mid-transmission arrival where the timing is anchored to the block
        # boundary rather than the true carrier edge, and the xcorr IQ has no
        # noise<->carrier step for the correlator to lock onto.
        self._validate_snippets: bool = False
        self._min_transition_db: float = 6.0

        # Periodic diagnostics: log noise floor / power stats every N windows.
        self._diag_interval: int = 10_000  # ~10 s at 64 kHz / 64-sample window
        self._diag_counter: int = 0
        self._diag_power_sum: float = 0.0
        self._diag_power_max: float = -200.0
        self._diag_power_min: float = 200.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> _State:
        return self._state

    @property
    def onset_threshold_db(self) -> float:
        return self._onset_db

    @property
    def offset_threshold_db(self) -> float:
        return self._offset_db

    @property
    def min_hold_windows(self) -> int:
        return self._min_hold

    @property
    def min_release_windows(self) -> int:
        return self._min_release

    @property
    def min_active_windows_for_offset(self) -> int:
        return self._min_active_for_offset

    @property
    def noise_floor_db(self) -> float:
        """Current noise floor estimate (EMA of idle-state power)."""
        return self._noise_floor_db

    # ------------------------------------------------------------------
    # Live threshold updates
    # ------------------------------------------------------------------

    def update_thresholds(
        self,
        onset_threshold_db: float | None = None,
        offset_threshold_db: float | None = None,
        min_hold_windows: int | None = None,
        min_release_windows: int | None = None,
        min_active_windows_for_offset: int | None = None,
    ) -> None:
        """Update detection thresholds on a live detector without resetting state.

        Only provided (non-None) parameters are changed.  The state machine,
        ring buffer, and noise floor EMA are preserved so detection continues
        seamlessly with the new thresholds.
        """
        onset = self._onset_db if onset_threshold_db is None else onset_threshold_db
        offset = self._offset_db if offset_threshold_db is None else offset_threshold_db
        if offset >= onset:
            raise ValueError(
                f"offset_threshold_db ({offset}) must be < onset_threshold_db ({onset})"
            )
        self._onset_db = onset
        self._offset_db = offset
        if min_hold_windows is not None:
            if min_hold_windows < 1:
                raise ValueError(f"min_hold_windows must be >= 1, got {min_hold_windows}")
            self._min_hold = min_hold_windows
        if min_release_windows is not None:
            if min_release_windows < 1:
                raise ValueError(f"min_release_windows must be >= 1, got {min_release_windows}")
            self._min_release = min_release_windows
        if min_active_windows_for_offset is not None:
            if min_active_windows_for_offset < 0:
                raise ValueError(
                    f"min_active_windows_for_offset must be >= 0, "
                    f"got {min_active_windows_for_offset}"
                )
            self._min_active_for_offset = min_active_windows_for_offset
        logger.info(
            "Thresholds updated: onset=%.1f offset=%.1f hold=%d release=%d "
            "min_active_for_offset=%d",
            self._onset_db, self._offset_db, self._min_hold, self._min_release,
            self._min_active_for_offset,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self, iq: np.ndarray, start_sample: int
    ) -> list[CarrierOnset | CarrierOffset]:
        """
        Scan an IQ buffer for state transitions.

        Parameters
        ----------
        iq : np.ndarray, complex64
            IQ samples at sample_rate_hz.
        start_sample : int
            Cumulative sample index of iq[0] in the continuous stream.

        Returns
        -------
        list of CarrierOnset / CarrierOffset
            Zero or more events, in chronological order.
            Sample indices are absolute (start_sample + offset within buffer).
        """
        events: list[CarrierOnset | CarrierOffset] = []
        n = len(iq)
        if n == 0:
            return events

        # Compute per-window averaged power in dBFS
        n_windows = n // self._window
        # Reshape to (n_windows, window_samples) and compute all power values at once,
        # avoiding n_windows separate numpy calls inside the Python loop.
        iq_windows = iq[:n_windows * self._window].reshape(n_windows, self._window)
        avg_powers = (np.abs(iq_windows) ** 2).mean(axis=1)
        powers_db = 10.0 * np.log10(avg_powers + 1e-30)

        for i in range(n_windows):
            window_iq = iq_windows[i]
            # Accumulate this window into the ring buffer (used for snippet capture).
            self._iq_ring.append(window_iq)
            self._windows_since_prime += 1

            # --- Periodic power diagnostics ---
            pwr = float(powers_db[i])
            self._diag_power_sum += pwr
            self._diag_power_max = max(self._diag_power_max, pwr)
            self._diag_power_min = min(self._diag_power_min, pwr)
            self._diag_counter += 1
            if self._diag_counter >= self._diag_interval:
                avg = self._diag_power_sum / self._diag_counter
                logger.debug(
                    "CarrierDetector diag: state=%s noise_floor=%.1f "
                    "avg_power=%.1f min=%.1f max=%.1f onset_th=%.1f offset_th=%.1f",
                    self._state, self._noise_floor_db,
                    avg, self._diag_power_min, self._diag_power_max,
                    self._onset_db, self._offset_db,
                )
                self._diag_counter = 0
                self._diag_power_sum = 0.0
                self._diag_power_max = -200.0
                self._diag_power_min = 200.0

            power_db = float(powers_db[i])
            # Sample index at the centre of this window
            window_sample = start_sample + i * self._window + self._window // 2

            # --- Deferred post-event collection ----------------------------------
            # If we are collecting post-event windows for a pending event, consume
            # this window before applying the normal state machine.
            if self._pending_event_type is not None:
                self._pending_post_buf.append(window_iq)
                self._pending_post_remaining -= 1

                # Check whether the *opposite* transition has fired, which requires
                # flushing the pending event immediately (partial post snippet).
                opposite_fired = False
                if self._pending_event_type == "onset" and self._state == "active":
                    # Check for offset transition while collecting post-onset IQ.
                    if power_db <= self._offset_db:
                        self._pre_offset_count += 1
                        if self._pre_offset_count >= self._min_release:
                            opposite_fired = True
                            self._state = "idle"
                            self._pre_onset_count = 0
                            self._pre_offset_count = 0
                    else:
                        self._pre_offset_count = 0
                elif self._pending_event_type == "offset" and self._state == "idle":
                    # Check for onset transition while collecting post-offset IQ.
                    if power_db >= self._onset_db:
                        self._pre_onset_count += 1
                        if self._pre_onset_count >= self._min_hold:
                            opposite_fired = True
                            self._state = "active"
                            self._pre_onset_count = 0
                            self._pre_offset_count = 0
                    else:
                        self._pre_onset_count = 0

                if self._pending_post_remaining <= 0 or opposite_fired:
                    ev: CarrierOnset | CarrierOffset
                    if self._pending_event_type == "onset":
                        _onset_bytes, _rise_idx = self._encode_combined(
                            self._pending_pre_snap, self._pending_post_buf
                        )
                        # sample_index = snippet anchor (argmax of power
                        # derivative = steepest carrier rise), matching the
                        # 25% position in the trimmed snippet.  This ensures
                        # sync_delta_ns and the xcorr reference the same
                        # physical time instant.
                        _onset_sample_index = (
                            self._pending_pre_snap_start + _rise_idx
                        )
                        ev = CarrierOnset(
                            sample_index=_onset_sample_index,
                            power_db=self._pending_power_db,
                            noise_floor_db=self._pending_noise_floor_db,
                            iq_snippet=_onset_bytes,
                        )
                        if _TIMING_DIAG:
                            _pre_start = self._pending_pre_snap_start
                            logger.info(
                                "TIMING_DIAG %s",
                                _json.dumps({
                                    "stage": "carrier",
                                    "event_type": "onset",
                                    "sample_index": ev.sample_index,
                                    "pre_snap_start": _pre_start,
                                    "rise_idx_in_buf": _rise_idx,
                                    "window_sample": self._pending_sample_index,
                                    "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                                    "power_db": round(ev.power_db, 1),
                                }),
                            )
                    else:
                        _offset_bytes, _cut_idx = self._encode_offset_snippet(
                            self._pending_pre_snap, self._pending_post_buf
                        )
                        ev = CarrierOffset(
                            sample_index=self._pending_pre_snap_start + _cut_idx,
                            power_db=self._pending_power_db,
                            iq_snippet=_offset_bytes,
                        )
                        if _TIMING_DIAG:
                            _pre_start = self._pending_pre_snap_start
                            logger.info(
                                "TIMING_DIAG %s",
                                _json.dumps({
                                    "stage": "carrier",
                                    "event_type": "offset",
                                    "sample_index": ev.sample_index,
                                    "pre_snap_start": _pre_start,
                                    "cut_idx_in_buf": _cut_idx,
                                    "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                                    "power_db": round(ev.power_db, 1),
                                }),
                            )
                    events.append(ev)
                    logger.debug(
                        "Deferred %s emitted at window %d (pre=%d post=%d snippet=%d bytes)",
                        type(ev).__name__, self._windows_since_prime,
                        len(self._pending_pre_snap), len(self._pending_post_buf),
                        len(ev.iq_snippet),
                    )
                    self._pending_event_type = None
                    self._pending_pre_snap = []
                    self._pending_post_buf = []
                    self._pending_post_remaining = 0

                    if opposite_fired:
                        # The opposite transition also needs to be emitted; defer or
                        # emit it immediately (it will go through the same deferral
                        # path on the next outer-loop iteration if _post_windows > 0).
                        if self._state == "idle":
                            # offset just fired during post-onset collection
                            if self._post_windows > 0:
                                self._pending_event_type = "offset"
                                self._pending_sample_index = window_sample
                                self._pending_power_db = power_db
                                self._pending_pre_snap = list(self._iq_ring)
                                self._pending_pre_snap_start = (
                                    window_sample - self._window // 2
                                    - (len(self._pending_pre_snap) - 1) * self._window
                                )
                                self._pending_post_buf = []
                                self._pending_post_remaining = self._post_windows
                            else:
                                _pre = list(self._iq_ring)
                                _pre_start = window_sample - self._window // 2 - (len(_pre) - 1) * self._window
                                _off_bytes, _cut = self._encode_offset_snippet(_pre)
                                events.append(CarrierOffset(
                                    sample_index=_pre_start + _cut,
                                    power_db=power_db,
                                    iq_snippet=_off_bytes,
                                ))
                        else:
                            # onset just fired during post-offset collection.
                            # This is a genuine onset; clear the block-start guard.
                            self._primed_active = False
                            self._active_window_count = 0
                            if self._post_windows > 0:
                                self._pending_event_type = "onset"
                                self._pending_sample_index = window_sample
                                self._pending_power_db = power_db
                                self._pending_noise_floor_db = self._noise_floor_db
                                self._pending_pre_snap = list(self._iq_ring)
                                self._pending_pre_snap_start = (
                                    window_sample - self._window // 2
                                    - (len(self._pending_pre_snap) - 1) * self._window
                                )
                                self._pending_post_buf = []
                                self._pending_post_remaining = self._post_windows
                            else:
                                opp_snippet = self._encode_snippet()
                                events.append(CarrierOnset(
                                    sample_index=window_sample,
                                    power_db=power_db,
                                    noise_floor_db=self._noise_floor_db,
                                    iq_snippet=opp_snippet,
                                ))
                continue  # window was consumed by post-collection; skip normal FSM
            # --- End deferred collection -----------------------------------------

            if self._state == "idle":
                if power_db >= self._onset_db:
                    self._pre_onset_count += 1
                    if self._pre_onset_count >= self._min_hold:
                        self._state = "active"
                        self._pre_onset_count = 0
                        self._pre_offset_count = 0
                        logger.debug(
                            "CarrierOnset detected at window %d since prime_state "
                            "(power=%.1f dB, idle_count=%d)",
                            self._windows_since_prime,
                            power_db,
                            self._idle_window_count,
                        )
                        if self._idle_window_count < self._min_idle_for_onset:
                            # Not enough idle history since the last prime_state()
                            # call.  This onset is either:
                            #   - carrier already present at block start (count=0)
                            #   - PLL settling artefact produced a single noisy
                            #     window that dipped below threshold (count=1)
                            # Either way the IQ snippet has no genuine noise->carrier
                            # transition and the timing is anchored to the block
                            # boundary, not the true carrier onset.  Suppress.
                            logger.debug(
                                "Suppressed block-start onset at sample %d "
                                "(only %d idle windows since prime_state, need %d; "
                                "freq_hop mid-transmission arrival)",
                                window_sample,
                                self._idle_window_count,
                                self._min_idle_for_onset,
                            )
                            # State is already "active"; idle_window_count continues
                            # to accumulate when the carrier later drops, so a
                            # genuine re-key in this block will pass the check.
                            # _active_window_count resets so this new active period
                            # accumulates from scratch for the offset guard.
                            self._active_window_count = 0
                        else:
                            # Genuine onset: clear the block-start offset guard so
                            # a subsequent offset in this block is not suppressed.
                            self._primed_active = False
                            self._active_window_count = 0
                            if self._post_windows > 0:
                                self._pending_event_type = "onset"
                                self._pending_sample_index = window_sample
                                self._pending_power_db = power_db
                                self._pending_noise_floor_db = self._noise_floor_db
                                self._pending_pre_snap = list(self._iq_ring)
                                self._pending_pre_snap_start = (
                                    window_sample - self._window // 2
                                    - (len(self._pending_pre_snap) - 1) * self._window
                                )
                                self._pending_post_buf = []
                                self._pending_post_remaining = self._post_windows
                            else:
                                snippet = self._encode_snippet()
                                events.append(CarrierOnset(
                                    sample_index=window_sample,
                                    power_db=power_db,
                                    noise_floor_db=self._noise_floor_db,
                                    iq_snippet=snippet,
                                ))
                else:
                    self._pre_onset_count = 0
                    # Idle window: signal is below onset threshold.
                    # Increment the idle window count so that subsequent onsets
                    # can verify they have enough noise history.
                    self._idle_window_count += 1
                    # Update noise floor EMA while clearly below the onset threshold.
                    self._noise_floor_db += self._noise_floor_alpha * (
                        power_db - self._noise_floor_db
                    )

            elif self._state == "active":
                if power_db <= self._offset_db:
                    self._pre_offset_count += 1
                    # These windows are below onset_db too (offset_db < onset_db),
                    # so they represent genuine idle signal.  Count them toward
                    # _idle_window_count so that a re-onset after a mid-block
                    # offset is not suppressed by _min_idle_for_onset: without
                    # this, release windows are processed in active state (where
                    # _idle_window_count doesn't increment), leaving it at zero
                    # and causing the subsequent re-onset to be suppressed even
                    # though the carrier genuinely dropped.
                    self._idle_window_count += 1
                    if self._pre_offset_count >= self._min_release:
                        self._state = "idle"
                        self._pre_onset_count = 0
                        self._pre_offset_count = 0
                        logger.debug(
                            "CarrierOffset detected at window %d since prime_state "
                            "(power=%.1f dB, ring_size=%d)",
                            self._windows_since_prime,
                            power_db,
                            len(self._iq_ring),
                        )
                        _active_count = self._active_window_count
                        _suppress = (
                            self._min_active_for_offset > 0
                            and self._primed_active
                            and _active_count < self._min_active_for_offset
                        )
                        self._active_window_count = 0
                        if _suppress:
                            logger.debug(
                                "Block-start offset suppressed at window %d: only %d "
                                "active windows since prime_state (need %d)",
                                self._windows_since_prime,
                                _active_count,
                                self._min_active_for_offset,
                            )
                        elif self._post_windows > 0:
                            self._pending_event_type = "offset"
                            self._pending_sample_index = window_sample
                            self._pending_power_db = power_db
                            self._pending_pre_snap = list(self._iq_ring)
                            self._pending_pre_snap_start = (
                                window_sample - self._window // 2
                                - (len(self._pending_pre_snap) - 1) * self._window
                            )
                            self._pending_post_buf = []
                            self._pending_post_remaining = self._post_windows
                        else:
                            _pre = list(self._iq_ring)
                            _pre_start = window_sample - self._window // 2 - (len(_pre) - 1) * self._window
                            _off_bytes, _cut = self._encode_offset_snippet(_pre)
                            events.append(CarrierOffset(
                                sample_index=_pre_start + _cut,
                                power_db=power_db,
                                iq_snippet=_off_bytes,
                            ))
                else:
                    # Signal above offset threshold - reset the release counter.
                    # Transient dips that recover do not accumulate toward offset.
                    self._pre_offset_count = 0
                    # Count windows where the carrier is genuinely above threshold.
                    # Used by the block-start offset guard (_min_active_for_offset).
                    self._active_window_count += 1

        self._cumulative_sample = start_sample + n

        # Flush any pending deferred event with a partial snippet rather than
        # leaving it to be silently abandoned by prime_state() on the next block.
        # This happens when an onset/offset fires near the end of the block and
        # there aren't enough remaining windows to collect the full post_windows.
        if self._pending_event_type is not None:
            ev: CarrierOnset | CarrierOffset
            if self._pending_event_type == "onset":
                _onset_bytes, _rise_idx = self._encode_combined(
                    self._pending_pre_snap, self._pending_post_buf
                )
                _onset_sample_index = (
                    self._pending_pre_snap_start + _rise_idx
                )
                ev = CarrierOnset(
                    sample_index=_onset_sample_index,
                    power_db=self._pending_power_db,
                    noise_floor_db=self._pending_noise_floor_db,
                    iq_snippet=_onset_bytes,
                )
                if _TIMING_DIAG:
                    _pre_start = self._pending_pre_snap_start
                    logger.info(
                        "TIMING_DIAG %s",
                        _json.dumps({
                            "stage": "carrier",
                            "event_type": "onset",
                            "sample_index": ev.sample_index,
                            "pre_snap_start": _pre_start,
                            "rise_idx_in_buf": _rise_idx,
                            "window_sample": self._pending_sample_index,
                            "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                            "power_db": round(ev.power_db, 1),
                            "partial_flush": True,
                        }),
                    )
            else:
                _offset_bytes, _cut_idx = self._encode_offset_snippet(
                    self._pending_pre_snap, self._pending_post_buf
                )
                ev = CarrierOffset(
                    sample_index=self._pending_pre_snap_start + _cut_idx,
                    power_db=self._pending_power_db,
                    iq_snippet=_offset_bytes,
                )
                if _TIMING_DIAG:
                    _pre_start = self._pending_pre_snap_start
                    logger.info(
                        "TIMING_DIAG %s",
                        _json.dumps({
                            "stage": "carrier",
                            "event_type": "offset",
                            "sample_index": ev.sample_index,
                            "pre_snap_start": _pre_start,
                            "cut_idx_in_buf": _cut_idx,
                            "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                            "power_db": round(ev.power_db, 1),
                            "partial_flush": True,
                        }),
                    )
            events.append(ev)
            logger.debug(
                "Partial flush %s at end of block (had %d/%d post windows)",
                type(ev).__name__,
                len(self._pending_post_buf),
                self._post_windows,
            )
            self._pending_event_type = None
            self._pending_pre_snap = []
            self._pending_post_buf = []
            self._pending_post_remaining = 0

        # After prime_state() (freq_hop target blocks), drop events whose IQ
        # snippets lack a genuine power transition.  This is a second line of
        # defence beyond _had_idle_window: it catches mid-transmission arrivals
        # that slip past the first-window power check (e.g. PLL settling
        # artefacts that produce a spurious idle window).
        if self._validate_snippets and events:
            validated: list[CarrierOnset | CarrierOffset] = []
            for ev in events:
                if self._snippet_has_transition(ev.iq_snippet):
                    validated.append(ev)
                else:
                    logger.debug(
                        "Dropped %s at sample %d: IQ snippet has no power "
                        "transition (freq_hop mid-transmission arrival)",
                        type(ev).__name__,
                        ev.sample_index,
                    )
            events = validated
        self._validate_snippets = False

        return events

    def _encode_snippet(self) -> bytes:
        """
        Encode the buffered IQ ring as interleaved int8 bytes for cross-correlation.

        The ring holds up to ceil(snippet_samples / window_samples) recent windows;
        concatenating and trimming to snippet_samples gives the most recent IQ around
        the event.

        Encoding: real[0], imag[0], real[1], imag[1], ...  (int8, range -127..127).
        The scale factor is chosen so the largest |sample| maps to +/-127; the
        server does not need to know the scale because cross-correlation lag is
        scale-independent.
        """
        assert self._iq_ring, "IQ ring is empty at snippet encode time - cannot happen"
        iq_cat = np.concatenate(list(self._iq_ring))
        iq_trim = iq_cat[-self._snippet_samples:]
        scale = float(np.max(np.abs(iq_trim))) + 1e-30
        normed = iq_trim / scale
        int8_ri = np.empty(len(normed) * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
        return int8_ri.tobytes()

    def _encode_combined(
        self, pre_snap: list[np.ndarray], post_buf: list[np.ndarray]
    ) -> tuple[bytes, int]:
        """
        Encode a transition-anchored onset snippet.

        Concatenates pre-event and post-event IQ, finds the carrier onset
        (peak positive power derivative = steepest carrier rise), and places
        it at 1/4 from the snippet start.  This mirrors _encode_offset_snippet
        (which places the PA shutoff at 3/4) and ensures the transition sits
        at a fixed position independent of min_hold_windows.

        Returns
        -------
        (bytes, rise_idx)
            bytes is the encoded snippet; rise_idx is the position of the
            onset (peak positive power derivative) within the concatenated
            pre+post buffer, in target-rate samples.  This mirrors
            ``_encode_offset_snippet`` and lets the caller log diagnostics
            anchored to the same point the snippet is trimmed around.
        """
        assert pre_snap or post_buf, "Both pre_snap and post_buf are empty - cannot happen"
        parts = list(pre_snap) + list(post_buf or [])
        iq_cat = np.concatenate(parts)
        assert len(iq_cat) >= 32, "Onset IQ data too short for derivative - cannot happen"

        # Smoothed power envelope and its derivative
        smooth = 16
        power = iq_cat.real.astype(np.float64) ** 2 + iq_cat.imag.astype(np.float64) ** 2
        kernel = np.ones(smooth) / smooth
        envelope = np.convolve(power, kernel, mode='same')
        deriv = np.diff(envelope)

        # Peak positive derivative = steepest carrier rise = onset moment
        rise_idx = int(np.argmax(deriv))

        # Place the onset at 1/4 from the start of the snippet.
        # This gives snippet/4 noise (pre-onset) + 3*snippet/4 carrier
        # (post-onset) for xcorr, matching the server's onset trim [:3N//4].
        pre_target = self._snippet_samples // 4
        start = max(0, rise_idx - pre_target)
        end = start + self._snippet_samples
        if end > len(iq_cat):
            end = len(iq_cat)
            start = max(0, end - self._snippet_samples)
        iq_trim = iq_cat[start:end]

        assert len(iq_trim) > 0, "Trimmed onset snippet is empty - cannot happen"
        scale = float(np.max(np.abs(iq_trim))) + 1e-30
        normed = iq_trim / scale
        int8_ri = np.empty(len(normed) * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
        return int8_ri.tobytes(), rise_idx

    def _encode_offset_snippet(
        self,
        pre_snap: list[np.ndarray],
        post_buf: list[np.ndarray] | None = None,
    ) -> tuple[bytes, int]:
        """
        Encode a snippet for a carrier-offset event anchored on the PA shutoff.

        The PA shutoff is an instantaneous transmitter-side event (bias cut)
        that appears at the same wall-clock time on all receivers.  Finding the
        peak negative power derivative in the combined pre+post IQ data identifies
        that moment, and the snippet is then placed so the shutoff lands at a
        fixed position (3/4 from the start) regardless of detection timing.

        Using post_buf (from the deferred emission path) is strongly recommended:
        it supplies post-cutoff noise samples that keep the target position inside
        the snippet even when detection fires only a few windows after the shutoff.
        Without post_buf the target position may be clamped, causing the cutoff to
        appear at different positions for nodes with different detection delays --
        which reintroduces the alignment problem this method is designed to solve.
        """
        parts = list(pre_snap) + list(post_buf or [])
        assert parts, "No IQ data for offset snippet - cannot happen"
        iq_cat = np.concatenate(parts)
        assert len(iq_cat) >= 32, "Offset IQ data too short for derivative - cannot happen"

        # Smoothed power envelope and its derivative
        smooth = 16
        power = iq_cat.real.astype(np.float64) ** 2 + iq_cat.imag.astype(np.float64) ** 2
        kernel = np.ones(smooth) / smooth
        envelope = np.convolve(power, kernel, mode='same')
        deriv = np.diff(envelope)
        assert len(deriv) > 0, "Empty derivative - cannot happen"

        # Peak negative derivative = fastest power drop = PA shutoff moment
        cut_idx = int(np.argmin(deriv))

        # Place the PA shutoff at 3/4 from the start of the snippet.
        # This uses snippet*3/4 samples of carrier (pre-cutoff) for xcorr
        # and snippet/4 samples of noise (post-cutoff) as confirmation.
        # Both nodes independently center here, so the cutoff lands at the
        # same position in both snippets regardless of detection timing.
        pre_target = (self._snippet_samples * 3) // 4
        start = max(0, cut_idx - pre_target)
        end = start + self._snippet_samples
        if end > len(iq_cat):
            end = len(iq_cat)
            start = max(0, end - self._snippet_samples)
        iq_trim = iq_cat[start:end]

        assert len(iq_trim) > 0, "Trimmed offset snippet is empty - cannot happen"
        scale = float(np.max(np.abs(iq_trim))) + 1e-30
        normed = iq_trim / scale
        int8_ri = np.empty(len(normed) * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)
        return int8_ri.tobytes(), cut_idx

    def _snippet_has_transition(self, snippet: bytes) -> bool:
        """Check that an encoded IQ snippet contains a genuine power transition.

        Mid-transmission arrivals produce snippets where the entire content is
        at carrier power (no noise->carrier or carrier->noise step).  The
        cross-correlator cannot lock onto these, and the timing measurement is
        anchored to the block boundary rather than the true carrier edge.

        Returns True if the snippet's per-window power dynamic range exceeds
        ``_min_transition_db``, indicating a real transition is present.
        Returns True (passes) for snippets that are too short to validate.
        """
        n_bytes = len(snippet)
        n_samples = n_bytes // 2          # interleaved real/imag pairs
        if n_samples < self._window * 2:
            return True                   # too short for meaningful validation

        arr = np.frombuffer(snippet, dtype=np.int8)
        real = arr[0::2].astype(np.float32)
        imag = arr[1::2].astype(np.float32)

        n_win = n_samples // self._window
        if n_win < 2:
            return True
        usable = n_win * self._window
        real_w = real[:usable].reshape(n_win, self._window)
        imag_w = imag[:usable].reshape(n_win, self._window)
        powers = np.mean(real_w ** 2 + imag_w ** 2, axis=1)
        powers_db = 10.0 * np.log10(powers + 1e-10)

        dynamic_range_db = float(np.max(powers_db) - np.min(powers_db))
        return dynamic_range_db >= self._min_transition_db

    def prime_state(self, iq: np.ndarray) -> None:
        """
        Set the detector state from the power of the first window of *iq*,
        without emitting any events.  Also clears the ring buffer and
        debounce counters.

        Call this at the start of each freq_hop target block so that the
        detector reflects the current carrier state rather than firing a
        spurious onset or offset for a transition that occurred while the
        SDR was tuned to the sync channel.

        State assignment rules (intentionally *without* hysteresis):

        * power >= onset_threshold_db -> ``active``
        * power <  onset_threshold_db -> ``idle``

        Using the onset threshold ensures that PLL settling transients
        (which often push power a few dB above the noise floor but well
        below onset_db) do not trigger a false ``active`` state that
        immediately produces a spurious carrier-tail offset.  A genuine
        carrier that was transmitting during the sync block will be well
        above onset_db and correctly detected as active.  The
        ``_min_idle_for_onset`` guard provides an additional safety net
        for the onset path: at least that many idle windows must be
        observed before an onset is emitted.

        If *iq* is shorter than one window, the state is forced to ``idle``
        (safe default for an uncharacterised block).
        """
        if len(iq) >= self._window:
            avg_power = float(np.mean(np.abs(iq[:self._window]) ** 2))
            power_db = 10.0 * np.log10(avg_power + 1e-30)
            if power_db >= self._onset_db:
                self._state = "active"
                self._primed_active = True
            else:
                self._state = "idle"
                self._primed_active = False
        else:
            self._state = "idle"
            self._primed_active = False
        self._pre_onset_count = 0
        self._pre_offset_count = 0
        self._active_window_count = 0
        self._windows_since_prime = 0
        self._iq_ring.clear()
        if self._pending_event_type is not None:
            logger.warning(
                "prime_state abandoned pending %s (had %d post windows collected)"
                " - partial flush should have caught this",
                self._pending_event_type,
                len(self._pending_post_buf),
            )
        self._pending_event_type = None
        self._pending_pre_snap = []
        self._pending_post_buf = []
        self._pending_post_remaining = 0
        self._pending_pre_snap_start = 0
        # Reset idle window count: require _min_idle_for_onset idle windows
        # before a new onset is emitted.  Catches carriers in the PLL
        # settling zone that slip past the first-window power check above.
        self._idle_window_count = 0
        # Arm snippet transition validation for the upcoming process() call.
        self._validate_snippets = True

    def reset(self) -> None:
        """Reset detector state."""
        self._state = "idle"
        self._cumulative_sample = 0
        self._pre_onset_count = 0
        self._pre_offset_count = 0
        self._iq_ring.clear()
        self._pending_event_type = None
        self._pending_pre_snap = []
        self._pending_post_buf = []
        self._pending_post_remaining = 0
        self._pending_pre_snap_start = 0
        self._idle_window_count = 1000   # normal reset: first block works like any other
        self._primed_active = False
        self._active_window_count = 0
