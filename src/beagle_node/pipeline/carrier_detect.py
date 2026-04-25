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
    sample_index: int           # Absolute stream sample index of the snippet's FIRST sample
    power_db: float             # Instantaneous power at detection
    noise_floor_db: float = -100.0  # EMA of idle-state power; used to compute SNR
    iq_snippet: bytes = b""    # int8-interleaved IQ bytes for cross-correlation
    transition_start: int = 0  # Knee-search hint: samples into snippet, start of transition zone
    transition_end: int = 0    # Knee-search hint: samples into snippet, end of transition zone


@dataclass(frozen=True)
class CarrierOffset:
    """LMR carrier has disappeared."""
    sample_index: int           # Absolute stream sample index of the snippet's FIRST sample
    power_db: float
    iq_snippet: bytes = b""    # int8-interleaved IQ bytes for cross-correlation
    transition_start: int = 0  # Knee-search hint: samples into snippet, start of transition zone
    transition_end: int = 0    # Knee-search hint: samples into snippet, end of transition zone


@dataclass(frozen=True)
class CarrierPlateau:
    """Periodic snapshot during a sustained carrier.

    Emitted on a wall-clock timer while the detector is in the active
    state, anchored (where possible) to a sync-pilot bit boundary so that
    plateau snippets from independent nodes cover the same physical time
    window.  Has no transition: ``transition_start`` / ``transition_end``
    bound the entire snippet so the server's xcorr / PHAT can run over
    the full plateau content.
    """
    sample_index: int           # Absolute stream sample index of the snippet's FIRST sample
    power_db: float
    iq_snippet: bytes = b""
    transition_start: int = 0   # 0 — full-snippet xcorr/PHAT window
    transition_end: int = 0     # set to len(iq_snippet) // 2 by encoder (full-snippet hint)


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
        auto_threshold_margins: bool = False,
        onset_margin_db: float = 12.0,
        offset_margin_db: float = 6.0,
        auto_threshold_update_interval_s: float = 2.0,
        plateau_event_interval_s: float = 0.0,
        plateau_max_per_active: int = 0,
    ) -> None:
        if plateau_max_per_active < 0:
            raise ValueError(
                f"plateau_max_per_active must be >= 0, got {plateau_max_per_active}"
            )
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
        if offset_margin_db >= onset_margin_db:
            raise ValueError(
                f"offset_margin_db ({offset_margin_db}) must be < "
                f"onset_margin_db ({onset_margin_db})"
            )
        if auto_threshold_update_interval_s <= 0.0:
            raise ValueError(
                f"auto_threshold_update_interval_s must be > 0, "
                f"got {auto_threshold_update_interval_s}"
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

        # Plateau-event emission while the carrier is sustained.  0.0
        # disables.  When > 0, emit a CarrierPlateau every N seconds of
        # wall-clock time while the detector is in the active state.
        # The snippet is taken from the ring + recently-processed windows;
        # cross-node alignment comes from NTP-synced wall clock (~10 ms),
        # and the per-event timing precision (sync_to_snippet_start_ns)
        # is sub-microsecond as for onset/offset.  Plateau snippets cover
        # the same physical-time window across nodes (modulo NTP skew)
        # because each node's wall-clock interval boundaries are nearly
        # simultaneous.
        self._plateau_interval_s: float = float(plateau_event_interval_s)
        self._last_plateau_wall_s: float | None = None

        # Stuck-active safety: cap plateau emissions per active period.
        # 0 = unlimited (legacy).  When > 0, after this many plateaus in a
        # single active period the emitter mutes itself; emission resumes
        # only after a state -> idle -> active transition (i.e. an offset
        # then a fresh onset).  Catches "stuck active" failures that would
        # otherwise emit one plateau per second indefinitely, exhausting
        # journal/disk on resource-constrained hosts.
        self._plateau_max_per_active: int = int(plateau_max_per_active)
        self._plateau_count_this_active: int = 0
        self._plateau_cap_warned: bool = False

        # Onset-edge tracker for first-plateau-of-active-period coverage.
        # Records the absolute target-stream sample at which the carrier
        # transitioned to active.  The plateau emitter requires the snippet
        # boundary to be at least edge_clearance samples *past* this value
        # before firing the first plateau, so the snippet contains clean
        # post-onset content rather than straddling the rising edge.
        # None when state is not "active".
        self._active_onset_sample: int | None = None

        self._state: _State = "idle"
        self._cumulative_sample: int = 0   # total samples seen so far

        # Invariant guard: track the last-emitted event type so we can
        # refuse to emit two events of the same type back-to-back.  This
        # catches any silent state transition path (cancel_pending,
        # prime_state, reset, and any future additions) that moves the
        # state machine between "idle" and "active" without emitting the
        # corresponding boundary event, which would otherwise allow a
        # spurious duplicate onset or offset on the next threshold
        # crossing.  All event-emission sites go through ``_emit``.
        self._last_emitted_type: str | None = None
        # Exponential moving average of power during idle (no-carrier) windows.
        # Initialised to offset_threshold_db (a reasonable upper bound on noise).
        # Alpha = 0.01 -> time constant ~ 100 windows (~ 100 ms at 64 kHz / 64-sample window).
        self._noise_floor_db: float = float(offset_threshold_db)
        self._noise_floor_alpha: float = 0.01

        # Auto-tracking thresholds relative to the EMA noise floor.  When
        # enabled, onset/offset thresholds are periodically set to
        # noise_floor + margin_db.  Static onset_db/offset_db only apply
        # until the EMA has warmed up enough (warmup_floor_updates) for the
        # tracked value to be meaningful.
        self._auto_threshold_margins: bool = bool(auto_threshold_margins)
        self._onset_margin_db: float = float(onset_margin_db)
        self._offset_margin_db: float = float(offset_margin_db)
        # Convert the update cadence from seconds to detector windows.
        self._auto_update_interval_windows: int = max(
            1, int(round(auto_threshold_update_interval_s * self._rate / self._window))
        )
        self._windows_since_auto_update: int = 0
        # Count of idle windows observed where the noise-floor EMA was
        # actually updated; this is what we use to determine "warmup"
        # rather than wall-clock windows, because the EMA is only moved
        # by idle-state samples.
        self._auto_floor_updates: int = 0
        # Require enough EMA updates for the estimate to be ~5 time constants
        # in from the initial (offset_threshold_db) value, i.e. settled to
        # within ~0.7% of the true floor.  alpha=0.01 -> 5/alpha = 500 updates.
        self._auto_warmup_floor_updates: int = max(
            50, int(round(5.0 / self._noise_floor_alpha))
        )
        # Minimum change to apply (dB) — avoids churning update_thresholds
        # on every sub-dB EMA jitter.  Small adjustments still happen; only
        # changes >= _auto_log_change_db are logged per-update.
        self._auto_min_change_db: float = 0.5
        # Threshold (dB) above which a per-update debug->info escalation
        # would be worth calling out; kept at debug below this to avoid
        # filling logs with tiny tracking adjustments.
        self._auto_log_change_db: float = 2.0
        # Maximum absolute threshold value (dBFS) for safety, so runaway
        # noise or strong interference doesn't push thresholds above where a
        # legitimate full-scale carrier would lie.
        self._auto_max_onset_db: float = -10.0
        # Heartbeat: emit a single info log every N windows confirming that
        # auto-tracking is active and reporting the current settings.
        # 10 minutes at 64 kHz / 64-sample window = 600 000 windows.
        self._auto_heartbeat_interval_windows: int = max(
            1, int(round(600.0 * self._rate / self._window))
        )
        self._windows_since_auto_heartbeat: int = 0
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
        #
        # Minimum viable depth: the concatenated pre+post buffer must be long
        # enough to fill snippet_samples, otherwise the emitted snippet is
        # silently truncated.  min_ring = ceil(snippet_samples/window) - post.
        _snippet_windows = max(1, -(-self._snippet_samples // self._window))
        _min_ring_for_full_snippet = max(1, _snippet_windows - self._post_windows)
        if ring_lookback_windows is None:
            _ring_capacity = max(_snippet_windows * 3, _min_ring_for_full_snippet)
        else:
            _ring_capacity = int(ring_lookback_windows)
            if _ring_capacity < _min_ring_for_full_snippet:
                logger.warning(
                    "ring_lookback_windows=%d is too small to fill snippet_samples=%d "
                    "at window_samples=%d with snippet_post_windows=%d. "
                    "Emitted snippets will be truncated to %d samples (%.1f ms at %.0f Hz) "
                    "instead of %d samples. Raise ring_lookback_windows to >= %d "
                    "(or leave it unset to auto-size).",
                    _ring_capacity, self._snippet_samples, self._window, self._post_windows,
                    (_ring_capacity + self._post_windows) * self._window,
                    (_ring_capacity + self._post_windows) * self._window / self._rate * 1000.0,
                    self._rate,
                    self._snippet_samples,
                    _min_ring_for_full_snippet,
                )
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
        plateau_event_interval_s: float | None = None,
        plateau_max_per_active: int | None = None,
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
        if plateau_event_interval_s is not None:
            if plateau_event_interval_s < 0.0:
                raise ValueError(
                    f"plateau_event_interval_s must be >= 0, got {plateau_event_interval_s}"
                )
            # Reset cadence anchor so the new interval takes effect on the
            # next aligned wall-clock boundary; otherwise we'd inherit the
            # old interval's last-fired timestamp.
            if self._plateau_interval_s != plateau_event_interval_s:
                self._last_plateau_wall_s = None
            self._plateau_interval_s = float(plateau_event_interval_s)
        if plateau_max_per_active is not None:
            if plateau_max_per_active < 0:
                raise ValueError(
                    f"plateau_max_per_active must be >= 0, got {plateau_max_per_active}"
                )
            self._plateau_max_per_active = int(plateau_max_per_active)
            # Re-arm the warn flag so a fresh cap value gets a fresh WARN
            # if/when it trips.
            self._plateau_cap_warned = False
        logger.info(
            "Thresholds updated: onset=%.1f offset=%.1f hold=%d release=%d "
            "min_active_for_offset=%d plateau_event_interval_s=%.2f "
            "plateau_max_per_active=%d",
            self._onset_db, self._offset_db, self._min_hold, self._min_release,
            self._min_active_for_offset, self._plateau_interval_s,
            self._plateau_max_per_active,
        )

    def _apply_auto_thresholds(self) -> None:
        """Re-evaluate onset/offset thresholds from the current noise floor EMA.

        Called periodically from ``process()`` once the noise-floor EMA has
        warmed up.  Computes target thresholds as
        ``noise_floor + {onset,offset}_margin_db``, clamps onset to
        ``_auto_max_onset_db`` for safety, and applies via
        ``update_thresholds`` if the change from the current settings is
        larger than ``_auto_min_change_db`` (to avoid churn on sub-dB EMA
        jitter).  Small adjustments (< ``_auto_log_change_db``) are applied
        silently at debug level; larger drifts get an info log so operators
        see unusual noise-floor movement.  A periodic info heartbeat
        (every ~10 minutes, emitted from ``process()``) confirms that
        auto-tracking is active regardless of adjustment size.
        """
        floor = self._noise_floor_db
        new_onset = min(floor + self._onset_margin_db, self._auto_max_onset_db)
        new_offset = min(floor + self._offset_margin_db,
                         new_onset - 1.0)  # preserve at least 1 dB hysteresis
        max_change = max(abs(new_onset - self._onset_db),
                         abs(new_offset - self._offset_db))
        if max_change < self._auto_min_change_db:
            return
        log_fn = logger.info if max_change >= self._auto_log_change_db else logger.debug
        log_fn(
            "Auto-threshold update: noise_floor=%.1f dB -> onset=%.1f offset=%.1f "
            "(was onset=%.1f offset=%.1f, max delta %.1f dB)",
            floor, new_onset, new_offset, self._onset_db, self._offset_db,
            max_change,
        )
        # Suppress update_thresholds' own info log for sub-threshold changes so
        # small tracking adjustments don't fill operator logs.
        prev_onset, prev_offset = self._onset_db, self._offset_db
        if max_change < self._auto_log_change_db:
            self._onset_db = new_onset
            self._offset_db = new_offset
        else:
            self.update_thresholds(
                onset_threshold_db=new_onset,
                offset_threshold_db=new_offset,
            )
        # Verify monotonicity was preserved by the clamp above.
        assert self._offset_db < self._onset_db, (
            f"auto update broke hysteresis: onset={self._onset_db} "
            f"offset={self._offset_db} (prev {prev_onset}/{prev_offset})"
        )

    # ------------------------------------------------------------------
    # Event emission invariant guard
    # ------------------------------------------------------------------

    def _emit(
        self, events_list: list,
        ev: "CarrierOnset | CarrierOffset | CarrierPlateau",
    ) -> None:
        """Append ``ev`` to ``events_list`` iff it doesn't duplicate the
        previously-emitted event's type.

        A CarrierDetector is a two-state machine (idle <-> active).  By
        definition each edge event marks a transition between those
        states, so the emitted edge type MUST alternate
        onset/offset/onset/...  If we ever try to emit two onsets (or two
        offsets) back-to-back it means some path silently transitioned
        the state without emitting the corresponding boundary event —
        the main culprits are ``cancel_pending`` on discontinuity,
        ``prime_state`` at block start, and ``reset``.

        Plateau events are not edge events and don't participate in the
        alternation invariant: many plateaus may be emitted in a row
        between an onset and an offset.  They bypass the duplicate-type
        check and don't update ``_last_emitted_type``.
        """
        if isinstance(ev, CarrierPlateau):
            events_list.append(ev)
            return
        etype = "onset" if isinstance(ev, CarrierOnset) else "offset"
        if etype == self._last_emitted_type:
            logger.warning(
                "State-machine invariant violated: %s immediately after "
                "%s at sample %d; suppressing duplicate.  This indicates "
                "a silent state transition (discontinuity, prime_state, "
                "or reset) bypassed the boundary event.",
                etype, self._last_emitted_type, ev.sample_index,
            )
            return
        events_list.append(ev)
        self._last_emitted_type = etype

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

            # Periodic auto-threshold tracking based on the noise-floor EMA.
            self._windows_since_auto_update += 1
            if (self._auto_threshold_margins
                    and self._windows_since_auto_update >= self._auto_update_interval_windows):
                self._windows_since_auto_update = 0
                if self._auto_floor_updates >= self._auto_warmup_floor_updates:
                    self._apply_auto_thresholds()
            # Heartbeat: confirm auto-tracking is alive even during long
            # periods with sub-threshold adjustments (those log at debug).
            if self._auto_threshold_margins:
                self._windows_since_auto_heartbeat += 1
                if self._windows_since_auto_heartbeat >= self._auto_heartbeat_interval_windows:
                    self._windows_since_auto_heartbeat = 0
                    if self._auto_floor_updates >= self._auto_warmup_floor_updates:
                        logger.info(
                            "Auto-threshold active: noise_floor=%.1f dB  "
                            "onset=%.1f  offset=%.1f  (margins %.1f/%.1f dB)",
                            self._noise_floor_db, self._onset_db, self._offset_db,
                            self._onset_margin_db, self._offset_margin_db,
                        )

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
                            self._active_onset_sample = window_sample
                            self._pre_onset_count = 0
                            self._pre_offset_count = 0
                    else:
                        self._pre_onset_count = 0

                if self._pending_post_remaining <= 0 or opposite_fired:
                    ev: CarrierOnset | CarrierOffset
                    if self._pending_event_type == "onset":
                        _onset_bytes, _snip_start, _t_start, _t_end = self._encode_combined(
                            self._pending_pre_snap, self._pending_post_buf
                        )
                        ev = CarrierOnset(
                            sample_index=self._pending_pre_snap_start + _snip_start,
                            power_db=self._pending_power_db,
                            noise_floor_db=self._pending_noise_floor_db,
                            iq_snippet=_onset_bytes,
                            transition_start=_t_start,
                            transition_end=_t_end,
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
                                    "snippet_start_in_buf": _snip_start,
                                    "transition_window": [_t_start, _t_end],
                                    "window_sample": self._pending_sample_index,
                                    "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                                    "power_db": round(ev.power_db, 1),
                                }),
                            )
                    else:
                        _offset_bytes, _snip_start, _t_start, _t_end = self._encode_offset_snippet(
                            self._pending_pre_snap, self._pending_post_buf
                        )
                        ev = CarrierOffset(
                            sample_index=self._pending_pre_snap_start + _snip_start,
                            power_db=self._pending_power_db,
                            iq_snippet=_offset_bytes,
                            transition_start=_t_start,
                            transition_end=_t_end,
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
                                    "snippet_start_in_buf": _snip_start,
                                    "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                                    "power_db": round(ev.power_db, 1),
                                }),
                            )
                    self._emit(events, ev)
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
                                _off_bytes, _snip_start, _t_s, _t_e = self._encode_offset_snippet(_pre)
                                self._emit(events, CarrierOffset(
                                    sample_index=_pre_start + _snip_start,
                                    power_db=power_db,
                                    iq_snippet=_off_bytes,
                                    transition_start=_t_s,
                                    transition_end=_t_e,
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
                                self._emit(events, CarrierOnset(
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
                        self._active_onset_sample = window_sample
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
                                self._emit(events, CarrierOnset(
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
                    # Count EMA updates for auto-threshold warmup tracking.
                    self._auto_floor_updates += 1

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
                            _off_bytes, _snip_start, _t_s, _t_e = self._encode_offset_snippet(_pre)
                            self._emit(events, CarrierOffset(
                                sample_index=_pre_start + _snip_start,
                                power_db=power_db,
                                iq_snippet=_off_bytes,
                                transition_start=_t_s,
                                transition_end=_t_e,
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
                _onset_bytes, _snip_start, _t_start, _t_end = self._encode_combined(
                    self._pending_pre_snap, self._pending_post_buf
                )
                ev = CarrierOnset(
                    sample_index=self._pending_pre_snap_start + _snip_start,
                    power_db=self._pending_power_db,
                    noise_floor_db=self._pending_noise_floor_db,
                    iq_snippet=_onset_bytes,
                    transition_start=_t_start,
                    transition_end=_t_end,
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
                            "snippet_start_in_buf": _snip_start,
                            "transition_window": [_t_start, _t_end],
                            "window_sample": self._pending_sample_index,
                            "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                            "power_db": round(ev.power_db, 1),
                            "partial_flush": True,
                        }),
                    )
            else:
                _offset_bytes, _snip_start, _t_start, _t_end = self._encode_offset_snippet(
                    self._pending_pre_snap, self._pending_post_buf
                )
                ev = CarrierOffset(
                    sample_index=self._pending_pre_snap_start + _snip_start,
                    power_db=self._pending_power_db,
                    iq_snippet=_offset_bytes,
                    transition_start=_t_start,
                    transition_end=_t_end,
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
                            "snippet_start_in_buf": _snip_start,
                            "buf_len": sum(len(w) for w in self._pending_pre_snap) + sum(len(w) for w in self._pending_post_buf),
                            "power_db": round(ev.power_db, 1),
                            "partial_flush": True,
                        }),
                    )
            self._emit(events, ev)
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

        # Plateau emission: while the carrier is sustained, periodically
        # snap a snippet of the recent IQ for cross-node averaging.  Done
        # after the FSM run so we don't interleave with onset/offset
        # state transitions; only fires while in active state and not
        # currently pending an onset/offset deferred emission.
        self._maybe_emit_plateau(events, start_sample, n_windows)

        return events

    def _maybe_emit_plateau(self, events: list, start_sample: int, n_windows: int) -> None:
        """If a plateau event is due, append a CarrierPlateau to ``events``.

        Eligibility:
          - plateau_event_interval_s > 0
          - state == "active"
          - no pending onset/offset deferred emission
          - IQ ring has at least snippet_samples worth of data
          - snippet boundary is past the onset transition + edge clearance
            (so the snippet is clean of the rising-edge transient)

        Cadence:
          - First plateau of an active period fires as soon as the IQ ring
            clears the onset edge — typically ~snippet_duration ms after
            the onset (e.g. ~65 ms for 16384 samples at 250 kHz).  This
            ensures even short transmissions (>= snippet_duration) yield
            at least one plateau measurement.
          - Subsequent plateaus fire on the wall-clock interval grid
            (floor(now/interval)*interval), so all nodes' plateaus are
            phase-locked to within NTP precision (~10 ms) for cross-node
            PHAT correlation.

        The snippet is taken from the most recent ring contents; the
        ``sample_index`` is the absolute target-stream sample of the
        snippet's first sample.  The DeltaComputer downstream will match
        this against the most recent sync-pilot bit boundary, giving a
        sub-microsecond ``sync_to_snippet_start_ns`` for the cross-node
        correlation.
        """
        if self._plateau_interval_s <= 0.0:
            return
        if self._state != "active":
            # Reset stuck-active counter, cadence anchor, and onset-edge
            # tracker so the next active period starts fresh: the WARN-once
            # flag re-arms, and the first plateau on the next active period
            # fires ASAP (after onset-edge clearance) rather than waiting
            # for the next wall-clock interval boundary.
            self._plateau_count_this_active = 0
            self._plateau_cap_warned = False
            self._last_plateau_wall_s = None
            return
        if self._pending_event_type is not None:
            return
        # Stuck-active safety cap.  After plateau_max_per_active emissions
        # in a single active period the emitter mutes itself; only a
        # state -> idle -> active transition (offset + onset) re-enables it.
        if (
            self._plateau_max_per_active > 0
            and self._plateau_count_this_active >= self._plateau_max_per_active
        ):
            if not self._plateau_cap_warned:
                logger.warning(
                    "Plateau cap reached at N=%d emissions during a single "
                    "active period; stuck-active suspected, emissions paused "
                    "until next idle->active cycle (offset+onset).",
                    self._plateau_max_per_active,
                )
                self._plateau_cap_warned = True
            return

        # Need enough recent IQ to fill snippet_samples.
        ring_total_samples = sum(len(w) for w in self._iq_ring)
        if ring_total_samples < self._snippet_samples:
            return

        # Use the most recent snippet_samples from the ring.
        # last_window_end_sample = sample of the last sample of the last
        # window we accumulated (== start_sample + n_windows*window - 1).
        last_window_end = start_sample + n_windows * self._window
        snippet_first_sample = last_window_end - self._snippet_samples

        # Onset-edge clearance: the first plateau of an active period must
        # wait until the IQ ring is filled with samples taken AFTER the
        # rising edge.  Otherwise the snippet straddles the edge and PHAT
        # locks onto the transition rather than the modulation content.
        # edge_clearance gives a small margin past the bare snippet_first
        # >= onset_sample condition for post-onset transient ringing.
        edge_clearance = self._window * 4
        if (
            self._active_onset_sample is not None
            and snippet_first_sample < self._active_onset_sample + edge_clearance
        ):
            return

        import time as _time
        now = _time.time()
        import math
        if self._last_plateau_wall_s is None:
            # First plateau of this active period.  The ring is clean of
            # onset-edge content (above check passed), so emit immediately
            # without waiting for the next wall-clock interval boundary;
            # this catches short transmissions that wouldn't otherwise
            # produce a plateau.  Anchor the cadence to the wall-clock
            # grid for subsequent plateaus so they remain phase-locked
            # across nodes.
            self._last_plateau_wall_s = (
                math.floor(now / self._plateau_interval_s) * self._plateau_interval_s
                - self._plateau_interval_s
            )
            # Falls through to emission below (now - last == interval, so
            # the elapsed-interval check passes).
        # If we've fallen far behind the schedule (e.g. an exceptionally
        # long active period crossed many intervals between process()
        # calls), snap forward to the current interval boundary rather
        # than emitting back-to-back plateaus to "catch up".  Without
        # this, every process() call could fire a plateau until
        # _last_plateau_wall_s catches up to wall time — bursting at
        # the chunk-arrival rate.
        if now - self._last_plateau_wall_s > 2 * self._plateau_interval_s:
            self._last_plateau_wall_s = (
                math.floor(now / self._plateau_interval_s) * self._plateau_interval_s
            )
        if now - self._last_plateau_wall_s < self._plateau_interval_s:
            return

        iq_cat = np.concatenate(list(self._iq_ring))
        iq_trim = iq_cat[-self._snippet_samples:]
        scale = float(np.max(np.abs(iq_trim))) + 1e-30
        normed = iq_trim / scale
        int8_ri = np.empty(len(normed) * 2, dtype=np.int8)
        int8_ri[0::2] = np.clip(np.round(normed.real * 127), -127, 127).astype(np.int8)
        int8_ri[1::2] = np.clip(np.round(normed.imag * 127), -127, 127).astype(np.int8)

        # Plateau "transition" hint covers the full snippet — no knee, the
        # server's PHAT/xcorr operates on the whole content.
        ev = CarrierPlateau(
            sample_index=snippet_first_sample,
            power_db=self._noise_floor_db + 20.0,  # rough; not used downstream
            iq_snippet=int8_ri.tobytes(),
            transition_start=0,
            transition_end=self._snippet_samples,
        )
        self._emit(events, ev)
        self._plateau_count_this_active += 1
        # Advance the cadence by exactly one interval so subsequent emissions
        # stay phase-locked to the original schedule.  (The "far behind" snap
        # above guarantees we won't emit more than once per process() call.)
        self._last_plateau_wall_s += self._plateau_interval_s
        logger.debug(
            "CarrierPlateau emitted at sample %d (interval %.1f s)",
            snippet_first_sample, self._plateau_interval_s,
        )

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
    ) -> tuple[bytes, int, int, int]:
        """
        Encode a natural-position onset snippet.

        Concatenates pre-event and post-event IQ and trims to snippet_samples
        centered on the detection point.  The IQ data is NOT repositioned —
        the PA transition stays at its natural position within the snippet,
        preserving the sample-boundary timing relationship needed for TDOA.

        Returns
        -------
        (bytes, snippet_start_in_iqcat, transition_start, transition_end)
            bytes: the encoded snippet.
            snippet_start_in_iqcat: index within the concatenated pre+post
                buffer where the trimmed snippet begins.  The caller combines
                this with ``pre_snap_start`` (the absolute stream sample of
                iq_cat[0]) to get the snippet's absolute stream sample index,
                which is what gets shipped to the server as the timing anchor.
            transition_start, transition_end: samples-into-snippet markers
                for the detection zone, used server-side as knee-search hints.
        """
        assert pre_snap or post_buf, "Both pre_snap and post_buf are empty - cannot happen"
        parts = list(pre_snap) + list(post_buf or [])
        iq_cat = np.concatenate(parts)
        assert len(iq_cat) >= 32, "Onset IQ data too short - cannot happen"

        # Detection point: boundary between pre_snap and post_buf.
        pre_len = sum(len(w) for w in pre_snap)
        det_idx = min(pre_len, len(iq_cat) - 1)

        # Trim to snippet_samples with detection at the midpoint.  Detection
        # sits a bit before the knee (knee = top of rise, detection = threshold
        # crossing during rise), so the detection-to-knee span straddles the
        # centre of the snippet with generous symmetric context on both sides
        # for the server's knee finder / xcorr.
        pre_target = self._snippet_samples // 2
        start = max(0, det_idx - pre_target)
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

        # Transition zone: detection is near bottom of rise; give the server
        # a generous window up the ramp so its knee-finder has room to work.
        t_start = max(0, det_idx - start)
        t_end = min(len(iq_trim), det_idx - start + 8 * self._window)
        return int8_ri.tobytes(), start, t_start, t_end

    def _encode_offset_snippet(
        self,
        pre_snap: list[np.ndarray],
        post_buf: list[np.ndarray] | None = None,
    ) -> tuple[bytes, int, int, int]:
        """
        Encode a natural-position offset snippet.

        Concatenates pre-event and post-event IQ and trims to snippet_samples
        centered on the detection point.  The IQ data is NOT repositioned —
        the PA transition stays at its natural position within the snippet,
        preserving the sample-boundary timing relationship needed for TDOA.

        Returns (bytes, snippet_start_in_iqcat, transition_start, transition_end).
        ``snippet_start_in_iqcat`` is the index within the concatenated pre+post
        buffer where the trimmed snippet begins; the caller combines it with
        ``pre_snap_start`` to get the snippet's absolute stream sample index.
        """
        parts = list(pre_snap) + list(post_buf or [])
        assert parts, "No IQ data for offset snippet - cannot happen"
        iq_cat = np.concatenate(parts)
        assert len(iq_cat) >= 32, "Offset IQ data too short - cannot happen"

        # Detection point: boundary between pre_snap and post_buf.
        pre_len = sum(len(w) for w in pre_snap)
        det_idx = min(pre_len, len(iq_cat) - 1)

        # Trim to snippet_samples with detection at the midpoint.  Detection
        # sits a bit after the knee (knee = top of fall, detection = threshold
        # crossing during fall), so the knee-to-detection span straddles the
        # centre of the snippet with generous symmetric context on both sides
        # for the server's knee finder / xcorr.
        pre_target = self._snippet_samples // 2
        start = max(0, det_idx - pre_target)
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

        # Transition zone: detection is near bottom of fall; plateau is a few
        # windows behind.  The server's knee-finder uses this window.
        t_start = max(0, det_idx - start - 8 * self._window)
        t_end = min(len(iq_trim), det_idx - start)
        return int8_ri.tobytes(), start, t_start, t_end

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
                # Treat prime_state-induced active as having an onset at the
                # current sample boundary: the carrier is already on, but we
                # need to wait for snippet_samples of post-prime IQ to
                # accumulate before the first plateau snippet is clean of
                # whatever was in the ring before this freq-hop block.
                self._active_onset_sample = self._cumulative_sample
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
        # prime_state may flip state between active <-> idle without
        # emitting a boundary event.  Reset the emission invariant
        # tracker so the next event of either type is accepted.
        self._last_emitted_type = None
        # Arm snippet transition validation for the upcoming process() call.
        self._validate_snippets = True

    def cancel_pending(self) -> None:
        """Discard any in-flight event (onset seen, collecting post-windows).

        Called on sample discontinuity — the pending event's sample position
        is relative to pre-gap timing and would produce a corrupt sync_delta.
        """
        if self._pending_event_type is not None:
            logger.warning(
                "Discontinuity: cancelled pending %s event "
                "(had %d/%d post windows)",
                self._pending_event_type,
                len(self._pending_post_buf),
                len(self._pending_post_buf) + self._pending_post_remaining,
            )
            self._pending_event_type = None
            self._pending_pre_snap = []
            self._pending_post_buf = []
            self._pending_post_remaining = 0
        # Also return to idle — if we were in "active" state tracking a
        # carrier, the sample continuity is broken and we should re-detect.
        self._state = "idle"
        self._pre_onset_count = 0
        self._pre_offset_count = 0
        # Require a fresh idle period before accepting a new onset.
        # Without this reset a carrier that persists past the discontinuity
        # (very likely, since we were in "active" state pre-gap) would
        # immediately re-trigger an onset on the next window, producing a
        # "mid-transmission" event whose snippet has no real noise->carrier
        # ramp for the server's knee finder to work with.  The
        # _min_idle_for_onset guard in the FSM then suppresses the first
        # onset attempt until enough genuinely-idle windows have passed;
        # when the carrier really drops, a clean offset fires and
        # subsequent onsets are real detections.
        self._idle_window_count = 0
        # The invariant-guard helper (_emit) assumes we know what the
        # previously emitted type was; after a forced state flip that
        # skipped a boundary event, the "last emitted" no longer matches
        # the physical state.  Reset to None so the very next emit of
        # either type is accepted.
        self._last_emitted_type = None

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
        self._active_onset_sample = None
        self._plateau_count_this_active = 0
        self._plateau_cap_warned = False
        self._last_plateau_wall_s = None
        # Reset the emission invariant tracker: next event of either type
        # is accepted (see _emit docstring).
        self._last_emitted_type = None
