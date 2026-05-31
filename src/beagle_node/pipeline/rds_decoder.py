# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Rolling-window RDS decoder service.

Wraps the offline ``rds_demodulator.decode()`` and ``rds_block_sync.BlockSync``
in a streaming-friendly facade that lives in the live pipeline.  Audio
chunks are appended to a rolling buffer (default 2 seconds); periodically
the whole buffer is re-decoded offline and the resulting groups are
exposed for lookup.

Why a rolling buffer instead of a true streaming demodulator?
-------------------------------------------------------------
The offline demodulator (commit 1) uses scipy.signal.resample_poly +
np.convolve for the DSP chain.  Neither supports incremental processing.
Building a fully-streaming version requires reimplementing polyphase
resampling and FIR filtering with explicit tail buffers and Costas/AGC
state continuity.  That's a substantial rewrite for what is — for the
current Commit 3 (visibility only) — unnecessary: a 2-second rolling
window with periodic re-decode gives sub-second latency at ~10% CPU,
which is plenty for logging / DeltaComputer queries.

This module exposes:

  ``RDSDecoderService`` — accepts FM-demodulated audio chunks, periodically
  re-decodes the rolling buffer, and offers:

    * ``latest_groups()``       — list[Group] from the most recent decode
    * ``lookup(sample_index)``  — find the Group whose block-A boundary
                                  is closest to (and ≤) ``sample_index``,
                                  also computes (block_letter, bit_in_block)
                                  for the queried position
    * ``stats``                  — decode latency, group count, BLER mean

A future commit (Commit 4) can replace this with a true streaming
implementation when DeltaComputer's per-event latency budget demands it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from beagle_node.pipeline.rds_block_sync import (
    BLOCK_LENGTH,
    BlockSync,
    Group,
    Offset,
    get_block_number_for_offset,
)
from beagle_node.pipeline.rds_demodulator import DecodedBit, decode

logger = logging.getLogger(__name__)

# A whole RDS group is 4 blocks × 26 bits = 104 bits.
GROUP_BITS: int = 4 * BLOCK_LENGTH


@dataclass(frozen=True)
class BlockContext:
    """Decoded block context for one sample-index position in the MPX stream."""
    block_letter: str        # "A", "B", "C", "C'", or "D"
    bit_in_block: int        # 0..25, where 0 is the MSB of the info word
    group_pi: Optional[int]  # 16-bit PI if block A of the group decoded
    group_type: Optional[str]  # e.g. "0A", "2A"
    group_anchor_sample: float  # MPX sample of bit 0 of block A in this group
    bler: float              # 0.0..1.0 of the parent group


@dataclass
class DecoderStats:
    decode_count: int = 0
    last_decode_duration_ms: float = 0.0
    last_decode_input_seconds: float = 0.0
    last_group_count: int = 0
    last_bler_mean: float = float("nan")
    last_decode_wallclock_ms: float = 0.0


class RDSDecoderService:
    """
    Live-pipeline RDS decoder facade with rolling-window re-decode.

    Parameters
    ----------
    fs_in : float
        Sample rate of the input FM-demodulated MPX audio (Hz).
    window_seconds : float
        Length of the rolling buffer.  Longer = better lock quality on
        noisy signals but more CPU per decode.  Default 2 s.
    decode_interval_ms : float
        Minimum wallclock interval between decode runs.  Calls to
        ``push_audio`` faster than this just buffer; the next decode
        happens when this interval has elapsed.  Default 1000 ms.
    use_fec : bool
        Enable (26,16) burst-error correction.  Default True.

    Notes
    -----
    Each decode is *independent*: we don't carry BlockSync state across
    decodes.  Each re-decode starts a fresh BlockSync.  This is safe
    because the input buffer is long enough (≥ 1 s) to fully reacquire
    sync.  The cost is a few groups of warmup at the start of each
    buffer; the lookup() API filters those out.
    """

    def __init__(
        self,
        fs_in: float,
        *,
        window_seconds: float = 2.0,
        decode_interval_ms: float = 1000.0,
        use_fec: bool = True,
    ) -> None:
        if window_seconds < 0.5:
            raise ValueError("window_seconds must be ≥ 0.5 (decoder needs time to lock)")
        self._fs_in = float(fs_in)
        self._max_samples = int(window_seconds * fs_in)
        self._decode_interval_ms = float(decode_interval_ms)
        self._use_fec = bool(use_fec)
        self._buf: list[tuple[int, np.ndarray]] = []  # (start_sample, audio chunk)
        self._latest_groups: list[Group] = []
        self._latest_decode_buffer_start_sample: int = 0
        self._last_decode_wallclock_ms: float = -math.inf
        self.stats = DecoderStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_audio(self, audio: np.ndarray, start_sample: int) -> Optional[list[Group]]:
        """Append a chunk of FM-demodulated audio.

        If at least ``decode_interval_ms`` has elapsed since the last
        decode and the rolling buffer has ≥ 0.5 s, re-decode immediately
        and return the new groups.  Otherwise return None (just buffered).

        Parameters
        ----------
        audio : np.ndarray
            FM-demodulated MPX audio at fs_in.
        start_sample : int
            Absolute MPX-sample index of audio[0] in the continuous stream.

        Returns
        -------
        list[Group] | None
            The latest decoded groups when a decode just ran; None when
            this call only buffered (no decode triggered).
        """
        if len(audio) == 0:
            return None
        self._buf.append((int(start_sample), np.ascontiguousarray(audio)))
        # Trim oldest chunks while exceeding window size.
        total = sum(len(a) for _, a in self._buf)
        while total > self._max_samples and len(self._buf) > 1:
            total -= len(self._buf[0][1])
            self._buf.pop(0)

        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_decode_wallclock_ms < self._decode_interval_ms:
            return None
        if total < int(0.5 * self._fs_in):
            return None
        return self._run_decode(now_ms)

    def latest_groups(self) -> list[Group]:
        """All groups from the most recent decode of the rolling buffer."""
        return list(self._latest_groups)

    def reset(self) -> None:
        """
        Drop the rolling audio buffer and the latest decoded groups.

        Called by the pipeline when an SDR discontinuity (overflow,
        re-tune) makes the buffered audio inconsistent.  Without this
        reset the rolling buffer carries pre-discontinuity audio for
        up to ``window_seconds`` after the event, causing stale or
        garbled decodes during recovery.
        """
        self._buf.clear()
        self._latest_groups = []
        self._latest_decode_buffer_start_sample = 0
        self._last_decode_wallclock_ms = -math.inf
        # Stats counters are intentionally preserved — they're cumulative
        # for the lifetime of the process so the server can compute
        # per-deployment drop rates.

    def find_a_bit0_anchor(
        self,
        carrier_sample: float,
        max_lookback_samples: float,
    ) -> Optional[BlockContext]:
        """
        Find the most recent block-A bit-0 anchor at or before
        ``carrier_sample``, within ``max_lookback_samples``.

        Returns a BlockContext where ``group_anchor_sample`` is the
        demodulator-derived sample position of bit 0 of block A.
        The DeltaComputer uses this as a rough indicator to find the
        SyncEvent closest to it; that SyncEvent becomes the actual
        TDOA anchor (sub-µs precision from the pilot path).

        Returns None when the decoder has no decoded block-A bit 0 in
        the lookback window (e.g., during a BLER gap or right after
        reset).  Callers should fail-closed in that case.
        """
        best_ctx: Optional[BlockContext] = None
        best_sample: float = float("-inf")
        for g in self._latest_groups:
            blk_a = g.blocks[0]
            if blk_a is None or not blk_a.is_received or math.isnan(blk_a.sample_index):
                continue
            if blk_a.sample_index > carrier_sample:
                continue
            if carrier_sample - blk_a.sample_index > max_lookback_samples:
                continue
            if blk_a.sample_index > best_sample:
                best_sample = blk_a.sample_index
                best_ctx = BlockContext(
                    block_letter="A",
                    bit_in_block=0,
                    group_pi=g.pi,
                    group_type=g.group_type,
                    group_anchor_sample=blk_a.sample_index,
                    bler=g.bler,
                )
        return best_ctx

    def block_a_bit0_anchors(self) -> list[float]:
        """All decoded block-A bit-0 sample positions in the current
        rolling-buffer decode, ascending (MPX-sample coordinates).

        Used by the pipeline's global-K-slot plateau emitter
        (``_maybe_emit_anchor_plateau``, step 3) to enumerate every
        candidate block-A bit-0 in the decode window, compute each
        one's global epoch, and emit the one(s) that land on a shared
        K-multiple slot.  Differs from ``find_a_bit0_anchor`` (which
        returns only the single most-recent anchor at-or-before a query
        point) by returning the whole set so the emitter can target a
        *specific* epoch rather than reacting to the newest anchor.

        Float values carry the demodulator's sub-sample precision; the
        caller is responsible for the ``ceil``-to-int conversion that
        keeps the downstream matcher from regressing one group (see the
        ``math.ceil`` rationale in ``_maybe_emit_anchor_plateau``).
        """
        out: list[float] = []
        for g in self._latest_groups:
            blk_a = g.blocks[0]
            if blk_a is None or not blk_a.is_received or math.isnan(blk_a.sample_index):
                continue
            out.append(blk_a.sample_index)
        out.sort()
        return out

    def lookup(self, sample_index: float) -> Optional[BlockContext]:
        """
        Find the block context for an MPX sample position.

        Returns ``None`` if no decoded group contains this position, or
        if the parent block didn't fully decode.

        Currently used only by legacy callers / tests.  The matcher in
        DeltaComputer now uses ``find_a_bit0_anchor`` instead, which
        sidesteps the sync-event-vs-block-sample alignment problem that
        a ±½ bit-width tolerance can't solve when the two timing paths
        drift apart (e.g., on a different sample rate or SDR chain).
        """
        # Walk the groups looking for the one whose blocks span sample_index.
        # Groups are time-ordered (BlockSync emits in order).
        bit_width_samples = self._fs_in / 1187.5
        half_bit = bit_width_samples / 2.0
        for g in self._latest_groups:
            for blk in g.blocks:
                if blk is None or not blk.is_received or math.isnan(blk.sample_index):
                    continue
                # Each block is 26 bits.  We treat a sample as belonging to
                # this block if it's within ±½ bit-width of the block's bit-
                # boundary range.
                start = blk.sample_index - half_bit
                end = blk.sample_index + BLOCK_LENGTH * bit_width_samples - half_bit
                if start <= sample_index < end:
                    # Found
                    bit_in_block = int(
                        (sample_index - blk.sample_index + half_bit) / bit_width_samples
                    )
                    bit_in_block = max(0, min(BLOCK_LENGTH - 1, bit_in_block))
                    letter = _offset_letter(blk.offset)
                    return BlockContext(
                        block_letter=letter,
                        bit_in_block=bit_in_block,
                        group_pi=g.pi,
                        group_type=g.group_type,
                        group_anchor_sample=g.sample_index_first_bit,
                        bler=g.bler,
                    )
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_decode(self, now_ms: float) -> list[Group]:
        if not self._buf:
            return []
        t0 = time.monotonic()
        joined = np.concatenate([a for _, a in self._buf])
        buffer_start = self._buf[0][0]

        bits = decode(joined, self._fs_in)
        # Adjust bit sample_indices to absolute MPX-stream coords.
        abs_bits: list[DecodedBit] = [
            DecodedBit(b.bit, b.sample_index + buffer_start, b.symbol)
            for b in bits
        ]

        # Fresh BlockSync each decode (state-free across decodes).
        block_sync = BlockSync(use_fec=self._use_fec)
        groups: list[Group] = []
        for b in abs_bits:
            groups.extend(block_sync.push(b))
        # Flush any final partial group
        groups.extend(block_sync.flush())

        self._latest_groups = groups
        self._latest_decode_buffer_start_sample = buffer_start
        self._last_decode_wallclock_ms = now_ms

        duration_ms = (time.monotonic() - t0) * 1000.0
        bler_vals = [g.bler for g in groups if g.pi is not None]
        self.stats = DecoderStats(
            decode_count=self.stats.decode_count + 1,
            last_decode_duration_ms=duration_ms,
            last_decode_input_seconds=len(joined) / self._fs_in,
            last_group_count=len(groups),
            last_bler_mean=float(np.mean(bler_vals)) if bler_vals else float("nan"),
            last_decode_wallclock_ms=now_ms,
        )
        logger.debug(
            "RDS decode: %.1f s buffer → %d groups (mean BLER %.2f) in %.1f ms",
            self.stats.last_decode_input_seconds,
            self.stats.last_group_count,
            self.stats.last_bler_mean if not math.isnan(self.stats.last_bler_mean) else 0.0,
            self.stats.last_decode_duration_ms,
        )
        return groups


def _offset_letter(off: Offset) -> str:
    if off == Offset.A:
        return "A"
    if off == Offset.B:
        return "B"
    if off == Offset.C:
        return "C"
    if off == Offset.C_PRIME:
        return "C'"
    if off == Offset.D:
        return "D"
    return "?"
