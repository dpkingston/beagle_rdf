# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
RDS block synchronization, syndrome decoding, and burst-error FEC.

Consumes the bit stream from ``rds_demodulator`` and produces ``Group``
records (the standard IEC 62106 unit: four 26-bit blocks decoded as
A, B, C or C', D).

Architecture
------------
Ported from windytan/redsea ``src/block_sync.cc``.  The (26,16) cyclic
code parity-check matrix and offset-word syndromes are taken verbatim
from IEC 62106:2015 Annex B.

Sync acquisition uses the 3-pulse mechanism: we keep a short history of
sync-pulse candidates (positions where a syndrome matched an offset
word) and declare lock when three of them line up at the correct cyclic
distances.  This tolerates a gap (up to ~6 blocks of bad data) between
the three pulses, which makes acquisition robust at low SNR.

Burst-error FEC corrects 1-bit and 2-bit error bursts by precomputing
the syndrome → error-vector map for every offset word.  This is the
Kopitz-Marks (1999) restricted error-correction approach used by
redsea.  Single- and double-bit bursts are corrected; longer bursts
are flagged but not corrected.

Output
------
``BlockSync.push(decoded_bit)`` returns an iterable of ``Group`` records;
each Group carries:
  - the four ``Block`` records (with raw 26 bits, decoded 16-bit info
    word, error/correction status, and sample index of the first bit)
  - the 16-bit PI from block A (if successfully decoded)
  - the group type string (e.g. ``"0A"``, ``"2A"``) from block B
  - the group's anchor sample index (first bit of block A in the
    input MPX stream — used by downstream consumers for sync alignment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Iterator, Optional

from beagle_node.pipeline.rds_demodulator import DecodedBit

BLOCK_LENGTH: int = 26
BLOCK_BITMASK: int = (1 << BLOCK_LENGTH) - 1
CHECKWORD_LENGTH: int = 10
INFO_LENGTH: int = 16


# ---------------------------------------------------------------------------
# Offset words (IEC 62106:2015 Annex B Table B.1)
# ---------------------------------------------------------------------------


class Offset(IntEnum):
    A = 0
    B = 1
    C = 2
    C_PRIME = 3
    D = 4
    INVALID = 5


# Offset words XORed into the checkword to make the code self-synchronizing.
OFFSET_WORDS: dict[Offset, int] = {
    Offset.A:        0b0011111100,
    Offset.B:        0b0110011000,
    Offset.C:        0b0101101000,
    Offset.C_PRIME:  0b1101010000,
    Offset.D:        0b0110110100,
}

# Pre-computed syndrome for each offset word (i.e., what calculateSyndrome
# returns when the only 1-bits in the input are those of the offset word).
SYNDROME_TO_OFFSET: dict[int, Offset] = {
    0b1111011000: Offset.A,
    0b1111010100: Offset.B,
    0b1001011100: Offset.C,
    0b1111001100: Offset.C_PRIME,
    0b1001011000: Offset.D,
}


# ---------------------------------------------------------------------------
# (26, 16) cyclic code parity-check matrix
# Ported verbatim from redsea src/block_sync.cc::calculateSyndrome.
# ---------------------------------------------------------------------------

_PARITY_CHECK_MATRIX: tuple[int, ...] = (
    0b1000000000, 0b0100000000, 0b0010000000, 0b0001000000, 0b0000100000,
    0b0000010000, 0b0000001000, 0b0000000100, 0b0000000010, 0b0000000001,
    0b1011011100, 0b0101101110, 0b0010110111, 0b1010000111, 0b1110011111,
    0b1100010011, 0b1101010101, 0b1101110110, 0b0110111011, 0b1000000001,
    0b1111011100, 0b0111101110, 0b0011110111, 0b1010100111, 0b1110001111,
    0b1100011011,
)


def calculate_syndrome(word26: int) -> int:
    """Compute the 10-bit parity-check syndrome of a 26-bit RDS block.

    Implements the matrix multiplication H · word^T mod 2 where H is the
    parity-check matrix of the (26, 16) cyclic code.

    See IEC 62106:2015 Annex B.1.1.
    """
    result = 0
    for k in range(BLOCK_LENGTH):
        if (word26 >> k) & 1:
            result ^= _PARITY_CHECK_MATRIX[BLOCK_LENGTH - 1 - k]
    return result


def get_offset_for_syndrome(syndrome: int) -> Offset:
    return SYNDROME_TO_OFFSET.get(syndrome, Offset.INVALID)


def get_block_number_for_offset(offset: Offset) -> int:
    """Block number 0..3 (A, B, C/C', D) for the given offset word."""
    if offset == Offset.A:
        return 0
    if offset == Offset.B:
        return 1
    if offset in (Offset.C, Offset.C_PRIME):
        return 2
    if offset == Offset.D:
        return 3
    # INVALID has no canonical block; callers should guard against this.
    return 0


def get_next_offset_for(offset: Offset) -> Offset:
    """The offset that should follow ``offset`` in the cyclic A/B/C/D sequence."""
    if offset == Offset.A:
        return Offset.B
    if offset == Offset.B:
        return Offset.C
    if offset == Offset.C or offset == Offset.C_PRIME:
        return Offset.D
    if offset == Offset.D:
        return Offset.A
    return Offset.A


# ---------------------------------------------------------------------------
# Burst-error FEC: precomputed syndrome → error vector table
# ---------------------------------------------------------------------------

def _make_error_lookup_table() -> dict[Offset, dict[int, int]]:
    """
    For each offset word, build a syndrome → error-vector map covering all
    1-bit and 2-bit consecutive-bit error patterns within the 26-bit block.

    See Kopitz & Marks (1999), "RDS: The Radio Data System", p. 224:
    "the error-correction system should be enabled, but should be
    restricted by attempting to correct bursts of errors spanning one or
    two bits."
    """
    table: dict[Offset, dict[int, int]] = {}
    for off in (Offset.A, Offset.B, Offset.C, Offset.C_PRIME, Offset.D):
        ow = OFFSET_WORDS[off]
        m: dict[int, int] = {}
        for error_bits in (0b1, 0b11):  # 1-bit or 2-bit burst
            for shift in range(BLOCK_LENGTH):
                error_vector = (error_bits << shift) & BLOCK_BITMASK
                syndrome = calculate_syndrome(error_vector ^ ow)
                # If two shifts produce the same syndrome, the first one wins
                # (1-bit corrections are tried before 2-bit because we iterate
                # 0b1 first).  Matches redsea's iteration order.
                if syndrome not in m:
                    m[syndrome] = error_vector
        table[off] = m
    return table


_ERROR_TABLE: dict[Offset, dict[int, int]] = _make_error_lookup_table()


def correct_burst_errors(raw: int, expected_offset: Offset) -> tuple[int, bool]:
    """
    Attempt 1- or 2-bit burst error correction on a 26-bit block.

    Returns
    -------
    (corrected_bits, succeeded)
        ``corrected_bits`` is the (possibly-corrected) raw 26-bit block.
        ``succeeded`` is True iff a syndrome match was found in the table
        for ``expected_offset``.

    See EN 50067:1998 Annex B.2.2 and Kopitz & Marks (1999).
    """
    syndrome = calculate_syndrome(raw)
    table = _ERROR_TABLE.get(expected_offset, {})
    err = table.get(syndrome)
    if err is None:
        return raw, False
    return raw ^ err, True


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One decoded RDS block (26 bits = 16-bit info + 10-bit checkword)."""
    offset: Offset
    raw: int                   # full 26-bit value as transmitted (possibly with errors)
    info: int                  # 16-bit info word (top 16 bits of raw, post-FEC)
    had_errors: bool
    is_received: bool          # True if this block decoded (with or without FEC)
    sample_index: float        # MPX-sample index of the FIRST bit of this block
    bit_position: int          # absolute bit index in the input stream of bit 0


# We keep the four blocks of a group in a list indexed 0..3.
# Index 0 is block A, 1 is B, 2 is C or C', 3 is D.

@dataclass
class Group:
    """A complete RDS group (4 blocks)."""
    blocks: list[Optional[Block]] = field(default_factory=lambda: [None] * 4)
    sample_index_first_bit: float = float("nan")
    bit_position_first_bit: int = -1

    @property
    def pi(self) -> Optional[int]:
        """Program Identification (block A's info word) if available."""
        a = self.blocks[0]
        return a.info if a is not None and a.is_received else None

    @property
    def group_type(self) -> Optional[str]:
        """Group-type string like ``"0A"`` or ``"2B"`` (from block B)."""
        b = self.blocks[1]
        if b is None or not b.is_received:
            return None
        # Block B layout: bits 15..12 = group number, bit 11 = version (0=A, 1=B)
        info = b.info
        num = (info >> 12) & 0xF
        ver = "B" if ((info >> 11) & 1) else "A"
        return f"{num}{ver}"

    @property
    def num_errors(self) -> int:
        """Number of blocks that did not decode (out of 4)."""
        return sum(1 for b in self.blocks if b is None or not b.is_received)

    @property
    def bler(self) -> float:
        """Block error rate, 0..1.  Mirrors redsea's ``getNumErrors()/4``."""
        return self.num_errors / 4.0

    def is_empty(self) -> bool:
        return all(b is None or not b.is_received for b in self.blocks)


# ---------------------------------------------------------------------------
# Sync-pulse buffer (3-pulse acquisition)
# ---------------------------------------------------------------------------

@dataclass
class _SyncPulse:
    offset: Offset = Offset.INVALID
    bit_position: int = 0

    def could_follow(self, other: "_SyncPulse") -> bool:
        """Could this pulse realistically follow ``other`` in a cyclic A/B/C/D sequence?"""
        if self.offset == Offset.INVALID or other.offset == Offset.INVALID:
            return False
        dist = self.bit_position - other.bit_position
        if dist <= 0 or dist % BLOCK_LENGTH != 0:
            return False
        block_dist = dist // BLOCK_LENGTH
        if block_dist > 6:
            return False
        expected_block = (get_block_number_for_offset(other.offset) + block_dist) % 4
        return expected_block == get_block_number_for_offset(self.offset)


_SYNC_PULSE_BUFFER_SIZE = 6


class _SyncPulseBuffer:
    """Rolling buffer of recent valid sync pulses; detects a cyclic sequence."""

    def __init__(self) -> None:
        self._pulses: list[_SyncPulse] = [
            _SyncPulse() for _ in range(_SYNC_PULSE_BUFFER_SIZE)
        ]

    def push(self, offset: Offset, bit_position: int) -> None:
        self._pulses = self._pulses[1:] + [_SyncPulse(offset, bit_position)]

    def is_sequence_found(self) -> bool:
        third = self._pulses[-1]
        for i_first in range(len(self._pulses) - 2):
            for i_second in range(i_first + 1, len(self._pulses) - 1):
                if (third.could_follow(self._pulses[i_second])
                        and self._pulses[i_second].could_follow(self._pulses[i_first])):
                    return True
        return False

    def clear(self) -> None:
        self._pulses = [_SyncPulse() for _ in range(_SYNC_PULSE_BUFFER_SIZE)]


# ---------------------------------------------------------------------------
# Block sync (consumes DecodedBit, emits Group)
# ---------------------------------------------------------------------------

# When out of sync, we check the offset every 1 bit; in sync, every 26 bits.
# Sync is dropped if too many blocks fail (matches redsea's 50-block window).
_MAX_TOLERABLE_BLER: int = 85  # percent, matches redsea kMaxTolerableBLER
_MAX_ERRORS_TOLERATED_OVER_50_BLOCKS: int = _MAX_TOLERABLE_BLER // 2


class BlockSync:
    """
    Streaming RDS block synchronizer.  Push bits one at a time, receive
    Group records when 4 consecutive blocks complete.

    Mirrors redsea's ``BlockStream`` class (src/block_sync.cc).
    """

    def __init__(self, *, use_fec: bool = True) -> None:
        self._use_fec = use_fec
        self._input_register: int = 0  # most recent 26 bits, latest in LSB
        self._bit_positions: list[int] = []  # absolute bit position for each of the 26 bits
        self._sample_indices: list[float] = []  # MPX sample index for each
        self._num_bits_until_next_block: int = 1
        self._bit_count: int = 0  # absolute bit counter (input stream)
        self._is_in_sync: bool = False
        self._expected_offset: Offset = Offset.A
        self._sync_buffer = _SyncPulseBuffer()
        self._current_group: Group = Group()
        self._block_error_history: list[bool] = []  # last 50 had_errors flags
        self._num_bits_since_sync_lost: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, decoded: DecodedBit) -> Iterator[Group]:
        """Feed one decoded bit; yield Group records as they complete."""
        # Shift bit into the 26-bit input register
        self._input_register = ((self._input_register << 1) | (decoded.bit & 1)) & BLOCK_BITMASK
        # Maintain parallel arrays of bit positions and sample indices,
        # so that when we extract a block we know where it started.
        self._bit_positions.append(self._bit_count)
        self._sample_indices.append(decoded.sample_index)
        if len(self._bit_positions) > BLOCK_LENGTH:
            self._bit_positions = self._bit_positions[-BLOCK_LENGTH:]
            self._sample_indices = self._sample_indices[-BLOCK_LENGTH:]
        self._bit_count += 1
        self._num_bits_until_next_block -= 1

        if self._num_bits_until_next_block == 0:
            yield from self._find_block_in_input_register()
            self._num_bits_until_next_block = BLOCK_LENGTH if self._is_in_sync else 1

    def push_bits(self, bits: Iterable[DecodedBit]) -> Iterator[Group]:
        """Convenience: feed many bits, yield all completed Groups."""
        for b in bits:
            yield from self.push(b)

    def flush(self) -> Iterator[Group]:
        """Emit any in-progress group at end-of-stream (may be partial)."""
        if not self._current_group.is_empty():
            yield self._current_group
        self._current_group = Group()

    @property
    def is_in_sync(self) -> bool:
        return self._is_in_sync

    @property
    def num_bits_since_sync_lost(self) -> int:
        return self._num_bits_since_sync_lost

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_block_in_input_register(self) -> Iterator[Group]:
        raw = self._input_register & BLOCK_BITMASK
        syndrome = calculate_syndrome(raw)
        offset = get_offset_for_syndrome(syndrome)

        # First-bit sample index of this 26-bit block (oldest bit in the register
        # corresponds to the first bit of the block).
        if len(self._sample_indices) >= BLOCK_LENGTH:
            block_sample_index = self._sample_indices[0]
            block_bit_position = self._bit_positions[0]
        else:
            block_sample_index = float("nan")
            block_bit_position = self._bit_count - len(self._sample_indices)

        # Acquire sync if not yet locked.  When the 3-pulse sequence is
        # found we fall through and process the current block as
        # `expected_offset = offset`, matching redsea's BlockStream
        # behavior (the acquiring block IS the first block of the new
        # sync run).
        if not self._is_in_sync:
            self._num_bits_since_sync_lost += 1
            if offset == Offset.INVALID:
                return  # nothing to do
            self._sync_buffer.push(offset, block_bit_position)
            if not self._sync_buffer.is_sequence_found():
                return  # still searching
            # Sync acquired
            self._is_in_sync = True
            self._expected_offset = offset
            self._current_group = Group()
            self._num_bits_since_sync_lost = 0
            self._block_error_history.clear()
            # Fall through to process this block as `offset`.

        # In sync: build the block, apply FEC if needed, drop sync if too many errors

        # Allow the C/C' substitution
        if self._expected_offset == Offset.C and offset == Offset.C_PRIME:
            self._expected_offset = Offset.C_PRIME

        had_errors = (offset != self._expected_offset)
        self._block_error_history.append(had_errors)
        if len(self._block_error_history) > 50:
            self._block_error_history = self._block_error_history[-50:]
        if sum(self._block_error_history) > _MAX_ERRORS_TOLERATED_OVER_50_BLOCKS:
            # Too many block errors → lose sync
            self._is_in_sync = False
            self._block_error_history.clear()
            self._sync_buffer.clear()
            return

        info = (raw >> CHECKWORD_LENGTH) & 0xFFFF
        is_received = (offset == self._expected_offset)

        # Burst-error FEC
        if had_errors and self._use_fec:
            corrected, ok = correct_burst_errors(raw, self._expected_offset)
            if ok:
                info = (corrected >> CHECKWORD_LENGTH) & 0xFFFF
                offset = self._expected_offset
                is_received = True

        block = Block(
            offset=offset,
            raw=raw,
            info=info,
            had_errors=had_errors,
            is_received=is_received,
            sample_index=block_sample_index,
            bit_position=block_bit_position,
        )

        if block.is_received:
            block_num = get_block_number_for_offset(self._expected_offset)
            self._current_group.blocks[block_num] = block
            if block_num == 0:
                # Start of a new group: anchor it
                self._current_group.sample_index_first_bit = block_sample_index
                self._current_group.bit_position_first_bit = block_bit_position

        next_offset = get_next_offset_for(self._expected_offset)
        if next_offset == Offset.A:
            # We just finished a group (the D block)
            yield self._current_group
            self._current_group = Group()

        self._expected_offset = next_offset
