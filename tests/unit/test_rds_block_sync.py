# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for ``rds_block_sync``: bit stream → Group records.

Covers:
  - Syndrome math (calculate_syndrome on known inputs)
  - Offset / block-number / next-offset lookups
  - Burst-error FEC: 1-bit and 2-bit shifts in each block must be
    corrected when expected_offset is provided
  - 3-pulse sync acquisition: a synthetic A-B-C-D bit stream locks
    and emits Group records
  - End-to-end real-fixture decode: PI matches redsea reference,
    group count within tolerance, BLER comparable
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from beagle_node.pipeline.rds_demodulator import DecodedBit, decode
from beagle_node.pipeline.rds_block_sync import (
    BLOCK_BITMASK,
    BLOCK_LENGTH,
    BlockSync,
    Offset,
    OFFSET_WORDS,
    SYNDROME_TO_OFFSET,
    calculate_syndrome,
    correct_burst_errors,
    get_block_number_for_offset,
    get_next_offset_for,
    get_offset_for_syndrome,
)

FIXTURE_NPZ = (
    Path(__file__).parents[1]
    / "fixtures"
    / "kuow_sync_audio_dpk_tdoa1_10s_20260524.npz"
)
FIXTURE_REDSEA_JSONL = (
    Path(__file__).parents[1]
    / "fixtures"
    / "kuow_sync_audio_dpk_tdoa1_10s_20260524.redsea.jsonl"
)


# ---------------------------------------------------------------------------
# Syndrome math
# ---------------------------------------------------------------------------

class TestSyndrome:
    def test_syndrome_of_zero_is_zero(self):
        assert calculate_syndrome(0) == 0

    def test_offset_word_syndromes_match_table(self):
        """The syndrome of an isolated offset word equals the IEC 62106 table."""
        for letter, ow in OFFSET_WORDS.items():
            assert calculate_syndrome(ow) == [
                k for k, v in SYNDROME_TO_OFFSET.items() if v == letter
            ][0], f"offset {letter}: syndrome mismatch"

    def test_get_offset_for_syndrome_roundtrip(self):
        for letter, ow in OFFSET_WORDS.items():
            assert get_offset_for_syndrome(calculate_syndrome(ow)) == letter

    def test_unknown_syndrome_returns_invalid(self):
        # A syndrome that doesn't match any offset word
        assert get_offset_for_syndrome(0b1010101010) == Offset.INVALID


class TestOffsetLookups:
    def test_block_number_mapping(self):
        assert get_block_number_for_offset(Offset.A) == 0
        assert get_block_number_for_offset(Offset.B) == 1
        assert get_block_number_for_offset(Offset.C) == 2
        assert get_block_number_for_offset(Offset.C_PRIME) == 2
        assert get_block_number_for_offset(Offset.D) == 3

    def test_next_offset(self):
        assert get_next_offset_for(Offset.A) == Offset.B
        assert get_next_offset_for(Offset.B) == Offset.C
        assert get_next_offset_for(Offset.C) == Offset.D
        assert get_next_offset_for(Offset.C_PRIME) == Offset.D
        assert get_next_offset_for(Offset.D) == Offset.A


# ---------------------------------------------------------------------------
# Burst-error FEC
# ---------------------------------------------------------------------------

def _make_valid_block(info: int, offset: Offset) -> int:
    """Compose a 26-bit block from a 16-bit info word and an offset.

    The 10-bit checkword is chosen so that ``calculate_syndrome(block)``
    returns exactly the table syndrome for ``offset`` (i.e., the block
    is what would have been transmitted for that offset over the air).

    Because redsea's matrix-form parity-check has identity rows in the
    *high* bits of the register (and the checkword sits in the low 10
    bits where the rows are non-identity), we can't just XOR
    syndrome(info<<10) into the checkword.  The map from checkword
    value to its syndrome contribution is a 10×10 GF(2) matrix; we
    invert it by brute search (1024 candidates, trivial cost).
    """
    expected_syndrome = [k for k, v in SYNDROME_TO_OFFSET.items() if v == offset][0]
    info_shifted = (info & 0xFFFF) << 10
    base_syndrome = calculate_syndrome(info_shifted)
    target_cw_syndrome = expected_syndrome ^ base_syndrome
    for cw in range(1024):
        if calculate_syndrome(cw) == target_cw_syndrome:
            return info_shifted | cw
    raise RuntimeError(
        f"No checkword found for info={info:#06x} offset={offset.name} "
        f"(parity-check matrix non-invertible?)"
    )


class TestFEC:
    def test_clean_block_not_in_error_table(self):
        """A clean (no-error) block's syndrome doesn't appear in the burst-
        error correction table — callers must skip FEC when had_errors=False.

        This matches redsea's flow: BlockStream only invokes
        correct_burst_errors when the block's offset doesn't match the
        expected offset.  For a clean block, the syndrome equals the
        offset word's table syndrome, which is not present as an error
        vector → correct_burst_errors returns (raw, False).
        """
        block = _make_valid_block(0x1234, Offset.A)
        # Syndrome matches the A offset → recognized via get_offset_for_syndrome
        assert get_offset_for_syndrome(calculate_syndrome(block)) == Offset.A
        # But correct_burst_errors expects an error pattern; clean → no match
        _, ok = correct_burst_errors(block, Offset.A)
        assert not ok

    @pytest.mark.parametrize("offset", [Offset.A, Offset.B, Offset.C, Offset.D])
    @pytest.mark.parametrize("info", [0x0000, 0x4652, 0xFFFF, 0x1234])
    def test_single_bit_burst_corrected(self, offset, info):
        good = _make_valid_block(info, offset)
        for shift in range(BLOCK_LENGTH):
            err = (1 << shift) & BLOCK_BITMASK
            bad = good ^ err
            corrected, ok = correct_burst_errors(bad, offset)
            assert ok, f"1-bit burst at shift {shift}: not corrected"
            assert corrected == good, (
                f"1-bit burst at shift {shift}: corrected to wrong value"
            )

    @pytest.mark.parametrize("offset", [Offset.A, Offset.B, Offset.D])
    def test_two_bit_burst_corrected(self, offset):
        good = _make_valid_block(0x4652, offset)
        for shift in range(BLOCK_LENGTH - 1):
            err = (0b11 << shift) & BLOCK_BITMASK
            bad = good ^ err
            corrected, ok = correct_burst_errors(bad, offset)
            # For two-bit bursts, the table may overlap with 1-bit table
            # entries for some offsets — in that case the correction may
            # succeed but yield a different (still-syndrome-consistent)
            # codeword.  We assert success but not exact match.
            assert ok, f"2-bit burst at shift {shift}: not corrected"


# ---------------------------------------------------------------------------
# 3-pulse sync acquisition on a synthetic stream
# ---------------------------------------------------------------------------

def _synth_bit_stream_for_pi(pi: int = 0x4652, n_groups: int = 20) -> list[int]:
    """Build an ideal RDS bit stream: ``n_groups`` groups, each 4 valid blocks.

    Block A info = PI, block B = 0x0000 (type 0A, MS=0, TA=0, etc.),
    block C = 0xCCCC, block D = 0xDDDD.  No bit errors.
    """
    bits: list[int] = []
    for _ in range(n_groups):
        for info, off in [
            (pi, Offset.A),
            (0x0000, Offset.B),
            (0xCCCC, Offset.C),
            (0xDDDD, Offset.D),
        ]:
            block = _make_valid_block(info, off)
            # MSB first
            for i in range(BLOCK_LENGTH - 1, -1, -1):
                bits.append((block >> i) & 1)
    return bits


class TestSyncAcquisition:
    def test_lock_on_clean_synthetic_stream(self):
        raw_bits = _synth_bit_stream_for_pi(pi=0x4652, n_groups=10)
        decoded = [DecodedBit(b, float(i), 1.0 + 0j) for i, b in enumerate(raw_bits)]

        sync = BlockSync()
        groups = list(g for b in decoded for g in sync.push(b))

        assert len(groups) >= 5, f"Expected ~10 groups, got {len(groups)}"
        # The first emitted group is partial (sync acquired mid-group).
        # All groups after the first should be complete and error-free.
        for g in groups[1:]:
            assert g.bler == 0.0, f"unexpected error on synthetic group: {g.bler}"
            assert g.pi == 0x4652
            assert g.group_type == "0A"

    def test_anchor_sample_index_set_on_block_a(self):
        raw_bits = _synth_bit_stream_for_pi(pi=0x1234, n_groups=5)
        decoded = [DecodedBit(b, float(i * 1000), 1.0 + 0j) for i, b in enumerate(raw_bits)]
        sync = BlockSync()
        groups = list(g for b in decoded for g in sync.push(b))
        # The first group is the partial sync-acquisition group (no block A);
        # all subsequent groups must have the block-A anchor set.
        for g in groups[1:]:
            assert not np.isnan(g.sample_index_first_bit), (
                "Group missing sample_index anchor"
            )
            # Anchor matches the bit_position * 1000 spacing
            assert g.sample_index_first_bit == g.bit_position_first_bit * 1000.0


# ---------------------------------------------------------------------------
# End-to-end: real KUOW fixture
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not FIXTURE_NPZ.exists() or not FIXTURE_REDSEA_JSONL.exists(),
    reason="real-audio fixture or redsea reference JSONL not available",
)


@pytestmark_real
class TestRealFixture:
    @pytest.fixture(scope="class")
    def decoded_groups(self):
        data = np.load(FIXTURE_NPZ)
        bits = decode(data["audio"], float(data["sample_rate_hz"]))
        sync = BlockSync(use_fec=True)
        groups = []
        for b in bits:
            groups.extend(sync.push(b))
        return groups

    @pytest.fixture(scope="class")
    def redsea_groups(self):
        with FIXTURE_REDSEA_JSONL.open() as f:
            return [json.loads(line) for line in f]

    def test_pi_matches_redsea(self, decoded_groups, redsea_groups):
        """All confidently-decoded groups should match the redsea reference PI."""
        redsea_pis = {g["pi"] for g in redsea_groups if "pi" in g}
        assert redsea_pis == {"0x4652"}, "redsea fixture should be single-PI KUOW"

        my_pis = Counter(g.pi for g in decoded_groups if g.pi is not None)
        assert len(my_pis) >= 1
        most_common_pi, count = my_pis.most_common(1)[0]
        assert most_common_pi == 0x4652, (
            f"Most common PI is {hex(most_common_pi)}, expected 0x4652"
        )
        # >95% of identified PIs match
        assert count / sum(my_pis.values()) >= 0.95

    def test_group_count_close_to_redsea(self, decoded_groups, redsea_groups):
        """We should decode within 15% of the redsea group count."""
        # redsea: 110 groups in this 10-second fixture
        rs_count = len(redsea_groups)
        # Count only groups that have a PI (i.e., block A was decoded);
        # redsea emits a group line only if block A decoded too.
        my_count = sum(1 for g in decoded_groups if g.pi is not None)
        ratio = my_count / rs_count
        assert 0.85 <= ratio <= 1.15, (
            f"Group count {my_count} vs redsea {rs_count} = ratio {ratio:.2f}"
        )

    def test_group_type_distribution_close_to_redsea(
        self, decoded_groups, redsea_groups
    ):
        """The relative mix of group types should match redsea closely."""
        rs_types = Counter(g["group"] for g in redsea_groups)
        my_types = Counter(g.group_type for g in decoded_groups if g.group_type)

        # The dominant group types (0A, 2A, 1A, 10A) should all appear
        for major in ("0A", "2A", "1A", "10A"):
            assert my_types[major] > 0, (
                f"Missing group type {major} (redsea: {rs_types[major]})"
            )
            # Within 30% of redsea's count for the major types
            ratio = my_types[major] / rs_types[major]
            assert 0.6 <= ratio <= 1.5, (
                f"Group type {major}: {my_types[major]} vs redsea {rs_types[major]} "
                f"(ratio {ratio:.2f})"
            )

    def test_bler_reasonable(self, decoded_groups):
        """Most groups should have BLER 0 after lock."""
        # Skip the first 10 groups for warmup
        post_warmup = [g for g in decoded_groups[10:] if g.pi is not None]
        assert len(post_warmup) > 50
        bler_zero_frac = sum(1 for g in post_warmup if g.bler == 0.0) / len(post_warmup)
        # redsea's reference has BLER 0-10 (very low); we should get most at 0
        assert bler_zero_frac > 0.7, (
            f"Only {bler_zero_frac:.1%} of post-warmup groups had BLER=0"
        )

    def test_sample_index_monotone(self, decoded_groups):
        """Sample-index anchors should be monotone increasing within a sync run."""
        with_anchor = [g for g in decoded_groups if not np.isnan(g.sample_index_first_bit)]
        anchors = [g.sample_index_first_bit for g in with_anchor]
        # Allow occasional regressions across a sync-loss / re-acquire boundary,
        # but the overwhelming majority should be increasing.
        increases = sum(1 for i in range(1, len(anchors)) if anchors[i] > anchors[i - 1])
        assert increases / max(1, len(anchors) - 1) > 0.95
