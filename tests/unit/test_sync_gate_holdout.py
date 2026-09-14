"""The gate.toml mirror edits ONE line, and the guarantee that says so can fail.

WHAT THIS MODULE IS FOR (code review WR-05)
--------------------------------------------
``scripts/sync_gate_holdout.py`` regenerates exactly one line of ``config/gate.toml``:
``[gate.seasons].holdout``. Everything else in that file, and the ``[baseline.*]`` block
above all, must stay BYTE-IDENTICAL -- it carries an undischarged 47-of-68-field
divergence and two DELIBERATE tripwires are red because of it. Re-serializing the file
would turn those tripwires green by accident, which is clearing a disclosure by
reformatting it.

The module documented three protections. Review WR-05 found that the headline one could
not fire and the other two were too loose:

1. **The other-bytes-unchanged refusal was dead code.** ``updated = list(lines)`` then
   ``updated[index] = replacement`` changes exactly one element, and both sides of the
   comparison masked that same index -- so ``before != after`` was a value compared with
   itself. It read as a guarantee and behaved as a comment.
2. **The key match was a PREFIX.** ``stripped.startswith(b"holdout")`` would also claim a
   future ``holdout_seasons`` key in the same table, and rewrite the wrong line while
   reporting success.
3. **The table boundary was "starts with [ and ends with ]".** A multi-line array value
   whose continuation line reads ``[2021, 2022]`` matched that, ending the scan early, so
   the mirror would report "no holdout assignment" for a key that is plainly there.

EVERY TEST HERE DRIVES A TMP COPY. ``config/gate.toml`` is never written by this module --
the real file's bytes are asserted to be untouched at the end of each write test.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conf.season_partition import default_season_partition
from scripts import sync_gate_holdout as mirror

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_GATE_TOML = REPO_ROOT / "config" / "gate.toml"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tmp_gate(tmp_path: Path, monkeypatch) -> Path:
    """A throwaway gate.toml with a STALE holdout, pointed at by the module constant."""
    target = tmp_path / "gate.toml"
    target.write_bytes(
        b"# a header comment that must not move\n"
        b"[baseline.wp]\n"
        b"pooled_clv = -0.0380  # a byte-identical block\n"
        b"\n"
        b"[gate.seasons]\n"
        b"holdout = [1999, 2000]\n"
        b"\n"
        b"[gate.secondary]\n"
        b"threshold = 0.5\n"
    )
    monkeypatch.setattr(mirror, "GATE_CONFIG_PATH", target)
    return target


class TestTheKeyMatchIsAnEqualityRatherThanAPrefix:
    """WR-05(b): a sibling key beginning with the same letters is not the mirror."""

    def test_a_holdout_seasons_sibling_is_not_claimed(self) -> None:
        lines = [
            b"[gate.seasons]",
            b"holdout_seasons = [1111, 2222]",
            b"holdout = [2024, 2025]",
        ]

        assert mirror._target_line_index(lines) == 2, (
            "the mirror claimed the holdout_seasons line. A prefix match rewrites the "
            "wrong key and reports success."
        )

    def test_a_lone_holdout_seasons_key_is_no_target_at_all(self) -> None:
        lines = [b"[gate.seasons]", b"holdout_seasons = [1111, 2222]"]

        with pytest.raises(ValueError, match="has no 'holdout' assignment"):
            mirror._target_line_index(lines)


class TestTheTableBoundaryIsAHeaderRatherThanAnyBracketedLine:
    """WR-05(c): an array continuation line is not a table header."""

    def test_a_multi_line_array_value_does_not_end_the_scan(self) -> None:
        lines = [
            b"[gate.seasons]",
            b"frozen = [",
            b"  [2021, 2022],",
            b"]",
            b"holdout = [2024, 2025]",
        ]

        assert mirror._target_line_index(lines) == 4, (
            "the scan stopped at an array continuation line, so a key that is plainly "
            "inside [gate.seasons] was reported missing."
        )

    def test_a_real_next_table_still_ends_the_scan(self) -> None:
        """The control: the boundary must still bite, or the scan is unbounded."""
        lines = [
            b"[gate.seasons]",
            b"other = 1",
            b"[gate.secondary]",
            b"holdout = [2024, 2025]",
        ]

        with pytest.raises(ValueError, match="has no 'holdout' assignment"):
            mirror._target_line_index(lines)


class TestTheOtherBytesGuaranteeCanActuallyFail:
    """WR-05(a): the headline refusal was a value compared with itself."""

    def test_a_write_that_moves_another_byte_raises_AND_restores(
        self, tmp_gate: Path, monkeypatch
    ) -> None:
        """The test that was impossible to write before the fix.

        A write is forced to corrupt a line the mirror does not own. The old check
        could not see it: it compared a list with a copy of itself and never looked at
        the file. The read-back does look, so it fails -- and the original bytes come
        back, because a half-correct gate.toml is worse than a stale one.
        """
        original = tmp_gate.read_bytes()
        real_write = Path.write_bytes
        corrupted: list[int] = []

        def _corrupting_write(self: Path, data: bytes) -> int:
            # Corrupt only the FIRST write (the mirror's own); the restore must land.
            if self == tmp_gate and not corrupted:
                corrupted.append(1)
                data = data.replace(b"a byte-identical block", b"MOVED")
            return real_write(self, data)

        monkeypatch.setattr(Path, "write_bytes", _corrupting_write)

        with pytest.raises(ValueError) as excinfo:
            mirror.sync_gate_holdout()

        message = str(excinfo.value)
        assert "RESTORED" in message, message
        assert "read back" in message, message
        assert tmp_gate.read_bytes() == original, (
            "the original bytes were NOT restored, so a failed mirror left a "
            "half-correct config/gate.toml on disk"
        )

    def test_the_happy_path_moves_exactly_one_line(self, tmp_gate: Path) -> None:
        """The control: the refusal is not unconditional."""
        before = tmp_gate.read_bytes().split(b"\n")

        report = mirror.sync_gate_holdout()

        after = tmp_gate.read_bytes().split(b"\n")
        assert len(before) == len(after)
        moved = [
            i for i, (x, y) in enumerate(zip(before, after, strict=True)) if x != y
        ]
        assert len(moved) == 1, f"lines that moved: {moved}"
        assert after[moved[0]].strip().startswith(b"holdout = ")
        assert "WROTE one line." in report
        assert "read back from disk" in report

    def test_the_written_value_is_the_committed_rule(self, tmp_gate: Path) -> None:
        mirror.sync_gate_holdout()

        rendered = ", ".join(str(s) for s in default_season_partition().holdout)
        assert f"holdout = [{rendered}]".encode() in tmp_gate.read_bytes()


class TestTheRealGateTomlIsNeverWrittenByThisModule:
    """The boundary this whole file runs inside."""

    def test_the_committed_gate_toml_is_byte_unchanged(self, tmp_gate: Path) -> None:
        digest_before = _digest(REAL_GATE_TOML)

        mirror.sync_gate_holdout()

        assert _digest(REAL_GATE_TOML) == digest_before
