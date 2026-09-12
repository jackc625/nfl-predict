"""The frozen gate baseline is a RECORD now, and a record has to still be there.

Phase 33, Plan 33-08 Task 2 (COLD-04, D33-11, T-33-37).

WHAT THE OWNER RULED, AND WHY THIS MODULE EXISTS
-------------------------------------------------
On 2026-09-12 the owner ruled ``live-rescore``: the five SECONDARY comparator scalars stop
reading ``config/gate.toml``'s ``[baseline.*]`` block and start reading a live paired
re-score of the deployed incumbent on the same gold the candidate was scored on. The block
itself stays BYTE-UNTOUCHED as a historical record.

"Stays byte-untouched" is a claim, and the whole point of this phase is that a claim
without an instrument is a sentence. Two of the five deliberate tripwires ARE the
gate-baseline disclosure -- the frozen block diverges from a fresh re-score in 47 of 68
fields, and it was deliberately NOT re-frozen. Re-freezing would turn both tripwires GREEN,
which is clearing a disclosure by making it pass. So the bytes must be pinned.

WHY A DIGEST AND NOT ``git diff``
----------------------------------
``git diff -- config/gate.toml`` reports a CLEAN tree for a change that was committed. The
verification a future reader needs is not "did somebody forget to commit an edit" but "is
this block the same block Phase 30 froze". Only a digest against a committed anchor answers
that, and only a digest survives the commit that would hide the edit.

WHY THE DIGEST IS NEWLINE-NORMALIZED
-------------------------------------
The idiom is stated in full at ``tests/unit/test_preregistration_ancestry.py:32-39``. This
repository has ``core.autocrlf=true`` and no ``.gitattributes``, so a tracked text file is
LF in the git blob and CRLF in a Windows working tree. A digest over raw working-tree bytes
would pin a value that holds only on the machine that measured it. ``Path.read_text``
applies universal-newline translation, so slicing ``.splitlines()`` and re-joining on
``\n`` reproduces the blob exactly -- checked, not assumed:
``git cat-file blob HEAD:config/gate.toml`` yields the same digest.

THE CONTROL
------------
An anchor that cannot fail is decoration. ``test_a_one_byte_edit_would_be_caught`` drives
the same digest function over a copy of the block with a single character changed and
asserts it does NOT match. Without that arm, a digest helper that returned a constant would
pass this module.

NO TEST HERE WRITES ANYTHING. The one-byte-edit control operates on an in-memory string;
``config/gate.toml`` is only ever read.

Run this module:  uv run pytest tests/unit/test_phase33_gate_toml_untouched.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_TOML = REPO_ROOT / "config" / "gate.toml"


def _normalized_digest(lines: list[str]) -> str:
    """The sha256 of *lines* joined with LF and terminated with LF.

    The single place this module hashes anything, so the anchor comparison and the
    one-byte-edit control are demonstrably using the SAME instrument.
    """
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _gate_toml_lines() -> list[str]:
    """Every line of ``config/gate.toml``, newline-normalized by ``read_text``."""
    return GATE_TOML.read_text(encoding="utf-8").splitlines()


def _baseline_block_lines() -> list[str]:
    """The ``[baseline.*]`` block, sliced by the committed bounds."""
    lo, hi = phase33_state.GATE_TOML_BASELINE_LINES
    return _gate_toml_lines()[lo - 1 : hi]


class TestTheCommittedBoundsDescribeTheRealFile:
    """The bounds are a measured fact about the file, not a number copied from a plan."""

    def test_the_recorded_upper_bound_is_the_last_line_of_the_file(self) -> None:
        """Plan 33-08's text says (168, 265); the file has 264 lines.

        A slice past the end of a list is silently harmless in Python, so the overshoot
        produced the RIGHT digest and would never have announced itself. This asserts the
        bound against the file rather than against the plan.
        """
        _, hi = phase33_state.GATE_TOML_BASELINE_LINES
        assert hi == len(_gate_toml_lines()), (
            "GATE_TOML_BASELINE_LINES's upper bound must be the last line of "
            "config/gate.toml, not a line number that runs past the end of it"
        )

    def test_the_block_starts_at_the_first_baseline_table(self) -> None:
        """The lower bound is the ``[baseline.wp.pooled]`` header, and the line before it is not."""
        lo, _ = phase33_state.GATE_TOML_BASELINE_LINES
        lines = _gate_toml_lines()
        assert lines[lo - 1] == "[baseline.wp.pooled]"
        assert not lines[lo - 2].startswith("[baseline")

    def test_every_table_header_inside_the_block_is_a_baseline_table(self) -> None:
        """Nothing that is not part of the frozen baseline is inside the digested span."""
        headers = [
            line
            for line in _baseline_block_lines()
            if line.startswith("[") and line.endswith("]")
        ]
        assert len(headers) == 15, headers
        assert all(h.startswith("[baseline.") for h in headers), headers

    def test_no_baseline_table_lives_outside_the_block(self) -> None:
        """The span is COMPLETE: a sixteenth baseline table added below it would fail here."""
        lo, hi = phase33_state.GATE_TOML_BASELINE_LINES
        outside = [
            (i, line)
            for i, line in enumerate(_gate_toml_lines(), start=1)
            if line.startswith("[baseline.") and not (lo <= i <= hi)
        ]
        assert outside == [], outside


class TestTheBlockIsByteIdenticalToItsCommittedAnchor:
    """D33-11's mechanism: the record is preserved, so its bytes are pinned."""

    def test_the_measured_digest_equals_the_committed_anchor(self) -> None:
        """The operative assertion. A re-freeze -- committed or not -- fails here."""
        measured = _normalized_digest(_baseline_block_lines())
        assert measured == phase33_state.GATE_TOML_BASELINE_SHA256, (
            "config/gate.toml's [baseline.*] block has MOVED. D33-11 forbids re-freezing "
            "it: two of the five deliberate tripwires assert the block reproduces, so "
            "regenerating it would clear a disclosure by making it pass."
        )

    def test_the_anchor_is_a_real_sha256_and_not_a_placeholder(self) -> None:
        """A 64-hex string, not an empty string and not 'TODO'."""
        anchor = phase33_state.GATE_TOML_BASELINE_SHA256
        assert len(anchor) == 64
        assert all(c in "0123456789abcdef" for c in anchor), anchor

    def test_a_one_byte_edit_would_be_caught(self) -> None:
        """The control. Without it a digest helper returning a constant would pass above."""
        tampered = list(_baseline_block_lines())
        assert tampered[1].startswith("mean = "), tampered[1]
        tampered[1] = tampered[1][:-1] + ("8" if tampered[1][-1] != "8" else "7")
        assert _normalized_digest(tampered) != phase33_state.GATE_TOML_BASELINE_SHA256

    def test_the_digest_reproduces_from_the_committed_git_blob(self) -> None:
        """Cross-platform proof: the LF blob hashes to the same value as the CRLF worktree.

        This is what makes the anchor portable rather than a fact about this machine.
        """
        proc = subprocess.run(
            ["git", "cat-file", "blob", "HEAD:config/gate.toml"],
            capture_output=True,
            cwd=REPO_ROOT,
        )
        if proc.returncode != 0:
            pytest.skip(
                "git blob for config/gate.toml is not resolvable in this checkout"
            )
        lo, hi = phase33_state.GATE_TOML_BASELINE_LINES
        blob_lines = proc.stdout.decode("utf-8").splitlines()[lo - 1 : hi]
        assert _normalized_digest(blob_lines) == phase33_state.GATE_TOML_BASELINE_SHA256


class TestTheWorkingTreeCopyIsUnmodified:
    """A committed edit is caught by the digest; an UNCOMMITTED one is caught here."""

    def test_git_reports_no_uncommitted_change_to_gate_toml(self) -> None:
        """``git diff`` on the path is empty, so the digest above is not vacuously right."""
        proc = subprocess.run(
            ["git", "diff", "--text", "--numstat", "--", "config/gate.toml"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if proc.returncode != 0:
            pytest.skip("git is not available in this checkout")
        assert proc.stdout.strip() == "", (
            "config/gate.toml has uncommitted changes; D33-11 requires it byte-untouched"
        )
