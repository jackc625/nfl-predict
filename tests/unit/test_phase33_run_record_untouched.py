"""The spent run record and its two provenance stamps are BYTE-UNTOUCHED (Plan 33-17; T-33-86/87).

WHAT IS BEING PROTECTED, AND FROM WHAT
---------------------------------------
Plan 33-17 corrects three refusal messages in ``backtest/weekly_bet_list.py`` that prescribed
re-running a measurement whose ledger records it as already spent. The obvious way to make that
correction is a repo-wide search-and-replace over the module's name. That would be a serious
mistake, and this module is the guard against it.

Two NEARBY mentions are not remedies at all:

    backtest/profitability_2025.py            the emitted generator header, three lines beginning
                                              "GENERATOR OUTPUT. Produced by"
    config/profitability_2025_verdict.toml    the same three lines, committed

They are ACCURATE PROVENANCE STAMPS: they record what produced an artifact. Rewriting them would
not fix a misleading instruction, it would destroy the record of where a published verdict came
from. The third path, ``config/profitability_2025_run_ledger.toml``, is the ledger itself -- the
evidence that the single-use 2025 split was spent, which is WHY the prescribed command was
unavailable. Editing any of the three to cover a new season is T-33-86, the tampering this phase's
overlay design exists to avoid.

ONE DIGEST REPRESENTATION, NOT TWO (Codex HIGH, folded into the plan)
----------------------------------------------------------------------
All three anchors are WHOLE-FILE digests. An earlier draft proposed hashing a REGION of
``backtest/profitability_2025.py`` -- just the stamp lines -- while the verification hashed whole
files; that is two contracts, and neither can validate the other. The chosen representation is
whole-file because the property being kept is that the spent run record is BYTE-UNTOUCHED, which is
a whole-file property. A region digest would permit the rest of the generator to change while the
stamp stayed put, which is exactly the state nobody wants to discover later. A future phase that
genuinely needs region anchors makes that a deliberate representation change with its own
migration; it does not mix the two.

WHY THE DIGESTS ARE NEWLINE-NORMALIZED
----------------------------------------
This repository has ``core.autocrlf=true`` and no ``.gitattributes``, so a tracked text file is LF
in the git blob and CRLF in a fresh Windows working tree. A digest over RAW working-tree bytes would
pin a value that holds on the machine that measured it and fails on every other checkout -- the
opposite of what a committed constant is for. Normalized, each digest equals the sha256 of
``git cat-file blob <commit>:<path>``, and that equality is ASSERTED below rather than assumed.
This is the idiom ``tests/unit/test_preregistration_ancestry.py`` established.

THE GENERATOR WAS RE-SEALED ONCE, DELIBERATELY (Plan 33.2-06, owner ruling 2026-09-21)
---------------------------------------------------------------------------------------
D33.2-24 deleted the O/U eligibility gate and its 48.0 boundary constant. The generator imported
that constant at module level, so commit ``c2257ea`` had to remove the dead gate wiring from it
(+7/-9; the stamp lines untouched). The owner ruled to keep that change and re-seal. The re-seal is
NOT an edit of ``PROVENANCE_STAMP_DIGESTS`` -- that slot is append-once and keeps its original three
values. It is a new slot, ``PLAN_33_2_06_RESEALED_PROVENANCE_DIGESTS``, which supersedes the
generator's entry and NOTHING ELSE:

* the generator is compared against its re-sealed digest, so a further one-byte change still fails;
* the verdict file and the ledger are compared against their ORIGINAL Plan 33-17 digests, and the
  re-seal slot is asserted to name the generator alone, so it cannot quietly absorb either of them;
* the superseded value is asserted to be the original pin, so the chain old -> new is checked.

Run this module:  uv run pytest tests/unit/test_phase33_run_record_untouched.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tests.phase33_state import (
    PLAN_33_2_06_RESEAL_COMMIT,
    PLAN_33_2_06_RESEAL_SUPERSEDED_DIGEST,
    PLAN_33_2_06_RESEALED_PROVENANCE_DIGESTS,
    PROVENANCE_STAMP_DIGESTS,
    PROVENANCE_STAMP_OPENING,
    SPENT_MEASUREMENT_TOKEN,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two paths carrying the emitted provenance stamp. The third anchored path is the ledger,
# which carries no stamp of its own -- it IS the evidence.
_STAMPED_PATHS: tuple[str, str] = (
    "backtest/profitability_2025.py",
    "config/profitability_2025_verdict.toml",
)

_LEDGER_PATH: str = "config/profitability_2025_run_ledger.toml"

_GENERATOR_PATH: str = "backtest/profitability_2025.py"

_VERDICT_PATH: str = "config/profitability_2025_verdict.toml"

# The digest each anchored file must have TODAY: the original Plan 33-17 pins, with the generator's
# entry superseded by the Plan 33.2-06 re-seal. Built from the two slots rather than restated, so
# neither value can drift from the manifest that records it.
EXPECTED_DIGESTS: dict[str, str] = {
    **PROVENANCE_STAMP_DIGESTS,
    **PLAN_33_2_06_RESEALED_PROVENANCE_DIGESTS,
}


def normalized_digest(data: bytes) -> str:
    """sha256 over newline-normalized bytes -- the ONE instrument this module uses."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def anchor_matches(relative_path: str, data: bytes) -> bool:
    """THE comparison: do *data* hash to the digest *relative_path* must have today?

    The real-file assertion and the planted controls both go through this one function, so a
    control that bites proves the assertion bites.
    """
    return normalized_digest(data) == EXPECTED_DIGESTS[relative_path]


# ---------------------------------------------------------------------------
# 1. Non-vacuity
# ---------------------------------------------------------------------------


def test_all_three_provenance_paths_are_anchored_and_present() -> None:
    """A two-entry anchor set would leave one file free to move unnoticed."""
    assert len(PROVENANCE_STAMP_DIGESTS) == 3
    assert sorted(PROVENANCE_STAMP_DIGESTS) == sorted([*_STAMPED_PATHS, _LEDGER_PATH])
    for relative_path in PROVENANCE_STAMP_DIGESTS:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_the_reseal_supersedes_the_generator_and_nothing_else() -> None:
    """The re-seal slot may move ONE pin. The verdict and the ledger stay on their originals.

    If the re-seal slot ever named the verdict or the ledger, a tampered spent record could be
    "re-sealed" into passing; this is what refuses that.
    """
    assert set(PLAN_33_2_06_RESEALED_PROVENANCE_DIGESTS) == {_GENERATOR_PATH}
    assert sorted(EXPECTED_DIGESTS) == sorted(PROVENANCE_STAMP_DIGESTS)
    for relative_path in (_LEDGER_PATH, _VERDICT_PATH):
        assert (
            EXPECTED_DIGESTS[relative_path] == PROVENANCE_STAMP_DIGESTS[relative_path]
        )
    assert (
        PROVENANCE_STAMP_DIGESTS[_GENERATOR_PATH]
        == PLAN_33_2_06_RESEAL_SUPERSEDED_DIGEST
    )
    assert EXPECTED_DIGESTS[_GENERATOR_PATH] != PLAN_33_2_06_RESEAL_SUPERSEDED_DIGEST


def test_the_superseded_digest_is_the_generator_just_before_the_reseal_commit() -> None:
    """The chain is checked against history: the old pin is the file at c2257ea's parent.

    That is what makes c2257ea the ONLY commit that moved the generator since Plan 33-17 pinned it,
    rather than a claim in a comment.
    """
    blob = subprocess.run(
        [
            "git",
            "cat-file",
            "blob",
            f"{PLAN_33_2_06_RESEAL_COMMIT}~1:{_GENERATOR_PATH}",
        ],
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if blob.returncode != 0:
        pytest.skip(
            "git history is unavailable (shallow clone or not a git checkout), so the pre-reseal "
            "blob cannot be read."
        )
    assert (
        hashlib.sha256(blob.stdout).hexdigest() == PLAN_33_2_06_RESEAL_SUPERSEDED_DIGEST
    )


# ---------------------------------------------------------------------------
# 2. The assertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_DIGESTS))
def test_each_anchored_file_is_byte_unchanged(relative_path: str) -> None:
    """T-33-86 / T-33-87. The primary evidence; the git diff is the corroborating one."""
    assert anchor_matches(relative_path, (REPO_ROOT / relative_path).read_bytes()), (
        f"{relative_path} has MOVED. If this was a search-and-replace over the spent "
        "measurement's name, it has destroyed a provenance stamp or the one-shot ledger rather "
        "than correcting a refusal -- the three refusals live in backtest/weekly_bet_list.py and "
        "nowhere else."
    )


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_DIGESTS))
def test_each_anchor_equals_the_committed_git_blob(relative_path: str) -> None:
    """The normalization claim, asserted rather than asserted-about.

    If the anchors were over raw bytes they would hold only on the machine that measured them.
    Equality with ``git cat-file blob HEAD:<path>`` is what makes them portable, so it is checked
    instead of being described in a comment.
    """
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if blob.returncode != 0:
        pytest.skip(
            "git history is unavailable (shallow clone or not a git checkout), so the blob "
            "comparison would fail for want of history rather than for want of identity."
        )
    assert hashlib.sha256(blob.stdout).hexdigest() == EXPECTED_DIGESTS[relative_path]


def test_the_two_provenance_stamps_still_say_what_produced_the_artifact() -> None:
    """The digests prove nothing MOVED; this proves what did not move is the thing named.

    Without it the anchors would be three opaque hashes and a future reader could not tell what
    property they were protecting.
    """
    for relative_path in _STAMPED_PATHS:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert PROVENANCE_STAMP_OPENING in text, relative_path
        assert SPENT_MEASUREMENT_TOKEN in text, relative_path


def test_the_ledger_still_records_the_split_as_spent() -> None:
    """The fact the corrected refusals now rest on: there is no force flag and no re-run."""
    text = (REPO_ROOT / _LEDGER_PATH).read_text(encoding="utf-8")
    assert 'state = "completed"' in text
    assert "NO force" in text


# ---------------------------------------------------------------------------
# 3. The fail-closed control
# ---------------------------------------------------------------------------


def test_a_one_byte_edit_moves_the_digest(tmp_path: Path) -> None:
    """A check only ever observed passing cannot be told from one that cannot fail.

    Run on a COPY in ``tmp_path``. Mutating a real anchored file to prove the guard works would
    be the tampering the guard exists to detect.
    """
    original = (REPO_ROOT / _LEDGER_PATH).read_bytes()
    before = normalized_digest(original)
    assert before == EXPECTED_DIGESTS[_LEDGER_PATH]

    copy = tmp_path / "ledger.toml"
    copy.write_bytes(original.replace(b'state = "completed"', b'state = "armed"', 1))
    after = normalized_digest(copy.read_bytes())

    assert after != before, (
        "a one-byte edit did not move the digest; the anchor would not detect a re-armed ledger"
    )


def test_a_one_byte_edit_to_the_resealed_generator_is_still_caught() -> None:
    """PLANTED CONTROL for the re-seal: the generator's new pin bites exactly as the old one did.

    The real bytes pass; the same bytes with one character added are fed through the SAME
    ``anchor_matches`` the real assertion uses and must fail. In memory -- the real generator is
    never mutated.
    """
    original = (REPO_ROOT / _GENERATOR_PATH).read_bytes()
    assert anchor_matches(_GENERATOR_PATH, original)

    marker = PROVENANCE_STAMP_OPENING.encode("ascii")
    assert marker in original
    tampered = original.replace(marker, marker.replace(b"Produced", b"Produced "), 1)
    assert tampered != original
    assert not anchor_matches(_GENERATOR_PATH, tampered), (
        "a one-byte edit to the generator passed the re-sealed pin; the re-seal has disarmed it"
    )


def test_the_superseded_generator_pin_no_longer_passes() -> None:
    """PLANTED CONTROL: the generator is compared against the RE-SEAL, not the old value.

    A digest mismatch fed through the same comparison: the real generator's bytes, judged against
    the superseded pin, must NOT match -- so the guard cannot be silently reading the old slot.
    """
    real = (REPO_ROOT / _GENERATOR_PATH).read_bytes()
    assert anchor_matches(_GENERATOR_PATH, real)
    assert normalized_digest(real) != PLAN_33_2_06_RESEAL_SUPERSEDED_DIGEST


@pytest.mark.parametrize("relative_path", [_LEDGER_PATH, _VERDICT_PATH])
def test_a_one_byte_edit_to_the_verdict_or_ledger_is_still_caught(
    relative_path: str,
) -> None:
    """PLANTED CONTROL: the re-seal did not loosen the two files it was not allowed to touch."""
    original = (REPO_ROOT / relative_path).read_bytes()
    assert anchor_matches(relative_path, original)
    assert not anchor_matches(relative_path, original + b"#")


def test_the_normalization_makes_crlf_and_lf_agree(tmp_path: Path) -> None:
    """The other half of the control: the instrument must be blind to line endings ONLY.

    Without this a digest that simply ignored the file would also pass the test above.
    """
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")

    assert normalized_digest(lf.read_bytes()) == normalized_digest(crlf.read_bytes())
    assert normalized_digest(b"alpha\nbeta\n") != normalized_digest(b"alpha\nbetaa\n")
