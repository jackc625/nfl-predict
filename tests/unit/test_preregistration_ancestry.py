"""The Phase-31 pre-registration is anchored to a git commit and locked to a recorded hash.

WHAT THIS PROVES, AND WHAT IT DELIBERATELY CANNOT
-------------------------------------------------
Everything Phase 31 publishes about 2025 rests on one mechanical fact: the rule commit strictly
precedes the measurement commit, and the rule's content did not change in between. This module
asserts the parts of that fact that are checkable BEFORE the verdict artifact exists -- the commit
resolves, it equals the recorded anchor, and each pre-registration file still hashes to its
recorded digest. Plan 31-14 extends this module with the remaining half: the ancestry assertion
between that commit and the verdict artifact's commit.

It cannot prove that nobody tuned a threshold after seeing the numbers. No test can prove intent.
That clause is judgment-tier and is discharged by the owner's CHECKPOINT 1 ratification.

WHY THE EXPECTED HASHES COME FROM A WITNESS MODULE AND NEVER FROM THE FILE BEING HASHED
---------------------------------------------------------------------------------------
REVIEW-CIRCULAR. A file that must CONTAIN and exactly REPRODUCE its own whole-file hash is
self-referential: writing the hash changes the bytes the hash was computed over, so no fixed point
exists without a canonical exclusion rule nobody has defined. A test written against such a marker
line would have to be relaxed -- to hash "everything except the marker line", or "everything above
it" -- and each relaxation is a new rule invented under execution pressure to make a failing
assertion pass. That is how a guard becomes decoration.

So the expected values live in ``tests/phase31_state.py``, OUTSIDE the files they witness,
appended in a LATER commit under that module's APPEND PROTOCOL. This is the pattern Phase 30 already
proved: ``backtest/group_gate.py``'s ``preregistration_commit`` RESOLVES the rule commit from git
and never transcribes it, and ``tests/phase30_state.py`` holds ``PRE_REGISTRATION_COMMIT``,
``MEASUREMENT_COMMIT`` and ``GROUP_VERDICT_FILE_SHA256``.

Recorded here so the next author does not reintroduce the self-reference.

WHY THE HASHES ARE NEWLINE-NORMALIZED
--------------------------------------
This repository has ``core.autocrlf=true`` and no ``.gitattributes``, so a tracked text file is LF
in the git blob and CRLF in a fresh Windows working tree. A digest over RAW working-tree bytes
would pin a value that holds on the machine that measured it and fails on every other checkout --
the exact opposite of what a tracked constant is for. Normalized, each digest equals the sha256 of
``git cat-file blob <commit>:<path>`` and reproduces on any platform.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from backtest.ev_chain_constants import PREREGISTRATION_PATHS
from tests import phase31_state

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The documented message the shallow / non-git guard skips with. Pinned as a constant so the test
# that proves the guard fires matches the SAME text the guard emits, rather than a paraphrase that
# could drift away from it. Copied in shape from tests/unit/test_gated_refit_readout_md.py.
SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout), so "
    "`git merge-base --is-ancestor` would fail for want of history rather than for want of "
    "ancestry -- and the two are indistinguishable from the exit code alone. Skipping BEFORE "
    "any ancestry call rather than reporting a false ancestry violation."
)


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run one git command in the repo root, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """Return True when git history cannot support an ancestry question.

    Two cases, both of which make a git history call fail for a reason that has nothing to do with
    ancestry: this is not a git checkout at all, or it is a shallow one whose history was truncated
    by ``--depth``.

    This is a module-level function precisely so it can be monkeypatched -- see
    ``test_the_shallow_checkout_guard_actually_fires``, which proves the guard fires rather than
    trusting that the probe was wired correctly.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _preregistration_commit() -> str:
    """Resolve the LAST commit to touch EITHER pre-registration path.

    The paths come from ``backtest.ev_chain_constants.PREREGISTRATION_PATHS``, never from a
    literal here: the two files are ONE rule in two artifacts, so the anchor is their COMBINED
    last-modifying commit and the rule itself is what names them.
    """
    result = _git("log", "-1", "--format=%H", "--", *PREREGISTRATION_PATHS)
    return result.stdout.strip()


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes."""
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _resolve_or_skip() -> str:
    """Return the pre-registration commit, skipping BEFORE any git history call if unavailable.

    Factored out of its tests so the shallow-guard proof can invoke exactly this code path under a
    monkeypatched probe. The skip is issued BEFORE any history-dependent git call, which is the
    whole point: in a ``--depth=1`` clone such a call fails for want of history, and the exit code
    cannot tell that apart from a genuine violation.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)
    return _preregistration_commit()


# ---------------------------------------------------------------------------
# The checks that need no git history -- they run on every checkout, shallow or not
# ---------------------------------------------------------------------------


def test_both_preregistration_files_exist_and_are_tracked() -> None:
    """The rule is TWO files, and both are in the index rather than merely on disk."""
    assert len(PREREGISTRATION_PATHS) == 2, PREREGISTRATION_PATHS
    tracked = set(_git("ls-files", *PREREGISTRATION_PATHS).stdout.split())
    for relative_path in PREREGISTRATION_PATHS:
        assert (REPO_ROOT / relative_path).is_file(), (
            f"the pre-registration path {relative_path} is missing from this checkout. It is "
            "TRACKED, so its absence means the checkout is broken, not that the check should be "
            "skipped."
        )
        assert relative_path in tracked, (
            f"{relative_path} is on disk but NOT tracked by git. An untracked pre-registration "
            "cannot anchor anything: there is no commit to assert ancestry from."
        )


def test_the_witness_covers_exactly_the_files_the_rule_names() -> None:
    """The recorded hash keys EQUAL ``PREREGISTRATION_PATHS``.

    A path added to the rule without a hash appended to the witness would be a silently
    unwitnessed file -- editable after the fact with nothing to catch it. Set equality is what
    makes that impossible rather than unlikely.
    """
    assert set(phase31_state.PRE_REGISTRATION_FILE_SHA256) == set(
        PREREGISTRATION_PATHS
    ), (
        "tests/phase31_state.PRE_REGISTRATION_FILE_SHA256 and "
        "backtest.ev_chain_constants.PREREGISTRATION_PATHS disagree about which files ARE the "
        f"pre-registration: witnessed {sorted(phase31_state.PRE_REGISTRATION_FILE_SHA256)} vs "
        f"declared {sorted(PREREGISTRATION_PATHS)}."
    )


def test_each_preregistration_file_still_hashes_to_its_recorded_digest() -> None:
    """The content lock: an edit of ONE BYTE to either file fails here.

    This is the assertion that turns "the rule was frozen" from a claim into a check. Recomputed
    from the working tree, compared against the witness module.
    """
    for relative_path, expected in phase31_state.PRE_REGISTRATION_FILE_SHA256.items():
        assert _SHA256_RE.match(expected), (
            f"the recorded digest for {relative_path} is not a 64-character sha256: {expected!r}"
        )
        actual = _normalized_sha256(relative_path)
        assert actual == expected, (
            f"{relative_path} has CHANGED since the pre-registration was frozen.\n"
            f"  recorded (tests/phase31_state.py): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "Editing the rule BEFORE the CHECKPOINT 1 ratification is legitimate, but it must be "
            "re-committed as a NEW single pre-registration commit and the witness re-measured. "
            "Editing it AFTER the measurement commit does not fix a bug -- it destroys the "
            "evidence, and there is no honest repair path."
        )


def test_neither_preregistration_file_contains_its_own_recorded_hash() -> None:
    """REVIEW-CIRCULAR: no file carries the digest that witnesses it.

    If it did, the digest would have no fixed point and this whole module would have to be relaxed
    into meaninglessness. Asserted rather than merely explained, so a future author who adds a
    convenience marker line finds out immediately.
    """
    for relative_path, expected in phase31_state.PRE_REGISTRATION_FILE_SHA256.items():
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected not in content, (
            f"{relative_path} CONTAINS its own recorded sha256 ({expected}). A file carrying its "
            "own whole-file hash is self-referential and has no fixed point: writing the hash "
            "changes the bytes it was computed over. The witness must stay in "
            "tests/phase31_state.py, outside the files it witnesses."
        )


def test_the_document_points_at_its_witness() -> None:
    """A reader of the prose document finds the anchor without knowing the convention.

    The document may not carry its own hash, but it MUST carry a machine-readable pointer to where
    the anchor is recorded -- otherwise the honest design (a witness outside the witnessed file)
    reads to an outsider as an anchor that simply is not there.
    """
    content = (REPO_ROOT / "PROFITABILITY-PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    assert "preregistration_anchor_recorded_in: tests/phase31_state.py" in content
    assert "PRE_REGISTRATION_COMMIT" in content
    assert "PRE_REGISTRATION_FILE_SHA256" in content


# ---------------------------------------------------------------------------
# The checks that DO need git history -- guarded by the shallow probe
# ---------------------------------------------------------------------------


def test_the_preregistration_commit_resolves_and_matches_the_witness() -> None:
    """The anchor is RESOLVED from git and equals the recorded slot.

    Resolved, never transcribed into the resolving code: the literal lives only in the witness
    module, and this test is what proves the two agree. If they ever disagree, the rule was
    re-committed without re-measuring the anchor.
    """
    resolved = _resolve_or_skip()
    assert _SHA1_RE.match(resolved), (
        "git could not resolve the last commit to touch "
        f"{list(PREREGISTRATION_PATHS)} (got {resolved!r}). Without it there is no rule commit "
        "to assert ancestry from."
    )
    assert resolved == phase31_state.PRE_REGISTRATION_COMMIT, (
        "the resolved pre-registration commit does NOT match the recorded anchor.\n"
        f"  resolved from git:                 {resolved}\n"
        f"  recorded (tests/phase31_state.py): {phase31_state.PRE_REGISTRATION_COMMIT}\n"
        "Either the rule was re-committed and the witness was not re-measured, or the witness "
        "was edited. The witness is APPEND-ONCE: a legitimate re-commit before ratification "
        "requires re-measuring it in the same breath."
    )


def test_the_preregistration_commit_contains_only_the_two_rule_files() -> None:
    """The commit's claim to BE the pre-registration is checkable, not merely asserted.

    A commit that also carried a test, a script or a data file would be a commit whose message
    says "pre-registration" while its contents say something else. Phase 30 made the same
    assertion about its single-file measurement commit for the same reason.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    commit = phase31_state.PRE_REGISTRATION_COMMIT
    listing = _git("show", "--name-only", "--format=", commit)
    assert listing.returncode == 0, listing.stderr
    touched = sorted(path for path in listing.stdout.split() if path)
    assert touched == sorted(PREREGISTRATION_PATHS), (
        f"the pre-registration commit {commit} touches {touched}, expected exactly "
        f"{sorted(PREREGISTRATION_PATHS)}. Anything else in that commit means the anchor points "
        "at a commit that did more than freeze the rule."
    )


def test_the_witness_landed_after_the_rule_it_witnesses() -> None:
    """The anchor is recorded in a LATER commit than the rule, and that ordering is the design.

    Recording the anchor in the SAME commit is impossible for the hash (it would be
    self-referential) and pointless for the SHA (a commit cannot contain its own id). The
    separation is what makes the witness a witness. Asserted so a future consolidation that
    "tidies" the two commits into one is caught.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    witness_commit = _git(
        "log", "-1", "--format=%H", "--", "tests/phase31_state.py"
    ).stdout.strip()
    assert _SHA1_RE.match(witness_commit), witness_commit
    assert witness_commit != phase31_state.PRE_REGISTRATION_COMMIT, (
        "the witness and the rule are the SAME commit. A file cannot record the hash of a commit "
        "it is part of, so this collapses the anchor into a claim about itself."
    )
    ancestry = _git(
        "merge-base",
        "--is-ancestor",
        phase31_state.PRE_REGISTRATION_COMMIT,
        witness_commit,
    )
    assert ancestry.returncode == 0, (
        f"the pre-registration commit {phase31_state.PRE_REGISTRATION_COMMIT} is NOT an ancestor "
        f"of the commit that records its anchor ({witness_commit}). The witness must be measured "
        "from a committed rule, never before it."
    )


def test_the_shallow_checkout_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed control on the probe itself: the skip is issued, with the pinned message.

    A guard that has only ever been observed NOT firing is indistinguishable from a guard that is
    not wired up. Monkeypatching the probe drives exactly the code path a shallow clone would take
    and asserts the skip happens BEFORE any history-dependent git call.
    """
    monkeypatch.setattr(
        "tests.unit.test_preregistration_ancestry._git_history_is_unavailable",
        lambda: True,
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _resolve_or_skip()
    assert str(excinfo.value) == SHALLOW_SKIP_MESSAGE


# ---------------------------------------------------------------------------
# The second half, completed by Plan 31-14: the rule provably PRECEDED the numbers
#
# Everything above is checkable before any 2025 number exists. What follows needs the verdict
# artifact, and it is the half SPEC R3 actually turns on: the pre-registration commit is a STRICT
# ancestor of the measurement commit, and the two are DIFFERENT commits.
#
# THE COMMIT IS READ FROM THE WITNESS, NEVER FROM THE ARTIFACT (REVIEW-CIRCULAR). An earlier draft
# had the verdict artifact carry its own measurement-commit marker. That cannot be satisfied: a
# commit hash is a function of the committed bytes, so writing the hash into the artifact changes
# the hash it claims to be, and the assertion built on it would have to be relaxed -- to "hash
# everything except the marker line", or "everything above it" -- until it asserted nothing.
# tests/phase31_state.py is a THIRD file that records facts about both frozen artifacts and is
# committed after them, which is exactly how tests/phase30_state.py resolved the same problem.
#
# The recorded constants are CROSS-CHECKED against git rather than trusted: a witness nobody
# verifies is a comment. And the artifact is asserted NOT to contain its own SHA, so the circular
# form cannot be reintroduced later as a "belt and braces" addition.
# ---------------------------------------------------------------------------

VERDICT_PATH = phase31_state.VERDICT_PATH


def test_the_measurement_commit_is_recorded_and_well_formed() -> None:
    """The witness carries a 40-character SHA and a 64-character digest for the artifact.

    Needs no git history, so it runs on a shallow clone: a malformed constant is a defect in the
    witness itself and should never hide behind an environment skip.
    """
    assert _SHA1_RE.match(phase31_state.MEASUREMENT_COMMIT), (
        "tests/phase31_state.MEASUREMENT_COMMIT is not a 40-character SHA: "
        f"{phase31_state.MEASUREMENT_COMMIT!r}"
    )
    assert _SHA256_RE.match(phase31_state.VERDICT_FILE_SHA256), (
        "tests/phase31_state.VERDICT_FILE_SHA256 is not a 64-character sha256: "
        f"{phase31_state.VERDICT_FILE_SHA256!r}"
    )


def test_the_verdict_artifact_exists_and_is_tracked() -> None:
    """The artifact is TRACKED, so its absence is a broken checkout and not a reason to skip.

    Deliberately NOT an evidence-backed skip. The committed generator-output form travels with
    the repository precisely so the 2025 figures survive a fresh clone; a skip here would let a
    checkout that lost the binding verdict report a green suite.
    """
    assert (REPO_ROOT / VERDICT_PATH).is_file(), (
        f"the verdict artifact {VERDICT_PATH} is missing from this checkout. It is TRACKED and "
        "the 2025 split is single-use, so it cannot be regenerated: this is a broken checkout."
    )
    tracked = _git("ls-files", VERDICT_PATH).stdout.split()
    assert VERDICT_PATH in tracked, (
        f"{VERDICT_PATH} is on disk but NOT tracked by git. An untracked verdict has no "
        "measurement commit, so there is no ancestry to assert."
    )


def test_the_verdict_artifact_still_hashes_to_its_recorded_digest() -> None:
    """The content lock on the MEASUREMENT, mirroring the one on the rule.

    A one-byte edit to the artifact fails here. That matters more for this file than for most:
    the 2025 split is single-use, so a hand-edited value cannot be regenerated and is
    indistinguishable from a tampered one.
    """
    actual = _normalized_sha256(VERDICT_PATH)
    assert actual == phase31_state.VERDICT_FILE_SHA256, (
        f"{VERDICT_PATH} has CHANGED since the measurement was committed.\n"
        f"  recorded (tests/phase31_state.py): {phase31_state.VERDICT_FILE_SHA256}\n"
        f"  recomputed from the working tree:  {actual}\n"
        "The verdict is GENERATOR OUTPUT over a single-use split. There is no honest repair "
        "path: it cannot be re-measured, so an edited artifact is simply a lost one."
    )


def test_the_digest_would_catch_a_one_byte_edit() -> None:
    """Fail-closed control on the lock above: flip ONE byte IN MEMORY, expect a different digest.

    A hash assertion that has only ever been observed passing is indistinguishable from one
    computed over the wrong bytes. This proves the discrimination WITHOUT touching the artifact
    on disk -- mutating a single-use measurement to test the test would be the exact carelessness
    the lock exists to catch.
    """
    raw = (REPO_ROOT / VERDICT_PATH).read_bytes().replace(b"\r\n", b"\n")
    mutated = bytearray(raw)
    mutated[-2] ^= 0x01
    assert (
        hashlib.sha256(bytes(mutated)).hexdigest() != phase31_state.VERDICT_FILE_SHA256
    ), (
        "a one-byte mutation of the verdict artifact produced the RECORDED digest, which means "
        "the digest is not being computed over the artifact's bytes at all."
    )


def test_the_verdict_artifact_does_not_contain_its_own_commit_sha() -> None:
    """REVIEW-CIRCULAR, asserted rather than merely explained.

    The full SHA is the criterion. The 12-character abbreviation is checked too because that is
    the form a future "belt and braces" addition would most plausibly take; 12 rather than 7
    because a 7-hex prefix that happened to be all decimal digits could collide with a rendered
    float and turn this guard into a flake.
    """
    content = (REPO_ROOT / VERDICT_PATH).read_text(encoding="utf-8")
    commit = phase31_state.MEASUREMENT_COMMIT
    assert commit not in content, (
        f"{VERDICT_PATH} CONTAINS its own measurement commit SHA ({commit}). A commit hash is a "
        "function of the committed bytes, so an artifact carrying its own has no fixed point and "
        "the ancestry assertion built on it could never be satisfied. The witness belongs in "
        "tests/phase31_state.py, outside the artifact it witnesses."
    )
    assert commit[:12] not in content, (
        f"{VERDICT_PATH} contains the abbreviated form of its own measurement commit "
        f"({commit[:12]}). Same self-reference, same absence of a fixed point."
    )


def test_the_measurement_commit_matches_what_git_resolves_for_the_artifact() -> None:
    """The recorded SHA is CROSS-CHECKED against git, never trusted on its own.

    A witness nobody verifies is a comment. If these disagree, either the artifact was
    re-committed after the witness was appended, or the witness was edited.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    resolved = _git("log", "-1", "--format=%H", "--", VERDICT_PATH).stdout.strip()
    assert _SHA1_RE.match(resolved), (
        f"git could not resolve the last commit to touch {VERDICT_PATH} (got {resolved!r})."
    )
    assert resolved == phase31_state.MEASUREMENT_COMMIT, (
        "the resolved measurement commit does NOT match the recorded one.\n"
        f"  resolved from git:                 {resolved}\n"
        f"  recorded (tests/phase31_state.py): {phase31_state.MEASUREMENT_COMMIT}\n"
    )


def test_the_recorded_digest_equals_the_committed_bytes() -> None:
    """The digest is measured against the COMMITTED blob, not merely the working tree.

    The working-tree check above can pass on a machine whose checkout differs from what was
    committed. Asking git for the blob at the measurement commit closes that gap.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    blob = subprocess.run(
        [
            "git",
            "cat-file",
            "blob",
            f"{phase31_state.MEASUREMENT_COMMIT}:{VERDICT_PATH}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert blob.returncode == 0, blob.stderr.decode("utf-8", "replace")
    assert (
        hashlib.sha256(blob.stdout).hexdigest() == phase31_state.VERDICT_FILE_SHA256
    ), (
        "the committed bytes of the verdict artifact do NOT hash to the recorded digest. The "
        "witness describes something other than what is in the measurement commit."
    )


def test_the_measurement_commit_contains_only_the_verdict_and_its_ledger() -> None:
    """The commit's claim to BE the measurement is checkable, not merely asserted.

    Exactly two paths: the verdict artifact and the COMPLETED run ledger. The record of what was
    spent has to travel in the same commit as what was measured, or the two can be separated
    later and only one of them believed.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    listing = _git("show", "--name-only", "--format=", phase31_state.MEASUREMENT_COMMIT)
    assert listing.returncode == 0, listing.stderr
    touched = sorted(path for path in listing.stdout.split() if path)
    expected = sorted([VERDICT_PATH, phase31_state.RUN_LEDGER_COMMITTED_PATH])
    assert touched == expected, (
        f"the measurement commit {phase31_state.MEASUREMENT_COMMIT} touches {touched}, expected "
        f"exactly {expected}."
    )


def test_the_rule_commit_is_a_strict_ancestor_of_the_measurement_commit() -> None:
    """SPEC R3, asserted between TWO SPECIFIC COMMITS and never against the current head.

    Asserting against HEAD is a weaker claim: HEAD moves, so it would pass for any rule committed
    at any point before now, including one committed after the numbers and then built on top of.
    Both SHAs are read from the witness and cross-checked against git by the tests above.

    The DIFFERENT-COMMITS check is not pedantry. A rule and the results it produced landing in
    one commit is not a pre-registration; it is only a claim of one.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    rule = phase31_state.PRE_REGISTRATION_COMMIT
    measurement = phase31_state.MEASUREMENT_COMMIT

    assert rule != measurement, (
        f"the pre-registration commit and the measurement commit are the SAME commit ({rule}). "
        "A rule that landed together with the numbers it produced was not registered in advance."
    )
    ancestry = _git("merge-base", "--is-ancestor", rule, measurement)
    assert ancestry.returncode == 0, (
        f"the pre-registration commit {rule} is NOT a strict ancestor of the measurement commit "
        f"{measurement} (git exit {ancestry.returncode}). Everything Phase 31 publishes about "
        "2025 rests on that ordering: without it the rule cannot be shown to have preceded the "
        "answer."
    )
    # The reverse must NOT hold. If it did, the measurement would precede the rule.
    reverse = _git("merge-base", "--is-ancestor", measurement, rule)
    assert reverse.returncode != 0, (
        f"the measurement commit {measurement} is ALSO an ancestor of the rule commit {rule}. "
        "That is only possible if they are the same commit, which the check above already "
        "excluded, so the repository state is inconsistent."
    )
