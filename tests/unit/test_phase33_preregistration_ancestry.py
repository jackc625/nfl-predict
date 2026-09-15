"""The Phase-33 cold-start pre-registration is anchored to a commit and locked to a hash.

Phase 33, Plan 33-16 Task 4 (COLD-07, CLEAN-01, D33-19, T-33-78/79/80/81/83).

WHAT THIS PROVES, AND WHAT IT DELIBERATELY CANNOT
--------------------------------------------------
Everything Phase 33 publishes about the 2026 cold start rests on one mechanical fact: the rule
commit strictly precedes the numbers it governs, and the rule's content did not change in
between. This module asserts the checkable parts -- the commit resolves, it equals the recorded
anchor, it carried ONLY the two rule files, each file still hashes to its recorded digest, and
the witness landed in a LATER commit than the rule it witnesses.

It cannot prove that nobody tuned a threshold after seeing the numbers. NO TEST CAN PROVE
INTENT. That clause is judgment-tier and was discharged by the owner's Task-3 ratification on
2026-09-14, recorded with its date in the plan's summary.

WHY A SECOND ANCHOR IS NEEDED AT ALL
--------------------------------------
``data/``, ``outputs/`` and ``artifacts/`` are ALL gitignored. So git ancestry alone can only
anchor the rule to a readout we write ourselves; it cannot reach wall-clock time, and therefore
cannot show the rule predates KICKOFF -- which is exactly the property that makes this a
pre-registration rather than a description written afterwards.

AND WHAT THE AUTHOR DATE CANNOT DO (the point that is easiest to overstate)
----------------------------------------------------------------------------
A git AUTHOR DATE IS LOCALLY SETTABLE. ``GIT_AUTHOR_DATE`` and ``git commit --date`` both set
it, so the wall-clock assertion below is CORROBORATION, NOT PROOF. It is asserted because a
recorded date that disagreed with the record would still be a finding, not because it carries
the pre-kickoff claim. The claim rests on an EXTERNAL anchor -- a signed tag pushed to the
remote, a remote push receipt, or a CI attestation -- produced by a system other than this
working tree.

THAT EXTERNAL ANCHOR IS DEFERRED, NOT MISSING (owner ruling, 2026-09-14)
--------------------------------------------------------------------------
It was NOT skipped and it is NOT unavailable. The owner ruled that the push happens ONCE at
phase end, still before the deadline, rather than mid-flight: this local branch is hundreds of
commits ahead of a stale ``origin/master`` whose real remote state the tracking ref has lost,
so a mid-phase push would publish an entire half-finished phase to settle a timestamp. Until it
lands, the pre-registration rests on git ancestry plus a corroborating author date only, and
both this module and ``COLD-START-PREREGISTRATION.md`` say exactly that rather than implying a
strength the evidence does not yet have. The arm below asserts the recorded kind is one of the
permitted values, PENDING included, and that the document discloses the weaker case while it is
pending.

WHY THE EXPECTED HASHES COME FROM A WITNESS MODULE AND NEVER FROM THE FILE BEING HASHED
-----------------------------------------------------------------------------------------
A file that must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential:
writing the hash changes the bytes the hash was computed over, so no fixed point exists without
a canonical exclusion rule nobody has defined. The expected values therefore live in
``tests/phase33_state.py``, OUTSIDE the files they witness, appended in a LATER commit under
that module's APPEND PROTOCOL -- the pattern Phases 30 and 31 already proved.

WHY THE HASHES ARE NEWLINE-NORMALIZED
---------------------------------------
This repository has ``core.autocrlf=true`` and no ``.gitattributes``, so a tracked text file is
LF in the git blob and CRLF in a fresh Windows working tree. A digest over RAW working-tree
bytes would pin a value that holds on the machine that measured it and fails on every other
checkout. Normalized, each digest equals ``sha256(git cat-file blob <commit>:<path>)`` and
reproduces on any platform -- which the committed-bytes arm below asserts directly rather than
assuming.

EVERY ``git show`` HERE PASSES THE SHA AS A REVISION, NEVER AFTER ``--``
-------------------------------------------------------------------------
``git show --name-only --format= -- <sha>`` reads the sha as a PATHSPEC, so it inspects HEAD
filtered to a nonexistent path and returns EMPTY -- which an "is this list exactly two files"
check reads as SUCCESS. The vacuous form is the one that looks right, so it is named here.

NO TEST HERE WRITES INTO THE REPOSITORY. The corrective-commit control builds a throwaway git
repository under a pytest ``tmp_path``; nothing in this module commits, stages or mutates this
checkout.

Run this module:  uv run pytest tests/unit/test_phase33_preregistration_ancestry.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from backtest.cold_start_constants import PREREGISTRATION_PATHS
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The external-anchor kinds this pre-registration may record. The first three are the anchors
# D33-19 permits; PENDING_DEFERRED_TO_PHASE_END is the OWNER'S 2026-09-14 ruling, which
# postdates the plan and is why the vocabulary has five members rather than four. The owner
# explicitly REFUSED "NONE_AVAILABLE" -- an anchor deliberately deferred to a dated point still
# inside the deadline is a different fact from one that could not be obtained, and collapsing
# the two would misreport the record in the safer-sounding direction.
PERMITTED_EXTERNAL_ANCHOR_KINDS: tuple[str, ...] = (
    "signed_pushed_tag",
    "push_receipt",
    "ci_attestation",
    "PENDING_DEFERRED_TO_PHASE_END",
    "NONE_AVAILABLE",
)


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run one git command, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """True when git history cannot support an ancestry question.

    Two cases, both of which make a history call fail for a reason that has nothing to do with
    ancestry: this is not a git checkout at all, or it is a shallow one truncated by
    ``--depth``. A module-level function precisely so it can be monkeypatched -- see
    ``test_the_shallow_checkout_guard_actually_fires``, which proves the guard fires rather
    than trusting that the probe was wired correctly.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _shallow_skip_message() -> str:
    """The pinned skip text, read from the witness so guard and control cannot drift apart."""
    return phase33_state.SHALLOW_SKIP_MESSAGE_PHASE33


def _readout_pending_skip_message() -> str:
    """The pinned skip text for the readout-ancestry arm, while no readout commit exists."""
    return phase33_state.READOUT_PENDING_SKIP_MESSAGE_PHASE33


def _preregistration_commit() -> str:
    """Resolve the LAST commit to touch EITHER pre-registration path.

    The paths come from ``backtest.cold_start_constants.PREREGISTRATION_PATHS``, never from a
    literal here: the two files are ONE rule in two artifacts, so the anchor is their COMBINED
    last-modifying commit and the rule itself is what names them.
    """
    return _git("log", "-1", "--format=%H", "--", *PREREGISTRATION_PATHS).stdout.strip()


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes."""
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _resolve_or_skip() -> str:
    """Return the pre-registration commit, skipping BEFORE any history call if unavailable."""
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())
    return _preregistration_commit()


def _readout_commit_or_none() -> str | None:
    """The Phase-33 readout commit, or None while Plan 33-18 has not appended it yet.

    Factored out so the fail-closed control can drive the absent case explicitly instead of
    relying on the constant genuinely being absent at the moment somebody runs the suite.
    """
    return getattr(phase33_state, "READOUT_COMMIT", None)


def _resolve_readout_or_skip() -> str:
    """Return the readout commit, skipping with the pinned message while it does not exist."""
    readout = _readout_commit_or_none()
    if not readout:
        pytest.skip(_readout_pending_skip_message())
    return readout


def uninvalidating_commits_touching_the_rule(
    repo_root: Path,
    rule_commit: str,
    paths: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Commits AFTER *rule_commit* that edit a rule file WITHOUT invalidating it by name.

    A pre-registration is corrected by a NEW, VISIBLY-LATER commit that explicitly says it
    INVALIDATES the earlier one, naming its sha. An in-place edit destroys the evidence; a
    quiet supersession is the same thing with better manners.

    Parameterised by *repo_root* so the fail-closed control can drive it against a synthetic
    repository rather than mutating this one.

    Returns:
        ``[(sha, subject)]`` for every offending commit, newest first. Empty is the good case.
    """
    listing = subprocess.run(
        ["git", "log", "--format=%H%x00%B%x1e", f"{rule_commit}..HEAD", "--", *paths],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    offenders: list[tuple[str, str]] = []
    for record in listing.stdout.split("\x1e"):
        if not record.strip():
            continue
        sha, _, message = record.strip().partition("\x00")
        short = message.strip().splitlines()[0] if message.strip() else ""
        lowered = message.lower()
        invalidates = "invalidat" in lowered and (
            rule_commit in message or rule_commit[:12] in message
        )
        if not invalidates:
            offenders.append((sha, short))
    return offenders


# ---------------------------------------------------------------------------
# The checks that need NO git history -- they run on every checkout, shallow or not
# ---------------------------------------------------------------------------


def test_both_preregistration_files_exist_and_are_tracked() -> None:
    """The rule is TWO files, and both are in the index rather than merely on disk."""
    assert len(PREREGISTRATION_PATHS) == 2, PREREGISTRATION_PATHS
    tracked = set(_git("ls-files", *PREREGISTRATION_PATHS).stdout.split())
    for relative_path in PREREGISTRATION_PATHS:
        assert (REPO_ROOT / relative_path).is_file(), (
            f"the pre-registration path {relative_path} is missing from this checkout. It is "
            "TRACKED, so its absence means the checkout is broken, not that the check should "
            "be skipped."
        )
        assert relative_path in tracked, (
            f"{relative_path} is on disk but NOT tracked by git. An untracked pre-registration "
            "cannot anchor anything: there is no commit to assert ancestry from."
        )


def test_the_witness_covers_exactly_the_files_the_rule_names() -> None:
    """The recorded hash keys EQUAL ``PREREGISTRATION_PATHS`` -- no more, no fewer.

    A path added to the rule without a hash appended to the witness would be a silently
    unwitnessed file, editable after the fact with nothing to catch it. Set equality is what
    makes that impossible rather than unlikely.
    """
    assert set(phase33_state.PRE_REGISTRATION_FILE_SHA256) == set(
        PREREGISTRATION_PATHS
    ), (
        "tests/phase33_state.PRE_REGISTRATION_FILE_SHA256 and "
        "backtest.cold_start_constants.PREREGISTRATION_PATHS disagree about which files ARE "
        f"the pre-registration: witnessed {sorted(phase33_state.PRE_REGISTRATION_FILE_SHA256)} "
        f"vs declared {sorted(PREREGISTRATION_PATHS)}."
    )


def test_the_recorded_anchor_constants_are_well_formed() -> None:
    """A malformed witness is a defect in the witness and must not hide behind a skip."""
    assert _SHA1_RE.match(phase33_state.PRE_REGISTRATION_COMMIT), (
        "tests/phase33_state.PRE_REGISTRATION_COMMIT is not a 40-character SHA: "
        f"{phase33_state.PRE_REGISTRATION_COMMIT!r}"
    )
    for relative_path, digest in phase33_state.PRE_REGISTRATION_FILE_SHA256.items():
        assert _SHA256_RE.match(digest), (
            f"the recorded digest for {relative_path} is not a 64-character sha256: {digest!r}"
        )


def test_each_preregistration_file_still_hashes_to_its_recorded_digest() -> None:
    """The content lock: an edit of ONE BYTE to either file fails here."""
    for relative_path, expected in phase33_state.PRE_REGISTRATION_FILE_SHA256.items():
        actual = _normalized_sha256(relative_path)
        assert actual == expected, (
            f"{relative_path} has CHANGED since the pre-registration was frozen.\n"
            f"  recorded (tests/phase33_state.py): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "Editing the rule after the anchor commit does not fix a bug -- it DESTROYS the "
            "evidence, and there is no honest repair path. A correction is a NEW, "
            "visibly-later commit that names the superseded sha."
        )


def test_the_digest_would_catch_a_one_byte_edit() -> None:
    """Fail-closed control on the lock above: flip ONE byte IN MEMORY, expect a different digest.

    A hash assertion that has only ever been observed passing is indistinguishable from one
    computed over the wrong bytes. This proves the discrimination WITHOUT touching either
    frozen file on disk -- mutating a frozen pre-registration to test the test would be the
    exact carelessness the lock exists to catch.
    """
    for relative_path, expected in phase33_state.PRE_REGISTRATION_FILE_SHA256.items():
        raw = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        mutated = bytearray(raw)
        mutated[-2] ^= 0x01
        assert hashlib.sha256(bytes(mutated)).hexdigest() != expected, (
            f"a one-byte mutation of {relative_path} produced the RECORDED digest, which means "
            "the digest is not being computed over that file's bytes at all."
        )


def test_neither_preregistration_file_contains_its_own_recorded_hash() -> None:
    """No file carries the digest that witnesses it; otherwise there is no fixed point."""
    for relative_path, expected in phase33_state.PRE_REGISTRATION_FILE_SHA256.items():
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected not in content, (
            f"{relative_path} CONTAINS its own recorded sha256 ({expected}). A file carrying "
            "its own whole-file hash is self-referential and has no fixed point: writing the "
            "hash changes the bytes it was computed over. The witness must stay in "
            "tests/phase33_state.py, outside the files it witnesses."
        )


def test_neither_preregistration_file_contains_its_own_commit_sha() -> None:
    """Same self-reference, the other way round: a commit cannot contain its own id."""
    commit = phase33_state.PRE_REGISTRATION_COMMIT
    for relative_path in PREREGISTRATION_PATHS:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert commit not in content, (
            f"{relative_path} CONTAINS its own anchor commit SHA ({commit}). A commit hash is a "
            "function of the committed bytes, so a file carrying its own has no fixed point."
        )
        assert commit[:12] not in content, (
            f"{relative_path} contains the abbreviated form of its own anchor commit "
            f"({commit[:12]}). Same self-reference, same absence of a fixed point."
        )


def test_the_document_points_at_its_witness() -> None:
    """A reader of the prose half finds the anchor without knowing the convention."""
    content = (REPO_ROOT / "COLD-START-PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "preregistration_anchor_recorded_in: tests/phase33_state.py" in content
    assert "PRE_REGISTRATION_COMMIT" in content
    assert "PRE_REGISTRATION_FILE_SHA256" in content
    assert "PRE_REGISTRATION_AUTHOR_DATE" in content


# ---------------------------------------------------------------------------
# The WALL-CLOCK anchor -- and the plain statement of what it cannot carry
# ---------------------------------------------------------------------------


def test_the_recorded_author_date_precedes_the_deadline() -> None:
    """The corroborating anchor: the rule was authored before Week 2 kicked off.

    CORROBORATION ONLY. A git author date is locally settable, so this cannot establish
    pre-kickoff existence on its own -- see this module's docstring and the arm below, which
    asserts that limitation is stated rather than merely known.
    """
    authored = datetime.fromisoformat(phase33_state.PRE_REGISTRATION_AUTHOR_DATE)
    deadline = datetime.fromisoformat(phase33_state.PRE_REGISTRATION_DEADLINE)
    assert authored.tzinfo is not None, (
        "the recorded author date is NAIVE. A wall-clock assertion against a naive instant "
        "compares two different clocks and quietly answers the wrong question."
    )
    assert deadline.tzinfo is not None, "the recorded deadline is NAIVE."
    assert authored < deadline, (
        f"the pre-registration was authored at {authored.isoformat()}, which is NOT before the "
        f"deadline {deadline.isoformat()}. A rule frozen after the numbers it governs arrived "
        "is not a pre-registration."
    )


def test_the_recorded_author_date_matches_what_git_reports() -> None:
    """The witness is CROSS-CHECKED against git, and against the AUTHOR date specifically.

    ``%aI`` not ``%cI``. A rebase or an amend moves the committer date and leaves the author
    date alone, so the two answer different questions and only one of them is the claim.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    resolved = _git(
        "show", "-s", "--format=%aI", phase33_state.PRE_REGISTRATION_COMMIT
    ).stdout.strip()
    assert resolved == phase33_state.PRE_REGISTRATION_AUTHOR_DATE, (
        "the recorded author date does NOT match git.\n"
        f"  resolved from git:                 {resolved}\n"
        f"  recorded (tests/phase33_state.py): {phase33_state.PRE_REGISTRATION_AUTHOR_DATE}\n"
    )


def test_this_module_states_that_the_author_date_is_corroboration_not_proof() -> None:
    """The limitation is IN the file, so a reader meets it beside the assertion.

    An assertion whose weakness lives only in a reviewer's memory will be read, one phase
    later, as the proof it is not.
    """
    doc = (__doc__ or "").upper()
    assert "LOCALLY SETTABLE" in doc
    assert "GIT_AUTHOR_DATE" in doc
    assert "CORROBORATION, NOT PROOF" in doc


def test_the_external_anchor_is_recorded_as_one_of_the_permitted_kinds() -> None:
    """Either an external anchor exists, or the weaker case is DISCLOSED rather than silent.

    The owner's 2026-09-14 ruling DEFERRED the anchor to phase end rather than declaring it
    unavailable, so ``PENDING_DEFERRED_TO_PHASE_END`` is a permitted value and
    ``NONE_AVAILABLE`` was explicitly refused. A pending anchor and an impossible one are
    different facts and the record must not collapse them.
    """
    kind = phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND
    assert kind in PERMITTED_EXTERNAL_ANCHOR_KINDS, (
        f"the recorded external-anchor kind {kind!r} is not one of "
        f"{PERMITTED_EXTERNAL_ANCHOR_KINDS}."
    )

    anchor = phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR
    anchor_present = bool(anchor) and kind in (
        "signed_pushed_tag",
        "push_receipt",
        "ci_attestation",
    )
    if anchor_present:
        return

    document = (
        (REPO_ROOT / "COLD-START-PREREGISTRATION.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert ("no external anchor" in document) or ("corroboration" in document), (
        "no external anchor has been obtained AND the document does not disclose that the "
        "pre-registration therefore rests on ancestry plus a corroborating author date only. "
        "Silently having neither is the failure this arm exists to catch."
    )


def test_a_pending_external_anchor_records_its_deadline_and_its_reason() -> None:
    """A deferral is only honest if it carries WHEN it must land and WHY it was deferred.

    Without both, "pending" is indistinguishable from "forgotten" -- and the difference between
    those two is the whole content of the owner's ruling.
    """
    if (
        phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND
        != "PENDING_DEFERRED_TO_PHASE_END"
    ):
        pytest.skip("no anchor is pending, so there is no deferral to check")

    record = phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_DEFERRAL
    assert record["deadline"] == phase33_state.PRE_REGISTRATION_DEADLINE
    assert record["ruled_on"] == "2026-09-14"
    assert record["reason"].strip(), (
        "a deferral with no stated reason is a skip with a label"
    )
    assert "ancestry" in record["until_it_lands"].lower()


# ---------------------------------------------------------------------------
# The checks that DO need git history -- guarded by the shallow probe
# ---------------------------------------------------------------------------


def test_the_preregistration_commit_resolves_and_matches_the_witness() -> None:
    """The anchor is RESOLVED from git and equals the recorded slot, never transcribed."""
    resolved = _resolve_or_skip()
    assert _SHA1_RE.match(resolved), (
        "git could not resolve the last commit to touch "
        f"{list(PREREGISTRATION_PATHS)} (got {resolved!r})."
    )
    assert resolved == phase33_state.PRE_REGISTRATION_COMMIT, (
        "the resolved pre-registration commit does NOT match the recorded anchor.\n"
        f"  resolved from git:                 {resolved}\n"
        f"  recorded (tests/phase33_state.py): {phase33_state.PRE_REGISTRATION_COMMIT}\n"
        "Either the rule was re-committed and the witness was not re-measured, or the witness "
        "was edited."
    )


def test_the_preregistration_commit_contains_only_the_two_rule_files() -> None:
    """The commit's claim to BE the pre-registration is checkable, not merely asserted.

    The sha is passed as a REVISION. After ``--`` git would read it as a pathspec, inspect HEAD
    filtered to a nonexistent path, return empty, and this assertion would pass vacuously.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    commit = phase33_state.PRE_REGISTRATION_COMMIT
    listing = _git("show", "--name-only", "--format=", commit)
    assert listing.returncode == 0, listing.stderr
    touched = sorted(path for path in listing.stdout.split() if path)
    assert touched == sorted(PREREGISTRATION_PATHS), (
        f"the pre-registration commit {commit} touches {touched}, expected exactly "
        f"{sorted(PREREGISTRATION_PATHS)}. Anything else in that commit means the anchor points "
        "at a commit that did more than freeze the rule."
    )
    assert touched, (
        "the commit-contents listing is EMPTY, which is what a sha passed after `--` produces. "
        "An empty listing must never read as success."
    )


def test_the_recorded_digests_equal_the_committed_bytes() -> None:
    """The digest is measured against the COMMITTED blob, not merely the working tree.

    The working-tree check can pass on a machine whose checkout differs from what was
    committed. Asking git for the blob at the anchor commit closes that gap, and is what makes
    the newline-normalization claim ("each digest equals sha256 of the blob") a check rather
    than an assertion in a docstring.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    for relative_path, expected in phase33_state.PRE_REGISTRATION_FILE_SHA256.items():
        blob = subprocess.run(
            [
                "git",
                "cat-file",
                "blob",
                f"{phase33_state.PRE_REGISTRATION_COMMIT}:{relative_path}",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert blob.returncode == 0, blob.stderr.decode("utf-8", "replace")
        assert hashlib.sha256(blob.stdout).hexdigest() == expected, (
            f"the COMMITTED bytes of {relative_path} do not hash to the recorded digest. The "
            "witness describes something other than what is in the anchor commit."
        )


def test_the_witness_landed_after_the_rule_it_witnesses() -> None:
    """The anchor is recorded in a LATER commit than the rule, and that ordering IS the design.

    Recording the anchor in the SAME commit is impossible for the hash (self-referential) and
    for the sha (a commit cannot contain its own id). The separation is what makes the witness
    a witness. Asserted so a future consolidation that "tidies" the two commits into one is
    caught rather than admired.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    witness_commit = _git(
        "log", "-1", "--format=%H", "--", "tests/phase33_state.py"
    ).stdout.strip()
    assert _SHA1_RE.match(witness_commit), witness_commit
    assert witness_commit != phase33_state.PRE_REGISTRATION_COMMIT, (
        "the witness and the rule are the SAME commit. A file cannot record the hash of a "
        "commit it is part of, so this collapses the anchor into a claim about itself."
    )
    ancestry = _git(
        "merge-base",
        "--is-ancestor",
        phase33_state.PRE_REGISTRATION_COMMIT,
        witness_commit,
    )
    assert ancestry.returncode == 0, (
        f"the pre-registration commit {phase33_state.PRE_REGISTRATION_COMMIT} is NOT a strict "
        f"ancestor of the commit that records its anchor ({witness_commit}). The witness must "
        "be measured from a committed rule, never before it."
    )


def test_no_later_commit_edits_the_rule_without_invalidating_it_by_name() -> None:
    """A correction is a NEW, visibly-later commit that SAYS what it invalidates.

    While no later commit touches either rule file this passes trivially -- which is exactly
    why it has its own fail-closed control below. An arm that has only ever been observed not
    firing is indistinguishable from one that is not wired up.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    offenders = uninvalidating_commits_touching_the_rule(
        REPO_ROOT, phase33_state.PRE_REGISTRATION_COMMIT, tuple(PREREGISTRATION_PATHS)
    )
    assert offenders == [], (
        "a commit AFTER the anchor edits a pre-registration file without explicitly "
        f"invalidating {phase33_state.PRE_REGISTRATION_COMMIT} by name: {offenders}. Editing a "
        "pre-registration in place destroys the evidence; a quiet supersession is the same "
        "thing with better manners. The remedy is a commit message that names the superseded "
        "sha and says it is invalidated."
    )


def test_the_corrective_commit_arm_actually_fires(tmp_path: Path) -> None:
    """Fail-closed control: a synthetic later commit that edits the rule IS reported.

    Built in a throwaway repository under ``tmp_path`` rather than by mutating this checkout.
    Driving the real repository to prove a guard fires would mean committing an edit to a
    frozen pre-registration, which is the act the guard exists to forbid.
    """
    repo = tmp_path / "synthetic"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", str(repo), cwd=tmp_path).returncode == 0
    _git("config", "user.email", "control@example.invalid", cwd=repo)
    _git("config", "user.name", "Fail Closed Control", cwd=repo)

    rule = repo / "COLD-START-PREREGISTRATION.md"
    rule.write_text("the rule\n", encoding="utf-8")
    _git("add", "COLD-START-PREREGISTRATION.md", cwd=repo)
    assert _git("commit", "-q", "-m", "freeze the rule", cwd=repo).returncode == 0
    rule_commit = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()

    rule.write_text("the rule, quietly edited\n", encoding="utf-8")
    _git("add", "COLD-START-PREREGISTRATION.md", cwd=repo)
    assert _git("commit", "-q", "-m", "tidy up the wording", cwd=repo).returncode == 0

    offenders = uninvalidating_commits_touching_the_rule(
        repo, rule_commit, ("COLD-START-PREREGISTRATION.md",)
    )
    assert len(offenders) == 1, (
        f"the corrective-commit arm did NOT fire on a quiet in-place edit: {offenders}. An arm "
        "that cannot detect the thing it forbids is decoration."
    )
    assert offenders[0][1] == "tidy up the wording"

    # And the honest form PASSES: a later commit that names what it invalidates is permitted.
    rule.write_text("the rule, corrected\n", encoding="utf-8")
    _git("add", "COLD-START-PREREGISTRATION.md", cwd=repo)
    _git(
        "commit",
        "-q",
        "-m",
        f"correct the rule\n\nThis commit INVALIDATES the pre-registration at {rule_commit}.",
        cwd=repo,
    )
    still = uninvalidating_commits_touching_the_rule(
        repo, rule_commit, ("COLD-START-PREREGISTRATION.md",)
    )
    assert len(still) == 1, (
        "the arm reported a commit that DOES name what it invalidates. The rule is that a "
        "correction must be visible and explicit, not that corrections are forbidden."
    )


def test_the_readout_ancestry_arm_asserts_once_the_readout_commit_exists() -> None:
    """D33-19 anchor (i): the rule strictly precedes the commit recording the readout.

    ``READOUT_COMMIT`` is appended by Plan 33-18 when the readout lands. Until then this SKIPS
    with a pinned message rather than passing, because an arm that quietly passes for want of
    an input is an arm nobody knows is not running.
    """
    if _git_history_is_unavailable():
        pytest.skip(_shallow_skip_message())

    readout = _resolve_readout_or_skip()
    assert _SHA1_RE.match(readout), readout
    assert readout != phase33_state.PRE_REGISTRATION_COMMIT, (
        "the rule and the readout are the SAME commit. That is not a pre-registration; it is "
        "only a claim of one."
    )
    ancestry = _git(
        "merge-base", "--is-ancestor", phase33_state.PRE_REGISTRATION_COMMIT, readout
    )
    assert ancestry.returncode == 0, (
        f"the pre-registration commit {phase33_state.PRE_REGISTRATION_COMMIT} is NOT a strict "
        f"ancestor of the readout commit ({readout})."
    )


def test_the_readout_pending_skip_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed control on the readout skip, with its pinned message."""
    monkeypatch.setattr(
        "tests.unit.test_phase33_preregistration_ancestry._readout_commit_or_none",
        lambda: None,
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _resolve_readout_or_skip()
    assert str(excinfo.value) == _readout_pending_skip_message()


def test_the_shallow_checkout_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed control on the probe itself: the skip is issued, with the pinned message.

    A guard that has only ever been observed NOT firing is indistinguishable from a guard that
    is not wired up. Monkeypatching the probe drives exactly the code path a shallow clone
    would take and asserts the skip happens BEFORE any history-dependent git call.
    """
    monkeypatch.setattr(
        "tests.unit.test_phase33_preregistration_ancestry._git_history_is_unavailable",
        lambda: True,
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _resolve_or_skip()
    assert str(excinfo.value) == _shallow_skip_message()
