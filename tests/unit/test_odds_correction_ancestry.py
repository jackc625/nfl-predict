"""The silver odds correction record is anchored to ONE commit that precedes everything later.

Plan 33.2-08 Task 4, SPEC R12: the correction's commit must be a strict git ancestor of every
fit or grading run in this phase that reads ``odds_snapshot``. That is provable only if the
record was committed exactly once, alone, and its witness was measured AFTER its last write
(33.2-08-PLAN.md ``<owned_protocol_witness>`` W1-W3). This module asserts all FOUR parts of W6:

(i)   ``git log -1 --format=%H -- config/odds_corrections.toml`` resolves EXACTLY to the
      recorded ``P332_08_ODDS_CORRECTIONS_COMMIT`` -- the assertion an earlier witness design
      (taken before the ``[[family]]`` append) could never pass;
(ii)  that commit is a STRICT ancestor of HEAD: ``merge-base --is-ancestor`` AND ``!= HEAD``,
      because ``--is-ancestor`` succeeds when the two are equal;
(iii) the witnessed commit's tree touches ONLY the record (W2);
(iv)  the record's current sha256 equals ``P332_08_ODDS_CORRECTIONS_FILE_SHA256``, so an edit
      after the witness fails HERE rather than in a later phase.

The expected values live in ``tests/phase33_state.py`` -- outside the file they witness, in a
strictly later commit -- for the REVIEW-CIRCULAR reason ``tests/unit/test_preregistration_
ancestry.py`` records: a file carrying its own hash has no fixed point.

Hashes are over NEWLINE-NORMALIZED bytes: ``core.autocrlf`` is true with no ``.gitattributes``,
so a raw working-tree hash would disagree with the blob on a correct Windows checkout.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = "config/odds_corrections.toml"

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Copied in shape from tests/unit/test_preregistration_ancestry.py so the two guards read alike.
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
    """True when this is not a git checkout, or a shallow one (module-level: monkeypatchable)."""
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes."""
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


class TestTheRecordedWitnessIsWellFormed:
    def test_the_recorded_values_have_their_shapes(self) -> None:
        assert _SHA1_RE.match(phase33_state.P332_08_ODDS_CORRECTIONS_COMMIT)
        assert _SHA256_RE.match(phase33_state.P332_08_ODDS_CORRECTIONS_FILE_SHA256)
        assert phase33_state.P332_08_ODDS_CORRECTIONS_MEASURED_AT.endswith("Z")

    def test_the_record_is_tracked(self) -> None:
        tracked = _git("ls-files", RECORD_PATH).stdout.split()
        assert tracked == [RECORD_PATH], (
            f"{RECORD_PATH} is not tracked: an untracked record has no commit to anchor."
        )

    def test_the_record_does_not_contain_its_own_hash(self) -> None:
        content = (REPO_ROOT / RECORD_PATH).read_text(encoding="utf-8")
        assert phase33_state.P332_08_ODDS_CORRECTIONS_FILE_SHA256 not in content


class TestTheFourPartsOfTheWitness:
    def test_i_git_resolves_exactly_the_recorded_commit(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        resolved = _git("log", "-1", "--format=%H", "--", RECORD_PATH).stdout.strip()
        assert resolved == phase33_state.P332_08_ODDS_CORRECTIONS_COMMIT, (
            f"{RECORD_PATH} was re-committed after its witness was taken.\n"
            f"  resolved from git: {resolved}\n"
            f"  recorded:          {phase33_state.P332_08_ODDS_CORRECTIONS_COMMIT}\n"
            "A new correction is a NEW record-final commit and a NEW appended P332_* "
            "witness -- never an edit under a stale sha."
        )

    def test_ii_the_commit_is_a_strict_ancestor_of_head(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        record = phase33_state.P332_08_ODDS_CORRECTIONS_COMMIT
        head = _git("rev-parse", "HEAD").stdout.strip()
        assert record != head, (
            f"the record commit IS HEAD ({record}); nothing follows it yet, so it cannot "
            "be shown to have preceded anything."
        )
        ancestry = _git("merge-base", "--is-ancestor", record, head)
        assert ancestry.returncode == 0, (
            f"the record commit {record} is NOT an ancestor of HEAD ({head})."
        )

    def test_iii_the_commit_touches_only_the_record(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        listing = _git(
            "show",
            "--name-only",
            "--format=",
            phase33_state.P332_08_ODDS_CORRECTIONS_COMMIT,
        )
        assert listing.returncode == 0, listing.stderr
        touched = sorted(path for path in listing.stdout.split() if path)
        assert touched == [RECORD_PATH], (
            f"the witnessed commit touches {touched}; a record committed alongside "
            "anything else cannot be shown to have preceded it."
        )

    def test_iv_the_record_still_hashes_to_its_recorded_digest(self) -> None:
        actual = _normalized_sha256(RECORD_PATH)
        assert actual == phase33_state.P332_08_ODDS_CORRECTIONS_FILE_SHA256, (
            f"{RECORD_PATH} CHANGED after its witness was taken.\n"
            f"  recorded: {phase33_state.P332_08_ODDS_CORRECTIONS_FILE_SHA256}\n"
            f"  now:      {actual}"
        )


class TestTheGuardsAreReal:
    def test_the_shallow_checkout_guard_actually_fires(self, monkeypatch) -> None:
        """The skip is reachable: with history reported unavailable, the ancestry check skips."""
        monkeypatch.setattr(
            sys.modules[__name__], "_git_history_is_unavailable", lambda: True
        )
        with pytest.raises(pytest.skip.Exception, match="history is unavailable"):
            TestTheFourPartsOfTheWitness().test_ii_the_commit_is_a_strict_ancestor_of_head()

    def test_a_one_byte_edit_would_move_the_digest(self) -> None:
        raw = (REPO_ROOT / RECORD_PATH).read_bytes().replace(b"\r\n", b"\n")
        edited = raw + b" "
        assert (
            hashlib.sha256(edited).hexdigest()
            != phase33_state.P332_08_ODDS_CORRECTIONS_FILE_SHA256
        )
