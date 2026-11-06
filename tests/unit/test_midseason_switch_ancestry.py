"""The 2026 mid-season switch record is witnessed, was corrected once by name, and is published.

The switch rests on ONE record in two files, landed together in commits that touch nothing else:

  * ``MIDSEASON-SWITCH-RECORD.md``        -- the human-readable half (a never-move root record);
  * ``backtest/midseason_switch_2026.py`` -- the constants the code reads.

The record was committed alone as ``a27898a`` (switch at week 13), then corrected ONCE by owner
ruling on 2026-11-06 (record section 8) in ``baa2d33`` (switch at week 10), a later commit touching
only the two record files whose message INVALIDATES ``a27898a`` by its full sha. The witness in
``tests/midseason_switch_state.py`` pins the CORRECTION, the last commit touching the record, since
that is the content in force. Against it this module asserts:

(i)   ``git log -1 --format=%H -- <path>`` resolves EXACTLY to the witnessed commit, for both paths;
(ii)  that commit is a STRICT ancestor of HEAD (``--is-ancestor`` succeeds on equality, so
      ``!= HEAD`` is checked too);
(iii) the commit touches EXACTLY the two record paths;
(iv)  each record's newline-normalized sha256 equals its witnessed digest;
(v)   the commit's committer date equals the witness and precedes the record's deadline
      (Tue 2026-11-10 17:00 ET, after week 9's last lock and before week 10's first);
(vi)  the commit is reachable from this clone's LOCAL tracking ref of GitHub's ``master``
      (``refs/remotes/origin/master``), read at test time -- as fresh as the last fetch; no
      network read is made.

And beside those: the witness landed in a commit after the record; every commit after the ORIGINAL
record commit that edits a record file says it INVALIDATES that commit by name (the correction
does, so it passes, and any quiet edit fails); and ``models/v2_recipe.py``, the recipe the record
installs, still hashes to its witness and was last committed where the record says.

WHAT THIS CANNOT PROVE. The committer date is set by the committing machine. Arm (vi) reads the
local tracking ref at test time, so it proves the record is published NOW, not WHEN it was
published. No test can prove that nobody chose a value after seeing a number; the ordering is what
makes that claim checkable at all.

Hashes are over NEWLINE-NORMALIZED bytes: ``core.autocrlf`` is true with no ``.gitattributes``.

No test here writes anything. The quiet-edit helper is imported from the Phase 33 test, and its
fail-closed synthetic-repository controls already run in
``tests/unit/test_phase38_preregistration_ancestry.py``, so they are not repeated here.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backtest import midseason_switch_2026 as record_module
from tests import midseason_switch_state
from tests.unit.test_phase33_preregistration_ancestry import (
    uninvalidating_commits_touching_the_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

RECORD_MARKDOWN_PATH = "MIDSEASON-SWITCH-RECORD.md"
RECORD_MODULE_PATH = "backtest/midseason_switch_2026.py"
RECORD_PATHS: tuple[str, ...] = (RECORD_MARKDOWN_PATH, RECORD_MODULE_PATH)
V2_RECIPE_PATH = "models/v2_recipe.py"
WITNESS_MODULE_PATH = "tests/midseason_switch_state.py"
REMOTE_MASTER_REF = "refs/remotes/origin/master"

# The ORIGINAL record commit (switch at week 13), which the witnessed correction INVALIDATES in
# part. Record section 8 names it by its full sha; a literal here, never in the state module, whose
# slots the record lists by name.
ORIGINAL_RECORD_COMMIT = "a27898aa1c1369ade3d8b47be8ab983cf16572b6"

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Copied in shape from tests/unit/test_phase38_preregistration_ancestry.py so the guards read alike.
SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (no git binary, not a git checkout, or a shallow or treeless "
    "clone), so "
    "`git merge-base --is-ancestor` would fail for want of history rather than for want of "
    "ancestry -- and the two are indistinguishable from the exit code alone. Skipping BEFORE "
    "any ancestry call rather than reporting a false ancestry violation."
)
NO_REMOTE_MASTER_SKIP_MESSAGE = (
    f"{REMOTE_MASTER_REF} is absent (no `origin` remote, or it was never fetched), so whether "
    "the record is published on GitHub's master cannot be asked of this checkout. Skipping "
    "rather than reporting an unpublished record."
)


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run one git command (in the repo root unless *cwd* is given), never raising on exit."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """True when history cannot answer (module-level: monkeypatchable).

    That is: no ``git`` binary on PATH, not a git checkout, a shallow clone, or a treeless
    partial clone (``--filter=tree:*``), whose ``git show --name-only`` / ``log -- <path>`` fail
    offline for want of trees rather than for want of the record.
    """
    try:
        if _git("rev-parse", "--git-dir").returncode != 0:
            return True
        shallow = _git("rev-parse", "--is-shallow-repository")
        if shallow.returncode != 0 or shallow.stdout.strip() == "true":
            return True
        partial = _git("config", "--get-regexp", r"^remote\..*\.partialclonefilter$")
    except FileNotFoundError:
        return True
    return any(
        line.split()[-1].startswith("tree:") for line in partial.stdout.splitlines()
    )


def _skip_without_history() -> None:
    """Skip BEFORE any history-dependent git call when history cannot answer it."""
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes."""
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _witness(relative_path: str) -> dict[str, str]:
    return midseason_switch_state.SWITCH_RECORD_WITNESS[relative_path]


def _v2_witness() -> dict[str, str]:
    return midseason_switch_state.V2_RECIPE_FILE_WITNESS[V2_RECIPE_PATH]


def _record_commit() -> str:
    return _witness(RECORD_MODULE_PATH)["commit"]


def _deadline_utc() -> datetime:
    return datetime.fromisoformat(midseason_switch_state.SWITCH_RECORD_DEADLINE_UTC)


def _record_text() -> str:
    return (REPO_ROOT / RECORD_MARKDOWN_PATH).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The witness itself, and the checks that need no git history
# ---------------------------------------------------------------------------


class TestTheWitnessIsWellFormed:
    def test_it_covers_exactly_the_two_records(self) -> None:
        assert set(midseason_switch_state.SWITCH_RECORD_WITNESS) == set(RECORD_PATHS), (
            "the witness and this module disagree about which files ARE the mid-season switch "
            "record; a record without a witness is editable after the fact."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_each_entry_has_its_shapes(self, relative_path: str) -> None:
        entry = _witness(relative_path)
        assert set(entry) == {"commit", "normalized_sha256"}, entry
        assert _SHA1_RE.match(entry["commit"]), entry["commit"]
        assert _SHA256_RE.match(entry["normalized_sha256"]), entry["normalized_sha256"]

    def test_both_records_name_the_same_commit(self) -> None:
        commits = {_witness(path)["commit"] for path in RECORD_PATHS}
        assert len(commits) == 1, (
            f"the two record files name different commits {sorted(commits)}; the record is ONE "
            "decision in two files and was last changed by one commit."
        )

    def test_the_witnessed_commit_is_the_correction_not_the_original(self) -> None:
        assert _record_commit() != ORIGINAL_RECORD_COMMIT, (
            "the witness pins the ORIGINAL record commit, but the content in force is its "
            "2026-11-06 correction (record section 8)."
        )
        assert ORIGINAL_RECORD_COMMIT in _record_text(), (
            f"record section 8 no longer names the original commit {ORIGINAL_RECORD_COMMIT}."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_each_record_is_tracked(self, relative_path: str) -> None:
        assert (REPO_ROOT / relative_path).is_file(), (
            f"{relative_path} is missing. It is TRACKED, so this is a broken checkout."
        )
        # A non-git export (or no git binary) cannot answer "is it tracked"; skip, never a false
        # "broken checkout" verdict.
        _skip_without_history()
        tracked = _git("ls-files", relative_path).stdout.split()
        assert tracked == [relative_path], (
            f"{relative_path} is not tracked: an untracked record has no commit to anchor."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_iv_each_record_still_hashes_to_its_witness(
        self, relative_path: str
    ) -> None:
        expected = _witness(relative_path)["normalized_sha256"]
        actual = _normalized_sha256(relative_path)
        assert actual == expected, (
            f"{relative_path} CHANGED after its witness was taken.\n"
            f"  recorded ({WITNESS_MODULE_PATH}): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "Editing the switch record mid-season does not fix a bug -- it destroys the "
            "evidence. A correction is a NEW commit whose message says it INVALIDATES "
            f"{_record_commit()}."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_no_record_contains_its_own_hash_or_commit(
        self, relative_path: str
    ) -> None:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert _witness(relative_path)["normalized_sha256"] not in content
        assert _witness(relative_path)["commit"] not in content
        assert _witness(relative_path)["commit"][:12] not in content

    def test_the_deadline_is_utc_and_is_the_records_own(self) -> None:
        deadline = _deadline_utc()
        assert deadline.utcoffset() == timedelta(0), (
            midseason_switch_state.SWITCH_RECORD_DEADLINE_UTC
        )
        assert (
            midseason_switch_state.SWITCH_RECORD_DEADLINE_UTC
            == record_module.PREREGISTRATION_DEADLINE_UTC
        ), (
            "the witness deadline and the record's PREREGISTRATION_DEADLINE_UTC disagree: "
            f"{midseason_switch_state.SWITCH_RECORD_DEADLINE_UTC} vs "
            f"{record_module.PREREGISTRATION_DEADLINE_UTC}."
        )

    def test_the_recorded_committed_at_precedes_the_deadline(self) -> None:
        committed_at = datetime.fromisoformat(
            midseason_switch_state.SWITCH_RECORD_COMMITTED_AT
        )
        assert committed_at.tzinfo is not None, (
            "the recorded committed-at is NAIVE; comparing it to a UTC deadline would compare "
            "two different clocks."
        )
        assert committed_at < _deadline_utc()


# ---------------------------------------------------------------------------
# The ordering: alone, before the deadline, under HEAD, on GitHub's master
# ---------------------------------------------------------------------------


class TestTheRecordWasCommittedAloneBeforeTheDeadline:
    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_i_git_resolves_exactly_the_witnessed_commit(
        self, relative_path: str
    ) -> None:
        _skip_without_history()
        resolved = _git("log", "-1", "--format=%H", "--", relative_path).stdout.strip()
        assert resolved == _witness(relative_path)["commit"], (
            f"{relative_path} was re-committed after its witness was taken.\n"
            f"  resolved from git: {resolved}\n"
            f"  recorded:          {_witness(relative_path)['commit']}"
        )

    def test_ii_the_commit_is_a_strict_ancestor_of_head(self) -> None:
        _skip_without_history()
        record = _record_commit()
        head = _git("rev-parse", "HEAD").stdout.strip()
        assert record != head, (
            f"the record commit IS HEAD ({record}); nothing follows it yet, so it cannot be "
            "shown to have preceded anything."
        )
        ancestry = _git("merge-base", "--is-ancestor", record, head)
        assert ancestry.returncode == 0, (
            f"the record commit {record} is NOT an ancestor of HEAD ({head})."
        )

    def test_iii_the_commit_touches_exactly_the_two_records(self) -> None:
        _skip_without_history()
        # The sha is passed as a REVISION, never after `--` (which would read it as a pathspec
        # and return an empty listing).
        listing = _git("show", "--name-only", "--format=", _record_commit())
        assert listing.returncode == 0, listing.stderr
        touched = sorted(path for path in listing.stdout.split() if path)
        assert touched == sorted(RECORD_PATHS), (
            f"the record commit touches {touched}; a record committed alongside anything else "
            "cannot be shown to have preceded it."
        )

    def test_v_the_committer_date_is_before_the_deadline(self) -> None:
        _skip_without_history()
        stamp = _git("show", "-s", "--format=%cI", _record_commit()).stdout.strip()
        assert stamp == midseason_switch_state.SWITCH_RECORD_COMMITTED_AT, (
            f"git reports committer date {stamp!r}, the witness records "
            f"{midseason_switch_state.SWITCH_RECORD_COMMITTED_AT!r}."
        )
        assert datetime.fromisoformat(stamp) < _deadline_utc(), (
            f"the record was committed at {stamp}, NOT before its deadline "
            f"({midseason_switch_state.SWITCH_RECORD_DEADLINE_UTC})."
        )

    def test_vi_the_commit_is_reachable_from_the_local_origin_master_ref(self) -> None:
        """Reachability from THIS clone's ``refs/remotes/origin/master`` tracking ref, NOW.

        No network read is made: the ref is as fresh as the last fetch, so this proves the record
        is published now, not when.
        """
        _skip_without_history()
        if _git("rev-parse", "--verify", "--quiet", REMOTE_MASTER_REF).returncode != 0:
            pytest.skip(NO_REMOTE_MASTER_SKIP_MESSAGE)
        record = _record_commit()
        reachable = _git("merge-base", "--is-ancestor", record, REMOTE_MASTER_REF)
        assert reachable.returncode == 0, (
            f"the record commit {record} is NOT reachable from {REMOTE_MASTER_REF}: the "
            "switch record was never published (or this clone's remote ref is stale -- "
            "`git fetch origin master` and re-run)."
        )

    def test_the_witness_landed_after_the_record(self) -> None:
        _skip_without_history()
        witness_commit = _git(
            "log", "-1", "--format=%H", "--", WITNESS_MODULE_PATH
        ).stdout.strip()
        record = _record_commit()
        assert _SHA1_RE.match(witness_commit), (
            f"git resolves no commit for {WITNESS_MODULE_PATH} ({witness_commit!r}); the "
            "witness must be committed before this arm can be asked."
        )
        assert witness_commit != record, (
            "the witness and the record are the SAME commit; a file cannot record the hash "
            "of a commit it is part of."
        )
        ancestry = _git("merge-base", "--is-ancestor", record, witness_commit)
        assert ancestry.returncode == 0, (
            f"the record commit {record} is NOT an ancestor of the commit that records its "
            f"witness ({witness_commit})."
        )


# ---------------------------------------------------------------------------
# No quiet edit after the original record (the correction names what it invalidates)
# ---------------------------------------------------------------------------


def test_no_later_commit_edits_the_record_quietly() -> None:
    """Every commit after the ORIGINAL record that edits a record file INVALIDATES it by name.

    Asked from the original commit, so the 2026-11-06 correction is in range and must pass on its
    own message ("INVALIDATES a27898aa... in part"), while any quiet edit fails.
    """
    _skip_without_history()
    offenders = uninvalidating_commits_touching_the_rule(
        REPO_ROOT, ORIGINAL_RECORD_COMMIT, RECORD_PATHS
    )
    assert offenders == [], (
        "a commit AFTER the original record edits a record file without saying it INVALIDATES "
        f"{ORIGINAL_RECORD_COMMIT} by name: {offenders}. Editing the switch record after the "
        "fact is the act this arm forbids."
    )


# ---------------------------------------------------------------------------
# The v2 recipe the record installs (record section 4)
# ---------------------------------------------------------------------------


class TestTheV2RecipeIsWitnessed:
    def test_it_covers_exactly_the_recipe_file(self) -> None:
        assert set(midseason_switch_state.V2_RECIPE_FILE_WITNESS) == {V2_RECIPE_PATH}
        entry = _v2_witness()
        assert set(entry) == {"commit", "normalized_sha256"}, entry
        assert _SHA1_RE.match(entry["commit"]), entry["commit"]
        assert _SHA256_RE.match(entry["normalized_sha256"]), entry["normalized_sha256"]

    def test_it_still_hashes_to_its_witness(self) -> None:
        expected = _v2_witness()["normalized_sha256"]
        actual = _normalized_sha256(V2_RECIPE_PATH)
        assert actual == expected, (
            f"{V2_RECIPE_PATH} CHANGED after its witness was taken.\n"
            f"  recorded ({WITNESS_MODULE_PATH}): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "The switch record installs the recipe as last committed; a changed recipe is not "
            "the recipe the record names."
        )

    def test_the_record_names_the_witnessed_commit(self) -> None:
        assert _v2_witness()["commit"] in _record_text(), (
            f"record section 4 does not name the witnessed recipe commit "
            f"{_v2_witness()['commit']}."
        )

    def test_git_resolves_exactly_the_witnessed_commit(self) -> None:
        _skip_without_history()
        resolved = _git("log", "-1", "--format=%H", "--", V2_RECIPE_PATH).stdout.strip()
        assert resolved == _v2_witness()["commit"], (
            f"{V2_RECIPE_PATH} was re-committed after its witness was taken.\n"
            f"  resolved from git: {resolved}\n"
            f"  recorded:          {_v2_witness()['commit']}"
        )


# ---------------------------------------------------------------------------
# Controls: the history guard can actually fire
# ---------------------------------------------------------------------------


class TestTheGuardsAreReal:
    def test_the_shallow_checkout_guard_actually_fires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys.modules[__name__], "_git_history_is_unavailable", lambda: True
        )
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_without_history()
        assert str(excinfo.value) == SHALLOW_SKIP_MESSAGE

    def test_a_history_arm_skips_before_any_ancestry_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []

        def _recording_git(
            *args: str, cwd: Path | None = None
        ) -> subprocess.CompletedProcess:
            calls.append(args)
            return subprocess.CompletedProcess(["git", *args], 0, "", "")

        monkeypatch.setattr(
            sys.modules[__name__], "_git_history_is_unavailable", lambda: True
        )
        monkeypatch.setattr(sys.modules[__name__], "_git", _recording_git)
        with pytest.raises(pytest.skip.Exception):
            TestTheRecordWasCommittedAloneBeforeTheDeadline().test_ii_the_commit_is_a_strict_ancestor_of_head()
        assert calls == []
