"""The 2026 rolling re-fit pre-registration was committed alone, before week 8, and published.

Plan 37-05 (RFIT-01, RFIT-06, D-03, D-23). The rolling re-fit rests on ONE pre-registration in two
files, landed together in one commit that touches nothing else:

  * ``ROLLING-REFIT-PREREGISTRATION.md``        -- the human-readable half (a never-move root record);
  * ``backtest/refit_preregistration_2026.py``  -- the constants the code reads.

Against the witness in ``tests/phase37_state.py`` this module asserts:

(i)   ``git log -1 --format=%H -- <path>`` resolves EXACTLY to the witnessed commit, for both paths;
(ii)  that commit is a STRICT ancestor of HEAD (``--is-ancestor`` succeeds on equality, so
      ``!= HEAD`` is checked too);
(iii) the commit touches EXACTLY the two record paths;
(iv)  each record's newline-normalized sha256 equals its witnessed digest;
(v)   the commit's committer date is BEFORE week 8's first decision run (D-03);
(vi)  the commit is reachable from this clone's LOCAL tracking ref of GitHub's ``master``
      (``refs/remotes/origin/master``) -- as fresh as the last fetch; no network read is made.

And beside those: the production witness ``refit/witness.py`` (which the runtime fingerprint
refusal reads, because production never imports ``tests.*``) agrees with the test witness and with
git; no later commit edits a record without INVALIDATES and the record sha in its message; the
first re-fit's code commit descends from the record; the frozen shadow pin and the old deploy-gate
files (RFIT-06) are byte-unchanged.

WHAT THIS CANNOT PROVE. The committer date is set by the committing machine (T-37-11, accepted).
Reachability from the remote master plus the push time recorded in the Plan 37-04 SUMMARY
(2026-10-20 15:41:32 ET) is the proportionate evidence. No test can prove that nobody tuned a
value after seeing a number; the ordering is what makes that claim checkable at all.

Hashes are over NEWLINE-NORMALIZED bytes: ``core.autocrlf`` is true with no ``.gitattributes``.

No test here writes into the repository or a production store: the synthetic-repo controls build
throwaway repositories under ``tmp_path``, and the first-re-fit arm only READS
``ledger/refit/dates.jsonl``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from backtest import refit_preregistration_2026 as record_module
from forward_ledger.store import LEDGER_DIR
from refit.records import read_records
from tests import phase37_state
from tests.unit.test_phase33_preregistration_ancestry import (
    uninvalidating_commits_touching_the_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

RECORD_MARKDOWN_PATH = "ROLLING-REFIT-PREREGISTRATION.md"
RECORD_MODULE_PATH = "backtest/refit_preregistration_2026.py"
RECORD_PATHS: tuple[str, ...] = (RECORD_MARKDOWN_PATH, RECORD_MODULE_PATH)
SHADOW_PIN_PATH = "backtest/shadow_pin_2026.py"
WITNESS_MODULE_PATH = "tests/phase37_state.py"
REMOTE_MASTER_REF = "refs/remotes/origin/master"

# The per-date record file of ledger/refit/ (Plan 37-06 names it DATES_FILENAME).
DATES_FILENAME = "dates.jsonl"

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Copied in shape from tests/unit/test_phase34_preregistration_ancestry.py so the guards read alike.
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
FIRST_REFIT_PENDING_SKIP_MESSAGE = (
    "no install_intent is recorded in ledger/refit/dates.jsonl, so no re-fit has reached the "
    "swap and none is serving. The first re-fit's ancestry is asserted from the first "
    "install_intent's own code_commit as soon as one exists."
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
    return phase37_state.P37_PREREGISTRATION_WITNESS[relative_path]


# Every digest the witness pins, by path: the two records, the shadow pin and the gate files.
_WITNESSED_DIGESTS: dict[str, str] = {
    **{
        path: entry["normalized_sha256"]
        for path, entry in phase37_state.P37_PREREGISTRATION_WITNESS.items()
    },
    **{
        path: entry["normalized_sha256"]
        for path, entry in phase37_state.P37_SHADOW_PIN_WITNESS.items()
    },
    **phase37_state.P37_UNTOUCHED_GATE_FILES,
}


def _record_commit() -> str:
    return _witness(RECORD_MODULE_PATH)["commit"]


def _deadline_utc() -> datetime:
    return datetime.fromisoformat(phase37_state.P37_DEADLINE_UTC)


def first_refit_intent(ledger_dir: Path) -> dict[str, Any] | None:
    """The FIRST ``install_intent`` in ``<ledger_dir>/refit/dates.jsonl``, or None.

    The first re-fit is the run that wrote the first install_intent, and its code commit is that
    intent's own ``code_commit``: the daily-run hook captures the commit once per attempt, before
    the intent is written, and writes the same value on the intent and on its attempt (Plans
    37-06, 37-11, 37-13). So nothing after the intent is needed -- an ``installed`` attempt, a
    ``crashed`` attempt after the swap (closed ``ran late`` by the next run with no new attempt),
    a ``refused`` attempt, or no attempt at all because the run was killed in between.

    Read-only, in file order, through the strict reader (a malformed line raises, never reads as
    "no intent"). None when the file is absent or holds no install_intent.
    """
    for event in read_records(ledger_dir, DATES_FILENAME):
        if event.get("event") == "install_intent":
            return event
    return None


def _first_refit_intent_or_skip() -> dict[str, Any]:
    """The production ledger's first install_intent, skipping with the pinned message if none."""
    intent = first_refit_intent(REPO_ROOT / LEDGER_DIR)
    if intent is None:
        pytest.skip(FIRST_REFIT_PENDING_SKIP_MESSAGE)
    return intent


# ---------------------------------------------------------------------------
# The witness itself, and the checks that need no git history
# ---------------------------------------------------------------------------


class TestTheWitnessIsWellFormed:
    def test_it_covers_exactly_the_two_records(self) -> None:
        assert set(phase37_state.P37_PREREGISTRATION_WITNESS) == set(RECORD_PATHS), (
            "the witness and this module disagree about which files ARE the Phase-37 "
            "pre-registration; a record without a witness is editable after the fact."
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
            f"the two record files name different commits {sorted(commits)}; the "
            "pre-registration is ONE rule in two files and was committed once."
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
            f"  recorded (tests/phase37_state.py): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "Editing the pre-registration mid-season does not fix a bug -- it destroys the "
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
        assert deadline.utcoffset() == timedelta(0), phase37_state.P37_DEADLINE_UTC
        assert (
            phase37_state.P37_DEADLINE_UTC == record_module.PREREGISTRATION_DEADLINE_UTC
        ), (
            "the witness deadline and the record's PREREGISTRATION_DEADLINE_UTC disagree: "
            f"{phase37_state.P37_DEADLINE_UTC} vs {record_module.PREREGISTRATION_DEADLINE_UTC}."
        )

    def test_the_recorded_committed_at_precedes_the_deadline(self) -> None:
        committed_at = datetime.fromisoformat(
            phase37_state.P37_PREREGISTRATION_COMMITTED_AT
        )
        assert committed_at.tzinfo is not None, (
            "the recorded committed-at is NAIVE; comparing it to a UTC deadline would compare "
            "two different clocks."
        )
        assert committed_at < _deadline_utc()


# ---------------------------------------------------------------------------
# The ordering: alone, before week 8's first decision run, under HEAD, on GitHub's master
# ---------------------------------------------------------------------------


class TestTheRecordWasCommittedAloneBeforeWeek8:
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

    def test_v_the_committer_date_is_before_week_8s_first_decision_run(self) -> None:
        _skip_without_history()
        stamp = _git("show", "-s", "--format=%cI", _record_commit()).stdout.strip()
        assert stamp == phase37_state.P37_PREREGISTRATION_COMMITTED_AT, (
            f"git reports committer date {stamp!r}, the witness records "
            f"{phase37_state.P37_PREREGISTRATION_COMMITTED_AT!r}."
        )
        assert datetime.fromisoformat(stamp) < _deadline_utc(), (
            f"the record was committed at {stamp}, NOT before week 8's first decision run "
            f"({phase37_state.P37_DEADLINE_UTC}, D-03)."
        )

    def test_vi_the_commit_is_reachable_from_the_local_origin_master_ref(self) -> None:
        """Reachability from THIS clone's ``refs/remotes/origin/master`` tracking ref.

        No network read is made: the ref is as fresh as the last fetch. The push itself was read
        back from GitHub once, at Plan 37-04 (15:41:32 ET, 2026-10-20).
        """
        _skip_without_history()
        if _git("rev-parse", "--verify", "--quiet", REMOTE_MASTER_REF).returncode != 0:
            pytest.skip(NO_REMOTE_MASTER_SKIP_MESSAGE)
        record = _record_commit()
        reachable = _git("merge-base", "--is-ancestor", record, REMOTE_MASTER_REF)
        assert reachable.returncode == 0, (
            f"the record commit {record} is NOT reachable from {REMOTE_MASTER_REF}: the "
            "pre-registration was never published (or this clone's remote ref is stale -- "
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
# The production witness the runtime fingerprint refusal reads (research Pitfall 9)
# ---------------------------------------------------------------------------


def test_the_production_witness_agrees() -> None:
    from refit import witness

    expected_digests = {
        path: _witness(path)["normalized_sha256"] for path in RECORD_PATHS
    }
    assert tuple(witness.PREREGISTRATION_PATHS) == RECORD_PATHS
    assert expected_digests == witness.PREREGISTRATION_SHA256
    assert _record_commit() == witness.PREREGISTRATION_COMMIT
    assert (
        witness.PREREGISTRATION_COMMITTED_AT
        == phase37_state.P37_PREREGISTRATION_COMMITTED_AT
    )
    assert witness.committed_at() == datetime.fromisoformat(
        phase37_state.P37_PREREGISTRATION_COMMITTED_AT
    )
    assert witness.committed_at().utcoffset() == timedelta(0)
    assert witness.REPO_ROOT == REPO_ROOT
    assert witness.fingerprint_mismatches() == [], (
        "the runtime fingerprint check reads a record that no longer hashes to its witness: "
        f"{witness.fingerprint_mismatches()}."
    )


def test_the_production_committed_at_is_what_git_reports() -> None:
    from refit import witness

    _skip_without_history()
    stamp = _git(
        "show", "-s", "--format=%cI", witness.PREREGISTRATION_COMMIT
    ).stdout.strip()
    assert stamp == witness.PREREGISTRATION_COMMITTED_AT, (
        f"refit/witness.py records {witness.PREREGISTRATION_COMMITTED_AT!r}; git reports "
        f"{stamp!r}."
    )


def test_the_fingerprint_refusal_names_an_edited_or_missing_record(
    tmp_path: Path,
) -> None:
    """Fail-closed control: a copy of the records with one edited and one missing is reported."""
    from refit import witness

    for relative_path in RECORD_PATHS:
        (tmp_path / relative_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative_path, tmp_path / relative_path)
    assert witness.fingerprint_mismatches(tmp_path) == []

    with (tmp_path / RECORD_MARKDOWN_PATH).open("ab") as handle:
        handle.write(b" ")
    assert witness.fingerprint_mismatches(tmp_path) == [RECORD_MARKDOWN_PATH]

    (tmp_path / RECORD_MODULE_PATH).unlink()
    assert witness.fingerprint_mismatches(tmp_path) == list(RECORD_PATHS)


# ---------------------------------------------------------------------------
# No quiet edit after the record (edge R1 idempotency, P4)
# ---------------------------------------------------------------------------


def test_no_later_commit_edits_the_record_quietly() -> None:
    _skip_without_history()
    offenders = uninvalidating_commits_touching_the_rule(
        REPO_ROOT, _record_commit(), RECORD_PATHS
    )
    assert offenders == [], (
        "a commit AFTER the record edits a record file without saying it INVALIDATES "
        f"{_record_commit()} by name: {offenders}. Editing the pre-registration mid-season "
        "to get a re-fit through is the act this arm forbids (P4)."
    )


def _synthetic_repo(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway repository holding one committed ``record.md``; returns (repo, its commit)."""
    repo = tmp_path / "synthetic"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", str(repo), cwd=tmp_path).returncode == 0
    _git("config", "user.email", "control@example.invalid", cwd=repo)
    _git("config", "user.name", "Fail Closed Control", cwd=repo)
    (repo / "record.md").write_text("the rule\n", encoding="utf-8")
    _git("add", "record.md", cwd=repo)
    assert _git("commit", "-q", "-m", "freeze the rule", cwd=repo).returncode == 0
    return repo, _git("rev-parse", "HEAD", cwd=repo).stdout.strip()


@pytest.mark.usefixtures("isolated_git_env")
def test_the_quiet_edit_arm_fires_and_passes_an_honest_invalidation(
    tmp_path: Path,
) -> None:
    """Fail-closed control in a synthetic repository, never this one."""
    repo, rule_commit = _synthetic_repo(tmp_path)
    record = repo / "record.md"

    record.write_text("the rule, quietly edited\n", encoding="utf-8")
    _git("add", "record.md", cwd=repo)
    assert _git("commit", "-q", "-m", "tidy up the wording", cwd=repo).returncode == 0
    offenders = uninvalidating_commits_touching_the_rule(
        repo, rule_commit, ("record.md",)
    )
    assert [subject for _, subject in offenders] == ["tidy up the wording"], offenders

    record.write_text("the rule, corrected\n", encoding="utf-8")
    _git("add", "record.md", cwd=repo)
    message = f"correct the rule\n\nThis commit INVALIDATES the pre-registration at {rule_commit}."
    assert _git("commit", "-q", "-m", message, cwd=repo).returncode == 0
    still = uninvalidating_commits_touching_the_rule(repo, rule_commit, ("record.md",))
    assert [subject for _, subject in still] == ["tidy up the wording"], (
        "the arm reported a commit that DOES name what it invalidates, or lost the quiet one: "
        f"{still}."
    )


@pytest.mark.usefixtures("isolated_git_env")
def test_a_failed_git_log_fails_closed(tmp_path: Path) -> None:
    """A rule commit git does not have makes ``git log`` exit non-zero: that RAISES, never ``[]``."""
    repo, _ = _synthetic_repo(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        uninvalidating_commits_touching_the_rule(repo, "f" * 40, ("record.md",))


# ---------------------------------------------------------------------------
# The first re-fit descends from the record (edge R1 adjacency)
# ---------------------------------------------------------------------------


def test_the_first_refit_descends_from_the_record() -> None:
    intent = _first_refit_intent_or_skip()
    code_commit = intent.get("code_commit")
    assert code_commit is not None, (
        f"the first install_intent (date {intent.get('date')}) records no code commit, so the "
        "first re-fit cannot be shown to descend from the pre-registration. The intent record "
        "is immutable; this is a real ordering gap, not a note."
    )
    assert not str(code_commit).endswith("+dirty"), (
        f"the first re-fit ran on a dirty tree ({code_commit}): its code is not a commit, so it "
        "cannot be shown to descend from the pre-registration."
    )
    assert _SHA1_RE.match(str(code_commit)), code_commit
    record = _record_commit()
    assert code_commit != record, (
        "the first re-fit ran AT the record commit; the record must strictly precede it."
    )
    _skip_without_history()
    forward = _git("merge-base", "--is-ancestor", record, code_commit)
    assert forward.returncode == 0, (
        f"the record commit {record} is NOT an ancestor of the first re-fit's code commit "
        f"{code_commit}: {forward.stderr.strip()}"
    )
    backward = _git("merge-base", "--is-ancestor", code_commit, record)
    assert backward.returncode != 0, (
        f"the first re-fit's code commit {code_commit} is an ancestor of the record commit."
    )


def test_the_first_refit_pending_skip_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules[__name__], "first_refit_intent", lambda _ledger_dir: None
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _first_refit_intent_or_skip()
    assert str(excinfo.value) == FIRST_REFIT_PENDING_SKIP_MESSAGE


def _intent(date_index: int, date: str, code_commit: str) -> dict[str, Any]:
    sha = f"{date_index}" * 64
    return {
        "event": "install_intent",
        "list_week": 8,
        "date_index": date_index,
        "date": date,
        "nfl_week": 9,
        "ids": {
            "wp": "wp_x",
            "ats": "ats_x",
            "ou": "ou_x",
            "blend": "b_x",
            "converter": "c_x",
        },
        "bet_rule_path": f"refit/bet_rules/{sha}.json",
        "bet_rule_sha256": sha,
        "code_commit": code_commit,
        "recorded_at_utc": f"{date}T22:05:00+00:00",
    }


def _attempt(
    date_index: int,
    date: str,
    outcome: str,
    code_commit: str,
    exception_class: str | None = None,
) -> dict[str, Any]:
    return {
        "event": "attempt",
        "list_week": 8,
        "date_index": date_index,
        "date": date,
        "nfl_week": 9,
        "cutoff_utc": f"{date}T22:00:00+00:00",
        "attempted_at_utc": f"{date}T22:01:00+00:00",
        "run_date_et": date,
        "code_commit": code_commit,
        "outcome": outcome,
        "check": "bundle_validation" if outcome == "refused" else None,
        "exception_class": exception_class,
        "before_ids": {},
        "after_ids": {},
        "no_final_score_ids": [],
        "recorded_at_utc": f"{date}T22:06:00+00:00",
    }


def _closed(date_index: int, date: str, status: str) -> dict[str, Any]:
    return {
        "event": "closed",
        "list_week": 8,
        "date_index": date_index,
        "date": date,
        "nfl_week": 9,
        "status": status,
        "closed_at_utc": f"{date}T22:07:00+00:00",
        "before_ids": {},
        "after_ids": {},
        "recorded_at_utc": f"{date}T22:07:00+00:00",
    }


def _write_dates(ledger_dir: Path, events: list[dict[str, Any]]) -> None:
    path = ledger_dir / "refit" / DATES_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def test_the_first_refit_is_found_however_its_install_was_recorded(
    tmp_path: Path,
) -> None:
    """Control: crash after the swap, killed before the attempt, and no intent at all."""
    first, second = "a" * 40, "b" * 40

    crashed = tmp_path / "crashed_after_swap"
    _write_dates(
        crashed,
        [
            _intent(1, "2026-11-03", first),
            _attempt(1, "2026-11-03", "crashed", first, exception_class="RuntimeError"),
            _closed(1, "2026-11-03", "ran late"),
            _intent(2, "2026-12-01", second),
            _attempt(2, "2026-12-01", "installed", second),
            _closed(2, "2026-12-01", "ran"),
        ],
    )
    found = first_refit_intent(crashed)
    assert found is not None
    assert (found["event"], found["date_index"], found["code_commit"]) == (
        "install_intent",
        1,
        first,
    )

    killed = tmp_path / "killed_before_attempt"
    _write_dates(killed, [_intent(1, "2026-11-03", first)])
    found = first_refit_intent(killed)
    assert found is not None
    assert (found["event"], found["code_commit"]) == ("install_intent", first)

    refused_only = tmp_path / "refused_without_intent"
    _write_dates(refused_only, [_attempt(1, "2026-11-03", "refused", first)])
    assert first_refit_intent(refused_only) is None
    assert first_refit_intent(tmp_path / "no_ledger") is None


# ---------------------------------------------------------------------------
# What Phase 37 must leave byte-unchanged: the shadow pin and the old gate path (RFIT-06)
# ---------------------------------------------------------------------------


def test_the_gate_files_are_byte_unchanged() -> None:
    assert set(phase37_state.P37_UNTOUCHED_GATE_FILES) == set(
        record_module.UNTOUCHED_GATE_FILES
    ), "the witness and the record disagree about which files are the old gate path."
    for relative_path, expected in phase37_state.P37_UNTOUCHED_GATE_FILES.items():
        actual = _normalized_sha256(relative_path)
        assert actual == expected, (
            f"{relative_path} CHANGED in Phase 37; the old deploy-gate path is byte-unchanged "
            f"by design (RFIT-06, R5).\n  recorded: {expected}\n  now:      {actual}"
        )


def test_the_shadow_pin_is_witnessed() -> None:
    entry = phase37_state.P37_SHADOW_PIN_WITNESS[SHADOW_PIN_PATH]
    assert set(phase37_state.P37_SHADOW_PIN_WITNESS) == {SHADOW_PIN_PATH}
    assert _normalized_sha256(SHADOW_PIN_PATH) == entry["normalized_sha256"], (
        f"{SHADOW_PIN_PATH} CHANGED after its witness was taken. A frozen arm that moved is not "
        "frozen."
    )
    _skip_without_history()
    resolved = _git("log", "-1", "--format=%H", "--", SHADOW_PIN_PATH).stdout.strip()
    assert resolved == entry["commit"], (
        f"{SHADOW_PIN_PATH} was re-committed: git resolves {resolved}, witnessed {entry['commit']}."
    )


# ---------------------------------------------------------------------------
# Controls: the guards above can actually fire
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
            TestTheRecordWasCommittedAloneBeforeWeek8().test_ii_the_commit_is_a_strict_ancestor_of_head()
        assert calls == []

    def test_no_git_binary_reads_as_unavailable_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_git(
            *_args: str, cwd: Path | None = None
        ) -> subprocess.CompletedProcess:
            raise FileNotFoundError("git")

        monkeypatch.setattr(sys.modules[__name__], "_git", _no_git)
        assert _git_history_is_unavailable() is True

    @pytest.mark.parametrize(
        ("partial_filter", "unavailable"),
        [("tree:0", True), ("blob:none", False), (None, False)],
    )
    def test_only_a_treeless_partial_clone_reads_as_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        partial_filter: str | None,
        unavailable: bool,
    ) -> None:
        def _fake_git(
            *args: str, cwd: Path | None = None
        ) -> subprocess.CompletedProcess:
            stdout = {
                "--git-dir": ".git\n",
                "--is-shallow-repository": "false\n",
            }.get(args[-1], "")
            if args[0] == "config":
                stdout = (
                    f"remote.origin.partialclonefilter {partial_filter}\n"
                    if partial_filter
                    else ""
                )
            return subprocess.CompletedProcess(["git", *args], 0, stdout, "")

        monkeypatch.setattr(sys.modules[__name__], "_git", _fake_git)
        assert _git_history_is_unavailable() is unavailable

    def test_history_is_available_on_this_checkout(self) -> None:
        """The guard does not skip a real, full checkout: here the ancestry checks RUN."""
        if not (REPO_ROOT / ".git").exists():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        assert _git_history_is_unavailable() is False

    @pytest.mark.parametrize("relative_path", sorted(_WITNESSED_DIGESTS))
    def test_a_one_byte_edit_would_move_the_digest(self, relative_path: str) -> None:
        raw = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(raw + b" ").hexdigest() != _WITNESSED_DIGESTS[relative_path]
        )
