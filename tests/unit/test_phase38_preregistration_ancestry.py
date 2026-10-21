"""The 2026 verdict pre-registration was committed alone, before week 8, and is published.

Plan 38-03 (CLEAN-04, D-11, D-12, SPEC R6 AC 19). How the 2026 forward profit verdict is judged
rests on ONE pre-registration in two files, landed together in one commit that touches nothing else:

  * ``SEASON-VERDICT-PREREGISTRATION.md`` -- the human-readable half (a never-move root record);
  * ``backtest/verdict_test_2026.py``     -- the constants the readout reads.

Against the witness in ``tests/phase38_state.py`` this module asserts:

(i)   ``git log -1 --format=%H -- <path>`` resolves EXACTLY to the witnessed commit, for both paths;
(ii)  that commit is a STRICT ancestor of HEAD (``--is-ancestor`` succeeds on equality, so
      ``!= HEAD`` is checked too);
(iii) the commit touches EXACTLY the two record paths;
(iv)  each record's newline-normalized sha256 equals its witnessed digest;
(v)   the commit's committer date is BEFORE week 8's first decision run (D-11);
(vi)  the commit is reachable from this clone's LOCAL tracking ref of GitHub's ``master``
      (``refs/remotes/origin/master``), read at test time -- as fresh as the last fetch; no
      network read is made.

And beside those: the witness landed in a commit after the record; no later commit edits a record
file without INVALIDATES and the record sha in its message.

THE FROZEN VALUES (D-12, research OQ5). The record does not copy alpha, the bootstrap settings or
the verdict words as literals: it IMPORTS them by identity from ``backtest.ev_chain_constants``,
whose alpha is in turn ``backtest.diagnose.SIGNIFICANCE_ALPHA``. An identity import means an edit
upstream would move the 2026 rule WITHOUT touching either record file, and the digest arms above
would stay green. So ``TestTheFrozenValuesStillHold`` pins each imported value as a literal HERE
(never in the record) and asserts each is the same object as in ``backtest.ev_chain_constants``.
It also re-derives the record's two disclosures from the stores: the weeks already graded when it
was committed, and the one tied game.

WHAT THIS CANNOT PROVE. The committer date is set by the committing machine (T-38-02, accepted).
Arm (vi) reads the local tracking ref at test time, so it proves the record is published NOW, not
WHEN it was published. The push time (2026-10-21 19:22:45 ET) and the post-push readback from
GitHub are recorded in the Plan 38-01 SUMMARY; that record plus this ordering is the proportionate
evidence. No test can prove that nobody tuned a value after seeing a number; the ordering is what
makes that claim checkable at all.

Hashes are over NEWLINE-NORMALIZED bytes: ``core.autocrlf`` is true with no ``.gitattributes``.

No test here writes into the repository or a production store: the synthetic-repo controls build
throwaway repositories under ``tmp_path``, and the disclosure checks only READ
``ledger/forward_2026.jsonl`` and ``data/silver/games.parquet``, stepping aside with an "absent"
message when either is missing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from api.cache_schema import BET_STATUS_LIVE
from backtest import ev_chain_constants
from backtest import verdict_test_2026 as record_module
from forward_ledger import declarations
from forward_ledger.store import LEDGER_DIR, entries_to_frame, ledger_path, read_entries
from forward_ledger.verdict import verdict_rows
from tests import phase38_state
from tests.unit.test_phase33_preregistration_ancestry import (
    uninvalidating_commits_touching_the_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

RECORD_MARKDOWN_PATH = "SEASON-VERDICT-PREREGISTRATION.md"
RECORD_MODULE_PATH = "backtest/verdict_test_2026.py"
RECORD_PATHS: tuple[str, ...] = (RECORD_MARKDOWN_PATH, RECORD_MODULE_PATH)
WITNESS_MODULE_PATH = "tests/phase38_state.py"
REMOTE_MASTER_REF = "refs/remotes/origin/master"

SILVER_GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"

# The one 2026 game that ended in a tie before the record commit (D-21, TIE_DISCLOSURE).
DISCLOSED_TIE_GAME_IDS: list[str] = ["2026_W05_NYG@WAS"]

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Copied in shape from tests/unit/test_phase37_preregistration_ancestry.py so the guards read alike.
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
LEDGER_ABSENT_SKIP_MESSAGE = (
    "the forward ledger store is absent on this checkout, so the record's disclosures cannot be "
    "re-derived from it. Skipping rather than reporting a disclosure the store never contradicted."
)
SILVER_GAMES_ABSENT_SKIP_MESSAGE = (
    "data/silver/games.parquet is absent on this checkout, so the tie disclosure cannot be "
    "re-derived from the schedule. Skipping rather than reporting a tie nobody can see."
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
    return phase38_state.P38_PREREGISTRATION_WITNESS[relative_path]


def _record_commit() -> str:
    return _witness(RECORD_MODULE_PATH)["commit"]


def _deadline_utc() -> datetime:
    return datetime.fromisoformat(phase38_state.P38_DEADLINE_UTC)


def _committed_at_utc() -> pd.Timestamp:
    """The witnessed committed-at instant as a tz-aware UTC timestamp."""
    committed_at = pd.Timestamp(phase38_state.P38_PREREGISTRATION_COMMITTED_AT)
    return committed_at.tz_convert("UTC")


def _ledger_entries_or_skip() -> list:
    """The production ledger's entries, read-only, skipping with an "absent" message if none."""
    ledger_dir = REPO_ROOT / LEDGER_DIR
    if not ledger_path(ledger_dir).exists():
        pytest.skip(LEDGER_ABSENT_SKIP_MESSAGE)
    return read_entries(ledger_dir)


# ---------------------------------------------------------------------------
# The witness itself, and the checks that need no git history
# ---------------------------------------------------------------------------


class TestTheWitnessIsWellFormed:
    def test_it_covers_exactly_the_two_records(self) -> None:
        assert set(phase38_state.P38_PREREGISTRATION_WITNESS) == set(RECORD_PATHS), (
            "the witness and this module disagree about which files ARE the 2026 verdict "
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
            f"  recorded (tests/phase38_state.py): {expected}\n"
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
        assert deadline.utcoffset() == timedelta(0), phase38_state.P38_DEADLINE_UTC
        assert (
            phase38_state.P38_DEADLINE_UTC == record_module.PREREGISTRATION_DEADLINE_UTC
        ), (
            "the witness deadline and the record's PREREGISTRATION_DEADLINE_UTC disagree: "
            f"{phase38_state.P38_DEADLINE_UTC} vs {record_module.PREREGISTRATION_DEADLINE_UTC}."
        )

    def test_the_recorded_committed_at_precedes_the_deadline(self) -> None:
        committed_at = datetime.fromisoformat(
            phase38_state.P38_PREREGISTRATION_COMMITTED_AT
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
        assert stamp == phase38_state.P38_PREREGISTRATION_COMMITTED_AT, (
            f"git reports committer date {stamp!r}, the witness records "
            f"{phase38_state.P38_PREREGISTRATION_COMMITTED_AT!r}."
        )
        assert datetime.fromisoformat(stamp) < _deadline_utc(), (
            f"the record was committed at {stamp}, NOT before week 8's first decision run "
            f"({phase38_state.P38_DEADLINE_UTC}, D-11)."
        )

    def test_vi_the_commit_is_reachable_from_the_local_origin_master_ref(self) -> None:
        """Reachability from THIS clone's ``refs/remotes/origin/master`` tracking ref, NOW.

        No network read is made: the ref is as fresh as the last fetch, so this proves the record
        is published now, not when. The push itself was read back from GitHub once, at Plan 38-01
        (19:22:45 ET, 2026-10-21).
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
# The identity-imported values and the record's disclosures (D-12, research OQ5)
# ---------------------------------------------------------------------------


class TestTheFrozenValuesStillHold:
    def test_the_imported_values_are_the_frozen_ones(self) -> None:
        """Each identity import still holds its frozen number, and is still the same object.

        The literals live HERE and never in the record: the record imports them, so an edit to
        backtest/diagnose.py or backtest/ou_monetization.py would otherwise move the 2026 rule
        without touching either record file.
        """
        assert record_module.ALPHA == 0.05
        assert record_module.BOOTSTRAP_B == 2000
        assert record_module.BOOTSTRAP_SEED == 2704
        assert record_module.ROI_MIN_ATTAINABLE_P == 1.0 / 2001
        assert tuple(record_module.VERDICT_TOKENS) == (
            "PROFITABLE_CLEAN",
            "UNPROFITABLE_CLEAN",
            "INCONCLUSIVE_CLEAN",
            "UNDISCHARGEABLE_NO_BETS",
            "UNDISCHARGEABLE_NO_CHAIN",
        )
        assert record_module.VERDICT_TOKENS[3] == record_module.UNDISCHARGEABLE_NO_BETS
        assert record_module.BH_FAMILY_SIZE == 3 == len(record_module.VERDICT_TARGETS)
        assert record_module.VERDICT_SCOPE_MODULE == declarations.VERDICT_SCOPE_MODULE

        for name in (
            "ALPHA",
            "BOOTSTRAP_B",
            "BOOTSTRAP_SEED",
            "ROI_MIN_ATTAINABLE_P",
            "ROI_P_VALUE_METHOD",
            "UNDISCHARGEABLE_NO_BETS",
            "VERDICT_TOKENS",
        ):
            assert getattr(record_module, name) is getattr(ev_chain_constants, name), (
                f"backtest.verdict_test_2026.{name} is no longer the backtest.ev_chain_constants "
                "object: the record froze an IDENTITY import, and a second copy is a second "
                "answer (D-12)."
            )

    def test_the_markdown_states_each_frozen_value(self) -> None:
        text = (REPO_ROOT / RECORD_MARKDOWN_PATH).read_text(encoding="utf-8")
        required = (
            *record_module.VERDICT_TOKENS,
            "2000",
            "2704",
            "0.05",
            "m = 3",
            "strictly less than",
            record_module.BINDING_RECORD_PATH,
        )
        missing = [phrase for phrase in required if phrase not in text]
        assert missing == [], (
            f"{RECORD_MARKDOWN_PATH} no longer states {missing}; the human half and the "
            "constants half of the record must state the same values."
        )

    def test_the_graded_weeks_disclosure_is_what_the_ledger_shows(self) -> None:
        entries = _ledger_entries_or_skip()
        rows = verdict_rows(entries, declarations.load_verdict_scope())
        graded_at = pd.to_datetime(rows["graded_at"], utc=True)
        graded_before_commit = rows[
            (rows["status"] == BET_STATUS_LIVE)
            & graded_at.notna()
            & (graded_at <= _committed_at_utc())
        ]
        weeks = tuple(sorted({int(week) for week in graded_before_commit["week"]}))
        assert weeks == record_module.WEEKS_GRADED_AT_RECORD_COMMIT, (
            f"the ledger shows counted live-arm bets graded at or before "
            f"{phase38_state.P38_PREREGISTRATION_COMMITTED_AT} in weeks {weeks}; the record "
            f"discloses {record_module.WEEKS_GRADED_AT_RECORD_COMMIT}."
        )

    def test_the_tie_disclosure_holds(self) -> None:
        if not SILVER_GAMES_PATH.exists():
            pytest.skip(SILVER_GAMES_ABSENT_SKIP_MESSAGE)
        entries = _ledger_entries_or_skip()
        games = pd.read_parquet(SILVER_GAMES_PATH)
        kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
        tied = games[
            (games["season"] == 2026)
            & games["home_score"].notna()
            & games["away_score"].notna()
            & (games["home_score"] == games["away_score"])
            & (kickoff < _committed_at_utc())
        ]
        tied_ids = sorted(tied["game_id"])
        assert tied_ids == DISCLOSED_TIE_GAME_IDS, (
            f"the 2026 games tied before the record commit are {tied_ids}; the record's "
            f"TIE_DISCLOSURE names {DISCLOSED_TIE_GAME_IDS}."
        )
        for game_id in DISCLOSED_TIE_GAME_IDS:
            assert game_id in record_module.TIE_DISCLOSURE

        ledger_game_ids = set(entries_to_frame(entries)["game_id"])
        assert ledger_game_ids.isdisjoint(tied_ids), (
            f"the ledger holds a row for a tied game {sorted(ledger_game_ids & set(tied_ids))}; "
            "the record disclosed that none exists, so no counted result was re-graded."
        )


# ---------------------------------------------------------------------------
# No quiet edit after the record (SPEC AC 19)
# ---------------------------------------------------------------------------


def test_no_later_commit_edits_the_record_quietly() -> None:
    _skip_without_history()
    offenders = uninvalidating_commits_touching_the_rule(
        REPO_ROOT, _record_commit(), RECORD_PATHS
    )
    assert offenders == [], (
        "a commit AFTER the record edits a record file without saying it INVALIDATES "
        f"{_record_commit()} by name: {offenders}. Editing the pre-registration mid-season "
        "after results exist is the act this arm forbids."
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

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_a_one_byte_edit_would_move_the_digest(self, relative_path: str) -> None:
        raw = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(raw + b" ").hexdigest()
            != _witness(relative_path)["normalized_sha256"]
        )
