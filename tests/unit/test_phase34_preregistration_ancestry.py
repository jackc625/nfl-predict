"""The three Phase-34 pre-registration records were committed alone, before week W, and published.

Plan 34-22 (LDGR-10, LDGR-11, D-06, D-15). The 2026 forward verdict rests on three records:

  * ``backtest/fill_conventions.py``   -- the pricing and sizing rule ``fill-v1`` (LDGR-11);
  * ``backtest/recipe_registry.py``    -- the recipe registry's first entry (LDGR-03);
  * ``backtest/verdict_scope_2026.py`` -- which weeks, arm and outcome the verdict counts (LDGR-10).

For EACH record this module asserts, against the witness in ``tests/phase34_state.py``:

(i)   ``git log -1 --format=%H -- <path>`` resolves EXACTLY to the witnessed commit;
(ii)  that commit is a STRICT ancestor of HEAD (``--is-ancestor`` succeeds on equality, so
      ``!= HEAD`` is checked too);
(iii) the commit touches ONLY that record;
(iv)  the record's newline-normalized sha256 equals the witnessed digest;
(v)   the commit's committer date is BEFORE week W's first lock;
(vi)  the commit is reachable from GitHub's ``master`` (``refs/remotes/origin/master``).

And, once, for the declaration: the witnessed first lock is ``utils.game_lock`` of the earliest
week-W kickoff in the recorded schedule, the declaration's constants are the registered values,
week 22 is in scope and week 23 is not, ``load_verdict_scope()`` returns the matching scope, and
no ledger row of a counted week lacks its model, blend, recipe or fill stamp.

WHAT THIS CANNOT PROVE. The committer date is set by the committing machine (T-34-75, accepted).
Reachability from the remote master plus the push time recorded in the Plan 34-22 SUMMARY is the
proportionate evidence; stronger timestamping is FUT-01.

The expected values live in ``tests/phase34_state.py`` -- outside the files they witness, in a
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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backtest import verdict_scope_2026 as declaration
from forward_ledger.declarations import (
    REQUIRED_DECLARATION_CONSTANTS,
    VerdictScope,
    load_verdict_scope,
    verdict_scope_label,
)
from forward_ledger.schema import VERDICT_SCOPE_PRE_VERDICT, VERDICT_SCOPE_VERDICT
from forward_ledger.store import LEDGER_DIR, entries_to_frame, ledger_path, read_entries
from forward_ledger.verdict import DECISION_RUN_LEAD
from tests import phase34_state
from utils.current_slate import default_schedule_path
from utils.date_utils import NFL_TOTAL_WEEKS
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

FILL_CONVENTIONS_PATH = "backtest/fill_conventions.py"
RECIPE_REGISTRY_PATH = "backtest/recipe_registry.py"
DECLARATION_PATH = "backtest/verdict_scope_2026.py"
RECORD_PATHS: tuple[str, ...] = (
    FILL_CONVENTIONS_PATH,
    RECIPE_REGISTRY_PATH,
    DECLARATION_PATH,
)
WITNESS_MODULE_PATH = "tests/phase34_state.py"
REMOTE_MASTER_REF = "refs/remotes/origin/master"

# The stamps a counted row must carry: the four the SPEC names for the LDGR-10 adjacency edge.
COUNTED_ROW_STAMP_COLUMNS: tuple[str, ...] = (
    "model_artifact_id",
    "blend_id",
    "recipe_id",
    "fill_convention_id",
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Copied in shape from tests/unit/test_preregistration_ancestry.py so the guards read alike.
SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout), so "
    "`git merge-base --is-ancestor` would fail for want of history rather than for want of "
    "ancestry -- and the two are indistinguishable from the exit code alone. Skipping BEFORE "
    "any ancestry call rather than reporting a false ancestry violation."
)
NO_REMOTE_MASTER_SKIP_MESSAGE = (
    f"{REMOTE_MASTER_REF} is absent (no `origin` remote, or it was never fetched), so whether "
    "the record is published on GitHub's master cannot be asked of this checkout. Skipping "
    "rather than reporting an unpublished record."
)
NO_SCHEDULE_SKIP_MESSAGE = (
    "the recorded schedule (silver games) is absent from this checkout -- data/ is not "
    "tracked -- so the witnessed first lock cannot be re-derived here."
)
NO_LEDGER_SKIP_MESSAGE = (
    "the forward ledger is absent from this checkout -- ledger/ is not tracked publicly -- so "
    "the counted rows' stamps cannot be read here."
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


def _skip_without_history() -> None:
    """Skip BEFORE any history-dependent git call when history cannot answer it."""
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes."""
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _witness(relative_path: str) -> dict[str, str]:
    return phase34_state.P34_PREREGISTRATION_WITNESS[relative_path]


def _first_lock_utc() -> datetime:
    return datetime.fromisoformat(phase34_state.P34_W_FIRST_LOCK_UTC)


def _expected_scope() -> VerdictScope:
    """The registered values, written out here rather than read back from the declaration."""
    return VerdictScope(
        season=2026,
        start_week=phase34_state.P34_W,
        end_week=22,
        includes_playoff_weeks=True,
        includes_neutral_site_games=True,
        counted_arm="live",
        outcome_rule="in_force_correction",
        fill_convention_id="fill-v1",
        bootstrap_regime_weeks=(2, 3, 4),
    )


# ---------------------------------------------------------------------------
# The witness itself, and the checks that need no git history
# ---------------------------------------------------------------------------


class TestTheWitnessIsWellFormed:
    def test_it_covers_exactly_the_three_records(self) -> None:
        assert set(phase34_state.P34_PREREGISTRATION_WITNESS) == set(RECORD_PATHS), (
            "the witness and this module disagree about which files ARE the Phase-34 "
            "pre-registration; a record without a witness is editable after the fact."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_each_entry_has_its_shapes(self, relative_path: str) -> None:
        entry = _witness(relative_path)
        assert set(entry) == {"commit", "normalized_sha256"}, entry
        assert _SHA1_RE.match(entry["commit"]), entry["commit"]
        assert _SHA256_RE.match(entry["normalized_sha256"]), entry["normalized_sha256"]

    def test_w_and_its_instants_are_consistent(self) -> None:
        first_lock = _first_lock_utc()
        first_run = datetime.fromisoformat(phase34_state.P34_W_FIRST_DECISION_RUN_UTC)
        assert first_lock.utcoffset() == timedelta(0), (
            phase34_state.P34_W_FIRST_LOCK_UTC
        )
        assert first_run.utcoffset() == timedelta(0), (
            phase34_state.P34_W_FIRST_DECISION_RUN_UTC
        )
        assert first_run == first_lock - DECISION_RUN_LEAD
        assert isinstance(phase34_state.P34_W, int)
        assert 1 <= phase34_state.P34_W <= NFL_TOTAL_WEEKS

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_each_record_is_tracked(self, relative_path: str) -> None:
        assert (REPO_ROOT / relative_path).is_file(), (
            f"{relative_path} is missing. It is TRACKED, so this is a broken checkout."
        )
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
            f"  recorded (tests/phase34_state.py): {expected}\n"
            f"  recomputed from the working tree:  {actual}\n"
            "Editing a pre-registration after week W's first lock does not fix a bug -- it "
            "destroys the evidence. A new rule is a NEW record with its own witness."
        )

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_no_record_contains_its_own_hash(self, relative_path: str) -> None:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert _witness(relative_path)["normalized_sha256"] not in content
        assert _witness(relative_path)["commit"] not in content


# ---------------------------------------------------------------------------
# The ordering: alone, before W's first lock, under HEAD, on GitHub's master
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative_path", RECORD_PATHS)
class TestEachRecordWasCommittedAloneBeforeWeekW:
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

    def test_ii_the_commit_is_a_strict_ancestor_of_head(
        self, relative_path: str
    ) -> None:
        _skip_without_history()
        record = _witness(relative_path)["commit"]
        head = _git("rev-parse", "HEAD").stdout.strip()
        assert record != head, (
            f"the record commit IS HEAD ({record}); nothing follows it yet, so it cannot be "
            "shown to have preceded anything."
        )
        ancestry = _git("merge-base", "--is-ancestor", record, head)
        assert ancestry.returncode == 0, (
            f"the record commit {record} is NOT an ancestor of HEAD ({head})."
        )

    def test_iii_the_commit_touches_only_the_record(self, relative_path: str) -> None:
        _skip_without_history()
        listing = _git(
            "show", "--name-only", "--format=", _witness(relative_path)["commit"]
        )
        assert listing.returncode == 0, listing.stderr
        touched = sorted(path for path in listing.stdout.split() if path)
        assert touched == [relative_path], (
            f"the witnessed commit touches {touched}; a record committed alongside anything "
            "else cannot be shown to have preceded it."
        )

    def test_v_the_commit_predates_week_ws_first_lock(self, relative_path: str) -> None:
        _skip_without_history()
        commit = _witness(relative_path)["commit"]
        stamp = _git("show", "-s", "--format=%cI", commit).stdout.strip()
        committed_at = datetime.fromisoformat(stamp)
        assert committed_at < _first_lock_utc(), (
            f"{relative_path} was committed at {stamp}, NOT before week "
            f"{phase34_state.P34_W}'s first lock ({phase34_state.P34_W_FIRST_LOCK_UTC})."
        )

    def test_vi_the_commit_is_on_githubs_master(self, relative_path: str) -> None:
        _skip_without_history()
        if _git("rev-parse", "--verify", "--quiet", REMOTE_MASTER_REF).returncode != 0:
            pytest.skip(NO_REMOTE_MASTER_SKIP_MESSAGE)
        commit = _witness(relative_path)["commit"]
        reachable = _git("merge-base", "--is-ancestor", commit, REMOTE_MASTER_REF)
        assert reachable.returncode == 0, (
            f"the record commit {commit} is NOT reachable from {REMOTE_MASTER_REF}: the "
            "pre-registration was never published (or this clone's remote ref is stale -- "
            "`git fetch origin master` and re-run)."
        )

    def test_the_witness_landed_after_the_record(self, relative_path: str) -> None:
        _skip_without_history()
        witness_commit = _git(
            "log", "-1", "--format=%H", "--", WITNESS_MODULE_PATH
        ).stdout.strip()
        record = _witness(relative_path)["commit"]
        assert _SHA1_RE.match(witness_commit), witness_commit
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
# The declaration's content, and the first lock it was registered before
# ---------------------------------------------------------------------------


class TestTheDeclarationSaysWhatWasRegistered:
    def test_the_constants_are_the_registered_values(self) -> None:
        declared = {
            name: getattr(declaration, name) for name in REQUIRED_DECLARATION_CONSTANTS
        }
        assert declared == {
            "VERDICT_SEASON": 2026,
            "VERDICT_START_WEEK": phase34_state.P34_W,
            "VERDICT_END_WEEK": 22,
            "INCLUDES_PLAYOFF_WEEKS": True,
            "INCLUDES_NEUTRAL_SITE_GAMES": True,
            "COUNTED_ARM": "live",
            "OUTCOME_RULE": "in_force_correction",
            "FILL_CONVENTION_ID": "fill-v1",
            "BOOTSTRAP_REGIME_WEEKS": (2, 3, 4),
        }

    def test_it_declares_every_required_constant_and_nothing_imported(self) -> None:
        for name in REQUIRED_DECLARATION_CONSTANTS:
            assert hasattr(declaration, name), name
        source = (REPO_ROOT / DECLARATION_PATH).read_text(encoding="utf-8")
        import_lines = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert import_lines == ["from __future__ import annotations"], import_lines

    def test_the_loader_returns_the_matching_scope(self) -> None:
        assert load_verdict_scope() == _expected_scope()

    def test_week_22_counts_and_there_is_no_week_23(self) -> None:
        scope = load_verdict_scope()
        assert scope.end_week == NFL_TOTAL_WEEKS == 22
        w = phase34_state.P34_W
        assert verdict_scope_label(2026, w - 1, scope) == VERDICT_SCOPE_PRE_VERDICT
        assert verdict_scope_label(2026, w, scope) == VERDICT_SCOPE_VERDICT
        assert verdict_scope_label(2026, 22, scope) == VERDICT_SCOPE_VERDICT
        assert verdict_scope_label(2026, 23, scope) == VERDICT_SCOPE_PRE_VERDICT
        assert verdict_scope_label(2025, w, scope) == VERDICT_SCOPE_PRE_VERDICT

    def test_the_bootstrap_weeks_are_all_before_w(self) -> None:
        assert max(declaration.BOOTSTRAP_REGIME_WEEKS) < phase34_state.P34_W

    def test_the_witnessed_first_lock_is_the_lock_of_week_ws_earliest_kickoff(
        self,
    ) -> None:
        schedule_path = default_schedule_path()
        if not schedule_path.is_file():
            pytest.skip(NO_SCHEDULE_SKIP_MESSAGE)
        games = pd.read_parquet(
            schedule_path, columns=["game_id", "season", "week", "kickoff_et"]
        )
        week_games = games[
            (games["season"] == 2026) & (games["week"] == phase34_state.P34_W)
        ]
        assert not week_games.empty, (
            f"week {phase34_state.P34_W} has no scheduled games"
        )
        earliest = week_games.sort_values("kickoff_et").iloc[0]
        lock = game_lock(earliest["kickoff_et"], game_id=str(earliest["game_id"]))
        assert lock == _first_lock_utc(), (
            f"the witnessed first lock {phase34_state.P34_W_FIRST_LOCK_UTC} is not the lock of "
            f"week {phase34_state.P34_W}'s earliest kickoff ({earliest['game_id']}, "
            f"{earliest['kickoff_et']}): {lock.isoformat()}."
        )

    def test_no_counted_row_lacks_a_stamp(self) -> None:
        if not ledger_path(REPO_ROOT / LEDGER_DIR).exists():
            pytest.skip(NO_LEDGER_SKIP_MESSAGE)
        frame = entries_to_frame(read_entries(REPO_ROOT / LEDGER_DIR))
        counted = frame[
            (frame["season"] == 2026) & (frame["week"] >= phase34_state.P34_W)
        ]
        unstamped = counted[counted[list(COUNTED_ROW_STAMP_COLUMNS)].isna().any(axis=1)]
        assert unstamped.empty, (
            f"{len(unstamped)} ledger row(s) of a counted week lack a model, blend, recipe or "
            f"fill stamp, so week {phase34_state.P34_W} holds an old-writer row and cannot be W:\n"
            f"{unstamped[['game_id', 'week', 'target', 'arm']].to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Controls: the guards above can actually fire
# ---------------------------------------------------------------------------


class TestTheGuardsAreReal:
    def test_the_shallow_checkout_guard_actually_fires(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sys.modules[__name__], "_git_history_is_unavailable", lambda: True
        )
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_without_history()
        assert str(excinfo.value) == SHALLOW_SKIP_MESSAGE

    @pytest.mark.parametrize("relative_path", RECORD_PATHS)
    def test_a_one_byte_edit_would_move_the_digest(self, relative_path: str) -> None:
        raw = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(raw + b" ").hexdigest()
            != _witness(relative_path)["normalized_sha256"]
        )
