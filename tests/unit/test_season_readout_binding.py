"""The season-close readout's binding mode (Phase 38, Plan 38-09; SPEC R8, R9).

The binding verdict is computed exactly as SEASON-VERDICT-PREREGISTRATION.md froze it: each target's
raw figures come from the preview's one computation (``scripts.season_readout.target_figures``),
its p-value from ``backtest.roi_significance.roi_ci_and_p`` over its counted bets' (season, week)
blocks, the correction from ``benjamini_hochberg_adjusted`` at m = 3, and the word from
``assign_verdict_token``. Decisions read unrounded values, and no closing-line value enters.

COLD-05: every store and every git repository these tests use lives under ``tmp_path`` -- never
the real ``docs/records/``, ``ledger/``, ``data/``, ``artifacts/`` or ``logs/``. The ledger rows are
written through the real store (``append_rows`` / ``apply_updates``) and read back through
``forward_ledger.verdict.verdict_rows``, the one reader of counted rows.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import backtest.verdict_test_2026
from backtest.midseason_switch_2026 import (
    DAY_BEFORE_RETIRED_FROM,
    R6_COVERAGE_END_WEEK,
)
from backtest.profitability_2025 import (
    TargetMeasurement,
    assign_verdict_token,
    benjamini_hochberg_adjusted,
)
from backtest.roi_significance import roi_ci_and_p
from backtest.verdict_test_2026 import (
    ALPHA,
    BH_FAMILY_SIZE,
    BINDING_RECORD_PATH,
    ROI_MIN_ATTAINABLE_P,
    UNDISCHARGEABLE_NO_BETS,
    VERDICT_TARGETS,
    VERDICT_TOKENS,
)
from forward_ledger.anchor import write_anchor_commit
from forward_ledger.declarations import load_verdict_scope
from forward_ledger.settle import DEFAULT_SILVER_DIR, SILVER_GAMES_FILENAME
from forward_ledger.shadow import SHADOW_PIN
from forward_ledger.store import (
    LEDGER_DIR,
    append_rows,
    apply_updates,
    ledger_head,
    ledger_path,
    read_entries,
)
from forward_ledger.verdict import verdict_rows
from pipeline.skip_log import SKIP_RECORD_PATH
from refit.records import DATES_FILENAME, refit_dir
from scripts import compare_refit_arms, season_readout
from tests import phase38_state
from tests.unit.test_forward_ledger_store import graded, key_of, make_row

# The real repository: only READ here (the R6 record files are copied into synthetic repos).
REAL_REPO_ROOT = Path(__file__).resolve().parents[2]
RULE_FILES: tuple[str, ...] = (
    "SEASON-VERDICT-PREREGISTRATION.md",
    "backtest/verdict_test_2026.py",
)

# The synthetic lock commit's committer date (the real R6 lock's instant).
LOCK_COMMITTED_AT: str = "2026-10-21T19:19:13-04:00"

# The coverage's last week and the switch week, read from the record so a moved switch needs no
# edit here. RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md section 8: the switch moved from
# week 13 to week 10, so these are now weeks 9 and 10 (they were literals 12 and 13).
LAST_COVERAGE_WEEK = R6_COVERAGE_END_WEEK
SWITCH_WEEK = DAY_BEFORE_RETIRED_FROM[1]


def _sunday_kickoff(week: int) -> datetime:
    """Week *week*'s Sunday, 13:00 ET: (week - 1) weeks after the 2026 week-1 Sunday, 2026-09-13."""
    return datetime.combine(
        date(2026, 9, 13) + timedelta(weeks=week - 1),
        time(13, 0),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)


# One ET kickoff day per week (the recorded schedule refuses two weeks on one day).
KICKOFF_BY_WEEK: dict[int, datetime] = {
    5: datetime(2026, 10, 11, 17, 0, tzinfo=UTC),
    6: datetime(2026, 10, 18, 17, 0, tzinfo=UTC),
    LAST_COVERAGE_WEEK: _sunday_kickoff(LAST_COVERAGE_WEEK),
    SWITCH_WEEK: _sunday_kickoff(SWITCH_WEEK),
    22: datetime(2027, 2, 14, 23, 30, tzinfo=UTC),
}

# The verdict words, by their VERDICT_TOKENS position (profitable, unprofitable, inconclusive,
# no bets, no chain) -- never retyped.
_PROFITABLE, _UNPROFITABLE, _INCONCLUSIVE, _NO_BETS, _NO_CHAIN = VERDICT_TOKENS
assert _NO_BETS == UNDISCHARGEABLE_NO_BETS

# The complete fixture's games: two in week 5, two in week 6, the Super Bowl in week 22.
WEEK5_GAMES: tuple[str, ...] = ("2026_W05_TB@DAL", "2026_W05_BUF@LA")
WEEK6_GAMES: tuple[str, ...] = ("2026_W06_SF@SEA", "2026_W06_MIA@NE")
SUPER_BOWL: str = "2026_W22_KC@PHI"
BET_GAMES: tuple[str, ...] = (*WEEK5_GAMES, *WEEK6_GAMES, SUPER_BOWL)

# A game whose only counted row is a suppressed (no-bet) WP candidate.
NO_BET_GAME: str = "2026_W06_DEN@LV"

# A scored game nobody decided: the binding run reads it as missed (it never blocks the run).
MISSED_GAME: str = "2026_W06_GB@CHI"

# Every counted bet's in-force grade, per (game, target).
COMPLETE_GRADES: dict[tuple[str, str], str] = {
    ("2026_W05_TB@DAL", "wp"): "win",
    ("2026_W05_TB@DAL", "ats"): "loss",
    ("2026_W05_TB@DAL", "ou"): "win",
    ("2026_W05_BUF@LA", "wp"): "loss",
    ("2026_W05_BUF@LA", "ats"): "win",
    ("2026_W05_BUF@LA", "ou"): "push",
    ("2026_W06_SF@SEA", "wp"): "win",
    ("2026_W06_SF@SEA", "ats"): "win",
    ("2026_W06_SF@SEA", "ou"): "loss",
    ("2026_W06_MIA@NE", "wp"): "loss",
    ("2026_W06_MIA@NE", "ats"): "push",
    ("2026_W06_MIA@NE", "ou"): "win",
    (SUPER_BOWL, "wp"): "win",
    (SUPER_BOWL, "ats"): "win",
    (SUPER_BOWL, "ou"): "loss",
}

# (payout_flat, realized_units) per grade, at -110 and a 1.25-unit stake.
PAYOUTS: dict[str, tuple[float, float]] = {
    "win": (0.9090909090909091, 1.1363636363636365),
    "loss": (-1.0, -1.25),
    "push": (0.0, 0.0),
}

# When each week's bets were graded: weeks 5 and 6 before the rule's lock, week 22 after it.
GRADED_AT: dict[int, datetime] = {
    5: datetime(2026, 10, 13, 21, 0, tzinfo=UTC),
    6: datetime(2026, 10, 20, 21, 0, tzinfo=UTC),
    22: datetime(2027, 2, 15, 21, 0, tzinfo=UTC),
}

_MODEL_VALUES: dict[str, float] = {"wp": 0.6, "ats": -3.2, "ou": 44.5}
_BET_SIDES: dict[str, str] = {"wp": "home", "ats": "home", "ou": "over"}
_CLV_COLUMNS: tuple[str, ...] = ("clv", "forward_clv", "closing_line", "closing_odds")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


@pytest.fixture
def coverage_through_week_22(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extend the coverage end to week 22 for the tests whose counted bets include the Super Bowl.

    RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md: R6 coverage now ends at week 9
    (``R6_COVERAGE_END_WEEK``) and a counted live row after it refuses the run, so a test built
    on the complete fixture's week-22 bets patches the end back to 22, the pre-switch span, to keep
    proving what it proved before. The post-switch behaviour has its own tests at the end of this
    file.
    """
    monkeypatch.setattr(season_readout, "R6_COVERAGE_END_WEEK", 22)


def _week_of(game_id: str) -> int:
    """The week a standard game id names."""
    return int(game_id.split("_W")[1][:2])


def ledger_row(
    game_id: str, target: str, *, status: str = "live", arm: str = "live"
) -> dict[str, Any]:
    """One ledger row for *game_id*, stamped with the frozen pin's ids (generation 0)."""
    return make_row(
        game_id=game_id,
        week=_week_of(game_id),
        target=target,
        status=status,
        arm=arm,
        bet_side=_BET_SIDES[target],
        model_value=_MODEL_VALUES[target],
        rejection_reason=None if status == "live" else "below_ev_floor",
        model_artifact_id=SHADOW_PIN.model_ids[target],
        blend_id=SHADOW_PIN.resolved().blend,
        recipe_id=SHADOW_PIN.recipe,
    )


def grade_update(grade: str, week: int) -> dict[str, Any]:
    """The grading update that moves a bet of *week* to *grade*."""
    payout, units = PAYOUTS[grade]
    return {**graded(grade, payout, units), "graded_at": GRADED_AT[week]}


def write_ledger(
    ledger_dir: Path,
    rows: Sequence[dict[str, Any]],
    grades: Mapping[tuple[str, str], str],
) -> None:
    """Append *rows* through the real store, then grade the live-arm bets *grades* names."""
    append_rows(ledger_dir, list(rows))
    updates = {
        key_of(row): grade_update(grades[(row["game_id"], row["target"])], row["week"])
        for row in rows
        if row["arm"] == "live" and (row["game_id"], row["target"]) in grades
    }
    if updates:
        apply_updates(ledger_dir, grading_updates=updates)


def complete_rows() -> list[dict[str, Any]]:
    """The complete fixture's rows: every bet, one suppressed row and one shadow twin."""
    return [
        *(
            ledger_row(game_id, target)
            for game_id in BET_GAMES
            for target in VERDICT_TARGETS
        ),
        ledger_row(NO_BET_GAME, "wp", status="suppressed"),
        ledger_row(SUPER_BOWL, "ats", arm="shadow"),
    ]


def switch_rows() -> list[dict[str, Any]]:
    """The complete fixture's rows as they stand after the switch (MIDSEASON-SWITCH-RECORD.md):
    no row on the Super Bowl, so no counted row after the coverage's last week (9)."""
    return [row for row in complete_rows() if row["game_id"] != SUPER_BOWL]


def counted_of(ledger_dir: Path) -> pd.DataFrame:
    """The counted rows of *ledger_dir*, read by the one reader of counted rows."""
    return verdict_rows(read_entries(ledger_dir), load_verdict_scope())


def complete_counted(tmp_path: Path) -> pd.DataFrame:
    """The complete fixture's counted frame (ledger under ``tmp_path/ledger``)."""
    ledger_dir = tmp_path / "ledger"
    write_ledger(ledger_dir, complete_rows(), COMPLETE_GRADES)
    return counted_of(ledger_dir)


def _per_bet(counted: pd.DataFrame, target: str) -> pd.DataFrame:
    """The test's own selection of *target*'s bets, for an independent recomputation."""
    keep = (counted["target"] == target) & (counted["status"] == "live")
    return counted.loc[keep, ["season", "week", "payout_flat", "flat_stake"]]


# ---------------------------------------------------------------------------
# Task 1: the verdict computation
# ---------------------------------------------------------------------------


def test_decisions_use_unrounded_values(tmp_path: Path) -> None:
    """SPEC AC 25 (the decision half): the floats the verdict reads are the functions' own."""
    counted = complete_counted(tmp_path)
    no_void: frozenset[str] = frozenset()

    results = season_readout.measure_targets(counted, no_void)

    assert tuple(result.figures for result in results) == season_readout.target_figures(
        counted, no_void
    )
    assert [result.figures.target for result in results] == list(VERDICT_TARGETS)
    stats = [roi_ci_and_p(_per_bet(counted, target)) for target in VERDICT_TARGETS]
    raw = [stat["p_value"] for stat in stats]
    assert all(p is not None for p in raw)
    adjusted = benjamini_hochberg_adjusted(raw, m=BH_FAMILY_SIZE)
    for result, stat, adjusted_p in zip(results, stats, adjusted, strict=True):
        assert result.figures.flat_roi == stat["point_estimate"]
        assert result.raw_p == stat["p_value"]
        assert result.adjusted_p == adjusted_p
        assert result.n_blocks == stat["n_blocks"] == 3
        assert result.p_absent_reason is None
        assert result.token in VERDICT_TOKENS


def test_the_verdict_reads_no_clv(tmp_path: Path) -> None:
    """CLV never enters: perturbing every closing-line column changes no figure and no word."""
    counted = complete_counted(tmp_path)
    perturbed = counted.copy()
    for offset, column in enumerate(_CLV_COLUMNS, start=1):
        assert column in perturbed.columns
        perturbed[column] = [1000.0 * offset + index for index in range(len(perturbed))]

    assert season_readout.measure_targets(
        perturbed, frozenset()
    ) == season_readout.measure_targets(counted, frozenset())


def test_the_degenerate_targets_keep_the_family_at_three(tmp_path: Path) -> None:
    """The zero-bet word, the no-chain word, an absent p for one block, and m = 3 throughout."""
    # Frame A: wp suppressed only, ats absent, ou's bets all in week 5 (one block, ROI > 0).
    a_dir = tmp_path / "a"
    write_ledger(
        a_dir,
        [
            ledger_row(WEEK5_GAMES[0], "wp", status="suppressed"),
            ledger_row(WEEK6_GAMES[0], "wp", status="suppressed"),
            ledger_row(WEEK5_GAMES[0], "ou"),
            ledger_row(WEEK5_GAMES[1], "ou"),
        ],
        {(WEEK5_GAMES[0], "ou"): "win", (WEEK5_GAMES[1], "ou"): "push"},
    )
    wp, ats, ou = season_readout.measure_targets(counted_of(a_dir), frozenset())

    assert (wp.figures.counted_rows, wp.figures.bets, wp.token) == (2, 0, _NO_BETS)
    assert (ats.figures.counted_rows, ats.token) == (0, _NO_CHAIN)
    one_block = roi_ci_and_p(_per_bet(counted_of(a_dir), "ou"))
    assert (ou.raw_p, ou.adjusted_p, ou.n_blocks) == (None, None, 1)
    assert ou.p_absent_reason == one_block["p_value_absent_reason"] is not None
    assert ou.figures.flat_roi is not None and ou.figures.flat_roi > 0
    assert ou.token == _INCONCLUSIVE

    # Frame B: wp bets across three blocks, ats in one block, ou suppressed only.
    b_dir = tmp_path / "b"
    b_rows = [
        *(
            ledger_row(game_id, "wp")
            for game_id in (WEEK5_GAMES[0], WEEK6_GAMES[0], SUPER_BOWL)
        ),
        ledger_row(WEEK5_GAMES[0], "ats"),
        ledger_row(WEEK5_GAMES[1], "ats"),
        ledger_row(WEEK6_GAMES[1], "ou", status="suppressed"),
    ]
    write_ledger(
        b_dir,
        b_rows,
        {
            (WEEK5_GAMES[0], "wp"): "win",
            (WEEK6_GAMES[0], "wp"): "loss",
            (SUPER_BOWL, "wp"): "win",
            (WEEK5_GAMES[0], "ats"): "win",
            (WEEK5_GAMES[1], "ats"): "loss",
        },
    )
    wp, ats, ou = season_readout.measure_targets(counted_of(b_dir), frozenset())

    assert wp.raw_p is not None and wp.n_blocks == 3
    assert wp.adjusted_p == benjamini_hochberg_adjusted([wp.raw_p], m=3)[0]
    assert wp.adjusted_p == min(1.0, 3 * wp.raw_p)
    assert (ats.raw_p, ats.adjusted_p, ats.n_blocks) == (None, None, 1)
    assert ats.figures.flat_roi is not None and ats.figures.flat_roi < 0
    assert ats.token == _UNPROFITABLE  # a non-positive return needs no p-value
    assert ou.token == _NO_BETS


# ---------------------------------------------------------------------------
# Binding fixture builders: the silver schedule, the evidence files, a synthetic repository
# ---------------------------------------------------------------------------


def schedule_game(game_id: str, *, scored: bool = True) -> dict[str, Any]:
    """One silver ``games`` row; the week comes from the standard game id."""
    week = _week_of(game_id)
    return {
        "game_id": game_id,
        "season": 2026,
        "week": week,
        "kickoff_et": pd.Timestamp(KICKOFF_BY_WEEK[week]),
        "game_type": "SB" if week == 22 else "REG",
        "home_score": 24.0 if scored else None,
        "away_score": 17.0 if scored else None,
    }


def complete_games() -> list[dict[str, Any]]:
    """Every game of the complete fixture, all scored (the missed game included)."""
    return [
        schedule_game(game_id) for game_id in (*BET_GAMES, NO_BET_GAME, MISSED_GAME)
    ]


def _jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def git(repo: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run one git command in *repo* (raising on failure) and return its stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env=None if env is None else {**os.environ, **env},
    )
    return completed.stdout


def synthetic_repo(root: Path) -> Path:
    """A throwaway repository whose one commit adds copies of both R6 record files (the lock)."""
    repo = root / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "binding@example.invalid")
    git(repo, "config", "user.name", "Binding Fixture")
    for name in RULE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REAL_REPO_ROOT / name).read_bytes())
    git(repo, "add", *RULE_FILES)
    dated = {
        "GIT_AUTHOR_DATE": LOCK_COMMITTED_AT,
        "GIT_COMMITTER_DATE": LOCK_COMMITTED_AT,
    }
    git(repo, "commit", "-q", "-m", "freeze the 2026 verdict rule", env=dated)
    return repo


def anchor_ledger(repo: Path, ledger_dir: Path) -> None:
    """Publish *ledger_dir*'s current head on *repo*'s ``ledger-anchor`` branch, as a sync does."""
    head, count = ledger_head(read_entries(ledger_dir))
    write_anchor_commit(repo, head, count)


def binding_fixture(
    root: Path,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    grades: Mapping[tuple[str, str], str] = COMPLETE_GRADES,
    games: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Write a complete binding fixture under *root*; return the binding argv that reads it.

    The synthetic repository is ``root / "repo"``; :func:`_run` makes it the run's
    ``REPO_ROOT``, the stand-in for the executing checkout (there is no repository option).

    By default: the complete ledger (every bet graded, one suppressed row, one shadow twin), a
    schedule holding every game scored (plus one missed game), a run log, empty skip records and
    a synthetic repository holding the R6 lock commit and the ledger's published anchor.
    """
    ledger_dir, silver_dir = root / "ledger", root / "silver"
    silver_dir.mkdir(parents=True)
    write_ledger(ledger_dir, complete_rows() if rows is None else rows, grades)
    schedule = complete_games() if games is None else list(games)
    pd.DataFrame(schedule).to_parquet(silver_dir / SILVER_GAMES_FILENAME)
    run_log, skip_records = root / "daily_lock_runs.jsonl", root / "skip_records.jsonl"
    no_games = {
        "game_ids": [],
        "outcome": "no_games",
        "recorded_at": "2026-10-10T21:00:00+00:00",
        "run_date_et": "2026-10-10",
    }
    _jsonl(run_log, [no_games])
    _jsonl(skip_records, [])
    repo = synthetic_repo(root)
    anchor_ledger(repo, ledger_dir)
    return [
        "--binding",
        "--ledger-dir",
        str(ledger_dir),
        "--silver-dir",
        str(silver_dir),
        "--run-log",
        str(run_log),
        "--skip-records",
        str(skip_records),
    ]


def _arg(argv: Sequence[str], name: str) -> Path:
    """The path the option *name* carries in *argv*."""
    return Path(argv[list(argv).index(name) + 1])


def _fixture_lock(repo: Path) -> str:
    """*repo*'s own lock commit, the witness's stand-in there; the real witness when it has none."""
    try:
        added = git(
            repo, "log", "--diff-filter=A", "--format=%H", "--", RULE_FILES[1]
        ).split()
    except (OSError, subprocess.CalledProcessError):
        added = []
    return added[-1] if added else season_readout.WITNESSED_LOCK_SHA


def _run(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
    repo: Path,
    witness: str | None = None,
) -> tuple[int, list[str]]:
    """Run the CLI with *repo* standing in for the executing checkout (``REPO_ROOT``).

    The witnessed lock is *repo*'s own lock commit unless *witness* names another.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(season_readout, "REPO_ROOT", repo)
        patch.setattr(
            season_readout, "WITNESSED_LOCK_SHA", witness or _fixture_lock(repo)
        )
        code = season_readout.main(argv)
    return code, capsys.readouterr().out.splitlines()


def _table_row(document: str, target: str) -> list[str]:
    """The cells of section 3's table row for *target*."""
    (row,) = [
        line for line in document.splitlines() if line.startswith(f"| {target} |")
    ]
    return [cell.strip() for cell in row.strip("|").split("|")]


SECTION_HEADINGS: tuple[str, ...] = (
    "## 1. Coverage",
    "## 2. Verdict scope",
    "## 3. The verdict per target",
    "## 4. The rule's lock",
    "## 5. Re-fit comparison (ROLLING-REFIT-PREREGISTRATION.md section 11)",
    "## 6. What this readout read",
)


# ---------------------------------------------------------------------------
# Task 2: the binding run writes its one document
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_a_binding_run_writes_the_document_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SPEC AC 23, 24, 25: six sections in order, figures equal an independent recomputation."""
    argv = binding_fixture(tmp_path)
    repo = tmp_path / "repo"
    ledger_dir, silver_dir = _arg(argv, "--ledger-dir"), _arg(argv, "--silver-dir")

    code, lines = _run(argv, capsys, repo)

    assert code == 0, lines
    record = repo / BINDING_RECORD_PATH
    assert lines[0] == f"BINDING_WRITTEN= {record.as_posix()}"
    raw = record.read_bytes()
    assert b"\r\n" not in raw
    document = raw.decode("ascii")
    positions = [document.index(heading) for heading in SECTION_HEADINGS]
    assert positions == sorted(positions)
    assert "Missed games: 1\n" in document
    assert "- Playoff weeks: counted (D40-08)\n" in document

    # Section 3 against a hand recomputation (bets, wins, units, flat ROI) and against the
    # reused statistic, correction and word functions called here directly.
    counted = counted_of(ledger_dir)
    stats = {
        target: roi_ci_and_p(_per_bet(counted, target)) for target in VERDICT_TARGETS
    }
    raws = [float(stats[target]["p_value"]) for target in VERDICT_TARGETS]
    adjusted = dict(
        zip(VERDICT_TARGETS, benjamini_hochberg_adjusted(raws, m=3), strict=True)
    )
    results = {
        result.figures.target: result
        for result in season_readout.measure_targets(counted, frozenset())
    }
    for target in VERDICT_TARGETS:
        target_grades = [COMPLETE_GRADES[(game_id, target)] for game_id in BET_GAMES]
        units = sum(PAYOUTS[grade][1] for grade in target_grades)
        # Flat stake 1.0 per bet, so the ROI is the mean payout.
        roi = sum(PAYOUTS[grade][0] for grade in target_grades) / len(BET_GAMES)
        stat = stats[target]
        token, _ = assign_verdict_token(
            TargetMeasurement(
                target=target,
                chain_resolved=True,
                control_passed=True,
                control_bet_count=0,
                bet_count=len(BET_GAMES),
                flat_roi=stat["point_estimate"],
                adjusted_p=adjusted[target],
                raw_p=stat["p_value"],
                p_absent_reason=None,
                n_blocks=stat["n_blocks"],
                n_push=target_grades.count("push"),
                n_ungraded=0,
                fallback_fired=False,
                fallback_trigger=None,
            ),
            alpha=ALPHA,
            min_attainable_p=ROI_MIN_ATTAINABLE_P,
        )
        assert _table_row(document, target) == [
            target,
            str(len(BET_GAMES)),  # the shadow twin and the suppressed row are no bet
            str(target_grades.count("win")),
            str(target_grades.count("loss")),
            str(target_grades.count("push")),
            f"{units:.6f}",
            f"{roi:.6f}",
            f"{stat['p_value']:.6g}",
            f"{adjusted[target]:.6g}",
            token,
        ]
        assert f"VERDICT_{target.upper()}= {token}" in lines
        # Decided on the unrounded float, shown rounded.
        unrounded = results[target].figures.flat_roi
        assert unrounded == stat["point_estimate"]
        assert repr(unrounded) not in document

    # Section 4: the lock commit, its time and the weeks graded by then.
    lock_sha = git(repo, "rev-parse", "HEAD").strip()
    assert f"- Lock commit: {lock_sha}\n" in document
    assert f"- Committed at: {LOCK_COMMITTED_AT}\n" in document
    assert "- Weeks with a counted bet graded at or before the lock: 5, 6\n" in document

    # Section 5: the comparison command's own stdout, byte for byte.
    refit_argv = ["--ledger-dir", str(ledger_dir), "--silver-dir", str(silver_dir)]
    assert compare_refit_arms.main(refit_argv) == 0
    refit_stdout = capsys.readouterr().out
    assert "```text\n" + refit_stdout + "```" in document

    # Section 6: the ledger head and entry count read.
    head, entries = ledger_head(read_entries(ledger_dir))
    assert f"- Ledger chain head: {head}\n" in document
    assert f"- Ledger entries: {entries}\n" in document
    # This fixture's silver lies outside the repository, so the document names it as given; a
    # store inside it is named repo-relative (test_the_document_names_repo_relative_paths).
    assert f"- Silver directory: {silver_dir.as_posix()}\n" in document


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_a_void_bet_leaves_the_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-20: a bet on a game never played is listed by name, in no figure, and never blocks."""
    void_game = "2026_W06_NYJ@BUF"  # unscored while week 22 is scored: not played
    base = binding_fixture(tmp_path / "base")
    code, _ = _run(base, capsys, tmp_path / "base" / "repo")
    assert code == 0
    base_document = (tmp_path / "base" / "repo" / BINDING_RECORD_PATH).read_text(
        "ascii"
    )

    argv = binding_fixture(
        tmp_path / "void",
        rows=[*complete_rows(), ledger_row(void_game, "wp")],
        games=[*complete_games(), schedule_game(void_game, scored=False)],
    )
    code, lines = _run(argv, capsys, tmp_path / "void" / "repo")

    assert code == 0, lines
    document = (tmp_path / "void" / "repo" / BINDING_RECORD_PATH).read_text("ascii")
    void_section = document.split("Void bets (D-20")[1].split("## 2.")[0]
    assert f"- {void_game} wp\n" in void_section
    assert (
        f"- {void_game} (week 6): inferred: no final score while week 22 is scored\n"
        in document
    )
    assert _table_row(document, "wp") == _table_row(base_document, "wp")
    assert _table_row(document, "wp")[1] == str(len(BET_GAMES))


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_every_row_of_a_not_played_game_leaves_the_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-A02: a not-played game's rows leave every figure whether or not a target bet there.

    On G a WP bet sits beside an ATS suppressed row; on H the ATS suppressed row is alone.
    Both games are not played, so neither ATS row counts.
    """
    with_bet, alone = "2026_W06_NYJ@BUF", "2026_W06_CAR@ATL"
    rows = [
        *complete_rows(),
        ledger_row(with_bet, "wp"),
        ledger_row(with_bet, "ats", status="suppressed"),
        ledger_row(alone, "ats", status="suppressed"),
    ]
    games = [
        *complete_games(),
        schedule_game(with_bet, scored=False),
        schedule_game(alone, scored=False),
    ]
    argv = binding_fixture(tmp_path, rows=rows, games=games)
    preview = [arg for arg in argv if arg != "--binding"]

    code, lines = _run(preview, capsys, tmp_path / "repo")

    assert code == 0, lines
    assert (
        f"NOT_PLAYED= {alone} week=6 reason=inferred: no final score while week 22 is scored"
        in lines
    )
    figures = {line.split()[1]: line for line in lines if line.startswith("FIGURES= ")}
    assert (
        f"counted_rows={len(BET_GAMES)} bets={len(BET_GAMES)} " in figures["target=ats"]
    )
    assert (
        f"counted_rows={len(BET_GAMES) + 1} bets={len(BET_GAMES)} "
        in figures["target=wp"]
    )


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_after_an_honest_invalidation_the_lock_section_reports_the_applied_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-A04: the applied rule was frozen at the INVALIDATES commit, so its instant and the
    weeks graded by then are reported beside the original lock's."""
    argv = binding_fixture(tmp_path)
    repo = tmp_path / "repo"
    lock_sha = git(repo, "rev-parse", "HEAD").strip()
    with (repo / RULE_FILES[1]).open("a", encoding="utf-8") as handle:
        handle.write("\n# corrected\n")
    frozen_at = "2027-02-16T12:00:00+00:00"  # after week 22 was graded
    message = f"correct the rule\n\nThis commit INVALIDATES the pre-registration at {lock_sha}."
    git(
        repo,
        "commit",
        "-q",
        "-am",
        message,
        env={"GIT_AUTHOR_DATE": frozen_at, "GIT_COMMITTER_DATE": frozen_at},
    )
    frozen_sha, frozen_iso = git(repo, "log", "-1", "--format=%H %cI").split()

    code, lines = _run(argv, capsys, repo)

    assert code == 0, lines
    document = (repo / BINDING_RECORD_PATH).read_text("ascii")
    assert f"- Lock commit: {lock_sha}\n" in document
    assert "- Weeks with a counted bet graded at or before the lock: 5, 6\n" in document
    assert (
        f"- The applied rule was frozen at: {frozen_sha}, committed at {frozen_iso}\n"
        in document
    )
    assert (
        "- Weeks with a counted bet graded at or before the applied rule was frozen: "
        "5, 6, 22\n" in document
    )


# ---------------------------------------------------------------------------
# Task 3: binding refuses until the season is graded, refuses an edited rule, and runs once
# ---------------------------------------------------------------------------

REFUSED = "BINDING_REFUSED= "


def _tree(root: Path) -> dict[str, int]:
    """Every file under *root* with its size."""
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def _refusal(lines: list[str]) -> str:
    """The one output line, asserted to be a refusal."""
    assert len(lines) == 1, lines
    assert lines[0].startswith(REFUSED), lines
    return lines[0]


def _assert_refuses(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
    root: Path,
    witness: str | None = None,
) -> str:
    """Run binding on *argv* against ``root / "repo"``: exit 3, one refusal line, and nothing
    new or changed under *root*."""
    before = _tree(root)
    code, lines = _run(argv, capsys, root / "repo", witness)
    assert code == 3, lines
    assert _tree(root) == before
    assert not (root / "repo" / BINDING_RECORD_PATH).exists()
    return _refusal(lines)


@pytest.mark.usefixtures("isolated_git_env")
def test_binding_refuses_on_todays_real_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SPEC AC 27, 28: the real ledger refuses, and its bytes do not move.

    It is judged against a COPY of the real schedule whose week-22 scores are blanked (WR-A08).
    While weeks 5-9 are unfinished, refusal 1 or 2 fires. Once they are graded and scored,
    refusal 3 fires, and after the real Super Bowl the copy's blanked week-22 game keeps it
    firing. Judged against the real schedule, the test would turn red on the day the binding run
    is due.

    RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md: coverage ends at week 9, so the blanked
    week-22 game is outside it and no longer trips refusal 2; refusal 3 ("no played week-22 game")
    is the honest answer from the end of week 9 until February, so it is accepted too.
    """
    ledger_file = REAL_REPO_ROOT / ledger_path(LEDGER_DIR)
    real_schedule = REAL_REPO_ROOT / DEFAULT_SILVER_DIR / SILVER_GAMES_FILENAME
    for path in (ledger_file, real_schedule):
        if not path.exists():
            pytest.skip(f"the real store {path.as_posix()} is absent on this machine")
    silver_copy = tmp_path / "silver"
    silver_copy.mkdir()
    schedule = pd.read_parquet(real_schedule)
    scope = load_verdict_scope()
    final_week = (schedule["season"] == scope.season) & (
        schedule["week"] == scope.end_week
    )
    schedule.loc[final_week, ["home_score", "away_score"]] = None
    schedule.to_parquet(silver_copy / SILVER_GAMES_FILENAME)
    stores = {
        "--ledger-dir": REAL_REPO_ROOT / LEDGER_DIR,
        "--silver-dir": silver_copy,
        "--run-log": REAL_REPO_ROOT / season_readout.DEFAULT_RUN_LOG,
        "--skip-records": REAL_REPO_ROOT / SKIP_RECORD_PATH,
    }
    repo = synthetic_repo(tmp_path)
    # The anchor is only READ from the real ledger.
    anchor_ledger(repo, stores["--ledger-dir"])
    before = hashlib.sha256(ledger_file.read_bytes()).hexdigest()
    argv = ["--binding"]
    for name, path in stores.items():
        argv += [name, str(path)]

    line = _assert_refuses(argv, capsys, tmp_path)

    assert (
        "counted bet(s) of played games are not graded yet" in line
        or "in-scope game(s) lack a final score" in line
        or "the schedule holds no played week-22 game" in line
    ), line
    assert hashlib.sha256(ledger_file.read_bytes()).hexdigest() == before


@pytest.mark.usefixtures("isolated_git_env")
def test_binding_refuses_an_absent_a_stale_or_an_unanchored_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refusal 0 (CR-A01): an absent or shortened ledger still verifies as a chain, so the
    season-end run must refuse it rather than spend its one shot on it."""
    # An absent ledger at season end: every other condition holds.
    absent = binding_fixture(tmp_path / "absent")
    ledger_path(_arg(absent, "--ledger-dir")).unlink()
    line = _assert_refuses(absent, capsys, tmp_path / "absent")
    assert "the ledger holds no entry" in line

    # A stale backup: one entry shorter than the published anchor.
    stale = binding_fixture(tmp_path / "stale")
    ledger_file = ledger_path(_arg(stale, "--ledger-dir"))
    kept = ledger_file.read_bytes().splitlines(keepends=True)[:-1]
    ledger_file.write_bytes(b"".join(kept))
    line = _assert_refuses(stale, capsys, tmp_path / "stale")
    assert "the ledger does not extend its published anchor" in line

    # No anchor at all while the ledger holds entries.
    unanchored = binding_fixture(tmp_path / "unanchored")
    git(
        tmp_path / "unanchored" / "repo", "update-ref", "-d", "refs/heads/ledger-anchor"
    )
    line = _assert_refuses(unanchored, capsys, tmp_path / "unanchored")
    assert "holds no ledger-anchor commit" in line


@pytest.mark.usefixtures("isolated_git_env")
def test_binding_refuses_without_a_played_week_22_game(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Super Bowl is REMOVED from the schedule and the ledger.

    RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md: the assertion is unchanged; only the
    reason given for removing the game moved. Before the switch an unscored week-22 game tripped
    refusal 2 first (the last coverage week is never 'not played'); coverage now ends at week 9,
    so week 22 is outside it, and refusal 3 alone waits for the Super Bowl.
    """
    argv = binding_fixture(
        tmp_path,
        rows=[row for row in complete_rows() if row["game_id"] != SUPER_BOWL],
        games=[game for game in complete_games() if game["game_id"] != SUPER_BOWL],
    )

    line = _assert_refuses(argv, capsys, tmp_path)

    assert "the schedule holds no played week-22 game" in line


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_a_graded_bet_on_a_game_inferred_not_played(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-A03: a week-6 score that vanished from silver would make its game "not played" and
    void its already-graded bet; the grade proves the score existed, so the run refuses."""
    lost_score = WEEK6_GAMES[0]
    games = [
        schedule_game(game["game_id"], scored=game["game_id"] != lost_score)
        for game in complete_games()
    ]
    argv = binding_fixture(tmp_path, games=games)

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line == (
        f"{REFUSED}1 game(s) inferred not played carry a graded counted bet: {lost_score}. "
        "A grade means a final score existed, so silver lost it; restore the schedule's score "
        "instead of waiting (voiding it would drop a settled result from the verdict)."
    )


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_while_a_counted_bet_is_ungraded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-24: one pending bet of a scored game refuses; the suppressed pending row never does."""
    ungraded = (WEEK6_GAMES[0], "ats")
    grades = {key: grade for key, grade in COMPLETE_GRADES.items() if key != ungraded}
    argv = binding_fixture(tmp_path, grades=grades)
    # The fixture also holds a suppressed row, which is never graded.
    suppressed = counted_of(_arg(argv, "--ledger-dir"))
    assert (suppressed["status"] == "suppressed").sum() == 1

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line == (
        f"{REFUSED}1 counted bet(s) of played games are not graded yet: "
        f"{ungraded[0]} {ungraded[1]}. Wait for the next 17:00 ET daily run to grade them."
    )


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_a_second_binding_run_refuses_even_after_the_record_is_deleted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-09: once written, never again -- not while the record exists, not after it is deleted,
    and not at a path the caller chooses (there is no such option)."""
    argv = binding_fixture(tmp_path)
    repo = tmp_path / "repo"
    record = repo / BINDING_RECORD_PATH
    code, _ = _run(argv, capsys, repo)
    assert code == 0 and record.exists()

    # The record exists: refused, and nothing anywhere under tmp_path changes.
    before = _tree(tmp_path)
    code, lines = _run(argv, capsys, repo)
    assert code == 3
    assert "already exists; the binding run happened" in _refusal(lines)
    assert _tree(tmp_path) == before

    # Committed, then deleted: git history still refuses it.
    git(repo, "add", BINDING_RECORD_PATH)
    git(repo, "commit", "-q", "-m", "the 2026 season-close readout")
    record.unlink()
    line = _assert_refuses(argv, capsys, tmp_path)
    assert f"{BINDING_RECORD_PATH} appears in git history" in line

    # A second run cannot be pointed at a fresh path, or at a clone lacking the record: neither
    # option exists, so the checked repository is always the executing one (WR-A01).
    with pytest.raises(SystemExit):
        season_readout.main([*argv, "--record-path", str(tmp_path / "elsewhere.md")])
    assert not (tmp_path / "elsewhere.md").exists()
    with pytest.raises(SystemExit):
        season_readout.main([*argv, "--repo-root", str(tmp_path / "another_clone")])


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_when_the_rule_moved(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusals 6 and 7: the rule binding reads must be the rule that was locked."""
    module = "backtest/verdict_test_2026.py"
    quiet_edit = "\n# a quiet edit after the lock\n"

    # (a) A later commit edits the record without naming the lock INVALIDATED.
    committed = binding_fixture(tmp_path / "committed")
    repo = tmp_path / "committed" / "repo"
    with (repo / module).open("a", encoding="utf-8") as handle:
        handle.write(quiet_edit)
    git(repo, "commit", "-q", "-am", "tidy the rule")
    quiet_sha = git(repo, "rev-parse", "HEAD").strip()
    line = _assert_refuses(committed, capsys, tmp_path / "committed")
    assert "the rule moved after its lock" in line
    assert f"{quiet_sha} tidy the rule" in line

    # (a) The same edit left uncommitted.
    uncommitted = binding_fixture(tmp_path / "uncommitted")
    repo = tmp_path / "uncommitted" / "repo"
    with (repo / module).open("a", encoding="utf-8") as handle:
        handle.write(quiet_edit)
    line = _assert_refuses(uncommitted, capsys, tmp_path / "uncommitted")
    assert f"uncommitted {module}" in line

    # (a) An honest commit naming the lock INVALIDATED is not a quiet edit.
    honest_repo = synthetic_repo(tmp_path / "honest")
    lock = season_readout.lock_commit(honest_repo)
    with (honest_repo / module).open("a", encoding="utf-8") as handle:
        handle.write("\n# corrected\n")
    message = f"correct the rule\n\nThis commit INVALIDATES the pre-registration at {lock.sha}."
    git(honest_repo, "commit", "-q", "-am", message)
    assert season_readout.quiet_rule_edits(honest_repo, lock.sha) == []
    assert season_readout.lock_commit(honest_repo).sha == lock.sha

    # (b) A frozen value moved upstream of the record.
    moved = binding_fixture(tmp_path / "moved")
    monkeypatch.setattr(backtest.verdict_test_2026, "ALPHA", 0.1)
    line = _assert_refuses(moved, capsys, tmp_path / "moved")
    assert (
        line == f"{REFUSED}the frozen value ALPHA is pinned at 0.05 but now holds 0.1."
    )


def test_the_witnessed_lock_is_the_recorded_witness() -> None:
    """The CLI's copy of the lock sha is the Plan 38-03 witness, and the real lock proves out."""
    witness = phase38_state.P38_PREREGISTRATION_WITNESS
    assert {witness[name]["commit"] for name in RULE_FILES} == {
        season_readout.WITNESSED_LOCK_SHA
    }
    lock = season_readout.lock_commit(REAL_REPO_ROOT)
    assert lock.sha == season_readout.WITNESSED_LOCK_SHA
    assert lock.committed_at == phase38_state.P38_PREREGISTRATION_COMMITTED_AT
    assert season_readout.unwitnessed_lock_problem(REAL_REPO_ROOT, lock) is None


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_a_lock_it_cannot_prove(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal 5 (CR-A02): a shallow clone, or a history whose "lock" is not the witnessed
    commit, refuses instead of writing the record under a wrong lock."""
    # A shallow clone: its boundary commit reads as adding every file.
    argv = binding_fixture(tmp_path / "origin")
    origin = tmp_path / "origin" / "repo"
    (origin / "notes.txt").write_text("a later commit\n", encoding="utf-8")
    git(origin, "add", "notes.txt")
    git(origin, "commit", "-q", "-m", "a later commit")
    clone = tmp_path / "shallow" / "repo"
    git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository").strip() == "true"
    anchor_ledger(clone, _arg(argv, "--ledger-dir"))
    line = _assert_refuses(argv, capsys, tmp_path / "shallow", _fixture_lock(origin))
    assert "the R6 lock commit cannot be found" in line and "shallow clone" in line

    # Rewritten history: the oldest commit adding the rule is not the witnessed one.
    line = _assert_refuses(argv, capsys, tmp_path / "origin", "f" * 40)
    assert "the R6 lock commit cannot be proven" in line
    assert "not the witnessed lock" in line

    # A "lock" committed after the deadline, or touching a third file, is not the lock either.
    late = synthetic_repo(tmp_path / "late")
    (late / "extra.txt").write_text("not part of the record\n", encoding="utf-8")
    git(late, "add", "extra.txt")
    after_deadline = "2026-10-29T12:00:00+00:00"
    git(
        late,
        "commit",
        "-q",
        "--amend",
        "--no-edit",
        env={"GIT_AUTHOR_DATE": after_deadline, "GIT_COMMITTER_DATE": after_deadline},
    )
    lock = season_readout.lock_commit(late)
    monkeypatch.setattr(season_readout, "WITNESSED_LOCK_SHA", lock.sha)
    problem = season_readout.unwitnessed_lock_problem(late, lock)
    assert problem is not None and "not before the pre-registration deadline" in problem
    git(
        late,
        "commit",
        "-q",
        "--amend",
        "--no-edit",
        env={
            "GIT_AUTHOR_DATE": LOCK_COMMITTED_AT,
            "GIT_COMMITTER_DATE": LOCK_COMMITTED_AT,
        },
    )
    lock = season_readout.lock_commit(late)
    monkeypatch.setattr(season_readout, "WITNESSED_LOCK_SHA", lock.sha)
    problem = season_readout.unwitnessed_lock_problem(late, lock)
    assert problem is not None and "not exactly the two R6 files" in problem


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_reordered_verdict_words(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR-A05: the 2026 meanings are zipped onto VERDICT_TOKENS by position, so a reorder
    upstream would print every meaning beside the wrong word; refusal 7 names it."""
    argv = binding_fixture(tmp_path)
    monkeypatch.setattr(
        backtest.verdict_test_2026, "VERDICT_TOKENS", tuple(reversed(VERDICT_TOKENS))
    )

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line.startswith(f"{REFUSED}the frozen value VERDICT_TOKENS is pinned at ")


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_fails_closed_on_every_git_branch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR-A09: no rule in the repository refuses at 5, a directory that is not a repository
    refuses at 0, and a git failure at refusal 4, 5 or 6 refuses rather than guessing."""
    argv = binding_fixture(tmp_path / "base")
    ledger_dir = _arg(argv, "--ledger-dir")

    # A repository holding the anchor but no commit that added the rule.
    no_rule = tmp_path / "no_rule" / "repo"
    no_rule.mkdir(parents=True)
    git(no_rule, "init", "-q", "-b", "main")
    git(no_rule, "config", "user.email", "binding@example.invalid")
    git(no_rule, "config", "user.name", "Binding Fixture")
    (no_rule / "README.md").write_text("no rule here\n", encoding="utf-8")
    git(no_rule, "add", "README.md")
    git(no_rule, "commit", "-q", "-m", "no rule")
    anchor_ledger(no_rule, ledger_dir)
    line = _assert_refuses(argv, capsys, tmp_path / "no_rule")
    assert line.startswith(
        f"{REFUSED}the R6 lock commit cannot be found: no commit in "
    )

    # A directory that is not a git repository: the first git read fails, closed.
    (tmp_path / "plain" / "repo").mkdir(parents=True)
    line = _assert_refuses(argv, capsys, tmp_path / "plain")
    assert "a git check failed, so the run refuses (fail closed)" in line

    # A git failure inside refusal 4 (history), 5 (shallow check) and 6 (status).
    real_git = season_readout._git
    for failing in ("--all", "--is-shallow-repository", "status"):

        def flaky(repo_root: Path, *args: str, _failing: str = failing) -> str:
            if _failing in args:
                raise season_readout.GitCheckError(f"planted failure of {_failing}")
            return real_git(repo_root, *args)

        monkeypatch.setattr(season_readout, "_git", flaky)
        line = _assert_refuses(argv, capsys, tmp_path / "base")
        assert line == (
            f"{REFUSED}a git check failed, so the run refuses (fail closed): planted "
            f"failure of {failing}"
        ), line
    monkeypatch.setattr(season_readout, "_git", real_git)


@pytest.mark.usefixtures("isolated_git_env")
def test_an_unscored_last_coverage_week_game_waits_only_until_a_later_week_is_scored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-A09, refusal 2: an unscored game of the coverage's last week refuses while nothing later
    is scored, even beside a scored game of its week; once a later week of the season is scored
    it is inferred not played, its bet is void, and it never blocks the run.

    RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md: coverage ends at week 9, but the
    not-played inference reads every later week of the season, so a cancelled week-9 game is
    voided once week 10 is scored instead of blocking the February run. (Before the switch this
    test pinned an unscored week-22 game refusing at refusal 2.) The ledger is the post-switch
    one plus a bet on the unscored game; the Super Bowl is unscored, so nothing later than week
    9 is scored until the week-10 game is. The weeks are read from the record.
    """
    unscored = f"2026_W{LAST_COVERAGE_WEEK:02d}_BUF@SF"
    scored = f"2026_W{LAST_COVERAGE_WEEK:02d}_NYJ@MIA"
    later = f"2026_W{SWITCH_WEEK:02d}_KC@DEN"
    rows = [*switch_rows(), ledger_row(unscored, "wp")]
    waiting_games = [
        schedule_game(game["game_id"], scored=game["game_id"] != SUPER_BOWL)
        for game in complete_games()
    ]
    waiting_games += [schedule_game(scored), schedule_game(unscored, scored=False)]

    # Nothing later than the coverage's last week is scored: refusal 2 waits for the score.
    waiting = binding_fixture(tmp_path / "waiting", rows=rows, games=waiting_games)
    line = _assert_refuses(waiting, capsys, tmp_path / "waiting")
    assert line == (
        f"{REFUSED}1 in-scope game(s) lack a final score: {unscored}. "
        "Wait for the next 17:00 ET daily run to record the scores."
    )

    # A switch-week game is scored: the last coverage week's game is not played, so refusal 2
    # passes and the run stops only at refusal 3, waiting for the Super Bowl.
    switch_week = binding_fixture(
        tmp_path / "switch_week",
        rows=rows,
        games=[*waiting_games, schedule_game(later)],
    )
    line = _assert_refuses(switch_week, capsys, tmp_path / "switch_week")
    assert "the schedule holds no played week-22 game" in line
    preview = [arg for arg in switch_week if arg != "--binding"]
    code, lines = _run(preview, capsys, tmp_path / "switch_week" / "repo")
    assert code == 0, lines
    assert (
        f"NOT_PLAYED= {unscored} week={LAST_COVERAGE_WEEK} reason=inferred: no final score "
        f"while week {SWITCH_WEEK} is scored" in lines
    )
    assert f"VOID_BETS= {unscored} target=wp" in lines


@pytest.mark.usefixtures("isolated_git_env", "coverage_through_week_22")
def test_binding_refuses_when_the_comparison_cannot_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refusal 8: a malformed re-fit date record makes the comparison exit 2."""
    argv = binding_fixture(tmp_path)
    dates = refit_dir(_arg(argv, "--ledger-dir")) / DATES_FILENAME
    dates.parent.mkdir(parents=True, exist_ok=True)
    dates.write_text("not json\n", encoding="utf-8")

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line.startswith(f"{REFUSED}the re-fit comparison could not run (exit 2): ")
    assert "STORE_UNREADABLE= " in line


# ---------------------------------------------------------------------------
# MIDSEASON-SWITCH-RECORD.md section 5: coverage ends at week 9; IN-A03; repo-relative paths
# ---------------------------------------------------------------------------

# A switch-week game: after the switch, the day-before system decides nothing there.
AFTER_SWITCH_GAME: str = f"2026_W{SWITCH_WEEK:02d}_NYJ@MIA"


def _coverage_text(document: str) -> str:
    """Section 1 of the binding document, heading excluded."""
    return document.partition("## 1. Coverage")[2].partition("## 2. ")[0]


def _coverage_weeks(document: str) -> list[int]:
    """The week numbers of section 1's per-week table rows, in order."""
    first_cells = [
        line.strip("|").partition("|")[0].strip()
        for line in _coverage_text(document).splitlines()
        if line.startswith("| ")
    ]
    return [int(cell) for cell in first_cells if cell.isdigit()]


@pytest.mark.usefixtures("isolated_git_env")
def test_coverage_ends_at_the_last_week_before_the_switch_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table ends at week 9 and cites the record, so the scheduled Super Bowl, which no live
    row decides, is not missed; the declared scope is still reported as weeks 5-22."""
    argv = binding_fixture(tmp_path, rows=switch_rows())
    repo = tmp_path / "repo"
    record = repo / BINDING_RECORD_PATH

    code, lines = _run(argv, capsys, repo)

    assert code == 0, lines
    assert lines[:2] == [
        f"BINDING_WRITTEN= {record.as_posix()}",
        f"COVERAGE_END= week={LAST_COVERAGE_WEEK} reason=MIDSEASON-SWITCH-RECORD.md",
    ]
    document = record.read_text("ascii")
    coverage = _coverage_text(document)
    assert _coverage_weeks(document) == list(range(5, LAST_COVERAGE_WEEK + 1))
    assert (
        f"Coverage ends at week {LAST_COVERAGE_WEEK}: under MIDSEASON-SWITCH-RECORD.md the "
        f"day-before system stopped deciding at week {SWITCH_WEEK}, so weeks {SWITCH_WEEK}-22 "
        "are not reported and none of their games is missed.\n" in coverage
    )
    # MISSED_GAME alone: the unbet Super Bowl would be the second miss under the old span.
    assert "Missed games: 1\n" in coverage
    assert "- Weeks: 5-22\n" in document


@pytest.mark.usefixtures("isolated_git_env")
@pytest.mark.parametrize("status", ["live", "suppressed"])
def test_a_counted_live_row_after_the_switch_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str
) -> None:
    """Record section 5: a counted live row in week 10 or later means the retirement leaked.

    "Counted live row" is any live-arm row the verdict keeps, a bet or a no-bet alike: from week
    10 the day-before system records no decision of either kind. The week-10 bet is ungraded on a
    scored game, so it would trip refusal 1 ("wait") if the switch check came later; waiting never
    clears this one, so it comes first.
    """
    argv = binding_fixture(
        tmp_path,
        rows=[*switch_rows(), ledger_row(AFTER_SWITCH_GAME, "wp", status=status)],
        games=[*complete_games(), schedule_game(AFTER_SWITCH_GAME)],
    )

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line == (
        f"{REFUSED}a counted live row in week {SWITCH_WEEK} is after the switch "
        f"(MIDSEASON-SWITCH-RECORD.md): {AFTER_SWITCH_GAME}"
    )


@pytest.mark.usefixtures("isolated_git_env")
def test_a_non_ascii_section_refuses_before_any_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IN-A03: a non-ASCII character refuses by section and code point before anything is
    written, so the one-shot record is never spent on a crash."""
    argv = binding_fixture(tmp_path, rows=switch_rows())
    monkeypatch.setattr(
        season_readout, "_scope_section", lambda *_: ["Scope \u2014 weeks 5-9"]
    )

    line = _assert_refuses(argv, capsys, tmp_path)

    assert line == (
        f"{REFUSED}section 2 (Verdict scope) of the document holds the non-ASCII character "
        "U+2014; the record is written as ASCII only, so nothing was written (IN-A03)."
    )
    assert not (tmp_path / "repo" / BINDING_RECORD_PATH).parent.exists()


@pytest.mark.usefixtures("isolated_git_env")
def test_the_document_names_repo_relative_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stores the real run reads lie inside the repository; the document names them relative
    to it, never by the machine's absolute path."""
    argv = binding_fixture(tmp_path, rows=switch_rows())
    repo = tmp_path / "repo"
    # Copy the stores to where the defaults point, and pass the relative defaults' shape: the
    # binding run resolves a relative store path against the repository root.
    inside = {
        "--silver-dir": "data/silver",
        "--run-log": "logs/daily_lock_runs.jsonl",
        "--skip-records": "config/skip_records.jsonl",
    }
    shutil.copytree(_arg(argv, "--silver-dir"), repo / inside["--silver-dir"])
    for option in ("--run-log", "--skip-records"):
        target = repo / inside[option]
        target.parent.mkdir(parents=True)
        target.write_bytes(_arg(argv, option).read_bytes())
    for option, relative in inside.items():
        argv[argv.index(option) + 1] = relative

    code, lines = _run(argv, capsys, repo)

    assert code == 0, lines
    document = (repo / BINDING_RECORD_PATH).read_text("ascii")
    assert "- Silver directory: data/silver\n" in document
    assert "- Evidence logs/daily_lock_runs.jsonl: 1 lines, sha256 " in document
    assert "- Evidence config/skip_records.jsonl: 0 lines, sha256 " in document
    assert tmp_path.as_posix() not in document
