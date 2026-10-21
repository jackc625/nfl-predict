"""What the 2026 season-close readout must show (Phase 38, Plan 38-04; SPEC R7, R9).

Every scheduled game of the verdict scope lands in exactly one coverage state, rolled up per week,
so a partly missed week can never read as covered: bet, no bet, failed or refused, missed, and in
preview only not yet due. A game inferred never played (D-26) leaves the denominator, is named
with the reason it was inferred, and is counted once. Each evidence file is listed from the one
read its records came from (D-08). The preview is labelled "season to date -- not the verdict"
and shows no test result.

COLD-05: fixture tests write only under ``tmp_path``. The real-store tests READ the production
ledger, silver games, run log and skip records (never write them) and step aside with a message
containing "absent" when a store is missing on the machine running them.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from forward_ledger.declarations import load_verdict_scope
from forward_ledger.settle import DEFAULT_SILVER_DIR, SILVER_GAMES_FILENAME
from forward_ledger.store import LEDGER_DIR, append_rows, ledger_path, read_entries
from forward_ledger.verdict import verdict_rows
from pipeline.skip_log import SKIP_RECORD_PATH
from scripts import season_readout
from scripts.season_coverage import (
    STATE_MISSED,
    STATE_NOT_YET_DUE,
    CoverageReport,
    classify_coverage,
)
from tests.unit.test_forward_ledger_store import make_row
from utils.current_slate import load_recorded_schedule

REAL_LEDGER = ledger_path(LEDGER_DIR)
REAL_SILVER_GAMES = DEFAULT_SILVER_DIR / SILVER_GAMES_FILENAME
REAL_RUN_LOG = season_readout.DEFAULT_RUN_LOG
REAL_SKIP_RECORDS = SKIP_RECORD_PATH

# One ET kickoff day per week (the recorded schedule refuses two weeks on one day).
KICKOFF_BY_WEEK: dict[int, datetime] = {
    4: datetime(2026, 10, 4, 17, 0, tzinfo=UTC),
    5: datetime(2026, 10, 11, 17, 0, tzinfo=UTC),
    6: datetime(2026, 10, 18, 17, 0, tzinfo=UTC),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_real_stores(*paths: Path) -> None:
    """Step aside, naming the store, when a real store is absent on this machine."""
    for path in paths:
        if not Path(path).exists():
            pytest.skip(
                f"the real store {Path(path).as_posix()} is absent on this machine"
            )


def _line(lines: list[str], prefix: str) -> str:
    """The one output line starting with *prefix*."""
    matches = [line for line in lines if line.startswith(prefix)]
    assert len(matches) == 1, f"expected one line starting {prefix!r}, got {matches}"
    return matches[0]


def _fields(line: str) -> dict[str, str]:
    """The ``key=value`` fields of one output line."""
    return dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)


def _game(game_id: str, *, scored: bool = True) -> dict[str, Any]:
    """One silver ``games`` row; the week comes from the standard game id."""
    week = int(game_id.split("_W")[1][:2])
    return {
        "game_id": game_id,
        "season": 2026,
        "week": week,
        "kickoff_et": pd.Timestamp(KICKOFF_BY_WEEK[week]),
        "game_type": "REG",
        "home_score": 24.0 if scored else None,
        "away_score": 17.0 if scored else None,
    }


def _row(
    game_id: str, target: str, status: str = "live", **overrides: Any
) -> dict[str, Any]:
    """One counted live-arm ledger row for *game_id*."""
    week = int(game_id.split("_W")[1][:2])
    bet_side = {"wp": "home", "ats": "home", "ou": "over"}[target]
    return make_row(
        game_id=game_id,
        week=week,
        target=target,
        status=status,
        bet_side=bet_side,
        rejection_reason=None if status == "live" else "below_ev_floor",
        **overrides,
    )


def _skip_record(game_id: str) -> dict[str, Any]:
    """One skip record carrying every required key."""
    return {
        "game_id": game_id,
        "information_time": None,
        "lock": None,
        "reason": "no_provenance",
        "recorded_at": "2026-10-11T21:09:23+00:00",
        "run_date_et": "2026-10-10",
        "run_id": "2026-10-10T17:00:12-04:00",
        "source": "weather",
    }


def _run_record(outcome: str, game_ids: Sequence[str]) -> dict[str, Any]:
    """One daily-run record, in the shape scripts/daily_lock_pipeline.py writes."""
    return {
        "game_ids": list(game_ids),
        "outcome": outcome,
        "recorded_at": "2026-10-10T21:00:00+00:00",
        "run_date_et": "2026-10-10",
    }


def _jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _stores(
    root: Path,
    games: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]] = (),
    run_log: Sequence[dict[str, Any]] | None = (),
    skip_records: Sequence[dict[str, Any]] | None = (),
) -> list[str]:
    """Write fixture stores under *root*; return the readout argv that reads them.

    ``None`` for an evidence file leaves it absent.
    """
    ledger_dir, silver_dir = root / "ledger", root / "silver"
    ledger_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)
    if rows:
        append_rows(ledger_dir, rows)
    pd.DataFrame(list(games)).to_parquet(silver_dir / SILVER_GAMES_FILENAME)
    run_log_path, skip_path = (
        root / "daily_lock_runs.jsonl",
        root / "skip_records.jsonl",
    )
    if run_log is not None:
        _jsonl(run_log_path, run_log)
    if skip_records is not None:
        _jsonl(skip_path, skip_records)
    return [
        "--ledger-dir",
        str(ledger_dir),
        "--silver-dir",
        str(silver_dir),
        "--run-log",
        str(run_log_path),
        "--skip-records",
        str(skip_path),
    ]


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str]]:
    code = season_readout.main(argv)
    return code, capsys.readouterr().out.splitlines()


def _report(argv: list[str], *, now: datetime | None) -> CoverageReport:
    """classify_coverage over the fixture stores *argv* names."""
    arg = dict(zip(argv[::2], argv[1::2], strict=True))
    scope = load_verdict_scope()
    counted = verdict_rows(read_entries(arg["--ledger-dir"]), scope)
    games_path = Path(arg["--silver-dir"]) / SILVER_GAMES_FILENAME
    _, refused = season_readout.read_refusal_evidence(
        Path(arg["--run-log"]), Path(arg["--skip-records"])
    )
    return classify_coverage(
        load_recorded_schedule(games_path),
        pd.read_parquet(games_path),
        counted,
        refused,
        scope,
        now=now,
    )


def _not_played_lines(lines: list[str]) -> list[str]:
    return [
        line
        for line in lines
        if line.startswith("NOT_PLAYED= ") and line != "NOT_PLAYED= none"
    ]


# ---------------------------------------------------------------------------
# Real stores (read only)
# ---------------------------------------------------------------------------


def test_the_preview_accounts_for_weeks_5_and_6_on_the_real_stores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SPEC AC 19: week 5 lost 12 of 15 games with the laptop off and BAL@ATL to a weather skip;
    week 6 missed WAS@SF only."""
    _require_real_stores(
        REAL_LEDGER, REAL_SILVER_GAMES, REAL_RUN_LOG, REAL_SKIP_RECORDS
    )

    code, lines = _run([], capsys)

    assert code == 0
    assert _line(lines, "COVERAGE= week=5 ").startswith(
        "COVERAGE= week=5 scheduled=15 decided=2 bet=2 no_bet=0 failed_or_refused=1 missed=12"
    )
    assert _line(lines, "COVERAGE= week=6 ").startswith(
        "COVERAGE= week=6 scheduled=14 decided=13 bet=11 no_bet=2 failed_or_refused=0 missed=1"
    )
    total = _fields(_line(lines, "COVERAGE_TOTAL= "))
    states = ("bet", "no_bet", "failed_or_refused", "missed", "not_yet_due")
    assert sum(int(total[state]) for state in states) == int(total["scheduled"])
    assert int(total["not_played"]) == len(_not_played_lines(lines))
    assert all("reason=inferred: " in line for line in _not_played_lines(lines))


# ---------------------------------------------------------------------------
# The coverage rule on fixtures
# ---------------------------------------------------------------------------


def test_each_game_lands_in_exactly_one_state_by_precedence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bet_game, no_bet_game = "2026_W05_TB@DAL", "2026_W05_BUF@LA"
    refused_game, missed_game = "2026_W05_BAL@ATL", "2026_W05_NYG@WAS"
    partial_game, early_game = "2026_W05_KC@JAX", "2026_W04_PIT@CLE"
    games = [
        _game(game_id)
        for game_id in (
            bet_game,
            no_bet_game,
            refused_game,
            missed_game,
            partial_game,
            early_game,
        )
    ]
    rows = [
        *(_row(bet_game, target) for target in ("wp", "ats", "ou")),
        *(_row(no_bet_game, target, "suppressed") for target in ("wp", "ats", "ou")),
        _row(no_bet_game, "ats", arm="shadow"),  # a shadow twin bet never counts
        _row(partial_game, "wp"),
        _row(partial_game, "ats", "suppressed"),
        _row(early_game, "wp"),  # week 4 is outside the scope
    ]
    argv = _stores(
        tmp_path,
        games,
        rows,
        run_log=[_run_record("run_failed", [bet_game])],  # rows outrank a refusal
        skip_records=[_skip_record(refused_game)],
    )

    code, lines = _run(argv, capsys)

    assert code == 0
    week5 = _fields(_line(lines, "COVERAGE= week=5 "))
    assert week5 == {
        "week": "5",
        "scheduled": "5",
        "decided": "3",
        "bet": "2",
        "no_bet": "1",
        "failed_or_refused": "1",
        "missed": "1",
        "not_yet_due": "0",
        "not_played": "0",
    }
    assert not any(line.startswith("COVERAGE= week=4 ") for line in lines)
    assert _line(lines, "PARTIAL= ") == f"PARTIAL= {partial_game} targets=ats,wp"
    assert _line(lines, "MISSED_GAMES= ") == "MISSED_GAMES= 1"
    assert _line(lines, "NOT_PLAYED= ") == "NOT_PLAYED= none"
    for line in lines:
        if line.startswith("COVERAGE= "):
            week = _fields(line)
            states = ("bet", "no_bet", "failed_or_refused", "missed", "not_yet_due")
            assert sum(int(week[state]) for state in states) == int(week["scheduled"])
    states_by_game = {
        game.game_id: game.state for game in _report(argv, now=None).games
    }
    assert states_by_game[missed_game] == STATE_MISSED
    assert early_game not in states_by_game


def test_a_game_never_played_leaves_the_denominator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unscored, scored = "2026_W05_TB@DAL", "2026_W06_SF@SEA"
    absent = "2026_W06_MIA@NE"  # named by the run log, never in the schedule
    argv = _stores(
        tmp_path,
        [_game(unscored, scored=False), _game("2026_W05_BUF@LA"), _game(scored)],
        run_log=[_run_record("run_failed", [absent])],
    )

    code, lines = _run(argv, capsys)

    assert code == 0
    assert _not_played_lines(lines) == [
        f"NOT_PLAYED= {unscored} week=5 reason=inferred: no final score while week 6 is scored",
        f"NOT_PLAYED= {absent} week=6 "
        "reason=inferred: named by run log but absent from the schedule",
    ]
    week5 = _fields(_line(lines, "COVERAGE= week=5 "))
    week6 = _fields(_line(lines, "COVERAGE= week=6 "))
    assert (week5["scheduled"], week5["not_played"]) == ("1", "1")
    assert (week6["scheduled"], week6["not_played"]) == ("1", "1")
    report = _report(argv, now=None)
    total = _fields(_line(lines, "COVERAGE_TOTAL= "))
    assert sum(week.not_played for week in report.weeks) == len(report.not_played)
    assert (
        len(report.not_played)
        == int(total["not_played"])
        == len(_not_played_lines(lines))
    )
    assert len(report.not_played) == 2


def test_a_postponed_game_reads_as_not_played_until_its_score_lands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The adopted D-26 / D-20 result: a postponed game and its bet read as not played (and
    void) while it is unscored, and return to the denominator once its score lands."""
    postponed, later = "2026_W05_TB@DAL", "2026_W06_SF@SEA"
    rows = [_row(postponed, "wp")]

    unscored_argv = _stores(
        tmp_path / "unscored", [_game(postponed, scored=False), _game(later)], rows
    )
    code, lines = _run(unscored_argv, capsys)
    assert code == 0
    assert _not_played_lines(lines) == [
        f"NOT_PLAYED= {postponed} week=5 reason=inferred: no final score while week 6 is scored"
    ]
    assert _fields(_line(lines, "COVERAGE= week=5 "))["scheduled"] == "0"

    scored_argv = _stores(tmp_path / "scored", [_game(postponed), _game(later)], rows)
    code, lines = _run(scored_argv, capsys)
    assert code == 0
    assert _line(lines, "NOT_PLAYED= ") == "NOT_PLAYED= none"
    week5 = _fields(_line(lines, "COVERAGE= week=5 "))
    assert (week5["scheduled"], week5["bet"]) == ("1", "1")


def test_a_future_lock_is_not_yet_due_in_preview_only(tmp_path: Path) -> None:
    game_id = "2026_W05_TB@DAL"
    argv = _stores(tmp_path, [_game(game_id, scored=False)])
    before_lock = datetime(
        2026, 10, 9, 12, 0, tzinfo=UTC
    )  # the lock is Sat 10 Oct 18:00 ET

    preview = _report(argv, now=before_lock)
    binding = _report(argv, now=None)

    assert [game.state for game in preview.games] == [STATE_NOT_YET_DUE]
    assert preview.weeks[0].not_yet_due == 1
    assert [game.state for game in binding.games] == [STATE_MISSED]
    assert binding.missed_total == 1


def test_a_malformed_run_log_line_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _stores(tmp_path, [_game("2026_W05_TB@DAL")])
    Path(argv[argv.index("--run-log") + 1]).write_text("not json\n", encoding="utf-8")

    code, lines = _run(argv, capsys)

    assert code == 2
    assert lines[-1].startswith("STORE_UNREADABLE= ")


def test_each_evidence_file_is_listed_or_named_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _stores(
        tmp_path,
        [_game("2026_W05_TB@DAL")],
        run_log=[
            _run_record("no_games", []),
            _run_record("run_failed", ["2026_W05_TB@DAL"]),
        ],
        skip_records=None,
    )
    run_log = argv[argv.index("--run-log") + 1]
    skip_records = argv[argv.index("--skip-records") + 1]

    code, lines = _run(argv, capsys)

    assert code == 0
    digest = hashlib.sha256(Path(run_log).read_bytes()).hexdigest()
    evidence = [line for line in lines if line.startswith("EVIDENCE= ")]
    assert evidence == [
        f"EVIDENCE= path={Path(run_log).as_posix()} lines=2 sha256={digest}",
        f"EVIDENCE= path={Path(skip_records).as_posix()} missing",
    ]
