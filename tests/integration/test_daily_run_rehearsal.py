"""Rehearsing the daily lock-time run on fixture days: one case per R15 acceptance line.

Phase 33.2, Plan 33.2-28 Task 2. Each case drives the REAL entry point,
``scripts.daily_lock_pipeline.main(["--date", ...])``, on a frozen clock against a sandboxed data
root, through the REAL orchestrator and the REAL daily step registry's ``close_collection`` and
``generate_predictions`` steps, so the prediction rows it asserts on are the ones the run wrote.

1. The Thursday game: the Wednesday run predicts exactly that game and nothing else.
2. An empty night: a day with no games tomorrow writes a no-op record and no predictions.
3. A passed lock, including after a missed day: refused before a single request; never back-filled.
4. The weekend split: the Friday run predicts the Saturday games, the Saturday run the Sunday
   games, and neither predicts the other's. A weekly run structurally cannot get this right.

REUSED, NOT REBUILT
-------------------
The fixture day is the one ``tests/integration/test_daily_run_skips.py`` (Plan 33.2-03) defines --
its Thursday game, its three Sunday games and their kickoffs, its season and week -- and the same
sandbox discipline: the boundary guards digest production BEFORE the working directory moves into
the sandbox, and it is moved back in this module's own teardown, before they compare. That
module's day drives the Friday registry's predictions phase; the daily entry point selects its
slate from the recorded schedule itself, so this module writes a schedule, not gold.

WHAT STANDS IN, EACH ONE NAMED
------------------------------
The clock (frozen in the entry point and the daily steps); the schedule refresh (the nflverse
capture and ingest -- a request counter that returns the season); the collection steps that reach
the network and the build steps (omitted from the registry by name: only ``close_collection`` and
``generate_predictions`` run); the model scoring inside ``build_predictions`` (a recorder that
returns one row per game it was asked for); the pre-flight and post-run gates, pinned healthy; and
the completion alerts, which are log-only and not under test. Every HTTP client's ``send`` is
replaced by a counter that records the URL, so "zero requests" is counted, not assumed.

The production ``data/`` and ``artifacts/`` trees are content-digested around every case
(``data_boundary_guard`` / ``artifacts_boundary_guard``). This module imports nothing that can
shell out.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.daily_lock_pipeline as daily
import scripts.generate_current_week_predictions as predictions_module
from pipeline import daily_steps, live_skip, orchestrator, skip_log, steps
from pipeline.alert import PipelineAlertManager
from pipeline.health import PipelineHealthChecker
from pipeline.staleness import StalenessGate
from pipeline.steps import RunStatus
from tests.integration.test_daily_run_skips import (
    SEASON,
    SUNDAY_GAMES,
    SUNDAY_KICKOFF,
    THURSDAY_GAME,
    THURSDAY_KICKOFF,
    WEEK,
)
from utils.date_utils import ET

# The fixture week the Thursday, empty-night and missed-day cases share: the week before it closes
# with a Monday night game, so the schedule names week 3 as the current slate from Tuesday on.
PRIOR_WEEK_GAME = ("2026_W02_DET@BUF", pd.Timestamp("2026-09-22T00:15:00+00:00"), 2)

# A late-season weekend with Saturday games (week 16), for the weekend split. Kickoffs in UTC;
# mid-December Eastern time is UTC-5.
WEEKEND_WEEK = 16
SATURDAY_GAMES = ("2026_W16_DEN@KC", "2026_W16_PHI@WAS")
SATURDAY_KICKOFFS = (
    pd.Timestamp("2026-12-19T21:30:00+00:00"),  # Sat 16:30 ET
    pd.Timestamp("2026-12-20T01:00:00+00:00"),  # Sat 20:00 ET
)
WEEKEND_SUNDAY_GAMES = ("2026_W16_BUF@NE", "2026_W16_DAL@NYG")
WEEKEND_SUNDAY_KICKOFF = pd.Timestamp("2026-12-20T18:00:00+00:00")  # Sun 13:00 ET
WEEKEND_PRIOR_GAME = ("2026_W15_LV@DEN", pd.Timestamp("2026-12-15T01:15:00+00:00"), 15)

# The steps of the real daily registry this module runs. Everything before them reaches the network
# (weather, snaps, injuries, odds); everything between them builds from production-sized stores.
_RUN_STEPS = ("close_collection", "generate_predictions")

_PREDICTION_STUB_COLUMNS = ("wp_prob", "ats_prediction", "ou_prediction")


def _et(day: date, hour: int, minute: int = 0) -> datetime:
    """An ET wall-clock instant."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)


# ---------------------------------------------------------------------------
# The sandboxed fixture day
# ---------------------------------------------------------------------------


@dataclass
class Rehearsal:
    """One sandbox and the handles its assertions read."""

    root: Path
    monkeypatch: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture[str]
    refreshes: list[date] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    asked_to_predict: list[frozenset[str]] = field(default_factory=list)
    registries_built: list[frozenset[str]] = field(default_factory=list)

    def run(self, run_date: date, *, clock: datetime) -> tuple[int, dict[str, str]]:
        """Drive the real entry point with ``--date`` at a frozen instant; parse its output."""
        _freeze_clock(self.monkeypatch, clock)
        self.capsys.readouterr()
        code = daily.main(["--date", run_date.isoformat()])
        out = self.capsys.readouterr().out
        contract = {
            line.split("= ", 1)[0]: line.split("= ", 1)[1]
            for line in out.splitlines()
            if "= " in line and line.split("= ", 1)[0].isupper()
        }
        return code, contract

    def predictions(self, week: int) -> pd.DataFrame:
        path = self.predictions_csv(week)
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def predictions_csv(self, week: int) -> Path:
        return (
            self.root
            / "outputs"
            / "predictions"
            / f"predictions_{SEASON}_week{week}.csv"
        )

    def run_records(self) -> list[dict[str, Any]]:
        path = self.root / daily.DAILY_RUN_RECORDS
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, instant: datetime) -> None:
    """Freeze ``datetime.now`` in the entry point and in the daily steps."""

    class _Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return instant if tz is None else instant.astimezone(tz)

    monkeypatch.setattr(daily, "datetime", _Clock)
    monkeypatch.setattr(daily_steps, "datetime", _Clock)


def _write_schedule(root: Path, games: list[tuple[str, pd.Timestamp, int]]) -> None:
    """The sandbox's silver ``games``: the recorded schedule the run selects its slate from."""
    silver = root / "data" / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    matchups = [game_id.split("_")[2].split("@") for game_id, _k, _w in games]
    pd.DataFrame(
        {
            "game_id": [game_id for game_id, _k, _w in games],
            "season": SEASON,
            "week": [week for _g, _k, week in games],
            "kickoff_et": [kickoff for _g, kickoff, _w in games],
            "game_type": "REG",
            "away_team": [away for away, _home in matchups],
            "home_team": [home for _away, home in matchups],
        }
    ).to_parquet(silver / "games.parquet", index=False)


def _week_three_schedule() -> list[tuple[str, pd.Timestamp, int]]:
    return [
        PRIOR_WEEK_GAME,
        (THURSDAY_GAME, THURSDAY_KICKOFF, WEEK),
        *((game_id, SUNDAY_KICKOFF, WEEK) for game_id in SUNDAY_GAMES),
    ]


def _weekend_schedule() -> list[tuple[str, pd.Timestamp, int]]:
    return [
        WEEKEND_PRIOR_GAME,
        *(
            (game_id, kickoff, WEEKEND_WEEK)
            for game_id, kickoff in zip(SATURDAY_GAMES, SATURDAY_KICKOFFS, strict=True)
        ),
        *(
            (game_id, WEEKEND_SUNDAY_KICKOFF, WEEKEND_WEEK)
            for game_id in WEEKEND_SUNDAY_GAMES
        ),
    ]


def _install_stand_ins(rehearsal: Rehearsal) -> None:
    """Every stand-in named in the module docstring, pointed at the sandbox."""
    monkeypatch, root = rehearsal.monkeypatch, rehearsal.root

    # Zero requests is COUNTED: every HTTP client's send records its URL instead of sending.
    import httpx
    import requests

    def _count(_client: Any, request: Any, *_a: Any, **_k: Any) -> Any:
        rehearsal.requests.append(str(getattr(request, "url", request)))
        raise AssertionError(f"a network request was made: {request}")

    async def _count_async(client: Any, request: Any, *a: Any, **k: Any) -> Any:
        return _count(client, request, *a, **k)

    monkeypatch.setattr(httpx.Client, "send", _count)
    monkeypatch.setattr(httpx.AsyncClient, "send", _count_async)
    monkeypatch.setattr(requests.Session, "send", _count)

    def _refresh(run_date_et: date, *, dry_run: bool) -> int:
        rehearsal.refreshes.append(run_date_et)
        return SEASON

    monkeypatch.setattr(daily, "_refresh_schedule", _refresh)

    real_registry = daily.build_daily_step_registry

    def _registry(slate: daily_steps.DailySlate) -> list[steps.StepDefinition]:
        rehearsal.registries_built.append(slate.game_ids)
        return [step for step in real_registry(slate) if step.name in _RUN_STEPS]

    monkeypatch.setattr(daily, "build_daily_step_registry", _registry)

    def _build_predictions(
        season: int, week: int, *, only_game_ids: frozenset[str], **_k: Any
    ) -> pd.DataFrame:
        rehearsal.asked_to_predict.append(frozenset(only_game_ids))
        ids = sorted(only_game_ids)
        frame = pd.DataFrame({"game_id": ids, "season": season, "week": week})
        for column in _PREDICTION_STUB_COLUMNS:
            frame[column] = 0.5
        return frame

    monkeypatch.setattr(predictions_module, "build_predictions", _build_predictions)
    monkeypatch.setattr(
        predictions_module,
        "build_game_context",
        lambda ids, _season, _week: pd.DataFrame({"game_id": list(ids)}),
    )

    # Every output location, pinned to the sandbox.
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", Path("logs/daily_lock_runs.jsonl"))
    monkeypatch.setattr(
        steps, "_predictions_output_dir", lambda: root / "outputs" / "predictions"
    )
    monkeypatch.setattr(
        orchestrator, "LOG_PATH", root / "logs" / "friday_pipeline.json"
    )
    monkeypatch.setattr(
        skip_log, "SKIP_RECORD_PATH", root / "config" / "skip_records.jsonl"
    )
    monkeypatch.setattr(orchestrator, "get_current_nfl_week", lambda: (SEASON, WEEK))

    # The gates and alerts, pinned: they are not what is under test.
    monkeypatch.setattr(
        StalenessGate,
        "run_all_checks",
        lambda _self: type(
            "Passed", (), {"passed": True, "warnings": [], "errors": []}
        )(),
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_preflight", lambda _self: {"status": "healthy"}
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_postrun", lambda _self: {"status": "healthy"}
    )
    for method in (
        "alert_pipeline_success",
        "alert_finished_with_skips",
        "alert_degraded_completion",
        "alert_pipeline_failure",
        "alert_staleness_warning",
    ):
        monkeypatch.setattr(PipelineAlertManager, method, lambda *_a, **_k: None)


@pytest.fixture
def make_rehearsal(
    data_boundary_guard: dict[str, str],
    artifacts_boundary_guard: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Iterator[Callable[[str, list[tuple[str, pd.Timestamp, int]]], Rehearsal]]:
    """Build (and switch into) a sandbox holding *schedule*.

    The guards are requested first so they digest production before the working directory moves;
    it is moved back in THIS teardown, which runs before theirs (the Plan 33.2-03 discipline).
    """
    original_cwd = Path.cwd()
    live_skip.reset_excluded_games()

    def _make(name: str, schedule: list[tuple[str, pd.Timestamp, int]]) -> Rehearsal:
        root = tmp_path / name
        _write_schedule(root, schedule)
        os.chdir(root)
        rehearsal = Rehearsal(root=root, monkeypatch=monkeypatch, capsys=capsys)
        _install_stand_ins(rehearsal)
        return rehearsal

    try:
        yield _make
    finally:
        os.chdir(original_cwd)
        live_skip.reset_excluded_games()


# ---------------------------------------------------------------------------
# R15 line 1: the Thursday game
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_wednesday_run_predicts_exactly_the_thursday_game(
    make_rehearsal: Callable[..., Rehearsal],
) -> None:
    rehearsal = make_rehearsal("thursday", _week_three_schedule())
    wednesday = date(2026, 9, 23)

    code, contract = rehearsal.run(wednesday, clock=_et(wednesday, 17, 0))

    assert code == 0
    assert contract["RUN_STATUS"] == RunStatus.SUCCESS.value
    assert contract["NEXT_DAY_GAMES"] == "1"
    predicted = rehearsal.predictions(WEEK)
    assert list(predicted["game_id"]) == [THURSDAY_GAME]
    assert rehearsal.asked_to_predict == [frozenset({THURSDAY_GAME})]
    for column in daily_steps.STAMP_COLUMNS:
        assert predicted[column].notna().all(), column
    # Non-vacuity: the week's Sunday games EXIST in the schedule, so "exactly one" is a selection.
    week_three = pd.read_parquet(rehearsal.root / "data" / "silver" / "games.parquet")
    assert set(SUNDAY_GAMES) <= set(
        week_three.loc[week_three["week"] == WEEK, "game_id"]
    )


# ---------------------------------------------------------------------------
# R15 line 2: an empty night
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_night_with_no_games_tomorrow_writes_a_no_op_record_and_no_predictions(
    make_rehearsal: Callable[..., Rehearsal],
) -> None:
    rehearsal = make_rehearsal("empty", _week_three_schedule())
    tuesday = date(2026, 9, 22)

    code, contract = rehearsal.run(tuesday, clock=_et(tuesday, 17, 0))

    assert code == 0
    assert contract["NEXT_DAY_GAMES"] == "0"
    (record,) = rehearsal.run_records()
    assert record["outcome"] == "no_games"
    assert record["run_date_et"] == tuesday.isoformat()
    assert rehearsal.predictions(WEEK).empty
    assert rehearsal.asked_to_predict == []
    # A success-with-no-op: no pipeline was even built, so no FINISHED_WITH_SKIPS can arise.
    assert rehearsal.registries_built == []
    assert "RUN_STATUS" not in contract
    assert record["outcome"] != RunStatus.FINISHED_WITH_SKIPS.value


# ---------------------------------------------------------------------------
# R15 line 3: a passed lock, including after a missed day
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_passed_lock_is_refused_and_a_missed_day_is_never_back_filled(
    make_rehearsal: Callable[..., Rehearsal],
) -> None:
    rehearsal = make_rehearsal("missed", _week_three_schedule())
    wednesday, thursday = date(2026, 9, 23), date(2026, 9, 24)
    one_second_late = _et(wednesday, 18, 0).replace(second=1)

    # The selector the entry point uses refuses a decision instant after the Thursday game's lock.
    schedule = pd.read_parquet(rehearsal.root / "data" / "silver" / "games.parquet")
    with pytest.raises(live_skip.LockPassedError, match=THURSDAY_GAME):
        daily.select_slate(schedule, wednesday, decided_at=one_second_late)

    # The Wednesday run never happened. Thursday morning, the catch-up run for Wednesday refuses
    # BEFORE collection: no schedule refresh, no HTTP request, no registry, no prediction row.
    code, contract = rehearsal.run(wednesday, clock=_et(thursday, 9, 0))
    assert code == 0
    assert contract["LOCK_PASSED_BEFORE_COLLECTION"] == "1"
    assert rehearsal.refreshes == []
    assert rehearsal.requests == []
    assert rehearsal.registries_built == []
    (record,) = rehearsal.run_records()
    assert record["outcome"] == "lock_passed"
    assert record["game_ids"] == [THURSDAY_GAME]

    # Thursday's own run predicts Friday's games -- none -- and never back-fills Thursday's.
    code, contract = rehearsal.run(thursday, clock=_et(thursday, 17, 0))
    assert code == 0
    assert contract["NEXT_DAY_GAMES"] == "0"
    assert rehearsal.asked_to_predict == []
    assert not rehearsal.predictions_csv(WEEK).exists()
    assert rehearsal.requests == []


# ---------------------------------------------------------------------------
# R15 line 4: the weekend split
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_friday_run_predicts_saturday_and_the_saturday_run_predicts_sunday(
    make_rehearsal: Callable[..., Rehearsal],
) -> None:
    rehearsal = make_rehearsal("weekend", _weekend_schedule())
    friday, saturday = date(2026, 12, 18), date(2026, 12, 19)

    code, contract = rehearsal.run(friday, clock=_et(friday, 17, 0))
    assert (code, contract["RUN_STATUS"]) == (0, RunStatus.SUCCESS.value)
    after_friday = rehearsal.predictions(WEEKEND_WEEK)
    assert sorted(after_friday["game_id"]) == sorted(SATURDAY_GAMES)

    code, contract = rehearsal.run(saturday, clock=_et(saturday, 17, 0))
    assert (code, contract["RUN_STATUS"]) == (0, RunStatus.SUCCESS.value)

    assert rehearsal.asked_to_predict == [
        frozenset(SATURDAY_GAMES),
        frozenset(WEEKEND_SUNDAY_GAMES),
    ]
    after_saturday = rehearsal.predictions(WEEKEND_WEEK).set_index("game_id")
    assert sorted(after_saturday.index) == sorted(SATURDAY_GAMES + WEEKEND_SUNDAY_GAMES)
    # The Saturday games keep the rows the FRIDAY run computed before their own lock; the
    # Saturday run neither re-predicted nor re-stamped them.
    friday_rows = after_friday.set_index("game_id")
    for game_id in SATURDAY_GAMES:
        assert (
            after_saturday.loc[game_id, "computed_at_utc"]
            == friday_rows.loc[game_id, "computed_at_utc"]
        )
    locks = pd.to_datetime(after_saturday["information_cutoff_utc"], utc=True)
    assert set(locks.loc[list(SATURDAY_GAMES)]) == {pd.Timestamp(_et(friday, 18, 0))}
    assert set(locks.loc[list(WEEKEND_SUNDAY_GAMES)]) == {
        pd.Timestamp(_et(saturday, 18, 0))
    }
