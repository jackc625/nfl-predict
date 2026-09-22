"""History stops and saves nothing; the live half's terminal status reaches every deciding site.

Phase 33.2, Plan 33.2-03 Task 2 (SPEC R3; D33.2-05).

PART 1 -- THE HISTORY HALF OF D33.2-05
--------------------------------------
A training/gold build that meets ONE information-time violation must stop and save nothing.
This module runs the REAL history build path -- ``scripts.build_features.main`` with
``--season 2002``, the CLI an operator runs -- against a SANDBOXED data root seeded with a
real, read-only 2002 slice (the same slice and the same storage-singleton seam as Plan
33.2-01's tracer, ``tests/integration/test_information_time_tracer.py``). One game's Elo
information time is moved to its lock plus one second, and the module asserts:

* ``InformationTimeViolation`` escapes ``main`` and names the planted game;
* a CALL COUNTER on the build module's store-write entry point (``save_dataframe``, both the
  name ``scripts.build_features`` binds and the one in ``data.storage``) reads ZERO at the moment
  the violation surfaces. A content digest alone cannot tell "never wrote" from "wrote and
  rolled back"; the counter can;
* the counter is LIVE, not decorative: the same build with the value moved to exactly its lock
  (admissible, at-lock) runs to completion and the counter sees the gold write;
* the PRODUCTION ``data/`` and ``artifacts/`` digests are identical before and after, through
  the ``data_boundary_guard`` / ``artifacts_boundary_guard`` fixtures AND an explicit bracket
  around the module's builds. No git query is used anywhere here: ``data/`` is gitignored,
  so a git-based check over it is structurally incapable of failing (RESEARCH P11, COLD-05).

PART 2 -- THE THIRD TERMINAL STATUS IS A STATUS, NOT A SYMBOL
--------------------------------------------------------------
The live half ends a run that dropped a game in ``RunStatus.FINISHED_WITH_SKIPS``. A status
defined in one module and honoured by none is decorative (T-33.2-03-06), so this module checks
each site that decides a run's outcome: the enum, the execution log's serialisation, the CLI's
exit code and log line, and the alert manager's own branch. The end-to-end proof through the
real orchestrator is ``tests/integration/test_daily_run_skips.py``.

Run this module:  uv run pytest tests/integration/test_history_build_stops_on_catch.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import data.storage as storage_mod
import scripts.build_features as build_features_mod
from data.storage import DuckDBConnection, ParquetManager, save_dataframe
from features.elo_features import EloFeatureBuilder
from features.provenance import InformationTimeViolation
from pipeline.execution_log import ExecutionLog, write_execution_log_atomic
from pipeline.steps import RunStatus, StepStatus
from tests.data_boundary import (
    PRODUCTION_ARTIFACTS_ROOT,
    PRODUCTION_DATA_ROOT,
    assert_tree_unchanged,
    content_digest_tree,
)
from utils import game_lock

HISTORY_SEASON = 2002

#: The real silver tables the 2002 build needs, copied read-only (the tracer's slice).
#: Plan 33.2-12 added silver ``weather``: the weather builder is now an information-time
#: supplier and dates each game from the silver row its one fence selected. That table
#: carries no ``season`` column, so it is sliced by the season in its ``game_id``.
SEEDED_TABLES: tuple[str, ...] = (
    "games",
    "elo_game_snapshots",
    "weather_features",
    "weather",
)

#: The REAL supplier, captured before any test wraps it.
_REAL_INFORMATION_TIMES = EloFeatureBuilder.information_times


# ---------------------------------------------------------------------------
# The sandboxed history build
# ---------------------------------------------------------------------------


@dataclass
class WriteCounter:
    """Counts every call into the build's store-write entry points, then calls through."""

    calls: list[str] = field(default_factory=list)

    def wrap(self, label: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def _counted(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(label)
            return real(*args, **kwargs)

        return _counted


@dataclass
class HistoryRun:
    """What the planted and the at-lock history builds observed, recorded once."""

    planted_game: str
    planted_error: BaseException | None
    writes_when_refused: list[str]
    gold_before_planted: dict[str, str]
    gold_after_planted: dict[str, str]
    at_lock_error: BaseException | None
    writes_at_lock: list[str]


def _seed(sandbox: Path) -> pd.DataFrame:
    """Copy the season's real rows of each seeded table from production, read-only."""
    for layer in ("bronze", "silver", "gold"):
        (sandbox / layer).mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame()
    for table in SEEDED_TABLES:
        frame = pd.read_parquet(PRODUCTION_DATA_ROOT / "silver" / f"{table}.parquet")
        season = (
            frame["season"]
            if "season" in frame.columns
            else frame["game_id"].str[:4].astype(int)
        )
        frame = frame.loc[season == HISTORY_SEASON].reset_index(drop=True)
        assert len(frame) > 0, f"production silver {table} has no {HISTORY_SEASON} rows"
        save_dataframe(frame, table, layer="silver", replace_mode=True)
        if table == "games":
            games = frame
    return games


def _move_one_row(game_id: str, when: object) -> Callable[..., pd.DataFrame]:
    """Wrap the REAL ``information_times`` and overwrite exactly one game's time."""

    def _wrapped(
        self: EloFeatureBuilder, games_df: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        frame = _REAL_INFORMATION_TIMES(self, games_df, **kwargs).copy()
        frame["information_time"] = frame["information_time"].astype(object)
        frame.loc[frame["game_id"] == game_id, "information_time"] = when
        return frame

    return _wrapped


def _run_history_cli(
    monkeypatch: pytest.MonkeyPatch, game_id: str, when: object
) -> tuple[BaseException | None, list[str]]:
    """Run ``scripts.build_features.main --season 2002`` with one row moved; count writes."""
    counter = WriteCounter()
    monkeypatch.setattr(
        EloFeatureBuilder, "information_times", _move_one_row(game_id, when)
    )
    monkeypatch.setattr(
        build_features_mod,
        "save_dataframe",
        counter.wrap("scripts.build_features.save_dataframe", save_dataframe),
    )
    monkeypatch.setattr(
        storage_mod,
        "save_dataframe",
        counter.wrap("data.storage.save_dataframe", storage_mod.save_dataframe),
    )
    monkeypatch.setattr(
        sys, "argv", ["build_features.py", "--season", str(HISTORY_SEASON)]
    )
    error: BaseException | None = None
    try:
        build_features_mod.main()
    except BaseException as exc:
        error = exc
    return error, list(counter.calls)


@pytest.fixture(scope="module")
def history_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[HistoryRun]:
    """Run the planted and the at-lock history builds once, inside a production bracket.

    The function-scoped boundary fixtures bracket each test; they cannot bracket a MODULE
    fixture, which runs before them, so this fixture brackets its own builds with the same
    instrument (``content_digest_tree`` + ``assert_tree_unchanged``).
    """
    data_before = content_digest_tree(PRODUCTION_DATA_ROOT)
    artifacts_before = content_digest_tree(PRODUCTION_ARTIFACTS_ROOT)
    sandbox = tmp_path_factory.mktemp("p332_history") / "data"
    sandbox.mkdir(parents=True)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(sandbox))
        )
        monkeypatch.setattr(
            storage_mod,
            "_db_connection",
            DuckDBConnection(str(sandbox / "sandbox.duckdb")),
        )
        games = _seed(sandbox)
        locks = game_lock.lock_frame(games)
        # A game from week 2 onward, whose Elo row is DATED (per_row) rather than undatable.
        planted_game = str(
            games.loc[games["week"] >= 2, "game_id"].sort_values().iloc[0]
        )
        planted_lock = pd.Timestamp(cast(Any, locks[planted_game]))

        gold_before = content_digest_tree(sandbox / "gold")
        planted_error, writes_when_refused = _run_history_cli(
            monkeypatch, planted_game, planted_lock + pd.Timedelta(seconds=1)
        )
        gold_after = content_digest_tree(sandbox / "gold")

        at_lock_error, writes_at_lock = _run_history_cli(
            monkeypatch, planted_game, planted_lock
        )

    assert_tree_unchanged(
        data_before, content_digest_tree(PRODUCTION_DATA_ROOT), PRODUCTION_DATA_ROOT
    )
    assert_tree_unchanged(
        artifacts_before,
        content_digest_tree(PRODUCTION_ARTIFACTS_ROOT),
        PRODUCTION_ARTIFACTS_ROOT,
    )
    yield HistoryRun(
        planted_game=planted_game,
        planted_error=planted_error,
        writes_when_refused=writes_when_refused,
        gold_before_planted=gold_before,
        gold_after_planted=gold_after,
        at_lock_error=at_lock_error,
        writes_at_lock=writes_at_lock,
    )


@pytest.mark.integration
@pytest.mark.usefixtures("data_boundary_guard", "artifacts_boundary_guard")
class TestAHistoryBuildStopsAndSavesNothing:
    """D33.2-05, history half, on the real CLI path."""

    def test_the_violation_escapes_the_history_build(
        self, history_run: HistoryRun
    ) -> None:
        assert isinstance(history_run.planted_error, InformationTimeViolation), (
            f"the planted post-lock value did not stop the build: {history_run.planted_error!r}"
        )

    def test_the_refusal_names_the_offending_game(
        self, history_run: HistoryRun
    ) -> None:
        error = history_run.planted_error
        assert isinstance(error, InformationTimeViolation)
        assert history_run.planted_game in str(error)
        assert history_run.planted_game in error.details["game_ids"]

    def test_no_store_write_had_happened_when_the_violation_surfaced(
        self, history_run: HistoryRun
    ) -> None:
        """A digest cannot tell 'never wrote' from 'wrote and rolled back'; a counter can."""
        assert history_run.writes_when_refused == [], (
            f"the history build wrote before refusing: {history_run.writes_when_refused}"
        )

    def test_the_sandbox_gold_is_byte_identical_after_the_refusal(
        self, history_run: HistoryRun
    ) -> None:
        assert history_run.gold_after_planted == history_run.gold_before_planted

    def test_the_write_counter_is_live_on_an_admissible_build(
        self, history_run: HistoryRun
    ) -> None:
        """Non-vacuity: at-lock is admissible, so the same path completes and IS counted."""
        assert history_run.at_lock_error is None, history_run.at_lock_error
        assert "scripts.build_features.save_dataframe" in history_run.writes_at_lock, (
            "the counter never saw the gold write, so its zero above would prove nothing"
        )

    def test_this_module_never_asks_git(self) -> None:
        """Structural: the boundary is a content digest here, never a git query."""
        text = Path(__file__).read_text(encoding="utf-8")
        assert ("git " + "status") not in text


# ---------------------------------------------------------------------------
# Part 2: the third terminal status, at every deciding site
# ---------------------------------------------------------------------------


class TestTheRunStatusVocabulary:
    def test_run_status_has_exactly_three_members(self) -> None:
        assert sorted(m.name for m in RunStatus) == [
            "FAILED",
            "FINISHED_WITH_SKIPS",
            "SUCCESS",
        ]

    def test_the_per_step_status_is_unchanged(self) -> None:
        assert sorted(m.name for m in StepStatus) == [
            "FAILED",
            "PENDING",
            "RETRIED",
            "RUNNING",
            "SKIPPED",
            "SUCCESS",
        ]

    def test_finished_with_skips_is_neither_success_nor_failure(self) -> None:
        skipped = RunStatus.FINISHED_WITH_SKIPS
        assert skipped is not RunStatus.SUCCESS
        assert skipped is not RunStatus.FAILED
        assert skipped.value not in (RunStatus.SUCCESS.value, RunStatus.FAILED.value)
        # The wire value is the one ExecutionLog.status carries; a success branch keyed on
        # the string "success" must not be taken for it.
        assert skipped.value != "success"
        assert skipped.value == "finished_with_skips"

    def test_the_health_endpoint_spells_the_same_wire_value(self) -> None:
        """api/ stays stdlib-only, so it spells the value; the spelling must not drift."""
        from api.routes import health

        assert (
            RunStatus.FINISHED_WITH_SKIPS.value
            == health.PIPELINE_STATUS_FINISHED_WITH_SKIPS
        )


class TestTheExecutionLogCarriesTheSkippedGames:
    def test_skipped_games_round_trips_through_the_atomic_writer(
        self, tmp_path: Path
    ) -> None:
        log = ExecutionLog(
            status=RunStatus.FINISHED_WITH_SKIPS.value,
            start_time="2026-09-19T17:30:00-04:00",
            season=2026,
            week=3,
            pid=1,
            skipped_games=["2026_W03_KC@BUF"],
        )
        path = tmp_path / "friday_pipeline.json"
        write_execution_log_atomic(log, path)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["status"] == "finished_with_skips"
        assert written["skipped_games"] == ["2026_W03_KC@BUF"]

    def test_a_clean_log_carries_an_empty_list(self) -> None:
        log = ExecutionLog(status="success", start_time="t", season=2026, week=3, pid=1)
        assert log.skipped_games == []


class _RecordingAlertManager:
    """Stands in for ``utils.alert_manager.AlertManager``: records, sends nothing."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []

    def create_alert(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return kwargs

    def send_alert(self, alert: dict[str, Any]) -> None:
        self.sent.append(alert)


class TestTheAlertManagerHasItsOwnBranch:
    def test_finished_with_skips_raises_a_warning_naming_the_games(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipeline.alert as alert_mod
        from utils.alert_manager import AlertLevel

        recorder = _RecordingAlertManager()
        monkeypatch.setattr(alert_mod, "get_alert_manager", lambda: recorder)
        manager = alert_mod.PipelineAlertManager()
        manager.alert_finished_with_skips(["2026_W03_KC@BUF"], 21, 2026, 3)

        assert len(recorder.sent) == 1
        alert = recorder.created[0]
        assert alert["level"] is AlertLevel.WARNING
        assert alert["details"]["skipped_games"] == ["2026_W03_KC@BUF"]
        assert "2026_W03_KC@BUF" in alert["message"]
        assert alert["title"] != "Pipeline Completed Successfully"


class TestTheCliExitsZeroForACompletedRunThatSkipped:
    def _run_main(
        self, monkeypatch: pytest.MonkeyPatch, status: str
    ) -> tuple[int, list[tuple[str, str, dict[str, Any]]]]:
        import pipeline.orchestrator as orchestrator_mod
        import scripts.friday_pipeline as cli

        final = ExecutionLog(
            status=status,
            start_time="2026-09-19T17:30:00-04:00",
            season=2026,
            week=3,
            pid=1,
            skipped_games=["2026_W03_KC@BUF"],
        )

        class _Pipeline:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def run(self) -> ExecutionLog:
                return final

        lines: list[tuple[str, str, dict[str, Any]]] = []

        class _Logger:
            def info(self, event: str, **kw: Any) -> None:
                lines.append(("info", event, kw))

            def warning(self, event: str, **kw: Any) -> None:
                lines.append(("warning", event, kw))

            def error(self, event: str, **kw: Any) -> None:
                lines.append(("error", event, kw))

        monkeypatch.setattr(orchestrator_mod, "FridayPipeline", _Pipeline)
        monkeypatch.setattr(cli, "logger", _Logger())
        monkeypatch.setattr(sys, "argv", ["friday_pipeline.py", "--force"])
        return cli.main(), lines

    def test_a_finished_with_skips_run_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, _lines = self._run_main(monkeypatch, RunStatus.FINISHED_WITH_SKIPS.value)
        assert code == 0

    def test_its_log_line_is_distinct_and_never_the_failure_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _code, lines = self._run_main(monkeypatch, RunStatus.FINISHED_WITH_SKIPS.value)
        events = [event for _level, event, _kw in lines]
        assert "Pipeline failed" not in events
        assert "Pipeline finished" not in events, (
            "a run that dropped games was reported in the same words as a clean one"
        )
        skipped_lines = [kw for _lv, _ev, kw in lines if kw.get("skipped_games")]
        assert skipped_lines == [
            {
                "status": "finished_with_skips",
                "skipped_games": ["2026_W03_KC@BUF"],
                "duration_ms": 0,
            }
        ]

    def test_an_unknown_status_still_exits_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: widening the accepted tuple did not accept everything."""
        code, _lines = self._run_main(monkeypatch, RunStatus.FAILED.value)
        assert code == 1
