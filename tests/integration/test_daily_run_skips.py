"""A live day drops ONLY the caught games, says so durably, and still finishes (D33.2-05).

Phase 33.2, Plan 33.2-03 Task 4 (SPEC R3; the standing prohibition on post-lock rows).

WHAT THIS MODULE DRIVES
-----------------------
The REAL entry points: ``pipeline.orchestrator.FridayPipeline.run()`` and
``scripts.friday_pipeline.main()``, through the REAL predictions-phase step registry. The
per-game refusal is caught by PRODUCTION code -- ``pipeline/live_skip.py`` wired at the
orchestrator's step seam -- never by this module. A test that caught the refusal itself would be
proving its own fixture (T-33.2-03-07).

What runs for real: the orchestrator loop and its status decision; ``step_build_features``'s body
(which threads the exclusion register into the build); the information-time gate
(``features.provenance.InformationTimeGate``) on a planted provenance frame; the build's own
``drop_excluded_games``; ``step_verify_gold_currency``; ``step_generate_predictions`` and
``generate_and_write`` (market join, edges, CSV writes); ``step_verify_prediction_currency``;
``step_generate_recommendations`` and ``generate_weekly_bet_list`` (schedule spine, odds join,
``BetSelector``, sizing, upsert, grading, both durable artifacts); ``step_export_artifacts``;
``step_validate_predictions``; the execution-log writer; the skip record; the CLI's exit code.

What stands in, each one named: the model objects (a recorder that also proves which games were
SCORED); ``backtest.diagnose.score_deployed_artifacts`` (the same recorder role Plan 33.2-02's
exclusion test uses); the bet-list strategy registry (``tests.fixtures.decision_frame``'s
Protocol-conforming mini strategies); the feature BUILDER's source loading (a fixture builder that
hands a planted provenance frame to the REAL gate after the REAL exclusion drop); the network and
production-store steps (odds ingest, market anchors, model validation, the web-cache rebuild),
which are no-op'd by name; and the staleness/health gates, which are pinned healthy. The decision
instant is pinned through ``pipeline.steps._decision_instant``, the one seam both consuming steps
read.

THE BINDING PROOF IS THE ZERO-ROWS ASSERTION, NOT A TypeError
--------------------------------------------------------------
If ``generate_weekly_bet_list`` lacked ``excluded_game_ids`` the bet-list step would raise a
``TypeError`` here -- but a parameter that exists and is not threaded to the schedule spine would
accept the argument, drop it, and leave that check green while the caught game was still priced
and written. So the binding assertion is behavioural: no row for the caught game in the bet-list
artifact AND in the tracker artifact, with the clean games' rows asserted PRESENT so an empty
frame for an unrelated reason cannot satisfy it.

WHY stake_units IS THE ONE COLUMN ALLOWED TO DIFFER
---------------------------------------------------
With a game skipped, every clean game's bet-list row equals the no-skip run's row except
``stake_units``: the pre-registered sizing shares ONE 10% weekly cap (D31-02) among fewer bets, so
a skipped game can only FREE room -- asserted as "never smaller". ``graded_at`` is also excluded
from the equality, for a different reason: it is the grading pass's own settlement clock, read
from the wall clock inside ``grade_pending_rows`` and not injectable through the weekly entry
point. It is a bookkeeping instant, not a decision fact, and it is asserted non-null instead.

NO ALERT CHANNEL
----------------
This phase writes the skip record and builds no notification channel: Phase 35 owns
notifications and will read ``config/skip_records.jsonl``. The fifth alert BRANCH
(``alert_finished_with_skips``) rides the existing log-only manager and is not a channel.

THE WRITE BOUNDARY
------------------
Every write lands in a per-test sandbox under ``tmp_path``. Production ``data/`` and
``artifacts/`` are content-digested around every test (``data_boundary_guard`` /
``artifacts_boundary_guard``). No git query is used, and this module imports nothing that could
shell out.

Run this module:  uv run pytest tests/integration/test_daily_run_skips.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import backtest.diagnose as diagnose_module
import scripts.build_features as build_features_module
import scripts.generate_current_week_predictions as predictions_module
from backtest import weekly_bet_list
from backtest.weekly_bet_list import (
    BET_LIST_ARTIFACT_NAME,
    BET_TRACKER_ARTIFACT_NAME,
    CANONICAL_TARGETS,
    LockPassedError,
    select_games_for_decision_instant,
)
from features.provenance import (
    PROVENANCE_COLUMNS,
    InformationTimeGate,
    build_lock_frame,
)
from pipeline import live_skip, orchestrator, skip_log, steps
from pipeline.alert import PipelineAlertManager
from pipeline.health import PipelineHealthChecker
from pipeline.orchestrator import FridayPipeline
from pipeline.staleness import StalenessGate
from pipeline.steps import RunStatus
from scripts import friday_pipeline
from tests.fixtures.decision_frame import chain_fit_record, mini_strategies

SEASON = 2026
WEEK = 3
_UNSCHEDULED_WEEK = 9

# The fixture day: three Sunday games (13:00 ET kickoffs lock Saturday 18:00 ET) ...
SUNDAY_KICKOFF = pd.Timestamp("2026-09-27T17:00:00+00:00")
SUNDAY_GAMES: tuple[str, ...] = (
    "2026_W03_KC@BUF",
    "2026_W03_NYJ@MIA",
    "2026_W03_DAL@PHI",
)
CAUGHT = SUNDAY_GAMES[1]
CLEAN: tuple[str, ...] = tuple(g for g in SUNDAY_GAMES if g != CAUGHT)

# ... and, for the missed-day case only, a Thursday night game (20:20 ET) that locked Wednesday.
THURSDAY_GAME = "2026_W03_GB@CHI"
THURSDAY_KICKOFF = pd.Timestamp("2026-09-25T00:20:00+00:00")
THURSDAY_LOCK = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)  # Wed 18:00 ET

# THE DECISION INSTANT every case pins: Friday noon ET -- after the Thursday game's lock and
# before the Sunday games'. The odds were captured before it.
DECISION_INSTANT = datetime(2026, 9, 25, 16, 0, tzinfo=UTC)
ODDS_CAPTURED_AT = "2026-09-25T12:00:00+00:00"

# The source name the planted violation is raised under, as the gate would name a registry key.
PLANTED_SOURCE = "injury"

# The prediction-phase steps whose REAL bodies would reach the network or a production store.
# No-op'd by derived adapter name; everything else in the phase runs for real.
_NO_OP_STEPS: tuple[str, ...] = (
    "ingest_odds",
    "build_market_anchors",
    "validate_features",
    "validate_models",
    "populate_web_cache",
)


# ---------------------------------------------------------------------------
# The sandboxed fixture day
# ---------------------------------------------------------------------------


@dataclass
class Recorders:
    """What the stand-ins observed during one day's runs."""

    predicted: list[str] = field(default_factory=list)
    priced: list[str] = field(default_factory=list)
    build_exclusions: list[frozenset[str]] = field(default_factory=list)
    alerts: dict[str, int] = field(default_factory=dict)


@dataclass
class Day:
    """One sandboxed live day and the handles its assertions read."""

    root: Path
    games: tuple[str, ...]
    recorders: Recorders

    @property
    def predictions_csv(self) -> Path:
        return (
            self.root
            / "outputs"
            / "predictions"
            / f"predictions_{SEASON}_week{WEEK}.csv"
        )

    @property
    def predictions_json(self) -> Path:
        return self.predictions_csv.with_suffix(".json")

    @property
    def context_csv(self) -> Path:
        return (
            self.root
            / "outputs"
            / "predictions"
            / f"game_context_{SEASON}_week{WEEK}.csv"
        )

    @property
    def bet_list(self) -> pd.DataFrame:
        return pd.read_parquet(
            self.root / "outputs" / "bet_list" / BET_LIST_ARTIFACT_NAME
        )

    @property
    def tracker(self) -> list[dict[str, Any]]:
        path = self.root / "outputs" / "bet_list" / BET_TRACKER_ARTIFACT_NAME
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def skip_record(self) -> Path:
        return self.root / "config" / "skip_records.jsonl"

    @property
    def run_log(self) -> dict[str, Any]:
        return json.loads(
            (self.root / "logs" / "friday_pipeline.json").read_text(encoding="utf-8")
        )


def _kickoff(game_id: str) -> pd.Timestamp:
    kickoff = THURSDAY_KICKOFF if game_id == THURSDAY_GAME else SUNDAY_KICKOFF
    return cast(pd.Timestamp, kickoff)


def _write_stores(root: Path, games: tuple[str, ...], *, schedule_week: int) -> None:
    """Silver schedule and odds, per-target gold, the frozen fit -- all under *root*."""
    silver = root / "data" / "silver"
    gold = root / "data" / "gold"
    for directory in (silver, gold, root / "artifacts", root / "outputs" / "p31"):
        directory.mkdir(parents=True, exist_ok=True)

    matchups = [game_id.split("_")[2].split("@") for game_id in games]
    pd.DataFrame(
        {
            "game_id": list(games),
            "season": [SEASON] * len(games),
            "week": [schedule_week] * len(games),
            "kickoff_et": [_kickoff(game_id) for game_id in games],
            "away_team": [away for away, _home in matchups],
            "home_team": [home for _away, home in matchups],
        }
    ).to_parquet(silver / "games.parquet", index=False)

    pd.DataFrame(
        {
            "game_id": list(games),
            "snapshot_ts": [ODDS_CAPTURED_AT] * len(games),
            "ml_home": [-130.0] * len(games),
            "ml_away": [110.0] * len(games),
            "spread": [2.5] * len(games),
            "total": [45.0] * len(games),
            "sportsbook": ["consensus"] * len(games),
            "is_live": [False] * len(games),
        }
    ).to_parquet(silver / "odds_snapshot.parquet", index=False)

    # Gold carries the week's rows plus each target's realized label, so the grading pass settles
    # every live bet and the tracker's counts are a real, non-zero measurement.
    gold_frame = pd.DataFrame(
        {
            "game_id": list(games),
            "season": [SEASON] * len(games),
            "week": [WEEK] * len(games),
            "game_key": list(range(len(games))),
            "home_win": [1] * len(games),
            "home_margin": [3.0] * len(games),
            "total_points": [44.0] * len(games),
        }
    )
    for target in CANONICAL_TARGETS:
        gold_frame.to_parquet(gold / f"features_{target}.parquet", index=False)

    (root / "outputs" / "p31" / "profitability_2025_verdict.json").write_text(
        json.dumps(chain_fit_record(0.0)), encoding="utf-8"
    )


class _ScoringModel:
    """Stands in for a deployed model and records every game it was asked to score."""

    def __init__(self, target: str, games: tuple[str, ...], sink: list[str]) -> None:
        self.target = target
        self.games = games
        self.sink = sink

    def _record(self, frame: pd.DataFrame) -> int:
        self.sink.extend(self.games[int(key)] for key in frame["game_key"])
        return len(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        n = self._record(frame)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        n = self._record(frame)
        return np.full(n, 2.0 if self.target == "ats" else 44.0)


_MODEL_VALUES: dict[str, dict[str, float]] = {
    "wp": {"model_prob": 0.61},
    "ats": {"model_spread": -3.5},
    "ou": {"model_total": 41.0},
}


def _fixture_builder(
    root: Path, planted: frozenset[str], sink: list[frozenset[str]]
) -> type:
    """A feature builder whose ONE source carries a post-lock time for each planted game.

    Everything that decides the outcome is real: the build's ``drop_excluded_games`` removes the
    games the register names, ``build_lock_frame`` derives each lock through the one rule, and
    ``InformationTimeGate.check`` raises the refusal the live half must catch.
    """

    class _Builder:
        def generate_feature_matrices(
            self, *, excluded_game_ids: frozenset[str] = frozenset(), **_kwargs: Any
        ) -> dict[str, pd.DataFrame]:
            sink.append(frozenset(excluded_game_ids))
            games = pd.read_parquet(root / "data" / "silver" / "games.parquet")
            games = games.loc[games["week"] == WEEK]
            sources = build_features_module.drop_excluded_games(
                {
                    "games": games,
                    PLANTED_SOURCE: pd.DataFrame(
                        {"game_id": games["game_id"], "value": 1.0}
                    ),
                },
                excluded_game_ids,
            )
            locks = build_lock_frame(sources["games"])
            provenance = cast(
                "pd.DataFrame",
                pd.DataFrame(
                    {
                        "game_id": list(locks.index),
                        "basis": "per_row",
                        "information_time": [
                            lock + timedelta(hours=1)
                            if game_id in planted
                            else lock - timedelta(hours=1)
                            for game_id, lock in locks.items()
                        ],
                    }
                )[list(PROVENANCE_COLUMNS)],
            )
            InformationTimeGate().check(
                PLANTED_SOURCE,
                sources[PLANTED_SOURCE],
                provenance,
                locks,
                no_information_signature={},
            )
            return {}

    return _Builder


@pytest.fixture
def make_day(
    data_boundary_guard: dict[str, str],
    artifacts_boundary_guard: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., Day]]:
    """Build (and switch into) a fresh sandboxed live day.

    The boundary guards are requested FIRST, so they digest production before the sandbox
    switches the working directory. The directory is switched with ``os.chdir`` and restored in
    THIS fixture's teardown, deliberately not through ``monkeypatch.chdir``: ``monkeypatch`` is
    set up earlier by autouse fixtures, so it is undone AFTER the guards compare -- and a guard
    comparing while the working directory is still the sandbox digests the sandbox's ``data/``
    instead of production's. This fixture depends on the guards, so its teardown runs first.
    """
    live_skip.reset_excluded_games()
    original_cwd = Path.cwd()

    def _make(
        name: str,
        *,
        planted: frozenset[str] = frozenset(),
        games: tuple[str, ...] = SUNDAY_GAMES,
        schedule_week: int = WEEK,
        run_week: int = WEEK,
    ) -> Day:
        root = tmp_path / name
        _write_stores(root, games, schedule_week=schedule_week)
        recorders = Recorders()
        os.chdir(root)

        # The week, the decision instant, and every output location, pinned to the sandbox.
        monkeypatch.setattr(
            orchestrator, "get_current_nfl_week", lambda: (SEASON, run_week)
        )
        monkeypatch.setattr(
            "utils.date_utils.get_current_nfl_week", lambda: (SEASON, run_week)
        )
        monkeypatch.setattr(steps, "_decision_instant", lambda: DECISION_INSTANT)
        monkeypatch.setattr(
            orchestrator, "LOG_PATH", root / "logs" / "friday_pipeline.json"
        )
        monkeypatch.setattr(
            steps, "_predictions_output_dir", lambda: root / "outputs" / "predictions"
        )
        monkeypatch.setattr(
            steps, "_bet_list_output_dir", lambda: root / "outputs" / "bet_list"
        )
        monkeypatch.setattr(
            skip_log, "SKIP_RECORD_PATH", root / "config" / "skip_records.jsonl"
        )

        # The pre-flight and post-run gates, pinned healthy: they are not what is under test.
        monkeypatch.setattr(
            StalenessGate,
            "run_all_checks",
            lambda _self: SimpleNamespace(passed=True, warnings=[], errors=[]),
        )
        monkeypatch.setattr(
            PipelineHealthChecker, "run_preflight", lambda _self: {"status": "healthy"}
        )
        monkeypatch.setattr(
            PipelineHealthChecker, "run_postrun", lambda _self: {"status": "healthy"}
        )

        # The stand-ins named in the module docstring.
        monkeypatch.setattr(
            predictions_module,
            "load_model_artifact",
            lambda target, artifacts_dir: {
                "model": _ScoringModel(target, games, recorders.predicted),
                "feature_list": ["game_key"],
                "calibrator": None,
            },
        )

        def _score(
            target: str, *, gold_df: pd.DataFrame, artifacts_dir: Path
        ) -> pd.DataFrame:
            recorders.priced.extend(gold_df["game_id"].astype(str))
            scored = cast("pd.DataFrame", gold_df[["game_id", "season", "week"]]).copy()
            for column, value in _MODEL_VALUES[target].items():
                scored[column] = value
            return scored

        monkeypatch.setattr(diagnose_module, "score_deployed_artifacts", _score)
        monkeypatch.setattr(
            weekly_bet_list, "build_strategies", lambda _fits: mini_strategies()
        )
        monkeypatch.setattr(
            build_features_module,
            "FeatureMatrixBuilder",
            _fixture_builder(root, planted, recorders.build_exclusions),
        )
        for name_ in _NO_OP_STEPS:
            adapter = f"step_{name_}"
            assert hasattr(steps, adapter), f"pipeline.steps has no {adapter}"
            monkeypatch.setattr(steps, adapter, lambda: None)

        # The alert spy: records which completion alert fired, by the method's own name.
        for method in (
            PipelineAlertManager.alert_finished_with_skips,
            PipelineAlertManager.alert_pipeline_success,
            PipelineAlertManager.alert_pipeline_failure,
            PipelineAlertManager.alert_degraded_completion,
        ):
            recorders.alerts[method.__name__] = 0

            def _spy(
                _self: Any, *_a: Any, _name: str = method.__name__, **_k: Any
            ) -> None:
                recorders.alerts[_name] += 1

            # The spy wears the replaced method's name, so assertions that read
            # ``PipelineAlertManager.<method>.__name__`` resolve the same key after patching.
            _spy.__name__ = method.__name__
            monkeypatch.setattr(PipelineAlertManager, method.__name__, _spy)

        return Day(root=root, games=games, recorders=recorders)

    try:
        yield _make
    finally:
        os.chdir(original_cwd)
        live_skip.reset_excluded_games()


def _run(day: Day) -> Any:
    """Drive the real orchestrator's predictions phase."""
    return FridayPipeline(mode="predictions-only", force=True).run()


def _run_cli(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(
        sys, "argv", ["friday_pipeline.py", "--predictions-only", "--force"]
    )
    return friday_pipeline.main()


def _prediction_ids(day: Day) -> list[str]:
    return sorted(pd.read_csv(day.predictions_csv)["game_id"].astype(str))


def _forward_graded(day: Day) -> int:
    return sum(int(block["bets_graded"]) for block in day.tracker)


# ---------------------------------------------------------------------------
# 1. One caught, two clean
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOneCaughtTwoClean:
    def test_exactly_the_clean_games_are_predicted_and_the_run_finishes_with_skips(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("one_caught", planted=frozenset({CAUGHT}))
        log = _run(day)

        assert RunStatus(log.status) is RunStatus.FINISHED_WITH_SKIPS
        assert _prediction_ids(day) == sorted(CLEAN)
        assert len(pd.read_csv(day.predictions_csv)) == 2
        assert log.skipped_games == [CAUGHT]

    def test_the_caught_game_was_never_scored_or_priced(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("never_scored", planted=frozenset({CAUGHT}))
        _run(day)

        assert CAUGHT not in day.recorders.predicted, "a model scored the dropped game"
        assert set(day.recorders.predicted) == set(CLEAN)
        assert CAUGHT not in day.recorders.priced, (
            "the bet list priced the dropped game"
        )
        assert set(day.recorders.priced) == set(CLEAN)

    def test_the_build_was_re_run_with_the_register_and_passed(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("rerun", planted=frozenset({CAUGHT}))
        _run(day)
        assert day.recorders.build_exclusions == [frozenset(), frozenset({CAUGHT})]

    def test_no_published_output_carries_a_row_for_the_caught_game(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("no_rows", planted=frozenset({CAUGHT}))
        _run(day)

        exported = pd.read_json(day.predictions_json)
        context = pd.read_csv(day.context_csv)
        bets = day.bet_list
        assert CAUGHT not in set(exported["game_id"])
        assert CAUGHT not in set(context["game_id"])
        assert CAUGHT not in set(bets["game_id"]), (
            "the bet list has a row for the dropped game; a suppressed row is still a row"
        )
        assert set(bets["game_id"]) == set(CLEAN), (
            "the clean games' bet rows are missing"
        )

    def test_clean_bet_rows_match_a_no_skip_day_except_the_pooled_stake(
        self, make_day: Callable[..., Day]
    ) -> None:
        """Binding: the caught game is absent from BOTH artifacts, the clean games present."""
        baseline_day = make_day("baseline")
        assert RunStatus(_run(baseline_day).status) is RunStatus.SUCCESS
        baseline = baseline_day.bet_list
        baseline_graded = _forward_graded(baseline_day)
        caught_live = int(
            ((baseline["game_id"] == CAUGHT) & (baseline["status"] == "live")).sum()
        )
        assert caught_live > 0, (
            "the control day never bet the game, so its absence proves nothing"
        )

        skipped_day = make_day("skipped", planted=frozenset({CAUGHT}))
        _run(skipped_day)
        skipped = skipped_day.bet_list

        keys = ["game_id", "target"]
        clean_baseline = (
            cast("pd.DataFrame", baseline[baseline["game_id"].isin(CLEAN)])
            .sort_values(keys)
            .reset_index(drop=True)
        )
        clean_skipped = skipped.sort_values(keys).reset_index(drop=True)
        not_decision_facts = ["stake_units", "graded_at"]
        pd.testing.assert_frame_equal(
            clean_baseline.drop(columns=not_decision_facts),
            clean_skipped.drop(columns=not_decision_facts),
        )
        assert bool(clean_skipped["graded_at"].notna().all())
        live = clean_skipped["status"] == "live"
        assert (
            clean_skipped.loc[live, "stake_units"].to_numpy()
            >= clean_baseline.loc[live, "stake_units"].to_numpy() - 1e-12
        ).all(), "skipping a game SHRANK a clean game's stake"

        # The tracker aggregates graded bets: the skipped day's count is the baseline's minus
        # exactly the caught game's live bets, and equals the clean live rows it graded.
        assert _forward_graded(skipped_day) == baseline_graded - caught_live
        assert _forward_graded(skipped_day) == int(live.sum()) > 0

    def test_exactly_one_skip_record_names_the_game_source_and_reason(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("one_record", planted=frozenset({CAUGHT}))
        _run(day)

        (record,) = skip_log.read_skip_records(day.skip_record)
        assert record["game_id"] == CAUGHT
        assert record["source"] == PLANTED_SOURCE
        assert record["reason"] == "post_lock"
        assert record["lock"] is not None and record["information_time"] is not None


# ---------------------------------------------------------------------------
# 2 and 2b. Every game caught -- and the unscheduled control
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAllCaught:
    def test_zero_predictions_and_finished_with_skips_never_no_games(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("all_caught", planted=frozenset(SUNDAY_GAMES))
        log = _run(day)

        assert RunStatus(log.status) is RunStatus.FINISHED_WITH_SKIPS
        assert log.status != RunStatus.SUCCESS.value
        assert pd.read_csv(day.predictions_csv).empty
        assert day.recorders.predicted == []
        assert sorted(log.skipped_games) == sorted(SUNDAY_GAMES)

    def test_no_refusal_escapes_the_bet_list_step(
        self, make_day: Callable[..., Day]
    ) -> None:
        """An empty-after-exclusion universe returns empty; it is not an empty schedule."""
        day = make_day("all_caught_bets", planted=frozenset(SUNDAY_GAMES))
        log = _run(day)

        recommendations = next(
            s for s in log.steps if s.name == "generate_recommendations"
        )
        assert recommendations.status == "success", recommendations.error
        assert all(s.status == "success" for s in log.steps), [
            (s.name, s.error) for s in log.steps if s.status != "success"
        ]
        assert day.bet_list.empty
        assert day.recorders.priced == []

    def test_a_day_with_no_scheduled_game_still_fails(
        self, make_day: Callable[..., Day]
    ) -> None:
        """The discriminating control: gold carries the week, the schedule does not."""
        day = make_day("unscheduled", schedule_week=_UNSCHEDULED_WEEK)
        with pytest.raises(RuntimeError, match="no scheduled games"):
            _run(day)

        log = day.run_log
        assert RunStatus(log["status"]) is RunStatus.FAILED
        failed = [s["name"] for s in log["steps"] if s["status"] == "failed"]
        assert failed == ["generate_recommendations"]
        assert not day.skip_record.exists()


# ---------------------------------------------------------------------------
# 3. Re-running the same live day
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_re_running_the_same_day_adds_no_skip_record(
    make_day: Callable[..., Day],
) -> None:
    day = make_day("rerun_day", planted=frozenset({CAUGHT}))
    first = _run(day)
    before = day.skip_record.read_bytes()

    second = _run(day)
    run_id = second.start_time
    assert run_id != first.start_time, (
        "the two runs share a run id; the proof needs two"
    )
    assert day.skip_record.read_bytes() == before

    (record,) = skip_log.read_skip_records(day.skip_record)
    assert record["run_id"] == first.start_time, "the record kept the FIRST run's id"
    natural_key = tuple(record[key] for key in skip_log.IDEMPOTENCY_KEY)
    assert skip_log.IDEMPOTENCY_KEY == ("run_date_et", "game_id", "source", "reason")
    assert natural_key[1:] == (CAUGHT, PLANTED_SOURCE, "post_lock")
    assert RunStatus(second.status) is RunStatus.FINISHED_WITH_SKIPS


# ---------------------------------------------------------------------------
# 4. A missed day never back-fills a passed lock
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAMissedDayNeverBackFills:
    def test_the_selection_fence_refuses_the_passed_game(self) -> None:
        """The real selector agrees: a decision after the Thursday lock is LockPassedError."""
        schedule = pd.DataFrame(
            {
                "game_id": [THURSDAY_GAME],
                "season": [SEASON],
                "week": [WEEK],
                "gameday": ["2026-09-24"],
            }
        )
        with pytest.raises(LockPassedError, match=THURSDAY_GAME):
            select_games_for_decision_instant(
                schedule, THURSDAY_LOCK, decided_at=DECISION_INSTANT
            )

    def test_the_catch_up_run_writes_no_row_for_the_passed_game(
        self, make_day: Callable[..., Day]
    ) -> None:
        day = make_day("missed_day", games=(THURSDAY_GAME, *SUNDAY_GAMES))
        log = _run(day)

        assert RunStatus(log.status) is RunStatus.FINISHED_WITH_SKIPS
        assert _prediction_ids(day) == sorted(SUNDAY_GAMES)
        assert THURSDAY_GAME not in set(day.bet_list["game_id"])
        assert THURSDAY_GAME not in day.recorders.predicted
        assert THURSDAY_GAME not in day.recorders.priced
        (record,) = skip_log.read_skip_records(day.skip_record)
        assert record["game_id"] == THURSDAY_GAME
        assert record["source"] == live_skip.DECISION_INSTANT_SOURCE
        assert record["reason"] == "post_lock"
        assert pd.Timestamp(record["lock"]) == pd.Timestamp(THURSDAY_LOCK)


# ---------------------------------------------------------------------------
# 5. The status survives the real entry point, end to end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTheStatusSurvivesTheRealEntryPoint:
    def test_the_cli_exits_zero_and_the_written_log_carries_the_skip(
        self, make_day: Callable[..., Day], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        day = make_day("cli", planted=frozenset({CAUGHT}))
        code = _run_cli(monkeypatch)

        assert code == 0
        written = day.run_log  # READ BACK from disk, not the in-memory object
        assert written["status"] == RunStatus.FINISHED_WITH_SKIPS.value
        assert written["skipped_games"] == [CAUGHT]

    def test_its_own_alert_fired_and_the_success_alert_did_not(
        self, make_day: Callable[..., Day], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        day = make_day("alerts", planted=frozenset({CAUGHT}))
        _run_cli(monkeypatch)

        alerts = day.recorders.alerts
        assert alerts[PipelineAlertManager.alert_finished_with_skips.__name__] == 1
        assert alerts[PipelineAlertManager.alert_pipeline_success.__name__] == 0, (
            "a run that dropped a game was alerted as a clean success"
        )
        assert alerts[PipelineAlertManager.alert_pipeline_failure.__name__] == 0


# ---------------------------------------------------------------------------
# 6. An ordinary failure is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_an_ordinary_critical_failure_still_fails_the_run(
    make_day: Callable[..., Day], monkeypatch: pytest.MonkeyPatch
) -> None:
    day = make_day("ordinary_failure")

    def _broken_validation() -> None:
        raise RuntimeError("models are unloadable")

    monkeypatch.setattr(steps, "step_validate_models", _broken_validation)
    code = _run_cli(monkeypatch)

    assert code == 1
    written = day.run_log
    assert RunStatus(written["status"]) is RunStatus.FAILED
    assert written["skipped_games"] == []
    assert not day.skip_record.exists()
    assert (
        day.recorders.alerts[PipelineAlertManager.alert_pipeline_failure.__name__] == 1
    )


# ---------------------------------------------------------------------------
# 7. A clean day is untouched
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_clean_day_writes_no_record_and_ends_success(
    make_day: Callable[..., Day], tmp_path: Path
) -> None:
    day = make_day("clean")
    log = _run(day)

    assert RunStatus(log.status) is RunStatus.SUCCESS
    assert log.skipped_games == []
    assert live_skip.excluded_games() == frozenset()
    assert not day.skip_record.exists()
    assert _prediction_ids(day) == sorted(SUNDAY_GAMES)
    assert (
        day.recorders.alerts[PipelineAlertManager.alert_pipeline_success.__name__] == 1
    )

    # An empty exclusion changes nothing: the pipeline's file is byte-identical to a direct
    # call that never mentions the parameter.
    direct = tmp_path / "direct"
    predictions_module.generate_and_write(season=SEASON, week=WEEK, output_dir=direct)
    assert (
        direct / day.predictions_csv.name
    ).read_bytes() == day.predictions_csv.read_bytes()
