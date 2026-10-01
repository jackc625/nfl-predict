"""A game the run chose not to bet produces NO bet row of any kind (D33.2-05, Plan 33.2-02).

THE INTERFACE THIS MODULE PROVES
--------------------------------
``excluded_game_ids: frozenset[str] = frozenset()``, keyword-only, on the five functions of
``backtest.weekly_bet_list`` that sit between the weekly entry point and the schedule spine:
``generate_weekly_bet_list`` -> ``build_weekly_decision_frame`` -> ``build_weekly_candidates``,
and ``build_freeze_instant_candidates`` -> ``select_games_for_decision_instant``. Plan 33.2-03
wires ``pipeline/steps.py`` to pass the live skip register through the first of them.

WHY THE SPINE, AND WHY THIS MODULE DRIVES THE REAL BUILDER
----------------------------------------------------------
``records_to_bet_list_frame`` stamps one row per (game, target) INCLUDING suppressed rows, so an
exclusion applied at the selector would still write a suppressed row with a rejection_reason --
still a row, still a write. The drop therefore lands on the schedule spine inside
``build_weekly_candidates``, before the odds join and before scoring. That placement is only
provable against the REAL builder, so this module writes a sandbox silver/gold tree and stands in
ONLY for ``backtest.diagnose.score_deployed_artifacts`` -- with a recorder, so "the excluded game
is never scored" is a measured fact rather than a reading of the source.

THE BINDING ASSERTION IS ON ``game_id``, NEVER ON ``status``. A suppressed row would satisfy a
status check while still being a written row, so every zero-rows claim below reads the frame's
``game_id`` column.

Everything is sandboxed under ``tmp_path``; nothing under ``data/``, ``outputs/`` or
``artifacts/`` is read or written.

Run this module:  uv run pytest tests/unit/test_weekly_candidates_exclusion.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import backtest.diagnose as diagnose_module
from backtest import weekly_bet_list
from backtest.weekly_bet_list import (
    CANONICAL_TARGETS,
    build_weekly_candidates,
    build_weekly_decision_frame,
)
from tests.fixtures.decision_frame import (
    FREEZE_AT_SUNDAY,
    N_GAMES,
    SEASON,
    SUNDAY_GAMEDAY,
    WEEK,
    fits_with_floor,
    mini_strategies,
)

_RUN_INSTANT = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
_UNSCHEDULED_WEEK = 9

_GAME_IDS = [
    f"{SEASON}_W{WEEK:02d}_A{index:02d}@H{index:02d}" for index in range(N_GAMES)
]
_CAUGHT = _GAME_IDS[1]
_CLEAN = [game_id for game_id in _GAME_IDS if game_id != _CAUGHT]

_MODEL_VALUES: dict[str, dict[str, float]] = {
    "wp": {"model_prob": 0.61},
    "ats": {"model_spread": -3.5},
    "ou": {"model_total": 41.0},
}

# The five functions the exclusion threads through, in call order on each path.
_THREADED_FUNCTIONS = (
    "generate_weekly_bet_list",
    "build_weekly_decision_frame",
    "build_weekly_candidates",
    "select_games_for_decision_instant",
    "build_freeze_instant_candidates",
)


def _write_sandbox(root: Path) -> tuple[Path, Path, Path]:
    """A one-week silver schedule, a pre-lock odds snapshot and a gold matrix per target."""
    silver = root / "silver"
    gold = root / "gold"
    silver.mkdir(parents=True)
    gold.mkdir(parents=True)

    pd.DataFrame(
        {
            "game_id": _GAME_IDS,
            "season": [SEASON] * N_GAMES,
            "week": [WEEK] * N_GAMES,
            "kickoff_et": [pd.Timestamp(f"{SUNDAY_GAMEDAY}T17:00:00+00:00")] * N_GAMES,
        }
    ).to_parquet(silver / "games.parquet", index=False)

    pd.DataFrame(
        {
            "game_id": _GAME_IDS,
            "snapshot_ts": [FREEZE_AT_SUNDAY] * N_GAMES,
            # The recorded capture time: the only information time (owner ruling 2026-09-22).
            "created_at": pd.to_datetime([FREEZE_AT_SUNDAY] * N_GAMES, utc=True),
            "ml_home": [-130.0] * N_GAMES,
            "ml_away": [110.0] * N_GAMES,
            "spread": [-2.5] * N_GAMES,
            "total": [45.0] * N_GAMES,
            "sportsbook": ["consensus"] * N_GAMES,
            "is_live": [False] * N_GAMES,
        }
    ).to_parquet(silver / "odds_snapshot.parquet", index=False)

    gold_frame = pd.DataFrame(
        {"game_id": _GAME_IDS, "season": [SEASON] * N_GAMES, "week": [WEEK] * N_GAMES}
    )
    for target in CANONICAL_TARGETS:
        gold_frame.to_parquet(gold / f"features_{target}.parquet", index=False)

    return silver, gold, root / "artifacts"


class _ScoringRecorder:
    """Stands in for ``score_deployed_artifacts`` and records every game it was asked to score."""

    def __init__(self) -> None:
        self.scored: dict[str, list[str]] = {}

    def __call__(
        self, target: str, *, gold_df: pd.DataFrame, artifacts_dir: Path
    ) -> pd.DataFrame:
        self.scored.setdefault(target, []).extend(gold_df["game_id"].astype(str))
        scored = gold_df[["game_id", "season", "week"]].copy()
        for column, value in _MODEL_VALUES[target].items():
            scored[column] = value
        return scored


@pytest.fixture
def sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, _ScoringRecorder]:
    silver, gold, artifacts = _write_sandbox(tmp_path)
    recorder = _ScoringRecorder()
    monkeypatch.setattr(diagnose_module, "score_deployed_artifacts", recorder)
    return silver, gold, artifacts, recorder


def _decide(silver: Path, gold: Path, artifacts: Path, **kwargs: Any) -> pd.DataFrame:
    return build_weekly_decision_frame(
        SEASON,
        WEEK,
        artifacts_dir=artifacts,
        gold_dir=gold,
        silver_dir=silver,
        fits=fits_with_floor(0.0),
        strategies=mini_strategies(),
        now=_RUN_INSTANT,
        **kwargs,
    )


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["game_id", "target"]).reset_index(drop=True)


class TestTheInterfaceIsThreadedThroughAllFive:
    """Checking two of the five would pass while the weekly path silently DROPPED the set."""

    @pytest.mark.parametrize("name", _THREADED_FUNCTIONS)
    def test_each_function_takes_a_keyword_only_empty_frozenset(
        self, name: str
    ) -> None:
        parameter = inspect.signature(getattr(weekly_bet_list, name)).parameters[
            "excluded_game_ids"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == frozenset()
        assert isinstance(parameter.default, frozenset)

    def test_the_weekly_entry_point_passes_the_set_to_the_decision_seam(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Link 1 -> 2, by a recording stub: the entry point reaches candidates ONLY via 2."""
        import json

        from api.cache import BET_LIST_COLUMNS
        from tests.fixtures.decision_frame import chain_fit_record

        fit_path = tmp_path / "verdict.json"
        fit_path.write_text(json.dumps(chain_fit_record(0.0)), encoding="utf-8")
        seen: list[frozenset[str]] = []

        def recorder(season: int, week: int, **kwargs: Any) -> pd.DataFrame:
            seen.append(kwargs.get("excluded_game_ids"))
            return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))

        monkeypatch.setattr(weekly_bet_list, "build_weekly_decision_frame", recorder)
        weekly_bet_list.generate_weekly_bet_list(
            SEASON,
            WEEK,
            output_dir=tmp_path / "bet_list",
            chain_fit_path=fit_path,
            now=_RUN_INSTANT,
            excluded_game_ids=frozenset({_CAUGHT}),
        )

        assert seen == [frozenset({_CAUGHT})], seen


class TestTheSpineDropHappensBeforeScoringAndPricing:
    """Link 3: the excluded game never reaches the odds join or the scorer."""

    def test_the_excluded_game_is_never_scored_for_any_target(self, sandbox) -> None:
        silver, gold, artifacts, recorder = sandbox
        build_weekly_candidates(
            SEASON,
            WEEK,
            artifacts_dir=artifacts,
            gold_dir=gold,
            silver_dir=silver,
            excluded_game_ids=frozenset({_CAUGHT}),
        )

        assert set(recorder.scored) == set(CANONICAL_TARGETS)
        for target, scored in recorder.scored.items():
            assert _CAUGHT not in scored, f"{_CAUGHT} was scored for {target!r}"
            assert set(scored) == set(_CLEAN), target

    def test_the_excluded_game_is_absent_from_candidates_and_the_spine(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        candidates, schedule = build_weekly_candidates(
            SEASON,
            WEEK,
            artifacts_dir=artifacts,
            gold_dir=gold,
            silver_dir=silver,
            excluded_game_ids=frozenset({_CAUGHT}),
        )

        assert _CAUGHT not in set(candidates["game_id"]), "the odds join priced it"
        assert _CAUGHT not in set(schedule["game_id"]), "it is still on the spine"
        assert set(schedule["game_id"]) == set(_CLEAN)


class TestNoRowOfAnyKindAndTheCleanGamesAreUntouched:
    """The binding D33.2-05 claims, on the stamped decision frame."""

    def test_the_excluded_game_has_zero_rows_live_or_suppressed(self, sandbox) -> None:
        silver, gold, artifacts, _recorder = sandbox
        frame = _decide(silver, gold, artifacts, excluded_game_ids=frozenset({_CAUGHT}))

        assert not frame.empty, "the frame came back empty for an unrelated reason"
        assert _CAUGHT not in set(frame["game_id"]), (
            "the excluded game still has a row; a suppressed row is still a written row"
        )
        assert set(frame["game_id"]) == set(_CLEAN)

    def test_every_clean_game_decision_is_identical_to_the_unexcluded_run(
        self, sandbox
    ) -> None:
        """One skipped game costs no clean game its prediction (D33.2-05).

        Every column of every clean game's row is identical to the unexcluded run EXCEPT
        ``stake_units``, and that one difference is the pre-registered sizing working, not a
        leak: the Kelly stakes pass through ONE pooled 10% weekly cap over the week's bets
        (D31-02). With four games the twelve bets hit the cap and are scaled down; with the
        caught game skipped, nine bets fit under it and each takes its uncapped stake. So a
        skipped game can only FREE room for the clean games -- asserted below as "never
        smaller" -- and it never removes, re-sides or re-prices one.
        """
        silver, gold, artifacts, _recorder = sandbox
        baseline = _decide(silver, gold, artifacts)
        excluded = _decide(
            silver, gold, artifacts, excluded_game_ids=frozenset({_CAUGHT})
        )
        assert _CAUGHT in set(baseline["game_id"]), (
            "the control run has no row for the game either, so the exclusion proves nothing"
        )

        clean_baseline = _sorted(baseline[baseline["game_id"].isin(_CLEAN)])
        clean_excluded = _sorted(excluded)
        pooled_sizing = ["stake_units"]
        pd.testing.assert_frame_equal(
            clean_baseline.drop(columns=pooled_sizing),
            clean_excluded.drop(columns=pooled_sizing),
        )
        assert (
            clean_excluded["stake_units"] >= clean_baseline["stake_units"] - 1e-12
        ).all(), "skipping a game SHRANK a clean game's stake"

    def test_an_empty_exclusion_set_is_byte_identical_to_the_default(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        pd.testing.assert_frame_equal(
            _sorted(_decide(silver, gold, artifacts)),
            _sorted(_decide(silver, gold, artifacts, excluded_game_ids=frozenset())),
        )


class TestAllExcludedIsNotUnscheduled:
    """SPEC R3: an all-caught day finishes with skips, never as "no games"."""

    def test_excluding_every_game_returns_empty_and_scores_nothing(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, recorder = sandbox
        candidates, schedule = build_weekly_candidates(
            SEASON,
            WEEK,
            artifacts_dir=artifacts,
            gold_dir=gold,
            silver_dir=silver,
            excluded_game_ids=frozenset(_GAME_IDS),
        )
        assert candidates.empty
        assert schedule.empty
        assert recorder.scored == {}, "an all-excluded week was still scored"

    def test_the_all_excluded_decision_frame_is_empty_and_does_not_raise(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        frame = _decide(silver, gold, artifacts, excluded_game_ids=frozenset(_GAME_IDS))
        assert frame.empty

    def test_a_week_with_no_scheduled_games_still_refuses(self, sandbox) -> None:
        """The discriminating control: exclusion must not turn a broken store into a quiet week."""
        silver, gold, artifacts, _recorder = sandbox
        with pytest.raises(ValueError, match="no scheduled games"):
            build_weekly_candidates(
                SEASON,
                _UNSCHEDULED_WEEK,
                artifacts_dir=artifacts,
                gold_dir=gold,
                silver_dir=silver,
                excluded_game_ids=frozenset(_GAME_IDS),
            )


class TestTheWeeklyListPricesTheLatestPreLockCapture:
    """33.2 review A WR-01 / C1 CR-02, on the REAL candidate builder.

    Live captures accumulate and all carry ``snapshot_ts = lock``. The bet must be priced at the
    latest capture known at the lock, and the selector's freshness fence must see that capture's
    REAL time -- never the lock label, which always passes.
    """

    def test_the_lock_day_capture_prices_the_game_and_its_capture_time_is_carried(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        lock = pd.Timestamp("2026-09-19T22:00:00Z")  # Saturday 18:00 ET, Sunday games
        live = _GAME_IDS[0]
        stored = pd.read_parquet(silver / "odds_snapshot.parquet")
        captures = pd.DataFrame(
            {
                "game_id": [live] * 3,
                "snapshot_ts": [lock.isoformat()] * 3,
                "created_at": pd.to_datetime(
                    [
                        "2026-09-16T15:00:00Z",  # opening line
                        "2026-09-19T21:00:00Z",  # lock day, before the lock
                        "2026-09-19T23:00:00Z",  # after the lock
                    ],
                    utc=True,
                ),
                "ml_home": [-130.0] * 3,
                "ml_away": [110.0] * 3,
                "spread": [-2.5, -4.0, -9.0],
                "total": [45.0] * 3,
                "sportsbook": ["draftkings"] * 3,
                "is_live": [False] * 3,
            }
        )
        others = stored[stored["game_id"] != live]
        pd.concat([captures, others], ignore_index=True).to_parquet(
            silver / "odds_snapshot.parquet", index=False
        )

        candidates, _schedule = build_weekly_candidates(
            SEASON, WEEK, artifacts_dir=artifacts, gold_dir=gold, silver_dir=silver
        )
        row = candidates[
            (candidates["game_id"] == live) & (candidates["target"] == "ats")
        ].iloc[0]
        assert row["closing_spread"] == -4.0
        assert pd.Timestamp(row["snapshot_ts"]) == pd.Timestamp("2026-09-19T21:00:00Z")


class TestAGameWithNoOddsRowGetsNoBet:
    """Batch 1a follow-up: one slate game with NO odds row must not sink the week's list."""

    def test_the_unpriced_game_is_suppressed_and_every_other_game_is_decided(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        unpriced = _GAME_IDS[0]
        stored = pd.read_parquet(silver / "odds_snapshot.parquet")
        stored[stored["game_id"] != unpriced].to_parquet(
            silver / "odds_snapshot.parquet", index=False
        )

        frame = _decide(silver, gold, artifacts)

        unpriced_rows = frame[frame["game_id"] == unpriced]
        assert set(unpriced_rows["status"]) == {"suppressed"}
        assert set(unpriced_rows["rejection_reason"]) == {"missing_snapshot"}
        assert set(frame["game_id"]) == set(_GAME_IDS)
        assert (frame.loc[frame["game_id"] != unpriced, "status"] == "live").any()

    def test_a_week_with_no_odds_at_all_places_no_bet_and_does_not_raise(
        self, sandbox
    ) -> None:
        silver, gold, artifacts, _recorder = sandbox
        stored = pd.read_parquet(silver / "odds_snapshot.parquet")
        stored.iloc[0:0].to_parquet(silver / "odds_snapshot.parquet", index=False)

        frame = _decide(silver, gold, artifacts)

        assert set(frame["game_id"]) == set(_GAME_IDS)
        assert set(frame["status"]) == {"suppressed"}
        assert set(frame["rejection_reason"]) == {"missing_snapshot"}


class TestTheBetListIsNeverPublishedAfterItsDeadline:
    """33.2 review C2 WR-08: a list ready after the lock writes nothing, and says so."""

    def _generate(self, tmp_path: Path, monkeypatch, publish_by):
        import json

        from api.cache import BET_LIST_COLUMNS
        from tests.fixtures.decision_frame import chain_fit_record

        fit_path = tmp_path / "verdict.json"
        fit_path.write_text(json.dumps(chain_fit_record(0.0)), encoding="utf-8")
        monkeypatch.setattr(
            weekly_bet_list,
            "build_weekly_decision_frame",
            lambda *_a, **_k: pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS)),
        )
        return weekly_bet_list.generate_weekly_bet_list(
            SEASON,
            WEEK,
            output_dir=tmp_path / "bet_list",
            chain_fit_path=fit_path,
            now=_RUN_INSTANT,
            publish_by=publish_by,
        )

    def test_a_list_ready_after_the_deadline_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passed = datetime(2020, 1, 1, tzinfo=UTC)
        with pytest.raises(weekly_bet_list.PublishDeadlinePassedError):
            self._generate(tmp_path, monkeypatch, passed)
        assert not (tmp_path / "bet_list").exists()

    def test_a_list_ready_before_the_deadline_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ahead = datetime(2100, 1, 1, tzinfo=UTC)
        self._generate(tmp_path, monkeypatch, ahead)
        assert (tmp_path / "bet_list").exists()
