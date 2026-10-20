"""The per-generation re-measure of the bet rule and the blend (Phase 37 Plan 37-09; SPEC R4, D-22).

WHAT THIS PINS
--------------
* D-22 folds: one fold per historical season 2017-2024 (cutoff = that season's first kickoff,
  predicting that season) and one fold per finished 2026 week before the generation's cutoff
  (cutoff = the week's first kickoff, predicting that week's finished games before the generation
  cutoff).
* Edge R4 adjacency: every predicted game kicked off at or after its fold's cutoff and before the
  generation's cutoff, and no fold's model trained on a game it predicts.
* The prediction frames: one row per predicted game, finite, WP inside [0, 1].

Every store is the synthetic fixture under a temporary directory (``tests/fixtures/refit_gold.py``)
and every fit is the one-thread ``refit.training`` fit; nothing here reads or writes ``data/``,
``artifacts/`` or ``ledger/``. The walk-forward runs ONCE per module (48 small fits).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from refit.remeasure import FoldSpec, fold_specs, walk_forward_predictions

from backtest.refit_preregistration_2026 import HISTORICAL_FOLD_SEASONS, REFIT_SEASON
from forward_ledger.settle import realized_values_from_scores
from refit import remeasure
from tests.fixtures.refit_gold import RefitFixture, build_refit_fixture

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")


@dataclass(frozen=True)
class _WalkForward:
    """The fixture store, its games, the fold specs and one walk-forward run with its spy."""

    store: RefitFixture
    games: pd.DataFrame
    specs: tuple[FoldSpec, ...]
    predictions: dict[str, pd.DataFrame]
    trained_ids: dict[tuple[str, datetime], frozenset[str]]


@pytest.fixture(scope="module")
def walk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_WalkForward]:
    store = build_refit_fixture(tmp_path_factory.mktemp("remeasure"))
    games = pd.read_parquet(store.silver_dir / "games.parquet")
    trained: dict[tuple[str, datetime], frozenset[str]] = {}
    real_fit = remeasure.fit_target_at_cutoff

    def spy(
        target: str,
        gold: pd.DataFrame,
        frame: pd.DataFrame,
        cutoff: datetime,
        **kw: Any,
    ):
        fit = real_fit(target, gold, frame, cutoff, **kw)
        trained[(target, cutoff)] = frozenset(fit.eligible.frame["game_id"].astype(str))
        return fit

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remeasure, "fit_target_at_cutoff", spy)
        predictions = walk_forward_predictions(
            store.gold_dir, games, store.cutoff_utc, snapshots=store.snapshots
        )
    yield _WalkForward(
        store=store,
        games=games,
        specs=fold_specs(games, store.cutoff_utc),
        predictions=predictions,
        trained_ids=trained,
    )


def _kickoffs(games: pd.DataFrame) -> pd.Series:
    return pd.Series(
        pd.to_datetime(games["kickoff_et"], utc=True).to_numpy(),
        index=games["game_id"].astype(str),
    )


def _finished(games: pd.DataFrame) -> pd.Series:
    home = pd.to_numeric(games["home_score"], errors="coerce")
    away = pd.to_numeric(games["away_score"], errors="coerce")
    return home.notna() & away.notna()


def test_fold_specs(walk: _WalkForward) -> None:
    """D-22: eight season folds 2017-2024, then one fold per 2026 week begun before the cutoff."""
    games, cutoff = walk.games, walk.store.cutoff_utc
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    finished = _finished(games)

    historical = walk.specs[: len(HISTORICAL_FOLD_SEASONS)]
    assert len(historical) == 8
    for spec, season in zip(historical, HISTORICAL_FOLD_SEASONS, strict=True):
        in_season = games["season"] == season
        assert spec.cutoff_utc == kickoff[in_season].min().to_pydatetime()
        assert set(spec.predict_game_ids) == set(
            games.loc[in_season & finished, "game_id"].astype(str)
        )

    weekly = walk.specs[len(HISTORICAL_FOLD_SEASONS) :]
    season_games = games["season"] == REFIT_SEASON
    first_kickoff = kickoff[season_games].groupby(games.loc[season_games, "week"]).min()
    expected_weeks = sorted(
        int(week)
        for week, first in first_kickoff.items()
        if first < pd.Timestamp(cutoff)
    )
    assert expected_weeks == list(range(1, 9))  # week 9's first game is AT the cutoff
    assert len(weekly) == len(expected_weeks)
    for spec, week in zip(weekly, expected_weeks, strict=True):
        assert spec.cutoff_utc == first_kickoff[week].to_pydatetime()
        in_week = season_games & (games["week"] == week)
        expected = games.loc[
            in_week & finished & (kickoff < pd.Timestamp(cutoff)), "game_id"
        ].astype(str)
        assert set(spec.predict_game_ids) == set(expected)

    every_predicted = {gid for spec in walk.specs for gid in spec.predict_game_ids}
    assert walk.store.at_cutoff_game_id not in every_predicted
    assert walk.store.postponed_game_id not in every_predicted
    assert walk.store.before_cutoff_game_id in every_predicted
    assert all(spec.cutoff_utc.tzinfo is not None for spec in walk.specs)


def test_no_prediction_from_a_model_that_saw_its_game(walk: _WalkForward) -> None:
    """Edge R4 adjacency: kickoff in [fold cutoff, generation cutoff) and never a training row."""
    kickoff = _kickoffs(walk.games)
    cutoff = pd.Timestamp(walk.store.cutoff_utc)
    for target in _TARGETS:
        frame = walk.predictions[target]
        assert not frame.empty
        for fold_label, rows in frame.groupby("fold"):
            fold_cutoff = pd.Timestamp(rows["fold_cutoff_utc"].iloc[0])
            ids = rows["game_id"].astype(str)
            assert (kickoff[ids].to_numpy() >= fold_cutoff.to_datetime64()).all(), (
                fold_label
            )
            assert (kickoff[ids].to_numpy() < cutoff.to_datetime64()).all(), fold_label
            trained = walk.trained_ids[(target, fold_cutoff.to_pydatetime())]
            assert trained, fold_label
            assert trained.isdisjoint(set(ids)), fold_label
    # One model per fold per target, each at its fold's own cutoff.
    assert len(walk.trained_ids) == len(_TARGETS) * len(walk.specs)


def test_predictions_shape(walk: _WalkForward) -> None:
    """game_id, season, week, prediction, actual: one row per predicted game, finite."""
    labels = realized_values_from_scores(walk.games)
    gold_ids = {
        target: set(
            pd.read_parquet(walk.store.gold_dir / f"features_{target}.parquet")[
                "game_id"
            ].astype(str)
        )
        for target in _TARGETS
    }
    for target in _TARGETS:
        frame = walk.predictions[target]
        assert {"game_id", "season", "week", "prediction", "actual"} <= set(
            frame.columns
        )
        assert not frame["game_id"].duplicated().any()
        expected = {
            gid for spec in walk.specs for gid in spec.predict_game_ids
        } & gold_ids[target]
        assert set(frame["game_id"]) == expected
        values = frame["prediction"].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        if target == "wp":
            assert ((values >= 0.0) & (values <= 1.0)).all()
        assert frame["actual"].to_numpy(dtype=float).tolist() == [
            labels[target][gid] for gid in frame["game_id"]
        ]
        assert set(frame["season"].astype(int)) == {
            *HISTORICAL_FOLD_SEASONS,
            REFIT_SEASON,
        }
    # The game with a score in silver but no gold row cannot be predicted by any fold.
    assert walk.store.no_gold_game_id not in set(walk.predictions["wp"]["game_id"])
    assert datetime(2026, 11, 3, 22, tzinfo=UTC) == walk.store.cutoff_utc
