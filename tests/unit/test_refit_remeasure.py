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
* The extended corpus: a finished 2026 game whose only line is timed after its lock is excluded
  and counted, and the closing-line guard runs on a fit-shaped projection of the corpus.
* The bet-rule numbers through the frozen estimators: the D-27 lean over the whole pool (owner
  ruling A: a game without an owned pre-lock line outside 2017-2019 enters neither the lean nor
  the error spread), the D-28 prior-only SD correction, the D-29 WP gate, the D-30 2026 slope, the
  D-36 edge bands, the 100-row floor and the fixed EV floors.
* The content-addressed bet-rule record in ``load_frozen_chain_fit``'s shape.

Every store is the synthetic fixture under a temporary directory (``tests/fixtures/refit_gold.py``)
or a hand-built frame, and every fit is the one-thread ``refit.training`` fit; nothing here reads
or writes ``data/``, ``artifacts/`` or ``ledger/``. The walk-forward runs ONCE per module (48 small
fits).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtest.refit_preregistration_2026 import (
    BIAS_SEED_SEASONS,
    EV_FLOORS,
    HISTORICAL_FOLD_SEASONS,
    HISTORICAL_TUNE_SEASONS,
    MARKET_CONVERTER_ID,
    OU_TOTAL_FLOOR,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    WP_EDGE_ANCHOR,
)
from forward_ledger.settle import realized_values_from_scores
from models.blending_data import (
    EXCLUDED_COLUMNS,
    EXCLUSION_NO_PRELOCK_LINE,
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
    TUNING_FRAME_COLUMNS,
    PrelockTuningCorpus,
)
from models.market_probability import ClosingLineInFitError
from refit import remeasure
from refit.records import read_bet_rule_record, write_bet_rule_record
from refit.remeasure import (
    BetRuleNumbers,
    FoldSpec,
    RemeasureRefusal,
    assert_corpus_is_prelock,
    bet_rule_payload,
    extended_corpus,
    extended_slopes,
    fold_specs,
    measure_bet_rule,
    walk_forward_predictions,
)
from scripts.derive_cold_start_constants import derive_threshold_pairs
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
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    kickoff.index = pd.Index(games["game_id"].astype(str))
    return kickoff


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
            assert (kickoff[ids] >= fold_cutoff).all(), fold_label
            assert (kickoff[ids] < cutoff).all(), fold_label
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


# ---------------------------------------------------------------------------
# Task 2: the extended corpus
# ---------------------------------------------------------------------------

_GENERATION_CUTOFF = datetime(2026, 11, 3, 22, 0, tzinfo=UTC)


def _silver_for_the_corpus(root: Path) -> pd.DataFrame:
    """A tiny silver store: one owned 2020-2024 game a season, plus the 2026 edge games."""
    root.mkdir(parents=True, exist_ok=True)
    games: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for season in HISTORICAL_TUNE_SEASONS:
        kickoff = datetime(season, 9, 13, 17, 0, tzinfo=UTC)
        game_id = f"{season}_01_AAA_BBB"
        games.append(
            {
                "game_id": game_id,
                "season": season,
                "week": 1,
                "game_type": "REG",
                "kickoff_et": kickoff,
                "home_score": 24.0,
                "away_score": 17.0,
            }
        )
        captured = kickoff - timedelta(days=2)
        timeline.append(
            {
                "game_id": game_id,
                "snapshot_ts": captured,
                "spread": 3.5,
                "total": 44.0,
                "created_at": captured,
            }
        )

    def game_2026(game_id: str, week: int, kickoff: datetime, scored: bool) -> None:
        games.append(
            {
                "game_id": game_id,
                "season": REFIT_SEASON,
                "week": week,
                "game_type": "REG",
                "kickoff_et": kickoff,
                "home_score": 20.0 if scored else np.nan,
                "away_score": 13.0 if scored else np.nan,
            }
        )

    sunday = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)  # lock: Sat 2026-10-31 22:00 UTC
    game_2026(
        "2026_08_AAA_BBB", 8, sunday, True
    )  # a pre-lock line and a post-lock line
    game_2026("2026_08_CCC_DDD", 8, sunday, True)  # only a post-lock line
    game_2026("2026_08_EEE_FFF", 8, datetime(2026, 11, 2, 1, 15, tzinfo=UTC), False)
    game_2026("2026_09_GGG_HHH", 9, datetime(2026, 11, 6, 1, 15, tzinfo=UTC), False)
    game_2026("2026_07_III_JJJ", 7, datetime(2026, 10, 25, 17, 0, tzinfo=UTC), True)

    def line(
        game_id: str, created: datetime, spread: float, total: float
    ) -> dict[str, Any]:
        return {
            "game_id": game_id,
            "sportsbook": "consensus",
            "ml_home": -150.0,
            "ml_away": 130.0,
            "spread": spread,
            "total": total,
            "snapshot_ts": created,
            "created_at": created,
        }

    snapshot = [
        line("2026_08_AAA_BBB", datetime(2026, 10, 30, 15, 0, tzinfo=UTC), -2.5, 47.0),
        line("2026_08_AAA_BBB", datetime(2026, 11, 1, 15, 0, tzinfo=UTC), -6.0, 49.0),
        line("2026_08_CCC_DDD", datetime(2026, 11, 1, 12, 0, tzinfo=UTC), 1.5, 41.0),
        line("2026_08_EEE_FFF", datetime(2026, 10, 30, 15, 0, tzinfo=UTC), 3.0, 45.0),
    ]
    frame = pd.DataFrame(games)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    frame.to_parquet(root / "games.parquet", index=False)
    pd.DataFrame(timeline).to_parquet(root / "odds_timeline.parquet", index=False)
    pd.DataFrame(snapshot).to_parquet(root / "odds_snapshot.parquet", index=False)
    return frame


def test_a_2026_game_without_a_prelock_line_is_excluded_and_counted(
    tmp_path: Path,
) -> None:
    """Edge R4 adjacency: no line at or before the lock -> excluded, counted, never filled."""
    silver = tmp_path / "silver"
    games = _silver_for_the_corpus(silver)
    corpus = extended_corpus(silver, games, _GENERATION_CUTOFF)

    frame = corpus.frame.set_index("game_id")
    assert list(corpus.frame.columns) == list(TUNING_FRAME_COLUMNS)
    assert set(frame.index) == {
        *(f"{season}_01_AAA_BBB" for season in HISTORICAL_TUNE_SEASONS),
        "2026_08_AAA_BBB",
    }
    # The 2026 line is the latest one timed at or before the lock, on the home-margin scale as
    # odds_snapshot stores it (identity); the timeline's is flipped by its one named function.
    assert frame.loc["2026_08_AAA_BBB", "market_spread"] == -2.5
    assert frame.loc["2026_08_AAA_BBB", "market_total"] == 47.0
    assert frame.loc["2020_01_AAA_BBB", "market_spread"] == -3.5
    assert (corpus.frame["snapshot_ts"] <= corpus.frame["lock"]).all()

    excluded = corpus.excluded.set_index("game_id")
    assert list(corpus.excluded.columns) == list(EXCLUDED_COLUMNS)
    assert excluded.loc["2026_08_CCC_DDD", "reason"] == EXCLUSION_NO_PRELOCK_LINE
    assert excluded.loc["2026_07_III_JJJ", "reason"] == EXCLUSION_NO_PRELOCK_LINE
    # Unfinished, or at/after the cutoff: neither in the corpus nor counted as an exclusion.
    assert "2026_08_EEE_FFF" not in excluded.index
    assert "2026_09_GGG_HHH" not in excluded.index

    assert_corpus_is_prelock(corpus.frame, games)
    planted = corpus.frame.copy()
    row = planted.index[planted["game_id"] == "2026_08_AAA_BBB"][0]
    planted.loc[row, "snapshot_ts"] = planted.loc[row, "lock"] + pd.Timedelta(seconds=1)
    with pytest.raises(ClosingLineInFitError):
        assert_corpus_is_prelock(planted, games)


# ---------------------------------------------------------------------------
# Task 2: the bet-rule numbers
# ---------------------------------------------------------------------------

_CURRENT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "wp": (0.05, 0.02),
    "ats": (1.0, 0.5),
    "ou": (0.02, 0.01),
}
_CONVERTER: dict[str, Any] = {
    "slope_beta": 0.151,
    "walk_forward_slopes": {"2021": 0.158, "2022": 0.136, "2023": 0.143, "2024": 0.143},
}
_NO_LINE_IDS: tuple[str, ...] = ("2022_NOLINE", "2026_NOLINE")


@dataclass(frozen=True)
class _HandBuilt:
    predictions: dict[str, pd.DataFrame]
    corpus: PrelockTuningCorpus


def _hand_built(
    games_per_season: int = 30,
    *,
    wp_prediction: float | None = None,
    seed: int = 9,
) -> _HandBuilt:
    """Predictions for seed + tune + 2026 games, and a corpus of the tune and 2026 games.

    Two no-line games (one 2022, one 2026) carry a planted residual of 100 (ATS / O/U; WP 1).
    With *wp_prediction* every WP prediction is that constant while the home team wins about a
    fifth of the time, a frame the frozen calibration gate must fail.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    for season in (*BIAS_SEED_SEASONS, *HISTORICAL_TUNE_SEASONS, REFIT_SEASON):
        for index in range(games_per_season):
            margin = float(rng.normal(0.0, 13.0))
            total = float(rng.normal(44.0, 10.0))
            p = float(rng.uniform(0.15, 0.85))
            win_rate = p if wp_prediction is None else 0.2
            rows.append(
                {
                    "game_id": f"{season}_{index:03d}",
                    "season": season,
                    "week": index % 8 + 1,
                    "wp": (
                        p if wp_prediction is None else wp_prediction,
                        float(rng.uniform() < win_rate),
                    ),
                    "ats": (0.3 * margin + float(rng.normal(1.0, 8.0)), margin),
                    "ou": (44.0 + float(rng.normal(0.5, 5.0)), total),
                }
            )
            if season in BIAS_SEED_SEASONS:
                continue
            lock = datetime(season, 9, 12, 22, 0, tzinfo=UTC) + timedelta(days=index)
            corpus_rows.append(
                {
                    "game_id": f"{season}_{index:03d}",
                    "season": season,
                    "week": index % 8 + 1,
                    "game_type": "REG",
                    "snapshot_ts": lock - timedelta(hours=3),
                    "lock": lock,
                    "created_at": lock - timedelta(hours=3),
                    "market_spread": float(rng.normal(0.0, 6.0)),
                    "market_total": 44.0 + float(rng.normal(0.0, 4.0)),
                }
            )
    for game_id in _NO_LINE_IDS:
        rows.append(
            {
                "game_id": game_id,
                "season": int(game_id[:4]),
                "week": 1,
                "wp": (0.0, 1.0),
                "ats": (0.0, 100.0),
                "ou": (40.0, 140.0),
            }
        )
    predictions = {
        target: pd.DataFrame(
            {
                "game_id": [row["game_id"] for row in rows],
                "season": [row["season"] for row in rows],
                "week": [row["week"] for row in rows],
                "prediction": [row[target][0] for row in rows],
                "actual": [row[target][1] for row in rows],
            }
        )
        for target in _TARGETS
    }
    excluded = pd.DataFrame(
        {
            "game_id": list(_NO_LINE_IDS),
            "season": [int(gid[:4]) for gid in _NO_LINE_IDS],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "reason": [EXCLUSION_NO_PRELOCK_LINE, EXCLUSION_NO_PRELOCK_LINE],
            "detail": ["absent", "absent"],
        }
    )
    return _HandBuilt(
        predictions=predictions,
        corpus=PrelockTuningCorpus(
            frame=pd.DataFrame(corpus_rows).loc[:, list(TUNING_FRAME_COLUMNS)],
            excluded=excluded,
            unjoined_timeline_ids=(),
        ),
    )


def _measure(built: _HandBuilt) -> BetRuleNumbers:
    return measure_bet_rule(
        built.predictions,
        built.corpus,
        extended_slopes(_CONVERTER),
        current_thresholds=_CURRENT_THRESHOLDS,
    )


@pytest.fixture(scope="module")
def built() -> _HandBuilt:
    return _hand_built()


@pytest.fixture(scope="module")
def numbers(built: _HandBuilt) -> BetRuleNumbers:
    return _measure(built)


def _residual_pool(built: _HandBuilt, target: str) -> pd.DataFrame:
    """Seed seasons whole plus the corpus games, with ``residual = actual - prediction``."""
    frame = built.predictions[target]
    corpus_ids = set(built.corpus.frame["game_id"])
    pool = frame.loc[
        frame["season"].isin(BIAS_SEED_SEASONS) | frame["game_id"].isin(corpus_ids)
    ]
    return pool.assign(residual=pool["actual"] - pool["prediction"])


def test_lean_is_the_pooled_mean_of_the_whole_pool(
    built: _HandBuilt, numbers: BetRuleNumbers
) -> None:
    """D-27 + owner ruling A: seed seasons whole plus corpus games only; no-line games stay out."""
    for target in _TARGETS:
        pool = _residual_pool(built, target)
        assert not set(pool["game_id"]) & set(_NO_LINE_IDS)
        expected = float(np.mean(pool["residual"].to_numpy(dtype=float)))
        lean = numbers.season_bias_by_season[target][REFIT_SEASON]
        assert lean == pytest.approx(expected, abs=1e-12)
        assert numbers.lean_by_target[target] == lean

    # The planted residual-100 games move neither the lean nor the error spread.
    without = _HandBuilt(
        predictions={
            target: frame.loc[~frame["game_id"].isin(_NO_LINE_IDS)].reset_index(
                drop=True
            )
            for target, frame in built.predictions.items()
        },
        corpus=built.corpus,
    )
    clean = _measure(without)
    for target in _TARGETS:
        assert clean.lean_by_target[target] == numbers.lean_by_target[target]
        assert (
            clean.chain_fits[target].frozen_sd == numbers.chain_fits[target].frozen_sd
        )


def test_sd_uses_prior_only_correction_for_2026(
    built: _HandBuilt, numbers: BetRuleNumbers
) -> None:
    """D-28: each tune block corrected by the pooled mean of strictly prior seasons; WP none."""
    for target in ("ats", "ou"):
        pool = _residual_pool(built, target)
        by_season = {
            int(season): rows["residual"].to_numpy(dtype=float)
            for season, rows in pool.groupby("season")
        }
        corrected: list[float] = []
        for season in (*HISTORICAL_TUNE_SEASONS, REFIT_SEASON):
            prior = np.concatenate([v for s, v in by_season.items() if s < season])
            corrected.extend((by_season[season] - float(np.mean(prior))).tolist())
        expected_sd = float(np.std(np.asarray(corrected), ddof=1))
        assert numbers.chain_fits[target].frozen_sd == pytest.approx(
            expected_sd, abs=1e-12
        )
        prior_2026 = np.concatenate(
            [v for s, v in by_season.items() if s < REFIT_SEASON]
        )
        assert numbers.sd_correction_bias_by_season[target][
            REFIT_SEASON
        ] == pytest.approx(float(np.mean(prior_2026)), abs=1e-12)
    assert numbers.chain_fits["wp"].frozen_sd is None


def test_wp_gate_is_rerun() -> None:
    """D-29: a WP frame that fails the frozen gate records the verdict and the fallback trigger."""
    failing = _measure(_hand_built(wp_prediction=0.9))
    assert failing.calibration_gate_passed["wp"] is False
    assert isinstance(failing.fallback_trigger["wp"], str)
    assert failing.calibration_gate_passed["ats"] is None
    assert failing.calibration_gate_passed["ou"] is None


def test_wp_2026_uses_slope_beta(built: _HandBuilt, numbers: BetRuleNumbers) -> None:
    """D-30: 2026 WP markets through slope_beta; 2020 WP rows excluded as no_prior_fold_converter."""
    slopes = extended_slopes(_CONVERTER)
    assert slopes[REFIT_SEASON] == _CONVERTER["slope_beta"]
    assert {season: slopes[season] for season in (2021, 2022, 2023, 2024)} == {
        int(k): v for k, v in _CONVERTER["walk_forward_slopes"].items()
    }
    edges = numbers.edges.set_index("game_id")
    corpus = built.corpus.frame.set_index("game_id")
    rows_2026 = corpus.index[corpus["season"] == REFIT_SEASON]
    spread = corpus.loc[rows_2026, "market_spread"].to_numpy(dtype=float)
    expected = 1.0 / (1.0 + np.exp(-_CONVERTER["slope_beta"] * spread))
    np.testing.assert_allclose(
        edges.loc[rows_2026, "wp_market_prob"].to_numpy(dtype=float),
        expected,
        rtol=0,
        atol=1e-15,
    )
    rows_2020 = corpus.index[corpus["season"] == 2020]
    assert numbers.excluded_counts[EXCLUSION_NO_PRIOR_FOLD_CONVERTER] == len(rows_2020)
    assert numbers.excluded_counts[EXCLUSION_NO_PRELOCK_LINE] == len(_NO_LINE_IDS)
    assert edges.loc[rows_2020, "wp_market_prob"].isna().all()
    assert numbers.rows_by_target["wp"]["usable"] == len(corpus) - len(rows_2020)


def test_edge_bands_anchor_on_wp(built: _HandBuilt, numbers: BetRuleNumbers) -> None:
    """D-36: WP keeps 0.05 / 0.02; ATS and O/U are the frozen derivation over the edges."""
    assert numbers.edge_tier_thresholds["wp"] == WP_EDGE_ANCHOR
    corpus = built.corpus.frame.set_index("game_id")
    preds = {
        t: f.set_index("game_id")["prediction"] for t, f in built.predictions.items()
    }
    slopes = extended_slopes(_CONVERTER)
    ids = corpus.index
    betas = corpus["season"].map(slopes).astype(float)
    market_prob = 1.0 / (1.0 + np.exp(-betas * corpus["market_spread"]))
    edges = pd.DataFrame(
        {
            "season": corpus["season"],
            "ats_edge": preds["ats"][ids] - corpus["market_spread"],
            "ou_edge": (preds["ou"][ids] - corpus["market_total"])
            / corpus["market_total"].clip(lower=OU_TOTAL_FLOOR),
            "wp_edge": preds["wp"][ids] - market_prob,
        }
    )
    expected = derive_threshold_pairs(edges, _CURRENT_THRESHOLDS)
    for target in ("ats", "ou"):
        assert expected.thresholds[target] is not None
        assert numbers.edge_tier_thresholds[target] == expected.thresholds[target]


def test_fewer_than_100_usable_rows_refuses(built: _HandBuilt) -> None:
    """Edge R4 empty: 99 usable ATS rows is an integrity refusal naming ATS, never no bets."""
    frame = built.corpus.frame.sort_values(
        ["season", "game_id"], ignore_index=True
    ).iloc[:99]
    short = PrelockTuningCorpus(
        frame=frame, excluded=built.corpus.excluded, unjoined_timeline_ids=()
    )
    with pytest.raises(RemeasureRefusal) as refusal:
        _measure(_HandBuilt(predictions=built.predictions, corpus=short))
    assert refusal.value.check == "remeasure_complete"
    assert "ats" in str(refusal.value)


def _payload(numbers: BetRuleNumbers) -> dict[str, Any]:
    return bet_rule_payload(
        numbers,
        artifact_ids={
            "wp": "wp_20261103_220501",
            "ats": "ats_20261103_220502",
            "ou": "ou_20261103_220503",
            "blend": "blend_20261103_221000",
            "converter": MARKET_CONVERTER_ID,
        },
        cutoff_utc=_GENERATION_CUTOFF,
        list_week=8,
        date_index=1,
        blend_weights={"wp": 0.0, "ats": 0.12, "ou": 0.4},
    )


def test_floors_are_fixed(numbers: BetRuleNumbers) -> None:
    """R4: the EV floors in force are the record's fixed 0.0 / 0.0 / 0.01."""
    assert numbers.ev_floors == EV_FLOORS
    payload = _payload(numbers)
    assert {t: payload["tune_fit"][t]["ev_floor_t"] for t in _TARGETS} == EV_FLOORS


def _keys(value: object) -> set[str]:
    """Every mapping key anywhere inside *value*."""
    if isinstance(value, Mapping):
        return {str(k) for k in value} | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list | tuple):
        return {k for item in value for k in _keys(item)}
    return set()


def test_payload_loads_like_a_chain_fit(
    tmp_path: Path, numbers: BetRuleNumbers
) -> None:
    """R4: the content-addressed record carries tune_fit / targets in load_frozen_chain_fit's shape."""
    path, sha = write_bet_rule_record(tmp_path / "ledger", _payload(numbers))
    record = read_bet_rule_record(path)
    assert path.name == f"{sha}.json"
    assert record["record_kind"] == "refit_generation_bet_rule"
    assert record["refit_recipe"] == REFIT_RECIPE_ID
    assert set(record["artifact_ids"]) == {"wp", "ats", "ou", "blend", "converter"}
    assert record["cutoff_utc"] == _GENERATION_CUTOFF.isoformat()
    assert (record["list_week"], record["date_index"]) == (8, 1)
    assert record["preregistration_sha256"] == dict(remeasure.PREREGISTRATION_SHA256)
    for key in (
        "edge_tier_thresholds",
        "blend_weights",
        "windows",
        "excluded_counts",
        "rows_by_target",
    ):
        assert key in record
    # The stamp-site rule: a record never names a ledger stamp column.
    assert not {"recipe_id", "blend_id", "model_artifact_id"} & _keys(record)

    for target in _TARGETS:
        block = record["tune_fit"][target]
        assert isinstance(block["ev_floor_t"], float)
        assert block["frozen_sd"] is None or isinstance(block["frozen_sd"], float)
        biases = block["season_bias_by_season"]
        assert str(REFIT_SEASON) in biases
        assert all(
            isinstance(k, str) and isinstance(v, float) for k, v in biases.items()
        )
        assert biases[str(REFIT_SEASON)] == numbers.lean_by_target[target]
        assert "fallback_trigger" in record["targets"][target]
    assert isinstance(record["tune_fit"]["wp"]["calibration_gate_passed"], bool)
    assert record["tune_fit"]["ats"]["calibration_gate_passed"] is None
    assert record["tune_fit"]["wp"]["frozen_sd"] is None
    assert record["edge_tier_thresholds"]["wp"] == list(WP_EDGE_ANCHOR)
