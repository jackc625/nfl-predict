"""Unit tests for MarketBlender.tune_weights: ONE fixed weight per target (Plan 33.2-24).

WHAT CHANGED, AND WHY (D33.2-10)
--------------------------------
The tuner used to grid-search [0.50, 0.70] for the weight maximising mean CLOSING-LINE value on
SYNTHETIC predictions over 2010-2017, and a second tuner fitted a week-varying sigmoid on the
same synthetic data. Both are retired: a closing line did not exist at a game's lock
(D33.2-03), and scoring a blend against the very line it blends with rewards whichever end of
the allowed range the search can reach. The tuner now scores each candidate weight by the
model's OWN outcome loss -- WP log loss, ATS and O/U absolute error -- on the corrected models'
walk-forward predictions blended with the market's PRE-LOCK opinion, over [0.00, 1.00].

HOW EACH EARLIER CASE WAS CARRIED
---------------------------------
* weights in range / grid shape / records game counts / updates the blender config: carried,
  over the full [0, 1] grid (101 candidates);
* "uses only tuning seasons": carried as ``seasons_by_target``;
* "perfect model prefers a higher weight" / "random model a lower one": carried, against the
  OUTCOME -- a model that IS the outcome takes weight 1.0, a noise model a lower weight;
* "maximises mean CLV": REPLACED -- the fitted weight MINIMISES the outcome loss, and the loss
  at the fitted weight equals the grid minimum;
* temporal isolation ("raises on a holdout season"): carried as "refuses a season no owned
  pre-lock line covers" -- 2025 is the spent hold, 2017 the retired closing-line era;
* artifact save / provenance / latest.json / from_artifacts: carried, with ``update_latest``
  now defaulting to False and the provenance required;
* DELETED BY RULING: ``TestDynamicSyntheticPredictions`` (5), ``TestSigmoidObjective`` (6),
  ``TestRunDynamicBlendTuning`` (3), ``TestSigmoidTuningMetadata`` (2) and the
  ``--dynamic`` / ``--n-trials`` / ``--rng-seed`` cases of ``TestDynamicCLIFlags`` (6): their
  subject -- the synthetic-prediction sigmoid tuner and its flags -- is removed. The CLI's one
  surviving option is pinned below.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.blending import (
    BLEND_WEIGHT_GRID,
    BlendProvenance,
    BlendTuningError,
    BlendWeights,
    MarketBlender,
    TuningResult,
)

_CONVERTER_ID = "market_probability_20990101_000000"
_CONVERTER_SLOPE = 0.15


# ---------------------------------------------------------------------------
# Synthetic tuning frames
# ---------------------------------------------------------------------------


def _ats_frame(
    seasons: list[int], model_noise: float, market_noise: float, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for index in range(120):
            margin = float(rng.normal(0, 13))
            rows.append(
                {
                    "game_id": f"{season}_W{index % 18 + 1:02d}_G{index:03d}",
                    "season": season,
                    "week": index % 18 + 1,
                    "model_spread": margin + float(rng.normal(0, model_noise)),
                    "market_spread": margin + float(rng.normal(0, market_noise)),
                    "home_margin": margin,
                }
            )
    return pd.DataFrame(rows)


def _ou_frame(seasons: list[int], seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for index in range(120):
            total = float(rng.normal(45, 9))
            rows.append(
                {
                    "game_id": f"{season}_W{index % 18 + 1:02d}_G{index:03d}",
                    "season": season,
                    "week": index % 18 + 1,
                    "model_total": total + float(rng.normal(0, 8)),
                    "market_total": total + float(rng.normal(0, 7)),
                    "total_points": total,
                }
            )
    return pd.DataFrame(rows)


def _wp_frame(seasons: list[int], quality: str, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for index in range(120):
            win = int(rng.integers(0, 2))
            if quality == "perfect":
                model = 0.999 if win else 0.001
            elif quality == "noise":
                model = float(rng.uniform(0.3, 0.7))
            else:
                model = float(
                    np.clip((0.65 if win else 0.35) + rng.normal(0, 0.1), 0.02, 0.98)
                )
            market = float(
                np.clip((0.6 if win else 0.4) + rng.normal(0, 0.1), 0.02, 0.98)
            )
            rows.append(
                {
                    "game_id": f"{season}_W{index % 18 + 1:02d}_G{index:03d}",
                    "season": season,
                    "week": index % 18 + 1,
                    "model_prob": model,
                    "market_prob_oof": market,
                    "home_win": win,
                }
            )
    return pd.DataFrame(rows)


def _frames() -> dict[str, pd.DataFrame]:
    seasons = [2021, 2022, 2023, 2024]
    return {
        "wp": _wp_frame(seasons, "decent"),
        "ats": _ats_frame([2020, *seasons], model_noise=10.0, market_noise=8.0),
        "ou": _ou_frame([2020, *seasons]),
    }


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class TestTuneWeights:
    def test_weights_are_in_the_unit_interval(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        assert isinstance(result, TuningResult)
        for weight in (
            result.weights.wp_model_weight,
            result.weights.ats_model_weight,
            result.weights.ou_model_weight,
        ):
            assert 0.0 <= weight <= 1.0

    def test_the_grid_is_zero_to_one_in_hundredths(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        for target in ("wp", "ats", "ou"):
            weights = [w for w, _ in result.grid_by_target[target]]
            assert len(weights) == 101
            assert weights[0] == 0.0
            assert weights[-1] == 1.0
        assert len(BLEND_WEIGHT_GRID) == 101

    def test_the_fitted_weight_minimises_the_outcome_loss(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        for target in ("wp", "ats", "ou"):
            grid = result.grid_by_target[target]
            best = min(loss for _, loss in grid)
            assert result.loss_by_target[target] == pytest.approx(best, abs=1e-12)
            assert (
                result.loss_by_target[target]
                <= result.market_only_loss_by_target[target]
            )
            assert (
                result.loss_by_target[target]
                <= result.model_only_loss_by_target[target]
            )

    def test_the_objective_is_the_models_own_metric_on_the_outcome(self) -> None:
        """At weight 1 the ATS loss IS the model's own mean absolute error vs the margin."""
        frame = _ats_frame([2022], model_noise=5.0, market_noise=5.0)
        result = MarketBlender().tune_weights({"ats": frame})
        model_mae = float(np.mean(np.abs(frame["model_spread"] - frame["home_margin"])))
        market_mae = float(
            np.mean(np.abs(frame["market_spread"] - frame["home_margin"]))
        )
        assert result.model_only_loss_by_target["ats"] == pytest.approx(model_mae)
        assert result.market_only_loss_by_target["ats"] == pytest.approx(market_mae)
        assert result.objective_by_target == {"ats": "mean_absolute_error"}

    def test_the_wp_objective_is_log_loss(self) -> None:
        frame = _wp_frame([2022], "decent")
        result = MarketBlender().tune_weights({"wp": frame})
        p = np.clip(frame["market_prob_oof"].to_numpy(), 0.001, 0.999)
        y = frame["home_win"].to_numpy()
        expected = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        assert result.market_only_loss_by_target["wp"] == pytest.approx(expected)
        assert result.objective_by_target == {"wp": "log_loss"}

    def test_a_model_that_is_the_outcome_takes_the_top_weight(self) -> None:
        frame = _ats_frame([2022], model_noise=0.0, market_noise=6.0)
        result = MarketBlender().tune_weights({"ats": frame})
        assert result.weights.ats_model_weight == 1.0

    def test_a_noise_model_takes_a_lower_weight_than_a_perfect_one(self) -> None:
        noise = MarketBlender().tune_weights({"wp": _wp_frame([2022], "noise")})
        perfect = MarketBlender().tune_weights({"wp": _wp_frame([2022], "perfect")})
        assert noise.weights.wp_model_weight < perfect.weights.wp_model_weight

    def test_the_tuning_seasons_are_recorded_per_target(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        assert result.seasons_by_target["wp"] == [2021, 2022, 2023, 2024]
        assert result.seasons_by_target["ats"] == [2020, 2021, 2022, 2023, 2024]

    def test_the_game_counts_are_recorded(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        assert result.n_games == {"wp": 480, "ats": 600, "ou": 600}

    def test_each_season_records_the_weight_it_alone_would_choose(self) -> None:
        result = MarketBlender().tune_weights(_frames())
        assert set(result.season_best_weight_by_target["ats"]) == {
            2020,
            2021,
            2022,
            2023,
            2024,
        }
        for weight in result.season_best_weight_by_target["ats"].values():
            assert 0.0 <= weight <= 1.0

    def test_the_fit_updates_the_blender_config(self) -> None:
        blender = MarketBlender()
        result = blender.tune_weights(_frames())
        assert blender.config.weights == result.weights

    def test_ties_resolve_to_the_lowest_weight(self) -> None:
        """Model and market identical: every weight scores the same; the first one wins."""
        frame = _ou_frame([2022]).assign(market_total=lambda f: f["model_total"])
        result = MarketBlender().tune_weights({"ou": frame})
        assert result.weights.ou_model_weight == 0.0


class TestTheTuningWindow:
    """Temporal isolation, re-expressed: only seasons an owned pre-lock line covers."""

    def test_a_spent_hold_season_is_refused(self) -> None:
        frame = _ats_frame([2024, 2025], model_noise=5.0, market_noise=5.0)
        with pytest.raises(BlendTuningError, match="2025"):
            MarketBlender().tune_weights({"ats": frame})

    def test_a_closing_line_era_season_is_refused(self) -> None:
        frame = _ats_frame([2017, 2021], model_noise=5.0, market_noise=5.0)
        with pytest.raises(BlendTuningError, match="2017"):
            MarketBlender().tune_weights({"ats": frame})

    def test_the_owned_seasons_are_accepted(self) -> None:
        frame = _ats_frame(
            [2020, 2021, 2022, 2023, 2024], model_noise=5.0, market_noise=5.0
        )
        assert MarketBlender().tune_weights({"ats": frame}).n_games["ats"] == 600

    def test_a_null_market_value_is_refused_not_filled(self) -> None:
        frame = _ats_frame([2022], model_noise=5.0, market_noise=5.0)
        frame.loc[3, "market_spread"] = np.nan
        with pytest.raises(BlendTuningError, match="nulls"):
            MarketBlender().tune_weights({"ats": frame})

    def test_a_missing_column_is_refused(self) -> None:
        frame = _ats_frame([2022], model_noise=5.0, market_noise=5.0).drop(
            columns=["home_margin"]
        )
        with pytest.raises(BlendTuningError, match="home_margin"):
            MarketBlender().tune_weights({"ats": frame})


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def _converter(root: Path) -> None:
    directory = root / _CONVERTER_ID
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "slope_beta": _CONVERTER_SLOPE,
                "walk_forward_slopes": {"2021": 0.14},
                "training_seasons": [2020, 2021],
                "n_games": 10,
                "input_digest": "0" * 64,
                "fitted_at": "2099-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


def _provenance() -> BlendProvenance:
    return BlendProvenance(
        gold_generation_digest="a" * 64,
        source_artifact_ids={"wp": "wp_x", "ats": "ats_x", "ou": "ou_x"},
        tuning_corpus_rows=600,
        excluded_counts={"no_prelock_line": 3, "no_prior_fold_converter": 120},
        thread_limit=1,
    )


def _bound() -> MarketBlender:
    return MarketBlender(
        market_probability_artifact_id=_CONVERTER_ID,
        market_probability_slope_beta=_CONVERTER_SLOPE,
    )


class TestBlendArtifacts:
    def test_save_writes_the_weights(self, tmp_path: Path) -> None:
        _converter(tmp_path)
        blender = _bound()
        result = blender.tune_weights(_frames())
        directory = blender.save_blend_artifacts(
            result, tmp_path, provenance=_provenance()
        )
        payload = json.loads((directory / "blend_weights.json").read_text("utf-8"))
        assert payload["weights"] == {
            "wp": result.weights.wp_model_weight,
            "ats": result.weights.ats_model_weight,
            "ou": result.weights.ou_model_weight,
        }

    def test_save_records_the_tuning_record(self, tmp_path: Path) -> None:
        _converter(tmp_path)
        blender = _bound()
        result = blender.tune_weights(_frames())
        directory = blender.save_blend_artifacts(
            result, tmp_path, provenance=_provenance()
        )
        payload = json.loads((directory / "blend_weights.json").read_text("utf-8"))
        assert payload["blender_version"] == "3.0"
        assert payload["tuning_seasons"]["wp"] == [2021, 2022, 2023, 2024]
        assert payload["n_games"] == {"wp": 480, "ats": 600, "ou": 600}
        assert payload["objective"]["wp"] == "log_loss"
        assert payload["weight_grid"] == {"min": 0.0, "max": 1.0, "step": 0.01}
        assert "tuned_at" in payload
        assert set(payload["loss_market_only"]) == {"wp", "ats", "ou"}
        assert "per_target_clv" not in payload

    def test_save_does_not_move_latest_json_by_default(self, tmp_path: Path) -> None:
        _converter(tmp_path)
        (tmp_path / "latest.json").write_text('{"blend": "x"}', encoding="utf-8")
        _bound().save_blend_artifacts(
            _bound().tune_weights(_frames()), tmp_path, provenance=_provenance()
        )
        assert json.loads((tmp_path / "latest.json").read_text()) == {"blend": "x"}

    def test_save_with_update_latest_points_the_manifest_at_it(
        self, tmp_path: Path
    ) -> None:
        _converter(tmp_path)
        directory = _bound().save_blend_artifacts(
            _bound().tune_weights(_frames()),
            tmp_path,
            provenance=_provenance(),
            update_latest=True,
        )
        assert json.loads((tmp_path / "latest.json").read_text())["blend"] == (
            directory.name
        )

    def test_from_artifacts_reads_back_the_saved_weights(self, tmp_path: Path) -> None:
        _converter(tmp_path)
        result = _bound().tune_weights(_frames())
        directory = _bound().save_blend_artifacts(
            result, tmp_path, provenance=_provenance()
        )
        loaded = MarketBlender.from_artifacts(tmp_path, version=directory.name)
        assert loaded.config.weights == result.weights
        assert loaded.market_probability_slope_beta == _CONVERTER_SLOPE
        assert loaded.provenance == _provenance()

    def test_weights_round_trip_as_blend_weights(self, tmp_path: Path) -> None:
        _converter(tmp_path)
        result = _bound().tune_weights(_frames())
        assert isinstance(result.weights, BlendWeights)


class TestTheCli:
    def test_the_parser_has_one_option(self) -> None:
        from backtest.tune import _build_cli_parser

        options = {
            option
            for option in vars(_build_cli_parser())["_option_string_actions"]
            if option not in ("-h", "--help")
        }
        assert options == {"--artifacts-dir"}
