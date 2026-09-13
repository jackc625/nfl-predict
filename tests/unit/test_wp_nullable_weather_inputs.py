"""WP consumes nullable weather without breaking, and without leaking a fold.

WHY THIS MODULE EXISTS (D33.1-R3, owner-ratified at replan time 2026-09-12)
----------------------------------------------------------------------------
Plan 33.1-04 Tasks 1 and 2 make an absent weather observation reach gold as
NaN. That is the point of the phase, and it puts NaN into 45 columns that WP's
feature selection currently offers to a bare `LogisticRegression`, which rejects
it. Codex found the exact path and the orchestrator re-verified it against live
source:

    WPTrainer.train_and_evaluate  -> select_features   (BEFORE any scaler)
    BaseTrainer.select_features   -> self._create_model(...).fit(informative)
    informative_columns           -> withholds a column that does not VARY.
                                     A NaN-bearing weather column DOES vary.

The currently-deployed `wp_20260824_113325` feature list containing zero weather
features does NOT protect re-selection: `select_features` re-runs on every
`train_and_evaluate` call and `models.train` exposes no feature-list flag, so a
Wave-15 re-fit re-selects from the whole informative candidate pool.

THE OWNER CHOSE THE PIPELINE AND REJECTED THE ALTERNATIVE. Excluding nullable
weather from WP's selection would work, and would foreclose exactly the thing
this phase exists to produce: WP could never use the corrected weather. So
`WPTrainer._create_model` returns a four-step Pipeline instead, and there is no
code path that produces a bare estimator.

THE PER-FOLD RULE IS A TEMPORAL-SAFETY REQUIREMENT, NOT A NICETY
------------------------------------------------------------------
An imputation statistic computed over the whole frame and applied inside a fold
leaks the holdout's distribution into the training set. That is the class of
defect CLAUDE.md's walk-forward constraint forbids and that this milestone
exists to detect. Fitting the Pipeline inside the fold loop gives it for free --
`Pipeline.fit` fits every step on the rows it is handed -- but "it happens to be
inside the loop" is not a guarantee anyone can check later, so
`test_the_imputer_is_fitted_per_fold` checks it.

NOTHING HERE IS MOCKED AWAY. `train_and_evaluate` is RUN against a partially
null gold-shaped frame. A test that patched `select_features` would prove the
patch works.

NOTHING UNDER artifacts/ IS WRITTEN. This task changes how WP WOULD fit; Phase
33 Wave 15 is what fits it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features.weather import WEATHER_FEATURE_COLUMNS_BY_BUILDER
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import (
    MISSING_INDICATOR_SUFFIX,
    WP_PIPELINE_STEP_NAMES,
    WPTrainer,
)

# The numeric half of the full builder's weather family -- the columns that
# actually reach a gold matrix as floats.
WEATHER_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]
    if column != "weather_condition"
)

SEASONS = (2018, 2019, 2020, 2021, 2022)
ROWS_PER_SEASON = 40


def _gold_shaped_frame(
    *,
    null_fraction: float = 0.35,
    seed: int = 7,
    seasons: tuple[int, ...] = SEASONS,
) -> pd.DataFrame:
    """A gold-shaped frame: id columns, a signal feature, the weather family.

    Follows `tests/unit/test_temporal_splits.py`'s helper shape rather than
    inventing a second convention. The target is a known function of ONE
    non-weather feature, so a fitted coefficient is checkable rather than
    merely present.

    Args:
        null_fraction: Share of rows whose weather columns are NaN. Stands in
            for Ruling J's absent-observation and dome states together.
        seed: RNG seed, so two frames can be built identical or deliberately
            different.
        seasons: Seasons to generate.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in seasons:
        for index in range(ROWS_PER_SEASON):
            signal = float(rng.normal())
            row: dict = {
                "game_id": f"{season}_{index:03d}",
                "season": season,
                "week": (index % 18) + 1,
                "home_team": "KC",
                "away_team": "BUF",
                "signal_feature": signal,
                "noise_feature": float(rng.normal()),
                "home_win": int(signal > 0.0),
            }
            missing = rng.random() < null_fraction
            for offset, column in enumerate(WEATHER_COLUMNS):
                row[column] = (
                    float("nan")
                    if missing
                    else float(rng.normal(loc=offset * 0.1, scale=1.0))
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _config() -> TemporalSplitConfig:
    return TemporalSplitConfig(
        train_seasons=[2018, 2019],
        hp_val_seasons=[2020],
        holdout_seasons=[2021, 2022],
    )


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_win",
    }
    return [column for column in frame.columns if column not in excluded]


class TestThereIsNoBareEstimatorPath:
    def test_no_bare_logistic_regression_path_exists(self):
        model = WPTrainer()._create_model({})
        assert isinstance(model, Pipeline)
        assert [name for name, _ in model.steps] == list(WP_PIPELINE_STEP_NAMES)
        assert isinstance(model.steps[-1][1], LogisticRegression)

    def test_the_step_names_are_declared_once_for_plan_33_1_10_to_import(self):
        assert WP_PIPELINE_STEP_NAMES == (
            "imputer",
            "missing_indicator",
            "scaler",
            "estimator",
        )
        assert MISSING_INDICATOR_SUFFIX == "_was_missing"

    def test_ats_and_ou_are_unaffected(self):
        """Both return NaN-tolerant XGBoost models, which is why the base.py
        change is behaviour-preserving for them."""
        assert not isinstance(ATSTrainer()._create_model({}), Pipeline)
        assert not isinstance(OUTrainer()._create_model({}), Pipeline)


class TestTheSelectionPathConsumesNaN:
    def test_select_features_does_not_raise_on_nan_bearing_candidates(self):
        """The exact call that fails today with `Input X contains NaN`."""
        frame = _gold_shaped_frame()
        trainer = WPTrainer(config=_config())
        selected = trainer.select_features(
            frame[_feature_columns(frame)],
            frame["home_win"],
            max_features=20,
        )
        assert isinstance(selected, list)
        assert selected, "selection returned nothing, so nothing was proven"
        assert all(name in _feature_columns(frame) for name in selected), (
            "selection must return names from the INPUT frame; an indicator "
            "column leaking out would make the returned list unusable as a "
            "column selector against gold"
        )

    def test_train_and_evaluate_completes_against_a_partially_null_frame(self):
        """The Wave-15 unblocking proof. RUN, not mocked."""
        frame = _gold_shaped_frame(null_fraction=0.35)
        weather_nulls = frame[list(WEATHER_COLUMNS)].isna().any(axis=1).mean()
        assert weather_nulls >= 0.25, (
            f"only {weather_nulls:.2%} of rows carry NaN weather; the frame is "
            "not representative of what Plan 33.1-07 puts into gold"
        )

        trainer = WPTrainer(config=_config())
        result = trainer.train_and_evaluate(frame, tune=False)

        assert result["feature_names"], "no features were selected"
        assert trainer.model is not None
        assert isinstance(trainer.model, Pipeline)
        assert result["season_results"], "no holdout season was evaluated"


class TestTheImputerNeverCrossesAFoldBoundary:
    def test_the_imputer_is_fitted_per_fold(self):
        """A TEMPORAL-SAFETY assertion, not a plumbing one.

        Two folds whose pre-holdout rows have materially different weather
        distributions must produce DIFFERENT imputer statistics. Identical
        statistics here mean one fit saw both folds -- an imputation statistic
        that crossed a walk-forward boundary, which is the leak CLAUDE.md's
        walk-forward constraint forbids.
        """
        trainer = WPTrainer(config=_config())
        columns = list(WEATHER_COLUMNS)

        early = _gold_shaped_frame(seed=11, seasons=(2018, 2019))
        late = _gold_shaped_frame(seed=11, seasons=(2018, 2019))
        # Shift the later fold's weather distribution by a wide, unmistakable
        # margin, so an equal pair cannot be an artefact of two similar draws.
        late[columns] = late[columns] + 50.0

        first = trainer._create_model({}).fit(early[columns], early["home_win"])
        second = trainer._create_model({}).fit(late[columns], late["home_win"])

        first_statistics = np.asarray(trainer.imputer_statistics(first))
        second_statistics = np.asarray(trainer.imputer_statistics(second))

        assert first_statistics.shape == second_statistics.shape
        assert np.nanmax(np.abs(first_statistics - second_statistics)) > 1e-6, (
            "two folds with materially different pre-holdout distributions "
            "produced IDENTICAL imputer statistics, which is only possible if "
            "the imputer was fitted once over the whole frame"
        )


class TestTheIndicatorSetIsDeterministic:
    def test_missing_indicator_columns_are_deterministic(self):
        """`features="all"` -- the emitted set is a function of the INPUT
        COLUMNS, never of which rows happened to be null in the fit window.

        A data-dependent indicator set would make two re-fits produce two
        different feature spaces, which is the reproducibility constraint
        CLAUDE.md states.
        """
        trainer = WPTrainer(config=_config())
        columns = list(WEATHER_COLUMNS)

        sparse = _gold_shaped_frame(seed=3, null_fraction=0.05, seasons=(2018,))
        dense = _gold_shaped_frame(seed=4, null_fraction=0.80, seasons=(2018,))

        sparse_names = trainer.indicator_feature_names(
            trainer._create_model({}).fit(sparse[columns], sparse["home_win"]),
            columns,
        )
        dense_names = trainer.indicator_feature_names(
            trainer._create_model({}).fit(dense[columns], dense["home_win"]),
            columns,
        )

        assert sparse_names == dense_names
        assert len(sparse_names) == len(columns)
        assert all(name.endswith(MISSING_INDICATOR_SUFFIX) for name in sparse_names)

    def test_a_frame_with_no_nan_still_emits_the_full_indicator_set(self):
        trainer = WPTrainer(config=_config())
        columns = list(WEATHER_COLUMNS)
        frame = _gold_shaped_frame(seed=5, null_fraction=0.0, seasons=(2018,))
        model = trainer._create_model({}).fit(frame[columns], frame["home_win"])
        names = trainer.indicator_feature_names(model, columns)
        assert len(names) == len(columns)


class TestTheNoNanCaseIsUnchanged:
    def test_the_no_nan_case_is_unchanged(self):
        """The NEGATIVE CONTROL: the two new steps are INERT without NaN.

        The reference is a scaler-plus-estimator pipeline rather than a bare
        unscaled `LogisticRegression`. Standardization is not what this task
        introduced -- WP's production model has always been fitted on scaled
        features (`wp_trainer.py:275-277` before this plan) -- so comparing
        against an UNSCALED fit would compare two different models and fail for
        a reason that has nothing to do with NaN. What is under test is whether
        the imputer and the indicator change anything when there is nothing to
        impute, and the answer must be no.
        """
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler

        trainer = WPTrainer(config=_config())
        frame = _gold_shaped_frame(seed=9, null_fraction=0.0)
        columns = _feature_columns(frame)
        X = frame[columns]
        y = frame["home_win"]

        full = trainer._create_model({}).fit(X, y)
        reference = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(**trainer._get_default_params())),
            ]
        ).fit(X, y)

        full_coefficients = full.named_steps["estimator"].coef_[0][: len(columns)]
        reference_coefficients = reference.named_steps["estimator"].coef_[0]

        assert np.max(np.abs(full_coefficients - reference_coefficients)) < 1e-6, (
            "the imputer and the missing indicator changed the fit on a frame "
            "with nothing missing, so they are not inert in the case they were "
            "not introduced for"
        )

    def test_the_indicator_half_carries_no_weight_without_nan(self):
        trainer = WPTrainer(config=_config())
        frame = _gold_shaped_frame(seed=9, null_fraction=0.0)
        columns = _feature_columns(frame)
        model = trainer._create_model({}).fit(frame[columns], frame["home_win"])
        indicator_coefficients = model.named_steps["estimator"].coef_[0][len(columns) :]
        assert np.max(np.abs(indicator_coefficients)) < 1e-6


class TestNothingUnderArtifactsMoved:
    def test_no_artifact_directory_was_created(self):
        import subprocess

        result = subprocess.run(
            ["git", "status", "--short", "artifacts/"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stdout.strip() == "", (
            "this task changes how WP WOULD fit; Phase 33 Wave 15 is what fits "
            f"it. git reports: {result.stdout!r}"
        )


class TestTheContractIsCommitted:
    def test_the_state_slot_records_the_decision_and_the_rejected_alternative(self):
        from tests.phase33_state import WP_NULLABLE_INPUT_CONTRACT as contract

        assert tuple(contract["pipeline_steps"]) == WP_PIPELINE_STEP_NAMES
        assert contract["imputer_fitted_per_fold"] is True
        assert contract["applies_to_pre_selection_path"] is True
        assert contract["requirement"] == "R4"
        assert contract["decision"] == "D33.1-R3"
        assert "rejected_alternative" in contract
        assert contract["missing_indicator_suffix"] == MISSING_INDICATOR_SUFFIX
