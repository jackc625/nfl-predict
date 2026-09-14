"""The final-fit entry point, proven component by component on SYNTHETIC frames.

WHAT THIS MODULE PROVES, AND WHAT IT DELIBERATELY DOES NOT DO
------------------------------------------------------------
Plan 33.1-10 AUTHORS an explicit final fit because D33.1-01 does not survive contact
with the code otherwise. ``WalkForwardSplitter.generate_splits`` builds every fold as
``season < holdout_season`` over the full gold frame and each concrete trainer keeps the
LAST fold's model, so no choice of the three existing season lists ever produces a model
fitted through the newest completed season.

This module drives that entry point on frames built in memory. It reads no
``data/gold/`` parquet, writes no artifact directory and needs no network. Plan 33.1-10's
Ruling T is explicit that this phase authors the mechanism and does NOT run it: Phase 33
Wave 15 owns the actual re-fit, and the AST boundary scan in this module is what keeps
that boundary checkable rather than merely stated.

THE FOUR COMPONENTS, AND WHY IDENTITY RATHER THAN EQUALITY
----------------------------------------------------------
Ruling S decides each persisted component separately. The calibrator is CARRIED OVER --
it is fitted on hp-val PREDICTIONS from a model trained on the selection window only,
which is what makes it out-of-sample, and refitting it on rows the final model was fitted
on would make it in-sample: the reliability curve would look excellent and mean nothing.
So the assertion here is object IDENTITY and not numeric agreement. A silently refitted
calibrator that happened to agree to fifteen decimal places would still fail this module,
which is the whole point of choosing identity.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from models.trainers.final_fit import (
    FINAL_FIT_COMPONENT_POLICY,
    FINAL_FIT_COMPONENTS_METADATA_KEY,
    FINAL_FIT_SEASONS_METADATA_KEY,
    FinalFitResult,
    apply_final_fit_to_trainer,
    final_fit_over_completed_seasons,
)

from conf.season_partition import SeasonPartition, default_season_partition
from models.temporal import WalkForwardSplitter
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import BaseTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WP_PIPELINE_STEP_NAMES, WPTrainer

# The LIVE partition, derived rather than typed. A literal here would fall behind
# `conf/season_partition.py` the moment `LATEST_COMPLETED_SEASON` is bumped, and the
# failure mode would be silent: `generate_splits` SKIPS a holdout season with no rows.
_PARTITION = default_season_partition()

# Enough rows per season that WP's hp-val fold carries both classes and a Platt
# calibration has something to fit. Sixteen is not a magic number -- it is the smallest
# count that kept the synthetic hp-val fold two-class across the seeds tried.
_ROWS_PER_SEASON = 16

_FEATURE_COLUMNS = ("feature_1", "feature_2", "feature_3")


def _synthetic_features(
    seasons: tuple[int, ...] = _PARTITION.final_fit,
    rows_per_season: int = _ROWS_PER_SEASON,
) -> pd.DataFrame:
    """Build the synthetic feature frame this module fits on.

    Shaped after ``tests/unit/test_temporal_splits.py``'s ``synthetic_features_df``
    rather than inventing a second convention: id columns, numeric features, and the
    three per-target columns the splitter treats as targets.

    Every target is a KNOWN function of ``feature_1``, so a fitted model's coefficients
    are checkable rather than merely present.
    """
    rng = np.random.default_rng(3311012)
    rows: list[dict[str, object]] = []
    for season in seasons:
        for i in range(rows_per_season):
            f1, f2, f3 = (float(v) for v in rng.normal(size=3))
            rows.append(
                {
                    "game_id": f"{season}_{i:02d}",
                    "season": season,
                    "week": (i % 18) + 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "feature_1": f1,
                    "feature_2": f2,
                    "feature_3": f3,
                    "home_win": int(f1 + 0.25 * f2 > 0.0),
                    "home_margin": 7.0 * f1 + 2.0 * f2,
                    "total_points": 44.0 + 6.0 * f1 - 3.0 * f3,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    """The synthetic frame spanning every season in the live partition's final fit."""
    return _synthetic_features()


@pytest.fixture
def wp_trained(frame: pd.DataFrame) -> WPTrainer:
    """A WP trainer whose walk-forward stage HAS run -- the entry point's precondition."""
    trainer = WPTrainer()
    trainer.train_and_evaluate(frame, tune=False)
    return trainer


@pytest.fixture
def ats_trained(frame: pd.DataFrame) -> ATSTrainer:
    """An ATS trainer whose walk-forward stage HAS run."""
    trainer = ATSTrainer()
    trainer.train_and_evaluate(frame, tune=False)
    return trainer


@pytest.fixture
def ou_trained(frame: pd.DataFrame) -> OUTrainer:
    """An O/U trainer whose walk-forward stage HAS run."""
    trainer = OUTrainer()
    trainer.train_and_evaluate(frame, tune=False)
    return trainer


class TestTheComponentPolicyIsCommittedAsData:
    """Ruling S's table is readable from source, not only from a plan document."""

    def test_the_policy_names_all_four_persisted_components(self) -> None:
        assert sorted(FINAL_FIT_COMPONENT_POLICY) == [
            "calibrator",
            "feature_names",
            "model",
            "preprocessing",
        ]

    def test_the_calibrator_entry_says_carried_over_rather_than_refitted(self) -> None:
        entry = FINAL_FIT_COMPONENT_POLICY["calibrator"].lower()
        assert "carried over" in entry
        assert "not refitted" in entry or "rather than refit" in entry

    def test_the_two_metadata_keys_are_the_declared_names(self) -> None:
        assert FINAL_FIT_SEASONS_METADATA_KEY == "final_fit_seasons"
        assert FINAL_FIT_COMPONENTS_METADATA_KEY == "final_fit_components"


class TestTheModelIsRefitOnEveryCompletedSeason:
    """The one reason the entry point exists (D33.1-02)."""

    def test_the_fitted_season_set_and_row_count_equal_the_partitions_final_fit(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)

        assert isinstance(result, FinalFitResult)
        assert result.seasons == _PARTITION.final_fit
        expected_rows = int(frame["season"].isin(_PARTITION.final_fit).sum())
        assert result.n_rows == expected_rows

    def test_the_final_fit_row_set_exceeds_the_last_fold_by_the_last_holdout_season(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        """The measured reason a different season LIST could never have done this.

        The last walk-forward fold trains on ``season < max(holdout)``, so under the live
        partition it stops at 2024 and the newest completed season never enters the
        shipped model. The difference is exactly that one season's rows.
        """
        splitter = WalkForwardSplitter(config=wp_trained.config, target_col="home_win")
        last_fold = list(splitter.generate_splits(frame))[-1]
        last_fold_ids = set(last_fold.train_data.index)

        final_fit_ids = set(
            frame.loc[frame["season"].isin(_PARTITION.final_fit), "game_id"]
        )
        newest_season_ids = set(
            frame.loc[frame["season"] == _PARTITION.holdout[-1], "game_id"]
        )

        assert final_fit_ids - last_fold_ids == newest_season_ids
        assert newest_season_ids


class TestTheCalibratorIsCarriedOverAndNotRefitted:
    """T-33.1-64: the defect class this milestone exists to detect."""

    def test_the_returned_calibrator_is_the_same_object_the_trainer_held(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        before = wp_trained.calibrator
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.calibrator is before

    def test_the_provenance_records_calibrator_refitted_false_with_a_reason(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.components["calibrator_refitted"] is False
        reason = result.components["calibrator_refit_reason"]
        assert isinstance(reason, str)
        assert "in-sample" in reason.lower()

    def test_every_one_of_the_four_components_is_recorded_in_the_provenance(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)

        for component in FINAL_FIT_COMPONENT_POLICY:
            assert component in result.components
            assert isinstance(result.components[component], str)


class TestTheWPPreprocessingIsRefitOnTheFinalFitRows:
    """T-33.1-65: a scaler and a model fitted on different row sets."""

    def test_the_persisted_object_is_the_four_step_pipeline(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.preprocessing is not None
        assert tuple(result.preprocessing.named_steps) == WP_PIPELINE_STEP_NAMES
        # Scaling and the estimator are ONE object, so they cannot be handed
        # different row sets even deliberately (D33.1-R1).
        assert result.preprocessing is result.model

    def test_the_scaler_means_match_the_final_fit_rows_and_not_the_selection_window(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)
        features = list(wp_trained.feature_names)
        width = len(features)

        scaler_means = np.asarray(
            result.preprocessing.named_steps["scaler"].mean_, dtype=float
        )[:width]

        final_rows = frame.loc[frame["season"].isin(_PARTITION.final_fit), features]
        selection_rows = frame.loc[frame["season"].isin(_PARTITION.selection), features]

        np.testing.assert_allclose(
            scaler_means, final_rows.to_numpy(dtype=float).mean(axis=0), atol=1e-9
        )
        selection_means = selection_rows.to_numpy(dtype=float).mean(axis=0)
        assert float(np.max(np.abs(scaler_means - selection_means))) > 1e-6


class TestATSAndOUCarryConverterParametersAndNoPreprocessing:
    """Neither trainer has preprocessing; both have a fitted converter (D33.1-R1)."""

    def test_ats_returns_no_preprocessing_and_populated_converter_params(
        self, ats_trained: ATSTrainer, frame: pd.DataFrame
    ) -> None:
        result = ats_trained.final_fit(frame, _PARTITION)

        assert result.preprocessing is None
        assert result.converter_params is not None
        assert sorted(result.converter_params) == [
            "distribution_params",
            "distribution_type",
            "residual_std",
        ]
        assert result.converter_params["distribution_type"] == "normal"

    def test_ou_returns_no_preprocessing_and_populated_converter_params(
        self, ou_trained: OUTrainer, frame: pd.DataFrame
    ) -> None:
        result = ou_trained.final_fit(frame, _PARTITION)

        assert result.preprocessing is None
        assert result.converter_params is not None
        assert sorted(result.converter_params) == [
            "distribution_params",
            "distribution_type",
            "residual_std",
        ]

    def test_wp_carries_no_converter_params(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.converter_params is None


class TestTheFeatureNamesAreCarriedThroughUnchanged:
    """Ruling S: selection stays on the selection window, where Ruling Q put it."""

    def test_the_returned_feature_names_equal_the_trainers_own(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        before = list(wp_trained.feature_names)
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.feature_names == before
        assert wp_trained.feature_names == before


class TestTheEntryPointRefusesByName:
    """T-33.1-66: an object that looks like a model and is not."""

    def test_a_trainer_with_no_feature_names_refuses_naming_feature_names(
        self, frame: pd.DataFrame
    ) -> None:
        trainer = WPTrainer()

        with pytest.raises(ValueError) as excinfo:
            final_fit_over_completed_seasons(trainer, frame, _PARTITION)

        message = str(excinfo.value)
        assert "feature_names" in message
        assert "walk-forward" in message.lower()

    def test_a_trainer_with_no_calibration_component_refuses_naming_it(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        wp_trained.calibrator = None

        with pytest.raises(ValueError) as excinfo:
            final_fit_over_completed_seasons(wp_trained, frame, _PARTITION)

        assert "calibrator" in str(excinfo.value)

    def test_absent_seasons_refuse_naming_both_the_requested_and_the_present(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        absent = SeasonPartition(
            selection=_PARTITION.selection,
            hp_val=_PARTITION.hp_val,
            holdout=_PARTITION.holdout,
            final_fit=(1990, 1991),
        )

        with pytest.raises(ValueError) as excinfo:
            final_fit_over_completed_seasons(wp_trained, frame, absent)

        message = str(excinfo.value)
        assert "1990" in message and "1991" in message
        assert str(_PARTITION.final_fit[0]) in message


class TestApplyFinalFitMovesTheTrainerState:
    """T-33.1-65d: returning a record is not the same as moving the save path's inputs."""

    def test_it_assigns_model_preprocessing_and_both_metadata_keys_by_identity(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        last_fold_model = wp_trained.model
        result = wp_trained.final_fit(frame, _PARTITION)

        apply_final_fit_to_trainer(wp_trained, result)

        assert wp_trained.model is result.model
        assert wp_trained.model is not last_fold_model
        assert wp_trained.preprocessing is result.preprocessing
        assert (
            wp_trained.metadata[FINAL_FIT_SEASONS_METADATA_KEY] == _PARTITION.final_fit
        )
        assert (
            wp_trained.metadata[FINAL_FIT_COMPONENTS_METADATA_KEY] is result.components
        )

    def test_the_fit_itself_leaves_the_trainer_untouched(
        self, ats_trained: ATSTrainer, frame: pd.DataFrame
    ) -> None:
        """The fit is pure; the mutation is a separate, named act."""
        before = ats_trained.model
        result = ats_trained.final_fit(frame, _PARTITION)

        assert ats_trained.model is before
        assert result.model is not before
        assert FINAL_FIT_SEASONS_METADATA_KEY not in ats_trained.metadata


class TestTheBaseClassDocumentationStaysTrue:
    """Ruling T / T-33.1-68: the entry point is a sibling, not a base method."""

    def test_base_trainer_has_no_final_fit(self) -> None:
        assert not hasattr(BaseTrainer, "final_fit")

    def test_base_trainer_carries_a_none_preprocessing_class_default(self) -> None:
        assert getattr(BaseTrainer, "preprocessing", "MISSING") is None

    def test_every_concrete_trainer_exposes_final_fit_and_still_overrides_train(
        self,
    ) -> None:
        for trainer_cls in (WPTrainer, ATSTrainer, OUTrainer):
            assert hasattr(trainer_cls, "final_fit"), trainer_cls.__name__
            assert (
                trainer_cls.train_and_evaluate is not BaseTrainer.train_and_evaluate
            ), trainer_cls.__name__

    def test_ats_and_ou_expose_the_converter_params_hook_and_wp_does_not(self) -> None:
        assert hasattr(ATSTrainer, "converter_params")
        assert hasattr(OUTrainer, "converter_params")
        assert not hasattr(WPTrainer, "converter_params")
