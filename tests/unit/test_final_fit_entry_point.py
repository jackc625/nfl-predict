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

THE BOUNDARY WAS INVERTED BY PLAN 33-15 TASK 1, NOT DELETED
-----------------------------------------------------------
Phase 33 Wave 15 IS the caller the old scan named, so the assertion moved: ``models/train.py``
now calls the fit once and the application once, and the other three production entry modules
still call NEITHER. The absent-module refusal and the planted-call control are unchanged in
kind, and the control now asserts the REAL call count plus one so it cannot pass on an
arithmetic coincidence. See the block comment above the boundary section for the full record
of what changed and which sentence of the old failure message authorised it.

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

import ast
import inspect
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models.train as models_train
from conf.season_partition import SeasonPartition, default_season_partition
from models.artifacts import (
    PREPROCESSING_FILENAME,
    PREPROCESSING_IS_MODEL_METADATA_KEY,
)
from models.temporal import WalkForwardSplitter
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import BaseTrainer
from models.trainers.final_fit import (
    FINAL_FIT_COMPONENT_POLICY,
    FINAL_FIT_COMPONENTS_METADATA_KEY,
    FINAL_FIT_SEASONS_METADATA_KEY,
    FinalFitResult,
    apply_final_fit_to_trainer,
    final_fit_over_completed_seasons,
)
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WP_PIPELINE_STEP_NAMES, WPTrainer
from tests.data_boundary import digest_file
from tests.gold_generation import gold_generation_key
from tests.phase33_state import (
    FINAL_FIT_NOT_RUN_IN_PHASE_331,
    GOLD_GENERATION_AFTER_ELO_REBUILD,
    GOLD_GENERATION_AT_REFIT,
    GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    P332_25_POST_SWAP_LATEST_JSON_SHA256,
    P332_25B_REFIT_GOLD_GENERATION,
    WP_PREPROCESSING_DEFECT_CLOSURE,
)
from tests.unit.test_weather_bridge_expiry import WEATHER_GENERATION_MARKER_KEY

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


class TestTheRecordedSeasonsAreTheSeasonsFitted:
    """Code review WR-04: final_fit recorded seasons the model may never have seen.

    ``seasons = tuple(partition.final_fit)`` was written unconditionally into
    ``components["final_fit_seasons"]``, into the model's own description string
    ("refit on N completed seasons (2002-2025)") and into the persisted
    ``FINAL_FIT_SEASONS_METADATA_KEY``. The only guard fired when NONE of the
    requested seasons was present. A frame holding 2002-2024 against a rule saying
    2002-2025 therefore fitted 23 seasons and recorded 24 -- naming a season the
    model never saw, in a milestone whose stated purpose is that records must not
    overstate.
    """

    def test_a_frame_missing_ONE_requested_season_refuses_by_name(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        dropped = _PARTITION.final_fit[-1]
        short = frame[frame["season"] != dropped]

        with pytest.raises(ValueError) as excinfo:
            wp_trained.final_fit(short, _PARTITION)

        message = str(excinfo.value)
        assert str(dropped) in message, (
            f"the refusal must NAME the season with no rows. Got: {message}"
        )
        assert "never saw" in message, (
            "the refusal must say what the silent alternative would have produced, "
            f"not merely that something is missing. Got: {message}"
        )

    def test_the_complete_frame_still_fits_and_records_every_season(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        """The control: the refusal must not fire on the normal case."""
        result = wp_trained.final_fit(frame, _PARTITION)

        assert result.components["final_fit_seasons"] == list(_PARTITION.final_fit)
        assert result.components["final_fit_rows"] == len(frame)


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


# ---------------------------------------------------------------------------
# THE BOUNDARY (Plan 33.1-10 Task 4), INVERTED BY PLAN 33-15 TASK 1(d).
#
# WHAT IT ASSERTED. Until Plan 33-15 this section asserted that NO production entry
# module called any final-fit symbol --
# `TestThisPhaseAuthorsTheEntryPointAndDoesNotRunIt
# ::test_no_production_module_calls_the_final_fit_entry_point` -- because
# `33.1-SPEC.md`'s Boundaries put ANY model re-fit out of scope by name, and authoring a
# re-fit mechanism was in scope while CALLING one was not.
#
# WHAT IT ASSERTS NOW. `models/train.py` calls the fit EXACTLY ONCE and the application
# EXACTLY ONCE; `scripts/promote_models.py`, `scripts/run_phase33_gate.py` and
# `pipeline/steps.py` still call NEITHER.
#
# WHICH SENTENCE OF ITS OWN DOCSTRING AUTHORISED THE CHANGE, quoted verbatim from the
# assertion message it used to carry: "Authoring the entry point is in scope for Phase
# 33.1; calling it is a model re-fit, which 33.1-SPEC.md's Boundaries put out of scope by
# name. Phase 33 Wave 15 is the intended caller." The boundary was scoped to Phase 33.1
# from the day it was written and it NAMED its successor. Plan 33-15 Task 1 is that
# successor, so the scan fired for exactly the reason it exists.
#
# THE BOUNDARY MOVED BY EXACTLY ONE MODULE; IT DID NOT DISSOLVE. Deleting the scan would
# have removed the only instrument in this repository that can say WHICH production entry
# module re-fits, at the precise moment one of them started to.
#
# IT IS NOT A TRIPWIRE. It is GREEN under a corrected assertion, not deliberately red, so
# no node id defined here is added to `tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`,
# which stays byte-unchanged at five entries.
# ---------------------------------------------------------------------------

#: The production entry modules a re-fit would have to be invoked from.
_PRODUCTION_ENTRY_MODULES: tuple[str, ...] = (
    "models/train.py",
    "scripts/promote_models.py",
    "scripts/run_phase33_gate.py",
    "pipeline/steps.py",
)

#: All THREE public symbols, not just the two that fit. `apply_final_fit_to_trainer`
#: MUTATES trainer state, and a call to it on a production path would be as much a
#: re-fit as a call to the fit itself -- it is what makes the save path persist a
#: different model.
_FINAL_FIT_SYMBOLS: frozenset[str] = frozenset(
    {
        "final_fit",
        "final_fit_over_completed_seasons",
        "apply_final_fit_to_trainer",
    }
)


#: The ONE production entry module Plan 33-15 Task 1 authorises to call the final fit.
_WAVE_15_CALLER: str = "models/train.py"

#: The production entry modules that must still call NEITHER symbol. Derived from
#: :data:`_PRODUCTION_ENTRY_MODULES` by subtraction rather than re-typed, so a module
#: added to the scanned list cannot be silently omitted from the must-not-call half.
_MODULES_THAT_STILL_CALL_NEITHER: tuple[str, ...] = tuple(
    module for module in _PRODUCTION_ENTRY_MODULES if module != _WAVE_15_CALLER
)


def _final_fit_calls(paths: tuple[str, ...]) -> list[tuple[str, int, str]]:
    """Return ``(path, lineno, symbol)`` for every final-fit call in *paths*.

    The SYMBOL is carried because the inverted boundary is a per-symbol claim -- one call
    to the fit and one to the application -- and a bare hit count cannot distinguish two
    fits from a fit and an application.

    A REQUIRED MODULE THAT IS ABSENT FAILS BY NAME rather than shortening the scanned
    list. The same guard the quarantine scan uses, and for the same reason: a boundary
    check that can pass by visiting nothing is not a boundary check.
    """
    calls: list[tuple[str, int, str]] = []
    for path in paths:
        source_path = Path(path)
        if not source_path.exists():
            msg = (
                f"required module {path} is absent from the checkout, so the boundary "
                "scan would pass by not visiting it. Fix the path or remove it from "
                "_PRODUCTION_ENTRY_MODULES with a recorded reason."
            )
            raise AssertionError(msg)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in _FINAL_FIT_SYMBOLS:
                calls.append((path, node.lineno, str(name)))
    return calls


def _scan_for_final_fit_calls(paths: tuple[str, ...]) -> list[str]:
    """Return ``path:lineno`` for every call to a final-fit symbol in *paths*.

    Unchanged in behaviour by Plan 33-15: it is the same scan, expressed over
    :func:`_final_fit_calls` so the two cannot drift. The absent-module refusal is
    inherited from that function rather than duplicated.
    """
    return [f"{path}:{lineno}" for path, lineno, _symbol in _final_fit_calls(paths)]


class TestPhase33Wave15IsTheCallerAndTheBoundaryMovedByOneModule:
    """T-33.1-67, INVERTED: the boundary moved by exactly one module.

    The old class name was ``TestThisPhaseAuthorsTheEntryPointAndDoesNotRunIt`` and its
    claim was that NO production entry module called the final fit. Plan 33-15 Task 1 is
    the caller that class's own failure message named, so the claim is now the narrower
    and stronger one: ONE named module calls it, in a proven order, and the other three
    still do not.
    """

    def test_models_train_calls_the_fit_once_and_the_application_once(self) -> None:
        """The positive half. A re-fit that never reaches the save path is not a re-fit."""
        by_symbol = Counter(
            symbol for _path, _lineno, symbol in _final_fit_calls((_WAVE_15_CALLER,))
        )

        assert by_symbol["final_fit"] == 1, (
            f"{_WAVE_15_CALLER} calls trainer.final_fit {by_symbol['final_fit']} times; "
            "Plan 33-15 Task 1(a) authorises exactly one, between train_and_evaluate and "
            f"trainer.save. Observed calls: {_final_fit_calls((_WAVE_15_CALLER,))}."
        )
        assert by_symbol["apply_final_fit_to_trainer"] == 1, (
            f"{_WAVE_15_CALLER} calls apply_final_fit_to_trainer "
            f"{by_symbol['apply_final_fit_to_trainer']} times; exactly one is "
            "authorised. Producing a FinalFitResult and NOT applying it leaves "
            "BaseTrainer.save persisting the LAST walk-forward fold's object while the "
            "record describes the final fit -- threat T-33.1-65d."
        )
        assert by_symbol["final_fit_over_completed_seasons"] == 0, (
            f"{_WAVE_15_CALLER} calls final_fit_over_completed_seasons directly. The "
            "authorised route is the concrete trainer's thin final_fit wrapper, so the "
            "three trainers keep one shared entry rather than two."
        )

    def test_the_other_three_production_entry_modules_still_call_neither(self) -> None:
        """The negative half. The boundary moved by one module; it did not dissolve."""
        hits = _scan_for_final_fit_calls(_MODULES_THAT_STILL_CALL_NEITHER)

        assert hits == [], (
            f"a production path that is NOT {_WAVE_15_CALLER} calls the final fit at "
            f"{hits}. Plan 33-15 authorises exactly one caller; a second one means the "
            "re-fit can start from a path whose verdict record nobody wrote."
        )

    def test_the_call_order_in_source_is_fit_then_apply_then_save(self) -> None:
        """The ORDER is the whole property: save reads what apply assigned."""
        source = Path(_WAVE_15_CALLER).read_text(encoding="utf-8")
        i_fit = source.find(".final_fit(")
        i_apply = source.find("apply_final_fit_to_trainer(")
        i_save = source.find("trainer.save(")

        assert 0 < i_fit < i_apply < i_save, (
            f"the three calls in {_WAVE_15_CALLER} are not in the order "
            f"final_fit -> apply_final_fit_to_trainer -> trainer.save (offsets "
            f"{i_fit}, {i_apply}, {i_save}). apply_final_fit_to_trainer assigns "
            "trainer.model, trainer.preprocessing and the two metadata keys, and "
            "BaseTrainer.save reads exactly those three -- so any other order persists "
            "the last walk-forward fold's model under a record describing the final fit."
        )

    def test_the_partition_is_read_from_the_rule_with_no_season_list_typed(
        self,
    ) -> None:
        """One rule, read once. A typed season list is the defect D33.1-03 abolished."""
        train_target_source = inspect.getsource(models_train.train_target)

        assert "default_season_partition()" in train_target_source, (
            "models.train.train_target does not read "
            "conf.season_partition.default_season_partition(). The partition handed to "
            "the final fit must come from the committed rule, never from the "
            "--config-*-seasons arguments: those three describe the walk-forward folds, "
            "and partition.final_fit is a FOURTH set that deliberately overlaps them."
        )
        typed_years = sorted(
            set(re.findall(r"\b(?:19|20)\d{2}\b", train_target_source))
        )
        assert typed_years == [], (
            f"models.train.train_target types season literals {typed_years}. The "
            "partition comes from conf/season_partition.py; a literal here is a second "
            "declaration that can drift away from the rule with nothing to notice."
        )

    def test_the_boundary_scan_reports_the_real_call_count_plus_one(
        self, tmp_path: Path
    ) -> None:
        """The control, re-based. A scan that never fires proves nothing.

        It used to assert exactly one hit, which was true only while the real count was
        zero. Now that ``models/train.py`` genuinely calls the entry point, asserting one
        would pass on an arithmetic coincidence rather than on the scan firing, so the
        control asserts the MEASURED count plus the planted call -- and asserts the
        measured count is non-zero, so "plus one" is not an increment on nothing.
        """
        real = len(_scan_for_final_fit_calls((_WAVE_15_CALLER,)))
        planted = tmp_path / "planted_train.py"
        original = Path(_WAVE_15_CALLER).read_text(encoding="utf-8")
        planted.write_text(
            original + "\n\napply_final_fit_to_trainer(trainer, result)\n",
            encoding="utf-8",
        )

        hits = _scan_for_final_fit_calls((str(planted),))

        assert real > 0, (
            f"{_WAVE_15_CALLER} carries zero final-fit calls, so this control would be "
            "proving only that the scan can count to one on a planted line. Plan 33-15 "
            "Task 1(a) makes that module the authorised caller."
        )
        assert len(hits) == real + 1, (hits, real)

    def test_the_boundary_scan_refuses_an_absent_required_module(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            _scan_for_final_fit_calls(("models/no_such_module.py",))

        assert "models/no_such_module.py" in str(excinfo.value)
        assert "pass by not visiting" in str(excinfo.value)

    def test_the_record_names_the_module_its_symbols_and_the_scanned_modules(
        self,
    ) -> None:
        record = FINAL_FIT_NOT_RUN_IN_PHASE_331

        assert record["entry_point_module"] == "models/trainers/final_fit.py"
        assert set(record["symbols"]) == set(_FINAL_FIT_SYMBOLS)
        assert tuple(record["scanned_modules"]) == _PRODUCTION_ENTRY_MODULES
        assert "Wave 15" in str(record["intended_caller"])

    def test_the_recorded_latest_json_digests_are_the_same_value_twice(self) -> None:
        """The claim IS that the two are equal: this plan moved nothing."""
        record = FINAL_FIT_NOT_RUN_IN_PHASE_331

        assert (
            record["latest_json_digest_before"] == record["latest_json_digest_after"]
        ), (
            "the recorded before and after digests differ, which would mean Plan 33.1-10 "
            "changed the deployed-model manifest"
        )

    def test_the_live_latest_json_digest_has_MOVED_off_the_phase_331_record(
        self,
    ) -> None:
        """RE-POINTED 2026-09-14 by Plan 33-15 Task 4, with the 33.1 record unedited.

        WHAT IT ASSERTED: the live ``artifacts/latest.json`` digest still equalled
        ``FINAL_FIT_NOT_RUN_IN_PHASE_331["latest_json_digest_after"]``, which was the whole
        of Phase 33.1's claim -- that phase authored the final-fit mechanism and moved no
        production pointer.

        WHY IT MOVED: Plan 33-15 is the first thing in this phase PERMITTED to move that
        digest, and on the owner's ruling of 2026-09-14 it moved all three target pointers
        in one owner-authorised pass. The Phase-33.1 record is NOT edited to agree --
        it is the record of what THAT phase left, and it is still true of that phase.

        WHAT IT ASSERTS NOW: the live digest equals the most recent authorised write and is
        NOT the Phase-33.1 value. Both halves are asserted, because "it changed" alone
        would be satisfied by any accident. RE-POINTED 2026-09-23 from Plan 33-15's
        ``POST_GATE_MANIFEST_DIGEST`` to Plan 33.2-25's post-swap digest; both records
        are unedited.
        """
        latest = Path("artifacts") / "latest.json"
        if not latest.exists():
            pytest.skip(
                "artifacts/latest.json is absent; it is gitignored and is written by "
                "models.artifacts.update_manifest via scripts/promote_models.py"
            )

        live = digest_file(latest)

        assert live == P332_25_POST_SWAP_LATEST_JSON_SHA256, (
            "the live manifest digest is not Plan 33.2-25's recorded post-swap state. "
            "Something moved the production swap surface outside an owner-authorised pass."
        )
        assert live != FINAL_FIT_NOT_RUN_IN_PHASE_331["latest_json_digest_after"]

    def test_the_entry_point_writes_nothing_under_production_artifacts(
        self, wp_trained: WPTrainer, frame: pd.DataFrame
    ) -> None:
        """Belt and braces beside the autouse COLD-05 guard, because it is cheap."""
        artifacts_root = Path("artifacts")
        before = sorted(p.name for p in artifacts_root.iterdir())

        result = wp_trained.final_fit(frame, _PARTITION)
        apply_final_fit_to_trainer(wp_trained, result)

        assert sorted(p.name for p in artifacts_root.iterdir()) == before


# ---------------------------------------------------------------------------
# THE GOLD-GENERATION MARKER (Plan 33-15 Task 1(b)).
#
# `tests/unit/test_weather_bridge_expiry.py` DEFINES and TESTS the marker key and
# deliberately does NOT write it -- writing it in Phase 33.1 would have been claiming a
# re-fit happened. Plan 33-15 Task 1 is the wave that writes it, from a command-line
# value rather than from an import: the ONE producer of the key is
# `tests.gold_generation.gold_generation_key`, and a production module importing from the
# tests package to reach it would be the wrong direction. The operator measures it once,
# records it in `tests.phase33_state.GOLD_GENERATION_AT_REFIT`, and passes it in; the
# record is then asserted against the ladder's own generation, so the marker is provably
# the generation Plan 33-14 produced rather than a string somebody typed.
# ---------------------------------------------------------------------------


class TestTheGoldGenerationMarkerIsDeclaredAndOptIn:
    """The flag exists, spells the key the bridge tests spell, and defaults to ABSENT."""

    def test_models_train_declares_the_key_the_bridge_module_defines(self) -> None:
        declared = getattr(models_train, "GOLD_GENERATION_METADATA_KEY", None)

        assert declared == WEATHER_GENERATION_MARKER_KEY, (
            "models.train.GOLD_GENERATION_METADATA_KEY must be spelled exactly as "
            "tests/unit/test_weather_bridge_expiry.py's WEATHER_GENERATION_MARKER_KEY "
            f"spells it ({WEATHER_GENERATION_MARKER_KEY!r}); it reads {declared!r}. A "
            "marker the bridge predicate cannot find is a marker that does not exist."
        )

    def test_the_parser_exposes_the_flag_and_defaults_it_to_none(self) -> None:
        parser = models_train.build_parser()
        options = {
            option for action in parser._actions for option in action.option_strings
        }

        assert "--gold-generation" in options, (
            f"models.train's parser exposes {sorted(options)} and not "
            "--gold-generation. Plan 33-15 Task 2 passes the measured generation key "
            "through this flag for every candidate it fits."
        )
        assert parser.parse_args(["--target", "wp"]).gold_generation is None, (
            "--gold-generation must default to None so an ORDINARY training run claims "
            "nothing about a weather generation."
        )


class TestTheFinalFitAndTheMarkerReachASavedArtifact:
    """The end-to-end claim, on a synthetic frame and a sandbox artifacts root.

    This is the shape the tracer run proves on real gold. It is here as well because a
    source-order assertion says the calls are in the right order and says nothing about
    whether the SAVED FILES carry what the record claims.
    """

    def test_a_saved_wp_artifact_carries_the_pipeline_seasons_and_the_marker(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = models_train.train_target(
            target="wp",
            features_df=frame,
            closing_odds_df=None,
            artifacts_dir=tmp_path,
            tune=False,
            gold_generation="synthetic-generation-key-for-this-test",
        )

        artifact_dir = Path(result["artifact_path"])
        metadata = json.loads(
            (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )

        assert (artifact_dir / PREPROCESSING_FILENAME).is_file(), (
            f"{artifact_dir} carries no {PREPROCESSING_FILENAME}. Under D33.1-R1 the WP "
            "transform and estimator travel as ONE inseparable Pipeline; without the "
            "file the serving path has nothing to recover the fitted scaler from, which "
            "is the live wp_20260824_113325 defect reproduced in a new artifact."
        )
        assert metadata[PREPROCESSING_IS_MODEL_METADATA_KEY] is True
        assert metadata[FINAL_FIT_SEASONS_METADATA_KEY] == list(_PARTITION.final_fit)
        components = metadata[FINAL_FIT_COMPONENTS_METADATA_KEY]
        for component in FINAL_FIT_COMPONENT_POLICY:
            assert component in components, (component, sorted(components))
        assert components["calibrator_refitted"] is False
        assert (
            metadata[WEATHER_GENERATION_MARKER_KEY]
            == "synthetic-generation-key-for-this-test"
        )

    def test_omitting_the_generation_leaves_the_key_ABSENT_not_empty(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Absent, never empty: an empty marker is a claim that reads as a non-claim."""
        result = models_train.train_target(
            target="ats",
            features_df=frame,
            closing_odds_df=None,
            artifacts_dir=tmp_path,
            tune=False,
        )

        metadata = json.loads(
            (Path(result["artifact_path"]) / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        assert WEATHER_GENERATION_MARKER_KEY not in metadata, (
            "an ordinary training run wrote a weather-generation marker. The key must be "
            f"ABSENT, not empty: {metadata.get(WEATHER_GENERATION_MARKER_KEY)!r}."
        )
        assert metadata[FINAL_FIT_SEASONS_METADATA_KEY] == list(_PARTITION.final_fit)


class TestTheRecordedGenerationIsTheLaddersOwn:
    """`GOLD_GENERATION_AT_REFIT` is MEASURED, never typed from memory."""

    def test_it_equals_the_generation_plan_33_14s_ladder_produced(self) -> None:
        assert GOLD_GENERATION_AT_REFIT == GOLD_GENERATION_AFTER_ELO_REBUILD

    def test_it_is_not_the_uncaptured_pre_rung_sentinel(self) -> None:
        """The sentinel is not a hex digest and can never be a real generation."""
        assert (
            GOLD_GENERATION_AT_REFIT != GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED
        )

    def test_the_live_gold_is_the_generation_the_served_models_were_fitted_on(
        self,
    ) -> None:
        """Live gold is what production's models were fitted on.

        This used to compare live gold with ``GOLD_GENERATION_AT_REFIT``, the Phase-33
        re-fit's generation. Plan 33.2-20 rebuilt gold and Plan 33.2-25 re-fitted on it,
        so that record is history and the live comparison is against the 33.2-25 re-fit's.
        """
        assert gold_generation_key() == P332_25B_REFIT_GOLD_GENERATION


class TestTheWPPreprocessingDefectClosureIsRecordedInOnePlace:
    """The disclosure travels in committed source, because `.planning/` is gitignored."""

    def test_it_names_the_deployed_artifact_and_that_the_defect_predates_phase_331(
        self,
    ) -> None:
        record = WP_PREPROCESSING_DEFECT_CLOSURE

        assert record["artifact"] == "wp_20260824_113325"
        assert record["predates_phase_331"] is True
        assert "scaled" in str(record["defect"]).lower()

    def test_it_states_that_closure_is_by_shipping_a_replacement(self) -> None:
        record = WP_PREPROCESSING_DEFECT_CLOSURE

        assert "replacement" in str(record["closes_by"]).lower()
        assert record["closed_at_task_1"] is False
        assert "Task 4" in str(record["closed_by"])
