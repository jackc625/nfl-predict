"""Fold-fitted imputation never erases a coverage flag (Plan 33.2-17 Task 2, D33.2-08 item 2).

WHAT IS PINNED. Before a family's first covered season its values are an honest unknown (NaN)
beside a coverage flag that says so. The WP logistic regression cannot take NaN, so WP's four-step
Pipeline (``models/trainers/wp_trainer.py``) median-imputes INSIDE the fold and appends a
``_was_missing`` indicator per column. This module proves, by ARRAY POSITION, that the flag and
the absence both survive that transformation:

* an N-column input becomes a 2N-column array;
* column i of the output is input column i imputed -- a flag is never NaN, so it comes through
  with its values unchanged, at its own position;
* column N+i is input column i's indicator: 1 exactly where input column i was NaN;
* ``WPTrainer.indicator_feature_names`` names position N+i after input column i.

Asserted by POSITION because the transformed object is a NumPy array with no column labels
(``_ImputeAndCarryRaw.transform`` and ``_FoldRawIntoMissingIndicator.transform`` both return
``np.hstack``); names are rebuilt separately by ``indicator_feature_names``.

A COLUMN THAT IS ENTIRELY NaN IN THE FIT WINDOW is the case that matters most once the
pre-coverage seasons are honest unknowns: a training window that ends before snaps' first covered
season sees the snap columns as all-NaN. sklearn's ``SimpleImputer`` DROPS such a column by
default, which would shift every later column one place left and misalign the ``[imputed | raw]``
split -- a flag would silently land at the wrong position. The pipeline keeps it
(``keep_empty_features=True``); the planted-violation test runs the stock imputer and shows the
position check catches exactly that.

A STANDARDIZED 0/1 FLAG IS STILL A LEGITIMATE INPUT. The third step, ``StandardScaler``, rescales
a flag column; that does not remove its information, and its coefficient is read on the scaled
column like every other feature's. The scaling is not a defect (Antigravity, L1305@f924749).

ATS and O/U are bare XGBoost, which takes NaN natively; their frames carry NaN untouched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import models.trainers.ats_trainer as ats_module
import models.trainers.base as base_module
import models.trainers.ou_trainer as ou_module
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import (
    MISSING_INDICATOR_SUFFIX,
    WPTrainer,
    _FoldRawIntoMissingIndicator,
    _ImputeAndCarryRaw,
)

FLAG_COLUMNS: tuple[str, ...] = ("home_snap_coverage", "home_injury_coverage")


def _frame(rows: int = 40) -> pd.DataFrame:
    """Snap and injury values NaN before coverage (the first half), flags saying so."""
    rng = np.random.default_rng(17)
    covered = np.arange(rows) >= rows // 2
    frame = pd.DataFrame(
        {
            "elo_diff": rng.normal(0.0, 50.0, rows),
            "home_snap_concentration": np.where(
                covered, rng.uniform(0.1, 0.3, rows), np.nan
            ),
            "home_snap_coverage": covered.astype(float),
            "home_qb_out_flag": np.where(covered, rng.integers(0, 2, rows), np.nan),
            "home_injury_coverage": covered.astype(float),
        }
    )
    return frame


def _impute_and_indicate(
    frame: pd.DataFrame, pipeline: Pipeline | None = None
) -> np.ndarray:
    """The fitted ``imputer`` and ``missing_indicator`` steps of the REAL WP pipeline."""
    model = pipeline if pipeline is not None else WPTrainer()._create_model({})
    head = model[:2]
    return np.asarray(head.fit_transform(frame.to_numpy(dtype=float)), dtype=float)


def _flags_survive_by_position(frame: pd.DataFrame, out: np.ndarray) -> bool:
    """Every flag unchanged at its input position, and every indicator exact."""
    width = frame.shape[1]
    if out.shape[1] != 2 * width:
        return False
    raw = frame.to_numpy(dtype=float)
    for position, column in enumerate(frame.columns):
        if column in FLAG_COLUMNS and not np.array_equal(
            out[:, position], raw[:, position]
        ):
            return False
        if not np.array_equal(
            out[:, width + position], np.isnan(raw[:, position]).astype(float)
        ):
            return False
    return True


class TestTheFlagSurvivesByPosition:
    def test_non_vacuity_the_input_carries_nan(self) -> None:
        frame = _frame()
        assert frame.isna().any().any()
        assert frame[list(FLAG_COLUMNS)].notna().all().all(), "a flag is never NaN"

    def test_the_output_is_exactly_twice_as_wide(self) -> None:
        frame = _frame()
        assert _impute_and_indicate(frame).shape == (len(frame), 2 * frame.shape[1])

    def test_each_flag_comes_through_unchanged_at_its_own_position(self) -> None:
        frame = _frame()
        out = _impute_and_indicate(frame)
        for column in FLAG_COLUMNS:
            position = list(frame.columns).index(column)
            np.testing.assert_array_equal(out[:, position], frame[column].to_numpy())

    def test_each_indicator_is_one_exactly_where_its_input_was_nan(self) -> None:
        frame = _frame()
        out = _impute_and_indicate(frame)
        width = frame.shape[1]
        for position, column in enumerate(frame.columns):
            expected = frame[column].isna().to_numpy().astype(float)
            np.testing.assert_array_equal(out[:, width + position], expected, column)

    def test_the_indicator_names_follow_the_input_order(self) -> None:
        frame = _frame()
        model = WPTrainer()._create_model({})
        model[:2].fit(frame.to_numpy(dtype=float))
        names = WPTrainer.indicator_feature_names(model, list(frame.columns))
        assert names == [f"{c}{MISSING_INDICATOR_SUFFIX}" for c in frame.columns]

    def test_a_column_entirely_nan_in_the_fit_window_keeps_its_place(self) -> None:
        # A training window that ends before snaps' first covered season: the snap column is
        # all-NaN. It must not be dropped, or every later column shifts one place left.
        frame = _frame()
        frame["home_snap_concentration"] = np.nan
        frame["home_snap_coverage"] = 0.0
        out = _impute_and_indicate(frame)
        assert _flags_survive_by_position(frame, out)

    def test_the_whole_pipeline_fits_and_predicts_on_the_frame(self) -> None:
        frame = _frame()
        assert frame["home_snap_concentration"].isna().any(), "non-vacuity"
        y = (np.arange(len(frame)) % 2).astype(int)
        model = WPTrainer()._create_model({})
        model.fit(frame, y)
        probabilities = model.predict_proba(frame)[:, 1]
        assert np.isfinite(probabilities).all()


class TestAPlantedViolationIsCaught:
    def test_a_stock_imputer_that_drops_an_empty_column_fails_the_position_check(
        self,
    ) -> None:
        class _DroppingImputer(_ImputeAndCarryRaw):
            """The stock sklearn behaviour: an all-NaN column is dropped from the output."""

            def fit(self, X, y=None):
                self.imputer_ = SimpleImputer(strategy=self.strategy)
                self.imputer_.fit(np.asarray(X, dtype=float))
                self.n_features_in_ = np.asarray(X, dtype=float).shape[1]
                return self

        frame = _frame()
        frame["home_snap_concentration"] = np.nan
        frame["home_snap_coverage"] = 0.0
        planted = Pipeline(
            [
                ("imputer", _DroppingImputer(strategy="median")),
                ("missing_indicator", _FoldRawIntoMissingIndicator()),
            ]
        )
        out = np.asarray(
            planted.fit_transform(frame.to_numpy(dtype=float)), dtype=float
        )
        assert not _flags_survive_by_position(frame, out)

    def test_a_reordered_output_fails_the_position_check(self) -> None:
        frame = _frame()
        out = _impute_and_indicate(frame)
        swapped = out.copy()
        swapped[:, [1, 2]] = swapped[:, [2, 1]]
        assert not _flags_survive_by_position(frame, swapped)


class TestAtsAndOuCarryNanUntouched:
    @pytest.mark.parametrize("trainer_class", [ATSTrainer, OUTrainer])
    def test_xgboost_fits_on_nan_without_any_imputation(self, trainer_class) -> None:
        frame = _frame()
        y = np.linspace(-7.0, 7.0, len(frame))
        model = trainer_class()._create_model({"n_estimators": 5})
        model.fit(frame, y)
        assert np.isfinite(model.predict(frame)).all()
        assert frame.isna().any().any(), "the frame handed to fit still carries NaN"

    @pytest.mark.parametrize("module", [base_module, ats_module, ou_module])
    def test_no_trainer_module_fills_or_imputes(self, module) -> None:
        tree = ast.parse(inspect.getsource(module))
        calls = {
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not calls & {"fillna", "SimpleImputer", "KNNImputer", "IterativeImputer"}
