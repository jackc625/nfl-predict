"""Coverage flags survive each SERVED target's real transform on the REAL gold matrix (R10).

``test_imputation_preserves_flags.py`` proves the mechanism on a 2-flag synthetic frame and the
first two WP pipeline steps. This module closes the other half: the matrices are the real gold
matrices, the feature sets are the ones the SERVED models (``artifacts/latest.json``) were
written with, and the transform is the one each target actually applies.

WHAT A FLAG IS: the repo's own definition -- ``scripts.build_features.LEVEL_PRESERVED_COLUMN_
SUFFIX`` (a ``*_coverage`` column is a flag), the same suffix the build uses to keep a flag
out of the z-score. Not a hand-typed list.

THE TRANSFORM PER TARGET, read from the code:
  * WP: a fitted four-step Pipeline (imputer -> missing_indicator -> scaler -> estimator,
    ``WPTrainer._create_model``), served through ``preprocessing.pkl``. Its first two steps
    emit ``[imputed | _was_missing]``, so a flag at input position i must be at output
    position i, unchanged.
  * ATS and O/U: NO transform. ``preprocessing_is_model`` is False, no ``preprocessing.pkl``
    exists, and the model reads ``frame[feature_list]`` directly (XGBoost takes NaN natively).

FINDING THAT SHAPES THE TEST: the served feature lists contain ZERO ``*_coverage`` columns (the
snap and injury families are excluded at train time and the selector kept no flag), so
"every flag in the served set survives" is true vacuously. The module says so and does not rest
there: the served-set check is kept (it fails if a flag is ever selected and mangled), and the
real transform is ALSO run on served features PLUS every gold flag, which is the case the
pipeline must survive the day a flag is selected.

Controls: a planted stock imputer that drops an all-NaN column fails the same position check on
the real frame, and a planted reordered output fails it too.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.artifacts import load_model_artifact
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import (
    WPTrainer,
    _FoldRawIntoMissingIndicator,
    _ImputeAndCarryRaw,
)
from scripts.build_features import LEVEL_PRESERVED_COLUMN_SUFFIX

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
GOLD_DIR = REPO_ROOT / "data" / "gold"
TARGETS = ("wp", "ats", "ou")

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_ROOT / "latest.json").exists()
    or not all((GOLD_DIR / f"features_{t}.parquet").exists() for t in TARGETS),
    reason="artifacts/latest.json or data/gold/features_*.parquet absent (gitignored)",
)


def flag_columns(columns) -> list[str]:
    """The coverage flags among *columns*, by the repo's own suffix rule."""
    return [c for c in columns if str(c).endswith(LEVEL_PRESERVED_COLUMN_SUFFIX)]


def flags_survive_by_position(frame: pd.DataFrame, out: np.ndarray) -> bool:
    """Every flag unchanged at its input position, every indicator exact, width doubled."""
    width = frame.shape[1]
    if out.shape[1] != 2 * width:
        return False
    raw = frame.to_numpy(dtype=float)
    flags = set(flag_columns(frame.columns))
    for position, column in enumerate(frame.columns):
        if column in flags and not np.array_equal(out[:, position], raw[:, position]):
            return False
        if not np.array_equal(
            out[:, width + position], np.isnan(raw[:, position]).astype(float)
        ):
            return False
    return True


@pytest.fixture(scope="module")
def gold() -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(GOLD_DIR / f"features_{t}.parquet") for t in TARGETS}


@pytest.fixture(scope="module")
def served() -> dict[str, dict]:
    return {t: load_model_artifact(t, artifacts_dir=ARTIFACTS_ROOT) for t in TARGETS}


def _augmented(gold_frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The served feature set plus every flag gold carries (flags not already selected)."""
    extra = [c for c in flag_columns(gold_frame.columns) if c not in features]
    return gold_frame[[*features, *extra]]


def test_non_vacuity_gold_carries_flags_and_the_suffix_rule_finds_them(gold) -> None:
    for target in TARGETS:
        found = flag_columns(gold[target].columns)
        assert len(found) >= 10, (target, found)
        assert "weather_coverage" in found


class TestWpFlagsSurviveTheServedPipeline:
    def test_served_feature_set_flags_survive_the_served_preprocessing(
        self, gold, served
    ) -> None:
        artifact = served["wp"]
        features = artifact["feature_list"]
        frame = gold["wp"][features]
        out = np.asarray(
            artifact["preprocessing"][:2].transform(frame.to_numpy(dtype=float))
        )
        assert flags_survive_by_position(frame, out)
        # The served set carries no flag today; if one is ever selected it is checked above.
        assert flag_columns(features) == [], (
            "a coverage flag is now in the served WP feature set; the position check above "
            "covers it, update this finding"
        )

    def test_every_gold_flag_survives_the_real_wp_pipeline_on_real_gold(
        self, gold, served
    ) -> None:
        frame = _augmented(gold["wp"], served["wp"]["feature_list"])
        assert len(flag_columns(frame.columns)) >= 10
        assert frame.isna().any().any(), "non-vacuity: the real frame carries NaN"
        out = np.asarray(
            WPTrainer()
            ._create_model({})[:2]
            .fit_transform(frame.to_numpy(dtype=float)),
            dtype=float,
        )
        assert flags_survive_by_position(frame, out)

    def test_a_planted_dropped_empty_column_fails_the_check_on_the_real_frame(
        self, gold, served
    ) -> None:
        class _DroppingImputer(_ImputeAndCarryRaw):
            """The stock sklearn behaviour: an all-NaN column is dropped from the output."""

            def fit(self, X, y=None):
                self.imputer_ = SimpleImputer(strategy=self.strategy)
                self.imputer_.fit(np.asarray(X, dtype=float))
                self.n_features_in_ = np.asarray(X, dtype=float).shape[1]
                return self

        features = served["wp"]["feature_list"]
        frame = _augmented(gold["wp"], features).copy()
        frame[features[0]] = np.nan  # a pre-coverage-style empty block
        planted = Pipeline(
            [
                ("imputer", _DroppingImputer(strategy="median")),
                ("missing_indicator", _FoldRawIntoMissingIndicator()),
            ]
        )
        out = np.asarray(
            planted.fit_transform(frame.to_numpy(dtype=float)), dtype=float
        )
        assert not flags_survive_by_position(frame, out)

    def test_a_planted_reordered_output_fails_the_check(self, gold, served) -> None:
        frame = _augmented(gold["wp"], served["wp"]["feature_list"])
        out = np.asarray(
            WPTrainer()
            ._create_model({})[:2]
            .fit_transform(frame.to_numpy(dtype=float)),
            dtype=float,
        )
        flag_position = list(frame.columns).index(flag_columns(frame.columns)[0])
        swapped = out.copy()
        swapped[:, [0, flag_position]] = swapped[:, [flag_position, 0]]
        assert not flags_survive_by_position(frame, swapped)


@pytest.mark.parametrize(
    ("target", "trainer_class"), [("ats", ATSTrainer), ("ou", OUTrainer)]
)
class TestAtsAndOuApplyNoTransform:
    def test_the_served_model_has_no_preprocessing_step(
        self, target, trainer_class, served
    ) -> None:
        artifact = served[target]
        assert artifact["metadata"]["preprocessing_is_model"] is False
        assert artifact["preprocessing"] is None
        assert not (artifact["artifact_dir"] / "preprocessing.pkl").exists()

    def test_the_served_model_reads_the_selected_columns_by_name_unchanged(
        self, target, trainer_class, gold, served
    ) -> None:
        artifact = served[target]
        features = artifact["feature_list"]
        assert list(artifact["model"].get_booster().feature_names) == features
        frame = gold[target][features]
        assert np.isfinite(artifact["model"].predict(frame)).all()
        # No flag is selected today; any that is must reach the model as the gold value.
        for flag in flag_columns(features):
            np.testing.assert_array_equal(
                frame[flag].to_numpy(), gold[target][flag].to_numpy()
            )

    def test_every_gold_flag_reaches_the_model_input_unchanged(
        self, target, trainer_class, gold, served
    ) -> None:
        """No transform exists, so a flag handed to the model is the gold value."""
        frame = _augmented(gold[target], served[target]["feature_list"])
        flags = flag_columns(frame.columns)
        assert len(flags) >= 10
        model = trainer_class()._create_model({"n_estimators": 3})
        model.fit(frame, np.linspace(-7.0, 7.0, len(frame)))
        assert list(model.get_booster().feature_names) == list(frame.columns)
        for flag in flags:
            np.testing.assert_array_equal(
                frame[flag].to_numpy(), gold[target][flag].to_numpy()
            )
