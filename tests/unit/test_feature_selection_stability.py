"""Feature selection must not depend on how much dead weight is in the frame.

Plan 30-17, Task 2. The lock on the property Task 1's census measured the absence of.

WHAT THE CENSUS FOUND, because these tests only make sense against it
--------------------------------------------------------------------
On the owner-accepted rung-4 gold, 82 of the 180 candidate columns are zero-variance
over the 2018-2019 window ``select_features`` fits on. Adding ten MORE columns that are
literally constant changed 5 of ATS's 25 selected features and 9 of O/U's 25; removing
all 82 changed 9 and 5. WP was unaffected.

The mechanism is NOT scikit-learn's threshold. Every target's cap is binding (56 / 85 /
91 features clear the resolved threshold against caps of 20 / 25 / 25), so the threshold
filter is a no-op and the rule is already pure top-K in effect. What moves is the
ESTIMATOR REFIT: ``select_features`` fits its scoring model on whatever columns are
present, and the two XGBoost targets run ``colsample_bytree=0.8``, so an added constant
changes which columns each tree sees and therefore the gain importances of the real
features. WP's ``LogisticRegression`` has no column sampling, which is exactly why WP
was already invariant.

That is why the fix is a zero-variance PRE-FILTER on the fit input rather than a
different threshold: identical post-filter input gives an identical fit, so invariance
holds by construction rather than by tuning.

WHAT THESE TESTS DELIBERATELY DO NOT CLAIM
------------------------------------------
Invariance to NOISE padding. A column drawn independently of the target is
information-free about the target but has variance, and at fit time it is
indistinguishable from a weak real signal -- the census measured noise columns actually
being SELECTED (up to 5 of ATS's 25, 8 of O/U's 25). No pre-filter can exclude them
without peeking at the target, and doing so would be a different selection rule
entirely. ``TestThePreFilterIsExactlyScoped`` pins that boundary instead of pretending
past it.

See ``SELECTION-CENSUS.md`` and ``scripts/selection_census.py``.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.trainers.ats_trainer import _ATS_MAX_FEATURES, ATSTrainer
from models.trainers.base import BaseTrainer
from models.trainers.ou_trainer import _OU_MAX_FEATURES, OUTrainer
from models.trainers.wp_trainer import _WP_MAX_FEATURES, WPTrainer

_GOLD_DIR = Path("data/gold")
_BASE_SOURCE = Path("models/trainers/base.py")

# (label, trainer class, budget, whether the target is binary)
_TARGETS: list[tuple[str, type[Any], int, bool]] = [
    ("wp", WPTrainer, _WP_MAX_FEATURES, True),
    ("ats", ATSTrainer, _ATS_MAX_FEATURES, False),
    ("ou", OUTrainer, _OU_MAX_FEATURES, False),
]

# The padding grid mirrors the census exactly, so a regression here is comparable to the
# committed measurement rather than to a differently-shaped probe.
_PAD_N = (10, 25, 50)
# 0.0 and 1.0 are both zero-variance, but only the non-zero one can carry an
# intercept-like linear coefficient -- a column of zeros cannot influence a linear score
# whatever coefficient it is given, so it is the WEAKER probe for the LogReg target.
_PAD_VALUES = (0.0, 1.0)


def _synthetic_frame(
    n_rows: int = 400, n_live: int = 40, n_dead: int = 12, seed: int = 7
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A small frame with the same SHAPE of problem as the real selection window.

    Live features with geometrically decaying true coefficients (so the ranking near the
    cap boundary is genuinely contested, which is where count-dependence shows up), plus
    a block of zero-variance columns standing in for gold's 82.
    """
    rng = np.random.default_rng(seed)
    live = [f"live_{i:02d}" for i in range(n_live)]
    frame = pd.DataFrame({name: rng.standard_normal(n_rows) for name in live})
    for i in range(n_dead):
        frame[f"dead_{i:02d}"] = 0.0
    beta = rng.standard_normal(n_live) * np.linspace(2.0, 0.05, n_live)
    continuous = frame[live].to_numpy() @ beta + rng.standard_normal(n_rows) * 0.5
    return frame, pd.Series(continuous), pd.Series((continuous > 0).astype(int))


def _pad(frame: pd.DataFrame, n: int, value: float) -> pd.DataFrame:
    padded = frame.copy()
    for i in range(n):
        padded[f"__pad_{i:03d}"] = value
    return padded


def _select(
    trainer_cls: type[Any], X: pd.DataFrame, y: pd.Series, cap: int
) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trainer_cls().select_features(X, y, max_features=cap)


class TestSelectionIsIndependentOfDeadColumnCount:
    """Adding information-free columns must not change WHICH features are selected."""

    @pytest.mark.parametrize(("label", "trainer_cls", "cap", "binary"), _TARGETS)
    @pytest.mark.parametrize("n_pad", _PAD_N)
    @pytest.mark.parametrize("pad_value", _PAD_VALUES)
    def test_padding_with_zero_variance_columns_does_not_move_the_selection(
        self,
        label: str,
        trainer_cls: type[Any],
        cap: int,
        binary: bool,
        n_pad: int,
        pad_value: float,
    ) -> None:
        frame, continuous, binary_target = _synthetic_frame()
        y = binary_target if binary else continuous

        baseline = _select(trainer_cls, frame, y, cap)
        padded = _select(trainer_cls, _pad(frame, n_pad, pad_value), y, cap)

        moved = sorted(set(baseline) ^ set(padded))
        assert moved == [], (
            f"{label}: adding {n_pad} zero-variance columns (value {pad_value}) moved "
            f"{len(moved)} of {len(baseline)} selected features: {moved}"
        )

    @pytest.mark.parametrize(("label", "trainer_cls", "cap", "binary"), _TARGETS)
    def test_removing_every_dead_column_does_not_move_the_selection(
        self,
        label: str,
        trainer_cls: type[Any],
        cap: int,
        binary: bool,
    ) -> None:
        """The ablation direction -- the one Plan 30-07's rung-3 drop actually took."""
        frame, continuous, binary_target = _synthetic_frame()
        y = binary_target if binary else continuous
        dead = [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
        assert dead, "the synthetic frame must contain dead columns for this to bite"

        baseline = _select(trainer_cls, frame, y, cap)
        ablated = _select(trainer_cls, frame.drop(columns=dead), y, cap)

        moved = sorted(set(baseline) ^ set(ablated))
        assert moved == [], (
            f"{label}: removing {len(dead)} zero-variance columns moved {len(moved)} of "
            f"{len(baseline)} selected features: {moved}"
        )

    @pytest.mark.parametrize(("label", "trainer_cls", "cap", "binary"), _TARGETS)
    def test_selection_is_deterministic_on_an_unchanged_frame(
        self,
        label: str,
        trainer_cls: type[Any],
        cap: int,
        binary: bool,
    ) -> None:
        """Without this, a passing invariance test could be luck rather than a property."""
        frame, continuous, binary_target = _synthetic_frame()
        y = binary_target if binary else continuous
        assert _select(trainer_cls, frame, y, cap) == _select(
            trainer_cls, frame, y, cap
        ), f"{label}: repeated selection on an unchanged frame did not reproduce"


@pytest.mark.skipif(
    not (_GOLD_DIR / "features_wp.parquet").exists(),
    reason="data/gold not built on this checkout",
)
class TestRealGold:
    """The same property on the REAL production selection window, not a stand-in.

    Deliberately narrower than the synthetic grid -- one padding size per target -- so
    the suite pays a few seconds rather than a few minutes for the live confirmation.
    The synthetic grid carries the breadth.
    """

    @pytest.mark.parametrize(
        ("label", "trainer_cls", "cap", "table"),
        [
            ("wp", WPTrainer, _WP_MAX_FEATURES, "features_wp"),
            ("ats", ATSTrainer, _ATS_MAX_FEATURES, "features_ats"),
            ("ou", OUTrainer, _OU_MAX_FEATURES, "features_ou"),
        ],
    )
    def test_padding_live_gold_does_not_move_the_selection(
        self, label: str, trainer_cls: type[Any], cap: int, table: str
    ) -> None:
        config = TemporalSplitConfig.default()
        trainer = trainer_cls(config=config)
        target_col = trainer._get_target_column()
        gold = pd.read_parquet(_GOLD_DIR / f"{table}.parquet")

        splitter = WalkForwardSplitter(config=config, target_col=target_col)
        split = splitter.get_train_val_split(gold)
        X, y = split.train_data, split.train_targets

        dead = int((X.nunique(dropna=False) <= 1).sum())
        assert dead > 0, (
            "this test is only meaningful while gold still carries dead columns; if "
            "that changed, re-derive the census rather than deleting the assertion"
        )

        baseline = _select(trainer_cls, X, y, cap)
        padded = _select(trainer_cls, _pad(X, 25, 0.0), y, cap)

        moved = sorted(set(baseline) ^ set(padded))
        assert moved == [], (
            f"{label}: on live gold ({X.shape[1]} candidates, {dead} of them dead), "
            f"adding 25 constant columns moved {len(moved)} of {len(baseline)} "
            f"selected features: {moved}"
        )


class TestThePreFilterIsExactlyScoped:
    """The pre-filter drops zero-variance columns and NOTHING else."""

    def test_only_zero_variance_columns_are_withheld_from_the_fit(self) -> None:
        from models.trainers.base import informative_columns

        frame = pd.DataFrame(
            {
                "varies": [1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0],
                "constant_zero": [0.0, 0.0, 0.0],
                "all_nan": [np.nan, np.nan, np.nan],
                "barely_varies": [0.0, 0.0, 1e-12],
            }
        )
        assert informative_columns(frame) == ["varies", "barely_varies"]

    def test_a_noise_column_is_not_withheld(self) -> None:
        """The guarantee stops at zero variance, and says so.

        A column of pure noise carries no information ABOUT THE TARGET but is
        indistinguishable from a weak signal without looking at the target. Excluding it
        would be a different selection rule; the census records that noise padding does
        move the selection, and that is reported rather than asserted away.
        """
        from models.trainers.base import informative_columns

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"noise": rng.standard_normal(50), "dead": np.zeros(50)})
        assert informative_columns(frame) == ["noise"]

    def test_a_wholly_constant_frame_falls_back_rather_than_selecting_from_nothing(
        self,
    ) -> None:
        """A degenerate frame must not become an empty fit. Exercised end to end."""
        frame = pd.DataFrame(
            {"a": [1.0] * 40, "b": [2.0] * 40, "c": [0.0] * 40},
        )
        y = pd.Series(np.arange(40, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = ATSTrainer().select_features(frame, y, max_features=2)
        assert set(selected) <= set(frame.columns)

    def test_the_estimator_is_fitted_on_the_filtered_frame(self) -> None:
        """Proves the filter is APPLIED, not merely present as a helper."""
        seen: dict[str, list[str]] = {}
        trainer = ATSTrainer()
        original_create = trainer._create_model

        def spy(params: dict) -> Any:
            model = original_create(params)
            original_fit = model.fit

            def fit(X: pd.DataFrame, y: Any, **kwargs: Any) -> Any:
                seen["columns"] = list(X.columns)
                return original_fit(X, y, **kwargs)

            model.fit = fit
            return model

        trainer._create_model = spy  # type: ignore[method-assign]
        frame, continuous, _ = _synthetic_frame(n_rows=120, n_live=8, n_dead=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.select_features(frame, continuous, max_features=4)

        assert seen["columns"] == [c for c in frame.columns if c.startswith("live_")]


class TestTheSemanticsAreStatedAtTheCallSite:
    """A reader must be able to predict the rule without knowing a library default."""

    @staticmethod
    def _select_from_model_call() -> ast.Call:
        tree = ast.parse(_BASE_SOURCE.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SelectFromModel"
        ]
        assert len(calls) == 1, (
            f"expected exactly one SelectFromModel construction in {_BASE_SOURCE}, "
            f"found {len(calls)}"
        )
        return calls[0]

    def test_threshold_is_passed_explicitly(self) -> None:
        call = self._select_from_model_call()
        keywords = {kw.arg for kw in call.keywords}
        assert "threshold" in keywords, (
            "SelectFromModel is constructed without an explicit threshold, so the rule "
            "is inherited from scikit-learn's threshold=None default (which resolves to "
            "the mean for a non-L1 estimator and to 1e-5 for an L1 one). State it."
        )

    def test_the_threshold_is_the_mean_that_the_default_already_resolved_to(
        self,
    ) -> None:
        call = self._select_from_model_call()
        threshold = next(kw.value for kw in call.keywords if kw.arg == "threshold")
        assert isinstance(threshold, ast.Constant) and threshold.value == "mean", (
            "the explicit threshold must be the same rule threshold=None already "
            "resolved to for LogisticRegression(l2) and XGBRegressor -- stating it is a "
            "documentation change, not a behavioural one"
        )

    def test_the_pre_filter_is_invoked_inside_select_features(self) -> None:
        source = inspect.getsource(BaseTrainer.select_features)
        assert "informative_columns(" in source, (
            "select_features must call informative_columns; without it the zero-variance "
            "pre-filter is dead code and the invariance above is accidental"
        )


class TestThisPlanChangedTheRuleNotTheBudget:
    """The per-target feature budgets are incumbent values and are explicitly out of scope."""

    def test_the_three_budgets_are_unchanged(self) -> None:
        assert (_WP_MAX_FEATURES, _ATS_MAX_FEATURES, _OU_MAX_FEATURES) == (20, 25, 25)


class TestTheBaseFallbackAsymmetry:
    """``base.py``'s own train_and_evaluate passes max_features=None -- a different rule.

    The census recorded it as unreachable. This pins the two facts that make it so, so
    that a future subclass which forgets to override ``train_and_evaluate`` fails here
    rather than silently selecting 56 / 85 / 91 features instead of 20 / 25 / 25.
    """

    def test_base_trainer_cannot_be_instantiated(self) -> None:
        assert BaseTrainer.__abstractmethods__

    @pytest.mark.parametrize(("label", "trainer_cls", "cap", "binary"), _TARGETS)
    def test_every_concrete_trainer_overrides_train_and_evaluate(
        self, label: str, trainer_cls: type[Any], cap: int, binary: bool
    ) -> None:
        assert trainer_cls.train_and_evaluate is not BaseTrainer.train_and_evaluate, (
            f"{label} inherits the uncapped base train_and_evaluate"
        )

    @pytest.mark.parametrize(("label", "trainer_cls", "cap", "binary"), _TARGETS)
    def test_every_concrete_trainer_passes_its_own_budget(
        self, label: str, trainer_cls: type[Any], cap: int, binary: bool
    ) -> None:
        source = inspect.getsource(trainer_cls.train_and_evaluate)
        assert "max_features=" in source, (
            f"{label}'s train_and_evaluate does not pass max_features, so it would fall "
            "through to the uncapped rule"
        )
