"""Temporal split infrastructure for walk-forward model training.

Provides:
- TemporalSplitConfig: Three-fold temporal split (train / HP-validation / holdout)
- WalkForwardSplitter: Expanding-window walk-forward splits within holdout
- make_temporal_cv_splits: Temporal CV folds for hyperparameter tuning

All splits enforce strict temporal ordering to prevent future data leakage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models.utils import TrainTestSplit
from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Three-Fold Temporal Split Configuration
# ---------------------------------------------------------------------------


@dataclass
class TemporalSplitConfig:
    """Three-fold temporal split configuration.

    Partitions seasons into three non-overlapping groups:
    - train_seasons: Feature selection and initial model fitting
    - hp_val_seasons: Hyperparameter tuning (temporal CV within this)
    - holdout_seasons: Walk-forward reporting only -- never tune here

    Attributes:
        train_seasons: Seasons for training.
        hp_val_seasons: Seasons for hyperparameter validation.
        holdout_seasons: Seasons for holdout walk-forward evaluation.
    """

    train_seasons: list[int] = field(default_factory=list)
    hp_val_seasons: list[int] = field(default_factory=list)
    holdout_seasons: list[int] = field(default_factory=list)

    def validate(self) -> None:
        """Ensure no overlap and strict temporal ordering.

        Raises:
            ValueError: If seasons overlap or temporal ordering is violated.
        """
        train_set = set(self.train_seasons)
        hp_val_set = set(self.hp_val_seasons)
        holdout_set = set(self.holdout_seasons)

        # Check for overlap
        if train_set & hp_val_set:
            overlap = sorted(train_set & hp_val_set)
            msg = f"Seasons overlap between train and hp_val: {overlap}"
            raise ValueError(msg)
        if train_set & holdout_set:
            overlap = sorted(train_set & holdout_set)
            msg = f"Seasons overlap between train and holdout: {overlap}"
            raise ValueError(msg)
        if hp_val_set & holdout_set:
            overlap = sorted(hp_val_set & holdout_set)
            msg = f"Seasons overlap between hp_val and holdout: {overlap}"
            raise ValueError(msg)

        # An empty hp_val fold is REJECTED, explicitly and by name.
        #
        # It is tempting to tolerate one for a screen run with tune=False, on the
        # reasoning that hp_val feeds only combined_train / combined_targets in
        # BaseTrainer.train_and_evaluate (base.py:355-360), consumed only inside
        # the `if tune` branch (base.py:361-368). That reasoning is TRUE of
        # BaseTrainer and FALSE of every concrete trainer: all three override
        # train_and_evaluate and fit a post-hoc conversion component on the
        # hp-val fold OUTSIDE the tune branch --
        #
        #   wp_trainer.py:310-326   Platt/isotonic probability calibrator
        #   ats_trainer.py:244-250  ResidualDistributionConverter on residuals
        #   ou_trainer.py:244-252   same pattern
        #
        # With an empty fold WP raises inside StandardScaler and ATS/OU fit a
        # converter with residual_std = np.std([]) = NaN -- i.e. one hard crash
        # and two SILENTLY degenerate models. Rejecting here with a message that
        # names the cause is worth more than the bare "min() iterable argument is
        # empty" this used to raise from the ordering check below.
        if not self.hp_val_seasons:
            msg = (
                "hp_val_seasons must not be empty: every trainer fits a "
                "calibration/conversion component on the hp-val fold OUTSIDE the "
                "tuning branch (wp_trainer.py:310-326 probability calibrator, "
                "ats_trainer.py:244-250 and ou_trainer.py:244-252 residual "
                "converter), so an empty fold crashes WP and silently gives "
                "ATS/OU a NaN-scale converter. Borrow a season into hp_val "
                "instead of leaving it empty."
            )
            raise ValueError(msg)

        # Check temporal ordering
        if max(self.train_seasons) >= min(self.hp_val_seasons):
            msg = (
                f"train seasons must precede hp_val seasons: "
                f"max(train)={max(self.train_seasons)} >= "
                f"min(hp_val)={min(self.hp_val_seasons)}"
            )
            raise ValueError(msg)

        if max(self.hp_val_seasons) >= min(self.holdout_seasons):
            msg = (
                f"hp_val seasons must precede holdout seasons: "
                f"max(hp_val)={max(self.hp_val_seasons)} >= "
                f"min(holdout)={min(self.holdout_seasons)}"
            )
            raise ValueError(msg)

        logger.info(
            "Temporal split config validated",
            train=self.train_seasons,
            hp_val=self.hp_val_seasons,
            holdout=self.holdout_seasons,
        )

    @classmethod
    def default(cls) -> TemporalSplitConfig:
        """Return the default temporal split configuration.

        Default boundaries:
        - Train: 2018-2019 (initial model fitting, feature selection)
        - HP-Validation: 2020 (COVID season absorbs anomaly)
        - Holdout: 2021-2024 (walk-forward reporting, CLV computation)
        """
        return cls(
            train_seasons=[2018, 2019],
            hp_val_seasons=[2020],
            holdout_seasons=[2021, 2022, 2023, 2024],
        )

    @property
    def all_seasons(self) -> list[int]:
        """Return sorted union of all three season lists."""
        return sorted(
            set(self.train_seasons)
            | set(self.hp_val_seasons)
            | set(self.holdout_seasons)
        )


# ---------------------------------------------------------------------------
# Walk-Forward Splitter
# ---------------------------------------------------------------------------

_DEFAULT_ID_COLS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "feature_timestamp",
    "target_wp",
    "target_ats",
    "target_ou",
    "home_win",
    "home_margin",
    "point_differential",
    "total_points",
    "home_covered_spread",
    "game_went_over",
]


class WalkForwardSplitter:
    """Walk-forward expanding-window splitter for holdout evaluation.

    For each holdout season Y, trains on ALL seasons < Y (expanding window),
    predicts season Y. This ensures no future data leakage and that the model
    accumulates more training data as holdout progresses.
    """

    def __init__(
        self,
        config: TemporalSplitConfig,
        target_col: str,
        id_cols: list[str] | None = None,
    ) -> None:
        """Initialize the splitter.

        Args:
            config: Temporal split configuration.
            target_col: Name of the target column.
            id_cols: Columns to exclude from features (metadata/ID columns).
                Defaults to ["game_id", "season", "week", "home_team", "away_team"].
        """
        self.config = config
        self.target_col = target_col
        self.id_cols = id_cols if id_cols is not None else list(_DEFAULT_ID_COLS)
        self.logger = get_logger(__name__)

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Determine feature columns by excluding ID, target, and non-numeric columns."""
        exclude = set(self.id_cols) | {self.target_col}
        numeric_df = df.select_dtypes(include=["number"])
        return [c for c in numeric_df.columns if c not in exclude]

    def generate_splits(self, features_df: pd.DataFrame) -> Iterator[TrainTestSplit]:
        """Generate expanding-window walk-forward splits for holdout seasons.

        For each holdout season Y:
        - Train on all rows where season < Y (includes train + hp_val + prior holdout)
        - Test on all rows where season == Y

        Args:
            features_df: DataFrame with ID cols, feature cols, and target col.

        Yields:
            TrainTestSplit for each holdout season.
        """
        feature_cols = self._feature_cols(features_df)

        # Set game_id as index so splits carry game IDs for CLV merge
        if "game_id" in features_df.columns:
            features_df = features_df.set_index("game_id", drop=True)

        for holdout_season in self.config.holdout_seasons:
            train_mask = features_df["season"] < holdout_season
            test_mask = features_df["season"] == holdout_season

            train_df = features_df[train_mask]
            test_df = features_df[test_mask]

            if len(train_df) == 0 or len(test_df) == 0:
                self.logger.warning(
                    "Skipping holdout season with no data",
                    holdout_season=holdout_season,
                    train_rows=len(train_df),
                    test_rows=len(test_df),
                )
                continue

            train_seasons = sorted(train_df["season"].unique().tolist())

            split = TrainTestSplit(
                train_data=train_df[feature_cols],
                train_targets=train_df[self.target_col],
                test_data=test_df[feature_cols],
                test_targets=test_df[self.target_col],
                train_seasons=train_seasons,
                test_season=holdout_season,
            )

            self.logger.info(
                "Generated walk-forward split",
                test_season=holdout_season,
                train_seasons=train_seasons,
                train_rows=len(train_df),
                test_rows=len(test_df),
                n_features=len(feature_cols),
            )

            yield split

    def get_train_val_split(self, features_df: pd.DataFrame) -> TrainTestSplit:
        """Return split for HP tuning: train=config.train_seasons, test=config.hp_val_seasons.

        Args:
            features_df: DataFrame with ID cols, feature cols, and target col.

        Returns:
            TrainTestSplit where train is the training window and test is the
            HP-validation window.
        """
        if "game_id" in features_df.columns:
            features_df = features_df.set_index("game_id", drop=True)

        feature_cols = self._feature_cols(features_df)

        train_mask = features_df["season"].isin(self.config.train_seasons)
        val_mask = features_df["season"].isin(self.config.hp_val_seasons)

        train_df = features_df[train_mask]
        val_df = features_df[val_mask]

        return TrainTestSplit(
            train_data=train_df[feature_cols],
            train_targets=train_df[self.target_col],
            test_data=val_df[feature_cols],
            test_targets=val_df[self.target_col],
            train_seasons=sorted(train_df["season"].unique().tolist()),
            test_season=self.config.hp_val_seasons[0],
        )


# ---------------------------------------------------------------------------
# Temporal CV Splits (for sklearn cross-validation)
# ---------------------------------------------------------------------------


def make_temporal_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produce temporal CV folds suitable for sklearn cross-validation.

    Sorts df by season then week and uses sklearn's TimeSeriesSplit to
    generate index arrays where each fold's test data is temporally after
    its train data.

    Args:
        df: DataFrame with at least "season" and "week" columns.
        n_splits: Number of temporal folds.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    sorted_df = df.sort_values(["season", "week"]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    for train_idx, test_idx in tscv.split(sorted_df):
        folds.append((train_idx, test_idx))

    logger.info(
        "Generated temporal CV splits",
        n_splits=n_splits,
        total_rows=len(sorted_df),
    )

    return folds
