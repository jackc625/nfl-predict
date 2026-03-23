"""Core backtest engine for walk-forward model evaluation.

Orchestrates per-season retraining of WP, ATS, and O/U models across
holdout seasons, collecting CLV-headlined metrics with per-season
breakdown. This is the computational backbone of Phase 6.

Provides:
- BacktestConfig: Configuration for backtest runs
- TargetResult: Per-target, per-season results container
- SeasonResult: Per-season results container
- BacktestResults: Full backtest output with headline CLV
- BacktestEngine: Orchestrator class

Key design decisions:
- Rolling HP-val: For holdout Y, hp_val=[Y-1], train=[first..Y-2]
- Fresh trainer instances per season (no state leakage)
- 2025 data filtered before processing
- CLV computed via models/clv.py (single source of truth)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.era import get_covid_hfa_annotation, get_season_total_weeks
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import BaseTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        holdout_seasons: Seasons to use as walk-forward holdout.
        first_data_season: Earliest season in the data.
        targets: Model targets to evaluate.
        max_backtest_season: Maximum season to include (filters incomplete seasons).
    """

    holdout_seasons: list[int] = field(
        default_factory=lambda: [2021, 2022, 2023, 2024]
    )
    first_data_season: int = 2018
    targets: list[str] = field(default_factory=lambda: ["wp", "ats", "ou"])
    max_backtest_season: int = 2024


@dataclass
class TargetResult:
    """Results for a single target within a single holdout season.

    Attributes:
        target: Model target type (wp/ats/ou).
        season: Holdout season.
        predictions_df: DataFrame of predictions for this season.
        clv_df: CLV computation results (from models/clv.py), or None.
        metrics: Metrics dict from the trainer.
        feature_names: Selected features for this run.
        best_params: Tuned hyperparameters.
    """

    target: str
    season: int
    predictions_df: pd.DataFrame
    clv_df: pd.DataFrame | None
    metrics: dict[str, Any]
    feature_names: list[str]
    best_params: dict


@dataclass
class SeasonResult:
    """Results for all targets within a single holdout season.

    Attributes:
        season: Holdout season.
        target_results: Mapping from target name to TargetResult.
        split_config: The temporal split used for this season.
    """

    season: int
    target_results: dict[str, TargetResult]
    split_config: TemporalSplitConfig


@dataclass
class BacktestResults:
    """Complete backtest output across all seasons and targets.

    Attributes:
        config: The BacktestConfig used.
        season_results: Per-season results list.
        all_predictions: All predictions per target, concatenated across seasons.
        all_clv: All CLV results per target, concatenated across seasons.
        headline_clv: Mean CLV per target across all seasons (the headline metric).
        odds_coverage: Games with/without closing odds per target.
        covid_annotation: COVID-2020 context from get_covid_hfa_annotation().
        era_info: Season total weeks mapping.
    """

    config: BacktestConfig
    season_results: list[SeasonResult]
    all_predictions: dict[str, pd.DataFrame]
    all_clv: dict[str, pd.DataFrame]
    headline_clv: dict[str, float]
    odds_coverage: dict[str, int]
    covid_annotation: dict
    era_info: dict


# -----------------------------------------------------------------------
# Trainer factory
# -----------------------------------------------------------------------

_TRAINER_MAP: dict[str, type[BaseTrainer]] = {
    "wp": WPTrainer,
    "ats": ATSTrainer,
    "ou": OUTrainer,
}


# -----------------------------------------------------------------------
# BacktestEngine
# -----------------------------------------------------------------------


class BacktestEngine:
    """Walk-forward backtest engine for NFL prediction models.

    For each holdout season, creates a fresh temporal split, instantiates
    fresh trainer instances, runs training + evaluation, and collects
    CLV-headlined results.

    Usage::

        engine = BacktestEngine()
        results = engine.run()
        print(results.headline_clv)
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize the backtest engine.

        Args:
            config: Backtest configuration. Defaults to BacktestConfig().
        """
        self.config = config or BacktestConfig()
        self.logger = get_logger(__name__)

    def _create_split_config(self, holdout_season: int) -> TemporalSplitConfig:
        """Create a TemporalSplitConfig for a single holdout season.

        Uses ROLLING hp_val strategy: for holdout Y, hp_val=[Y-1],
        train=[first_data_season .. Y-2]. This satisfies temporal
        ordering constraints and recalibrates per season.

        Args:
            holdout_season: The season to hold out for evaluation.

        Returns:
            Validated TemporalSplitConfig.
        """
        train_seasons = list(
            range(self.config.first_data_season, holdout_season - 1)
        )
        hp_val_seasons = [holdout_season - 1]
        holdout_seasons = [holdout_season]

        config = TemporalSplitConfig(
            train_seasons=train_seasons,
            hp_val_seasons=hp_val_seasons,
            holdout_seasons=holdout_seasons,
        )
        config.validate()
        return config

    def _create_trainer(
        self, target: str, config: TemporalSplitConfig
    ) -> BaseTrainer:
        """Create a fresh trainer instance for the given target.

        Always creates a NEW instance to prevent state leakage between
        holdout seasons.

        Args:
            target: Model target type (wp/ats/ou).
            config: Temporal split configuration.

        Returns:
            Fresh BaseTrainer subclass instance.

        Raises:
            ValueError: If target is not recognized.
        """
        trainer_class = _TRAINER_MAP.get(target)
        if trainer_class is None:
            msg = f"Unknown target: '{target}'. Must be one of {list(_TRAINER_MAP.keys())}"
            raise ValueError(msg)
        return trainer_class(config=config)

    def _load_features(self, target: str) -> pd.DataFrame:
        """Load Gold feature matrix for the given target.

        Filters to seasons <= max_backtest_season to exclude incomplete
        seasons (e.g. 2025).

        Args:
            target: Model target type (wp/ats/ou).

        Returns:
            Filtered feature DataFrame.
        """
        features_path = Path(f"data/gold/features_{target}.parquet")
        self.logger.info(
            "Loading features",
            target=target,
            path=str(features_path),
        )
        df = pd.read_parquet(features_path)

        # Filter to max_backtest_season
        pre_filter_count = len(df)
        df = df[df["season"] <= self.config.max_backtest_season].copy()
        post_filter_count = len(df)

        if pre_filter_count != post_filter_count:
            self.logger.info(
                "Filtered future seasons from features",
                target=target,
                removed=pre_filter_count - post_filter_count,
                remaining=post_filter_count,
            )

        return df

    def _load_closing_odds(self) -> pd.DataFrame:
        """Load closing odds from Silver layer.

        Returns:
            DataFrame with game_id, ml_home, ml_away, spread, total columns.
        """
        odds_path = Path("data/silver/odds_historical.parquet")
        self.logger.info("Loading closing odds", path=str(odds_path))
        df = pd.read_parquet(odds_path)
        return df

    def run(self) -> BacktestResults:
        """Execute the full walk-forward backtest.

        For each holdout season:
        1. Create temporal split config (rolling hp_val)
        2. For each target: load features, create trainer, train_and_evaluate
        3. Collect per-season, per-target results

        After all seasons:
        - Concatenate predictions and CLV across seasons per target
        - Compute headline CLV
        - Compute odds coverage stats
        - Add COVID annotation and era info

        Returns:
            BacktestResults with all metrics, predictions, and CLV.
        """
        run_start = time.monotonic()
        self.logger.info(
            "Starting backtest",
            holdout_seasons=self.config.holdout_seasons,
            targets=self.config.targets,
        )

        # Load closing odds once for all targets/seasons
        closing_odds_df = self._load_closing_odds()

        season_results: list[SeasonResult] = []
        all_predictions: dict[str, list[pd.DataFrame]] = {
            t: [] for t in self.config.targets
        }
        all_clv: dict[str, list[pd.DataFrame]] = {
            t: [] for t in self.config.targets
        }

        for holdout_season in self.config.holdout_seasons:
            season_start = time.monotonic()
            split_config = self._create_split_config(holdout_season)

            self.logger.info(
                "Processing holdout season",
                season=holdout_season,
                train=split_config.train_seasons,
                hp_val=split_config.hp_val_seasons,
            )

            target_results: dict[str, TargetResult] = {}

            for target in self.config.targets:
                target_start = time.monotonic()

                # Load features and create fresh trainer
                features_df = self._load_features(target)
                trainer = self._create_trainer(target, split_config)

                # Train and evaluate
                result = trainer.train_and_evaluate(features_df, closing_odds_df)

                # Extract predictions from trainer result
                # The trainer's season_results contains per-season metrics
                # and clv_results is the CLV DataFrame
                season_metrics_list = result.get("season_results", [])

                # Find the metrics for this holdout season
                season_metrics = {}
                for sm in season_metrics_list:
                    if sm.get("season") == holdout_season:
                        season_metrics = sm
                        break

                # Build predictions DataFrame from CLV results if available
                predictions_df = pd.DataFrame()
                clv_df = result.get("clv_results")

                if clv_df is not None and not clv_df.empty:
                    # Filter to this holdout season
                    if "season" in clv_df.columns:
                        season_preds = clv_df[
                            clv_df["season"] == holdout_season
                        ].copy()
                    else:
                        season_preds = clv_df.copy()

                    predictions_df = season_preds
                    all_clv[target].append(season_preds)

                all_predictions[target].append(predictions_df)

                target_results[target] = TargetResult(
                    target=target,
                    season=holdout_season,
                    predictions_df=predictions_df,
                    clv_df=clv_df,
                    metrics=season_metrics,
                    feature_names=result.get("feature_names", []),
                    best_params=result.get("best_params", {}),
                )

                elapsed = time.monotonic() - target_start
                self.logger.info(
                    "Target complete",
                    target=target,
                    season=holdout_season,
                    duration_s=round(elapsed, 1),
                    metrics=season_metrics,
                )

            season_results.append(
                SeasonResult(
                    season=holdout_season,
                    target_results=target_results,
                    split_config=split_config,
                )
            )

            season_elapsed = time.monotonic() - season_start
            self.logger.info(
                "Season complete",
                season=holdout_season,
                duration_s=round(season_elapsed, 1),
            )

        # Concatenate all predictions and CLV per target
        concat_predictions: dict[str, pd.DataFrame] = {}
        concat_clv: dict[str, pd.DataFrame] = {}

        for target in self.config.targets:
            pred_frames = [df for df in all_predictions[target] if not df.empty]
            clv_frames = [df for df in all_clv[target] if not df.empty]

            concat_predictions[target] = (
                pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
            )
            concat_clv[target] = (
                pd.concat(clv_frames, ignore_index=True) if clv_frames else pd.DataFrame()
            )

        # Compute headline CLV: mean probability_clv for games with closing odds
        headline_clv: dict[str, float] = {}
        for target in self.config.targets:
            clv_data = concat_clv[target]
            if not clv_data.empty and "probability_clv" in clv_data.columns:
                valid_clv = clv_data[clv_data["has_closing_odds"] == True]  # noqa: E712
                if not valid_clv.empty:
                    headline_clv[target] = float(
                        valid_clv["probability_clv"].mean()
                    )

        # Compute odds coverage
        odds_coverage: dict[str, int] = {}
        for target in self.config.targets:
            clv_data = concat_clv[target]
            if not clv_data.empty and "has_closing_odds" in clv_data.columns:
                odds_coverage[f"{target}_with_odds"] = int(
                    clv_data["has_closing_odds"].sum()
                )
                odds_coverage[f"{target}_without_odds"] = int(
                    (~clv_data["has_closing_odds"]).sum()
                )

        # Build era info
        era_info: dict[int, int] = {}
        for season in self.config.holdout_seasons:
            era_info[season] = get_season_total_weeks(season)

        total_elapsed = time.monotonic() - run_start
        self.logger.info(
            "Backtest complete",
            duration_s=round(total_elapsed, 1),
            headline_clv=headline_clv,
        )

        return BacktestResults(
            config=self.config,
            season_results=season_results,
            all_predictions=concat_predictions,
            all_clv=concat_clv,
            headline_clv=headline_clv,
            odds_coverage=odds_coverage,
            covid_annotation=get_covid_hfa_annotation(),
            era_info=era_info,
        )
