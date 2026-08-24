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
- PER-RUN Optuna identity (Plan 30-16, D30-OWNER-02): each BacktestEngine instance mints
  its own run id and opts every trainer it builds into it, so a "tuned" backtest genuinely
  searches instead of resuming a study that was already at budget. Within one run the
  earliest holdout season fills the study and the later ones reuse it, which is why
  ascending holdout order is hard-enforced; every run writes tuning_provenance.json naming
  the trials it actually added.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.era import get_covid_hfa_annotation, get_season_total_weeks
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import BACKTEST_TUNING_STORAGE_DIR, BaseTrainer
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
        blend_config: Market blending configuration. None means no blending
            (Phase 6 baseline behavior).
    """

    holdout_seasons: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024])
    first_data_season: int = 2018
    targets: list[str] = field(default_factory=lambda: ["wp", "ats", "ou"])
    max_backtest_season: int = 2024
    blend_config: Any | None = None  # BlendConfig from models.blending


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
        is_blended: Whether market blending was applied to predictions.
    """

    config: BacktestConfig
    season_results: list[SeasonResult]
    all_predictions: dict[str, pd.DataFrame]
    all_clv: dict[str, pd.DataFrame]
    headline_clv: dict[str, float]
    odds_coverage: dict[str, int]
    covid_annotation: dict
    era_info: dict
    is_blended: bool = False


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

        Mints this instance's Optuna run id. It is per INSTANCE and not per second on
        purpose: ``backtest.run.run_backtest(blend=True)`` constructs and runs the engine
        TWICE (blended primary, then the unblended baseline), typically inside the same
        clock second, and a colliding id would make the second run a resume-at-budget of
        the first -- the exact vacuity Plan 30-16 exists to end (D30-DEFER-01,
        D30-OWNER-02). The timestamp is kept because a human reading
        ``outputs/optuna/backtest/`` needs to know WHEN, and the random suffix is what makes
        it unique.

        Args:
            config: Backtest configuration. Defaults to BacktestConfig().
        """
        self.config = config or BacktestConfig()
        self.logger = get_logger(__name__)
        self.run_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"
        )
        # Per (target, holdout season) record of what the tuned search actually did, filled
        # during run() and written out as tuning_provenance.json.
        self._tuning_provenance: list[dict[str, Any]] = []

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
        train_seasons = list(range(self.config.first_data_season, holdout_season - 1))
        hp_val_seasons = [holdout_season - 1]
        holdout_seasons = [holdout_season]

        config = TemporalSplitConfig(
            train_seasons=train_seasons,
            hp_val_seasons=hp_val_seasons,
            holdout_seasons=holdout_seasons,
        )
        config.validate()
        return config

    def _create_trainer(self, target: str, config: TemporalSplitConfig) -> BaseTrainer:
        """Create a fresh trainer instance for the given target.

        Always creates a NEW instance to prevent state leakage between
        holdout seasons.

        Every trainer is opted into THIS engine run's Optuna identity here rather than at
        the call site, so the identity cannot be forgotten for one target or one holdout
        season. Before Plan 30-16 the trainer kept the LEGACY identity and its "tuned"
        search resumed a study written 2026-03-31 that was already at budget: zero trials
        run, stored v2.0 parameters returned, a full trial count reported (D30-DEFER-01).

        Args:
            target: Model target type (wp/ats/ou).
            config: Temporal split configuration.

        Returns:
            Fresh BaseTrainer subclass instance, opted into this run's tuning identity.

        Raises:
            ValueError: If target is not recognized.
        """
        trainer_class = _TRAINER_MAP.get(target)
        if trainer_class is None:
            msg = f"Unknown target: '{target}'. Must be one of {list(_TRAINER_MAP.keys())}"
            raise ValueError(msg)
        trainer = trainer_class(config=config)
        trainer.use_backtest_tuning(self.run_id)
        return trainer

    def _validate_holdout_order(self) -> None:
        """Reject a holdout season list that is not STRICTLY ASCENDING.

        This guards a temporal property, not a style preference. One Optuna study per target
        per run is filled by the FIRST holdout season processed and RESUMED by every later
        one, so the whole run trains under hyperparameters searched on the first-processed
        season's train + hp_val window. Ascending order makes that safe -- later folds use
        parameters chosen from strictly prior data. A descending or shuffled list inverts it:
        holdout 2024's window (train 2018-2022, hp_val 2023) would supply the hyperparameters
        used to backtest holdout 2021, which is future information reaching a past fold.

        Raises rather than silently sorting, because silently reordering what the caller
        asked for hides the problem instead of reporting it. Called at the very top of
        ``run()`` so a bad config fails in a second rather than twenty minutes in.

        Raises:
            ValueError: If holdout_seasons is not strictly ascending.
        """
        seasons = list(self.config.holdout_seasons)
        if seasons == sorted(set(seasons)) and len(seasons) == len(set(seasons)):
            return

        msg = (
            f"holdout_seasons must be STRICTLY ASCENDING, got {seasons}. This is a temporal "
            "requirement, not a style rule: one Optuna study per target per run is filled by "
            "the FIRST holdout season processed and resumed by the rest, so the whole run "
            "trains under hyperparameters searched on that first season's train + hp_val "
            "window. Out of order, a LATER season's window would supply the hyperparameters "
            "used to backtest an EARLIER fold -- future information reaching a past fold. "
            f"Remediation: pass sorted(set(...)) = {sorted(set(seasons))}."
        )
        raise ValueError(msg)

    def _write_tuning_provenance(self) -> Path | None:
        """Write this run's tuning provenance record beside its study files.

        The record is what makes the honesty claim CHECKABLE by a reader instead of asserted
        by a plan: per (target, holdout season) it names the study, the trials it already
        held, and the trials this run actually ADDED. A genuine run shows a full budget added
        on the first season of each target and zero on the rest; the old vacuous behaviour
        showed zero added on ALL of them against a study written 2026-03-31.

        Returns:
            The path written, or None when the run tuned nothing (tune=False paths).
        """
        if not self._tuning_provenance:
            return None

        totals: dict[str, int] = {}
        for row in self._tuning_provenance:
            added = row["trials_added"]
            totals[row["target"]] = totals.get(row["target"], 0) + (
                int(added) if added is not None else 0
            )

        record = {
            "run_id": self.run_id,
            "written_at_utc": datetime.now(UTC).isoformat(),
            "holdout_seasons": list(self.config.holdout_seasons),
            "targets": list(self.config.targets),
            "per_target_trials_added": totals,
            "folds": self._tuning_provenance,
            "how_to_read_this": (
                "trials_added > 0 means the search genuinely ran that many trials. Within ONE "
                "run the study name carries the run and not the holdout season, so the "
                "EARLIEST season fills the study and every later season resumes it and adds "
                "ZERO -- those zeros are expected and are stated within-run reuse, not the "
                "cross-run resume Plan 30-16 exists to end. Ascending holdout order is what "
                "makes that reuse temporally safe and is hard-enforced by "
                "BacktestEngine._validate_holdout_order. Before Plan 30-16 this same command "
                "added zero trials on EVERY fold and returned parameters searched 2026-03-31 "
                "while reporting a full trial count (D30-DEFER-01, D30-OWNER-02)."
            ),
        }

        out_dir = BACKTEST_TUNING_STORAGE_DIR / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tuning_provenance.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.logger.info(
            "Tuning provenance written",
            path=str(out_path),
            per_target_trials_added=totals,
        )
        return out_path

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
        odds_path = Path("data/silver/odds_snapshot.parquet")
        self.logger.info("Loading closing odds", path=str(odds_path))
        df = pd.read_parquet(odds_path)

        # Normalize team abbreviations in game_ids (e.g. LAR -> LA)
        from utils.exceptions import DataValidationError
        from utils.team_data import normalize_team_abbreviation

        def _normalize_game_id(gid: str) -> str:
            parts = gid.split("_")
            if len(parts) >= 3:
                matchup = parts[2]
                sep = "@" if "@" in matchup else "_"
                teams = matchup.split(sep)
                if len(teams) == 2:
                    try:
                        normalized = sep.join(
                            normalize_team_abbreviation(t) for t in teams
                        )
                        return "_".join([*parts[:2], normalized, *parts[3:]])
                    except (ValueError, KeyError, DataValidationError):
                        pass
            return gid

        df["game_id"] = df["game_id"].apply(_normalize_game_id)
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
        # Before ANY loading, so a bad config fails in a second rather than twenty minutes in.
        self._validate_holdout_order()

        run_start = time.monotonic()
        self.logger.info(
            "Starting backtest",
            run_id=self.run_id,
            holdout_seasons=self.config.holdout_seasons,
            targets=self.config.targets,
        )

        # Load closing odds once for all targets/seasons
        closing_odds_df = self._load_closing_odds()

        season_results: list[SeasonResult] = []
        all_predictions: dict[str, list[pd.DataFrame]] = {
            t: [] for t in self.config.targets
        }
        all_clv: dict[str, list[pd.DataFrame]] = {t: [] for t in self.config.targets}

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

                # Record what this fold's tuned search actually did. None means the fold did
                # not tune at all (a tune=False path), which is recorded as such rather than
                # as a zero -- a zero and an absence are different claims.
                self._tuning_provenance.append(
                    {
                        "target": target,
                        "holdout_season": holdout_season,
                        "train_seasons": list(split_config.train_seasons),
                        "hp_val_seasons": list(split_config.hp_val_seasons),
                        "study_name": trainer.last_tuning_study_name,
                        "trials_before": trainer.last_tuning_trials_before,
                        "trials_added": trainer.last_tuning_trials_added,
                        "tuned": trainer.last_tuning_study_name is not None,
                    }
                )

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
                        season_preds = clv_df[clv_df["season"] == holdout_season].copy()
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
                pd.concat(pred_frames, ignore_index=True)
                if pred_frames
                else pd.DataFrame()
            )
            concat_clv[target] = (
                pd.concat(clv_frames, ignore_index=True)
                if clv_frames
                else pd.DataFrame()
            )

        # Apply market blending if configured
        is_blended = False
        if self.config.blend_config is not None:
            from models.blending import MarketBlender

            blender = MarketBlender(config=self.config.blend_config)
            for target in self.config.targets:
                if not concat_predictions[target].empty:
                    concat_predictions[target] = blender.blend_predictions(
                        concat_predictions[target], closing_odds_df, target
                    )
                    # Run edge rate diagnostic
                    edge_report = blender.check_weekly_edge_rate(
                        concat_predictions[target], closing_odds_df, target
                    )
                    self.logger.info(
                        "Blend edge diagnostic",
                        target=target,
                        mean_flag_rate=edge_report["mean_rate"],
                        n_warnings=len(edge_report["warnings"]),
                    )

            # Recompute CLV from blended predictions.
            # Drop existing CLV/odds columns first to avoid merge conflicts,
            # since concat_predictions already contains merged odds from
            # the initial CLV computation.
            from models.clv import compute_clv_for_predictions

            clv_odds_cols = [
                "probability_clv",
                "fair_closing_prob",
                "has_closing_odds",
                "line_clv",
                "ml_home",
                "ml_away",
                "spread",
                "total",
            ]
            for target in self.config.targets:
                if not concat_predictions[target].empty:
                    drop_cols = [
                        c
                        for c in clv_odds_cols
                        if c in concat_predictions[target].columns
                    ]
                    clean_preds = concat_predictions[target].drop(columns=drop_cols)
                    concat_clv[target] = compute_clv_for_predictions(
                        clean_preds, closing_odds_df, target
                    )
                    # Update concat_predictions with recomputed CLV data
                    concat_predictions[target] = concat_clv[target]

            is_blended = True

        # Compute headline CLV: mean probability_clv for games with closing odds
        headline_clv: dict[str, float] = {}
        for target in self.config.targets:
            clv_data = concat_clv[target]
            if not clv_data.empty and "probability_clv" in clv_data.columns:
                valid_clv = clv_data[clv_data["has_closing_odds"] == True]  # noqa: E712
                if not valid_clv.empty:
                    headline_clv[target] = float(valid_clv["probability_clv"].mean())

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

        self._write_tuning_provenance()

        total_elapsed = time.monotonic() - run_start
        self.logger.info(
            "Backtest complete",
            run_id=self.run_id,
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
            is_blended=is_blended,
        )
