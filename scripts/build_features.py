#!/usr/bin/env python3
"""
Unified Feature Building Pipeline

This script combines all feature sources into complete feature matrices:
- Team form metrics (rolling 4-week EPA, success rates)
- Elo ratings (with uncertainty and home field advantage)
- Contextual features (travel, weather, venue, rest)
- Weather features (wind, temperature, precipitation - outdoor only)
- Market anchor features (opening/snapshot lines, devigged probabilities)

Feature processing includes:
- Missing data imputation and outlier handling (winsorization)
- Expanding-window normalization (per-season, with prior-season bootstrap)
- LeakageGate hard-fail validation (per-builder + combined matrix)
- Separate feature matrices for WP, ATS, and O/U targets
- Storage in gold layer for model consumption

Usage:
    python scripts/build_features.py --season 2024 --week 1
    python scripts/build_features.py --season 2024  # All weeks in season
    python scripts/build_features.py  # All available data
    python scripts/build_features.py --season 2024 --as-of 2024-10-04T18:00:00
"""

import argparse
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.contextual import ContextualFeaturesCalculator
from features.elo_features import EloFeatureBuilder
from features.injury import InjuryBuilder
from features.line_movement import LineMovementBuilder
from features.market_anchors import MarketAnchorFeaturesCalculator
from features.normalization import compute_prior_season_stats, expanding_normalize
from features.opponent_adj import OpponentAdjuster
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder
from features.team_form import TeamFormCalculator
from features.validation import LeakageGate, LeakageViolation
from features.weather import WeatherFeaturesCalculator
from utils import get_logger
from utils.date_utils import ET
from utils.exceptions import DataIngestionError

logger = get_logger(__name__)

# Every optional-source guard below catches THIS tuple, and nothing else.
#
# This is not a line-movement problem. ``DataIngestionError`` derives from
# ``NFLPredictException`` and therefore from ``Exception`` directly -- it is NOT a
# subclass of ValueError, OSError or FileNotFoundError -- and ``load_dataframe``
# raises it (``data/storage.py:923``) whenever a table is absent from both DuckDB
# and parquet. Every optional silver source below guards a ``load_dataframe`` call
# or a builder that makes one, so before CR-03 they ALL shared the same false
# graceful-degradation contract: on a fresh clone the guard could not fire, the
# error escaped, and gold could not be built at all -- which falsifies the
# degradation contract this module asserts in prose for the line-movement builder.
#
# One constant is what stops the twelve sites drifting apart again.
_SOURCE_LOAD_ERRORS = (
    DataIngestionError,
    ValueError,
    KeyError,
    TypeError,
    FileNotFoundError,
    OSError,
)


class FeatureMatrixBuilder:
    """
    Build unified feature matrices from all feature sources.

    Combines team form, Elo, contextual, weather, and market features
    into complete feature matrices ready for model training.
    """

    # WR-06: the minimum number of non-null values a FIT SOURCE must hold before it
    # can support an imputation median or a winsorization bound. The threshold is
    # preserved verbatim from the pre-WR-06 shape (``notna().sum() > 10``); what
    # moved is WHAT it is applied to -- the strictly-prior slice the statistic is
    # actually estimated from, rather than the whole multi-season column.
    _MIN_FIT_POINTS = 10

    def __init__(self):
        """Initialize feature matrix builder."""
        self.logger = get_logger(__name__)

        # Feature calculators
        self.team_form_calc = TeamFormCalculator()
        self.elo_calc = EloFeatureBuilder()
        self.contextual_calc = ContextualFeaturesCalculator()
        self.weather_calc = WeatherFeaturesCalculator()
        self.market_calc = MarketAnchorFeaturesCalculator()
        self.qb_tracker = QBTracker()
        self.opponent_adj = OpponentAdjuster(window=10, min_opponent_games=4)

        # Snaps are constructed (and invoked) BEFORE injuries (D-09 build order):
        # InjuryBuilder consumes the SnapCountBuilder per-position prior shares as
        # its availability weights via the constructor handoff (review #6), which
        # LOCKS the snaps-before-injuries order rather than relying on an implicit
        # re-load. The contextual situational-spot features (Plan 28-04) already
        # flow through self.contextual_calc; no separate builder is needed here.
        self.snap_builder = SnapCountBuilder()
        self.injury_builder = InjuryBuilder(snap_builder=self.snap_builder)

        # Line-movement features (Phase 29, SIG-04) read the additive
        # `odds_timeline` trajectory silver. The builder degrades to neutral,
        # non-null defaults plus `line_movement_coverage` = 0.0 when the table is
        # absent or a game has no pre-freeze trajectory, so it is safe to
        # construct unconditionally.
        self.line_movement_builder = LineMovementBuilder()

        # Leakage gate for hard-fail validation
        self.leakage_gate = LeakageGate()

        # Feature processing parameters
        self.outlier_percentiles = (1, 99)  # Winsorization bounds
        self.min_games_for_stats = 10  # Minimum games for normalization

        # WR-06: the machine-readable self-fit flag. Maps a column name to the
        # seasons whose imputation median / winsorization bounds were fitted on
        # their OWN rows because no usable strictly-prior slice existed. Reset at
        # the start of every ``handle_missing_data_and_outliers`` call and logged at
        # its end, so the flag is observable in a build log AND assertable by a
        # test. A self-fit outside the earliest data-bearing season is a per-column
        # coverage floor -- a fact worth surfacing, which a log line alone would
        # never make checkable.
        self.self_fit_seasons: dict[str, list[int]] = {}

    def load_all_feature_sources(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all feature sources from silver layer.

        Args:
            target_season: Specific season to load
            target_week: Specific week to load
            as_of_datetime: Time-fence cutoff for builders that need it
                (QBTracker, OpponentAdjuster). Defaults to ``datetime.now(ET)``.

        Returns:
            Dictionary with all feature DataFrames
        """
        # CR-01: the default MUST be tz-aware. A naive ``datetime.now()`` is a LOCAL
        # wall clock, and every downstream consumer re-labels it as UTC without
        # shifting it (``ensure_utc_aware`` reinterprets, and
        # ``LeakageGate.check_time_fence`` tz_localizes). On the owner's ET machine
        # the Friday 18:05 ET orchestrator slot became 18:05Z == 14:05 ET, silently
        # fencing OUT the Friday-6PM-ET freeze snapshot that the whole D-12 cadence
        # exists to capture; east of UTC the same bug points the other way and LEAKS.
        # ET is the project's canonical wall clock (kickoffs, the Friday freeze), so
        # that is what "now" means here; being aware, it CONVERTS correctly instead
        # of being reinterpreted.
        if as_of_datetime is None:
            as_of_datetime = datetime.now(ET)
        logger.info(
            "Loading all feature sources",
            target_season=target_season,
            target_week=target_week,
        )

        feature_sources = {}

        try:
            # Core game data.
            #
            # WR-10: filter by season when a season is given, and narrow by week only
            # when a week is ALSO given. This previously required BOTH, while
            # LineMovementBuilder.build_features filters on each independently and
            # main() declares --season and --week as independent options -- so a
            # season-only build left most games unfiltered here, produced
            # line-movement rows only for the target season, and left the rest NaN
            # after the left merge. Those NaNs were then filled with the column
            # median, which for line_movement_coverage is 1.0, stamping every
            # uncovered game as covered and inverting the flag the family's whole
            # semantics rest on.
            games_df = load_dataframe("games", layer="silver")
            if target_season:
                games_df = games_df[games_df["season"] == target_season]
                if target_week:
                    games_df = games_df[games_df["week"] == target_week]
            feature_sources["games"] = games_df
            logger.info("Loaded games data", records=len(games_df))

            # Team form features
            try:
                team_form_df = load_dataframe("team_form_features", layer="silver")
                if target_season and target_week:
                    team_form_df = team_form_df[
                        (team_form_df["target_season"] == target_season)
                        & (team_form_df["target_week"] == target_week)
                    ]
                feature_sources["team_form"] = team_form_df
                logger.info("Loaded team form features", records=len(team_form_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load team form features", error=str(e))
                feature_sources["team_form"] = pd.DataFrame()

            # Elo features (computed on-the-fly via EloFeatureBuilder)
            try:
                elo_df = self.elo_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["elo"] = elo_df
                logger.info("Built Elo features", records=len(elo_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build Elo features", error=str(e))
                feature_sources["elo"] = pd.DataFrame()

            # Contextual features (computed on-the-fly for Phase 5 additions)
            try:
                contextual_df = self.contextual_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["contextual"] = contextual_df
                logger.info("Built contextual features", records=len(contextual_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build contextual features", error=str(e))
                feature_sources["contextual"] = pd.DataFrame()

            # Weather features
            try:
                weather_df = load_dataframe("weather_features", layer="silver")
                if target_season and target_week:
                    weather_df = weather_df[
                        (weather_df["season"] == target_season)
                        & (weather_df["week"] == target_week)
                    ]
                feature_sources["weather"] = weather_df
                logger.info("Loaded weather features", records=len(weather_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load weather features", error=str(e))
                feature_sources["weather"] = pd.DataFrame()

            # Market anchor features (computed on-the-fly via MarketAnchorFeaturesCalculator)
            try:
                market_df = self.market_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["market"] = market_df
                logger.info("Built market anchor features", records=len(market_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build market anchor features", error=str(e))
                feature_sources["market"] = pd.DataFrame()

            # QB tracking features (computed via QBTracker, not loaded from silver)
            try:
                qb_features_df = self.qb_tracker.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["qb_tracking"] = qb_features_df
                logger.info("Loaded QB tracking features", records=len(qb_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load QB tracking features", error=str(e))
                feature_sources["qb_tracking"] = pd.DataFrame()

            # Snap-count features (computed via SnapCountBuilder). CRITICAL build
            # order (D-09): snaps are invoked BEFORE injuries so the per-position
            # prior snap shares are available to the injury availability metric.
            try:
                snap_features_df = self.snap_builder.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["snaps"] = snap_features_df
                logger.info("Built snap-count features", records=len(snap_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build snap-count features", error=str(e))
                feature_sources["snaps"] = pd.DataFrame()

            # Injury features (computed via InjuryBuilder AFTER snaps; the builder
            # draws its D-09 availability weights from the constructor-injected
            # SnapCountBuilder's per-position prior shares, review #6).
            try:
                injury_features_df = self.injury_builder.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["injury"] = injury_features_df
                logger.info("Built injury features", records=len(injury_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build injury features", error=str(e))
                feature_sources["injury"] = pd.DataFrame()

            # Line-movement features (Phase 29, SIG-04; computed via
            # LineMovementBuilder from the `odds_timeline` trajectory silver).
            # Registration here routes the source through the LeakageGate; the
            # EXPLICIT merge block in combine_features is what actually lands the
            # columns in gold (the 28-06 lesson -- both seams are mandatory).
            try:
                line_movement_df = self.line_movement_builder.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["line_movement"] = line_movement_df
                logger.info(
                    "Built line-movement features", records=len(line_movement_df)
                )
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build line-movement features", error=str(e))
                feature_sources["line_movement"] = pd.DataFrame()

            return feature_sources

        except _SOURCE_LOAD_ERRORS as e:
            logger.error("Failed to load feature sources", error=str(e))
            raise

    def combine_features(
        self, feature_sources: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all feature sources into unified feature matrix.

        Args:
            feature_sources: Dictionary with all feature DataFrames

        Returns:
            Combined feature matrix
        """
        logger.info("Combining all feature sources")

        # Start with games as base
        games_df = feature_sources["games"].copy()

        if len(games_df) == 0:
            logger.error("No games data available")
            return pd.DataFrame()

        # Initialize combined features with game identifiers
        combined_features = games_df[
            ["game_id", "season", "week", "home_team", "away_team"]
        ].copy()

        # Add game results if available (for target creation)
        if "home_score" in games_df.columns:
            combined_features["home_score"] = games_df["home_score"]
        if "away_score" in games_df.columns:
            combined_features["away_score"] = games_df["away_score"]

        # Merge each feature source
        feature_counts = {}

        # Team form features (need to handle home/away separately)
        team_form_df = feature_sources.get("team_form", pd.DataFrame())
        if len(team_form_df) > 0:
            home_form = self._get_team_features(
                team_form_df, combined_features, "home_team", "home"
            )
            away_form = self._get_team_features(
                team_form_df, combined_features, "away_team", "away"
            )
            combined_features = combined_features.merge(
                home_form, on="game_id", how="left"
            )
            combined_features = combined_features.merge(
                away_form, on="game_id", how="left"
            )
            feature_counts["team_form"] = len(
                [col for col in combined_features.columns if "form_" in col]
            )

        # Elo features (game-level, already has home/away columns from EloFeatureBuilder)
        elo_df = feature_sources.get("elo", pd.DataFrame())
        if len(elo_df) > 0:
            from features.elo_features import ELO_FEATURE_COLUMNS

            merge_cols = ["game_id"] + [
                c for c in ELO_FEATURE_COLUMNS if c in elo_df.columns
            ]
            combined_features = combined_features.merge(
                elo_df[merge_cols], on="game_id", how="left"
            )
            feature_counts["elo"] = len(
                [col for col in combined_features.columns if "elo_" in col]
            )

        # Contextual features (game-level)
        contextual_df = feature_sources.get("contextual", pd.DataFrame())
        if len(contextual_df) > 0:
            merge_cols = ["game_id"]
            contextual_features = contextual_df.drop(
                columns=["season", "week", "home_team", "away_team"], errors="ignore"
            )
            combined_features = combined_features.merge(
                contextual_features, on=merge_cols, how="left"
            )
            feature_counts["contextual"] = len(
                [col for col in contextual_features.columns if col != "game_id"]
            )

        # Weather features (game-level)
        weather_df = feature_sources.get("weather", pd.DataFrame())
        if len(weather_df) > 0:
            merge_cols = ["game_id"]
            weather_features = weather_df.drop(
                columns=["season", "week"], errors="ignore"
            )
            combined_features = combined_features.merge(
                weather_features, on=merge_cols, how="left"
            )
            feature_counts["weather"] = len(
                [col for col in weather_features.columns if col != "game_id"]
            )

        # Market anchor features (game-level, compressed 5 features)
        market_df = feature_sources.get("market", pd.DataFrame())
        if len(market_df) > 0:
            market_feature_cols = [
                "snapshot_spread",
                "snapshot_total",
                "snapshot_ml_prob_home_fair",
                "spread_movement",
                "total_movement",
            ]
            merge_cols = ["game_id"] + [
                c for c in market_feature_cols if c in market_df.columns
            ]
            combined_features = combined_features.merge(
                market_df[merge_cols], on="game_id", how="left"
            )
            feature_counts["market"] = len(
                [c for c in market_feature_cols if c in market_df.columns]
            )

        # QB adjustment features (one value per team per game)
        qb_df = feature_sources.get("qb_tracking", pd.DataFrame())
        if len(qb_df) > 0 and "qb_adjustment" in qb_df.columns:
            # Merge home QB adjustment
            home_qb = qb_df[["game_id", "team", "qb_adjustment"]].copy()
            home_qb = home_qb.merge(
                combined_features[["game_id", "home_team"]].drop_duplicates(),
                left_on=["game_id", "team"],
                right_on=["game_id", "home_team"],
                how="inner",
            )
            home_qb = home_qb[["game_id", "qb_adjustment"]].rename(
                columns={"qb_adjustment": "home_qb_adjustment"}
            )
            combined_features = combined_features.merge(
                home_qb, on="game_id", how="left"
            )

            # Merge away QB adjustment
            away_qb = qb_df[["game_id", "team", "qb_adjustment"]].copy()
            away_qb = away_qb.merge(
                combined_features[["game_id", "away_team"]].drop_duplicates(),
                left_on=["game_id", "team"],
                right_on=["game_id", "away_team"],
                how="inner",
            )
            away_qb = away_qb[["game_id", "qb_adjustment"]].rename(
                columns={"qb_adjustment": "away_qb_adjustment"}
            )
            combined_features = combined_features.merge(
                away_qb, on="game_id", how="left"
            )

            feature_counts["qb_tracking"] = 2  # home + away qb_adjustment

        # Snap-count features (game-level; SnapCountBuilder already emits the
        # home_/away_-expanded columns, so merge on game_id like the elo block).
        # combine_features has NO generic loop over feature_sources keys -- without
        # this explicit block the registered snap columns pass the LeakageGate but
        # are SILENTLY DROPPED from gold (review #1, orchestrator-verified).
        snaps_df = feature_sources.get("snaps", pd.DataFrame())
        if len(snaps_df) > 0:
            snap_cols = [c for c in snaps_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                snaps_df[["game_id", *snap_cols]], on="game_id", how="left"
            )
            feature_counts["snaps"] = len(snap_cols)

        # Injury features (game-level; InjuryBuilder already emits the home_/away_-
        # expanded columns). Same rationale as the snap block (review #1): merge on
        # game_id so the columns actually reach all three gold matrices.
        injury_df = feature_sources.get("injury", pd.DataFrame())
        if len(injury_df) > 0:
            injury_cols = [c for c in injury_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                injury_df[["game_id", *injury_cols]], on="game_id", how="left"
            )
            feature_counts["injury"] = len(injury_cols)

        # Line-movement features (game-level; LineMovementBuilder emits one
        # un-prefixed row per game -- a line trajectory belongs to the game, not
        # to a side). Same rationale as the snap/injury blocks: combine_features
        # has NO generic loop over feature_sources, so without this EXPLICIT merge
        # the registered line-movement columns pass the LeakageGate and are then
        # SILENTLY DROPPED from gold (the load-bearing 28-06 lesson).
        line_movement_df = feature_sources.get("line_movement", pd.DataFrame())
        if len(line_movement_df) > 0:
            lm_cols = [c for c in line_movement_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                line_movement_df[["game_id", *lm_cols]], on="game_id", how="left"
            )
            feature_counts["line_movement"] = len(lm_cols)

        # Add feature timestamp (tz-aware UTC; the storage layer rejects naive
        # datetimes, and feature_timestamp is persisted into every gold matrix)
        combined_features["feature_timestamp"] = datetime.now(UTC)

        logger.info(
            "Combined all features",
            total_games=len(combined_features),
            total_features=len(combined_features.columns),
            feature_breakdown=feature_counts,
        )

        return combined_features

    def _get_team_features(
        self,
        team_form_df: pd.DataFrame,
        games_df: pd.DataFrame,
        team_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Get team form features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's offensive features
            team_off = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "offense")
            ]

            # Get team's defensive features
            team_def = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "defense")
            ]

            game_features = {"game_id": game_id}

            # Add offensive features
            if len(team_off) > 0:
                off_row = team_off.iloc[0]
                for col in off_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_off_{col}"
                        game_features[feature_name] = off_row[col]

            # Add defensive features
            if len(team_def) > 0:
                def_row = team_def.iloc[0]
                for col in def_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_def_{col}"
                        game_features[feature_name] = def_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def _get_team_elo_features(
        self, elo_df: pd.DataFrame, games_df: pd.DataFrame, team_col: str, prefix: str
    ) -> pd.DataFrame:
        """Get Elo features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's Elo rating
            team_elo = elo_df[
                (elo_df["team"] == team)
                & (elo_df["season"] == season)
                & (elo_df["week"] == week)
            ]

            game_features = {"game_id": game_id}

            if len(team_elo) > 0:
                elo_row = team_elo.iloc[0]
                for col in elo_row.index:
                    if col not in ["team", "season", "week"]:
                        feature_name = f"{prefix}_{col}"
                        game_features[feature_name] = elo_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def handle_missing_data_and_outliers(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Handle missing data and outliers with winsorization.

        WR-06: every distributional statistic below is fitted on the seasons
        STRICTLY BEFORE the season it is applied to. Before this, the imputation
        median and the q01/q99 bounds were computed over the whole multi-season
        frame, so season 2025 could move a 2021 feature value -- the temporal
        boundary this phase turns on, and the reason SPEC R2's byte-identity
        control exists.

        DELIBERATE DIVERGENCE FROM THE HOUSE PRECEDENT. ``features.normalization.
        compute_prior_season_stats`` fits on season Y-1 ONLY; this fits on ALL
        seasons before Y. Both are point-in-time; they differ in sample size, and
        D30-16's own rationale asks for a full-season-or-more sample so
        winsorization stays stable -- a q01/q99 estimate from roughly 285 games is
        materially noisier than one from thousands. The two also run at different
        stages on different statistics (mean and standard deviation for a z-score
        bootstrap, versus median and quantiles for imputation and clipping), so
        they are NOT required to agree. Do not "fix" one to match the other.

        ACCEPTED RESIDUAL, documented rather than left unstated: within-season
        lookahead remains. A self-fitting season's week-1 bound still sees that
        season's week 18, and ``_impute_team_features``' team mean and season mean
        are still within-season. D30-16 accepts both, and neither can be moved by
        adding a LATER season's rows -- which is exactly what SPEC R2 asserts.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from processing

        Returns:
            Processed feature matrix
        """
        logger.info(
            "Handling missing data and outliers", features=len(features_df.columns)
        )

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        processed_df = features_df.copy()

        # Get numeric feature columns
        feature_cols = [
            col for col in processed_df.columns if col not in target_columns
        ]
        numeric_cols = (
            processed_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        missing_stats = {}
        outlier_stats = {}

        # WR-06: the per-season passes, computed ONCE. ``season`` sits in the
        # target-column exclusion list above, which removes it from ``numeric_cols``
        # but leaves the column in the frame -- so a per-season pass can group on it
        # directly. Each entry is (season, season_mask, strictly_prior_mask).
        self.self_fit_seasons = {}
        season_passes = self._season_passes(processed_df)
        earliest_season = season_passes[0][0] if season_passes else None

        # WR-10: the line-movement family must NEVER be median-imputed. Its neutral
        # state is a defined thing -- LEAGUE_AVERAGE_TOTAL for the opening anchors,
        # 0.0 for the drift/path families and 0.0 for the coverage flag -- and the
        # column median for line_movement_coverage is 1.0, so a median fill would
        # fabricate coverage for games that have none. Fill from the builder's own
        # neutral defaults instead, so a gap can only ever read as "not covered".
        neutral_line_movement = self.line_movement_builder._neutral_features(
            emit_spread=True
        )

        for col in numeric_cols:
            original_missing = processed_df[col].isna().sum()

            # Handle missing data
            if original_missing > 0:
                if col in neutral_line_movement:
                    processed_df[col] = processed_df[col].fillna(
                        neutral_line_movement[col]
                    )
                # For team-based features, use team's season average
                elif any(prefix in col for prefix in ["home_", "away_"]):
                    processed_df[col] = self._impute_team_features(processed_df, col)
                else:
                    # WR-06 surface 1: for game-level features this was
                    # ``processed_df[col].median()`` over the WHOLE frame, so a 2002
                    # gap was filled from a statistic that saw 2025. It is now a
                    # per-season, strictly-prior median.
                    processed_df[col] = self._impute_game_level_features(
                        processed_df, col, season_passes, earliest_season
                    )

                missing_stats[col] = original_missing

            # CR-02: a DISCRETE INDICATOR has no outliers to winsorize, and
            # clipping one destroys the distinction it exists to encode.
            #
            # The WR-10 guard above correctly refuses to median-impute the
            # line-movement family, because the median of line_movement_coverage is
            # 1.0. But that guard used to sit inside `if original_missing > 0:` and
            # end in `continue`. In the normal case the builder emits a row for
            # every game, so original_missing == 0, the guard never ran, and
            # execution fell straight into the unconditional winsorization below.
            #
            # On a single covered season the uncovered fraction is far below 1%
            # (2023 is 271/272 covered), so q01 == q99 == 1.0 and the clip stamped
            # EVERY uncovered game as COVERED: `--season 2023` produced gold whose
            # line_movement_coverage was a constant 1.0. That is precisely the
            # fabrication WR-10 was written to prevent, arriving through a different
            # door, and it additionally destroyed the column's variance before
            # expanding_normalize saw it. The full-history rebuild happened to be
            # safe (q01 = 0.0 at ~13% uncovered), which is why the published gold is
            # unaffected and why nothing caught it. It is a general defect for any
            # rare binary flag -- `saturday_game` is another candidate.
            #
            # The old shape was also internally inconsistent: the `continue` skipped
            # winsorization entirely whenever the family DID have NaNs, so the same
            # column was winsorized or not depending on whether a gap happened to
            # exist. Missing-handling and outlier-handling are now independent.
            #
            # The test is deliberately a VALUE test, not a name test: any column
            # whose values are all indicator levels is discrete, however it is
            # spelled. Continuous line-movement columns (opening_total, the drift
            # and path families) are NOT exempted -- they are genuine continuous
            # measurements with genuine outliers, and the published readout's
            # argument about what the model saw rests on their winsorization bound.
            #
            # WR-06 note on WHERE this test sits. It is evaluated ONCE per column,
            # over the whole frame, OUTSIDE the per-season loop below. That is a
            # correctness requirement, not tidiness: a column that is {0, 1} overall
            # is discrete and a per-season evaluation would still classify it
            # correctly, but a column that is continuous overall yet happens to be
            # constant at one of -1 / 0 / 1 within a single season would be
            # misclassified as an indicator for that season and would silently
            # escape winsorization there (T-30-29). It is evaluated AFTER the
            # missing-handling above, exactly as it was before this rewrite, so the
            # predicate still sees the already-imputed column its docstring names.
            if self._is_discrete_indicator(processed_df[col]):
                continue

            # Handle outliers with winsorization, per season, on strictly-prior
            # bounds (WR-06 surface 2). The pre-WR-06 shape computed one q01/q99 pair
            # from the whole frame and clipped every row against it.
            outliers_count = 0
            for season, season_mask, prior_mask in season_passes:
                fit_source, self_fit = self._season_fit_source(
                    self._column(processed_df, col),
                    season,
                    season_mask,
                    prior_mask,
                    earliest_season,
                )

                # The pre-WR-06 minimum-data-points condition, now applied to the fit
                # source rather than to the whole column.
                if fit_source.notna().sum() <= self._MIN_FIT_POINTS:
                    continue

                lower_bound = fit_source.quantile(self.outlier_percentiles[0] / 100)
                upper_bound = fit_source.quantile(self.outlier_percentiles[1] / 100)
                if pd.isna(lower_bound) or pd.isna(upper_bound):
                    continue

                if self_fit:
                    self._record_self_fit(col, season)

                season_values = processed_df.loc[season_mask, col]
                season_outliers = int(
                    (
                        (season_values < lower_bound) | (season_values > upper_bound)
                    ).sum()
                )
                if season_outliers > 0:
                    processed_df.loc[season_mask, col] = season_values.clip(
                        lower=lower_bound, upper=upper_bound
                    )
                    outliers_count += season_outliers

            if outliers_count > 0:
                outlier_stats[col] = outliers_count

        logger.info(
            "Completed missing data and outlier handling",
            missing_imputed=len(missing_stats),
            outliers_winsorized=len(outlier_stats),
            discrete_indicators_exempt=sum(
                1 for c in numeric_cols if self._is_discrete_indicator(processed_df[c])
            ),
            total_features_processed=len(numeric_cols),
            # WR-06: the self-fit flag, made observable in a build log. A season
            # other than the earliest appearing here is a per-column coverage floor
            # -- the column's upstream source simply starts later -- and is a finding
            # worth reading, not an error.
            self_fit_columns=len(self.self_fit_seasons),
            self_fit_seasons=sorted(
                {
                    season
                    for seasons in self.self_fit_seasons.values()
                    for season in seasons
                }
            ),
        )

        return processed_df

    # ------------------------------------------------------------------
    # WR-06 helpers: prior-seasons-only fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _column(df: pd.DataFrame, col: str) -> pd.Series:
        """Return ``df[col]`` narrowed to a Series.

        pandas types ``df[col]`` as ``Series | DataFrame`` because a DUPLICATED
        column label yields a frame. The gold matrices carry no duplicate labels, so
        the frame arm is unreachable here; stating that once beats repeating a cast
        at every call site, and it keeps the WR-06 helpers honestly typed.
        """
        values = df[col]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return values

    @staticmethod
    def _season_passes(df: pd.DataFrame) -> list[tuple]:
        """Return ``[(season, season_mask, strictly_prior_mask), ...]`` ascending.

        Computed once per call and reused across every column, because the masks are
        the expensive part of a per-season pass over two hundred columns.

        A frame with no ``season`` column is never the production path -- the column
        is carried through ``combine_features`` and is required by
        ``_impute_team_features`` -- but some unit frames omit it. Such a frame
        degrades to ONE self-fitting pass over the whole frame, i.e. to the
        pre-WR-06 whole-frame behaviour, rather than to no processing at all.
        """
        if "season" not in df.columns:
            everything = pd.Series(True, index=df.index)
            return [(None, everything, pd.Series(False, index=df.index))]

        seasons = sorted(df["season"].dropna().unique())
        return [
            (season, df["season"] == season, df["season"] < season)
            for season in seasons
        ]

    def _record_self_fit(self, col: str, season) -> None:
        """Record that *col*'s statistic for *season* was fitted on its own rows."""
        if season is None:
            return
        seasons = self.self_fit_seasons.setdefault(col, [])
        if int(season) not in seasons:
            seasons.append(int(season))

    def _season_fit_source(
        self,
        values: pd.Series,
        season,
        season_mask: pd.Series,
        prior_mask: pd.Series,
        earliest_season,
    ) -> tuple[pd.Series, bool]:
        """Return ``(fit_source, self_fit)`` for *season*.

        The fit source is the strictly-prior slice. The earliest data-bearing season
        has none, and a column whose upstream source starts mid-history has an EMPTY
        prior slice in its first populated season -- a naive prior-only rewrite would
        hand both of them NaN bounds or raise (T-30-55). Both cases fall back to the
        season's own rows under the SAME documented flag, so a per-column coverage
        floor is handled by the general rule rather than by a per-family exception.
        """
        if season != earliest_season:
            prior = values.loc[prior_mask]
            if prior.notna().sum() > self._MIN_FIT_POINTS:
                return prior, False
        return values.loc[season_mask], True

    def _impute_game_level_features(
        self,
        df: pd.DataFrame,
        col: str,
        season_passes: list[tuple],
        earliest_season,
    ) -> pd.Series:
        """Fill a game-level column's gaps with a prior-seasons-only median (WR-06).

        Replaces ``df[col].fillna(df[col].median())``, whose median saw every future
        season. A season with no usable fit source at all keeps its NaNs rather than
        borrowing a value from the future: that is the deliberate consequence of the
        fix, not an oversight, and downstream ``expanding_normalize`` already maps an
        un-normalizable position to the neutral 0.0 z-score.
        """
        result = self._column(df, col).copy()

        for season, season_mask, prior_mask in season_passes:
            season_values = result.loc[season_mask]
            if not season_values.isna().any():
                continue

            fit_source, self_fit = self._season_fit_source(
                result, season, season_mask, prior_mask, earliest_season
            )
            median_value = float(fit_source.median())
            if np.isnan(median_value):
                continue

            if self_fit:
                self._record_self_fit(col, season)
            result.loc[season_mask] = season_values.fillna(median_value)

        return result

    @staticmethod
    def _is_discrete_indicator(series: pd.Series) -> bool:
        """True when a column's values are indicator levels, not measurements.

        CR-02. Winsorization answers "is this value an implausible extreme of a
        continuous distribution". That question is meaningless for a column whose
        values are drawn from {-1, 0, 1} -- a coverage flag, a sign, a boolean
        game-context marker. For such a column the 1st and 99th percentiles are
        simply the modal level whenever the minority level is rarer than 1%, so
        the clip does not remove an outlier: it OVERWRITES every minority row with
        the majority level and leaves a constant.

        That is how ``--season 2023`` produced gold whose
        ``line_movement_coverage`` was a constant 1.0, encoding "we measured this
        game's trajectory" for games that have none -- the exact conflation the
        flag exists to prevent.

        Args:
            series: The (already imputed) numeric feature column.

        Returns:
            True iff every non-null value is one of -1.0, 0.0 or 1.0.
        """
        values = pd.unique(series.dropna())
        return len(values) > 0 and set(values.tolist()) <= {-1.0, 0.0, 1.0}

    def _impute_team_features(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Impute missing team features using team's season average.

        WR-14: the two ``col.replace("home_", "")`` / ``col.replace("away_", "")``
        statements that used to sit here were no-ops -- ``str`` is immutable and
        the results were discarded -- so they looked like they computed a base
        column name and did not. Nothing downstream ever needed one: the imputation
        works on ``col`` itself and only needs to know WHICH team column to group
        by. The dead lines are gone rather than "fixed", because there was no bug
        to fix, only a false suggestion that a base name was in play.

        WR-06 SURFACE 3, named by neither the SPEC nor CONTEXT (T-30-28). Both
        last-resort fallbacks below used to be ``df[col].median()`` over the WHOLE
        frame, and this is the branch every ``home_*`` / ``away_*`` column takes --
        the large majority of features. Any 2021-2024 row reaching either of them
        WOULD move when the N-01 re-sync adds 2025 rows, failing SPEC R2's
        byte-identity control for a cause unrelated to the two surfaces D30-16
        names. Both now fit on the strictly-prior seasons.

        The two WITHIN-SEASON statistics -- the team mean and the season mean -- are
        DELIBERATELY left exactly as they are. Neither can be moved by adding a
        later season's rows, so neither threatens SPEC R2, and converting them would
        be a larger behavioural change than D30-16 authorises in the
        highest-blast-radius file in this phase.
        """
        season_passes = self._season_passes(df)
        earliest_season = season_passes[0][0] if season_passes else None

        if col.startswith("home_"):
            team_col = "home_team"
        elif col.startswith("away_"):
            team_col = "away_team"
        else:
            # Not a team feature. The dispatch in handle_missing_data_and_outliers
            # routes on ``"home_" in col`` while this method routes on
            # ``col.startswith("home_")``, so a column carrying the substring
            # anywhere but the front lands here. WR-06 surface 3, first site.
            return self._impute_game_level_features(
                df, col, season_passes, earliest_season
            )

        result = self._column(df, col).copy()

        # For each team with missing data, use their season average
        for season, season_mask, prior_mask in season_passes:
            season_data = df.loc[season_mask]
            # Computed LAZILY, and only when the two within-season statistics have
            # both come back NaN: computing it eagerly would record a self-fit for
            # every season of a late-arriving column, whose prior slices are empty
            # but whose fallback is never actually reached.
            prior_median = None
            prior_median_computed = False

            for team in season_data[team_col].unique():
                team_mask = season_mask & (df[team_col] == team)
                team_values = df.loc[team_mask, col]

                if team_values.isna().any():
                    team_mean = team_values.mean()
                    if pd.isna(team_mean):
                        # Use season average if team has no data
                        team_mean = season_data[col].mean()
                    if pd.isna(team_mean):
                        # WR-06 surface 3, second site: the last resort is the
                        # median of the seasons STRICTLY BEFORE this one, not of
                        # the whole frame.
                        if not prior_median_computed:
                            prior_median = self._prior_season_median(
                                result,
                                col,
                                season,
                                season_mask,
                                prior_mask,
                                earliest_season,
                            )
                            prior_median_computed = True
                        if prior_median is None:
                            # Nothing at or before this season can fill the
                            # gap. Leave it NaN rather than borrow from the
                            # future -- the deliberate consequence of WR-06.
                            continue
                        team_mean = prior_median

                    result.loc[team_mask & result.isna()] = team_mean

        return result

    def _prior_season_median(
        self,
        values: pd.Series,
        col: str,
        season,
        season_mask: pd.Series,
        prior_mask: pd.Series,
        earliest_season,
    ) -> float | None:
        """Return the strictly-prior-seasons median for *col* in *season* (WR-06)."""
        fit_source, self_fit = self._season_fit_source(
            values, season, season_mask, prior_mask, earliest_season
        )
        median_value = float(fit_source.median())
        if np.isnan(median_value):
            return None
        if self_fit:
            self._record_self_fit(col, season)
        return float(median_value)

    def normalize_features_within_seasons(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Apply Z-score normalization within seasons.

        .. deprecated::
            This method uses full-season mean/std which leaks future data.
            Use ``expanding_normalize`` from ``features.normalization`` instead.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from normalization

        Returns:
            Normalized feature matrix
        """
        warnings.warn(
            "normalize_features_within_seasons is deprecated due to future data leakage. "
            "Use expanding_normalize from features.normalization instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info("Normalizing features within seasons")

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        normalized_df = features_df.copy()

        # Get numeric feature columns to normalize
        feature_cols = [
            col for col in normalized_df.columns if col not in target_columns
        ]
        numeric_cols = (
            normalized_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        normalization_stats = {}

        # Normalize within each season
        for season in normalized_df["season"].unique():
            season_mask = normalized_df["season"] == season
            season_data = normalized_df.loc[season_mask]

            if len(season_data) < self.min_games_for_stats:
                logger.warning(
                    "Insufficient data for normalization",
                    season=season,
                    games=len(season_data),
                )
                continue

            season_normalized_cols = 0

            for col in numeric_cols:
                col_data = season_data[col]

                if col_data.notna().sum() >= self.min_games_for_stats:
                    mean_val = col_data.mean()
                    std_val = col_data.std()

                    if std_val > 0:  # Avoid division by zero
                        normalized_df.loc[season_mask, col] = (
                            col_data - mean_val
                        ) / std_val
                        season_normalized_cols += 1

            normalization_stats[season] = season_normalized_cols

        logger.info(
            "Completed within-season normalization",
            seasons_processed=len(normalization_stats),
            avg_features_per_season=np.mean(list(normalization_stats.values())),
        )

        return normalized_df

    def create_target_variables(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for WP, ATS, and O/U prediction.

        Args:
            features_df: Feature matrix with game scores

        Returns:
            Feature matrix with target variables added
        """
        logger.info("Creating target variables")

        target_df = features_df.copy()

        # Only create targets if we have scores
        if (
            "home_score" not in target_df.columns
            or "away_score" not in target_df.columns
        ):
            logger.warning("No score data available for target creation")
            return target_df

        # Convert score columns to numeric (they may be stored as strings)
        target_df["home_score"] = pd.to_numeric(
            target_df["home_score"], errors="coerce"
        )
        target_df["away_score"] = pd.to_numeric(
            target_df["away_score"], errors="coerce"
        )

        # Remove games with missing scores
        score_mask = target_df["home_score"].notna() & target_df["away_score"].notna()
        target_df = target_df[score_mask].copy()

        # Win Probability target (1 = home win, 0 = away win)
        target_df["target_wp"] = (
            target_df["home_score"] > target_df["away_score"]
        ).astype(int)
        # Trainer-expected column: strict binary (ties = 0, same as trainer derivation)
        target_df["home_win"] = target_df["target_wp"].copy()

        # Handle ties (rare in NFL)
        ties = target_df["home_score"] == target_df["away_score"]
        if ties.sum() > 0:
            logger.info("Found tied games", count=ties.sum())
            # For WP regression target, ties as 0.5
            target_df.loc[ties, "target_wp"] = 0.5
            # For classifier target, ties stay 0 (not a home win)

        # Point differential (for ATS calculation if spreads available)
        target_df["point_differential"] = (
            target_df["home_score"] - target_df["away_score"]
        )
        # Trainer-expected column (alias for ATS regression target)
        target_df["home_margin"] = target_df["point_differential"].copy()

        # Total points (for O/U calculation if totals available)
        target_df["total_points"] = target_df["home_score"] + target_df["away_score"]

        # ATS target (requires market data)
        if "snapshot_spread" in target_df.columns:
            # ATS = actual margin - spread (positive = home team covered)
            target_df["target_ats"] = (
                target_df["point_differential"] - target_df["snapshot_spread"]
            )
            target_df["home_covered_spread"] = (target_df["target_ats"] > 0).astype(int)

        # O/U target (requires market data)
        if "snapshot_total" in target_df.columns:
            # O/U = actual total - market total (positive = over)
            target_df["target_ou"] = (
                target_df["total_points"] - target_df["snapshot_total"]
            )
            target_df["game_went_over"] = (target_df["target_ou"] > 0).astype(int)

        logger.info(
            "Created target variables",
            wp_targets=target_df["target_wp"].notna().sum(),
            ats_targets=target_df.get("target_ats", pd.Series()).notna().sum(),
            ou_targets=target_df.get("target_ou", pd.Series()).notna().sum(),
        )

        return target_df

    def generate_feature_matrices(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate complete feature matrices for all prediction targets.

        Pipeline flow:
        1. Load feature sources
        2. Stage 1: Per-source LeakageGate.check_time_fence
        3. Combine features
        4. Stage 2: LeakageGate.validate_combined_matrix
        5. Handle missing data and outliers
        6. Expanding-window normalization (replaces within-season Z-scores)
        7. Create target variables
        8. Split into per-target matrices

        Args:
            target_season: Specific season to process.
            target_week: Specific week to process.
            as_of_datetime: Time-fence cutoff for leakage validation.
                Defaults to ``datetime.now(ET)`` if not provided -- tz-AWARE, see
                the CR-01 note on ``load_all_feature_sources``.

        Returns:
            Dictionary with feature matrices for each target.
        """
        # CR-01: tz-aware ET, never a naive local clock. ``pipeline/steps.py:235``
        # calls this with no arguments, so this default IS the live orchestrator
        # fence.
        if as_of_datetime is None:
            as_of_datetime = datetime.now(ET)

        logger.info(
            "Generating feature matrices",
            target_season=target_season,
            target_week=target_week,
            as_of_datetime=str(as_of_datetime),
        )

        try:
            # Load all feature sources
            feature_sources = self.load_all_feature_sources(
                target_season, target_week, as_of_datetime=as_of_datetime
            )

            # -- Stage 1: Per-source time-fence check --
            for source_name, source_df in feature_sources.items():
                if source_name == "games":
                    continue  # Games are the base, not a builder output
                if len(source_df) == 0:
                    continue

                try:
                    self.leakage_gate.check_time_fence(
                        source_df, as_of_datetime, source_name
                    )
                except LeakageViolation as e:
                    self.leakage_gate.write_diagnostic_report(e)
                    raise

            # Combine features
            combined_features = self.combine_features(feature_sources)

            if len(combined_features) == 0:
                logger.error("No features to process")
                return {}

            # -- Replace raw EPA with opponent-adjusted EPA --
            # OpponentAdjuster needs per-game stats (with game_id, raw EPA),
            # not the rolling averages from the silver table.
            games_df = feature_sources["games"]
            try:
                per_game_stats = self.team_form_calc.get_per_game_stats(
                    as_of_datetime,
                    target_season=target_season,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                logger.warning(
                    "Failed to get per-game stats for opponent adjustment",
                    error=str(e),
                )
                per_game_stats = pd.DataFrame()
            if len(per_game_stats) > 0:
                try:
                    adjusted_df = self.opponent_adj.build_features(
                        games_df,
                        as_of_datetime,
                        target_season=target_season,
                        target_week=target_week,
                        team_game_stats=per_game_stats,
                    )

                    if len(adjusted_df) > 0:
                        # Merge opponent-adjusted features using same _get_team_features pattern
                        for prefix, team_col in [
                            ("home", "home_team"),
                            ("away", "away_team"),
                        ]:
                            adj_team = self._get_team_features(
                                adjusted_df, combined_features, team_col, prefix
                            )
                            # Only keep the rolling_opp_adj_* columns (not duplicate other rolling cols)
                            opp_adj_cols = [
                                c for c in adj_team.columns if "opp_adj" in c
                            ]
                            if opp_adj_cols:
                                adj_team = adj_team[["game_id", *opp_adj_cols]]
                                combined_features = combined_features.merge(
                                    adj_team, on="game_id", how="left"
                                )

                        # Drop old raw EPA columns that are now replaced by opp_adj versions
                        raw_epa_suffixes = [
                            "rolling_epa_per_play",
                            "rolling_pass_epa_per_play",
                            "rolling_rush_epa_per_play",
                        ]
                        cols_to_drop = []
                        for pfx in ["home", "away"]:
                            for side in ["off", "def"]:
                                for suffix in raw_epa_suffixes:
                                    col_name = f"{pfx}_{side}_{suffix}"
                                    if col_name in combined_features.columns:
                                        cols_to_drop.append(col_name)

                        if cols_to_drop:
                            combined_features = combined_features.drop(
                                columns=cols_to_drop
                            )
                            logger.info(
                                "Replaced raw EPA with opponent-adjusted EPA",
                                dropped_columns=cols_to_drop,
                                n_dropped=len(cols_to_drop),
                            )
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(
                        "Failed to apply opponent adjustment, keeping raw EPA",
                        error=str(e),
                    )

            # -- Stage 2: Combined matrix validation --
            try:
                self.leakage_gate.validate_combined_matrix(
                    combined_features, as_of_datetime
                )
            except LeakageViolation as e:
                self.leakage_gate.write_diagnostic_report(e)
                raise

            # Handle missing data and outliers
            processed_features = self.handle_missing_data_and_outliers(
                combined_features
            )

            # Preserve the un-normalized composite severity for display before
            # normalization runs (mirrors the raw_wind_mph / raw_precip_mm
            # family). The normalized weather_severity_score stays the model
            # feature; this raw sibling is excluded below so it is never
            # z-scored and the cache can surface a human-meaningful value.
            processed_features["raw_weather_severity"] = processed_features[
                "weather_severity_score"
            ]

            # -- Expanding-window normalization (replaces within-season Z-scores) --
            exclude_cols = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
                "raw_wind_mph",
                "raw_temp_f",
                "raw_precip_prob",
                "raw_precip_mm",
                "raw_humidity_pct",
                "raw_weather_severity",
            ]
            feature_cols = [
                col for col in processed_features.columns if col not in exclude_cols
            ]
            numeric_feature_cols = (
                processed_features[feature_cols]
                .select_dtypes(include=[np.number])
                .columns.tolist()
            )

            # Compute prior-season stats for bootstrap and normalize
            if target_season:
                # Single-season mode: compute prior stats once
                prior_stats = compute_prior_season_stats(
                    processed_features, numeric_feature_cols, target_season - 1
                )
                normalized_features = expanding_normalize(
                    processed_features,
                    feature_cols=numeric_feature_cols,
                    group_col="season",
                    sort_cols=["season", "week"],
                    min_periods=4,
                    prior_season_stats=prior_stats,
                )
            else:
                # Batch mode: compute prior-season stats per season
                seasons = sorted(processed_features["season"].unique())
                normalized_parts = []
                for s in seasons:
                    season_df = processed_features[
                        processed_features["season"] == s
                    ].copy()
                    prior_stats = compute_prior_season_stats(
                        processed_features, numeric_feature_cols, s - 1
                    )
                    norm_part = expanding_normalize(
                        season_df,
                        feature_cols=numeric_feature_cols,
                        group_col="season",
                        sort_cols=["season", "week"],
                        min_periods=4,
                        prior_season_stats=prior_stats,
                    )
                    normalized_parts.append(norm_part)
                normalized_features = pd.concat(normalized_parts, ignore_index=False)

            # Create target variables
            final_features = self.create_target_variables(normalized_features)

            # Generate separate matrices for each prediction target
            feature_matrices = {}

            # Base feature columns (exclude identifiers and targets)
            exclude_cols = [
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

            feature_cols = [
                col for col in final_features.columns if col not in exclude_cols
            ]

            # Metadata columns to carry through for target derivation
            meta_cols = ["game_id", "season", "week", "feature_timestamp"]
            score_cols = [
                c for c in ["home_score", "away_score"] if c in final_features.columns
            ]

            # Win Probability matrix
            wp_target_cols = [
                c for c in ["target_wp", "home_win"] if c in final_features.columns
            ]
            wp_matrix = final_features[
                [*meta_cols, *score_cols, *wp_target_cols, *feature_cols]
            ].copy()
            feature_matrices["wp"] = wp_matrix

            # ATS matrix (only include games with spread data)
            if "target_ats" in final_features.columns:
                ats_target_cols = [
                    c
                    for c in ["target_ats", "home_margin", "point_differential"]
                    if c in final_features.columns
                ]
                ats_games = final_features["target_ats"].notna()
                ats_matrix = final_features.loc[
                    ats_games,
                    [*meta_cols, *score_cols, *ats_target_cols, *feature_cols],
                ].copy()
                feature_matrices["ats"] = ats_matrix

            # O/U matrix (only include games with total data)
            if "target_ou" in final_features.columns:
                ou_target_cols = [
                    c
                    for c in ["target_ou", "total_points"]
                    if c in final_features.columns
                ]
                ou_games = final_features["target_ou"].notna()
                ou_matrix = final_features.loc[
                    ou_games,
                    [*meta_cols, *score_cols, *ou_target_cols, *feature_cols],
                ].copy()
                feature_matrices["ou"] = ou_matrix

            logger.info(
                "Generated feature matrices",
                wp_games=len(feature_matrices.get("wp", [])),
                ats_games=len(feature_matrices.get("ats", [])),
                ou_games=len(feature_matrices.get("ou", [])),
                total_features=len(feature_cols),
            )

            return feature_matrices

        except _SOURCE_LOAD_ERRORS as e:
            logger.error("Failed to generate feature matrices", error=str(e))
            raise

    def save_feature_matrices(
        self,
        feature_matrices: dict[str, pd.DataFrame],
        target_season: int | None = None,
    ) -> None:
        """
        Save feature matrices to gold layer.

        Each matrix is written as a single self-contained Parquet file with
        game_id latest-wins dedup (no directory partitioning), so both the
        full-rebuild (target_season=None) and current-week/per-season paths
        are idempotent. target_season no longer controls partitioning; it is
        retained for call-site compatibility and logged for observability.

        Args:
            feature_matrices: Dictionary with feature matrices
            target_season: Season the matrices were built for (informational
                only; does not affect the single-file write).
        """
        logger.info(
            "Saving feature matrices to gold layer", target_season=target_season
        )

        for target, matrix_df in feature_matrices.items():
            if len(matrix_df) == 0:
                logger.warning("Empty feature matrix", target=target)
                continue

            # Coerce non-numeric feature columns (string artifacts from append)
            non_meta = [
                c
                for c in matrix_df.columns
                if c
                not in [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "feature_timestamp",
                    "weather_condition",
                ]
            ]
            for col in non_meta:
                if matrix_df[col].dtype == object:
                    matrix_df[col] = pd.to_numeric(matrix_df[col], errors="coerce")

            # Drop non-numeric string columns that aren't features
            if "weather_condition" in matrix_df.columns:
                matrix_df = matrix_df.drop(columns=["weather_condition"])

            # Drop columns that are 100% NaN (offense-only metrics on defense side)
            all_nan_cols = [
                c
                for c in matrix_df.columns
                if matrix_df[c].isna().all() and c not in ["game_id"]
            ]
            if all_nan_cols:
                matrix_df = matrix_df.drop(columns=all_nan_cols)
                logger.info(
                    "Dropped all-NaN columns",
                    target=target,
                    dropped=all_nan_cols,
                )

            table_name = f"features_{target}"

            # Write the gold matrix as a single self-contained file (no
            # directory partitioning). The gold matrices are one-row-per-game
            # tables; carrying game_id, they dedup cleanly with latest-wins.
            #
            # The prior partition_cols=["season"] if target_season else None
            # re-introduced the shared-root partitioned-append antipattern that
            # 25c364f eradicated from every silver builder: pq.write_to_dataset
            # writes season=YYYY/ partition directories into the SHARED
            # data/gold/ root, where all three matrices (features_wp/ats/ou)
            # collide in the same season=YYYY/ directory and each current-week
            # run appends a NEW hash-named parquet instead of overwriting --
            # silently multiplying gold cardinality and cross-contaminating the
            # three matrices.
            #
            # save_dataframe's default append_mode=True path reads any existing
            # single-file gold table, concats the rebuilt rows, drops duplicate
            # game_ids keeping the latest, and writes one file. So the
            # current-week/per-season path is now idempotent (re-running cannot
            # append-bloat or cross-contaminate), and the full-rebuild path
            # (target_season=None) still writes the complete single file as
            # before. This mirrors scripts/build_weather.py /
            # scripts/build_contextual.py and pipeline/steps.py. (CR-01, D-10)
            save_dataframe(
                matrix_df,
                table_name=table_name,
                layer="gold",
            )

            logger.info(
                "Saved feature matrix",
                target=target,
                table_name=table_name,
                records=len(matrix_df),
                features=len(matrix_df.columns) - 5,
            )  # Exclude metadata columns


def main():
    """Build unified feature matrices."""
    parser = argparse.ArgumentParser(description="Build unified feature matrices")
    parser.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--as-of",
        type=str,
        help="As-of datetime (ISO format) for time-fence enforcement",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save feature matrices to gold layer (default: on)",
    )
    # The bare --save flag was a no-op (store_true with default=True can never
    # turn saving OFF). --no-save is the real toggle for a read-only build.
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Build the feature matrices without writing them to the gold layer",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate feature matrices after building",
    )

    args = parser.parse_args()

    # Parse --as-of datetime if provided.
    #
    # CR-01: a bare ISO string on the command line means ET -- the freeze this
    # project fences on is "Friday 6 PM ET", and an operator typing
    # `--as-of 2024-10-04T18:00:00` means 18:00 ET, not 18:00 UTC. Attaching ET
    # here makes the value tz-AWARE, so `ensure_utc_aware` downstream CONVERTS it
    # (22:00Z) instead of re-labelling it (18:00Z == 14:00 ET, four hours early).
    # An explicit offset in the string is honoured as given.
    as_of_dt = None
    if args.as_of:
        as_of_dt = datetime.fromisoformat(args.as_of)
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=ET)

    logger.info("Building unified feature matrices", season=args.season, week=args.week)

    try:
        # Initialize feature matrix builder
        builder = FeatureMatrixBuilder()

        # Generate feature matrices
        feature_matrices = builder.generate_feature_matrices(
            target_season=args.season,
            target_week=args.week,
            as_of_datetime=as_of_dt,
        )

        if not feature_matrices:
            logger.warning("No feature matrices generated")
            return

        # Display summary
        print("\nFeature matrices summary:")
        print("=" * 60)

        total_features = 0
        for target, matrix_df in feature_matrices.items():
            feature_count = len(matrix_df.columns) - 5  # Exclude metadata
            total_features = feature_count  # They should all have same feature count

            print(f"{target.upper()} Matrix:")
            print(f"  Games: {len(matrix_df)}")
            print(f"  Features: {feature_count}")

            if len(matrix_df) > 0:
                # Show target distribution
                target_col = f"target_{target}"
                if target_col in matrix_df.columns:
                    if target == "wp":
                        home_wins = (matrix_df[target_col] == 1).sum()
                        away_wins = (matrix_df[target_col] == 0).sum()
                        ties = (matrix_df[target_col] == 0.5).sum()
                        print(
                            f"  Home wins: {home_wins}, Away wins: {away_wins}, Ties: {ties}"
                        )
                    else:
                        target_mean = matrix_df[target_col].mean()
                        target_std = matrix_df[target_col].std()
                        print(
                            f"  Target mean: {target_mean:.3f}, std: {target_std:.3f}"
                        )

        print(f"\nTotal unique features across all matrices: {total_features}")

        # Sample feature types
        if feature_matrices:
            sample_matrix = next(iter(feature_matrices.values()))
            feature_cols = [
                col
                for col in sample_matrix.columns
                if col
                not in [
                    "game_id",
                    "season",
                    "week",
                    "feature_timestamp",
                    "target_wp",
                    "target_ats",
                    "target_ou",
                ]
            ]

            feature_types = {}
            for col in feature_cols:
                if "elo_" in col:
                    feature_types.setdefault("Elo", []).append(col)
                elif any(
                    x in col
                    for x in ["home_off_", "away_off_", "home_def_", "away_def_"]
                ):
                    feature_types.setdefault("Team Form", []).append(col)
                elif any(
                    x in col
                    for x in ["travel_", "rest_", "venue_", "thursday_", "short_"]
                ):
                    feature_types.setdefault("Contextual", []).append(col)
                elif any(x in col for x in ["wind_", "temp_", "precip_", "weather_"]):
                    feature_types.setdefault("Weather", []).append(col)
                elif any(
                    x in col
                    for x in ["ml_", "spread_", "total_", "opening_", "snapshot_"]
                ):
                    feature_types.setdefault("Market", []).append(col)
                else:
                    feature_types.setdefault("Other", []).append(col)

            print("\nFeature breakdown by type:")
            for ftype, fcols in feature_types.items():
                print(f"  {ftype}: {len(fcols)} features")

        # Save feature matrices if requested
        if args.save:
            builder.save_feature_matrices(feature_matrices, args.season)
            logger.info("Saved all feature matrices to gold layer")

        logger.info("Feature matrix building completed successfully")

    except _SOURCE_LOAD_ERRORS as e:
        logger.error("Failed to build feature matrices", error=str(e))
        raise


if __name__ == "__main__":
    main()
