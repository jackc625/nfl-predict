"""QB Starter Tracking and Quality Metric Feature Builder.

This module implements QB starter detection from depth chart data and
computes a composite QB quality metric for the NFL prediction system.

QB Starter Detection (FEAT-13):
- Primary: nflreadpy depth charts (position=="QB", depth_team=="1")
- Historical ground truth: PBP passer_player_id (most pass attempts per game)
- Change detection: week-over-week gsis_id comparison

QB Quality Metric (FEAT-12):
- Composite: 0.7 * normalized_rolling_qb_epa + 0.3 * normalized_rolling_cpoe
- Rolling metrics use expanding window with prior-season bootstrap
  (same dynamic window logic as TeamFormCalculator)
- Z-score normalization within the window
- Default to 0.0 (league average) for QBs with no history

Key constraints:
- No data leakage: features for Week N use only data from weeks < N
- Canonical team abbreviations: LA is Rams (not LAR), per Phase 2 decision
- Single composite metric per team per game (D-03)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from features.provenance import DECLARED_GAME_DURATION, InformationTimeViolation
from utils import get_logger
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Composite QB quality weights (D-01 recommendation from RESEARCH.md)
EPA_WEIGHT = 0.7
CPOE_WEIGHT = 0.3

# Dynamic window parameters (matching TeamFormCalculator)
MAX_PRIOR_GAMES = 8


class DepthChartSchemaError(LookupError):
    """A depth-chart frame carries neither a ``dt`` nor a ``week`` column.

    Neither schema can be resolved, so no starter can be named. A ``LookupError``, which is
    NOT a member of ``scripts.build_features._SOURCE_LOAD_ERRORS`` (``KeyError`` is a
    subclass of ``LookupError``, not the other way round), so it reaches the caller instead
    of becoming an empty QB frame -- the same shape as ``UnknownSurfaceError``.
    """


class UntimeablePlayByPlayError(LookupError):
    """Play-by-play rows cannot be matched to a scheduled game, so their end is unknown.

    A play with no end instant cannot be compared to a lock. Refused by name rather than
    admitted, for the reason ``DepthChartSchemaError`` gives about its base class.
    """


@dataclass(frozen=True)
class DepthChartQBs:
    """A team's QB1 and QB2 as a depth chart published them, and WHEN it did.

    ``published_at`` is the upstream ``dt`` of the snapshot used (2025+ schema) -- a real
    publication time. It is ``None`` on the week-keyed 2002-2024 schema, whose charts carry
    no publication time at all; that is stated, not manufactured.
    """

    qb1: str | None
    qb2: str | None
    published_at: pd.Timestamp | None


def to_aware_utc(values: pd.Series, *, column: str) -> pd.Series:
    """*values* as tz-aware UTC instants; a naive value is REFUSED, never relabelled.

    ``pd.to_datetime(..., utc=True)`` would stamp a naive wall clock as UTC, which is the
    convert-never-relabel defect D33.2-01 forbids. The refusal is an
    ``InformationTimeViolation`` so the build's optional-source handler cannot swallow it.

    Args:
        values: A series of instants: tz-aware datetimes, ISO-8601 strings carrying an
            offset, or nulls.
        column: The column the values came from, named in a refusal.

    Returns:
        A ``datetime64[..., UTC]`` series on the same index (nulls stay NaT).
    """
    if len(values) == 0 or bool(values.isna().all()):
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_dtype(values):
        _refuse_naive(column)
    try:
        parsed = pd.to_datetime(values, format="ISO8601")
    except (ValueError, TypeError):
        # Mixed offsets cannot be parsed as one vector without utc=True, which would
        # relabel any naive member; parse element by element instead.
        return values.map(lambda value: _one_aware_utc(value, column))
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        _refuse_naive(column)
    return parsed.dt.tz_convert("UTC")


def _one_aware_utc(value: Any, column: str) -> pd.Timestamp:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT  # type: ignore[return-value]
    instant = pd.Timestamp(value)
    if instant.tzinfo is None:
        _refuse_naive(column)
    return instant.tz_convert("UTC")


def _refuse_naive(column: str) -> None:
    msg = (
        f"the {column!r} column carries a naive (tz-unaware) instant. It cannot be "
        "compared to a game's lock, and a naive value is refused rather than relabelled."
    )
    raise InformationTimeViolation(
        msg, {"column": column, "violation_type": "naive_information_time"}
    )


def lock_as_utc(lock: Any) -> pd.Timestamp:
    """A lock (from ``utils.game_lock``) as a UTC ``Timestamp``; a naive lock is refused."""
    instant = pd.Timestamp(lock)
    if instant.tzinfo is None:
        _refuse_naive("lock")
    return instant.tz_convert("UTC")


class QBTracker:
    """Track QB starters and compute composite QB quality metric.

    Conforms to the FeatureBuilder Protocol with as_of_datetime time-fence.
    Produces a single `qb_adjustment` feature per team per game.

    The composite metric combines:
    - 70% normalized rolling QB EPA (outcome correlation)
    - 30% normalized rolling CPOE (sustainable talent signal)
    """

    def __init__(self, max_prior_games: int = MAX_PRIOR_GAMES) -> None:
        """Initialize QB tracker.

        Args:
            max_prior_games: Maximum prior-season games to include in the
                rolling window when current-season data is sparse.
        """
        self.max_prior_games = max_prior_games

        # Caches to avoid repeated nflreadpy loads
        self._depth_chart_cache: dict[int, pd.DataFrame] = {}
        self._pbp_cache: dict[int, pd.DataFrame] = {}
        self._games_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # QB Starter Detection (FEAT-13)
    # ------------------------------------------------------------------

    def get_starters_from_depth_charts(
        self, depth_charts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract QB1 starters from depth chart data.

        Filters to position=="QB" and depth_team=="1", normalizes team
        abbreviations, and detects week-over-week starter changes.

        Args:
            depth_charts_df: Depth chart DataFrame with columns:
                season, week, club_code, position, depth_team, full_name, gsis_id

        Returns:
            DataFrame with columns: season, week, team, gsis_id, full_name,
            starter_changed (bool)
        """
        # Filter to QB1 only
        qb1 = depth_charts_df[
            (depth_charts_df["position"] == "QB")
            & (depth_charts_df["depth_team"] == "1")
        ].copy()

        if len(qb1) == 0:
            logger.warning("No QB1 entries found in depth charts")
            return pd.DataFrame(
                columns=[
                    "season",
                    "week",
                    "team",
                    "gsis_id",
                    "full_name",
                    "starter_changed",
                ]
            )

        # Normalize team abbreviations
        qb1["team"] = qb1["club_code"].apply(self._safe_normalize_team)

        # Select relevant columns
        starters = qb1[["season", "week", "team", "gsis_id", "full_name"]].copy()

        # Sort for change detection
        starters = starters.sort_values(["team", "season", "week"]).reset_index(
            drop=True
        )

        # Detect starter changes: compare gsis_id to previous week for same team
        starters["prev_gsis_id"] = starters.groupby("team")["gsis_id"].shift(1)
        starters["starter_changed"] = (
            starters["gsis_id"] != starters["prev_gsis_id"]
        ) & starters["prev_gsis_id"].notna()

        # Clean up -- drop the helper column
        starters = starters.drop(columns=["prev_gsis_id"])

        logger.info(
            "Extracted QB starters from depth charts",
            total_entries=len(starters),
            teams=len(starters["team"].unique()),
            changes=int(starters["starter_changed"].sum()),
        )

        return starters

    # ------------------------------------------------------------------
    # Per-Game QB Stats from PBP (FEAT-12)
    # ------------------------------------------------------------------

    def compute_per_game_qb_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-game QB statistics from play-by-play data.

        For each game-team combination, identifies the primary passer
        (most pass attempts) and computes:
        - mean_qb_epa: mean of qb_epa across all pass plays
        - mean_cpoe: mean of cpoe across non-null plays only
        - pass_attempts: count of pass plays

        Args:
            pbp_df: PBP DataFrame with columns:
                game_id, season, week, posteam, passer_player_id, qb_epa, cpoe

        Returns:
            DataFrame with one row per primary passer per game-team,
            columns: game_id, season, week, posteam, passer_player_id,
            pass_attempts, mean_qb_epa, mean_cpoe
        """
        if len(pbp_df) == 0:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "week",
                    "posteam",
                    "passer_player_id",
                    "pass_attempts",
                    "mean_qb_epa",
                    "mean_cpoe",
                ]
            )

        # Filter to plays with a passer
        pass_plays = pbp_df[pbp_df["passer_player_id"].notna()].copy()

        if len(pass_plays) == 0:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "week",
                    "posteam",
                    "passer_player_id",
                    "pass_attempts",
                    "mean_qb_epa",
                    "mean_cpoe",
                ]
            )

        # Compute per-game stats for each passer
        grouped = pass_plays.groupby(
            ["game_id", "season", "week", "posteam", "passer_player_id"]
        )

        qb_game_stats = grouped.agg(
            pass_attempts=("qb_epa", "count"),
            mean_qb_epa=("qb_epa", "mean"),
            mean_cpoe=(
                "cpoe",
                "mean",
            ),  # NaN cpoe values are automatically excluded by mean
        ).reset_index()

        # Identify primary passer per game-team (most pass attempts)
        primary_qb = (
            qb_game_stats.sort_values("pass_attempts", ascending=False)
            .groupby(["game_id", "posteam"])
            .first()
            .reset_index()
        )

        logger.info(
            "Computed per-game QB stats",
            total_passer_entries=len(qb_game_stats),
            primary_passers=len(primary_qb),
        )

        return primary_qb

    # ------------------------------------------------------------------
    # Rolling QB Metrics
    # ------------------------------------------------------------------

    def _select_dynamic_window(
        self,
        qb_games: pd.DataFrame,
        target_season: int,
        target_week: int,
    ) -> pd.DataFrame:
        """Select games for the dynamic expanding window.

        Same logic as TeamFormCalculator._select_dynamic_window:
        - current_games: all games in target_season with week < target_week
        - prior_games: last N of (target_season - 1) where
          N = max(0, max_prior_games - current_count)

        Args:
            qb_games: All historical games for one QB, sorted chronologically.
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            DataFrame subset with the games to include in the window.
        """
        # Filter to games before the target
        historical = qb_games[
            (qb_games["season"] < target_season)
            | ((qb_games["season"] == target_season) & (qb_games["week"] < target_week))
        ]

        current_season_games = historical[historical["season"] == target_season]
        prior_season_games = historical[historical["season"] == target_season - 1]

        current_count = len(current_season_games)
        prior_count = max(0, self.max_prior_games - current_count)

        prior_tail = prior_season_games.tail(prior_count)
        combined = pd.concat([prior_tail, current_season_games])
        return combined.sort_values(["season", "week"])

    def compute_rolling_qb_metrics(
        self,
        pbp_df: pd.DataFrame,
        target_season: int,
        target_week: int,
    ) -> pd.DataFrame:
        """Compute rolling QB metrics for all passers as of a target week.

        Uses the dynamic expanding window (prior-season bootstrap) and
        z-score normalization within the window.

        Args:
            pbp_df: PBP DataFrame for relevant seasons.
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            DataFrame with one row per QB, columns: passer_player_id,
            rolling_qb_epa, rolling_cpoe, norm_rolling_qb_epa,
            norm_rolling_cpoe, qb_quality
        """
        qb_stats = self.compute_per_game_qb_stats(pbp_df)

        if len(qb_stats) == 0:
            return pd.DataFrame(
                columns=[
                    "passer_player_id",
                    "rolling_qb_epa",
                    "rolling_cpoe",
                    "norm_rolling_qb_epa",
                    "norm_rolling_cpoe",
                    "qb_quality",
                ]
            )

        results = []

        for passer_id, passer_games in qb_stats.groupby("passer_player_id"):
            passer_games = passer_games.sort_values(["season", "week"])

            # Select games within the dynamic window
            window_games = self._select_dynamic_window(
                passer_games, target_season, target_week
            )

            if len(window_games) == 0:
                continue

            # Compute rolling metrics with recency weighting
            weights = np.arange(1, len(window_games) + 1, dtype=float)
            weights = weights / weights.sum()

            epa_values = window_games["mean_qb_epa"].values
            cpoe_values = window_games["mean_cpoe"].values

            # Weighted mean for EPA (all values should be present)
            valid_epa_mask = ~np.isnan(epa_values)
            if valid_epa_mask.sum() > 0:
                valid_epa = epa_values[valid_epa_mask]
                valid_epa_weights = weights[valid_epa_mask]
                valid_epa_weights = valid_epa_weights / valid_epa_weights.sum()
                rolling_epa = float(np.average(valid_epa, weights=valid_epa_weights))
            else:
                rolling_epa = 0.0

            # Weighted mean for CPOE (may have NaN from sack-heavy games)
            valid_cpoe_mask = ~np.isnan(cpoe_values)
            if valid_cpoe_mask.sum() > 0:
                valid_cpoe = cpoe_values[valid_cpoe_mask]
                valid_cpoe_weights = weights[valid_cpoe_mask]
                valid_cpoe_weights = valid_cpoe_weights / valid_cpoe_weights.sum()
                rolling_cpoe = float(np.average(valid_cpoe, weights=valid_cpoe_weights))
            else:
                rolling_cpoe = 0.0

            results.append(
                {
                    "passer_player_id": passer_id,
                    "rolling_qb_epa": rolling_epa,
                    "rolling_cpoe": rolling_cpoe,
                    "games_used": len(window_games),
                }
            )

        if not results:
            return pd.DataFrame(
                columns=[
                    "passer_player_id",
                    "rolling_qb_epa",
                    "rolling_cpoe",
                    "norm_rolling_qb_epa",
                    "norm_rolling_cpoe",
                    "qb_quality",
                ]
            )

        rolling_df = pd.DataFrame(results)

        # Z-score normalization across all QBs in the window
        rolling_df["norm_rolling_qb_epa"] = self._zscore(rolling_df["rolling_qb_epa"])
        rolling_df["norm_rolling_cpoe"] = self._zscore(rolling_df["rolling_cpoe"])

        # Composite quality
        rolling_df["qb_quality"] = rolling_df.apply(
            lambda row: self.compute_composite_quality(
                row["norm_rolling_qb_epa"], row["norm_rolling_cpoe"]
            ),
            axis=1,
        )

        return rolling_df

    # ------------------------------------------------------------------
    # Composite Quality Metric
    # ------------------------------------------------------------------

    @staticmethod
    def compute_composite_quality(
        norm_epa: float | None,
        norm_cpoe: float | None,
    ) -> float:
        """Compute the composite QB quality metric.

        Formula: qb_quality = 0.7 * normalized_rolling_qb_epa
                             + 0.3 * normalized_rolling_cpoe

        When either input is None or NaN, defaults to 0.0 (league average).

        Args:
            norm_epa: Z-scored rolling QB EPA.
            norm_cpoe: Z-scored rolling CPOE.

        Returns:
            Composite QB quality value. 0.0 means league average.
        """
        if norm_epa is None or (isinstance(norm_epa, float) and np.isnan(norm_epa)):
            return 0.0
        if norm_cpoe is None or (isinstance(norm_cpoe, float) and np.isnan(norm_cpoe)):
            return 0.0

        qb_quality = EPA_WEIGHT * norm_epa + CPOE_WEIGHT * norm_cpoe
        return float(qb_quality)

    # ------------------------------------------------------------------
    # Per-game lock helpers (Phase 33.2, Plan 33.2-13): shared with InjuryBuilder
    # ------------------------------------------------------------------

    def resolve_depth_chart_qbs(
        self,
        depth_charts: pd.DataFrame,
        season: int,
        week: int,
        team: str,
        lock: Any,
    ) -> DepthChartQBs:
        """A team's QB1 / QB2 as known at *lock*, branching on the depth-chart SCHEMA.

        * A ``dt`` column (the 2025+ schema): nflverse stopped assigning weeks after 2024
          and appends each update with an ISO-8601 publication time instead. The snapshot
          used is the LATEST one whose ``dt`` is at or before the lock; a snapshot published
          one second later is not seen. ``dt`` is a genuine upstream publication time, so
          the identity carries a REAL information time -- better provenance than the
          week-keyed path can offer.
        * A ``week`` column (2002-2024): the chart filed for the target week, and never a
          later week's. That chart has no publication time, so ``published_at`` is None.

        The branch is on the column, never on a season literal. The 2025+ frames also carry
        ``espn_id`` beside ``gsis_id`` -- the bridge for anything ESPN-sourced. It is
        recorded here and deliberately unused (Plan 33.2-15 settles the ESPN question).

        Args:
            depth_charts: A depth-chart frame in the loader's normalised column names
                (``club_code``, ``position``, ``depth_team``, ``gsis_id``).
            season: The target game's season.
            week: The target game's week.
            team: Canonical team abbreviation.
            lock: The target game's lock (``utils.game_lock``), tz-aware.

        Returns:
            The resolved QB1 / QB2 and the publication time of the snapshot used.

        Raises:
            DepthChartSchemaError: the frame carries neither ``dt`` nor ``week``.
        """
        if depth_charts is None or len(depth_charts) == 0:
            return DepthChartQBs(None, None, None)
        if "dt" not in depth_charts.columns and "week" not in depth_charts.columns:
            msg = (
                "a depth-chart frame carries neither 'dt' (2025+) nor 'week' (2002-2024), "
                f"so no starter can be resolved; columns: {sorted(depth_charts.columns)}"
            )
            raise DepthChartSchemaError(msg)

        qbs = depth_charts.loc[depth_charts["position"] == "QB"]
        qbs = qbs.loc[qbs["club_code"].map(self._safe_normalize_team) == team]
        if len(qbs) == 0:
            return DepthChartQBs(None, None, None)

        published_at: pd.Timestamp | None = None
        if "dt" in qbs.columns:
            published = to_aware_utc(qbs["dt"], column="dt")
            # The at-lock-admissible comparison utils.game_lock.is_admissible states (<=),
            # applied to a vector of publication times against this game's one lock.
            admitted = published.notna() & (published <= lock_as_utc(lock))
            if not bool(admitted.any()):
                return DepthChartQBs(None, None, None)
            latest = published.loc[admitted].max()
            snapshot = qbs.loc[admitted & (published == latest)]
            published_at = pd.Timestamp(latest)
        else:
            snapshot = qbs
            if "season" in snapshot.columns:
                snapshot = snapshot.loc[snapshot["season"] == season]
            snapshot = snapshot.loc[snapshot["week"] == week]

        depth = snapshot["depth_team"].astype(str)
        qb1 = snapshot.loc[depth == "1"]
        qb2 = snapshot.loc[depth == "2"]
        return DepthChartQBs(
            qb1=str(qb1.iloc[0]["gsis_id"]) if len(qb1) > 0 else None,
            qb2=str(qb2.iloc[0]["gsis_id"]) if len(qb2) > 0 else None,
            published_at=published_at,
        )

    def pbp_game_end_times(
        self, pbp: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.Series:
        """When each play-by-play game's RESULT became known: kickoff plus the duration.

        The duration is ``features.provenance.DECLARED_GAME_DURATION`` (4 h) -- the SAME
        value ``audit.elo_replay.DEFAULT_GAME_DURATION_HOURS`` declares, reused rather than
        re-declared, so the builder and the replay cannot disagree about when a game ended.

        PBP ids (``2024_01_BUF_KC``) and silver ids (``2024_W01_BUF@KC``) never match, so a
        PBP game is keyed to its scheduled game by (season, week, home team, away team).
        A PBP game with no scheduled match in *games_df* has no known end and is left out
        of the result -- it is therefore never admitted, which is the conservative answer.

        Returns:
            PBP ``game_id`` -> tz-aware UTC end instant.

        Raises:
            UntimeablePlayByPlayError: non-empty PBP without ``home_team`` / ``away_team``.
        """
        empty = pd.Series(dtype="datetime64[ns, UTC]", name="end")
        if pbp is None or len(pbp) == 0 or len(games_df) == 0:
            return empty
        missing = [c for c in ("home_team", "away_team") if c not in pbp.columns]
        if missing:
            msg = (
                f"play-by-play carries no {missing} column(s), so its games cannot be "
                "matched to the schedule and have no end instant to compare to a lock"
            )
            raise UntimeablePlayByPlayError(msg)

        keys = ["season", "week", "home_team", "away_team"]
        pbp_games = pbp[["game_id", *keys]].drop_duplicates("game_id").copy()
        schedule = games_df[[*keys, "kickoff_et"]].copy()
        for frame in (pbp_games, schedule):
            for column in ("home_team", "away_team"):
                frame[column] = frame[column].map(self._safe_normalize_team)
            for column in ("season", "week"):
                frame[column] = pd.to_numeric(frame[column]).astype("int64")
        schedule["end"] = (
            to_aware_utc(schedule["kickoff_et"], column="kickoff_et")
            + DECLARED_GAME_DURATION
        )
        matched = pbp_games.merge(
            schedule.drop(columns="kickoff_et").drop_duplicates(keys),
            on=keys,
            how="inner",
        )
        return pd.Series(
            matched["end"].to_numpy(), index=matched["game_id"].to_numpy(), name="end"
        )

    @staticmethod
    def admitted_pbp(
        pbp: pd.DataFrame, game_ends: pd.Series, lock: Any
    ) -> pd.DataFrame:
        """The plays from games whose result was known at or before *lock*.

        Uses the at-lock-admissible comparison ``utils.game_lock.is_admissible`` states
        (``end <= lock``), vectorised over the game ends.
        """
        if pbp is None or len(pbp) == 0 or len(game_ends) == 0:
            return pbp.iloc[0:0] if pbp is not None else pd.DataFrame()
        admitted_ids = game_ends.index[game_ends <= lock_as_utc(lock)]
        return pbp.loc[pbp["game_id"].isin(admitted_ids)]

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build QB adjustment features for games.

        Conforms to the FeatureBuilder Protocol. Uses as_of_datetime to
        enforce the time-fence: only depth chart and PBP data from before
        this cutoff are used.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.

        Returns:
            DataFrame with columns: game_id, team, qb_adjustment
        """
        logger.info(
            "Building QB adjustment features",
            as_of_datetime=str(as_of_datetime),
            target_season=target_season,
            target_week=target_week,
        )

        if target_season is None or target_week is None:
            # Build for ALL season/week combos in games_df
            if len(games_df) == 0:
                return pd.DataFrame(columns=["game_id", "team", "qb_adjustment"])
            self._games_cache = games_df
            all_results = []
            season_weeks = (
                games_df[["season", "week"]]
                .drop_duplicates()
                .sort_values(["season", "week"])
            )
            for _, row in season_weeks.iterrows():
                try:
                    chunk = self.build_features(
                        games_df,
                        as_of_datetime,
                        target_season=int(row["season"]),
                        target_week=int(row["week"]),
                    )
                    if len(chunk) > 0:
                        all_results.append(chunk)
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(
                        "Skipping QB features for season/week",
                        season=int(row["season"]),
                        week=int(row["week"]),
                        error=str(e),
                    )
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            return pd.DataFrame(columns=["game_id", "team", "qb_adjustment"])

        # Store games for get_features_for_game lookups
        self._games_cache = games_df

        # Load data (from cache or nflreadpy)
        depth_charts = self._load_depth_charts(target_season)
        pbp = self._load_pbp_data(target_season)

        # Apply as_of_datetime filter to PBP data
        # Only use PBP data from games before the time-fence.
        # Uses season/week pairs instead of game_id matching because the
        # silver-layer games table has a different game_id format than PBP
        # (e.g. "2024_W01_KC@BUF" vs "2024_01_KC_BUF"). See 05-06-PLAN.md.
        if "kickoff_et" in games_df.columns:
            kickoff_col = pd.to_datetime(games_df["kickoff_et"])
            cutoff = pd.Timestamp(as_of_datetime)
            if kickoff_col.dt.tz is not None and cutoff.tz is None:
                cutoff = cutoff.tz_localize(kickoff_col.dt.tz)
            completed_games = games_df[kickoff_col < cutoff]
            if len(pbp) > 0 and len(completed_games) > 0:
                completed_sw = completed_games[["season", "week"]].drop_duplicates()
                pbp = pbp.merge(completed_sw, on=["season", "week"], how="inner")

        # Get rolling QB metrics
        rolling_qb = self.compute_rolling_qb_metrics(pbp, target_season, target_week)

        # Get starters for the target week
        starters = self.get_starters_from_depth_charts(depth_charts)
        target_starters = starters[
            (starters["season"] == target_season) & (starters["week"] == target_week)
        ]

        # Also use PBP primary passers as fallback for historical games
        primary_passers = self.compute_per_game_qb_stats(pbp)

        # Build feature rows: one per team per game in the target week
        target_games = games_df[
            (games_df["season"] == target_season) & (games_df["week"] == target_week)
        ]

        feature_rows = []
        for _, game in target_games.iterrows():
            game_id = game["game_id"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            for team in [home_team, away_team]:
                qb_adj = self._get_qb_adjustment_for_team(
                    team,
                    target_starters,
                    primary_passers,
                    rolling_qb,
                    target_season,
                    target_week,
                )
                feature_rows.append(
                    {
                        "game_id": game_id,
                        "team": team,
                        "qb_adjustment": qb_adj,
                    }
                )

        result = pd.DataFrame(feature_rows)

        logger.info(
            "Built QB adjustment features",
            records=len(result),
            target_season=target_season,
            target_week=target_week,
        )

        return result

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get QB adjustment features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary with "home_qb_adjustment" and "away_qb_adjustment".
        """
        # Parse game_id to extract season, week, teams
        # Expected format: {season}_{week:02d}_{away}_{home} or similar
        parts = game_id.split("_")
        if len(parts) < 4:
            logger.warning(
                "Cannot parse game_id for QB tracking lookup",
                game_id=game_id,
            )
            return {"home_qb_adjustment": 0.0, "away_qb_adjustment": 0.0}

        season = int(parts[0])
        week = int(parts[1])

        # Look up teams from games cache or parse from game_id
        home_team, away_team = self._resolve_teams_for_game(game_id, parts)

        # Load data
        depth_charts = self._load_depth_charts(season)
        pbp = self._load_pbp_data(season)

        # Apply time fence using season/week pairs (format-agnostic).
        # See build_features() comment for rationale on season/week vs game_id.
        if self._games_cache is not None and "kickoff_et" in self._games_cache.columns:
            kickoff_col = pd.to_datetime(self._games_cache["kickoff_et"])
            cutoff = pd.Timestamp(as_of_datetime)
            if kickoff_col.dt.tz is not None and cutoff.tz is None:
                cutoff = cutoff.tz_localize(kickoff_col.dt.tz)
            completed_games = self._games_cache[kickoff_col < cutoff]
            if len(pbp) > 0 and len(completed_games) > 0:
                completed_sw = completed_games[["season", "week"]].drop_duplicates()
                pbp = pbp.merge(completed_sw, on=["season", "week"], how="inner")

        rolling_qb = self.compute_rolling_qb_metrics(pbp, season, week)
        starters = self.get_starters_from_depth_charts(depth_charts)
        target_starters = starters[
            (starters["season"] == season) & (starters["week"] == week)
        ]
        primary_passers = self.compute_per_game_qb_stats(pbp)

        home_adj = self._get_qb_adjustment_for_team(
            home_team, target_starters, primary_passers, rolling_qb, season, week
        )
        away_adj = self._get_qb_adjustment_for_team(
            away_team, target_starters, primary_passers, rolling_qb, season, week
        )

        return {
            "home_qb_adjustment": float(home_adj),
            "away_qb_adjustment": float(away_adj),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_qb_adjustment_for_team(
        self,
        team: str,
        starters: pd.DataFrame,
        primary_passers: pd.DataFrame,
        rolling_qb: pd.DataFrame,
        target_season: int,
        target_week: int,
    ) -> float:
        """Look up the QB quality for a team's starter.

        Strategy:
        1. Check depth chart starters for the target week
        2. Fall back to most recent primary passer from PBP
        3. If no QB found or no rolling data, return 0.0 (league avg)

        Args:
            team: Canonical team abbreviation.
            starters: Depth chart starters DataFrame.
            primary_passers: PBP-derived primary passers.
            rolling_qb: Rolling QB metrics DataFrame.
            target_season: Target season.
            target_week: Target week.

        Returns:
            QB quality value (0.0 = league average).
        """
        qb_id = None

        # Try depth chart first
        team_starter = starters[starters["team"] == team]
        if len(team_starter) > 0:
            qb_id = team_starter.iloc[0]["gsis_id"]

        # Fall back to most recent primary passer from PBP
        if qb_id is None and len(primary_passers) > 0:
            team_passers = primary_passers[
                primary_passers["posteam"] == team
            ].sort_values(["season", "week"], ascending=False)
            if len(team_passers) > 0:
                qb_id = team_passers.iloc[0]["passer_player_id"]

        if qb_id is None:
            return 0.0

        # Look up rolling quality
        if len(rolling_qb) == 0:
            return 0.0

        qb_row = rolling_qb[rolling_qb["passer_player_id"] == qb_id]
        if len(qb_row) == 0:
            return 0.0

        return float(qb_row.iloc[0]["qb_quality"])

    def _resolve_teams_for_game(
        self, game_id: str, parts: list[str]
    ) -> tuple[str, str]:
        """Resolve home and away teams for a game_id.

        Checks games cache first, then falls back to parsing the game_id.

        Args:
            game_id: The game identifier.
            parts: Pre-split parts of game_id.

        Returns:
            Tuple of (home_team, away_team).
        """
        # Check cache
        if self._games_cache is not None:
            game_row = self._games_cache[self._games_cache["game_id"] == game_id]
            if len(game_row) > 0:
                return (
                    str(game_row.iloc[0]["home_team"]),
                    str(game_row.iloc[0]["away_team"]),
                )

        # Parse from game_id: {season}_{week}_{home}_{away}
        if len(parts) >= 4:
            return parts[2], parts[3]

        return "UNK", "UNK"

    def _load_depth_charts(self, season: int) -> pd.DataFrame:
        """Load depth chart data, with caching.

        Args:
            season: Season year.

        Returns:
            Depth chart DataFrame.
        """
        if season in self._depth_chart_cache:
            return self._depth_chart_cache[season]

        try:
            from data import upstream_pin

            # PINNED read, not a live fetch. ``upstream_pin`` raises UpstreamPinError,
            # which is deliberately NOT a RuntimeError/ValueError/ImportError, so the
            # except clause below cannot swallow a pin refusal into an empty frame.
            dc = upstream_pin.load_depth_charts(season)

            # nflreadpy changed depth chart schema in 2025+:
            #   Old (<=2024): club_code, position, depth_team, full_name, week, gsis_id
            #   New (>=2025): team, pos_abb, pos_rank, player_name, dt, gsis_id
            # Normalize to old schema for consistency.
            if "pos_abb" in dc.columns and "position" not in dc.columns:
                rename_map = {
                    "team": "club_code",
                    "pos_abb": "position",
                    "pos_rank": "depth_team",
                    "player_name": "full_name",
                }
                dc = dc.rename(columns=rename_map)
                # depth_team needs to be string "1" for QB1
                dc["depth_team"] = dc["depth_team"].astype(str)
                # Add season column if missing
                if "season" not in dc.columns:
                    dc["season"] = season
                # Add week column from dt if missing
                if "week" not in dc.columns and "dt" in dc.columns:
                    # dt column may encode week; default to 0 if not parseable
                    dc["week"] = 0

            self._depth_chart_cache[season] = dc
            return dc
        except (ImportError, ValueError, RuntimeError) as e:
            logger.error(
                "Failed to load depth charts",
                season=season,
                error=str(e),
            )
            return pd.DataFrame(
                columns=[
                    "season",
                    "week",
                    "club_code",
                    "position",
                    "depth_team",
                    "full_name",
                    "gsis_id",
                ]
            )

    def _load_pbp_data(self, season: int) -> pd.DataFrame:
        """Load play-by-play data, with caching.

        Loads the target season and prior season for dynamic window.

        Args:
            season: Target season year.

        Returns:
            PBP DataFrame filtered to pass plays.
        """
        seasons_needed = [season - 1, season]
        all_pbp = []

        for s in seasons_needed:
            if s in self._pbp_cache:
                all_pbp.append(self._pbp_cache[s])
                continue

            try:
                from data import upstream_pin

                # PINNED read; see the note in _load_depth_charts about why a pin
                # refusal cannot be caught by the handler below.
                pbp = upstream_pin.load_pbp([s])
                # Filter to pass plays with passer info
                pbp = pbp[pbp["passer_player_id"].notna()].copy()
                self._pbp_cache[s] = pbp
                all_pbp.append(pbp)
            except (ImportError, ValueError, RuntimeError) as e:
                logger.warning(
                    "Failed to load PBP data",
                    season=s,
                    error=str(e),
                )

        if all_pbp:
            return pd.concat(all_pbp, ignore_index=True)
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "posteam",
                "passer_player_id",
                "qb_epa",
                "cpoe",
                "play_id",
            ]
        )

    @staticmethod
    def _safe_normalize_team(team: str) -> str:
        """Normalize team abbreviation, falling back to uppercase if unknown.

        Args:
            team: Raw team abbreviation.

        Returns:
            Canonical team abbreviation.
        """
        if not team or pd.isna(team):
            return str(team)
        from utils.exceptions import DataValidationError

        try:
            return normalize_team_abbreviation(str(team))
        except DataValidationError:
            return str(team).upper().strip()

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Compute z-scores for a series. Returns 0.0 if std is 0.

        Args:
            series: Input series of values.

        Returns:
            Z-scored series.
        """
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
