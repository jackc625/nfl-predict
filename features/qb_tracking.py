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
- No data leakage: every input is fenced at the TARGET GAME'S OWN LOCK -- 18:00 ET on
  the ET calendar day before kickoff (utils.game_lock, D33.2-01). QB1 comes from the
  depth chart as known at the lock; play-by-play is admitted per GAME, only when that
  game ended (kickoff + DECLARED_GAME_DURATION) at or before the lock. The rolling
  window additionally reads only weeks < N.
- The 2025+ depth charts carry a real upstream publication time (`dt`) instead of a
  week; the 2002-2024 week-keyed charts carry none (Plan 33.2-13).
- Canonical team abbreviations: LA is Rams (not LAR), per Phase 2 decision
- Single composite metric per team per game (D-03)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from features.provenance import (
    DECLARED_GAME_DURATION,
    PROVENANCE_COLUMNS,
    InformationBasis,
    InformationTimeViolation,
)
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


@dataclass
class _WeekContext:
    """What one (season, week) reads, loaded once and shared by all its games."""

    season: int
    week: int
    depth: pd.DataFrame
    pbp: pd.DataFrame
    game_ends: pd.Series
    window_ids: frozenset[str]
    rolling: dict[frozenset[str], pd.DataFrame] = field(default_factory=dict)
    primary: dict[frozenset[str], pd.DataFrame] = field(default_factory=dict)
    # One resolution per (game, lock): information_times re-reads exactly what
    # build_features resolved rather than resolving it a second time.
    inputs: dict[tuple[str, pd.Timestamp], "_GameQBInputs"] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _GameQBInputs:
    """Both teams' adjustments for one game, and when the inputs behind them were known."""

    adjustments: dict[str, float]
    published: tuple[pd.Timestamp, ...]
    latest_end: pd.Timestamp | None


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


def _one_aware_utc(value: Any, column: str) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
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
    return cast(pd.Timestamp, instant.tz_convert("UTC"))


class QBTracker:
    """Track QB starters and compute composite QB quality metric.

    Conforms to the FeatureBuilder Protocol AND to
    ``features.protocol.InformationTimeProvider``: every input is fenced at the target
    game's own lock, and ``information_times`` reports when the inputs used were known.
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
        # Per-(season, week) contexts, each held beside the games frame it was built
        # from and reused only for that SAME frame object (a narrowed frame -- e.g. a
        # live run's exclusions -- times PBP against a different schedule).
        self._context_memo: dict[
            tuple[int, int], tuple[pd.DataFrame, _WeekContext]
        ] = {}

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
        codes = qbs["club_code"]
        canonical = {code: self._safe_normalize_team(code) for code in codes.unique()}
        qbs = qbs.loc[codes.map(canonical) == team]
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
            published_at = cast(pd.Timestamp, pd.Timestamp(latest))
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
        pbp_games = pd.DataFrame(pbp[["game_id", *keys]]).drop_duplicates(
            subset=["game_id"]
        )
        schedule = pd.DataFrame(games_df[[*keys, "kickoff_et"]])
        for frame in (pbp_games, schedule):
            for column in ("home_team", "away_team"):
                frame[column] = pd.Series(frame[column]).map(self._safe_normalize_team)
            for column in ("season", "week"):
                frame[column] = pd.Series(
                    pd.to_numeric(pd.Series(frame[column]))
                ).astype("int64")
        schedule["end"] = (
            to_aware_utc(pd.Series(schedule["kickoff_et"]), column="kickoff_et")
            + DECLARED_GAME_DURATION
        )
        matched = pbp_games.merge(
            schedule.drop(columns="kickoff_et").drop_duplicates(subset=keys),
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
        admitted_ids = game_ends.index[game_ends <= lock_as_utc(lock)].tolist()
        return pbp.loc[pbp["game_id"].isin(admitted_ids)]

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods, each game fenced at its OWN lock
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        lock_frame: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build QB adjustment features, each game fenced at its own lock (D33.2-01).

        For every target game: QB1 comes from the depth chart as known at that game's
        lock (:meth:`resolve_depth_chart_qbs`), and the rolling quality comes only from
        play-by-play games whose RESULT was known at that lock (kickoff plus
        ``DECLARED_GAME_DURATION``, :meth:`pbp_game_end_times`). The retired fence
        compared every kickoff with one frame-wide ``as_of_datetime`` (``now`` in
        production) and admitted PBP by (season, week) pair, so one finished game in a
        week admitted that whole week -- the target game's own plays included.

        Args:
            games_df: The build's games (``game_id``, ``season``, ``week``, teams and a
                tz-aware ``kickoff_et``). Also the schedule PBP games are timed against.
            as_of_datetime: Carried for the ``FeatureBuilder`` Protocol ONLY. It is NOT a
                fence and no selection reads it.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.
            lock_frame: The build's ``game_id`` -> lock frame, built ONCE by the caller.
                When ``None`` it is built here, once, through
                ``utils.game_lock.lock_frame``.

        Returns:
            DataFrame with columns: game_id, team, qb_adjustment
        """
        logger.info(
            "Building QB adjustment features",
            target_season=target_season,
            target_week=target_week,
        )
        columns = ["game_id", "team", "qb_adjustment"]
        if len(games_df) == 0:
            return pd.DataFrame(columns=columns)
        self._games_cache = games_df
        target_games = self._target_games(games_df, target_season, target_week)
        if len(target_games) == 0:
            return pd.DataFrame(columns=columns)
        if lock_frame is None:
            lock_frame = lock_rule.lock_frame(target_games)

        # A fan-out over every week keeps its historical tolerance for one unbuildable
        # week. That week's games then have no QB rows, which the information-time gate
        # refuses by name (a provenance row with no source row), so it cannot pass quietly.
        fan_out = target_season is None or target_week is None
        chunks: list[pd.DataFrame] = []
        for key, week_games in target_games.groupby(["season", "week"], sort=True):
            season, week = cast(tuple[int, int], key)
            try:
                chunks.append(
                    self._build_week(
                        games_df, week_games, int(season), int(week), lock_frame
                    )
                )
            except (ValueError, KeyError, TypeError) as e:
                if not fan_out:
                    raise
                logger.warning(
                    "Skipping QB features for season/week",
                    season=int(season),
                    week=int(week),
                    error=str(e),
                )
        chunks = [chunk for chunk in chunks if len(chunk) > 0]
        result = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=columns)
        )
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
        """QB adjustment for one game, fenced at its lock.

        The game is resolved from the frame handed to the last ``build_features`` call,
        because its lock needs the kickoff; a game that cannot be resolved has no lock and
        is refused by name rather than built from a manufactured cutoff.
        ``as_of_datetime`` is carried for the Protocol only.

        Returns:
            Dictionary with "home_qb_adjustment" and "away_qb_adjustment".

        Raises:
            utils.game_lock.MissingKickoffError: the game is not in the cached frame.
        """
        game = self._resolve_game(game_id)
        if game is None or self._games_cache is None:
            msg = (
                f"game {game_id} is not in the games frame this tracker was given, so it "
                "has no kickoff and therefore no lock; refusing rather than guessing one"
            )
            raise lock_rule.MissingKickoffError(msg)
        season, week = int(game["season"]), int(game["week"])
        lock = lock_rule.game_lock(game.get("kickoff_et"), game_id=str(game_id))
        context = self._week_context(self._games_cache, season, week)
        inputs = self._game_inputs(game, lock, context)
        home = self._safe_normalize_team(game["home_team"])
        away = self._safe_normalize_team(game["away_team"])
        return {
            "home_qb_adjustment": float(inputs.adjustments[home]),
            "away_qb_adjustment": float(inputs.adjustments[away]),
        }

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game with no dated input has no rolling history: ``qb_adjustment`` 0.0.

        ``no_information`` is reported only when no depth-chart snapshot and no finished
        play-by-play game was known at the lock -- and with no admitted PBP the rolling
        metrics are empty, so both teams' adjustment is the league-average 0.0.
        """
        return {"qb_adjustment": 0.0}

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game: WHEN the inputs this builder used were known.

        The time is the MAXIMUM of (the ``dt`` of each depth-chart snapshot used, 2025+)
        and (the end instant of the latest play-by-play game admitted at the lock) --
        what the builder actually read, from the same resolution ``build_features`` runs.
        A week-keyed 2002-2024 chart carries no publication time, so it contributes none;
        no time is manufactured for it. A game with neither input is
        ``basis="no_information"`` with a NULL time, value-checked by the gate.

        Returns:
            A frame with exactly ``PROVENANCE_COLUMNS``.
        """
        target_games = self._target_games(games_df, target_season, target_week)
        records: list[dict[str, Any]] = []
        if len(target_games) > 0:
            locks = lock_rule.lock_frame(target_games)
            for key, week_games in target_games.groupby(["season", "week"], sort=True):
                season, week = cast(tuple[int, int], key)
                context = self._week_context(games_df, int(season), int(week))
                for game in week_games.to_dict("records"):
                    game_id = str(game["game_id"])
                    inputs = self._game_inputs(game, locks[game_id], context)
                    known = [
                        t
                        for t in (*inputs.published, inputs.latest_end)
                        if t is not None
                    ]
                    latest = max(known) if known else None
                    records.append(
                        {
                            "game_id": game_id,
                            "basis": (
                                InformationBasis.PER_ROW.value
                                if latest is not None
                                else InformationBasis.NO_INFORMATION.value
                            ),
                            "information_time": latest,
                        }
                    )
        return pd.DataFrame(records, columns=list(PROVENANCE_COLUMNS))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_week(
        self,
        games_df: pd.DataFrame,
        week_games: pd.DataFrame,
        season: int,
        week: int,
        lock_frame: pd.Series,
    ) -> pd.DataFrame:
        """One row per team per game for one (season, week), each at its game's lock."""
        context = self._week_context(games_df, season, week)
        rows: list[dict[str, Any]] = []
        for game in week_games.to_dict("records"):
            inputs = self._game_inputs(game, lock_frame[str(game["game_id"])], context)
            for column in ("home_team", "away_team"):
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "team": game[column],
                        "qb_adjustment": float(
                            inputs.adjustments[self._safe_normalize_team(game[column])]
                        ),
                    }
                )
        return pd.DataFrame(rows, columns=["game_id", "team", "qb_adjustment"])

    def _week_context(
        self, games_df: pd.DataFrame, season: int, week: int
    ) -> _WeekContext:
        """Everything one (season, week) reads, loaded once for all its games."""
        memo = self._context_memo.get((season, week))
        if memo is not None and memo[0] is games_df:
            return memo[1]
        depth = self._load_depth_charts(season)
        if len(depth) > 0 and "position" in depth.columns:
            depth = depth.loc[depth["position"] == "QB"]
        pbp = self._load_pbp_data(season)
        ends = self.pbp_game_end_times(pbp, games_df)
        if len(pbp) > 0:
            meta = pbp[["game_id", "season", "week"]].drop_duplicates("game_id")
            windowed = (meta["season"] < season) | (
                (meta["season"] == season) & (meta["week"] < week)
            )
            window_ids = frozenset(meta.loc[windowed, "game_id"])
        else:
            window_ids = frozenset()
        context = _WeekContext(
            season=season,
            week=week,
            depth=depth,
            pbp=pbp,
            game_ends=ends,
            window_ids=window_ids,
        )
        self._context_memo[(season, week)] = (games_df, context)
        return context

    def _game_inputs(
        self, game: Mapping[str, Any], lock: Any, context: _WeekContext
    ) -> _GameQBInputs:
        """Resolve both teams' QB and quality for one game from what its lock admits."""
        lock_utc = lock_as_utc(lock)
        memo_key = (str(game["game_id"]), lock_utc)
        if memo_key in context.inputs:
            return context.inputs[memo_key]
        admitted_ends = context.game_ends.loc[context.game_ends <= lock_utc]
        latest_end = (
            cast(pd.Timestamp, pd.Timestamp(admitted_ends.max()))
            if len(admitted_ends)
            else None
        )
        admitted_ids = frozenset(admitted_ends.index)

        # The rolling window reads only games in earlier weeks, so the rolling result
        # depends only on the admitted games INSIDE that window -- one computation per
        # distinct set rather than one per lock.
        window = admitted_ids & context.window_ids
        rolling = context.rolling.get(window)
        if rolling is None:
            rolling = self.compute_rolling_qb_metrics(
                context.pbp.loc[context.pbp["game_id"].isin(sorted(window))]
                if len(context.pbp) > 0
                else context.pbp,
                context.season,
                context.week,
            )
            context.rolling[window] = rolling

        adjustments: dict[str, float] = {}
        published: list[pd.Timestamp] = []
        for column in ("home_team", "away_team"):
            team = self._safe_normalize_team(game[column])
            qbs = self.resolve_depth_chart_qbs(
                context.depth, context.season, context.week, team, lock
            )
            if qbs.published_at is not None:
                published.append(qbs.published_at)
            qb_id = qbs.qb1
            if qb_id is None:
                qb_id = self._latest_admitted_primary_passer(
                    context, admitted_ids, team
                )
            adjustments[team] = self._quality_of(rolling, qb_id)
        inputs = _GameQBInputs(
            adjustments=adjustments, published=tuple(published), latest_end=latest_end
        )
        context.inputs[memo_key] = inputs
        return inputs

    def _latest_admitted_primary_passer(
        self, context: _WeekContext, admitted_ids: frozenset[str], team: str
    ) -> str | None:
        """The team's most recent primary passer among games FINISHED at the lock.

        Used only when no depth chart names a QB1 (87 of 12,428 team-games in 2002-2024,
        none in 2025, measured 2026-09-21). It can no longer name the season's final
        passer: it reads only play-by-play whose game ended at or before THIS game's lock,
        so a passer who first appears later -- or in this game -- is never seen.
        """
        if not admitted_ids or len(context.pbp) == 0:
            return None
        primary = context.primary.get(admitted_ids)
        if primary is None:
            primary = self.compute_per_game_qb_stats(
                context.pbp.loc[context.pbp["game_id"].isin(sorted(admitted_ids))]
            )
            if len(primary) > 0:
                primary = primary.assign(
                    _team=primary["posteam"].map(self._safe_normalize_team)
                )
            context.primary[admitted_ids] = primary
        if len(primary) == 0:
            return None
        team_passers = primary.loc[primary["_team"] == team].sort_values(
            ["season", "week"], ascending=False
        )
        if len(team_passers) == 0:
            return None
        return str(team_passers.iloc[0]["passer_player_id"])

    @staticmethod
    def _quality_of(rolling: pd.DataFrame, qb_id: str | None) -> float:
        """The QB's rolling quality, or 0.0 (league average) with no id or no history."""
        if qb_id is None or len(rolling) == 0:
            return 0.0
        match = rolling.loc[rolling["passer_player_id"] == qb_id]
        if len(match) == 0:
            return 0.0
        return float(match.iloc[0]["qb_quality"])

    @staticmethod
    def _target_games(
        games_df: pd.DataFrame, target_season: int | None, target_week: int | None
    ) -> pd.DataFrame:
        """The games a build covers: one (season, week) when both are given, else all."""
        if len(games_df) == 0:
            return games_df
        if target_season is not None and target_week is not None:
            return games_df.loc[
                (games_df["season"] == target_season)
                & (games_df["week"] == target_week)
            ]
        return games_df

    def _resolve_game(self, game_id: str) -> dict[str, Any] | None:
        """The cached games-frame row for *game_id*, or ``None``."""
        if self._games_cache is None or len(self._games_cache) == 0:
            return None
        match = self._games_cache.loc[self._games_cache["game_id"] == game_id]
        if len(match) == 0:
            return None
        return match.iloc[0].to_dict()

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
                # NO WEEK IS MANUFACTURED (Plan 33.2-13). This block used to stamp every
                # 2025+ row with a constant week, so no chart matched any target week and
                # the starter fell through to the season's final primary passer. The 2025+
                # schema has no week by design: each update carries its ISO-8601 upstream
                # publication time `dt`, parsed here ONCE (aware, never relabelled) so
                # `resolve_depth_chart_qbs` can take the latest snapshot at or before a
                # game's lock.
                if "dt" in dc.columns:
                    dc["dt"] = to_aware_utc(pd.Series(dc["dt"]), column="dt")

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
