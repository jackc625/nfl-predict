"""
Team Form Metrics Calculator

This module calculates rolling team performance metrics from play-by-play data
using a dynamic expanding window that adapts to season progress:

- Early season (Week 1-4): blends prior-season tail with current-season games
- Mid season (Week 5-8): expanding current-season window, shrinking prior-season
- Late season (Week 9+): current-season data dominates, prior-season drops off

Metrics calculated:
- EPA/play: Expected Points Added per play (overall, pass, rush) x (offense, defense)
- Success rate: Percentage of plays that increase win probability (overall, pass, rush)
- Neutral situation pass rate
- Red zone TD rate
- Third down conversion rate
- Rest days since last game
- Team-level rolling CPOE (FEAT-15): mean cpoe from non-null pass plays
- Average drive starting field position (FEAT-20): mean yardline_100 from first play per drive
- Neutral-situation pace (FEAT-21): count of neutral-situation plays per game

FEAT-16 (Win Totals Prior): SKIPPED -- no free programmatic data source
available (nflverse, nfelo). Existing Elo 75/25 season carryover (Phase 3)
serves as the calibration fallback. Per D-09 decision.

Key constraints:
- No data leakage: a target game's window admits a prior team-game only when that
  game's RESULT existed at the target's OWN lock -- its END (kickoff plus
  ``features.provenance.DECLARED_GAME_DURATION``) at or before 18:00 ET on the
  calendar day before the target's kickoff (``utils.game_lock``, Plan 33.2-14). On an
  ordinary schedule that is exactly "the games before Week N"; the lock-keyed form is
  the rule itself and also excludes a rescheduled game played after the lock.
- Recency weighting: linear weights [1, 2, ..., N] where more recent = higher weight
- Canonical team abbreviations: LA is Rams (not LAR), per Phase 2 decision
"""

import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from conf.season_partition import SELECTION_WINDOW_FIRST_SEASON
from conf.settings import get_settings
from data import upstream_pin
from data.storage import load_dataframe, save_dataframe
from features.provenance import (
    DECLARED_GAME_DURATION,
    PROVENANCE_COLUMNS,
    InformationBasis,
)
from features.qb_tracking import to_aware_utc
from utils import get_logger

logger = get_logger(__name__)

#: The twelve rolling metrics one team side carries (the ``rolling_*`` names
#: ``calculate_rolling_averages`` writes). Gold reads them as
#: ``{home,away}_{off,def}_rolling_*`` through ``scripts/build_features``.
ROLLING_COLUMNS: tuple[str, ...] = (
    "rolling_epa_per_play",
    "rolling_pass_epa_per_play",
    "rolling_rush_epa_per_play",
    "rolling_success_rate",
    "rolling_pass_success_rate",
    "rolling_rush_success_rate",
    "rolling_neutral_pass_rate",
    "rolling_red_zone_td_rate",
    "rolling_third_down_conversion_rate",
    "rolling_cpoe",
    "rolling_avg_drive_start_yardline",
    "rolling_neutral_pace",
)

#: THE METRICS THE PINNED PLAY-BY-PLAY DOES NOT CARRY IN EVERY SEASON (Plan 33.2-17 Task 2).
#: Measured 2026-09-22 on silver team form rebuilt from the pin back to 2002: ``rolling_cpoe``
#: is NULL in every 2002-2005 row -- the play-by-play carries no completion probability before
#: 2006 -- and every other rolling metric is populated from 2002. The gold build gives each
#: offensive copy a ``*_coverage`` flag (1.0 where the value was computed) and keeps the
#: unmeasured value NaN, never the neutral 0.0 a model would read as "exactly average". The
#: metric is offense-only, so its defensive copy is never populated and is not flagged. Which
#: seasons lack it is read from the data, never stated as a literal.
SOURCE_LIMITED_ROLLING_COLUMNS: tuple[str, ...] = ("rolling_cpoe",)


def source_limited_gold_columns() -> tuple[tuple[str, str], ...]:
    """``(value column, coverage flag)`` in gold for each source-limited metric, per side."""
    return tuple(
        (f"{prefix}_off_{column}", f"{prefix}_off_{column}_coverage")
        for prefix in ("home", "away")
        for column in SOURCE_LIMITED_ROLLING_COLUMNS
    )


#: The team-schedule frame's columns (``team_game_schedule``).
TEAM_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "schedule_game_id",
    "_end",
    "_lock",
)


def team_game_schedule(games_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team): the game's id, when its RESULT existed, its lock.

    THE ONE TIMING of a team-game for every lock-keyed rolling window (team form and
    snaps, D-12). ``_end`` is kickoff plus ``features.provenance.DECLARED_GAME_DURATION``
    -- the same four hours the Elo replay declares -- and ``_lock`` is the game's own
    lock from the one rule, ``utils.game_lock.lock_frame``. A scheduled game with no
    kickoff has neither and is left out: a team-game that cannot be timed is never
    admitted. A naive kickoff is refused by name (``to_aware_utc``), never relabelled.

    Args:
        games_df: Silver ``games`` shape: ``game_id``, ``season``, ``week``,
            ``home_team``, ``away_team`` (canonical abbreviations) and a tz-aware
            ``kickoff_et``.

    Returns:
        A frame with exactly ``TEAM_SCHEDULE_COLUMNS``.
    """
    games = games_df.loc[games_df["kickoff_et"].notna()]
    if len(games) == 0:
        return pd.DataFrame(columns=list(TEAM_SCHEDULE_COLUMNS))
    locks = lock_rule.lock_frame(games)
    ends = (
        to_aware_utc(pd.Series(games["kickoff_et"]), column="kickoff_et")
        + DECLARED_GAME_DURATION
    )
    sides = []
    for column in ("home_team", "away_team"):
        side = pd.DataFrame(
            {
                "season": games["season"].astype(int).to_numpy(),
                "week": games["week"].astype(int).to_numpy(),
                "team": games[column].astype(str).to_numpy(),
                "schedule_game_id": games["game_id"].astype(str).to_numpy(),
                "_end": ends.to_numpy(),
            }
        )
        side["_lock"] = side["schedule_game_id"].map(locks.to_dict())
        sides.append(side)
    return pd.concat(sides, ignore_index=True)[list(TEAM_SCHEDULE_COLUMNS)]


def week_team_locks(
    schedule: pd.DataFrame, target_season: int, target_week: int
) -> dict[str, Any]:
    """``team -> lock`` of each team's scheduled game in (season, week), from a schedule.

    A team with no game that week (a bye) has no lock and is absent: nothing is admitted
    for a game that does not exist.
    """
    week = schedule.loc[
        (schedule["season"] == target_season) & (schedule["week"] == target_week)
    ]
    return dict(zip(week["team"], week["_lock"], strict=True))


# ---------------------------------------------------------------------------
# THE PER-GAME SEASON POOL (Plan 33.1-07 Task 4; reworded by Plan 33.2-17 Task 1).
#
# WHAT WAS WRONG ONCE. `get_per_game_stats` resolved its full-build pool as a hardcoded
# `range(...)` whose exclusive upper bound silently stopped one season short, so the live
# season's twelve opponent-adjusted columns carried two distinct values -- the imputation that
# runs when the real column is absent, wearing the column's name. The UPPER bound is therefore
# DERIVED from the seasons the data carries (see `per_game_seasons`), never typed.
#
# THE FLOOR IS THE RULE MODULE'S, BY IDENTITY. It is whatever
# `conf.season_partition.SELECTION_WINDOW_FIRST_SEASON` says, and the reason it sits where it
# does -- and which coverage floors it once stood in for -- is recorded in that module's
# `RULE_EVIDENCE`, not here. This binding exists so the two cannot drift: a decision to move
# the selection window moves this pool with it, and nothing in this file restates a season or
# a column count that could go stale when it does.
# ---------------------------------------------------------------------------

TEAM_FORM_PER_GAME_FIRST_SEASON: int = SELECTION_WINDOW_FIRST_SEASON


class TeamFormCalculator:
    """Calculate rolling team form metrics from play-by-play data.

    Uses a dynamic expanding window: early in the season the window leans
    on prior-season games; as the current season progresses, those prior-
    season games are replaced by current-season data.

    Features calculated (12 metrics total -- original 9 + 3 PBP-derived):
    - Offensive EPA/play (overall, pass, rush)
    - Defensive EPA/play (overall, pass, rush)
    - Offensive/Defensive success rate (overall, pass, rush)
    - Neutral situation pass rate
    - Red zone TD rate
    - Third down conversion rate
    - Rest days since last game
    - Team-level rolling CPOE (FEAT-15)
    - Average drive starting field position (FEAT-20)
    - Neutral-situation pace of play (FEAT-21)

    FEAT-16 (Win Totals Prior): SKIPPED -- no free programmatic data source
    available (nflverse, nfelo). Existing Elo 75/25 season carryover (Phase 3)
    serves as the calibration fallback. Per D-09 decision.
    """

    def __init__(
        self,
        max_prior_games: int = 8,
        schedule_df: pd.DataFrame | None = None,
        form_df: pd.DataFrame | None = None,
    ):
        """Initialize team form calculator.

        Args:
            max_prior_games: Maximum number of prior-season games to include
                in the window when current-season data is sparse. As current-
                season games accumulate, prior-season games are shed:
                prior_count = max(0, max_prior_games - current_season_count).
            schedule_df: Optional schedule (silver ``games`` shape) team-games are
                TIMED against and target games LOCKED from. When ``None`` the
                calculator loads silver ``games``. The test-injection seam.
            form_df: Optional rolling team-form table (silver
                ``team_form_features`` shape) the provenance reads. When ``None``
                the calculator loads silver ``team_form_features``.
        """
        self.max_prior_games = max_prior_games
        self.settings = get_settings()
        self._schedule_df = schedule_df
        self._form_df = form_df
        self._team_schedule_cache: pd.DataFrame | None = None

        # Team name mapping for consistency with our game data
        self.team_mapping = self._build_team_mapping()

    def _build_team_mapping(self) -> dict[str, str]:
        """Build mapping from nflreadpy team names to our canonical abbreviations.

        Phase 2 established LA as the canonical Rams abbreviation (not LAR).
        nflreadpy uses LA for Rams, which matches our canonical form, so no
        mapping is needed for the Rams.
        """
        return {
            # LA is canonical for Rams -- no mapping needed (identity)
            "LV": "LV",  # Las Vegas Raiders
            "GB": "GB",  # Green Bay Packers
            # Add any other mappings needed
        }

    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to our canonical format."""
        if not team or pd.isna(team):
            return team
        team_upper = str(team).upper().strip()
        return self.team_mapping.get(team_upper, team_upper)

    def fetch_pbp_data(self, seasons: list[int]) -> pd.DataFrame:
        """Fetch play-by-play data for specified seasons.

        Args:
            seasons: List of seasons to fetch

        Returns:
            DataFrame with play-by-play data
        """
        try:
            logger.info("Fetching play-by-play data", seasons=seasons)

            # Read the PINNED play-by-play snapshot (data/upstream_pin.py). This used
            # to be a live ``nfl.load_pbp(seasons).to_pandas()`` with no cache, which is
            # how an nflverse re-release moved twelve opponent-adjusted columns from
            # season 2020 onward between two Phase-31 gold builds. The loader REFUSES
            # rather than falling back to the network when a season is not pinned.
            pbp_df = upstream_pin.load_pbp(seasons)

            # Normalize team names
            pbp_df["posteam"] = pbp_df["posteam"].apply(self._normalize_team_name)
            pbp_df["defteam"] = pbp_df["defteam"].apply(self._normalize_team_name)

            # Filter to regular season only (weeks 1-18)
            pbp_df = pbp_df[pbp_df["week"].between(1, 18)]

            # Filter to meaningful plays (exclude special teams, penalties, etc.)
            meaningful_plays = pbp_df[
                (pbp_df["play_type"].isin(["pass", "run"]))
                & (pbp_df["posteam"].notna())
                & (pbp_df["defteam"].notna())
                & (pbp_df["epa"].notna())
            ].copy()

            logger.info(
                "Fetched play-by-play data",
                total_plays=len(pbp_df),
                meaningful_plays=len(meaningful_plays),
                seasons=seasons,
            )

            return meaningful_plays

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(
                "Failed to fetch play-by-play data", seasons=seasons, error=str(e)
            )
            raise

    def _identify_neutral_situations(self, pbp_df: pd.DataFrame) -> pd.Series:
        """Identify neutral game situations for pass rate analysis.

        Neutral situation criteria:
        - Down 1 or 2
        - 5+ yards to go
        - Not in red zone (>20 yards from goal)
        - Score differential within 14 points
        - Not in final 2 minutes of half

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            Boolean series indicating neutral situations
        """
        return (
            (pbp_df["down"].isin([1, 2]))
            & (pbp_df["ydstogo"] >= 5)
            & (pbp_df["yardline_100"] > 20)
            & (pbp_df["score_differential"].abs() <= 14)
            & (pbp_df["half_seconds_remaining"] > 120)
        )

    def calculate_team_game_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team-level statistics for each game from play-by-play data.

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            DataFrame with team-game level statistics
        """
        logger.info("Calculating team-game statistics", plays=len(pbp_df))

        # Add neutral situation indicator
        pbp_df = pbp_df.copy()
        pbp_df["neutral_situation"] = self._identify_neutral_situations(pbp_df)

        # Separate offensive and defensive stats
        offense_stats = []
        defense_stats = []

        # Group by game and team (offensive perspective)
        for (game_id, season, week, team), group in pbp_df.groupby(
            ["game_id", "season", "week", "posteam"]
        ):
            if team is None or pd.isna(team):
                continue

            plays = len(group)
            if plays == 0:
                continue

            # Basic EPA metrics
            total_epa = group["epa"].sum()
            epa_per_play = group["epa"].mean()

            # Pass vs Run EPA
            pass_plays = group[group["play_type"] == "pass"]
            rush_plays = group[group["play_type"] == "run"]

            pass_epa_per_play = pass_plays["epa"].mean() if len(pass_plays) > 0 else 0
            rush_epa_per_play = rush_plays["epa"].mean() if len(rush_plays) > 0 else 0

            # Success rate
            success_rate = group["success"].mean()
            pass_success_rate = (
                pass_plays["success"].mean() if len(pass_plays) > 0 else 0
            )
            rush_success_rate = (
                rush_plays["success"].mean() if len(rush_plays) > 0 else 0
            )

            # Neutral situation metrics
            neutral_plays = group[group["neutral_situation"]]
            neutral_pass_rate = 0
            if len(neutral_plays) > 0:
                neutral_pass_rate = (neutral_plays["play_type"] == "pass").mean()

            # Red zone metrics (within 20 yards of goal)
            red_zone_plays = group[group["yardline_100"] <= 20]
            red_zone_td_rate = 0
            if len(red_zone_plays) > 0:
                red_zone_td_rate = red_zone_plays["touchdown"].mean()

            # Third down metrics
            third_down_plays = group[group["down"] == 3]
            third_down_conversion_rate = 0
            if len(third_down_plays) > 0:
                third_down_conversion_rate = third_down_plays["first_down"].mean()

            # FEAT-15: Team-level CPOE -- mean of cpoe where cpoe is not null
            # Sacks, scrambles, and spikes have null cpoe; filter them out
            team_cpoe = np.nan
            if "cpoe" in group.columns:
                valid_cpoe = group[group["cpoe"].notna()]["cpoe"]
                if len(valid_cpoe) > 0:
                    team_cpoe = valid_cpoe.mean()

            # FEAT-20: Average drive start yard line -- yardline_100 from
            # first play of each fixed_drive (offense only)
            # Use yardline_100 (numeric, 0-100) NOT drive_start_yard_line (string)
            avg_drive_start_yardline = np.nan
            if "fixed_drive" in group.columns:
                drive_first_plays = group.groupby("fixed_drive").first()
                if (
                    len(drive_first_plays) > 0
                    and "yardline_100" in drive_first_plays.columns
                ):
                    avg_drive_start_yardline = drive_first_plays["yardline_100"].mean()

            # FEAT-21: Neutral-situation pace -- count of neutral-situation plays
            # Uses the neutral_situation column computed by _identify_neutral_situations()
            neutral_pace = 0
            if "neutral_situation" in group.columns:
                neutral_pace = int(group["neutral_situation"].sum())

            offense_stats.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "side": "offense",
                    "plays": plays,
                    "total_epa": total_epa,
                    "epa_per_play": epa_per_play,
                    "pass_epa_per_play": pass_epa_per_play,
                    "rush_epa_per_play": rush_epa_per_play,
                    "success_rate": success_rate,
                    "pass_success_rate": pass_success_rate,
                    "rush_success_rate": rush_success_rate,
                    "neutral_pass_rate": neutral_pass_rate,
                    "red_zone_td_rate": red_zone_td_rate,
                    "third_down_conversion_rate": third_down_conversion_rate,
                    "pass_attempts": len(pass_plays),
                    "rush_attempts": len(rush_plays),
                    "team_cpoe": team_cpoe,
                    "avg_drive_start_yardline": avg_drive_start_yardline,
                    "neutral_pace": neutral_pace,
                }
            )

        # Group by game and team (defensive perspective)
        for (game_id, season, week, team), group in pbp_df.groupby(
            ["game_id", "season", "week", "defteam"]
        ):
            if team is None or pd.isna(team):
                continue

            plays = len(group)
            if plays == 0:
                continue

            # Defensive EPA (opponent's EPA against this defense)
            total_epa_allowed = group["epa"].sum()
            epa_per_play_allowed = group["epa"].mean()

            # Pass vs Run EPA allowed
            pass_plays = group[group["play_type"] == "pass"]
            rush_plays = group[group["play_type"] == "run"]

            pass_epa_per_play_allowed = (
                pass_plays["epa"].mean() if len(pass_plays) > 0 else 0
            )
            rush_epa_per_play_allowed = (
                rush_plays["epa"].mean() if len(rush_plays) > 0 else 0
            )

            # Success rate allowed
            success_rate_allowed = group["success"].mean()
            pass_success_rate_allowed = (
                pass_plays["success"].mean() if len(pass_plays) > 0 else 0
            )
            rush_success_rate_allowed = (
                rush_plays["success"].mean() if len(rush_plays) > 0 else 0
            )

            # Red zone defense
            red_zone_plays = group[group["yardline_100"] <= 20]
            red_zone_td_rate_allowed = 0
            if len(red_zone_plays) > 0:
                red_zone_td_rate_allowed = red_zone_plays["touchdown"].mean()

            # Third down defense
            third_down_plays = group[group["down"] == 3]
            third_down_conversion_rate_allowed = 0
            if len(third_down_plays) > 0:
                third_down_conversion_rate_allowed = third_down_plays[
                    "first_down"
                ].mean()

            defense_stats.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "side": "defense",
                    "plays": plays,
                    "total_epa": total_epa_allowed,
                    "epa_per_play": epa_per_play_allowed,
                    "pass_epa_per_play": pass_epa_per_play_allowed,
                    "rush_epa_per_play": rush_epa_per_play_allowed,
                    "success_rate": success_rate_allowed,
                    "pass_success_rate": pass_success_rate_allowed,
                    "rush_success_rate": rush_success_rate_allowed,
                    "neutral_pass_rate": np.nan,  # Not applicable for defense
                    "red_zone_td_rate": red_zone_td_rate_allowed,
                    "third_down_conversion_rate": third_down_conversion_rate_allowed,
                    "pass_attempts": len(pass_plays),
                    "rush_attempts": len(rush_plays),
                    "team_cpoe": np.nan,  # Offense-only metric
                    "avg_drive_start_yardline": np.nan,  # Offense-only metric
                    "neutral_pace": np.nan,  # Offense-only metric
                }
            )

        # Combine offense and defense stats
        all_stats = offense_stats + defense_stats
        stats_df = pd.DataFrame(all_stats)

        logger.info(
            "Calculated team-game statistics",
            total_records=len(stats_df),
            offense_records=len(offense_stats),
            defense_records=len(defense_stats),
        )

        return stats_df

    def _select_dynamic_window(
        self,
        team_group: pd.DataFrame,
        target_season: int,
        target_week: int,
    ) -> pd.DataFrame:
        """Select games for the dynamic expanding window.

        The window blends prior-season and current-season data:
        - current_games: all games in target_season with week < target_week
        - prior_games: last N games of (target_season - 1), where
          N = max(0, max_prior_games - len(current_games))
        - Combined: prior_games.tail(N) + current_games (all of them)

        This means:
        - Week 1: 8 prior-season games, 0 current (100% prior)
        - Week 5: 4 prior + 4 current (50/50)
        - Week 9+: 0 prior + all current (current dominates)

        Args:
            team_group: All historical games for one team+side, sorted
                chronologically and already filtered to < target week.
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            DataFrame subset with the games to include in the window.
        """
        current_season_games = team_group[team_group["season"] == target_season]
        prior_season_games = team_group[team_group["season"] == target_season - 1]

        current_count = len(current_season_games)
        prior_count = max(0, self.max_prior_games - current_count)

        # Take last prior_count games from prior season
        prior_tail = prior_season_games.tail(prior_count)

        # Combine: prior tail + all current season games
        combined = pd.concat([prior_tail, current_season_games])
        return combined.sort_values(["season", "week"])

    def _team_schedule(self) -> pd.DataFrame:
        """The timing schedule (``team_game_schedule``), built once per calculator."""
        if self._team_schedule_cache is None:
            games = (
                self._schedule_df
                if self._schedule_df is not None
                else load_dataframe("games", layer="silver")
            )
            self._team_schedule_cache = team_game_schedule(games)
        return self._team_schedule_cache

    @staticmethod
    def time_team_games(
        team_games: pd.DataFrame, schedule: pd.DataFrame
    ) -> pd.DataFrame:
        """*team_games* with ``_end``: its scheduled game's end, keyed (season, week, team).

        Play-by-play ids (``2024_01_BUF_KC``) and silver ids never match, so a team-game
        is keyed to its scheduled game by (season, week, team). One with no scheduled
        match has no known end (NaT) and is therefore never admitted.
        """
        timing = schedule[["season", "week", "team", "_end"]]
        return team_games.drop(columns=["_end"], errors="ignore").merge(
            timing, on=["season", "week", "team"], how="left"
        )

    def admitted_window(
        self,
        team_games: pd.DataFrame,
        target_season: int,
        target_week: int,
        lock: Any,
    ) -> pd.DataFrame:
        """ONE team's (and side's) rolling window for a target game, at that game's LOCK.

        A team-game is admitted only when its END (``_end``) is at or before *lock*,
        the target game's own lock (at-lock admissible, the ``<=`` of
        ``utils.game_lock.is_admissible``); the admitted games then go through the
        dynamic window. THE ONE SELECTION the rolling averages and the provenance both
        read.

        WHY LOCK-KEYED AND NOT WEEK-KEYED (Plan 33.2-14). The retired window read the
        WEEK LABEL: ``season < target_season`` or the same season with ``week <
        target_week``. The window is per team and a team plays at most once a week, so
        on an ordinary schedule the two select the same games -- D33.2-01 measured 0
        games in 2002-2026 whose week-keyed inputs include a result that ended after
        their lock. The lock-keyed window IS the admissibility rule rather than a proxy
        that happens to agree with it, and it is the only form that excludes a game
        whose week label precedes the target week but which was PLAYED after the
        target's lock -- a rescheduled or postponed game.

        Args:
            team_games: One team's (and side's) timed team-games (``_end``).
            target_season: Season being predicted.
            target_week: Week being predicted.
            lock: The target game's lock, tz-aware.

        Returns:
            The window rows (possibly empty), sorted chronologically.
        """
        admitted = cast(
            "pd.DataFrame", team_games[team_games["_end"] <= lock]
        ).sort_values(["season", "week"])
        if len(admitted) == 0:
            return admitted
        return self._select_dynamic_window(admitted, target_season, target_week)

    def contributing_games(
        self,
        team_games: pd.DataFrame,
        target_season: int,
        target_week: int,
        team_locks: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Every team-game the rolling window admits for the week's target games.

        The team-form counterpart of ``features.snaps.SnapCountBuilder.
        contributing_games``: the window each team actually reads at its own target
        game's lock, concatenated. Grouped by (team, side) when the frame carries a
        ``side``, by team otherwise.

        Args:
            team_games: Timed team-games (``_end``) for any number of teams.
            target_season: Season being predicted.
            target_week: Week being predicted.
            team_locks: ``team -> lock`` for the week's target games.

        Returns:
            The admitted rows. Empty when nothing was admitted.
        """
        keys = ["team", "side"] if "side" in team_games.columns else ["team"]
        windows = [
            window
            for group_key, group in team_games.groupby(keys, sort=True)
            if (lock := team_locks.get(str(cast("tuple", group_key)[0]))) is not None
            and len(
                window := self.admitted_window(group, target_season, target_week, lock)
            )
            > 0
        ]
        if not windows:
            return team_games.iloc[0:0]
        return pd.concat(windows, ignore_index=True)

    def calculate_rolling_averages(
        self,
        team_stats_df: pd.DataFrame,
        target_season: int,
        target_week: int,
        *,
        schedule: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculate rolling averages using a dynamic expanding window.

        The window adapts to season progress:
        - Early season: more prior-season data for stability
        - Late season: dominated by current-season data for responsiveness

        Recency weighting is applied via linear weights [1, 2, ..., N]
        where N = number of games in the window.

        EACH TEAM'S WINDOW IS ADMITTED AT ITS TARGET GAME'S LOCK
        (:meth:`admitted_window`, Plan 33.2-14), so a team with no game in
        (target_season, target_week) -- a bye -- has no lock and no row.

        Args:
            team_stats_df: Team-game statistics DataFrame. Timed against *schedule*
                unless it already carries ``_end``.
            target_season: Season to calculate rolling averages for
            target_week: Week to calculate rolling averages for
            schedule: The timing schedule (``team_game_schedule``). When ``None``
                the calculator's own (injected, else silver ``games``) is used.

        Returns:
            DataFrame with rolling averages for each team
        """
        logger.info(
            "Calculating rolling averages",
            target_season=target_season,
            target_week=target_week,
            max_prior_games=self.max_prior_games,
        )

        schedule = schedule if schedule is not None else self._team_schedule()
        timed = (
            team_stats_df
            if "_end" in team_stats_df.columns
            else self.time_team_games(team_stats_df, schedule)
        )
        team_locks = week_team_locks(schedule, target_season, target_week)

        if len(timed) == 0 or not team_locks:
            logger.warning("No historical games found for rolling averages")
            return pd.DataFrame()

        rolling_stats = []

        # Calculate rolling averages for each team and side
        for (team, side), group in timed.groupby(["team", "side"]):
            lock = team_locks.get(str(team))
            if lock is None:
                continue
            recent_games = self.admitted_window(group, target_season, target_week, lock)

            if len(recent_games) == 0:
                continue

            # Calculate weighted averages (more recent games weighted higher)
            weights = np.arange(1, len(recent_games) + 1)
            weights = weights / weights.sum()

            # Calculate weighted averages for key metrics
            # Original 9 metrics + 3 new PBP-derived metrics (FEAT-15, 20, 21)
            # Offense-only metrics are set to NaN for the defense side
            offense_only_metrics = {
                "neutral_pass_rate",
                "team_cpoe",
                "avg_drive_start_yardline",
                "neutral_pace",
            }

            # Build metrics list dynamically: original 9 + new 3
            all_metrics = [
                "epa_per_play",
                "pass_epa_per_play",
                "rush_epa_per_play",
                "success_rate",
                "pass_success_rate",
                "rush_success_rate",
                "neutral_pass_rate",
                "red_zone_td_rate",
                "third_down_conversion_rate",
                "team_cpoe",
                "avg_drive_start_yardline",
                "neutral_pace",
            ]

            avg_stats = {}
            for metric in all_metrics:
                # Use custom rolling column names for new metrics
                if metric == "team_cpoe":
                    rolling_name = "rolling_cpoe"
                elif metric == "avg_drive_start_yardline":
                    rolling_name = "rolling_avg_drive_start_yardline"
                elif metric == "neutral_pace":
                    rolling_name = "rolling_neutral_pace"
                else:
                    rolling_name = f"rolling_{metric}"

                # Skip offense-only metrics for defense side
                if metric in offense_only_metrics and side == "defense":
                    avg_stats[rolling_name] = np.nan
                elif metric not in recent_games.columns:
                    # Graceful handling when column is missing (backward compat)
                    avg_stats[rolling_name] = np.nan
                else:
                    values = recent_games[metric].values
                    # Handle NaN values
                    valid_mask = ~np.isnan(values.astype(float))
                    if valid_mask.sum() > 0:
                        valid_values = values[valid_mask]
                        valid_weights = weights[valid_mask]
                        valid_weights = valid_weights / valid_weights.sum()
                        avg_stats[rolling_name] = np.average(
                            valid_values, weights=valid_weights
                        )
                    else:
                        avg_stats[rolling_name] = np.nan

            rolling_stats.append(
                {
                    "team": team,
                    "side": side,
                    "target_season": target_season,
                    "target_week": target_week,
                    "games_used": len(recent_games),
                    "weeks_span": (
                        recent_games["season"].max() - recent_games["season"].min()
                    )
                    * 18
                    + (recent_games["week"].max() - recent_games["week"].min()),
                    **avg_stats,
                }
            )

        rolling_df = pd.DataFrame(rolling_stats)

        logger.info(
            "Calculated rolling averages",
            teams=len(rolling_df["team"].unique()) if len(rolling_df) > 0 else 0,
            records=len(rolling_df),
        )

        return rolling_df

    def calculate_rest_days(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest days for each team since their last game.

        Args:
            games_df: Games DataFrame with kickoff times

        Returns:
            DataFrame with rest days added
        """
        games_with_rest = []

        for team in games_df[["home_team", "away_team"]].values.flatten():
            if pd.isna(team):
                continue

            # Get all games for this team, sorted chronologically
            team_games = (
                games_df[
                    (games_df["home_team"] == team) | (games_df["away_team"] == team)
                ]
                .sort_values("kickoff_et")
                .copy()
            )

            team_games["rest_days"] = np.nan

            for i in range(1, len(team_games)):
                current_game = team_games.iloc[i]
                previous_game = team_games.iloc[i - 1]

                rest_days = (
                    current_game["kickoff_et"] - previous_game["kickoff_et"]
                ).days
                team_games.iloc[i, team_games.columns.get_loc("rest_days")] = rest_days

            # Set first game of season to standard rest (7 days)
            if len(team_games) > 0:
                team_games.iloc[0, team_games.columns.get_loc("rest_days")] = 7

            games_with_rest.append(team_games)

        if games_with_rest:
            result_df = pd.concat(games_with_rest).drop_duplicates(subset=["game_id"])
            return result_df.sort_values(["season", "week", "kickoff_et"])
        return games_df.copy()

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
        """Build team form features using only data available before as_of_datetime.

        Conforms to the FeatureBuilder Protocol. Each team's window is admitted
        at its target game's own lock (:meth:`admitted_window`, Plan 33.2-14).

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Carried for the ``FeatureBuilder`` Protocol ONLY. It is
                NOT a fence and no selection reads it.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.

        Returns:
            DataFrame with team form rolling averages.
        """
        logger.info(
            "Building team form features (Protocol)",
            target_season=target_season,
            target_week=target_week,
        )

        # Determine seasons to include
        if target_season is not None:
            seasons = [target_season - 1, target_season]
        else:
            all_seasons = sorted(games_df["season"].unique())
            seasons = list(all_seasons)
            if len(all_seasons) > 0:
                prior = all_seasons[0] - 1
                seasons = [prior, *seasons]

        # Fetch play-by-play data and compute team game stats, each team-game TIMED
        # (its END) against the schedule once; every window below is admitted at its
        # target game's lock. There is no frame-wide ``as_of_datetime`` fence: the
        # ``now`` it carries in production admits everything already played.
        pbp_df = self.fetch_pbp_data(seasons)
        schedule = self._team_schedule()
        team_stats_df = self.time_team_games(
            self.calculate_team_game_stats(pbp_df), schedule
        )

        # Calculate rolling averages for the target
        if target_season and target_week:
            return self.calculate_rolling_averages(
                team_stats_df, target_season, target_week, schedule=schedule
            )

        # Fall back to all weeks in all seasons
        all_rolling_stats = []
        for season in seasons:
            for week in range(1, 19):
                rolling_df = self.calculate_rolling_averages(
                    team_stats_df, season, week, schedule=schedule
                )
                if len(rolling_df) > 0:
                    all_rolling_stats.append(rolling_df)

        if all_rolling_stats:
            return pd.concat(all_rolling_stats, ignore_index=True)
        return pd.DataFrame()

    def per_game_seasons(self, covered_seasons: object) -> list[int]:
        """The per-game pool for *covered_seasons*, floored and sorted.

        The FLOOR is ``TEAM_FORM_PER_GAME_FIRST_SEASON`` and the CEILING is
        whatever the caller's data reaches. A season below the floor is
        DROPPED rather than fetched: widening the pool downward is the rule
        module's decision, carried here by the identity binding (see the
        constant's comment), never a per-call choice.

        THERE IS DELIBERATELY NO UPPER BOUND, and code review WR-07 asked for
        one, so the reasoning is recorded here rather than left implicit. A
        ceiling at the most recent COMPLETED season would drop the LIVE season
        from the per-game pool -- which is precisely the defect Plan 33.1-07
        Task 4 removed, when a hardcoded ``range(2018, 2025)`` stopped at 2024
        and left season 2025's twelve opponent-adjusted columns carrying two
        distinct values across 285 games. Deriving that same ceiling from a
        constant would make it the same bug with a better provenance.

        WHAT HAPPENS INSTEAD, stated so the boundary is not a surprise: a
        season above ``data.upstream_pin.SEALED_THROUGH_SEASON`` is served from
        the LIVE zone, and if that season has not been captured the pin refuses
        by name (``UpstreamLiveCaptureMissing`` / ``UpstreamPinMissing``, which
        are not in ``scripts/build_features``' swallow tuple, so they propagate
        and stop the build). A loud stop naming the uncaptured season is the
        correct outcome: it is recoverable by running the capture, whereas a
        silent ceiling produces a full-looking gold matrix with the live
        season's opponent-adjusted family imputed and nothing to notice it.

        Args:
            covered_seasons: Any iterable of season labels -- a games frame's
                ``season`` column, a list of integers, anything sortable to
                integers. Duplicates are collapsed, which is what lets a games
                frame be passed straight in.

        Returns:
            The sorted, de-duplicated seasons at or above the floor.
        """
        return sorted(
            {
                int(season)
                for season in covered_seasons
                if int(season) >= TEAM_FORM_PER_GAME_FIRST_SEASON
            }
        )

    def _covered_seasons_from_silver(self) -> list[int]:
        """The seasons silver ``games`` carries, for a caller that named none.

        FAIL-CLOSED, and typed as ``RuntimeError`` on purpose.
        ``scripts/build_features`` guards its ``get_per_game_stats`` call with
        ``except (ValueError, KeyError, TypeError, AttributeError)`` and falls
        back to an EMPTY per-game frame, which silently drops the entire
        opponent-adjusted family from gold. A refusal typed as any of those four
        would be swallowed into exactly the shape this fix exists to remove: a
        full-looking build with a family quietly missing. ``RuntimeError`` is
        outside that tuple and propagates.

        Raises:
            RuntimeError: silver ``games`` cannot be read, or carries no season.
        """
        # THE COLUMN READ IS INSIDE THE TRY (code review WR-07). It used to sit
        # after it, so a `games` frame that LOADED but carried no `season`
        # column raised a bare KeyError -- which IS in the caller's
        # `except (ValueError, KeyError, TypeError, AttributeError)` tuple and
        # was therefore swallowed into a warning plus an empty per-game frame,
        # silently dropping the whole opponent-adjusted family from gold. That
        # is the exact outcome the RuntimeError typing exists to prevent, so
        # the one read that could produce it belonged in here all along.
        try:
            games = load_dataframe("games", "silver", "parquet")
            raw_seasons = games["season"].dropna().tolist()
        except Exception as error:
            msg = (
                "cannot resolve the per-game season pool: silver `games` could "
                f"not be read, or carries no `season` column ({error!r}). The "
                "pool used to be the hardcoded range(2018, 2025), which "
                "silently stopped at 2024 and left season 2025's "
                "opponent-adjusted columns imputed. Refusing rather than "
                "guessing a range. Pass `seasons=` explicitly if the caller "
                "already holds a games frame."
            )
            raise RuntimeError(msg) from error

        seasons = self.per_game_seasons(raw_seasons)
        if not seasons:
            msg = (
                "silver `games` carries no season at or above "
                f"{TEAM_FORM_PER_GAME_FIRST_SEASON}, so the per-game pool would "
                "be empty and every opponent-adjusted column would be imputed. "
                "That is the defect this resolution replaced, so it refuses "
                "rather than returning an empty frame."
            )
            raise RuntimeError(msg)
        return seasons

    def get_per_game_stats(
        self,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        seasons: object = None,
    ) -> pd.DataFrame:
        """Get per-game team stats (not rolling) for opponent adjustment.

        Loads PBP data and computes per-game stats with game_id, team,
        season, week, side, and raw EPA metrics. Used by OpponentAdjuster
        as input for opponent strength adjustment.

        THE POOL IS DERIVED, NEVER TYPED (Plan 33.1-07 Task 4). The three
        branches below are ordered by how much the caller knows:

        1. *seasons* -- the caller holds the games frame and says what it
           covers. ``scripts/build_features`` reads ``feature_sources["games"]``
           two lines before this call, so re-deriving the same fact from a
           second store read would be a second source of truth for it.
        2. *target_season* -- a scoped build. Byte-unchanged: the target and its
           predecessor, exactly as before.
        3. Neither -- read the coverage off silver ``games``. This is the branch
           that replaced ``list(range(2018, 2025))``.

        Args:
            as_of_datetime: Time-fence cutoff.
            target_season: If set, loads this season and prior.
            seasons: The seasons the caller's data covers. Floored at
                ``TEAM_FORM_PER_GAME_FIRST_SEASON``. Takes precedence over
                *target_season*, because a caller that names its own coverage
                has more information than a season label does.

        Returns:
            DataFrame with per-game team stats.
        """
        if seasons is not None:
            resolved = self.per_game_seasons(seasons)
        elif target_season is not None:
            resolved = [target_season - 1, target_season]
        else:
            resolved = self._covered_seasons_from_silver()

        pbp_df = self.fetch_pbp_data(resolved)
        team_stats_df = self.calculate_team_game_stats(pbp_df)
        return team_stats_df

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get team form features for a single game.

        Conforms to the FeatureBuilder Protocol. Wraps the existing
        get_team_form_for_game with as_of_datetime enforcement.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            # Load team form features
            form_df = load_dataframe("team_form_features", layer="silver")

            # Parse game_id to extract teams and week info
            # Expected format: {prefix}_{season}_W{week}_{away}@{home}
            parts = game_id.split("_")
            if len(parts) < 4:
                logger.warning(
                    "Cannot parse game_id for team form lookup",
                    game_id=game_id,
                )
                return {}

            season = int(parts[1])
            week_str = parts[2]
            week = int(week_str.replace("W", ""))

            # Filter by season/week
            week_form = form_df[
                (form_df["target_season"] == season) & (form_df["target_week"] == week)
            ]

            result: dict[str, float] = {}
            for _, row in week_form.iterrows():
                team = row["team"]
                side = row["side"]
                prefix = f"{team}_{side[:3]}"
                for col in row.index:
                    if col.startswith("rolling_"):
                        result[f"{prefix}_{col}"] = float(row[col])

            return result

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(
                "Failed to get features for game",
                game_id=game_id,
                error=str(e),
            )
            return {}

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game neither of whose teams carries a rolling row: every column NULL.

        The ``team_form`` source frame is one row per game with the
        ``{home,away}_{off,def}_rolling_*`` columns ``scripts/build_features`` lays out
        from silver ``team_form_features``; a game with no rolling row for either team
        carries NULL in all of them.
        """
        return {
            f"{prefix}_{side}_{column}": None
            for prefix in ("home", "away")
            for side in ("off", "def")
            for column in ROLLING_COLUMNS
        }

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game: the END of the latest team-game its window admitted.

        For each team that CARRIES a rolling row for the game's (season, week) in the
        rolling table, the time is the latest ``_end`` in :meth:`admitted_window` at the
        game's own lock -- the one selection the rolling averages read -- over the
        team's scheduled games. A game with no rolling row for either team (the seasons
        before the table's coverage, the postseason) is ``basis="no_information"`` with
        a NULL time, value-checked against :meth:`no_information_signature`.

        Returns:
            A frame with exactly ``PROVENANCE_COLUMNS``.
        """
        target = games_df
        if target_season is not None and target_week is not None:
            target = games_df[
                (games_df["season"] == target_season)
                & (games_df["week"] == target_week)
            ]
        records: list[dict[str, Any]] = []
        if len(target) == 0:
            return pd.DataFrame(records, columns=list(PROVENANCE_COLUMNS))

        form = (
            self._form_df
            if self._form_df is not None
            else load_dataframe("team_form_features", layer="silver")
        )
        carried = {
            (str(team), int(season), int(week))
            for team, season, week in zip(
                form["team"], form["target_season"], form["target_week"], strict=True
            )
        }
        schedule = self._team_schedule()
        by_team = {str(team): rows for team, rows in schedule.groupby("team")}
        locks = lock_rule.lock_frame(target)
        for game in target.to_dict("records"):
            season, week = int(game["season"]), int(game["week"])
            lock = locks[str(game["game_id"])]
            ends = []
            for column in ("home_team", "away_team"):
                team = str(game[column])
                if (team, season, week) not in carried or team not in by_team:
                    continue
                window = self.admitted_window(by_team[team], season, week, lock)
                if len(window) > 0:
                    ends.append(pd.Timestamp(window["_end"].max()))
            latest = max(ends) if ends else None
            records.append(
                {
                    "game_id": str(game["game_id"]),
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
    # Deprecated aliases (kept for backward compatibility)
    # ------------------------------------------------------------------

    def build_team_form_features(
        self,
        seasons: list[int],
        target_season: int | None = None,
        target_week: int | None = None,
        *,
        bootstrap_seasons: Sequence[int] = (),
    ) -> pd.DataFrame:
        """Build complete team form features for specified seasons.

        .. deprecated::
            Use :meth:`build_features` instead, which enforces the
            ``as_of_datetime`` time-fence.

        THE BOOTSTRAP SEASONS (Plan 33.2-17 Task 1). A season's week-1 window rolls over the
        PRIOR season's games, so the first season of a corpus build needs the season before it
        fetched too -- otherwise its week 1 has no row at all. *bootstrap_seasons* are fetched
        and timed exactly like the targets, but no rolling row and no ``team_game_stats`` row
        is computed or persisted for them. Every window is still admitted at its target game's
        lock (:meth:`admitted_window`); fetching an extra season widens what is AVAILABLE,
        never what a single game may see.

        Args:
            seasons: Seasons to process play-by-play data for
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for
            bootstrap_seasons: Seasons fetched only to supply earlier windows. The caller's
                schedule must time them (a team-game that cannot be timed is never admitted).

        Returns:
            DataFrame with team form features
        """
        warnings.warn(
            "build_team_form_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building team form features (deprecated path)",
            seasons=seasons,
            target_season=target_season,
            target_week=target_week,
        )

        # The incremental/current-week invocation (build_for_current_week ->
        # target_season AND target_week set) builds only a 3-season subset
        # [current-2, current-1, current]. The full-rebuild invocations
        # (build_for_seasons via --season/--seasons/--all-seasons/default) pass
        # the complete season set with no target. Only the full path is the
        # canonical producer of the persisted silver tables.
        incremental = target_season is not None and target_week is not None

        try:
            # Fetch play-by-play data: the targets plus any bootstrap season
            pbp_df = self.fetch_pbp_data(
                sorted({*map(int, bootstrap_seasons), *map(int, seasons)})
            )

            # Calculate team-game statistics (persisted UNTIMED below; timed for the
            # lock-keyed windows, Plan 33.2-14)
            team_stats_df = self.calculate_team_game_stats(pbp_df)
            schedule = self._team_schedule()
            timed_stats_df = self.time_team_games(team_stats_df, schedule)

            # Persist team statistics to silver layer ONLY on the full-rebuild
            # path. On the full path team_stats_df IS the complete team-game
            # stat table, so replace_mode writes a single self-replacing file
            # that is idempotent (no directory-partition append bloat -- the
            # F-02 ~127k duplicate (game_id, team) rows came from blind appends
            # into the shared data/silver/season=YYYY/ root). (FIX-01, D-13)
            #
            # On the incremental/--current path we must NOT replace the
            # persisted table: team_stats_df holds only ~3 seasons there, and
            # replace_mode would shrink the on-disk 2002-2024 history down to
            # those 3 seasons -- a silent data-loss regression (WR-01). We skip
            # the persisted write entirely rather than append, because (a) the
            # full path is the canonical producer of the complete table and
            # (b) no code path reads team_game_stats from disk -- build_features
            # recomputes per-game stats in-memory via
            # TeamFormCalculator.get_per_game_stats() -- so the current-week run
            # has no need to mutate the artifact. Idempotency therefore holds
            # within a fixed seasons argument; switching between full and
            # current builds no longer mutates the table's season span.
            if not incremental:
                # The targets only: a bootstrap season is fetched to supply windows, never
                # persisted as if it were part of the corpus.
                save_dataframe(
                    team_stats_df[team_stats_df["season"].isin(seasons)],
                    "team_game_stats",
                    layer="silver",
                    replace_mode=True,
                )
            else:
                logger.info(
                    "Skipping persisted team_game_stats write on incremental "
                    "(--current) path to preserve full-history table (WR-01)",
                    seasons=seasons,
                    target_season=target_season,
                    target_week=target_week,
                )

            # If specific target provided, calculate rolling averages for that point
            if target_season and target_week:
                rolling_df = self.calculate_rolling_averages(
                    timed_stats_df, target_season, target_week, schedule=schedule
                )
                return rolling_df

            # Otherwise, calculate rolling averages for all weeks
            all_rolling_stats = []

            for season in seasons:
                for week in range(1, 19):  # Weeks 1-18
                    rolling_df = self.calculate_rolling_averages(
                        timed_stats_df, season, week, schedule=schedule
                    )
                    if len(rolling_df) > 0:
                        all_rolling_stats.append(rolling_df)

            if all_rolling_stats:
                final_df = pd.concat(all_rolling_stats, ignore_index=True)

                # Save rolling team form features.
                # replace_mode: final_df is the complete rolling team-form table
                # for the seasons built; a single self-replacing file keeps the
                # rebuild idempotent. The prior partition_cols=["target_season"]
                # wrote into the shared data/silver/target_season=YYYY/ root,
                # which both mixed tables and appended non-idempotently. This is
                # the table build_features.py consumes, so de-bloating it is the
                # enabling correctness fix for the gold rebuild. (FIX-01, D-13)
                save_dataframe(
                    final_df,
                    "team_form_features",
                    layer="silver",
                    replace_mode=True,
                )

                logger.info(
                    "Built team form features",
                    total_records=len(final_df),
                    seasons=seasons,
                )

                return final_df
            return pd.DataFrame()

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(
                "Failed to build team form features", seasons=seasons, error=str(e)
            )
            raise

    def get_team_form_for_game(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, dict[str, float]]:
        """Get team form features for a specific game.

        .. deprecated::
            Use :meth:`get_features_for_game` instead, which enforces the
            ``as_of_datetime`` time-fence.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dictionary with form features for both teams
        """
        warnings.warn(
            "get_team_form_for_game is deprecated; use get_features_for_game instead",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            # Load team form features
            form_df = load_dataframe("team_form_features", layer="silver")

            # Filter to the specific season/week
            week_form = form_df[
                (form_df["target_season"] == season) & (form_df["target_week"] == week)
            ]

            result = {}

            for team in [home_team, away_team]:
                result[team] = {}

                # Get offensive and defensive form
                team_offense = week_form[
                    (week_form["team"] == team) & (week_form["side"] == "offense")
                ]
                team_defense = week_form[
                    (week_form["team"] == team) & (week_form["side"] == "defense")
                ]

                # Combine offense and defense metrics
                if len(team_offense) > 0:
                    offense_row = team_offense.iloc[0]
                    for col in offense_row.index:
                        if col.startswith("rolling_"):
                            result[team][f"off_{col}"] = offense_row[col]

                if len(team_defense) > 0:
                    defense_row = team_defense.iloc[0]
                    for col in defense_row.index:
                        if col.startswith("rolling_"):
                            result[team][f"def_{col}"] = defense_row[col]

            return result

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(
                "Failed to get team form for game",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def validate_form_features(self, form_df: pd.DataFrame) -> bool:
        """Validate team form features for data quality.

        Args:
            form_df: Team form features DataFrame

        Returns:
            True if validation passes
        """
        if len(form_df) == 0:
            logger.error("No team form features found")
            return False

        # Check for required columns
        required_cols = [
            "team",
            "side",
            "target_season",
            "target_week",
            "rolling_epa_per_play",
        ]
        missing_cols = set(required_cols) - set(form_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check for reasonable EPA ranges
        epa_cols = [col for col in form_df.columns if "epa_per_play" in col]
        for col in epa_cols:
            values = form_df[col].dropna()
            if len(values) > 0:
                if values.min() < -2.0 or values.max() > 2.0:
                    logger.warning(
                        f"EPA values outside reasonable range for {col}",
                        min_value=values.min(),
                        max_value=values.max(),
                    )

        # Check success rate ranges (should be 0-1)
        success_cols = [col for col in form_df.columns if "success_rate" in col]
        for col in success_cols:
            values = form_df[col].dropna()
            if len(values) > 0 and (values.min() < 0 or values.max() > 1):
                logger.warning(
                    f"Success rate outside 0-1 range for {col}",
                    min_value=values.min(),
                    max_value=values.max(),
                )

        logger.info(
            "Team form features validation completed",
            records=len(form_df),
            teams=len(form_df["team"].unique()),
            seasons=sorted(form_df["target_season"].unique()),
        )

        return True
