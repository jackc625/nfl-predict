"""Opponent-Adjusted EPA Feature (FEAT-14), selected at every game's own lock.

Single-pass opponent strength adjustment for EPA metrics, following the
Open Source Football methodology:
https://opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/

This is a post-processor: it takes per-game EPA stats (from TeamFormCalculator)
and produces opponent-adjusted rolling EPA values that replace the raw versions.

Algorithm, for each team-game (one row per team and side):

1. Resolve the play-by-play game to its scheduled game through the ONE canonical id
   mapping, ``utils.game_id_utils.convert_legacy_game_id``: that yields the opponent, the
   game's END (kickoff plus ``features.provenance.DECLARED_GAME_DURATION``) and its LOCK.
2. Look up the opponent's rolling level on the opposite side (its defense, to adjust an
   offense; its offense, to adjust a defense) as it stood at THIS game's lock: the mean over
   the opponent's last ``window`` games that ENDED at or before the lock.
3. adjustment = league_avg - opponent_level (positive against a worse-than-average unit),
   and adjusted_epa = raw_epa + adjustment.
4. The rolling feature for a target game is a recency-weighted mean of the team's adjusted
   per-game values over its games that ENDED at or before the TARGET's lock.

WHAT PLAN 33.2-16 CHANGED, AND WHY (SPEC R9 / R2)
-------------------------------------------------
* THE ADJUSTMENT NEVER RAN. The old lookup keyed the play-by-play id (``2023_01_ARI_WAS``)
  into silver ``games`` (``2023_W01_ARI@WAS``); nothing ever matched, so every opponent was
  ``""``, every history count NaN, and the RAW value was kept under the adjusted name -- 0 of
  1,088 rows adjusted for 2023. Opponents now resolve through the canonical converter, and an
  id that does not resolve RAISES :class:`OpponentResolutionError` rather than becoming ``""``.
* A ROW THE ADJUSTMENT CANNOT REACH IS SAID TO BE SO. Below the minimum opponent history the
  adjusted value is NaN beside an explicit coverage flag (:data:`OPP_ADJ_COVERAGE_COLUMN`),
  never raw EPA wearing an adjusted name.
* EVERY INPUT IS TIMED AT THE LOCK (D33.2-01). A prior game counts only once it ENDED at or
  before the lock of the game it informs (at-lock admissible, the ``<=`` of
  ``utils.game_lock.is_admissible``), and :meth:`OpponentAdjuster.information_times` reports
  the latest instant actually read, so the information-time gate checks the family.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from data.storage import load_dataframe
from features.provenance import PROVENANCE_COLUMNS, InformationBasis
from features.team_form import team_game_schedule
from utils import get_logger
from utils.exceptions import DataValidationError
from utils.game_id_utils import convert_legacy_game_id

logger = get_logger(__name__)

# EPA metrics to adjust (3 per side = 6 total)
_EPA_METRICS = ["epa_per_play", "pass_epa_per_play", "rush_epa_per_play"]

# Rolling output column names for each metric
_ROLLING_NAMES = {
    "epa_per_play": "rolling_opp_adj_epa_per_play",
    "pass_epa_per_play": "rolling_opp_adj_pass_epa",
    "rush_epa_per_play": "rolling_opp_adj_rush_epa",
}

#: The per-side coverage flag: 1.0 when the rolling values were computed from at least one
#: genuinely adjusted game, 0.0 when none (the values are then NaN).
#:
#: THE NAME IS LOAD-BEARING -- it must pass TWO filters in scripts/build_features.py, and the
#: obvious name passes only one. The family reaches gold ONLY through
#: ``FeatureMatrixBuilder._get_team_features``, which copies ONLY columns starting with
#: ``rolling_`` (renaming them ``{prefix}_off_{col}`` / ``{prefix}_def_{col}``), and THEN
#: through the merge's ``"opp_adj" in c`` filter. ``opp_adj_coverage`` passes the second and
#: fails the first, so it would be dropped before the second ever ran and never reach gold.
#: ``rolling_opp_adj_coverage`` passes both and emerges as FOUR gold columns per matrix:
#: ``home_off_`` / ``home_def_`` / ``away_off_`` / ``away_def_rolling_opp_adj_coverage``. Its
#: ``_coverage`` suffix also makes it level-preserved (never z-scored, never winsorized).
#: A rename that drops either substring silently loses the flag;
#: tests/unit/test_opponent_adjustment_runs.py pushes it through the real merge code.
#:
#: It is a GOLD-level feature column that never passes through a silver schema, so it is NOT
#: declared in data/schemas.py (Plan 33.2-15's schema ruling): the guard is the gold-survival
#: assertion in tests/integration/test_p332_rung7_attribution.py.
OPP_ADJ_COVERAGE_COLUMN: str = "rolling_opp_adj_coverage"

#: The information-time gate's name for this family. It is NOT a ``feature_sources`` registry
#: key: the family is merged after the Stage-1 loop and checked at that merge site.
OPP_ADJ_SOURCE_NAME: str = "opponent_adj"

_GOLD_PREFIXES = ("home", "away")
_GOLD_SIDES = {"offense": "off", "defense": "def"}
_OPPOSITE_SIDE = {"offense": "defense", "defense": "offense"}


def opponent_adjusted_gold_columns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(value_columns, flag_columns)`` the family lands in gold, in the merge's naming.

    Twelve values (home/away x off/def x the three rolling EPA metrics) and four flags.
    """
    values = tuple(
        f"{prefix}_{side}_{name}"
        for prefix in _GOLD_PREFIXES
        for side in _GOLD_SIDES.values()
        for name in _ROLLING_NAMES.values()
    )
    flags = tuple(
        f"{prefix}_{side}_{OPP_ADJ_COVERAGE_COLUMN}"
        for prefix in _GOLD_PREFIXES
        for side in _GOLD_SIDES.values()
    )
    return values, flags


class OpponentResolutionError(RuntimeError):
    """A play-by-play game id that does not resolve to exactly one scheduled team-game.

    A ``RuntimeError`` ON PURPOSE, so the refusal reaches the caller. The production merge
    site in ``scripts/build_features.py`` wraps the adjustment in ``except (ValueError,
    KeyError, TypeError)`` and logs "keeping raw EPA", and every optional source there is
    wrapped in ``_SOURCE_LOAD_ERRORS`` (``DataIngestionError``, ``ValueError``, ``KeyError``,
    ``TypeError``, ``FileNotFoundError``, ``OSError``). A ``ValueError`` here would be
    swallowed by both and turned straight back into the silent raw-EPA degradation this
    refusal exists to end. ``RuntimeError`` is in neither tuple and is a superclass of no
    member of either, so it propagates -- the exclusion-by-TYPE precedent
    ``ProvisionalSnapshotAsTrainingInputError`` set, and for the same reason a re-raising
    ``except`` handler would be the wrong fix (it would have to track the hierarchy forever).
    """


def _utc_ns(values: pd.Series) -> np.ndarray:
    """Tz-aware instants as int64 UTC nanoseconds, for ``np.searchsorted``."""
    instants = pd.to_datetime(values, utc=True)
    return (
        instants.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")
    )


def _from_utc_ns(value: int) -> pd.Timestamp:
    """The inverse of :func:`_utc_ns` for one instant."""
    return pd.Timestamp(value, unit="ns", tz="UTC")


class OpponentAdjuster:
    """Single-pass opponent strength adjustment for EPA metrics, timed at each game's lock.

    Adjusts a team's EPA by comparing their opponents' quality to the league average. Uses
    Open Source Football methodology:
    https://opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/

    Conforms to the FeatureBuilder Protocol via build_features() and
    get_features_for_game(), and to ``features.protocol.InformationTimeProvider`` via
    :meth:`information_times` and :meth:`no_information_signature`.
    """

    def __init__(
        self,
        window: int = 10,
        min_opponent_games: int = 4,
        *,
        schedule_df: pd.DataFrame | None = None,
    ):
        """Initialize opponent adjuster.

        Args:
            window: Number of games for the opponent rolling average (default 10).
            min_opponent_games: Minimum opponent games at the lock before the adjustment
                applies (default 4). The boundary is INCLUSIVE; below it the row is the
                flagged unknown (NaN), never raw EPA.
            schedule_df: Optional schedule (silver ``games`` shape) team-games are resolved
                and TIMED against. When ``None`` the adjuster loads silver ``games`` -- the
                FULL schedule, so a scoped build still times the prior season's games. The
                test-injection seam.
        """
        self.window = window
        self.min_opponent_games = min_opponent_games
        self._schedule_df = schedule_df
        self._team_games_cache: pd.DataFrame | None = None
        # The last build's rolling rows, which the provenance reads (one selection for both).
        self._rolling_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Resolution: the ONE canonical id mapping, and the loud miss
    # ------------------------------------------------------------------

    def _scheduled_team_games(self) -> pd.DataFrame:
        """One row per scheduled (game, team): opponent, END and LOCK. Built once."""
        if self._team_games_cache is None:
            games = (
                self._schedule_df
                if self._schedule_df is not None
                else load_dataframe("games", layer="silver")
            )
            timed = team_game_schedule(games)
            teams = games[["game_id", "home_team", "away_team"]].rename(
                columns={"game_id": "schedule_game_id"}
            )
            timed = timed.merge(teams, on="schedule_game_id", how="left")
            timed["opponent"] = np.where(
                timed["team"] == timed["home_team"],
                timed["away_team"],
                timed["home_team"],
            )
            self._team_games_cache = timed[
                ["schedule_game_id", "team", "opponent", "_end", "_lock"]
            ]
        return self._team_games_cache

    def resolve_team_games(self, team_game_stats: pd.DataFrame) -> pd.DataFrame:
        """*team_game_stats* with its scheduled game: ``schedule_game_id``, ``opponent``,
        ``_end`` and ``_lock`` (see :meth:`_add_opponent_column`).

        Raises:
            OpponentResolutionError: naming every id that does not resolve.
        """
        return self._add_opponent_column(team_game_stats)

    def _add_opponent_column(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Resolve every row's scheduled game and opponent through the canonical mapping.

        Every play-by-play id goes through ``convert_legacy_game_id`` (``2023_01_ARI_WAS`` ->
        ``2023_W01_ARI@WAS``), which normalizes abbreviations and refuses an unknown team or a
        week past 22; the canonical id must then name a scheduled, timed game that the row's
        team played in. There is no second id parser here.

        Raises:
            OpponentResolutionError: naming every id that fails any of those steps. Nothing
                unresolvable becomes an empty opponent.
        """
        stats = stats.drop(
            columns=["schedule_game_id", "opponent", "_end", "_lock"], errors="ignore"
        )
        scheduled = self._scheduled_team_games()
        lookup = {
            (str(game_id), str(team)): (str(opponent), end, lock)
            for game_id, team, opponent, end, lock in zip(
                scheduled["schedule_game_id"],
                scheduled["team"],
                scheduled["opponent"],
                scheduled["_end"],
                scheduled["_lock"],
                strict=True,
            )
        }

        resolved: dict[tuple[str, str], tuple[str, str, Any, Any]] = {}
        unresolvable: set[str] = set()
        pairs = stats[["game_id", "team"]].astype(str).drop_duplicates()
        for legacy_id, team in zip(pairs["game_id"], pairs["team"], strict=True):
            try:
                resolved[(legacy_id, team)] = self._get_opponent(
                    legacy_id, team, lookup
                )
            except OpponentResolutionError:
                unresolvable.add(legacy_id)
        if unresolvable:
            named = sorted(unresolvable)
            msg = (
                f"{len(named)} play-by-play game id(s) do not resolve to a scheduled, timed "
                f"team-game through utils.game_id_utils.convert_legacy_game_id: "
                f"{named[:10]}{' ...' if len(named) > 10 else ''}. The opponent adjustment "
                "refuses rather than adjusting against an unknown opponent."
            )
            raise OpponentResolutionError(msg)

        keys = list(
            zip(stats["game_id"].astype(str), stats["team"].astype(str), strict=True)
        )
        stats = stats.copy()
        stats["schedule_game_id"] = [resolved[key][0] for key in keys]
        stats["opponent"] = [resolved[key][1] for key in keys]
        stats["_end"] = pd.to_datetime([resolved[key][2] for key in keys], utc=True)
        stats["_lock"] = pd.to_datetime([resolved[key][3] for key in keys], utc=True)
        return stats

    @staticmethod
    def _get_opponent(
        legacy_id: str,
        team: str,
        lookup: Mapping[tuple[str, str], tuple[str, Any, Any]],
    ) -> tuple[str, str, Any, Any]:
        """``(schedule_game_id, opponent, end, lock)`` for one play-by-play team-game.

        Raises:
            OpponentResolutionError: the id does not convert, names no scheduled and timed
                game, or names one *team* did not play in.
        """
        try:
            canonical = convert_legacy_game_id(legacy_id)
        except (ValueError, DataValidationError) as exc:
            msg = f"{legacy_id!r} does not convert to a canonical game id: {exc}"
            raise OpponentResolutionError(msg) from exc
        entry = lookup.get((canonical, team))
        if entry is None:
            msg = (
                f"{legacy_id!r} -> {canonical!r} is not a scheduled, timed game that "
                f"{team!r} played in"
            )
            raise OpponentResolutionError(msg)
        opponent, end, lock = entry
        return canonical, opponent, end, lock

    # ------------------------------------------------------------------
    # Opponent levels and league averages, each as it stood at a lock
    # ------------------------------------------------------------------

    def _side_states(self, timed: pd.DataFrame, side: str) -> pd.DataFrame:
        """Each team's rolling level on *side* after each of its games.

        One row per team-game: the mean of each metric over the team's last ``window`` games
        through this one (NaN-skipping), how many games that window holds, and the END of
        this game -- the instant the level became known.
        """
        rows = timed.loc[timed["side"] == side].sort_values(["team", "_end"])
        parts = []
        for team, group in rows.groupby("team", sort=True):
            state = pd.DataFrame(
                {
                    "team": team,
                    "_end_ns": _utc_ns(group["_end"]),
                    "games_in_window": np.minimum(
                        np.arange(1, len(group) + 1), self.window
                    ),
                }
            )
            for metric in _EPA_METRICS:
                state[metric] = (
                    group[metric]
                    .astype(float)
                    .rolling(self.window, min_periods=1)
                    .mean()
                    .to_numpy()
                )
            parts.append(state)
        if not parts:
            return pd.DataFrame(
                columns=["team", "_end_ns", "games_in_window", *_EPA_METRICS]
            )
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _states_at(
        states: pd.DataFrame, teams: pd.Series, locks_ns: np.ndarray
    ) -> pd.DataFrame:
        """For each (team, lock), the team's latest state whose game ENDED at or before it.

        Returns one row per query, aligned to *teams*: ``games_in_window`` (0 when the team
        had played nothing by then), each metric (NaN then) and ``_end_ns`` (-1 then).
        """
        result = pd.DataFrame(
            {"games_in_window": 0, "_end_ns": -1}, index=range(len(teams))
        )
        for metric in _EPA_METRICS:
            result[metric] = np.nan
        by_team = dict(list(states.groupby("team", sort=False)))
        team_values = teams.to_numpy()
        for team in pd.unique(team_values):
            positions = np.flatnonzero(team_values == team)
            group = by_team.get(team)
            if group is None:
                continue
            # at-lock admissible: a game ending exactly AT the lock counts (``side="right"``)
            index = (
                np.searchsorted(
                    group["_end_ns"].to_numpy(), locks_ns[positions], "right"
                )
                - 1
            )
            found = index >= 0
            hit = positions[found]
            chosen = group.iloc[index[found]]
            result.loc[hit, "games_in_window"] = chosen["games_in_window"].to_numpy()
            result.loc[hit, "_end_ns"] = chosen["_end_ns"].to_numpy()
            for metric in _EPA_METRICS:
                result.loc[hit, metric] = chosen[metric].to_numpy()
        return result

    def league_average_at(
        self, states: pd.DataFrame, locks_ns: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``metric -> (league average, latest END read)`` at each lock in *locks_ns*.

        The mean over every team's rolling level on the WHOLE loaded frame (the pre-Plan
        33.2-16 rule, pending Task 2), broadcast to each lock.
        """
        averages: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        latest = int(states["_end_ns"].max()) if len(states) else -1
        for metric in _EPA_METRICS:
            mean = float(states[metric].mean()) if len(states) else np.nan
            averages[metric] = (
                np.full(len(locks_ns), mean),
                np.full(len(locks_ns), latest, dtype="int64"),
            )
        return averages

    def per_game_adjusted(self, team_game_stats: pd.DataFrame) -> pd.DataFrame:
        """Every per-game row resolved, timed and -- where the rule allows -- adjusted.

        Adds ``schedule_game_id``, ``opponent``, ``_end``, ``_lock``,
        ``opponent_prior_games`` (the opponent's games at this game's lock, capped at
        ``window``), ``opp_adj_{metric}`` for the three metrics, ``adjusted`` and
        ``information_end`` (the latest instant the adjusted value read; NaT when not
        adjusted).

        Raises:
            OpponentResolutionError: see :meth:`resolve_team_games`.
        """
        timed = self.resolve_team_games(team_game_stats).reset_index(drop=True)
        timed["opponent_prior_games"] = 0
        timed["adjusted"] = False
        timed["information_end"] = pd.Series(
            pd.NaT, index=timed.index, dtype="datetime64[ns, UTC]"
        )
        for metric in _EPA_METRICS:
            timed[f"opp_adj_{metric}"] = np.nan

        for side, opposite in _OPPOSITE_SIDE.items():
            rows = timed.index[timed["side"] == side]
            if len(rows) == 0:
                continue
            states = self._side_states(timed, opposite)
            locks_ns = _utc_ns(timed.loc[rows, "_lock"])
            opponent = self._states_at(
                states, timed.loc[rows, "opponent"].reset_index(drop=True), locks_ns
            )
            league = self.league_average_at(states, locks_ns)

            # THE BOUNDARY IS INCLUSIVE: an opponent with EXACTLY ``min_opponent_games``
            # games at the lock is adjusted; one fewer is the flagged unknown.
            has_minimum_history = (
                opponent["games_in_window"].to_numpy() >= self.min_opponent_games
            )
            computable = has_minimum_history.copy()
            for metric in _EPA_METRICS:
                computable &= opponent[metric].notna().to_numpy()
                computable &= ~np.isnan(league[metric][0])

            latest_read = np.maximum(
                opponent["_end_ns"].to_numpy(), _utc_ns(timed.loc[rows, "_end"])
            )
            for metric in _EPA_METRICS:
                raw = timed.loc[rows, metric].astype(float).to_numpy()
                adjustment = league[metric][0] - opponent[metric].to_numpy()
                timed.loc[rows, f"opp_adj_{metric}"] = np.where(
                    computable, raw + adjustment, np.nan
                )
                latest_read = np.maximum(latest_read, league[metric][1])

            timed.loc[rows, "opponent_prior_games"] = opponent[
                "games_in_window"
            ].to_numpy()
            timed.loc[rows, "adjusted"] = computable
            timed.loc[rows, "information_end"] = [
                _from_utc_ns(int(ns)) if ok else pd.NaT
                for ns, ok in zip(latest_read, computable, strict=True)
            ]
        return timed

    # ------------------------------------------------------------------
    # The rolling feature, at each target game's lock
    # ------------------------------------------------------------------

    def _rolling_for_targets(
        self, adjusted: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """The rolling rows for each target team-game, from the team's games at its lock.

        Recency weighting [1, 2, ..., N] over the team's history (N = games that ENDED at or
        before the target's lock), renormalized over the genuinely adjusted games only; a
        target with history but no adjusted game gets NaN values and the flag 0.0, and a
        target with no history at all gets no row (the layout then carries the unknown).
        """
        rows: list[dict[str, Any]] = []
        grouped = {
            key: group.sort_values(["_end", "season", "week"])
            for key, group in adjusted.groupby(["team", "side"], sort=True)
        }
        for (team, side), history in grouped.items():
            team_targets = targets.loc[targets["team"] == team]
            if len(team_targets) == 0:
                continue
            ends_ns = _utc_ns(history["_end"])
            weights = np.arange(1, len(history) + 1, dtype=float)
            mask = history["adjusted"].to_numpy(dtype=bool)
            weighted_mask = np.cumsum(weights * mask)
            weighted_values = {
                metric: np.cumsum(
                    weights
                    * np.where(mask, history[f"opp_adj_{metric}"].to_numpy(float), 0.0)
                )
                for metric in _EPA_METRICS
            }
            info_ns = np.where(
                mask,
                _utc_ns(history["information_end"].fillna(pd.Timestamp(0, tz="UTC"))),
                -1,
            )
            latest_info = np.maximum.accumulate(info_ns)
            counts = np.searchsorted(ends_ns, _utc_ns(team_targets["lock"]), "right")
            for target, count in zip(
                team_targets.to_dict("records"), counts, strict=True
            ):
                if count == 0:
                    continue
                last = count - 1
                covered = weighted_mask[last] > 0
                row: dict[str, Any] = {
                    "team": team,
                    "side": side,
                    "target_season": int(target["season"]),
                    "target_week": int(target["week"]),
                    "games_used": int(count),
                }
                for metric in _EPA_METRICS:
                    row[_ROLLING_NAMES[metric]] = (
                        float(weighted_values[metric][last] / weighted_mask[last])
                        if covered
                        else np.nan
                    )
                row[OPP_ADJ_COVERAGE_COLUMN] = 1.0 if covered else 0.0
                row["latest_information_end"] = (
                    _from_utc_ns(int(latest_info[last])) if covered else pd.NaT
                )
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _target_team_games(games_df: pd.DataFrame) -> pd.DataFrame:
        """One row per (game, team) of *games_df*: team, season, week and the game's lock."""
        if len(games_df) == 0:
            return pd.DataFrame(columns=["game_id", "team", "season", "week", "lock"])
        locks = lock_rule.lock_frame(games_df)
        sides = [
            pd.DataFrame(
                {
                    "game_id": games_df["game_id"].astype(str).to_numpy(),
                    "team": games_df[column].astype(str).to_numpy(),
                    "season": games_df["season"].astype(int).to_numpy(),
                    "week": games_df["week"].astype(int).to_numpy(),
                }
            )
            for column in ("home_team", "away_team")
        ]
        targets = pd.concat(sides, ignore_index=True)
        targets["lock"] = targets["game_id"].map(locks.to_dict())
        return targets

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
        team_game_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build opponent-adjusted rolling EPA features for the games in *games_df*.

        Conforms to the FeatureBuilder Protocol. ``as_of_datetime`` is NOT the fence: every
        input is admitted at the lock of the game it informs (Plan 33.2-16). It is kept for
        the Protocol's call shape and logged.

        Args:
            games_df: The games to build features for (``game_id``, ``season``, ``week``,
                ``home_team``, ``away_team``, tz-aware ``kickoff_et``).
            as_of_datetime: The Protocol's cutoff argument; see above.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.
            team_game_stats: Per-game stats from TeamFormCalculator (play-by-play ids).

        Returns:
            One row per (team, side) per target game with a history: ``team``, ``side``,
            ``target_season``, ``target_week``, ``games_used``, the three
            ``rolling_opp_adj_*`` values, :data:`OPP_ADJ_COVERAGE_COLUMN` and
            ``latest_information_end``.

        Raises:
            OpponentResolutionError: a play-by-play id that does not resolve.
        """
        logger.info(
            "Building opponent-adjusted EPA features",
            as_of_datetime=str(as_of_datetime),
            target_season=target_season,
            target_week=target_week,
        )

        if team_game_stats is None:
            msg = (
                "team_game_stats must be provided. OpponentAdjuster is a "
                "post-processor that requires pre-computed per-game stats "
                "from TeamFormCalculator."
            )
            raise ValueError(msg)

        stats = team_game_stats.copy()
        # Filter to only relevant seasons (target + prior)
        if target_season is not None:
            stats = stats[stats["season"].isin([target_season - 1, target_season])]

        targets_games = games_df
        if target_season is not None:
            targets_games = targets_games[targets_games["season"] == target_season]
        if target_week is not None:
            targets_games = targets_games[targets_games["week"] == target_week]

        adjusted = self.per_game_adjusted(stats)
        rolling = self._rolling_for_targets(
            adjusted, self._target_team_games(targets_games)
        )
        self._rolling_df = rolling
        return rolling

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get opponent-adjusted EPA features for a single game.

        Conforms to the FeatureBuilder Protocol. Returns a dictionary
        mapping feature names to values for the specified game.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        # This method requires access to stored features or a full
        # computation pipeline. For now, return empty dict -- the primary
        # interface is build_features().
        logger.warning(
            "get_features_for_game not yet implemented for OpponentAdjuster",
            game_id=game_id,
        )
        return {}

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game whose teams read no adjusted game: every value NULL, every flag 0.0."""
        values, flags = opponent_adjusted_gold_columns()
        return {**dict.fromkeys(values), **dict.fromkeys(flags, 0.0)}

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game: the latest instant any of its rolling values read.

        Read from the SAME rolling rows the last :meth:`build_features` produced, keyed
        exactly as the gold merge keys them -- (team, target season, target week, side) for
        the home and away team. A game with at least one genuinely adjusted rolling value is
        ``basis="per_row"`` at the latest ``latest_information_end`` among them (the END of
        the latest game read, or of the latest opponent or league-average input, whichever
        is later); a game with none -- a team's first game, a history below the minimum --
        is ``basis="no_information"`` with a NULL time, value-checked against
        :meth:`no_information_signature`.

        Returns:
            A frame with exactly ``PROVENANCE_COLUMNS``.
        """
        target = games_df
        if target_season is not None:
            target = target[target["season"] == target_season]
        if target_week is not None:
            target = target[target["week"] == target_week]

        latest: dict[tuple[str, int, int], pd.Timestamp] = {}
        rolling = self._rolling_df
        if rolling is not None and len(rolling) > 0:
            covered = rolling.loc[rolling[OPP_ADJ_COVERAGE_COLUMN] == 1.0]
            for (team, season, week), group in covered.groupby(
                ["team", "target_season", "target_week"]
            ):
                latest[(str(team), int(season), int(week))] = pd.Timestamp(
                    group["latest_information_end"].max()
                )

        records: list[dict[str, Any]] = []
        for game in target.to_dict("records"):
            season, week = int(game["season"]), int(game["week"])
            ends = [
                latest[key]
                for key in (
                    (str(game["home_team"]), season, week),
                    (str(game["away_team"]), season, week),
                )
                if key in latest
            ]
            when = max(ends) if ends else None
            records.append(
                {
                    "game_id": str(game["game_id"]),
                    "basis": (
                        InformationBasis.PER_ROW.value
                        if when is not None
                        else InformationBasis.NO_INFORMATION.value
                    ),
                    "information_time": when,
                }
            )
        return pd.DataFrame(records, columns=list(PROVENANCE_COLUMNS))
