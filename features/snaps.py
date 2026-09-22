"""Snap-Count Feature Builder (SIG-02).

This module exposes the player-level snap-count silver table (produced by the
Plan 28-02 ingest) as backward-rolling, leakage-safe TEAM features. Where
``team_form`` already owns *how well* a team plays, the snap signal captures
*who plays and how stably* -- the week-over-week churn of the snap-share
distribution and how concentrated snaps are across position groups.

Two model-facing team features (D-11):

- ``snap_continuity`` -- weighted week-over-week retention of snap share among
  returning players (1.0 = an identical snap-share distribution to last game,
  lower = more churn / new faces taking snaps).
- ``snap_concentration`` -- weighted Herfindahl-Hirschman index (HHI) of the
  positional snap-share distribution (higher = snaps concentrated in fewer
  position groups).

PLUS the per-position PRIOR snap shares (D-11) -- ``rolling_snap_share_*`` --
exposed both as model columns and, via :meth:`get_position_prior_shares`, as the
availability weights the Plan 28-05 ``InjuryBuilder`` (D-09) consumes. This is
the FIRST builder in the snaps->injuries dependency chain.

Key constraints:

- No data leakage: snaps are POST-game, so a team-game's snaps are known once
  that game has ENDED (kickoff plus ``features.provenance.DECLARED_GAME_DURATION``,
  timed against the schedule), and a target game's window admits a prior game only
  when that end is at or before the target's OWN lock (``utils.game_lock``, Plan
  33.2-14). The window used to be keyed on the WEEK LABEL (``week < target_week``);
  on an ordinary schedule the two agree, and the lock-keyed form is the rule itself
  -- the only one that also excludes a rescheduled prior game played after the
  target's lock.
- The rolling window is REUSED from ``team_form`` (D-12): the dynamic expanding
  window + linear recency weights are NOT re-implemented here.
- Derived feature names use ``snap_continuity`` / ``snap_concentration`` /
  ``rolling_snap_share_*`` -- never ``_pct`` or the raw count spellings
  (``offense_snaps`` / ``defense_snaps`` / ``st_snaps``), so the D-13
  ``LEAKAGE_KEYWORDS`` substring guard catches only true raw leaks.
- Canonical team abbreviations via ``normalize_team_abbreviation`` (snaps back
  to 2013 carry SD / STL / OAK historical codes; hard-fail on unknowns).
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from data.storage import load_dataframe
from features.provenance import (
    PROVENANCE_COLUMNS,
    InformationBasis,
)
from features.team_form import TeamFormCalculator, team_game_schedule, week_team_locks
from utils import get_logger
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Dynamic window parameters (matching TeamFormCalculator / QBTracker)
MAX_PRIOR_GAMES = 8

# Raw snap-count columns carried by the silver table. Each player's total
# participation is the sum of these three; they are NEVER emitted as features
# (they are the D-13 leakage-keyword targets).
RAW_SNAP_COUNT_COLUMNS = ["offense_snaps", "defense_snaps", "st_snaps"]

# Map raw nflreadpy position codes to stable, model-facing position GROUPS.
# Grouping keeps the per-position share columns deterministic (a fixed column
# set every build) regardless of which exact position codes appear in a week.
POSITION_GROUP_MAP = {
    "QB": "qb",
    "RB": "rb",
    "FB": "rb",
    "HB": "rb",
    "WR": "wr",
    "TE": "te",
    "T": "ol",
    "G": "ol",
    "C": "ol",
    "OL": "ol",
    "OT": "ol",
    "OG": "ol",
    "LT": "ol",
    "RT": "ol",
    "LG": "ol",
    "RG": "ol",
    "DE": "dl",
    "DT": "dl",
    "NT": "dl",
    "DL": "dl",
    "EDGE": "lb",
    "LB": "lb",
    "ILB": "lb",
    "OLB": "lb",
    "MLB": "lb",
    "CB": "db",
    "S": "db",
    "SS": "db",
    "FS": "db",
    "DB": "db",
}

# The fixed, sorted set of position groups exposed as share columns. "other"
# (kickers, punters, long-snappers, unmapped codes) still counts toward the
# team snap denominator but is not a key availability group (D-09).
KEY_POSITION_GROUPS = ["db", "dl", "lb", "ol", "qb", "rb", "te", "wr"]

# THE SNAP COVERAGE FLAG (Plan 33.2-17 Task 2, D33.2-08 item 2). 1.0 when a team's value was
# computed from at least one admitted snap team-game, 0.0 when its window admitted none -- then
# every snap value beside it is NaN, never 0.0. A 0.0 in a z-scored or centred feature reads as
# "exactly average", a confident claim about a game nobody measured; that is the defect this
# flag and the NaN replace, and before snaps' first covered season it spanned every snap column
# of every model. The FIRST COVERED SEASON IS DERIVED FROM THE DATA, never a literal: a season
# before any snap row exists has no admitted game in any window, so the boundary (2012 / 2013 on
# today's feed) emerges from what upstream supplied (D30-02: one answer on disk). The name ends
# in ``_coverage``, so the gold build keeps its level (never z-scored, never winsorized).
SNAP_COVERAGE_COLUMN: str = "snap_coverage"


class SnapCountBuilder:
    """Build backward-rolling team snap features from player-level snap silver.

    Conforms to the FeatureBuilder Protocol (``build_features`` with the
    ``as_of_datetime`` time-fence + ``get_features_for_game``). Emits per game
    the home/away-expanded ``snap_continuity`` / ``snap_concentration`` /
    ``rolling_snap_share_*`` columns and exposes the per-position prior shares
    that the Plan 28-05 InjuryBuilder consumes as availability weights (D-09).

    The dynamic expanding window and linear recency weights are REUSED from
    :class:`TeamFormCalculator` (D-12), not re-implemented.
    """

    def __init__(
        self,
        snaps_df: pd.DataFrame | None = None,
        max_prior_games: int = MAX_PRIOR_GAMES,
        schedule_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the snap-count builder.

        Args:
            snaps_df: Optional player-level snap frame to use directly (the
                test-injection / constructor-handoff seam). When ``None`` the
                builder loads ``snap_counts`` from the silver layer.
            max_prior_games: Maximum prior-season games in the rolling window
                when current-season data is sparse (matches TeamFormCalculator).
            schedule_df: Optional schedule (silver ``games`` shape: ``game_id``,
                ``season``, ``week``, ``home_team``, ``away_team``, a tz-aware
                ``kickoff_et``) the snap team-games are TIMED against and the
                target games are LOCKED from -- the same injection seam as
                *snaps_df*. When ``None`` the builder loads silver ``games``.
        """
        self._snaps_df = snaps_df
        self._schedule_df = schedule_df
        self.max_prior_games = max_prior_games

        # The team_form calculator owns the dynamic window (D-12). We reuse its
        # _select_dynamic_window verbatim rather than re-implementing it.
        self._window_source = TeamFormCalculator(max_prior_games=max_prior_games)

        # Caches keyed by nothing (single load); populated lazily.
        self._team_game_cache: pd.DataFrame | None = None
        self._games_cache: pd.DataFrame | None = None
        self._team_schedule_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Silver loading + team-game aggregation
    # ------------------------------------------------------------------

    def _load_snaps(self) -> pd.DataFrame:
        """Load the player-level snap silver (or the injected frame)."""
        if self._snaps_df is not None:
            return self._snaps_df
        return load_dataframe("snap_counts", layer="silver")

    @staticmethod
    def _position_group(position: object) -> str:
        """Map a raw position code to a stable model-facing position group."""
        if position is None or (isinstance(position, float) and np.isnan(position)):
            return "other"
        return POSITION_GROUP_MAP.get(str(position).upper().strip(), "other")

    def _team_game_aggregates(self, snaps_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player-level snaps to one row per (game_id, team).

        Produces, per team-game:
        - ``snap_concentration`` -- HHI of the positional snap-share distribution
        - ``snap_share_{group}`` -- each key position group's share of team snaps
        - ``_player_shares`` -- {player_id: snap_share} dict (for continuity)

        Args:
            snaps_df: Player-level snap rows (one per game_id+player).

        Returns:
            Team-game aggregate DataFrame sorted chronologically, with a
            per-team ``snap_continuity`` column derived from consecutive games.
        """
        # AN EMPTY SNAP TABLE IS "NO TEAM-GAMES", NOT A CRASH (Plan 33.2-15, D33.2-16). It used
        # to raise KeyError('team') here, which the gold build's optional-source handler turned
        # into a zero-row frame the information-time gate could not date. Now every target game
        # reads NULL in every snap column and reports basis="no_information", which the gate
        # value-checks against no_information_signature -- the honest unknown, checked.
        if len(snaps_df) == 0:
            return pd.DataFrame()

        df = snaps_df.copy()

        # Coerce raw snap counts to numeric; a player's participation is the sum
        # of offense + defense + special-teams snaps. The 0.0 here is a PLAYER's missing count
        # INSIDE a covered team-game (he took no snaps of that kind), not a stand-in for a game
        # nobody measured: a season with no snap rows never reaches this line, and its games
        # read NaN with ``snap_coverage`` 0.0 (see SNAP_COVERAGE_COLUMN).
        for col in RAW_SNAP_COUNT_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["player_snaps"] = df[RAW_SNAP_COUNT_COLUMNS].sum(axis=1)

        # Drop players who took no snaps (they carry no signal and would only
        # add zero-weight noise to the shares).
        df = df[df["player_snaps"] > 0].copy()

        # Canonical team + stable position group.
        df["team"] = df["team"].apply(normalize_team_abbreviation)
        df["pos_group"] = df["position"].apply(self._position_group)

        records: list[dict] = []
        for (game_id, season, week, team), group in df.groupby(
            ["game_id", "season", "week", "team"]
        ):
            total = float(group["player_snaps"].sum())
            if total <= 0:
                continue

            # Positional snap shares over ALL groups (incl. "other") -> HHI.
            pos_snaps = group.groupby("pos_group")["player_snaps"].sum()
            pos_share = pos_snaps / total
            concentration = float((pos_share**2).sum())

            record: dict = {
                "game_id": game_id,
                "season": int(season),
                "week": int(week),
                "team": team,
                "snap_concentration": concentration,
            }

            # Per-key-group share columns (0.0 when a group is absent this game).
            for grp in KEY_POSITION_GROUPS:
                record[f"snap_share_{grp}"] = float(pos_share.get(grp, 0.0))

            # Player-level snap shares keep continuity computable downstream.
            player_shares = (
                group.groupby("pfr_player_id")["player_snaps"].sum() / total
            ).to_dict()
            record["_player_shares"] = player_shares

            records.append(record)

        if not records:
            return pd.DataFrame()

        team_game = pd.DataFrame(records).sort_values(["team", "season", "week"])

        # Week-over-week continuity: for each team game, the retained snap share
        # vs the team's immediately-preceding game = sum over players present in
        # BOTH games of min(share_now, share_prev). Depends only on the current
        # and prior game, never on future games (the withhold-future property).
        team_game["snap_continuity"] = np.nan
        for _team, team_rows in team_game.groupby("team"):
            prev_shares: dict | None = None
            for idx in team_rows.index:
                cur_shares = team_game.at[idx, "_player_shares"]
                if prev_shares is not None:
                    shared_players = set(cur_shares) & set(prev_shares)
                    continuity = sum(
                        min(cur_shares[p], prev_shares[p]) for p in shared_players
                    )
                    team_game.at[idx, "snap_continuity"] = float(continuity)
                prev_shares = cur_shares

        return team_game

    def _team_schedule(self) -> pd.DataFrame:
        """The timing schedule, built once: ``features.team_form.team_game_schedule``.

        THE ONE TIMING of a team-game shared with team form (D-12): its END (kickoff
        plus ``DECLARED_GAME_DURATION``, the Elo replay's duration) and its lock (the
        one rule). A scheduled game with no kickoff has neither, so its snaps are never
        admitted and it is never a target; a naive kickoff is refused by name.
        """
        if self._team_schedule_cache is None:
            games = (
                self._schedule_df
                if self._schedule_df is not None
                else load_dataframe("games", layer="silver")
            )
            self._team_schedule_cache = team_game_schedule(games)
        return self._team_schedule_cache

    def _get_team_game(self) -> pd.DataFrame:
        """Return the cached team-game aggregate frame (computed once), each row TIMED.

        ``_end`` is the scheduled game's end instant, matched by (season, week, team)
        -- the snap ids (``2023_01_BUF_KC``) and the schedule's never match. A
        team-game with no scheduled match has no known end (NaT) and is therefore
        never admitted, which is the conservative answer.
        """
        if self._team_game_cache is None:
            team_game = self._team_game_aggregates(self._load_snaps())
            if len(team_game) > 0:
                timing = self._team_schedule()[["season", "week", "team", "_end"]]
                team_game = team_game.merge(
                    timing, on=["season", "week", "team"], how="left"
                )
            self._team_game_cache = team_game
        return self._team_game_cache

    def _week_team_locks(self, target_season: int, target_week: int) -> dict[str, Any]:
        """``team -> lock`` of each team's scheduled game in (season, week)."""
        return week_team_locks(self._team_schedule(), target_season, target_week)

    # ------------------------------------------------------------------
    # Backward-rolling feature computation (window REUSED from team_form)
    # ------------------------------------------------------------------

    def _window_for_team(
        self,
        team_game: pd.DataFrame,
        team: str,
        target_season: int,
        target_week: int,
        lock: Any,
    ) -> pd.DataFrame:
        """Select the rolling window for one team, admitted at the target's LOCK.

        A team-game is admitted only when its END instant (``_end``) is at or before
        *lock*, the target game's own lock (at-lock admissible, the ``<=`` of
        ``utils.game_lock.is_admissible``). The admitted games then go through the
        REUSED ``TeamFormCalculator._select_dynamic_window`` (D-12).

        WHY LOCK-KEYED AND NOT WEEK-KEYED (Plan 33.2-14). The retired fence read the
        WEEK LABEL (``week < target_week``). The window is per team and a team plays
        at most once a week, so on an ordinary schedule the two select the same
        games (D33.2-01 measured 0 games in 2002-2026 whose week-keyed inputs include
        a result that ended after their lock). The lock-keyed window IS the
        admissibility rule rather than a proxy that happens to agree with it, and it
        is the only one that excludes a game whose week label precedes the target
        week but which was PLAYED after the target's lock -- a rescheduled game.

        Args:
            team_game: Team-game aggregate frame (timed: ``_end``).
            team: Canonical team abbreviation.
            target_season: Season being predicted.
            target_week: Week being predicted.
            lock: The target game's lock, tz-aware.

        Returns:
            The window subset of admitted prior games for the team (may be empty).
        """
        group = team_game[team_game["team"] == team]
        fenced = group[group["_end"] <= lock].sort_values("_end")

        if len(fenced) == 0:
            return fenced

        # REUSE the team_form dynamic window (do not re-implement, D-12).
        return self._window_source._select_dynamic_window(
            fenced, target_season, target_week
        )

    def contributing_games(
        self,
        target_season: int,
        target_week: int,
        team_locks: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return the snap team-game rows that feed the rolling features.

        Every row here ENDED at or before its target game's lock (``_end``); the
        time-fence proof asserts exactly that, and this frame is the honest source of
        the builder's provenance (the latest ``_end`` a game's two teams admitted).

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.
            team_locks: ``team -> lock`` for the week's target games. When ``None``
                each team's own scheduled game in (season, week) supplies it.

        Returns:
            Concatenation of each team's window games (game_id, season, week, team,
            ``_end``), de-duplicated. Empty when no prior snaps were admitted.
        """
        columns = ["game_id", "season", "week", "team", "_end"]
        team_game = self._get_team_game()
        if len(team_game) == 0:
            return pd.DataFrame(columns=columns)
        locks = (
            dict(team_locks)
            if team_locks is not None
            else self._week_team_locks(target_season, target_week)
        )

        windows = []
        for team, lock in locks.items():
            window = self._window_for_team(
                team_game, team, target_season, target_week, lock
            )
            if len(window) > 0:
                windows.append(window[columns])

        if not windows:
            return pd.DataFrame(columns=columns)
        return pd.concat(windows, ignore_index=True).drop_duplicates(
            subset=["game_id", "team"]
        )

    def compute_team_snap_features(
        self,
        target_season: int,
        target_week: int,
        team_locks: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute backward-rolling team snap features for a target week.

        For each team, takes the prior-games-only dynamic window, applies the
        linear recency weights (D-12, REUSED shape from team_form), and produces
        weighted-mean ``snap_continuity`` / ``snap_concentration`` and per
        position-group ``rolling_snap_share_*`` columns.

        Each team's window is admitted at ITS target game's lock, so a team with no
        scheduled game in (season, week) -- a bye -- has no lock and no row.

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.
            team_locks: ``team -> lock`` for the week's target games. When ``None``
                each team's own scheduled game in (season, week) supplies it.

        Returns:
            One row per team with the rolling snap feature columns. Empty when
            no prior snaps were admitted.
        """
        team_game = self._get_team_game()
        if len(team_game) == 0:
            return pd.DataFrame()
        locks = (
            dict(team_locks)
            if team_locks is not None
            else self._week_team_locks(target_season, target_week)
        )

        share_cols = [f"snap_share_{grp}" for grp in KEY_POSITION_GROUPS]
        rows: list[dict] = []

        for team in sorted(locks):
            window = self._window_for_team(
                team_game, team, target_season, target_week, locks[team]
            )
            if len(window) == 0:
                continue

            # Linear recency weights [1, 2, ..., N] normalized (team_form:480-481).
            weights = np.arange(1, len(window) + 1, dtype=float)
            weights = weights / weights.sum()

            row: dict = {
                "team": team,
                "target_season": target_season,
                "target_week": target_week,
                "snap_games_used": len(window),
                # Measured from at least one admitted snap team-game.
                SNAP_COVERAGE_COLUMN: 1.0,
            }

            # Concentration: weighted mean over the window.
            row["snap_concentration"] = float(
                np.average(
                    window["snap_concentration"].to_numpy(dtype=float), weights=weights
                )
            )

            # Continuity: weighted mean over window games that HAVE a continuity
            # value (the first game of a team's history has none).
            cont_vals = window["snap_continuity"].to_numpy(dtype=float)
            valid = ~np.isnan(cont_vals)
            if valid.sum() > 0:
                w = weights[valid]
                w = w / w.sum()
                row["snap_continuity"] = float(np.average(cont_vals[valid], weights=w))
            else:
                row["snap_continuity"] = np.nan

            # Per-position prior shares: weighted mean of each group's share.
            for col in share_cols:
                grp = col.removeprefix("snap_share_")
                row[f"rolling_snap_share_{grp}"] = float(
                    np.average(window[col].to_numpy(dtype=float), weights=weights)
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def get_position_prior_shares(
        self, target_season: int, target_week: int
    ) -> pd.DataFrame:
        """Expose per-position prior snap shares (the D-09 availability weights).

        This is the explicit hand-off to the Plan 28-05 InjuryBuilder: the
        snap-weighted positional availability metric weights each position group
        by its prior snap share. Returns a long (team, position_group, share)
        frame so the consumer can join at the (team, position) grain.

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            Long DataFrame: team, position_group, prior_snap_share.
        """
        team_features = self.compute_team_snap_features(target_season, target_week)
        if len(team_features) == 0:
            return pd.DataFrame(columns=["team", "position_group", "prior_snap_share"])

        records = []
        for _, row in team_features.iterrows():
            for grp in KEY_POSITION_GROUPS:
                records.append(
                    {
                        "team": row["team"],
                        "position_group": grp,
                        "prior_snap_share": float(row[f"rolling_snap_share_{grp}"]),
                    }
                )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def _feature_columns(self) -> list[str]:
        """Per-team rolling feature column names (pre home/away expansion).

        ``snap_coverage`` IS IN THIS LIST ON PURPOSE: ``build_features`` emits only
        ``home_{c}`` / ``away_{c}`` for the columns named here, so a flag left off it would be
        computed and never reach gold.
        """
        return [
            "snap_continuity",
            "snap_concentration",
            *[f"rolling_snap_share_{grp}" for grp in KEY_POSITION_GROUPS],
            SNAP_COVERAGE_COLUMN,
        ]

    @staticmethod
    def _unmeasured_value(column: str) -> float:
        """What a team whose window admitted no snap game carries: NaN, its flag 0.0."""
        return 0.0 if column == SNAP_COVERAGE_COLUMN else float("nan")

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        lock_frame: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build home/away-expanded snap features, each game at its OWN lock.

        Conforms to the FeatureBuilder Protocol. Both teams' windows admit only the
        team-games that ENDED at or before the target game's lock (Plan 33.2-14).

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Carried for the ``FeatureBuilder`` Protocol ONLY. It is
                NOT a fence and no selection reads it.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.
            lock_frame: The build's ``game_id`` -> lock frame, built ONCE by the
                caller. When ``None`` it is built here from the target games.

        Returns:
            One row per game with home_/away_ snap feature columns.

        Raises:
            utils.game_lock.MissingKickoffError: a target game has no kickoff.
        """
        logger.info(
            "Building snap-count features",
            target_season=target_season,
            target_week=target_week,
        )

        self._games_cache = games_df

        feature_cols = self._feature_columns()
        empty_cols = (
            ["game_id"]
            + [f"home_{c}" for c in feature_cols]
            + [f"away_{c}" for c in feature_cols]
        )

        if len(games_df) == 0:
            return pd.DataFrame(columns=empty_cols)

        # Fan out over every (season, week) when no single target is given.
        if target_season is None or target_week is None:
            all_results = []
            season_weeks = (
                games_df[["season", "week"]]
                .drop_duplicates()
                .sort_values(["season", "week"])
            )
            locks = (
                lock_frame if lock_frame is not None else lock_rule.lock_frame(games_df)
            )
            for _, sw in season_weeks.iterrows():
                chunk = self.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=int(sw["season"]),
                    target_week=int(sw["week"]),
                    lock_frame=locks,
                )
                if len(chunk) > 0:
                    all_results.append(chunk)
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            return pd.DataFrame(columns=empty_cols)

        target_games = games_df[
            (games_df["season"] == target_season) & (games_df["week"] == target_week)
        ]
        team_features = self.compute_team_snap_features(
            target_season,
            target_week,
            team_locks=self._target_team_locks(target_games, lock_frame),
        )
        team_lookup: dict[str, dict] = {}
        if len(team_features) > 0:
            for _, row in team_features.iterrows():
                team_lookup[row["team"]] = {c: row[c] for c in feature_cols}

        rows: list[dict] = []
        for _, game in target_games.iterrows():
            record: dict = {"game_id": game["game_id"]}
            for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
                team = self._safe_normalize(game[team_col])
                team_feats = team_lookup.get(team, {})
                for col in feature_cols:
                    record[f"{prefix}_{col}"] = float(
                        team_feats.get(col, self._unmeasured_value(col))
                    )
            rows.append(record)

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

    def _target_team_locks(
        self, target_games: pd.DataFrame, lock_frame: pd.Series | None
    ) -> dict[str, Any]:
        """``team -> its target game's lock`` for one (season, week) of target games.

        The lock comes from the caller's lock frame (per-game lookup by ``game_id``),
        or from the rule applied to *target_games* when none was handed in.
        """
        if len(target_games) == 0:
            return {}
        locks = (
            lock_frame if lock_frame is not None else lock_rule.lock_frame(target_games)
        )
        team_locks: dict[str, Any] = {}
        for game in target_games.to_dict("records"):
            lock = locks[str(game["game_id"])]
            for column in ("home_team", "away_team"):
                team = self._safe_normalize(game[column])
                if team is not None:
                    team_locks[team] = lock
        return team_locks

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game whose two teams admitted no snap team-game: every value NULL, both flags 0.0."""
        return {
            f"{prefix}_{column}": (0.0 if column == SNAP_COVERAGE_COLUMN else None)
            for prefix in ("home", "away")
            for column in self._feature_columns()
        }

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game: the END of the latest snap team-game it admitted.

        DERIVED FROM :meth:`contributing_games` -- the window the features were computed
        from -- rather than from a second rule: the latest ``_end`` among both teams'
        admitted team-games at the target game's lock. A game whose teams admitted
        nothing (before snaps coverage begins in 2013, or a team's first game) is
        ``basis="no_information"`` with a NULL time, value-checked against
        :meth:`no_information_signature`.

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
        if len(target) > 0:
            locks = lock_rule.lock_frame(target)
            for key, week_games in target.groupby(["season", "week"], sort=True):
                season, week = int(key[0]), int(key[1])
                contributing = self.contributing_games(
                    season, week, self._target_team_locks(week_games, locks)
                )
                for game in week_games.to_dict("records"):
                    teams = {
                        self._safe_normalize(game["home_team"]),
                        self._safe_normalize(game["away_team"]),
                    }
                    ends = contributing.loc[contributing["team"].isin(teams), "_end"]
                    latest = pd.Timestamp(ends.max()) if len(ends) > 0 else None
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

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get home/away snap features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dict of home_/away_ snap feature values (NaN when no prior snaps).
        """
        feature_cols = self._feature_columns()
        empty = {
            f"{prefix}_{col}": self._unmeasured_value(col)
            for prefix in ("home", "away")
            for col in feature_cols
        }

        season, week, home_team, away_team = self._resolve_game(game_id)
        if season is None or week is None:
            return empty

        team_features = self.compute_team_snap_features(season, week)
        if len(team_features) == 0:
            return empty

        team_lookup = {
            row["team"]: {c: float(row[c]) for c in feature_cols}
            for _, row in team_features.iterrows()
        }

        result = dict(empty)
        for prefix, team in (("home", home_team), ("away", away_team)):
            if team is None:
                continue
            feats = team_lookup.get(self._safe_normalize(team), {})
            for col in feature_cols:
                if col in feats:
                    result[f"{prefix}_{col}"] = feats[col]
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_normalize(team: object) -> str | None:
        """Normalize a team abbreviation, returning None for missing values."""
        if team is None or (isinstance(team, float) and np.isnan(team)):
            return None
        return normalize_team_abbreviation(str(team))

    def _resolve_game(
        self, game_id: str
    ) -> tuple[int | None, int | None, str | None, str | None]:
        """Resolve (season, week, home_team, away_team) for a game_id.

        Checks the games cache first, then falls back to parsing the canonical
        ``{season}_{week}_{away}_{home}`` snap game_id form.
        """
        if self._games_cache is not None:
            match = self._games_cache[self._games_cache["game_id"] == game_id]
            if len(match) > 0:
                r = match.iloc[0]
                return (
                    int(r["season"]),
                    int(r["week"]),
                    str(r["home_team"]),
                    str(r["away_team"]),
                )

        parts = str(game_id).split("_")
        if len(parts) >= 4:
            try:
                season = int(parts[0])
                week = int(parts[1])
            except ValueError:
                return None, None, None, None
            # Snap game_id is {season}_{week}_{away}_{home}.
            away, home = parts[2], parts[3]
            return season, week, home, away

        return None, None, None, None
