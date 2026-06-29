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

- No data leakage: snaps are POST-game, so the builder fences by
  ``week < target_week`` (prior games only) -- exactly the ``team_form`` week
  idiom. Snaps carry no timestamp column, so ``check_time_fence`` is not used.
- The rolling window is REUSED from ``team_form`` (D-12): the dynamic expanding
  window + linear recency weights are NOT re-implemented here.
- Derived feature names use ``snap_continuity`` / ``snap_concentration`` /
  ``rolling_snap_share_*`` -- never ``_pct`` or the raw count spellings
  (``offense_snaps`` / ``defense_snaps`` / ``st_snaps``), so the D-13
  ``LEAKAGE_KEYWORDS`` substring guard catches only true raw leaks.
- Canonical team abbreviations via ``normalize_team_abbreviation`` (snaps back
  to 2013 carry SD / STL / OAK historical codes; hard-fail on unknowns).
"""

from datetime import datetime

import numpy as np
import pandas as pd

from data.storage import load_dataframe
from features.team_form import TeamFormCalculator
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
    ) -> None:
        """Initialize the snap-count builder.

        Args:
            snaps_df: Optional player-level snap frame to use directly (the
                test-injection / constructor-handoff seam). When ``None`` the
                builder loads ``snap_counts`` from the silver layer.
            max_prior_games: Maximum prior-season games in the rolling window
                when current-season data is sparse (matches TeamFormCalculator).
        """
        self._snaps_df = snaps_df
        self.max_prior_games = max_prior_games

        # The team_form calculator owns the dynamic window (D-12). We reuse its
        # _select_dynamic_window verbatim rather than re-implementing it.
        self._window_source = TeamFormCalculator(max_prior_games=max_prior_games)

        # Caches keyed by nothing (single load); populated lazily.
        self._team_game_cache: pd.DataFrame | None = None
        self._games_cache: pd.DataFrame | None = None

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
        df = snaps_df.copy()

        # Coerce raw snap counts to numeric; a player's participation is the sum
        # of offense + defense + special-teams snaps.
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

    def _get_team_game(self) -> pd.DataFrame:
        """Return the cached team-game aggregate frame (computed once)."""
        if self._team_game_cache is None:
            self._team_game_cache = self._team_game_aggregates(self._load_snaps())
        return self._team_game_cache

    # ------------------------------------------------------------------
    # Backward-rolling feature computation (window REUSED from team_form)
    # ------------------------------------------------------------------

    def _window_for_team(
        self,
        team_game: pd.DataFrame,
        team: str,
        target_season: int,
        target_week: int,
    ) -> pd.DataFrame:
        """Select the prior-games-only rolling window for one team.

        Fences to ``week < target_week`` (prior games only, the team_form week
        idiom) and then REUSES ``TeamFormCalculator._select_dynamic_window``
        (D-12) to pick the dynamic expanding window over those prior games.

        Args:
            team_game: Team-game aggregate frame.
            team: Canonical team abbreviation.
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            The window subset of prior games for the team (may be empty).
        """
        group = team_game[team_game["team"] == team]

        # Prior-games-only fence (snaps are post-game; fence by week, not by a
        # timestamp). Identical in shape to team_form.py:451-458.
        fenced = group[
            (group["season"] < target_season)
            | ((group["season"] == target_season) & (group["week"] < target_week))
        ].sort_values(["season", "week"])

        if len(fenced) == 0:
            return fenced

        # REUSE the team_form dynamic window (do not re-implement, D-12).
        return self._window_source._select_dynamic_window(
            fenced, target_season, target_week
        )

    def contributing_games(self, target_season: int, target_week: int) -> pd.DataFrame:
        """Return the snap team-game rows that feed the rolling features.

        Every row here is a PRIOR game (``season < target_season`` or
        ``week < target_week``); the time-fence proof asserts exactly that.

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            Concatenation of each team's window games (game_id, season, week,
            team), de-duplicated. Empty when no prior snaps exist.
        """
        team_game = self._get_team_game()
        if len(team_game) == 0:
            return pd.DataFrame(columns=["game_id", "season", "week", "team"])

        windows = []
        for team in team_game["team"].unique():
            window = self._window_for_team(team_game, team, target_season, target_week)
            if len(window) > 0:
                windows.append(window[["game_id", "season", "week", "team"]])

        if not windows:
            return pd.DataFrame(columns=["game_id", "season", "week", "team"])
        return pd.concat(windows, ignore_index=True).drop_duplicates()

    def compute_team_snap_features(
        self, target_season: int, target_week: int
    ) -> pd.DataFrame:
        """Compute backward-rolling team snap features for a target week.

        For each team, takes the prior-games-only dynamic window, applies the
        linear recency weights (D-12, REUSED shape from team_form), and produces
        weighted-mean ``snap_continuity`` / ``snap_concentration`` and per
        position-group ``rolling_snap_share_*`` columns.

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.

        Returns:
            One row per team with the rolling snap feature columns. Empty when
            no prior snaps exist.
        """
        team_game = self._get_team_game()
        if len(team_game) == 0:
            return pd.DataFrame()

        share_cols = [f"snap_share_{grp}" for grp in KEY_POSITION_GROUPS]
        rows: list[dict] = []

        for team in sorted(team_game["team"].unique()):
            window = self._window_for_team(team_game, team, target_season, target_week)
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
        """Per-team rolling feature column names (pre home/away expansion)."""
        return [
            "snap_continuity",
            "snap_concentration",
            *[f"rolling_snap_share_{grp}" for grp in KEY_POSITION_GROUPS],
        ]

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build home/away-expanded snap features for games.

        Conforms to the FeatureBuilder Protocol. The ``as_of_datetime``
        time-fence is honored structurally: snaps have no timestamp, so the
        builder only ever consumes PRIOR games (``week < target_week``), which
        are by construction before the as-of cutoff for the target week.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff (informational for snaps; the
                prior-week fence is the binding control).
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.

        Returns:
            One row per game with home_/away_ snap feature columns.
        """
        logger.info(
            "Building snap-count features",
            as_of_datetime=str(as_of_datetime),
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
            for _, sw in season_weeks.iterrows():
                chunk = self.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=int(sw["season"]),
                    target_week=int(sw["week"]),
                )
                if len(chunk) > 0:
                    all_results.append(chunk)
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            return pd.DataFrame(columns=empty_cols)

        team_features = self.compute_team_snap_features(target_season, target_week)
        team_lookup: dict[str, dict] = {}
        if len(team_features) > 0:
            for _, row in team_features.iterrows():
                team_lookup[row["team"]] = {c: row[c] for c in feature_cols}

        target_games = games_df[
            (games_df["season"] == target_season) & (games_df["week"] == target_week)
        ]

        rows: list[dict] = []
        for _, game in target_games.iterrows():
            record: dict = {"game_id": game["game_id"]}
            for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
                team = self._safe_normalize(game[team_col])
                team_feats = team_lookup.get(team, {})
                for col in feature_cols:
                    record[f"{prefix}_{col}"] = float(team_feats.get(col, np.nan))
            rows.append(record)

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

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
            f"{prefix}_{col}": float("nan")
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
