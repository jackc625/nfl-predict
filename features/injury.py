"""Injury Signal Feature Builder (SIG-01).

This module exposes the weekly-final injury silver table (produced by the Plan
28-02 ingest) as three leakage-safe, pre-game team features, all fenced to the
Friday 6 PM ET freeze:

- ``qb_out_flag`` -- 1.0 when the team's depth-chart QB1 is listed ``Out`` or
  ``Doubtful`` in the Friday-fenced report, else 0.0 (D-07).
- ``backup_quality_delta`` -- ``quality(QB1) - quality(QB2)`` on the REUSED
  QBTracker quality scale (D-06/D-08); 0.0 when the starter is not out. A
  no-history backup is valued at the EXPLICIT below-average
  ``REPLACEMENT_LEVEL_QB_QUALITY`` constant, never a silent 0.0 mislabeled
  "replacement-level" (review bonus, qb_tracking.py:385).
- ``availability_fraction`` -- snap-weighted positional availability at the
  (team, position) grain (D-09), weighting each key non-QB position group by the
  SnapCountBuilder's per-position PRIOR snap share.

The load-bearing temporal control is the per-row ``date_modified <=
as_of_datetime`` fence (D-07): only injury reports KNOWN at/before the Friday
freeze enter a feature, so the ~7.5% Saturday/Sunday game-day updates (the SC1
leakage event) are excluded. The fence is tolerant of a dropped ``date_modified``
column (2025+, Pitfall 4): it falls back to a week-based snapshot fence and sets
a ``*_coverage`` flag instead of raising a ``KeyError``.

Snaps -> injuries dependency (review #6): the per-position snap-share availability
weights are obtained from a CONSTRUCTED ``SnapCountBuilder`` handed in at
construction (``InjuryBuilder(snap_builder=...)``), NOT an implicit independent
re-load of the snap silver. This LOCKS the snaps-before-injuries build order
(Plan 28-06 must construct ``SnapCountBuilder()`` first, then
``InjuryBuilder(snap_builder=self.snap_builder)``). The FeatureBuilder Protocol
``build_features`` signature is unchanged -- the dependency is injected, not
threaded through a new argument.

The backup-quality delta REUSES QBTracker's ``compute_composite_quality`` /
rolling-metric quality path (D-06); this builder authors NO new EPA/CPOE quality
metric.

Per D-10, the builder emits neutral defaults (no QB out = 0.0, availability =
1.0) plus ``*_coverage`` flags marking gaps, so the paired-lift population stays
intact rather than dropping rows.

Key constraints:

- No data leakage: the binding control is ``date_modified <= as_of_datetime``;
  availability reads the ``report_status`` column (there is no game-day inactive
  status field to lean on).
- Team x position availability grain deliberately avoids the pfr_player_id (snaps)
  vs gsis_id (injuries) cross-ID bridge that would silently zero-match (Pitfall 5).
- Canonical team abbreviations via ``normalize_team_abbreviation`` (reused through
  QBTracker's ``_safe_normalize_team``); hard-fail-tolerant on unknowns.
"""

from collections import defaultdict
from datetime import datetime

import pandas as pd

from data.storage import load_dataframe
from features.qb_tracking import QBTracker
from features.snaps import KEY_POSITION_GROUPS, POSITION_GROUP_MAP, SnapCountBuilder
from utils import get_logger

logger = get_logger(__name__)

# Report-status values that mark a player as NOT available pre-game. Read from
# the `report_status` column (the silver vocabulary is {None, Questionable, Out,
# Doubtful}); there is no game-day inactive status field in the silver schema.
OUT_STATUSES = frozenset({"Out", "Doubtful"})

# EXPLICIT below-average backup-QB quality default (review bonus). QBTracker's
# `compute_composite_quality` returns 0.0 = LEAGUE AVERAGE on no history
# (qb_tracking.py:385); a true replacement-level backup is BELOW average, so
# defaulting an unknown backup to 0.0 would UNDERSTATE the downgrade and shrink
# `backup_quality_delta`. This constant is a CHOSEN, not-tuned, in-scale value
# (~0.5 z-score below the league-average 0.0 on QBTracker's normalized quality
# scale) -- the same "defensible chosen starting point" posture as the D-16 Elo
# spot step. It is VISIBLE here and in the docstring, never a silent mislabeled
# 0.0.
REPLACEMENT_LEVEL_QB_QUALITY = -0.5

# Per-Out/Doubtful-player availability hit within a position group (D-09). Each
# unavailable player in a key group reduces that group's availability by this
# fraction, capped at a full (1.0) group loss. A CHOSEN, not-tuned constant: at
# 0.25, it takes four simultaneously-out players in one group to zero that
# group's availability -- a deliberately conservative, interpretable scale at the
# coarse team x position grain (no per-player snap weighting, which would require
# the forbidden pfr/gsis bridge).
OUT_PLAYER_AVAILABILITY_WEIGHT = 0.25

# Key non-QB position groups whose availability the metric aggregates. The QB
# group is excluded here because the QB signal is carried separately by
# `qb_out_flag` + `backup_quality_delta`.
_NON_QB_KEY_GROUPS = [grp for grp in KEY_POSITION_GROUPS if grp != "qb"]

# Per-team feature columns (pre home/away expansion).
_FEATURE_COLUMNS = [
    "qb_out_flag",
    "backup_quality_delta",
    "availability_fraction",
    "injury_coverage",
    "availability_coverage",
    "date_modified_coverage",
]


class InjuryBuilder:
    """Build leakage-safe pre-game injury features from the injury silver.

    Conforms to the FeatureBuilder Protocol (``build_features`` with the
    ``as_of_datetime`` Friday-freeze fence + ``get_features_for_game``). Emits
    per game the home/away-expanded ``qb_out_flag`` / ``backup_quality_delta`` /
    ``availability_fraction`` columns plus ``*_coverage`` flags (D-10).

    The snaps -> injuries dependency is LOCKED at construction (review #6): the
    per-position prior snap shares (the D-09 availability weights) come from a
    constructed :class:`SnapCountBuilder` passed via ``snap_builder=...``, never
    an implicit re-load. The backup delta REUSES :class:`QBTracker`'s quality
    path (D-06).
    """

    def __init__(
        self,
        snap_builder: SnapCountBuilder,
        *,
        injuries_df: pd.DataFrame | None = None,
        depth_charts_df: pd.DataFrame | None = None,
        pbp_df: pd.DataFrame | None = None,
        qb_tracker: QBTracker | None = None,
    ) -> None:
        """Initialize the injury builder.

        Args:
            snap_builder: The CONSTRUCTED :class:`SnapCountBuilder` (Plan 28-03).
                Its :meth:`SnapCountBuilder.get_position_prior_shares` supplies
                the D-09 availability weights via constructor handoff (review #6),
                locking the snaps-before-injuries build order. Required.
            injuries_df: Optional player-level injury frame to use directly (the
                test-injection seam). When ``None`` the builder loads
                ``injuries`` from the silver layer.
            depth_charts_df: Optional depth-chart frame for QB1/QB2 resolution
                (test-injection seam). When ``None`` the builder loads depth
                charts per season through the reused QBTracker loader.
            pbp_df: Optional play-by-play frame for the backup-quality rolling
                metrics (test-injection seam). When ``None`` the builder loads
                PBP per season through the reused QBTracker loader.
            qb_tracker: Optional :class:`QBTracker` to reuse for the quality
                scale + depth-chart/PBP loaders (D-06). Defaults to a fresh
                ``QBTracker()``.
        """
        if snap_builder is None:
            raise ValueError(
                "InjuryBuilder requires a constructed SnapCountBuilder "
                "(snap_builder=...) -- the snaps->injuries handoff is locked at "
                "the constructor (review #6)."
            )
        self.snap_builder = snap_builder
        self._injuries_df = injuries_df
        self._depth_charts_df = depth_charts_df
        self._pbp_df = pbp_df
        self._qb_tracker = qb_tracker if qb_tracker is not None else QBTracker()

        self._games_cache: pd.DataFrame | None = None
        # Cache rolling QB metrics per season so the quality path runs once.
        self._rolling_cache: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Silver loading + Friday fence
    # ------------------------------------------------------------------

    def _load_injuries(self) -> pd.DataFrame:
        """Load the player-level injury silver (or the injected frame)."""
        if self._injuries_df is not None:
            return self._injuries_df
        return load_dataframe("injuries", layer="silver")

    def fenced_injuries(
        self,
        target_season: int,
        target_week: int,
        as_of_datetime: datetime,
    ) -> tuple[pd.DataFrame, bool]:
        """Return the week's injury rows known at/before the Friday freeze.

        Filters to the target (season, week) weekly-final report, then applies
        the load-bearing ``date_modified <= as_of_datetime`` fence (D-07) so the
        Saturday/Sunday game-day updates are excluded. Tolerant of a missing /
        all-null ``date_modified`` column (2025+, Pitfall 4): falls back to the
        week-based snapshot report and reports ``date_modified_applied=False``.

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.
            as_of_datetime: The Friday 6 PM ET freeze cutoff.

        Returns:
            ``(fenced_rows, date_modified_applied)`` -- the fenced injury rows
            for the week and whether the per-row date fence was applied.
        """
        inj = self._load_injuries()
        if inj is None or len(inj) == 0:
            return inj if inj is not None else pd.DataFrame(), False

        week_rows = inj[
            (inj["season"] == target_season) & (inj["week"] == target_week)
        ].copy()
        if len(week_rows) == 0:
            return week_rows, False

        has_dm = (
            "date_modified" in week_rows.columns
            and week_rows["date_modified"].notna().any()
        )
        if not has_dm:
            # 2025+ drop / all-null: best available is the weekly-final report.
            return week_rows, False

        dm = pd.to_datetime(week_rows["date_modified"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(as_of_datetime)
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")

        # Rows with an unparseable date_modified cannot be proven pre-freeze and
        # are excluded (conservative: never admit an un-fenceable row).
        fenced = week_rows[dm <= cutoff]
        return fenced, True

    # ------------------------------------------------------------------
    # QB1 / QB2 depth-chart resolution (reuses the QBTracker loaders)
    # ------------------------------------------------------------------

    def _load_depth_charts(self, season: int) -> pd.DataFrame:
        """Load depth charts for a season (injected frame or QBTracker loader)."""
        if self._depth_charts_df is not None:
            return self._depth_charts_df
        return self._qb_tracker._load_depth_charts(season)

    def _resolve_qb_ids(
        self, season: int, week: int, team: str
    ) -> tuple[str | None, str | None]:
        """Resolve a team's QB1 and QB2 gsis_ids from depth charts (D-07).

        Reuses QBTracker's depth-chart loader + team normalization. QB1 is
        ``depth_team == "1"``, QB2 is ``depth_team == "2"`` for ``position ==
        "QB"`` in the target (season, week).

        Returns:
            ``(qb1_gsis_id, qb2_gsis_id)`` -- either may be None when absent.
        """
        dc = self._load_depth_charts(season)
        if dc is None or len(dc) == 0 or "club_code" not in dc.columns:
            return None, None

        qbs = dc[dc["position"] == "QB"].copy()
        if "season" in qbs.columns:
            qbs = qbs[qbs["season"] == season]
        if "week" in qbs.columns:
            qbs = qbs[qbs["week"] == week]
        if len(qbs) == 0:
            return None, None

        qbs["_team"] = qbs["club_code"].apply(self._qb_tracker._safe_normalize_team)
        qbs = qbs[qbs["_team"] == team]
        if len(qbs) == 0:
            return None, None

        depth = qbs["depth_team"].astype(str)
        qb1 = qbs[depth == "1"]
        qb2 = qbs[depth == "2"]
        qb1_id = str(qb1.iloc[0]["gsis_id"]) if len(qb1) > 0 else None
        qb2_id = str(qb2.iloc[0]["gsis_id"]) if len(qb2) > 0 else None
        return qb1_id, qb2_id

    # ------------------------------------------------------------------
    # Backup-quality delta (REUSES the QBTracker quality scale, D-06)
    # ------------------------------------------------------------------

    def _rolling_qb(self, season: int, week: int) -> pd.DataFrame:
        """Compute (and cache) the reused QBTracker rolling quality for a season."""
        if season in self._rolling_cache:
            return self._rolling_cache[season]
        if self._pbp_df is not None:
            pbp = self._pbp_df
        else:
            pbp = self._qb_tracker._load_pbp_data(season)
        rolling = self._qb_tracker.compute_rolling_qb_metrics(pbp, season, week)
        self._rolling_cache[season] = rolling
        return rolling

    def _quality_for(
        self, rolling: pd.DataFrame, gsis_id: str | None, default: float
    ) -> float:
        """Look up a QB's reused-quality value, falling back to ``default``."""
        if gsis_id is None or len(rolling) == 0:
            return default
        match = rolling[rolling["passer_player_id"] == gsis_id]
        if len(match) == 0:
            return default
        return float(match.iloc[0]["qb_quality"])

    def _backup_quality_delta(
        self,
        season: int,
        week: int,
        qb1_id: str | None,
        qb2_id: str | None,
    ) -> float:
        """``quality(QB1) - quality(QB2)`` on the reused QBTracker scale (D-08).

        Only called when the starter is out. A no-history backup is valued at the
        EXPLICIT below-average ``REPLACEMENT_LEVEL_QB_QUALITY`` (never a silent
        league-average 0.0); the starter falls back to 0.0 (league average) when
        their rolling history is unavailable.
        """
        rolling = self._rolling_qb(season, week)
        starter_quality = self._quality_for(rolling, qb1_id, default=0.0)
        backup_quality = self._quality_for(
            rolling, qb2_id, default=REPLACEMENT_LEVEL_QB_QUALITY
        )
        return float(starter_quality - backup_quality)

    # ------------------------------------------------------------------
    # Snap-weighted positional availability (D-09, team x position grain)
    # ------------------------------------------------------------------

    @staticmethod
    def _out_counts_by_group(team_injuries: pd.DataFrame) -> dict[str, int]:
        """Count Out/Doubtful players per key non-QB position group for a team."""
        counts: dict[str, int] = defaultdict(int)
        for _, row in team_injuries.iterrows():
            status = row.get("report_status")
            if status not in OUT_STATUSES:
                continue
            grp = POSITION_GROUP_MAP.get(
                str(row.get("position")).upper().strip(), "other"
            )
            if grp in _NON_QB_KEY_GROUPS:
                counts[grp] += 1
        return dict(counts)

    def _availability_fraction(
        self,
        prior_shares: dict[str, float],
        out_counts: dict[str, int],
    ) -> tuple[float, bool]:
        """Snap-weighted availability over key non-QB groups (D-09).

        ``Sigma_g w_g * available_g`` where ``w_g`` is the team's prior per-group
        snap share (normalized over the non-QB key groups so a fully-healthy team
        scores 1.0) and ``available_g = 1 - min(1, n_out_g *
        OUT_PLAYER_AVAILABILITY_WEIGHT)``. The (team, position) grain avoids the
        pfr/gsis cross-ID bridge (Pitfall 5).

        Returns:
            ``(availability_fraction, has_share_coverage)``. When no prior snap
            shares exist the neutral 1.0 default is returned with coverage False.
        """
        weights = {grp: prior_shares.get(grp, 0.0) for grp in _NON_QB_KEY_GROUPS}
        total_w = sum(weights.values())
        if total_w <= 0:
            # No prior snap-share weights -> neutral default, flagged as a gap.
            return 1.0, False

        availability = 0.0
        for grp in _NON_QB_KEY_GROUPS:
            w = weights[grp] / total_w
            n_out = out_counts.get(grp, 0)
            available_g = 1.0 - min(1.0, n_out * OUT_PLAYER_AVAILABILITY_WEIGHT)
            availability += w * available_g
        return float(availability), True

    # ------------------------------------------------------------------
    # Per-(season, week) feature assembly
    # ------------------------------------------------------------------

    def compute_injury_features(
        self,
        target_season: int,
        target_week: int,
        as_of_datetime: datetime,
        teams: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute per-team injury features for one target (season, week).

        Args:
            target_season: Season being predicted.
            target_week: Week being predicted.
            as_of_datetime: The Friday 6 PM ET freeze cutoff.
            teams: Canonical team abbreviations to build features for.

        Returns:
            ``{team: {feature: value}}`` with neutral defaults + coverage flags.
        """
        fenced, dm_applied = self.fenced_injuries(
            target_season, target_week, as_of_datetime
        )
        week_has_data = 1.0 if len(fenced) > 0 else 0.0
        dm_coverage = 1.0 if dm_applied else 0.0

        # Per-position prior snap shares (the D-09 availability weights) come from
        # the constructor-injected SnapCountBuilder, NOT an independent re-load.
        prior = self.snap_builder.get_position_prior_shares(target_season, target_week)
        team_shares: dict[str, dict[str, float]] = defaultdict(dict)
        if prior is not None and len(prior) > 0:
            for _, row in prior.iterrows():
                team_shares[str(row["team"])][str(row["position_group"])] = float(
                    row["prior_snap_share"]
                )

        results: dict[str, dict[str, float]] = {}
        for team in teams:
            team_inj = (
                fenced[
                    fenced["team"].apply(self._qb_tracker._safe_normalize_team) == team
                ]
                if len(fenced) > 0
                else fenced
            )

            qb1_id, qb2_id = self._resolve_qb_ids(target_season, target_week, team)
            out_ids = (
                set(team_inj[team_inj["report_status"].isin(OUT_STATUSES)]["gsis_id"])
                if len(team_inj) > 0
                else set()
            )
            qb_out = 1.0 if (qb1_id is not None and qb1_id in out_ids) else 0.0

            backup_delta = (
                self._backup_quality_delta(target_season, target_week, qb1_id, qb2_id)
                if qb_out == 1.0
                else 0.0
            )

            out_counts = (
                self._out_counts_by_group(team_inj) if len(team_inj) > 0 else {}
            )
            availability, has_share = self._availability_fraction(
                team_shares.get(team, {}), out_counts
            )

            results[team] = {
                "qb_out_flag": qb_out,
                "backup_quality_delta": float(backup_delta),
                "availability_fraction": float(availability),
                "injury_coverage": week_has_data,
                "availability_coverage": 1.0 if has_share else 0.0,
                "date_modified_coverage": dm_coverage,
            }
        return results

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
        """Build home/away-expanded injury features for games.

        Conforms to the FeatureBuilder Protocol. The binding time-fence is the
        per-row ``date_modified <= as_of_datetime`` filter applied inside
        :meth:`fenced_injuries` (D-07): only reports known at/before the Friday
        freeze enter a feature.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: The Friday 6 PM ET freeze cutoff.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.

        Returns:
            One row per game with ``home_`` / ``away_`` injury feature columns.
        """
        logger.info(
            "Building injury features",
            as_of_datetime=str(as_of_datetime),
            target_season=target_season,
            target_week=target_week,
        )

        self._games_cache = games_df

        empty_cols = (
            ["game_id"]
            + [f"home_{c}" for c in _FEATURE_COLUMNS]
            + [f"away_{c}" for c in _FEATURE_COLUMNS]
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

        target_games = games_df[
            (games_df["season"] == target_season) & (games_df["week"] == target_week)
        ]
        if len(target_games) == 0:
            return pd.DataFrame(columns=empty_cols)

        teams: list[str] = []
        for _, game in target_games.iterrows():
            for col in ("home_team", "away_team"):
                team = self._qb_tracker._safe_normalize_team(game[col])
                if team not in teams:
                    teams.append(team)

        team_features = self.compute_injury_features(
            target_season, target_week, as_of_datetime, teams
        )

        rows: list[dict] = []
        for _, game in target_games.iterrows():
            record: dict = {"game_id": game["game_id"]}
            for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
                team = self._qb_tracker._safe_normalize_team(game[team_col])
                feats = team_features.get(team, self._neutral_features())
                for col in _FEATURE_COLUMNS:
                    record[f"{prefix}_{col}"] = float(feats[col])
            rows.append(record)

        return pd.DataFrame(rows)

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get home/away injury features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: The Friday 6 PM ET freeze cutoff.

        Returns:
            Dict of ``home_`` / ``away_`` injury feature values.
        """
        empty = {
            f"{prefix}_{col}": float(self._neutral_features()[col])
            for prefix in ("home", "away")
            for col in _FEATURE_COLUMNS
        }

        season, week, home_team, away_team = self._resolve_game(game_id)
        if season is None or week is None:
            return empty

        home = self._qb_tracker._safe_normalize_team(home_team)
        away = self._qb_tracker._safe_normalize_team(away_team)
        team_features = self.compute_injury_features(
            season, week, as_of_datetime, [home, away]
        )

        result = dict(empty)
        for prefix, team in (("home", home), ("away", away)):
            feats = team_features.get(team, self._neutral_features())
            for col in _FEATURE_COLUMNS:
                result[f"{prefix}_{col}"] = float(feats[col])
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_features() -> dict[str, float]:
        """Neutral defaults (D-10): no QB out, full availability, no coverage."""
        return {
            "qb_out_flag": 0.0,
            "backup_quality_delta": 0.0,
            "availability_fraction": 1.0,
            "injury_coverage": 0.0,
            "availability_coverage": 0.0,
            "date_modified_coverage": 0.0,
        }

    def _resolve_game(
        self, game_id: str
    ) -> tuple[int | None, int | None, str | None, str | None]:
        """Resolve (season, week, home_team, away_team) for a game_id."""
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
            away, home = parts[2], parts[3]
            return season, week, home, away

        return None, None, None, None
