"""Injury Signal Feature Builder (SIG-01), fenced at EACH GAME'S OWN LOCK (Plan 33.2-13).

This module exposes the weekly-final injury silver table (produced by the Plan 28-02
ingest) as three pre-game team features:

- ``qb_out_flag`` -- 1.0 when the team's depth-chart QB1 is listed ``Out`` or
  ``Doubtful`` in the latest report admitted at the lock, else 0.0 (D-07).
- ``backup_quality_delta`` -- ``quality(QB1) - quality(QB2)`` on the REUSED QBTracker
  quality scale (D-06/D-08); 0.0 when the starter is not out. A no-history backup is valued
  at the EXPLICIT below-average ``REPLACEMENT_LEVEL_QB_QUALITY`` constant, never a silent
  0.0 mislabeled "replacement-level" (review bonus, qb_tracking.py:385).
- ``availability_fraction`` -- snap-weighted positional availability at the (team,
  position) grain (D-09), weighting each key non-QB position group by the
  SnapCountBuilder's per-position PRIOR snap share.

THE FENCE IS THE PER-GAME LOCK (D33.2-01, SPEC R5)
--------------------------------------------------
A report is admitted for a game only when its information time is AT or BEFORE that
game's lock -- 18:00 America/New_York on the ET calendar day before kickoff
(``utils.game_lock``). The retired fence compared every row with ONE frame-wide
``as_of_datetime``, which is ``datetime.now(ET)`` in production and therefore admitted
everything that had already happened. ``as_of_datetime`` survives only because the
``FeatureBuilder`` Protocol carries it; no selection reads it.

A row's information time, and the SELECTION RULE that admitted it:

* ``date_modified`` -- the per-row upstream modification time (2010-2024 in full, a
  handful of 2009 rows). The honest per-row time wherever it exists.
* ``capture_stamp`` -- the frame column named by :data:`UPSTREAM_CAPTURE_COLUMN`: the
  upstream publication time of the file a row was captured from (D33.2-16). It admits a
  row only when AT or BEFORE the game's lock, which is the forward daily-capture case.
  A season fetched after its games carries a stamp after every lock and admits nothing.
* otherwise the row is UNDATABLE and is NOT ADMITTED -- exactly as a post-lock row is not.

A game that admitted nothing is ``none_admitted``: its values are NaN -- the honest
unknown -- with the ``*_coverage`` flags at 0.0, and its provenance is
``basis="no_information"`` with a NULL time, which the information-time gate CHECKS against
:meth:`InjuryBuilder.no_information_signature` rather than believes (RESEARCH P2).

WHY NaN AND NOT THE OLD NEUTRAL DEFAULTS (Plan 33.2-17 Task 2, D33.2-08 item 2). A team with no
admitted report used to read "no QB out, no backup delta, full availability" -- 0.0 / 0.0 /
1.0. A 0.0 in a z-scored or centred feature reads as "exactly average", a confident claim about
a game nobody measured, and before the feed's first covered season it filled every one of these
columns. The flags already said "unknown"; the values now agree with them. The first covered
season is DERIVED from the data (no admitted report before the feed begins), never a literal.
The same rule covers ``availability_fraction`` when no prior snap shares exist to weight it:
NaN beside ``availability_coverage`` 0.0, never the neutral 1.0.

WHAT THIS REPLACED, AND WHY IT IS NOT A WEEK JOIN ANY MORE. Until Plan 33.2-13 a week whose
rows carried no ``date_modified`` fell back to the week's report UNFENCED and returned a
flag saying the date fence had not been applied -- a value that looked computed but was not
fenced. That fallback is deleted. An undatable row is simply not admitted.

THE 2025+ SCHEMA CARRIES NO PER-ROW TIME AT ALL. Measured 2026-09-15 (RESEARCH 5.1):
``injuries_2025.parquet`` (6,068 rows) and ``injuries_2026.parquet`` have NO
``date_modified`` column -- upstream dropped it. ``nflreadr``'s own documentation still says
the injury source "died after the 2024 season", and that page is stale (the 2026 asset
refreshed at 12:38 UTC on 2026-09-15), but the source visibly changed, which is what dropped
the column. Upstream injury availability is therefore FRAGILE, and its absence is a flagged
unknown, never a zero. The real cost, stated rather than worked around: for 2025 and 2026
games already played, nobody captured the file before their locks, so the injury family
reads as unknown-with-a-flag for them. Plan 33.2-14's rung 5 predicts that movement.

THE CAPTURE BRANCH READS A FRAME COLUMN, NOT A MODULE. The stamp arrives as DATA on the
injury frame, so this module imports nothing from the capture machinery Plan 33.2-15
builds; Plan 33.2-15 declares the same column on the real ``InjurySchema`` and asserts the
two names are identical (D30-02: one literal, one checked tie).

Snaps -> injuries dependency (review #6): the per-position snap-share availability weights
come from a CONSTRUCTED ``SnapCountBuilder`` handed in at construction
(``InjuryBuilder(snap_builder=...)``), which LOCKS the snaps-before-injuries build order.

The backup-quality delta REUSES QBTracker's quality path (D-06), fed only play-by-play from
games whose result was known at the lock; QB1 / QB2 come from QBTracker's lock-aware
depth-chart resolver, so a 2025+ chart published after the lock is not seen either.

Key constraints:

- Team x position availability grain deliberately avoids the pfr_player_id (snaps) vs
  gsis_id (injuries) cross-ID bridge that would silently zero-match (Pitfall 5).
- Canonical team abbreviations via ``normalize_team_abbreviation`` (reused through
  QBTracker's ``_safe_normalize_team``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import pandas as pd

import utils.game_lock as lock_rule
from data.storage import load_dataframe
from features.provenance import PROVENANCE_COLUMNS, InformationBasis
from features.qb_tracking import QBTracker, lock_as_utc, to_aware_utc
from features.snaps import KEY_POSITION_GROUPS, POSITION_GROUP_MAP, SnapCountBuilder
from utils import get_logger

logger = get_logger(__name__)

#: THE INPUT CONTRACT for the capture-time selection rule: the column the injury ingest
#: writes each captured row's upstream publication stamp into (tz-aware). Plan 33.2-15
#: declares the same name on ``data.schemas.InjurySchema`` and a test there asserts the two
#: are identical, so the builder and the ingest share one literal plus one checked tie.
UPSTREAM_CAPTURE_COLUMN: str = "upstream_captured_at"

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

#: The per-team columns this builder emits, public so the gold build can lay out an EMPTY
#: injury source frame (Plan 33.2-15's empty-source-family guard in scripts/build_features.py)
#: from the builder's own list rather than a second hand-kept one.
INJURY_FEATURE_COLUMNS: tuple[str, ...] = tuple(_FEATURE_COLUMNS)

# The per-team values a team that admitted NO report carries: the honest unknown -- NaN in
# every value column -- with the injury coverage flags at 0.0 (Plan 33.2-17 Task 2; these were
# the neutral defaults 0.0 / 0.0 / 1.0, read by a model as "exactly average").
# ``availability_coverage`` is NOT here: it reports whether PRIOR SNAP SHARES exist, which is a
# snap fact, not an injury one.
_UNKNOWN_TEAM_VALUES: dict[str, float] = {
    "qb_out_flag": float("nan"),
    "backup_quality_delta": float("nan"),
    "availability_fraction": float("nan"),
    "injury_coverage": 0.0,
    "date_modified_coverage": 0.0,
}

#: The injury columns whose unknown is NaN (the value columns); the rest are flags.
_UNKNOWN_IS_NULL: frozenset[str] = frozenset(
    column for column, value in _UNKNOWN_TEAM_VALUES.items() if math.isnan(value)
)


class SelectionRule(StrEnum):
    """Which admission rule applied to a game's injury reports. Builder-internal.

    NOT the provenance basis. ``features.provenance.InformationBasis`` has exactly two
    members; :meth:`InjuryBuilder.information_times` maps ``date_modified`` and
    ``capture_stamp`` onto ``per_row`` and ``none_admitted`` onto ``no_information``.
    These names only let a test tell the two dated paths apart.
    """

    DATE_MODIFIED = "date_modified"
    CAPTURE_STAMP = "capture_stamp"
    NONE_ADMITTED = "none_admitted"


@dataclass(frozen=True)
class InjurySelection:
    """The injury reports one game was allowed to see, and how they were admitted.

    Attributes:
        rows: The admitted reports, ONE per (team, player): the latest admitted at or
            before the lock, not the latest overall.
        rule: The selection rule of the report that sets :attr:`information_time`, or
            ``NONE_ADMITTED`` when nothing was admitted.
        information_time: The latest information time among the admitted reports (the
            provenance time), or ``None`` when nothing was admitted.
        undatable_teams: Teams with at least one report excluded because it carried no
            information time at all -- their dated coverage is incomplete.
    """

    rows: pd.DataFrame
    rule: SelectionRule
    information_time: pd.Timestamp | None
    undatable_teams: frozenset[str]


class InjuryBuilder:
    """Build pre-game injury features, each game fenced at its own lock.

    Conforms to the ``FeatureBuilder`` Protocol (``build_features`` +
    ``get_features_for_game``) AND to ``features.protocol.InformationTimeProvider``
    (``information_times`` + ``no_information_signature``), so the information-time gate
    checks it on every build.

    The snaps -> injuries dependency is LOCKED at construction (review #6). The backup delta
    REUSES :class:`QBTracker`'s quality path (D-06).
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
                ``injuries`` from the silver layer, once.
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
        # The injury frame, loaded once and grouped by (season, week).
        self._injuries_by_week: dict[tuple[int, int], pd.DataFrame] | None = None
        # Rolling QB quality per (season, week, lock): the admitted play-by-play differs by
        # lock, so a per-season cache would reuse one week's window for every week.
        self._rolling_cache: dict[tuple[int, int, pd.Timestamp], pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Silver loading + the per-game lock fence
    # ------------------------------------------------------------------

    def _load_injuries(self) -> pd.DataFrame:
        """Load the player-level injury silver (or the injected frame)."""
        if self._injuries_df is not None:
            return self._injuries_df
        return load_dataframe("injuries", layer="silver")

    def _week_rows(self, season: int, week: int) -> pd.DataFrame:
        """The injury rows filed for one (season, week), from a frame loaded once."""
        if self._injuries_by_week is None:
            injuries = self._load_injuries()
            if injuries is None or len(injuries) == 0:
                self._injuries_by_week = {}
            else:
                grouped = injuries.groupby(["season", "week"])
                self._injuries_by_week = {
                    (int(key[0]), int(key[1])): rows
                    for key, rows in ((cast(tuple[int, int], k), r) for k, r in grouped)
                }
        return self._injuries_by_week.get((int(season), int(week)), pd.DataFrame())

    def fenced_injuries(self, game: Mapping[str, Any], lock: Any) -> InjurySelection:
        """The injury reports *game* may see: those known at or before ITS lock.

        Replaces both the frame-wide ``date_modified <= as_of_datetime`` fence and the
        week-keyed fallback that stood beside it. That fallback returned a week's rows
        UNFENCED whenever ``date_modified`` was absent and reported the fact through a
        ``date_modified_applied`` flag in a ``(rows, flag)`` tuple; it is gone, and so is
        the tuple. Now each row's information time is its ``date_modified`` or, failing
        that, the capture stamp in :data:`UPSTREAM_CAPTURE_COLUMN`; a row with neither is
        UNDATABLE and is NOT ADMITTED, exactly like a row timed after the lock. Among the
        admitted rows, each player's LATEST report wins.

        Args:
            game: The target game (``season``, ``week``, ``home_team``, ``away_team``).
            lock: The target game's lock (``utils.game_lock``), tz-aware.

        Returns:
            The admitted reports, the selection rule and the provenance time.
        """
        season, week = int(game["season"]), int(game["week"])
        teams = {
            self._qb_tracker._safe_normalize_team(game[column])
            for column in ("home_team", "away_team")
        }
        week_rows = self._week_rows(season, week)
        if len(week_rows) == 0:
            return InjurySelection(
                week_rows, SelectionRule.NONE_ADMITTED, None, frozenset()
            )

        normalized = week_rows["team"].map(self._qb_tracker._safe_normalize_team)
        rows = week_rows.loc[normalized.isin(sorted(teams))].copy()
        rows["_team"] = normalized.loc[rows.index]
        if len(rows) == 0:
            return InjurySelection(rows, SelectionRule.NONE_ADMITTED, None, frozenset())

        times, rules = self._row_information_times(rows)
        undatable_teams = frozenset(rows.loc[times.isna(), "_team"])
        lock_utc = lock_as_utc(lock)
        # The at-lock-admissible comparison utils.game_lock.is_admissible states (<=),
        # vectorised over this game's candidate rows against its one lock.
        admitted_mask = times.notna() & (times <= lock_utc)
        admitted = rows.loc[admitted_mask].copy()
        if len(admitted) == 0:
            return InjurySelection(
                admitted, SelectionRule.NONE_ADMITTED, None, undatable_teams
            )

        admitted["_information_time"] = times.loc[admitted.index]
        admitted["_selection_rule"] = rules.loc[admitted.index]
        latest = (
            admitted.sort_values("_information_time", kind="stable")
            .groupby(["_team", "gsis_id"], dropna=False, sort=False)
            .tail(1)
        )
        top = latest.sort_values("_information_time", kind="stable").iloc[-1]
        return InjurySelection(
            rows=latest,
            rule=SelectionRule(top["_selection_rule"]),
            information_time=cast(pd.Timestamp, pd.Timestamp(top["_information_time"])),
            undatable_teams=undatable_teams,
        )

    @staticmethod
    def _row_information_times(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Each row's information time and the selection rule that supplies it.

        ``date_modified`` wins where present; the capture stamp covers a row only where
        ``date_modified`` is absent. A naive value in either column is refused.
        """
        if "date_modified" in rows.columns:
            modified = to_aware_utc(
                pd.Series(rows["date_modified"]), column="date_modified"
            )
        else:
            modified = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
        if UPSTREAM_CAPTURE_COLUMN in rows.columns:
            captured = to_aware_utc(
                pd.Series(rows[UPSTREAM_CAPTURE_COLUMN]), column=UPSTREAM_CAPTURE_COLUMN
            )
        else:
            captured = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")

        times = modified.where(modified.notna(), captured)
        rules = pd.Series(None, index=rows.index, dtype="object")
        rules.loc[modified.notna()] = SelectionRule.DATE_MODIFIED.value
        rules.loc[modified.isna() & captured.notna()] = (
            SelectionRule.CAPTURE_STAMP.value
        )
        return times, rules

    # ------------------------------------------------------------------
    # QB1 / QB2 depth-chart resolution (reuses QBTracker's lock-aware resolver)
    # ------------------------------------------------------------------

    def _load_depth_charts(self, season: int) -> pd.DataFrame:
        """Load depth charts for a season (injected frame or QBTracker loader)."""
        if self._depth_charts_df is not None:
            return self._depth_charts_df
        return self._qb_tracker._load_depth_charts(season)

    def _resolve_qb_ids(
        self, season: int, week: int, team: str, lock: Any
    ) -> tuple[str | None, str | None]:
        """A team's QB1 and QB2 gsis_ids as known at the game's lock (D-07)."""
        resolved = self._qb_tracker.resolve_depth_chart_qbs(
            self._load_depth_charts(season), season, week, team, lock
        )
        return resolved.qb1, resolved.qb2

    # ------------------------------------------------------------------
    # Backup-quality delta (REUSES the QBTracker quality scale, D-06)
    # ------------------------------------------------------------------

    def _rolling_qb(self, season: int, week: int, lock: Any) -> pd.DataFrame:
        """The reused QBTracker rolling quality from PBP known at the lock (cached)."""
        key = (int(season), int(week), lock_as_utc(lock))
        if key in self._rolling_cache:
            return self._rolling_cache[key]
        pbp = (
            self._pbp_df
            if self._pbp_df is not None
            else self._qb_tracker._load_pbp_data(season)
        )
        games = self._games_cache if self._games_cache is not None else pd.DataFrame()
        ends = self._qb_tracker.pbp_game_end_times(pbp, games)
        admitted = self._qb_tracker.admitted_pbp(pbp, ends, lock)
        rolling = self._qb_tracker.compute_rolling_qb_metrics(admitted, season, week)
        self._rolling_cache[key] = rolling
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
        lock: Any,
        qb1_id: str | None,
        qb2_id: str | None,
    ) -> float:
        """``quality(QB1) - quality(QB2)`` on the reused QBTracker scale (D-08).

        Only called when the starter is out. A no-history backup is valued at the
        EXPLICIT below-average ``REPLACEMENT_LEVEL_QB_QUALITY`` (never a silent
        league-average 0.0); the starter falls back to 0.0 (league average) when
        their rolling history is unavailable.
        """
        rolling = self._rolling_qb(season, week, lock)
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
            shares exist the value is NaN -- nothing weights it, so it is unknown,
            never the neutral 1.0 -- with coverage False.
        """
        weights = {grp: prior_shares.get(grp, 0.0) for grp in _NON_QB_KEY_GROUPS}
        total_w = sum(weights.values())
        if total_w <= 0:
            # No prior snap-share weights: the honest unknown, flagged as a gap.
            return float("nan"), False

        availability = 0.0
        for grp in _NON_QB_KEY_GROUPS:
            w = weights[grp] / total_w
            n_out = out_counts.get(grp, 0)
            available_g = 1.0 - min(1.0, n_out * OUT_PLAYER_AVAILABILITY_WEIGHT)
            availability += w * available_g
        return float(availability), True

    # ------------------------------------------------------------------
    # Per-game feature assembly
    # ------------------------------------------------------------------

    def _team_shares(self, season: int, week: int) -> dict[str, dict[str, float]]:
        """Per-team prior snap shares from the constructor-injected SnapCountBuilder."""
        prior = self.snap_builder.get_position_prior_shares(season, week)
        team_shares: dict[str, dict[str, float]] = defaultdict(dict)
        if prior is not None and len(prior) > 0:
            for _, row in prior.iterrows():
                team_shares[str(row["team"])][str(row["position_group"])] = float(
                    row["prior_snap_share"]
                )
        return team_shares

    def compute_game_injury_features(
        self,
        game: Mapping[str, Any],
        lock: Any,
        team_shares: Mapping[str, Mapping[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Per-team injury features for one game, from the reports known at its lock.

        Args:
            game: The target game (``season``, ``week``, ``home_team``, ``away_team``).
            lock: The target game's lock, tz-aware.
            team_shares: Per-team prior snap shares (the D-09 weights).

        Returns:
            ``{team: {feature: value}}`` for the game's two teams.
        """
        season, week = int(game["season"]), int(game["week"])
        selection = self.fenced_injuries(game, lock)

        results: dict[str, dict[str, float]] = {}
        for column in ("home_team", "away_team"):
            team = self._qb_tracker._safe_normalize_team(game[column])
            team_inj = (
                selection.rows.loc[selection.rows["_team"] == team]
                if len(selection.rows) > 0
                else selection.rows
            )
            shares = dict(team_shares.get(team, {}))
            _, has_share = self._availability_fraction(shares, {})

            if len(team_inj) == 0:
                # Nothing admitted for this team: the honest unknown (NaN values), with
                # the coverage flags saying so.
                results[team] = {
                    **_UNKNOWN_TEAM_VALUES,
                    "availability_coverage": 1.0 if has_share else 0.0,
                }
                continue

            qb1_id, qb2_id = self._resolve_qb_ids(season, week, team, lock)
            out_ids = set(
                team_inj.loc[
                    team_inj["report_status"].isin(sorted(OUT_STATUSES)), "gsis_id"
                ]
            )
            qb_out = 1.0 if (qb1_id is not None and qb1_id in out_ids) else 0.0
            backup_delta = (
                self._backup_quality_delta(season, week, lock, qb1_id, qb2_id)
                if qb_out == 1.0
                else 0.0
            )
            availability, _ = self._availability_fraction(
                shares, self._out_counts_by_group(team_inj)
            )
            dated_by_row = bool(
                (team_inj["_selection_rule"] == SelectionRule.DATE_MODIFIED.value).any()
            )
            results[team] = {
                "qb_out_flag": qb_out,
                "backup_quality_delta": float(backup_delta),
                "availability_fraction": float(availability),
                "injury_coverage": 1.0,
                "availability_coverage": 1.0 if has_share else 0.0,
                # 1.0 only when this team's reports were dated per row by
                # date_modified AND none of its rows was dropped as undatable.
                "date_modified_coverage": (
                    1.0
                    if dated_by_row and team not in selection.undatable_teams
                    else 0.0
                ),
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
        lock_frame: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build home/away-expanded injury features, each game fenced at its own lock.

        Args:
            games_df: DataFrame of games to build features for (``kickoff_et`` required:
                a game with no kickoff has no lock and is refused by name).
            as_of_datetime: Carried for the ``FeatureBuilder`` Protocol ONLY. It is NOT a
                fence and no selection reads it; the fence is each game's lock.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.
            lock_frame: The build's ``game_id`` -> lock frame, built ONCE by the caller
                (``scripts/build_features.py``). When ``None`` it is built here, once,
                from *games_df* through ``utils.game_lock.lock_frame``.

        Returns:
            One row per game with ``home_`` / ``away_`` injury feature columns.
        """
        logger.info(
            "Building injury features",
            target_season=target_season,
            target_week=target_week,
        )
        self._games_cache = games_df

        empty_cols = (
            ["game_id"]
            + [f"home_{c}" for c in _FEATURE_COLUMNS]
            + [f"away_{c}" for c in _FEATURE_COLUMNS]
        )
        target_games = self._target_games(games_df, target_season, target_week)
        if len(target_games) == 0:
            return pd.DataFrame(columns=empty_cols)
        if lock_frame is None:
            lock_frame = lock_rule.lock_frame(target_games)

        rows: list[dict] = []
        for key, week_games in target_games.groupby(["season", "week"], sort=True):
            season, week = cast(tuple[int, int], key)
            team_shares = self._team_shares(int(season), int(week))
            for game in week_games.to_dict("records"):
                lock = lock_frame[str(game["game_id"])]
                team_features = self.compute_game_injury_features(
                    game, lock, team_shares
                )
                record: dict = {"game_id": game["game_id"]}
                for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
                    team = self._qb_tracker._safe_normalize_team(game[team_col])
                    feats = team_features.get(team, self._neutral_features())
                    for col in _FEATURE_COLUMNS:
                        record[f"{prefix}_{col}"] = float(feats[col])
                rows.append(record)

        return pd.DataFrame(rows, columns=empty_cols)

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Home/away injury features for one game, fenced at its lock.

        The game is resolved from the frame handed to the last ``build_features`` call,
        because its lock needs the kickoff; a game that cannot be resolved has no lock and
        is refused by name (``utils.game_lock.MissingKickoffError``) rather than built from
        a manufactured cutoff. ``as_of_datetime`` is carried for the Protocol only.
        """
        game = self._resolve_game(game_id)
        if game is None:
            msg = (
                f"game {game_id} is not in the games frame this builder was given, so it "
                "has no kickoff and therefore no lock; refusing rather than guessing one"
            )
            raise lock_rule.MissingKickoffError(msg)
        lock = lock_rule.game_lock(game.get("kickoff_et"), game_id=str(game_id))
        team_features = self.compute_game_injury_features(
            game, lock, self._team_shares(int(game["season"]), int(game["week"]))
        )

        result: dict[str, float] = {}
        for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
            team = self._qb_tracker._safe_normalize_team(game[team_col])
            feats = team_features.get(team, self._neutral_features())
            for col in _FEATURE_COLUMNS:
                result[f"{prefix}_{col}"] = float(feats[col])
        return result

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """What a game that admitted NO report must carry, per source-frame column.

        Both teams take the honest unknown: every value column NULL, and both injury coverage
        flags at 0.0 (Plan 33.2-17 Task 2 -- the declared unknown moved with the value, so the
        gate keeps checking the claim against what is actually emitted). Non-empty by
        construction, so the gate value-checks every ``no_information`` row instead of
        trusting it.
        """
        return {
            f"{prefix}_{column}": (None if column in _UNKNOWN_IS_NULL else value)
            for prefix in ("home", "away")
            for column, value in _UNKNOWN_TEAM_VALUES.items()
        }

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game, from the SAME selection ``build_features`` runs.

        * The game admitted reports (``date_modified`` or ``capture_stamp`` rule):
          ``basis="per_row"`` with the MAXIMUM information time among the reports it
          actually admitted -- the selector is the only thing that knows what it used.
        * The game admitted nothing (``none_admitted``): ``basis="no_information"`` with a
          NULL time, value-checked by the gate against :meth:`no_information_signature`.

        There is never a ``per_row`` row with a null time, so the gate's undated refusal is
        unreachable from this builder.

        Returns:
            A frame with exactly ``PROVENANCE_COLUMNS``.
        """
        target_games = self._target_games(games_df, target_season, target_week)
        records: list[dict[str, Any]] = []
        if len(target_games) > 0:
            locks = lock_rule.lock_frame(target_games)
            for game in target_games.to_dict("records"):
                game_id = str(game["game_id"])
                selection = self.fenced_injuries(game, locks[game_id])
                dated = selection.rule is not SelectionRule.NONE_ADMITTED
                records.append(
                    {
                        "game_id": game_id,
                        "basis": (
                            InformationBasis.PER_ROW.value
                            if dated
                            else InformationBasis.NO_INFORMATION.value
                        ),
                        "information_time": (
                            selection.information_time if dated else None
                        ),
                    }
                )
        return pd.DataFrame(records, columns=list(PROVENANCE_COLUMNS))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _neutral_features() -> dict[str, float]:
        """A team the game frame names but no computation reached: the honest unknown.

        Every value NaN, every coverage flag 0.0 (Plan 33.2-17 Task 2; this used to be the
        neutral defaults 0.0 / 0.0 / 1.0).
        """
        return {
            **_UNKNOWN_TEAM_VALUES,
            "availability_coverage": 0.0,
        }

    def _resolve_game(self, game_id: str) -> dict[str, Any] | None:
        """The cached games-frame row for *game_id*, or ``None``."""
        if self._games_cache is None or len(self._games_cache) == 0:
            return None
        match = self._games_cache.loc[self._games_cache["game_id"] == game_id]
        if len(match) == 0:
            return None
        return match.iloc[0].to_dict()
