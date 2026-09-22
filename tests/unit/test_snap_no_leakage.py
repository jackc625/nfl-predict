"""Leakage proof for the snap-count signal (SIG-02 / SC2).

Three-part canonical standard (D-18a), following the
`tests/unit/test_elo_no_leakage.py` convention:

  1. LEAKAGE_KEYWORDS hard-fail (LIVE) -- a stray raw `offense_snaps` column in
     the combined matrix raises LeakageViolation, while the derived
     `rolling_snap_continuity` / `snap_concentration` names do NOT trip the
     substring guard (the D-13 substring-collision contract).
  2. Time-fence assertion (LIVE, Plan 28-03) -- SnapCountBuilder only consumes
     prior-week snap rows (week < target_week within target_season, or any
     earlier season); no current/future-week row feeds the backward-rolling
     features. Since Plan 33.2-14 the window is admitted at the target game's
     LOCK (a team-game counts once it has ENDED at or before it), which on this
     ordinary weekly schedule selects exactly the prior weeks; the synthetic
     one-second boundary is proven in tests/unit/test_builder_lock_cutoffs.py.
  3. Withhold-future byte-unchanged (LIVE, Plan 28-03) -- revealing the target
     week's post-game snap row leaves every rolling_snap_* / snap_continuity /
     snap_concentration value byte-identical.

The keyword tests depend only on the LeakageGate keyword guard (Plan 28-01).
The builder-dependent parts (2, 3) exercise the SnapCountBuilder landed in
Plan 28-03 with small polars->pandas snap fixtures.
"""

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from features.snaps import KEY_POSITION_GROUPS, SnapCountBuilder
from features.validation import LeakageGate, LeakageViolation

# Player-level snap silver schema (mirrors data/silver/snap_counts.parquet).
_SNAP_COLUMNS = [
    "game_id",
    "pfr_player_id",
    "player",
    "position",
    "team",
    "opponent",
    "season",
    "week",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]

# Base roster per team: (pfr_player_id, position, offense_snaps, defense_snaps,
# st_snaps). Offensive starters take offense snaps; defenders take defense snaps.
_KC_ROSTER = [
    ("kc_qb", "QB", 70, 0, 0),
    ("kc_rb", "RB", 45, 0, 6),
    ("kc_wr1", "WR", 60, 0, 4),
    ("kc_wr2", "WR", 50, 0, 9),
    ("kc_te", "TE", 55, 0, 3),
    ("kc_lt", "T", 71, 0, 0),
    ("kc_de", "DE", 0, 60, 5),
    ("kc_cb", "CB", 0, 65, 2),
]
_BUF_ROSTER = [
    ("buf_qb", "QB", 68, 0, 0),
    ("buf_rb", "RB", 40, 0, 7),
    ("buf_wr1", "WR", 62, 0, 3),
    ("buf_wr2", "WR", 48, 0, 8),
    ("buf_te", "TE", 50, 0, 4),
    ("buf_lt", "T", 68, 0, 0),
    ("buf_de", "DE", 0, 58, 6),
    ("buf_cb", "CB", 0, 63, 1),
]


def _roster_for_week(base_roster: list[tuple], week: int) -> list[tuple]:
    """Introduce a small week-2 roster change so continuity is non-degenerate.

    In week 2 the second receiver is a different player (a returning-player
    churn event), so snap_continuity is a meaningful < 1.0 value.
    """
    if week != 2:
        return base_roster
    prefix = base_roster[0][0].split("_")[0]
    swapped = []
    for row in base_roster:
        if row[0].endswith("_wr2"):
            swapped.append((f"{prefix}_wr3", "WR", 50, 0, 9))
        else:
            swapped.append(row)
    return swapped


def _snap_fixture(weeks: list[int], season: int = 2023) -> pd.DataFrame:
    """Build a player-level snap fixture (polars -> pandas, the ingest boundary).

    One KC-vs-BUF game per requested week, with full player rosters for both
    teams so the team-level aggregation, positional concentration, and
    week-over-week continuity are all well-defined.

    Args:
        weeks: Weeks to populate (e.g. [1, 2] or [1, 2, 3]).
        season: Season year.

    Returns:
        pandas DataFrame matching the snap_counts silver schema.
    """
    rows: list[dict] = []
    for week in weeks:
        game_id = f"{season}_{week:02d}_BUF_KC"
        for team, opponent, base in (
            ("KC", "BUF", _KC_ROSTER),
            ("BUF", "KC", _BUF_ROSTER),
        ):
            for pid, pos, off, deff, st in _roster_for_week(base, week):
                total = off + deff + st
                rows.append(
                    {
                        "game_id": game_id,
                        "pfr_player_id": pid,
                        "player": pid,
                        "position": pos,
                        "team": team,
                        "opponent": opponent,
                        "season": season,
                        "week": week,
                        "offense_snaps": float(off),
                        "offense_pct": float(off) / total if total else 0.0,
                        "defense_snaps": float(deff),
                        "defense_pct": float(deff) / total if total else 0.0,
                        "st_snaps": float(st),
                        "st_pct": float(st) / total if total else 0.0,
                    }
                )
    # Round-trip through polars to mirror the nflreadpy .to_pandas() boundary.
    return pl.DataFrame(rows, schema_overrides={"season": pl.Int32, "week": pl.Int32})[
        _SNAP_COLUMNS
    ].to_pandas()


def _kickoff(season: int, week: int) -> pd.Timestamp:
    """A Sunday 13:00 ET kickoff for *week* (week 1 on the second Sunday of September)."""
    return pd.Timestamp(f"{season}-09-10 17:00", tz="UTC") + pd.Timedelta(
        weeks=week - 1
    )


def _schedule(season: int = 2023, weeks: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """The schedule the snap team-games are TIMED against and the targets LOCKED from.

    One KC (home) vs BUF (away) game per week. Injected so the builder never reads the
    local data lake: since Plan 33.2-14 the window admits a team-game only once it has
    ENDED at or before the target game's lock, which needs each game's kickoff.
    """
    return pd.DataFrame(
        [
            {
                "game_id": f"{season}_{week:02d}_BUF_KC",
                "season": season,
                "week": week,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": _kickoff(season, week),
            }
            for week in weeks
        ]
    )


def _make_test_games(season: int = 2023, week: int = 3) -> pd.DataFrame:
    """Minimal games_df for a single KC (home) vs BUF (away) target game."""
    return _schedule(season, (week,))


def _make_combined_matrix(extra_columns: dict[str, list] | None = None) -> pd.DataFrame:
    """Build a minimal combined feature matrix that satisfies the required
    feature groups (elo_, rolling_) so the only thing under test is the
    leakage-keyword scan.

    Args:
        extra_columns: Optional extra columns to splice in (the snap columns
            under test).

    Returns:
        A 3-row combined feature matrix DataFrame.
    """
    df = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "elo_home": [1520.0, 1530.0, 1525.0],
            "elo_away": [1480.0, 1470.0, 1475.0],
            "rolling_epa_home": [0.05, 0.08, 0.06],
            "rolling_epa_away": [-0.02, 0.01, -0.01],
        }
    )
    if extra_columns:
        for col, values in extra_columns.items():
            df[col] = values
    return df


class TestSnapKeywordHardFail:
    """LIVE: the raw snap spellings hard-fail the LeakageGate keyword guard."""

    def test_raw_offense_snaps_column_raises(self):
        """A stray raw `offense_snaps` column in the combined matrix raises
        LeakageViolation with violation_type=leakage_keyword (D-13, SIG-02)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix({"offense_snaps": [55, 60, 58]})

        with pytest.raises(LeakageViolation) as exc_info:
            gate.validate_combined_matrix(df, as_of)

        assert exc_info.value.details["violation_type"] == "leakage_keyword"
        assert "offense_snaps" in exc_info.value.details["affected_features"]

    @pytest.mark.parametrize(
        "raw_col",
        ["defense_snaps", "st_snaps", "offense_pct", "defense_pct", "st_pct"],
    )
    def test_every_raw_snap_spelling_raises(self, raw_col):
        """Each of the six raw snap spellings trips the guard, not just
        offense_snaps (proves the full D-13 keyword set is live)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix({raw_col: [0.5, 0.6, 0.55]})

        with pytest.raises(LeakageViolation) as exc_info:
            gate.validate_combined_matrix(df, as_of)

        assert exc_info.value.details["violation_type"] == "leakage_keyword"
        assert raw_col in exc_info.value.details["affected_features"]


class TestSnapDerivedNamesNoFalsePositive:
    """LIVE: derived snap-feature names must NOT trip the substring guard."""

    def test_derived_snap_names_do_not_raise(self):
        """A matrix whose only snap-derived columns are `rolling_snap_continuity`
        and `snap_concentration` does NOT raise -- the bare `snap`/`snaps`
        keyword is deliberately absent from LEAKAGE_KEYWORDS (D-13)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix(
            {
                "rolling_snap_continuity": [0.9, 0.85, 0.92],
                "snap_concentration": [0.3, 0.35, 0.31],
                "rolling_snap_off_share": [0.7, 0.72, 0.69],
            }
        )

        # Must NOT raise: derived names use _share / _concentration, never the
        # raw spellings or a bare snap/snaps keyword.
        gate.validate_combined_matrix(df, as_of)

    def test_real_builder_columns_do_not_raise(self):
        """The ACTUAL columns SnapCountBuilder emits must clear the guard.

        Builds real home/away-expanded features and feeds them (alongside the
        required elo_/rolling_ groups) through the gate -- proves no derived
        snap name substring-collides with the six raw spellings."""
        gate = LeakageGate()
        as_of = datetime(2023, 10, 26, 18, 0)
        builder = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2]), schedule_df=_schedule()
        )
        feats = builder.build_features(
            _make_test_games(2023, 3), as_of, target_season=2023, target_week=3
        )

        df = _make_combined_matrix()
        for col in feats.columns:
            if col == "game_id":
                continue
            df[col] = list(feats[col]) + [0.0] * (len(df) - len(feats))

        # Must NOT raise -- every derived snap column is leakage-keyword-safe.
        gate.validate_combined_matrix(df, as_of)


class TestSnapTimeFence:
    """LIVE: SnapCountBuilder only uses prior-week snap rows (Plan 28-03)."""

    def test_snap_builder_respects_week_fence(self):
        """No snap row feeding the rolling features for target (2023, week 3)
        may be from week >= 3 within season 2023 (the prior-games-only fence)."""
        # Fixture carries weeks 1, 2 AND 3; week 3 is the target's own post-game
        # data that must be excluded.
        builder = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2, 3]), schedule_df=_schedule()
        )

        contributing = builder.contributing_games(target_season=2023, target_week=3)

        assert len(contributing) > 0, "expected prior-week snaps to contribute"
        # Every contributing row is strictly prior to the target week.
        is_prior = (contributing["season"] < 2023) | (
            (contributing["season"] == 2023) & (contributing["week"] < 3)
        )
        assert is_prior.all(), (
            "time-fence violation: a contributing snap row is from "
            f"week >= target_week:\n{contributing[~is_prior]}"
        )
        # Specifically, NO week-3 row leaked in.
        assert (contributing["week"] != 3).all()


class TestSnapWithholdFuture:
    """LIVE: revealing current-week post-game snaps does not move features."""

    def test_revealing_current_week_snaps_does_not_change_features(self):
        """Building the rolling snap features for target (2023, week 3) once
        WITHOUT the week-3 post-game snap row and once WITH it must produce
        byte-identical rolling_snap_* / snap_continuity / snap_concentration
        values -- the backward-rolling feature is invariant to revealing the
        current week's post-game snaps."""
        rolling_cols = [
            "snap_continuity",
            "snap_concentration",
            *[f"rolling_snap_share_{grp}" for grp in KEY_POSITION_GROUPS],
        ]

        # WITHOUT the target week's post-game snaps (weeks 1-2 only).
        withheld = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2]), schedule_df=_schedule()
        )
        feats_withheld = withheld.compute_team_snap_features(2023, 3)

        # WITH the target week's post-game snaps revealed (weeks 1-2-3).
        revealed = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2, 3]), schedule_df=_schedule()
        )
        feats_revealed = revealed.compute_team_snap_features(2023, 3)

        # Same teams, same ordering.
        feats_withheld = feats_withheld.sort_values("team").reset_index(drop=True)
        feats_revealed = feats_revealed.sort_values("team").reset_index(drop=True)

        assert list(feats_withheld["team"]) == list(feats_revealed["team"])
        assert len(feats_withheld) > 0

        # Byte-identical rolling feature values.
        pd.testing.assert_frame_equal(
            feats_withheld[rolling_cols],
            feats_revealed[rolling_cols],
            check_exact=True,
        )

    def test_withhold_future_holds_end_to_end_through_build_features(self):
        """The same invariance holds through the production build_features entry
        point: the home_/away_ snap columns for the week-3 game are byte-identical
        whether or not the week-3 post-game snap row is present."""
        as_of = datetime(2023, 10, 26, 18, 0)
        games = _make_test_games(2023, 3)

        withheld = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2]), schedule_df=_schedule()
        )
        out_withheld = withheld.build_features(
            games, as_of, target_season=2023, target_week=3
        )

        revealed = SnapCountBuilder(
            snaps_df=_snap_fixture([1, 2, 3]), schedule_df=_schedule()
        )
        out_revealed = revealed.build_features(
            games, as_of, target_season=2023, target_week=3
        )

        pd.testing.assert_frame_equal(out_withheld, out_revealed, check_exact=True)
