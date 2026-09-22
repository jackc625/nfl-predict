"""An EMPTY feature family reads as unknown, never as a historical median (Plan 33.2-15 Task 2).

D33.2-16 found the defect on the snap family: with no current snap data, 2025 weeks 3 and 15
produced IDENTICAL snap values in columns the ATS and O/U models consume -- a plausible number
standing where there was no data. The fix has two halves, and this module asserts both plus the
ingest wiring that makes them matter:

* THE GUARD (``scripts.build_features.EMPTY_SOURCE_GUARDED_FAMILIES``). A family whose SOURCE
  FRAME carries nothing -- zero rows, or every value column NULL -- lands in gold as NaN, is
  excluded from the generic imputer and keeps its NaN through normalization; the injury family's
  ``*_coverage`` flags read 0.0. Its provenance is ``no_information``.
* THE SCOPE CONTROL, which is why this module exists in this shape. A NON-empty family with a
  WITHIN-SEASON gap is still imputed (Plan 33.2-15 owns this case; since p332_ step 7b the fill
  reads only games ended by the gap's lock, and a gap at the end of a season -- the one pinned
  here -- takes the same team mean either way).
* THE WHOLE-SEASON CASE (Plan 33.2-17 Task 2 owns it; it first reaches gold at rung 8). A family
  that declares a coverage flag and has NO value in a column for a whole season stays NaN --
  not the strictly-prior-seasons median -- and the NaN survives normalization. A family with no
  flag is untouched by it (the control). And the snap / injury BUILDERS emit that honest
  unknown -- NaN beside a false flag, never 0.0 -- for every game before the feed's first
  covered season.
* A SOURCE SCAN over ``scripts/build_features.py`` with its four structural controls.
* A REGRESSION CONTROL: two weeks with genuinely different snap inputs produce different snap
  values (the measured defect was two weeks producing the same ones).
* THE INGEST: both ingesters route every batch through ``validate_bronze_to_silver`` (a batch
  missing the capture stamp is refused), a repeat ingest leaves silver byte-identical, and a
  same-second bronze collision raises ``FileExistsError`` rather than overwriting.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.build_features as build_features_module
from data.quality_gates import validate_bronze_to_silver
from data.schemas import InjurySchema, SnapCountSchema
from features.injury import INJURY_FEATURE_COLUMNS
from features.provenance import InformationBasis
from features.snaps import SnapCountBuilder
from scripts.build_features import (
    EMPTY_SOURCE_GUARDED_FAMILIES,
    FeatureMatrixBuilder,
    source_family_is_empty,
)
from utils.exceptions import DataValidationError

ET = "America/New_York"


# ---------------------------------------------------------------------------
# Fixtures: a small, time-coherent schedule and the families built on it.
# ---------------------------------------------------------------------------


def _games(season: int = 2024, weeks: int = 4) -> pd.DataFrame:
    """Two teams playing each other every week, Sundays at 13:00 ET, week order = time order."""
    first_sunday = pd.Timestamp(f"{season}-09-08 13:00", tz=ET)
    rows = []
    for week in range(1, weeks + 1):
        home, away = ("BUF", "MIA") if week % 2 else ("MIA", "BUF")
        rows.append(
            {
                "game_id": f"{season}_W{week:02d}_{away}@{home}",
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "kickoff_et": first_sunday + pd.Timedelta(days=7 * (week - 1)),
                "home_score": 20 + week,
                "away_score": 17,
            }
        )
    return pd.DataFrame(rows)


def _snap_rows(season: int = 2024, weeks: int = 4) -> pd.DataFrame:
    """Player snaps whose distribution CHANGES week to week (so the windows genuinely differ)."""
    rows = []
    for week in range(1, weeks + 1):
        home, away = ("BUF", "MIA") if week % 2 else ("MIA", "BUF")
        for team, opponent in ((home, away), (away, home)):
            for index, position in enumerate(("QB", "WR", "WR", "T", "CB", "LB")):
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "pfr_player_id": f"{team}{position}{index}w{week % 3}",
                        "player": f"{team} {position} {index}",
                        "position": position,
                        "team": team,
                        "opponent": opponent,
                        "season": season,
                        "week": week,
                        "offense_snaps": float(40 + 5 * index + 7 * week),
                        "offense_pct": 0.9,
                        "defense_snaps": float(10 * (index % 2) + week),
                        "defense_pct": 0.2,
                        "st_snaps": float(index),
                        "st_pct": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def _combined(
    builder: FeatureMatrixBuilder, games: pd.DataFrame, snaps: pd.DataFrame, injury
) -> pd.DataFrame:
    """``combine_features`` over the games plus the two guarded families only."""
    return builder.combine_features({"games": games, "snaps": snaps, "injury": injury})


def _snap_columns() -> list[str]:
    """The snap VALUE columns.

    Was: every key of ``no_information_signature()``. Plan 33.2-17 added ``snap_coverage``, a
    FLAG that reads 0.0 (not NaN) for an unmeasured game, so the value columns exclude it and
    the flag is asserted separately.
    """
    signature = SnapCountBuilder(snaps_df=pd.DataFrame()).no_information_signature()
    return [c for c in signature if not c.endswith("_coverage")]


def _snap_flags() -> list[str]:
    signature = SnapCountBuilder(snaps_df=pd.DataFrame()).no_information_signature()
    return [c for c in signature if c.endswith("_coverage")]


def _injury_columns() -> tuple[list[str], list[str]]:
    expanded = [f"{p}_{c}" for p in ("home", "away") for c in INJURY_FEATURE_COLUMNS]
    flags = [c for c in expanded if c.endswith("_coverage")]
    return [c for c in expanded if c not in flags], flags


def _injury_frame(games: pd.DataFrame) -> pd.DataFrame:
    """A NON-empty injury source frame: every game its documented unknown values."""
    values, flags = _injury_columns()
    frame = pd.DataFrame({"game_id": games["game_id"]})
    for column in values:
        frame[column] = 1.0 if column.endswith("availability_fraction") else 0.0
    for column in flags:
        frame[column] = 0.0
    return frame


# ---------------------------------------------------------------------------
# The guard itself.
# ---------------------------------------------------------------------------


class TestAnEmptySnapFamilyIsUnknown:
    def test_a_zero_row_snap_frame_yields_nan_in_every_snap_column(self) -> None:
        games = _games()
        builder = FeatureMatrixBuilder()
        combined = _combined(builder, games, pd.DataFrame(), _injury_frame(games))
        processed = builder.handle_missing_data_and_outliers(combined)
        columns = _snap_columns()
        assert columns, "non-vacuity: the snap family declares its columns"
        for column in columns:
            assert column in processed.columns, column
            assert processed[column].isna().all(), column
        for flag in _snap_flags():
            assert (processed[flag] == 0.0).all(), flag
        assert builder.empty_source_families == {"snaps": tuple(columns)}

    def test_an_all_null_snap_frame_keeps_its_nan_through_normalization(self) -> None:
        # The live shape: the builder read an EMPTY silver table and emitted a row per game
        # whose every value is NULL. Before the guard, normalization turned that NaN into the
        # neutral 0.0 z-score.
        games = _games()
        snap_builder = SnapCountBuilder(snaps_df=pd.DataFrame(), schedule_df=games)
        snap_frame = snap_builder.build_features(games, datetime.now(UTC))
        assert len(snap_frame) == len(games)
        builder = FeatureMatrixBuilder()
        # Normalization refuses to run without a merged weather frame (the fail-closed
        # coverage-flag rule), so a minimal full-builder weather frame rides along.
        weather = pd.DataFrame(
            {"game_id": games["game_id"], "weather_coverage": 1.0, "temp_f": 60.0}
        )
        combined = builder.combine_features(
            {
                "games": games,
                "snaps": snap_frame,
                "injury": _injury_frame(games),
                "weather": weather,
            }
        )
        assert builder.empty_source_families.get("snaps"), "the all-null arm fired"
        processed = builder.handle_missing_data_and_outliers(combined)
        normalized = builder.normalize_combined_features(processed)
        for column in _snap_columns():
            assert normalized[column].isna().all(), column

    def test_no_value_equal_to_a_historical_median_appears(self) -> None:
        # A PRIOR season with real snap values exists in the same build; the empty 2025
        # family must not borrow its median (the measured defect's mechanism).
        prior = _games(2024)
        target = _games(2025)
        games = pd.concat([prior, target], ignore_index=True)
        builder = FeatureMatrixBuilder()
        combined = _combined(builder, games, pd.DataFrame(), _injury_frame(games))
        combined.loc[combined["season"] == 2024, _snap_columns()] = 0.4
        # 2024 now carries values -- but the SOURCE frame was empty, so the family is the
        # recorded unknown and nothing is imputed into 2025.
        processed = builder.handle_missing_data_and_outliers(combined)
        late = processed.loc[processed["season"] == 2025, _snap_columns()]
        assert late.isna().all().all()
        assert not (late == 0.4).any().any()

    def test_its_provenance_reports_no_information(self) -> None:
        games = _games()
        snap_builder = SnapCountBuilder(snaps_df=pd.DataFrame(), schedule_df=games)
        provenance = snap_builder.information_times(games)
        assert len(provenance) == len(games)
        assert set(provenance["basis"]) == {InformationBasis.NO_INFORMATION.value}
        assert provenance["information_time"].isna().all()


class TestAnEmptyInjuryFamilyIsUnknownWithItsFlag:
    def test_a_zero_row_injury_frame_yields_nan_plus_the_coverage_flags_at_zero(
        self,
    ) -> None:
        games = _games()
        builder = FeatureMatrixBuilder()
        snaps = SnapCountBuilder(
            snaps_df=_snap_rows(), schedule_df=games
        ).build_features(games, datetime.now(UTC))
        combined = _combined(builder, games, snaps, pd.DataFrame())
        processed = builder.handle_missing_data_and_outliers(combined)
        values, flags = _injury_columns()
        assert values and flags, "non-vacuity"
        for column in values:
            assert processed[column].isna().all(), column
        for column in flags:
            assert (processed[column] == 0.0).all(), column
        assert "injury" in builder.empty_source_families
        assert "snaps" not in builder.empty_source_families

    def test_the_injury_builders_own_unknown_values_are_not_an_empty_frame(
        self,
    ) -> None:
        # A frame of documented unknown values (0.0 / 1.0 with the flags at 0.0) is VALUES the
        # gate value-checks, not an absence.
        games = _games()
        values, _flags = _injury_columns()
        assert not source_family_is_empty(_injury_frame(games), values)
        assert source_family_is_empty(pd.DataFrame(), values)


class TestTheGuardedFamiliesAreDeclared:
    def test_exactly_snaps_and_injury(self) -> None:
        assert EMPTY_SOURCE_GUARDED_FAMILIES == ("snaps", "injury")

    def test_the_flag_declaring_families_are_the_same_two(self) -> None:
        assert build_features_module.COVERAGE_FLAGGED_FAMILIES == ("snaps", "injury")


# ---------------------------------------------------------------------------
# THE SCOPE CONTROL: a within-season gap in a non-empty family is imputed exactly as before.
# ---------------------------------------------------------------------------


class TestAWithinSeasonGapIsImputedExactlyAsBefore:
    def test_a_within_season_gap_takes_the_teams_own_season_mean(self) -> None:
        games = _games(2024, weeks=6)
        builder = FeatureMatrixBuilder()
        snaps = SnapCountBuilder(snaps_df=_snap_rows(2024, 6), schedule_df=games)
        snap_frame = snaps.build_features(games, datetime.now(UTC))
        column = "home_snap_concentration"
        # Punch ONE within-season hole in a family that otherwise has data.
        hole = snap_frame.index[snap_frame[column].notna()][-1]
        snap_frame.loc[hole, column] = np.nan
        combined = _combined(builder, games, snap_frame, _injury_frame(games))
        assert builder.empty_source_families == {}
        # The unguarded generic imputer, called directly, is the "before".
        expected = builder._impute_team_features(combined, column)
        processed = builder.handle_missing_data_and_outliers(combined)
        filled = processed.loc[hole, column]
        assert np.isfinite(filled)
        assert filled == expected.loc[hole]
        # ...and it IS the team's own within-season mean, the imputer's first rule.
        team = combined.loc[hole, "home_team"]
        mask = (combined["home_team"] == team) & (combined["season"] == 2024)
        assert filled == combined.loc[mask, column].mean()


# ---------------------------------------------------------------------------
# THE WHOLE-SEASON CASE (Plan 33.2-17 Task 2): a flag-declaring family's empty season is NaN.
# ---------------------------------------------------------------------------


def _two_season_snap_frame() -> tuple[FeatureMatrixBuilder, pd.DataFrame]:
    """2024 carries real snap values; 2025 carries NONE (every value NaN, the flag 0.0)."""
    # Twelve 2024 weeks: more than ``_MIN_FIT_POINTS`` prior values, so a prior median EXISTS
    # and the guard (not a thin prior season) is what keeps 2025 empty.
    games = pd.concat(
        [_games(2024, weeks=12), _games(2025, weeks=6)], ignore_index=True
    )
    builder = FeatureMatrixBuilder()
    snaps = SnapCountBuilder(snaps_df=_snap_rows(2024, 12), schedule_df=games)
    snap_frame = snaps.build_features(games, datetime.now(UTC))
    late = snap_frame["game_id"].str.startswith("2025")
    snap_frame.loc[late, _snap_columns()] = np.nan
    snap_frame.loc[late, _snap_flags()] = 0.0
    combined = _combined(builder, games, snap_frame, _injury_frame(games))
    assert builder.empty_source_families == {}, "the source frame is NOT empty"
    return builder, combined


class TestAFlagDeclaringFamilysEmptySeasonStaysNaN:
    def test_non_vacuity_2024_carries_values(self) -> None:
        _builder, combined = _two_season_snap_frame()
        early = combined.loc[combined["season"] == 2024, _snap_columns()]
        assert early.notna().any().any()

    def test_the_empty_season_is_not_filled_from_the_prior_median(self) -> None:
        builder, combined = _two_season_snap_frame()
        processed = builder.handle_missing_data_and_outliers(combined)
        late = processed.loc[processed["season"] == 2025, _snap_columns()]
        assert late.isna().all().all(), (
            "a season with no snap data was filled with a borrowed value -- the mechanism "
            "behind 2025 weeks 3 and 15 coming out identical"
        )

    def test_the_empty_season_keeps_its_nan_through_normalization(self) -> None:
        builder, combined = _two_season_snap_frame()
        combined["weather_coverage"] = 1.0
        combined["temp_f"] = 60.0
        weather = combined[["game_id", "weather_coverage", "temp_f"]]
        builder.record_missing_preserving_columns(weather)
        processed = builder.handle_missing_data_and_outliers(combined)
        normalized = builder.normalize_combined_features(processed)
        late = normalized.loc[normalized["season"] == 2025, _snap_columns()]
        assert late.isna().all().all()

    def test_a_family_with_no_flag_still_takes_the_prior_median(self) -> None:
        # The control: the guard is keyed on the family's flag, not applied to every column.
        builder, combined = _two_season_snap_frame()
        combined["home_off_rolling_cpoe"] = np.where(
            combined["season"] == 2024, np.linspace(1.0, 2.0, len(combined)), np.nan
        )
        processed = builder.handle_missing_data_and_outliers(combined)
        late = processed.loc[processed["season"] == 2025, "home_off_rolling_cpoe"]
        assert late.notna().all()


class TestTheBuildersEmitTheHonestUnknownBeforeCoverage:
    """The first covered season is DERIVED from the rows upstream supplied, never a literal."""

    @staticmethod
    def _snap_output() -> tuple[pd.DataFrame, pd.DataFrame]:
        games = pd.concat(
            [_games(2012, weeks=4), _games(2013, weeks=4)], ignore_index=True
        )
        frame = SnapCountBuilder(
            snaps_df=_snap_rows(2013, 4), schedule_df=games
        ).build_features(games, datetime.now(UTC))
        return games, frame.merge(games[["game_id", "season", "week"]], on="game_id")

    def test_before_the_first_snap_season_every_value_is_nan_and_the_flag_false(
        self,
    ) -> None:
        _games_frame, frame = self._snap_output()
        before = frame[frame["season"] == 2012]
        assert len(before) > 0, "non-vacuity"
        assert before[_snap_columns()].isna().all().all()
        assert (before[_snap_flags()] == 0.0).all().all()
        assert not (before[_snap_columns()] == 0.0).any().any()

    def test_the_first_snap_season_is_populated_with_the_flag_true(self) -> None:
        _games_frame, frame = self._snap_output()
        first = frame[frame["season"] == 2013]
        for side in ("home", "away"):
            flag = f"{side}_snap_coverage"
            values = [c for c in _snap_columns() if c.startswith(f"{side}_")]
            measured = first[flag] == 1.0
            assert measured.any(), "populated from the first season with snap rows"
            populated = first.loc[measured, values].drop(
                columns=[f"{side}_snap_continuity"]
            )
            assert populated.notna().all().all()
            # A team whose window admitted no snap game yet (its first game) is the unknown.
            assert first.loc[~measured, values].isna().all().all()

    def test_snap_coverage_is_in_the_list_the_builder_emits(self) -> None:
        columns = SnapCountBuilder(snaps_df=pd.DataFrame())._feature_columns()
        assert "snap_coverage" in columns
        _games_frame, frame = self._snap_output()
        assert {"home_snap_coverage", "away_snap_coverage"} <= set(frame.columns)

    def test_an_injury_game_with_nothing_admitted_is_nan_with_its_flags_false(
        self,
    ) -> None:
        from features.injury import InjuryBuilder

        games = _games(2008, weeks=2)
        snap_builder = SnapCountBuilder(snaps_df=pd.DataFrame(), schedule_df=games)
        builder = InjuryBuilder(
            snap_builder,
            injuries_df=pd.DataFrame(),
            depth_charts_df=pd.DataFrame(),
            pbp_df=pd.DataFrame(),
        )
        frame = builder.build_features(games, datetime.now(UTC))
        values, flags = _injury_columns()
        assert frame[values].isna().all().all()
        assert (frame[flags] == 0.0).all().all()
        assert not (frame[values] == 0.0).any().any()
        assert not (frame[values] == 1.0).any().any()

    def test_the_injury_signature_declares_the_same_unknown(self) -> None:
        from features.injury import InjuryBuilder

        signature = InjuryBuilder(
            SnapCountBuilder(snaps_df=pd.DataFrame())
        ).no_information_signature()
        assert signature, "non-empty: the gate value-checks every no-information row"
        for column, declared in signature.items():
            if column.endswith("_coverage"):
                assert declared == 0.0, column
            else:
                assert declared is None, column


class TestAMetricThePinnedPlayByPlayLacksIsFlagged:
    """``rolling_cpoe`` before 2006: the pinned play-by-play carries no completion probability.

    Task 1's corpus widening computes every team-form metric back to 2002, and every one is
    real there except cpoe, which is NULL in every 2002-2005 row of silver team form (measured
    2026-09-22). Left alone it would reach gold as the neutral 0.0 after rung 8 -- an unflagged
    pre-coverage block no other route closes -- so it is the honest unknown with its own flag,
    under the same coverage-floor cause (Plan 33.2-17 Task 3's routing rule).
    """

    @staticmethod
    def _frames() -> tuple[FeatureMatrixBuilder, pd.DataFrame]:
        games = pd.concat(
            [_games(2005, weeks=12), _games(2006, weeks=12)], ignore_index=True
        )
        form = pd.DataFrame({"game_id": games["game_id"]})
        for prefix in ("home", "away"):
            form[f"{prefix}_off_rolling_cpoe"] = np.where(
                games["season"] == 2006, np.linspace(-2.0, 3.0, len(games)), np.nan
            )
            form[f"{prefix}_off_rolling_success_rate"] = np.linspace(
                0.4, 0.5, len(games)
            )
        builder = FeatureMatrixBuilder()
        combined = builder.combine_features({"games": games, "team_form": form})
        return builder, combined

    def test_the_flag_states_where_cpoe_was_measured(self) -> None:
        _builder, combined = self._frames()
        for prefix in ("home", "away"):
            flag = combined[f"{prefix}_off_rolling_cpoe_coverage"]
            value = combined[f"{prefix}_off_rolling_cpoe"]
            assert ((flag == 1.0) == value.notna()).all()
            assert (flag[combined["season"] == 2005] == 0.0).all()

    def test_a_metric_that_is_always_present_gets_no_flag(self) -> None:
        _builder, combined = self._frames()
        assert "home_off_rolling_success_rate_coverage" not in combined.columns

    def test_the_unmeasured_seasons_stay_nan_through_normalization(self) -> None:
        builder, combined = self._frames()
        combined["weather_coverage"] = 1.0
        combined["temp_f"] = 60.0
        builder.record_missing_preserving_columns(
            combined[["game_id", "weather_coverage", "temp_f"]]
        )
        processed = builder.handle_missing_data_and_outliers(combined)
        normalized = builder.normalize_combined_features(processed)
        early = normalized.loc[normalized["season"] == 2005]
        assert early["home_off_rolling_cpoe"].isna().all()
        assert (early["home_off_rolling_cpoe_coverage"] == 0.0).all()
        late = normalized.loc[normalized["season"] == 2006, "home_off_rolling_cpoe"]
        assert late.notna().all()


# ---------------------------------------------------------------------------
# THE SOURCE SCAN, with its four structural controls.
# ---------------------------------------------------------------------------

IMPUTERS = ("_impute_team_features", "_impute_game_level_features")
GUARD_NAME = "preserve_this_column"
EMPTY_SET_NAME = "empty_family_columns"


def _is_median_fill(node: ast.AST) -> bool:
    """A call to one of the imputers, or ``.fillna(<something>.median() / .mean())``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name in IMPUTERS:
        return True
    if name == "fillna":
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in ("median", "mean")
            for arg in node.args
            for sub in ast.walk(arg)
        )
    return False


def _unguarded_median_fills(function_source: str) -> tuple[list[int], bool, int]:
    """``(unguarded fill lines, guard names the empty set, fills seen)`` for one function.

    A fill is GUARDED when an enclosing ``if`` test reads ``preserve_this_column``; the guard
    counts only when ``preserve_this_column`` is assigned from an expression naming
    ``empty_family_columns``.
    """
    tree = ast.parse(textwrap.dedent(function_source))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    guard_names_empty_set = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == GUARD_NAME for t in node.targets)
        and any(
            isinstance(n, ast.Name) and n.id == EMPTY_SET_NAME
            for n in ast.walk(node.value)
        )
        for node in ast.walk(tree)
    )

    unguarded: list[int] = []
    fills = [node for node in ast.walk(tree) if _is_median_fill(node)]
    for fill in fills:
        node, guarded = fill, False
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If) and any(
                isinstance(n, ast.Name) and n.id == GUARD_NAME
                for n in ast.walk(node.test)
            ):
                guarded = True
                break
        if not guarded:
            unguarded.append(fill.lineno)
    return sorted(unguarded), guard_names_empty_set, len(fills)


class TestTheSourceScan:
    DISPATCHER = FeatureMatrixBuilder.handle_missing_data_and_outliers

    def test_non_vacuity_the_scan_reads_the_real_dispatcher_and_sees_its_fills(
        self,
    ) -> None:
        source = inspect.getsource(self.DISPATCHER)
        tree = ast.parse(textwrap.dedent(source))
        assert sum(1 for _ in ast.walk(tree)) > 200
        _lines, _guard, fills = _unguarded_median_fills(source)
        assert fills >= 2, "the dispatcher's two imputer calls were not seen"

    def test_no_median_fill_can_reach_an_empty_family(self) -> None:
        lines, guard_names_empty_set, _fills = _unguarded_median_fills(
            inspect.getsource(self.DISPATCHER)
        )
        assert guard_names_empty_set, (
            f"{GUARD_NAME} is not assigned from {EMPTY_SET_NAME}: an empty family would reach "
            "the generic imputer"
        )
        assert lines == []

    def test_a_planted_fillna_median_on_the_empty_family_path_is_flagged(self) -> None:
        source = textwrap.dedent(inspect.getsource(self.DISPATCHER))
        anchor = "    for col in numeric_cols:\n"  # the dedented method body
        assert anchor in source
        planted = source.replace(
            anchor,
            anchor + "        processed_df[col] = processed_df[col].fillna("
            "processed_df[col].median())\n",
            1,
        )
        lines, _guard, _fills = _unguarded_median_fills(planted)
        assert len(lines) == 1

    def test_the_legitimate_within_family_imputation_is_not_flagged(self) -> None:
        # _impute_team_features IS a median fill by design (its last resort). The scan judges the
        # DISPATCH to it, so the imputer's own body is not a violation, and the dispatcher's
        # guarded call to it is not either.
        dispatcher_lines, _guard, _fills = _unguarded_median_fills(
            inspect.getsource(self.DISPATCHER)
        )
        assert dispatcher_lines == []
        imputer = inspect.getsource(FeatureMatrixBuilder._impute_team_features)
        assert "median" in imputer, "non-vacuity: the imputer still performs its fill"


# ---------------------------------------------------------------------------
# THE REGRESSION CONTROL: different weeks, different snap values.
# ---------------------------------------------------------------------------


class TestDifferentWeeksGetDifferentSnapValues:
    def test_two_weeks_with_different_snap_inputs_produce_different_values(
        self,
    ) -> None:
        games = _games(2024, weeks=6)
        frame = SnapCountBuilder(
            snaps_df=_snap_rows(2024, 6), schedule_df=games
        ).build_features(games, datetime.now(UTC))
        by_week = frame.merge(games[["game_id", "week"]], on="game_id")
        week3 = by_week.loc[by_week["week"] == 3, _snap_columns()].iloc[0]
        week5 = by_week.loc[by_week["week"] == 5, _snap_columns()].iloc[0]
        assert week3.notna().any() and week5.notna().any(), "non-vacuity"
        assert not week3.equals(week5)


# ---------------------------------------------------------------------------
# THE INGEST: the promotion gate, the stamp, the repeat run and the bronze collision.
# ---------------------------------------------------------------------------

STAMP = datetime(2026, 9, 21, 11, 3, 43, tzinfo=UTC)


def _stamp_reader(_asset: str, *, fresh: bool = False) -> datetime:
    return STAMP


def _raw_snaps(season: int = 2026) -> pd.DataFrame:
    raw = _snap_rows(season, 2)
    raw["game_type"] = "REG"
    raw["pfr_game_id"] = "x"
    return raw


def _raw_injuries(season: int = 2026) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season, season],
            "season_type": ["REG", "REG"],
            "game_type": ["REG", "REG"],
            "team": ["BUF", "MIA"],
            "week": [1, 1],
            "gsis_id": ["00-1", "00-2"],
            "position": ["QB", "WR"],
            "full_name": ["A", "B"],
            "first_name": ["A", "B"],
            "last_name": ["A", "B"],
            "report_primary_injury": ["Ankle", None],
            "report_secondary_injury": [None, None],
            "report_status": ["Questionable", "Out"],
            "practice_primary_injury": ["Ankle", "Knee"],
            "practice_secondary_injury": [None, None],
            "practice_status": ["Limited", "Did Not Participate"],
        }
    )


class _BronzeClock(datetime):
    """A ``datetime`` whose ``now`` hands out queued instants (the bronze filename clock)."""

    queue: list[datetime] = []

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.queue.pop(0)


@pytest.fixture
def bronze_clock(monkeypatch):
    from data import storage

    _BronzeClock.queue = []
    monkeypatch.setattr(storage, "datetime", _BronzeClock)
    return _BronzeClock


class TestTheIngestersPassThePromotionGate:
    def test_both_ingesters_call_the_gate_between_bronze_and_silver(self) -> None:
        import scripts.ingest_injuries as injuries
        import scripts.ingest_snaps as snaps

        for module in (snaps, injuries):
            names = [
                node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                for node in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(node, ast.Call)
                and (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
                in (
                    "save_bronze_snapshot",
                    "validate_bronze_to_silver",
                    "upsert_silver",
                )
            ]
            assert names == [
                "save_bronze_snapshot",
                "validate_bronze_to_silver",
                "upsert_silver",
            ], module.__name__

    def test_a_batch_missing_the_stamp_is_refused(self) -> None:
        raw = _raw_snaps()
        batch = raw[
            [c for c in SnapCountSchema.model_fields if c in raw.columns]
        ].copy()
        with pytest.raises(DataValidationError):
            validate_bronze_to_silver(batch, SnapCountSchema)
        injuries = _raw_injuries().assign(
            game_id="2026_W01_MIA@BUF", date_modified=pd.NaT
        )
        with pytest.raises(DataValidationError):
            validate_bronze_to_silver(
                injuries[
                    [c for c in InjurySchema.model_fields if c in injuries.columns]
                ],
                InjurySchema,
            )

    def test_the_snap_ingest_lands_every_kept_column_and_the_stamp(
        self, tmp_path: Path, bronze_clock
    ) -> None:
        from scripts.ingest_snaps import RAW_SNAP_COLUMNS, ingest_snaps_season

        bronze_clock.queue = [datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)]
        ingest_snaps_season(
            2026,
            loader=lambda _s: _raw_snaps(),
            stamp_reader=_stamp_reader,
            base_path=tmp_path,
        )
        silver = pd.read_parquet(tmp_path / "silver" / "snap_counts.parquet")
        assert len(silver) == len(_raw_snaps())
        assert silver[RAW_SNAP_COLUMNS].notna().all().all()
        assert (silver["upstream_captured_at"] == pd.Timestamp(STAMP)).all()

    def test_the_injury_ingest_carries_a_null_date_modified_and_the_stamp(
        self, tmp_path: Path, bronze_clock
    ) -> None:
        from scripts.ingest_injuries import ingest_injuries_season

        bronze_clock.queue = [datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)]
        games = _games(2026, weeks=1)
        ingest_injuries_season(
            2026,
            loader=lambda _s: _raw_injuries(),
            stamp_reader=_stamp_reader,
            games=games,
            base_path=tmp_path,
        )
        silver = pd.read_parquet(tmp_path / "silver" / "injuries.parquet")
        assert len(silver) == 2
        assert silver["date_modified"].isna().all()
        assert str(silver["upstream_captured_at"].dtype).startswith("datetime64")
        assert (silver["upstream_captured_at"] == pd.Timestamp(STAMP)).all()


class TestARepeatIngest:
    def test_running_twice_leaves_silver_byte_identical(
        self, tmp_path: Path, bronze_clock
    ) -> None:
        from scripts.ingest_snaps import ingest_snaps_season

        first = datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)
        bronze_clock.queue = [first, first + timedelta(seconds=5)]
        kwargs = {
            "loader": lambda _s: _raw_snaps(),
            "stamp_reader": _stamp_reader,
            "base_path": tmp_path,
        }
        ingest_snaps_season(2026, **kwargs)
        silver = tmp_path / "silver" / "snap_counts.parquet"
        once = silver.read_bytes()
        ingest_snaps_season(2026, **kwargs)
        assert silver.read_bytes() == once
        assert len(list((tmp_path / "bronze").glob("snap_counts_raw_bronze_*"))) == 2

    def test_a_same_second_bronze_collision_raises_rather_than_overwrites(
        self, tmp_path: Path, bronze_clock
    ) -> None:
        from scripts.ingest_injuries import ingest_injuries_season

        instant = datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)
        bronze_clock.queue = [instant, instant]
        kwargs = {
            "loader": lambda _s: _raw_injuries(),
            "stamp_reader": _stamp_reader,
            "games": _games(2026, weeks=1),
            "base_path": tmp_path,
        }
        ingest_injuries_season(2026, **kwargs)
        bronze = next((tmp_path / "bronze").glob("injuries_raw_bronze_*"))
        original = bronze.read_bytes()
        with pytest.raises(FileExistsError):
            ingest_injuries_season(2026, **kwargs)
        assert bronze.read_bytes() == original


class TestTheStampIsBracketed:
    def test_a_republish_during_the_download_is_refused(self) -> None:
        from data.upstream_asset_stamp import UpstreamStampUnavailable, fetch_with_stamp

        reads = iter([STAMP, STAMP + timedelta(hours=6)])

        def reader(_asset: str, *, fresh: bool = False) -> datetime:
            return next(reads)

        with pytest.raises(UpstreamStampUnavailable, match="republished"):
            fetch_with_stamp(
                "injuries_2026.parquet", lambda: "bytes", stamp_reader=reader
            )

    def test_the_second_read_is_fresh_and_its_value_is_returned(self) -> None:
        from data.upstream_asset_stamp import fetch_with_stamp

        calls: list[bool] = []

        def reader(_asset: str, *, fresh: bool = False) -> datetime:
            calls.append(fresh)
            return STAMP

        fetched, stamp = fetch_with_stamp(
            "injuries_2026.parquet", lambda: "bytes", stamp_reader=reader
        )
        assert (fetched, stamp) == ("bytes", STAMP)
        assert calls == [False, True]


def test_module_imports_are_real() -> None:
    """Guard against a renamed seam turning this module into a no-op."""
    assert hasattr(
        build_features_module.FeatureMatrixBuilder, "_lay_out_empty_source_families"
    )
