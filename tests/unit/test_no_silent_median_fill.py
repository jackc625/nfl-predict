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
  WITHIN-SEASON gap is imputed exactly as before. The WHOLE-SEASON-absence case (a family with no
  rows for one season while other seasons have them) is deliberately NOT pinned here: Plan
  33.2-17 changes it at rung 8 and extends this module to assert it.
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
    return list(SnapCountBuilder(snaps_df=pd.DataFrame()).no_information_signature())


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
