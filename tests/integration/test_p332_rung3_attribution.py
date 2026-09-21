"""Rung 3 of the Phase-33.2 gold ladder: the emergency schedule moves.

Plan 33.2-10 Task 4 (D33.2-20, SPEC R8). Rung 3 is registered AND dispatched per Plan
33.2-08's <owned_protocol_rung_registration>: its cause, signature and attributor sit in the
`p332_` tables, and it is judged by ``_attribute_p332_schedule_move`` -- never by the generic
rung-3 path, which is Phase 30's and expects a ``line_movement`` column REMOVAL. That trap
is sharpest at this rung number: a correct rebuild that removes nothing would be refused
against another phase's cause. The dispatch is ASSERTED below.

THE EXPLAINABLE SET WAS DERIVED BEFORE THE REBUILD RAN. Each reverted move kind reaches a
known family of contextual columns; today the owner-ratified table reverts exactly one
venue move (2003_W08_MIA@LAC), so the set is the stadium-dependent set rung 2 derives from
source, and the season floor is 2003.

THE ROW RULE is checked against a copy of the before-gold taken just before the rebuild
(``outputs/p332_rung3_gold_before/``, gitignored). The live half reads gitignored evidence
and skips, naming it, on a checkout that has not run the ladder.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import scripts.fingerprint_gold as fg
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_RETAKEN_BASELINES,
    PHASE332_RUNG_PREFIX,
    PHASE332_SCHEDULE_MOVE_RUNG,
    PHASE332_VENUE_RUNG,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    require_rung_ladder,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG2 = rung_document_path(FINGERPRINT_DIR, PHASE332_VENUE_RUNG, PHASE332_RUNG_PREFIX)
RUNG3 = rung_document_path(
    FINGERPRINT_DIR, PHASE332_SCHEDULE_MOVE_RUNG, PHASE332_RUNG_PREFIX
)
BASELINE_CONFIRM = (
    FINGERPRINT_DIR / fg.PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION_DOCUMENT
)
GOLD_BEFORE_DIR = Path("outputs/p332_rung3_gold_before")
GOLD_AFTER_DIR = Path("data/gold")

NEUTRALISED_GAME = "2003_W08_MIA@LAC"
SEASON_FLOOR = 2003

# DERIVED 2026-09-21 by scripts.fingerprint_gold.phase332_schedule_fact_columns(), BEFORE
# the rung-3 rebuild ran: one reverted venue move reaches the stadium-dependent set.
DERIVED_SCHEDULE_FACT_COLUMNS: tuple[str, ...] = (
    "away_abs_timezone_diff_hours",
    "away_cross_country_travel",
    "away_eastward_travel",
    "away_timezone_diff_hours",
    "away_travel_distance_miles",
    "away_travel_fatigue_score",
    "away_westward_travel",
    "surface_mismatch",
    "venue_capacity",
    "venue_cold_climate",
    "venue_elevation_ft",
    "venue_high_altitude",
    "venue_indoor",
    "venue_large_stadium",
    "venue_outdoor",
    "venue_retractable",
    "venue_warm_climate",
)


def _matrix(changed: dict[str, list[str]], removed: list[str] | None = None) -> dict:
    """One synthetic compare_fingerprints matrix entry: same shape, given moves."""
    return {
        "width_before": 195,
        "width_after": 195 - len(removed or []),
        "rows_before": 6499,
        "rows_after": 6499,
        "rows_per_season_before": {},
        "rows_per_season_after": {},
        "columns_added": [],
        "columns_removed": list(removed or []),
        "columns_changed": changed,
        "column_details": {
            column: {
                "seasons": seasons,
                "seasons_values": seasons,
                "seasons_storage": [],
                "move_kind": "values",
                "reasons": ["values"],
                "dtype_before": "float64",
                "dtype_after": "float64",
                "null_count_before": 0,
                "null_count_after": 0,
                "discrete_indicator_before": False,
                "discrete_indicator_after": False,
            }
            for column, seasons in changed.items()
        },
    }


def _report(changed: dict[str, list[str]]) -> dict:
    return {matrix: _matrix(changed) for matrix in fg.GOLD_MATRICES}


class TestTheRungIsRegisteredAndDispatched:
    def test_the_three_names_and_both_table_entries_exist(self) -> None:
        causes = fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert (
            causes[PHASE332_SCHEDULE_MOVE_RUNG] == fg.PHASE332_SCHEDULE_MOVE_RUNG_CAUSE
        )
        assert (
            fg.PHASE332_RUNG_SIGNATURES[PHASE332_SCHEDULE_MOVE_RUNG]
            is fg.PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_SCHEDULE_MOVE_RUNG]
            is fg._attribute_p332_schedule_move
        )

    def test_the_signature_handed_out_is_a_copy(self) -> None:
        signature = fg._expected_signature(
            PHASE332_SCHEDULE_MOVE_RUNG, prefix=PHASE332_RUNG_PREFIX
        )
        assert signature == fg.PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE
        assert signature is not fg.PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert "line_movement" not in json.dumps(signature)

    def test_rung_three_is_judged_by_its_own_attributor(self, monkeypatch) -> None:
        calls: list[str] = []
        original = fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_SCHEDULE_MOVE_RUNG]

        def spy(*args, **kwargs):
            calls.append("p332_schedule_move")
            return original(*args, **kwargs)

        monkeypatch.setitem(
            fg.PHASE332_RUNG_ATTRIBUTORS, PHASE332_SCHEDULE_MOVE_RUNG, spy
        )
        verdict = attribute_rung(
            _report({"venue_elevation_ft": [str(SEASON_FLOOR)]}),
            PHASE332_SCHEDULE_MOVE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332_schedule_move"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]
        assert verdict["cause"] == fg.PHASE332_SCHEDULE_MOVE_RUNG_CAUSE

    def test_the_generic_rung_three_path_would_refuse_a_correct_rebuild(self) -> None:
        """The trap, measured: Phase 30's rung 3 wants a line_movement removal."""
        diff = _report({"venue_elevation_ft": [str(SEASON_FLOOR)]})
        phase30 = attribute_rung(diff, 3)
        own = attribute_rung(
            diff, PHASE332_SCHEDULE_MOVE_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert own["ok"], own["failures"]
        assert not phase30["ok"]

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_rung_three_registers_no_retaken_baseline(self) -> None:
        assert PHASE332_SCHEDULE_MOVE_RUNG not in PHASE332_RETAKEN_BASELINES
        assert "CONFIRMED" in fg.PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION


class TestTheExplainableSetIsDerived:
    def test_the_reverted_moves_are_the_published_post_lock_list(self) -> None:
        from features.schedule_moves import post_lock_summary

        reverted = fg.phase332_post_lock_real_moves()
        assert reverted == {NEUTRALISED_GAME: ("venue",)}
        assert sorted(reverted) == post_lock_summary()["real_moves"]

    def test_the_pinned_set_is_what_the_derivation_returns(self) -> None:
        assert fg.phase332_schedule_fact_columns() == DERIVED_SCHEDULE_FACT_COLUMNS
        assert fg.phase332_schedule_move_earliest_season() == SEASON_FLOOR

    def test_no_weather_column_is_in_the_set(self) -> None:
        from features.weather import WEATHER_FEATURE_COLUMNS

        assert not set(DERIVED_SCHEDULE_FACT_COLUMNS) & set(WEATHER_FEATURE_COLUMNS)


class TestTheAttributorRefusesWhatTheCauseCannotExplain:
    def test_a_schedule_fact_column_on_or_after_the_floor_is_attributed(self) -> None:
        verdict = attribute_rung(
            _report({c: ["2003", "2011"] for c in DERIVED_SCHEDULE_FACT_COLUMNS}),
            PHASE332_SCHEDULE_MOVE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]

    def test_a_schedule_fact_column_before_the_floor_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"venue_outdoor": ["2002", "2003"]}),
            PHASE332_SCHEDULE_MOVE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]
        assert verdict["matrices"]["features_wp"]["unattributed"] == ["venue_outdoor"]

    def test_a_weekday_column_is_unattributed_while_no_date_move_is_reverted(
        self,
    ) -> None:
        verdict = attribute_rung(
            _report({"monday_game": ["2003"]}),
            PHASE332_SCHEDULE_MOVE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_a_weather_column_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"weather_severity_score": ["2003"]}),
            PHASE332_SCHEDULE_MOVE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        verdict = attribute_rung(
            _report({}), PHASE332_SCHEDULE_MOVE_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not verdict["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG2.is_file() and RUNG3.is_file()),
    reason=(
        "the p332_ rung 2/3 fingerprint documents are not present under "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveRung:
    def test_the_ladder_holds_rung_three_judged_against_rung_two(self) -> None:
        names = [
            p.name
            for p in require_rung_ladder(FINGERPRINT_DIR, 4, PHASE332_RUNG_PREFIX)
        ]
        assert RUNG2.name in names and RUNG3.name in names
        assert (
            phase332_baseline_document_path(
                FINGERPRINT_DIR, PHASE332_SCHEDULE_MOVE_RUNG
            )
            == RUNG2
        )

    def test_every_moved_column_is_a_schedule_fact_on_or_after_the_floor(
        self,
    ) -> None:
        before = json.loads(RUNG2.read_text(encoding="utf-8"))
        after = json.loads(RUNG3.read_text(encoding="utf-8"))
        report = compare_fingerprints(before, after)
        verdict = attribute_rung(
            report,
            PHASE332_SCHEDULE_MOVE_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert verdict["non_clock_moves"], "an empty rung must be recorded as not run"
        assert set(verdict["non_clock_moves"]) <= set(DERIVED_SCHEDULE_FACT_COLUMNS)
        for matrix in fg.GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [] and detail["columns_removed"] == []
            assert detail["rows_before"] == detail["rows_after"]
            for column, seasons in detail["columns_changed"].items():
                if not fg._is_build_clock(column):
                    assert all(int(s) >= SEASON_FLOOR for s in seasons), (
                        matrix,
                        column,
                        seasons,
                    )

    def test_the_baseline_confirmation_matches_rung_two(self) -> None:
        if not BASELINE_CONFIRM.is_file():
            pytest.skip(f"{BASELINE_CONFIRM} is absent (gitignored runtime evidence)")
        rung2 = json.loads(RUNG2.read_text(encoding="utf-8"))
        confirm = json.loads(BASELINE_CONFIRM.read_text(encoding="utf-8"))
        report = compare_fingerprints(rung2, confirm)
        for matrix in fg.GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [] and detail["columns_removed"] == []
            moved = [c for c in detail["columns_changed"] if not fg._is_build_clock(c)]
            assert moved == [], (matrix, moved)

    def test_the_committed_diff_records_rung_three(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"][str(PHASE332_SCHEDULE_MOVE_RUNG)]
        assert rung["cause"] == fg.PHASE332_SCHEDULE_MOVE_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_schedule_move"
        assert rung["baseline_document"] == RUNG2.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["declared_columns"] == list(DERIVED_SCHEDULE_FACT_COLUMNS)
        assert rung["post_lock_real_moves"] == [NEUTRALISED_GAME]
        assert set(rung["moved_columns"]) <= set(DERIVED_SCHEDULE_FACT_COLUMNS)


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason=(
        "outputs/p332_rung3_gold_before/ (the before-gold copy taken for the row rule) "
        "is absent -- gitignored runtime evidence"
    ),
)


def _gold_moves(matrix: str) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    before = pd.read_parquet(GOLD_BEFORE_DIR / f"{matrix}.parquet").set_index("game_id")
    after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index("game_id")
    rung3 = json.loads(RUNG3.read_text(encoding="utf-8"))[matrix]
    # The VALUE digests (not only column_meta): a later rung that moves values without
    # changing a dtype must read as "moved on", or this would judge the wrong rebuild.
    live = fg.fingerprint_matrix(after.reset_index())["columns"]
    if {c: v for c, v in live.items() if not fg._is_build_clock(c)} != {
        c: v for c, v in rung3["columns"].items() if not fg._is_build_clock(c)
    }:
        pytest.skip("live gold has moved on since the rung-3 document was written")
    assert list(before.index) == list(after.index)
    moved: dict[str, set[str]] = {}
    for column in before.columns:
        if fg._is_build_clock(column):
            continue
        left, right = before[column], after[column]
        same = (left == right) | (left.isna() & right.isna())
        if (~same).any():
            moved[column] = set(before.index[~same])
    return before, moved


@needs_gold_copy
@needs_ladder
class TestTheRowRule:
    """Measured row by row, never inferred from the per-season digests."""

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_moved_rows_are_on_or_after_the_floor_in_the_derived_columns(
        self, matrix: str
    ) -> None:
        before, moved = _gold_moves(matrix)
        assert set(moved) <= set(DERIVED_SCHEDULE_FACT_COLUMNS), sorted(moved)
        for column, rows in moved.items():
            seasons = set(before.loc[sorted(rows), "season"].astype(int))
            assert min(seasons) >= SEASON_FLOOR, (column, seasons)

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_the_neutralised_game_moved(self, matrix: str) -> None:
        _, moved = _gold_moves(matrix)
        assert NEUTRALISED_GAME in set().union(*moved.values())

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_same_season_rows_before_the_game_did_not_move(self, matrix: str) -> None:
        """Expanding within-season statistics cannot reach a row sorted before the game."""
        before, moved = _gold_moves(matrix)
        game_week = int(before.at[NEUTRALISED_GAME, "week"])
        for column, rows in moved.items():
            same_season = before.loc[sorted(rows)]
            same_season = same_season[same_season["season"] == SEASON_FLOOR]
            assert int(same_season["week"].min()) >= game_week, column

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_off_game_moves_are_only_in_rescaled_columns(self, matrix: str) -> None:
        before, moved = _gold_moves(matrix)
        season_rows = before[before["season"] == SEASON_FLOOR]
        for column, rows in moved.items():
            if not rows - {NEUTRALISED_GAME}:
                continue
            levels = set(season_rows[column].dropna().unique())
            assert not levels <= {0.0, 1.0}, (column, sorted(rows)[:5])
