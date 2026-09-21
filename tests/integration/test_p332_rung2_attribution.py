"""Rung 2 of the Phase-33.2 gold ladder: the 2025 international venues.

Plan 33.2-09 Task 3 (D33.2-20, SPEC R8 venue half). Rung 2 is registered AND dispatched
per Plan 33.2-08's <owned_protocol_rung_registration>: its cause, signature and attributor
sit in the `p332_` tables, and it is judged by ``_attribute_p332_venue`` -- never by Phase
30's ``_attribute_rung2``, a blanket judge that cannot fail on a moved column and would
print a clean verdict for a wrong rebuild just as readily as for a right one. The dispatch
is ASSERTED below, because a pass under the wrong judge is indistinguishable from a pass
under the right one.

THE EXPLAINABLE SET WAS DERIVED BEFORE THE REBUILD RAN and is pinned here as
``DERIVED_STADIUM_DEPENDENT_COLUMNS``: the contextual columns whose value differs when one
synthetic game is placed at each venue in data/venues.json. Weather is deliberately not in
it -- gold weather is read from silver ``weather_features`` by game id, which this rung does
not touch.

THE ROW RULE is checked against a copy of the before-gold taken just before the rebuild
(``outputs/p332_rung2_gold_before/``, gitignored): only the seven corrected games may move
before normalization, and a row outside them may move only in a column expanding
normalization rescales, only in 2025. The live half reads gitignored evidence and skips,
naming it, on a checkout that has not run the ladder.
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
    PHASE332_VENUE_RUNG,
    assert_ladder_is_recoverable,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    require_rung_ladder,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
CORRECTIONS = Path("config/international_venue_corrections.toml")
RUNG1 = rung_document_path(FINGERPRINT_DIR, 1, PHASE332_RUNG_PREFIX)
RUNG2 = rung_document_path(FINGERPRINT_DIR, PHASE332_VENUE_RUNG, PHASE332_RUNG_PREFIX)
BASELINE_CONFIRM = (
    FINGERPRINT_DIR / fg.PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION_DOCUMENT
)
GOLD_BEFORE_DIR = Path("outputs/p332_rung2_gold_before")
GOLD_AFTER_DIR = Path("data/gold")

# DERIVED 2026-09-21 by scripts.fingerprint_gold.phase332_stadium_dependent_columns(),
# BEFORE the rung-2 rebuild ran: the contextual columns whose value differs across one
# probe game placed at every venue in data/venues.json.
DERIVED_STADIUM_DEPENDENT_COLUMNS: tuple[str, ...] = (
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

CORRECTED_SEASON = "2025"


def _corrected_game_ids() -> set[str]:
    record = tomllib.loads(CORRECTIONS.read_text(encoding="utf-8"))
    return {entry["game_id"] for entry in record["correction"]}


def _matrix(changed: dict[str, list[str]]) -> dict:
    """One synthetic compare_fingerprints matrix entry: same shape, given moves."""
    return {
        "width_before": 195,
        "width_after": 195,
        "rows_before": 6499,
        "rows_after": 6499,
        "rows_per_season_before": {},
        "rows_per_season_after": {},
        "columns_added": [],
        "columns_removed": [],
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
        assert causes[PHASE332_VENUE_RUNG] == fg.PHASE332_VENUE_RUNG_CAUSE
        assert fg.PHASE332_VENUE_RUNG_CAUSE
        assert (
            fg.PHASE332_RUNG_SIGNATURES[PHASE332_VENUE_RUNG]
            is fg.PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_VENUE_RUNG]
            is fg._attribute_p332_venue
        )

    def test_the_signature_handed_out_is_a_copy(self) -> None:
        signature = fg._expected_signature(
            PHASE332_VENUE_RUNG, prefix=PHASE332_RUNG_PREFIX
        )
        assert signature == fg.PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE
        assert signature is not fg.PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True

    def test_rung_two_is_judged_by_its_own_attributor_not_phase_30s(
        self, monkeypatch
    ) -> None:
        calls: list[str] = []
        original = fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_VENUE_RUNG]

        def spy(*args, **kwargs):
            calls.append("p332_venue")
            return original(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "p332_ rung 2 fell through to Phase 30's _attribute_rung2"
            )

        monkeypatch.setitem(fg.PHASE332_RUNG_ATTRIBUTORS, PHASE332_VENUE_RUNG, spy)
        monkeypatch.setattr(fg, "_attribute_rung2", forbidden)
        verdict = attribute_rung(
            _report({"venue_elevation_ft": [CORRECTED_SEASON]}),
            PHASE332_VENUE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332_venue"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]
        assert verdict["cause"] == fg.PHASE332_VENUE_RUNG_CAUSE

    def test_the_phase_30_judge_would_have_passed_a_wrong_rebuild(self) -> None:
        """The trap, measured: an Elo column moving is waved through with no prefix."""
        diff = _report({"home_elo_rating": ["2019", CORRECTED_SEASON]})
        phase30 = attribute_rung(diff, 2)
        own = attribute_rung(
            diff, PHASE332_VENUE_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not own["ok"]
        assert "home_elo_rating" in own["matrices"]["features_wp"]["unattributed"]
        assert (
            "home_elo_rating" not in phase30["matrices"]["features_wp"]["unattributed"]
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        """Registration goes through the tables; the `p332_` branch count stays at two."""
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_rung_two_registers_no_retaken_baseline(self) -> None:
        """The baseline was confirmed unchanged, so rung 2 judges against rung 1."""
        assert PHASE332_VENUE_RUNG not in PHASE332_RETAKEN_BASELINES
        assert "CONFIRMED" in fg.PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION


class TestTheExplainableSetIsDerived:
    def test_the_pinned_set_is_what_the_derivation_returns(self) -> None:
        assert (
            fg.phase332_stadium_dependent_columns() == DERIVED_STADIUM_DEPENDENT_COLUMNS
        )

    def test_no_weather_column_is_in_the_set(self) -> None:
        from features.weather import WEATHER_FEATURE_COLUMNS

        assert not set(DERIVED_STADIUM_DEPENDENT_COLUMNS) & set(WEATHER_FEATURE_COLUMNS)

    def test_a_venue_independent_contextual_column_is_not_in_the_set(self) -> None:
        for column in ("home_rest_days", "thursday_game", "is_divisional"):
            assert column in fg.phase331_venue_family()
            assert column not in DERIVED_STADIUM_DEPENDENT_COLUMNS


class TestTheAttributorRefusesWhatTheCauseCannotExplain:
    def test_a_stadium_dependent_column_moving_only_in_2025_is_attributed(self) -> None:
        verdict = attribute_rung(
            _report({c: [CORRECTED_SEASON] for c in DERIVED_STADIUM_DEPENDENT_COLUMNS}),
            PHASE332_VENUE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]

    def test_a_stadium_dependent_column_moving_outside_2025_is_unattributed(
        self,
    ) -> None:
        verdict = attribute_rung(
            _report({"venue_outdoor": ["2024", CORRECTED_SEASON]}),
            PHASE332_VENUE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]
        assert verdict["matrices"]["features_wp"]["unattributed"] == ["venue_outdoor"]

    def test_a_weather_column_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"weather_severity_score": [CORRECTED_SEASON]}),
            PHASE332_VENUE_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        verdict = attribute_rung(
            _report({}), PHASE332_VENUE_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not verdict["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG1.is_file() and RUNG2.is_file()),
    reason=(
        "the p332_ rung 1/2 fingerprint documents are not present under "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveRung:
    def test_both_documents_exist_and_the_ladder_is_recoverable(self) -> None:
        names = [
            p.name
            for p in require_rung_ladder(FINGERPRINT_DIR, 3, PHASE332_RUNG_PREFIX)
        ]
        assert names[-2:] == [RUNG1.name, RUNG2.name]
        assert_ladder_is_recoverable(FINGERPRINT_DIR, 3, PHASE332_RUNG_PREFIX)
        assert phase332_baseline_document_path(
            FINGERPRINT_DIR, PHASE332_VENUE_RUNG
        ) == (RUNG1)

    def test_every_moved_column_is_stadium_dependent_and_moved_only_in_2025(
        self,
    ) -> None:
        before = json.loads(RUNG1.read_text(encoding="utf-8"))
        after = json.loads(RUNG2.read_text(encoding="utf-8"))
        report = compare_fingerprints(before, after)
        verdict = attribute_rung(
            report,
            PHASE332_VENUE_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert verdict["non_clock_moves"], "an empty rung must be recorded as not run"
        assert set(verdict["non_clock_moves"]) <= set(DERIVED_STADIUM_DEPENDENT_COLUMNS)
        for matrix in fg.GOLD_MATRICES:
            for column, seasons in report[matrix]["columns_changed"].items():
                if not fg._is_build_clock(column):
                    assert seasons == [CORRECTED_SEASON], (matrix, column, seasons)

    def test_the_baseline_confirmation_matches_rung_one(self) -> None:
        if not BASELINE_CONFIRM.is_file():
            pytest.skip(f"{BASELINE_CONFIRM} is absent (gitignored runtime evidence)")
        rung1 = json.loads(RUNG1.read_text(encoding="utf-8"))
        confirm = json.loads(BASELINE_CONFIRM.read_text(encoding="utf-8"))
        report = compare_fingerprints(rung1, confirm)
        for matrix in fg.GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [] and detail["columns_removed"] == []
            moved = [c for c in detail["columns_changed"] if not fg._is_build_clock(c)]
            assert moved == [], (matrix, moved)

    def test_the_committed_diff_records_rung_two(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"][str(PHASE332_VENUE_RUNG)]
        assert rung["cause"] == fg.PHASE332_VENUE_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_venue"
        assert rung["baseline_document"] == RUNG1.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["declared_columns"] == list(DERIVED_STADIUM_DEPENDENT_COLUMNS)
        assert set(rung["moved_columns"]) <= set(DERIVED_STADIUM_DEPENDENT_COLUMNS)
        assert all(
            seasons == [CORRECTED_SEASON] for seasons in rung["moved_seasons"].values()
        )


def _value_digests(document: dict) -> dict:
    """Per-column per-season value digests, without the build clock."""
    return {
        column: digests
        for column, digests in document["columns"].items()
        if not fg._is_build_clock(column)
    }


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason=(
        "outputs/p332_rung2_gold_before/ (the before-gold copy taken for the row rule) "
        "is absent -- gitignored runtime evidence"
    ),
)


@needs_gold_copy
@needs_ladder
class TestTheRowRule:
    """Measured row by row, never inferred from the per-season digests."""

    @staticmethod
    def _moves(matrix: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
        before = pd.read_parquet(GOLD_BEFORE_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        after_digest = fg.fingerprint_matrix(after.reset_index())
        rung2 = json.loads(RUNG2.read_text(encoding="utf-8"))[matrix]
        # The VALUE digests, not only column_meta (dtype and null counts): a later rung
        # that moves values without changing any dtype must still read as "moved on".
        if _value_digests(after_digest) != _value_digests(rung2):
            pytest.skip("live gold has moved on since the rung-2 document was written")
        assert list(before.index) == list(after.index)
        moved: dict[str, set[str]] = {}
        for column in before.columns:
            if fg._is_build_clock(column):
                continue
            left, right = before[column], after[column]
            same = (left == right) | (left.isna() & right.isna())
            changed = set(before.index[~same])
            if changed:
                moved[column] = changed
        return before, after, moved

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_moved_rows_are_2025_and_in_the_derived_columns(self, matrix: str) -> None:
        before, _, moved = self._moves(matrix)
        assert set(moved) <= set(DERIVED_STADIUM_DEPENDENT_COLUMNS), sorted(moved)
        for column, rows in moved.items():
            seasons = set(before.loc[sorted(rows), "season"].astype(int))
            assert seasons == {2025}, (column, seasons)

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_every_corrected_game_moved(self, matrix: str) -> None:
        _, _, moved = self._moves(matrix)
        touched = set().union(*moved.values()) if moved else set()
        assert _corrected_game_ids() <= touched

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_off_game_moves_are_only_in_rescaled_columns(self, matrix: str) -> None:
        """A level-preserved indicator can move on the seven games only.

        A column whose 2025 before-values are all in {0, 1} was left at its level by
        normalization, so it carries no expanding statistic that could reach another row.
        """
        before, _, moved = self._moves(matrix)
        corrected = _corrected_game_ids()
        before_2025 = before[before["season"] == 2025]
        for column, rows in moved.items():
            off_game = rows - corrected
            if not off_game:
                continue
            levels = set(before_2025[column].dropna().unique())
            assert not levels <= {0.0, 1.0}, (
                f"{column} is a level-preserved indicator yet moved on "
                f"{sorted(off_game)[:5]}"
            )
