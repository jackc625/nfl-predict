"""Rung 4 of the Phase-33.2 gold ladder: the day-before weather forecast.

Plan 33.2-12 Task 3 (SPEC R6, D33.2-20: one declared cause, every moved column explained by
it). Registered per Plan 33.2-08's owned registration protocol AND dispatched: a registered
but undispatched p332_ rung 4 would fall through to Phase 30's generic path, whose ``if rung
== 4`` routes to ``_attribute_rung4`` and would still print a verdict.

THE PREDICTION IS RECORDED HERE, BEFORE THE REBUILD (commit order is the evidence): the
removed columns, the widths before (measured on ``p332_rung3c.json``) and after (each exactly
two narrower), and the explainable family, which is derived from the weather builder's own
declaration. The live classes then confirm the rebuild against it.

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
    PHASE332_KICKOFF_HOUR_STEP,
    PHASE332_RUNG_PREFIX,
    PHASE332_WEATHER_RUNG,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    phase332_weather_columns,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP3C = rung_document_path(
    FINGERPRINT_DIR, PHASE332_KICKOFF_HOUR_STEP, PHASE332_RUNG_PREFIX
)
RUNG4 = rung_document_path(FINGERPRINT_DIR, PHASE332_WEATHER_RUNG, PHASE332_RUNG_PREFIX)
GOLD_BEFORE_DIR = Path("outputs/p332_rung4_gold_before")
GOLD_AFTER_DIR = Path("data/gold")

# ---------------------------------------------------------------------------
# THE PREDICTION, recorded before the rebuild ran.
# ---------------------------------------------------------------------------

PREDICTED_REMOVED: tuple[str, ...] = ("precip_mm", "raw_precip_mm")
#: MEASURED on p332_rung3c.json (the rung's baseline), features_wp / _ats / _ou.
PREDICTED_WIDTHS_BEFORE: tuple[int, int, int] = (195, 196, 195)
PREDICTED_WIDTHS_AFTER: tuple[int, int, int] = (193, 194, 193)
PREDICTED_ROWS: int = 6499


def _report(
    changed: dict[str, list[str]],
    *,
    removed: tuple[str, ...] = PREDICTED_REMOVED,
    width_after: int = 193,
) -> dict:
    matrix = {
        "width_before": 195,
        "width_after": width_after,
        "rows_before": PREDICTED_ROWS,
        "rows_after": PREDICTED_ROWS,
        "rows_per_season_before": {},
        "rows_per_season_after": {},
        "columns_added": [],
        "columns_removed": list(removed),
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
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


class TestTheRungIsRegisteredAndDispatched:
    def test_it_is_numbered_four_with_a_cause(self):
        assert PHASE332_WEATHER_RUNG == 4
        causes = fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert causes[PHASE332_WEATHER_RUNG] == fg.PHASE332_WEATHER_RUNG_CAUSE

    def test_both_dispatch_tables_carry_it(self):
        assert (
            fg.PHASE332_RUNG_SIGNATURES[PHASE332_WEATHER_RUNG]
            is fg.PHASE332_WEATHER_RUNG_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_WEATHER_RUNG]
            is fg._attribute_p332_weather
        )

    def test_it_is_judged_by_its_own_attributor_not_phase_30s(self, monkeypatch):
        calls: list[str] = []
        original = fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_WEATHER_RUNG]

        def spy(*args, **kwargs):
            calls.append("p332_weather")
            return original(*args, **kwargs)

        monkeypatch.setitem(fg.PHASE332_RUNG_ATTRIBUTORS, PHASE332_WEATHER_RUNG, spy)
        verdict = attribute_rung(
            _report({"temp_f": ["2002", "2025"]}),
            PHASE332_WEATHER_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332_weather"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]

    def test_no_new_prefix_branch_was_added(self):
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_it_is_judged_against_step_3c(self, tmp_path):
        assert (
            phase332_baseline_document_path(tmp_path, PHASE332_WEATHER_RUNG).name
            == STEP3C.name
        )

    def test_the_signature_records_the_provider_mismatch(self):
        signature = fg.PHASE332_WEATHER_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["columns_removed"] == PREDICTED_REMOVED
        assert "+1.56 mph" in str(signature["provider_mismatch"])
        assert "5.0 mph" in str(signature["provider_mismatch"])


class TestThePredictionAndTheJudge:
    def test_the_family_is_derived_from_the_builder_declaration(self):
        from features.weather import WEATHER_FEATURE_COLUMNS_BY_BUILDER

        family = phase332_weather_columns()
        assert set(WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]) <= family
        assert "raw_weather_severity" in family
        assert not family & {"home_rest_days", "venue_cold_climate", "elo_diff"}

    def test_the_predicted_widths_are_exactly_two_narrower(self):
        assert (
            tuple(b - len(PREDICTED_REMOVED) for b in PREDICTED_WIDTHS_BEFORE)
            == PREDICTED_WIDTHS_AFTER
        )

    def test_a_non_weather_column_moving_is_unattributed(self):
        verdict = attribute_rung(
            _report({"temp_f": ["2010"], "home_rest_days": ["2010"]}),
            PHASE332_WEATHER_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_removing_anything_but_the_two_columns_blocks(self):
        verdict = attribute_rung(
            _report({"temp_f": ["2010"]}, removed=("precip_mm",), width_after=194),
            PHASE332_WEATHER_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["blocking"]

    def test_a_width_that_does_not_fall_by_two_blocks(self):
        verdict = attribute_rung(
            _report({"temp_f": ["2010"]}, width_after=195),
            PHASE332_WEATHER_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["blocking"]


needs_ladder = pytest.mark.skipif(
    not (STEP3C.is_file() and RUNG4.is_file()),
    reason=(
        "the p332_ step 3c / rung 4 fingerprint documents are not present under "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveRung:
    def _documents(self) -> tuple[dict, dict]:
        return (
            json.loads(STEP3C.read_text(encoding="utf-8")),
            json.loads(RUNG4.read_text(encoding="utf-8")),
        )

    def test_the_widths_before_and_after_are_the_predicted_ones(self):
        before, after = self._documents()
        assert tuple(before[m]["width"] for m in fg.GOLD_MATRICES) == (
            PREDICTED_WIDTHS_BEFORE
        )
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == (
            PREDICTED_WIDTHS_AFTER
        )
        assert all(after[m]["rows"] == PREDICTED_ROWS for m in fg.GOLD_MATRICES)

    def test_the_attribution_is_clean(self):
        before, after = self._documents()
        verdict = attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_WEATHER_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= phase332_weather_columns()

    def test_the_committed_diff_records_the_rung_and_the_mismatch(self):
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"][str(PHASE332_WEATHER_RUNG)]
        assert rung["cause"] == fg.PHASE332_WEATHER_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_weather"
        assert rung["baseline_document"] == STEP3C.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["removed_columns"] == list(PREDICTED_REMOVED)
        assert "+1.56 mph" in rung["provider_mismatch"]


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason="outputs/p332_rung4_gold_before/ is absent -- gitignored runtime evidence",
)


@needs_gold_copy
@needs_ladder
class TestTheRowRule:
    @staticmethod
    def _frames(matrix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        before = pd.read_parquet(GOLD_BEFORE_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        live = fg.fingerprint_matrix(after.reset_index())["columns"]
        recorded = json.loads(RUNG4.read_text(encoding="utf-8"))[matrix]["columns"]
        if {c: v for c, v in live.items() if not fg._is_build_clock(c)} != {
            c: v for c, v in recorded.items() if not fg._is_build_clock(c)
        }:
            pytest.skip("live gold has moved on since the rung-4 document was written")
        return before, after

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_the_same_games_and_only_the_two_columns_left(self, matrix):
        before, after = self._frames(matrix)
        assert list(before.index) == list(after.index)
        assert set(before.columns) - set(after.columns) == set(PREDICTED_REMOVED)
        assert set(after.columns) <= set(before.columns)

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_no_non_weather_column_moved_on_any_row(self, matrix):
        before, after = self._frames(matrix)
        family = phase332_weather_columns()
        moved = []
        for column in after.columns:
            if fg._is_build_clock(column) or column.lower() in family:
                continue
            left, right = before[column], after[column]
            same = (left == right) | (left.isna() & right.isna())
            if not same.all():
                moved.append(column)
        assert moved == [], f"non-weather columns moved: {moved}"
