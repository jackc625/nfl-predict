"""Extra step 8d of the Phase-33.2 gold ladder: an unscorable cell is blank, not zero.

Owner ruling 2026-09-22 (deferred-items.md, "Normalization maps an early-season MEASURED
value to the neutral 0.0 ...", option (a) -- "LEAVE THE CELL BLANK"), run THIRD in Plan
33.2-19's dispatch, after steps 8b and 8c and before step 8e and rung 9. ONE cause:
``expanding_normalize``'s terminal fill writes the neutral 0.0 only where the INPUT VALUE
was absent, so a position whose value EXISTS and whose expanding statistic could not be
formed at all comes back BLANK, beside whatever coverage flag it already carries.

THE PREDICTION, declared before the rebuild in commit ``87fb610`` (the constants below
equal ``scripts.fingerprint_gold``'s ``PHASE332_UNSCORABLE_BLANK_STEP_*``) and MEASURED on
a full scratch build whose every other cell reproduces production gold EXACTLY:

* exactly 124 columns move, each only in the seasons declared beside it, identically in
  all three matrices;
* exactly 182 cells per matrix move, and every one of them moves from exactly 0.0 to
  blank -- so each declared column's NULL COUNT rises by exactly its declared cell count
  and no other null count moves. This step is a null-count step, so the counts are half
  the prediction rather than a footnote;
* the 69 feature columns the terminal fill cannot reach are declared UNMOVED: the
  level-preserved coverage flags and weather indicators, the display-only ``raw_*``
  passthroughs, the five constant market columns, the eight never-populated defensive
  team-form copies, and the families whose unscorable cells are already blank through
  ``preserve_missing_cols``;
* nothing added or removed; widths 201/202/201 and rows 6,499 unchanged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import scripts.fingerprint_gold as fg
import tests.phase33_state as state
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_LOCK_ORDER_STEP,
    PHASE332_RUNG_PREFIX,
    PHASE332_UNSCORABLE_BLANK_STEP,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP8C = rung_document_path(
    FINGERPRINT_DIR, PHASE332_LOCK_ORDER_STEP, PHASE332_RUNG_PREFIX
)
STEP8D = rung_document_path(
    FINGERPRINT_DIR, PHASE332_UNSCORABLE_BLANK_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step8d_before.json")
DIGEST_AFTER = Path("outputs/p332_step8d_after.json")

PREDICTED_WIDTHS: tuple[int, int, int] = (201, 202, 201)
PREDICTED_ROWS: int = 6499
PREDICTED_CELLS: int = 182

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)


def _report(
    changed: dict[str, list[str]], nulls: dict[str, tuple[int, int]] | None = None
) -> dict:
    cells = fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS_BY_COLUMN
    nulls = nulls or {}
    matrix = {
        "width_before": 201,
        "width_after": 201,
        "rows_before": PREDICTED_ROWS,
        "rows_after": PREDICTED_ROWS,
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
                "null_count_before": nulls.get(column, (0, cells.get(column, 0)))[0],
                "null_count_after": nulls.get(column, (0, cells.get(column, 0)))[1],
                "discrete_indicator_before": False,
                "discrete_indicator_after": False,
            }
            for column, seasons in changed.items()
        },
    }
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


def _all_declared(**overrides) -> dict[str, list[str]]:
    changed = {
        column: list(seasons)
        for column, seasons in (
            fg.PHASE332_UNSCORABLE_BLANK_STEP_SEASONS_BY_COLUMN.items()
        )
    }
    changed.update(overrides)
    return changed


def _verdict(
    changed: dict[str, list[str]], nulls: dict[str, tuple[int, int]] | None = None
) -> dict:
    return attribute_rung(
        _report(changed, nulls),
        PHASE332_UNSCORABLE_BLANK_STEP,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_follows_rung_eight_and_is_judged_against_step_8c(
        self, tmp_path
    ) -> None:
        assert PHASE332_UNSCORABLE_BLANK_STEP == "8d"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["8d"] == 8
        assert (
            phase332_baseline_document_path(tmp_path, "8d").name == "p332_rung8c.json"
        )
        assert fg._ladder_predecessors("8d", PHASE332_RUNG_PREFIX)[-1] == "8c"

    def test_it_is_dispatched(self) -> None:
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["8d"]
            == fg.PHASE332_UNSCORABLE_BLANK_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["8d"]
            is fg.PHASE332_UNSCORABLE_BLANK_STEP_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["8d"]
            is fg._attribute_p332_unscorable_blank
        )

    def test_its_id_cannot_collide_with_rung_nine(self) -> None:
        assert PHASE332_UNSCORABLE_BLANK_STEP not in (8, 9, "8", "9")
        assert rung_document_path(FINGERPRINT_DIR, "8d", PHASE332_RUNG_PREFIX).name == (
            "p332_rung8d.json"
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_UNSCORABLE_BLANK_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"
        assert signature["columns_added"] == "empty"
        assert signature["columns_removed"] == "empty"


class TestThePrediction:
    def test_all_three_halves_are_non_vacuous_and_agree(self) -> None:
        declared = fg.PHASE332_UNSCORABLE_BLANK_STEP_SEASONS_BY_COLUMN
        cells = fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS_BY_COLUMN
        unmoved = fg.PHASE332_UNSCORABLE_BLANK_STEP_PREDICTED_UNMOVED
        assert len(declared) == 124
        assert set(declared) == set(cells)
        assert len(unmoved) == 69
        assert not set(declared) & set(unmoved)
        assert all(seasons for seasons in declared.values())
        assert all(count > 0 for count in cells.values())
        assert fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS == PREDICTED_CELLS

    def test_the_unmoved_half_names_the_reasons_a_column_is_out_of_reach(self) -> None:
        unmoved = set(fg.PHASE332_UNSCORABLE_BLANK_STEP_PREDICTED_UNMOVED)
        assert {"home_snap_coverage", "weather_coverage"} <= unmoved  # level-preserved
        assert {"raw_temp_f", "raw_wind_mph"} <= unmoved  # display-only
        assert {"snapshot_spread", "total_movement"} <= unmoved  # constant since rung 5
        assert {"home_def_rolling_cpoe"} <= unmoved  # never populated
        assert {"home_off_rolling_opp_adj_epa_per_play"} <= unmoved  # already blank

    def test_the_declared_cells_sum_to_the_recorded_total(self) -> None:
        cells = fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS_BY_COLUMN
        assert sum(cells.values()) == state.P332_19_STEP8D_CELLS_PER_MATRIX


class TestTheJudge:
    def test_the_declared_moves_are_attributed(self) -> None:
        verdict = _verdict(_all_declared())
        assert verdict["ok"], verdict["failures"]

    def test_a_move_outside_a_columns_declared_seasons_is_unattributed(self) -> None:
        assert not _verdict(_all_declared(is_home_game=["1999"]))["ok"]

    def test_a_move_in_a_column_declared_unmoved_is_unattributed(self) -> None:
        for column in ("weather_coverage", "raw_temp_f", "snapshot_spread"):
            assert not _verdict(_all_declared(**{column: ["2019"]}))["ok"], column

    def test_a_declared_column_that_did_not_move_fails(self) -> None:
        changed = _all_declared()
        del changed["is_home_game"]
        assert not _verdict(changed)["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]

    def test_a_null_count_that_moved_by_the_wrong_amount_fails(self) -> None:
        """The null delta IS the step, so a value move without it is a finding."""
        assert not _verdict(_all_declared(), nulls={"is_home_game": (0, 99)})["ok"]

    def test_a_declared_column_whose_null_count_did_not_move_fails(self) -> None:
        assert not _verdict(_all_declared(), nulls={"is_home_game": (0, 0)})["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP8C.is_file() and STEP8D.is_file()),
    reason="the p332_ step-8c or step-8d fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(STEP8C.read_text(encoding="utf-8")),
            json.loads(STEP8D.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_UNSCORABLE_BLANK_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert verdict["non_clock_moves"] == sorted(state.P332_19_STEP8D_MOVED_COLUMNS)
        assert verdict["build_clock_moves"] == ["feature_timestamp"]

    def test_the_widths_and_rows_did_not_move(self) -> None:
        before, after = self._documents()
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["width"] == after[matrix]["width"]
            assert before[matrix]["rows"] == after[matrix]["rows"] == PREDICTED_ROWS
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == PREDICTED_WIDTHS

    def test_every_moved_column_moved_in_all_three_matrices_in_the_same_seasons(
        self,
    ) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        declared = fg.PHASE332_UNSCORABLE_BLANK_STEP_SEASONS_BY_COLUMN
        for matrix in fg.GOLD_MATRICES:
            moved = {
                column: tuple(seasons)
                for column, seasons in report[matrix]["columns_changed"].items()
                if not fg._is_build_clock(column)
            }
            assert set(moved) == set(declared), matrix
            for column, seasons in moved.items():
                assert seasons == declared[column], (matrix, column)

    def test_no_column_declared_unmoved_moved(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        unmoved = set(fg.PHASE332_UNSCORABLE_BLANK_STEP_PREDICTED_UNMOVED)
        for matrix in fg.GOLD_MATRICES:
            assert not unmoved & set(report[matrix]["columns_changed"]), matrix

    def test_every_null_count_moved_by_exactly_its_declared_cell_count(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        cells = fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS_BY_COLUMN
        for matrix in fg.GOLD_MATRICES:
            total = 0
            for column, move in report[matrix]["column_details"].items():
                if fg._is_build_clock(column):
                    continue
                delta = move["null_count_after"] - move["null_count_before"]
                assert delta == cells.get(column, 0), (matrix, column, delta)
                total += delta
            assert total == PREDICTED_CELLS, matrix

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_UNSCORABLE_BLANK_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_19_STEP8D_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["8d"]
        assert step["cause"] == fg.PHASE332_UNSCORABLE_BLANK_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_unscorable_blank"
        assert step["baseline_document"] == STEP8C.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]
        assert step["declared_cells"] == PREDICTED_CELLS
        assert step["measured_cells"] == [PREDICTED_CELLS] * 3
        assert step["moved_columns"] == sorted(state.P332_19_STEP8D_MOVED_COLUMNS)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-8d digest bracket documents are gitignored runtime state",
)
class TestTheDigestBracket:
    def test_only_the_declared_paths_moved(self) -> None:
        from tests.data_boundary import diff_digests

        before = json.loads(DIGEST_BEFORE.read_text(encoding="utf-8"))
        after = json.loads(DIGEST_AFTER.read_text(encoding="utf-8"))
        diff = diff_digests(before, after)
        moved = set(diff["added"]) | set(diff["removed"]) | set(diff["changed"])
        assert moved - DECLARED_PATHS == set()
        assert not diff.get("mixed")
        assert diff["removed"] == []
        assert diff["added"] == []
