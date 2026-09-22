"""Extra step 8c of the Phase-33.2 gold ladder: normalization statistics ordered by lock.

Owner ruling 2026-09-22 (deferred-items.md, "(b) normalization statistics ordered by lock
instant"), run SECOND in Plan 33.2-19's dispatch, after step 8b and before step 8d and rung
9. ONE cause: a row's expanding mean and standard deviation are computed over every row of
its season whose game LOCK is at or before that row's lock, so a same-week game whose lock
is LATER can no longer enter an earlier-locking game's statistic, and games sharing a lock
share one statistic whatever order their rows arrive in.

THE PREDICTION, declared before the rebuild in commit ``5a21f5e`` (the constants below equal
``scripts.fingerprint_gold``'s ``PHASE332_LOCK_ORDER_STEP_*``), measured on the same tail
replay step 8b used, whose CONTROL -- the same replay from a worktree at rung 8's commit
``2d0eb97`` -- reproduces production gold exactly:

* exactly 138 columns move -- every column the normalizer z-scores -- each only in the
  seasons declared beside it, and identically in all three matrices;
* the 55 feature columns the normalizer never z-scores are declared UNMOVED: the
  level-preserved coverage flags and weather indicators, the display-only ``raw_*``
  passthroughs, the five market columns constant since rung 5, and the eight
  never-populated defensive team-form copies. That NEGATIVE half is what makes a
  prediction covering every normalized column discriminating rather than a blanket;
* nothing added or removed; widths 201/202/201 and rows 6,499 unchanged; no null count
  moves -- re-ordering a statistic neither creates nor removes a missing value.

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
    PHASE332_WINSORIZATION_STEP,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP8B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_WINSORIZATION_STEP, PHASE332_RUNG_PREFIX
)
STEP8C = rung_document_path(
    FINGERPRINT_DIR, PHASE332_LOCK_ORDER_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step8c_before.json")
DIGEST_AFTER = Path("outputs/p332_step8c_after.json")

PREDICTED_WIDTHS: tuple[int, int, int] = (201, 202, 201)
PREDICTED_ROWS: int = 6499

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)


def _report(changed: dict[str, list[str]]) -> dict:
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
                "null_count_before": 0,
                "null_count_after": 0,
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
        for column, seasons in fg.PHASE332_LOCK_ORDER_STEP_SEASONS_BY_COLUMN.items()
    }
    changed.update(overrides)
    return changed


def _verdict(changed: dict[str, list[str]]) -> dict:
    return attribute_rung(
        _report(changed), PHASE332_LOCK_ORDER_STEP, rung_prefix=PHASE332_RUNG_PREFIX
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_follows_rung_eight_and_is_judged_against_step_8b(
        self, tmp_path
    ) -> None:
        assert PHASE332_LOCK_ORDER_STEP == "8c"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["8c"] == 8
        assert (
            phase332_baseline_document_path(tmp_path, "8c").name == "p332_rung8b.json"
        )
        assert fg._ladder_predecessors("8c", PHASE332_RUNG_PREFIX)[-1] == "8b"

    def test_it_is_dispatched(self) -> None:
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["8c"]
            == fg.PHASE332_LOCK_ORDER_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["8c"]
            is fg.PHASE332_LOCK_ORDER_STEP_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["8c"] is fg._attribute_p332_lock_order

    def test_its_id_cannot_collide_with_rung_nine(self) -> None:
        assert PHASE332_LOCK_ORDER_STEP not in (8, 9, "8", "9")
        assert rung_document_path(FINGERPRINT_DIR, "8c", PHASE332_RUNG_PREFIX).name == (
            "p332_rung8c.json"
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_LOCK_ORDER_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"
        assert signature["columns_added"] == "empty"
        assert signature["columns_removed"] == "empty"


class TestThePrediction:
    def test_both_halves_are_non_vacuous_and_disjoint(self) -> None:
        declared = fg.PHASE332_LOCK_ORDER_STEP_SEASONS_BY_COLUMN
        unmoved = fg.PHASE332_LOCK_ORDER_STEP_PREDICTED_UNMOVED
        assert len(declared) == 138
        assert len(unmoved) == 55
        assert not set(declared) & set(unmoved)
        assert all(seasons for seasons in declared.values())

    def test_the_unmoved_half_names_the_four_reasons_a_column_is_not_z_scored(
        self,
    ) -> None:
        unmoved = set(fg.PHASE332_LOCK_ORDER_STEP_PREDICTED_UNMOVED)
        assert {"home_snap_coverage", "weather_coverage"} <= unmoved  # level-preserved
        assert {"raw_temp_f", "raw_wind_mph"} <= unmoved  # display-only
        assert {"snapshot_spread", "total_movement"} <= unmoved  # constant since rung 5
        assert {"home_def_rolling_cpoe"} <= unmoved  # never populated

    def test_no_declared_column_is_a_coverage_flag(self) -> None:
        assert not [
            c for c in fg.PHASE332_LOCK_ORDER_STEP_COLUMNS if c.endswith("_coverage")
        ]


class TestTheJudge:
    def test_the_declared_moves_are_attributed(self) -> None:
        verdict = _verdict(_all_declared())
        assert verdict["ok"], verdict["failures"]

    def test_a_move_outside_a_columns_declared_seasons_is_unattributed(self) -> None:
        assert not _verdict(_all_declared(home_elo=["2001", "2002"]))["ok"]

    def test_a_move_in_a_column_declared_unmoved_is_unattributed(self) -> None:
        for column in ("weather_coverage", "raw_temp_f", "snapshot_spread"):
            assert not _verdict(_all_declared(**{column: ["2019"]}))["ok"], column

    def test_a_declared_column_that_did_not_move_fails(self) -> None:
        changed = _all_declared()
        del changed["home_elo"]
        assert not _verdict(changed)["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP8B.is_file() and STEP8C.is_file()),
    reason="the p332_ step-8b or step-8c fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(STEP8B.read_text(encoding="utf-8")),
            json.loads(STEP8C.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_LOCK_ORDER_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert verdict["non_clock_moves"] == sorted(state.P332_19_STEP8C_MOVED_COLUMNS)
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
        declared = fg.PHASE332_LOCK_ORDER_STEP_SEASONS_BY_COLUMN
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
        unmoved = set(fg.PHASE332_LOCK_ORDER_STEP_PREDICTED_UNMOVED)
        for matrix in fg.GOLD_MATRICES:
            assert not unmoved & set(report[matrix]["columns_changed"]), matrix

    def test_no_null_count_moved(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        for matrix in fg.GOLD_MATRICES:
            for column, move in report[matrix]["column_details"].items():
                if fg._is_build_clock(column):
                    continue
                assert move["null_count_before"] == move["null_count_after"], (
                    matrix,
                    column,
                )

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_LOCK_ORDER_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_19_STEP8C_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["8c"]
        assert step["cause"] == fg.PHASE332_LOCK_ORDER_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_lock_order"
        assert step["baseline_document"] == STEP8B.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]
        assert step["moved_columns"] == sorted(state.P332_19_STEP8C_MOVED_COLUMNS)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-8c digest bracket documents are gitignored runtime state",
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
