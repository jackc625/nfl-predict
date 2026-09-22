"""Extra step 8b of the Phase-33.2 gold ladder: winsorization stops fitting on the season it clips.

Owner ruling 2026-09-22 (deferred-items.md, "(a) no winsorization in a self-fitting season"),
run FIRST in Plan 33.2-19's dispatch, before steps 8c / 8d and rung 9. ONE cause: a season's
q01/q99 clip bounds are fitted only on strictly-prior seasons, and a season with no usable
strictly-prior fit -- the earliest season, or a column's first populated season -- is left
UNCLIPPED instead of fitting a bound on the season it is applied to. NOT inside rung 8 or rung
9: it follows rung 8 and step 8c is judged against it (D33.2-20).

THE PREDICTION, declared before the rebuild in commit ``eb9154f`` (the constants below equal
``scripts.fingerprint_gold``'s ``PHASE332_WINSORIZATION_STEP_*``), measured on a replay of the
build's tail whose CONTROL -- the same replay run from a worktree at rung 8's own commit
``2d0eb97`` -- reproduces production gold exactly:

* exactly 97 columns move, each only in the seasons declared beside it, in all three matrices;
* the shape is one rule seen twice: a column's first fitted season S moves because it is no
  longer clipped, and S+1 moves because its normalization bootstrap reads S's statistics. The
  five display-only ``raw_*`` columns move in S ALONE -- they are excluded from
  ``expanding_normalize``, and clipping to q01/q99 does not move q01/q99, so S+1's BOUND is
  unchanged everywhere;
* nothing added or removed; widths 201/202/201 and rows 6,499 unchanged; no null count moves.

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
    PHASE332_RUNG_PREFIX,
    PHASE332_WINDOW_RUNG,
    PHASE332_WINSORIZATION_STEP,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG8 = rung_document_path(FINGERPRINT_DIR, PHASE332_WINDOW_RUNG, PHASE332_RUNG_PREFIX)
STEP8B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_WINSORIZATION_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step8b_before.json")
DIGEST_AFTER = Path("outputs/p332_step8b_after.json")

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded BEFORE the rebuild.
# ---------------------------------------------------------------------------

#: The five display-only passthroughs. Excluded from ``expanding_normalize``, so they move
#: in their first fitted season ALONE -- no prior-season bootstrap carries them forward.
PREDICTED_DISPLAY_ONLY: tuple[str, ...] = (
    "raw_humidity_pct",
    "raw_precip_prob",
    "raw_temp_f",
    "raw_weather_severity",
    "raw_wind_mph",
)

#: Each column's FIRST fitted season -- the one that used to fit its bound on itself.
PREDICTED_FIRST_SEASON: dict[str, str] = {
    column: seasons[0]
    for column, seasons in fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN.items()
}

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
        for column, seasons in fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN.items()
    }
    changed.update(overrides)
    return changed


def _verdict(changed: dict[str, list[str]]) -> dict:
    return attribute_rung(
        _report(changed),
        PHASE332_WINSORIZATION_STEP,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_is_an_extra_step_following_rung_eight(self) -> None:
        assert PHASE332_WINSORIZATION_STEP == "8b"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["8b"] == 8
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["8b"]
            == fg.PHASE332_WINSORIZATION_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["8b"]
            is fg.PHASE332_WINSORIZATION_STEP_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["8b"] is fg._attribute_p332_winsorization
        )

    def test_it_is_judged_against_rung_eight_and_sits_in_the_ladder_before_rung_nine(
        self, tmp_path
    ) -> None:
        assert phase332_baseline_document_path(tmp_path, "8b").name == "p332_rung8.json"
        assert "8b" in fg._ladder_predecessors(9, PHASE332_RUNG_PREFIX)

    def test_its_id_cannot_collide_with_rung_nine(self) -> None:
        assert PHASE332_WINSORIZATION_STEP not in (8, 9, "8", "9")
        assert rung_document_path(FINGERPRINT_DIR, "8b", PHASE332_RUNG_PREFIX).name == (
            "p332_rung8b.json"
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_WINSORIZATION_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"
        assert signature["columns_added"] == "empty"
        assert signature["columns_removed"] == "empty"


class TestThePrediction:
    def test_the_declared_columns_are_non_vacuous_and_exact(self) -> None:
        declared = fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN
        assert len(declared) == 97
        assert tuple(sorted(declared)) == fg.PHASE332_WINSORIZATION_STEP_COLUMNS
        assert all(seasons for seasons in declared.values())

    def test_every_column_moves_in_its_first_fitted_season(self) -> None:
        """S is always the first declared season, and the set of S values is small."""
        assert sorted(set(PREDICTED_FIRST_SEASON.values())) == [
            "2002",
            "2006",
            "2010",
            "2013",
        ]

    def test_a_normalized_column_also_moves_in_the_season_after_its_first(self) -> None:
        for column, seasons in fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN.items():
            if column in PREDICTED_DISPLAY_ONLY:
                continue
            assert len(seasons) == 2, column
            assert int(seasons[1]) == int(seasons[0]) + 1, column

    def test_the_display_only_columns_move_in_one_season_alone(self) -> None:
        for column in PREDICTED_DISPLAY_ONLY:
            seasons = fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN[column]
            assert seasons == ("2002",), column

    def test_no_coverage_flag_is_declared(self) -> None:
        """A coverage flag is a discrete indicator and is exempt from winsorization."""
        assert not [
            c for c in fg.PHASE332_WINSORIZATION_STEP_COLUMNS if c.endswith("_coverage")
        ]


class TestTheJudge:
    def test_the_declared_moves_are_attributed(self) -> None:
        verdict = _verdict(_all_declared())
        assert verdict["ok"], verdict["failures"]

    def test_a_move_outside_a_columns_declared_seasons_is_unattributed(self) -> None:
        assert not _verdict(_all_declared(home_elo=["2002", "2003", "2004"]))["ok"]

    def test_a_column_the_step_never_clips_is_unattributed(self) -> None:
        for column in ("home_snap_coverage", "divisional_game", "target_wp"):
            assert not _verdict(_all_declared(**{column: ["2002"]}))["ok"], column

    def test_a_declared_column_that_did_not_move_fails(self) -> None:
        changed = _all_declared()
        del changed["home_elo"]
        assert not _verdict(changed)["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG8.is_file() and STEP8B.is_file()),
    reason="the p332_ rung-8 or step-8b fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(RUNG8.read_text(encoding="utf-8")),
            json.loads(STEP8B.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_WINSORIZATION_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert verdict["non_clock_moves"] == sorted(state.P332_19_STEP8B_MOVED_COLUMNS)
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
        per_matrix = {
            matrix: {
                column: tuple(seasons)
                for column, seasons in report[matrix]["columns_changed"].items()
                if not fg._is_build_clock(column)
            }
            for matrix in fg.GOLD_MATRICES
        }
        declared = fg.PHASE332_WINSORIZATION_STEP_SEASONS_BY_COLUMN
        for matrix, moved in per_matrix.items():
            assert set(moved) == set(declared), matrix
            for column, seasons in moved.items():
                assert seasons == declared[column], (matrix, column)

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
            fg.PHASE332_WINSORIZATION_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_19_STEP8B_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["8b"]
        assert step["cause"] == fg.PHASE332_WINSORIZATION_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_winsorization"
        assert step["baseline_document"] == RUNG8.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]
        assert step["moved_columns"] == sorted(state.P332_19_STEP8B_MOVED_COLUMNS)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-8b digest bracket documents are gitignored runtime state",
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
