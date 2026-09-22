"""Extra step 8e of the Phase-33.2 gold ladder: the never-populated defensive copies go.

Owner ruling 2026-09-22 (deferred-items.md, "Eight gold team-form columns are a constant
0.0 in EVERY season", option (a) -- "REMOVE the eight columns from gold"), run FOURTH in
Plan 33.2-19's dispatch, after steps 8b, 8c and 8d and before rung 9; RENUMBERED from 8d
when the owner's blank-cell ruling took that slot ahead of it. ONE cause: the gold
team-form layout stops copying the DEFENSIVE column of every metric
``features.team_form`` declares offence-only.

THE PREDICTION, declared before the rebuild in commit ``b0b1a43`` and DERIVED rather than
previewed -- the four metrics are offence-only in the SOURCE, so their defensive copies
carry nothing for any other column to depend on:

* exactly the eight columns are REMOVED, in all three matrices, and nothing is added;
* every width falls by exactly 8 (201/202/201 -> 193/194/193) and rows stay 6,499;
* NO surviving column moves at all -- the changed set must be EMPTY. A single moved value
  is a finding that halts the step, because removing a column that nothing reads cannot
  change another column's value.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import scripts.fingerprint_gold as fg
import tests.phase33_state as state
from features.team_form import (
    OFFENSE_ONLY_METRICS,
    offense_only_gold_columns,
)
from scripts.data_qa import GOLD_FEATURE_MATRICES
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_OFFENSE_ONLY_STEP,
    PHASE332_RUNG_PREFIX,
    PHASE332_UNSCORABLE_BLANK_STEP,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
GOLD_DIR = Path("data/gold")
STEP8D = rung_document_path(
    FINGERPRINT_DIR, PHASE332_UNSCORABLE_BLANK_STEP, PHASE332_RUNG_PREFIX
)
STEP8E = rung_document_path(
    FINGERPRINT_DIR, PHASE332_OFFENSE_ONLY_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step8e_before.json")
DIGEST_AFTER = Path("outputs/p332_step8e_after.json")

WIDTHS_BEFORE: tuple[int, int, int] = (201, 202, 201)
PREDICTED_WIDTHS: tuple[int, int, int] = (193, 194, 193)
PREDICTED_ROWS: int = 6499

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)


def _report(
    removed: list[str],
    changed: dict[str, list[str]] | None = None,
    added: list[str] | None = None,
    width_after: int | None = None,
) -> dict:
    changed = changed or {}
    matrix = {
        "width_before": 201,
        "width_after": 201 - len(removed) if width_after is None else width_after,
        "rows_before": PREDICTED_ROWS,
        "rows_after": PREDICTED_ROWS,
        "rows_per_season_before": {},
        "rows_per_season_after": {},
        "columns_added": added or [],
        "columns_removed": removed,
        "columns_changed": changed,
        "column_details": {},
    }
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


def _verdict(**kwargs) -> dict:
    kwargs.setdefault("removed", list(fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS))
    return attribute_rung(
        _report(**kwargs),
        PHASE332_OFFENSE_ONLY_STEP,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_follows_rung_eight_and_is_judged_against_step_8d(
        self, tmp_path
    ) -> None:
        assert PHASE332_OFFENSE_ONLY_STEP == "8e"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["8e"] == 8
        assert (
            phase332_baseline_document_path(tmp_path, "8e").name == "p332_rung8d.json"
        )
        assert fg._ladder_predecessors("8e", PHASE332_RUNG_PREFIX)[-1] == "8d"

    def test_rung_nine_is_judged_against_it(self) -> None:
        """The order the orchestrator set: 8e (-8) runs BEFORE rung 9, so rung 9 is
        still the LAST width-moving rung and still writes the FINAL pin."""
        assert fg._ladder_predecessors(9, PHASE332_RUNG_PREFIX)[-1] == "8e"

    def test_it_is_dispatched(self) -> None:
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["8e"]
            == fg.PHASE332_OFFENSE_ONLY_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["8e"]
            is fg.PHASE332_OFFENSE_ONLY_STEP_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["8e"] is fg._attribute_p332_offense_only
        )

    def test_its_id_cannot_collide_with_rung_nine(self) -> None:
        assert PHASE332_OFFENSE_ONLY_STEP not in (8, 9, "8", "9")
        assert rung_document_path(FINGERPRINT_DIR, "8e", PHASE332_RUNG_PREFIX).name == (
            "p332_rung8e.json"
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_OFFENSE_ONLY_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["columns_added"] == "empty"
        assert signature["predicted_changed"] == ()


class TestThePrediction:
    def test_the_declared_set_is_the_registrys_own(self) -> None:
        """The ladder's declaration is DERIVED from ``features.team_form``, not retyped.

        If the two ever disagree the step is declaring a removal the build does not
        perform, or performing one it did not declare.
        """
        assert tuple(sorted(fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS)) == tuple(
            sorted(offense_only_gold_columns())
        )
        assert len(fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS) == 8
        assert len(OFFENSE_ONLY_METRICS) == 4

    def test_the_removed_set_is_defensive_copies_only(self) -> None:
        for column in fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS:
            assert "_def_rolling_" in column, column
            assert "_off_" not in column, column


class TestTheJudge:
    def test_the_declared_removal_is_attributed(self) -> None:
        verdict = _verdict()
        assert verdict["ok"], verdict["failures"]

    def test_removing_one_column_too_few_blocks(self) -> None:
        short = list(fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS)[:-1]
        verdict = _verdict(removed=short)
        assert not verdict["ok"]
        assert verdict["blocking"]

    def test_removing_an_undeclared_column_blocks(self) -> None:
        wide = [*fg.PHASE332_OFFENSE_ONLY_STEP_REMOVED_COLUMNS, "home_elo"]
        assert not _verdict(removed=wide)["ok"]

    def test_adding_a_column_blocks(self) -> None:
        assert not _verdict(added=["home_something_new"])["ok"]

    def test_a_surviving_value_move_is_a_finding(self) -> None:
        """The whole point of the negative half: this step changes no value."""
        verdict = _verdict(changed={"home_elo": ["2019"]})
        assert not verdict["ok"]
        assert "home_elo" in verdict["matrices"]["features_wp"]["unattributed"]

    def test_a_width_that_did_not_fall_by_eight_blocks(self) -> None:
        assert not _verdict(width_after=201)["ok"]

    def test_removing_nothing_fails(self) -> None:
        assert not _verdict(removed=[])["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP8D.is_file() and STEP8E.is_file()),
    reason="the p332_ step-8d or step-8e fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(STEP8D.read_text(encoding="utf-8")),
            json.loads(STEP8E.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_OFFENSE_ONLY_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert verdict["non_clock_moves"] == []
        assert verdict["build_clock_moves"] == ["feature_timestamp"]

    def test_exactly_the_eight_left_and_nothing_arrived(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        expected = set(state.P332_19_STEP8E_REMOVED_COLUMNS)
        for matrix in fg.GOLD_MATRICES:
            assert set(report[matrix]["columns_removed"]) == expected, matrix
            assert report[matrix]["columns_added"] == [], matrix

    def test_each_width_fell_by_exactly_eight_and_rows_held(self) -> None:
        before, after = self._documents()
        for matrix in fg.GOLD_MATRICES:
            assert after[matrix]["width"] == before[matrix]["width"] - 8, matrix
            assert before[matrix]["rows"] == after[matrix]["rows"] == PREDICTED_ROWS
        assert tuple(before[m]["width"] for m in fg.GOLD_MATRICES) == WIDTHS_BEFORE
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == PREDICTED_WIDTHS

    def test_no_surviving_column_moved(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        for matrix in fg.GOLD_MATRICES:
            moved = [
                column
                for column in report[matrix]["columns_changed"]
                if not fg._is_build_clock(column)
            ]
            assert moved == [], (matrix, moved)

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_OFFENSE_ONLY_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_19_STEP8E_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["8e"]
        assert step["cause"] == fg.PHASE332_OFFENSE_ONLY_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_offense_only"
        assert step["baseline_document"] == STEP8D.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["moved_columns"] == []
        assert step["added_columns"] == []
        assert sorted(step["removed_columns"]) == sorted(
            state.P332_19_STEP8E_REMOVED_COLUMNS
        )


@pytest.mark.skipif(
    not all((GOLD_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason="the gold matrices are absent on this checkout (data/ is gitignored)",
)
class TestTheLiveGold:
    def test_none_of_the_eight_survives_in_any_matrix(self) -> None:
        for matrix in fg.GOLD_MATRICES:
            columns = set(pd.read_parquet(GOLD_DIR / f"{matrix}.parquet").columns)
            assert not columns & set(offense_only_gold_columns()), matrix

    def test_the_offensive_copies_all_survive(self) -> None:
        """No over-reach: the four metrics are offence-only, not absent."""
        for matrix in fg.GOLD_MATRICES:
            columns = set(pd.read_parquet(GOLD_DIR / f"{matrix}.parquet").columns)
            for name in offense_only_gold_columns():
                assert name.replace("_def_", "_off_") in columns, (matrix, name)

    def test_the_width_pin_still_describes_the_gold_on_disk(self) -> None:
        """The pin equals the live matrices -- whatever LATER rungs have moved it to.

        THIS NODE DOES NOT PIN 193/194/193. It did, and that was a premise error of
        exactly the kind this phase keeps correcting: a step's test must not assert a
        LIVE artifact that a later ladder step legitimately moves, or it goes red for a
        reason that has nothing to do with the step. Rung 9 ran after this one and took
        the pin to 188/188/187. This step's OWN width claim -- 201/202/201 -> 193/194/193
        -- is asserted against its two fingerprint DOCUMENTS in
        ``test_each_width_fell_by_exactly_eight_and_rows_held``, which no later rung can
        move, and ``tests/unit/test_data_qa_gold_width.py`` owns the live pin.
        """
        measured = {
            matrix: pd.read_parquet(GOLD_DIR / f"{matrix}.parquet").shape[1]
            for matrix in GOLD_FEATURE_MATRICES
        }
        assert measured == dict(GOLD_FEATURE_MATRICES)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-8e digest bracket documents are gitignored runtime state",
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
