"""Rung 9 of the Phase-33.2 gold ladder: no betting line is a model input for any target.

The LAST rung of the `p332_` ladder (Plan 33.2-19, D33.2-03), judged against extra step
8e. ONE cause: the five market-line columns leave all three gold matrices through the ONE
registry group ``market`` and the ONE drop mechanism every dropped group uses, and the
four line-derived TARGET columns go with their parents as arithmetic children of exactly
those columns.

THE PREDICTION, declared before the rebuild in commit ``75e39ea`` (the constants below
equal ``scripts.fingerprint_gold``'s ``PHASE332_MARKET_RUNG_*``):

* every matrix removes ``snapshot_spread``, ``snapshot_total``,
  ``snapshot_ml_prob_home_fair``, ``spread_movement`` and ``total_movement``;
* ``features_ats`` also removes ``target_ats`` and ``features_ou`` also removes
  ``target_ou``. ``home_covered_spread`` and ``game_went_over`` never reached a matrix, so
  the width delta is -5 / -6 / -6 and NOT the uniform -5 the plan's text assumed:
  193/194/193 -> 188/188/187;
* nothing is added and rows stay 6,499 -- the ats and ou row filters moved from the
  line-derived target to ``home_margin`` and ``total_points``, their trainers' own targets,
  both non-null on every game in gold;
* NO surviving column moves at all. Every market value has been a constant 0.0 since rung
  5, so a moved value would mean a downstream step reads the market columns, which is a
  finding that halts the run.

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
from backtest.signal_lift import group_columns
from scripts.data_qa import GOLD_FEATURE_MATRICES
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_MARKET_RUNG,
    PHASE332_OFFENSE_ONLY_STEP,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
GOLD_DIR = Path("data/gold")
STEP8E = rung_document_path(
    FINGERPRINT_DIR, PHASE332_OFFENSE_ONLY_STEP, PHASE332_RUNG_PREFIX
)
RUNG9 = rung_document_path(FINGERPRINT_DIR, PHASE332_MARKET_RUNG, PHASE332_RUNG_PREFIX)
DIGEST_BEFORE = Path("outputs/p332_rung9_before.json")
DIGEST_AFTER = Path("outputs/p332_rung9_after.json")

WIDTHS_BEFORE: tuple[int, int, int] = (193, 194, 193)
PREDICTED_WIDTHS: tuple[int, int, int] = (188, 188, 187)
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
    removed: dict[str, list[str]] | None = None,
    changed: dict[str, list[str]] | None = None,
    added: list[str] | None = None,
    width_after: dict[str, int] | None = None,
) -> dict:
    removed = (
        removed
        if removed is not None
        else {
            matrix: list(columns)
            for matrix, columns in fg.PHASE332_MARKET_RUNG_REMOVED_BY_MATRIX.items()
        }
    )
    widths_before = dict(zip(fg.GOLD_MATRICES, WIDTHS_BEFORE, strict=True))
    report = {}
    for matrix in fg.GOLD_MATRICES:
        gone = removed.get(matrix, [])
        report[matrix] = {
            "width_before": widths_before[matrix],
            "width_after": (
                widths_before[matrix] - len(gone)
                if width_after is None
                else width_after[matrix]
            ),
            "rows_before": PREDICTED_ROWS,
            "rows_after": PREDICTED_ROWS,
            "rows_per_season_before": {},
            "rows_per_season_after": {},
            "columns_added": added or [],
            "columns_removed": gone,
            "columns_changed": changed or {},
            "column_details": {},
        }
    return report


def _verdict(**kwargs) -> dict:
    return attribute_rung(
        _report(**kwargs), PHASE332_MARKET_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
    )


class TestTheRungIsRegisteredAndDispatched:
    def test_it_is_the_ladders_last_rung_and_follows_step_8e(self, tmp_path) -> None:
        assert PHASE332_MARKET_RUNG == 9
        assert sorted(fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]) == list(
            range(1, 10)
        )
        assert phase332_baseline_document_path(tmp_path, 9).name == "p332_rung8e.json"
        assert fg._ladder_predecessors(9, PHASE332_RUNG_PREFIX)[-1] == "8e"

    def test_it_is_dispatched_not_merely_registered(self) -> None:
        """A registered-but-undispatched p332_ rung falls through to Phase 30's path."""
        assert PHASE332_MARKET_RUNG in fg.PHASE332_RUNG_SIGNATURES
        assert PHASE332_MARKET_RUNG in fg.PHASE332_RUNG_ATTRIBUTORS
        assert (
            fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_MARKET_RUNG]
            is fg._attribute_p332_market
        )
        assert 9 in set(fg.PHASE332_RUNG_SIGNATURES) & set(fg.PHASE332_RUNG_ATTRIBUTORS)

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_MARKET_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["columns_added"] == "empty"
        assert signature["predicted_changed"] == ()


class TestThePrediction:
    def test_the_declared_market_set_is_the_registrys_own(self) -> None:
        """The rung's five names and the predicate's matched set are the SAME answer."""
        frame = pd.DataFrame(
            columns=pd.Index(
                [
                    *fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS,
                    "total_points",
                    "rolling_total_epa",
                ]
            )
        )
        assert set(group_columns(frame, "market")) == set(
            fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS
        )
        assert len(fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS) == 5

    def test_each_matrix_loses_the_five_and_at_most_one_target(self) -> None:
        by_matrix = fg.PHASE332_MARKET_RUNG_REMOVED_BY_MATRIX
        market = set(fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS)
        assert set(by_matrix) == set(fg.GOLD_MATRICES)
        for matrix, columns in by_matrix.items():
            extra = set(columns) - market
            assert market <= set(columns), matrix
            assert extra <= set(fg.PHASE332_MARKET_RUNG_TARGET_COLUMNS), matrix
            assert len(extra) <= 1, matrix
        assert by_matrix["features_wp"] == fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS
        assert "target_ats" in by_matrix["features_ats"]
        assert "target_ou" in by_matrix["features_ou"]


class TestTheJudge:
    def test_the_declared_removal_is_attributed(self) -> None:
        verdict = _verdict()
        assert verdict["ok"], verdict["failures"]

    def test_a_matrix_that_kept_a_market_column_blocks(self) -> None:
        removed = {
            matrix: [c for c in columns if c != "snapshot_total"]
            for matrix, columns in fg.PHASE332_MARKET_RUNG_REMOVED_BY_MATRIX.items()
        }
        verdict = _verdict(removed=removed)
        assert not verdict["ok"]
        assert verdict["blocking"]

    def test_removing_an_undeclared_column_blocks(self) -> None:
        removed = {
            matrix: [*columns, "home_elo"]
            for matrix, columns in fg.PHASE332_MARKET_RUNG_REMOVED_BY_MATRIX.items()
        }
        assert not _verdict(removed=removed)["ok"]

    def test_removing_both_targets_from_one_matrix_blocks(self) -> None:
        removed = {
            matrix: [*fg.PHASE332_MARKET_RUNG_MARKET_COLUMNS, "target_ats", "target_ou"]
            for matrix in fg.GOLD_MATRICES
        }
        assert not _verdict(removed=removed)["ok"]

    def test_adding_a_column_blocks(self) -> None:
        assert not _verdict(added=["home_something_new"])["ok"]

    def test_a_surviving_value_move_is_a_finding(self) -> None:
        verdict = _verdict(changed={"home_elo": ["2019"]})
        assert not verdict["ok"]
        assert "home_elo" in verdict["matrices"]["features_wp"]["unattributed"]

    def test_a_width_that_did_not_fall_by_what_was_removed_blocks(self) -> None:
        widths = dict(zip(fg.GOLD_MATRICES, WIDTHS_BEFORE, strict=True))
        assert not _verdict(width_after=widths)["ok"]

    def test_removing_nothing_fails(self) -> None:
        empty: dict[str, list[str]] = {matrix: [] for matrix in fg.GOLD_MATRICES}
        assert not _verdict(removed=empty)["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP8E.is_file() and RUNG9.is_file()),
    reason="the p332_ step-8e or rung-9 fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveRung:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(STEP8E.read_text(encoding="utf-8")),
            json.loads(RUNG9.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_MARKET_RUNG,
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

    def test_the_ladder_is_complete_and_recoverable_end_to_end(self) -> None:
        """All nine rungs and every extra step, proved as a CHAIN rather than rung by rung."""
        fg.require_rung_ladder(FINGERPRINT_DIR, 9, PHASE332_RUNG_PREFIX)
        fg.assert_ladder_is_recoverable(FINGERPRINT_DIR, 9, PHASE332_RUNG_PREFIX)

    def test_each_matrix_removed_exactly_what_was_declared_for_it(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        for matrix in fg.GOLD_MATRICES:
            assert set(report[matrix]["columns_removed"]) == set(
                state.P332_19_RUNG9_REMOVED_BY_MATRIX[matrix]
            ), matrix
            assert report[matrix]["columns_added"] == [], matrix

    def test_the_widths_fell_as_declared_and_rows_held(self) -> None:
        before, after = self._documents()
        for matrix in fg.GOLD_MATRICES:
            declared = fg.PHASE332_MARKET_RUNG_REMOVED_BY_MATRIX[matrix]
            assert after[matrix]["width"] == before[matrix]["width"] - len(declared), (
                matrix
            )
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
            fg.PHASE332_MARKET_RUNG_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_19_RUNG9_CAUSE_DIGEST

    def test_the_committed_diff_records_the_rung(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"]["9"]
        assert rung["cause"] == fg.PHASE332_MARKET_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_market"
        assert rung["baseline_document"] == STEP8E.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["moved_columns"] == []
        assert rung["added_columns"] == []
        for matrix in fg.GOLD_MATRICES:
            assert sorted(rung["removed_by_matrix"][matrix]) == sorted(
                state.P332_19_RUNG9_REMOVED_BY_MATRIX[matrix]
            )


@pytest.mark.skipif(
    not all((GOLD_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason="the gold matrices are absent on this checkout (data/ is gitignored)",
)
class TestTheLiveGold:
    @staticmethod
    def _frame(matrix: str) -> pd.DataFrame:
        return pd.read_parquet(GOLD_DIR / f"{matrix}.parquet")

    def test_no_betting_line_survives_in_any_matrix(self) -> None:
        for matrix in fg.GOLD_MATRICES:
            assert group_columns(self._frame(matrix), "market") == [], matrix

    def test_the_line_derived_targets_are_gone(self) -> None:
        for matrix in fg.GOLD_MATRICES:
            columns = set(self._frame(matrix).columns)
            assert not columns & {
                "target_ats",
                "target_ou",
                "home_covered_spread",
                "game_went_over",
            }, matrix

    def test_each_matrix_still_carries_its_trainers_own_target(self) -> None:
        """No over-reach: the real targets are derived from the SCORES, not from a line."""
        assert "home_win" in set(self._frame("features_wp").columns)
        assert "home_margin" in set(self._frame("features_ats").columns)
        assert "total_points" in set(self._frame("features_ou").columns)

    def test_every_matrix_still_carries_every_game(self) -> None:
        """No matrix drops a game the others keep -- whatever the live row count is.

        THIS NODE DOES NOT PIN 6,499. It did, and that was the premise error
        ``74afaaf`` corrected for step 8e one rung earlier: a rung's test must not
        assert a LIVE artifact that later work legitimately moves, or it goes red for
        a reason that has nothing to do with the rung. Plan 33.2-20's clean
        production build added the 17 played 2026 games (6,499 -> 6,516), which this
        node predates; nothing about rung 9 changed.

        Rung 9's OWN row claim -- 6,499 before and after, because the ats and ou row
        filters moved from the removed line-derived targets to ``home_margin`` and
        ``total_points`` -- is asserted against this rung's two fingerprint DOCUMENTS
        in ``test_the_widths_fell_as_declared_and_rows_held``, which no later build
        can move. What stays true of LIVE gold for as long as those filters hold is
        the EQUALITY: every matrix carries the same games, so a filter that started
        dropping rows on one target would still be caught here.
        """
        measured = {matrix: len(self._frame(matrix)) for matrix in fg.GOLD_MATRICES}
        assert len(set(measured.values())) == 1, measured
        assert min(measured.values()) > 0, measured

    def test_the_width_pin_agrees_with_the_built_gold(self) -> None:
        """The pin equals the live matrices -- whatever LATER work has moved it to.

        The ``PREDICTED_WIDTHS`` clause was removed here for the same reason
        ``74afaaf`` removed it from step 8e's live node: rung 9's own width claim
        (193/194/193 -> 188/188/187) is asserted against its fingerprint DOCUMENTS
        in ``test_the_widths_fell_as_declared_and_rows_held``, and the live pin is
        owned by ``tests/unit/test_data_qa_gold_width.py``. Rung 9 happens to be the
        last rung, so the two agree today; the re-fit's gold step (Plan 33.2-23) may
        move the width, and this node must not go red when it does.
        """
        measured = {
            matrix: self._frame(matrix).shape[1] for matrix in GOLD_FEATURE_MATRICES
        }
        assert measured == dict(GOLD_FEATURE_MATRICES)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the rung-9 digest bracket documents are gitignored runtime state",
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
