"""Rung 1 of the Phase-33.2 gold ladder: declared, rebuilt, and attributed by its OWN judge.

Plan 33.2-08 Task 4 (D33.2-20). The `p332_` prefix is registered AND dispatched: rung 1
is judged by ``_attribute_p332_odds`` against its own declared signature. A registered
prefix without a dispatch branch would fall through to Phase 30's rung-1 semantics
(``_attribute_rung1``, indicator-only), which would read the continuous market columns
this rung moves as unattributed on a CORRECT rebuild -- the synthetic control below proves
that trap is real rather than assumed.

RETAKEN BASELINE (owner ruling 2026-09-21, "retake a stale baseline, never widen a rung's
cause"). The first rung-0 document was gold as it stood on disk, last built before Plan
33.2-05 re-sorted ``elo_game_snapshots``; 2002 week-1 Elo ties break by row order, so a
rebuild from today's inputs moved 2002-2003 Elo rank/percentile whatever rung 1 did. Rung 1
is therefore judged against ``p332_rung0_retaken.json`` -- gold rebuilt in a scratch data
root from today's inputs with only the 12 corrected odds values put back -- and the
original-to-retaken difference is asserted to be exactly that measured carry-in.
``target_ats`` (final margin minus spread) is attributed as a run-time-disclosed child of
``snapshot_spread``.

The live half reads the gitignored fingerprint documents under ``outputs/fingerprints``
and skips, naming the absent evidence, on a checkout that has not run the ladder.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import scripts.fingerprint_gold as fg
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_ODDS_MARKET_SOURCES,
    PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN,
    PHASE332_ODDS_RUNG,
    PHASE332_RETAKEN_BASELINES,
    PHASE332_RUNG_PREFIX,
    MissingPredecessorFingerprintError,
    assert_ladder_is_recoverable,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    require_rung_ladder,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG0 = rung_document_path(FINGERPRINT_DIR, 0, PHASE332_RUNG_PREFIX)
RUNG1 = rung_document_path(FINGERPRINT_DIR, PHASE332_ODDS_RUNG, PHASE332_RUNG_PREFIX)
RUNG0_RETAKEN = FINGERPRINT_DIR / PHASE332_RETAKEN_BASELINES[PHASE332_ODDS_RUNG]
MARKET_COLUMNS = set(PHASE332_ODDS_MARKET_SOURCES)
EXPLAINABLE_COLUMNS = MARKET_COLUMNS | set(PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN)

# MEASURED 2026-09-21 (original p332_rung0.json -> p332_rung0_retaken.json): the pre-ladder
# carry-in of Plan 33.2-05's elo_game_snapshots re-sort. Exactly these columns, exactly
# these seasons; anything else in the carry-in would be a cause nobody has named.
CARRY_IN_COLUMNS = {
    "home_elo_rank",
    "away_elo_rank",
    "home_elo_percentile",
    "away_elo_percentile",
}
CARRY_IN_SEASONS = {"2002", "2003"}

needs_ladder = pytest.mark.skipif(
    not (RUNG0.is_file() and RUNG0_RETAKEN.is_file() and RUNG1.is_file()),
    reason=(
        "the p332_ rung fingerprint document is not present under outputs/fingerprints "
        "-- outputs/ is gitignored runtime state"
    ),
)


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


LATE_SEASONS = [str(s) for s in range(2021, 2026)]


class TestTheRungIsDispatchedNotMerelyRegistered:
    def test_rung_one_is_judged_by_its_own_attributor(self, monkeypatch) -> None:
        calls: list[str] = []
        original = fg.PHASE332_RUNG_ATTRIBUTORS[PHASE332_ODDS_RUNG]

        def spy(*args, **kwargs):
            calls.append("p332")
            return original(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "p332_ rung 1 fell through to Phase 30's _attribute_rung1"
            )

        monkeypatch.setitem(fg.PHASE332_RUNG_ATTRIBUTORS, PHASE332_ODDS_RUNG, spy)
        monkeypatch.setattr(fg, "_attribute_rung1", forbidden)
        verdict = attribute_rung(
            _report({"snapshot_spread": LATE_SEASONS}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]
        assert verdict["cause"] == fg.PHASE332_ODDS_RUNG_CAUSE

    def test_the_phase_30_judge_would_have_misread_a_correct_rebuild(self) -> None:
        """The trap the branch exists for, measured: same diff, no prefix, not attributed."""
        verdict = attribute_rung(_report({"snapshot_spread": LATE_SEASONS}), 1)
        assert not verdict["ok"]
        assert "snapshot_spread" in verdict["matrices"]["features_wp"]["unattributed"]

    def test_an_unregistered_rung_is_refused_by_name_at_both_tables(self) -> None:
        with pytest.raises(ValueError, match="PHASE332_RUNG_SIGNATURES"):
            fg._phase332_table_entry(
                fg.PHASE332_RUNG_SIGNATURES, 2, "PHASE332_RUNG_SIGNATURES"
            )
        with pytest.raises(ValueError, match="PHASE332_RUNG_ATTRIBUTORS"):
            fg._attribute_one_matrix(
                2,
                _matrix({}),
                {},
                {},
                lambda message: None,
                rung_prefix=PHASE332_RUNG_PREFIX,
            )

    def test_an_unknown_prefix_still_refuses(self) -> None:
        with pytest.raises(ValueError, match="Unknown rung prefix"):
            fg._rung_causes("p999_")

    def test_only_rung_one_is_declared(self) -> None:
        assert sorted(fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]) == [1]
        assert sorted(fg.PHASE332_RUNG_SIGNATURES) == [1]
        assert sorted(fg.PHASE332_RUNG_ATTRIBUTORS) == [1]


class TestTheRunTimeDisclosedChild:
    """target_ats is attributed only as the arithmetic child of an attributed snapshot_spread."""

    def test_it_is_attributed_with_its_parent_in_the_same_seasons(self) -> None:
        verdict = attribute_rung(
            _report({"snapshot_spread": LATE_SEASONS, "target_ats": LATE_SEASONS}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert "target_ats" in verdict["matrices"]["features_ats"]["attributed"]

    def test_it_is_unattributed_without_its_parent(self) -> None:
        verdict = attribute_rung(
            _report({"target_ats": LATE_SEASONS}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]
        assert verdict["matrices"]["features_ats"]["unattributed"] == ["target_ats"]

    def test_it_is_unattributed_in_a_season_its_parent_did_not_move(self) -> None:
        verdict = attribute_rung(
            _report(
                {"snapshot_spread": ["2021", "2022"], "target_ats": ["2021", "2023"]}
            ),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]
        assert "target_ats" in verdict["matrices"]["features_ats"]["unattributed"]

    def test_the_disclosure_is_kept_apart_from_the_declared_columns(self) -> None:
        assert "target_ats" not in PHASE332_ODDS_MARKET_SOURCES
        signature = fg._expected_signature(
            PHASE332_ODDS_RUNG, prefix=PHASE332_RUNG_PREFIX
        )
        assert signature["declared_before_the_rebuild"] is True
        assert "target_ats" in signature["disclosed_at_run_time"]


class TestTheRetakenBaselineIsRegisteredAndRequired:
    def test_rung_one_is_judged_against_the_retaken_document(self, tmp_path) -> None:
        (tmp_path / PHASE332_RETAKEN_BASELINES[PHASE332_ODDS_RUNG]).write_text("{}")
        assert phase332_baseline_document_path(tmp_path, PHASE332_ODDS_RUNG) == (
            tmp_path / "p332_rung0_retaken.json"
        )

    def test_a_missing_retaken_baseline_is_refused_not_replaced(self, tmp_path) -> None:
        rung_document_path(tmp_path, 0, PHASE332_RUNG_PREFIX).write_text("{}")
        with pytest.raises(MissingPredecessorFingerprintError, match="retaken"):
            phase332_baseline_document_path(tmp_path, PHASE332_ODDS_RUNG)

    def test_an_unregistered_rung_falls_back_to_its_predecessor(self, tmp_path) -> None:
        assert phase332_baseline_document_path(tmp_path, 2) == rung_document_path(
            tmp_path, 1, PHASE332_RUNG_PREFIX
        )


class TestTheAttributorRefusesWhatTheCauseCannotExplain:
    def test_a_non_market_column_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"home_elo_rating": LATE_SEASONS}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]
        assert verdict["matrices"]["features_wp"]["unattributed"] == ["home_elo_rating"]

    def test_a_market_column_moving_before_its_earliest_correction_is_unattributed(
        self,
    ) -> None:
        floor = fg.phase332_odds_corrected_seasons()["snapshot_spread"]
        verdict = attribute_rung(
            _report({"snapshot_spread": [str(floor - 1), str(floor)]}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_a_market_column_whose_source_was_only_confirmed_is_unattributed(
        self,
    ) -> None:
        # The only total in the record is CONFIRMED (28.5 on 2023_W18_NYJ@NE): it changed
        # nothing, so snapshot_total moving would be unexplained.
        assert "snapshot_total" not in fg.phase332_odds_corrected_seasons()
        verdict = attribute_rung(
            _report({"snapshot_total": LATE_SEASONS}),
            PHASE332_ODDS_RUNG,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]


@needs_ladder
class TestTheLiveRung:
    def test_the_ladder_is_intact_and_recoverable(self) -> None:
        assert [
            p.name
            for p in require_rung_ladder(FINGERPRINT_DIR, 2, PHASE332_RUNG_PREFIX)
        ] == [
            RUNG0.name,
            RUNG1.name,
        ]
        assert_ladder_is_recoverable(FINGERPRINT_DIR, 2, PHASE332_RUNG_PREFIX)
        assert (
            phase332_baseline_document_path(FINGERPRINT_DIR, PHASE332_ODDS_RUNG)
            == RUNG0_RETAKEN
        )

    def test_the_carry_in_is_exactly_the_measured_elo_tie_break(self) -> None:
        import json

        original = json.loads(RUNG0.read_text(encoding="utf-8"))
        retaken = json.loads(RUNG0_RETAKEN.read_text(encoding="utf-8"))
        report = compare_fingerprints(original, retaken)
        moved: dict[str, set[str]] = {}
        for matrix in fg.GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [] and detail["columns_removed"] == []
            assert detail["rows_before"] == detail["rows_after"]
            for column, seasons in detail["columns_changed"].items():
                if not fg._is_build_clock(column):
                    moved.setdefault(column, set()).update(seasons)
        assert set(moved) == CARRY_IN_COLUMNS
        assert set().union(*moved.values()) == CARRY_IN_SEASONS

    def test_every_moved_column_is_explained_by_the_declared_cause(self) -> None:
        import json

        before = json.loads(RUNG0_RETAKEN.read_text(encoding="utf-8"))
        after = json.loads(RUNG1.read_text(encoding="utf-8"))
        verdict = attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_ODDS_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert verdict["non_clock_moves"], "an empty rung must be recorded as not run"
        assert set(verdict["non_clock_moves"]) <= EXPLAINABLE_COLUMNS

    def test_the_committed_diff_names_the_rung_prefix_and_cause(self) -> None:
        assert DIFF_TOML.is_file()
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        assert diff["rung_prefix"] == PHASE332_RUNG_PREFIX
        rung = diff["rung"][str(PHASE332_ODDS_RUNG)]
        assert rung["cause"] == fg.PHASE332_ODDS_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_odds"
        assert rung["baseline_document"] == RUNG0_RETAKEN.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert set(rung["moved_columns"]) <= EXPLAINABLE_COLUMNS
        assert rung["disclosed_at_run_time_columns"] == ["target_ats"]

    def test_the_committed_diff_records_the_carry_in_outside_rung_one(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung0 = diff["rung"]["0"]
        assert rung0["document"] == RUNG0.name
        assert rung0["retaken"]["document"] == RUNG0_RETAKEN.name
        carry = rung0["carry_in"]
        assert "33.2-05" in carry["cause"]
        assert set(carry["moved_columns"]) == CARRY_IN_COLUMNS
        assert not set(carry["moved_columns"]) & set(
            diff["rung"][str(PHASE332_ODDS_RUNG)]["moved_columns"]
        )
