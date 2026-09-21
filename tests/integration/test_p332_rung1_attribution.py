"""Rung 1 of the Phase-33.2 gold ladder: declared, rebuilt, and attributed by its OWN judge.

Plan 33.2-08 Task 4 (D33.2-20). The `p332_` prefix is registered AND dispatched: rung 1
is judged by ``_attribute_p332_odds`` against its own declared signature. A registered
prefix without a dispatch branch would fall through to Phase 30's rung-1 semantics
(``_attribute_rung1``, indicator-only), which would read the continuous market columns
this rung moves as unattributed on a CORRECT rebuild -- the synthetic control below proves
that trap is real rather than assumed.

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
    PHASE332_ODDS_RUNG,
    PHASE332_RUNG_PREFIX,
    assert_ladder_is_recoverable,
    attribute_rung,
    compare_fingerprints,
    require_rung_ladder,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG0 = rung_document_path(FINGERPRINT_DIR, 0, PHASE332_RUNG_PREFIX)
RUNG1 = rung_document_path(FINGERPRINT_DIR, PHASE332_ODDS_RUNG, PHASE332_RUNG_PREFIX)
MARKET_COLUMNS = set(PHASE332_ODDS_MARKET_SOURCES)

needs_ladder = pytest.mark.skipif(
    not (RUNG0.is_file() and RUNG1.is_file()),
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

    def test_every_moved_column_is_explained_by_the_declared_cause(self) -> None:
        import json

        before = json.loads(RUNG0.read_text(encoding="utf-8"))
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
        assert set(verdict["non_clock_moves"]) <= MARKET_COLUMNS

    def test_the_committed_diff_names_the_rung_prefix_and_cause(self) -> None:
        assert DIFF_TOML.is_file()
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        assert diff["rung_prefix"] == PHASE332_RUNG_PREFIX
        rung = diff["rung"][str(PHASE332_ODDS_RUNG)]
        assert rung["cause"] == fg.PHASE332_ODDS_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_odds"
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert set(rung["moved_columns"]) <= MARKET_COLUMNS
