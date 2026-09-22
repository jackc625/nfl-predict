"""Rung 5 of the Phase-33.2 gold ladder: every builder selection cutoff on the per-game lock.

Plan 33.2-14 Task 3 (SPEC R5, D33.2-01, D33.2-20). ONE cause -- a builder admits a row only
when it was known at or before the target game's own day-before lock -- applied uniformly to
eight selections: injury reports, starting-QB identity, QB play-by-play, the letdown flag,
the rest-days window, the market-odds selection, the snap window and the team-form rolling
window. The day-before forecast fence is NOT among them (rung 4's cause, Plan 33.2-12).

THE PREDICTION IS PER BUILDER and was declared before the rebuild (``PREDICTED_BY_BUILDER``
below, equal to ``scripts.fingerprint_gold.PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER``):

* injury, qb -- EMPTY: their movement landed at extra step 4b as Plan 33.2-13's measured
  carry-in, and rung 5 is judged against ``p332_rung4b.json``;
* contextual, snaps -- EMPTY: the pre- and post-change builders were measured byte-identical
  over 2002-2025; a row here would have to be a named rescheduled game;
* team_form -- EMPTY: D33.2-01 measured 0 games whose week-keyed prior-game inputs include a
  result that ended after their day-before lock (the window is per team; a team plays once a
  week);
* market -- the five market columns plus their arithmetic children ``target_ats`` and
  ``target_ou``. Owner ruling 2026-09-22 ("Only real capture times"): a line counts only with
  a recorded capture time (``created_at``) at or before the lock, so no stored 2018-2025 line
  qualifies and every 2018-2025 market value becomes the honest unknown.

What is asserted: the rung is registered AND dispatched (no fifth prefix branch); the cause
names the team-form window and neither the forecast fence nor any day-of-week claim; the
per-builder prediction and its union; a market column in a pre-2018 season, a child without
its parent, and any column of an empty-subset builder are unattributed; and, live against the
gitignored ladder, the attribution is clean, every builder's measured subset sits inside its
prediction, and the committed diff and the state witness record it.

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
    PHASE332_CUTOFF_RUNG,
    PHASE332_RETRACTABLE_ROOF_STEP,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG5 = rung_document_path(FINGERPRINT_DIR, PHASE332_CUTOFF_RUNG, PHASE332_RUNG_PREFIX)
STEP4B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_RETRACTABLE_ROOF_STEP, PHASE332_RUNG_PREFIX
)

#: THE PREDICTION, PER BUILDER, declared before the rebuild (Plan 33.2-14 Task 3).
PREDICTED_BY_BUILDER: dict[str, tuple[str, ...]] = {
    "injury": (),
    "qb": (),
    "contextual": (),
    "snaps": (),
    "team_form": (),
    "market": (
        "snapshot_spread",
        "snapshot_total",
        "snapshot_ml_prob_home_fair",
        "spread_movement",
        "total_movement",
        "target_ats",
        "target_ou",
    ),
}
PREDICTED_UNION: frozenset[str] = frozenset(
    column for columns in PREDICTED_BY_BUILDER.values() for column in columns
)


def _report(changed: dict[str, list[str]], width: int = 193) -> dict:
    matrix = {
        "width_before": width,
        "width_after": width,
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
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


def _verdict(changed: dict[str, list[str]]) -> dict:
    return attribute_rung(
        _report(changed), PHASE332_CUTOFF_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
    )


class TestTheRungIsRegisteredAndDispatched:
    def test_rung_five_is_in_every_table(self) -> None:
        assert PHASE332_CUTOFF_RUNG == 5
        assert (
            fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][5]
            == fg.PHASE332_CUTOFF_RUNG_CAUSE
        )
        assert (
            fg.PHASE332_RUNG_SIGNATURES[5] is fg.PHASE332_CUTOFF_RUNG_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_RUNG_ATTRIBUTORS[5] is fg._attribute_p332_cutoff

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_the_signature_was_declared_before_the_rebuild(self) -> None:
        signature = fg._expected_signature(5, prefix=PHASE332_RUNG_PREFIX)
        assert signature == fg.PHASE332_CUTOFF_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"

    def test_rung_five_is_judged_against_step_4b(self, tmp_path) -> None:
        assert fg._ladder_predecessors(5, PHASE332_RUNG_PREFIX)[-1] == "4b"
        assert phase332_baseline_document_path(tmp_path, 5).name == "p332_rung4b.json"
        assert 5 not in fg.PHASE332_RETAKEN_BASELINES


class TestTheCauseSaysExactlyWhatChanged:
    def test_it_names_the_team_form_window_and_the_eight_selections(self) -> None:
        cause = fg.PHASE332_CUTOFF_RUNG_CAUSE.lower()
        assert "team-form rolling window" in cause
        for selection in (
            "injury reports",
            "starting-qb identity",
            "qb play-by-play",
            "letdown",
            "rest-days",
            "market-odds",
            "snap window",
        ):
            assert selection in cause, selection
        assert "one rule change" in cause

    def test_it_names_neither_the_forecast_fence_nor_a_day_of_week_claim(self) -> None:
        cause = fg.PHASE332_CUTOFF_RUNG_CAUSE.lower()
        assert "weather" not in cause
        assert "thursday" not in cause


class TestThePredictionIsPerBuilder:
    def test_the_declared_dict_is_the_judges_dict(self) -> None:
        assert PREDICTED_BY_BUILDER == fg.PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER

    def test_the_expected_empty_subsets_are_recorded_not_omitted(self) -> None:
        for builder in ("injury", "qb", "contextual", "snaps", "team_form"):
            assert builder in PREDICTED_BY_BUILDER
            assert PREDICTED_BY_BUILDER[builder] == ()

    def test_the_union_is_the_market_family(self) -> None:
        assert (
            set(fg.PHASE332_CUTOFF_RUNG_MARKET_COLUMNS)
            | set(fg.PHASE332_CUTOFF_RUNG_MARKET_CHILDREN)
            == PREDICTED_UNION
        )

    def test_every_predicted_column_belongs_to_its_builder(self) -> None:
        families = fg.phase332_cutoff_builder_columns()
        for builder, columns in PREDICTED_BY_BUILDER.items():
            assert set(columns) <= families[builder], builder


class TestTheJudge:
    def test_the_market_family_moving_from_2018_is_attributed(self) -> None:
        seasons = [str(s) for s in range(2018, 2026)]
        verdict = _verdict(
            {
                "snapshot_spread": seasons,
                "snapshot_total": seasons,
                "snapshot_ml_prob_home_fair": seasons,
                "target_ats": seasons,
                "target_ou": seasons,
            }
        )
        assert verdict["ok"], verdict["failures"]

    def test_a_market_column_before_2018_is_unattributed(self) -> None:
        assert not _verdict({"snapshot_spread": ["2017", "2018"]})["ok"]

    def test_a_child_without_its_parent_is_unattributed(self) -> None:
        assert not _verdict({"target_ats": ["2019"]})["ok"]

    def test_a_child_outside_its_parents_seasons_is_unattributed(self) -> None:
        assert not _verdict(
            {"snapshot_total": ["2019"], "target_ou": ["2019", "2020"]}
        )["ok"]

    @pytest.mark.parametrize(
        "column",
        [
            "home_rest_days",  # contextual: EMPTY subset
            "home_letdown_spot",  # contextual: EMPTY subset
            "home_snap_concentration",  # snaps: EMPTY subset
            "home_off_rolling_epa_per_play",  # team_form: EMPTY subset
            "home_qb_out_flag",  # injury: EMPTY subset (landed at step 4b)
            "home_qb_adjustment",  # qb: EMPTY subset (landed at step 4b)
            "temp_f",  # the forecast: rung 4's cause
        ],
    )
    def test_any_other_column_is_unattributed_and_names_its_builder(
        self, column: str
    ) -> None:
        verdict = _verdict({column: ["2019"]})
        assert not verdict["ok"]
        assert any("builder:" in failure for failure in verdict["failures"])

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG5.is_file() and STEP4B.is_file()),
    reason=(
        "the p332_ step-4b or rung-5 fingerprint documents are absent from "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveRung:
    @staticmethod
    def _verdict() -> dict:
        before = json.loads(STEP4B.read_text(encoding="utf-8"))
        after = json.loads(RUNG5.read_text(encoding="utf-8"))
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_CUTOFF_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= PREDICTED_UNION
        assert verdict["non_clock_moves"] == sorted(state.P332_14_RUNG5_MOVED_COLUMNS)

    def test_every_builders_measured_subset_is_inside_its_prediction(self) -> None:
        split = fg.phase332_cutoff_moved_by_builder(self._verdict()["non_clock_moves"])
        assert split["unmapped"] == []
        for builder, predicted in PREDICTED_BY_BUILDER.items():
            assert set(split[builder]) <= set(predicted), builder
        assert {
            builder: tuple(columns) for builder, columns in split.items()
        } == state.P332_14_RUNG5_MOVED_BY_BUILDER

    def test_no_forecast_column_moved(self) -> None:
        moved = set(self._verdict()["non_clock_moves"])
        assert not moved & fg.phase332_weather_columns()

    def test_the_widths_and_rows_did_not_move(self) -> None:
        before = json.loads(STEP4B.read_text(encoding="utf-8"))
        after = json.loads(RUNG5.read_text(encoding="utf-8"))
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["width"] == after[matrix]["width"]
            assert before[matrix]["rows"] == after[matrix]["rows"]

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_CUTOFF_RUNG_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_14_RUNG5_CAUSE_DIGEST

    def test_the_committed_diff_records_the_rung(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"]["5"]
        assert rung["cause"] == fg.PHASE332_CUTOFF_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_cutoff"
        assert rung["baseline_document"] == STEP4B.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["widths_before"] == rung["widths_after"]
        assert rung["moved_by_builder"]["unmapped"] == []
        for builder, predicted in PREDICTED_BY_BUILDER.items():
            assert rung["predicted_by_builder"][builder] == list(predicted)
            assert set(rung["moved_by_builder"][builder]) <= set(predicted)
