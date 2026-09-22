"""Extra step 4b of the Phase-33.2 gold ladder: a retractable roof is unknown at the lock.

Plan 33.2-14, orchestrator-assigned (owner ruling 2026-09-21, "Treat as unknown at lock";
D33.2-20: one cause, its own rebuild, its own attribution). Rung 4 turned 620 games played
with a retractable roof CLOSED into domes with no weather; whether such a roof closes is
decided near kickoff, often because of the weather, so it is post-lock information. Step 4b
gives every retractable-venue game the day-before forecast, as an outdoor game does.

It follows rung 4, is judged against a RETAKEN baseline (Plan 33.2-13 changed the injury and
QB builders after rung 4 was built), and rung 5 is judged against ``p332_rung4b.json``.

What is asserted:

* the step is registered in the extra-step tables and judged by
  ``_attribute_p332_retractable_roof``; its id cannot collide with a numbered rung; no new
  prefix branch was added;
* the ladder order: step 4b needs rungs 0-4 and steps 3b, 3c; rung 5 needs step 4b last; the
  step reads its retaken baseline, never the stale predecessor;
* the explainable set is the derived weather family; a non-weather column is unattributed;
* the carry-in (p332_rung4.json -> the retaken baseline) stays inside the injury and QB
  prediction declared before it was measured, and is recorded beside the step, not in it;
* live, against gitignored evidence: the attribution is clean, and row by row exactly the 620
  redecided games changed ``weather_affects_game`` while ``venue_retractable`` did not move.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import scripts.fingerprint_gold as fg
import tests.phase33_state as state
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_RETRACTABLE_ROOF_STEP,
    PHASE332_RUNG_PREFIX,
    PHASE332_WEATHER_RUNG,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP4B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_RETRACTABLE_ROOF_STEP, PHASE332_RUNG_PREFIX
)
RETAKEN = FINGERPRINT_DIR / fg.PHASE332_RETRACTABLE_ROOF_STEP_BASELINE_DOCUMENT
RUNG4 = rung_document_path(FINGERPRINT_DIR, PHASE332_WEATHER_RUNG, PHASE332_RUNG_PREFIX)
GOLD_AFTER_DIR = Path("data/gold")
SILVER_BEFORE = Path("outputs/p332_rung4b_silver_before/weather.parquet")
SILVER_AFTER = Path("data/silver/weather.parquet")


def _report(changed: dict[str, list[str]]) -> dict:
    matrix = {
        "width_before": 193,
        "width_after": 193,
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


class TestTheStepIsRegisteredAndDispatched:
    def test_the_step_is_an_extra_step_that_follows_rung_four(self) -> None:
        assert PHASE332_RETRACTABLE_ROOF_STEP == "4b"
        steps = fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert steps[PHASE332_RETRACTABLE_ROOF_STEP] == PHASE332_WEATHER_RUNG
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["4b"]
            == fg.PHASE332_RETRACTABLE_ROOF_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["4b"]
            is fg._attribute_p332_retractable_roof
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["4b"]
            is fg.PHASE332_RETRACTABLE_ROOF_STEP_EXPECTED_SIGNATURE
        )

    def test_the_numbered_tables_are_untouched_by_the_step(self) -> None:
        for table in (
            fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX],
            fg.PHASE332_RUNG_SIGNATURES,
            fg.PHASE332_RUNG_ATTRIBUTORS,
        ):
            assert all(isinstance(key, int) for key in table)
            assert "4b" not in table

    def test_the_signature_was_declared_before_the_rebuild(self) -> None:
        signature = fg._expected_signature("4b", prefix=PHASE332_RUNG_PREFIX)
        assert signature == fg.PHASE332_RETRACTABLE_ROOF_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        import hashlib

        digest = hashlib.sha256(
            fg.PHASE332_RETRACTABLE_ROOF_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_14_STEP4B_CAUSE_DIGEST

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1


class TestTheLadderOrder:
    def test_step_4b_needs_rung_four_and_every_earlier_step(self) -> None:
        assert fg._ladder_predecessors("4b", PHASE332_RUNG_PREFIX) == [
            0,
            1,
            2,
            3,
            "3b",
            "3c",
            4,
        ]

    def test_rung_five_is_judged_against_step_4b(self, tmp_path) -> None:
        assert fg._ladder_predecessors(5, PHASE332_RUNG_PREFIX)[-1] == "4b"
        assert phase332_baseline_document_path(tmp_path, 5).name == "p332_rung4b.json"

    def test_step_4b_reads_its_retaken_baseline_and_refuses_without_it(
        self, tmp_path
    ) -> None:
        with pytest.raises(fg.MissingPredecessorFingerprintError):
            phase332_baseline_document_path(tmp_path, "4b")
        (tmp_path / RETAKEN.name).write_text("{}", encoding="utf-8")
        assert phase332_baseline_document_path(tmp_path, "4b").name == RETAKEN.name


class TestTheExplainableSetIsTheWeatherFamily:
    def test_a_weather_column_is_attributed(self) -> None:
        verdict = attribute_rung(
            _report({"temp_f": ["2002", "2019"]}),
            "4b",
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]

    @pytest.mark.parametrize(
        "column", ["venue_retractable", "home_qb_adjustment", "home_rest_days"]
    )
    def test_a_non_weather_column_is_unattributed(self, column: str) -> None:
        verdict = attribute_rung(
            _report({column: ["2019"]}), "4b", rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not verdict["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        verdict = attribute_rung(_report({}), "4b", rung_prefix=PHASE332_RUNG_PREFIX)
        assert not verdict["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP4B.is_file() and RETAKEN.is_file() and RUNG4.is_file()),
    reason=(
        "the p332_ rung-4, retaken-baseline or step-4b fingerprint documents are absent "
        "from outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveStep:
    def test_the_attribution_is_clean(self) -> None:
        before = json.loads(RETAKEN.read_text(encoding="utf-8"))
        after = json.loads(STEP4B.read_text(encoding="utf-8"))
        verdict = attribute_rung(
            compare_fingerprints(before, after),
            "4b",
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert set(verdict["non_clock_moves"]) <= fg.phase332_weather_columns()
        assert (
            len(verdict["non_clock_moves"]) == state.P332_14_STEP4B_MOVED_COLUMN_COUNT
        )

    def test_the_carry_in_is_inside_its_declared_prediction(self) -> None:
        carry_in = fg.phase332_step4b_carry_in()
        assert carry_in["outside_prediction"] == []
        assert carry_in["by_builder"] == {
            builder: list(columns)
            for builder, columns in state.P332_14_STEP4B_CARRY_IN_BY_BUILDER.items()
        }
        for matrix in fg.GOLD_MATRICES:
            structure = carry_in["structure"][matrix]  # type: ignore[index]
            assert structure["added"] == [] and structure["removed"] == []
            assert structure["rows"][0] == structure["rows"][1]

    def test_the_committed_diff_records_the_step_and_its_carry_in(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["4b"]
        assert step["cause"] == fg.PHASE332_RETRACTABLE_ROOF_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_retractable_roof"
        assert step["follows_rung"] == PHASE332_WEATHER_RUNG
        assert step["baseline_document"] == RETAKEN.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]
        assert step["carry_in"]["outside_prediction"] == []
        assert step["carry_in"]["from_document"] == RUNG4.name


needs_gold = pytest.mark.skipif(
    not (SILVER_BEFORE.is_file() and SILVER_AFTER.is_file()),
    reason="the step-4b silver copy under outputs/ is absent -- gitignored evidence",
)


@needs_gold
@needs_ladder
class TestTheRowRule:
    def test_exactly_the_redecided_games_moved_in_silver(self) -> None:
        before = pd.read_parquet(SILVER_BEFORE).set_index("game_id")
        after = pd.read_parquet(SILVER_AFTER).set_index("game_id")
        redecided = before.index[
            ~before["is_outdoor"].astype(bool)
            & after.loc[before.index, "is_outdoor"].astype(bool)
        ]
        assert len(redecided) == state.P332_14_STEP4B_REDECIDED_GAMES
        assert not any(game.startswith("2026") for game in redecided)
        assert state.P332_14_STEP4B_ARCHIVE_GAP_GAME in redecided

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_the_redecided_games_now_weather_apply_and_the_flag_did_not_move(
        self, matrix: str
    ) -> None:
        gold = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        live = fg.fingerprint_matrix(gold.reset_index())["columns"]
        recorded = json.loads(STEP4B.read_text(encoding="utf-8"))[matrix]["columns"]
        if {c: v for c, v in live.items() if not fg._is_build_clock(c)} != {
            c: v for c, v in recorded.items() if not fg._is_build_clock(c)
        }:
            pytest.skip("live gold has moved on since the step-4b document was written")
        before = pd.read_parquet(SILVER_BEFORE).set_index("game_id")
        redecided = before.index[~before["is_outdoor"].astype(bool)].intersection(
            gold.index[gold["weather_affects_game"] == 1.0]
        )
        assert len(redecided) == state.P332_14_STEP4B_REDECIDED_GAMES
        gap = gold.loc[state.P332_14_STEP4B_ARCHIVE_GAP_GAME]
        assert gap["weather_affects_game"] == 1.0 and gap["weather_coverage"] == 0.0
        retaken = json.loads(RETAKEN.read_text(encoding="utf-8"))[matrix]["columns"]
        assert retaken["venue_retractable"] == recorded["venue_retractable"]
