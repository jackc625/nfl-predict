"""Extra step 3c of the Phase-33.2 gold ladder: the kickoff-hour correction.

Plan 33.2-12, orchestrator-assigned (D33.2-20: one cause, its own rebuild, its own
attribution). The 68 2002-2005 Monday and Thursday night games stored at 09:00 ET move to
21:00 ET on the same date. The fix runs BEFORE rung 4, whose forecast-hour selection reads
the kickoff hour, so it is registered as the extra step ``"3c"``: it follows rung 3 after
step 3b, is judged against ``p332_rung3b.json``, and rung 4 is judged against
``p332_rung3c.json``.

What is asserted:

* the step is registered in the extra-step tables and judged by
  ``_attribute_p332_kickoff_hour``, its id cannot collide with a numbered rung, and no new
  prefix branch was added;
* the ladder order: step 3c needs rungs 0-3 and step 3b, rung 4 needs step 3c last;
* the explainable set is derived: only the eight rest-day columns, only from the earliest
  season holding a game whose rest count moves (2002), and no weekday or weather column;
* live, against gitignored evidence: the attribution is clean, and row by row every moved
  row is in a rest-day column and every game whose rest count moves did move.

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
    PHASE332_SCHEDULE_MOVE_RUNG,
    PHASE332_SURFACE_STEP,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    require_rung_ladder,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP3B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_SURFACE_STEP, PHASE332_RUNG_PREFIX
)
STEP3C = rung_document_path(
    FINGERPRINT_DIR, PHASE332_KICKOFF_HOUR_STEP, PHASE332_RUNG_PREFIX
)
GOLD_BEFORE_DIR = Path("outputs/p332_rung3c_gold_before")
GOLD_AFTER_DIR = Path("data/gold")

SEASON_FLOOR = 2002
REST_COLUMNS = frozenset(fg.PHASE332_KICKOFF_HOUR_STEP_COLUMNS)
WEEKDAY_COLUMNS = ("thursday_game", "monday_game", "short_week", "game_day_of_week")


def _report(changed: dict[str, list[str]]) -> dict:
    matrix = {
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
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


class TestTheStepIsRegisteredAndDispatched:
    def test_the_step_is_an_extra_step_that_follows_rung_three(self) -> None:
        assert PHASE332_KICKOFF_HOUR_STEP == "3c"
        steps = fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert steps[PHASE332_KICKOFF_HOUR_STEP] == PHASE332_SCHEDULE_MOVE_RUNG
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][
                PHASE332_KICKOFF_HOUR_STEP
            ]
            == fg.PHASE332_KICKOFF_HOUR_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_KICKOFF_HOUR_STEP]
            is fg._attribute_p332_kickoff_hour
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES[PHASE332_KICKOFF_HOUR_STEP]
            is fg.PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE
        )

    def test_the_numbered_tables_are_untouched_by_the_step(self) -> None:
        for table in (
            fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX],
            fg.PHASE332_RUNG_SIGNATURES,
            fg.PHASE332_RUNG_ATTRIBUTORS,
        ):
            assert all(isinstance(key, int) for key in table)
            assert PHASE332_KICKOFF_HOUR_STEP not in table

    def test_the_step_is_judged_by_its_own_attributor(self, monkeypatch) -> None:
        calls: list[str] = []
        original = fg.PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_KICKOFF_HOUR_STEP]

        def spy(*args, **kwargs):
            calls.append("p332_kickoff_hour")
            return original(*args, **kwargs)

        monkeypatch.setitem(
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS, PHASE332_KICKOFF_HOUR_STEP, spy
        )
        verdict = attribute_rung(
            _report({"home_rest_days": ["2002", "2003"]}),
            PHASE332_KICKOFF_HOUR_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332_kickoff_hour"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]
        assert verdict["cause"] == fg.PHASE332_KICKOFF_HOUR_STEP_CAUSE

    def test_the_signature_was_declared_before_the_rebuild(self) -> None:
        signature = fg._expected_signature(
            PHASE332_KICKOFF_HOUR_STEP, prefix=PHASE332_RUNG_PREFIX
        )
        assert signature == fg.PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE
        assert signature is not fg.PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1


class TestTheLadderOrder:
    def test_step_3c_needs_step_3b_after_rung_three(self) -> None:
        assert fg._ladder_predecessors(
            PHASE332_KICKOFF_HOUR_STEP, PHASE332_RUNG_PREFIX
        ) == [0, 1, 2, 3, "3b"]

    def test_rung_four_is_judged_against_step_3c(self, tmp_path) -> None:
        assert fg._ladder_predecessors(4, PHASE332_RUNG_PREFIX) == [
            0,
            1,
            2,
            3,
            "3b",
            "3c",
        ]
        assert (
            phase332_baseline_document_path(tmp_path, PHASE332_KICKOFF_HOUR_STEP).name
            == "p332_rung3b.json"
        )
        assert phase332_baseline_document_path(tmp_path, 4).name == "p332_rung3c.json"

    def test_step_3b_is_still_judged_against_rung_three(self, tmp_path) -> None:
        assert (
            phase332_baseline_document_path(tmp_path, PHASE332_SURFACE_STEP).name
            == "p332_rung3.json"
        )


class TestTheExplainableSetIsDerived:
    def test_only_rest_columns_from_2002(self) -> None:
        assert fg.phase332_kickoff_hour_step_earliest_season() == SEASON_FLOOR
        assert not REST_COLUMNS & set(WEEKDAY_COLUMNS)
        assert not any(
            "weather" in c or "wind" in c or "temp" in c for c in REST_COLUMNS
        )

    def test_every_rest_change_is_a_corrected_game_or_its_teams_next_game(self) -> None:
        from scripts.ingest_games import load_kickoff_hour_corrections

        changed = fg.phase332_kickoff_hour_rest_changes()
        corrections = load_kickoff_hour_corrections()
        assert len(changed) > 0
        assert set(changed.values()) <= {2002, 2003, 2004, 2005, 2006}
        corrected_teams = {
            (game_id[:4], team)
            for game_id in corrections
            for team in game_id.split("_")[2].split("@")
        }
        for game_id in changed:
            teams = set(game_id.split("_")[2].split("@"))
            assert game_id in corrections or any(
                (game_id[:4], t) in corrected_teams for t in teams
            ), game_id

    def test_a_weekday_column_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"monday_game": ["2003"]}),
            PHASE332_KICKOFF_HOUR_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_a_weather_column_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"wind_moderate": ["2003"]}),
            PHASE332_KICKOFF_HOUR_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        verdict = attribute_rung(
            _report({}), PHASE332_KICKOFF_HOUR_STEP, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not verdict["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP3B.is_file() and STEP3C.is_file()),
    reason=(
        "the p332_ step 3b / step 3c fingerprint documents are not present under "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveStep:
    def test_the_ladder_is_complete_through_the_step(self) -> None:
        names = [
            p.name
            for p in require_rung_ladder(
                FINGERPRINT_DIR, PHASE332_KICKOFF_HOUR_STEP, PHASE332_RUNG_PREFIX
            )
        ]
        assert names[-1] == STEP3B.name

    def test_the_attribution_is_clean(self) -> None:
        before = json.loads(STEP3B.read_text(encoding="utf-8"))
        after = json.loads(STEP3C.read_text(encoding="utf-8"))
        verdict = attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_KICKOFF_HOUR_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert set(verdict["non_clock_moves"]) <= REST_COLUMNS
        assert verdict["non_clock_moves"]

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"][PHASE332_KICKOFF_HOUR_STEP]
        assert step["cause"] == fg.PHASE332_KICKOFF_HOUR_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_kickoff_hour"
        assert step["follows_rung"] == PHASE332_SCHEDULE_MOVE_RUNG
        assert step["baseline_document"] == STEP3B.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert set(step["moved_columns"]) <= REST_COLUMNS
        assert step["widths_before"] == step["widths_after"]


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason="outputs/p332_rung3c_gold_before/ is absent -- gitignored runtime evidence",
)


@needs_gold_copy
@needs_ladder
class TestTheRowRule:
    @staticmethod
    def _moves(matrix: str) -> tuple[pd.DataFrame, dict[str, set[str]]]:
        before = pd.read_parquet(GOLD_BEFORE_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        step = json.loads(STEP3C.read_text(encoding="utf-8"))[matrix]
        live = fg.fingerprint_matrix(after.reset_index())["columns"]
        if {c: v for c, v in live.items() if not fg._is_build_clock(c)} != {
            c: v for c, v in step["columns"].items() if not fg._is_build_clock(c)
        }:
            pytest.skip("live gold has moved on since the step-3c document was written")
        moved: dict[str, set[str]] = {}
        for column in before.columns:
            if fg._is_build_clock(column):
                continue
            left, right = before[column], after[column]
            same = (left == right) | (left.isna() & right.isna())
            if (~same).any():
                moved[column] = set(before.index[~same])
        return before, moved

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_only_rest_columns_moved_on_or_after_the_floor(self, matrix) -> None:
        before, moved = self._moves(matrix)
        assert set(moved) <= REST_COLUMNS
        rows = set().union(*moved.values())
        seasons = set(before.loc[sorted(rows), "season"].astype(int))
        assert min(seasons) >= SEASON_FLOOR

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_every_game_whose_rest_count_moves_moved(self, matrix) -> None:
        """Every derived game moved in gold, bar the season-opening rows gold cannot scale.

        The derivation scores the CONTEXTUAL rest count. Gold then z-scores the column
        within each season, and a week-1 row has no earlier row of its season to be
        scaled against, so it reads exactly 0.0 whatever its raw count. MEASURED
        2026-09-21: 199 of the 201 derived games moved; the other two are 2003 week-1
        games (a team whose previous game was a 2002 Monday-night game) reading 0.0 in
        both rest columns before and after. The exception is that shape exactly, not a
        list of games.
        """
        before, moved = self._moves(matrix)
        rest_rows = moved.get("home_rest_days", set()) | moved.get(
            "away_rest_days", set()
        )
        after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        changed = {
            g for g, s in fg.phase332_kickoff_hour_rest_changes().items() if s <= 2025
        }
        masked = sorted(changed - rest_rows)
        for game_id in masked:
            assert int(before.loc[game_id, "week"]) == 1, game_id
            for column in ("home_rest_days", "away_rest_days"):
                assert before.loc[game_id, column] == 0.0, (game_id, column)
                assert after.loc[game_id, column] == 0.0, (game_id, column)
        assert len(masked) < len(changed) / 10
