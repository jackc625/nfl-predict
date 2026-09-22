"""Extra step 3b of the Phase-33.2 gold ladder: the surface-classification fix.

Plan 33.2-10, orchestrator-assigned (D33.2-20: one cause, its own rebuild, its own
attribution). The fix is NOT a numbered rung -- rungs 4-9 belong to later plans -- so it is
registered as the extra step ``"3b"``, which FOLLOWS rung 3: it is judged against
``p332_rung3.json``, and rung 4 will be judged against ``p332_rung3b.json``.

What is asserted:

* the step is registered in the extra-step tables and judged by ``_attribute_p332_surface``,
  its id cannot collide with a numbered rung, and the numbered-rung cause table is unchanged
  (its keys stay integers the tooling can sort);
* the ladder order: step 3b needs rungs 0-3, rung 4 needs rungs 0-3 AND step 3b, and rung 4's
  baseline is the step's document;
* the explainable set is derived: only ``surface_mismatch``, only from the earliest season
  holding a reclassified game (2005);
* live, against gitignored evidence: the attribution is clean, and row by row every moved
  row is in a season on or after 2005 and every reclassified game through 2025 moved.

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
RUNG3 = rung_document_path(
    FINGERPRINT_DIR, PHASE332_SCHEDULE_MOVE_RUNG, PHASE332_RUNG_PREFIX
)
STEP3B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_SURFACE_STEP, PHASE332_RUNG_PREFIX
)
GOLD_BEFORE_DIR = Path("outputs/p332_rung3b_gold_before")
GOLD_AFTER_DIR = Path("data/gold")

SEASON_FLOOR = 2005


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
        assert PHASE332_SURFACE_STEP == "3b"
        steps = fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert steps[PHASE332_SURFACE_STEP] == PHASE332_SCHEDULE_MOVE_RUNG
        # The first extra step registered on this ladder (later ones follow it).
        assert next(iter(steps)) == PHASE332_SURFACE_STEP
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_SURFACE_STEP]
            == fg.PHASE332_SURFACE_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_SURFACE_STEP]
            is fg._attribute_p332_surface
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES[PHASE332_SURFACE_STEP]
            is fg.PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE
        )

    def test_the_numbered_tables_are_untouched_by_the_step(self) -> None:
        """Integer keys only: rungs 4-9 stay free, and the tooling can still sort them."""
        for table in (
            fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX],
            fg.PHASE332_RUNG_SIGNATURES,
            fg.PHASE332_RUNG_ATTRIBUTORS,
        ):
            assert all(isinstance(key, int) for key in table)
            assert PHASE332_SURFACE_STEP not in table
        # The numbered rungs stay a contiguous run of integers the tooling can range over
        # (rung 4, Plan 33.2-12, was registered after this step without disturbing it).
        numbered = sorted(fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX])
        assert numbered == list(range(1, numbered[-1] + 1))

    def test_the_step_is_judged_by_its_own_attributor(self, monkeypatch) -> None:
        calls: list[str] = []
        original = fg.PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_SURFACE_STEP]

        def spy(*args, **kwargs):
            calls.append("p332_surface")
            return original(*args, **kwargs)

        monkeypatch.setitem(
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS, PHASE332_SURFACE_STEP, spy
        )
        verdict = attribute_rung(
            _report({"surface_mismatch": ["2005", "2025"]}),
            PHASE332_SURFACE_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert calls == ["p332_surface"] * len(fg.GOLD_MATRICES)
        assert verdict["ok"], verdict["failures"]
        assert verdict["cause"] == fg.PHASE332_SURFACE_STEP_CAUSE

    def test_the_signature_handed_out_is_a_copy(self) -> None:
        signature = fg._expected_signature(
            PHASE332_SURFACE_STEP, prefix=PHASE332_RUNG_PREFIX
        )
        assert signature == fg.PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE
        assert signature is not fg.PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True

    def test_an_unregistered_string_step_is_refused(self) -> None:
        registered = fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]
        unregistered = next(
            f"3{letter}"
            for letter in "bcdefghijklmnopqrstuvwxyz"
            if f"3{letter}" not in registered
        )
        with pytest.raises(ValueError, match="Unknown"):
            attribute_rung(
                _report({"surface_mismatch": ["2005"]}),
                unregistered,
                rung_prefix=PHASE332_RUNG_PREFIX,
            )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1


class TestTheLadderOrder:
    def test_step_3b_needs_rungs_zero_to_three(self) -> None:
        assert fg._ladder_predecessors(PHASE332_SURFACE_STEP, PHASE332_RUNG_PREFIX) == [
            0,
            1,
            2,
            3,
        ]

    def test_rung_four_needs_step_3b_after_rung_three(self) -> None:
        """Step 3b sits directly after rung 3; a step registered later (3c, Plan
        33.2-12) follows it, so rung 4 is judged against the LAST step after rung 3."""
        predecessors = fg._ladder_predecessors(4, PHASE332_RUNG_PREFIX)
        assert predecessors[:5] == [0, 1, 2, 3, "3b"]
        assert all(isinstance(entry, str) for entry in predecessors[5:])
        assert fg._ladder_predecessors(3, PHASE332_RUNG_PREFIX) == [0, 1, 2]

    def test_baselines_follow_the_ladder_order(self, tmp_path) -> None:
        assert (
            phase332_baseline_document_path(tmp_path, PHASE332_SURFACE_STEP).name
            == "p332_rung3.json"
        )
        last = fg._ladder_predecessors(4, PHASE332_RUNG_PREFIX)[-1]
        assert phase332_baseline_document_path(tmp_path, 4).name == (
            f"p332_rung{last}.json"
        )

    def test_other_prefixes_have_no_extra_steps(self) -> None:
        assert fg._ladder_predecessors(3, "p331_") == [0, 1, 2]


class TestTheExplainableSetIsDerived:
    def test_only_surface_mismatch_from_2005(self) -> None:
        assert fg.PHASE332_SURFACE_STEP_COLUMNS == ("surface_mismatch",)
        assert fg.phase332_surface_step_earliest_season() == SEASON_FLOOR

    def test_every_reclassified_game_is_at_a_reclassified_venue(self) -> None:
        import pandas as pd

        games = pd.read_parquet("data/silver/games.parquet").set_index("game_id")
        reclassified = fg.phase332_surface_reclassified_games()
        assert reclassified, "the fix must change at least one game"
        venues = {
            "LON00",
            "LON01",
            "LON02",
            "MEX00",
            "FRA00",
            "GER00",
            "SAO00",
            "DUB00",
            "BER00",
        }
        assert {games.at[g, "stadium_id"] for g in reclassified} <= venues

    def test_a_column_other_than_surface_mismatch_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"venue_outdoor": ["2010"]}),
            PHASE332_SURFACE_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_surface_mismatch_before_the_floor_is_unattributed(self) -> None:
        verdict = attribute_rung(
            _report({"surface_mismatch": ["2004", "2005"]}),
            PHASE332_SURFACE_STEP,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert not verdict["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        verdict = attribute_rung(
            _report({}), PHASE332_SURFACE_STEP, rung_prefix=PHASE332_RUNG_PREFIX
        )
        assert not verdict["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG3.is_file() and STEP3B.is_file()),
    reason=(
        "the p332_ rung 3 / step 3b fingerprint documents are not present under "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveStep:
    def test_the_ladder_is_complete_through_the_step(self) -> None:
        names = [
            p.name
            for p in require_rung_ladder(
                FINGERPRINT_DIR, PHASE332_SURFACE_STEP, PHASE332_RUNG_PREFIX
            )
        ]
        assert names[-1] == RUNG3.name
        assert STEP3B.is_file()

    def test_the_attribution_is_clean(self) -> None:
        before = json.loads(RUNG3.read_text(encoding="utf-8"))
        after = json.loads(STEP3B.read_text(encoding="utf-8"))
        report = compare_fingerprints(before, after)
        verdict = attribute_rung(
            report,
            PHASE332_SURFACE_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )
        assert verdict["ok"], verdict["failures"]
        assert verdict["non_clock_moves"] == ["surface_mismatch"]

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"][PHASE332_SURFACE_STEP]
        assert step["cause"] == fg.PHASE332_SURFACE_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_surface"
        assert step["follows_rung"] == PHASE332_SCHEDULE_MOVE_RUNG
        assert step["baseline_document"] == RUNG3.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["moved_columns"] == ["surface_mismatch"]


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in fg.GOLD_MATRICES),
    reason="outputs/p332_rung3b_gold_before/ is absent -- gitignored runtime evidence",
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
        step = json.loads(STEP3B.read_text(encoding="utf-8"))[matrix]
        live = fg.fingerprint_matrix(after.reset_index())["columns"]
        if {c: v for c, v in live.items() if not fg._is_build_clock(c)} != {
            c: v for c, v in step["columns"].items() if not fg._is_build_clock(c)
        }:
            pytest.skip("live gold has moved on since the step-3b document was written")
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
    def test_only_surface_mismatch_moved_on_or_after_the_floor(self, matrix) -> None:
        before, moved = self._moves(matrix)
        assert set(moved) == {"surface_mismatch"}
        seasons = set(
            before.loc[sorted(moved["surface_mismatch"]), "season"].astype(int)
        )
        assert min(seasons) >= SEASON_FLOOR

    @pytest.mark.parametrize("matrix", fg.GOLD_MATRICES)
    def test_every_reclassified_game_through_2025_moved(self, matrix) -> None:
        _, moved = self._moves(matrix)
        reclassified = {
            g for g, s in fg.phase332_surface_reclassified_games().items() if s <= 2025
        }
        assert reclassified <= moved["surface_mismatch"]
