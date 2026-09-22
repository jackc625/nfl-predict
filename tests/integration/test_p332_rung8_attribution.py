"""Rung 8 of the Phase-33.2 gold ladder: the coverage floors are retired.

Plan 33.2-18 Task 3 (D33.2-14, D33.2-08 item 2, SPEC R10). ONE cause, seen from several sides:
the input-selection window's 2018 was a coverage floor, and retiring it moves the window to 2002
(``conf/season_partition.py``) and releases every family 2018 floored -- team form computed back
to 2002, the opponent-adjusted pool widened through the identity binding, and snaps / injury /
completion probability an honest unknown beside a flag before their coverage. Attributed FACET
BY FACET so a surprise is bisected without a second rung.

THE PREDICTION, declared before the rebuild (the constants below, equal to
``scripts.fingerprint_gold``'s ``PHASE332_WINDOW_RUNG_*``):

* FACET A (team form): 28 value columns may move in any season 2002-2025 and MUST move in
  2002-2019; two ``{side}_off_rolling_cpoe_coverage`` flags are added;
* FACET B (opponent-adjusted): 12 values and their 4 flags may move in any season 2002-2025 and
  MUST move in 2002-2017; nothing added;
* FACET C (snaps and injury): 20 snap and 12 injury-family columns may move in any season
  2002-2025 and MUST move in 2002-2012; two ``{side}_snap_coverage`` flags are added;
* nothing outside the union moves -- including the eight never-populated defensive team-form
  copies, predicted UNMOVED -- nothing is removed, rows hold, and every width grows by four.

The rung is judged against a RETAKEN baseline: Plan 33.2-17 Task 1's silver rebuild also
re-derived 2020 onward for reasons that are not this rung's, and that carry-in is measured
between ``p332_rung7b.json`` and the retaken baseline, inside a prediction declared before it.

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
from features.team_form import TeamFormCalculator
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_IMPUTATION_STEP,
    PHASE332_RUNG_PREFIX,
    PHASE332_WINDOW_RUNG,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP7B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_IMPUTATION_STEP, PHASE332_RUNG_PREFIX
)
RETAKEN = FINGERPRINT_DIR / "p332_rung7b_retaken.json"
RUNG8 = rung_document_path(FINGERPRINT_DIR, PHASE332_WINDOW_RUNG, PHASE332_RUNG_PREFIX)
DIGEST_BEFORE = Path("outputs/p332_rung8_before.json")
DIGEST_AFTER = Path("outputs/p332_rung8_after.json")
GOLD = Path("data/gold")

#: THE DETERMINATION, made from the consumers BEFORE the rebuild -- never from its diff.
EXPECTED_CASE: str = "runs"
"""Rung 8 RUNS: it moves gold independently of model selection.

Consumer trace, read before the rebuild: ``features/team_form.py`` binds
``TEAM_FORM_PER_GAME_FIRST_SEASON = SELECTION_WINDOW_FIRST_SEASON`` by identity, and
``TeamFormCalculator.per_game_seasons`` drops every season below it from the opponent-adjusted
per-game pool that ``get_per_game_stats`` loads. Moving the window from 2018 to 2002 therefore
widens that pool and changes the twelve opponent-adjusted gold columns on its own, whatever any
fit later selects -- so "selection only, the diff may be empty" is not a live possibility, and an
empty diff here is a FINDING that halts the run, not a declared-but-not-run record.
"""

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded BEFORE the rebuild.
# ---------------------------------------------------------------------------

FACET_A_CHANGED: tuple[str, ...] = tuple(
    sorted(
        f"{side}_{unit}_rolling_{metric}"
        for side in ("home", "away")
        for unit, metrics in (
            (
                "off",
                (
                    "avg_drive_start_yardline",
                    "cpoe",
                    "neutral_pace",
                    "neutral_pass_rate",
                    "pass_success_rate",
                    "red_zone_td_rate",
                    "rush_success_rate",
                    "success_rate",
                    "third_down_conversion_rate",
                ),
            ),
            (
                "def",
                (
                    "pass_success_rate",
                    "red_zone_td_rate",
                    "rush_success_rate",
                    "success_rate",
                    "third_down_conversion_rate",
                ),
            ),
        )
        for metric in metrics
    )
)
FACET_A_ADDED: tuple[str, ...] = (
    "away_off_rolling_cpoe_coverage",
    "home_off_rolling_cpoe_coverage",
)
FACET_B_CHANGED: tuple[str, ...] = tuple(
    sorted(
        f"{side}_{unit}_rolling_opp_adj_{name}"
        for side in ("home", "away")
        for unit in ("off", "def")
        for name in ("coverage", "epa_per_play", "pass_epa", "rush_epa")
    )
)
FACET_C_SNAP: tuple[str, ...] = tuple(
    sorted(
        [
            f"{side}_rolling_snap_share_{position}"
            for side in ("home", "away")
            for position in ("db", "dl", "lb", "ol", "qb", "rb", "te", "wr")
        ]
        + [
            f"{side}_snap_{name}"
            for side in ("home", "away")
            for name in ("concentration", "continuity")
        ]
    )
)
FACET_C_INJURY: tuple[str, ...] = tuple(
    sorted(
        f"{side}_{name}"
        for side in ("home", "away")
        for name in (
            "availability_coverage",
            "availability_fraction",
            "backup_quality_delta",
            "date_modified_coverage",
            "injury_coverage",
            "qb_out_flag",
        )
    )
)
FACET_C_CHANGED: tuple[str, ...] = tuple(sorted(FACET_C_SNAP + FACET_C_INJURY))
FACET_C_ADDED: tuple[str, ...] = ("away_snap_coverage", "home_snap_coverage")
PREDICTED_UNMOVED: tuple[str, ...] = tuple(
    sorted(
        f"{side}_def_rolling_{metric}"
        for side in ("home", "away")
        for metric in (
            "avg_drive_start_yardline",
            "cpoe",
            "neutral_pace",
            "neutral_pass_rate",
        )
    )
)
REQUIRED_SEASONS: dict[str, tuple[str, ...]] = {
    "A_team_form": tuple(str(s) for s in range(2002, 2020)),
    "B_opponent_adjusted": tuple(str(s) for s in range(2002, 2018)),
    "C_snaps_injury": tuple(str(s) for s in range(2002, 2013)),
}
PREDICTED_WIDTHS_BEFORE: tuple[int, int, int] = (197, 198, 197)
PREDICTED_WIDTHS_AFTER: tuple[int, int, int] = (201, 202, 201)
PREDICTED_ROWS: int = 6499

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)

FACET_CHANGED: dict[str, tuple[str, ...]] = {
    "A_team_form": FACET_A_CHANGED,
    "B_opponent_adjusted": FACET_B_CHANGED,
    "C_snaps_injury": FACET_C_CHANGED,
}


def _report(
    changed: dict[str, list[str]],
    *,
    added: tuple[str, ...] = tuple(sorted(FACET_A_ADDED + FACET_C_ADDED)),
    removed: tuple[str, ...] = (),
    width_delta: int = 4,
) -> dict:
    matrix = {
        "width_before": 197,
        "width_after": 197 + width_delta,
        "rows_before": PREDICTED_ROWS,
        "rows_after": PREDICTED_ROWS,
        "rows_per_season_before": {},
        "rows_per_season_after": {},
        "columns_added": list(added),
        "columns_removed": list(removed),
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


def _complete_diff() -> dict[str, list[str]]:
    """A diff moving one column of each facet across that facet's required seasons."""
    return {
        FACET_A_CHANGED[0]: [*REQUIRED_SEASONS["A_team_form"], "2023"],
        FACET_B_CHANGED[0]: list(REQUIRED_SEASONS["B_opponent_adjusted"]),
        FACET_C_CHANGED[0]: [*REQUIRED_SEASONS["C_snaps_injury"], "2025"],
    }


def _verdict(changed: dict[str, list[str]], **kwargs) -> dict:
    return attribute_rung(
        _report(changed, **kwargs),
        PHASE332_WINDOW_RUNG,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheRungIsRegisteredAndDispatched:
    def test_it_is_rung_eight_with_one_cause(self) -> None:
        assert PHASE332_WINDOW_RUNG == 8
        causes = fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert causes[8] == fg.PHASE332_WINDOW_RUNG_CAUSE
        cause = fg.PHASE332_WINDOW_RUNG_CAUSE
        assert "THE COVERAGE FLOORS ARE RETIRED" in cause
        assert "faces of the one retirement" in cause

    def test_both_dispatch_tables_carry_it(self) -> None:
        assert (
            fg.PHASE332_RUNG_SIGNATURES[8] is fg.PHASE332_WINDOW_RUNG_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_RUNG_ATTRIBUTORS[8] is fg._attribute_p332_window

    def test_it_is_judged_against_its_retaken_baseline(self, tmp_path) -> None:
        assert fg._ladder_predecessors(8, PHASE332_RUNG_PREFIX)[-1] == "7b"
        with pytest.raises(fg.MissingPredecessorFingerprintError):
            phase332_baseline_document_path(tmp_path, 8)
        (tmp_path / RETAKEN.name).write_text("{}", encoding="utf-8")
        assert phase332_baseline_document_path(tmp_path, 8).name == RETAKEN.name

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild_and_expected_to_run(self) -> None:
        signature = fg.PHASE332_WINDOW_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["expected_case"] == EXPECTED_CASE == "runs"
        assert fg.PHASE332_WINDOW_RUNG_EXPECTED_CASE == EXPECTED_CASE


class TestThePredictionIsFacetPartitioned:
    def test_each_facet_equals_the_declared_constant(self) -> None:
        for name, changed in FACET_CHANGED.items():
            declared = fg.PHASE332_WINDOW_RUNG_FACETS[name]
            assert tuple(sorted(declared["changed"])) == changed, name
            assert declared["required_seasons"] == REQUIRED_SEASONS[name], name
        assert tuple(fg.PHASE332_WINDOW_RUNG_FACETS["A_team_form"]["added"]) == (
            FACET_A_ADDED
        )
        assert tuple(fg.PHASE332_WINDOW_RUNG_FACETS["C_snaps_injury"]["added"]) == (
            FACET_C_ADDED
        )
        assert fg.PHASE332_WINDOW_RUNG_FACETS["B_opponent_adjusted"]["added"] == ()

    def test_the_facets_are_disjoint_and_sized(self) -> None:
        assert len(FACET_A_CHANGED) == 28
        assert len(FACET_B_CHANGED) == 16
        assert len(FACET_C_CHANGED) == 32
        union = set(FACET_A_CHANGED) | set(FACET_B_CHANGED) | set(FACET_C_CHANGED)
        assert len(union) == 28 + 16 + 32
        assert not union & set(PREDICTED_UNMOVED)
        assert tuple(fg.PHASE332_WINDOW_RUNG_PREDICTED_UNMOVED) == PREDICTED_UNMOVED

    def test_the_widths_grow_by_the_four_added_flags(self) -> None:
        assert fg.PHASE332_WINDOW_RUNG_WIDTH_DELTA == 4
        assert len(FACET_A_ADDED) + len(FACET_C_ADDED) == 4
        assert fg.PHASE332_WINDOW_RUNG_WIDTHS_BEFORE == PREDICTED_WIDTHS_BEFORE
        assert fg.PHASE332_WINDOW_RUNG_WIDTHS_AFTER == PREDICTED_WIDTHS_AFTER
        assert all(
            after - before == 4
            for before, after in zip(
                PREDICTED_WIDTHS_BEFORE, PREDICTED_WIDTHS_AFTER, strict=True
            )
        )

    def test_the_opponent_adjusted_pool_now_starts_at_2002(self) -> None:
        """The widening Plan 33.2-17 handed forward BY NAME, through the identity binding."""
        assert min(TeamFormCalculator().per_game_seasons(range(2002, 2026))) == 2002


class TestTheJudge:
    def test_the_predicted_diff_is_attributed(self) -> None:
        verdict = _verdict(_complete_diff())
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]

    @pytest.mark.parametrize(
        "column",
        [
            "home_elo_rating",
            "temp_f",
            "home_qb_adjustment",
            "home_def_rolling_cpoe",
        ],
    )
    def test_a_column_outside_every_facet_is_unattributed(self, column: str) -> None:
        diff = _complete_diff() | {column: ["2010"]}
        verdict = _verdict(diff)
        assert not verdict["ok"]
        assert any(column in failure for failure in verdict["failures"])

    @pytest.mark.parametrize("facet", sorted(REQUIRED_SEASONS))
    def test_a_facet_missing_its_required_seasons_fails(self, facet: str) -> None:
        diff = _complete_diff()
        column = FACET_CHANGED[facet][0]
        diff[column] = ["2024"]
        verdict = _verdict(diff)
        assert not verdict["ok"]
        assert any(facet in failure for failure in verdict["failures"])

    def test_a_missing_added_flag_blocks(self) -> None:
        verdict = _verdict(
            _complete_diff(),
            added=FACET_A_ADDED + FACET_C_ADDED[:1],
            width_delta=3,
        )
        assert verdict["blocking"]

    def test_an_unexpected_added_column_blocks(self) -> None:
        verdict = _verdict(
            _complete_diff(),
            added=(*FACET_A_ADDED, *FACET_C_ADDED, "home_team_form_coverage"),
            width_delta=5,
        )
        assert verdict["blocking"]

    def test_a_removed_column_blocks(self) -> None:
        verdict = _verdict(_complete_diff(), removed=("temp_f",), width_delta=3)
        assert verdict["blocking"]

    def test_an_empty_diff_is_a_finding(self) -> None:
        verdict = _verdict({})
        assert not verdict["ok"]
        assert any("FINDING" in failure for failure in verdict["failures"])


needs_ladder = pytest.mark.skipif(
    not (STEP7B.is_file() and RETAKEN.is_file() and RUNG8.is_file()),
    reason=(
        "the p332_ step-7b, retaken-baseline or rung-8 fingerprint documents are absent "
        "(outputs/ is gitignored runtime state)"
    ),
)


@needs_ladder
class TestTheLiveRung:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(RETAKEN.read_text(encoding="utf-8")),
            json.loads(RUNG8.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_WINDOW_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_rung_ran_and_the_attribution_is_clean(self) -> None:
        import tests.phase33_state as state

        verdict = self._verdict()
        assert verdict["non_clock_moves"], (
            "EXPECTED_CASE is 'runs'; an empty diff halts"
        )
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert verdict["non_clock_moves"] == sorted(state.P332_18_RUNG8_MOVED_COLUMNS)

    @pytest.mark.parametrize("facet", sorted(FACET_CHANGED))
    def test_each_facet_moved_inside_its_own_prediction(self, facet: str) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        moved = fg._phase332_moved_seasons(report)
        facet_moved = {c: s for c, s in moved.items() if c in set(FACET_CHANGED[facet])}
        assert facet_moved, f"facet {facet} moved nothing"
        seasons = {s for values in facet_moved.values() for s in values}
        assert set(REQUIRED_SEASONS[facet]) <= seasons, facet
        assert seasons <= {str(s) for s in range(2002, 2026)}, facet

    def test_nothing_outside_the_three_facets_moved(self) -> None:
        before, after = self._documents()
        moved = set(fg._phase332_moved_seasons(compare_fingerprints(before, after)))
        union = set(FACET_A_CHANGED) | set(FACET_B_CHANGED) | set(FACET_C_CHANGED)
        assert sorted(moved - union) == []
        assert sorted(moved & set(PREDICTED_UNMOVED)) == []

    def test_the_widths_grew_by_the_four_flags_and_rows_held(self) -> None:
        before, after = self._documents()
        assert tuple(before[m]["width"] for m in fg.GOLD_MATRICES) == (
            PREDICTED_WIDTHS_BEFORE
        )
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == (
            PREDICTED_WIDTHS_AFTER
        )
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["rows"] == after[matrix]["rows"] == PREDICTED_ROWS
            added = sorted(
                set(after[matrix]["columns"]) - set(before[matrix]["columns"])
            )
            assert added == sorted(FACET_A_ADDED + FACET_C_ADDED), matrix

    def test_the_carry_in_is_inside_its_declared_prediction(self) -> None:
        import tests.phase33_state as state

        carry_in = fg.phase332_rung8_carry_in()
        assert carry_in["outside_prediction"] == []
        assert carry_in["outside_seasons"] == []
        assert sorted(carry_in["moved_seasons"]) == sorted(  # type: ignore[arg-type]
            state.P332_18_RUNG8_CARRY_IN_COLUMNS
        )
        for matrix in fg.GOLD_MATRICES:
            structure = carry_in["structure"][matrix]  # type: ignore[index]
            assert structure["added"] == [] and structure["removed"] == []
            assert structure["rows"][0] == structure["rows"][1]

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        import tests.phase33_state as state

        digest = hashlib.sha256(
            fg.PHASE332_WINDOW_RUNG_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_18_RUNG8_CAUSE_DIGEST

    def test_the_committed_diff_records_the_rung_and_its_carry_in(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"]["8"]
        assert rung["cause"] == fg.PHASE332_WINDOW_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_window"
        assert rung["expected_case"] == EXPECTED_CASE
        assert rung["baseline_document"] == RETAKEN.name
        assert rung["rebuilt"] is True
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert tuple(int(w) for w in rung["widths_before"]) == PREDICTED_WIDTHS_BEFORE
        assert tuple(int(w) for w in rung["widths_after"]) == PREDICTED_WIDTHS_AFTER
        assert sorted(rung["added_columns"]) == sorted(FACET_A_ADDED + FACET_C_ADDED)
        assert rung["carry_in"]["outside_prediction"] == []
        assert rung["carry_in"]["outside_seasons"] == []
        assert rung["carry_in"]["from_document"] == STEP7B.name
        assert sorted(rung["facets"]) == sorted(FACET_CHANGED)


@pytest.mark.skipif(
    not all((GOLD / f"features_{t}.parquet").is_file() for t in ("wp", "ats", "ou")),
    reason="production gold is absent (data/ is gitignored)",
)
class TestTheAddedFlagsSurviveIntoGold:
    """THE GOLD-SURVIVAL GUARD: the four added flags PRESENT in all three matrices, as levels."""

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_every_added_flag_is_present_and_a_level(self, target: str) -> None:
        frame = pd.read_parquet(GOLD / f"features_{target}.parquet")
        flags = list(FACET_A_ADDED + FACET_C_ADDED)
        missing = [name for name in flags if name not in frame.columns]
        assert missing == [], f"features_{target} lacks {missing}"
        values = frame[flags]
        assert values.isin([0.0, 1.0]).all().all()
        assert (values == 1.0).any().all(), "every flag must be set somewhere"
        assert (values == 0.0).any().all(), "every flag must be unset somewhere"

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_cpoe_before_2006_is_nan_beside_an_unset_flag(self, target: str) -> None:
        """2002-2005 carry no completion probability: NaN beside 0.0, never the neutral 0.0.

        A set flag always carries a value. (A within-season gap in a covered season may still
        be filled point-in-time beside a 0.0 flag -- step 7b's rule -- so the converse is not
        asserted.)
        """
        frame = pd.read_parquet(GOLD / f"features_{target}.parquet")
        early = frame["season"].between(2002, 2005)
        for side in ("home", "away"):
            value = frame[f"{side}_off_rolling_cpoe"]
            flag = frame[f"{side}_off_rolling_cpoe_coverage"]
            assert value[flag == 1.0].notna().all(), (target, side)
            assert (flag[early] == 0.0).all(), (target, side)
            assert value[early].isna().all(), (target, side)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the rung-8 digest bracket documents are gitignored runtime state",
)
class TestTheDigestBracket:
    def test_only_the_declared_paths_moved(self) -> None:
        from tests.data_boundary import diff_digests

        before = json.loads(DIGEST_BEFORE.read_text(encoding="utf-8"))
        after = json.loads(DIGEST_AFTER.read_text(encoding="utf-8"))
        diff = diff_digests(before, after)
        moved = set(diff["added"]) | set(diff["removed"]) | set(diff["changed"])
        assert sorted(moved - DECLARED_PATHS) == []
        assert not diff.get("mixed")
        assert diff["removed"] == []
