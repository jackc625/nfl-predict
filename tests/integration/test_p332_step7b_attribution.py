"""Extra step 7b of the Phase-33.2 gold ladder: within-season imputation made point-in-time.

Owner ruling 2026-09-22 (deferred-items.md: "fill a mid-season gap from EARLIER games only"),
run first in Plan 33.2-17's dispatch. ONE cause: the gold imputer fills a gap only from games
that had ENDED by that gap's own lock -- the team's pre-lock mean, then the league's pre-lock
mean that season, then the strictly-prior-seasons median (never a self-fit) -- and where nothing
known at the lock exists the cell stays blank through normalization. NOT inside rung 7 or rung
8: it follows rung 7 and rung 8 is judged against it (D33.2-20).

THE PREDICTION, declared before the rebuild (the constants below, equal to
``scripts.fingerprint_gold``'s ``PHASE332_IMPUTATION_STEP_*``), measured read-only on the exact
frame the imputer receives:

* only the 50 imputed columns can move -- 28 team-form (from 2020), 20 snap (from 2013) and the
  two Elo momentum columns (from 2002) -- each from its first imputed season through 2025;
* each gains exactly its declared blank cells (16; 32 for each snap_continuity column);
* nothing added or removed; widths 197/198/197 and rows 6,499 unchanged.

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
    PHASE332_IMPUTATION_STEP,
    PHASE332_OPPADJ_RUNG,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG7 = rung_document_path(FINGERPRINT_DIR, PHASE332_OPPADJ_RUNG, PHASE332_RUNG_PREFIX)
STEP7B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_IMPUTATION_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step7b_before.json")
DIGEST_AFTER = Path("outputs/p332_step7b_after.json")

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded BEFORE the rebuild.
# ---------------------------------------------------------------------------

_TEAM_FORM_METRICS = {
    "def": (
        "pass_success_rate",
        "red_zone_td_rate",
        "rush_success_rate",
        "success_rate",
        "third_down_conversion_rate",
    ),
    "off": (
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
}
PREDICTED_TEAM_FORM = tuple(
    f"{side}_{unit}_rolling_{metric}"
    for side in ("home", "away")
    for unit, metrics in _TEAM_FORM_METRICS.items()
    for metric in metrics
)
PREDICTED_SNAP = tuple(
    f"{side}_{name}"
    for side in ("home", "away")
    for name in (
        *(
            f"rolling_snap_share_{p}"
            for p in ("db", "dl", "lb", "ol", "qb", "rb", "te", "wr")
        ),
        "snap_concentration",
        "snap_continuity",
    )
)
PREDICTED_ELO = ("home_elo_momentum", "away_elo_momentum")
PREDICTED_FIRST_SEASON: dict[str, int] = (
    dict.fromkeys(PREDICTED_TEAM_FORM, 2020)
    | dict.fromkeys(PREDICTED_SNAP, 2013)
    | dict.fromkeys(PREDICTED_ELO, 2002)
)
PREDICTED_BLANKS: dict[str, int] = {
    c: (32 if c.endswith("snap_continuity") else 16) for c in PREDICTED_FIRST_SEASON
}
PREDICTED_WIDTHS: tuple[int, int, int] = (197, 198, 197)

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)


def _report(
    changed: dict[str, list[str]], nulls: dict[str, tuple[int, int]] | None = None
) -> dict:
    nulls = nulls or {}
    matrix = {
        "width_before": 197,
        "width_after": 197,
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
                "null_count_before": nulls.get(
                    column, (0, PREDICTED_BLANKS.get(column, 0))
                )[0],
                "null_count_after": nulls.get(
                    column, (0, PREDICTED_BLANKS.get(column, 0))
                )[1],
                "discrete_indicator_before": False,
                "discrete_indicator_after": False,
            }
            for column, seasons in changed.items()
        },
    }
    return {name: dict(matrix) for name in fg.GOLD_MATRICES}


def _all_declared(**overrides) -> dict[str, list[str]]:
    changed = {c: [str(s)] for c, s in PREDICTED_FIRST_SEASON.items()}
    changed.update(overrides)
    return changed


def _verdict(changed: dict[str, list[str]], nulls=None) -> dict:
    return attribute_rung(
        _report(changed, nulls),
        PHASE332_IMPUTATION_STEP,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_is_an_extra_step_following_rung_seven(self) -> None:
        assert PHASE332_IMPUTATION_STEP == "7b"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["7b"] == 7
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["7b"]
            == fg.PHASE332_IMPUTATION_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["7b"]
            is fg.PHASE332_IMPUTATION_STEP_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["7b"] is fg._attribute_p332_imputation

    def test_it_is_judged_against_rung_seven_and_rung_eight_against_it(
        self, tmp_path
    ) -> None:
        assert phase332_baseline_document_path(tmp_path, "7b").name == "p332_rung7.json"
        assert fg._ladder_predecessors(8, PHASE332_RUNG_PREFIX)[-1] == "7b"

    def test_its_id_cannot_collide_with_rungs_eight_and_nine(self) -> None:
        assert PHASE332_IMPUTATION_STEP not in (8, 9, "8", "9")
        assert rung_document_path(FINGERPRINT_DIR, "7b", PHASE332_RUNG_PREFIX).name == (
            "p332_rung7b.json"
        )

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_IMPUTATION_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"


class TestThePrediction:
    def test_the_fifty_columns_and_their_first_seasons(self) -> None:
        assert (
            PREDICTED_FIRST_SEASON == fg.PHASE332_IMPUTATION_STEP_FIRST_SEASON_BY_COLUMN
        )
        assert len(PREDICTED_FIRST_SEASON) == 50
        assert (
            tuple(sorted(PREDICTED_FIRST_SEASON)) == fg.PHASE332_IMPUTATION_STEP_COLUMNS
        )

    def test_the_blank_cells(self) -> None:
        assert PREDICTED_BLANKS == fg.PHASE332_IMPUTATION_STEP_BLANK_CELLS_BY_COLUMN
        assert sum(PREDICTED_BLANKS.values()) == 832

    def test_no_opponent_adjusted_value_is_declared(self) -> None:
        # Plan 33.2-16 keeps the twelve *_rolling_opp_adj_* values out of the imputer.
        assert not any("opp_adj" in c for c in PREDICTED_FIRST_SEASON)


class TestTheJudge:
    def test_the_declared_moves_are_attributed(self) -> None:
        verdict = _verdict(_all_declared(home_off_rolling_cpoe=["2020", "2025"]))
        assert verdict["ok"], verdict["failures"]

    def test_a_move_before_the_first_imputed_season_is_unattributed(self) -> None:
        assert not _verdict(_all_declared(home_snap_concentration=["2012", "2013"]))[
            "ok"
        ]

    def test_a_column_the_imputer_never_fills_is_unattributed(self) -> None:
        for column in (
            "home_off_rolling_opp_adj_epa_per_play",
            "temp_f",
            "home_injury_coverage",
        ):
            assert not _verdict(_all_declared(**{column: ["2021"]}))["ok"], column

    def test_a_blank_count_off_by_one_fails(self) -> None:
        nulls = {"home_elo_momentum": (0, 17)}
        assert not _verdict(_all_declared(), nulls)["ok"]

    def test_a_declared_blank_column_that_did_not_move_fails(self) -> None:
        changed = _all_declared()
        del changed["away_snap_continuity"]
        assert not _verdict(changed)["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG7.is_file() and STEP7B.is_file()),
    reason="the p332_ rung-7 or step-7b fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(RUNG7.read_text(encoding="utf-8")),
            json.loads(STEP7B.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_IMPUTATION_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= set(PREDICTED_FIRST_SEASON)
        assert verdict["non_clock_moves"] == sorted(state.P332_17_STEP7B_MOVED_COLUMNS)

    def test_the_widths_and_rows_did_not_move(self) -> None:
        before, after = self._documents()
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["width"] == after[matrix]["width"]
            assert before[matrix]["rows"] == after[matrix]["rows"]
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == PREDICTED_WIDTHS

    def test_every_declared_blank_is_blank_in_every_matrix(self) -> None:
        before, after = self._documents()
        report = compare_fingerprints(before, after)
        for matrix in fg.GOLD_MATRICES:
            details = report[matrix]["column_details"]
            for column, blanks in PREDICTED_BLANKS.items():
                grown = (
                    details[column]["null_count_after"]
                    - details[column]["null_count_before"]
                )
                assert grown == blanks, (matrix, column, grown)

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_IMPUTATION_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_17_STEP7B_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["7b"]
        assert step["cause"] == fg.PHASE332_IMPUTATION_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_imputation"
        assert step["baseline_document"] == RUNG7.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]
        assert step["declared_blank_cells"] == step["measured_blank_cells_wp"]


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-7b digest bracket documents are gitignored runtime state",
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
