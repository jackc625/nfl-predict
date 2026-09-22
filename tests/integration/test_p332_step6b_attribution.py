"""Extra step 6b of the Phase-33.2 gold ladder: postseason injury reports ingested and timed.

Plan 33.2-15 (the question Plan 33.2-13 routed to it). ONE cause: the injury ingest no longer
drops non-REG game types, and the 2009-2025 postseason reports upstream publishes are added to
silver injuries, each admitted for its game only when its own ``date_modified`` is at or before
that game's lock. NOT inside rung 6: it moves pre-2025 injury values, which rung 6 predicted it
would not (D33.2-20).

THE PREDICTION, declared before the rebuild (the constants below, equal to
``scripts.fingerprint_gold``'s ``PHASE332_POSTSEASON_INJURY_STEP_*``):

* ten injury columns can move -- home/away of qb_out_flag, backup_quality_delta,
  availability_fraction, injury_coverage and date_modified_coverage. availability_coverage
  cannot (it reports whether prior snap shares exist);
* the two coverage flags only on 2010-2024 postseason games (level-preserved, never
  winsorized); the three value columns within 2010-2025 (z-scored within season, bootstrapped
  from the prior season, winsorized on strictly-prior seasons) -- never before 2010;
* every QB, snap and other column EMPTY; widths unchanged.

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
    PHASE332_FEED_RUNG,
    PHASE332_POSTSEASON_INJURY_STEP,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG6 = rung_document_path(FINGERPRINT_DIR, PHASE332_FEED_RUNG, PHASE332_RUNG_PREFIX)
STEP6B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_POSTSEASON_INJURY_STEP, PHASE332_RUNG_PREFIX
)
DIGEST_BEFORE = Path("outputs/p332_step6b_before.json")
DIGEST_AFTER = Path("outputs/p332_step6b_after.json")

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded BEFORE the rebuild.
# ---------------------------------------------------------------------------

PREDICTED_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        f"{side}_{column}"
        for side in ("home", "away")
        for column in (
            "qb_out_flag",
            "backup_quality_delta",
            "availability_fraction",
            "injury_coverage",
            "date_modified_coverage",
        )
    )
)
PREDICTED_FLAG_SEASONS: tuple[str, ...] = tuple(str(s) for s in range(2010, 2025))
PREDICTED_VALUE_SEASONS: tuple[str, ...] = tuple(str(s) for s in range(2010, 2026))
PREDICTED_WIDTHS: tuple[int, int, int] = (193, 194, 193)

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "silver/injuries.parquet",
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)
DECLARED_BRONZE_PREFIX = "bronze/injuries_raw_bronze_"


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
        _report(changed),
        PHASE332_POSTSEASON_INJURY_STEP,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheStepIsRegisteredAndDispatched:
    def test_it_is_an_extra_step_following_rung_six(self) -> None:
        assert PHASE332_POSTSEASON_INJURY_STEP == "6b"
        assert fg.EXTRA_STEPS_BY_PREFIX[PHASE332_RUNG_PREFIX]["6b"] == 6
        assert (
            fg.EXTRA_STEP_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]["6b"]
            == fg.PHASE332_POSTSEASON_INJURY_STEP_CAUSE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_SIGNATURES["6b"]
            is fg.PHASE332_POSTSEASON_INJURY_STEP_EXPECTED_SIGNATURE
        )
        assert (
            fg.PHASE332_EXTRA_STEP_ATTRIBUTORS["6b"]
            is fg._attribute_p332_postseason_injury
        )

    def test_it_is_judged_against_rung_six_and_rung_seven_against_it(
        self, tmp_path
    ) -> None:
        assert phase332_baseline_document_path(tmp_path, "6b").name == "p332_rung6.json"
        assert fg._ladder_predecessors(7, PHASE332_RUNG_PREFIX)[-1] == "6b"

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_POSTSEASON_INJURY_STEP_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"


class TestThePrediction:
    def test_the_ten_columns_and_not_availability_coverage(self) -> None:
        assert PREDICTED_COLUMNS == fg.PHASE332_POSTSEASON_INJURY_STEP_COLUMNS
        assert len(PREDICTED_COLUMNS) == 10
        assert "home_availability_coverage" not in PREDICTED_COLUMNS
        injury = set(fg.PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS["injury"])
        assert set(PREDICTED_COLUMNS) <= injury

    def test_the_seasons(self) -> None:
        assert PREDICTED_FLAG_SEASONS == fg.PHASE332_POSTSEASON_INJURY_STEP_FLAG_SEASONS
        assert (
            PREDICTED_VALUE_SEASONS == fg.PHASE332_POSTSEASON_INJURY_STEP_VALUE_SEASONS
        )


class TestTheJudge:
    def test_flags_in_2010_2024_and_values_through_2025_are_attributed(self) -> None:
        verdict = _verdict(
            {
                "home_injury_coverage": ["2010", "2024"],
                "home_qb_out_flag": ["2012", "2025"],
            }
        )
        assert verdict["ok"], verdict["failures"]

    def test_a_flag_moving_in_2025_is_unattributed(self) -> None:
        assert not _verdict({"home_injury_coverage": ["2025"]})["ok"]

    def test_anything_before_2010_is_unattributed(self) -> None:
        assert not _verdict({"away_availability_fraction": ["2009", "2012"]})["ok"]

    @pytest.mark.parametrize(
        "column",
        [
            "home_availability_coverage",
            "home_qb_adjustment",
            "home_snap_continuity",
            "temp_f",
        ],
    )
    def test_any_other_column_is_unattributed(self, column) -> None:
        assert not _verdict({column: ["2015"]})["ok"]

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG6.is_file() and STEP6B.is_file()),
    reason="the p332_ rung-6 or step-6b fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveStep:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(RUNG6.read_text(encoding="utf-8")),
            json.loads(STEP6B.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_POSTSEASON_INJURY_STEP,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= set(PREDICTED_COLUMNS)
        assert verdict["non_clock_moves"] == sorted(state.P332_15_STEP6B_MOVED_COLUMNS)

    def test_the_coverage_flag_moved(self) -> None:
        assert "home_injury_coverage" in self._verdict()["non_clock_moves"]

    def test_the_widths_and_rows_did_not_move(self) -> None:
        before, after = self._documents()
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["width"] == after[matrix]["width"]
            assert before[matrix]["rows"] == after[matrix]["rows"]
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == PREDICTED_WIDTHS

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(
            fg.PHASE332_POSTSEASON_INJURY_STEP_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_15_STEP6B_CAUSE_DIGEST

    def test_the_committed_diff_records_the_step(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        step = diff["rung"]["6b"]
        assert step["cause"] == fg.PHASE332_POSTSEASON_INJURY_STEP_CAUSE
        assert step["attributor"] == "_attribute_p332_postseason_injury"
        assert step["baseline_document"] == RUNG6.name
        assert step["attribution_ok"] is True
        assert step["unattributed_columns"] == []
        assert step["widths_before"] == step["widths_after"]


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the step-6b digest bracket documents are gitignored runtime state",
)
class TestTheDigestBracket:
    def test_only_declared_paths_moved_and_a_bronze_injury_file_appeared(self) -> None:
        from tests.data_boundary import diff_digests

        before = json.loads(DIGEST_BEFORE.read_text(encoding="utf-8"))
        after = json.loads(DIGEST_AFTER.read_text(encoding="utf-8"))
        diff = diff_digests(before, after)
        moved = set(diff["added"]) | set(diff["removed"]) | set(diff["changed"])
        undeclared = sorted(
            k
            for k in moved - DECLARED_PATHS
            if not k.startswith(DECLARED_BRONZE_PREFIX)
        )
        assert undeclared == []
        assert not diff.get("mixed")
        assert diff["removed"] == []
        assert any(k.startswith(DECLARED_BRONZE_PREFIX) for k in diff["added"])
