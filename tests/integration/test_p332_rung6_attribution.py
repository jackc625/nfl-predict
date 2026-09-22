"""Rung 6 of the Phase-33.2 gold ladder: the snap and injury feeds wired and populated.

Plan 33.2-15 Task 4 (D33.2-16, D33.2-20). ONE cause: both ingests run for 2025 and 2026, every
row validated through a real silver schema and stamped with its release asset's publication
time as capture provenance, and the median fill for an EMPTY family replaced by NaN.

THE PREDICTION, declared before the rebuild from the prospective-stamp ruling (the constants
below, equal to ``scripts.fingerprint_gold.PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER``):

* snaps -- the 20 snap columns MOVE, in 2025 only: real 2025 snaps replace the last eight 2024
  games every 2025 game's window used to read. 2026 is not in gold (owner ruling: every rung
  rebuilds ``--through-season 2025``).
* injury, qb -- EMPTY: every 2025 row the ingest wrote carries the asset's 2026-09-07 stamp,
  after every 2025 lock, so nothing is admitted; an injury movement would mean a post-lock
  stamp was admitted.
* no pre-2025 row moves and the widths do not change.

What is asserted: the rung is registered AND dispatched (no new prefix branch); the cause names
the one change; the judge refuses a pre-2025 snap move, any injury / QB move and any column of
another builder; and, live against the gitignored ladder, the attribution is clean, exactly the
declared family moved in exactly 2025, the digest bracket named only declared paths, and the
committed diff and the state witness record it.

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
    PHASE332_FEED_RUNG,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
RUNG5 = rung_document_path(FINGERPRINT_DIR, PHASE332_CUTOFF_RUNG, PHASE332_RUNG_PREFIX)
RUNG6 = rung_document_path(FINGERPRINT_DIR, PHASE332_FEED_RUNG, PHASE332_RUNG_PREFIX)
DIGEST_BEFORE = Path("outputs/p332_rung6_before.json")
DIGEST_AFTER = Path("outputs/p332_rung6_after.json")

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded here BEFORE the rebuild (Plan 33.2-15 Task 4).
# ---------------------------------------------------------------------------

#: The 20 snap columns (home/away x continuity, concentration and eight rolling shares).
PREDICTED_SNAP_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        f"{side}_{column}"
        for side in ("home", "away")
        for column in (
            "snap_continuity",
            "snap_concentration",
            *(
                f"rolling_snap_share_{g}"
                for g in ("db", "dl", "lb", "ol", "qb", "rb", "te", "wr")
            ),
        )
    )
)

#: EMPTY, and recorded rather than omitted: an injury or QB move is a named finding.
PREDICTED_INJURY_QB_COLUMNS: tuple[str, ...] = ()

#: The only season that can move: 2026 is not in gold under the --through-season 2025 ruling.
PREDICTED_MOVED_SEASONS: tuple[str, ...] = ("2025",)

#: Widths unchanged (rung 5's).
PREDICTED_WIDTHS: tuple[int, int, int] = (193, 194, 193)

#: The declared write set of the rung, relative to data/. The plan text named the bronze
#: PREFIXES as bronze/snap_counts/ and bronze/injuries/; data.storage.save_bronze_snapshot
#: actually writes flat <table>_raw_bronze_<season>_W<week>_<stamp>.parquet files under bronze/,
#: so the prefixes are declared as the names it writes (a plan-text correction, recorded).
DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "silver/snap_counts.parquet",
        "silver/injuries.parquet",
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)
DECLARED_BRONZE_PREFIXES: tuple[str, ...] = (
    "bronze/snap_counts_raw_bronze_",
    "bronze/injuries_raw_bronze_",
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


def _verdict(changed: dict[str, list[str]], width: int = 193) -> dict:
    return attribute_rung(
        _report(changed, width), PHASE332_FEED_RUNG, rung_prefix=PHASE332_RUNG_PREFIX
    )


class TestTheRungIsRegisteredAndDispatched:
    def test_rung_six_is_in_every_table(self) -> None:
        assert PHASE332_FEED_RUNG == 6
        assert (
            fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][6]
            == fg.PHASE332_FEED_RUNG_CAUSE
        )
        assert (
            fg.PHASE332_RUNG_SIGNATURES[6] is fg.PHASE332_FEED_RUNG_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_RUNG_ATTRIBUTORS[6] is fg._attribute_p332_feed

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_the_signature_was_declared_before_the_rebuild(self) -> None:
        signature = fg._expected_signature(6, prefix=PHASE332_RUNG_PREFIX)
        assert signature == fg.PHASE332_FEED_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert signature["width"] == "unchanged"

    def test_rung_six_is_judged_against_rung_five_with_no_retake(
        self, tmp_path
    ) -> None:
        assert fg._ladder_predecessors(6, PHASE332_RUNG_PREFIX)[-1] == 5
        assert phase332_baseline_document_path(tmp_path, 6).name == "p332_rung5.json"
        assert 6 not in fg.PHASE332_RETAKEN_BASELINES


class TestTheCauseSaysExactlyWhatChanged:
    def test_it_names_the_one_change(self) -> None:
        cause = fg.PHASE332_FEED_RUNG_CAUSE.lower()
        for phrase in (
            "snap and injury feeds wired",
            "snapcountschema",
            "injuryschema",
            "validate_bronze_to_silver",
            "upstream_captured_at",
            "capture provenance",
            "empty",
            "nan",
        ):
            assert phrase in cause, phrase

    def test_the_limitation_and_the_espn_ruling_are_recorded_together(self) -> None:
        signature = fg.PHASE332_FEED_RUNG_EXPECTED_SIGNATURE
        assert "08:38 ET" in signature["injury_freshness_limitation"]
        assert "ESPN" in signature["espn_ruling"]
        assert "not adopted" in signature["espn_ruling"].lower()


class TestThePredictionIsRecordedBeforeTheRebuild:
    def test_the_snap_set_is_the_builders_own_declaration(self) -> None:
        assert len(PREDICTED_SNAP_COLUMNS) == 20
        assert (
            fg.PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER["snaps"]
            == PREDICTED_SNAP_COLUMNS
        )
        assert (
            set(PREDICTED_SNAP_COLUMNS) == fg.phase332_feed_builder_columns()["snaps"]
        )

    def test_the_injury_and_qb_subsets_are_recorded_empty(self) -> None:
        assert fg.PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER["injury"] == ()
        assert fg.PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER["qb"] == ()
        assert PREDICTED_INJURY_QB_COLUMNS == ()

    def test_the_moved_seasons_are_2025_only(self) -> None:
        assert PREDICTED_MOVED_SEASONS == fg.PHASE332_FEED_RUNG_PREDICTED_SEASONS
        assert set(PREDICTED_MOVED_SEASONS) <= set(
            fg.PHASE332_FEED_RUNG_ALLOWED_SEASONS
        )


class TestTheJudge:
    def test_the_snap_family_moving_in_2025_is_attributed(self) -> None:
        verdict = _verdict({column: ["2025"] for column in PREDICTED_SNAP_COLUMNS})
        assert verdict["ok"], verdict["failures"]

    def test_a_snap_column_moving_before_2025_is_unattributed(self) -> None:
        verdict = _verdict({"home_snap_continuity": ["2024", "2025"]})
        assert not verdict["ok"]
        assert any("SECOND cause" in failure for failure in verdict["failures"])

    @pytest.mark.parametrize(
        "column",
        ["home_qb_out_flag", "away_availability_coverage", "home_qb_adjustment"],
    )
    def test_any_injury_or_qb_move_is_an_admitted_post_lock_stamp(self, column) -> None:
        verdict = _verdict({column: ["2025"]})
        assert not verdict["ok"]
        assert any("ADMITTED" in failure for failure in verdict["failures"])

    @pytest.mark.parametrize("column", ["temp_f", "home_rest_days", "snapshot_spread"])
    def test_a_column_of_another_builder_is_unattributed(self, column) -> None:
        verdict = _verdict({column: ["2025"]})
        assert not verdict["ok"]
        assert any("builder:" in failure for failure in verdict["failures"])

    def test_an_empty_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (RUNG5.is_file() and RUNG6.is_file()),
    reason=(
        "the p332_ rung-5 or rung-6 fingerprint documents are absent from "
        "outputs/fingerprints -- outputs/ is gitignored runtime state"
    ),
)


@needs_ladder
class TestTheLiveRung:
    @staticmethod
    def _verdict() -> dict:
        before = json.loads(RUNG5.read_text(encoding="utf-8"))
        after = json.loads(RUNG6.read_text(encoding="utf-8"))
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_FEED_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= set(PREDICTED_SNAP_COLUMNS)
        assert verdict["non_clock_moves"] == sorted(state.P332_15_RUNG6_MOVED_COLUMNS)

    def test_no_injury_or_qb_column_moved(self) -> None:
        split = fg.phase332_feed_moved_by_builder(self._verdict()["non_clock_moves"])
        assert split["injury"] == [] and split["qb"] == [] and split["unmapped"] == []

    def test_every_moved_column_moved_in_2025_only(self) -> None:
        before = json.loads(RUNG5.read_text(encoding="utf-8"))
        after = json.loads(RUNG6.read_text(encoding="utf-8"))
        seasons = fg._phase332_moved_seasons(compare_fingerprints(before, after))
        moved = {s for column, values in seasons.items() for s in values}
        assert moved == set(PREDICTED_MOVED_SEASONS)

    def test_the_widths_and_rows_did_not_move(self) -> None:
        before = json.loads(RUNG5.read_text(encoding="utf-8"))
        after = json.loads(RUNG6.read_text(encoding="utf-8"))
        for matrix in fg.GOLD_MATRICES:
            assert before[matrix]["width"] == after[matrix]["width"]
            assert before[matrix]["rows"] == after[matrix]["rows"]
        assert tuple(after[m]["width"] for m in fg.GOLD_MATRICES) == PREDICTED_WIDTHS

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        digest = hashlib.sha256(fg.PHASE332_FEED_RUNG_CAUSE.encode("utf-8")).hexdigest()
        assert digest == state.P332_15_RUNG6_CAUSE_DIGEST

    def test_the_committed_diff_records_the_rung(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"]["6"]
        assert rung["cause"] == fg.PHASE332_FEED_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_feed"
        assert rung["baseline_document"] == RUNG5.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        assert rung["widths_before"] == rung["widths_after"]
        assert rung["moved_season_union"] == list(PREDICTED_MOVED_SEASONS)
        assert rung["moved_by_builder"]["injury"] == []
        assert rung["moved_by_builder"]["qb"] == []
        assert "08:38 ET" in rung["injury_freshness_limitation"]
        assert "ESPN" in rung["espn_ruling"]


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the rung-6 digest bracket documents are gitignored runtime state",
)
class TestTheDigestBracket:
    def test_only_declared_paths_moved_and_both_bronze_prefixes_gained_a_file(
        self,
    ) -> None:
        from tests.data_boundary import diff_digests

        before = json.loads(DIGEST_BEFORE.read_text(encoding="utf-8"))
        after = json.loads(DIGEST_AFTER.read_text(encoding="utf-8"))
        diff = diff_digests(before, after)
        moved = set(diff["added"]) | set(diff["removed"]) | set(diff["changed"])
        undeclared = sorted(
            key
            for key in moved - DECLARED_PATHS
            if not key.startswith(DECLARED_BRONZE_PREFIXES)
        )
        assert undeclared == []
        assert not diff.get("mixed")
        assert diff["removed"] == []
        for prefix in DECLARED_BRONZE_PREFIXES:
            assert any(key.startswith(prefix) for key in diff["added"]), prefix
