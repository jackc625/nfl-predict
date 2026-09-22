"""Rung 7 of the Phase-33.2 gold ladder: the opponent adjustment made to run, honestly.

Plan 33.2-16 Task 3 (SPEC R9 / R2, D33.2-08 item 1). ONE cause: opponents resolved through the
canonical game id mapping so the adjustment actually runs, every input admitted at the lock of
the game it informs (per-lock league averages replacing whole-frame means), and an explicit
coverage flag -- the value left NaN, never imputed -- replacing the silent fall-through to raw
EPA.

THE PREDICTION, declared before the rebuild (the constants below, equal to
``scripts.fingerprint_gold``'s ``PHASE332_OPPADJ_RUNG_*``):

* the twelve ``*_rolling_opp_adj_*`` value columns move, in any season 2002-2025 (new values
  2018-2025; the neutral 0.0 stand-in becomes the preserved NaN before the play-by-play pool);
* exactly four columns are ADDED to every matrix -- ``home_off_`` / ``home_def_`` /
  ``away_off_`` / ``away_def_rolling_opp_adj_coverage`` -- and they are asserted PRESENT by name
  in all three gold matrices (the gold-survival guard that replaces a ``data/schemas.py``
  declaration: the flag is a gold-level column that never passes a silver schema);
* nothing else moves, nothing is removed, rows are unchanged, and every width grows by four.

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
from features.opponent_adj import OPP_ADJ_COVERAGE_COLUMN
from scripts.fingerprint_gold import (
    FINGERPRINT_DIR,
    PHASE332_OPPADJ_RUNG,
    PHASE332_POSTSEASON_INJURY_STEP,
    PHASE332_RUNG_PREFIX,
    attribute_rung,
    compare_fingerprints,
    phase332_baseline_document_path,
    rung_document_path,
)

DIFF_TOML = Path("config/phase332_gold_rebuild_diff.toml")
STEP6B = rung_document_path(
    FINGERPRINT_DIR, PHASE332_POSTSEASON_INJURY_STEP, PHASE332_RUNG_PREFIX
)
RUNG7 = rung_document_path(FINGERPRINT_DIR, PHASE332_OPPADJ_RUNG, PHASE332_RUNG_PREFIX)
DIGEST_BEFORE = Path("outputs/p332_rung7_before.json")
DIGEST_AFTER = Path("outputs/p332_rung7_after.json")
GOLD = Path("data/gold")

# ---------------------------------------------------------------------------
# THE PREDICTION -- recorded BEFORE the rebuild.
# ---------------------------------------------------------------------------

PREDICTED_CHANGED: tuple[str, ...] = tuple(
    sorted(
        f"{prefix}_{side}_{name}"
        for prefix in ("home", "away")
        for side in ("off", "def")
        for name in (
            "rolling_opp_adj_epa_per_play",
            "rolling_opp_adj_pass_epa",
            "rolling_opp_adj_rush_epa",
        )
    )
)
PREDICTED_ADDED: tuple[str, ...] = tuple(
    sorted(
        f"{prefix}_{side}_{OPP_ADJ_COVERAGE_COLUMN}"
        for prefix in ("home", "away")
        for side in ("off", "def")
    )
)
PREDICTED_ALLOWED_SEASONS: tuple[str, ...] = tuple(str(s) for s in range(2002, 2026))
PREDICTED_WIDTHS_BEFORE: tuple[int, int, int] = (193, 194, 193)
PREDICTED_WIDTHS_AFTER: tuple[int, int, int] = (197, 198, 197)
PREDICTED_ROWS: int = 6499

DECLARED_PATHS: frozenset[str] = frozenset(
    {
        "gold/features_wp.parquet",
        "gold/features_ats.parquet",
        "gold/features_ou.parquet",
        "nfl_predictions.duckdb",
    }
)


def _report(
    changed: dict[str, list[str]],
    *,
    added: tuple[str, ...] = PREDICTED_ADDED,
    removed: tuple[str, ...] = (),
    width_delta: int = 4,
) -> dict:
    matrix = {
        "width_before": 193,
        "width_after": 193 + width_delta,
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


def _verdict(changed: dict[str, list[str]], **kwargs) -> dict:
    return attribute_rung(
        _report(changed, **kwargs),
        PHASE332_OPPADJ_RUNG,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )


class TestTheRungIsRegisteredAndDispatched:
    def test_it_is_rung_seven_with_one_cause(self) -> None:
        assert PHASE332_OPPADJ_RUNG == 7
        causes = fg.RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX]
        assert causes[7] == fg.PHASE332_OPPADJ_RUNG_CAUSE
        assert fg.PHASE332_OPPADJ_RUNG_CAUSE

    def test_both_dispatch_tables_carry_it(self) -> None:
        assert (
            fg.PHASE332_RUNG_SIGNATURES[7] is fg.PHASE332_OPPADJ_RUNG_EXPECTED_SIGNATURE
        )
        assert fg.PHASE332_RUNG_ATTRIBUTORS[7] is fg._attribute_p332_oppadj

    def test_it_is_judged_against_step_6b(self, tmp_path) -> None:
        assert phase332_baseline_document_path(tmp_path, 7).name == "p332_rung6b.json"

    def test_no_new_prefix_branch_was_added(self) -> None:
        source = Path(fg.__file__).read_text(encoding="utf-8")
        assert source.count("if prefix == PHASE332_RUNG_PREFIX:") == 1
        assert source.count("if rung_prefix == PHASE332_RUNG_PREFIX:") == 1

    def test_declared_before_the_rebuild(self) -> None:
        signature = fg.PHASE332_OPPADJ_RUNG_EXPECTED_SIGNATURE
        assert signature["declared_before_the_rebuild"] is True
        assert tuple(signature["columns_added"]) == PREDICTED_ADDED


class TestThePrediction:
    def test_the_twelve_value_columns(self) -> None:
        assert PREDICTED_CHANGED == fg.PHASE332_OPPADJ_RUNG_PREDICTED_CHANGED
        assert len(PREDICTED_CHANGED) == 12

    def test_the_four_flags_come_from_the_one_constant(self) -> None:
        assert PREDICTED_ADDED == fg.PHASE332_OPPADJ_RUNG_PREDICTED_ADDED
        assert len(PREDICTED_ADDED) == 4
        assert all(name.endswith(OPP_ADJ_COVERAGE_COLUMN) for name in PREDICTED_ADDED)

    def test_the_seasons_and_the_width_delta(self) -> None:
        assert PREDICTED_ALLOWED_SEASONS == fg.PHASE332_OPPADJ_RUNG_ALLOWED_SEASONS
        assert fg.PHASE332_OPPADJ_RUNG_WIDTH_DELTA == 4
        assert all(
            after - before == 4
            for before, after in zip(
                PREDICTED_WIDTHS_BEFORE, PREDICTED_WIDTHS_AFTER, strict=True
            )
        )


class TestTheJudge:
    def test_the_predicted_diff_is_attributed(self) -> None:
        verdict = _verdict(
            {
                "home_off_rolling_opp_adj_epa_per_play": ["2002", "2018", "2025"],
                "away_def_rolling_opp_adj_rush_epa": ["2019"],
            }
        )
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]

    def test_a_column_outside_the_family_is_unattributed(self) -> None:
        verdict = _verdict(
            {
                "home_off_rolling_opp_adj_epa_per_play": ["2018"],
                "home_off_rolling_epa_per_play": ["2018"],
            }
        )
        assert not verdict["ok"]

    def test_a_season_outside_2002_2025_is_unattributed(self) -> None:
        assert not _verdict({"home_off_rolling_opp_adj_pass_epa": ["2026"]})["ok"]

    def test_a_missing_flag_blocks(self) -> None:
        verdict = _verdict(
            {"home_off_rolling_opp_adj_epa_per_play": ["2018"]},
            added=PREDICTED_ADDED[:3],
            width_delta=3,
        )
        assert verdict["blocking"]

    def test_an_unexpected_added_column_blocks(self) -> None:
        verdict = _verdict(
            {"home_off_rolling_opp_adj_epa_per_play": ["2018"]},
            added=(*PREDICTED_ADDED, "home_off_opp_adj_coverage"),
            width_delta=5,
        )
        assert verdict["blocking"]

    def test_a_removed_column_blocks(self) -> None:
        verdict = _verdict(
            {"home_off_rolling_opp_adj_epa_per_play": ["2018"]},
            removed=("temp_f",),
            width_delta=3,
        )
        assert verdict["blocking"]

    def test_an_empty_value_diff_is_refused(self) -> None:
        assert not _verdict({})["ok"]


needs_ladder = pytest.mark.skipif(
    not (STEP6B.is_file() and RUNG7.is_file()),
    reason="the p332_ step-6b or rung-7 fingerprint documents are absent (gitignored)",
)


@needs_ladder
class TestTheLiveRung:
    @staticmethod
    def _documents() -> tuple[dict, dict]:
        return (
            json.loads(STEP6B.read_text(encoding="utf-8")),
            json.loads(RUNG7.read_text(encoding="utf-8")),
        )

    def _verdict(self) -> dict:
        before, after = self._documents()
        return attribute_rung(
            compare_fingerprints(before, after),
            PHASE332_OPPADJ_RUNG,
            before=before,
            after=after,
            ladder_directory=FINGERPRINT_DIR,
            rung_prefix=PHASE332_RUNG_PREFIX,
        )

    def test_the_attribution_is_clean(self) -> None:
        import tests.phase33_state as state

        verdict = self._verdict()
        assert verdict["ok"], verdict["failures"]
        assert not verdict["blocking"]
        assert set(verdict["non_clock_moves"]) <= set(PREDICTED_CHANGED)
        assert verdict["non_clock_moves"] == sorted(state.P332_16_RUNG7_MOVED_COLUMNS)

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

    def test_the_cause_digest_matches_the_state_record(self) -> None:
        import tests.phase33_state as state

        digest = hashlib.sha256(
            fg.PHASE332_OPPADJ_RUNG_CAUSE.encode("utf-8")
        ).hexdigest()
        assert digest == state.P332_16_RUNG7_CAUSE_DIGEST

    def test_the_committed_diff_records_the_rung(self) -> None:
        diff = tomllib.loads(DIFF_TOML.read_text(encoding="utf-8"))
        rung = diff["rung"]["7"]
        assert rung["cause"] == fg.PHASE332_OPPADJ_RUNG_CAUSE
        assert rung["attributor"] == "_attribute_p332_oppadj"
        assert rung["baseline_document"] == STEP6B.name
        assert rung["attribution_ok"] is True
        assert rung["unattributed_columns"] == []
        # The generator writes the widths as TOML strings (``_toml_array``).
        assert tuple(int(w) for w in rung["widths_before"]) == PREDICTED_WIDTHS_BEFORE
        assert tuple(int(w) for w in rung["widths_after"]) == PREDICTED_WIDTHS_AFTER
        assert sorted(rung["added_columns"]) == list(PREDICTED_ADDED)


@pytest.mark.skipif(
    not all((GOLD / f"features_{t}.parquet").is_file() for t in ("wp", "ats", "ou")),
    reason="production gold is absent (data/ is gitignored)",
)
class TestTheFlagsSurviveIntoGold:
    """THE GOLD-SURVIVAL GUARD: all four prefixed flags PRESENT in all three matrices."""

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_every_flag_is_present_and_a_level(self, target: str) -> None:
        frame = pd.read_parquet(GOLD / f"features_{target}.parquet")
        missing = [name for name in PREDICTED_ADDED if name not in frame.columns]
        assert missing == [], f"features_{target} lacks {missing}"
        flags = frame[list(PREDICTED_ADDED)]
        assert flags.isin([0.0, 1.0]).all().all()
        assert (flags == 1.0).any().all(), "every flag must be set somewhere"
        assert (flags == 0.0).any().all(), "every flag must be unset somewhere"

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_a_value_is_nan_exactly_where_its_flag_is_unset(self, target: str) -> None:
        frame = pd.read_parquet(GOLD / f"features_{target}.parquet")
        for prefix in ("home", "away"):
            for side in ("off", "def"):
                flag = frame[f"{prefix}_{side}_{OPP_ADJ_COVERAGE_COLUMN}"]
                for name in (
                    "rolling_opp_adj_epa_per_play",
                    "rolling_opp_adj_pass_epa",
                    "rolling_opp_adj_rush_epa",
                ):
                    value = frame[f"{prefix}_{side}_{name}"]
                    assert (value.isna() == (flag == 0.0)).all(), (target, name)


@pytest.mark.skipif(
    not (DIGEST_BEFORE.is_file() and DIGEST_AFTER.is_file()),
    reason="the rung-7 digest bracket documents are gitignored runtime state",
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
