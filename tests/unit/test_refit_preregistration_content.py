"""The frozen 2026 re-fit record agrees with the code it describes (Phase 37, Plan 37-04).

The pre-registration (``backtest/refit_preregistration_2026.py`` plus the root
``ROLLING-REFIT-PREREGISTRATION.md``) restates values the code also holds: the input caps, the
trainers' settings, the EV floors, alpha, the excluded groups, the frozen re-measure constants, the
shadow pin. These tests prove the two agree at the moment of freezing. They were run against the
UNCOMMITTED record before the one-way push, because a disagreement found after the push can only
be repaired by a correction commit that INVALIDATES the record by sha.

They also check RELATIONSHIPS, not only that strings appear: every date list follows the D-02
formula on the record's own constants, and the markdown's list table carries exactly those dates
and weeks, in order.

The markdown is read from the repository root by path.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import backtest.refit_preregistration_2026 as record
from api.cache_schema import BET_LIST_FILL_COLUMNS
from backtest import shadow_pin_2026 as pin
from backtest.diagnose import SIGNIFICANCE_ALPHA
from backtest.neutral_hfa_ev_chain_constants import (
    CORRECTED_BIAS_SEED_SEASONS,
    CORRECTED_DERIVATION_SEASONS,
    CORRECTED_EV_FLOOR_BY_TARGET,
    CORRECTED_WP_DERIVATION_SEASONS,
)
from backtest.recipe_registry import RECIPE_REGISTRY
from conf.season_partition import CORPUS_FIRST_SEASON
from config.blend_sources import BLEND_CONVERTER_ARTIFACT_ID
from forward_ledger import store
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1
from forward_ledger.shadow import SHADOW_COMMIT_HEADROOM
from forward_ledger.verdict import DECISION_RUN_LEAD
from models.train import GROUP_GATE_VERDICT_PATH, verdict_exclusion
from models.trainers.ats_trainer import _ATS_MAX_FEATURES, ATSTrainer
from models.trainers.ou_trainer import _OU_MAX_FEATURES, OUTrainer
from models.trainers.wp_trainer import _WP_MAX_FEATURES, WPTrainer
from scripts import derive_cold_start_constants as cold_start
from scripts import derive_corrected_ev_chain as ev_chain

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_MD = REPO_ROOT / "ROLLING-REFIT-PREREGISTRATION.md"
ET = ZoneInfo("America/New_York")
TUESDAY = 1  # date.weekday(): Monday is 0
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@pytest.fixture(scope="module")
def markdown() -> str:
    """The human-readable half, read as text (universal newlines)."""
    return RECORD_MD.read_text(encoding="utf-8")


def _last_tuesday_strictly_before(day: date) -> date:
    days_back = (day.weekday() - TUESDAY) % 7 or 7
    return day - timedelta(days=days_back)


def _first_game_et_date(week: int) -> date:
    """The ET calendar date of a week's first game, from the record's own evidence."""
    if week in record.FIRST_KICKOFF_EVIDENCE_UTC:
        instant = datetime.fromisoformat(record.FIRST_KICKOFF_EVIDENCE_UTC[week])
        return instant.astimezone(ET).date()
    return date.fromisoformat(record.PLAYOFF_FIRST_GAME_DATES[week])


# ---------------------------------------------------------------------------
# The training recipe restates the trainers
# ---------------------------------------------------------------------------


def test_caps_equal_the_trainers_caps() -> None:
    assert record.INPUT_CAPS == {
        "wp": _WP_MAX_FEATURES,
        "ats": _ATS_MAX_FEATURES,
        "ou": _OU_MAX_FEATURES,
    }


def test_settings_equal_each_trainers_defaults() -> None:
    defaults = {
        "wp": WPTrainer()._get_default_params(),
        "ats": ATSTrainer()._get_default_params(),
        "ou": OUTrainer()._get_default_params(),
    }
    assert defaults == record.MODEL_SETTINGS


def test_training_start_is_the_corpus_floor() -> None:
    assert record.TRAINING_START_SEASON == CORPUS_FIRST_SEASON


def test_excluded_groups_and_digest_equal_the_ratified_verdict() -> None:
    groups, digest = verdict_exclusion()
    assert groups == record.EXCLUDED_FEATURE_GROUPS
    assert digest == record.GROUP_VERDICT_NORMALIZED_SHA256
    assert (
        GROUP_GATE_VERDICT_PATH.relative_to(REPO_ROOT).as_posix()
        == record.GROUP_VERDICT_PATH
    )


# ---------------------------------------------------------------------------
# The bet rule, the statistics and the A/A list restate the code
# ---------------------------------------------------------------------------


def test_ev_floors_alpha_and_converter_equal_the_code() -> None:
    assert record.EV_FLOORS == CORRECTED_EV_FLOOR_BY_TARGET
    assert record.ALPHA == SIGNIFICANCE_ALPHA
    assert record.MARKET_CONVERTER_ID == pin.SHADOW_CONVERTER
    assert record.MARKET_CONVERTER_ID == BLEND_CONVERTER_ARTIFACT_ID


def test_aa_exclusions_are_real_field_names() -> None:
    expected = ("arm", "seq", "chain_hash", *BET_LIST_FILL_COLUMNS)
    assert expected == record.AA_EXCLUDED_FIELDS
    assert "arm" in IMMUTABLE_COLUMNS_V1
    assert {"seq", "chain_hash"} <= set(store._LINE_KEYS)


def test_remeasure_constants_equal_the_frozen_procedure() -> None:
    assert record.BIAS_SEED_SEASONS == ev_chain.BIAS_SEED_SEASONS
    assert record.BIAS_SEED_SEASONS == CORRECTED_BIAS_SEED_SEASONS
    assert record.HISTORICAL_TUNE_SEASONS == ev_chain.DERIVATION_SEASONS
    assert record.HISTORICAL_TUNE_SEASONS == CORRECTED_DERIVATION_SEASONS
    assert record.WP_HISTORICAL_TUNE_SEASONS == CORRECTED_WP_DERIVATION_SEASONS
    folds = (*record.BIAS_SEED_SEASONS, *record.HISTORICAL_TUNE_SEASONS)
    assert folds == record.HISTORICAL_FOLD_SEASONS
    assert record.WP_EDGE_ANCHOR == (
        cold_start.WP_HIGH_THRESHOLD,
        cold_start.WP_MEDIUM_THRESHOLD,
    )
    assert record.OU_TOTAL_FLOOR == cold_start.OU_TOTAL_FLOOR
    assert record.EDGE_QUANTILE_METHOD == cold_start.QUANTILE_METHOD
    assert record.MIN_USABLE_REMEASURE_ROWS == cold_start.MIN_HONEST_THRESHOLD_ROWS


# ---------------------------------------------------------------------------
# The frozen arm restates the row-19 registry entry
# ---------------------------------------------------------------------------


def test_the_pin_equals_the_row19_registry_entry() -> None:
    entry = RECIPE_REGISTRY[pin.SHADOW_RECIPE]
    assert dict(pin.SHADOW_MODEL_IDS) == dict(entry.model_artifact_ids)
    assert entry.blend_id == pin.SHADOW_BLEND
    assert entry.converter_id == pin.SHADOW_CONVERTER
    assert entry.chain_fit_record_path == pin.SHADOW_CHAIN_FIT_PATH
    assert entry.chain_fit_record_sha256 == pin.SHADOW_CHAIN_FIT_SHA256
    assert record.SHADOW_RECIPE_ID == pin.SHADOW_RECIPE
    assert record.REFIT_RECIPE_ID not in RECIPE_REGISTRY
    assert entry.fill_convention_id == record.FILL_CONVENTION_ID


# ---------------------------------------------------------------------------
# The calendar follows the D-02 formula on the record's own constants
# ---------------------------------------------------------------------------


def test_the_deadline_is_week_8s_first_decision_run() -> None:
    first_game = _first_game_et_date(record.AA_FIRST_WEEK)
    first_lock = datetime.combine(
        first_game - timedelta(days=1), datetime.min.time(), tzinfo=ET
    ).replace(hour=18)
    first_run = first_lock - DECISION_RUN_LEAD
    assert first_run == datetime.fromisoformat(record.PREREGISTRATION_DEADLINE_UTC)


def test_cadence_tuesdays_are_the_last_tuesday_before_each_first_game() -> None:
    assert sorted(record.CADENCE_TUESDAY_BY_WEEK) == list(
        range(record.AA_FIRST_WEEK + 1, record.LAST_NFL_WEEK + 1)
    )
    for week, tuesday in record.CADENCE_TUESDAY_BY_WEEK.items():
        expected = _last_tuesday_strictly_before(_first_game_et_date(week))
        assert date.fromisoformat(tuesday) == expected, week


def test_date_lists_follow_the_formula() -> None:
    weeks = list(range(record.AA_FIRST_WEEK, record.AA_LAST_WEEK + 1))
    assert sorted(record.REFIT_DATE_LISTS) == weeks
    assert sorted(record.AA_EVALUATION_TUESDAY_BY_WEEK) == weeks
    spacing = record.CADENCE_SPACING_NFL_WEEKS
    for first_clean, dates in record.REFIT_DATE_LISTS.items():
        evaluation = record.AA_EVALUATION_TUESDAY_BY_WEEK[first_clean]
        assert evaluation == record.CADENCE_TUESDAY_BY_WEEK[first_clean + 1]
        assert dates[0] == (evaluation, first_clean + 1)
        for listed, week in dates:
            assert listed == record.CADENCE_TUESDAY_BY_WEEK[week]
            expected = _last_tuesday_strictly_before(_first_game_et_date(week))
            assert date.fromisoformat(listed) == expected
        list_weeks = [week for _, week in dates]
        assert all(
            later - earlier == spacing for earlier, later in pairwise(list_weeks)
        )
        assert list_weeks[-1] <= record.LAST_NFL_WEEK
        assert list_weeks[-1] + spacing > record.LAST_NFL_WEEK


def test_every_listed_cutoff_is_2200_utc() -> None:
    hour, minute = (int(part) for part in record.CUTOFF_LOCAL_TIME.split(":"))
    zone = ZoneInfo(record.CUTOFF_TIMEZONE)
    for dates in record.REFIT_DATE_LISTS.values():
        for listed, _ in dates:
            cutoff = datetime.combine(
                date.fromisoformat(listed), datetime.min.time(), tzinfo=zone
            ).replace(hour=hour, minute=minute)
            in_utc = cutoff.astimezone(UTC)
            assert (in_utc.hour, in_utc.minute) == (22, 0), listed


# ---------------------------------------------------------------------------
# The markdown states what the constants hold
# ---------------------------------------------------------------------------


def test_the_markdown_carries_the_frozen_sentences_and_names(markdown: str) -> None:
    for dates in record.REFIT_DATE_LISTS.values():
        for listed, _ in dates:
            assert listed in markdown
    assert record.GATE_RELATIONSHIP in markdown
    assert record.LEAN_POOL_RULE in markdown
    assert record.NO_CLEAN_WEEK_RULE in markdown
    for check in record.INTEGRITY_CHECKS:
        assert f"`{check}`" in markdown, check
    for gate_file in record.UNTOUCHED_GATE_FILES:
        assert gate_file in markdown
    assert record.REFIT_RECIPE_ID in markdown
    assert record.SHADOW_RECIPE_ID in markdown
    headroom = int(SHADOW_COMMIT_HEADROOM.total_seconds())
    assert f"{headroom} seconds" in markdown


def test_the_anchor_block_names_both_paths(markdown: str) -> None:
    assert (
        "preregistration_paths: ROLLING-REFIT-PREREGISTRATION.md, "
        "backtest/refit_preregistration_2026.py"
    ) in markdown
    deadline = datetime.fromisoformat(record.PREREGISTRATION_DEADLINE_UTC)
    local = deadline.astimezone(ET).isoformat()
    assert f"preregistration_deadline: {local}" in markdown


def test_integrity_checks_are_eleven_in_run_order() -> None:
    checks = record.INTEGRITY_CHECKS
    assert len(checks) == 11
    assert len(set(checks)) == 11
    assert checks.index("dry_stamp_and_validate") + 1 == checks.index(
        "clean_code_commit"
    )
    assert checks.index("clean_code_commit") + 1 == checks.index("bundle_validation")
    assert checks[-1] == "bundle_validation"


def test_the_aa_completion_reason_is_frozen(markdown: str) -> None:
    section_5 = markdown.index("\n## 5. ")
    section_6 = markdown.index("\n## 6. ")
    first = markdown.index("week_not_complete")
    assert section_5 < first < section_6


def test_the_markdown_list_table_matches_the_constants(markdown: str) -> None:
    lines = markdown.splitlines()
    for first_clean, dates in record.REFIT_DATE_LISTS.items():
        rows = [line for line in lines if line.startswith(f"| {first_clean} |")]
        assert len(rows) == 1, (first_clean, rows)
        row = rows[0]
        cells = [f"{listed} (week {week})" for listed, week in dates]
        missing = [cell for cell in cells if cell not in row]
        assert missing == [], (first_clean, missing)
        positions = [row.index(cell) for cell in cells]
        assert positions == sorted(positions), row
        assert ISO_DATE.findall(row) == [listed for listed, _ in dates], row
