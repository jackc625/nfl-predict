"""Every recorded Phase-33 acceptance artifact carries a VALUE, never only a status.

WHY (T-33-90). The defect this phase exists to fix produced GREEN RUNS while serving imputed
Elo to the deployed WP model. A zero exit code, a boolean "passed", or a verdict word with no
number behind it is therefore not evidence of anything this phase claims. This module walks
each committed acceptance artifact and asserts the recorded evidence is a measurement:

* ``config/phase33_gate_verdict.toml`` -- every verdict word has a paired statistic, a p-value
  inside [0, 1] and a positive pair count; the blend carries numeric CLV before AND after.
* ``config/phase33_gold_rebuild_diff.toml`` -- every rung carries its added/removed column and
  season LISTS and its widths, present even where empty ("absent" and "measured nothing" must
  not read the same).
* ``backtest/cold_start_constants.py`` -- the pre-registered thresholds, biases and counts are
  NUMBERS. The module is FROZEN at 11761c7; it is read here, never written.
* ``tests/phase33_state.py`` -- no boolean stands in for a measurement. Each of the few boolean
  slots is paired with the measured value it summarises, and that value is not a boolean.

Plus the non-vacuity control: the artifact list is non-empty and every artifact yields at least
one item to check, so an artifact that stopped parsing cannot pass by having nothing to say.

READ-ONLY, no ``data/`` access. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
import tomllib
from numbers import Real
from pathlib import Path

import pytest

import backtest.cold_start_constants as frozen
import tests.phase33_state as state

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_VERDICT = REPO_ROOT / "config" / "phase33_gate_verdict.toml"
GOLD_REBUILD_DIFF = REPO_ROOT / "config" / "phase33_gold_rebuild_diff.toml"
FROZEN_MODULE = REPO_ROOT / "backtest" / "cold_start_constants.py"
STATE_MODULE = REPO_ROOT / "tests" / "phase33_state.py"

ACCEPTANCE_ARTIFACTS: tuple[Path, ...] = (
    GATE_VERDICT,
    GOLD_REBUILD_DIFF,
    FROZEN_MODULE,
    STATE_MODULE,
)

# Each boolean slot in tests/phase33_state.py, paired with the MEASURED slot(s) it summarises.
# A new boolean slot fails this module until it is paired here with a measurement.
BOOLEAN_SLOT_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "W8X_FAILURE_SET_IS_EXACTLY_THE_TRIPWIRES": (
        "W8X_POST_TASK_FAILURE_SET",
        "W8X_POST_TASK_COLLECTED",
    ),
    "W8X_WALL_CLOCK_COMPARISON_IS_CONFOUNDED": ("W8X_WALL_CLOCK_DELTA_SECONDS",),
    "SILVER_GAMES_HAS_TWO_COPIES": (
        "SILVER_GAMES_ROWS_AFTER_IDENTITY",
        "SILVER_GAMES_TABLE_DIGEST_AFTER_RESYNC",
    ),
    "PRECIP_PARTITION_PROMOTED_TO_ITS_OWN_RUNG": ("PRECIP_PARTITION_PREMISE",),
    "P332_20_CLEAN_BUILD_PIN_EQUALS_MEASURED": (
        "P332_20_CLEAN_BUILD_PIN_WIDTHS",
        "P332_20_CLEAN_BUILD_WIDTHS",
    ),
    "P332_20_CLEAN_BUILD_MEASURED_EQUALS_LADDER": (
        "P332_20_CLEAN_BUILD_WIDTHS",
        "P332_20_CLEAN_BUILD_LADDER_WIDTHS",
    ),
}


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class TestTheArtifactListIsNotVacuous:
    def test_there_are_artifacts_to_check(self):
        assert len(ACCEPTANCE_ARTIFACTS) == 4

    @pytest.mark.parametrize("path", ACCEPTANCE_ARTIFACTS, ids=lambda p: p.name)
    def test_every_artifact_exists(self, path):
        assert path.is_file(), f"missing acceptance artifact: {path}"


class TestEveryGateVerdictCarriesItsStatistic:
    @pytest.fixture(scope="class")
    def verdicts(self) -> dict:
        document = _load_toml(GATE_VERDICT)
        assert document["verdicts"], "the gate verdict document carries no verdicts"
        return document["verdicts"]

    def test_all_three_targets_are_recorded(self, verdicts):
        assert set(verdicts) == {"wp", "ats", "ou"}

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_the_verdict_word_has_a_paired_statistic_and_p_value(
        self, verdicts, target
    ):
        row = verdicts[target]
        assert row["verdict"] in ("PASS", "FAIL", "UNTESTABLE_REFUSAL")
        assert _is_number(row["paired_statistic"]) and math.isfinite(
            row["paired_statistic"]
        ), f"{target}: the verdict {row['verdict']!r} has no finite paired statistic"
        assert _is_number(row["p_value"]) and 0.0 <= row["p_value"] <= 1.0
        assert _is_number(row["n_paired"]) and row["n_paired"] > 0

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_the_blend_clv_is_a_number_before_and_after(self, target):
        blend = _load_toml(GATE_VERDICT)["blend"]
        for side in ("clv_before", "clv_after"):
            value = blend[side][target]
            assert _is_number(value) and math.isfinite(value), (
                f"blend {side}[{target}] is {value!r}, not a measured number"
            )


class TestEveryRebuildRungCarriesItsLists:
    @pytest.fixture(scope="class")
    def rungs(self) -> dict:
        rungs = _load_toml(GOLD_REBUILD_DIFF)["rung"]
        assert rungs, "the rebuild diff carries no rungs"
        return rungs

    def test_every_rung_carries_every_list_even_when_empty(self, rungs):
        required = (
            "added_columns",
            "removed_columns",
            "added_seasons",
            "removed_seasons",
            "widths_before",
            "widths_after",
        )
        for number, rung in rungs.items():
            missing = [key for key in required if not isinstance(rung.get(key), list)]
            assert not missing, f"rung {number} is missing the lists {missing}"

    def test_every_rung_records_three_widths_on_each_side(self, rungs):
        for number, rung in rungs.items():
            assert len(rung["widths_before"]) == len(rung["widths_after"]) == 3, number


class TestTheFrozenConstantsAreNumbers:
    """Read-only over the module frozen at 11761c7."""

    def test_the_edge_tier_thresholds_are_numbers(self):
        assert frozen.EDGE_TIER_THRESHOLDS_BY_TARGET
        for target, pair in frozen.EDGE_TIER_THRESHOLDS_BY_TARGET.items():
            assert len(pair) == 2 and all(_is_number(v) for v in pair), target

    def test_the_2026_chain_fit_bias_is_numeric(self):
        assert frozen.CHAIN_FIT_BIAS_2026
        assert all(_is_number(v) for v in frozen.CHAIN_FIT_BIAS_2026.values())

    def test_the_band_movement_counts_are_integers(self):
        assert frozen.GAMES_CHANGING_BAND
        assert all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in frozen.GAMES_CHANGING_BAND.values()
        )

    def test_the_fix_cycle_allowance_is_a_count(self):
        allowance = frozen.FIX_CYCLE_ALLOWANCE
        assert isinstance(allowance, int) and not isinstance(allowance, bool)


class TestNoStateSlotIsABooleanStandingInForAMeasurement:
    def test_every_boolean_slot_is_paired_with_a_measurement(self):
        booleans = sorted(
            name
            for name in dir(state)
            if name.isupper() and isinstance(getattr(state, name), bool)
        )
        assert booleans, "no boolean slots found; the scan is not reading the module"
        unpaired = [name for name in booleans if name not in BOOLEAN_SLOT_MEASUREMENTS]
        assert not unpaired, (
            f"boolean slots with no paired measurement: {unpaired}. A boolean records a "
            "conclusion; the number it was concluded from must be recorded beside it."
        )

    @pytest.mark.parametrize("flag", sorted(BOOLEAN_SLOT_MEASUREMENTS))
    def test_each_paired_measurement_exists_and_is_not_a_boolean(self, flag):
        for measured in BOOLEAN_SLOT_MEASUREMENTS[flag]:
            assert hasattr(state, measured), f"{flag} names a missing slot {measured}"
            value = getattr(state, measured)
            assert not isinstance(value, bool), f"{measured} is itself a boolean"
            assert value not in (None, "", ()), f"{measured} is empty"
