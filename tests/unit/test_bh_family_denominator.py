"""The multiplicity denominator is the PRE-REGISTERED one, and its exclusions are arithmetic.

An argued denominator is a post-hoc denominator. ``BH_FAMILY_SPEC`` enumerated the family
BEFORE any 2025 number existed -- three primary verdicts, three robustness cuts counted once
each, plus one entry per target whose registered calibration fallback actually fired -- so 6
with no fallback and 7, 8 or 9 with one, two or three. This module asserts that the runner's
family is exactly that enumeration, and that the two exclusions are counted rather than
described:

* the TUNE-SIDE ``EV_FLOOR_GRID`` sweep is RECORDED in the registry and does NOT enter the
  family, because those grid cells ran entirely on the tune split and never touched the hold.
  Counting all fifteen would correct for fourteen tests never performed on the hold, making a
  real result harder to detect for ceremonial rather than statistical reasons;
* every ``entry_kind = control`` row is RECORDED and excluded, because the shifted-edge
  counterfactual has no p-value to correct and counting it would statistically penalise the
  verdict for running a code-liveness check.

Registry row count and denominator therefore DIFFER, visibly and by design. That difference is
asserted arithmetically here, which is what stops it being read as a bug.

THE DUPLICATE CUT COUNTS ONCE, and the reason is measured rather than argued:
``backtest/ou_monetization.py:690-710`` assigns BOTH ``regular_season_only`` and
``playoffs_excluded`` the same value from the same ``week <= 18`` filter, so they compute the
BYTE-IDENTICAL frame. One statistic reported twice is not two hypotheses.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from backtest.ev_chain_constants import (
    BH_FAMILY_SPEC,
    ROBUSTNESS_CUT_BH_COUNT,
    ROBUSTNESS_CUTS_P31,
    TRIAL_ENTRY_KIND_CONTROL,
    TRIAL_ENTRY_KIND_INFERENCE,
    TRIAL_REGISTRY_FIELDS_P31,
)
from backtest.ou_ev_chain import EV_FLOOR_GRID
from backtest.profitability_2025 import (
    CANONICAL_TARGETS,
    _robustness_cut_frames,
    benjamini_hochberg_adjusted,
    bh_family_members,
    registry_exclusion_counts,
)

HOLD_LABEL = "hold_2024"
TUNE_LABEL = "tune_2021_2023"

PREREGISTERED_DENOMINATOR = int(BH_FAMILY_SPEC["denominator_no_fallback"])


def _entry(
    entry_id: str,
    target: str,
    *,
    sample_window: str,
    entry_kind: str = TRIAL_ENTRY_KIND_INFERENCE,
    raw_p: float | None = 0.2,
) -> dict[str, object]:
    """A registry row in the frozen field order, with only the fields the family rule reads."""
    row: dict[str, object] = {"entry_id": entry_id, "target": target}
    row.update(dict.fromkeys(TRIAL_REGISTRY_FIELDS_P31))
    row["sample_window"] = sample_window
    row["entry_kind"] = entry_kind
    row["raw_p"] = raw_p
    return row


def _registry(fallback_targets: tuple[str, ...] = ()) -> list[dict[str, object]]:
    """A registry with the runner's own shape: sweep cells, primaries, cuts and controls."""
    rows: list[dict[str, object]] = []
    for target in CANONICAL_TARGETS:
        rows.extend(
            _entry(
                f"{target}/sweep/t={floor}",
                target,
                sample_window=TUNE_LABEL,
                raw_p=None,
            )
            for floor in EV_FLOOR_GRID
        )
    for target in CANONICAL_TARGETS:
        rows.append(_entry(f"{target}/primary", target, sample_window=HOLD_LABEL))
        rows.append(_entry(f"{target}/robustness", target, sample_window=HOLD_LABEL))
        if target in fallback_targets:
            rows.append(
                _entry(
                    f"{target}/calibration_fallback", target, sample_window=HOLD_LABEL
                )
            )
        rows.append(
            _entry(
                f"{target}/control/shifted_edge",
                target,
                sample_window=HOLD_LABEL,
                entry_kind=TRIAL_ENTRY_KIND_CONTROL,
                raw_p=None,
            )
        )
    return rows


class TestTheDenominatorIsThePreRegisteredEnumeration:
    """6 with no fallback, and exactly one more per fired fallback."""

    def test_a_no_fallback_run_has_the_pre_registered_denominator(self) -> None:
        family = bh_family_members(_registry(), HOLD_LABEL)
        assert len(family) == PREREGISTERED_DENOMINATOR == 6
        assert sorted(entry["entry_id"] for entry in family) == [
            "ats/primary",
            "ats/robustness",
            "ou/primary",
            "ou/robustness",
            "wp/primary",
            "wp/robustness",
        ]

    @pytest.mark.parametrize(
        "fired",
        [(), ("wp",), ("wp", "ats"), ("wp", "ats", "ou")],
    )
    def test_each_fired_fallback_adds_exactly_one(self, fired: tuple[str, ...]) -> None:
        family = bh_family_members(_registry(fired), HOLD_LABEL)
        assert len(family) == PREREGISTERED_DENOMINATOR + len(fired)
        assert len(family) in (6, 7, 8, 9), (
            "the pre-registration states the denominator is 6 with no fallback and 7, 8 or 9 "
            "with one, two or three."
        )

    def test_the_frozen_spec_still_says_six(self) -> None:
        """A guard on the RULE, so a later edit to the frozen module turns this red."""
        assert BH_FAMILY_SPEC["denominator_no_fallback"] == 6
        assert "7, 8 or 9" in str(BH_FAMILY_SPEC["denominator_with_fallbacks"])


class TestTheExclusionsAreCountedNotDescribed:
    """Row count minus denominator equals the excluded rows, exactly."""

    def test_the_tune_side_sweep_is_recorded_and_excluded(self) -> None:
        registry = _registry()
        sweep_ids = {
            entry["entry_id"]
            for entry in registry
            if entry["sample_window"] == TUNE_LABEL
        }
        assert len(sweep_ids) == len(EV_FLOOR_GRID) * len(CANONICAL_TARGETS) == 15, (
            "the sweep must be RECORDED for transparency; excluding it from the family is not "
            "the same as leaving it out of the registry."
        )
        family_ids = {
            entry["entry_id"] for entry in bh_family_members(registry, HOLD_LABEL)
        }
        assert not (sweep_ids & family_ids), sorted(sweep_ids & family_ids)

    def test_control_entries_are_recorded_and_excluded(self) -> None:
        registry = _registry()
        control_ids = {
            entry["entry_id"]
            for entry in registry
            if entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL
        }
        assert len(control_ids) == len(CANONICAL_TARGETS)
        family_ids = {
            entry["entry_id"] for entry in bh_family_members(registry, HOLD_LABEL)
        }
        assert not (control_ids & family_ids)

    @pytest.mark.parametrize("fired", [(), ("ou",), ("wp", "ats", "ou")])
    def test_rows_minus_denominator_equals_the_excluded_count(
        self, fired: tuple[str, ...]
    ) -> None:
        counts = registry_exclusion_counts(_registry(fired), HOLD_LABEL)
        assert (
            counts["registry_rows"]
            == counts["bh_family_size"]
            + counts["excluded_control_entries"]
            + counts["excluded_tune_side_sweep_cells"]
        ), counts
        assert counts["registry_rows"] > counts["bh_family_size"], (
            "the registry must carry MORE rows than the denominator counts. Every read of the "
            "hold is on the record; the denominator counts only actual inferences."
        )

    def test_the_live_run_reproduces_that_arithmetic(self, p31_rehearsal_run) -> None:
        """The same identity on the runner's OWN registry, not only on a hand-built one."""
        run = p31_rehearsal_run["result"]["run"]
        assert (
            run["registry_rows"]
            == run["bh_denominator"]
            + run["excluded_control_entries"]
            + run["excluded_tune_side_sweep_cells"]
        ), run
        assert (
            run["bh_denominator"]
            == PREREGISTERED_DENOMINATOR + run["bh_fallbacks_fired"]
        ), (
            "the live run's denominator is not the pre-registered enumeration plus its fired "
            "fallbacks."
        )
        assert run["excluded_tune_side_sweep_cells"] == len(EV_FLOOR_GRID) * len(
            CANONICAL_TARGETS
        )
        assert run["excluded_control_entries"] == len(CANONICAL_TARGETS)


class TestTheDuplicateCutCountsOnce:
    """Measured identity, not an argument about what the two cut names mean."""

    def test_the_two_named_cuts_compute_the_identical_frame(self) -> None:
        per_bet = pd.DataFrame(
            [
                {"season": 2024, "week": week, "flat_stake": 1.0, "payout_flat": 0.5}
                for week in (1, 9, 18, 19, 20, 21)
            ]
        )
        cuts = _robustness_cut_frames(per_bet)
        assert set(cuts) == set(ROBUSTNESS_CUTS_P31)
        first = cuts[ROBUSTNESS_CUTS_P31[0]]
        for name, frame in cuts.items():
            assert frame.equals(first), (
                f"cut {name!r} is not the byte-identical frame the frozen Phase-27 code "
                "produces for both names. One statistic reported twice is not two "
                "hypotheses, and that claim rests on this identity."
            )

    def test_the_cut_is_genuine_and_not_vacuous_under_the_playoffs_everywhere_scope(
        self,
    ) -> None:
        """D31-38 put playoff rows in the population, so ``week <= 18`` actually bites."""
        per_bet = pd.DataFrame(
            [
                {"season": 2024, "week": week, "flat_stake": 1.0, "payout_flat": 0.5}
                for week in (1, 9, 18, 19, 20, 21)
            ]
        )
        cut = _robustness_cut_frames(per_bet)[ROBUSTNESS_CUTS_P31[0]]
        assert len(cut) == 3 < len(per_bet), (
            "the regular-season cut did not remove the playoff rows, so it is the vacuous cut "
            "Phase 27 disclosed rather than the genuine one D31-38 created."
        )

    def test_the_pair_contributes_one_entry_to_the_family(self) -> None:
        assert ROBUSTNESS_CUT_BH_COUNT == 1
        family = bh_family_members(_registry(), HOLD_LABEL)
        cut_entries = [
            entry for entry in family if str(entry["entry_id"]).endswith("/robustness")
        ]
        assert len(cut_entries) == len(CANONICAL_TARGETS), (
            "each target contributes exactly ONE robustness entry; two would correct for a "
            "duplicate and make a real result harder to detect for ceremonial reasons."
        )

    def test_both_cut_names_are_retained_for_lineage(self) -> None:
        assert ROBUSTNESS_CUTS_P31 == ("regular_season_only", "playoffs_excluded")

    def test_the_live_run_records_the_identity_it_claims(
        self, p31_rehearsal_run
    ) -> None:
        for target, record in p31_rehearsal_run["result"]["targets"].items():
            assert record["robustness_cuts_identical"] is True, target


class TestTheCorrectionUsesThePreRegisteredM:
    """The denominator is enumerated in advance, never inferred from how many p's exist."""

    def test_it_agrees_with_scipy_when_m_equals_the_input_length(self) -> None:
        pvalues = [0.001, 0.04, 0.03, 0.2, 0.9, 0.01]
        mine = benjamini_hochberg_adjusted(pvalues, len(pvalues))
        theirs = list(false_discovery_control(pvalues, method="bh"))
        assert mine == pytest.approx(theirs, abs=1e-15), (mine, theirs)

    def test_a_larger_m_makes_the_correction_strictly_more_conservative(self) -> None:
        pvalues = [0.001, 0.02]
        at_two = benjamini_hochberg_adjusted(pvalues, 2)
        at_six = benjamini_hochberg_adjusted(pvalues, 6)
        assert all(big >= small for big, small in zip(at_six, at_two, strict=True))
        assert at_six != at_two, (
            "correcting a two-test subset at the pre-registered m=6 must differ from "
            "correcting it at m=2; otherwise the enumerated denominator has no effect and a "
            "target with no bets would quietly make the surviving tests easier."
        )

    def test_the_adjusted_values_are_monotone_in_the_raw_ones(self) -> None:
        pvalues = [0.001, 0.04, 0.03, 0.2, 0.9, 0.01]
        adjusted = benjamini_hochberg_adjusted(pvalues, 6)
        pairs = sorted(zip(pvalues, adjusted, strict=True))
        assert all(later >= earlier for (_, earlier), (_, later) in pairwise(pairs)), (
            pairs
        )

    def test_every_adjusted_value_is_a_probability(self) -> None:
        adjusted = benjamini_hochberg_adjusted([0.5, 0.9, 0.99], 6)
        assert all(0.0 <= value <= 1.0 for value in adjusted), adjusted

    def test_an_m_smaller_than_the_tests_performed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="smaller than the"):
            benjamini_hochberg_adjusted([0.1, 0.2, 0.3], 2)

    def test_an_empty_family_returns_nothing_rather_than_raising(self) -> None:
        assert benjamini_hochberg_adjusted([], 6) == []


class TestAFiredFallbackDoesNotDuplicateItsPrimarysPValue:
    """WR-11: the fallback row is a bookkeeping row, and a duplicate p makes the family LENIENT.

    When a target's calibration fallback fires, the strategy is built WITH the gate, so there is
    exactly ONE selection and ONE statistic. The ``<target>/calibration_fallback`` registry row
    records that same statistic a second time -- ``raw_p`` and ``roi`` are the identical numbers
    as that target's ``/primary`` row. It discloses that the fallback fired and what it produced;
    it is not an independent hypothesis.

    Feeding the same p into the BH input twice is not neutral. A tied p occupies two adjacent
    ranks and the step-up takes ``min_{j>=i} m*p_(j)/j``, so the duplicate at rank ``i+1`` gives
    ``m*p/(i+1) < m*p/i`` and pulls the PRIMARY's q DOWN. Measured on the shipped run's p-value
    shape with the family growing 6 -> 7, that is 0.600 -> 0.560, against 0.700 when the duplicate
    stays out of the ranking. Firing a fallback made the verdict EASIER to call significant, which
    is the opposite of what a multiplicity correction is for.

    What the FROZEN ``BH_FAMILY_SPEC`` pre-registers is preserved exactly: the membership ("0 to 3
    entries") and the denominator ("7, 8 or 9"). ``bh_family_members`` selects on
    ``entry_kind``/``sample_window``, never on ``raw_p``, so the row still grows m.

    No fallback fired in the shipped run (bh_denominator = 6), so nothing published is affected.
    """

    # The shipped run's six family p-values, in registry order (3 primary + 3 robustness),
    # rounded from PROFITABILITY-READOUT.md. Used only to show the DIRECTION of the effect on a
    # realistic shape; nothing here re-derives a published figure.
    _SHIPPED_SHAPE = [0.336, 0.761, 0.336, 0.400, 0.800, 0.400]

    def test_a_duplicated_p_value_makes_its_own_primary_easier_not_harder(self) -> None:
        """The mechanism, demonstrated on the correction function directly."""
        with_duplicate = benjamini_hochberg_adjusted(
            [*self._SHIPPED_SHAPE, self._SHIPPED_SHAPE[0]], 7
        )
        without_duplicate = benjamini_hochberg_adjusted(self._SHIPPED_SHAPE, 7)

        assert with_duplicate[0] < without_duplicate[0], (
            "duplicating a p-value no longer pulls its primary's q down, so this module is "
            "pinning the wrong mechanism"
        )
        # And it is even more lenient than the SMALLER family it replaced -- growing m from 6 to
        # 7 should never make a verdict easier.
        at_six = benjamini_hochberg_adjusted(self._SHIPPED_SHAPE, 6)
        assert with_duplicate[0] < at_six[0]
        assert without_duplicate[0] > at_six[0]

    def test_the_runner_does_not_feed_the_fallback_p_into_the_ranking(self) -> None:
        """The fix, at the seam. A source check: reaching this branch needs a full scoring run.

        ``tests/unit/test_sportsbook_preference.py`` uses the same shape for the same reason.
        """
        import inspect

        from backtest.profitability_2025 import _measure_and_judge

        source = inspect.getsource(_measure_and_judge)

        assert "raw_by_entry[fallback_id]" not in source, (
            "the fallback row's p-value is fed back into the BH input, so it enters the ranking "
            "as a tie with its own primary"
        )
        assert "fallback_mirrors[fallback_id] = primary_id" in source, (
            "the fallback row no longer mirrors its primary's adjusted q, so the row would be "
            "recorded with no q at all"
        )

    def test_the_frozen_denominator_rule_is_untouched(self) -> None:
        """The membership and the denominator are pre-registered; only the ranking changed."""
        assert BH_FAMILY_SPEC["denominator_no_fallback"] == 6
        assert "7, 8 or 9" in str(BH_FAMILY_SPEC["denominator_with_fallbacks"])
        assert "ONLY IF that target's fallback" in " ".join(
            str(item) for item in BH_FAMILY_SPEC["included"]
        )
