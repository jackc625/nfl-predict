"""The Phase-33 fix-cycle allowance is ZERO, and it was declared before any verdict.

Phase 33, Plan 33-08 Task 3 (COLD-04, D33-12, T-33-40).

WHAT A PRE-REGISTRATION TEST CAN AND CANNOT PROVE
----------------------------------------------------
It can prove the declaration EXISTS, that it says zero, and that the module says in its
own words that the declaration precedes any verdict. It cannot prove nobody would have
edited it after seeing a number -- no test can prove intent. What makes the declaration
load-bearing is that it lives in COMMITTED SOURCE with a commit date, so an edit after
the measurement commit is visible in git history rather than invisible in a habit.

WHY ZERO AND NOT ONE
---------------------
Phase 30 pre-registered a single fix-cycle lever and it went UNSPENT for both failing
targets -- not because it was overlooked, but because it had no unspent move. Phase 33
says the same thing up front. Every candidate lever is already spoken for: the training
window belongs to Phase 37's recipe, the feature groups were bindingly ruled in Phase 30,
and hyperparameter search is out of scope. An allowance of one would have nothing
legitimate to spend, and the shape it would license -- re-running a live gate until it
agrees -- is p-hacking with extra steps.

THE PRE-FLIGHT IS NOT A RETRY (D33-33)
----------------------------------------
``preflight_health_check`` was added in answer to "what if the environment breaks
mid-run", and the answer is to fail the environment BEFORE any number exists rather than
allow a second look after one does. That distinction is easy to erode in a later reading,
so it is asserted here: the allowance is still zero, and the pre-flight's own docstring
says in those words that it creates no retry state.

PLAN 33-16 EXTENDS THIS MODULE with the six edge thresholds. It is structured so that
extension is an APPEND -- a new class at the bottom -- rather than a rewrite.

NO TEST HERE WRITES ANYTHING.

Run this module:  uv run pytest tests/unit/test_phase33_preregistration.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import json
import math
import re
from decimal import Decimal
from pathlib import Path

from backtest import cold_start_constants
from scripts import run_phase33_gate as runner
from tests import phase33_state
from utils import edge_tier

# The ONE edit above the Plan 33-16 append line, and it is the imports the appended classes
# need. No existing assertion was touched.
REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTheFixCycleAllowanceIsZero:
    """D33-12, declared in committed source before any Phase-33 verdict exists."""

    def test_the_allowance_is_zero(self) -> None:
        assert runner.PHASE33_FIX_CYCLE_ALLOWANCE == 0

    def test_the_allowance_matches_the_committed_manifest(self) -> None:
        """Two homes, one value; a drift between them fails here rather than silently."""
        assert runner.PHASE33_FIX_CYCLE_ALLOWANCE == phase33_state.FIX_CYCLE_ALLOWANCE

    def test_the_module_states_that_the_declaration_precedes_any_verdict(self) -> None:
        """The claim is IN the file, so a reader meets it before the numbers."""
        doc = (inspect.getdoc(runner) or "").upper()
        assert "BEFORE ANY VERDICT EXISTS" in doc

    def test_the_module_states_that_editing_it_later_destroys_the_evidence(
        self,
    ) -> None:
        """The ev_chain_constants voice: there is no honest repair path."""
        doc = (inspect.getdoc(runner) or "").upper()
        assert "DESTROYS" in doc and "EVIDENCE" in doc

    def test_the_module_names_why_each_candidate_lever_is_unusable(self) -> None:
        """Not an assertion of scarcity -- an enumeration of it."""
        doc = (inspect.getdoc(runner) or "").lower()
        assert "training window" in doc
        assert "phase 37" in doc
        assert "feature groups" in doc
        assert "phase 30" in doc
        assert "hyperparameter search" in doc

    def test_the_named_consequence_of_an_ats_failure_is_stated_in_advance(self) -> None:
        """If ATS fails, production RETAINS ats_20260605_220128, and that is right."""
        doc = inspect.getdoc(runner) or ""
        assert "ats_20260605_220128" in doc
        assert "61.2%" in doc

    def test_a_second_candidate_for_a_judged_target_is_refused_by_name(self) -> None:
        """The allowance is enforced, not merely declared."""
        assert issubclass(runner.FixCycleAllowanceExceededError, Exception)
        src = inspect.getsource(runner._refuse_second_candidate)
        assert "FixCycleAllowanceExceededError" in src
        assert "PHASE33_FIX_CYCLE_ALLOWANCE" in src


class TestThePreflightDoesNotCreateARetryState:
    """D33-33: address the environmental fault, do NOT loosen the rule."""

    def test_the_preflight_docstring_says_no_retry_state_is_created(self) -> None:
        doc = (inspect.getdoc(runner.preflight_health_check) or "").upper()
        assert "NO RETRY STATE" in doc

    def test_it_says_the_zero_fix_cycle_rule_is_unchanged_and_absolute(self) -> None:
        doc = (inspect.getdoc(runner.preflight_health_check) or "").upper()
        assert "UNCHANGED" in doc
        assert "ABSOLUTE" in doc

    def test_it_forecloses_the_environmental_abort_escape_hatch_by_name(self) -> None:
        doc = (inspect.getdoc(runner.preflight_health_check) or "").lower()
        assert "escape hatch" in doc
        assert "one run is the run" in doc

    def test_the_allowance_is_still_zero_with_the_preflight_in_place(self) -> None:
        """The one assertion that would catch the pre-flight being read as a licence."""
        assert runner.PHASE33_FIX_CYCLE_ALLOWANCE == 0


# ---------------------------------------------------------------------------
# THE SIX EDGE THRESHOLDS AND THE 2026 CHAIN-FIT BIAS.
#
# APPENDED by Plan 33-16 Task 4. No assertion above this line was edited; the only
# change above it is the import block the classes below need. The module was created
# by Plan 33-08 with the allowance half and structured so that this extension would
# be an APPEND rather than a rewrite.
#
# The thresholds are checked in TWO ways a single assertion could not separate: their
# VALUES (against the witness, so the frozen module and the witness cannot drift
# apart) and their committed PRECISION (against the source TEXT, because a float's
# repr cannot tell 0.05 from 0.0500 and the pre-registered property is four decimal
# places, not at-most-four).
# ---------------------------------------------------------------------------


class TestTheSixEdgeThresholdsAreFrozenToFourDecimalPlaces:
    """D33-20, frozen before Week 2 and never recomputed in-season."""

    def test_there_are_exactly_three_targets(self) -> None:
        assert sorted(cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET) == [
            "ats",
            "ou",
            "wp",
        ]

    def test_wp_is_unchanged_at_the_values_edge_tier_already_uses(self) -> None:
        """WP labels must not move AT ALL.

        That is what keeps the 23-point ``_EDGE_TIER_SNAPSHOT``, recorded BEFORE the Phase-31
        helper collapse, valid as pre-collapse regression evidence instead of a rewrite with a
        new expectation. Bound to ``utils.edge_tier``'s own constants rather than to literals,
        so a change to either side fails here.
        """
        assert tuple(cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET["wp"]) == (
            edge_tier.EDGE_TIER_HIGH_THRESHOLD,
            edge_tier.EDGE_TIER_MEDIUM_THRESHOLD,
        )
        assert tuple(cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET["wp"]) == (
            0.05,
            0.02,
        )

    def test_every_pair_orders_high_above_medium(self) -> None:
        """A pair whose "high" sat below its "medium" would make the middle band empty."""
        for target, (
            high,
            medium,
        ) in cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET.items():
            assert high > medium > 0, f"{target}: ({high}, {medium})"

    def test_the_module_and_the_witness_agree_on_all_six(self) -> None:
        """Two homes, one set of values; a drift between them fails here, not silently."""
        assert dict(phase33_state.EDGE_THRESHOLDS_FROZEN) == {
            target: tuple(pair)
            for target, pair in cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET.items()
        }

    def test_all_six_are_stated_to_four_decimal_places_in_the_committed_text(
        self,
    ) -> None:
        """The pre-registered PRECISION, checked against the source rather than a float repr.

        ``len(str(value).split('.')[-1]) <= 4`` accepts ``0.05`` and proves only at-most
        precision. ``Decimal`` over the literal TEXT is what distinguishes ``0.05`` from
        ``0.0500``, and four decimal places in the committed bytes is what was frozen.
        """
        source = (REPO_ROOT / "backtest/cold_start_constants.py").read_text(
            encoding="utf-8"
        )
        block = source.split("EDGE_TIER_THRESHOLDS_BY_TARGET", 1)[1].split("\n\n", 1)[0]
        literals = re.findall(r"(?<![\w.])(\d+\.\d+)", block)
        assert len(literals) == 6, (
            f"expected exactly six threshold literals in the committed text, found {literals}."
        )
        for literal in literals:
            assert abs(Decimal(literal).as_tuple().exponent) == 4, (
                f"{literal} is not stated to FOUR decimal places in the committed source. The "
                "pre-registered property is four places, not at most four."
            )

    def test_each_threshold_pair_carries_the_unit_it_belongs_to(self) -> None:
        """A threshold without a stated unit is a number nobody can check.

        Applying ONE pair to three incompatible units is the defect (DEF-31-17) this
        pre-registration exists to retire, so the units are not decoration.
        """
        units = cold_start_constants.EDGE_TIER_THRESHOLD_UNITS
        assert set(units) == set(cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET)
        assert "points" in units["ats"]
        assert "ratio" in units["ou"]
        assert "probability" in units["wp"]

    def test_the_allowance_is_still_zero_beside_the_thresholds(self) -> None:
        """The one assertion that would catch the frozen module drifting from Plan 33-08."""
        assert cold_start_constants.FIX_CYCLE_ALLOWANCE == 0
        assert (
            cold_start_constants.FIX_CYCLE_ALLOWANCE
            == phase33_state.FIX_CYCLE_ALLOWANCE
        )
        assert (
            runner.PHASE33_FIX_CYCLE_ALLOWANCE
            == cold_start_constants.FIX_CYCLE_ALLOWANCE
        )


class TestThe2026ChainFitBiasIsFrozenWithItsProvenance:
    """D33-03 / D33-21: one estimator, a strictly-prior pool, a stated source per target."""

    def test_the_bias_covers_all_three_targets_and_is_not_none(self) -> None:
        assert sorted(cold_start_constants.CHAIN_FIT_BIAS_2026) == ["ats", "ou", "wp"]
        for target, value in cold_start_constants.CHAIN_FIT_BIAS_2026.items():
            assert isinstance(value, float), target
            assert not math.isnan(value), f"{target}: the bias is NaN"

    def test_the_pool_is_strictly_prior_and_covers_2021_through_2025(self) -> None:
        assert cold_start_constants.CHAIN_FIT_BIAS_SEASONS == (
            2021,
            2022,
            2023,
            2024,
            2025,
        )
        assert 2026 not in cold_start_constants.CHAIN_FIT_BIAS_SEASONS, (
            "the 2026 bias must NEVER be estimated from 2026's own data. The walk-forward "
            "rule is strictly-prior seasons only."
        )

    def test_the_module_and_the_witness_agree_on_the_bias(self) -> None:
        assert dict(phase33_state.CHAIN_FIT_BIAS_2026_FROZEN) == dict(
            cold_start_constants.CHAIN_FIT_BIAS_2026
        )

    def test_each_target_records_whether_it_came_from_a_refit_or_an_incumbent(
        self,
    ) -> None:
        """A promoted re-fit with a non-PASS verdict is visible AS THAT, not smoothed away.

        The gate FAILED ATS and O/U and the owner promoted them anyway under a recorded
        override. The verdict was never softened to match the ruling, and this is where a
        reader meets both facts at once.
        """
        sources = cold_start_constants.CHAIN_FIT_BIAS_SOURCE_BY_TARGET
        assert sorted(sources) == ["ats", "ou", "wp"]
        for target, row in sources.items():
            assert row["kind"] in ("promoted_refit", "retained_incumbent"), target
            assert row["verdict"] in ("PASS", "FAIL", "UNTESTABLE_REFUSAL"), target
            assert (
                row["artifact"] == phase33_state.POST_GATE_ARTIFACT_MANIFEST[target]
            ), f"{target}: the residual source is not the DEPLOYED artifact."
            expected = row["kind"] == "promoted_refit" and row["verdict"] != "PASS"
            assert bool(row["promoted_against_verdict"]) is expected, target

    def test_the_empty_pool_refusal_is_stated_rather_than_defaulted(self) -> None:
        """R10 edge: no bias is invented and there is no fallback to the target's own data."""
        doc = " ".join((inspect.getdoc(cold_start_constants) or "").upper().split())
        assert "EMPTY STRICTLY-PRIOR RESIDUAL POOL REFUSES BY NAME" in doc
        assert "NO FALLBACK TO THE TARGET SEASON'S OWN DATA" in doc

    def test_the_json_string_versus_int_key_rule_is_stated(self) -> None:
        """JSON round-trips season keys as strings while the lookup is by int."""
        doc = " ".join((inspect.getdoc(cold_start_constants) or "").upper().split())
        assert (
            "JSON ROUND-TRIPS SEASON KEYS AS STRINGS WHILE THE LOOKUP IS BY INT" in doc
        )

    def test_a_json_round_trip_of_the_bias_finds_2026_under_a_string_key(self) -> None:
        """The stated rule, exercised rather than only described."""
        payload = json.dumps({str(2026): cold_start_constants.CHAIN_FIT_BIAS_2026})
        restored = json.loads(payload)
        assert list(restored) == ["2026"]
        assert restored.get(str(2026)) is not None
        assert restored.get(2026) is None, (
            "a JSON-reloaded mapping answers to the STRING key only; a consumer looking up "
            "the int would silently miss, which is why the rule is stated in the module."
        )


class TestTheDerivationCircularityIsStatedNotDiscovered:
    """D33-20: the caveat is kept and named, rather than traded for an unstated mismatch."""

    def test_the_module_states_the_circularity(self) -> None:
        doc = (inspect.getdoc(cold_start_constants) or "").lower()
        assert "circular" in doc

    def test_the_population_names_the_seasons_it_was_derived_on(self) -> None:
        population = cold_start_constants.THRESHOLD_DERIVATION_POPULATION
        assert "2021-2024" in population
        assert "d31-04" in population.lower()

    def test_the_document_states_it_too(self) -> None:
        text = (REPO_ROOT / "COLD-START-PREREGISTRATION.md").read_text(encoding="utf-8")
        assert "circular" in text.lower()


class TestTheLabelMovementIsRecordedAsCountsAndNotOnlyAsShares:
    """A share rounds; a count does not. Both are recorded so neither can stand alone."""

    def test_the_after_shares_cover_all_three_targets(self) -> None:
        after = phase33_state.ATS_BAND_SHARES_AFTER
        assert sorted(after) == ["ats", "ou", "wp"]
        for target, row in after.items():
            assert sorted(row) == ["high", "low", "medium"], target
            assert abs(sum(row.values()) - 1.0) < 0.001, target

    def test_the_witness_and_the_module_agree_on_the_after_shares(self) -> None:
        assert dict(phase33_state.ATS_BAND_SHARES_AFTER) == {
            target: dict(row)
            for target, row in cold_start_constants.ATS_BAND_SHARES_AFTER.items()
        }

    def test_zero_wp_games_change_band(self) -> None:
        """By construction -- WP's thresholds do not move -- and asserted, not assumed."""
        assert cold_start_constants.GAMES_CHANGING_BAND["wp"] == 0

    def test_the_ats_and_ou_movements_are_recorded_as_game_counts(self) -> None:
        moving = cold_start_constants.GAMES_CHANGING_BAND
        assert moving["ats"] > 0 and moving["ou"] > 0
        eligible = cold_start_constants.DERIVATION_ELIGIBLE_COUNTS["ats"][
            "threshold_rows"
        ]
        assert moving["ats"] <= eligible
        assert moving["ou"] <= eligible

    def test_the_recorded_edge_tier_consumers_match_a_live_source_scan(self) -> None:
        """The movement changes a PRINTED LABEL and changes NO BET -- scanned, not asserted.

        This is the fact that made accepting 522 moved ATS labels a narrow decision rather
        than a broad one, so it is guarded rather than left in a comment. The scan also
        caught the first version of the record naming ONE consumer when there are TWO;
        understating the reach of a published-label change is the direction an honest record
        must not err in, which is why the correction is recorded beside the original.
        """
        recorded = set(
            phase33_state.EDGE_TIER_DISPLAY_ONLY_CORRECTED["non_test_consumers"]
        )
        found = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob("*.py")
            if ".venv" not in path.parts
            and "tests" not in path.parts
            and path.relative_to(REPO_ROOT).as_posix() != "utils/edge_tier.py"
            and re.search(
                r"^from utils\.edge_tier import", path.read_text(encoding="utf-8"), re.M
            )
        }
        assert found == recorded, (
            f"the recorded edge_tier consumers {sorted(recorded)} disagree with a live source "
            f"scan {sorted(found)}. A NEW consumer means the band's reach has changed and the "
            "display-only conclusion must be re-established, not inherited."
        )

    def test_no_bet_selection_path_reads_the_edge_band(self) -> None:
        """The other half of the same claim: selection bands by ev_tier, never by edge_tier."""
        for module in (
            "backtest/weekly_bet_list.py",
            "backtest/selector_strategies.py",
            "scripts/generate_bet_list.py",
        ):
            source = (REPO_ROOT / module).read_text(encoding="utf-8")
            assert "from utils.edge_tier import" not in source, module
            assert "edge_tier_series(" not in source, module

    def test_every_band_count_sums_to_the_eligible_row_count(self) -> None:
        """The counts are a partition of the population, not a sample of it."""
        for target, row in cold_start_constants.ATS_BAND_COUNTS_AFTER.items():
            expected = cold_start_constants.DERIVATION_ELIGIBLE_COUNTS[target][
                "threshold_rows"
            ]
            assert sum(row.values()) == expected, target
