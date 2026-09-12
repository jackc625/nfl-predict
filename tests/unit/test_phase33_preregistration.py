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

from scripts import run_phase33_gate as runner
from tests import phase33_state


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
