"""Both halves of each chain's positive control ran, and neither one entered the correction.

WHY A ZERO NEEDS A CONTROL AT ALL
---------------------------------
The 2025 verdict rests on a claim about the ABSENCE of an edge, and absence has two possible
causes that look identical from the outside: there is genuinely no edge, or the chain is broken.
A missing column, a NaN, a silently empty merge and a real efficient market all produce the same
number. So a zero -- or a thin, unimpressive result -- is only interpretable if something on the
SAME frame can be shown to have moved the selector.

THE SYNTHETIC UNIT CONTROL IS NOT ENOUGH, AND THAT IS THE WHOLE POINT
---------------------------------------------------------------------
A control on a hand-built frame passes even when the real frame is broken: it proves the
mechanism, not the wiring that fed 2025 into it. So there are TWO halves, and this module asserts
both:

* **The synthetic unit control.** ``_shift_edge_for_control`` manufactures a positive edge and the
  selector admits it, on frames this repository built. Exercised through the shared
  ``p31_rehearsal_run`` fixture -- the DISJOINT rehearsal window (tune 2021-2023, hold 2024) over
  the synthetic fixture, which refuses to generate 2025 at all. This half proves the CONTROL
  MECHANISM is live.
* **The shifted-edge counterfactual over the ACTUAL 2025 frame.** Run inside the SAME one-shot
  invocation that produced the verdict, on the same rows, and recorded in the committed artifact.
  This half proves the frame the verdict was measured on can be moved. It is the half that makes a
  2025 zero mean "no edge" rather than "unexplained".

Both halves are asserted PER TARGET. A control that passed for two targets out of three defends
two thirds of the verdict, and the third would still be an unexplained number.

WHY THE COUNTERFACTUAL IS RECORDED BUT NOT CORRECTED
-----------------------------------------------------
Every read of 2025 is on the record: the counterfactual gets its own registry row, so nobody can
later claim the hold was read a third time in silence. But it carries ``entry_kind = control`` and
is EXCLUDED from the Benjamini-Hochberg denominator, because it has no p-value to correct.
Counting it would enlarge the family and make the real verdict harder to reach -- statistically
penalising the measurement for running a code-liveness check. Registry row count and denominator
therefore differ, visibly and by design, and the difference is asserted arithmetically here rather
than explained in a comment.

WHAT THIS MODULE READS
----------------------
The COMMITTED artifact at ``config/profitability_2025_verdict.toml``, which is git-tracked and
travels with the repository. It does NOT re-run the 2025 chain and could not: the split is spent,
the ledger reads ``completed``, and the runner refuses both on the ledger and on the existing
artifact. A control module that re-measured the thing it audits would be a second read of a
single-use season.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import pytest

from backtest.ev_chain_constants import (
    TRIAL_ENTRY_KIND_CONTROL,
    TRIAL_ENTRY_KIND_INFERENCE,
    VERDICT_TOKENS,
)
from backtest.profitability_2025 import (
    CANONICAL_TARGETS,
    UNDISCHARGEABLE_NO_BETS,
    bh_family_members,
    registry_exclusion_counts,
)
from tests import phase31_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# The registry entry id the runner gives the counterfactual pass, per target.
CONTROL_ENTRY_ID = "{target}/control/shifted_edge"


@pytest.fixture(scope="module")
def committed_verdict() -> dict[str, Any]:
    """The COMMITTED 2025 verdict artifact, parsed.

    NOT an evidence-backed skip. The artifact is git-tracked precisely so the binding figures
    survive a fresh clone; if it is absent, the checkout is broken and this module must say so
    rather than quietly not running.
    """
    path = REPO_ROOT / phase31_state.VERDICT_PATH
    assert path.is_file(), (
        f"the committed verdict artifact {phase31_state.VERDICT_PATH} is missing. It is TRACKED "
        "and the 2025 split is single-use, so it cannot be regenerated."
    )
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hold_window_label(committed_verdict: dict[str, Any]) -> str:
    """The hold window label the registry rows are tagged with, read off the artifact."""
    return "hold_" + "_".join(
        str(season) for season in committed_verdict["run"]["hold_seasons"]
    )


def _entries(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    return list(verdict["registry"])


def _control_entry(verdict: dict[str, Any], target: str) -> dict[str, Any]:
    wanted = CONTROL_ENTRY_ID.format(target=target)
    matches = [entry for entry in _entries(verdict) if entry["entry_id"] == wanted]
    assert len(matches) == 1, (
        f"expected exactly one counterfactual control entry {wanted!r} in the committed "
        f"registry, found {len(matches)}."
    )
    return matches[0]


class TestTheSyntheticUnitControlHalf:
    """The control MECHANISM is live: a manufactured edge is admitted on frames we built."""

    def test_every_target_records_a_synthetic_control_that_selected_bets(
        self, p31_rehearsal_run
    ) -> None:
        """On the DISJOINT rehearsal split, the counterfactual selects for all three targets.

        This runs on synthetic frames spanning 2018-2024; the fixture that builds them REFUSES to
        generate 2025. It proves ``_shift_edge_for_control`` plus the selection path can turn a
        manufactured edge into admitted bets -- which is what makes the 2025 half's silence, if it
        were ever silent, attributable to the frame rather than to the mechanism.
        """
        registry = p31_rehearsal_run["result"]["registry"]
        for target in CANONICAL_TARGETS:
            wanted = CONTROL_ENTRY_ID.format(target=target)
            matches = [entry for entry in registry if entry["entry_id"] == wanted]
            assert len(matches) == 1, (
                f"the rehearsal registry has {len(matches)} entries for {wanted!r}, expected 1."
            )
            entry = matches[0]
            assert entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL, entry
            assert entry["bet_count"] > 0, (
                f"[{target}] the SYNTHETIC unit control selected zero bets. A manufactured "
                "positive edge that the selector will not admit means the control mechanism "
                "itself is dead, and every zero it is supposed to defend is undefended."
            )

    def test_the_synthetic_control_is_recorded_but_never_corrected(
        self, p31_rehearsal_run
    ) -> None:
        """The exclusion rule is a property of the CODE, not of the 2025 artifact alone."""
        result = p31_rehearsal_run["result"]
        label = result["run"]["hold_window_label"]
        family_ids = {
            entry["entry_id"] for entry in bh_family_members(result["registry"], label)
        }
        for target in CANONICAL_TARGETS:
            assert CONTROL_ENTRY_ID.format(target=target) not in family_ids


class TestTheCounterfactualOverTheActual2025Frame:
    """The half that defends the real numbers: the 2025 frame itself can be moved."""

    def test_every_target_records_a_counterfactual_over_the_hold_frame(
        self, committed_verdict: dict[str, Any], hold_window_label: str
    ) -> None:
        """Per target, one control row, tagged with the HOLD window and not the tune window."""
        for target in CANONICAL_TARGETS:
            entry = _control_entry(committed_verdict, target)
            assert entry["target"] == target, entry
            assert entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL, entry
            assert entry["sample_window"] == hold_window_label, (
                f"[{target}] the counterfactual is tagged {entry['sample_window']!r}, not the "
                f"hold window {hold_window_label!r}. A control run on the TUNE frame proves "
                "nothing about the frame the verdict was measured on -- which is the entire "
                "reason this half exists."
            )

    def test_the_counterfactual_moved_the_selector_for_every_target(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """A manufactured edge admitted at least one bet on the ACTUAL 2025 rows, per target.

        This is the assertion the verdict's interpretability rests on. If it failed for a target,
        that target's result would be an unexplained number rather than a measurement, and the
        runner would have assigned ``UNDISCHARGEABLE_NO_CHAIN`` on a zero rather than
        ``UNDISCHARGEABLE_NO_BETS``.
        """
        for target in CANONICAL_TARGETS:
            entry = _control_entry(committed_verdict, target)
            record = committed_verdict["targets"][target]
            assert entry["bet_count"] > 0, (
                f"[{target}] the shifted-edge counterfactual selected ZERO bets on the actual "
                "2025 frame. Nothing on that frame is then known to be able to move the "
                "selector, so the measured result cannot be distinguished from a broken chain."
            )
            assert record["control_passed"] is True, record
            assert record["control_bet_count"] == entry["bet_count"], (
                f"[{target}] the per-target record says the control selected "
                f"{record['control_bet_count']} bets and the registry row says "
                f"{entry['bet_count']}. Two numbers for one fact."
            )

    def test_the_counterfactual_and_the_verdict_came_from_ONE_invocation(
        self, committed_verdict: dict[str, Any], hold_window_label: str
    ) -> None:
        """One attempt, one registry, both passes -- not a control bolted on afterwards.

        A counterfactual produced by a second, later invocation would be a control over a second
        read of the hold, and it would defend a frame nobody can now prove was the same one. The
        artifact carries a single ``attempt_id``, and the ledger it was committed with carries
        that same id in state ``completed``.
        """
        run = committed_verdict["run"]
        assert run["window_is_the_preregistered_rule"] is True, run
        attempt = run["attempt_id"]
        assert attempt, "the artifact records no attempt id"

        ledger = tomllib.loads(
            (REPO_ROOT / phase31_state.RUN_LEDGER_COMMITTED_PATH).read_text(
                encoding="utf-8"
            )
        )
        assert ledger["state"] == "completed", ledger["state"]
        assert ledger["attempt_id"] == attempt, (
            "the ledger and the verdict artifact name DIFFERENT attempts "
            f"({ledger['attempt_id']!r} vs {attempt!r}), so the record of what was spent does "
            "not describe what was measured."
        )

        # Primary and control for every target live in this ONE registry.
        ids = {entry["entry_id"] for entry in _entries(committed_verdict)}
        for target in CANONICAL_TARGETS:
            assert f"{target}/primary" in ids
            assert CONTROL_ENTRY_ID.format(target=target) in ids
        holds = {
            entry["sample_window"]
            for entry in _entries(committed_verdict)
            if entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL
        }
        assert holds == {hold_window_label}, holds


class TestControlsAreOnTheRecordAndOutOfTheFamily:
    """Recorded, and excluded. Both, and the difference is arithmetic rather than argued."""

    def test_every_control_entry_is_absent_from_the_correction_family(
        self, committed_verdict: dict[str, Any], hold_window_label: str
    ) -> None:
        family = bh_family_members(_entries(committed_verdict), hold_window_label)
        family_ids = {entry["entry_id"] for entry in family}
        for entry in _entries(committed_verdict):
            if entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL:
                assert entry["entry_id"] not in family_ids, (
                    f"the control entry {entry['entry_id']!r} entered the BH family. A control "
                    "has no p-value to correct, so counting it would enlarge the denominator "
                    "and penalise the verdict for running a liveness check."
                )
        assert all(
            entry["entry_kind"] == TRIAL_ENTRY_KIND_INFERENCE for entry in family
        )

    def test_no_control_entry_carries_a_p_value_of_any_kind(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """A control has nothing to correct, so it must carry neither a raw nor an adjusted p.

        Rendered as the TOML float ``nan`` rather than omitted, because the artifact's shape does
        not change with the data -- an absent value and a forgotten field would otherwise be
        indistinguishable.
        """
        for target in CANONICAL_TARGETS:
            entry = _control_entry(committed_verdict, target)
            for field in ("raw_p", "adjusted_p"):
                value = entry[field]
                assert isinstance(value, float) and math.isnan(value), (
                    f"[{target}] the control entry carries {field}={value!r}. A control is not "
                    "an inference and has no p-value; a number here would eventually be read as "
                    "one."
                )

    def test_the_row_count_accounts_for_every_exclusion_arithmetically(
        self, committed_verdict: dict[str, Any], hold_window_label: str
    ) -> None:
        """``rows == family + controls + tune-side cells``, recomputed from the artifact itself.

        The identity is what stops the gap between the registry's size and the denominator being
        read as a bug. Recomputed here rather than trusting the run block, then compared against
        it, so the two would have to be wrong in the same way to agree.
        """
        run = committed_verdict["run"]
        recomputed = registry_exclusion_counts(
            _entries(committed_verdict), hold_window_label
        )
        assert recomputed["registry_rows"] == run["registry_rows"]
        assert recomputed["bh_family_size"] == run["bh_denominator"]
        assert recomputed["excluded_control_entries"] == run["excluded_control_entries"]
        assert (
            recomputed["excluded_tune_side_sweep_cells"]
            == run["excluded_tune_side_sweep_cells"]
        )
        assert recomputed["registry_rows"] == (
            recomputed["bh_family_size"]
            + recomputed["excluded_control_entries"]
            + recomputed["excluded_tune_side_sweep_cells"]
        ), recomputed

    def test_there_is_exactly_one_control_per_target_and_no_more(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """Three targets, three controls. A fourth would be a read of 2025 nobody declared."""
        controls = [
            entry
            for entry in _entries(committed_verdict)
            if entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL
        ]
        assert len(controls) == len(CANONICAL_TARGETS), [
            entry["entry_id"] for entry in controls
        ]
        assert sorted(entry["target"] for entry in controls) == sorted(
            CANONICAL_TARGETS
        )


class TestEveryTargetsResultIsDefendedOnTheFrameItWasMeasuredOn:
    """The join between the control and the token, target by target."""

    def test_every_token_is_from_the_frozen_vocabulary(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        for target in CANONICAL_TARGETS:
            token = committed_verdict["targets"][target]["verdict_token"]
            assert token in VERDICT_TOKENS, (
                f"[{target}] the token {token!r} is not in the frozen vocabulary "
                f"{list(VERDICT_TOKENS)}."
            )

    def test_a_zero_bet_target_carries_the_no_bets_token_and_a_passing_control(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """Whatever the data said, a zero is only a RESULT when the control passed on that frame.

        Written to bind in both directions rather than to describe what this particular run
        happened to produce: a target with zero bets must carry the no-bets token AND no return,
        and a target carrying the no-bets token must have selected zero bets.
        """
        for target in CANONICAL_TARGETS:
            record = committed_verdict["targets"][target]
            if record["bets_selected"] == 0:
                assert record["control_passed"] is True, record
                assert record["verdict_token"] == UNDISCHARGEABLE_NO_BETS, record
                assert record["has_return"] is False, record
                assert math.isnan(record["flat_roi"]), (
                    f"[{target}] zero bets was reported with a return of "
                    f"{record['flat_roi']!r}. A return of zero is a measured break-even; no bets "
                    "is not, and the two must never render alike."
                )
            if record["verdict_token"] == UNDISCHARGEABLE_NO_BETS:
                assert record["bets_selected"] == 0, record

    def test_a_reported_return_is_backed_by_bets_and_a_passing_control(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """A return is only reported when bets exist, and only defended when the control passed."""
        for target in CANONICAL_TARGETS:
            record = committed_verdict["targets"][target]
            if record["has_return"]:
                assert record["bets_selected"] > 0, record
                assert not math.isnan(record["flat_roi"]), record
                assert record["control_passed"] is True, (
                    f"[{target}] a return is reported but the counterfactual over the same 2025 "
                    "frame did NOT pass. The number is then unattributable."
                )

    def test_the_clv_figures_are_carried_beside_the_verdict_and_never_inside_it(
        self, committed_verdict: dict[str, Any]
    ) -> None:
        """Report-only, structurally: the CLV p-value is disclosed and drove no token.

        The firewall itself is proven in ``tests/unit/test_verdict_tokens.py`` (a frozen
        measurement dataclass with no CLV field, plus an AST scan of the verdict call graph).
        What this asserts is the ARTIFACT side of the same rule: the figures are present, under
        names that say what they are, and the run block declares the policy.
        """
        assert committed_verdict["run"]["clv_p_value_is_report_only"] is True
        for target in CANONICAL_TARGETS:
            record = committed_verdict["targets"][target]
            for field in (
                "clv_report_only_mean",
                "clv_report_only_p",
                "clv_report_only_n",
            ):
                assert field in record, (
                    f"[{target}] the report-only CLV field {field!r} is missing. Suppressing it "
                    "would be as dishonest as letting it drive the verdict."
                )
