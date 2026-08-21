"""The BINDING Stage-1 feature-group judge for Phase 30 (SPEC R4, PROD-01).

This module decides; it measures nothing. ``backtest/signal_lift.py`` produces the paired
add-one-in CLV deltas and ``backtest/diagnose.py`` owns the significance primitive, the per-target
CLV column map and alpha. The judge imports both and adds ONLY the multiplicity correction and the
verdict -- exactly the relationship ``models/deploy_gate.py`` has to ``backtest/diagnose.py``
(D24-13), applied one level down at the feature-group grain (D30-04).

The SCREEN and the JUDGE are deliberately separate modules. ``signal_lift.decide_group_keep`` is
the PERMISSIVE Phase-28 D-05 screen rule (keep unless significantly-negative, grid reported raw,
correction explicitly deferred to this phase). ``decide_group_verdicts`` below REPLACES it for the
binding decision -- it does not extend it. Keeping the two rules in one module is the
screen-versus-deploy conflation this phase exists to end.

The frozen rule itself lives in ``backtest/group_gate_constants.py`` and is one-way after the
pre-registration commit. The mechanics HERE stay editable by design: that split is the whole
reason the constants module can carry a stable git-ancestry assertion (D30-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from typing import Any

from scipy.stats import false_discovery_control

# Import-the-primitive parity seam (D24-13, retargeted from models/deploy_gate.py:95-102). These
# symbols are IMPORTED, never re-declared here: this module re-derives NO metric. It must judge on
# the exact same CLV column, sample-size floor and significance level the honest diagnosis uses,
# or the screen and the gate could silently diverge (Pitfall 2, T-30-15).
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    MIN_CLV_SAMPLE,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from backtest.group_gate_constants import (
    ALPHA,
    BH_DENOMINATOR,
    CORRECTION_METHOD,
    FIX_CYCLE_LEVERS,
    GRID_GROUPS,
    GRID_TARGETS,
    MDE_POWER,
    VERDICT_DROP,
    VERDICT_KEEP,
    VERDICT_NOT_MEASURED,
    VERDICT_UNDETERMINED,
    bh_family_exclusion_reason,
    bh_rank_order,
    bh_reject,
    mde_for_cell,
)

# The measurement side. ``ALL_REGISTERED_GROUPS`` is the load-bearing baseline pin the Stage-1
# orchestrator must pass (the module default is the three-name Phase-28 deny-list, which would
# silently measure something else); the other three are the screen entry points the orchestrator
# calls. The orchestrating ``run_group_gate`` and its CLI are Plan 30-05's, deliberately NOT this
# plan's -- mechanics that may later need a fix belong outside the frozen pre-registration, and
# the seam they bind to is fixed here so the judge's dependency surface is part of what the owner
# ratifies.
from backtest.signal_lift import (
    ALL_REGISTERED_GROUPS,
    group_columns,
    run_signal_lift_screen,
    select_group_columns,
)

__all__ = [
    "ALL_REGISTERED_GROUPS",
    "ALPHA",
    "BH_DENOMINATOR",
    "CLV_COLUMN_FOR",
    "CORRECTION_METHOD",
    "FIX_CYCLE_LEVERS",
    "GRID_GROUPS",
    "GRID_TARGETS",
    "MDE_POWER",
    "SIGNIFICANCE_ALPHA",
    "VERDICT_DROP",
    "VERDICT_KEEP",
    "VERDICT_NOT_MEASURED",
    "VERDICT_UNDETERMINED",
    "clv_significance",
    "decide_group_verdicts",
    "group_columns",
    "run_signal_lift_screen",
    "select_group_columns",
]


def _usable_pvalue(cell: dict[str, Any]) -> float | None:
    """The raw two-sided p, or None when it is absent or NaN.

    NaN is DETECTED, never coerced. A zero-variance cell (both legs identical because the group's
    columns were all discarded by feature selection) returns ``t=nan, p=nan`` from
    ``ttest_1samp`` with no warning emitted, and both ``nan <= alpha`` and
    ``false_discovery_control`` handle that badly -- the first silently never rejects, the second
    raises. See T-30-18.
    """
    pvalue = cell.get("delta_p")
    if pvalue is None:
        return None
    pvalue = float(pvalue)
    if math.isnan(pvalue):
        return None
    return pvalue


def decide_group_verdicts(screen_result: dict[str, Any]) -> dict[str, Any]:
    """Apply the FROZEN Stage-1 rule to a signal-lift screen output. PURE.

    A pure function of the screen output, so every arm and edge case is unit-testable on a
    synthetic grid without running two walk-forward trainers per cell.

    Three passes:

    1. **Measurability.** Each (group, target) cell is MEASURED or NOT MEASURED under the frozen
       ``bh_family_exclusion_reason`` predicate, whose check order is itself pre-registered --
       ``n_group_columns == 0`` is tested before the p-value is inspected at all (SPEC R4). The
       cell's own ``paired_sufficient`` and ``measurable`` accounting is CONSUMED, never
       re-derived: ``signal_lift`` already models the small-sample refusal and
       ``clv_significance`` already owns it.
    2. **Correction.** The BH family is assembled from the MEASURED cells ONLY, with ``m`` equal
       to that count, keys ``(group, target)`` and p-values the raw two-sided p taken AS-IS
       (D30-09 -- a two-sided p is roughly twice the one-sided p, so this is strictly
       conservative, and it honours SPEC R4's "no bespoke re-derivation" literally). scipy's
       adjusted q-values are emitted alongside, labelled display-only; the binding decision is the
       vendored inclusive step-up.
    3. **Verdict.** Per group, the four arms are applied in the pre-registered order KEEP, DROP,
       UNDETERMINED, NOT MEASURED. Direction is read from the sign of ``delta_mean`` exactly as
       ``backtest/diagnose.py`` lines 317-322 do, using a STRICT ``> 0.0`` comparison with no
       epsilon anywhere. NOTE the arm order is load-bearing where a group is BH-rejected in both
       directions: KEEP wins, which REVERSES the permissive Phase-28 veto. The DROP arm is the
       simpler pre-registered form -- BH-rejected negative on any target, OR no positive point
       estimate on any target -- so the rule contains no judgement call about which effect sizes
       are "of interest"; every positive-but-unrejected case routes to UNDETERMINED, including one
       exactly equal to its cell's MDE.

    Args:
        screen_result: A ``run_signal_lift_screen`` return dict.

    Returns:
        ``{"screen": <the untouched input>, "verdicts": {group: {...}}, ...}``. The raw screen is
        passed through unmodified so the readout can publish the raw grid beside the corrected
        one, mirroring ``evaluate_target``'s discipline that every metric field was produced
        upstream and the judge adds only the decision.
    """
    groups_in: dict[str, Any] = screen_result.get("groups", {})

    # --- pass 1: measurability -------------------------------------------------------------
    exclusions: dict[tuple[str, str], str | None] = {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for group, group_out in groups_in.items():
        for target, cell in group_out.get("per_target", {}).items():
            key = (group, target)
            cells[key] = cell
            exclusions[key] = bh_family_exclusion_reason(cell)

    # --- pass 2: correction ----------------------------------------------------------------
    measured_keys = sorted(key for key, reason in exclusions.items() if reason is None)
    pvalues = [float(cells[key]["delta_p"]) for key in measured_keys]
    denominator = len(measured_keys)
    rejected = bh_reject(pvalues, measured_keys)
    rank_of = {
        key: rank
        for rank, key in enumerate(bh_rank_order(pvalues, measured_keys), start=1)
    }
    if pvalues:
        q_display = {
            key: float(q)
            for key, q in zip(
                measured_keys, false_discovery_control(pvalues, method="bh")
            )
        }
    else:
        q_display = {}

    # --- pass 3: verdict -------------------------------------------------------------------
    verdicts: dict[str, Any] = {}
    for group, group_out in groups_in.items():
        group_cells: dict[str, Any] = {}
        for target, cell in group_out.get("per_target", {}).items():
            key = (group, target)
            reason = exclusions[key]
            measured = reason is None
            group_cells[target] = {
                "target": target,
                "clv_column": cell.get("clv_column", CLV_COLUMN_FOR.get(target)),
                "measured": measured,
                "exclusion_reason": reason,
                "n_paired": cell.get("n_paired"),
                "n_group_columns": cell.get("n_group_columns"),
                "n_group_columns_selected": cell.get("n_group_columns_selected"),
                "delta_mean": cell.get("delta_mean"),
                "delta_p": _usable_pvalue(cell),
                "delta_ci95": cell.get("delta_ci95"),
                "bh_rejected": bool(rejected.get(key, False)),
                "bh_rank": rank_of.get(key),
                "q_display": q_display.get(key),
                "mde": mde_for_cell(
                    {"n": cell.get("n_paired", 0), "ci95": cell.get("delta_ci95")}
                ),
                "selection_churn": bool(
                    int(cell.get("n_group_columns", 0)) > 0
                    and int(cell.get("n_group_columns_selected", 0)) == 0
                ),
            }

        measured_targets = sorted(t for t, c in group_cells.items() if c["measured"])
        positive_targets = sorted(
            t
            for t in measured_targets
            if group_cells[t]["delta_mean"] is not None
            and group_cells[t]["delta_mean"] > 0.0
        )
        rejected_positive = [
            t for t in positive_targets if group_cells[t]["bh_rejected"]
        ]
        rejected_negative = sorted(
            t
            for t in measured_targets
            if group_cells[t]["bh_rejected"]
            and group_cells[t]["delta_mean"] is not None
            and group_cells[t]["delta_mean"] < 0.0
        )

        largest_positive: dict[str, Any] | None = None
        if positive_targets:
            best = max(positive_targets, key=lambda t: group_cells[t]["delta_mean"])
            mde = group_cells[best]["mde"]
            largest_positive = {
                "target": best,
                "delta_mean": group_cells[best]["delta_mean"],
                "mde": mde,
                # Inclusive at the MDE: an effect exactly equal to its cell's MDE lands INSIDE the
                # UNDETERMINED band (SPEC R4 boundary).
                "at_or_below_mde": (
                    None
                    if mde is None
                    else bool(group_cells[best]["delta_mean"] <= mde)
                ),
            }

        if not measured_targets:
            verdict = VERDICT_NOT_MEASURED
            reasons = sorted(
                {
                    r
                    for t, c in group_cells.items()
                    if (r := c["exclusion_reason"]) is not None
                }
            )
            reason_text = (
                f"NOT MEASURED: no cell of this group entered the BH family ({'; '.join(reasons)}). "
                "This is a refusal to rule, NOT a DROP -- no evidence about this group was "
                f"produced. The CLV sample-size floor is MIN_CLV_SAMPLE={MIN_CLV_SAMPLE}."
            )
        elif rejected_positive:
            verdict = VERDICT_KEEP
            reason_text = (
                f"KEEP: significantly positive after Benjamini-Hochberg correction on "
                f"{rejected_positive} at alpha={ALPHA} with m={denominator} "
                f"({BH_DENOMINATOR} family). Carried into the Stage-2 candidate feature set."
            )
        elif rejected_negative:
            verdict = VERDICT_DROP
            reason_text = (
                f"DROP: significantly NEGATIVE after correction on {rejected_negative}; "
                "this group measurably hurt a target and is dropped, not silently retained."
            )
        elif not positive_targets:
            verdict = VERDICT_DROP
            reason_text = (
                "DROP: no positive point estimate on any measured target (a delta of exactly 0.0 "
                f"is not positive). Measured targets: {measured_targets}."
            )
        else:
            verdict = VERDICT_UNDETERMINED
            reason_text = (
                f"UNDETERMINED: positive point estimate on {positive_targets} but no cell "
                f"survived Benjamini-Hochberg correction at alpha={ALPHA} with m={denominator}. "
                "Resolves to DROP for the deploy decision and is REPORTED as UNDETERMINED -- "
                "'we could not tell', not 'it did not help'."
            )

        verdicts[group] = {
            "verdict": verdict,
            "reason": reason_text,
            "cells": group_cells,
            "measured_targets": measured_targets,
            "positive_targets": positive_targets,
            "rejected_positive_targets": rejected_positive,
            "rejected_negative_targets": rejected_negative,
            "largest_positive_effect": largest_positive,
            "measurability_note": group_out.get("measurability", {}).get("note", ""),
            "coverage_span": group_out.get("coverage_span"),
            "bh_denominator": denominator,
        }

    return {
        "screen": screen_result,
        "verdicts": verdicts,
        "alpha": ALPHA,
        "correction_method": CORRECTION_METHOD,
        "bh_denominator_rule": BH_DENOMINATOR,
        "bh_denominator": denominator,
        "bh_family": list(measured_keys),
        "bh_rank_order": bh_rank_order(pvalues, measured_keys),
        "excluded_cells": {
            f"{group}/{target}": reason
            for (group, target), reason in sorted(exclusions.items())
            if reason is not None
        },
        "mde_power": MDE_POWER,
        "fix_cycle_levers": list(FIX_CYCLE_LEVERS),
        "q_values_are_display_only": True,
    }
