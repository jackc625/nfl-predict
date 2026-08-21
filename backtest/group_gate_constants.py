"""The FROZEN Stage-1 pre-registration for the Phase-30 feature-group gate (SPEC R4, PROD-01).

ONE-WAY BY CONSTRUCTION (D30-05, D30-11). This module is the pre-registration itself, not a
description of one. Its LAST-MODIFYING COMMIT is the git-ancestry anchor that SPEC R4 asserts
against: ``tests/unit/test_gated_refit_readout_md.py`` (Plan 30-13) requires that commit to be a
strict git ANCESTOR of the commit recording the Stage-1 measurement output. That assertion is the
only mechanical proof this phase has that the rule was fixed before the numbers existed.

The consequence, stated plainly because it is easy to forget six plans later: EDITING THIS FILE
AFTER THE MEASUREMENT COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. There is no honest
repair path. A value that is wrong here is wrong for the remainder of the phase, and the only
legitimate response is to say so in the readout, not to amend the file. That is why the entire
rule is implemented and unit-tested BEFORE the owner ratifies it (D30-10 checkpoint 1), and why
the mechanics that may later need a fix -- the orchestrator, the CLI, the readout writer -- live
OUTSIDE this module in ``backtest/group_gate.py`` and Plan 30-05's additions. A file that changes
only when the RULE changes is what makes the ancestry assertion stable and meaningful.

What is frozen here: alpha, the MDE formula and its power, the correction method, the BH family
and denominator rule, the 9-cell grid definition, the four-value verdict vocabulary, the
measurement-exclusion rule that determines the denominator, and the permitted fix-cycle lever
list.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scipy import stats

# Import-the-primitive parity seam (D24-13, applied one level down by D30-04). alpha is IMPORTED,
# never re-declared: the group gate must judge at the exact same significance level the honest
# diagnosis and the model deploy gate use, or the three could silently diverge. This mirrors
# models/deploy_gate.py:95-102 from the code side and config/gate.toml:29-30 from the config side.
from backtest.diagnose import SIGNIFICANCE_ALPHA

__all__ = [
    "ALPHA",
    "BH_DENOMINATOR",
    "CORRECTION_METHOD",
    "EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE",
    "EXCLUSION_NO_PVALUE",
    "EXCLUSION_NO_SELECTED_COLUMN",
    "EXCLUSION_REASONS",
    "EXCLUSION_ZERO_COLUMNS_IN_GOLD",
    "FIX_CYCLE_LEVERS",
    "GRID_GROUPS",
    "GRID_TARGETS",
    "MDE_POWER",
    "VERDICT_DROP",
    "VERDICT_KEEP",
    "VERDICT_NOT_MEASURED",
    "VERDICT_UNDETERMINED",
    "bh_family_exclusion_reason",
    "bh_rank_order",
    "bh_reject",
    "mde_for_cell",
]


# ---------------------------------------------------------------------------
# FORKING-PATHS GUARD: the values below are frozen HERE and are NOT adjusted after seeing
# results (the D26-08 / D24-07 pre-registration discipline, mirroring backtest/ou_divergence.py,
# backtest/ou_ev_chain.py and backtest/ou_monetization.py).
#
# backtest/diagnose.py owns the significance primitive, the per-target CLV column map and alpha.
# backtest/signal_lift.py owns the feature-group registry and the paired add-one-in measurement.
# This module adds ONLY the multiplicity correction, the effect-size bar and the verdict
# thresholds. It re-derives no metric and declares no second alpha.
# ---------------------------------------------------------------------------

# D30-08: the significance level is the ONE that already governs every CLV call in this repo.
# Bound by identity to the imported symbol so a test can assert `ALPHA is SIGNIFICANCE_ALPHA`;
# a float literal here would be a second alpha even if it happened to read 0.05 today.
ALPHA: float = SIGNIFICANCE_ALPHA

# D30-07: the MDE is pre-registered as a FORMULA at 80% power, not as a number, so nothing can be
# shopped after the fact and the bar survives the rebuild moving the underlying standard
# deviations. 90% power was considered and rejected as stricter than this phase needs: it would
# widen the UNDETERMINED band without making any KEEP more trustworthy.
MDE_POWER: float = 0.80

# SPEC R4's named correction. Benjamini-Hochberg controls the false-discovery rate, which is the
# right error to control when the question is "did ANY of these groups help ANY of these targets".
CORRECTION_METHOD: str = "benjamini-hochberg"

# The BH family and its denominator, stated ONCE and in exactly one form:
#
#   the FAMILY is the full 9-cell grid -- 3 groups x 3 targets, as opposed to three per-target
#   families of 3 -- and `m` is the count of MEASURED cells within that family, which equals 9
#   when no cell is excluded.
#
# The SPEC's own R4 text says the correction is applied "across the full 9-cell grid". This is the
# STRICTER of the two available choices: at m = 9 the smallest p in the grid must be at or below
# 0.00556 (= 1 * alpha / 9) for ANY rejection to occur, where three per-target families of 3 would
# need only 0.0167. The strictness is the intent. The KEEP rule pools evidence ACROSS targets
# ("significantly positive on >= 1 target after correction"), so the family IS the grid; switching
# to per-target families after seeing that the full grid rejects nothing would be textbook
# rule-shopping, and is exactly what the git-ancestry assertion exists to detect.
BH_DENOMINATOR: str = "full-grid"

# The grid. line_movement is deliberately absent: Plan 30-07 drops its fifteen columns from gold,
# and a group with zero columns in gold reports NOT MEASURED rather than entering the family.
GRID_GROUPS: tuple[str, ...] = ("injury", "snap", "situational")
GRID_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The four-value verdict vocabulary. UNDETERMINED resolves to DROP for the DEPLOY decision and is
# nonetheless REPORTED as UNDETERMINED and never collapsed into DROP in any published record
# (SPEC R4/R8) -- "we could not tell" and "we measured it and it did not help" are different
# findings and the readout must be able to say which one happened. NOT MEASURED is not a verdict
# about the group at all; it is a refusal to rule.
VERDICT_KEEP: str = "KEEP"
VERDICT_DROP: str = "DROP"
VERDICT_UNDETERMINED: str = "UNDETERMINED"
VERDICT_NOT_MEASURED: str = "NOT MEASURED"

# The measurement-exclusion rule. This is pre-registered rather than left to run-time because it
# DETERMINES the BH denominator: every excluded cell reduces `m`, and a denominator chosen after
# seeing which cells were awkward would be the same rule-shopping the correction exists to
# prevent. The denominator rule is the SAME single form stated at BH_DENOMINATOR above -- the
# family is the full 9-cell grid and `m` is the count of MEASURED cells within it, equal to 9 when
# no cell is excluded.
#
# A cell is EXCLUDED from the BH family when, in this order:
EXCLUSION_ZERO_COLUMNS_IN_GOLD: str = "zero columns of this group are present in gold"
EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE: str = (
    "the paired sample is below the CLV sample-size floor"
)
EXCLUSION_NO_PVALUE: str = "the paired delta has no usable p-value (None or NaN)"
EXCLUSION_NO_SELECTED_COLUMN: str = (
    "no column of this group was selected by this target's own feature selection, so the "
    "candidate model never saw the group and the delta is selection churn, not lift"
)
EXCLUSION_REASONS: tuple[str, ...] = (
    EXCLUSION_ZERO_COLUMNS_IN_GOLD,
    EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE,
    EXCLUSION_NO_PVALUE,
    EXCLUSION_NO_SELECTED_COLUMN,
)

# D30-11: the ONE permitted candidate-side fix-cycle lever, named before anyone knows which target
# will fail. A lever committed to in ignorance cannot be shopped; adding one after seeing the
# failure is precisely what the git-ancestry assertion detects. At most one documented
# candidate-side fix-cycle per failing target (SPEC R5).
FIX_CYCLE_LEVERS: tuple[str, ...] = (
    "D25-05 feature-selection train-window widening (widen the SelectFromModel train window to "
    "the incumbent's own selection window, as Phase 25 did for ATS)",
)


# ---------------------------------------------------------------------------
# The vendored inclusive BH step-up
# ---------------------------------------------------------------------------


def bh_rank_order(
    pvalues: Sequence[float],
    keys: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return the pre-registered 1-based rank order of the BH family.

    Ranks ascend by raw p-value, and TIES break on the deterministic ``(group, target)``
    lexicographic secondary key (SPEC R4 adjacency + ordering). The explicit secondary key is
    load-bearing, not decoration: numpy's default ``argsort`` kind is quicksort, which is
    unstable, so the tie order of any library route is not guaranteed. The BH REJECTION SET is
    tie-order-invariant, but the reported RANK is not -- and the readout publishes ranks.

    Args:
        pvalues: Raw two-sided p-values, one per family member, in the same order as ``keys``.
        keys: The ``(group, target)`` identity of each family member.

    Returns:
        The keys, ordered from rank 1 upward.
    """
    if len(pvalues) != len(keys):
        msg = (
            f"bh_rank_order requires pvalues and keys of the same length; got "
            f"{len(pvalues)} and {len(keys)}."
        )
        raise ValueError(msg)
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], keys[index]))
    return [keys[index] for index in order]


def bh_reject(
    pvalues: Sequence[float],
    keys: Sequence[tuple[str, str]],
    alpha: float = ALPHA,
) -> dict[tuple[str, str], bool]:
    """Benjamini-Hochberg step-up, written in SPEC R4's LITERAL inclusive form.

    Rejects the LARGEST 1-based rank ``i`` whose p satisfies ``p_(i) <= i * alpha / m``, and every
    rank below it. The comparison is INCLUSIVE and is made on the RAW two-sided p from
    ``clv_significance``; rounding is for display only.

    WHY THIS IS VENDORED RATHER THAN DELEGATED TO ``scipy.stats.false_discovery_control``.
    scipy returns ADJUSTED p-values (q-values), computed as ``ps *= m / i`` followed by a REVERSE
    CUMULATIVE MINIMUM, and comparing those to alpha is mathematically -- but not
    float-equivalently -- the same rule. Verified live against scipy 1.17.1 in this venv:

      * The artifact is in the scaling. At m = 9, alpha = 0.05 the products ``p_3 * (9/3)`` and
        ``p_6 * (9/6)`` both evaluate to 0.05000000000000001, i.e. strictly greater than alpha,
        while ``p_i <= i * alpha / m`` is True at both ranks. Ranks 3 and 6 are the only two
        affected ranks of the nine.
      * The reverse cumulative minimum can MASK that artifact. On a vector where every rank sits
        exactly on its own boundary, rank 5's smaller adjusted value propagates back over ranks 3
        and 4, so scipy happens to agree there.
      * The divergence SURFACES whenever the affected boundary rank is the LARGEST rank
        satisfying the inclusive comparison, because then no later rank has a smaller adjusted
        value to rescue it. In that construction scipy rejects NOTHING while the pre-registered
        rule rejects ranks 1..3 (or 1..6). Both constructions are pinned by committed tests.

    scipy REMAINS the house tool for DISPLAY q-values, and is used for exactly that in
    ``backtest/group_gate.py``. ``backtest/ou_monetization.py`` must NOT be harmonized onto this
    implementation: its BH usage is itself pre-registered as-is under Phase 27.

    Args:
        pvalues: Raw two-sided p-values of the MEASURED cells only. A None or NaN entry is a
            programming error here, not a data condition -- such cells are excluded from the
            family BEFORE it is assembled (T-30-18) -- so one raises rather than being coerced.
        keys: The ``(group, target)`` identity of each family member.
        alpha: The pre-registered significance level. Defaults to the frozen ``ALPHA``.

    Returns:
        ``{key: rejected}`` for every key, with ``m`` implicitly the family size.

    Raises:
        ValueError: If the inputs differ in length, or if any p-value is None or NaN.
    """
    if len(pvalues) != len(keys):
        msg = (
            f"bh_reject requires pvalues and keys of the same length; got "
            f"{len(pvalues)} and {len(keys)}."
        )
        raise ValueError(msg)

    rejected: dict[tuple[str, str], bool] = dict.fromkeys(keys, False)
    m = len(pvalues)
    if m == 0:
        return rejected

    for key, pvalue in zip(keys, pvalues):
        if pvalue is None or math.isnan(float(pvalue)):
            msg = (
                f"bh_reject received a None/NaN p-value for cell {key}. A cell with no usable "
                "p-value must be routed to NOT MEASURED and EXCLUDED from the family before it "
                "is assembled -- never coerced to 1.0 and never passed in (T-30-18)."
            )
            raise ValueError(msg)

    order = sorted(range(m), key=lambda index: (pvalues[index], keys[index]))
    largest_passing_rank = 0
    for rank, index in enumerate(order, start=1):
        if pvalues[index] <= rank * alpha / m:
            largest_passing_rank = rank
    for rank, index in enumerate(order, start=1):
        if rank <= largest_passing_rank:
            rejected[keys[index]] = True
    return rejected


# ---------------------------------------------------------------------------
# The minimum detectable effect (D30-07)
# ---------------------------------------------------------------------------


def mde_for_cell(sig: Mapping[str, Any], power: float = MDE_POWER) -> float | None:
    """The pre-registered per-cell MDE: ``(t_{0.975, n-1} + t_{power, n-1}) * SE``.

    SE is RECOVERED from the canonical primitive's own ``ci95`` half-width rather than re-derived
    from the raw values: ``clv_significance`` computes ``se = std(ddof=1) / sqrt(n)`` but does not
    return it, and inverting ``half = t.ppf(0.975, n-1) * se`` recovers it exactly. The
    independent cross-check ``SE = mean / t`` (from ``ttest_1samp`` against zero) agrees to
    machine precision and is asserted in a committed test; the ``ci95`` route is nonetheless the
    implementation because it stays correct when the mean is exactly zero, where ``mean / t`` is
    0/NaN.

    The formula SELF-SCALES per cell, which is the load-bearing reason it is a formula and not a
    number. ``CLV_COLUMN_FOR`` is ``probability_clv`` for WP and ``line_clv`` for ATS/OU, and the
    two differ by roughly two orders of magnitude; a single scalar MDE would make WP
    unfalsifiable and ATS/OU untouchable. At the grid's real n the critical values are essentially
    normal, so the published MDE should read close to ``2.802 * SE`` and can be eyeballed against
    the published ``ci95``.

    Args:
        sig: A ``clv_significance`` result. Only ``n`` and ``ci95`` are read.
        power: The pre-registered power. Defaults to the frozen ``MDE_POWER``.

    Returns:
        The MDE in the cell's own CLV units, or None when ``ci95`` is None -- which is exactly the
        shape a below-floor cell arrives in, since ``clv_significance`` returns ``p``, ``t`` and
        ``ci95`` all None below ``MIN_CLV_SAMPLE``. Returning None rather than raising is what
        lets a NOT MEASURED cell pass through the verdict rule without an exception escaping.
    """
    ci95 = sig.get("ci95")
    if ci95 is None:
        return None
    degrees_of_freedom = int(sig["n"]) - 1
    half_width = (float(ci95[1]) - float(ci95[0])) / 2.0
    t_crit = float(stats.t.ppf(0.975, degrees_of_freedom))
    standard_error = half_width / t_crit
    return float(
        (t_crit + float(stats.t.ppf(power, degrees_of_freedom))) * standard_error
    )


# ---------------------------------------------------------------------------
# The measurement-exclusion predicate (it determines m, so it is pre-registered)
# ---------------------------------------------------------------------------


def bh_family_exclusion_reason(cell: Mapping[str, Any]) -> str | None:
    """Return why a screen cell is EXCLUDED from the BH family, or None if it is MEASURED.

    The check order is itself pre-registered. ``n_group_columns == 0`` is tested FIRST, before the
    p-value is inspected at all: SPEC R4 names "a group with zero columns present in gold reports
    NOT MEASURED, never DROP" as an acceptance criterion, and a group absent from gold produces
    two identical legs whose paired delta is exactly 0.0 with a NaN p -- so inspecting p first
    would reach the right answer for the wrong reason and would stop doing so the moment the
    numerics changed. This is the branch that keeps ``line_movement`` honest after the Plan 30-07
    drop.

    Args:
        cell: One ``backtest/signal_lift.py`` ``_screen_target`` result dict.

    Returns:
        One of ``EXCLUSION_REASONS``, or None when the cell is MEASURED and belongs in the family.
    """
    if int(cell.get("n_group_columns", 0)) == 0:
        return EXCLUSION_ZERO_COLUMNS_IN_GOLD
    if not cell.get("paired_sufficient", False):
        return EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE
    pvalue = cell.get("delta_p")
    if pvalue is None or math.isnan(float(pvalue)):
        return EXCLUSION_NO_PVALUE
    if int(cell.get("n_group_columns_selected", 0)) == 0:
        return EXCLUSION_NO_SELECTED_COLUMN
    return None
