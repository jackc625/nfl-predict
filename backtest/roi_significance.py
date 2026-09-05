"""The PRE-REGISTERED ROI hypothesis test (Phase 31, plan 31-12; REVIEW-ROI, SPEC R3).

WHY THIS MODULE EXISTS AT ALL
-----------------------------
No ROI hypothesis test existed in this repository. ``backtest/diagnose.py``'s
``clv_significance`` tests CLV -- its own docstring says it mirrors the ``ttest_1samp(clv, 0)``
idiom -- and ``backtest/ou_monetization.py``'s ``_block_by_week_bootstrap_ci`` returns a
PERCENTILE INTERVAL with no p-value at all. So the only p-value within reach of a
profitability verdict was the CLV one, and substituting it would silently convert
"significant CLV" into "profitable": precisely the D25-14 and D26-09 trap this milestone
exists to avoid. A superficially valid but dishonest verdict is worse than a failed run.

The test implemented here was frozen in ``backtest.ev_chain_constants.ROI_SIGNIFICANCE_SPEC``
and in ``PROFITABILITY-PREREGISTRATION.md`` section 7, BEFORE any 2025 number existed. This
module implements exactly that and nothing else.

WHY RE-EXPRESSING A FROZEN RESAMPLER IS ADMISSIBLE HERE
------------------------------------------------------
``backtest/ou_monetization.py`` is BYTE-FROZEN -- the Phase-27 and Phase-30 published record
depends on it, and plan 31-13 asserts it is unchanged -- so the frozen
``_block_by_week_bootstrap_ci`` cannot be extended in place to also return what it discards.
Re-expressing a frozen statistic is a real risk of minting a SECOND ANSWER, so the risk is
closed MECHANICALLY rather than argued: ``tests/unit/test_roi_significance.py`` asserts that
:func:`roi_ci_and_p`'s confidence interval is BYTE-IDENTICAL to the frozen helper's on the
same frame, rendered through the shared 17-significant-digit specifier and compared as
STRINGS with no numeric tolerance, across hand-built frames including both degenerate cases.

**This module is admissible ONLY because it reproduces the frozen interval exactly while
additionally returning the replicate array the frozen one throws away.** If the two ever
disagree, this module is wrong and the test says so.

``BOOTSTRAP_B``, ``BOOTSTRAP_SEED`` and ``BOOTSTRAP_CI_TYPE`` are IMPORTED, never re-declared:
a copied literal that happened to read 2000 today would be a second declaration that can
drift. ``_flat_roi_from_records`` is likewise IMPORTED rather than re-written, so the
payout-over-stake ratio in a replicate is literally the same function the frozen helper and
the Phase-27 published ROI both used. Importing a private name is deliberate: the alternative
is a second ratio implementation, which is the exact failure this whole module is designed
around.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.ev_chain_constants import (
    ROI_MIN_ATTAINABLE_P,
    ROI_P_VALUE_METHOD,
)

# The frozen Phase-27 bootstrap configuration and the frozen flat-stake ratio, CONSUMED. See
# the module docstring on why the private ratio helper is imported rather than restated.
from backtest.ou_monetization import (
    BOOTSTRAP_B,
    BOOTSTRAP_CI_TYPE,
    BOOTSTRAP_SEED,
    _flat_roi_from_records,
)

__all__ = [
    "BLOCK_KEYS",
    "BOOTSTRAP_B",
    "BOOTSTRAP_CI_TYPE",
    "BOOTSTRAP_SEED",
    "MIN_BLOCKS_FOR_P",
    "P_ABSENT_EMPTY_FRAME",
    "P_ABSENT_NO_REPLICATES",
    "P_ABSENT_SINGLE_BLOCK",
    "block_by_week_roi_replicates",
    "flat_roi",
    "roi_bootstrap_p_value",
    "roi_ci_and_p",
]


# The BLOCK. The (season, week) pair, exactly as the frozen helper groups. Resampling whole
# weeks preserves within-week correlation -- a half-point line move correlates the games on a
# slate -- and resampling individual bets would understate the spread.
BLOCK_KEYS: tuple[str, str] = ("season", "week")

# A one-block frame gets NO p-value, and that refusal is load-bearing rather than defensive.
# With a single block every replicate resamples the SAME block, so every replicate ROI equals
# the point estimate exactly; the recentred replicates are then all 0.0 and a positive point
# estimate would score the SMALLEST ATTAINABLE p on a resampling space of one. That is a
# manufactured significance, not a measured one. The frozen CI helper reports such a run with
# ``n_blocks == 1`` and a collapsed interval; this module reports it with a NULL p and a
# stated reason, which is the same honesty applied to the quantity that drives a verdict.
MIN_BLOCKS_FOR_P: int = 2

P_ABSENT_EMPTY_FRAME: str = (
    "no bets were selected on the hold, so there is no ROI to test; a zero-bet target is a "
    "first-class RESULT (UNDISCHARGEABLE_NO_BETS) and never a return of 0."
)
P_ABSENT_NO_REPLICATES: str = (
    "every bootstrap replicate was degenerate (no replicate carried a positive total stake), "
    "so no reference distribution exists to compute an achieved significance level against."
)
P_ABSENT_SINGLE_BLOCK: str = (
    f"the hold bets span fewer than {MIN_BLOCKS_FOR_P} (season, week) blocks, so every "
    "replicate reproduces the point estimate exactly and the achieved significance level "
    "would be manufactured by the resampling space rather than measured. Reported as a NULL "
    "p-value with this reason."
)


def flat_roi(per_bet: pd.DataFrame) -> float | None:
    """The flat-stake ROI ``sum(payout_flat) / sum(flat_stake)``, or None with no stake.

    A thin re-export of the FROZEN ``backtest.ou_monetization._flat_roi_from_records`` so
    callers reach exactly one implementation of the statistic the null is stated about. No
    new estimator is introduced (``ROI_SIGNIFICANCE_SPEC["statistic"]``).
    """
    return _flat_roi_from_records(per_bet)


def block_by_week_roi_replicates(per_bet: pd.DataFrame) -> list[float]:
    """The block-by-week bootstrap ROI replicates the frozen CI helper computes and DISCARDS.

    Reproduces ``backtest.ou_monetization._block_by_week_bootstrap_ci`` step for step: the
    ``(season, week)`` pair as the block, ``np.random.default_rng(BOOTSTRAP_SEED)``,
    ``rng.integers(0, n_blocks, size=n_blocks)``, ``BOOTSTRAP_B`` replicates, the same
    payout-over-stake ratio, and the same skip of a degenerate replicate. The ONLY difference
    is that the replicate array is RETURNED instead of being reduced to two percentiles and
    dropped.

    Args:
        per_bet: The per-bet frame, carrying ``season``, ``week``, ``payout_flat`` and
            ``flat_stake``. HOLD bets only -- the reference distribution never reaches
            outside the hold.

    Returns:
        The replicate ROIs, in draw order. EMPTY for an empty frame (the frozen helper's
        first degenerate early return) and empty when no replicate carried a positive total
        stake (its second).
    """
    if per_bet.empty:
        return []

    blocks = [grp for _, grp in per_bet.groupby(list(BLOCK_KEYS), sort=True)]
    n_blocks = len(blocks)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    roi_reps: list[float] = []
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        resampled = pd.concat([blocks[i] for i in idx], ignore_index=True)
        roi = _flat_roi_from_records(resampled)
        if roi is not None:
            roi_reps.append(roi)
    return roi_reps


def roi_bootstrap_p_value(
    replicates: Any, point_estimate: float | None
) -> float | None:
    """The pre-registered ONE-SIDED achieved significance level, RECENTRED AT THE NULL.

    ``H0``: the POPULATION flat-stake ROI over the hold is LESS THAN OR EQUAL TO ZERO. Each
    replicate is recentred by SUBTRACTING the observed ROI, which is what turns a percentile
    interval's resampling distribution into a null distribution::

        p = (1 + #{recentred replicates >= observed ROI}) / (BOOTSTRAP_B + 1)

    The plus-one in BOTH numerator and denominator makes ``p`` STRICTLY POSITIVE and never
    zero, so a run in which no recentred replicate reaches the observed ROI reports the
    smallest attainable p rather than a ``0.0`` that would overstate the evidence.

    The denominator is the FROZEN ``BOOTSTRAP_B + 1``, not the number of replicates that
    survived. Dividing by a survivor count would make the p-value depend on how many
    replicates happened to be degenerate, which is a property of the draw rather than of the
    hypothesis.

    Args:
        replicates: The replicate ROIs from :func:`block_by_week_roi_replicates`.
        point_estimate: The OBSERVED flat-stake ROI over the hold bets.

    Returns:
        The one-sided p in ``(0, 1]``, or None when there is no point estimate or no
        replicate to compare against.
    """
    if point_estimate is None:
        return None
    reps = np.asarray(replicates, dtype=float)
    if reps.size == 0:
        return None
    observed = float(point_estimate)
    recentred = reps - observed
    at_or_beyond = int(np.count_nonzero(recentred >= observed))
    return float((1 + at_or_beyond) / (BOOTSTRAP_B + 1))


def roi_ci_and_p(per_bet: pd.DataFrame) -> dict[str, Any]:
    """The percentile CI AND the pre-registered p-value, from ONE resampling pass.

    The CI fields are byte-identical to what the frozen
    ``backtest.ou_monetization._block_by_week_bootstrap_ci`` produces on the same frame; that
    identity is asserted on strings under the shared 17-significant-digit specifier in
    ``tests/unit/test_roi_significance.py`` and is the whole reason this module is allowed to
    exist. The p-value is the addition.

    Args:
        per_bet: The HOLD per-bet frame (``season``, ``week``, ``payout_flat``,
            ``flat_stake``).

    Returns:
        The frozen helper's field set -- ``point_estimate``, ``ci_lo``, ``ci_hi``,
        ``n_blocks``, ``b``, ``seed``, ``ci_type``, ``scope`` -- plus ``p_value``,
        ``p_value_absent_reason``, ``n_replicates``, ``p_value_method`` and
        ``min_attainable_p``. A degenerate frame returns a NULL p with a stated reason
        rather than raising.
    """
    base: dict[str, Any] = {
        "b": BOOTSTRAP_B,
        "seed": BOOTSTRAP_SEED,
        "ci_type": BOOTSTRAP_CI_TYPE,
        "scope": "within_holdout_only",
        "p_value_method": ROI_P_VALUE_METHOD,
        "min_attainable_p": ROI_MIN_ATTAINABLE_P,
    }

    # Degenerate case 1, mirroring the frozen helper's first early return.
    if per_bet.empty:
        return {
            **base,
            "point_estimate": None,
            "ci_lo": None,
            "ci_hi": None,
            "n_blocks": 0,
            "n_replicates": 0,
            "p_value": None,
            "p_value_absent_reason": P_ABSENT_EMPTY_FRAME,
        }

    point = _flat_roi_from_records(per_bet)
    n_blocks = int(per_bet.groupby(list(BLOCK_KEYS), sort=True).ngroups)
    replicates = block_by_week_roi_replicates(per_bet)

    # Degenerate case 2, mirroring the frozen helper's second early return.
    if not replicates:
        return {
            **base,
            "point_estimate": point,
            "ci_lo": None,
            "ci_hi": None,
            "n_blocks": n_blocks,
            "n_replicates": 0,
            "p_value": None,
            "p_value_absent_reason": P_ABSENT_NO_REPLICATES,
        }

    ci_lo = float(np.percentile(replicates, 2.5))
    ci_hi = float(np.percentile(replicates, 97.5))

    if n_blocks < MIN_BLOCKS_FOR_P:
        p_value: float | None = None
        absent_reason: str | None = P_ABSENT_SINGLE_BLOCK
    else:
        p_value = roi_bootstrap_p_value(replicates, point)
        absent_reason = None

    return {
        **base,
        "point_estimate": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_blocks": n_blocks,
        "n_replicates": len(replicates),
        "p_value": p_value,
        "p_value_absent_reason": absent_reason,
    }
