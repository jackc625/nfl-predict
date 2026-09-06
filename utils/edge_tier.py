"""The ONE edge-band helper, collapsed from two duplicates and renamed (Phase 31, 31-17; D31-23).

WHAT WAS COLLAPSED
------------------
Two byte-equivalent implementations of the same three-band rule lived in the tree:

    api/cache.py                                 ``_compute_confidence(edge: Series) -> Series``
    scripts/generate_current_week_predictions.py ``compute_confidence(edge: float) -> str``

Both mapped ``|edge| > 0.05 -> high``, ``|edge| > 0.02 -> medium``, else ``low``. They agreed
value-for-value across a 23-point grid spanning every boundary, which is what made collapsing them
safe -- and it is asserted in ``tests/api/test_cache_betting.py`` against a snapshot recorded
BEFORE the collapse, so the regression test is a genuine regression test rather than a rewrite with
a new expectation.

WHY THE CONCEPT IS RENAMED
---------------------------
The band is now an EDGE BAND, ``edge_tier``. ``/bets`` owns a different and incompatible concept,
``ev_tier``, whose bands are absolute per-bet expected value (D31-24). Before the rename both were
called "confidence" and both rendered the word "high", so "high" meant two incompatible things
across two surfaces.

``backtest/simulation.py`` warns in its own comments that on the SELECTOR path the edge field
carries per-bet EXPECTED VALUE rather than a points or probability edge, and that such rows must
not reach this helper. That separation is NOW IN PLACE: selector-produced rows live in the
``bet_list`` table and are banded by ``backtest.ev_chain_constants.assign_ev_tier``; nothing routes
a per-bet EV into ``edge_tier``. The risk that comment raises is therefore recorded as CLOSED, and
``tests/api/test_cache_betting.py`` asserts the separation structurally.

THE SURVIVING DEFECT, NAMED RATHER THAN HIDDEN (DEF-31-17)
-----------------------------------------------------------
This helper applies ONE threshold pair to THREE INCOMPATIBLE UNITS:

    wp_edge   a probability delta          (dimensionless, typically |.| < 0.15)
    ats_edge  a fraction of |spread|       (a ratio whose denominator is a point count)
    ou_edge   a fraction of the total      (a ratio whose denominator is ~45)

So a "high" WP edge and a "high" ATS edge are not comparable quantities. Plan 31-17
DE-DUPLICATES and RENAMES this rule; it does NOT repair it. Repairing it would move a published
label on ``/`` and ``/betting``, which D31-04 pins, and there is no measurement showing new bands
would be better. It is recorded as a named deferred item and carried into the project's open-items
document by the closing plan.

WHICH WRITE IS AUTHORITATIVE
-----------------------------
Both, for DIFFERENT artifacts -- measured, not assumed. ``api/cache.py::_load_predictions`` derives
the band for the ``predictions`` TABLE from ``outputs/backtest/predictions_all.csv``, which carries
no band column of its own (verified: its 21 columns include no ``*_confidence``).
``scripts/generate_current_week_predictions.py`` derives it for the CURRENT-WEEK CSV under
``outputs/predictions/``, which the cache never reads (``populate_cache`` is given
``outputs/backtest``). So neither write overwrites the other; the defect was two implementations of
one rule, and after this collapse there is one implementation behind two call sites.

The stored/rendered COLUMN names stay ``wp_confidence`` / ``ats_confidence`` / ``ou_confidence``.
Renaming them is a schema change that would move every export header -- a published figure -- and
is out of scope here; the concept rename is in the helper, which is where the ambiguity lived.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "EDGE_TIER_HIGH_THRESHOLD",
    "EDGE_TIER_LABELS",
    "EDGE_TIER_MEDIUM_THRESHOLD",
    "edge_tier",
    "edge_tier_series",
]

# The two thresholds, in ONE place. Both are STRICT: a value exactly at a threshold falls in the
# LOWER band, which is the behaviour both retired helpers had (``>`` not ``>=``) and which the
# boundary rows of the pre-collapse snapshot pin.
EDGE_TIER_HIGH_THRESHOLD: float = 0.05
EDGE_TIER_MEDIUM_THRESHOLD: float = 0.02

# The closed three-label vocabulary, low to high.
EDGE_TIER_LABELS: tuple[str, str, str] = ("low", "medium", "high")


def edge_tier(edge: float | None) -> str:
    """The edge band for ONE edge value.

    THE single implementation. :func:`edge_tier_series` dispatches to it and computes nothing of
    its own, so there is one definition of the rule in the repository rather than a scalar one and
    a vectorized one that can drift apart.

    Args:
        edge: A signed edge on its target's own scale, or None/NaN when no edge was computable.
            The band is taken on the MAGNITUDE: a large edge against the home side is as strong a
            signal as a large edge for it.

    Returns:
        ``"high"``, ``"medium"`` or ``"low"``. An absent edge is ``"low"`` -- the behaviour both
        retired helpers had, in one case through ``np.where``'s NaN-comparison result and in the
        other through a ``pd.notna`` guard at the call site. It is stated HERE now, so the two
        call sites cannot answer the absent case differently.
    """
    if edge is None or pd.isna(edge):
        return EDGE_TIER_LABELS[0]
    magnitude = abs(float(edge))
    if magnitude > EDGE_TIER_HIGH_THRESHOLD:
        return EDGE_TIER_LABELS[2]
    if magnitude > EDGE_TIER_MEDIUM_THRESHOLD:
        return EDGE_TIER_LABELS[1]
    return EDGE_TIER_LABELS[0]


def edge_tier_series(edge: pd.Series) -> pd.Series:
    """The edge band for a whole column -- a DISPATCH to :func:`edge_tier`, not a second rule.

    Returns a Series of labels aligned to *edge*'s index. It performs no comparison of its own,
    which is what makes "exactly one function computes the edge band" checkable by a source scan.
    """
    return edge.map(edge_tier).astype(object)
