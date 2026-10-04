"""The ONE edge-band helper -- now ONE RULER PER UNIT (Phase 33, plan 33-17; CLEAN-01, D33-22).

WHAT WAS COLLAPSED (Phase 31, 31-17; D31-23)
---------------------------------------------
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
The band is an EDGE BAND, ``edge_tier``. ``/bets`` owns a different and incompatible concept,
``ev_tier``, whose bands are absolute per-bet expected value (D31-24). Before the rename both were
called "confidence" and both rendered the word "high", so "high" meant two incompatible things
across two surfaces.

``backtest/simulation.py`` warns in its own comments that on the SELECTOR path the edge field
carries per-bet EXPECTED VALUE rather than a points or probability edge, and that such rows must
not reach this helper. That separation is IN PLACE: selector-produced rows live in the
``bet_list`` table and are banded by ``backtest.ev_chain_constants.assign_ev_tier``; nothing routes
a per-bet EV into ``edge_tier``. The risk that comment raises is therefore recorded as CLOSED, and
``tests/api/test_cache_betting.py`` asserts the separation structurally.

THE SURVIVING DEFECT IS NOW REPAIRED (DEF-31-17 -> CLEAN-01, D33-22)
----------------------------------------------------------------------
This helper used to apply ONE threshold pair to THREE INCOMPATIBLE UNITS:

    wp_edge   a probability delta          model minus the market WP (spread through the
                                           blend's converter, since 33.2 review C2 WR-05)
    ats_edge  POINTS                       signed model-minus-market home margin (R13, D33-05)
    ou_edge   a fraction of the total      (model total - market total) / max(market total, 30)

So a "high" WP edge and a "high" ATS edge were not comparable quantities, and after Plan 33-10
put ``ats_edge`` on a POINTS scale the 0.05 pair put 98.80% of ATS games -- 1,074 of 1,087 --
into "high". Plan 31-17 DE-DUPLICATED and RENAMED the rule and deliberately did NOT repair it,
because repairing it moves a published label on ``/`` and ``/betting`` (D31-04) and no measurement
then existed showing new bands would be better.

That measurement now exists and is FROZEN, and since quick task 261003-vke (WINDOWS row 19) the
pair this helper reads is the ROW-19 RE-MEASURE:
``backtest.neutral_hfa_cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET`` carries one
``(high, medium)`` pair per target, each on its target's OWN unit, re-derived by the Plan 33.2-26
recipe on the models re-fitted after the neutral-site Elo fix and the owned pre-lock lines. It
supersedes 9bb7568's ``backtest.corrected_cold_start_constants``, which superseded the 11761c7
original ``backtest.cold_start_constants``; both stay byte-unchanged and importable as the record
of what was measured and when, and this helper reads neither.

A TARGET WITH NO HONEST THRESHOLD IS UNBANDED. The correction records ``None`` for a target whose
honest pool was too small to derive a pair (SPEC R14). :func:`edge_tier` then returns ``None`` --
UNBANDED -- which is a different claim from ``"low"`` (banded, below medium): it is never unpacked
into a ``TypeError`` and never lent another target's pair. ``target`` is a REQUIRED argument here with NO DEFAULT: a call
site that forgot would silently band an ATS point edge against WP's probability pair, which is the
exact defect being removed, reintroduced as a default. Omitting it is a ``TypeError`` from the
interpreter -- the one refusal nobody can forget to write.

WP's pair is UNCHANGED at 0.05 / 0.02 by design (D33-20), in the original and in the correction
alike, so zero WP games change band and the 23-point pre-collapse snapshot keeps its evidentiary
value instead of being rewritten.

WHY THE FROZEN MAPPING IS IMPORTED LAZILY
------------------------------------------
``backtest.neutral_hfa_cold_start_constants`` is a pure-constants module with no imports of its
own, but
reaching it executes ``backtest/__init__.py``, which pulls ``backtest.engine`` and
``backtest.simulation`` and through them ``models.*``, scikit-learn, XGBoost and Plotly. Two
MEASURED reasons, in that order of weight:

  1. LAYERING. ``utils/`` sits BELOW ``backtest/``: ``backtest.engine``, ``.simulation``,
     ``.report``, ``.run`` and ``.metrics`` each do ``from utils import ...``. A module-level
     ``utils.edge_tier -> backtest`` edge inverts that, and the first ``backtest`` module to
     import this band would then meet a half-initialized ``utils``. No such importer exists today
     -- ``tests/unit/test_phase33_preregistration.py`` asserts three selection modules import no
     band at all -- and this keeps it a design property rather than a coincidence.

  2. COST. Importing this leaf helper is 0.63 s and loads NO scikit-learn, XGBoost, Plotly or
     ``models.*``; importing the frozen constants module standalone is 1.94 s and loads all of
     them (measured 2026-09-15 on the 11761c7 original; the correction has the same shape). Deferred, that cost lands once on the first band actually
     taken (1.37 s) rather than on every importer of this module.

The import is therefore deferred into :func:`_thresholds_by_target` and cached after the first
resolution. There is still exactly ONE source for the numbers; only the moment of reading moved.

WHICH WRITE IS AUTHORITATIVE
-----------------------------
Both, for DIFFERENT artifacts -- measured, not assumed. ``api/cache.py::_load_predictions`` derives
the band for the ``predictions`` TABLE from ``outputs/backtest/predictions_all.csv``, which carries
no band column of its own (verified: its 21 columns include no ``*_confidence``).
``scripts/generate_current_week_predictions.py`` derives it for the CURRENT-WEEK CSV under
``outputs/predictions/``, which the cache never reads (``populate_cache`` is given
``outputs/backtest``). So neither write overwrites the other. Both are DISPLAY-AND-REPORT writers
and neither feeds bet selection, which is the finding
``tests.phase33_state.EDGE_TIER_DISPLAY_ONLY_CORRECTED`` records and
``tests/unit/test_phase33_preregistration.py`` guards.

The stored/rendered COLUMN names stay ``wp_confidence`` / ``ats_confidence`` / ``ou_confidence``.
Renaming them is a schema change that would move every export header -- a published figure -- and
is out of scope here; the concept rename is in the helper, which is where the ambiguity lived.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ``EDGE_TIER_HIGH_THRESHOLD`` and ``EDGE_TIER_MEDIUM_THRESHOLD`` are DELIBERATELY ABSENT from
# this list while remaining importable. They are served by the module ``__getattr__`` at the foot
# of the file (PEP 562) and are LEGACY: a bare "high threshold" with no unit attached is exactly
# the quantity this plan retired, and new code should read ``EDGE_TIER_TARGETS`` and call
# ``edge_tier(edge, target)``. Keeping them out of ``__all__`` is the accurate signal -- they are
# still resolvable for the one drift test that binds to them, but they are not promoted as part
# of the star-export surface. (It also keeps both ruff and pyright honest: neither models PEP 562
# here, and suppressing two tools to advertise two deprecated names would be the wrong trade.)
__all__ = [
    "EDGE_TIER_LABELS",
    "EDGE_TIER_TARGETS",
    "UnknownEdgeTargetError",
    "edge_tier",
    "edge_tier_series",
]

# The closed three-label vocabulary, low to high.
EDGE_TIER_LABELS: tuple[str, str, str] = ("low", "medium", "high")

# The closed TARGET vocabulary. Stated as a tuple here and asserted equal to the frozen
# mapping's keys by ``tests/unit/test_edge_tier_per_target.py``, so a fourth target cannot be
# invented on one side without the other noticing -- and so the refusal below can name the valid
# set without importing the frozen module just to build an error message.
EDGE_TIER_TARGETS: tuple[str, str, str] = ("ats", "ou", "wp")

# Resolved once, on the first band taken. See the module docstring for why this is not a
# module-level import.
_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float] | None] | None = None


class UnknownEdgeTargetError(ValueError):
    """A band was asked for on a target the frozen pre-registration does not price.

    Raised by name, carrying BOTH the offending value and the valid vocabulary. A refusal that
    says only "unknown target" sends the reader into the source to discover what the
    alternatives are, which this repository treats as worse than no message at all.
    """


def _thresholds_by_target() -> dict[str, tuple[float, float] | None]:
    """The CORRECTED ``{target -> (high, medium) | None}`` mapping, imported on first use and cached.

    The deferred import is deliberate and the module docstring gives both measured reasons:
    reaching the constants module executes ``backtest/__init__.py``, which inverts the ``utils``
    -> ``backtest`` layering and loads the whole modelling stack. It reads the row-19 re-measure
    (quick task 261003-vke), never the superseded 9bb7568 correction or the 11761c7 original.
    """
    global _THRESHOLDS_BY_TARGET
    if _THRESHOLDS_BY_TARGET is None:
        from backtest.neutral_hfa_cold_start_constants import (
            EDGE_TIER_THRESHOLDS_BY_TARGET,
        )

        _THRESHOLDS_BY_TARGET = EDGE_TIER_THRESHOLDS_BY_TARGET
    return _THRESHOLDS_BY_TARGET


def _require_thresholds(target: str) -> tuple[float, float] | None:
    """This target's ``(high, medium)`` pair, or None when it has NO honest threshold.

    Refuses an UNKNOWN target by name. A KNOWN target whose pair is ``None`` is not an error: it
    is the correction's recorded verdict that no honest threshold exists, and it is returned as
    ``None`` for the caller to render as unbanded.
    """
    try:
        return _thresholds_by_target()[target]
    except KeyError:
        msg = (
            f"no frozen edge-band thresholds for target {target!r}; the pre-registered "
            f"vocabulary is {sorted(EDGE_TIER_TARGETS)}. Each target's pair is on its OWN unit "
            "-- ats in POINTS, ou as a ratio of the market total, wp in probability -- so there "
            "is no neutral pair to fall back to and none is invented here. The pairs live in "
            "backtest/neutral_hfa_cold_start_constants.EDGE_TIER_THRESHOLDS_BY_TARGET; inspect "
            'them with `uv run python -c "import backtest.neutral_hfa_cold_start_constants as c; '
            'print(c.EDGE_TIER_THRESHOLDS_BY_TARGET, c.EDGE_TIER_THRESHOLD_UNITS)"`.'
        )
        raise UnknownEdgeTargetError(msg) from None


def edge_tier(edge: float | None, target: str) -> str | None:
    """The edge band for ONE edge value, on *target*'s own unit.

    THE single implementation. :func:`edge_tier_series` dispatches to it and computes nothing of
    its own, so there is one definition of the rule in the repository rather than a scalar one and
    a vectorized one that can drift apart.

    Both comparisons are STRICT: a value exactly at a threshold falls in the LOWER band. That is
    the behaviour both retired helpers had (``>``, not ``>=``) and the boundary rows of all three
    recorded grids pin it, for all three targets.

    Args:
        edge: A signed edge on *target*'s own scale, or None/NaN when no edge was computable.
            The band is taken on the MAGNITUDE: a large edge against the home side is as strong a
            signal as a large edge for it.
        target: One of :data:`EDGE_TIER_TARGETS`. REQUIRED, with no default -- see the module
            docstring. A defaulted target would let a forgetful call site band an ATS point edge
            against WP's probability pair, which is the defect this signature exists to remove.

    Returns:
        ``"high"``, ``"medium"`` or ``"low"``. ``None`` -- UNBANDED -- in two cases, both
        distinct from ``"low"``, which is a band: the target has NO honest threshold (SPEC R14),
        or the edge is ABSENT (a game with no market line has no edge to band).

        AN ABSENT EDGE USED TO BE ``"low"`` (33.2 review C2 CR-04 = B WR-11). Both retired helpers
        answered it that way, and every game with no line was then stored, rendered and exported
        with the band "low" -- a claim about an edge that does not exist. It is answered HERE, so
        the call sites cannot answer the absent case differently.

    Raises:
        UnknownEdgeTargetError: when *target* is outside the frozen vocabulary. Raised BEFORE the
            absent-edge shortcut, so a bad target is reported even on a null edge.
    """
    pair = _require_thresholds(target)
    if pair is None:
        return None
    high, medium = pair
    if edge is None or pd.isna(edge):
        return None
    magnitude = abs(float(edge))
    if magnitude > high:
        return EDGE_TIER_LABELS[2]
    if magnitude > medium:
        return EDGE_TIER_LABELS[1]
    return EDGE_TIER_LABELS[0]


def edge_tier_series(edge: pd.Series, target: str) -> pd.Series:
    """The edge band for a whole column -- a DISPATCH to :func:`edge_tier`, not a second rule.

    Returns a Series of labels aligned to *edge*'s index. It performs no comparison of its own,
    which is what makes "exactly one function computes the edge band" checkable -- by an AST walk
    for ``ast.Compare`` nodes rather than by a substring search, since a substring search would
    false-positive on an annotation, a docstring or a comment.

    The target is validated UP FRONT rather than per element, so an unknown target is refused even
    when the column is EMPTY -- which is exactly what a weekly slate with no priced game produces,
    and therefore exactly the case a per-element check would pass silently.

    A PER-ELEMENT PYTHON DISPATCH IS THE POINT, not an oversight. Vectorizing the comparisons here
    would put threshold logic in a second place, which is the defect CLEAN-01 exists to remove,
    reintroduced as an optimisation. The weekly slate is a few dozen rows.
    """
    _require_thresholds(target)
    return edge.map(lambda value: edge_tier(value, target)).astype(object)


def __getattr__(name: str) -> Any:
    """Resolve the two LEGACY threshold constants from the frozen WP entry (PEP 562).

    They survive because ``tests/unit/test_phase33_preregistration.py`` binds the frozen WP pair
    to them -- that is the drift test proving WP's labels do not move -- and they are RE-POINTED
    rather than re-typed: a second literal ``0.05`` in this module would be a second source for a
    number the pre-registration froze, and two sources for one number is the drift this repository
    has been bitten by three times.

    They are served through the module ``__getattr__`` rather than assigned at import time for the
    same reason the mapping is imported lazily: assigning them would force
    ``backtest.neutral_hfa_cold_start_constants`` -- and with it the whole modelling stack -- into every
    importer of this module, including the cache path that is built to avoid it.

    New code should read :data:`EDGE_TIER_TARGETS` and call :func:`edge_tier` with an explicit
    target instead. A bare "high threshold" is exactly the unit-free quantity this plan retired.
    """
    if name in ("EDGE_TIER_HIGH_THRESHOLD", "EDGE_TIER_MEDIUM_THRESHOLD"):
        pair = _require_thresholds("wp")
        if pair is None:
            msg = f"{name} is unavailable: WP has NO honest threshold in the correction"
            raise AttributeError(msg)
        return pair[0] if name == "EDGE_TIER_HIGH_THRESHOLD" else pair[1]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
