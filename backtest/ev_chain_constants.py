"""The Phase-31 pre-registration for the three-target EV chain (SPEC R1/R3/R7, PROD-02).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/group_gate_constants.py`` (D30-05/D30-11).
This module IS the Phase-31 pre-registration, not a description of one. Its LAST-MODIFYING COMMIT
is the git-ancestry anchor: the readout guard requires that commit to be a strict git ANCESTOR of
the commit recording the 2025 measurement output. That ancestry assertion is the only mechanical
proof this phase has that the rule was fixed before the numbers existed.

Stated plainly because it is easy to forget fifteen plans later: EDITING THIS FILE AFTER THE
MEASUREMENT COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. There is no honest repair path.
A value that is wrong here is wrong for the remainder of the phase, and the only legitimate
response is to say so in the readout, not to amend the file.

Scope of THIS commit (Plan 31-01, the tracer). Only the two facts D31-24 and D31-36 already fix
are declared: the EV tier bands and the verdict-token vocabulary. Plan 31-05 lands the measurement
constants (the tune/hold season split, the BH family spec, the ATS residual contract, the
robustness cuts, the run-ledger states, the readout guards) and FREEZES this module at CHECKPOINT
1. Adding a constant before that freeze is expected; changing one after it is not.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math

__all__ = [
    "EV_TIER_BANDS",
    "EV_TIER_HIGH",
    "EV_TIER_LABELS",
    "EV_TIER_LOW",
    "EV_TIER_MEDIUM",
    "VERDICT_TOKENS",
    "assign_ev_tier",
]


# ---------------------------------------------------------------------------
# EV tier bands (D31-24, SPEC R7)
# ---------------------------------------------------------------------------

EV_TIER_LOW = "low"
EV_TIER_MEDIUM = "medium"
EV_TIER_HIGH = "high"

# The lower bound of the LOW band is the pre-registered EV floor ``t``, which is a per-target
# tuned scalar rather than a constant, so it is carried as ``None`` here and resolved from the
# ``ev_floor_t`` argument at call time. Recording it as a literal would freeze a number this
# module does not own.
_EV_FLOOR_SENTINEL: float | None = None

# Ordered half-open [lo, hi) bands. LOWER bound INCLUSIVE, UPPER bound EXCLUSIVE, so a value
# landing exactly on a boundary falls in the HIGHER band: 0.03 is medium (not low) and 0.05 is
# high (not medium). This is the SPEC R7 adjacency edge, fixed here rather than in the renderer.
EV_TIER_BANDS: tuple[tuple[str, float | None, float], ...] = (
    (EV_TIER_LOW, _EV_FLOOR_SENTINEL, 0.03),
    (EV_TIER_MEDIUM, 0.03, 0.05),
    (EV_TIER_HIGH, 0.05, math.inf),
)

EV_TIER_LABELS: tuple[str, ...] = tuple(label for label, _lo, _hi in EV_TIER_BANDS)


def assign_ev_tier(per_bet_ev: float, ev_floor_t: float) -> str:
    """Return the pre-registered EV band label for *per_bet_ev* under floor *ev_floor_t*.

    The bands are the half-open [lo, hi) triples in :data:`EV_TIER_BANDS`, with the LOW band's
    lower bound resolved to ``ev_floor_t``. A boundary value falls in the HIGHER band.

    A non-finite EV does NOT receive a label. Under the SPEC R7 suppression taxonomy such a row
    is rejected upstream with reason ``ev_not_finite`` and never reaches the live list, so
    returning a band here would manufacture a tier for a bet that is not a bet. The same applies
    to an EV below the floor: it was not admitted, so it has no band.

    Args:
        per_bet_ev: The calibrated per-bet expected value, in units of stake.
        ev_floor_t: The pre-registered per-target EV floor ``t`` (the LOW band's lower bound).

    Returns:
        One of ``"low"`` / ``"medium"`` / ``"high"``.

    Raises:
        ValueError: if *per_bet_ev* or *ev_floor_t* is non-finite, or if *per_bet_ev* is below
            *ev_floor_t* (an unadmitted row has no band).
    """
    if not math.isfinite(per_bet_ev):
        msg = (
            f"assign_ev_tier: per_bet_ev={per_bet_ev!r} is not finite; a non-finite EV is "
            "suppressed with reason 'ev_not_finite' (SPEC R7) and is never tiered."
        )
        raise ValueError(msg)
    if not math.isfinite(ev_floor_t):
        msg = (
            f"assign_ev_tier: ev_floor_t={ev_floor_t!r} is not finite; the EV floor is a "
            "pre-registered finite scalar and must resolve before any band is assigned."
        )
        raise ValueError(msg)
    if per_bet_ev < ev_floor_t:
        msg = (
            f"assign_ev_tier: per_bet_ev={per_bet_ev!r} is below the EV floor {ev_floor_t!r}; "
            "such a candidate is rejected with reason 'ev_below_floor' and has no band."
        )
        raise ValueError(msg)

    for label, lo, hi in EV_TIER_BANDS:
        lower = ev_floor_t if lo is _EV_FLOOR_SENTINEL else lo
        assert lower is not None  # narrowed by the sentinel resolution above
        if lower <= per_bet_ev < hi:
            return label

    # Unreachable by construction: the bands are contiguous from ev_floor_t to +inf and the
    # below-floor case is rejected above. Raise rather than return a label if it ever happens.
    msg = (
        f"assign_ev_tier: per_bet_ev={per_bet_ev!r} fell through every band under floor "
        f"{ev_floor_t!r}; EV_TIER_BANDS is no longer contiguous."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Verdict vocabulary (D31-36, SPEC R3/R11)
# ---------------------------------------------------------------------------

# The CLOSED set of verdict tokens the 2025 profitability run may emit. A target that selected
# zero bets reports UNDISCHARGEABLE_NO_BETS -- never "ROI 0" -- and a target whose EV chain did
# not resolve reports UNDISCHARGEABLE_NO_CHAIN. Both are first-class outcomes, not failures.
VERDICT_TOKENS: tuple[str, ...] = (
    "PROFITABLE_CLEAN",
    "UNPROFITABLE_CLEAN",
    "INCONCLUSIVE_CLEAN",
    "UNDISCHARGEABLE_NO_BETS",
    "UNDISCHARGEABLE_NO_CHAIN",
)
