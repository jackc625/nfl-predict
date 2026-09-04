"""Per-target selection strategies behind the ``BetSelector`` facade (Phase 31, plan 31-06; D31-01).

D31-01 splits the single-source selector into a TARGET-AGNOSTIC CORE plus one strategy per target,
both behind the unchanged ``backtest.bet_selector.BetSelector`` facade. This module holds the
strategies; the core stays in ``backtest/bet_selector.py`` because SPEC R4's source scan pins that
module as the ONE import target for a bet decision.

Why the seam exists, and why here:

  * The 10% weekly exposure cap is POOLED over the union of a week's bets across targets (D31-02),
    and the correlated de-weight groups by same-GAME across targets (D31-03). Neither has a natural
    home in three sibling selectors, so a shared core is not a stylistic preference -- it is the
    only place those two rules can be applied once.
  * The O/U numeric path moves here VERBATIM as ``OUStrategy``. Phase-27 reproduction is therefore
    STRUCTURAL -- a property of where the code sits -- rather than a test result that has to stay
    green through a rewrite. ``_bet_side``, ``_totals_regime``, ``_season_bias``,
    ``_calibrated_p_side`` and ``_subpop_label`` below are the Phase-27 bodies unchanged.

``TargetStrategy`` is a STRUCTURAL-SUBTYPING Protocol, following the ``FeatureBuilder`` precedent in
``features/protocol.py``: implementers declare conformance in their DOCSTRING and are checked
structurally (pyright statically, ``isinstance`` at test time), never by inheritance.

Only ``OUStrategy`` exists here. ``ATSStrategy`` and ``WPStrategy`` are deliberately ABSENT rather
than present-and-stubbed: a strategy whose methods raise ``NotImplementedError`` can be registered,
and a registered strategy that cannot decide is worse than a missing one -- it turns a loud
"unregistered target" error into a runtime failure mid-selection. Plan 31-10 adds them complete.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.ou_ev_chain import calibrated_p_over
from backtest.simulation import (
    SLIPPAGE_POINTS,
    BettingSimulator,
    SimulationConfig,
    apply_slippage_total,
)
from models.clv import compute_line_clv

__all__ = [
    "OUStrategy",
    "TargetStrategy",
    "UnregisteredTargetError",
    "require_finite_high_total_boundary",
]


class UnregisteredTargetError(LookupError):
    """Raised when a candidate names a target with no registered strategy (T-31-27).

    A missing strategy must be a LOUD FAILURE, not an empty result: silently producing no bets for
    a target looks identical to a target that genuinely had no +EV bets that week, and the
    profitability readout cannot tell those two apart after the fact.
    """


def require_finite_high_total_boundary(value: float) -> float:
    """Return ``value`` as a float, hard-failing on a non-finite high-total boundary (WR-03).

    Without this guard ``closing_total > NaN`` is always False, which silently collapses the O/U
    under-OR-high UNION to under-only. Kept as ONE implementation consumed by both ``OUStrategy``
    (which uses the boundary) and ``BetSelector`` (which publishes it as an attribute), so the
    check can never be stated two ways.

    Raises:
        ValueError: when ``value`` is not finite.
    """
    boundary = float(value)
    if not math.isfinite(boundary):
        msg = (
            "high_total_boundary must be finite; got a non-finite value (the pre-hold "
            "boundary did not resolve -- the silver odds lake is required, LOCKED-1). A NaN "
            "boundary would silently collapse the under-OR-high UNION to under-only (WR-03)."
        )
        raise ValueError(msg)
    return boundary


@runtime_checkable
class TargetStrategy(Protocol):
    """The per-target seam the target-agnostic selector core dispatches through (D31-01).

    Every member is something the CORE cannot know and EVERY target must answer. The core owns
    everything else: provenance, EV admission against the floor, Kelly sizing, the pooled weekly
    cap, the report-only CLV summary, and the shape of the decision record.

    Members:
        target: The target code (``"ou"``, and later ``"ats"`` / ``"wp"``). It is the registry key
            and it is stamped onto every decision record, so a pooled week can be split back out.
        required_market_fields: The market columns this target needs on a candidate row. The core
            copies them onto the decision record verbatim and raises a NAMED error when one is
            absent, so a malformed candidate frame fails by name rather than by ``KeyError``.
        resolve_bet_side: The side, delegating to the LOCKED
            ``BettingSimulator._determine_bet_side_*`` method for the target (D-18). A strategy
            NEVER re-implements the side convention.
        eligibility: ``None`` when the candidate is eligible, otherwise the rejection reason
            (a member of ``bet_selector.REJECTION_REASONS``). D31-05 gives WP and ATS no
            eligibility gate, so their implementations will return ``None`` unconditionally.
        eligibility_label: A human-readable label naming WHICH eligibility arm(s) the candidate
            satisfies. Only O/U currently has a sub-population concept (D27-04/05); targets
            without one report a constant label.
        side_probability: The calibrated ``P(side)`` and the slipped line, as a pair. This is the
            number Kelly consumes (BET-02) -- never a points distance.
        decision_extras: Target-specific REPORTING fields merged into the decision record (for O/U:
            the totals regime and the model-edge CLV). They keep target vocabulary out of the core.
        grade: The push-aware outcome, delegating to the LOCKED ``_resolve_*_outcome`` resolver.
    """

    target: str
    required_market_fields: tuple[str, ...]

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        """Return the bet side for ``row``, or None when there is no side."""
        ...

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        """Return None when eligible, else the rejection reason for ``row``."""
        ...

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        """Return the sub-population label for ``row``."""
        ...

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float]:
        """Return ``(calibrated_p_side, slipped_line)`` for an eligible ``row``."""
        ...

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """Return the target-specific reporting fields for ``row``."""
        ...

    def grade(self, record: dict[str, Any]) -> bool | None:
        """Return True (win), False (loss) or None (push / ungraded) for ``record``."""
        ...


class OUStrategy:
    """The Phase-27 O/U selection path, MOVED VERBATIM behind the D31-01 seam.

    Satisfies the ``TargetStrategy`` Protocol via structural subtyping (the ``FeatureBuilder``
    precedent in ``features/protocol.py``): conformance is declared here, in the docstring, and
    checked structurally -- there is no base class.

    The five private helpers below are the Phase-27 ``BetSelector`` methods with their bodies
    UNCHANGED: ``_bet_side`` (the LOCKED ``_determine_bet_side_ou`` delegation), ``_totals_regime``
    (the leakage-clean PRE-HOLD boundary, LOCKED-1), ``_season_bias`` (the no-silent-fallback
    prior-season walk-forward bias, D27-07), ``_calibrated_p_side`` (the half-point-slipped
    calibrated P(side), BET-02) and ``_subpop_label``. Moving rather than rewriting them is what
    makes the Phase-27 reproduction structural.

    Eligibility is the sub-pop UNION (D27-04/05): a candidate is eligible iff its side is ``under``
    OR its totals regime is ``high``. A None side (the model agrees with the market inside the side
    threshold) is not a bet of either side and is therefore not eligible.
    """

    target = "ou"
    # The two market columns an O/U decision reads. The realized ``actual`` total is NOT a market
    # field -- the core stashes it privately for grading.
    required_market_fields: tuple[str, ...] = ("model_total", "closing_total")

    def __init__(
        self,
        frozen_sd: float,
        season_bias_by_season: dict[int, float],
        high_total_boundary: float = HIGH_TOTAL_BOUNDARY_PREHOLD,
        slippage_points: float = SLIPPAGE_POINTS,
        simulator: BettingSimulator | None = None,
    ) -> None:
        self.frozen_sd = float(frozen_sd)
        self.season_bias_by_season = dict(season_bias_by_season)
        self.high_total_boundary = require_finite_high_total_boundary(
            high_total_boundary
        )
        self.slippage_points = float(slippage_points)
        # Wrap -- never re-implement -- the LOCKED side/slippage/outcome convention (D-18). The
        # facade injects its own simulator so exactly one exists per selector.
        self._sim = (
            simulator if simulator is not None else BettingSimulator(SimulationConfig())
        )

    # -- side / regime / EV helpers (Phase-27 bodies, moved verbatim) ----------

    def _bet_side(self, model_total: float, closing_total: float) -> str | None:
        """Determine the O/U bet side via the LOCKED ``_determine_bet_side_ou`` (D-18)."""
        return self._sim._determine_bet_side_ou(model_total, closing_total)

    def _totals_regime(self, closing_total: float) -> str:
        """High iff the closing total exceeds the leakage-clean PRE-HOLD boundary (LOCKED-1)."""
        return "high" if closing_total > self.high_total_boundary else "not_high"

    def _season_bias(self, season: int) -> float:
        """Prior-season walk-forward bias for ``season`` (NEGATIVE for an over-biased model)."""
        if season not in self.season_bias_by_season:
            msg = (
                f"no prior-season bias provided for season {season}; the caller must supply a "
                "walk-forward bias (ou_ev_chain.estimate_prior_season_bias) for every candidate "
                "season (no silent fallback to the raw biased total, D27-07)."
            )
            raise ValueError(msg)
        return float(self.season_bias_by_season[season])

    def _calibrated_p_side(
        self, bet_side: str, model_total: float, closing_total: float, season: int
    ) -> tuple[float, float]:
        """Calibrated P(side) and the slipped line for one candidate (BET-02 input).

        The P(side) is evaluated against the HALF-POINT-SLIPPED line (the line moves against the
        bettor, the LOCKED ``apply_slippage_total``), so a high-total OVER's over-bias is not
        rewarded: the slipped line + the bias correction pull the calibrated P(over) down. Returns
        ``(p_side, slipped_line)``.
        """
        slipped_line = apply_slippage_total(
            closing_total, bet_side, self.slippage_points
        )
        season_bias = self._season_bias(season)
        p_over = float(
            calibrated_p_over(model_total, slipped_line, self.frozen_sd, season_bias)
        )
        p_side = p_over if bet_side == "over" else (1.0 - p_over)
        return p_side, slipped_line

    @staticmethod
    def _subpop_label(is_under: bool, is_high: bool) -> str:
        """A human-readable sub-pop label for the UNION arms a candidate satisfies."""
        if is_under and is_high:
            return "under+high_total"
        if is_under:
            return "under"
        if is_high:
            return "high_total"
        return "none"

    # -- TargetStrategy Protocol surface --------------------------------------

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        """The O/U side via the LOCKED simulator convention."""
        return self._bet_side(float(row["model_total"]), float(row["closing_total"]))

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        """The sub-pop UNION gate (D27-04/05): eligible iff under-pick OR high-total.

        Returns None when eligible and ``"not_subpop"`` otherwise. A None side is not a bet of
        either side, so it never satisfies the union.
        """
        is_under, is_high = self._union_arms(row, bet_side)
        if bet_side is not None and (is_under or is_high):
            return None
        return "not_subpop"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        """The sub-pop label naming which UNION arm(s) the candidate satisfies."""
        is_under, is_high = self._union_arms(row, bet_side)
        return self._subpop_label(is_under, is_high)

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float]:
        """The calibrated P(side) and the half-point-slipped line (BET-02)."""
        return self._calibrated_p_side(
            bet_side,
            float(row["model_total"]),
            float(row["closing_total"]),
            int(row["season"]),
        )

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """The O/U reporting fields: the totals regime and the REPORT-ONLY model-edge CLV.

        The CLV here is ``compute_line_clv(model_total, closing_total, direction="total")`` -- the
        MODEL EDGE vs the line, DISTINCT from the freeze-vs-close forward metric (structurally ~0
        in backtest). It is reported, never a gate (D27-06/12).
        """
        model_total = float(row["model_total"])
        closing_total = float(row["closing_total"])
        return {
            "totals_regime": self._totals_regime(closing_total),
            "clv": compute_line_clv(model_total, closing_total, direction="total"),
        }

    def grade(self, record: dict[str, Any]) -> bool | None:
        """Grade a selected bet via the LOCKED ``_resolve_ou_outcome`` (push-aware, T-27-23).

        Returns True (win), False (loss), or None (push). The push (actual == slipped line) is
        carried as None, never coerced. When the candidate carries no ``actual`` (a forward,
        not-yet-played game), the outcome is None (ungraded) -- distinct from a push but both
        represented by None here; downstream stores both as SQL NULL (Plan 04).
        """
        actual = record.get("_actual_total")
        if actual is None:
            return None
        return self._sim._resolve_ou_outcome(
            record["bet_side"], float(actual), record["slipped_line"]
        )

    # -- internals ------------------------------------------------------------

    def _union_arms(
        self, row: dict[str, Any], bet_side: str | None
    ) -> tuple[bool, bool]:
        """The two UNION arms for ``row``: (is_under_pick, is_high_total)."""
        is_under = bet_side == "under"
        is_high = self._totals_regime(float(row["closing_total"])) == "high"
        return is_under, is_high
