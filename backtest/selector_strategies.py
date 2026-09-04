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

Plan 31-10 completes the set: ``ATSStrategy`` and ``WPStrategy`` land here fully implemented, and
``default_strategies`` builds the three-target registry a mixed week is selected through. They were
deliberately ABSENT rather than present-and-stubbed until then: a strategy whose methods raise
``NotImplementedError`` can be registered, and a registered strategy that cannot decide is worse
than a missing one -- it turns a loud "unregistered target" error into a runtime failure
mid-selection.

TWO SCALES AND TWO PRICES, STATED ONCE HERE BECAUSE BOTH ARE EASY TO GET SILENTLY WRONG
----------------------------------------------------------------------------------------

  * The ATS target carries TWO SIGN CONVENTIONS. ``model_spread`` is a predicted home MARGIN
    (POSITIVE when the home team is expected to win); ``closing_spread`` is a market LINE (NEGATIVE
    when the home team is favored). ``BettingSimulator._determine_bet_side_ats`` compares two
    LINES and ``BettingSimulator._resolve_ats_outcome`` compares an actual MARGIN against a
    threshold on the MARGIN scale, so ``ATSStrategy`` CONVERTS AT BOTH SEAMS -- the model's implied
    line is the negated margin going in, and the cover threshold is the negated slipped line coming
    out. It does NOT re-implement either LOCKED helper; it hands each one arguments in the
    convention that helper was written for. ``backtest/ats_ev_chain.py``'s module docstring records
    the same conversion and the legacy simulator path that omits it.
  * The WP target is quoted PER GAME. Its per-bet EV and its Kelly stake are computed at the side's
    own moneyline through the OPTIONAL ``bet_odds`` member, never at the flat -110 the spread and
    totals markets are quoted at. Pricing a -320 favourite at -110 turns a losing bet into a
    +0.43 EV one, which is the same class of defect as sizing Kelly off a points distance.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.ou_ev_chain import calibrated_p_over
from backtest.simulation import (
    SLIPPAGE_POINTS,
    BettingSimulator,
    SimulationConfig,
    apply_slippage_spread,
    apply_slippage_total,
)
from models.clv import compute_line_clv, compute_probability_clv

__all__ = [
    "NO_SUBPOPULATION_LABEL",
    "ATSStrategy",
    "OUStrategy",
    "TargetStrategy",
    "UnregisteredTargetError",
    "WPStrategy",
    "default_strategies",
    "require_finite_high_total_boundary",
]

# The eligibility label the two targets WITHOUT a sub-population report (D31-05). It is a constant
# rather than an empty string or None so the page renders a definite statement -- "this target has
# no sub-population" -- instead of a blank cell a reader would have to interpret.
NO_SUBPOPULATION_LABEL: str = "no_subpopulation"


# ---------------------------------------------------------------------------
# The two Phase-31 chains, imported LAZILY. This is a cycle break, not a soft dependency.
#
# ``backtest.ats_ev_chain`` -> ``backtest.ev_chain_constants`` -> ``backtest.ou_monetization`` ->
# ``backtest.bet_selector`` -> THIS MODULE. A module-scope import here therefore fails at
# collection with "cannot import name 'OUStrategy' from partially initialized module". The same
# break, for the same reason, is already used by ``BetSelector._freshness_context``.
# ---------------------------------------------------------------------------


def _ats_chain() -> Any:
    """The ATS EV chain module (lazy -- see the cycle note above)."""
    from backtest import ats_ev_chain

    return ats_ev_chain


def _wp_chain() -> Any:
    """The WP EV chain module (lazy -- see the cycle note above)."""
    from backtest import wp_ev_chain

    return wp_ev_chain


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
            eligibility GATE; they still refuse a candidate with no bet side, because there is no
            bet to price, and they report that with ``"no_bet_side"`` rather than with the O/U
            sub-population reason.
        eligibility_label: A human-readable label naming WHICH eligibility arm(s) the candidate
            satisfies. Only O/U has a sub-population concept (D27-04/05); targets without one
            report the constant :data:`NO_SUBPOPULATION_LABEL`.
        side_probability: The calibrated ``P(side)`` and the slipped line, as a pair. This is the
            number Kelly consumes (BET-02) -- never a points distance. The line is ``None`` for a
            target that has no line to slip: WP is a moneyline bet, and returning 0.0 there would
            be a made-up line rather than an absent one.
        decision_extras: Target-specific REPORTING fields merged into the decision record (for O/U:
            the totals regime and the model-edge CLV). They keep target vocabulary out of the core.
        grade: The push-aware outcome, delegating to the LOCKED ``_resolve_*_outcome`` resolver.

    TWO OPTIONAL MEMBERS, DELIBERATELY OUTSIDE THIS PROTOCOL. Both are read by the core through
    ``getattr`` with a documented default, so adding either to a strategy does not un-conform every
    other strategy -- and so a target that does not need one carries no empty implementation:

      * ``required_prediction_fields`` (plan 31-09): the subset of ``required_market_fields`` that
        are MODEL outputs, which splits ``missing_prediction`` from ``missing_snapshot``. Undeclared,
        the core classifies by the ``model_`` naming convention.
      * ``bet_odds(row, bet_side) -> int`` (plan 31-10): the American odds for the side actually
        bet. Undeclared, the core prices and sizes at its own reference juice (-110), which is what
        the spread and totals markets are quoted at. WP declares it because a moneyline is quoted
        per game and per side.
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
    ) -> tuple[float, float | None]:
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


class ATSStrategy:
    """The spread target's selection path (Phase 31, plan 31-10; D31-04/05, SPEC R1).

    Satisfies the ``TargetStrategy`` Protocol via structural subtyping (the ``FeatureBuilder``
    precedent in ``features/protocol.py``): conformance is declared here, in the docstring, and
    checked structurally -- there is no base class.

    NO ELIGIBILITY GATE (D31-05). The calibrated chain runs on every candidate and the EV floor
    alone decides. O/U's under-OR-high-total UNION was earned by an entire phase of pre-registered
    sub-population sweeping with multiplicity correction; ATS (pooled CLV -0.0015, p 0.990) has had
    no such diagnosis, and a zero-bet chain is an explicitly defined PASS. A pre-registered
    sub-population search per target was REJECTED as an unscoped diagnosis that would discover a
    rule on contaminated, partly-burned data and then spend the single clean 2025 split validating
    it; declaring the target report-only by construction was REJECTED because it pre-decides the
    verdict, and a guaranteed answer is not evidence.

    THE TWO SEAMS THIS CLASS CONVERTS AT, AND WHY EACH ONE MATTERS. ``model_spread`` is a predicted
    home MARGIN; ``closing_spread`` is a market LINE. They are numerically opposite for the same
    opinion, so handing either helper the wrong one silently prices the wrong side of every game:

      * ``resolve_bet_side`` passes the model's IMPLIED LINE (``-model_spread``) against the market
        line, because ``BettingSimulator._determine_bet_side_ats`` compares two LINES. Its own
        docstring's "model thinks home wins by more than market" only holds under that reading.
      * ``grade`` passes the NEGATED slipped line as the cover threshold, because
        ``BettingSimulator._resolve_ats_outcome`` compares the actual home MARGIN against it and
        ``models/train.py:200-202`` grades a home cover as ``actual_margin + spread > 0``, i.e.
        ``actual_margin > -spread``.

    Neither LOCKED helper is re-implemented; each is called with arguments in the convention it was
    written for, which is the same conversion ``backtest.ats_ev_chain.price_ats_candidates``
    performs.

    ``slipped_line`` is carried on the MARKET convention -- the price the bettor actually got, which
    is what a bet list renders -- and the margin-scale conversion happens at the grading seam.
    """

    target = "ats"
    # The predicted home MARGIN and the market LINE. ``model_spread`` carries the ``model_``
    # prefix, so the core's D31-19 classifier reports its absence as ``missing_prediction`` and
    # ``closing_spread``'s as ``missing_snapshot`` without a declared override.
    required_market_fields: tuple[str, ...] = ("model_spread", "closing_spread")

    def __init__(
        self,
        frozen_sd: float,
        season_bias_by_season: Mapping[int, float],
        slippage_points: float = SLIPPAGE_POINTS,
        simulator: BettingSimulator | None = None,
    ) -> None:
        """Build the spread strategy.

        Args:
            frozen_sd: The single frozen residual SD on the HOME-MARGIN scale, fit on
                bias-corrected TUNE residuals only. It is NOT the O/U frozen SD: that one is on the
                total scale and the two are different quantities that happen to be similar numbers.
            season_bias_by_season: Target season -> the prior-season walk-forward mean residual.
            slippage_points: The half-point slippage, applied through the LOCKED
                ``apply_slippage_spread`` in the LINE convention it was written for.
            simulator: The injected simulator, so exactly one exists per selector.
        """
        self.frozen_sd = float(frozen_sd)
        self.season_bias_by_season = dict(season_bias_by_season)
        self.slippage_points = float(slippage_points)
        self._sim = (
            simulator if simulator is not None else BettingSimulator(SimulationConfig())
        )

    # -- TargetStrategy Protocol surface --------------------------------------

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        """The ATS side via the LOCKED convention, on the model's IMPLIED LINE.

        The side is resolved on the RAW prediction, exactly as the O/U strategy resolves its side
        on the raw model total; the bias correction enters the PROBABILITY, not the side.
        """
        return self._sim._determine_bet_side_ats(
            -float(row["model_spread"]), float(row["closing_spread"])
        )

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        """No eligibility gate (D31-05); a candidate with no side has no bet to price.

        The sideless case is reported as ``"no_bet_side"`` and NOT as the O/U sub-population
        reason: this target has no sub-population, so claiming a candidate fell outside one would
        assert a gate that does not exist. It is not reported as an EV failure either -- nothing was
        priced, so no expected value was measured.
        """
        return None if bet_side is not None else "no_bet_side"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        """The constant label for a target with no sub-population (D31-05)."""
        return NO_SUBPOPULATION_LABEL

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float | None]:
        """The calibrated P(side) and the half-point-slipped MARKET line (BET-02).

        Slippage is applied in the LINE convention ``apply_slippage_spread`` was written for, then
        NEGATED into the margin-scale cover threshold the converter uses. Returns the slipped LINE
        so the record carries the price the bettor got; ``grade`` performs the same negation.
        """
        chain = _ats_chain()
        model_spread = float(row["model_spread"])
        closing_spread = float(row["closing_spread"])
        slipped_line = apply_slippage_spread(
            closing_spread, bet_side, self.slippage_points
        )
        cover_threshold = -slipped_line
        season_bias = chain.season_bias_for(
            int(row["season"]), self.season_bias_by_season, target=self.target
        )
        p_home_cover = float(
            chain.calibrated_p_cover(
                model_spread, cover_threshold, self.frozen_sd, season_bias
            )
        )
        return chain.ats_side_probability(bet_side, p_home_cover), slipped_line

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """The REPORT-ONLY model-edge line CLV (``closing_spread - model_spread``).

        Reported, never a gate (D27-06). This is the MODEL EDGE against the line, DISTINCT from the
        freeze-vs-close forward metric, which is structurally ~0 in backtest.
        """
        return {
            "clv": compute_line_clv(
                float(row["model_spread"]),
                float(row["closing_spread"]),
                direction="spread",
            )
        }

    def grade(self, record: dict[str, Any]) -> bool | None:
        """Grade via the LOCKED ``_resolve_ats_outcome`` on the MARGIN scale (push-aware).

        The realized value the core stashed under ``_actual_total`` is, for this target, the actual
        home MARGIN (``home_score - away_score``) -- the stash is target-agnostic and named for the
        target that introduced it. The cover threshold handed to the resolver is the NEGATED slipped
        line, so a home cover is graded as ``actual_margin > -slipped_line``, which is exactly the
        rule ``models/train.py`` states. The push (margin exactly on the threshold) is carried as
        None, never coerced; a candidate with no realized value is ungraded, also None.
        """
        actual_margin = record.get("_actual_total")
        if actual_margin is None or record.get("slipped_line") is None:
            return None
        return self._sim._resolve_ats_outcome(
            record["bet_side"], float(actual_margin), -float(record["slipped_line"])
        )


class WPStrategy:
    """The winner target's selection path (Phase 31, plan 31-10; D31-05/07, SPEC R1).

    Satisfies the ``TargetStrategy`` Protocol via structural subtyping, declared here in the
    docstring and checked structurally -- there is no base class.

    NO ELIGIBILITY GATE (D31-05), on the same rule as the spread target: WP (pooled CLV -0.0380,
    t -15.52) has had no sub-population diagnosis, and inventing one on contaminated, partly-burned
    data would spend the single clean split validating a rule discovered on it.

    THE PROBABILITY IS THE MODEL'S OWN (D31-07 default). The deployed artifact carries an isotonic
    calibrator and the bet list is priced off THAT model, unchanged. The REGISTERED fallback -- a
    prior-season probability-scale shift -- fires only when a ``WPGateResult`` reporting a failed
    tune-split calibration gate is supplied, and it can never fire silently: ``fallback_fired`` and
    ``fallback_trigger`` travel onto every decision record.

    THE PRICE IS THE GAME'S OWN. A moneyline is quoted per game and per side, so this strategy
    declares the optional ``bet_odds`` member and the core prices its EV and sizes its Kelly stake
    at that price. The flat -110 default the spread and totals markets carry would turn a -320
    favourite priced at a true 0.75 win probability from a losing bet into a +0.43 EV one.
    """

    target = "wp"
    # The deployed isotonic P(home) and BOTH real moneylines. Both prices are REQUIRED: a moneyline
    # market has a real two-sided price by construction, so a missing one is missing DATA and
    # defaulting it would invent a market no book offered.
    required_market_fields: tuple[str, ...] = ("model_prob", "ml_home", "ml_away")

    def __init__(
        self,
        season_bias_by_season: Mapping[int, float] | None = None,
        gate: Any | None = None,
        simulator: BettingSimulator | None = None,
    ) -> None:
        """Build the winner strategy.

        Args:
            season_bias_by_season: Target season -> the prior-season walk-forward probability-scale
                bias. Consulted ONLY when the registered fallback fired; an empty mapping is
                correct on the default path and raises by name if the fallback is later switched on
                without one.
            gate: The TUNE-split ``WPGateResult``. None is the DEFAULT path -- the deployed
                probability used unchanged, no fallback -- and the fallback fields are stamped onto
                every record either way, so "no gate was run" is never mistaken for "the gate
                passed".
            simulator: The injected simulator, so exactly one exists per selector.
        """
        self.season_bias_by_season = dict(season_bias_by_season or {})
        self.gate = gate
        self.fallback_fired = bool(gate is not None and gate.fallback_fired)
        self.fallback_trigger = gate.fallback_trigger if gate is not None else None
        self._sim = (
            simulator if simulator is not None else BettingSimulator(SimulationConfig())
        )

    # -- internals ------------------------------------------------------------

    def _p_home(self, row: dict[str, Any]) -> float:
        """P(home) for ``row``: the deployed probability, or the registered fallback's correction.

        ONE implementation, consumed by the side rule, the probability and the CLV, because WP's
        side rule reads the probability itself -- a fired fallback that moves a probability across
        the side threshold must move the side with it.
        """
        model_prob = float(row["model_prob"])
        if not self.fallback_fired:
            return model_prob
        chain = _wp_chain()
        season_bias = chain.season_bias_for(
            int(row["season"]), self.season_bias_by_season, target=self.target
        )
        return float(chain.apply_wp_fallback_correction(model_prob, season_bias))

    # -- TargetStrategy Protocol surface --------------------------------------

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        """The WP side via the LOCKED ``_determine_bet_side_wp``, on the priced probability."""
        return self._sim._determine_bet_side_wp(self._p_home(row))

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        """No eligibility gate (D31-05); a candidate with no side has no bet to price.

        ``model_prob`` inside the LOCKED no-bet band around 0.5 yields no side, which is common on
        this target rather than exotic. It is reported as ``"no_bet_side"`` for the same reason the
        spread target reports it that way: this target has no sub-population to fall outside of,
        and nothing was priced, so no expected value was measured.
        """
        return None if bet_side is not None else "no_bet_side"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        """The constant label for a target with no sub-population (D31-05)."""
        return NO_SUBPOPULATION_LABEL

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float | None]:
        """The side-correct probability, and NO line.

        A moneyline bet has no line to slip, so the second element is None rather than 0.0: a
        zero would be a made-up line, and ``BetRecord.slipped_line`` already documents None as the
        WP value.
        """
        return (
            float(_wp_chain().calibrated_p_home_side(self._p_home(row), bet_side)),
            None,
        )

    def bet_odds(self, row: dict[str, Any], bet_side: str) -> int:
        """The American odds for the side actually bet, via the LOCKED ``_get_wp_odds``.

        The OPTIONAL Protocol member (see ``TargetStrategy``): declaring it is what makes the core
        price and size this target at the game's own moneyline instead of at its reference juice.
        """
        return self._sim._get_wp_odds(
            bet_side, float(row["ml_home"]), float(row["ml_away"])
        )

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """The fallback registration and the REPORT-ONLY probability CLV.

        The two fallback fields are stamped on EVERY record, sided or not, because their job is to
        make a fired fallback impossible to infer from the numbers rather than read off the row.
        The CLV is measured against the devigged fair closing price through the LOCKED
        ``compute_probability_clv``; it is reported, never a gate (D27-06).
        """
        extras: dict[str, Any] = {
            "fallback_fired": self.fallback_fired,
            "fallback_trigger": self.fallback_trigger,
        }
        if bet_side is None:
            return extras
        p_side = float(_wp_chain().calibrated_p_home_side(self._p_home(row), bet_side))
        extras["clv"] = compute_probability_clv(
            model_prob=p_side,
            closing_ml_home=float(row["ml_home"]),
            closing_ml_away=float(row["ml_away"]),
            side=bet_side,
        )["probability_clv"]
        return extras

    def grade(self, record: dict[str, Any]) -> bool | None:
        """Grade via the LOCKED ``_resolve_wp_outcome``.

        The realized value the core stashed under ``_actual_total`` is, for this target, the 0/1
        home-win label. A moneyline bet cannot push, so the only None here is UNGRADED -- a forward
        game that has not been played.
        """
        actual_home_win = record.get("_actual_total")
        if actual_home_win is None:
            return None
        return self._sim._resolve_wp_outcome(record["bet_side"], int(actual_home_win))


# ---------------------------------------------------------------------------
# The three-target registry
# ---------------------------------------------------------------------------


def default_strategies(
    *,
    ou_frozen_sd: float,
    ou_season_bias_by_season: Mapping[int, float],
    ats_frozen_sd: float,
    ats_season_bias_by_season: Mapping[int, float],
    wp_season_bias_by_season: Mapping[int, float] | None = None,
    wp_gate: Any | None = None,
    high_total_boundary: float = HIGH_TOTAL_BOUNDARY_PREHOLD,
    slippage_points: float = SLIPPAGE_POINTS,
    simulator: BettingSimulator | None = None,
) -> list[Any]:
    """Build the three production strategies, in the repository's canonical target order.

    Passing the result as ``BetSelector(..., strategies=...)`` is what makes a mixed week produce
    records for all three targets in ONE ``select`` call, with the 10% weekly exposure cap pooled
    over their union (D31-02) and the correlated de-weight grouping by same-GAME across them
    (D31-03).

    THE PER-TARGET FIT PARAMETERS ARE SEPARATE ARGUMENTS ON PURPOSE. ``ou_frozen_sd`` is on the
    TOTAL scale and ``ats_frozen_sd`` is on the HOME-MARGIN scale; they are different quantities
    that happen to be similar numbers, and one shared ``frozen_sd`` would be a category error that
    typechecks. WP fits no residual SD at all (D31-07): a calibrated classifier has no residual to
    take a standard deviation of, and inventing a logit-space one was explicitly rejected.

    ``BetSelector``'s own default registry is NOT changed by this function and still registers the
    O/U strategy alone, so every pre-D31-01 call site -- including the simulator's O/U routing,
    whose candidate rows carry no ``target`` column -- behaves exactly as before.

    Args:
        ou_frozen_sd: The O/U frozen residual SD (total scale).
        ou_season_bias_by_season: The O/U prior-season walk-forward bias per season.
        ats_frozen_sd: The ATS frozen residual SD (home-margin scale).
        ats_season_bias_by_season: The ATS prior-season walk-forward bias per season.
        wp_season_bias_by_season: The WP probability-scale bias per season; consulted only when
            ``wp_gate`` reports a failed calibration gate.
        wp_gate: The WP TUNE-split ``WPGateResult``, or None for the default path.
        high_total_boundary: The leakage-clean PRE-HOLD O/U eligibility boundary (LOCKED-1).
        slippage_points: The half-point slippage applied to the two line targets.
        simulator: One injected simulator shared by all three strategies.

    Returns:
        ``[WPStrategy, ATSStrategy, OUStrategy]`` -- the ``wp / ats / ou`` order this repository
        uses everywhere, so a registry listing reads the same as every other per-target table.
    """
    return [
        WPStrategy(
            season_bias_by_season=wp_season_bias_by_season,
            gate=wp_gate,
            simulator=simulator,
        ),
        ATSStrategy(
            frozen_sd=ats_frozen_sd,
            season_bias_by_season=ats_season_bias_by_season,
            slippage_points=slippage_points,
            simulator=simulator,
        ),
        OUStrategy(
            frozen_sd=ou_frozen_sd,
            season_bias_by_season=dict(ou_season_bias_by_season),
            high_total_boundary=high_total_boundary,
            slippage_points=slippage_points,
            simulator=simulator,
        ),
    ]
