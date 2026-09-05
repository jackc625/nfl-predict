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

  * The ATS target's two spread quantities are ON THE SAME SCALE (DEF-31-01, ruled 2026-09-04).
    ``model_spread`` is a predicted home MARGIN and ``closing_spread`` is the stored nflverse
    ``spread_line``, POSITIVE when the home team is favored and already the cover threshold on
    that same margin scale. What DOES need converting is the pair of LOCKED helpers written in the
    opposite line convention: ``BettingSimulator._determine_bet_side_ats`` (which returns
    ``home_cover`` when its first argument is the smaller) and ``apply_slippage_spread`` (which
    moves a LINE against the bettor). ``ATSStrategy`` negates INTO those two and negates BACK OUT,
    and re-implements neither. ``BettingSimulator._resolve_ats_outcome`` already grades
    ``actual_margin > slipped_line`` and needs no conversion at all.
    ``backtest/ats_ev_chain.py``'s module docstring records the measurement, the two halves of the
    ruling that were declined, and the legacy simulator path left on its older reading.
  * EVERY target is priced at the price a book actually offered, through the OPTIONAL ``bet_odds``
    member. WP reads the side's own moneyline. The two LINE targets read the stored two-sided
    juice -- ``spread_ju_home`` / ``spread_ju_away`` for the spread and ``total_over_ju`` /
    ``total_under_ju`` for the total -- through the EXISTING devig, and fall back to the selector's
    reference juice (-110) only when a row carries no real two-sided price. Pricing a -320
    favourite at -110 turns a losing bet into a +0.43 EV one, which is the same class of defect as
    sizing Kelly off a points distance.

    THE TWO LINE TARGETS DID NOT ALWAYS DO THIS, AND THE CHANGE WAS AN OWNER RULING (DEF-31-13,
    ruled 2026-09-05). D31-04 called the spread and totals markets "the two flat-quoted targets"
    and gave ``bet_odds`` to ``WPStrategy`` alone, so ATS and O/U priced, sized and paid out at a
    flat -110 while the frozen ``PROFITABILITY-PREREGISTRATION.md`` said in two places (sections
    3.2 step 4 and 3.3 step 4) that both chains devig the real two-sided prices. Two ratified
    documents disagreed; the owner ruled that the frozen pre-registration governs the 2025 verdict,
    so that half of D31-04's characterisation is SUPERSEDED for the selection path. The juice is
    real and asymmetric: Plan 31-02 measured 1,992 of 2,120 distinct ``(game_id, sportsbook)``
    pairs carrying a price other than -110.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.ou_ev_chain import calibrated_p_over, devig
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
    "OU_JUICE_FIELDS",
    "OU_SIDES",
    "REAL_TWO_SIDED_METHOD",
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

# The LOCKED O/U side vocabulary, from ``BettingSimulator._determine_bet_side_ou``. Stated in the
# same (over, under) order as ``OU_JUICE_FIELDS`` and as ``devig``'s two arguments, so the side a
# bet is on and the column its price is read from cannot be paired up wrongly.
OU_SIDES: tuple[str, str] = ("over", "under")

# The two stored O/U juice columns (DEF-31-13, ruled 2026-09-05). Named as literals here for the
# same reason ``backtest.ats_ev_chain.ATS_JUICE_FIELDS`` names the spread pair as literals: the
# four juice columns are named individually in the frozen pre-registration (section 4.3 clause 2),
# so they are named individually in code. ``tests/unit/test_bet_selector.py`` pins these two
# against ``scripts.ingest_historical_odds.JUICE_COLUMNS`` -- the module that WRITES them -- so the
# reader's spelling and the writer's spelling cannot drift apart.
OU_JUICE_FIELDS: tuple[str, str] = ("total_over_ju", "total_under_ju")

# ``devig``'s label for a price it ACTUALLY devigged, as opposed to its flat -110 default. The
# strategies below read this off the returned dict rather than re-deciding "are both prices
# present?" locally, so there is exactly ONE definition of a real two-sided price in the codebase.
REAL_TWO_SIDED_METHOD: str = "real_two_sided"


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


# ---------------------------------------------------------------------------
# Reading a stored two-sided price (DEF-31-13, ruled 2026-09-05)
# ---------------------------------------------------------------------------


def _juice_price(row: dict[str, Any], column: str) -> float | None:
    """One stored juice value as a float, or None when it is ABSENT for pricing purposes.

    ABSENT means None, a DataFrame's NaN cell, or a nullable dtype's NA -- the three shapes a
    missing price actually arrives in. It does NOT mean MALFORMED: a value that is present and is
    not a number raises, naming the column, because a broken price is a data defect and pricing it
    at the flat reference juice would file it under the same label an honestly absent price
    carries. The two absences have different causes and different fixes, which is the same
    distinction ``BetSelector`` already draws between a missing market value and a missing model
    output.

    Args:
        row: The candidate row.
        column: The stored juice column to read.

    Returns:
        The price as a float, or None when the row carries none.

    Raises:
        ValueError: naming the column, when the value is present but is not a number.
    """
    value = row.get(column)
    if value is None:
        return None
    try:
        price = float(value)
    except TypeError:
        # A value that refuses float conversion BY TYPE is an absent cell (pandas' NA is the
        # case this exists for), not a malformed one.
        return None
    except ValueError as exc:
        msg = (
            f"stored juice column {column!r} holds {value!r}, which is not a number. An ABSENT "
            "price falls back to the flat reference juice (DEF-31-13, D27-13); a MALFORMED one "
            "is a data defect and is refused rather than quietly priced."
        )
        raise ValueError(msg) from exc
    if not math.isfinite(price):
        # A DataFrame writes a missing cell as NaN, so this is the ORDINARY absent case rather
        # than an exotic one. An infinite value is refused by the same branch: it is not a price.
        return None
    return price


def _side_odds_from_devig(method: str, side_price: float | None) -> int | None:
    """The American odds for the side actually bet, or None when there is no two-sided price.

    ``method`` comes from the EXISTING devig, never from a local "are both prices present?" test.

    None is a deliberate signal rather than a value: ``BetSelector._strategy_bet_odds`` maps it to
    the selector's own ``default`` (``STANDARD_VIG_ODDS``, the flat -110 the ATS chain has always
    documented for an absent price, D27-13). Returning that constant from here instead would
    silently ignore a selector configured with a different reference juice.

    The ``int`` cast matches the LOCKED ``BettingSimulator._get_wp_odds``, which casts a stored
    moneyline the same way; American odds are integral, and every stored juice value in the silver
    odds table is.
    """
    if method != REAL_TWO_SIDED_METHOD or side_price is None:
        return None
    return int(side_price)


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
      * ``bet_odds(row, bet_side) -> int | None`` (plan 31-10; widened by DEF-31-13, ruled
        2026-09-05): the American odds for the side actually bet, or None when this row carries no
        real two-sided price. Undeclared -- or declared and returning None -- the core prices and
        sizes at its own reference juice (-110). ALL THREE production strategies declare it: WP
        because a moneyline is quoted per game and per side, and the two line targets because the
        frozen pre-registration devigs the stored two-sided spread and total prices (sections 3.2
        step 4 and 3.3 step 4). The None return is what keeps the core's ``default`` meaningful:
        a strategy says "I have no price", and the SELECTOR decides what a missing price costs.
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

    THE PRICE IS THE STORED TWO-SIDED TOTAL PRICE (DEF-31-13, ruled 2026-09-05). ``bet_odds`` reads
    ``total_over_ju`` / ``total_under_ju`` through the EXISTING ``devig`` and returns the side's own
    price, so the per-bet EV, the Kelly stake and the flat payout are all struck at a price a book
    actually offered. A row carrying no two-sided price falls back to the selector's reference
    juice, and ``decision_extras`` stamps ``devig_method`` so that fallback is legible on the record
    rather than indistinguishable from a genuine -110. This is a PRICE change only: the eligibility
    rule, the side rule, the bias correction, the frozen SD and the slippage are all untouched, and
    a row whose stored price IS -110 reproduces the pre-ruling numbers exactly.
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

    def _two_sided_juice(
        self, row: dict[str, Any]
    ) -> tuple[float | None, float | None, str]:
        """The stored ``(over, under)`` prices and the METHOD the EXISTING devig reports for them.

        CONSUMES ``backtest.ou_ev_chain.devig`` -- the O/U-NATIVE primitive, whose parameters are
        literally ``over_odds`` and ``under_odds``, and which
        ``backtest.ats_ev_chain.ats_two_sided_prices`` is itself only an adapter around. No
        proportional arithmetic is written here (the pre-registration's "no new devig
        implementation"), and the question "is there a real two-sided price on this row?" is
        answered in exactly one place for both line targets.

        Returns:
            ``(over_price, under_price, method)``; either price is None when absent, and ``method``
            is ``devig``'s own label (:data:`REAL_TWO_SIDED_METHOD` or its flat -110 default).
        """
        over = _juice_price(row, OU_JUICE_FIELDS[0])
        under = _juice_price(row, OU_JUICE_FIELDS[1])
        return over, under, str(devig(over_odds=over, under_odds=under)["method"])

    def bet_odds(self, row: dict[str, Any], bet_side: str) -> int | None:
        """The stored price for the side actually bet, or None when the row carries none.

        The OPTIONAL Protocol member (see ``TargetStrategy``). The frozen pre-registration
        section 3.3 step 4 devigs "the real two-sided total prices (``total_over_ju``,
        ``total_under_ju``)", and DEF-31-13's 2026-09-05 ruling is that the frozen rule governs.
        Returning None hands the flat -110 fallback back to the selector rather than asserting one
        here.

        Raises:
            ValueError: naming the LOCKED O/U side vocabulary, for any other side string. A side
                outside it would otherwise silently read the WRONG half of the two-sided price.
        """
        over, under, method = self._two_sided_juice(row)
        over_side, under_side = OU_SIDES
        if bet_side == over_side:
            side_price = over
        elif bet_side == under_side:
            side_price = under
        else:
            msg = (
                f"unknown O/U bet side {bet_side!r}; the LOCKED O/U side vocabulary is "
                f"{list(OU_SIDES)} (BettingSimulator._determine_bet_side_ou). A side outside it "
                "is a caller error, never a default."
            )
            raise ValueError(msg)
        return _side_odds_from_devig(method, side_price)

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """The O/U reporting fields: the totals regime, the model-edge CLV and the devig method.

        The CLV here is ``compute_line_clv(model_total, closing_total, direction="total")`` -- the
        MODEL EDGE vs the line, DISTINCT from the freeze-vs-close forward metric (structurally ~0
        in backtest). It is reported, never a gate (D27-06/12).

        ``devig_method`` is what makes an ABSENT price distinguishable from a real -110 in the
        record (DEF-31-13). Without it the two are the same number with two different meanings,
        and a reader could not tell a bet a book actually quoted at -110 from one nobody quoted.
        It is a statement about the STORED MARKET DATA on this row, so it is stamped whether or
        not a side was found -- the same treatment ``totals_regime`` already gets.
        """
        model_total = float(row["model_total"])
        closing_total = float(row["closing_total"])
        _over, _under, method = self._two_sided_juice(row)
        return {
            "totals_regime": self._totals_regime(closing_total),
            "clv": compute_line_clv(model_total, closing_total, direction="total"),
            "devig_method": method,
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

    THE ONE SCALE, AND THE TWO HELPERS THAT ARE NOT ON IT (DEF-31-01, ruled 2026-09-04).
    ``model_spread`` is a predicted home MARGIN and ``closing_spread`` is the stored nflverse
    ``spread_line`` -- POSITIVE when the home team is favored, and already the margin the home team
    must EXCEED to cover. They are directly comparable. Two LOCKED helpers are written in the
    OPPOSITE line convention and are negated into, never re-implemented:

      * ``resolve_bet_side`` negates BOTH margins into ``BettingSimulator._determine_bet_side_ats``,
        which returns ``home_cover`` when its first argument is the smaller. The net rule is the
        plain one: bet the home side when the model expects a bigger home margin than the market.
      * ``side_probability`` negates the stored spread into ``apply_slippage_spread`` and negates
        the result back, so a home-cover bet's threshold RISES by the slippage rather than falling.

    ``grade`` converts NOTHING: ``BettingSimulator._resolve_ats_outcome`` already grades
    ``home_covers = actual_margin > slipped_line``, which is the measured convention exactly. This
    is the same arrangement ``backtest.ats_ev_chain.price_ats_candidates`` uses.

    ``slipped_line`` is therefore both the LINE the bettor actually got (what a bet list renders)
    and the cover threshold the game must clear -- one number, not two.

    THE PRICE IS THE STORED TWO-SIDED SPREAD PRICE (DEF-31-13, ruled 2026-09-05). ``bet_odds`` reads
    ``spread_ju_home`` / ``spread_ju_away`` through ``ats_ev_chain.ats_two_sided_prices`` -- the
    adapter that already devigs exactly this pair through the EXISTING ``devig`` and "writes no
    proportional arithmetic of its own" -- and returns the side's own price. The per-bet EV, the
    Kelly stake and the flat payout are therefore all struck at a price a book actually offered,
    which is what the frozen pre-registration section 3.2 step 4 says this chain does. A row
    carrying no two-sided price falls back to the selector's reference juice, and
    ``decision_extras`` stamps ``devig_method`` so that fallback is legible on the record rather
    than indistinguishable from a genuine -110.
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
        """The ATS side via the LOCKED helper, with BOTH margins negated into its convention.

        The side is resolved on the RAW prediction, exactly as the O/U strategy resolves its side
        on the raw model total; the bias correction enters the PROBABILITY, not the side.
        """
        return self._sim._determine_bet_side_ats(
            -float(row["model_spread"]), -float(row["closing_spread"])
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
        """The calibrated P(side) and the half-point-slipped stored SPREAD (BET-02).

        Slippage is applied by negating into the line convention ``apply_slippage_spread`` was
        written for and negating its result back, which moves the stored spread AGAINST the bettor
        on the margin scale. The result IS the cover threshold, so no further conversion happens
        here or at the grading seam -- ``grade`` hands this same number to the resolver.
        """
        chain = _ats_chain()
        model_spread = float(row["model_spread"])
        closing_spread = float(row["closing_spread"])
        slipped_line = -apply_slippage_spread(
            -closing_spread, bet_side, self.slippage_points
        )
        cover_threshold = slipped_line
        season_bias = chain.season_bias_for(
            int(row["season"]), self.season_bias_by_season, target=self.target
        )
        p_home_cover = float(
            chain.calibrated_p_cover(
                model_spread, cover_threshold, self.frozen_sd, season_bias
            )
        )
        return chain.ats_side_probability(bet_side, p_home_cover), slipped_line

    def _two_sided_juice(
        self, row: dict[str, Any]
    ) -> tuple[float | None, float | None, str]:
        """The stored ``(home_cover, away_cover)`` prices and the devig METHOD for them.

        CONSUMES ``backtest.ats_ev_chain.ats_two_sided_prices``, which devigs this exact pair
        through the shared ``devig`` and explicitly "writes no proportional arithmetic of its own".
        The column names come from that module's ``ATS_JUICE_FIELDS`` rather than being restated
        here, so the chain and the selection path read the same two columns by construction.

        Returns:
            ``(home_cover_price, away_cover_price, method)``; either price is None when absent.
        """
        chain = _ats_chain()
        home = _juice_price(row, chain.ATS_JUICE_FIELDS[0])
        away = _juice_price(row, chain.ATS_JUICE_FIELDS[1])
        return home, away, str(chain.ats_two_sided_prices(home, away)["method"])

    def bet_odds(self, row: dict[str, Any], bet_side: str) -> int | None:
        """The stored price for the side actually bet, or None when the row carries none.

        The OPTIONAL Protocol member (see ``TargetStrategy``). The frozen pre-registration
        section 3.2 step 4 devigs "the real two-sided spread prices (``spread_ju_home``,
        ``spread_ju_away``)", and DEF-31-13's 2026-09-05 ruling is that the frozen rule governs.
        Returning None hands the flat -110 fallback back to the selector rather than asserting one
        here.

        Raises:
            ValueError: naming the LOCKED ATS side vocabulary, for any other side string. A side
                outside it would otherwise silently read the WRONG half of the two-sided price.
        """
        home, away, method = self._two_sided_juice(row)
        home_side, away_side = _ats_chain().ATS_SIDES
        if bet_side == home_side:
            side_price = home
        elif bet_side == away_side:
            side_price = away
        else:
            msg = (
                f"unknown ATS bet side {bet_side!r}; the LOCKED ATS side vocabulary is "
                f"{list(_ats_chain().ATS_SIDES)} "
                "(BettingSimulator._determine_bet_side_ats). A side outside it is a caller "
                "error, never a default."
            )
            raise ValueError(msg)
        return _side_odds_from_devig(method, side_price)

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        """The REPORT-ONLY model-edge line CLV, and the devig method the price came from.

        Reported, never a gate (D27-06). This is the MODEL EDGE against the market number, DISTINCT
        from the freeze-vs-close forward metric, which is structurally ~0 in backtest.

        READ THE SIGN OFF THE FORMULA, NOT OFF ``compute_line_clv``'s docstring example. That
        example is written in the line convention; here both arguments are home MARGINS
        (DEF-31-01), so a POSITIVE value means the market favors the home team MORE than the model
        does. The DEF-31-01 ruling deliberately left this report-only number where it was rather
        than re-signing a published figure outside the scope it ruled on.

        ``devig_method`` is what makes an ABSENT price distinguishable from a real -110 in the
        record (DEF-31-13). It is a statement about the STORED MARKET DATA on this row, so it is
        stamped whether or not a side was found.
        """
        _home, _away, method = self._two_sided_juice(row)
        return {
            "clv": compute_line_clv(
                float(row["model_spread"]),
                float(row["closing_spread"]),
                direction="spread",
            ),
            "devig_method": method,
        }

    def grade(self, record: dict[str, Any]) -> bool | None:
        """Grade via the LOCKED ``_resolve_ats_outcome``, converting NOTHING (push-aware).

        The realized value the core stashed under ``_actual_total`` is, for this target, the actual
        home MARGIN (``home_score - away_score``) -- the stash is target-agnostic and named for the
        target that introduced it. The resolver already grades
        ``home_covers = actual_margin > slipped_line``, and under the measured convention
        (DEF-31-01) the slipped stored spread IS that threshold, so it is handed over un-negated.
        The push (margin exactly on the threshold) is carried as None, never coerced; a candidate
        with no realized value is ungraded, also None.
        """
        actual_margin = record.get("_actual_total")
        if actual_margin is None or record.get("slipped_line") is None:
            return None
        return self._sim._resolve_ats_outcome(
            record["bet_side"], float(actual_margin), float(record["slipped_line"])
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
    at that price. The selector's flat -110 reference juice would turn a -320 favourite priced at a
    true 0.75 win probability from a losing bet into a +0.43 EV one.

    THIS STRATEGY NEVER FALLS BACK, and that is why it declares no ``devig_method``. Both
    moneylines are REQUIRED market fields, so a row without them is suppressed as
    ``missing_snapshot`` before it is ever priced; there is no shape in which a WP bet is struck at
    the reference juice. The two line targets stamp ``devig_method`` precisely because they CAN
    fall back (DEF-31-13).
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
