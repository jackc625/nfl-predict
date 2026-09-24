"""Single-source bet-decision engine (Phase 27, plan 27-03; BET-01/02, OUM-04/06).

``BetSelector.select()`` is the ONE place bet decisions are made (BET-01, LOCKED-2). It is a
THIN HARNESS in the house style of ``backtest/diagnose.py`` and ``backtest/ou_divergence.py``: it
CALLS the LOCKED scorers and never re-derives a metric. Responsibilities, in order:

  1. Provenance hard-fail (OUM-06, T-27-07): ``assert_real_odds`` rejects any odds row whose
     sportsbook is outside {consensus, draftkings} or whose ``is_live`` is True, raising a
     ValueError naming the offending game_ids. Called BEFORE any selection.
  2. Eligibility: NO target has an eligibility gate. D31-05 gave WP and ATS none, and D33.2-24
     DELETED the O/U under-OR-high-total UNION (and the pre-hold high-total boundary it compared
     against) that used to sit here. A candidate is eligible iff its strategy resolves a bet side;
     a sideless candidate is rejected as ``no_bet_side``. The EV floor alone decides the rest.
  3. EV admission (D27-14): within the eligible set, a bet is admitted iff its per-bet EV is at or
     above the EV-floor scalar ``t``. The comparison REJECTS strictly below the floor, so EV
     EXACTLY EQUAL to the floor is ADMITTED (the SPEC R1 adjacency edge). The EV uses the
     calibrated P(side) from the Plan-01 EV chain (``ou_ev_chain.calibrated_p_over``) evaluated
     against the half-point-slipped line (the line moves against the bettor, via the LOCKED
     ``apply_slippage_total``).
  4. Sizing (BET-02 fix, T-27-08): Kelly consumes the CALIBRATED P(side) -- never the points
     distance ``implied + abs(model_total - closing_total)`` -- through
     ``KellyCalculator.calculate_optimal_bet_size(model_prob=p_side, ...)``, then the Plan-02
     LOCKED-order ``apply_sizing_pipeline`` (kelly -> 5% per-bet -> same-game-and-side
     de-weight -> 10% weekly cap), ONCE per week over the POOLED union of that week's bets across
     every registered target (D31-02). Unit = 1% of bankroll (D27-09).
  5. Push handling (#8, T-27-23): grading uses the LOCKED ``_resolve_ou_outcome``; a push is carried
     as ``outcome=None`` on the record, never coerced to win/loss.
  6. CLV reporting (D27-06/12, REPORT-ONLY, T-27-10): per bet ``compute_line_clv(model_total,
     closing_total, direction="total")`` -- the MODEL EDGE vs the line (the non-zero line_clv,
     ~+1.11 pooled) -- is attached and summarized via ``diagnose.clv_significance``. This is
     DISTINCT from the separate freeze-vs-close forward metric, which is structurally ~0 here
     because stored silver carries freeze == close (one synthetic snapshot per game); the backtest
     CLV-vs-close is a LIVE-ONLY FORWARD metric (D27-12). CLV NEVER gates a bet -- it is computed
     and reported only. Do not imply the backtest line_clv is meaningful historical forward
     evidence.

D31-01 (Phase 31, plan 31-06) split this module into a TARGET-AGNOSTIC CORE (here) plus one
strategy per target (``backtest/selector_strategies.py``). The facade is UNCHANGED and deliberately
so: SPEC R4's source scan pins ``backtest.bet_selector`` as the ONE import target for a bet
decision, and ``BetSelector`` / ``SelectionResult`` / ``assert_real_odds`` / ``REJECTION_REASONS``
remain its entire public surface. Constructed without a ``strategies`` argument the selector
registers the O/U strategy alone, so every pre-D31-01 call site behaves exactly as before. All three
strategies exist as of Plan 31-10; a caller wanting the pooled three-target week passes
``backtest.selector_strategies.default_strategies(...)``, whose per-target fit parameters this
constructor could not supply -- the ATS frozen SD is on the home-margin scale and the O/U one is on
the total scale, and WP fits no residual SD at all. The DEFAULT registry stays O/U-only precisely so
that the simulator's O/U routing, whose candidate rows carry no ``target`` column, is untouched.

D31-04 (Phase 31, plan 31-10) made the price PER TARGET AND PER SIDE. An eligible record carries
``selected_odds`` -- the game's own moneyline for WP, and, since DEF-31-13 was ruled on 2026-09-05,
the stored two-sided spread or total price for the two LINE targets -- and the per-bet EV and the
Kelly stake are both computed from it. A row that carries no real two-sided price falls back to
this selector's reference juice (-110) and says so on the record through ``devig_method``, so the
fallback is legible rather than indistinguishable from a genuine -110. See ``_strategy_bet_odds``
for the ruling: the frozen pre-registration devigs the real prices, D31-04 called those two targets
flat-quoted, and the owner ruled that the frozen rule governs the 2025 verdict.

D31-17/18/19 (Phase 31, plan 31-09) moved SUPPRESSION inside this module. Given a schedule,
``select()`` builds the candidate universe as every scheduled game times every REGISTERED target
and returns one record per pair -- live, or suppressed with exactly one reason from the SAME
``REJECTION_REASONS`` taxonomy the rejections already used. R6's "mirroring the rejected[]
taxonomy" is therefore literal identity rather than two lists kept in sync, which is the failure
mode this repo has already had once. Its precedent is ``assert_real_odds``: a pure data-provenance
check that already lived here and already ran before any selection.

The admissibility fence takes its lock instant and its comparison from ``utils.game_lock``,
through ``scripts.ingest_historical_odds`` -- the module that STAMPS ``snapshot_ts`` at ingest
(plan 31-08, D31-37) -- so the value written and the value compared come from ONE rule
(D33.2-01). Those imports are DEFERRED into ``_freshness_context`` because that module reaches
back to this one through ``backtest.ev_chain_constants`` -> ``backtest.ou_monetization``; see that
function's docstring.

``select()`` returns BOTH the FILTERED decision set (the acceptance basis) and the UNFILTERED
whole-population cross-check (D27-04), plus the REJECTED candidates with rejection reasons drawn
from the eight-member ``REJECTION_REASONS``.

It WRAPS -- never re-implements -- the LOCKED ``BettingSimulator`` side/slippage/outcome convention
(D-18). It makes NO change to ``models/clv.py``, ``config/gate.toml``, or any production artifact
(the LOCKED self-judge boundary). It does NOT import the exploratory Phase-26 EV preview from the
divergence harness (the no-leak guard, T-26-08); it consumes the production-grade Plan-01 EV chain.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from backtest.diagnose import clv_significance
from backtest.ou_ev_chain import (
    american_to_payout,
    per_bet_ev,
)
from backtest.selector_strategies import (
    OUStrategy,
    TargetStrategy,
    UnregisteredTargetError,
    is_absent,
)
from backtest.simulation import (
    SLIPPAGE_POINTS,
    STANDARD_VIG_ODDS,
    BettingSimulator,
    SimulationConfig,
)
from utils import get_logger
from utils.kelly_criterion import (
    KellyCalculator,
    KellyMode,
    apply_sizing_pipeline,
    apply_weekly_exposure_cap,
)

logger = get_logger(__name__)

# The rejection-reason taxonomy surfaced on rejected records (BET-01). An eligible candidate that
# is not bet is tagged with exactly one of these; the taxonomy is exported so callers/tests do not
# redefine the strings.
REJECTION_REASONS: tuple[str, ...] = (
    # outside the retired O/U UNION {under} OR {high-total} (D27-04/05). NO production strategy
    # emits it since D33.2-24 deleted that gate; it stays in the vocabulary because published
    # pre-33.2 records carry it and the page must still be able to label them (R16: the old-rule
    # record is kept and labelled, not erased).
    "not_subpop",
    "ev_below_floor",  # eligible but per-bet EV < the EV-floor t (D27-14)
    "real_odds_failed",  # provenance hard-fail (OUM-06) -- raised before selection
    "zero_kelly_stake",  # admitted by EV but the Kelly calculator zeroed the stake (WR-07)
    # snapshot NOT ADMISSIBLE at that game's OWN lock -- captured AFTER it (D33.2-01, Plan
    # 33.2-02). The token is kept because it is a published rejection_reason value; its meaning
    # moved from "before the freeze" (a staleness test) to "after the lock" (admissibility).
    "stale_line",
    "missing_snapshot",  # no market data for THAT target on that game (D31-19)
    "missing_prediction",  # no model output for that game -- a pipeline gap (D31-19)
    "ev_not_finite",  # a non-finite per-bet EV: suppressed, never tiered (SPEC R7, D31-24)
    "no_bet_side",  # the model agrees with the market inside the LOCKED no-bet band (D31-05)
    # the target has NO honest EV floor: no grid floor admitted a bet on the honest pre-lock
    # pool, so the whole target places no bets and each of its games is recorded, not dropped
    # (Plan 33.2-29). Emitted by ``backtest.weekly_bet_list.select_weekly_bets``, not here.
    "no_honest_ev_floor",
    # a 2026 WIN bet cleared its price test but its edge over the SPREAD-DERIVED market
    # probability did not clear the target's corrected edge threshold (D33.2-11, Plan 33.2-26).
    "edge_below_threshold",
    # the target has NO honest edge threshold: too little honest pre-lock data to derive one,
    # so it places no bets and each of its games is recorded (Plan 33.2-26, SPEC R14). Emitted
    # by ``backtest.weekly_bet_list.select_weekly_bets``, not here.
    "no_honest_edge_threshold",
    # a 2026 WIN bet whose second test cannot run because NO converter is bound: the deployed
    # blend binds none, so no spread can be turned into a market win probability. A missing
    # converter, not a market-data gap -- ``missing_snapshot`` would send the operator to the
    # odds feed for a fault in the artifacts (A33.2-review IN-06).
    "no_bound_converter",
)

#: The candidate column carrying the SPREAD-DERIVED market probability of a home win, converted
#: with the live blend's bound serving slope (``models.market_probability``). Read only for a
#: target registered in ``BetSelector(edge_threshold_by_target=...)``.
SPREAD_MARKET_PROB_FIELD: str = "market_prob_spread"
# ``no_bet_side`` is plan 31-10's addition, and it is a NINTH reason rather than a reuse of one of
# the eight. D31-05 gives the WP and ATS targets NO eligibility gate (and D33.2-24 took O/U's away),
# so ``not_subpop`` would assert a sub-population that does not exist; and nothing is priced for a
# candidate with no side,
# so ``ev_below_floor`` would claim an expected value that was never measured. The taxonomy is
# designed to grow -- it is the ONE exported list the page maps to labels, and the tests that
# enumerate it are what make an undeclared reason fail.

__all__ = [
    "REJECTION_REASONS",
    "BetSelector",
    "SelectionResult",
    "assert_real_odds",
]

# The naming convention that tells a MODEL output apart from a MARKET value inside a strategy's
# ``required_market_fields``. The two absences have different causes and different fixes --
# collapsing them would hide a broken prediction path behind an odds-coverage label (D31-19) --
# and the core cannot use target vocabulary to tell them apart.
#
# A strategy MAY declare ``required_prediction_fields`` to name its model columns explicitly; that
# member is deliberately NOT part of the ``TargetStrategy`` Protocol, because adding a required
# member mid-wave would un-conform the strategies plan 31-10 is writing against the current
# eight-member Protocol. Undeclared, the convention below classifies.
_PREDICTION_FIELD_PREFIX = "model_"


# The canonical absence predicate now lives in ``backtest.selector_strategies`` (WR-06), which
# this module already imports -- the three ``grade()`` implementations there need the same
# predicate, and importing this module from there would close a cycle. Re-bound under the existing
# private name so the six call sites below read unchanged.
_is_absent = is_absent


def _prediction_field_names(strategy: TargetStrategy) -> frozenset[str]:
    """The subset of ``strategy.required_market_fields`` that are MODEL outputs (D31-19)."""
    declared = getattr(strategy, "required_prediction_fields", None)
    if declared is not None:
        return frozenset(declared)
    return frozenset(
        name
        for name in strategy.required_market_fields
        if name.startswith(_PREDICTION_FIELD_PREFIX)
    )


def _strategy_bet_odds(
    strategy: TargetStrategy, row: dict[str, Any], bet_side: str, default: int
) -> int:
    """The American odds for the side actually bet, via the OPTIONAL ``bet_odds`` member (D31-04).

    EVERY production strategy declares this member, and each is priced and sized at the price a
    book actually offered: WP at the side's own moneyline, and the two line targets at the stored
    two-sided juice devigged through the existing chain helpers. A flat -110 payout on a -320
    favourite would turn a losing bet into a +0.43 EV one -- the same class of defect as sizing
    Kelly off a points distance.

    THE TWO LINE TARGETS WERE PRICED FLAT UNTIL DEF-31-13 WAS RULED (2026-09-05). D31-04 named ATS
    and O/U "the two flat-quoted targets" and gave this member to ``WPStrategy`` alone, so both
    fell through to ``default`` and their EV, Kelly stake and flat payout were all computed at
    -110. The frozen ``PROFITABILITY-PREREGISTRATION.md`` says otherwise in two places -- the ATS
    chain devigs ``spread_ju_home`` / ``spread_ju_away`` (section 3.2 step 4) and the O/U chain
    devigs ``total_over_ju`` / ``total_under_ju`` (section 3.3 step 4) -- and the juice is real:
    1,992 of 2,120 distinct ``(game_id, sportsbook)`` pairs carry a price other than -110 (Plan
    31-02). Two ratified documents disagreed, and choosing between them is an owner ruling rather
    than an executor's: the owner ruled that the FROZEN PRE-REGISTRATION governs the 2025 verdict,
    so D31-04's "flat-quoted" characterisation is SUPERSEDED for the selection path.

    ``default`` REMAINS the price of an ABSENT two-sided price -- the flat -110 fallback
    ``backtest.ats_ev_chain`` has documented since D27-13. A strategy signals that by returning
    None rather than by asserting ``STANDARD_VIG_ODDS`` itself, so a selector configured with a
    different reference juice is honoured instead of silently ignored. The fallback is never
    invisible: the two line strategies stamp ``devig_method`` onto the decision record, so a real
    -110 and an absent-juice -110 are told apart in the record rather than by inference.

    Read through ``getattr`` for the same reason ``required_prediction_fields`` is (plan 31-09):
    adding a required member to ``TargetStrategy`` would un-conform every strategy that has no use
    for it. The member stays OPTIONAL and off the Protocol; the ruling changed WHICH strategies
    declare it, not how the core reaches it.
    """
    resolver = getattr(strategy, "bet_odds", None)
    if resolver is None:
        return default
    resolved = resolver(row, bet_side)
    if resolved is None:
        return default
    return int(resolved)


def _side_market_probability(market_home: float, bet_side: str) -> float:
    """The spread-derived market probability of the SIDE bet.

    Delegates to the WP chain's one side-correct helper -- the same function that turns the
    model's P(home) into P(side) -- so the two sides of the edge are made side-correct by one
    definition. Imported lazily for the cycle ``backtest.selector_strategies._wp_chain`` names.
    """
    from backtest.wp_ev_chain import calibrated_p_home_side

    return float(calibrated_p_home_side(market_home, bet_side))


def _freshness_context(
    row: dict[str, Any],
) -> tuple[datetime | None, datetime | None, bool | None]:
    """The candidate's snapshot instant, the lock it is judged against, and the verdict.

    Both instants are returned as tz-aware datetimes expressed in EASTERN, because the market's
    own zone is the one the lock is defined in and re-expressing it for display is where a
    UTC-anchored reading gets reintroduced (the WR-02 lesson). The second element is still
    carried on the record as ``freeze_ts`` -- a published column name kept by ruling (HOST-07) --
    but its VALUE is the game's day-before-kickoff lock (D33.2-01).

    The lock is PER-GAME (D31-18), taken from that game's own Eastern gameday: a Thursday game
    locks on the Wednesday and that week's Sunday games on the Saturday.

    The lock is ``utils.game_lock.game_lock`` and the verdict is ``utils.game_lock.is_admissible``
    -- the ONE comparison every admissibility reader in the tree now calls directly, since Plan
    33.2-20 deleted the uncalled ``scripts.ingest_historical_odds.is_admissible_at_lock`` wrapper
    that used to delegate to it -- each called exactly ONCE here and never restated. Both are reached as module attributes at
    call time, so the phase's identity scan sees them. It is an ADMISSIBILITY verdict, not a
    staleness one (Plan 33.2-02): a snapshot captured AT or BEFORE the lock is admissible, one
    captured after it is not. Both sides are parsed strictly -- the stored column holds
    offset-carrying strings in two spellings, so a string comparison would be wrong, and a naive
    value raises rather than being anchored.

    A row with no ``gameday`` has no per-game lock and the verdict is None (not evaluated): that
    is a historical backtest frame, which makes no forward claim. The universe path refuses a
    schedule without kickoff dates, so the forward path cannot reach this branch and quietly
    skip the fence.

    The helpers are imported HERE rather than at module scope to break a REAL import cycle:
    ``scripts.ingest_historical_odds`` imports ``backtest.ev_chain_constants``, which imports
    ``backtest.ou_monetization``, which imports THIS module. A module-scope import fails at
    collection with a partially-initialized-module ImportError. The deferred import is the cycle
    break and not an attempt to soften the dependency, which the module docstring states plainly.

    Returns:
        ``(snapshot_instant, lock_instant, is_admissible)``; any of the three may be None.
    """
    import utils.game_lock as lock_rule
    from scripts.ingest_historical_odds import (
        EASTERN,
        eastern_gameday_start,
        normalize_snapshot_ts,
    )

    gameday = row.get("gameday")
    snapshot_value = row.get("snapshot_ts")

    snapshot = (
        None if _is_absent(snapshot_value) else normalize_snapshot_ts(snapshot_value)
    )
    if _is_absent(gameday):
        return snapshot, None, None

    # THE ONE CALL SITE resolving a lock in this module (D31-18, D33.2-01), and the one
    # comparison. The start of the game's Eastern day is handed to the rule because the rule
    # reads only the Eastern date.
    lock = lock_rule.game_lock(eastern_gameday_start(gameday)).astimezone(EASTERN)
    if snapshot is None:
        return None, lock, None
    return snapshot, lock, lock_rule.is_admissible(snapshot_value, lock)


#: The bookmaker keys a REAL live Odds API response returns, MEASURED rather than recalled: every
#: key on the two genuine live pulls held in bronze (``odds_raw_bronze_2025_W01.parquet`` and
#: ``_W05.parquet``, read 2026-09-24; their per-market ``last_update`` stamps and "DraftKings"-style
#: titles distinguish them from the mock generator's output). Plan 33.2-27 Task 1.
LIVE_CAPTURE_SPORTSBOOKS: frozenset[str] = frozenset(
    {
        "betmgm",
        "betonlineag",
        "betrivers",
        "betus",
        "bovada",
        "draftkings",
        "fanduel",
        "lowvig",
        "mybookieag",
    }
)

#: The provenance guard's allowlist. WIDENED DELIBERATELY (Plan 33.2-27 Task 1): it used to be
#: Phase 26's ``{consensus, draftkings}``, which rejected every other real book a live capture
#: returns -- so the first genuine live row would have raised. Its PURPOSE is unchanged: a row
#: with no sportsbook, or a name no real source uses (a mock or placeholder), is still refused.
#: ``consensus`` is the historical closing-line label, ``consensus_median`` the owned line
#: history's own label. ``backtest.ou_divergence._ALLOWED_SPORTSBOOKS`` is Phase 26's LOCKED
#: harness constant and is deliberately NOT widened; this module no longer reads it.
_ALLOWED_SPORTSBOOKS: frozenset[str] = LIVE_CAPTURE_SPORTSBOOKS | {
    "consensus",
    "consensus_median",
}


def assert_real_odds(raw_odds_df: pd.DataFrame) -> None:
    """Hard-fail on mock/synthetic odds BEFORE any selection (OUM-06, T-27-07).

    A row is offending if its ``sportsbook`` is absent or outside :data:`_ALLOWED_SPORTSBOOKS`.
    Raises a ValueError naming the offending game_ids; a clean frame passes silently.

    ``is_live`` IS NO LONGER A CONTAMINATION SIGNAL (Plan 33.2-27 Task 1). Phase 26 treated it as
    one because no stored row had ever carried it; a genuine live capture legitimately can, and
    whether a row's information is admissible at its game's lock is the lock fence's question
    (``utils.game_lock``), not a provenance one.

    Args:
        raw_odds_df: A frame carrying at least ``game_id`` and ``sportsbook`` (and optionally
            ``is_live``). When ``sportsbook`` is absent the frame is treated as not carrying
            provenance (no check possible) and passes -- callers wanting the guard must supply it.

    Raises:
        ValueError: naming the offending game_ids when a mock/synthetic-odds contamination signal
            is present.
    """
    if (
        raw_odds_df is None
        or raw_odds_df.empty
        or "sportsbook" not in raw_odds_df.columns
    ):
        return

    offending_mask = ~raw_odds_df["sportsbook"].isin(_ALLOWED_SPORTSBOOKS)
    if not offending_mask.any():
        return

    # Name offenders by game_id when present; fall back to row indices on a frame missing the
    # game_id column so a malformed provenance frame still raises the named ValueError (never a
    # bare KeyError, WR-05).
    if "game_id" in raw_odds_df.columns:
        offenders = raw_odds_df.loc[offending_mask, "game_id"].head(10).tolist()
        offender_label = f"offending game_ids={offenders}"
    else:
        offenders = raw_odds_df.index[offending_mask].tolist()[:10]
        offender_label = f"offending row indices (no game_id column)={offenders}"
    bad_books = sorted(
        set(raw_odds_df.loc[offending_mask, "sportsbook"].dropna().unique())
    )
    absent_books = int(raw_odds_df.loc[offending_mask, "sportsbook"].isna().sum())
    msg = (
        "Odds provenance check FAILED (mock/synthetic-odds contamination, OUM-06): "
        f"unexpected sportsbooks={bad_books}, rows with no sportsbook={absent_books}; "
        f"{offender_label}"
    )
    raise ValueError(msg)


@dataclass
class SelectionResult:
    """The BetSelector output (BET-01).

    Attributes:
        selected: Per-bet records the selector BET (eligible AND EV >= floor).
        rejected: Per-candidate records NOT bet, each carrying a ``rejection_reason`` in
            ``REJECTION_REASONS`` (for example no_bet_side / ev_below_floor).
        filtered: The eligible acceptance-basis records (every sided candidate; both selected and
            EV-floor-rejected eligible candidates carry their decision metadata here).
        unfiltered: The whole-population cross-check (every candidate, eligible or not) -- a
            REPORTED cross-check only (D27-04), never the acceptance basis.
        clv_report: A report-only CLV summary over the SELECTED bets (mean / t / p / 95% CI via
            ``clv_significance``). NEVER used to gate (D27-06). None when no bets were selected.
    """

    selected: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    filtered: list[dict[str, Any]] = field(default_factory=list)
    unfiltered: list[dict[str, Any]] = field(default_factory=list)
    clv_report: dict[str, Any] | None = None


class BetSelector:
    """Single-source bet-decision engine (BET-01/02, OUM-04/06, LOCKED-1/2, D31-01/02).

    Construct with the frozen residual SD and the per-season prior-season-walk-forward bias (both
    fit on tune-only data by the caller -- the no-leak fence is the caller's responsibility, per the
    Plan-01 EV-chain contract), the EV-floor scalar ``ev_floor_t`` (Plan 04 tunes it) and the
    bankroll. ``select()`` is the ONE source of bet decisions (LOCKED-2). There is no high-total
    boundary argument: it fed only the O/U eligibility gate D33.2-24 deleted, and a parameter with
    no reader is a second answer waiting to be revived.

    ``strategies`` is the D31-01 seam. It defaults to the O/U strategy ALONE, built from the
    frozen-SD / bias / slippage arguments above, so every pre-D31-01 call site is
    unaffected. Supplying it registers additional targets; the weekly exposure cap is then POOLED
    over the union of a week's bets across them (D31-02), which is the concrete reason this core
    exists rather than three sibling selectors.
    """

    def __init__(
        self,
        frozen_sd: float,
        season_bias_by_season: dict[int, float],
        ev_floor_t: float | Mapping[str, float] = 0.0,
        bankroll: float = 10_000.0,
        slippage_points: float = SLIPPAGE_POINTS,
        odds: int = STANDARD_VIG_ODDS,
        strategies: list[TargetStrategy] | None = None,
        edge_threshold_by_target: Mapping[str, float] | None = None,
    ) -> None:
        self.frozen_sd = float(frozen_sd)
        # THE SECOND TEST OF A 2026 WIN BET (D33.2-11, Plan 33.2-26). A target named here is bet
        # only when BOTH its EV at the price the market quoted clears the EV floor AND its edge
        # over the SPREAD-DERIVED market probability, on the side bet, is strictly above this
        # threshold. Both are required, and neither makes the other redundant: the spread
        # conversion can manufacture an apparent edge at every spread, which is why the
        # threshold test alone is not enough; and the threshold was derived against a
        # spread-derived probability and never calibrated on moneyline prices, which is why the
        # price test alone is not enough. Empty -- every pre-33.2-26 caller -- changes nothing.
        self.edge_threshold_by_target: dict[str, float] = {
            str(target): float(value)
            for target, value in (edge_threshold_by_target or {}).items()
        }
        self.season_bias_by_season = dict(season_bias_by_season)
        # The EV floor is a SCALAR or a PER-TARGET mapping (plan 31-12). The pre-registration
        # selects ``t`` per target by a tune-side sweep and carries ONE scalar per target to the
        # hold, so a single shared floor could not express the frozen rule for a pooled
        # three-target week. A scalar still behaves exactly as it always did, which is what
        # every pre-31-12 call site passes.
        if isinstance(ev_floor_t, Mapping):
            self.ev_floor_by_target: dict[str, float] | None = {
                str(target): float(floor) for target, floor in ev_floor_t.items()
            }
            self.ev_floor_t: float | None = None
        else:
            self.ev_floor_by_target = None
            self.ev_floor_t = float(ev_floor_t)
        self.bankroll = float(bankroll)
        self.slippage_points = float(slippage_points)
        self.odds = int(odds)

        # Wrap -- never re-implement -- the LOCKED side/slippage/outcome convention (D-18). The
        # simulator is used ONLY for its grading helpers; the strategies own the target rules and
        # the BetSelector owns the decision. One simulator per selector, injected downwards.
        self._sim = BettingSimulator(SimulationConfig())

        if strategies is None:
            strategies = [
                OUStrategy(
                    frozen_sd=self.frozen_sd,
                    season_bias_by_season=self.season_bias_by_season,
                    slippage_points=self.slippage_points,
                    simulator=self._sim,
                )
            ]
        if not strategies:
            msg = (
                "BetSelector requires at least one target strategy; an empty registry can only "
                "produce an empty bet list, which is indistinguishable from a target that "
                "genuinely had no +EV bets (T-31-27)."
            )
            raise ValueError(msg)

        registry: dict[str, TargetStrategy] = {}
        for strategy in strategies:
            code = strategy.target
            if code in registry:
                msg = (
                    f"duplicate strategy registered for target {code!r}; the registry is keyed by "
                    "target code and each target has exactly one strategy (D31-01)."
                )
                raise ValueError(msg)
            registry[code] = strategy
        self._strategies = registry

        # Kelly sizing (BET-02): the calibrated P(side) is fed to the EXISTING calculator. unit = 1%
        # of bankroll (D27-09); quarter-Kelly via default_kelly_fraction=0.25; 5% per-bet cap.
        self._kelly = KellyCalculator(
            starting_bankroll=self.bankroll,
            max_bet_pct=0.05,
            base_unit_size=self.bankroll * 0.01,
            default_kelly_fraction=0.25,
            # The EV chain owns admission; do not double-gate sizing on a points-edge threshold.
            confidence_threshold=0.0,
        )

    @property
    def strategies(self) -> dict[str, TargetStrategy]:
        """The registered strategies, keyed by target code (a copy -- the registry is fixed)."""
        return dict(self._strategies)

    def ev_floor_for(self, target: str | None) -> float:
        """The EV floor ``t`` this selector admits ``target`` at (plan 31-12).

        A scalar floor answers for every target, which is what every pre-31-12 caller
        configured. A per-target mapping answers only for the targets it names and RAISES for
        one it does not: a silent fallback to 0.0 would admit a target at a floor nobody swept
        for, which is precisely the un-pre-registered threshold the whole apparatus exists to
        prevent.

        Raises:
            UnregisteredTargetError: when a per-target mapping has no floor for ``target``.
        """
        if self.ev_floor_by_target is None:
            return float(self.ev_floor_t or 0.0)
        if target is None and len(self.ev_floor_by_target) == 1:
            return next(iter(self.ev_floor_by_target.values()))
        if target not in self.ev_floor_by_target:
            msg = (
                f"no EV floor is registered for target {target!r}; the per-target floors are "
                f"{sorted(self.ev_floor_by_target)}. There is NO silent fallback to 0.0: the "
                "floor is a PRE-REGISTERED per-target scalar chosen by a tune-side sweep, and "
                "admitting a target at an unswept floor is an un-pre-registered threshold."
            )
            raise UnregisteredTargetError(msg)
        return self.ev_floor_by_target[target]

    # -- strategy dispatch ----------------------------------------------------

    def _strategy_for(self, target: str | None) -> TargetStrategy:
        """Resolve the strategy for ``target``, failing LOUDLY when there is none (T-31-27).

        A candidate row (or a decision record) may omit ``target`` entirely -- every pre-D31-01
        caller does. That is unambiguous ONLY while a single strategy is registered, and it is
        resolved that way. With more than one registered there is no defensible default, so it
        raises rather than guessing: a wrong guess would book a bet under the wrong target's rules.

        Raises:
            UnregisteredTargetError: when the target is unknown, or is absent and ambiguous.
        """
        if target is None:
            if len(self._strategies) == 1:
                return next(iter(self._strategies.values()))
            msg = (
                "candidate carries no 'target' but this selector has multiple registered "
                f"targets {sorted(self._strategies)}; the target is ambiguous -- tag each "
                "candidate row with its target code."
            )
            raise UnregisteredTargetError(msg)
        if target not in self._strategies:
            msg = (
                f"no strategy is registered for target {target!r}; the registered targets are "
                f"{sorted(self._strategies)}. A missing strategy is a failure, not an empty "
                "bet list (T-31-27)."
            )
            raise UnregisteredTargetError(msg)
        return self._strategies[target]

    # -- main entry point -----------------------------------------------------

    def select(
        self,
        candidates: pd.DataFrame | list[dict[str, Any]],
        raw_odds_df: pd.DataFrame | None = None,
        scheduled_games: pd.DataFrame | list[dict[str, Any]] | None = None,
    ) -> SelectionResult:
        """The single bet-decision source (BET-01/02, LOCKED-2, D31-17/19).

        Args:
            candidates: Per-game candidate rows (a DataFrame or list of dicts) carrying at least
                ``game_id``, ``season``, ``week``, the registered strategy's
                ``required_market_fields``, and (for grading) ``actual``. A row may carry a
                ``target`` code selecting its strategy; when absent and exactly one strategy is
                registered, that strategy is used. Rows may also carry ``sportsbook`` / ``is_live``
                provenance columns, which are validated when present, and ``snapshot_ts`` /
                ``gameday``, which drive the freshness fence.
            raw_odds_df: Optional raw odds provenance frame. When supplied, it is validated FIRST
                via ``assert_real_odds`` (OUM-06) before any selection. When None, the candidates
                frame itself is checked for provenance columns.
            scheduled_games: Optional schedule (``game_id``, ``season``, ``week``, ``gameday``).
                When supplied, the candidate universe becomes every scheduled game times every
                REGISTERED target (D31-19) and every pair yields a record, live or suppressed, so
                a candidate is never silently dropped. When None the rows ARE the universe, which
                is exactly what every pre-31-09 caller expects.

        Returns:
            A :class:`SelectionResult` with selected / rejected / filtered / unfiltered records and
            a report-only ``clv_report``.
        """
        rows = self._to_records(candidates)
        if scheduled_games is not None:
            rows = self._build_universe(scheduled_games, rows)

        # (1) Provenance hard-fail BEFORE any decision (OUM-06, T-27-07). Validate the explicit
        # raw odds frame if supplied; otherwise validate the candidates' own provenance columns.
        #
        # THE FRAME IS NARROWED; THE CHECK IS NOT (T-31-43b). ``assert_real_odds`` raises when a
        # row's sportsbook is outside the allowlist, and a MISSING sportsbook is outside it by
        # construction (``~Series.isin(...)`` is True for NaN). A skeleton row -- one carrying
        # neither a sportsbook nor any market value -- would therefore RAISE before it could be
        # labelled ``missing_snapshot``. ``_provenance_frame`` drops exactly those rows and no
        # others: a row carrying a sportsbook is always checked, so a disallowed book and an
        # ``is_live`` row each still raise. ``assert_real_odds`` itself is untouched -- it is the
        # LOCKED Phase-27 guard and the Phase-27/30 record depends on its behaviour.
        if raw_odds_df is not None:
            assert_real_odds(raw_odds_df)
        else:
            assert_real_odds(self._provenance_frame(rows))

        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []
        unfiltered: list[dict[str, Any]] = []

        # Eligible candidates needing sizing, grouped by week ACROSS TARGETS. The key is
        # (season, week) and deliberately NOT (target, season, week): D31-02 pools the 10% weekly
        # exposure cap over the union of a week's bets, because D27-10 pre-registered it as a
        # per-week TOTAL exposure cap and the bankroll does not grow because targets were added.
        eligible_by_week: dict[tuple[int, int], list[dict[str, Any]]] = {}

        for row in rows:
            strategy = self._strategy_for(row.get("target"))
            record, rejection_reason = self._build_decision_record(row, strategy)
            unfiltered.append(record)

            if rejection_reason is not None:
                rejected.append({**record, "rejection_reason": rejection_reason})
                continue

            # Eligible: it is part of the acceptance basis regardless of the EV decision.
            filtered.append(record)
            key = (int(record["season"]), int(record["week"]))
            eligible_by_week.setdefault(key, []).append(record)

        # (3)/(4) EV admission + LOCKED-order sizing, once per POOLED week.
        for (_season, _week), week_records in eligible_by_week.items():
            self._admit_and_size_week(week_records, selected, rejected)

        clv_report = self._clv_report(selected)

        logger.info(
            "BetSelector.select complete",
            n_candidates=len(rows),
            n_eligible=len(filtered),
            n_selected=len(selected),
            n_rejected=len(rejected),
            ev_floor_t=self.ev_floor_t,
            ev_floor_by_target=self.ev_floor_by_target,
            targets=sorted(self._strategies),
        )

        return SelectionResult(
            selected=selected,
            rejected=rejected,
            filtered=filtered,
            unfiltered=unfiltered,
            clv_report=clv_report,
        )

    # -- the candidate universe (D31-19) --------------------------------------

    def _build_universe(
        self,
        scheduled_games: pd.DataFrame | list[dict[str, Any]],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expand the schedule to every scheduled game times every REGISTERED target.

        A pair with no candidate row becomes a SKELETON -- the game_id, its week and its kickoff
        date, and nothing else -- which the decision path then suppresses as ``missing_snapshot``.
        Building the universe first is what makes "never silently dropped" structural: a game
        absent from the universe could not be reported as suppressed at all, and reconciling an
        odds-joined universe afterwards would be a second computation that can disagree with the
        first.

        Args:
            scheduled_games: The week's schedule, each row carrying ``game_id``, ``season``,
                ``week`` and ``gameday``.
            rows: The normalized candidate rows.

        Returns:
            One row per (scheduled game, registered target), in schedule order then registry order.

        Raises:
            ValueError: when a scheduled game is missing a required column or is listed twice, when
                two candidates claim the same (game_id, target), or when a candidate names a game
                that is not in the schedule.
        """
        if isinstance(scheduled_games, pd.DataFrame):
            games = scheduled_games.to_dict("records")
        else:
            games = [dict(game) for game in scheduled_games]

        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["game_id"]), self._strategy_for(row.get("target")).target)
            if key in by_key:
                msg = (
                    f"two candidate rows claim game_id/target {key}; the universe is keyed by "
                    "that pair and a duplicate would double-count the week."
                )
                raise ValueError(msg)
            by_key[key] = row

        universe: list[dict[str, Any]] = []
        seen: set[str] = set()
        for game in games:
            missing = [
                name
                for name in ("game_id", "season", "week", "gameday")
                if _is_absent(game.get(name))
            ]
            if missing:
                msg = (
                    f"scheduled game {game.get('game_id')!r} is missing {missing}. A schedule row "
                    "must carry its own gameday: the freshness fence is measured against that "
                    "game's OWN preceding-Friday freeze (D31-18), and without a kickoff date the "
                    "fence would silently never fire."
                )
                raise ValueError(msg)

            game_id = str(game["game_id"])
            if game_id in seen:
                msg = (
                    f"scheduled game {game_id!r} is listed twice; a duplicated game would be "
                    "counted twice in the universe and reported twice on the page."
                )
                raise ValueError(msg)
            seen.add(game_id)

            for target in self._strategies:
                # The schedule is authoritative for identity and timing, so it is applied LAST.
                skeleton = {
                    "game_id": game_id,
                    "season": int(game["season"]),
                    "week": int(game["week"]),
                    "gameday": game["gameday"],
                    "target": target,
                }
                candidate = by_key.pop((game_id, target), None)
                universe.append(
                    skeleton if candidate is None else {**candidate, **skeleton}
                )

        if by_key:
            orphans = sorted({game_id for game_id, _target in by_key})[:10]
            msg = (
                f"candidate rows name game_ids that are not in the schedule: {orphans}. A "
                "candidate outside the universe could not be reported as live or as suppressed, "
                "so it is refused rather than dropped (D31-19)."
            )
            raise ValueError(msg)

        return universe

    def _provenance_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """The subset of ``rows`` that carries provenance for ``assert_real_odds`` to judge.

        A row carrying NEITHER a sportsbook NOR any of its target's MARKET values is a skeleton:
        it makes no provenance claim, so there is nothing to check and it is labelled rather than
        raised on. Every other row -- in particular every row carrying a sportsbook -- is handed
        to the guard unchanged. ``assert_real_odds`` already treats a frame with no ``sportsbook``
        column as carrying no provenance; this applies the SAME rule per row.

        A MODEL value is not a market value (33.2 review, batch 1a follow-up). The weekly
        candidates are a left join of the scored games onto the odds store, so a game the model
        scored and no book priced carries its prediction beside a NaN sportsbook. Counting the
        prediction as a provenance claim sent that one game into the guard, and the guard's
        "no sportsbook" refusal sank the WHOLE bet list; the game is a market gap, labelled
        ``missing_snapshot`` downstream, and gets no bet.
        """

        def _market_fields(row: dict[str, Any]) -> list[str]:
            strategy = self._strategy_for(row.get("target"))
            model_fields = _prediction_field_names(strategy)
            return [
                name
                for name in strategy.required_market_fields
                if name not in model_fields
            ]

        checked = [
            row
            for row in rows
            if not _is_absent(row.get("sportsbook"))
            or any(not _is_absent(row.get(name)) for name in _market_fields(row))
        ]
        return pd.DataFrame(checked)

    # -- decision-record construction -----------------------------------------

    def _build_decision_record(
        self, row: dict[str, Any], strategy: TargetStrategy
    ) -> tuple[dict[str, Any], str | None]:
        """Build the per-candidate decision record, dispatching every target rule to ``strategy``.

        The core owns the record SHAPE and the generic fields; the strategy owns the side, the
        eligibility rule, the sub-pop label, the calibrated P(side) and the target-specific
        reporting extras. No target vocabulary appears here.

        A SUPPRESSED candidate short-circuits before any strategy method that reads market or
        model data. It is not priced at all: a stale line and an absent column are both refusals to
        form an opinion, and publishing an EV computed from either would be the thing suppression
        exists to prevent. Such a record therefore carries the CORE schema only -- no target
        ``decision_extras`` -- because those extras are derived from data the row does not have.

        Returns:
            ``(record, rejection_reason)``. The reason is None when the candidate is eligible. It
            is returned ALONGSIDE the record rather than stored on it, so an eligible record never
            carries a nullable reason field it can never use.
        """
        game_id = row["game_id"]
        season = int(row["season"])
        week = int(row["week"])

        market, missing_market, missing_prediction = self._read_required_fields(
            row, strategy
        )
        snapshot_instant, freeze_instant, is_fresh = _freshness_context(row)

        record: dict[str, Any] = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "target": strategy.target,
            "bet_side": None,
            # None means NOT ASKED -- distinct from the label a strategy reports once asked
            # (``NO_SUBPOPULATION_LABEL`` for every production target since D33.2-24).
            "subpop_label": None,
            **market,
            "eligible": False,
            "calibrated_p_side": None,
            "per_bet_ev": None,
            "slipped_line": None,
            # The American odds the bet was priced and sized at, resolved per target and per side
            # (D31-04). None on a candidate that was never priced -- a suppressed row got no price,
            # and stamping the reference juice onto it would claim one it never had.
            "selected_odds": None,
            "kelly_stake": 0.0,
            "outcome": None,
            # CLV defaults to None -- NOT REPORTED -- and a strategy that has a closing-line value
            # overrides it in ``decision_extras``. Defaulting to 0.0 instead would let a target
            # that measures no CLV drag a published CLV mean toward zero and read as "no edge
            # measured" rather than "not measured"; ``_clv_report`` drops the Nones.
            "clv": None,
            # Both instants travel ON the record so the page and the export render them without
            # recomputing either (D31-17).
            "snapshot_ts": snapshot_instant,
            "freeze_ts": freeze_instant,
            # Carry the private realized-total stash forward so ``_grade`` can resolve the push-aware
            # outcome (the LOCKED ``_resolve_ou_outcome``). ``_to_records`` stashes the candidate's
            # ``actual`` under ``_actual_total``; without carrying it onto the decision record every
            # graded outcome would be None (no win/loss ever resolved) -- the bug that zeroed every
            # graded ROI (Rule 1, Plan 27-04). It is NOT in the public schema; downstream stores a
            # missing/ungraded outcome as SQL NULL.
            "_actual_total": row.get("_actual_total"),
        }

        # Market gaps and model gaps are DIFFERENT causes with different fixes (D31-19), so the
        # market check runs first and a present-market/absent-model row is reported as the model
        # gap it is rather than as an odds-coverage problem.
        two_test = strategy.target in self.edge_threshold_by_target
        if two_test:
            record[SPREAD_MARKET_PROB_FIELD] = None
            record["spread_market_edge"] = None
        if missing_market:
            return record, "missing_snapshot"
        if missing_prediction:
            return record, "missing_prediction"
        # The spread-derived probability is MARKET data the threshold test needs; without it the
        # second test cannot run, so the row is suppressed rather than priced on one test alone.
        # A column ABSENT from the candidate means no converter was bound (the weekly layer
        # attaches it only when one is); a PRESENT-but-empty value means this game had no spread.
        if two_test and SPREAD_MARKET_PROB_FIELD not in row:
            return record, "no_bound_converter"
        if two_test and _is_absent(row.get(SPREAD_MARKET_PROB_FIELD)):
            return record, "missing_snapshot"

        # A row that HAS market data and a kickoff date but no timestamp is a pipeline bug, and it
        # is the one shape that would make this fence unable to fire. Assuming it fresh would
        # silently admit whatever the cache builder dropped the column on; calling it stale would
        # hide the bug behind a data label. So it raises, naming the column.
        if freeze_instant is not None and snapshot_instant is None:
            msg = (
                f"candidate {game_id!r} for target {strategy.target!r} carries market data and a "
                "gameday but no 'snapshot_ts'; the freshness fence cannot be evaluated and a "
                "missing freeze instant is never treated as fresh (D31-17/18)."
            )
            raise ValueError(msg)
        if is_fresh is False:
            # Suppressed BEFORE pricing: a line captured after the game's lock is information the
            # decision could not have had, and an EV computed from it would be exactly the number
            # suppression exists to withhold.
            return record, "stale_line"

        bet_side = strategy.resolve_bet_side(row)
        rejection_reason = strategy.eligibility(row, bet_side)
        eligible = rejection_reason is None
        record.update(
            {
                "bet_side": bet_side,
                "subpop_label": strategy.eligibility_label(row, bet_side),
                "eligible": eligible,
                # The strategy's reporting extras (for O/U: the REPORT-ONLY model-edge CLV, which
                # never gates -- D27-06, and the devig method).
                **strategy.decision_extras(row, bet_side),
            }
        )

        if eligible:
            if bet_side is None:
                # A core invariant, not a target rule: there is no bet without a side, so a
                # strategy that admits a sideless candidate is a bug and must say so rather
                # than price one half of a coin flip.
                msg = (
                    f"strategy {strategy.target!r} declared candidate "
                    f"{game_id!r} eligible with no bet side; an eligible candidate must have "
                    "a side."
                )
                raise ValueError(msg)
            p_side, slipped_line = strategy.side_probability(row, bet_side)
            record["calibrated_p_side"] = p_side
            record["slipped_line"] = slipped_line
            # The price the bet is judged at, resolved per target and per side (D31-04, DEF-31-13).
            # It is the price the market actually quoted for THIS side -- a moneyline for WP, the
            # devigged stored juice for the two line targets -- falling back to ``self.odds`` only
            # when the row carries no two-sided price. It travels ON the record so the EV, the Kelly
            # stake and the published price are all the SAME number rather than three that agree by
            # convention.
            selected_odds = _strategy_bet_odds(strategy, row, bet_side, self.odds)
            record["selected_odds"] = selected_odds
            record["per_bet_ev"] = per_bet_ev(p_side, american_to_payout(selected_odds))
            if two_test:
                market_home = float(row[SPREAD_MARKET_PROB_FIELD])
                record[SPREAD_MARKET_PROB_FIELD] = market_home
                record["spread_market_edge"] = p_side - _side_market_probability(
                    market_home, bet_side
                )

        return record, rejection_reason

    def _clears_edge_threshold(self, record: dict[str, Any]) -> bool:
        """The second test of a win bet: the side edge is STRICTLY above the threshold.

        STRICT, the same comparison the edge band uses, so an edge exactly at the threshold is
        in the band below it and is not a bet. A target with no threshold registered passes.
        """
        threshold = self.edge_threshold_by_target.get(str(record.get("target")))
        if threshold is None:
            return True
        edge = record.get("spread_market_edge")
        return edge is not None and float(edge) > threshold

    @staticmethod
    def _read_required_fields(
        row: dict[str, Any], strategy: TargetStrategy
    ) -> tuple[dict[str, float | None], tuple[str, ...], tuple[str, ...]]:
        """Read the strategy's required fields off ``row``, naming which are absent and why.

        Every required name lands on the record -- as a float when present, as None when not -- so
        the record schema does not change shape with coverage. The two absence lists are what the
        caller turns into ``missing_snapshot`` or ``missing_prediction``.

        Before D31-19 an absent field raised a named ``KeyError`` here. It is now a FIRST-CLASS
        suppression instead: an absent column is exactly the partial-coverage case R6 requires be
        reported per target, and raising would have made the universe unbuildable.

        Returns:
            ``(values, missing_market_names, missing_prediction_names)``.
        """
        prediction_names = _prediction_field_names(strategy)
        values: dict[str, float | None] = {}
        missing_market: list[str] = []
        missing_prediction: list[str] = []
        for name in strategy.required_market_fields:
            value = row.get(name)
            # ``value is None`` is stated first so the type checker narrows the else branch;
            # ``_is_absent`` additionally catches the NaN a DataFrame puts in an empty cell.
            if value is None or _is_absent(value):
                values[name] = None
                if name in prediction_names:
                    missing_prediction.append(name)
                else:
                    missing_market.append(name)
            else:
                values[name] = float(value)
        return values, tuple(missing_market), tuple(missing_prediction)

    # -- EV admission + sizing (per POOLED week) ------------------------------

    def _admit_and_size_week(
        self,
        week_records: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> None:
        """Admit eligible records by the EV floor, then size the admitted set via the LOCKED order.

        ``week_records`` is the UNION of one week's eligible candidates across every registered
        target (D31-02). The 10% weekly exposure cap is applied ONCE over that union, pro-rata, so
        every bet in the week carries the same weekly factor and the week's TOTAL exposure -- not
        each target's -- is what the cap bounds. A per-target cap would be a 30% total weekly
        ceiling, a post-hoc tripling of a pre-registered ruin guard; per-target sub-caps would be a
        second threshold nobody pre-registered.

        EV admission: bet iff ``per_bet_ev(p_side) >= ev_floor_t`` (D27-14) -- the comparison below
        rejects STRICTLY BELOW the floor, so EV exactly equal to the floor is ADMITTED (SPEC R1
        adjacency). Sizing: Kelly on the CALIBRATED P(side) (BET-02 fix) -> the Plan-02
        ``apply_sizing_pipeline`` LOCKED cap order. Grading dispatches to the record's strategy.

        ADMISSION STRUCTURALLY PRECEDES SIZING (D31-03): the EV-floor loop below compares
        ``per_bet_ev`` against the floor and reads no stake at all, so no change to the
        de-weighting rule -- including the pooling above -- can change WHICH bets are selected.
        The later zero-stake guard reads the RAW Kelly stake, computed before and independently
        of de-weighting.
        """
        admitted: list[dict[str, Any]] = []
        for record in week_records:
            per_bet = record["per_bet_ev"]
            ev_floor_t = self.ev_floor_for(record.get("target"))
            # The non-finite refusal precedes the floor comparison, and must (SPEC R7, T-31-43):
            # ``NaN < floor`` is False, so a NaN EV would otherwise fall THROUGH the floor and be
            # booked, then reach ``assign_ev_tier`` -- which raises -- with a bet already made.
            if per_bet is not None and not math.isfinite(per_bet):
                logger.warning(
                    "BetSelector suppressed a candidate whose per-bet EV is not finite; it is "
                    "never tiered and never bet (SPEC R7).",
                    game_id=record.get("game_id"),
                    target=record.get("target"),
                    calibrated_p_side=record.get("calibrated_p_side"),
                    per_bet_ev=per_bet,
                )
                rejected.append({**record, "rejection_reason": "ev_not_finite"})
                continue
            if per_bet is None or per_bet < ev_floor_t:
                rejected.append({**record, "rejection_reason": "ev_below_floor"})
                continue
            # The second test (D33.2-11): a price that clears the floor is not enough on its own.
            if not self._clears_edge_threshold(record):
                rejected.append({**record, "rejection_reason": "edge_below_threshold"})
                continue
            admitted.append(record)

        if not admitted:
            return

        # Kelly stake on the CALIBRATED P(side) (BET-02 fix -- NEVER implied + points_edge).
        # An admitted bet that the inner Kelly calculator zeroes is the EV/Kelly double-gate
        # boundary (WR-07): the EV floor admitted it, but Kelly stakes nothing because its
        # calibrated P(side) is at/below the -110 breakeven. At ev_floor_t >= 0 this is
        # zero-measure (admission already requires p_side >= breakeven); for a NEGATIVE floor the
        # two gates diverge. Rather than silently book a zero-stake "selected" bet, surface it:
        # reject with reason "zero_kelly_stake" and warn, so `selected` holds only genuinely
        # stakeable bets and the latent double-gate is observable.
        kelly_inputs: list[dict[str, Any]] = []
        staked_admitted: list[dict[str, Any]] = []
        for record in admitted:
            kelly_result = self._kelly.calculate_optimal_bet_size(
                # The RECORD's price, not the selector's reference juice: an eligible record was
                # priced at the odds its own market quoted, and sizing at a different number would
                # be sizing a bet nobody could place (D31-04).
                model_prob=record["calibrated_p_side"],
                market_odds=int(record["selected_odds"]),
                mode=KellyMode.FRACTIONAL,
            )
            if kelly_result.recommended_bet <= 0.0:
                logger.warning(
                    "BetSelector admitted a bet the Kelly calculator zeroed (EV/Kelly "
                    "double-gate boundary); rejecting instead of booking a zero stake.",
                    game_id=record.get("game_id"),
                    target=record.get("target"),
                    calibrated_p_side=record.get("calibrated_p_side"),
                    per_bet_ev=record.get("per_bet_ev"),
                    ev_floor_t=self.ev_floor_for(record.get("target")),
                )
                rejected.append({**record, "rejection_reason": "zero_kelly_stake"})
                continue
            staked_admitted.append(record)
            kelly_inputs.append(
                {
                    # game_id feeds the D31-03 same-game grouping inside the de-weight step. It
                    # is REQUIRED by the sizing seam (no silent fallback to same-side-only
                    # grouping, T-31-17). It became load-bearing here once D31-02 pooled the week
                    # across targets: a WP "home" bet and an ATS "home_cover" bet on the SAME game
                    # are close to one leveraged wager, and only the game key catches that.
                    "game_id": record["game_id"],
                    "bet_side": record["bet_side"],
                    "stake": kelly_result.recommended_bet,
                }
            )

        if not kelly_inputs:
            return

        # LOCKED-order sizing pipeline (Plan-02, extended by D31-03): kelly -> 5% per-bet ->
        # same-game-and-side de-weight -> 10% weekly cap. The BetSelector consumes this helper
        # and does NOT re-order the steps. Called EXACTLY ONCE per week over the pooled union.
        sized = apply_sizing_pipeline(kelly_inputs, self.bankroll)

        # ``apply_sizing_pipeline`` collapses the weekly pro-rata factor into a COMBINED
        # de-weight x weekly ``scale_factor`` that differs per bet, so the week's single pro-rata
        # factor is not directly readable from it. Recover it by re-reading the SAME pure LOCKED
        # helper on the SAME inputs -- a read, not a second application of the cap. Deriving it as
        # a ratio instead would be inexact in the last bit and would break the "one factor for the
        # whole week" guarantee this publishes.
        weekly = apply_weekly_exposure_cap(
            [sized_rec["deweighted_stake"] for sized_rec in sized], self.bankroll
        )

        for record, sized_rec, weekly_rec in zip(
            staked_admitted, sized, weekly, strict=True
        ):
            if weekly_rec["weekly_scaled_stake"] != sized_rec["weekly_scaled_stake"]:
                msg = (
                    "weekly-cap re-read disagreed with the sizing pipeline "
                    f"({weekly_rec['weekly_scaled_stake']!r} != "
                    f"{sized_rec['weekly_scaled_stake']!r}); the published weekly factor would "
                    "not be the factor that produced the stake."
                )
                raise ValueError(msg)
            record["kelly_stake"] = sized_rec["weekly_scaled_stake"]
            # The caps made VISIBLE on the record rather than inferable from the ordering, so
            # /bets can show a stake beside its EV and the reader can see why it is that size.
            record["same_side_group_size"] = sized_rec["same_side_group_size"]
            record["same_game_group_size"] = sized_rec["same_game_group_size"]
            record["binding_group"] = sized_rec["binding_group"]
            record["weekly_scale_factor"] = weekly_rec["scale_factor"]
            record["outcome"] = self._grade(record)
            selected.append(record)

    def _grade(self, record: dict[str, Any]) -> bool | None:
        """Grade a selected bet via its target's LOCKED outcome resolver (push-aware, T-27-23).

        Dispatches to the record's strategy; for O/U that is the LOCKED ``_resolve_ou_outcome``,
        unchanged. Returns True (win), False (loss), or None (push or ungraded).
        """
        return self._strategy_for(record.get("target")).grade(record)

    # -- report-only CLV ------------------------------------------------------

    @staticmethod
    def _clv_report(selected: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Report-only CLV summary over the selected bets (D27-06/12, NEVER a gate).

        Computes the mean / t / p / 95% CI of the per-bet model-edge line_clv via the LOCKED
        ``clv_significance``. This is the model-edge line_clv, DISTINCT from the freeze-vs-close
        forward metric (~0 in backtest because freeze == close); CLV-vs-close is a live-only
        forward metric (D27-12). Returns None when no bets were selected.
        """
        if not selected:
            return None
        clv_values = [r["clv"] for r in selected if r["clv"] is not None]
        if not clv_values:
            return None
        report = dict(clv_significance(clv_values))
        report["metric"] = (
            "model_edge_line_clv (model_total - closing_total); REPORT-ONLY (D27-06)"
        )
        return report

    # -- input normalization --------------------------------------------------

    @staticmethod
    def _to_records(
        candidates: pd.DataFrame | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize candidates to a list of dicts, stashing the grading ``actual`` privately.

        The ``actual`` total (when present) is kept under ``_actual_total`` so the public record
        never confuses the model/closing totals with the realized outcome.
        """
        if isinstance(candidates, pd.DataFrame):
            raw = candidates.to_dict("records")
        else:
            raw = [dict(c) for c in candidates]

        records: list[dict[str, Any]] = []
        for row in raw:
            rec = dict(row)
            if "actual" in rec:
                # NORMALIZED TO None AT THE STASH (WR-06). A DataFrame candidate spells an absent
                # realized value as NaN, and every downstream absence contract in this repository
                # is written against ``None`` -- the three ``grade()`` guards (now defended in
                # their own right) and
                # ``profitability_2025._measure_and_judge``'s push-versus-ungraded split, which
                # would otherwise count an UNPLAYED game as a PUSH on the verdict artifact.
                # Normalizing here means an absent result is absent in ONE spelling everywhere.
                value = rec.get("actual")
                rec["_actual_total"] = None if is_absent(value) else value
            records.append(rec)
        return records
