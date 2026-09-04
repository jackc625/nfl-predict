"""WP EV chain (Phase 31, plan 31-07; SPEC R1, PROD-02) -- thin BY DESIGN, not by omission.

    deployed isotonic P(home)                                        (D31-07 DEFAULT)
      -> reliability + Brier calibration gate, TUNE SPLIT ONLY       (D27-07 params, consumed)
      -> [gate FAILS only] prior-season PROBABILITY-scale correction (D31-07 REGISTERED FALLBACK)
      -> real two-sided moneyline devig                              (D27-13, consumed verbatim)
      -> per-bet EV with the devig-derived EXPLICIT payout           (D27-14, consumed)

D31-07 INVERTS the O/U structure for a model that is ALREADY calibrated. In the O/U chain
prior-season bias subtraction is the DEFAULT and isotonic is the fallback, because the O/U
model is not calibrated. The WP model is: its deployed artifact carries an isotonic
calibrator, and the bet list is priced off THAT model, unchanged. Re-fitting calibration on
the tune split was REJECTED -- the bet list would then be priced off a model nobody runs,
and the 2025 verdict would describe a model nobody runs.

"THE SAME APPARATUS" FOR WP MEANS THE SAME CONTRACT, NOT THE SAME STEPS
-----------------------------------------------------------------------
This is narrowing 3 of the ratified pre-registration, and it is why this module is short.
A calibrated classifier has NO RESIDUAL to take a standard deviation of. Forcing the O/U
shape onto WP would mean inventing a logit-space residual SD: manufacturing a fitted
parameter the model does not need, on the target with the worst measured closing-line value
(pooled CLV -0.0380, t -15.52). That was REJECTED, and this module accordingly contains no
residual SD fit and no log-space transform of any kind --
``tests/unit/test_ev_chain_positive_control.py`` asserts both by AST scan rather than by
substring, so this paragraph's own naming of the rejected alternative cannot trip its guard.

What WP DOES owe, and delivers: a pre-registered chain, a leakage fence, a calibration gate
on tune data only, a registered fallback that cannot fire silently, a real-price devig, a
per-bet EV, and one EV floor.

THE FALLBACK CANNOT FIRE SILENTLY
----------------------------------
:class:`WPGateResult` carries ``fallback_fired`` and ``fallback_trigger``, and both travel
onto every priced record. ``fallback_trigger`` is one of the two Phase-31 additions to the
trial-registry field tuple (``TRIAL_REGISTRY_FIELDS_P31``) precisely so a fired fallback is
a row in the registry and adds one entry to the BH family, rather than a difference a reader
would have to infer from the numbers.

A NOTE ON THE GATE'S BRIER ARM, STATED RATHER THAN LEFT TO BE DISCOVERED
------------------------------------------------------------------------
``calibration_gate`` passes on ``brier_corrected <= brier_raw`` AND
``max_bin_deviation <= RELIABILITY_TOLERANCE``. On WP's DEFAULT path there is no corrected
series -- the deployed probability IS the series -- so the same array is passed twice and the
Brier arm is satisfied by construction. The RELIABILITY arm is what actually binds for WP.
Passing the deployed probability twice is honest about that; fabricating a second series to
give the Brier arm something to compare would be a comparison against a model nobody runs.

This module is a PURE numeric transform: NO file I/O, NO DuckDB, NO request path, NO model
re-fit, NO production swap.

WHY THIS IS A SEPARATE FILE. Plan 31-07 asked for it to be folded into
``backtest/selector_strategies.py`` if it came in under roughly forty lines after Task 2.
It did not: the module carries five public callables plus a result dataclass, and its
executable body is well past that threshold before docstrings. Folding it in would also put
WP chain numerics inside the module whose stated scope is per-target SELECTION strategies,
and would place them beside an ``OUStrategy`` that deliberately consumes its chain from
elsewhere. The exact line count is recorded in the plan SUMMARY.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# The shared Phase-31 chain scaffolding. It is hosted in the ATS module rather than in a
# fourth file because tests/unit/test_p31_constants_isolation.py enumerates the scanned
# Phase-31 modules BY NAME, and a module absent from that registry is never AST-scanned.
from backtest.ats_ev_chain import (
    ChainFit,
    ChainPricing,
    assert_fit_window_p31,
    chain_clv_report,
    chain_order_key,
    season_bias_for,
)
from backtest.ev_chain_constants import WP_CHAIN_POLICY

# The primitives, CONSUMED from the Phase-27 chain. Note what is ABSENT: no
# fit_frozen_residual_sd, because WP fits no residual SD (D31-07).
from backtest.ou_ev_chain import (
    P_OVER_CLIP,
    RELIABILITY_TOLERANCE,
    american_to_payout,
    calibration_gate,
    devig,
    per_bet_ev,
)
from backtest.simulation import BettingSimulator, SimulationConfig
from models.clv import compute_probability_clv

__all__ = [
    "WP_CHAIN_POLICY",
    "WP_CLV_METRIC",
    "WP_PROB_CLIP",
    "WP_REQUIRED_FIELDS",
    "WP_SIDES",
    "WPGateResult",
    "apply_wp_fallback_correction",
    "calibrated_p_home_side",
    "price_wp_candidates",
    "run_wp_calibration_gate",
    "wp_two_sided_prices",
]

# The probability clip band, BOUND BY IDENTITY to the O/U band so all three targets clip in
# one place. It applies ONLY to the registered fallback's corrected probability: the
# DEFAULT path returns the deployed isotonic probability untouched, because clipping a
# calibrated model's own output would be a silent edit to the model being judged.
WP_PROB_CLIP: tuple[float, float] = P_OVER_CLIP

# The LOCKED WP side vocabulary, from ``BettingSimulator._determine_bet_side_wp``.
WP_SIDES: tuple[str, str] = ("home", "away")

# The market columns a WP candidate row must carry. Both moneylines are REQUIRED: a
# moneyline market has a real two-sided price by construction, so a missing one is missing
# DATA and pricing it at a flat default would invent a market.
WP_REQUIRED_FIELDS: tuple[str, ...] = ("model_prob", "ml_home", "ml_away")

WP_CLV_METRIC: str = (
    "probability_clv (side probability - devigged fair closing probability); "
    "REPORT-ONLY (D27-06, CLV_P_VALUE_IS_REPORT_ONLY)"
)


@dataclass(frozen=True)
class WPGateResult:
    """The TUNE-split calibration gate verdict, carrying its own fallback registration.

    Attributes:
        passed: The gate verdict. True means the deployed isotonic probability is used
            UNCHANGED (the D31-07 default).
        report: The full ``calibration_gate`` report -- bin table, per-bin counts, ECE, max
            bin deviation and both Brier scores. Carried whole so a reader can check the
            verdict rather than trust it.
        fallback_fired: Whether the registered prior-season probability-scale correction
            fired. Exactly ``not passed``, held as its own field because it is what the
            trial registry records and what the BH family counts.
        fallback_trigger: The reason string written into the registry's ``fallback_trigger``
            field, naming the measured quantity that breached its tolerance. None when the
            fallback did not fire.
    """

    passed: bool
    report: dict[str, Any]
    fallback_fired: bool
    fallback_trigger: str | None


def calibrated_p_home_side(model_prob: float, bet_side: str) -> float:
    """The side-correct probability: ``model_prob`` for home, its complement for away.

    Mirrors EXACTLY what ``backtest/simulation.py:531-533`` already computes for WP
    (``model_value = model_prob if bet_side == "home" else (1.0 - model_prob)``), so the
    chain and the simulator cannot diverge on what "the probability of the side bet" means.

    Raises:
        ValueError: naming the LOCKED side vocabulary, for any other side string.
    """
    if bet_side == "home":
        return float(model_prob)
    if bet_side == "away":
        return 1.0 - float(model_prob)
    msg = (
        f"unknown WP bet side {bet_side!r}; the LOCKED WP side vocabulary is "
        f"{list(WP_SIDES)} (BettingSimulator._determine_bet_side_wp). A side outside it is "
        "a caller error, never a default."
    )
    raise ValueError(msg)


def wp_two_sided_prices(ml_home: float | None, ml_away: float | None) -> dict[str, Any]:
    """Devig BOTH real moneylines through the EXISTING ``devig`` (D27-13, D31-07 step 4).

    D27-13 built ``devig`` PLUGGABLE with explicit odds arguments for exactly this: a real
    two-sided price slots in with no rework. This function passes the prices and writes no
    proportional arithmetic of its own -- the same proportional method
    ``models.clv.compute_probability_clv`` already uses for moneylines.

    Both prices are REQUIRED. ``devig``'s flat -110 default exists for the O/U total, whose
    stored rows carry no two-sided juice; a moneyline market always has a real two-sided
    price, so a missing one is missing DATA and defaulting it would invent a market and
    quietly price a bet at a number no book offered.

    Returns:
        ``{fair_home, fair_away, payout_home, payout_away, breakeven, method}`` with
        ``method == "real_two_sided"``.

    Raises:
        ValueError: when either moneyline is absent.
    """
    if ml_home is None or ml_away is None:
        msg = (
            "wp_two_sided_prices requires BOTH moneylines (ml_home="
            f"{ml_home!r}, ml_away={ml_away!r}). A moneyline market has a real two-sided "
            "price by construction, so a missing one is missing DATA; the flat -110 "
            "fallback in devig() exists for the O/U total and must never invent a "
            "moneyline that no book offered."
        )
        raise ValueError(msg)

    priced = devig(over_odds=ml_home, under_odds=ml_away)
    return {
        "fair_home": priced["fair_over"],
        "fair_away": priced["fair_under"],
        "payout_home": american_to_payout(float(ml_home)),
        "payout_away": american_to_payout(float(ml_away)),
        "breakeven": priced["breakeven"],
        "method": priced["method"],
    }


def run_wp_calibration_gate(
    p_home_tune: np.ndarray, realized_home_win_tune: np.ndarray
) -> WPGateResult:
    """Run the frozen reliability + Brier gate on the TUNE split only (D31-07 step 2).

    CONSUMES ``backtest.ou_ev_chain.calibration_gate`` with its frozen parameters (5
    equal-count quantile bins, at least 20 observations per bin, a maximum per-bin
    ``|mean_pred - realized|`` of 0.10). The deployed probability is passed as BOTH the
    corrected and the raw series, because on WP's default path there is no correction; see
    the module docstring on why the reliability arm is the one that binds.

    An UNDER-FILLED tune split RAISES and is NOT converted into a gate failure. The
    difference matters: a failure fires the registered fallback, so silently turning "too
    few observations to judge" into "judged and failed" would apply a correction the run
    never had the evidence to justify.

    Args:
        p_home_tune: The deployed isotonic P(home) over the TUNE seasons.
        realized_home_win_tune: The realized 0/1 home-win outcomes over the same rows.

    Returns:
        A :class:`WPGateResult`.

    Raises:
        ValueError: from ``calibration_gate``, when the tune split cannot fill its bins.
    """
    probabilities = np.asarray(p_home_tune, dtype=float)
    realized = np.asarray(realized_home_win_tune, dtype=float)

    passed, report = calibration_gate(probabilities, probabilities, realized)

    if passed:
        return WPGateResult(
            passed=True, report=report, fallback_fired=False, fallback_trigger=None
        )

    trigger = (
        "WP calibration gate FAILED on the tune split: max_bin_deviation="
        f"{round(float(report['max_bin_deviation']), 4)} against the frozen "
        f"RELIABILITY_TOLERANCE={RELIABILITY_TOLERANCE}; brier_corrected="
        f"{round(float(report['brier_corrected']), 6)} vs brier_raw="
        f"{round(float(report['brier_raw']), 6)}. The REGISTERED prior-season "
        "probability-scale bias correction fires (D31-07); this string is what the trial "
        "registry's fallback_trigger field records, and a fired fallback adds one entry to "
        "the BH family."
    )
    return WPGateResult(
        passed=False, report=report, fallback_fired=True, fallback_trigger=trigger
    )


def apply_wp_fallback_correction(model_prob: float, season_bias: float) -> float:
    """The REGISTERED fallback: a prior-season PROBABILITY-scale shift, then a clip.

    ``corrected = clip(model_prob + season_bias, *WP_PROB_CLIP)``. The bias is the
    prior-season walk-forward mean of ``realized - p_home``, produced by the existing
    ``estimate_prior_season_bias`` over probability-scale residuals; no new estimator is
    introduced and no log-space transform is applied.

    The clip is what keeps a large correction inside the unit interval; a probability at a
    bound CAPS the Kelly fraction rather than demanding an unbounded stake.

    This function is reached ONLY when :func:`run_wp_calibration_gate` returned a failure.
    On the default path the deployed probability is returned untouched by
    :func:`price_wp_candidates`, which never calls this.
    """
    corrected = float(model_prob) + float(season_bias)
    return float(np.clip(corrected, WP_PROB_CLIP[0], WP_PROB_CLIP[1]))


def price_wp_candidates(
    rows: Iterable[Mapping[str, Any]],
    fit: ChainFit,
    gate: WPGateResult | None = None,
    *,
    simulator: BettingSimulator | None = None,
) -> ChainPricing:
    """Price every WP candidate end to end, in the canonical publication order.

    This function DECIDES NOTHING. It prices: side, side-correct probability, devigged fair
    price and payout, and per-bet EV. Admission against the EV floor ``t``, sizing and
    grading stay in the single bet-decision source (``backtest.bet_selector.BetSelector``,
    LOCKED-2); Plan 31-10 registers the ``WPStrategy`` that routes this pricing into it.

    An EMPTY candidate frame returns an empty record list and a NULL CLV report, and raises
    nothing (SPEC R1 empty-case).

    Args:
        rows: Candidate rows carrying ``game_id``, ``season``, ``week``, ``model_prob`` (the
            deployed isotonic P(home)) and both real moneylines.
        fit: The Phase-31 fit provenance. ``fit.frozen_sd`` is None for WP BY DESIGN, and
            ``fit.season_bias_by_season`` is consulted ONLY when the gate failed.
        gate: The TUNE-split gate result. None is treated as the DEFAULT path -- the
            deployed probability used unchanged, no fallback -- which is what a caller that
            has not yet run the gate must not silently get away with claiming, so the
            fallback fields are stamped onto every record either way.
        simulator: An injected simulator, so exactly one exists per run.

    Returns:
        A :class:`ChainPricing`.

    Raises:
        LeakageError: when any fitted parameter on ``fit`` saw a hold season. WP fits no
            residual SD, but its calibration-gate tune split and its registered fallback's
            bias pool are both fitted inputs and are both fenced.
        KeyError: naming the column, when a required market field is absent.
        ValueError: when the fallback fired and a candidate season has no prior-season bias.
    """
    fence_report = assert_fit_window_p31(fit)

    sim = simulator if simulator is not None else BettingSimulator(SimulationConfig())
    fallback_fired = bool(gate is not None and gate.fallback_fired)
    fallback_trigger = gate.fallback_trigger if gate is not None else None

    records: list[dict[str, Any]] = []
    for row in rows:
        missing = [name for name in WP_REQUIRED_FIELDS if row.get(name) is None]
        if missing:
            msg = (
                f"WP candidate {row.get('game_id')!r} is missing required market field(s) "
                f"{missing}; required fields are {list(WP_REQUIRED_FIELDS)}."
            )
            raise KeyError(msg)

        season = int(row["season"])
        model_prob = float(row["model_prob"])
        ml_home = float(row["ml_home"])
        ml_away = float(row["ml_away"])

        if fallback_fired:
            season_bias = season_bias_for(
                season, fit.season_bias_by_season, target="wp"
            )
            p_home = apply_wp_fallback_correction(model_prob, season_bias)
        else:
            season_bias = None
            p_home = model_prob

        # The side comes from the LOCKED simulator convention, resolved on the probability
        # the chain will actually price -- unlike ATS and O/U, WP's side rule reads the
        # probability itself, so a fired fallback that moves a probability across the side
        # threshold must move the side with it.
        bet_side = sim._determine_bet_side_wp(p_home)
        prices = wp_two_sided_prices(ml_home, ml_away)

        record: dict[str, Any] = {
            "game_id": row["game_id"],
            "season": season,
            "week": int(row["week"]),
            "target": "wp",
            "bet_side": bet_side,
            "model_prob": model_prob,
            "p_home": p_home,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "season_bias": season_bias,
            "fallback_fired": fallback_fired,
            "fallback_trigger": fallback_trigger,
            "fair_prob": None,
            "calibrated_p_side": None,
            "payout": None,
            "devig_method": prices["method"],
            "per_bet_ev": None,
            "clv": None,
        }

        if bet_side is not None:
            p_side = calibrated_p_home_side(p_home, bet_side)
            payout = (
                prices["payout_home"] if bet_side == "home" else prices["payout_away"]
            )
            fair_prob = (
                prices["fair_home"] if bet_side == "home" else prices["fair_away"]
            )
            record["calibrated_p_side"] = p_side
            record["fair_prob"] = fair_prob
            record["payout"] = payout
            record["per_bet_ev"] = per_bet_ev(p_side, payout)
            # REPORT-ONLY probability CLV against the devigged fair closing price, through
            # the existing LOCKED helper. Never a gate (D27-06); the CLV p-value is
            # forbidden from the multiplicity family and from any verdict token.
            record["clv"] = compute_probability_clv(
                model_prob=p_side,
                closing_ml_home=ml_home,
                closing_ml_away=ml_away,
                side=bet_side,
            )["probability_clv"]

        records.append(record)

    records.sort(key=chain_order_key)
    return ChainPricing(
        records=records,
        clv_report=chain_clv_report(records, WP_CLV_METRIC),
        fence_report=fence_report,
    )
