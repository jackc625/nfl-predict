"""ATS EV chain (Phase 31, plan 31-07; SPEC R1, PROD-02) plus the shared chain scaffolding.

A PURE numeric transform that converts a scored ATS model spread into a calibrated
P(cover) and then a per-bet EV:

    model_spread (a predicted HOME MARGIN)
      -> bias-correct with the PRIOR-SEASON walk-forward mean residual   (D31-13, ATS_BIAS_CARRIED)
      -> P(home cover) = 1 - norm.cdf(z) with a SINGLE FROZEN SD          (SPEC R1)
      -> devig (EXPLICIT prices; flat -110 when the juice is absent)      (D27-13, consumed)
      -> per-bet EV with an EXPLICIT payout                               (D27-14, consumed)

It is a PURE numeric transform: NO file I/O, NO DuckDB, NO request path, NO model re-fit,
NO production swap. It fits nothing itself -- every fitted nuisance parameter arrives on a
:class:`ChainFit` that names the seasons it was fit on, which is what makes the Phase-31
leakage fence possible at all.

WHAT IS CONSUMED, NEVER REDEFINED
---------------------------------
``devig``, ``per_bet_ev``, ``estimate_prior_season_bias``, ``fit_frozen_residual_sd`` and
``calibration_gate`` are IMPORTED from ``backtest.ou_ev_chain`` and re-exported here. A
second implementation of any of them would be a second answer to a settled question
(T-31-31). ``ATS_RESIDUAL_CONTRACT`` and ``P_COVER_CLIP`` are likewise BOUND BY IDENTITY to
the frozen pre-registration and to the O/U clip band rather than restated: the
pre-registration is ratified and frozen, so a restated copy could drift with no legitimate
repair path.

THE ATS RESIDUAL DIRECTION IS THE OPPOSITE OF O/U'S
---------------------------------------------------
The measured tune-window POOLED bias is POSITIVE (+0.58937727047262922 over n=1139): the
model UNDER-predicts the home margin across the window, so adding the prior-season bias
pushes the corrected margin UP and RAISES the cover probability. In the O/U chain the bias
is NEGATIVE and adding it LOWERS P(over). A sign guard copied from the O/U guard would
therefore assert the wrong direction, which is why ``tests/unit/test_ats_ev_chain.py``
hand-computes its own. The full contract, including the one measured tune season (2022)
whose mean is negative and the reason a per-season walk-forward correction does not depend
on a constant sign, is :data:`ATS_RESIDUAL_CONTRACT`.

THE MARKET SPREAD AND THE MODEL SPREAD ARE ON THE SAME SCALE (DEF-31-01, RULED 2026-09-04)
-------------------------------------------------------------------------------------------
This is load-bearing and is stated here because getting it wrong silently prices the wrong
side of every ATS bet -- which is what this module did until the ruling.

  * The MARKET SPREAD (``spread`` in silver / gold, ``closing_spread`` on a candidate row) is
    nflverse ``spread_line``, written straight through by
    ``scripts/ingest_historical_odds.py:557`` with NO negation at ingest, on load, or in the
    feature build. It is POSITIVE when the home team is favored, and it IS the cover
    threshold on the HOME-MARGIN scale: the home team covers iff
    ``actual_margin > spread``.
  * The MODEL PREDICTION (``model_spread``) is a predicted HOME MARGIN: POSITIVE when the
    home team is expected to win. ``models/train_ats.py`` regresses ``actual_margin``, and
    ``scripts/audit_odds_preingest.measure_ats_residual_bias`` -- the Plan 31-02 measurement
    the frozen contract quotes -- computes ``residual = actual - model_prob`` directly on
    that margin scale.

So the two are DIRECTLY COMPARABLE and neither needs converting into the other. The measured
evidence, three independent cross-checks, is recorded in the phase's ``deferred-items.md``
under DEF-31-01: ``corr(spread, ml_home) = -0.9525`` (n=1856); big HOME favourites
(``ml_home <= -300``) carry a mean stored spread of ``+10.184`` against ``-9.587`` for big
AWAY favourites; and ``actual_margin > spread`` grades a 47.2% home-cover rate where
``actual_margin + spread > 0`` grades an implausible 56.2%.

WHERE A NEGATION IS STILL REQUIRED, AND WHY IT IS ONLY THERE. Two LOCKED helpers are written
in the OPPOSITE "line" convention (negative = home favored) and are NOT re-implemented here:

  * ``BettingSimulator._determine_bet_side_ats(a, b)`` returns ``home_cover`` when ``a < b``,
    which is the line reading. Both of its arguments are therefore NEGATED on the way in.
  * ``apply_slippage_spread(line, side)`` returns ``line - slippage`` for ``home_cover``,
    which moves a LINE against the bettor. Its argument is negated on the way in and its
    result negated back out, giving ``spread + slippage`` for a home-cover bet -- a HIGHER
    margin to clear, which is what "against the bettor" means on this scale.

``BettingSimulator._resolve_ats_outcome(side, actual_margin, slipped_line)`` needs NO
conversion: it already grades ``home_covers = actual_margin > slipped_line``, which is the
measured convention exactly. The slipped market spread reaches it un-negated.

TWO THINGS THIS RULING DELIBERATELY DID NOT CHANGE, recorded so neither reads as an oversight
(both are in DEF-31-01's ruling entry):

  * ``models/train.py:200-202`` keeps its comment and its ``actual_margin + spread > 0``
    baseline rule. That ``actual_cover`` feeds ONLY ``_compute_ats_baseline``, a reported
    market-accuracy diagnostic that trains nothing, and correcting it would move a published
    figure from 0.5618 to about 0.472. The owner declined that half.
  * The legacy no-selector simulator ATS path keeps its pre-existing convention (DEF-31-02),
    because D31-04 forbids moving /betting's published 1073-row spread population. The
    phase's bet list and 2025 verdict route through the selector, not that branch.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from backtest.diagnose import clv_significance
from backtest.ev_chain_constants import (
    ATS_BIAS_CARRIED,
    HOLD_SEASONS_P31,
    PRIOR_RESIDUAL_SEASONS_P31,
    REHEARSAL_PROXY_SPLIT,
    TUNE_SEASONS_P31,
)
from backtest.ev_chain_constants import (
    ATS_RESIDUAL_CONTRACT as _FROZEN_ATS_RESIDUAL_CONTRACT,
)

# The leakage-clean PRE-HOLD high-total boundary derivation, re-run by fence check (d) so a
# drifted or hold-informed O/U eligibility boundary is caught rather than trusted (LOCKED-1).
from backtest.ou_divergence import derive_high_total_boundary

# The five numeric primitives, CONSUMED from the Phase-27 chain and re-exported so callers
# reach exactly one implementation of each (T-31-31). Importing them does NOT import the
# Phase-27 WINDOW -- TUNE_SEASONS / HOLD_SEASONS are deliberately not read anywhere in
# Phase-31 code (D31-14), which tests/unit/test_p31_constants_isolation.py asserts by AST
# scan.
from backtest.ou_ev_chain import (
    MINUS_110_PAYOUT,
    P_OVER_CLIP,
    american_to_payout,
    calibration_gate,
    devig,
    estimate_prior_season_bias,
    fit_frozen_residual_sd,
    per_bet_ev,
)

# LeakageError already means "the fit-window fence was violated" in this repository, so the
# Phase-31 fence RAISES IT rather than minting a second exception for one fact. Note that
# only the EXCEPTION is taken from that module -- the Phase-27 fence helper itself reads the
# Phase-27 window and is deliberately not imported (D31-14).
from backtest.ou_monetization import LeakageError
from backtest.simulation import (
    SLIPPAGE_POINTS,
    BettingSimulator,
    SimulationConfig,
    apply_slippage_spread,
)
from models.clv import compute_line_clv

__all__ = [
    "ATS_BIAS_CARRIED",
    "ATS_RESIDUAL_CONTRACT",
    "ATS_SIDES",
    "FENCE_STAGE_BIAS",
    "FENCE_STAGE_BOUNDARY",
    "FENCE_STAGE_THRESHOLD",
    "FENCE_STAGE_TUNE_FIT",
    "FENCE_WINDOW_P31",
    "FENCE_WINDOW_REHEARSAL",
    "P_COVER_CLIP",
    "REGISTERED_FENCE_WINDOWS",
    "THRESHOLD_WINDOW_P31",
    "ChainFit",
    "ChainPricing",
    "FenceWindow",
    "LeakageError",
    "assert_fit_window_p31",
    "ats_side_probability",
    "ats_two_sided_prices",
    "calibrated_p_cover",
    "calibration_gate",
    "chain_clv_report",
    "chain_order_key",
    "devig",
    "estimate_prior_season_bias",
    "fit_frozen_residual_sd",
    "per_bet_ev",
    "price_ats_candidates",
    "season_bias_for",
    "threshold_window_label",
]


# ---------------------------------------------------------------------------
# The contract and the clip band -- BOUND BY IDENTITY, never restated
# ---------------------------------------------------------------------------

# The frozen pre-registration's ATS residual contract, re-exported under the name the chain
# reads. It is the SAME OBJECT, not a copy: ``backtest/ev_chain_constants.py`` is frozen and
# ratified, so a restated copy here would be a second source of truth with no legitimate
# repair path if the two ever disagreed.
ATS_RESIDUAL_CONTRACT: str = _FROZEN_ATS_RESIDUAL_CONTRACT

# The cover-probability clip band. BOUND BY IDENTITY to the O/U band because the ratified
# pre-registration says ATS is "clipped to the same disclosed ``P_OVER_CLIP`` band". The
# clip bounds the Kelly tail -- a probability at the bound CAPS the Kelly fraction rather
# than demanding an unbounded stake -- and extreme z-scores resolve to these bounds rather
# than to 0.0, 1.0 or NaN. Disclosed in ``calibrated_p_cover``'s docstring.
P_COVER_CLIP: tuple[float, float] = P_OVER_CLIP

# The LOCKED ATS side vocabulary, from ``BettingSimulator._determine_bet_side_ats``.
ATS_SIDES: tuple[str, str] = ("home_cover", "away_cover")


def threshold_window_label(tune_seasons: Sequence[int]) -> str:
    """The EV-floor sweep's window label for a tune window. ONE derivation, never a literal.

    The label travels on a :class:`ChainFit` and fence check (b) compares it against the
    label derived from the window being fenced, so a sweep logged on one window can never
    satisfy a fence built for another.
    """
    return f"tune_{tune_seasons[0]}_{tune_seasons[-1]}"


# The threshold-tuning window label the Phase-31 EV-floor sweep must carry. Derived from the
# frozen Phase-31 window so it cannot drift away from it; never a literal.
THRESHOLD_WINDOW_P31: str = threshold_window_label(TUNE_SEASONS_P31)


@dataclass(frozen=True)
class FenceWindow:
    """The tune/hold/seed triple a fence run is measured against (plan 31-12).

    WHY THE FENCE IS PARAMETERISED AT ALL, AND WHY THAT IS NOT A WEAKENING. The Phase-31
    rehearsal runs on the DISJOINT ``REHEARSAL_PROXY_SPLIT`` (tune 2021-2023, hold 2024) so
    the plumbing can be proven on an already-burned season without spending 2025. A fence
    hard-wired to the frozen window would then either pass VACUOUSLY -- 2024 is a frozen tune
    season, so no check would bind and the report would name 2025 as the hold, which is
    false -- or force the rehearsal to run with no fence at all. Both outcomes are worse than
    a parameter.

    What stops the parameter from becoming an escape hatch is that only REGISTERED windows
    are accepted (:data:`REGISTERED_FENCE_WINDOWS`). An arbitrary tune/hold pair cannot be
    handed to :func:`assert_fit_window_p31`; it raises. So there are exactly two windows a
    fence run can be measured against, both declared in the frozen pre-registration, and the
    armed production run uses the frozen one by construction.

    Attributes:
        tune_seasons: The seasons a tune-only fit may consume.
        hold_seasons: The seasons no fitted parameter may see.
        prior_residual_seasons: The strictly-prior bias seed, admissible in a bias pool.
        threshold_window: The label the EV-floor sweep must have logged.
        label: A short machine-readable name for the window, carried into the fence report.
        is_the_preregistered_rule: True ONLY for the frozen Phase-31 window. The rehearsal
            window says False about itself, so a report cannot be mistaken for a binding one.
    """

    tune_seasons: tuple[int, ...]
    hold_seasons: tuple[int, ...]
    prior_residual_seasons: tuple[int, ...]
    threshold_window: str
    label: str
    is_the_preregistered_rule: bool


# The FROZEN Phase-31 window: tune 2021-2024, hold 2025, seed 2018-2020.
FENCE_WINDOW_P31: FenceWindow = FenceWindow(
    tune_seasons=TUNE_SEASONS_P31,
    hold_seasons=HOLD_SEASONS_P31,
    prior_residual_seasons=PRIOR_RESIDUAL_SEASONS_P31,
    threshold_window=THRESHOLD_WINDOW_P31,
    label="frozen_p31",
    is_the_preregistered_rule=True,
)

# The DISJOINT rehearsal window, built FROM the frozen REHEARSAL_PROXY_SPLIT rather than from
# literals, so the two cannot drift apart.
_REHEARSAL_TUNE: tuple[int, ...] = tuple(
    int(season)
    for season in REHEARSAL_PROXY_SPLIT["tune_seasons"]  # type: ignore[union-attr]
)
_REHEARSAL_HOLD: tuple[int, ...] = tuple(
    int(season)
    for season in REHEARSAL_PROXY_SPLIT["hold_seasons"]  # type: ignore[union-attr]
)

FENCE_WINDOW_REHEARSAL: FenceWindow = FenceWindow(
    tune_seasons=_REHEARSAL_TUNE,
    hold_seasons=_REHEARSAL_HOLD,
    prior_residual_seasons=PRIOR_RESIDUAL_SEASONS_P31,
    threshold_window=threshold_window_label(_REHEARSAL_TUNE),
    label="rehearsal_proxy",
    is_the_preregistered_rule=False,
)

# The CLOSED set of windows a fence run may be measured against. A window outside it is
# refused, so "parameterised" never becomes "arbitrary".
REGISTERED_FENCE_WINDOWS: tuple[FenceWindow, ...] = (
    FENCE_WINDOW_P31,
    FENCE_WINDOW_REHEARSAL,
)


# ---------------------------------------------------------------------------
# Shared Phase-31 chain scaffolding
#
# WHY IT LIVES HERE AND NOT IN A FOURTH MODULE. tests/unit/test_p31_constants_isolation.py
# enumerates the Phase-31 modules BY NAME in ``P31_MODULES`` and AST-scans the ones present.
# A module absent from that registry is a module the scan never visits -- which is exactly
# the T-31-19 failure the scan exists to catch, since a passing suite is what that failure
# produces. ``ats_ev_chain`` and ``wp_ev_chain`` are both already registered, so the shared
# pieces live in one of the two rather than in an unscanned new file.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainFit:
    """The fitted nuisance parameters for one target, WITH the seasons they were fit on.

    Bundling the parameters with their provenance is what makes the Phase-31 leakage fence
    possible: a bare ``frozen_sd`` float carries no evidence about which seasons produced
    it, so a fence over one could only ever assert that a number is a number.

    Attributes:
        target: The target code (``"wp"`` / ``"ats"`` / ``"ou"``).
        frozen_sd: The single frozen residual SD, or None for a target that fits none. WP
            fits none BY DESIGN (D31-07): a calibrated classifier has no residual to take a
            standard deviation of, and inventing a logit-space one was explicitly REJECTED.
        season_bias_by_season: Target season -> the prior-season walk-forward bias applied
            to it, produced by ``estimate_prior_season_bias``.
        tune_fit_seasons: The seasons consumed by the TUNE-ONLY fit -- the frozen-SD fit for
            ATS and O/U, the calibration-gate tune split for WP. Empty for a target that
            fits nothing on the tune split.
        threshold_window: The window label the EV-floor sweep logged. Must be
            :data:`THRESHOLD_WINDOW_P31`.
        bias_pool_by_season: Target season -> the seasons its bias estimate actually
            consumed. The fence asserts each pool is STRICTLY PRIOR and hold-free.
        high_total_boundary: O/U's eligibility boundary, or None for a target that has no
            eligibility gate (D31-05 gives WP and ATS none).
    """

    target: str
    frozen_sd: float | None
    season_bias_by_season: Mapping[int, float]
    tune_fit_seasons: tuple[int, ...]
    threshold_window: str
    bias_pool_by_season: Mapping[int, tuple[int, ...]]
    high_total_boundary: float | None = None


@dataclass(frozen=True)
class ChainPricing:
    """The output of a per-target chain over a candidate frame.

    Attributes:
        records: The priced per-candidate records, in the canonical publication order
            (``game_id`` then ``target``, both ascending). EMPTY for an empty candidate
            frame -- never a raise.
        clv_report: The REPORT-ONLY CLV summary over the priced records, or None when there
            is nothing to summarize. NEVER a gate (D27-06), and NEVER consulted by the
            verdict path (``CLV_P_VALUE_IS_REPORT_ONLY``).
        fence_report: The Phase-31 fit-window fence report, proving which seasons each fit
            consumed. Empty until the fence runs.
    """

    records: list[dict[str, Any]]
    clv_report: dict[str, Any] | None
    fence_report: dict[str, Any]


def chain_order_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """The canonical publication order for a chain record: ``game_id`` then ``target``.

    ONE implementation, so the ordering the determinism test asserts is the ordering the
    readout publishes. Both components are stringified so a mixed-type ``game_id`` column
    cannot raise mid-sort and produce a partially ordered list.
    """
    return (str(record["game_id"]), str(record["target"]))


def chain_clv_report(
    records: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any] | None:
    """REPORT-ONLY CLV summary over ``records`` via the LOCKED ``clv_significance``.

    Records whose ``clv`` is None are DROPPED rather than counted as zero. A CLV that was
    never measured entering a published mean as 0.0 would read as "no edge" rather than as
    "not measured" -- the ``_clv_report`` defaulting bug Plan 31-06 already fixed once.

    Args:
        records: Priced chain records, each optionally carrying a ``clv`` float.
        metric: A human-readable name for WHICH closing-line value this is, carried into the
            report so a reader cannot mistake a model-edge line CLV for a forward one.

    Returns:
        The ``clv_significance`` dict extended with ``metric``, or None when no record
        carries a measured CLV.
    """
    values = [
        float(record["clv"]) for record in records if record.get("clv") is not None
    ]
    if not values:
        return None
    report = dict(clv_significance(values))
    report["metric"] = metric
    return report


def season_bias_for(
    season: int, bias_by_season: Mapping[int, float], *, target: str
) -> float:
    """The prior-season walk-forward bias for ``season``, RAISING when it is absent.

    The no-silent-fallback shape the selector already uses (``OUStrategy._season_bias``):
    falling back to the raw, uncorrected prediction would silently re-inject the very bias
    the chain exists to remove, and would do so invisibly.

    Args:
        season: The season being priced.
        bias_by_season: Target season -> prior-season walk-forward bias.
        target: The target code, named in the error so a multi-target run says WHICH chain
            is missing a bias.

    Returns:
        The bias as a float.

    Raises:
        ValueError: naming the season, the target and the estimator, when the season is
            absent from the mapping.
    """
    if season not in bias_by_season:
        msg = (
            f"no prior-season bias provided for season {season} on target {target!r}; the "
            "caller must supply a walk-forward bias from "
            "backtest.ou_ev_chain.estimate_prior_season_bias for every candidate season. "
            "There is NO silent fallback to the raw, uncorrected prediction (D27-07)."
        )
        raise ValueError(msg)
    return float(bias_by_season[season])


# ---------------------------------------------------------------------------
# The Phase-31 fit-window / leakage fence (T-31-28, SPEC R1)
#
# WHY THIS IS A NEW FUNCTION AND NOT A REUSE. backtest/ou_monetization's Phase-27 fence
# helper READS the Phase-27 window constants. A Phase-31 runner that reused it would fence
# against 2023-2024, and 2023-2024 are TUNE seasons here -- so the fence would still PASS,
# nothing would raise at run time, and its report would name the wrong hold seasons. A
# passing suite is exactly what that failure produces, which is why D31-14 forbids reading
# the Phase-27 window at all and tests/unit/test_p31_constants_isolation.py enforces it by
# AST scan. The four checks below MIRROR that helper's four; only the constants differ.
#
# The EXCEPTION TYPE is reused: LeakageError already means "the fit-window fence was
# violated" in this repository, and a second exception class for the same condition would be
# a second vocabulary for one fact.
# ---------------------------------------------------------------------------

# The stage labels every fence message names, so a raise says WHICH fit leaked rather than
# only that something did.
FENCE_STAGE_TUNE_FIT: str = "tune-only fit input"
FENCE_STAGE_THRESHOLD: str = "EV-floor threshold tuning"
FENCE_STAGE_BIAS: str = "prior-season bias estimate"
FENCE_STAGE_BOUNDARY: str = "high-total boundary derivation"

_SEASON_IN_LABEL = re.compile(r"\d{4}")


def assert_fit_window_p31(
    fit: ChainFit, window: FenceWindow = FENCE_WINDOW_P31
) -> dict[str, Any]:
    """Prove NO hold season fed any fitted parameter on ``fit`` (T-31-28, SPEC R1).

    Mirrors the four checks the Phase-27 fence performs, parameterised on the FROZEN
    Phase-31 constants (``TUNE_SEASONS_P31`` 2021-2024, ``HOLD_SEASONS_P31`` 2025,
    ``PRIOR_RESIDUAL_SEASONS_P31`` 2018-2020):

      (a) the TUNE-ONLY fit input -- the frozen-SD fit for ATS and O/U, the calibration-gate
          tune split for WP -- must be disjoint from the hold and inside the tune window;
      (b) the EV-floor sweep must have been tuned on the Phase-31 window label;
      (c) every per-season bias pool must be STRICTLY PRIOR to the season it debiases, must
          be hold-free, and must lie inside the tune window plus the 2018-2020 seed;
      (d) when a high-total boundary is supplied (O/U only), it must equal the leakage-clean
          PRE-HOLD derivation, so a drifted or hold-informed boundary is caught.

    Check (c) asserts STRICT PRIORITY as well as hold-freeness. The Phase-27 helper checked
    only the latter, which is sufficient when tune strictly precedes hold; here the hold is a
    single later season, so a pool containing the target season itself would slip past a
    hold-only check while being the most direct leak available.

    Args:
        fit: The target's fitted parameters WITH the seasons they were fit on.
        window: The REGISTERED window to fence against. Defaults to the frozen Phase-31 one,
            so every pre-existing call site is unchanged. The only other admissible value is
            :data:`FENCE_WINDOW_REHEARSAL`; see :class:`FenceWindow` for why the parameter
            exists and why it is not an escape hatch.

    Returns:
        A structured fence report naming the seasons each fit consumed and the hold it was
        measured against, so the report a run publishes cannot name the wrong hold seasons.

    Raises:
        LeakageError: naming the offending season(s) and the fit stage, or naming an
            unregistered window.
    """
    if not any(window == registered for registered in REGISTERED_FENCE_WINDOWS):
        msg = (
            f"[{fit.target}] refusing to fence against the UNREGISTERED window "
            f"{window!r}. Exactly two windows are admissible -- the frozen Phase-31 rule and "
            "the disjoint rehearsal proxy, both declared in the frozen pre-registration. An "
            "arbitrary tune/hold pair handed to the fence is a weakened fence, which is the "
            "one failure a leakage guard must never permit (T-31-28)."
        )
        raise LeakageError(msg)

    hold = set(window.hold_seasons)
    tune = set(window.tune_seasons)
    seed = set(window.prior_residual_seasons)
    expected_threshold_window = window.threshold_window
    target = fit.target

    # (a) The tune-only fit input.
    tune_fit = {int(season) for season in fit.tune_fit_seasons}
    leaked = sorted(tune_fit & hold)
    if leaked:
        msg = (
            f"[{target}] {FENCE_STAGE_TUNE_FIT} consumed HOLD season(s) {leaked}; the "
            f"Phase-31 hold is {sorted(hold)} and the tune window is {sorted(tune)}. "
            "No fitted parameter may see 2025 (T-31-28)."
        )
        raise LeakageError(msg)
    outside = sorted(tune_fit - tune)
    if outside:
        msg = (
            f"[{target}] {FENCE_STAGE_TUNE_FIT} consumed season(s) {outside} outside the "
            f"Phase-31 tune window {sorted(tune)} (T-31-28)."
        )
        raise LeakageError(msg)

    # (b) The threshold-tuning window label.
    if fit.threshold_window != expected_threshold_window:
        offending = sorted(
            {
                int(found)
                for found in _SEASON_IN_LABEL.findall(fit.threshold_window)
                if int(found) in hold
            }
        )
        detail = (
            f" It names HOLD season(s) {offending}."
            if offending
            else " It is not the Phase-31 window label."
        )
        msg = (
            f"[{target}] {FENCE_STAGE_THRESHOLD} logged window "
            f"{fit.threshold_window!r}, which is not {expected_threshold_window!r}.{detail} "
            "The EV floor t cannot be tuned on data the hold contains (T-31-28)."
        )
        raise LeakageError(msg)

    # (c) Every per-season bias pool.
    for raw_season, raw_pool in fit.bias_pool_by_season.items():
        season = int(raw_season)
        pool = {int(entry) for entry in raw_pool}

        leaked = sorted(pool & hold)
        if leaked:
            msg = (
                f"[{target}] {FENCE_STAGE_BIAS} for season {season} consumed HOLD "
                f"season(s) {leaked}; the Phase-31 hold is {sorted(hold)} (T-31-28)."
            )
            raise LeakageError(msg)

        not_strictly_prior = sorted(entry for entry in pool if entry >= season)
        if not_strictly_prior:
            msg = (
                f"[{target}] {FENCE_STAGE_BIAS} for season {season} consumed season(s) "
                f"{not_strictly_prior} that are NOT strictly prior to {season}. A "
                "walk-forward estimate reads only earlier seasons; including the target "
                "season debiases it with its own outcomes (T-31-28)."
            )
            raise LeakageError(msg)

        outside = sorted(pool - (tune | seed))
        if outside:
            msg = (
                f"[{target}] {FENCE_STAGE_BIAS} for season {season} consumed season(s) "
                f"{outside} outside the tune window {sorted(tune)} plus the pre-registered "
                f"bias seed {sorted(seed)} (T-31-28)."
            )
            raise LeakageError(msg)

    # (d) The O/U high-total eligibility boundary, when one is supplied.
    pre_hold_boundary: float | None = None
    if fit.high_total_boundary is not None:
        boundary = float(fit.high_total_boundary)
        if not np.isfinite(boundary):
            msg = (
                f"[{target}] {FENCE_STAGE_BOUNDARY} produced a non-finite boundary; the "
                "leakage-clean PRE-HOLD derivation is unavailable, and a NaN boundary would "
                "silently collapse the under-OR-high UNION to under-only (T-31-28, WR-03)."
            )
            raise LeakageError(msg)
        pre_hold_boundary = float(derive_high_total_boundary())
        if abs(boundary - pre_hold_boundary) > 1e-9:
            msg = (
                f"[{target}] {FENCE_STAGE_BOUNDARY}: the supplied boundary {boundary} "
                f"drifted from the leakage-clean pre-hold derivation {pre_hold_boundary} "
                "(LOCKED-1, T-31-28)."
            )
            raise LeakageError(msg)

    return {
        "target": target,
        "tune_fit_seasons": sorted(tune_fit),
        "threshold_window": fit.threshold_window,
        "bias_seasons": sorted(int(season) for season in fit.bias_pool_by_season),
        "tune_seasons": list(window.tune_seasons),
        "hold_seasons": list(window.hold_seasons),
        "prior_residual_seed_seasons": list(window.prior_residual_seasons),
        "window_label": window.label,
        "window_is_the_preregistered_rule": window.is_the_preregistered_rule,
        "high_total_boundary": fit.high_total_boundary,
        "pre_hold_boundary_rederived": pre_hold_boundary,
        "fence_held": True,
    }


# ---------------------------------------------------------------------------
# The ATS converter: bias-correct -> frozen-SD normal CDF -> clip
# ---------------------------------------------------------------------------


def calibrated_p_cover(
    model_spread: float | np.ndarray,
    line: float | np.ndarray,
    frozen_sd: float,
    season_bias: float,
) -> float | np.ndarray:
    """Bias-corrected, clipped P(home cover) -- the direct analogue of ``calibrated_p_over``.

    THE DIRECTION IS PINNED BY FORMULA, not by prose::

        corrected = model_spread + season_bias   # season_bias > 0 pushes the margin UP
        z         = (line - corrected) / frozen_sd
        p_cover   = clip(1 - norm.cdf(z), *P_COVER_CLIP)

    ``1 - norm.cdf(z)`` is identically ``norm.cdf((corrected - line) / frozen_sd)``, i.e.
    ``P(actual home margin > line)`` under a Normal(corrected, frozen_sd) margin. That is
    exactly the event ``backtest/simulation.py:347`` grades as a home cover
    (``home_covers = actual_margin > slipped_line``). The complementary side is exact:

        P(away cover) = 1 - P(home cover)

    and ``ats_side_probability`` is the one place that mapping is applied, so the two sides
    can never be priced as two independent numbers.

    BOTH ``model_spread`` AND ``line`` ARE ON THE HOME-MARGIN SCALE. ``model_spread`` is the
    predicted home margin (POSITIVE when the home team is expected to win). ``line`` is the
    COVER THRESHOLD the actual home margin must EXCEED -- which, under the measured
    convention DEF-31-01 ruled on, IS the stored market spread (POSITIVE when the home team
    is favored), slipped but NOT negated. A caller who negated it first would price the
    wrong side of every game; the module docstring records the measurement.

    The measured tune-window pooled ``season_bias`` is POSITIVE, so the correction RAISES
    the cover probability -- the OPPOSITE direction to the O/U case, where a negative bias
    LOWERS P(over). See :data:`ATS_RESIDUAL_CONTRACT`.

    Vectorized: accepts scalars or numpy arrays. Extreme z-scores resolve to the disclosed
    :data:`P_COVER_CLIP` bounds, never to 0.0, 1.0 or NaN; the clip also bounds the Kelly
    tail, so a probability at a bound CAPS the Kelly fraction rather than demanding an
    unbounded stake.

    Args:
        model_spread: The scored predicted home margin(s).
        line: The cover threshold(s) on the home-margin scale -- the slipped stored spread,
            un-negated.
        frozen_sd: The single frozen residual SD, fit on bias-corrected TUNE residuals only
            via ``fit_frozen_residual_sd``.
        season_bias: The prior-season walk-forward mean residual. Must not be None.

    Returns:
        P(home cover) -- a float for scalar inputs, an ndarray for array inputs -- clipped
        to :data:`P_COVER_CLIP`.

    Raises:
        ValueError: when ``season_bias`` is None (no silent fallback to the raw spread).
    """
    if season_bias is None:
        msg = (
            "season_bias is required (ATS_BIAS_CARRIED is True): it must be the "
            "prior-season walk-forward mean residual; never fall back to the raw biased "
            "spread."
        )
        raise ValueError(msg)

    model_spread_arr = np.asarray(model_spread, dtype=float)
    line_arr = np.asarray(line, dtype=float)

    corrected_margin = model_spread_arr + season_bias
    z = (line_arr - corrected_margin) / frozen_sd
    p_cover = np.clip(1.0 - norm.cdf(z), P_COVER_CLIP[0], P_COVER_CLIP[1])

    # Preserve scalar-in / scalar-out for the hand-computed sign-guard test.
    if np.ndim(model_spread) == 0 and np.ndim(line) == 0:
        return float(p_cover)
    return p_cover


def ats_side_probability(bet_side: str, p_home_cover: float) -> float:
    """Map P(home cover) to the probability of the side actually bet.

    The one place the complement is taken, so ``P(home_cover) + P(away_cover)`` is exactly
    1.0 by construction rather than by two computations agreeing.

    Raises:
        ValueError: naming the LOCKED side vocabulary, for any other side string.
    """
    if bet_side == "home_cover":
        return float(p_home_cover)
    if bet_side == "away_cover":
        return 1.0 - float(p_home_cover)
    msg = (
        f"unknown ATS bet side {bet_side!r}; the LOCKED ATS side vocabulary is "
        f"{list(ATS_SIDES)} (BettingSimulator._determine_bet_side_ats). A side outside it "
        "is a caller error, never a default."
    )
    raise ValueError(msg)


def ats_two_sided_prices(
    home_cover_odds: float | None = None,
    away_cover_odds: float | None = None,
) -> dict[str, Any]:
    """Devig the two ATS spread prices through the EXISTING pluggable ``devig`` (D27-13).

    CONSUMES ``backtest.ou_ev_chain.devig`` and writes no proportional arithmetic of its
    own. ``devig``'s parameters are named for the O/U sides because that is where it was
    built; its two-sided method is target-agnostic and D27-13 built it EXPLICITLY so a real
    price slots in with no rework. The first price is the home-cover price and the second
    the away-cover price.

    When either price is absent the flat -110 symmetric default applies, which is what the
    stored silver spread rows carry today.

    Returns:
        ``{fair_home_cover, fair_away_cover, payout_home_cover, payout_away_cover,
        breakeven, method}``.
    """
    priced = devig(over_odds=home_cover_odds, under_odds=away_cover_odds)
    if priced["method"] == "real_two_sided":
        payout_home = american_to_payout(float(home_cover_odds))  # type: ignore[arg-type]
        payout_away = american_to_payout(float(away_cover_odds))  # type: ignore[arg-type]
    else:
        payout_home = MINUS_110_PAYOUT
        payout_away = MINUS_110_PAYOUT
    return {
        "fair_home_cover": priced["fair_over"],
        "fair_away_cover": priced["fair_under"],
        "payout_home_cover": payout_home,
        "payout_away_cover": payout_away,
        "breakeven": priced["breakeven"],
        "method": priced["method"],
    }


# ---------------------------------------------------------------------------
# The ATS pricing pass over a candidate frame
# ---------------------------------------------------------------------------

# The market columns an ATS candidate row must carry. The juice columns are OPTIONAL: the
# historical rows carry them (Plan 31-02 measured 1,992 of 2,120 distinct pairs at a price
# other than -110), and a row without them prices at the flat -110 default.
ATS_REQUIRED_FIELDS: tuple[str, ...] = ("model_spread", "closing_spread")
ATS_JUICE_FIELDS: tuple[str, str] = ("spread_ju_home", "spread_ju_away")

# The REPORT-ONLY model-edge CLV label. It NAMES ITS FORMULA rather than an interpretation,
# which is what keeps it true across the DEF-31-01 ruling: with both quantities on the
# home-margin scale, ``closing_spread - model_spread`` is POSITIVE when the market favors the
# home team MORE than the model does -- the OPPOSITE reading to the line-convention example in
# ``models/clv.compute_line_clv``'s docstring. The number is unchanged by the ruling and is
# deliberately left where it is: it is never a gate (D27-06), and re-signing a published
# report-only figure was outside the ruled scope. See DEF-31-01's ruling entry.
ATS_CLV_METRIC: str = (
    "model_edge_line_clv (closing_spread - model_spread); REPORT-ONLY (D27-06)"
)


def price_ats_candidates(
    rows: Iterable[Mapping[str, Any]],
    fit: ChainFit,
    *,
    slippage_points: float = SLIPPAGE_POINTS,
    simulator: BettingSimulator | None = None,
) -> ChainPricing:
    """Price every ATS candidate end to end, in the canonical publication order.

    This function DECIDES NOTHING. It prices: side, calibrated P(side), devigged payout and
    per-bet EV. Admission against the EV floor ``t``, sizing and grading stay in the single
    bet-decision source (``backtest.bet_selector.BetSelector``, LOCKED-2); Plan 31-10
    registers the ``ATSStrategy`` that routes this pricing into it. Keeping admission out of
    here is what stops a second bet-decision path existing.

    An EMPTY candidate frame returns an empty record list and a NULL CLV report, and raises
    nothing (SPEC R1 empty-case).

    Args:
        rows: Candidate rows carrying ``game_id``, ``season``, ``week``, ``model_spread``
            (a predicted home MARGIN) and ``closing_spread`` (the stored market spread,
            POSITIVE when the home team is favored, on the SAME margin scale), and
            optionally the two juice columns.
        fit: The fitted nuisance parameters and the seasons they were fit on.
        slippage_points: The half-point slippage, applied through the LOCKED
            ``apply_slippage_spread`` by negating into the LINE convention it was written
            for and negating its result back onto the margin scale.
        simulator: An injected simulator, so exactly one exists per run.

    Returns:
        A :class:`ChainPricing`.

    Raises:
        LeakageError: when any fitted parameter on ``fit`` saw a hold season. The fence runs
            FIRST, before any candidate is priced, so a leaking fit cannot produce a bet list
            that would then have to be retracted.
        ValueError: when ``fit.frozen_sd`` is absent (ATS fits one BY DESIGN), or when a
            candidate season has no prior-season bias.
        KeyError: naming the column, when a required market field is absent.
    """
    fence_report = assert_fit_window_p31(fit)

    if fit.frozen_sd is None:
        msg = (
            "the ATS chain requires a frozen residual SD fit on the TUNE split only "
            "(fit_frozen_residual_sd); ChainFit.frozen_sd is None. Unlike WP, ATS is a "
            "point-prediction target and its converter cannot run without one."
        )
        raise ValueError(msg)
    frozen_sd = float(fit.frozen_sd)

    sim = simulator if simulator is not None else BettingSimulator(SimulationConfig())

    records: list[dict[str, Any]] = []
    for row in rows:
        missing = [name for name in ATS_REQUIRED_FIELDS if row.get(name) is None]
        if missing:
            msg = (
                f"ATS candidate {row.get('game_id')!r} is missing required market "
                f"field(s) {missing}; required fields are {list(ATS_REQUIRED_FIELDS)}."
            )
            raise KeyError(msg)

        season = int(row["season"])
        model_spread = float(row["model_spread"])
        closing_spread = float(row["closing_spread"])

        # The side is resolved on the RAW prediction, exactly as the O/U strategy resolves
        # its side on the raw model total; the bias correction enters the PROBABILITY. The
        # two margins are directly comparable (DEF-31-01), and BOTH are negated into the
        # LOCKED helper's line convention because it returns home_cover when a < b.
        bet_side = sim._determine_bet_side_ats(-model_spread, -closing_spread)

        record: dict[str, Any] = {
            "game_id": row["game_id"],
            "season": season,
            "week": int(row["week"]),
            "target": "ats",
            "bet_side": bet_side,
            "model_spread": model_spread,
            "closing_spread": closing_spread,
            "season_bias": None,
            "corrected_margin": None,
            "slipped_spread": None,
            "cover_threshold": None,
            "p_home_cover": None,
            "calibrated_p_side": None,
            "payout": None,
            "devig_method": None,
            "per_bet_ev": None,
            # REPORT-ONLY model-edge line CLV. Never a gate (D27-06).
            "clv": compute_line_clv(model_spread, closing_spread, direction="spread"),
        }

        if bet_side is not None:
            season_bias = season_bias_for(
                season, fit.season_bias_by_season, target="ats"
            )
            # Slippage is applied by negating into the LINE convention the LOCKED helper was
            # written for and negating its result back, which moves the stored spread AGAINST
            # the bettor on the margin scale: a home-cover bet's threshold RISES. The result
            # is already the cover threshold the converter and _resolve_ats_outcome both use,
            # so ``cover_threshold`` and ``slipped_spread`` are the SAME number under this
            # convention. Both keys are kept: one names the price the bettor got, the other
            # names the margin the game must clear, and a reader should not have to infer
            # that they coincide.
            slipped_spread = -apply_slippage_spread(
                -closing_spread, bet_side, slippage_points
            )
            cover_threshold = slipped_spread
            p_home_cover = float(
                calibrated_p_cover(
                    model_spread, cover_threshold, frozen_sd, season_bias
                )
            )
            p_side = ats_side_probability(bet_side, p_home_cover)
            prices = ats_two_sided_prices(
                row.get(ATS_JUICE_FIELDS[0]), row.get(ATS_JUICE_FIELDS[1])
            )
            payout = (
                prices["payout_home_cover"]
                if bet_side == "home_cover"
                else prices["payout_away_cover"]
            )

            record["season_bias"] = season_bias
            record["corrected_margin"] = model_spread + season_bias
            record["slipped_spread"] = slipped_spread
            record["cover_threshold"] = cover_threshold
            record["p_home_cover"] = p_home_cover
            record["calibrated_p_side"] = p_side
            record["payout"] = payout
            record["devig_method"] = prices["method"]
            record["per_bet_ev"] = per_bet_ev(p_side, payout)

        records.append(record)

    records.sort(key=chain_order_key)
    return ChainPricing(
        records=records,
        clv_report=chain_clv_report(records, ATS_CLV_METRIC),
        fence_report=fence_report,
    )
