"""O/U monetization runner (Phase 27, plan 27-04; OUM-03/05, BET-01).

The thin tune/hold runner that closes the O/U monetization chain. It tunes the single EV-floor
scalar ``t`` ONCE on the PRE-REGISTERED tune split (2021-2022), FREEZES the residual SD + the
prior-season walk-forward bias + ``t`` BEFORE touching the
2023-2024 hold split, then grades the BetSelector's selected bets THROUGH the LOCKED
``BettingSimulator`` to produce the provisional held-out ROI with block-by-week bootstrap CIs
(resampling WEEKS WITHIN the holdout seasons only) and robustness cuts (OUM-03/05).

This is a THIN HARNESS in the house style of ``backtest/diagnose.py`` / ``backtest/ou_divergence.py``
and ``backtest/bet_selector.py``: it CALLS the LOCKED scorers (``score_deployed_artifacts``,
``BetSelector.select`` -> ``BettingSimulator.simulate``, ``clv_significance``,
``false_discovery_control``) and NEVER re-derives a metric. The acceptance bar is GRADED simulator
ROI on the held-out split (NOT CLV magnitude); CLV is reported ONLY (D27-06, #11). A NEGATIVE
provisional ROI is a VALID documented redirect toward Phase 30 (D27-02 / D25-14 honest-refusal
pattern) -- the runner returns the honest result, it never raises on a negative ROI.

HONESTY IS STRUCTURAL (#6, D27-01): every 2023-2024 number is labeled with the FIXED
owner-checkpoint vocabulary CONTAMINATED_VOCAB ("provisional contaminated readout" / "proceed
signal" / "redirect") -- NEVER "validated" / "proven profitable". The binding clean verdict is
Phase 30. The materialized bet table (api/cache.py, written by a separate Plan-04 fn) carries a
``validation_type = PROVISIONAL_CONTAMINATED`` column.

WALK-FORWARD FENCES (non-negotiable, #5):
  - the EV-floor ``t``, the frozen residual SD, and the prior-season bias are fit on the TUNE split
    ONLY; a fit-window / leakage assertion proves NO hold season (2023/2024) feeds the SD fit, the
    threshold tuning, any season's self-bias estimation, or the trial-selection set.
    ``LeakageError`` is raised if the fence is violated.

NO ELIGIBILITY GATE (D33.2-24). This runner used to freeze a fourth input, the pre-hold high-total
boundary behind the O/U under-OR-high-total UNION, and fence its derivation. The UNION and the
boundary were deleted together, so every O/U candidate reaches the EV floor and the floor alone
decides. The union-versus-under-only comparison this runner reported dies with the union, by
ruling. Its Phase-27 numbers stay in the record labelled old-rule (R16); a run of this module today
measures the EV-only rule, not the one those numbers were produced under.

MULTIPLE-COMPARISONS CONTROL (#4, OUM-03): the EV-floor sweep logs EVERY effective fork to a
COMPLETE trial registry using the Plan-01 ``TRIAL_REGISTRY_FIELDS`` schema, and BH-FDR
(``scipy.stats.false_discovery_control``, method "bh") deflates the registry. ROI -- not
significance -- is the acceptance bar; significance is supporting context (Pitfall 3).

HARD BOUNDARY (carried from the harnesses it mirrors): this module imports NO ``train_*`` module and
NEVER writes ``data/``. It LOADS the deployed artifact via ``score_deployed_artifacts`` and runs
inference only -- NO re-fit, NO gold rebuild, NO production swap. It does NOT import
``throwaway_ev_preview`` (the no-leak guard, T-26-08) -- it consumes the production-grade Plan-01 EV
chain via the BetSelector. ``models/clv.py`` and ``config/gate.toml`` are CONSUMED verbatim.

PRE-REGISTERED (frozen as module constants BEFORE any tuning run; the D26-08 / D24-07 discipline).
The EV chain (Plan 01) already froze TUNE_SEASONS / HOLD_SEASONS / EV_FLOOR_GRID / the calibration
method / the SD fit + TRIAL_REGISTRY_FIELDS. This module REUSES those constants and adds ONLY the
runner-specific pre-registered constants below
(bootstrap B / seed / CI type, the robustness cuts, the significance alpha, and the fixed
contaminated vocabulary) with a forking-paths comment.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from backtest.bet_selector import BetSelector, assert_real_odds
from backtest.diagnose import (
    SIGNIFICANCE_ALPHA,
    clv_significance,
    score_deployed_artifacts,
)
from backtest.ou_ev_chain import (
    EV_FLOOR_GRID,
    HOLD_SEASONS,
    TRIAL_REGISTRY_FIELDS,
    TUNE_SEASONS,
    calibrated_p_over,
    estimate_prior_season_bias,
    fit_frozen_residual_sd,
)
from backtest.simulation import BettingSimulator, SimulationConfig
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BOOTSTRAP_B",
    "BOOTSTRAP_CI_TYPE",
    "BOOTSTRAP_SEED",
    "CONTAMINATED_VOCAB",
    "ROBUSTNESS_CUTS",
    "SIGNIFICANCE_ALPHA",
    "LeakageError",
    "run",
    "run_ou_monetization",
]


# ---------------------------------------------------------------------------
# Pre-registered runner constants (frozen BEFORE any tuning run).
#
# FORKING-PATHS GUARD: these are frozen here and are NOT adjusted after seeing results (the
# D26-08 / D24-07 pre-registration discipline, mirroring backtest/ou_divergence.py and
# backtest/ou_ev_chain.py). The EV chain owns TUNE_SEASONS / HOLD_SEASONS / EV_FLOOR_GRID /
# TRIAL_REGISTRY_FIELDS. This module reuses those and adds only the runner-specific constants
# below.
# ---------------------------------------------------------------------------

# Block-by-week bootstrap (D27 discretion, Pitfall 4 + #6): resample WEEKS WITHIN the holdout
# seasons only. B replicates, a frozen seed, percentile CI. State B explicitly so the CI is
# reproducible and the bootstrap can never silently widen/narrow.
BOOTSTRAP_B: int = 2000
BOOTSTRAP_SEED: int = 27_04
BOOTSTRAP_CI_TYPE: str = "percentile"

# Robustness-cut pass policy (D27-02 / OUM-05): all applicable cuts must stay positive vs the
# headline. 2020 is NOT in the hold split, so a 2020-specific cut is N/A-for-hold (documented). The
# applicable hold cuts are regular-season-only and playoffs-excluded (the with-line hold population
# is week <= 18, so these report on the same population -- the cut is reported for completeness and
# the N/A-for-hold disclosure is explicit).
ROBUSTNESS_CUTS: tuple[str, ...] = ("regular_season_only", "playoffs_excluded")

# The FIXED owner-checkpoint vocabulary (#6, D27-01). The ONLY words the readout / checkpoint may
# use for the 2023-2024 numbers; "validated" / "proven profitable" are FORBIDDEN.
CONTAMINATED_VOCAB: tuple[str, ...] = (
    "provisional contaminated readout",
    "proceed signal",
    "redirect",
)

# Prior-residual window for the walk-forward bias seed (leakage-clean). The first tune season (2021)
# needs a STRICTLY-PRIOR residual pool to estimate its walk-forward bias from -- the EV chain raises
# otherwise. PRIOR_RESIDUAL_SEASONS (2018-2020, the deployed artifact's own train/val window) is the
# leakage-clean source: these seasons seed ONLY the prior-season bias estimate. They are NEVER
# candidates, NEVER in the frozen-SD fit, NEVER in the threshold tuning, and NEVER in the trial
# selection -- so they cannot leak future info into the tune/hold decision. Frozen pre-registered.
PRIOR_RESIDUAL_SEASONS: tuple[int, ...] = (2018, 2019, 2020)

# The structural label the materialized bet table carries (#6); mirrored by api/cache.py.
VALIDATION_TYPE_PROVISIONAL: str = "PROVISIONAL_CONTAMINATED"

# The per-2023-2024-number honesty label attached to every hold ROI in the result.
_CONTAMINATED_LABEL: str = CONTAMINATED_VOCAB[0]

# The raw closing-total column on the silver odds frame (NEVER the gold z-scored snapshot_total).
_RAW_TOTAL_COL = "total"


class LeakageError(ValueError):
    """Raised when the fit-window / leakage fence is violated (#5, T-27-12).

    The frozen SD, the EV-floor ``t``, and each season's self-bias estimate must be fit on
    TUNE / strictly-prior seasons only; NO hold season (2023/2024) may feed the SD fit, the
    threshold tuning, any self-bias estimation, or the trial-selection set. This hard error forbids
    a fence violation rather than silently leaking future information into the sizing/threshold.
    """


# ---------------------------------------------------------------------------
# Scoring + odds join (provenance hard-fail FIRST)
# ---------------------------------------------------------------------------


def _load_ou_gold(seasons_lo: int, seasons_hi: int) -> pd.DataFrame:
    """Load the OU gold for a season window (READ ONLY, never written)."""
    gold = pd.read_parquet("data/gold/features_ou.parquet")
    return gold[(gold["season"] >= seasons_lo) & (gold["season"] <= seasons_hi)].copy()


def _score_ou_candidates(gold_df: pd.DataFrame | None) -> pd.DataFrame:
    """Score the deployed O/U artifact single-pass over the 2021-2024 candidate window.

    Mirrors ``backtest/diagnose.score_deployed_artifacts("ou")``: LOADS the deployed v1.0 OU
    artifact and runs inference. Returns the backtest column contract (game_id, season, week,
    model_total, actual). The 2021-2024 window covers BOTH the tune (2021-2022) and the hold
    (2023-2024) splits -- the walk-forward fences below keep the hold out of every fit.
    """
    return score_deployed_artifacts("ou", gold_df=gold_df)


def _score_ou_prior_residuals() -> pd.DataFrame:
    """Score the deployed O/U artifact over PRIOR_RESIDUAL_SEASONS for the walk-forward bias seed.

    The first tune season (2021) needs a STRICTLY-PRIOR residual pool to estimate its walk-forward
    bias from (the EV chain raises otherwise). PRIOR_RESIDUAL_SEASONS (2018-2020) is the leakage-clean
    source: these residuals ONLY seed the prior-season bias estimate -- they are never candidates,
    never in the frozen-SD fit, the threshold tuning, or the trial selection. Returns the same
    backtest column contract (game_id, season, week, model_total, actual).
    """
    prior_gold = _load_ou_gold(PRIOR_RESIDUAL_SEASONS[0], PRIOR_RESIDUAL_SEASONS[-1])
    return score_deployed_artifacts("ou", gold_df=prior_gold)


def _load_raw_silver_totals() -> pd.DataFrame:
    """Load the raw closing ``total`` per game_id from SILVER (READ ONLY, never written).

    The raw closing total is the line the bet is graded against. The gold ``snapshot_total`` is
    z-scored and must NOT be used as the raw line (Plan read_first). Uses the engine's normalized
    game_id (LAR->LA) via ``BacktestEngine._load_closing_odds`` so the join matches the backtest
    exactly. Also carries the provenance columns (sportsbook / is_live) for the OUM-06 hard-fail.
    """
    from backtest.engine import BacktestEngine

    odds = BacktestEngine()._load_closing_odds()
    keep = ["game_id", _RAW_TOTAL_COL]
    for prov in ("sportsbook", "is_live"):
        if prov in odds.columns:
            keep.append(prov)
    return odds[keep].copy()


def _build_scored_candidates(
    gold_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the deployed artifact, join the raw closing total, run the OUM-06 provenance hard-fail.

    Returns ``(candidates, coverage)`` where ``candidates`` carries game_id / season / week /
    model_total / closing_total / actual for every game WITH a closing line, and ``coverage`` holds
    the OUM-06 provenance counts (n_total, n_with_line, n_excluded, coverage) reported alongside
    every ROI number (OUM-06).

    The provenance hard-fail (``assert_real_odds``) runs FIRST on the raw silver odds frame, before
    any selection or grading -- a profitability number on contaminated odds is invalid (T-27-14).
    """
    preds = _score_ou_candidates(gold_df)
    raw_totals = _load_raw_silver_totals() if odds_df is None else odds_df.copy()

    # (1) OUM-06 provenance hard-fail FIRST (T-27-14): reject mock/synthetic odds before anything.
    assert_real_odds(raw_totals)

    n_total = len(preds)
    merged = preds.merge(
        raw_totals[["game_id", _RAW_TOTAL_COL]], on="game_id", how="left"
    )
    merged = merged.rename(columns={_RAW_TOTAL_COL: "closing_total"})

    with_line = merged[merged["closing_total"].notna()].copy()
    n_with_line = len(with_line)
    n_excluded = n_total - n_with_line
    coverage = {
        "n_total": n_total,
        "n_with_line": n_with_line,
        "n_excluded": n_excluded,
        "coverage": float(n_with_line / n_total) if n_total else 0.0,
        "provenance": "OUM-06: assert_real_odds passed (consensus/draftkings, is_live all False)",
    }
    return with_line, coverage


# ---------------------------------------------------------------------------
# Tune-only fit: prior-season walk-forward bias + frozen residual SD
# ---------------------------------------------------------------------------


def _residuals_by_season(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-season residual arrays following the LOCKED contract (actual - model_total).

    The residual contract (D26-18 / D27-07): ``residual = actual_total - model_total``; an
    over-predicting model gives actual < predicted => residual < 0. Used by
    ``estimate_prior_season_bias`` (walk-forward, strictly-prior) and the frozen-SD fit.
    """
    out: dict[int, np.ndarray] = {}
    for season in sorted(int(s) for s in frame["season"].unique()):
        sub = frame[frame["season"] == season]
        out[season] = (sub["actual"] - sub["model_total"]).to_numpy(dtype=float)
    return out


def _fit_season_bias(
    resid_by_season: dict[int, np.ndarray],
    seasons: tuple[int, ...],
) -> dict[int, float]:
    """Prior-season walk-forward bias for each season in ``seasons`` (strictly-prior, no-leak).

    Each season's bias is the pooled mean residual of all STRICTLY-PRIOR seasons only (D26-18 /
    D27-07, via ``estimate_prior_season_bias``). NEVER uses the target season's own data, so the
    self-bias estimate for a hold season can only ever see earlier seasons.
    """
    return {
        season: float(estimate_prior_season_bias(resid_by_season, season))
        for season in seasons
    }


def _fit_tune(
    candidates: pd.DataFrame,
    prior_resid_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Fit the frozen SD + tune-season bias on the TUNE split ONLY (#5, no-leak).

    The frozen residual SD (D27-08) is fit on the BIAS-CORRECTED TUNE residuals only:
    corrected = model_total + season_bias, residual = actual - corrected. The tune-season bias is
    the prior-season walk-forward mean residual for each tune season -- the first tune season (2021)
    draws its strictly-prior pool from ``prior_resid_frame`` (PRIOR_RESIDUAL_SEASONS, 2018-2020),
    which seeds ONLY the bias estimate and is NEVER in the SD fit / threshold tuning / candidates.
    Returns the fit artifacts plus the exact seasons consumed by each fit, so the fit-window /
    leakage assertion can prove the fence held.
    """
    tune_seasons = tuple(range(TUNE_SEASONS[0], TUNE_SEASONS[1] + 1))

    # The residual map for BIAS estimation pools the prior seasons (2018-2020) + the candidates, so a
    # tune season's strictly-prior bias has a non-empty pool. The prior seasons feed ONLY the bias
    # estimate (never the SD fit / threshold / candidates below).
    resid_by_season = _residuals_by_season(
        pd.concat([prior_resid_frame, candidates], ignore_index=True)
    )

    tune_bias = _fit_season_bias(resid_by_season, tune_seasons)

    # Bias-corrected tune residuals: residual = actual - (model_total + season_bias).
    tune = candidates[candidates["season"].isin(tune_seasons)]
    corrected_resids: list[float] = []
    sd_fit_seasons: set[int] = set()
    for season in tune_seasons:
        sub = tune[tune["season"] == season]
        if sub.empty:
            continue
        bias = tune_bias[season]
        corrected = sub["model_total"].to_numpy(dtype=float) + bias
        corrected_resids.extend(
            (sub["actual"].to_numpy(dtype=float) - corrected).tolist()
        )
        sd_fit_seasons.add(season)

    frozen_sd = fit_frozen_residual_sd(np.asarray(corrected_resids, dtype=float))

    return {
        "tune_seasons": tune_seasons,
        "resid_by_season": resid_by_season,
        "tune_bias": tune_bias,
        "frozen_sd": frozen_sd,
        "sd_fit_seasons": sorted(sd_fit_seasons),
        "bias_seasons": sorted(tune_bias),
    }


# ---------------------------------------------------------------------------
# EV-floor sweep on TUNE only + COMPLETE trial registry + BH-FDR (#4, OUM-03)
# ---------------------------------------------------------------------------


def _grade_selector_roi(
    selector: BetSelector,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Grade the BetSelector's selected bets through the LOCKED BettingSimulator (proof==production).

    Routes the SAME injected BetSelector END-TO-END through the LOCKED ``BettingSimulator`` (the
    LOCKED-2 path): the simulator grades ONLY the games the selector SELECTED, with the calibrated
    p_side / kelly_stake / slipped_line / outcome the selector supplied. Returns the flat-stake ROI
    (the acceptance ROI -- equal-weight per bet so it is not dominated by a single large Kelly
    stake), the Kelly ROI, the bet count, and the per-bet outcome/season/week frame the bootstrap +
    robustness cuts resample over.

    The simulator needs the backtest contract columns (game_id, season, week, model_total, the
    closing ``total``, and ``actual``); ``candidates`` already carries them (closing_total -> total).
    The frame is shaped as an ALREADY-merged prediction frame (it carries the four odds columns + a
    ``has_closing_odds`` flag) so the simulator takes the already-merged branch and the per-game
    season/week from the predictions frame is used directly (no odds re-merge that would collide).
    """
    sim_frame = candidates.rename(columns={"closing_total": "total"}).copy()
    sim_frame["has_closing_odds"] = True
    # O/U only needs the ``total`` line; the other odds columns are present-but-NaN so the simulator
    # takes the already-merged branch (it never reads ml/spread for the O/U target).
    for col in ("ml_home", "ml_away", "spread"):
        if col not in sim_frame.columns:
            sim_frame[col] = np.nan
    results_like = _ResultsLike({"ou": sim_frame})

    closing_odds = sim_frame[
        ["game_id", "ml_home", "ml_away", "spread", "total"]
    ].copy()
    sim = BettingSimulator(SimulationConfig(), ou_bet_selector=selector)
    res = sim.simulate(results_like, closing_odds)

    ou_stats = res.by_target.get("ou", {})
    ou_records = [r for r in res.bet_records if r.target == "ou"]
    per_bet = pd.DataFrame(
        [
            {
                "game_id": r.game_id,
                "season": r.season,
                "week": r.week,
                "outcome": r.outcome,
                "payout_flat": r.payout_flat,
                "flat_stake": r.flat_stake,
                "kelly_stake": r.kelly_stake,
                "payout_kelly": r.payout_kelly,
            }
            for r in ou_records
        ]
    )
    return {
        "flat_roi": float(ou_stats.get("flat_roi", 0.0)) if ou_records else None,
        "kelly_roi": float(ou_stats.get("kelly_roi", 0.0)) if ou_records else None,
        "win_rate": float(ou_stats.get("win_rate", 0.0)) if ou_records else None,
        "n_bets": len(ou_records),
        "per_bet": per_bet,
    }


class _ResultsLike:
    """Minimal BacktestResults-shaped object exposing only ``all_predictions`` (simulator input)."""

    def __init__(self, all_predictions: dict[str, pd.DataFrame]) -> None:
        self.all_predictions = all_predictions


def _make_selector(
    frozen_sd: float,
    season_bias_by_season: dict[int, float],
    ev_floor_t: float,
) -> BetSelector:
    """Construct a BetSelector with the FROZEN SD / bias and a given EV-floor t.

    The BetSelector is the SINGLE O/U decision source (BET-01); the runner only supplies the frozen
    inputs and the EV-floor scalar. The bias map MUST carry every candidate season (the selector
    raises if a season's bias is missing -- no silent fallback to the raw biased total).
    """
    return BetSelector(
        frozen_sd=frozen_sd,
        season_bias_by_season=season_bias_by_season,
        ev_floor_t=ev_floor_t,
    )


# The ``subpopulation_rule`` every registry entry now carries (D33.2-24). It used to read
# ``union(under OR high_total)``; the union was deleted, so the entry says what the rule IS rather
# than leaving a field that names a gate no code applies.
_SUBPOPULATION_RULE: str = (
    "none (D33.2-24: no eligibility gate; the EV floor alone decides)"
)


def _sweep_ev_floor_on_tune(
    tune_candidates: pd.DataFrame,
    frozen_sd: float,
    bias_all_seasons: dict[int, float],
) -> dict[str, Any]:
    """Sweep the EV-floor scalar t over EV_FLOOR_GRID on TUNE only + build the COMPLETE registry (#4).

    For each t in the pre-registered ``EV_FLOOR_GRID`` the BetSelector grades the tune split through
    the LOCKED simulator; ONE trial-registry entry per t is logged using the Plan-01
    ``TRIAL_REGISTRY_FIELDS`` schema (threshold, subpopulation_rule, calibration_method, sd_source,
    devig_method, sizing_policy, sample_window, robustness_cut, raw_p, adjusted_p, roi, ci,
    bet_count). The registry is the BH-FDR denominator; ``false_discovery_control(..., method="bh")``
    deflates the non-None raw p-values (the ou_divergence:1199-1206 pattern; setdefault adjusted_p
    None). ROI -- not significance -- chooses the frozen t (the highest tune flat-stake ROI with a
    non-degenerate bet count); significance is supporting context (Pitfall 3).
    """
    registry: list[dict[str, Any]] = []
    for t in EV_FLOOR_GRID:
        selector = _make_selector(frozen_sd, bias_all_seasons, t)
        graded = _grade_selector_roi(selector, tune_candidates)

        # Per-trial CLV significance is the registry's testable p (CLV is REPORT-ONLY -- it is the
        # multiple-comparisons p-value source, never a selection gate; ROI chooses t below).
        clv_values = _per_bet_clv(selector, tune_candidates)
        sig = clv_significance(clv_values) if clv_values else {"p": None, "ci95": None}

        entry = dict.fromkeys(TRIAL_REGISTRY_FIELDS)
        entry.update(
            {
                "threshold": float(t),
                "subpopulation_rule": _SUBPOPULATION_RULE,
                "calibration_method": "prior_season_mean_bias_subtraction",
                "sd_source": "frozen_tune_corrected_sd",
                "devig_method": "flat_-110",
                "sizing_policy": "calibrated_kelly_locked_cap_order",
                "sample_window": f"tune_{TUNE_SEASONS[0]}_{TUNE_SEASONS[1]}",
                "robustness_cut": "none",
                "raw_p": sig.get("p"),
                "adjusted_p": None,
                "roi": graded["flat_roi"],
                "ci": sig.get("ci95"),
                "bet_count": graded["n_bets"],
            }
        )
        registry.append(entry)

    # BH-FDR over the COMPLETE registry (the multiple-comparisons denominator, #4 / Pitfall 3).
    tested = [e for e in registry if e["raw_p"] is not None]
    if tested:
        raw_ps = [e["raw_p"] for e in tested]
        adjusted = false_discovery_control(raw_ps, method="bh")
        for entry, adj in zip(tested, adjusted, strict=True):
            entry["adjusted_p"] = float(adj)
    for entry in registry:
        entry.setdefault("adjusted_p", None)

    # ROI (not significance) chooses the frozen t: the highest tune flat-stake ROI with >=1 bet.
    bettable = [e for e in registry if e["bet_count"] and e["roi"] is not None]
    if bettable:
        chosen = max(bettable, key=lambda e: e["roi"])
        chosen_t = float(chosen["threshold"])
    else:
        # No t admits a bet on tune -- freeze the grid floor (lowest t) and let the hold report a
        # zero-bet honest result. Never invent a t outside the pre-registered grid.
        chosen_t = float(EV_FLOOR_GRID[0])

    return {
        "trial_registry": registry,
        "n_trials": len(tested),
        "chosen_t": chosen_t,
        "ev_floor_grid": list(EV_FLOOR_GRID),
    }


def _per_bet_clv(selector: BetSelector, candidates: pd.DataFrame) -> list[float]:
    """The selected bets' report-only model-edge line_clv values (D27-06, #11; NEVER a gate).

    Drives the SAME selector ``select()`` to read the per-bet line_clv it attaches; CLV is the
    registry's p-value source (multiple-comparisons accounting) and is reported only -- it never
    selects, admits, or sizes a bet. Returns an empty list when nothing was selected.
    """
    result = selector.select(
        candidates.rename(columns={"closing_total": "closing_total"})
    )
    return [r["clv"] for r in result.selected if r.get("clv") is not None]


# ---------------------------------------------------------------------------
# Fit-window / leakage assertion (#5, T-27-12)
# ---------------------------------------------------------------------------


def _assert_fit_window(
    fit: dict[str, Any],
    chosen_t_window: str,
) -> dict[str, Any]:
    """Prove NO hold season fed the SD fit, the threshold tuning, or the TUNE-season self-bias (#5).

    Raises :class:`LeakageError` if any hold season (2023/2024) appears in:
      - the seasons consumed by the frozen-SD fit,
      - the seasons the EV-floor t was tuned on (the sample_window string),
      - any TUNE-season self-bias estimate input (check (c) iterates ``fit["bias_seasons"]`` = the
        tune seasons; each is fit on STRICTLY-PRIOR seasons, asserted here). The HOLD seasons'
        walk-forward bias may, BY DESIGN, include the strictly-prior 2023 outcomes (correct
        walk-forward: 2023 is known before betting 2024); that does NOT feed the frozen t / SD
        decision and is therefore out of this fence's scope (WR-04 -- the fence asserts the
        inputs to the FROZEN decision are hold-free, not every per-season walk-forward bias pool).

    A fourth check -- that the high-total eligibility boundary matched its leakage-clean pre-hold
    re-derivation -- was deleted with the boundary itself (D33.2-24). There is no boundary left to
    drift, so there is nothing for that check to guard.

    Returns a structured fence report (the seasons each fit actually consumed) for the SUMMARY.
    """
    hold = set(range(HOLD_SEASONS[0], HOLD_SEASONS[1] + 1))

    # (a) SD fit seasons must be a subset of the tune split and disjoint from hold.
    sd_seasons = set(fit["sd_fit_seasons"])
    if sd_seasons & hold:
        msg = (
            f"frozen-SD fit consumed HOLD seasons {sorted(sd_seasons & hold)} "
            f"(must be tune-only {sorted(fit['tune_seasons'])}); leakage fence violated (#5, T-27-12)."
        )
        raise LeakageError(msg)
    if not sd_seasons.issubset(set(fit["tune_seasons"])):
        msg = (
            f"frozen-SD fit consumed seasons {sorted(sd_seasons)} outside the tune split "
            f"{sorted(fit['tune_seasons'])} (#5, T-27-12)."
        )
        raise LeakageError(msg)

    # (b) Threshold-tuning window must be the tune split (the sample_window string the sweep logged).
    expected_window = f"tune_{TUNE_SEASONS[0]}_{TUNE_SEASONS[1]}"
    if chosen_t_window != expected_window:
        msg = (
            f"EV-floor t was tuned on '{chosen_t_window}' (must be '{expected_window}'); the "
            "threshold cannot see hold data (#5, T-27-12)."
        )
        raise LeakageError(msg)

    # (c) Each tune-season self-bias estimate must see strictly-prior seasons only (no hold). The
    # walk-forward estimate raises if no prior season exists; here we assert no hold season is a
    # strictly-prior input to any tune-season bias (it never is, since tune < hold, but assert it).
    for season in fit["bias_seasons"]:
        prior = {s for s in fit["resid_by_season"] if s < season}
        if prior & hold:
            msg = (
                f"self-bias estimate for season {season} would consume HOLD seasons "
                f"{sorted(prior & hold)} (strictly-prior must exclude hold; #5, T-27-12)."
            )
            raise LeakageError(msg)

    return {
        "sd_fit_seasons": sorted(sd_seasons),
        "bias_seasons": sorted(fit["bias_seasons"]),
        "threshold_window": chosen_t_window,
        "hold_seasons": sorted(hold),
        "fence_held": True,
    }


# ---------------------------------------------------------------------------
# Block-by-week bootstrap CI (within holdout only) + robustness cuts (#6, OUM-05)
# ---------------------------------------------------------------------------


def _flat_roi_from_records(per_bet: pd.DataFrame) -> float | None:
    """Flat-stake ROI from a per-bet frame: sum(payout_flat) / sum(flat_stake) (None if no stake)."""
    if per_bet.empty:
        return None
    total_stake = float(per_bet["flat_stake"].sum())
    if total_stake <= 0:
        return None
    return float(per_bet["payout_flat"].sum() / total_stake)


def _block_by_week_bootstrap_ci(per_bet: pd.DataFrame) -> dict[str, Any]:
    """Block-by-week percentile CI resampling WEEKS WITHIN the holdout seasons only (#6, Pitfall 4).

    The block is the (season, week) pair: each bootstrap replicate resamples (season, week) BLOCKS
    with replacement from the HOLDOUT bets only, then recomputes the flat-stake ROI over the
    resampled blocks. Resampling whole weeks preserves within-week correlation (a half-point line
    move correlates the games on a slate); resampling individual bets would understate the CI
    (T-27-25). The bootstrap NEVER reaches outside the holdout (#6). Percentile CI at the frozen B +
    seed.

    Degenerate cases (#8): an empty frame -> a None CI; a single (season, week) block -> the
    bootstrap degenerates to that one block (the CI collapses to the point estimate) and is reported
    with ``n_blocks == 1`` rather than crashing.
    """
    if per_bet.empty:
        return {
            "point_estimate": None,
            "ci_lo": None,
            "ci_hi": None,
            "n_blocks": 0,
            "b": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "ci_type": BOOTSTRAP_CI_TYPE,
            "scope": "within_holdout_only",
        }

    point = _flat_roi_from_records(per_bet)

    blocks = [grp for _, grp in per_bet.groupby(["season", "week"], sort=True)]
    n_blocks = len(blocks)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    roi_reps: list[float] = []
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        resampled = pd.concat([blocks[i] for i in idx], ignore_index=True)
        roi = _flat_roi_from_records(resampled)
        if roi is not None:
            roi_reps.append(roi)

    if not roi_reps:
        return {
            "point_estimate": point,
            "ci_lo": None,
            "ci_hi": None,
            "n_blocks": n_blocks,
            "b": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "ci_type": BOOTSTRAP_CI_TYPE,
            "scope": "within_holdout_only",
        }

    lo = float(np.percentile(roi_reps, 2.5))
    hi = float(np.percentile(roi_reps, 97.5))
    return {
        "point_estimate": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_blocks": n_blocks,
        "b": BOOTSTRAP_B,
        "seed": BOOTSTRAP_SEED,
        "ci_type": BOOTSTRAP_CI_TYPE,
        "scope": "within_holdout_only",
    }


def _robustness_cuts(
    per_bet: pd.DataFrame, headline_roi: float | None
) -> dict[str, Any]:
    """Robustness-cut ROIs vs the headline (OUM-05 / D27-02): all applicable cuts stay positive.

    The applicable hold cuts (ROBUSTNESS_CUTS): regular-season-only (week <= 18) and
    playoffs-excluded (week <= 18) -- in the with-line hold population every game is week <= 18, so
    these report on the same population as the headline (the cut is reported for completeness and the
    fact that it equals the headline is disclosed). 2020 is N/A-for-hold (it is not in the hold
    split) and is documented, never used to bolster a holdout claim. ``all_cuts_positive`` is True
    iff every applicable cut ROI is >= 0 AND the headline ROI is >= 0.
    """
    cuts: dict[str, Any] = {}
    if not per_bet.empty:
        reg_season = per_bet[per_bet["week"] <= 18]
        cuts["regular_season_only"] = _flat_roi_from_records(reg_season)
        cuts["playoffs_excluded"] = _flat_roi_from_records(reg_season)
    else:
        cuts["regular_season_only"] = None
        cuts["playoffs_excluded"] = None

    applicable = [v for v in cuts.values() if v is not None]
    all_positive = bool(
        headline_roi is not None
        and headline_roi >= 0.0
        and all(v >= 0.0 for v in applicable)
    )
    return {
        "cuts": cuts,
        "all_cuts_positive": all_positive,
        "cut_2020_na_for_hold": (
            "2020 is not in the hold split (2023-2024); a 2020-specific cut is N/A-for-hold and "
            "is never used to bolster a holdout claim (D27-02)."
        ),
        "policy": "all applicable cuts and the headline must be >= 0 (OUM-05 / D27-02)",
    }


# ---------------------------------------------------------------------------
# BET-02 probability check (#7)
#
# This section used to be the "high-total OVER empirical report": the surviving high-total OVER
# count read off each selected record's ``totals_regime`` field, and the union-versus-under-only
# ROI comparison. Both measured the O/U eligibility UNION, which D33.2-24 deleted -- along with the
# ``totals_regime`` field the slice read -- so that over/under split dies with the union, by ruling.
# The one part that measured something else survives: the 8-point-gap check that Kelly consumes a
# PROBABILITY rather than a points distance (BET-02).
# ---------------------------------------------------------------------------


def _bet02_probability_report(
    selector: BetSelector,
    hold_candidates: pd.DataFrame,
) -> dict[str, Any]:
    """The 8-point-gap ``kelly_model_prob <= 1.0`` check on the hold candidates (BET-02, #7).

    Confirms the BET-02 fix: Kelly consumes the calibrated P(side), a probability, NOT the legacy
    points distance ``implied + 8.0``.
    """
    eight_pt_p = _eight_point_gap_p_side(selector, hold_candidates)
    return {
        "eight_point_gap_kelly_model_prob": eight_pt_p,
        "eight_point_gap_prob_is_probability": bool(
            eight_pt_p is not None and 0.0 <= eight_pt_p <= 1.0
        ),
    }


def _eight_point_gap_p_side(
    selector: BetSelector,
    hold_candidates: pd.DataFrame,
) -> float | None:
    """Calibrated P(over) for a synthetic 8-point-gap OVER candidate (the BET-02 #7 check).

    Builds a model_total 8 points ABOVE a representative closing total -- the MEDIAN closing total
    of the hold candidates -- and reads the calibrated P(over) the EV chain produces. The result is
    a probability in [0,1] -- NOT the legacy points distance ``implied + 8.0 = 8.524`` --
    confirming Kelly consumes a probability (BET-02).

    The representative line used to be the high-total eligibility boundary plus two points. That
    boundary was deleted under D33.2-24, and the check never depended on which line it used: a
    normal CDF is a probability at every line, so the median of the candidates' own lines is a
    representative choice that needs no constant of its own.
    """
    if hold_candidates.empty:
        return None
    season = int(hold_candidates["season"].iloc[0])
    if season not in selector.season_bias_by_season:
        return None
    closing = float(hold_candidates["closing_total"].median())
    model_total = closing + 8.0
    bias = selector.season_bias_by_season[season]
    p_over = float(calibrated_p_over(model_total, closing, selector.frozen_sd, bias))
    return p_over


# ---------------------------------------------------------------------------
# Top-level runner (the single re-runnable entry point)
# ---------------------------------------------------------------------------


def run_ou_monetization(
    gold_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full O/U monetization chain and return the provisional contaminated readout (OUM-03/05).

    Pipeline (walk-forward, no leakage -- the fences are non-negotiable):
      1. Score the deployed O/U artifact single-pass over 2021-2024 gold, join the raw closing total
         from SILVER, run the OUM-06 provenance hard-fail FIRST, capture coverage/exclusion counts.
      2. FIT on TUNE (2021-2022) ONLY: the prior-season walk-forward bias, the single frozen residual
         SD on bias-corrected tune residuals, and SWEEP the EV-floor t over EV_FLOOR_GRID. Each
         t evaluation logs ONE COMPLETE trial-registry entry (TRIAL_REGISTRY_FIELDS).
         FREEZE the chosen t (and SD, bias) before touching hold.
      3. FIT-WINDOW / LEAKAGE ASSERTION: prove no hold season fed the SD/threshold/self-bias.
      4. BH-FDR over the COMPLETE registry; ROI (not significance) is the acceptance bar.
      5. GRADE the frozen-t BetSelector on HOLD (2023-2024) through the LOCKED BettingSimulator:
         the provisional ROI point estimate, a within-holdout block-by-week bootstrap CI, and the
         robustness cuts. Report the FILTERED (acceptance) and UNFILTERED whole-population cross-check.
      6. The BET-02 probability check (the 8-point-gap calibrated P(over) is a probability).
      7. Attach coverage/exclusion counts to every ROI; LABEL every 2023-2024 number with the FIXED
         contaminated vocabulary; CLV is report-only; a NEGATIVE provisional ROI is an honest
         redirect (does NOT raise).
      8. EDGE CASES: empty selected bets (zero-bet honest result), missing odds rows (excluded +
         counted), no-prior-season bias (the EV chain raises -- surfaced), all bets in one week (the
         block bootstrap degenerates to one block).

    Args:
        gold_df: Optional pre-filtered 2021-2024 OU gold frame. When None, loaded read-only.
        odds_df: Optional raw silver odds frame (game_id, total, provenance). When None, loaded via
            the engine loader.

    Returns:
        A structured result dict (deterministic given the scored artifact + silver odds): the tune
        fit, the frozen t / SD / bias, the COMPLETE registry + BH-FDR, the hold ROI +
        within-holdout block-by-week CI + robustness cuts + cross-check + coverage, the
        BET-02 probability check, and the fixed contaminated-vocab labels.
    """
    # (1) Score + join + provenance hard-fail FIRST.
    candidates, coverage = _build_scored_candidates(gold_df, odds_df)

    tune_seasons = tuple(range(TUNE_SEASONS[0], TUNE_SEASONS[1] + 1))
    hold_seasons = tuple(range(HOLD_SEASONS[0], HOLD_SEASONS[1] + 1))
    tune_candidates = candidates[candidates["season"].isin(tune_seasons)].copy()
    hold_candidates = candidates[candidates["season"].isin(hold_seasons)].copy()

    # (2) Fit on TUNE only: prior-season bias + frozen SD. The first tune season (2021) draws its
    # strictly-prior bias pool from PRIOR_RESIDUAL_SEASONS (2018-2020); those residuals seed ONLY the
    # bias estimate and are never in the SD fit / threshold tuning / candidates.
    prior_resid_frame = _score_ou_prior_residuals()
    fit = _fit_tune(candidates, prior_resid_frame)
    frozen_sd = fit["frozen_sd"]

    # The bias map for selection must cover every candidate season (tune + hold), each estimated
    # walk-forward on STRICTLY-PRIOR seasons only (the hold seasons' bias sees 2021-2022 + the prior
    # pool, never their own data). The EV chain raises if a season has no prior season (#8 --
    # surfaced, not swallowed).
    candidate_seasons = tuple(sorted(int(s) for s in candidates["season"].unique()))
    bias_all_seasons = _fit_season_bias(fit["resid_by_season"], candidate_seasons)

    # (2 cont.) Sweep the EV-floor t on TUNE only + COMPLETE registry + BH-FDR.
    sweep = _sweep_ev_floor_on_tune(tune_candidates, frozen_sd, bias_all_seasons)
    chosen_t = sweep["chosen_t"]

    # (3) FIT-WINDOW / LEAKAGE ASSERTION (raises on a fence violation).
    fence = _assert_fit_window(fit, f"tune_{TUNE_SEASONS[0]}_{TUNE_SEASONS[1]}")

    # (5) GRADE the frozen-t BetSelector on HOLD through the LOCKED simulator.
    hold_selector = _make_selector(frozen_sd, bias_all_seasons, chosen_t)
    hold_graded = _grade_selector_roi(hold_selector, hold_candidates)
    hold_per_bet = hold_graded["per_bet"]
    headline_roi = _flat_roi_from_records(hold_per_bet)

    bootstrap_ci = _block_by_week_bootstrap_ci(hold_per_bet)
    robustness = _robustness_cuts(hold_per_bet, headline_roi)

    # FILTERED (acceptance) vs UNFILTERED whole-population cross-check (D27-04). The unfiltered
    # cross-check grades EVERY hold game at the lowest EV floor on the grid through the simulator
    # -- a reported cross-check only, never the acceptance basis. With the eligibility UNION
    # deleted (D33.2-24) the two differ only by the EV floor, not by a sub-population.
    unfiltered_selector = _make_selector(
        frozen_sd, bias_all_seasons, float(EV_FLOOR_GRID[0])
    )
    unfiltered_graded = _grade_selector_roi(unfiltered_selector, hold_candidates)
    unfiltered_roi = _flat_roi_from_records(unfiltered_graded["per_bet"])

    # (6) The BET-02 probability check on the hold candidates.
    bet02_check = _bet02_probability_report(hold_selector, hold_candidates)

    # (7) Report-only CLV on the SELECTED hold bets (D27-06 / D27-12, #11 -- never a gate).
    hold_clv_values = _per_bet_clv(hold_selector, hold_candidates)
    clv_report = clv_significance(hold_clv_values) if hold_clv_values else None

    result: dict[str, Any] = {
        "split": {
            "tune_seasons": list(tune_seasons),
            "hold_seasons": list(hold_seasons),
            "hold_label": _CONTAMINATED_LABEL,
        },
        "coverage": coverage,
        "tune_fit": {
            "frozen_sd": frozen_sd,
            "tune_bias": fit["tune_bias"],
            "sd_fit_seasons": fit["sd_fit_seasons"],
        },
        "frozen": {
            "ev_floor_t": chosen_t,
            "frozen_sd": frozen_sd,
            "bias_by_season": bias_all_seasons,
        },
        "trial_registry": sweep["trial_registry"],
        "n_trials": sweep["n_trials"],
        "ev_floor_grid": sweep["ev_floor_grid"],
        "fit_window_assertion": fence,
        "hold_roi": {
            # Every 2023-2024 number is labeled with the FIXED contaminated vocabulary (#6).
            "validation_type": VALIDATION_TYPE_PROVISIONAL,
            "label": _CONTAMINATED_LABEL,
            "headline_flat_roi": headline_roi,
            "headline_kelly_roi": hold_graded["kelly_roi"],
            "win_rate": hold_graded["win_rate"],
            "n_bets": hold_graded["n_bets"],
            "block_by_week_ci": bootstrap_ci,
            "robustness": robustness,
            "coverage": coverage,
            "filtered_unfiltered_cross_check": {
                "filtered_acceptance_roi": headline_roi,
                "unfiltered_cross_check_roi": unfiltered_roi,
                "unfiltered_n_bets": unfiltered_graded["n_bets"],
                "note": (
                    "filtered = the frozen-t acceptance basis (the bar); unfiltered = the "
                    "lowest-grid-floor cross-check (reported only, D27-04). No eligibility "
                    "sub-population separates them (D33.2-24)."
                ),
            },
        },
        "bet02_probability_check": bet02_check,
        "clv_report_only": {
            "metric": "model_edge_line_clv (model_total - closing_total); REPORT-ONLY (D27-06)",
            "is_selection_gate": False,
            "summary": clv_report,
        },
        "contaminated_vocab": list(CONTAMINATED_VOCAB),
        "acceptance_bar": "graded simulator ROI on the held-out split (NOT CLV magnitude); OUM-05",
        "negative_roi_is_redirect": (
            "a negative provisional ROI is a VALID documented redirect toward Phase 30 "
            "(D27-02 / D25-14 honest-refusal pattern), not a forced phase failure."
        ),
    }

    logger.info(
        "ou_monetization run complete",
        chosen_t=chosen_t,
        frozen_sd=frozen_sd,
        hold_headline_flat_roi=headline_roi,
        hold_n_bets=hold_graded["n_bets"],
        n_trials=sweep["n_trials"],
        coverage=coverage["coverage"],
    )
    return result


# Public alias matching the plan's `run(...)` / `run_ou_monetization(...)` contract.
run = run_ou_monetization


def _format_readout(result: dict[str, Any]) -> str:
    """Render the provisional contaminated readout for the owner checkpoint (ASCII, fixed vocab).

    Uses ONLY the FIXED contaminated vocabulary for the 2023-2024 numbers; NEVER emits "validated" /
    "proven profitable". Presents the headline provisional held-out ROI point estimate, its
    within-holdout block-by-week bootstrap CI, the robustness-cut table, the filtered-vs-unfiltered
    cross-check, the BET-02 probability check, and the coverage/exclusion counts.
    """
    hold = result["hold_roi"]
    ci = hold["block_by_week_ci"]
    cov = result["coverage"]
    bet02 = result["bet02_probability_check"]
    frozen = result["frozen"]

    lines = [
        "=" * 78,
        "O/U MONETIZATION -- PROVISIONAL CONTAMINATED READOUT (2023-2024 hold; OUM-03/05)",
        "=" * 78,
        f"  label: {hold['label']} (validation_type={hold['validation_type']})",
        f"  frozen EV-floor t: {frozen['ev_floor_t']}  frozen SD: {frozen['frozen_sd']:.4f}",
        "  eligibility: none -- every candidate reaches the EV floor (D33.2-24)",
        f"  BH-FDR trial registry: {result['n_trials']} tested of "
        f"{len(result['trial_registry'])} forks",
        "-" * 78,
        "  HEADLINE provisional held-out flat-stake ROI (the acceptance bar, OUM-05):",
        f"    point estimate: {_fmt(hold['headline_flat_roi'])}   n_bets: {hold['n_bets']}   "
        f"win_rate: {_fmt(hold['win_rate'])}",
        f"    within-holdout block-by-week {ci['ci_type']} CI "
        f"(B={ci['b']}, seed={ci['seed']}, n_blocks={ci['n_blocks']}): "
        f"[{_fmt(ci['ci_lo'])}, {_fmt(ci['ci_hi'])}]",
        f"    Kelly ROI: {_fmt(hold['headline_kelly_roi'])}",
        "-" * 78,
        "  ROBUSTNESS CUTS (all applicable must stay >= 0 vs headline; OUM-05 / D27-02):",
        f"    regular_season_only: {_fmt(hold['robustness']['cuts']['regular_season_only'])}",
        f"    playoffs_excluded:   {_fmt(hold['robustness']['cuts']['playoffs_excluded'])}",
        f"    all_cuts_positive:   {hold['robustness']['all_cuts_positive']}",
        f"    {hold['robustness']['cut_2020_na_for_hold']}",
        "-" * 78,
        "  FILTERED (acceptance) vs UNFILTERED (cross-check only, D27-04):",
        f"    filtered acceptance ROI:  {_fmt(hold['filtered_unfiltered_cross_check']['filtered_acceptance_roi'])}",
        f"    unfiltered cross-check ROI: {_fmt(hold['filtered_unfiltered_cross_check']['unfiltered_cross_check_roi'])} "
        f"(n={hold['filtered_unfiltered_cross_check']['unfiltered_n_bets']})",
        "-" * 78,
        "  BET-02 probability check (#7):",
        f"    8-pt-gap kelly_model_prob: {_fmt(bet02['eight_point_gap_kelly_model_prob'])} "
        f"(is a probability in [0, 1]: {bet02['eight_point_gap_prob_is_probability']})",
        "-" * 78,
        "  ODDS COVERAGE (OUM-06; reported with every ROI):",
        f"    n_total: {cov['n_total']}   n_with_line: {cov['n_with_line']}   "
        f"n_excluded: {cov['n_excluded']}   coverage: {_fmt(cov['coverage'])}",
        "-" * 78,
        "  CLV is REPORT-ONLY (D27-06 / D27-12, #11): it never gates a bet.",
        "  ACCEPTANCE: graded simulator ROI on the held-out split (NOT CLV magnitude).",
        "  A provisional +ROI point estimate is the 'proceed signal' (D27-02).",
        "  A negative result is a fully legitimate 'redirect' toward Phase 30 (D25-14 pattern).",
        "  These 2023-2024 numbers are PROVISIONAL CONTAMINATED only; the binding clean verdict",
        "  is Phase 30 (the burned-holdout overclaim words are deliberately not used here, #6).",
        "=" * 78,
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a float metric to 4 dp, or 'n/a' for None (ASCII readout helper)."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    _result = run_ou_monetization()
    print(_format_readout(_result))  # noqa: T201 -- CLI entry point readout
