"""Thin DIAG harness for the honest accuracy diagnosis (Phase 22, DIAG-01..05).

This is the "thin entry where none fits" allowance (carried D-13): ``BacktestEngine`` has no
external-artifact injection point, so the DIAG-05 deployed-artifact scoring (D-07/D-08) lives in
this harness that CALLS existing ``backtest/`` + ``models/`` functions on the canonical
Phase-20-rebuilt gold. The harness adds only TWO genuinely-new measurements -- the all-games
straight-pick hit-rate (D-01 population a, via the simulator at ``min_edge_threshold=0.0``) and the
per-game CLV t-stat/p-value/95% CI (Pitfall 1, via ``scipy.stats.ttest_1samp``) -- and reuses
everything else unchanged. The hit-rate uses ONLY the LOCKED BettingSimulator sign convention --
never the simplified, non-line-graded heuristics in ``backtest/metrics.py`` (D-01).

HARD BOUNDARY (carried D-01, the milestone's namesake): this module imports NO ``train_*`` module
and NEVER writes ``data/gold/``. The deployed-artifact path LOADS artifacts via
``load_model_artifact`` and runs inference only -- NO re-fit, NO gold rebuild. The
``BacktestEngine`` itself fits FRESH per-fold models internally (inference-for-evaluation, the
established backtest behavior); that is invoked only via ``run_diagnosis(run_backtest_half=True)``
and is distinct from a deployed-artifact re-fit.

Public API:
  - ``score_deployed_artifacts(target, gold_df=None, artifacts_dir="artifacts")`` -- DIAG-05 raw
    deployed-artifact scoring in the BACKTEST column contract, mirroring
    ``scripts.generate_current_week_predictions.run_predictions`` EXACTLY.
  - ``both_population_hit_rates(results_like, closing_odds_df)`` -- DIAG-01 straight-pick (0.0) +
    edge-filtered (0.02) hit-rates via the BettingSimulator honest convention, with the gap.
  - ``clv_significance(clv_values)`` -- DIAG-03 {n, mean, t, p, ci95} via ttest_1samp.
  - ``clv_verdict(clv_values)`` -- DIAG-04 significance-gated CLV verdict string.
  - ``apply_blended_cut(preds, closing_odds_df, target, artifacts_dir="artifacts")`` -- D-02/D-08
    blended stream mirroring the engine's blend-then-recompute-CLV path.
  - ``run_diagnosis(...)`` -- assembles the DIAG-05 2x2 (raw/blended x prod/backtest) + DIAG-01/02
    + per-target CLV significance into a structured dict.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from backtest.metrics import compute_wp_metrics
from backtest.simulation import BettingSimulator, SimulationConfig
from models.artifacts import load_model_artifact
from models.blending import MarketBlender
from models.clv import compute_clv_for_predictions
from utils import get_logger

if TYPE_CHECKING:
    from backtest.engine import BacktestResults

logger = get_logger(__name__)

# The DIAG harness's holdout window (walk-forward, 2021-2024).
#
# FROZEN BY DESIGN, and NOT the live partition (review WR-14). The comment here used to
# read "Matches BacktestConfig defaults", which stopped being true when Phase 33.1 moved
# BacktestConfig onto conf/season_partition.py: the live holdout is now the two most
# recent completed seasons. These two constants are deliberately NOT moved with it --
# they fence a PUBLISHED measurement (the Phase-22 diagnosis, and every consumer that
# calls score_deployed_artifacts without passing its own gold_df), and
# scripts/audit_odds_preingest.py asserts this exact pair by name before measuring, so
# changing them here would silently re-window a pre-registered result.
#
# Every gate and promotion caller passes gold_df explicitly and is unaffected by these.
# The frozen window is stated ONCE, as a window (Plan 33.2-17, SPEC R10): a `*_FIRST_SEASON`
# name assigned a year literal reads as a COVERAGE FLOOR, and every coverage floor lives in
# conf/season_partition.py alone. This pair is NOT a floor -- it is a pinned measurement span
# -- so its literal lives under a name that says so and the two bounds derive from it. The
# values are unchanged.
HOLDOUT_WINDOW_SEASONS: tuple[int, int] = (2021, 2024)
HOLDOUT_FIRST_SEASON = HOLDOUT_WINDOW_SEASONS[0]
HOLDOUT_LAST_SEASON = HOLDOUT_WINDOW_SEASONS[1]

# CLV significance: require a reasonable sample (mirrors clv_tracking.py:561).
MIN_CLV_SAMPLE = 10

# DIAG-04 verdict gate (D-05): only call CLV "systematically negative" when mean < 0 AND p < 0.05.
SIGNIFICANCE_ALPHA = 0.05

# Per-target gold label columns (the backtest "actual" contract).
_LABEL_COLUMN = {"wp": "home_win", "ats": "home_margin", "ou": "total_points"}

# Per-target CLV column used for the DIAG-03 significance test (Pitfall 2):
#   WP -> probability_clv (devigged); ATS/OU -> line_clv (spread/total units).
CLV_COLUMN_FOR = {"wp": "probability_clv", "ats": "line_clv", "ou": "line_clv"}


# ---------------------------------------------------------------------------
# (1) DIAG-05: deployed-artifact scoring (mirror run_predictions EXACTLY)
# ---------------------------------------------------------------------------


def _load_gold_holdout(target: str) -> pd.DataFrame:
    """Load the 2021-2024 gold holdout for a target (READ ONLY, never written)."""
    gold = pd.read_parquet(f"data/gold/features_{target}.parquet")
    return gold[
        (gold["season"] >= HOLDOUT_FIRST_SEASON)
        & (gold["season"] <= HOLDOUT_LAST_SEASON)
    ].copy()


def score_deployed_artifacts(
    target: str,
    gold_df: pd.DataFrame | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> pd.DataFrame:
    """Score a deployed v1.0 artifact over 2021-2024 gold (DIAG-05 raw-prod cut).

    Mirrors ``scripts.generate_current_week_predictions.run_predictions`` EXACTLY:
    ``load_model_artifact`` -> ``X = gold[feature_list]`` (unscaled) -> for WP
    ``model.predict_proba(X)[:, 1]`` then ``calibrator.predict(raw)`` (Pitfall 4 -- NO external
    scaler); for ATS/OU ``model.predict(X)``. Produces the BACKTEST column contract so the result
    feeds the existing CLV / simulator / metrics functions unchanged.

    Args:
        target: One of "wp", "ats", "ou".
        gold_df: Optional pre-filtered 2021-2024 gold frame. When None, loads from disk
            (read-only). Passing the shared fixture frame avoids a redundant parquet read.
        artifacts_dir: Root directory for the deployed model artifacts.

    Returns:
        DataFrame with columns: ``game_id``, ``season``, ``week``, ``model_prob``, ``actual``,
        plus ``model_spread`` (ATS) or ``model_total`` (OU).

    Raises:
        KeyError: If a required feature is absent from the gold matrix (should not happen on
            canonical gold -- verified 0 missing for all three targets).
    """
    artifact = load_model_artifact(target, artifacts_dir=Path(artifacts_dir))
    model = artifact["model"]
    feature_list = artifact["feature_list"]
    calibrator = artifact["calibrator"]

    gold = gold_df if gold_df is not None else _load_gold_holdout(target)

    missing = [f for f in feature_list if f not in gold.columns]
    if missing:
        msg = (
            f"Missing features in gold matrix for target '{target}': "
            f"{missing[:10]} (n_missing={len(missing)})"
        )
        raise KeyError(msg)

    features = gold[feature_list]
    out = gold[["game_id", "season", "week"]].copy().reset_index(drop=True)

    label_col = _LABEL_COLUMN[target]
    if target == "wp":
        raw_probs = model.predict_proba(features)[:, 1]
        # Production convention: UNSCALED features, then isotonic/Platt calibrator (Pitfall 4).
        out["model_prob"] = (
            calibrator.predict(raw_probs) if calibrator is not None else raw_probs
        )
    elif target == "ats":
        margins = model.predict(features)
        # The backtest contract expects model_prob to carry the raw margin for ATS.
        out["model_prob"] = margins
        out["model_spread"] = margins
    elif target == "ou":
        totals = model.predict(features)
        out["model_prob"] = totals
        out["model_total"] = totals
    else:
        msg = f"Unknown target: '{target}'. Must be 'wp', 'ats', or 'ou'."
        raise ValueError(msg)

    out["actual"] = gold[label_col].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Lightweight BacktestResults shim for the simulator/CLV consumers
# ---------------------------------------------------------------------------


def _results_like(predictions: dict[str, pd.DataFrame]) -> BacktestResults:
    """Wrap a {target -> predictions} dict in a BacktestResults-shaped object.

    The BettingSimulator only reads ``.all_predictions``; this shim avoids re-running the engine
    when only the deployed-artifact predictions are needed.
    """
    from backtest.engine import BacktestConfig, BacktestResults

    return BacktestResults(
        config=BacktestConfig(),
        season_results=[],
        all_predictions=predictions,
        all_clv={},
        headline_clv={},
        odds_coverage={},
        covid_annotation={},
        era_info={},
        is_blended=False,
    )


# ---------------------------------------------------------------------------
# (2) DIAG-01: both-population hit-rate via the simulator honest convention
# ---------------------------------------------------------------------------


def both_population_hit_rates(
    results_like: BacktestResults,
    closing_odds_df: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Compute both DIAG-01 hit-rate populations per target via the BettingSimulator.

    Population (a) straight_pick: ``min_edge_threshold=0.0`` -- every game graded against the
    line. Population (b) edge_filtered: the canonical default ``min_edge_threshold=0.02``. The
    ``straight_pick - edge_filtered`` gap is itself the D-01 reportable finding. Uses the LOCKED
    simulator sign convention only -- never the simplified, non-line-graded metrics.py heuristics.

    Args:
        results_like: A BacktestResults (or shim) exposing ``.all_predictions``.
        closing_odds_df: Normalized closing odds (game_id, ml_home, ml_away, spread, total).

    Returns:
        Dict per target with ``straight_pick``, ``edge_filtered``, and ``gap`` win-rates plus the
        companion ``n_bets_*`` counts. A win-rate field is ``None`` when that population graded
        ZERO bets (the simulator omits a zero-bet target from ``by_target``), which is distinct
        from a 0.0 (0%) win-rate. ``gap`` is ``None`` whenever either side is ``None`` (no graded
        bets to compare). With the canonical n~1087-per-target population every target grades
        well over the minimum, so these fields are non-None in practice.
    """
    sim_all = BettingSimulator(SimulationConfig(min_edge_threshold=0.0))
    res_all = sim_all.simulate(results_like, closing_odds_df)

    sim_edge = BettingSimulator(SimulationConfig(min_edge_threshold=0.02))
    res_edge = sim_edge.simulate(results_like, closing_odds_df)

    def _win_rate(by_target: dict[str, Any], target: str) -> float | None:
        """Win-rate for a target, or None when it graded no bets ("no bets" != "0% wins")."""
        stats_for_target = by_target.get(target, {})
        if not stats_for_target.get("n_bets", 0):
            return None
        return float(stats_for_target["win_rate"])

    out: dict[str, dict[str, float | None]] = {}
    targets = set(res_all.by_target) | set(res_edge.by_target)
    for target in targets:
        straight = _win_rate(res_all.by_target, target)
        edge = _win_rate(res_edge.by_target, target)
        gap = straight - edge if straight is not None and edge is not None else None
        out[target] = {
            "straight_pick": straight,
            "edge_filtered": edge,
            "gap": gap,
            "n_bets_straight_pick": int(
                res_all.by_target.get(target, {}).get("n_bets", 0)
            ),
            "n_bets_edge_filtered": int(
                res_edge.by_target.get(target, {}).get("n_bets", 0)
            ),
        }
    return out


# ---------------------------------------------------------------------------
# (3) DIAG-03: per-game CLV significance (t-stat / p-value / 95% CI)
# ---------------------------------------------------------------------------


def clv_significance(clv_values: Any) -> dict[str, Any]:
    """Compute CLV significance + 95% CI from a per-game CLV array (D-05).

    Mirrors the ``clv_tracking.py`` ``ttest_1samp(clv, 0)`` idiom (Pitfall 1) and adds the 95% CI
    as ``mean +/- t.ppf(0.975, n-1) * std(ddof=1) / sqrt(n)``. NaNs are dropped. Below
    ``MIN_CLV_SAMPLE`` games the test returns None for t/p/ci95 (insufficient sample).

    Args:
        clv_values: 1-D array-like of per-game CLV values (probability_clv for WP, line_clv for
            ATS/OU), filtered to games with closing odds by the caller.

    Returns:
        Dict with ``n``, ``mean``, ``t``, ``p``, ``ci95`` (a (lo, hi) tuple or None).
    """
    arr = np.asarray(clv_values, dtype=float)
    clv = arr[~np.isnan(arr)]
    n = len(clv)

    if n < MIN_CLV_SAMPLE:
        return {
            "n": n,
            "mean": float(np.mean(clv)) if n else None,
            "t": None,
            "p": None,
            "ci95": None,
        }

    t_stat, p_value = stats.ttest_1samp(clv, 0.0)
    mean = float(np.mean(clv))
    se = float(np.std(clv, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": mean,
        "t": float(t_stat),
        "p": float(p_value),
        "ci95": (mean - half, mean + half),
    }


# ---------------------------------------------------------------------------
# (4) DIAG-04: significance-gated CLV verdict
# ---------------------------------------------------------------------------


def clv_verdict(clv_values: Any) -> str:
    """Render the D-05 significance-gated CLV verdict for a per-game CLV array.

    A negative CLV is only called a "systematically negative" methodology concern when it is
    statistically distinguishable from zero (mean < 0 AND p < 0.05). A non-significant small
    negative (or positive) CLV reads "indistinguishable from zero" -- consistent with an
    efficient-market ceiling.

    Args:
        clv_values: 1-D array-like of per-game CLV values.

    Returns:
        A human-readable verdict string (ASCII only).
    """
    sig = clv_significance(clv_values)
    if sig["t"] is None:
        return (
            f"Insufficient sample (n={sig['n']}) to test CLV significance; "
            "indistinguishable from zero."
        )

    mean = sig["mean"]
    p = sig["p"]
    if mean < 0 and p < SIGNIFICANCE_ALPHA:
        return (
            f"Systematically negative CLV (mean={mean:.5f}, p={p:.4f}): statistically "
            "distinguishable from zero -- a methodology concern."
        )
    if mean > 0 and p < SIGNIFICANCE_ALPHA:
        return (
            f"Systematically positive CLV (mean={mean:.5f}, p={p:.4f}): statistically "
            "distinguishable from zero -- a real edge."
        )
    return (
        f"CLV indistinguishable from zero (mean={mean:.5f}, p={p:.4f}): "
        "consistent with an efficient-market ceiling."
    )


# ---------------------------------------------------------------------------
# (5) D-02 / D-08: blended cut (mirror the engine blend-then-recompute path)
# ---------------------------------------------------------------------------

# CLV/odds columns dropped before recomputing CLV on a blended frame (mirrors engine.py:450-459).
_CLV_ODDS_COLS = [
    "probability_clv",
    "fair_closing_prob",
    "has_closing_odds",
    "line_clv",
    "ml_home",
    "ml_away",
    "spread",
    "total",
]


def apply_blended_cut(
    preds: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    target: str,
    artifacts_dir: str | Path = "artifacts",
    silver_dir: str | Path = "data/silver",
) -> pd.DataFrame:
    """Produce the market-blended cut for a target (D-02/D-08), mirroring the engine's blend.

    Loads the deployed blend via ``MarketBlender.from_artifacts`` (one fixed weight per target
    since D33.2-10; the retired dynamic schedule is refused at load), blends the predictions
    through ``models.blending_data.blend_historical_predictions``, then DROPS any existing
    CLV/odds columns and recomputes CLV on the clean blended frame so the blended cut is
    symmetric with the engine's blended path.

    THE WP MARKET SIDE IS OUT OF FOLD (A33.2-review WR-04). These are HISTORICAL games, and
    the serving blend converts a spread with the bound serving slope -- fitted partly on these
    same games' outcomes -- over whatever line the caller passed, which here is the CLOSING
    line. Each WP game is instead blended against its OWNED pre-lock spread converted with its
    own season's prior-only slope, the column the blend weight was tuned on. A game with no
    owned pre-lock line (or in a season with no prior-fold slope) has no honest market opinion
    and is left out of the WP cut, counted in the log. The closing line is still what CLV is
    GRADED against -- grading is its legitimate use.

    ATS and O/U blend the market line in *closing_odds_df*; a game with no line is left out
    of the cut rather than kept unblended in a blended column.

    Args:
        preds: Raw predictions frame in the backtest contract (from ``score_deployed_artifacts``
            or the engine). Must carry the target's model column.
        closing_odds_df: Normalized closing odds (the ATS/O/U line source and the CLV grade).
        target: One of "wp", "ats", "ou".
        artifacts_dir: Root directory for the blend artifact and its bound converter.
        silver_dir: Silver root holding the owned ``odds_timeline`` (WP market side).

    Returns:
        The blended predictions frame with recomputed CLV columns.
    """
    from models.blending_data import blend_historical_predictions

    # Strip any pre-existing CLV/odds columns so blend_predictions does not produce *_market
    # suffixed duplicates (the engine's all_predictions frames already carry merged odds).
    pre_drop = [c for c in _CLV_ODDS_COLS if c in preds.columns]
    base = preds.drop(columns=pre_drop) if pre_drop else preds

    blender = MarketBlender.from_artifacts(Path(artifacts_dir))
    blended = blend_historical_predictions(
        blender,
        base,
        closing_odds_df,
        target,
        artifacts_dir=Path(artifacts_dir),
        silver_dir=Path(silver_dir),
    )

    drop_cols = [c for c in _CLV_ODDS_COLS if c in blended.columns]
    clean = blended.drop(columns=drop_cols)
    return compute_clv_for_predictions(clean, closing_odds_df, target)


def _headline_clv(clv_df: pd.DataFrame) -> float | None:
    """Mean probability_clv over games with closing odds (mirrors engine.py:483)."""
    if clv_df.empty or "probability_clv" not in clv_df.columns:
        return None
    valid = clv_df[clv_df["has_closing_odds"] == True]  # noqa: E712
    if valid.empty:
        return None
    return float(valid["probability_clv"].mean())


# ---------------------------------------------------------------------------
# Per-target measurement bundle (reused for both the prod and backtest halves)
# ---------------------------------------------------------------------------


def _measure_target(
    target: str,
    raw_preds: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    """Compute the full per-target DIAG bundle (raw + blended) for one prediction stream.

    Returns a dict with pooled accuracy (WP), calibration (WP ECE/Brier), headline CLV + the
    significance bundle on the right CLV column, and the blended-cut headline CLV.

    The engine's ``all_predictions`` frames already carry merged CLV/odds columns; those are
    dropped before recomputing CLV so the left-merge in ``compute_clv_for_predictions`` does not
    duplicate odds columns (mirrors the engine's blend-then-recompute discipline).
    """
    raw_drop = [c for c in _CLV_ODDS_COLS if c in raw_preds.columns]
    raw_base = raw_preds.drop(columns=raw_drop) if raw_drop else raw_preds
    raw_clv = compute_clv_for_predictions(raw_base, closing_odds_df, target)
    valid_raw = raw_clv[raw_clv["has_closing_odds"]]
    clv_col = CLV_COLUMN_FOR[target]
    raw_sig = clv_significance(valid_raw[clv_col].to_numpy())

    blended_clv = apply_blended_cut(
        raw_preds, closing_odds_df, target, artifacts_dir=artifacts_dir
    )
    valid_blended = blended_clv[blended_clv["has_closing_odds"]]
    blended_sig = clv_significance(valid_blended[clv_col].to_numpy())

    bundle: dict[str, Any] = {
        "n_games": len(raw_preds),
        "headline_clv": _headline_clv(raw_clv),
        "clv_significance_raw": raw_sig,
        "clv_verdict_raw": clv_verdict(valid_raw[clv_col].to_numpy()),
        "blended_headline_clv": _headline_clv(blended_clv),
        "clv_significance_blended": blended_sig,
    }

    if target == "wp":
        wp_metrics = compute_wp_metrics(
            raw_preds["actual"].to_numpy(), raw_preds["model_prob"].to_numpy()
        )
        bundle["pooled_accuracy"] = wp_metrics["accuracy"]
        bundle["ece"] = wp_metrics["ece"]
        bundle["brier_score"] = wp_metrics["brier_score"]
        bundle["brier_reliability"] = wp_metrics["brier_reliability"]
        bundle["brier_resolution"] = wp_metrics["brier_resolution"]

    return bundle


# ---------------------------------------------------------------------------
# Backtest-half predictions (the engine fits fresh per-fold models internally)
# ---------------------------------------------------------------------------


def _backtest_predictions(targets: list[str]) -> dict[str, pd.DataFrame]:
    """Run the walk-forward engine and return per-target predictions in the backtest contract.

    The engine fits FRESH per-fold models internally (inference-for-evaluation, the established
    backtest behavior) -- this is NOT a deployed-artifact re-fit and writes no gold. The returned
    frames carry the trainer's ``model_prob`` / ``model_spread`` / ``model_total`` and ``actual``.
    """
    from backtest.engine import BacktestConfig, BacktestEngine

    engine = BacktestEngine(BacktestConfig(targets=targets))
    results = engine.run()
    out: dict[str, pd.DataFrame] = {}
    for target in targets:
        preds = results.all_predictions.get(target, pd.DataFrame())
        out[target] = preds.copy()
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator: the DIAG-05 2x2 + DIAG-01/02/03
# ---------------------------------------------------------------------------


def run_diagnosis(
    gold: dict[str, pd.DataFrame] | None = None,
    odds: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    run_backtest_half: bool = True,
) -> dict[str, Any]:
    """Assemble the DIAG-01/02/03 measurements and the DIAG-05 2x2 into a structured dict.

    The 2x2 (D-08): raw-prod / blended-prod (deployed artifacts) and raw-backtest /
    blended-backtest (engine per-fold models). Each cut reuses the SAME CLV / simulator / metrics
    functions; the only new measurements are the both-population hit-rate (D-01) and the CLV
    significance bundle (D-05). NO model re-fit and NO ``data/gold/`` write occurs here.

    Args:
        gold: Optional {target -> 2021-2024 gold frame}. When None, loaded read-only from disk.
        odds: Optional normalized closing odds. When None, loaded via the engine loader.
        targets: Targets to diagnose. Defaults to ["wp", "ats", "ou"].
        artifacts_dir: Root directory for the deployed model + blend artifacts.
        run_backtest_half: When True, run the walk-forward engine for the backtest half of the
            2x2. When False, only the deployed-artifact (production) half is computed.

    Returns:
        Dict with ``production`` (per-target deployed-artifact bundle), ``hit_rates`` (DIAG-01
        both-population), and -- when ``run_backtest_half`` -- ``backtest`` (per-target engine
        bundle). Each per-target bundle carries pooled accuracy (WP), calibration, headline CLV,
        and the significance + verdict bundle for raw and blended cuts.
    """
    targets = targets or ["wp", "ats", "ou"]

    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()

    # -- Production half: deployed-artifact scoring (DIAG-05 raw-prod + blended-prod) --
    prod_preds: dict[str, pd.DataFrame] = {}
    for target in targets:
        gold_df = gold.get(target) if gold is not None else None
        prod_preds[target] = score_deployed_artifacts(
            target, gold_df=gold_df, artifacts_dir=artifacts_dir
        )

    production: dict[str, Any] = {}
    for target in targets:
        production[target] = _measure_target(
            target, prod_preds[target], odds, artifacts_dir
        )

    # -- DIAG-01 both-population hit-rate on the deployed-artifact stream --
    hit_rates = both_population_hit_rates(_results_like(prod_preds), odds)

    diagnosis: dict[str, Any] = {
        "production": production,
        "hit_rates": hit_rates,
    }

    # -- Backtest half: engine per-fold models (raw-backtest + blended-backtest) --
    if run_backtest_half:
        backtest_preds = _backtest_predictions(targets)
        backtest: dict[str, Any] = {}
        for target in targets:
            preds = backtest_preds[target]
            if preds.empty:
                backtest[target] = {"n_games": 0}
                continue
            backtest[target] = _measure_target(target, preds, odds, artifacts_dir)
        diagnosis["backtest"] = backtest
        diagnosis["backtest_hit_rates"] = both_population_hit_rates(
            _results_like(backtest_preds), odds
        )

    return diagnosis
