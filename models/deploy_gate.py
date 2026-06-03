"""The single, frozen, significance-tested per-target model deploy gate (Phase 24).

This module is THE JUDGE for the per-target deploy decision (ACTV-01, ACTV-02). It
decides, per target, whether a freshly-fit candidate model may replace the production
model in ``artifacts/latest.json``. It is import-parity-locked to ``backtest/diagnose.py``
(D24-13): it IMPORTS the per-target CLV column map (``CLV_COLUMN_FOR``), the significance
constant (``SIGNIFICANCE_ALPHA``), and the t-test (``clv_significance``) directly from the
diagnosis, and NEVER re-declares them. The gate and the honest diagnosis therefore cannot
silently diverge onto different metrics (Pitfall 2 -- the gate must judge on the SAME CLV
column the diagnosis reports: ``probability_clv`` for WP, ``line_clv`` for ATS/OU).

Why a HARD block: in v2.0 an ungated swap shipped a regression -- the deployed v1.0 pre-Elo
models were retained only because per-target gating caught the regression after the fact
(the D-17 precedent). v3.0 deliberately re-fits (lifting the v2.1 D-01 boundary), so this
gate is the safety rail: nothing ships that regresses. The frozen thresholds + baseline live
in the git-tracked ``config/gate.toml``; a threshold change is its own reviewed config edit.

The CLV floor (D24-01) is the LOGICAL COMPLEMENT of ``diagnose.clv_verdict``'s
"systematically negative" branch (``mean < 0 and p < SIGNIFICANCE_ALPHA``): a target PASSES
the floor unless its per-game CLV is significantly negative. A non-significant negative (or
any positive) CLV passes -- consistent with an efficient-market ceiling rather than a
methodology leak. Per-season-must-pass (D24-04) applies this same floor to every holdout
season individually so one lucky season cannot carry a model whose other seasons are
significantly negative. The secondary non-regression gates (accuracy/MAE, and for WP the
calibration ECE/Brier per D24-05) are evaluated POOLED (Open Question A: ~270-game per-season
secondary slices fire on sampling noise; per-season secondary deltas are a readout, not a
blocker). diagnose.py is pooled-only, so per-season CLV slicing is the one genuinely-new
piece here -- it reuses the season-agnostic ``clv_significance`` on per-season array slices.

The candidate/baseline bundle contract (the output of ``build_candidate_bundle``):

    {
        "clv_values": np.ndarray,          # per-game CLV on CLV_COLUMN_FOR[target]
        "mean": float | None,              # pooled CLV mean (clv_significance)
        "t": float | None,                 # pooled CLV t-stat
        "p": float | None,                 # pooled CLV two-sided p-value
        "per_season": {int: {...}, ...},   # {season -> clv_significance bundle}
        # WP only:
        "accuracy": float, "ece": float, "brier_score": float,
        # ATS/OU only:
        "mae": float,
    }

``build_candidate_bundle`` is the SINGLE source of this bundle shape. ``scripts/promote_models.py``,
the rewired ``scripts/retrain_models.py``, and the gate tests MUST all construct candidate
bundles through it, so the three call sites cannot drift into subtly different bundle shapes
and the forced-FAIL/PASS tests have ONE well-defined monkeypatch seam (mirrors
``diagnose._measure_target``; because ``compute_clv_for_predictions`` RECOMPUTES the CLV column
from ``model_prob``, a pre-filled CLV column on the scored frame would be overwritten -- so the
bundle builder, not ``score_deployed_artifacts``, is the correct seam to force a CLV value).

The ``{passed, reasons, v1_metrics, v2_metrics}`` reason-dict shape is kept compatible with
``scripts.retrain_models.print_gating_summary`` so the Plan 24-04 rewire is drop-in.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Import-the-diagnosis parity seam (D24-13). These three symbols are IMPORTED, never
# re-declared here: the gate must judge on the exact same CLV column + significance test
# the honest diagnosis uses, or the two could silently diverge (Pitfall 2).
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from backtest.metrics import compute_wp_metrics
from models.clv import compute_clv_for_predictions
from utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Re-export the imported names so callers (and the parity test) can reference
# ``gate.CLV_COLUMN_FOR`` / ``gate.SIGNIFICANCE_ALPHA``. These are the SAME objects as in
# diagnose.py (identity, not copies): ``gate.CLV_COLUMN_FOR is diagnose.CLV_COLUMN_FOR``.
__all__ = [
    "CLV_COLUMN_FOR",
    "SIGNIFICANCE_ALPHA",
    "build_candidate_bundle",
    "clv_floor_passes",
    "evaluate_target",
    "load_gate_config",
    "per_season_clv",
    "validate_gate_config",
]

# The frozen holdout window (matches BacktestConfig + diagnose.py). per_season_clv slices
# the per-game CLV arrays by these seasons; load_gate_config asserts a populated baseline
# season table matches this set exactly.
HOLDOUT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)

# CLV/odds columns dropped from a scored frame before recomputing CLV, so the left-merge in
# compute_clv_for_predictions does not produce duplicate odds columns (mirrors the
# diagnose._measure_target "raw_drop" discipline at diagnose.py:416-417).
_CLV_ODDS_COLS = (
    "probability_clv",
    "fair_closing_prob",
    "has_closing_odds",
    "line_clv",
    "ml_home",
    "ml_away",
    "spread",
    "total",
)


# ---------------------------------------------------------------------------
# (1) The CLV floor: the logical complement of clv_verdict's negative branch
# ---------------------------------------------------------------------------


def clv_floor_passes(clv_values: Any, alpha: float = SIGNIFICANCE_ALPHA) -> bool:
    """Return True unless the CLV array is SIGNIFICANTLY NEGATIVE (the D24-01 floor).

    The logical complement of ``diagnose.clv_verdict``'s "systematically negative" branch
    (``mean < 0 and p < SIGNIFICANCE_ALPHA``). A target passes the floor when its per-game
    CLV is NOT significantly negative -- i.e. a non-significant negative, a zero, or any
    positive CLV all pass. Insufficient sample (below ``MIN_CLV_SAMPLE``, where the t-test
    cannot run) is a STRICT FAIL: the gate refuses to deploy on an untestable CLV.

    Args:
        clv_values: 1-D array-like of per-game CLV values (``probability_clv`` for WP,
            ``line_clv`` for ATS/OU), already filtered to games with closing odds.
        alpha: Significance level for the negative-tail test. Defaults to the imported
            ``SIGNIFICANCE_ALPHA`` (0.05) so the floor and the diagnosis share one alpha.

    Returns:
        True if the CLV is not significantly negative (passes the floor); False if it is
        significantly negative OR the sample is too small to test.
    """
    sig = clv_significance(clv_values)
    if sig["t"] is None:
        # n < MIN_CLV_SAMPLE: the t-test cannot run, so the CLV is untestable. A model whose
        # CLV cannot be shown non-negative does not clear the floor (strict fail).
        return False
    return not (sig["mean"] < 0 and sig["p"] < alpha)


# ---------------------------------------------------------------------------
# (2) Per-season CLV slicing (the one genuinely-new piece; diagnose.py is pooled-only)
# ---------------------------------------------------------------------------


def per_season_clv(
    valid_preds: pd.DataFrame,
    target: str,
    seasons: tuple[int, ...] = HOLDOUT_SEASONS,
) -> dict[int, dict[str, Any]]:
    """Slice per-game CLV by season and run ``clv_significance`` on each slice.

    diagnose.py computes CLV significance POOLED only; this reuses the same season-agnostic
    ``clv_significance`` on per-season slices of the per-game CLV array (the D24-04
    per-season-must-pass input). The returned keys are INTEGER seasons.

    Args:
        valid_preds: A predictions frame already filtered to ``has_closing_odds`` and
            carrying the integer ``season`` column plus the target's CLV column
            (``CLV_COLUMN_FOR[target]``) -- i.e. the post-merge frame from
            ``compute_clv_for_predictions``.
        target: One of "wp", "ats", "ou".
        seasons: Seasons to slice. Defaults to the 2021-2024 holdout.

    Returns:
        ``{season: clv_significance_bundle}`` with one entry per requested season (integer
        keys). A season with no rows yields a bundle with ``n == 0`` and ``t is None``.
    """
    col = CLV_COLUMN_FOR[target]
    return {
        int(s): clv_significance(
            valid_preds.loc[valid_preds["season"] == s, col].to_numpy()
        )
        for s in seasons
    }


# ---------------------------------------------------------------------------
# (3) The single bundle builder -- the ONE source of the evaluate_target input
# ---------------------------------------------------------------------------


def build_candidate_bundle(
    target: str,
    scored_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Turn a scored frame + closing odds + cfg into the canonical ``evaluate_target`` bundle.

    This mirrors ``diagnose._measure_target`` (diagnose.py:416-446) and is the SINGLE source
    of the bundle shape ``evaluate_target`` consumes. ``promote_models.py``, the rewired
    ``retrain_models.py``, and the gate tests all build candidate bundles through this one
    function, so the three call sites cannot drift into different bundle shapes and the
    forced-FAIL/PASS tests have one monkeypatch seam.

    Because ``compute_clv_for_predictions`` RECOMPUTES the CLV column from ``model_prob`` /
    ``model_spread`` / ``model_total``, any pre-existing CLV/odds columns on ``scored_df`` are
    dropped first (same discipline as ``_measure_target``'s ``raw_drop``) to avoid duplicate
    merged odds columns -- and a CLV value pre-baked onto ``scored_df`` would be overwritten,
    which is why this builder (not ``score_deployed_artifacts``) is the seam tests monkeypatch.

    Args:
        target: One of "wp", "ats", "ou".
        scored_df: Predictions in the backtest contract -- ``game_id``, ``season``,
            ``model_prob`` (+ ``model_spread`` for ATS / ``model_total`` for OU), ``actual``.
        odds_df: Normalized closing odds (``game_id``, ``ml_home``, ``ml_away``, and
            ``spread`` / ``total`` as needed by the target's line-CLV computation).
        cfg: The loaded gate config (currently unused for the bundle math; threaded for
            forward-compatibility and a uniform call signature across the three call sites).

    Returns:
        The candidate bundle (see the module docstring for the full contract): ``clv_values``,
        pooled ``mean`` / ``t`` / ``p``, ``per_season`` (int season keys), and either
        ``accuracy`` / ``ece`` / ``brier_score`` (WP) or ``mae`` (ATS/OU).
    """
    _ = cfg  # threaded for a uniform signature across promote / retrain / tests
    if target not in CLV_COLUMN_FOR:
        msg = f"Unknown target: '{target}'. Must be one of {sorted(CLV_COLUMN_FOR)}."
        raise ValueError(msg)

    # Drop any pre-existing CLV/odds columns before the recompute-merge (mirrors
    # diagnose._measure_target raw_drop) so compute_clv_for_predictions's left-merge does not
    # duplicate odds columns.
    pre_drop = [c for c in _CLV_ODDS_COLS if c in scored_df.columns]
    base = scored_df.drop(columns=pre_drop) if pre_drop else scored_df

    clv_df = compute_clv_for_predictions(base, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]

    col = CLV_COLUMN_FOR[target]
    clv_values = valid[col].to_numpy()
    pooled = clv_significance(clv_values)

    bundle: dict[str, Any] = {
        "clv_values": clv_values,
        "mean": pooled["mean"],
        "t": pooled["t"],
        "p": pooled["p"],
        "n": pooled["n"],
        "per_season": per_season_clv(valid, target),
    }

    if target == "wp":
        wp_metrics = compute_wp_metrics(
            valid["actual"].to_numpy(), valid["model_prob"].to_numpy()
        )
        bundle["accuracy"] = wp_metrics["accuracy"]
        bundle["ece"] = wp_metrics["ece"]
        bundle["brier_score"] = wp_metrics["brier_score"]
    else:
        # Regression MAE is measured against the EXPLICIT line column (margin for ATS,
        # total for OU), NEVER the overloaded "model_prob" (CR-01). model_prob is a
        # convention-only alias the three producers (score_deployed_artifacts, ats_trainer,
        # ou_trainer) happen to set equal to the line value today; if any future producer
        # set model_prob to a cover/over PROBABILITY (the column name literally says it is)
        # while leaving the line value in model_spread/model_total, a model_prob-based MAE
        # would compute mean(|margin - probability|) ~= the raw margin magnitude and compare
        # it to a ~9.5 baseline -- a meaningless pass/fail. Reading the line column and
        # asserting its presence makes the units explicit and the gate robust to that drift.
        line_col = "model_spread" if target == "ats" else "model_total"
        if line_col not in valid.columns:
            msg = (
                f"{target} candidate frame missing required '{line_col}' column for the "
                "regression MAE (the line value must be carried explicitly, not via model_prob)"
            )
            raise ValueError(msg)
        bundle["mae"] = float(
            np.mean(np.abs(valid["actual"].to_numpy() - valid[line_col].to_numpy()))
        )

    return bundle


# ---------------------------------------------------------------------------
# (4) Config loader + validator (git-tracked config/gate.toml, season-key int-normalized)
# ---------------------------------------------------------------------------


def load_gate_config(path: Path = Path("config/gate.toml")) -> dict[str, Any]:
    """Load ``config/gate.toml`` and int-normalize the baseline season keys.

    tomllib requires binary mode. Crucially, tomllib parses the dotted-integer season keys
    (``[baseline.wp.season.2021]``) as STRING keys ("2021"); this normalizes every
    ``baseline.<target>.season`` sub-table to INTEGER keys so the rest of the gate can index
    seasons as ints (matching ``per_season_clv``'s integer keys).

    Args:
        path: Path to the gate config. Defaults to the committed ``config/gate.toml``.

    Returns:
        The parsed config dict with baseline season keys normalized to int.
    """
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    for target_cfg in cfg.get("baseline", {}).values():
        season = target_cfg.get("season")
        if isinstance(season, dict):
            target_cfg["season"] = {int(k): v for k, v in season.items()}

    return cfg


def validate_gate_config(cfg: dict[str, Any]) -> None:
    """Validate the gate config structure, raising ``ValueError`` on a malformed config.

    Input-validation mitigation (T-24-INTEGRITY): a malformed or partial gate config could
    silently change the deploy decision, so the structure is checked explicitly. Required:
    ``gate.alpha``, the four ``gate.secondary`` tolerance keys, and the ``baseline.{wp,ats,ou}``
    TABLES. The baseline tables may be EMPTY in Wave 1 (Plan 24-02 ships empty baselines; Plan
    24-03 fills the numbers), so a present-but-empty ``baseline.<target>`` is NOT an error here.
    For any target whose ``baseline.<target>.season`` table IS populated, its (int-normalized)
    key set must equal the 2021-2024 holdout exactly.

    Args:
        cfg: A loaded gate config (ideally from ``load_gate_config`` so season keys are ints).

    Raises:
        ValueError: If a required key/table is missing, or a populated season table has the
            wrong key set.
    """
    gate = cfg.get("gate")
    if not isinstance(gate, dict):
        msg = "gate config missing required [gate] table"
        raise ValueError(msg)

    if "alpha" not in gate:
        msg = "gate config missing required key gate.alpha"
        raise ValueError(msg)

    secondary = gate.get("secondary")
    if not isinstance(secondary, dict):
        msg = "gate config missing required [gate.secondary] table"
        raise ValueError(msg)

    required_secondary = (
        "wp_accuracy_max_drop",
        "regression_mae_max_increase",
        "wp_ece_max_increase",
        "wp_brier_max_increase",
    )
    missing_secondary = [k for k in required_secondary if k not in secondary]
    if missing_secondary:
        msg = (
            "gate config missing required gate.secondary tolerance keys: "
            f"{missing_secondary}"
        )
        raise ValueError(msg)

    baseline = cfg.get("baseline")
    if not isinstance(baseline, dict):
        msg = "gate config missing required [baseline] table"
        raise ValueError(msg)

    for target in ("wp", "ats", "ou"):
        if target not in baseline:
            msg = f"gate config missing required baseline table baseline.{target}"
            raise ValueError(msg)
        # Tolerate an empty Wave-1 baseline table (numeric values are Plan 24-03's job).
        season = baseline[target].get("season")
        if isinstance(season, dict) and season:
            season_keys = {int(k) for k in season}
            if season_keys != set(HOLDOUT_SEASONS):
                msg = (
                    f"baseline.{target}.season has wrong holdout key set {sorted(season_keys)}; "
                    f"expected {sorted(HOLDOUT_SEASONS)}"
                )
                raise ValueError(msg)


# ---------------------------------------------------------------------------
# (5) The per-target deploy decision
# ---------------------------------------------------------------------------


def _secondary_reasons(
    target: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    secondary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply the POOLED secondary non-regression gates; return (passed, reasons).

    WP gates on accuracy (must not drop more than ``wp_accuracy_max_drop``); ATS/OU gate on
    MAE (must not increase more than ``regression_mae_max_increase``). A secondary check whose
    baseline value is absent is recorded as skipped and does NOT fail the target (the baseline
    is filled by Plan 24-03; Wave-1 synthetic baselines may omit a metric).
    """
    reasons: list[str] = []
    passed = True

    if target == "wp":
        cand_acc = candidate.get("accuracy")
        base_acc = baseline.get("accuracy")
        if cand_acc is None or base_acc is None:
            reasons.append("Accuracy comparison skipped (metric not available)")
        else:
            drop = base_acc - cand_acc
            max_drop = secondary["wp_accuracy_max_drop"]
            if drop > max_drop:
                passed = False
                reasons.append(
                    f"Accuracy dropped by {drop:.4f} (max allowed {max_drop})"
                )
            else:
                reasons.append(
                    f"Accuracy delta {cand_acc - base_acc:+.4f} (within {max_drop})"
                )
    else:
        cand_mae = candidate.get("mae")
        base_mae = baseline.get("mae")
        if cand_mae is None or base_mae is None:
            reasons.append("MAE comparison skipped (metric not available)")
        else:
            increase = cand_mae - base_mae
            max_increase = secondary["regression_mae_max_increase"]
            if increase > max_increase:
                passed = False
                reasons.append(
                    f"MAE increased by {increase:.4f} (max allowed {max_increase})"
                )
            else:
                reasons.append(f"MAE delta {increase:+.4f} (within {max_increase})")

    return passed, reasons


def _calibration_reasons(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    secondary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply the WP calibration non-regression gates (D24-05); return (passed, reasons).

    WP-only: ECE and Brier must not exceed the baseline by more than their tolerances. A
    missing baseline calibration metric is recorded as skipped (does not fail).
    """
    reasons: list[str] = []
    passed = True

    for metric, tol_key in (
        ("ece", "wp_ece_max_increase"),
        ("brier_score", "wp_brier_max_increase"),
    ):
        cand_val = candidate.get(metric)
        base_val = baseline.get(metric)
        if cand_val is None or base_val is None:
            reasons.append(f"{metric} comparison skipped (metric not available)")
            continue
        increase = cand_val - base_val
        max_increase = secondary[tol_key]
        if increase > max_increase:
            passed = False
            reasons.append(
                f"{metric} increased by {increase:.4f} (max allowed {max_increase})"
            )
        else:
            reasons.append(f"{metric} delta {increase:+.4f} (within {max_increase})")

    return passed, reasons


def evaluate_target(
    target: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether a candidate model may ship for one target (the per-target deploy gate).

    The decision is the AND of all applicable checks:
      1. POOLED CLV floor on ``candidate["clv_values"]`` (D24-01 -- not significantly negative).
      2. If ``gate.per_season_must_pass``: the CLV floor on EVERY season's slice in
         ``candidate["per_season"]`` (D24-04 -- a single significantly-negative season fails
         the target; an insufficient-sample season also fails per the strict floor rule).
      3. POOLED secondary non-regression vs baseline (D24-05 secondary: WP accuracy / ATS-OU MAE).
      4. For WP, if ``gate.calibration_in_gate``: ECE + Brier non-regression vs baseline.

    Args:
        target: One of "wp", "ats", "ou".
        candidate: The candidate bundle from ``build_candidate_bundle``.
        baseline: The frozen baseline bundle for this target (from ``config/gate.toml``;
            same key shape as the candidate -- ``clv_values``/``mean``/``per_season`` plus
            ``accuracy``/``ece``/``brier_score`` for WP or ``mae`` for ATS/OU).
        cfg: The loaded gate config (reads ``gate.alpha``, ``gate.per_season_must_pass``,
            ``gate.calibration_in_gate``, ``gate.secondary``).

    Returns:
        ``{"passed": bool, "reasons": list[str], "candidate": {...}, "baseline": {...},
        "per_season": {...}, "v1_metrics": {...}, "v2_metrics": {...}}``. The
        ``v1_metrics``/``v2_metrics`` aliases keep the shape compatible with
        ``scripts.retrain_models.print_gating_summary`` (Plan 24-04 rewire).
    """
    gate = cfg["gate"]
    alpha = gate["alpha"]
    secondary = gate["secondary"]
    reasons: list[str] = []
    passed = True

    # (1) Pooled CLV floor.
    pooled_clv = candidate.get("clv_values")
    if clv_floor_passes(pooled_clv, alpha=alpha):
        reasons.append(
            f"Pooled CLV floor PASS (mean={candidate.get('mean')}, p={candidate.get('p')})"
        )
    else:
        passed = False
        reasons.append(
            f"Pooled CLV significantly negative or untestable "
            f"(mean={candidate.get('mean')}, p={candidate.get('p')})"
        )

    # (2) Per-season-must-pass CLV floor.
    if gate.get("per_season_must_pass"):
        per_season = candidate.get("per_season", {})
        for season in sorted(per_season):
            season_sig = per_season[season]
            season_arr = season_sig.get("clv_values")
            # Synthetic test bundles may carry the raw array under "clv_values"; the real
            # build_candidate_bundle stores clv_significance bundles. Re-test from the array
            # when present, else re-derive the pass/fail from the stored {mean, t, p}.
            if season_arr is not None:
                season_pass = clv_floor_passes(season_arr, alpha=alpha)
            elif season_sig.get("t") is None:
                season_pass = False
            else:
                season_pass = not (season_sig["mean"] < 0 and season_sig["p"] < alpha)
            if not season_pass:
                passed = False
                reasons.append(
                    f"Season {season} CLV floor FAIL "
                    f"(mean={season_sig.get('mean')}, p={season_sig.get('p')}, "
                    f"n={season_sig.get('n')})"
                )
        if passed:
            reasons.append("Per-season CLV floor PASS (all holdout seasons)")

    # (3) Pooled secondary non-regression.
    sec_passed, sec_reasons = _secondary_reasons(target, candidate, baseline, secondary)
    passed = passed and sec_passed
    reasons.extend(sec_reasons)

    # (4) WP calibration-in-gate (D24-05).
    if target == "wp" and gate.get("calibration_in_gate"):
        cal_passed, cal_reasons = _calibration_reasons(candidate, baseline, secondary)
        passed = passed and cal_passed
        reasons.extend(cal_reasons)

    return {
        "passed": passed,
        "reasons": reasons,
        "candidate": candidate,
        "baseline": baseline,
        "per_season": candidate.get("per_season", {}),
        # Aliases kept for print_gating_summary compatibility (Plan 24-04 rewire).
        "v1_metrics": baseline,
        "v2_metrics": candidate,
    }
