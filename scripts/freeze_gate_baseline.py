"""Generate the FROZEN v1.0 deploy-gate baseline (D24-07) as a ready-to-paste TOML block.

This one-shot, re-runnable generator re-scores the DEPLOYED v1.0 production artifacts
(``artifacts/latest.json`` -> the v1.0 pre-Elo WP/ATS/OU models) on the canonical
Phase-20 gold, using the SAME ``backtest/diagnose.py`` path the honest accuracy diagnosis
used (``run_diagnosis(run_backtest_half=False)`` -- the deployed-artifact production half).
It captures, per target:

  - pooled CLV significance (mean / t / p / 95% CI) on the per-target CLV column
    (``probability_clv`` for WP, ``line_clv`` for ATS/OU -- imported via the parity seam),
  - pooled secondary metric (WP: accuracy + ECE + Brier; ATS/OU: MAE),
  - per-season (2021-2024) CLV significance via ``models.deploy_gate.per_season_clv``.

The OUTPUT is a COMPLETE, deterministic, ready-to-paste TOML rendering of the ENTIRE
``[baseline.*]`` section (every pooled table and every per-season sub-table). The human
pastes that one block into ``config/gate.toml`` to FREEZE the baseline (Plan 24-03). The
generator is re-runnable, but the embedded values are frozen by hand and committed in
``config/gate.toml`` -- separating the judge from the judged so the gate's reference can
never silently drift at gate-time (Pitfall 1/3).

Why PRINT (not write into config/): there is no stdlib TOML *writer*, and the repo
.gitignore silently ignores ``config/*.json`` (only ``conf/*.json`` is negated), so a
sibling JSON baseline would be untracked. Embedding the frozen numbers inside the tracked
``config/gate.toml`` is the D24-06/D24-07 anti-gitignore-landmine control. This script
therefore PRINTS the frozen numbers as JSON and the complete TOML block.

HARD discipline (mirrors diagnose.py): this script LOADS + scores only. It imports NO
trainer and NEVER writes ``data/gold/`` (no ``.to_parquet`` call anywhere). The re-score is
inference over the deployed artifacts -- never a re-fit, never a gold rebuild.

Freshness anchors (MODEL-DIAGNOSIS DIAG-05 / AUDIT): the pooled WP accuracy should land near
the deployed-artifact ~0.66023 region (the DIAG-05 raw-prod WP accuracy) and the WP headline
CLV near the recorded deployed value -- ``main`` prints a freshness line so a drift in the
deployed artifacts is visible at a glance. (Plan 24-05's ``test_frozen_baseline_matches_rescore``
re-verifies ALL frozen pooled + per-season values against a fresh re-score, not only these
anchors.)

Usage:
    python -m scripts.freeze_gate_baseline
    python -m scripts.freeze_gate_baseline --artifacts-dir artifacts

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import-the-diagnosis parity seam (D24-13): the per-target CLV column map is IMPORTED, never
# re-declared, so the frozen baseline is measured on the EXACT column the gate + diagnosis use.
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    clv_significance,
    run_diagnosis,
    score_deployed_artifacts,
)
from models.clv import compute_clv_for_predictions
from models.deploy_gate import HOLDOUT_SEASONS, per_season_clv
from utils import get_logger

logger = get_logger(__name__)

# Tolerance for the WR-05 cross-check that the pooled CLV mean from run_diagnosis matches the
# pooled CLV mean re-derived from the single score_deployed_artifacts pass below. Both are
# deterministic inference over the SAME artifacts on the SAME gold, so they must agree to
# floating-point noise; a wider gap means the two paths filtered different games and the frozen
# pooled-vs-per-season values would describe different populations.
_POOLED_CLV_CROSSCHECK_TOL = 1e-6

# Canonical-gold presence guard: the re-score needs the Phase-20 rebuilt gold + closing odds.
_GOLD_WP_PATH = Path("data/gold/features_wp.parquet")

# The targets frozen into the baseline (every gate target).
_TARGETS = ("wp", "ats", "ou")

# MODEL-DIAGNOSIS DIAG-05 raw-prod (DEPLOYED v1.0) freshness anchors. The baseline re-scores
# the DEPLOYED artifacts (run_backtest_half=False), so the right anchors are the DIAG-05
# raw-prod row (WP accuracy 0.66023, raw probability_clv -0.05668), NOT the backtest-model
# anchors. These are sanity bounds printed by main(), not exact targets.
_ANCHOR_WP_ACCURACY_PROD = 0.66023
_ANCHOR_WP_HEADLINE_CLV_PROD = -0.05668

# Deterministic float formatting: a fixed precision so a re-run on unchanged artifacts renders
# a byte-identical TOML block (the Plan 24-05 freshness test compares value-by-value).
_FLOAT_PRECISION = 8


def _fmt(value: float | None) -> str:
    """Render a float (or None) deterministically for the TOML block.

    A None significance field (insufficient sample) is rendered as the TOML float ``nan`` so the
    table key is always present and ``tomllib`` parses it as a float. Real baseline rows on the
    full 2021-2024 population always have a testable n, so ``nan`` is a defensive fallback.

    Args:
        value: The numeric value, or None for an untestable significance field.

    Returns:
        A deterministic TOML float literal (fixed precision) or ``nan``.
    """
    if value is None:
        return "nan"
    return f"{float(value):.{_FLOAT_PRECISION}f}"


def _pooled_clv_block(sig: dict[str, Any]) -> dict[str, float | None]:
    """Flatten a ``clv_significance`` bundle into the pooled baseline CLV keys.

    The 95% CI tuple is split into ``ci95_lo`` / ``ci95_hi`` scalar keys so every baseline
    value is a flat TOML float (tomllib has no tuple type).

    Args:
        sig: A ``clv_significance`` result ``{n, mean, t, p, ci95}``.

    Returns:
        ``{mean, t, p, ci95_lo, ci95_hi}`` with scalar floats (or None).
    """
    ci95 = sig.get("ci95")
    ci_lo, ci_hi = ci95 if ci95 is not None else (None, None)
    return {
        "mean": sig.get("mean"),
        "t": sig.get("t"),
        "p": sig.get("p"),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


def _season_clv_block(sig: dict[str, Any]) -> dict[str, float | int | None]:
    """Flatten a per-season ``clv_significance`` bundle into the per-season baseline keys.

    Args:
        sig: A ``clv_significance`` result for one season slice.

    Returns:
        ``{mean, t, p, n}`` -- the per-season CLV floor inputs (n kept as an int sample size).
    """
    return {
        "mean": sig.get("mean"),
        "t": sig.get("t"),
        "p": sig.get("p"),
        "n": int(sig.get("n", 0)),
    }


def compute_baseline(artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Re-score the deployed v1.0 artifacts on canonical gold and build the frozen baseline dict.

    Loads the 2021-2024 canonical gold + normalized closing odds via the engine loaders
    (matching the ``test_diag_diagnosis`` fixture so the LAR->LA team-abbreviation mapping is
    canonical), runs ``diagnose.run_diagnosis(run_backtest_half=False)`` for the pooled
    deployed-artifact bundle (per-target CLV significance + WP accuracy/ECE/Brier), then scores
    each target ONCE with ``score_deployed_artifacts`` and derives the pooled regression MAE
    (ATS/OU), the per-season CLV slices (``deploy_gate.per_season_clv``), AND a pooled-CLV
    cross-check from that single frame (WR-05). The cross-check asserts the pooled CLV mean from
    ``run_diagnosis`` matches the pooled mean re-derived from the single score (same artifacts,
    same gold, deterministic), so the frozen pooled and per-season CLV cannot describe different
    game populations.

    LOAD + score only -- no trainer import, no ``data/gold/`` write (no ``.to_parquet``).

    Args:
        artifacts_dir: Root directory for the deployed model artifacts (the production swap
            surface). Defaults to ``artifacts``.

    Returns:
        A nested dict shaped EXACTLY to the ``config/gate.toml`` baseline schema::

            {
              "wp":  {"pooled": {mean, t, p, ci95_lo, ci95_hi, accuracy, ece, brier},
                      "season": {2021: {mean, t, p, n}, ..., 2024: {...}}},
              "ats": {"pooled": {mean, t, p, ci95_lo, ci95_hi, mae},
                      "season": {2021: {...}, ..., 2024: {...}}},
              "ou":  {"pooled": {mean, t, p, ci95_lo, ci95_hi, mae},
                      "season": {2021: {...}, ..., 2024: {...}}},
            }

    Raises:
        FileNotFoundError: If the canonical gold is absent (the caller / main handles this as a
            clean skip; raised here so importing callers get an explicit signal).
    """
    artifacts_dir = Path(artifacts_dir)
    if not _GOLD_WP_PATH.exists():
        msg = (
            f"Canonical gold not present at {_GOLD_WP_PATH}; cannot re-score the deployed "
            "artifacts. Build the Phase-20 gold first (see PIPELINE.md)."
        )
        raise FileNotFoundError(msg)

    # Load gold + normalized odds via the engine loaders so team-abbrev mapping matches the
    # backtest/diagnosis exactly (the test_diag_diagnosis fixture convention).
    from backtest.engine import BacktestEngine

    engine = BacktestEngine()
    gold: dict[str, Any] = {}
    for target in _TARGETS:
        df = engine._load_features(target)
        gold[target] = df[
            (df["season"] >= HOLDOUT_SEASONS[0]) & (df["season"] <= HOLDOUT_SEASONS[-1])
        ].copy()
    odds = engine._load_closing_odds()

    # Pooled per-target bundle via the SAME diagnosis path (production half = deployed v1.0).
    diag = run_diagnosis(
        gold=gold,
        odds=odds,
        targets=list(_TARGETS),
        artifacts_dir=artifacts_dir,
        run_backtest_half=False,
    )
    production = diag["production"]

    baseline: dict[str, Any] = {}
    for target in _TARGETS:
        bundle = production[target]
        pooled = _pooled_clv_block(bundle["clv_significance_raw"])

        # Score each target ONCE and derive MAE, per-season slices, AND a pooled-CLV
        # cross-check from that single frame (WR-05). Previously the per-season path and
        # _pooled_mae each re-scored independently (2-3 score passes per target), relying on
        # determinism without asserting it; if run_diagnosis and this path ever filtered
        # different games, the frozen pooled CLV and the per-season CLV would describe
        # different populations with no warning.
        scored = score_deployed_artifacts(
            target, gold_df=gold[target], artifacts_dir=artifacts_dir
        )
        clv_df = compute_clv_for_predictions(scored, odds, target)
        valid = clv_df.loc[clv_df["has_closing_odds"]]

        # Cross-check: the pooled CLV mean re-derived from THIS single score must match the
        # pooled CLV mean run_diagnosis reported (same artifacts, same gold, deterministic).
        crosscheck = clv_significance(valid[CLV_COLUMN_FOR[target]].to_numpy())
        diag_mean = bundle["clv_significance_raw"].get("mean")
        cross_mean = crosscheck.get("mean")
        if diag_mean is not None and cross_mean is not None:
            delta = abs(float(diag_mean) - float(cross_mean))
            if delta > _POOLED_CLV_CROSSCHECK_TOL:
                msg = (
                    f"{target} pooled CLV mean from run_diagnosis ({diag_mean}) disagrees with "
                    f"the single-score re-derivation ({cross_mean}) by {delta:.3e} "
                    f"(> {_POOLED_CLV_CROSSCHECK_TOL:.0e}); the two scoring paths filtered "
                    "different games, so the frozen pooled and per-season CLV would describe "
                    "different populations. Refusing to emit an inconsistent baseline."
                )
                raise ValueError(msg)

        if target == "wp":
            pooled["accuracy"] = bundle["pooled_accuracy"]
            pooled["ece"] = bundle["ece"]
            pooled["brier"] = bundle["brier_score"]
        else:
            pooled["mae"] = _mae_from_valid(target, valid)

        # Per-season CLV slices over the SAME valid frame (the D24-04 per-season floor inputs).
        season_sigs = per_season_clv(valid, target, seasons=HOLDOUT_SEASONS)

        baseline[target] = {
            "pooled": pooled,
            "season": {
                int(season): _season_clv_block(sig)
                for season, sig in season_sigs.items()
            },
        }

    return baseline


def _mae_from_valid(target: str, valid: Any) -> float:
    """Compute pooled MAE for a regression target from an already-scored, odds-filtered frame.

    Mirrors ``models.deploy_gate.build_candidate_bundle``'s MAE so the frozen baseline MAE is
    measured exactly like the candidate MAE the gate compares against (like-for-like): both
    read the EXPLICIT line column (``model_spread`` for ATS, ``model_total`` for OU), never the
    overloaded ``model_prob`` alias (CR-01). Takes the already-scored ``has_closing_odds`` frame
    (the single ``compute_baseline`` score, WR-05) rather than re-scoring, so MAE is over the
    exact same population as the CLV significance and per-season slices.

    Args:
        target: "ats" or "ou".
        valid: The scored, ``has_closing_odds``-filtered predictions frame for the target.

    Returns:
        Pooled mean absolute error between predicted and actual (margin/total).
    """
    import numpy as np

    line_col = "model_spread" if target == "ats" else "model_total"
    if line_col not in valid.columns:
        msg = (
            f"{target} baseline frame missing required '{line_col}' column for the "
            "regression MAE (the line value must be carried explicitly, not via model_prob)"
        )
        raise ValueError(msg)
    return float(
        np.mean(np.abs(valid["actual"].to_numpy() - valid[line_col].to_numpy()))
    )


def render_baseline_toml(baseline: dict[str, Any]) -> str:
    """Serialize the baseline dict into the COMPLETE, deterministic ``[baseline.*]`` TOML block.

    Emits the ENTIRE baseline section -- every ``[baseline.<t>.pooled]`` table and every
    ``[baseline.<t>.season.YYYY]`` sub-table for all three targets and all four holdout
    seasons -- so the human pastes ONE block rather than transcribing individual numbers
    (review concern #6, T-24-12 hand-typo mitigation). The rendering is deterministic (fixed
    target order ``wp, ats, ou``; fixed season order; fixed key order per table; fixed float
    precision) so a re-run on unchanged artifacts yields a byte-identical string.

    Args:
        baseline: The nested dict from ``compute_baseline``.

    Returns:
        A ready-to-paste TOML string covering the entire ``[baseline.*]`` section, terminated by
        a trailing newline. Parsing it with ``tomllib`` yields ``baseline.<t>.pooled`` and
        ``baseline.<t>.season.<YYYY>`` for every target/season.
    """
    # Fixed pooled key order per target (CLV keys first, then the secondary metric(s)).
    pooled_keys: dict[str, tuple[str, ...]] = {
        "wp": ("mean", "t", "p", "ci95_lo", "ci95_hi", "accuracy", "ece", "brier"),
        "ats": ("mean", "t", "p", "ci95_lo", "ci95_hi", "mae"),
        "ou": ("mean", "t", "p", "ci95_lo", "ci95_hi", "mae"),
    }
    season_keys: tuple[str, ...] = ("mean", "t", "p", "n")

    lines: list[str] = [
        "# =============================================================================",
        "# FROZEN BASELINE VALUES -- generated by scripts/freeze_gate_baseline.py",
        "# Re-score of the DEPLOYED INCUMBENT production artifacts on canonical Phase-20",
        "# gold (run_diagnosis production half). Post-Phase-25 the incumbents are the",
        "# activated WP/ATS re-fits with OU retained on v1.0 (D25-11). Do NOT hand-edit",
        "# values; re-run the generator and paste its complete block to refresh.",
        "# Per-season keys parse as strings via tomllib; deploy_gate.load_gate_config",
        "# normalizes them to int.",
        "# =============================================================================",
    ]

    for target in _TARGETS:
        target_block = baseline[target]
        lines.append("")
        lines.append(f"[baseline.{target}.pooled]")
        pooled = target_block["pooled"]
        for key in pooled_keys[target]:
            if key == "n":
                lines.append(f"{key} = {int(pooled[key])}")
            else:
                lines.append(f"{key} = {_fmt(pooled.get(key))}")

        for season in HOLDOUT_SEASONS:
            season_block = target_block["season"][int(season)]
            lines.append("")
            lines.append(f"[baseline.{target}.season.{season}]")
            for key in season_keys:
                if key == "n":
                    lines.append(f"{key} = {int(season_block[key])}")
                else:
                    lines.append(f"{key} = {_fmt(season_block.get(key))}")

    return "\n".join(lines) + "\n"


def _freshness_line(baseline: dict[str, Any]) -> str:
    """Render the WP freshness anchor line comparing the re-score to the DIAG-05 anchors.

    Args:
        baseline: The nested dict from ``compute_baseline``.

    Returns:
        A human-readable freshness line (ASCII) reporting the WP pooled accuracy and headline
        (pooled) CLV mean against the MODEL-DIAGNOSIS DIAG-05 raw-prod anchors, with the deltas.
    """
    wp_pooled = baseline["wp"]["pooled"]
    acc = wp_pooled.get("accuracy")
    clv_mean = wp_pooled.get("mean")
    acc_delta = (acc - _ANCHOR_WP_ACCURACY_PROD) if acc is not None else float("nan")
    clv_delta = (
        (clv_mean - _ANCHOR_WP_HEADLINE_CLV_PROD)
        if clv_mean is not None
        else float("nan")
    )
    return (
        "FRESHNESS (DIAG-05 raw-prod anchors): "
        f"WP accuracy={acc:.5f} (anchor {_ANCHOR_WP_ACCURACY_PROD:.5f}, "
        f"delta {acc_delta:+.5f}); "
        f"WP pooled probability_clv={clv_mean:.5f} "
        f"(anchor {_ANCHOR_WP_HEADLINE_CLV_PROD:.5f}, delta {clv_delta:+.5f})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Parsed arguments namespace with ``artifacts_dir``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Re-score the deployed v1.0 artifacts on canonical gold and PRINT the frozen "
            "deploy-gate baseline as a complete, ready-to-paste TOML block (D24-07)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Deployed production artifacts dir to re-score (default: artifacts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate and PRINT the frozen baseline (JSON + complete TOML block + freshness line).

    PRINTS three things, in order:
      1. the baseline as JSON (machine-readable record of the frozen numbers),
      2. the COMPLETE ready-to-paste ``[baseline.*]`` TOML block (paste into config/gate.toml),
      3. a freshness line comparing pooled WP accuracy + headline CLV to the DIAG-05 anchors.

    Writes nothing into ``config/`` (no stdlib TOML writer; ``config/*.json`` is gitignored).
    Exits cleanly with a clear message when canonical gold is absent (offseason-safe), so a
    routine run on a checkout without gold does not crash.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Exit code: 0 on success OR a clean gold-absent skip; non-zero only on an unexpected
        error.
    """
    args = parse_args(argv)

    print("=" * 70)
    print("NFL Prediction System -- Frozen Deploy-Gate Baseline Generator (D24-07)")
    print("=" * 70)
    print(f"  Deployed artifacts dir: {args.artifacts_dir}")
    print(f"  Holdout seasons: {list(HOLDOUT_SEASONS)}")
    print(f"  CLV columns (parity with diagnose): {CLV_COLUMN_FOR}")
    print("=" * 70)

    if not _GOLD_WP_PATH.exists():
        print(
            f"\nCanonical gold not present at {_GOLD_WP_PATH}; skipping re-score. "
            "Build the Phase-20 gold first (see PIPELINE.md), then re-run."
        )
        return 0

    print("\n[1/3] Re-scoring deployed v1.0 artifacts on canonical gold...")
    baseline = compute_baseline(artifacts_dir=args.artifacts_dir)

    print("\n[2/3] Frozen baseline (JSON):")
    print(json.dumps(baseline, indent=2, default=str))

    print(
        "\n[3/3] Ready-to-paste TOML block (paste into config/gate.toml [baseline.*]):"
    )
    print()
    print(render_baseline_toml(baseline))

    print(_freshness_line(baseline))
    print(
        "\nFREEZE STEP: replace the [baseline.*] tables in config/gate.toml with the block "
        "above (one block-replace), then commit config/gate.toml."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
