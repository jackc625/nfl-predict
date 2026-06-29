"""Add-one-in signal-lift SCREEN for the Phase-28 new feature groups (SIG-05).

This is a thin ``backtest/diagnose.py`` / ``backtest/ou_divergence.py``-style orchestrator:
per feature group (injury / snap / situational) and per target (WP / ATS / OU) it measures
each group's PAIRED incremental Closing-Line-Value (CLV) lift and applies the D-05 keep/drop
rule. It composes the canonical significance primitives (``clv_significance``,
``CLV_COLUMN_FOR``, ``SIGNIFICANCE_ALPHA``) -- it NEVER re-derives a bespoke paired-delta or
t-test (the D24-13 import-parity lesson).

MECHANISM (review #2 / #3 -- the load-bearing correction):
  The lift is anchored on an IN-PROCESS per-season walk-forward re-fit via
  ``BaseTrainer.train_and_evaluate(features_df, closing_odds_df, tune=False)`` (D24-12) over a
  selected-column dataframe -- NOT ``score_deployed_artifacts`` whole-frame scoring. Codex
  (orchestrator-verified) found that ``models.train --no-tune`` saves a single final model
  trained through ~2023 and ``score_deployed_artifacts`` then scores the WHOLE 2021-2024 gold
  frame, so 2021-2023 are IN-SAMPLE for the candidate. That is NOT the "train <=Y-1, measure Y
  per holdout season" walk-forward D-18c promises, and SIG-05 keep/drop must not rest on an
  in-sample-contaminated number. The trainer's returned ``clv_results`` IS the per-split
  out-of-sample walk-forward CLV (the loop in ``models/trainers/base.py``); that is the lift
  anchor here. The CLI (``models/train.py``) reads ``data/gold/features_{target}.parquet``
  WHOLESALE with no feature-group flag, so the add-one-in seam MUST be in-process on a
  selected-column dataframe; this module NEVER shells out to it.

ADD-ONE-IN (D-02): per group G the candidate dataframe = baseline feature columns PLUS ONLY
group G's NEW columns (the ``select_group_columns`` helper). Dropping a feature column from the
dataframe excludes it from the trainer's feature set (``WalkForwardSplitter`` derives features
as the numeric non-ID columns), so the baseline leg simply omits every Phase-28 new column and
each candidate leg re-admits exactly one group. Both legs go through the SAME walk-forward, so
the per-game delta merged on ``game_id`` is PAIRED and free of whole-frame in-sample
contamination. The D25-11 re-frozen ``config/gate.toml`` post-activation baseline remains the
DOCUMENTED reference cross-check (D-04 paired intent preserved) -- it is not the lift anchor.

D-05 keep/drop rule (per group, applied across WP/ATS/OU):
  KEEP iff the point-estimate delta is > 0 on >=1 target AND no target is significantly-negative
  (mean < 0 AND p < SIGNIFICANCE_ALPHA). A 3x3 grid is reported RAW with a multiplicity NOTE; the
  binding p<0.05 multiple-comparison correction stays in Phase 30's deploy gate (D-05).

SITUATIONAL honesty (SC3 / D-17): the new look-ahead / letdown / off-bye spots are documented as
WEAK / largely PRICED-IN / not a standing bet angle, regardless of the screen outcome.

HARD BOUNDARY (T-28-19): this is a READ-ONLY measurement harness. It NEVER writes under ``data/``
and never mutates ``data/gold``; the selection helper returns an in-memory copy. Invoke via the
project venv: ``.venv\\Scripts\\python.exe -m backtest.signal_lift``.

SCREEN-NOT-DEPLOY (D-01 / D-20): Phase 28 SCREENS; Phase 30 runs the binding deploy gate. Kept
groups are CARRIED into widened gold for Phase 30; dropped groups are documented and never enter
gold. This harness and the ``SIGNAL-LIFT-READOUT.md`` it feeds say "screened / carried to Phase
30", NEVER "deployed" / "proven".

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.diagnose import (
    CLV_COLUMN_FOR,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from utils import get_logger

logger = get_logger(__name__)

# The three targets and the three NEW feature groups under screen.
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
GROUPS: tuple[str, ...] = ("injury", "snap", "situational")

# Per-group data-coverage floors (RESEARCH): outside coverage the gold carries neutral defaults
# + a coverage flag (D-10 / D-18d). The 2021-2024 measurement window is fully inside every floor;
# these spans are REPORTED beside each lift number so the readout states the covered span (D-18d).
GROUP_COVERAGE: dict[str, str] = {
    "injury": "injuries 2009+ (measured 2021-2024)",
    "snap": "snaps 2013+ (measured 2021-2024)",
    "situational": "full history (measured 2021-2024)",
}

# The walk-forward measurement window (matches config/gate.toml [gate.seasons].holdout + diagnose).
MEASURE_WINDOW = "2021-2024"

# The lift anchor is the trainer's own per-split out-of-sample walk-forward CLV, NOT
# score_deployed_artifacts whole-frame scoring (review #2). This string is asserted by the seam
# test and recorded in the readout's METHOD line.
LIFT_ANCHOR = "BaseTrainer.train_and_evaluate(tune=False)"

_TRAINER_FOR: dict[str, type] = {
    "wp": WPTrainer,
    "ats": ATSTrainer,
    "ou": OUTrainer,
}

# Default gold/odds locations (read-only). Overridable via run_signal_lift_screen args for tests.
_GOLD_PATH_FOR = {t: Path(f"data/gold/features_{t}.parquet") for t in TARGETS}
_ODDS_PATH = Path("data/silver/odds_snapshot.parquet")


# ---------------------------------------------------------------------------
# (1) Feature-group column-selection helper (review #3) -- read-only, in-memory
# ---------------------------------------------------------------------------


def _is_snap_col(col: str) -> bool:
    """Match a derived snap-count feature (home_/away_ prefixed).

    Matches snap CONTINUITY / CONCENTRATION / rolling-position-share columns only. Deliberately
    does NOT match ``snapshot_spread`` / ``snapshot_total`` / ``snapshot_ml_prob_home_fair`` --
    those carry the substring "snap" but are odds-SNAPSHOT columns, not snap-count features
    (RESEARCH: never add a bare ``snap`` substring guard -- it collides with rolling_snap_*).
    """
    cl = col.lower()
    if not cl.startswith(("home_", "away_")):
        return False
    return (
        "snap_continuity" in cl
        or "snap_concentration" in cl
        or "rolling_snap_share" in cl
    )


def _is_injury_col(col: str) -> bool:
    """Match an injury-availability feature (home_/away_injury_coverage)."""
    return "injury" in col.lower()


# The genuinely-NEW situational spots (D-15/D-16). Existing rest/travel/short-week/bye features are
# already in the baseline, so the situational group under screen is ONLY these new spot flags.
_SITUATIONAL_SUFFIXES = ("off_bye", "look_ahead_spot", "letdown_spot")


def _is_situational_col(col: str) -> bool:
    """Match a NEW situational spot flag (home_/away_ off_bye / look_ahead_spot / letdown_spot)."""
    cl = col.lower()
    return cl.startswith(("home_", "away_")) and cl.endswith(_SITUATIONAL_SUFFIXES)


_GROUP_PREDICATE = {
    "snap": _is_snap_col,
    "injury": _is_injury_col,
    "situational": _is_situational_col,
}


def group_columns(gold_df: pd.DataFrame, group: str) -> list[str]:
    """Return the NEW columns belonging to ``group`` present in ``gold_df`` (sorted)."""
    if group not in _GROUP_PREDICATE:
        msg = f"Unknown group '{group}'. Must be one of {sorted(_GROUP_PREDICATE)}."
        raise ValueError(msg)
    predicate = _GROUP_PREDICATE[group]
    return sorted(c for c in gold_df.columns if predicate(c))


def phase28_new_columns(gold_df: pd.DataFrame) -> list[str]:
    """Return every Phase-28 NEW column (union of all three groups) present in ``gold_df``."""
    cols: list[str] = []
    for group in GROUPS:
        cols.extend(group_columns(gold_df, group))
    return sorted(set(cols))


def select_group_columns(gold_df: pd.DataFrame, group: str | None) -> pd.DataFrame:
    """Return baseline columns + ONLY the requested group's NEW columns (in memory, read-only).

    The baseline is the widened gold with EVERY Phase-28 new column removed (the pre-Phase-28
    activated feature set + ID/target cols + the unrelated odds-snapshot/rest/travel cols).
    Passing a ``group`` re-admits exactly that group's new columns (the add-one-in seam, D-02);
    passing ``group=None`` returns the baseline leg. NEVER mutates ``data/gold`` -- it returns a
    fresh in-memory copy (HARD BOUNDARY, review #3).

    Args:
        gold_df: The Plan 28-06 widened gold feature matrix for one target.
        group: One of "snap" / "injury" / "situational", or None for the baseline leg.

    Returns:
        A copy of ``gold_df`` restricted to baseline columns + (when ``group`` is not None) that
        group's new columns.
    """
    new_all = set(phase28_new_columns(gold_df))
    group_cols = set(group_columns(gold_df, group)) if group is not None else set()
    keep = [c for c in gold_df.columns if c not in new_all or c in group_cols]
    return gold_df[keep].copy()


# ---------------------------------------------------------------------------
# (2) In-process walk-forward CLV anchor (review #2) -- NOT score_deployed_artifacts
# ---------------------------------------------------------------------------


def _walkforward_clv_series(clv_results: pd.DataFrame, target: str) -> pd.DataFrame:
    """Extract the per-game out-of-sample walk-forward CLV (game_id + clv) for one target.

    Reads the target's CLV column via ``CLV_COLUMN_FOR`` (probability_clv for WP, line_clv for
    ATS/OU) from the trainer's ``clv_results``, restricted to games with closing odds and dropping
    any NaN CLV. This is the trainer's OWN per-split walk-forward CLV (the lift anchor), never a
    whole-frame deployed-artifact re-score.
    """
    clv_col = CLV_COLUMN_FOR[target]
    if clv_col not in clv_results.columns:
        msg = (
            f"clv_results for target '{target}' is missing the expected CLV column "
            f"'{clv_col}'. Columns: {list(clv_results.columns)}"
        )
        raise KeyError(msg)
    valid = clv_results[clv_results["has_closing_odds"]]
    out = (
        valid[["game_id", clv_col]]
        .dropna(subset=[clv_col])
        .rename(columns={clv_col: "clv"})
        .reset_index(drop=True)
    )
    return out


def _walkforward_clv(
    target: str,
    features_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    config: TemporalSplitConfig,
) -> pd.DataFrame:
    """Run ONE in-process walk-forward re-fit (tune=False) and return its per-game CLV.

    A fresh trainer is instantiated per call (trainers carry fit state). The trainer's
    ``train_and_evaluate(..., tune=False)`` performs the train<=Y-1 / measure-Y walk-forward and
    returns ``clv_results``; this function extracts the per-game CLV series. No artifact is saved
    and nothing is written under ``data/`` (the trainer's ``save()`` is never called here).
    """
    trainer_cls = _TRAINER_FOR[target]
    trainer = trainer_cls(config=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = trainer.train_and_evaluate(features_df, closing_odds_df, tune=False)
    clv_results = result.get("clv_results")
    if clv_results is None or len(clv_results) == 0:
        return pd.DataFrame(columns=["game_id", "clv"])
    return _walkforward_clv_series(clv_results, target)


def _paired_delta(
    baseline_clv: pd.DataFrame,
    candidate_clv: pd.DataFrame,
) -> tuple[Any, int]:
    """Merge baseline + candidate per-game CLV on ``game_id`` and return (delta_array, n_paired).

    The paired per-game delta is ``candidate - baseline`` over the games BOTH legs scored (an
    inner merge on game_id). This preserves the D-04 paired intent while the D-18c walk-forward
    governs the mechanism.
    """
    merged = baseline_clv.merge(
        candidate_clv, on="game_id", how="inner", suffixes=("_base", "_cand")
    )
    delta = (merged["clv_cand"] - merged["clv_base"]).to_numpy()
    return delta, len(merged)


# ---------------------------------------------------------------------------
# (3) Per-target screen + (4) the D-05 keep/drop rule
# ---------------------------------------------------------------------------


def _screen_target(
    target: str,
    gold_df: pd.DataFrame,
    baseline_clv: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    config: TemporalSplitConfig,
    group: str,
) -> dict[str, Any]:
    """Screen one (group, target): paired incremental CLV delta + the per-target keep/veto flags."""
    candidate_df = select_group_columns(gold_df, group=group)
    candidate_clv = _walkforward_clv(target, candidate_df, closing_odds_df, config)
    delta, n_paired = _paired_delta(baseline_clv, candidate_clv)
    sig = clv_significance(delta)

    mean = sig["mean"]
    p = sig["p"]
    keep_target = mean is not None and mean > 0
    veto = mean is not None and mean < 0 and p is not None and p < SIGNIFICANCE_ALPHA
    return {
        "target": target,
        "clv_column": CLV_COLUMN_FOR[target],
        "n_paired": int(n_paired),
        "delta_mean": mean,
        "delta_t": sig["t"],
        "delta_p": p,
        "delta_ci95": sig["ci95"],
        "keep_target": bool(keep_target),
        "veto": bool(veto),
    }


def decide_group_keep(per_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the D-05 keep/drop rule to a group's per-target screen results.

    D-05: KEEP iff the point-estimate delta is > 0 on >=1 target AND no target is
    significantly-negative (mean < 0 AND p < SIGNIFICANCE_ALPHA, the per-target veto). A group that
    is flat/negative everywhere, or significantly-negative on ANY target, is DROPPED. The binding
    multiple-comparison correction stays in the Phase-30 deploy gate (multiplicity note).

    Args:
        per_target: Mapping target -> the ``_screen_target`` result dict (carries keep_target/veto).

    Returns:
        Dict with ``keep`` (bool), ``any_positive``, ``any_veto``, ``positive_targets``,
        ``veto_targets``, and a human-readable ``reason``.
    """
    positive_targets = [t for t, r in per_target.items() if r["keep_target"]]
    veto_targets = [t for t, r in per_target.items() if r["veto"]]
    any_positive = bool(positive_targets)
    any_veto = bool(veto_targets)
    keep = any_positive and not any_veto

    if any_veto:
        reason = (
            f"DROP: significantly-negative on {veto_targets} (D-05 veto); "
            "dropped, not silently retained"
        )
    elif not any_positive:
        reason = (
            "DROP: no positive point-estimate on any target (no incremental CLV lift); "
            "dropped, not silently retained"
        )
    else:
        reason = (
            f"KEEP: positive point-estimate on {positive_targets} and not "
            "significantly-negative on any target -- carried to Phase 30 for the binding gate"
        )

    return {
        "keep": keep,
        "any_positive": any_positive,
        "any_veto": any_veto,
        "positive_targets": positive_targets,
        "veto_targets": veto_targets,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# (5) Single re-runnable orchestrator -- one structured dict over all groups x targets
# ---------------------------------------------------------------------------

_MULTIPLICITY_NOTE = (
    "Multiplicity: this is a 3x3 (group x target) screen reported RAW. Per-target p-values are "
    "NOT multiple-comparison-corrected here -- the screen is a permissive add-one-in filter and "
    "the binding BH-FDR / p<0.05 correction stays in the Phase-30 deploy gate (D-05). Treat any "
    "single nominally-significant cell as a screening signal, not a deploy decision."
)


def run_signal_lift_screen(
    gold_by_target: dict[str, pd.DataFrame] | None = None,
    closing_odds_df: pd.DataFrame | None = None,
    config: TemporalSplitConfig | None = None,
    targets: tuple[str, ...] | list[str] = TARGETS,
    groups: tuple[str, ...] | list[str] = GROUPS,
) -> dict[str, Any]:
    """Run the add-one-in lift screen over all groups x targets into ONE structured dict.

    Per target the baseline leg (no Phase-28 columns) is computed ONCE and reused across groups;
    each group's candidate leg adds exactly that group's new columns. Every leg is an in-process
    ``train_and_evaluate(tune=False)`` walk-forward re-fit (the lift anchor, review #2), so the
    output is deterministic and re-runnable -- the anti-rot guard the D-20 readout doc-drift test
    runs against. NEVER writes ``data/``.

    Args:
        gold_by_target: Optional {target -> widened gold frame}. When None, loaded read-only from
            ``data/gold/features_{target}.parquet``. Injectable for the deterministic fixture test.
        closing_odds_df: Optional normalized closing odds. When None, loaded read-only from
            ``data/silver/odds_snapshot.parquet``.
        config: Optional temporal split config. Defaults to ``TemporalSplitConfig.default()``
            (train 2018-2019 / hp_val 2020 / holdout 2021-2024).
        targets: Targets to screen. Defaults to ("wp", "ats", "ou").
        groups: Feature groups to screen. Defaults to ("injury", "snap", "situational").

    Returns:
        Dict with ``measure_window``, ``alpha``, ``anchor`` (the LIFT_ANCHOR string),
        ``multiplicity_note``, ``targets``, and ``groups`` -- the last a mapping group ->
        {``coverage_span``, ``per_target`` (target -> screen result), ``decision`` (the D-05
        keep/drop)}.
    """
    config = config or TemporalSplitConfig.default()
    targets = list(targets)
    groups = list(groups)

    if closing_odds_df is None:
        closing_odds_df = pd.read_parquet(_ODDS_PATH)
    if gold_by_target is None:
        gold_by_target = {t: pd.read_parquet(_GOLD_PATH_FOR[t]) for t in targets}

    # Baseline per target: computed ONCE (no Phase-28 columns) and reused for every group.
    baseline_clv_by_target: dict[str, pd.DataFrame] = {}
    for target in targets:
        baseline_df = select_group_columns(gold_by_target[target], group=None)
        baseline_clv_by_target[target] = _walkforward_clv(
            target, baseline_df, closing_odds_df, config
        )
        logger.info(
            "Baseline walk-forward CLV computed",
            target=target,
            n_games=len(baseline_clv_by_target[target]),
        )

    groups_out: dict[str, Any] = {}
    for group in groups:
        per_target: dict[str, Any] = {}
        for target in targets:
            per_target[target] = _screen_target(
                target,
                gold_by_target[target],
                baseline_clv_by_target[target],
                closing_odds_df,
                config,
                group,
            )
        decision = decide_group_keep(per_target)
        groups_out[group] = {
            "coverage_span": GROUP_COVERAGE.get(group, "full history"),
            "per_target": per_target,
            "decision": decision,
        }
        logger.info(
            "Group screened",
            group=group,
            keep=decision["keep"],
            reason=decision["reason"],
        )

    return {
        "measure_window": MEASURE_WINDOW,
        "alpha": SIGNIFICANCE_ALPHA,
        "anchor": LIFT_ANCHOR,
        "multiplicity_note": _MULTIPLICITY_NOTE,
        "targets": targets,
        "groups": groups_out,
    }


# ---------------------------------------------------------------------------
# CLI: print the structured screen (never writes data/)
# ---------------------------------------------------------------------------


def _format_screen_report(result: dict[str, Any]) -> str:
    """Render the structured screen result as an ASCII report for the CLI / readout authoring."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  SIGNAL-LIFT SCREEN (SIG-05) -- add-one-in paired incremental CLV")
    lines.append("=" * 78)
    lines.append(f"  Anchor        : {result['anchor']} (out-of-sample walk-forward)")
    lines.append(f"  Measure window: {result['measure_window']}")
    lines.append(f"  Alpha         : {result['alpha']}")
    lines.append("")
    for group, gdata in result["groups"].items():
        lines.append("-" * 78)
        lines.append(f"  GROUP: {group}   [coverage: {gdata['coverage_span']}]")
        lines.append("-" * 78)
        header = (
            f"    {'target':<8}{'n_paired':<10}{'delta_mean':<14}"
            f"{'t':<10}{'p':<12}{'keep':<6}{'veto':<6}"
        )
        lines.append(header)
        for target, r in gdata["per_target"].items():
            mean = r["delta_mean"]
            t_stat = r["delta_t"]
            p = r["delta_p"]
            mean_s = f"{mean:+.6f}" if mean is not None else "n/a"
            t_s = f"{t_stat:+.3f}" if t_stat is not None else "n/a"
            p_s = f"{p:.5f}" if p is not None else "n/a"
            lines.append(
                f"    {target:<8}{r['n_paired']:<10}{mean_s:<14}"
                f"{t_s:<10}{p_s:<12}{r['keep_target']!s:<6}{r['veto']!s:<6}"
            )
        decision = gdata["decision"]
        lines.append(f"    DECISION: {'KEEP' if decision['keep'] else 'DROP'}")
        lines.append(f"      {decision['reason']}")
        lines.append("")
    lines.append("-" * 78)
    lines.append("  " + result["multiplicity_note"])
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    """Run the screen on the canonical widened gold and print the structured report.

    READ-ONLY: loads gold + odds, prints the report, and exits. Never writes ``data/``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_signal_lift_screen()
    print(_format_screen_report(result))  # noqa: T201 -- CLI report to stdout (read-only harness)


if __name__ == "__main__":
    main()
