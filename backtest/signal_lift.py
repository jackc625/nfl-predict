"""Add-one-in signal-lift SCREEN for the Phase-28 / Phase-29 new feature groups (SIG-05, SIG-04).

This is a thin ``backtest/diagnose.py`` / ``backtest/ou_divergence.py``-style orchestrator:
per feature group (injury / snap / situational / line_movement) and per target (WP / ATS / OU)
it measures each group's PAIRED incremental Closing-Line-Value (CLV) lift and applies the
keep/drop rule. It composes the canonical significance primitives (``clv_significance``,
``CLV_COLUMN_FOR``, ``SIGNIFICANCE_ALPHA``) -- it NEVER re-derives a bespoke paired-delta or
t-test (the D24-13 import-parity lesson).

PHASE-29 BASELINE (SIG-04, review 29-07 HIGH -- load-bearing): the Phase-28 screen removes the
union over ``GROUPS`` from the baseline leg. Phase 29 screens ``line_movement`` and must measure
it INCREMENTAL TO the post-Phase-28 feature set, because the D-02 question the whole phase exists
for is "is line-movement redundant WITH the injury signal the market reacts to?". Registering
``line_movement`` in ``GROUPS`` would strip injury/snap/situational from the baseline too and
answer a different, useless question. So ``line_movement`` lives in ``_GROUP_PREDICATE`` ONLY,
``select_group_columns`` takes an ``exclude_groups`` parameter (default ``GROUPS`` --
Phase-28 behaviour byte-preserved) and ``run_signal_lift_screen`` takes
``baseline_exclude_groups`` (default ``GROUPS``) threaded into BOTH legs. The Phase-29
invocation is ``groups=("line_movement",), baseline_exclude_groups=("line_movement",)``, run
by ``python -m backtest.signal_lift --phase 29``.

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
as the numeric non-ID columns), so the baseline leg simply omits every column belonging to
``baseline_exclude_groups`` and each candidate leg re-admits exactly one group. Both legs go
through the SAME walk-forward and the SAME exclusion set, so they differ by exactly the group
under screen and the per-game delta merged on ``game_id`` is PAIRED and free of whole-frame in-sample
contamination. The D25-11 re-frozen ``config/gate.toml`` post-activation baseline remains the
DOCUMENTED reference cross-check (D-04 paired intent preserved) -- it is not the lift anchor.

D-05 (Phase 28) / D-13 (Phase 29) keep/drop rule (per group, applied across WP/ATS/OU):
  KEEP iff the point-estimate delta is > 0 on >=1 target AND no target is significantly-negative
  (mean < 0 AND p < SIGNIFICANCE_ALPHA). The grid is reported RAW with a multiplicity NOTE; the
  binding p<0.05 multiple-comparison correction stays in Phase 30's deploy gate (D-05).

SITUATIONAL honesty (SC3 / D-17): the new look-ahead / letdown / off-bye spots are documented as
WEAK / largely PRICED-IN / not a standing bet angle, regardless of the screen outcome.

HARD BOUNDARY (T-28-19): this is a READ-ONLY measurement harness. It NEVER writes under ``data/``
and never mutates ``data/gold``; the selection helper returns an in-memory copy. Invoke via the
project venv: ``.venv\\Scripts\\python.exe -m backtest.signal_lift``.

SCREEN-NOT-DEPLOY (D-01 / D-20, D-16): Phases 28 and 29 SCREEN; Phase 30 runs the binding deploy
gate. Kept groups are CARRIED to Phase 30; dropped groups are documented. This harness and the
readouts it feeds (``SIGNAL-LIFT-READOUT.md``, ``LINE-MOVEMENT-READOUT.md``) say "screened /
carried to Phase 30", NEVER "deployed" / "proven". A flat or negative screen is a COMPLETE
result, not a failure: it is what stops Phase 30 chasing a signal that is not there.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
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

# The three targets and the three PHASE-28 feature groups under screen. ``GROUPS`` is the
# Phase-28 baseline-exclusion set (the union removed from the baseline leg); Phase-29's
# ``line_movement`` is deliberately NOT a member (see the module docstring / review 29-07 HIGH).
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
GROUPS: tuple[str, ...] = ("injury", "snap", "situational")

# The Phase-29 (SIG-04) screen: line_movement measured against a baseline that KEEPS the Phase-28
# groups, so the delta answers the D-02 "redundant WITH the injury signal?" question.
PHASE29_GROUPS: tuple[str, ...] = ("line_movement",)

# Per-group data-coverage floors (RESEARCH): outside coverage the gold carries neutral defaults
# + a coverage flag (D-10 / D-18d). The 2021-2024 measurement window is fully inside every
# Phase-28 floor; these spans are REPORTED beside each lift number so the readout states the
# covered span (D-18d). line_movement is the exception worth naming: its archive floor is
# 2020-06-06 and the stored 2020 rows are additionally orphaned (D29-06-01), so every gold row
# before 2021 carries neutral defaults and the measure-2021 training fold sees NO line movement.
GROUP_COVERAGE: dict[str, str] = {
    "injury": "injuries 2009+ (measured 2021-2024)",
    "snap": "snaps 2013+ (measured 2021-2024)",
    "situational": "full history (measured 2021-2024)",
    "line_movement": (
        "odds-timeline 2020-06-06+ (measured 2021-2024; the stored 2020 rows are orphaned "
        "by D29-06-01, so all pre-2021 rows carry neutral defaults)"
    ),
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


# The InjuryBuilder's per-side feature basenames (features/injury.py:96-103,
# enumerated in scripts/data_qa.py:38-39). A bare ``"injury" in col`` substring
# test matches ONLY ``*_injury_coverage`` (2 of the 12 injury columns), so the
# screen measured a near-constant coverage flag instead of the real injury signal
# and the missed columns silently survived into the "baseline" leg (CR-01). Match
# by known basename instead -- mirroring ``_is_situational_col`` below.
_INJURY_COLUMN_BASENAMES = (
    "qb_out_flag",
    "backup_quality_delta",
    "availability_fraction",
    "injury_coverage",
    "availability_coverage",
    "date_modified_coverage",
)


def _is_injury_col(col: str) -> bool:
    """Match an injury feature (home_/away_ + an InjuryBuilder basename)."""
    cl = col.lower()
    return cl.startswith(("home_", "away_")) and cl.endswith(_INJURY_COLUMN_BASENAMES)


# The genuinely-NEW situational spots (D-15/D-16). Existing rest/travel/short-week/bye features are
# already in the baseline, so the situational group under screen is ONLY these new spot flags.
_SITUATIONAL_SUFFIXES = ("off_bye", "look_ahead_spot", "letdown_spot")


def _is_situational_col(col: str) -> bool:
    """Match a NEW situational spot flag (home_/away_ off_bye / look_ahead_spot / letdown_spot)."""
    cl = col.lower()
    return cl.startswith(("home_", "away_")) and cl.endswith(_SITUATIONAL_SUFFIXES)


# The Phase-29 LineMovementBuilder family (D-09), game-level -- a line trajectory belongs to the
# game, not to a side, so these columns carry NO home_/away_ prefix.
#
# Tier (a) WAS bought at Plan 29-05 (`--backfill 2020 2024 --markets totals spreads`), so gold
# carries the seven spread siblings alongside the seven totals features. Both halves are listed
# here DELIBERATELY (review 29-07 HIGH): a suffix set covering only the totals half would leave
# spread movement sitting in the "baseline" leg, so the measured delta would be
# line-movement-incremental-to-line-movement -- a contaminated, meaningless number.
#
# ``line_movement_coverage`` is a member of the family for the same reason: it is a Phase-29
# column, and leaving it in the baseline would hand the baseline leg a Phase-29 signal.
#
# Every entry is an exact endswith SUFFIX, never a bare substring: ``"total" in col`` matches
# ``snapshot_total`` (the freeze anchor, a pre-existing baseline feature) and ``total_points``
# (the OU target), and ``"spread" in col`` matches ``snapshot_spread`` / ``spread_movement``.
# This is the ``_is_snap_col`` lesson (RESEARCH D-13) applied to a second family.
_LINE_MV_SUFFIXES: tuple[str, ...] = (
    "line_movement_coverage",
    # totals family (D-07 primary)
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    # spread siblings (Tier (a), bought at 29-05)
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)


def _is_line_movement_col(col: str) -> bool:
    """Match a Phase-29 line-movement feature by exact suffix (never a bare ``total`` substring).

    Deliberately does NOT match ``snapshot_total`` / ``snapshot_spread`` (the already-modelled
    freeze anchors), ``total_movement`` / ``spread_movement`` (the pre-existing, historically
    identically-0.0 MarketAnchor columns) or ``total_points`` (the OU target).
    """
    return col.lower().endswith(_LINE_MV_SUFFIXES)


_GROUP_PREDICATE = {
    "snap": _is_snap_col,
    "injury": _is_injury_col,
    "situational": _is_situational_col,
    # Phase 29 (SIG-04). Registered HERE and NOT in ``GROUPS`` on purpose: ``GROUPS`` is the
    # default baseline-exclusion set, and adding line_movement to it would strip the kept
    # Phase-28 signal from the baseline leg (review 29-07 HIGH).
    "line_movement": _is_line_movement_col,
}


def group_columns(gold_df: pd.DataFrame, group: str) -> list[str]:
    """Return the NEW columns belonging to ``group`` present in ``gold_df`` (sorted)."""
    if group not in _GROUP_PREDICATE:
        msg = f"Unknown group '{group}'. Must be one of {sorted(_GROUP_PREDICATE)}."
        raise ValueError(msg)
    predicate = _GROUP_PREDICATE[group]
    return sorted(c for c in gold_df.columns if predicate(c))


def excluded_columns(
    gold_df: pd.DataFrame, exclude_groups: tuple[str, ...] | list[str] = GROUPS
) -> list[str]:
    """Return the union of ``exclude_groups``' columns -- the set the baseline leg removes."""
    cols: list[str] = []
    for group in exclude_groups:
        cols.extend(group_columns(gold_df, group))
    return sorted(set(cols))


def phase28_new_columns(gold_df: pd.DataFrame) -> list[str]:
    """Return every Phase-28 NEW column (union of all three GROUPS) present in ``gold_df``."""
    return excluded_columns(gold_df, GROUPS)


def select_group_columns(
    gold_df: pd.DataFrame,
    group: str | None,
    exclude_groups: tuple[str, ...] | list[str] = GROUPS,
) -> pd.DataFrame:
    """Return baseline columns + ONLY the requested group's NEW columns (in memory, read-only).

    The baseline is the widened gold with every column belonging to ``exclude_groups`` removed.
    Passing a ``group`` re-admits exactly that group's new columns (the add-one-in seam, D-02);
    passing ``group=None`` returns the baseline leg. NEVER mutates ``data/gold`` -- it returns a
    fresh in-memory copy (HARD BOUNDARY, review #3).

    ``exclude_groups`` defaults to ``GROUPS``, which is the Phase-28 behaviour byte-for-byte
    (baseline = the pre-Phase-28 activated feature set). Phase 29 passes
    ``exclude_groups=("line_movement",)`` so the baseline RETAINS the kept Phase-28
    injury/snap/situational columns and only line-movement is the add-one-in delta -- the only
    wiring under which the D-02 redundancy question is answerable (review 29-07 HIGH).

    Args:
        gold_df: The widened gold feature matrix for one target.
        group: A registered group name, or None for the baseline leg.
        exclude_groups: The groups removed from the baseline. Defaults to the Phase-28 ``GROUPS``.

    Returns:
        A copy of ``gold_df`` restricted to baseline columns + (when ``group`` is not None) that
        group's new columns.
    """
    removed = set(excluded_columns(gold_df, exclude_groups))
    group_cols = set(group_columns(gold_df, group)) if group is not None else set()
    keep = [c for c in gold_df.columns if c not in removed or c in group_cols]
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
    exclude_groups: tuple[str, ...] | list[str] = GROUPS,
) -> dict[str, Any]:
    """Screen one (group, target): paired incremental CLV delta + the per-target keep/veto flags.

    ``exclude_groups`` MUST be the same set used for the baseline leg, otherwise the two legs
    differ by more than the one group under screen and the delta is not an add-one-in.
    """
    candidate_df = select_group_columns(
        gold_df, group=group, exclude_groups=exclude_groups
    )
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
    baseline_exclude_groups: tuple[str, ...] | list[str] = GROUPS,
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
        baseline_exclude_groups: The groups removed from the BASELINE leg. Defaults to ``GROUPS``
            (the Phase-28 behaviour). Phase 29 passes ``("line_movement",)`` so the baseline
            RETAINS the kept Phase-28 groups and the delta is line-movement incremental to the
            post-Phase-28 feature set (review 29-07 HIGH). It is threaded into BOTH legs, so the
            two legs differ by exactly the group under screen.

    Returns:
        Dict with ``measure_window``, ``alpha``, ``anchor`` (the LIFT_ANCHOR string),
        ``baseline_excludes`` (what the baseline leg dropped), ``multiplicity_note``, ``targets``,
        and ``groups`` -- the last a mapping group -> {``coverage_span``, ``per_target``
        (target -> screen result), ``decision`` (the keep/drop)}.
    """
    config = config or TemporalSplitConfig.default()
    targets = list(targets)
    groups = list(groups)
    baseline_exclude_groups = list(baseline_exclude_groups)

    if closing_odds_df is None:
        closing_odds_df = pd.read_parquet(_ODDS_PATH)
    if gold_by_target is None:
        gold_by_target = {t: pd.read_parquet(_GOLD_PATH_FOR[t]) for t in targets}

    # Baseline per target: computed ONCE (minus baseline_exclude_groups) and reused per group.
    baseline_clv_by_target: dict[str, pd.DataFrame] = {}
    for target in targets:
        baseline_df = select_group_columns(
            gold_by_target[target], group=None, exclude_groups=baseline_exclude_groups
        )
        baseline_clv_by_target[target] = _walkforward_clv(
            target, baseline_df, closing_odds_df, config
        )
        logger.info(
            "Baseline walk-forward CLV computed",
            target=target,
            n_games=len(baseline_clv_by_target[target]),
            baseline_excludes=list(baseline_exclude_groups),
            n_baseline_cols=len(baseline_df.columns),
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
                exclude_groups=baseline_exclude_groups,
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
        "baseline_excludes": list(baseline_exclude_groups),
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
    excludes = result.get("baseline_excludes")
    if excludes is not None:
        lines.append(
            f"  Baseline drops: {excludes} (every OTHER feature group stays in the baseline)"
        )
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so the argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Add-one-in signal-lift screen. --phase 28 (default) screens the Phase-28 groups "
            "against a pre-Phase-28 baseline; --phase 29 screens line_movement against a "
            "baseline that KEEPS the Phase-28 groups (SIG-04)."
        )
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(28, 29),
        default=28,
        help=(
            "28 = the Phase-28 injury/snap/situational screen (default, unchanged). "
            "29 = the SIG-04 line_movement screen: groups=('line_movement',), "
            "baseline_exclude_groups=('line_movement',)."
        ),
    )
    return parser


def screen_kwargs_for_phase(phase: int) -> dict[str, Any]:
    """Return the ``run_signal_lift_screen`` kwargs for a phase's screen.

    Phase 28 uses the module defaults (baseline = pre-Phase-28 feature set). Phase 29 screens
    ONLY ``line_movement`` and excludes ONLY ``line_movement`` from the baseline, so the kept
    Phase-28 groups stay in the baseline and the delta is incremental to the post-Phase-28
    feature set (review 29-07 HIGH). Exposed as a function so the readout doc-drift guard runs
    exactly the invocation the CLI runs -- the doc and the command cannot drift apart.
    """
    if phase == 29:
        return {
            "groups": PHASE29_GROUPS,
            "baseline_exclude_groups": PHASE29_GROUPS,
        }
    return {}


def main(argv: list[str] | None = None) -> None:
    """Run the screen on the canonical widened gold and print the structured report.

    READ-ONLY: loads gold + odds, prints the report, and exits. Never writes ``data/``.
    """
    args = _build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_signal_lift_screen(**screen_kwargs_for_phase(args.phase))
    print(_format_screen_report(result))  # noqa: T201 -- CLI report to stdout (read-only harness)


if __name__ == "__main__":
    main()
