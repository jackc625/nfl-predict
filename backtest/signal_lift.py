"""Add-one-in signal-lift SCREEN for the Phase-28 / Phase-29 new feature groups (SIG-05, SIG-04).

This is a thin ``backtest/diagnose.py`` / ``backtest/ou_divergence.py``-style orchestrator:
per feature group (injury / snap / situational / line_movement) and per target (WP / ATS / OU)
it measures each group's PAIRED incremental Closing-Line-Value (CLV) lift and applies the
keep/drop rule. It composes the canonical significance primitives (``clv_significance``,
``CLV_COLUMN_FOR``, ``SIGNIFICANCE_ALPHA``) -- it NEVER re-derives a bespoke paired-delta or
t-test (the D24-13 import-parity lesson).

PHASE-28 BASELINE PIN (load-bearing): the Phase-28 screen removes the union over
``ALL_REGISTERED_GROUPS`` -- every registered signal group -- from its baseline leg, so its
baseline is the non-signal core and STAYS the non-signal core as later phases widen gold. It
previously removed only ``GROUPS``, a deny-list of three names, which let Phase-29's fifteen
line-movement columns into the Phase-28 baseline and silently changed what the recorded Phase-28
grid measured. See ``ALL_REGISTERED_GROUPS`` for the full account and the numbers it moved.

PHASE-29 BASELINE (SIG-04, review 29-07 HIGH -- load-bearing): Phase 29 screens ``line_movement``
and must measure it INCREMENTAL TO the post-Phase-28 feature set, because the D-02 question the
whole phase exists for is "is line-movement redundant WITH the injury signal the market reacts
to?". Excluding every group from its baseline would strip injury/snap/situational too and answer
a different, useless question. So ``select_group_columns`` takes an ``exclude_groups`` parameter
and ``run_signal_lift_screen`` takes ``baseline_exclude_groups`` (both defaulting to ``GROUPS``,
so every pre-existing caller is byte-preserved) threaded into BOTH legs, and each phase states its
own baseline explicitly at the ``screen_kwargs_for_phase`` seam. The Phase-29 invocation is
``groups=("line_movement",), baseline_exclude_groups=("line_movement",)``, run by
``python -m backtest.signal_lift --phase 29``; the Phase-28 invocation is
``baseline_exclude_groups=ALL_REGISTERED_GROUPS``, run by ``--phase 28`` (the default).

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
    MIN_CLV_SAMPLE,
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


def _REQUIREMENT_FOR_GROUPS(groups) -> str:
    """The requirement ID the screened groups belong to (WR-13).

    ``_format_screen_report`` printed the literal ``SIG-05`` for EVERY run,
    including ``--phase 29``, which is SIG-04 -- so the header on the Phase-29
    report named the wrong requirement. Derived from the groups screened rather
    than passed in, so it cannot drift from what was actually measured.
    """
    names = set(groups)
    if names and names <= set(PHASE29_GROUPS):
        return "SIG-04"
    if names & set(PHASE29_GROUPS):
        return "SIG-04/SIG-05"
    return "SIG-05"


# Per-group data-coverage floors (RESEARCH): outside coverage the gold carries neutral defaults
# + a coverage flag (D-10 / D-18d). The 2021-2024 measurement window is fully inside every
# Phase-28 floor; these spans are REPORTED beside each lift number so the readout states the
# covered span (D-18d). line_movement is the exception worth naming: its archive floor is
# 2020-06-06, so 2018-2019 has ZERO trajectory coverage and those rows carry neutral defaults.
# The stored 2020 rows WERE orphaned by a game_id off-by-one (D29-06-01) and were re-keyed in
# place by quick task 260816-u0e, so 2020 now carries real trajectories; 2018-2019 remain
# uncovered because no archive exists before the floor.
GROUP_COVERAGE: dict[str, str] = {
    "injury": "injuries 2009+ (measured 2021-2024)",
    "snap": "snaps 2013+ (measured 2021-2024)",
    "situational": "full history (measured 2021-2024)",
    "line_movement": (
        "odds-timeline 2020-06-06+ (measured 2021-2024; the stored 2020 rows were re-keyed "
        "in place by quick task 260816-u0e and now carry real trajectories, while 2018-2019 "
        "predate the archive floor and carry neutral defaults)"
    ),
}

# The walk-forward measurement window (matches config/gate.toml [gate.seasons].holdout + diagnose).
MEASURE_WINDOW = "2021-2024"

# The lift anchor is the trainer's own per-split out-of-sample walk-forward CLV, NOT
# score_deployed_artifacts whole-frame scoring (review #2). This string is asserted by the seam
# test and recorded in the readout's METHOD line.
LIFT_ANCHOR = "BaseTrainer.train_and_evaluate(tune=False)"

# A DIAGNOSTIC temporal config for groups whose data floor lands after the canonical selection
# window (Plan 29-07). Every trainer selects its features on ``config.train_seasons`` ONLY and
# LOCKS that set for the whole holdout walk-forward. The canonical window is train 2018-2019, and
# the odds-timeline archive floor is 2020-06-06, so under the canonical config every
# line-movement column is constant across the entire selection window, has zero importance by
# construction, and CANNOT be selected -- the screen then measures selection churn instead of the
# group. This config slides train/hp_val into covered seasons so the selector can actually see
# the family.
#
# It is a DIAGNOSTIC, never the canonical measurement: it trains on holdout seasons, leaving a
# single measured season (2024, ~255 paired games), so it is low-powered and consumes holdout.
# Results from it are reported as such and never presented as the pre-registered screen.
#
# FROZEN BY DESIGN -- DO NOT DERIVE THESE FROM conf/season_partition.py (review WR-14).
# These seasons are the window a PUBLISHED measurement was taken on, not a live default.
# Moving them to the committed rule would silently re-window SIGNAL-LIFT-READOUT.md's
# numbers so that the document no longer describes the run that produced it. A literal
# here is the correct shape: it is a record, and a record does not roll forward.
COVERAGE_WINDOW_CONFIG = TemporalSplitConfig(
    train_seasons=[2021, 2022],
    hp_val_seasons=[2023],
    holdout_seasons=[2024],
)

# The COVERED SELECTION WINDOW (quick task 260816-u0e, D-Q2). It answers the objection that
# COVERAGE_WINDOW_CONFIG above trains on holdout seasons and measures a single season: here the
# selection window is slid forward only as far as 2020 -- which became a covered season once the
# orphaned 2020 archive rows were re-keyed -- so the selector can actually see the family, while
# 2022-2024 (~764 paired games) is measured out of sample.
#
# 2021 is spent as the hp-val fold, and that cost is REAL and was accepted deliberately. The
# original design left ``hp_val_seasons`` EMPTY to keep the whole 2021-2024 holdout as the
# measurement span (~1,019 games), on the reasoning that an hp-val fold does nothing under
# ``tune=False``. That reasoning was WRONG, and it is worth recording why rather than quietly
# fixing it: it is true of ``BaseTrainer``, where hp_val feeds only ``combined_train`` /
# ``combined_targets`` (``models/trainers/base.py:355-360``) inside the ``if tune`` branch
# (``base.py:361-368``) -- but all three CONCRETE trainers override ``train_and_evaluate`` and fit
# a post-hoc conversion component on the hp-val fold OUTSIDE that branch:
#
#   models/trainers/wp_trainer.py:310-326   Platt/isotonic probability calibrator
#   models/trainers/ats_trainer.py:244-250  ResidualDistributionConverter on hp-val residuals
#   models/trainers/ou_trainer.py:244-252   same pattern
#
# An empty fold therefore crashes WP inside StandardScaler and hands ATS/OU a converter with
# ``residual_std = np.std([]) = NaN``. Borrowing 2021 is what keeps all three targets measured by
# models of the SAME CLASS as the canonical grid's -- the only version of this window whose numbers
# are comparable to 4a's. ``TemporalSplitConfig.validate`` now rejects an empty hp_val by name.
#
# This is a NON-DEFAULT configuration reachable only through the explicit
# ``--covered-selection-window`` flag. ``TemporalSplitConfig.default()`` is NOT touched, so Phase
# 30's binding gate and every trainer keep the canonical 2018-2019 selection window unless Phase
# 30 adopts a covered window deliberately (D29-07-01).
#
# FROZEN BY DESIGN -- DO NOT DERIVE THESE FROM conf/season_partition.py (review WR-14).
# Same reason as COVERAGE_WINDOW_CONFIG above: this is the window the published 4c-bis
# registration was measured on. The next season-literal consolidation pass must leave
# both of these alone.
COVERED_SELECTION_WINDOW_CONFIG = TemporalSplitConfig(
    train_seasons=[2018, 2019, 2020],
    hp_val_seasons=[2021],
    holdout_seasons=[2022, 2023, 2024],
)


def _span(config: TemporalSplitConfig) -> str:
    """The measured season span of a config, DERIVED from its holdout seasons.

    WR-13: every human-facing label for a window is now computed from the config
    it names. Two prose sites said "COVERED_SELECTION_WINDOW_CONFIG measures
    2021-2024" when it measures 2022-2024 -- and that number is what the whole
    4c-bis registration turns on, so it is not a cosmetic error.
    """
    holdout = sorted(config.holdout_seasons or [])
    if not holdout:
        return MEASURE_WINDOW
    if len(holdout) == 1:
        return str(holdout[0])
    return f"{holdout[0]}-{holdout[-1]}"


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


# The weather inputs NO FORECAST CAN SUPPLY (Plan 33.2-12, p332_ rung 4). History is now the
# archived day-before NWS MOS bulletin, which carries a precipitation PROBABILITY and an ordinal
# QPF CATEGORY but no millimetre AMOUNT -- and the category is banded, never turned into a
# fabricated millimetre value. So ``precip_mm`` and ``raw_precip_mm`` are NULL on every historical
# row and leave the model-input candidate set here, through the registry, rather than through a
# hand-written list. Exact ``endswith`` suffixes, never a bare substring: ``"precip" in col``
# would also match ``precip_prob``, ``precip_none``/``light``/``moderate``/``heavy`` and
# ``precip_impact_score``, every one of which the bulletins DO supply.
_WEATHER_UNSUPPLIED_SUFFIXES: tuple[str, ...] = ("precip_mm", "raw_precip_mm")


def _is_weather_unsupplied_col(col: str) -> bool:
    """Match a weather input no forecast supplies (``precip_mm``, ``raw_precip_mm``)."""
    return col.lower().endswith(_WEATHER_UNSUPPLIED_SUFFIXES)


# THE BETTING LINE, WHICH IS NOT A MODEL INPUT FOR ANY TARGET (Plan 33.2-19, p332_ rung 9,
# D33.2-03). All three deployed models selected one -- WP ``snapshot_spread``; ATS
# ``snapshot_spread`` plus ``snapshot_ml_prob_home_fair``; O/U ``snapshot_total``, its rank-1
# input of 25 -- and for 2018-2025 those are CLOSING lines, which did not exist at the lock.
# None of the three targets needs one: WP predicts the winner, ATS the margin, O/U total
# points. The line is used AFTERWARDS, to price and decide a bet, and it stays available for
# grading and CLV; it simply stops being a thing a model is fitted on.
#
# Every entry is an exact endswith SUFFIX, never a bare substring -- the ``_is_line_movement_col``
# lesson applied to a third family. ``"total" in col`` matches ``total_points`` (the O/U
# target) and ``rolling_total_epa``; ``"spread" in col`` matches every line-movement spread
# sibling. The suffix rule still works through a PREFIX, so a hypothetical
# ``home_snapshot_spread`` is matched while ``rolling_total_epa`` is not.
_MARKET_SUFFIXES: tuple[str, ...] = (
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
)


def _is_market_col(col: str) -> bool:
    """Match a betting-line column by exact suffix (never a bare ``total`` substring).

    Deliberately does NOT match ``total_points`` (the O/U target), ``rolling_total_epa`` or
    any other column that merely contains the word.
    """
    return col.lower().endswith(_MARKET_SUFFIXES)


_GROUP_PREDICATE = {
    "snap": _is_snap_col,
    "injury": _is_injury_col,
    "situational": _is_situational_col,
    # Phase 29 (SIG-04). Registered HERE and NOT in ``GROUPS`` on purpose: ``GROUPS`` is the
    # default baseline-exclusion set, and adding line_movement to it would strip the kept
    # Phase-28 signal from the baseline leg (review 29-07 HIGH).
    "line_movement": _is_line_movement_col,
    # Plan 33.2-12 (SPEC R6). Also NOT in ``GROUPS``, for the same reason: this group is never
    # screened into a model, it is DROPPED from gold by scripts.build_features.
    # _enforce_groups_dropped, the one drop mechanism line_movement already uses.
    "weather_unsupplied": _is_weather_unsupplied_col,
    # Plan 33.2-19 (D33.2-03), p332_ rung 9. Registered here for DROPPING, not admitted to
    # ``GROUPS`` for SCREENING -- two different jobs. Adding it to ``GROUPS`` would silently
    # change the Phase-28 baseline, which the ``ALL_REGISTERED_GROUPS`` docstring below
    # explains at length. It leaves gold through ``_enforce_groups_dropped``, the SAME drop
    # mechanism line_movement and weather_unsupplied use -- one mechanism for three groups.
    #
    # ``ALL_REGISTERED_GROUPS`` is DERIVED from this dict, so ``market`` joins it
    # automatically: the Phase-28 baseline pin and the Phase-30 group-gate baseline exclude it
    # without any further edit. That is the intended effect and the reason the pin is derived
    # rather than listed.
    "market": _is_market_col,
}


# EVERY registered signal group -- the Phase-28 baseline PIN.
#
# ``GROUPS`` is a DENY-LIST of three group names, and a deny-list cannot name a group that does
# not exist yet. When Phase 29 widened gold by fifteen ``line_movement`` columns, those columns
# were matched by no entry in ``GROUPS`` and so fell straight through into the BASELINE leg of the
# Phase-28 screen. Nothing failed; the recorded Phase-28 numbers simply stopped being reproducible,
# because re-running them now measured each group incremental to a baseline that had silently
# grown. The situational-OU cell moved from +0.177334 to -0.195371 and flipped the D-05 ruling
# from KEEP to a veto -- entirely from baseline composition, not from any change to the group.
#
# The Phase-28 baseline was never "gold minus these three names". It was "gold minus every signal
# column we know about" -- which happened to equal the three names at the time. Deriving the pin
# from ``_GROUP_PREDICATE`` states that invariant directly, so a group registered by a LATER phase
# is excluded from the Phase-28 baseline automatically and the recorded grid stays reproducible.
# Appending "line_movement" to ``GROUPS`` would fix today's symptom and re-arm the identical trap
# for the next phase that widens gold.
#
# This is deliberately NOT the default of ``select_group_columns`` / ``run_signal_lift_screen``:
# those keep ``GROUPS`` so every existing caller is byte-preserved, and the Phase-29 invocation
# still passes its own explicit ``("line_movement",)`` to keep the kept Phase-28 groups IN its
# baseline (review 29-07 HIGH). The pin is applied at the Phase-28 invocation seam only, in
# ``screen_kwargs_for_phase``.
ALL_REGISTERED_GROUPS: tuple[str, ...] = tuple(_GROUP_PREDICATE)


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
) -> tuple[pd.DataFrame, list[str]]:
    """Run ONE in-process walk-forward re-fit (tune=False); return per-game CLV + the LOCKED
    feature set.

    A fresh trainer is instantiated per call (trainers carry fit state). The trainer's
    ``train_and_evaluate(..., tune=False)`` performs the train<=Y-1 / measure-Y walk-forward and
    returns ``clv_results``; this function extracts the per-game CLV series. No artifact is saved
    and nothing is written under ``data/`` (the trainer's ``save()`` is never called here).

    The second element is the trainer's SELECTED feature list. Every trainer runs its own
    ``SelectFromModel`` pass on the ``config.train_seasons`` window and LOCKS the result for the
    whole holdout walk-forward, so a column that is present in the candidate dataframe has still
    not necessarily reached a single model. Returning the selection is what lets the screen tell
    "this group did not help" apart from "this group was never used" (Plan 29-07, SIG-04).
    """
    trainer_cls = _TRAINER_FOR[target]
    trainer = trainer_cls(config=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = trainer.train_and_evaluate(features_df, closing_odds_df, tune=False)
    selected = list(
        result.get("feature_names") or getattr(trainer, "feature_names", [])
    )
    clv_results = result.get("clv_results")
    if clv_results is None or len(clv_results) == 0:
        return pd.DataFrame(columns=["game_id", "clv"]), selected
    return _walkforward_clv_series(clv_results, target), selected


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
    candidate_clv, selected = _walkforward_clv(
        target, candidate_df, closing_odds_df, config
    )
    delta, n_paired = _paired_delta(baseline_clv, candidate_clv)
    sig = clv_significance(delta)

    # Did the group under screen actually REACH the model? A column can sit in the candidate
    # dataframe and still be discarded by the trainer's train-window SelectFromModel pass. When
    # NONE of the group's columns are selected, the candidate leg's model saw none of them, so the
    # delta is NOT this group's lift -- it is selection churn among the OTHER features (adding
    # columns to the pool perturbs the fitted importances and hence which features get locked).
    # Reporting that delta as a lift would be reporting noise as a finding.
    group_cols = group_columns(gold_df, group)
    selected_group_cols = sorted(set(selected) & set(group_cols))

    mean = sig["mean"]
    p = sig["p"]
    keep_target = mean is not None and mean > 0
    veto = mean is not None and mean < 0 and p is not None and p < SIGNIFICANCE_ALPHA

    # WR-02: n_paired was recorded and never read by any rule, so a cell that
    # measured NOTHING was ruled on as if it had. clv_significance returns all-None
    # at n == 0 (both flags False -> "DROP: no positive point-estimate on any
    # target", a substantive negative finding published for an empty measurement)
    # and a mean with NO p-value at 1 <= n < MIN_CLV_SAMPLE (so keep_target can be
    # True off a three-game fluke while the veto is structurally impossible).
    #
    # ``paired_sufficient`` mirrors the existing measurability accounting so an
    # unmeasurable cell can be REFUSED rather than ruled on. At the screen's actual
    # n of roughly 764 this is inert by construction -- see decide_group_keep.
    paired_sufficient = bool(n_paired >= MIN_CLV_SAMPLE)
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
        "n_group_columns": len(group_cols),
        "n_group_columns_selected": len(selected_group_cols),
        "group_columns_selected": selected_group_cols,
        "paired_sufficient": paired_sufficient,
        "measurable": bool(selected_group_cols) and paired_sufficient,
    }


def decide_group_keep(per_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the D-05 keep/drop rule to a group's per-target screen results.

    D-05: KEEP iff the point-estimate delta is > 0 on >=1 target AND no target is
    significantly-negative (mean < 0 AND p < SIGNIFICANCE_ALPHA, the per-target veto). A group that
    is flat/negative everywhere, or significantly-negative on ANY target, is DROPPED. The binding
    multiple-comparison correction stays in the Phase-30 deploy gate (multiplicity note).

    WR-02 REFUSAL ARM, and its exact adjudication -- written down because it is
    unreachable at the screen's actual n and would otherwise be discovered only by
    whoever next runs the harness at lower n:

    * A per-target row with ``paired_sufficient`` False measured nothing usable, so
      it contributes NO evidence in either direction and is EXCLUDED from both the
      positive set and the veto set.
    * If NO target is sufficient, the group is REFUSED: ``keep`` is False, and the
      reason reads NOT MEASURED with the per-target n_paired, rather than the
      substantive "DROP: no positive point-estimate on any target" the pre-WR-02
      code published for a cell that measured nothing.
    * MIXED case (some targets sufficient, some not): the group IS ruled on, using
      the sufficient targets ONLY, and the insufficient ones are named in the reason
      so the ruling's evidence base is visible. Ruling on the sufficient subset is
      the conservative choice in both directions -- an insufficient row can never
      carry a veto anyway (``clv_significance`` returns no p-value below
      MIN_CLV_SAMPLE, so ``veto`` is structurally False there), while it CAN set
      ``keep_target`` True off a handful of games, so excluding it can only make a
      KEEP harder to obtain, never easier.

    D-R3 INTEGRITY. This function is the implementation the frozen Section 4c-bis
    pre-registration points at. The exclusion above is a no-op when every cell is at
    or above MIN_CLV_SAMPLE -- the positive and veto sets are then identical to the
    pre-WR-02 sets, so the KEEP and DROP arms and their reason strings are
    byte-identical. ``tests/integration/test_signal_lift.py`` walks the whole truth
    table at n >= 10 to demonstrate that, rather than asserting it.

    Args:
        per_target: Mapping target -> the ``_screen_target`` result dict (carries keep_target/veto
            and paired_sufficient).

    Returns:
        Dict with ``keep`` (bool), ``any_positive``, ``any_veto``, ``positive_targets``,
        ``veto_targets``, ``insufficient_targets``, ``measured`` and a human-readable ``reason``.
    """
    sufficient = {
        t: r for t, r in per_target.items() if r.get("paired_sufficient", True)
    }
    insufficient_targets = [t for t in per_target if t not in sufficient]

    positive_targets = [t for t, r in sufficient.items() if r["keep_target"]]
    veto_targets = [t for t, r in sufficient.items() if r["veto"]]
    any_positive = bool(positive_targets)
    any_veto = bool(veto_targets)
    keep = any_positive and not any_veto

    if not sufficient:
        counts = ", ".join(
            f"{t}={per_target[t]['n_paired']}" for t in sorted(per_target)
        )
        return {
            "keep": False,
            "any_positive": False,
            "any_veto": False,
            "positive_targets": [],
            "veto_targets": [],
            "insufficient_targets": insufficient_targets,
            "measured": False,
            "reason": (
                f"NOT MEASURED: n_paired below MIN_CLV_SAMPLE={MIN_CLV_SAMPLE} on every "
                f"target ({counts}); this cell measured nothing, so no keep/drop ruling "
                "is made on it"
            ),
        }

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

    if insufficient_targets:
        counts = ", ".join(
            f"{t}={per_target[t]['n_paired']}" for t in sorted(insufficient_targets)
        )
        reason = (
            f"{reason} [ruled on the sufficient targets only; NOT MEASURED on "
            f"{sorted(insufficient_targets)} ({counts}), below "
            f"MIN_CLV_SAMPLE={MIN_CLV_SAMPLE}]"
        )

    return {
        "keep": keep,
        "any_positive": any_positive,
        "any_veto": any_veto,
        "positive_targets": positive_targets,
        "veto_targets": veto_targets,
        "insufficient_targets": insufficient_targets,
        "measured": True,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# (5) Single re-runnable orchestrator -- one structured dict over all groups x targets
# ---------------------------------------------------------------------------


def _multiplicity_note(n_groups: int, n_targets: int) -> str:
    """The RAW-reporting multiplicity note, sized to the grid ACTUALLY run.

    WR-13: this was a constant reading "this is a 3x3 (group x target) screen",
    printed verbatim under a Phase-29 run, which is 1x3. The multiplicity count is
    load-bearing in the readout's own garden-of-forking-paths argument, so a note
    that overstates the grid is not a cosmetic error.
    """
    return (
        f"Multiplicity: this is a {n_groups}x{n_targets} (group x target) screen reported RAW. "
        "Per-target p-values are "
        "NOT multiple-comparison-corrected here -- the screen is a permissive add-one-in filter "
        "and the binding BH-FDR / p<0.05 correction stays in the Phase-30 deploy gate (D-05). "
        "Treat any single nominally-significant cell as a screening signal, not a deploy "
        "decision."
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
        baseline_clv_by_target[target], _ = _walkforward_clv(
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
        measurable_targets = [t for t, r in per_target.items() if r["measurable"]]
        groups_out[group] = {
            "coverage_span": GROUP_COVERAGE.get(group, "full history"),
            "per_target": per_target,
            "decision": decision,
            "measurability": {
                "measurable": bool(measurable_targets),
                "measurable_targets": measurable_targets,
                "note": (
                    ""
                    if measurable_targets
                    else (
                        "NOT MEASURED: no column of this group was selected by ANY target's "
                        "feature selector, so no candidate model ever saw the group. The deltas "
                        "below are selection churn among the other features, NOT this group's "
                        "lift, and the keep/drop ruling they produce is not evidence about this "
                        "group."
                    )
                ),
            },
        }
        logger.info(
            "Group screened",
            group=group,
            keep=decision["keep"],
            reason=decision["reason"],
            measurable=bool(measurable_targets),
        )

    # Report the window ACTUALLY measured, derived from the config's holdout seasons. A run under
    # COVERAGE_WINDOW_CONFIG measures 2024 alone; printing the canonical "2021-2024" there would
    # mislabel a one-season diagnostic as the four-season screen.
    measure_window = _span(config)

    return {
        "measure_window": measure_window,
        "alpha": SIGNIFICANCE_ALPHA,
        "anchor": LIFT_ANCHOR,
        "baseline_excludes": list(baseline_exclude_groups),
        # WR-13: sized to the grid actually run, not hardcoded 3x3.
        "multiplicity_note": _multiplicity_note(len(groups_out), len(targets)),
        "requirement": _REQUIREMENT_FOR_GROUPS(groups_out),
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
    lines.append(
        f"  SIGNAL-LIFT SCREEN ({result.get('requirement', 'SIG-05')}) "
        f"-- add-one-in paired incremental CLV"
    )
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
            f"{'t':<10}{'p':<12}{'keep':<6}{'veto':<6}{'grp_cols_used':<14}"
        )
        lines.append(header)
        for target, r in gdata["per_target"].items():
            mean = r["delta_mean"]
            t_stat = r["delta_t"]
            p = r["delta_p"]
            mean_s = f"{mean:+.6f}" if mean is not None else "n/a"
            t_s = f"{t_stat:+.3f}" if t_stat is not None else "n/a"
            p_s = f"{p:.5f}" if p is not None else "n/a"
            used = (
                f"{r.get('n_group_columns_selected', 0)}/{r.get('n_group_columns', 0)}"
            )
            lines.append(
                f"    {target:<8}{r['n_paired']:<10}{mean_s:<14}"
                f"{t_s:<10}{p_s:<12}{r['keep_target']!s:<6}{r['veto']!s:<6}{used:<14}"
            )
        measurability = gdata.get("measurability", {})
        if measurability and not measurability.get("measurable", True):
            lines.append("")
            lines.append("    *** " + measurability["note"])
        decision = gdata["decision"]
        lines.append(
            f"    DECISION (rule as written): {'KEEP' if decision['keep'] else 'DROP'}"
        )
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
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--coverage-window",
        action="store_true",
        help=(
            "DIAGNOSTIC: run under COVERAGE_WINDOW_CONFIG (train 2021-2022 / hp_val 2023 / "
            "measure 2024) so a group whose data floor lands after the canonical 2018-2019 "
            "selection window can actually be selected. Low-powered (one season) and it trains "
            "on holdout seasons -- it is NOT the canonical screen."
        ),
    )
    window.add_argument(
        "--covered-selection-window",
        action="store_true",
        help=(
            "Run under COVERED_SELECTION_WINDOW_CONFIG (train 2018-2020 / hp_val 2021 / measure "
            "2022-2024). Slides the selection window forward only as far as 2020 so the selector "
            "can see a group whose archive floor is 2020-06-06. 2021 is spent as the calibration "
            "fold because every concrete trainer fits a conversion component there, so the fold "
            "cannot be empty. NON-DEFAULT: the canonical training window is not mutated."
        ),
    )
    return parser


def screen_kwargs_for_phase(
    phase: int,
    coverage_window: bool = False,
    covered_selection_window: bool = False,
) -> dict[str, Any]:
    """Return the ``run_signal_lift_screen`` kwargs for a phase's screen.

    Phase 28 PINS its baseline to ``ALL_REGISTERED_GROUPS`` -- "gold minus every signal column we
    know about", which is what its recorded grid was actually measured against. It used to rely on
    the module default ``GROUPS``, a deny-list of three names; Phase 29's fifteen ``line_movement``
    columns matched none of them, fell through into the baseline leg, and silently changed what
    the Phase-28 screen measured (see the ``ALL_REGISTERED_GROUPS`` note). Phase 29 screens
    ONLY ``line_movement`` and excludes ONLY ``line_movement`` from the baseline, so the kept
    Phase-28 groups stay in the baseline and the delta is incremental to the post-Phase-28
    feature set (review 29-07 HIGH). ``coverage_window`` swaps in the diagnostic
    ``COVERAGE_WINDOW_CONFIG``; ``covered_selection_window`` swaps in
    ``COVERED_SELECTION_WINDOW_CONFIG``. The two window flags are mutually exclusive -- they name
    different measurement spans, so a run under both would be ambiguous. Exposed as a function so
    the readout doc-drift guard runs exactly the invocation the CLI runs -- the doc and the
    command cannot drift apart.

    WR-13: the span labels below are DERIVED from each config's ``holdout_seasons``
    rather than written out. Both the docstring and the error message used to say
    "COVERED_SELECTION_WINDOW_CONFIG measures 2021-2024"; it measures 2022-2024,
    and that number is what the whole 4c-bis registration turns on.
    """
    if coverage_window and covered_selection_window:
        msg = (
            "coverage_window and covered_selection_window are mutually exclusive: "
            f"COVERAGE_WINDOW_CONFIG measures {_span(COVERAGE_WINDOW_CONFIG)} while "
            f"COVERED_SELECTION_WINDOW_CONFIG measures "
            f"{_span(COVERED_SELECTION_WINDOW_CONFIG)}. Pick one."
        )
        raise ValueError(msg)

    kwargs: dict[str, Any] = {}
    if phase == 28:
        # The PIN. Not ``GROUPS``: see ALL_REGISTERED_GROUPS for why a deny-list of three names
        # let Phase-29 columns into the Phase-28 baseline and moved a published number.
        kwargs["baseline_exclude_groups"] = ALL_REGISTERED_GROUPS
    if phase == 29:
        kwargs["groups"] = PHASE29_GROUPS
        kwargs["baseline_exclude_groups"] = PHASE29_GROUPS
    if coverage_window:
        kwargs["config"] = COVERAGE_WINDOW_CONFIG
    elif covered_selection_window:
        kwargs["config"] = COVERED_SELECTION_WINDOW_CONFIG
    return kwargs


def main(argv: list[str] | None = None) -> None:
    """Run the screen on the canonical widened gold and print the structured report.

    READ-ONLY: loads gold + odds, prints the report, and exits. Never writes ``data/``.
    """
    args = _build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_signal_lift_screen(
            **screen_kwargs_for_phase(
                args.phase,
                args.coverage_window,
                args.covered_selection_window,
            )
        )
    print(_format_screen_report(result))  # noqa: T201 -- CLI report to stdout (read-only harness)


if __name__ == "__main__":
    main()
