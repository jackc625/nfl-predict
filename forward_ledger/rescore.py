"""The ONE re-score routine: put a model's outputs back onto a decision's own candidates (Phase 37).

Two callers re-score a decision that was already made, and both call this module so there is one
re-score routine, not two that can drift apart:

  * replay (``forward_ledger.replay``) re-scores a stored snapshot's gold rows with the ids its rows
    were stamped with, from the ledger's own artifact copies, and NOTES every stored candidate
    input the re-scoring did not reproduce;
  * the frozen shadow arm (``forward_ledger.shadow``) re-scores the LIVE decision's own candidates
    and gold rows with the pinned row-19 ids, from the production artifacts directory, and keeps
    no notes.

EXPLICIT VERSIONS ONLY. Every scorer call passes ``version=`` and the blend slope is read with
``version=``, so nothing here resolves an artifact through ``artifacts/latest.json``: a manifest
swap can never change which model a re-score used.

Reads only: this module writes nothing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from backtest.weekly_bet_list import (
    CANONICAL_TARGETS,
    SPREAD_MARKET_PROB_COLUMN,
    _attach_spread_market_probability,
    _bound_spread_slope,
)

__all__ = [
    "REPLAY_TOLERANCE",
    "SCORED_COLUMNS",
    "Notes",
    "Scorer",
    "column_or_nan",
    "note_differences",
    "numbers_agree",
    "reattach_spread_probability",
    "rescore_candidates",
]

# Two numbers agree when they differ by no more than this (replay's comparison tolerance).
REPLAY_TOLERANCE: float = 1e-9

# The prediction columns each target's scorer writes onto a candidate (``backtest.diagnose``).
SCORED_COLUMNS: dict[str, tuple[str, ...]] = {
    "wp": ("model_prob",),
    "ats": ("model_prob", "model_spread"),
    "ou": ("model_prob", "model_total"),
}

Scorer = Callable[..., pd.DataFrame]

# Per (game_id, target): the stored candidate inputs a re-score did not reproduce.
Notes = dict[tuple[str, str], list[str]]


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def numbers_agree(stored: Any, replayed: Any) -> bool:
    """True when both are null, or both are numbers within :data:`REPLAY_TOLERANCE`."""
    if _is_null(stored) or _is_null(replayed):
        return _is_null(stored) and _is_null(replayed)
    return abs(float(stored) - float(replayed)) <= REPLAY_TOLERANCE


def column_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    """*frame*'s *column*, or an all-NaN series on its index when the column is absent."""
    if column in frame.columns:
        return cast("pd.Series", frame[column])
    return pd.Series(float("nan"), index=frame.index)


def note_differences(
    notes: Notes,
    rows: pd.DataFrame,
    target: str,
    column: str,
    replayed: pd.Series,
) -> None:
    """Record, per game, a stored candidate input the re-score did not reproduce."""
    stored = column_or_nan(rows, column)
    for game_id, before, after in zip(rows["game_id"], stored, replayed, strict=True):
        if not numbers_agree(before, after):
            notes.setdefault((str(game_id), target), []).append(
                f"candidate {column} stored={before!r} replayed={after!r}"
            )


def rescore_candidates(
    candidates: pd.DataFrame,
    gold_by_target: Mapping[str, pd.DataFrame],
    artifacts_dir: Path,
    ids: Mapping[str, str],
    scorer: Scorer,
    notes: Notes | None = None,
) -> None:
    """Re-score every target from its gold rows with ``ids[target]``; substitute in place.

    Args:
        candidates: The decision's candidates frame; its model columns are overwritten.
        gold_by_target: Each target's gold rows exactly as the decision scored them.
        artifacts_dir: Where the artifacts named by *ids* live.
        ids: Model target -> the artifact id to score with (passed as ``version=``).
        scorer: The model scorer (``backtest.diagnose.score_deployed_artifacts``).
        notes: When given, every stored input the re-score did not reproduce is recorded here.
    """
    for target in CANONICAL_TARGETS:
        mask = candidates["target"] == target
        gold = gold_by_target[target]
        if not bool(mask.any()) or gold.empty:
            continue
        scored = scorer(
            target, gold_df=gold, artifacts_dir=artifacts_dir, version=ids[target]
        )
        by_game = scored.set_index("game_id")
        rows = candidates.loc[mask]
        for column in SCORED_COLUMNS[target]:
            replayed = rows["game_id"].map(by_game[column])
            if notes is not None:
                note_differences(notes, rows, target, column, replayed)
            candidates.loc[mask, column] = replayed


def reattach_spread_probability(
    candidates: pd.DataFrame,
    artifacts_dir: Path,
    blend_id: str,
    notes: Notes | None = None,
) -> pd.DataFrame:
    """Recompute WP's spread-derived market probability from the blend *blend_id*'s bound slope."""
    slope = _bound_spread_slope(artifacts_dir, version=blend_id)
    mask = candidates["target"] == "wp"
    if not bool(mask.any()) or (
        slope is None and SPREAD_MARKET_PROB_COLUMN not in candidates.columns
    ):
        return candidates
    rows = candidates.loc[mask]
    unattached = rows.drop(columns=[SPREAD_MARKET_PROB_COLUMN], errors="ignore")
    replayed = column_or_nan(
        _attach_spread_market_probability(unattached, slope), SPREAD_MARKET_PROB_COLUMN
    )
    if notes is not None:
        note_differences(notes, rows, "wp", SPREAD_MARKET_PROB_COLUMN, replayed)
    out = candidates.copy()
    out.loc[mask, SPREAD_MARKET_PROB_COLUMN] = replayed
    return out
