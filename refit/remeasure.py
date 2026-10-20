"""The per-generation re-measure of the bet rule and the blend (Phase 37; SPEC R4, D-22, D-27..D-30).

WHAT THIS IS
------------
A re-fit changes the models, so the numbers that describe the models -- the error spread, the
lean, the edge bands, the WP calibration verdict and the blend weights -- are re-measured for every
generation by the FROZEN row-19 procedure (the code behind ``outputs/row19/neutral_hfa_chain_fit.json``,
``backtest/neutral_hfa_ev_chain_constants.py``, ``backtest/neutral_hfa_cold_start_constants.py`` and
the blend tune in ``backtest/tune.py``), re-run per generation and changing ONLY what the record
changes (``ROLLING-REFIT-PREREGISTRATION.md`` section 9):

* every walk-forward prediction comes from a ``refit.training`` model fitted at its fold's cutoff
  by the frozen recipe, so inputs are re-picked inside every walk-forward step; and
* the finished 2026 games are predicted week by week, one fold per week.

THE FOLDS (D-22)
----------------
One fold per historical season 2017-2024, cutoff = that season's first kickoff, predicting that
season; one fold per finished 2026 week before the generation's cutoff, cutoff = that week's first
kickoff, predicting that week's finished games before the generation's cutoff. No prediction comes
from a model that trained on its game, and nothing at or after the generation's cutoff is measured.

"Finished" means both scores are present in the ``games`` frame the CALLER passes; silver is never
re-read here. On a retry the hook passes a copy with the first attempt's no-final-score games
masked (R10), so a held game is neither trained on, nor predicted, nor in the corpus.

EVERY ESTIMATOR IS CALLED, NEVER COPIED
---------------------------------------
The lean, the frozen SD and the WP gate go through ``backtest.profitability_2025._fit_target_on_tune``
(which calls ``estimate_prior_season_bias``, ``fit_frozen_residual_sd`` and
``run_wp_calibration_gate``); the edge bands through
``scripts.derive_cold_start_constants.derive_threshold_pairs``; the blend through
``backtest.tune.run_blend_tuning``. The EV floors are the record's fixed ``EV_FLOORS``: no sweep
runs. Every fit runs inside ``models.train.pinned_thread_pool(THREAD_LIMIT)``.

A target with fewer than ``MIN_USABLE_REMEASURE_ROWS`` usable rows, or any threshold the frozen
derivation refuses, is an integrity refusal (:class:`RemeasureRefusal`, check
``remeasure_complete``), never a generation that silently places no bets.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from backtest.refit_preregistration_2026 import (
    HISTORICAL_FOLD_SEASONS,
    REFIT_SEASON,
    THREAD_LIMIT,
)
from forward_ledger.settle import realized_values_from_scores
from models.train import pinned_thread_pool
from refit.training import fit_target_at_cutoff, predict_rows

__all__ = [
    "CHECK_REMEASURE_COMPLETE",
    "FoldSpec",
    "RemeasureRefusal",
    "fold_specs",
    "walk_forward_predictions",
]

#: The pre-registered integrity check every refusal here names
#: (``backtest.refit_preregistration_2026.INTEGRITY_CHECKS``).
CHECK_REMEASURE_COMPLETE: str = "remeasure_complete"

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
_ROW_ORDER: list[str] = ["season", "week", "game_id"]


class RemeasureRefusal(Exception):
    """The re-measure could not be completed honestly, so the re-fit is refused.

    Inherits bare ``Exception``: several call sites catch ``ValueError`` / ``KeyError`` and
    degrade quietly, and a refused re-measure degraded into "carry on" would install a generation
    whose bet-rule numbers were never measured.

    Attributes:
        check: Always ``remeasure_complete``, the pre-registered integrity check.
    """

    check: str = CHECK_REMEASURE_COMPLETE

    def __init__(self, message: str) -> None:
        super().__init__(f"{CHECK_REMEASURE_COMPLETE}: {message}")


@dataclass(frozen=True)
class FoldSpec:
    """One walk-forward fold: the instant its model is fitted to, and the games it predicts.

    Attributes:
        label: ``season_2017`` .. ``season_2024`` or ``2026_week_01`` ...
        cutoff_utc: The fold's training cutoff (tz-aware UTC): the first kickoff of its season
            or week. Every training row kicked off strictly before it.
        predict_game_ids: The finished games the fold predicts, sorted.
    """

    label: str
    cutoff_utc: datetime
    predict_game_ids: tuple[str, ...]


def _as_utc(instant: datetime) -> datetime:
    """*instant* as aware UTC; a naive instant is refused, never guessed."""
    if instant.tzinfo is None:
        msg = (
            f"the generation cutoff {instant!r} is naive; a cutoff must carry its zone"
        )
        raise RemeasureRefusal(msg)
    return instant.astimezone(UTC)


def _numeric(column: Any) -> pd.Series:
    """*column* as numbers, anything unreadable as missing."""
    return cast("pd.Series", pd.to_numeric(column, errors="coerce"))


def _kickoffs(games: pd.DataFrame) -> pd.Series:
    """Each game's kickoff as a tz-aware UTC instant, in *games* order."""
    return cast("pd.Series", pd.to_datetime(games["kickoff_et"], utc=True))


def _finished(games: pd.DataFrame) -> pd.Series:
    """Both scores present in *games* (the caller's frame, never a re-read of silver)."""
    return _numeric(games["home_score"]).notna() & _numeric(games["away_score"]).notna()


def _sorted_ids(games: pd.DataFrame, mask: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(games.loc[mask, "game_id"].astype(str)))


def _first_kickoff(kickoff: pd.Series, mask: pd.Series, what: str) -> datetime:
    """The earliest kickoff under *mask*, as an aware datetime; a missing one is refused."""
    first = pd.Timestamp(kickoff[mask].min())
    if first is pd.NaT:
        msg = f"{what} has no kickoff time, so its fold has no cutoff"
        raise RemeasureRefusal(msg)
    return cast("datetime", first.to_pydatetime())


def fold_specs(
    games: pd.DataFrame, generation_cutoff: datetime
) -> tuple[FoldSpec, ...]:
    """The pre-registered folds for a generation whose cutoff is *generation_cutoff* (D-22).

    Args:
        games: Silver ``games`` (``game_id``, ``season``, ``week``, tz-aware ``kickoff_et``,
            scores) as the caller holds it.
        generation_cutoff: The generation's scheduled cutoff, tz-aware.

    Returns:
        The eight season folds in season order, then one fold per 2026 week whose first game
        kicked off before the cutoff and which has a finished game before it, in week order.

    Raises:
        RemeasureRefusal: a naive cutoff, or a historical fold season absent from *games*.
    """
    cutoff = _as_utc(generation_cutoff)
    kickoff = _kickoffs(games)
    measurable = _finished(games) & (kickoff < pd.Timestamp(cutoff))

    specs: list[FoldSpec] = []
    for season in HISTORICAL_FOLD_SEASONS:
        in_season = games["season"].astype(int) == season
        if not bool(in_season.any()):
            msg = f"season {season} has no game in the schedule, so its fold has no cutoff"
            raise RemeasureRefusal(msg)
        specs.append(
            FoldSpec(
                label=f"season_{season}",
                cutoff_utc=_first_kickoff(kickoff, in_season, f"season {season}"),
                predict_game_ids=_sorted_ids(games, in_season & measurable),
            )
        )

    in_refit_season = games["season"].astype(int) == REFIT_SEASON
    week_of = games["week"].astype(int)
    for week in sorted(set(week_of[in_refit_season].tolist())):
        in_week = in_refit_season & (week_of == week)
        first = _first_kickoff(kickoff, in_week, f"{REFIT_SEASON} week {week}")
        if not first < cutoff:
            continue
        predicted = _sorted_ids(games, in_week & measurable)
        if not predicted:
            continue
        specs.append(
            FoldSpec(
                label=f"{REFIT_SEASON}_week_{week:02d}",
                cutoff_utc=first,
                predict_game_ids=predicted,
            )
        )
    return tuple(specs)


def walk_forward_predictions(
    gold_dir: Path | str,
    games: pd.DataFrame,
    generation_cutoff: datetime,
    *,
    snapshots: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Every fold's out-of-sample predictions under the new recipe, per target.

    For each target and each fold, ONE ``refit.training.fit_target_at_cutoff`` model is fitted
    at the fold's cutoff and predicts the fold's games (those with a gold row); labels come from
    the scores in *games* by the gold label formula. Everything runs inside
    ``pinned_thread_pool(THREAD_LIMIT)``.

    Args:
        gold_dir: The directory holding ``features_{wp,ats,ou}.parquet``.
        games: Silver ``games`` as the caller holds it (held games already masked).
        generation_cutoff: The generation's scheduled cutoff, tz-aware.
        snapshots: Backing Elo snapshots for the training provisional-row guard.

    Returns:
        ``{target: frame}`` with ``game_id``, ``season``, ``week``, ``prediction``, ``actual``,
        ``fold`` and ``fold_cutoff_utc``: one row per predicted game.

    Raises:
        RemeasureRefusal: a fold predicted a game its own model trained on, or one game was
            predicted twice.
        refit.training.RefitTrainingError: a fold's training refused.
    """
    specs = fold_specs(games, generation_cutoff)
    labels = realized_values_from_scores(games)
    root = Path(gold_dir)

    predictions: dict[str, pd.DataFrame] = {}
    with pinned_thread_pool(THREAD_LIMIT):
        for target in _TARGETS:
            gold = pd.read_parquet(root / f"features_{target}.parquet")
            gold_ids = gold["game_id"].astype(str)
            frames: list[pd.DataFrame] = []
            for fold in specs:
                rows = gold.loc[gold_ids.isin(fold.predict_game_ids)].copy()
                if rows.empty:
                    continue
                rows["game_id"] = rows["game_id"].astype(str)
                rows = rows.sort_values(_ROW_ORDER, kind="mergesort").reset_index(
                    drop=True
                )
                fit = fit_target_at_cutoff(
                    target, gold, games, fold.cutoff_utc, snapshots=snapshots
                )
                seen = set(fit.eligible.frame["game_id"].astype(str)) & set(
                    rows["game_id"]
                )
                if seen:
                    msg = (
                        f"{target} fold {fold.label} would predict {sorted(seen)[:10]}, which "
                        "its own model trained on"
                    )
                    raise RemeasureRefusal(msg)
                frames.append(
                    pd.DataFrame(
                        {
                            "game_id": rows["game_id"],
                            "season": rows["season"].astype(int),
                            "week": rows["week"].astype(int),
                            "prediction": predict_rows(fit, rows),
                            "actual": [labels[target][gid] for gid in rows["game_id"]],
                            "fold": fold.label,
                            "fold_cutoff_utc": pd.Timestamp(fold.cutoff_utc),
                        }
                    )
                )
            frame = pd.concat(frames, ignore_index=True)
            repeated = sorted(frame.loc[frame["game_id"].duplicated(), "game_id"])
            if repeated:
                msg = f"{target} predicted {repeated[:10]} in more than one fold"
                raise RemeasureRefusal(msg)
            predictions[target] = frame
    return predictions
