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

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.ats_ev_chain import ChainFit, FenceWindow
from backtest.ou_divergence import (
    dedupe_odds_by_book_preference,
    odds_information_time,
)
from backtest.profitability_2025 import _MODEL_COLUMN, _fit_target_on_tune
from backtest.refit_preregistration_2026 import (
    BIAS_SEED_SEASONS,
    EV_FLOORS,
    HISTORICAL_FOLD_SEASONS,
    HISTORICAL_TUNE_SEASONS,
    MIN_USABLE_REMEASURE_ROWS,
    OU_TOTAL_FLOOR,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    THREAD_LIMIT,
)
from forward_ledger.settle import realized_values_from_scores
from models.blending import home_fav_margin_from_prelock_spread
from models.blending_data import (
    DETAIL_POST_LOCK_ONLY,
    EXCLUDED_COLUMNS,
    EXCLUSION_NO_PRELOCK_LINE,
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
    TUNING_FRAME_COLUMNS,
    PrelockTuningCorpus,
    assert_one_row_per_game,
    load_tuning_period_data,
)
from models.market_probability import assert_no_closing_line, oof_market_probability
from models.train import pinned_thread_pool
from refit.training import fit_target_at_cutoff, predict_rows
from refit.witness import PREREGISTRATION_SHA256
from scripts.derive_cold_start_constants import derive_threshold_pairs
from scripts.derive_corrected_ev_chain import (
    DerivationWindowError,
    refuse_seasons_outside,
)

__all__ = [
    "BET_RULE_RECORD_KIND",
    "CHECK_REMEASURE_COMPLETE",
    "BetRuleNumbers",
    "FoldSpec",
    "RemeasureRefusal",
    "assert_corpus_is_prelock",
    "bet_rule_payload",
    "extended_corpus",
    "extended_slopes",
    "fit_frame",
    "fold_specs",
    "measure_bet_rule",
    "walk_forward_predictions",
]

#: The pre-registered integrity check every refusal here names
#: (``backtest.refit_preregistration_2026.INTEGRITY_CHECKS``).
CHECK_REMEASURE_COMPLETE: str = "remeasure_complete"

#: The kind a generation's bet-rule record declares (stored under ``ledger/refit/bet_rules/``).
BET_RULE_RECORD_KIND: str = "refit_generation_bet_rule"

#: The window label the frozen fit carries for a re-measure (it is not the Phase-31 window).
REMEASURE_WINDOW_LABEL: str = "refit_prelock_2020_2024_2026"

#: Why a finished 2026 game has no pre-lock line: the live store holds no row for it at all.
#: (A game whose every row is timed after its lock carries ``DETAIL_POST_LOCK_ONLY``.)
DETAIL_ABSENT_FROM_ODDS_SNAPSHOT: str = "absent_from_odds_snapshot"

#: The silver table holding the live Odds API captures, the 2026 pre-lock source.
ODDS_SNAPSHOT_FILENAME: str = "odds_snapshot.parquet"

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
_ROW_ORDER: list[str] = ["season", "week", "game_id"]
_PREDICTION_COLUMNS: tuple[str, ...] = ("game_id", "season", "prediction", "actual")
_RECORD_ARTIFACT_KEYS: tuple[str, ...] = ("ats", "blend", "converter", "ou", "wp")


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


# ---------------------------------------------------------------------------
# The extended pre-lock corpus (record section 9, "The corpus")
# ---------------------------------------------------------------------------


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(list(columns)))


def _season_2026_corpus(
    silver_dir: Path, games: pd.DataFrame, cutoff: datetime
) -> PrelockTuningCorpus:
    """The finished 2026 games before *cutoff*: each one's latest owned line at or before its lock.

    The selection is the decision path's own (``backtest.weekly_bet_list``): one row per game by
    ``dedupe_odds_by_book_preference(..., locks=..., keep_inadmissible=True)``, judged on
    ``odds_information_time`` (the recorded capture instant), kept only when it is at or before
    the game's lock. A row with no spread or no total is not a line. A game with no such line is
    returned in ``excluded`` under ``no_prelock_line``, never filled.
    """
    kickoff = _kickoffs(games)
    finished_before = (
        (games["season"].astype(int) == REFIT_SEASON)
        & _finished(games)
        & (kickoff < pd.Timestamp(cutoff))
    )
    schedule = games.loc[
        finished_before, ["game_id", "season", "week", "game_type", "kickoff_et"]
    ].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    if schedule.empty:
        return PrelockTuningCorpus(
            frame=_empty(TUNING_FRAME_COLUMNS),
            excluded=_empty(EXCLUDED_COLUMNS),
            unjoined_timeline_ids=(),
        )

    # THE lock, from THE rule. Imported here, not at module level, so the one-lock-rule identity
    # delegate sees the call (tests/unit/test_one_lock_rule_source_scan.py, control 5).
    from utils.game_lock import lock_frame

    locks = lock_frame(schedule)
    odds = pd.read_parquet(silver_dir / ODDS_SNAPSHOT_FILENAME)
    odds = odds.loc[
        odds["game_id"].astype(str).isin(sorted(schedule["game_id"]))
    ].copy()
    odds["game_id"] = odds["game_id"].astype(str)
    lines = odds.dropna(subset=["spread", "total"])

    chosen = dedupe_odds_by_book_preference(lines, locks=locks, keep_inadmissible=True)
    lock_by_game: dict[str, Any] = {str(gid): value for gid, value in locks.items()}
    information_time = odds_information_time(chosen)
    lock = pd.Series(
        pd.to_datetime([lock_by_game[str(gid)] for gid in chosen["game_id"]], utc=True),
        index=chosen.index,
    )
    admissible = information_time.notna() & (information_time <= lock)
    kept = chosen.loc[admissible]

    frame = (
        pd.DataFrame(
            {
                "game_id": kept["game_id"].astype(str),
                "snapshot_ts": information_time[admissible],
                "lock": lock[admissible],
                "created_at": information_time[admissible],
                # THE SIGN: odds_snapshot is already on the home-margin scale; the named
                # identity says so (models/blending.py), never a negation here.
                "market_spread": home_fav_margin_from_prelock_spread(kept["spread"]),
                "market_total": kept["total"].astype(float),
            }
        )
        .merge(
            schedule[["game_id", "season", "week", "game_type"]],
            on="game_id",
            how="inner",
            validate="one_to_one",
        )
        .astype({"season": int, "week": int, "game_type": str})
        .loc[:, list(TUNING_FRAME_COLUMNS)]
        .sort_values("game_id", ignore_index=True)
    )

    seen = set(odds["game_id"])
    missing = schedule.loc[~schedule["game_id"].isin(set(frame["game_id"]))].copy()
    missing["reason"] = EXCLUSION_NO_PRELOCK_LINE
    missing["detail"] = [
        DETAIL_POST_LOCK_ONLY if game_id in seen else DETAIL_ABSENT_FROM_ODDS_SNAPSHOT
        for game_id in missing["game_id"]
    ]
    excluded = (
        missing.loc[:, list(EXCLUDED_COLUMNS)]
        .astype({"season": int, "week": int, "game_type": str})
        .sort_values("game_id", ignore_index=True)
    )
    return PrelockTuningCorpus(frame=frame, excluded=excluded, unjoined_timeline_ids=())


def assert_corpus_is_prelock(frame: pd.DataFrame, games: pd.DataFrame) -> None:
    """Run the one closing-line refusal on a FIT-SHAPED PROJECTION of a corpus frame.

    ``models.market_probability.assert_no_closing_line`` requires ``home_fav_margin`` and
    ``home_win``, which the corpus columns (``TUNING_FRAME_COLUMNS``) do not carry, so it is run
    on ``frame.assign(home_fav_margin=market_spread, home_win=<silver result>)`` -- the shape
    ``load_owned_prelock_lines`` returns. ``market_spread`` is already on the home-margin scale for
    the 2020-2024 and the 2026 rows alike, so the margin is the identity, never negated. The
    projection is checked and discarded; it never enters the corpus.

    Raises:
        models.market_probability.ClosingLineInFitError: a closing-line column, an undated row or
            a row timed after its own lock.
    """
    home_win: dict[str, float] = realized_values_from_scores(games)["wp"]
    projection = frame.assign(
        home_fav_margin=frame["market_spread"],
        home_win=[home_win.get(str(gid)) for gid in frame["game_id"]],
    )
    assert_no_closing_line(projection)


def extended_corpus(
    silver_dir: Path | str, games: pd.DataFrame, generation_cutoff: datetime
) -> PrelockTuningCorpus:
    """The owned pre-lock corpus 2020-2024 extended by the finished 2026 games before the cutoff.

    2020-2024 is ``models.blending_data.load_tuning_period_data(silver_dir)`` unchanged (the
    owned ``odds_timeline``). 2026 is the live ``odds_snapshot`` captures of every game finished
    in *games* before *generation_cutoff*, selected as the decision path selects a line; a 2026
    game without a line at or before its lock is excluded and counted under ``no_prelock_line``.

    Args:
        silver_dir: The silver root (``odds_timeline``, ``games``, ``odds_snapshot``).
        games: Silver ``games`` as the caller holds it (held games already masked).
        generation_cutoff: The generation's scheduled cutoff, tz-aware.

    Raises:
        models.market_probability.ClosingLineInFitError: a post-lock line reached the corpus.
        models.blending_data.DuplicateTuningRowError: a game carries two corpus rows.
    """
    cutoff = _as_utc(generation_cutoff)
    root = Path(silver_dir)
    historical = load_tuning_period_data(root)
    current = _season_2026_corpus(root, games, cutoff)

    frame = pd.concat(
        [part for part in (historical.frame, current.frame) if not part.empty],
        ignore_index=True,
    ).sort_values("game_id", ignore_index=True)
    assert_one_row_per_game(frame)
    excluded_parts = [p for p in (historical.excluded, current.excluded) if not p.empty]
    excluded = (
        pd.concat(excluded_parts, ignore_index=True).sort_values(
            "game_id", ignore_index=True
        )
        if excluded_parts
        else _empty(EXCLUDED_COLUMNS)
    )
    assert_corpus_is_prelock(frame, games)
    return PrelockTuningCorpus(
        frame=frame,
        excluded=excluded,
        unjoined_timeline_ids=historical.unjoined_timeline_ids,
    )


def extended_slopes(converter: Mapping[str, Any]) -> dict[int, float]:
    """The converter's prior-only ``walk_forward_slopes`` plus 2026 priced by ``slope_beta`` (D-30).

    ``slope_beta`` of ``market_probability_20260923_195443`` was fitted on 2020-2024, strictly
    before 2026, so it is prior-only for every 2026 row. A 2020 row stays without a slope and is
    excluded as ``no_prior_fold_converter``.

    Raises:
        RemeasureRefusal: the converter already carries a 2026 walk-forward slope (it would then
            have been fitted on 2026 games).
    """
    slopes = {
        int(season): float(slope)
        for season, slope in converter["walk_forward_slopes"].items()
    }
    if REFIT_SEASON in slopes:
        msg = (
            f"the converter carries a {REFIT_SEASON} walk-forward slope, so it was fitted on "
            f"{REFIT_SEASON} games; the record prices {REFIT_SEASON} by a slope fitted before it"
        )
        raise RemeasureRefusal(msg)
    slopes[REFIT_SEASON] = float(converter["slope_beta"])
    return slopes


# ---------------------------------------------------------------------------
# The bet-rule numbers (record section 9: lean, error spread, WP gate, edge bands, floors)
# ---------------------------------------------------------------------------


def _refuse_outside(frame: pd.DataFrame, allowed: tuple[int, ...], what: str) -> None:
    """The frozen season refusal (a 2025 row named as the spent hold), as a re-measure refusal."""
    try:
        refuse_seasons_outside(frame, allowed, what)
    except DerivationWindowError as refusal:
        raise RemeasureRefusal(str(refusal)) from refusal


def _narrow(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """*frame* reduced to the prediction columns, ids as text, one row per game."""
    missing = [column for column in _PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"the {target} predictions are missing {missing}"
        raise RemeasureRefusal(msg)
    narrowed = frame.loc[:, list(_PREDICTION_COLUMNS)].copy()
    narrowed["game_id"] = narrowed["game_id"].astype(str)
    narrowed["season"] = narrowed["season"].astype(int)
    repeated = sorted(narrowed.loc[narrowed["game_id"].duplicated(), "game_id"])
    if repeated:
        msg = f"the {target} predictions repeat {repeated[:10]}"
        raise RemeasureRefusal(msg)
    return narrowed


def fit_frame(
    target: str,
    predictions: Mapping[str, pd.DataFrame],
    corpus_game_ids: Collection[str],
) -> pd.DataFrame:
    """The residual rows the frozen procedure pools for *target*: the D-27 / D-28 population.

    The population is exactly the frozen record's ``LEAN_POOL_RULE``
    (``backtest.refit_preregistration_2026``), as the owner's ruling A of 2026-10-20 clarified
    D-27 (37-CONTEXT.md): every game of the seed seasons 2017-2019, plus ONLY the 2020-2024 and
    finished 2026 games that have an owned pre-lock line (the corpus; the filter of
    ``scripts/derive_corrected_ev_chain.py::fit_frame_for``). A 2020-2024 or 2026 game without
    an owned pre-lock line enters neither the lean nor the error spread, exactly as today.

    Returns:
        ``game_id``, ``season``, the model column as the frozen code names it (``model_prob`` /
        ``model_spread`` / ``model_total``) and ``actual``, sorted by (season, game_id).

    Raises:
        RemeasureRefusal: a prediction from a season outside 2017-2024 and 2026 (a 2025 row is
            named as the spent hold).
    """
    preds = _narrow(predictions[target], target)
    _refuse_outside(
        preds,
        (*BIAS_SEED_SEASONS, *HISTORICAL_TUNE_SEASONS, REFIT_SEASON),
        f"the {target} predictions",
    )
    corpus_ids = {str(game_id) for game_id in corpus_game_ids}
    keep = preds["season"].isin(BIAS_SEED_SEASONS) | preds["game_id"].isin(
        sorted(corpus_ids)
    )
    return (
        preds.loc[keep]
        .rename(columns={"prediction": _MODEL_COLUMN[target]})
        .sort_values(["season", "game_id"], ignore_index=True)
    )


@dataclass(frozen=True)
class BetRuleNumbers:
    """One generation's re-measured bet-rule numbers, per target.

    Attributes:
        chain_fits: The frozen ``_fit_target_on_tune`` result per target (its
            ``season_bias_by_season`` is the PRIOR-ONLY series the error spread used).
        calibration_gate_passed: The D-29 WP gate verdict; None for ATS and O/U.
        fallback_trigger: The WP gate's recorded trigger when it failed; otherwise None.
        lean_by_target: The D-27 lean: the pooled mean residual of the whole pool.
        season_bias_by_season: The bet rule's bias per season: the prior-only series with 2026
            set to the D-27 lean, the value applied to 2026 bets.
        sd_correction_bias_by_season: The prior-only series as the error spread consumed it
            (2026 = the 2017-2024 lean), kept as evidence.
        edge_tier_thresholds: ``(high, medium)`` per target (D-36).
        ev_floors: The record's fixed EV floors.
        windows: The seasons each number was measured on, per target.
        excluded_counts: ``no_prelock_line`` and ``no_prior_fold_converter`` game counts.
        rows_by_target: ``usable`` candidate rows, ``residual_pool`` rows and ``edge_rows``.
        edges: The per-game edge frame the bands were derived from (evidence; not recorded).
    """

    chain_fits: dict[str, ChainFit]
    calibration_gate_passed: dict[str, bool | None]
    fallback_trigger: dict[str, str | None]
    lean_by_target: dict[str, float]
    season_bias_by_season: dict[str, dict[int, float]]
    sd_correction_bias_by_season: dict[str, dict[int, float]]
    edge_tier_thresholds: dict[str, tuple[float, float]]
    ev_floors: dict[str, float]
    windows: dict[str, dict[str, list[int]]]
    excluded_counts: dict[str, int]
    rows_by_target: dict[str, dict[str, int]]
    edges: pd.DataFrame


def _edge_frame(
    corpus_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    slopes: Mapping[int, float],
) -> pd.DataFrame:
    """Every corpus game's three edges, by the frozen formulas of ``build_prelock_edge_frame``.

    ``build_prelock_edge_frame`` itself refuses any season outside 2020-2024, so its formulas
    are applied here to the extended corpus: ATS ``model_spread - market_spread``; O/U
    ``(model_total - market_total) / max(market_total, OU_TOTAL_FLOOR)``; WP ``model_prob -
    market probability``, the market converted through ``oof_market_probability`` with each
    season's prior-only slope (2026: ``slope_beta``) and NaN where a season has none.
    """
    frame = corpus_frame.loc[
        :, ["game_id", "season", "market_spread", "market_total"]
    ].copy()
    for target in _TARGETS:
        frame = frame.merge(
            predictions[target]
            .loc[:, ["game_id", "prediction"]]
            .rename(columns={"prediction": _MODEL_COLUMN[target]}),
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    frame["ats_edge"] = frame["model_spread"] - frame["market_spread"]
    frame["ou_edge"] = (frame["model_total"] - frame["market_total"]) / frame[
        "market_total"
    ].clip(lower=OU_TOTAL_FLOOR)
    covered = frame["season"].isin(sorted(slopes))
    frame["wp_market_prob"] = np.nan
    if bool(covered.any()):
        converted = frame.loc[covered]
        frame.loc[covered, "wp_market_prob"] = oof_market_probability(
            converted.assign(home_fav_margin=converted["market_spread"]),
            {int(season): float(slope) for season, slope in slopes.items()},
        )
    frame["wp_edge"] = frame["model_prob"] - frame["wp_market_prob"]
    return frame.sort_values("game_id", ignore_index=True)


def measure_bet_rule(
    predictions: Mapping[str, pd.DataFrame],
    corpus: PrelockTuningCorpus,
    slopes: Mapping[int, float],
    *,
    current_thresholds: Mapping[str, tuple[float, float]],
) -> BetRuleNumbers:
    """Re-measure one generation's bet rule through the frozen estimators (record section 9).

    Args:
        predictions: :func:`walk_forward_predictions` output (``game_id``, ``season``,
            ``prediction``, ``actual`` per target).
        corpus: :func:`extended_corpus`.
        slopes: :func:`extended_slopes` of the bound converter.
        current_thresholds: The edge-tier pairs in force, for the frozen derivation's
            label-movement report.

    Returns:
        The :class:`BetRuleNumbers`. The EV floors are the record's ``EV_FLOORS``; nothing is
        swept.

    Raises:
        RemeasureRefusal: a corpus game without a prediction, a corpus or prediction season
            outside the record's windows, fewer than ``MIN_USABLE_REMEASURE_ROWS`` usable rows
            for any target, an under-filled WP gate, or a threshold the frozen derivation refuses.
    """
    corpus_frame = corpus.frame.copy()
    corpus_frame["game_id"] = corpus_frame["game_id"].astype(str)
    corpus_frame["season"] = corpus_frame["season"].astype(int)
    _refuse_outside(
        corpus_frame, (*HISTORICAL_TUNE_SEASONS, REFIT_SEASON), "the extended corpus"
    )
    narrowed = {target: _narrow(predictions[target], target) for target in _TARGETS}
    for target in _TARGETS:
        unpredicted = sorted(
            set(corpus_frame["game_id"]) - set(narrowed[target]["game_id"])
        )
        if unpredicted:
            msg = (
                f"{len(unpredicted)} corpus game(s) have no {target} walk-forward prediction, "
                f"e.g. {unpredicted[:10]}; refusing rather than dropping them uncounted"
            )
            raise RemeasureRefusal(msg)

    slope_map = {int(season): float(slope) for season, slope in slopes.items()}
    covered = corpus_frame["season"].isin(sorted(slope_map))
    usable = {
        "wp": int(covered.sum()),
        "ats": len(corpus_frame),
        "ou": len(corpus_frame),
    }
    short = {target: n for target, n in usable.items() if n < MIN_USABLE_REMEASURE_ROWS}
    if short:
        named = ", ".join(f"{target} {n}" for target, n in short.items())
        msg = (
            f"fewer than {MIN_USABLE_REMEASURE_ROWS} usable re-measure rows ({named}); a "
            "generation with too few rows is refused, never installed as a no-bets generation"
        )
        raise RemeasureRefusal(msg)

    candidate_seasons = {
        "wp": sorted({int(s) for s in corpus_frame.loc[covered, "season"]}),
        "ats": sorted({int(s) for s in corpus_frame["season"]}),
        "ou": sorted({int(s) for s in corpus_frame["season"]}),
    }
    corpus_ids = frozenset(corpus_frame["game_id"])

    chain_fits: dict[str, ChainFit] = {}
    gate_passed: dict[str, bool | None] = {}
    triggers: dict[str, str | None] = {}
    leans: dict[str, float] = {}
    bet_rule_bias: dict[str, dict[int, float]] = {}
    prior_only_bias: dict[str, dict[int, float]] = {}
    windows: dict[str, dict[str, list[int]]] = {}
    pool_rows: dict[str, int] = {}
    for target in _TARGETS:
        pool = fit_frame(target, narrowed, corpus_ids)
        window = FenceWindow(
            tune_seasons=tuple(candidate_seasons[target]),
            hold_seasons=(),
            prior_residual_seasons=BIAS_SEED_SEASONS,
            threshold_window=REMEASURE_WINDOW_LABEL,
            label=REMEASURE_WINDOW_LABEL,
            is_the_preregistered_rule=False,
        )
        try:
            chain_fit, gate = _fit_target_on_tune(target, pool, window)
        except ValueError as exc:
            msg = f"the frozen {target} fit could not be completed: {exc}"
            raise RemeasureRefusal(msg) from exc

        residuals = pool["actual"].to_numpy(dtype=float) - pool[
            _MODEL_COLUMN[target]
        ].to_numpy(dtype=float)
        lean = float(np.mean(residuals))
        prior_only = {
            int(season): float(bias)
            for season, bias in sorted(chain_fit.season_bias_by_season.items())
        }
        chain_fits[target] = chain_fit
        gate_passed[target] = None if gate is None else bool(gate.passed)
        triggers[target] = None if gate is None else gate.fallback_trigger
        leans[target] = lean
        prior_only_bias[target] = prior_only
        bet_rule_bias[target] = {**prior_only, REFIT_SEASON: lean}
        pool_rows[target] = len(pool)
        windows[target] = {
            "candidate_seasons": list(candidate_seasons[target]),
            "bias_seed_seasons": list(BIAS_SEED_SEASONS),
            "tune_fit_seasons": [int(s) for s in chain_fit.tune_fit_seasons],
            "lean_pool_seasons": sorted({int(s) for s in pool["season"]}),
        }

    edges = _edge_frame(corpus_frame, narrowed, slope_map)
    derivation = derive_threshold_pairs(edges, current_thresholds)
    refused = {
        target: derivation.refusals.get(target, "no pair derived")
        for target in _TARGETS
        if derivation.thresholds.get(target) is None
    }
    if refused:
        msg = f"the frozen edge-band derivation refused {sorted(refused)}: {refused}"
        raise RemeasureRefusal(msg)
    thresholds = {
        target: cast("tuple[float, float]", derivation.thresholds[target])
        for target in _TARGETS
    }

    return BetRuleNumbers(
        chain_fits=chain_fits,
        calibration_gate_passed=gate_passed,
        fallback_trigger=triggers,
        lean_by_target=leans,
        season_bias_by_season=bet_rule_bias,
        sd_correction_bias_by_season=prior_only_bias,
        edge_tier_thresholds={
            target: (float(pair[0]), float(pair[1]))
            for target, pair in thresholds.items()
        },
        ev_floors=dict(EV_FLOORS),
        windows=windows,
        excluded_counts={
            EXCLUSION_NO_PRELOCK_LINE: len(corpus.excluded),
            EXCLUSION_NO_PRIOR_FOLD_CONVERTER: int((~covered).sum()),
        },
        rows_by_target={
            target: {
                "usable": usable[target],
                "residual_pool": pool_rows[target],
                "edge_rows": int(derivation.eligible_rows[target]),
            }
            for target in _TARGETS
        },
        edges=edges,
    )


def _by_season_text(values: Mapping[int, float]) -> dict[str, float]:
    return {str(season): float(value) for season, value in sorted(values.items())}


def bet_rule_payload(
    numbers: BetRuleNumbers,
    *,
    artifact_ids: Mapping[str, str],
    cutoff_utc: datetime,
    list_week: int,
    date_index: int,
    blend_weights: Mapping[str, float],
) -> dict[str, Any]:
    """The generation's bet-rule record, ready for ``refit.records.write_bet_rule_record``.

    ``tune_fit`` and ``targets`` are in the shape ``backtest.weekly_bet_list.load_frozen_chain_fit``
    reads (``ev_floor_t``, ``frozen_sd``, ``season_bias_by_season`` keyed by season text and
    including 2026, ``calibration_gate_passed``; ``fallback_trigger``). The record names its
    artifacts under ``artifact_ids`` and its recipe under ``refit_recipe``, never under a ledger
    stamp column's name.

    Raises:
        ValueError: *artifact_ids* lacks one of wp, ats, ou, blend, converter, or
            *blend_weights* lacks a target.
    """
    missing = [key for key in _RECORD_ARTIFACT_KEYS if key not in artifact_ids]
    if missing:
        msg = f"the bet-rule record needs artifact ids for {missing}"
        raise ValueError(msg)
    missing_weights = [target for target in _TARGETS if target not in blend_weights]
    if missing_weights:
        msg = f"the bet-rule record needs blend weights for {missing_weights}"
        raise ValueError(msg)

    tune_fit: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for target in _TARGETS:
        frozen_sd = numbers.chain_fits[target].frozen_sd
        tune_fit[target] = {
            "ev_floor_t": float(numbers.ev_floors[target]),
            "frozen_sd": None if frozen_sd is None else float(frozen_sd),
            "season_bias_by_season": _by_season_text(
                numbers.season_bias_by_season[target]
            ),
            "sd_correction_bias_by_season": _by_season_text(
                numbers.sd_correction_bias_by_season[target]
            ),
            "lean": float(numbers.lean_by_target[target]),
            "calibration_gate_passed": numbers.calibration_gate_passed[target],
        }
        targets[target] = {"fallback_trigger": numbers.fallback_trigger[target]}

    return {
        "record_kind": BET_RULE_RECORD_KIND,
        "refit_recipe": REFIT_RECIPE_ID,
        "artifact_ids": {key: str(artifact_ids[key]) for key in _RECORD_ARTIFACT_KEYS},
        "cutoff_utc": _as_utc(cutoff_utc).isoformat(),
        "list_week": int(list_week),
        "date_index": int(date_index),
        "tune_fit": tune_fit,
        "targets": targets,
        "edge_tier_thresholds": {
            target: [float(high), float(medium)]
            for target, (high, medium) in numbers.edge_tier_thresholds.items()
        },
        "blend_weights": {target: float(blend_weights[target]) for target in _TARGETS},
        "windows": numbers.windows,
        "excluded_counts": dict(numbers.excluded_counts),
        "rows_by_target": numbers.rows_by_target,
        "preregistration_sha256": dict(PREREGISTRATION_SHA256),
    }
