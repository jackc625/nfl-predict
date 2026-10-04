"""Re-derive the EV floor and the frozen residual SD on the corrected models (Plan 33.2-29, D33.2-25).

Run via: uv run python -m scripts.derive_corrected_ev_chain

WHY
---
The per-target EV floor (the number that decides whether a bet is placed at all) and the frozen
residual SD (the scale that turns a model's miss into a bet's expected value) were chosen in Phase
31 against CLOSING-line outcomes on models this phase replaced. D33.2-25 rules them dead with those
models. This script re-derives both on the three corrected models and on the lines we owned before
each game's lock, and publishes the result as a SUPERSEDING correction: the frozen ``ee20773``
pre-registration (``backtest/ev_chain_constants.py`` + ``PROFITABILITY-PREREGISTRATION.md``) is
never edited, and the spent 2025 run ledger is read-only here.

WHAT IS SWAPPED AND WHAT IS HELD
--------------------------------
Swapped: the models (the three corrected artifacts now in production), the market source (the
owned ``odds_timeline`` line at or before each game's lock, never a closing line), and the corpus
(2020-2024). Held, and REUSED rather than re-implemented: the frozen ``EV_FLOOR_GRID``, the
ROI-best choice over it, the tune-only fit (``backtest.profitability_2025._fit_target_on_tune``:
the walk-forward prior-season bias and ``fit_frozen_residual_sd``), the single bet-decision path
(``BetSelector`` through ``_select``) and the band edges.

ONE DELIBERATE DIVERGENCE: the Phase-31 sweep froze ``EV_FLOOR_GRID[0]`` (0.00, the most
permissive value) when no floor admitted a bet. Here that case RAISES
:class:`InsufficientHonestDataForFloorError` and the target's floor is recorded as ``None`` -- no
honest floor, therefore no bets. A grid floor inherited without evidence looks like a threshold and
is a relic. No numeric stand-in is used: ``float('inf')`` is refused by name inside
``backtest.ev_chain_constants.assign_ev_tier`` and ``0.0`` is a real grid value that admits
everything.

THE WINDOW
----------
2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any price and
2025 has no free pre-lock source. The single-use 2025 hold stays spent: no 2025 row enters any fit
(a named refusal), and the run ledger's bytes are compared before and after.

The first season of each target's walk-forward bias needs a strictly-prior residual pool, exactly
as the Phase-31 recipe seeded its first tune season from the three seasons before it. The seed here
is the corrected models' own walk-forward residuals for 2017-2019 -- model misses only, no betting
line of any kind -- and those seasons feed ONLY the bias estimate: never a candidate, never the
frozen SD, never the floor sweep.

WP: the market side of every historical row is converted OUT OF FOLD through
``models.market_probability.oof_market_probability`` with the converter the live blend binds, never
through the serving slope. 2020 has no prior fold, so its games leave the WP sweep as
``no_prior_fold_converter`` (counted, never filled). The owned timeline carries no moneyline, so a
WP candidate is priced from that out-of-fold probability WITH the bookmaker's cut put back on
(:data:`WP_WIN_PAYOUT_FACTOR`): pricing it at the no-vig line would pay every winning bet more
than any book does and flatter the sweep. WP fits no residual SD by design (D31-07).

WHAT IT WRITES
--------------
``outputs/p332/corrected_chain_fit.json`` (the run record, in the shape
``backtest.weekly_bet_list.load_frozen_chain_fit`` reads, plus provenance) and
``backtest/corrected_ev_chain_constants.py`` (the committed superseding module). It does NOT touch
``artifacts/latest.json`` or ``data/``, and it does NOT repoint the live reader: Plan 33.2-26 Task 3
repoints ``DEFAULT_CHAIN_FIT_PATH`` together with the cold-start bias import in one commit.

THE CORRECTION SELECTOR (quick task 261003-vke, WINDOWS row 19)
---------------------------------------------------------------
``--correction`` picks WHICH superseding correction a run measures: its inputs (the served ids,
the re-fit, the gold generation and the converter, as ``tests.phase33_state`` slots), its output
paths, the record it supersedes and the ruling text it quotes (:class:`BetRuleCorrection`). The
default, ``p332_25b``, is the 33.2 correction above, every path and text unchanged, so the
8c9675e record stays reproducible. ``row19`` re-measures on the models re-fitted after the
neutral-site Elo fix (owner ruling 2026-10-03, "Re-measure, same recipe"). The RECIPE is not a
selector field: windows, seed, grid, estimators, thread pin and refusals are shared.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.ats_ev_chain import ChainFit, FenceWindow
from backtest.ev_chain_constants import HOLD_SEASONS_P31, RUN_LEDGER_PATH
from backtest.ou_ev_chain import EV_FLOOR_GRID, MINUS_110_PAYOUT
from backtest.profitability_2025 import (
    _fit_target_on_tune,
    _per_bet_frame,
    _select,
    _strategy_for_target,
)
from backtest.roi_significance import flat_roi
from backtest.wp_ev_chain import WPGateResult
from config.tuning_preregistration import PINNED_THREAD_COUNT
from models.blending_data import (
    EXCLUDED_COLUMNS,
    EXCLUSION_NO_PRELOCK_LINE,
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
    assert_one_row_per_game,
)
from models.market_probability import assert_no_closing_line, oof_market_probability
from utils.probability_utils import probability_to_moneyline

TARGETS: tuple[str, str, str] = ("wp", "ats", "ou")

#: The owned pre-lock corpus seasons -- the whole honest corpus (see the module docstring).
DERIVATION_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

#: The strictly-prior residual seed for the first derivation season's bias ONLY.
BIAS_SEED_SEASONS: tuple[int, ...] = (2017, 2018, 2019)

WINDOW_REASON: str = (
    "2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any "
    "price and 2025 has no free pre-lock source. The single-use 2025 hold is spent and is not "
    "re-read."
)

THRESHOLD_WINDOW_LABEL: str = "corrected_prelock_2020_2024"

WP_NO_FROZEN_SD_REASON: str = (
    "D31-07: WP fits no residual SD by design. A calibrated classifier has no residual to take "
    "a standard deviation of, and inventing a logit-space one was rejected "
    "(backtest/wp_ev_chain.py)."
)

#: The bookmaker's cut put back on a historical WP price: a winning WP bet pays 100/110 of what
#: the no-vig price would pay -- the standard -110 cut (a 0.0476 two-way hold at even odds), the
#: same price the ATS and O/U legs are graded at here (``backtest.ats_ev_chain`` and
#: ``backtest.ou_ev_chain.devig`` fall back to flat -110 when no juice is stored, which is every
#: row of the owned timeline). No live PRE-LOCK moneyline is stored in silver --
#: ``odds_snapshot``'s moneylines are the nflverse backfill -- so there is no owned measurement to
#: use instead. Cross-check: the live Odds API captures in bronze (14 games, 9 books, all
#: pre-lock, 2025 week 5) show a median per-game hold of 0.042, so -110 is realistic and slightly
#: conservative. Those 2025 quotes are NOT read here (the hold is spent).
WP_WIN_PAYOUT_FACTOR: float = MINUS_110_PAYOUT

RECORD_PATH: Path = Path("outputs") / "p332" / "corrected_chain_fit.json"
MODULE_PATH: Path = Path("backtest") / "corrected_ev_chain_constants.py"
PHASE31_RECORD_PATH: Path = Path("outputs") / "p31" / "profitability_2025_verdict.json"
LATEST_MANIFEST_PATH: Path = Path("artifacts") / "latest.json"

_MODEL_COLUMN: Mapping[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}
_PREDICTION_COLUMNS: tuple[str, ...] = ("game_id", "season", "prediction", "actual")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class DerivationWindowError(Exception):
    """A row from outside the derivation window reached a fit.

    Inherits ``Exception`` rather than ``ValueError`` so none of the repository's
    degrade-quietly ``(ValueError, KeyError, ...)`` catch tuples can turn it into "fit anyway".
    """


class SpentHoldSeasonError(DerivationWindowError):
    """A row from the SPENT 2025 hold reached a fit. The hold is single-use and stays spent."""


class InsufficientHonestDataForFloorError(Exception):
    """No value on the frozen grid admits a single bet on the honest pool, so there is no floor.

    The Phase-31 recipe froze ``EV_FLOOR_GRID[0]`` in this case; this derivation refuses instead
    and records ``None``, so the target places no bets. Like
    ``data.sealed_probe_log.SealedProbeLogCorrupt`` it inherits ``Exception`` directly, NOT
    ``ValueError`` / ``KeyError`` / ``RuntimeError`` / ``LookupError``: the selection path and the
    weekly bet list catch those tuples for absent inputs, and a refused floor degraded into "use
    a default floor" is exactly the relic this refusal replaces.
    """


# ---------------------------------------------------------------------------
# The candidate frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateFrames:
    """Each target's sweep candidates, and the WP rows the converter cannot price.

    Attributes:
        frames: ``{wp, ats, ou}`` -> one row per game, in the columns the strategies read. The
            strategies' market field names (``closing_spread``, ``closing_total``, ``ml_home``,
            ``ml_away``) are their LEGACY names; here they hold the owned PRE-LOCK line and the
            moneyline implied by the out-of-fold market probability plus the -110 cut. The
            closing-line guard runs on the inputs BEFORE this rename.
        wp_excluded: The ``no_prior_fold_converter`` rows (:data:`EXCLUDED_COLUMNS`).
        corpus_game_ids: The owned pre-lock corpus games.
        closing_line_rows: Post-lock or undated line rows found in the inputs (always 0: the
            guard raises on any).
    """

    frames: dict[str, pd.DataFrame]
    wp_excluded: pd.DataFrame
    corpus_game_ids: frozenset[str]
    closing_line_rows: int


def refuse_seasons_outside(
    frame: pd.DataFrame, allowed: tuple[int, ...], what: str
) -> None:
    """Refuse a frame carrying a season outside *allowed*; a 2025 row is named as the spent hold.

    Raises:
        SpentHoldSeasonError: when any row is from the spent hold (``HOLD_SEASONS_P31``).
        DerivationWindowError: when any other row falls outside *allowed*.
    """
    seasons = {int(season) for season in frame["season"]}
    spent = sorted(seasons & set(HOLD_SEASONS_P31))
    if spent:
        msg = (
            f"{what} carries season(s) {spent}: the single-use 2025 hold is SPENT "
            f"({RUN_LEDGER_PATH}, state completed, no force flag) and no row of it may enter "
            "any fit."
        )
        raise SpentHoldSeasonError(msg)
    outside = sorted(seasons - set(allowed))
    if outside:
        msg = f"{what} carries season(s) {outside} outside the derivation window {list(allowed)}"
        raise DerivationWindowError(msg)


def refuse_closing_lines(inputs: pd.DataFrame) -> int:
    """Refuse a closing line in the fit inputs, by column or by timing. Returns the count (0).

    Delegates to ``models.market_probability.assert_no_closing_line`` -- the one closing-line
    refusal -- which rejects a closing-line column and any row timed after its own lock.
    """
    assert_no_closing_line(inputs)
    snapshot = pd.to_datetime(inputs["snapshot_ts"], utc=True)
    lock = pd.to_datetime(inputs["lock"], utc=True)
    return int((snapshot > lock).sum() + (snapshot.isna() | lock.isna()).sum())


def vigged_moneyline(
    fair_probability: float, payout_factor: float = WP_WIN_PAYOUT_FACTOR
) -> int:
    """The American moneyline a book would post for *fair_probability*, cut included.

    A win pays *payout_factor* of the no-vig profit, so at 0.5 both sides post -110. The cut is
    taken from the winnings rather than added to the probability, so every probability below 1
    still has a posted price (a proportional mark-up would exceed 1 on a heavy favourite).
    """
    fair_profit = (1.0 - fair_probability) / fair_probability
    return probability_to_moneyline(1.0 / (1.0 + fair_profit * payout_factor))


def _narrow_predictions(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    missing = [column for column in _PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"the {target} predictions are missing {missing}"
        raise ValueError(msg)
    narrowed = frame.loc[:, list(_PREDICTION_COLUMNS)].copy()
    narrowed["game_id"] = narrowed["game_id"].astype(str)
    narrowed["season"] = narrowed["season"].astype(int)
    assert_one_row_per_game(narrowed)
    return narrowed


def build_candidate_frames(
    corpus_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    walk_forward_slopes: Mapping[int | str, float],
) -> CandidateFrames:
    """Join the owned pre-lock corpus to each corrected model's walk-forward predictions.

    Args:
        corpus_frame: ``models.blending_data.PrelockTuningCorpus.frame`` (one row per game with
            a line at or before its lock; ``market_spread`` on the home-margin scale).
        predictions: ``{wp, ats, ou}`` -> ``game_id``, ``season``, ``prediction``, ``actual``.
        walk_forward_slopes: The bound converter's prior-only slopes.

    Raises:
        SpentHoldSeasonError / DerivationWindowError: a corpus row outside 2020-2024.
        models.market_probability.ClosingLineInFitError: a closing line reached the inputs.
        ValueError: a corpus game has no prediction for some target.
    """
    refuse_seasons_outside(
        corpus_frame, DERIVATION_SEASONS, "the owned pre-lock corpus"
    )
    corpus = corpus_frame.copy()
    corpus["game_id"] = corpus["game_id"].astype(str)
    corpus["season"] = corpus["season"].astype(int)

    narrowed = {
        target: _narrow_predictions(predictions[target], target) for target in TARGETS
    }
    for target in TARGETS:
        missing = sorted(set(corpus["game_id"]) - set(narrowed[target]["game_id"]))
        if missing:
            msg = (
                f"{len(missing)} owned pre-lock game(s) have no {target} walk-forward "
                f"prediction, e.g. {missing[:10]}; refusing rather than dropping them uncounted."
            )
            raise ValueError(msg)

    home_win = narrowed["wp"].set_index("game_id")["actual"]
    inputs = corpus.assign(
        home_fav_margin=corpus["market_spread"],
        home_win=corpus["game_id"].map(home_win),
    )
    closing_line_rows = refuse_closing_lines(inputs)

    base = corpus.loc[:, ["game_id", "season", "week", "market_spread", "market_total"]]
    frames: dict[str, pd.DataFrame] = {}
    joined = {
        target: base.merge(
            narrowed[target].drop(columns="season"), on="game_id", how="inner"
        )
        for target in TARGETS
    }

    # THE LEGACY FIELD NAMES: the strategies read ``closing_spread`` / ``closing_total``; the
    # values are the owned PRE-LOCK line, already cleared by the guard above.
    frames["ats"] = (
        joined["ats"]
        .rename(columns={"prediction": "model_spread"})
        .assign(closing_spread=joined["ats"]["market_spread"], target="ats")
    )
    frames["ou"] = (
        joined["ou"]
        .rename(columns={"prediction": "model_total"})
        .assign(closing_total=joined["ou"]["market_total"], target="ou")
    )

    wp = joined["wp"].rename(columns={"prediction": "model_prob"})
    covered = {int(season) for season in walk_forward_slopes}
    convertible = wp["season"].isin(covered)
    priced = wp.loc[convertible].copy()
    priced["market_prob"] = oof_market_probability(
        priced.assign(home_fav_margin=priced["market_spread"]), walk_forward_slopes
    )
    priced["ml_home"] = [vigged_moneyline(q) for q in priced["market_prob"]]
    priced["ml_away"] = [vigged_moneyline(1.0 - q) for q in priced["market_prob"]]
    priced["target"] = "wp"
    frames["wp"] = priced

    dropped = wp.loc[~convertible].copy()
    dropped["game_type"] = (
        corpus.set_index("game_id").loc[dropped["game_id"], "game_type"].to_numpy()
    )
    dropped["reason"] = EXCLUSION_NO_PRIOR_FOLD_CONVERTER
    dropped["detail"] = [
        f"season {int(season)} has no prior-fold converter slope"
        for season in dropped["season"]
    ]
    wp_excluded = dropped.loc[:, list(EXCLUDED_COLUMNS)].reset_index(drop=True)

    return CandidateFrames(
        frames={
            t: frames[t].sort_values("game_id", ignore_index=True) for t in TARGETS
        },
        wp_excluded=wp_excluded,
        corpus_game_ids=frozenset(corpus["game_id"]),
        closing_line_rows=closing_line_rows,
    )


def fit_frame_for(
    target: str,
    predictions: Mapping[str, pd.DataFrame],
    frames: CandidateFrames,
) -> pd.DataFrame:
    """The rows one target's tune-only fit reads: the bias seed plus every corpus game.

    Seed-season rows (:data:`BIAS_SEED_SEASONS`) feed only the walk-forward bias; the corpus
    rows are the games with an owned pre-lock line. Model residuals only -- no line enters.

    Raises:
        SpentHoldSeasonError / DerivationWindowError: a prediction outside seed + corpus seasons.
    """
    refuse_seasons_outside(
        predictions[target],
        (*BIAS_SEED_SEASONS, *DERIVATION_SEASONS),
        f"the {target} predictions",
    )
    preds = _narrow_predictions(predictions[target], target)
    keep = preds["season"].isin(BIAS_SEED_SEASONS) | preds["game_id"].isin(
        frames.corpus_game_ids
    )
    return (
        preds.loc[keep]
        .rename(columns={"prediction": _MODEL_COLUMN[target]})
        .sort_values(["season", "game_id"], ignore_index=True)
    )


# ---------------------------------------------------------------------------
# The fit and the sweep (the Phase-31 recipe, reused)
# ---------------------------------------------------------------------------


def fit_target(
    target: str,
    fit_frame: pd.DataFrame,
    tune_seasons: tuple[int, ...] = DERIVATION_SEASONS,
) -> tuple[ChainFit, WPGateResult | None]:
    """The tune-only fit through the Phase-31 ``_fit_target_on_tune``, on the corrected window.

    The window names NO hold season (the spent 2025 hold is never read) and the bias seed as its
    prior-residual seasons.
    """
    window = FenceWindow(
        tune_seasons=tuple(tune_seasons),
        hold_seasons=(),
        prior_residual_seasons=BIAS_SEED_SEASONS,
        threshold_window=THRESHOLD_WINDOW_LABEL,
        label=THRESHOLD_WINDOW_LABEL,
        is_the_preregistered_rule=False,
    )
    return _fit_target_on_tune(target, fit_frame, window)


@dataclass(frozen=True)
class FloorSweep:
    """One target's sweep over the frozen grid: the ROI-best floor (or None) and every cell."""

    target: str
    best_t: float | None
    best_roi: float | None
    rows: list[dict[str, Any]] = field(default_factory=list)


def sweep_ev_floor(
    target: str,
    candidates: pd.DataFrame,
    chain_fit: ChainFit,
    gate: WPGateResult | None,
) -> FloorSweep:
    """Sweep the frozen ``EV_FLOOR_GRID`` and keep the ROI-best floor that admits a bet.

    The Phase-31 loop (``backtest.profitability_2025._measure_and_judge``) with its selection
    path, its per-bet frame and its strict ``roi > best_roi`` choice -- ties keep the lower floor.
    """
    strategy = _strategy_for_target(target, chain_fit, gate)
    rows: list[dict[str, Any]] = []
    best_t: float | None = None
    best_roi: float | None = None
    for floor in EV_FLOOR_GRID:
        selection = _select(
            candidates,
            [strategy],
            float(floor),
            frozen_sd=float(chain_fit.frozen_sd or 1.0),
        )
        per_bet = _per_bet_frame(selection.selected)
        roi = None if per_bet.empty else flat_roi(per_bet)
        rows.append(
            {
                "threshold": float(floor),
                "bet_count": len(selection.selected),
                "roi": roi,
            }
        )
        if (
            len(selection.selected)
            and roi is not None
            and (best_roi is None or roi > best_roi)
        ):
            best_roi = roi
            best_t = float(floor)
    return FloorSweep(target=target, best_t=best_t, best_roi=best_roi, rows=rows)


def choose_ev_floor(target: str, sweep: FloorSweep) -> float:
    """The swept floor, or a named refusal where the Phase-31 recipe fell back to the grid floor.

    Raises:
        InsufficientHonestDataForFloorError: no grid value admitted a single bet.
    """
    if sweep.best_t is None:
        counts = {row["threshold"]: row["bet_count"] for row in sweep.rows}
        msg = (
            f"[{target}] no value on the frozen EV_FLOOR_GRID {list(EV_FLOOR_GRID)} admits a "
            f"single bet on the honest pre-lock pool (bets per floor {counts}). There is no "
            "honest floor, so this target places no bets. The Phase-31 recipe would have frozen "
            f"EV_FLOOR_GRID[0] = {EV_FLOOR_GRID[0]} here; that is refused."
        )
        raise InsufficientHonestDataForFloorError(msg)
    return sweep.best_t


def floor_or_refusal(target: str, sweep: FloorSweep) -> tuple[float | None, str | None]:
    """``(floor, None)`` or ``(None, refusal message)`` -- the refusal is recorded, never coerced."""
    try:
        return choose_ev_floor(target, sweep), None
    except InsufficientHonestDataForFloorError as refusal:
        return None, str(refusal)


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainDerivation:
    """Everything one derivation produced, for the record, the module and the CLI."""

    floors: dict[str, float | None]
    refusals: dict[str, str]
    chain_fits: dict[str, ChainFit]
    gates: dict[str, WPGateResult | None]
    sweeps: dict[str, FloorSweep]
    candidate_seasons: dict[str, tuple[int, ...]]
    n_candidates: dict[str, int]
    wp_excluded_no_prior_fold: int
    closing_line_rows: int
    input_digest: str


def _input_digest(frames: Mapping[str, pd.DataFrame]) -> str:
    """sha256 over the rows that entered the derivation, canonically ordered and formatted."""
    digest = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name]
        ordered = frame.reindex(columns=sorted(frame.columns)).sort_values(
            ["season", "game_id"], ignore_index=True
        )
        digest.update(name.encode("utf-8"))
        digest.update(ordered.to_csv(index=False, float_format="%.12g").encode("utf-8"))
    return digest.hexdigest()


def derive_chain(
    corpus_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    walk_forward_slopes: Mapping[int | str, float],
) -> ChainDerivation:
    """Fit, sweep and choose (or refuse) each target's floor and frozen SD. Pure: no I/O."""
    frames = build_candidate_frames(corpus_frame, predictions, walk_forward_slopes)

    floors: dict[str, float | None] = {}
    refusals: dict[str, str] = {}
    chain_fits: dict[str, ChainFit] = {}
    gates: dict[str, WPGateResult | None] = {}
    sweeps: dict[str, FloorSweep] = {}
    candidate_seasons: dict[str, tuple[int, ...]] = {}
    digest_inputs: dict[str, pd.DataFrame] = {}

    for target in TARGETS:
        candidates = frames.frames[target]
        seasons = tuple(sorted({int(season) for season in candidates["season"]}))
        fit_frame = fit_frame_for(target, predictions, frames)
        chain_fit, gate = fit_target(target, fit_frame, seasons)
        sweep = sweep_ev_floor(target, candidates, chain_fit, gate)
        floor, refusal = floor_or_refusal(target, sweep)

        floors[target] = floor
        if refusal is not None:
            refusals[target] = refusal
        chain_fits[target] = chain_fit
        gates[target] = gate
        sweeps[target] = sweep
        candidate_seasons[target] = seasons
        digest_inputs[f"{target}_candidates"] = candidates
        digest_inputs[f"{target}_fit"] = fit_frame

    return ChainDerivation(
        floors=floors,
        refusals=refusals,
        chain_fits=chain_fits,
        gates=gates,
        sweeps=sweeps,
        candidate_seasons=candidate_seasons,
        n_candidates={t: len(frames.frames[t]) for t in TARGETS},
        wp_excluded_no_prior_fold=len(frames.wp_excluded),
        closing_line_rows=frames.closing_line_rows,
        input_digest=_input_digest(digest_inputs),
    )


# ---------------------------------------------------------------------------
# The correction selector (quick task 261003-vke, WINDOWS row 19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetRuleCorrection:
    """WHICH superseding correction of the 2026 bet rule a derivation run measures and emits.

    ONE selector for both derivations: this script and ``scripts.derive_cold_start_constants
    --corrected``, which imports it. It names the inputs (fully qualified ``tests.*_state`` slot
    names, resolved lazily at run time exactly as the old function-local imports were), the
    output paths, the record each half supersedes, and the ruling text the renderers quote.

    The RECIPE is deliberately NOT a field: the windows, the bias seed, the floor grid, the
    estimators, WP's 0.05 / 0.02 anchor pair, the quantile method, the thread pin and the
    reproduction refusal are shared by every correction, so a selector cannot change them.

    Attributes:
        name: The ``--correction`` value.
        swap_ids_slot / refit_ids_slot / refit_gold_generation_slot / converter_slot: The served
            manifest, the re-fit the blend was tuned on, the gold those re-fits were fitted on,
            and the converter the live blend binds.
        ev_record_path / ev_module_path / ev_record_id_prefix: Where the EV-chain half writes.
        ev_superseded_record_path / ev_superseded_commit_slot / ev_superseded_short: The EV-chain
            record this correction supersedes, its commit (a slot) and its short sha (for text).
        ev_module_docstring: The EV module's docstring template (``str.format`` fields:
            ``commit``, ``window_reason``, ``seed``, ``wp_seasons``, ``record_path``).
        ev_chain_label: How the cold-start half names the EV-chain correction it continues.
        cold_module_path / cold_document_path: Where the cold-start half writes.
        cold_superseded_module / cold_superseded_short / cold_superseded_paths: The cold-start
            record this correction supersedes: its importable module (read for the old pairs and
            bias), its short sha, and its ``(document, module)`` paths.
        cold_*: The paragraphs the cold-start renderers quote. ``cold_module_supersedes``,
            ``cold_document_why``, ``cold_document_live_change`` and ``cold_document_footer`` are
            ``str.format`` templates over ``short`` (and ``full`` for the first).
    """

    name: str
    swap_ids_slot: str
    refit_ids_slot: str
    refit_gold_generation_slot: str
    converter_slot: str
    ev_record_path: Path
    ev_module_path: Path
    ev_record_id_prefix: str
    ev_superseded_record_path: Path
    ev_superseded_commit_slot: str
    ev_superseded_short: str
    ev_module_docstring: str
    ev_chain_label: str
    cold_module_path: str
    cold_document_path: str
    cold_superseded_module: str
    cold_superseded_short: str
    cold_superseded_paths: tuple[str, str]
    cold_title: str
    cold_module_supersedes: str
    cold_module_ev_half: str
    cold_source_verdict: str
    cold_document_why: str
    cold_document_bias_contrast: str
    cold_document_ev_half: str
    cold_document_live_change: str
    cold_document_footer: str

    @staticmethod
    def resolve(slot: str) -> Any:
        """A fully qualified state slot (``tests.phase33_state.NAME``), imported on demand."""
        module_name, _, name = slot.rpartition(".")
        return getattr(importlib.import_module(module_name), name)

    def swap_ids(self) -> dict[str, str]:
        return dict(self.resolve(self.swap_ids_slot))

    def refit_ids(self) -> dict[str, str]:
        return dict(self.resolve(self.refit_ids_slot))

    def refit_gold_generation(self) -> str:
        return str(self.resolve(self.refit_gold_generation_slot))

    def converter_id(self) -> str:
        return str(self.resolve(self.converter_slot))

    def ev_superseded_commit(self) -> str:
        return str(self.resolve(self.ev_superseded_commit_slot))

    def cli_flag(self) -> str:
        """The flag a reader must add to reproduce this correction ("" for the default)."""
        return "" if self.name == DEFAULT_CORRECTION else f" --correction {self.name}"


_MODULE_DOCSTRING = '''"""SUPERSEDING CORRECTION of two Phase-31 bet-decision scalars (Plan 33.2-29, D33.2-25).

GENERATED by ``scripts/derive_corrected_ev_chain.py`` -- do not edit by hand; re-run the script.

WHAT IT SUPERSEDES. Pre-registration commit ``ee20773`` ({commit}) froze the Phase-31 EV chain
in ``backtest/ev_chain_constants.py`` + ``PROFITABILITY-PREREGISTRATION.md``. The per-target EV
floor and the frozen residual SD it produced were swept against CLOSING-line outcomes on models
fitted on inputs later found defective, and those models are gone (D33.2-25, the same shape as the
48.0 cutoff D33.2-24 refused). This module SUPERSEDES those two scalars. The originals are
byte-unchanged and remain the record of what was frozen and when; nothing here edits them.

WHAT MOVED, symbol by symbol:
  * the EV floor per target -> ``CORRECTED_EV_FLOOR_BY_TARGET`` (old values beside it in
    ``SUPERSEDED_EV_FLOOR_BY_TARGET``). ``None`` means NO HONEST FLOOR COULD BE SWEPT and
    therefore NO BETS for that target (``CORRECTED_FLOOR_REFUSALS`` says why) -- never 0.0,
    which is a real grid value that admits everything, and never ``inf``, which
    ``assign_ev_tier`` refuses by name;
  * the frozen residual SD -> ``CORRECTED_FROZEN_SD_BY_TARGET`` (old values in
    ``SUPERSEDED_FROZEN_SD_BY_TARGET``); ATS and O/U only, WP none by design (D31-07);
  * the replay-snapshot policy sentence -> ``CORRECTED_REPLAY_SNAPSHOT_TS_POLICY``.

WHAT DID NOT MOVE, and is READ from the frozen modules rather than re-declared here: the grid the
floor is chosen from (``backtest.ou_ev_chain.EV_FLOOR_GRID``), the ROI-best rule over it, the
frozen-SD estimator (``fit_frozen_residual_sd``), the walk-forward bias estimator and the EV tier
band edges (``backtest.ev_chain_constants.EV_TIER_BANDS``). No uppercase name in this module
repeats one defined in either frozen module.

ONE DELIBERATE DIVERGENCE from the Phase-31 recipe: where no grid floor admits a bet, Phase 31
froze the grid floor 0.00; this correction refuses and records ``None``.

THE WINDOW. {window_reason} The bias seed ({seed}) is the corrected models' own walk-forward
residuals and feeds only the first seasons' bias. WP's first owned season has no prior-fold
converter slope, so WP is swept on {wp_seasons}.

NOT CLEAN EVIDENCE (D33.2-07): these are re-measured past seasons. They select a threshold; they
do not establish profitability. Only the 2026 season, recorded live, counts.

STAGED, NOT DEPLOYED: the live bet-list reader still resolves the Phase-31 record. Plan 33.2-26
Task 3 repoints it to ``{record_path}`` together with the cold-start bias, in one commit.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""
'''

_ROW19_MODULE_DOCSTRING = '''"""SUPERSEDING CORRECTION of the 2026 EV floor and frozen residual SD recorded at 8c9675e (row 19).

GENERATED by ``scripts/derive_corrected_ev_chain.py --correction row19`` (quick task 261003-vke)
-- do not edit by hand; re-run the script.

WHAT IT SUPERSEDES. Commit ``8c9675e`` ({commit}) recorded the per-target EV floor and the
frozen residual SD in ``backtest/corrected_ev_chain_constants.py`` (run record
``outputs/p332/corrected_chain_fit.json``), measured on the three models production served from
2026-09-23. Owner ruling 2026-10-03 ~22:12 ET (WINDOWS row 19, "No home boost at any neutral
site"): Elo home-field advantage is zero at every neutral site and neutral games are left out of
HFA learning. Elo, gold and all three models were re-derived under that rule and swapped into
production on 2026-10-04, so scalars measured on the old models describe models that no longer
serve. Owner ruling 2026-10-03 ~22:50 ET ("Re-measure, same recipe"): this module is the SAME
recipe re-run, unchanged, on the new production models. The 8c9675e module and record are
byte-unchanged and remain the record of what was measured and when; nothing here edits them.

WHAT MOVED, symbol by symbol:
  * the EV floor per target -> ``CORRECTED_EV_FLOOR_BY_TARGET`` (the 8c9675e values beside it
    in ``SUPERSEDED_EV_FLOOR_BY_TARGET``). ``None`` means NO HONEST FLOOR COULD BE SWEPT and
    therefore NO BETS for that target (``CORRECTED_FLOOR_REFUSALS`` says why) -- never 0.0,
    which is a real grid value that admits everything, and never ``inf``, which
    ``assign_ev_tier`` refuses by name. Equal old and new values are a legitimate outcome of a
    re-measure: the floor is chosen from a coarse grid;
  * the frozen residual SD -> ``CORRECTED_FROZEN_SD_BY_TARGET`` (the 8c9675e values in
    ``SUPERSEDED_FROZEN_SD_BY_TARGET``); ATS and O/U only, WP none by design (D31-07);
  * the replay-snapshot policy sentence -> ``CORRECTED_REPLAY_SNAPSHOT_TS_POLICY`` (unchanged
    text, carried so this module has the 8c9675e module's shape).

WHAT DID NOT MOVE, and is READ from the frozen modules rather than re-declared here: the grid the
floor is chosen from (``backtest.ou_ev_chain.EV_FLOOR_GRID``), the ROI-best rule over it, the
frozen-SD estimator (``fit_frozen_residual_sd``), the walk-forward bias estimator and the EV tier
band edges (``backtest.ev_chain_constants.EV_TIER_BANDS``). No uppercase name in this module
repeats one defined in either frozen module.

THE ONE DELIBERATE DIVERGENCE from the Phase-31 recipe is inherited unchanged from 8c9675e: where
no grid floor admits a bet, Phase 31 froze the grid floor 0.00; this correction refuses and
records ``None``.

THE WINDOW. {window_reason} The bias seed ({seed}) is the new models' own walk-forward
residuals and feeds only the first seasons' bias. WP's first owned season has no prior-fold
converter slope, so WP is swept on {wp_seasons}.

NOT CLEAN EVIDENCE (D33.2-07): these are re-measured past seasons. They select a threshold; they
do not establish profitability. Only the 2026 season, recorded live, counts.

STAGED UNTIL THE ROW-19 REPOINT COMMIT: the live bet-list reader resolves the 8c9675e record
until one commit repoints it to ``{record_path}`` together with the re-measured cold-start bias
and edge thresholds.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""
'''

DEFAULT_CORRECTION = "p332_25b"

#: The 33.2 correction (Plans 33.2-26 / 33.2-29): every value, path and text exactly as before
#: the selector existed, so the 8c9675e and 9bb7568 records stay reproducible.
P332_25B_CORRECTION = BetRuleCorrection(
    name=DEFAULT_CORRECTION,
    swap_ids_slot="tests.phase33_state.P332_25B_SWAP_ARTIFACT_IDS",
    refit_ids_slot="tests.phase33_state.P332_25B_REFIT_ARTIFACT_IDS",
    refit_gold_generation_slot="tests.phase33_state.P332_25B_REFIT_GOLD_GENERATION",
    converter_slot="tests.phase33_state.P332_25B_BLEND_CONVERTER_ARTIFACT_ID",
    ev_record_path=RECORD_PATH,
    ev_module_path=MODULE_PATH,
    ev_record_id_prefix="corrected_chain_fit",
    ev_superseded_record_path=PHASE31_RECORD_PATH,
    ev_superseded_commit_slot="tests.phase31_state.PRE_REGISTRATION_COMMIT",
    ev_superseded_short="ee20773",
    ev_module_docstring=_MODULE_DOCSTRING,
    ev_chain_label="Plan 33.2-29",
    cold_module_path="backtest/corrected_cold_start_constants.py",
    cold_document_path="COLD-START-CORRECTION.md",
    cold_superseded_module="backtest.cold_start_constants",
    cold_superseded_short="11761c7",
    cold_superseded_paths=(
        "COLD-START-PREREGISTRATION.md",
        "backtest/cold_start_constants.py",
    ),
    cold_title=(
        "SUPERSEDING CORRECTION of the frozen 2026 cold-start rule (Plan 33.2-26, SPEC R14)."
    ),
    cold_module_supersedes=(
        "WHAT IT SUPERSEDES. Commit ``{short}`` ({full}) froze the 2026 edge-band\n"
        "thresholds and the 2026 chain-fit bias in ``backtest/cold_start_constants.py`` +\n"
        "``COLD-START-PREREGISTRATION.md``. Those values were derived from models fitted on "
        "inputs later\n"
        "found defective and from CLOSING lines, and those models are gone. This module "
        "SUPERSEDES them.\n"
        "The originals are byte-unchanged and remain the record of what was frozen and when; "
        "nothing here\n"
        "edits them. ``COLD-START-CORRECTION.md`` is the human-readable half of this record."
    ),
    cold_module_ev_half=(
        "THE OTHER TWO MOVED PARTS OF THE 2026 BET RULE -- the EV floor and the frozen residual "
        "SD -- are\n"
        "superseded separately, naming ``ee20773``: ``backtest.corrected_ev_chain_constants`` "
        "and\n"
        "``EV-CHAIN-CORRECTION.md`` (Plan 33.2-29)."
    ),
    cold_source_verdict="NOT_GATED (SPEC R13: owner readiness ruling, no pass/fail gate)",
    cold_document_why=(
        "**Why it is superseded.** `{short}` froze the 2026 edge thresholds and the 2026 "
        "chain-fit\n"
        "bias from models fitted on inputs later found defective (Phase 33.2) and from CLOSING "
        "lines, which\n"
        "did not exist at a game's lock. Those models are gone. A value derived from them sits "
        "inside the\n"
        "live 2026 bet rule, so it is replaced visibly rather than edited quietly."
    ),
    cold_document_bias_contrast=(
        "The superseded value was pooled IN-SAMPLE over the retired models' own training "
        "seasons; this\n"
        "one is out of sample."
    ),
    cold_document_ev_half=(
        "The 2026 bet rule is these thresholds and this bias PLUS the per-target **EV floor** "
        "(the number\n"
        "that decides whether a bet is placed at all) and the **frozen residual SD** (the scale "
        "that turns\n"
        "a model's miss into a bet's expected value). D33.2-25 rules that all of them move "
        "together. The\n"
        "EV floor and the residual SD are superseded separately -- naming `ee20773`, the "
        "Phase-31\n"
        "pre-registration they came from -- by Plan 33.2-29: "
        "`backtest/corrected_ev_chain_constants.py`\n"
        "and `EV-CHAIN-CORRECTION.md`. Their values, as the corrected chain fit records them:"
    ),
    cold_document_live_change=(
        "The live rule changes in ONE commit (Plan 33.2-26 Task 3), which moves every live "
        "reader\n"
        "together: the chain-fit record path, this bias, and the edge bands read by the web "
        "cache and the\n"
        "current-week predictions. Before that commit the live bet list read the Phase-31 "
        "chain-fit record\n"
        "and the `{short}` bias; no run can judge corrected floors against an uncorrected bias,\n"
        "or the reverse. From that commit on, a 2026 WIN bet must pass two tests:"
    ),
    cold_document_footer=(
        "*Phase: 33.2-information-time-integrity-day-before-kickoff-lock-and-hones*\n"
        "*Plan 33.2-26, superseding `{short}`*"
    ),
)

#: The row-19 re-measure (quick task 261003-vke): the same recipe on the models re-fitted after
#: the neutral-site Elo fix, superseding 8c9675e (EV floor, frozen SD) and 9bb7568 (edge
#: thresholds, 2026 chain-fit bias). STAGED until the row-19 repoint commit.
ROW19_CORRECTION = BetRuleCorrection(
    name="row19",
    swap_ids_slot="tests.phase33_state.ROW19_SWAP_ARTIFACT_IDS",
    refit_ids_slot="tests.phase33_state.ROW19_REFIT_ARTIFACT_IDS",
    refit_gold_generation_slot="tests.phase33_state.ROW19_REFIT_GOLD_GENERATION",
    converter_slot="tests.phase33_state.P332_24B_CONVERTER_ARTIFACT_ID",
    ev_record_path=Path("outputs") / "row19" / "neutral_hfa_chain_fit.json",
    ev_module_path=Path("backtest") / "neutral_hfa_ev_chain_constants.py",
    ev_record_id_prefix="neutral_hfa_chain_fit",
    ev_superseded_record_path=RECORD_PATH,
    ev_superseded_commit_slot="tests.phase33_state.EV_CHAIN_CORRECTIVE_COMMIT_SHA",
    ev_superseded_short="8c9675e",
    ev_module_docstring=_ROW19_MODULE_DOCSTRING,
    ev_chain_label="quick task 261003-vke",
    cold_module_path="backtest/neutral_hfa_cold_start_constants.py",
    cold_document_path="NEUTRAL-HFA-BET-RULE-CORRECTION.md",
    cold_superseded_module="backtest.corrected_cold_start_constants",
    cold_superseded_short="9bb7568",
    cold_superseded_paths=(
        "COLD-START-CORRECTION.md",
        "backtest/corrected_cold_start_constants.py",
    ),
    cold_title=(
        "SUPERSEDING CORRECTION of the 2026 cold-start rule recorded at 9bb7568 (row 19)."
    ),
    cold_module_supersedes=(
        "WHAT IT SUPERSEDES. Commit ``{short}`` ({full}) recorded the 2026 edge-band\n"
        "thresholds and the 2026 chain-fit bias in "
        "``backtest/corrected_cold_start_constants.py`` +\n"
        "``COLD-START-CORRECTION.md``, measured on the three models production served from "
        "2026-09-23.\n"
        "Owner ruling 2026-10-03 ~22:12 ET (WINDOWS row 19, quick task 261003-vke): zero Elo\n"
        "home-field advantage at every neutral site, neutral games left out of HFA learning. "
        "Elo, gold\n"
        "and all three models were re-derived under that rule and swapped into production on "
        "2026-10-04.\n"
        'Owner ruling 2026-10-03 ~22:50 ET ("Re-measure, same recipe"): the unchanged recipe is '
        "re-run on\n"
        "the new production models, and this module SUPERSEDES the {short} values. The {short} "
        "files are\n"
        "byte-unchanged and stay importable as the record of what was measured and when; "
        "nothing here\n"
        "edits them. ``NEUTRAL-HFA-BET-RULE-CORRECTION.md`` is the human-readable half of this "
        "record."
    ),
    cold_module_ev_half=(
        "THE OTHER TWO MOVED PARTS OF THE 2026 BET RULE -- the EV floor and the frozen residual "
        "SD -- are\n"
        "re-measured by the same recipe on the same new models, superseding the 8c9675e "
        "record:\n"
        "``backtest.neutral_hfa_ev_chain_constants`` (record "
        "``outputs/row19/neutral_hfa_chain_fit.json``)."
    ),
    cold_source_verdict=(
        "NOT_GATED (owner standing rule: models fitted on the pre-fix Elo are dead; the "
        "row-19 re-fits are promoted regardless of any adverse measurement)"
    ),
    cold_document_why=(
        "**Why it is superseded.** `{short}` measured the 2026 edge thresholds and the 2026 "
        "chain-fit\n"
        "bias on the three models production served from 2026-09-23. Owner ruling 2026-10-03 "
        "~22:12 ET\n"
        '(WINDOWS row 19, "No home boost at any neutral site"): Elo home-field advantage is '
        "zero at every\n"
        "neutral site and neutral games are left out of HFA learning. Elo, gold and all three "
        "models were\n"
        "re-derived under that rule and swapped into production on 2026-10-04, so values "
        "measured on the\n"
        'old models describe models that no longer serve. Owner ruling 2026-10-03 ~22:50 ET ("Re-measure,\n'
        'same recipe"): the recipe that produced `{short}` is re-run, unchanged, on the new '
        "production\n"
        "models, and the result replaces it visibly rather than by an edit in place."
    ),
    cold_document_bias_contrast=(
        "The superseded value was pooled the same way, out of sample, over the walk-forward "
        "predictions of\n"
        "the 2026-09-23 models' recipes; only the models moved."
    ),
    cold_document_ev_half=(
        "The 2026 bet rule is these thresholds and this bias PLUS the per-target **EV floor** "
        "(the number\n"
        "that decides whether a bet is placed at all) and the **frozen residual SD** (the scale "
        "that turns\n"
        "a model's miss into a bet's expected value). All of them move together. The EV floor "
        "and the\n"
        "residual SD are re-measured by the same recipe on the same new models, superseding the "
        "`8c9675e`\n"
        "record: `backtest/neutral_hfa_ev_chain_constants.py` (record\n"
        "`outputs/row19/neutral_hfa_chain_fit.json`). Their values, as that chain fit records "
        "them:"
    ),
    cold_document_live_change=(
        "The live rule changes in ONE commit (the row-19 repoint), which moves every live "
        "reader\n"
        "together: the chain-fit record path, this bias, and the edge bands read by the web "
        "cache and the\n"
        "current-week predictions. Before that commit the live bet list read the `8c9675e` "
        "chain-fit record\n"
        "and the `{short}` bias; no run can judge re-measured floors against a superseded bias, "
        "or the\n"
        "reverse. The bet itself is unchanged from `{short}`: a 2026 WIN bet must pass two "
        "tests:"
    ),
    cold_document_footer=(
        "*Quick task 261003-vke: zero Elo home-field advantage at neutral sites (WINDOWS row "
        "19)*\n"
        "*Superseding `{short}` by the same recipe*"
    ),
)

#: Every correction ``--correction`` accepts, by name.
CORRECTIONS: dict[str, BetRuleCorrection] = {
    correction.name: correction
    for correction in (P332_25B_CORRECTION, ROW19_CORRECTION)
}


# ---------------------------------------------------------------------------
# The record and the module
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_superseded_values(
    path: Path = PHASE31_RECORD_PATH,
) -> dict[str, dict[str, Any]]:
    """The Phase-31 floor and frozen SD per target, READ from its run record (never edited)."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        target: {
            "ev_floor_t": record["tune_fit"][target]["ev_floor_t"],
            "frozen_sd": record["tune_fit"][target]["frozen_sd"],
        }
        for target in TARGETS
    }


def build_record(
    derivation: ChainDerivation,
    *,
    record_id: str,
    fit_time: str,
    artifact_ids: Mapping[str, str],
    converter_artifact_id: str,
    gold_generation_digest: str,
    excluded_no_prelock_line: int,
    correction: BetRuleCorrection = P332_25B_CORRECTION,
) -> dict[str, Any]:
    """The run record: ``tune_fit`` in the shape ``load_frozen_chain_fit`` reads, plus provenance."""
    tune_fit: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for target in TARGETS:
        chain_fit = derivation.chain_fits[target]
        gate = derivation.gates[target]
        sweep = derivation.sweeps[target]
        tune_fit[target] = {
            "ev_floor_t": derivation.floors[target],
            "frozen_sd": chain_fit.frozen_sd,
            "season_bias_by_season": {
                str(season): bias
                for season, bias in sorted(chain_fit.season_bias_by_season.items())
            },
            "calibration_gate_passed": None if gate is None else bool(gate.passed),
            "tune_bet_counts": {
                str(row["threshold"]): row["bet_count"] for row in sweep.rows
            },
            "sweep": sweep.rows,
            "floor_refusal": derivation.refusals.get(target),
            "frozen_sd_absent_reason": WP_NO_FROZEN_SD_REASON
            if target == "wp"
            else None,
            "candidate_seasons": list(derivation.candidate_seasons[target]),
            "n_candidates": derivation.n_candidates[target],
            "tune_fit_seasons": list(chain_fit.tune_fit_seasons),
            "bias_pool_by_season": {
                str(season): list(pool)
                for season, pool in sorted(chain_fit.bias_pool_by_season.items())
            },
        }
        targets[target] = {
            "fallback_trigger": None if gate is None else gate.fallback_trigger
        }

    return {
        "record_id": record_id,
        "supersedes": {
            "preregistration_commit": correction.ev_superseded_commit(),
            "record": correction.ev_superseded_record_path.as_posix(),
        },
        "artifact_ids": dict(artifact_ids),
        "market_probability_artifact_id": converter_artifact_id,
        "gold_generation_digest": gold_generation_digest,
        "derivation_window": {
            "seasons": list(DERIVATION_SEASONS),
            "wp_seasons": list(derivation.candidate_seasons["wp"]),
            "bias_seed_seasons": list(BIAS_SEED_SEASONS),
            "reason": WINDOW_REASON,
        },
        "exclusions": {
            EXCLUSION_NO_PRELOCK_LINE: excluded_no_prelock_line,
            EXCLUSION_NO_PRIOR_FOLD_CONVERTER: derivation.wp_excluded_no_prior_fold,
        },
        "closing_line_rows": derivation.closing_line_rows,
        "input_digest": derivation.input_digest,
        "fit_time": fit_time,
        "thread_limit": PINNED_THREAD_COUNT,
        "wp_win_payout_factor": WP_WIN_PAYOUT_FACTOR,
        "tune_fit": tune_fit,
        "targets": targets,
    }


def _render_mapping(values: Mapping[str, Any]) -> str:
    return (
        "{"
        + ", ".join(f"{key!r}: {values[key]!r}" for key in TARGETS if key in values)
        + "}"
    )


def render_corrected_module(
    derivation: ChainDerivation,
    superseded: Mapping[str, Mapping[str, Any]],
    record: Mapping[str, Any],
    correction: BetRuleCorrection = P332_25B_CORRECTION,
) -> str:
    """The text of the correction's EV-chain module (``correction.ev_module_path``). Pure; ASCII."""
    commit = correction.ev_superseded_commit()
    wp_seasons = derivation.candidate_seasons["wp"]
    docstring = correction.ev_module_docstring.format(
        commit=commit,
        window_reason=WINDOW_REASON,
        seed=f"{BIAS_SEED_SEASONS[0]}-{BIAS_SEED_SEASONS[-1]}",
        wp_seasons=f"{wp_seasons[0]}-{wp_seasons[-1]}",
        record_path=correction.ev_record_path.as_posix(),
    )
    sds = {t: derivation.chain_fits[t].frozen_sd for t in TARGETS}
    policy = (
        "Every replay row's information time is judged against THAT GAME'S OWN LOCK: 18:00 "
        "America/New_York on the ET calendar day before its kickoff (D33.2-01, "
        "utils.game_lock). Information timed at the lock is admissible; one second after is "
        "not. This SUPERSEDES -- it does not replace -- the frozen sentence of commit ee20773 "
        "(backtest/ev_chain_constants.py), which describes the retired preceding-Friday 6 PM "
        "Eastern freeze and the retired get_synthetic_snapshot_ts; that sentence stays "
        "byte-unchanged as the record of what was frozen and when."
    )
    lines = [
        docstring,
        "from __future__ import annotations",
        "",
        f"SUPERSEDED_PREREGISTRATION_COMMIT: str = {commit!r}",
        f"CORRECTED_CHAIN_FIT_RECORD_PATH: str = {correction.ev_record_path.as_posix()!r}",
        f"CORRECTED_CHAIN_FIT_RECORD_ID: str = {record['record_id']!r}",
        "CORRECTED_SOURCE_ARTIFACT_IDS: dict[str, str] = "
        + "{"
        + ", ".join(f"{k!r}: {v!r}" for k, v in record["artifact_ids"].items())
        + "}",
        f"CORRECTED_MARKET_PROBABILITY_ARTIFACT_ID: str = "
        f"{record['market_probability_artifact_id']!r}",
        f"CORRECTED_GOLD_GENERATION: str = {record['gold_generation_digest']!r}",
        f"CORRECTED_DERIVATION_SEASONS: tuple[int, ...] = {DERIVATION_SEASONS!r}",
        f"CORRECTED_WP_DERIVATION_SEASONS: tuple[int, ...] = {tuple(wp_seasons)!r}",
        f"CORRECTED_WP_EXCLUDED_NO_PRIOR_FOLD: int = {derivation.wp_excluded_no_prior_fold!r}",
        f"CORRECTED_BIAS_SEED_SEASONS: tuple[int, ...] = {BIAS_SEED_SEASONS!r}",
        f"CORRECTED_WINDOW_REASON: str = {WINDOW_REASON!r}",
        "CORRECTED_EV_FLOOR_BY_TARGET: dict[str, float | None] = "
        + _render_mapping(derivation.floors),
        "SUPERSEDED_EV_FLOOR_BY_TARGET: dict[str, float] = "
        + _render_mapping({t: superseded[t]["ev_floor_t"] for t in TARGETS}),
        "CORRECTED_FLOOR_REFUSALS: dict[str, str] = "
        + _render_mapping(derivation.refusals),
        "CORRECTED_FROZEN_SD_BY_TARGET: dict[str, float | None] = "
        + _render_mapping(sds),
        "SUPERSEDED_FROZEN_SD_BY_TARGET: dict[str, float | None] = "
        + _render_mapping({t: superseded[t]["frozen_sd"] for t in TARGETS}),
        f"CORRECTED_WP_NO_FROZEN_SD_REASON: str = {WP_NO_FROZEN_SD_REASON!r}",
        f"CORRECTED_THREAD_LIMIT: int = {PINNED_THREAD_COUNT!r}",
        f"CORRECTED_WP_WIN_PAYOUT_FACTOR: float = {WP_WIN_PAYOUT_FACTOR!r}",
        f"CORRECTED_REPLAY_SNAPSHOT_TS_POLICY: str = {policy!r}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The production run
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the one option: which correction to derive (the default is the 33.2 one)."""
    parser = argparse.ArgumentParser(
        prog="derive_corrected_ev_chain",
        description=(
            "Re-derive the EV floor and the frozen residual SD on the served models and emit "
            "the superseding run record and constants module."
        ),
    )
    parser.add_argument(
        "--correction",
        choices=sorted(CORRECTIONS),
        default=DEFAULT_CORRECTION,
        help=(
            f"Which superseding correction to derive (default {DEFAULT_CORRECTION}, the 33.2 "
            "correction; row19 = the re-measure after the neutral-site Elo fix)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Derive, write the record and the module, and print the machine-readable summary."""
    from backtest.tune import (
        _gold_predictions_fn,
        common_gold_generation,
        read_source_recipes,
    )
    from models.blending_data import load_tuning_period_data
    from models.market_probability import load_market_probability_artifact
    from scripts.derive_cold_start_constants import ruff_format
    from tests.gold_generation import gold_generation_key

    correction = CORRECTIONS[parse_args(argv).correction]
    record_path = correction.ev_record_path
    module_path = correction.ev_module_path
    converter_id = correction.converter_id()

    ledger_path = Path(RUN_LEDGER_PATH)
    ledger_before = _sha256(ledger_path)
    latest_before = _sha256(LATEST_MANIFEST_PATH)

    swap_ids = correction.swap_ids()
    model_ids = correction.refit_ids()
    if any(swap_ids[target] != model_ids[target] for target in TARGETS):
        msg = f"the served models {swap_ids} are not the recorded re-fit {model_ids}"
        raise DerivationWindowError(msg)

    artifacts_dir = Path("artifacts")
    recipes = read_source_recipes(model_ids, artifacts_dir)
    gold_digest = common_gold_generation(recipes, correction.refit_gold_generation())
    live_gold = gold_generation_key()
    if live_gold != gold_digest:
        msg = (
            f"the gold on disk is generation {live_gold}, but the corrected models were "
            f"fitted on {gold_digest}; their walk-forward predictions cannot be reproduced."
        )
        raise DerivationWindowError(msg)

    converter = load_market_probability_artifact(converter_id, artifacts_dir)
    corpus = load_tuning_period_data(Path("data/silver"))
    predict = _gold_predictions_fn(Path("data/gold"))
    seasons = [*BIAS_SEED_SEASONS, *DERIVATION_SEASONS]
    # THE THREAD PIN (Plan 33.2-22): XGBoost answers differently at different OpenMP thread
    # counts; every published fit is pinned and the value recorded.
    with threadpool_limits(limits=PINNED_THREAD_COUNT):
        predictions = {
            target: predict(target, recipes[target], seasons) for target in TARGETS
        }

    derivation = derive_chain(
        corpus.frame, predictions, converter["walk_forward_slopes"]
    )

    fit_time = datetime.now(tz=UTC)
    record = build_record(
        derivation,
        record_id=f"{correction.ev_record_id_prefix}_{fit_time:%Y%m%d_%H%M%S}",
        fit_time=fit_time.isoformat(),
        artifact_ids=swap_ids,
        converter_artifact_id=converter_id,
        gold_generation_digest=gold_digest,
        excluded_no_prelock_line=len(corpus.excluded),
        correction=correction,
    )
    superseded = read_superseded_values(correction.ev_superseded_record_path)

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    module_text = ruff_format(
        render_corrected_module(derivation, superseded, record, correction),
        module_path.as_posix(),
    )
    module_path.write_text(module_text, encoding="utf-8", newline="\n")

    if (
        _sha256(ledger_path) != ledger_before
        or _sha256(LATEST_MANIFEST_PATH) != latest_before
    ):
        msg = "the spent 2025 run ledger or artifacts/latest.json moved during the derivation"
        raise DerivationWindowError(msg)

    print(f"DERIVATION_SEASONS= {list(DERIVATION_SEASONS)}")
    print(f"WP_DERIVATION_SEASONS= {list(derivation.candidate_seasons['wp'])}")
    print(f"WP_EXCLUDED_NO_PRIOR_FOLD= {derivation.wp_excluded_no_prior_fold}")
    print(f"BIAS_SEED_SEASONS= {list(BIAS_SEED_SEASONS)}")
    print(f"EXCLUDED_NO_PRELOCK_LINE= {len(corpus.excluded)}")
    print(f"CLOSING_LINE_ROWS= {derivation.closing_line_rows}")
    print(f"CANDIDATES= {derivation.n_candidates}")
    print(f"FLOORS= {derivation.floors}")
    print(f"SUPERSEDED_FLOORS= { {t: superseded[t]['ev_floor_t'] for t in TARGETS} }")
    print(f"FROZEN_SD= { {t: derivation.chain_fits[t].frozen_sd for t in TARGETS} }")
    print(f"SUPERSEDED_FROZEN_SD= { {t: superseded[t]['frozen_sd'] for t in TARGETS} }")
    for target in TARGETS:
        print(f"SWEEP= {target} {derivation.sweeps[target].rows}")
        gate = derivation.gates[target]
        if gate is not None:
            print(f"WP_GATE_PASSED= {gate.passed}")
    print(f"REFUSALS= {sorted(derivation.refusals)}")
    print(f"THREAD_LIMIT= {PINNED_THREAD_COUNT}")
    print(f"WP_WIN_PAYOUT_FACTOR= {WP_WIN_PAYOUT_FACTOR}")
    print(f"GOLD_GENERATION= {gold_digest}")
    print(f"INPUT_DIGEST= {derivation.input_digest}")
    print(f"LEDGER_SHA= {ledger_before}")
    print(f"LATEST_SHA= {latest_before}")
    print(f"CORRECTION= {correction.name}")
    print(f"MODULE_WRITTEN= {module_path.as_posix()}")
    print(f"RECORD_WRITTEN= {record_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
