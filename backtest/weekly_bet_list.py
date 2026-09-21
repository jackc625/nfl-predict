"""The ONE weekly bet-selection path (Phase 31, plan 31-17; PROD-02, SPEC R4, D31-31/32).

WHAT THIS MODULE IS
-------------------
The single entry point the Friday pipeline's recommendation step delegates to. It routes a week
through ``backtest.bet_selector.BetSelector`` -- the LOCKED-2 single bet-decision source -- maps
the decision records onto ``api.cache.BET_LIST_COLUMNS``, and persists them to the DURABLE
artifact PAIR under ``outputs/bet_list/`` -- the bet list AND its companion tracker blocks, both
written by :func:`generate_weekly_bet_list` -- that the Plan 31-18 cache population step reads.

It REPLACES a legacy step body that filtered on a confidence tier with no expected value, no
sizing and no suppression, and wrote a JSON file no code ever read.

WHY THE WRITE IS AN ARTIFACT AND NOT THE LIVE CACHE (REVIEW-CACHE)
------------------------------------------------------------------
``api.cache.populate_cache`` does NOT update the live database. It opens a FRESH temporary DuckDB,
loads its source artifacts into it, then does ``db_path.unlink()`` followed by
``tmp_path.rename(db_path)``. Anything this step wrote into the live cache would be silently
DELETED by the next population run, along with every preserved forward row. Every other cache table
already follows the working pattern -- a source artifact under ``artifacts/`` or ``outputs/`` that
the build READS -- and the bet list joins it.

WHY THE ARTIFACT IS ALSO THE DURABLE HOME OF FORWARD HISTORY
-------------------------------------------------------------
A forward row is the claim that on this Friday, before these games were played, the system
recommended this bet at this size. Regenerating it from today's models would silently rewrite what
was recommended. The cache is rebuilt from scratch on every population run, so a forward row
preserved only in the cache would not survive one. The upsert is therefore keyed on
``BET_LIST_ROW_KEY`` and split on the game's OWN freeze instant (D31-18):

  * BEFORE that game's freeze -- the latest run WINS. The games have not been priced and frozen
    yet and late odds can still land, so replacing is legitimate.
  * AT OR AFTER that game's freeze -- the STORED row wins, whole. Not merely its
    ``BET_LIST_IMMUTABLE_COLUMNS``: the entire stored row is carried forward untouched, which is
    strictly stronger and additionally stops a later run restating the decision-time CLV.

IMMUTABILITY COVERS THE RECOMMENDATION FACTS ONLY (REVIEW-FWD-GRADE)
---------------------------------------------------------------------
Whole-row-forever immutability would be self-defeating: at freeze the game has not been played, so
a row that can never change afterwards stays ``pending`` forever and the self-grading honesty loop
is silently disabled -- the tracker would show a permanently zero-graded forward block, and a
reader could not tell that from "we recommended nothing that won". So a frozen row is immutable
against RE-SELECTION and mutable ONLY through :func:`grade_pending_rows`, which transitions
``grading_status`` ONE-WAY and ONE-TIME out of ``pending``.

REPLAY rows are derived and fully regenerable, so they are rebuilt whenever inputs change; only
the FORWARD provenance carries the freeze fence.

WHY THE FROZEN FIT IS READ AND NEVER RE-FIT
--------------------------------------------
The per-target EV floor ``t`` and the frozen residual SD are PRE-REGISTERED quantities swept and
fitted on tune-only data by ``backtest.profitability_2025``. This module READS them from that run's
record and refuses to invent one. There is NO silent fallback to a 0.0 floor: admitting a target
at an unswept floor is an un-pre-registered threshold, which is the failure the whole apparatus
exists to prevent (the same rule ``BetSelector.ev_floor_for`` states for its own mapping).

Run the tests:  .venv/Scripts/python.exe -m pytest tests/unit/test_weekly_bet_list.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    BET_STATUS_LIVE,
    BET_TRACKER_BLOCK_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_FORWARD,
    RUN_MODE_FORWARD,
    RUN_MODE_REPLAY,
    assert_grading_transition,
    stamp_bet_list_provenance,
)
from backtest.bet_selector import BetSelector, SelectionResult

# MODULE-LEVEL, not lazy, and the distinction is a claim rather than a style choice. There is no
# cycle to break here: ``backtest.bet_tracker`` imports ``api.cache`` and ``utils`` and imports
# NOTHING from this module, and both import orders were run live before this import was added. A
# deferred import would imply a constraint that does not exist -- which is how the split this
# import exists to close was justified in the first place. Contrast
# ``build_bet_week_schedule``'s import of ``scripts.ingest_historical_odds``, which IS deferred
# and whose docstring names the real cycle it breaks.
from backtest.bet_tracker import aggregate_all_blocks, to_tracker_frame
from backtest.cold_start_constants import (
    CHAIN_FIT_BIAS_2026,
    CHAIN_FIT_BIAS_SEASONS,
)
from backtest.ev_chain_constants import assign_ev_tier
from backtest.ou_divergence import (
    HIGH_TOTAL_BOUNDARY_PREHOLD,
    dedupe_odds_by_book_preference,
)
from backtest.ou_ev_chain import american_to_payout
from backtest.selector_strategies import default_strategies
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BET_LIST_ARTIFACT_NAME",
    "BET_LIST_READ_COLUMNS",
    "BET_LIST_ROW_KEY",
    "BET_TRACKER_ARTIFACT_NAME",
    "DECIDED_AT_COLUMN",
    "DEFAULT_BET_LIST_DIR",
    "AlreadyGradedError",
    "BetListCacheSources",
    "ChainFitOverlayDisagreementError",
    "DecidedAfterFreezeError",
    "EmptyPriorResidualPoolError",
    "FrozenChainFitError",
    "LockPassedError",
    "MissingDecidedAtError",
    "WeeklyChainFit",
    "assert_decided_at_before_freeze",
    "build_bet_week_schedule",
    "build_freeze_instant_candidates",
    "build_weekly_candidates",
    "build_weekly_decision_frame",
    "frozen_overlay_season",
    "generate_weekly_bet_list",
    "grade_pending_rows",
    "grade_row",
    "load_frozen_chain_fit",
    "read_bet_list_artifact",
    "read_bet_list_cache_sources",
    "read_bet_list_with_schema_shim",
    "read_bet_tracker_artifact",
    "records_to_bet_list_frame",
    "select_games_for_decision_instant",
    "select_weekly_bets",
    "upsert_bet_list_rows",
    "write_bet_list_artifact",
    "write_bet_tracker_artifact",
]


# ---------------------------------------------------------------------------
# Artifact locations and the row key
# ---------------------------------------------------------------------------

# ``outputs/`` is gitignored, which is the same status every other cache SOURCE artifact has and is
# correct for derived data. The durable-history claim rests on the file surviving a cache rebuild,
# not on it being in git.
DEFAULT_BET_LIST_DIR: Path = Path("outputs") / "bet_list"
BET_LIST_ARTIFACT_NAME: str = "bet_list.parquet"
BET_TRACKER_ARTIFACT_NAME: str = "bet_tracker.json"

# The upsert key. A (game, target) pair is decided ONCE per week, so this quadruple identifies a
# row uniquely; the season and week are carried in the key rather than derived from the game_id so
# a malformed identifier cannot silently merge two weeks.
BET_LIST_ROW_KEY: tuple[str, ...] = ("game_id", "season", "week", "target")

# The row's OWN observation time: the instant this run decided this bet (D33-27). An ISO-8601
# string with an explicit UTC offset, matching its ``snapshot_ts`` / ``freeze_ts`` siblings
# rather than ``graded_at``'s TIMESTAMP type -- the three instants are read and compared by the
# same parse path, so one representation is what keeps that path single.
DECIDED_AT_COLUMN: str = "decided_at_utc"

# The column order the schema SHIM returns: ``BET_LIST_COLUMNS`` with ``decided_at_utc``
# present at the END of the immutable half.
#
# ONE EXPRESSION RATHER THAN A BRANCH, and that is what makes it safe across the schema bump.
# ``dict.fromkeys`` preserves order and DEDUPES, so before ``api.cache`` carries the column
# this tuple inserts it (22 + 1 + 6 = 29) and afterwards the explicit insertion collapses
# against the entry already in ``BET_LIST_IMMUTABLE_COLUMNS`` and this tuple IS
# ``BET_LIST_COLUMNS``. The import-time check at the foot of this module asserts that equality
# once the bump has landed, so the two cannot drift into two different 29-column orders.
BET_LIST_READ_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [*BET_LIST_IMMUTABLE_COLUMNS, DECIDED_AT_COLUMN, *BET_LIST_GRADING_COLUMNS]
    )
)

# The pre-registered run record carrying the tune-only fit. Gitignored (it is generator output),
# which is why its absence RAISES a named error instead of defaulting.
DEFAULT_CHAIN_FIT_PATH: Path = (
    Path("outputs") / "p31" / "profitability_2025_verdict.json"
)

# 1 unit = 1% of the notional bankroll (D27-09). The selector sizes in dollars; the bet list
# publishes units, so this is the ONE conversion in the mapping and it is not a metric.
DEFAULT_BANKROLL: float = 10_000.0
UNIT_FRACTION_OF_BANKROLL: float = 0.01

# The flat stake every row is graded against. 1.0 makes ``payout_flat`` a per-unit-staked profit,
# which is exactly what ``backtest.bet_tracker._flat_return_units`` sums, and it matches
# ``backtest.profitability_2025._per_bet_frame`` field for field.
FLAT_STAKE: float = 1.0

# The gold label column carrying each target's REALIZED value, taken from ``backtest.diagnose``'s
# contract rather than restated: WP the 0/1 home win, ATS the home margin, O/U the total points.
_REALIZED_LABEL_COLUMN: dict[str, str] = {
    "wp": "home_win",
    "ats": "home_margin",
    "ou": "total_points",
}

# The model column and the market columns each target's candidate frame must carry, and the silver
# odds column each market column is renamed FROM. Same contracts as
# ``backtest.profitability_2025``, restated here because this module builds a CURRENT-week frame
# from the odds SNAPSHOT rather than a historical frame from the closing odds.
_MODEL_COLUMN: dict[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}
_MARKET_COLUMNS: dict[str, tuple[str, ...]] = {
    "wp": ("ml_home", "ml_away"),
    "ats": ("closing_spread",),
    "ou": ("closing_total",),
}
_ODDS_SOURCE_COLUMN: dict[str, dict[str, str]] = {
    "wp": {},
    "ats": {"spread": "closing_spread"},
    "ou": {"total": "closing_total"},
}

# The LINE column each target publishes on the bet list, or None for a target that has no line. A
# moneyline is a PRICE, not a line, so WP publishes None rather than a fabricated zero -- the same
# reason ``WPStrategy.side_probability`` returns None for its slipped line.
_LINE_COLUMN: dict[str, str | None] = {
    "wp": None,
    "ats": "closing_spread",
    "ou": "closing_total",
}

CANONICAL_TARGETS: tuple[str, str, str] = ("wp", "ats", "ou")


class FrozenChainFitError(RuntimeError):
    """The pre-registered tune-only fit could not be resolved, so no week can be selected.

    Raised rather than defaulted. Every quantity it carries -- the per-target EV floor, the frozen
    residual SD, the prior-season walk-forward bias -- was fitted on tune-only data under a rule
    frozen before any 2025 number existed. Substituting a default would admit bets at a threshold
    nobody swept for.
    """


class ChainFitOverlayDisagreementError(RuntimeError):
    """The frozen overlay and the run record both price one season, and they DISAGREE (D33-21).

    Raised by name, carrying the season and BOTH values, rather than applying a precedence rule.
    A precedence rule is how two sources of truth quietly become one answer: whichever side is
    preferred wins silently, the losing value stays on disk looking authoritative, and nobody can
    later say which one a published bet was struck under.

    The AGREEING case is not an error. When the run record and the overlay carry the same value
    for the same season there is one answer and nothing to attribute, so the load proceeds.
    """


class EmptyPriorResidualPoolError(RuntimeError):
    """A bias was asked for over an EMPTY strictly-prior residual pool (R10 edge, T-33-88).

    Raised rather than answered with zero. A pooled mean over no seasons is not a small bias --
    it is no bias at all -- and the two ways of papering over that are both worse than refusing:
    inventing a value gives every candidate a correction nobody measured, and falling back to the
    TARGET season's own residuals debiases a season with the data it is being used to predict,
    which is the leak the whole walk-forward construction exists to prevent (D27-08).

    The estimator this rule was pre-registered under,
    ``backtest.ou_ev_chain.estimate_prior_season_bias``, refuses the same case in the same
    direction; this is that refusal named at the point the frozen pool is consumed.
    """


class LockPassedError(RuntimeError):
    """A game was offered for selection with a DECISION instant after its own lock (R6, R1).

    The successor of the former freeze-passed refusal (Plan 33.2-02). Its subject changed from
    the instant the RUN started to the instant the DECISION's inputs were captured: under
    D33.2-18 the daily run captures before the lock and builds after it, so a run that starts
    at or after a lock is normal, and only a decision whose inputs were captured after the lock
    is refused.

    Raised by name rather than skipped or dropped. Emitting a row whose decision post-dates
    its game's lock would publish a post-hoc pick wearing a pre-game timestamp -- the
    repudiation failure COLD-03 exists to prevent -- and dropping it silently would hide the
    same fact behind a shorter list nobody could audit.

    A ``RuntimeError`` subclass, deliberately OUTSIDE ``ValueError``: the selection path's
    callers catch ``ValueError`` for absent inputs (a missing schedule, an empty week), and a
    temporal-integrity refusal must not be convertible into one of those by an existing
    handler. Same ruling ``ProvisionalSnapshotAsTrainingInputError`` records (Plan 33-04).
    """


class MissingDecidedAtError(ValueError):
    """A FORWARD row carries no instant for the write-time assertion to compare (D33-27).

    Covers either missing side: a NULL ``decided_at_utc`` (the row makes no claim about when
    it was decided) and a NULL ``freeze_ts`` (there is nothing to compare that claim against).
    Both are named refusals rather than skips, because a forward row that cannot be checked is
    exactly the row the check exists for.
    """


class DecidedAfterFreezeError(ValueError):
    """A FORWARD row claims an observation time AFTER its own game freeze (R7, T-33-21)."""


class AlreadyGradedError(ValueError):
    """A row whose ``grading_status`` is already terminal was handed to the grader again.

    The transition is ONE-WAY and ONE-TIME (REVIEW-FWD-GRADE). ``api.cache.assert_grading_transition``
    permits an idempotent re-assertion of the SAME terminal status so a re-run of a batch grader is
    safe; this error is the stricter single-row rule, so a caller that reaches a settled bet at all
    is told rather than quietly allowed to restate a result.
    """


@dataclass(frozen=True)
class WeeklyChainFit:
    """One target's PRE-REGISTERED tune-only fit, read and never re-derived.

    Attributes:
        target: The canonical target code.
        ev_floor_t: The pre-registered per-target EV floor.
        frozen_sd: The frozen residual SD, or None for WP, which fits none by design (D31-07).
        season_bias_by_season: The prior-season walk-forward bias per season.
        calibration_gate_passed: The TUNE-split calibration gate verdict, or None when no gate was
            run for this target. CARRIED (WR-12) because without it ``build_strategies`` had no way
            to know the gate had failed, so ``WPStrategy.fallback_fired`` was always False and the
            registered fallback could not fire on the weekly path AT ALL -- inverting the D31-07
            guarantee that it "cannot fire silently" into "cannot fire, silently".
        fallback_trigger: The reason string the run recorded, or None. Carried alongside the
            verdict so a reconstructed gate stamps the SAME trigger onto every weekly decision
            record that the profitability run stamped onto its own.
    """

    target: str
    ev_floor_t: float
    frozen_sd: float | None
    season_bias_by_season: dict[int, float]
    calibration_gate_passed: bool | None = None
    fallback_trigger: str | None = None


# ---------------------------------------------------------------------------
# The frozen fit
# ---------------------------------------------------------------------------


def load_frozen_chain_fit(
    path: Path | str = DEFAULT_CHAIN_FIT_PATH,
) -> dict[str, WeeklyChainFit]:
    """Read the pre-registered per-target tune-only fit from the profitability run record.

    Args:
        path: The run record written by ``backtest.profitability_2025``.

    Returns:
        ``{target -> WeeklyChainFit}`` for all three canonical targets.

    Raises:
        FrozenChainFitError: when the record is absent, unreadable, or missing a target's fit.
            Never a default: see the class docstring.
    """
    fit_path = Path(path)
    if not fit_path.exists():
        msg = (
            f"the pre-registered tune-only fit is not on disk at {fit_path.as_posix()}; the "
            "weekly bet list cannot be selected without the frozen per-target EV floor and "
            "residual SD. IT CANNOT BE REGENERATED: it is generator output from a SINGLE-USE "
            "2025 hold split, and the committed one-shot run ledger under config/ records that "
            'split as already spent -- state "completed", no force flag -- so the generator '
            "would REFUSE rather than rebuild it. Naming that command here would send you to a "
            "locked door, which is why this message does not. Find the ledger and the path this "
            'record was written to with `uv run python -c "import pathlib, tomllib; '
            "p = next(pathlib.Path('config').glob('*run_ledger.toml')); "
            "print(p.as_posix()); "
            "print(tomllib.loads(p.read_text(encoding='utf-8'))['verdict_run_record'])\"`, then "
            "restore the file from there; a NEW measurement requires an owner ruling written "
            "into that ledger first. There is NO fallback: a defaulted floor would admit bets at "
            "a threshold nobody swept for."
        )
        raise FrozenChainFitError(msg)

    try:
        record = json.loads(fit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"the tune-only fit at {fit_path.as_posix()} could not be read: {exc}"
        raise FrozenChainFitError(msg) from exc

    tune_fit = record.get("tune_fit")
    if not isinstance(tune_fit, dict):
        msg = (
            f"{fit_path.as_posix()} carries no 'tune_fit' block; it is not a profitability run "
            "record, or it predates the block this module reads."
        )
        raise FrozenChainFitError(msg)

    fits: dict[str, WeeklyChainFit] = {}
    for target in CANONICAL_TARGETS:
        block = tune_fit.get(target)
        if not isinstance(block, dict) or "ev_floor_t" not in block:
            msg = (
                f"{fit_path.as_posix()} carries no tune-only fit for target {target!r}; every "
                "registered target needs its own pre-registered EV floor."
            )
            raise FrozenChainFitError(msg)
        frozen_sd = block.get("frozen_sd")
        gate_passed = block.get("calibration_gate_passed")
        # The trigger lives on the per-target VERDICT record, not in ``tune_fit`` -- the run writes
        # the gate verdict in one place and the reason it fired in another. Read both, so a
        # reconstructed gate stamps the same trigger the profitability run stamped (WR-12).
        target_record = record.get("targets")
        trigger = None
        if isinstance(target_record, dict) and isinstance(
            target_record.get(target), dict
        ):
            trigger = target_record[target].get("fallback_trigger")
        fits[target] = WeeklyChainFit(
            target=target,
            ev_floor_t=float(block["ev_floor_t"]),
            frozen_sd=None if frozen_sd is None else float(frozen_sd),
            season_bias_by_season={
                int(season): float(bias)
                for season, bias in (block.get("season_bias_by_season") or {}).items()
            },
            calibration_gate_passed=(
                None if gate_passed is None else bool(gate_passed)
            ),
            fallback_trigger=None if trigger is None else str(trigger),
        )
    return _overlay_frozen_chain_fit(fits)


def frozen_overlay_season() -> int:
    """The season the committed Phase-33 bias debiases -- DERIVED, never re-typed.

    The frozen bias is the pooled mean residual over the STRICTLY PRIOR seasons in
    ``backtest.cold_start_constants.CHAIN_FIT_BIAS_SEASONS``, so the season it corrects is the one
    immediately after that pool. Deriving it keeps "the season the bias is FOR" and "the seasons it
    was pooled FROM" one fact instead of two that can be edited apart; a hard-coded 2026 here would
    be the second copy, and this repository has been bitten by a second copy three times.
    ``tests/unit/test_chain_fit_2026_overlay.py`` asserts the result agrees with the frozen
    constant's own name.

    Raises:
        EmptyPriorResidualPoolError: when the pool is empty. ``max(())`` would otherwise raise a
            bare ``ValueError`` saying nothing about biases, and the honest refusal is the named
            one -- see the class docstring for why no value is invented in its place.
    """
    if not CHAIN_FIT_BIAS_SEASONS:
        msg = (
            "the frozen chain-fit bias records an EMPTY strictly-prior residual pool "
            "(backtest.cold_start_constants.CHAIN_FIT_BIAS_SEASONS is empty), so there is no "
            "season it could be the bias FOR and no pool it could have been estimated from. No "
            "bias is invented here and there is NO fallback to the target season's own "
            "residuals: debiasing a season with its own data is the leak the walk-forward "
            "construction exists to prevent (D27-08). Check what the pre-registration actually "
            'records with `uv run python -c "import backtest.cold_start_constants as c; '
            'print(c.CHAIN_FIT_BIAS_SEASONS, c.CHAIN_FIT_BIAS_2026)"`.'
        )
        raise EmptyPriorResidualPoolError(msg)
    return max(CHAIN_FIT_BIAS_SEASONS) + 1


def _overlay_frozen_chain_fit(
    fits: dict[str, WeeklyChainFit],
) -> dict[str, WeeklyChainFit]:
    """Overlay the committed Phase-33 bias for ONE season onto the run record's own (D33-21).

    The run record REMAINS the source for the seasons it covers (2021-2025). This adds the single
    season the record cannot cover, because the measurement that would have extended it was a
    single-use hold split the ledger marks as spent.

    REJECTED, recorded so neither is re-proposed: copying 2021-2025 into the frozen module (two
    copies that can drift), and writing the new season into the run record (editing a spent
    one-shot measurement, T-33-86).

    THE INT-KEYING IS PRESERVED, NOT RE-DONE. :func:`load_frozen_chain_fit` already normalizes
    JSON's string season keys at exactly one place -- ``{int(season): float(bias) ...}`` -- and
    this merges into the mapping that call produced. A second ``int()`` pass here would be a
    second place a future change could diverge, which is the defect being avoided rather than a
    belt-and-braces improvement.

    Args:
        fits: The per-target fits exactly as read from the run record, TARGET-keyed.

    Returns:
        A new mapping, same keys, each fit's ``season_bias_by_season`` extended by the overlay
        season. The inputs are frozen dataclasses and are not mutated.

    Raises:
        ChainFitOverlayDisagreementError: when the record already prices the overlay season with a
            DIFFERENT value.
        EmptyPriorResidualPoolError: when the frozen pool is empty.
    """
    season = frozen_overlay_season()
    overlaid: dict[str, WeeklyChainFit] = {}
    for target, fit in fits.items():
        frozen_bias = CHAIN_FIT_BIAS_2026.get(target)
        if frozen_bias is None:
            # A target the pre-registration does not price is left exactly as read. Refusing here
            # would make an unrelated fourth target impossible to load at all, and the season
            # gate already refuses any season it genuinely has no bias for.
            overlaid[target] = fit
            continue

        recorded = fit.season_bias_by_season.get(season)
        if recorded is not None and recorded != frozen_bias:
            msg = (
                f"two sources price season {season} for target {target!r} and they DISAGREE: the "
                f"run record carries {recorded!r} and the committed pre-registration carries "
                f"{frozen_bias!r}. This load REFUSES rather than preferring one of them. There is "
                "deliberately no precedence rule: whichever side a rule picked would win "
                "silently, the other value would stay on disk looking authoritative, and no one "
                "could later say which one a published bet was struck under. Decide which is "
                "correct and remove the other. What the pre-registration holds: "
                '`uv run python -c "import backtest.cold_start_constants as c; '
                'print(c.CHAIN_FIT_BIAS_2026, c.CHAIN_FIT_BIAS_SEASONS)"`.'
            )
            raise ChainFitOverlayDisagreementError(msg)

        merged = {**fit.season_bias_by_season, season: float(frozen_bias)}
        overlaid[target] = replace(fit, season_bias_by_season=merged)
    return overlaid


def _require_season_covered(fits: dict[str, WeeklyChainFit], season: int) -> None:
    """Refuse a season the pre-registered walk-forward bias does not cover.

    The bias is a per-season quantity estimated from STRICTLY PRIOR realized residuals. A season
    outside the fitted mapping has no bias, and the two line strategies raise by name when asked
    for one. Refusing HERE names the whole problem once, before a week is half-selected, and says
    what to re-run -- rather than surfacing as a per-candidate ValueError deep inside a strategy.
    """
    # WP is exempt ONLY while its registered fallback is inactive (WR-12). When the gate failed,
    # ``WPStrategy._p_home`` consults ``season_bias_for`` for every candidate and raises by name
    # deep inside the strategy if the season is uncovered -- which is exactly the per-candidate
    # failure this function exists to turn into one named refusal up front.
    uncovered = [
        target
        for target, fit in fits.items()
        if (target != "wp" or wp_fallback_is_active(fits))
        and season not in fit.season_bias_by_season
    ]
    if uncovered:
        covered = sorted(
            {s for fit in fits.values() for s in fit.season_bias_by_season}
        )
        msg = (
            f"the pre-registered walk-forward bias does not cover season {season} for target(s) "
            f"{sorted(uncovered)}; the fitted seasons are {covered}. The measurement that "
            "produced those seasons was a SINGLE-USE hold split and cannot be run again, so this "
            "is NOT fixed by regenerating anything -- which is why this message does not tell "
            "you to. For the FIRST season after that pool the bias is already committed: "
            "backtest/cold_start_constants.py carries CHAIN_FIT_BIAS_2026 per target and "
            "CHAIN_FIT_BIAS_SEASONS names the strictly-prior seasons it was pooled over, and "
            "load_frozen_chain_fit OVERLAYS it automatically -- so seeing THAT season here means "
            "the overlay did not reach this fit. Check what is committed with "
            '`uv run python -c "import backtest.cold_start_constants as c; '
            'print(c.CHAIN_FIT_BIAS_2026); print(c.CHAIN_FIT_BIAS_SEASONS)"`. A season BEYOND '
            "that one has no pre-registered bias at all, and none is invented here: a raw biased "
            "total is exactly what the walk-forward correction exists to remove (D27-07)."
        )
        raise FrozenChainFitError(msg)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


SCHEDULE_COLUMNS: tuple[str, ...] = ("game_id", "season", "week", "gameday")


def _with_gameday(games: pd.DataFrame) -> pd.DataFrame:
    """Add the EASTERN calendar ``gameday`` each game's lock is measured from.

    One derivation shared by the week-scoped and full-season loaders, and no longer a private
    one: the date is read through ``utils.date_utils.kickoff_wall_clock_et`` -- the SAME
    accessor ``utils.game_lock.game_lock`` reads the kickoff's ET date through -- so this
    module and the lock rule cannot disagree about which calendar day a game is on. The date
    is taken in EASTERN, never in UTC: an 8:15 PM Eastern Monday kickoff is 00:15 UTC on
    TUESDAY, and a UTC-dated gameday would move that game's lock by a whole day.
    """
    from utils.date_utils import kickoff_wall_clock_et

    dated = games.copy()
    # A null kickoff yields a null gameday -- no date, and therefore no lock -- rather than a
    # manufactured one; the lock helper refuses it by name if anything asks for its lock.
    dated["gameday"] = [
        None if pd.isna(kickoff) else kickoff_wall_clock_et(kickoff).date().isoformat()
        for kickoff in games["kickoff_et"]
    ]
    subset = cast("pd.DataFrame", dated[list(SCHEDULE_COLUMNS)])
    return subset.reset_index(drop=True)


def _read_silver_games(silver_dir: Path) -> pd.DataFrame:
    games_path = Path(silver_dir) / "games.parquet"
    if not games_path.exists():
        msg = f"cannot build the weekly universe -- schedule missing: {games_path.as_posix()}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(games_path)


def _load_week_schedule(season: int, week: int, silver_dir: Path) -> pd.DataFrame:
    """The week's schedule, carrying the ``gameday`` the per-game freeze fence is measured from.

    READ ONLY. The schedule -- not the odds join -- is the universe's spine (D31-19): a game
    absent from it could not be reported as suppressed at all.
    """
    games = _read_silver_games(silver_dir)
    week_games = cast(
        "pd.DataFrame", games[(games["season"] == season) & (games["week"] == week)]
    ).copy()
    if week_games.empty:
        games_path = Path(silver_dir) / "games.parquet"
        msg = (
            f"no scheduled games for {season} week {week} in "
            f"{games_path.as_posix()}; an empty universe is refused rather than published as a "
            "week in which nothing was recommended."
        )
        raise ValueError(msg)

    return _with_gameday(week_games)


def _load_full_schedule(silver_dir: Path) -> pd.DataFrame:
    """Every scheduled game with its ``gameday``. READ ONLY.

    The full season rather than one week, because one lock instant can span TWO weeks (a
    Saturday lock covers week N's Sunday slate; a Wednesday lock covers week N+1's Thursday
    game) and a week-scoped read cannot see the second one.
    """
    return _with_gameday(_read_silver_games(silver_dir))


def _drop_excluded(
    schedule: pd.DataFrame, excluded_game_ids: frozenset[str]
) -> pd.DataFrame:
    """*schedule* without the games the run has decided not to bet (D33.2-05)."""
    if not excluded_game_ids:
        return schedule
    keep = ~schedule["game_id"].astype(str).isin(sorted(excluded_game_ids))
    return cast("pd.DataFrame", schedule[keep]).reset_index(drop=True)


def build_weekly_candidates(
    season: int,
    week: int,
    *,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    excluded_game_ids: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the deployed artifacts over one week and join the stored odds snapshot. READ ONLY.

    Mirrors ``backtest.profitability_2025._load_candidate_frames`` one week at a time, with one
    deliberate difference: the market side comes from ``data/silver/odds_snapshot.parquet``,
    carrying its ``snapshot_ts`` and its provenance columns, so the selector's admissibility
    fence and its OUM-06 provenance hard-fail both have something real to judge.

    Args:
        season: The season to build.
        week: The week to build.
        artifacts_dir: Where the deployed model artifacts live.
        gold_dir: Where the per-target gold matrices live.
        silver_dir: Where the schedule and the odds snapshot live.
        excluded_game_ids: Games the run has decided not to bet (D33.2-05's live skip). They
            are removed from the schedule SPINE before anything else reads it, so an excluded
            game is never scored, never priced and never written -- not even as a suppressed
            row. Empty by default, so every existing caller is unchanged.

    Returns:
        ``(candidates, schedule)``. ``candidates`` carries one row per (game, target) for which a
        model output exists; ``schedule`` is the week's spine for the universe build. Both are
        EMPTY, and nothing raises, when every scheduled game is excluded: that day is honestly
        all-skipped, which is a different fact from a week with no scheduled games (still
        refused by ``_load_week_schedule``).
    """
    from backtest.diagnose import score_deployed_artifacts

    schedule = _load_week_schedule(season, week, silver_dir)
    # THE EXCLUSION LANDS ON THE SPINE, HERE, AND NOWHERE LATER (D33.2-05, T-33.2-02-14). The
    # schedule -- not the odds join -- is the universe's spine (D31-19, `_load_week_schedule`):
    # a game absent from it cannot be reported as suppressed at all. Dropping it any later
    # would still produce a SUPPRESSED row with a rejection_reason, because
    # `records_to_bet_list_frame` stamps one row per (game, target) including suppressed ones,
    # and a suppressed row is still a row and still a write. Dropping it before `game_ids` is
    # taken means the odds join and the returned spine never see it; the gold rows handed to
    # the scorer are narrowed by the same set below, because the scorer reads gold by WEEK, not
    # by the spine's game ids.
    schedule = _drop_excluded(schedule, excluded_game_ids)
    if schedule.empty:
        logger.info(
            "Every scheduled game was excluded; the week is all-skipped, not unscheduled",
            season=season,
            week=week,
            n_excluded=len(excluded_game_ids),
        )
        return pd.DataFrame(columns=pd.Index(["game_id", "target"])), schedule
    game_ids = set(schedule["game_id"])

    odds_path = silver_dir / "odds_snapshot.parquet"
    if not odds_path.exists():
        msg = f"cannot price the weekly bet list -- odds snapshot missing: {odds_path.as_posix()}"
        raise FileNotFoundError(msg)
    odds = pd.read_parquet(odds_path)
    # The book is chosen BY NAME, not by parquet row order (WR-08). ``keep="first"`` picked
    # whichever row happened to appear first in the file, so appending a second book's row for a
    # game -- or any rewrite that changed row order -- silently changed which book's price the
    # published bet was struck at, with nothing on the record to attribute the change to.
    odds = dedupe_odds_by_book_preference(odds[odds["game_id"].isin(game_ids)])

    frames: list[pd.DataFrame] = []
    for target in CANONICAL_TARGETS:
        gold_path = gold_dir / f"features_{target}.parquet"
        if not gold_path.exists():
            msg = f"cannot select {target!r} -- gold matrix missing: {gold_path.as_posix()}"
            raise FileNotFoundError(msg)
        gold = pd.read_parquet(gold_path)
        gold = gold[(gold["season"] == season) & (gold["week"] == week)].copy()
        if gold.empty:
            msg = (
                f"gold matrix {gold_path.as_posix()} has no rows for {season} week {week}; the "
                "week cannot be selected for target " + repr(target)
            )
            raise ValueError(msg)
        # An excluded game is never SCORED either (D33.2-05). Gold is read by week, so without
        # this the scorer would still see the game and emit a candidate the selector would then
        # refuse as outside the schedule. Applied after the week-emptiness check, so that check
        # keeps meaning "gold carries this week at all".
        gold = _drop_excluded(cast("pd.DataFrame", gold), excluded_game_ids)

        scored = score_deployed_artifacts(
            target, gold_df=gold, artifacts_dir=artifacts_dir
        )
        keep = ["game_id", "snapshot_ts", "ml_home", "ml_away", "spread", "total"]
        keep += [c for c in ("sportsbook", "is_live") if c in odds.columns]
        keep += [
            c
            for c in (
                "spread_ju_home",
                "spread_ju_away",
                "total_over_ju",
                "total_under_ju",
            )
            if c in odds.columns
        ]
        merged = scored.merge(odds[keep], on="game_id", how="left")
        merged = merged.rename(columns=dict(_ODDS_SOURCE_COLUMN[target]))
        merged["target"] = target
        frames.append(merged)

    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.merge(
        schedule[["game_id", "gameday"]], on="game_id", how="left"
    )
    logger.info(
        "Built weekly candidates",
        season=season,
        week=week,
        n_scheduled=len(schedule),
        n_candidates=len(candidates),
    )
    return candidates, schedule


def _game_locks(schedule: pd.DataFrame) -> list[datetime]:
    """Each game's lock, in *schedule* order, from the ONE rule. Never re-derived here.

    Every value is ``scripts.ingest_historical_odds.gameday_lock`` of the game's Eastern
    ``gameday``, which hands that date to ``utils.game_lock.game_lock`` at call time. The import
    is deferred for the cycle ``build_bet_week_schedule``'s docstring names.
    """
    from scripts.ingest_historical_odds import gameday_lock, require_aware_snapshot_ts

    return [
        require_aware_snapshot_ts(gameday_lock(str(gameday)))
        for gameday in schedule["gameday"]
    ]


def _games_at_lock(schedule: pd.DataFrame, instant: datetime) -> pd.DataFrame:
    """The rows of *schedule* whose own lock IS *instant* -- scoping only, never refusing."""
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    if schedule.empty:
        return schedule.iloc[0:0].copy()
    target = require_aware_snapshot_ts(instant)
    in_scope = pd.Series(
        [lock == target for lock in _game_locks(schedule)], index=schedule.index
    )
    return cast("pd.DataFrame", schedule[in_scope]).copy()


def select_games_for_decision_instant(
    schedule: pd.DataFrame,
    instant: datetime,
    *,
    decided_at: datetime,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """The games belonging to ONE lock instant, refusing any decided AFTER its own lock.

    THE SELECTION UNIT IS THE LOCK INSTANT, NOT THE WEEK (D33-28, D33.2-01). A game locks at
    18:00 Eastern on the Eastern day before its kickoff, so one instant covers every game on
    one calendar day -- the Saturday lock covers the whole Sunday slate, the Wednesday lock a
    Thursday game -- and games of two different NFL weeks can share a run.

    THE REFUSAL'S SUBJECT IS THE DECISION INSTANT, NOT THE RUN'S START (D33.2-18). The daily
    run captures its inputs BEFORE the lock and builds AFTER it, so a run that STARTS at or
    after a lock is normal. What is refused is a decision whose inputs were captured after the
    lock: ``decided_at > lock`` raises :class:`LockPassedError`. AT-LOCK IS ADMISSIBLE
    (``decided_at <= lock``), the same operator as ``utils.game_lock.is_admissible``, which
    makes the comparison here. *decided_at* is passed in explicitly and is never the clock.

    EXCLUSION PRECEDES THE REFUSAL (D33.2-05). A game skipped for a post-lock input is usually
    exactly a game whose lock has passed; checking before excluding would raise for a game the
    run had already decided not to bet, turning one clean skip into a failure that denies that
    day's clean games their predictions.

    SCOPING IS NOT REFUSING. A game whose lock is a DIFFERENT instant is simply absent from the
    result and nothing raises -- it belongs to another run.

    Args:
        schedule: Any frame carrying ``game_id`` and ``gameday``. Rows are FILTERED, never
            reshaped, so a caller's extra columns survive.
        instant: The lock instant this run is scoped to.
        decided_at: The instant the decision's inputs were captured. Must be timezone-aware.
        excluded_game_ids: Games the run has decided not to bet. Removed before the check.

    Returns:
        The subset of *schedule* whose lock equals *instant*, minus the excluded games.

    Raises:
        LockPassedError: when an in-scope game's lock is before *decided_at*.
        NaiveTimestampError: when *decided_at* or *instant* carries no timezone.
    """
    import utils.game_lock as lock_rule
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    decision_instant = require_aware_snapshot_ts(decided_at)
    target = require_aware_snapshot_ts(instant)

    selected = _drop_excluded(_games_at_lock(schedule, target), excluded_game_ids)
    if selected.empty:
        return selected

    passed = [
        str(game_id)
        for game_id in selected["game_id"]
        if not lock_rule.is_admissible(decision_instant, target)
    ]
    if passed:
        msg = (
            f"refusing to emit a bet row decided at {decision_instant.isoformat()} for "
            f"{len(passed)} game(s) whose own lock {target.isoformat()} is BEFORE that "
            f"decision: {', '.join(passed)}. At-lock is admissible and one second later is "
            "not. A row emitted here would be a post-hoc pick wearing a pre-game timestamp; "
            "the honest outcome is a MISSING row with a dated reason."
        )
        raise LockPassedError(msg)

    logger.info(
        "Selected games for lock instant",
        instant=target.isoformat(),
        decided_at=decision_instant.isoformat(),
        n_scheduled=len(schedule),
        n_selected=len(selected),
        n_excluded=len(excluded_game_ids),
    )
    return selected


def build_freeze_instant_candidates(
    instant: datetime,
    *,
    decided_at: datetime,
    schedule: pd.DataFrame | None = None,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    excluded_game_ids: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidates for every game sharing ONE lock instant, across the weeks it spans.

    A WRAPPER OVER THE WEEK-SCOPED BUILDER, NOT A SECOND IMPLEMENTATION.
    :func:`build_weekly_candidates` takes one ``(season, week)`` and returns a 2-tuple, and a
    lock instant can span two of them, so the week-scoped function cannot serve an instant with
    one call. This groups the instant's games by ``(season, week)``, calls the EXISTING builder
    once per group, RESTRICTS each group's result to the ``game_id``s the instant actually
    selected, and concatenates. Candidate construction is not duplicated.

    THE RESTRICTION IS LOAD-BEARING. A week-scoped call returns the WHOLE week, which includes
    games belonging to OTHER lock instants. Concatenating without restricting would publish them
    under this run and restate a decision made on another day.

    Args:
        instant: The lock instant this run is scoped to.
        decided_at: The instant the decision's inputs were captured; the fence judges it.
        schedule: An explicit schedule to select from; read from *silver_dir* when omitted.
        artifacts_dir: Passed through to the week-scoped builder.
        gold_dir: Passed through to the week-scoped builder.
        silver_dir: Passed through to the week-scoped builder, and the schedule source.
        excluded_game_ids: Games the run has decided not to bet, passed to BOTH the selection
            fence (removed before its check) and the week-scoped builder (removed from its
            spine), so no row of any kind is produced for them.

    Returns:
        ``(candidates, schedule)`` -- the same 2-tuple shape :func:`build_weekly_candidates`
        returns. Both are EMPTY, and nothing raises, when every game at the instant is
        excluded.

    Raises:
        ValueError: when the instant covers no scheduled game at all -- a different fact from
            every covered game being excluded, and still refused.
        LockPassedError: propagated from the selection fence.
    """
    universe = _load_full_schedule(silver_dir) if schedule is None else schedule
    scheduled_at_instant = _games_at_lock(universe, instant)
    if scheduled_at_instant.empty:
        msg = (
            f"no scheduled game locks at {instant.isoformat()}; an empty universe is refused "
            "rather than published as a run in which nothing was recommended. Check the "
            "instant against utils.game_lock.game_lock for the games it should cover."
        )
        raise ValueError(msg)

    selected = select_games_for_decision_instant(
        universe,
        instant,
        decided_at=decided_at,
        excluded_game_ids=excluded_game_ids,
    )
    if selected.empty:
        # Every game at this instant was excluded: an honestly all-skipped day (SPEC R3), not
        # an unscheduled one, so it returns empty rather than refusing.
        logger.info(
            "Every game at the lock instant was excluded; the run is all-skipped",
            instant=instant.isoformat(),
            n_scheduled=len(scheduled_at_instant),
        )
        return pd.DataFrame(columns=pd.Index(["game_id", "target"])), selected

    # Grouped by an EXPLICIT distinct-pair pass rather than ``groupby``: the loop needs the
    # (season, week) pair as two plain ints to hand to the week-scoped builder, and a groupby key
    # arrives as an opaque Hashable that has to be unpacked on trust.
    pairs = (
        cast("pd.DataFrame", selected[["season", "week"]])
        .drop_duplicates()
        .sort_values(["season", "week"])
    )
    candidate_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    for season, week in zip(pairs["season"], pairs["week"], strict=True):
        group = selected[(selected["season"] == season) & (selected["week"] == week)]
        wanted = sorted({str(game_id) for game_id in group["game_id"]})
        candidates, week_schedule = build_weekly_candidates(
            int(season),
            int(week),
            artifacts_dir=artifacts_dir,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            excluded_game_ids=excluded_game_ids,
        )
        candidate_frames.append(
            cast(
                "pd.DataFrame",
                candidates[candidates["game_id"].astype(str).isin(wanted)],
            )
        )
        schedule_frames.append(
            cast(
                "pd.DataFrame",
                week_schedule[week_schedule["game_id"].astype(str).isin(wanted)],
            )
        )

    merged_candidates = pd.concat(candidate_frames, ignore_index=True)
    merged_schedule = pd.concat(schedule_frames, ignore_index=True)
    logger.info(
        "Built lock-instant candidates",
        instant=instant.isoformat(),
        n_groups=len(candidate_frames),
        n_games=len(selected),
        n_candidates=len(merged_candidates),
    )
    return merged_candidates, merged_schedule


def require_frozen_sd(fit: WeeklyChainFit) -> float:
    """The frozen residual SD, REFUSING an absent, non-finite or non-positive one (WR-05).

    ``load_frozen_chain_fit`` permits ``frozen_sd`` to be absent -- ``block.get("frozen_sd")``
    yields ``None`` -- and ``build_strategies`` used to write ``float(fit.frozen_sd or 0.0)``,
    which is silent in exactly the case that matters. A zero SD reaches ``calibrated_p_cover`` /
    ``calibrated_p_over``, where ``z = (line - corrected) / sd`` divides by zero, yielding
    ``+/-inf`` -> ``norm.cdf`` -> 0.0 or 1.0 -> clipped to ``P_OVER_CLIP``. Every candidate then
    prices at 0.999 or 0.001, every 0.999 clears any EV floor, and Kelly stakes it at the 5%
    per-bet cap: a week's bet list maximally staked on a model that produced no probability at
    all, with nothing raising.

    The PRICING path already refuses this input by name (``ats_ev_chain.py``: "the ATS chain
    requires a frozen residual SD ... ChainFit.frozen_sd is None"), so without this the SELECTION
    path was strictly weaker than the pricing path it is supposed to mirror. ``or 0.0`` also
    swallowed a legitimately-0.0 stored value, which is the same fatal input arriving a different
    way.

    Args:
        fit: The per-target frozen fit.

    Returns:
        The SD as a positive, finite float.

    Raises:
        FrozenChainFitError: naming the target and the offending value.
    """
    value = fit.frozen_sd
    if value is None or not math.isfinite(value) or value <= 0:
        msg = (
            f"target {fit.target!r} has no usable frozen residual SD ({value!r}); a zero, "
            "absent or non-finite SD divides by zero in the calibrated-probability converter "
            "and clips every candidate to the probability bound, which Kelly then stakes at the "
            "per-bet cap. The SD is READ from the run record and cannot be re-fitted here: it "
            "came from a single-use hold split the committed one-shot ledger under config/ "
            "records as spent, so there is no command that would produce a new one and this "
            "message names none. Inspect what the record actually carries for each target with "
            '`uv run python -c "import json; from backtest.weekly_bet_list import '
            "DEFAULT_CHAIN_FIT_PATH as p; "
            "print({t: b.get('frozen_sd') for t, b in "
            "json.loads(p.read_text(encoding='utf-8'))['tune_fit'].items()})\"`, then repair or "
            "restore the record rather than defaulting the value."
        )
        raise FrozenChainFitError(msg)
    return float(value)


def wp_fallback_is_active(fits: Mapping[str, WeeklyChainFit]) -> bool:
    """True when the run record says WP's TUNE-split calibration gate FAILED (WR-12).

    ``None`` -- no gate was run -- is NOT a failure and is the D31-07 default path. Only an
    explicit ``False`` activates the registered fallback.
    """
    return fits["wp"].calibration_gate_passed is False


def _wp_gate_from_fit(fit: WeeklyChainFit) -> Any | None:
    """Rebuild the ``WPGateResult`` the profitability run recorded, or None on the default path.

    The weekly path reads a run RECORD, not a live gate, so the report body is not available and
    is reconstructed as an explicit marker rather than an empty dict pretending to be one. Only
    the three fields ``WPStrategy`` actually reads -- ``passed``, ``fallback_fired`` and
    ``fallback_trigger`` -- are load-bearing here.
    """
    if fit.calibration_gate_passed is None:
        return None
    from backtest.wp_ev_chain import WPGateResult

    passed = bool(fit.calibration_gate_passed)
    return WPGateResult(
        passed=passed,
        report={
            "source": "reconstructed from the profitability run record; the bin table lives on "
            "that artifact, not here"
        },
        fallback_fired=not passed,
        fallback_trigger=fit.fallback_trigger,
    )


def build_strategies(fits: dict[str, WeeklyChainFit]) -> list[Any]:
    """The three registered strategies, built ONCE from the frozen fit.

    Built once and shared between selection and grading, so a bet is graded under exactly the rule
    it was selected under. Two independent constructions could drift apart on a parameter and the
    disagreement would show up as a mis-graded result rather than as an error.

    The two line targets' residual SDs go through :func:`require_frozen_sd`, which REFUSES an
    absent or non-positive value rather than substituting 0.0 (WR-05).
    """
    return default_strategies(
        ou_frozen_sd=require_frozen_sd(fits["ou"]),
        ou_season_bias_by_season=fits["ou"].season_bias_by_season,
        ats_frozen_sd=require_frozen_sd(fits["ats"]),
        ats_season_bias_by_season=fits["ats"].season_bias_by_season,
        wp_season_bias_by_season=fits["wp"].season_bias_by_season,
        # The gate verdict the run RECORDED, carried through (WR-12). Without it ``wp_gate``
        # defaulted to None, so ``WPStrategy.fallback_fired`` was always False, ``_p_home`` always
        # returned the deployed probability unchanged, and every weekly row was stamped
        # ``fallback_fired = False`` -- a registered fallback that could not fire at all, silently,
        # which is the inverse of the D31-07 guarantee.
        wp_gate=_wp_gate_from_fit(fits["wp"]),
        high_total_boundary=float(HIGH_TOTAL_BOUNDARY_PREHOLD),
    )


def select_weekly_bets(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    fits: dict[str, WeeklyChainFit],
    *,
    strategies: list[Any] | None = None,
    bankroll: float = DEFAULT_BANKROLL,
) -> SelectionResult:
    """Route the week through the SINGLE bet-decision source (BET-01, LOCKED-2).

    Every pricing, admission, sizing and grading decision happens inside ``BetSelector.select``.
    This function supplies the frozen inputs and the schedule that makes the universe complete;
    it prices nothing of its own, which is what keeps exactly one bet-decision path in the tree.
    """
    if strategies is None:
        strategies = build_strategies(fits)
    selector = BetSelector(
        frozen_sd=float(fits["ou"].frozen_sd or 0.0),
        season_bias_by_season={},
        ev_floor_t={target: fit.ev_floor_t for target, fit in fits.items()},
        bankroll=bankroll,
        high_total_boundary=float(HIGH_TOTAL_BOUNDARY_PREHOLD),
        strategies=strategies,
    )
    return selector.select(candidates, scheduled_games=schedule)


# ---------------------------------------------------------------------------
# The selector-record -> bet_list mapping
# ---------------------------------------------------------------------------


def _instant_text(value: Any) -> str | None:
    """Render a timestamp for the bet list's VARCHAR instant columns without re-deriving it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    return text or None


def _row_from_record(
    record: dict[str, Any],
    *,
    status: str,
    rejection_reason: str | None,
    ev_floor_t: float,
    unit: float,
    decided_at: str | None,
) -> dict[str, Any]:
    """Map ONE selector decision record onto the locked 29-column bet_list schema.

    The only transforms are a UNIT conversion (dollars -> units) and the two honesty labels, which
    are stamped separately by ``stamp_bet_list_provenance``. NO metric is re-derived: the EV, the
    calibrated P(side), the line, the slipped line and the price are carried through unchanged.

    A SUPPRESSED row was never priced, so its ``ev_tier`` is None -- ``assign_ev_tier`` raises on a
    non-finite or below-floor EV by design, and tiering a bet nobody made would claim a band it was
    never in.

    *decided_at* is THIS RUN's observation time, already rendered and already validated by
    :func:`records_to_bet_list_frame` -- it is None for a replay run, which observes nothing. It
    arrives as a DICT KEY here rather than as a later column assignment on purpose: the
    retroactive-stamp scan in ``tests/unit/test_decided_at_utc.py`` flags every
    ``frame[decided_at_utc] = value`` outside the emission path, and a row built with the value in
    place has no such assignment to flag or to have to except.
    """
    target = str(record["target"])
    line_column = _LINE_COLUMN[target]
    per_bet_ev = record.get("per_bet_ev")
    kelly_stake = record.get("kelly_stake")

    ev_tier: str | None = None
    if status == BET_STATUS_LIVE and per_bet_ev is not None:
        ev_tier = assign_ev_tier(float(per_bet_ev), ev_floor_t)

    return {
        "game_id": record["game_id"],
        "season": int(record["season"]),
        "week": int(record["week"]),
        "target": target,
        "bet_side": record.get("bet_side"),
        "model_value": record.get(_MODEL_COLUMN[target]),
        "market_value": None if line_column is None else record.get(line_column),
        "line": None if line_column is None else record.get(line_column),
        "slipped_line": record.get("slipped_line"),
        "calibrated_p_side": record.get("calibrated_p_side"),
        "per_bet_ev": per_bet_ev,
        "stake_units": None if kelly_stake is None else float(kelly_stake) / unit,
        "ev_tier": ev_tier,
        "status": status,
        "rejection_reason": rejection_reason,
        "eligibility_label": record.get("subpop_label"),
        "snapshot_ts": _instant_text(record.get("snapshot_ts")),
        "freeze_ts": _instant_text(record.get("freeze_ts")),
        "selected_odds": record.get("selected_odds"),
        "flat_stake": FLAT_STAKE if status == BET_STATUS_LIVE else None,
        "provenance": None,  # stamped by stamp_bet_list_provenance
        "validation_type": None,  # stamped by stamp_bet_list_provenance
        # THE ROW'S OWN OBSERVATION TIME. Both LIVE and SUPPRESSED rows carry it: a suppressed
        # row is part of the record (D31-21) and was decided at the same instant the live rows
        # were, so a stamp on only the live half would give the completeness query's two classes
        # different evidentiary weight.
        DECIDED_AT_COLUMN: decided_at,
        "grading_status": GRADING_STATUS_PENDING,
        # The decision-time CLV the selector measured, carried onto the row unchanged. It is the
        # REPORT-ONLY model-edge CLV (D27-06) and it is NOT the forward freeze-vs-close metric --
        # the silver odds store holds exactly ONE snapshot per game, so no closing line exists to
        # compare a freeze price against. Recorded as a deferred item rather than fabricated.
        "clv": record.get("clv"),
        "outcome": None,
        "payout_flat": None,
        "realized_units": None,
        "graded_at": None,
    }


def records_to_bet_list_frame(
    result: SelectionResult,
    fits: dict[str, WeeklyChainFit],
    *,
    run_mode: str,
    bankroll: float = DEFAULT_BANKROLL,
    decided_at: datetime | None = None,
) -> pd.DataFrame:
    """Turn a whole ``SelectionResult`` into the stamped ``BET_LIST_COLUMNS`` frame.

    Selected records become ``status='live'``; every rejected record becomes a SUPPRESSED row
    carrying its own ``rejection_reason`` from the selector's own taxonomy. The two sets are the
    EXACT partition of the universe, so a candidate is never silently dropped (SPEC R6).

    THE OBSERVATION TIME IS STAMPED HERE, AND ONLY HERE (D33-27, R7). Every row a FORWARD run
    emits carries *decided_at* -- this run's own wall clock -- rendered through
    :func:`scripts.ingest_historical_odds.require_aware_snapshot_ts`, so a NAIVE clock raises
    rather than being assumed UTC or Eastern (T-33-23). A FORWARD run with no *decided_at* takes
    the current instant, because a forward row that makes no claim about when it was decided is
    refused downstream anyway and defaulting to "now" is the only honest value available.

    A REPLAY RUN STAMPS NOTHING, and a caller who passes *decided_at* anyway is REFUSED rather
    than silently ignored. A replay row is derived and fully regenerable, so it observed nothing;
    dropping the argument quietly would leave the caller believing a time had been recorded.

    Args:
        result: The selector's whole output -- both partitions.
        fits: The per-target frozen fits, read for their EV floors only.
        run_mode: ``"forward"`` or ``"replay"``.
        bankroll: The notional bankroll the unit conversion is expressed against.
        decided_at: This run's instant. Injected rather than read from the clock so the stamp is
            testable; required to be timezone-aware; forbidden under ``run_mode='replay'``.

    Raises:
        ValueError: when *decided_at* is supplied under ``run_mode='replay'``.
        NaiveTimestampError: when *decided_at* carries no timezone.
    """
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    if run_mode == RUN_MODE_REPLAY:
        if decided_at is not None:
            msg = (
                "records_to_bet_list_frame: decided_at was supplied under run_mode='replay'. A "
                "replay row is derived and fully regenerable, so it observed nothing and carries "
                "no observation time; the argument is refused rather than dropped, because "
                "dropping it would leave the caller believing a time had been recorded."
            )
            raise ValueError(msg)
        stamp: str | None = None
    else:
        stamp = require_aware_snapshot_ts(
            decided_at if decided_at is not None else datetime.now(tz=UTC)
        ).isoformat()

    unit = bankroll * UNIT_FRACTION_OF_BANKROLL
    rows: list[dict[str, Any]] = [
        _row_from_record(
            record,
            status=BET_STATUS_LIVE,
            rejection_reason=None,
            ev_floor_t=fits[str(record["target"])].ev_floor_t,
            unit=unit,
            decided_at=stamp,
        )
        for record in result.selected
    ]
    rows += [
        _row_from_record(
            record,
            status="suppressed",
            rejection_reason=str(record["rejection_reason"]),
            ev_floor_t=fits[str(record["target"])].ev_floor_t,
            unit=unit,
            decided_at=stamp,
        )
        for record in result.rejected
    ]

    frame = pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))
    stamped = stamp_bet_list_provenance(frame, run_mode)
    return stamped[BET_LIST_COLUMNS]


def _require_run_mode(run_mode: str) -> None:
    """Refuse a run mode outside the two-value vocabulary, before anything is loaded."""
    if run_mode not in (RUN_MODE_FORWARD, RUN_MODE_REPLAY):
        msg = (
            f"run_mode {run_mode!r} is outside the vocabulary "
            f"('{RUN_MODE_FORWARD}', '{RUN_MODE_REPLAY}')."
        )
        raise ValueError(msg)


def build_weekly_decision_frame(
    season: int,
    week: int,
    *,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    chain_fit_path: Path | str = DEFAULT_CHAIN_FIT_PATH,
    run_mode: str = RUN_MODE_FORWARD,
    bankroll: float = DEFAULT_BANKROLL,
    now: datetime | None = None,
    fits: dict[str, WeeklyChainFit] | None = None,
    strategies: list[Any] | None = None,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """One week's decisions -- every ``status`` and ``rejection_reason`` -- and NO write.

    THE PURE SEAM. It selects the week and stamps the result through the EXISTING
    :func:`records_to_bet_list_frame`, then RETURNS the frame. It persists nothing: no
    durable artifact, no tracker, no ``outputs/`` path touched, no cache connection opened.

    WHY IT HAS TO EXIST (Codex HIGH, raised on both Plan 33-07 and Plan 33-18).
    :func:`build_weekly_candidates` returns feature/odds candidates carrying neither
    decision column; ``status`` and ``rejection_reason`` are created by
    :func:`records_to_bet_list_frame`, and until this function the only caller that reached
    it also WROTE the result. So "what did this week decide?" could not be asked without
    also recording that the week had been decided. R9's no-edge-week criterion needs the
    first without the second, and so does Plan 33-18's acceptance run.

    IT IS THE SAME DECISION PATH THE PERSISTING ONE USES, NOT A SECOND ONE. The weekly
    entry point delegates HERE and then persists what comes back, so there is one decision
    path with a persistence step bolted on rather than two that agree today and drift
    tomorrow. This repository has been bitten by duplicated logic three times, which is why
    the delegation is a call rather than a copy.

    Args:
        season: The season to select.
        week: The week to select.
        artifacts_dir: Where the deployed model artifacts live.
        gold_dir: Where the per-target gold matrices live.
        silver_dir: Where the schedule and the odds snapshot live.
        chain_fit_path: The pre-registered tune-only fit, read when *fits* is omitted.
        run_mode: ``"forward"`` or ``"replay"``.
        bankroll: The notional bankroll the unit conversion is expressed against.
        now: This run's instant. Injected rather than read from the clock so the stamp is
            testable; a FORWARD run with none takes the current instant.
        fits: Pre-resolved per-target frozen fits. Passed in by a caller that has already
            read them, so the record is read ONCE per run rather than twice.
        strategies: A pre-built strategy registry, for the same reason: selection and
            grading must share one registry or a bet can be graded under a rule it was not
            selected under.
        excluded_game_ids: Games the run has decided not to bet (D33.2-05). Passed straight
            to :func:`build_weekly_candidates`, which drops them from the schedule spine, so
            they produce NO row of any kind. This pass-through is load-bearing: the weekly
            entry point reaches the candidates ONLY through this function.

    Returns:
        The stamped ``BET_LIST_COLUMNS`` frame -- one row per (game, target), live or
        suppressed, with every suppression carrying its own reason. Empty when every
        scheduled game is excluded.

    Raises:
        ValueError: for a run mode outside the vocabulary, a week with no scheduled games,
            or a gold matrix carrying no rows for the week.
        FrozenChainFitError: when the pre-registered fit cannot be resolved.
    """
    _require_run_mode(run_mode)

    resolved_fits = load_frozen_chain_fit(chain_fit_path) if fits is None else fits
    _require_season_covered(resolved_fits, season)
    registry = build_strategies(resolved_fits) if strategies is None else strategies

    candidates, schedule = build_weekly_candidates(
        season,
        week,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        excluded_game_ids=excluded_game_ids,
    )
    if schedule.empty:
        # Every scheduled game was excluded: the week is all-skipped (SPEC R3). There is no
        # universe to select over, so the honest frame is empty -- not a refusal, and not a
        # suppressed row for a game the run chose not to bet.
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    result = select_weekly_bets(
        candidates, schedule, resolved_fits, strategies=registry, bankroll=bankroll
    )

    # ONE run instant, so the caller's fence and this stamp cannot disagree by whatever the
    # selection took on a slow week.
    run_instant = now if now is not None else datetime.now(tz=UTC)
    return records_to_bet_list_frame(
        result,
        resolved_fits,
        run_mode=run_mode,
        bankroll=bankroll,
        decided_at=run_instant if run_mode == RUN_MODE_FORWARD else None,
    )


# ---------------------------------------------------------------------------
# The durable artifact: read, upsert, write
# ---------------------------------------------------------------------------


def read_bet_list_with_schema_shim(path: Path | str) -> pd.DataFrame:
    """Read a stored bet list, filling an ABSENT ``decided_at_utc`` with NULL. READ ONLY.

    THE SHIM IS A READ, AND ONLY A READ. A bet-list parquet written before this phase carries
    TWENTY-EIGHT columns (22 immutable + 6 grading) -- measured, not assumed: the production
    artifact is 234 rows by 28 columns. The ``decided_at_utc`` bump takes the schema to 29, and
    without this shim every existing reader of that file would raise the moment the constant
    moved.

    IT MUST NEVER WRITE THE SHIMMED COLUMN BACK, and that is this plan's own named prohibition
    rather than an implementation preference. Every stored row is a ``backtest_replay`` row that
    is already past its freeze, so stamping an observation time onto it now -- from
    ``snapshot_ts``, from the file's mtime, from anything -- would record a time at which
    nobody observed anything. The 234 rows take NULL and are never backfilled.

    Args:
        path: The stored ``bet_list.parquet``.

    Returns:
        The frame in ``BET_LIST_READ_COLUMNS`` order, with ``decided_at_utc`` present.

    Raises:
        ValueError: when a column OTHER than ``decided_at_utc`` is missing. A file written by a
            different schema is refused rather than merged; only the one column this phase adds
            is filled.
    """
    stored = pd.read_parquet(Path(path))
    if DECIDED_AT_COLUMN not in stored.columns:
        stored = stored.copy()
        stored[DECIDED_AT_COLUMN] = None

    missing = [c for c in BET_LIST_READ_COLUMNS if c not in stored.columns]
    if missing:
        msg = (
            f"the stored bet list at {Path(path).as_posix()} is missing column(s) {missing}; it "
            "was written by a different schema and is refused rather than merged."
        )
        raise ValueError(msg)
    return stored[list(BET_LIST_READ_COLUMNS)]


def read_bet_list_artifact(output_dir: Path = DEFAULT_BET_LIST_DIR) -> pd.DataFrame:
    """The stored bet list, or an EMPTY frame with the locked columns when none exists yet.

    THE ONE READER OF THE STORED ARTIFACT, and therefore the one place the back-compat shim has
    to be wired: ``data.graded_weeks`` and ``read_bet_list_cache_sources`` both come through
    here rather than reading the parquet themselves, so routing this function through
    :func:`read_bet_list_with_schema_shim` routes every reader in the tree.

    The return is narrowed to ``BET_LIST_COLUMNS`` -- the LIVE locked order -- so this function's
    contract is unchanged by the shim: before the schema bump the shimmed column is dropped
    here, and after it the two orders are identical.
    """
    path = Path(output_dir) / BET_LIST_ARTIFACT_NAME
    if not path.exists():
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    return read_bet_list_with_schema_shim(path)[BET_LIST_COLUMNS]


def read_bet_tracker_artifact(
    output_dir: Path = DEFAULT_BET_LIST_DIR,
) -> pd.DataFrame:
    """The stored tracker blocks, or an EMPTY frame with the locked columns when none exists yet.

    The mirror of :func:`write_bet_tracker_artifact`. JSON ``null`` is read back as ``None`` and
    left alone: a NULL ``hit_rate`` beside ``bets_graded = 0`` means the rate was NOT COMPUTED,
    which is a different claim from a measured zero, and coercing it to NaN here would erase the
    distinction the tracker exists to preserve.

    ABSENT AND CORRUPT ARE DIFFERENT, AND BOTH ARE HANDLED (WR-04). An absent artifact degrades
    to an empty frame, because a first-ever build legitimately has nothing to read and
    :func:`read_bet_list_cache_sources` promises exactly that. A TRUNCATED or otherwise
    unparseable one is refused by NAME instead: it used to reach ``json.loads`` unguarded and
    raise ``json.JSONDecodeError``, which escaped ``read_bet_list_cache_sources`` -- whose
    docstring promised degradation -- and aborted the whole cache population, a step registered
    NON-CRITICAL precisely so it could not do that. It is re-raised as the same ``ValueError``
    shape the schema-mismatch branch below uses, so a caller has ONE exception type to catch and
    a message that says which file and why.
    """
    path = Path(output_dir) / BET_TRACKER_ARTIFACT_NAME
    if not path.exists():
        return pd.DataFrame(columns=pd.Index(BET_TRACKER_BLOCK_COLUMNS))

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = (
            f"the stored bet tracker at {path.as_posix()} is not parseable JSON ({exc}); it is "
            "truncated or corrupt and is refused rather than merged. Regenerate the artifact "
            "pair with `uv run python scripts/generate_bet_list.py`."
        )
        raise ValueError(msg) from exc
    if not records:
        return pd.DataFrame(columns=pd.Index(BET_TRACKER_BLOCK_COLUMNS))

    stored = pd.DataFrame(records)
    missing = [c for c in BET_TRACKER_BLOCK_COLUMNS if c not in stored.columns]
    if missing:
        msg = (
            f"the stored bet tracker at {path.as_posix()} is missing column(s) {missing}; it was "
            "written by a different schema and is refused rather than merged."
        )
        raise ValueError(msg)
    return stored[BET_TRACKER_BLOCK_COLUMNS]


def build_bet_week_schedule(silver_dir: Path = Path("data/silver")) -> pd.DataFrame:
    """The full schedule with each game's OWN day-before-kickoff lock instant. READ ONLY.

    The source of BOTH schedule-derived cache tables: ``available_bet_weeks`` (navigation) and
    ``bet_week_freeze`` (the threshold the stale-cache hard-block compares the populated-at marker
    against). Deriving them from the SCHEDULE rather than from bet rows is load-bearing in two
    separate ways:

    * a scheduled week carrying no prediction row is still reachable in the week selector; and
    * the freeze threshold exists even when the week has NO bet rows at all -- which is precisely
      the state a failed bet-list insertion leaves behind, and the state the hard-block was built
      for (REVIEW-STALE). A guard reading its own threshold from the data it is checking could not
      fire in that case.

    THE LOCK IS PER-GAME, NOT PER-WEEK (D31-18, D33.2-01), and it is not re-derived here: every
    value comes from ``utils.game_lock`` through ``scripts.ingest_historical_odds.gameday_lock``.
    A Thursday game locks on the Wednesday and that week's Sunday games on the Saturday, so a
    second implementation that rounded to a week would be wrong for every Thursday game. The
    column keeps its published name ``game_freeze_ts`` (renaming it is HOST-07's schema change);
    ``api.cache.materialize_bet_week_freeze`` takes the per-week MAXIMUM of these values.

    The import is deferred because ``scripts.ingest_historical_odds`` imports
    ``backtest.ev_chain_constants``, which imports ``backtest.ou_monetization``, which imports
    ``backtest.bet_selector`` -- which THIS module imports at module scope. A module-level import
    would close that cycle and fail at collection. This is the same cycle break
    ``backtest.bet_selector._freshness_context`` documents, for the same reason.

    Args:
        silver_dir: The silver layer holding ``games.parquet``.

    Returns:
        A frame of ``game_id``, ``season``, ``week``, ``game_freeze_ts`` -- one row per scheduled
        game. An EMPTY frame with those columns when the schedule is absent, so a checkout without
        a silver layer still builds a cache whose ``/bets`` renders its no-current-week state
        rather than failing the population step.
    """
    from scripts.ingest_historical_odds import gameday_lock

    games_path = Path(silver_dir) / "games.parquet"
    columns = ["game_id", "season", "week", "game_freeze_ts"]
    if not games_path.exists():
        logger.warning(
            "Schedule absent -- bet navigation and freeze tables will be empty",
            path=games_path.as_posix(),
        )
        return pd.DataFrame(columns=pd.Index(columns))

    games = pd.read_parquet(
        games_path, columns=["game_id", "season", "week", "kickoff_et"]
    )
    if games.empty:
        return pd.DataFrame(columns=pd.Index(columns))

    # The Eastern gameday through the ONE date accessor, exactly as `_with_gameday` reads it.
    gameday = _with_gameday(games)["gameday"]
    gameday.index = games.index

    # One lock per DISTINCT gameday rather than per game (about 1,300 against 6,500), then
    # mapped back. Same values, and it keeps the single-source rule affordable over full history.
    # Stored as UTC, the representation this column has always carried.
    lock_by_gameday = {
        day: gameday_lock(day).astimezone(UTC)
        for day in sorted(gameday.dropna().unique())
    }

    schedule = games[["game_id", "season", "week"]].copy()
    schedule["game_freeze_ts"] = gameday.map(lock_by_gameday)
    return schedule.dropna(subset=["game_freeze_ts"]).reset_index(drop=True)[columns]


@dataclass(frozen=True)
class BetListCacheSources:
    """The three frames ``api.cache.populate_cache`` loads into its TEMPORARY database.

    Bundled into one object so the scheduled step (``pipeline/steps.py``) and the manual recovery
    command (``scripts/populate_cache.py``) cannot read a DIFFERENT set of sources -- the manual
    command is the one the ``/bets`` refusal text tells the reader to run, so a cache it builds
    that lacks the freeze table would leave the refusal permanently unrecoverable.
    """

    bet_list: pd.DataFrame
    tracker: pd.DataFrame
    schedule: pd.DataFrame


def read_bet_list_cache_sources(
    output_dir: Path = DEFAULT_BET_LIST_DIR,
    silver_dir: Path = Path("data/silver"),
) -> BetListCacheSources:
    """Read every bet-list cache source: the durable rows, the tracker blocks and the schedule.

    THE SEAM (REVIEW-IMPORT). ``api/cache.py`` may import no ``backtest`` module, so it cannot know
    these filenames or derive a per-game freeze. This function -- called from ``pipeline/steps.py``
    and ``scripts/populate_cache.py``, both of which may -- reads them and hands over frames.

    Each reader degrades to an EMPTY frame rather than raising when its source is ABSENT, because
    the cache population step is registered NON-CRITICAL: a raise here would degrade the whole
    Friday run for a first-ever build that simply has nothing to load yet. An absent bet list is
    then represented HONESTLY downstream -- zero rows and no populated-at marker -- which is what
    makes ``/bets`` refuse the week rather than render it as one in which nothing was recommended.

    ABSENT IS NOT CORRUPT (WR-04). A source that EXISTS but cannot be read is refused loudly, as
    a named ``ValueError``, and is not degraded: silently treating a corrupt ledger as "nothing
    recorded yet" would publish a cache that claims no bets were recommended when the record of
    them is merely unreadable. That is the one thing this whole path exists not to do. The
    downstream population step catches nothing, so the refusal reaches the operator.
    """
    return BetListCacheSources(
        bet_list=read_bet_list_artifact(output_dir),
        tracker=read_bet_tracker_artifact(output_dir),
        schedule=build_bet_week_schedule(silver_dir),
    )


def _replace_atomically(write: Callable[[Path], None], path: Path) -> None:
    """Run *write* against a sibling temp file, then ``os.replace`` it onto *path*.

    WHY EVERY WRITE IN THIS MODULE GOES THROUGH HERE (WR-05). ``bet_list.parquet`` is the
    durable home of forward recommendation history -- it carries the FROZEN forward rows
    :func:`upsert_bet_list_rows` protects under the D31-18 per-game freeze fence -- and it is
    gitignored, so there is no second copy of it anywhere. Writing straight onto the final path
    means an interrupted or failing write (Ctrl-C, a full disk, a killed process) leaves a
    TRUNCATED file, and the next :func:`read_bet_list_artifact` raises inside ``pd.read_parquet``
    on the critical path of :func:`generate_weekly_bet_list`. At that point the frozen forward
    rows for every prior week of the season are simply gone and cannot be regenerated -- that is
    what "durable record" means.

    Plan 31-22 did not introduce the mechanism, it multiplied the OCCASIONS: before it, the only
    writer was step 15 of a scheduled orchestrator run, and now there is a hand-runnable CLI
    whose documented invocation omits ``--output-dir`` and therefore targets the production
    ledger directly -- including the recovery runs ``/bets`` itself instructs the reader to
    perform. ``Path.replace`` (i.e. ``os.replace``) is atomic on POSIX and on Windows, where it maps to
    ``MoveFileEx`` with ``MOVEFILE_REPLACE_EXISTING``, so a failure now leaves the PREVIOUS
    complete artifact in place rather than a corrupt ledger.

    The temp file is a SIBLING, deliberately: the replace is only atomic within a filesystem,
    so a temp under the system temp directory could land on another volume and silently degrade
    to a copy. It is removed on failure so a crashed run does not leave a half-written file that
    the next run's ``os.replace`` would publish.

    THIS IS PER-FILE ATOMICITY, NOT PAIR ATOMICITY. It does not make the bet-list/tracker PAIR
    both-old-or-both-new; see :func:`generate_weekly_bet_list` for what remains open there and
    what makes the remaining case audible instead of silent.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write(tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_bet_list_artifact(
    frame: pd.DataFrame, output_dir: Path = DEFAULT_BET_LIST_DIR
) -> Path:
    """Persist the bet list to its durable parquet artifact and return the path.

    Written through :func:`_replace_atomically`, so this call either leaves the artifact
    completely replaced or leaves the previous one completely intact. See that function for why
    an in-place write of this particular file is not an acceptable failure mode.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BET_LIST_ARTIFACT_NAME
    payload = frame[BET_LIST_COLUMNS]
    _replace_atomically(lambda tmp: payload.to_parquet(tmp, index=False), path)
    logger.info("Wrote bet-list artifact", path=str(path), n_rows=len(frame))
    return path


def write_bet_tracker_artifact(
    tracker_frame: pd.DataFrame, output_dir: Path = DEFAULT_BET_LIST_DIR
) -> Path:
    """Persist the PRECOMPUTED tracker blocks as JSON records and return the path.

    JSON rather than parquet because the frame is a handful of aggregate rows whose ``None`` rate
    fields must survive the round trip as null -- the not-measured / measured-zero distinction the
    tracker exists to preserve. Parquet would promote them to NaN.

    Written through :func:`_replace_atomically` for the same reason its sibling is, and with one
    extra consequence: a truncated JSON file is a shape :func:`read_bet_tracker_artifact` used to
    hit at ``json.loads`` with no guard.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BET_TRACKER_ARTIFACT_NAME
    records = tracker_frame.astype(object).where(tracker_frame.notna(), None)
    text = json.dumps(records.to_dict("records"), indent=2, default=str)
    _replace_atomically(lambda tmp: tmp.write_text(text, encoding="utf-8"), path)
    logger.info(
        "Wrote bet-tracker artifact", path=str(path), n_blocks=len(tracker_frame)
    )
    return path


def _key_frame(frame: pd.DataFrame) -> pd.Series:
    """The row key as a single joined string, so a merge/index cannot mis-pair the parts."""
    if frame.empty:
        return pd.Series([], dtype=object)
    return frame[list(BET_LIST_ROW_KEY)].astype(str).agg("|".join, axis=1)


def _is_frozen(row: pd.Series, now: datetime) -> bool:
    """True when THIS row's own game freeze has passed, so its recommendation facts are settled.

    The fence is PER-GAME (D31-18) and applies to FORWARD rows only: a replay row is derived and
    fully regenerable, so freezing it would pin a reproduction rather than a record.
    """
    if row.get("provenance") != PROVENANCE_FORWARD:
        return False
    freeze_text = row.get("freeze_ts")
    if freeze_text is None or (
        isinstance(freeze_text, float) and math.isnan(freeze_text)
    ):
        return False
    freeze = pd.to_datetime(freeze_text, errors="coerce")
    if pd.isna(freeze):
        return False
    # THE SILENT UTC ASSUMPTION IS GONE (D33-27, T-33-23). A naive stored freeze used to be
    # read AS UTC here, which moves a market-local 6 PM instant four or five hours earlier and
    # silently un-freezes rows for the rest of that window. It now RAISES through the one strict
    # parse helper. Unreachable from real data -- all 234 stored rows carry a tz-aware Eastern
    # string -- which is exactly why converting it to a raise costs nothing and closes the hole.
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    freeze_dt = require_aware_snapshot_ts(freeze)
    return now >= freeze_dt


def assert_decided_at_before_freeze(row: Mapping[str, Any] | pd.Series) -> None:
    """A FORWARD row's own observation time must be AT OR BEFORE its own game freeze (R7).

    THE ASSERTION IS ``<=``, NOT ``<``, and it is the SAME operator as the selection fence.
    Since Plan 33.2-02 the ``freeze_ts`` value is the game's day-before-kickoff LOCK, and the
    selection fence (:func:`select_games_for_decision_instant`) refuses only a decision AFTER
    the lock, so a row decided exactly AT its lock is admitted there and accepted here. The two
    fences agree on the boundary rather than sitting on opposite sides of it, and
    ``tests/unit/test_selection_scoped_by_freeze_instant.py`` proves the conjunction rather than
    arguing it.

    SCOPED TO ``provenance == forward``. A replay row is derived and fully regenerable, so it
    carries no observation time and needs none; the 234 stored ``backtest_replay`` rows take
    NULL and are never backfilled. Inside the forward scope a NULL is a NAMED REFUSAL, because
    a forward row that cannot be checked is exactly the row the check exists for.

    Both instants go through the ONE strict parse helper, so a naive value on either side raises
    rather than being assumed UTC or Eastern.

    Args:
        row: A bet-list row as a mapping (a dict) or as a ``pandas.Series``. Both are accepted
            because both are what the callers actually hold -- the upsert iterates rows as
            Series, and the tests construct dicts -- and a ``Series`` is not a ``Mapping`` to a
            type checker even though ``.get`` behaves identically on it.

    Raises:
        MissingDecidedAtError: a forward row with no ``decided_at_utc`` or no ``freeze_ts``.
        DecidedAfterFreezeError: a forward row decided after its own freeze.
        NaiveTimestampError: either instant carries no timezone.
    """
    if row.get("provenance") != PROVENANCE_FORWARD:
        return

    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    game_id = row.get("game_id")
    decided_text = row.get(DECIDED_AT_COLUMN)
    freeze_text = row.get("freeze_ts")
    for label, value in ((DECIDED_AT_COLUMN, decided_text), ("freeze_ts", freeze_text)):
        if value is None or _is_null(value):
            msg = (
                f"forward row {game_id!r}/{row.get('target')!r} carries no {label}; a forward "
                "row must say when it was decided AND which freeze that claim is measured "
                "against. A row that cannot be checked is exactly the row this check exists "
                "for, so it is refused rather than written."
            )
            raise MissingDecidedAtError(msg)

    decided = require_aware_snapshot_ts(decided_text)
    freeze = require_aware_snapshot_ts(freeze_text)
    if decided > freeze:
        msg = (
            f"refusing to write forward row {game_id!r}/{row.get('target')!r}: its "
            f"{DECIDED_AT_COLUMN} is {decided.isoformat()}, which is AFTER its own game freeze "
            f"{freeze.isoformat()}. That is a post-hoc pick wearing a pre-game timestamp -- the "
            "one claim the forward ledger exists to make unforgeable."
        )
        raise DecidedAfterFreezeError(msg)


def _is_null(value: Any) -> bool:
    """True when *value* is a scalar null. Array-likes are never null for this purpose."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _assert_stored_stamps_untouched(
    stored: pd.DataFrame, carried: pd.DataFrame
) -> None:
    """The stored half's observation times must come out of a merge BYTE-IDENTICAL (T-33-22).

    THE PROHIBITION IS RUNTIME, NOT ONLY STRUCTURAL. ``tests/unit/test_decided_at_utc.py`` scans
    the production tree by AST and proves nobody WROTE a retroactive stamp; this proves nobody
    DOES. The two catch different mistakes: the scan cannot see a stamp introduced by a merge, a
    reindex or a ``fillna`` that never names the column, and those are exactly the ways a value
    appears on a row nobody meant to touch.

    A stored row is a record of a decision already made. Writing an observation time onto it now
    -- from ``snapshot_ts``, from the run clock, from a neighbouring row -- records a time at
    which nobody observed anything, which is the one thing this phase exists to make impossible.
    NULL is a legitimate stored value (all 234 ``backtest_replay`` rows carry it) and must survive
    as NULL; the comparison below therefore treats two NULLs as equal rather than as unequal
    floats.

    Args:
        stored: The stored frame as it was READ, before the merge.
        carried: The subset of it the merge is about to carry forward.

    Raises:
        RuntimeError: when any carried row's ``decided_at_utc`` differs from the stored one.
    """
    if (
        DECIDED_AT_COLUMN not in stored.columns
        or DECIDED_AT_COLUMN not in carried.columns
    ):
        return
    original = stored.loc[carried.index, DECIDED_AT_COLUMN]
    now_values = carried[DECIDED_AT_COLUMN]
    changed = [
        str(stored.loc[index, "game_id"])
        for index, before, after in zip(
            carried.index, original, now_values, strict=True
        )
        if not (_is_null(before) and _is_null(after)) and before != after
    ]
    if changed:
        msg = (
            f"refusing the merge: it altered {DECIDED_AT_COLUMN} on {len(changed)} row(s) it did "
            f"not create ({', '.join(changed[:5])}). A stored row is a record of a decision "
            "already made; stamping an observation time onto it now would record a time at which "
            "nobody observed anything. The 234 stored replay rows keep their NULL."
        )
        raise RuntimeError(msg)


def upsert_bet_list_rows(
    stored: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    now: datetime,
) -> pd.DataFrame:
    """Merge a fresh selection into the stored bet list under the D31-18 per-game freeze fence.

    * A stored FORWARD row whose own game freeze has PASSED wins WHOLE -- the incoming row for
      that key is discarded. Not merely its ``BET_LIST_IMMUTABLE_COLUMNS``: carrying the entire
      stored row forward is strictly stronger and additionally stops a later run restating the
      decision-time CLV, which is a grading-half column and would otherwise be overwritten.
    * Every other key takes the incoming row, because before the freeze the games have not been
      priced and frozen yet and late odds can still land.
    * A stored key the incoming selection does not mention is KEPT. A week the run did not touch
      is history, not an absence.

    TWO THINGS ARE ASSERTED HERE THAT ARE NOT ABOUT THE MERGE ITSELF (Phase 33, R7, D33-27):

    * EVERY INCOMING FORWARD ROW is checked by :func:`assert_decided_at_before_freeze` BEFORE any
      merging happens, so a row decided after its own game freeze -- or one that makes no claim
      at all -- is refused by name rather than written. The check is here because this is the one
      merge every writer goes through; an assertion function nobody calls is a comment. It is
      applied to the INCOMING half only: the stored half was checked when it was written, and
      re-checking it would make a legitimately NULL-stamped replay row unwritable forever.
    * THE UPSERT NEVER SETS ``decided_at_utc`` ON A ROW IT DID NOT CREATE. Stored rows are carried
      through by column selection alone and :func:`_assert_stored_stamps_untouched` proves it at
      runtime. That is this plan's named prohibition, and it is why the 234 stored
      ``backtest_replay`` rows keep their NULL through every future run.

    Args:
        stored: The bet list already on disk (possibly empty).
        incoming: This run's freshly selected rows.
        now: The instant the freeze is judged against. Injected rather than read from the clock so
            the before-freeze and after-freeze behaviours are both testable.

    Returns:
        The merged frame in ``BET_LIST_COLUMNS`` order.

    Raises:
        MissingDecidedAtError: an incoming forward row carries no observation time.
        DecidedAfterFreezeError: an incoming forward row was decided after its own game freeze.
        RuntimeError: the merge altered a stored row's observation time.
    """
    for _index, row in incoming.iterrows():
        assert_decided_at_before_freeze(row)

    if stored.empty:
        return incoming[BET_LIST_COLUMNS].reset_index(drop=True)
    if incoming.empty:
        return stored[BET_LIST_COLUMNS].reset_index(drop=True)

    stored = stored.reset_index(drop=True)
    incoming = incoming.reset_index(drop=True)
    stored_keys = _key_frame(stored)
    incoming_keys = _key_frame(incoming)

    frozen_keys = {
        key
        for key, (_, row) in zip(stored_keys, stored.iterrows(), strict=True)
        if _is_frozen(row, now)
    }

    superseded = set(incoming_keys) - frozen_keys
    kept_stored = stored[~stored_keys.isin(superseded)]
    accepted_incoming = incoming[~incoming_keys.isin(frozen_keys)]
    _assert_stored_stamps_untouched(stored, cast("pd.DataFrame", kept_stored))

    merged = pd.concat(
        [kept_stored[BET_LIST_COLUMNS], accepted_incoming[BET_LIST_COLUMNS]],
        ignore_index=True,
    )
    logger.info(
        "Upserted bet-list rows",
        n_stored=len(stored),
        n_incoming=len(incoming),
        n_frozen_preserved=len(frozen_keys & set(incoming_keys)),
        n_merged=len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# The grading pass
# ---------------------------------------------------------------------------


def _payout_multiple(outcome: bool | None, selected_odds: float | None) -> float:
    """The per-unit-staked profit of a settled bet, at the price the bet was ACTUALLY struck at.

    Identical in form to ``backtest.profitability_2025._per_bet_frame``: a win pays
    ``american_to_payout(selected_odds)``, a loss pays -1.0, a push pays 0.0. A flat -110 payout on
    a -320 favourite would turn a losing bet into a winning one on paper.
    """
    if outcome is True:
        if selected_odds is None:
            msg = (
                "cannot settle a winning bet with no stored price; ``selected_odds`` is written "
                "at selection and a settled row without one is a corrupt record, not a default."
            )
            raise ValueError(msg)
        return float(american_to_payout(int(selected_odds)))
    if outcome is False:
        return -1.0
    return 0.0


def grade_row(
    row: dict[str, Any],
    outcome: bool | None,
    *,
    graded_at: datetime,
) -> dict[str, Any]:
    """Settle ONE pending row, ONE-WAY and ONE-TIME (REVIEW-FWD-GRADE, T-31-86b).

    Every ``BET_LIST_IMMUTABLE_COLUMNS`` value is carried through byte-identical; only the six
    grading columns move.

    Args:
        row: The stored bet-list row.
        outcome: True (win), False (loss) or None (push). A row whose realized value is UNKNOWN
            must not be passed here at all -- it stays ``pending``.
        graded_at: The settlement instant.

    Returns:
        A NEW row dict; the caller's row is not mutated.

    Raises:
        AlreadyGradedError: when the row's ``grading_status`` is already terminal. A later run may
            not quietly restate a result.
    """
    current = str(row.get("grading_status") or GRADING_STATUS_PENDING)
    if current != GRADING_STATUS_PENDING:
        msg = (
            f"refusing to re-grade {row.get('game_id')!r}/{row.get('target')!r}: its "
            f"grading_status is already {current!r}. The transition out of "
            f"{GRADING_STATUS_PENDING!r} is one-way and one-time, so a later run cannot restate a "
            "settled result."
        )
        raise AlreadyGradedError(msg)

    if outcome is True:
        new_status = GRADING_STATUS_WIN
    elif outcome is False:
        new_status = GRADING_STATUS_LOSS
    else:
        new_status = GRADING_STATUS_PUSH
    assert_grading_transition(current, new_status)

    multiple = _payout_multiple(outcome, row.get("selected_odds"))
    stake_units = row.get("stake_units")
    graded = dict(row)
    graded.update(
        {
            "grading_status": new_status,
            "outcome": outcome,
            "payout_flat": FLAT_STAKE * multiple,
            "realized_units": (
                None if stake_units is None else float(stake_units) * multiple
            ),
            "graded_at": graded_at,
        }
    )
    return graded


def _realized_values(
    frame: pd.DataFrame, gold_dir: Path
) -> dict[str, dict[str, float]]:
    """The realized label per (target, game_id) for every game the frame mentions. READ ONLY."""
    wanted = set(frame["game_id"].astype(str))
    realized: dict[str, dict[str, float]] = {}
    for target in CANONICAL_TARGETS:
        path = Path(gold_dir) / f"features_{target}.parquet"
        label = _REALIZED_LABEL_COLUMN[target]
        if not path.exists():
            realized[target] = {}
            continue
        gold = pd.read_parquet(path, columns=["game_id", label])
        gold = gold[gold["game_id"].astype(str).isin(wanted)]
        gold = gold[gold[label].notna()]
        realized[target] = {
            str(game_id): float(value)
            for game_id, value in zip(gold["game_id"], gold[label], strict=True)
        }
    return realized


def grade_pending_rows(
    frame: pd.DataFrame,
    strategies: dict[str, Any],
    *,
    gold_dir: Path = Path("data/gold"),
    graded_at: datetime | None = None,
    realized: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Settle every LIVE, PENDING row whose game result is now known.

    The outcome is resolved by the row's OWN target strategy -- the LOCKED
    ``_resolve_wp_outcome`` / ``_resolve_ats_outcome`` / ``_resolve_ou_outcome`` convention -- and
    is never re-implemented here. A row whose realized value is still unknown stays ``pending``,
    which is what lets the tracker say "not measured" instead of "measured zero".

    The pass FILTERS to pending rows, so re-running it is idempotent; :func:`grade_row` is the
    enforcement point that refuses a second attempt on a settled row.

    Args:
        frame: The merged bet list.
        strategies: ``{target -> TargetStrategy}``, taken from the selector's own registry.
        gold_dir: Where the realized labels are read from.
        graded_at: The settlement instant. Defaults to now (UTC).
        realized: Pre-resolved ``{target -> {game_id -> realized value}}``, used INSTEAD of reading
            gold. The injection seam the tests drive.

    Returns:
        A NEW frame; the caller's frame is not mutated.
    """
    if frame.empty:
        return frame

    settled_at = graded_at or datetime.now(tz=UTC)
    lookup = realized if realized is not None else _realized_values(frame, gold_dir)

    graded_rows: list[dict[str, Any]] = []
    n_graded = 0
    for row in frame.to_dict("records"):
        pending = str(row.get("grading_status") or "") == GRADING_STATUS_PENDING
        live = row.get("status") == BET_STATUS_LIVE
        target = str(row.get("target"))
        value = lookup.get(target, {}).get(str(row.get("game_id")))
        if not (pending and live) or value is None:
            graded_rows.append(row)
            continue

        strategy = strategies.get(target)
        if strategy is None:
            msg = (
                f"no strategy registered for target {target!r}; a stored row cannot be graded by "
                "a rule that is not present, and guessing one would book a result under the "
                "wrong target's convention."
            )
            raise ValueError(msg)

        outcome = strategy.grade(
            {
                "bet_side": row.get("bet_side"),
                "slipped_line": row.get("slipped_line"),
                "_actual_total": value,
            }
        )
        graded_rows.append(grade_row(row, outcome, graded_at=settled_at))
        n_graded += 1

    logger.info("Graded pending bet-list rows", n_rows=len(frame), n_graded=n_graded)
    return pd.DataFrame(graded_rows, columns=pd.Index(BET_LIST_COLUMNS))


# ---------------------------------------------------------------------------
# The ONE weekly entry point
# ---------------------------------------------------------------------------


def generate_weekly_bet_list(
    season: int,
    week: int,
    *,
    output_dir: Path = DEFAULT_BET_LIST_DIR,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    chain_fit_path: Path | str = DEFAULT_CHAIN_FIT_PATH,
    run_mode: str = RUN_MODE_FORWARD,
    bankroll: float = DEFAULT_BANKROLL,
    now: datetime | None = None,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Select one week, merge it into the durable artifacts, grade what is settled, and persist.

    *excluded_game_ids* (keyword-only, empty by default) names games the run has decided not
    to bet (D33.2-05's live skip, wired by Plan 33.2-03). They are passed through
    :func:`build_weekly_decision_frame` to :func:`build_weekly_candidates`, which drops them
    from the schedule spine, so neither artifact of the pair gains a row of any kind for them.

    THE single weekly selection entry point (SPEC R4, D31-31). It writes NOTHING under ``data/``
    and opens NO connection to the live cache: the write is the PAIR of artifacts under
    ``output_dir``, which the Plan 31-18 population step loads into the temp cache build before
    the atomic swap.

    THE WRITE IS THE PAIR, AND NO CALLER CAN PRODUCE HALF OF IT (Plan 31-22, T-31-114). Both
    ``BET_LIST_ARTIFACT_NAME`` and ``BET_TRACKER_ARTIFACT_NAME`` are written here, from the SAME
    graded frame. ``/bets`` reads both through ONE reader,
    :func:`read_bet_list_cache_sources`, so a caller that produced only the parquet would leave
    the realized-versus-expected tracker permanently EMPTY while the page still looked correct --
    a page that has quietly stopped grading itself and does not say so. That is not a cosmetic
    gap: the tracker is the honesty half of the feature.

    WHAT "INDIVISIBLE" DOES AND DOES NOT MEAN (WR-04, correcting an earlier overclaim). An
    earlier version of this docstring said the pair was indivisible full stop. That is true of
    CALLERS -- the structural guard in ``tests/unit/test_bet_list_entry_point.py`` makes a second
    producer of half the pair impossible -- and it was never true of EXECUTION. The two writes
    below are sequential statements with an aggregation between them, so a process that dies, or
    an ``aggregate_all_blocks`` that raises, still leaves the first artifact written and the
    second not. Two things now bound that:

    * Each write is individually ATOMIC (:func:`_replace_atomically`), so neither file can be
      left TRUNCATED. Whatever is on disk afterwards is a complete artifact.
    * The remaining case -- a new parquet beside the PREVIOUS run's complete tracker, or beside
      none at all -- is now AUDIBLE rather than silent: ``api.cache.populate_cache`` warns when
      it is handed bet rows without tracker blocks
      (``api.cache.bet_tracker_source_is_absent``), and a corrupt tracker is refused by name
      rather than escaping as ``json.JSONDecodeError``
      (:func:`read_bet_tracker_artifact`).

    What is still OPEN, and is recorded rather than glossed: a new parquet beside a STALE but
    complete tracker is not detectable by absence, so the population cannot warn about it. Making
    the pair genuinely both-old-or-both-new needs a two-phase commit -- stage both temp files,
    then publish both -- which changes the writers' interface and is a larger change than a
    review fix should make to the path that owns the durable ledger.

    WHY THE AGGREGATION LIVES HERE (REVIEW-IMPORT, T-31-117). It used to live in
    ``pipeline/steps.py::step_generate_recommendations``, which meant the tracker was written by
    the SCHEDULED STEP rather than by this function -- so any second caller of this function
    produced half the pair. It is here now so the scheduled step and the manual command
    (``scripts/generate_bet_list.py``) cannot produce different artifact SETS. It is NOT in
    ``api/cache.py``, and could not be: that module may import no ``backtest`` module (UIAP-01,
    ``tests/api/test_import_guard_bets.py``), whose allow-list this move does not widen. The
    pure-persistence seam is unchanged; only the aggregation's position INSIDE ``backtest/``
    moved.

    Returns:
        The merged, graded bet list -- the same frame both artifacts were written from.
    """
    _require_run_mode(run_mode)

    fits = load_frozen_chain_fit(chain_fit_path)
    _require_season_covered(fits, season)
    # ONE registry, shared by selection and grading, so a bet is graded under exactly the rule it
    # was selected under. Built HERE and handed to the decision seam rather than built inside it,
    # because the grading pass below needs the same objects.
    strategies = build_strategies(fits)
    # ONE run instant, used for BOTH the stamp and the fence. Two clock reads would let a row
    # claim an observation time the freeze was not judged at, which is a gap of milliseconds
    # today and a gap of whatever the selection takes on a slow week.
    run_instant = now if now is not None else datetime.now(tz=UTC)

    # THE DECISION IS DELEGATED, NOT DUPLICATED (Plan 33-07 Task 1). Everything from the
    # candidate build through the status stamp lives in :func:`build_weekly_decision_frame`,
    # which writes nothing; this function is that call plus persistence. Two copies of the
    # decision would agree until the first change to either.
    incoming = build_weekly_decision_frame(
        season,
        week,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        run_mode=run_mode,
        bankroll=bankroll,
        now=run_instant,
        fits=fits,
        strategies=strategies,
        excluded_game_ids=excluded_game_ids,
    )

    stored = read_bet_list_artifact(output_dir)
    merged = upsert_bet_list_rows(stored, incoming, now=run_instant)

    graded = grade_pending_rows(
        merged,
        {strategy.target: strategy for strategy in strategies},
        gold_dir=gold_dir,
    )
    write_bet_list_artifact(graded, output_dir)
    # The tracker is aggregated from the SAME graded frame that was just written, not from a
    # re-read of the artifact: a re-read would let the two halves describe different rows if a
    # write partially failed.
    blocks = aggregate_all_blocks(graded)
    write_bet_tracker_artifact(to_tracker_frame(blocks), output_dir=output_dir)

    logger.info(
        "Weekly bet list generated",
        season=season,
        week=week,
        run_mode=run_mode,
        n_live=int((graded["status"] == BET_STATUS_LIVE).sum()),
        n_rows=len(graded),
        n_tracker_blocks=len(blocks),
    )
    return graded


# The ONE assertion this module makes about the schema it writes: the immutable half named by
# ``api.cache`` is exactly the half the freeze fence protects. Stated as an import-time check so a
# column added to one list and not the other cannot ship silently.
if set(BET_LIST_IMMUTABLE_COLUMNS) - set(BET_LIST_COLUMNS):  # pragma: no cover
    msg = "BET_LIST_IMMUTABLE_COLUMNS names a column absent from BET_LIST_COLUMNS"
    raise RuntimeError(msg)

# The second import-time check: once ``api.cache`` carries ``decided_at_utc``, the shim's read
# order must BE the locked order rather than a second 29-column order that agrees with it today.
# Stated here so the collapse the ``dict.fromkeys`` derivation relies on cannot fail silently.
if (
    DECIDED_AT_COLUMN in BET_LIST_COLUMNS
    and tuple(BET_LIST_COLUMNS) != BET_LIST_READ_COLUMNS
):  # pragma: no cover
    msg = (
        "BET_LIST_READ_COLUMNS has drifted from BET_LIST_COLUMNS: the schema shim would return "
        "a different column order from the one the writer and the DDL use"
    )
    raise RuntimeError(msg)
