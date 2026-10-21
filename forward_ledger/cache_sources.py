"""The web-cache sources, cutover-aware: ONE reader for both cache builders (Phase 34, Plan 34-16).

WHY THIS MODULE EXISTS
----------------------
``api.cache.populate_cache`` takes every bet-list source as a FRAME, because ``api/`` may import no
``backtest`` or ``forward_ledger`` module (UIAP-01, D-18). Its two callers -- the scheduled
``pipeline/steps.py::step_populate_web_cache`` and the manual recovery command
``scripts/populate_cache.py`` -- both read through :func:`read_bet_cache_sources`, so they cannot
build caches from different sources.

THE TWO SWITCH STATES (``forward_ledger.cutover``)
--------------------------------------------------
* OFF (today): exactly ``backtest.weekly_bet_list.read_bet_list_cache_sources`` -- the durable
  ``outputs/bet_list`` rows, their stored tracker and the schedule -- with no corrections and an
  undeclared verdict context. Nothing about today's cache changes.
* ON (after go-live): the forward rows come from the CHAIN-VERIFIED ledger and are UNIONED with
  the 2025 ``backtest_replay`` rows that stay in ``outputs/bet_list``; the tracker is recomputed
  over that union with the in-force corrections applied (D-06). Forward rows live in exactly ONE
  place, so a forward row still found in ``outputs/bet_list`` is refused by name -- a half-done
  migration must never be counted twice (research Pitfall 2).

ABSENT IS NOT EMPTY
-------------------
With the switch on, an absent ledger is a cutover without its migration, and a ledger whose chain
does not verify is unreadable; both RAISE by name. Serving either as "nothing recommended" would
publish a record that claims no bets were made. The cache step is non-critical, so a refusal
degrades the run and the ``/bets`` stale-cache hard-block fires (T-34-57).

In both states the ``/bets`` result strip's rows are ``backtest.bet_tracker.graded_outcome_rows``
over the very bet-list frame returned, with the same corrections the tracker uses, so the strip and
the tiles are drawn from one frame (review finding 4). Last week's actual vs expected
(``backtest.bet_week_results.aggregate_bet_weeks``, Phase 35 SPEC R5) is computed from that same
frame and those same corrections, so it can never disagree with the tracker about which rows count.

THE RE-FIT COMPARISON PANEL (Phase 37, D-12..D-14)
--------------------------------------------------
With the switch on, the Season page panel's raw numbers and A/A status come from
``refit.comparison.panel_frames`` over the same verified ledger entries, the silver scores and the
ledger's ``refit/`` records. An absent ``games.parquet`` reads as an empty games frame -- every
period then counts 0 paired games -- the same degradation ``build_bet_week_schedule`` applies, so
a cache build never fails for want of scores. A panel that cannot be computed (a stamped id no
generation knows, an unreadable re-fit record, a period with no start date) never stops the build
either (WR-D05): the panel is display-only, so its frames are left empty, the exception CLASS is
logged and handed on as ``refit_panel_failure``, and the Season page says "not available" while
``/bets`` and every other page are built as usual. Ledger chain failures stay fatal. With the
switch off there is no forward ledger to compare, and both frames are empty.

THE EDGE BANDS OF DECIDED GAMES (Phase 37, D-36)
------------------------------------------------
With the switch on, ``refit.generations.decided_edge_thresholds`` maps every game a live bet rule
decided to the edge-band thresholds of the generation that decided it, over the same verified
entries, so the cache bands that game with the band its live bet rule used. With the switch off
the mapping is empty and every game is banded with the in-force generation, as before.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_CORRECTIONS_COLUMNS,
    BET_WEEK_RESULTS_COLUMNS,
    PROVENANCE_FORWARD,
    REFIT_AA_STATUS_COLUMNS,
    REFIT_PANEL_COLUMNS,
)
from backtest.bet_tracker import (
    aggregate_all_blocks,
    graded_outcome_rows,
    to_tracker_frame,
)
from backtest.bet_week_results import aggregate_bet_weeks
from backtest.weekly_bet_list import (
    DEFAULT_BET_LIST_DIR,
    build_bet_week_schedule,
    read_bet_list_artifact,
    read_bet_list_cache_sources,
)
from forward_ledger.corrections import in_force_outcomes
from forward_ledger.cutover import forward_rows_go_to_ledger
from forward_ledger.declarations import VerdictScopeUndeclaredError, load_verdict_scope
from forward_ledger.settle import SILVER_GAMES_FILENAME, load_silver_games
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    LedgerEntry,
    entries_to_frame,
    ledger_path,
    read_entries,
    verify_chain,
)
from refit.comparison import panel_frames
from refit.generations import (
    EdgeThresholds,
    UnknownGenerationError,
    decided_edge_thresholds,
)
from refit.records import RefitRecordFormatError
from utils import get_logger

__all__ = [
    "BetCacheSources",
    "ForwardRowsOutsideLedgerError",
    "LedgerAbsentAfterCutoverError",
    "read_bet_cache_sources",
]

logger = get_logger(__name__)

# The panel's own failures (refit.comparison.panel_frames): an unknown stamped id, an unreadable
# re-fit record, and the ValueError of a period whose cadence date has no event.
_PANEL_FAILURES: tuple[type[Exception], ...] = (
    UnknownGenerationError,
    RefitRecordFormatError,
    ValueError,
)

# The context stamped when no verdict scope is declared (or the switch is off).
_UNDECLARED: dict[str, Any] = {"declared": False, "season": None, "start_week": None}


class ForwardRowsOutsideLedgerError(Exception):
    """Forward rows were found in ``outputs/bet_list`` after the cutover: a second forward store.

    Bare ``Exception`` (the ``forward_ledger.store`` rule): broad ``ValueError`` handlers elsewhere
    degrade to empty results, and a refused source must never read as "nothing recommended".
    """


class LedgerAbsentAfterCutoverError(Exception):
    """The switch is on but the ledger file does not exist: a cutover without its migration."""


def _empty_week_results() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(BET_WEEK_RESULTS_COLUMNS))


def _empty_refit_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(REFIT_PANEL_COLUMNS))


def _empty_refit_aa_status() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(REFIT_AA_STATUS_COLUMNS))


# The silver ``games`` columns the panel reads, for the empty frame an absent store reads as.
_PANEL_GAMES_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_score",
    "away_score",
)


@dataclass(frozen=True)
class BetCacheSources:
    """Every frame ``api.cache.populate_cache`` loads for ``/bets`` and Track Record.

    Attributes:
        bet_list: The bet-list rows in ``BET_LIST_COLUMNS`` order.
        tracker: The precomputed tracker blocks (``BET_TRACKER_BLOCK_COLUMNS``).
        schedule: The schedule with each game's own lock (``build_bet_week_schedule``).
        corrections: The in-force corrections (``BET_LIST_CORRECTIONS_COLUMNS``); empty when off.
        graded_outcomes: The result strip's rows (``BET_GRADED_OUTCOMES_COLUMNS``).
        verdict_context: ``{"declared", "season", "start_week"}`` for the 2026 forward verdict.
        week_results: Last week's actual vs expected rows (``BET_WEEK_RESULTS_COLUMNS``), one per
            season, week and counting class, aggregated from the same rows and corrections as
            the tracker (SPEC R5). Defaults to an empty frame with those columns.
        refit_panel: The Season page panel's raw numbers (``REFIT_PANEL_COLUMNS``, Phase 37).
            Defaults to an empty frame with those columns (the switch off).
        refit_aa_status: The A/A week status rows (``REFIT_AA_STATUS_COLUMNS``). Defaults to an
            empty frame with those columns.
        decided_edge_thresholds: ``game_id -> thresholds`` of the generation whose live bet rule
            decided each game (``refit.generations.decided_edge_thresholds``, D-36). Defaults to
            empty (the switch off): every game is then banded with the in-force generation.
        refit_panel_failure: The exception class when the panel could not be computed (WR-D05);
            the two panel frames are then empty. None when it was computed.
    """

    bet_list: pd.DataFrame
    tracker: pd.DataFrame
    schedule: pd.DataFrame
    corrections: pd.DataFrame
    graded_outcomes: pd.DataFrame
    verdict_context: dict[str, Any]
    week_results: pd.DataFrame = field(default_factory=_empty_week_results)
    refit_panel: pd.DataFrame = field(default_factory=_empty_refit_panel)
    refit_aa_status: pd.DataFrame = field(default_factory=_empty_refit_aa_status)
    decided_edge_thresholds: dict[str, EdgeThresholds | None] = field(
        default_factory=dict
    )
    refit_panel_failure: str | None = None


def _empty_corrections() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(BET_LIST_CORRECTIONS_COLUMNS))


def _verified_ledger_entries(ledger_dir: Path) -> list[LedgerEntry]:
    """The ledger's entries, refused by name when absent or when the chain does not verify."""
    path = ledger_path(ledger_dir)
    if not path.exists():
        msg = (
            f"the forward-rows cutover is on but the ledger {path.as_posix()} does not exist; "
            "a cutover without its migration has no forward record to serve, and serving the "
            "replay rows alone would claim no forward bet was ever made."
        )
        raise LedgerAbsentAfterCutoverError(msg)

    entries = read_entries(ledger_dir)
    verdict = verify_chain(entries)
    if not verdict.ok:
        msg = (
            f"the ledger at {path.as_posix()} does not verify: entry seq "
            f"{verdict.first_broken_seq} (key {verdict.first_broken_key}) -- {verdict.reason}. "
            "A tampered or damaged ledger is refused, never served as an empty record."
        )
        raise LedgerChainBrokenError(msg)
    return entries


def _in_force_corrections(entries: Sequence[LedgerEntry]) -> pd.DataFrame:
    """One row per key whose IN-FORCE outcome is a correction: that entry's recorded facts."""
    by_seq = {entry.seq: entry for entry in entries}
    records = [
        {
            name: by_seq[state.correction_seq].immutable[name]
            for name in BET_LIST_CORRECTIONS_COLUMNS
        }
        for state in in_force_outcomes(entries).values()
        if state.corrected and state.correction_seq is not None
    ]
    if not records:
        return _empty_corrections()
    return pd.DataFrame.from_records(records, columns=BET_LIST_CORRECTIONS_COLUMNS)


def _replay_rows_only(output_dir: Path) -> pd.DataFrame:
    """The ``outputs/bet_list`` rows, refused if any forward row is still among them."""
    stored = read_bet_list_artifact(output_dir)
    forward = int((stored["provenance"] == PROVENANCE_FORWARD).sum())
    if forward:
        msg = (
            f"{forward} forward row(s) found in {Path(output_dir).as_posix()} after the cutover; "
            "forward rows live only in the ledger, and counting them from both stores would "
            "double the forward record. Finish the migration "
            "(`uv run python scripts/migrate_forward_rows.py --finish`) before rebuilding the cache."
        )
        raise ForwardRowsOutsideLedgerError(msg)
    return stored


def _union(replay: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Replay rows then ledger rows, in ``BET_LIST_COLUMNS`` order (an empty side is skipped)."""
    parts = [
        frame.reindex(columns=BET_LIST_COLUMNS)
        for frame in (replay, forward)
        if not frame.empty
    ]
    if not parts:
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    if len(parts) == 1:
        return parts[0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)


def _panel_games(silver_dir: Path) -> pd.DataFrame:
    """The silver ``games`` the panel scores against; an empty frame when the store is absent.

    ``load_silver_games`` refuses an absent store by name, which the settle and verify passes need;
    the panel instead counts 0 paired games, as every other source of this build degrades.
    """
    if not (Path(silver_dir) / SILVER_GAMES_FILENAME).exists():
        return pd.DataFrame(columns=pd.Index(_PANEL_GAMES_COLUMNS))
    return load_silver_games(silver_dir)


def _refit_panel(
    entries: Sequence[LedgerEntry], silver_dir: Path, ledger_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """The panel's two frames, or two empty frames and the exception class when it fails."""
    try:
        panel, aa_status = panel_frames(entries, _panel_games(silver_dir), ledger_dir)
    except _PANEL_FAILURES as error:
        failure = type(error).__name__
        logger.warning(
            "refit panel not computed; the page says not available", error_class=failure
        )
        return _empty_refit_panel(), _empty_refit_aa_status(), failure
    return panel, aa_status, None


def _verdict_context() -> dict[str, Any]:
    """The committed declaration's season and start week; undeclared while none exists.

    A MALFORMED declaration still raises (``VerdictScopeMalformedError``): only an absent one is
    the honest "not declared yet".
    """
    try:
        scope = load_verdict_scope()
    except VerdictScopeUndeclaredError:
        return dict(_UNDECLARED)
    return {"declared": True, "season": scope.season, "start_week": scope.start_week}


def read_bet_cache_sources(
    output_dir: Path = DEFAULT_BET_LIST_DIR,
    ledger_dir: Path = LEDGER_DIR,
    silver_dir: Path = Path("data/silver"),
) -> BetCacheSources:
    """Read every bet-related web-cache source, following the forward-rows cutover switch.

    Args:
        output_dir: The durable ``outputs/bet_list`` directory.
        ledger_dir: The forward ledger directory (read only when the switch is on).
        silver_dir: The silver layer holding ``games.parquet`` (the schedule).

    Returns:
        The :class:`BetCacheSources`.

    Raises:
        ForwardRowsOutsideLedgerError: switch on and ``outputs/bet_list`` holds forward rows.
        LedgerAbsentAfterCutoverError: switch on and the ledger file does not exist.
        forward_ledger.store.LedgerChainBrokenError: switch on and the chain does not verify.
        forward_ledger.store.LedgerFormatError: switch on and a ledger line cannot be read.
        Everything ``read_bet_list_cache_sources`` raises for a corrupt artifact. A panel failure
        never raises here: it is reported as ``refit_panel_failure`` (WR-D05).
    """
    if not forward_rows_go_to_ledger():
        today = read_bet_list_cache_sources(output_dir, silver_dir)
        return BetCacheSources(
            bet_list=today.bet_list,
            tracker=today.tracker,
            schedule=today.schedule,
            corrections=_empty_corrections(),
            graded_outcomes=graded_outcome_rows(today.bet_list),
            verdict_context=dict(_UNDECLARED),
            week_results=aggregate_bet_weeks(today.bet_list),
        )

    replay = _replay_rows_only(Path(output_dir))
    entries = _verified_ledger_entries(Path(ledger_dir))
    union = _union(replay, entries_to_frame(entries))
    corrections = _in_force_corrections(entries)
    refit_panel, refit_aa_status, refit_panel_failure = _refit_panel(
        entries, Path(silver_dir), Path(ledger_dir)
    )
    return BetCacheSources(
        bet_list=union,
        tracker=to_tracker_frame(aggregate_all_blocks(union, corrections=corrections)),
        schedule=build_bet_week_schedule(silver_dir),
        corrections=corrections,
        graded_outcomes=graded_outcome_rows(union, corrections=corrections),
        verdict_context=_verdict_context(),
        week_results=aggregate_bet_weeks(union, corrections=corrections),
        refit_panel=refit_panel,
        refit_aa_status=refit_aa_status,
        decided_edge_thresholds=decided_edge_thresholds(entries, Path(ledger_dir)),
        refit_panel_failure=refit_panel_failure,
    )
