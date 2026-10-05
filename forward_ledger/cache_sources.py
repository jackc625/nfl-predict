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
the tiles are drawn from one frame (review finding 4).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_CORRECTIONS_COLUMNS,
    PROVENANCE_FORWARD,
)
from backtest.bet_tracker import (
    aggregate_all_blocks,
    graded_outcome_rows,
    to_tracker_frame,
)
from backtest.weekly_bet_list import (
    DEFAULT_BET_LIST_DIR,
    build_bet_week_schedule,
    read_bet_list_artifact,
    read_bet_list_cache_sources,
)
from forward_ledger.corrections import in_force_outcomes
from forward_ledger.cutover import forward_rows_go_to_ledger
from forward_ledger.declarations import VerdictScopeUndeclaredError, load_verdict_scope
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    LedgerEntry,
    entries_to_frame,
    ledger_path,
    read_entries,
    verify_chain,
)

__all__ = [
    "BetCacheSources",
    "ForwardRowsOutsideLedgerError",
    "LedgerAbsentAfterCutoverError",
    "read_bet_cache_sources",
]

# The context stamped when no verdict scope is declared (or the switch is off).
_UNDECLARED: dict[str, Any] = {"declared": False, "season": None, "start_week": None}


class ForwardRowsOutsideLedgerError(Exception):
    """Forward rows were found in ``outputs/bet_list`` after the cutover: a second forward store.

    Bare ``Exception`` (the ``forward_ledger.store`` rule): broad ``ValueError`` handlers elsewhere
    degrade to empty results, and a refused source must never read as "nothing recommended".
    """


class LedgerAbsentAfterCutoverError(Exception):
    """The switch is on but the ledger file does not exist: a cutover without its migration."""


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
    """

    bet_list: pd.DataFrame
    tracker: pd.DataFrame
    schedule: pd.DataFrame
    corrections: pd.DataFrame
    graded_outcomes: pd.DataFrame
    verdict_context: dict[str, Any]


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
        Everything ``read_bet_list_cache_sources`` raises for a corrupt artifact.
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
        )

    replay = _replay_rows_only(Path(output_dir))
    entries = _verified_ledger_entries(Path(ledger_dir))
    union = _union(replay, entries_to_frame(entries))
    corrections = _in_force_corrections(entries)
    return BetCacheSources(
        bet_list=union,
        tracker=to_tracker_frame(aggregate_all_blocks(union, corrections=corrections)),
        schedule=build_bet_week_schedule(silver_dir),
        corrections=corrections,
        graded_outcomes=graded_outcome_rows(union, corrections=corrections),
        verdict_context=_verdict_context(),
    )
