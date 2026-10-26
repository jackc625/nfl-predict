"""The per-game coverage rule of the 2026 season-close readout (Phase 38; SPEC R7, D-08, D-20, D-26).

THE STATES
----------
Every scheduled game of the verdict scope (season 2026, weeks ``start_week..end_week`` of the
committed declaration, read through ``forward_ledger.declarations.load_verdict_scope`` and never
restated) lands in exactly ONE state, by this precedence:

* ``not played``        -- inferred never played (below); it leaves the denominator;
* ``bet``               -- it has counted live-arm rows and at least one of them has status ``live``;
* ``no bet``            -- it has counted rows and none of them is a bet (every row ``suppressed``);
* ``failed or refused`` -- no counted row, but a run-log record or a skip record names it;
* ``not yet due``       -- PREVIEW ONLY: no row, no record, and its lock (``utils.game_lock``) is
  still in the future;
* ``missed``            -- no row and no record, and its lock has passed (the binding run passes
  no ``now``, so there every such game is missed).

A partly missed week must never read as covered: a week with any row looks attempted, so the rule
is applied per GAME and then rolled up per week, with a week that has no scheduled game reading
all zeros. Rows outrank a refusal record: a game with rows AND a refusal is decided.

NOT PLAYED IS AN INFERENCE, READ EACH RUN (D-26, D-20)
------------------------------------------------------
A game is not played when (a) it is in scope and has no final score while a game of a LATER scope
week already has one, or (b) its id is named by ledger, run-log or skip-record evidence for an
in-scope week but is absent from the schedule. Each is printed with the reason it was inferred. A
postponed game therefore reads as not played (and its bets as void, D-20) until its score lands,
then returns to the denominator with its bets; the binding run reads it once, after week 22.
One count: an (a) game leaves its week's scheduled count and adds 1 to that week's
``not_played``; a (b) id adds 1 to the ``not_played`` of the week its id names and leaves that
week's scheduled count alone (it was never scheduled). So the per-week ``not_played`` counters
sum to the number of named not-played games.

THE EVIDENCE FILES (D-08)
-------------------------
"Failed or refused" rests on ``logs/daily_lock_runs.jsonl`` and ``config/skip_records.jsonl``,
neither of them backed up. :func:`read_evidence` reads each file's bytes ONCE; the line count and
SHA-256 the readout prints and the records it parses come from those same bytes, so an append
between two reads can never make the digest and the classification disagree. A malformed line
refuses (:class:`EvidenceUnreadableError`, or ``pipeline.skip_log.SkipRecordCorrupt`` for the
skip records); it is never read as absent. A lost file can only move a game from "failed or
refused" to "missed", never make a missed game look covered.

Read-only: nothing here writes. Nothing on the daily run and no module under ``api/`` imports
this module.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import utils.game_lock as lock_rule
from api.cache import BET_STATUS_LIVE
from backtest.verdict_test_2026 import VERDICT_TARGETS
from forward_ledger.declarations import VerdictScope
from forward_ledger.settle import realized_values_from_scores
from utils.game_id_utils import parse_game_id

__all__ = [
    "EVIDENCE_LEDGER",
    "EVIDENCE_RUN_LOG",
    "EVIDENCE_SKIP_RECORDS",
    "STATE_BET",
    "STATE_FAILED_OR_REFUSED",
    "STATE_MISSED",
    "STATE_NOT_PLAYED",
    "STATE_NOT_YET_DUE",
    "STATE_NO_BET",
    "CoverageReport",
    "EvidenceFile",
    "EvidenceUnreadableError",
    "GameCoverage",
    "NotPlayedGame",
    "WeekCoverage",
    "classify_coverage",
    "parse_run_log",
    "read_evidence",
    "refused_game_ids",
]

STATE_BET: str = "bet"
STATE_NO_BET: str = "no bet"
STATE_FAILED_OR_REFUSED: str = "failed or refused"
STATE_MISSED: str = "missed"
STATE_NOT_YET_DUE: str = "not yet due"
STATE_NOT_PLAYED: str = "not played"

# The names an evidence source goes by in a not-played reason.
EVIDENCE_LEDGER: str = "ledger"
EVIDENCE_RUN_LOG: str = "run log"
EVIDENCE_SKIP_RECORDS: str = "skip records"


class EvidenceUnreadableError(Exception):
    """An evidence file or id cannot be read. NOT a claim that it is absent.

    Inherits bare ``Exception`` for the ``forward_ledger.store.LedgerFormatError`` reason: broad
    ``ValueError`` handlers elsewhere degrade to empty results, and an unreadable refusal record
    read as "no refusal" would turn a failed game into a quiet "missed".
    """


@dataclass(frozen=True)
class EvidenceFile:
    """One evidence file as read: its line count and SHA-256, both None when it is missing."""

    path: str
    lines: int | None
    sha256: str | None

    @property
    def missing(self) -> bool:
        """True when the file did not exist."""
        return self.sha256 is None


@dataclass(frozen=True)
class GameCoverage:
    """One scheduled game of the scope and the one state it landed in."""

    game_id: str
    week: int
    state: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class NotPlayedGame:
    """A game inferred never played, with the reason it was inferred (D-26)."""

    game_id: str
    week: int
    reason: str


@dataclass(frozen=True)
class WeekCoverage:
    """One scope week's per-state game counts.

    ``scheduled`` excludes the week's not-played games, so the five other states sum to it.
    """

    week: int
    scheduled: int
    bet: int
    no_bet: int
    failed_or_refused: int
    missed: int
    not_yet_due: int
    not_played: int

    @property
    def decided(self) -> int:
        """Games the live arm decided: a bet or a considered no-bet."""
        return self.bet + self.no_bet


@dataclass(frozen=True)
class CoverageReport:
    """Every scope game's state, the per-week roll-up, and the named exceptions.

    Attributes:
        games: Every scheduled scope game, in week then game id order.
        weeks: One entry per scope week, in week order.
        partial: Decided games whose rows cover fewer targets than the verdict has.
        not_played: Every game inferred never played, sorted by game id.
    """

    games: tuple[GameCoverage, ...]
    weeks: tuple[WeekCoverage, ...]
    partial: tuple[GameCoverage, ...]
    not_played: tuple[NotPlayedGame, ...]

    @property
    def missed_total(self) -> int:
        """Games missed across the whole scope."""
        return sum(week.missed for week in self.weeks)


# ---------------------------------------------------------------------------
# The evidence files
# ---------------------------------------------------------------------------


def read_evidence(path: Path | str) -> tuple[EvidenceFile, str | None]:
    """Read one evidence file ONCE: its provenance and its decoded text, from the same bytes.

    Returns:
        ``(EvidenceFile, text)``; an absent file is ``(EvidenceFile(path, None, None), None)``.

    Raises:
        EvidenceUnreadableError: the file exists but cannot be read or is not UTF-8.
    """
    name = Path(path).as_posix()
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return EvidenceFile(name, None, None), None
    except OSError as error:
        msg = f"the evidence file {name} exists but cannot be read: {error}"
        raise EvidenceUnreadableError(msg) from error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"the evidence file {name} is not UTF-8: {error}"
        raise EvidenceUnreadableError(msg) from error
    evidence = EvidenceFile(
        name, len(data.splitlines()), hashlib.sha256(data).hexdigest()
    )
    return evidence, text


def parse_run_log(text: str | None, path: Path | str) -> list[dict[str, Any]]:
    """The daily-run records of ``logs/daily_lock_runs.jsonl``, in file order.

    The writer is ``scripts/daily_lock_pipeline.py`` (``DAILY_RUN_RECORDS``). ``None`` text (the
    file is missing) is ``[]``.

    Raises:
        EvidenceUnreadableError: a line does not decode to a mapping, or its ``game_ids`` is
            present but not a list of strings -- named by path and line number, never read as
            absent.
    """
    if text is None:
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        where = f"the run log {Path(path).as_posix()}, line {number}"
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"{where} does not decode ({error}); it is refused, never read as absent"
            raise EvidenceUnreadableError(msg) from error
        if not isinstance(decoded, dict):
            msg = f"{where} is a {type(decoded).__name__}, not a mapping"
            raise EvidenceUnreadableError(msg)
        game_ids = decoded.get("game_ids", [])
        if not isinstance(game_ids, list) or not all(
            isinstance(g, str) for g in game_ids
        ):
            msg = (
                f"{where} carries game_ids {game_ids!r}, not a list of game id strings"
            )
            raise EvidenceUnreadableError(msg)
        records.append(decoded)
    return records


def refused_game_ids(
    run_records: Sequence[Mapping[str, Any]],
    skip_records: Sequence[Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    """``{game_id -> the evidence that named it}`` over every run record and skip record.

    A run record names the games of its ``game_ids`` whatever its outcome (a ``no_games`` record
    names none); a skip record names its ``game_id``.
    """
    named: dict[str, set[str]] = {}
    for record in run_records:
        for game_id in record.get("game_ids", []):
            named.setdefault(str(game_id), set()).add(EVIDENCE_RUN_LOG)
    for record in skip_records:
        named.setdefault(str(record["game_id"]), set()).add(EVIDENCE_SKIP_RECORDS)
    return {game_id: frozenset(sources) for game_id, sources in named.items()}


# ---------------------------------------------------------------------------
# The coverage rule
# ---------------------------------------------------------------------------


def _scope_games(schedule: pd.DataFrame, scope: VerdictScope) -> pd.DataFrame:
    """The schedule's games of the scope: the declared season, weeks start..end inclusive."""
    in_scope = (
        (schedule["season"] == scope.season)
        & (schedule["week"] >= scope.start_week)
        & (schedule["week"] <= scope.end_week)
    )
    return schedule.loc[in_scope].sort_values(["week", "game_id"])


def _rows_by_game(counted: pd.DataFrame) -> dict[str, tuple[set[str], bool]]:
    """``{game_id -> (its counted rows' targets, whether any row is a bet)}``."""
    by_game: dict[str, tuple[set[str], bool]] = {}
    if counted.empty:  # an empty ledger's frame may carry no columns at all
        return by_game
    for game_id, target, status in zip(
        counted["game_id"], counted["target"], counted["status"], strict=True
    ):
        targets, has_bet = by_game.get(str(game_id), (set(), False))
        targets.add(str(target))
        by_game[str(game_id)] = (targets, has_bet or status == BET_STATUS_LIVE)
    return by_game


def _in_scope_week(game_id: str, scope: VerdictScope) -> int | None:
    """The scope week an evidence id names, or None when it names a game outside the scope.

    Raises:
        EvidenceUnreadableError: the id is not a standard game id.
    """
    try:
        parsed = parse_game_id(game_id)
    except ValueError as error:
        msg = (
            f"the evidence names {game_id!r}, which is not a standard game id: {error}"
        )
        raise EvidenceUnreadableError(msg) from error
    week = int(parsed["week"])
    if parsed["season"] == scope.season and scope.start_week <= week <= scope.end_week:
        return week
    return None


def _absent_from_schedule(
    schedule: pd.DataFrame,
    rows_by_game: Mapping[str, object],
    refused: Mapping[str, frozenset[str]],
    scope: VerdictScope,
) -> list[NotPlayedGame]:
    """Rule (b): in-scope evidence ids the schedule does not hold."""
    sources: dict[str, set[str]] = {}
    for game_id in rows_by_game:
        sources.setdefault(game_id, set()).add(EVIDENCE_LEDGER)
    for game_id, named_by in refused.items():
        sources.setdefault(game_id, set()).update(named_by)

    scheduled = set(schedule["game_id"].astype(str))
    absent: list[NotPlayedGame] = []
    for game_id, named_by in sources.items():
        week = _in_scope_week(game_id, scope)
        if week is None or game_id in scheduled:
            continue
        order = (EVIDENCE_LEDGER, EVIDENCE_RUN_LOG, EVIDENCE_SKIP_RECORDS)
        names = "+".join(name for name in order if name in named_by)
        reason = f"inferred: named by {names} but absent from the schedule"
        absent.append(NotPlayedGame(game_id, week, reason))
    return absent


def _live_state(
    game_id: str,
    kickoff: Any,
    rows_by_game: Mapping[str, tuple[set[str], bool]],
    refused: Mapping[str, frozenset[str]],
    now: datetime | None,
) -> str:
    """bet > no bet > failed or refused > (not yet due | missed), for a game that was played."""
    if game_id in rows_by_game:
        return STATE_BET if rows_by_game[game_id][1] else STATE_NO_BET
    if game_id in refused:
        return STATE_FAILED_OR_REFUSED
    if now is not None and lock_rule.game_lock(kickoff, game_id=game_id) > now:
        return STATE_NOT_YET_DUE
    return STATE_MISSED


def classify_coverage(
    schedule: pd.DataFrame,
    scores: pd.DataFrame,
    counted: pd.DataFrame,
    refused: Mapping[str, frozenset[str]],
    scope: VerdictScope,
    *,
    now: datetime | None,
) -> CoverageReport:
    """Every scope game's coverage state, rolled up per week.

    Args:
        schedule: The recorded schedule (``utils.current_slate.load_recorded_schedule``).
        scores: The silver ``games`` table (``forward_ledger.settle.load_silver_games``); a game is
            played when ``realized_values_from_scores`` gives it a value (THE final-score rule).
        counted: The counted live-arm rows (``forward_ledger.verdict.verdict_rows``).
        refused: :func:`refused_game_ids` over the run log and the skip records.
        scope: The committed verdict scope.
        now: The preview's instant (tz-aware), below which a lock is "not yet due"; None in the
            binding run, where nothing is ever not yet due.

    Raises:
        EvidenceUnreadableError: an evidence id is not a standard game id.
    """
    rows_by_game = _rows_by_game(counted)
    played = set(realized_values_from_scores(scores)["ats"])
    universe = _scope_games(schedule, scope)
    universe_ids = universe["game_id"].astype(str)
    played_weeks = [
        int(week)
        for game_id, week in zip(universe_ids, universe["week"], strict=True)
        if game_id in played
    ]
    last_scored_week = max(played_weeks, default=None)

    games: list[GameCoverage] = []
    not_played = _absent_from_schedule(schedule, rows_by_game, refused, scope)
    for game_id, week, kickoff in zip(
        universe_ids, universe["week"].astype(int), universe["kickoff_et"], strict=True
    ):
        targets = (
            tuple(sorted(rows_by_game[game_id][0])) if game_id in rows_by_game else ()
        )
        if (
            game_id not in played
            and last_scored_week is not None
            and week < last_scored_week
        ):
            reason = f"inferred: no final score while week {last_scored_week} is scored"
            not_played.append(NotPlayedGame(game_id, week, reason))
            games.append(GameCoverage(game_id, week, STATE_NOT_PLAYED, targets))
            continue
        state = _live_state(game_id, kickoff, rows_by_game, refused, now)
        games.append(GameCoverage(game_id, week, state, targets))

    weeks: list[WeekCoverage] = []
    for week in range(scope.start_week, scope.end_week + 1):
        states = [game.state for game in games if game.week == week]
        weeks.append(
            WeekCoverage(
                week=week,
                scheduled=len(states) - states.count(STATE_NOT_PLAYED),
                bet=states.count(STATE_BET),
                no_bet=states.count(STATE_NO_BET),
                failed_or_refused=states.count(STATE_FAILED_OR_REFUSED),
                missed=states.count(STATE_MISSED),
                not_yet_due=states.count(STATE_NOT_YET_DUE),
                not_played=sum(1 for game in not_played if game.week == week),
            )
        )
    partial = tuple(
        game
        for game in games
        if game.state in (STATE_BET, STATE_NO_BET)
        and len(game.targets) < len(VERDICT_TARGETS)
    )
    return CoverageReport(
        games=tuple(games),
        weeks=tuple(weeks),
        partial=partial,
        not_played=tuple(sorted(not_played, key=lambda game: game.game_id)),
    )
