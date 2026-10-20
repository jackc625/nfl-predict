"""The A/A proof and the per-week comparability invariant (Phase 37; SPEC R7, R8, D-02, D-03).

WHAT A CLEAN A/A WEEK IS
------------------------
Before the first re-fit the frozen (``shadow``) arm runs the same recipe as the ``live`` arm, so a
finished NFL week's shadow rows must equal its live rows exactly. A week is clean when every live
row has one shadow twin -- the same ``(game_id, season, week, target)`` -- identical in every
immutable column, in the envelope ``kind`` and ``canon_v``, in the grading half and in the
closing half. The ONLY fields ignored are the pre-registered ``AA_EXCLUDED_FIELDS`` (imported
from the frozen record, never re-declared): ``arm`` (the one column that names the arm),
the envelope ``seq`` and ``chain_hash`` (a row's position in the file, which two rows can never
share), and the five real-fill columns (a fill is recorded on the live row only). Every other
field is compared with the store's exact-text rule, so ``-0.0`` and ``0.0`` differ as they do in
the chain. The grading and closing halves are compared as each arm's CURRENT in-force record:
two still-pending twins agree, and one settle commit grades both arms identically afterwards.

ORDER OF THE CHECKS
-------------------
1. D-03: the pre-registration commit's committer date must precede the week's first decision run
   (its earliest game lock minus ``DECISION_RUN_LEAD``), else ``preregistration_after_first_run``.
2. Completeness (ROLLING-REFIT-PREREGISTRATION.md section 5): a week with any silver game kicking
   off at or after its evaluation instant (17:00 ET on its evaluation Tuesday) is
   ``week_not_complete`` -- told apart from a mismatch, and checked before any twin is compared,
   so no row can join a week after its result is binding.
3. A week with zero live rows does not qualify (``no_live_rows``). Games with no row in either arm
   (runs missed with the laptop off) do not void a week.
4. In file order, each live row needs its twin (``missing_twin``) and must match it
   (``field_mismatch``, naming the first differing field in column order); then a shadow row with
   no live twin is ``orphan_shadow``. A twin problem names the field ``twin``.

An unclean week never refuses a re-fit (SPEC amendment 2): it is recorded with its first mismatch
in ``ledger/refit/aa_weeks.jsonl`` and the schedule waits for the next clean week (the trigger in
``refit.cadence`` reads the record).

The functions here are pure over ``forward_ledger.store.read_entries`` output; only
:func:`record_week` writes, through ``refit.records.append_aa_record``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.refit_preregistration_2026 import AA_EXCLUDED_FIELDS
from forward_ledger.canonical import ENTRY_KIND_ROW, IMMUTABLE_COLUMNS_V1
from forward_ledger.store import LedgerEntry
from forward_ledger.verdict import DECISION_RUN_LEAD
from refit.cadence import aa_evaluation_instant
from refit.records import append_aa_record
from utils.game_lock import game_lock

__all__ = [
    "AA_EXCLUDED_FIELDS",
    "AAWeekResult",
    "evaluate_week",
    "record_week",
    "week_first_decision_run",
]

_LIVE = "live"
_SHADOW = "shadow"
_TWIN_FIELD = "twin"

# The immutable columns an A/A comparison reads: every v1 column outside the exclusion list.
_COMPARED_IMMUTABLE: tuple[str, ...] = tuple(
    name for name in IMMUTABLE_COLUMNS_V1 if name not in AA_EXCLUDED_FIELDS
)

# A key absent from one half compares unequal to any stored value, including JSON null.
_ABSENT = "<absent>"


@dataclass(frozen=True)
class AAWeekResult:
    """One week's A/A verdict.

    Attributes:
        season: The season evaluated.
        week: The NFL week evaluated.
        live_rows: The week's live rows.
        twins: The live rows that have a shadow twin.
        clean: True only when the week qualifies as a clean A/A week.
        reason: ``clean``, ``no_live_rows``, ``preregistration_after_first_run``,
            ``week_not_complete``, ``missing_twin``, ``field_mismatch`` or ``orphan_shadow``.
        first_mismatch: ``{game_id, target, field}`` of the first twin problem, else None.
        week_first_decision_run_utc: The week's first decision run (D-03).
        preregistration_committed_at: The record commit's committer date the check used.
    """

    season: int
    week: int
    live_rows: int
    twins: int
    clean: bool
    reason: str
    first_mismatch: dict[str, str] | None
    week_first_decision_run_utc: datetime
    preregistration_committed_at: datetime


def _exact(value: Any) -> str:
    """A value's exact JSON text: ``-0.0`` and ``0.0`` differ, as they do in the chain."""
    return json.dumps(value, allow_nan=False)


def _twin_key(entry: LedgerEntry) -> tuple[Any, ...]:
    """``(game_id, season, week, target)``: the ledger row key without its ``arm``."""
    return tuple(
        entry.immutable[name] for name in ("game_id", "season", "week", "target")
    )


def _week_rows(
    entries: Sequence[LedgerEntry], season: int, week: int, arm: str
) -> list[LedgerEntry]:
    """The row entries of (*season*, *week*) in *arm*, in file order."""
    return [
        entry
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW
        and entry.immutable["season"] == season
        and entry.immutable["week"] == week
        and entry.immutable["arm"] == arm
    ]


def _week_games(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """The silver games of (*season*, *week*); refuses a week the schedule does not hold."""
    week_games = games.loc[(games["season"] == season) & (games["week"] == week)]
    if week_games.empty:
        msg = f"the schedule holds no game of season {season} week {week}"
        raise ValueError(msg)
    return week_games


def week_first_decision_run(games: pd.DataFrame, season: int, week: int) -> datetime:
    """The week's earliest ``utils.game_lock`` lock minus ``DECISION_RUN_LEAD``, in UTC.

    Raises:
        ValueError: the schedule holds no game of the week.
    """
    week_games = _week_games(games, season, week)
    first_lock = min(
        game_lock(kickoff, game_id=str(game_id))
        for game_id, kickoff in zip(
            week_games["game_id"], week_games["kickoff_et"], strict=True
        )
    )
    return (first_lock - DECISION_RUN_LEAD).astimezone(UTC)


def _half_difference(
    prefix: str, live: Mapping[str, Any] | None, shadow: Mapping[str, Any] | None
) -> str | None:
    """The first key (as ``prefix.key``) on which two mutable halves differ, else None."""
    live_half = dict(live or {})
    shadow_half = dict(shadow or {})
    keys = [*live_half, *(key for key in shadow_half if key not in live_half)]
    for key in keys:
        if _exact(live_half.get(key, _ABSENT)) != _exact(shadow_half.get(key, _ABSENT)):
            return f"{prefix}.{key}"
    return None


def _first_difference(live: LedgerEntry, shadow: LedgerEntry) -> str | None:
    """The first field outside ``AA_EXCLUDED_FIELDS`` on which two twins differ, else None."""
    for name in _COMPARED_IMMUTABLE:
        if _exact(live.immutable[name]) != _exact(shadow.immutable[name]):
            return name
    if live.kind != shadow.kind:
        return "kind"
    if live.canon_v != shadow.canon_v:
        return "canon_v"
    return _half_difference(
        "grading", live.grading, shadow.grading
    ) or _half_difference("closing", live.closing, shadow.closing)


def _mismatch(entry: LedgerEntry, field: str) -> dict[str, str]:
    return {
        "game_id": str(entry.immutable["game_id"]),
        "target": str(entry.immutable["target"]),
        "field": field,
    }


def evaluate_week(
    entries: Sequence[LedgerEntry],
    games: pd.DataFrame,
    season: int,
    week: int,
    *,
    committed_at: datetime,
) -> AAWeekResult:
    """Whether (*season*, *week*) is a clean A/A week (see the module doc for the order).

    Args:
        entries: The ledger, in file order (``read_entries``).
        games: A silver-``games``-shaped schedule (``game_id``, ``season``, ``week``,
            tz-aware ``kickoff_et``).
        season: The season evaluated.
        week: The NFL week evaluated.
        committed_at: The pre-registration commit's committer date (``refit.witness``).

    Raises:
        ValueError: the schedule holds no game of the week.
    """
    first_run = week_first_decision_run(games, season, week)
    live = _week_rows(entries, season, week, _LIVE)
    shadow = _week_rows(entries, season, week, _SHADOW)
    shadow_by_key = {_twin_key(entry): entry for entry in shadow}
    twins = sum(1 for entry in live if _twin_key(entry) in shadow_by_key)

    def result(
        reason: str, first_mismatch: dict[str, str] | None = None
    ) -> AAWeekResult:
        return AAWeekResult(
            season=season,
            week=week,
            live_rows=len(live),
            twins=twins,
            clean=reason == "clean",
            reason=reason,
            first_mismatch=first_mismatch,
            week_first_decision_run_utc=first_run,
            preregistration_committed_at=committed_at,
        )

    if committed_at >= first_run:
        return result("preregistration_after_first_run")
    instant = aa_evaluation_instant(week)
    if any(
        pd.Timestamp(kickoff) >= instant
        for kickoff in _week_games(games, season, week)["kickoff_et"]
    ):
        return result("week_not_complete")
    if not live:
        return result("no_live_rows")
    for entry in live:
        twin = shadow_by_key.get(_twin_key(entry))
        if twin is None:
            return result("missing_twin", _mismatch(entry, _TWIN_FIELD))
        field = _first_difference(entry, twin)
        if field is not None:
            return result("field_mismatch", _mismatch(entry, field))
    live_keys = {_twin_key(entry) for entry in live}
    for entry in shadow:
        if _twin_key(entry) not in live_keys:
            return result("orphan_shadow", _mismatch(entry, _TWIN_FIELD))
    return result("clean")


def record_week(
    ledger_dir: Path | str, result: AAWeekResult, *, code_commit: str | None
) -> None:
    """Append *result* to ``<ledger>/refit/aa_weeks.jsonl`` (``refit.records``)."""
    append_aa_record(
        ledger_dir,
        {
            "season": result.season,
            "week": result.week,
            "evaluated_at_utc": datetime.now(tz=UTC).isoformat(),
            "clean": result.clean,
            "reason": result.reason,
            "live_rows": result.live_rows,
            "twins": result.twins,
            "first_mismatch": result.first_mismatch,
            "week_first_decision_run_utc": result.week_first_decision_run_utc.isoformat(),
            "preregistration_committed_at": result.preregistration_committed_at.isoformat(),
            "code_commit": code_commit,
        },
    )
