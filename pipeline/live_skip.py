"""The live-skip POLICY: a caught game is dropped and recorded; the day's clean games go ahead.

Phase 33.2, Plan 33.2-03 (SPEC R3; D33.2-05 live half; RESEARCH pitfall P10).

WHAT D33.2-05 ASKS FOR, AND WHERE EACH HALF LIVES
-------------------------------------------------
When the information-time check catches a problem, the two kinds of run must react
DIFFERENTLY:

* a HISTORY (training/gold) build stops and saves nothing -- the refusal simply propagates, and
  this module is never consulted (the orchestrator's history branch comes first);
* a LIVE run leaves out ONLY the caught games: no prediction, no bet, the reason recorded
  durably, and every clean game still predicted.

This module is the live half's POLICY, in production code. The orchestrator's step seam
(``pipeline/orchestrator.py``, ``FridayPipeline._execute_step``) catches a refusal from
:data:`LIVE_SKIP_EXCEPTIONS`, hands it to :func:`apply_skip_policy`, and re-runs the step on the
remainder. Before this, ``_execute_step`` turned EVERY exception into a failed step, and a
critical step's failure is fatal to the run -- so one post-lock value denied the whole night's
clean games their predictions, which is exactly what the live half exists to prevent (P10).

THE EXCLUSION REGISTER LIVES HERE, AND WHY IT IS A REGISTER
-----------------------------------------------------------
``StepDefinition.callable`` is ``Callable[[], None]`` (``pipeline/steps.py``). Threading the
excluded games through as a step ARGUMENT would rewrite all 22 registry entries for the benefit
of the few steps that read it. So the games travel through ONE process-level register owned by
this module -- :func:`exclude_games`, :func:`excluded_games`, :func:`reset_excluded_games` --
which the steps that need it READ (``pipeline/steps.py`` never writes it).

``reset_excluded_games`` is called at the START of every run's step loop, so a register left
populated by an earlier in-process run cannot suppress tonight's games (T-33.2-03-10). Nothing
at import time seeds it.

WHAT IS AND IS NOT CAUGHT
-------------------------
:data:`LIVE_SKIP_EXCEPTIONS` is a CLOSED two-member tuple: the per-game refusals. Anything else
is an ordinary step failure and keeps today's behaviour exactly -- a ``FAILED`` step, fatal when
the step is critical. Nothing is ever added to the tuple to make a broad failure survivable
(T-33.2-03-08). A caught refusal that names NO game, or whose kind the closed reason vocabulary
cannot express, is refused by :class:`UnskippableRefusalError` rather than defaulted: a silent
fall-back reason would mislabel the record Phase 35 reads, and an unnamed game cannot be
excluded at all.

A BUILD THAT IGNORES THE EXCLUSION IS LOUD
------------------------------------------
If a re-run raises again for a game already in the register, the step is not honouring the
exclusion, and retrying would loop. :func:`apply_skip_policy` raises
:class:`SkipNotConvergingError` on the FIRST such repeat, and the orchestrator bounds the rounds
by :func:`max_skip_rounds` in any case (T-33.2-03-09).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.weekly_bet_list import LockPassedError
from features.provenance import (
    InformationTimeViolation,
    ProvenanceCoverageError,
    UndatedSourceError,
)
from pipeline import skip_log

__all__ = [
    "DECISION_INSTANT_SOURCE",
    "LIVE_SKIP_EXCEPTIONS",
    "GamesLockPassedError",
    "SkipNotConvergingError",
    "UnskippableRefusalError",
    "apply_skip_policy",
    "exclude_games",
    "excluded_games",
    "excluded_ids_from",
    "max_skip_rounds",
    "refuse_passed_locks",
    "reset_excluded_games",
    "skip_reason_for",
    "skip_records_from",
]

# THE CLOSED SET OF PER-GAME REFUSALS the live half may catch.
#
# Exactly two classes, and a unit test compares them by IDENTITY so a same-named class elsewhere
# cannot satisfy it. ``InformationTimeViolation`` covers its two subclasses (``UndatedSourceError``,
# ``ProvenanceCoverageError``) because each names the games it refuses. Nothing is added here to
# make a broad failure survivable: anything outside this tuple is an ordinary step failure and
# keeps today's fatality.
LIVE_SKIP_EXCEPTIONS: tuple[type[Exception], ...] = (
    InformationTimeViolation,
    LockPassedError,
)

# The ``source`` a passed-lock refusal is recorded under. The information that post-dates the lock
# is the DECISION itself -- the instant the run would have decided the game -- not any one feature
# source, so it is named as such rather than borrowed from a source it did not come from.
DECISION_INSTANT_SOURCE = "decision_instant"

# The gate's ``violation_type`` values (features/provenance.py) that name a per-game fact, and the
# reason each maps onto. A naive information time is recorded as ``undated``: the time could not be
# established honestly, which is what ``undated`` means. Any other value -- ``provenance_column``,
# raised when a sidecar column reaches gold -- is a whole-build defect, not a game to drop.
_VIOLATION_TYPE_REASONS: dict[str, str] = {
    "information_time": "post_lock",
    "naive_information_time": "undated",
    "lock_passed": "post_lock",
}

# The process exclusion register. Module state by design (see the module docstring); read through
# :func:`excluded_games`, which returns a SNAPSHOT so no caller holds a live handle to it.
_EXCLUDED: set[str] = set()


class SkipNotConvergingError(Exception):
    """A re-run refused a game the run had ALREADY excluded, or the rounds hit their cap.

    Inherits ``Exception`` for the reason ``pipeline.skip_log.SkipRecordCorrupt`` does: this
    repository's catch-tuples swallow ``RuntimeError`` / ``ValueError`` / ``ImportError`` and
    degrade quietly, and "the exclusion is being ignored" degraded into "carry on" would publish a
    prediction for a post-lock game.
    """


class UnskippableRefusalError(Exception):
    """A caught refusal the policy cannot turn into a named, reasoned skip.

    Raised when the refusal names no game (there is nothing to exclude, and parsing prose to find
    one would be a guess), or when its kind is outside the closed reason vocabulary. Never a
    default: a silent fall-back to ``post_lock`` would mislabel the record Phase 35 reads.
    Inherits ``Exception`` for the same catch-tuple reason as :class:`SkipNotConvergingError`.
    """


class GamesLockPassedError(LockPassedError):
    """A :class:`backtest.weekly_bet_list.LockPassedError` that NAMES its games in ``details``.

    ``LockPassedError`` itself carries only a message, so a live run could not tell WHICH games to
    drop without parsing prose. This subclass carries the gate's payload shape --
    ``{"source", "game_ids", "information_times", "locks", "violation_type"}`` -- and, being a
    ``LockPassedError``, is caught by :data:`LIVE_SKIP_EXCEPTIONS` without widening the set.
    """

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details


def _details(exc: BaseException) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    return details if isinstance(details, dict) else {}


def skip_reason_for(exc: BaseException) -> str:
    """Map a caught refusal onto the closed :data:`pipeline.skip_log.SKIP_REASONS` vocabulary.

    Raises:
        UnskippableRefusalError: for an exception outside :data:`LIVE_SKIP_EXCEPTIONS`, or an
            ``InformationTimeViolation`` whose ``violation_type`` names no per-game fact.
    """
    # The subclasses first: each is also an InformationTimeViolation.
    if isinstance(exc, LockPassedError):
        return "post_lock"
    if isinstance(exc, UndatedSourceError):
        return "undated"
    if isinstance(exc, ProvenanceCoverageError):
        return "no_provenance"
    if isinstance(exc, InformationTimeViolation):
        violation_type = _details(exc).get("violation_type")
        if violation_type in _VIOLATION_TYPE_REASONS:
            return _VIOLATION_TYPE_REASONS[str(violation_type)]
    msg = (
        f"cannot map {type(exc).__name__} ({_details(exc).get('violation_type')!r}) onto a skip "
        f"reason in {skip_log.SKIP_REASONS}. It is not a per-game refusal the live half may "
        f"drop a game for, so the step fails as it would have before: {exc}"
    )
    raise UnskippableRefusalError(msg)


def excluded_ids_from(exc: BaseException) -> frozenset[str]:
    """The games a caught refusal names, read off its ``details["game_ids"]``.

    Raises:
        UnskippableRefusalError: when the refusal names no game.
    """
    game_ids = _details(exc).get("game_ids")
    if not game_ids:
        msg = (
            f"{type(exc).__name__} names no game in its details, so the live half has nothing "
            "it could drop -- and reading a game id out of the message would be a guess. The "
            f"step fails as it would have before: {exc}"
        )
        raise UnskippableRefusalError(msg)
    return frozenset(str(game_id) for game_id in game_ids)


def _aligned(values: object, index: int, count: int) -> object:
    """The *index*-th entry of a per-game list the payload carries, or None if it carries none."""
    if isinstance(values, list | tuple) and len(values) == count:
        return values[index]
    return None


def skip_records_from(
    exc: BaseException, *, run_id: str, run_date_et: str
) -> list[dict[str, Any]]:
    """One skip record per (game, source, reason) the refusal names. BUILDS; never writes.

    Every record carries all eight :data:`pipeline.skip_log.REQUIRED_ENTRY_KEYS`. The information
    time and lock are taken from the refusal's own payload when it carries them, aligned by
    position with ``game_ids``, and are NULL when it does not: an ``undated`` refusal has no time
    to record, and a coverage refusal does not state its lock.

    Raises:
        UnskippableRefusalError: via :func:`skip_reason_for` / :func:`excluded_ids_from`, or when
            the refusal names no source.
    """
    reason = skip_reason_for(exc)
    details = _details(exc)
    ordered_ids = [str(game_id) for game_id in details.get("game_ids") or []]
    if not ordered_ids:
        excluded_ids_from(exc)  # raises, naming the missing games
    source = details.get("source")
    if not source:
        msg = (
            f"{type(exc).__name__} names no source, so its skip record could not say which "
            f"input post-dated the lock: {exc}"
        )
        raise UnskippableRefusalError(msg)

    recorded_at = datetime.now(UTC).isoformat()
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for position, game_id in enumerate(ordered_ids):
        key = (game_id, str(source), reason)
        records.setdefault(
            key,
            {
                "run_id": run_id,
                "run_date_et": run_date_et,
                "game_id": game_id,
                "source": str(source),
                "information_time": _aligned(
                    details.get("information_times"), position, len(ordered_ids)
                ),
                "lock": _aligned(details.get("locks"), position, len(ordered_ids)),
                "reason": reason,
                "recorded_at": recorded_at,
            },
        )
    return list(records.values())


def _iso(value: object) -> str:
    """An instant as ISO-8601 text; ``pd.Timestamp`` is a ``datetime``, so both take this path."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


def exclude_games(game_ids: Iterable[str]) -> None:
    """Add *game_ids* to the process exclusion register."""
    _EXCLUDED.update(str(game_id) for game_id in game_ids)


def excluded_games() -> frozenset[str]:
    """A SNAPSHOT of the games excluded so far in this process's current run."""
    return frozenset(_EXCLUDED)


def reset_excluded_games() -> None:
    """Empty the register. Called at the start of every run's step loop (T-33.2-03-10)."""
    _EXCLUDED.clear()


def apply_skip_policy(
    exc: BaseException,
    *,
    run_id: str,
    run_date_et: str,
    path: Path | str | None = None,
) -> frozenset[str]:
    """The one composition: map the reason, record each game durably, exclude it, return them.

    The order is chosen so that a refusal writes NOTHING: the repeat check and the record build
    both run before the first append, so an already-excluded game or an unmappable refusal leaves
    the skip record byte-identical. Each append is idempotent on its natural key
    (``pipeline.skip_log``), so re-running the same live day adds no line.

    Args:
        exc: The caught refusal.
        run_id: The run's identity (its ISO start instant).
        run_date_et: The run's Eastern calendar date, ``YYYY-MM-DD``.
        path: A skip-record override, for tests. ``None`` uses the committed store.

    Returns:
        The games excluded by this refusal.

    Raises:
        UnskippableRefusalError: for an exception outside :data:`LIVE_SKIP_EXCEPTIONS`, or one
            the policy cannot map or that names no game.
        SkipNotConvergingError: when the refusal names a game already excluded -- the step is
            not honouring the exclusion, and retrying would loop.
    """
    if not isinstance(exc, LIVE_SKIP_EXCEPTIONS):
        msg = (
            f"{type(exc).__name__} is not a per-game refusal ({LIVE_SKIP_EXCEPTIONS}); the live "
            f"half never drops a game for it: {exc}"
        )
        raise UnskippableRefusalError(msg)

    game_ids = excluded_ids_from(exc)
    repeated = sorted(game_ids & excluded_games())
    if repeated:
        msg = (
            f"the live skip did not converge: a re-run refused {repeated}, which this run had "
            "ALREADY excluded, so the step is not honouring the exclusion register. Retrying "
            f"would loop; the step fails instead. Refusal: {exc}"
        )
        raise SkipNotConvergingError(msg) from exc

    records = skip_records_from(exc, run_id=run_id, run_date_et=run_date_et)
    for record in records:
        skip_log.append_skip_record(record, path=path)
    exclude_games(game_ids)
    return game_ids


def max_skip_rounds() -> int:
    """The most skip rounds one step may take: the feature-source count plus one. DERIVED.

    One ``InformationTimeViolation`` already names EVERY offending game in its source, because
    the gate's mask covers the whole source frame. So a step honouring the register can raise at
    most once per registered source; the ``+ 1`` is the round for a refusal no feature source
    raises -- the decision instant's passed-lock refusal. Read from the registry
    ``scripts.build_features`` walks, never a literal, because plans 33.2-12 .. 33.2-17 grow it.
    The import is deferred: the build module is heavy and only a run that actually skips needs
    the number.
    """
    from scripts.build_features import FEATURE_SOURCE_KEYS

    return len(FEATURE_SOURCE_KEYS) + 1


def refuse_passed_locks(
    schedule: pd.DataFrame,
    *,
    decided_at: datetime,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> None:
    """Refuse every game whose lock is BEFORE the decision instant, naming each one.

    The standing prohibition (SPEC R3, R15): no prediction or bet row for a game whose lock had
    already passed -- including after a MISSED daily run, when a catch-up run would otherwise
    back-fill yesterday's slate. The comparison is the ONE rule: each game's lock comes from
    ``utils.game_lock.lock_frame`` and the test is ``utils.game_lock.is_admissible(decided_at,
    lock)``, the same ``<=`` the bet list's selection fence uses, so AT-LOCK IS ADMISSIBLE and one
    second later is not. Both are reached as module attributes at call time, so the one-rule
    identity scan sees this reader.

    EXCLUSION PRECEDES THE REFUSAL, for the reason the selection fence states: a game already
    dropped for a post-lock input is usually exactly a game whose lock has passed, and refusing it
    again would turn one clean skip into a failure.

    Args:
        schedule: The games in scope, carrying ``game_id`` and ``kickoff_et``.
        decided_at: The instant the run would decide these games. Must be timezone-aware; a
            naive value raises through the lock rule's strict parser rather than being relabelled.
        excluded_game_ids: Games already dropped this run.

    Raises:
        GamesLockPassedError: naming every in-scope game whose lock is before *decided_at*.
    """
    import utils.game_lock as lock_rule

    if schedule.empty:
        return
    in_scope = schedule.loc[
        ~schedule["game_id"].astype(str).isin(sorted(excluded_game_ids))
    ]
    if in_scope.empty:
        return

    locks = lock_rule.lock_frame(in_scope)
    passed = [
        (str(game_id), lock)
        for game_id, lock in locks.items()
        if not lock_rule.is_admissible(decided_at, lock)
    ]
    if not passed:
        return

    decided_text = _iso(decided_at)
    game_ids = [game_id for game_id, _lock in passed]
    msg = (
        f"refusing {len(passed)} game(s) whose own lock is BEFORE the decision instant "
        f"{decided_text}: {game_ids}. At-lock is admissible and one second later is not; a "
        "prediction or bet decided now for these games would be post-hoc, so the honest outcome "
        "is no row and a recorded skip."
    )
    raise GamesLockPassedError(
        msg,
        {
            "source": DECISION_INSTANT_SOURCE,
            "game_ids": game_ids,
            "information_times": [decided_text] * len(passed),
            "locks": [_iso(lock) for _game_id, lock in passed],
            "violation_type": "lock_passed",
        },
    )
