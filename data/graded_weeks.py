"""The graded-weeks seam (D32-11, Plan 32-03).

WHAT THIS ANSWERS
-----------------
One question, and only one: *which weeks of this season have already been graded, so a
revision touching one of them can escalate?* PIN-03's rule -- a 2026 revision touching an
ALREADY-GRADED week is louder than one touching a week nobody has settled -- is unevaluable
without that set.

HOW IT ANSWERS IT
-----------------
From wherever forward rows actually live, decided by the committed cutover switch
(``forward_ledger.cutover.forward_rows_go_to_ledger``):

* SWITCH OFF (until Phase 34's go-live, Plan 34-19): Phase 31's bet-list grading state, the
  durable artifact ``outputs/bet_list/bet_list.parquet``, read through
  ``backtest.weekly_bet_list.read_bet_list_artifact``.
* SWITCH ON: the forward ledger ``ledger/forward_2026.jsonl``, read through
  ``forward_ledger.store.read_entries`` and accepted only when ``verify_chain`` passes. Both
  arms count -- a correction can be owed for a live or a shadow row alike.

Either way a row counts as graded when its ``grading_status`` is outside ``pending`` -- see
:data:`TERMINAL_GRADING_STATUSES` -- and :func:`graded_weeks_source` names the store that was
actually read, so a record never claims a source it did not use.

THE PHASE-34 REPOINTING NOTE (why this seam exists at all)
-----------------------------------------------------------
Stated as a prose reason in the style of ``scripts/pin_upstream_snapshot.py``'s
``NOT_PINNED`` entries, because a reader six months from now needs the reason and not just
the fact:

    Phase 34's LDGR-01 RELOCATES the durable forward-bet store out of ``outputs/``. A
    detector that read ``outputs/bet_list/bet_list.parquet`` directly would therefore break
    on a change that is ALREADY ON THE CALENDAR -- and it would break inside a detector,
    where a silent failure looks exactly like "nothing to report". So every caller asks
    :func:`graded_weeks`, and Phase 34 repoints THIS ONE FUNCTION (with its private
    :func:`_artifact_location`, the only other place in the repository that knows where the
    store lives) at the forward ledger. Nothing else changes, and
    :func:`graded_weeks_record` follows for free because it is a thin wrapper with no read
    of its own.

D32-11 DISCHARGED BY PHASE 34 (Plan 34-15): the repoint is made, behind the cutover switch, in
:func:`graded_weeks` and :func:`_artifact_location`. :data:`GRADED_WEEKS_SOURCE` now names the
ledger (the repointed seam's source, updated in the same edit as the module docstring requires);
:data:`GRADED_WEEKS_SOURCE_BET_LIST` keeps the old string for the switch-off reads.

ABSENT AND UNREADABLE ARE DIFFERENT FACTS
------------------------------------------
* NO STORE -- a legitimate state (a checkout that has never generated a bet list). It
  resolves to an EXPLICITLY RECORDED empty set: :func:`graded_weeks_record` returns
  ``weeks: []`` with a ``reason`` naming the absent path, so a verdict can never say "no
  weeks were graded" without also saying where it looked.
* UNREADABLE -- schema mismatch, truncation, corruption, or (switch on) a ledger whose
  chain does not verify. It RAISES :class:`GradedWeeksUnavailable`. Returning an empty set there would make "I could not
  tell" indistinguishable from "nothing is graded", which silently downgrades a CRITICAL to
  informational -- the exact failure D32-11 was written to prevent.

IMPORT POSITION
---------------
``backtest.weekly_bet_list`` and ``api.cache`` are imported INSIDE function bodies, the
idiom ``data/upstream_pin.py`` uses for ``nflreadpy``. ``data/`` is the lowest layer in this
repository and ``api/`` is the highest; a module-scope import here would invert that and
would be a standing invitation to an import cycle the day ``api/`` grows a ``data/`` import.

ASCII only, no emoji (CLAUDE.md hard constraint).

Run the tests:  .venv/Scripts/python.exe -m pytest tests/unit/test_graded_weeks.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "GRADED_WEEKS_SOURCE",
    "GRADED_WEEKS_SOURCE_BET_LIST",
    "GRADED_WEEKS_SOURCE_LEDGER",
    "TERMINAL_GRADING_STATUSES",
    "GradedWeeksUnavailable",
    "graded_weeks",
    "graded_weeks_record",
    "graded_weeks_source",
]


# The human-readable provenance strings, one per store. EVERY :func:`graded_weeks_record`
# result -- including the empty ones -- carries the one :func:`graded_weeks_source` names, so a
# record whose weeks came from the forward ledger never claims they came from the bet list.
GRADED_WEEKS_SOURCE_BET_LIST: str = (
    "backtest.weekly_bet_list.read_bet_list_artifact over "
    "outputs/bet_list/bet_list.parquet"
)
GRADED_WEEKS_SOURCE_LEDGER: str = (
    "forward_ledger.store.read_entries over ledger/forward_2026.jsonl (chain verified)"
)

# The repointed seam's source (Phase 34 updated it in the same edit that repointed the functions
# below). Read :func:`graded_weeks_source` for the store a given call actually reads.
GRADED_WEEKS_SOURCE: str = GRADED_WEEKS_SOURCE_LEDGER


def _reads_the_ledger() -> bool:
    """Whether forward rows -- and so their grading -- live in the ledger (the cutover switch)."""
    from forward_ledger.cutover import forward_rows_go_to_ledger

    return forward_rows_go_to_ledger()


def graded_weeks_source() -> str:
    """The provenance string of the store :func:`graded_weeks` reads right now."""
    if _reads_the_ledger():
        return GRADED_WEEKS_SOURCE_LEDGER
    return GRADED_WEEKS_SOURCE_BET_LIST


def _derive_terminal_grading_statuses() -> tuple[str, ...]:
    """The grading statuses that mean SETTLED, derived from the one source vocabulary.

    ``api.cache.GRADING_STATUSES`` is the CLOSED four-state vocabulary and the only place it
    is spelled. Re-typing the words here is the second-list failure this repository has
    already paid for once -- ``backtest/bet_tracker.py``'s docstring says so about the label
    vocabulary -- so the terminal set is the imported tuple MINUS ``GRADING_STATUS_PENDING``
    and is never a literal.

    A push is deliberately IN. It is a settled result that turned over money at zero profit,
    and ``backtest/bet_tracker.py``'s "three distinctions" section is the authority: a push
    is counted as graded, while an ungraded row is excluded from every figure.
    """
    from api.cache import GRADING_STATUS_PENDING, GRADING_STATUSES

    return tuple(
        status for status in GRADING_STATUSES if status != GRADING_STATUS_PENDING
    )


# Bound at IMPORT so that the vocabulary is a constant a caller can read, compare and log
# without first making a call -- a value that exists only sometimes is not a constant. The
# import that derives it is function-local (see the module docstring on import position);
# that keeps ``api`` out of this module's static import graph, but the derivation call below
# does mean ``api.cache`` is imported when this module is. The heavier PARQUET read path
# (``backtest.weekly_bet_list`` plus pandas) stays genuinely deferred to call time.
TERMINAL_GRADING_STATUSES: tuple[str, ...] = _derive_terminal_grading_statuses()


class GradedWeeksUnavailable(Exception):
    """The graded-week state could not be determined. NOT a claim that nothing is graded.

    Inherits ``Exception`` and deliberately NOT ``RuntimeError`` / ``ValueError`` /
    ``ImportError``, for the same reason ``data.upstream_pin.UpstreamPinError`` does.
    ``features/qb_tracking.py`` catches ``(ImportError, ValueError, RuntimeError)`` around
    its loaders and degrades to an empty frame; ``features/team_form.py`` and
    ``scripts/ingest_games.py`` catch similar tuples. A refusal that landed in one of those
    handlers would be converted into "no weeks are graded", which silently downgrades a
    CRITICAL verdict to an informational one. Escaping every broad handler is the point.
    """


def _artifact_location(output_dir: Path | str | None) -> tuple[Path, Path]:
    """Return ``(directory, store path)`` for the store the seam reads now.

    The ONLY place in this module that knows where the store lives: the ledger directory and
    ``forward_2026.jsonl`` when the cutover switch is on, else the bet-list pair. Kept private
    because a caller that wanted the path would be reading the store directly, which is the
    thing this seam exists to stop.
    """
    if _reads_the_ledger():
        from forward_ledger.store import LEDGER_DIR, ledger_path

        ledger_dir = Path(output_dir) if output_dir is not None else LEDGER_DIR
        return ledger_dir, ledger_path(ledger_dir)

    from backtest.weekly_bet_list import BET_LIST_ARTIFACT_NAME, DEFAULT_BET_LIST_DIR

    directory = Path(output_dir) if output_dir is not None else DEFAULT_BET_LIST_DIR
    return directory, directory / BET_LIST_ARTIFACT_NAME


def graded_weeks(season: int, *, output_dir: Path | str | None = None) -> set[int]:
    """Return the weeks of *season* whose bets have already been graded.

    THE SEAM. Phase 34 repoints this function at the forward ledger; see the module
    docstring.

    Args:
        season: The season to ask about.
        output_dir: The store directory: the bet-list directory (default
            ``backtest.weekly_bet_list.DEFAULT_BET_LIST_DIR``), or the ledger directory
            (default ``forward_ledger.store.LEDGER_DIR``) when the cutover switch is on.
            Tests pass a ``tmp_path``.

    Returns:
        The set of graded week numbers, possibly empty. An EMPTY set means "the store was
        read and nothing in it is graded for this season" -- never "the store could not be
        read". Use :func:`graded_weeks_record` when that distinction has to be recorded.

    Raises:
        GradedWeeksUnavailable: If a store exists but cannot be read -- a schema mismatch,
            a truncated or corrupt file, an I/O failure, or a ledger whose chain does not
            verify. The message names the path and the underlying error, and the original
            exception is chained.
    """
    if _reads_the_ledger():
        return _ledger_graded_weeks(season, output_dir)

    from backtest.weekly_bet_list import read_bet_list_artifact

    directory, path = _artifact_location(output_dir)
    try:
        frame = read_bet_list_artifact(directory)
    except (OSError, ValueError) as exc:
        # Translate AT THE FETCH BOUNDARY -- the rule 32-RESEARCH.md Pitfall 4 states for
        # nflreadpy, applied to the store. ``read_bet_list_artifact`` raises ValueError on a
        # schema mismatch; pyarrow's ArrowInvalid (a corrupt or truncated parquet) is itself
        # a ValueError subclass and ArrowIOError an OSError one, so these two arms cover the
        # family without naming pyarrow and taking a dependency on its exception layout.
        msg = (
            f"The graded-week state for season {season} could not be determined: the bet "
            f"list at '{path}' exists but could not be read ({exc.__class__.__name__}: "
            f"{exc}). This is NOT a claim that no weeks are graded -- that claim needs a "
            "readable store. Regenerate the artifact pair with "
            "`uv run python scripts/generate_bet_list.py`, or record the verdict as UNKNOWN."
        )
        raise GradedWeeksUnavailable(msg) from exc

    if frame.empty:
        return set()
    settled = frame[
        (frame["season"] == season)
        & frame["grading_status"].isin(TERMINAL_GRADING_STATUSES)
    ]
    return {int(week) for week in settled["week"]}


def _ledger_graded_weeks(season: int, output_dir: Path | str | None) -> set[int]:
    """The graded weeks of *season* in the forward ledger, accepted only on a verified chain.

    Raises:
        GradedWeeksUnavailable: the ledger exists but a line cannot be read, or its chain does
            not verify -- chained from the underlying refusal.
    """
    from forward_ledger.canonical import ENTRY_KIND_ROW
    from forward_ledger.store import (
        LedgerChainBrokenError,
        LedgerFormatError,
        read_entries,
        verify_chain,
    )

    directory, path = _artifact_location(output_dir)
    unreadable = (
        f"The graded-week state for season {season} could not be determined: the forward "
        f"ledger at '{path}' exists but"
    )
    try:
        entries = read_entries(directory)
    except LedgerFormatError as exc:
        msg = (
            f"{unreadable} could not be read ({exc}). This is NOT a claim that no weeks are "
            "graded -- that claim needs a readable ledger. Record the verdict as UNKNOWN."
        )
        raise GradedWeeksUnavailable(msg) from exc

    verdict = verify_chain(entries)
    if not verdict.ok:
        broken = LedgerChainBrokenError(
            f"chain broken at seq {verdict.first_broken_seq} (key "
            f"{verdict.first_broken_key}): {verdict.reason}"
        )
        msg = (
            f"{unreadable} its chain does not verify ({broken}). A tampered or damaged ledger "
            "is never read as a graded-week answer. Record the verdict as UNKNOWN."
        )
        raise GradedWeeksUnavailable(msg) from broken

    return {
        int(entry.immutable["week"])
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW
        and entry.immutable["season"] == season
        and (entry.grading or {}).get("grading_status") in TERMINAL_GRADING_STATUSES
    }


def graded_weeks_record(
    season: int, *, output_dir: Path | str | None = None
) -> dict[str, Any]:
    """Return :func:`graded_weeks`'s answer as a recordable block.

    The wrapper that makes an empty answer EXPLICITLY RECORDED rather than merely returned.
    ``source`` is present on every path, so a verdict can never report "no weeks graded"
    without also reporting where it looked; ``weeks`` is a sorted ``list[int]`` rather than
    a ``set`` so the block serialises straight into a committed JSON manifest.

    It deliberately does NOT catch :class:`GradedWeeksUnavailable`. An unresolvable store is
    the CALLER's decision to record as ``UNKNOWN`` (with its reason); swallowing it here
    would put the silent downgrade straight back. ``resolved`` is therefore ``True`` on
    every path that returns at all -- it is a field for readers, not a branch for callers.

    Args:
        season: The season to ask about.
        output_dir: The bet-list directory, as for :func:`graded_weeks`.

    Returns:
        ``{"season": int, "weeks": list[int], "source": str, "resolved": True,
        "reason": str | None}``. ``source`` is :func:`graded_weeks_source` -- the store
        actually read. ``reason`` names the absent store path when the store does not exist,
        and is ``None`` when it does -- so "there is no store" and "the store holds nothing
        graded yet" stay distinguishable in the record.

    Raises:
        GradedWeeksUnavailable: Propagated unchanged from :func:`graded_weeks`.
    """
    weeks = graded_weeks(season, output_dir=output_dir)
    _, path = _artifact_location(output_dir)
    store = "forward ledger" if _reads_the_ledger() else "bet list"
    reason: str | None = None
    if not path.exists():
        reason = (
            f"no {store} at '{path}', so no week of season {season} has been graded yet. "
            "This is the state of a checkout that has never generated one, and is recorded "
            "rather than inferred."
        )
    return {
        "season": int(season),
        "weeks": sorted(weeks),
        "source": graded_weeks_source(),
        "resolved": True,
        "reason": reason,
    }
