"""The two kinds of push the owner gets -- "failed" and "bets" -- built from a run's report (D-06).

PURE BUILDERS
-------------
Every function here SELECTS and FORMATS stored values: what the run recorded on its
:class:`notify.report.RunReport`, or the rows it wrote. None of them computes an EV, a stake, a
return or a rate (SPEC R6), and none of them sends anything: each returns
:class:`notify.events.PushMessage` values for the closed router in :mod:`notify.events`.

WHAT "FAILED" MEANS (SPEC R2, D-23)
-----------------------------------
Anything less than a clean run. A daily-run outcome other than clean gives exactly ONE
top-priority message; each ledger or closing failure gives one normal-priority message of its own.
The classification keys on ``report.status`` alone, so a critical step failure -- which the
orchestrator raises, so the daily run records it as ``run_failed`` -- is one event, never two
(RESEARCH Pitfall 5). Push failures are read from the sync's error fields, never from its
``*_pushed`` booleans: nothing is pushed on a normal no-new-row day (Pitfall 6). A frozen-arm hole
and a re-fit problem (Phase 37: refused, crashed, ran late, an unclean A/A week) add no push when
the run already owes one: each is merged into the first.

THE LOCK TIME IS NEVER WRITTEN HERE
-----------------------------------
The next-action line shows the lock instant the daily run supplies (``RunReport.lock``, built by
``utils.game_lock``), converted to ``utils.game_lock.ET``. This module never restates the lock
hour, so its prose cannot drift from the one rule.

THE BETS PUSH (D-14)
--------------------
It lists only the rows the run newly APPENDED to the forward ledger with the live arm and live
status (``RunReport.appended_rows``, exactly as written), in ``/bets`` order, rendered through the
shared ``components/_bet_pick.html`` macros so the pick reads as it does on ``/bets``.

NO SECRET IN A MESSAGE (D-10)
-----------------------------
Error text is one line, with every configured secret value replaced by ``<redacted>`` and every URL
cut to its scheme and host, ASCII, at most :data:`ONE_LINE_ERROR_LIMIT` characters.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from api.cache import ARM_LIVE, BET_STATUS_LIVE
from notify.events import PRIORITY_NORMAL, PRIORITY_TOP, PushMessage, ascii_text
from notify.rendering import template_environment
from notify.report import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_FINISHED_WITH_SKIPS,
    STATUS_LOCK_PASSED,
    STATUS_RUN_FAILED,
    RunReport,
)
from notify.transport import scrub_text
from utils.game_lock import ET

if TYPE_CHECKING:  # typing only: notify never imports refit at load time
    from refit.aa import AAWeekResult
    from refit.hook import RefitOutcome

__all__ = [
    "ONE_LINE_ERROR_LIMIT",
    "RERUN_COMMAND",
    "bets_message",
    "closing_failure_message",
    "email_failed_message",
    "failed_messages",
    "game_labels",
    "lock_clock_text",
    "merge_into_first_failed",
    "next_action_line",
    "one_line_error",
]

ONE_LINE_ERROR_LIMIT = 160
RERUN_COMMAND = "uv run python -m scripts.daily_lock_pipeline"

# The commands docs/guides/RUNBOOK.md names for each ledger or closing failure.
_SYNC_COMMAND = "uv run python -m scripts.sync_ledger"
_VERIFY_COMMAND = "uv run python -m scripts.verify_ledger"
_CLOSING_CHECK_COMMAND = (
    "uv run python deployment/setup_scheduling.py --verify-installed"
)
_PUSH_SITE_COMMAND = "uv run python -m scripts.push_site"

# The website push outcomes that alarm (Phase 36, SPEC R7). ``succeeded`` and every ``not_pushed``
# send nothing: a failed run already sends its own top-priority push. Plain strings, because
# ``notify`` never imports ``pipeline`` at load time (``pipeline.site_push.OUTCOME_*``).
_WEBSITE_PUSH_ALARMS = frozenset({"failed", "refused"})

# The closing-trigger skips that leave the coming kickoff windows unregistered. A skip because the
# closing task is running right now is clean: the previous registration still covers the week.
_FAILING_TRIGGER_SKIPS = frozenset(
    {"closing_task_not_installed", "machine_not_eastern"}
)

# A URL, with any user:password@ part, kept as its scheme and host only.
_URL = re.compile(r"(https?)://(?:[^\s/?#@]*@)?([^\s/?#:]+)\S*")

_NOT_AFFECTED = "Games: none (predictions are not affected)"

BETS_PUSH_TEMPLATE = "bets_push.txt.j2"
# English weekday names by date.weekday(), independent of the machine's locale.
_WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
# English short weekday (by date.weekday()) and month (by date.month - 1) names, never the locale's.
_WEEKDAY_ABBR: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

# The A/A reason whose first mismatch names a FIELD; every other unclean reason is named itself.
_AA_FIELD_MISMATCH = "field_mismatch"
_SEASON_END = "the season ends"
_REFIT_STATUS_LATE = "ran late"

# What each top-priority daily-run outcome means, after "the daily run for <date>".
_TOP_WHAT_FAILED: dict[str, str] = {
    STATUS_RUN_FAILED: "stopped before it finished",
    STATUS_FAILED: "stopped at a critical step",
    STATUS_FINISHED_WITH_SKIPS: "left games out: an input arrived after their lock",
    STATUS_DEGRADED: "finished, but a non-critical step failed",
    STATUS_LOCK_PASSED: (
        "started after tomorrow's lock, so nothing was collected or predicted"
    ),
}


def game_labels(game_ids: Sequence[str] | None) -> str:
    """``AWAY @ HOME`` for each game id, comma-separated; ``unknown`` for None, ``none`` if empty.

    The matchup is the id's ``AWAY@HOME`` suffix, read as ``/bets`` reads it; an id without one is
    shown whole.
    """
    if game_ids is None:
        return "unknown"
    if not game_ids:
        return "none"
    labels: list[str] = []
    for game_id in game_ids:
        matchup = str(game_id).split("_")[-1]
        labels.append(matchup.replace("@", " @ ") if "@" in matchup else str(game_id))
    return ", ".join(labels)


def _safe_line(text: str, secret_values: Iterable[str]) -> str:
    """The first line of *text*: secrets redacted, URLs cut to scheme and host, ASCII, capped."""
    lines = text.splitlines()
    first = lines[0] if lines else ""
    scrubbed = scrub_text(first, secret_values)
    reduced = _URL.sub(lambda match: f"{match.group(1)}://{match.group(2)}", scrubbed)
    return ascii_text(reduced).strip()[:ONE_LINE_ERROR_LIMIT]


def one_line_error(
    exception_class: str, exception_text: str | None, secret_values: Iterable[str]
) -> str:
    """``Class: first line`` as one safe line (D-10).

    Every value in *secret_values* is replaced by ``<redacted>`` (the longest first), every URL is
    reduced to its scheme and host, non-ASCII becomes ``?``, and the line is cut to
    :data:`ONE_LINE_ERROR_LIMIT` characters.
    """
    text = f"{exception_class}: {exception_text}" if exception_text else exception_class
    return _safe_line(text, secret_values)


def lock_clock_text(lock: datetime) -> str:
    """The lock instant as an ET wall clock, ``<h>:<MM> AM|PM ET``.

    Built from the hour and minute, because the platform ``strftime`` flag for an unpadded
    12-hour clock is not portable to Windows. It formats the instant it is given and never
    restates the lock hour.
    """
    local = lock.astimezone(ET)
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    return f"{hour}:{local.minute:02d} {meridiem} ET"


def next_action_line(lock: datetime, now: datetime) -> str:
    """Before the lock, the re-run command; at or after it, that the games can no longer be predicted."""
    clock = lock_clock_text(lock)
    if now < lock:
        return f"Next: re-run before {clock} today: {RERUN_COMMAND}"
    return (
        f"Next: these games can no longer be predicted (their {clock} lock has passed)."
    )


def _games_without_prediction(report: RunReport) -> tuple[str, ...] | None:
    """The games the run left without a prediction; None when that cannot be read.

    What the run recorded comes first. A degraded run predicted every game; a finished-with-skips
    run left out exactly its skipped games; otherwise -- the slate was never selected -- it is
    tomorrow's games per the recorded schedule.
    """
    if report.unpredicted_game_ids is not None:
        return report.unpredicted_game_ids
    if report.status == STATUS_DEGRADED:
        return ()
    if report.status == STATUS_FINISHED_WITH_SKIPS:
        return report.skipped_game_ids
    return report.tomorrow_game_ids


def _is_top_failure(report: RunReport) -> bool:
    """True when the daily-run outcome itself is not clean (SPEC R2's top-priority list)."""
    if report.status in (
        STATUS_FAILED,
        STATUS_DEGRADED,
        STATUS_FINISHED_WITH_SKIPS,
        STATUS_RUN_FAILED,
    ):
        return True
    if report.status == STATUS_LOCK_PASSED:
        # Unknown games are a failure: silence could hide lost games.
        return report.tomorrow_game_ids is None or len(report.tomorrow_game_ids) > 0
    return False


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _top_title(report: RunReport, games: tuple[str, ...] | None) -> str:
    if report.status == STATUS_DEGRADED:
        return f"FAILED: daily run degraded ({_plural(len(report.failed_steps), 'step')} failed)"
    if games is None:
        return "FAILED: daily run, tomorrow's games unknown"
    if not games:
        return "FAILED: daily run (0 games not predicted)"
    return f"FAILED: {_plural(len(games), 'game')} not predicted"


def _top_message(
    report: RunReport, *, now: datetime, secret_values: Iterable[str]
) -> PushMessage:
    games = _games_without_prediction(report)
    what = _TOP_WHAT_FAILED[str(report.status)]
    lines = [
        f"What failed: the daily run for {report.run_date_et.isoformat()} {what}",
        f"Games: {game_labels(games)}",
    ]
    if report.exception_class is not None:
        error = one_line_error(
            report.exception_class, report.exception_text, secret_values
        )
        lines.append(f"Error: {error}")
    elif report.status == STATUS_DEGRADED:
        lines.append(f"Steps: {', '.join(report.failed_steps) or 'unknown'}")
    lines.append(next_action_line(report.lock, now))
    return PushMessage(
        event_class="failed",
        title=_top_title(report, games),
        header_lines=tuple(lines),
        item_lines=(),
        priority=PRIORITY_TOP,
    )


def _normal_message(
    title: str, what: str, error: str | None, next_line: str
) -> PushMessage:
    lines = [f"What failed: {what}", _NOT_AFFECTED]
    if error:
        lines.append(f"Error: {error}")
    lines.append(next_line)
    return PushMessage(
        event_class="failed",
        title=title,
        header_lines=tuple(lines),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )


def _settle_messages(
    report: RunReport, secret_values: Iterable[str]
) -> list[PushMessage]:
    retry = "Next: nothing to run now; the next daily run retries it."
    messages: list[PushMessage] = []
    if report.settle_error is not None:
        messages.append(
            _normal_message(
                "FAILED: ledger settle",
                "the ledger settle pass (grading, corrections, closing lines) raised",
                one_line_error(report.settle_error, None, secret_values),
                retry,
            )
        )
    deferred = getattr(report.settle, "closing_deferred", None)
    if deferred:
        messages.append(
            _normal_message(
                "FAILED: ledger closing finalization deferred",
                "grading ran, but closing-line finalization was deferred",
                _safe_line(str(deferred), secret_values),
                retry,
            )
        )
    return messages


def _closing_schedule_message(
    report: RunReport, secret_values: Iterable[str]
) -> PushMessage | None:
    triggers = report.triggers
    if triggers is None:
        return None
    reason = _safe_line(str(triggers.reason or ""), secret_values) or None
    next_line = f"Next: check the closing task: {_CLOSING_CHECK_COMMAND}"
    if triggers.outcome == "failed" or (
        triggers.outcome == "skipped" and triggers.reason in _FAILING_TRIGGER_SKIPS
    ):
        return _normal_message(
            "FAILED: closing line schedule",
            f"the closing-line wake triggers were not registered ({triggers.outcome})",
            reason,
            next_line,
        )
    if triggers.outcome == "registered" and not triggers.readback_match:
        return _normal_message(
            "FAILED: closing line schedule read back wrong",
            "the closing-line wake task was registered but did not read back as expected",
            reason,
            next_line,
        )
    return None


def _sync_messages(
    report: RunReport, secret_values: Iterable[str]
) -> list[PushMessage]:
    messages: list[PushMessage] = []
    push_next = f"Next: push it with {_SYNC_COMMAND}"
    sync = report.sync
    if sync is not None and sync.anchor_error:
        messages.append(
            _normal_message(
                "FAILED: ledger anchor push",
                "the ledger anchor was not committed or pushed",
                _safe_line(sync.anchor_error, secret_values),
                push_next,
            )
        )
    if sync is not None and sync.backup_error:
        messages.append(
            _normal_message(
                "FAILED: ledger backup push",
                "the ledger backup was not committed or pushed",
                _safe_line(sync.backup_error, secret_values),
                push_next,
            )
        )
    refused_next = (
        f"Next: run {_VERIFY_COMMAND}, then follow the restore procedure in "
        "docs/guides/RUNBOOK.md"
    )
    if sync is not None and sync.refused_reason:
        error: str | None = _safe_line(sync.refused_reason, secret_values)
    elif report.sync_error is not None:
        error = one_line_error(report.sync_error, None, secret_values)
    elif report.sync_ran and sync is None:
        error = "the sync did not complete (recorded in logs/ledger_runs.jsonl)"
    else:
        return messages
    messages.append(
        _normal_message(
            "FAILED: ledger sync refused",
            "the end-of-run ledger sync refused or did not complete",
            error,
            refused_next,
        )
    )
    return messages


def _site_push_messages(
    report: RunReport, secret_values: Iterable[str]
) -> list[PushMessage]:
    """The one normal-priority message for a website push that failed or refused (SPEC R7).

    Read from the stored ``PushOutcome`` (its outcome, reason and data-format versions), never
    re-derived. A raise that escaped the push routine left no record, only its class on the
    report, and is named by that class.
    """
    outcome = report.site_push
    lines: list[str] = []
    if report.site_push_error is not None:
        lines.append("What failed: the website push (failed: raised)")
        lines.append(
            f"Error: {one_line_error(report.site_push_error, None, secret_values)}"
        )
    elif outcome is not None and outcome.outcome in _WEBSITE_PUSH_ALARMS:
        lines.append(
            _safe_line(
                f"What failed: the website push ({outcome.outcome}: "
                f"{outcome.reason or 'unknown'})",
                secret_values,
            )
        )
        if str(outcome.reason).startswith("format_mismatch"):
            lines.append(
                _safe_line(
                    f"Data format: the data file is {outcome.data_format_found}, the pages "
                    f"expect {outcome.data_format_expected}",
                    secret_values,
                )
            )
    else:
        return []
    lines.append(
        f"Next: run {_PUSH_SITE_COMMAND}, or wait for the next 5 PM run",
    )
    return [
        PushMessage(
            event_class="failed",
            title="FAILED: website push",
            header_lines=tuple(lines),
            item_lines=(),
            priority=PRIORITY_NORMAL,
        )
    ]


def merge_into_first_failed(
    messages: list[PushMessage], *, standalone: PushMessage, title_suffix: str
) -> list[PushMessage]:
    """Fold a new failure into the run's FIRST failed message, or send it on its own (SPEC R14).

    When *messages* is empty, the result is *messages* followed by *standalone*. Otherwise the new
    failure adds no push of its own: the first message is replaced by one titled
    ``<its title> + <title_suffix>`` whose header lines are its own followed by *standalone*'s, so
    the merged push names both failures; it keeps its own priority and event class. Every later
    message is returned untouched -- this does not collapse the run's other failed messages into
    one. Plan 37-14 reuses it for re-fit trouble.
    """
    if not messages:
        return [*messages, standalone]
    first, *rest = messages
    merged = PushMessage(
        event_class=first.event_class,
        title=f"{first.title} + {title_suffix}",
        header_lines=(*first.header_lines, *standalone.header_lines),
        item_lines=first.item_lines,
        priority=first.priority,
    )
    return [merged, *rest]


def _shadow_hole_count(report: RunReport) -> str:
    """``<n> row(s)``: how many new live rows the frozen arm could not twin."""
    return _plural(len(report.shadow_holes), "row")


def _shadow_hole_message(
    report: RunReport, secret_values: Iterable[str]
) -> PushMessage | None:
    """The normal-priority push for the frozen arm's holes; None when it twinned every row (R8).

    Read from ``report.shadow_holes`` as the recommend step recorded them (D-04: never
    recomputed). The error line names exception CLASSES only, never a message (D-10).
    """
    holes = report.shadow_holes
    if not holes:
        return None
    week = report.slate_week if report.slate_week is not None else holes[0][0][2]
    reasons = dict.fromkeys(str(reason) for _key, reason in holes)
    error = ", ".join(one_line_error(reason, None, secret_values) for reason in reasons)
    return PushMessage(
        event_class="failed",
        title=f"FAILED: frozen arm missed {_shadow_hole_count(report)} (week {week})",
        header_lines=(
            f"What failed: the frozen comparison arm could not score {len(holes)} of "
            "tonight's rows; the live rows were written",
            f"Error: {error}",
            "Next: nothing to run; the comparison uses paired games only and the hole is "
            "recorded in ledger/refit/shadow_holes.jsonl",
        ),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )


def _short_day(day: date | str) -> str:
    """``Tue Nov 3``: a date (or its ISO text) with English names, never the machine's locale."""
    value = date.fromisoformat(day) if isinstance(day, str) else day
    return (
        f"{_WEEKDAY_ABBR[value.weekday()]} {_MONTH_ABBR[value.month - 1]} {value.day}"
    )


def _refit_date_facts(
    outcome: RefitOutcome, iso_date: str
) -> tuple[int | None, int | None]:
    """``(date index, NFL week)`` of the cadence date *iso_date* of the outcome's list.

    The outcome's own date carries both. Another date of the same list -- a recovered close in a
    run that then attempted a later date -- is looked up in the frozen list (lazy import: notify
    never imports refit at load time).
    """
    if iso_date == outcome.date:
        return outcome.date_index, outcome.nfl_week
    if outcome.list_week is None:
        return None, None
    from refit.cadence import cadence_dates

    for cadence_date in cadence_dates(outcome.list_week):
        if cadence_date.date.isoformat() == iso_date:
            return cadence_date.index, cadence_date.nfl_week
    return None, None


def _week_suffix(nfl_week: int | None) -> str:
    return "" if nfl_week is None else f" (week {nfl_week})"


def _retry_line(next_date: str | None) -> str:
    until = _short_day(next_date) if next_date else _SEASON_END
    return f"Next: it retries at every daily run until {until}."


def _aa_part(
    result: AAWeekResult, secrets: Sequence[str]
) -> tuple[str, str, list[str]]:
    """The push part of one unclean A/A week: (title fragment, merge fragment, lines)."""
    mismatch = result.first_mismatch or {}
    where = (
        f" (game {game_labels((mismatch['game_id'],))}, target {mismatch['target']})"
        if mismatch.get("game_id") and mismatch.get("target")
        else ""
    )
    if result.reason == _AA_FIELD_MISMATCH:
        what = (
            f"did not match the live arm on {mismatch.get('field', 'unknown')}{where}"
        )
    else:
        what = f"is not a clean A/A week ({result.reason}){where}"
    return (
        f"frozen arm mismatch (week {result.week})",
        "frozen arm mismatch",
        [
            _safe_line(f"The frozen arm's week {result.week} {what}.", secrets),
            "Next: nothing to run; the first re-fit waits for the next clean week.",
        ],
    )


def _late_part(
    index: int | None, iso_date: str, nfl_week: int | None, swapped_on: date
) -> tuple[str, str, list[str]]:
    """The push part of a re-fit installed after its own day: still FAILED (D-16).

    *swapped_on* is the ET day the swap happened, so the new models serve from the next day's run.
    """
    number = "" if index is None else f" {index}"
    live_from = _short_day(swapped_on + timedelta(days=1))
    return (
        f"model re-fit ran late{_week_suffix(nfl_week)}",
        "model re-fit ran late",
        [
            f"Re-fit{number}, due {_short_day(iso_date)}, ran late, on "
            f"{_short_day(swapped_on)}. New models are live from {live_from}.",
            "Next: nothing to run; the late re-fit is recorded.",
        ],
    )


def _attempt_part(
    outcome: RefitOutcome, report: RunReport, secrets: Sequence[str]
) -> tuple[str, str, list[str]] | None:
    """The push part of the attempt's own status; None for every status that is not trouble.

    None also when the status describes the recovered close itself (a run that only recovered a
    pending install), which its own part already reports (WR-C01).
    """
    if outcome.date is None:
        return None
    if (
        outcome.recovered_close is not None
        and outcome.recovered_close[0] == outcome.date
    ):
        return None
    if outcome.status == _REFIT_STATUS_LATE:
        return _late_part(
            outcome.date_index, outcome.date, outcome.nfl_week, report.run_date_et
        )
    due = f"Re-fit {outcome.date_index}, due {_short_day(outcome.date)}"
    week = _week_suffix(outcome.nfl_week)
    if outcome.status == "refused":
        check = _safe_line(str(outcome.check or "unknown"), secrets)
        return (
            f"model re-fit refused{week}",
            "model re-fit refused",
            [
                f"{due}, was refused by the check: {check}. The current models keep serving.",
                _retry_line(outcome.next_date),
            ],
        )
    if outcome.status == "crashed":
        error = one_line_error(outcome.exception_class or "unknown", None, secrets)
        return (
            f"model re-fit crashed{week}",
            "model re-fit crashed",
            [
                f"{due}, crashed. The current models keep serving.",
                f"Error: {error}",
                _retry_line(outcome.next_date),
            ],
        )
    return None


def _refit_message(
    report: RunReport, secrets: Sequence[str]
) -> tuple[PushMessage, str] | None:
    """One normal-priority push for every re-fit problem this run produced, and its merge suffix.

    Read from the side step's outcome as it returned it (D-04: never recomputed), in order: each
    unclean A/A week it evaluated, a recovered close that ran late, the attempt's own status
    (refused, crashed, ran late), then an exception that escaped the side step. A catch-up run's
    earlier mismatch is therefore alerted even when the run then installs on time. The title
    joins every problem with `` + `` (D-16); the merge suffix joins the same problems without
    their weeks, as the frozen-arm hole merge does. None when there is no problem (ran on time,
    waiting with clean or no A/A results, not_due, busy, no_refit_this_season). Class names only
    (D-10).
    """
    parts: list[tuple[str, str, list[str]]] = []
    outcome = report.refit
    if outcome is not None:
        parts.extend(
            _aa_part(result, secrets)
            for result in outcome.aa_results
            if not result.clean
        )
        if outcome.recovered_close is not None:
            iso_date, status, swapped_on = outcome.recovered_close
            if status == _REFIT_STATUS_LATE:
                index, nfl_week = _refit_date_facts(outcome, iso_date)
                parts.append(
                    _late_part(
                        index, iso_date, nfl_week, date.fromisoformat(swapped_on)
                    )
                )
        attempt = _attempt_part(outcome, report, secrets)
        if attempt is not None:
            parts.append(attempt)
    if report.refit_error is not None:
        error = f"Error: {one_line_error(report.refit_error, None, secrets)}"
        if report.refit_error_location is not None:
            # A torn or damaged record never heals by itself, so say which one and how (WR-C05).
            lines = [
                _safe_line(
                    f"The re-fit record {report.refit_error_location} could not be read.",
                    secrets,
                ),
                error,
                'Next: repair that record (RUNBOOK.md, "A re-fit record that will not '
                'read"); the re-fit stays stopped until it reads.',
            ]
        else:
            lines = [
                "The re-fit side step raised before it reported an outcome.",
                error,
                "Next: nothing to run; the next daily run tries again.",
            ]
        parts.append(("model re-fit crashed", "model re-fit crashed", lines))
    if not parts:
        return None
    message = PushMessage(
        event_class="failed",
        title="FAILED: " + " + ".join(title for title, _merge, _lines in parts),
        header_lines=tuple(line for *_titles, lines in parts for line in lines),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )
    suffix = " + ".join(dict.fromkeys(merge for _title, merge, _lines in parts))
    return message, suffix


def failed_messages(
    report: RunReport, *, now: datetime, secret_values: Iterable[str] = ()
) -> list[PushMessage]:
    """Every failed push this run owes, top priority first (SPEC R2, D-23).

    At most one top-priority message for the daily-run outcome, then one normal-priority message
    per ledger, closing or website failure in this fixed order: settle raised, closing finalization
    deferred, closing schedule, anchor push, backup push, sync refused, website push. A clean run
    gives none.

    Last, the frozen arm's holes and then the re-fit's problems (Phase 37, SPEC R8/R14, D-16):
    each adds no push of its own when the run already sends a failed message -- it is merged into
    the FIRST one (:func:`merge_into_first_failed`), which keeps its priority -- and is otherwise
    sent on its own at normal priority (D-35). The other failed messages stay unchanged.
    """
    secrets = tuple(secret_values)
    messages: list[PushMessage] = []
    if _is_top_failure(report):
        messages.append(_top_message(report, now=now, secret_values=secrets))
    messages.extend(_settle_messages(report, secrets))
    closing = _closing_schedule_message(report, secrets)
    if closing is not None:
        messages.append(closing)
    messages.extend(_sync_messages(report, secrets))
    messages.extend(_site_push_messages(report, secrets))
    holes = _shadow_hole_message(report, secrets)
    if holes is not None:
        messages = merge_into_first_failed(
            messages,
            standalone=holes,
            title_suffix=f"frozen arm missed {_shadow_hole_count(report)}",
        )
    refit = _refit_message(report, secrets)
    if refit is not None:
        standalone, suffix = refit
        messages = merge_into_first_failed(
            messages, standalone=standalone, title_suffix=suffix
        )
    return messages


def email_failed_message(run_date_et: date, failure_kind: str) -> PushMessage:
    """The normal-priority push for a weekly email that was not delivered or not built (SPEC R8).

    *failure_kind* is an HTTP status (``http_500``), ``missing_secret``, or
    ``build_<ExceptionClass>`` for an email that could not be built.
    """
    return PushMessage(
        event_class="failed",
        title="FAILED: weekly email",
        header_lines=(
            f"What failed: the weekly email for {run_date_et.isoformat()} "
            f"({ascii_text(failure_kind)})",
            "Next: the email is not re-sent automatically; this week's list is on /bets.",
        ),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )


def closing_failure_message(
    game_ids: Sequence[str] | None, error_type: str
) -> PushMessage:
    """The closing task's normal-priority push for a failed capture (SPEC R2).

    *error_type* is the exception's class name only: a message can carry a request URL.
    """
    return PushMessage(
        event_class="failed",
        title="FAILED: closing line capture",
        header_lines=(
            "What failed: the closing line capture",
            f"Games: {game_labels(game_ids)}",
            f"Error: {ascii_text(error_type)[:ONE_LINE_ERROR_LIMIT]}",
            "Next: nothing to re-run; their closing columns stay empty with reason "
            "capture_failed.",
        ),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )


def _bets_page_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*rows* in ``/bets`` order: ``per_bet_ev`` descending with NULLs last, then season, week,
    game_id, target (``DataService.get_bet_list``'s ORDER BY).

    A selection, not arithmetic: one stable sort on the tie-break, then a stable descending sort
    on the stored EV, with the rows that have none kept last in tie-break order.
    """
    ranked = sorted(rows, key=itemgetter("season", "week", "game_id", "target"))
    with_ev = [row for row in ranked if row["per_bet_ev"] is not None]
    without_ev = [row for row in ranked if row["per_bet_ev"] is None]
    return [*sorted(with_ev, key=itemgetter("per_bet_ev"), reverse=True), *without_ev]


def bets_message(report: RunReport) -> PushMessage | None:
    """The one bets push for the live bets this run newly appended, or None when there are none.

    Only rows with the live arm AND live status count (the ledger also stores suppressed rows, and
    Phase 37's shadow arm writes its own). An identical re-run appends nothing, so it gives None;
    a re-run that adds a bet lists exactly that bet. The 4,000-byte cap and ``... and N more`` are
    applied by the router (``notify.events.fit_lines``) when it sends the message.
    """
    live = [
        row
        for row in report.appended_rows
        if row["arm"] == ARM_LIVE and row["status"] == BET_STATUS_LIVE
    ]
    if not live:
        return None
    ordered = _bets_page_order(live)
    season = (
        report.slate_season if report.slate_season is not None else ordered[0]["season"]
    )
    week = report.slate_week if report.slate_week is not None else ordered[0]["week"]
    slate_day = _WEEKDAYS[(report.run_date_et + timedelta(days=1)).weekday()]
    rendered = (
        template_environment().get_template(BETS_PUSH_TEMPLATE).render(bets=ordered)
    )
    item_lines = tuple(line.strip() for line in rendered.splitlines() if line.strip())
    noun = "new bet" if len(live) == 1 else "new bets"
    return PushMessage(
        event_class="bets",
        title=f"{len(live)} {noun}: {season} week {week}, {slate_day}",
        header_lines=(f"{season} week {week}, {slate_day} games",),
        item_lines=item_lines,
        priority=PRIORITY_NORMAL,
    )
