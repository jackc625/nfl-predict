"""The Saturday digest email: which run sends it, what it reads and how it is sent (SPEC R4, D-07, D-19).

WHICH RUN SENDS IT
------------------
The run whose slate holds the week's Sunday games -- the Saturday 17:00 ET run -- sends one email
every in-season week, playoffs included, and also when that run failed. A week with no Sunday game
falls back to its LAST slate day. "Which week is tomorrow" and "which day is its digest day" are read
from the RECORDED schedule (``utils.current_slate``), which this very run refreshed, so the rule holds
on the failure path too (:func:`digest_week_for_run`).

READ, NEVER RECOMPUTE (NOTIF-05)
--------------------------------
Every figure comes from the web cache through ``api.services.DataService`` -- the getters ``/bets``
reads -- opened read-only, with the getters' TTL cache cleared first (it is keyed without the
connection, RESEARCH Pitfall 9) and the connection closed afterwards (an open handle can block the
next cache rebuild on Windows). The builders here SELECT and FORMAT stored values: no EV, stake,
return, rate or count arithmetic. The bet count in the subject is the length of the selected list.

"BETS UNKNOWN", NEVER "0 BETS"
------------------------------
The cache is CURRENT for this run when its per-week ``bet_list_populated_at`` marker (the project's
own freshness rule, D31-29) is at or after the run's start. An absent or older cache gives a stale
notice and ``bets unknown``: a missing list is never rendered as zero bets (RESEARCH Pitfall 11).

UPSTREAM DATA CHANGES AND CORRECTED GRADES (D-03, D-21)
--------------------------------------------------------
After the season-to-date block comes the week's closing section (SPEC R4 item 7, D-23: skipped or
missed closing readings, first-pick refusals and closing-schedule read-back failures from the ledger
run log). Then, before the not-advice line, the email lists the latest probe
line's unacknowledged sealed changes, the live season's WARNING-or-louder verdicts and the corrected
grades recorded since the previous delivered digest (:mod:`notify.digest_sections`). Each source is
read on its own: one that cannot be read is named as unreadable in the section, never shown as
nothing, and never costs the rest of the email.

A SCHEDULE THAT CANNOT BE READ (D-24)
------------------------------------
When the recorded schedule cannot be read, the run cannot tell which week is tomorrow or whether
it is the digest run. On a SATURDAY it emails anyway, so a schedule-read failure never costs the
weekly signal: the week comes from the cache's stamped current slate (else the run's selected
slate, else ``week unknown``), the status reads ``RUN FAILED``, the bets read ``bets unknown`` (no
schedule confirms the list is this week's) and the games section says the schedule could not be
read. On any other weekday no email is attempted.

ONE EMAIL PER RUN
-----------------
The Resend Idempotency-Key is ``digest/<run_date_et>/<the run's start instant, UTC>``: the message is
built once, so it is stable across that run's retries, and every run -- including two lock-passed
re-runs of one day, which carry no run id -- sends its own email instead of replaying the first
send's response. A same-day re-send is marked ``(update)`` in the subject (D-07).

ASCII only, no emoji, no images, no links (CLAUDE.md hard constraint; D-19).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from api.cache import ARM_LIVE
from api.presentation import et_timestamp_label, last_week_display
from api.services import DataService, clear_cache
from data.sealed_probe_log import latest_entry
from data.upstream_live import load_live_manifest
from notify import digest_sections
from notify.events import (
    EmailMessage,
    SendContext,
    ascii_text,
    record_unbuilt_email,
    send_email,
)
from notify.log import read_notifications
from notify.messages import game_labels
from notify.rendering import template_environment
from notify.report import RunReport
from notify.transport import OUTCOME_DELIVERED, Delivery
from utils.current_slate import load_recorded_schedule
from utils.game_lock import ET, game_lock

if (
    TYPE_CHECKING
):  # pragma: no cover - typing only; pandas is loaded by the schedule reader
    import pandas as pd

__all__ = [
    "LOAD_SCHEDULE",
    "SCHEDULE_UNREADABLE_LINE",
    "SCHEDULE_UNREADABLE_TITLE",
    "STALE_NOTICE",
    "WEB_CACHE_PATH",
    "ZERO_BETS_LINE",
    "DigestInputs",
    "collect_digest_inputs",
    "digest_subject",
    "digest_week_for_run",
    "open_cache_service",
    "render_digest",
    "send_digest_if_due",
]

#: The web cache the daily run's last step rebuilds. Read at call time, so tests point it elsewhere.
WEB_CACHE_PATH = Path("data/web_cache.duckdb")

#: Seam, read at call time: the recorded schedule (tests hand in a prepared frame).
LOAD_SCHEDULE = load_recorded_schedule

DIGEST_HTML_TEMPLATE = "digest.html.j2"
DIGEST_TEXT_TEMPLATE = "digest.txt.j2"

# No character HTML escapes, so the notice reads byte-identical in both parts of the email.
STALE_NOTICE = (
    "The web cache is not current for this run: bets unknown. "
    "The list is not shown until the cache is rebuilt."
)
ZERO_BETS_LINE = "0 bets this week"

# D-24: the status line and the games-section line of a Saturday whose schedule cannot be read.
# Like STALE_NOTICE, no character HTML escapes, so the line reads the same in both parts.
SCHEDULE_UNREADABLE_TITLE = "FAILED: recorded schedule could not be read"
SCHEDULE_UNREADABLE_LINE = (
    "The recorded schedule could not be read, so the games of this week are not listed."
)

STATE_BET = "bet"
STATE_NO_BET = "no bet"
STATE_NOT_PREDICTED = "not predicted"
STATE_NOT_SHOWN = "result not shown -- the web cache is not current"

# The forward record's two classes, verdict record first, with /bets' own section headings.
_TRACKER_CLASS_ORDER: tuple[str, ...] = ("forward_realized", "pre_verdict")
_TRACKER_HEADINGS: dict[str, str] = {
    "forward_realized": "Forward record -- recommended before kickoff",
    "pre_verdict": "Forward record -- decided before the counting start week",
}

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
_SATURDAY = 5
_SUNDAY = 6
_FAILED_PREFIX = "FAILED: "


@dataclass(frozen=True)
class DigestInputs:
    """Everything one digest shows, selected from stored output. Nothing here was computed.

    Attributes:
        run_date_et: The sending run's ET date.
        run_start: The sending run's start instant (the idempotency key's per-run part).
        season: The digest week's season, or None when unknown.
        week: The digest week's number (playoff rounds continue it), or None when unknown.
        cache_current: True when the week's bet list was populated at or after the run's start.
        cache_populated_at: The week's stored populated-at marker, as stored, or None.
        bets: The week's live-arm live bets, in the cache's /bets order; empty when not current.
        games: One ``{game_id, label, state}`` per game of the week, in schedule order.
        last_week: ``api.presentation.last_week_display`` of last week's stored rows; None when
            no cache could be read (unavailable, never "nothing graded").
        tracker_blocks: The forward tracker blocks, verdict record first; None when no cache could
            be read.
        failed_titles: The titles of this run's failed messages; empty for a clean run.
        is_update: True when a digest for this run date was already delivered (D-07).
        label_prefix: Prepended to the subject and heading (``TEST - `` for the TEST command).
        schedule_readable: False when the recorded schedule could not be read (D-24): the games
            section then says so instead of listing games.
        data_changes: :func:`notify.digest_sections.data_change_section`'s output (D-03, D-21);
            None renders no section.
        closing: :func:`notify.digest_sections.closing_section`'s output (SPEC R4 item 7, D-23);
            None renders no section.
    """

    run_date_et: date
    run_start: datetime
    season: int | None
    week: int | None
    cache_current: bool
    cache_populated_at: str | None
    bets: list[dict[str, Any]]
    games: list[dict[str, Any]]
    last_week: dict[str, Any] | None
    tracker_blocks: list[dict[str, Any]] | None
    failed_titles: list[str]
    is_update: bool
    label_prefix: str = ""
    schedule_readable: bool = True
    data_changes: dict[str, Any] | None = None
    closing: dict[str, Any] | None = None


def _week_of_day(schedule: pd.DataFrame, day: date) -> tuple[int, int] | None:
    """The (season, week) of the games whose ET kickoff day is *day*, or None when there are none.

    One ET day names one slate (``utils.current_slate.prepare_schedule`` refuses otherwise).
    """
    games = schedule[schedule["et_day"] == day]
    if games.empty:
        return None
    first = games.iloc[0]
    return int(first["season"]), int(first["week"])


def _week_games(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """The week's games, in the prepared schedule's order (ET day, kickoff, game id)."""
    return schedule.loc[(schedule["season"] == season) & (schedule["week"] == week)]


def digest_week_for_run(
    schedule: pd.DataFrame, run_date_et: date
) -> tuple[int, int] | None:
    """The (season, week) this run emails about, or None when it is not that week's digest run.

    Tomorrow's games name the week. The week's digest day is its Sunday ET day when it has one,
    else its LAST ET day; the run is the digest run when tomorrow is that day.
    """
    tomorrow = run_date_et + timedelta(days=1)
    week_key = _week_of_day(schedule, tomorrow)
    if week_key is None:
        return None
    days = sorted(set(_week_games(schedule, *week_key)["et_day"]))
    sundays = [day for day in days if day.weekday() == _SUNDAY]
    digest_day = sundays[0] if sundays else days[-1]
    return week_key if tomorrow == digest_day else None


@contextmanager
def open_cache_service(path: Path | None = None) -> Iterator[DataService | None]:
    """A ``DataService`` over the web cache opened read-only, or None when it cannot be read.

    An absent file and a file that exists but cannot be opened (``duckdb.connect`` raising
    ``duckdb.Error`` -- a locked or corrupt file) both yield None: the email then says the cache
    is not current and reads ``bets unknown``, never ``0 bets``. An unopenable file prints
    ``NOTIFY_DIGEST_CACHE= unreadable <ExceptionClass>`` (the class name only).

    The getters' TTL cache is cleared before the read (and after it, so no figure outlives the
    connection), and the connection is closed on exit, on every path.

    Args:
        path: The cache file (default: :data:`WEB_CACHE_PATH`, read at call time).
    """
    cache_path = path if path is not None else WEB_CACHE_PATH
    if not cache_path.exists():
        yield None
        return
    clear_cache()
    conn: duckdb.DuckDBPyConnection | None
    try:
        conn = duckdb.connect(str(cache_path), read_only=True)
    except duckdb.Error as error:
        print(f"NOTIFY_DIGEST_CACHE= unreadable {type(error).__name__}")  # noqa: T201
        conn = None
    if conn is None:
        yield None
        return
    try:
        yield DataService(conn)
    finally:
        conn.close()
        clear_cache()


def _instant(value: object) -> datetime | None:
    """A stored timestamp as an aware instant (a naive value is UTC, as the cache writes it)."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _digest_week(
    report: RunReport, schedule: pd.DataFrame | None, service: DataService | None
) -> tuple[int | None, int | None]:
    """Tomorrow's week from the recorded schedule, else the run's selected slate, else unknown.

    When the schedule could not be read at all (D-24), the cache's stamped current slate comes
    first: it is the week the cache build itself resolved.
    """
    if schedule is not None:
        week_key = _week_of_day(schedule, report.run_date_et + timedelta(days=1))
        if week_key is not None:
            return week_key
    elif service is not None:
        stamped = service.get_current_slate()
        if stamped is not None:
            return stamped
    return report.slate_season, report.slate_week


def _game_states(
    report: RunReport,
    *,
    schedule: pd.DataFrame | None,
    season: int | None,
    week: int | None,
    current: bool,
    bet_game_ids: set[str],
    row_game_ids: set[str],
    now: datetime,
) -> list[dict[str, Any]]:
    """Every game of the week in one state (SPEC R4 item 3; the discretion block's stale rule)."""
    if schedule is None or season is None or week is None:
        return []
    unpredicted = set(report.unpredicted_game_ids or ())
    games: list[dict[str, Any]] = []
    week_games = _week_games(schedule, season, week)
    for game_value, kickoff in zip(
        week_games["game_id"], week_games["kickoff_et"], strict=True
    ):
        game_id = str(game_value)
        lock = game_lock(kickoff, game_id=game_id)
        if current and game_id in bet_game_ids:
            state = STATE_BET
        elif current and game_id in row_game_ids:
            state = STATE_NO_BET
        elif game_id in unpredicted:
            state = STATE_NOT_PREDICTED
        elif now < lock:
            state = f"picked {_WEEKDAYS[lock.astimezone(ET).weekday()]} evening"
        elif current:
            state = STATE_NOT_PREDICTED
        else:
            state = STATE_NOT_SHOWN
        games.append(
            {"game_id": game_id, "label": game_labels((game_id,)), "state": state}
        )
    return games


def _forward_tracker_blocks(
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The forward provenance's blocks, verdict record first, each with /bets' section heading."""
    ordered: list[dict[str, Any]] = []
    for validation_type in _TRACKER_CLASS_ORDER:
        for block in blocks:
            if (
                block["provenance"] == "forward"
                and block["validation_type"] == validation_type
            ):
                ordered.append({**block, "heading": _TRACKER_HEADINGS[validation_type]})
    return ordered


def _already_delivered(
    notifications: Sequence[Mapping[str, Any]], run_date_et: date
) -> bool:
    """True when the log holds a delivered daily-run digest for *run_date_et* (D-07 "(update)")."""
    return any(
        record.get("event_class") == "digest"
        and record.get("source") == "daily_run"
        and record.get("outcome") == OUTCOME_DELIVERED
        and record.get("run_date_et") == run_date_et.isoformat()
        for record in notifications
    )


def _data_change_section(
    season: int | None, service: DataService | None, cutoff: datetime | None
) -> dict[str, Any]:
    """Read the three stored sources of the D-03 section, each on its own, and select its lines.

    A source that cannot be read -- a corrupt probe log, an unreadable manifest, an unknown season,
    no cache -- is named as unreadable in the section and never costs the other two. The seams
    ``digest_sections.PROBE_LOG_PATH`` / ``MANIFEST_DIR`` are looked up at call time.
    """
    unreadable: list[str] = []
    probe_line: dict[str, Any] | None = None
    try:
        probe_line = latest_entry(digest_sections.PROBE_LOG_PATH)
    except Exception:  # noqa: BLE001 - named in the section, never read as "no change"
        unreadable.append(digest_sections.UNREADABLE_PROBE_LOG)
    manifest: dict[str, Any] | None = None
    if season is None:
        unreadable.append(digest_sections.UNREADABLE_MANIFEST)
    else:
        try:
            manifest = load_live_manifest(
                season, manifest_dir=digest_sections.MANIFEST_DIR
            )
        except Exception:  # noqa: BLE001 - named in the section, never read as "no change"
            unreadable.append(digest_sections.UNREADABLE_MANIFEST)
    corrections: list[dict[str, Any]] = []
    if service is None:
        unreadable.append(digest_sections.UNREADABLE_CORRECTIONS)
    else:
        corrections = list(service.get_bet_list_corrections(None, None).values())
    return digest_sections.data_change_section(
        probe_line=probe_line,
        manifest=manifest,
        corrections=corrections,
        cutoff=cutoff,
        unreadable=unreadable,
    )


def collect_digest_inputs(
    report: RunReport,
    *,
    schedule: pd.DataFrame | None,
    service: DataService | None,
    notifications: Sequence[Mapping[str, Any]],
    failed_titles: Sequence[str],
    now: datetime,
    label_prefix: str = "",
) -> DigestInputs:
    """Select everything the digest shows. Nothing is computed.

    Args:
        report: What the sending run did.
        schedule: The prepared recorded schedule, or None when it could not be read.
        service: The cache, or None when no cache could be read -- read as not current, with
            every cache-derived part unavailable, never zero.
        notifications: The notification log's records (for the "(update)" mark and the
            previous delivered digest's cutoff).
        failed_titles: The titles of this run's failed messages.
        now: The instant a game's lock is judged against.
        label_prefix: Prepended to the subject and heading.
    """
    season, week = _digest_week(report, schedule, service)
    cutoff = digest_sections.previous_digest_cutoff(notifications, report.run_date_et)
    populated_at: str | None = None
    current = False
    bets: list[dict[str, Any]] = []
    row_game_ids: set[str] = set()
    last_week: dict[str, Any] | None = None
    tracker: list[dict[str, Any]] | None = None
    if service is not None:
        stored = service.get_bet_list_populated_at(season, week)
        populated_at = None if stored is None else str(stored)
        marker = _instant(stored)
        # Without the schedule nothing confirms the list is this week's (D-24): bets unknown.
        current = schedule is not None and marker is not None and marker >= report.start
        if current:
            # The exact call /bets' ranked list makes, so the two rank the same rows (NOTIF-05).
            bets = service.get_bet_list(season, week, arm=ARM_LIVE)
            row_game_ids = set(service.get_bet_game_ids(season, week))
        last_rows = (
            service.get_bet_week_results(season, week - 1)
            if season is not None and week is not None and week > 1
            else []
        )
        last_week = last_week_display(last_rows, season, week)
        tracker = _forward_tracker_blocks(service.get_bet_tracker_blocks())
    games = _game_states(
        report,
        schedule=schedule,
        season=season,
        week=week,
        current=current,
        bet_game_ids={str(row["game_id"]) for row in bets},
        row_game_ids=row_game_ids,
        now=now,
    )
    return DigestInputs(
        run_date_et=report.run_date_et,
        run_start=report.start,
        season=season,
        week=week,
        cache_current=current,
        cache_populated_at=populated_at,
        bets=bets,
        games=games,
        last_week=last_week,
        tracker_blocks=tracker,
        failed_titles=list(failed_titles),
        is_update=_already_delivered(notifications, report.run_date_et),
        label_prefix=label_prefix,
        schedule_readable=schedule is not None,
        data_changes=_data_change_section(season, service, cutoff),
        closing=digest_sections.closing_section(cutoff),
    )


def _week_label(inputs: DigestInputs) -> str:
    if inputs.season is None or inputs.week is None:
        return "week unknown"
    return f"{inputs.season} week {inputs.week}"


def _bets_label(inputs: DigestInputs) -> str:
    """``3 bets`` / ``1 bet`` / ``0 bets`` only when the cache is current, else ``bets unknown``."""
    if not inputs.cache_current:
        return "bets unknown"
    count = len(inputs.bets)
    return f"{count} bet" if count == 1 else f"{count} bets"


def digest_subject(inputs: DigestInputs) -> str:
    """``NFL Predict - <season> week <N> - <bets> - <status>`` (D-07).

    A run that was not clean puts its status first, as the owner-approved example does
    (``... - RUN FAILED - bets unknown``); a same-day re-send ends `` (update)``.
    """
    bets = _bets_label(inputs)
    tail = f"RUN FAILED - {bets}" if inputs.failed_titles else f"{bets} - run clean"
    update = " (update)" if inputs.is_update else ""
    return f"{inputs.label_prefix}NFL Predict - {_week_label(inputs)} - {tail}{update}"


def _status_lines(inputs: DigestInputs) -> list[str]:
    """What broke, one line per failed title without its ``FAILED: `` prefix."""
    return [title.removeprefix(_FAILED_PREFIX) for title in inputs.failed_titles]


def render_digest(inputs: DigestInputs) -> EmailMessage:
    """The digest as one email: Jinja HTML (inline styles) plus its plain-text alternative.

    Raises:
        jinja2.TemplateError: a template could not render (``StrictUndefined``: a typo is loud).
    """
    subject = digest_subject(inputs)
    environment = template_environment()
    context = {
        "inputs": inputs,
        "heading": f"{inputs.label_prefix}NFL Predict - {_week_label(inputs)}",
        "status_lines": _status_lines(inputs),
        "populated_label": et_timestamp_label(inputs.cache_populated_at)
        or "not recorded",
        "stale_notice": STALE_NOTICE,
        "zero_bets_line": ZERO_BETS_LINE,
        "schedule_unreadable_line": SCHEDULE_UNREADABLE_LINE,
    }
    html = environment.get_template(DIGEST_HTML_TEMPLATE).render(**context)
    text = environment.get_template(DIGEST_TEXT_TEMPLATE).render(**context)
    run_start = inputs.run_start.astimezone(UTC).isoformat()
    return EmailMessage(
        event_class="digest",
        subject=subject,
        html=ascii_text(html),
        text=ascii_text(text),
        idempotency_key=f"digest/{inputs.run_date_et.isoformat()}/{run_start}",
    )


def send_digest_if_due(
    report: RunReport,
    *,
    context: SendContext,
    failed_titles: Sequence[str],
    now: datetime,
) -> Delivery | None:
    """Send this week's digest when this run is its digest run; None on any other day.

    The schedule is read through :data:`LOAD_SCHEDULE` and the cache at :data:`WEB_CACHE_PATH`,
    both at call time.

    A DUE DIGEST NEVER VANISHES. Once the run is known to be the digest run, any exception while
    opening the cache, reading the notification log, collecting or rendering becomes ONE recorded
    failed digest (attempts 0, ``build_<ExceptionClass>``, the class name only) returned for the
    caller to escalate. That one handler is the only treatment of a notification-log read failure:
    a malformed log raises ``ValueError`` (Plan 35-01) and is never read as "no earlier digest".
    The cache connection is closed before the handler runs. Only ``Exception`` is caught, so a
    Ctrl+C and the test suite's network guard still propagate. A failed SEND is not an exception:
    :func:`notify.events.send_email` already returns a failed delivery.

    A SCHEDULE THAT CANNOT BE READ (D-24): any ``Exception`` from :data:`LOAD_SCHEDULE` prints
    ``NOTIFY_DIGEST_SCHEDULE= unreadable <ExceptionClass>``. On a Saturday the email is still sent,
    its status naming :data:`SCHEDULE_UNREADABLE_TITLE`; on any other weekday this returns None.
    """
    schedule: pd.DataFrame | None
    try:
        schedule = LOAD_SCHEDULE()
    except Exception as exc:  # noqa: BLE001 - D-24: an unreadable schedule never costs the email
        print(f"NOTIFY_DIGEST_SCHEDULE= unreadable {type(exc).__name__}")  # noqa: T201
        if report.run_date_et.weekday() != _SATURDAY:
            return None
        schedule = None
        failed_titles = (*failed_titles, SCHEDULE_UNREADABLE_TITLE)
    else:
        if digest_week_for_run(schedule, report.run_date_et) is None:
            return None
    try:
        with open_cache_service(WEB_CACHE_PATH) as service:
            notifications = read_notifications()
            inputs = collect_digest_inputs(
                report,
                schedule=schedule,
                service=service,
                notifications=notifications,
                failed_titles=failed_titles,
                now=now,
            )
        message = render_digest(inputs)
    except Exception as exc:  # noqa: BLE001 - a digest that cannot be built is escalated, not lost
        return record_unbuilt_email(f"build_{type(exc).__name__}", context)
    return send_email(message, context)
