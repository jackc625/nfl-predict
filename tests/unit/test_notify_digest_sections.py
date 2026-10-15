"""The weekly email's data-change section (Phase 35 D-03, D-21, NOTIF-05).

* The sealed zone lists the LATEST probe line's unacknowledged findings when that line ranks at
  WARNING or louder, in plain words (dataset, season, what moved); acknowledged and informational
  findings are never listed.
* The live zone lists the manifest's WARNING-or-louder verdicts recorded since the previous
  delivered digest (all of them before the first digest), each read from its CAPTURE's instant.
* ``fixed by ledger correction`` is claimed only for a correction that provably belongs to the
  verdict: same season, a scope week, and ``detected_at_utc`` equal to the capture's instant.
* Corrected grades since the previous digest are listed from the cache's corrections.
* Nothing to list is one line; the section says runs are unaffected and names the acknowledge
  command; severity is ranked, never compared as text.
* SPEC R4 item 7 / D-23: the ledger run log's skipped, failed or missed closing readings, first-pick
  refusals and mismatched closing-schedule read-backs since the previous digest are listed before
  the data-change section; an unreadable run log is named with its line number.

Every input is a ``tmp_path`` file or a dict: the real probe log, live manifest and web cache are
never read, and every request goes through an ``httpx.MockTransport``.

Run this module:  uv run python -m pytest tests/unit/test_notify_digest_sections.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from api.cache import BET_LIST_CORRECTIONS_COLUMNS, materialize_bet_list_corrections
from api.presentation import NOT_ADVICE_LINE
from data.sealed_probe_log import latest_entry
from forward_ledger import run_log
from notify import digest, digest_sections
from tests.unit.test_notify_channels import email_secrets
from tests.unit.test_notify_digest import (
    WEEK6,
    build_cache,
    install_router,
    saturday_report,
    schedule_frame,
    week6_bets,
)
from tests.unit.test_notify_transport import (
    RUN_DATE,
    daily_context,
    install_sleep_recorder,
)

SCHEDULES_2025_LINE = "schedules 2025: the upstream file changed after it was pinned (content digest moved)"
GRADED_WEEK5_LINE = "pbp 2026 week(s) 5: upstream restated already-graded data"
FIXED = " -- fixed by ledger correction"

FIRST_CAPTURE_AT = "2026-10-05T21:00:14.410510+00:00"
SECOND_CAPTURE_AT = "2026-10-12T21:00:09.120000+00:00"
UNRELATED_AT = "2026-10-08T21:00:00+00:00"


# ---------------------------------------------------------------------------
# Stored shapes: a probe line, a live manifest and a cache correction row
# ---------------------------------------------------------------------------


def finding(
    dataset: str = "schedules",
    season: int = 2025,
    *,
    strategy: str = "content",
    acknowledged: bool = False,
) -> dict[str, Any]:
    """One probe finding, in the shape ``data.sealed_probe`` writes it."""
    return {
        "acknowledged": acknowledged,
        "baseline_sha256": "07de247b",
        "dataset": dataset,
        "event_class": "known_divergence_stable" if acknowledged else "sealed_revision",
        "observed_sha256": "0d8a27c2",
        "ruled_at_utc": None,
        "ruled_by": None,
        "season": season,
        "strategy": strategy,
        "superseded_acknowledgement": False,
    }


def probe_line(
    *findings: dict[str, Any],
    severity: str = "critical",
    event_class: str = "sealed_revision",
) -> dict[str, Any]:
    """One probe log line, in the real shape (the 2026-10-14 CRITICAL schedules line)."""
    return {
        "checked": 76,
        "event_class": event_class,
        "expected": 76,
        "findings": list(findings),
        "probed_at_utc": "2026-10-14T21:00:15.103188+00:00",
        "season": 2026,
        "severity": severity,
        "unresolved": [],
        "verdict_schema_version": 1,
    }


def write_probe_log(path: Path, *lines: dict[str, Any]) -> Path:
    """A tmp probe log holding *lines*, oldest first, one JSON object per line."""
    path.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
        encoding="utf-8",
    )
    return path


def verdict(
    event_class: str,
    severity: str,
    *,
    dataset: str = "pbp",
    weeks: tuple[int, ...] = (),
) -> dict[str, Any]:
    """A live capture's verdict block; a graded one owes a correction for *weeks*."""
    owed = event_class == "live_revision_graded"
    return {
        "dataset": dataset,
        "season": 2026,
        "event_class": event_class,
        "severity": severity,
        "correction_owed": owed,
        "correction_owed_scope": (
            {"season": 2026, "weeks": list(weeks), "dataset": dataset} if owed else None
        ),
    }


def capture(
    sequence: int, captured_at: str, block: dict[str, Any] | None, *, week: int = 5
) -> dict[str, Any]:
    """A manifest capture entry; its verdict rides under ``CAPTURE_VERDICT_KEY`` with no instant."""
    entry: dict[str, Any] = {
        "week": week,
        "sequence": sequence,
        "captured_at_utc": captured_at,
    }
    if block is not None:
        entry["revision"] = block
    return entry


def manifest(**captures: list[dict[str, Any]]) -> dict[str, Any]:
    """A live manifest for 2026: ``dataset=[captures]``."""
    return {
        "schema_version": 1,
        "season": 2026,
        "zone": "live",
        "datasets": {name: {"captures": rows} for name, rows in captures.items()},
    }


def correction(
    *,
    week: int = 5,
    detected_at: str = FIRST_CAPTURE_AT,
    corrected_at: str = "2026-10-14T21:30:00+00:00",
    game_id: str = "2026_05_BUF@LA",
    target: str = "ats",
    arm: str | None = "live",
) -> dict[str, Any]:
    """A cache ``bet_list_corrections`` row."""
    return {
        "game_id": game_id,
        "season": 2026,
        "week": week,
        "target": target,
        "arm": arm,
        "original_grading_status": "win",
        "corrected_grading_status": "loss",
        "corrected_outcome": False,
        "corrected_payout_flat": -1.0,
        "corrected_realized_units": -1.0,
        "realized_value": -2.5,
        "detected_at_utc": detected_at,
        "corrected_at_utc": corrected_at,
    }


def graded_manifest() -> dict[str, Any]:
    """One CRITICAL graded pbp verdict on 2026 week 5, captured at :data:`FIRST_CAPTURE_AT`."""
    return manifest(
        pbp=[
            capture(
                7,
                FIRST_CAPTURE_AT,
                verdict("live_revision_graded", "critical", weeks=(5,)),
            )
        ]
    )


def render_text_and_html(section: dict[str, Any]) -> tuple[str, str]:
    """The digest's text and HTML parts around *section* (no cache, no schedule needed)."""
    start = datetime(2026, 10, 17, 21, 0, tzinfo=UTC)
    inputs = digest.DigestInputs(
        run_date_et=RUN_DATE,
        run_start=start,
        season=2026,
        week=6,
        cache_current=False,
        cache_populated_at=None,
        bets=[],
        games=[],
        last_week=None,
        tracker_blocks=None,
        failed_titles=[],
        is_update=False,
        data_changes=section,
    )
    message = digest.render_digest(inputs)
    return message.text, message.html


def text_section(text: str) -> list[str]:
    """The text part's data-change section, heading excluded, up to its blank line."""
    body = text.split("UPSTREAM DATA CHANGES AND CORRECTED GRADES\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


# ---------------------------------------------------------------------------
# Task 1 (tracer): stored verdicts and corrections -> the email's data-change section
# ---------------------------------------------------------------------------


@pytest.fixture
def section_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Week 6 behind ``LOAD_SCHEDULE``; tmp cache, probe log and manifest directory; no sleeps."""
    install_sleep_recorder(monkeypatch)
    schedule = schedule_frame(WEEK6)
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: schedule)
    paths = {
        "cache": tmp_path / "web_cache.duckdb",
        "probe_log": tmp_path / "upstream_probe_log.jsonl",
        "manifest_dir": tmp_path / "upstream_live",
    }
    paths["manifest_dir"].mkdir()
    monkeypatch.setattr(digest, "WEB_CACHE_PATH", paths["cache"])
    monkeypatch.setattr(digest_sections, "PROBE_LOG_PATH", paths["probe_log"])
    monkeypatch.setattr(digest_sections, "MANIFEST_DIR", paths["manifest_dir"])
    return paths


def test_the_section_names_the_sealed_change_in_plain_words(
    monkeypatch: pytest.MonkeyPatch, section_env: dict[str, Path]
) -> None:
    """Tracer: probe log + live manifest + cache corrections -> templates -> the sent email."""
    write_probe_log(
        section_env["probe_log"],
        probe_line(severity="informational", event_class="clean"),
        probe_line(finding()),
    )
    (section_env["manifest_dir"] / "2026.json").write_text(
        json.dumps(graded_manifest()), encoding="utf-8"
    )
    report = saturday_report()
    build_cache(
        section_env["cache"],
        bets=week6_bets(),
        stamped_at=report.start + timedelta(minutes=12),
    )
    conn = duckdb.connect(str(section_env["cache"]))
    try:
        row = correction()
        assert list(row) == BET_LIST_CORRECTIONS_COLUMNS
        materialize_bet_list_corrections(conn, pd.DataFrame([row]))
    finally:
        conn.close()
    requests = install_router(monkeypatch)

    delivery = digest.send_digest_if_due(
        report,
        context=daily_context(email_secrets()),
        failed_titles=(),
        now=report.start + timedelta(minutes=20),
    )

    assert delivery is not None and delivery.delivered
    (request,) = requests["api.resend.com"]
    sent = json.loads(request.content)
    text, raw_html = sent["text"], sent["html"]
    readable_html = html.unescape(raw_html)
    expected = [
        SCHEDULES_2025_LINE,
        GRADED_WEEK5_LINE + FIXED,
        "BUF @ LA Spread: graded win, corrected to loss",
    ]
    assert text_section(text)[: len(expected)] == [f"- {line}" for line in expected]
    for line in expected:
        assert f"<li data-digest-data-change>{line}</li>" in readable_html
    for part in (text, readable_html):
        assert digest_sections.ACKNOWLEDGE_COMMAND in part
    # After the season-to-date block (the closing section's position), before the not-advice line.
    assert (
        text.index("SEASON TO DATE")
        < text.index("UPSTREAM DATA CHANGES AND CORRECTED GRADES")
        < text.index(NOT_ADVICE_LINE)
    )
    assert (
        raw_html.index(">Season to date<")
        < raw_html.index("data-digest-data-changes")
        < raw_html.index("data-not-advice")
    )
    assert text.rstrip().endswith(NOT_ADVICE_LINE)


def test_only_the_latest_probe_line_counts(tmp_path: Path) -> None:
    path = write_probe_log(
        tmp_path / "probe.jsonl",
        probe_line(finding("pbp", 2024, strategy="metadata"), finding()),
        probe_line(finding()),
    )

    assert digest_sections.sealed_changes(latest_entry(path)) == [SCHEDULES_2025_LINE]


def test_acknowledged_and_informational_findings_are_not_listed() -> None:
    critical = probe_line(
        finding("pbp", 2024, acknowledged=True),
        finding("depth_charts", 2023, strategy="metadata"),
    )
    quiet = probe_line(
        finding(acknowledged=True),
        severity="informational",
        event_class="known_divergence_stable",
    )

    assert digest_sections.sealed_changes(critical) == [
        "depth_charts 2023: the upstream file changed after it was pinned "
        "(updated time or size moved)"
    ]
    assert digest_sections.sealed_changes(quiet) == []
    assert digest_sections.sealed_changes(None) == []


def test_live_changes_since_the_previous_digest_are_listed_once() -> None:
    live = manifest(
        depth_charts=[
            capture(
                2,
                "2026-10-06T12:00:00+00:00",
                verdict("unknown", "warning", dataset="depth_charts"),
            )
        ],
        pbp=[
            capture(
                8,
                SECOND_CAPTURE_AT,
                verdict("live_revision_graded", "critical", weeks=(5,)),
            )
        ],
    )
    previous = {
        "event_class": "digest",
        "source": "daily_run",
        "outcome": "delivered",
        "run_date_et": "2026-10-10",
        "sent_at": "2026-10-10T21:20:00+00:00",
    }
    ignored = [
        # A failed digest, a same-day send and a test send never move the cutoff.
        {**previous, "outcome": "failed", "sent_at": "2026-10-13T21:00:00+00:00"},
        {
            **previous,
            "run_date_et": "2026-10-17",
            "sent_at": "2026-10-17T21:30:00+00:00",
        },
        {**previous, "source": "test_command", "sent_at": "2026-10-16T09:00:00+00:00"},
    ]
    cutoff = digest_sections.previous_digest_cutoff([previous, *ignored], RUN_DATE)

    assert cutoff == datetime(2026, 10, 10, 21, 20, tzinfo=UTC)
    assert digest_sections.live_changes(live, cutoff, []) == [GRADED_WEEK5_LINE]
    assert digest_sections.previous_digest_cutoff([], RUN_DATE) is None
    assert digest_sections.live_changes(live, None, []) == [
        "depth_charts 2026: a change that could not be compared (not ruled on)",
        GRADED_WEEK5_LINE,
    ]


def test_a_discharged_live_change_reads_fixed_by_ledger_correction() -> None:
    lines = digest_sections.live_changes(graded_manifest(), None, [correction()])

    assert lines == [GRADED_WEEK5_LINE + FIXED]
    assert lines[0].endswith("fixed by ledger correction")


def test_only_the_correction_that_belongs_to_a_verdict_marks_it() -> None:
    """Review 35-14: an unrelated correction in the same week never marks a verdict fixed."""
    two_verdicts = manifest(
        pbp=[
            capture(
                7,
                FIRST_CAPTURE_AT,
                verdict("live_revision_graded", "critical", weeks=(5,)),
            ),
            capture(
                8,
                SECOND_CAPTURE_AT,
                verdict("live_revision_graded", "critical", weeks=(5,)),
            ),
        ]
    )
    # The same instant written with another offset is still the second capture's instant.
    second_instant = "2026-10-12T17:00:09.120000-04:00"
    belongs = correction(detected_at=second_instant)
    unrelated = correction(detected_at=UNRELATED_AT, game_id="2026_05_TB@DAL")

    assert digest_sections.live_changes(two_verdicts, None, [belongs, unrelated]) == [
        GRADED_WEEK5_LINE,
        GRADED_WEEK5_LINE + FIXED,
    ]
    assert digest_sections.live_changes(two_verdicts, None, [unrelated]) == [
        GRADED_WEEK5_LINE,
        GRADED_WEEK5_LINE,
    ]
    # Same instant, but a week outside the verdict's scope: not this verdict's correction.
    other_week = correction(week=4, detected_at=SECOND_CAPTURE_AT)
    assert digest_sections.live_changes(two_verdicts, None, [other_week]) == [
        GRADED_WEEK5_LINE,
        GRADED_WEEK5_LINE,
    ]


def test_corrected_grades_since_the_cutoff_are_listed() -> None:
    cutoff = datetime(2026, 10, 10, 21, 20, tzinfo=UTC)
    rows = [
        correction(corrected_at="2026-10-08T21:30:00+00:00", game_id="2026_04_KC@DEN"),
        correction(target="ou"),
        correction(target="wp", arm="shadow"),  # the email is live-arm only
    ]

    assert digest_sections.corrected_grades(rows, cutoff) == [
        "BUF @ LA Totals: graded win, corrected to loss"
    ]
    assert len(digest_sections.corrected_grades(rows, None)) == 2


def test_nothing_to_list_is_one_line() -> None:
    section = digest_sections.data_change_section(
        probe_line=probe_line(severity="informational", event_class="clean"),
        manifest=manifest(
            pbp=[capture(3, FIRST_CAPTURE_AT, verdict("clean", "informational"))]
        ),
        corrections=[],
        cutoff=None,
    )

    text, raw_html = render_text_and_html(section)

    assert section["lines"] == []
    assert text_section(text)[:1] == [
        "No unacknowledged upstream data change and no corrected grade since the last digest."
    ]
    assert text_section(text)[1] == digest_sections.UNAFFECTED_LINE
    assert "data-digest-data-change>" not in raw_html
    assert raw_html.count(digest_sections.DATA_CHANGE_EMPTY_LINE) == 1

    # A source that could not be read is named, never shown as "nothing".
    unreadable = digest_sections.data_change_section(
        probe_line=None,
        manifest=None,
        corrections=[],
        cutoff=None,
        unreadable=[digest_sections.UNREADABLE_PROBE_LOG],
    )
    text, _raw_html = render_text_and_html(unreadable)
    assert text_section(text)[0] == f"- {digest_sections.UNREADABLE_PROBE_LOG}"
    assert digest_sections.DATA_CHANGE_EMPTY_LINE not in text


def test_runs_are_said_to_be_unaffected() -> None:
    section = digest_sections.data_change_section(
        probe_line=probe_line(finding()), manifest=None, corrections=[], cutoff=None
    )

    text, raw_html = render_text_and_html(section)

    assert section["unaffected_line"] == (
        "Runs are unaffected: every run reads pinned bytes."
    )
    assert section["unaffected_line"] in text
    assert section["unaffected_line"] in raw_html


def test_severity_is_ranked_not_compared_as_text() -> None:
    """Alphabetically "critical" < "informational" < "warning"; the section ranks instead."""
    loudness = ["informational", "warning", "critical"]
    assert sorted(loudness) == ["critical", "informational", "warning"]
    live = manifest(
        pbp=[
            capture(4, FIRST_CAPTURE_AT, verdict("live_revision", "informational")),
            capture(5, SECOND_CAPTURE_AT, verdict("unknown", "warning")),
        ]
    )

    assert digest_sections.live_changes(live, None, []) == [
        "pbp 2026: a change that could not be compared (not ruled on)"
    ]
    assert digest_sections.sealed_changes(
        probe_line(finding(), severity="warning", event_class="unknown")
    ) == [SCHEDULES_2025_LINE]
    assert (
        digest_sections.sealed_changes(probe_line(finding(), severity="informational"))
        == []
    )


# ---------------------------------------------------------------------------
# Task 2: the week's closing gaps, first-pick refusals and read-back failures (R4 item 7, D-23)
# ---------------------------------------------------------------------------

CLOSING_CUTOFF = datetime(2026, 10, 10, 21, 20, tzinfo=UTC)
BUF_LA_ATS = ["2026_05_BUF@LA", 2026, 5, "ats", "live"]


def event(kind: str, at_utc: str = "2026-10-12T20:05:00+00:00", **fields: Any) -> dict:
    """One ledger run-log record, as ``forward_ledger.run_log.record_event`` writes it."""
    return {"event": kind, "at_utc": at_utc, "run_id": None, **fields}


def test_skipped_and_failed_readings_since_the_cutoff_are_listed() -> None:
    events = [
        event(
            "closing_capture",
            outcome="skipped",
            reason="credit_reserve",
            game_ids=["2026_05_TB@DAL", "2026_05_BUF@LA"],
            remaining_credits=40,
        ),
        event(
            "closing_capture",
            outcome="failed",
            reason="capture_failed",
            game_ids=["2026_05_KC@JAX"],
            error_type="RuntimeError",
        ),
        event("closing_capture", outcome="captured", game_ids=["2026_05_SF@LAR"]),
        event("closing_capture", outcome="no_games", game_ids=[]),
    ]

    assert digest_sections.closing_section_lines(events, CLOSING_CUTOFF) == [
        "closing reading skipped (credit_reserve): TB @ DAL, BUF @ LA",
        "closing reading failed (capture_failed): KC @ JAX",
    ]


def test_a_missed_reading_is_listed_from_the_finalized_closing_reason() -> None:
    events = [
        event(
            "closing_finalized",
            key=BUF_LA_ATS,
            closing_null_reason="capture_missed",
            forward_clv=None,
        ),
        event(
            "closing_finalized",
            key=["2026_05_TB@DAL", 2026, 5, "ou", "live"],
            closing_null_reason=None,
            forward_clv=-1.0,
        ),
    ]

    assert digest_sections.closing_section_lines(events, None) == [
        "BUF @ LA Spread: no closing reading (capture_missed)"
    ]


def test_first_pick_refusals_are_listed() -> None:
    events = [
        event(
            "first_pick_refusal",
            key=["2026_06_CIN@CLE", 2026, 6, "wp", "live"],
            fields=["bet_side", "per_bet_ev"],
        )
    ]

    assert digest_sections.closing_section_lines(events, None) == [
        "CIN @ CLE Winner: a later pick was refused, the first pick stands "
        "(differing bet_side, per_bet_ev)"
    ]


def test_a_mismatched_closing_schedule_is_listed() -> None:
    """D-23: a registration that read back wrong is listed; a clean one is not."""
    events = [
        event(
            "closing_triggers",
            outcome="registered",
            reason=None,
            readback_match=False,
            trigger_count=9,
        ),
        event(
            "closing_triggers",
            outcome="registered",
            reason=None,
            readback_match=True,
            trigger_count=9,
        ),
    ]

    assert digest_sections.closing_section_lines(events, None) == [
        "the closing-line schedule read back wrong"
    ]


def test_events_before_the_cutoff_are_not_listed() -> None:
    before = "2026-10-10T20:05:00+00:00"
    events = [
        event(
            "closing_capture",
            before,
            outcome="failed",
            reason="capture_failed",
            game_ids=["2026_05_KC@JAX"],
        ),
        event(
            "closing_finalized",
            before,
            key=BUF_LA_ATS,
            closing_null_reason="capture_missed",
        ),
        event(
            "closing_triggers",
            before,
            outcome="registered",
            readback_match=False,
        ),
    ]

    assert digest_sections.closing_section_lines(events, CLOSING_CUTOFF) == []
    assert len(digest_sections.closing_section_lines(events, None)) == 3


def test_an_unreadable_run_log_is_named() -> None:
    log_path = run_log.LEDGER_RUN_LOG  # the suite's per-test quarantined run log
    log_path.write_text(
        json.dumps(event("closing_capture", outcome="no_games", game_ids=[]))
        + "\nthis line is not JSON\n",
        encoding="utf-8",
    )

    section = digest_sections.closing_section(None)

    (line,) = section["lines"]
    assert line == (
        f"The ledger run log {log_path.as_posix()} could not be read at line 2, so missed "
        "closing readings and first-pick refusals are not listed."
    )
    assert section["empty_line"] not in line


def test_nothing_to_list_in_the_closing_section_is_one_line(
    monkeypatch: pytest.MonkeyPatch, section_env: dict[str, Path]
) -> None:
    """An empty week's closing section is one line, between the tracker totals and the data
    changes, both of which come before the not-advice line."""
    run_log.record_event("closing_capture", outcome="no_games", game_ids=[])
    report = saturday_report()
    build_cache(
        section_env["cache"],
        bets=week6_bets(),
        stamped_at=report.start + timedelta(minutes=12),
    )
    requests = install_router(monkeypatch)

    delivery = digest.send_digest_if_due(
        report,
        context=daily_context(email_secrets()),
        failed_titles=(),
        now=report.start + timedelta(minutes=20),
    )

    assert delivery is not None and delivery.delivered
    sent = json.loads(requests["api.resend.com"][0].content)
    text, raw_html = sent["text"], sent["html"]
    body = text.split("CLOSING READINGS AND FIRST-PICK REFUSALS\n", 1)[1]
    assert body.split("\n\n", 1)[0].splitlines() == [
        "No skipped or missed closing reading and no first-pick refusal since the last digest."
    ]
    assert "data-digest-closing-line>" not in raw_html
    assert raw_html.count(digest_sections.CLOSING_EMPTY_LINE) == 1
    assert (
        text.index("Return (flat, units)")
        < text.index("CLOSING READINGS AND FIRST-PICK REFUSALS")
        < text.index("UPSTREAM DATA CHANGES AND CORRECTED GRADES")
        < text.index(NOT_ADVICE_LINE)
    )
    assert (
        raw_html.rindex("data-digest-tracker-block")
        < raw_html.index("data-digest-closing")
        < raw_html.index("data-digest-data-changes")
        < raw_html.index("data-not-advice")
    )
