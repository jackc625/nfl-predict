"""The Phase 37 cadence records: dates, A/A weeks, bet rules, the attempt lock (D-19, R9, RFIT-05).

Every cadence-date event is one append-only line in ``ledger/refit/dates.jsonl`` with closed
vocabularies, validated the same way on write and on read; A/A results go to
``ledger/refit/aa_weeks.jsonl``; each generation's bet rule is a content-addressed, write-once JSON
under ``ledger/refit/bet_rules/``. Install-critical writes are fsynced. Every file here lives under
``tmp_path``: no test writes a production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from data.write_sink import RecordingSink, active_sink
from refit.records import (
    AA_WEEKS_FILENAME,
    ATTEMPT_OUTCOMES,
    BET_RULES_DIRNAME,
    CLOSED_STATUSES,
    DATE_EVENTS,
    DATES_FILENAME,
    REFIT_DIRNAME,
    RefitRecordFormatError,
    append_aa_record,
    append_date_event,
    attempt_lock,
    bet_rule_record_path,
    bet_rule_relpath,
    code_commit,
    read_aa_records,
    read_bet_rule_record,
    read_date_events,
    write_bet_rule_record,
)
from utils.file_lock import FileLockHeldError

_SHA_A = "a" * 40
_BET_RULE: dict[str, Any] = {"generation": 1, "thresholds": {"wp": 0.03, "ats": 0.02}}


def _common(date_index: int = 1, date: str = "2026-11-03") -> dict[str, Any]:
    return {
        "event": "attempt",
        "list_week": 8,
        "date_index": date_index,
        "date": date,
        "nfl_week": 9,
    }


def _attempt(outcome: str, **overrides: Any) -> dict[str, Any]:
    event = {
        **_common(),
        "cutoff_utc": "2026-11-03T22:00:00+00:00",
        "attempted_at_utc": "2026-11-03T22:01:00+00:00",
        "run_date_et": "2026-11-03",
        "code_commit": _SHA_A,
        "outcome": outcome,
        "check": "bundle_validation" if outcome == "refused" else None,
        "exception_class": None,
        "before_ids": {},
        "after_ids": {},
        "no_final_score_ids": ["2026_09_AAA_BBB"] if outcome != "refused" else None,
    }
    event.update(overrides)
    return event


def _closed(status: str) -> dict[str, Any]:
    return {
        **_common(),
        "event": "closed",
        "status": status,
        "closed_at_utc": "2026-11-03T22:07:00+00:00",
        "before_ids": {},
        "after_ids": {},
    }


def _intent(sha256: str, bet_rule_path: str, **overrides: Any) -> dict[str, Any]:
    event = {
        **_common(),
        "event": "install_intent",
        "ids": {
            "wp": "wp_x",
            "ats": "ats_x",
            "ou": "ou_x",
            "blend": "b_x",
            "converter": "c_x",
        },
        "bet_rule_path": bet_rule_path,
        "bet_rule_sha256": sha256,
        "code_commit": _SHA_A,
    }
    event.update(overrides)
    return event


def _dates_path(ledger: Path) -> Path:
    return ledger / REFIT_DIRNAME / DATES_FILENAME


def test_date_events_round_trip_in_time_order(tmp_path: Path) -> None:
    events = [_attempt("refused"), _attempt("installed"), _closed("ran")]
    for event in events:
        assert append_date_event(tmp_path, event) is True

    read = read_date_events(tmp_path)
    assert len(read) == 3
    for written, stored in zip(events, read, strict=True):
        assert {key: stored[key] for key in written} == written
    stamps = [stored["recorded_at_utc"] for stored in read]
    assert stamps == sorted(stamps)


@pytest.mark.parametrize(
    ("event", "bad_value", "vocabulary"),
    [
        ({**_attempt("installed"), "event": "retry"}, "retry", DATE_EVENTS),
        (_attempt("skipped"), "skipped", ATTEMPT_OUTCOMES),
        (_closed("ran early"), "ran early", CLOSED_STATUSES),
    ],
)
def test_unknown_event_outcome_or_status_is_refused(
    tmp_path: Path, event: dict[str, Any], bad_value: str, vocabulary: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="vocabulary") as raised:
        append_date_event(tmp_path, event)
    assert repr(bad_value) in str(raised.value)
    assert str(vocabulary) in str(raised.value)
    assert not _dates_path(tmp_path).exists()


def test_aa_records_round_trip(tmp_path: Path) -> None:
    records = [
        {"season": 2026, "week": 8, "clean": False},
        {"season": 2026, "week": 9, "clean": True},
    ]
    for record in records:
        assert append_aa_record(tmp_path, record) is True

    read = read_aa_records(tmp_path)
    assert [
        {key: stored[key] for key in written}
        for written, stored in zip(records, read, strict=True)
    ] == records
    assert all("recorded_at_utc" in stored for stored in read)
    assert (tmp_path / REFIT_DIRNAME / AA_WEEKS_FILENAME).exists()


def test_bet_rule_record_is_content_addressed_and_write_once(tmp_path: Path) -> None:
    path, sha = write_bet_rule_record(tmp_path, _BET_RULE)
    assert path == bet_rule_record_path(tmp_path, sha)
    assert path.name == f"{sha}.json"
    assert path.parent == tmp_path / REFIT_DIRNAME / BET_RULES_DIRNAME
    assert read_bet_rule_record(path) == _BET_RULE

    before = path.stat().st_mtime_ns
    again_path, again_sha = write_bet_rule_record(
        tmp_path, dict(reversed(_BET_RULE.items()))
    )
    assert (again_path, again_sha) == (path, sha)
    assert path.stat().st_mtime_ns == before

    path.write_text(json.dumps({"generation": 2}), encoding="utf-8")
    with pytest.raises(RefitRecordFormatError, match=sha):
        read_bet_rule_record(path)
    with pytest.raises(RefitRecordFormatError, match=sha):
        write_bet_rule_record(tmp_path, _BET_RULE)


def test_install_intent_stores_the_one_bet_rule_path_convention(tmp_path: Path) -> None:
    written_path, sha = write_bet_rule_record(tmp_path, _BET_RULE)
    other_sha = "f" * 64
    relpath = bet_rule_relpath(sha)
    assert relpath == f"refit/bet_rules/{sha}.json"

    wrong_paths = [
        str(written_path),
        relpath.replace("/", "\\"),
        bet_rule_relpath(other_sha),
    ]
    for wrong in wrong_paths:
        with pytest.raises(ValueError) as raised:
            append_date_event(tmp_path, _intent(sha, wrong))
        assert repr(wrong) in str(raised.value)
        assert repr(relpath) in str(raised.value)
    assert not _dates_path(tmp_path).exists()

    no_commit = _intent(sha, relpath)
    del no_commit["code_commit"]
    with pytest.raises(ValueError, match="code_commit"):
        append_date_event(tmp_path, no_commit)
    with pytest.raises(ValueError, match="'abcdef1'"):
        append_date_event(tmp_path, _intent(sha, relpath, code_commit="abcdef1"))
    assert not _dates_path(tmp_path).exists()

    dirty = "c" * 40 + "+dirty"
    assert append_date_event(tmp_path, _intent(sha, relpath, code_commit=dirty)) is True
    stored = read_date_events(tmp_path)[-1]
    assert stored["code_commit"] == dirty
    assert bet_rule_record_path(tmp_path, stored["bet_rule_sha256"]) == written_path


def test_a_second_attempt_lock_is_refused(tmp_path: Path) -> None:
    # The first lock is held while the second is tried; the second is refused at once.
    with (
        attempt_lock(tmp_path),
        pytest.raises(FileLockHeldError),
        attempt_lock(tmp_path),
    ):
        pass
    with attempt_lock(tmp_path):
        pass


def test_a_malformed_dates_line_is_never_read_as_empty(tmp_path: Path) -> None:
    append_date_event(tmp_path, _attempt("refused"))
    path = _dates_path(tmp_path)
    good = path.read_text(encoding="utf-8")

    path.write_text(good + '{"event": "attem', encoding="utf-8")
    with pytest.raises(RefitRecordFormatError) as raised:
        read_date_events(tmp_path)
    message = str(raised.value)
    assert DATES_FILENAME in message
    assert "line 2" in message
    assert "copy the file aside, then delete that one line" in message

    path.write_text("not json\n" + good, encoding="utf-8")
    with pytest.raises(RefitRecordFormatError) as raised:
        read_date_events(tmp_path)
    message = str(raised.value)
    assert DATES_FILENAME in message
    assert "line 1" in message
    assert "compare the file with the private backup" in message
    assert "delete that one line" not in message


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ({**_attempt("installed"), "event": "retry"}, "event"),
        (
            {
                key: value
                for key, value in _attempt("installed").items()
                if key != "outcome"
            },
            "outcome",
        ),
        (_intent("e" * 64, "refit\\bet_rules\\" + "e" * 64 + ".json"), "bet_rule_path"),
    ],
)
def test_a_well_formed_but_invalid_dates_line_is_refused_on_read(
    tmp_path: Path, line: dict[str, Any], field: str
) -> None:
    path = _dates_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(RefitRecordFormatError) as raised:
        read_date_events(tmp_path)
    message = str(raised.value)
    assert DATES_FILENAME in message
    assert "line 1" in message
    assert field in message


def test_a_well_formed_but_invalid_aa_line_is_refused_on_read(tmp_path: Path) -> None:
    path = tmp_path / REFIT_DIRNAME / AA_WEEKS_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"season": 2026, "week": 8}) + "\n", encoding="utf-8")
    with pytest.raises(RefitRecordFormatError) as raised:
        read_aa_records(tmp_path)
    message = str(raised.value)
    assert AA_WEEKS_FILENAME in message
    assert "line 1" in message
    assert "clean" in message


def test_install_critical_writes_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)

    append_date_event(tmp_path, _attempt("refused"))
    assert calls == []

    _, sha = write_bet_rule_record(tmp_path, _BET_RULE)
    assert len(calls) == 1

    append_date_event(tmp_path, _intent(sha, bet_rule_relpath(sha)))
    assert len(calls) == 2

    before = _dates_path(tmp_path).read_bytes()
    with active_sink(RecordingSink()) as sink:
        assert append_date_event(tmp_path, _intent(sha, bet_rule_relpath(sha))) is False
    assert _dates_path(tmp_path).read_bytes() == before
    assert len(sink.intended_writes) == 1
    assert len(calls) == 2


@pytest.mark.usefixtures("isolated_git_env")
def test_code_commit_reports_head_and_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    git("init", "-q")
    (tmp_path / "refit").mkdir()
    module = tmp_path / "refit" / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("outside every code path\n", encoding="utf-8")
    git("add", "refit/module.py")
    git("commit", "-q", "-m", "initial")
    head = git("rev-parse", "HEAD")

    assert code_commit(tmp_path) == head

    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert code_commit(tmp_path) == f"{head}+dirty"

    assert code_commit(tmp_path / "not_a_repo_child_missing") is None
