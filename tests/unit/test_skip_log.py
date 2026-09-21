"""The durable, append-only skip record (D33.2-05 live half, Plan 33.2-03 Task 1).

WHAT THIS MODULE PROVES
-----------------------
``pipeline/skip_log.py`` is the one place a skipped game leaves a trace that outlives the run.
``logs/friday_pipeline.json`` is rewritten by every run, so "we skipped a game" recorded only
there survives until the next night. This module proves the record's four properties:

* it REFUSES an incomplete entry, naming every missing key, and writes nothing;
* it is IDEMPOTENT on its natural key ``(run_date_et, game_id, source, reason)`` and never on
  ``run_id`` -- keying on ``run_id`` would make every re-run a new record by construction;
* it is APPEND-ONLY: a second append leaves the first line's bytes untouched;
* an unreadable line BLOCKS the read rather than being skipped.

Every file lives under ``tmp_path``; nothing here touches ``config/``.

Run this module:  uv run pytest tests/unit/test_skip_log.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline import skip_log
from pipeline.skip_log import (
    REQUIRED_ENTRY_KEYS,
    SKIP_REASONS,
    SkipRecordCorrupt,
    append_skip_record,
    read_skip_records,
    skip_records_for_run_date,
)


def _entry(**overrides: Any) -> dict[str, Any]:
    """One complete skip record; each test overrides only the field it is about."""
    entry: dict[str, Any] = {
        "run_id": "2026-09-19T17:30:00-04:00",
        "run_date_et": "2026-09-19",
        "game_id": "2026_W03_KC@BUF",
        "source": "elo",
        "information_time": "2026-09-19T22:41:00+00:00",
        "lock": "2026-09-19T22:00:00+00:00",
        "reason": "post_lock",
        "recorded_at": "2026-09-19T21:35:12+00:00",
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "config" / "skip_records.jsonl"


class TestTheVocabulary:
    def test_the_eight_required_keys_are_declared(self) -> None:
        assert REQUIRED_ENTRY_KEYS == (
            "run_id",
            "run_date_et",
            "game_id",
            "source",
            "information_time",
            "lock",
            "reason",
            "recorded_at",
        )

    def test_the_reasons_are_a_closed_three_member_set(self) -> None:
        assert SKIP_REASONS == ("post_lock", "undated", "no_provenance")

    def test_the_refusal_escapes_every_quiet_catch_tuple(self) -> None:
        """Several call sites catch RuntimeError/ValueError/ImportError and carry on."""
        assert SkipRecordCorrupt.__bases__ == (Exception,)
        assert not issubclass(
            SkipRecordCorrupt, (RuntimeError, ValueError, ImportError)
        )

    def test_the_default_store_lives_under_config(self) -> None:
        assert Path("config/skip_records.jsonl") == skip_log.SKIP_RECORD_PATH


class TestAppending:
    def test_one_entry_writes_exactly_one_json_line(self, log_path: Path) -> None:
        assert append_skip_record(_entry(), path=log_path) is True

        lines = log_path.read_bytes().split(b"\n")
        assert lines[-1] == b"", "the file must end in exactly one newline"
        assert len(lines) == 2
        assert json.loads(lines[0]) == _entry()

    def test_the_file_is_created_on_first_append_not_seeded(
        self, log_path: Path
    ) -> None:
        assert not log_path.exists()
        append_skip_record(_entry(), path=log_path)
        assert read_skip_records(log_path) == [_entry()]

    def test_lines_are_lf_terminated_and_key_sorted(self, log_path: Path) -> None:
        """A committed file must not acquire CRLF on Windows, and bytes follow content."""
        append_skip_record(_entry(), path=log_path)
        raw = log_path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.decode("utf-8") == json.dumps(_entry(), sort_keys=True) + "\n"

    def test_a_second_append_leaves_the_first_line_byte_identical(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(), path=log_path)
        first = log_path.read_bytes()
        append_skip_record(_entry(game_id="2026_W03_NYJ@MIA"), path=log_path)
        assert log_path.read_bytes().startswith(first)
        assert len(read_skip_records(log_path)) == 2


class TestTheNaturalKey:
    def test_re_appending_the_same_logical_entry_is_a_byte_identical_no_op(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(), path=log_path)
        before = log_path.read_bytes()
        assert append_skip_record(_entry(), path=log_path) is False
        assert log_path.read_bytes() == before

    def test_two_entries_differing_only_in_run_id_collapse_to_one_line(
        self, log_path: Path
    ) -> None:
        """Keying on run_id would make every re-run a new record by construction."""
        append_skip_record(_entry(), path=log_path)
        second = _entry(
            run_id="2026-09-19T19:05:00-04:00", recorded_at="2026-09-19T23:05:00+00:00"
        )
        assert append_skip_record(second, path=log_path) is False
        assert len(read_skip_records(log_path)) == 1

    def test_the_same_game_from_two_sources_is_two_records(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(source="elo"), path=log_path)
        assert append_skip_record(_entry(source="injury"), path=log_path) is True
        assert [r["source"] for r in read_skip_records(log_path)] == ["elo", "injury"]

    def test_the_same_game_on_another_run_date_is_a_new_record(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(), path=log_path)
        assert append_skip_record(_entry(run_date_et="2026-09-20"), path=log_path)
        assert len(read_skip_records(log_path)) == 2

    def test_the_same_game_for_another_reason_is_a_new_record(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(), path=log_path)
        assert append_skip_record(
            _entry(reason="undated", information_time=None), path=log_path
        )
        assert len(read_skip_records(log_path)) == 2


class TestRefusals:
    def test_an_entry_missing_lock_raises_and_writes_nothing(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(game_id="2026_W03_NYJ@MIA"), path=log_path)
        before = log_path.read_bytes()
        incomplete = _entry()
        del incomplete["lock"]

        with pytest.raises(SkipRecordCorrupt, match="lock"):
            append_skip_record(incomplete, path=log_path)
        assert log_path.read_bytes() == before

    def test_the_refusal_names_every_missing_key(self, log_path: Path) -> None:
        incomplete = _entry()
        for key in ("run_id", "source", "recorded_at"):
            del incomplete[key]
        with pytest.raises(SkipRecordCorrupt) as caught:
            append_skip_record(incomplete, path=log_path)
        for key in ("run_id", "source", "recorded_at"):
            assert key in str(caught.value)
        assert not log_path.exists()

    @pytest.mark.parametrize("key", REQUIRED_ENTRY_KEYS)
    def test_each_required_key_is_individually_required(
        self, log_path: Path, key: str
    ) -> None:
        incomplete = _entry()
        del incomplete[key]
        with pytest.raises(SkipRecordCorrupt, match=key):
            append_skip_record(incomplete, path=log_path)
        assert not log_path.exists()

    def test_an_unknown_reason_is_refused_not_recorded(self, log_path: Path) -> None:
        """There is no reason that means 'checked and waived'."""
        with pytest.raises(SkipRecordCorrupt, match="waived"):
            append_skip_record(_entry(reason="waived"), path=log_path)
        assert not log_path.exists()

    @pytest.mark.parametrize(
        "key", ["run_id", "run_date_et", "game_id", "source", "reason", "recorded_at"]
    )
    def test_an_identity_field_may_not_be_null(self, log_path: Path, key: str) -> None:
        with pytest.raises(SkipRecordCorrupt, match=key):
            append_skip_record(_entry(**{key: None}), path=log_path)
        assert not log_path.exists()

    def test_a_null_information_time_is_the_honest_undated_value(
        self, log_path: Path
    ) -> None:
        """An undated value HAS no information time; null is the truth, not a gap."""
        assert append_skip_record(
            _entry(reason="undated", information_time=None), path=log_path
        )

    def test_a_non_mapping_entry_is_refused(self, log_path: Path) -> None:
        with pytest.raises(SkipRecordCorrupt, match="mapping"):
            append_skip_record(["not", "a", "mapping"], path=log_path)  # type: ignore[arg-type]


class TestReading:
    def test_an_absent_file_reads_as_no_records(self, log_path: Path) -> None:
        assert read_skip_records(log_path) == []

    def test_a_malformed_line_raises_naming_its_line_number(
        self, log_path: Path
    ) -> None:
        append_skip_record(_entry(), path=log_path)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("{not json\n")
        with pytest.raises(SkipRecordCorrupt, match="line 2"):
            read_skip_records(log_path)

    def test_a_line_missing_a_required_key_raises_on_read(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True)
        partial = _entry()
        del partial["reason"]
        log_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")
        with pytest.raises(SkipRecordCorrupt, match="reason"):
            read_skip_records(log_path)

    def test_a_blank_line_is_an_anomaly_not_whitespace(self, log_path: Path) -> None:
        append_skip_record(_entry(), path=log_path)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n")
        with pytest.raises(SkipRecordCorrupt, match="BLANK"):
            read_skip_records(log_path)

    def test_a_corrupt_file_also_blocks_the_append(self, log_path: Path) -> None:
        """The idempotency scan must not treat an unreadable line as absent."""
        log_path.parent.mkdir(parents=True)
        log_path.write_text("{truncated\n", encoding="utf-8")
        before = log_path.read_bytes()
        with pytest.raises(SkipRecordCorrupt):
            append_skip_record(_entry(), path=log_path)
        assert log_path.read_bytes() == before

    def test_records_for_one_run_date_are_selected(self, log_path: Path) -> None:
        append_skip_record(_entry(), path=log_path)
        append_skip_record(_entry(run_date_et="2026-09-20"), path=log_path)
        append_skip_record(_entry(game_id="2026_W03_NYJ@MIA"), path=log_path)

        selected = skip_records_for_run_date("2026-09-19", path=log_path)
        assert [r["game_id"] for r in selected] == [
            "2026_W03_KC@BUF",
            "2026_W03_NYJ@MIA",
        ]
        assert skip_records_for_run_date("2026-09-21", path=log_path) == []


def test_the_default_path_is_resolved_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test (or Plan 33.2-27's sink) can redirect the store without re-binding defaults."""
    redirected = tmp_path / "elsewhere.jsonl"
    monkeypatch.setattr(skip_log, "SKIP_RECORD_PATH", redirected)
    append_skip_record(_entry())
    assert read_skip_records() == [_entry()]
    assert redirected.exists()
