"""The sealed probe log is append-only, and a log that cannot be read BLOCKS.

TEST CLASS: plain unit tests. Every case writes inside ``tmp_path``; nothing here touches
``config/upstream_probe_log.jsonl``, which Plan 32-05 deliberately does not create -- the
file comes into existence when Plan 32-08 runs the first probe, so its first committed
lines are real verdicts rather than a seeded placeholder. No network, no data lake.

Three properties are proved, and each one is the reason a specific failure cannot happen:

1. APPEND-ONLY. A second append leaves the first line's raw bytes untouched, and one line
   lands per run for ``clean`` and ``unknown`` alike. That is what makes a season of lines
   the proof the detector was alive, and a gap in them evidence that it was not.
2. AN UNREADABLE LOG BLOCKS. A truncated line raises and names the line number rather than
   reading as absent -- a crashed append must not look exactly like a clean slate.
3. AN EMPTY LOG IS STALE. A detector that never ran and a detector that ran and found
   nothing are different facts, and only one of them leaves a line.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from data.sealed_probe_log import (
    REQUIRED_ENTRY_KEYS,
    SEALED_PROBE_LOG_PATH,
    SealedProbeLogCorrupt,
    append_probe_entry,
    latest_entry,
    probe_log_staleness,
    read_probe_log,
)


def entry(
    *,
    event_class: str = "clean",
    severity: str = "informational",
    probed_at_utc: str = "2026-09-10T00:00:00+00:00",
    checked: int = 76,
    expected: int = 76,
    **extra: Any,
) -> dict[str, Any]:
    """A minimal verdict line, in the shape ``data.sealed_probe.probe_sealed`` returns."""
    return {
        "event_class": event_class,
        "severity": severity,
        "probed_at_utc": probed_at_utc,
        "checked": checked,
        "expected": expected,
        **extra,
    }


class TestTheLogIsAppendOnly:
    """D32-08: one line per run, and no run ever touches another run's line."""

    def test_a_second_append_leaves_the_first_line_byte_unchanged(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"

        append_probe_entry(entry(probed_at_utc="2026-09-10T00:00:00+00:00"), path=log)
        after_first = log.read_bytes()

        append_probe_entry(entry(probed_at_utc="2026-09-17T00:00:00+00:00"), path=log)
        after_second = log.read_bytes()

        assert after_second.startswith(after_first), (
            "the second append rewrote the first line. Append mode is the only thing "
            "making the append-only claim true; if it is not in force the whole log is "
            "decoration."
        )
        assert len(after_second.splitlines()) == 2
        entries = read_probe_log(log)
        assert [item["probed_at_utc"] for item in entries] == [
            "2026-09-10T00:00:00+00:00",
            "2026-09-17T00:00:00+00:00",
        ]

    @pytest.mark.parametrize(
        ("event_class", "severity", "checked"),
        [
            pytest.param("clean", "informational", 76, id="clean"),
            pytest.param("unknown", "warning", 75, id="unknown"),
        ],
    )
    def test_one_line_is_appended_per_run_including_clean_and_unknown(
        self, tmp_path: Path, event_class: str, severity: str, checked: int
    ) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        record = entry(event_class=event_class, severity=severity, checked=checked)

        append_probe_entry(record, path=log)
        assert len(log.read_text(encoding="utf-8").splitlines()) == 1

        append_probe_entry(record, path=log)
        assert len(log.read_text(encoding="utf-8").splitlines()) == 2
        assert len(read_probe_log(log)) == 2

    @pytest.mark.parametrize("missing", ["checked", "expected"])
    def test_an_entry_missing_its_coverage_fields_is_refused(
        self, tmp_path: Path, missing: str
    ) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        record = entry()
        del record[missing]

        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            append_probe_entry(record, path=log)

        assert missing in str(excinfo.value)
        assert not log.exists(), "a refused entry must not create the log"

    @pytest.mark.parametrize("missing", list(REQUIRED_ENTRY_KEYS))
    def test_every_required_key_is_actually_required(
        self, tmp_path: Path, missing: str
    ) -> None:
        record = entry()
        del record[missing]
        with pytest.raises(SealedProbeLogCorrupt):
            append_probe_entry(record, path=tmp_path / "log.jsonl")

    def test_a_non_mapping_entry_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            append_probe_entry(["not", "a", "mapping"], path=tmp_path / "log.jsonl")
        assert "mapping" in str(excinfo.value)

    def test_the_same_verdict_produces_the_same_bytes(self, tmp_path: Path) -> None:
        # sort_keys=True makes a line a function of its CONTENT alone, so a diff between
        # two runs means a real difference rather than a dict-ordering accident.
        first = tmp_path / "a.jsonl"
        second = tmp_path / "b.jsonl"
        append_probe_entry({**entry(), "findings": [], "unresolved": []}, path=first)
        append_probe_entry(
            {"unresolved": [], "findings": [], **entry()},
            path=second,
        )
        assert first.read_bytes() == second.read_bytes()

    def test_the_parent_directory_is_created(self, tmp_path: Path) -> None:
        log = tmp_path / "nested" / "deeper" / "log.jsonl"
        append_probe_entry(entry(), path=log)
        assert log.is_file()


class TestAnUnreadableLogBlocks:
    """``read_run_ledger``'s contract, reused: a truncated write is not a clean slate."""

    def test_a_truncated_line_raises_rather_than_reading_as_absent(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(), path=log)
        with log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write('{"event_class": "clea')

        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            read_probe_log(log)

        assert "line 2" in str(excinfo.value)

    def test_an_absent_log_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_probe_log(tmp_path / "never-written.jsonl") == []
        assert latest_entry(tmp_path / "never-written.jsonl") is None

    def test_a_blank_line_is_an_anomaly_not_whitespace(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(), path=log)
        with log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n")
            handle.write('{"x": 1}\n')

        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            read_probe_log(log)

        assert "BLANK" in str(excinfo.value)
        assert "line 2" in str(excinfo.value)

    def test_a_non_mapping_line_is_refused(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        log.write_text("[1, 2, 3]\n", encoding="utf-8", newline="\n")
        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            read_probe_log(log)
        assert "line 1" in str(excinfo.value)

    def test_the_exception_sits_outside_every_existing_handler(self) -> None:
        assert not issubclass(
            SealedProbeLogCorrupt, RuntimeError | ValueError | ImportError
        )


class TestTheLogProvesTheDetectorWasAlive:
    """The cheap liveness check: a detector that stopped running leaves a gap."""

    NOW = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)

    def test_an_empty_log_is_stale(self, tmp_path: Path) -> None:
        report = probe_log_staleness(
            path=tmp_path / "never-written.jsonl", max_age_days=8, now=self.NOW
        )
        assert report["stale"] is True
        assert report["latest_probed_at_utc"] is None
        assert report["age_days"] is None
        assert report["entries"] == 0

    def test_a_gap_longer_than_the_cadence_is_stale(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(probed_at_utc="2026-09-01T00:00:00+00:00"), path=log)

        report = probe_log_staleness(path=log, max_age_days=8, now=self.NOW)

        assert report["stale"] is True
        assert report["age_days"] == pytest.approx(17.0)
        assert report["entries"] == 1

    def test_a_recent_entry_is_not_stale(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(probed_at_utc="2026-09-11T00:00:00+00:00"), path=log)

        report = probe_log_staleness(path=log, max_age_days=8, now=self.NOW)

        assert report["stale"] is False
        assert report["age_days"] == pytest.approx(7.0)
        assert report["latest_probed_at_utc"] == "2026-09-11T00:00:00+00:00"

    def test_the_newest_entry_is_the_last_appended_not_the_newest_stamp(
        self, tmp_path: Path
    ) -> None:
        # Positional, never re-sorted: the order in the file is the order the runs
        # happened, and a wrong clock must not be able to reorder history.
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(probed_at_utc="2026-09-17T00:00:00+00:00"), path=log)
        append_probe_entry(entry(probed_at_utc="2026-09-11T00:00:00+00:00"), path=log)

        assert latest_entry(log)["probed_at_utc"] == "2026-09-11T00:00:00+00:00"
        report = probe_log_staleness(path=log, max_age_days=8, now=self.NOW)
        assert report["latest_probed_at_utc"] == "2026-09-11T00:00:00+00:00"

    def test_a_naive_probed_at_is_refused_rather_than_silently_ranked(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(probed_at_utc="2026-09-11T00:00:00"), path=log)
        with pytest.raises(SealedProbeLogCorrupt) as excinfo:
            probe_log_staleness(path=log, max_age_days=8, now=self.NOW)
        assert "NAIVE" in str(excinfo.value)

    def test_a_naive_now_is_refused(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(), path=log)
        with pytest.raises(ValueError) as excinfo:
            probe_log_staleness(
                path=log, max_age_days=8, now=datetime(2026, 9, 18, 0, 0)
            )
        assert "timezone-aware" in str(excinfo.value)

    def test_lines_use_lf_endings(self, tmp_path: Path) -> None:
        log = tmp_path / "upstream_probe_log.jsonl"
        append_probe_entry(entry(), path=log)
        append_probe_entry(entry(event_class="unknown", severity="warning"), path=log)

        raw = log.read_bytes()

        assert b"\r\n" not in raw, (
            "the log is committed; a CRLF flip would rewrite every line in the diff and "
            "destroy the append-only reading of its history."
        )
        assert raw.endswith(b"\n")


class TestThisPlanDoesNotCreateTheCommittedLog:
    """The first committed lines must be real verdicts, not a seeded placeholder."""

    def test_the_committed_log_path_is_where_it_is_expected(self) -> None:
        assert Path("config/upstream_probe_log.jsonl") == SEALED_PROBE_LOG_PATH

    def test_plan_32_05_did_not_create_it(self) -> None:
        if SEALED_PROBE_LOG_PATH.is_file():
            pytest.skip(
                f"{SEALED_PROBE_LOG_PATH} is present at "
                f"{SEALED_PROBE_LOG_PATH.resolve()} -- Plan 32-08 has run the first probe"
            )
        assert read_probe_log() == []
        assert latest_entry() is None
