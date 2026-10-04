"""The graded-weeks seam (D32-11, Plan 32-03 Task 2).

`data/graded_weeks.py` answers ONE question -- "which weeks of this season have already been
graded, so a revision touching one of them can escalate" -- and Phase 34 repoints it at the
forward ledger when LDGR-01 relocates the durable store out of ``outputs/``.

The seam sits INSIDE a detector, which is what makes its failure mode dangerous: a silent
failure there looks exactly like "nothing to report". So the tests below spend most of their
weight on the distinction between the two empty-looking answers:

* NO STORE AT ALL -- a legitimate state (a checkout that has never generated a bet list).
  Resolves to an EXPLICITLY RECORDED empty set whose ``reason`` names the absent path.
* A STORE THAT CANNOT BE READ -- schema mismatch, truncation, corruption. RAISES
  ``GradedWeeksUnavailable``, which is deliberately outside ``(RuntimeError, ValueError,
  ImportError)`` so a caller's broad ``except`` cannot convert "I could not tell whether
  this week was graded" into "no weeks are graded" and silently downgrade a CRITICAL.

No test here reads the real ``outputs/bet_list/``; every fixture is built inside ``tmp_path``.

ASCII only, no emoji (CLAUDE.md hard constraint).

Run:  .venv/Scripts/python.exe -m pytest tests/unit/test_graded_weeks.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    GRADING_STATUS_PENDING,
    GRADING_STATUSES,
)
from backtest.weekly_bet_list import BET_LIST_ARTIFACT_NAME
from data import graded_weeks as graded_weeks_module
from data.graded_weeks import (
    GRADED_WEEKS_SOURCE,
    TERMINAL_GRADING_STATUSES,
    GradedWeeksUnavailable,
    graded_weeks,
    graded_weeks_record,
)

# The four grading statuses, taken from the ONE source rather than re-typed here either.
_WIN, _LOSS, _PUSH = (
    status for status in GRADING_STATUSES if status != GRADING_STATUS_PENDING
)


def _write_bet_list(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a bet-list artifact carrying *rows* under *tmp_path*; return the directory.

    Every locked column is materialised so ``read_bet_list_artifact`` accepts the frame --
    the same "build the real shape in tmp_path" discipline as
    ``tests/unit/test_upstream_pin.py::_write_pin``.
    """
    directory = tmp_path / "bet_list"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in BET_LIST_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame[BET_LIST_COLUMNS].to_parquet(
        directory / BET_LIST_ARTIFACT_NAME,
        index=False,
    )
    return directory


def _row(season: int, week: int, status: str, target: str = "ou") -> dict[str, object]:
    """One bet-list row, carrying only the columns this seam reads."""
    return {
        "game_id": f"{season}_{week:02d}_HOME_AWAY",
        "season": season,
        "week": week,
        "target": target,
        "grading_status": status,
    }


class TestAnAbsentStoreIsARecordedEmptySet:
    """A checkout that never generated a bet list is a legitimate state, not a failure."""

    def test_no_artifact_yields_an_empty_set(self, tmp_path: Path) -> None:
        assert graded_weeks(2026, output_dir=tmp_path) == set()

    def test_the_record_resolves_with_an_empty_week_list(self, tmp_path: Path) -> None:
        record = graded_weeks_record(2026, output_dir=tmp_path)
        assert record["weeks"] == []
        assert record["resolved"] is True

    def test_the_record_names_the_absent_path_as_its_reason(
        self, tmp_path: Path
    ) -> None:
        # The empty set must be EXPLICITLY RECORDED, not merely returned: a verdict can
        # never say "no weeks were graded" without also saying where it looked.
        record = graded_weeks_record(2026, output_dir=tmp_path)
        assert record["reason"] is not None
        assert BET_LIST_ARTIFACT_NAME in str(record["reason"])

    def test_the_record_carries_the_source_on_the_absent_path(
        self, tmp_path: Path
    ) -> None:
        record = graded_weeks_record(2026, output_dir=tmp_path)
        assert record["source"] == GRADED_WEEKS_SOURCE

    def test_an_existing_store_with_no_graded_rows_has_no_reason(
        self, tmp_path: Path
    ) -> None:
        # "The store exists and nothing is graded yet" is a THIRD state, distinct from
        # "there is no store". Both yield [], and the reason is what tells them apart.
        directory = _write_bet_list(tmp_path, [_row(2026, 4, GRADING_STATUS_PENDING)])
        record = graded_weeks_record(2026, output_dir=directory)
        assert record["weeks"] == []
        assert record["reason"] is None
        assert record["resolved"] is True


class TestTheTerminalStatusesDecideWhatCounts:
    """pending is excluded; a push is INCLUDED, because a push is a settled result."""

    def test_pending_is_excluded_and_a_push_is_included(self, tmp_path: Path) -> None:
        directory = _write_bet_list(
            tmp_path,
            [
                _row(2026, 2, _WIN),
                _row(2026, 3, _PUSH),
                _row(2026, 4, GRADING_STATUS_PENDING),
            ],
        )
        assert graded_weeks(2026, output_dir=directory) == {2, 3}

    def test_a_loss_also_counts_as_graded(self, tmp_path: Path) -> None:
        directory = _write_bet_list(tmp_path, [_row(2026, 5, _LOSS)])
        assert graded_weeks(2026, output_dir=directory) == {5}

    def test_a_week_with_one_graded_and_one_pending_row_counts(
        self, tmp_path: Path
    ) -> None:
        directory = _write_bet_list(
            tmp_path,
            [
                _row(2026, 6, GRADING_STATUS_PENDING, target="wp"),
                _row(2026, 6, _WIN, target="ou"),
            ],
        )
        assert graded_weeks(2026, output_dir=directory) == {6}

    def test_the_vocabulary_is_derived_from_the_one_source(self) -> None:
        expected = tuple(
            status for status in GRADING_STATUSES if status != GRADING_STATUS_PENDING
        )
        assert expected == TERMINAL_GRADING_STATUSES
        assert GRADING_STATUS_PENDING not in TERMINAL_GRADING_STATUSES
        assert _PUSH in TERMINAL_GRADING_STATUSES

    def test_the_module_never_re_types_the_grading_vocabulary(self) -> None:
        # A second hand-written list is the failure this repository has already paid for
        # once (``backtest/bet_tracker.py``'s docstring says so about the label vocabulary).
        source = Path(graded_weeks_module.__file__).read_text(encoding="utf-8")
        for status in GRADING_STATUSES:
            assert f'"{status}"' not in source
            assert f"'{status}'" not in source


class TestAnotherSeasonNeverLeaksIn:
    """The season filter is part of the answer, not an assumption about the store."""

    def test_rows_from_a_different_season_are_excluded(self, tmp_path: Path) -> None:
        directory = _write_bet_list(
            tmp_path,
            [
                _row(2025, 9, _WIN),
                _row(2026, 2, _WIN),
            ],
        )
        assert graded_weeks(2026, output_dir=directory) == {2}

    def test_the_other_season_is_still_answerable(self, tmp_path: Path) -> None:
        directory = _write_bet_list(
            tmp_path,
            [
                _row(2025, 9, _WIN),
                _row(2026, 2, _WIN),
            ],
        )
        assert graded_weeks(2025, output_dir=directory) == {9}


class TestAnUnreadableStoreRaisesAndNeverReturnsEmpty:
    """Absent and unreadable are different facts, and neither is a silent empty."""

    def test_a_schema_mismatch_raises(self, tmp_path: Path) -> None:
        directory = tmp_path / "bet_list"
        directory.mkdir(parents=True)
        pd.DataFrame({"season": [2026], "week": [2]}).to_parquet(
            directory / BET_LIST_ARTIFACT_NAME, index=False
        )
        with pytest.raises(GradedWeeksUnavailable):
            graded_weeks(2026, output_dir=directory)

    def test_the_schema_mismatch_message_names_the_path(self, tmp_path: Path) -> None:
        directory = tmp_path / "bet_list"
        directory.mkdir(parents=True)
        pd.DataFrame({"season": [2026]}).to_parquet(
            directory / BET_LIST_ARTIFACT_NAME, index=False
        )
        with pytest.raises(GradedWeeksUnavailable) as excinfo:
            graded_weeks(2026, output_dir=directory)
        assert BET_LIST_ARTIFACT_NAME in str(excinfo.value)

    def test_the_schema_mismatch_is_chained_from_the_original_error(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "bet_list"
        directory.mkdir(parents=True)
        pd.DataFrame({"season": [2026]}).to_parquet(
            directory / BET_LIST_ARTIFACT_NAME, index=False
        )
        with pytest.raises(GradedWeeksUnavailable) as excinfo:
            graded_weeks(2026, output_dir=directory)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_a_corrupt_parquet_raises_rather_than_returning_empty(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "bet_list"
        directory.mkdir(parents=True)
        (directory / BET_LIST_ARTIFACT_NAME).write_bytes(b"not a parquet file at all")
        with pytest.raises(GradedWeeksUnavailable):
            graded_weeks(2026, output_dir=directory)

    def test_the_record_does_not_swallow_the_refusal(self, tmp_path: Path) -> None:
        # graded_weeks_record must NOT catch it: an unresolvable store is the CALLER's
        # decision to record as UNKNOWN. Swallowing it here puts the silent downgrade back.
        directory = tmp_path / "bet_list"
        directory.mkdir(parents=True)
        (directory / BET_LIST_ARTIFACT_NAME).write_bytes(b"truncated")
        with pytest.raises(GradedWeeksUnavailable):
            graded_weeks_record(2026, output_dir=directory)


class TestTheExceptionIsOutsideEveryBroadHandler:
    """The same discipline ``UpstreamPinError`` states, for the same reason."""

    def test_it_is_not_a_runtime_value_or_import_error(self) -> None:
        assert not issubclass(
            GradedWeeksUnavailable, (RuntimeError, ValueError, ImportError)
        )

    def test_it_is_an_exception(self) -> None:
        assert issubclass(GradedWeeksUnavailable, Exception)

    def test_a_wired_call_sites_except_tuple_does_not_catch_it(self) -> None:
        # The literal tuple ``features/qb_tracking.py`` and friends catch around loaders.
        caught = False
        try:
            raise GradedWeeksUnavailable("probe")
        except (ImportError, ValueError, RuntimeError):
            caught = True
        except GradedWeeksUnavailable:
            caught = False
        assert caught is False


class TestTheRecordSerialisesStraightIntoAManifest:
    """``weeks`` is a sorted list of int, and ``source`` is present on every path."""

    def test_weeks_is_a_sorted_list_of_int(self, tmp_path: Path) -> None:
        directory = _write_bet_list(
            tmp_path,
            [
                _row(2026, 7, _WIN),
                _row(2026, 2, _LOSS),
                _row(2026, 5, _PUSH),
            ],
        )
        record = graded_weeks_record(2026, output_dir=directory)
        assert record["weeks"] == [2, 5, 7]
        assert isinstance(record["weeks"], list)
        assert all(type(week) is int for week in record["weeks"])

    def test_the_record_carries_the_season_and_source(self, tmp_path: Path) -> None:
        directory = _write_bet_list(tmp_path, [_row(2026, 2, _WIN)])
        record = graded_weeks_record(2026, output_dir=directory)
        assert record["season"] == 2026
        assert record["source"] == GRADED_WEEKS_SOURCE

    def test_the_source_names_where_the_answer_came_from(self) -> None:
        assert "weekly_bet_list" in GRADED_WEEKS_SOURCE
        assert "bet_list.parquet" in GRADED_WEEKS_SOURCE


class TestTheSeamKeepsItsDependenciesOutOfImportPosition:
    """No module-level ``backtest``/``api`` import, and the repointing note is in source."""

    def test_no_top_level_backtest_or_api_import(self) -> None:
        tree = ast.parse(Path(graded_weeks_module.__file__).read_text(encoding="utf-8"))
        modules = [
            node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
        ]
        modules += [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        offenders = [
            module for module in modules if module.split(".")[0] in {"backtest", "api"}
        ]
        assert offenders == []

    def test_the_docstring_names_phase_34_and_the_relocation(self) -> None:
        doc = graded_weeks_module.__doc__ or ""
        assert "Phase 34" in doc
        assert "LDGR-01" in doc
        assert "outputs/" in doc

    def test_the_module_is_ascii_only(self) -> None:
        source = Path(graded_weeks_module.__file__).read_text(encoding="utf-8")
        assert source.isascii()
