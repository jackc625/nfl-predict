"""The no-write execution seam: every write primitive the daily path reaches asks ONE sink.

WHY (Plan 33.2-27 Task 2b, reviews round ``f924749``, Codex 33.2-27 HIGH)
------------------------------------------------------------------------
A dry run that relied on a digest comparison AFTER the fact could only report that a write had
happened, never prevent one -- and the live capture appends to two git-tracked config records.
``data.write_sink`` puts a ``ContextVar`` in front of every write primitive, on the
``data.upstream_live.as_of_capture`` pattern: under ``RecordingSink`` each primitive records the
write it would have made and persists NOTHING; under the default ``ProductionSink`` nothing
changes. The content-digest bracket (``tests/data_boundary.py``) stays as the SECOND line, and a
planted writer that bypasses the sink proves it still fires.

EVERYTHING HERE IS SANDBOXED under ``tmp_path``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data.storage as storage_mod
from data.sealed_probe_log import append_probe_entry
from data.upstream_live import write_live_manifest
from pipeline.skip_log import append_skip_record
from tests.data_boundary import (
    DataBoundaryViolation,
    assert_tree_unchanged,
    digest_tree,
)

SANDBOX_SUFFIXES = (".parquet", ".json", ".jsonl")
CAPTURED_AT = datetime(2026, 9, 26, 21, 30, tzinfo=UTC)
LOCK = datetime(2026, 9, 26, 22, 0, tzinfo=UTC)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_W04_DAL@PHI"],
            "sportsbook": ["draftkings"],
            "snapshot_ts": [LOCK],
            "created_at": [CAPTURED_AT],
            "spread": [3.5],
        }
    )


def _probe_entry() -> dict[str, Any]:
    return {
        "event_class": "none",
        "severity": "none",
        "probed_at_utc": CAPTURED_AT.isoformat(),
        "checked": [],
        "expected": [],
    }


def _skip_entry() -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "run_date_et": "2026-09-26",
        "game_id": "2026_W04_DAL@PHI",
        "source": "decision_instant",
        "information_time": None,
        "lock": LOCK.isoformat(),
        "reason": "post_lock",
        "recorded_at": CAPTURED_AT.isoformat(),
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "season": 2026,
        "zone": "live",
        "source": "test",
        "datasets": {},
    }


def _primitives(root: Path) -> dict[str, Callable[[], Any]]:
    """Every write primitive the daily path reaches, each pointed at *root*."""
    return {
        "save_dataframe": lambda: storage_mod.save_dataframe(
            _frame(), "odds_snapshot", save_to_db=False
        ),
        "save_bronze_snapshot": lambda: storage_mod.save_bronze_snapshot(
            _frame(), "odds", season=2026, week=4, base_path=root
        ),
        "upsert_silver": lambda: storage_mod.upsert_silver(
            _frame(), "games_probe", base_path=root
        ),
        "upsert_silver_composite": lambda: storage_mod.upsert_silver_composite(
            _frame(), "odds_timeline_probe", base_path=root
        ),
        "append_odds_captures": lambda: storage_mod.append_odds_captures(
            _frame(), base_path=root
        ),
        "append_probe_entry": lambda: append_probe_entry(
            _probe_entry(), path=root / "config" / "upstream_probe_log.jsonl"
        ),
        "write_live_manifest": lambda: write_live_manifest(
            _manifest(), manifest_dir=root / "config" / "upstream_live"
        ),
        "append_skip_record": lambda: append_skip_record(
            _skip_entry(), path=root / "config" / "skip_records.jsonl"
        ),
    }


@pytest.fixture
def write_sink():
    """``data.write_sink``, or a named FAILURE (not a collection error) while it is absent."""
    import importlib

    try:
        return importlib.import_module("data.write_sink")
    except ModuleNotFoundError:
        pytest.fail(
            "data.write_sink does not exist: no sink stands in front of the write "
            "primitives, so a dry run cannot avoid writing"
        )


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sandbox root holding one pre-existing file, with the parquet manager pointed at it."""
    root = tmp_path / "lake"
    (root / "silver").mkdir(parents=True)
    _frame().to_parquet(root / "silver" / "seed.parquet")
    monkeypatch.setattr(
        storage_mod, "_parquet_manager", storage_mod.ParquetManager(str(root))
    )
    monkeypatch.setattr(storage_mod, "_keep_duckdb_copy_in_step", lambda *a, **k: None)
    return root


class TestTheSinkModule:
    """``current_sink`` / ``active_sink`` on the ``as_of_capture`` ContextVar pattern."""

    def test_the_default_sink_is_production(self, write_sink):
        assert isinstance(write_sink.current_sink(), write_sink.ProductionSink)

    def test_active_sink_sets_and_resets(self, write_sink):
        recorder = write_sink.RecordingSink()
        with write_sink.active_sink(recorder):
            assert write_sink.current_sink() is recorder
        assert isinstance(write_sink.current_sink(), write_sink.ProductionSink)

    def test_a_raise_inside_the_block_still_resets_the_sink(self, write_sink):
        recorder = write_sink.RecordingSink()
        with pytest.raises(RuntimeError), write_sink.active_sink(recorder):
            raise RuntimeError("dry run failed")
        assert isinstance(write_sink.current_sink(), write_sink.ProductionSink), (
            "a failing dry run left recording on for the next real run"
        )

    def test_the_docstring_names_what_the_seam_does_not_cover(self, write_sink):
        assert "step_populate_web_cache" in (write_sink.__doc__ or "")


class TestUnderRecordingNothingIsPersisted:
    """Each primitive records its intended write and the sandbox stays byte-identical."""

    @pytest.mark.parametrize(
        "primitive",
        [
            "save_dataframe",
            "save_bronze_snapshot",
            "upsert_silver",
            "upsert_silver_composite",
            "append_odds_captures",
            "append_probe_entry",
            "write_live_manifest",
            "append_skip_record",
        ],
    )
    def test_the_primitive_records_and_persists_nothing(
        self, write_sink, sandbox: Path, primitive: str
    ):
        before = digest_tree(sandbox, SANDBOX_SUFFIXES)
        files_before = sorted(p.relative_to(sandbox) for p in sandbox.rglob("*"))
        recorder = write_sink.RecordingSink()

        with write_sink.active_sink(recorder):
            _primitives(sandbox)[primitive]()

        assert_tree_unchanged(before, digest_tree(sandbox, SANDBOX_SUFFIXES), sandbox)
        assert (
            sorted(p.relative_to(sandbox) for p in sandbox.rglob("*")) == files_before
        ), f"{primitive} created a file or directory under a RecordingSink"
        assert [write.kind for write in recorder.intended_writes] == [primitive]
        assert recorder.production_write_count == 0

    def test_a_whole_recording_run_records_a_non_zero_count(
        self, write_sink, sandbox: Path
    ):
        """Non-vacuity: 'nothing was written' cannot pass because nothing was attempted."""
        recorder = write_sink.RecordingSink()
        with write_sink.active_sink(recorder):
            for call in _primitives(sandbox).values():
                call()
        assert len(recorder.intended_writes) == 8
        assert all(write.target for write in recorder.intended_writes)


class TestUnderProductionEachPrimitiveWrites:
    """The default sink writes exactly as before."""

    def test_every_primitive_writes_under_the_default_sink(
        self, write_sink, sandbox: Path
    ):
        before = digest_tree(sandbox, SANDBOX_SUFFIXES)
        for call in _primitives(sandbox).values():
            call()
        after = digest_tree(sandbox, SANDBOX_SUFFIXES)
        added = set(after) - set(before)
        assert "silver/odds_snapshot.parquet" in added
        assert "silver/games_probe.parquet" in added
        assert "silver/odds_timeline_probe.parquet" in added
        assert "config/upstream_probe_log.jsonl" in added
        assert "config/upstream_live/2026.json" in added
        assert "config/skip_records.jsonl" in added
        assert any(
            name.startswith("bronze/odds_raw_bronze_2026_W04_") for name in added
        )


class TestTheDigestBracketIsTheSecondLine:
    """A writer that BYPASSES the sink is still caught by the content digest."""

    def test_a_planted_bypassing_writer_is_caught(self, write_sink, sandbox: Path):
        before = digest_tree(sandbox, SANDBOX_SUFFIXES)
        with write_sink.active_sink(write_sink.RecordingSink()):
            _frame().to_parquet(sandbox / "silver" / "seed.parquet")
            _frame().assign(spread=9.5).to_parquet(sandbox / "silver" / "seed.parquet")
        with pytest.raises(DataBoundaryViolation):
            assert_tree_unchanged(
                before, digest_tree(sandbox, SANDBOX_SUFFIXES), sandbox
            )
