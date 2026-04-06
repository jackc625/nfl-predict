"""Tests for pipeline execution log."""

import json
import os
from unittest.mock import patch

from pipeline.execution_log import (
    ExecutionLog,
    StepLogEntry,
    write_execution_log_atomic,
)


class TestStepLogEntry:
    """Tests for StepLogEntry Pydantic model."""

    def test_step_log_entry_fields(self):
        """StepLogEntry captures name, status, duration_ms, retry_count, error."""
        entry = StepLogEntry(
            name="ingest_games",
            status="success",
            duration_ms=1500.0,
            retry_count=0,
            error=None,
        )
        assert entry.name == "ingest_games"
        assert entry.status == "success"
        assert entry.duration_ms == 1500.0
        assert entry.retry_count == 0
        assert entry.error is None

    def test_step_log_entry_with_error(self):
        """StepLogEntry records error message when step fails."""
        entry = StepLogEntry(
            name="ingest_games",
            status="failed",
            duration_ms=500.0,
            retry_count=2,
            error="ConnectionError: timeout",
        )
        assert entry.status == "failed"
        assert entry.retry_count == 2
        assert entry.error == "ConnectionError: timeout"


class TestExecutionLog:
    """Tests for ExecutionLog Pydantic model."""

    def test_execution_log_serialization(self):
        """ExecutionLog serializes to valid JSON with all fields."""
        log = ExecutionLog(
            status="running",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=12345,
        )
        json_str = log.model_dump_json()
        data = json.loads(json_str)
        assert data["status"] == "running"
        assert data["start_time"] == "2025-10-03T17:00:00"
        assert data["season"] == 2025
        assert data["week"] == 5
        assert data["pid"] == 12345

    def test_execution_log_has_pid_field(self):
        """ExecutionLog has pid field as int."""
        log = ExecutionLog(
            status="running",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=99999,
        )
        assert isinstance(log.pid, int)
        assert log.pid == 99999

    def test_execution_log_default_values(self):
        """ExecutionLog has correct defaults: steps=[], warnings=[], error=None, forced=False, mode='full'."""
        log = ExecutionLog(
            status="running",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=1,
        )
        assert log.steps == []
        assert log.warnings == []
        assert log.error is None
        assert log.forced is False
        assert log.mode == "full"

    def test_execution_log_status_values(self):
        """ExecutionLog accepts 'running', 'success', 'failed', 'degraded' statuses."""
        for status in ["running", "success", "failed", "degraded"]:
            log = ExecutionLog(
                status=status,
                start_time="2025-10-03T17:00:00",
                season=2025,
                week=5,
                pid=1,
            )
            assert log.status == status


class TestWriteExecutionLogAtomic:
    """Tests for write_execution_log_atomic function."""

    def test_write_execution_log_atomic_creates_file(self, tmp_path):
        """write_execution_log_atomic creates valid JSON file."""
        log = ExecutionLog(
            status="success",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=1,
        )
        log_path = tmp_path / "logs" / "pipeline.json"
        write_execution_log_atomic(log, log_path)

        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert data["status"] == "success"
        assert data["season"] == 2025

    def test_write_execution_log_atomic_uses_absolute_path(self, tmp_path):
        """write_execution_log_atomic resolves relative paths."""
        log = ExecutionLog(
            status="running",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=1,
        )
        # Use relative-looking path within tmp_path
        log_path = tmp_path / "test_log.json"
        write_execution_log_atomic(log, log_path)
        assert log_path.exists()

    def test_write_execution_log_atomic_is_atomic(self, tmp_path):
        """write_execution_log_atomic uses os.replace for atomic write."""
        log = ExecutionLog(
            status="success",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=1,
        )
        log_path = tmp_path / "atomic_test.json"
        with patch(
            "pipeline.execution_log.os.replace", wraps=os.replace
        ) as mock_replace:
            write_execution_log_atomic(log, log_path)
            mock_replace.assert_called_once()

    def test_write_execution_log_atomic_overwrites_existing(self, tmp_path):
        """write_execution_log_atomic overwrites existing file."""
        log_path = tmp_path / "overwrite_test.json"

        log1 = ExecutionLog(
            status="running",
            start_time="2025-10-03T17:00:00",
            season=2025,
            week=5,
            pid=1,
        )
        write_execution_log_atomic(log1, log_path)

        log2 = ExecutionLog(
            status="success",
            start_time="2025-10-03T17:00:00",
            end_time="2025-10-03T17:30:00",
            season=2025,
            week=5,
            pid=1,
        )
        write_execution_log_atomic(log2, log_path)

        with open(log_path) as f:
            data = json.load(f)
        assert data["status"] == "success"
