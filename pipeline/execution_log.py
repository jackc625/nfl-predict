"""Execution log models and atomic file writer for the Friday pipeline.

Provides Pydantic models for structured execution logging and an atomic
write function that ensures the log file is never left in a partial state.
"""

import contextlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


class StepLogEntry(BaseModel):
    """Log entry for a single pipeline step execution."""

    name: str
    status: str
    duration_ms: float
    retry_count: int = 0
    error: str | None = None


class ExecutionLog(BaseModel):
    """Top-level execution log for a pipeline run.

    Captures overall run status, timing, per-step results, and metadata
    needed for monitoring and debugging.
    """

    # "running" while in flight; then one terminal value: "success", "failed",
    # "finished_with_skips" (pipeline.steps.RunStatus -- the run completed and dropped at
    # least one game for a post-lock input, D33.2-05), or "degraded" (a non-critical step
    # failed). The RunStatus members' VALUES are these strings, so the enum and the log
    # share one vocabulary.
    status: str
    start_time: str  # ISO format
    end_time: str | None = None
    season: int
    week: int
    total_duration_ms: float = 0
    forced: bool = False
    mode: str = "full"  # "full", "data-only", "predictions-only"
    pid: int
    steps: list[StepLogEntry] = []
    warnings: list[str] = []
    error: str | None = None
    # WHICH games this run dropped under the live-skip rule (D33.2-05), sorted. This file is
    # OVERWRITTEN by every run, so it is the cross-reference, not the record: the durable,
    # append-only record is config/skip_records.jsonl (pipeline/skip_log.py), and this field
    # is what lets an operator reading one file find the other.
    skipped_games: list[str] = []


def write_execution_log_atomic(log: ExecutionLog, path: Path) -> None:
    """Write execution log to disk atomically.

    Uses a temp file + os.replace pattern so that a crash during write
    never leaves a partial JSON file on disk.

    Args:
        log: The execution log model to persist.
        path: Destination file path (resolved to absolute internally).
    """
    # Resolve to absolute path to avoid ambiguity
    resolved = path.resolve()

    # Ensure parent directory exists
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory, then atomically rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved.parent),
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(log.model_dump_json(indent=2))
        # NOTE: We use os.replace deliberately here rather than Path.replace()
        # because Path.replace() has different semantics (it returns a new Path
        # and may not behave atomically on all platforms the same way).
        os.replace(tmp_path, str(resolved))  # noqa: PTH105
    except Exception:
        # Clean up temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
