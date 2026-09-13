"""The run-level corpus lock, proven against a SECOND PROCESS.

WHY A SAME-THREAD DOUBLE-ACQUIRE WOULD NOT BE THIS TEST
-------------------------------------------------------
`save_bronze_snapshot(..., exclusive=True)` prevents two writers creating the same
snapshot FILENAME and nothing more. The corpus-level hazard is somewhere else entirely:
`data.storage.upsert_silver` (`data/storage.py:1110-1123`) reads the existing parquet,
filters it by key, concats and only THEN writes atomically. The WRITE is atomic; the
READ-MODIFY-WRITE is not. Two promoters that each read the pre-state produce two full
frames, and the second write silently discards the first's rows -- with no error, and
with a digest bracket that still reports exactly one CHANGED path, exactly as declared.

Resume determination has the same shape: two runs can each call
`seasons_still_missing()`, see the same gap, and fetch the same season twice, spending
double budget for one snapshot.

The hazard is therefore CROSS-PROCESS, and a single-process check cannot observe it: an
in-process flag, a thread lock or a same-thread double-acquire would all pass against an
implementation that gives no cross-process guarantee at all. So the proof here spawns a
real `subprocess` that holds the lock while the parent attempts to acquire it. Do NOT
downgrade this to a same-thread check.

THE STALE-LOCK RULE IS A REFUSAL, NOT A TAKEOVER
------------------------------------------------
An existing lock makes the run REFUSE, printing the recorded pid, the age and the exact
`--force-unlock` command. The run never breaks a lock automatically on an age heuristic:
a paced 81-minute run legitimately holds the lock for 81 minutes, so any age threshold
short enough to be useful is short enough to break a healthy run, and breaking a healthy
run mid-corpus is the one failure this phase cannot afford.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import pytest

import scripts.backfill_historical_weather as backfill
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# Generous: the child imports pandas, httpx and tenacity before it can take the lock.
CHILD_STARTUP_TIMEOUT_SECONDS = 180.0

_CHILD_SOURCE = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path

    import scripts.backfill_historical_weather as backfill

    root = Path(sys.argv[1])
    release_signal = Path(sys.argv[2])

    with backfill.acquire_corpus_lock(base_path=root):
        print("LOCKED", flush=True)
        deadline = time.monotonic() + 120.0
        while not release_signal.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    print("RELEASED", flush=True)
    """
)


@pytest.fixture
def lock_holder(tmp_path, sandbox_data_root):
    """Spawn a child process that holds the corpus lock until signalled."""
    script = tmp_path / "_hold_corpus_lock.py"
    script.write_text(_CHILD_SOURCE, encoding="utf-8")
    release_signal = tmp_path / "release.signal"

    child = subprocess.Popen(
        [sys.executable, str(script), str(sandbox_data_root), str(release_signal)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + CHILD_STARTUP_TIMEOUT_SECONDS
    line = ""
    while time.monotonic() < deadline:
        line = child.stdout.readline()
        if line.strip() == "LOCKED" or not line:
            break
    if line.strip() != "LOCKED":
        child.kill()
        stderr = child.stderr.read()
        pytest.fail(f"the lock-holding child never acquired the lock.\n{stderr}")

    try:
        yield child
    finally:
        release_signal.touch()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()


class TestASecondProcessCannotTakeTheLock:
    def test_the_parent_refuses_while_a_child_holds_the_lock(
        self, sandbox_data_root, lock_holder
    ):
        with pytest.raises(backfill.CorpusLockedError) as excinfo:
            backfill.acquire_corpus_lock(base_path=sandbox_data_root)

        message = str(excinfo.value)
        assert str(lock_holder.pid) in message, (
            "the refusal does not name the pid recorded in the lock file, so a stale "
            f"lock is mysterious rather than diagnosable. Message: {message}"
        )
        assert "--force-unlock" in message

    def test_the_lock_file_records_the_holders_identity(
        self, sandbox_data_root, lock_holder
    ):
        payload = json.loads(
            backfill.corpus_lock_path(sandbox_data_root).read_text(encoding="utf-8")
        )

        assert payload["pid"] == lock_holder.pid
        assert payload["acquired_at_utc"].endswith("+00:00")
        assert payload["corpus"]["table"] == backfill.BACKFILL_BRONZE_TABLE
        assert payload["corpus"]["first_season"] == backfill.CORPUS_FIRST_SEASON
        assert payload["corpus"]["last_season"] == backfill.CORPUS_LAST_SEASON


class TestTheLockLifecycle:
    def test_a_clean_exit_removes_the_lock(self, sandbox_data_root):
        path = backfill.corpus_lock_path(sandbox_data_root)
        with backfill.CorpusLock(sandbox_data_root):
            assert path.is_file()
        assert not path.exists()

    def test_the_lock_survives_an_exception_inside_the_context(self, sandbox_data_root):
        """A CRASH leaves a CLEARABLE stale lock rather than a vanished one.

        This is threat T-33.1-31c stated as an assertion. An exception mid-corpus means
        the corpus is in an unknown half-written state; a lock that silently removed
        itself would let the very next run proceed straight over it.
        """
        path = backfill.corpus_lock_path(sandbox_data_root)

        with pytest.raises(RuntimeError), backfill.CorpusLock(sandbox_data_root):
            raise RuntimeError("simulated crash mid-corpus")

        assert path.is_file()

    def test_force_unlock_clears_a_stale_lock(self, sandbox_data_root):
        path = backfill.corpus_lock_path(sandbox_data_root)
        with pytest.raises(RuntimeError), backfill.CorpusLock(sandbox_data_root):
            raise RuntimeError("simulated crash mid-corpus")
        assert path.is_file()

        with backfill.acquire_corpus_lock(base_path=sandbox_data_root, force=True):
            assert path.is_file()

        assert not path.exists()

    def test_a_second_acquisition_in_one_process_also_refuses(self, sandbox_data_root):
        """The in-process control. It proves the refusal is reachable at all; the
        cross-process test above is what proves the PROPERTY."""
        with backfill.CorpusLock(sandbox_data_root):
            with pytest.raises(backfill.CorpusLockedError):
                backfill.acquire_corpus_lock(base_path=sandbox_data_root)


class TestTheLockSpansResumeDeterminationAndPromotion:
    """A lock released between the pre-state digest and the promotion protects nothing.

    Both observations are taken by wrapping the module-level names `backfill_corpus`
    actually calls, so the assertion needs no production hook and cannot be satisfied by
    a lock that is merely taken somewhere in the run.
    """

    def test_the_lock_file_exists_at_both_points(self, sandbox_data_root, monkeypatch):
        observed: dict[str, bool] = {}
        lock_path = backfill.corpus_lock_path(sandbox_data_root)

        real_missing = backfill.seasons_still_missing
        real_upsert = backfill.upsert_silver

        def _missing(*args, **kwargs):
            observed["resume_determination"] = lock_path.is_file()
            return real_missing(*args, **kwargs)

        def _upsert(*args, **kwargs):
            observed["promotion"] = lock_path.is_file()
            return real_upsert(*args, **kwargs)

        monkeypatch.setattr(backfill, "seasons_still_missing", _missing)
        monkeypatch.setattr(backfill, "upsert_silver", _upsert)

        snapshot = (
            sandbox_data_root
            / "bronze"
            / f"{backfill.BACKFILL_BRONZE_TABLE}_raw_bronze_2003_W00_20260101T000000"
            ".parquet"
        )
        pd.DataFrame(
            {
                "game_id": ["2003_W01_SF@NYG"],
                "weather_source": ["archive"],
                "forecast_time": [pd.Timestamp("2026-09-13T00:00:00Z")],
                "game_time": [pd.Timestamp("2003-09-08T00:30:00Z")],
                "temp_f": [55.0],
                "temp_c": [12.8],
                "wind_mph": [8.0],
                "wind_direction": [180.0],
                "humidity_pct": [60.0],
                "precip_prob": [None],
                "precip_mm": [0.0],
                "condition": [None],
                "condition_code": [0],
                "visibility_km": [None],
                "dew_point_f": [40.0],
                "apparent_temp_f": [54.0],
                "snowfall_cm": [0.0],
                "wind_gusts_mph": [12.0],
                "cloud_cover_pct": [30.0],
                "is_outdoor": [True],
                "weather_coverage": [True],
                "is_cold": [False],
                "is_windy": [False],
                "is_precipitation": [False],
            }
        ).to_parquet(snapshot, index=False)

        backfill.backfill_corpus(
            base_path=sandbox_data_root,
            fetch=False,
            promote=True,
            verify_archive_floor=False,
        )

        assert observed.get("resume_determination") is True
        assert observed.get("promotion") is True

    def test_the_recorded_contract_names_both_points(self):
        scope = " ".join(
            str(item) for item in phase33_state.CORPUS_LOCK_CONTRACT["scope"]
        ).lower()
        assert "resume determination" in scope
        assert "promotion" in scope
