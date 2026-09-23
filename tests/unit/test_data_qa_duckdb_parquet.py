"""The D30-18 DuckDB-versus-parquet consistency guard, and its own positive control.

WHAT THE GUARD IS FOR (N-01). ``data.storage.load_dataframe(source="auto")`` resolves DuckDB
FIRST whenever the table exists and only falls back to parquet
(``data/storage.py`` ``load_dataframe``). So a DuckDB copy that has fallen behind its parquet
makes every ``upsert_silver`` write since the divergence INVISIBLE to the whole pipeline --
silently, with no error anywhere. That is not hypothetical here: at Phase-30 start the silver
``games`` parquet held 6,499 rows against 6,292 in DuckDB, a 207-row divergence, all of them
season 2025 (``tests/phase30_state.py``, measured by Plan 30-04 Task 2).

WHY THE POSITIVE CONTROL EXISTS. A check that has only ever been observed returning ``pass``
is indistinguishable from a check that CANNOT fail. The hermetic half below constructs the
exact N-01 shape -- a DuckDB copy missing rows the parquet has -- in a temp directory, and
asserts the guard catches it and NAMES the missing ids rather than only reporting a count
mismatch.

TEST CLASS (Plan 30-04's phase-wide rule -- every test module this phase adds declares its
kind in its docstring), and this module has BOTH kinds, deliberately:

* ``TestHermeticPositiveControl`` and ``TestNoLockIsLeftHeld`` are **plain unit tests**. No
  marker, no generated file read, and they pass on a fresh checkout with ``data/``,
  ``artifacts/`` and ``outputs/`` all absent.
* ``TestLiveSilverGames`` and ``TestLiveGoldMirrorWidth`` are **integration / slow**. They
  carry ``@pytest.mark.integration`` and skip cleanly, with a remediation-carrying message,
  when the live lake is absent.

THE LIVE HALF WAS HONEST IN BOTH PHASE STATES, AND IS NOW FLIPPED. Before Plan 30-08's rung-4
re-sync the live check LEGITIMATELY returned ``fail``, so the live half asserted the divergence
SHAPE and the reported counts rather than asserting ``pass``. Plan 30-08 re-synced the stale
copy, so the live half now asserts ``pass`` -- and, IN THE SAME CLASS, re-asserts that the very
same check still returns ``fail`` on an injected divergence. Both must hold: a check that now
only ever passes would be indistinguishable from a check that cannot fail (T-30-24), which is
the whole reason the hermetic control below exists in the first place. The
expected counts come from the git-TRACKED ``tests.phase30_state``, never from the gitignored
``outputs/n01/divergence_before.json`` -- which is absent on any checkout that did not just run
Plan 30-04 Task 2. Where that scratch document IS present it is read only to CROSS-CHECK the
tracked constants, and a disagreement between the two is itself a failure (T-30-52): a scratch
file regenerated out of step with the manifest must never silently supersede it.

DUCKDB HANDLES (T-30-53). Every connection this module opens is opened through a context
manager, as a read-only attachment wherever only reading is needed, and is closed on every path
including the error path. This host is Windows 11, where a handle left open by one test raises
``duckdb.IOException`` for the next opener rather than being tolerated as it would be on Linux
-- which turns a rapid or concurrent suite run into an intermittent failure that looks like a
data problem and is not. ``TestNoLockIsLeftHeld`` proves it directly.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from data.storage import load_dataframe
from scripts.data_qa import (
    _DUCKDB_PARQUET_CONSISTENCY_TABLES,
    GOLD_FEATURE_MATRICES,
    DataQualityMonitor,
)
from tests.phase30_state import (
    N01_DB_ROWS_BEFORE,
    N01_DIVERGENCE_BEFORE,
    N01_PARQUET_ROWS_BEFORE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"
_GOLD_DIR = REPO_ROOT / "data" / "gold"
_SCRATCH_CAPTURE = REPO_ROOT / "outputs" / "n01" / "divergence_before.json"


# ---------------------------------------------------------------------------
# Hermetic fixtures -- a temp parquet and a temp DuckDB, nothing generated read
# ---------------------------------------------------------------------------


def _rows(n: int, start: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"2025_W{i:02d}_AAA@BBB" for i in range(start, start + n)],
            "season": [2025] * n,
            "week": list(range(start, start + n)),
        }
    )


def _bare_monitor() -> DataQualityMonitor:
    """A monitor built WITHOUT ``__init__``.

    The consistency check must touch no instance state -- no settings, no shared DuckDB
    singleton, no read path. Driving it from a bare instance proves that mechanically and
    keeps the positive control hermetic: nothing here needs a configured lake.
    """
    return DataQualityMonitor.__new__(DataQualityMonitor)


def _injected_loader(tmp_path: Path, parquet: pd.DataFrame, db: pd.DataFrame):
    """Return a ``load_dataframe``-shaped loader over a temp parquet and a temp DuckDB."""
    parquet_path = tmp_path / "silver_games.parquet"
    parquet.to_parquet(parquet_path, index=False)

    db_path = tmp_path / "hermetic.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.register("incoming", db)
        con.execute("CREATE TABLE games AS SELECT * FROM incoming")

    def loader(
        table_name: str, layer: str = "silver", source: str = "auto"
    ) -> pd.DataFrame:
        if source == "parquet":
            return pd.read_parquet(parquet_path)
        if source == "db":
            with duckdb.connect(str(db_path), read_only=True) as con:
                return con.execute(f"SELECT * FROM {table_name}").df()
        raise ValueError(f"Invalid source: {source}")

    return loader


# ---------------------------------------------------------------------------
# The positive control -- proof the guard can fail
# ---------------------------------------------------------------------------


class TestHermeticPositiveControl:
    """The N-01 shape, injected: a DuckDB copy missing rows the parquet has."""

    def test_a_duckdb_copy_missing_two_rows_is_caught_and_the_ids_named(
        self, tmp_path: Path
    ):
        parquet = _rows(10)
        db = parquet.iloc[:8]
        result = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, db)
        )

        entry = result["checks"]["games"]
        assert entry["status"] == "fail"
        assert entry["parquet_rows"] == 10
        assert entry["db_rows"] == 8
        assert entry["only_in_parquet_count"] == 2
        assert entry["only_in_duckdb_count"] == 0
        assert entry["only_in_parquet_sample"] == sorted(
            parquet["game_id"].iloc[8:].tolist()
        ), "the guard must NAME the missing ids, not only report a count mismatch"

    def test_the_result_follows_the_house_per_check_contract(self, tmp_path: Path):
        parquet = _rows(10)
        result = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, parquet)
        )

        assert set(result) >= {"timestamp", "checks"}
        for entry in result["checks"].values():
            assert "status" in entry

    def test_identical_copies_pass(self, tmp_path: Path):
        parquet = _rows(10)
        result = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, parquet)
        )
        assert result["checks"]["games"]["status"] == "pass"

    def test_equal_row_counts_with_different_membership_is_a_failure(
        self, tmp_path: Path
    ):
        """The subtler failure: the integer matches and the rows do not.

        This is the 29-06 lesson applied to rows -- pair the count with a named-delta
        assertion, so a swapped row cannot hide behind a matching total.
        """
        parquet = _rows(10)
        db = pd.concat([parquet.iloc[:9], _rows(1, start=99)], ignore_index=True)
        result = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, db)
        )

        entry = result["checks"]["games"]
        assert entry["db_rows"] == entry["parquet_rows"] == 10
        assert entry["status"] == "fail"
        assert entry["only_in_parquet_count"] == 1
        assert entry["only_in_duckdb_count"] == 1

    def test_a_duckdb_copy_ahead_of_its_parquet_is_also_caught(self, tmp_path: Path):
        parquet = _rows(8)
        db = _rows(10)
        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, db)
        )["checks"]["games"]

        assert entry["status"] == "fail"
        assert entry["only_in_duckdb_count"] == 2

    def test_the_sample_of_missing_ids_is_bounded(self, tmp_path: Path):
        parquet = _rows(200)
        db = parquet.iloc[:1]
        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, db)
        )["checks"]["games"]

        assert entry["only_in_parquet_count"] == 199
        assert len(entry["only_in_parquet_sample"]) <= 20, (
            "a QA report that inlines 199 ids is a report nobody reads"
        )

    def test_the_configured_table_registry_is_data_not_logic(self):
        assert "games" in _DUCKDB_PARQUET_CONSISTENCY_TABLES, (
            "silver games is the table the N-01 divergence was measured on; adding a "
            "future table must not require touching the comparison logic"
        )


class TestAGuardThatCannotRunIsNotAGuardThatPassed:
    """WR-06: only genuine ABSENCE may degrade to the uncounted not_applicable status.

    ``generate_quality_report`` skips ``not_applicable`` when counting
    (``if status in ("expected_gap", "not_applicable"): continue``). The read used to be
    wrapped in a tuple spanning ``duckdb.Error``, ``OSError``, ``ValueError``,
    ``KeyError`` and ``TypeError`` as well as the two absence signals -- so a locked or
    corrupt DuckDB file, the precise failure this guard exists to detect, produced a QA
    report with the check silently missing from the totals and no fail recorded.
    ``TypeError`` and ``KeyError`` are programming errors, so a bug inside the guard was
    indistinguishable from a legitimately inapplicable check.
    """

    @staticmethod
    def _raising_loader(exc: Exception):
        def loader(table_name: str, layer: str = "silver", source: str = "auto"):
            raise exc

        return loader

    def test_a_genuinely_absent_table_is_still_not_applicable(self):
        from utils.exceptions import DataIngestionError

        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=self._raising_loader(
                DataIngestionError("Table games not found in DB or Parquet")
            )
        )["checks"]["games"]

        assert entry["status"] == "not_applicable"

    def test_a_missing_file_is_still_not_applicable(self):
        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=self._raising_loader(FileNotFoundError("games.parquet"))
        )["checks"]["games"]

        assert entry["status"] == "not_applicable"

    @pytest.mark.parametrize(
        "exc",
        [
            duckdb.Error("database is locked"),
            OSError("permission denied"),
            ValueError("Invalid source: duckdb"),
            KeyError("game_id"),
            TypeError("unsupported operand"),
        ],
        ids=["duckdb_error", "oserror", "valueerror", "keyerror", "typeerror"],
    )
    def test_anything_that_is_not_absence_is_a_FAILURE(self, exc: Exception):
        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=self._raising_loader(exc)
        )["checks"]["games"]

        assert entry["status"] == "fail", (
            f"{type(exc).__name__} degraded to {entry['status']!r}. "
            "generate_quality_report does not count not_applicable, so this check "
            "would vanish from the totals with no failure recorded -- and a locked or "
            "corrupt store is exactly what this guard exists to catch."
        )
        assert "guard that cannot run is not a guard that passed" in entry["message"]

    def test_a_table_present_in_both_stores_but_missing_its_key_is_reported_not_raised(
        self, tmp_path: Path
    ):
        """WR-06: the key access sat OUTSIDE the try, so this aborted the whole report."""
        keyless = _rows(4).drop(columns=["game_id"])
        parquet_path = tmp_path / "silver_games.parquet"
        keyless.to_parquet(parquet_path, index=False)

        def loader(table_name: str, layer: str = "silver", source: str = "auto"):
            return pd.read_parquet(parquet_path)

        result = _bare_monitor().check_duckdb_parquet_consistency(loader=loader)

        entry = result["checks"]["games"]
        assert entry["status"] == "fail"
        assert "KeyError" in entry["message"]

    def test_one_broken_table_does_not_abort_the_whole_report(self, tmp_path: Path):
        """The gold-width arm must still run when the membership arm fails."""
        parquet = _rows(4)
        calls: list[tuple[str, str]] = []

        def loader(table_name: str, layer: str = "silver", source: str = "auto"):
            calls.append((table_name, layer))
            if table_name == "games":
                raise duckdb.Error("database is locked")
            return parquet

        result = _bare_monitor().check_duckdb_parquet_consistency(loader=loader)

        assert result["checks"]["games"]["status"] == "fail"
        for matrix in GOLD_FEATURE_MATRICES:
            assert f"{matrix}_width" in result["checks"], (
                "a failure on one table aborted the rest of the consistency report"
            )


class TestNoLockIsLeftHeld:
    """T-30-53: on Windows an unclosed handle blocks the next opener outright."""

    def test_the_check_can_be_run_twice_back_to_back(self, tmp_path: Path):
        parquet = _rows(10)
        loader = _injected_loader(tmp_path, parquet, parquet.iloc[:8])

        first = _bare_monitor().check_duckdb_parquet_consistency(loader=loader)
        second = _bare_monitor().check_duckdb_parquet_consistency(loader=loader)

        assert first["checks"]["games"]["status"] == "fail"
        assert second["checks"]["games"]["status"] == "fail"

    def test_a_writer_can_still_open_the_database_afterwards(self, tmp_path: Path):
        parquet = _rows(10)
        loader = _injected_loader(tmp_path, parquet, parquet)
        _bare_monitor().check_duckdb_parquet_consistency(loader=loader)

        with duckdb.connect(str(tmp_path / "hermetic.duckdb")) as con:
            con.execute("CREATE TABLE proof AS SELECT 1 AS a")
        assert (tmp_path / "hermetic.duckdb").exists()


# ---------------------------------------------------------------------------
# The live half -- honest in BOTH phase states
# ---------------------------------------------------------------------------


def _require_live_silver_games() -> None:
    if not _SILVER_GAMES.exists():
        pytest.skip(
            f"live silver games not present at {_SILVER_GAMES} -- data/ is gitignored "
            "runtime state. Run the ingestion pipeline (see PIPELINE.md) to populate it."
        )


@pytest.mark.integration
class TestLiveSilverGames:
    """N-01 is CLOSED: the live check now passes, and can still fail."""

    def test_the_live_check_now_reports_pass_for_silver_games(self):
        _require_live_silver_games()
        entry = DataQualityMonitor().check_duckdb_parquet_consistency()["checks"][
            "games"
        ]

        assert entry["status"] == "pass", (
            "the N-01 divergence is back. Plan 30-08 rung 4 re-synced the DuckDB copy of "
            f"silver games from its authoritative parquet; this reads {entry['db_rows']} "
            f"against {entry['parquet_rows']}, with {entry['only_in_parquet_count']} rows "
            f"only in parquet and {entry['only_in_duckdb_count']} only in DuckDB. "
            "load_dataframe(source='auto') prefers DuckDB, so a re-divergence makes every "
            "upsert_silver write since it invisible to the whole pipeline, silently. "
            "Re-sync with `python -m scripts.resync_games_duckdb --apply`."
        )
        # RE-ANCHORED (Plan 33.2-20). This pinned both copies to the parquet's
        # PRE-re-sync count. The live 2026 capture has since ingested 272 games into
        # the parquet -- through upsert_silver, not through the re-sync, which writes
        # the parquet not at all -- so the count moved for a reason that is not a
        # divergence. The claim was that the re-sync brought the STALE copy up and left
        # the AUTHORITATIVE one alone, and the durable form is: the two AGREE, and the
        # authoritative copy never went below what it held before. A copy that fell
        # behind again, or a parquet that lost rows, still fails.
        assert entry["db_rows"] == entry["parquet_rows"]
        assert entry["parquet_rows"] >= N01_PARQUET_ROWS_BEFORE, (
            f"the authoritative parquet reads {entry['parquet_rows']} against the "
            f"{N01_PARQUET_ROWS_BEFORE} it held before the re-sync -- fewer rows than "
            "it started with, which no ingest produces"
        )
        assert entry["only_in_parquet_count"] == 0
        assert entry["only_in_duckdb_count"] == 0, (
            "N-01 was a mirror that fell BEHIND; rows present only in DuckDB would be a "
            "different and worse defect"
        )

    def test_the_check_that_now_passes_is_still_capable_of_failing(
        self, tmp_path: Path
    ):
        """T-30-24, and this is why it lives in the LIVE class rather than only above.

        The moment a check flips from 'observed failing' to 'observed passing' it stops
        carrying its own evidence: a passing check and a check that cannot fail are
        indistinguishable from the outside. So the flip to a pass assertion above is
        paired, in the same class, with the same check being driven to a fail on an
        injected divergence. Both must hold or the guard has become decorative.
        """
        parquet = _rows(10)
        entry = _bare_monitor().check_duckdb_parquet_consistency(
            loader=_injected_loader(tmp_path, parquet, parquet.iloc[:8])
        )["checks"]["games"]

        assert entry["status"] == "fail"
        assert entry["only_in_parquet_count"] == 2

    def test_the_pre_resync_divergence_is_still_recorded_and_was_real(self):
        """The closed defect keeps its measurement; a fixed bug is not an unrecorded one."""
        assert N01_PARQUET_ROWS_BEFORE - N01_DB_ROWS_BEFORE == N01_DIVERGENCE_BEFORE
        assert N01_DIVERGENCE_BEFORE > 0

    def test_the_scratch_capture_agrees_with_the_tracked_manifest(self):
        """T-30-52: a regenerated scratch file must not silently supersede the manifest."""
        if not _SCRATCH_CAPTURE.exists():
            pytest.skip(
                f"{_SCRATCH_CAPTURE} absent -- it is gitignored scratch (.gitignore:26), "
                "written by Plan 30-04 Task 2. The tracked manifest is the source of "
                "truth; this test only cross-checks the scratch copy when it is present."
            )
        captured = json.loads(_SCRATCH_CAPTURE.read_text(encoding="utf-8"))

        assert captured["parquet_rows"] == N01_PARQUET_ROWS_BEFORE
        assert captured["db_rows"] == N01_DB_ROWS_BEFORE
        assert captured["divergence"] == N01_DIVERGENCE_BEFORE, (
            "outputs/n01/divergence_before.json disagrees with tests/phase30_state.py. "
            "One of them was regenerated out of step with the other. The TRACKED manifest "
            "is the expectation; do not edit it to match a re-run."
        )

    def test_the_manifest_divergence_is_strictly_positive(self):
        assert N01_DIVERGENCE_BEFORE > 0, (
            "a zero divergence means the re-sync already happened, leaving Plan 30-08's "
            "positive control with nothing to prove"
        )
        assert N01_PARQUET_ROWS_BEFORE - N01_DB_ROWS_BEFORE == N01_DIVERGENCE_BEFORE


@pytest.mark.integration
class TestLiveGoldMirrorWidth:
    """The gold DuckDB mirror and the gold parquet are read by DIFFERENT consumers."""

    def test_all_three_matrices_agree_on_width_across_both_stores(self):
        missing = [
            matrix
            for matrix in GOLD_FEATURE_MATRICES
            if not (_GOLD_DIR / f"{matrix}.parquet").exists()
        ]
        if missing:
            pytest.skip(
                f"live gold absent ({', '.join(missing)}) -- run "
                "`python -m scripts.build_features --all` to populate data/gold."
            )

        result = DataQualityMonitor().check_duckdb_parquet_consistency()
        for matrix in GOLD_FEATURE_MATRICES:
            entry = result["checks"][f"{matrix}_width"]
            assert entry["status"] == "pass", (
                f"{matrix}'s DuckDB mirror is {entry['db_width']} columns wide against "
                f"{entry['parquet_width']} in the parquet. check_gold_integrity reads gold "
                "through the DuckDB-preferring auto path while models/train.py, "
                "backtest/diagnose.py, backtest/signal_lift.py and "
                "scripts/fingerprint_gold.py all read the parquet directly -- so the width "
                "tripwire and the training path would be checking different artifacts."
            )

    def test_the_width_check_covers_every_gold_matrix(self):
        assert len(GOLD_FEATURE_MATRICES) == 3

    def test_the_shared_read_path_is_untouched(self):
        """D30-18: the guard sits where the pipeline already looks; it changes nothing.

        The ``upsert_silver`` / ``load_dataframe`` write-path asymmetry is explicitly out of
        scope for this phase. Fixing the read path would be a change to the seam every
        builder, trainer and backtest goes through, made in the same phase that is trying to
        measure a gold rebuild -- exactly the confound this phase is built to avoid.
        """
        assert load_dataframe.__module__ == "data.storage"
        parameters = inspect.signature(load_dataframe).parameters
        defaults = (parameters["layer"].default, parameters["source"].default)
        assert defaults == ("silver", "auto"), (
            "load_dataframe's defaults moved. The D30-18 guard exists precisely BECAUSE "
            "source defaults to 'auto' and 'auto' prefers DuckDB; a change here changes "
            "what the whole pipeline reads."
        )
