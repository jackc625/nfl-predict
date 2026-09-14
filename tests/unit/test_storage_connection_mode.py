"""QT-W8X-01: a caller that only READS never takes a read-write DuckDB lock.

WHAT THIS MODULE IS FOR
-----------------------
``data/storage.py`` connected with no ``read_only`` flag, which is read-WRITE, and the
connection is a module global held for the whole of any process that touched
``load_dataframe``. MEASURED on duckdb 1.5.0 / Windows 11:

===============  ===============  ========================================
holder           second opener    result
===============  ===============  ========================================
read-only        read-only        both OPEN, concurrently
read-write       read-only        ``IOException: ... used by another process``
read-write       read-write       ``IOException: ... used by another process``
===============  ===============  ========================================

So a pytest session that merely READ the production store made that store unopenable
by anything else -- including a plain ``open(path, "rb")``, which is why
``tests/data_boundary.digest_file`` grew a metadata fallback in the first place.

The fix is a DEFAULT, not a heuristic: ``DuckDBConnection.connect()`` takes the safe
mode, and a caller that needs to write says ``connect(write=True)`` at the call site.
Write intent is NEVER inferred from the text of a query -- an implicit default that
reads the SQL is the same implicit default this module exists to remove, and
``TestWriteIntentIsNamedAndNeverSniffed`` asserts against it by AST.

EVERY DATABASE HERE IS UNDER ``tmp_path``, with exactly one named exception:
``TestTheProductionPathIsOpenedReadOnly`` READS the real
``data/nfl_predictions.duckdb`` and writes nothing. It carries no
``writes_production_store`` marker and must never need one -- if it ever does, the
default regressed.

TEST CLASS: plain unit test.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from data.storage import DuckDBConnection
from utils import DataIngestionError

REPO_ROOT = Path(__file__).resolve().parents[2]

# A second OS process, because the property under test is a cross-process file lock
# and it cannot be shown in-process: duckdb caches the database instance per process
# and answers the second connect from the cache.
_SECOND_OPENER = """
import sys

import duckdb

try:
    connection = duckdb.connect(sys.argv[1], read_only=True)
    connection.execute("SELECT 1").fetchall()
    connection.close()
except Exception as exc:  # noqa: BLE001
    sys.stdout.write("SECOND-OPENER-BLOCKED:" + type(exc).__name__)
    raise SystemExit(1) from exc
sys.stdout.write("SECOND-OPENER-OPENED")
"""


def _second_process_opens_read_only(db_path: Path) -> subprocess.CompletedProcess:
    """Try to open *db_path* read-only from another OS process."""
    return subprocess.run(
        [sys.executable, "-c", _SECOND_OPENER, str(db_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """An EXISTING duckdb file with one table, closed before the test starts."""
    path = tmp_path / "seeded.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE seeded AS SELECT 1 AS a")
    connection.close()
    return path


@pytest.fixture
def restore_global_connection():
    """Save, and afterwards restore, ``data.storage``'s module-global connection.

    The production-path test reads through the global, so without this a mode it
    left behind would leak into whatever runs next in the session.
    """
    from data import storage

    previous = storage._db_connection
    yield
    if storage._db_connection is not None and storage._db_connection is not previous:
        storage._db_connection.close()
    storage._db_connection = previous


# ---------------------------------------------------------------------------
# 1-2. The lock matrix, measured across processes
# ---------------------------------------------------------------------------


class TestTheLockMatrixAcrossProcesses:
    """READ+READ share the file. WRITE excludes everyone. Both halves asserted."""

    def test_a_read_handle_lets_a_second_process_open_the_same_file(
        self, seeded_db: Path
    ) -> None:
        holder = DuckDBConnection(str(seeded_db))
        holder.connect()
        try:
            completed = _second_process_opens_read_only(seeded_db)
        finally:
            holder.close()

        assert completed.returncode == 0, (
            "a second process could NOT open a database this one is only READING. "
            "That is the whole defect: a read-only workload was holding an exclusive "
            f"write lock.\n\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
        assert "SECOND-OPENER-OPENED" in completed.stdout, completed.stdout

    def test_a_write_handle_still_excludes_a_second_process(
        self, seeded_db: Path
    ) -> None:
        """Today's behaviour, and it must REMAIN true for a genuine writer.

        This case is not a regression test on the lock; it is what makes the case
        above mean something. If both modes let a second process in, the first test
        would be measuring the filesystem rather than the connection mode.
        """
        holder = DuckDBConnection(str(seeded_db))
        holder.connect(write=True)
        try:
            completed = _second_process_opens_read_only(seeded_db)
        finally:
            holder.close()

        assert completed.returncode != 0, (
            "a second process opened a database held for WRITING. duckdb's exclusive "
            f"lock is what serialises writers.\n\nstdout: {completed.stdout}"
        )
        assert "SECOND-OPENER-BLOCKED" in completed.stdout, completed.stdout


# ---------------------------------------------------------------------------
# 3. A read handle REFUSES a write
# ---------------------------------------------------------------------------


class TestAReadHandleRefusesAWrite:
    def test_a_create_table_through_a_read_handle_raises(self, seeded_db: Path) -> None:
        """Silently succeeding would make the read-only mode decorative."""
        connection = DuckDBConnection(str(seeded_db))
        try:
            with pytest.raises(DataIngestionError):
                connection.execute("CREATE TABLE sneaked AS SELECT 2 AS b")
        finally:
            connection.close()

        # And the write really did not land.
        verifier = duckdb.connect(str(seeded_db), read_only=True)
        try:
            tables = {
                row[0]
                for row in verifier.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
        finally:
            verifier.close()
        assert "sneaked" not in tables, (
            "the CREATE raised but the table exists. The exception would then be "
            "reporting something other than what happened on disk."
        )


# ---------------------------------------------------------------------------
# 4. The upgrade: read first, then write on the SAME object
# ---------------------------------------------------------------------------


class TestTheUpgradeFromReadToWrite:
    def test_a_read_then_a_write_on_one_object_succeeds_and_reports_write_mode(
        self, seeded_db: Path
    ) -> None:
        """Close-then-reopen is REQUIRED here, not stylistic.

        MEASURED on duckdb 1.5.0: a second connection onto one file under a different
        configuration raises ``ConnectionException: Can't open a connection to same
        database file with a different configuration than existing connections``. So
        the read handle can be neither reused for the write nor held alongside a write
        handle; it has to be closed.
        """
        connection = DuckDBConnection(str(seeded_db))

        assert connection.table_exists("seeded") is True
        assert connection.connection_mode == "read", (
            "a plain read took a write handle. The default is the fix."
        )

        connection.create_table_from_df(
            pd.DataFrame({"b": [1, 2, 3]}), "upgraded", if_exists="replace"
        )
        assert connection.connection_mode == "write", (
            "the object still reports read mode after a write. Either the write went "
            "through a read handle or the recorded mode is not tracking reality."
        )
        assert connection.table_exists("upgraded") is True

        # The write handle is exclusive, exactly as case 2 showed.
        completed = _second_process_opens_read_only(seeded_db)
        connection.close()
        assert completed.returncode != 0, (
            "the upgraded handle is not actually a write handle -- a second process "
            f"opened the file while it was held.\n\nstdout: {completed.stdout}"
        )


# ---------------------------------------------------------------------------
# 5. A MISSING database in read mode still opens
# ---------------------------------------------------------------------------


class TestAMissingDatabaseStillOpensInReadMode:
    def test_a_fresh_temp_root_opens_and_answers_table_exists_as_false(
        self, tmp_path: Path
    ) -> None:
        """The sandbox path ~770 integration tests depend on.

        MEASURED: ``duckdb.connect(path, read_only=True)`` on a file that does not
        exist raises ``IOException: Cannot open database ... in read-only mode:
        database does not exist``. There is nothing to protect in an absent file, so
        the read path keeps today's create-on-open behaviour and a `load_dataframe`
        against a fresh root still falls through to parquet exactly as before.
        """
        missing = tmp_path / "nested" / "absent.duckdb"
        connection = DuckDBConnection(str(missing))
        try:
            assert connection.table_exists("anything") is False
            assert missing.exists(), (
                "the read path on a missing database did not create it. Every "
                "sandboxed test that writes to a fresh temp root depends on this."
            )
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# 6. close() resets the mode -- the de-escalation seam
# ---------------------------------------------------------------------------


class TestCloseResetsTheMode:
    def test_close_then_connect_yields_a_read_handle_again(
        self, seeded_db: Path
    ) -> None:
        """This is what makes ``close_db_connection`` a real de-escalation.

        ``tests/data_boundary.close_probable_holders`` calls
        ``data.storage.close_db_connection`` at every guard checkpoint. If ``close()``
        left the recorded mode at write, the next lazy open would re-take the
        exclusive lock and the seam would be a no-op.
        """
        connection = DuckDBConnection(str(seeded_db))
        connection.create_table_from_df(
            pd.DataFrame({"c": [1]}), "written", if_exists="replace"
        )
        assert connection.connection_mode == "write"

        connection.close()
        assert connection.connection_mode is None, (
            "a closed connection still reports a mode"
        )

        connection.connect()
        try:
            assert connection.connection_mode == "read", (
                "the next lazy open after close() took a write handle again, so "
                "closing handles does not de-escalate anything."
            )
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# 7. THE PRODUCTION PATH. Reads only. Carries no marker and must not need one.
# ---------------------------------------------------------------------------


class TestTheProductionPathIsOpenedReadOnly:
    def test_a_pure_read_through_load_dataframe_leaves_a_read_handle(
        self, restore_global_connection
    ) -> None:
        from data import storage

        production = REPO_ROOT / "data" / "nfl_predictions.duckdb"
        if not production.exists():
            pytest.skip(
                "data/nfl_predictions.duckdb is gitignored and not present at "
                f"{production}"
            )

        # Start from a closed global so the mode measured is the one this read took,
        # not one a previous test in the session left behind.
        storage.close_db_connection()

        frame = storage.load_dataframe(
            "games", layer="silver", source="db", columns=["game_id"]
        )
        assert not frame.empty, "the production read returned nothing to judge"

        connection = storage.get_db_connection()
        assert connection.connection_mode == "read", (
            "a pytest session that only READ the production store is holding a "
            "read-write lock on it. Nothing else -- not another test process, not "
            f"the content-digest guard -- can open it.\n\nmode: "
            f"{connection.connection_mode!r}"
        )

    def test_the_held_read_handle_does_not_block_a_plain_byte_read(
        self, restore_global_connection
    ) -> None:
        """MEASURED, and it is why ``digest_file``'s fallback stops firing.

        duckdb 1.5.0 / Windows 11: a read-only handle leaves the file openable with
        ``open(path, "rb")``; a read-write handle raises ``PermissionError``. The
        content-digest guard's whole claim is content-based evidence, so which of
        those two the ordinary session holds decides whether the guard can prove
        anything about this file at all.
        """
        from data import storage

        production = REPO_ROOT / "data" / "nfl_predictions.duckdb"
        if not production.exists():
            pytest.skip(
                "data/nfl_predictions.duckdb is gitignored and not present at "
                f"{production}"
            )

        storage.close_db_connection()
        storage.get_db_connection().connect()

        with production.open("rb") as handle:
            head = handle.read(4096)
        assert head, "the production store read back zero bytes"


# ---------------------------------------------------------------------------
# 8. SOURCE CONTRACT: write intent is NAMED, never sniffed from SQL
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = (
    "CREATE",
    "DROP",
    "INSERT",
    "UPDATE",
    "DELETE",
    "ALTER",
    "TRUNCATE",
    "REPLACE",
    "BEGIN",
    "COMMIT",
    "VACUUM",
    "ANALYZE",
    "ATTACH",
    "COPY",
)


def _connect_ast() -> ast.FunctionDef:
    """``connect``'s AST with its DOCSTRING REMOVED.

    The docstring says in words what the function does -- "get or create a
    connection" -- and the word CREATE is one of the SQL keywords scanned for below.
    Scanning the prose that explains the behaviour rather than the behaviour is the
    same mistake a substring scan makes, one level in.
    """
    import textwrap

    source = textwrap.dedent(inspect.getsource(DuckDBConnection.connect))
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    if ast.get_docstring(function):
        function.body = function.body[1:]
    return function


class TestWriteIntentIsNamedAndNeverSniffed:
    def test_connect_takes_a_keyword_only_write_parameter_defaulting_to_false(
        self,
    ) -> None:
        signature = inspect.signature(DuckDBConnection.connect)
        write = signature.parameters.get("write")
        assert write is not None, "connect has no `write` parameter"
        assert write.kind is inspect.Parameter.KEYWORD_ONLY, (
            "`write` is positional. A positional boolean at a call site reads as "
            "`connect(True)`, which says nothing about what it grants."
        )
        assert write.default is False, (
            "the default is not False. The default IS the fix: a caller that says "
            "nothing gets the safe mode."
        )

    def test_connect_never_decides_write_intent_by_reading_sql_text(self) -> None:
        """A regex over query text is the implicit default this task removes.

        Scanned by AST rather than by substring: the prose in this module names
        every SQL keyword below, and a substring scan would flag its own rationale.
        """
        function = _connect_ast()

        literals = {
            node.value.upper()
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        offenders = sorted(
            keyword
            for keyword in _SQL_KEYWORDS
            if any(keyword in literal for literal in literals)
        )
        assert not offenders, (
            f"connect's source mentions the SQL keyword(s) {offenders}. Write "
            "capability must be granted by the `write=` argument at the call site "
            "and never inferred from what a query happens to say."
        )

        comparisons = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Compare)
            for op in node.ops
            if isinstance(op, ast.In | ast.NotIn)
        ]
        for node in comparisons:
            operands = [node.left, *node.comparators]
            assert not any(
                isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                for operand in operands
            ), (
                "connect performs a membership test against a string literal. That "
                "is SQL-text sniffing however it is spelled."
            )


# ---------------------------------------------------------------------------
# 9. The session settings on a READ handle -- probed, not assumed
# ---------------------------------------------------------------------------


class TestTheSessionSettingsSurviveOnAReadHandle:
    def test_both_settings_are_still_issued_and_take_effect_read_only(
        self, seeded_db: Path
    ) -> None:
        """PROBED on duckdb 1.5.0 before this test was written.

        Both ``SET memory_limit='4GB'`` and ``SET threads=4`` are ACCEPTED on a
        read-only handle -- neither is a write to the database, both are session
        configuration -- so the read path issues them unchanged and there is no skip
        to assert. Were either rejected, the read path would have to skip it with a
        comment naming which and why; it is not, so it does not.
        """
        connection = DuckDBConnection(str(seeded_db))
        handle = connection.connect()
        try:
            assert connection.connection_mode == "read"
            threads = handle.execute("SELECT current_setting('threads')").fetchone()
            memory = handle.execute("SELECT current_setting('memory_limit')").fetchone()
        finally:
            connection.close()

        assert threads is not None and threads[0] == 4, (
            f"SET threads=4 did not take effect on the read handle: {threads!r}"
        )
        assert memory is not None and "GiB" in str(memory[0]), (
            f"SET memory_limit did not take effect on the read handle: {memory!r}"
        )
