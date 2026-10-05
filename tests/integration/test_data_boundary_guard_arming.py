"""COLD-05: the production-store write guard is ARMED BY DEFAULT, and it fires.

WHAT THIS PROVES, AND WHY IT NEEDED A NESTED SESSION
----------------------------------------------------
``tests/data_boundary.py`` has existed since Plan 31-11 and is OPT-IN per module: a
module that must not touch a production store requests ``data_boundary_guard``
explicitly. That design has one failure mode and this phase is paying for it -- a NEW
test is UNGUARDED BY DEFAULT. Four production overwrites went unreported during Phase 31
for exactly that reason.

Plan 33-01 arms the guard for every test in the session. Proving that cannot be done from
inside the session it is proving: a test that deliberately writes a production store to
watch the guard fire would BE the boundary crossing. So this module drives a SECOND,
nested pytest session against a generated mini-suite whose guarded roots are redirected
onto ``tmp_path``, and asserts on the child's exit status and captured output.

HARNESS RULING (Plan 33-01 Task 1(e), answering the Codex MEDIUM review finding).
``pytester`` is NOT enabled in this repository: ``tests/conftest.py`` declares no
``pytest_plugins`` and ``pyproject.toml``'s ``[tool.pytest.ini_options]`` does not load
it. Enabling it repo-wide to serve one module would change every session's plugin set,
which is an uninstructed change. A recursive ``pytest.main()`` inside a running session
is worse still -- it shares the parent's plugin manager and fixture state, which is the
exact contamination risk the reviewer named. The child therefore runs as a SUBPROCESS:
``subprocess.run([sys.executable, "-m", "pytest", ...])``. A child process cannot
contaminate the parent, and its stdout is the evidence.

THE GENERATED MINI-SUITE IS SEVERAL MODULES, NOT ONE (QT-W8X-02)
----------------------------------------------------------------
The guard now sweeps once per MODULE rather than once per test -- a measured 36.7 ms x
4,872 tests, ~179 s, 18% of a whole-suite run, given back at the cost of file-level
rather than test-level attribution. The mini-suite had to be re-authored to match: six
cases in ONE child module would collapse into a SINGLE teardown error and every per-case
assertion would go vacuous while still passing. So each claim gets its own child module,
and the report is keyed by the module the guard's own header names.

They run in filename order (``-p no:randomly``), and the ORDER IS PART OF THE PROOF: the
module carrying the marker runs FIRST, so every later module demonstrates that one
file's declaration stood the guard down for nobody else.

1. ``test_1_marked_writer.py`` -- a MARKED test writes its DECLARED path (permitted) and
   an unmarked sibling in the SAME file writes a DIFFERENT path. The module FAILS, and
   the report names the undeclared path ALONE. Module scope is a union of declared
   PATHS, never a licence for the file.
2. ``test_2_accepted_loss.py`` -- THE ACCEPTED LOSS, recorded as a test rather than as
   prose. One test declares ``data/nfl_predictions.duckdb``; a DIFFERENT unmarked test
   in the same file writes that SAME path and is now PERMITTED. That is precisely what
   module granularity costs and the owner accepted it explicitly.
3. ``test_3_wave6_overwrite.py`` -- THE PHASE 33 WAVE 6 INCIDENT, reproduced. A module
   with NO marker anywhere in it overwrites all three production gold matrices, exactly
   as a test that omitted the ``gold_lake`` sandbox fixture did. The session FAILS, the
   report names all three matrices, and it names the module that wrote them.
4. ``test_4_creation_and_deletion.py`` -- ADDED and REMOVED are still reported as their
   own kinds, with digests.
5. ``test_5_metadata_restoring.py`` -- a write that restores its own ``st_size`` and
   ``st_mtime_ns`` is INVISIBLE to the per-module stat prefilter and is STILL caught, by
   the session-end full content sweep. Two assertions, because the point of the case is
   WHICH instrument catches it. This case is unchanged in substance and is the backstop
   the whole speed/attribution trade rests on.

THE THIRD ROOT, ``ledger`` (Plan 34-02, D-10)
--------------------------------------------
Phase 34 stores the forward bet ledger as JSON lines under a new top-level ``ledger/``
directory, which is itself a git repository pushed to a private backup (D-01). Two more
nested sessions prove the guard watches it, with every root redirected onto ``tmp_path``:

6. ``test_6_ledger_store_write.py`` -- an unmarked module appends to
   ``forward_2026.jsonl``. The session FAILS and the report names the module, the test
   and the ``.jsonl`` file. ``.jsonl`` is tracked under the ledger root ONLY; the data and
   artifacts roots keep the old suffix set so existing digest documents keep their
   meaning.
7. ``test_7_ledger_git_internals.py`` -- in the SAME session, a module creates a new file
   under ``.git/objects/`` and rewrites an existing file under ``.git/``. Both carry
   tracked suffixes on purpose, so only the ``.git`` exclusion -- not the suffix filter --
   can be what keeps them out. Nothing is reported, by the per-module pass or by the
   closing sweep: the backup repository's internals are not the store.
8. A separate session whose one module writes nothing exits 0 -- the third root adds no
   false positive.

TEST CLASS: **integration**. It spawns a real pytest process. It writes nothing outside
``tmp_path``; the ``data_boundary_guard`` request on every test is the belt-and-braces
control that says so, the same request the already-sandboxed
``tests/integration/test_elo_integration.py`` uses.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA256_RE = re.compile(r"\b[0-9a-f]{64}\b")

# The guard's own header is what keys the report now. Under module scope pytest heads
# the block `ERROR at teardown of <test>` naming whichever test happened to run LAST in
# the file -- an innocent one -- so parsing THAT would attribute every finding to the
# wrong place. The guard states the module itself; this reads what it states.
_GUARD_MODULE_HEADER_RE = re.compile(r"PRODUCTION STORE WRITE GUARD -- MODULE (\S+) ")

# Anything that ends a per-module report region. `-{4,}` is a `write_sep("-", ...)` rule
# from the terminal summary; `_{4,}` is the next failure/error block header; `=` opens
# the session summary.
_SEPARATOR_RULE_RE = re.compile(r"^[-_]{4,}")

CLOSING_SWEEP_HEADER = "SESSION-END FULL CONTENT SWEEP"

MARKED_MODULE = "test_1_marked_writer.py"
ACCEPTED_LOSS_MODULE = "test_2_accepted_loss.py"
WAVE6_MODULE = "test_3_wave6_overwrite.py"
ADD_REMOVE_MODULE = "test_4_creation_and_deletion.py"
METADATA_MODULE = "test_5_metadata_restoring.py"

# Eight generated tests across five generated modules. Pinned so a mini-suite that
# silently stopped collecting cannot leave every assertion below vacuously true.
EXPECTED_CHILD_TESTS = 8

_CHILD_CONFTEST = '''\
"""Generated mini-suite conftest: borrow the REAL guard, guard a sandbox root."""

import sys

sys.path.insert(0, {repo_root!r})

from tests.conftest import (  # noqa: E402,F401
    _production_store_baseline,
    production_store_write_guard,
    pytest_sessionfinish,
    pytest_terminal_summary,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "writes_production_store: this test legitimately writes the production "
        "store paths named in its paths= argument (COLD-05, D33-24)",
    )
'''

_CHILD_MODULES: dict[str, str] = {
    MARKED_MODULE: '''\
"""A declared write, and an undeclared one, in the SAME file."""

import os
import pathlib

import pytest

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])


@pytest.mark.writes_production_store(paths=["data/nfl_predictions.duckdb"])
def test_marked_writer_is_permitted():
    (DATA / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v2")


def test_an_unmarked_sibling_writing_a_DIFFERENT_path_is_still_rejected():
    (DATA / "silver" / "games.parquet").write_bytes(b"GAMES-REWRITTEN-BY-A-TEST")
''',
    ACCEPTED_LOSS_MODULE: '''\
"""THE ACCEPTED LOSS. Module granularity exempts the PATH for the whole file."""

import os
import pathlib

import pytest

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])


@pytest.mark.writes_production_store(paths=["data/nfl_predictions.duckdb"])
def test_the_declaring_test_writes_what_it_declared():
    (DATA / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v3")


def test_an_unmarked_sibling_writing_the_SAME_declared_path_is_now_permitted():
    (DATA / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v4-UNMARKED")
''',
    WAVE6_MODULE: '''\
"""THE PHASE 33 WAVE 6 INCIDENT. NO marker anywhere in this file."""

import os
import pathlib

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])

SYNTHETIC = b"48-SYNTHETIC-ROWS-FROM-A-HELPER-THAT-SKIPPED-THE-SANDBOX"


def test_a_helper_without_the_sandbox_fixture_overwrote_all_three_gold_matrices():
    for name in ("features_wp", "features_ats", "features_ou"):
        (DATA / "gold" / (name + ".parquet")).write_bytes(SYNTHETIC)
''',
    ADD_REMOVE_MODULE: '''\
"""Creation and deletion are their own kinds, not rewrites."""

import os
import pathlib

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])


def test_unmarked_creation_is_rejected():
    (DATA / "gold" / "features_new.parquet").write_bytes(b"A-BRAND-NEW-MATRIX")


def test_unmarked_deletion_is_rejected():
    (DATA / "gold" / "features_gone.parquet").unlink()
''',
    METADATA_MODULE: '''\
"""A write that restores its own size and mtime. The prefilter cannot see it."""

import os
import pathlib

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])


def test_metadata_restoring_write_slips_past_the_prefilter():
    target = DATA / "gold" / "features_meta.parquet"
    before = target.stat()
    target.write_bytes(b"MTA-v2")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
''',
}

_CHILD_INI = """\
[pytest]
markers =
    writes_production_store: this test legitimately writes the production store paths named in its paths= argument (COLD-05, D33-24)
"""


LEDGER_STORE_MODULE = "test_6_ledger_store_write.py"
LEDGER_GIT_MODULE = "test_7_ledger_git_internals.py"
LEDGER_STORE_TEST = "test_an_unmarked_append_to_the_forward_ledger_is_rejected"
CLEAN_MODULE = "test_8_writes_nothing.py"

# Two generated tests in two generated modules; one of them is reported.
EXPECTED_LEDGER_CHILD_TESTS = 2

_LEDGER_CHILD_MODULES: dict[str, str] = {
    LEDGER_STORE_MODULE: f'''\
"""An unmarked append to the forward ledger store. NO marker anywhere in this file."""

import os
import pathlib

LEDGER = pathlib.Path(os.environ["NFL_GUARD_LEDGER_ROOT"])


def {LEDGER_STORE_TEST}():
    with open(LEDGER / "forward_2026.jsonl", "ab") as handle:
        handle.write(b'{{"seq": 2, "row": "APPENDED-BY-A-TEST"}}\\n')
''',
    LEDGER_GIT_MODULE: '''\
"""Writes inside the backup repository's .git directory only. Must not be reported."""

import os
import pathlib

LEDGER = pathlib.Path(os.environ["NFL_GUARD_LEDGER_ROOT"])


def test_new_and_modified_files_under_dot_git_are_not_the_store():
    (LEDGER / ".git" / "objects" / "ab" / "cd.jsonl").write_bytes(b"NEW-GIT-OBJECT")
    (LEDGER / ".git" / "packed.json").write_bytes(b'{"refs": "REWRITTEN"}')
''',
}

_CLEAN_CHILD_MODULES: dict[str, str] = {
    CLEAN_MODULE: '''\
"""Writes nothing under any guarded root."""


def test_a_test_that_writes_nothing():
    assert 1 + 1 == 2
''',
}


def _build_store(root: Path) -> tuple[Path, Path, Path]:
    """Lay down a stand-in production tree under *root*/store: data, artifacts, ledger."""
    store = root / "store"
    data = store / "data"
    artifacts = store / "artifacts"
    ledger = store / "ledger"
    (data / "gold").mkdir(parents=True)
    (data / "silver").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (ledger / ".git" / "objects" / "ab").mkdir(parents=True)
    (ledger / "forward_2026.jsonl").write_bytes(b'{"seq": 1, "row": "v1"}\n')
    (ledger / ".git" / "packed.json").write_bytes(b'{"refs": "v1"}')
    return data, artifacts, ledger


def _write_suite(root: Path, modules: dict[str, str]) -> Path:
    """Write a generated mini-suite (conftest, ini, modules) under *root*/suite."""
    suite = root / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        _CHILD_CONFTEST.format(repo_root=str(REPO_ROOT)), encoding="utf-8"
    )
    for name, source in modules.items():
        (suite / name).write_text(source, encoding="utf-8")
    (suite / "pytest.ini").write_text(_CHILD_INI, encoding="utf-8")
    return suite


def _build_sandbox(root: Path) -> tuple[Path, Path]:
    """Lay down a stand-in production tree and the mini-suite that writes into it."""
    store = root / "store"
    data = store / "data"
    artifacts = store / "artifacts"
    (data / "gold").mkdir(parents=True)
    (data / "silver").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (store / "ledger").mkdir(parents=True)

    # The three production gold matrices the Wave-6 incident overwrote.
    (data / "gold" / "features_wp.parquet").write_bytes(b"WP-COLUMN-BYTES-v1")
    (data / "gold" / "features_ats.parquet").write_bytes(b"ATS-COLUMN-BYTES-v1")
    (data / "gold" / "features_ou.parquet").write_bytes(b"OU-COLUMN-BYTES-v1")
    # Exactly six bytes, so the metadata-restoring case can rewrite it with six
    # DIFFERENT bytes and leave st_size untouched. A prefilter that reads size and
    # mtime cannot see that write. Kept on its OWN path so the Wave-6 module's
    # overwrite of the gold matrices cannot be confused with it.
    (data / "gold" / "features_meta.parquet").write_bytes(b"MTA-v1")
    (data / "gold" / "features_gone.parquet").write_bytes(b"GONE-v1")
    (data / "silver" / "games.parquet").write_bytes(b"GAMES-v1")
    (data / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v1")
    (artifacts / "latest.json").write_bytes(b'{"wp": "v1"}')

    return _write_suite(root, _CHILD_MODULES), data


def _run_child(
    suite: Path, data: Path, artifacts: Path, ledger: Path
) -> subprocess.CompletedProcess:
    """Run the generated mini-suite with EVERY guarded root redirected into tmp."""
    env = os.environ.copy()
    env["NFL_GUARD_DATA_ROOT"] = str(data)
    env["NFL_GUARD_ARTIFACTS_ROOT"] = str(artifacts)
    env["NFL_GUARD_LEDGER_ROOT"] = str(ledger)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "--no-header",
            # Explicit rather than inherited: `--tb=auto` renders the middle failures
            # of a multi-block run differently from the first and last, and this
            # module PARSES those blocks. One shape, deterministically.
            "--tb=long",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
        ],
        capture_output=True,
        text=True,
        cwd=str(suite),
        env=env,
        check=False,
    )


def _blocks(output: str) -> dict[str, str]:
    """Split a pytest report into per-MODULE regions, keyed as the guard names them."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in output.splitlines():
        header = _GUARD_MODULE_HEADER_RE.search(line)
        if header:
            current = header.group(1)
            blocks.setdefault(current, [])
            continue
        # Anything that ends the per-module report region closes the open block. The
        # closing full sweep in particular is printed by `pytest_terminal_summary`
        # AFTER the last error block, under a `write_sep("-", ...)` rule -- letting it
        # leak into the block above would attribute a session-level finding to an
        # innocent module, which is the exact mis-attribution the sweep is reported at
        # session level to avoid.
        if (
            line.startswith("=")
            or line.startswith("short test summary")
            or CLOSING_SWEEP_HEADER in line
            or _SEPARATOR_RULE_RE.match(line)
        ):
            current = None
            continue
        if current:
            blocks[current].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


@pytest.fixture(scope="module")
def nested_session(tmp_path_factory) -> dict[str, object]:
    """Run the generated mini-suite ONCE; every assertion below reads this one run."""
    root = tmp_path_factory.mktemp("guard_arming")
    suite, data = _build_sandbox(root)
    completed = _run_child(
        suite, data, root / "store" / "artifacts", root / "store" / "ledger"
    )
    output = completed.stdout + completed.stderr
    return {
        "completed": completed,
        "output": output,
        "blocks": _blocks(output),
        "data": data,
    }


@pytest.fixture(scope="module")
def ledger_session(tmp_path_factory) -> dict[str, object]:
    """Run the two ledger modules ONCE in their own nested session."""
    root = tmp_path_factory.mktemp("guard_arming_ledger")
    data, artifacts, ledger = _build_store(root)
    suite = _write_suite(root, _LEDGER_CHILD_MODULES)
    completed = _run_child(suite, data, artifacts, ledger)
    output = completed.stdout + completed.stderr
    return {"completed": completed, "output": output, "blocks": _blocks(output)}


@pytest.fixture(scope="module")
def clean_session(tmp_path_factory) -> dict[str, object]:
    """Run a nested session whose one module writes nothing anywhere."""
    root = tmp_path_factory.mktemp("guard_arming_clean")
    data, artifacts, ledger = _build_store(root)
    suite = _write_suite(root, _CLEAN_CHILD_MODULES)
    completed = _run_child(suite, data, artifacts, ledger)
    return {"completed": completed, "output": completed.stdout + completed.stderr}


class TestTheChildSessionIsTheEvidenceItClaimsToBe:
    """Anti-vacuity. Every assertion below reads one run's output."""

    def test_the_child_session_failed(
        self, nested_session, data_boundary_guard
    ) -> None:
        completed = nested_session["completed"]
        assert completed.returncode != 0, (
            "the nested session exited 0. A session in which several modules wrote a "
            "guarded production store without declaring the write MUST fail.\n\n"
            + str(nested_session["output"])
        )
        assert "no tests ran" not in nested_session["output"], (
            "the generated mini-suite collected nothing, so every assertion in this "
            "module would be vacuous.\n\n" + str(nested_session["output"])
        )

    def test_every_generated_test_ran(
        self, nested_session, data_boundary_guard
    ) -> None:
        """A mini-suite that silently stopped collecting would pass by ABSENCE.

        Three assertions below are of the form "this module is NOT in the report",
        and a module that was never collected satisfies them for entirely the wrong
        reason. The child's own count line is the instrument.

        The generated tests all PASS -- they only write files. The guard fires in
        module teardown, so the findings are reported as ERRORS, not as failures.
        """
        output = str(nested_session["output"])
        assert re.search(rf"\b{EXPECTED_CHILD_TESTS} passed\b", output), (
            f"the child session's summary does not report {EXPECTED_CHILD_TESTS} "
            "passed tests, so the mini-suite is not the one this module "
            f"generated.\n\n{output}"
        )
        errors = re.search(r"\b(\d+) errors?\b", output)
        assert errors and int(errors.group(1)) == 3, (
            "exactly three of the five generated modules must be reported -- the "
            "Wave-6 overwrite, the marked writer's undeclared neighbour, and the "
            "creation/deletion pair. The other two are the accepted loss (permitted) "
            f"and the metadata-restoring write (caught at session end).\n\n{output}"
        )


class TestThePhase33Wave6IncidentIsStillCaught:
    """THE GUARANTEE. An unmarked module that overwrites production gold FAILS."""

    def test_the_three_gold_matrices_are_all_named_with_both_digests(
        self, nested_session, data_boundary_guard
    ) -> None:
        block = nested_session["blocks"].get(WAVE6_MODULE)
        assert block, (
            "the Wave-6 reproduction was NOT reported. A module with no marker "
            "anywhere in it overwrote all three production gold matrices and the "
            "guard said nothing -- which is the incident this guard exists to "
            "catch.\n\n" + str(nested_session["output"])
        )
        for key in (
            "gold/features_wp.parquet",
            "gold/features_ats.parquet",
            "gold/features_ou.parquet",
        ):
            assert key in block, f"{key} is missing from the report.\n\n{block}"
        assert "REWRITTEN:" in block, block
        digests = _SHA256_RE.findall(block)
        assert len(digests) >= 6, (
            "the violation must carry BOTH digests for EACH of the three matrices, so "
            f"the report is checkable against the files. Found {len(digests)}.\n\n"
            f"{block}"
        )

    def test_the_finding_is_attributed_to_the_module_that_wrote_them(
        self, nested_session, data_boundary_guard
    ) -> None:
        """Module-level attribution, asserted POSITIVELY rather than by absence."""
        output = str(nested_session["output"])
        assert f"MODULE {WAVE6_MODULE}" in output, (
            "the report does not name the module that wrote the gold matrices. Under "
            "module scope the pytest block header names whichever test ran last in the "
            f"file, so the guard must state the module itself.\n\n{output}"
        )

    def test_a_marker_in_an_earlier_module_stood_the_guard_down_for_nobody(
        self, nested_session, data_boundary_guard
    ) -> None:
        """``test_1_marked_writer.py`` runs FIRST. This module still fails."""
        block = nested_session["blocks"][WAVE6_MODULE]
        assert "nfl_predictions.duckdb" not in block, (
            "the Wave-6 report named the path an EARLIER module declared. The two "
            f"modules' verdicts have leaked into each other.\n\n{block}"
        )


class TestTheExemptionIsPathScopedWithinAModule:
    """A module that declares one store gets nothing for any other."""

    def test_the_marked_writers_module_fails_for_the_path_it_did_not_declare(
        self, nested_session, data_boundary_guard
    ) -> None:
        block = nested_session["blocks"].get(MARKED_MODULE)
        assert block, (
            "a module containing a marked writer wrote an UNDECLARED path and was not "
            "reported. Module scope is a union of declared PATHS, never a licence for "
            "the file.\n\n" + str(nested_session["output"])
        )
        assert "silver/games.parquet" in block, block
        assert "nfl_predictions.duckdb" not in block, (
            "the violation named the module's DECLARED path. A path-scoped marker must "
            f"permit exactly the write it declares.\n\n{block}"
        )


class TestTheAcceptedAttributionLoss:
    """What module granularity COSTS, recorded as a test and not as a docstring.

    In a module where one test declares ``data/nfl_predictions.duckdb``, a DIFFERENT
    unmarked test in that same module writing that same path is now PERMITTED. Under
    per-test scope it would have failed. The owner accepted that trade explicitly in
    exchange for the measured ~179 s, and a degradation that lives only in prose is a
    degradation nobody re-reads -- so it lives here, under a name that says what it is.

    The recovery is real and is stated in the guard's own violation message: re-run
    that one file on its own and the window narrows to the tests in it.
    """

    def test_an_unmarked_sibling_writing_the_same_declared_path_is_permitted(
        self, nested_session, data_boundary_guard
    ) -> None:
        assert ACCEPTED_LOSS_MODULE not in nested_session["blocks"], (
            "the accepted loss did not occur -- the unmarked sibling was reported. "
            "That is a STRICTER guard than this change describes, so either the union "
            "is not module-wide or this test is describing the wrong behaviour. Fix "
            "the description, not the guard.\n\n" + str(nested_session["output"])
        )

    def test_the_session_end_sweep_did_not_report_it_either(
        self, nested_session, data_boundary_guard
    ) -> None:
        """The rebase is what keeps the permitted write out of the closing sweep."""
        output = str(nested_session["output"])
        tail = (
            output.split(CLOSING_SWEEP_HEADER, 1)[1]
            if CLOSING_SWEEP_HEADER in output
            else ""
        )
        assert "nfl_predictions.duckdb" not in tail, (
            "a PERMITTED write surfaced in the session-end sweep, which means the "
            f"per-module rebase did not land.\n\n{tail}"
        )


class TestCreationAndDeletionAreReportedAsTheirOwnKinds:
    def test_added_and_removed_both_appear_with_their_digests(
        self, nested_session, data_boundary_guard
    ) -> None:
        block = nested_session["blocks"].get(ADD_REMOVE_MODULE)
        assert block, str(nested_session["output"])

        assert "ADDED:" in block, (
            "a file that did not exist before must be reported as ADDED -- reporting "
            f"it as a rewrite loses the distinction that says what happened.\n\n{block}"
        )
        assert "gold/features_new.parquet" in block, block

        assert "REMOVED:" in block, (
            f"a deleted store is REMOVED, not REWRITTEN.\n\n{block}"
        )
        assert "gold/features_gone.parquet" in block, block

        assert "REWRITTEN:" not in block, (
            "nothing in this module rewrote an existing store, so a REWRITTEN section "
            f"means the kinds have blurred.\n\n{block}"
        )
        assert len(_SHA256_RE.findall(block)) >= 2, (
            "the ADDED file's after-digest and the REMOVED file's before-digest must "
            f"both appear.\n\n{block}"
        )


class TestTheClosingSweepCatchesWhatThePrefilterCannot:
    """D33-23 review hardening: the prefilter buys speed, the closing sweep buys truth.

    UNCHANGED IN SUBSTANCE by the move to module scope, and deliberately so. The
    per-module pass is still a stat prefilter, so it still cannot see a write that
    restores its own size and mtime; the session-end full content sweep is still what
    makes the SESSION's verdict content-based. That is the backstop the whole
    speed-for-attribution trade rests on.
    """

    def test_the_prefilter_reported_nothing_for_the_metadata_restoring_write(
        self, nested_session, data_boundary_guard
    ) -> None:
        assert METADATA_MODULE not in nested_session["blocks"], (
            "the per-module stat prefilter reported the metadata-restoring write. If "
            "it can see that write on this filesystem the case is not exercising what "
            "it was written for -- check that st_size and st_mtime_ns really were "
            "restored.\n\n" + str(nested_session["output"])
        )

    def test_the_session_still_failed_via_the_closing_full_sweep(
        self, nested_session, data_boundary_guard
    ) -> None:
        output = str(nested_session["output"])
        assert CLOSING_SWEEP_HEADER in output, (
            "no session-end full content sweep was reported. A write that restores its "
            "own size and mtime is invisible to the stat prefilter, so without the "
            "closing sweep the session's verdict is not content-based at all.\n\n"
            + output
        )
        tail = output.split(CLOSING_SWEEP_HEADER, 1)[1]
        assert "gold/features_meta.parquet" in tail, (
            "the closing sweep fired but did not name the file whose bytes moved.\n\n"
            + tail
        )
        assert len(_SHA256_RE.findall(tail)) >= 2, (
            "the closing sweep must carry both digests, exactly as a per-module "
            f"violation does.\n\n{tail}"
        )


class TestTheHarnessIsTheOneTheRulingNamed:
    """The nested session runs OUT OF PROCESS. Asserted, not assumed."""

    def test_this_module_uses_a_subprocess_and_neither_pytest_main_nor_pytester(
        self,
    ) -> None:
        """Scanned by AST, not by substring.

        The prose above DESCRIBES the two harnesses this module rejects, so a substring
        scan would flag its own rationale. The AST sees calls and parameters, which is
        what the ruling is actually about.
        """
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        subprocess_runs = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert subprocess_runs, (
            "the nested session must be driven by subprocess.run, so the child cannot "
            "touch the parent's plugin manager or fixture state."
        )

        pytest_mains = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "main"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ]
        assert not pytest_mains, (
            "a recursive pytest.main() shares the parent's plugin manager and fixture "
            "state -- the contamination risk the harness ruling exists to avoid."
        )

        requested = {
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            for argument in node.args.args
        }
        assert "pytester" not in requested, (
            "pytester is not enabled in this repository and enabling it to serve one "
            "module would change every session's plugin set."
        )


# ---------------------------------------------------------------------------
# Plan 34-02 / D-10: the third guarded root, `ledger`.
# ---------------------------------------------------------------------------

# The suffix set the data and artifacts roots were guarded under before Phase 34,
# spelled out so a widening of the global set fails here by value.
_PRE_PHASE34_TRACKED_SUFFIXES = (
    ".parquet",
    ".duckdb",
    ".db",
    ".json",
    ".csv",
    ".pkl",
    ".joblib",
)


def _ledger_root_with_git_internals(root: Path) -> Path:
    """A stand-in ledger root: the store, plus .git files with and without suffixes."""
    ledger = root / "ledger"
    (ledger / ".git" / "objects" / "ab").mkdir(parents=True)
    (ledger / "forward_2026.jsonl").write_bytes(b'{"seq": 1}\n')
    (ledger / ".git" / "objects" / "ab" / "cd").write_bytes(b"GIT-OBJECT")
    (ledger / ".git" / "objects" / "ab" / "cd.jsonl").write_bytes(b"GIT-OBJECT")
    (ledger / ".git" / "packed.json").write_bytes(b'{"refs": "v1"}')
    return ledger


class TestTheLedgerRootIsTheThirdGuardedRoot:
    """Direct checks on the root list and the per-root policy. No nested session."""

    def test_guarded_roots_returns_three_roots_with_the_ledger_last(
        self, monkeypatch, tmp_path
    ) -> None:
        from tests.conftest import GUARD_LEDGER_ROOT_ENV, _guarded_roots
        from tests.data_boundary import PRODUCTION_LEDGER_ROOT

        assert GUARD_LEDGER_ROOT_ENV == "NFL_GUARD_LEDGER_ROOT"
        monkeypatch.delenv(GUARD_LEDGER_ROOT_ENV, raising=False)
        roots = _guarded_roots()
        assert [label for label, _ in roots] == ["data", "artifacts", "ledger"]
        assert roots[2][1] == PRODUCTION_LEDGER_ROOT == Path("ledger")

        monkeypatch.setenv(GUARD_LEDGER_ROOT_ENV, str(tmp_path))
        assert _guarded_roots()[2] == ("ledger", tmp_path)

    def test_the_data_and_artifacts_suffixes_are_the_pre_phase34_set(self) -> None:
        from tests.data_boundary import (
            LEDGER_EXTRA_SUFFIXES,
            TRACKED_SUFFIXES,
            root_policy,
        )

        assert TRACKED_SUFFIXES == _PRE_PHASE34_TRACKED_SUFFIXES
        for label in ("data", "artifacts"):
            assert root_policy(label) == (_PRE_PHASE34_TRACKED_SUFFIXES, frozenset())
        assert LEDGER_EXTRA_SUFFIXES == (".jsonl",)
        assert root_policy("ledger") == (
            (*_PRE_PHASE34_TRACKED_SUFFIXES, ".jsonl"),
            frozenset({".git"}),
        )

    def test_both_sweeps_use_the_ledger_root_policy(self, tmp_path) -> None:
        """The pre-test stat pass and the content-digest pass read ONE policy.

        Exercised separately, so neither pass can silently keep the old global
        suffix set. The .git files carry tracked suffixes on purpose: only the
        directory exclusion can be what keeps them out.
        """
        from tests.conftest import _content_sweep, _stat_sweep

        ledger = _ledger_root_with_git_internals(tmp_path)

        assert set(_stat_sweep(ledger, "ledger")) == {"forward_2026.jsonl"}
        assert set(_content_sweep("ledger", ledger)) == {"forward_2026.jsonl"}

        # Control: under the data policy the .git file IS visible and the .jsonl
        # is not, so the ledger result above is the policy's doing.
        assert set(_stat_sweep(ledger, "data")) == {".git/packed.json"}
        assert set(_content_sweep("data", ledger)) == {".git/packed.json"}


class TestALedgerStoreWriteIsCaughtAndDotGitIsNot:
    """Nested session: the .jsonl append trips the guard; the .git writes do not."""

    def test_the_ledger_session_failed_with_exactly_one_report(
        self, ledger_session, data_boundary_guard
    ) -> None:
        output = str(ledger_session["output"])
        assert ledger_session["completed"].returncode != 0, (
            "the nested session exited 0, so an unmarked append to the forward ledger "
            f"store went unreported.\n\n{output}"
        )
        assert re.search(rf"\b{EXPECTED_LEDGER_CHILD_TESTS} passed\b", output), output
        errors = re.search(r"\b(\d+) errors?\b", output)
        assert errors and int(errors.group(1)) == 1, (
            "exactly one module -- the store append -- must be reported; the .git "
            f"module must not be.\n\n{output}"
        )

    def test_the_report_names_the_module_the_test_and_the_jsonl_file(
        self, ledger_session, data_boundary_guard
    ) -> None:
        output = str(ledger_session["output"])
        block = ledger_session["blocks"].get(LEDGER_STORE_MODULE)
        assert block, f"the ledger store write was NOT reported.\n\n{output}"
        assert f"ERROR at teardown of {LEDGER_STORE_TEST}" in output, output
        assert "forward_2026.jsonl" in block, block
        assert "REWRITTEN:" in block, block
        assert len(_SHA256_RE.findall(block)) >= 2, block
        assert (
            'writes_production_store(paths=["ledger/forward_2026.jsonl"])' in block
        ), f"the suggested marker must route the path to the ledger label.\n\n{block}"

    def test_new_and_modified_files_under_dot_git_are_not_reported(
        self, ledger_session, data_boundary_guard
    ) -> None:
        output = str(ledger_session["output"])
        assert LEDGER_GIT_MODULE not in ledger_session["blocks"], output
        assert ".git/" not in output, (
            f"a file inside the backup repository's .git was reported.\n\n{output}"
        )
        assert CLOSING_SWEEP_HEADER not in output, (
            "the closing sweep found something no per-module pass saw; under the "
            f"ledger policy it must find nothing here.\n\n{output}"
        )


class TestANestedSessionThatWritesNothingPasses:
    """No false positive: three guarded roots, nothing written, exit 0."""

    def test_the_clean_session_exits_zero(
        self, clean_session, data_boundary_guard
    ) -> None:
        output = str(clean_session["output"])
        assert clean_session["completed"].returncode == 0, output
        assert re.search(r"\b1 passed\b", output), output
        assert not re.search(r"\b\d+ errors?\b", output), output
        assert CLOSING_SWEEP_HEADER not in output, output
