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

THE SIX GENERATED CASES, and what each one is for:

1. an UNMARKED test that REWRITES a tracked store            -> must fail, REWRITTEN
2. a MARKED test that writes its DECLARED path               -> must pass
3. a SECOND unmarked test that rewrites a different store    -> must fail (the marker
   suppressed the guard for nobody but case 2)
4. an UNMARKED test that CREATES a tracked file              -> must fail, ADDED
5. an UNMARKED test that DELETES a tracked file              -> must fail, REMOVED
6. an UNMARKED test that writes different bytes and then RESTORES ``st_size`` and
   ``st_mtime_ns`` -> INVISIBLE to the per-test stat prefilter, and STILL caught, by the
   session-end full content sweep. Two assertions, because the point of the case is
   WHICH instrument catches it.

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

# pytest prints a teardown error block headed `ERROR at teardown of <name>` and a call
# failure block headed `___ <name> ___`. The guard fires in teardown, but matching both
# keeps the parser from silently returning nothing if that ever changes.
_TEARDOWN_HEADER_RE = re.compile(r"ERROR at teardown of (test_\w+)")
_FAILURE_HEADER_RE = re.compile(r"^_+ (test_\w+) _+$")
_SEPARATOR_RULE_RE = re.compile(r"^-{4,}")

CLOSING_SWEEP_HEADER = "SESSION-END FULL CONTENT SWEEP"

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

_CHILD_TESTS = '''\
"""Generated mini-suite: six writes, one of them declared."""

import os
import pathlib

import pytest

DATA = pathlib.Path(os.environ["NFL_GUARD_DATA_ROOT"])


def test_unmarked_rewrite_is_rejected():
    (DATA / "gold" / "features_wp.parquet").write_bytes(b"WP-REWRITTEN-BY-A-TEST")


@pytest.mark.writes_production_store(paths=["data/nfl_predictions.duckdb"])
def test_marked_writer_is_permitted():
    (DATA / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v2")


def test_second_unmarked_rewrite_is_still_rejected():
    (DATA / "silver" / "games.parquet").write_bytes(b"GAMES-REWRITTEN-BY-A-TEST")


def test_unmarked_creation_is_rejected():
    (DATA / "gold" / "features_new.parquet").write_bytes(b"A-BRAND-NEW-MATRIX")


def test_unmarked_deletion_is_rejected():
    (DATA / "gold" / "features_ou.parquet").unlink()


def test_metadata_restoring_write_slips_past_the_prefilter():
    target = DATA / "gold" / "features_ats.parquet"
    before = target.stat()
    target.write_bytes(b"ATS-v2")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
'''

_CHILD_INI = """\
[pytest]
markers =
    writes_production_store: this test legitimately writes the production store paths named in its paths= argument (COLD-05, D33-24)
"""


def _build_sandbox(root: Path) -> tuple[Path, Path]:
    """Lay down a stand-in production tree and the mini-suite that writes into it."""
    store = root / "store"
    data = store / "data"
    artifacts = store / "artifacts"
    (data / "gold").mkdir(parents=True)
    (data / "silver").mkdir(parents=True)
    artifacts.mkdir(parents=True)

    (data / "gold" / "features_wp.parquet").write_bytes(b"WP-COLUMN-BYTES-v1")
    # Exactly six bytes, so case 6 can rewrite it with six DIFFERENT bytes and leave
    # st_size untouched. A prefilter that reads size and mtime cannot see that write.
    (data / "gold" / "features_ats.parquet").write_bytes(b"ATS-v1")
    (data / "gold" / "features_ou.parquet").write_bytes(b"OU-COLUMN-BYTES-v1")
    (data / "silver" / "games.parquet").write_bytes(b"GAMES-v1")
    (data / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v1")
    (artifacts / "latest.json").write_bytes(b'{"wp": "v1"}')

    suite = root / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        _CHILD_CONFTEST.format(repo_root=str(REPO_ROOT)), encoding="utf-8"
    )
    (suite / "test_mini.py").write_text(_CHILD_TESTS, encoding="utf-8")
    (suite / "pytest.ini").write_text(_CHILD_INI, encoding="utf-8")
    return suite, data


def _run_child(suite: Path, data: Path, artifacts: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["NFL_GUARD_DATA_ROOT"] = str(data)
    env["NFL_GUARD_ARTIFACTS_ROOT"] = str(artifacts)
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
    """Split a pytest report into per-test blocks keyed by test function name."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in output.splitlines():
        teardown = _TEARDOWN_HEADER_RE.search(line)
        failure = _FAILURE_HEADER_RE.match(line.strip())
        if teardown:
            current = teardown.group(1)
            blocks.setdefault(current, [])
            continue
        if failure:
            current = failure.group(1)
            blocks.setdefault(current, [])
            continue
        # Anything that ends the per-test report region closes the open block. The
        # closing full sweep in particular is printed by `pytest_terminal_summary`
        # AFTER the last error block, under a `write_sep("-", ...)` rule -- letting it
        # leak into the block above would attribute a session-level finding to an
        # innocent test, which is the exact mis-attribution the sweep is reported at
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
    completed = _run_child(suite, data, root / "store" / "artifacts")
    output = completed.stdout + completed.stderr
    return {
        "completed": completed,
        "output": output,
        "blocks": _blocks(output),
        "data": data,
    }


class TestTheGuardFiresOnAnUndeclaredWrite:
    """An unmarked test that writes a production store fails, and the message says so."""

    def test_the_child_session_failed(
        self, nested_session, data_boundary_guard
    ) -> None:
        completed = nested_session["completed"]
        assert completed.returncode != 0, (
            "the nested session exited 0. A session in which five tests wrote a guarded "
            "production store and only one of them declared the write MUST fail.\n\n"
            + str(nested_session["output"])
        )
        assert "no tests ran" not in nested_session["output"], (
            "the generated mini-suite collected nothing, so every assertion in this "
            "module would be vacuous.\n\n" + str(nested_session["output"])
        )

    def test_the_unmarked_rewrite_names_the_file_and_both_digests(
        self, nested_session, data_boundary_guard
    ) -> None:
        block = nested_session["blocks"].get("test_unmarked_rewrite_is_rejected")
        assert block, "no failure was reported for the unmarked rewrite.\n\n" + str(
            nested_session["output"]
        )
        assert "gold/features_wp.parquet" in block, block
        assert "REWRITTEN:" in block, block
        digests = _SHA256_RE.findall(block)
        assert len(digests) >= 2, (
            "the violation must carry BOTH digests -- before and after -- so the report "
            f"is checkable against the file. Found {len(digests)}.\n\n{block}"
        )
        assert "test_unmarked_rewrite_is_rejected" in block, block

    def test_the_marked_writer_is_permitted(
        self, nested_session, data_boundary_guard
    ) -> None:
        assert "test_marked_writer_is_permitted" not in nested_session["blocks"], (
            "the marked writer was reported as a violation. A path-scoped marker must "
            "permit exactly the write it declares.\n\n" + str(nested_session["output"])
        )

    def test_the_marker_suppressed_nothing_for_the_second_unmarked_writer(
        self, nested_session, data_boundary_guard
    ) -> None:
        block = nested_session["blocks"].get(
            "test_second_unmarked_rewrite_is_still_rejected"
        )
        assert block, (
            "the second unmarked writer was NOT reported. One test's marker must never "
            "stand the guard down for the rest of the session.\n\n"
            + str(nested_session["output"])
        )
        assert "silver/games.parquet" in block, block
        assert "nfl_predictions.duckdb" not in block, (
            "the second violation named the marked writer's declared path. The two "
            "tests' verdicts have leaked into each other.\n\n" + block
        )

    def test_creation_and_deletion_are_reported_as_their_own_kinds(
        self, nested_session, data_boundary_guard
    ) -> None:
        created = nested_session["blocks"].get("test_unmarked_creation_is_rejected")
        deleted = nested_session["blocks"].get("test_unmarked_deletion_is_rejected")
        assert created, str(nested_session["output"])
        assert deleted, str(nested_session["output"])

        assert "ADDED:" in created and "REWRITTEN:" not in created, (
            "a file that did not exist before is ADDED, not REWRITTEN -- reporting both "
            f"as one kind loses the distinction that says what happened.\n\n{created}"
        )
        assert "gold/features_new.parquet" in created, created

        assert "REMOVED:" in deleted and "REWRITTEN:" not in deleted, (
            f"a deleted store is REMOVED, not REWRITTEN.\n\n{deleted}"
        )
        assert "gold/features_ou.parquet" in deleted, deleted


class TestTheClosingSweepCatchesWhatThePrefilterCannot:
    """D33-23 review hardening: the prefilter buys speed, the closing sweep buys truth."""

    def test_the_prefilter_reported_nothing_for_the_metadata_restoring_write(
        self, nested_session, data_boundary_guard
    ) -> None:
        assert (
            "test_metadata_restoring_write_slips_past_the_prefilter"
            not in nested_session["blocks"]
        ), (
            "the per-test stat prefilter reported the metadata-restoring write. If it "
            "can see that write on this filesystem the case is not exercising what it "
            "was written for -- check that st_size and st_mtime_ns really were "
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
        assert "gold/features_ats.parquet" in tail, (
            "the closing sweep fired but did not name the file whose bytes moved.\n\n"
            + tail
        )
        assert len(_SHA256_RE.findall(tail)) >= 2, (
            "the closing sweep must carry both digests, exactly as a per-test violation "
            f"does.\n\n{tail}"
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
