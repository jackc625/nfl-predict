"""D33-24: the write-guard's opt-out is PATH-SCOPED, and the guard never stands down.

WHAT THIS MODULE IS FOR
-----------------------
``tests/integration/test_data_boundary_guard_arming.py`` proves the guard FIRES, end to
end, through a nested pytest session. This module proves the three properties that decide
whether the guard is worth having once it does:

1. **The marker exempts a PATH, not a test.** A marked test that writes outside its
   declared ``paths=`` still fails, and the failure names the undeclared path. An
   exemption that widened to "this test may write anything" would be the same
   blanket permission the opt-in design already gave every unmarked module.
2. **The prefilter is not the verdict** (D33-23). The per-test pass reads
   ``(st_size, st_mtime_ns)`` to decide WHERE to look. A file whose stat signature moved
   but whose bytes did not is NOT a violation -- if it were, the guard would report noise
   and be switched off inside a week.
3. **The guard refuses rather than skips** (T-33-03). Under a configuration it cannot be
   correct in -- a multi-worker session sharing one filesystem -- it raises a NAMED error.
   ``pytest.skip`` would leave the session green with nothing guarding it, which is
   strictly worse than not having the guard at all.

Every assertion drives the guard's own helpers directly against a ``tmp_path`` root. No
real production store is read or written here; the sandbox is constructed per test.

THE INVENTORY TEST IS THE ONE THAT TOUCHES THE LIVE SUITE, and it only COLLECTS it. A
marker is a standing write permission, so the complete set of them is pinned to
``tests/phase33_state.MARKED_PRODUCTION_WRITERS`` and drift is a failure in both
directions: a new unrecorded marker is an ungoverned exemption, and a vanished one means
a permission was removed without the manifest being told.

TEST CLASS: plain unit test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _assert_single_worker,
    _guard_verdict,
    _marker_paths_for_item,
    _stat_sweep,
    _StoreBaseline,
    _suspect_paths,
)
from tests.data_boundary import digest_tree
from tests.phase33_state import (
    DELIBERATE_TRIPWIRE_NODE_IDS,
    MARKED_PRODUCTION_WRITERS,
    WRITES_PRODUCTION_STORE_MARKER,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_TRIPWIRES_IN_N01 = (
    "tests/integration/test_n01_resync_control.py::"
    "TestEvery2021To2024ValueIsByteIdentical::"
    "test_every_data_column_reproduces_its_pre_resync_digest_exactly",
    "tests/integration/test_n01_resync_control.py::"
    "TestEvery2021To2024ValueIsByteIdentical::"
    "test_the_moved_set_is_exactly_the_build_clock",
)


# ---------------------------------------------------------------------------
# Sandbox plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A stand-in `data/` root: two gold matrices and a duckdb store."""
    root = tmp_path / "data"
    (root / "gold").mkdir(parents=True)
    (root / "gold" / "features_wp.parquet").write_bytes(b"WP-v1")
    (root / "gold" / "features_ats.parquet").write_bytes(b"ATS-v1")
    (root / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-v1")
    return root


def _baseline(root: Path) -> _StoreBaseline:
    return _StoreBaseline("data", root, digest_tree(root), _stat_sweep(root))


class _StubMarker:
    def __init__(self, paths) -> None:
        self.args = ()
        self.kwargs = {"paths": list(paths)}


class _StubItem:
    """The only two attributes `_marker_paths_for_item` uses."""

    def __init__(
        self, paths=None, nodeid: str = "tests/unit/stub.py::test_stub"
    ) -> None:
        self.nodeid = nodeid
        self._marker = None if paths is None else _StubMarker(paths)

    def get_closest_marker(self, name: str):
        return self._marker if name == WRITES_PRODUCTION_STORE_MARKER else None


class _StubConfig:
    """A session config carrying only what the arming check interrogates."""

    def __init__(self, workerinput=None, numprocesses=None) -> None:
        if workerinput is not None:
            self.workerinput = workerinput
        self._options = {"numprocesses": numprocesses}

    def getoption(self, name, default=None):
        return self._options.get(name, default)


# ---------------------------------------------------------------------------
# 1. The marker exempts a PATH, not a test
# ---------------------------------------------------------------------------


class TestTheMarkerIsPathScoped:
    def test_a_marked_test_writing_outside_its_declared_paths_still_fails(
        self, sandbox: Path
    ) -> None:
        baseline = _baseline(sandbox)
        declared = _marker_paths_for_item(
            _StubItem(paths=["data/gold/features_wp.parquet"])
        )["data"]

        (sandbox / "gold" / "features_ats.parquet").write_bytes(b"ATS-v2-UNDECLARED")

        message = _guard_verdict(baseline, declared)
        assert message, (
            "a marked test wrote a path its marker does not name and the guard said "
            "nothing. The exemption is for the declared PATHS, never for the test."
        )
        assert "gold/features_ats.parquet" in message, message
        assert "gold/features_wp.parquet" not in message, (
            "the violation named the DECLARED path. Only the undeclared write is the "
            f"fault here.\n\n{message}"
        )

    def test_a_declared_path_that_is_never_written_grants_nothing_and_fails_nothing(
        self, sandbox: Path
    ) -> None:
        baseline = _baseline(sandbox)
        declared = _marker_paths_for_item(
            _StubItem(paths=["data/gold/features_never_written.parquet"])
        )["data"]

        assert _guard_verdict(baseline, declared) is None, (
            "a marker that names a path the test does not write is not itself a "
            "violation -- an over-cautious declaration must not redden a clean test."
        )

        (sandbox / "gold" / "features_wp.parquet").write_bytes(b"WP-v2")
        message = _guard_verdict(baseline, declared)
        assert message and "gold/features_wp.parquet" in message, (
            "the unused declaration granted an exemption to a DIFFERENT path.\n\n"
            f"{message}"
        )

    def test_a_declared_path_outside_every_guarded_root_is_an_error(self) -> None:
        """Silently dropping it would grant an exemption the author believes they have."""
        with pytest.raises(ValueError) as excinfo:
            _marker_paths_for_item(_StubItem(paths=["outputs/scratch.json"]))
        assert "repo-relative" in str(excinfo.value)

    def test_the_marker_routes_to_the_artifacts_root_as_well_as_data(self) -> None:
        scoped = _marker_paths_for_item(_StubItem(paths=["artifacts/latest.json"]))
        assert scoped["artifacts"] == frozenset({"latest.json"})
        assert scoped["data"] == frozenset()


# ---------------------------------------------------------------------------
# 2. The rebase after a permitted write is NARROW -- in both directions
# ---------------------------------------------------------------------------


class TestThePermittedWriteRebasesExactlyItsDeclaredPaths:
    def test_the_rebase_leaks_sideways_to_nothing_and_leaves_nothing_stale(
        self, sandbox: Path
    ) -> None:
        """Three steps, one test, because the property has two failure directions.

        A rebase that is too WIDE would silently absolve the neighbouring path in step
        2. A rebase that never happened at all would leave the stale pre-write digest in
        the baseline, and step 3's second write to the same path would then compare
        against a value two writes old -- which, for a write that restored the original
        bytes, would read as clean. Asserting the steps separately in three test
        functions would let either half pass while the other silently regressed.
        """
        baseline = _baseline(sandbox)
        declared = _marker_paths_for_item(
            _StubItem(paths=["data/gold/features_wp.parquet"])
        )["data"]

        # (1) the marked test writes its declared path
        (sandbox / "gold" / "features_wp.parquet").write_bytes(b"WP-v2-DECLARED")
        assert _guard_verdict(baseline, declared) is None, (
            "the declared write was reported as a violation"
        )

        # (2) an unmarked test writes a NEIGHBOURING path in the same directory
        (sandbox / "gold" / "features_ats.parquet").write_bytes(b"ATS-v2-UNDECLARED")
        neighbour = _guard_verdict(baseline, frozenset())
        assert neighbour, "the rebase widened into a directory-level exemption"
        assert "gold/features_ats.parquet" in neighbour, neighbour
        assert "gold/features_wp.parquet" not in neighbour, (
            "step 1's permitted write was re-reported against step 2's test.\n\n"
            f"{neighbour}"
        )

        # (3) an unmarked test writes the REBASED path again
        (sandbox / "gold" / "features_wp.parquet").write_bytes(b"WP-v3-UNDECLARED-NOW")
        second = _guard_verdict(baseline, frozenset())
        assert second and "gold/features_wp.parquet" in second, (
            "the rebased path was not judged against its CURRENT state -- one declared "
            f"write cannot exempt every later write to the same file.\n\n{second}"
        )


# ---------------------------------------------------------------------------
# 3. Creation and deletion are their own kinds
# ---------------------------------------------------------------------------


class TestCreationAndDeletionAreNotRewrites:
    def test_an_unmarked_creation_is_reported_as_ADDED(self, sandbox: Path) -> None:
        baseline = _baseline(sandbox)
        (sandbox / "gold" / "features_ou.parquet").write_bytes(b"OU-NEW")

        message = _guard_verdict(baseline, frozenset())
        assert message, "a new tracked store appeared and the guard said nothing"
        assert "ADDED:" in message, message
        assert "REWRITTEN:" not in message, (
            f"a file that did not exist before is ADDED, not REWRITTEN.\n\n{message}"
        )
        assert "gold/features_ou.parquet" in message, message

    def test_an_unmarked_deletion_is_reported_as_REMOVED(self, sandbox: Path) -> None:
        baseline = _baseline(sandbox)
        (sandbox / "gold" / "features_ats.parquet").unlink()

        message = _guard_verdict(baseline, frozenset())
        assert message, "a tracked store vanished and the guard said nothing"
        assert "REMOVED:" in message, message
        assert "REWRITTEN:" not in message, (
            f"a deleted store is REMOVED, not REWRITTEN.\n\n{message}"
        )
        assert "gold/features_ats.parquet" in message, message


# ---------------------------------------------------------------------------
# 4. The prefilter decides WHERE to look; it never renders the verdict
# ---------------------------------------------------------------------------


class TestTheStatSweepIsAPrefilterAndNotAVerdict:
    def test_a_moved_stat_signature_over_unchanged_bytes_is_not_a_violation(
        self, sandbox: Path
    ) -> None:
        """D33-23, stated as a property rather than as a comment.

        Both halves are asserted in ONE test on purpose: "the verdict list is empty" is
        trivially satisfiable by a sweep that found nothing to look at. It only means
        something alongside "the prefilter DID flag this key".
        """
        baseline = _baseline(sandbox)
        target = sandbox / "gold" / "features_wp.parquet"
        future = target.stat().st_mtime_ns + 5_000_000_000
        os.utime(target, ns=(future, future))

        moved = _suspect_paths(baseline.stats, _stat_sweep(sandbox))
        assert "gold/features_wp.parquet" in moved, (
            "the stat prefilter did not flag a file whose mtime moved, so this test is "
            "not exercising the discrimination it was written for."
        )

        assert _guard_verdict(baseline, frozenset()) is None, (
            "a file whose bytes are unchanged was reported as a data move. The stat "
            "sweep decides where to look; the content hash decides what happened."
        )


# ---------------------------------------------------------------------------
# 5. The guard REFUSES under multiple workers. It never skips.
# ---------------------------------------------------------------------------


class TestTheMultiWorkerRefusal:
    def test_a_worker_session_raises_a_named_error(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_single_worker(_StubConfig(workerinput={"workerid": "gw0"}))

        message = str(excinfo.value)
        assert "multi-worker" in message, message
        assert "single-worker" in message, message

    def test_the_refusal_is_an_error_and_never_a_stand_down(self) -> None:
        """A skip would leave the session green with nothing guarding it."""
        with pytest.raises(RuntimeError) as excinfo:
            _assert_single_worker(_StubConfig(numprocesses=4))
        assert not isinstance(excinfo.value, pytest.skip.Exception), (
            "the guard stood down instead of refusing. A guard that skips under a "
            "configuration it does not understand still reports a green suite, which "
            "is worse than having no guard: the green is now evidence of nothing."
        )

    def test_a_single_worker_session_arms_without_complaint(self) -> None:
        assert _assert_single_worker(_StubConfig()) is None

    def test_the_single_worker_precondition_is_measured_not_assumed(self) -> None:
        assert importlib.util.find_spec("xdist") is None, (
            "pytest-xdist is installed on this checkout. Phase 33 resolves the "
            "guard's concurrency edge BY EXCLUSION -- one session baseline, rebased in "
            "test order -- so installing xdist activates a case the guard refuses to "
            "arm under. Either uninstall it or reopen D33-24."
        )


# ---------------------------------------------------------------------------
# 6. The live inventory of standing write permissions
# ---------------------------------------------------------------------------


_INVENTORY_PLUGIN = '''\
"""Collect-time inventory of every test carrying the write-guard marker."""

import json
import os


def pytest_collection_modifyitems(session, config, items):
    marked = []
    for item in items:
        marker = item.get_closest_marker({marker_name!r})
        if marker is None:
            continue
        paths = list(marker.kwargs.get("paths") or ())
        for value in marker.args:
            paths.extend([value] if isinstance(value, str) else list(value))
        marked.append([item.nodeid.replace("\\\\", "/"), paths])
    with open(os.environ["GUARD_INVENTORY_OUT"], "w", encoding="utf-8") as handle:
        json.dump({{"collected": len(items), "marked": marked}}, handle)
'''


@pytest.fixture(scope="module")
def collected_inventory(tmp_path_factory) -> dict:
    """Collect the LIVE suite once and report which items carry the marker.

    Collection only -- no test in the child runs, nothing is written, and the guard's
    session fixture never fires. Run out of process for the same reason the arming
    proof is: a pytest collection driven from inside a pytest session would share the
    parent's plugin manager.
    """
    workdir = tmp_path_factory.mktemp("guard_inventory")
    (workdir / "guard_inventory_plugin.py").write_text(
        _INVENTORY_PLUGIN.format(marker_name=WRITES_PRODUCTION_STORE_MARKER),
        encoding="utf-8",
    )
    out = workdir / "inventory.json"

    env = os.environ.copy()
    env["GUARD_INVENTORY_OUT"] = str(out)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(workdir), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-p",
            "guard_inventory_plugin",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    assert out.exists(), (
        "the inventory plugin never wrote its document, so collection did not "
        f"complete.\n\n{completed.stdout}\n{completed.stderr}"
    )
    return json.loads(out.read_text(encoding="utf-8"))


class TestTheMarkedWriterInventoryIsPinned:
    def test_the_collection_visited_a_non_empty_item_list(
        self, collected_inventory: dict
    ) -> None:
        """Anti-vacuity. A collection that found nothing would pass every set
        comparison below while proving nothing about the live suite."""
        assert collected_inventory["collected"] > 1000, (
            f"collection visited {collected_inventory['collected']} items -- far too "
            "few to be this suite. The inventory comparison below would be vacuous."
        )

    def test_the_collected_marked_set_equals_the_tracked_manifest(
        self, collected_inventory: dict
    ) -> None:
        collected = {
            (node_id, tuple(paths)) for node_id, paths in collected_inventory["marked"]
        }
        tracked = set(MARKED_PRODUCTION_WRITERS)

        assert collected, (
            "no test in the live suite carries the write-guard marker, but "
            "tests/phase33_state.MARKED_PRODUCTION_WRITERS records one. Either the "
            "marker was removed from the n01 idempotency test or the marker NAME "
            "drifted -- with --strict-markers off, a typo is silent."
        )
        assert tracked, "the tracked inventory is empty, so this comparison is vacuous"
        assert collected == tracked, (
            "the live set of standing production-write permissions has drifted from "
            "tests/phase33_state.MARKED_PRODUCTION_WRITERS.\n"
            f"  collected but not recorded: {sorted(collected - tracked)}\n"
            f"  recorded but not collected: {sorted(tracked - collected)}\n"
            "A marker is a standing permission to write a production store. A new one "
            "that nobody recorded is an ungoverned exemption; a recorded one that "
            "vanished means a permission changed without the manifest being told."
        )

    def test_neither_n01_tripwire_carries_the_marker(
        self, collected_inventory: dict
    ) -> None:
        marked_ids = {node_id for node_id, _ in collected_inventory["marked"]}
        for node_id in _TRIPWIRES_IN_N01:
            assert node_id not in marked_ids, (
                f"{node_id} carries a write exemption. It is a DELIBERATE TRIPWIRE "
                "encoding an owner-accepted fact, and its value depends on being "
                "judged against a real lake with no permission of any kind."
            )
            assert node_id in DELIBERATE_TRIPWIRE_NODE_IDS, (
                f"{node_id} is no longer in DELIBERATE_TRIPWIRE_NODE_IDS -- the two "
                "halves of this assertion have drifted apart."
            )
