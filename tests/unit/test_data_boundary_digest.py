"""The `data/` boundary guard must be capable of FAILING, and the git check it replaces must not.

Plan 31-11, Step 2 of the owner's C-MODIFIED ruling.

THE INSTRUMENT DEFECT THIS PINS
-------------------------------
Every "data/ hard boundary held" report in Phases 28 through 31 was produced by::

    git status --porcelain data/

`.gitignore:22` blankets `data/`. That command returns empty output whether the archive
is byte-intact, silently rewritten by a stray test, or deleted outright. It has no
failure mode. Two production writes went unreported behind it during Phase 31 alone:
`tests/integration/test_elo_integration.py` rewriting `data/silver/elo_game_snapshots.parquet`,
and `tests/integration/test_lift_validation.py` retraining into production `artifacts/`.

This module proves BOTH halves of the replacement:

1. the git check really is vacuous -- demonstrated against the live repository, on a
   real gold file, so the claim is measured rather than asserted; and
2. the content check really does fire -- demonstrated by mutating a temporary copy and
   then reverting it, so the guard is shown working rather than assumed working.

TEST CLASS: plain unit test. Reads (never writes) `data/`; all mutation happens under
`tmp_path`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.data_boundary import (
    PRODUCTION_ARTIFACTS_ROOT,
    PRODUCTION_DATA_ROOT,
    TRACKED_SUFFIXES,
    DataBoundaryViolation,
    _main,
    assert_tree_unchanged,
    diff_digests,
    digest_file,
    digest_tree,
    format_digest_diff,
    is_clean,
    is_stat_signature,
)

_GOLD_WP = PRODUCTION_DATA_ROOT / "gold" / "features_wp.parquet"


def _write_store(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def sandbox_store(tmp_path: Path) -> Path:
    """A stand-in production tree: two parquet stores and a duckdb file.

    Nested one level under ``tmp_path`` so a snapshot DOCUMENT can be written beside
    the tree without landing inside it -- a document written into the guarded root
    registers as an ADDED store, which is the guard behaving correctly and the harness
    behaving badly.
    """
    root = tmp_path / "store"
    _write_store(root, "gold/features_wp.parquet", b"WP-COLUMN-BYTES-v1")
    _write_store(root, "silver/elo_game_snapshots.parquet", b"ELO-SNAPSHOTS-v1")
    _write_store(root, "nfl_predictions.duckdb", b"DUCKDB-PAGES-v1")
    return root


# ---------------------------------------------------------------------------
# 1. The check being replaced cannot fail
# ---------------------------------------------------------------------------


class TestTheGitCheckIsVacuous:
    """`git status --porcelain data/` is not evidence. Measure it, do not assume it."""

    @staticmethod
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path.cwd()),
        )

    def test_a_real_gold_file_is_ignored_by_git(self) -> None:
        """The file whose movement the phase HARD STOPPED on is invisible to git."""
        if not _GOLD_WP.exists():
            pytest.skip(
                "canonical gold is not present at data/gold/features_wp.parquet -- "
                "data/ is gitignored runtime state."
            )
        if self._git("rev-parse", "--git-dir").returncode != 0:
            pytest.skip(
                "git history is unavailable (shallow clone or not a git checkout), so "
                "the ignore rule cannot be interrogated."
            )

        ignored = self._git("check-ignore", "-q", _GOLD_WP.as_posix())
        assert ignored.returncode == 0, (
            "data/gold/features_wp.parquet is NOT gitignored on this checkout. If that "
            "is now true, the vacuity argument below changes and this module must be "
            "revisited rather than deleted."
        )

    def test_git_status_reports_nothing_for_data_even_though_gold_was_rebuilt(
        self,
    ) -> None:
        """Gold was fully rewritten at 2026-09-04 22:36 and git still says nothing."""
        if not _GOLD_WP.exists():
            pytest.skip(
                "canonical gold is not present at data/gold/features_wp.parquet -- "
                "data/ is gitignored runtime state."
            )
        status = self._git("status", "--porcelain", "data/")
        if status.returncode != 0:
            pytest.skip(
                "git history is unavailable (shallow clone or not a git checkout), so "
                "`git status` cannot be run."
            )
        assert status.stdout.strip() == "", (
            "`git status --porcelain data/` produced output, which would make it a "
            "usable boundary check after all. It does not on a normal checkout: the "
            "ignore rule blankets the directory. Investigate before trusting either "
            "instrument."
        )

    def test_the_content_check_sees_what_git_cannot(self) -> None:
        """The same directory git calls empty has real, comparable content."""
        digests = digest_tree(PRODUCTION_DATA_ROOT)
        if not digests:
            pytest.skip(
                "data/ is not populated on this checkout -- it is gitignored runtime "
                "state."
            )
        assert len(digests) > 0
        for key, value in digests.items():
            if is_stat_signature(value):
                # A locked store (DuckDB holds an exclusive handle on Windows). The
                # value declares itself, which is the point.
                continue
            assert len(value) == 64, (
                f"{key} produced a {len(value)}-character digest. A sha256 is 64 hex "
                "characters; a short value means the file was read partially and the "
                "comparison would be unsound."
            )


# ---------------------------------------------------------------------------
# 2. The replacement CAN fail -- demonstrated, then reverted
# ---------------------------------------------------------------------------


class TestTheContentGuardFires:
    """Mutate a temp copy, watch the guard fire, revert, watch it fall silent."""

    def test_an_unchanged_tree_passes(self, sandbox_store: Path) -> None:
        before = digest_tree(sandbox_store)
        assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)

    def test_a_rewritten_store_fires_and_names_the_file(
        self, sandbox_store: Path
    ) -> None:
        before = digest_tree(sandbox_store)
        target = sandbox_store / "silver" / "elo_game_snapshots.parquet"
        original = target.read_bytes()

        # MUTATE -- this is the exact shape of the Phase-31 defect: a test rebuilding
        # Elo and writing the result over the production silver snapshot table.
        target.write_bytes(b"ELO-SNAPSHOTS-REBUILT-BY-A-TEST")

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)

        message = str(excinfo.value)
        assert "silver/elo_game_snapshots.parquet" in message, (
            "the violation must NAME the file. A report that says only 'data/ changed' "
            "sends the reader back to the filesystem, which is the failure the git "
            "check already had."
        )
        assert "REWRITTEN:" in message
        assert digest_file(target) in message, (
            "the after-digest must appear so the report is checkable against the file."
        )

        # REVERT -- and the guard must fall silent again, or it is reporting noise.
        target.write_bytes(original)
        assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)

    def test_a_deleted_store_fires(self, sandbox_store: Path) -> None:
        before = digest_tree(sandbox_store)
        (sandbox_store / "gold" / "features_wp.parquet").unlink()

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)
        assert "REMOVED:" in str(excinfo.value)
        assert "gold/features_wp.parquet" in str(excinfo.value)

    def test_an_added_store_fires(self, sandbox_store: Path) -> None:
        before = digest_tree(sandbox_store)
        _write_store(sandbox_store, "gold/features_ou.parquet", b"NEW-MATRIX")

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)
        assert "ADDED:" in str(excinfo.value)
        assert "gold/features_ou.parquet" in str(excinfo.value)

    def test_a_duckdb_rewrite_fires_too(self, sandbox_store: Path) -> None:
        """The DuckDB half of every gold write is guarded, not just the parquet half."""
        before = digest_tree(sandbox_store)
        (sandbox_store / "nfl_predictions.duckdb").write_bytes(b"DUCKDB-PAGES-v2")

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)
        assert "nfl_predictions.duckdb" in str(excinfo.value)

    def test_the_violation_message_says_why_git_could_not_have_caught_it(
        self, sandbox_store: Path
    ) -> None:
        before = digest_tree(sandbox_store)
        (sandbox_store / "nfl_predictions.duckdb").write_bytes(b"x")
        diff = diff_digests(before, digest_tree(sandbox_store))
        message = format_digest_diff(
            diff, before, digest_tree(sandbox_store), sandbox_store
        )
        assert "gitignore" in message.lower()
        assert "tmp_path" in message, (
            "the message must state the FIX (redirect the write), not only the fault."
        )


class TestTheLockedFileFallback:
    """DuckDB holds an exclusive handle on Windows; the guard must still report.

    ``data/nfl_predictions.duckdb`` is open for the whole of any pytest session that
    called ``load_dataframe``, so opening it to hash raises ``PermissionError``. Two
    wrong answers were available -- crash (a guard that errors gets disabled) and skip
    silently (a guard that lies). The third is a stat signature that DECLARES itself.
    """

    def test_an_unreadable_file_falls_back_to_a_self_describing_signature(
        self, sandbox_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        target = sandbox_store / "nfl_predictions.duckdb"
        real_open = builtins.open

        def locked_open(file, *args, **kwargs):
            if Path(file) == target:
                raise PermissionError(13, "Permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", locked_open)

        value = digest_file(target)
        assert is_stat_signature(value), (
            "a locked file must yield a signature that says what it is, so a digest "
            "document can never mix a content hash on one side with a stat signature "
            "on the other and read the difference as a data move."
        )
        assert str(target.stat().st_size) in value
        assert str(target.stat().st_mtime_ns) in value

    def test_a_content_hash_is_not_mistaken_for_a_signature(
        self, sandbox_store: Path
    ) -> None:
        assert not is_stat_signature(
            digest_file(sandbox_store / "gold" / "features_wp.parquet")
        )

    def test_a_write_to_a_locked_file_still_fires_the_guard(
        self, sandbox_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The weaker instrument must still detect the write it exists to detect."""
        import builtins
        import os
        import time

        target = sandbox_store / "nfl_predictions.duckdb"
        real_open = builtins.open

        def locked_open(file, *args, **kwargs):
            if Path(file) == target:
                raise PermissionError(13, "Permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", locked_open)
        before = digest_tree(sandbox_store)
        assert is_stat_signature(before["nfl_predictions.duckdb"])

        monkeypatch.undo()
        target.write_bytes(b"DUCKDB-PAGES-v2-LONGER-THAN-BEFORE")
        # Make the mtime unambiguously different even on a coarse-resolution clock.
        future = time.time() + 2
        os.utime(target, (future, future))
        monkeypatch.setattr(builtins, "open", locked_open)

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)
        assert "nfl_predictions.duckdb" in str(excinfo.value)


class TestTheDiffPrimitives:
    """`diff_digests` and `is_clean` are the discrimination everything else rests on."""

    def test_identical_mappings_are_clean(self) -> None:
        mapping = {"a.parquet": "0" * 64}
        assert is_clean(diff_digests(mapping, dict(mapping)))

    def test_each_category_is_reported_separately(self) -> None:
        before = {"same": "1" * 64, "moved": "2" * 64, "gone": "3" * 64}
        after = {"same": "1" * 64, "moved": "9" * 64, "new": "4" * 64}
        diff = diff_digests(before, after)
        assert diff == {"added": ["new"], "removed": ["gone"], "changed": ["moved"]}
        assert not is_clean(diff)

    def test_an_empty_pair_is_clean(self) -> None:
        assert is_clean(diff_digests({}, {}))


class TestTheTreeWalk:
    """What gets hashed decides what can be missed."""

    def test_a_missing_root_yields_an_empty_mapping_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        assert digest_tree(tmp_path / "does-not-exist") == {}

    def test_untracked_suffixes_are_skipped(self, sandbox_store: Path) -> None:
        _write_store(sandbox_store, "gold/notes.txt", b"scratch")
        assert "gold/notes.txt" not in digest_tree(sandbox_store)

    def test_the_duckdb_and_manifest_suffixes_are_tracked(self) -> None:
        """A parquet-only guard would miss both stores an accident is worst in."""
        assert ".duckdb" in TRACKED_SUFFIXES, (
            "data/nfl_predictions.duckdb is the second half of every gold write; "
            "leaving it untracked would let half a write pass unreported."
        )
        assert ".json" in TRACKED_SUFFIXES, (
            "artifacts/latest.json is the deployed-model manifest; leaving it "
            "untracked would let a model swap pass unreported."
        )

    def test_keys_are_posix_relative_so_a_snapshot_travels(
        self, sandbox_store: Path
    ) -> None:
        keys = digest_tree(sandbox_store)
        assert "gold/features_wp.parquet" in keys
        assert not any("\\" in key for key in keys)

    def test_the_artifacts_root_is_guarded_as_well_as_data(self) -> None:
        """`artifacts/` holds the deployed models; Phase 31 saw a test retrain into it."""
        assert Path("artifacts") == PRODUCTION_ARTIFACTS_ROOT


class TestTheCommandLineInstrument:
    """The guard must be usable around a long run pytest is not driving."""

    def test_snapshot_then_verify_round_trips(
        self, sandbox_store: Path, tmp_path: Path
    ) -> None:
        document = tmp_path / "snap.json"
        assert _main(["snapshot", str(sandbox_store), str(document)]) == 0
        assert json.loads(document.read_text())["gold/features_wp.parquet"]
        assert _main(["verify", str(sandbox_store), str(document)]) == 0

    def test_verify_returns_nonzero_after_a_write(
        self, sandbox_store: Path, tmp_path: Path
    ) -> None:
        document = tmp_path / "snap.json"
        _main(["snapshot", str(sandbox_store), str(document)])
        (sandbox_store / "gold" / "features_wp.parquet").write_bytes(b"MOVED")
        assert _main(["verify", str(sandbox_store), str(document)]) == 1

    def test_a_bad_invocation_returns_the_usage_code(self) -> None:
        assert _main([]) == 2
        assert _main(["frobnicate", "data", "out.json"]) == 2


# ---------------------------------------------------------------------------
# NF-02 / D33-32 (Plan 33-01 Task 3): an UNDECIDED comparison is neither a data
# move nor a pass, and an UNPROVABLE one is a named refusal.
# ---------------------------------------------------------------------------
#
# The module docstring above has forbidden mixing a content hash on one side with
# a stat signature on the other since Plan 31-11 wrote it, and `is_stat_signature`
# was added to make the mixing detectable -- but nothing ever called it. A key that
# was content-hashed before a DuckDB connection opened and stat-signed after it did
# would compare UNEQUAL as raw strings and be reported as a REWRITTEN production
# store. That is a false alarm of exactly the kind that gets a guard switched off.
#
# The reachability is not hypothetical. The session-scoped baseline is held across
# a whole tier, which is precisely the window in which a DuckDB lock is acquired.
#
# D33-32 goes further, and the owner ruled on it against the reviewer who called the
# fallback a strength: a `stat-size-mtime:` value is an INABILITY TO PROVE content
# integrity, never a verdict. The guard closes handles, retries, and if it still
# cannot read the bytes it RAISES BY NAME. A legitimately-locked production file now
# fails a normal test run -- and that is the point, because the alternative is a
# green suite resting on metadata.


class TestTheMixedInstrumentRule:
    """A content hash on one side and a stat signature on the other is UNDECIDED."""

    def test_a_content_to_signature_pair_is_classified_mixed_and_not_changed(
        self,
    ) -> None:
        from tests.data_boundary import diff_digests, mixed_instrument_keys

        before = {"a.parquet": "0" * 64, "b.duckdb": "stat-size-mtime:10:20"}
        after = {
            "a.parquet": "stat-size-mtime:5:6",
            "b.duckdb": "stat-size-mtime:11:20",
        }

        assert mixed_instrument_keys(before, after) == ["a.parquet"]
        diff = diff_digests(before, after)
        assert diff["mixed"] == ["a.parquet"]
        assert "a.parquet" not in diff["changed"], (
            "the instrument changed between the two observations. Reporting that as a "
            "data move is a false alarm, and a false alarm is how a guard gets deleted."
        )

    def test_a_mixed_key_is_neither_clean_nor_silent(self) -> None:
        from tests.data_boundary import diff_digests, format_digest_diff, is_clean

        before = {"a.parquet": "0" * 64}
        after = {"a.parquet": "stat-size-mtime:5:6"}
        diff = diff_digests(before, after)

        assert not is_clean(diff), (
            "an undecided comparison passed. 'We could not tell' must never read as "
            "'nothing happened'."
        )
        rendered = format_digest_diff(diff, before, after, "data")
        assert "UNDECIDED" in rendered, rendered
        assert "0" * 64 in rendered and "stat-size-mtime:5:6" in rendered, rendered

    def test_two_stat_signatures_that_differ_are_still_a_rewrite(self) -> None:
        from tests.data_boundary import diff_digests

        before = {"b.duckdb": "stat-size-mtime:10:20"}
        after = {"b.duckdb": "stat-size-mtime:11:20"}
        assert diff_digests(before, after)["changed"] == ["b.duckdb"], (
            "both sides came from the SAME instrument, so the comparison is decided "
            "and the file grew by a byte. That is a rewrite."
        )

    def test_two_content_hashes_that_differ_are_still_a_rewrite(self) -> None:
        """No-false-positive control on the mixed detector itself."""
        from tests.data_boundary import diff_digests, mixed_instrument_keys

        before = {"a.parquet": "0" * 64}
        after = {"a.parquet": "1" * 64}
        assert mixed_instrument_keys(before, after) == []
        diff = diff_digests(before, after)
        assert diff["changed"] == ["a.parquet"]
        assert not diff.get("mixed")

    def test_the_mixed_detector_fires_on_a_planted_pair(self) -> None:
        """Fail-closed control: a detector only ever seen returning [] proves nothing."""
        from tests.data_boundary import mixed_instrument_keys

        assert mixed_instrument_keys(
            {"planted.parquet": "0" * 64},
            {"planted.parquet": "stat-size-mtime:1:2"},
        ) == ["planted.parquet"]

    def test_assert_tree_unchanged_treats_a_mixed_key_as_a_hard_failure(
        self, sandbox_store: Path
    ) -> None:
        from tests.data_boundary import DataBoundaryViolation, digest_tree

        before = dict(digest_tree(sandbox_store))
        before["nfl_predictions.duckdb"] = "stat-size-mtime:9:9"

        with pytest.raises(DataBoundaryViolation) as excinfo:
            assert_tree_unchanged(before, digest_tree(sandbox_store), sandbox_store)
        assert "UNDECIDED" in str(excinfo.value), str(excinfo.value)


class TestRequireContentDigestRefusesToDegrade:
    """D33-32: retry after closing handles, then raise BY NAME. Never a signature."""

    def test_a_readable_file_digests_exactly_as_digest_file_does(
        self, sandbox_store: Path
    ) -> None:
        from tests.data_boundary import require_content_digest

        target = sandbox_store / "gold" / "features_wp.parquet"
        value = require_content_digest(target)
        assert len(value) == 64
        assert value == digest_file(target)

    def test_the_retry_budget_is_at_least_one(self) -> None:
        from tests.data_boundary import DIGEST_RETRY_ATTEMPTS

        assert DIGEST_RETRY_ATTEMPTS >= 1, (
            "a zero-retry budget makes the whole D33-32 ruling a rename: the point of "
            "closing handles is to get a second chance at the bytes."
        )

    def test_a_first_call_permission_error_recovers_on_the_retry(
        self, sandbox_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        from tests.data_boundary import require_content_digest

        target = sandbox_store / "nfl_predictions.duckdb"
        real_open = builtins.open
        state = {"calls": 0}

        def flaky_open(file, *args, **kwargs):
            if Path(file) == target:
                state["calls"] += 1
                if state["calls"] == 1:
                    raise PermissionError(13, "Permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", flaky_open)

        value = require_content_digest(target)
        assert len(value) == 64, (
            "the retry produced something other than a content hash -- the whole "
            "point of closing the holder is to get the BYTES."
        )
        assert state["calls"] >= 2, "the retry never happened"

    def test_a_file_that_never_reads_raises_by_name(
        self, sandbox_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        from tests.data_boundary import (
            DIGEST_RETRY_ATTEMPTS,
            LockedStoreDigestError,
            require_content_digest,
        )

        target = sandbox_store / "nfl_predictions.duckdb"
        real_open = builtins.open

        def locked_open(file, *args, **kwargs):
            if Path(file) == target:
                raise PermissionError(13, "Permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", locked_open)

        with pytest.raises(LockedStoreDigestError) as excinfo:
            require_content_digest(target)

        message = str(excinfo.value)
        assert "nfl_predictions.duckdb" in message, message
        assert "PermissionError" in message, message
        assert str(DIGEST_RETRY_ATTEMPTS) in message, message
        assert "HELD HANDLE" in message, (
            "the diagnostic must name the likely holder. This repository already has "
            "that wording in scripts/promote_models.py; a second phrasing of it would "
            "be a second answer to the same question.\n\n" + message
        )

    def test_a_locked_store_failure_is_a_boundary_violation(self) -> None:
        """So every existing `pytest.raises(DataBoundaryViolation)` still catches it."""
        from tests.data_boundary import DataBoundaryViolation, LockedStoreDigestError

        assert issubclass(LockedStoreDigestError, DataBoundaryViolation)

    def test_digest_file_still_declares_a_stat_signature_on_a_locked_file(
        self, sandbox_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D33-32 changes what the GUARD does with a signature, not how one is made.

        The CLI snapshot/verify path and every existing caller depend on the
        self-declaring fallback, and the self-declaration is what makes the mixed
        comparison detectable at all.
        """
        import builtins

        target = sandbox_store / "nfl_predictions.duckdb"
        real_open = builtins.open

        def locked_open(file, *args, **kwargs):
            if Path(file) == target:
                raise PermissionError(13, "Permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", locked_open)
        assert is_stat_signature(digest_file(target))


class TestNoStatSignatureCanReachAVerdict:
    """Asserted by source scan, because the routing IS the D33-32 mitigation."""

    def test_the_autouse_guard_routes_through_require_content_digest(self) -> None:
        import inspect

        from tests import conftest

        source = inspect.getsource(conftest)
        assert "require_content_digest" in source, (
            "the write guard can still render a verdict from a stat signature."
        )

    def test_assert_tree_unchanged_routes_through_require_content_digest(self) -> None:
        import inspect

        from tests import data_boundary

        source = inspect.getsource(data_boundary.assert_tree_unchanged)
        assert "require_content_digest" in source or "_content_resolved" in source, (
            "assert_tree_unchanged still compares whatever it was handed, so a stat "
            "signature can reach a verdict through the opt-in fixtures."
        )
