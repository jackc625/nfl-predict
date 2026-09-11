"""The EFFECT assertion over the owner's single real 2026 capture (D32-01).

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT
----------------------------------------------------
Phase 32's success criterion 1 asks for something no fixture can supply: that a REAL
2026 week is ingested and the sealed pre-2026 zone's content digest is byte-identical
before and after. Every other behaviour in this phase was proven against fixture data
roots, and the suite writes nothing under ``data/`` by design.

D32-01 reconciles that with the ROADMAP's "writes ``config/`` only" constraint by making
the real capture a SINGLE deliberate owner-run step OUTSIDE pytest, bracketed on both
sides by a ``python -m tests.data_boundary snapshot`` digest document. This module reads
those two documents and asserts the effect.

So: it writes nothing, fetches nothing and runs no capture. It is an assertion over
evidence somebody else produced. That separation is the point -- a test that took the
real capture itself would be the suite writing to the production lake, which is the exact
prohibition D32-01 exists to honour.

WHY THE EVIDENCE IS READ RATHER THAN `verify` RE-RUN
-----------------------------------------------------
``python -m tests.data_boundary verify`` returns ``1`` for ANY difference, additions
included -- and a capture ADDS files by definition. Its exit code is therefore not the
verdict for this run; the PARTITION is. Nothing under ``REWRITTEN:``, nothing under
``REMOVED:``, and every path under ``ADDED:`` beginning with ``bronze/``. Reading the two
snapshot documents is what lets that partition be asserted directly instead of inferred
from an integer that cannot express it.

Re-running ``verify`` live would also be wrong in a second way: every later capture, and
any ordinary gold rebuild, moves ``data/`` again. The two documents are a FROZEN record of
one moment, and freezing it is what makes the claim auditable a season later.

ABSENT EVIDENCE SKIPS AUDIBLY, NEVER PASSES QUIETLY (T-32-44)
---------------------------------------------------------------
``outputs/`` is gitignored (``.gitignore:26``), so on any checkout other than the one that
took the capture the two documents are absent. Every skip below carries the REGISTERED
marker ``not present at`` from ``tests/conftest._EVIDENCE_SKIP_MARKERS``, so the terminal
summary names this module in the evidence-backed-controls aggregate rather than letting a
green suite silently exclude it.

THE ONE THING THAT MUST FAIL RATHER THAN SKIP (T-32-43)
---------------------------------------------------------
``tests/data_boundary.digest_file`` falls back to a ``stat-size-mtime:`` signature when a
file cannot be opened -- which on Windows is a LIVE possibility rather than a theoretical
one, because DuckDB holds an exclusive lock on ``data/nfl_predictions.duckdb`` for the
whole of any session that touched it. A comparison that put a content hash on one side and
a stat signature on the other would mean NOTHING, and would report "unchanged" for a file
nobody actually hashed. That case FAILS here. It never skips and it never passes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from data.upstream_pin import LIVE_ZONE_FIRST_SEASON
from tests.data_boundary import (
    PRODUCTION_DATA_ROOT,
    diff_digests,
    format_digest_diff,
    is_stat_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two documents the owner's run produced, in the order it produced them.
_BEFORE = REPO_ROOT / "outputs" / "phase32" / "data_before.json"
_AFTER = REPO_ROOT / "outputs" / "phase32" / "data_after.json"

# The committed record of the same event. Git-TRACKED, so its absence is a broken checkout
# rather than an environment fact -- but it is still read through the same skip idiom,
# because a test that hard-failed on a shallow or partial checkout would be disabled rather
# than fixed.
#
# It was originally tracked only because it had been FORCE-ADDED past .gitignore:254's
# blanket ``*.json``. CR-02 fixed that at the source with a narrow
# ``!config/upstream_live/*.json`` re-include; ``TestTheCommittedRecordSurvivesACheckout``
# below is the guard that keeps the rule in place.
_LIVE_MANIFEST = (
    REPO_ROOT / "config" / "upstream_live" / f"{LIVE_ZONE_FIRST_SEASON}.json"
)

# The three files Phase 32 added to the COMMITTED half of the two-zone record. Each one is
# a record whose whole value is surviving a fresh checkout of a repository whose ``data/``
# tree does not.
_COMMITTED_PHASE_32_RECORDS = (
    "config/upstream_pin.sealed.lock",
    "config/upstream_probe_log.jsonl",
    f"config/upstream_live/{LIVE_ZONE_FIRST_SEASON}.json",
)

# The one directory a capture is allowed to add to. ``data/bronze/`` is the append-only
# archive; everything else under ``data/`` is derived state a capture has no business
# touching.
_BRONZE_PREFIX = "bronze/"

# Every skip reason this module can emit, each carrying the registered ``not present at``
# marker. Listed here so the phrasings are auditable in one place rather than scattered
# across call sites.
_SKIP_REASONS_THIS_MODULE_CAN_EMIT = (
    (
        "the pre-capture boundary evidence is not present at "
        "outputs/phase32/data_before.json -- outputs/ is gitignored, so this control runs "
        "only on the checkout that took the single real capture (D32-01)."
    ),
    (
        "the post-capture boundary evidence is not present at "
        "outputs/phase32/data_after.json -- outputs/ is gitignored, so this control runs "
        "only on the checkout that took the single real capture (D32-01)."
    ),
    (
        "the live-zone manifest is not present at "
        f"config/upstream_live/{LIVE_ZONE_FIRST_SEASON}.json -- no real capture has been "
        "recorded on this checkout."
    ),
)


def _load_digest_document(path: Path, which: str) -> dict[str, str]:
    """Read one snapshot document, skipping AUDIBLY when the evidence is absent."""
    if not path.exists():
        index = 0 if which == "pre" else 1
        pytest.skip(_SKIP_REASONS_THIS_MODULE_CAN_EMIT[index])
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def before_digests() -> dict[str, str]:
    """The per-file content digests of ``data/`` immediately BEFORE the real capture."""
    return _load_digest_document(_BEFORE, "pre")


@pytest.fixture(scope="module")
def after_digests() -> dict[str, str]:
    """The per-file content digests of ``data/`` immediately AFTER the real capture."""
    return _load_digest_document(_AFTER, "post")


class TestTheRealCaptureLeftTheSealedZoneUnmoved:
    """Success criterion 1, asserted from the owner run's own bracketing evidence."""

    def test_no_tracked_path_was_rewritten_or_removed(
        self,
        before_digests: dict[str, str],
        after_digests: dict[str, str],
    ) -> None:
        """THE criterion: not one byte of the pre-existing lake moved.

        A capture writes a NEW timestamped snapshot into the append-only bronze archive.
        If it rewrote or removed anything it reached outside its own contract, and the
        bytes some published verdict was measured against are gone.
        """
        diff = diff_digests(before_digests, after_digests)
        offenders = diff["changed"] + diff["removed"]
        assert not offenders, format_digest_diff(
            diff, before_digests, after_digests, PRODUCTION_DATA_ROOT
        )

    def test_every_added_path_is_a_bronze_addition(
        self,
        before_digests: dict[str, str],
        after_digests: dict[str, str],
    ) -> None:
        """The capture added files, and every one of them landed in ``data/bronze/``.

        The non-emptiness half matters as much as the prefix half: a comparison of two
        identical snapshots would satisfy every OTHER assertion in this class trivially,
        and a capture that added nothing did not happen.
        """
        added = diff_digests(before_digests, after_digests)["added"]
        assert added, (
            "the two boundary snapshots are identical, so no capture happened between "
            "them. This module asserts the EFFECT of the single real 2026 capture "
            "(D32-01); with nothing added there is no effect to assert and every other "
            "assertion here would pass vacuously."
        )

        strays = [key for key in added if not key.startswith(_BRONZE_PREFIX)]
        assert not strays, (
            "the capture added path(s) OUTSIDE the append-only bronze archive: "
            f"{strays}. A live capture writes exactly one timestamped snapshot per "
            f"dataset under {_BRONZE_PREFIX}; anything else under data/ is derived state "
            "it has no business creating."
        )

    def test_no_compared_pair_mixes_a_content_hash_with_a_stat_signature(
        self,
        before_digests: dict[str, str],
        after_digests: dict[str, str],
    ) -> None:
        """T-32-43. A comparison that mixes the two instruments means NOTHING.

        FAILS rather than skips, deliberately. ``tests.data_boundary.digest_file`` falls
        back to ``stat-size-mtime:`` for a file it cannot open, and DuckDB holds
        ``data/nfl_predictions.duckdb`` locked for the whole of any session that touched
        it -- so this is a live hazard on this machine, not a theoretical one. A stat
        signature on either side of a pair would let an unread file report "unchanged",
        which is the precise false negative success criterion 1 cannot afford.
        """
        weakened = sorted(
            {
                key
                for document in (before_digests, after_digests)
                for key, value in document.items()
                if is_stat_signature(value)
            }
        )
        assert not weakened, (
            "these path(s) were digested by the LOCKED-FILE FALLBACK rather than by a "
            f"content hash: {weakened}. A stat-size-mtime signature is a weaker "
            "instrument, and a boundary claim asserted partly through it is not the "
            "byte-identity claim success criterion 1 asks for. Re-take both snapshots "
            "with nothing holding data/nfl_predictions.duckdb open."
        )

    def test_the_live_manifest_records_the_capture_the_evidence_describes(
        self,
        before_digests: dict[str, str],
        after_digests: dict[str, str],
    ) -> None:
        """The committed record and the on-disk additions are the SAME event.

        Without this the two halves of the proof are two coincidences: a manifest that
        claims a capture, and a directory that grew. Joining them on the recorded ``path``
        is what makes them one fact.
        """
        if not _LIVE_MANIFEST.exists():
            pytest.skip(_SKIP_REASONS_THIS_MODULE_CAN_EMIT[2])

        manifest = json.loads(_LIVE_MANIFEST.read_text(encoding="utf-8"))
        recorded = {
            capture["path"]
            for block in manifest.get("datasets", {}).values()
            for capture in block.get("captures", [])
        }
        added = set(diff_digests(before_digests, after_digests)["added"])

        assert recorded & added, (
            "no path recorded in "
            f"config/upstream_live/{LIVE_ZONE_FIRST_SEASON}.json appears among the "
            f"{len(added)} path(s) the boundary evidence says were added. The committed "
            "manifest and the filesystem are describing different events, so neither "
            "corroborates the other.\n"
            f"  manifest records: {sorted(recorded)}\n"
            f"  evidence added:   {sorted(added)}"
        )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one git command at the repository root and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheCommittedRecordSurvivesACheckout:
    """CR-02. The committed half of the two-zone claim has to actually be committed.

    ``data/upstream_live.py``'s module docstring states the premise outright: "``config/``
    is committed and ``data/`` is gitignored, for both zones: the record of WHICH upstream
    revision a verdict was measured against survives a fresh checkout even when the parquet
    bytes do not." That was never CONFIGURED for the live-zone directory --
    ``.gitignore``'s blanket ``*.json`` matched ``config/upstream_live/<season>.json``, and
    the 2026 file was tracked only because somebody force-added it once.

    Tracking is a property of the INDEX, not of the path, so the force-add covered exactly
    one file. 2027.json at the season roll -- and any manifest recreated on a checkout where
    it had ever been removed -- would have been ignored silently: ``git add
    config/upstream_live/`` skipping it without a word, ``git status`` not showing it, no
    error anywhere. A record that fails to exist with no error is the precise failure shape
    this phase spends itself preventing, so the rule gets a test rather than a convention.
    """

    @pytest.mark.parametrize("relative", _COMMITTED_PHASE_32_RECORDS)
    def test_the_committed_record_is_tracked_by_git(self, relative: str) -> None:
        """Each Phase-32 record is in the index, not merely on disk."""
        tracked = _git("ls-files", "--", relative)
        assert tracked.returncode == 0, (
            f"`git ls-files -- {relative}` failed: {tracked.stderr.strip()}"
        )
        assert tracked.stdout.strip(), (
            f"{relative} is NOT tracked by git. .gitignore's blanket *.json (or a future "
            "rule) swallowed it, so the committed record does not survive a fresh "
            "checkout -- and the parquet it accounts for lives under gitignored data/, so "
            "nothing anywhere would say which upstream revision a verdict was measured "
            "against."
        )

    def test_a_future_seasons_live_manifest_is_not_ignored(self) -> None:
        """The rule, not the index: the season AFTER the live one must be trackable too.

        This is the assertion the force-add could never make. ``--no-index`` asks purely
        about the ignore rules, and ``-q`` is the spelling that answers the question --
        without it ``check-ignore`` exits 0 for a NEGATED match too, reporting "a pattern
        matched" rather than "this path is ignored".
        """
        future = f"config/upstream_live/{LIVE_ZONE_FIRST_SEASON + 1}.json"
        ignored = _git("check-ignore", "-q", "--no-index", "--", future)

        assert ignored.returncode == 1, (
            f"{future} is IGNORED by .gitignore, so next season's committed capture "
            "record would silently fail to be tracked. Keep the narrow "
            "`!config/upstream_live/*.json` re-include in .gitignore.\n"
            "  matching rule: "
            + (
                _git("check-ignore", "-v", "--no-index", "--", future).stdout.strip()
                or "(none reported)"
            )
        )

    def test_the_re_include_stays_narrow_and_does_not_re_track_every_json(self) -> None:
        """The fix re-includes ONE directory's manifests, and nothing else.

        ``.gitignore:254``'s blanket ``*.json`` is load-bearing -- it is what keeps every
        generated JSON under ``outputs/`` and ``data/`` out of history. A re-include that
        widened past the live-zone directory would trade one silent failure for a much
        larger one.
        """
        stray = "outputs/phase32/data_before.json"
        ignored = _git("check-ignore", "-q", "--no-index", "--", stray)

        assert ignored.returncode == 0, (
            f"{stray} is no longer ignored, so the CR-02 re-include reached past "
            "config/upstream_live/ and started tracking generated JSON."
        )


class TestThisModuleCannotGoQuiet:
    """The guard on the guard: every skip path stays registered and visible."""

    def test_every_skip_reason_carries_a_registered_evidence_marker(self) -> None:
        """A new skip phrasing must be registered, not invented silently.

        ``tests/conftest.py``'s evidence-backed-controls footer matches on the skip REASON
        text. A reason outside that vocabulary makes this module's non-run invisible in the
        aggregate -- which is T-32-44 exactly: a green suite that quietly excluded the one
        control asserting success criterion 1.
        """
        from tests.conftest import is_evidence_backed_skip

        unregistered = [
            reason
            for reason in _SKIP_REASONS_THIS_MODULE_CAN_EMIT
            if not is_evidence_backed_skip(reason)
        ]
        assert not unregistered, (
            "these skip reason(s) carry no marker from "
            f"tests/conftest._EVIDENCE_SKIP_MARKERS: {unregistered}. Register the "
            "phrasing there or reword the reason; an unregistered skip is a control that "
            "did not run and did not say so."
        )
