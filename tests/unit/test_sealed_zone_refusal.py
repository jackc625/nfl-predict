"""The sealed zone REFUSES, and the refusal can be shown to bind in both directions.

TEST CLASS: plain unit tests. Every pin these tests mutate is a copy made inside
``tmp_path``, and every capture is driven through a stubbed ``fetch_live``, so the module
passes on a fresh checkout with no ``data/`` and no network. The one class that reads the
COMMITTED records skips with an evidence-backed reason when they are absent -- using the
vocabulary ``tests/conftest.py`` matches on, so an absent-evidence skip is surfaced in the
aggregate rather than disappearing into a green summary line.

Phase 32 (PIN-01, D32-02) gives the sealed zone -- seasons at or before
:data:`data.upstream_pin.SEALED_THROUGH_SEASON` -- BOTH halves of the guarantee, because
either half alone leaves the hazard live:

1. DETECTION. ``config/upstream_pin.sealed.lock`` is a committed lock on the sealed half
   of ``config/upstream_pin.json``. A hand edit that never runs the capture tool shows
   only as a git diff, and a git diff is something a reader has to happen to notice.
   Divergence FAILS this module instead, naming the dataset, the season and both digests.
2. PREVENTION. ``scripts/pin_upstream_snapshot.py::capture`` today REPLACES an
   already-pinned season's manifest entry (its own docstring said so). That is right for
   EXTENDING a pin and wrong for a SEALED one, so the write is now refused before any
   fetch unless it carries an explicit, attributed override.
3. Every refusal names a command that the write gate would actually accept. A refusal
   that prints a command the tool rejects is worse than printing no advice at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from data.upstream_pin import (
    LIVE_ZONE_FIRST_SEASON,
    MANIFEST_PATH,
    SEALED_LOCK_PATH,
    load_manifest,
    load_sealed_lock,
    sealed_lock_problems,
)
from scripts.pin_upstream_snapshot import refresh_sealed_lock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_committed_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the committed manifest and lock into *tmp_path*; return (manifest, lock).

    The mutation tests deliberately run against COPIES of the real committed records
    rather than a synthetic two-season stand-in: the comparator has to hold on the shape
    that actually exists (76 sealed pairs across three datasets), not on a shape written
    to make it pass.
    """
    source_manifest = REPO_ROOT / MANIFEST_PATH
    source_lock = REPO_ROOT / SEALED_LOCK_PATH
    for source in (source_manifest, source_lock):
        if not source.is_file():
            pytest.skip(
                f"the sealed-zone record is not present at {source} -- no pin has been "
                "captured on this checkout."
            )
    manifest_path = tmp_path / "upstream_pin.json"
    lock_path = tmp_path / "upstream_pin.sealed.lock"
    shutil.copyfile(source_manifest, manifest_path)
    shutil.copyfile(source_lock, lock_path)
    return manifest_path, lock_path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _first_sealed_pair(lock: dict) -> tuple[str, str]:
    """Return the first ``(dataset, season)`` the lock records, as strings."""
    dataset = sorted(lock["datasets"])[0]
    season = sorted(lock["datasets"][dataset], key=int)[0]
    return dataset, season


class TestTheSealedLockBindsTheCommittedManifest:
    """D32-02 detection: a sealed-zone change FAILS the suite, not merely a git diff."""

    def test_the_committed_lock_and_the_committed_manifest_agree(self) -> None:
        """The one case that reads the REAL committed pair. It must run, not skip.

        ``sealed_lock_problems`` returns a LIST rather than a bool precisely so this
        assertion prints the offenders. A bool would report that something moved and
        leave the reader to go find out what.
        """
        manifest_path = REPO_ROOT / MANIFEST_PATH
        lock_path = REPO_ROOT / SEALED_LOCK_PATH
        for path in (manifest_path, lock_path):
            if not path.is_file():
                pytest.skip(
                    f"the sealed-zone record is not present at {path} -- no pin has "
                    "been captured on this checkout."
                )

        problems = sealed_lock_problems(
            load_sealed_lock(lock_path), load_manifest(manifest_path)
        )

        assert problems == [], (
            "the committed sealed pin no longer matches the committed sealed lock. A "
            "sealed season's bytes are immutable by definition, so this is either a "
            "hand edit to the manifest or an unattributed re-capture:\n  "
            + "\n  ".join(problems)
        )

    def test_a_moved_sha256_in_the_manifest_is_reported(self, tmp_path: Path) -> None:
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        lock = _read(lock_path)
        dataset, season = _first_sealed_pair(lock)
        locked_digest = lock["datasets"][dataset][season]["sha256"]
        moved_digest = "0" * 64

        manifest = _read(manifest_path)
        manifest["datasets"][dataset]["seasons"][season]["sha256"] = moved_digest
        _write(manifest_path, manifest)

        problems = sealed_lock_problems(
            load_sealed_lock(lock_path), load_manifest(manifest_path)
        )

        assert len(problems) == 1, (
            f"one moved digest should produce exactly one problem, got {problems}"
        )
        assert dataset in problems[0]
        assert season in problems[0]
        assert locked_digest in problems[0], "the locked digest is not reported"
        assert moved_digest in problems[0], "the manifest's digest is not reported"

    def test_a_season_dropped_from_the_lock_is_reported(self, tmp_path: Path) -> None:
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        lock = _read(lock_path)
        dataset, season = _first_sealed_pair(lock)
        del lock["datasets"][dataset][season]
        _write(lock_path, lock)

        problems = sealed_lock_problems(
            load_sealed_lock(lock_path), load_manifest(manifest_path)
        )

        assert [p for p in problems if "MISSING FROM LOCK" in p and season in p], (
            f"a sealed season pinned but not locked was not reported: {problems}"
        )

    def test_a_season_dropped_from_the_manifest_is_reported(
        self, tmp_path: Path
    ) -> None:
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        lock = _read(lock_path)
        dataset, season = _first_sealed_pair(lock)

        manifest = _read(manifest_path)
        del manifest["datasets"][dataset]["seasons"][season]
        _write(manifest_path, manifest)

        problems = sealed_lock_problems(
            load_sealed_lock(lock_path), load_manifest(manifest_path)
        )

        assert [p for p in problems if "MISSING FROM MANIFEST" in p and season in p], (
            f"a locked season deleted from the manifest was not reported: {problems}"
        )

    def test_a_live_zone_season_in_the_sealed_lock_is_reported(
        self, tmp_path: Path
    ) -> None:
        """The two records must never merge: their diffs mean opposite things."""
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        lock = _read(lock_path)
        dataset, season = _first_sealed_pair(lock)
        lock["datasets"][dataset][str(LIVE_ZONE_FIRST_SEASON)] = dict(
            lock["datasets"][dataset][season]
        )
        _write(lock_path, lock)

        problems = sealed_lock_problems(
            load_sealed_lock(lock_path), load_manifest(manifest_path)
        )

        offenders = [p for p in problems if "LIVE-ZONE SEASON IN SEALED LOCK" in p]
        assert offenders, (
            f"a live-zone season sitting in the SEALED lock was not reported: {problems}"
        )
        assert str(LIVE_ZONE_FIRST_SEASON) in offenders[0]
        assert "upstream_live" in offenders[0], (
            "the refusal does not name the record that season actually belongs in"
        )

    def test_regeneration_preserves_an_acknowledgement(self, tmp_path: Path) -> None:
        """D32-10: an acknowledgement is a committed, attributed ruling.

        Re-deriving the lock must never silently drop one -- that would re-arm a
        CRITICAL the owner has already ruled on, with nothing to tell them their ruling
        had evaporated.
        """
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        lock = _read(lock_path)
        dataset, season = _first_sealed_pair(lock)
        acknowledgement = {
            "ruled_at_utc": "2026-09-11T00:00:00+00:00",
            "ruled_by": "owner",
            "reason": "known divergence, stable; the pin is NOT re-captured to match",
            "observed_signature": {"etag": 'W/"deadbeef"', "content_length": 123456},
        }
        lock["datasets"][dataset][season]["acknowledgement"] = acknowledgement
        lock["datasets"][dataset][season]["upstream_updated_at"] = (
            "Mon, 01 Sep 2026 00:00:00 GMT"
        )
        lock["datasets"][dataset][season]["upstream_size"] = 123456
        _write(lock_path, lock)

        regenerated = refresh_sealed_lock(manifest_path, lock_path)

        entry = regenerated["datasets"][dataset][season]
        assert entry["acknowledgement"] == acknowledgement, (
            "regeneration erased a recorded acknowledgement"
        )
        assert entry["upstream_updated_at"] == "Mon, 01 Sep 2026 00:00:00 GMT"
        assert entry["upstream_size"] == 123456
        assert _read(lock_path)["datasets"][dataset][season] == entry, (
            "the regenerated document on disk differs from the one returned"
        )

    def test_regeneration_leaves_the_signature_slots_null_where_none_was_recorded(
        self, tmp_path: Path
    ) -> None:
        """``null`` means "not yet observed", which is not the same claim as absent."""
        manifest_path, lock_path = _copy_committed_pair(tmp_path)

        regenerated = refresh_sealed_lock(manifest_path, lock_path)

        for dataset, seasons in regenerated["datasets"].items():
            for season, entry in seasons.items():
                for field in (
                    "sha256",
                    "rows",
                    "bytes",
                    "upstream_updated_at",
                    "upstream_size",
                    "acknowledgement",
                ):
                    assert field in entry, f"{dataset} {season} has no {field!r} slot"
                assert entry["acknowledgement"] is None
