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
import shlex
import shutil
from pathlib import Path

import pandas as pd
import pytest

from data import upstream_pin
from data.upstream_pin import (
    LIVE_ZONE_FIRST_SEASON,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    PBP_PINNED_COLUMNS,
    SEALED_LOCK_PATH,
    UpstreamPinCorrupt,
    UpstreamPinMissing,
    ZoneWriteRefused,
    load_manifest,
    load_sealed_lock,
    sealed_lock_problems,
)
from scripts import pin_upstream_snapshot as pin
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

    def test_a_repin_and_its_lock_refresh_keep_every_untouched_top_level_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2026-09-30: a two-season re-pin rewrote not_pinned / captured_at_utc, and the lock
        refresh dropped the signature seeding record. Only the re-pinned entries may move."""
        manifest_path, lock_path = _copy_committed_pair(tmp_path)
        manifest_before, lock_before = _read(manifest_path), _read(lock_path)
        monkeypatch.setattr(
            pin, "fetch_live", lambda dataset, season: _pbp_frame(season)
        )

        pin.capture(
            {"pbp": [2024]},
            manifest_path=manifest_path,
            data_root=tmp_path / "data",
            allow_sealed_rewrite=True,
            sealed_rewrite_reason="test: re-pin one sealed season",
        )
        refresh_sealed_lock(manifest_path, lock_path)

        manifest_after, lock_after = _read(manifest_path), _read(lock_path)
        for record in (manifest_before, manifest_after):
            del record["datasets"]
        for record in (lock_before, lock_after):
            del record["datasets"], record["generated_at_utc"]
        assert manifest_after == manifest_before
        assert lock_after == lock_before


def _tmp_pin(tmp_path: Path, dataset: str, seasons: list[int]) -> Path:
    """Write a manifest under *tmp_path* pinning *seasons*; return its path.

    The recorded bytes are fictional on purpose. Every test that uses this manifest
    asserts a refusal that happens BEFORE anything is fetched, read or written, so a
    real parquet would only prove that the refusal came too late to matter.
    """
    entries = {
        str(season): {
            "path": f"bronze/{dataset}_raw_bronze_{season}_W00_20260905T000000.parquet",
            "sha256": "a" * 64,
            "bytes": 1024,
            "rows": 128,
            "columns": list(PBP_PINNED_COLUMNS),
            "upstream_width": 372,
            "captured_at_utc": "2026-09-05T00:00:00+00:00",
        }
        for season in seasons
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "test fixture",
        "captured_at_utc": "2026-09-05T00:00:00+00:00",
        "datasets": {dataset: {"loader": "test", "seasons": entries}},
    }
    manifest_path = tmp_path / "upstream_pin.json"
    _write(manifest_path, manifest)
    return manifest_path


def _pbp_frame(season: int) -> pd.DataFrame:
    """A minimal frame carrying enough of the pinned play-by-play column set."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_01_HOME_AWAY", f"{season}_01_HOME_AWAY"],
            "season": [season, season],
            "week": [1, 1],
            "posteam": ["HOME", "AWAY"],
            "defteam": ["AWAY", "HOME"],
            "epa": [0.25, -0.25],
        }
    )


@pytest.fixture
def fetching_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any nflverse fetch an outright failure, and record the attempts.

    ``scripts.pin_upstream_snapshot.fetch_live`` is the ONE place the capture tool can
    reach the network. Replacing it with a raising stub means a test that passes
    provably refused BEFORE fetching, rather than merely refusing eventually.
    """
    attempts: list[str] = []

    def _explode(dataset: str, season: int) -> pd.DataFrame:
        attempts.append(f"{dataset}:{season}")
        msg = "the network was reached, but this test asserts the write was refused"
        raise AssertionError(msg)

    monkeypatch.setattr(pin, "fetch_live", _explode)
    return attempts


class TestTheSealedZoneRefusesARewrite:
    """D32-02 prevention: the one tool that can rewrite the sealed zone refuses to."""

    def test_capturing_an_already_pinned_sealed_season_refuses_and_writes_nothing(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        """Refused BEFORE the loop and before any fetch, so nothing moves at all."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        before = manifest_path.read_bytes()

        with pytest.raises(ZoneWriteRefused):
            pin.capture(
                {"pbp": [2024]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            )

        assert manifest_path.read_bytes() == before, (
            "the manifest was rewritten by a refused run"
        )
        assert fetching_is_a_failure == [], (
            "the refusal arrived AFTER a fetch. A check that runs after the work is a "
            "check nobody can afford to trust."
        )
        assert not (tmp_path / "data").exists(), "a refused run created a data root"

    def test_the_refusal_names_the_override_flag_and_requires_a_reason(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])

        with pytest.raises(ZoneWriteRefused) as error:
            pin.capture(
                {"pbp": [2024]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            )

        message = str(error.value)
        assert "--allow-sealed-rewrite" in message
        assert "--sealed-rewrite-reason" in message
        assert "2026-08-22" in message, (
            "the refusal cites no concrete incident, so it reads as bureaucracy"
        )

    def test_the_override_without_a_reason_still_refuses(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        """An unattributed override is the same silent rewrite with an extra flag."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        before = manifest_path.read_bytes()

        for blank in (None, "", "   "):
            with pytest.raises(ZoneWriteRefused) as error:
                pin.capture(
                    {"pbp": [2024]},
                    manifest_path=manifest_path,
                    data_root=tmp_path / "data",
                    allow_sealed_rewrite=True,
                    sealed_rewrite_reason=blank,
                )
            assert "--sealed-rewrite-reason" in str(error.value)

        assert manifest_path.read_bytes() == before
        assert fetching_is_a_failure == []

    def test_the_override_with_a_reason_proceeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        monkeypatch.setattr(
            pin, "fetch_live", lambda dataset, season: _pbp_frame(season)
        )

        manifest = pin.capture(
            {"pbp": [2024]},
            manifest_path=manifest_path,
            data_root=tmp_path / "data",
            allow_sealed_rewrite=True,
            sealed_rewrite_reason="re-captured after the 2026-09-11 audit",
        )

        entry = manifest["datasets"]["pbp"]["seasons"]["2024"]
        assert entry["sha256"] != "a" * 64, "the pinned entry was not replaced"
        assert entry["rows"] == 2
        assert _read(manifest_path)["datasets"]["pbp"]["seasons"]["2024"] == entry
        assert "re-captured after the 2026-09-11 audit" in capsys.readouterr().err, (
            "an accepted override left no attribution where an operator would see it"
        )

    def test_a_sealed_season_not_already_pinned_captures_with_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extending the pin is the ordinary use of this tool and stays unguarded."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        monkeypatch.setattr(
            pin, "fetch_live", lambda dataset, season: _pbp_frame(season)
        )

        manifest = pin.capture(
            {"pbp": [2023]},
            manifest_path=manifest_path,
            data_root=tmp_path / "data",
        )

        assert sorted(manifest["datasets"]["pbp"]["seasons"]) == ["2023", "2024"]

    def test_capturing_a_live_zone_season_refuses_and_names_the_live_cli(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        """The exact mirror of the refusal capture_live_season raises for a sealed one."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        before = manifest_path.read_bytes()

        with pytest.raises(ZoneWriteRefused) as error:
            pin.capture(
                {"pbp": [LIVE_ZONE_FIRST_SEASON]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            )

        message = str(error.value)
        assert "scripts.capture_live_season" in message
        assert "LIVE zone" in message
        assert "--allow-sealed-rewrite" not in message, (
            "a live-zone write has NO override; naming one would invite the attempt"
        )
        assert manifest_path.read_bytes() == before
        assert fetching_is_a_failure == []

    def test_capturing_a_2027_season_refuses_and_names_the_deferred_seal_tool(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        """Was: refused as "NO zone owns it" (step 27b: every later season is LIVE). Intent
        kept: the SEALED pin still refuses it before any fetch, with no override, and names
        the one-way human seal as the only way it ever becomes sealed."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        before = manifest_path.read_bytes()

        with pytest.raises(ZoneWriteRefused) as error:
            pin.capture(
                {"pbp": [LIVE_ZONE_FIRST_SEASON + 1]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            )

        message = str(error.value)
        assert "LIVE zone" in message
        assert "--allow-sealed-rewrite" not in message
        assert manifest_path.read_bytes() == before
        assert "seal_season.py" in message
        assert "SEALED_THROUGH_SEASON" in message
        assert "one-way" in message.lower()
        assert fetching_is_a_failure == []

    def test_the_gate_runs_before_the_loop_so_a_mixed_request_writes_nothing(
        self, tmp_path: Path, fetching_is_a_failure: list[str]
    ) -> None:
        """One refusable season in the request refuses the WHOLE request.

        Capturing the allowed half first and refusing the rest would leave the manifest
        half-written by a run the operator was told had failed.
        """
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        before = manifest_path.read_bytes()

        with pytest.raises(ZoneWriteRefused):
            pin.capture(
                {"pbp": [2023, LIVE_ZONE_FIRST_SEASON]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            )

        assert manifest_path.read_bytes() == before
        assert fetching_is_a_failure == []

    @pytest.mark.parametrize("swallowed", [ImportError, ValueError, RuntimeError])
    def test_every_refusal_is_outside_the_wired_call_sites_except_clause(
        self, swallowed: type[Exception]
    ) -> None:
        """PinCaptureError is a RuntimeError; a refusal must not be.

        ``features/qb_tracking.py`` catches ``(ImportError, ValueError, RuntimeError)``
        around its loaders and returns an EMPTY frame. A write refusal caught there
        would become a silently degraded gold matrix -- worse than the drift.
        """
        assert not issubclass(ZoneWriteRefused, swallowed)
        assert issubclass(pin.PinCaptureError, RuntimeError), (
            "this test is vacuous unless PinCaptureError really is the swallowed kind"
        )


# The literal marker a printed module invocation carries. The repository's own convention
# is the interpreter's full path -- ".venv/Scripts/python.exe -m scripts.<module>" -- so
# this, and not "python -m scripts.", is the substring that actually appears in a message
# and distinguishes a runnable command from prose about one.
_MODULE_INVOCATION = "-m scripts."
_PIN_INVOCATION = "-m scripts.pin_upstream_snapshot"


def _pin_commands(message: str) -> list[list[str]]:
    """Every CONCRETE pin invocation in *message*, shlex-split into argv tokens.

    TEMPLATE lines are skipped. A message may legitimately print the FORM of a command
    (``--seasons <S> <S>``) when it is teaching the shape rather than answering a
    specific situation; those cannot be parsed and are not what this check is about.
    A line is a template when any token outside a quoted ``--sealed-rewrite-reason``
    value is an angle-bracket placeholder.
    """
    commands: list[list[str]] = []
    for line in message.splitlines():
        if _PIN_INVOCATION not in line:
            continue
        _, _, tail = line.partition(_PIN_INVOCATION)
        tokens = shlex.split(tail)
        placeholders = [
            token
            for index, token in enumerate(tokens)
            if token.startswith("<")
            and (index == 0 or tokens[index - 1] != "--sealed-rewrite-reason")
        ]
        if placeholders:
            continue
        commands.append(tokens)
    return commands


def _refusal_text(raises: type[Exception], call) -> str:
    with pytest.raises(raises) as error:
        call()
    return str(error.value)


class TestEveryRefusalCarriesARunnableRecoveryCommand:
    """A refusal that prints a command the write gate rejects is worse than silence.

    Task 2 turned two of the commands this package PRINTS as recovery advice into
    commands that would now be refused. This class is the "refusals carry the recovery
    command" pattern applied to itself: every scenario provokes a REAL refusal inside
    ``tmp_path`` and then feeds the command it names back through ``build_parser`` and
    ``assert_write_allowed``.
    """

    # Each row is an id, the builder method's name, whether the refusal is about an
    # UNCOVERED season, and whether it names a runnable command at all.
    #
    # ``uncovered_later_live_season`` named no command until step 27b, when every season
    # after the first live one became LIVE: the live capture opens it once the season before
    # it has ended, so its refusal names that capture like any live season's.
    SCENARIOS = (
        ("uncovered_sealed_season", "_uncovered_sealed", True, True),
        ("uncovered_live_season", "_uncovered_live", True, True),
        ("uncovered_later_live_season", "_uncovered_beyond", True, True),
        ("pinned_season_whose_bytes_are_absent", "_missing_bytes", False, True),
        ("pinned_season_whose_digest_moved", "_digest_moved", False, True),
        ("sealed_rewrite_without_the_override", "_sealed_rewrite", False, True),
        ("live_zone_write_aimed_at_the_sealed_pin", "_live_zone_write", False, True),
    )

    @staticmethod
    def _uncovered_sealed(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            UpstreamPinMissing,
            lambda: upstream_pin.load_pbp(
                [2001],
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            ),
        )
        return message, load_manifest(manifest_path)

    @staticmethod
    def _uncovered_live(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            UpstreamPinMissing,
            lambda: upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON],
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            ),
        )
        return message, load_manifest(manifest_path)

    @staticmethod
    def _uncovered_beyond(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            UpstreamPinMissing,
            lambda: upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON + 1],
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            ),
        )
        return message, load_manifest(manifest_path)

    @staticmethod
    def _missing_bytes(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        manifest = load_manifest(manifest_path)
        assert manifest is not None
        message = _refusal_text(
            UpstreamPinCorrupt,
            lambda: upstream_pin._read_pinned_frame(
                "pbp", 2024, manifest, tmp_path / "data"
            ),
        )
        return message, manifest

    @staticmethod
    def _digest_moved(tmp_path: Path) -> tuple[str, dict | None]:
        data_root = tmp_path / "data"
        (data_root / "bronze").mkdir(parents=True)
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        manifest = load_manifest(manifest_path)
        assert manifest is not None
        entry = manifest["datasets"]["pbp"]["seasons"]["2024"]
        _pbp_frame(2024).to_parquet(data_root / entry["path"], index=False)
        message = _refusal_text(
            UpstreamPinCorrupt,
            lambda: upstream_pin._read_pinned_frame("pbp", 2024, manifest, data_root),
        )
        return message, manifest

    @staticmethod
    def _sealed_rewrite(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            ZoneWriteRefused,
            lambda: pin.capture(
                {"pbp": [2024]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            ),
        )
        return message, load_manifest(manifest_path)

    @staticmethod
    def _live_zone_write(tmp_path: Path) -> tuple[str, dict | None]:
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            ZoneWriteRefused,
            lambda: pin.capture(
                {"pbp": [LIVE_ZONE_FIRST_SEASON]},
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
            ),
        )
        return message, load_manifest(manifest_path)

    @pytest.mark.parametrize(
        ("builder", "is_uncovered_refusal", "names_a_command"),
        [
            (builder, uncovered, commanded)
            for _, builder, uncovered, commanded in SCENARIOS
        ],
        ids=[name for name, _, _, _ in SCENARIOS],
    )
    def test_the_refusal_names_a_command_the_write_gate_would_accept(
        self,
        tmp_path: Path,
        builder: str,
        is_uncovered_refusal: bool,
        names_a_command: bool,
    ) -> None:
        message, manifest = getattr(self, builder)(tmp_path)

        if names_a_command:
            assert _MODULE_INVOCATION in message, (
                "the refusal names no runnable module invocation, only prose:\n"
                + message
            )
        else:
            assert _MODULE_INVOCATION not in message, (
                "this refusal has no runnable recovery and must not imply one:\n"
                + message
            )
            assert "SEALED_THROUGH_SEASON" in message and "one-way" in message, (
                "a refusal with no command must at least name the human act that IS "
                "the recovery:\n" + message
            )
        if is_uncovered_refusal:
            assert "zone" in message.lower(), (
                "an uncovered-season refusal that does not name the ZONE cannot point "
                "at the right capture tool:\n" + message
            )

        for tokens in _pin_commands(message):
            args = pin.build_parser().parse_args(tokens)
            if not args.dataset or not args.seasons:
                continue
            first, last = args.seasons
            requested = {
                name: list(range(first, last + 1)) for name in sorted(set(args.dataset))
            }
            pin.assert_write_allowed(
                requested,
                manifest,
                allow_sealed_rewrite=args.allow_sealed_rewrite,
                sealed_rewrite_reason=args.sealed_rewrite_reason,
            )

    def test_at_least_one_scenario_names_a_concrete_checkable_command(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the loop above passes by finding nothing to check."""
        checked = 0
        for index, (_, builder, _, _) in enumerate(self.SCENARIOS):
            scenario_dir = tmp_path / str(index)
            scenario_dir.mkdir()
            message, _ = getattr(self, builder)(scenario_dir)
            checked += len(_pin_commands(message))
        assert checked >= 3, (
            f"only {checked} concrete pin command(s) across all scenarios -- the gate "
            "check is close to vacuous"
        )

    def test_the_historical_incident_paragraph_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """It is why anyone reads the refusal at all, so it stays verbatim."""
        manifest_path = _tmp_pin(tmp_path, "pbp", [2024])
        message = _refusal_text(
            UpstreamPinMissing,
            lambda: upstream_pin.load_pbp(
                [2001],
                manifest_path=manifest_path,
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            ),
        )
        assert "2026-08-22" in message
        assert "sixteen opponent-adjusted and QB columns" in message
        assert "NFL_PREDICT_ALLOW_LIVE_UPSTREAM=1 <your command>" in message
        assert "UpstreamPinBypassedWarning" in message

    def test_a_pinned_seasons_recovery_command_is_the_override_form(
        self, tmp_path: Path
    ) -> None:
        """The specific reconciliation Task 3 exists for, asserted directly."""
        message, _ = self._missing_bytes(tmp_path)
        assert "--allow-sealed-rewrite" in message
        assert "--sealed-rewrite-reason" in message

        message, _ = self._digest_moved(tmp_path / "digest")
        assert "--allow-sealed-rewrite" in message
        assert "--sealed-rewrite-reason" in message

    def test_an_unpinned_sealed_season_is_still_told_the_plain_command(
        self, tmp_path: Path
    ) -> None:
        """Extending the pin never needed an override and must not be told it does."""
        message, _ = self._uncovered_sealed(tmp_path)
        assert "--dataset pbp --seasons 2001 2001" in message
        assert "--allow-sealed-rewrite" not in message
