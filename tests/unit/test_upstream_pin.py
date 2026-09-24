"""The upstream pin BINDS, and it can be shown to fail.

TEST CLASS: plain unit tests. Every pin these tests read is built inside ``tmp_path`` from
frames constructed in the test, so the module passes on a fresh checkout with no
``data/``, no network and no captured pin. The two classes that read the COMMITTED
manifest or the real snapshots skip with an evidence-backed reason when those are absent.

A pin nobody can show binding is decoration. Four separate things are proved here, and
each is proved in BOTH directions:

1. With a pin present, the loader does not reach the network -- proved by a monkeypatched
   ``nflreadpy`` whose every loader raises, so a network read would fail the test.
2. The frame really comes FROM the pin -- proved by POISONING the pinned parquet (and
   updating its recorded digest to match, so the poisoned bytes are legitimately the
   pin) and reading the poisoned value back out.
3. A pin whose bytes no longer match its recorded digest is REFUSED, not used.
4. A missing pin REFUSES rather than silently refetching, and the live loader is proved
   uncalled. Live refetching happens only under the explicit environment opt-in, and it
   warns when it does.

Phase 32 adds a fifth, and it is proved in both directions too: WHICH live capture a read
serves is a process-level as-of (D32-15), so one setting reaches all three independent
builder call chains without any of them being modified. The positive direction is three
tests that set the as-of and read back the marker value of the addressed capture through
the exact call shapes ``features/team_form.py``, ``features/qb_tracking.py`` and
``scripts/ingest_games.py`` use today. The negative direction is an ``ast`` assertion that
none of those three modules passes an ``as_of`` keyword -- the threaded-keyword design
D32-15 rejected, and the one a later change would drift back toward.
"""

from __future__ import annotations

import ast
import json
import sys
import types
import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

from data import upstream_pin
from data.upstream_live import (
    AS_OF_ENV,
    LIVE_MANIFEST_SCHEMA_VERSION,
    AsOfCapture,
    as_of_capture,
    current_as_of,
)
from data.upstream_pin import (
    LIVE_OPT_IN_ENV,
    LIVE_ZONE_FIRST_SEASON,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    PBP_PINNED_COLUMNS,
    SEALED_THROUGH_SEASON,
    ZONE_LIVE,
    ZONE_SEALED,
    UpstreamLiveCaptureMissing,
    UpstreamLiveCorrupt,
    UpstreamPinBypassedWarning,
    UpstreamPinCorrupt,
    UpstreamPinError,
    UpstreamPinMissing,
    UpstreamSeasonWindowRefused,
    ZoneWriteRefused,
    digest_file,
    zone_for_season,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The value the poisoning test writes into the pinned parquet. It is deliberately absurd:
# no real EPA is 999.0, so seeing it come back out of a loader is unambiguous evidence
# that the loader read the file rather than the network.
POISON_EPA = 999.0


def _pbp_frame(season: int, epa: float = 0.25) -> pd.DataFrame:
    """A minimal frame carrying the pinned play-by-play column set."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_01_HOME_AWAY", f"{season}_01_HOME_AWAY"],
            "season": [season, season],
            "week": [1, 1],
            "posteam": ["HOME", "AWAY"],
            "defteam": ["AWAY", "HOME"],
            "epa": [epa, -epa],
        }
    )


def _write_pin(
    tmp_path: Path,
    dataset: str,
    frames: dict[int, pd.DataFrame],
) -> tuple[Path, Path]:
    """Write a pin for *frames* under *tmp_path*; return (manifest_path, data_root)."""
    data_root = tmp_path / "data"
    (data_root / "bronze").mkdir(parents=True, exist_ok=True)
    seasons: dict[str, dict] = {}
    for season, frame in frames.items():
        relative = f"bronze/{dataset}_raw_bronze_{season}_W00_20260905T000000.parquet"
        path = data_root / relative
        frame.to_parquet(path, index=False)
        seasons[str(season)] = {
            "path": relative,
            "sha256": digest_file(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "columns": list(frame.columns),
            "captured_at_utc": "2026-09-05T00:00:00+00:00",
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "test fixture",
        "captured_at_utc": "2026-09-05T00:00:00+00:00",
        "nflreadpy_version": "0.1.5",
        "datasets": {dataset: {"loader": "test", "seasons": seasons}},
    }
    manifest_path = tmp_path / "upstream_pin.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, data_root


# The marker EPA each LIVE capture carries, keyed by the ``(week, sequence)`` address that
# holds it. The digits are the address itself, so a failing assertion says WHICH capture
# was read rather than only that the value was wrong -- and reading "the newest" when an
# as-of asked for week 6 sequence 1 shows up as 7.1 instead of 6.1.
CAPTURE_MARKERS: dict[tuple[int, int], float] = {
    (6, 1): 6.1,
    (6, 2): 6.2,
    (7, 1): 7.1,
}

# The live capture addresses the two-zone fixture below writes, in append order.
CAPTURE_ADDRESSES: tuple[tuple[int, int], ...] = ((6, 1), (6, 2), (7, 1))

# The datasets the fixture captures. All three, because the three wired call chains
# between them read all three: team_form and qb_tracking read pbp, qb_tracking reads
# depth_charts, ingest_games reads schedules and pbp.
LIVE_DATASETS: tuple[str, ...] = ("pbp", "schedules", "depth_charts")


def _capture_frame(week: int, sequence: int) -> pd.DataFrame:
    """The frame stored at one live capture address.

    The same play-by-play shape stands in for all three datasets on purpose: what these
    tests are about is which CAPTURE a read resolves to, not what each dataset's columns
    look like. Giving each dataset its own schema here would add nothing an assertion
    could read and would hide the marker behind three different column names.
    """
    return _pbp_frame(LIVE_ZONE_FIRST_SEASON, epa=CAPTURE_MARKERS[(week, sequence)])


class _TwoZones(NamedTuple):
    """A sealed pin and a live manifest, both inside ``tmp_path``."""

    manifest_path: Path
    data_root: Path
    live_dir: Path

    def loader_kwargs(self) -> dict[str, Path]:
        """The three REDIRECT keywords, so a test never touches the committed records.

        Deliberately only the redirects: the SEASON argument stays written out at every
        call site below, because matching the call shapes the wired modules actually use
        is the whole point of those tests.
        """
        return {
            "manifest_path": self.manifest_path,
            "data_root": self.data_root,
            "live_manifest_dir": self.live_dir,
        }


def _write_live_manifest(
    tmp_path: Path,
    data_root: Path,
    *,
    season: int = LIVE_ZONE_FIRST_SEASON,
) -> Path:
    """Write a live-zone manifest holding :data:`CAPTURE_ADDRESSES`; return its directory.

    The sibling of :func:`_write_pin` for the LIVE half. Two captures exist for week 6
    (sequences 1 and 2) and one for week 7, each holding a distinguishable frame, so every
    test below can say exactly which capture was read by looking at the value it got back.
    """
    (data_root / "bronze").mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict] = {}
    for dataset in LIVE_DATASETS:
        captures: list[dict] = []
        for week, sequence in CAPTURE_ADDRESSES:
            frame = _capture_frame(week, sequence)
            relative = (
                f"bronze/{dataset}_raw_bronze_{season}_W{week:02d}_S{sequence}"
                "_20260911T000000.parquet"
            )
            path = data_root / relative
            frame.to_parquet(path, index=False)
            captures.append(
                {
                    "week": week,
                    "sequence": sequence,
                    "captured_at_utc": f"2026-09-{10 + sequence:02d}T00:00:00+00:00",
                    "path": relative,
                    "sha256": digest_file(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                }
            )
        datasets[dataset] = {
            "loader": f"nflreadpy.load_{dataset}",
            "captures": captures,
        }

    manifest = {
        "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
        "season": season,
        "zone": "live",
        "source": "test fixture",
        "datasets": datasets,
    }
    live_dir = tmp_path / "upstream_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / f"{season}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return live_dir


@pytest.fixture
def two_zones(tmp_path: Path) -> _TwoZones:
    """Both zones, hermetically: a sealed 2024/2025 pbp pin and a live 2026 manifest.

    Nothing here reads ``config/``, ``data/`` or the network, so these tests pass on a
    fresh checkout exactly as the rest of the module does.
    """
    manifest_path, data_root = _write_pin(
        tmp_path,
        "pbp",
        {
            SEALED_THROUGH_SEASON - 1: _pbp_frame(SEALED_THROUGH_SEASON - 1),
            SEALED_THROUGH_SEASON: _pbp_frame(SEALED_THROUGH_SEASON),
        },
    )
    live_dir = _write_live_manifest(tmp_path, data_root)
    return _TwoZones(manifest_path, data_root, live_dir)


@pytest.fixture
def network_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any nflverse fetch an outright failure, and record attempts.

    ``upstream_pin._fetch_live`` is the ONE place the module can reach the network.
    Replacing it with a raising stub means a test that passes provably did not fetch.
    """
    attempts: list[str] = []

    def _explode(dataset: str, seasons: list[int]) -> pd.DataFrame:
        attempts.append(f"{dataset}:{seasons}")
        msg = "the network was reached, but this test asserts the pin was used instead"
        raise AssertionError(msg)

    monkeypatch.setattr(upstream_pin, "_fetch_live", _explode)
    return attempts


class TestAPresentPinIsRead:
    """With the pin present, the loader reads the file and never reaches nflverse."""

    def test_a_pinned_season_is_served_without_touching_the_network(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})

        frame = upstream_pin.load_pbp(
            [2024], manifest_path=manifest_path, data_root=data_root
        )

        assert list(frame["epa"]) == [0.25, -0.25]
        assert network_is_a_failure == [], (
            "the loader reached the network even though the season was pinned: "
            f"{network_is_a_failure}"
        )

    def test_multiple_pinned_seasons_concatenate_in_the_requested_order(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        manifest_path, data_root = _write_pin(
            tmp_path,
            "pbp",
            {2023: _pbp_frame(2023), 2024: _pbp_frame(2024)},
        )

        frame = upstream_pin.load_pbp(
            [2023, 2024], manifest_path=manifest_path, data_root=data_root
        )

        assert list(frame["season"]) == [2023, 2023, 2024, 2024]
        assert network_is_a_failure == []

    def test_the_returned_frame_is_not_a_view_a_caller_can_mutate_into_the_cache(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """``features/team_form.py`` mutates the frame it is handed (it rewrites
        ``posteam``/``defteam`` in place). Two loads must not see each other's edits."""
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})

        first = upstream_pin.load_pbp(
            [2024], manifest_path=manifest_path, data_root=data_root
        )
        first["posteam"] = "MUTATED"
        second = upstream_pin.load_pbp(
            [2024], manifest_path=manifest_path, data_root=data_root
        )

        assert list(second["posteam"]) == ["HOME", "AWAY"]


class TestThePinCanBeShownToFail:
    """A guard that has never been observed failing is a guard nobody has tested."""

    def test_a_poisoned_pin_changes_what_the_loader_returns(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """The strongest available proof that the VALUES come from the pinned file.

        The parquet is rewritten with an impossible EPA and the manifest digest is
        updated to match, so the poisoned bytes ARE legitimately the pin. If the loader
        were reaching upstream, or serving anything cached, the poison would not appear.
        """
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["datasets"]["pbp"]["seasons"]["2024"]
        pinned = data_root / entry["path"]

        poisoned = _pbp_frame(2024, epa=POISON_EPA)
        poisoned.to_parquet(pinned, index=False)
        entry["sha256"] = digest_file(pinned)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        frame = upstream_pin.load_pbp(
            [2024], manifest_path=manifest_path, data_root=data_root
        )

        assert list(frame["epa"]) == [POISON_EPA, -POISON_EPA], (
            "the poisoned pin did NOT reach the caller, so this loader is not actually "
            "reading the pinned file and the binding proof is vacuous"
        )
        assert network_is_a_failure == []

    def test_bytes_that_no_longer_match_the_recorded_digest_are_refused(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """Poisoning WITHOUT updating the manifest is tampering, and is refused."""
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned = data_root / manifest["datasets"]["pbp"]["seasons"]["2024"]["path"]
        _pbp_frame(2024, epa=POISON_EPA).to_parquet(pinned, index=False)

        with pytest.raises(UpstreamPinCorrupt) as error:
            upstream_pin.load_pbp(
                [2024], manifest_path=manifest_path, data_root=data_root
            )

        assert "does NOT match the digest recorded" in str(error.value)
        assert network_is_a_failure == [], (
            "a tampered pin fell through to a live fetch, which is the silent fallback "
            "this module exists to remove"
        )

    def test_a_pinned_file_that_is_absent_is_refused_by_name(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """A fresh checkout has the committed manifest but not the gitignored bytes."""
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (data_root / manifest["datasets"]["pbp"]["seasons"]["2024"]["path"]).unlink()

        with pytest.raises(UpstreamPinCorrupt) as error:
            upstream_pin.load_pbp(
                [2024], manifest_path=manifest_path, data_root=data_root
            )

        assert "pin_upstream_snapshot" in str(error.value)
        assert network_is_a_failure == []

    def test_a_manifest_from_a_future_schema_is_refused(self, tmp_path: Path) -> None:
        manifest_path, _ = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(UpstreamPinCorrupt):
            upstream_pin.load_manifest(manifest_path)


class TestAMissingPinRefusesRatherThanRefetching:
    """The silent fallback is the defect. Fail closed."""

    def test_no_manifest_at_all_refuses_and_does_not_fetch(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [2024],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        message = str(error.value)
        assert "nothing (no pin captured)" in message
        assert "pin_upstream_snapshot" in message, (
            "a refusal that does not say how to satisfy it is a refusal operators route "
            "around"
        )
        assert LIVE_OPT_IN_ENV in message
        assert network_is_a_failure == []

    def test_a_partially_covering_pin_names_the_missing_seasons(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        manifest_path, data_root = _write_pin(tmp_path, "pbp", {2024: _pbp_frame(2024)})

        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [2022, 2023, 2024], manifest_path=manifest_path, data_root=data_root
            )

        message = str(error.value)
        assert "2022, 2023" in message
        assert "It covers 2024-2024" in message
        assert network_is_a_failure == []

    def test_the_pin_error_is_outside_every_call_sites_except_clause(self) -> None:
        """The swallow hazard, held as a test rather than as a comment.

        ``features/qb_tracking.py`` catches ``(ImportError, ValueError, RuntimeError)``
        around both of its loaders and returns an EMPTY DataFrame. If ``UpstreamPinError``
        were any of those, a refusal would become an empty play-by-play frame and a
        silently degraded gold matrix -- strictly worse than the drift the pin removes.
        """
        for swallowed in (
            RuntimeError,
            ValueError,
            ImportError,
            KeyError,
            TypeError,
            ConnectionError,
            TimeoutError,
            OSError,
        ):
            assert not issubclass(UpstreamPinError, swallowed), (
                f"UpstreamPinError inherits {swallowed.__name__}, which the wired call "
                "sites catch and convert into an empty frame. A pin refusal MUST escape "
                "every existing handler."
            )


class TestAPinRefusalEscapesTheIngestGamesHandler:
    """WR-10: ``scripts/ingest_games`` caught bare ``Exception``, so the type discipline was moot.

    The class above proves ``UpstreamPinError`` inherits none of the types the wired handlers
    name. That is necessary and it is not sufficient: ``ingest_games.ingest_games`` wrapped its
    play-by-play fetch in ``except Exception``, which catches the refusal regardless of what it
    inherits. ``fetch_pbp_data`` catches only ``(ConnectionError, TimeoutError, ValueError)``, so
    the pin error arrived at that handler untouched.

    The consequence is the one ``upstream_pin``'s docstring names: after a season roll the
    manifest covers pbp through 2025 and nothing beyond, ``load_pbp`` raises its explicit refusal,
    ONE warning line is logged, and silver ``games`` is written with no ``home_score`` /
    ``away_score`` merged -- the ingest step reporting success while every downstream label built
    from those scores is wrong or absent.

    So the docstring's claim is ENFORCED here rather than asserted.
    """

    def test_a_pin_refusal_raised_inside_fetch_pbp_propagates_out_of_ingest_games(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.ingest_games import GameDataIngester

        ingester = GameDataIngester()
        monkeypatch.setattr(
            ingester,
            "fetch_schedule_data",
            lambda *a, **k: pd.DataFrame([{"game_id": "2026_W01_AAA@BBB"}]),
        )
        monkeypatch.setattr(
            ingester, "transform_schedule_data", lambda df: df.assign(season=2026)
        )

        def _refuse(*_args: object, **_kwargs: object):
            raise UpstreamPinMissing(
                "the pin covers pbp through 2025 and does not cover 2026"
            )

        monkeypatch.setattr(ingester, "fetch_pbp_data", _refuse)

        with pytest.raises(UpstreamPinError, match="does not cover 2026"):
            ingester.ingest_games(seasons=[2026], include_results=True)

    def test_an_ordinary_fetch_failure_still_degrades_rather_than_failing_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. The handler exists for a reason and must keep working.

        Without this the fix could have been "re-raise everything", which turns a transient
        results fetch into a failed ingest.
        """
        from scripts.ingest_games import GameDataIngester

        ingester = GameDataIngester()
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            ingester,
            "fetch_schedule_data",
            lambda *a, **k: pd.DataFrame([{"game_id": "2026_W01_AAA@BBB"}]),
        )
        monkeypatch.setattr(
            ingester, "transform_schedule_data", lambda df: df.assign(season=2026)
        )
        monkeypatch.setattr(
            ingester,
            "fetch_pbp_data",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("upstream is down")),
        )

        # Stop the run right after the handler, so this test asserts the handler's behaviour
        # and nothing about validation, bronze or silver.
        def _stop(df, schema):
            captured["reached_validation"] = True
            raise _StopAfterHandler

        monkeypatch.setattr("scripts.ingest_games.validate_bronze_to_silver", _stop)

        with pytest.raises(_StopAfterHandler):
            ingester.ingest_games(seasons=[2026], include_results=True)

        assert captured.get("reached_validation") is True, (
            "an ordinary ConnectionError from the results fetch now fails the whole ingest; "
            "the degrade-on-results-failure handler is meant to survive"
        )


class _StopAfterHandler(BaseException):
    """A sentinel that is NOT an Exception, so the handler under test cannot catch it."""


class TestLiveRefetchIsAnExplicitLoudOptIn:
    def test_the_opt_in_env_var_is_required_and_sufficient(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched: list[str] = []

        def _fake_fetch(dataset: str, seasons: list[int]) -> pd.DataFrame:
            fetched.append(f"{dataset}:{seasons}")
            return _pbp_frame(seasons[0])

        monkeypatch.setattr(upstream_pin, "_fetch_live", _fake_fetch)
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")

        with pytest.warns(UpstreamPinBypassedWarning) as recorded:
            frame = upstream_pin.load_pbp(
                [2024],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        assert fetched == ["pbp:[2024]"]
        assert len(frame) == 2
        assert "NOT reproducible" in str(recorded[0].message), (
            "the bypass warning must say what was given up, not merely that it happened"
        )

    def test_an_empty_env_var_is_not_an_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "   ")
        assert upstream_pin.live_upstream_allowed() is False

    def test_the_bypass_is_not_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bypass that produced no warning could be made routine without anyone noticing."""
        monkeypatch.setattr(
            upstream_pin, "_fetch_live", lambda dataset, seasons: _pbp_frame(2024)
        )
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "yes")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            upstream_pin.load_pbp(
                [2024],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        assert [w for w in caught if issubclass(w.category, UpstreamPinBypassedWarning)]


class TestAnEmptySeasonRequestIsRefusedByName:
    """WR-09. An empty request is a CALLER BUG, and the old answers said otherwise.

    ``if seasons and not missing`` deliberately excluded the empty list, which then fell
    through to the bypass block and produced one of two dishonest answers:

    * With the opt-in UNSET, ``UpstreamPinMissing`` reading "The upstream pin does not
      cover pbp season(s) ." -- a refusal naming no season, with an empty zone table and a
      recovery option list built from nothing.
    * With it SET, ``_fetch_live(dataset, [])`` reached ``frames[0]`` for ``depth_charts``
      and raised ``IndexError``.

    Returning an EMPTY FRAME would be worse than either: its emptiness reads as "upstream
    had nothing for these seasons", which is the silent-wrong-number failure this module
    exists to prevent.
    """

    @pytest.mark.parametrize("loader", ["load_pbp", "load_schedules"])
    def test_it_refuses_rather_than_naming_no_season(
        self, tmp_path: Path, loader: str
    ) -> None:
        with pytest.raises(UpstreamPinError) as error:
            getattr(upstream_pin, loader)(
                [],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        message = str(error.value)
        assert "NO seasons" in message, (
            f"the refusal does not name the empty request:\n{message}"
        )
        assert "season(s) ." not in message, (
            "the refusal still renders an empty season list as though a season had been "
            f"asked for:\n{message}"
        )

    def test_it_refuses_before_the_opt_in_can_reach_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even WITH the live opt-in set, an empty request never reaches nflverse.

        Under the opt-in the empty list used to fall into ``_fetch_live``, which is the
        only place this module can touch the network -- to answer a question nobody asked.
        """

        def _explode(dataset: str, seasons: list[int]):
            msg = "the empty request reached the live fetch"
            raise AssertionError(msg)

        monkeypatch.setattr(upstream_pin, "_fetch_live", _explode)
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")

        with pytest.raises(UpstreamPinError, match="NO seasons"):
            upstream_pin.load_pbp(
                [],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

    def test_it_is_not_an_upstream_pin_missing_coverage_question(
        self, tmp_path: Path
    ) -> None:
        """The TYPE carries the meaning: a caller bug is not a coverage gap.

        ``UpstreamPinMissing`` means "capture this season"; there is no season here to
        capture, and a recovery instruction that cannot be followed is worse than none.
        """
        with pytest.raises(UpstreamPinError) as error:
            upstream_pin.load_pbp(
                [],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        assert not isinstance(error.value, UpstreamPinMissing)


class TestTheBypassFetchCarriesTheSameColumnSetAsBothZones:
    """WR-05. The mixed-request branch must not concatenate ragged halves.

    ``_read_pinned_frame`` returns bytes narrowed at PIN time and ``_read_live_frame``
    bytes narrowed at CAPTURE time -- 23 columns each for ``pbp``. ``_fetch_live`` returned
    ``nfl.load_pbp(...)`` RAW, at roughly 372. ``_load``'s mixed branch ``pd.concat``s all
    three, so the result was a ragged union: the zone-served halves gained ~349 all-NaN
    columns and every extra column materialised for the live half alone.

    ``scripts/capture_live_season.py`` treats this as load-bearing rather than cosmetic:
    the mixed ``[2025, 2026]`` request "is exactly the shape ``features/team_form.py``
    issues", and that builder "branches on ``"cpoe" in group.columns``". So the one code
    path already declared non-reproducible was ALSO silently changing which branches the
    feature builders take -- a difference that produces plausible numbers and no error.
    """

    @staticmethod
    def _stub_nflreadpy(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
        """Stand in for ``nflreadpy`` so the REAL ``_fetch_live`` body runs offline."""

        class _Polars:
            def __init__(self, payload: pd.DataFrame) -> None:
                self._payload = payload

            def to_pandas(self) -> pd.DataFrame:
                return self._payload.copy()

        module = types.SimpleNamespace(
            load_pbp=lambda seasons: _Polars(frame),
            load_schedules=lambda seasons: _Polars(frame),
            load_depth_charts=lambda season: _Polars(frame),
        )
        monkeypatch.setitem(sys.modules, "nflreadpy", module)

    def test_the_live_pbp_fetch_is_narrowed_to_the_pinned_allowlist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wide = pd.DataFrame(
            {
                **{column: [1, 2] for column in PBP_PINNED_COLUMNS},
                # The columns the raw upstream frame carries and the pin does not.
                "xpass": [0.4, 0.6],
                "vegas_wp": [0.5, 0.5],
                "desc": ["a", "b"],
            }
        )
        self._stub_nflreadpy(monkeypatch, wide)

        fetched = upstream_pin._fetch_live("pbp", [2026])

        assert list(fetched.columns) == list(PBP_PINNED_COLUMNS), (
            "the bypass fetch returned a column set neither zone produces, so a mixed "
            f"request concatenates ragged halves. Extra: "
            f"{sorted(set(fetched.columns) - set(PBP_PINNED_COLUMNS))}"
        )

    def test_narrowing_preserves_absence_rather_than_inventing_a_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``narrow`` is IMPORTED, so its preserve-absence contract comes with it.

        Materialising an allowlisted-but-absent column as all-null would hand the builders
        a column the live path never gave them, and ``features/team_form.py`` branches on
        ``"cpoe" in group.columns``.
        """
        without_cpoe = pd.DataFrame(
            {column: [1, 2] for column in PBP_PINNED_COLUMNS if column != "cpoe"}
        )
        self._stub_nflreadpy(monkeypatch, without_cpoe)

        fetched = upstream_pin._fetch_live("pbp", [2026])

        assert "cpoe" not in fetched.columns, (
            "an absent allowlisted column was materialised, which changes which branch "
            "the feature builder takes"
        )

    def test_a_whole_pinned_dataset_is_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``schedules`` and ``depth_charts`` are pinned WHOLE, so narrow is a no-op.

        Pinned here as behaviour: a fix that narrowed them to some invented allowlist
        would be a second column judgement inside the reproducibility claim, which
        ``DATASET_COLUMNS`` deliberately refuses.
        """
        frame = pd.DataFrame({"game_id": ["g1"], "anything": [1], "else_": [2]})
        self._stub_nflreadpy(monkeypatch, frame)

        for dataset in ("schedules", "depth_charts"):
            assert upstream_pin.DATASET_COLUMNS[dataset] is None
            fetched = upstream_pin._fetch_live(dataset, [2026])
            assert list(fetched.columns) == list(frame.columns), dataset

    def test_a_mixed_zone_request_produces_one_column_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end-to-end shape: no NaN-padded union at the season boundary."""
        narrowed = pd.DataFrame({column: [1, 2] for column in PBP_PINNED_COLUMNS})
        wide = pd.DataFrame(
            {
                **{column: [3, 4] for column in PBP_PINNED_COLUMNS},
                "xpass": [0.4, 0.6],
                "desc": ["a", "b"],
            }
        )
        self._stub_nflreadpy(monkeypatch, wide)
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")
        monkeypatch.setattr(
            upstream_pin,
            "_read_pinned_frame",
            lambda dataset, season, manifest, root: narrowed.copy(),
        )
        monkeypatch.setattr(
            upstream_pin,
            "_partition_seasons",
            lambda dataset, seasons, manifest, manifest_dir=None: {
                "sealed": [2025],
                "live": [],
                "unknown": [2026],
            },
        )

        with pytest.warns(UpstreamPinBypassedWarning):
            combined = upstream_pin.load_pbp(
                [2025, 2026],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
            )

        assert list(combined.columns) == list(PBP_PINNED_COLUMNS), (
            "the mixed frame is a ragged union of two different column sets: "
            f"{sorted(set(combined.columns) - set(PBP_PINNED_COLUMNS))}"
        )
        assert len(combined) == 4
        assert not combined.isna().any().any(), (
            "the concat NaN-padded one half against the other's extra columns"
        )


class TestTheGoldRebuildCallSitesReadThePin:
    """Wiring the pin into a module nobody calls would prove nothing.

    The scan is on the AST, not on the source text: a comment mentioning
    ``nfl.load_pbp`` must not fail the check, and a live call hidden behind a rename must
    not pass it.
    """

    LIVE_LOADERS = {"load_pbp", "load_schedules", "load_depth_charts"}

    WIRED_MODULES = (
        "features/team_form.py",
        "features/qb_tracking.py",
        "scripts/ingest_games.py",
        "scripts/ingest_historical_odds.py",
    )

    @pytest.mark.parametrize("relative", WIRED_MODULES)
    def test_the_module_makes_no_direct_nflreadpy_load_call(
        self, relative: str
    ) -> None:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))

        offenders = [
            f"line {node.lineno}: {ast.unparse(node.func)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in self.LIVE_LOADERS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id != "upstream_pin"
        ]

        assert offenders == [], (
            f"{relative} still calls nflverse directly at {offenders}. A gold rebuild "
            "that reaches upstream live cannot be reproduced from a recorded snapshot, "
            "which is the defect the pin exists to close."
        )

    @pytest.mark.parametrize("relative", WIRED_MODULES)
    def test_the_module_imports_the_pin(self, relative: str) -> None:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "upstream_pin" in source, (
            f"{relative} does not reference data.upstream_pin, so the previous test "
            "passes vacuously -- a module with no loader call at all would satisfy it."
        )


class TestTheCommittedManifestIsTheProvenanceRecord:
    """``data/`` is gitignored, so the record has to live somewhere tracked."""

    @staticmethod
    def _manifest() -> dict:
        path = REPO_ROOT / MANIFEST_PATH
        if not path.is_file():
            pytest.skip(
                f"the upstream pin manifest is not present at {path} -- no pin has been "
                "captured on this checkout."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_records_the_source_and_the_capture_identity(self) -> None:
        manifest = self._manifest()
        for field in (
            "source",
            "captured_at_utc",
            "nflreadpy_version",
            "pandas_version",
            "pyarrow_version",
            "python_version",
        ):
            assert manifest.get(field), (
                f"the manifest records no {field!r}. A snapshot whose producing "
                "environment is unrecorded cannot be re-derived."
            )

    def test_every_pinned_season_carries_a_digest_and_a_row_count(self) -> None:
        manifest = self._manifest()
        assert manifest["datasets"], "the manifest pins no dataset at all"
        for dataset, record in manifest["datasets"].items():
            assert record["seasons"], f"{dataset} pins no season"
            for season, entry in record["seasons"].items():
                assert len(entry["sha256"]) == 64, f"{dataset} {season}: no sha256"
                assert entry["rows"] > 0, f"{dataset} {season}: zero rows pinned"
                assert entry["columns"], f"{dataset} {season}: no column list"
                assert entry["path"].startswith("bronze/"), (
                    f"{dataset} {season}: pinned outside the bronze snapshot layer at "
                    f"{entry['path']}"
                )

    def test_the_play_by_play_pin_carries_every_column_the_builders_read(self) -> None:
        manifest = self._manifest()
        pbp = manifest["datasets"].get("pbp")
        if pbp is None:
            pytest.skip("play-by-play is not pinned on this checkout.")
        for season, entry in pbp["seasons"].items():
            missing = sorted(set(PBP_PINNED_COLUMNS) - set(entry["columns"]))
            assert missing == [], (
                f"pbp {season} is pinned without {missing}, which "
                "data.upstream_pin.PBP_PINNED_COLUMNS says a builder reads. The build "
                "would raise, or silently take a different branch."
            )

    def test_it_names_what_was_deliberately_left_unpinned(self) -> None:
        manifest = self._manifest()
        assert manifest.get("not_pinned"), (
            "the manifest claims no exclusions. The pin does NOT cover every nflverse "
            "reader in the repository, and a provenance record that does not say where "
            "its own boundary is overstates the reproducibility claim."
        )
        for excluded in manifest["not_pinned"]:
            assert excluded.get("reason"), (
                f"{excluded.get('loader')} has no reason given"
            )


class TestTheZoneBoundaryIsAFixedLiteral:
    """Phase 32 (PIN-01): the SEALED boundary is a literal; the live zone follows it.

    Step 27b: the third "unknown" zone for every season after the first live one is gone --
    August 2027 needed a human edit before the daily run could capture its schedule. When a
    later season may first be WRITTEN is the live capture's schedule rule, pinned in
    ``tests/unit/test_daily_season_rollover.py``.
    """

    def test_every_season_at_or_before_the_boundary_is_sealed(self) -> None:
        for season in (1999, 2002, 2020, SEALED_THROUGH_SEASON):
            assert zone_for_season(season) == ZONE_SEALED

    def test_the_live_zone_starts_right_after_the_sealed_zone(self) -> None:
        assert zone_for_season(LIVE_ZONE_FIRST_SEASON) == ZONE_LIVE
        assert LIVE_ZONE_FIRST_SEASON == SEALED_THROUGH_SEASON + 1

    def test_every_later_season_is_live_and_never_sealed(self) -> None:
        """Was: 2027 belonged to no zone (step 27b). Intent kept: nothing after the
        boundary is ever SEALED by a load or a capture -- sealing is still the human edit
        pinned below -- and a later season is written only under the live-zone rules."""
        for season in (LIVE_ZONE_FIRST_SEASON + 1, LIVE_ZONE_FIRST_SEASON + 5):
            assert zone_for_season(season) == ZONE_LIVE

    def test_the_boundary_is_a_literal_and_not_a_computed_current_season(self) -> None:
        """``get_current_season()`` flips on the Thursday after Labor Day.

        A computed boundary would move the sealed zone silently and mid-week. This test
        is the reason the constant is an ``int`` and not a call.
        """
        tree = ast.parse(
            (REPO_ROOT / "data" / "upstream_pin.py").read_text(encoding="utf-8")
        )
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SEALED_THROUGH_SEASON"
        ]
        assert len(assignments) == 1, (
            "SEALED_THROUGH_SEASON is not assigned exactly once"
        )
        value = assignments[0].value
        assert isinstance(value, ast.Constant) and value.value == 2025, (
            "SEALED_THROUGH_SEASON is not a plain integer literal. A computed boundary "
            "moves the sealed zone silently and mid-week; only a human edit may move it."
        )


class TestEveryNewRefusalEscapesTheWiredHandlers:
    """The swallow hazard applies to the zone exceptions exactly as it does to the pin."""

    NEW_REFUSALS = (
        UpstreamLiveCaptureMissing,
        UpstreamSeasonWindowRefused,
        ZoneWriteRefused,
        UpstreamLiveCorrupt,
    )

    @pytest.mark.parametrize("refusal", NEW_REFUSALS)
    def test_it_is_an_upstream_pin_error(self, refusal: type[Exception]) -> None:
        assert issubclass(refusal, UpstreamPinError)

    def test_the_corrupt_capture_is_a_corrupt_pin(self) -> None:
        """So every handler that already treats a corrupt pin as fatal covers it."""
        assert issubclass(UpstreamLiveCorrupt, UpstreamPinCorrupt)

    @pytest.mark.parametrize("refusal", NEW_REFUSALS)
    def test_it_is_outside_every_call_sites_except_clause(
        self, refusal: type[Exception]
    ) -> None:
        for swallowed in (
            RuntimeError,
            ValueError,
            ImportError,
            KeyError,
            TypeError,
            ConnectionError,
            TimeoutError,
            OSError,
        ):
            assert not issubclass(refusal, swallowed), (
                f"{refusal.__name__} inherits {swallowed.__name__}, which the wired "
                "call sites catch and convert into an empty frame."
            )


class TestTheRefusalNamesTheZoneAndItsTool:
    """A refusal that names the wrong capture tool is a refusal operators route around."""

    def test_a_live_zone_season_is_pointed_at_the_live_capture(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        manifest_path, data_root = _write_pin(
            tmp_path, "pbp", {SEALED_THROUGH_SEASON: _pbp_frame(SEALED_THROUGH_SEASON)}
        )

        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [SEALED_THROUGH_SEASON, LIVE_ZONE_FIRST_SEASON],
                manifest_path=manifest_path,
                data_root=data_root,
                live_manifest_dir=tmp_path / "upstream_live",
            )

        message = str(error.value)
        assert f"{LIVE_ZONE_FIRST_SEASON}  zone live" in message
        assert "scripts.capture_live_season" in message
        assert network_is_a_failure == []

    def test_a_sealed_zone_season_still_names_the_sealed_tool(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """The pre-existing message contract, unchanged for the sealed zone."""
        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [2024],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            )

        message = str(error.value)
        assert "2024  zone sealed" in message
        assert "scripts.pin_upstream_snapshot" in message
        assert "capture_live_season" not in message
        assert network_is_a_failure == []

    def test_an_uncaptured_later_season_is_refused_and_names_the_live_tool(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        """Was: "told no tool captures it" (step 27b). Intent kept: a later season nobody
        captured is REFUSED by name with no network fetch; it now names the live capture,
        which is the tool that opens it once the season before it has ended."""
        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON + 1],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            )

        message = str(error.value)
        assert f"{LIVE_ZONE_FIRST_SEASON + 1}  zone live" in message
        assert "scripts.capture_live_season" in message
        assert network_is_a_failure == []


class TestThePinnedFilesOnDiskMatchTheCommittedManifest:
    """The committed record and the gitignored bytes still agree on this checkout."""

    def test_every_recorded_digest_matches(self) -> None:
        manifest_path = REPO_ROOT / MANIFEST_PATH
        if not manifest_path.is_file():
            pytest.skip(
                f"the upstream pin manifest is not present at {manifest_path} -- no pin "
                "has been captured on this checkout."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # WR-11. RESOLVED THE WAY PRODUCTION RESOLVES IT, not hardcoded to REPO_ROOT/"data".
        # ``default_data_root`` reads ``settings.config.data.root_path``, which
        # ``DATA_ROOT_PATH`` redirects -- ``data.upstream_pin``'s own docstring says so at
        # its definition. On any checkout or CI job with a redirected root, the hardcoded
        # path made this control -- the ONLY one that checks the committed manifest against
        # the bytes actually IN USE -- skip with a reason that reads like a
        # gitignored-evidence skip rather than the mis-scoped test it was. A control going
        # quiet is the exact property this phase asserts everywhere else.
        data_root = upstream_pin.default_data_root()
        first_entry = next(
            iter(next(iter(manifest["datasets"].values()))["seasons"].values())
        )
        if not (data_root / first_entry["path"]).is_file():
            pytest.skip(
                "the pinned bronze snapshots are gitignored and are absent on this "
                f"checkout (looked under the configured data root {data_root})."
            )

        mismatches: list[str] = []
        for dataset, record in sorted(manifest["datasets"].items()):
            for season, entry in sorted(record["seasons"].items()):
                path = data_root / entry["path"]
                if not path.is_file():
                    mismatches.append(f"{dataset} {season}: MISSING {path}")
                elif digest_file(path) != entry["sha256"]:
                    mismatches.append(f"{dataset} {season}: DIGEST MOVED {path}")

        assert mismatches == [], (
            "the pinned snapshots on disk no longer match the committed manifest:\n  "
            + "\n  ".join(mismatches)
        )


class TestTheAsOfContextReachesEveryWiredCallSite:
    """D32-15, in both directions: the setting reaches all three chains, unthreaded.

    ``features/team_form.py``, ``features/qb_tracking.py`` and ``scripts/ingest_games.py``
    each call the pin independently, at different depths inside their own call chains. A
    capture-selection keyword threaded through those three builders -- and through
    everything that calls them -- has one path that gets missed, and that path silently
    reads today's newest bytes and returns a number that looks entirely normal: no
    traceback, no empty frame, nothing to notice. That is why D32-15 is a process-level
    context and why 32-CONTEXT.md rates it costly.

    Each test below sets the as-of once and reads back through the exact call shape its
    module uses TODAY. A shape that stopped reaching the context would return the week-7
    marker (7.1) instead of the week-6-sequence-1 marker (6.1), which is precisely the
    "normal-looking wrong number" this class exists to make impossible to miss.
    """

    WIRED_MODULES = (
        "features/team_form.py",
        "features/qb_tracking.py",
        "scripts/ingest_games.py",
    )

    PINNED_LOADERS = {"load_pbp", "load_schedules", "load_depth_charts"}

    @staticmethod
    def _live_epa(frame: pd.DataFrame) -> list[float]:
        """The EPA markers of the LIVE-zone rows in *frame*, in frame order."""
        return list(frame.loc[frame["season"] == LIVE_ZONE_FIRST_SEASON, "epa"])

    def test_team_form_reads_the_active_as_of(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """``features/team_form.py:123``, with the ``[season - 1, season]`` request.

        That request shape is built at ``features/team_form.py:654`` and ``:712`` and is
        the one PIN-02 protects: it spans both zones at once, so a call site that reached
        the as-of for neither half and a call site that reached it for the sealed half
        would both be wrong here, and differently.
        """
        with as_of_capture(week=6, sequence=1):
            frame = upstream_pin.load_pbp(
                [SEALED_THROUGH_SEASON, LIVE_ZONE_FIRST_SEASON],
                **two_zones.loader_kwargs(),
            )

        assert self._live_epa(frame) == [6.1, -6.1], (
            "features/team_form.py's [season - 1, season] play-by-play request did NOT "
            "reach the process-level as-of: it read a capture other than week 6 sequence "
            "1. A missed path here returns a normal-looking number computed from bytes "
            "nobody asked for."
        )
        assert network_is_a_failure == []

    def test_qb_tracking_reads_the_active_as_of(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """``features/qb_tracking.py:779-783`` (pbp) and ``:715`` (depth charts).

        Two shapes, both unmodified. The per-season ``load_pbp([s])`` loop dodges PIN-02's
        bug by call shape rather than by design, so it is exactly the kind of site a
        threaded keyword would have been added to last, or not at all.
        """
        with as_of_capture(week=6, sequence=1):
            pbp = upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs()
            )
            depth_charts = upstream_pin.load_depth_charts(
                LIVE_ZONE_FIRST_SEASON, **two_zones.loader_kwargs()
            )

        assert self._live_epa(pbp) == [6.1, -6.1], (
            "features/qb_tracking.py's per-season load_pbp([s]) shape did not reach the "
            "as-of"
        )
        assert self._live_epa(depth_charts) == [6.1, -6.1], (
            "features/qb_tracking.py's load_depth_charts(season) shape did not reach the "
            "as-of -- the depth-chart read is a separate call chain from the pbp one and "
            "has to be proved separately"
        )
        assert network_is_a_failure == []

    def test_ingest_games_reads_the_active_as_of(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """``scripts/ingest_games.py:134`` (schedules) and ``:175`` (play-by-play)."""
        with as_of_capture(week=6, sequence=1):
            schedules = upstream_pin.load_schedules(
                [LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs()
            )
            pbp = upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs()
            )

        assert self._live_epa(schedules) == [6.1, -6.1], (
            "scripts/ingest_games.py's load_schedules shape did not reach the as-of"
        )
        assert self._live_epa(pbp) == [6.1, -6.1], (
            "scripts/ingest_games.py's load_pbp shape did not reach the as-of"
        )
        assert network_is_a_failure == []

    def test_the_same_shapes_read_the_newest_capture_when_no_as_of_is_set(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """The control, without which the three tests above could pass vacuously.

        If every read returned the week-6-sequence-1 frame regardless -- because the
        fixture only really held one capture, say -- the positive tests would pass while
        proving nothing. Unset, the same three shapes must return the week-7 marker.
        """
        kwargs = two_zones.loader_kwargs()
        assert current_as_of() is None, "an as-of leaked in from an earlier test"

        team_form = upstream_pin.load_pbp(
            [SEALED_THROUGH_SEASON, LIVE_ZONE_FIRST_SEASON], **kwargs
        )
        depth_charts = upstream_pin.load_depth_charts(LIVE_ZONE_FIRST_SEASON, **kwargs)
        schedules = upstream_pin.load_schedules([LIVE_ZONE_FIRST_SEASON], **kwargs)

        for name, frame in (
            ("team_form pbp", team_form),
            ("qb_tracking depth_charts", depth_charts),
            ("ingest_games schedules", schedules),
        ):
            assert self._live_epa(frame) == [7.1, -7.1], (
                f"{name} did not read the NEWEST capture with no as-of set, so the "
                "fixture cannot distinguish 'reached the as-of' from 'always returns the "
                "same frame' and the positive proofs above are vacuous"
            )
        assert network_is_a_failure == []

    @pytest.mark.parametrize("relative", WIRED_MODULES)
    def test_the_wired_modules_pass_no_as_of_argument(self, relative: str) -> None:
        """The STRUCTURAL proof, on the AST rather than on the source text.

        The three tests above prove the context reaches each call chain. This one proves
        it is the CONTEXT doing that work and not a keyword somebody quietly threaded
        through a builder -- the design D32-15 rejected, and the one a later change would
        drift back toward, because threading a keyword always looks like the smaller diff
        at the moment it is written.
        """
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))

        threaded = [
            f"line {node.lineno}: {ast.unparse(node.func)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in self.PINNED_LOADERS
            and any(keyword.arg == "as_of" for keyword in node.keywords)
        ]

        assert threaded == [], (
            f"{relative} threads an as_of keyword into a pinned loader at {threaded}. "
            "Capture selection is a PROCESS-LEVEL context precisely so these three "
            "modules do not have to carry it: a keyword threaded through three builders "
            "and everything that calls them has one path that gets missed, and that path "
            "silently reads today's newest bytes (D32-15)."
        )


class TestTheAsOfPrecedenceAndItsFailureModes:
    """One precedence order, and every way of getting it wrong is loud."""

    def test_the_per_call_as_of_keyword_beats_the_context(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        with as_of_capture(week=7):
            frame = upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON],
                as_of=AsOfCapture(6, 1),
                **two_zones.loader_kwargs(),
            )

        assert list(frame["epa"]) == [6.1, -6.1], (
            "the per-call as_of keyword did not override the enclosing context, so the "
            "one caller that needs a different capture from the surrounding process "
            "cannot ask for one"
        )
        assert network_is_a_failure == []

    def test_the_context_beats_the_environment(
        self,
        two_zones: _TwoZones,
        network_is_a_failure: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit ``with`` block is more specific than a process-wide setting."""
        monkeypatch.setenv(AS_OF_ENV, "7")

        with as_of_capture(week=6, sequence=1):
            frame = upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs()
            )

        assert list(frame["epa"]) == [6.1, -6.1], (
            f"{AS_OF_ENV} overrode an enclosing as_of_capture block. A rebuild that set "
            "the variable once would then silently ignore every deliberate replay inside "
            "it."
        )
        assert network_is_a_failure == []

    def test_no_as_of_reads_the_newest_capture(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """D32-14's default read, unchanged by the existence of an as-of mechanism."""
        frame = upstream_pin.load_pbp(
            [LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs()
        )

        assert list(frame["epa"]) == [7.1, -7.1], (
            "with nothing set, the read did not serve the newest capture"
        )
        assert network_is_a_failure == []

    def test_a_sequence_addresses_one_exact_capture(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """Week 6 was captured twice; both remain addressable forever (D32-14)."""
        kwargs = two_zones.loader_kwargs()

        first = upstream_pin.load_pbp(
            [LIVE_ZONE_FIRST_SEASON], as_of=AsOfCapture(6, 1), **kwargs
        )
        second = upstream_pin.load_pbp(
            [LIVE_ZONE_FIRST_SEASON], as_of=AsOfCapture(6, 2), **kwargs
        )
        newest_of_the_week = upstream_pin.load_pbp(
            [LIVE_ZONE_FIRST_SEASON], as_of=AsOfCapture(6), **kwargs
        )

        assert list(first["epa"]) == [6.1, -6.1]
        assert list(second["epa"]) == [6.2, -6.2], (
            "sequence 2 returned the same frame as sequence 1, so a re-capture is not "
            "separately addressable and a Phase-34 replay address means nothing"
        )
        assert list(newest_of_the_week["epa"]) == [6.2, -6.2], (
            "a week with no sequence did not resolve to the NEWEST capture of that week; "
            "an ordinary read must see the Saturday re-run, not the superseded Friday one"
        )
        assert network_is_a_failure == []

    def test_a_replay_of_a_week_with_no_capture_lists_the_captures_that_exist(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """D32-16. A missing week is NEVER served from a different one."""
        with (
            pytest.raises(UpstreamLiveCaptureMissing) as error,
            as_of_capture(week=12),
        ):
            upstream_pin.load_pbp([LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs())

        message = str(error.value)
        assert "week 6" in message and "week 7" in message, (
            "the refusal does not list the weeks that DO have captures, so an operator "
            f"cannot tell what IS replayable:\n{message}"
        )
        assert "(6, 1), (6, 2), (7, 1)" in message, (
            "the refusal does not print the addressable (week, sequence) pairs in the "
            "form the operator would retype"
        )
        assert isinstance(error.value, UpstreamPinError), (
            "the replay refusal is outside the pin's exception family"
        )
        for swallowed in (ValueError, RuntimeError, ImportError):
            assert not isinstance(error.value, swallowed), (
                f"the replay refusal is a {swallowed.__name__}, which the wired call "
                "sites catch and convert into an empty frame"
            )
        assert network_is_a_failure == []

    @pytest.mark.parametrize("raw", ("six", "x", "-1", "6:2:3", "6:x"))
    def test_a_malformed_environment_as_of_raises_rather_than_reading_the_newest(
        self,
        raw: str,
        two_zones: _TwoZones,
        network_is_a_failure: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """T-32-29. This silent fallback would be invisible, which is what makes it bad.

        A malformed as-of that fell back to the newest capture would finish the run,
        report success, and return a number that looks entirely normal -- computed from a
        capture nobody asked for. There is no traceback to read and no empty frame to
        notice, so the refusal is the only thing standing between a typo and a wrong
        number in a published prediction.
        """
        monkeypatch.setenv(AS_OF_ENV, raw)

        with pytest.raises(UpstreamPinError) as error:
            upstream_pin.load_pbp([LIVE_ZONE_FIRST_SEASON], **two_zones.loader_kwargs())

        message = str(error.value)
        assert AS_OF_ENV in message, (
            "the refusal does not name the variable that caused it"
        )
        assert repr(raw) in message, "the refusal does not quote the value it rejected"
        assert network_is_a_failure == []

    def test_the_context_does_not_survive_an_exception_in_its_body(
        self, network_is_a_failure: list[str]
    ) -> None:
        """T-32-31. An as-of that outlived its block would re-address an unrelated load."""
        assert current_as_of() is None

        with pytest.raises(_BoomInsideTheBlock):
            with as_of_capture(week=6, sequence=1):
                assert current_as_of() == AsOfCapture(6, 1)
                raise _BoomInsideTheBlock

        assert current_as_of() is None, (
            "the as-of survived an exception raised inside its block, so every later "
            "load in this process would silently read week 6 sequence 1"
        )

    @pytest.mark.parametrize("week", (6, 7))
    def test_an_as_of_never_changes_which_bytes_a_sealed_season_reads(
        self, week: int, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """T-32-32. A sealed season has exactly one pinned file per dataset.

        Parametrized over two DIFFERENT as-ofs on purpose: one would prove only that the
        sealed read survived an as-of, not that it is indifferent to which one.
        """
        kwargs = two_zones.loader_kwargs()
        baseline = upstream_pin.load_pbp([SEALED_THROUGH_SEASON - 1], **kwargs)

        with as_of_capture(week=week):
            under_as_of = upstream_pin.load_pbp([SEALED_THROUGH_SEASON - 1], **kwargs)

        pd.testing.assert_frame_equal(
            under_as_of,
            baseline,
            obj=(
                f"the sealed {SEALED_THROUGH_SEASON - 1} frame read under "
                f"as_of_capture(week={week}) differs from the frame read with no as-of. "
                "An as-of selects among LIVE captures; a sealed season has exactly one "
                "pinned file and nothing for an as-of to choose between."
            ),
        )
        assert network_is_a_failure == []

    def test_a_mixed_request_applies_the_as_of_only_to_the_live_half(
        self, two_zones: _TwoZones, network_is_a_failure: list[str]
    ) -> None:
        """The two zones in one request, in the ORIGINALLY REQUESTED season order."""
        with as_of_capture(week=6, sequence=1):
            frame = upstream_pin.load_pbp(
                [SEALED_THROUGH_SEASON, LIVE_ZONE_FIRST_SEASON],
                **two_zones.loader_kwargs(),
            )

        assert list(frame["season"]) == [
            SEALED_THROUGH_SEASON,
            SEALED_THROUGH_SEASON,
            LIVE_ZONE_FIRST_SEASON,
            LIVE_ZONE_FIRST_SEASON,
        ], "the mixed request did not preserve the requested season order"
        assert list(frame["epa"]) == [0.25, -0.25, 6.1, -6.1], (
            "the as-of did not land on the live half alone: the sealed rows must carry "
            "their pinned values and the live rows the addressed capture's marker"
        )
        assert network_is_a_failure == []


class _BoomInsideTheBlock(RuntimeError):
    """Raised inside an ``as_of_capture`` block to prove the token is reset anyway."""
