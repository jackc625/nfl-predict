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
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from data import upstream_pin
from data.upstream_pin import (
    LIVE_OPT_IN_ENV,
    LIVE_ZONE_FIRST_SEASON,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    PBP_PINNED_COLUMNS,
    SEALED_THROUGH_SEASON,
    ZONE_LIVE,
    ZONE_SEALED,
    ZONE_UNKNOWN,
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
    """Phase 32 (PIN-01): three zones, and the third is not a tidy-up case."""

    def test_every_season_at_or_before_the_boundary_is_sealed(self) -> None:
        for season in (1999, 2002, 2020, SEALED_THROUGH_SEASON):
            assert zone_for_season(season) == ZONE_SEALED

    def test_exactly_one_season_is_live(self) -> None:
        assert zone_for_season(LIVE_ZONE_FIRST_SEASON) == ZONE_LIVE
        assert LIVE_ZONE_FIRST_SEASON == SEALED_THROUGH_SEASON + 1

    def test_beyond_the_live_zone_belongs_to_no_zone(self) -> None:
        """2027 must REFUSE rather than be silently admitted to the live zone.

        The one-way promotion that moves the boundary forward cannot be exercised until
        the live season actually ends. A season that quietly became "live" would be
        captured under semantics nobody ratified.
        """
        for season in (LIVE_ZONE_FIRST_SEASON + 1, LIVE_ZONE_FIRST_SEASON + 5):
            assert zone_for_season(season) == ZONE_UNKNOWN

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

    def test_a_season_beyond_the_live_zone_is_told_no_tool_captures_it(
        self, tmp_path: Path, network_is_a_failure: list[str]
    ) -> None:
        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [LIVE_ZONE_FIRST_SEASON + 1],
                manifest_path=tmp_path / "absent.json",
                data_root=tmp_path / "data",
                live_manifest_dir=tmp_path / "upstream_live",
            )

        message = str(error.value)
        assert f"{LIVE_ZONE_FIRST_SEASON + 1}  zone unknown" in message
        assert "one-way" in message
        assert "SEALED_THROUGH_SEASON" in message
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
        data_root = REPO_ROOT / "data"
        first_entry = next(
            iter(next(iter(manifest["datasets"].values()))["seasons"].values())
        )
        if not (data_root / first_entry["path"]).is_file():
            pytest.skip(
                "the pinned bronze snapshots are gitignored and are absent on this "
                "checkout."
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
