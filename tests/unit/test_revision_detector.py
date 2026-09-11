"""The sealed-zone probe rules honestly, and it can be shown to fail.

TEST CLASS: plain unit tests. Every GitHub payload these tests read comes from
``tests/fixtures/github_releases/``, captured once from the live API and committed (see
that directory's README). **This module makes NO network call**, and neither does any
other test in Phase 32. The GitHub API rate limit is 60 requests per hour per IP; a suite
that probed live would exhaust it and would also make its own result depend on what
nflverse published that morning.

A detector nobody can show failing is decoration. Six things are proved here:

1. The probe watches the file the LOADER reads -- the resolved asset URL is compared
   against the one ``nflreadpy`` itself builds, with the base read from
   ``nflreadpy.downloader`` rather than re-typed.
2. The release payload is UNTRUSTED input. Every malformed shape is refused by name.
3. A real sealed re-release is reported as CRITICAL, naming the dataset and the season.
4. A probe that went BLIND says so. An absent asset and an unseeded baseline both produce
   UNKNOWN, never clean -- the F2 test: a dead detector and a healthy system must not be
   the same observable.
5. A transport failure is raised as ``SealedProbeUnavailable``, outside every broad
   ``except`` in this repository, rather than swallowed into "upstream is fine".
6. The ``schedules`` strategy is CONTENT, not metadata. The 27-of-27 false-positive
   measurement is pinned as a regression.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import requests

from data.revision_events import RevisionEventClass, RevisionSeverity
from data.sealed_probe import (
    PROBE_STRATEGY_CONTENT,
    PROBE_STRATEGY_METADATA,
    SEALED_PROBE_STRATEGY,
    VERDICT_KEYS,
    SealedProbeUnavailable,
    asset_download_url,
    asset_name_for,
    fetch_release_assets,
    parse_release_payload,
    probe_sealed,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "github_releases"
TAGS = ("pbp", "schedules", "depth_charts")

# The loader path each ``nflreadpy`` call site supplies to its own downloader, cited to
# source so a reader can check them: ``load_pbp.py:45``, ``load_depth_charts.py:47``,
# ``load_schedules.py:27``. Only the PATH is transcribed; the BASE URL is read from
# ``nflreadpy.downloader`` below so a change upstream fails this test rather than passing
# silently.
LOADER_PATHS = {
    "pbp": "pbp/play_by_play_{season}",
    "depth_charts": "depth_charts/depth_charts_{season}",
    "schedules": "schedules/games",
}


def _stored_payload(tag: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{tag}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stored_assets() -> dict[str, dict[str, dict]]:
    """``{tag: {asset name: signature}}`` from the committed payloads. No network."""
    return {tag: parse_release_payload(_stored_payload(tag)) for tag in TAGS}


def _metadata_entry(assets: dict[str, dict], dataset: str, season: int) -> dict:
    """A sealed-lock entry whose baseline IS today's stored upstream signature."""
    observed = assets[asset_name_for(dataset, season)]
    return {
        "sha256": f"pinned-local-digest-{dataset}-{season}",
        "rows": 1,
        "bytes": 1,
        "upstream_updated_at": observed["updated_at"],
        "upstream_size": observed["size"],
        "acknowledgement": None,
    }


def _content_entry(dataset: str, season: int) -> dict:
    """A sealed-lock entry for a CONTENT-strategy pair.

    ``sha256`` is the pinned per-season digest this project wrote -- NOT the upstream
    asset's bytes. ``probe_sealed``'s ``content_digests`` argument is documented to be on
    that same basis, and these tests honour it.
    """
    return {
        "sha256": f"pinned-local-digest-{dataset}-{season}",
        "rows": 1,
        "bytes": 1,
        "upstream_updated_at": None,
        "upstream_size": None,
        "acknowledgement": None,
    }


def build_lock(
    stored_assets: dict[str, dict[str, dict]],
    *,
    pbp_seasons: tuple[int, ...] = (2021, 2022),
    depth_seasons: tuple[int, ...] = (2021,),
    schedules_seasons: tuple[int, ...] = (2021,),
) -> dict:
    """A small sealed lock whose metadata baselines already match the stored payloads."""
    datasets: dict[str, dict[str, dict]] = {}
    if pbp_seasons:
        datasets["pbp"] = {
            str(season): _metadata_entry(stored_assets["pbp"], "pbp", season)
            for season in pbp_seasons
        }
    if depth_seasons:
        datasets["depth_charts"] = {
            str(season): _metadata_entry(
                stored_assets["depth_charts"], "depth_charts", season
            )
            for season in depth_seasons
        }
    if schedules_seasons:
        datasets["schedules"] = {
            str(season): _content_entry("schedules", season)
            for season in schedules_seasons
        }
    return {
        "schema_version": 1,
        "sealed_through_season": 2025,
        "datasets": datasets,
    }


def matching_content_digests(lock: dict) -> dict[tuple[str, int], str]:
    """The content digest every CONTENT-strategy pair in *lock* would produce when clean."""
    digests: dict[tuple[str, int], str] = {}
    for dataset, seasons in lock["datasets"].items():
        if SEALED_PROBE_STRATEGY.get(dataset) != PROBE_STRATEGY_CONTENT:
            continue
        for season, entry in seasons.items():
            digests[(dataset, int(season))] = entry["sha256"]
    return digests


class TestTheProbeWatchesTheFileTheLoaderReads:
    """A probe watching a URL nflreadpy does not read is worse than no probe (T-32-10)."""

    @pytest.mark.parametrize("dataset", ["pbp", "depth_charts", "schedules"])
    def test_the_resolved_url_equals_nflreadpys_own(self, dataset: str) -> None:
        from nflreadpy.config import DataFormat
        from nflreadpy.downloader import NflverseDownloader

        season = 2021
        # Build the comparison URL through nflreadpy's OWN construction rather than a
        # second hand-written literal. ``__new__`` skips ``__init__``, which would open a
        # requests.Session and a cache manager; ``_build_url`` uses neither.
        downloader = NflverseDownloader.__new__(NflverseDownloader)
        expected = downloader._build_url(
            "nflverse-data",
            LOADER_PATHS[dataset].format(season=season),
            DataFormat.PARQUET,
        )

        assert asset_download_url(dataset, season) == expected

    def test_schedules_resolves_to_one_file_for_every_season(self) -> None:
        # The decisive fact behind the dataset split: one monolithic asset, so two
        # different seasons resolve to the SAME url.
        assert asset_download_url("schedules", 1999) == asset_download_url(
            "schedules", 2025
        )
        assert asset_download_url("pbp", 1999) != asset_download_url("pbp", 2025)

    def test_an_unknown_dataset_is_refused_and_names_the_known_set(self) -> None:
        with pytest.raises(SealedProbeUnavailable) as excinfo:
            asset_download_url("injuries", 2021)
        assert "injuries" in str(excinfo.value)
        assert "depth_charts" in str(excinfo.value)


class TestTheReleasePayloadIsUntrustedInput:
    """V5: third-party JSON crossing into a sealed decision is validated field by field."""

    @pytest.mark.parametrize(
        ("payload", "offending_field"),
        [
            pytest.param(
                ["not", "a", "mapping"], "mapping", id="payload_not_a_mapping"
            ),
            pytest.param({"tag_name": "pbp"}, "assets", id="assets_absent"),
            pytest.param({"assets": {"a": 1}}, "assets", id="assets_not_a_list"),
            pytest.param({"assets": ["nope"]}, "index 0", id="entry_not_a_mapping"),
            pytest.param(
                {
                    "assets": [
                        {"name": "", "size": 1, "updated_at": "2026-01-01T00:00:00Z"}
                    ]
                },
                "name",
                id="name_empty",
            ),
            pytest.param(
                {
                    "assets": [
                        {
                            "name": "play_by_play_2021.parquet",
                            "size": "20597560",
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
                "size",
                id="size_is_a_string",
            ),
            pytest.param(
                {"assets": [{"name": "play_by_play_2021.parquet", "size": 1}]},
                "updated_at",
                id="updated_at_absent",
            ),
            pytest.param(
                {
                    "assets": [
                        {
                            "name": "play_by_play_2021.parquet",
                            "size": 1,
                            "updated_at": "2026-01-01T00:00:00",
                        }
                    ]
                },
                "updated_at",
                id="updated_at_naive",
            ),
        ],
    )
    def test_a_malformed_payload_is_unavailable_and_names_the_field(
        self, payload: object, offending_field: str
    ) -> None:
        with pytest.raises(SealedProbeUnavailable) as excinfo:
            parse_release_payload(payload)
        assert offending_field in str(excinfo.value)

    def test_the_naive_case_says_the_timestamp_carried_no_timezone(self) -> None:
        payload = {
            "assets": [
                {
                    "name": "play_by_play_2021.parquet",
                    "size": 1,
                    "updated_at": "2026-01-01T00:00:00",
                }
            ]
        }
        with pytest.raises(SealedProbeUnavailable) as excinfo:
            parse_release_payload(payload)
        assert "NO TIMEZONE" in str(excinfo.value)

    def test_the_stored_payloads_parse_and_carry_only_size_and_updated_at(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        for tag in TAGS:
            assert stored_assets[tag], f"{tag} payload resolved no assets"
            for signature in stored_assets[tag].values():
                # created_at and the ETag are deliberately NOT carried through.
                assert set(signature) == {"size", "updated_at"}

    def test_a_boolean_size_is_a_malformed_payload_not_a_small_file(self) -> None:
        payload = {
            "assets": [
                {
                    "name": "play_by_play_2021.parquet",
                    "size": True,
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        with pytest.raises(SealedProbeUnavailable) as excinfo:
            parse_release_payload(payload)
        assert "size" in str(excinfo.value)


class TestASealedRevisionIsReportedWithItsSeverity:
    """T-32-01: the empirically live threat -- four such events in the last nine months."""

    def test_a_shifted_updated_at_is_critical_and_names_the_pair(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        moved = copy.deepcopy(stored_assets)
        moved["pbp"]["play_by_play_2021.parquet"]["updated_at"] = (
            "2027-01-08T05:57:29+00:00"
        )

        verdict = probe_sealed(lock, assets_by_tag=moved)

        assert verdict["event_class"] == RevisionEventClass.SEALED_REVISION
        assert verdict["severity"] == RevisionSeverity.CRITICAL
        assert verdict["checked"] == verdict["expected"]
        assert len(verdict["findings"]) == 1
        finding = verdict["findings"][0]
        assert finding["dataset"] == "pbp"
        assert finding["season"] == 2021
        assert finding["strategy"] == PROBE_STRATEGY_METADATA
        assert finding["observed_updated_at"] == "2027-01-08T05:57:29+00:00"
        assert "pbp 2021" in verdict["reason"]

    def test_the_finding_records_size_beside_updated_at_without_ruling_on_it(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        moved = copy.deepcopy(stored_assets)
        moved["pbp"]["play_by_play_2021.parquet"]["updated_at"] = (
            "2027-01-08T05:57:29+00:00"
        )
        verdict = probe_sealed(lock, assets_by_tag=moved)
        finding = verdict["findings"][0]
        # size is RECORDED on both sides -- weak evidence of a cosmetic rebuild when it is
        # unchanged -- but the class came from the moved timestamp, not from the size.
        assert finding["baseline_size"] == finding["observed_size"]
        assert finding["event_class"] == RevisionEventClass.SEALED_REVISION

    def test_an_unmoved_record_is_clean_with_full_coverage(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets)
        verdict = probe_sealed(
            lock,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(lock),
        )
        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["severity"] == RevisionSeverity.INFORMATIONAL
        assert verdict["findings"] == []
        assert verdict["checked"] == verdict["expected"] == 4

    def test_the_verdict_key_set_is_frozen_and_json_serialisable(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets)
        verdict = probe_sealed(
            lock,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(lock),
            rate_limit_remaining=57,
        )
        assert tuple(verdict) == VERDICT_KEYS
        assert verdict["rate_limit_remaining"] == 57
        # D32-08 appends this to a committed JSONL log; it has to render.
        json.loads(json.dumps(verdict))


class TestABlindProbeSaysSoRatherThanSayingClean:
    """PITFALLS F2, the named failure mode of the whole recommendation."""

    def test_an_absent_asset_is_unknown_and_names_the_unresolved_season(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        blinded = copy.deepcopy(stored_assets)
        del blinded["pbp"]["play_by_play_2021.parquet"]

        verdict = probe_sealed(lock, assets_by_tag=blinded)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["event_class"] != RevisionEventClass.CLEAN
        assert verdict["severity"] == RevisionSeverity.WARNING
        assert verdict["checked"] < verdict["expected"]
        assert [
            (item["dataset"], item["season"]) for item in verdict["unresolved"]
        ] == [("pbp", 2021)]
        assert "2021" in verdict["reason"]

    def test_an_unseeded_baseline_is_unknown_not_clean(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        lock["datasets"]["pbp"]["2022"]["upstream_updated_at"] = None
        lock["datasets"]["pbp"]["2022"]["upstream_size"] = None

        verdict = probe_sealed(lock, assets_by_tag=stored_assets)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["checked"] == verdict["expected"] - 1
        assert verdict["unresolved"][0]["season"] == 2022
        assert "null" in verdict["unresolved"][0]["why"]

    def test_a_whole_tag_that_did_not_fetch_is_unknown(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        blinded = dict(stored_assets)
        blinded["pbp"] = None

        verdict = probe_sealed(lock, assets_by_tag=blinded)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["checked"] == 1
        assert verdict["expected"] == 3

    def test_coverage_beats_a_finding_when_both_are_present(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        # A run that both MISSED a pair and SAW a move reports UNKNOWN, because the
        # statement "I saw everything and one thing moved" is the one it cannot make.
        lock = build_lock(stored_assets, schedules_seasons=())
        blinded = copy.deepcopy(stored_assets)
        del blinded["pbp"]["play_by_play_2021.parquet"]
        blinded["pbp"]["play_by_play_2022.parquet"]["updated_at"] = (
            "2027-02-12T12:00:00+00:00"
        )

        verdict = probe_sealed(lock, assets_by_tag=blinded)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN

    def test_a_content_pair_with_no_supplied_digest_is_unknown(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, pbp_seasons=(), depth_seasons=())
        verdict = probe_sealed(lock, assets_by_tag=stored_assets)
        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["checked"] == 0
        assert verdict["expected"] == 1


class TestATransportFailureIsRecordedNotSwallowed:
    """D32-07: the probe cannot see, so it says so -- outside every broad ``except``."""

    def test_a_connection_error_becomes_sealed_probe_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(*args: object, **kwargs: object) -> None:
            raise requests.ConnectionError("name resolution failed")

        monkeypatch.setattr(requests, "get", _explode)

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            fetch_release_assets("pbp")

        assert "name resolution failed" in str(excinfo.value)
        assert "ConnectionError" in str(excinfo.value)

    def test_the_exception_sits_outside_every_existing_handler(self) -> None:
        assert not issubclass(
            SealedProbeUnavailable, RuntimeError | ValueError | ImportError
        )
        assert not isinstance(SealedProbeUnavailable("x"), ConnectionError)

    def test_a_rate_limited_403_is_legible_as_a_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Response:
            status_code = 403
            headers = {"x-ratelimit-remaining": "0", "x-ratelimit-limit": "60"}
            content = b""

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
        rate_limits: dict[str, int | None] = {}

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            fetch_release_assets("pbp", rate_limits=rate_limits)

        assert "403" in str(excinfo.value)
        assert "x-ratelimit-remaining=0" in str(excinfo.value)
        assert "60/hr" in str(excinfo.value)
        assert rate_limits == {"pbp": 0}

    def test_an_oversized_body_is_refused_before_it_is_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Response:
            status_code = 200
            headers: dict[str, str] = {}
            content = b"x" * (8 * 1024 * 1024 + 1)

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            fetch_release_assets("pbp")

        assert "ceiling" in str(excinfo.value)

    def test_a_stored_payload_round_trips_through_the_fetch_path(
        self, monkeypatch: pytest.MonkeyPatch, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        body = (FIXTURE_DIR / "pbp.json").read_bytes()

        class _Response:
            status_code = 200
            headers = {"x-ratelimit-remaining": "57"}
            content = body

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
        rate_limits: dict[str, int | None] = {}

        assets = fetch_release_assets("pbp", rate_limits=rate_limits)

        assert assets == stored_assets["pbp"]
        assert rate_limits == {"pbp": 57}


class TestTheSchedulesStrategyIsContentNotMetadata:
    """The 27-of-27 false-positive measurement, pinned as a regression (D32-05)."""

    def test_the_strategy_table_is_exactly_the_measured_split(self) -> None:
        assert SEALED_PROBE_STRATEGY == {
            "pbp": PROBE_STRATEGY_METADATA,
            "depth_charts": PROBE_STRATEGY_METADATA,
            "schedules": PROBE_STRATEGY_CONTENT,
        }

    def test_a_months_newer_schedules_timestamp_with_matching_content_is_no_finding(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        # This is the regression. The real games.parquet asset already carries an
        # updated_at that postdates the pin; push it months further and the verdict must
        # STILL be clean, because the bytes this project pinned did not move.
        lock = build_lock(stored_assets, pbp_seasons=(), depth_seasons=())
        moved = copy.deepcopy(stored_assets)
        moved["schedules"]["games.parquet"]["updated_at"] = "2027-03-01T01:06:14+00:00"
        moved["schedules"]["games.parquet"]["size"] = 999_999

        verdict = probe_sealed(
            lock,
            assets_by_tag=moved,
            content_digests=matching_content_digests(lock),
        )

        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["findings"] == []
        assert verdict["strategy"] == {"schedules": PROBE_STRATEGY_CONTENT}

    def test_all_27_pinned_schedules_seasons_stay_clean_under_a_moved_asset(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        seasons = tuple(range(1999, 2026))
        assert len(seasons) == 27
        lock = build_lock(
            stored_assets,
            pbp_seasons=(),
            depth_seasons=(),
            schedules_seasons=seasons,
        )
        moved = copy.deepcopy(stored_assets)
        moved["schedules"]["games.parquet"]["updated_at"] = "2027-03-01T01:06:14+00:00"

        verdict = probe_sealed(
            lock,
            assets_by_tag=moved,
            content_digests=matching_content_digests(lock),
        )

        assert verdict["checked"] == 27
        assert verdict["event_class"] == RevisionEventClass.CLEAN

    def test_a_moved_schedules_digest_is_a_finding(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, pbp_seasons=(), depth_seasons=())
        digests = matching_content_digests(lock)
        digests[("schedules", 2021)] = "a" * 64

        verdict = probe_sealed(
            lock, assets_by_tag=stored_assets, content_digests=digests
        )

        assert verdict["event_class"] == RevisionEventClass.SEALED_REVISION
        assert verdict["findings"][0]["strategy"] == PROBE_STRATEGY_CONTENT
        assert verdict["findings"][0]["observed_sha256"] == "a" * 64
