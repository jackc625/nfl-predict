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
import hashlib
import json
from pathlib import Path

import pytest
import requests

from data.revision_events import RevisionEventClass, RevisionSeverity
from data.sealed_probe import (
    MAX_CONTENT_STREAM_BYTES,
    MAX_RELEASE_PAYLOAD_BYTES,
    PROBE_STRATEGY_CONTENT,
    PROBE_STRATEGY_METADATA,
    SEALED_PROBE_STRATEGY,
    VERDICT_KEYS,
    SealedProbeUnavailable,
    acknowledge_divergence,
    asset_download_url,
    asset_name_for,
    content_digest_for,
    fetch_release_assets,
    parse_release_payload,
    probe_sealed,
    seed_signatures,
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

    # CR-01. THE ZERO-PAIR LOCK IS THE LIMIT CASE OF THIS WHOLE CLASS, and it was the
    # one left un-ruled. ``data.upstream_pin.load_sealed_lock`` returns ``None`` for an
    # ABSENT file BY DESIGN, so a mistyped or relocated ``--sealed-lock`` reached
    # ``probe_sealed`` as ``None`` -- and ``checked < expected`` is False when both are
    # zero, so the run ruled CLEAN and appended a clean/informational line to the
    # committed config/upstream_probe_log.jsonl. A season of those is indistinguishable
    # from a season of real clean lines, which is exactly the observable D32-08's log
    # exists to make distinguishable.
    @pytest.mark.parametrize(
        ("lock", "shape"),
        [
            (None, "an ABSENT lock file, which load_sealed_lock returns None for"),
            ({}, "a lock document with no 'datasets' key at all"),
            ({"datasets": {}}, "a lock whose 'datasets' mapping is empty"),
        ],
    )
    def test_a_lock_with_zero_pairs_is_unknown_and_never_clean(
        self, lock: dict | None, shape: str
    ) -> None:
        verdict = probe_sealed(lock, assets_by_tag={})

        assert verdict["event_class"] == str(RevisionEventClass.UNKNOWN), (
            f"{shape} ruled {verdict['event_class']!r}. A probe with no baseline to "
            "compare against did not look at anything, and 'checked 0 of 0' must never "
            "read as clean -- a dead detector and a healthy system must not be the same "
            "observable."
        )
        assert verdict["event_class"] != str(RevisionEventClass.CLEAN)
        assert verdict["severity"] == str(RevisionSeverity.WARNING)
        assert verdict["expected"] == 0
        assert verdict["checked"] == 0

    def test_the_zero_pair_reason_names_what_was_missed_and_how_to_fix_it(self) -> None:
        reason = probe_sealed(None, assets_by_tag={})["reason"]

        assert "ZERO" in reason, (
            f"the reason does not name the zero-pair fact:\n{reason}"
        )
        assert "--refresh-sealed-lock" in reason, (
            "the reason does not name the recovery command, so an operator who hit a "
            f"mistyped --sealed-lock has nothing to act on:\n{reason}"
        )
        assert "matched its recorded baseline" not in reason, (
            "the zero-pair run still claims every pair matched its baseline, which is "
            f"the false sentence CR-01 reported:\n{reason}"
        )


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

    def test_the_metadata_body_is_streamed_so_the_ceiling_can_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WR-03. ``stream=True`` is what makes the ceiling a ceiling.

        ``getter(url, timeout=timeout)`` was a NON-STREAMING ``requests.get``, so the
        ENTIRE body was read into memory inside that call and
        ``len(body) > MAX_RELEASE_PAYLOAD_BYTES`` only decided whether to PARSE bytes that
        were already resident. An 8 GB response exhausted the run before the check could
        execute -- the constant's own comment claimed it "refuses an unbounded or hostile
        body rather than handing it to a parser", which the code did not deliver.
        """
        seen: dict[str, object] = {}

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, size: int):
                yield (FIXTURE_DIR / "pbp.json").read_bytes()

        def _get(url, **kwargs):
            seen.update(kwargs)
            return _Response()

        monkeypatch.setattr(requests, "get", _get)
        fetch_release_assets("pbp")

        assert seen.get("stream") is True, (
            "the metadata fetch is not streamed, so the byte ceiling can only be applied "
            f"to a body that is already fully resident. kwargs: {seen}"
        )

    def test_an_unbounded_body_is_abandoned_mid_stream_not_after_it_lands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The running total drops the connection rather than buffering to the end."""
        served = {"chunks": 0}
        chunk = b"x" * (1024 * 1024)

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, size: int):
                while True:  # an endless body, as a hostile server would serve
                    served["chunks"] += 1
                    yield chunk

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            fetch_release_assets("pbp")

        assert "ABANDONED" in str(excinfo.value)
        assert "ceiling" in str(excinfo.value)
        # The endless generator was stopped just past the ceiling instead of being drained.
        assert served["chunks"] <= MAX_RELEASE_PAYLOAD_BYTES // len(chunk) + 2, (
            f"the body was read {served['chunks']} MB past an "
            f"{MAX_RELEASE_PAYLOAD_BYTES}-byte ceiling, so the ceiling did not bind"
        )

    def test_the_content_stream_has_a_ceiling_of_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WR-03's other half: ``content_digest_for`` had NO size ceiling at all.

        It streamed and digested whatever it was served. A separate, much looser constant
        bounds it, because the two ceilings bound subjects three orders of magnitude apart
        -- the largest real asset is the ~20 MB ``games.parquet`` -- and one number serving
        both would have to be the looser of the two.
        """
        assert MAX_CONTENT_STREAM_BYTES > MAX_RELEASE_PAYLOAD_BYTES

        chunk = b"y" * (1024 * 1024)
        served = {"chunks": 0}

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, size: int):
                while True:
                    served["chunks"] += 1
                    yield chunk

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            content_digest_for("schedules", 2010)

        assert "ABANDONED" in str(excinfo.value)
        assert served["chunks"] <= MAX_CONTENT_STREAM_BYTES // len(chunk) + 2

    def test_an_ordinary_content_stream_still_digests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ceiling must not change the answer for a body under it."""
        payload = b"the bytes upstream published"

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, size: int):
                yield payload[:5]
                yield payload[5:]

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

        assert (
            content_digest_for("schedules", 2010) == hashlib.sha256(payload).hexdigest()
        )

    def test_a_stream_failing_mid_read_is_unavailable_not_a_short_digest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated stream must never produce a digest over the bytes that arrived."""

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, size: int):
                yield b"first"
                raise requests.ConnectionError("connection reset mid-stream")

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            content_digest_for("schedules", 2010)

        assert "failed mid-read" in str(excinfo.value)
        assert "connection reset mid-stream" in str(excinfo.value)

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


class TestAKnownDivergenceIsAcknowledgedNotReFrozen:
    """D32-10: rule on a divergence once, with a name and a date, and never re-freeze."""

    RULED_BY = "jack (owner)"
    RULED_AT = "2026-09-11T07:00:00+00:00"
    REASON = (
        "nflverse re-released pbp 2021 on 2026-01-08; the content re-fetch showed the "
        "pinned narrowed columns unchanged, so the pin stands and the move is recorded."
    )
    MOVED_AT = "2027-01-08T05:57:29+00:00"
    MOVED_AGAIN_AT = "2027-06-30T11:11:11+00:00"

    def _moved(
        self, stored_assets: dict[str, dict[str, dict]], updated_at: str
    ) -> dict[str, dict[str, dict]]:
        moved = copy.deepcopy(stored_assets)
        moved["pbp"]["play_by_play_2021.parquet"]["updated_at"] = updated_at
        return moved

    def _acknowledged_lock(
        self, stored_assets: dict[str, dict[str, dict]], updated_at: str
    ) -> dict:
        lock = build_lock(stored_assets, schedules_seasons=())
        observed = self._moved(stored_assets, updated_at)["pbp"][
            "play_by_play_2021.parquet"
        ]
        return acknowledge_divergence(
            lock,
            dataset="pbp",
            season=2021,
            observed_updated_at=observed["updated_at"],
            observed_size=observed["size"],
            observed_sha256=None,
            ruled_by=self.RULED_BY,
            ruled_at_utc=self.RULED_AT,
            reason=self.REASON,
        )

    def test_a_probe_matching_the_acknowledged_signature_is_informational(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._acknowledged_lock(stored_assets, self.MOVED_AT)

        verdict = probe_sealed(
            lock, assets_by_tag=self._moved(stored_assets, self.MOVED_AT)
        )

        assert verdict["event_class"] == RevisionEventClass.KNOWN_DIVERGENCE_STABLE
        assert verdict["severity"] == RevisionSeverity.INFORMATIONAL
        assert verdict["checked"] == verdict["expected"]
        finding = verdict["findings"][0]
        assert finding["acknowledged"] is True
        assert finding["ruled_by"] == self.RULED_BY
        assert finding["ruled_at_utc"] == self.RULED_AT
        assert finding["superseded_acknowledgement"] is False

    def test_a_new_move_on_top_of_an_acknowledgement_re_fires_critical(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._acknowledged_lock(stored_assets, self.MOVED_AT)

        verdict = probe_sealed(
            lock, assets_by_tag=self._moved(stored_assets, self.MOVED_AGAIN_AT)
        )

        assert verdict["event_class"] == RevisionEventClass.SEALED_REVISION
        assert verdict["severity"] == RevisionSeverity.CRITICAL
        finding = verdict["findings"][0]
        assert finding["acknowledged"] is False
        assert finding["superseded_acknowledgement"] is True
        assert finding["observed_updated_at"] == self.MOVED_AGAIN_AT

    def test_a_signature_back_at_the_baseline_is_not_a_finding_at_all(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        # Precedence: BASELINE first, acknowledgement second. An acknowledgement does not
        # make the baseline stop being the baseline.
        lock = self._acknowledged_lock(stored_assets, self.MOVED_AT)

        verdict = probe_sealed(lock, assets_by_tag=stored_assets)

        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["findings"] == []

    def test_the_committed_pin_is_byte_identical_across_both(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        pin_path = Path("config/upstream_pin.json")
        if not pin_path.is_file():
            pytest.skip(f"the committed pin is not present at {pin_path}")
        before = pin_path.read_bytes()

        lock = build_lock(stored_assets, schedules_seasons=())
        probe_sealed(lock, assets_by_tag=self._moved(stored_assets, self.MOVED_AT))
        acknowledged = self._acknowledged_lock(stored_assets, self.MOVED_AT)
        probe_sealed(
            acknowledged, assets_by_tag=self._moved(stored_assets, self.MOVED_AT)
        )

        assert pin_path.read_bytes() == before, (
            "an acknowledgement that moved config/upstream_pin.json is a RE-FREEZE, "
            "which D32-10 refuses: it would erase the evidence that upstream moved."
        )

    def test_the_acknowledgement_never_touches_the_input_lock(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        snapshot = copy.deepcopy(lock)
        acknowledge_divergence(
            lock,
            dataset="pbp",
            season=2021,
            observed_updated_at=self.MOVED_AT,
            observed_size=1,
            observed_sha256=None,
            ruled_by=self.RULED_BY,
            ruled_at_utc=self.RULED_AT,
            reason=self.REASON,
        )
        assert lock == snapshot

    @pytest.mark.parametrize(
        ("ruled_by", "reason", "blank_field"),
        [
            pytest.param("   ", "a real reason", "ruled_by", id="blank_ruled_by"),
            pytest.param("jack (owner)", "\t\n ", "reason", id="blank_reason"),
            pytest.param("", "", "ruled_by", id="both_blank"),
        ],
    )
    def test_an_acknowledgement_without_an_attributed_ruler_or_reason_is_refused(
        self,
        stored_assets: dict[str, dict[str, dict]],
        ruled_by: str,
        reason: str,
        blank_field: str,
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        with pytest.raises(ValueError) as excinfo:
            acknowledge_divergence(
                lock,
                dataset="pbp",
                season=2021,
                observed_updated_at=self.MOVED_AT,
                observed_size=1,
                observed_sha256=None,
                ruled_by=ruled_by,
                ruled_at_utc=self.RULED_AT,
                reason=reason,
            )
        message = str(excinfo.value)
        assert blank_field in message
        assert "suppression" in message

    def test_a_second_acknowledgement_moves_the_first_into_supersedes(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        first = self._acknowledged_lock(stored_assets, self.MOVED_AT)
        second = acknowledge_divergence(
            first,
            dataset="pbp",
            season=2021,
            observed_updated_at=self.MOVED_AGAIN_AT,
            observed_size=2,
            observed_sha256=None,
            ruled_by="jack (owner), second ruling",
            ruled_at_utc="2027-07-01T00:00:00+00:00",
            reason="a second re-release, re-checked and again content-equivalent",
        )

        block = second["datasets"]["pbp"]["2021"]["acknowledgement"]
        assert block["observed_updated_at"] == self.MOVED_AGAIN_AT
        assert block["supersedes"]["observed_updated_at"] == self.MOVED_AT
        assert block["supersedes"]["ruled_by"] == self.RULED_BY
        # The first document is untouched -- the history is append-shaped.
        assert first["datasets"]["pbp"]["2021"]["acknowledgement"]["supersedes"] is None

    def test_the_run_severity_is_the_highest_contribution(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._acknowledged_lock(stored_assets, self.MOVED_AT)
        assets = self._moved(stored_assets, self.MOVED_AT)
        # ... and a SECOND pair that moved and was never ruled on.
        assets["pbp"]["play_by_play_2022.parquet"]["updated_at"] = self.MOVED_AGAIN_AT

        verdict = probe_sealed(lock, assets_by_tag=assets)

        classes = {finding["event_class"] for finding in verdict["findings"]}
        assert classes == {
            RevisionEventClass.KNOWN_DIVERGENCE_STABLE,
            RevisionEventClass.SEALED_REVISION,
        }
        assert verdict["event_class"] == RevisionEventClass.SEALED_REVISION
        assert verdict["severity"] == RevisionSeverity.CRITICAL

    def test_acknowledging_a_pair_the_lock_does_not_track_is_refused(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = build_lock(stored_assets, schedules_seasons=())
        with pytest.raises(ValueError) as excinfo:
            acknowledge_divergence(
                lock,
                dataset="pbp",
                season=1999,
                observed_updated_at=self.MOVED_AT,
                observed_size=1,
                observed_sha256=None,
                ruled_by=self.RULED_BY,
                ruled_at_utc=self.RULED_AT,
                reason=self.REASON,
            )
        assert "pbp 1999" in str(excinfo.value)


class TestSeedingABaselineIsDeliberateAndAttributed:
    """The one-time baseline writer 32-09 calls. It never touches the pin either."""

    SEEDED_AT = "2026-09-11T07:00:00+00:00"
    SEEDED_BY = "jack (owner), Plan 32-09 attributed seeding run"

    def _unseeded(self, stored_assets: dict[str, dict[str, dict]]) -> dict:
        lock = build_lock(stored_assets)
        for dataset in ("pbp", "depth_charts"):
            for entry in lock["datasets"][dataset].values():
                entry["upstream_updated_at"] = None
                entry["upstream_size"] = None
        return lock

    def test_it_fills_every_metadata_baseline_and_stamps_its_attribution(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)

        seeded = seed_signatures(
            lock,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(lock),
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
        )

        assert seeded["signatures_seeded_at_utc"] == self.SEEDED_AT
        assert seeded["signatures_seeded_by"] == self.SEEDED_BY
        assert "signatures_unseeded" not in seeded
        for dataset in ("pbp", "depth_charts"):
            for season, entry in seeded["datasets"][dataset].items():
                observed = stored_assets[dataset][asset_name_for(dataset, int(season))]
                assert entry["upstream_updated_at"] == observed["updated_at"]
                assert entry["upstream_size"] == observed["size"]

    def test_a_seeded_lock_then_probes_clean(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)
        seeded = seed_signatures(
            lock,
            assets_by_tag=stored_assets,
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
        )
        verdict = probe_sealed(
            seeded,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(seeded),
        )
        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["checked"] == verdict["expected"]

    def test_it_leaves_the_content_strategy_sha256_untouched(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)
        before = copy.deepcopy(lock["datasets"]["schedules"])
        seeded = seed_signatures(
            lock,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(lock),
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
        )
        assert seeded["datasets"]["schedules"] == before

    def test_it_refuses_to_re_seed_a_recorded_baseline_without_permission(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        already_seeded = build_lock(stored_assets)
        with pytest.raises(ValueError) as excinfo:
            seed_signatures(
                already_seeded,
                assets_by_tag=stored_assets,
                seeded_at_utc=self.SEEDED_AT,
                seeded_by=self.SEEDED_BY,
            )
        assert "refusing to re-seed" in str(excinfo.value)
        assert "overwrite=True" in str(excinfo.value)

    def test_overwrite_true_with_an_attributed_seeder_is_permitted(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        already_seeded = build_lock(stored_assets)
        already_seeded["datasets"]["pbp"]["2021"]["upstream_updated_at"] = (
            "1999-01-01T00:00:00+00:00"
        )

        reseeded = seed_signatures(
            already_seeded,
            assets_by_tag=stored_assets,
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
            overwrite=True,
        )

        observed = stored_assets["pbp"]["play_by_play_2021.parquet"]
        assert (
            reseeded["datasets"]["pbp"]["2021"]["upstream_updated_at"]
            == observed["updated_at"]
        )

    @pytest.mark.parametrize(
        ("seeded_by", "seeded_at"),
        [
            pytest.param("  ", "2026-09-11T07:00:00+00:00", id="blank_seeded_by"),
            pytest.param("jack (owner)", "", id="blank_seeded_at"),
        ],
    )
    def test_an_unattributed_seeding_run_is_refused(
        self,
        stored_assets: dict[str, dict[str, dict]],
        seeded_by: str,
        seeded_at: str,
    ) -> None:
        lock = self._unseeded(stored_assets)
        with pytest.raises(ValueError) as excinfo:
            seed_signatures(
                lock,
                assets_by_tag=stored_assets,
                seeded_at_utc=seeded_at,
                seeded_by=seeded_by,
            )
        assert "suppression" in str(excinfo.value)

    def test_it_leaves_an_existing_acknowledgement_untouched(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)
        lock = acknowledge_divergence(
            lock,
            dataset="pbp",
            season=2021,
            observed_updated_at="2027-01-08T05:57:29+00:00",
            observed_size=7,
            observed_sha256=None,
            ruled_by="jack (owner)",
            ruled_at_utc="2026-09-11T07:00:00+00:00",
            reason="recorded before the baseline was ever seeded",
        )
        block = copy.deepcopy(lock["datasets"]["pbp"]["2021"]["acknowledgement"])

        seeded = seed_signatures(
            lock,
            assets_by_tag=stored_assets,
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
        )

        assert seeded["datasets"]["pbp"]["2021"]["acknowledgement"] == block

    def test_an_unresolvable_asset_is_named_rather_than_seeded_silently(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)
        blinded = copy.deepcopy(stored_assets)
        del blinded["pbp"]["play_by_play_2021.parquet"]

        seeded = seed_signatures(
            lock,
            assets_by_tag=blinded,
            seeded_at_utc=self.SEEDED_AT,
            seeded_by=self.SEEDED_BY,
        )

        assert seeded["datasets"]["pbp"]["2021"]["upstream_updated_at"] is None
        assert [
            (item["dataset"], item["season"]) for item in seeded["signatures_unseeded"]
        ] == [("pbp", 2021)]
        # And the next probe reports it UNRESOLVED rather than clean.
        verdict = probe_sealed(
            seeded,
            assets_by_tag=stored_assets,
            content_digests=matching_content_digests(seeded),
        )
        assert verdict["event_class"] == RevisionEventClass.UNKNOWN

    def test_seeding_over_an_unruled_content_divergence_is_refused(
        self, stored_assets: dict[str, dict[str, dict]]
    ) -> None:
        lock = self._unseeded(stored_assets)
        digests = matching_content_digests(lock)
        digests[("schedules", 2021)] = "b" * 64

        with pytest.raises(SealedProbeUnavailable) as excinfo:
            seed_signatures(
                lock,
                assets_by_tag=stored_assets,
                content_digests=digests,
                seeded_at_utc=self.SEEDED_AT,
                seeded_by=self.SEEDED_BY,
            )
        assert "refusing to seed" in str(excinfo.value)
