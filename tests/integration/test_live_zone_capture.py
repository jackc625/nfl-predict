"""The live-zone capture writes a 2026 week WITHOUT moving one sealed byte.

All tests use tmp_path as both the data root and the live manifest directory, so no real
data lake is touched. The ``data_boundary_guard`` fixture is belt and braces on top of
that: it content-hashes the PRODUCTION ``data/`` tree around every test in this module,
so a capture that escaped the fixture root would be reported here rather than discovered
months later.

THE CLAIM IS MADE BY CONTENT DIGEST, NEVER BY GIT. ``git status --porcelain data/``
cannot fail -- ``.gitignore:22`` blankets the tree, so it returns empty whether the
archive is intact or destroyed. Every "the sealed zone did not move" assertion below
compares ``tests.data_boundary.digest_tree`` snapshots taken either side of the capture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data import storage as data_storage
from data import upstream_live, upstream_pin
from data.upstream_pin import (
    LIVE_OPT_IN_ENV,
    MANIFEST_SCHEMA_VERSION,
    UpstreamPinMissing,
    ZoneWriteRefused,
    digest_file,
)
from scripts import capture_live_season, pin_upstream_snapshot
from tests.data_boundary import diff_digests, digest_tree, is_stat_signature

SEALED_SEASON = upstream_pin.SEALED_THROUGH_SEASON
LIVE_SEASON = upstream_pin.LIVE_ZONE_FIRST_SEASON

# Plan 32-08 wired both detectors INTO the capture, so every call below now runs them.
# ``sealed_probe_offline`` (tests/conftest.py) keeps that offline and off the committed
# probe log; the detector records the stubbed failure as an explicit UNKNOWN, which is the
# guard working rather than being bypassed.
pytestmark = pytest.mark.usefixtures("sealed_probe_offline")

# The week being PREDICTED (D32-13, ratified 2026-09-11). Deliberately NOT related by
# arithmetic to the weeks the captured frame carries -- see CAPTURED_WEEKS below.
PREDICTED_WEEK = 3

# The captured 2026 frame spans weeks 2 and 3. The max is EQUAL to the predicted week,
# not one below it: the Thursday game that opens week 3 is played before the Friday
# freeze, so a week-3 capture normally already holds week-3 rows. Nothing here asserts
# a computed relationship between the label and the content.
CAPTURED_WEEKS = (2, 3)


def _pbp_frame(season: int, weeks: tuple[int, ...] = (1,), epa: float = 0.25):
    """A minimal frame carrying the pinned play-by-play column names."""
    rows = []
    for week in weeks:
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_HOME_AWAY",
                "season": season,
                "week": week,
                "posteam": "HOME",
                "defteam": "AWAY",
                "epa": epa,
            }
        )
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_HOME_AWAY",
                "season": season,
                "week": week,
                "posteam": "AWAY",
                "defteam": "HOME",
                "epa": -epa,
            }
        )
    return pd.DataFrame(rows)


def _write_pin(
    tmp_path: Path,
    dataset: str,
    frames: dict[int, pd.DataFrame],
) -> tuple[Path, Path]:
    """Write a SEALED pin for *frames* under *tmp_path*; return (manifest, data_root).

    The same shape as ``tests/unit/test_upstream_pin.py::_write_pin`` -- the sealed zone
    this module proves did not move has to be built the way the sealed zone really is.
    """
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


def _assert_content_hashes(digests: dict[str, str], side: str) -> None:
    """Refuse to compare a stat signature against a content hash.

    ``tests.data_boundary.digest_file`` falls back to ``stat-size-mtime:`` for a locked
    file and SAYS SO in the value. Comparing that against a real sha256 would report a
    move that is really an instrument change, or hide one that is real. A comparison that
    cannot be trusted must FAIL, never quietly pass.
    """
    degraded = sorted(key for key, value in digests.items() if is_stat_signature(value))
    assert degraded == [], (
        f"the {side} digest snapshot fell back to a stat signature for {degraded}, so "
        "this comparison would mix a content hash with a size/mtime signature. The "
        "fixture root holds no locked file; investigate rather than relaxing this."
    )


@pytest.fixture
def live_root(tmp_path: Path) -> dict[str, Path]:
    """A sealed pin, a live manifest dir and a data root, all inside tmp_path."""
    manifest_path, data_root = _write_pin(
        tmp_path, "pbp", {SEALED_SEASON: _pbp_frame(SEALED_SEASON)}
    )
    return {
        "manifest_path": manifest_path,
        "data_root": data_root,
        "manifest_dir": tmp_path / "upstream_live",
    }


@pytest.fixture
def captured_live_frame(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Make the live fetch return a synthetic 2026 frame instead of reaching nflverse."""
    frame = _pbp_frame(LIVE_SEASON, weeks=CAPTURED_WEEKS, epa=0.5)

    def _fake_fetch(dataset: str, season: int) -> pd.DataFrame:
        assert dataset == "pbp", f"unexpected dataset fetched: {dataset}"
        assert season == LIVE_SEASON, f"unexpected season fetched: {season}"
        return frame.copy()

    monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _fake_fetch)
    return frame


class TestOneLiveWeekIsCapturedWithoutMovingTheSealedZone:
    """The whole shape of the phase, end to end, on the thinnest real path."""

    def test_capture_read_back_and_the_sealed_digests_do_not_move(
        self,
        live_root: dict[str, Path],
        captured_live_frame: pd.DataFrame,
        data_boundary_guard: dict[str, str],
    ) -> None:
        data_root = live_root["data_root"]
        before = digest_tree(data_root)
        _assert_content_hashes(before, "before")
        assert before, "the sealed fixture pin wrote nothing, so this proves nothing"

        code = capture_live_season.main(
            [
                "--season",
                str(LIVE_SEASON),
                "--week",
                str(PREDICTED_WEEK),
                "--dataset",
                "pbp",
                "--data-root",
                str(data_root),
                "--manifest-dir",
                str(live_root["manifest_dir"]),
            ]
        )
        assert code == capture_live_season.EXIT_OK

        after = digest_tree(data_root)
        _assert_content_hashes(after, "after")
        diff = diff_digests(before, after)

        assert diff["changed"] == [], (
            "the live capture REWROTE a file the sealed zone owns:\n  "
            + "\n  ".join(diff["changed"])
        )
        assert diff["removed"] == [], (
            "the live capture REMOVED a file:\n  " + "\n  ".join(diff["removed"])
        )
        assert len(diff["added"]) == 1, (
            f"expected exactly one new bronze snapshot, got {diff['added']}"
        )
        assert all(key.startswith("bronze/") for key in diff["added"]), (
            f"the live capture wrote outside the bronze snapshot layer: {diff['added']}"
        )

        combined = upstream_pin.load_pbp(
            [SEALED_SEASON, LIVE_SEASON],
            manifest_path=live_root["manifest_path"],
            data_root=data_root,
            live_manifest_dir=live_root["manifest_dir"],
        )

        sealed_frame = _pbp_frame(SEALED_SEASON)
        head = combined.iloc[: len(sealed_frame)].reset_index(drop=True)
        tail = combined.iloc[len(sealed_frame) :].reset_index(drop=True)
        pd.testing.assert_frame_equal(head, sealed_frame)
        pd.testing.assert_frame_equal(tail, captured_live_frame.reset_index(drop=True))

    def test_the_manifest_records_the_predicted_week_and_the_observed_content(
        self,
        live_root: dict[str, Path],
        captured_live_frame: pd.DataFrame,
        data_boundary_guard: dict[str, str],
    ) -> None:
        """D32-13: the label is stated, the content is OBSERVED, neither is derived."""
        capture_live_season.main(
            [
                "--season",
                str(LIVE_SEASON),
                "--week",
                str(PREDICTED_WEEK),
                "--dataset",
                "pbp",
                "--data-root",
                str(live_root["data_root"]),
                "--manifest-dir",
                str(live_root["manifest_dir"]),
            ]
        )

        manifest = upstream_live.load_live_manifest(
            LIVE_SEASON, manifest_dir=live_root["manifest_dir"]
        )
        assert manifest is not None
        entry = upstream_live.resolve_capture(manifest, "pbp")

        assert entry["week"] == PREDICTED_WEEK
        assert entry["sequence"] == 1
        assert entry["week_label_means"] == upstream_live.WEEK_LABEL_MEANS
        assert "PREDICTED" in entry["week_label_means"]
        assert entry["content_through_week"] == max(CAPTURED_WEEKS)
        assert entry["rows"] == len(captured_live_frame)
        assert entry["path"].startswith("bronze/")
        assert len(entry["sha256"]) == 64

    def test_a_sealed_season_is_refused_before_anything_is_fetched(
        self,
        live_root: dict[str, Path],
        data_boundary_guard: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The zone boundary is un-crossable in the write direction (T-32-08)."""

        def _explode(dataset: str, season: int) -> pd.DataFrame:
            msg = "a sealed season reached the fetch, so the refusal is not pre-fetch"
            raise AssertionError(msg)

        monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _explode)
        before = digest_tree(live_root["data_root"])

        with pytest.raises(ZoneWriteRefused) as error:
            capture_live_season.capture_live_dataset(
                "pbp",
                SEALED_SEASON,
                PREDICTED_WEEK,
                data_root=live_root["data_root"],
                manifest_dir=live_root["manifest_dir"],
            )

        message = str(error.value)
        assert "pin_upstream_snapshot" in message
        assert "sealed" in message
        assert diff_digests(before, digest_tree(live_root["data_root"])) == {
            "added": [],
            "removed": [],
            "changed": [],
        }

    def test_the_cli_turns_that_refusal_into_a_usage_exit_code(
        self,
        live_root: dict[str, Path],
        data_boundary_guard: dict[str, str],
    ) -> None:
        code = capture_live_season.main(
            [
                "--season",
                str(SEALED_SEASON),
                "--week",
                str(PREDICTED_WEEK),
                "--dataset",
                "pbp",
                "--data-root",
                str(live_root["data_root"]),
                "--manifest-dir",
                str(live_root["manifest_dir"]),
            ]
        )
        assert code == capture_live_season.EXIT_USAGE


class TestTheZonePartitionDecidesWhatIsFetched:
    """PIN-02, proved through the real capture path rather than a stubbed manifest."""

    def test_an_uncovered_live_season_fetches_ITSELF_only(
        self,
        live_root: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        data_boundary_guard: dict[str, str],
    ) -> None:
        """The literal defect: line 354 passed the WHOLE request, not the missing half."""
        fetched: list[str] = []

        def _record(dataset: str, seasons: list[int]) -> pd.DataFrame:
            fetched.append(f"{dataset}:{seasons}")
            return _pbp_frame(seasons[0], weeks=CAPTURED_WEEKS)

        monkeypatch.setattr(upstream_pin, "_fetch_live", _record)
        monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")

        with pytest.warns(upstream_pin.UpstreamPinBypassedWarning):
            combined = upstream_pin.load_pbp(
                [SEALED_SEASON, LIVE_SEASON],
                manifest_path=live_root["manifest_path"],
                data_root=live_root["data_root"],
                live_manifest_dir=live_root["manifest_dir"],
            )

        assert fetched == [f"pbp:[{LIVE_SEASON}]"], (
            "the uncovered live season sent the PINNED season back to nflverse too, "
            f"which is PIN-02: {fetched}"
        )
        assert list(combined["season"]) == [SEALED_SEASON] * 2 + [LIVE_SEASON] * 4

    def test_without_the_opt_in_the_whole_request_refuses_and_names_the_zone(
        self,
        live_root: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        data_boundary_guard: dict[str, str],
    ) -> None:
        """All-or-nothing is preserved; the refusal gained the zone and the tool."""
        attempts: list[str] = []

        def _explode(dataset: str, seasons: list[int]) -> pd.DataFrame:
            attempts.append(f"{dataset}:{seasons}")
            msg = "a refusal fell through to the network"
            raise AssertionError(msg)

        monkeypatch.setattr(upstream_pin, "_fetch_live", _explode)
        monkeypatch.delenv(LIVE_OPT_IN_ENV, raising=False)

        with pytest.raises(UpstreamPinMissing) as error:
            upstream_pin.load_pbp(
                [SEALED_SEASON, LIVE_SEASON],
                manifest_path=live_root["manifest_path"],
                data_root=live_root["data_root"],
                live_manifest_dir=live_root["manifest_dir"],
            )

        message = str(error.value)
        assert str(LIVE_SEASON) in message
        assert "live" in message
        assert "capture_live_season" in message
        assert f"{LIVE_SEASON}  zone live" in message, (
            "the refusal does not say WHICH ZONE the uncovered season belongs to, so it "
            "cannot point at the right capture tool"
        )
        assert f"season(s) {LIVE_SEASON}." in message, (
            "the refusal named a season that IS covered among the uncovered ones"
        )
        assert attempts == []


def _capture_argv(
    live_root: dict[str, Path], week: int, dataset: str = "pbp"
) -> list[str]:
    """The CLI arguments for one capture against the fixture roots."""
    return [
        "--season",
        str(LIVE_SEASON),
        "--week",
        str(week),
        "--dataset",
        dataset,
        "--data-root",
        str(live_root["data_root"]),
        "--manifest-dir",
        str(live_root["manifest_dir"]),
    ]


def _live_manifest(live_root: dict[str, Path]) -> dict:
    manifest = upstream_live.load_live_manifest(
        LIVE_SEASON, manifest_dir=live_root["manifest_dir"]
    )
    assert manifest is not None, "the capture wrote no live manifest at all"
    return manifest


@pytest.fixture
def sequenced_live_frames(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Hand out a DIFFERENT frame on each fetch, so two captures are distinguishable.

    Identical frames would make the append test pass for the wrong reason: two entries
    pointing at byte-identical parquet cannot show that the FIRST one's bytes survived.
    """
    frames = [
        _pbp_frame(LIVE_SEASON, weeks=CAPTURED_WEEKS, epa=0.5),
        _pbp_frame(LIVE_SEASON, weeks=CAPTURED_WEEKS, epa=0.75),
    ]
    handed: list[int] = []

    def _fake_fetch(dataset: str, season: int) -> pd.DataFrame:
        assert dataset == "pbp", f"unexpected dataset fetched: {dataset}"
        assert season == LIVE_SEASON, f"unexpected season fetched: {season}"
        index = min(len(handed), len(frames) - 1)
        handed.append(index)
        return frames[index].copy()

    monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _fake_fetch)
    return frames


class TestASecondCaptureAppendsRatherThanRewrites:
    """D32-14: a re-capture of the same week is a recorded fact, never an erasure."""

    def test_two_captures_for_the_same_week_both_resolve_and_the_first_is_unchanged(
        self,
        live_root: dict[str, Path],
        sequenced_live_frames: list[pd.DataFrame],
        data_boundary_guard: dict[str, str],
    ) -> None:
        data_root = live_root["data_root"]

        assert capture_live_season.main(_capture_argv(live_root, PREDICTED_WEEK)) == (
            capture_live_season.EXIT_OK
        )
        first = _live_manifest(live_root)["datasets"]["pbp"]["captures"][0]
        first_path = first["path"]
        first_sha = first["sha256"]
        before_second = digest_tree(data_root)
        _assert_content_hashes(before_second, "before the second capture")

        assert capture_live_season.main(_capture_argv(live_root, PREDICTED_WEEK)) == (
            capture_live_season.EXIT_OK
        )

        captures = _live_manifest(live_root)["datasets"]["pbp"]["captures"]
        assert len(captures) == 2, (
            f"the second capture did not APPEND -- captures list is {captures}"
        )
        assert [capture["sequence"] for capture in captures] == [1, 2]

        assert captures[0]["path"] == first_path, (
            "the second capture rewrote the FIRST entry's recorded path"
        )
        assert captures[0]["sha256"] == first_sha, (
            "the second capture rewrote the FIRST entry's recorded digest"
        )
        assert captures[1]["path"] != first_path, (
            "both captures point at the same bronze file, so one overwrote the other"
        )

        after_second = digest_tree(data_root)
        _assert_content_hashes(after_second, "after the second capture")
        diff = diff_digests(before_second, after_second)
        assert diff["changed"] == [], (
            "the second capture REWROTE bytes the first capture owns:\n  "
            + "\n  ".join(diff["changed"])
        )
        assert diff["removed"] == [], (
            "the second capture REMOVED a file:\n  " + "\n  ".join(diff["removed"])
        )
        assert len(diff["added"]) == 1, (
            f"expected exactly one new bronze snapshot, got {diff['added']}"
        )

        manifest = _live_manifest(live_root)
        newest = upstream_live.resolve_capture(manifest, "pbp", week=PREDICTED_WEEK)
        assert newest["sequence"] == 2, (
            "a default read of a week did not take its NEWEST capture"
        )
        oldest = upstream_live.resolve_capture(
            manifest, "pbp", week=PREDICTED_WEEK, sequence=1
        )
        assert oldest["sequence"] == 1

        pd.testing.assert_frame_equal(
            upstream_live.read_live_frame(oldest, data_root),
            sequenced_live_frames[0].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            upstream_live.read_live_frame(newest, data_root),
            sequenced_live_frames[1].reset_index(drop=True),
        )

    def test_addressing_a_sequence_that_does_not_exist_lists_the_ones_that_do(
        self,
        live_root: dict[str, Path],
        sequenced_live_frames: list[pd.DataFrame],
        data_boundary_guard: dict[str, str],
    ) -> None:
        capture_live_season.main(_capture_argv(live_root, PREDICTED_WEEK))
        capture_live_season.main(_capture_argv(live_root, PREDICTED_WEEK))

        manifest = _live_manifest(live_root)
        with pytest.raises(upstream_pin.UpstreamLiveCaptureMissing) as error:
            upstream_live.resolve_capture(
                manifest, "pbp", week=PREDICTED_WEEK, sequence=9
            )

        message = str(error.value)
        assert f"({PREDICTED_WEEK}, 1)" in message, (
            "the refusal does not list the (week, sequence) pairs that DO exist, so an "
            f"operator cannot retype a valid address:\n{message}"
        )
        assert f"({PREDICTED_WEEK}, 2)" in message

    def test_a_same_second_second_capture_never_overwrites_the_first(
        self,
        live_root: dict[str, Path],
        sequenced_live_frames: list[pd.DataFrame],
        data_boundary_guard: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The known second-resolution filename defect, made non-silent (T-32-09).

        BOTH clocks are pinned, deliberately. ``capture_live_season._utc_stamp_now`` is
        the reservation's seam and ``data.storage.datetime`` is the writer's; in
        production they are the same wall clock, so pinning only one would simulate a
        disagreement that cannot happen and would leave the real collision unexercised.
        """
        frozen = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)

        class _FrozenClock:
            @staticmethod
            def now(tz=None):
                return frozen

        monkeypatch.setattr(
            capture_live_season, "_utc_stamp_now", lambda: "20260911T120000"
        )
        monkeypatch.setattr(data_storage, "datetime", _FrozenClock)
        monkeypatch.setattr(capture_live_season.time, "sleep", lambda _seconds: None)

        first = capture_live_season.capture_live_dataset(
            "pbp",
            LIVE_SEASON,
            PREDICTED_WEEK,
            data_root=live_root["data_root"],
            manifest_dir=live_root["manifest_dir"],
        )
        first_file = live_root["data_root"] / first["path"]
        first_sha = digest_file(first_file)

        with pytest.raises(pin_upstream_snapshot.PinCaptureError) as error:
            capture_live_season.capture_live_dataset(
                "pbp",
                LIVE_SEASON,
                PREDICTED_WEEK,
                data_root=live_root["data_root"],
                manifest_dir=live_root["manifest_dir"],
            )

        message = str(error.value)
        assert first_file.name in message, (
            f"the refusal does not name the colliding file:\n{message}"
        )
        assert digest_file(first_file) == first_sha, (
            "the same-second re-capture OVERWROTE the first snapshot's bytes"
        )
        captures = _live_manifest(live_root)["datasets"]["pbp"]["captures"]
        assert len(captures) == 1, (
            f"a refused capture still acquired a manifest entry: {captures}"
        )


class TestTheWeekLabelMeansThePredictedWeek:
    """D32-13: the label is a CONVENTION, the content is an OBSERVATION."""

    def test_the_entry_states_the_convention_and_the_observed_content(
        self,
        live_root: dict[str, Path],
        data_boundary_guard: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Weeks ``[5, 5, 6]`` under the label ``6`` -- the ORDINARY case, measured.

        32-RESEARCH.md measured live 2026 play-by-play already holding week-1 rows on the
        Thursday of week 1. So a week-6 capture normally carries week-6 rows, and the
        observed content depth is NOT one less than the label. Nothing here derives one
        field from the other; both are read straight off the recorded entry, and no test
        in this phase asserts any arithmetic between them.
        """
        label_week = 6
        frame = pd.DataFrame(
            [
                {
                    "game_id": "2026_05_HOME_AWAY",
                    "season": LIVE_SEASON,
                    "week": 5,
                    "posteam": "HOME",
                    "defteam": "AWAY",
                    "epa": 0.11,
                },
                {
                    "game_id": "2026_05_HOME_AWAY",
                    "season": LIVE_SEASON,
                    "week": 5,
                    "posteam": "AWAY",
                    "defteam": "HOME",
                    "epa": -0.11,
                },
                {
                    "game_id": "2026_06_THU_NIGHT",
                    "season": LIVE_SEASON,
                    "week": 6,
                    "posteam": "HOME",
                    "defteam": "AWAY",
                    "epa": 0.42,
                },
            ]
        )
        monkeypatch.setattr(
            pin_upstream_snapshot, "fetch_live", lambda dataset, season: frame.copy()
        )

        entry = capture_live_season.capture_live_dataset(
            "pbp",
            LIVE_SEASON,
            label_week,
            data_root=live_root["data_root"],
            manifest_dir=live_root["manifest_dir"],
        )

        assert entry["week"] == label_week
        assert entry["content_through_week"] == label_week
        assert entry["weeks_present"] == [5, 6]
        assert "PREDICTED" in entry["week_label_means"]
        assert entry["week_label_semantics_version"] == (
            upstream_live.WEEK_LABEL_SEMANTICS_VERSION
        )
