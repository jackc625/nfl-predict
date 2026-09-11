"""The live zone's CAPTURE BOUNDARY: what it digests, what it records, what it refuses.

TEST CLASS: plain unit tests. Every frame is constructed IN the test and every file is
written under ``tmp_path``, so this module passes on a fresh checkout with no ``data/``,
no network and no captured pin. Nothing here reaches nflverse: the one fetch seam,
``scripts.pin_upstream_snapshot.fetch_live``, is monkeypatched in every test that
exercises it.

Four things are proved, and each one is a place where the obvious implementation is
quietly wrong:

1. **The per-week, per-column digest (D32-06).** A corrected value in one week must move
   exactly that week's digest and exactly that column's digest inside it -- otherwise a
   routine stat correction and a wholesale EPA recompute are the same observable, which is
   the alert-fatigue failure the detector's calibration depends on avoiding. The digest
   must also be independent of the order upstream happened to emit rows in, or it reports
   a revision every single week.
2. **The recorded empty capture (D32-03).** "Upstream had nothing for this dataset at this
   moment" is a fact worth pinning in the LIVE zone -- and the SEALED zone's opposite rule,
   which refuses a zero-row pin, must be left exactly as it is.
3. **The season-window refusal.** ``nflreadpy`` raises ``ValueError`` for two of the three
   datasets when a season is outside its window, and ``ValueError`` is precisely what every
   wired call site catches and converts into an EMPTY frame. A refusal that arrived as a
   bare ``ValueError`` would therefore be indistinguishable from "upstream had no data".
4. **Preserve-absence narrowing on the live path.** All 23 ``PBP_PINNED_COLUMNS`` are
   present in live 2026 today, so this branch is NOT exercised by current evidence -- which
   is exactly why it is held by a test rather than an assumption. The risk is not
   hypothetical: ``depth_charts`` went from a 15-column schema to a 12-column one with ZERO
   name overlap, inside the sealed pin's own coverage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data import upstream_live
from data.upstream_pin import (
    LIVE_ZONE_FIRST_SEASON,
    PBP_PINNED_COLUMNS,
    SEALED_THROUGH_SEASON,
    UpstreamPinError,
    UpstreamSeasonWindowRefused,
)
from scripts import capture_live_season, pin_upstream_snapshot
from scripts.pin_upstream_snapshot import PinCaptureError

LIVE_SEASON = LIVE_ZONE_FIRST_SEASON

# Plan 32-08 wired both detectors INTO the capture, so every call below now runs them.
# ``sealed_probe_offline`` (tests/conftest.py) keeps that offline and off the committed
# probe log; the detector records the stubbed failure as an explicit UNKNOWN, which is the
# guard working rather than being bypassed.
pytestmark = pytest.mark.usefixtures("sealed_probe_offline")


def _pbp_frame(weeks: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """A play-by-play-shaped frame with a UNIQUE ``game_id`` per row.

    Unique identity per row makes the canonical digest ordering total, so a test that
    changes one value is changing exactly one thing.
    """
    rows = []
    for week in weeks:
        for slot in (0, 1):
            rows.append(
                {
                    "game_id": f"{LIVE_SEASON}_{week:02d}_G{slot}",
                    "season": LIVE_SEASON,
                    "week": week,
                    "posteam": "HOME" if slot == 0 else "AWAY",
                    "epa": round(0.1 * week + 0.01 * slot, 4),
                }
            )
    return pd.DataFrame(rows)


def _depth_charts_frame() -> pd.DataFrame:
    """A frame with NO ``week`` column -- the shape ``depth_charts`` can arrive in."""
    return pd.DataFrame(
        [
            {
                "gsis_id": "00-0011111",
                "club_code": "HOME",
                "position": "QB",
                "depth_team": 1,
                "season": LIVE_SEASON,
            },
            {
                "gsis_id": "00-0022222",
                "club_code": "AWAY",
                "position": "QB",
                "depth_team": 1,
                "season": LIVE_SEASON,
            },
        ]
    )


# The EXACT column layout the single real 2026 depth_charts capture recorded, transcribed
# from config/upstream_live/2026.json. WR-01 was reproduced on this layout and nowhere else
# matters: it is the frame the production capture actually digests, 509,781 rows of it, in
# one whole-frame bucket.
_DEPTH_CHARTS_2026_COLUMNS = (
    "dt",
    "team",
    "player_name",
    "espn_id",
    "gsis_id",
    "pos_grp_id",
    "pos_grp",
    "pos_id",
    "pos_name",
    "pos_abb",
    "pos_slot",
    "pos_rank",
)


def _depth_charts_2026_frame() -> pd.DataFrame:
    """Three roster slots in the committed 2026 ``depth_charts`` column layout.

    The player names deliberately sort in the OPPOSITE order to the roster slots, so a
    value-ordered digest and an identity-ordered one produce different row orders and the
    difference between them is observable.
    """
    rows = [
        {
            "dt": "2026-09-05",
            "team": "BUF",
            "player_name": "Zeta Adams",
            "espn_id": "3001",
            "gsis_id": "00-0000001",
            "pos_grp_id": 1,
            "pos_grp": "OFF",
            "pos_id": 1,
            "pos_name": "Quarterback",
            "pos_abb": "QB",
            "pos_slot": 1,
            "pos_rank": 1,
        },
        {
            "dt": "2026-09-05",
            "team": "BUF",
            "player_name": "Mike Brown",
            "espn_id": "3002",
            "gsis_id": "00-0000002",
            "pos_grp_id": 1,
            "pos_grp": "OFF",
            "pos_id": 2,
            "pos_name": "Running Back",
            "pos_abb": "RB",
            "pos_slot": 2,
            "pos_rank": 1,
        },
        {
            "dt": "2026-09-05",
            "team": "BUF",
            "player_name": "Alan Cole",
            "espn_id": "3003",
            "gsis_id": "00-0000003",
            "pos_grp_id": 1,
            "pos_grp": "OFF",
            "pos_id": 3,
            "pos_name": "Wide Receiver",
            "pos_abb": "WR",
            "pos_slot": 3,
            "pos_rank": 1,
        },
    ]
    return pd.DataFrame(rows)[list(_DEPTH_CHARTS_2026_COLUMNS)]


def _entry_for(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    dataset: str = "pbp",
    season: int = LIVE_SEASON,
    week: int = 3,
    raw: pd.DataFrame | None = None,
) -> dict:
    """Write *frame* under *tmp_path* and build the capture entry that describes it."""
    data_root = tmp_path / "data"
    (data_root / "bronze").mkdir(parents=True, exist_ok=True)
    path = data_root / "bronze" / f"{dataset}_{season}_W{week:02d}.parquet"
    frame.to_parquet(path, index=False)
    return upstream_live.build_capture_entry(
        dataset,
        season,
        week,
        raw if raw is not None else frame,
        frame,
        path,
        data_root,
    )


class TestTheCommittedManifestSurvivesAnInterruptedWrite:
    """WR-04. Append-only held in memory and NOT on disk.

    ``append_capture``'s docstring argues that append-only "holds literally, not by
    convention: the only mutation of the capture list is a single ``list.append``". True in
    memory; false on disk. ``write_live_manifest`` used ``path.write_text`` -- a TRUNCATING,
    non-atomic rewrite of the file that holds EVERY capture of the season. An interruption
    or a disk-full part way through left a truncated JSON document, and
    ``load_live_manifest`` would then raise for every subsequent read: the whole season's
    committed record gone, including the verdicts ``attach_verdict`` exists to keep welded
    to their bytes.

    ``data/sealed_probe_log.py`` makes exactly this argument when it chooses JSONL append
    over a re-rendered array -- "A record whose integrity rests on a serialiser
    round-tripping identically for a whole season is not a record" -- and this file was the
    counterexample.
    """

    def test_a_failed_write_leaves_the_previous_manifest_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decisive one: an interrupted write must not truncate the record."""
        manifest_dir = tmp_path / "upstream_live"
        good = upstream_live.empty_live_manifest(LIVE_SEASON)
        good["datasets"]["pbp"] = {"loader": "test", "captures": [{"week": 1}]}
        path = upstream_live.write_live_manifest(good, manifest_dir=manifest_dir)
        before = path.read_bytes()

        def _explode(*args: object, **kwargs: object) -> None:
            msg = "disk full part way through the write"
            raise OSError(msg)

        monkeypatch.setattr(upstream_live.os, "replace", _explode)

        with pytest.raises(OSError, match="disk full"):
            upstream_live.write_live_manifest(good, manifest_dir=manifest_dir)

        assert path.read_bytes() == before, (
            "the failed write damaged the committed record. A truncated manifest makes "
            "load_live_manifest raise for the REST OF THE SEASON, taking every earlier "
            "capture and its verdict with it."
        )
        assert (
            upstream_live.load_live_manifest(LIVE_SEASON, manifest_dir=manifest_dir)
            == good
        )

    def test_no_temp_file_is_left_behind_by_a_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A litter of ``.tmp`` files in a COMMITTED directory is its own problem."""
        manifest_dir = tmp_path / "upstream_live"
        manifest = upstream_live.empty_live_manifest(LIVE_SEASON)
        upstream_live.write_live_manifest(manifest, manifest_dir=manifest_dir)

        def _explode(*args: object, **kwargs: object) -> None:
            msg = "interrupted"
            raise OSError(msg)

        monkeypatch.setattr(upstream_live.os, "replace", _explode)
        with pytest.raises(OSError, match="interrupted"):
            upstream_live.write_live_manifest(manifest, manifest_dir=manifest_dir)

        strays = [
            entry.name for entry in manifest_dir.iterdir() if ".tmp" in entry.name
        ]
        assert strays == [], f"a failed write left temp file(s) behind: {strays}"

    def test_an_ordinary_write_still_round_trips(self, tmp_path: Path) -> None:
        """The atomic path must not change what lands on disk."""
        manifest_dir = tmp_path / "upstream_live"
        manifest = upstream_live.empty_live_manifest(LIVE_SEASON)
        manifest["datasets"]["pbp"] = {
            "loader": "test",
            "captures": [{"week": 6, "sequence": 1}],
        }

        path = upstream_live.write_live_manifest(manifest, manifest_dir=manifest_dir)

        assert json.loads(path.read_text(encoding="utf-8")) == manifest
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_the_bronze_archive_refuses_a_collision_rather_than_overwriting(
        self, tmp_path: Path
    ) -> None:
        """WR-04(b). A second-resolution collision destroyed the earlier snapshot.

        ``save_bronze_snapshot`` builds its filename from a SECOND-resolution UTC stamp, so
        two captures of the same ``(season, week)`` inside one second produced the SAME
        path and the second silently overwrote the first -- breaking ``data/bronze/``'s
        append-only contract literally, and destroying the bytes a published week-N
        prediction was made from. The capture path now creates EXCLUSIVELY, which decides
        the race at the only place it can be decided (the create), and holds across two
        processes where a check-then-write cannot.
        """
        from data import storage

        first = storage.save_bronze_snapshot(
            _pbp_frame(weeks=(1,)),
            table_name="pbp",
            season=LIVE_SEASON,
            week=1,
            base_path=tmp_path,
            exclusive=True,
        )
        original = first.read_bytes()

        # FREEZE THE CLOCK so the second call lands on the same second-resolution stamp.
        # Waiting for a real collision would make the test a coin flip about scheduling --
        # the exact non-determinism the guard exists because of.
        frozen = datetime.strptime(
            first.stem.rsplit("_", 1)[-1], "%Y%m%dT%H%M%S"
        ).replace(tzinfo=UTC)

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen

        monkeypatch_target = storage.datetime
        storage.datetime = _FrozenDatetime  # type: ignore[misc]
        try:
            with pytest.raises(FileExistsError):
                storage.save_bronze_snapshot(
                    _pbp_frame(weeks=(2,)),
                    table_name="pbp",
                    season=LIVE_SEASON,
                    week=1,
                    base_path=tmp_path,
                    exclusive=True,
                )
        finally:
            storage.datetime = monkeypatch_target  # type: ignore[misc]

        assert first.read_bytes() == original, (
            "the colliding write destroyed the earlier snapshot's bytes, which are the "
            "evidence an earlier verdict was measured against"
        )

    def test_the_default_write_is_unchanged_for_every_other_caller(
        self, tmp_path: Path
    ) -> None:
        """The exclusive guard is OPT-IN, and deliberately so.

        ``scripts/ingest_odds_timeline`` writes one bronze snapshot per trajectory
        timestamp for a single ``(season, week)``, so a fast backfill legitimately produces
        several calls inside one second and RELIES on the overwrite today. That reliance is
        a latent data-loss bug in THAT ingester, but it is pre-existing, it belongs to the
        line-movement work rather than to the upstream pin, and turning it into a hard
        failure for every caller at once would convert a quiet defect into a broken
        production path. Pinned here so the scoping is a recorded decision rather than an
        oversight a later reader has to guess at.
        """
        from data.storage import save_bronze_snapshot

        path = save_bronze_snapshot(
            _pbp_frame(weeks=(1,)),
            table_name="pbp",
            season=LIVE_SEASON,
            week=1,
            base_path=tmp_path,
        )
        assert path.is_file()

        # The same path, written again WITHOUT the guard: still permitted.
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.Table.from_pandas(_pbp_frame(weeks=(2,))), path, compression="snappy"
        )
        assert len(pd.read_parquet(path)) == 2


class TestTheCaptureDigestsPerWeekAndPerColumn:
    """D32-06, whose shape is frozen for the whole season the first entry is written."""

    def test_a_changed_value_in_one_week_moves_only_that_weeks_digest(self) -> None:
        before = _pbp_frame()
        after = before.copy()
        target = after.index[(after["week"] == 3) & (after["posteam"] == "HOME")][0]
        after.loc[target, "epa"] = 9.99

        first = upstream_live.week_digests(before)
        second = upstream_live.week_digests(after)

        assert sorted(first) == sorted(second) == ["1", "2", "3"]
        assert first["3"]["frame_sha256"] != second["3"]["frame_sha256"], (
            "a changed value did not move its own week's digest, so a revision in that "
            "week would be invisible"
        )
        assert first["3"]["columns"]["epa"] != second["3"]["columns"]["epa"]

        for week in ("1", "2"):
            assert first[week] == second[week], (
                f"week {week} moved although nothing in it changed -- a detector built on "
                "this would report a revision of every week on every correction"
            )
        unchanged_columns = [
            column for column in first["3"]["columns"] if column != "epa"
        ]
        assert unchanged_columns, (
            "the fixture has no other column to prove isolation with"
        )
        for column in unchanged_columns:
            assert first["3"]["columns"][column] == second["3"]["columns"][column], (
                f"column {column!r} moved although only 'epa' changed, so a routine stat "
                "correction is indistinguishable from a wholesale recompute"
            )

    def test_row_order_does_not_change_any_digest(self) -> None:
        frame = _pbp_frame()
        shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
        assert not shuffled.equals(frame), (
            "the shuffle was a no-op, so this proves nothing"
        )

        assert upstream_live.week_digests(shuffled) == upstream_live.week_digests(
            frame
        ), (
            "the digest depends on the order upstream emitted rows in, so it would report "
            "a revision every week"
        )

    def test_the_recorded_row_counts_are_per_week(self) -> None:
        digests = upstream_live.week_digests(_pbp_frame(weeks=(1, 2, 2, 3)))
        assert {week: bucket["rows"] for week, bucket in digests.items()} == {
            "1": 2,
            "2": 4,
            "3": 2,
        }

    def test_a_frame_without_a_week_column_records_the_whole_frame_bucket(
        self, tmp_path: Path
    ) -> None:
        frame = _depth_charts_frame()
        digests = upstream_live.week_digests(frame, "depth_charts")

        assert list(digests) == [upstream_live.NO_WEEK_COLUMN_BUCKET]
        assert digests[upstream_live.NO_WEEK_COLUMN_BUCKET]["rows"] == len(frame)

        entry = _entry_for(tmp_path, frame, dataset="depth_charts")
        assert entry["week_partition"] == upstream_live.WEEK_PARTITION_WHOLE_FRAME
        assert "no week column" in entry["week_partition"]
        assert entry["week_digests"] == digests, (
            "build_capture_entry digested the frame under a different ordering rule from "
            "the one week_digests applies for the same dataset, so the recorded digests "
            "would not reproduce"
        )

    def test_depth_charts_isolates_per_column_on_its_real_production_layout(
        self,
    ) -> None:
        """WR-01. The whole-frame dataset gets the SAME isolation ``pbp`` gets.

        ``_digest_ordering``'s version-1 fallback was ``keys = list(frame.columns)`` -- a
        sort by every column, BY VALUE -- which is the precise thing its own docstring
        calls the calibration failure D32-06 exists to prevent. It was not a dead branch:
        ``depth_charts`` carries no ``game_id``, so the single 509,781-row production
        bucket took it. MEASURED on this exact layout before the fix: ONE corrected
        ``player_name`` moved SEVEN of twelve column digests, making a routine roster
        correction indistinguishable from a wholesale recompute.

        The pre-fix test suite could not catch this. Per-column isolation was proven only
        on a ``game_id``-bearing frame, and the ``depth_charts`` test checked the bucket
        key and the row count but never the per-column map.
        """
        before = _depth_charts_2026_frame()
        after = before.copy()
        after.loc[0, "player_name"] = "Aaron Adams"

        bucket = upstream_live.NO_WEEK_COLUMN_BUCKET
        first = upstream_live.week_digests(before, "depth_charts")[bucket]
        second = upstream_live.week_digests(after, "depth_charts")[bucket]

        moved = sorted(
            name
            for name, digest in first["columns"].items()
            if second["columns"][name] != digest
        )
        assert moved == ["player_name"], (
            f"a ONE-CELL correction moved {len(moved)} of "
            f"{len(_DEPTH_CHARTS_2026_COLUMNS)} column digests: {moved}. The rows are "
            "being ordered by VALUE, so a corrected cell reorders the frame and drags "
            "every other column's digest with it -- a routine roster correction and a "
            "wholesale recompute become the same observable."
        )
        assert first["frame_sha256"] != second["frame_sha256"], (
            "the whole-frame digest did not move, so the correction is invisible"
        )

    def test_the_depth_charts_ordering_is_recorded_as_an_identity_not_a_value_sort(
        self,
    ) -> None:
        """The ordering is a STATED fact on the bucket, not something to re-derive."""
        bucket = upstream_live.week_digests(_depth_charts_2026_frame(), "depth_charts")[
            upstream_live.NO_WEEK_COLUMN_BUCKET
        ]

        assert bucket["ordering"].startswith("identity: "), (
            f"depth_charts recorded ordering {bucket['ordering']!r}, which is not an "
            "identity ordering"
        )
        for key in upstream_live.DIGEST_SORT_KEYS["depth_charts"]:
            assert key in bucket["ordering"]

    def test_depth_charts_digests_do_not_depend_on_upstream_row_order(self) -> None:
        """The other half of WR-01: identity ordering must still be order-independent."""
        frame = _depth_charts_2026_frame()
        # REVERSED rather than sampled. A random permutation of three rows lands on the
        # identity often enough to make the test vacuous, and a seeded `sample` that
        # happens to be a no-op proves nothing.
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        assert not shuffled.equals(frame), (
            "the reordering was a no-op, so this proves nothing"
        )

        assert upstream_live.week_digests(
            shuffled, "depth_charts"
        ) == upstream_live.week_digests(frame, "depth_charts")

    def test_a_frame_carrying_no_identity_key_is_recorded_unordered_not_value_sorted(
        self,
    ) -> None:
        """The refusal, rather than the silent fallback that caused WR-01.

        A frame with none of its dataset's identity columns cannot be put in a meaningful
        row order, so the per-column map is OMITTED rather than filled with numbers that do
        not answer "did this column change". ``rows`` and ``frame_sha256`` still detect the
        movement; only the per-column ATTRIBUTION is lost, and the bucket says so itself.
        """
        frame = pd.DataFrame({"alpha": [3, 1, 2], "beta": ["x", "y", "z"]})
        bucket = upstream_live.week_digests(frame, "depth_charts")[
            upstream_live.NO_WEEK_COLUMN_BUCKET
        ]

        assert bucket["ordering"] == upstream_live.DIGEST_ORDERING_UNORDERED
        assert bucket["columns"] == {}, (
            "per-column digests were published for an arbitrary row order, which is the "
            "value-ordering failure under a different name"
        )
        assert bucket["rows"] == 3
        assert bucket["frame_sha256"]

    def test_an_incomparable_identity_column_is_unordered_not_a_failed_capture(
        self,
    ) -> None:
        """WR-01's secondary risk: ``sort_values`` raising AFTER the bronze bytes exist.

        ``week_digests`` runs after ``save_bronze_snapshot``, so a ``TypeError`` from a
        mixed-type sort key failed the whole capture and orphaned a snapshot in the
        append-only archive -- a detector detail costing a week's record, which D32-07
        forbids. The ordering is recorded as UNORDERED instead.

        Exercised through a SINGLE-key dataset, because that is the shape that actually
        raises: ``pandas`` compares a single object column directly (``'<' not supported
        between instances of 'str' and 'float'``), while a multi-key sort factorises each
        column first and tolerates the mixture. ``pbp`` and ``schedules`` are the
        single-key datasets, so they are where the hazard lives.
        """
        frame = _pbp_frame(weeks=(1,))
        assert upstream_live.DIGEST_SORT_KEYS["pbp"] == ("game_id",), (
            "this test targets the SINGLE-key sort; pbp is no longer single-key"
        )
        frame["game_id"] = [f"{LIVE_SEASON}_01_G0", 2.5]

        bucket = upstream_live.week_digests(frame, "pbp")["1"]
        assert bucket["ordering"] == upstream_live.DIGEST_ORDERING_UNORDERED
        assert bucket["columns"] == {}
        assert bucket["rows"] == 2

    def test_the_pbp_isolation_is_unchanged_by_the_per_dataset_keys(self) -> None:
        """The dataset that already worked must keep working, explicitly."""
        before = _pbp_frame()
        after = before.copy()
        target = after.index[(after["week"] == 3) & (after["posteam"] == "HOME")][0]
        after.loc[target, "epa"] = 9.99

        first = upstream_live.week_digests(before, "pbp")["3"]
        second = upstream_live.week_digests(after, "pbp")["3"]

        moved = sorted(
            name
            for name, digest in first["columns"].items()
            if second["columns"][name] != digest
        )
        assert moved == ["epa"]
        assert first["ordering"] == "identity: game_id"

    def test_a_zero_row_frame_yields_an_empty_map_and_records_the_empty_case(
        self, tmp_path: Path
    ) -> None:
        empty = _pbp_frame().iloc[0:0]
        assert upstream_live.week_digests(empty) == {}

        entry = _entry_for(tmp_path, empty)
        assert entry["week_digests"] == {}
        assert entry["week_partition"] == upstream_live.WEEK_PARTITION_EMPTY
        assert entry["rows"] == 0

    def test_an_unreadable_week_value_is_refused_rather_than_silently_dropped(
        self,
    ) -> None:
        frame = _pbp_frame(weeks=(1,))
        frame.loc[0, "week"] = None

        with pytest.raises(PinCaptureError) as error:
            upstream_live.week_digests(frame)
        assert "no week bucket" in str(error.value)

    def test_a_frame_carrying_another_season_is_refused(self, tmp_path: Path) -> None:
        """nflreadpy.load_schedules downloads 1999-2026 and filters in memory."""
        leaked = _pbp_frame(weeks=(1,)).copy()
        leaked.loc[0, "season"] = LIVE_SEASON - 1

        with pytest.raises(PinCaptureError) as error:
            _entry_for(tmp_path, leaked, dataset="schedules")

        message = str(error.value)
        assert str(LIVE_SEASON - 1) in message, (
            f"the refusal does not name the leaked season:\n{message}"
        )

    def test_every_entry_carries_the_digest_block_and_its_schema_version(
        self, tmp_path: Path
    ) -> None:
        entry = _entry_for(tmp_path, _pbp_frame())

        assert entry["week_digest_schema_version"] == (
            upstream_live.WEEK_DIGEST_SCHEMA_VERSION
        )
        assert entry["week_partition"] == upstream_live.WEEK_PARTITION_PER_WEEK
        assert sorted(entry["week_digests"]) == ["1", "2", "3"]
        # ``ordering`` is schema version 2's addition (WR-01) and rides on EVERY bucket,
        # never only on the unusual branch -- a bucket shape that depended on which branch
        # produced it is not a shape a reader can diff a season later.
        assert set(entry["week_digests"]["1"]) == {
            "rows",
            "ordering",
            "frame_sha256",
            "columns",
        }
        assert sorted(entry["week_digests"]["1"]["columns"]) == sorted(
            _pbp_frame().columns
        )

    def test_the_digest_block_survives_a_json_round_trip_with_stable_keys(
        self, tmp_path: Path
    ) -> None:
        """The bucket key is ``str(int(week))`` precisely so this is a no-op."""
        entry = _entry_for(tmp_path, _pbp_frame())
        assert json.loads(json.dumps(entry))["week_digests"] == entry["week_digests"]


class TestNarrowIsTheSealedZonesOwnAllowlist:
    """Discretion item 7, recorded: the live path narrows with the SAME allowlist."""

    def test_the_live_capture_module_imports_narrow_and_never_redefines_it(
        self,
    ) -> None:
        import inspect

        from scripts import capture_live_season

        source = inspect.getsource(capture_live_season)
        assert "narrow" in source
        assert "def narrow" not in source, (
            "the live path defines its own narrow(), so the two zones' column sets can "
            "drift apart and a mixed [2025, 2026] frame would be ragged at the boundary"
        )
        assert "def fetch_live(" not in source
        assert "def assert_round_trip_faithful" not in source

    def test_the_allowlist_is_the_sealed_zones_tuple(self) -> None:
        from data.upstream_pin import DATASET_COLUMNS

        assert DATASET_COLUMNS["pbp"] is PBP_PINNED_COLUMNS


# The pinned play-by-play columns whose upstream values are text rather than numbers.
# Named so a fixture frame carries plausible dtypes through the parquet round trip that
# ``assert_round_trip_faithful`` checks.
_TEXT_COLUMNS = frozenset(
    {
        "game_id",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "play_type",
        "passer_player_id",
    }
)


def _full_pbp_frame(
    *, omit: tuple[str, ...] = (), extra: tuple[str, ...] = ()
) -> pd.DataFrame:
    """A two-row frame carrying EVERY ``PBP_PINNED_COLUMNS`` name, minus *omit*.

    *omit* simulates an upstream frame that does not supply an allowlisted column --
    ``narrow``'s preserve-absence branch. *extra* simulates an upstream column outside the
    allowlist, which must be dropped.
    """
    data: dict[str, list] = {}
    for index, column in enumerate(PBP_PINNED_COLUMNS):
        if column in omit:
            continue
        if column == "season":
            data[column] = [LIVE_SEASON, LIVE_SEASON]
        elif column == "week":
            data[column] = [1, 2]
        elif column in _TEXT_COLUMNS:
            data[column] = [f"{column}_a", f"{column}_b"]
        else:
            data[column] = [float(index), float(index) + 0.5]
    for column in extra:
        data[column] = ["upstream_a", "upstream_b"]
    return pd.DataFrame(data)


@pytest.fixture
def live_roots(tmp_path: Path) -> dict[str, Path]:
    """A data root and a live manifest directory, both inside ``tmp_path``."""
    data_root = tmp_path / "data"
    (data_root / "bronze").mkdir(parents=True, exist_ok=True)
    return {"data_root": data_root, "manifest_dir": tmp_path / "upstream_live"}


def _seed_manifest(live_roots: dict[str, Path]) -> tuple[Path, bytes]:
    """Write a capture-less live manifest and return its path and its bytes.

    A refusal has to be shown to leave the COMMITTED record byte-unchanged, and "the file
    does not exist" is a weaker claim than "the file is exactly what it was".
    """
    path = upstream_live.write_live_manifest(
        upstream_live.empty_live_manifest(LIVE_SEASON),
        manifest_dir=live_roots["manifest_dir"],
    )
    return path, path.read_bytes()


def _capture(live_roots: dict[str, Path], dataset: str = "pbp", week: int = 3) -> dict:
    return capture_live_season.capture_live_dataset(
        dataset,
        LIVE_SEASON,
        week,
        data_root=live_roots["data_root"],
        manifest_dir=live_roots["manifest_dir"],
    )


class TestTheLiveZoneRecordsAnEmptyCapture:
    """D32-03: the two zones take OPPOSITE positions on zero rows, on purpose."""

    def test_an_empty_live_capture_is_recorded_with_rows_zero(
        self, live_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2026 play-by-play is legitimately empty before the season's first game."""
        empty = _full_pbp_frame().iloc[0:0]
        monkeypatch.setattr(
            pin_upstream_snapshot, "fetch_live", lambda dataset, season: empty.copy()
        )

        code = capture_live_season.main(
            [
                "--season",
                str(LIVE_SEASON),
                "--week",
                "3",
                "--dataset",
                "pbp",
                "--data-root",
                str(live_roots["data_root"]),
                "--manifest-dir",
                str(live_roots["manifest_dir"]),
            ]
        )
        assert code == capture_live_season.EXIT_OK, (
            "the live zone REFUSED an empty capture; D32-03 scopes that refusal to the "
            "sealed zone only"
        )

        manifest = upstream_live.load_live_manifest(
            LIVE_SEASON, manifest_dir=live_roots["manifest_dir"]
        )
        assert manifest is not None
        entry = upstream_live.resolve_capture(manifest, "pbp", week=3)

        assert entry["rows"] == 0
        assert len(entry["sha256"]) == 64, "an empty capture must still be digested"
        assert entry["week_digests"] == {}
        assert entry["week_partition"] == upstream_live.WEEK_PARTITION_EMPTY
        assert "empty" in entry["week_partition"]
        assert (live_roots["data_root"] / entry["path"]).is_file(), (
            "the entry names bytes that are not on disk"
        )

    def test_the_sealed_zone_still_refuses_a_zero_row_capture(
        self, live_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allowance is scoped to the LIVE zone; the sealed refusal is untouched."""
        empty = _full_pbp_frame().iloc[0:0]
        monkeypatch.setattr(
            pin_upstream_snapshot, "fetch_live", lambda dataset, season: empty.copy()
        )

        with pytest.raises(PinCaptureError) as error:
            pin_upstream_snapshot.capture_season(
                "pbp", SEALED_THROUGH_SEASON, live_roots["data_root"]
            )
        assert "ZERO rows" in str(error.value)


class TestASeasonOutsideTheUpstreamWindowRefuses:
    """The fetch boundary: a refusal must never arrive as a bare ``ValueError``."""

    @pytest.mark.parametrize("dataset", ["pbp", "schedules", "depth_charts"])
    def test_a_season_window_value_error_becomes_a_pin_family_refusal(
        self,
        dataset: str,
        live_roots: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manifest_path, before = _seed_manifest(live_roots)

        def _outside_window(requested: str, season: int) -> pd.DataFrame:
            raise ValueError("Season must be between 1999 and 2026")

        monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _outside_window)

        with pytest.raises(UpstreamSeasonWindowRefused) as error:
            _capture(live_roots, dataset=dataset)

        raised = error.value
        assert isinstance(raised, UpstreamPinError)
        assert not isinstance(raised, ValueError), (
            "the refusal is a ValueError, which is exactly what every wired call site "
            "catches and converts into an EMPTY frame -- so it would be indistinguishable "
            "from 'upstream had no data'"
        )
        assert capture_live_season.SEASON_WINDOW_HINTS[dataset] in str(raised), (
            "the refusal does not tell the operator WHEN the season becomes requestable"
        )
        assert isinstance(raised.__cause__, ValueError)
        assert manifest_path.read_bytes() == before, (
            "a refused capture moved the committed live manifest"
        )

    def test_a_transport_failure_becomes_an_upstream_pin_error_not_a_connection_error(
        self, live_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead network and a season with no data must not be the same observable."""
        manifest_path, before = _seed_manifest(live_roots)
        original = ConnectionError("nflverse release assets unreachable")

        def _no_transport(dataset: str, season: int) -> pd.DataFrame:
            raise original

        monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _no_transport)

        with pytest.raises(UpstreamPinError) as error:
            _capture(live_roots)

        raised = error.value
        assert not isinstance(raised, ConnectionError)
        assert raised.__cause__ is original
        assert manifest_path.read_bytes() == before


class TestNarrowPreservesAbsenceOnTheLivePath:
    """The branch today's live 2026 frame does NOT exercise, held by test instead."""

    def test_an_allowlisted_column_absent_upstream_stays_absent(
        self, live_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``features/team_form.py`` branches on ``"cpoe" in group.columns``."""
        upstream = _full_pbp_frame(omit=("cpoe",))
        assert "cpoe" not in upstream.columns, "the fixture did not omit the column"
        monkeypatch.setattr(
            pin_upstream_snapshot, "fetch_live", lambda dataset, season: upstream.copy()
        )

        entry = _capture(live_roots)

        written = pd.read_parquet(live_roots["data_root"] / entry["path"])
        assert "cpoe" not in written.columns, (
            "the live capture MATERIALISED an allowlisted column upstream never supplied, "
            "so the builders would compute something the live path did not give them"
        )
        assert "cpoe" not in entry["columns"]
        assert set(entry["columns"]) == set(PBP_PINNED_COLUMNS) - {"cpoe"}, (
            "preserve-absence dropped more than the absent column"
        )

    def test_an_upstream_column_outside_the_allowlist_is_dropped(
        self, live_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        upstream = _full_pbp_frame(extra=("some_new_nflverse_column",))
        monkeypatch.setattr(
            pin_upstream_snapshot, "fetch_live", lambda dataset, season: upstream.copy()
        )

        entry = _capture(live_roots)

        written = pd.read_parquet(live_roots["data_root"] / entry["path"])
        assert "some_new_nflverse_column" not in written.columns
        assert "some_new_nflverse_column" not in entry["columns"]
        assert entry["columns"] == list(PBP_PINNED_COLUMNS)
        assert entry["upstream_width"] == len(PBP_PINNED_COLUMNS) + 1, (
            "upstream_width must record the FULL width nflverse returned, or the record "
            "cannot say what was dropped"
        )
