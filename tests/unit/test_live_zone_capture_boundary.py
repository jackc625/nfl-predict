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
        digests = upstream_live.week_digests(frame)

        assert list(digests) == [upstream_live.NO_WEEK_COLUMN_BUCKET]
        assert digests[upstream_live.NO_WEEK_COLUMN_BUCKET]["rows"] == len(frame)

        entry = _entry_for(tmp_path, frame, dataset="depth_charts")
        assert entry["week_partition"] == upstream_live.WEEK_PARTITION_WHOLE_FRAME
        assert "no week column" in entry["week_partition"]
        assert entry["week_digests"] == digests

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
        assert set(entry["week_digests"]["1"]) == {"rows", "frame_sha256", "columns"}
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
