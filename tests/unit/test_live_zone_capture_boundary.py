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
from data.upstream_pin import LIVE_ZONE_FIRST_SEASON, PBP_PINNED_COLUMNS
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
