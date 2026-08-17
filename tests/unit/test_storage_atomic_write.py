"""A crash partway through a full-table parquet rewrite must not destroy the table.

CR-02 of the Phase-29 deep code review. ``upsert_silver_composite`` is the recurring
writer for the 9,957-row odds_timeline archive that cost 7,210 real API credits, and
Plan 29-08 wires the current-week capture into the Friday orchestrator so it runs
unattended every week. Before this fix its write tail was a bare
``pq.write_table(table, silver_path)`` straight over the live path: an interrupt, a
power loss or a full disk partway through left a truncated file where the whole
archive used to be, with no second copy to recover from.

``upsert_silver`` shares the identical shape and is the writer for ``games``,
``weather``, ``odds_snapshot``, ``injuries`` and ``snap_counts``;
``ParquetManager.save``'s single-file branch is the gold-matrix writer, so an
interrupted 27-minute gold rebuild truncated gold.

These tests drive the two public upsert functions on a temporary ``base_path`` rather
than calling the helper in isolation, so what is proven is the behaviour of the real
call paths.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import storage
from data.storage import _atomic_write_parquet, upsert_silver, upsert_silver_composite
from utils.exceptions import DataIngestionError


def _games_frame(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": [2023] * len(game_ids),
            "home_score": range(len(game_ids)),
        }
    )


def _timeline_frame(game_ids: list[str], hour: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "snapshot_ts": [pd.Timestamp(f"2023-11-24 {hour:02d}:00:00", tz="UTC")]
            * len(game_ids),
            "total": [44.0] * len(game_ids),
        }
    )


class _ExplodingWrite:
    """A ``pq.write_table`` stand-in that emits partial bytes and then fails."""

    def __init__(self) -> None:
        self.path_written: Path | None = None

    def __call__(self, table, where, **kwargs):
        self.path_written = Path(where)
        Path(where).write_bytes(b"PAR1-truncated-garbage")
        raise OSError("simulated disk full partway through the write")


class TestAtomicWriteHelper:
    """The shared tail itself."""

    def test_successful_write_leaves_no_tmp_sibling(self, tmp_path: Path) -> None:
        target = tmp_path / "table.parquet"
        _atomic_write_parquet(pa.Table.from_pandas(_games_frame(["a"])), target)

        assert target.exists()
        assert not (tmp_path / "table.parquet.tmp").exists()
        assert list(pd.read_parquet(target)["game_id"]) == ["a"]

    def test_temp_file_is_a_sibling_of_the_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.replace is atomic only WITHIN a filesystem, so the temp file must
        live beside the target and never in a system temp directory."""
        target = tmp_path / "nested" / "table.parquet"
        target.parent.mkdir(parents=True)

        seen: list[Path] = []
        real_write = pq.write_table

        def _record(table, where, **kwargs):
            seen.append(Path(where))
            real_write(table, where, **kwargs)

        monkeypatch.setattr(storage.pq, "write_table", _record)
        _atomic_write_parquet(pa.Table.from_pandas(_games_frame(["a"])), target)

        assert seen == [target.parent / "table.parquet.tmp"]
        assert seen[0].parent == target.parent

    def test_failed_write_does_not_touch_the_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "table.parquet"
        _atomic_write_parquet(pa.Table.from_pandas(_games_frame(["a", "b"])), target)
        original_bytes = target.read_bytes()

        exploding = _ExplodingWrite()
        monkeypatch.setattr(storage.pq, "write_table", exploding)

        with pytest.raises(OSError, match="simulated disk full"):
            _atomic_write_parquet(pa.Table.from_pandas(_games_frame(["c"])), target)

        assert target.read_bytes() == original_bytes
        assert exploding.path_written == tmp_path / "table.parquet.tmp"


class TestUpsertSilverSurvivesAMidWriteCrash:
    """``upsert_silver`` -- the writer for games/weather/odds_snapshot/injuries/snaps."""

    def test_original_file_survives_intact_and_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        upsert_silver(_games_frame(["2023_W01_A@B"]), "games", base_path=tmp_path)
        silver = tmp_path / "silver" / "games.parquet"
        before_bytes = silver.read_bytes()
        before_frame = pd.read_parquet(silver)

        monkeypatch.setattr(storage.pq, "write_table", _ExplodingWrite())
        with pytest.raises(OSError, match="simulated disk full"):
            upsert_silver(_games_frame(["2023_W02_C@D"]), "games", base_path=tmp_path)

        assert silver.read_bytes() == before_bytes
        pd.testing.assert_frame_equal(pd.read_parquet(silver), before_frame)

    def test_successful_upsert_leaves_no_tmp_sibling(self, tmp_path: Path) -> None:
        upsert_silver(_games_frame(["2023_W01_A@B"]), "games", base_path=tmp_path)
        upsert_silver(_games_frame(["2023_W02_C@D"]), "games", base_path=tmp_path)

        silver_dir = tmp_path / "silver"
        assert not list(silver_dir.glob("*.tmp"))
        assert sorted(pd.read_parquet(silver_dir / "games.parquet")["game_id"]) == [
            "2023_W01_A@B",
            "2023_W02_C@D",
        ]


class TestUpsertSilverCompositeSurvivesAMidWriteCrash:
    """``upsert_silver_composite`` -- the paid odds_timeline archive's writer (CR-02)."""

    def test_original_archive_survives_intact_and_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        upsert_silver_composite(
            _timeline_frame(["2023_W12_MIA@NYJ"]), "odds_timeline", base_path=tmp_path
        )
        silver = tmp_path / "silver" / "odds_timeline.parquet"
        before_bytes = silver.read_bytes()
        before_frame = pd.read_parquet(silver)

        monkeypatch.setattr(storage.pq, "write_table", _ExplodingWrite())
        with pytest.raises(OSError, match="simulated disk full"):
            upsert_silver_composite(
                _timeline_frame(["2023_W13_LV@KC"], hour=13),
                "odds_timeline",
                base_path=tmp_path,
            )

        assert silver.read_bytes() == before_bytes
        pd.testing.assert_frame_equal(pd.read_parquet(silver), before_frame)

    def test_successful_write_leaves_no_tmp_sibling(self, tmp_path: Path) -> None:
        upsert_silver_composite(
            _timeline_frame(["2023_W12_MIA@NYJ"]), "odds_timeline", base_path=tmp_path
        )
        upsert_silver_composite(
            _timeline_frame(["2023_W13_LV@KC"], hour=13),
            "odds_timeline",
            base_path=tmp_path,
        )

        assert not list((tmp_path / "silver").glob("*.tmp"))

    def test_composite_dedupe_and_idempotency_are_unchanged(
        self, tmp_path: Path
    ) -> None:
        """The atomic tail must not disturb the (game_id, snapshot_ts) semantics."""
        frame = _timeline_frame(["2023_W12_MIA@NYJ", "2023_W13_LV@KC"])
        upsert_silver_composite(frame, "odds_timeline", base_path=tmp_path)
        upsert_silver_composite(frame, "odds_timeline", base_path=tmp_path)

        stored = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
        assert len(stored) == 2
        assert len(stored.drop_duplicates(subset=["game_id", "snapshot_ts"])) == 2

        # A distinct snapshot_ts for the same game_id coexists rather than clobbering.
        upsert_silver_composite(
            _timeline_frame(["2023_W12_MIA@NYJ"], hour=17),
            "odds_timeline",
            base_path=tmp_path,
        )
        stored = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
        assert len(stored) == 3


class TestParquetManagerSaveIsAtomic:
    """The gold-matrix writer -- an interrupted rebuild must not truncate gold."""

    def test_existing_gold_survives_a_failed_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pm = storage.ParquetManager(str(tmp_path))
        pm.save(_games_frame(["a", "b"]), "gold/features_wp.parquet")
        target = tmp_path / "gold" / "features_wp.parquet"
        before_bytes = target.read_bytes()

        monkeypatch.setattr(storage.pq, "write_table", _ExplodingWrite())
        with pytest.raises(DataIngestionError, match="Parquet save failed"):
            pm.save(_games_frame(["c"]), "gold/features_wp.parquet")

        assert target.read_bytes() == before_bytes


def test_partitioned_writes_are_untouched(tmp_path: Path) -> None:
    """Only the single-file branch changed; ``write_to_dataset`` is unaffected."""
    pm = storage.ParquetManager(str(tmp_path))
    pm.save(
        _games_frame(["a", "b"]),
        "silver/partitioned/games.parquet",
        partition_cols=["season"],
    )
    assert list((tmp_path / "silver" / "partitioned").glob("season=*"))


def test_bronze_snapshot_is_deliberately_not_atomic() -> None:
    """``save_bronze_snapshot`` writes a NEW timestamped file per call.

    There is no complete previous file for a partial write to destroy, so the
    atomic tail buys nothing there. Pinned so a future "consistency" sweep does
    not convert it and then claim the conversion mattered.
    """
    source = Path("data/storage.py").read_text(encoding="utf-8")
    bronze_body = source.split("def save_bronze_snapshot(")[1].split(
        "def upsert_silver("
    )[0]
    assert "Deliberately NOT _atomic_write_parquet" in bronze_body
    assert "pq.write_table(table, filepath" in bronze_body
    assert "_atomic_write_parquet(table" not in bronze_body
