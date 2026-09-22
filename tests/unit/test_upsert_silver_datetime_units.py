"""An upsert never stores a datetime column as text (found and fixed by Plan 33.2-15).

``data.storage.upsert_silver`` concatenates the table it reads back from parquet with the new
rows. A tz-aware column read back is ``datetime64[us, UTC]`` under a pytz zone; the same column
built fresh is ``datetime64[ns, UTC]`` under ``datetime.timezone.utc``. pandas could not unify
that pair, the concat fell back to ``object``, and the parquet normalizer then wrote every value
as a STRING -- measured on 2026-09-22, when the first 2025 injury upsert turned all 76,767 stored
``date_modified`` instants into text (repaired from the bronze capture the same day).

Asserted here, each in a temporary data root (never production):

* the exact failing pair now round-trips as a datetime with every instant unchanged;
* a table whose columns already agree is written exactly as before (no unit change);
* a column that would still come out ``object`` is REFUSED by name, not stringified;
* replacing every stored row writes the new frame as the table.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.storage import SilverDatetimeDriftError, upsert_silver


def _seed(root: Path, frame: pd.DataFrame, table: str = "t") -> Path:
    (root / "silver").mkdir(parents=True, exist_ok=True)
    path = root / "silver" / f"{table}.parquet"
    frame.to_parquet(path)
    return path


def _stored_type(path: Path, column: str) -> str:
    return str(pq.read_schema(path).field(column).type)


class TestTheFailingPairRoundTrips:
    def test_a_microsecond_stored_column_plus_fresh_nanosecond_nulls_stays_a_datetime(
        self, tmp_path: Path
    ) -> None:
        stored = pd.to_datetime(["2024-09-06 19:05:30.123456"], utc=True).as_unit("us")
        path = _seed(
            tmp_path, pd.DataFrame({"game_id": ["g1"], "date_modified": stored})
        )
        new = pd.DataFrame(
            {
                "game_id": ["g2"],
                "date_modified": pd.to_datetime(
                    pd.Series([None], dtype=object), utc=True
                ),
            }
        )
        assert (
            pd.concat([pd.read_parquet(path), new], ignore_index=True)[
                "date_modified"
            ].dtype
            == object
        ), "non-vacuity: the raw concat still drifts"

        upsert_silver(new, "t", base_path=tmp_path)

        assert _stored_type(path, "date_modified").startswith("timestamp")
        back = pd.read_parquet(path)["date_modified"]
        assert back.iloc[0] == pd.Timestamp("2024-09-06 19:05:30.123456", tz="UTC")
        assert pd.isna(back.iloc[1])

    def test_a_fresh_value_joins_a_stored_column_as_a_datetime(
        self, tmp_path: Path
    ) -> None:
        stored = pd.to_datetime(["2024-09-06 19:05:30"], utc=True).as_unit("us")
        path = _seed(tmp_path, pd.DataFrame({"game_id": ["g1"], "d": stored}))
        fresh = pd.DataFrame(
            {"game_id": ["g2"], "d": pd.to_datetime(["2026-01-01 00:00"], utc=True)}
        )
        upsert_silver(fresh, "t", base_path=tmp_path)
        back = pd.read_parquet(path)["d"]
        assert list(back) == [
            pd.Timestamp("2024-09-06 19:05:30", tz="UTC"),
            pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        ]


class TestAnAgreeingTableIsUntouched:
    def test_matching_dtypes_keep_their_unit(self, tmp_path: Path) -> None:
        stored = pd.to_datetime(["2024-09-06 19:05:30"], utc=True).as_unit("us")
        path = _seed(tmp_path, pd.DataFrame({"game_id": ["g1"], "d": stored}))
        same = pd.DataFrame(
            {
                "game_id": ["g2"],
                "d": pd.read_parquet(path)["d"].iloc[[0]].reset_index(drop=True),
            }
        )
        upsert_silver(same, "t", base_path=tmp_path)
        assert _stored_type(path, "d") == "timestamp[us, tz=UTC]"


class TestAColumnThatWouldBeTextIsRefused:
    def test_a_stored_string_column_meeting_a_datetime_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = _seed(
            tmp_path,
            pd.DataFrame({"game_id": ["g1"], "d": ["2024-09-06 19:05:30.000000Z"]}),
        )
        before = path.read_bytes()
        new = pd.DataFrame(
            {"game_id": ["g2"], "d": pd.to_datetime(["2026-01-01"], utc=True)}
        )
        with pytest.raises(SilverDatetimeDriftError, match="'d'"):
            upsert_silver(new, "t", base_path=tmp_path)
        assert path.read_bytes() == before

    def test_the_refusal_dodges_the_degrading_catch_tuples(self) -> None:
        assert SilverDatetimeDriftError.__bases__ == (Exception,)


class TestReplacingEveryRow:
    def test_the_new_frame_becomes_the_table(self, tmp_path: Path) -> None:
        path = _seed(
            tmp_path,
            pd.DataFrame({"game_id": ["g1"], "d": ["2024-09-06 19:05:30.000000Z"]}),
        )
        replacement = pd.DataFrame(
            {"game_id": ["g1"], "d": pd.to_datetime(["2024-09-06 19:05:30"], utc=True)}
        )
        upsert_silver(replacement, "t", base_path=tmp_path)
        assert _stored_type(path, "d").startswith("timestamp")
        assert len(pd.read_parquet(path)) == 1
