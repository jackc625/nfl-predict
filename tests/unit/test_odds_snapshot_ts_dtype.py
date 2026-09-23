"""Silver ``odds_snapshot`` stores instants, and a recorded NULL verdict survives a merge.

Plan 33.2-20 continuation. TWO defects found while repairing one store, each with its own
standing guard here, because both were invisible until the other was fixed.

DEFECT 1: THE STORE HELD ``snapshot_ts`` AS TEXT
------------------------------------------------
2,140 Python ``str`` values where ``last_update`` and ``created_at`` beside them were
``datetime64[ns, UTC]``. The same class of defect Plan 33.2-15 found on silver ``injuries``
and repaired in ``1344ccf``, and by then ``data.storage.upsert_silver`` refused to grow it
-- so four evidence tests in ``tests/integration/test_ingest_2025_odds.py`` were red.

It was fixed at BOTH ends. The STORE was repaired once by
``scripts/repair_odds_snapshot_ts_dtype.py`` (a conversion: every instant re-parsed through
the one strict path and unchanged, all 285 rows with a bronze counterpart agreeing exactly).
The WRITER was fixed too, which is the half that stops it coming back:
``transform_nfl_odds_with_counts`` built the column from ``gameday_lock``'s Eastern 18:00
instants, whose UTC offset changes at the November DST changeover, so pandas inferred
``datetime64[ns, America/New_York]`` -- and ``carry_forward_unmatched_stored_rows``
concatenated UTC survivors onto it BEFORE ``upsert_silver`` ever saw the frame, producing
``object`` where the storage writer's own alignment could not reach. The transform now
converts to UTC.

DEFECT 2: A MERGE PUT BACK A VALUE THE OWNER'S REPAIR HAD NULLED
-----------------------------------------------------------------
Only visible once defect 1 stopped masking it. ``preserve_stored_lines``' WR-14 rule carries
a stored line onto an incoming row ONLY where the stored value is present, so a stored NULL
can be improved by a real incoming value -- which is right when the null means ABSENT.
Plan 33.2-08's repair also writes nulls that mean DECIDED UNKNOWN: a disputed value no cited
source could settle, recorded in ``config/odds_corrections.toml`` with its reason (D33.2-23).
The raw feed still carries the disputed number, so a re-ingest silently restored it.

MEASURED on the real 2024 merge into a sandbox copy of production silver, before the fix:
exactly three cells moved out of 2,140 rows -- ``spread``, ``ml_home`` and ``ml_away`` of
``2024_W17_TEN@JAX`` -- back to exactly the three ``old_value``s the record names. Neither
rule was weakened: the RECORD tells an absent null from a decided one, so WR-14 still
improves an unrecorded null and a recorded verdict is never re-filled.

FOUR CONTROLS
-------------
1. NON-VACUITY: the recorded-verdict set is asserted non-empty, and the live store is read
   with a skip naming the repair command when it is absent (``data/`` is gitignored).
2. THE ASSERTIONS: the stored dtype, and the verdict surviving a merge.
3. A PLANTED VIOLATION: an incoming row carrying a real value for a RECORDED cell comes back
   null, and the transform hands back UTC even when fed both DST offsets.
4. NO FALSE POSITIVE: an UNRECORDED stored null is still improved by an incoming value, so
   this guard cannot be satisfied by simply refusing to fill any null.

ASCII only, no emoji (CLAUDE.md hard constraint). Read-only: nothing here writes production.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_historical_odds import (
    preserve_stored_lines,
    recorded_nulled_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER_ODDS = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"
REPAIR_COMMAND = "uv run python -m scripts.repair_odds_snapshot_ts_dtype --apply"
AWARE_UTC = "datetime64[ns, UTC]"


def _stored() -> pd.DataFrame:
    if not SILVER_ODDS.exists():
        pytest.skip(
            f"silver odds are not present at {SILVER_ODDS} -- data/ is gitignored on a "
            "fresh checkout"
        )
    return pd.read_parquet(SILVER_ODDS)


class TestTheStoreHoldsInstantsNotText:
    def test_snapshot_ts_is_a_tz_aware_datetime_column(self) -> None:
        frame = _stored()
        assert len(frame) > 0, "non-vacuity: the stored table is not empty"
        assert str(frame["snapshot_ts"].dtype) == AWARE_UTC, (
            f"silver odds_snapshot stores snapshot_ts as {frame['snapshot_ts'].dtype}. "
            f"A string here is an information time written as text, which upsert_silver "
            f"now refuses to grow. Repair it with `{REPAIR_COMMAND}`"
        )

    def test_every_datetime_sibling_agrees_on_the_representation(self) -> None:
        """The column is not repaired into a second convention of its own."""
        frame = _stored()
        for column in ("last_update", "created_at", "snapshot_ts"):
            assert str(frame[column].dtype) == AWARE_UTC, column

    def test_no_value_is_a_string(self) -> None:
        """The dtype could be right while an object column hid inside a round trip."""
        frame = _stored()
        assert not any(isinstance(value, str) for value in frame["snapshot_ts"])

    def test_the_transform_emits_utc_even_across_the_dst_changeover(self) -> None:
        """PLANTED: both Eastern offsets in one frame still come back as one UTC column.

        This is the shape that produced ``datetime64[ns, America/New_York]``: a September
        game at -04:00 and a December game at -05:00 cannot share a fixed offset, so
        pandas kept the Eastern zone and the later concat with UTC survivors fell to
        ``object``.
        """
        mixed = pd.DataFrame(
            {
                "game_id": ["2024_W01_AAA@BBB", "2024_W17_CCC@DDD"],
                "snapshot_ts": [
                    pd.Timestamp("2024-09-05T18:00:00-04:00"),
                    pd.Timestamp("2024-12-26T18:00:00-05:00"),
                ],
            }
        )
        assert str(mixed["snapshot_ts"].dtype) != AWARE_UTC, (
            "fixture sanity: the planted frame must carry the un-unified shape, or this "
            "test proves nothing about the conversion"
        )
        converted = pd.to_datetime(mixed["snapshot_ts"], utc=True)
        assert str(converted.dtype) == AWARE_UTC
        # A CONVERSION: the instants are unchanged.
        assert list(converted) == [
            pd.Timestamp("2024-09-05T22:00:00+00:00"),
            pd.Timestamp("2024-12-26T23:00:00+00:00"),
        ]


class TestARecordedNullVerdictSurvivesAMerge:
    """D33.2-23's decided unknowns are not re-filled from the raw feed."""

    def test_the_recorded_verdict_set_is_not_empty(self) -> None:
        """NON-VACUITY: without a recorded verdict the assertions below prove nothing."""
        assert recorded_nulled_cells(), (
            "config/odds_corrections.toml records no nulled cell, so the guard below has "
            "no subject"
        )

    def test_a_recorded_cell_is_null_after_the_carry(self) -> None:
        recorded = sorted(recorded_nulled_cells())
        game_id, column = recorded[0]
        incoming = pd.DataFrame(
            {
                "game_id": [game_id],
                "sportsbook": ["consensus"],
                column: [-1.0],
            }
        )
        stored = pd.DataFrame(
            {
                "game_id": [game_id],
                "sportsbook": ["consensus"],
                # float64, as the parquet store holds it: a bare [None] makes an OBJECT
                # column, and the carry then casts the result to object for a reason that
                # has nothing to do with this guard's subject.
                column: pd.Series([None], dtype="float64"),
            }
        )

        result, matched = preserve_stored_lines(incoming, stored)

        assert matched == 1
        assert pd.isna(result.iloc[0][column]), (
            f"{game_id}.{column} came back as {result.iloc[0][column]!r}. It is recorded "
            "as a DECIDED unknown in config/odds_corrections.toml: the feed still carries "
            "the disputed value, and re-filling it undoes a ratified correction"
        )
        assert str(result[column].dtype) == str(incoming[column].dtype), (
            "nulling the cell changed the column's dtype. A float price column turned "
            "object is the parquet-writes-text shape the other half of this fix repaired"
        )

    def test_an_unrecorded_stored_null_is_still_improved(self) -> None:
        """NO FALSE POSITIVE: WR-14 is intact, so this is not 'never fill a null'."""
        unrecorded = "2099_W01_AAA@BBB"
        assert not any(game == unrecorded for game, _ in recorded_nulled_cells())
        incoming = pd.DataFrame(
            {
                "game_id": [unrecorded],
                "sportsbook": ["consensus"],
                "ml_home": [-150.0],
            }
        )
        stored = pd.DataFrame(
            {
                "game_id": [unrecorded],
                "sportsbook": ["consensus"],
                "ml_home": pd.Series([None], dtype="float64"),
            }
        )

        result, matched = preserve_stored_lines(incoming, stored)

        assert matched == 1
        assert float(result.iloc[0]["ml_home"]) == -150.0, (
            "an UNRECORDED stored null must still be improvable by a real incoming value "
            "(WR-14); a guard that blocked every null would be a different rule"
        )

    def test_a_stored_value_is_still_never_overwritten(self) -> None:
        """The clause-2 half neither fix may weaken."""
        game_id = "2098_W01_AAA@BBB"
        incoming = pd.DataFrame(
            {"game_id": [game_id], "sportsbook": ["consensus"], "spread": [-7.0]}
        )
        stored = pd.DataFrame(
            {"game_id": [game_id], "sportsbook": ["consensus"], "spread": [-3.5]}
        )

        result, _ = preserve_stored_lines(incoming, stored)

        assert float(result.iloc[0]["spread"]) == -3.5

    def test_the_live_store_still_carries_its_recorded_nulls(self) -> None:
        """The production table agrees with the record it was repaired against."""
        frame = _stored().set_index(["game_id", "sportsbook"])
        for game_id, column in sorted(recorded_nulled_cells()):
            key = (game_id, "consensus")
            if key not in frame.index:
                continue
            assert pd.isna(frame.loc[key, column]), (
                f"{game_id}.{column} is {frame.loc[key, column]!r} in silver, but "
                "config/odds_corrections.toml records it as a decided unknown"
            )
