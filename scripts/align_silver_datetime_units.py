"""One-shot, declared-write alignment of a silver table's tz-aware datetime UNITS.

Plan 33.2-20 continuation (deferred-items entry found by Plan 33.2-16, not caused by it).

WHY THIS EXISTS
---------------
``data/silver/weather.parquet`` stores ``created_at`` as ``datetime64[us, UTC]`` while
its three siblings in the SAME table -- ``forecast_time``, ``game_time`` and
``forecast_issue_time`` -- are ``datetime64[ns, UTC]``. The table was written before
Plan 33.2-15's ``1344ccf``, which made ``data.storage.upsert_silver`` align a mismatched
tz-aware pair to ``datetime64[ns, UTC]`` before concatenating. So an offline regeneration
of silver weather produces ``ns`` and the comparison against production fails on DTYPE,
before a single VALUE is compared
(``tests/integration/test_weather_rebuild_offline.py::...::test_regeneration_is_offline_idempotent_and_reproduces_production``).

WHY THE STORE MOVES AND NOT THE COMPARISON
-------------------------------------------
Two routes were open. Making the WRITER preserve whatever unit happens to be stored
would fix the comparison, but it would make the store's representation a function of its
own history: the same table would be ``us`` or ``ns`` depending on when it was last
written, and a later reader could not say which without looking. Aligning the STORE
gives the column one representation -- the one ``upsert_silver`` already converges on,
and the one its three siblings already carry -- so the writer's rule and the store agree.
Relaxing the comparison was never an option: it is the only thing asserting that an
offline regeneration reproduces production.

A CONVERSION, NEVER A RELABEL (D33.2-01). ``us`` to ``ns`` is a widening of the same
tz-aware instants: every value is unchanged and none can overflow (pandas' ``ns`` range
covers 1677-2262). Both sides are compared AS INSTANTS after the change, not as bytes.

IDEMPOTENT
----------
A column already at the target unit is counted and left alone; a naive datetime column
is REFUSED by name rather than localized, because guessing a zone is the relabel D33.2-01
forbids. If nothing needs aligning, nothing is written.

WRITES
------
``--dry-run`` (the default) writes nothing. ``--apply`` writes exactly
``silver/<table>.parquet`` (one atomic full-table write through ``upsert_silver``, every
row in its original position) and, through that writer, the DuckDB copy IF the table has
one. The post-state is asserted on the frame before it is written and on the table read
back.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.storage import upsert_silver

#: The ONE representation a silver tz-aware datetime column is stored in. Not invented
#: here: it is the unit ``data.storage._align_aware_datetime_columns`` converges a
#: mismatched pair to, added by Plan 33.2-15's ``1344ccf``.
TARGET_DTYPE = "datetime64[ns, UTC]"


class SilverDatetimeUnitRepairError(AssertionError):
    """The repaired table is not exactly the declared change. Nothing is trusted."""


@dataclass(frozen=True)
class UnitAlignmentPlan:
    """What a run would do: the repaired frame and the per-column disposition."""

    before: pd.DataFrame
    after: pd.DataFrame
    aligned: tuple[str, ...]
    already_aligned: tuple[str, ...]


def load_table(data_root: Path, table: str) -> pd.DataFrame:
    """The silver parquet for *table* under *data_root*, in its stored row order."""
    return pd.read_parquet(data_root / "silver" / f"{table}.parquet")


def plan_alignment(frame: pd.DataFrame) -> UnitAlignmentPlan:
    """Convert every tz-aware datetime column that is not already at the target unit.

    Raises:
        SilverDatetimeUnitRepairError: a NAIVE datetime column is present. Localizing it
            would be a relabel, which D33.2-01 forbids; it is surfaced instead.
    """
    naive = sorted(
        column
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
        and not isinstance(frame[column].dtype, pd.DatetimeTZDtype)
    )
    if naive:
        raise SilverDatetimeUnitRepairError(
            f"naive datetime column(s) {naive} carry no zone. Giving them one here would "
            "be a relabel, not a conversion (D33.2-01). Fix them at the writer."
        )

    after = frame.copy()
    aligned: list[str] = []
    already: list[str] = []
    for column in frame.columns:
        dtype = frame[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype):
            continue
        if str(dtype) == TARGET_DTYPE:
            already.append(column)
            continue
        after[column] = frame[column].dt.tz_convert("UTC").astype(TARGET_DTYPE)
        aligned.append(column)
    return UnitAlignmentPlan(frame, after, tuple(aligned), tuple(already))


def assert_post_state(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Exactly the declared change, or refuse.

    * same rows, same order, same columns;
    * no NON-datetime column moved at all, dtype included;
    * every tz-aware datetime column is at the target unit;
    * every datetime INSTANT is unchanged -- compared after converting BOTH sides to the
      target unit, which is what makes this a conversion rather than an edit.
    """
    if list(before.columns) != list(after.columns) or len(before) != len(after):
        raise SilverDatetimeUnitRepairError("the table's shape changed")

    for column in before.columns:
        left, right = before[column], after[column]
        if isinstance(left.dtype, pd.DatetimeTZDtype):
            if str(right.dtype) != TARGET_DTYPE:
                raise SilverDatetimeUnitRepairError(
                    f"column {column!r} is {right.dtype}, not {TARGET_DTYPE}"
                )
            moved = (left.dt.tz_convert("UTC").astype(TARGET_DTYPE) != right) & ~(
                left.isna() & right.isna()
            )
            if bool(moved.any()):
                raise SilverDatetimeUnitRepairError(
                    f"{int(moved.sum())} instant(s) in column {column!r} moved. This "
                    "repair is a unit conversion and moves no value."
                )
            if int(left.isna().sum()) != int(right.isna().sum()):
                raise SilverDatetimeUnitRepairError(
                    f"column {column!r} changed its null count"
                )
            continue
        if left.dtype != right.dtype:
            raise SilverDatetimeUnitRepairError(f"column {column!r} changed dtype")
        if not left.equals(right):
            raise SilverDatetimeUnitRepairError(
                f"column {column!r} moved; only a datetime UNIT may change"
            )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align a silver table's tz-aware datetime columns to datetime64[ns, UTC]."
        )
    )
    parser.add_argument(
        "--table",
        required=True,
        help="the silver table name, e.g. weather",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="perform the declared write")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    before = load_table(args.data_root, args.table)
    plan = plan_alignment(before)
    assert_post_state(before, plan.after)

    print(f"TABLE= {args.table}")
    print(f"ROWS= {len(before)}")
    print(f"ALIGNED= {list(plan.aligned)}")
    print(f"ALREADY_ALIGNED= {list(plan.already_aligned)}")
    if not args.apply:
        print("MODE= dry-run (nothing written)")
        return 0

    if plan.aligned:
        upsert_silver(
            plan.after.reset_index(drop=True), args.table, base_path=args.data_root
        )
        stored = load_table(args.data_root, args.table)
        assert_post_state(before, stored)
        print(f"ROWS_WRITTEN= {len(stored)} (read back and re-asserted)")
    print("MODE= apply")
    return 0


if __name__ == "__main__":
    sys.exit(run())
