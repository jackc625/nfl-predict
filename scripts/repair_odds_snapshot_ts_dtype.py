"""One-shot, declared-write repair of silver ``odds_snapshot``'s TEXT ``snapshot_ts``.

Plan 33.2-20 continuation (deferred-items entry found by Plan 33.2-20 Task 3, not caused
by it). The SAME CLASS of defect Plan 33.2-15 found and repaired for silver ``injuries``
in ``1344ccf``: a store holding instants as STRINGS, and a writer that now correctly
refuses to grow it.

WHY THIS EXISTS
---------------
``data/silver/odds_snapshot.parquet`` stores ``snapshot_ts`` as ``object`` -- 2,140 Python
``str`` values -- while its ``last_update`` and ``created_at`` siblings are
``datetime64[ns, UTC]``. Since ``1344ccf``, ``data.storage.upsert_silver`` refuses to
concatenate a fresh datetime column onto a stored object one rather than writing text, so
ANY partial upsert of this table now raises ``SilverDatetimeDriftError``. Four evidence
tests in ``tests/integration/test_ingest_2025_odds.py`` do exactly that against a
temporary copy and have been red because of it. The writer is right; the STORE is stale.

TWO WRITE PATHS PUT THE STRINGS THERE, and both are already fixed upstream:

1. the 2018-2024 historical ingest wrote an ISO-8601 STRING with an Eastern offset
   (``2018-09-19T18:00:00-04:00``). ``scripts/ingest_historical_odds.py`` now stamps a
   tz-aware ``datetime`` per game through ``gameday_lock`` (clause 3, D31-37);
2. the 2025 backfill of 2026-09-05 handed ``upsert_silver`` real
   ``datetime64[ns, UTC]`` values, which pandas could not unify with the stored object
   column, so ``pd.concat`` fell back to ``object`` and every fresh instant was
   stringified into ``2025-08-29 22:00:00+00:00``. That is the ``1344ccf`` defect exactly,
   and ``upsert_silver`` now refuses it instead of repeating it.

So nothing in the code writes a string here any more, and this script does not change any
code path. It repairs the residue ONE TIME.

A CONVERSION, NEVER A RELABEL (D33.2-01)
----------------------------------------
Every stored string carries an EXPLICIT UTC offset, so each names an unambiguous instant.
The repair parses each value through the module that owns the rule --
``scripts.ingest_historical_odds.require_aware_snapshot_ts``, the strict half of the ONE
parse path -- and stores the same instant as ``datetime64[ns, UTC]``. A 2018 value reading
``2018-09-19T18:00:00-04:00`` is stored as ``2018-09-19 22:00:00+00:00``: the same moment,
written in the column's own unit. No naive value is relabelled and no instant moves.

THE VALUES ARE STILL MANUFACTURED, AND THIS DOES NOT CHANGE THAT. The owner ruled on
2026-09-22 (Plan 33.2-14's market-odds checkpoint) that a stored line counts only with a
genuinely recorded capture time in ``created_at``; ``snapshot_ts`` is a LABEL the ingest
stamps, is never read as an information time, and no fence in this phase treats it as
evidence. Repairing its dtype makes the label storable; it does not promote it.

IDEMPOTENT
----------
A value already ``datetime64[ns, UTC]`` is counted and left alone; a string is converted;
a null or an unparseable value raises through the one parse path and NOTHING is written.
``CONVERTED=`` and ``ALREADY_AWARE=`` sum to the stored row count.

WRITES
------
``--dry-run`` (the default) writes nothing. ``--apply`` writes exactly
``silver/odds_snapshot.parquet`` (one atomic full-table write through ``upsert_silver``,
every row in its original position) and, through that writer, the DuckDB
``odds_snapshot`` copy. Only ``snapshot_ts``'s DTYPE changes; the post-state is asserted
on the frame before it is written and on the table read back, and the instants are
asserted equal row-for-row in both directions.

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
from scripts.ingest_historical_odds import require_aware_snapshot_ts

ODDS_TABLE = "odds_snapshot"
SNAPSHOT_COLUMN = "snapshot_ts"
TARGET_DTYPE = "datetime64[ns, UTC]"


class OddsSnapshotTsRepairError(AssertionError):
    """The repaired table is not exactly the declared change. Nothing is trusted."""


@dataclass(frozen=True)
class SnapshotTsRepairPlan:
    """What a run would do: the repaired frame and the per-row disposition."""

    before: pd.DataFrame
    after: pd.DataFrame
    converted: int
    already_aware: int


def load_odds(data_root: Path) -> pd.DataFrame:
    """The silver ``odds_snapshot`` parquet under *data_root*, in its stored row order."""
    return pd.read_parquet(data_root / "silver" / f"{ODDS_TABLE}.parquet")


def plan_repair(odds: pd.DataFrame) -> SnapshotTsRepairPlan:
    """Parse every stored ``snapshot_ts`` through the one strict path.

    Raises:
        OddsSnapshotTsRepairError: the column is absent.
        ValueError: a value is null, empty or unparseable -- raised by
            ``require_aware_snapshot_ts`` itself, so the refusal wording is the one rule's.
    """
    if SNAPSHOT_COLUMN not in odds.columns:
        raise OddsSnapshotTsRepairError(
            f"silver {ODDS_TABLE} carries no {SNAPSHOT_COLUMN!r} column"
        )
    stored = odds[SNAPSHOT_COLUMN]
    already = int(sum(isinstance(value, pd.Timestamp) for value in stored))
    parsed = pd.to_datetime(
        pd.Series(
            [require_aware_snapshot_ts(value) for value in stored], index=odds.index
        ),
        utc=True,
    )
    after = odds.copy()
    after[SNAPSHOT_COLUMN] = parsed.astype(TARGET_DTYPE)
    return SnapshotTsRepairPlan(odds, after, len(odds) - already, already)


def assert_post_state(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Exactly the declared change, or refuse.

    * same rows, same order, same columns;
    * no column other than ``snapshot_ts`` moved on any row, dtype included;
    * ``snapshot_ts`` is ``datetime64[ns, UTC]`` with no nulls;
    * every row's instant is UNCHANGED -- both sides re-parsed through the one strict
      path and compared row-for-row, which is what makes this a conversion and not an
      edit. A string-vs-timestamp comparison would be meaningless, so neither side is
      compared raw.
    """
    if list(before.columns) != list(after.columns) or len(before) != len(after):
        raise OddsSnapshotTsRepairError("the table's shape changed")
    for key in ("game_id", "sportsbook"):
        if key in before.columns and list(before[key]) != list(after[key]):
            raise OddsSnapshotTsRepairError(f"the table's {key} order changed")
    for column in before.columns:
        if column == SNAPSHOT_COLUMN:
            continue
        if before[column].dtype != after[column].dtype:
            raise OddsSnapshotTsRepairError(f"column {column!r} changed dtype")
        if not before[column].equals(after[column]):
            raise OddsSnapshotTsRepairError(
                f"column {column!r} moved; only {SNAPSHOT_COLUMN!r} may change"
            )

    if str(after[SNAPSHOT_COLUMN].dtype) != TARGET_DTYPE:
        raise OddsSnapshotTsRepairError(
            f"{SNAPSHOT_COLUMN!r} is {after[SNAPSHOT_COLUMN].dtype}, not {TARGET_DTYPE}"
        )
    if int(after[SNAPSHOT_COLUMN].isna().sum()):
        raise OddsSnapshotTsRepairError(f"{SNAPSHOT_COLUMN!r} gained a null")

    left = [require_aware_snapshot_ts(v) for v in before[SNAPSHOT_COLUMN]]
    right = [require_aware_snapshot_ts(v) for v in after[SNAPSHOT_COLUMN]]
    if left != right:
        moved = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
        raise OddsSnapshotTsRepairError(
            f"{len(moved)} instant(s) moved, e.g. row {moved[0]}: "
            f"{left[moved[0]]} -> {right[moved[0]]}. This repair is a conversion."
        )


def bronze_agreement(data_root: Path, after: pd.DataFrame) -> tuple[int, int]:
    """``(matched, agreeing)`` rows of *after* against every bronze odds capture.

    The 33.2-15 discipline: a repaired store is checked against the capture it came from,
    not only against itself. Only the ``odds_raw_bronze_`` captures are read (the
    ``odds_timeline_raw_bronze_`` family is a different table). Rows with no bronze
    counterpart -- the 2018-2024 schedule-derived lines, which were never captured -- are
    simply not matched, so this is a check where evidence exists and is silent where none
    does.
    """
    captures = sorted(
        path
        for path in (data_root / "bronze").glob("odds_raw_bronze_*.parquet")
        if not path.name.startswith("odds_timeline_")
    )
    if not captures:
        return 0, 0
    frames = []
    for path in captures:
        frame = pd.read_parquet(path)
        if {"game_id", "sportsbook", SNAPSHOT_COLUMN} <= set(frame.columns):
            frames.append(frame[["game_id", "sportsbook", SNAPSHOT_COLUMN]])
    if not frames:
        return 0, 0
    bronze = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["game_id", "sportsbook"], keep="last"
    )
    bronze[SNAPSHOT_COLUMN] = pd.to_datetime(bronze[SNAPSHOT_COLUMN], utc=True)
    merged = after[["game_id", "sportsbook", SNAPSHOT_COLUMN]].merge(
        bronze.rename(columns={SNAPSHOT_COLUMN: "_bronze"}),
        on=["game_id", "sportsbook"],
        how="inner",
    )
    agreeing = int((merged[SNAPSHOT_COLUMN] == merged["_bronze"]).sum())
    return len(merged), agreeing


def write_odds_table(data_root: Path, repaired: pd.DataFrame) -> Path:
    """One atomic full-table write.

    Every stored ``game_id`` is in *repaired*, so ``upsert_silver``'s latest-wins key
    replacement empties the stored side and writes this frame AS the table -- the path
    ``1344ccf`` added for exactly this case, which is also why the drift refusal does not
    fire on a full replacement.
    """
    return upsert_silver(
        repaired.reset_index(drop=True), ODDS_TABLE, base_path=data_root
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair silver odds_snapshot's TEXT snapshot_ts to datetime64[ns, UTC]."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="perform the declared write")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    before = load_odds(args.data_root)
    plan = plan_repair(before)
    assert_post_state(before, plan.after)
    matched, agreeing = bronze_agreement(args.data_root, plan.after)

    print(f"STORED_ROWS= {len(before)}")
    print(f"STORED_DTYPE= {before[SNAPSHOT_COLUMN].dtype}")
    print(f"CONVERTED= {plan.converted}")
    print(f"ALREADY_AWARE= {plan.already_aware}")
    print(f"REPAIRED_DTYPE= {plan.after[SNAPSHOT_COLUMN].dtype}")
    print(f"DISTINCT_INSTANTS_BEFORE= {before[SNAPSHOT_COLUMN].astype(str).nunique()}")
    print(f"BRONZE_MATCHED= {matched}")
    print(f"BRONZE_AGREEING= {agreeing}")
    if not args.apply:
        print("MODE= dry-run (nothing written)")
        return 0

    if plan.converted:
        write_odds_table(args.data_root, plan.after)
        stored = load_odds(args.data_root)
        assert_post_state(before, stored)
        print(f"ROWS_WRITTEN= {len(stored)} (read back and re-asserted)")
    print("MODE= apply")
    return 0


if __name__ == "__main__":
    sys.exit(run())
