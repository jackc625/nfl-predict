"""PROMOTE the historical four juice columns into the flat silver odds table.

WHAT THIS EXECUTES, AND WHY IT IS A SEPARATE MODULE
---------------------------------------------------
``PROFITABILITY-PREREGISTRATION.md`` section 4.2 is a FROZEN clause, ratified at CHECKPOINT 1:

    The historical four juice columns are PROMOTED from the partitioned silver store, not
    re-ingested across the network.

That promote had not reached ``data/silver/odds_snapshot.parquet``. Measured on 2026-09-05, the
flat table carried the four juice columns for 285 of 2,140 rows and every one of those 285 was
2025, while the partitioned store carried them for all 1,855 historical rows. With the DEF-31-13
devig ruling live, that split priced the 2025 HOLD on devigged real juice and the 2021-2024 TUNE
window on the flat -110 fallback -- so the EV floor ``t`` would have been swept in one pricing
regime and applied in another, permanently, the moment Plan 31-14 armed the hold.

This module is that promote. It is a SEPARATE, NAMED step rather than a mode of
``scripts/ingest_historical_odds.py`` for the reason
:func:`~scripts.ingest_historical_odds.remove_synthetic_stored_rows` states about itself: it
WRITES production silver, so it must be invoked, counted and reported on its own rather than
folded into a run whose report is about something else. An ingest reaches the network; a promote
never does. They are different operations and they fail in different ways.

IT WRITES THROUGH THE PRE-REGISTERED WRITE PATH, NEVER AROUND IT
----------------------------------------------------------------
The one write is :func:`~scripts.ingest_historical_odds.write_odds_additively` -- Plan 31-08's
gated, ``base_path``-parameterised merge. ``ingest_historical_odds_for_seasons`` is NOT used (it
hard-codes the production root and reaches nflreadpy) and ``upsert_silver`` is NOT called directly
(that path runs no gate and, because it replaces a whole row by key, overwrites a stored line by
construction).

WHAT IS ADDED, AND WHAT PROVABLY IS NOT
----------------------------------------
The incoming frame is built as an EXACT COPY of the stored rows, with ONLY the four juice columns
filled in. Every other column -- ``spread``, ``total``, ``ml_home``, ``ml_away``, ``sportsbook``,
``snapshot_ts``, ``is_live``, ``last_update``, ``created_at`` -- is carried from the stored row
unchanged, so "nothing but juice moved" is true BY CONSTRUCTION rather than by inspection
afterwards. ``preserve_stored_lines`` then re-carries the four protected line values on top of
that, which is a no-op here and is deliberately left in the path: the clause-2 guarantee should
not depend on this module having built its frame the careful way.

THE 2025 PARTITION IS NOT A SOURCE, BY CONSTRUCTION
----------------------------------------------------
Section 4.1 states that the partitioned store is NOT the 2025 odds source and says so in advance
precisely so the choice cannot be argued after the fact: the ``2025-10-03`` partition holds nine
sportsbooks of which the OUM-06 allowlist admits exactly one, and ``consensus`` does not appear in
it at all. This module therefore drops every row whose season is in the frozen
:data:`~backtest.ev_chain_constants.HOLD_SEASONS_P31` before it reads a single value, and the
season is taken from each row's own ``game_id`` rather than from the partition directory name --
a directory name is a label, an id is the fact.

THE FOUR REFUSALS
-----------------
Each one is a hard failure naming the offending keys, never a warning, and each exists because a
promote that silently misses or silently clobbers a row is worse than one that refuses:

1. **The accumulated copies must agree.** ``pq.write_to_dataset`` never deletes prior files, so
   the store ACCUMULATES: 7,829 rows on disk hold 2,120 distinct ``(game_id, sportsbook)`` pairs
   (Plan 31-02). Deduplicating a key whose copies DISAGREE would silently pick a price by file
   order. :func:`read_partitioned_juice` refuses instead.
2. **A null juice value is investigated, not written.** The same rule the ingest already applies
   (Plan 31-08: "a missing juice value is a FAILURE TO INVESTIGATE, never a -110 default to
   fill"). Measured: zero nulls across all four columns in the historical partitions.
3. **The PROMOTE branch rule is re-evaluated at write time.** D31-39's third clause is that the
   partitioned rows agree with the flat table row-for-row on every shared value column with zero
   mismatches. That was measured in Plan 31-02 and is CHECKED AGAIN here against the four
   protected line columns, because a branch rule trusted from a report is a branch rule nobody
   re-ran.
4. **A stored juice value is never overwritten with a different one.** Re-promoting an identical
   value is idempotent and allowed; changing one is refused. This is what makes a second run
   safe and a silent price change impossible.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backtest.ev_chain_constants import HOLD_SEASONS_P31
from scripts.ingest_historical_odds import (
    JUICE_COLUMNS,
    PROTECTED_LINE_COLUMNS,
    OddsWriteReport,
    canonical_game_id,
    load_features_ou,
    write_odds_additively,
)
from utils import get_logger

logger = get_logger(__name__)

# The silver table the promote reads from and writes into. One name, used for both, because the
# partitioned files and the flat file ARE one logical table (D31-39, Plan 31-02: the opaque
# filenames are per-write pyarrow GUIDs, not ParquetManager table-name hashes).
SILVER_ODDS_TABLE = "odds_snapshot"

# How the partitioned store's directories are named on disk. ``snapshot_ts`` is the partition key,
# so it lives in the directory name and is ABSENT as a column from the files themselves -- which
# is exactly why the incoming frame is built from the stored rows rather than from these.
PARTITION_GLOB = "snapshot_ts=*/*.parquet"

# The pair a promoted value is keyed on. ``game_id`` alone would be ambiguous in the 2025
# partition, which carries nine sportsbooks for one game; the historical partitions carry one.
PROMOTE_KEY_COLUMNS: tuple[str, ...] = ("game_id", "sportsbook")


def _season_of(game_id: str) -> int | None:
    """The season a canonical ``game_id`` names, or None when it does not name one.

    Args:
        game_id: A project-standard game id.

    Returns:
        The four-digit season as an int, or None when the leading token is not one.
    """
    head = str(game_id)[:4]
    return int(head) if head.isdigit() else None


@dataclass(frozen=True)
class JuicePromoteReport:
    """What the promote did, as counted return values.

    Attributes:
        rows_in_partitions: Rows read off the partitioned store, before any filter.
        rows_after_hold_exclusion: Rows left once the frozen hold seasons were dropped.
        partition_rows_rekeyed: Partition rows whose ``game_id`` was non-canonical and was
            normalized before matching. A non-zero count is the Rams resolution doing its job on
            the SOURCE side; the stored side was normalized in an earlier plan.
        distinct_juice_keys: Distinct ``(game_id, sportsbook)`` pairs carrying juice after the
            accumulated copies were collapsed.
        stored_rows_matched: Stored rows a promoted value was found for.
        stored_rows_already_juiced: Matched stored rows that already carried the identical value,
            so the promote was a no-op for them (this is what makes a re-run idempotent).
        stored_rows_not_served: Stored rows the partitioned store offered NO juice for, by season.
            This is NOT the same claim as "these rows have no juice": a hold-season row is
            deliberately not served here and already carries the juice its own ingest wrote.
        stored_rows_still_without_juice: Stored rows that carry a null in at least one of the four
            juice columns AFTER the promote, by season, measured on what is on disk. THIS is the
            coverage gap, and it is reported rather than implied away by a matched count.
        write: The merge record, or None on a dry run.
    """

    rows_in_partitions: int
    rows_after_hold_exclusion: int
    partition_rows_rekeyed: int
    distinct_juice_keys: int
    stored_rows_matched: int
    stored_rows_already_juiced: int
    stored_rows_not_served: dict[int, int] = field(default_factory=dict)
    stored_rows_still_without_juice: dict[int, int] = field(default_factory=dict)
    write: OddsWriteReport | None = None


def read_partitioned_juice(base_path: Path) -> tuple[pd.DataFrame, int, int, int]:
    """Read the HISTORICAL juice off the partitioned silver store, one row per key.

    Args:
        base_path: The data root whose ``silver/`` holds the partition directories.

    Returns:
        ``(frame, rows_in_partitions, rows_after_hold_exclusion, rows_rekeyed)`` where *frame*
        carries ``game_id``, ``sportsbook``, the four juice columns and the four protected line
        columns, with exactly one row per ``(game_id, sportsbook)``.

    Raises:
        ValueError: when the accumulated copies of one key disagree, or when any juice value is
            null. Both are refusals, not warnings -- see the module docstring.
    """
    silver = Path(base_path) / "silver"
    files = sorted(silver.glob(PARTITION_GLOB))
    if not files:
        msg = (
            f"no partitioned odds files under {silver.as_posix()}/{PARTITION_GLOB}. The "
            "section-4.2 PROMOTE reads the partitioned silver store and never the network; with "
            "no partition to read there is nothing to promote and nothing to guess."
        )
        raise ValueError(msg)

    frames = [pd.read_parquet(path) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    rows_in_partitions = len(raw)

    raw = raw.copy()
    original_ids = raw["game_id"].astype(str)
    raw["game_id"] = original_ids.map(canonical_game_id)
    rows_rekeyed = int((raw["game_id"] != original_ids).sum())

    # The hold is excluded BY CONSTRUCTION, before any value is read (section 4.1).
    seasons = raw["game_id"].map(_season_of)
    raw = raw[~seasons.isin(HOLD_SEASONS_P31)]
    rows_after_hold_exclusion = len(raw)

    wanted = [*PROMOTE_KEY_COLUMNS, *JUICE_COLUMNS, *PROTECTED_LINE_COLUMNS]
    missing = [column for column in wanted if column not in raw.columns]
    if missing:
        msg = (
            f"the partitioned odds store is missing {missing}. The PROMOTE branch (D31-39) was "
            "chosen because every partition file carries all four juice columns; a file that "
            "does not is a different store from the one the branch rule was evaluated on."
        )
        raise ValueError(msg)
    raw = raw[wanted]

    _refuse_disagreeing_copies(raw)
    _refuse_null_juice(raw)

    deduped = raw.drop_duplicates(
        subset=list(PROMOTE_KEY_COLUMNS), keep="first"
    ).reset_index(drop=True)
    return deduped, rows_in_partitions, rows_after_hold_exclusion, rows_rekeyed


def _refuse_disagreeing_copies(frame: pd.DataFrame) -> None:
    """Raise when the accumulated copies of one key hold more than one value (refusal 1)."""
    columns = [*JUICE_COLUMNS, *PROTECTED_LINE_COLUMNS]
    spread = frame.groupby(list(PROMOTE_KEY_COLUMNS))[columns].nunique(dropna=False)
    offenders = spread[(spread > 1).any(axis=1)]
    if offenders.empty:
        return
    named = [f"{gid} / {book}" for gid, book in list(offenders.index)[:10]]
    msg = (
        f"{len(offenders)} partitioned odds key(s) carry DISAGREEING values across their "
        f"accumulated copies: {named}. pq.write_to_dataset never deletes a prior file, so the "
        "store accumulates one copy per write; deduplicating a disagreeing key would pick a "
        "price by file order. This is a refusal, not a warning."
    )
    raise ValueError(msg)


def _refuse_null_juice(frame: pd.DataFrame) -> None:
    """Raise when any partitioned juice value is null (refusal 2)."""
    null_mask = frame[list(JUICE_COLUMNS)].isna().any(axis=1)
    if not bool(null_mask.any()):
        return
    offenders = frame.loc[null_mask, list(PROMOTE_KEY_COLUMNS)]
    named = [
        f"{row.game_id} / {row.sportsbook}" for row in offenders.head(10).itertuples()
    ]
    msg = (
        f"{int(null_mask.sum())} partitioned odds row(s) carry a NULL juice value: {named}. A "
        "missing price is a failure to investigate, never a -110 default to fill -- a fabricated "
        "price moves the per-bet breakeven and corrupts the EV floor it feeds."
    )
    raise ValueError(msg)


def build_promote_frame(
    stored: pd.DataFrame, promoted: pd.DataFrame
) -> tuple[pd.DataFrame, int, dict[int, int]]:
    """Build the incoming frame: the stored rows, with ONLY the four juice columns filled.

    Args:
        stored: The flat silver odds table as it stands.
        promoted: One row per ``(game_id, sportsbook)`` carrying the juice to add.

    Returns:
        ``(incoming, already_juiced, not_served_by_season)``. *incoming* holds exactly the stored
        rows a value was found for, with the stored column set and the stored values, and only the
        four juice columns changed. *not_served_by_season* counts the stored rows the partitioned
        store offered nothing for -- which is NOT the same claim as "those rows have no juice".

    Raises:
        ValueError: when a partitioned line value disagrees with the stored one (refusal 3), or
            when a stored juice value would be overwritten with a different one (refusal 4).
    """
    keys = list(PROMOTE_KEY_COLUMNS)
    lookup = promoted.set_index(keys)

    stored = stored.copy()
    index = pd.MultiIndex.from_arrays(
        [stored["game_id"].astype(str), stored["sportsbook"].astype(str)]
    )
    matched = index.isin(lookup.index)

    incoming = stored.loc[matched].copy()
    matched_index = index[matched]

    _refuse_line_disagreement(incoming, lookup, matched_index)

    already_juiced = 0
    for column in JUICE_COLUMNS:
        new_values = pd.Series(
            lookup[column].reindex(matched_index).to_numpy(), index=incoming.index
        )
        if column in incoming.columns:
            old = incoming[column]
            conflicting = old.notna() & (old != new_values)
            if bool(conflicting.any()):
                offenders = (
                    incoming.loc[conflicting, "game_id"].astype(str).head(10).tolist()
                )
                msg = (
                    f"the promote would OVERWRITE {int(conflicting.sum())} stored "
                    f"'{column}' value(s) with a different one: {offenders}. Section 4.2 adds "
                    "the juice; it does not restate it. A value that already differs is a "
                    "disagreement to investigate, not a value to replace."
                )
                raise ValueError(msg)
            already_juiced = max(
                already_juiced, int((old.notna() & (old == new_values)).sum())
            )
        incoming[column] = new_values

    unmatched = stored.loc[~matched]
    return (
        incoming.reset_index(drop=True),
        already_juiced,
        _count_by_season(unmatched["game_id"]),
    )


def _count_by_season(game_ids: pd.Series) -> dict[int, int]:
    """Count *game_ids* per season, skipping any id that does not name one."""
    counts: dict[int, int] = {}
    for game_id in game_ids.astype(str):
        season = _season_of(game_id)
        if season is None:
            continue
        counts[season] = counts.get(season, 0) + 1
    return counts


def rows_still_without_juice(frame: pd.DataFrame) -> dict[int, int]:
    """Rows carrying a null in ANY of the four juice columns, counted per season.

    Measured on the frame that is ON DISK after the write, never inferred from a matched count.
    "1,855 rows matched" and "no row lacks juice" are different claims, and only the second one
    is the coverage statement the DEF-31-13 devig ruling depends on.

    Args:
        frame: A silver odds table.

    Returns:
        ``{season -> count}``, empty when every row carries all four values.
    """
    present = [column for column in JUICE_COLUMNS if column in frame.columns]
    if len(present) < len(JUICE_COLUMNS):
        return _count_by_season(frame["game_id"])
    incomplete = frame.loc[frame[present].isna().any(axis=1)]
    return _count_by_season(incomplete["game_id"])


def _refuse_line_disagreement(
    incoming: pd.DataFrame, lookup: pd.DataFrame, matched_index: pd.MultiIndex
) -> None:
    """Raise when a partitioned line value disagrees with the stored one (refusal 3).

    This RE-EVALUATES D31-39's third branch clause -- that the partitioned rows agree with the
    flat table row-for-row on every shared value column with zero mismatches -- at write time,
    against the four columns clause 2 protects. A branch rule trusted from an earlier report is a
    branch rule nobody re-ran.
    """
    mismatched: dict[str, list[str]] = {}
    for column in PROTECTED_LINE_COLUMNS:
        if column not in incoming.columns or column not in lookup.columns:
            continue
        stored_values = incoming[column]
        partition_values = pd.Series(
            lookup[column].reindex(matched_index).to_numpy(), index=incoming.index
        )
        differs = ~(
            (stored_values == partition_values)
            | (stored_values.isna() & partition_values.isna())
        )
        if bool(differs.any()):
            mismatched[column] = (
                incoming.loc[differs, "game_id"].astype(str).head(10).tolist()
            )

    if not mismatched:
        return
    msg = (
        "the partitioned odds store DISAGREES with the flat table on a protected line column, so "
        f"the PROMOTE branch rule (D31-39 clause 3) no longer holds: {mismatched}. The branch was "
        "chosen because the two agreed row-for-row with zero mismatches; promoting juice off a "
        "store that has since diverged would attach one market's price to another market's line."
    )
    raise ValueError(msg)


def promote_historical_juice(
    *,
    base_path: Path,
    features_ou_df: pd.DataFrame,
    dry_run: bool = False,
) -> JuicePromoteReport:
    """Execute the section-4.2 PROMOTE against the silver odds table under *base_path*.

    Args:
        base_path: The data root to read and write under.
        features_ou_df: The gold O/U matrix the synthetic-id gate checks membership against.
        dry_run: Build and check the incoming frame, and write nothing.

    Returns:
        The counted record of what the promote did.

    Raises:
        ValueError: from any of the four refusals above, or from any gate inside
            :func:`~scripts.ingest_historical_odds.write_odds_additively`.
    """
    base_path = Path(base_path)
    stored_path = base_path / "silver" / f"{SILVER_ODDS_TABLE}.parquet"
    if not stored_path.is_file():
        msg = (
            f"no stored odds table at {stored_path.as_posix()}. A promote ADDS columns' values to "
            "rows that already exist; with no destination there is nothing to promote INTO, and "
            "creating one here would be an ingest wearing a promote's name."
        )
        raise ValueError(msg)

    stored = pd.read_parquet(stored_path)
    (
        promoted,
        rows_in_partitions,
        rows_after_hold_exclusion,
        rows_rekeyed,
    ) = read_partitioned_juice(base_path)

    incoming, already_juiced, not_served = build_promote_frame(stored, promoted)

    write: OddsWriteReport | None = None
    if not dry_run and not incoming.empty:
        write = write_odds_additively(
            incoming,
            base_path=base_path,
            features_ou_df=features_ou_df,
            table_name=SILVER_ODDS_TABLE,
        )

    # The coverage gap is measured on what is ON DISK after the write -- or, on a dry run, on the
    # frame the write would have produced. A matched count is a claim about the merge; this is the
    # claim about the table, and only the second one answers "does any row still lack juice?".
    settled = pd.read_parquet(write.path) if write is not None else stored
    report = JuicePromoteReport(
        rows_in_partitions=rows_in_partitions,
        rows_after_hold_exclusion=rows_after_hold_exclusion,
        partition_rows_rekeyed=rows_rekeyed,
        distinct_juice_keys=len(promoted),
        stored_rows_matched=len(incoming),
        stored_rows_already_juiced=already_juiced,
        stored_rows_not_served=not_served,
        stored_rows_still_without_juice=rows_still_without_juice(settled),
        write=write,
    )
    logger.info(
        "Promoted historical juice into the flat silver odds table",
        rows_in_partitions=report.rows_in_partitions,
        rows_after_hold_exclusion=report.rows_after_hold_exclusion,
        partition_rows_rekeyed=report.partition_rows_rekeyed,
        distinct_juice_keys=report.distinct_juice_keys,
        stored_rows_matched=report.stored_rows_matched,
        stored_rows_already_juiced=report.stored_rows_already_juiced,
        stored_rows_not_served=report.stored_rows_not_served,
        stored_rows_still_without_juice=report.stored_rows_still_without_juice,
        dry_run=dry_run,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "PROMOTE the historical four juice columns from the partitioned silver odds store "
            "into the flat table (PROFITABILITY-PREREGISTRATION.md section 4.2). Additive: the "
            "stored spread, total and both moneylines are never overwritten."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("data"),
        help="The data root to read and write under (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and check the incoming frame, then write nothing.",
    )
    return parser


def main() -> None:
    """CLI entry point for the section-4.2 promote."""
    args = build_parser().parse_args()
    try:
        report = promote_historical_juice(
            base_path=args.base_path,
            features_ou_df=load_features_ou(),
            dry_run=args.dry_run,
        )
    except ValueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"partition rows read            : {report.rows_in_partitions}")
    print(f"rows after hold exclusion      : {report.rows_after_hold_exclusion}")
    print(f"partition rows re-keyed to canon: {report.partition_rows_rekeyed}")
    print(f"distinct juice keys            : {report.distinct_juice_keys}")
    print(f"stored rows matched            : {report.stored_rows_matched}")
    print(f"stored rows already juiced     : {report.stored_rows_already_juiced}")
    print(f"stored rows NOT SERVED here    : {report.stored_rows_not_served}")
    print(f"rows STILL without juice       : {report.stored_rows_still_without_juice}")
    if report.write is None:
        print("DRY RUN -- nothing written.")
        return
    print(f"rows before                    : {report.write.rows_before}")
    print(f"rows after                     : {report.write.rows_after}")
    print(f"stored ids normalized          : {report.write.stored_ids_normalized}")
    print(f"rows with stored lines preserved: {report.write.rows_with_lines_preserved}")
    print(f"wrote                          : {report.write.path}")


if __name__ == "__main__":
    main()
