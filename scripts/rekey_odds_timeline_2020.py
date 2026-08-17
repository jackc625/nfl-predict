"""Re-key the orphaned 2020 rows of the paid ``odds_timeline`` archive.

Quick task 260816-u0e. Background, stated plainly because the correct tool here
is NOT the obvious one:

``get_nfl_season_start`` derived the season opener as the first Thursday in
September instead of the Thursday after Labor Day. The two rules agree in every
season except those where Sept 1 falls on a Tuesday -- 2020 and 2026. For 2020
the old rule returned 2020-09-03 instead of 2020-09-10, so every week number
derived from it ran exactly one week high. The FUNCTION was fixed in ``45bff24``;
the 1,780 rows already written into ``data/silver/odds_timeline.parquet`` under
the old rule were never re-keyed, so 261 of 262 distinct 2020 ``game_id`` values
fail to join ``games`` silver and the paid 2020 trajectory contributes nothing to
gold.

This tool recovers those keys WITHOUT re-purchasing anything. It never imports or
invokes the Odds API client: the 9,957-row archive cost 7,210 real credits and is
irreplaceable at that price.

Three mutually exclusive modes:

* ``--bronze-completeness-report`` -- the D-Q1 gate. Proves (or disproves) that
  ``odds_timeline`` silver is reproducible from the bronze snapshots. Exits
  non-zero when bronze is short for any season, which FORBIDS the
  rebuild-from-bronze branch and mandates the in-place re-key below.
* ``--dry-run`` -- builds the map and asserts every invariant ``--apply`` will
  enforce, writing nothing.
* ``--apply`` -- the full-table atomic rewrite.

Why a full-table rewrite and not ``upsert_silver_composite``: that function is a
pure insert-or-update on ``(game_id, snapshot_ts)`` with NO delete path
(``data/storage.py:1149-1160``). Re-keyed rows carry NEW keys, so upserting them
would leave all 1,780 old wrong-keyed rows in place -- 11,737 rows with 2020
present twice under two key sets, the orphans invisible to coverage checks. The
whole table is therefore read, re-keyed in memory, asserted, and written back as
one file via a temp path + ``os.replace`` so a crash cannot leave a truncated
archive.

``odds_snapshot`` is never read or written by this tool (D-11).
"""

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from conf.settings import get_settings
from data.storage import (
    ParquetManager,
    _canonicalize_snapshot_ts_utc,
    get_db_connection,
    load_dataframe,
)
from utils import get_logger
from utils.date_utils import ET
from utils.game_id_utils import create_standard_game_id, parse_game_id

logger = get_logger(__name__)

# The only season the season-start bug touched in the archive's 2020-2024 span.
TARGET_SEASON = 2020

# Seasons that MUST come through the rewrite byte-identical.
FROZEN_SEASONS = (2021, 2022, 2023, 2024)

# The archive state this tool was built against, measured 2026-08-16. A mismatch
# means the archive is not in the state the quick task was planned against, and
# the tool refuses to write rather than guess.
EXPECTED_TOTAL_ROWS = 9957
EXPECTED_TOTAL_PAIRS = 9957
EXPECTED_SEASON_ROWS = {
    2020: 1780,
    2021: 1219,
    2022: 1452,
    2023: 2759,
    2024: 2747,
}
EXPECTED_STORED_2020_IDS = 262
EXPECTED_TRUE_2020_IDS = 256

# The OLD ``NFL_SEASON_START_DAY_RANGE`` lower bound, as it stood before
# ``45bff24`` widened the check to (4, 10). Load-bearing -- see
# :func:`old_rule_season_start`.
_OLD_SEASON_START_DAY_MIN = 3

# Weekday index of Thursday (Monday == 0), used by the frozen replay.
_THURSDAY = 3

_BRONZE_GLOB = "odds_timeline_raw_bronze_*.parquet"

_KEY_COLUMNS = ["game_id", "snapshot_ts"]


class RekeyInvariantError(RuntimeError):
    """Raised when a re-key invariant is breached. The archive is left untouched."""


# ----------------------------------------------------------------------
# The frozen replay of the fixed bug
# ----------------------------------------------------------------------


def old_rule_season_start(season: int) -> datetime:
    """Return the season opener as the PRE-``45bff24`` rule computed it.

    This is a FROZEN REPLAY OF A FIXED BUG. It deliberately reproduces the exact
    body ``utils.date_utils.get_nfl_season_start`` had before commit ``45bff24``,
    so the wrong ``game_id`` values already sitting in the archive can be
    re-derived and inverted. It is NOT a date utility, it must never be called by
    production code, and it must never be "corrected" -- correcting it would make
    it stop matching the stored keys, which is the only thing it is for.

    The ``if first_thursday.day < 3`` clause is the load-bearing part. It mirrors
    the OLD ``NFL_SEASON_START_DAY_RANGE`` lower bound of 3. Omitting it makes
    2021 and 2022 appear affected when they are not.

    Args:
        season: NFL season year.

    Returns:
        The season opener under the old rule (midnight ET on that Thursday).
    """
    sept_first = datetime(season, 9, 1, tzinfo=ET)
    days_to_thursday = (_THURSDAY - sept_first.weekday()) % 7
    first_thursday = sept_first + timedelta(days=days_to_thursday)
    if first_thursday.day < _OLD_SEASON_START_DAY_MIN:
        first_thursday += timedelta(days=7)
    return first_thursday


def old_rule_week(kickoff_et: datetime, season: int) -> int:
    """Return the week number the old rule derived for *kickoff_et*.

    Mirrors ``scripts/ingest_odds_timeline._derive_season_week`` (lines 170-172),
    which is what actually produced the stored ids: weeks elapsed since the
    season opener, clamped to 1..22.
    """
    days_since_start = (kickoff_et - old_rule_season_start(season)).days
    return max(1, min(days_since_start // 7 + 1, 22))


def kickoff_as_et(value: object) -> datetime:
    """Return a games-row kickoff as an ET-localized datetime.

    ``games.kickoff_et`` is stored as a tz-aware column whose WALL CLOCK is
    already Eastern (the column name is the contract), so the wall clock is read
    off and re-attached to ET rather than converted.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=ET)


# ----------------------------------------------------------------------
# The re-key map
# ----------------------------------------------------------------------


def build_rekey_map(
    games_2020: pd.DataFrame, stored_ids: Iterable[str]
) -> dict[str, str]:
    """Map each stored 2020 ``game_id`` onto its true ``games`` silver id.

    Each stored id is resolved in a strict three-step order:

    1. **Fixed point.** If the stored id is already a true 2020 ``games`` id it
       maps to itself. This is what makes a second ``--apply`` run a no-op.
    2. **Old-rule inverse.** The old rule is replayed FORWARD over every 2020 REG
       game's kickoff to produce the id that game WOULD have been stored under,
       and that map is inverted. This is an exact re-derivation, not a guess.
    3. **REG matchup fallback.** ``(away_team, home_team)`` is unique across all
       256 2020 REG games, which resolves the residue: early-September board
       listings whose provisional dates later moved, so no week arithmetic can
       recover them.

    Raises rather than warns on any unmapped id -- a ``.map()`` that quietly
    yields NaN, or a ``.fillna(original)``, would leave paid rows orphaned under
    a key that silently fails to join.

    Args:
        games_2020: The 2020 slice of ``games`` silver (all game types).
        stored_ids: The distinct 2020 ``game_id`` values in the archive.

    Returns:
        ``{stored_id: true_id}`` covering every stored id.

    Raises:
        RekeyInvariantError: On a duplicate map key or any unmapped stored id.
    """
    true_ids = set(games_2020["game_id"])
    reg = games_2020[games_2020["game_type"] == "REG"]

    old_inverse: dict[str, str] = {}
    matchup: dict[tuple[str, str], str] = {}
    for row in reg.itertuples(index=False):
        week = old_rule_week(kickoff_as_et(row.kickoff_et), TARGET_SEASON)
        old_id = create_standard_game_id(
            season=TARGET_SEASON,
            week=week,
            away_team=row.away_team,
            home_team=row.home_team,
        )
        if old_inverse.get(old_id, row.game_id) != row.game_id:
            raise RekeyInvariantError(
                f"Old-rule inverse is ambiguous: {old_id!r} maps to both "
                f"{old_inverse[old_id]!r} and {row.game_id!r}"
            )
        old_inverse[old_id] = row.game_id

        key = (row.away_team, row.home_team)
        if matchup.get(key, row.game_id) != row.game_id:
            raise RekeyInvariantError(
                f"REG matchup map is ambiguous: {key} maps to both "
                f"{matchup[key]!r} and {row.game_id!r}"
            )
        matchup[key] = row.game_id

    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for stored_id in stored_ids:
        if stored_id in true_ids:
            mapping[stored_id] = stored_id
            continue
        if stored_id in old_inverse:
            mapping[stored_id] = old_inverse[stored_id]
            continue
        parsed = parse_game_id(stored_id)
        key = (parsed["away_team"], parsed["home_team"])
        if key in matchup:
            mapping[stored_id] = matchup[key]
            continue
        unmapped.append(stored_id)

    if unmapped:
        raise RekeyInvariantError(
            f"{len(unmapped)} stored 2020 game_id values could not be resolved to "
            f"a games silver id: {sorted(unmapped)[:10]}. Refusing to write -- an "
            "unmapped id must never be silently left orphaned."
        )

    return mapping


def resolution_breakdown(
    games_2020: pd.DataFrame, mapping: dict[str, str]
) -> dict[str, int]:
    """Count how many stored ids each resolution step accounted for."""
    true_ids = set(games_2020["game_id"])
    fixed_point = sum(1 for stored in mapping if stored in true_ids)
    shifted = len(mapping) - fixed_point
    return {"fixed_point": fixed_point, "shifted": shifted}


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------


def data_root(base_path: Path | None = None) -> Path:
    """Return the data root, defaulting to the configured one."""
    if base_path is not None:
        return Path(base_path)
    return Path(get_settings().config.data.root_path)


def timeline_path(base_path: Path | None = None) -> Path:
    """Path to the ``odds_timeline`` silver parquet."""
    return data_root(base_path) / "silver" / "odds_timeline.parquet"


def _season_series(game_ids: pd.Series) -> pd.Series:
    """Season number parsed from the ``{season}_W..`` game_id prefix."""
    return game_ids.str.slice(0, 4).astype(int)


def _per_season_rows(df: pd.DataFrame) -> dict[int, int]:
    counts = _season_series(df["game_id"]).value_counts().sort_index()
    return {int(season): int(n) for season, n in counts.items()}


def _distinct_pairs(df: pd.DataFrame) -> int:
    return len(df.drop_duplicates(subset=_KEY_COLUMNS))


def _frozen_slice_hash(df: pd.DataFrame) -> str:
    """A content hash of the 2021-2024 slice, sorted for stability."""
    seasons = _season_series(df["game_id"])
    frozen = df[seasons.isin(FROZEN_SEASONS)].copy()
    frozen = frozen.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(frozen, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


# ----------------------------------------------------------------------
# D-Q1 gate: bronze completeness
# ----------------------------------------------------------------------


def bronze_completeness_report(base_path: Path | None = None) -> int:
    """Print the D-Q1 bronze-vs-silver completeness proof. Returns an exit code.

    Silver is only reproducible from bronze if bronze holds at least as many
    rows and at least as many distinct ``(game_id, snapshot_ts)`` pairs for every
    season. Any shortfall means a rebuild-from-bronze would DESTROY paid rows, so
    the gate exits non-zero and the in-place re-key is mandated instead.
    """
    root = data_root(base_path)
    silver = pd.read_parquet(timeline_path(base_path), engine="pyarrow")
    bronze_files = sorted((root / "bronze").glob(_BRONZE_GLOB))

    print("D-Q1 BRONZE COMPLETENESS PROOF (odds_timeline)")
    print(f"  bronze files on disk: {len(bronze_files)}")
    if not bronze_files:
        print("  FAIL: no bronze snapshots found at all.")
        return 1

    bronze = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in bronze_files],
        ignore_index=True,
    )
    bronze["snapshot_ts"] = pd.to_datetime(bronze["snapshot_ts"], utc=True)
    silver["snapshot_ts"] = pd.to_datetime(silver["snapshot_ts"], utc=True)

    silver_rows = _per_season_rows(silver)
    bronze_rows = _per_season_rows(bronze)
    silver_pairs = silver.drop_duplicates(subset=_KEY_COLUMNS).pipe(_per_season_rows)
    bronze_pairs = bronze.drop_duplicates(subset=_KEY_COLUMNS).pipe(_per_season_rows)

    print()
    print(
        f"  {'season':>7} {'silver rows':>12} {'bronze rows':>12} "
        f"{'silver pairs':>13} {'bronze pairs':>13} {'shortfall':>10}"
    )
    failed_seasons: list[int] = []
    for season in sorted(silver_rows):
        s_rows = silver_rows.get(season, 0)
        b_rows = bronze_rows.get(season, 0)
        s_pairs = silver_pairs.get(season, 0)
        b_pairs = bronze_pairs.get(season, 0)
        shortfall = b_pairs - s_pairs
        if b_rows < s_rows or b_pairs < s_pairs:
            failed_seasons.append(season)
        print(
            f"  {season:>7} {s_rows:>12} {b_rows:>12} "
            f"{s_pairs:>13} {b_pairs:>13} {shortfall:>10}"
        )

    silver_keys = set(map(tuple, silver[_KEY_COLUMNS].to_numpy()))
    bronze_keys = set(map(tuple, bronze[_KEY_COLUMNS].to_numpy()))
    print()
    print(f"  pairs in silver but NOT in bronze: {len(silver_keys - bronze_keys)}")
    print(f"  pairs in bronze but NOT in silver: {len(bronze_keys - silver_keys)}")
    print(
        f"  distinct snapshot_ts  silver={silver['snapshot_ts'].nunique()} "
        f"bronze={bronze['snapshot_ts'].nunique()}"
    )

    if not failed_seasons:
        print()
        print("  PASS: every season is reproducible from bronze.")
        return 0

    print()
    print(f"  FAIL: bronze is short for seasons {failed_seasons}.")
    print("  The rebuild-from-bronze branch is FORBIDDEN: rebuilding would")
    print("  permanently destroy the paid rows bronze no longer holds.")
    print("  The in-place 2020-only re-key (--apply) is the mandated fallback.")
    print()
    print("  PROVEN CAUSE: save_bronze_snapshot builds its filename from a")
    print("  SECOND-resolution timestamp (data/storage.py:985-986) and")
    print("  pq.write_table OVERWRITES, so cadence writes landing in the same")
    print("  wall-clock second silently clobber each other. Of the 360 expected")
    print(f"  snapshots only {len(bronze_files)} files survived.")
    return 1


# ----------------------------------------------------------------------
# The re-key itself
# ----------------------------------------------------------------------


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RekeyInvariantError(message)


def prepare_rekey(base_path: Path | None = None) -> dict:
    """Load the archive, build the map, apply it in memory, assert everything.

    Writes nothing. Returns a report dict carrying the re-keyed frame plus every
    measured invariant, so ``--dry-run`` and ``--apply`` enforce an identical
    check set and can never drift apart.
    """
    before = pd.read_parquet(timeline_path(base_path), engine="pyarrow")
    before["snapshot_ts"] = pd.to_datetime(before["snapshot_ts"], utc=True)

    games = load_dataframe("games", layer="silver")
    games_2020 = games[games["season"] == TARGET_SEASON]
    true_ids = set(games_2020["game_id"])

    seasons_before = _season_series(before["game_id"])
    slice_2020_before = before[seasons_before == TARGET_SEASON]
    stored_ids = sorted(slice_2020_before["game_id"].unique())
    orphans = [gid for gid in stored_ids if gid not in true_ids]
    already_rekeyed = not orphans

    # --- before-state must match the recorded constants ---------------
    _assert(
        len(before) == EXPECTED_TOTAL_ROWS,
        f"archive has {len(before)} rows, expected {EXPECTED_TOTAL_ROWS}; the "
        "archive is not in the state this tool was planned against",
    )
    _assert(
        _distinct_pairs(before) == EXPECTED_TOTAL_PAIRS,
        f"archive has {_distinct_pairs(before)} distinct (game_id, snapshot_ts) "
        f"pairs, expected {EXPECTED_TOTAL_PAIRS}",
    )
    rows_before = _per_season_rows(before)
    _assert(
        rows_before == EXPECTED_SEASON_ROWS,
        f"per-season row counts {rows_before} != expected {EXPECTED_SEASON_ROWS}",
    )
    expected_stored = (
        EXPECTED_TRUE_2020_IDS if already_rekeyed else EXPECTED_STORED_2020_IDS
    )
    _assert(
        len(stored_ids) == expected_stored,
        f"2020 slice carries {len(stored_ids)} distinct ids, expected "
        f"{expected_stored}",
    )

    pairs_2020_before = _distinct_pairs(slice_2020_before)
    frozen_hash_before = _frozen_slice_hash(before)

    mapping = build_rekey_map(games_2020, stored_ids)

    after = before.copy()
    mask_2020 = seasons_before == TARGET_SEASON
    remapped = after.loc[mask_2020, "game_id"].map(mapping)
    _assert(
        int(remapped.isna().sum()) == 0,
        "the re-key map left rows unmapped; refusing to write",
    )
    after.loc[mask_2020, "game_id"] = remapped

    seasons_after = _season_series(after["game_id"])
    slice_2020_after = after[seasons_after == TARGET_SEASON]

    # --- post-state invariants ----------------------------------------
    _assert(
        len(slice_2020_after) == EXPECTED_SEASON_ROWS[TARGET_SEASON],
        f"2020 slice holds {len(slice_2020_after)} rows after the re-key, "
        f"expected {EXPECTED_SEASON_ROWS[TARGET_SEASON]}",
    )
    pairs_2020_after = _distinct_pairs(slice_2020_after)
    _assert(
        pairs_2020_after == pairs_2020_before == EXPECTED_SEASON_ROWS[TARGET_SEASON],
        f"2020 distinct pairs moved {pairs_2020_before} -> {pairs_2020_after}; "
        "a merged id silently swallowed a paid row",
    )
    true_id_count = int(slice_2020_after["game_id"].nunique())
    _assert(
        true_id_count == EXPECTED_TRUE_2020_IDS,
        f"2020 collapsed to {true_id_count} distinct ids, expected "
        f"{EXPECTED_TRUE_2020_IDS}",
    )
    missing = sorted(set(slice_2020_after["game_id"]) - true_ids)
    _assert(
        not missing,
        f"{len(missing)} re-keyed 2020 ids do not join games silver: {missing[:10]}",
    )
    _assert(
        _distinct_pairs(after) == EXPECTED_TOTAL_PAIRS,
        f"whole-table distinct pairs became {_distinct_pairs(after)}, expected "
        f"{EXPECTED_TOTAL_PAIRS}: a re-keyed 2020 pair collided with another row",
    )
    _assert(
        len(after) == EXPECTED_TOTAL_ROWS,
        f"whole-table row count became {len(after)}, expected {EXPECTED_TOTAL_ROWS}",
    )
    rows_after = _per_season_rows(after)
    _assert(
        rows_after == EXPECTED_SEASON_ROWS,
        f"per-season row counts after the re-key {rows_after} != "
        f"{EXPECTED_SEASON_ROWS}",
    )
    frozen_hash_after = _frozen_slice_hash(after)
    _assert(
        frozen_hash_after == frozen_hash_before,
        "the 2021-2024 slice hash changed; the re-key touched a frozen season",
    )

    return {
        "already_rekeyed": already_rekeyed,
        "orphan_ids": len(orphans),
        "stored_2020_ids": len(stored_ids),
        "true_2020_ids": true_id_count,
        "unmapped": 0,
        "collisions": EXPECTED_TOTAL_PAIRS - _distinct_pairs(after),
        "pairs_2020_before": pairs_2020_before,
        "pairs_2020_after": pairs_2020_after,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "frozen_slice_hash": frozen_hash_before,
        "resolution": resolution_breakdown(games_2020, mapping),
        "mapping": mapping,
        "after": after,
    }


def _refresh_duckdb_copy(df: pd.DataFrame) -> bool:
    """Keep a DuckDB ``odds_timeline`` table (if any) in step with the parquet.

    ``load_dataframe(..., source="auto")`` resolves DuckDB FIRST and only falls
    back to parquet (``data/storage.py:916-921``), so a stale DuckDB copy would
    silently defeat the parquet rewrite and feed the gold rebuild wrong-keyed
    rows. Returns True when a table existed and was replaced.
    """
    db = get_db_connection()
    if not db.table_exists("odds_timeline"):
        return False
    db.create_table_from_df(df, "odds_timeline", if_exists="replace")
    return True


def _atomic_write(df: pd.DataFrame, path: Path, base_path: Path | None) -> None:
    """Write *df* to *path* as one file, atomically.

    Mirrors the write tail of ``upsert_silver_composite``
    (``data/storage.py:1162-1167``) -- canonicalize ``snapshot_ts`` to tz-aware
    UTC, normalize datetimes, snappy parquet -- but REPLACES the table instead of
    merging into it, then an ``os.replace`` (via ``Path.replace``) swaps the
    finished file into position so a crash cannot leave a truncated archive.
    """
    canonical = _canonicalize_snapshot_ts_utc(df)
    pm = ParquetManager(str(data_root(base_path)))
    normalized = pm._normalize_parquet_datetime_columns(canonical)
    table = pa.Table.from_pandas(normalized)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression="snappy")
    tmp_path.replace(path)


def apply_rekey(base_path: Path | None = None) -> dict:
    """Run the full-table atomic rewrite, then verify through ``load_dataframe``."""
    path = timeline_path(base_path)
    report = prepare_rekey(base_path)

    if report["already_rekeyed"]:
        print("Archive is already re-keyed: 0 orphaned 2020 ids. Nothing written.")
        report["written"] = False
        return report

    _atomic_write(report["after"], path, base_path)
    report["written"] = True
    report["duckdb_refreshed"] = _refresh_duckdb_copy(report["after"])

    # The decisive assertion goes THROUGH load_dataframe, not a direct parquet
    # read, so it proves what the feature builders will actually see.
    reloaded = load_dataframe("odds_timeline", layer="silver")
    reloaded["snapshot_ts"] = pd.to_datetime(reloaded["snapshot_ts"], utc=True)
    _assert(
        str(reloaded["snapshot_ts"].dtype) == "datetime64[ns, UTC]",
        f"snapshot_ts round-tripped as {reloaded['snapshot_ts'].dtype}",
    )
    _assert(
        len(reloaded) == EXPECTED_TOTAL_ROWS
        and _distinct_pairs(reloaded) == EXPECTED_TOTAL_PAIRS,
        "the reloaded archive does not match the expected row/pair counts",
    )
    games = load_dataframe("games", layer="silver")
    true_ids = set(games[games["season"] == TARGET_SEASON]["game_id"])
    reloaded_2020 = reloaded[_season_series(reloaded["game_id"]) == TARGET_SEASON]
    _assert(
        int(reloaded_2020["game_id"].nunique()) == EXPECTED_TRUE_2020_IDS,
        "the reloaded 2020 slice does not carry the expected distinct id count",
    )
    _assert(
        set(reloaded_2020["game_id"]) <= true_ids,
        "a reloaded 2020 id fails to join games silver",
    )
    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _print_report(report: dict, *, mode: str) -> None:
    print(f"2020 ODDS_TIMELINE RE-KEY ({mode})")
    print(f"  already re-keyed:        {report['already_rekeyed']}")
    print(f"  stored 2020 ids:         {report['stored_2020_ids']}")
    print(f"  true 2020 ids after:     {report['true_2020_ids']}")
    print(f"  unmapped ids:            {report['unmapped']}")
    print(f"  (new_id, ts) collisions: {report['collisions']}")
    print(
        f"  2020 distinct pairs:     {report['pairs_2020_before']} -> "
        f"{report['pairs_2020_after']}"
    )
    print(f"  per-season rows before:  {report['rows_before']}")
    print(f"  per-season rows after:   {report['rows_after']}")
    print(f"  2021-2024 slice hash:    {report['frozen_slice_hash'][:16]}")
    print(f"  resolution:              {report['resolution']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-key the orphaned 2020 rows of the paid odds_timeline archive. "
            "Makes no Odds API call of any kind."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--bronze-completeness-report",
        action="store_true",
        help="D-Q1 gate: prove (or disprove) that silver is reproducible from "
        "bronze. Non-zero exit forbids the rebuild-from-bronze branch.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the map and assert every --apply invariant, writing nothing.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the whole odds_timeline table with the 2020 rows re-keyed.",
    )
    parser.add_argument(
        "--backup-to",
        type=Path,
        help="Copy the archive to this path before --apply writes.",
    )
    return parser


def main() -> None:
    """CLI entry point for the 2020 archive re-key."""
    args = build_parser().parse_args()

    try:
        if args.bronze_completeness_report:
            sys.exit(bronze_completeness_report())

        if args.dry_run:
            _print_report(prepare_rekey(), mode="dry run -- nothing written")
            sys.exit(0)

        if args.backup_to is not None:
            args.backup_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(timeline_path(), args.backup_to)
            print(f"Backed up archive to {args.backup_to}")

        report = apply_rekey()
        _print_report(report, mode="applied" if report["written"] else "no-op")
        sys.exit(0)

    except RekeyInvariantError as exc:
        logger.error("Re-key invariant breached", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
