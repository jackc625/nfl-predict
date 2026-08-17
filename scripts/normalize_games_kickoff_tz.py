"""One-time normalization of the stale ``games.kickoff_et`` cohort (WR-06).

THE DIRECTION, FIRST, because getting it backwards is worse than not fixing it.

``games.kickoff_et`` is a TRUE INSTANT whose ET wall clock is the kickoff. This tool
shifts the rows that violate that contract SO THE STORED INSTANT BECOMES TRUE. It
never relabels a correct row. Applying the opposite reading -- treating the true-UTC
rows as though their wall clock were already ET -- turns 156 Thursday, Sunday and
Monday NIGHT games into phantom Friday 00:15-to-01:30 ET kickoffs, each of which
would then be fenced at a Friday 18:00 ET freeze roughly EIGHTEEN HOURS AFTER its
real kickoff. That is a leak two orders of magnitude larger than the Black-Friday
leak this remediation exists to close.

WHAT IS ACTUALLY WRONG, corrected against the review. The review attributes the
defect to the ingest path producing a naive Timestamp that reaches storage as
naive-labelled-UTC. That is NOT true of the current code:
``GameSchema.validate_timestamps`` (``data/schemas.py:98-102``) ET-localizes any
naive value, and feeding the exact shape ``scripts/ingest_games.py:221-223`` produces
through ``GameSchema`` yields the true-UTC form. THE WRITE PATH IS ALREADY CORRECT.
The defect is entirely 1,926 stale rows written by an older validator in a single
2025-09-28 ingest run -- demonstrated by a 2024-Week-1 re-ingest seventeen minutes
later in the same season producing true UTC. ``scripts/ingest_games.py`` is NOT
changed by this tool.

BOTH COPIES OR NEITHER. ``load_dataframe(..., source="auto")`` resolves DuckDB FIRST
(``data/storage.py:916-921``) and ``db.table_exists("games")`` is True, so a fix
applied only to the parquet is invisible to the entire pipeline. A fix applied only
to DuckDB leaves the parquet as a divergent fallback that activates the moment the
table is dropped. This tool writes both, inside one invariant-guarded run.

WHAT IT DELIBERATELY DOES NOT DO: it never re-syncs the 207 missing 2025 rows into
the DuckDB copy. That would change the gold row set and confound the corrected
line-movement screen. Row coverage comes through byte-unchanged at 6,499 parquet /
6,292 DuckDB, and the coverage divergence is surfaced as a separate open finding.

Two mutually exclusive modes, mirroring ``scripts/rekey_odds_timeline_2020.py``:

* ``--dry-run`` -- computes the shift and asserts every invariant ``--apply`` will
  enforce, writing nothing.
* ``--apply`` -- rewrites both copies through the shared atomic write tail.

The backup is ON by default (``<games.parquet>.bak-<UTC timestamp>``) and skipping it
requires an explicit ``--no-backup``.

This tool makes no Odds API call of any kind and never touches ``odds_snapshot`` or
``odds_timeline``.
"""

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from conf.settings import get_settings
from data.storage import (
    ParquetManager,
    _atomic_write_parquet,
    get_db_connection,
    load_dataframe,
)
from utils import get_logger
from utils.date_utils import ET

logger = get_logger(__name__)

# The single ingest run that wrote ET wall clocks mislabelled as UTC. Identified by
# its created_at stamp: the 14:47:59 run is shifted, the 15:04:05 run seventeen
# minutes later in the same season is not.
SHIFTED_COHORT_CREATED_AT = pd.Timestamp("2025-09-28T14:47:59.211678Z")

# The state this tool was built against, measured 2026-08-17. A mismatch means the
# data is not in the state the quick task was planned against, and the tool refuses
# to write rather than guess.
EXPECTED_SHIFTED_ROWS = 1926
EXPECTED_PARQUET_ROWS = 6499
EXPECTED_DUCKDB_ROWS = 6292

# The named regression anchor: the 2023 Black Friday game whose in-play line is the
# leak this whole remediation exists to close. After normalization its ET wall clock
# must read Friday 2023-11-24 15:00.
ANCHOR_GAME_ID = "2023_W12_MIA@NYJ"
ANCHOR_ET_WALL_CLOCK = "2023-11-24 15:00"

# The densest kickoff window, used as the DST-correlation probe.
_EARLY_WINDOW_ET_HOUR = 13


class NormalizeInvariantError(RuntimeError):
    """Raised when an invariant is breached. Both copies are left untouched."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        logger.error("Kickoff-tz normalization invariant breached", error=message)
        raise NormalizeInvariantError(message)


def games_parquet_path(base_path: Path | None = None) -> Path:
    """Return the path of the silver games parquet."""
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    return root / "silver" / "games.parquet"


def default_backup_path(base_path: Path | None = None) -> Path:
    """Return ``<games.parquet>.bak-<UTC timestamp>`` -- the default backup target."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = games_parquet_path(base_path)
    return path.with_suffix(path.suffix + f".bak-{stamp}")


def shifted_mask(df: pd.DataFrame) -> pd.Series:
    """Return the boolean mask of rows belonging to the stale ingest cohort."""
    created = pd.to_datetime(df["created_at"], utc=True)
    return created == SHIFTED_COHORT_CREATED_AT


def cohort_is_stale(df: pd.DataFrame) -> bool:
    """Return True when the stale cohort's kickoffs still need shifting.

    IDEMPOTENCY, and why membership is the wrong test. The normalization shifts
    ``kickoff_et`` and deliberately does NOT touch ``created_at``, so the cohort
    still has its 1,926 rows AFTER a successful run. Keying "already normalized" on
    cohort membership would therefore never fire, and a second ``--apply`` would
    shift the same rows a SECOND time -- silently pushing every one of them four or
    five hours past its real kickoff. (The first ``--apply`` run failed its
    post-write assertion for exactly this reason: it asserted the cohort was empty,
    which it never is.)

    The staleness test is the DST correlation restricted to the cohort. A correctly
    stored cohort shifts its UTC hour by one across the November boundary for a fixed
    ET wall-clock window; a cohort still holding ET wall clocks labelled UTC does not,
    and its 1 PM ET probe band is empty because those games read 08:00/09:00 ET.
    """
    cohort = df[shifted_mask(df)]
    if cohort.empty:
        return False
    correlation = dst_correlation_holds(cohort)
    if not correlation:
        return True
    return not all(correlation.values())


def normalize_kickoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with the stale cohort's kickoff INSTANTS corrected.

    The stale rows carry an ET wall clock labelled UTC. The correct instant is that
    wall clock READ AS ET, i.e. shifted forward by the ET offset in force on that
    date (4 hours EDT, 5 hours EST). Doing it per row via ``tz_localize`` is what
    makes the DST boundary come out right; a fixed ``+4h`` would corrupt every
    December and January game by an hour.

    THE MISLABELLED WALL CLOCK IS THE **UTC** ONE, so the column is converted to UTC
    BEFORE its label is dropped. This matters because the two silver copies are typed
    differently: the parquet is UTC-typed and the DuckDB table is
    ``America/New_York``-typed, yet they hold the SAME instants. Reading each copy's
    own local wall clock would shift the parquet by five hours and leave DuckDB
    untouched, silently splitting two copies that agree today. (The instants-agree
    invariant in :func:`prepare` catches exactly that, and did.)

    Correct rows are returned untouched, and the kickoff DATE is preserved for every
    row in the cohort (the minimum stale ET hour is 08:00, so a forward shift of 4-5
    hours never crosses midnight).
    """
    out = df.copy()
    mask = shifted_mask(out)
    if not mask.any() or not cohort_is_stale(out):
        # Already correct: shifting again would push every cohort row four or five
        # hours PAST its real kickoff. The no-op is what makes --apply re-runnable.
        return out

    column = pd.to_datetime(out["kickoff_et"])
    original_tz = column.dt.tz  # preserved, so the column's dtype does not change

    # Read the UTC wall clock -- that is the mislabelled one, in BOTH copies.
    in_utc = column[mask] if original_tz is None else column[mask].dt.tz_convert(UTC)

    # Drop the (wrong) label, then re-attach ET to the SAME wall clock -- that is
    # what turns "13:00 labelled UTC" into "13:00 ET", i.e. 17:00/18:00 UTC.
    wall_clock = in_utc.dt.tz_localize(None)
    corrected = wall_clock.dt.tz_localize(
        ET, ambiguous=True, nonexistent="shift_forward"
    )
    if original_tz is not None:
        corrected = corrected.dt.tz_convert(original_tz)

    column.loc[mask] = corrected
    out["kickoff_et"] = column
    return out


def _et_wall_clock(series: pd.Series) -> pd.Series:
    """Return the ET wall clock of a kickoff column, whatever tz it is typed in."""
    values = pd.to_datetime(series)
    if values.dt.tz is None:
        return values
    return values.dt.tz_convert(ET).dt.tz_localize(None)


def dst_correlation_holds(df: pd.DataFrame) -> dict[str, bool]:
    """Per-season: does the STORED hour shift across the November DST boundary?

    The decisive discriminator (RESEARCH A.1). For a fixed ET wall-clock window a
    true UTC instant must shift by one hour across the EDT-to-EST boundary; an ET
    wall clock mislabelled as UTC does not. Reported per season so a single surviving
    stale cohort cannot hide inside an aggregate.

    The probe is the 1 PM ET early window, which is the densest. An EMPTY band is
    reported as FAILING, never skipped: a regular season with no 1 PM ET kickoffs at
    all is not a season with an unusual schedule, it is a season whose wall clocks are
    wrong -- the stale cohort's 1 PM ET games read 08:00/09:00 ET, which is not an NFL
    kickoff time at a domestic venue. Skipping the empty band is what made an earlier
    version of this check return a vacuous all-clear on the pre-fix data.
    """
    values = pd.to_datetime(df["kickoff_et"], utc=True)
    et_wall = _et_wall_clock(df["kickoff_et"])
    et_hour = et_wall.dt.hour
    stored_hour = values.dt.hour
    month = et_wall.dt.month

    result: dict[str, bool] = {}
    for season, group_idx in df.groupby("season").groups.items():
        idx = list(group_idx)
        has_edt = any(month.loc[i] in (9, 10) for i in idx)
        has_est = any(month.loc[i] in (12, 1, 2) for i in idx)
        if not (has_edt and has_est):
            # A partial season (e.g. an in-progress one) cannot answer the question.
            continue

        early = [i for i in idx if et_hour.loc[i] == _EARLY_WINDOW_ET_HOUR]
        edt = {stored_hour.loc[i] for i in early if month.loc[i] in (9, 10)}
        est = {stored_hour.loc[i] for i in early if month.loc[i] in (12, 1, 2)}
        if not edt or not est:
            result[str(season)] = False
            continue
        result[str(season)] = min(est) == min(edt) + 1
    return result


def prepare(base_path: Path | None = None) -> dict:
    """Compute the corrected frames and assert every invariant. Writes nothing."""
    parquet_path = games_parquet_path(base_path)
    _assert(parquet_path.exists(), f"{parquet_path} does not exist")

    parquet_before = pd.read_parquet(parquet_path, engine="pyarrow")

    db = get_db_connection()
    has_duckdb = db.table_exists("games")
    duckdb_before = db.fetch_df("SELECT * FROM games") if has_duckdb else None

    shifted_parquet = int(shifted_mask(parquet_before).sum())
    shifted_duckdb = (
        int(shifted_mask(duckdb_before).sum()) if duckdb_before is not None else 0
    )
    # Staleness, NOT cohort membership -- see cohort_is_stale for why.
    parquet_stale = cohort_is_stale(parquet_before)
    duckdb_stale = duckdb_before is not None and cohort_is_stale(duckdb_before)
    already_normalized = not parquet_stale and not duckdb_stale

    # --- pre-assertions ------------------------------------------------
    if not already_normalized:
        _assert(
            shifted_parquet == EXPECTED_SHIFTED_ROWS,
            f"the parquet holds {shifted_parquet} rows in the stale cohort, "
            f"expected {EXPECTED_SHIFTED_ROWS}",
        )
        _assert(
            len(parquet_before) == EXPECTED_PARQUET_ROWS,
            f"the parquet holds {len(parquet_before)} rows, expected "
            f"{EXPECTED_PARQUET_ROWS}",
        )
        if duckdb_before is not None:
            _assert(
                shifted_duckdb == EXPECTED_SHIFTED_ROWS,
                f"the DuckDB copy holds {shifted_duckdb} rows in the stale cohort, "
                f"expected {EXPECTED_SHIFTED_ROWS}",
            )
            _assert(
                len(duckdb_before) == EXPECTED_DUCKDB_ROWS,
                f"the DuckDB copy holds {len(duckdb_before)} rows, expected "
                f"{EXPECTED_DUCKDB_ROWS}",
            )

    parquet_after = normalize_kickoffs(parquet_before)
    duckdb_after = (
        normalize_kickoffs(duckdb_before) if duckdb_before is not None else None
    )

    # --- post-assertions (on the in-memory result) ---------------------
    _assert(
        len(parquet_after) == len(parquet_before),
        "the parquet row count moved; refusing to write",
    )
    _assert(
        set(parquet_after["game_id"]) == set(parquet_before["game_id"]),
        "the parquet game_id set moved; refusing to write",
    )
    _assert(
        int(shifted_mask(parquet_after).sum()) == shifted_parquet,
        "the created_at cohort itself was modified; refusing to write",
    )
    # The kickoff DATE is preserved for every row -- the shift never crosses midnight.
    _assert(
        (
            _et_wall_clock(parquet_after["kickoff_et"]).dt.date
            == _et_wall_clock(parquet_before["kickoff_et"]).dt.date
        ).all(),
        "a kickoff DATE moved; the shift crossed midnight, which it must never do",
    )

    if duckdb_after is not None:
        _assert(
            len(duckdb_after) == len(duckdb_before),
            "the DuckDB row count moved; refusing to write",
        )
        _assert(
            set(duckdb_after["game_id"]) == set(duckdb_before["game_id"]),
            "the DuckDB game_id set moved; refusing to write",
        )
        # The two copies must agree on kickoff INSTANTS for every shared row (N-01).
        pq_inst = pd.to_datetime(
            parquet_after.set_index("game_id")["kickoff_et"], utc=True
        )
        db_inst = pd.to_datetime(
            duckdb_after.set_index("game_id")["kickoff_et"], utc=True
        )
        shared = pq_inst.index.intersection(db_inst.index)
        disagreements = int((pq_inst.loc[shared] != db_inst.loc[shared]).sum())
        _assert(
            disagreements == 0,
            f"{disagreements} shared rows disagree on the kickoff instant after "
            "normalization; refusing to write",
        )

    # The named anchor reads Friday 15:00 ET.
    anchor_rows = parquet_after[parquet_after["game_id"] == ANCHOR_GAME_ID]
    anchor_wall_clock = None
    if len(anchor_rows) == 1:
        anchor_wall_clock = _et_wall_clock(anchor_rows["kickoff_et"]).iloc[0]
        _assert(
            anchor_wall_clock.strftime("%Y-%m-%d %H:%M") == ANCHOR_ET_WALL_CLOCK
            and anchor_wall_clock.strftime("%a") == "Fri",
            f"{ANCHOR_GAME_ID} reads {anchor_wall_clock} ET after normalization, "
            f"expected {ANCHOR_ET_WALL_CLOCK} (a Friday)",
        )

    dst_before = dst_correlation_holds(parquet_before)
    dst_after = dst_correlation_holds(parquet_after)
    failing_after = sorted(season for season, ok in dst_after.items() if not ok)
    _assert(
        not failing_after,
        f"the DST correlation still fails for seasons {failing_after} after "
        "normalization; a stale cohort survived",
    )

    return {
        "already_normalized": already_normalized,
        "shifted_parquet": shifted_parquet,
        "shifted_duckdb": shifted_duckdb,
        "parquet_rows": len(parquet_after),
        "duckdb_rows": len(duckdb_after) if duckdb_after is not None else 0,
        "has_duckdb": has_duckdb,
        "anchor_et": str(anchor_wall_clock) if anchor_wall_clock is not None else None,
        "dst_seasons_failing_before": sorted(
            season for season, ok in dst_before.items() if not ok
        ),
        "dst_seasons_failing_after": failing_after,
        "parquet_after": parquet_after,
        "duckdb_after": duckdb_after,
    }


def apply(base_path: Path | None = None) -> dict:
    """Normalize BOTH copies, then verify through ``load_dataframe``."""
    report = prepare(base_path)

    if report["already_normalized"]:
        print(
            "games.kickoff_et is already normalized: the cohort's kickoffs are "
            "true instants. Nothing written. (The created_at cohort itself keeps "
            "its rows -- this tool shifts kickoff_et, never created_at.)"
        )
        report["written"] = False
        return report

    parquet_path = games_parquet_path(base_path)
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )

    pm = ParquetManager(str(root))
    normalized = pm._normalize_parquet_datetime_columns(report["parquet_after"])
    _atomic_write_parquet(pa.Table.from_pandas(normalized), parquet_path)

    if report["duckdb_after"] is not None:
        db = get_db_connection()
        db.create_table_from_df(report["duckdb_after"], "games", if_exists="replace")

    report["written"] = True

    # The decisive check goes THROUGH load_dataframe, not a direct parquet read, so
    # it proves what the feature builders will actually see.
    reloaded = load_dataframe("games", layer="silver")
    report["reloaded_dtype"] = str(pd.to_datetime(reloaded["kickoff_et"]).dtype)
    report["reloaded_rows"] = len(reloaded)
    _assert(
        not cohort_is_stale(reloaded),
        "the cohort still reads as stale after the write -- the shift did not take",
    )
    _assert(
        int(shifted_mask(reloaded).sum()) == report["shifted_duckdb"]
        or int(shifted_mask(reloaded).sum()) == report["shifted_parquet"],
        "the created_at cohort itself changed size across the write",
    )
    failing = sorted(
        season for season, ok in dst_correlation_holds(reloaded).items() if not ok
    )
    _assert(
        not failing,
        f"the reloaded copy still fails the DST correlation for seasons {failing}",
    )
    return report


def _print_report(report: dict, mode: str) -> None:
    print(f"GAMES KICKOFF-TZ NORMALIZATION ({mode})")
    print(f"  already normalized:      {report['already_normalized']}")
    print(f"  stale rows (parquet):    {report['shifted_parquet']}")
    print(f"  stale rows (duckdb):     {report['shifted_duckdb']}")
    print(f"  parquet rows:            {report['parquet_rows']}")
    print(f"  duckdb rows:             {report['duckdb_rows']}")
    print(f"  {ANCHOR_GAME_ID} ET:  {report['anchor_et']}")
    print(f"  DST failing before:      {report['dst_seasons_failing_before']}")
    print(f"  DST failing after:       {report['dst_seasons_failing_after']}")
    if "reloaded_dtype" in report:
        print(f"  reloaded dtype:          {report['reloaded_dtype']}")
        print(f"  reloaded rows:           {report['reloaded_rows']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the stale games.kickoff_et cohort to true UTC in BOTH silver "
            "copies. Makes no Odds API call of any kind."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the shift and assert every --apply invariant, writing nothing.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite both the parquet and the DuckDB games copy.",
    )
    parser.add_argument(
        "--backup-to",
        type=Path,
        help="Copy games.parquet here before --apply writes. Defaults to "
        "<games.parquet>.bak-<UTC timestamp>.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the pre-write backup. Must be requested explicitly.",
    )
    return parser


def main() -> None:
    """CLI entry point for the kickoff-tz normalization."""
    args = build_parser().parse_args()

    try:
        if args.dry_run:
            _print_report(prepare(), mode="dry run -- nothing written")
            sys.exit(0)

        if not args.no_backup:
            backup_to = (
                args.backup_to if args.backup_to is not None else default_backup_path()
            )
            backup_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(games_parquet_path(), backup_to)
            print(f"Backed up games.parquet to {backup_to}")
        else:
            print("Skipping the pre-write backup (--no-backup was requested)")

        report = apply()
        _print_report(report, mode="applied" if report["written"] else "no-op")
        sys.exit(0)

    except NormalizeInvariantError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
