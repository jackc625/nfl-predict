"""Re-key the ``odds_timeline`` rows a retired week count filed under the wrong week.

Plan 33.2-24 step 24b (orchestrator-assigned, before the Plan 33.2-25 swap). The owner's
standing rule decides it: a known data bug is fixed at the root, in the same work -- the
ingest AND the stored rows AND everything fitted on them.

WHAT WAS WRONG
--------------
``scripts/ingest_odds_timeline`` keyed each captured event by COUNTING whole weeks from the
season's opening Thursday to the event's ``commence_time``. A game played before the
Thursday of its own week landed in the previous week. In the paid archive that hit exactly
two games -- the Wednesday 2024 Christmas games -- whose lines are stored under
``2024_W16_KC@PIT`` / ``2024_W16_BAL@HOU``, ids no scheduled game carries, while silver
``games`` has them in week 17. Both of the owned timeline's readers (the converter fit and
the blend corpus) therefore never saw them, although both games carry lines captured before
their locks. The ingest is fixed in the same step (every event is now keyed to its
scheduled game); this tool repairs the rows the old key already wrote.

HOW A MISFILED ROW IS TOLD APART FROM A ROW THAT NAMES NO GAME
--------------------------------------------------------------
Measured, never assumed. The archive stores no ``commence_time``, so the classification
is an EXACT RE-DERIVATION, the method quick task 260816-u0e used for the 2020 archive: the
retired week count is replayed FORWARD over every scheduled game's kickoff, producing the
id that game WOULD have been stored under, and that map is inverted. An unjoined stored id
found in it is MISFILED and is re-keyed to the scheduled game it came from. An unjoined id
not found in it names no scheduled game at all -- the suspended ``2022_W17_BUF@CIN`` and the
speculative 2024 wild-card pairings priced before week 18 settled -- and is left exactly as
it is.

:func:`retired_week_count` is a FROZEN REPLAY OF A FIXED BUG. It must never be called by
production code and must never be "corrected": correcting it would stop it matching the
stored keys, which is the only thing it is for.

WHY A WHOLE-TABLE REWRITE
-------------------------
``data.storage.upsert_silver_composite`` is insert-or-update on ``(game_id, snapshot_ts)``
with no delete path, so upserting re-keyed rows would leave the wrong-keyed originals in
place beside them. The table is read, the misfiled rows are re-keyed IN PLACE (same row
positions, every other column untouched), every invariant is asserted, and the result is
written through the storage module's own write tail -- ``_canonicalize_snapshot_ts_utc``,
``ParquetManager._normalize_parquet_datetime_columns``, ``_atomic_write_parquet`` and
``_keep_duckdb_copy_in_step`` -- the same four calls ``upsert_silver_composite`` makes.

It never imports or calls the Odds API client: the archive cost real credits and nothing is
re-bought. ``odds_snapshot`` is never read or written. Bronze is append-only raw record and
is left as captured.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa

from conf.settings import get_settings
from data.storage import (
    ParquetManager,
    _atomic_write_parquet,
    _canonicalize_snapshot_ts_utc,
    _keep_duckdb_copy_in_step,
)
from utils import get_logger
from utils.date_utils import get_nfl_season_start, kickoff_wall_clock_et
from utils.game_id_utils import create_standard_game_id

logger = get_logger(__name__)

TABLE = "odds_timeline"
KEY_COLUMNS = ["game_id", "snapshot_ts"]

# The highest week a game id can carry. The retired count REFUSED anything outside
# 1..22, so a scheduled game whose replayed week falls outside it was never stored under
# any key and has nothing to re-key.
_MAX_STORED_WEEK = 22

# The production archive this tool was built against, MEASURED 2026-09-23. A mismatch
# means the archive is not in the state the step was planned against, and --apply refuses
# rather than guessing. (A second --apply finds nothing misfiled and is a no-op.)
EXPECTED_TOTAL_ROWS = 9957
EXPECTED_REKEY_MAP: dict[str, str] = {
    "2024_W16_BAL@HOU": "2024_W17_BAL@HOU",
    "2024_W16_KC@PIT": "2024_W17_KC@PIT",
}
EXPECTED_ROWS_REKEYED = 16
EXPECTED_NO_GAME_IDS: tuple[str, ...] = (
    "2022_W17_BUF@CIN",
    "2024_W19_DET@LA",
    "2024_W19_LAC@BAL",
    "2024_W19_PIT@HOU",
)

# Where the pre-write copy goes: outside data/, so the digest bracket over data/ sees the
# one declared rewrite and nothing else.
DEFAULT_BACKUP = Path("outputs") / "p332_24b_odds_timeline_before.parquet"


class MisfiledRekeyError(RuntimeError):
    """A re-key invariant was breached. The archive is left untouched."""


def retired_week_count(kickoff_et: object) -> tuple[int, int]:
    """``(season, week)`` exactly as the RETIRED ingest counted it from a kickoff.

    FROZEN REPLAY -- see the module docstring. Whole weeks elapsed since the season's
    opening Thursday (``get_nfl_season_start``, the corrected Thursday-after-Labor-Day
    rule the ingest used from ``45bff24`` until step 24b), plus one; a January or February
    kickoff belongs to the previous calendar year's season.
    """
    kickoff = kickoff_wall_clock_et(kickoff_et)
    season = kickoff.year if kickoff.month >= 8 else kickoff.year - 1
    return season, (kickoff - get_nfl_season_start(season)).days // 7 + 1


@dataclass(frozen=True)
class Classification:
    """Every stored id that joins no scheduled game, sorted into its two kinds."""

    rekey_map: dict[str, str] = field(default_factory=dict)
    no_game_ids: tuple[str, ...] = ()


def classify_unjoined_ids(
    timeline: pd.DataFrame, games: pd.DataFrame
) -> Classification:
    """Sort the stored ids that join no scheduled game into misfiled and no-game.

    Raises:
        MisfiledRekeyError: when the replay is ambiguous (two scheduled games would have
            been stored under one id) or would have produced a REAL scheduled game's id
            for a different game -- in either case the stored rows cannot be attributed.
    """
    scheduled = set(games["game_id"].astype(str))
    replay: dict[str, str] = {}
    for row in games.itertuples(index=False):
        season, week = retired_week_count(row.kickoff_et)
        if not 1 <= week <= _MAX_STORED_WEEK:
            continue
        derived = create_standard_game_id(
            season=season, week=week, away_team=row.away_team, home_team=row.home_team
        )
        if derived == row.game_id:
            continue
        if derived in scheduled:
            raise MisfiledRekeyError(
                f"the retired count keys {row.game_id} as {derived}, which is ANOTHER "
                "scheduled game's id; rows stored there cannot be attributed to either"
            )
        if replay.get(derived, row.game_id) != row.game_id:
            raise MisfiledRekeyError(
                f"the retired count keys both {replay[derived]} and {row.game_id} as "
                f"{derived}; the stored rows cannot be attributed to one of them"
            )
        replay[derived] = str(row.game_id)

    unjoined = sorted(set(timeline["game_id"].astype(str)) - scheduled)
    return Classification(
        rekey_map={gid: replay[gid] for gid in unjoined if gid in replay},
        no_game_ids=tuple(gid for gid in unjoined if gid not in replay),
    )


def pair_list_sha256(frame: pd.DataFrame) -> str:
    """sha256 of the sorted ``(game_id, snapshot_ts)`` pair list.

    The SAME expression ``tests/integration/test_odds_timeline_archive_integrity.py``
    pins the archive with (the Phase-30 baseline), so the before and after values this
    tool prints are directly comparable with that record.
    """
    pairs = (
        frame[KEY_COLUMNS]
        .drop_duplicates()
        .assign(snapshot_ts=lambda f: f["snapshot_ts"].astype(str))
        .sort_values(KEY_COLUMNS)
    )
    joined = "\n".join(
        f"{game_id}\x1f{snapshot_ts}"
        for game_id, snapshot_ts in zip(
            pairs["game_id"], pairs["snapshot_ts"], strict=True
        )
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def content_sha256(frame: pd.DataFrame) -> str:
    """sha256 over every row's values, in stored row order, index excluded."""
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise MisfiledRekeyError(message)


@dataclass(frozen=True)
class RepairReport:
    """What the repair found and would write (or wrote). Every value is measured."""

    classification: Classification
    rows_rekeyed: int
    before: pd.DataFrame
    after: pd.DataFrame
    pair_list_sha256_before: str
    pair_list_sha256_after: str
    content_sha256_before: str
    content_sha256_after: str
    untouched_content_sha256: str

    @property
    def already_repaired(self) -> bool:
        return not self.classification.rekey_map


def prepare_repair(timeline: pd.DataFrame, games: pd.DataFrame) -> RepairReport:
    """Classify, re-key in memory, and assert every invariant. Writes nothing.

    Raises:
        MisfiledRekeyError: on any breached invariant.
    """
    before = timeline.reset_index(drop=True)
    classification = classify_unjoined_ids(before, games)
    rekey_map = classification.rekey_map

    mask = before["game_id"].isin(list(rekey_map))
    after = before.copy()
    after.loc[mask, "game_id"] = after.loc[mask, "game_id"].map(rekey_map)

    # No re-keyed id may already carry rows: a merge onto an existing trajectory would
    # need a decision about which capture to keep, and that is not this tool's to make.
    targets_present = sorted(set(rekey_map.values()) & set(before["game_id"]))
    _assert(
        not targets_present,
        f"re-key targets already carry rows in the archive: {targets_present}",
    )
    _assert(
        len(after) == len(before),
        f"the row count moved {len(before)} -> {len(after)}",
    )
    _assert(
        len(after.drop_duplicates(subset=KEY_COLUMNS)) == len(after),
        "a re-keyed (game_id, snapshot_ts) pair collided with another row",
    )

    # Every re-keyed row is a board listing of an UPCOMING game: captured before the
    # scheduled kickoff of the game it is re-keyed to.
    kickoffs = pd.to_datetime(
        games.set_index("game_id")["kickoff_et"], utc=True
    ).reindex(after.loc[mask, "game_id"])
    captured = pd.to_datetime(after.loc[mask, "snapshot_ts"], utc=True)
    late = after.loc[mask, "game_id"][captured.to_numpy() >= kickoffs.to_numpy()]
    _assert(
        late.empty,
        f"re-keyed rows captured at or after their game's kickoff: {sorted(set(late))}",
    )

    # Only game_id moves, and only on the misfiled rows.
    untouched_before = content_sha256(before.loc[~mask])
    untouched_after = content_sha256(after.loc[~mask])
    _assert(
        untouched_before == untouched_after,
        "a row that is not misfiled changed",
    )
    others = [c for c in before.columns if c != "game_id"]
    _assert(
        content_sha256(before.loc[mask, others])
        == content_sha256(after.loc[mask, others]),
        "a re-keyed row changed in a column other than game_id",
    )

    return RepairReport(
        classification=classification,
        rows_rekeyed=int(mask.sum()),
        before=before,
        after=after,
        pair_list_sha256_before=pair_list_sha256(before),
        pair_list_sha256_after=pair_list_sha256(after),
        content_sha256_before=content_sha256(before),
        content_sha256_after=content_sha256(after),
        untouched_content_sha256=untouched_before,
    )


def assert_planned_state(report: RepairReport) -> None:
    """Refuse a production --apply unless the archive is the one this step measured."""
    if report.already_repaired:
        return
    _assert(
        len(report.before) == EXPECTED_TOTAL_ROWS,
        f"archive has {len(report.before)} rows, expected {EXPECTED_TOTAL_ROWS}",
    )
    _assert(
        report.classification.rekey_map == EXPECTED_REKEY_MAP,
        f"misfiled map {report.classification.rekey_map} != {EXPECTED_REKEY_MAP}",
    )
    _assert(
        report.rows_rekeyed == EXPECTED_ROWS_REKEYED,
        f"{report.rows_rekeyed} rows to re-key, expected {EXPECTED_ROWS_REKEYED}",
    )
    _assert(
        report.classification.no_game_ids == EXPECTED_NO_GAME_IDS,
        f"no-game ids {report.classification.no_game_ids} != {EXPECTED_NO_GAME_IDS}",
    )


def data_root(base_path: Path | None = None) -> Path:
    """The data root, defaulting to the configured one."""
    if base_path is not None:
        return Path(base_path)
    return Path(get_settings().config.data.root_path)


def load_inputs(base_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored timeline and the silver ``games`` schedule, read directly."""
    silver = data_root(base_path) / "silver"
    timeline = pd.read_parquet(silver / f"{TABLE}.parquet", engine="pyarrow")
    timeline["snapshot_ts"] = pd.to_datetime(timeline["snapshot_ts"], utc=True)
    games = pd.read_parquet(
        silver / "games.parquet",
        columns=["game_id", "season", "week", "home_team", "away_team", "kickoff_et"],
        engine="pyarrow",
    )
    return timeline, games


def write_repaired(after: pd.DataFrame, base_path: Path | None = None) -> Path:
    """Write the repaired table through the storage module's own write tail."""
    root = data_root(base_path)
    path = root / "silver" / f"{TABLE}.parquet"
    canonical = _canonicalize_snapshot_ts_utc(after)
    normalized = ParquetManager(str(root))._normalize_parquet_datetime_columns(
        canonical
    )
    _atomic_write_parquet(pa.Table.from_pandas(normalized), path)
    _keep_duckdb_copy_in_step(TABLE, path, root)
    return path


def apply_repair(
    base_path: Path | None = None,
    *,
    backup_to: Path | None = DEFAULT_BACKUP,
    enforce_planned_state: bool = True,
) -> tuple[RepairReport, bool]:
    """Re-key the misfiled rows and verify the written table. Returns ``(report, written)``."""
    timeline, games = load_inputs(base_path)
    report = prepare_repair(timeline, games)
    if enforce_planned_state:
        assert_planned_state(report)
    if report.already_repaired:
        return report, False

    path = data_root(base_path) / "silver" / f"{TABLE}.parquet"
    if backup_to is not None:
        backup_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_to)
    write_repaired(report.after, base_path)

    reread, _games = load_inputs(base_path)
    _assert(
        pair_list_sha256(reread) == report.pair_list_sha256_after,
        "the written archive does not digest to the prepared pair list",
    )
    _assert(
        not classify_unjoined_ids(reread, games).rekey_map,
        "a misfiled id survived the write",
    )
    return report, True


def _print_report(report: RepairReport, *, mode: str, written: bool) -> None:
    c = report.classification
    print(f"ODDS_TIMELINE MISFILED RE-KEY ({mode})")
    print(f"ROWS= {len(report.before)}")
    print(f"MISFILED= {dict(sorted(c.rekey_map.items()))}")
    print(f"NO_GAME_IDS_LEFT_ALONE= {list(c.no_game_ids)}")
    print(f"ROWS_REKEYED= {report.rows_rekeyed}")
    print(f"PAIR_LIST_SHA256_BEFORE= {report.pair_list_sha256_before}")
    print(f"PAIR_LIST_SHA256_AFTER= {report.pair_list_sha256_after}")
    print(f"CONTENT_SHA256_BEFORE= {report.content_sha256_before}")
    print(f"CONTENT_SHA256_AFTER= {report.content_sha256_after}")
    print(f"UNTOUCHED_ROWS_CONTENT_SHA256= {report.untouched_content_sha256}")
    print(f"WRITTEN= {written}")


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-key the odds_timeline rows the retired week count mis-filed. "
            "Makes no Odds API call of any kind."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Measure; write nothing.")
    mode.add_argument("--apply", action="store_true", help="Rewrite the table.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    try:
        if args.dry_run:
            timeline, games = load_inputs()
            report = prepare_repair(timeline, games)
            assert_planned_state(report)
            _print_report(report, mode="dry run -- nothing written", written=False)
            sys.exit(0)
        report, written = apply_repair()
        _print_report(report, mode="applied" if written else "no-op", written=written)
        sys.exit(0)
    except MisfiledRekeyError as exc:
        logger.error("Misfiled re-key invariant breached", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
