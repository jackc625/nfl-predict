"""One-shot, declared-write repair of the 68 2002-2005 night kickoffs in silver ``games``.

Plan 33.2-12, p332_ extra step 3c (orchestrator-assigned; deferred-items entry found by
Plan 33.2-10).

WHY THIS EXISTS
---------------
Every 2002-2005 Monday and Thursday night game was stored at 09:00 ET, because the feed's
``gametime`` is a 12-hour AM/PM error. The date is right, so no lock moves; the HOUR is
wrong, and it feeds rest-day counts and the day-before forecast hour.

THE STORE HALF OF A TWO-HALF FIX
--------------------------------
The ingest half: ``scripts/ingest_games.py`` applies ``config/kickoff_hour_corrections.toml``
to every row through ``resolve_kickoff_gametime``. This script calls THE SAME FUNCTION with
each recorded game's CURRENT silver clock, and builds the new ``kickoff_et`` exactly as the
ingest does (``gameday + " " + clock`` read as America/New_York, stored in UTC), so the
store and the next ingest of 2002-2005 write the same instant.

IDEMPOTENT
----------
A game whose silver clock is the recorded wrong one is corrected; one already carrying the
correction is counted and NOT rewritten; any other value raises
``KickoffCorrectionDriftError`` and nothing is written. ``CORRECTED=`` and
``ALREADY_CORRECTED=`` sum to 68.

WRITES
------
``--dry-run`` (the default) writes nothing. ``--apply`` writes exactly
``silver/games.parquet`` (one atomic full-table write through ``upsert_silver``, every row
in its original position) and, through that writer, the DuckDB ``games`` copy. Only
``kickoff_et`` of the recorded games changes; the post-state is asserted on the frame
before it is written and on the table read back.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.storage import upsert_silver
from scripts.ingest_games import (
    KickoffHourCorrection,
    load_kickoff_hour_corrections,
    resolve_kickoff_gametime,
)

GAMES_TABLE = "games"
KICKOFF_COLUMN = "kickoff_et"
ET = "America/New_York"


class KickoffRepairPostStateError(AssertionError):
    """The repaired table is not exactly the declared change. Nothing is trusted."""


@dataclass(frozen=True)
class KickoffRepairPlan:
    """What a run would do: the repaired frame and the per-game disposition."""

    before: pd.DataFrame
    after: pd.DataFrame
    corrected: tuple[str, ...]
    already_corrected: tuple[str, ...]


def load_games(data_root: Path) -> pd.DataFrame:
    """The silver ``games`` parquet under *data_root*, in its stored row order."""
    return pd.read_parquet(data_root / "silver" / f"{GAMES_TABLE}.parquet")


def et_wall_clock(kickoff: pd.Timestamp) -> tuple[str, str]:
    """``(gameday, HH:MM)`` of a stored UTC kickoff, read on the Eastern wall clock."""
    local = kickoff.tz_convert(ET)
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M")


def kickoff_instant(gameday: str, clock: str, like: pd.Series) -> pd.Timestamp:
    """The stored instant for an Eastern ``gameday clock``, in *like*'s UTC resolution."""
    instant = pd.Timestamp(f"{gameday} {clock}").tz_localize(ET).tz_convert("UTC")
    return pd.Series([instant]).astype(like.dtype).iloc[0]


def plan_repair(
    games: pd.DataFrame, corrections: Mapping[str, KickoffHourCorrection]
) -> KickoffRepairPlan:
    """Resolve every recorded game against its CURRENT silver clock.

    Raises:
        KickoffRepairPostStateError: a recorded game is absent from silver or appears twice.
        KickoffCorrectionDriftError: a recorded game's silver date or clock is neither the
            recorded wrong value nor the correction.
    """
    after = games.copy()
    corrected: list[str] = []
    already: list[str] = []
    for game_id in sorted(corrections):
        positions = after.index[after["game_id"] == game_id]
        if len(positions) != 1:
            raise KickoffRepairPostStateError(
                f"{game_id} appears {len(positions)} time(s) in silver games; the record "
                "names games that must each exist exactly once."
            )
        position = positions[0]
        gameday, clock = et_wall_clock(after.loc[position, KICKOFF_COLUMN])
        resolved = resolve_kickoff_gametime(
            game_id, gameday, clock, corrections=corrections
        )
        if resolved == clock:
            already.append(game_id)
            continue
        after.loc[position, KICKOFF_COLUMN] = kickoff_instant(
            gameday, resolved, after[KICKOFF_COLUMN]
        )
        corrected.append(game_id)
    return KickoffRepairPlan(games, after, tuple(corrected), tuple(already))


def assert_post_state(
    before: pd.DataFrame,
    after: pd.DataFrame,
    corrections: Mapping[str, KickoffHourCorrection],
) -> list[str]:
    """Exactly the declared change, or refuse. Returns the game ids whose rows changed.

    * same rows, same order, same columns and dtypes;
    * no column other than ``kickoff_et`` moved on any row;
    * every recorded game reads its corrected Eastern clock on its recorded date;
    * every row that changed is a recorded game, and it moved by exactly +12 hours.
    """
    if list(before.columns) != list(after.columns) or len(before) != len(after):
        raise KickoffRepairPostStateError("the table's shape changed")
    if list(before["game_id"]) != list(after["game_id"]):
        raise KickoffRepairPostStateError("the table's row order changed")
    for column in before.columns:
        if before[column].dtype != after[column].dtype:
            raise KickoffRepairPostStateError(f"column {column!r} changed dtype")
        if column != KICKOFF_COLUMN and not before[column].equals(after[column]):
            raise KickoffRepairPostStateError(
                f"column {column!r} moved; only {KICKOFF_COLUMN!r} may change"
            )

    indexed = after.set_index("game_id")[KICKOFF_COLUMN]
    for game_id, correction in corrections.items():
        if et_wall_clock(indexed[game_id]) != (
            correction.gameday,
            correction.corrected_gametime,
        ):
            raise KickoffRepairPostStateError(
                f"{game_id} reads {et_wall_clock(indexed[game_id])}, not "
                f"{(correction.gameday, correction.corrected_gametime)}"
            )

    moved = before[KICKOFF_COLUMN] != after[KICKOFF_COLUMN]
    for position in before.index[moved]:
        game_id = before.loc[position, "game_id"]
        if game_id not in corrections:
            raise KickoffRepairPostStateError(f"{game_id} changed but is not recorded")
        shift = (
            after.loc[position, KICKOFF_COLUMN] - before.loc[position, KICKOFF_COLUMN]
        )
        if shift != pd.Timedelta(hours=12):
            raise KickoffRepairPostStateError(
                f"{game_id} moved by {shift}, not the +12h an AM/PM correction is"
            )
    return list(before.loc[moved, "game_id"])


def write_games_table(data_root: Path, repaired: pd.DataFrame) -> Path:
    """One atomic full-table write. Every stored game_id is in *repaired*, so the key
    replacement in ``upsert_silver`` replaces the whole table with this frame, in order."""
    return upsert_silver(
        repaired.reset_index(drop=True), GAMES_TABLE, base_path=data_root
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct the 68 2002-2005 night kickoffs in silver games."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="perform the declared write")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    corrections = load_kickoff_hour_corrections()
    before = load_games(args.data_root)
    plan = plan_repair(before, corrections)
    changed = assert_post_state(before, plan.after, corrections)
    if sorted(changed) != sorted(plan.corrected):
        raise KickoffRepairPostStateError(
            f"changed rows {sorted(changed)} differ from corrected games "
            f"{sorted(plan.corrected)}"
        )

    print(f"RECORDED= {len(corrections)}")
    print(f"CORRECTED= {len(plan.corrected)}")
    print(f"ALREADY_CORRECTED= {len(plan.already_corrected)}")
    if not args.apply:
        print("MODE= dry-run (nothing written)")
        return 0

    if plan.corrected:
        write_games_table(args.data_root, plan.after)
        stored = load_games(args.data_root)
        assert_post_state(before, stored, corrections)
        print(f"ROWS_WRITTEN= {len(stored)} (read back and re-asserted)")
    print("MODE= apply")
    return 0


if __name__ == "__main__":
    sys.exit(run())
