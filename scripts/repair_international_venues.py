"""One-shot, declared-write repair of the seven 2025 international games in silver ``games``.

Plan 33.2-09 Task 2 (SPEC R8 venue half, T-33.2-09-03, T-33.2-09-06).

WHY THIS EXISTS
---------------
The nflverse schedule feed records every 2025 game played outside the United States at the
US home team's own stadium (33.2-RESEARCH.md section 8.3). Silver ``games`` copied that
verbatim, so the Sao Paulo opener sits at SoFi Stadium under an indoor roof (it would get
no weather at all) and the Dublin, London, Berlin and Madrid games sit at Pittsburgh,
Cleveland, New York, Jacksonville, Indianapolis and Miami. Venue, roof, travel and
time-zone features are wrong for all seven, and the weather backfill would fetch a US
airport forecast for a game played in Europe.

THE STORE HALF OF A TWO-HALF FIX
--------------------------------
The ingest half landed first: ``scripts/ingest_games.py`` applies
``config/international_venue_corrections.toml`` to every row it ingests, through
``resolve_venue_override``. This script is the store half and calls THE SAME FUNCTION with
each game's CURRENT silver values as the upstream input, so the store and the next ingest
of 2025 write the same three values. It derives no venue, name or roof of its own.

IDEMPOTENT
----------
A game whose silver ``stadium_id`` is the recorded wrong value is corrected. One already
carrying the correction is counted and NOT rewritten, so a re-run after an interrupted
attempt is a no-op. Any other value raises ``VenueOverrideDriftError`` and nothing is
written. Every run prints ``CORRECTED=`` and ``ALREADY_CORRECTED=``; they sum to 7.

WRITES
------
``--dry-run`` (the default) writes nothing. ``--apply`` writes exactly
``silver/games.parquet`` (one atomic full-table write through ``upsert_silver``, every row
in its original position) and, through that writer, the DuckDB ``games`` copy under the
same root. Only ``stadium_id``, ``venue`` and ``venue_roof`` of the recorded games change;
the post-state is asserted on the frame before it is written and on the table read back.

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
    GameDataIngester,
    VenueOverride,
    load_venue_overrides,
    resolve_venue_override,
)

GAMES_TABLE = "games"
VENUE_COLUMNS: tuple[str, ...] = ("stadium_id", "venue", "venue_roof")
CORRECTED_SEASON = 2025


class VenueRepairPostStateError(AssertionError):
    """The repaired table is not exactly the declared change. Nothing is trusted."""


@dataclass(frozen=True)
class VenueRepairPlan:
    """What a run would do: the repaired frame and the per-game disposition."""

    before: pd.DataFrame
    after: pd.DataFrame
    corrected: tuple[str, ...]
    already_corrected: tuple[str, ...]


def load_games(data_root: Path) -> pd.DataFrame:
    """The silver ``games`` parquet under *data_root*, in its stored row order."""
    return pd.read_parquet(data_root / "silver" / f"{GAMES_TABLE}.parquet")


def usual_home_stadium(games: pd.DataFrame) -> pd.Series:
    """Each (season, home_team)'s MODAL stadium over its NON-neutral home games."""
    home = games[~games["neutral_site"].astype(bool)]
    return home.groupby(["season", "home_team"])["stadium_id"].agg(
        lambda ids: ids.mode().iat[0]
    )


def plan_repair(
    games: pd.DataFrame,
    overrides: Mapping[str, VenueOverride],
    roof_resolver,
) -> VenueRepairPlan:
    """Resolve every recorded game against its CURRENT silver values.

    Raises:
        VenueRepairPostStateError: a recorded game is absent from silver or appears twice.
        VenueOverrideDriftError: a recorded game's silver stadium_id is neither the
            recorded wrong value nor the correction.
    """
    after = games.copy()
    corrected: list[str] = []
    already: list[str] = []
    for game_id, override in sorted(overrides.items()):
        positions = after.index[after["game_id"] == game_id]
        if len(positions) != 1:
            raise VenueRepairPostStateError(
                f"{game_id} appears {len(positions)} time(s) in silver games; the record "
                "names games that must each exist exactly once."
            )
        position = positions[0]
        current = tuple(after.loc[position, list(VENUE_COLUMNS)])
        resolved = resolve_venue_override(
            game_id,
            current[0],
            current[1],
            None,
            roof_resolver=roof_resolver,
            overrides=overrides,
        )
        if current == resolved:
            already.append(game_id)
            continue
        for column, value in zip(VENUE_COLUMNS, resolved, strict=True):
            after.loc[position, column] = value
        corrected.append(override.game_id)
    return VenueRepairPlan(games, after, tuple(corrected), tuple(already))


def assert_post_state(
    before: pd.DataFrame,
    after: pd.DataFrame,
    overrides: Mapping[str, VenueOverride],
    roof_resolver,
) -> list[str]:
    """Exactly the declared change, or refuse. Returns the game ids whose rows changed.

    * same rows, same order, same columns and dtypes;
    * no column outside ``VENUE_COLUMNS`` moved on any row;
    * every recorded game carries the triple ``resolve_venue_override`` returns for it;
    * every row that changed is a recorded game, in 2025, that previously pointed at the
      home team's usual stadium.
    """
    if list(before.columns) != list(after.columns) or len(before) != len(after):
        raise VenueRepairPostStateError("the table's shape changed")
    if list(before["game_id"]) != list(after["game_id"]):
        raise VenueRepairPostStateError("the table's row order changed")
    for column in before.columns:
        if column in VENUE_COLUMNS:
            continue
        if before[column].dtype != after[column].dtype or not before[column].equals(
            after[column]
        ):
            raise VenueRepairPostStateError(
                f"column {column!r} moved; only {list(VENUE_COLUMNS)} may change"
            )

    for game_id, override in overrides.items():
        row = after.loc[after["game_id"] == game_id].iloc[0]
        expected = resolve_venue_override(
            game_id,
            override.old_stadium_id,
            None,
            None,
            roof_resolver=roof_resolver,
            overrides=overrides,
        )
        if tuple(row[list(VENUE_COLUMNS)]) != expected:
            raise VenueRepairPostStateError(
                f"{game_id} carries {tuple(row[list(VENUE_COLUMNS)])!r}, not {expected!r}"
            )

    left, right = before[list(VENUE_COLUMNS)], after[list(VENUE_COLUMNS)]
    same = (left == right) | (left.isna() & right.isna())
    moved = ~same.all(axis=1)
    changed = before.loc[moved]
    usual = usual_home_stadium(before)
    for row in changed.itertuples():
        if row.game_id not in overrides:
            raise VenueRepairPostStateError(
                f"{row.game_id} changed but is not recorded"
            )
        if row.season != CORRECTED_SEASON:
            raise VenueRepairPostStateError(f"{row.game_id} is not a 2025 game")
        if row.stadium_id != usual.get((row.season, row.home_team)):
            raise VenueRepairPostStateError(
                f"{row.game_id} did not previously point at {row.home_team}'s usual "
                "stadium, so it is not the defect this repair corrects"
            )
    return list(changed["game_id"])


def write_games_table(data_root: Path, repaired: pd.DataFrame) -> Path:
    """One atomic full-table write. Every stored game_id is in *repaired*, so the key
    replacement in ``upsert_silver`` replaces the whole table with this frame, in order."""
    return upsert_silver(
        repaired.reset_index(drop=True), GAMES_TABLE, base_path=data_root
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct the seven 2025 international games in silver games."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="perform the declared write")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    overrides = load_venue_overrides()
    roof_resolver = GameDataIngester()._get_venue_roof_type
    before = load_games(args.data_root)
    plan = plan_repair(before, overrides, roof_resolver)
    changed = assert_post_state(before, plan.after, overrides, roof_resolver)
    if sorted(changed) != sorted(plan.corrected):
        raise VenueRepairPostStateError(
            f"changed rows {sorted(changed)} differ from corrected games "
            f"{sorted(plan.corrected)}"
        )

    print(f"RECORDED= {len(overrides)}")
    print(f"CANDIDATES= {len(plan.corrected) + len(plan.already_corrected)}")
    for game_id in sorted(overrides):
        row = before.loc[before["game_id"] == game_id].iloc[0]
        new = plan.after.loc[plan.after["game_id"] == game_id].iloc[0]
        print(
            f"  {game_id}: {tuple(row[list(VENUE_COLUMNS)])} -> "
            f"{tuple(new[list(VENUE_COLUMNS)])}"
        )
    print(f"CORRECTED= {len(plan.corrected)}")
    print(f"ALREADY_CORRECTED= {len(plan.already_corrected)}")

    if not args.apply:
        print("MODE= dry-run (nothing written)")
        return 0

    if plan.corrected:
        write_games_table(args.data_root, plan.after)
        stored = load_games(args.data_root)
        assert_post_state(before, stored, overrides, roof_resolver)
        print(f"ROWS_WRITTEN= {len(stored)} (read back and re-asserted)")
    print("MODE= apply")
    return 0


if __name__ == "__main__":
    sys.exit(run())
