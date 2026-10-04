"""An independent, read-only replay of every pre-game Elo rating (D33.2-06, SPEC R4).

WHY THIS MODULE EXISTS
----------------------
The ``elo`` provenance supplier (Plan 33.2-01) is RULE-derived: it dates each Elo row by
the rule the builder already follows. A rule checked with the same rule proves the rule,
not the content (RESEARCH pitfall P3). This module is the CONTENT evidence. For every game
it recomputes the pre-game ratings from ONLY the results whose kickoff plus a declared
duration is at or before that game's lock (``utils.game_lock``: 18:00 ET on the ET calendar
day before kickoff), and compares them with exact ``==`` -- no tolerance, no rounding --
against what silver ``elo_game_snapshots`` actually holds.

THIS MODULE IS DELIBERATELY INDEPENDENT OF ``scripts/build_elo.py``
--------------------------------------------------------------------
It may not import ``scripts.build_elo`` -- not its chain, not its writers, not even a
column constant, because importing a constant still imports the module and runs its body.
It may not read the four legacy tables (``games_with_elo``, ``elo_rating_history``,
``elo_ratings_current``, ``elo_ratings.json``). And it may not call
``EloRatingSystem.process_season_chronologically`` (a whole-season loop that hands the
ordering back to the library), ``save_ratings`` (which writes ``data/silver/
elo_ratings.json`` -- a production write from a read-only audit) or ``load_ratings``.

A source scan is the only instrument that works here, because the wrong thing RETURNS A
VALUE and therefore leaves a green suite behind it: a replay that imported the builder's
chain would pass every equality assertion it makes, on every row, and prove nothing. The
scan is ``tests/unit/test_elo_replay_independence.py``. It reads the parsed tree across
three families -- imports, legacy table names matched by EQUALITY, and forbidden calls --
and its exclusions are ASSERTED, not assumed: each forbidden name has a planted violation
the scan must flag.

WHAT THE REPLAY OWNS, AND WHAT IT BORROWS
-----------------------------------------
Its independence is in SELECTION and TIMING, not in arithmetic. It owns which results each
game may see (the lock), the order games are applied in, when a new season's carryover is
applied (the first game of that season), and when home-field advantage is learned (from
prior-season completed games only). Every rating operation goes through an
``EloRatingSystem`` instance's per-game formulas -- ``get_or_create_rating``,
``apply_season_carryover``, ``update_ratings``, ``learn_home_field_advantage`` and
``predict_game`` -- called by name. A local copy of the expected-score, K-factor or
uncertainty equations would agree with the builder for the wrong reason.

READ-ONLY
---------
Both stores are read with ``pandas.read_parquet``. There is no ``to_parquet``, no DuckDB
handle and no ``save_*`` anywhere in this module, and the integration test brackets a
production run with the content-digest guard.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ratings.elo import EloRatingSystem, is_divisional_game
from utils.game_lock import lock_frame

__all__ = [
    "CHAIN_START_SEASON",
    "COMPARED_COLUMNS",
    "DEFAULT_GAME_DURATION_HOURS",
    "ELO_SNAPSHOT_SOURCE",
    "GAMES_SOURCE",
    "EmptySnapshotSourceError",
    "ReplayResult",
    "VacuousRowPolicy",
    "duration_insensitivity_bound",
    "main",
    "replay",
    "tied_kickoff_groups",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# The six snapshot columns compared, DECLARED HERE as literals. They are not imported from
# tests.phase33_state.ELO_GOLD_JOIN_SUBSET -- production audit code that imports from
# `tests` inverts the dependency direction -- and not from scripts.build_elo, which is the
# forbidden surface. tests/unit/test_elo_replay_independence.py pins this tuple against
# ELO_GOLD_JOIN_SUBSET minus its `game_id` join key, so the two cannot drift.
COMPARED_COLUMNS: tuple[str, ...] = (
    "home_elo_pre",
    "away_elo_pre",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
    "elo_prob_home",
    "hfa_used",
)

# The assumed game length: a result is known at kickoff + this duration. DECLARED, NOT
# LOAD-BEARING. The prior prototype measured identical results at 0, 4 and 8 h under the
# retired Friday freeze; under the day-before lock (15.0-29.5 h lock-to-kickoff) the band is
# re-measured by tests/integration/test_elo_information_time_replay.py, which asserts 0, 4
# and 8 h identical and reports the exact upper edge from duration_insensitivity_bound().
DEFAULT_GAME_DURATION_HOURS: float = 4.0

# Compare against SILVER snapshots -- the values gold joins -- never against z-scored gold.
ELO_SNAPSHOT_SOURCE: str = "data/silver/elo_game_snapshots.parquet"

# The schedule and results the replay selects from.
GAMES_SOURCE: str = "data/silver/games.parquet"

# The first season of the rating chain. Declared here rather than imported from
# scripts.elo_generation so the replay's notion of where the chain starts is its own.
CHAIN_START_SEASON: int = 2002

_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "kickoff_et",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    # Carried through so the rating verbs apply the one neutral-site HFA rule (row 19).
    "neutral_site",
)
_SNAPSHOT_KEY_COLUMNS: tuple[str, ...] = ("game_id", "season", "week")


class EmptySnapshotSourceError(ValueError):
    """The snapshot frame holds no row for the requested seasons.

    Raised rather than reported as zero mismatches: a comparison over nothing agrees with
    everything, and "0 mismatches" over zero rows is the reading this refusal prevents.
    """


@dataclass(frozen=True)
class VacuousRowPolicy:
    """Which stored values agree with ANY replay, so they cannot count as evidence.

    In the chain's first season, week 1, every team sits at the 1500 initial rating and the
    350 initial uncertainty, and the win probability is a function of home-field advantage
    alone: those rows agree trivially. Every first-season ``hfa_used`` is the 48-point
    initial value (times the divisional factor), or 0.0 at a neutral site, because there
    is no prior season to learn from. Both are still COMPARED -- a mismatch there is still a mismatch -- but they are
    reported separately and excluded from the evidence count, so a headline "N rows
    matched" is N rows that could have disagreed.
    """

    first_season: int = CHAIN_START_SEASON
    first_week: int = 1
    first_season_constant_columns: tuple[str, ...] = ("hfa_used",)

    def vacuous_rows(self, snapshots: pd.DataFrame) -> pd.Series:
        """Boolean mask of whole rows that agree trivially."""
        return (snapshots["season"] == self.first_season) & (
            snapshots["week"] == self.first_week
        )

    def vacuous_cells(self, snapshots: pd.DataFrame) -> int:
        """First-season constant cells in rows that are NOT already whole-row vacuous."""
        in_first_season = snapshots["season"] == self.first_season
        outside_rows = in_first_season & ~self.vacuous_rows(snapshots)
        return int(outside_rows.sum()) * len(self.first_season_constant_columns)


@dataclass(frozen=True)
class ReplayResult:
    """Everything one replay measured, including what it could NOT compare.

    ``ok`` is True only when there are zero mismatches, zero missing snapshot rows, zero
    orphan snapshot rows and at least one non-vacuous comparison.
    """

    seasons: tuple[int, ...]
    duration_hours: float
    as_of: datetime
    compared: int
    vacuous: int
    compared_cells: int
    vacuous_cells: int
    mismatches: pd.DataFrame
    missing_snapshot_rows: tuple[str, ...]
    orphan_snapshot_rows: tuple[str, ...]
    replayed: pd.DataFrame
    tied_kickoff_groups: int
    tied_kickoff_games: int
    ordering_note: str = field(default="")

    @property
    def ok(self) -> bool:
        return (
            self.mismatches.empty
            and not self.missing_snapshot_rows
            and not self.orphan_snapshot_rows
            and self.compared > 0
        )


# The two orderings, named once so the CLI, the result and the SUMMARY quote one sentence.
ORDERING_NOTE: str = (
    "The replay orders games sharing a kickoff instant by (kickoff, game_id); the "
    "canonical chain sorts on kickoff_et alone (scripts/build_elo.py:411), which fixes no "
    "order among ties. Exact equality between the two is evidence only because tied games "
    "involve disjoint team pairs whose updates commute -- a claim the tied-kickoff "
    "permutation test measures rather than assumes."
)


def _read_source(relative_path: str) -> pd.DataFrame:
    """Read one silver parquet, read-only, relative to the repository root."""
    return pd.read_parquet(REPO_ROOT / relative_path)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"the {name} frame has no {missing} column(s) (columns: {sorted(frame.columns)})"
        raise KeyError(msg)


def _aware_utc(value: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        msg = (
            f"as_of must be timezone-aware, got naive {value!r}; refusing to relabel it"
        )
        raise ValueError(msg)
    return cast(pd.Timestamp, stamp.tz_convert("UTC"))


def _chain_frame(
    games_df: pd.DataFrame,
    duration_hours: float,
    tie_rank: Mapping[str, int] | None,
) -> pd.DataFrame:
    """The games the chain runs over, each with its lock, end and tie-break key."""
    _require_columns(games_df, _GAME_COLUMNS, "games")
    chain = games_df.loc[
        games_df["season"] >= CHAIN_START_SEASON, list(_GAME_COLUMNS)
    ].copy()
    chain["game_id"] = chain["game_id"].astype(str)
    locks = lock_frame(chain).dt.tz_convert("UTC")
    chain["lock"] = chain["game_id"].map(locks)
    chain["end"] = chain["kickoff_et"] + pd.Timedelta(hours=float(duration_hours))
    chain["completed"] = chain["home_score"].notna() & chain["away_score"].notna()

    # The tie-break is DECLARED, not incidental: sorting on kickoff alone leaves the order of
    # tied games to the sort algorithm. game_id is the default; tie_rank lets the permutation
    # test impose any other order and prove the replayed values do not move.
    if tie_rank is None:
        chain["tie"] = chain["game_id"]
    else:
        unranked = sorted(set(chain["game_id"]) - set(tie_rank))
        if unranked:
            msg = f"tie_rank does not rank {len(unranked)} game(s), e.g. {unranked[:5]}"
            raise ValueError(msg)
        chain["tie"] = chain["game_id"].map(tie_rank)
    return chain


def tied_kickoff_groups(games: pd.DataFrame) -> tuple[int, int]:
    """(groups, games) for kickoff instants shared by more than one game in *games*."""
    sizes = games.groupby("kickoff_et").size()
    shared = sizes[sizes > 1]
    return len(shared), int(shared.sum())


def duration_insensitivity_bound(games_df: pd.DataFrame) -> timedelta:
    """The largest assumed duration under which no replayed value can change.

    A game's own teams' ratings depend only on those teams' prior results. For each team,
    the tightest case is its previous completed game against its next game's lock: any
    duration up to ``min(lock(next) - kickoff(previous))`` still applies every such result
    (the boundary is inclusive), and one second more drops at least one. The value is
    derived from the schedule, and the integration test confirms it by replaying on both
    sides of it.
    """
    chain = _chain_frame(games_df, 0.0, None)
    sides = []
    for side in ("home_team", "away_team"):
        part = chain[["game_id", "kickoff_et", "lock", "completed", side]]
        sides.append(
            part.set_axis(
                ["game_id", "kickoff_et", "lock", "completed", "team"], axis=1
            )
        )
    by_team = pd.concat(sides, ignore_index=True).sort_values(["team", "kickoff_et"])
    previous_kickoff = by_team.groupby("team")["kickoff_et"].shift(1)
    previous_completed = by_team.groupby("team")["completed"].shift(1)
    # ``eq(True)`` reads a team's first game (no previous game, NaN) as False without the
    # object-dtype downcast ``fillna(False)`` would warn about.
    gaps = (by_team["lock"] - previous_kickoff)[previous_completed.eq(True)]
    return gaps.min().to_pytimedelta()


class _Chain:
    """The replay's own rating state, advanced strictly in the order it chooses."""

    def __init__(self, completed_games: pd.DataFrame) -> None:
        self.system = EloRatingSystem()
        self.season: int | None = None
        self._completed = completed_games
        self._seasons = sorted(int(s) for s in completed_games["season"].unique())

    def enter_season(self, season: int) -> None:
        """Carry over and learn HFA for every chain season up to *season*, once each.

        The replay chooses this point: the first game of a new season, applied or predicted,
        whichever comes first. HFA is learned from PRIOR-season completed games only; every
        one of them ended months before any lock in *season*.
        """
        if self.season is not None and season <= self.season:
            return
        start = CHAIN_START_SEASON if self.season is None else self.season + 1
        for entering in range(start, season + 1):
            if entering != season and entering not in self._seasons:
                continue
            self.system.apply_season_carryover(entering)
            prior = cast(
                pd.DataFrame, self._completed[self._completed["season"] < entering]
            )
            self.system.learn_home_field_advantage(prior, entering)
        self.season = season

    def apply(self, game: Any) -> None:
        """Apply one completed result."""
        season = int(game.season)
        self.enter_season(season)
        home, away = str(game.home_team), str(game.away_team)
        self.system.update_ratings(
            home_team=home,
            away_team=away,
            home_score=int(game.home_score),
            away_score=int(game.away_score),
            season=season,
            game_date=game.kickoff_et,
            game_id=str(game.game_id),
            is_divisional=is_divisional_game(home, away),
            neutral_site=game.neutral_site,
        )

    def pre_game(self, game: Any) -> dict[str, Any]:
        """The six pre-game values for *game* from the current state."""
        season = int(game.season)
        self.enter_season(season)
        home, away = str(game.home_team), str(game.away_team)
        home_rating = self.system.get_or_create_rating(home, season)
        away_rating = self.system.get_or_create_rating(away, season)
        # The game's raw neutral flag goes to the library's one HFA rule (zero at a
        # neutral site, row 19), exactly as the chain being replayed passes it; matching
        # it is a property of the rating model, not of information timing.
        prediction = self.system.predict_game(
            home,
            away,
            season,
            neutral_site=game.neutral_site,
            is_divisional=is_divisional_game(home, away),
        )
        return {
            "game_id": str(game.game_id),
            "home_elo_pre": home_rating.rating,
            "away_elo_pre": away_rating.rating,
            "home_elo_uncertainty": home_rating.uncertainty,
            "away_elo_uncertainty": away_rating.uncertainty,
            "elo_prob_home": prediction["home_win_prob"],
            "hfa_used": prediction["hfa_used"],
        }


def _compare(replayed: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Every cell where replayed and stored are not exactly ``==``."""
    joined = replayed.merge(
        stored[["game_id", *COMPARED_COLUMNS]],
        on="game_id",
        how="inner",
        suffixes=("_replayed", "_stored"),
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for column in COMPARED_COLUMNS:
        left = joined[f"{column}_replayed"]
        right = joined[f"{column}_stored"]
        differs = ~(left == right)
        for game_id, value, stored_value in zip(
            joined.loc[differs, "game_id"],
            left[differs],
            right[differs],
            strict=True,
        ):
            records.append(
                {
                    "game_id": game_id,
                    "column": column,
                    "replayed": value,
                    "stored": stored_value,
                }
            )
    return pd.DataFrame(
        records, columns=pd.Index(["game_id", "column", "replayed", "stored"])
    )


def replay(
    seasons: Iterable[int],
    duration_hours: float = DEFAULT_GAME_DURATION_HOURS,
    games_df: pd.DataFrame | None = None,
    snapshots_df: pd.DataFrame | None = None,
    *,
    as_of: datetime | None = None,
    tie_rank: Mapping[str, int] | None = None,
    vacuous_policy: VacuousRowPolicy | None = None,
) -> ReplayResult:
    """Recompute every stored pre-game Elo row in *seasons* and compare it with ``==``.

    The chain always starts at ``CHAIN_START_SEASON``; *seasons* only selects which stored
    rows are compared and which games are required to have one.

    Args:
        seasons: Seasons whose snapshot rows are compared.
        duration_hours: Assumed game length. A result is applied to a game when its
            kickoff plus this duration is AT OR BEFORE that game's lock.
        games_df: Silver ``games`` rows (default: read from ``GAMES_SOURCE``).
        snapshots_df: Silver ``elo_game_snapshots`` rows (default: ``ELO_SNAPSHOT_SOURCE``).
        as_of: The instant "required" is judged at: a completed game, or an unplayed game
            whose lock is at or before *as_of*, must have a snapshot row. Default: now.
        tie_rank: An explicit total order for games sharing a kickoff instant (default:
            ``game_id``). Exists so the permutation test can impose a different order.
        vacuous_policy: Which rows and cells agree trivially (default: the chain's first
            season, week 1, and first-season ``hfa_used``).

    Returns:
        A :class:`ReplayResult`.

    Raises:
        EmptySnapshotSourceError: when no snapshot row exists for *seasons*.
    """
    wanted = tuple(sorted({int(season) for season in seasons}))
    as_of_utc = _aware_utc(as_of if as_of is not None else datetime.now(UTC))
    policy = vacuous_policy if vacuous_policy is not None else VacuousRowPolicy()
    games = games_df if games_df is not None else _read_source(GAMES_SOURCE)
    snapshots = (
        snapshots_df if snapshots_df is not None else _read_source(ELO_SNAPSHOT_SOURCE)
    )
    _require_columns(
        snapshots, (*_SNAPSHOT_KEY_COLUMNS, *COMPARED_COLUMNS), "snapshots"
    )

    stored = cast(pd.DataFrame, snapshots[snapshots["season"].isin(wanted)]).copy()
    stored["game_id"] = stored["game_id"].astype(str)
    if stored.empty:
        msg = (
            f"elo_game_snapshots holds no row for seasons {list(wanted)}; there is nothing "
            "to compare, and a comparison over nothing would read as agreement."
        )
        raise EmptySnapshotSourceError(msg)

    chain = _chain_frame(games, duration_hours, tie_rank)
    stored_ids = sorted(set(stored["game_id"]))
    in_scope = chain["season"].isin(wanted)
    required = in_scope & (chain["completed"] | (chain["lock"] <= as_of_utc))
    missing = tuple(sorted(set(chain.loc[required, "game_id"]) - set(stored_ids)))
    orphans = tuple(sorted(set(stored_ids) - set(chain["game_id"])))

    targets = cast(pd.DataFrame, chain[in_scope & chain["game_id"].isin(stored_ids)])
    targets = targets.sort_values(by=["kickoff_et", "tie"])
    completed = cast(pd.DataFrame, chain[chain["completed"]])
    applicable = completed.sort_values(by=["end", "tie"])
    state = _Chain(completed)

    rows: list[dict[str, Any]] = []
    pending: Iterator[Any] = iter(applicable.itertuples(index=False))
    next_game: Any = next(pending, None)
    target: Any
    for target in targets.itertuples(index=False):
        # Apply every result that ended AT OR BEFORE this game's lock (inclusive).
        while next_game is not None and next_game.end <= target.lock:
            state.apply(next_game)
            next_game = next(pending, None)
        rows.append(state.pre_game(target))

    replayed = pd.DataFrame(rows, columns=pd.Index(["game_id", *COMPARED_COLUMNS]))
    mismatches = _compare(replayed, stored)

    replayed_ids = sorted(set(replayed["game_id"]))
    compared_rows = cast(pd.DataFrame, stored[stored["game_id"].isin(replayed_ids)])
    vacuous_mask = policy.vacuous_rows(compared_rows)
    vacuous = int(vacuous_mask.sum())
    compared = len(compared_rows) - vacuous
    vacuous_cells = policy.vacuous_cells(compared_rows)
    compared_cells = compared * len(COMPARED_COLUMNS) - vacuous_cells
    groups, tied_games = tied_kickoff_groups(targets)

    return ReplayResult(
        seasons=wanted,
        duration_hours=float(duration_hours),
        as_of=as_of_utc.to_pydatetime(),
        compared=compared,
        vacuous=vacuous,
        compared_cells=compared_cells,
        vacuous_cells=vacuous_cells,
        mismatches=mismatches,
        missing_snapshot_rows=missing,
        orphan_snapshot_rows=orphans,
        replayed=replayed,
        tied_kickoff_groups=groups,
        tied_kickoff_games=tied_games,
        ordering_note=ORDERING_NOTE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seasons(text: str) -> list[int]:
    """``2002-2026``, ``2024,2025`` or ``2025``."""
    seasons: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            if last < first:
                msg = f"season range {part!r} runs backwards"
                raise argparse.ArgumentTypeError(msg)
            seasons.extend(range(first, last + 1))
        elif part:
            seasons.append(int(part))
    if not seasons:
        msg = f"no seasons in {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return seasons


def _parse_as_of(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        msg = f"--as-of must carry a UTC offset, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _emit(line: str) -> None:
    """Write one CLI line to stdout (the CLI's output IS its interface)."""
    sys.stdout.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay against production silver and print the counts. Writes nothing."""
    parser = argparse.ArgumentParser(
        prog="python -m audit.elo_replay",
        description=(
            "Independently replay every pre-game Elo rating from only pre-lock results "
            "and compare it with exact equality against silver elo_game_snapshots. "
            "Read-only."
        ),
    )
    parser.add_argument(
        "--seasons",
        type=_parse_seasons,
        required=True,
        help="Seasons to compare: a range (2002-2026), a list (2024,2025) or one season.",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=DEFAULT_GAME_DURATION_HOURS,
        help="Assumed game length in hours (default: %(default)s).",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=None,
        help="Aware ISO instant 'required' is judged at (default: now).",
    )
    args = parser.parse_args(argv)

    result = replay(args.seasons, duration_hours=args.duration_hours, as_of=args.as_of)
    _emit(f"SEASONS= {result.seasons[0]}-{result.seasons[-1]}")
    _emit(f"DURATION_HOURS= {result.duration_hours}")
    _emit(f"AS_OF= {result.as_of.isoformat()}")
    _emit(f"COMPARED= {result.compared}")
    _emit(f"VACUOUS= {result.vacuous}")
    _emit(f"COMPARED_CELLS= {result.compared_cells}")
    _emit(f"VACUOUS_CELLS= {result.vacuous_cells}")
    _emit(f"MISMATCHES= {len(result.mismatches)}")
    _emit(f"MISSING_SNAPSHOT_ROWS= {len(result.missing_snapshot_rows)}")
    _emit(f"ORPHAN_SNAPSHOT_ROWS= {len(result.orphan_snapshot_rows)}")
    _emit(f"TIED_KICKOFF_GROUPS= {result.tied_kickoff_groups}")
    _emit(f"TIED_KICKOFF_GAMES= {result.tied_kickoff_games}")
    _emit(f"ORDERING= {result.ordering_note}")
    for game_id in result.missing_snapshot_rows:
        _emit(f"MISSING: {game_id}")
    for game_id in result.orphan_snapshot_rows:
        _emit(f"ORPHAN: {game_id}")
    if not result.mismatches.empty:
        _emit(result.mismatches.head(50).to_string(index=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
