"""A small synthetic gold + silver + Elo-snapshot store for the re-fit training tests (Phase 37).

WHY IT EXISTS
-------------
``refit.training`` trains on every finished game from 2002 through a cutoff. A test that pointed it
at the real gold would train on 6,500 rows per target and read production stores; this fixture
writes a real-shaped store of about 16 games a season, 2002-2026, under ``tmp_path`` instead, so a
whole three-target re-fit takes seconds and touches nothing outside the test's own directory.

WHAT IT CARRIES, AND WHY EACH PIECE IS THERE
--------------------------------------------
* Gold ``features_{wp,ats,ou}.parquet`` with the real id and label columns (``home_win`` /
  ``target_wp``, ``home_margin`` / ``point_differential``, ``total_points``), six numeric inputs,
  and ONE column of each excluded group (injury, situational, snap). The excluded columns are the
  most predictive inputs in the frame on purpose: if the exclusion ever stopped working they would
  be selected, so a test that finds none of them selected is not passing by accident.
* Silver ``games.parquet`` with tz-aware ``kickoff_et`` and final scores. The 2026 season holds,
  around the chosen cutoff (Tue 2026-11-03 17:00 America/New_York = 22:00 UTC):
    - a game kicking off EXACTLY at the cutoff, with a final score (must be excluded);
    - a game kicking off ONE SECOND before it, with a final score but no gold label yet (must be
      included, labelled from silver);
    - a postponed game that kicked off before the cutoff with no final score (excluded, counted
      under ``no_final_score``);
    - a finished game with no gold row (excluded, counted under ``no_gold_row``);
    - unplayed games after the cutoff, present in gold with no label, as a live slate is.
* An Elo snapshot frame naming every gold game, with no provisional row.

Synthetic and tmp-only: nothing here reads or writes ``data/``, ``artifacts/`` or ``ledger/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from conf.season_partition import CORPUS_FIRST_SEASON

__all__ = [
    "FIXTURE_CUTOFF_UTC",
    "FIXTURE_FIRST_SEASON",
    "FIXTURE_GAMES_PER_SEASON",
    "FIXTURE_LAST_SEASON",
    "RefitFixture",
    "build_refit_fixture",
]

FIXTURE_FIRST_SEASON: int = CORPUS_FIRST_SEASON
FIXTURE_LAST_SEASON: int = 2026
FIXTURE_GAMES_PER_SEASON: int = 16

# Tue 2026-11-03 17:00 America/New_York (EST after the 2026-11-01 DST change) = 22:00 UTC.
FIXTURE_CUTOFF_UTC: datetime = datetime(2026, 11, 3, 22, 0, tzinfo=UTC)

# The six model inputs, and one column of each excluded group (their registry predicates:
# injury = home_/away_ + an InjuryBuilder basename, situational = home_/away_ + a spot suffix,
# snap = home_/away_ + snap_continuity).
FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "rest_advantage",
    "venue_elevation_ft",
    "weather_severity_score",
    "season_progress",
    "temp_f",
)
EXCLUDED_GROUP_COLUMNS: tuple[str, ...] = (
    "home_qb_out_flag",
    "home_off_bye",
    "home_snap_continuity",
)

_TEAMS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
_SEED = 37


@dataclass(frozen=True)
class RefitFixture:
    """Where the fixture store lives and the named games the edge tests need.

    Attributes:
        root: The fixture root (under ``tmp_path``).
        gold_dir: The three gold matrices.
        silver_dir: Silver ``games.parquet``.
        artifacts_dir: An empty artifacts root for saved re-fits.
        snapshots: Elo snapshot rows for every gold game, none provisional.
        cutoff_utc: The scheduled cutoff the edge games sit around.
        at_cutoff_game_id: Kicks off exactly at the cutoff, with a final score.
        before_cutoff_game_id: Kicks off one second before the cutoff; final score in silver,
            no label in gold yet.
        postponed_game_id: Kicked off before the cutoff, no final score.
        no_gold_game_id: Finished before the cutoff, absent from gold.
    """

    root: Path
    gold_dir: Path
    silver_dir: Path
    artifacts_dir: Path
    snapshots: pd.DataFrame
    cutoff_utc: datetime
    at_cutoff_game_id: str
    before_cutoff_game_id: str
    postponed_game_id: str
    no_gold_game_id: str


def _game_id(season: int, week: int, home: str, away: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def _schedule() -> list[dict[str, object]]:
    """Every fixture game: id, season, week, teams, kickoff and whether it is played/in gold."""
    games: list[dict[str, object]] = []
    for season in range(FIXTURE_FIRST_SEASON, FIXTURE_LAST_SEASON):
        opener = datetime(season, 9, 8, 17, 0, tzinfo=UTC)
        for index in range(FIXTURE_GAMES_PER_SEASON):
            week = index // 2 + 1
            home = _TEAMS[(index * 2) % len(_TEAMS)]
            away = _TEAMS[(index * 2 + 1) % len(_TEAMS)]
            games.append(
                {
                    "game_id": _game_id(season, week, home, away),
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": opener
                    + timedelta(days=7 * (week - 1), hours=index % 2),
                    "played": True,
                    "in_gold": True,
                }
            )

    # 2026: ordinary finished games in weeks 1-7, then the edge games around the cutoff.
    opener = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
    for index in range(14):
        week = index // 2 + 1
        home = _TEAMS[(index * 2) % len(_TEAMS)]
        away = _TEAMS[(index * 2 + 1) % len(_TEAMS)]
        games.append(
            {
                "game_id": _game_id(2026, week, home, away),
                "season": 2026,
                "week": week,
                "home_team": home,
                "away_team": away,
                "kickoff_et": opener + timedelta(days=7 * (week - 1), hours=index % 2),
                "played": True,
                "in_gold": True,
            }
        )
    edges = (
        ("2026_08_PPP_QQQ", 8, datetime(2026, 11, 1, 18, 0, tzinfo=UTC), False, True),
        ("2026_08_RRR_SSS", 8, datetime(2026, 11, 2, 1, 15, tzinfo=UTC), True, False),
        ("2026_08_TTT_UUU", 8, FIXTURE_CUTOFF_UTC - timedelta(seconds=1), True, True),
        ("2026_09_VVV_WWW", 9, FIXTURE_CUTOFF_UTC, True, True),
        ("2026_09_XXX_YYY", 9, datetime(2026, 11, 6, 1, 15, tzinfo=UTC), False, True),
        ("2026_09_ZZZ_AAA", 9, datetime(2026, 11, 8, 18, 0, tzinfo=UTC), False, True),
    )
    for game_id, week, kickoff, played, in_gold in edges:
        away, home = game_id.split("_")[2], game_id.split("_")[3]
        games.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "home_team": home,
                "away_team": away,
                "kickoff_et": kickoff,
                "played": played,
                "in_gold": in_gold,
            }
        )
    return games


def build_refit_fixture(root: Path) -> RefitFixture:
    """Write the fixture store under *root* and return where everything is.

    Args:
        root: A fresh directory (a test's ``tmp_path``).

    Returns:
        The :class:`RefitFixture` describing the written store.
    """
    rng = np.random.default_rng(_SEED)
    schedule = pd.DataFrame(_schedule())
    n_games = len(schedule)

    strength = rng.normal(0.0, 1.0, n_games)
    pace = rng.normal(0.0, 1.0, n_games)
    margin = np.round(7.0 * strength + rng.normal(0.0, 6.0, n_games))
    total = np.round(44.0 + 6.0 * pace + rng.normal(0.0, 5.0, n_games))
    home_score = np.clip(np.round((total + margin) / 2.0), 0.0, None)
    away_score = np.clip(total - home_score, 0.0, None)

    played = schedule["played"].to_numpy(dtype=bool)
    games = schedule[
        ["game_id", "season", "week", "home_team", "away_team", "kickoff_et"]
    ].copy()
    games["kickoff_et"] = pd.to_datetime(games["kickoff_et"], utc=True)
    games["home_score"] = np.where(played, home_score, np.nan)
    games["away_score"] = np.where(played, away_score, np.nan)

    inputs = pd.DataFrame(
        {
            "elo_diff": strength + rng.normal(0.0, 0.8, n_games),
            "rest_advantage": rng.normal(0.0, 1.0, n_games),
            "venue_elevation_ft": pace + rng.normal(0.0, 0.8, n_games),
            "weather_severity_score": rng.normal(0.0, 1.0, n_games),
            "season_progress": rng.uniform(0.0, 1.0, n_games),
            "temp_f": rng.normal(55.0, 15.0, n_games),
            # The excluded groups: deliberately the strongest signals in the frame.
            "home_qb_out_flag": strength + rng.normal(0.0, 0.1, n_games),
            "home_off_bye": pace + rng.normal(0.0, 0.1, n_games),
            "home_snap_continuity": strength - pace,
        }
    )
    # A few missing readings, as real weather columns carry.
    inputs.loc[inputs.index % 23 == 0, "temp_f"] = np.nan

    base = pd.concat(
        [games[["game_id", "season", "week"]].reset_index(drop=True), inputs], axis=1
    )
    base["home_score"] = games["home_score"].to_numpy()
    base["away_score"] = games["away_score"].to_numpy()

    gold_scored = base["home_score"].notna()
    home = base["home_score"]
    away = base["away_score"]
    labels = {
        "home_win": ((home > away).astype(float)).where(gold_scored),
        "home_margin": (home - away).where(gold_scored),
        "total_points": (home + away).where(gold_scored),
    }

    # The game one second before the cutoff has its score in silver but not yet in gold: gold
    # labels lag on a Tuesday, which is why the labels come from silver.
    before_cutoff_id = "2026_08_TTT_UUU"
    lagging = base["game_id"] == before_cutoff_id
    labels = {name: values.mask(lagging) for name, values in labels.items()}
    base.loc[lagging, ["home_score", "away_score"]] = np.nan

    in_gold = schedule["in_gold"].to_numpy(dtype=bool)
    gold_base = base.loc[in_gold].reset_index(drop=True)
    gold_labels = {
        name: values.loc[in_gold].reset_index(drop=True)
        for name, values in labels.items()
    }

    gold_dir = root / "gold"
    silver_dir = root / "silver"
    artifacts_dir = root / "artifacts"
    for directory in (gold_dir, silver_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    wp = gold_base.copy()
    wp["target_wp"] = gold_labels["home_win"]
    wp["home_win"] = gold_labels["home_win"]
    ats = gold_base.copy()
    ats["point_differential"] = gold_labels["home_margin"]
    ats["home_margin"] = gold_labels["home_margin"]
    ou = gold_base.copy()
    ou["total_points"] = gold_labels["total_points"]
    for target, frame in (("wp", wp), ("ats", ats), ("ou", ou)):
        frame.to_parquet(gold_dir / f"features_{target}.parquet", index=False)

    games.to_parquet(silver_dir / "games.parquet", index=False)

    snapshots = gold_base[["game_id", "season", "week"]].copy()
    snapshots["is_provisional"] = False

    return RefitFixture(
        root=root,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        artifacts_dir=artifacts_dir,
        snapshots=snapshots,
        cutoff_utc=FIXTURE_CUTOFF_UTC,
        at_cutoff_game_id="2026_09_VVV_WWW",
        before_cutoff_game_id=before_cutoff_id,
        postponed_game_id="2026_08_PPP_QQQ",
        no_gold_game_id="2026_08_RRR_SSS",
    )
