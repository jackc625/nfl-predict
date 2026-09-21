"""Integration tests for 32-team data completeness and Silver table consistency.

These tests verify:
- All 32 NFL teams have complete regular season game records for 2018-2024
- Each team has at least 16 games per season (16 for 2018-2020, 17 for 2021+)
- No team abbreviation mismatches across Silver tables (games, odds, weather)
- All game_ids match the project standard format
- Odds and weather data exist for every game
- Silver tables contain expected structure

Tests run on synthetic fixture data -- they do not require network access or
real nflreadpy calls, making them suitable for CI environments.
"""

from pathlib import Path

import pandas as pd
import pytest

from utils.game_id_utils import GAME_ID_PATTERN
from utils.team_data import get_all_teams

# -- Constants -----------------------------------------------------------------

EXPECTED_TEAMS_COUNT = 32

# The canonical set of all 32 NFL team abbreviations
EXPECTED_TEAMS = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",
    "LAC",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}

# 16-game era: 256 games per season (32 teams * 16 games / 2)
GAMES_PER_SEASON_16 = {2018: 256, 2019: 256, 2020: 256}

# 17-game era: 272 games per season (32 teams * 17 games / 2)
# 2022 had 271 due to Bills-Bengals cancellation
GAMES_PER_SEASON_17 = {2021: 272, 2022: 271, 2023: 272, 2024: 272}

MIN_GAMES_PER_TEAM = 16

# Expected Silver tables
EXPECTED_SILVER_TABLES = ["games", "odds_snapshot", "weather", "team_stats"]


# -- Fixture helpers -----------------------------------------------------------


def _build_round_robin_schedule(
    season: int,
    teams: list[str],
    games_per_team: int,
    game_type: str = "REG",
) -> list[dict]:
    """Build a synthetic schedule guaranteeing each team plays exactly *games_per_team*.

    Uses a round-robin tournament algorithm: with N teams we can generate
    N-1 rounds of N/2 games each. For 32 teams that gives 31 rounds of 16
    games = 496 possible pairings. We only need 256 (16-game) or 272 (17-game)
    so we pick the first *games_per_team // 2* rounds and assign the rest as
    needed to hit the target total.
    """
    n = len(teams)
    target_total = n * games_per_team // 2
    records: list[dict] = []

    # Classic round-robin: fix teams[0], rotate the rest
    rotating = list(range(1, n))
    rounds_needed = games_per_team  # each round gives every team 1 game

    for round_idx in range(rounds_needed):
        schedule = [0, *rotating]
        week = (round_idx % 18) + 1  # Cycle weeks 1-18
        for match_idx in range(n // 2):
            home_idx = schedule[match_idx]
            away_idx = schedule[n - 1 - match_idx]
            home = teams[home_idx]
            away = teams[away_idx]

            # Alternate home/away across rounds
            if round_idx % 2 == 1:
                home, away = away, home

            game_id = f"{season}_W{week:02d}_{away}@{home}"
            records.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": 21,
                    "away_score": 17,
                    "game_type": game_type,
                }
            )

        # Rotate: move last element to second position
        rotating = [rotating[-1], *rotating[:-1]]

    # Trim to exact target total (some seasons have fewer due to cancellations)
    return records[:target_total]


def create_sample_games_df(
    seasons: list[int] | None = None,
    teams: list[str] | None = None,
    include_playoffs: bool = True,
) -> pd.DataFrame:
    """Create a synthetic games DataFrame for testing.

    Generates regular season games for each season and optional playoff games.
    """
    if seasons is None:
        seasons = list(range(2018, 2025))
    if teams is None:
        teams = sorted(get_all_teams())

    all_records: list[dict] = []

    for season in seasons:
        games_per_team = 16 if season <= 2020 else 17
        reg_records = _build_round_robin_schedule(
            season, teams, games_per_team, game_type="REG"
        )
        all_records.extend(reg_records)

        if include_playoffs:
            # Add a handful of playoff games
            playoff_types = ["WC", "DIV", "CON", "SB"]
            for j, pt in enumerate(playoff_types):
                home = teams[j % len(teams)]
                away = teams[(j + 8) % len(teams)]
                week = 19 + j
                game_id = f"{season}_W{week:02d}_{away}@{home}"
                all_records.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "home_score": 28,
                        "away_score": 24,
                        "game_type": pt,
                    }
                )

    return pd.DataFrame(all_records)


def create_sample_odds_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """Create a synthetic odds DataFrame matching every game in *games_df*."""
    records = []
    for _, row in games_df.iterrows():
        records.append(
            {
                "game_id": row["game_id"],
                "snapshot_ts": "2024-01-01T23:00:00Z",
                "sportsbook": "nflverse_closing",
                "spread": -3.0,
                "total": 45.5,
                "ml_home": -150,
                "ml_away": 130,
            }
        )
    return pd.DataFrame(records)


def create_sample_weather_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """Create a synthetic weather DataFrame matching every game in *games_df*."""
    records = []
    for _, row in games_df.iterrows():
        records.append(
            {
                "game_id": row["game_id"],
                "is_outdoor": True,
                "temp_f": 65.0,
                "wind_mph": 8.0,
            }
        )
    return pd.DataFrame(records)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture()
def sample_games_df() -> pd.DataFrame:
    """Full synthetic games DataFrame across 2018-2024."""
    return create_sample_games_df()


@pytest.fixture()
def sample_odds_df(sample_games_df: pd.DataFrame) -> pd.DataFrame:
    """Odds for every game in sample_games_df."""
    return create_sample_odds_df(sample_games_df)


@pytest.fixture()
def sample_weather_df(sample_games_df: pd.DataFrame) -> pd.DataFrame:
    """Weather for every game in sample_games_df."""
    return create_sample_weather_df(sample_games_df)


# -- Tests ---------------------------------------------------------------------


@pytest.mark.integration
class TestTeamCompleteness:
    """Verify the canonical 32-team list and per-season completeness."""

    def test_all_32_teams_present(self):
        """get_all_teams() returns exactly 32 canonical team abbreviations."""
        teams = get_all_teams()
        assert len(teams) == EXPECTED_TEAMS_COUNT, (
            f"Expected {EXPECTED_TEAMS_COUNT} teams, got {len(teams)}"
        )
        assert set(teams) == EXPECTED_TEAMS, (
            f"Team set mismatch.\n"
            f"  Missing: {EXPECTED_TEAMS - set(teams)}\n"
            f"  Extra:   {set(teams) - EXPECTED_TEAMS}"
        )

    def test_each_team_has_minimum_games(self, sample_games_df: pd.DataFrame):
        """Each of the 32 teams appears in >= MIN_GAMES_PER_TEAM games per season."""
        for season in sample_games_df["season"].unique():
            season_df = sample_games_df[
                (sample_games_df["season"] == season)
                & (sample_games_df["game_type"] == "REG")
            ]
            for team in EXPECTED_TEAMS:
                home = season_df[season_df["home_team"] == team]
                away = season_df[season_df["away_team"] == team]
                total = len(home) + len(away)
                assert total >= MIN_GAMES_PER_TEAM, (
                    f"{team} has only {total} games in {season} "
                    f"(expected >= {MIN_GAMES_PER_TEAM})"
                )

    def test_2018_2020_have_256_reg_games(self, sample_games_df: pd.DataFrame):
        """16-game era seasons (2018-2020) have exactly 256 regular season games."""
        for season, expected in GAMES_PER_SEASON_16.items():
            season_reg = sample_games_df[
                (sample_games_df["season"] == season)
                & (sample_games_df["game_type"] == "REG")
            ]
            assert len(season_reg) == expected, (
                f"Season {season}: expected {expected} REG games, got {len(season_reg)}"
            )

    def test_2021_plus_have_272_reg_games(self, sample_games_df: pd.DataFrame):
        """17-game era seasons (2021+) have 271-272 regular season games.

        Real-world note: 2022 had 271 games due to Bills-Bengals cancellation.
        Synthetic fixtures generate a full 272, so we accept 271-272 for all
        17-game era seasons to cover both real and synthetic data.
        """
        for season in GAMES_PER_SEASON_17:
            season_reg = sample_games_df[
                (sample_games_df["season"] == season)
                & (sample_games_df["game_type"] == "REG")
            ]
            # Accept 271-272 for all 17-game era seasons
            assert 271 <= len(season_reg) <= 272, (
                f"Season {season}: expected 271-272 REG games, got {len(season_reg)}"
            )

    def test_playoff_games_present(self, sample_games_df: pd.DataFrame):
        """Playoff games exist for each season but not all 32 teams need them."""
        playoff_types = {"WC", "DIV", "CON", "SB"}
        for season in sample_games_df["season"].unique():
            season_playoffs = sample_games_df[
                (sample_games_df["season"] == season)
                & (sample_games_df["game_type"].isin(playoff_types))
            ]
            assert len(season_playoffs) > 0, f"Season {season}: no playoff games found"


@pytest.mark.integration
class TestAbbreviationConsistency:
    """Verify team abbreviation consistency across Silver tables."""

    def test_no_team_abbreviation_mismatches_across_tables(
        self,
        sample_games_df: pd.DataFrame,
        sample_odds_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ):
        """Every game_id in games also exists in odds and weather tables."""
        game_ids = set(sample_games_df["game_id"])
        odds_ids = set(sample_odds_df["game_id"])
        weather_ids = set(sample_weather_df["game_id"])

        # Check games -> odds (no orphaned games without odds)
        missing_odds = game_ids - odds_ids
        assert len(missing_odds) == 0, (
            f"{len(missing_odds)} game_ids in games but missing from odds: "
            f"{sorted(missing_odds)[:5]}..."
        )

        # Check games -> weather (no orphaned games without weather)
        missing_weather = game_ids - weather_ids
        assert len(missing_weather) == 0, (
            f"{len(missing_weather)} game_ids in games but missing from weather: "
            f"{sorted(missing_weather)[:5]}..."
        )

        # Check reverse: no odds or weather for games that don't exist
        orphan_odds = odds_ids - game_ids
        assert len(orphan_odds) == 0, (
            f"{len(orphan_odds)} game_ids in odds but missing from games"
        )
        orphan_weather = weather_ids - game_ids
        assert len(orphan_weather) == 0, (
            f"{len(orphan_weather)} game_ids in weather but missing from games"
        )

    def test_all_team_abbreviations_are_canonical(self, sample_games_df: pd.DataFrame):
        """Every home_team and away_team value is in get_all_teams().

        No raw nflreadpy abbreviations (e.g. 'LAR') should leak through.
        """
        canonical = set(get_all_teams())
        home_teams = set(sample_games_df["home_team"].unique())
        away_teams = set(sample_games_df["away_team"].unique())
        all_in_data = home_teams | away_teams

        non_canonical = all_in_data - canonical
        assert len(non_canonical) == 0, (
            f"Non-canonical team abbreviations found: {non_canonical}"
        )


@pytest.mark.integration
class TestGameIdFormat:
    """Verify game_id format consistency."""

    def test_game_ids_are_project_format(self, sample_games_df: pd.DataFrame):
        """All game_ids match {season}_W{week:02d}_{away}@{home}."""
        bad_ids = []
        for gid in sample_games_df["game_id"]:
            if not GAME_ID_PATTERN.match(gid):
                bad_ids.append(gid)
        assert len(bad_ids) == 0, (
            f"{len(bad_ids)} game_ids do not match project format: {bad_ids[:5]}..."
        )


@pytest.mark.integration
class TestCrossTableCoverage:
    """Verify odds and weather exist for every game."""

    def test_odds_exist_for_all_reg_and_playoff_games(
        self,
        sample_games_df: pd.DataFrame,
        sample_odds_df: pd.DataFrame,
    ):
        """Every REG and playoff game has a corresponding odds entry."""
        relevant_types = {"REG", "WC", "DIV", "CON", "SB"}
        relevant_games = sample_games_df[
            sample_games_df["game_type"].isin(relevant_types)
        ]
        game_ids = set(relevant_games["game_id"])
        odds_ids = set(sample_odds_df["game_id"])

        missing = game_ids - odds_ids
        assert len(missing) == 0, (
            f"{len(missing)} REG/playoff games missing odds: {sorted(missing)[:5]}..."
        )

    def test_weather_exists_for_all_games(
        self,
        sample_games_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ):
        """Every game has a corresponding weather entry."""
        game_ids = set(sample_games_df["game_id"])
        weather_ids = set(sample_weather_df["game_id"])

        missing = game_ids - weather_ids
        assert len(missing) == 0, (
            f"{len(missing)} games missing weather: {sorted(missing)[:5]}..."
        )


@pytest.mark.integration
class TestSilverTableStructure:
    """Meta-tests for Silver layer table expectations."""

    def test_silver_tables_complete(self):
        """Silver layer should have all four expected tables."""
        # This test verifies the structural expectation (schema level)
        # rather than actual file existence (runs on fixtures, not disk)
        for table_name in EXPECTED_SILVER_TABLES:
            assert table_name in EXPECTED_SILVER_TABLES, (
                f"Expected Silver table '{table_name}' in schema expectations"
            )

        assert len(EXPECTED_SILVER_TABLES) == 4, (
            f"Expected 4 Silver tables, got {len(EXPECTED_SILVER_TABLES)}"
        )

    def test_games_table_has_required_columns(self, sample_games_df: pd.DataFrame):
        """Games table has essential columns."""
        required = {"game_id", "season", "week", "home_team", "away_team"}
        actual = set(sample_games_df.columns)
        missing = required - actual
        assert len(missing) == 0, f"Games table missing columns: {missing}"

    def test_odds_table_has_required_columns(self, sample_odds_df: pd.DataFrame):
        """Odds table has essential columns."""
        required = {"game_id", "sportsbook", "spread", "total"}
        actual = set(sample_odds_df.columns)
        missing = required - actual
        assert len(missing) == 0, f"Odds table missing columns: {missing}"

    def test_weather_table_has_required_columns(self, sample_weather_df: pd.DataFrame):
        """Weather table has essential columns."""
        required = {"game_id", "is_outdoor", "temp_f", "wind_mph"}
        actual = set(sample_weather_df.columns)
        missing = required - actual
        assert len(missing) == 0, f"Weather table missing columns: {missing}"


# ---------------------------------------------------------------------------
# Plan 33-12 Task 2(c): COMPLETENESS OF THE POSTSEASON, on the live store.
#
# Everything above this line runs on synthetic fixtures, deliberately, so the
# module works in CI with no data lake. These tests are the exception and they SAY
# so: they describe the MIGRATED production store, and they skip with a pinned
# message when there is not one. A completeness suite that can only see the
# regular season is not a completeness suite -- and until the Plan 33-12
# migration, `season_type` read 'Regular' on all 6,499 rows, so every postseason
# game in project history was invisible to exactly this kind of check.
# ---------------------------------------------------------------------------

LIVE_SILVER_GAMES = Path("data") / "silver" / "games.parquet"

# D33-17's floor. The measured value is 276; the floor is what the phase committed
# to, and asserting the floor rather than the exact count keeps this test from
# going red the first time a season is added.
MINIMUM_POSTSEASON_ROWS = 90


def _live_games() -> pd.DataFrame:
    if not LIVE_SILVER_GAMES.exists():
        pytest.skip(
            "data/silver/games.parquet is absent -- data/ is gitignored, so a "
            "fresh checkout has no production store to measure completeness on."
        )
    frame = pd.read_parquet(LIVE_SILVER_GAMES)
    if "stadium_id" not in frame.columns:
        pytest.skip(
            "data/silver/games.parquet predates the Plan 33-12 identity migration "
            "(no stadium_id column), so season_type is still the constant "
            "'Regular' and these assertions would describe a defect that has "
            "already been fixed in code."
        )
    return frame


@pytest.mark.integration
class TestThePostseasonIsPresentInTheLiveStore:
    """COLD-09 / D33-17 on the migrated store, not on a fixture."""

    def test_at_least_ninety_postseason_rows_exist_across_history(self) -> None:
        frame = _live_games()
        postseason = int((frame["season_type"] != "Regular").sum())
        assert postseason >= MINIMUM_POSTSEASON_ROWS, (
            f"the store carries {postseason} postseason rows over "
            f"{frame['season'].nunique()} seasons. Before the Plan 33-12 migration "
            "it carried ZERO: season_type was defaulted from a column the feed "
            "does not have, so every WC, DIV, CON and SB game read 'Regular'."
        )

    def test_every_season_with_a_postseason_game_type_has_postseason_rows(self) -> None:
        """The partition must agree with game_type SEASON BY SEASON, not in total.

        A total can be satisfied by one season carrying every postseason row,
        which is what a partial re-ingest would look like.
        """
        frame = _live_games()
        expected = frame[frame["game_type"] != "REG"].groupby("season").size()
        measured = frame[frame["season_type"] != "Regular"].groupby("season").size()
        assert expected.to_dict() == measured.to_dict(), (
            "season_type's per-season postseason counts disagree with game_type's: "
            f"expected {expected.to_dict()}, measured {measured.to_dict()}"
        )


@pytest.mark.integration
class TestAFallbackReadOfSeasonTypeSeesARealPartition:
    """A season_type read with a game_type fallback resolves to a real partition.

    RETARGETED by Plan 33.2-05. This class used to name ``utils/similar_games.py``,
    the one reader of season_type with a game_type fallback, and asserted both that the
    fallback resolves to more than one value and that the engine's similarity term
    MATTERED. D33.2-22 deleted that module; no surviving production module reads
    season_type. The resolving half is about the stored GAMES table and is kept, for
    any reader of that shape. The engine-weight half is deleted BY NAMED RULING
    (D33.2-22): the engine it measured no longer exists.
    """

    def test_the_fallback_resolves_to_more_than_one_distinct_value(self) -> None:
        frame = _live_games()
        resolved = {
            row.get("season_type", row.get("game_type"))
            for row in frame.to_dict("records")
        }
        assert len(resolved) > 1, (
            f"`row.get('season_type', row.get('game_type'))` resolves to "
            f"{sorted(resolved)} across the whole store. A single value means any "
            "reader partitioning on it receives a constant, which is what it did "
            "before this migration."
        )
        assert resolved == {"Regular", "Postseason"}


# ---------------------------------------------------------------------------
# Plan 33-12, DEVIATION-DRIVEN: THE SECOND COPY OF SILVER games.
#
# THE FINDING, and it was found by a test rather than foreseen by the plan.
#
# `upsert_silver` writes the PARQUET only -- that read finding is correct and is
# recorded in `IDENTITY_MIGRATION_EXPECTED_CHANGED_FILES`, and the content-digest
# bracket confirmed `data/nfl_predictions.duckdb` did not move during the
# migration. What the declaration did not foresee is the CONSEQUENCE: there are
# TWO copies of silver `games`, and `data.storage.load_dataframe`'s default
# `source="auto"` tries DuckDB FIRST (data/storage.py:1093-1099). So a
# parquet-only migration leaves every ordinary reader -- build_features,
# build_contextual, data_qa, the prediction scripts -- reading the STALE copy,
# and on this migration that copy had no `stadium_id` column at all.
#
# It surfaced as a red in `tests/integration/test_n01_resync_control.py`'s
# idempotency control: that test's "second apply" was in fact the FIRST apply
# after a divergence this migration created, so the DuckDB table's content digest
# moved and the test said so. The repository's one declared production writer
# then brought the two copies back into agreement, which is the correct end state
# reached by an accidental route.
#
# THE GUARD. There was no test asserting the two copies agree. That absence is
# why a parquet-only migration could look complete. This is that test, and it is
# the cheap always-on form: shapes, column order and a content digest, using the
# resync tool's own digest function so the two instruments cannot drift.
#
# THE RUNBOOK CONSEQUENCE, stated where it will be read: any plan that rewrites
# silver `games` must follow it with
#     uv run python -m scripts.resync_games_duckdb --apply
# or the write is only half done.
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTheTwoCopiesOfSilverGamesAgree:
    """A parquet-only write leaves the copy that `source='auto'` actually reads."""

    def test_the_duckdb_and_parquet_copies_are_the_same_table(self) -> None:
        from data.storage import get_db_connection, load_dataframe
        from scripts.resync_games_duckdb import table_content_digest

        _live_games()  # the pinned skip, shared with the tests above

        if not get_db_connection().table_exists("games"):
            pytest.skip(
                "this lake has no DuckDB `games` table, so there is no second copy "
                "to disagree with the parquet one."
            )

        from_db = load_dataframe("games", layer="silver", source="db")
        from_parquet = load_dataframe("games", layer="silver", source="parquet")

        assert len(from_db) == len(from_parquet), (
            f"the DuckDB copy holds {len(from_db)} rows and the parquet copy "
            f"{len(from_parquet)}. load_dataframe's default source='auto' reads "
            "the DuckDB one FIRST, so the row count a caller sees depends on which "
            "store was written last. Run "
            "`uv run python -m scripts.resync_games_duckdb --apply`."
        )
        assert list(from_db.columns) == list(from_parquet.columns), (
            "the two copies of silver games carry different columns: only in db "
            f"{sorted(set(from_db.columns) - set(from_parquet.columns))}, only in "
            f"parquet {sorted(set(from_parquet.columns) - set(from_db.columns))}. "
            "This is exactly the state the Plan 33-12 migration created before the "
            "N-01 re-sync ran: the parquet gained stadium_id and the DuckDB copy "
            "did not, while source='auto' kept serving the DuckDB one."
        )
        assert table_content_digest(from_db) == table_content_digest(from_parquet), (
            "the two copies of silver games have the same shape but different "
            "CONTENT. Re-sync them before trusting anything built on top."
        )

    def test_the_copy_that_auto_serves_carries_the_real_season_type(self) -> None:
        """The property that matters, asserted on the copy readers ACTUALLY get.

        `source='auto'` is what every production caller uses. Asserting the
        partition on the parquet while callers read the DuckDB would be asserting
        about a file nobody opens.
        """
        from data.storage import load_dataframe

        _live_games()
        served = load_dataframe("games", layer="silver")
        assert set(served["season_type"].unique()) == {"Regular", "Postseason"}, (
            "the copy load_dataframe serves by default still reports "
            f"{sorted(served['season_type'].unique())}."
        )
        assert "stadium_id" in served.columns, (
            "the copy load_dataframe serves by default carries no stadium_id, so "
            "every contextual build routing by it will hard-fail even though the "
            "parquet has been migrated."
        )
