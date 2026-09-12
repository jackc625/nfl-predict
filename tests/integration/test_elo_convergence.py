"""Integration tests for Elo rating convergence across full historical data.

These tests process the full 2002-2024 dataset and verify that:
- Ratings converge to meaningful differentiation by 2018 (after 16 years burn-in)
- Known good/bad teams are ranked correctly (2007 Patriots top 3, 2017 Browns bottom 3)

All tests are marked @pytest.mark.slow as they process 6000+ games.
"""

import pytest

from ratings.elo import EloRatingSystem


def _load_all_games():
    """Load all games from silver layer.

    Returns:
        DataFrame with all available games, sorted by season.

    Raises:
        pytest.skip: If silver layer is not available.
    """
    try:
        from data.storage import load_dataframe

        games_df = load_dataframe("games", layer="silver")
        if len(games_df) == 0:
            pytest.skip("No games data in silver layer")
        return games_df
    except (FileNotFoundError, OSError, ValueError) as e:
        pytest.skip(f"Silver layer not available: {e}")


def _process_all_seasons(games_df):
    """Process all seasons chronologically through EloRatingSystem.

    Args:
        games_df: DataFrame with all games.

    Returns:
        Tuple of (EloRatingSystem, list of per-season rating DataFrames).
    """
    elo = EloRatingSystem()
    seasons = sorted(games_df["season"].unique())

    season_ratings = {}
    for season in seasons:
        season_games = games_df[games_df["season"] == season]
        elo.process_season_chronologically(season_games, season)
        # Snapshot end-of-season ratings
        season_ratings[season] = elo.get_current_ratings()

    return elo, season_ratings


@pytest.mark.slow
class TestEloConvergence:
    """Test that Elo ratings converge by 2018 after burn-in from 2002."""

    def test_elo_convergence_by_2018(self):
        """Process 2002-2024 games. By 2018, ratings should be meaningfully
        differentiated (std > 50) but not divergent (std < 200), with mean
        near 1500 +/- 50.
        """
        games_df = _load_all_games()

        # Verify we have data from 2002
        min_season = games_df["season"].min()
        if min_season > 2002:
            pytest.skip(
                f"Need data from 2002 for burn-in, earliest available: {min_season}"
            )

        elo, season_ratings = _process_all_seasons(games_df)

        # Check 2018 ratings
        assert 2018 in season_ratings, "No 2018 season ratings found"
        ratings_2018 = season_ratings[2018]

        # Rating standard deviation should show meaningful differentiation
        rating_std = ratings_2018["rating"].std()
        assert rating_std > 50, (
            f"Rating std should be > 50 by 2018 (differentiated), got {rating_std:.1f}"
        )
        assert rating_std < 200, (
            f"Rating std should be < 200 by 2018 (not divergent), got {rating_std:.1f}"
        )

        # Mean should be near 1500
        rating_mean = ratings_2018["rating"].mean()
        assert abs(rating_mean - 1500) < 50, (
            f"Mean rating should be near 1500, got {rating_mean:.1f}"
        )

    def test_known_team_fixtures(self):
        """After full 2002-2024 processing:
        - 2007 Patriots (16-0 regular season) rank top 3 by end-of-season Elo
        - 2017 Browns (0-16) rank bottom 3
        """
        games_df = _load_all_games()

        min_season = games_df["season"].min()
        if min_season > 2002:
            pytest.skip(
                f"Need data from 2002 for burn-in, earliest available: {min_season}"
            )

        elo, season_ratings = _process_all_seasons(games_df)

        # Check 2007 Patriots -- should be top 3
        assert 2007 in season_ratings, "No 2007 season ratings found"
        ratings_2007 = season_ratings[2007].reset_index(drop=True)
        # Ratings are sorted descending by default from get_current_ratings
        top_3_teams_2007 = ratings_2007.head(3)["team"].tolist()
        assert "NE" in top_3_teams_2007, (
            f"2007 Patriots (NE) should be top 3, but top 3 are: {top_3_teams_2007}. "
            f"NE rating: {ratings_2007[ratings_2007['team'] == 'NE']['rating'].values}"
        )

        # Check 2017 Browns -- should be bottom 3
        assert 2017 in season_ratings, "No 2017 season ratings found"
        ratings_2017 = season_ratings[2017].reset_index(drop=True)
        bottom_3_teams_2017 = ratings_2017.tail(3)["team"].tolist()
        assert "CLE" in bottom_3_teams_2017, (
            f"2017 Browns (CLE) should be bottom 3, but bottom 3 are: {bottom_3_teams_2017}. "
            f"CLE rating: {ratings_2017[ratings_2017['team'] == 'CLE']['rating'].values}"
        )


# ---------------------------------------------------------------------------
# THE RERUN-IDENTITY SUITE (Plan 33-03 Task 3, COLD-04, T-33-16b)
#
# WHAT WAS WRONG. `EloBuilder.update_current_season` called
# `self.elo_system.load_ratings()` -- FINAL ratings that already contain this season's
# completed games from any prior run -- and then reprocessed EVERY completed game in
# that season. There is no processed-game watermark anywhere, so a weekly rerun applied
# every completed game a SECOND time. The weekly path is precisely where that fires,
# and it would silently inflate the ratings the deployed WP, ATS and O/U models read.
#
# WHAT REPLACED IT. The current season is REBUILT deterministically from the prior
# season's terminal state, which is itself re-derived from the canonical chain. The
# result is a pure function of (prior terminal state, this season's completed games),
# so a rerun is idempotent BY CONSTRUCTION rather than by bookkeeping. A persisted
# watermark was considered and rejected: it is a second piece of state that can drift
# from the ratings it describes, and `load_ratings` already repopulates `hfa_by_season`
# from JSON (ratings/elo.py:638-639) -- adding a second JSON-backed guard beside a
# fragile one is not an improvement.
#
# CARRYOVER-ONCE HAS A STATIC HALF AND A DYNAMIC HALF. R2's 4-decimal carryover
# criterion is the STATIC half: it checks the 75/25 formula on one application. This
# suite is the DYNAMIC half -- that the application happens exactly ONCE across repeat
# runs. Before this plan, carryover-once was enforced only incidentally, by
# `process_season_chronologically`'s `if season not in self.hfa_by_season` guard
# (ratings/elo.py:450-452), a guard coupled to an HFA dict that `load_ratings`
# repopulates from JSON -- so its answer depended on whether a file happened to exist.
#
# "UNCHANGED" IS NOT ENOUGH ON ITS OWN. A writer that did nothing at all satisfies it.
# So the suite also asserts the row digests, that a THIRD run equals the second, that a
# rerun equals a SINGLE clean run from the same prior state, and that adding one game
# advances exactly that game's effect once rather than twice.
# ---------------------------------------------------------------------------

import copy

import pandas as pd

from tests.fixtures.elo_sandbox import (
    make_season_games,
    per_season_row_digests,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
    seed_sandbox_games,
)

PRIOR_SEASON = 2025
LIVE_SEASON = 2026
ROW_TABLES = ("elo_game_snapshots", "games_with_elo", "elo_rating_history")


def _two_season_frame(live_weeks: int = 2) -> pd.DataFrame:
    return pd.concat(
        [
            make_season_games(PRIOR_SEASON, weeks=4),
            make_season_games(LIVE_SEASON, weeks=live_weeks),
        ],
        ignore_index=True,
    )


def _weekly_run(builder):
    """One whole weekly Elo run: re-derive the live season and persist it."""
    update = builder.update_current_season(season=LIVE_SEASON)
    builder.save_live_append(
        update.season,
        snapshots=update.snapshots,
        games_with_elo=update.games_with_elo,
        rating_history=update.rating_history,
    )
    return update


def _ratings_map(builder) -> dict[str, float]:
    """Every team's rating to FOUR DECIMAL PLACES, which is R2's stated resolution."""
    return {
        team: round(rating.rating, 4)
        for team, rating in builder.elo_system.ratings.items()
    }


def _digest_map(sandbox) -> dict[str, dict[str, str]]:
    return {
        table: per_season_row_digests(read_sandbox_table(sandbox, table))
        for table in ROW_TABLES
    }


class TestAWeeklyRerunIsIdenticalToASingleCleanRun:
    """Five arms. Together they say "idempotent", which "unchanged" alone does not."""

    def test_a_second_run_leaves_every_rating_and_every_digest_identical(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, _two_season_frame())

        _weekly_run(builder)
        first_ratings = _ratings_map(builder)
        first_digests = _digest_map(sandbox)
        assert first_ratings, "the first run rated nobody, so nothing below means much"
        assert first_digests["elo_game_snapshots"], "the first run wrote no snapshots"

        _weekly_run(builder)
        second_ratings = _ratings_map(builder)
        second_digests = _digest_map(sandbox)

        moved = {
            team: (first_ratings[team], second_ratings.get(team))
            for team in first_ratings
            if first_ratings[team] != second_ratings.get(team)
        }
        assert not moved, (
            "a second weekly run MOVED team ratings, which is the signature of every "
            f"completed game being applied twice: {moved}"
        )
        assert second_digests == first_digests, (
            "the row digests moved on a rerun. Rating equality alone would pass if the "
            "second run had written nothing at all, which is why both are asserted."
        )

    def test_a_third_run_equals_the_second(self, tmp_path, monkeypatch) -> None:
        """Idempotence, not a one-off coincidence of run two."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, _two_season_frame())

        _weekly_run(builder)
        _weekly_run(builder)
        second = _ratings_map(builder)
        _weekly_run(builder)
        third = _ratings_map(builder)

        disagreements = {
            team: (second[team], third.get(team))
            for team in second
            if second[team] != third.get(team)
        }
        assert not disagreements, (
            f"the third run disagreed with the second: {disagreements}"
        )

    def test_a_rerun_equals_a_single_clean_run_from_the_same_prior_state(
        self, tmp_path, monkeypatch
    ) -> None:
        """The POSITIVE statement of "no double application".

        "Unchanged" says the second run did not move anything. It does not say the
        FIRST run was right. This compares the reran state against a fresh single run
        over the same games in a store that has never been written, which is the only
        comparison that can catch a first run that was already double-applied.
        """
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = _two_season_frame()
        rerun_builder = sandbox_builder(sandbox, games)

        _weekly_run(rerun_builder)
        _weekly_run(rerun_builder)
        reran = _ratings_map(rerun_builder)

        clean_sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path / "clean")
        clean_builder = sandbox_builder(clean_sandbox, games)
        _weekly_run(clean_builder)
        clean = _ratings_map(clean_builder)

        divergences = {
            team: (reran.get(team), clean.get(team))
            for team in set(reran) | set(clean)
            if reran.get(team) != clean.get(team)
        }
        assert not divergences, (
            "a rerun did not reproduce a single clean run from the same prior-season "
            f"terminal state: {divergences}"
        )

    def test_one_more_completed_game_advances_exactly_that_game_once(
        self, tmp_path, monkeypatch
    ) -> None:
        """Exactly two teams move, and each by ONE application's amount."""
        from ratings.elo import is_divisional_game

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, _two_season_frame(live_weeks=2))

        _weekly_run(builder)
        before = _ratings_map(builder)
        state_before = copy.deepcopy(builder.elo_system)

        # Week 3 arrives with exactly ONE graded game; the rest stay ungraded, so it is
        # the last game chronologically and nothing downstream of it can also move.
        three_weeks = _two_season_frame(live_weeks=3)
        week_three = (three_weeks["season"] == LIVE_SEASON) & (three_weeks["week"] == 3)
        three_weeks.loc[week_three, ["home_score", "away_score"]] = None
        new_index = three_weeks.index[week_three][0]
        three_weeks.loc[new_index, "home_score"] = 31.0
        three_weeks.loc[new_index, "away_score"] = 13.0
        new_game = three_weeks.loc[new_index]
        seed_sandbox_games(three_weeks)

        # What ONE application of that game produces, from the state run one ended in.
        expected_system = copy.deepcopy(state_before)
        expected_system.update_ratings(
            home_team=new_game["home_team"],
            away_team=new_game["away_team"],
            home_score=int(new_game["home_score"]),
            away_score=int(new_game["away_score"]),
            season=LIVE_SEASON,
            game_date=new_game["kickoff_et"],
            game_id=new_game["game_id"],
            is_divisional=is_divisional_game(
                new_game["home_team"], new_game["away_team"]
            ),
        )
        expected = {
            team: round(rating.rating, 4)
            for team, rating in expected_system.ratings.items()
        }

        _weekly_run(builder)
        after = _ratings_map(builder)

        moved = sorted(team for team in before if before[team] != after.get(team))
        assert moved == sorted([new_game["home_team"], new_game["away_team"]]), (
            f"exactly the two teams in the new game must move, but {moved} did"
        )
        for team in moved:
            single = expected[team] - before[team]
            actual = after[team] - before[team]
            assert abs(actual - single) < 1e-4, (
                f"{team} moved by {actual:.4f}; ONE application of "
                f"{new_game['game_id']} produces {single:.4f}, and a double "
                f"application would produce about {2 * single:.4f}."
            )

    def test_a_starting_state_that_already_contains_the_season_is_refused(
        self, tmp_path, monkeypatch
    ) -> None:
        """Defence in depth, raised BY NAME on the real path.

        The re-derivation is idempotent by construction, so this guard should never
        fire in production. That is exactly why it is tested: a property that is merely
        intended is not enforced, and the next author to change how the prior state is
        obtained needs something that objects.
        """
        from scripts.build_elo import EloBuilder, EloSeasonReapplicationError

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = _two_season_frame()
        poisoner = sandbox_builder(sandbox, games)

        # A state that has ALREADY processed the live season -- which is what
        # load_ratings() used to hand the re-derivation every single week.
        poisoner.build_season_frames(LIVE_SEASON, games=games, learn_from=games)
        contaminated = copy.deepcopy(poisoner.elo_system)

        builder = EloBuilder(data_root=sandbox)
        monkeypatch.setattr(
            type(builder),
            "build_prior_terminal_state",
            lambda self, *a, **k: contaminated,
            raising=False,
        )

        with pytest.raises(EloSeasonReapplicationError) as excinfo:
            builder.update_current_season(season=LIVE_SEASON)

        assert str(LIVE_SEASON) in str(excinfo.value), (
            f"the refusal must name the season it refused to re-apply: {excinfo.value}"
        )
