"""
Build team form metrics from play-by-play data.

This script processes NFL play-by-play data to calculate rolling team performance metrics
including EPA/play, success rates, and situational statistics.

Usage:
    python scripts/build_team_form.py --season 2024           # Process single season
    python scripts/build_team_form.py --seasons 2022 2023    # Process multiple seasons
    python scripts/build_team_form.py --all-seasons          # Process all available
    python scripts/build_team_form.py --current              # Process current season
"""

import argparse
import sys
from datetime import datetime
from typing import Any

import pandas as pd

# Add project root to path
sys.path.append(".")

from conf.settings import get_settings
from data.storage import load_dataframe, save_dataframe
from features.team_form import TeamFormCalculator
from utils import get_current_nfl_week, get_logger, setup_logging
from utils.date_utils import ET

logger = get_logger(__name__)


class TeamFormBuilder:
    """Build and manage team form metrics."""

    def __init__(self, rolling_weeks: int = 4):
        """Initialize team form builder."""
        self.settings = get_settings()
        self.calculator = TeamFormCalculator(max_prior_games=rolling_weeks)

    def build_for_seasons(self, seasons: list[int]) -> pd.DataFrame:
        """
        Build team form features for specified seasons.

        Args:
            seasons: List of seasons to process

        Returns:
            DataFrame with team form features
        """
        logger.info("Building team form features for seasons", seasons=seasons)

        try:
            # Build features for all seasons
            form_df = self.calculator.build_team_form_features(seasons)

            # Validate results
            if len(form_df) > 0:
                self.calculator.validate_form_features(form_df)

            return form_df

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error(
                "Failed to build team form features", seasons=seasons, error=str(e)
            )
            raise

    def build_for_current_week(self) -> pd.DataFrame:
        """
        Build team form features for the current NFL week.

        Routes through the time-fenced ``TeamFormCalculator.build_features``
        Protocol method (NOT the deprecated ``build_team_form_features``) so the
        live current-week build enforces the same leakage guard as the canonical
        gold path (F-01, AUDIT-REPORT.md).

        OPERATIVE LEAKAGE GUARD: the ``week < target_week`` filter inside
        ``TeamFormCalculator.calculate_rolling_averages`` (features/team_form.py)
        is what actually excludes the target week's own games from the rolling
        window. The ``kickoff_et < as_of_datetime`` fence inside ``build_features``
        is currently DEAD because ``calculate_team_game_stats`` output lacks a
        ``kickoff_et`` column (WR-03, deferred -- do NOT activate it here).

        Returns:
            DataFrame with current week team form features
        """
        current_season, current_week = get_current_nfl_week()

        logger.info(
            "Building team form for current week (time-fenced Protocol path)",
            season=current_season,
            week=current_week,
        )

        try:
            # Friday 6 PM ET snapshot freeze (documented data freeze, CLAUDE.md /
            # PROJECT.md). The as_of_datetime is tz-aware ET so the time-fence is
            # evaluated against the canonical Eastern wall-clock.
            as_of_datetime = datetime.now(tz=ET)

            # The time-fenced Protocol path computes its own per-game stats from
            # play-by-play for [target_season - 1, target_season]; the games_df
            # argument is unused when target_season/target_week are supplied, so an
            # empty frame is sufficient (it never determines the season set here).
            form_df = self.calculator.build_features(
                pd.DataFrame(),
                as_of_datetime,
                target_season=current_season,
                target_week=current_week,
            )

            # Persist the current-week rolling rows to the silver
            # ``team_form_features`` table that ``scripts/build_features.py`` reads.
            # WR-01 invariant: the current-week path produces only the target's
            # rolling rows, so a full-history ``replace_mode=True`` write would
            # shrink the on-disk season span. Use a target-keyed upsert instead --
            # drop only the existing rows for THIS (target_season, target_week) and
            # append the freshly-built ones, preserving all prior history.
            if len(form_df) > 0:
                self._upsert_current_week_form(form_df, current_season, current_week)

            return form_df

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error(
                "Failed to build current week team form",
                season=current_season,
                week=current_week,
                error=str(e),
            )
            raise

    def _upsert_current_week_form(
        self, form_df: pd.DataFrame, target_season: int, target_week: int
    ) -> None:
        """Upsert current-week rolling rows into silver ``team_form_features``.

        Latest-wins on the ``(target_season, target_week)`` key: any existing rows
        for the target are dropped and replaced with the freshly-built ones, while
        all other on-disk history is preserved. This is the team-form analog of the
        ``upsert_silver`` latest-wins pattern; we cannot use ``replace_mode=True``
        (it would discard the rest of the table -- the WR-01 data-loss hazard) nor
        the default ``append_mode`` (team-form has no ``game_id`` key, so the
        append-merge would not de-duplicate and the table would grow each run).
        """
        try:
            existing = load_dataframe("team_form_features", layer="silver")
        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError):
            existing = pd.DataFrame()

        if len(existing) > 0 and {"target_season", "target_week"}.issubset(
            existing.columns
        ):
            mask = ~(
                (existing["target_season"] == target_season)
                & (existing["target_week"] == target_week)
            )
            combined = pd.concat([existing[mask], form_df], ignore_index=True)
        else:
            combined = form_df

        # replace_mode writes a single self-replacing file from the FULL combined
        # table (preserved history + refreshed target rows), keeping the write
        # idempotent without growing the lake or shrinking its span.
        save_dataframe(
            combined,
            "team_form_features",
            layer="silver",
            replace_mode=True,
        )
        logger.info(
            "Upserted current-week team_form_features (history preserved)",
            target_season=target_season,
            target_week=target_week,
            target_rows=len(form_df),
            total_rows=len(combined),
        )

    def update_with_new_games(self, new_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update team form features with new game results.

        Args:
            new_games_df: DataFrame with new game results

        Returns:
            Updated team form features
        """
        if len(new_games_df) == 0:
            logger.warning("No new games to update team form with")
            return pd.DataFrame()

        seasons = sorted(new_games_df["season"].unique())
        logger.info(
            "Updating team form with new games",
            games=len(new_games_df),
            seasons=seasons,
        )

        # Rebuild features for affected seasons
        return self.build_for_seasons(seasons)

    def get_team_matchup_features(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get team form features for a specific matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dictionary with matchup features
        """
        try:
            # Get individual team form
            team_forms = self.calculator.get_team_form_for_game(
                home_team, away_team, season, week
            )

            if not team_forms:
                return {}

            # Create matchup features by combining team forms
            matchup_features = {}

            home_form = team_forms.get(home_team, {})
            away_form = team_forms.get(away_team, {})

            # Direct team features
            for team_name, form in [
                (f"home_{home_team}", home_form),
                (f"away_{away_team}", away_form),
            ]:
                for metric, value in form.items():
                    if pd.notna(value):
                        matchup_features[f"{team_name}_{metric}"] = value

            # Matchup differentials (home offense vs away defense, etc.)
            try:
                # Home offense vs Away defense
                if (
                    "off_rolling_epa_per_play" in home_form
                    and "def_rolling_epa_per_play" in away_form
                ):
                    matchup_features["home_off_vs_away_def_epa"] = (
                        home_form["off_rolling_epa_per_play"]
                        - away_form["def_rolling_epa_per_play"]
                    )

                # Away offense vs Home defense
                if (
                    "off_rolling_epa_per_play" in away_form
                    and "def_rolling_epa_per_play" in home_form
                ):
                    matchup_features["away_off_vs_home_def_epa"] = (
                        away_form["off_rolling_epa_per_play"]
                        - home_form["def_rolling_epa_per_play"]
                    )

                # Overall EPA differential
                if (
                    "off_rolling_epa_per_play" in home_form
                    and "off_rolling_epa_per_play" in away_form
                ):
                    matchup_features["epa_differential"] = (
                        home_form["off_rolling_epa_per_play"]
                        - away_form["off_rolling_epa_per_play"]
                    )

                # Success rate differentials
                if (
                    "off_rolling_success_rate" in home_form
                    and "off_rolling_success_rate" in away_form
                ):
                    matchup_features["success_rate_differential"] = (
                        home_form["off_rolling_success_rate"]
                        - away_form["off_rolling_success_rate"]
                    )

            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
                logger.warning(
                    "Failed to calculate matchup differentials", error=str(e)
                )

            return matchup_features

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error(
                "Failed to get team matchup features",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def analyze_team_trends(self, team: str, seasons: list[int]) -> dict[str, Any]:
        """
        Analyze trends for a specific team across seasons.

        Args:
            team: Team abbreviation
            seasons: Seasons to analyze

        Returns:
            Dictionary with trend analysis
        """
        try:
            # Load team form features
            form_df = load_dataframe("team_form_features", layer="silver")

            # Filter to specific team and seasons
            team_data = form_df[
                (form_df["team"] == team) & (form_df["target_season"].isin(seasons))
            ]

            if len(team_data) == 0:
                return {"error": f"No data found for team {team}"}

            trends = {"team": team, "seasons_analyzed": seasons}

            # Analyze offensive trends
            offense_data = team_data[team_data["side"] == "offense"]
            if len(offense_data) > 0:
                trends["offense"] = {
                    "avg_epa_per_play": offense_data["rolling_epa_per_play"].mean(),
                    "avg_success_rate": offense_data["rolling_success_rate"].mean(),
                    "avg_pass_rate": offense_data["rolling_neutral_pass_rate"].mean(),
                    "consistency_epa": offense_data["rolling_epa_per_play"].std(),
                }

            # Analyze defensive trends
            defense_data = team_data[team_data["side"] == "defense"]
            if len(defense_data) > 0:
                trends["defense"] = {
                    "avg_epa_allowed": defense_data["rolling_epa_per_play"].mean(),
                    "avg_success_rate_allowed": defense_data[
                        "rolling_success_rate"
                    ].mean(),
                    "consistency_epa": defense_data["rolling_epa_per_play"].std(),
                }

            return trends

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error(
                "Failed to analyze team trends",
                team=team,
                seasons=seasons,
                error=str(e),
            )
            return {"error": str(e)}


def main():
    """CLI entry point for team form builder."""
    parser = argparse.ArgumentParser(description="Build NFL team form metrics")
    parser.add_argument("--season", type=int, help="Process single season")
    parser.add_argument(
        "--seasons", nargs="+", type=int, help="Process multiple seasons"
    )
    parser.add_argument(
        "--all-seasons", action="store_true", help="Process all available seasons"
    )
    parser.add_argument(
        "--current", action="store_true", help="Process current week only"
    )
    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=4,
        help="Number of weeks for rolling averages (default: 4)",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2020,
        help="Starting season for all-seasons build (default: 2020)",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing features"
    )
    parser.add_argument(
        "--analyze-team", type=str, help="Analyze trends for specific team"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging()

        builder = TeamFormBuilder(rolling_weeks=args.rolling_weeks)

        if args.validate_only:
            # Load and validate existing features
            try:
                form_df = load_dataframe("team_form_features", layer="silver")
                success = builder.calculator.validate_form_features(form_df)
                if success:
                    print("Team form features validation passed")
                else:
                    print("Team form features validation failed")
                    sys.exit(1)
            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
                print(f"No existing features found: {e}")
                sys.exit(1)
            return

        if args.analyze_team:
            # Analyze specific team trends
            seasons = [2022, 2023, 2024]  # Default recent seasons
            if args.seasons:
                seasons = args.seasons
            elif args.season:
                seasons = [args.season]

            trends = builder.analyze_team_trends(args.analyze_team, seasons)
            print(f"\nTrend analysis for {args.analyze_team}:")
            import json

            print(json.dumps(trends, indent=2, default=str))
            return

        # Determine what to process
        form_df = pd.DataFrame()

        if args.current:
            form_df = builder.build_for_current_week()
        elif args.all_seasons:
            current_season, _ = get_current_nfl_week()
            seasons = list(range(args.start_season, current_season + 1))
            form_df = builder.build_for_seasons(seasons)
        elif args.seasons:
            form_df = builder.build_for_seasons(args.seasons)
        elif args.season:
            form_df = builder.build_for_seasons([args.season])
        else:
            # Default: process recent seasons
            current_season, _ = get_current_nfl_week()
            seasons = [current_season - 1, current_season]
            form_df = builder.build_for_seasons(seasons)

        # Print summary
        if len(form_df) > 0:
            print("\nTeam form features built successfully!")
            print(f"Records: {len(form_df)}")
            print(f"Teams: {len(form_df['team'].unique())}")
            print(f"Seasons: {sorted(form_df['target_season'].unique())}")

            # Show sample of recent features
            recent_features = form_df[
                form_df["target_season"] == form_df["target_season"].max()
            ].head()

            if len(recent_features) > 0:
                print("\nSample recent features:")
                key_cols = [
                    "team",
                    "side",
                    "target_week",
                    "rolling_epa_per_play",
                    "rolling_success_rate",
                ]
                available_cols = [
                    col for col in key_cols if col in recent_features.columns
                ]
                print(recent_features[available_cols].to_string(index=False))
        else:
            print("No team form features generated")

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Team form building failed", error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
