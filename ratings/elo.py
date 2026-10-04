"""
Elo Rating System for NFL Teams

This module implements a comprehensive Elo rating system for NFL teams with:
- Margin-of-victory adjustments with dynamic K-factor
- Home-field advantage learning per season
- Season carryover with regression to mean
- Chronological updates across multiple seasons
- Optional Glicko-style uncertainty tracking

References:
- FiveThirtyEight NFL Elo methodology
- Elo rating system (Arpad Elo, 1978)
- Glicko rating system (Mark Glickman, 1995)
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from conf.settings import get_settings
from utils import get_logger

logger = get_logger(__name__)


# ---- NFL Division Data ----
# Uses canonical team abbreviations from utils/team_data.py:
#   LA = Rams (NFC West), LAC = Chargers (AFC West)
NFL_DIVISIONS: dict[str, list[str]] = {
    "AFC_East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC_North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC_South": ["HOU", "IND", "JAX", "TEN"],
    "AFC_West": ["DEN", "KC", "LAC", "LV"],
    "NFC_East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC_North": ["CHI", "DET", "GB", "MIN"],
    "NFC_South": ["ATL", "CAR", "NO", "TB"],
    "NFC_West": ["ARI", "LA", "SEA", "SF"],
}

# Divisional HFA multiplier: ~46% reduction from nfelo research
# Non-divisional HFA 2.95 pts, divisional HFA 1.59 pts -> 1.59/2.95 = 0.54
DIVISIONAL_HFA_FACTOR: float = 0.54

# Pre-build a team-to-division lookup for O(1) checks
_TEAM_TO_DIVISION: dict[str, str] = {}
for _div_name, _div_teams in NFL_DIVISIONS.items():
    for _team in _div_teams:
        _TEAM_TO_DIVISION[_team] = _div_name


def is_divisional_game(home_team: str, away_team: str) -> bool:
    """Check if two teams are in the same division.

    Args:
        home_team: Home team canonical abbreviation.
        away_team: Away team canonical abbreviation.

    Returns:
        True if both teams are in the same division.
    """
    home_div = _TEAM_TO_DIVISION.get(home_team)
    away_div = _TEAM_TO_DIVISION.get(away_team)
    if home_div is None or away_div is None:
        return False
    return home_div == away_div


# ---- Neutral sites (WINDOWS row 19, owner ruling 2026-10-03) ----
# A game silver ``games`` flags ``neutral_site == True`` gets NO home-field advantage --
# Super Bowls, international games, relocated games, all of them -- and zero overrides the
# divisional reduction. Such games are also left out of HFA LEARNING: a home win at a
# neutral site says nothing about home field. The rule is spelled once, here, through the
# three names below; every chain caller passes the raw flag and lets them validate it.


class NeutralSiteFlagError(ValueError):
    """A game's ``neutral_site`` flag is not a boolean.

    Refused rather than coerced: ``bool(float("nan"))`` is True, so coercing a null flag
    would silently make the game NEUTRAL and zero its home-field advantage.
    """


def is_neutral_site(value: object) -> bool:
    """THE one reader of a game's neutral-site flag.

    Args:
        value: The raw ``neutral_site`` cell of one game.

    Returns:
        The flag, for a Python or numpy boolean.

    Raises:
        NeutralSiteFlagError: Quoting the value, for None, pandas NA, a float NaN or any
            other non-boolean.
    """
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise NeutralSiteFlagError(
        f"a game's neutral_site flag must be a boolean, got {value!r} "
        f"({type(value).__name__}). A null or non-boolean flag is refused rather than "
        "coerced, because bool(NaN) is True and would make the game neutral."
    )


def hfa_learning_mask(games_df: pd.DataFrame, season: int) -> pd.Series:
    """THE one home-field-advantage learning filter for *season*.

    Selects the PRIOR season's games (``season - 1``) that have both scores, are not tied
    and were NOT played at a neutral site. Neutral games are excluded per the owner ruling
    of 2026-10-03 (WINDOWS row 19): a neutral site has no home field to learn from.

    Args:
        games_df: Games frame (may span seasons); must carry ``neutral_site``.
        season: The season HFA is being learned FOR.

    Returns:
        A boolean Series aligned to *games_df*'s index.

    Raises:
        KeyError: Naming ``neutral_site`` when the frame does not carry it.
        NeutralSiteFlagError: When a prior-season flag is null or non-boolean.
    """
    if "neutral_site" not in games_df.columns:
        raise KeyError(
            "the frame handed to the HFA learner has no 'neutral_site' column, so neutral "
            f"games cannot be left out of HFA learning (columns: {sorted(games_df.columns)})"
        )
    prior = (games_df["season"] == season - 1).to_numpy(dtype=bool)
    flags = games_df["neutral_site"].to_numpy(dtype=object)
    neutral = np.zeros(len(games_df), dtype=bool)
    for position in np.flatnonzero(prior):
        neutral[position] = is_neutral_site(flags[position])
    decided = (
        games_df["home_score"].notna()
        & games_df["away_score"].notna()
        & (games_df["home_score"] != games_df["away_score"])
    ).to_numpy(dtype=bool)
    return pd.Series(prior & decided & ~neutral, index=games_df.index)


@dataclass
class EloRating:
    """Individual team Elo rating with metadata."""

    team: str
    rating: float = 1500.0
    games_played: int = 0
    last_updated: datetime | None = None
    uncertainty: float = 350.0  # Glicko-style rating deviation
    season: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        EVERY NUMERIC FIELD IS CAST TO A NATIVE PYTHON TYPE HERE, and that is load
        bearing rather than tidy. A JSON encoder called with ``default=str`` silently
        STRINGIFIES a numpy scalar. That is how the persisted Elo state file (deleted
        with the legacy pass, D33.2-22) came to carry ``"season": "2025"``, and a
        string season makes
        ``apply_season_carryover``'s ``rating.season < season`` raise ``TypeError: '<'
        not supported between instances of 'str' and 'int'`` the moment the current
        season is carried forward (Plan 33-03).
        """
        return {
            "team": str(self.team),
            "rating": float(self.rating),
            "games_played": int(self.games_played),
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "uncertainty": float(self.uncertainty),
            "season": int(self.season) if self.season is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EloRating":
        """Create from dictionary.

        Coerces on the way IN as well as on the way out, so a rating file written by an
        older build -- every one on disk today -- loads with an integer season rather
        than the string that file actually holds. The input mapping is copied because
        mutating a caller's dict while parsing it is a surprise nobody asked for.
        """
        data = dict(data)
        if data.get("last_updated"):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        if data.get("season") is not None:
            data["season"] = int(data["season"])
        if data.get("games_played") is not None:
            data["games_played"] = int(data["games_played"])
        for numeric in ("rating", "uncertainty"):
            if data.get(numeric) is not None:
                data[numeric] = float(data[numeric])
        return cls(**data)


class EloRatingSystem:
    """
    NFL Elo Rating System with advanced features.

    Features:
    - Base Elo calculations (initialized at 1500)
    - Margin-of-victory adjustments with dynamic K-factor
    - Home-field advantage learning per season
    - Season carryover with regression to mean (25 Elo shrinkage)
    - Chronological updates across multiple seasons
    - Optional Glicko-style uncertainty tracking
    """

    def __init__(
        self,
        base_k: float = 20.0,
        hfa_init: float = 48.0,
        mov_multiplier: float = 2.2,
        season_carryover: float = 0.75,
        uncertainty_decay: float = 15.0,
        min_uncertainty: float = 50.0,
        max_uncertainty: float = 350.0,
    ):
        """
        Initialize Elo rating system.

        Args:
            base_k: Base K-factor for rating updates
            hfa_init: Initial home field advantage
            mov_multiplier: Margin of victory multiplier
            season_carryover: Fraction of rating carried over to next season (0.75 = 25 point regression)
            uncertainty_decay: How much uncertainty increases per game
            min_uncertainty: Minimum rating uncertainty
            max_uncertainty: Maximum rating uncertainty
        """
        self.base_k = base_k
        self.hfa_init = hfa_init
        self.mov_multiplier = mov_multiplier
        self.season_carryover = season_carryover
        self.uncertainty_decay = uncertainty_decay
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty

        # Current ratings by team
        self.ratings: dict[str, EloRating] = {}

        # Home field advantage by season
        self.hfa_by_season: dict[int, float] = {}

        # Game history for analysis
        self.game_history: list[dict[str, Any]] = []

        self.settings = get_settings()

    def _expected_score(
        self, rating_a: float, rating_b: float, hfa: float = 0.0
    ) -> float:
        """
        Calculate expected score for team A vs team B.

        Args:
            rating_a: Team A's Elo rating
            rating_b: Team B's Elo rating
            hfa: Home field advantage for team A (if home)

        Returns:
            Expected score (0-1) for team A
        """
        rating_diff = rating_a - rating_b + hfa
        return 1 / (1 + 10 ** (-rating_diff / 400))

    def _calculate_k_factor(
        self, base_k: float, mov: int, elo_diff: float, uncertainty: float | None = None
    ) -> float:
        """
        Calculate dynamic K-factor based on margin of victory and rating difference.

        Uses FiveThirtyEight methodology with uncertainty adjustment.

        Args:
            base_k: Base K-factor
            mov: Margin of victory (absolute)
            elo_diff: Rating difference (winner - loser)
            uncertainty: Rating uncertainty (for Glicko-style adjustment)

        Returns:
            Adjusted K-factor
        """
        # Base margin of victory adjustment
        mov_factor = math.log(max(1, mov)) + 1.0

        # Elo difference adjustment (blowout of strong team less impressive)
        elo_factor = 2.2 / ((elo_diff * 0.001) + 2.2)

        # Uncertainty adjustment (higher K for uncertain ratings)
        uncertainty_factor = 1.0
        if uncertainty is not None:
            uncertainty_factor = 1 + (uncertainty - self.min_uncertainty) / (
                self.max_uncertainty - self.min_uncertainty
            )
            uncertainty_factor = max(
                0.5, min(2.0, uncertainty_factor)
            )  # Clamp between 0.5-2.0

        return base_k * mov_factor * elo_factor * uncertainty_factor

    def _update_uncertainty(self, rating: EloRating, k_factor: float) -> None:
        """Update rating uncertainty after a game (Glicko-style)."""
        # Uncertainty decreases with game experience but increases over time
        rating.uncertainty = max(
            self.min_uncertainty,
            rating.uncertainty * 0.95 - (k_factor * 0.5) + self.uncertainty_decay,
        )
        rating.uncertainty = min(self.max_uncertainty, rating.uncertainty)

    def get_or_create_rating(self, team: str, season: int) -> EloRating:
        """Get existing rating or create new one for team."""
        if team not in self.ratings:
            self.ratings[team] = EloRating(
                team=team,
                rating=1500.0,
                season=season,
                uncertainty=self.max_uncertainty,
            )
        return self.ratings[team]

    def apply_season_carryover(self, season: int) -> None:
        """
        Apply season carryover (regression to mean) for all teams.

        Args:
            season: New season to apply carryover for
        """
        logger.info(
            f"Applying season carryover for {season}",
            carryover_factor=self.season_carryover,
        )

        for team, rating in self.ratings.items():
            if rating.season and rating.season < season:
                # Regress to mean (1500) by (1 - carryover_factor)
                old_rating = rating.rating
                rating.rating = rating.rating * self.season_carryover + 1500 * (
                    1 - self.season_carryover
                )

                # Reset uncertainty to higher value for new season
                rating.uncertainty = min(
                    self.max_uncertainty, rating.uncertainty + 50.0
                )
                rating.season = season

                logger.debug(
                    f"Season carryover for {team}",
                    old_rating=old_rating,
                    new_rating=rating.rating,
                )

    def home_field_advantage(
        self, season: int, *, neutral_site: object, is_divisional: bool
    ) -> float:
        """THE one home-field-advantage rule for one game (WINDOWS row 19).

        0.0 at a neutral site, overriding the divisional reduction; otherwise the
        season's learned HFA (``hfa_init`` before one is learned), times
        ``DIVISIONAL_HFA_FACTOR`` for a divisional game.

        Args:
            season: Season year.
            neutral_site: The game's raw neutral-site flag, read through
                :func:`is_neutral_site` (a null or non-boolean flag is refused).
            is_divisional: Whether this is a divisional game.

        Returns:
            The HFA in Elo points.
        """
        if is_neutral_site(neutral_site):
            return 0.0
        hfa = self.hfa_by_season.get(season, self.hfa_init)
        return hfa * DIVISIONAL_HFA_FACTOR if is_divisional else hfa

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        season: int,
        game_date: datetime,
        game_id: str | None = None,
        is_divisional: bool = False,
        neutral_site: bool = False,
    ) -> tuple[float, float]:
        """Update Elo ratings for both teams after a game.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            home_score: Home team score.
            away_score: Away team score.
            season: Season year.
            game_date: Game date.
            game_id: Optional game identifier.
            is_divisional: Whether this is a divisional game. Divisional
                games get reduced HFA (multiplied by DIVISIONAL_HFA_FACTOR).
            neutral_site: Whether the game was played at a neutral site (zero HFA,
                overriding the divisional reduction). Defaults to False, mirroring
                ``predict_game``; every chain caller passes the game's raw flag.

        Returns:
            Tuple of (home_rating_change, away_rating_change).
        """
        # Get or create ratings
        home_rating = self.get_or_create_rating(home_team, season)
        away_rating = self.get_or_create_rating(away_team, season)

        hfa = self.home_field_advantage(
            season, neutral_site=neutral_site, is_divisional=is_divisional
        )

        # Pre-game ratings
        home_pre = home_rating.rating
        away_pre = away_rating.rating

        # Expected scores
        home_expected = self._expected_score(home_pre, away_pre, hfa)
        away_expected = 1 - home_expected

        # Actual scores (0-1)
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif away_score > home_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5  # Tie

        # Margin of victory
        mov = abs(home_score - away_score)

        # Calculate K-factors
        elo_diff = abs(home_pre - away_pre)
        home_k = self._calculate_k_factor(
            self.base_k, mov, elo_diff, home_rating.uncertainty
        )
        away_k = self._calculate_k_factor(
            self.base_k, mov, elo_diff, away_rating.uncertainty
        )

        # Update ratings
        home_change = home_k * (home_actual - home_expected)
        away_change = away_k * (away_actual - away_expected)

        home_rating.rating += home_change
        away_rating.rating += away_change

        # Update metadata
        home_rating.games_played += 1
        away_rating.games_played += 1
        home_rating.last_updated = game_date
        away_rating.last_updated = game_date

        # Update uncertainties
        self._update_uncertainty(home_rating, home_k)
        self._update_uncertainty(away_rating, away_k)

        # Log the update
        logger.debug(
            f"Elo update: {game_id or 'Unknown'}",
            home_team=home_team,
            away_team=away_team,
            home_change=home_change,
            away_change=away_change,
            home_new=home_rating.rating,
            away_new=away_rating.rating,
        )

        # Store game history
        self.game_history.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_rating_pre": home_pre,
                "away_rating_pre": away_pre,
                "home_rating_post": home_rating.rating,
                "away_rating_post": away_rating.rating,
                "home_change": home_change,
                "away_change": away_change,
                "hfa_used": hfa,
                "mov": mov,
            }
        )

        return home_change, away_change

    def learn_home_field_advantage(self, games_df: pd.DataFrame, season: int) -> float:
        """Learn home field advantage from PRIOR season data only (no lookahead).

        For the first season (no prior-season data available), returns
        hfa_init (48). For subsequent seasons, learns HFA from the prior
        season's home win rate and blends with the running estimate.

        The games learned from are :func:`hfa_learning_mask`'s: the prior season's
        scored, non-tied, NON-NEUTRAL games. Neutral-site games are excluded per the
        owner ruling of 2026-10-03 (WINDOWS row 19).

        Args:
            games_df: DataFrame with game results (may span multiple seasons); must
                carry ``neutral_site``.
            season: Season to learn HFA for. Uses season-1 data.

        Returns:
            Learned home field advantage value (Elo points).
        """
        # Filter to PRIOR season only -- this is the key fix for the
        # lookahead bug. Never use same-season data for HFA learning.
        prior_games = games_df[hfa_learning_mask(games_df, season)]

        if len(prior_games) == 0:
            # No prior-season data -- use initial value
            self.hfa_by_season[season] = self.hfa_init
            return self.hfa_init

        home_wins = (prior_games["home_score"] > prior_games["away_score"]).sum()
        home_win_rate = home_wins / len(prior_games)

        # Convert win rate to Elo points (approximately)
        # 50% win rate = 0 Elo advantage
        # Each 1% above 50% ~ 8 Elo points
        hfa_from_data = (home_win_rate - 0.5) * 800

        # Smooth with previous HFA estimate and clamp to reasonable range
        prev_hfa = self.hfa_by_season.get(season - 1, self.hfa_init)
        learned_hfa = 0.7 * hfa_from_data + 0.3 * prev_hfa
        learned_hfa = max(20.0, min(80.0, learned_hfa))  # Clamp between 20-80

        self.hfa_by_season[season] = learned_hfa

        logger.info(
            f"Learned HFA for {season} from {season - 1} data",
            prior_games_analyzed=len(prior_games),
            home_win_rate=home_win_rate,
            raw_hfa=hfa_from_data,
            final_hfa=learned_hfa,
        )

        return learned_hfa

    def process_season_chronologically(
        self, games_df: pd.DataFrame, season: int, learn_hfa: bool = True
    ) -> pd.DataFrame:
        """
        Process all games in a season chronologically.

        Args:
            games_df: DataFrame with games for the season
            season: Season year
            learn_hfa: Whether to learn home field advantage from data

        Returns:
            DataFrame with pre/post game ratings added
        """
        logger.info(
            f"Processing {season} season chronologically", total_games=len(games_df)
        )

        # Apply season carryover if this is a new season
        if season not in self.hfa_by_season:
            self.apply_season_carryover(season)

        # Learn home field advantage for this season
        if learn_hfa:
            self.learn_home_field_advantage(games_df, season)

        # Sort games chronologically
        games_sorted = games_df.sort_values("kickoff_et").copy()

        # Process each completed game
        rating_updates = []

        for _idx, game in games_sorted.iterrows():
            # Skip games without results
            if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
                continue

            home_team = game["home_team"]
            away_team = game["away_team"]

            # Get pre-game ratings
            home_rating_pre = self.get_or_create_rating(home_team, season).rating
            away_rating_pre = self.get_or_create_rating(away_team, season).rating

            # Auto-detect divisional games using static division lookup
            divisional = is_divisional_game(home_team, away_team)

            # Update ratings
            home_change, away_change = self.update_ratings(
                home_team=home_team,
                away_team=away_team,
                home_score=int(game["home_score"]),
                away_score=int(game["away_score"]),
                season=season,
                game_date=game["kickoff_et"],
                game_id=game["game_id"],
                is_divisional=divisional,
                neutral_site=game["neutral_site"],
            )

            # Get post-game ratings
            home_rating_post = self.ratings[home_team].rating
            away_rating_post = self.ratings[away_team].rating

            rating_updates.append(
                {
                    "game_id": game["game_id"],
                    "home_rating_pre": home_rating_pre,
                    "away_rating_pre": away_rating_pre,
                    "home_rating_post": home_rating_post,
                    "away_rating_post": away_rating_post,
                    "home_change": home_change,
                    "away_change": away_change,
                }
            )

        # Merge rating updates back into games DataFrame
        if rating_updates:
            updates_df = pd.DataFrame(rating_updates)
            games_sorted = games_sorted.merge(updates_df, on="game_id", how="left")

        logger.info(
            f"Completed {season} season processing", games_processed=len(rating_updates)
        )

        return games_sorted

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        neutral_site: bool = False,
        is_divisional: bool = False,
    ) -> dict[str, float]:
        """Predict game outcome using current Elo ratings.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            season: Season year.
            neutral_site: Whether game is at neutral site (zero HFA, overriding the
                divisional reduction).
            is_divisional: Whether this is a divisional game. Divisional
                games get reduced HFA (multiplied by DIVISIONAL_HFA_FACTOR).

        Returns:
            Dictionary with win probabilities and rating info.
        """
        home_rating = self.get_or_create_rating(home_team, season)
        away_rating = self.get_or_create_rating(away_team, season)

        hfa = self.home_field_advantage(
            season, neutral_site=neutral_site, is_divisional=is_divisional
        )

        home_win_prob = self._expected_score(
            home_rating.rating, away_rating.rating, hfa
        )
        away_win_prob = 1 - home_win_prob

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_rating": home_rating.rating,
            "away_rating": away_rating.rating,
            "rating_diff": home_rating.rating - away_rating.rating,
            "hfa_used": hfa,
            "home_uncertainty": home_rating.uncertainty,
            "away_uncertainty": away_rating.uncertainty,
        }

    def get_current_ratings(self, season: int | None = None) -> pd.DataFrame:
        """
        Get current ratings for all teams.

        Args:
            season: Optional season filter

        Returns:
            DataFrame with current team ratings
        """
        ratings_data = []
        for team, rating in self.ratings.items():
            if season is None or rating.season == season:
                ratings_data.append(
                    {
                        "team": team,
                        "rating": rating.rating,
                        "games_played": rating.games_played,
                        "uncertainty": rating.uncertainty,
                        "season": rating.season,
                        "last_updated": rating.last_updated,
                    }
                )

        df = pd.DataFrame(ratings_data)
        if len(df) > 0:
            return df.sort_values("rating", ascending=False)
        return df
