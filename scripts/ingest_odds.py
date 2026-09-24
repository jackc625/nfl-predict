"""Odds data ingestion from external APIs.

ONE REQUEST, FOUR DISTINCT INSTANTS (Plan 33.2-02, D33.2-01)
-------------------------------------------------------------
The live capture used to take ONE ``snapshot_time`` and use it for four incompatible
jobs. They are kept apart now, and the plumbing names each one:

* the KICKOFF-time SELECTION window (``commence_from`` / ``commence_to``) -- which games
  to ask the API about. The Odds API filters ``commenceTimeFrom`` / ``commenceTimeTo`` on
  KICKOFF, so this is a game-selection window and never a time fence. It is a request
  parameter and is never stored;
* the observed CAPTURE instant (``captured_at``) -- read once, when the response
  returns, and stored as ``created_at``. It is never supplied by a caller;
* each row's UPSTREAM bookmaker time (``last_update``) -- NULL when the bookmaker gives
  none or an unparseable one, never our own instant;
* each game's own LOCK (``snapshot_ts``) -- ``utils.game_lock.lock_frame`` over the
  matched schedule rows, derived only after the payload game has been matched to its
  silver ``games`` row. The schedule, not the payload, says when a game starts.

All three stored values already have ``OddsSchema`` columns, so there is no schema
change. The public input is the slate's ``schedule``: the lock map is keyed by game ids
that only exist after the match, so no caller could supply it.
"""

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import utils.game_lock as lock_rule
from conf.settings import get_settings
from data.schemas import OddsSchema
from data.storage import (
    append_odds_captures,
    get_db_connection,
    load_dataframe,
    save_dataframe,
)
from utils import (
    DataIngestionError,
    ExternalAPIError,
    get_current_nfl_week,
    get_logger,
    log_data_operation,
)
from utils.game_id_utils import is_valid_game_id

logger = get_logger(__name__)

# The kickoff SELECTION window is the slate's own kickoff span widened by this much on
# each side, so a game whose listed commence_time drifts from the schedule by a few hours
# is still returned and then REPORTED, rather than silently missing from the response.
COMMENCE_WINDOW_MARGIN = timedelta(hours=12)

# A payload commence_time further than this from the matched schedule kickoff is
# reported by game id as a possible schedule move (R8 / D33.2-04). The schedule kickoff
# is used either way; resolving a move is not an ingest script's job.
COMMENCE_TIME_TOLERANCE = timedelta(hours=1)

# The silver ``games`` columns the per-game match and the lock need.
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff_et",
)


def _observe_capture_instant() -> datetime:
    """The instant a response is OBSERVED, read once per request.

    A named seam so a test fixes the capture instant by patching the clock, never by
    passing a value: an instant a caller or operator could assert would be a
    manufactured timestamp (RESEARCH P1).
    """
    return datetime.now(UTC)


def _parse_commence_time(value: Any) -> datetime | None:
    """A payload ``commence_time`` as a UTC instant, or None when absent or unparseable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _parse_upstream_instant(value: Any) -> datetime | None:
    """A payload ``last_update`` as an aware instant, or None when absent, unparseable or naive.

    An upstream time the payload does not give is UNKNOWN and stays NULL; it is never filled
    with our own capture instant (RESEARCH P1).
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo is not None else None


def latest_market_update(bookmaker: Mapping[str, Any]) -> datetime | None:
    """The latest per-market ``last_update`` a bookmaker entry carries (Plan 33.2-27).

    Real Odds API payloads stamp every market (h2h, spreads, totals) with its own
    ``last_update`` (measured on bronze ``odds_raw_bronze_2025_W01.parquet``). One stored row
    combines a bookmaker's markets, so it keeps the LATEST of those stamps: none of the row's
    prices is fresher than that. None when no market carries a parseable aware stamp.
    """
    stamps = [
        stamp
        for market in bookmaker.get("markets", [])
        if (stamp := _parse_upstream_instant(market.get("last_update"))) is not None
    ]
    return max(stamps) if stamps else None


def load_schedule_slice(season: int, week: int) -> pd.DataFrame:
    """The silver ``games`` rows for one slate, READ ONLY, in the shape the ingest needs.

    Raises:
        DataIngestionError: when the slate has no scheduled game. A request with no
            schedule has no per-game lock to stamp, and guessing one is the defect.
    """
    games = load_dataframe("games", layer="silver")
    missing = [c for c in SCHEDULE_COLUMNS if c not in games.columns]
    if missing:
        msg = f"silver games is missing {missing}; the odds ingest cannot match games"
        raise DataIngestionError(msg)
    slate = games[(games["season"] == season) & (games["week"] == week)]
    if slate.empty:
        msg = (
            f"no scheduled games for {season} week {week} in silver games; the odds "
            "ingest refuses to run without a schedule to derive per-game locks from"
        )
        raise DataIngestionError(msg)
    return cast("pd.DataFrame", slate[list(SCHEDULE_COLUMNS)]).reset_index(drop=True)


@dataclass(frozen=True)
class LiveOddsMatchReport:
    """What the last transform could not match, and where payload and schedule disagreed.

    Attributes:
        unmatched_games: Payload games with no schedule row, named by teams and listed
            commence_time. Each was SKIPPED: it has no kickoff, so it has no lock.
        kickoff_disagreements: Game ids whose payload commence_time differs from the
            schedule kickoff by more than :data:`COMMENCE_TIME_TOLERANCE`. The schedule
            kickoff was used for every one of them.
    """

    unmatched_games: tuple[str, ...] = ()
    kickoff_disagreements: tuple[str, ...] = ()


# Sentinel substituted for the live API key in any logged params dict so the
# secret never reaches the DEBUG log (review 29-03 MED).
_API_KEY_REDACTION = "***REDACTED***"


def _redact_api_key(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a shallow copy of *params* with ``apiKey`` redacted.

    Defense-in-depth so the live ``ODDS_API_KEY`` never appears in the DEBUG
    params log even when DEBUG logging is enabled (review 29-03 MED). The
    original params dict passed to the HTTP client is unchanged.
    """
    if not params or "apiKey" not in params:
        return params
    return {**params, "apiKey": _API_KEY_REDACTION}


class OddsAPIClient:
    """Client for fetching odds from external APIs."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize odds API client.

        Args:
            api_key: API key for odds service
            base_url: Base URL for odds API
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.odds_api_key
        self.base_url = (
            base_url or self.settings.config.external_apis.odds_api["base_url"]
        )

        if not self.api_key:
            logger.warning("No odds API key provided - using mock data")
            self.mock_mode = True
        else:
            self.mock_mode = False

        # API configuration
        self.timeout = self.settings.config.external_apis.odds_api["timeout"]
        self.retries = self.settings.config.external_apis.odds_api["retries"]
        self.rate_limit = self.settings.config.external_apis.odds_api[
            "rate_limit_per_hour"
        ]

        # HTTP client
        self.client = httpx.Client(timeout=self.timeout)

    def close(self):
        """Close HTTP client."""
        self.client.close()

    # WR-11: RETRY ONLY ExternalAPIError. tenacity's default retries EVERY
    # exception, so narrowing the except arms below is not by itself enough --
    # a MemoryError, a KeyboardInterrupt-adjacent bug or any other programming
    # error escaping the body was still retried three times with exponential
    # backoff, i.e. three PAID HTTP calls to recover from something that cannot
    # be recovered from. Every recoverable failure is already mapped to
    # ExternalAPIError inside the body, so this predicate loses no retry coverage.
    @retry(
        retry=retry_if_exception_type(ExternalAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        with_headers: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], httpx.Headers]:
        """Make an HTTP request to the odds API, with retries.

        WR-11: this used to have a near-verbatim twin, ``_make_request_with_headers``
        -- about 45 lines duplicating this method character-for-character apart
        from the return value (same retry decorator, same mock-mode branch, same
        three except arms, same messages) -- with no test on the copy and no
        production caller. Any change to auth, retry policy or error mapping had to
        be made twice. One implementation, one optional return shape.

        Args:
            endpoint: Path under ``base_url``.
            params: Query parameters; ``apiKey`` is added here.
            with_headers: When True return ``(json, response.headers)`` so the
                caller can read the Odds API credit headers
                (``x-requests-last`` / ``x-requests-remaining``) that the paid
                backfill's cost guard depends on.

        Returns:
            The decoded JSON, or ``(json, headers)`` when ``with_headers`` is True.

        Raises:
            ExternalAPIError: On an HTTP status error, a transport error, or a
                malformed response body.
        """
        if self.mock_mode:
            mock = self._generate_mock_odds()
            return (mock, httpx.Headers({})) if with_headers else mock

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        try:
            # Redact the live apiKey before logging params at DEBUG -- the key
            # must never reach the log even with DEBUG enabled (review 29-03 MED;
            # defense-in-depth beyond "keep DEBUG off in automation").
            logger.debug(
                "Making odds API request", url=url, params=_redact_api_key(params)
            )

            response = self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            logger.info(
                "Odds API request successful", url=url, status_code=response.status_code
            )

            return (data, response.headers) if with_headers else data

        except httpx.HTTPStatusError as e:
            logger.error(
                "Odds API HTTP error",
                url=url,
                status_code=e.response.status_code,
                response=e.response.text,
            )
            raise ExternalAPIError(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            ) from e

        except httpx.RequestError as e:
            logger.error("Odds API request error", url=url, error=str(e))
            raise ExternalAPIError(f"Request error: {e}") from e

        # WR-11: NARROWED from a bare `except Exception`. A blanket catch here
        # converted programming errors (a KeyError, a typo in the response
        # handling) into ExternalAPIError, which tenacity then RETRIED three times
        # with exponential backoff -- three PAID HTTP calls to recover from a bug
        # that cannot be recovered from. These three are the genuine
        # malformed-body failures; anything else propagates un-retried, with its
        # own traceback. `from e` preserves the cause, which neither copy did.
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Odds API malformed response", url=url, error=str(e))
            raise ExternalAPIError(f"Malformed response: {e}") from e

    def _generate_mock_odds(
        self, season: int | None = None, week: int | None = None
    ) -> dict[str, Any]:
        """Generate mock odds data for testing using real historical games."""
        import random

        from data.storage import get_parquet_manager

        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week

        # Load real games data for the specified season/week
        try:
            pm = get_parquet_manager()
            games_df = pm.load("silver/games.parquet")

            # Filter for the specific season and week
            season_games = games_df[
                (games_df["season"] == season) & (games_df["week"] == week)
            ].copy()

            if season_games.empty:
                logger.warning(
                    f"No games found for season {season} week {week}, generating minimal mock data"
                )
                # Fallback to basic mock data
                season_games = self._generate_basic_mock_games()
            else:
                logger.info(
                    f"Generating mock odds for {len(season_games)} real games from {season} Week {week}"
                )

        except Exception as e:
            logger.warning(
                f"Could not load games data: {e}, generating basic mock data"
            )
            season_games = self._generate_basic_mock_games()

        # Generate realistic odds for each game
        mock_games = []
        for idx, game in season_games.iterrows():
            game_id = game.get("game_id", f"mock_game_{idx}")
            home_team = game.get("home_team", "HOME")
            away_team = game.get("away_team", "AWAY")

            # Generate realistic spreads and totals based on team strength
            base_spread = random.uniform(-14.0, 14.0)  # Home team spread
            base_total = random.uniform(38.0, 58.0)  # Game total

            # Create odds for multiple sportsbooks
            bookmakers = []
            sportsbook_names = [
                "draftkings",
                "fanduel",
                "mybookieag",
                "betus",
                "betonlineag",
                "lowvig",
                "betrivers",
                "bovada",
                "betmgm",
            ]

            for book_key in sportsbook_names:
                # Add slight variations between sportsbooks
                spread_variation = random.uniform(-1.0, 1.0)
                total_variation = random.uniform(-2.0, 2.0)

                game_spread = round(base_spread + spread_variation, 1)
                game_total = round(base_total + total_variation, 1)

                bookmaker = {
                    "key": book_key,
                    "title": book_key.replace("ag", ".ag").title(),
                    "last_update": datetime.now().isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": home_team,
                                    "price": random.randint(-200, -105),
                                },
                                {
                                    "name": away_team,
                                    "price": random.randint(-200, -105),
                                },
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {
                                    "name": home_team,
                                    "price": random.randint(-115, -105),
                                    "point": game_spread,
                                },
                                {
                                    "name": away_team,
                                    "price": random.randint(-115, -105),
                                    "point": -game_spread,
                                },
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "price": random.randint(-115, -105),
                                    "point": game_total,
                                },
                                {
                                    "name": "Under",
                                    "price": random.randint(-115, -105),
                                    "point": game_total,
                                },
                            ],
                        },
                    ],
                }
                bookmakers.append(bookmaker)

            mock_game = {
                "id": game_id,
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": datetime.now().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "bookmakers": bookmakers,
            }
            mock_games.append(mock_game)

        logger.info(
            "Generated mock odds data", games=len(mock_games), season=season, week=week
        )
        return mock_games

    def _generate_basic_mock_games(self):
        """Generate basic mock games when real games data is unavailable."""
        import pandas as pd

        teams = ["BUF", "KC", "DAL", "SF", "GB", "MIA", "LV", "PHI", "LAR", "MIN"]

        games_data = []
        for i in range(3):
            games_data.append(
                {
                    "game_id": f"mock_game_{i}",
                    "home_team": teams[i % len(teams)],
                    "away_team": teams[(i + 5) % len(teams)],
                }
            )

        return pd.DataFrame(games_data)

    def get_nfl_odds(
        self,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch NFL odds from API.

        Args:
            markets: List of markets to fetch ('h2h', 'spreads', 'totals')
            bookmakers: List of bookmaker keys
            date_from: Start date for games
            date_to: End date for games

        Returns:
            List of game odds data
        """
        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        params = {
            "sport": "americanfootball_nfl",
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        if date_from:
            # Convert to UTC and format for The Odds API (requires YYYY-MM-DDTHH:MM:SSZ format)

            utc_time = date_from.astimezone(UTC)
            params["commenceTimeFrom"] = utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        if date_to:
            # Convert to UTC and format for The Odds API

            utc_time = date_to.astimezone(UTC)
            params["commenceTimeTo"] = utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info("Fetching NFL odds", markets=markets, bookmakers=bookmakers)

        # Use different endpoint based on API structure
        endpoint = "sports/americanfootball_nfl/odds"

        data = self._make_request(endpoint, params)

        # Handle different response formats
        if isinstance(data, list):
            games = data
        elif isinstance(data, dict) and "data" in data:
            games = data["data"]
        else:
            games = []

        logger.info("Fetched NFL odds", total_games=len(games))
        return games

    def get_historical_nfl_odds(
        self,
        date_iso: str,
        markets: list[str],
        regions: str = "us",
        return_headers: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], httpx.Headers]:
        """Fetch ONE historical NFL odds snapshot at/near ``date_iso``.

        Calls the PAID historical endpoint
        ``historical/sports/americanfootball_nfl/odds`` through the inherited
        ``_make_request`` (auth via ``apiKey`` + tenacity retry), returning the
        ``{timestamp, previous_timestamp, next_timestamp, data}`` envelope. The
        envelope's ``timestamp`` is the ACTUAL snapshot at/earlier than the
        requested ``date_iso`` (the closest available, per the Odds API v4 docs)
        -- callers must stamp the stored row from that envelope ``timestamp``,
        NOT the requested ``date_iso`` (review 29-03 HIGH).

        One call returns the WHOLE NFL board for that snapshot. Cost =
        ``10 x len(markets) x 1 region`` credits per call, PAID plan ONLY.

        Args:
            date_iso: ISO8601 ``Z`` timestamp, e.g. ``2021-10-15T22:00:00Z``.
            markets: Markets to fetch (e.g. ``["totals"]`` or
                ``["totals", "spreads"]``).
            regions: Odds region (default ``"us"``).
            return_headers: When ``True``, return ``(envelope, response.headers)``
                so the backfill's cost guard can read ``x-requests-last`` /
                ``x-requests-remaining`` (review 29-05 HIGH). Default ``False``
                preserves the JSON-only return.

        Returns:
            The trajectory envelope dict, or ``(envelope, headers)`` when
            ``return_headers=True``.
        """
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": date_iso,
        }
        endpoint = "historical/sports/americanfootball_nfl/odds"

        logger.info(
            "Fetching historical NFL odds snapshot",
            date=date_iso,
            markets=markets,
            regions=regions,
        )

        return self._make_request(endpoint, params, with_headers=return_headers)


class OddsDataIngester:
    """NFL odds data ingestion and processing."""

    def __init__(self, api_key: str | None = None):
        """Initialize odds data ingester."""
        self.settings = get_settings()
        self.db = get_db_connection()
        self.api_client = OddsAPIClient(api_key)

        # Sportsbook priority for consensus odds
        self.sportsbook_priority = [
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "pointsbet",
            "barstool",
            "unibet",
        ]

    def close(self):
        """Close API client."""
        self.api_client.close()

    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to canonical abbreviation."""
        # Team name mapping for odds APIs
        team_mapping = {
            # Common variations in odds APIs
            "Kansas City Chiefs": "KC",
            "Buffalo Bills": "BUF",
            "Miami Dolphins": "MIA",
            "New England Patriots": "NE",
            "Baltimore Ravens": "BAL",
            "Cincinnati Bengals": "CIN",
            "Cleveland Browns": "CLE",
            "Pittsburgh Steelers": "PIT",
            "Houston Texans": "HOU",
            "Indianapolis Colts": "IND",
            "Jacksonville Jaguars": "JAX",
            "Tennessee Titans": "TEN",
            "Denver Broncos": "DEN",
            "Las Vegas Raiders": "LV",
            "Los Angeles Chargers": "LAC",
            "Chicago Bears": "CHI",
            "Detroit Lions": "DET",
            "Green Bay Packers": "GB",
            "Minnesota Vikings": "MIN",
            "Atlanta Falcons": "ATL",
            "Carolina Panthers": "CAR",
            "New Orleans Saints": "NO",
            "Tampa Bay Buccaneers": "TB",
            "Dallas Cowboys": "DAL",
            "New York Giants": "NYG",
            "Philadelphia Eagles": "PHI",
            "Washington Commanders": "WAS",
            "Arizona Cardinals": "ARI",
            "Los Angeles Rams": "LA",
            "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA",
            "New York Jets": "NYJ",
        }

        # Try direct mapping first
        if team in team_mapping:
            return team_mapping[team]

        # Try to extract abbreviation
        team_upper = team.upper().strip()

        # Common abbreviation patterns
        abbrev_mapping = {
            "KANSAS CITY": "KC",
            "KC": "KC",
            "BUFFALO": "BUF",
            "BUF": "BUF",
            "NEW ENGLAND": "NE",
            "NE": "NE",
            "MIAMI": "MIA",
            "MIA": "MIA",
            "BALTIMORE": "BAL",
            "BAL": "BAL",
            "CINCINNATI": "CIN",
            "CIN": "CIN",
            "CLEVELAND": "CLE",
            "CLE": "CLE",
            "PITTSBURGH": "PIT",
            "PIT": "PIT",
            "LAS VEGAS": "LV",
            "RAIDERS": "LV",
            "LV": "LV",
            "GREEN BAY": "GB",
            "PACKERS": "GB",
            "GB": "GB",
            "SAN FRANCISCO": "SF",
            "49ERS": "SF",
            "SF": "SF",
            "NEW YORK GIANTS": "NYG",
            "NYG": "NYG",
            "NEW YORK JETS": "NYJ",
            "NYJ": "NYJ",
        }

        for key, abbrev in abbrev_mapping.items():
            if key in team_upper:
                return abbrev

        # Default: return as-is (will likely fail validation)
        logger.warning("Could not normalize team name", team=team)
        return team_upper[:5]  # Truncate to max 5 chars

    def _create_game_id_from_odds(
        self, game_data: dict[str, Any], season: int, week: int
    ) -> str:
        """Create a game ID from odds data and the MATCHED schedule row's season/week.

        The season and week come from the game's own schedule row, never from the
        request: the request window spans eight days and so two NFL weeks, and stamping
        every payload game with the requesting week mis-keyed the second week's games.
        """
        home_team = self._normalize_team_name(game_data["home_team"])
        away_team = self._normalize_team_name(game_data["away_team"])

        game_id = f"{season}_W{week:02d}_{away_team}@{home_team}"

        # Validate the generated game ID
        if not is_valid_game_id(game_id):
            raise ValueError(f"Generated invalid game ID: {game_id}")

        return game_id

    def _extract_market_odds(
        self,
        bookmaker: dict[str, Any],
        market_key: str,
        *,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        """Extract odds for a specific market, sided by TEAM NAME against the event.

        SIDES COME FROM THE EVENT'S ``home_team`` / ``away_team``, NEVER FROM LIST
        POSITION OR FROM WHICH SIDE IS FAVOURED (Plan 33-18, owner-authorised
        2026-09-15). Two defects lived here:

        * h2h prices were written under team-named keys (``ml_atl``) that
          ``OddsSchema`` drops, so ``ml_home`` / ``ml_away`` were None on every live
          row and WP could never be priced from a pull;
        * the favourite's point was stored as ``spread`` and its price as
          ``spread_ju_home`` whichever side was home, so the stored spread was always
          negative.

        THE STORED CONVENTION IS POSITIVE = HOME FAVOURED, i.e. ``spread`` is the
        NEGATED home line (Atlanta -3.5 at home stores +3.5). Measured on
        ``silver/odds_snapshot.parquet``: 2,130 of 2,140 rows follow it. This makes new
        live rows match that history; whether the ATS code expects it is DEF-31-01 and
        is not decided here.

        The partition fix in ``ingest_odds`` made these values REACHABLE, which is why
        this could not wait: before it a live pull landed where nothing read it, after
        it the same pull would have written blank moneylines and wrong-sign spreads into
        the file training reads.

        An outcome whose name matches NEITHER side is skipped with a warning rather than
        guessed onto one, because a mis-sided price is worse than a missing one.
        """
        market_data: dict[str, Any] = {}
        home_abbrev = self._normalize_team_name(home_team)
        away_abbrev = self._normalize_team_name(away_team)
        if home_abbrev == away_abbrev:
            logger.warning(
                "Cannot side odds outcomes: home and away normalize to the same team",
                home_team=home_team,
                away_team=away_team,
            )
            return market_data

        def side_of(outcome_name: str) -> str | None:
            abbrev = self._normalize_team_name(outcome_name)
            if abbrev == home_abbrev:
                return "home"
            if abbrev == away_abbrev:
                return "away"
            logger.warning(
                "Odds outcome matches neither side; skipped",
                outcome=outcome_name,
                home_team=home_team,
                away_team=away_team,
            )
            return None

        for market in bookmaker.get("markets", []):
            if market["key"] == market_key:
                outcomes = market.get("outcomes", [])

                if market_key == "h2h":
                    # Moneyline odds, one per side.
                    for outcome in outcomes:
                        side = side_of(outcome["name"])
                        if side is not None:
                            market_data[f"ml_{side}"] = outcome.get("price")

                elif market_key == "spreads":
                    # Spread juice per side; the line in the HOME-FAVOURED-POSITIVE
                    # convention. The away point is the negated home point, so the
                    # stored value is -(home point) == (away point). The home outcome
                    # wins when both are present.
                    for outcome in outcomes:
                        side = side_of(outcome["name"])
                        if side is None:
                            continue
                        market_data[f"spread_ju_{side}"] = outcome.get("price", -110)
                        point = outcome.get("point")
                        if point is None:
                            continue
                        if side == "home":
                            market_data["spread"] = -float(point)
                        elif "spread" not in market_data:
                            market_data["spread"] = float(point)

                elif market_key == "totals":
                    # Over/Under odds
                    for outcome in outcomes:
                        if outcome["name"].lower() == "over":
                            market_data["total"] = outcome.get("point")
                            market_data["total_over_ju"] = outcome.get("price", -110)
                        elif outcome["name"].lower() == "under":
                            market_data["total_under_ju"] = outcome.get("price", -110)

                break

        return market_data

    def _match_schedule_row(
        self, game_data: dict[str, Any], schedule: pd.DataFrame
    ) -> pd.Series | None:
        """The payload game's silver ``games`` row, or None when it has none.

        Matched by canonical home and away team. When a slate holds the same pairing
        twice (a window spanning a rematch), the row whose kickoff is nearest the listed
        commence_time wins; with no usable commence_time that case is unmatched rather
        than guessed.
        """
        home = self._normalize_team_name(game_data["home_team"])
        away = self._normalize_team_name(game_data["away_team"])
        candidates = schedule[
            (schedule["home_team"] == home) & (schedule["away_team"] == away)
        ]
        if len(candidates) == 1:
            return candidates.iloc[0]
        if candidates.empty:
            return None
        commence = _parse_commence_time(game_data.get("commence_time"))
        if commence is None:
            return None
        gaps = (pd.to_datetime(candidates["kickoff_et"], utc=True) - commence).abs()
        return candidates.loc[gaps.idxmin()]

    def _process_game_odds(
        self,
        game_data: dict[str, Any],
        *,
        schedule: pd.DataFrame,
        locks: Mapping[str, datetime],
        captured_at: datetime,
    ) -> list[dict[str, Any]]:
        """Process odds for a single game, stamping each row with that game's OWN lock.

        Args:
            game_data: One event from the Odds API response.
            schedule: The slate's silver ``games`` rows.
            locks: ``game_id -> lock`` from ``utils.game_lock.lock_frame(schedule)``.
            captured_at: The instant the response was observed; becomes ``created_at``.

        Returns:
            One record per bookmaker, or ``[]`` when the game matches no schedule row --
            a game with no kickoff has no lock, and guessing one is the failure this
            interface exists to end.
        """
        matched = self._match_schedule_row(game_data, schedule)
        if matched is None:
            self._unmatched.append(
                f"{game_data.get('away_team')} @ {game_data.get('home_team')} "
                f"(commence_time {game_data.get('commence_time')})"
            )
            logger.warning(
                "Odds payload game has no schedule row; skipped rather than stamped "
                "with a guessed lock",
                home_team=game_data.get("home_team"),
                away_team=game_data.get("away_team"),
                commence_time=game_data.get("commence_time"),
            )
            return []

        game_id = self._create_game_id_from_odds(
            game_data, int(matched["season"]), int(matched["week"])
        )
        if game_id != str(matched["game_id"]) or game_id not in locks:
            # The id built from the matched row must BE that row's id, or the lock looked
            # up below would belong to a different key than the row written.
            self._unmatched.append(f"{game_id} (schedule id {matched['game_id']})")
            logger.warning(
                "Odds game id does not equal its matched schedule id; skipped",
                game_id=game_id,
                schedule_game_id=str(matched["game_id"]),
            )
            return []
        lock = locks[game_id]

        commence = _parse_commence_time(game_data.get("commence_time"))
        scheduled = cast(
            "datetime", pd.Timestamp(cast("Any", matched["kickoff_et"])).to_pydatetime()
        )
        if commence is not None and abs(commence - scheduled) > COMMENCE_TIME_TOLERANCE:
            self._kickoff_disagreements.append(game_id)
            logger.warning(
                "Payload commence_time disagrees with the schedule kickoff; possible "
                "schedule move (R8). The SCHEDULE kickoff and its lock are used.",
                game_id=game_id,
                payload_commence_time=commence.isoformat(),
                schedule_kickoff=scheduled.isoformat(),
            )

        odds_records = []

        for bookmaker in game_data.get("bookmakers", []):
            sportsbook = bookmaker.get("key", "unknown")
            raw_update = bookmaker.get("last_update")

            # An absent or unparseable bookmaker time is UNKNOWN and stays NULL. Filling
            # it with our own capture instant (or the lock) would make a manufactured
            # stamp read as the bookmaker's own (RESEARCH P1, T-33.2-02-12). The
            # market-level stamps are kept beside it in market_last_update (Plan 33.2-27).
            last_update = _parse_upstream_instant(raw_update)

            # Extract all market types
            sides = {
                "home_team": game_data["home_team"],
                "away_team": game_data["away_team"],
            }
            h2h_odds = self._extract_market_odds(bookmaker, "h2h", **sides)
            spread_odds = self._extract_market_odds(bookmaker, "spreads", **sides)
            total_odds = self._extract_market_odds(bookmaker, "totals", **sides)

            # Combine all odds data
            odds_record = {
                "game_id": game_id,
                "snapshot_ts": lock,
                "sportsbook": sportsbook,
                "last_update": last_update,
                "market_last_update": latest_market_update(bookmaker),
                "created_at": captured_at,
                "is_live": False,  # Assume pre-game for now
                **h2h_odds,
                **spread_odds,
                **total_odds,
            }

            odds_records.append(odds_record)

        return odds_records

    def transform_odds_data(
        self,
        raw_odds: list[dict[str, Any]],
        *,
        schedule: pd.DataFrame,
        locks: Mapping[str, datetime],
        captured_at: datetime,
    ) -> pd.DataFrame:
        """Transform raw odds data to schema format, one lock per matched game.

        Records what could not be matched, and where payload and schedule disagreed, on
        ``self.last_match_report`` (:class:`LiveOddsMatchReport`).
        """
        logger.info(
            "Transforming odds data",
            input_games=len(raw_odds),
            scheduled_games=len(schedule),
            captured_at=captured_at.isoformat(),
        )

        self._unmatched: list[str] = []
        self._kickoff_disagreements: list[str] = []
        all_odds_records = []

        for game_data in raw_odds:
            try:
                game_odds = self._process_game_odds(
                    game_data, schedule=schedule, locks=locks, captured_at=captured_at
                )
                all_odds_records.extend(game_odds)

            except Exception as e:
                logger.warning(
                    "Failed to process game odds", game_data=game_data, error=str(e)
                )
                continue

        odds_df = pd.DataFrame(all_odds_records)
        self.last_match_report = LiveOddsMatchReport(
            unmatched_games=tuple(self._unmatched),
            kickoff_disagreements=tuple(self._kickoff_disagreements),
        )

        logger.info(
            "Transformed odds data",
            input_games=len(raw_odds),
            output_records=len(odds_df),
            unmatched_games=len(self._unmatched),
            kickoff_disagreements=len(self._kickoff_disagreements),
        )

        return odds_df

    def validate_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Validate odds data against schema."""
        logger.info("Validating odds data", input_rows=len(odds_df))

        valid_records = []
        validation_errors = []

        for idx, row in odds_df.iterrows():
            try:
                # Fill NaN values with None for validation
                row_dict = row.where(pd.notna(row), None).to_dict()

                # Validate against schema
                odds = OddsSchema(**row_dict)
                valid_records.append(odds.model_dump())

            except Exception as e:
                validation_errors.append(f"Row {idx}: {e!s}")
                logger.warning(
                    "Odds data validation failed", row_index=idx, error=str(e)
                )

        if validation_errors:
            logger.warning(
                "Odds data validation issues",
                total_errors=len(validation_errors),
                sample_errors=validation_errors[:5],
            )

        validated_df = pd.DataFrame(valid_records)

        logger.info(
            "Odds data validation completed",
            input_rows=len(odds_df),
            output_rows=len(validated_df),
            errors=len(validation_errors),
        )

        return validated_df

    def ingest_odds(
        self,
        season: int | None = None,
        week: int | None = None,
        *,
        schedule: pd.DataFrame,
        commence_from: datetime | None = None,
        commence_to: datetime | None = None,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Full odds data ingestion pipeline, stamping every row with its own game's lock.

        Args:
            season: Season the slate belongs to; names the bronze file (default: current).
            week: Week the slate belongs to; names the bronze file (default: current).
            schedule: The slate's silver ``games`` rows. THE one input both the
                game-id match and each game's lock are derived from.
            commence_from: Start of the KICKOFF-time selection window. Default: the
                schedule's earliest kickoff minus :data:`COMMENCE_WINDOW_MARGIN`.
            commence_to: End of the KICKOFF-time selection window. Default: the
                schedule's latest kickoff plus :data:`COMMENCE_WINDOW_MARGIN`.
            markets: Markets to fetch
            bookmakers: Bookmakers to include

        Returns:
            Ingested and validated odds data
        """
        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week

        missing = [c for c in SCHEDULE_COLUMNS if c not in schedule.columns]
        if missing or schedule.empty:
            msg = (
                f"ingest_odds needs a non-empty schedule carrying {list(SCHEDULE_COLUMNS)}"
                f" (missing: {missing}, rows: {len(schedule)}); without it no game has a "
                "lock to stamp"
            )
            raise DataIngestionError(msg)

        # The per-game locks, from the ONE rule, over the schedule's own kickoffs.
        locks: dict[str, datetime] = {
            str(game_id): lock.to_pydatetime()
            for game_id, lock in lock_rule.lock_frame(schedule).items()
        }

        # The KICKOFF-time selection window. commenceTimeFrom / commenceTimeTo filter on
        # KICKOFF, so this chooses which games to ask about -- it is not a time fence.
        kickoffs = pd.to_datetime(schedule["kickoff_et"], utc=True)
        if commence_from is None:
            earliest = cast("pd.Timestamp", kickoffs.min())
            commence_from = earliest.to_pydatetime() - COMMENCE_WINDOW_MARGIN
        if commence_to is None:
            latest = cast("pd.Timestamp", kickoffs.max())
            commence_to = latest.to_pydatetime() + COMMENCE_WINDOW_MARGIN

        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        logger.info(
            "Starting odds data ingestion",
            season=season,
            week=week,
            scheduled_games=len(schedule),
            commence_from=commence_from.isoformat(),
            commence_to=commence_to.isoformat(),
            markets=markets,
            bookmakers=bookmakers,
        )

        try:
            # Fetch odds from API
            raw_odds = self.api_client.get_nfl_odds(
                markets=markets,
                bookmakers=bookmakers,
                date_from=commence_from,
                date_to=commence_to,
            )
            # Read ONCE, when the response returns: the one capture instant every row
            # of this response carries as created_at.
            captured_at = _observe_capture_instant()

            # Transform to our schema
            odds_df = self.transform_odds_data(
                raw_odds, schedule=schedule, locks=locks, captured_at=captured_at
            )

            if odds_df.empty:
                logger.warning("No odds data to process")
                return odds_df

            # Validate data. created_at arrives on every row as the observed capture
            # instant; no second clock read happens at the write.
            validated_df = self.validate_odds_data(odds_df)

            # Save to bronze layer (raw)
            save_dataframe(
                pd.DataFrame(raw_odds),
                f"odds_raw_bronze_{season}_W{week:02d}",
                layer="bronze",
                save_to_db=False,
            )

            # Save to silver layer (processed) - skip DuckDB for now due to timezone issues
            #
            # ONE FILE, NEVER PARTITIONED (Plan 33-18, owner-authorised 2026-09-15). This
            # write used to pass partition_cols=["snapshot_ts"] above 100 rows -- which is
            # every real week -- routing it to pq.write_to_dataset in the SHARED silver
            # root. That left silver/odds_snapshot.parquet untouched, so the week's lines
            # reached none of its readers (models/train.py,
            # scripts/generate_current_week_predictions.py, api/routes/health.py), and it
            # rewrote the whole merged history as hash-named files under snapshot_ts=
            # directories: the G-01 cross-table contamination. Same correction CR-01 /
            # D-10 made for gold and 25c364f made for the silver builders.
            #
            # THE STORE ACCUMULATES (Plan 33.2-27 Task 1). save_dataframe's append merge
            # removed every stored row sharing an incoming game_id, so a second pull of a
            # game destroyed the first. append_odds_captures keys each row on the CAPTURE
            # (game_id, sportsbook, snapshot_ts, created_at): earlier captures and the
            # historical rows both survive.
            append_odds_captures(validated_df)

            log_data_operation(
                operation="ingest",
                table="odds_snapshot",
                rows=len(validated_df),
                season=season,
                week=week,
                captured_at=captured_at.isoformat(),
            )

            logger.info(
                "Odds data ingestion completed successfully",
                total_records=len(validated_df),
                unique_games=validated_df["game_id"].nunique(),
                sportsbooks=validated_df["sportsbook"].nunique(),
            )

            return validated_df

        except Exception as e:
            logger.error("Odds data ingestion failed", error=str(e))
            raise DataIngestionError(f"Odds ingestion failed: {e}")

        finally:
            self.close()


def main():
    """CLI entry point for odds data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL odds data")

    # Add standardized ingestion arguments
    from utils.ingestion_args import (
        add_standard_ingestion_args,
        parse_season_week_args,
    )

    parser = add_standard_ingestion_args(parser)

    # Add odds-specific arguments. There is deliberately NO flag for the capture time: the
    # capture instant is observed when the response returns, and a flag letting an
    # operator assert it would manufacture a timestamp (RESEARCH P1, Plan 33.2-02).
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["h2h", "spreads", "totals"],
        default=["h2h", "spreads", "totals"],
        help="Markets to fetch",
    )
    parser.add_argument(
        "--bookmakers", nargs="+", help="Specific bookmakers to include"
    )
    parser.add_argument("--api-key", type=str, help="Odds API key (overrides config)")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock data instead of API"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        # Parse standardized season/week arguments
        seasons, weeks = parse_season_week_args(args)

        # Odds ingestion currently supports single season/week only
        if len(seasons) > 1:
            print(
                "Warning: Odds ingestion only supports single season. Using first season."
            )
        season = seasons[0]

        if weeks and len(weeks) > 1:
            print(
                "Warning: Odds ingestion only supports single week. Using first week."
            )
            week = weeks[0]
        elif weeks:
            week = weeks[0]
        else:
            # Default to current week if no week specified
            _, current_week = get_current_nfl_week()
            week = current_week

        # Log what we're about to ingest
        logger.info(f"Starting odds data ingestion for season {season}, week {week}")

        # The slate's schedule: the one input each game's id and lock come from.
        schedule = load_schedule_slice(season, week)

        # Initialize ingester
        api_key = args.api_key if not args.mock else None
        ingester = OddsDataIngester(api_key)

        if args.mock:
            ingester.api_client.mock_mode = True

        # Run ingestion
        odds_df = ingester.ingest_odds(
            season=season,
            week=week,
            schedule=schedule,
            markets=args.markets,
            bookmakers=args.bookmakers,
        )

        print(f"Successfully ingested {len(odds_df)} odds records")
        print(f"Season: {season}, Week: {week}")
        print(f"Unique games: {odds_df['game_id'].nunique()}")
        print(f"Sportsbooks: {', '.join(odds_df['sportsbook'].unique())}")

        # Show sample data
        if not odds_df.empty:
            print("\nSample odds data:")
            sample_cols = [
                "game_id",
                "sportsbook",
                "ml_home",
                "ml_away",
                "spread",
                "total",
            ]
            available_cols = [col for col in sample_cols if col in odds_df.columns]
            print(odds_df[available_cols].head())

    except Exception as e:
        logger.error("Odds ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
