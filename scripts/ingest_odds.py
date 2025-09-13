"""Odds data ingestion from external APIs."""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from conf.settings import get_settings
from data.storage import save_dataframe, get_db_connection
from data.schemas import OddsSchema
from utils import (
    get_logger, 
    get_current_nfl_week,
    get_snapshot_time,
    DataIngestionError,
    ExternalAPIError,
    log_data_operation
)


logger = get_logger(__name__)


class OddsAPIClient:
    """Client for fetching odds from external APIs."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize odds API client.
        
        Args:
            api_key: API key for odds service
            base_url: Base URL for odds API
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.odds_api_key
        self.base_url = base_url or self.settings.config.external_apis.odds_api['base_url']
        
        if not self.api_key:
            logger.warning("No odds API key provided - using mock data")
            self.mock_mode = True
        else:
            self.mock_mode = False
        
        # API configuration
        self.timeout = self.settings.config.external_apis.odds_api['timeout']
        self.retries = self.settings.config.external_apis.odds_api['retries']
        self.rate_limit = self.settings.config.external_apis.odds_api['rate_limit_per_hour']
        
        # HTTP client
        self.client = httpx.Client(timeout=self.timeout)
    
    def close(self):
        """Close HTTP client."""
        self.client.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to odds API with retries."""
        if self.mock_mode:
            return self._generate_mock_odds()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to params
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        try:
            logger.debug("Making odds API request", url=url, params=params)
            
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Odds API request successful", 
                       url=url, status_code=response.status_code)
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error("Odds API HTTP error", 
                        url=url, status_code=e.response.status_code, 
                        response=e.response.text)
            raise ExternalAPIError(f"HTTP error {e.response.status_code}: {e.response.text}")
            
        except httpx.RequestError as e:
            logger.error("Odds API request error", url=url, error=str(e))
            raise ExternalAPIError(f"Request error: {e}")
            
        except Exception as e:
            logger.error("Odds API unexpected error", url=url, error=str(e))
            raise ExternalAPIError(f"Unexpected error: {e}")
    
    def _generate_mock_odds(self) -> Dict[str, Any]:
        """Generate mock odds data for testing."""
        current_season, current_week = get_current_nfl_week()
        
        # Generate mock games for current week
        mock_games = [
            {
                "id": f"mock_game_{i}",
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": (datetime.now() + timedelta(days=i)).isoformat(),
                "home_team": ["BUF", "KC", "DAL", "SF", "GB"][i % 5],
                "away_team": ["MIA", "LV", "PHI", "LAR", "MIN"][i % 5],
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": datetime.now().isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {
                                        "name": ["BUF", "KC", "DAL", "SF", "GB"][i % 5],
                                        "price": -110 + (i * 10)
                                    },
                                    {
                                        "name": ["MIA", "LV", "PHI", "LAR", "MIN"][i % 5], 
                                        "price": -110 - (i * 10)
                                    }
                                ]
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": ["BUF", "KC", "DAL", "SF", "GB"][i % 5],
                                        "price": -110,
                                        "point": -3.5 + i
                                    },
                                    {
                                        "name": ["MIA", "LV", "PHI", "LAR", "MIN"][i % 5],
                                        "price": -110,
                                        "point": 3.5 - i
                                    }
                                ]
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "price": -110,
                                        "point": 45.5 + i
                                    },
                                    {
                                        "name": "Under", 
                                        "price": -110,
                                        "point": 45.5 + i
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            for i in range(3)  # Generate 3 mock games
        ]
        
        logger.info("Generated mock odds data", games=len(mock_games))
        return mock_games
    
    def get_nfl_odds(
        self,
        markets: List[str] = None,
        bookmakers: List[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
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
            markets = ['h2h', 'spreads', 'totals']
        
        params = {
            'sport': 'americanfootball_nfl',
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        if date_from:
            params['commenceTimeFrom'] = date_from.isoformat()
        
        if date_to:
            params['commenceTimeTo'] = date_to.isoformat()
        
        logger.info("Fetching NFL odds", markets=markets, bookmakers=bookmakers)
        
        # Use different endpoint based on API structure
        endpoint = "sports/americanfootball_nfl/odds"
        
        data = self._make_request(endpoint, params)
        
        # Handle different response formats
        if isinstance(data, list):
            games = data
        elif isinstance(data, dict) and 'data' in data:
            games = data['data']
        else:
            games = []
        
        logger.info("Fetched NFL odds", total_games=len(games))
        return games


class OddsDataIngester:
    """NFL odds data ingestion and processing."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize odds data ingester."""
        self.settings = get_settings()
        self.db = get_db_connection()
        self.api_client = OddsAPIClient(api_key)
        
        # Sportsbook priority for consensus odds
        self.sportsbook_priority = [
            'draftkings', 'fanduel', 'betmgm', 'caesars',
            'pointsbet', 'barstool', 'unibet'
        ]
    
    def close(self):
        """Close API client."""
        self.api_client.close()
    
    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to canonical abbreviation."""
        # Team name mapping for odds APIs
        team_mapping = {
            # Common variations in odds APIs
            'Kansas City Chiefs': 'KC',
            'Buffalo Bills': 'BUF',
            'Miami Dolphins': 'MIA',
            'New England Patriots': 'NE',
            'Baltimore Ravens': 'BAL',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Pittsburgh Steelers': 'PIT',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Tennessee Titans': 'TEN',
            'Denver Broncos': 'DEN',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Chicago Bears': 'CHI',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Minnesota Vikings': 'MIN',
            'Atlanta Falcons': 'ATL',
            'Carolina Panthers': 'CAR',
            'New Orleans Saints': 'NO',
            'Tampa Bay Buccaneers': 'TB',
            'Dallas Cowboys': 'DAL',
            'New York Giants': 'NYG',
            'Philadelphia Eagles': 'PHI',
            'Washington Commanders': 'WAS',
            'Arizona Cardinals': 'ARI',
            'Los Angeles Rams': 'LAR',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'New York Jets': 'NYJ'
        }
        
        # Try direct mapping first
        if team in team_mapping:
            return team_mapping[team]
        
        # Try to extract abbreviation
        team_upper = team.upper().strip()
        
        # Common abbreviation patterns
        abbrev_mapping = {
            'KANSAS CITY': 'KC', 'KC': 'KC',
            'BUFFALO': 'BUF', 'BUF': 'BUF',
            'NEW ENGLAND': 'NE', 'NE': 'NE',
            'MIAMI': 'MIA', 'MIA': 'MIA',
            'BALTIMORE': 'BAL', 'BAL': 'BAL',
            'CINCINNATI': 'CIN', 'CIN': 'CIN',
            'CLEVELAND': 'CLE', 'CLE': 'CLE',
            'PITTSBURGH': 'PIT', 'PIT': 'PIT',
            'LAS VEGAS': 'LV', 'RAIDERS': 'LV', 'LV': 'LV',
            'GREEN BAY': 'GB', 'PACKERS': 'GB', 'GB': 'GB',
            'SAN FRANCISCO': 'SF', '49ERS': 'SF', 'SF': 'SF',
            'NEW YORK GIANTS': 'NYG', 'NYG': 'NYG',
            'NEW YORK JETS': 'NYJ', 'NYJ': 'NYJ'
        }
        
        for key, abbrev in abbrev_mapping.items():
            if key in team_upper:
                return abbrev
        
        # Default: return as-is (will likely fail validation)
        logger.warning("Could not normalize team name", team=team)
        return team_upper[:5]  # Truncate to max 5 chars
    
    def _create_game_id_from_odds(self, game_data: Dict[str, Any], season: int, week: int) -> str:
        """Create game ID from odds data."""
        home_team = self._normalize_team_name(game_data['home_team'])
        away_team = self._normalize_team_name(game_data['away_team'])
        
        return f"{season}_W{week:02d}_{away_team}@{home_team}"
    
    def _extract_market_odds(
        self, 
        bookmaker: Dict[str, Any], 
        market_key: str
    ) -> Dict[str, Any]:
        """Extract odds for a specific market from bookmaker data."""
        market_data = {}
        
        for market in bookmaker.get('markets', []):
            if market['key'] == market_key:
                outcomes = market.get('outcomes', [])
                
                if market_key == 'h2h':
                    # Moneyline odds
                    for outcome in outcomes:
                        team = self._normalize_team_name(outcome['name'])
                        market_data[f'ml_{team.lower()}'] = outcome.get('price')
                
                elif market_key == 'spreads':
                    # Spread odds
                    for outcome in outcomes:
                        team = self._normalize_team_name(outcome['name'])
                        point = outcome.get('point', 0)
                        price = outcome.get('price', -110)
                        
                        if point < 0:  # This team is favored
                            market_data['spread'] = point
                            market_data['spread_ju_home'] = price
                        else:  # This team is underdog
                            market_data['spread_ju_away'] = price
                
                elif market_key == 'totals':
                    # Over/Under odds
                    for outcome in outcomes:
                        if outcome['name'].lower() == 'over':
                            market_data['total'] = outcome.get('point')
                            market_data['total_over_ju'] = outcome.get('price', -110)
                        elif outcome['name'].lower() == 'under':
                            market_data['total_under_ju'] = outcome.get('price', -110)
                
                break
        
        return market_data
    
    def _process_game_odds(
        self, 
        game_data: Dict[str, Any], 
        snapshot_time: datetime,
        season: int,
        week: int
    ) -> List[Dict[str, Any]]:
        """Process odds for a single game."""
        game_id = self._create_game_id_from_odds(game_data, season, week)
        
        odds_records = []
        
        for bookmaker in game_data.get('bookmakers', []):
            sportsbook = bookmaker.get('key', 'unknown')
            last_update = bookmaker.get('last_update')
            
            if last_update:
                try:
                    last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                except:
                    last_update = snapshot_time
            else:
                last_update = snapshot_time
            
            # Extract all market types
            h2h_odds = self._extract_market_odds(bookmaker, 'h2h')
            spread_odds = self._extract_market_odds(bookmaker, 'spreads')
            total_odds = self._extract_market_odds(bookmaker, 'totals')
            
            # Combine all odds data
            odds_record = {
                'game_id': game_id,
                'snapshot_ts': snapshot_time,
                'sportsbook': sportsbook,
                'last_update': last_update,
                'is_live': False,  # Assume pre-game for now
                **h2h_odds,
                **spread_odds,
                **total_odds
            }
            
            odds_records.append(odds_record)
        
        return odds_records
    
    def transform_odds_data(
        self, 
        raw_odds: List[Dict[str, Any]], 
        snapshot_time: datetime,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """Transform raw odds data to schema format."""
        logger.info("Transforming odds data", 
                   input_games=len(raw_odds), season=season, week=week)
        
        all_odds_records = []
        
        for game_data in raw_odds:
            try:
                game_odds = self._process_game_odds(game_data, snapshot_time, season, week)
                all_odds_records.extend(game_odds)
                
            except Exception as e:
                logger.warning("Failed to process game odds", 
                              game_data=game_data, error=str(e))
                continue
        
        odds_df = pd.DataFrame(all_odds_records)
        
        logger.info("Transformed odds data", 
                   input_games=len(raw_odds),
                   output_records=len(odds_df))
        
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
                valid_records.append(odds.dict())
                
            except Exception as e:
                validation_errors.append(f"Row {idx}: {str(e)}")
                logger.warning("Odds data validation failed", 
                              row_index=idx, error=str(e))
        
        if validation_errors:
            logger.warning("Odds data validation issues",
                          total_errors=len(validation_errors),
                          sample_errors=validation_errors[:5])
        
        validated_df = pd.DataFrame(valid_records)
        
        logger.info("Odds data validation completed",
                   input_rows=len(odds_df),
                   output_rows=len(validated_df),
                   errors=len(validation_errors))
        
        return validated_df
    
    def ingest_odds(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        snapshot_time: Optional[datetime] = None,
        markets: List[str] = None,
        bookmakers: List[str] = None
    ) -> pd.DataFrame:
        """
        Full odds data ingestion pipeline.
        
        Args:
            season: Season to ingest (default: current)
            week: Week to ingest (default: current)
            snapshot_time: Time of odds snapshot (default: Friday 6PM ET)
            markets: Markets to fetch
            bookmakers: Bookmakers to include
            
        Returns:
            Ingested and validated odds data
        """
        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week
        
        if snapshot_time is None:
            snapshot_time = get_snapshot_time()
        
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']
        
        logger.info("Starting odds data ingestion",
                   season=season, week=week, 
                   snapshot_time=snapshot_time.isoformat(),
                   markets=markets, bookmakers=bookmakers)
        
        try:
            # Fetch odds from API
            raw_odds = self.api_client.get_nfl_odds(
                markets=markets,
                bookmakers=bookmakers,
                date_from=snapshot_time - timedelta(days=1),
                date_to=snapshot_time + timedelta(days=7)
            )
            
            # Transform to our schema
            odds_df = self.transform_odds_data(raw_odds, snapshot_time, season, week)
            
            if odds_df.empty:
                logger.warning("No odds data to process")
                return odds_df
            
            # Validate data
            validated_df = self.validate_odds_data(odds_df)
            
            # Save to bronze layer (raw)
            save_dataframe(
                pd.DataFrame(raw_odds),
                f'odds_raw_bronze_{season}_W{week:02d}',
                layer='bronze',
                save_to_db=False
            )
            
            # Save to silver layer (processed)
            save_dataframe(
                validated_df,
                'odds_snapshot',
                layer='silver',
                partition_cols=['snapshot_ts'] if len(validated_df) > 100 else None
            )
            
            log_data_operation(
                operation='ingest',
                table='odds_snapshot',
                rows=len(validated_df),
                season=season,
                week=week,
                snapshot_time=snapshot_time.isoformat()
            )
            
            logger.info("Odds data ingestion completed successfully",
                       total_records=len(validated_df),
                       unique_games=validated_df['game_id'].nunique(),
                       sportsbooks=validated_df['sportsbook'].nunique())
            
            return validated_df
            
        except Exception as e:
            logger.error("Odds data ingestion failed", error=str(e))
            raise DataIngestionError(f"Odds ingestion failed: {e}")
        
        finally:
            self.close()


def main():
    """CLI entry point for odds data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL odds data")
    parser.add_argument('--season', type=int, help='Season to ingest (default: current)')
    parser.add_argument('--week', type=int, help='Specific week to ingest (default: current)')
    parser.add_argument('--current', action='store_true', help='Ingest current week')
    parser.add_argument('--snapshot-time', type=str, 
                       help='Snapshot time (ISO format, default: Friday 6PM ET)')
    parser.add_argument('--markets', nargs='+', choices=['h2h', 'spreads', 'totals'],
                       default=['h2h', 'spreads', 'totals'], help='Markets to fetch')
    parser.add_argument('--bookmakers', nargs='+', help='Specific bookmakers to include')
    parser.add_argument('--api-key', type=str, help='Odds API key (overrides config)')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of API')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        from utils import setup_logging
        setup_logging()
        
        # Determine season and week
        if args.current or (not args.season and not args.week):
            current_season, current_week = get_current_nfl_week()
            season = args.season or current_season
            week = args.week or current_week
        else:
            season = args.season
            week = args.week
        
        # Parse snapshot time
        snapshot_time = None
        if args.snapshot_time:
            try:
                snapshot_time = datetime.fromisoformat(args.snapshot_time)
            except ValueError:
                print(f"Invalid snapshot time format: {args.snapshot_time}")
                sys.exit(1)
        
        # Initialize ingester
        api_key = args.api_key if not args.mock else None
        ingester = OddsDataIngester(api_key)
        
        if args.mock:
            ingester.api_client.mock_mode = True
        
        # Run ingestion
        odds_df = ingester.ingest_odds(
            season=season,
            week=week,
            snapshot_time=snapshot_time,
            markets=args.markets,
            bookmakers=args.bookmakers
        )
        
        print(f"Successfully ingested {len(odds_df)} odds records")
        print(f"Season: {season}, Week: {week}")
        print(f"Unique games: {odds_df['game_id'].nunique()}")
        print(f"Sportsbooks: {', '.join(odds_df['sportsbook'].unique())}")
        
        # Show sample data
        if not odds_df.empty:
            print("\nSample odds data:")
            sample_cols = ['game_id', 'sportsbook', 'ml_home', 'ml_away', 'spread', 'total']
            available_cols = [col for col in sample_cols if col in odds_df.columns]
            print(odds_df[available_cols].head())
        
    except Exception as e:
        logger.error("Odds ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()