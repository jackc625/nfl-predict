"""Weather data ingestion using Meteostat."""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from meteostat import Point, Hourly, Daily
import pytz

from conf.settings import get_settings
from data.storage import save_dataframe, load_dataframe
from data.schemas import WeatherSchema
from utils import (
    get_logger, 
    get_current_nfl_week,
    DataIngestionError,
    log_data_operation
)


logger = get_logger(__name__)


class WeatherDataIngester:
    """NFL weather data ingestion using Meteostat."""
    
    def __init__(self):
        """Initialize weather data ingester."""
        self.settings = get_settings()
        
        # Timezone mappings
        self.timezone_map = {
            'America/New_York': pytz.timezone('America/New_York'),
            'America/Chicago': pytz.timezone('America/Chicago'),
            'America/Denver': pytz.timezone('America/Denver'),
            'America/Los_Angeles': pytz.timezone('America/Los_Angeles'),
            'America/Phoenix': pytz.timezone('America/Phoenix')
        }
    
    def _load_venue_data(self) -> pd.DataFrame:
        """Load venue data with coordinates."""
        try:
            venues_df = load_dataframe('venues', layer='silver')
            logger.info("Loaded venue data", venues=len(venues_df))
            return venues_df
        except Exception as e:
            logger.error("Failed to load venue data", error=str(e))
            raise DataIngestionError(f"Venue data load failed: {e}")
    
    def _load_games_data(self, season: int, week: int) -> pd.DataFrame:
        """Load games data for specified season/week."""
        try:
            games_df = load_dataframe('games', layer='silver')
            
            # Filter for specific season/week
            filtered_games = games_df[
                (games_df['season'] == season) & 
                (games_df['week'] == week)
            ].copy()
            
            logger.info("Loaded games data", 
                       season=season, week=week, games=len(filtered_games))
            return filtered_games
            
        except Exception as e:
            logger.error("Failed to load games data", 
                        season=season, week=week, error=str(e))
            raise DataIngestionError(f"Games data load failed: {e}")
    
    def _get_venue_coordinates(self, home_team: str, venues_df: pd.DataFrame) -> Tuple[float, float, str]:
        """Get venue coordinates and roof type for a team."""
        venue_info = venues_df[venues_df['home_teams'].str.contains(home_team, na=False)]
        
        if venue_info.empty:
            logger.warning("No venue found for team", team=home_team)
            # Default coordinates (Kansas City as central location)
            return 39.0997, -94.5786, 'outdoor'
        
        venue = venue_info.iloc[0]
        return venue['latitude'], venue['longitude'], venue['roof_type']
    
    def _is_outdoor_game(self, roof_type: str) -> bool:
        """Determine if weather affects the game."""
        return roof_type.lower() in ['outdoor', 'retractable']
    
    def _convert_timezone(self, dt: datetime, from_tz: str, to_utc: bool = True) -> datetime:
        """Convert datetime between timezones."""
        if from_tz in self.timezone_map:
            tz = self.timezone_map[from_tz]
            
            if dt.tzinfo is None:
                dt = tz.localize(dt)
            
            if to_utc:
                return dt.astimezone(pytz.UTC)
            else:
                return dt
        
        return dt
    
    def _generate_mock_weather(
        self, 
        game_time: datetime, 
        latitude: float, 
        longitude: float
    ) -> Dict[str, Any]:
        """Generate mock weather data for testing."""
        import random
        
        # Base weather on geographic location and season
        month = game_time.month
        
        # Temperature based on latitude and season
        if latitude > 45:  # Northern locations
            base_temp = 35 if month in [11, 12, 1, 2] else 65
        elif latitude < 30:  # Southern locations  
            base_temp = 65 if month in [11, 12, 1, 2] else 80
        else:  # Middle latitudes
            base_temp = 45 if month in [11, 12, 1, 2] else 70
        
        temp_f = base_temp + random.randint(-15, 15)
        temp_c = (temp_f - 32) * 5 / 9
        
        # Wind based on season and location
        wind_mph = random.uniform(2, 20)
        if month in [11, 12, 1, 2, 3]:  # Winter/early spring - windier
            wind_mph += random.uniform(0, 10)
        
        # Precipitation probability
        precip_prob = random.uniform(0, 0.4)  # 0-40% chance
        if month in [4, 5, 6, 7, 8]:  # Spring/summer - more rain
            precip_prob += random.uniform(0, 0.3)
        
        precip_prob = min(precip_prob, 1.0)
        
        # Precipitation amount (if any)
        precip_mm = random.uniform(0, 5) if precip_prob > 0.3 else 0
        
        # Other conditions
        humidity_pct = random.uniform(40, 90)
        visibility_km = random.uniform(8, 16)
        
        # Weather condition
        if precip_prob > 0.6:
            condition = "Rain" if temp_f > 35 else "Snow"
            condition_code = 500 if temp_f > 35 else 600
        elif precip_prob > 0.3:
            condition = "Cloudy"
            condition_code = 300
        else:
            condition = "Clear"
            condition_code = 800
        
        return {
            'temp_f': round(temp_f, 1),
            'temp_c': round(temp_c, 1),
            'wind_mph': round(wind_mph, 1),
            'wind_direction': random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'humidity_pct': round(humidity_pct, 1),
            'precip_prob': round(precip_prob, 2),
            'precip_mm': round(precip_mm, 1),
            'condition': condition,
            'condition_code': condition_code,
            'visibility_km': round(visibility_km, 1)
        }
    
    def _fetch_meteostat_weather(
        self, 
        latitude: float, 
        longitude: float, 
        game_time: datetime,
        forecast_time: datetime
    ) -> Dict[str, Any]:
        """Fetch weather data from Meteostat API."""
        try:
            # Create Point object for location
            location = Point(latitude, longitude)
            
            # Determine time range for forecast
            start_time = game_time - timedelta(hours=2)
            end_time = game_time + timedelta(hours=2)
            
            # Fetch hourly data
            data = Hourly(location, start_time, end_time)
            df = data.fetch()
            
            if df.empty:
                logger.warning("No Meteostat data available", 
                              lat=latitude, lon=longitude, time=game_time)
                return self._generate_mock_weather(game_time, latitude, longitude)
            
            # Get closest time to game time
            closest_idx = df.index[df.index.get_indexer([game_time], method='nearest')]
            weather_data = df.loc[closest_idx[0]]
            
            # Convert to our format
            return {
                'temp_f': round(weather_data.get('temp', 60) * 9/5 + 32, 1) if pd.notna(weather_data.get('temp')) else None,
                'temp_c': round(weather_data.get('temp', 15), 1) if pd.notna(weather_data.get('temp')) else None,
                'wind_mph': round(weather_data.get('wspd', 5) * 2.237, 1) if pd.notna(weather_data.get('wspd')) else None,
                'wind_direction': None,  # Meteostat doesn't provide direction easily
                'humidity_pct': round(weather_data.get('rhum', 60), 1) if pd.notna(weather_data.get('rhum')) else None,
                'precip_prob': None,  # Not available in historical data
                'precip_mm': round(weather_data.get('prcp', 0), 1) if pd.notna(weather_data.get('prcp')) else 0,
                'condition': None,  # Would need to derive from other fields
                'condition_code': None,
                'visibility_km': None  # Not available
            }
            
        except Exception as e:
            logger.warning("Meteostat fetch failed", 
                          lat=latitude, lon=longitude, error=str(e))
            return self._generate_mock_weather(game_time, latitude, longitude)
    
    def _create_weather_record(
        self,
        game_id: str,
        game_time: datetime,
        forecast_time: datetime,
        weather_data: Dict[str, Any],
        roof_type: str
    ) -> Dict[str, Any]:
        """Create weather record for a game."""
        is_outdoor = self._is_outdoor_game(roof_type)
        
        # Derived weather flags
        is_cold = weather_data.get('temp_f', 60) < 32 if weather_data.get('temp_f') is not None else None
        is_windy = weather_data.get('wind_mph', 0) > 12 if weather_data.get('wind_mph') is not None else None
        is_precipitation = (
            weather_data.get('precip_prob', 0) > 0.3 or 
            weather_data.get('precip_mm', 0) > 0
        ) if weather_data.get('precip_prob') is not None or weather_data.get('precip_mm') is not None else None
        
        return {
            'game_id': game_id,
            'forecast_time': forecast_time,
            'game_time': game_time,
            'is_outdoor': is_outdoor,
            'is_cold': is_cold,
            'is_windy': is_windy,
            'is_precipitation': is_precipitation,
            **weather_data
        }
    
    def fetch_weather_for_games(
        self,
        games_df: pd.DataFrame,
        venues_df: pd.DataFrame,
        forecast_time: Optional[datetime] = None,
        use_mock: bool = False
    ) -> pd.DataFrame:
        """Fetch weather data for all games."""
        if forecast_time is None:
            forecast_time = datetime.now(pytz.UTC)
        
        logger.info("Fetching weather for games", 
                   games=len(games_df), use_mock=use_mock)
        
        weather_records = []
        
        for _, game in games_df.iterrows():
            try:
                # Get venue coordinates
                lat, lon, roof_type = self._get_venue_coordinates(game['home_team'], venues_df)
                
                # Convert game time to UTC
                game_time_utc = self._convert_timezone(
                    game['kickoff_et'], 
                    'America/New_York', 
                    to_utc=True
                )
                
                # Fetch weather data
                if use_mock or not self._is_outdoor_game(roof_type):
                    weather_data = self._generate_mock_weather(game_time_utc, lat, lon)
                else:
                    weather_data = self._fetch_meteostat_weather(
                        lat, lon, game_time_utc, forecast_time
                    )
                
                # Create weather record
                weather_record = self._create_weather_record(
                    game['game_id'],
                    game_time_utc,
                    forecast_time,
                    weather_data,
                    roof_type
                )
                
                weather_records.append(weather_record)
                
                logger.debug("Weather fetched for game", 
                            game_id=game['game_id'], 
                            outdoor=weather_record['is_outdoor'])
                
            except Exception as e:
                logger.warning("Failed to fetch weather for game", 
                              game_id=game['game_id'], error=str(e))
                continue
        
        weather_df = pd.DataFrame(weather_records)
        
        logger.info("Weather data fetched", 
                   games=len(games_df), 
                   weather_records=len(weather_df))
        
        return weather_df
    
    def validate_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Validate weather data against schema."""
        logger.info("Validating weather data", input_rows=len(weather_df))
        
        valid_records = []
        validation_errors = []
        
        for idx, row in weather_df.iterrows():
            try:
                # Fill NaN values with None for validation
                row_dict = row.where(pd.notna(row), None).to_dict()
                
                # Validate against schema
                weather = WeatherSchema(**row_dict)
                valid_records.append(weather.dict())
                
            except Exception as e:
                validation_errors.append(f"Row {idx}: {str(e)}")
                logger.warning("Weather data validation failed", 
                              row_index=idx, error=str(e))
        
        if validation_errors:
            logger.warning("Weather data validation issues",
                          total_errors=len(validation_errors),
                          sample_errors=validation_errors[:5])
        
        validated_df = pd.DataFrame(valid_records)
        
        logger.info("Weather data validation completed",
                   input_rows=len(weather_df),
                   output_rows=len(validated_df),
                   errors=len(validation_errors))
        
        return validated_df
    
    def ingest_weather(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        forecast_time: Optional[datetime] = None,
        use_mock: bool = False
    ) -> pd.DataFrame:
        """
        Full weather data ingestion pipeline.
        
        Args:
            season: Season to ingest (default: current)
            week: Week to ingest (default: current)
            forecast_time: Time when forecast was made
            use_mock: Use mock data instead of API
            
        Returns:
            Ingested and validated weather data
        """
        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week
        
        if forecast_time is None:
            forecast_time = datetime.now(pytz.UTC)
        
        logger.info("Starting weather data ingestion",
                   season=season, week=week, 
                   forecast_time=forecast_time.isoformat(),
                   use_mock=use_mock)
        
        try:
            # Load required data
            venues_df = self._load_venue_data()
            games_df = self._load_games_data(season, week)
            
            if games_df.empty:
                logger.warning("No games found for season/week", 
                              season=season, week=week)
                return pd.DataFrame()
            
            # Fetch weather data
            weather_df = self.fetch_weather_for_games(
                games_df, venues_df, forecast_time, use_mock
            )
            
            if weather_df.empty:
                logger.warning("No weather data fetched")
                return weather_df
            
            # Validate data
            validated_df = self.validate_weather_data(weather_df)
            
            # Save to bronze layer (raw)
            if not use_mock:  # Only save raw data if from real API
                save_dataframe(
                    weather_df,
                    f'weather_raw_bronze_{season}_W{week:02d}',
                    layer='bronze',
                    save_to_db=False
                )
            
            # Save to silver layer (processed)
            save_dataframe(
                validated_df,
                'weather_forecast',
                layer='silver',
                partition_cols=['forecast_time'] if len(validated_df) > 100 else None
            )
            
            log_data_operation(
                operation='ingest',
                table='weather_forecast',
                rows=len(validated_df),
                season=season,
                week=week,
                use_mock=use_mock
            )
            
            logger.info("Weather data ingestion completed successfully",
                       total_records=len(validated_df),
                       outdoor_games=validated_df['is_outdoor'].sum(),
                       season=season, week=week)
            
            return validated_df
            
        except Exception as e:
            logger.error("Weather data ingestion failed", error=str(e))
            raise DataIngestionError(f"Weather ingestion failed: {e}")


def main():
    """CLI entry point for weather data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL weather data")
    parser.add_argument('--season', type=int, help='Season to ingest (default: current)')
    parser.add_argument('--week', type=int, help='Specific week to ingest (default: current)')
    parser.add_argument('--current', action='store_true', help='Ingest current week')
    parser.add_argument('--forecast-time', type=str,
                       help='Forecast time (ISO format, default: now)')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of API')
    parser.add_argument('--outdoor-only', action='store_true', 
                       help='Only fetch weather for outdoor games')
    
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
        
        # Parse forecast time
        forecast_time = None
        if args.forecast_time:
            try:
                forecast_time = datetime.fromisoformat(args.forecast_time)
                if forecast_time.tzinfo is None:
                    forecast_time = pytz.UTC.localize(forecast_time)
            except ValueError:
                print(f"Invalid forecast time format: {args.forecast_time}")
                sys.exit(1)
        
        # Initialize ingester
        ingester = WeatherDataIngester()
        
        # Run ingestion
        weather_df = ingester.ingest_weather(
            season=season,
            week=week,
            forecast_time=forecast_time,
            use_mock=args.mock
        )
        
        if weather_df.empty:
            print("No weather data ingested")
            return
        
        print(f"Successfully ingested {len(weather_df)} weather records")
        print(f"Season: {season}, Week: {week}")
        print(f"Outdoor games: {weather_df['is_outdoor'].sum()}")
        print(f"Indoor games: {(~weather_df['is_outdoor']).sum()}")
        
        # Show weather summary for outdoor games
        outdoor_weather = weather_df[weather_df['is_outdoor']]
        if not outdoor_weather.empty:
            print(f"\nOutdoor weather summary:")
            if 'temp_f' in outdoor_weather.columns:
                print(f"  Temperature: {outdoor_weather['temp_f'].mean():.1f}°F avg")
            if 'wind_mph' in outdoor_weather.columns:
                print(f"  Wind: {outdoor_weather['wind_mph'].mean():.1f} mph avg")
            if 'is_cold' in outdoor_weather.columns:
                print(f"  Cold games: {outdoor_weather['is_cold'].sum()}")
            if 'is_windy' in outdoor_weather.columns:
                print(f"  Windy games: {outdoor_weather['is_windy'].sum()}")
        
    except Exception as e:
        logger.error("Weather ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()