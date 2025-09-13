"""Load venue/stadium data from static JSON file."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from conf.settings import get_settings
from data.storage import save_dataframe
from data.schemas import VenueSchema
from utils import get_logger, DataIngestionError, log_data_operation


logger = get_logger(__name__)


class VenueDataLoader:
    """Load and process venue/stadium data."""
    
    def __init__(self):
        """Initialize venue data loader."""
        self.settings = get_settings()
        self.venues_file = Path("data/venues.json")
        
    def load_venues_json(self) -> Dict[str, Any]:
        """Load venues from JSON file."""
        if not self.venues_file.exists():
            raise DataIngestionError(f"Venues file not found: {self.venues_file}")
        
        try:
            with open(self.venues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info("Loaded venues JSON", file=str(self.venues_file))
            return data
            
        except Exception as e:
            logger.error("Failed to load venues JSON", 
                        file=str(self.venues_file), error=str(e))
            raise DataIngestionError(f"Venues JSON load failed: {e}")
    
    def validate_venues(self, venues_data: List[Dict[str, Any]]) -> List[VenueSchema]:
        """Validate venue data against schema."""
        logger.info("Validating venue data", count=len(venues_data))
        
        validated_venues = []
        validation_errors = []
        
        for i, venue_data in enumerate(venues_data):
            try:
                venue = VenueSchema(**venue_data)
                validated_venues.append(venue)
                
            except Exception as e:
                validation_errors.append(f"Venue {i}: {str(e)}")
                logger.warning("Venue validation failed", 
                              venue_index=i, error=str(e))
        
        if validation_errors:
            logger.warning("Venue validation issues",
                          total_errors=len(validation_errors),
                          sample_errors=validation_errors[:3])
        
        logger.info("Venue validation completed",
                   input_count=len(venues_data),
                   validated_count=len(validated_venues),
                   errors=len(validation_errors))
        
        return validated_venues
    
    def create_venue_mappings(self, venues: List[VenueSchema]) -> Dict[str, Dict[str, Any]]:
        """Create venue mapping dictionaries for easy lookup."""
        mappings = {
            'by_name': {},
            'by_team': {},
            'by_id': {}
        }
        
        for venue in venues:
            venue_dict = venue.dict()
            
            # Map by venue name (normalized)
            name_key = venue.venue_name.lower().replace(' ', '_').replace('-', '_')
            mappings['by_name'][name_key] = venue_dict
            
            # Map by team
            for team in venue.home_teams:
                mappings['by_team'][team] = venue_dict
            
            # Map by ID
            mappings['by_id'][venue.venue_id] = venue_dict
        
        return mappings
    
    def process_venues(self) -> pd.DataFrame:
        """Complete venue data processing pipeline."""
        logger.info("Starting venue data processing")
        
        try:
            # Load JSON data
            venues_json = self.load_venues_json()
            venues_data = venues_json.get('venues', [])
            
            # Validate venues
            validated_venues = self.validate_venues(venues_data)
            
            # Convert to DataFrame
            venues_df = pd.DataFrame([venue.dict() for venue in validated_venues])
            
            # Create venue mappings
            mappings = self.create_venue_mappings(validated_venues)
            
            # Save venues data
            save_dataframe(
                venues_df,
                'venues',
                layer='silver',
                save_to_db=True,
                save_to_parquet=True
            )
            
            # Save mappings as JSON for easy lookup
            mappings_path = self.settings.get_data_path("silver") / "venue_mappings.json"
            with open(mappings_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, default=str)
            
            log_data_operation(
                operation='load',
                table='venues',
                rows=len(venues_df),
                source='static_json'
            )
            
            logger.info("Venue data processing completed",
                       total_venues=len(venues_df))
            
            return venues_df
            
        except Exception as e:
            logger.error("Venue data processing failed", error=str(e))
            raise DataIngestionError(f"Venue processing failed: {e}")
    
    def get_venue_by_team(self, team: str) -> Dict[str, Any]:
        """Get venue information for a team."""
        mappings_path = self.settings.get_data_path("silver") / "venue_mappings.json"
        
        if not mappings_path.exists():
            # Process venues if mappings don't exist
            self.process_venues()
        
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            return mappings['by_team'].get(team.upper())
            
        except Exception as e:
            logger.error("Failed to get venue by team", team=team, error=str(e))
            return None
    
    def get_venue_by_name(self, venue_name: str) -> Dict[str, Any]:
        """Get venue information by name."""
        mappings_path = self.settings.get_data_path("silver") / "venue_mappings.json"
        
        if not mappings_path.exists():
            self.process_venues()
        
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            name_key = venue_name.lower().replace(' ', '_').replace('-', '_')
            return mappings['by_name'].get(name_key)
            
        except Exception as e:
            logger.error("Failed to get venue by name", venue=venue_name, error=str(e))
            return None
    
    def update_game_venues(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Update game data with enhanced venue information."""
        logger.info("Updating game venues", games=len(games_df))
        
        # Load venue mappings
        try:
            mappings_path = self.settings.get_data_path("silver") / "venue_mappings.json"
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except:
            # Process venues if mappings don't exist
            self.process_venues()
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        
        updated_games = games_df.copy()
        
        for idx, row in updated_games.iterrows():
            home_team = row['home_team']
            venue_info = mappings['by_team'].get(home_team)
            
            if venue_info:
                # Update venue information
                updated_games.loc[idx, 'venue'] = venue_info['venue_name']
                updated_games.loc[idx, 'venue_roof'] = venue_info['roof_type']
                
                # Add additional venue metadata if columns exist
                for field in ['latitude', 'longitude', 'elevation_ft', 'timezone', 'climate_zone']:
                    if field not in updated_games.columns:
                        updated_games[field] = None
                    updated_games.loc[idx, field] = venue_info.get(field)
        
        logger.info("Game venues updated", total_games=len(updated_games))
        return updated_games


def main():
    """CLI entry point for venue data loading."""
    parser = argparse.ArgumentParser(description="Load NFL venue data")
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only validate venue data without saving')
    parser.add_argument('--update-games', action='store_true',
                       help='Update existing games data with venue info')
    parser.add_argument('--team', type=str, help='Get venue info for specific team')
    parser.add_argument('--venue', type=str, help='Get info for specific venue')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        from utils import setup_logging
        setup_logging()
        
        loader = VenueDataLoader()
        
        if args.team:
            # Get venue for team
            venue_info = loader.get_venue_by_team(args.team)
            if venue_info:
                print(f"Venue for {args.team}:")
                print(f"  Name: {venue_info['venue_name']}")
                print(f"  City: {venue_info['city']}, {venue_info['state']}")
                print(f"  Roof: {venue_info['roof_type']}")
                print(f"  Capacity: {venue_info.get('capacity', 'Unknown'):,}")
            else:
                print(f"No venue found for team: {args.team}")
            return
        
        if args.venue:
            # Get specific venue info
            venue_info = loader.get_venue_by_name(args.venue)
            if venue_info:
                print(f"Venue: {venue_info['venue_name']}")
                print(f"  Location: {venue_info['city']}, {venue_info['state']}")
                print(f"  Home teams: {', '.join(venue_info['home_teams'])}")
                print(f"  Roof: {venue_info['roof_type']}")
                print(f"  Surface: {venue_info.get('surface', 'Unknown')}")
                print(f"  Capacity: {venue_info.get('capacity', 'Unknown'):,}")
            else:
                print(f"No venue found: {args.venue}")
            return
        
        if args.validate_only:
            # Validate venues without saving
            venues_json = loader.load_venues_json()
            venues_data = venues_json.get('venues', [])
            validated_venues = loader.validate_venues(venues_data)
            print(f"Validated {len(validated_venues)} venues")
            return
        
        if args.update_games:
            # Update games data with venue info
            from data.storage import load_dataframe, save_dataframe
            
            try:
                games_df = load_dataframe('games', layer='silver')
                updated_games = loader.update_game_venues(games_df)
                
                save_dataframe(
                    updated_games,
                    'games',
                    layer='silver',
                    save_to_db=True,
                    save_to_parquet=True
                )
                
                print(f"Updated {len(updated_games)} games with venue information")
                
            except Exception as e:
                print(f"Failed to update games: {e}")
                sys.exit(1)
            
            return
        
        # Default: process all venues
        venues_df = loader.process_venues()
        
        print(f"Successfully loaded {len(venues_df)} venues")
        print(f"Roof types: {venues_df['roof_type'].value_counts().to_dict()}")
        print(f"States: {venues_df['state'].value_counts().head().to_dict()}")
        
    except Exception as e:
        logger.error("Venue loading CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()