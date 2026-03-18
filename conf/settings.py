"""Settings and configuration management using Pydantic."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DataPaths(BaseModel):
    """Data storage paths configuration."""

    bronze: str = "./data/bronze"
    silver: str = "./data/silver"
    gold: str = "./data/gold"
    outputs: str = "./outputs"
    artifacts: str = "./artifacts"


class DataSources(BaseModel):
    """Data source configuration."""

    games: str = "nfl_data_py"
    odds: str = "theoddsapi"
    weather: str = "meteostat"
    venues: str = "static_json"


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    root_path: str = "./data"
    snapshot_time_et: str = "Friday 18:00"
    sources: DataSources = DataSources()
    paths: DataPaths = DataPaths()
    retention_days: int = 365


class EloConfig(BaseModel):
    """Elo rating system configuration."""

    initial_rating: int = 1500
    k_factor_base: int = 20
    margin_multiplier_cap: float = 2.0
    season_carryover_shrink: int = 25


class FormConfig(BaseModel):
    """Team form metrics configuration."""

    rolling_weeks: int = 4
    min_games: int = 2


class WeatherConfig(BaseModel):
    """Weather features configuration."""

    wind_threshold: int = 12
    temp_threshold: int = 32


class MarketConfig(BaseModel):
    """Market features configuration."""

    devig_method: str = "proportional"


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    elo: EloConfig = EloConfig()
    form: FormConfig = FormConfig()
    weather: WeatherConfig = WeatherConfig()
    market: MarketConfig = MarketConfig()


class ModelConfig(BaseModel):
    """Individual model configuration."""

    algorithm: str
    calibration: bool = False


class ModelsConfig(BaseModel):
    """Models configuration."""

    wp: dict[str, Any] = {
        "algorithm": "logistic_regression",
        "calibration": True,
        "solver": "lbfgs",
        "regularization": "l2",
        "max_iter": 1000,
    }
    ats: dict[str, Any] = {
        "algorithm": "xgboost_regression",
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    ou: dict[str, Any] = {
        "algorithm": "xgboost_regression",
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    calibration: dict[str, Any] = {"method": "isotonic", "cv_folds": 5}
    features: FeaturesConfig = FeaturesConfig()


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    seasons: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    validation_method: str = "walk_forward"
    min_train_seasons: int = 2
    metrics: dict[str, list[str]] = {
        "classification": ["log_loss", "brier_score", "accuracy", "calibration_error"],
        "regression": ["mae", "rmse", "mape"],
        "betting": ["roi", "sharpe", "max_drawdown", "hit_rate"],
    }


class BettingConfig(BaseModel):
    """Betting and edge calculation configuration."""

    edge_threshold: float = 0.02
    max_edge_threshold: float = 0.15
    kelly_fraction: float = 0.25
    max_bet_size: float = 0.05
    standard_juice: int = -110
    min_probability: float = 0.01
    max_probability: float = 0.99


class APIConfig(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "NFL Prediction System API"
    description: str = "Pre-game predictions for NFL Win Probability, ATS, and O/U"
    version: str = "1.0.0"
    rate_limit: dict[str, int] = {"requests_per_minute": 60, "requests_per_hour": 1000}
    cors: dict[str, list[str]] = {
        "allow_origins": ["*"],
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["*"],
    }


class WebConfig(BaseModel):
    """Web UI configuration."""

    title: str = "NFL Predictions"
    theme: str = "light"
    default_week: str = "current"
    games_per_page: int = 16
    edge_color_thresholds: dict[str, float] = {
        "low": 0.02,
        "medium": 0.05,
        "high": 0.10,
    }
    enable_csv_download: bool = True
    enable_json_download: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    console: bool = True
    file: bool = True
    file_path: str = "./logs/nfl-predict.log"
    include_request_id: bool = True
    include_timestamp: bool = True
    include_module: bool = True


class ExternalAPIConfig(BaseModel):
    """External API configuration."""

    odds_api: dict[str, Any] = {
        "base_url": "https://api.the-odds-api.com/v4",
        "timeout": 30,
        "retries": 3,
        "rate_limit_per_hour": 500,
    }
    weather_api: dict[str, Any] = {"provider": "meteostat", "timeout": 30, "retries": 3}


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""

    enable_health_checks: bool = True
    health_check_interval: int = 300
    data_quality: dict[str, float] = {
        "min_games_per_week": 14,
        "max_missing_odds_pct": 0.1,
        "max_missing_weather_pct": 0.2,
    }
    model_performance: dict[str, float] = {
        "max_log_loss_increase": 0.1,
        "min_calibration_score": 0.8,
        "max_prediction_time": 60,
    }


class DevelopmentConfig(BaseModel):
    """Development settings."""

    debug: bool = False
    reload: bool = False
    test_data_path: str = "./tests/data"
    mock_external_apis: bool = False
    enable_profiling: bool = False
    profile_output_path: str = "./profiles"


class ProductionConfig(BaseModel):
    """Production settings."""

    secret_key_min_length: int = 32
    worker_processes: int = 4
    max_request_size: str = "10MB"
    request_timeout: int = 300
    enable_redis_cache: bool = False
    cache_ttl: int = 3600


class ConfigFromYAML(BaseModel):
    """Configuration loaded from YAML file."""

    data: DataConfig = DataConfig()
    models: ModelsConfig = ModelsConfig()
    backtest: BacktestConfig = BacktestConfig()
    betting: BettingConfig = BettingConfig()
    api: APIConfig = APIConfig()
    web: WebConfig = WebConfig()
    logging: LoggingConfig = LoggingConfig()
    external_apis: ExternalAPIConfig = ExternalAPIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    development: DevelopmentConfig = DevelopmentConfig()
    production: ProductionConfig = ProductionConfig()


class Settings(BaseSettings):
    """Main settings class combining YAML config and environment variables."""

    # Environment variables (override YAML config)
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # API Keys
    odds_api_key: str | None = Field(default=None, env="ODDS_API_KEY")
    secret_key: str | None = Field(default=None, env="SECRET_KEY")

    # Database
    duckdb_path: str | None = Field(
        default="data/nfl_predictions.duckdb", env="DUCKDB_PATH"
    )
    data_root_path: str | None = Field(default=None, env="DATA_ROOT_PATH")

    # API
    api_host: str | None = Field(default=None, env="API_HOST")
    api_port: int | None = Field(default=None, env="API_PORT")

    # Logging
    log_level: str | None = Field(default=None, env="LOG_LEVEL")
    log_format: str | None = Field(default=None, env="LOG_FORMAT")

    # Timezone
    timezone: str = Field(default="America/New_York", env="TIMEZONE")

    # Feature flags
    enable_live_odds: bool = Field(default=True, env="ENABLE_LIVE_ODDS")
    enable_weather_data: bool = Field(default=True, env="ENABLE_WEATHER_DATA")
    enable_betting_recommendations: bool = Field(
        default=True, env="ENABLE_BETTING_RECOMMENDATIONS"
    )
    enable_email_reports: bool = Field(default=False, env="ENABLE_EMAIL_REPORTS")
    enable_slack_notifications: bool = Field(
        default=False, env="ENABLE_SLACK_NOTIFICATIONS"
    )
    mock_external_apis: bool = Field(default=False, env="MOCK_EXTERNAL_APIS")

    # System settings
    locale: str = Field(default="en_US", env="LOCALE")

    # YAML configuration
    _yaml_config: ConfigFromYAML | None = None

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, config_path: str = "conf/config.yaml", **kwargs):
        """Initialize settings with YAML config and environment variables."""
        super().__init__(**kwargs)
        self._load_yaml_config(config_path)

    def _load_yaml_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                self._yaml_config = ConfigFromYAML(**yaml_data)
        else:
            # Use default configuration if file doesn't exist
            self._yaml_config = ConfigFromYAML()

    @property
    def config(self) -> ConfigFromYAML:
        """Get the YAML configuration with environment variable overrides."""
        if self._yaml_config is None:
            self._yaml_config = ConfigFromYAML()

        # Apply environment variable overrides
        config_dict = self._yaml_config.dict()

        # Override logging settings
        if self.log_level:
            config_dict["logging"]["level"] = self.log_level
        if self.log_format:
            config_dict["logging"]["format"] = self.log_format

        # Override API settings
        if self.api_host:
            config_dict["api"]["host"] = self.api_host
        if self.api_port:
            config_dict["api"]["port"] = self.api_port

        # Override data paths
        if self.data_root_path:
            config_dict["data"]["root_path"] = self.data_root_path

        # Override development settings
        config_dict["development"]["debug"] = self.debug
        config_dict["development"]["mock_external_apis"] = self.mock_external_apis

        return ConfigFromYAML(**config_dict)

    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Validate secret key length in production."""
        if v and len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    def get_data_path(self, path_type: str) -> Path:
        """Get a data path with proper resolution."""
        paths = self.config.data.paths.dict()
        if path_type in paths:
            return Path(paths[path_type]).resolve()
        raise ValueError(f"Unknown path type: {path_type}")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        paths_to_create = [
            self.get_data_path("bronze"),
            self.get_data_path("silver"),
            self.get_data_path("gold"),
            self.get_data_path("outputs"),
            self.get_data_path("artifacts"),
            Path("logs"),
        ]

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings(config_path: str = "conf/config.yaml") -> Settings:
    """Get cached settings instance."""
    return Settings(config_path=config_path)
