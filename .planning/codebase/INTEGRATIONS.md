# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**Odds Data:**
- The Odds API - Live and historical sports betting odds
  - SDK/Client: httpx (custom client in `scripts/ingest_odds.py`)
  - Auth: `ODDS_API_KEY` environment variable
  - Base URL: `https://api.the-odds-api.com/v4`
  - Timeout: 30 seconds (configurable)
  - Retries: 3 attempts with exponential backoff (4-60s wait)
  - Rate limit: 500 requests per hour

**NFL Historical Game Data:**
- nfl_data_py Python library - Schedules, results, player stats
  - SDK/Client: nfl_data_py package (`scripts/ingest_games.py`)
  - Auth: None (public data)
  - Data includes: Team names, game dates, scores, venues
  - Usage: Primary source for historical game information 2018-present

**Weather Data:**
- Meteostat - Historical and current weather data
  - SDK/Client: meteostat package (`scripts/ingest_weather.py`)
  - Auth: None (public API)
  - Data includes: Temperature, wind speed, precipitation
  - Query pattern: Coordinates-based (latitude/longitude from venue data)
  - Venue coordinates: Static JSON mapping in `data/venues.json`

## Data Storage

**Databases:**
- DuckDB (embedded)
  - Connection: File-based at `data/nfl_predictions.duckdb`
  - Purpose: Query engine for cleaned/processed data (silver/gold layers)
  - Client: duckdb package (native Python bindings)
  - Configuration: 4GB memory limit, 4 threads (configurable)
  - Read/write access: Data service (`api/services.py`), scripts

**File Storage:**
- Local filesystem (Parquet files)
  - Bronze layer: Raw snapshots from external sources
  - Silver layer: Cleaned, deduplicated tables (games, odds, weather, team stats)
  - Gold layer: Feature matrices (per prediction target: WP, ATS, O/U)
  - Outputs: Prediction artifacts, backtest results
  - Artifacts: Trained model files (joblib format)
  - Root paths configurable via `DATA_ROOT_PATH` env var or `conf/config.yaml`

**Caching:**
- Optional Redis cache (disabled by default)
  - Flag: `enable_redis_cache` in production settings
  - TTL: 3600 seconds (configurable)
  - Currently not integrated; feature for future scaling

## Authentication & Identity

**Auth Provider:**
- Custom (none centralized)
- Approach:
  - API Key for external services: `ODDS_API_KEY` (The Odds API)
  - Secret key for CSRF/session: `SECRET_KEY` environment variable (32+ chars)
  - No user authentication implemented (internal/analytical system)

**Feature Flags:**
- Environment-based toggles in settings:
  - `ENABLE_LIVE_ODDS` - Live odds data ingestion
  - `ENABLE_WEATHER_DATA` - Weather feature engineering
  - `ENABLE_BETTING_RECOMMENDATIONS` - Recommendation generation
  - `ENABLE_EMAIL_REPORTS` - Email notifications (disabled by default)
  - `ENABLE_SLACK_NOTIFICATIONS` - Slack alerts (disabled by default)
  - `MOCK_EXTERNAL_APIS` - Use mock data for development

## Monitoring & Observability

**Error Tracking:**
- None (logging-based error capture)
- Errors logged to structured logs with context
- Log format: JSON (configurable via `LOG_FORMAT` setting)

**Logs:**
- Approach: structlog with JSON output
- Destinations:
  - Console (always enabled)
  - File: `logs/nfl-predict.log` (optional, enabled by default)
- Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (via `LOG_LEVEL` env var)
- Fields: timestamp, module name, request ID, message

**Health Checks:**
- Endpoint: `GET /health` returns component status
  - Components checked: database connectivity, models availability, cache status
  - Status values: healthy, degraded, unhealthy
- Implementation: `HealthChecker` in `scripts/health_check.py`
- Interval: 300 seconds (configurable in `conf/config.yaml`)

**Data Quality Monitoring:**
- Configured thresholds in `conf/config.yaml`:
  - Min games per week: 14
  - Max missing odds: 10%
  - Max missing weather: 20%
  - Max data age: 6 hours
  - Max duplicate games: 0
  - Min odds coverage: 90%
- Feature drift detection: 20% threshold
- Extreme prediction threshold: 10% max

**Model Performance Monitoring:**
- Alerts on metric degradation:
  - Accuracy drop: 5% threshold
  - Log loss increase: 0.1 threshold
  - Calibration score: min 0.8
  - Prediction time: max 60 seconds
- Trend tracking: Last 4 weeks, 8 weeks, current season ROI

## CI/CD & Deployment

**Hosting:**
- Not preconfigured (Docker-ready via Uvicorn)
- Deployment target: Any ASGI-compatible server
- Expected: Cloud platforms (AWS, GCP, Azure) or self-hosted

**CI Pipeline:**
- None configured (not in scope for current phase)
- GitHub Actions directory exists at `.github/` (not implemented)
- Placeholder for future: Build, test, deploy automation

**Server Configuration:**
- Framework: Uvicorn ASGI server
- Workers: Configurable (default 1 in dev, 4 in production)
- Host/Port: 0.0.0.0:8000 (configurable)
- Reload: Enabled in debug mode, disabled in production
- Access logging: Enabled

## Environment Configuration

**Required env vars:**
```
ODDS_API_KEY           # The Odds API key (required for live odds)
SECRET_KEY             # CSRF/security token (min 32 chars)
ENVIRONMENT            # "development" or "production"
DEBUG                  # "true" or "false"
```

**Optional env vars:**
```
API_HOST               # Server host (default: 0.0.0.0)
API_PORT               # Server port (default: 8000)
DATA_ROOT_PATH         # Override data lake root (default: ./data)
DUCKDB_PATH            # DuckDB file path (default: data/nfl_predictions.duckdb)
LOG_LEVEL              # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT             # "json" or "text"
TIMEZONE               # IANA timezone (default: America/New_York)
LOCALE                 # Locale code (default: en_US)
ENABLE_LIVE_ODDS       # Enable odds ingestion (default: true)
ENABLE_WEATHER_DATA    # Enable weather features (default: true)
ENABLE_BETTING_RECOMMENDATIONS  # Enable recommendations (default: true)
ENABLE_EMAIL_REPORTS   # Enable email notifications (default: false)
ENABLE_SLACK_NOTIFICATIONS      # Enable Slack alerts (default: false)
MOCK_EXTERNAL_APIS     # Use mock data for development (default: false)
```

**Secrets location:**
- `.env` file (Git-ignored, never committed)
- Template: `.env.example` (safe defaults for development)
- Production: Environment variables set via infrastructure (Docker, K8s, cloud console)

## Webhooks & Callbacks

**Incoming:**
- None configured (system is read-only from external APIs)

**Outgoing:**
- None configured (potential future: Slack webhook for alerts, email notifications)
- Placeholder infrastructure: Email SMTP settings in `.env.example`
  - Fields: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, NOTIFICATION_EMAIL
  - Status: Disabled by default, awaiting implementation

## Rate Limiting & API Quotas

**The Odds API:**
- Rate limit: 500 requests per hour
- Retry strategy: 3 attempts with exponential backoff (implemented in `OddsAPIClient`)
- Mock fallback: If API unavailable or key missing, generates mock data

**Meteostat:**
- Rate limit: Not documented (appears unlimited for reasonable usage)
- No retries configured (weather failures are non-critical)

**FastAPI Endpoints:**
- Rate limiting: Disabled by default (`RATE_LIMIT_ENABLED=false`)
- Configurable: Requests per minute (default 60)
- Implementation: Ready in middleware but not active

## Data Leakage Prevention

**Design patterns to prevent leakage:**
- Walk-forward validation: Training only on historical data ≤ season-1
- Snapshot timing: Friday 6 PM ET (configured in `conf/config.yaml`)
- Data-driven feature boundaries:
  - Form metrics: Only data strictly before prediction week
  - Opponent stats: No post-game stats for current week
  - Odds: Only Friday snapshot, never live odds before prediction
- Time-split validation only (no random cross-validation)

## External Data Sources Summary

| Source | Provider | Auth | Frequency | Storage |
|--------|----------|------|-----------|---------|
| Game Schedules | nfl_data_py | None | Weekly (snapshot) | `data/silver/games.parquet` |
| Game Results | nfl_data_py | None | After games end | `data/silver/games.parquet` |
| Betting Odds | The Odds API | API Key | Friday 6 PM ET | `data/silver/odds.parquet` |
| Weather | Meteostat | None | Weekly (historical) | `data/silver/weather.parquet` |
| Team Stats | nfl_data_py | None | Weekly | `data/silver/team_stats.parquet` |
| Venue Data | Static JSON | None | Static | `data/venues.json` |

---

*Integration audit: 2026-03-18*
