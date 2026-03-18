# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Data Pipeline with Layered Feature Engineering → ML Models → Prediction API

**Key Characteristics:**
- Three-layer data architecture (Bronze/Silver/Gold) with Parquet + DuckDB storage
- Walk-forward temporal validation preventing data leakage
- Modular feature engineering (Elo, weather, form, market anchors)
- Three separate specialized models (WP, ATS, O/U) unified via prediction pipeline
- FastAPI REST API with web UI frontend for predictions and analytics
- Reproducible backtesting with comprehensive metrics and reporting

## Layers

**Data Ingestion (Bronze):**
- Purpose: Ingest raw data from external sources
- Location: `scripts/ingest_games.py`, `scripts/ingest_odds.py`, `scripts/ingest_weather.py`
- Contains: Raw snapshots from nfl_data_py, TheOddsAPI, Meteostat
- Depends on: External APIs, configuration
- Used by: Data validation and Silver layer transformation

**Data Transformation (Silver):**
- Purpose: Clean and standardize data into reusable tables
- Location: `data/silver/` (Parquet partitioned by season/snapshot_ts)
- Contains: games, odds_snapshot, weather, team_stats tables
- Depends on: Bronze layer data, data quality validation
- Used by: Feature builders and Gold layer aggregation

**Feature Engineering:**
- Purpose: Create domain-specific features for models
- Location: `features/` - elo_features.py, weather.py, team_form.py, contextual.py, market_anchors.py
- Contains: Elo ratings, rolling form metrics, weather conditions, venue/context, market devig
- Depends on: Silver layer data, configuration (e.g., rolling_weeks=4, wind_threshold=12)
- Used by: Gold layer and model training

**Modeling (Gold):**
- Purpose: Create feature matrices for model training
- Location: `data/gold/` - wp.parquet, ats.parquet, ou.parquet
- Contains: Unified feature matrices per prediction target (WP/ATS/O/U)
- Depends on: Feature builders, validation logic
- Used by: Model training pipelines

**Model Training:**
- Purpose: Train and calibrate predictive models
- Location: `models/train_wp.py`, `models/train_ats.py`, `models/train_ou.py`
- Contains: WinProbabilityModel (logistic regression), ATSModel (XGBoost), OUModel (XGBoost)
- Depends on: Gold layer features, calibration module
- Used by: Prediction pipeline and backtesting

**Backtesting:**
- Purpose: Walk-forward validation across 2018-2024 with no data leakage
- Location: `backtest/walkforward.py`, `backtest/metrics.py`, `backtest/reporting.py`
- Contains: SeasonSplit definitions, metric calculation, HTML/CSV reporting
- Depends on: Model training results, historical data
- Used by: Performance validation and reporting

**API/Web:**
- Purpose: Serve predictions and analytics via REST API and HTML UI
- Location: `api/main.py` (FastAPI app), `web/` (templates + static assets)
- Contains: Endpoints for games, predictions, recommendations, backtest analytics
- Depends on: Prediction pipeline, data storage, services layer
- Used by: Web browsers and external API consumers

## Data Flow

**Snapshot (Weekly Production Run):**

1. External ingestion: `scripts/ingest_games.py --current` → Bronze
2. Raw odds snapshot: `scripts/ingest_odds.py --snapshot-time "2025-10-10T18:00:00-04:00"` → Bronze
3. Weather data: `scripts/ingest_weather.py --current` → Bronze
4. Quality validation: `scripts/data_qa.py --current-week --strict`
5. Feature building (incremental):
   - Elo updates: `scripts/build_elo.py --incremental --current-week`
   - Team form: `scripts/build_team_form.py --incremental --current-week`
   - Context/venue: `scripts/build_contextual.py --current-week`
   - Weather processing: `scripts/build_weather.py --current-week`
   - Market anchors: `scripts/build_market_anchors.py --current-week`
6. Feature matrix creation: `scripts/build_features.py --current-week --targets wp,ats,ou` → Gold
7. Feature leakage validation: `scripts/validate_features.py --current-week --check-leakage`
8. Model predictions: Unified pipeline generates WP/ATS/O/U predictions
9. Betting recommendations: Edge calculation and Kelly sizing
10. API serves predictions via `/games`, `/predictions/week/{season}/{week}`, `/recommendations/week/{season}/{week}`

**Backtest (Historical Validation):**

1. Historical data preparation: `scripts/validate_historical_data.py --seasons 2018-2024`
2. Walk-forward splits by season: For season Y, train on ≤Y-1, validate on Y (weeks 5-18)
3. For each split:
   - Feature building for training + validation periods
   - Model training on training seasons only
   - Prediction on validation season
   - Metric calculation (accuracy, ROI, calibration)
4. Aggregated metrics: Seasonal breakdown, cumulative performance, trend analysis
5. Report generation: HTML interactive charts, CSV downloads, calibration curves

**State Management:**
- Configuration: `conf/config.yaml` (Pydantic-based with env var overrides)
- Model artifacts: `artifacts/models/` (joblib/pickle serialized models)
- Elo ratings: Embedded in feature builder, persisted with each training run
- Prediction outputs: `outputs/` (Parquet for raw predictions, JSON for API responses)
- Backtest results: `outputs/backtest/` (metrics, reports, betting simulation)

## Key Abstractions

**DuckDBConnection:**
- Purpose: Manage DuckDB connections for in-memory and file-based databases
- Examples: `data/storage.py` lines 21-167
- Pattern: Singleton-like factory with connection pooling, table sanitization, query execution

**ParquetManager:**
- Purpose: Handle Parquet file I/O with partitioning support
- Examples: `data/storage.py` lines 235-523
- Pattern: Abstracts complex partition directory structures, handles timezone normalization

**WalkForwardValidator:**
- Purpose: Orchestrate temporal train-test splits with strict chronological ordering
- Examples: `models/utils.py`
- Pattern: Prevents look-ahead bias by ensuring validation always on future data

**NFLPredictionPipeline:**
- Purpose: Unified interface combining WP, ATS, O/U models with fair lines and edge calculations
- Examples: `models/prediction_pipeline.py` lines 348-836
- Pattern: Coordinates model outputs, generates betting recommendations, calculates Kelly fractions

**EloRatingSystem:**
- Purpose: Maintain and update Elo ratings with dynamic K factors based on margin
- Examples: `ratings/elo.py`
- Pattern: Stateful rating persistence, margin-of-victory adjustment, season carryover shrinking

**BacktestMetricsCalculator:**
- Purpose: Compute classification/regression/betting metrics for model evaluation
- Examples: `backtest/metrics.py`
- Pattern: Metric aggregation per season and overall, confidence interval calculation

**DataService (API):**
- Purpose: Query and format prediction data for REST endpoints
- Examples: `api/services.py`
- Pattern: Caches game lists, filters by week/team/status, formats response DTOs

## Entry Points

**Data Snapshot (Makefile):**
- Location: `make snapshot` (defined in `Makefile` lines 45-73)
- Triggers: Friday 6 PM ET (via cron or manual invocation)
- Responsibilities: Orchestrate weekly data ingestion, feature building, validation

**Prediction Generation:**
- Location: `make predict` (defined in `Makefile` lines 99-110)
- Triggers: After snapshot completes
- Responsibilities: Load trained models, generate predictions for current week, export to Parquet/JSON

**API Server:**
- Location: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Entry point: `api/main.py` (FastAPI app created at line 287)
- Lifespan hooks: Initialize models (line 70) and database (line 88) on startup
- Responsibilities: Serve REST endpoints, render HTML templates, handle CORS/rate limiting

**Backtest Runner:**
- Location: `make backtest` (defined in `Makefile` lines 75-97)
- Triggers: Manual or CI/CD pipeline
- Responsibilities: Run walk-forward validation 2018-2024, generate metrics and HTML report

## Error Handling

**Strategy:** Layered validation with exceptions propagating up for logging

**Patterns:**
- Data ingestion errors caught in scripts with retry logic (tenacity decorator)
- Feature validation warnings logged but not fatal (bad data rows dropped)
- Model training errors halt execution to prevent invalid artifacts
- API errors return structured JSON responses with HTTP status codes (500, 422, 404)
- Database transaction rollback on query failure (`data/storage.py` lines 554-566)

**Custom Exceptions:**
- `DataIngestionError`: Raised by storage layer on I/O failures (`data/storage.py` line 15)
- `BacktestError`: Raised by backtesting system for validation failures
- `ModelTrainingError`: Raised by model modules on training issues

## Cross-Cutting Concerns

**Logging:**
- Implementation: Pydantic-based config at `conf/settings.py` (LoggingConfig class)
- Usage: `from utils import get_logger; logger = get_logger(__name__)` in all modules
- Format: JSON structured logs with module, timestamp, request_id fields
- File: `logs/nfl-predict.log` (configured in settings)

**Validation:**
- Feature leakage prevention: `scripts/validate_features.py` checks no future data used
- Data quality checks: Minimum games per week, missing odds/weather thresholds
- Input sanitization: DuckDB table names sanitized against SQL injection (`data/storage.py` line 136)
- Request validation: Pydantic schemas in `api/schemas.py` validate all API inputs

**Authentication:**
- Strategy: None (research system, no user auth)
- Rate limiting: Configured in `conf/settings.py` APIConfig (requests_per_minute/hour)
- CORS: Configured to allow origins and methods per settings

**Configuration Management:**
- Environment: `.env` file with secrets (ODDS_API_KEY, SECRET_KEY)
- Settings: `conf/config.yaml` for feature params (K-factors, thresholds, model hyperparams)
- Precedence: Environment variables override YAML, YAML overrides defaults
- Validation: Pydantic validators check secret_key length in production mode

---

*Architecture analysis: 2026-03-18*
