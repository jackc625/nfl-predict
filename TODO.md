# NFL Prediction System - Task Manager

This document outlines the comprehensive implementation plan for the NFL prediction system based on the PRD requirements.

## 🎯 Project Goals Recap
- Generate calibrated pre-game probabilities (WP, ATS, O/U) by Friday 6:00 PM ET
- End-to-end reproducibility with deterministic data snapshots
- Walk-forward backtest evaluation (≥5 seasons) with betting metrics
- Simple web UI for predictions and recommended bets

## 📋 Implementation Phases

### Phase 1: Project Foundation & Setup
**Timeline: Days 1-3**

- [x] **1.1** Set up project directory structure according to PRD specification
  ```
  project/
  ├─ data/ (bronze/silver/gold)   ✅ COMPLETE
  ├─ scripts/ (ingestion)         ✅ COMPLETE
  ├─ ratings/ (elo)              ✅ COMPLETE
  ├─ models/ (training)          ✅ COMPLETE
  ├─ backtest/ (validation)      ✅ COMPLETE
  ├─ api/ (FastAPI)              ✅ COMPLETE
  ├─ web/ (templates/static)     ✅ COMPLETE (need to add templates/static subdirs)
  ├─ conf/ (config)              ✅ COMPLETE
  ├─ tests/                      ✅ COMPLETE
  ├─ outputs/                    ✅ COMPLETE
  ```

- [x] **1.2** Create `pyproject.toml` with all dependencies from PRD
  - Core: python>=3.11, pandas, numpy, pyarrow, duckdb  ✅ COMPLETE
  - Modeling: scikit-learn, xgboost, statsmodels        ✅ COMPLETE
  - API/UI: fastapi, uvicorn, jinja2, pydantic         ✅ COMPLETE
  - HTTP/ETL: httpx, tenacity, pytz, python-dateutil   ✅ COMPLETE
  - Data: nfl_data_py, meteostat                       ✅ COMPLETE
  - Testing: pytest, pytest-cov                       ✅ COMPLETE

- [x] **1.3** Set up Python virtual environment and install dependencies
  - Create `.venv/` directory                          ✅ COMPLETE
  - Install all required packages                      ✅ COMPLETE
  - Verify installation with import tests              ✅ COMPLETE

- [x] **1.4** Create configuration system
  - `conf/config.yaml` with all settings from PRD        ✅ COMPLETE
  - `.env.example` template for API keys                 ✅ COMPLETE  
  - Configuration loading utilities with pydantic-settings ✅ COMPLETE

- [x] **1.5** Initialize git repository and create initial commit
  - Set up `.gitignore` (exclude .env, .venv/, data/, outputs/) ✅ COMPLETE
  - Create initial commit with project structure              ✅ COMPLETE

- [x] **1.6** Create basic logging and utilities
  - Structured logging setup with request IDs        ✅ COMPLETE
  - Common utilities (date handling, probability conversions) ✅ COMPLETE  
  - Exception handling classes                       ✅ COMPLETE

### Phase 2: Data Infrastructure & Ingestion
**Timeline: Days 4-8**

- [ ] **2.1** Implement data storage foundation
  - Create data directory structure (bronze/silver/gold)
  - Set up DuckDB connection utilities
  - Implement Parquet read/write helpers

- [ ] **2.2** Create data schemas and validation
  - Define schema classes for games, odds, weather, team_form tables
  - Implement data validation functions
  - Create data quality check utilities

- [ ] **2.3** Implement game data ingestion (`scripts/ingest_games.py`)
  - Connect to nfl_data_py
  - Pull season schedules and historical results
  - Normalize team names to canonical IDs
  - Store raw data in bronze, cleaned in silver
  - Handle incremental updates

- [ ] **2.4** Implement odds data ingestion (`scripts/ingest_odds.py`)
  - Set up API connection (TheOddsAPI or similar)
  - Implement Friday 6PM ET snapshot timing
  - Store odds snapshots with proper timestamps
  - Handle multiple sportsbooks and line movements
  - Implement rate limiting and retry logic

- [ ] **2.5** Implement weather data ingestion (`scripts/ingest_weather.py`)
  - Connect to Meteostat API
  - Get forecasts for stadium locations at kickoff times
  - Handle indoor/outdoor/retractable venue types
  - Store weather data with game linkage

- [ ] **2.6** Create stadium/venue data
  - Static JSON mapping of venues with coordinates
  - Venue roof types (indoor/outdoor/retractable)
  - Time zone information for travel calculations

- [ ] **2.7** Implement data QA and monitoring
  - Row count validation per week
  - Missing data detection (odds/weather)
  - Sanity checks (spreads in [-20,+20], totals in [30,65])
  - Duplicate game detection

### Phase 3: Feature Engineering System
**Timeline: Days 9-14**

- [ ] **3.1** Implement Elo rating system (`ratings/elo.py`)
  - Base Elo calculations (init at 1500)
  - Margin-of-victory adjustments with dynamic K-factor
  - Home-field advantage learning per season
  - Season carryover with 25 Elo shrinkage
  - Chronological updates across multiple seasons

- [ ] **3.2** Implement team form metrics
  - Rolling 4-week EPA/play calculations (offense/defense)
  - Success rate metrics (offense/defense)
  - Neutral situation pass rate calculations
  - Rest days since last game
  - Ensure no data leakage (only past games)

- [ ] **3.3** Implement contextual features
  - Travel time-zone difference calculations
  - Short week detection
  - Venue roof type encoding
  - Home/away team indicators

- [ ] **3.4** Implement weather features
  - Wind speed (primary factor)
  - Temperature, precipitation probability
  - Outdoor game filtering
  - Weather impact only for outdoor/retractable venues

- [ ] **3.5** Implement market anchor features
  - Opening line capture and storage
  - Current line at snapshot time
  - Moneyline to probability conversions with devig
  - Never use closing lines (strict no-leakage)

- [ ] **3.6** Create feature building pipeline (`scripts/build_features.py`)
  - Combine all feature sources
  - Handle missing data and outliers (winsorization)
  - Z-score normalization within seasons
  - Generate separate feature matrices for WP/ATS/OU
  - Store features in gold layer

- [ ] **3.7** Implement feature validation
  - Check for data leakage in feature construction
  - Validate feature distributions
  - Test feature pipeline end-to-end

### Phase 4: Modeling Infrastructure
**Timeline: Days 15-18**

- [ ] **4.1** Create model training utilities (`models/utils.py`)
  - Walk-forward validation framework
  - Data splitting by season/week
  - Cross-validation utilities
  - Model serialization/loading

- [ ] **4.2** Implement probability calibration (`models/calibrate.py`)
  - Isotonic regression calibration
  - Platt scaling as fallback
  - Within-season calibration on validation folds
  - Calibration curve generation

- [ ] **4.3** Implement baseline Elo model
  - Pure Elo-based win probability
  - Logistic regression on Elo difference
  - Home field advantage integration
  - Serve as baseline comparison

- [ ] **4.4** Create model evaluation framework
  - LogLoss, Brier score for WP
  - MAE, RMSE for regression targets
  - Calibration metrics (ECE, reliability diagrams)
  - Betting simulation metrics (ROI, CLV)

### Phase 5: Core Prediction Models
**Timeline: Days 19-22**

- [ ] **5.1** Implement Win Probability model (`models/train_wp.py`)
  - Logistic regression with L1/L2 regularization
  - Feature selection and hyperparameter tuning
  - Walk-forward training protocol
  - Calibration integration
  - Model persistence and loading

- [ ] **5.2** Implement ATS model (`models/train_ats.py`)
  - XGBoost/LightGBM regression for expected margin
  - Convert to cover probability using residual distribution
  - Handle spread betting mechanics
  - Feature importance tracking

- [ ] **5.3** Implement Over/Under model (`models/train_ou.py`)
  - XGBoost/LightGBM regression for total points
  - Convert to over/under probabilities
  - Weather impact modeling for totals
  - Handle total betting mechanics

- [ ] **5.4** Create prediction pipeline
  - Combine all models for unified predictions
  - Generate fair lines and probabilities
  - Calculate edges vs market lines
  - Output structured prediction format

### Phase 6: Backtesting & Evaluation System
**Timeline: Days 23-26**

- [ ] **6.1** Implement walk-forward backtesting (`backtest/walkforward.py`)
  - Season-by-season validation (2018-2024)
  - Strict temporal ordering
  - No data leakage validation
  - Progress tracking and logging

- [ ] **6.2** Create evaluation metrics (`backtest/metrics.py`)
  - Model performance metrics (LogLoss, Brier, MAE, RMSE)
  - Calibration analysis (ECE, reliability diagrams)
  - Edge bucket analysis
  - Statistical significance testing

- [ ] **6.3** Implement betting simulation
  - ROI calculation at -110 juice
  - Kelly criterion sizing (fractional)
  - CLV tracking if closing lines available
  - Bankroll simulation over time

- [ ] **6.4** Create backtest reporting
  - HTML report generation with charts
  - Season-by-season performance breakdown
  - Cohort analysis (weather, venue, travel)
  - CSV export for detailed analysis

- [ ] **6.5** Validate backtest with known results
  - Test on subset of historical data
  - Compare against frozen artifacts
  - Ensure reproducibility

### Phase 7: API & Web Interface
**Timeline: Days 27-30**

- [ ] **7.1** Create FastAPI application (`api/main.py`)
  - API endpoint structure per PRD specification
  - Response schemas (`api/schemas.py`)
  - Error handling and validation
  - Request/response logging

- [ ] **7.2** Implement core API endpoints
  - `GET /api/weeks/current` - current season/week metadata
  - `GET /api/games` - list games with predictions
  - `GET /api/games/{game_id}` - single game details
  - `GET /api/reports/backtest` - summary metrics
  - `GET /api/reports/calibration` - reliability data

- [ ] **7.3** Create web UI templates (`web/templates/`)
  - Base template with Tailwind CSS
  - Current week games table
  - Game detail view
  - Backtest reports page
  - Mobile-responsive design

- [ ] **7.4** Implement web UI functionality
  - Server-rendered Jinja templates
  - Client-side filtering and sorting
  - Edge threshold sliders
  - CSV download capabilities
  - "As of" timestamp display

- [ ] **7.5** Create static assets (`web/static/`)
  - Tailwind CSS compilation
  - JavaScript for interactivity
  - Charts for calibration curves
  - Mobile-optimized styling

### Phase 8: Integration & Testing
**Timeline: Days 31-34**

- [ ] **8.1** Create comprehensive test suite
  - Unit tests for all feature builders
  - Unit tests for Elo updates and probability conversions
  - Integration tests for end-to-end pipeline
  - API endpoint contract tests
  - HTML snapshot tests for UI

- [ ] **8.2** End-to-end integration testing
  - Run complete pipeline on historical week
  - Validate all outputs and artifacts
  - Test API responses and UI rendering
  - Performance benchmarking

- [ ] **8.3** Create operational procedures
  - Data validation checks
  - Model drift monitoring
  - Alert thresholds and notifications
  - Recovery procedures for failures

- [ ] **8.4** Documentation and examples
  - API documentation
  - Feature engineering documentation
  - Model methodology documentation
  - Operational runbooks

### Phase 9: Deployment & Orchestration
**Timeline: Days 35-38**

- [ ] **9.1** Create Make commands per acceptance criteria
  - `make snapshot` - produce silver/gold tables
  - `make backtest` - run walk-forward validation
  - `make predict` - generate prediction artifacts
  - `make serve` - start web UI/API

- [ ] **9.2** Set up scheduling system
  - Friday 5:00 PM ET: data updates
  - Friday 6:00 PM ET: odds snapshot → predictions
  - Cron job configuration
  - GitHub Actions workflow (optional)

- [ ] **9.3** Implement monitoring and alerting
  - Data ingestion success/failure alerts
  - Model prediction output validation
  - API health checks
  - Performance monitoring

- [ ] **9.4** Create deployment configuration
  - Uvicorn/Gunicorn production setup
  - Environment variable configuration
  - Static asset serving
  - Security considerations

## 🏁 Acceptance Criteria Validation

Before considering the project complete, verify all acceptance criteria from the PRD:

1. ✅ `make snapshot` produces silver/gold tables for current week
2. ✅ `make backtest` runs walk-forward 2018-2024 with metrics report
3. ✅ `make predict` writes prediction artifacts (Parquet + JSON)
4. ✅ Web UI at `/` lists current week games with model outputs and edges
5. ✅ Calibration page shows reliability curves for WP and ATS

## 🔍 Quality Gates

Each phase should include:
- [ ] Code review and testing
- [ ] Documentation updates
- [ ] Performance validation
- [ ] Security review (no secrets in code)
- [ ] User acceptance if UI changes

## ⚠️ Critical Dependencies

- **Phase 2 → 3**: Data ingestion must be complete before feature engineering
- **Phase 3 → 4**: Features must be validated before model training
- **Phase 4 → 5**: Infrastructure must be solid before implementing models
- **Phase 5 → 6**: Models must be trained before backtesting
- **Phase 6 → 7**: Evaluation complete before UI implementation

## 📊 Progress Tracking

Use this document to track progress. Mark items as complete with ✅ and note any blockers or changes needed.

**Current Status**: ✅ Phase 1 COMPLETE - Project foundation established
**Next Milestone**: Begin Phase 2 - Data Infrastructure & Ingestion (Days 4-8)