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

- [x] **2.1** Implement data storage foundation
  - Create data directory structure (bronze/silver/gold)   ✅ COMPLETE
  - Set up DuckDB connection utilities                     ✅ COMPLETE
  - Implement Parquet read/write helpers                   ✅ COMPLETE

- [x] **2.2** Create data schemas and validation
  - Define schema classes for games, odds, weather, team_form tables ✅ COMPLETE
  - Implement data validation functions                            ✅ COMPLETE
  - Create data quality check utilities                            ✅ COMPLETE

- [x] **2.3** Implement game data ingestion (`scripts/ingest_games.py`)
  - Connect to nfl_data_py                                ✅ COMPLETE
  - Pull season schedules and historical results          ✅ COMPLETE
  - Normalize team names to canonical IDs                 ✅ COMPLETE
  - Store raw data in bronze, cleaned in silver           ✅ COMPLETE
  - Handle incremental updates                             ✅ COMPLETE

- [x] **2.4** Implement odds data ingestion (`scripts/ingest_odds.py`)
  - Set up API connection (TheOddsAPI or similar)         ✅ COMPLETE
  - Implement Friday 6PM ET snapshot timing               ✅ COMPLETE
  - Store odds snapshots with proper timestamps           ✅ COMPLETE
  - Handle multiple sportsbooks and line movements        ✅ COMPLETE
  - Implement rate limiting and retry logic               ✅ COMPLETE

- [x] **2.5** Implement weather data ingestion (`scripts/ingest_weather.py`)
  - Connect to Meteostat API                              ✅ COMPLETE
  - Get forecasts for stadium locations at kickoff times  ✅ COMPLETE
  - Handle indoor/outdoor/retractable venue types         ✅ COMPLETE
  - Store weather data with game linkage                  ✅ COMPLETE

- [x] **2.6** Create stadium/venue data
  - Static JSON mapping of venues with coordinates        ✅ COMPLETE
  - Venue roof types (indoor/outdoor/retractable)         ✅ COMPLETE
  - Time zone information for travel calculations         ✅ COMPLETE

- [x] **2.7** Implement data QA and monitoring
  - Row count validation per week                         ✅ COMPLETE
  - Missing data detection (odds/weather)                 ✅ COMPLETE
  - Sanity checks (spreads in [-20,+20], totals in [30,65]) ✅ COMPLETE
  - Duplicate game detection                              ✅ COMPLETE

### Phase 3: Feature Engineering System
**Timeline: Days 9-14**

- [x] **3.1** Implement Elo rating system (`ratings/elo.py`)
  - Base Elo calculations (init at 1500)                     ✅ COMPLETE
  - Margin-of-victory adjustments with dynamic K-factor     ✅ COMPLETE
  - Home-field advantage learning per season                ✅ COMPLETE
  - Season carryover with 25 Elo shrinkage                  ✅ COMPLETE
  - Chronological updates across multiple seasons           ✅ COMPLETE
  - Consider Glicko rating system as Elo alternative        ✅ COMPLETE (uncertainty tracking implemented)

- [x] **3.2** Implement team form metrics
  - Rolling 4-week EPA/play calculations (offense/defense)     ✅ COMPLETE
  - Success rate metrics (offense/defense)                    ✅ COMPLETE
  - Neutral situation pass rate calculations                  ✅ COMPLETE
  - Rest days since last game                                 ✅ COMPLETE
  - Ensure no data leakage (only past games)                  ✅ COMPLETE

- [x] **3.3** Implement contextual features
  - Travel time-zone difference calculations               ✅ COMPLETE
  - Short week detection                                   ✅ COMPLETE
  - Venue roof type encoding                              ✅ COMPLETE
  - Home/away team indicators                             ✅ COMPLETE

- [x] **3.4** Implement weather features
  - Wind speed (primary factor)                           ✅ COMPLETE
  - Temperature, precipitation probability                 ✅ COMPLETE
  - Outdoor game filtering                                 ✅ COMPLETE
  - Weather impact only for outdoor/retractable venues    ✅ COMPLETE

- [x] **3.5** Implement market anchor features
  - Opening line capture and storage                      ✅ COMPLETE
  - Current line at snapshot time                         ✅ COMPLETE
  - Moneyline to probability conversions with devig       ✅ COMPLETE
  - Never use closing lines (strict no-leakage)           ✅ COMPLETE

- [x] **3.6** Create feature building pipeline (`scripts/build_features.py`)
  - Combine all feature sources                              ✅ COMPLETE
  - Handle missing data and outliers (winsorization)         ✅ COMPLETE
  - Z-score normalization within seasons                     ✅ COMPLETE
  - Generate separate feature matrices for WP/ATS/OU         ✅ COMPLETE
  - Store features in gold layer                             ✅ COMPLETE

- [x] **3.7** Implement feature validation
  - Check for data leakage in feature construction                ✅ COMPLETE
  - Validate feature distributions                               ✅ COMPLETE
  - Test feature pipeline end-to-end                             ✅ COMPLETE

### Phase 4: Modeling Infrastructure
**Timeline: Days 15-18**

- [x] **4.1** Create model training utilities (`models/utils.py`)
  - Walk-forward validation framework                   ✅ COMPLETE
  - Data splitting by season/week                       ✅ COMPLETE
  - Cross-validation utilities                          ✅ COMPLETE
  - Model serialization/loading                         ✅ COMPLETE

- [x] **4.2** Implement probability calibration (`models/calibrate.py`)
  - Isotonic regression calibration                     ✅ COMPLETE
  - Platt scaling as fallback                           ✅ COMPLETE
  - Within-season calibration on validation folds       ✅ COMPLETE
  - Calibration curve generation                        ✅ COMPLETE

- [x] **4.3** Implement baseline Elo model                     ✅ COMPLETE
  - Pure Elo-based win probability                          ✅ COMPLETE
  - Logistic regression on Elo difference                   ✅ COMPLETE
  - Home field advantage integration                        ✅ COMPLETE
  - Serve as baseline comparison                            ✅ COMPLETE

- [x] **4.4** Create model evaluation framework             ✅ COMPLETE
  - LogLoss, Brier score for WP                          ✅ COMPLETE
  - MAE, RMSE for regression targets                     ✅ COMPLETE
  - Calibration metrics (ECE, reliability diagrams)     ✅ COMPLETE
  - Betting simulation metrics (ROI, CLV)               ✅ COMPLETE

### Phase 5: Core Prediction Models
**Timeline: Days 19-22**

- [x] **5.1** Implement Win Probability model (`models/train_wp.py`)      ✅ COMPLETE
  - Logistic regression with L1/L2 regularization               ✅ COMPLETE
  - Feature selection and hyperparameter tuning                 ✅ COMPLETE
  - Walk-forward training protocol                              ✅ COMPLETE
  - Calibration integration                                     ✅ COMPLETE
  - Model persistence and loading                               ✅ COMPLETE

- [x] **5.2** Implement ATS model (`models/train_ats.py`)           ✅ COMPLETE
  - XGBoost/LightGBM regression for expected margin            ✅ COMPLETE
  - Convert to cover probability using residual distribution   ✅ COMPLETE
  - Implement both classification and regression approaches     ✅ COMPLETE
  - Handle spread betting mechanics                             ✅ COMPLETE
  - Feature importance tracking                                 ✅ COMPLETE

- [x] **5.3** Implement Over/Under model (`models/train_ou.py`)     ✅ COMPLETE
  - XGBoost/LightGBM regression for total points              ✅ COMPLETE
  - Convert to over/under probabilities                       ✅ COMPLETE
  - Optional: Poisson score model for total points simulation ✅ COMPLETE
  - Weather impact modeling for totals                        ✅ COMPLETE
  - Handle total betting mechanics                             ✅ COMPLETE

- [x] **5.4** Create prediction pipeline                    ✅ COMPLETE
  - Combine all models for unified predictions            ✅ COMPLETE
  - Generate fair lines and probabilities                ✅ COMPLETE
  - Calculate edges vs market lines                       ✅ COMPLETE
  - Output structured prediction format                   ✅ COMPLETE

### Phase 5.5: Betting Utilities & Selection  
**Timeline: Days 22.5-23.5**

- [x] **5.5.1** Implement Expected Value calculations
  - American odds to implied probability conversion                ✅ COMPLETE
  - EV calculation for different bet types (spread, total, ML)     ✅ COMPLETE
  - Devig utilities for removing sportsbook margin                ✅ COMPLETE
  
- [x] **5.5.2** Implement Kelly Criterion bet sizing
  - Full Kelly and fractional Kelly calculations                 ✅ COMPLETE
  - Bankroll management with drawdown limits                     ✅ COMPLETE
  - Unit size determination based on edge and confidence         ✅ COMPLETE

- [x] **5.5.3** Create bet selection and filtering                     ✅ COMPLETE
  - Minimum edge thresholds (configurable, default 2%)              ✅ COMPLETE
  - Confidence requirements (|prob - 0.5| >= threshold)             ✅ COMPLETE
  - Maximum bets per week limits (top N edges)                      ✅ COMPLETE
  - Disagreement vs closing line detection                          ✅ COMPLETE

- [x] **5.5.4** Implement bet recommendation engine                      ✅ COMPLETE
  - Rank potential bets by EV and confidence                        ✅ COMPLETE
  - Generate recommended unit sizes                                 ✅ COMPLETE
  - Output structured bet recommendations                           ✅ COMPLETE

### Phase 6: Backtesting & Evaluation System
**Timeline: Days 24-27**

- [x] **6.1** Implement walk-forward backtesting (`backtest/walkforward.py`) ✅ COMPLETE
  - Season-by-season validation (2018-2024)                         ✅ COMPLETE
  - Strict temporal ordering                                         ✅ COMPLETE
  - No data leakage validation                                       ✅ COMPLETE
  - Progress tracking and logging                                    ✅ COMPLETE

- [x] **6.2** Create evaluation metrics (`backtest/metrics.py`)            ✅ COMPLETE
  - Model performance metrics (LogLoss, Brier, MAE, RMSE)               ✅ COMPLETE
  - Calibration analysis (ECE, reliability diagrams)                    ✅ COMPLETE
  - Edge bucket analysis                                                 ✅ COMPLETE
  - Statistical significance testing                                     ✅ COMPLETE

- [x] **6.3** Implement betting simulation                               ✅ COMPLETE
  - ROI calculation at -110 juice                                      ✅ COMPLETE
  - Kelly criterion sizing (fractional)                                ✅ COMPLETE
  - CLV tracking if closing lines available                            ✅ COMPLETE
  - Bankroll simulation over time                                      ✅ COMPLETE

- [x] **6.4** Create backtest reporting                                      ✅ COMPLETE
  - HTML report generation with charts                                ✅ COMPLETE
  - Season-by-season performance breakdown                            ✅ COMPLETE
  - Cohort analysis (weather, venue, travel)                         ✅ COMPLETE
  - Sensitivity analysis on EV thresholds and Kelly fractions        ✅ COMPLETE
  - CSV export for detailed analysis                                  ✅ COMPLETE

- [x] **6.5** Validate backtest with known results                        ✅ COMPLETE
  - Test on subset of historical data                               ✅ COMPLETE
  - Compare against frozen artifacts                                ✅ COMPLETE
  - Ensure reproducibility                                          ✅ COMPLETE

### Phase 7: API & Web Interface
**Timeline: Days 28-31**

- [x] **7.1** Create FastAPI application (`api/main.py`)                    ✅ COMPLETE
  - API endpoint structure per PRD specification                    ✅ COMPLETE
  - Response schemas (`api/schemas.py`)                              ✅ COMPLETE
  - Error handling and validation                                   ✅ COMPLETE
  - Request/response logging                                        ✅ COMPLETE

- [x] **7.2** Implement core API endpoints                                ✅ COMPLETE
  - `GET /current-week` - current season/week metadata                   ✅ COMPLETE
  - `GET /games` - list games with predictions                           ✅ COMPLETE
  - `GET /games/{game_id}` - single game details                         ✅ COMPLETE
  - `GET /backtest` - summary metrics                                    ✅ COMPLETE
  - `GET /calibration` - reliability data                                ✅ COMPLETE

- [x] **7.3** Create web UI templates (`web/templates/`)                    ✅ COMPLETE
  - Base template with Tailwind CSS                              ✅ COMPLETE
  - Current week games table                                      ✅ COMPLETE
  - Game detail view                                              ✅ COMPLETE
  - Backtest reports page                                         ✅ COMPLETE
  - Mobile-responsive design                                      ✅ COMPLETE

- [x] **7.4** Implement web UI functionality                          ✅ COMPLETE
  - Server-rendered Jinja templates                                ✅ COMPLETE
  - Client-side filtering and sorting                             ✅ COMPLETE
  - Edge threshold sliders                                         ✅ COMPLETE
  - CSV download capabilities                                      ✅ COMPLETE
  - "As of" timestamp display                                      ✅ COMPLETE

- [x] **7.5** Create static assets (`web/static/`)                            ✅ COMPLETE
  - Tailwind CSS compilation                                      ✅ COMPLETE
  - JavaScript for interactivity                                 ✅ COMPLETE
  - Charts for calibration curves                                ✅ COMPLETE
  - Mobile-optimized styling                                     ✅ COMPLETE

### Phase 8: Integration & Testing
**Timeline: Days 32-35**

- [x] **8.1** Create comprehensive test suite                              ✅ COMPLETE
  - Unit tests for all feature builders                               ✅ COMPLETE
  - Unit tests for Elo updates and probability conversions            ✅ COMPLETE
  - Integration tests for end-to-end pipeline                         ✅ COMPLETE
  - API endpoint contract tests                                       ✅ COMPLETE
  - HTML snapshot tests for UI                                        ✅ COMPLETE

- [x] **8.2** End-to-end integration testing                                 ✅ COMPLETE
  - Run complete pipeline on historical week                          ✅ COMPLETE
  - Validate all outputs and artifacts                                ✅ COMPLETE
  - Test API responses and UI rendering                               ✅ COMPLETE
  - Performance benchmarking                                          ✅ COMPLETE

- [x] **8.3** Create operational procedures                                     ✅ COMPLETE
  - Data validation checks                                            ✅ COMPLETE
  - Model drift monitoring                                            ✅ COMPLETE
  - Alert thresholds and notifications                               ✅ COMPLETE
  - Recovery procedures for failures                                 ✅ COMPLETE

- [x] **8.4** Documentation and examples                                     ✅ COMPLETE
  - API documentation                                               ✅ COMPLETE
  - Feature engineering documentation                               ✅ COMPLETE
  - Model methodology documentation                                 ✅ COMPLETE
  - Operational runbooks                                            ✅ COMPLETE

### Phase 9: Deployment & Orchestration
**Timeline: Days 36-39**

- [x] **9.1** Create Make commands per acceptance criteria                ✅ COMPLETE
  - `make snapshot` - produce silver/gold tables                        ✅ COMPLETE
  - `make backtest` - run walk-forward validation                       ✅ COMPLETE
  - `make predict` - generate prediction artifacts                      ✅ COMPLETE
  - `make serve` - start web UI/API                                     ✅ COMPLETE

- [x] **9.2** Set up scheduling system                                  ✅ COMPLETE
  - Friday 5:00 PM ET: data updates                                   ✅ COMPLETE
  - Friday 6:00 PM ET: odds snapshot → predictions                    ✅ COMPLETE
  - Cron job configuration                                            ✅ COMPLETE
  - GitHub Actions workflow (optional)                                ✅ COMPLETE

- [x] **9.3** Implement monitoring and alerting                        ✅ COMPLETE
  - Data ingestion success/failure alerts                            ✅ COMPLETE
  - Model prediction output validation                               ✅ COMPLETE
  - API health checks                                                ✅ COMPLETE
  - Performance monitoring                                           ✅ COMPLETE

- [x] **9.4** Create deployment configuration                            ✅ COMPLETE
  - Uvicorn/Gunicorn production setup                                 ✅ COMPLETE
  - Environment variable configuration                               ✅ COMPLETE
  - Static asset serving                                             ✅ COMPLETE
  - Security considerations                                          ✅ COMPLETE

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

**Current Status**: ✅ Phase 9.4 COMPLETE - Comprehensive deployment configuration implemented with production-ready Gunicorn/Uvicorn setup, enhanced security middleware, static asset serving with caching, environment variable configuration, Docker containerization, nginx reverse proxy configuration, and complete deployment documentation. System ready for production deployment.
**Next Milestone**: Project deployment and production operations