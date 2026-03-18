# Codebase Structure

**Analysis Date:** 2026-03-18

## Directory Layout

```
nfl-predict/
├── api/                    # FastAPI REST API and web services
│   ├── __init__.py
│   ├── main.py             # FastAPI app with all endpoints
│   ├── config.py           # API-specific configuration
│   ├── schemas.py          # Pydantic DTOs (request/response models)
│   ├── services.py         # DataService, BacktestService abstractions
│   ├── middleware.py       # CORS, rate limiting, logging middleware
│   └── exceptions.py       # Custom exceptions and error handlers
│
├── backtest/               # Walk-forward backtesting system
│   ├── __init__.py
│   ├── walkforward.py      # WalkForwardBacktester, SeasonSplit, BacktestConfig
│   ├── metrics.py          # BacktestMetricsCalculator, metric definitions
│   ├── clv_tracking.py     # Closing Line Value analysis for bet evaluation
│   ├── reporting.py        # BacktestReporter, HTML/CSV generation
│   ├── visualization.py    # Chart generation for reports
│   └── templates/          # HTML templates for reports (Jinja2)
│
├── conf/                   # Configuration management
│   ├── __init__.py
│   ├── config.yaml         # All settings (data paths, model params, API config)
│   └── settings.py         # Pydantic Settings class, env var merging
│
├── data/                   # Data storage directories (auto-created)
│   ├── bronze/             # Raw ingested data (not versioned)
│   ├── silver/             # Cleaned tables
│   │   ├── season=2018/    # Games, odds_snapshot, weather (partitioned)
│   │   ├── season=2019/
│   │   └── ...
│   ├── gold/               # Feature matrices
│   │   ├── wp.parquet      # WP model features
│   │   ├── ats.parquet     # ATS model features
│   │   └── ou.parquet      # O/U model features
│   ├── schemas.py          # Data table schema definitions
│   └── storage.py          # DuckDBConnection, ParquetManager abstractions
│
├── features/               # Feature engineering modules
│   ├── __init__.py
│   ├── elo_features.py     # EloFeatureBuilder (Elo diff, predictions)
│   ├── team_form.py        # Rolling EPA, success rates, team form
│   ├── weather.py          # Wind, temp, precipitation processing
│   ├── contextual.py       # Rest days, travel distance, neutral site
│   ├── market_anchors.py   # Opening line, devigging, market probability
│   ├── validation.py       # Feature validation, leakage detection
│   └── __init__.py
│
├── models/                 # ML model training and prediction
│   ├── __init__.py
│   ├── train_wp.py         # WinProbabilityModel (LogisticRegression)
│   ├── train_ats.py        # ATSModel (XGBoost margin regression)
│   ├── train_ou.py         # OUModel (XGBoost total regression)
│   ├── prediction_pipeline.py  # NFLPredictionPipeline, fair lines, edges
│   ├── calibrate.py        # ProbabilityCalibrator (Isotonic Regression)
│   ├── evaluation.py       # ModelEvaluationFramework, metrics
│   ├── baseline_elo.py     # Baseline Elo-only model for comparison
│   └── utils.py            # WalkForwardValidator, ModelManager, ModelMetadata
│
├── ratings/                # Elo rating system
│   ├── __init__.py
│   └── elo.py              # EloRatingSystem (K-factors, margin multiplier)
│
├── scripts/                # Standalone CLI scripts for data/model operations
│   ├── ingest_games.py          # Fetch games from nfl_data_py
│   ├── ingest_odds.py           # Fetch odds snapshot from TheOddsAPI
│   ├── ingest_weather.py        # Fetch weather from Meteostat
│   ├── build_elo.py             # Incremental Elo rating updates
│   ├── build_team_form.py       # Calculate rolling form metrics
│   ├── build_contextual.py      # Add venue/context features
│   ├── build_weather.py         # Process weather into features
│   ├── build_market_anchors.py  # Add market-derived features
│   ├── build_features.py        # Merge all features into Gold layer
│   ├── data_qa.py               # Data quality validation
│   ├── validate_features.py     # Check for data leakage
│   ├── validate_historical_data.py  # Prepare historical data for backtest
│   ├── run_backtest.py          # Execute walk-forward validation
│   ├── health_check.py          # System health checks
│   ├── friday_data_update.py    # Weekly Friday data ingestion
│   ├── friday_predictions_run.py # Weekly prediction generation
│   ├── generate_current_week_predictions.py  # Ad-hoc predictions
│   ├── cleanup_data.py          # Data directory cleanup
│   └── test_*.py / demo_*.py / validate_*.py  # Testing scripts
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── logging_config.py        # get_logger() function, JSON logging
│   ├── date_utils.py            # get_current_nfl_season, get_current_nfl_week
│   ├── probability_utils.py     # Odds conversions, probability math
│   ├── validation.py            # Input validation helpers
│   ├── exceptions.py            # Custom exception classes
│   ├── team_data.py             # Team info (name, city, conference, division)
│   ├── game_utils.py            # Game ID generation, game parsing
│   ├── game_id_utils.py         # Game ID formatting/parsing
│   ├── ingestion_args.py        # CLI argument parsing for scripts
│   ├── similar_games.py         # SimilarGameCriteria, similar game finding
│   ├── betting_utils.py         # Edge calculation, Kelly criterion
│   ├── kelly_criterion.py       # Kelly fraction calculation
│   ├── unit_sizing.py           # Position sizing strategies
│   ├── bet_recommender.py       # BetRecommender, recommendation logic
│   ├── bet_selector.py          # Bet filtering/ranking
│   ├── bankroll_manager.py      # Bankroll tracking
│   ├── alert_manager.py         # Slack/email notifications
│   ├── api_metrics_bridge.py    # Metrics for API responses
│   └── api_recommendation_bridge.py  # Recommendation generation for API
│
├── web/                    # Web UI (frontend)
│   ├── static/
│   │   ├── css/
│   │   │   ├── tailwind.css      # Tailwind input CSS
│   │   │   ├── tailwind-compiled.css  # Compiled CSS (auto-generated)
│   │   │   └── custom.css        # Custom CSS overrides
│   │   └── js/
│   │       ├── common.js         # Shared JS utilities (API calls, formatting)
│   │       ├── games.js          # Games list page functionality
│   │       ├── game-detail.js    # Game detail page functionality
│   │       ├── backtest.js       # Backtest analytics page
│   │       ├── calibration.js    # Model calibration charts
│   │       └── recommendations.js # Betting recommendations UI
│   └── templates/
│       ├── base.html             # Base template (header, nav, footer)
│       ├── games.html            # Games list page
│       ├── game_detail.html      # Game detail page with similar games
│       ├── backtest.html         # Backtest performance dashboard
│       ├── calibration.html      # Model calibration curves
│       └── recommendations.html  # Weekly recommendations page
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures and configuration
│   └── test_runner.py      # Test discovery and execution
│
├── outputs/                # Generated artifacts (not versioned)
│   └── backtest/           # Backtest reports and CSV exports
│
├── artifacts/              # Model artifacts (not versioned)
│   └── models/             # Serialized models (wp.joblib, ats.joblib, etc.)
│
├── logs/                   # Log files (not versioned)
│   └── nfl-predict.log     # JSON structured logs
│
├── deployment/             # Deployment configuration
│   ├── gunicorn.conf.py    # Gunicorn server configuration
│   ├── uvicorn.conf.py     # Uvicorn ASGI config
│   ├── run_production.py   # Production run script
│   ├── setup_scheduling.py # Cron job setup for Friday runs
│   ├── security.py         # Security headers, secret validation
│   └── static_config.py    # Static deployment configuration
│
├── docs/                   # Documentation
│   └── (various markdown files)
│
├── .github/                # GitHub Actions CI/CD
│   └── workflows/          # Automated testing and deployment
│
├── Makefile                # Command definitions (snapshot, backtest, predict, serve)
├── pyproject.toml          # Python package configuration, dependencies
├── conf/config.yaml        # Configuration file
├── .env.example            # Template for .env secrets file
├── README.md               # Project overview
├── PRD.md                  # Product Requirements Document
└── TODO.md                 # Implementation progress tracking
```

## Directory Purposes

**api/:**
- Purpose: FastAPI REST API server and data access layer
- Contains: Endpoints for games, predictions, recommendations, analytics, web UI rendering
- Key files: `main.py` (1143 lines, all endpoints), `services.py` (data queries), `schemas.py` (DTOs)
- Depends on: models, data, utils, backtest modules

**backtest/:**
- Purpose: Historical validation system with no look-ahead bias
- Contains: Walk-forward orchestration, metric calculation, HTML/CSV reporting
- Key files: `walkforward.py` (orchestrator), `metrics.py` (computation), `reporting.py` (outputs)
- Depends on: models, data, utils

**conf/:**
- Purpose: Centralized configuration management
- Contains: YAML config file, Pydantic Settings class with env var merging
- Key files: `config.yaml` (all settings), `settings.py` (validation and access)
- Depends on: None (foundational)

**data/:**
- Purpose: Data storage abstractions and layer management
- Contains: DuckDB and Parquet utilities, schema definitions, save/load functions
- Key files: `storage.py` (DuckDBConnection, ParquetManager), `schemas.py` (table schemas)
- Depends on: conf, utils

**features/:**
- Purpose: Domain-specific feature engineering pipelines
- Contains: Elo, weather, form, context, market features; validation
- Key files: `elo_features.py`, `team_form.py`, `weather.py`, `market_anchors.py`
- Depends on: data, ratings, utils, conf

**models/:**
- Purpose: ML model training, calibration, and unified prediction
- Contains: WP, ATS, O/U models, prediction pipeline, evaluation
- Key files: `train_wp.py`, `train_ats.py`, `train_ou.py`, `prediction_pipeline.py`
- Depends on: features, data, utils, calibrate

**ratings/:**
- Purpose: Elo rating system maintenance
- Contains: Elo algorithm with margin-of-victory adjustment, dynamic K factors
- Key files: `elo.py`
- Depends on: utils

**scripts/:**
- Purpose: Standalone CLI tools for data and model operations
- Contains: 40+ scripts for ingestion, feature building, validation, testing
- Key files: `ingest_*.py` (external data), `build_*.py` (features), `run_backtest.py`
- Depends on: All other modules

**utils/:**
- Purpose: Shared utilities and helpers
- Contains: Logging, date utilities, validation, team data, betting math
- Key files: `logging_config.py` (logging), `date_utils.py`, `probability_utils.py`
- Depends on: conf (for settings)

**web/:**
- Purpose: HTML/CSS/JavaScript frontend
- Contains: Jinja2 templates, Tailwind CSS, vanilla JS for interactivity
- Key files: `base.html` (layout), `games.html`, `game_detail.html`, `*.js` (frontend logic)
- Depends on: api (via REST calls)

**tests/:**
- Purpose: Test suite
- Contains: Pytest fixtures and test discovery
- Key files: `conftest.py`, `test_runner.py`
- Depends on: All modules

**deployment/:**
- Purpose: Production deployment configuration
- Contains: Uvicorn/Gunicorn config, security, scheduling setup
- Key files: `uvicorn.conf.py`, `run_production.py`, `setup_scheduling.py`
- Depends on: api, conf

## Key File Locations

**Entry Points:**
- `api/main.py`: REST API server (1143 lines)
- `Makefile`: Command entry points (snapshot, backtest, predict, serve)
- `scripts/friday_predictions_run.py`: Weekly production run
- `scripts/run_backtest.py`: Backtest execution

**Configuration:**
- `conf/config.yaml`: All settings (data paths, feature params, model hyperparams)
- `conf/settings.py`: Pydantic Settings validation and access
- `deployment/static_config.py`: Production environment configuration
- `.env`: Secrets (ODDS_API_KEY, SECRET_KEY) - not versioned

**Core Logic:**
- `models/prediction_pipeline.py`: Unified prediction pipeline (NFLPredictionPipeline class, 836 lines)
- `data/storage.py`: DuckDB and Parquet abstractions (848 lines)
- `backtest/walkforward.py`: Walk-forward validation orchestrator
- `ratings/elo.py`: Elo rating system

**Feature Engineering:**
- `features/elo_features.py`: Elo-based features
- `features/team_form.py`: Rolling form metrics
- `features/weather.py`: Weather processing
- `features/market_anchors.py`: Market-derived features
- `features/contextual.py`: Venue and context features

**API Responses:**
- `api/schemas.py`: Pydantic models for all request/response types
- `api/services.py`: DataService and BacktestService abstractions
- `api/middleware.py`: Middleware for CORS, logging, rate limiting

**Web UI:**
- `web/templates/base.html`: Layout template (Jinja2)
- `web/templates/games.html`: Games list page
- `web/templates/game_detail.html`: Single game detail
- `web/static/js/common.js`: Shared JS utilities

**Testing & Validation:**
- `scripts/validate_features.py`: Feature leakage detection
- `scripts/data_qa.py`: Data quality checks
- `backtest/metrics.py`: Metric definitions and calculation

## Naming Conventions

**Files:**
- `train_*.py`: Model training modules (train_wp.py, train_ats.py, train_ou.py)
- `build_*.py`: Feature building scripts (build_elo.py, build_team_form.py)
- `ingest_*.py`: Data ingestion scripts (ingest_games.py, ingest_odds.py)
- `test_*.py` / `validate_*.py`: Testing and validation scripts
- `*_bridge.py`: Bridge/adapter modules (api_recommendation_bridge.py, api_metrics_bridge.py)

**Directories:**
- `data/{layer}/`: Data storage by layer (bronze, silver, gold)
- `data/silver/season={year}/`: Silver layer partitioned by season
- `data/silver/snapshot_ts={timestamp}/`: Odds snapshot partitioned by timestamp
- `artifacts/models/`: Model artifact storage
- `outputs/backtest/`: Backtest result artifacts

**Classes:**
- Model classes: `WinProbabilityModel`, `ATSModel`, `OUModel`
- Feature builders: `EloFeatureBuilder`, `WeatherFeatureBuilder`, etc.
- Data abstractions: `DuckDBConnection`, `ParquetManager`
- Pipeline: `NFLPredictionPipeline`
- Backtesting: `WalkForwardBacktester`, `BacktestMetricsCalculator`, `BacktestReporter`

**Functions:**
- Utilities: snake_case (e.g., `get_logger`, `get_current_nfl_season`)
- Feature functions: `get_*_features_for_game` (e.g., `get_elo_features_for_game`)

## Where to Add New Code

**New Feature:**
- Primary code: Create file in `features/` (e.g., `features/my_feature.py`)
- Integration: Add feature builder call to `scripts/build_features.py`
- Tests: Add test script `scripts/test_my_feature.py`
- Config: Add feature params to `conf/config.yaml` under `models.features` section

**New Model Type (e.g., draw prediction):**
- Implementation: Create `models/train_draw.py` following `train_wp.py` pattern
- Pipeline integration: Add to `NFLPredictionPipeline` in `models/prediction_pipeline.py`
- Training script: Add `scripts/build_draw_model.py`
- API endpoint: Add to `api/main.py` (e.g., `/predictions/draw`)
- Config: Add `draw` section to `conf/config.yaml` under `models`

**New API Endpoint:**
- Implementation: Add function to `api/main.py` with `@app.get()` or `@app.post()` decorator
- Schema: Define request/response DTOs in `api/schemas.py`
- Service: Add method to appropriate service class in `api/services.py` if needed
- Documentation: Add docstring with description and parameter documentation

**Utilities/Helpers:**
- Shared helpers: Add to appropriate file in `utils/` (e.g., `utils/betting_utils.py`)
- Domain-specific: Create new file if warranted (e.g., `utils/my_helper.py`)
- Tests: Add test script in `scripts/test_my_helper.py`

**Data Ingestion Source:**
- Implementation: Create `scripts/ingest_my_source.py` following pattern of existing ingest scripts
- Schema: Define table schema in `data/schemas.py`
- Integration: Add call to `Makefile` snapshot target
- Storage: Data saved to `data/bronze/` initially

## Special Directories

**data/bronze/:**
- Purpose: Raw ingested data snapshots
- Generated: Yes (auto-created on ingestion)
- Committed: No (.gitignore excludes data/)
- Contents: JSON/CSV snapshots from external APIs, not normalized

**data/silver/:**
- Purpose: Cleaned, standardized data tables
- Generated: Yes (by transformation scripts)
- Committed: No (.gitignore excludes data/)
- Partitioning: By season and snapshot_ts for query efficiency
- Contents: games.parquet, odds_snapshot.parquet, weather.parquet, team_stats.parquet

**data/gold/:**
- Purpose: Feature matrices ready for model input
- Generated: Yes (by build_features.py)
- Committed: No (.gitignore excludes data/)
- Contents: wp.parquet, ats.parquet, ou.parquet (feature matrices per target)

**artifacts/models/:**
- Purpose: Trained model serialization
- Generated: Yes (by model training scripts)
- Committed: No (.gitignore excludes artifacts/)
- Contents: wp_model.joblib, ats_model.joblib, ou_model.joblib, calibrators, scalers

**outputs/backtest/:**
- Purpose: Backtest reports and metrics
- Generated: Yes (by backtest runner)
- Committed: No (.gitignore excludes outputs/)
- Contents: backtest_report.html, metrics_summary.json, betting_simulation.csv

**logs/:**
- Purpose: Application logs
- Generated: Yes (at runtime)
- Committed: No (.gitignore excludes logs/)
- Contents: nfl-predict.log (JSON structured logs)

---

*Structure analysis: 2026-03-18*
