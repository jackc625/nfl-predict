# NFL Prediction System

A full-stack NFL prediction engine that generates pre-game Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions for every weekly matchup. The system is built around a three-layer data pipeline, walk-forward temporal validation with strict no-leakage enforcement, and a market-blended prediction strategy evaluated through multi-season backtesting with betting simulation.

## Overview

This project ingests game data, odds, weather, and play-by-play statistics from multiple sources, engineers 55-70 features per game, trains three specialized ML models (one per prediction target), blends model outputs with market prices, and serves predictions through a FastAPI web dashboard with interactive backtest reporting.

The entire pipeline is designed around one constraint: **no future data leakage**. Every feature builder enforces a time-fence cutoff, a LeakageGate validator hard-fails on violations, and all evaluation uses expanding-window walk-forward splits -- never random cross-validation.

## Key Features

### Prediction Targets
- **Win Probability (WP)** -- Calibrated home/away win probabilities using logistic regression with isotonic calibration
- **Against the Spread (ATS)** -- Point margin predictions and cover probabilities via XGBoost regression with residual distribution conversion
- **Over/Under (O/U)** -- Total points predictions with weather impact modeling via XGBoost regression

### Data Pipeline
- **Three-layer architecture** (Bronze/Silver/Gold) with append-only raw snapshots, validated upsert tables, and model-ready feature matrices
- **Pydantic schema validation** at every layer boundary with hard-fail on violations
- **Canonical team abbreviation mapping** that resolves all historical variations (JAC/JAX, STL/LA, SD/LAC, OAK/LV, WSH/WAS) with hard-fail on unknowns
- **DuckDB + Parquet dual storage** for SQL query flexibility and columnar read performance

### Feature Engineering
- **Elo ratings** with margin-of-victory K-factor adjustment, Glicko-style uncertainty tracking, divisional HFA reduction, and season carryover regression
- **Team form** from play-by-play data: rolling EPA/play (overall, pass, rush), success rates, neutral pass rate, red zone efficiency, 3rd-down conversion
- **Opponent-adjusted EPA** that accounts for strength of schedule
- **Market anchors** with devigged moneyline probabilities, opening/snapshot line movement tracking
- **Weather modeling** with wind impact scoring, cold wind chill, precipitation indicators (outdoor venues only)
- **Contextual signals**: rest days, travel distance, surface mismatch, divisional indicator, season progress
- **QB tracking**: team-level CPOE (completion percentage over expected)
- **Expanding-window normalization** with prior-season bootstrap and per-season reset to prevent cross-season contamination

### Temporal Safety
- **FeatureBuilder Protocol** -- every builder accepts an `as_of_datetime` time-fence parameter; data after the cutoff is inaccessible
- **LeakageGate** -- two-stage validator that (1) checks each builder's output against the time fence and (2) scans the combined feature matrix for leakage keywords (`closing`, `final_score`, `outcome`, etc.) with hard-fail on detection
- **Elo ordering check** -- verifies chronological update order within each season
- **Walk-forward-only evaluation** -- expanding-window splits (train on years < Y, evaluate on Y) across 2021-2024 holdout seasons

### Market Blending
- **Log-odds blending for WP** -- model and market probabilities combined in logit space to respect the non-linearity of probabilities
- **Linear interpolation for ATS/O/U** -- point spreads and totals blended directly in linear space
- **Walk-forward weight tuning** on 2010-2017 pre-backtest data with temporal isolation enforced
- **Per-target edge threshold calibration** -- diagnostic flagging of games where model-market disagreement exceeds a threshold

### Backtesting
- **Walk-forward engine** across 2021-2024 with per-season model retraining and rolling HP-validation windows
- **Brier score decomposition** (Murphy 1973): reliability, resolution, uncertainty
- **CLV computation** -- probability-based and line-based closing line value for all three targets
- **Betting simulation** -- parallel flat-stake and quarter-Kelly strategies with half-point slippage, standard vig, equity curve tracking, max drawdown measurement
- **Interactive HTML reports** with Plotly charts (calibration reliability, cumulative CLV, season heatmap, equity curves) and CSV/JSON export

### Web Dashboard
- **FastAPI monolith** serving HTML pages and JSON exports from a read-only DuckDB cache
- **HTMX-powered navigation** -- week switching, sort controls, and season filtering update the page without full reloads
- **Game detail drill-down** with prediction vs. market comparison, feature importance visualization, team context (Elo, last 5, H2H), and result overlay for completed games
- **Performance dashboard** with season-level metrics and summary cards
- **Backtest visualization** with four-chart layout (calibration, CLV, heatmap, equity curves)
- **CSV/JSON export** for predictions and backtest data
- **Mobile-responsive** with Tailwind CSS v4 and touch-optimized controls

## Technical Highlights

**Temporal integrity as a first-class concern.** The `FeatureBuilder` protocol, `LeakageGate` validator, and walk-forward splitter form a three-layer defense against future data contamination. This isn't just a convention -- it's enforced at build time with hard-fail exceptions.

**Feature compression without information loss.** Raw weather data (38 columns) compresses to 4-6 dominant features through categorical bucketing and impact scoring. Market data (40+ odds columns) reduces to ~5 features via model-based feature selection. This keeps dimensionality manageable relative to training set size.

**Principled model-market integration.** Rather than treating model predictions and market prices as competing signals, the blending system combines them in mathematically appropriate spaces (logit for probabilities, linear for points) with weights tuned on temporally isolated pre-backtest data.

**Decoupled cache architecture.** The web layer reads exclusively from a denormalized DuckDB cache -- no model imports, no training code, no feature pipeline. This means the API starts in milliseconds and cannot be broken by changes to the ML pipeline.

**Production scheduling.** A GitHub Actions workflow runs the full pipeline on Fridays at 5 PM ET (data update) and 6 PM ET (predictions), matching the NFL's Friday odds snapshot timing. Cron, Docker, and Windows Task Scheduler configurations are also provided.

## Architecture

```
External Sources ──> Ingestion (ETL) ──> Data Lake (Parquet/DuckDB) ──> Feature Build ──> Models ──> Predictions
                                                                                                        │
                                                                                                        v
                                                                              DuckDB Cache <── Populate Script
                                                                                   │
                                                                                   v
                                                                        FastAPI ──> Jinja2/HTMX Web UI
```

### Data Layer

| Layer | Location | Format | Semantics |
|-------|----------|--------|-----------|
| Bronze | `data/bronze/` | Parquet | Append-only raw snapshots, timestamped |
| Silver | `data/silver/` | Parquet | Cleaned, validated tables (latest-wins upsert by `game_id`) |
| Gold | `data/gold/` | Parquet | Feature matrices per target (`features_wp`, `features_ats`, `features_ou`) |
| Cache | `data/web_cache.duckdb` | DuckDB | Denormalized tables for web serving |

### ML Models

| Target | Algorithm | Output | Calibration |
|--------|-----------|--------|-------------|
| Win Probability | Logistic Regression | P(home win) [0, 1] | Isotonic regression on HP-validation set |
| Against the Spread | XGBoost Regressor | Predicted margin, converted to cover probability | Residual distribution converter |
| Over/Under | XGBoost Regressor | Predicted total, converted to over/under probability | Total distribution converter |

Feature selection uses `SelectFromModel` on the training window, locked for all holdout evaluations. Hyperparameter tuning uses `RandomizedSearchCV` with temporal CV folds (never random).

### Web Layer

| Component | Technology | Role |
|-----------|------------|------|
| Server | FastAPI + Uvicorn | HTML pages, HTMX fragments, JSON/CSV export endpoints |
| Templates | Jinja2 with jinja2-fragments | Full pages and partial block responses for HTMX |
| Interactivity | HTMX 2.0 | Fragment swaps for week/sort/season changes without page reloads |
| Styling | Tailwind CSS v4 (standalone compiler) | Responsive layout, NFL-branded color tokens |
| Charts | Plotly | Feature importance bars, calibration plots, equity curves |
| Data Source | DuckDB (read-only) | Fresh connection per request, no model imports in API layer |

### Middleware Stack

Security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, CSP), CORS with configurable origins, request logging with sensitive field redaction, performance monitoring with slow-request flagging, and structured error responses with request ID tracking.

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12+ |
| Data | pandas, DuckDB, Parquet (PyArrow), NumPy |
| ML | scikit-learn, XGBoost, joblib |
| API | FastAPI, Uvicorn, Pydantic, Pydantic Settings |
| Frontend | Jinja2, HTMX 2.0, Tailwind CSS v4, Plotly |
| Data Sources | nflreadpy (games/PBP), The Odds API (odds), Open-Meteo (weather) |
| HTTP | httpx, tenacity (retry logic) |
| Config | YAML (conf/config.yaml), .env, Pydantic Settings |
| Logging | structlog (JSON + text modes) |
| Package Manager | uv |
| Linting | Ruff (13 rule categories), Pyright (standard mode) |
| Testing | pytest, pytest-cov, pytest-asyncio, Hypothesis |
| Deployment | Docker, Gunicorn, Nginx, GitHub Actions |

## Feature Deep Dive

### Elo Rating System

The `ratings/elo.py` module implements an Elo system inspired by FiveThirtyEight's NFL methodology. Ratings carry forward between seasons with regression toward the mean. Each game update uses a margin-of-victory-adjusted K-factor. Home-field advantage is learned per season with a divisional reduction factor (0.54x) based on research showing divisional games exhibit ~46% less HFA. Optional Glicko-style uncertainty tracking widens confidence intervals for teams with fewer recent games.

### Walk-Forward Backtest Engine

The `backtest/engine.py` orchestrates full walk-forward evaluation across holdout seasons (2021-2024). For each holdout year Y:

1. A fresh model is trained on all data from seasons < Y-1
2. Hyperparameters are tuned on the HP-validation year (Y-1)
3. Predictions are generated for year Y
4. CLV is computed against closing lines
5. Metrics are aggregated (Brier score, ECE, MAE, RMSE, accuracy)

The `BettingSimulator` runs two parallel strategies (flat-stake and quarter-Kelly) with half-point slippage on ATS/O/U and standard -110 vig, producing equity curves, ROI, and max drawdown.

### Prediction Pipeline

`models/prediction_pipeline.py` orchestrates end-to-end prediction generation. Given a target week, it:

1. Loads the latest model artifacts (versioned with timestamps)
2. Builds features through the FeatureBuilder pipeline with time-fence enforcement
3. Generates raw predictions
4. Applies calibration (WP only)
5. Optionally blends with market prices
6. Exports to Parquet, JSON, and CSV

### Web Dashboard Pages

- **This Week's Predictions** (`/`) -- Game cards with WP, ATS, O/U predictions, confidence badges, edge indicators, and HTMX-powered week/sort controls
- **Performance** (`/performance`) -- Summary cards (total games, overall CLV, WP accuracy, Brier score) and per-season metrics table with season selector
- **Backtest** (`/backtest`) -- Four-chart Plotly layout: calibration reliability, cumulative CLV, season comparison heatmap, betting equity curves
- **Game Detail** (`/games/{id}`) -- Prediction vs. market comparison table, top-10 feature importance bar chart, team context (Elo ratings, last 5 results, head-to-head record), venue/weather info, result overlay for completed games

## Data Model

### External Sources

| Source | Data | Usage |
|--------|------|-------|
| nflreadpy | Game schedules, scores, play-by-play | Game outcomes, EPA, success rates, CPOE |
| The Odds API | Pre-game odds (spreads, totals, moneylines) | Market anchors, devigged probabilities, CLV computation |
| Open-Meteo | Historical hourly weather | Wind, temperature, precipitation for outdoor venues |
| Static JSON | Stadium coordinates, surface type, roof type | Venue features, travel distance, surface mismatch |

### Feature Matrix (Gold Layer)

Each row represents one game. Feature groups:

| Group | Features | Source |
|-------|----------|--------|
| Elo | Home/away Elo, diff, win probability, uncertainty, HFA | Elo rating system |
| Team Form | EPA/play (overall, pass, rush), success rates, neutral pace, red zone, 3rd down | Play-by-play |
| Market | Opening/snapshot spreads and totals, line movement, devigged ML probabilities | Odds API |
| Weather | Wind mph, wind impact score, temp, cold indicator, precipitation | Open-Meteo |
| Contextual | Rest days, travel zones, surface mismatch, divisional, season progress | Game metadata |
| QB | Team CPOE (completion % over expected) | Play-by-play |
| Opponent-Adjusted | EPA adjusted for opponent defensive strength | Play-by-play |

### Model Artifacts

Versioned artifacts are stored in `artifacts/` with the structure:
```
artifacts/
  wp_20260319_180000/
    model.pkl              # Serialized model (joblib)
    metadata.json          # Training metadata, metrics, parameters
    feature_list.json      # Ordered feature names
    calibrator.pkl         # Isotonic calibrator (WP only)
  latest.json              # Manifest pointing to production models
```

### DuckDB Web Cache

Nine denormalized tables optimized for read-only web access:

- `predictions` -- Game predictions with market data, edges, and blended values
- `feature_importances` -- Per-game, per-target feature attribution
- `backtest_metrics` -- Aggregated performance by (season, target, metric)
- `backtest_predictions` -- Individual backtest predictions with CLV
- `simulation_results` -- Betting simulation metrics by strategy
- `equity_curve` -- Bankroll progression by strategy
- `chart_cache` -- Pre-rendered Plotly HTML divs
- `game_context` -- Team Elo, recent results, H2H (JSON columns)
- `cache_meta` -- Last updated timestamp, prediction counts

## Project Structure

```
nfl-predict/
  api/                     # FastAPI application
    main.py                #   App factory, router registration, lifespan
    routes/                #   Page, fragment, export, and health endpoints
    middleware.py           #   Security headers, CORS, logging, performance
    services.py            #   DuckDB read-only data access layer
    cache.py               #   Cache population and schema management
    charts.py              #   Plotly chart generation
    schemas.py             #   Pydantic response models
    exceptions.py          #   Structured error handling with request tracking
  backtest/                # Walk-forward backtesting
    engine.py              #   Core backtest orchestrator
    simulation.py          #   Flat-stake and Kelly betting simulator
    metrics.py             #   Brier decomposition, ECE, accuracy, MAE/RMSE
    report.py              #   Jinja2 + Plotly HTML report generation
    tune.py                #   Blend weight tuning on pre-backtest data
  conf/                    # Configuration
    config.yaml            #   260+ line central config (paths, models, thresholds)
    settings.py            #   Pydantic settings models
  data/                    # Data storage (not committed)
    bronze/                #   Raw snapshots (append-only, timestamped)
    silver/                #   Cleaned tables (latest-wins upsert)
    gold/                  #   Feature matrices per target
    storage.py             #   DuckDB + Parquet I/O with dual-save
    schemas.py             #   Pydantic data contracts
    quality_gates.py       #   Schema validation at layer boundaries
  features/                # Feature engineering
    protocol.py            #   FeatureBuilder protocol (time-fence contract)
    validation.py          #   LeakageGate (two-stage hard-fail validator)
    elo_features.py        #   Elo rating features
    team_form.py           #   Rolling EPA, success rates from PBP
    market_anchors.py      #   Devigged odds, line movement
    weather.py             #   Wind, temp, precipitation (outdoor only)
    contextual.py          #   Rest, travel, surface, divisional
    opponent_adj.py        #   Opponent-adjusted EPA
    normalization.py       #   Expanding-window with prior-season bootstrap
  models/                  # ML training and prediction
    trainers/              #   WP (LogReg), ATS (XGBoost), O/U (XGBoost)
    calibrate.py           #   Isotonic + Platt calibration
    blending.py            #   Log-odds WP blending, linear ATS/O/U blending
    clv.py                 #   Closing line value computation
    temporal.py            #   Walk-forward splitter, temporal CV
    prediction_pipeline.py #   End-to-end prediction orchestration
    artifacts.py           #   Versioned model storage
    evaluation.py          #   Metric computation and comparison
  ratings/                 # Elo rating system
    elo.py                 #   MoV-adjusted Elo with Glicko uncertainty
  scripts/                 # Operational scripts (~37 scripts)
    ingest_*.py            #   Data ingestion (games, odds, weather)
    build_*.py             #   Feature building (Elo, form, weather, etc.)
    validate_*.py          #   Data, feature, model, prediction validation
    friday_*.py            #   Friday production run orchestration
    health_check.py        #   System health monitoring
  tests/                   # Test suite (~55 files)
    unit/                  #   Core logic tests
    integration/           #   Pipeline and report tests
    api/                   #   Endpoint tests
    ui/                    #   HTML snapshot tests
  utils/                   # Shared utilities (19 modules)
    team_data.py           #   Canonical abbreviations, hard-fail mapping
    kelly_criterion.py     #   Kelly fraction sizing
    bet_recommender.py     #   Bet recommendation engine
    similar_games.py       #   Historical similarity finder
  web/                     # Web frontend
    templates/             #   Jinja2 templates (base, pages, components)
    static/css/            #   Tailwind CSS v4 (compiled) + custom theme
    static/js/             #   HTMX helpers, chart rendering, game detail
  deployment/              # Production deployment
    Dockerfile             #   Multi-stage build
    docker-compose.yml     #   API, Nginx, Redis, optional monitoring stack
    nginx.conf             #   Reverse proxy, static serving, rate limiting
    gunicorn.conf.py       #   Production WSGI server config
    crontab.txt            #   Friday pipeline scheduling
  .github/workflows/       # CI/CD
    friday-production.yml  #   Automated Friday data + prediction pipeline
```

## Production Considerations

### Scheduling

The prediction pipeline is timed around the NFL's Friday odds snapshot:
- **5:00 PM ET Friday** -- Data ingestion (games, odds, weather) with quality validation
- **6:00 PM ET Friday** -- Feature build, model predictions, cache population, web update
- Supported via GitHub Actions (scheduled + manual dispatch), cron, Docker scheduler, or Windows Task Scheduler

### Monitoring

- Health check endpoint (`/health`) reports cache status and last-updated time
- Operational monitoring scripts check data freshness, model health, and prediction pipeline status
- Request logging middleware with unique request IDs, sensitive field redaction, and slow-request flagging
- Structured JSON logging via structlog

### Error Handling

- Custom exception hierarchy (`DataNotFoundError`, `ValidationError`, `ModelUnavailableError`, `DataStaleError`, `RateLimitError`) with standardized JSON error responses
- Input validation for season/week ranges and team abbreviations
- Quality gates at every data layer boundary prevent silent corruption

### Deployment

Docker Compose configuration provides:
- FastAPI application container with health checks
- Nginx reverse proxy with SSL termination and static file serving
- Optional Redis for caching and rate limiting
- Optional Prometheus + Grafana monitoring stack
- Optional ELK logging stack (Elasticsearch, Logstash, Kibana)

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone <repo-url>
cd nfl-predict
uv sync
cp .env.example .env
# Add your API keys (The Odds API key required for live odds)
```

### Core Commands

```bash
make snapshot          # Ingest data, build features, produce Gold tables
make train             # Train WP, ATS, O/U models
make backtest          # Walk-forward backtest across 2021-2024
make backtest-blend    # Blended backtest with market integration
make predict           # Generate predictions for current week
make serve             # Start web dashboard at localhost:8000
make test              # Run full test suite (unit, integration, API, UI)
```

## Extensibility

The system is designed for straightforward extension:

- **New feature builders** implement the `FeatureBuilder` protocol with the `as_of_datetime` time-fence contract and plug into the feature matrix pipeline
- **New prediction targets** follow the `BaseTrainer` abstract class pattern with walk-forward evaluation built in
- **New data sources** follow the Bronze/Silver ingest pattern with Pydantic schema validation at boundaries
- **Configuration-driven** -- model hyperparameters, feature toggles, edge thresholds, and API settings are all managed through `conf/config.yaml` and environment variables

## License

MIT
