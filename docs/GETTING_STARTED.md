# Getting Started - NFL Prediction System

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This guide will walk you through setting up and running the complete NFL Prediction System pipeline from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Complete Setup (30 minutes)](#complete-setup-30-minutes)
4. [Running the Full Pipeline](#running-the-full-pipeline)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements
- **Python 3.9+**
- **Git** (for version control)
- **4GB+ RAM** (for model training)
- **2GB+ disk space** (for data and artifacts)

### Check Your Environment
```bash
# Verify Python version
python --version

# Check if you're in the project directory
pwd
ls -la  # Should see: Makefile, api/, models/, etc.
```

---

## Quick Start (5 minutes)

**Goal**: Get the system running with existing data to see the web interface.

### Step 1: Check System Status
```bash
# Run basic health check
python scripts/health_check.py --mode basic
```
**Expected**: Database connection should be healthy

### Step 2: Start the API Server
```bash
# Start the web server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: View the Interface
Open your browser and go to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**🎉 Success**: You should see a beautiful NFL prediction interface!

---

## Complete Setup (30 minutes)

**Goal**: Set up the complete system with fresh data and working predictions.

### Step 1: System Health Check
```bash
# Comprehensive system check
python scripts/health_check.py --mode comprehensive

# Check what data we currently have
ls -la data/bronze/ data/silver/ data/gold/
```

### Step 2: Data Quality Assessment
```bash
# Run data quality checks
python scripts/data_qa.py --current --save

# Check results
cat outputs/reports/data_qa_report_*.json | head -20
```

### Step 3: Fix Known Issues (if needed)

**Schema Issue Fix** (if you see "game_date not found" errors):
```bash
# Check current database schema
python -c "import duckdb; conn = duckdb.connect('data/nfl_predictions.duckdb'); print('Games schema:'); print(conn.execute('DESCRIBE games').fetchall()[:5])"
```

If the schema shows `kickoff_et` instead of `game_date`, the system will work but with some API warnings.

### Step 4: Test Core Components

**Test Data Pipeline**:
```bash
# Test data ingestion (should complete quickly with existing data)
# --current now ingests weeks 1 through current week of current season
python scripts/ingest_games.py --current
python scripts/ingest_odds.py --current
python scripts/ingest_weather.py --current  # Use --mock only for testing without API access
```

**Test Model System**:
```bash
# Check model availability
ls -la artifacts/models/

# Test model validation
python scripts/validate_models.py --models wp,ats,ou
```

---

## Standardized Command Interface

All data ingestion scripts now use **consistent, standardized arguments** for better usability:

### **Universal Arguments (All Scripts)**
```bash
# Season selection
--season SEASON                    # Single season (e.g., --season 2024)
--seasons SEASON [SEASON...]       # Multiple seasons (e.g., --seasons 2022 2023 2024)

# Week selection
--week WEEK                        # Single week (e.g., --week 5)
--weeks WEEK [WEEK...]             # Multiple weeks (e.g., --weeks 1 2 3)
--current                          # Weeks 1 through current week (e.g., weeks 1-3 if current is week 3)
--all                              # All weeks of specified season(s)
```

### **Key Improvement: Fixed `--current` Behavior**
**Before**: `--current` only got the single current week (e.g., just Week 3)
**After**: `--current` gets **weeks 1 through current week** (e.g., Weeks 1, 2, 3)

This ensures you always have complete season data when using `--current`.

### **Script-Specific Arguments**
```bash
# ingest_odds.py only
--snapshot-time TIME               # Snapshot timing (ISO format)
--markets h2h spreads totals       # Which markets to fetch
--bookmakers BOOK [BOOK...]        # Specific bookmakers
--mock                             # Use mock data for testing

# ingest_weather.py only
--forecast-time TIME               # Forecast timing (ISO format)
--mock                             # Use mock data for testing
--outdoor-only                     # Only outdoor games

# ingest_games.py only
--no-results                       # Skip fetching results
--validate-only                    # Only validate existing data
```

### **Example Usage**
```bash
# Get all historical data
python scripts/ingest_games.py --seasons 2018 2019 2020 2021 2022 2023 2024
python scripts/ingest_weather.py --seasons 2018 2019 2020 2021 2022 2023 2024 --all

# Get current season data (weeks 1 through current)
python scripts/ingest_games.py --current
python scripts/ingest_weather.py --current

# Get specific weeks
python scripts/ingest_games.py --season 2024 --weeks 15 16 17
python scripts/ingest_weather.py --season 2024 --weeks 15 16 17

# Single week
python scripts/ingest_odds.py --season 2024 --week 17
```

---

## Pipeline Operations

The NFL prediction system has two distinct operational modes:

### **🏗️ First-Time Setup (Complete Historical Pipeline)**

**When to use**: Initial deployment, fresh installation, or rebuilding from scratch.

**Goal**: Build the complete system with 7 years of historical data (2018-2024) for robust backtesting and model training.

**⏱️ Time Required**: 2-4 hours (depending on system performance)

#### **Method A: Automated Setup (Recommended)**

```bash
# Check if make is available
make --version

# Run system setup and directory creation
make setup

# NOTE: Historical data ingestion must be done manually - see Method B below
```

#### **Method B: Manual Setup (Step-by-Step)**

**When to use**: If `make` is not available on your system (common on Windows) or you prefer manual control.

**Step 1: Environment Setup (5 minutes)**
```bash
# Create directory structure manually
mkdir -p data/{bronze,silver,gold}
mkdir -p outputs/{predictions,backtest,reports}
mkdir -p artifacts/{models,features}
mkdir -p logs
mkdir -p web/{templates,static}

# Install dependencies (now uses uv; black/isort/flake8 replaced by Ruff in Phase 1)
pip install -e ".[all]"

# Setup configuration
cp .env.example .env
# Edit .env file with your API keys

# Verify system health
python scripts/health_check.py --mode basic
```

**Step 2: Historical Data Ingestion (30-60 minutes)**

*Note: The system uses two different odds ingestion methods:*
- *`ingest_historical_odds.py` - Real historical data from nflreadpy (previously nfl_data_py, replaced in Phase 1) (for backtesting)*
- *`ingest_odds.py` - Live current odds from TheOddsAPI (for production predictions)*

```bash
# Ingest 7 years of NFL game data (2018-2024) - NEW: standardized arguments
python scripts/ingest_games.py --seasons 2018 2019 2020 2021 2022 2023 2024

# Ingest historical betting odds (real data from nflreadpy (previously nfl_data_py, replaced in Phase 1) for backtesting)
# This ingests 1,855 real consensus betting lines for all games 2018-2024
python scripts/ingest_historical_odds.py

# Ingest historical weather data (for feature engineering) - NEW: consistent arguments
# Note: Use --mock only for testing; production requires real weather data for accuracy
python scripts/ingest_weather.py --seasons 2018 2019 2020 2021 2022 2023 2024 --all

# OR ingest one season at a time (if preferred for progress monitoring):
# python scripts/ingest_weather.py --season 2018
# python scripts/ingest_weather.py --season 2019
# python scripts/ingest_weather.py --season 2020
# python scripts/ingest_weather.py --season 2021
# python scripts/ingest_weather.py --season 2022
# python scripts/ingest_weather.py --season 2023 
# python scripts/ingest_weather.py --season 2024

# Add current season (2025) data - NEW: --current gets weeks 1 through current week
python scripts/ingest_games.py --current      # Gets weeks 1-3 (not just week 3)
python scripts/ingest_odds.py --current       # Gets current week only (API limitation)
python scripts/ingest_weather.py --current    # Gets weeks 1-3 (weather supports multiple weeks)

# Comprehensive data quality validation
python scripts/data_qa.py --comprehensive --save
```

**Step 3: Feature Engineering (45-90 minutes)**
```bash
# Build Elo ratings from all historical data
python scripts/build_elo.py --all-seasons

# Build team form metrics (EPA, success rates)
python scripts/build_team_form.py --all-seasons

# Build contextual features (travel, rest, venue)
python scripts/build_contextual.py --all-seasons

# Build weather impact features
python scripts/build_weather.py --all-seasons

# Build market anchor features from odds
python scripts/build_market_anchors.py --all-seasons

# Create unified feature matrices for all targets (WP, ATS, O/U)
python scripts/build_features.py --all-data --targets wp,ats,ou --save --validate

# Validate features for data leakage
python scripts/validate_features.py --all-data --strict
```

**Step 4: Model Training (30-60 minutes)**
```bash
# Train Win Probability model (logistic regression)
python models/train_wp.py --full-training

# Train Against the Spread model (XGBoost)
python models/train_ats.py --full-training

# Train Over/Under model (XGBoost)
python models/train_ou.py --full-training

# Validate all trained models
python scripts/validate_models.py --models wp,ats,ou --required
```

**Step 5: Backtesting Validation (30-90 minutes)**
```bash
# Prepare historical data
python scripts/validate_historical_data.py --seasons 2018-2024

# Run walk-forward validation
python -c "from backtest.walkforward import WalkForwardBacktester; bt = WalkForwardBacktester(); results = bt.run_backtest(start_season=2018, end_season=2024); print('Backtest completed')"

# Generate metrics report
python -c "from backtest.reporting import BacktestReporter; reporter = BacktestReporter(); reporter.generate_full_report(start_season=2018, end_season=2024, output_dir='outputs/backtest')"

# Validate backtest reproducibility
python scripts/validate_backtest_reproducibility.py --seasons 2018-2024
```

**Step 6: Generate Initial Predictions**
```bash
# Validate model availability
python scripts/validate_models.py --models wp,ats,ou --required

# Generate predictions
python -c "from models.prediction_pipeline import NFLPredictionPipeline; pipeline = NFLPredictionPipeline(); results = pipeline.predict_current_week(); print(f'Generated {len(results)} predictions')"

# Generate bet recommendations
python -c "from models.bet_recommender import BetRecommendationEngine; engine = BetRecommendationEngine(); recs = engine.generate_weekly_recommendations(); print(f'Generated {len(recs)} recommendations')"

# Export prediction artifacts
python -c "from models.prediction_pipeline import PredictionExporter; exporter = PredictionExporter(); exporter.export_current_week_predictions(formats=['parquet', 'json', 'csv'])"

# Validate prediction outputs
python scripts/validate_predictions.py --current-week --check-completeness
```

**Step 7: Start Web Interface**
```bash
# Pre-flight checks
python scripts/health_check.py --pre-startup

# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

---

### **🔄 Weekly Operations (Production Cycle)**

**When to use**: Regular weekly operation after initial setup is complete.

**Goal**: Generate fresh predictions for current week using Friday 6 PM ET snapshot timing.

**⏱️ Time Required**: 15-30 minutes

#### **Method A: Automated Weekly Pipeline (Recommended)**

```bash
# Complete weekly update (Tuesday-Thursday)
make weekly-update

# Friday 5:00 PM ET: Data preparation
make friday-data-update

# Friday 6:00 PM ET: Prediction generation
make friday-predictions

# Start/restart web interface
make serve
```

#### **Method B: Manual Weekly Operations**

**When to use**: If `make` is not available or you want granular control over each step.

**Tuesday-Thursday: Data Preparation**
```bash
# Update current week games - NEW: gets all weeks 1 through current
python scripts/ingest_games.py --current

# Update weather forecasts (use real data for production) - NEW: gets all weeks 1 through current
python scripts/ingest_weather.py --current

# Data quality check
python scripts/data_qa.py --current --save

# Build current week features
python scripts/build_elo.py --current-week
python scripts/build_team_form.py --current-week
python scripts/build_contextual.py --current-week
python scripts/build_weather.py --current-week

# Validate features
python scripts/validate_features.py --current-week
```

**Friday 6:00 PM ET: Manual Production Run**
```bash
# Step 1: Capture odds snapshot at exactly 6:00 PM ET
python scripts/ingest_odds.py --snapshot "$(date -Iseconds)" --production

# Step 2: Build market anchor features
python scripts/build_market_anchors.py --current-week

# Step 3: Create unified feature matrices
python scripts/build_features.py --current-week --targets wp,ats,ou

# Step 4: Validate features for data leakage
python scripts/validate_features.py --current-week --check-leakage

# Step 5: Validate model availability
python scripts/validate_models.py --models wp,ats,ou --required

# Step 6: Generate predictions
python -c "from models.prediction_pipeline import NFLPredictionPipeline; pipeline = NFLPredictionPipeline(); results = pipeline.predict_current_week(); print(f'Generated {len(results)} predictions')"

# Step 7: Generate bet recommendations
python -c "from models.bet_recommender import BetRecommendationEngine; engine = BetRecommendationEngine(); recs = engine.generate_weekly_recommendations(); print(f'Generated {len(recs)} recommendations')"

# Step 8: Export prediction artifacts
python -c "from models.prediction_pipeline import PredictionExporter; exporter = PredictionExporter(); exporter.export_current_week_predictions(formats=['parquet', 'json', 'csv'])"

# Step 9: Validate prediction outputs
python scripts/validate_predictions.py --current-week --check-completeness

# Step 10: Health check
python scripts/health_check.py --comprehensive
```

**Post-Prediction: Monitoring**
```bash
# Monitor system health
python scripts/health_check.py --mode comprehensive

# Check prediction quality
python scripts/validate_predictions.py --current-week

# Start web interface (if not running)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### **⚡ Quick Commands for Development**

**For testing current week only (no historical data)**:
```bash
# Quick current week pipeline
make snapshot    # Current week data only
make predict     # Generate predictions
make serve       # Start web interface
```

**For model development**:
```bash
# Test feature engineering
make features-build

# Test model training
make models-train

# Run test suite
make test
```

---

## Expected Outputs

### After Data Snapshot (`make snapshot`)
```
✅ Bronze data files in data/bronze/
✅ Silver tables in data/silver/ (games.parquet, odds_snapshot.parquet)
✅ Gold feature matrices in data/gold/
✅ Feature validation passes with no leakage detected
```

### After Backtest (`make backtest`)
```
✅ Backtest report: outputs/backtest/backtest_report.html
✅ Metrics summary: outputs/backtest/metrics_summary.json
✅ Betting simulation: outputs/backtest/betting_simulation.csv
✅ Performance metrics: WP accuracy >65%, ATS >52%, Betting ROI >5%
```

### After Predictions (`make predict`)
```
✅ Prediction files: outputs/predictions/current_week_predictions.parquet
✅ JSON format: outputs/predictions/current_week_predictions.json
✅ Recommendations: outputs/predictions/current_week_recommendations.json
✅ Summary CSV: outputs/predictions/current_week_summary.csv
```

### After Web Server (`make serve`)
```
✅ API available at: http://localhost:8000
✅ Health endpoint: http://localhost:8000/health returns "healthy"
✅ Web interface shows current week games and predictions
✅ Swagger docs: http://localhost:8000/docs
```

---

## Troubleshooting

### Common Issues

#### 1. "game_date not found" Error
**Symptoms**: API returns empty games, warnings in logs
**Cause**: Database schema uses `kickoff_et` instead of `game_date`
**Impact**: System works but API queries return no results
**Status**: Known issue, system partially functional

#### 2. "Model artifacts unhealthy"
**Symptoms**: Health check fails on models
**Cause**: No trained models in `artifacts/models/`
**Solution**: Run model training or use demo mode
```bash
# Quick demo prediction
python scripts/generate_current_week_predictions.py --demo-mode
```

#### 3. "Odds data stale"
**Symptoms**: Data quality warnings about old odds
**Cause**: Odds data >6 hours old
**Solution**: Refresh odds data
```bash
python scripts/ingest_odds.py --snapshot "$(date -Iseconds)" --force
```

#### 4. "`make` command not found"
**Cause**: Make utility not installed (common on Windows)
**Solution**: Use Manual Commands method above, or install make:
```bash
# On Windows with Chocolatey
choco install make

# Or use Git Bash / WSL which includes make
```

#### 5. External API Errors (Odds/Weather)
**Symptoms**: Odds ingestion fails with "Invalid commenceTimeFrom parameter"
**Cause**: External API requires specific date formats or valid API keys
**Solution**: Use mock data for testing/development only
```bash
# For testing pipeline without API access
python scripts/ingest_odds.py --current --mock
python scripts/ingest_weather.py --current --mock

# For production, use real data (requires API keys)
python scripts/ingest_odds.py --current
python scripts/ingest_weather.py --current
```

#### 6. Empty Games in Web Interface
**Symptoms**: Web shows "0 games" despite having data
**Cause**: Schema mismatch in API queries
**Status**: System functional for backtest/calibration, API needs schema fix

### Data Cleanup for Testing

**Use Case**: Clean pipeline testing, recovering from failed ingestion, or starting fresh

The system includes a comprehensive data cleanup utility for safe pipeline testing:

```bash
# Clean only weather data (safe for weather pipeline testing)
python scripts/cleanup_data.py --weather-only

# Clean all data files (nuclear option - removes bronze/silver/gold)
python scripts/cleanup_data.py --all-data

# Clean output files only (predictions, backtest reports)
python scripts/cleanup_data.py --outputs

# Clean Python caches and temporary files
python scripts/cleanup_data.py --caches

# Clean everything (data + outputs + caches)
python scripts/cleanup_data.py --everything

# Skip confirmation prompts
python scripts/cleanup_data.py --weather-only --force
```

**Alternative: Use Makefile commands** (if `make` is available):
```bash
# Clean all data files (with confirmation)
make clean-data

# Clean output files
make clean-outputs

# Clean Python caches
make clean
```

**Recommended Testing Workflow**:
1. Clean specific data: `python scripts/cleanup_data.py --weather-only`
2. Test single week: `python scripts/ingest_weather.py --season 2018 --week 1 --mock`
3. Test full season: `python scripts/ingest_weather.py --season 2018 --mock`
4. Verify results: Check for proper deduplication and file structure

**Production vs Testing Weather Data**:
- **Testing**: Use `--mock` flag to generate fake weather data for pipeline validation
- **Production**: Omit `--mock` flag to use real weather data for accurate predictions
- Weather conditions (wind, temperature, precipitation) significantly impact NFL game outcomes

### Debug Commands

```bash
# Check system status
python scripts/health_check.py --mode comprehensive

# Check data files
ls -la data/bronze/ data/silver/ data/gold/

# Check database tables
python -c "import duckdb; conn = duckdb.connect('data/nfl_predictions.duckdb'); print(conn.execute('SHOW TABLES').fetchall())"

# Check API endpoints manually
curl http://localhost:8000/health
curl http://localhost:8000/backtest
curl http://localhost:8000/calibration

# View recent logs
tail -f logs/nfl_predict.log

# Check operational monitoring
python scripts/operational_monitoring.py --mode full
```

---

## Next Steps

### After Getting Started

1. **Explore the Web Interface**
   - Browse http://localhost:8000 to see the UI
   - Check backtest performance at http://localhost:8000/backtest
   - View model calibration at http://localhost:8000/calibration

2. **Review System Performance**
   ```bash
   # Check backtest results
   python -c "import json; data=json.load(open('outputs/backtest/metrics_summary.json')); print(f'WP Accuracy: {data[\"wp_accuracy\"]:.1%}'); print(f'Betting ROI: {data[\"betting_roi\"]:.1%}')"
   ```

3. **Set Up Automated Operations**
   - Review `docs/operational_runbooks.md` for weekly schedule
   - Set up the Friday 6 PM ET production run
   - Configure monitoring and alerts

4. **Customize for Your Needs**
   - Adjust edge thresholds in `conf/config.yaml`
   - Modify UI preferences in `web/templates/`
   - Add custom features in `features/`

### Production Deployment

When ready for production:
1. Follow `deployment/DEPLOYMENT.md` for Docker setup
2. Configure environment variables in `.env`
3. Set up proper SSL certificates
4. Enable monitoring and alerting

---

## Support

### Documentation
- **API Reference**: `docs/api_documentation.md`
- **Model Details**: `docs/model_methodology.md`
- **Operations Guide**: `docs/operational_runbooks.md`
- **Troubleshooting**: `docs/operational_procedures.md`

### Getting Help
- Check the Swagger API docs at http://localhost:8000/docs
- Review error logs in `logs/` directory
- Run diagnostic commands above
- Check system status with `python scripts/health_check.py --mode comprehensive`

---

## Quick Reference

### **First-Time Setup Commands (NEW: Standardized Arguments)**
```bash
# System setup
make setup

# Manual historical data ingestion (required) - NEW: consistent syntax
python scripts/ingest_games.py --seasons 2018 2019 2020 2021 2022 2023 2024

# Historical odds (real data from nflreadpy (previously nfl_data_py, replaced in Phase 1) for backtesting)
# This ingests 1,855 real consensus betting lines for all games 2018-2024
python scripts/ingest_historical_odds.py

# Historical weather - NEW: can do all seasons at once or individually
# Option 1: All seasons at once (faster)
python scripts/ingest_weather.py --seasons 2018 2019 2020 2021 2022 2023 2024 --all

# Option 2: One season at a time (better progress monitoring)
# python scripts/ingest_weather.py --season 2018
# python scripts/ingest_weather.py --season 2019
# python scripts/ingest_weather.py --season 2020
# python scripts/ingest_weather.py --season 2021
# python scripts/ingest_weather.py --season 2022
# python scripts/ingest_weather.py --season 2023
# python scripts/ingest_weather.py --season 2024

# Current season data - NEW: --current gets weeks 1 through current week
python scripts/ingest_games.py --current      # Gets weeks 1-3 (not just week 3)
python scripts/ingest_weather.py --current    # Gets weeks 1-3

# Feature engineering and training
python scripts/build_elo.py --all-seasons
make features-build
make models-train

# Validation and testing
make backtest
make predict
make serve
```

### **Weekly Operations Commands**
```bash
# Automated weekly cycle
make weekly-update          # Complete Tuesday-Saturday process
make friday-production      # Friday 6 PM ET production run
make serve                  # Start web interface

# Manual weekly operations
make data-ingest           # Update current week data
make features-build        # Build current week features
make snapshot              # Create silver/gold tables
make predict               # Generate predictions
```

### **Development & Testing Commands**
```bash
# Current week only (for testing)
make snapshot               # Current week data
make predict               # Generate predictions
make serve                 # Start web interface

# Health checks
python scripts/health_check.py --mode basic
python scripts/health_check.py --mode comprehensive

# Data quality
python scripts/data_qa.py --current --save

# Model validation
python scripts/validate_models.py --models wp,ats,ou

# Manual API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Common Paths
- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Data Files**: `data/bronze/`, `data/silver/`, `data/gold/`
- **Outputs**: `outputs/predictions/`, `outputs/backtest/`
- **Logs**: `logs/nfl_predict.log`

---

**🏈 Ready to predict some NFL games!**

This NFL prediction system provides two distinct operational modes:

- **🎯 First-Time Setup**: Complete historical pipeline with 7 years of data for robust backtesting
- **⚡ Weekly Operations**: Streamlined production cycle for generating fresh predictions

The system is designed to be professional-grade with comprehensive monitoring, beautiful interfaces, and robust ML models. Choose the setup method that matches your needs and follow the clear step-by-step instructions above.