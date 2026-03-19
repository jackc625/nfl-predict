# NFL Prediction System Makefile
#
# This Makefile provides standardized commands for running the NFL prediction system
# in accordance with the PRD acceptance criteria.

.PHONY: help snapshot backtest predict serve test clean install setup lint build-css train train-wp train-ats train-ou

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := uv run python
VENV_DIR := .venv
CONDA_ENV := nfl-predict
API_HOST := 0.0.0.0
API_PORT := 8000
WORKERS := 4

# Current date/time variables
CURRENT_DATE := $(shell date +%Y-%m-%d)
CURRENT_TIMESTAMP := $(shell date +%Y-%m-%dT%H:%M:%S%z)

# Color codes for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "NFL Prediction System - Available Commands"
	@echo "=========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Configuration:"
	@echo "  Python: $(PYTHON)"
	@echo "  API Host: $(API_HOST):$(API_PORT)"
	@echo "  Workers: $(WORKERS)"
	@echo ""

# =============================================================================
# CORE ACCEPTANCE CRITERIA COMMANDS
# =============================================================================

snapshot: ## Produce silver/gold tables for current week (PRD Acceptance Criteria #1)
	@echo "$(GREEN)📊 Producing silver/gold tables for current week...$(NC)"
	@echo "Timestamp: $(CURRENT_TIMESTAMP)"
	@echo ""

	@echo "$(YELLOW)Step 1: Ingesting raw data (Bronze layer)...$(NC)"
	$(PYTHON) scripts/ingest_games.py --current
	$(PYTHON) scripts/ingest_odds.py --snapshot-time "$(CURRENT_TIMESTAMP)" --current
	$(PYTHON) scripts/ingest_weather.py --current

	@echo "$(YELLOW)Step 2: Data quality validation...$(NC)"
	$(PYTHON) scripts/data_qa.py --current-week --strict

	@echo "$(YELLOW)Step 3: Building feature components...$(NC)"
	$(PYTHON) scripts/build_elo.py --incremental --current-week
	$(PYTHON) scripts/build_team_form.py --incremental --current-week
	$(PYTHON) scripts/build_contextual.py --current-week
	$(PYTHON) scripts/build_weather.py --current-week
	$(PYTHON) scripts/build_market_anchors.py --current-week

	@echo "$(YELLOW)Step 4: Creating unified feature matrices (Gold layer)...$(NC)"
	$(PYTHON) scripts/build_features.py --current-week --targets wp,ats,ou

	@echo "$(YELLOW)Step 5: Feature validation...$(NC)"
	$(PYTHON) scripts/validate_features.py --current-week --check-leakage

	@echo "$(GREEN)✅ Snapshot complete! Silver/gold tables updated for current week.$(NC)"
	@echo "Output location: data/silver/ and data/gold/"
	@echo ""

backtest: ## Run walk-forward validation 2018-2024 with metrics report (PRD Acceptance Criteria #2)
	@echo "$(GREEN)🔄 Running walk-forward backtest validation (2018-2024)...$(NC)"
	@echo "This may take 30-60 minutes depending on system performance."
	@echo ""

	@echo "$(YELLOW)Step 1: Preparing historical data...$(NC)"
	$(PYTHON) scripts/validate_historical_data.py --seasons 2018-2024

	@echo "$(YELLOW)Step 2: Running walk-forward validation...$(NC)"
	$(PYTHON) -c "from backtest.walkforward import WalkForwardBacktester; from backtest.metrics import BacktestMetricsCalculator; bt = WalkForwardBacktester(); results = bt.run_backtest(start_season=2018, end_season=2024); print('Backtest completed successfully')"

	@echo "$(YELLOW)Step 3: Generating metrics report...$(NC)"
	$(PYTHON) -c "from backtest.reporting import BacktestReporter; reporter = BacktestReporter(); reporter.generate_full_report(start_season=2018, end_season=2024, output_dir='outputs/backtest')"

	@echo "$(YELLOW)Step 4: Validating backtest reproducibility...$(NC)"
	$(PYTHON) scripts/validate_backtest_reproducibility.py --seasons 2018-2024

	@echo "$(GREEN)✅ Backtest complete! Reports available in outputs/backtest/$(NC)"
	@echo "Key files:"
	@echo "  - outputs/backtest/backtest_report.html (Main report)"
	@echo "  - outputs/backtest/metrics_summary.json (Summary metrics)"
	@echo "  - outputs/backtest/betting_simulation.csv (Betting results)"
	@echo ""

predict: ## Generate prediction artifacts (Parquet + JSON) (PRD Acceptance Criteria #3)
	@echo "$(GREEN)🎯 Generating prediction artifacts for current week...$(NC)"
	@echo "Timestamp: $(CURRENT_TIMESTAMP)"
	@echo ""

	@echo "$(YELLOW)Step 1: Validating model availability...$(NC)"
	$(PYTHON) scripts/validate_models.py --models wp,ats,ou --required

	@echo "$(YELLOW)Step 2: Generating predictions...$(NC)"
	$(PYTHON) -c "from models.prediction_pipeline import NFLPredictionPipeline; pipeline = NFLPredictionPipeline(); results = pipeline.predict_current_week(); print(f'Generated {len(results)} predictions')"

	@echo "$(YELLOW)Step 3: Generating bet recommendations...$(NC)"
	$(PYTHON) -c "from models.bet_recommender import BetRecommendationEngine; engine = BetRecommendationEngine(); recs = engine.generate_weekly_recommendations(); print(f'Generated {len(recs)} bet recommendations')"

	@echo "$(YELLOW)Step 4: Exporting prediction artifacts...$(NC)"
	$(PYTHON) -c "from models.prediction_pipeline import PredictionExporter; exporter = PredictionExporter(); exporter.export_current_week_predictions(formats=['parquet', 'json', 'csv'])"

	@echo "$(YELLOW)Step 5: Validating prediction outputs...$(NC)"
	$(PYTHON) scripts/validate_predictions.py --current-week --check-completeness

	@echo "$(GREEN)✅ Prediction artifacts generated!$(NC)"
	@echo "Output files:"
	@echo "  - outputs/predictions/current_week_predictions.parquet"
	@echo "  - outputs/predictions/current_week_predictions.json"
	@echo "  - outputs/predictions/current_week_recommendations.json"
	@echo "  - outputs/predictions/current_week_summary.csv"
	@echo ""

serve: ## Start web UI/API server (PRD Acceptance Criteria #4)
	@echo "$(GREEN)🌐 Starting NFL Prediction API and Web UI...$(NC)"
	@echo "Server will be available at: http://$(API_HOST):$(API_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@echo ""

	@echo "$(YELLOW)Pre-flight checks...$(NC)"
	@$(PYTHON) scripts/health_check.py --pre-startup

	@echo "$(YELLOW)Starting FastAPI server with $(WORKERS) workers...$(NC)"
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --workers $(WORKERS) --log-level info

# =============================================================================
# DEVELOPMENT AND TESTING COMMANDS
# =============================================================================

test: ## Run the full test suite
	@echo "$(GREEN)🧪 Running test suite...$(NC)"
	@echo ""

	@echo "$(YELLOW)Step 1: Unit tests...$(NC)"
	pytest tests/unit/ -v --tb=short

	@echo "$(YELLOW)Step 2: Integration tests...$(NC)"
	pytest tests/integration/ -v --tb=short

	@echo "$(YELLOW)Step 3: API tests...$(NC)"
	pytest tests/api/ -v --tb=short

	@echo "$(YELLOW)Step 4: UI tests...$(NC)"
	pytest tests/ui/ -v --tb=short

	@echo "$(GREEN)✅ All tests completed!$(NC)"

test-quick: ## Run quick test suite (unit tests only)
	@echo "$(GREEN)⚡ Running quick test suite...$(NC)"
	pytest tests/unit/ -v --tb=line -x

test-models: ## Test model training and prediction pipeline
	@echo "$(GREEN)Testing model pipeline...$(NC)"
	pytest tests/unit/test_elo_and_probabilities.py -v --tb=short

test-features: ## Test feature engineering pipeline
	@echo "$(GREEN)Testing feature pipeline...$(NC)"
	pytest tests/unit/test_feature_builders.py -v --tb=short

test-api: ## Test API endpoints
	@echo "$(GREEN)Testing API endpoints...$(NC)"
	pytest tests/ -k "api" -v --tb=short
	@echo "Note: Start the server with 'make serve' in another terminal for full API testing"

# =============================================================================
# DATA MANAGEMENT COMMANDS
# =============================================================================

data-ingest: ## Ingest all data sources for current week
	@echo "$(GREEN)📥 Ingesting data for current week...$(NC)"
	$(PYTHON) scripts/ingest_games.py --current
	$(PYTHON) scripts/ingest_odds.py --current
	$(PYTHON) scripts/ingest_weather.py --current
	$(PYTHON) scripts/data_qa.py --current-week

data-ingest-season: ## Ingest full season data (specify SEASON=2024)
	@echo "$(GREEN)📥 Ingesting full season data for $(SEASON)...$(NC)"
	$(PYTHON) scripts/ingest_games.py --season $(SEASON)
	$(PYTHON) scripts/ingest_odds.py --season $(SEASON)
	$(PYTHON) scripts/ingest_weather.py --season $(SEASON)
	$(PYTHON) scripts/data_qa.py --season $(SEASON)

data-validate: ## Validate data quality and integrity
	@echo "$(GREEN)✔️ Validating data quality...$(NC)"
	$(PYTHON) scripts/data_qa.py --comprehensive
	$(PYTHON) scripts/validate_features.py --check-leakage
	$(PYTHON) scripts/operational_monitoring.py --check-data-quality

features-build: ## Build all features for current week
	@echo "$(GREEN)🔧 Building features for current week...$(NC)"
	$(PYTHON) scripts/build_elo.py --current-week
	$(PYTHON) scripts/build_team_form.py --current-week
	$(PYTHON) scripts/build_contextual.py --current-week
	$(PYTHON) scripts/build_weather.py --current-week
	$(PYTHON) scripts/build_market_anchors.py --current-week
	$(PYTHON) scripts/build_features.py --current-week

features-validate: ## Validate feature engineering pipeline
	@echo "$(GREEN)Validating features...$(NC)"
	$(PYTHON) scripts/validate_features.py --comprehensive

# =============================================================================
# MODEL MANAGEMENT COMMANDS
# =============================================================================

train: ## Train all models (WP, ATS, O/U) with walk-forward temporal validation
	@echo "$(GREEN)Training all models with walk-forward validation...$(NC)"
	$(PYTHON) -m models.train --target all

train-wp: ## Train WP model only
	$(PYTHON) -m models.train --target wp

train-ats: ## Train ATS model only
	$(PYTHON) -m models.train --target ats

train-ou: ## Train O/U model only
	$(PYTHON) -m models.train --target ou

models-train: train ## Legacy alias

# models-train-incremental removed -- walk-forward training replaces incremental

models-validate: ## Validate trained models
	@echo "$(GREEN)✔️ Validating models...$(NC)"
	$(PYTHON) scripts/validate_models.py --comprehensive
	$(PYTHON) scripts/operational_monitoring.py --check-model-health

# =============================================================================
# MONITORING AND HEALTH COMMANDS
# =============================================================================

health-check: ## Run comprehensive health check
	@echo "$(GREEN)🏥 Running health check...$(NC)"
	$(PYTHON) scripts/health_check.py --comprehensive

health-monitor: ## Run operational monitoring
	@echo "$(GREEN)📊 Running operational monitoring...$(NC)"
	$(PYTHON) scripts/operational_monitoring.py --comprehensive

status: ## Show system status
	@echo "$(GREEN)📋 System Status$(NC)"
	@echo "==============="
	@echo "Date: $(CURRENT_DATE)"
	@echo "Timestamp: $(CURRENT_TIMESTAMP)"
	@echo ""
	@$(PYTHON) scripts/health_check.py --brief
	@echo ""
	@echo "Data Status:"
	@$(PYTHON) -c "import os; from pathlib import Path; print(f'Bronze tables: {len(list(Path(\"data/bronze\").glob(\"*.parquet\")))} files') if Path('data/bronze').exists() else print('Bronze: Not found')"
	@$(PYTHON) -c "import os; from pathlib import Path; print(f'Silver tables: {len(list(Path(\"data/silver\").glob(\"*.parquet\")))} files') if Path('data/silver').exists() else print('Silver: Not found')"
	@$(PYTHON) -c "import os; from pathlib import Path; print(f'Gold tables: {len(list(Path(\"data/gold\").glob(\"*.parquet\")))} files') if Path('data/gold').exists() else print('Gold: Not found')"
	@echo ""

# =============================================================================
# SETUP AND INSTALLATION COMMANDS
# =============================================================================

setup: ## Full system setup and installation
	@echo "$(GREEN)🚀 Setting up NFL Prediction System...$(NC)"
	@make install
	@make setup-directories
	@make setup-config
	@echo "$(GREEN)✅ Setup complete!$(NC)"

install: ## Install Python dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	uv sync

setup-directories: ## Create required directory structure
	@echo "$(GREEN)📁 Creating directory structure...$(NC)"
	mkdir -p data/{bronze,silver,gold}
	mkdir -p outputs/{predictions,backtest,reports}
	mkdir -p artifacts/{models,features}
	mkdir -p logs
	mkdir -p web/{templates,static}

setup-config: ## Setup configuration files
	@echo "$(GREEN)⚙️ Setting up configuration...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit with your API keys."; \
	fi

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

clean: ## Clean generated files and caches
	@echo "$(GREEN)🧹 Cleaning generated files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf build dist
	@echo "Cache cleaned!"

clean-data: ## Clean all data files (WARNING: destructive)
	@echo "$(RED)⚠️  WARNING: This will delete all data files!$(NC)"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || exit 1
	rm -rf data/bronze/*
	rm -rf data/silver/*
	rm -rf data/gold/*
	@echo "Data files cleaned!"

clean-outputs: ## Clean output files
	@echo "$(GREEN)🗑️ Cleaning output files...$(NC)"
	rm -rf outputs/predictions/*
	rm -rf outputs/backtest/*
	rm -rf outputs/reports/*
	@echo "Output files cleaned!"

logs: ## Show recent log entries
	@echo "$(GREEN)📋 Recent log entries:$(NC)"
	@if [ -f logs/nfl_predict.log ]; then \
		tail -n 50 logs/nfl_predict.log; \
	else \
		echo "No log file found at logs/nfl_predict.log"; \
	fi

demo: ## Run system demonstration
	@echo "$(GREEN)🎬 Running system demonstration...$(NC)"
	$(PYTHON) scripts/demo_prediction_pipeline.py
	$(PYTHON) scripts/demo_wp_model.py
	$(PYTHON) scripts/demo_ats_model.py
	$(PYTHON) scripts/demo_ou_model.py

version: ## Show version information
	@echo "NFL Prediction System"
	@echo "===================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"

# =============================================================================
# WEEKLY AUTOMATION COMMANDS
# =============================================================================

weekly-update: ## Complete weekly update process (Tuesday-Saturday)
	@echo "$(GREEN)📅 Running complete weekly update...$(NC)"
	@echo "This runs the full weekly process as outlined in operational runbooks"
	@echo ""

	@echo "$(YELLOW)Tuesday: Data validation...$(NC)"
	@make data-validate

	@echo "$(YELLOW)Wednesday: Feature refresh...$(NC)"
	@make features-build
	@make features-validate

	@echo "$(YELLOW)Thursday: Model updates (if needed)...$(NC)"
	@$(PYTHON) scripts/operational_monitoring.py --check-model-drift
	@make models-validate

	@echo "$(YELLOW)Friday: Snapshot and predictions...$(NC)"
	@make snapshot
	@make predict

	@echo "$(YELLOW)Saturday: Final validation...$(NC)"
	@make health-check

	@echo "$(GREEN)✅ Weekly update complete!$(NC)"

friday-production: ## Friday 6 PM ET production run
	@echo "$(GREEN)🏈 Friday Production Run - NFL Predictions$(NC)"
	@echo "Time: $(CURRENT_TIMESTAMP)"
	@echo "This is the official Friday 6 PM ET production process"
	@echo ""

	@echo "$(YELLOW)Step 1: Market snapshot...$(NC)"
	$(PYTHON) scripts/ingest_odds.py --snapshot-time "$(CURRENT_TIMESTAMP)" --current

	@echo "$(YELLOW)Step 2: Final data preparation...$(NC)"
	@make snapshot

	@echo "$(YELLOW)Step 3: Generate predictions...$(NC)"
	@make predict

	@echo "$(YELLOW)Step 4: Health validation...$(NC)"
	@make health-check

	@echo "$(GREEN)🎯 Production run complete! Predictions ready for weekend.$(NC)"

# =============================================================================
# FRONTEND BUILD
# =============================================================================

build-css: ## Build Tailwind CSS
	./tools/tailwindcss -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify

# =============================================================================
# DEVELOPMENT SHORTCUTS
# =============================================================================

dev-serve: ## Start development server with auto-reload
	@echo "$(GREEN)🔧 Starting development server...$(NC)"
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --reload --log-level debug

dev-test: ## Development testing with coverage
	@echo "$(GREEN)🧪 Running development tests with coverage...$(NC)"
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint: ## Run linting (Ruff check + format check)
	uv run ruff check . && uv run ruff format --check .

dev-lint: lint ## Alias for lint

dev-format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	uv run ruff format .
	uv run ruff check . --fix

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Allow overriding configuration via environment variables
ifdef PRODUCTION
API_HOST := 0.0.0.0
API_PORT := 8000
WORKERS := 8
endif

ifdef API_HOST_OVERRIDE
API_HOST := $(API_HOST_OVERRIDE)
endif

ifdef API_PORT_OVERRIDE
API_PORT := $(API_PORT_OVERRIDE)
endif

ifdef WORKERS_OVERRIDE
WORKERS := $(WORKERS_OVERRIDE)
endif

# Season override for data commands
ifndef SEASON
SEASON := 2024
endif