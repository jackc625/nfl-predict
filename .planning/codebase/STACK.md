# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- Python 3.11+ - Core application, data pipeline, modeling, API, scripts
- JavaScript - Frontend build tooling (Tailwind CSS compilation)
- HTML/Jinja2 - Web UI templates

**Secondary:**
- SQL (DuckDB) - Data querying and analytics

## Runtime

**Environment:**
- Python 3.11 or 3.12 (as specified in `pyproject.toml`)

**Package Manager:**
- pip (via setuptools/pyproject.toml)
- npm (for frontend CSS building)
- Lockfile: None (pip uses `pyproject.toml` with version constraints)

## Frameworks

**Core:**
- FastAPI 0.100.0+ - REST API framework for predictions and reporting
- Uvicorn 0.23.0+ - ASGI application server

**Data Processing:**
- pandas 2.0.0+ - Data manipulation and analysis
- polars 0.18.0+ - Alternative/supplementary dataframe library
- pyarrow 10.0.0+ - Columnar data format and Apache Arrow support
- duckdb 0.8.0+ - Embedded database for data storage and querying

**Modeling & ML:**
- scikit-learn 1.3.0+ - Logistic regression (WP model), calibration, preprocessing
- xgboost 1.7.0+ - Gradient boosted trees for ATS and O/U models
- lightgbm 4.0.0+ - Alternative gradient boosting library
- statsmodels 0.14.0+ - Statistical modeling and analysis

**Web Framework:**
- Jinja2 3.1.0+ - HTML template rendering for web UI
- pydantic 2.0.0+ - Data validation and API schemas
- pydantic-settings 2.0.0+ - Configuration management from environment variables

**Testing:**
- pytest 7.4.0+ - Test runner and framework
- pytest-cov 4.1.0+ - Code coverage reporting
- pytest-asyncio 0.21.0+ - Async test support
- hypothesis 6.82.0+ - Property-based testing

**Code Quality:**
- black 23.7.0+ - Code formatter (88 char line length)
- isort 5.12.0+ - Import sorter (black-compatible profile)
- flake8 6.0.0+ - Linter
- mypy 1.5.0+ - Static type checker (strict mode)
- pre-commit 3.3.0+ - Git hook framework

**Development:**
- jupyter 1.0.0+ - Notebook environment for exploration
- ipython 8.14.0+ - Interactive Python shell

**Frontend Build:**
- tailwindcss 3.3.0+ - CSS framework (via npm)
- concurrently 8.2.0+ - Run multiple npm scripts in parallel
- @tailwindcss/forms 0.5.7+ - Form styling plugin
- @tailwindcss/typography 0.5.10+ - Typography styling plugin

## Key Dependencies

**Critical:**
- httpx 0.24.0+ - Async HTTP client for external API calls (odds, weather)
- tenacity 8.2.0+ - Retry logic with exponential backoff
- nfl-data-py 0.3.0+ - NFL historical game data source
- meteostat 1.6.0+ - Historical weather data provider

**Infrastructure:**
- python-dotenv 1.0.0+ - Environment variable loading from .env
- pyyaml - YAML configuration file parsing
- structlog 23.1.0+ - Structured logging with JSON output
- click 8.1.0+ - CLI framework for command-line utilities
- pytz 2023.3+ - Timezone handling
- python-dateutil 2.8.0+ - Date/time utilities
- pendulum 2.1.0+ - Alternative datetime library
- tqdm 4.65.0+ - Progress bars
- joblib 1.3.0+ - Parallel computing and caching
- python-multipart 0.0.6+ - Form data parsing
- itsdangerous 2.1.0+ - Security utilities (token generation, signing)

**Visualization (for reports):**
- matplotlib 3.7.0+ - Static plotting
- plotly 5.15.0+ - Interactive charts and graphs
- seaborn 0.12.0+ - Statistical visualization

**Testing Utilities:**
- beautifulsoup4 4.12.0+ - HTML parsing for UI snapshot tests

## Configuration

**Environment:**
- Configuration via `conf/config.yaml` (YAML-based, Pydantic models in `conf/settings.py`)
- Secrets and overrides via `.env` file (never committed)
- Environment variable mapping: `ODDS_API_KEY`, `SECRET_KEY`, `API_HOST`, `API_PORT`, `LOG_LEVEL`, `TIMEZONE`, etc.

**Build:**
- `pyproject.toml`: Python package configuration and dependency management
- `package.json`: Frontend build tool configuration
- `Makefile`: Standardized commands for data pipeline, backtesting, prediction, serving
- `tailwind.config.js`: Tailwind CSS customization

**Logging:**
- JSON-formatted structured logging (via `structlog`)
- Output to console and optional file (`logs/nfl-predict.log`)
- Configurable levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Platform Requirements

**Development:**
- Python 3.11+
- Node.js (for npm/Tailwind CSS)
- 4GB+ RAM (DuckDB configured with 4GB memory limit)
- Git

**Production:**
- Python 3.11+
- Uvicorn-compatible server (Docker recommended)
- 4GB+ RAM for DuckDB operations
- Filesystem access for data storage (parquet, DuckDB files)
- 10+ GB disk for data lake (bronze/silver/gold layers)

## Key Architecture Patterns

**Configuration Management:**
- Two-tier system: YAML defaults (`conf/config.yaml`) + Pydantic settings (`conf/settings.py`)
- Environment variable overrides via BaseSettings
- Runtime access via cached singleton: `get_settings()`

**Data Storage:**
- Multi-layer data lake: bronze (raw snapshots) → silver (cleaned) → gold (features)
- Format: Parquet files (columnar) for batch data, DuckDB for queries
- Connection pooling managed by `DuckDBConnection` utility class

**Async I/O:**
- httpx for async HTTP calls to external APIs
- FastAPI async endpoints for non-blocking request handling

**Type Safety:**
- Strict MyPy configuration (disallow untyped defs, implicit optionals)
- Pydantic schemas for API request/response validation
- Type hints throughout codebase

---

*Stack analysis: 2026-03-18*
