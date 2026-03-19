# NFL Prediction System

A Python-based system that produces pre-game predictions for every NFL matchup each week: Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) -- backed by walk-forward backtesting with strict no-leakage validation.

## What It Does

- **Win Probability** -- calibrated home/away win probabilities using Elo + team form + contextual features
- **Against the Spread** -- margin predictions and cover probabilities via gradient-boosted models
- **Over/Under** -- total points predictions with weather impact modeling
- **Walk-Forward Backtesting** -- rigorous temporal validation across 2018-2024 seasons (train on years <= Y-1, test on Y)
- **Web Dashboard** -- FastAPI backend serving predictions through an HTMX-powered, mobile-responsive UI

## Tech Stack

**Core:** Python 3.13, pandas, DuckDB, Parquet, scikit-learn, XGBoost

**API/UI:** FastAPI, Jinja2, Tailwind CSS v4 (standalone), HTMX 2.0

**Data Sources:** nflreadpy (game data), The Odds API (odds), Open-Meteo (weather)

**Tooling:** uv (package management), Ruff (linting/formatting), pyright (type checking)

## Architecture

```
External Sources -> Ingestion (ETL) -> Data Lake (Parquet/DuckDB) -> Feature Build -> Models -> Predictions API -> Web UI
```

**Three-layer data pipeline:**
- **Bronze** -- Raw data snapshots, append-only with timestamps
- **Silver** -- Cleaned, validated tables (games, odds_snapshot, weather, team_stats)
- **Gold** -- Feature matrices per prediction target (WP, ATS, O/U)

Quality gates enforce schema validation at each boundary. A canonical team abbreviation module resolves all historical variations (JAC/JAX, STL/LA, SD/LAC, OAK/LV, WSH/WAS) with hard-fail on unknowns.

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Foundation Hardening | Dependency upgrades, error handling, toolchain | Complete |
| 2. Data Pipeline Audit | Ingestion hardening, quality gates, team mapping | Complete |
| 3. Feature Engineering Correctness | Leakage audit, time-fence enforcement, feature compression | Next |
| 4. Core Model Training | Walk-forward WP/ATS/O/U with CLV evaluation | Planned |
| 5. Differentiating Features | QB adjustment, opponent-adjusted EPA, CPOE | Planned |
| 6. Backtest Reporting | Full walk-forward reports with CLV and betting simulation | Planned |
| 7. Market Reversion | Model-market blending, Kelly sizing, edge calibration | Planned |
| 8. API and Web UI | Dashboard, drill-down views, CSV export, mobile design | Planned |

See [`.planning/ROADMAP.md`](.planning/ROADMAP.md) for full phase details and requirements mapping.

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Setup

```bash
git clone https://github.com/yourusername/nfl-predict.git
cd nfl-predict

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (The Odds API, etc.)
```

### Run Development Server

```bash
uvicorn api.main:app --reload --port 8000
```

Visit http://localhost:8000 for the web interface, http://localhost:8000/docs for the API docs.

## Project Structure

```
nfl-predict/
  data/                  # Data storage (excluded from git)
    bronze/              # Raw data snapshots
    silver/              # Cleaned tables
    gold/                # Feature matrices
  scripts/               # Data ingestion and processing scripts
  ratings/               # Elo rating system
  features/              # Feature engineering modules
  models/                # Model training and prediction
  backtest/              # Walk-forward validation
  api/                   # FastAPI backend
  web/                   # Web UI (Jinja2 templates, Tailwind CSS, HTMX)
    templates/
    static/
  conf/                  # Configuration (config.yaml)
  tests/                 # Test suite
  deployment/            # Docker, Nginx, scheduling configs
  .planning/             # Project planning (roadmap, requirements, state)
  outputs/               # Prediction artifacts (excluded from git)
  artifacts/             # Model artifacts (excluded from git)
```

## License

Personal project. Not licensed for redistribution.
