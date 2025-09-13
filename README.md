# NFL Prediction System

A Python-based system that produces pre-game predictions for every NFL matchup each week, including Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions.

## Project Structure

```
project/
├─ data/                    # Data storage (excluded from git)
│   ├─ bronze/             # Raw data snapshots
│   ├─ silver/             # Cleaned tables (games, odds, weather, team_stats)
│   └─ gold/               # Feature matrices per target (wp, ats, ou)
├─ scripts/                # Data ingestion and processing
├─ ratings/                # Elo rating system
├─ models/                 # Model training and prediction
├─ backtest/              # Walk-forward validation and evaluation
├─ api/                   # FastAPI backend
├─ web/                   # Web UI templates and static files
│   ├─ templates/         # Jinja2 templates
│   └─ static/           # CSS, JS, images
├─ conf/                  # Configuration files
├─ tests/                 # Test suite
├─ outputs/               # Prediction artifacts (excluded from git)
├─ artifacts/             # Model artifacts (excluded from git)
├─ CLAUDE.md             # Claude Code guidance
├─ TODO.md               # Implementation task manager
└─ PRD.md                # Product Requirements Document
```

## Getting Started

This project is currently in development. See `TODO.md` for the implementation roadmap.

## Key Features (Planned)

- **Reproducible data pipeline** with deterministic snapshots
- **Walk-forward backtesting** with strict no-leakage validation  
- **Elo-based baseline** with advanced ML models
- **Probability calibration** using isotonic regression
- **Simple web UI** for browsing predictions and edges
- **Friday 6PM ET snapshot timing** for consistent predictions

## Data Sources

- **Games & Stats**: nfl_data_py
- **Odds**: TheOddsAPI-style JSON feed
- **Weather**: Meteostat
- **Venues**: Static JSON mapping

## Models

- **WP**: Logistic Regression on Elo diff + features
- **ATS & O/U**: Gradient Boosted Trees (XGBoost/LightGBM)
- **Calibration**: Isotonic Regression within seasons