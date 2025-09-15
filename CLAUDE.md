# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL Prediction System that generates pre-game predictions for NFL games including Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions. The system is designed with a reproducible data pipeline, modeling stack, walk-forward backtesting, and a simple web UI.

**Current Status**: Project is in planning phase with comprehensive PRD but no implementation yet.


## Persistent Instructions for Claude

* Always read entire files. Otherwise, you don’t know what you don’t know, and will end up making mistakes, duplicating code that already exists, or misunderstanding the architecture.
* Commit early and often. When working on large tasks, your task could be broken down into multiple logical milestones. After a certain milestone is completed and confirmed to be ok by the user, you should commit it. If you do not, if something goes wrong in further steps, we would need to end up throwing away all the code, which is expensive and time consuming.
* Your internal knowledgebase of libraries might not be up to date. When working with any external library, unless you are 100% sure that the library has a super stable interface, you will look up the latest syntax and usage via either Perplexity (first preference) or web search (less preferred, only use if Perplexity is not available)
* Do not say things like: “x library isn’t working so I will skip it”. Generally, it isn’t working because you are using the incorrect syntax or patterns. This applies doubly when the user has explicitly asked you to use a specific library, if the user wanted to use another library they wouldn’t have asked you to use a specific one in the first place.
* Always run linting after making major changes. Otherwise, you won’t know if you’ve corrupted a file or made syntax errors, or are using the wrong methods, or using methods in the wrong way.
* Please organise code into separate files wherever appropriate, and follow general coding best practices about variable naming, modularity, function complexity, file sizes, commenting, etc.
* Code is read more often than it is written, make sure your code is always optimised for readability
* Unless explicitly asked otherwise, the user never wants you to do a “dummy” implementation of any given task. Never do an implementation where you tell the user: “This is how it *would* look like”. Just implement the thing.
* Whenever you are starting a new task, it is of utmost importance that you have clarity about the task. You should ask the user follow up questions if you do not, rather than making incorrect assumptions.
* Do not carry out large refactors unless explicitly instructed to do so.
* When starting on a new task, you should first understand the current architecture, identify the files you will need to modify, and come up with a Plan. In the Plan, you will think through architectural aspects related to the changes you will be making, consider edge cases, and identify the best approach for the given task. Get your Plan approved by the user before writing a single line of code.
* If you are running into repeated issues with a given task, figure out the root cause instead of throwing random things at the wall and seeing what sticks, or throwing in the towel by saying “I’ll just use another library / do a dummy implementation”.
* You are an incredibly talented and experienced polyglot with decades of experience in diverse areas such as software architecture, system design, development, UI & UX, copywriting, and more.
* When doing UI & UX work, make sure your designs are both aesthetically pleasing, easy to use, and follow UI / UX best practices. You pay attention to interaction patterns, micro-interactions, and are proactive about creating smooth, engaging user interfaces that delight users.
* When you receive a task that is very large in scope or too vague, you will first try to break it down into smaller subtasks. If that feels difficult or still leaves you with too many open questions, push back to the user and ask them to consider breaking down the task for you, or guide them through that process. This is important because the larger the task, the more likely it is that things go wrong, wasting time and energy for everyone involved.
* If I am ever wrong, please point it out, I need honest feedback on my code.
* Do not use Emojis in code


## Architecture

The system follows a data pipeline architecture:
- **External Sources** → **Ingestion (ETL)** → **Data Lake (Parquet/DuckDB)**
- **Feature Build** → **Models/Train + Backtests** → **Predictions API (FastAPI)** → **Web UI**

## Data Storage Structure

- `data/bronze/` - Raw data snapshots
- `data/silver/` - Cleaned tables (games, odds, weather, team_stats)  
- `data/gold/` - Feature matrices per target (wp, ats, ou)
- `outputs/` - Prediction artifacts
- `artifacts/` - Model artifacts

## Key Components (Planned)

### Data Sources
- Games & Team Performance: via `nfl_data_py`
- Odds: TheOddsAPI-style JSON feed
- Weather: Meteostat JSON/Python
- Stadium/Meta: Static JSON mapping

### Models
- **WP**: Logistic Regression on Elo diff + features
- **ATS & O/U**: Gradient Boosted Trees (XGBoost/LightGBM)
- **Calibration**: Isotonic Regression

### Key Features
- Elo ratings with margin-of-victory and dynamic K
- Rolling 4-week EPA/play and success rates
- Weather conditions (wind, temp, precip)
- Market anchors (opening/current lines)
- Context (rest days, travel, venue type)

## Development Commands (Planned)

Once implemented, the project will use these commands:

```bash
# Data ingestion & feature building
python scripts/ingest_games.py --season 2025 --week current
python scripts/ingest_odds.py --snapshot "2025-10-10T18:00:00-04:00"  
python scripts/ingest_weather.py --season 2025 --week current
python scripts/build_features.py --season 2025 --week current

# Model training & prediction
python models/train_wp.py --season 2025 --week current
python models/train_ats.py --season 2025 --week current  
python models/train_ou.py --season 2025 --week current

# Backtesting
make backtest  # Runs walk-forward 2018-2024

# Data snapshot
make snapshot  # Produces silver/gold tables for current week

# Predictions
make predict   # Writes prediction artifacts

# Web UI/API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

- Configuration via `conf/config.yaml`
- Secrets in `.env` (never commit)
- Snapshot time: Friday 6:00 PM ET (default)
- Backtest seasons: 2018-2024
- Default edge threshold: 2.0%

## Key Constraints

### Data Leakage Prevention
- Use only data strictly **before** prediction week for form metrics
- No opponent post-game stats for current week
- Use odds only from snapshot time (Friday 6 PM ET)
- Walk-forward validation only (never random CV)

### Model Training Protocol  
- Walk-forward: For season Y, train on ≤ Y-1, validate on Y
- Time-split validation only
- Calibration within each season using held-out folds

## Testing Strategy (Planned)

- Unit tests: Feature builders, Elo updates, probability transforms
- Integration tests: End-to-end dry run on past week
- Backtest tests: Spot-check known season vs frozen artifacts
- UI tests: Endpoint contracts + HTML snapshot tests

## Dependencies (Planned)

**Core**: pandas, numpy, pyarrow, duckdb  
**Modeling**: scikit-learn, xgboost, statsmodels  
**API/UI**: fastapi, uvicorn, jinja2, pydantic  
**Data Sources**: nfl_data_py, meteostat  
**HTTP/ETL**: httpx, tenacity, pytz

## Important Notes

- All predictions are for research/entertainment only
- System designed for Friday 6 PM ET snapshot timing
- Emphasis on reproducibility and no-leakage validation
- Conservative edge thresholds for betting viability
- Mobile-friendly UI with CSV download capabilities