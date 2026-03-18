# NFL Prediction System

## What This Is

An NFL game prediction system that generates pre-game Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions. Built as a personal tool for informed NFL analysis, doubling as a portfolio project, with architecture that keeps the door open for a future public-facing product.

## Core Value

Produce trustworthy, well-calibrated NFL predictions backed by rigorous methodology — when the model says 70%, it should win ~70% of the time, and over a season it should find real edges against the market.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — existing code has never been run end-to-end; full audit needed before anything can be considered validated)

### Active

- [ ] Sound modeling methodology — features, model choices, and calibration approach that are defensible and grounded in sports analytics best practices
- [ ] Reliable end-to-end pipeline — ingest data, build features, train models, generate predictions in a single reproducible flow
- [ ] Accurate, well-calibrated predictions for all three targets (WP, ATS, O/U)
- [ ] Positive CLV (closing line value) in backtests — model finds real edges vs the market
- [ ] Profitable backtests with realistic bet sizing across 2018-2024 seasons
- [ ] Clean, maintainable codebase — no broad exception swallowing, no silent fallbacks, proper error handling
- [ ] Walk-forward backtesting that rigorously proves model performance with no data leakage
- [ ] Web UI with quick-glance dashboard and drill-down analysis views
- [ ] FastAPI backend serving predictions, historical performance, and model diagnostics
- [ ] Automated weekly workflow — Friday snapshot through prediction generation
- [ ] Production-quality code suitable for a portfolio (clean architecture, good tests, documentation)

### Out of Scope

- Real-time in-game predictions — pre-game only
- Player-level prop bets — team-level game outcomes only
- Daily fantasy optimization — this is a game prediction tool, not a DFS optimizer
- Mobile native app — web-first, responsive design is sufficient

## Context

**Current state:** A substantial codebase exists (~Phase 9.4 of a prior development effort) with a three-layer data pipeline (Bronze/Silver/Gold), three trained models, walk-forward backtesting, a FastAPI API with 30+ endpoints, and a Tailwind CSS web UI. However, the pipeline has never been run end-to-end, and the owner is uncertain whether the modeling approach is sound.

**What needs to happen:** A full audit of the existing system — evaluate the modeling methodology, identify what code is worth keeping vs rewriting, fix tech debt and code quality issues, then get everything working correctly and validated.

**Known issues from codebase mapping:**
- Broad `except Exception` handling masks failures across 7+ files
- Silent DuckDB fallback returns empty results instead of errors
- Fallback HTML reports show fake data instead of clear error states
- Synchronous DB queries with no connection pooling
- Full DataFrame loads where DuckDB WHERE clauses would suffice
- Model artifacts loaded fresh each prediction (no caching)
- Datetime timezone assumptions may be incorrect
- Season/week inference uses hardcoded date offsets

**Existing tech stack:** Python, pandas, DuckDB/Parquet, scikit-learn, XGBoost, FastAPI, Jinja2/Tailwind CSS

**Data sources in codebase:** nfl_data_py (game data), The Odds API (odds), Meteostat (weather)

## Constraints

- **Methodology**: Walk-forward validation only — no random CV, no future data leakage
- **Snapshot timing**: Friday 6 PM ET for odds/data freeze (aligns with betting market timing)
- **Reproducibility**: Any prediction must be reproducible given the same input data snapshot
- **Architecture**: Must support future transition to multi-user product without major rewrite

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full audit before new features | Owner hasn't validated existing code works; building on uncertain foundation is risky | — Pending |
| Keep existing data layer pattern (Bronze/Silver/Gold) | Well-established pattern, good separation of concerns | — Pending |
| Re-evaluate model choices during audit | Current choices (LogReg for WP, XGBoost for ATS/OU) may not be optimal | — Pending |
| Python-only stack | Existing codebase, owner's language, rich ML ecosystem | — Pending |

---
*Last updated: 2026-03-18 after initialization*
