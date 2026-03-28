# NFL Prediction System

## What This Is

An NFL game prediction system that generates pre-game Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions. Built as a personal tool for informed NFL analysis, doubling as a portfolio project, with architecture that keeps the door open for a future public-facing product. Shipped v1.0 MVP with end-to-end pipeline: data ingestion through feature engineering, walk-forward model training, backtest reporting, market blending, and a FastAPI + HTMX web dashboard.

## Core Value

Produce trustworthy, well-calibrated NFL predictions backed by rigorous methodology -- when the model says 70%, it should win ~70% of the time, and over a season it should find real edges against the market.

## Current State

**Shipped:** v1.0 MVP (2026-03-28)

**What was built:**
- Modern Python toolchain (uv/Ruff/pyright) with Tailwind v4 + HTMX 2.0
- Reliable Bronze/Silver/Gold data pipeline with Pydantic v2 quality gates and canonical team mapping
- Temporal-safe feature engineering with FeatureBuilder Protocol, LeakageGate, and compressed features (weather 38->4, market 40->5)
- Walk-forward trained models: WP (LogReg+isotonic), ATS/O/U (XGBoost regression) with CLV as primary metric
- Full backtest system (2021-2024) with Brier decomposition, betting simulation, and interactive HTML reports
- Market blending (log-odds WP, linear ATS/O/U) with pre-2018 weight tuning and edge calibration
- FastAPI web dashboard with HTMX interactions, game detail drill-down, feature importance charts, CSV/JSON exports
- DuckDB cache with 1139 predictions and 1991 game contexts

**Codebase:** ~35.5K LOC Python (production), ~72K LOC Python (tests), ~44K LOC HTML/templates

**Tech stack:** Python 3.13 (uv + Ruff + pyright), pandas, DuckDB/Parquet, scikit-learn, XGBoost, FastAPI, Jinja2/Tailwind CSS v4, HTMX 2.0

## Requirements

### Validated

- Canonical team mapping as single source of truth with hard-fail on unknown abbreviations -- v1.0
- Bronze/Silver data layer pattern with timestamped append-only snapshots and latest-wins upsert -- v1.0
- Quality gates at Bronze-to-Silver and Silver-to-Gold boundaries with Pydantic v2 schema validation -- v1.0
- 32-team data completeness across 2018-2024 proven by integration tests -- v1.0
- Sound feature engineering with temporal safety (as_of_datetime enforcement), leakage gate, and expanding-window normalization -- v1.0
- Sound modeling methodology -- walk-forward temporal validation, LogReg+calibration for WP, XGBoost regression for ATS/OU, CLV computation, market baseline comparison -- v1.0
- Differentiating features -- QB adjustment, opponent-adjusted EPA, CPOE, contextual features integrated and validated -- v1.0
- Model-market blending with temporal isolation -- log-odds WP blending, linear ATS/O/U interpolation, weights tuned on pre-2018 data only, edge threshold calibration with 30% cap -- v1.0
- Disciplined bet sizing -- flat-stake baseline alongside Kelly sizing, confirming Kelly does not merely amplify noise -- v1.0
- Walk-forward backtesting that rigorously proves model performance with no data leakage -- v1.0
- Clean, maintainable codebase -- no broad exception swallowing, no silent fallbacks, proper error handling -- v1.0
- Web UI with quick-glance dashboard and drill-down analysis views -- v1.0
- FastAPI backend serving predictions, historical performance, and model diagnostics -- v1.0
- API serves precomputed artifacts only -- no model inference in request path -- v1.0

### Active

- [ ] Accurate, well-calibrated predictions for all three targets (WP, ATS, O/U) -- ongoing validation
- [ ] Positive CLV (closing line value) in backtests -- model finds real edges vs the market
- [ ] Profitable backtests with realistic bet sizing across 2018-2024 seasons
- [ ] Automated weekly workflow -- Friday snapshot through prediction generation
- [ ] Production-quality code suitable for a portfolio (clean architecture, good tests, documentation)
- [ ] Reliable end-to-end pipeline -- ingest data, build features, train models, generate predictions in a single reproducible flow

### Out of Scope

- Real-time in-game predictions -- pre-game only
- Player-level prop bets -- team-level game outcomes only
- Daily fantasy optimization -- this is a game prediction tool, not a DFS optimizer
- Mobile native app -- web-first, responsive design is sufficient
- Automated bet placement -- legal complexity; predictions are informational only

## Context

**Data sources:** nflreadpy (game data), The Odds API (odds), Open-Meteo (weather)

**Known tech debt (from v1.0 audit):**
- Synchronous DB queries with no connection pooling
- Full DataFrame loads where DuckDB WHERE clauses would suffice
- Model artifacts loaded fresh each prediction (no caching)
- Season/week inference uses hardcoded date offsets
- All Elo values show 1500.0 in game detail (gold features never had real Elo computed)
- Legacy templates (recommendations.html, games.html, calibration.html) not cleaned up

## Constraints

- **Methodology**: Walk-forward validation only -- no random CV, no future data leakage
- **Snapshot timing**: Friday 6 PM ET for odds/data freeze (aligns with betting market timing)
- **Reproducibility**: Any prediction must be reproducible given the same input data snapshot
- **Architecture**: Must support future transition to multi-user product without major rewrite

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full audit before new features | Owner hadn't validated existing code works; building on uncertain foundation is risky | Validated -- systematic audit across 10 phases found and fixed hundreds of issues |
| Keep existing data layer pattern (Bronze/Silver/Gold) | Well-established pattern, good separation of concerns | Validated -- Bronze snapshots append-only, Silver latest-wins upsert, quality gates at boundaries |
| Re-evaluate model choices during audit | Current choices (LogReg for WP, XGBoost for ATS/OU) may not be optimal | Validated -- LogReg+isotonic for WP, XGBoost regression for ATS/OU confirmed as sound |
| Python-only stack | Existing codebase, owner's language, rich ML ecosystem | Validated -- no need for additional languages |
| LA (not LAR) as canonical Rams abbreviation | Aligns with nflreadpy primary data source | Validated -- consistent across all data sources |
| FeatureBuilder Protocol (not ABC) | Structural subtyping without forcing inheritance changes | Validated -- all 5 builders conform |
| Log-odds blending for WP, linear for ATS/O/U | Probability-correct interpolation in log-odds space | Validated -- blend improves CLV |
| Pre-2018 data for blend weight tuning | Strict temporal isolation from backtest period | Validated -- no information leakage |
| Precomputed artifacts for API (no model inference) | Keeps API fast, simple, and decoupled from model code | Validated -- UIAP-01 compliance enforced by import guard test |

---
*Last updated: 2026-03-28 after v1.0 milestone*
