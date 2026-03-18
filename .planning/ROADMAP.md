# Roadmap: NFL Prediction System

## Overview

Transform the existing NFL prediction codebase from an unvalidated partial implementation into a trustworthy, end-to-end system that produces well-calibrated WP, ATS, and O/U predictions with demonstrable CLV against the market. The journey follows the dependency chain: fix the foundation (deprecated deps, silent failures), harden the data pipeline, audit feature engineering for leakage, establish a baseline model with walk-forward validation, add differentiating features (QB adjustment, opponent-adjusted efficiency), build the full backtest reporting system, layer in market reversion and edge calibration, and finally serve everything through a clean API and web UI.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation Hardening** - Replace deprecated dependencies, fix silent failure modes, modernize toolchain (completed 2026-03-18)
- [ ] **Phase 2: Data Pipeline Audit** - Reliable ingestion, canonical team mapping, hard-fail quality gates
- [ ] **Phase 3: Feature Engineering Correctness** - Audit existing features for leakage, enforce time-fence abstraction, compress overengineered features
- [ ] **Phase 4: Core Model Training** - Walk-forward training of WP/ATS/O/U models with correct temporal validation protocol
- [ ] **Phase 5: Differentiating Features** - QB adjustment, opponent-adjusted efficiency, CPOE, and other high-impact new features
- [ ] **Phase 6: Backtest Reporting** - Full walk-forward backtest system with CLV tracking, era analysis, and interactive reports
- [ ] **Phase 7: Market Reversion and Edge Calibration** - Model-market blending, Kelly sizing, edge threshold tuning
- [ ] **Phase 8: API and Web UI** - FastAPI serving precomputed artifacts, HTMX dashboard, mobile-responsive design

## Phase Details

### Phase 1: Foundation Hardening
**Goal**: The codebase has no deprecated dependencies, no silent failure modes, and a modern Python toolchain -- every subsequent phase builds on a clean, honest foundation
**Depends on**: Nothing (first phase)
**Requirements**: FOUN-01, FOUN-02, FOUN-03, FOUN-04, FOUN-05, FOUN-06, FOUN-07, FOUN-08, FOUN-09, FOUN-10, FOUN-11, FOUN-12
**Success Criteria** (what must be TRUE):
  1. Running any script that previously used nfl_data_py now uses nflreadpy and successfully fetches game data for the 2024 season
  2. Running any script that previously used Meteostat now uses Open-Meteo and successfully fetches weather data for a known NFL game venue
  3. Every `except` block in the codebase catches a specific exception type -- no bare `except Exception` remains in any production code file
  4. A DuckDB query failure (e.g., missing table, bad SQL) raises an immediate error with a descriptive message instead of returning empty results
  5. `uv sync` installs all dependencies from a locked pyproject.toml, and `ruff check .` passes with zero violations on the entire codebase
**Plans**: 4 plans

Plans:
- [x] 01-01-PLAN.md -- Toolchain modernization: uv, Ruff, pyright, dependency cleanup, version upgrades, exception hierarchy
- [x] 01-02-PLAN.md -- Data library migration: nfl_data_py to nflreadpy, Meteostat to Open-Meteo
- [x] 01-03-PLAN.md -- Error handling hardening: except Exception removal, DuckDB fail-loud, HTML fallback removal
- [x] 01-04-PLAN.md -- Frontend toolchain and cleanup: Tailwind v4 standalone, HTMX 2.0, dead file removal, pytz elimination

### Phase 2: Data Pipeline Audit
**Goal**: Data flows reliably from all three sources through Bronze and Silver layers with hard-fail quality gates, canonical team names, and verified completeness for all 32 teams across 2018-2024
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08
**Success Criteria** (what must be TRUE):
  1. A canonical team abbreviation lookup resolves every historical variant (JAC/JAX, STL/LA, SD/LAC, OAK/LV, WSH/WAS) to a single standard abbreviation, and an integration test confirms all 32 teams have complete game records for each season 2018-2024
  2. Running Bronze-to-Silver transformation on data with a missing required column or null value in a required field causes an immediate hard failure with a descriptive error -- not a silent empty result
  3. Running Silver-to-Gold transformation on data with out-of-range feature values or temporal ordering violations causes an immediate hard failure
  4. Re-running any ingestion script for the same season/week produces identical output (idempotent) and Bronze snapshots are append-only with timestamps
  5. Silver layer contains clean, queryable games, odds_snapshot, weather, and team_stats tables for 2018-2024 with no team abbreviation mismatches across tables
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD
- [ ] 02-03: TBD

### Phase 3: Feature Engineering Correctness
**Goal**: Every existing feature builder is audited for temporal correctness, overengineered features are compressed, and a hard leakage gate blocks the pipeline if any temporal violation is detected
**Depends on**: Phase 2
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, FEAT-07, FEAT-08, FEAT-09, FEAT-10, FEAT-11
**Success Criteria** (what must be TRUE):
  1. Every feature builder function accepts an `as_of_datetime` parameter and produces identical output for the same cutoff -- a test confirms that features computed for Week 8 of 2023 use only data from before that week's games
  2. Elo ratings update sequentially with corrected HFA (48-55 points), divisional game HFA reduction, and a test confirms the Elo for a specific team at a specific week matches an independently calculated value
  3. The Gold feature matrix for WP contains no more than 16 features, ATS no more than 22, and O/U no more than 21 -- with weather compressed to 3-5 features, market anchors to 4-5, and anti-features (turnover margin, raw rushing yards, penalty stats, win streaks) removed
  4. Running the feature pipeline with an intentionally introduced temporal violation (e.g., a future game's stats leaking into a feature) triggers a hard failure from the leakage validation gate before any model can train
  5. Normalization uses expanding-window only (no within-season Z-scores that see future data)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Core Model Training
**Goal**: Three models (WP, ATS, O/U) are trained with strict walk-forward temporal validation, producing calibrated predictions with CLV as the primary evaluation metric and a baseline comparison against the market
**Depends on**: Phase 3
**Requirements**: MODL-01, MODL-02, MODL-03, MODL-04, MODL-05, MODL-06, MODL-07, MODL-08, MODL-13
**Success Criteria** (what must be TRUE):
  1. WP model (Logistic Regression + calibration) trained on seasons up to Y-1 produces predictions for season Y, and a calibration curve shows predicted probabilities track actual win rates (ECE < 5% on initial baseline, targeting < 3% after tuning)
  2. ATS model (XGBoost) and O/U model (XGBoost) produce margin and total predictions respectively, each trained with strict walk-forward splits and a three-fold temporal split (train / hyperparameter validation / holdout)
  3. CLV is computed for every prediction across 2021-2024 validation seasons and reported as the headline metric -- win/loss ROI is secondary
  4. A baseline comparison report shows how each model performs against simply using the market closing line (the market-only baseline)
  5. No model training step uses random cross-validation -- every split is temporal and verifiable from the training logs
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Differentiating Features
**Goal**: The highest-impact non-market features are added to the system -- QB adjustment (worth 3-7 spread points), opponent-adjusted efficiency, CPOE, and contextual features -- with validated lift measured against the Phase 4 baseline
**Depends on**: Phase 4
**Requirements**: FEAT-12, FEAT-13, FEAT-14, FEAT-15, FEAT-16, FEAT-17, FEAT-18, FEAT-19, FEAT-20, FEAT-21
**Success Criteria** (what must be TRUE):
  1. QB starter changes are detected from an ingested data source (injury reports or depth charts), and a QB quality metric (QB Elo, rolling QB EPA, or CPOE-based) adjusts predictions when a backup QB starts
  2. EPA/play metrics are adjusted for opponent strength, and a test confirms that a team's opponent-adjusted EPA differs from raw EPA for a season with a notably easy or hard schedule
  3. Each new feature is validated for lift: re-running the walk-forward evaluation with the feature included shows improvement (or at minimum no degradation) in CLV compared to the Phase 4 baseline
  4. Season-week position features, divisional game indicator, and surface mismatch flag are present in the Gold feature matrices
  5. Feature count constraints are still respected after additions (WP ~16, ATS ~22, O/U ~21)
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Backtest Reporting
**Goal**: A complete walk-forward backtest system runs the full pipeline (features through predictions) across 2021-2024 seasons, producing CLV-headlined reports with era analysis, betting simulation, and interactive HTML output
**Depends on**: Phase 5
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04, BACK-05, BACK-06, BACK-07, BACK-08, BACK-09, BACK-10
**Success Criteria** (what must be TRUE):
  1. A single command runs the full walk-forward backtest across 2021-2024 validation seasons with strict temporal isolation, and CLV is reported as the headline metric in the output
  2. The backtest report includes season-by-season breakdown of accuracy, CLV, Brier score, and calibration curves -- with the COVID 2020 season flagged for special HFA analysis
  3. A betting simulation report shows results with realistic vig and slippage, comparing flat-stake and Kelly criterion strategies side by side
  4. Interactive HTML reports with Plotly charts are generated and viewable in a browser, alongside CSV exports of all predictions and metrics
  5. 16-game vs 17-game era normalization is applied (week number as proportion of season length)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD
- [ ] 06-03: TBD

### Phase 7: Market Reversion and Edge Calibration
**Goal**: Model predictions are intelligently blended with market lines using weights tuned on pre-backtest data, edge thresholds are calibrated with Kelly sizing, and the system identifies genuine betting edges with disciplined constraints
**Depends on**: Phase 6
**Requirements**: MODL-09, MODL-10, MODL-11, MODL-12
**Success Criteria** (what must be TRUE):
  1. Model-market blend weights are tuned on pre-backtest period data (2010-2017 or available historical window) and applied without modification to the 2021-2024 backtest -- blend weights are never optimized on the reporting data
  2. Edge thresholds are calibrated using Kelly criterion with sanity checks: no more than 30% of games per week are flagged as edges
  3. A flat-stake baseline comparison exists alongside Kelly results, confirming that Kelly sizing does not merely amplify noise from a poorly calibrated model
  4. Re-running the full backtest with market reversion shows CLV improvement over the unblended model from Phase 6
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: API and Web UI
**Goal**: Precomputed prediction artifacts are served through a FastAPI backend and rendered in an HTMX-powered, mobile-responsive web dashboard with drill-down analysis and export capabilities
**Depends on**: Phase 7
**Requirements**: UIAP-01, UIAP-02, UIAP-03, UIAP-04, UIAP-05, UIAP-06, UIAP-07, UIAP-08
**Success Criteria** (what must be TRUE):
  1. The API serves predictions from precomputed artifact files only -- importing any model class or running inference in the API codebase causes a test failure
  2. A user visiting the dashboard sees this week's predictions for all games with WP, ATS, and O/U values plus confidence indicators, and can filter/sort without full page reloads (HTMX)
  3. A user can navigate to historical performance showing season-by-season accuracy, CLV trends, and calibration charts, and can drill into any individual game for feature contributions and model explanations
  4. A user can download predictions and historical data as CSV or JSON from any view
  5. The dashboard renders correctly and is usable on a mobile phone screen (no horizontal scrolling, touch-friendly controls)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD
- [ ] 08-03: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation Hardening | 4/4 | Complete   | 2026-03-18 |
| 2. Data Pipeline Audit | 0/TBD | Not started | - |
| 3. Feature Engineering Correctness | 0/TBD | Not started | - |
| 4. Core Model Training | 0/TBD | Not started | - |
| 5. Differentiating Features | 0/TBD | Not started | - |
| 6. Backtest Reporting | 0/TBD | Not started | - |
| 7. Market Reversion and Edge Calibration | 0/TBD | Not started | - |
| 8. API and Web UI | 0/TBD | Not started | - |
