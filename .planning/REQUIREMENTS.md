# Requirements: NFL Prediction System

**Defined:** 2026-03-18
**Core Value:** Produce trustworthy, well-calibrated NFL predictions backed by rigorous methodology -- when the model says 70%, it should win ~70% of the time, and over a season it should find real edges against the market.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation & Tooling

- [x] **FOUN-01**: Replace deprecated nfl_data_py with nflreadpy for all NFL data ingestion
- [x] **FOUN-02**: Replace deprecated Meteostat with Open-Meteo for weather data
- [ ] **FOUN-03**: Replace broad `except Exception` handling with specific exception types across all files
- [ ] **FOUN-04**: Remove silent DuckDB fallback -- fail loudly with clear error messages instead of returning empty results
- [ ] **FOUN-05**: Remove fallback HTML report generators -- return structured errors instead of fake data
- [x] **FOUN-06**: Upgrade XGBoost to 3.2+, scikit-learn to 1.8+, DuckDB to 1.3+
- [x] **FOUN-07**: Migrate package management from pip to uv with lockfile
- [x] **FOUN-08**: Replace black + isort + flake8 with Ruff for linting/formatting
- [ ] **FOUN-09**: Add HTMX 2.0 for interactive UI without JavaScript build toolchain
- [ ] **FOUN-10**: Migrate to Tailwind CSS v4 standalone CLI (eliminate Node.js dependency)
- [x] **FOUN-11**: Remove unused dependencies (polars, lightgbm, statsmodels, seaborn, pytz, pendulum)
- [x] **FOUN-12**: Pin all dependency versions in pyproject.toml for reproducibility

### Data Pipeline

- [ ] **DATA-01**: Canonical team abbreviation mapping module handling all historical variations (JAC/JAX, STL/LA, SD/LAC, OAK/LV, WSH/WAS)
- [ ] **DATA-02**: Hard-fail data quality gates at Bronze-to-Silver boundary (schema validation, null checks, type enforcement)
- [ ] **DATA-03**: Hard-fail data quality gates at Silver-to-Gold boundary (feature completeness, range checks, temporal ordering)
- [ ] **DATA-04**: Integration tests confirming complete game data for all 32 teams across 2018-2024 seasons
- [ ] **DATA-05**: Idempotent Bronze ingestion from nflreadpy with append-only timestamped snapshots
- [ ] **DATA-06**: Idempotent Bronze ingestion from The Odds API with Friday 6 PM ET snapshot timing
- [ ] **DATA-07**: Idempotent Bronze ingestion from Open-Meteo with venue-based coordinate lookups
- [ ] **DATA-08**: Silver layer transformation producing clean games, odds_snapshot, weather, and team_stats tables

### Feature Engineering -- Audit & Correction

- [ ] **FEAT-01**: Time-fence abstraction -- every feature builder enforces an `as_of_datetime` cutoff parameter with no exceptions
- [ ] **FEAT-02**: Elo audit -- validate sequential update correctness, reduce HFA from 65 to 48-55 Elo points, verify MOV adjustment and season carryover (75/25)
- [ ] **FEAT-03**: Elo add divisional game HFA reduction (~50% cut for divisional matchups)
- [ ] **FEAT-04**: EPA/play audit -- validate rolling window has no off-by-one leakage, implement dynamic window sizing (more prior-season data early, more current-season data late)
- [ ] **FEAT-05**: Market anchors consolidation -- compress from ~15 line movement features to 4-5 (spread_movement, total_movement, ml_prob_change, significant_movement)
- [ ] **FEAT-06**: Market anchors audit -- validate no closing-line contamination, Friday 6 PM ET snapshot only
- [ ] **FEAT-07**: Weather compression -- reduce from ~30 sub-features to 3-5 for model input (weather_severity, wind_mph, is_precipitation, is_outdoor, is_snow)
- [ ] **FEAT-08**: Weather features applied to outdoor games only -- zero all weather features for indoor/dome games
- [ ] **FEAT-09**: Remove anti-features from model input -- turnover margin, raw rushing yards volume, penalty stats, win/loss streaks
- [ ] **FEAT-10**: Leakage validation as hard pipeline gate -- feature validation blocks pipeline if temporal violations detected
- [ ] **FEAT-11**: Replace within-season Z-score normalization with expanding-window normalization to prevent future-data leakage

### Feature Engineering -- New Features

- [ ] **FEAT-12**: QB adjustment feature -- detect QB starter changes and compute QB quality metric (QB Elo, rolling QB EPA, or CPOE-based)
- [ ] **FEAT-13**: QB starter tracking data source -- ingest injury reports or depth charts to detect QB changes before games
- [ ] **FEAT-14**: Opponent-adjusted efficiency metrics -- adjust EPA/play for schedule difficulty
- [ ] **FEAT-15**: CPOE (Completion % Over Expected) as rolling team-level metric from play-by-play data
- [ ] **FEAT-16**: Win totals prior -- ingest preseason win totals for initial Elo calibration (fixes early-season cold start)
- [ ] **FEAT-17**: Season-week position features -- week_number, early_season flag (weeks 1-4), late_season flag (weeks 14+)
- [ ] **FEAT-18**: Surface type mismatch feature -- flag when visiting team's home surface differs from game venue
- [ ] **FEAT-19**: Divisional game indicator as explicit model feature
- [ ] **FEAT-20**: Drive starting field position for O/U model (average drive start yard line)
- [ ] **FEAT-21**: Pace of play indicators for O/U model (neutral-situation plays per game)

### Modeling

- [ ] **MODL-01**: Walk-forward temporal validation only -- train on seasons <= Y-1, validate on Y, never random CV
- [ ] **MODL-02**: WP model -- Logistic Regression with isotonic or Platt calibration
- [ ] **MODL-03**: ATS model -- XGBoost regression for margin prediction
- [ ] **MODL-04**: O/U model -- XGBoost regression for total points prediction
- [ ] **MODL-05**: CLV (Closing Line Value) as primary evaluation metric, not win/loss ROI
- [ ] **MODL-06**: Calibration curves verified -- when model says X%, outcome occurs ~X% of the time (ECE < 3%)
- [ ] **MODL-07**: Feature count enforcement -- WP ~16, ATS ~22, O/U ~21 features maximum
- [ ] **MODL-08**: Baseline comparison -- measure model performance against market closing line accuracy
- [ ] **MODL-09**: Market reversion/blend -- intelligent model-market blending with tuned weights (50-70% model / 30-50% market)
- [ ] **MODL-10**: Blend weights tuned on pre-backtest period (2010-2017), not on backtest data
- [ ] **MODL-11**: Edge threshold calibration with Kelly criterion bet sizing and sanity checks (max 30% games flagged/week)
- [ ] **MODL-12**: Flat-stake baseline comparison alongside Kelly to validate calibration
- [ ] **MODL-13**: Three-fold temporal split -- train / hyperparameter validation / holdout -- never tune on reporting data

### Backtesting & Reporting

- [ ] **BACK-01**: Walk-forward backtest across 2021-2024 validation seasons with strict temporal isolation
- [ ] **BACK-02**: CLV tracking as headline backtest metric
- [ ] **BACK-03**: Calibration curve reporting with Brier score decomposition
- [ ] **BACK-04**: Season-by-season breakdown of all metrics
- [ ] **BACK-05**: COVID season (2020) flagged for special HFA analysis
- [ ] **BACK-06**: 16-vs-17 game era normalization (week number as proportion of season)
- [ ] **BACK-07**: Betting simulation with slippage modeling and realistic vig
- [ ] **BACK-08**: Flat-stake vs Kelly criterion comparison in betting simulation
- [ ] **BACK-09**: HTML reports with interactive charts (Plotly)
- [ ] **BACK-10**: CSV export of all backtest results and predictions

### API & Web UI

- [ ] **UIAP-01**: API serves precomputed prediction artifacts only -- no model inference in the request path
- [ ] **UIAP-02**: This week's predictions dashboard -- all games with WP, ATS, O/U predictions and confidence levels
- [ ] **UIAP-03**: Historical performance view -- season-by-season accuracy, CLV trends, calibration charts
- [ ] **UIAP-04**: Game detail drill-down -- individual game analysis with feature contributions and model explanations
- [ ] **UIAP-05**: Backtest results dashboard with interactive charts
- [ ] **UIAP-06**: CSV and JSON export for predictions and historical data
- [ ] **UIAP-07**: Mobile-responsive design
- [ ] **UIAP-08**: HTMX-powered interactive filtering and updates without full page reloads

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Features

- **ADV-01**: Individual player impact modeling beyond QB position
- **ADV-02**: Real-time odds streaming and live line monitoring
- **ADV-03**: Advanced special teams DVOA integration
- **ADV-04**: Social media sentiment analysis
- **ADV-05**: Player-level injury impact modeling (beyond QB starter status)

### Platform

- **PLAT-01**: Multi-user authentication and personalized dashboards
- **PLAT-02**: Email/push notifications for high-edge predictions
- **PLAT-03**: Public API with rate limiting for external consumers
- **PLAT-04**: Mobile native app

## Out of Scope

| Feature | Reason |
|---------|--------|
| In-game / live predictions | Pre-game only system -- different architecture, different data needs |
| Daily fantasy optimization | Different problem domain -- this is game outcome prediction |
| Player prop bets | Team-level outcomes only -- player props require roster-level modeling |
| Real-time chat/community | Not a social platform |
| Automated bet placement | Legal and regulatory complexity; predictions are informational only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUN-01 | Phase 1 | Complete |
| FOUN-02 | Phase 1 | Complete |
| FOUN-03 | Phase 1 | Pending |
| FOUN-04 | Phase 1 | Pending |
| FOUN-05 | Phase 1 | Pending |
| FOUN-06 | Phase 1 | Complete |
| FOUN-07 | Phase 1 | Complete |
| FOUN-08 | Phase 1 | Complete |
| FOUN-09 | Phase 1 | Pending |
| FOUN-10 | Phase 1 | Pending |
| FOUN-11 | Phase 1 | Complete |
| FOUN-12 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-05 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| DATA-07 | Phase 2 | Pending |
| DATA-08 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Pending |
| FEAT-02 | Phase 3 | Pending |
| FEAT-03 | Phase 3 | Pending |
| FEAT-04 | Phase 3 | Pending |
| FEAT-05 | Phase 3 | Pending |
| FEAT-06 | Phase 3 | Pending |
| FEAT-07 | Phase 3 | Pending |
| FEAT-08 | Phase 3 | Pending |
| FEAT-09 | Phase 3 | Pending |
| FEAT-10 | Phase 3 | Pending |
| FEAT-11 | Phase 3 | Pending |
| FEAT-12 | Phase 5 | Pending |
| FEAT-13 | Phase 5 | Pending |
| FEAT-14 | Phase 5 | Pending |
| FEAT-15 | Phase 5 | Pending |
| FEAT-16 | Phase 5 | Pending |
| FEAT-17 | Phase 5 | Pending |
| FEAT-18 | Phase 5 | Pending |
| FEAT-19 | Phase 5 | Pending |
| FEAT-20 | Phase 5 | Pending |
| FEAT-21 | Phase 5 | Pending |
| MODL-01 | Phase 4 | Pending |
| MODL-02 | Phase 4 | Pending |
| MODL-03 | Phase 4 | Pending |
| MODL-04 | Phase 4 | Pending |
| MODL-05 | Phase 4 | Pending |
| MODL-06 | Phase 4 | Pending |
| MODL-07 | Phase 4 | Pending |
| MODL-08 | Phase 4 | Pending |
| MODL-09 | Phase 7 | Pending |
| MODL-10 | Phase 7 | Pending |
| MODL-11 | Phase 7 | Pending |
| MODL-12 | Phase 7 | Pending |
| MODL-13 | Phase 4 | Pending |
| BACK-01 | Phase 6 | Pending |
| BACK-02 | Phase 6 | Pending |
| BACK-03 | Phase 6 | Pending |
| BACK-04 | Phase 6 | Pending |
| BACK-05 | Phase 6 | Pending |
| BACK-06 | Phase 6 | Pending |
| BACK-07 | Phase 6 | Pending |
| BACK-08 | Phase 6 | Pending |
| BACK-09 | Phase 6 | Pending |
| BACK-10 | Phase 6 | Pending |
| UIAP-01 | Phase 8 | Pending |
| UIAP-02 | Phase 8 | Pending |
| UIAP-03 | Phase 8 | Pending |
| UIAP-04 | Phase 8 | Pending |
| UIAP-05 | Phase 8 | Pending |
| UIAP-06 | Phase 8 | Pending |
| UIAP-07 | Phase 8 | Pending |
| UIAP-08 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 72 total
- Mapped to phases: 72
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after roadmap creation*
