# NFL Prediction System

A pre-game NFL prediction system that produces calibrated Win Probability (WP),
Against the Spread (ATS), and Over/Under (O/U) forecasts for every weekly matchup.
Models are trained strictly walk-forward, blended with the market in the
mathematically appropriate space (log-odds for probabilities, point space for
spreads and totals), and served from a read-only DuckDB cache by a FastAPI +
HTMX + Tailwind v4 web app.

The project is both a personal decision-support tool and a portfolio piece.
v1.0 MVP shipped on 2026-03-28; a v2.0 accuracy, automation, and polish
milestone is in flight.

---

## Overview

**What it is.** A Friday-snapshot prediction engine for NFL regular-season and
playoff games. Every Friday at 18:00 ET, the system ingests the latest games,
odds, and weather; rebuilds the feature matrices; runs three target-specific
models; blends those outputs with the closing-ish market snapshot; and
refreshes a denormalized web cache that the dashboard reads from.

**Who it serves.** A single analyst (me). Predictions are informational only,
not wagering automation; sizing and bet selection are surfaced as suggestions,
not actions.

**What problem it solves.** NFL markets are efficient but not perfect, and
the signal lives in carefully constructed, temporally-safe features combined
with disciplined market blending. The system exists to make that edge-hunting
reproducible, auditable, and honest about where it fails.

**Why it is technically interesting.**

- An end-to-end data lakehouse (Bronze / Silver / Gold Parquet + DuckDB) with
  Pydantic v2 quality gates at each layer transition.
- Temporal-safety is enforced *at three levels*: a runtime-checkable
  `FeatureBuilder` Protocol with a mandatory `as_of_datetime` time-fence,
  a `LeakageGate` that scans the combined feature matrix for post-game
  keywords, and a walk-forward splitter that hard-fails if train / validation /
  holdout seasons overlap.
- The API layer and the ML layer are architecturally quarantined. A pytest
  AST-walking import guard (`tests/api/test_import_guard.py`) fails CI if
  `api/` ever imports from `models/`, `features/`, or `ratings/`.
- Market blend weights are tuned strictly on pre-2018 seasons, so the
  2018-2024 backtest window is never seen during weight selection — a
  harder constraint than the backtest itself.

---

## Status and Scope

**Shipped (v1.0 MVP, 2026-03-28).** Data pipeline, feature engineering,
model training, walk-forward backtest, market blending, and the web
dashboard are all in production.

**In flight (v2.0 accuracy / automation / polish).** Phases 11-16 are
complete: Elo rebuild with snapshot-then-update, Optuna-driven retraining
infrastructure, `DynamicBlendWeights` (week-of-season sigmoid schedule),
a unified Friday orchestrator with retry and health gates, and data-quality
monitoring. Phases 17-18 (betting dashboard + season tracking page) are
not started.

**What is explicitly *not* in scope.** Automated bet placement (legal
complexity — informational only), in-game / real-time predictions,
player props, DFS optimization, multi-user access, native mobile apps.
None of these exist in the codebase.

**What is in code but not yet serving production.** `DynamicBlendWeights`
shipped in Phase 13 but the v2.0 retrain meant to adopt it failed
per-target gating, so the production blend artifact is
`blend_20260324_023118` (the v1.0 static blend). See "Current
Limitations" — dynamic blending is ready to activate the moment a
future retrain clears gating.

---

## Key Features

### Prediction outputs
- **Win Probability (WP)** — calibrated home-team win probability from a
  `LogisticRegression` + `StandardScaler` + `IsotonicRegression` pipeline.
  Calibration is fit on the HP-validation fold, never on training data.
- **Against the Spread (ATS)** — predicted home margin from an `XGBRegressor`,
  converted to cover probabilities via an empirical `ResidualDistributionConverter`.
- **Over/Under (O/U)** — predicted total points from an `XGBRegressor`,
  converted to over/under probabilities via a `TotalDistributionConverter`.

### Data lakehouse
- **Bronze** — append-only timestamped Parquet snapshots from every ingestion.
- **Silver** — cleaned, schema-validated, latest-wins tables, Hive-partitioned
  by `season=YYYY/` and `snapshot_ts=…` (URL-encoded ET ISO-8601), so the
  Friday 18:00 ET snapshot is a first-class storage attribute.
- **Gold** — one feature matrix per target: `features_wp.parquet`,
  `features_ats.parquet`, `features_ou.parquet`.
- **Pydantic v2 quality gates** at every layer boundary. One bad row fails
  the whole batch (intentional — silent row-skipping causes downstream
  corruption that is harder to debug than a hard failure).

### Feature engineering
Elo, rolling team form (EPA / success / pace / red zone / third-down),
market anchors (opening + Friday-snapshot lines; closing lines *never* read
into features), weather (indoor games zeroed), contextual (rest, travel,
divisional, surface mismatch), composite QB quality
(`0.7·z(rolling_qb_epa) + 0.3·z(rolling_cpoe)`), and opponent-adjusted
EPA. Normalization is **expanding-window** grouped by season with
`min_periods=4` and prior-season bootstrap, explicitly replacing the
leakier within-season Z-score approach.

### Temporal safety
Three independent layers: `features/protocol.py::FeatureBuilder`
(`@runtime_checkable` Protocol with mandatory `as_of_datetime`);
`features/validation.py::LeakageGate` (per-builder time-fence,
combined-matrix keyword scan, Elo chronological ordering — violations
raise `LeakageViolation` and emit a JSON diagnostic); and
`models/temporal.py::TemporalSplitConfig.validate()` which raises if
train / HP-validation / holdout seasons overlap or violate ordering.

### Market blending
WP blended in log-odds space via `scipy.special.logit` / `expit`;
ATS / O/U blended linearly in point space. Weights tuned strictly on
**pre-2018 seasons** (`TUNING_SEASONS` in `models/blending_data.py`) —
temporally disjoint from the 2021-2024 backtest window.
`DynamicBlendWeights` (Phase 13) adds a per-target sigmoid schedule in
week-of-season with per-target gating; implemented in code, ready to
activate when the next retrain clears gating.

### Walk-forward backtesting
Expanding-window walk-forward over 2021-2024, per-season retraining.
`backtest/metrics.py::brier_decomposition` implements Murphy (1973)
reliability / resolution / uncertainty by hand.
`backtest/simulation.py::BettingSimulator` runs flat-stake *and*
quarter-Kelly side by side, with `SLIPPAGE_POINTS = 0.5` and -110 vig on
ATS / O/U. CLV (`models/clv.py`) — probability and line — is the
primary quality metric.

### Web dashboard
FastAPI monolith with a shared read-only DuckDB connection. Jinja2
templates rendered through `jinja2-fragments`'s `Jinja2Blocks` so
full-page requests and HTMX fragment swaps share the same templates.
Four pages (**This Week**, **Performance**, **Backtest**, **Insights**),
game detail drill-down (`/games/{id}`), HTMX partials, CSV + JSON
exports, and a `/health` endpoint.

---

## Architecture

```
   +-----------------+   +------------------+   +-------------------+   +----------------+
   |  nflreadpy      |   |  The Odds API    |   |  Open-Meteo       |   |  venues.json   |
   |  (games, PBP)   |   |  (spread/ML/OU)  |   |  (hist. weather)  |   |  (static)      |
   +--------+--------+   +---------+--------+   +---------+---------+   +-------+--------+
            |                      |                       |                     |
            v                      v                       v                     v
   +---------------------------------------------------------------------------------------+
   | data/bronze/  (append-only, timestamped parquet — raw source snapshots)               |
   +---------------------------------------------------------------------------------------+
              |  validate_bronze_to_silver (Pydantic v2; 1 bad row fails the batch)
              v
   +---------------------------------------------------------------------------------------+
   | data/silver/  (Hive-partitioned: season=YYYY/, snapshot_ts=YYYY-MM-DDT18%3A00…)       |
   |   tables: games, odds_snapshot, weather, contextual, elo_game_snapshots               |
   +---------------------------------------------------------------------------------------+
              |  FeatureBuilder Protocol + LeakageGate + expanding-window normalization
              |  + validate_silver_to_gold (numeric range guards)
              v
   +---------------------------------------------------------------------------------------+
   | data/gold/  (features_wp.parquet, features_ats.parquet, features_ou.parquet)          |
   +---------------------------------------------------------------------------------------+
              |
              v
   +---------------------------------------------------------------------------------------+
   | Model training  (walk-forward; Optuna TPE + Hyperband; SQLite studies)                |
   |   WP  : LogReg + StandardScaler + IsotonicRegression (calibrated on HP-val fold)      |
   |   ATS : XGBRegressor(margin) -> ResidualDistributionConverter -> cover prob.          |
   |   O/U : XGBRegressor(total)  -> TotalDistributionConverter    -> over/under prob.     |
   |   Artifacts: artifacts/{target}_{UTCtimestamp}/ + artifacts/latest.json manifest      |
   +---------------------------------------------------------------------------------------+
              |
              v
   +---------------------------------------------------------------------------------------+
   | Market blending (models/blending.py)                                                  |
   |   WP  : log-odds interpolation  |  ATS/O/U : linear in point space                    |
   |   Weights tuned strictly on pre-2018 seasons (TUNING_SEASONS)                         |
   +---------------------------------------------------------------------------------------+
              |
              v                     [---- UIAP-01 ARCHITECTURAL BOUNDARY ----]
   +---------------------------------------------------------------------------------------+
   | scripts/populate_cache.py  ->  data/web_cache.duckdb  (read-only; ~6 MB)              |
   |   Pre-rendered Plotly HTML, prediction rows, backtest rows, game context.             |
   +---------------------------------------------------------------------------------------+
              |  api/ MUST NOT import models/, features/, ratings/ (enforced by test)
              v
   +---------------------------------------------------------------------------------------+
   | FastAPI  (api/main.py lifespan: shared read-only DuckDB; single-worker envelope)      |
   |   Pages:     /, /performance, /backtest, /insights, /games/{id}                       |
   |   HTMX:      /fragments/games, /fragments/performance                                 |
   |   Exports:   /api/export/csv, /api/export/json                                        |
   |   Health:    /health                                                                  |
   +---------------------------------------------------------------------------------------+
              |
              v
   +---------------------------------------------------------------------------------------+
   | Web UI (Jinja2Blocks + HTMX 2.0.4 + Tailwind v4 + Plotly)                             |
   +---------------------------------------------------------------------------------------+
```

The Friday 18:00 ET snapshot is both *computed* (`utils/date_utils.py::get_snapshot_time`)
and *stored as a directory partition* under `data/silver/`, which is why
any prediction is reproducible given the same input snapshot.

---

## Model and Feature Architecture

```
  +-------------------------------------------------------------+
  |  TemporalSplitConfig                                        |
  |    train seasons  <  HP-validation season  <  holdout       |
  |    (validate() raises on overlap or ordering violation)     |
  +------+----------------------+-------------------------------+
         |                      |
         v                      v
  +--------------+     +------------------+
  |  train set   |     |  HP-val set      |
  +--------------+     +------------------+
         |                      |
         +----------+-----------+
                    |
                    v
     +---------------------------------------+
     |  OptunaTuner  (TPE + Hyperband)       |
     |    resumable SQLite studies:          |
     |      data/optuna/wp_tuning_v1.db      |
     |      data/optuna/ats_tuning_v1.db     |
     |      data/optuna/ou_tuning_v1.db      |
     +-------------------+-------------------+
                         |
                         v
           +-------------+-------------+
           |  WalkForwardSplitter      |
           |    expanding-window over  |
           |    the holdout horizon    |
           +---+--------+--------+-----+
               |        |        |
               v        v        v
       +---------+ +---------+ +---------+
       |  WP     | |  ATS    | |  O/U    |
       | LogReg  | | XGB     | | XGB     |
       | + Iso   | | +Resid  | | +Total  |
       +----+----+ +----+----+ +----+----+
            |           |           |
            v           v           v
        +-----------------------------------+
        |  Market blending                  |
        |    WP  : log-odds interpolation   |
        |    ATS : linear, point space      |
        |    O/U : linear, point space      |
        |  Weights tuned on pre-2018 only   |
        +-----------------+-----------------+
                          |
                          v
        +-----------------------------------+
        |  artifacts/{target}_{ts}/         |
        |    model.pkl                      |
        |    metadata.json                  |
        |    feature_list.json              |
        |    calibrator.pkl  (WP only)      |
        |    {target}_params.json           |
        |  artifacts/latest.json (manifest) |
        +-----------------------------------+
```

`models/prediction_pipeline.py` imports *only* `load_model_artifact` — no
trainer classes on the prediction path. The manifest at `artifacts/latest.json`
points at the production versions; on 2026-04-16 those are
`wp_20260327_114739`, `ats_20260326_163724`, `ou_20260326_163930`, and
`blend_20260324_023118`.

---

## Tech Stack

| Layer | Choice | Version |
|-------|--------|---------|
| Language | Python | `>=3.12` (runtime 3.13) |
| Package / env | uv | `uv.lock` |
| Lint / format | Ruff | `>=0.9` |
| Type check | Pyright | `>=1.1.390`, `typeCheckingMode = "standard"` |
| Dataframes | pandas | `>=2.2, <3` |
| Analytics DB | DuckDB | `>=1.5, <2` |
| Columnar IO | pyarrow | `>=18.0` |
| WP model | scikit-learn `LogisticRegression` + `IsotonicRegression` | `>=1.8, <2` |
| ATS / O/U model | XGBoost `XGBRegressor` | `>=3.2, <4` |
| Hyperparameter tuning | Optuna (TPE + Hyperband, SQLite studies) | `==4.8.0` (pinned exact) |
| Web framework | FastAPI | `>=0.115, <1` |
| ASGI / production servers | uvicorn + gunicorn | uvicorn `>=0.34, <1` |
| Templating | Jinja2 + `jinja2-fragments` (`Jinja2Blocks`) | `jinja2-fragments>=1.11` |
| Frontend interactivity | HTMX 2.0.4 (CDN) | — |
| Charts | Plotly (Python pre-render + JS CDN) | `plotly>=6.0, <7` |
| Styling | Tailwind CSS v4 (vendored CLI at `tools/tailwindcss.exe`) | — |
| Data sources | nflreadpy, The Odds API, Open-Meteo | `nflreadpy>=0.1.5` |
| HTTP client / retries | httpx + tenacity | `httpx>=0.28`, `tenacity>=9.0` |
| Config | Pydantic v2 + pydantic-settings + YAML | `pydantic>=2.10` |
| Structured logging | structlog | `>=24.4, <25` |
| In-process cache | `cachetools.TTLCache` | `>=7.0.5` |
| Tests | pytest + pytest-asyncio + pytest-cov + hypothesis + beautifulsoup4 | dev deps |

Pinning Optuna to an exact version (`==4.8.0`) is deliberate: study state
persists to SQLite across runs and minor Optuna changes have historically
altered search behaviour enough to drift results.

---

## Data and Domain Model

### Layered lakehouse

| Layer | Location | Shape | Semantics |
|-------|----------|-------|-----------|
| Bronze | `data/bronze/` | Parquet | Append-only timestamped raw snapshots |
| Silver | `data/silver/` | Parquet (Hive-partitioned) | Schema-validated, latest-wins upsert |
| Gold | `data/gold/` | Parquet (one file per target) | Feature matrices ready for training |
| Web cache | `data/web_cache.duckdb` | DuckDB (~6 MB) | Read-only, API-facing |
| Analytics DB | `data/nfl_predictions.duckdb` | DuckDB (~37 MB) | Ad-hoc SQL and notebooks |

The two DuckDB files are deliberately separate. The web cache contains
only what the API needs and is rebuilt by `scripts/populate_cache.py`;
the analytics DB is for exploration and never touches the request path.

### Core Pydantic v2 schemas (`data/schemas.py`)

| Schema | Role | Notable fields |
|--------|------|----------------|
| `GameSchema` | Silver-layer games | `game_id` (e.g. `2024_W06_KC@BUF`), `season` (2000-2030), `week` (1-22), `kickoff_et`, `venue_roof: VenueRoof`, `result: GameResult` |
| `OddsSchema` | Silver-layer odds snapshot | per-book spread / total / moneylines + `snapshot_ts` |
| `WeatherSchema` | Silver-layer weather | Open-Meteo shape + `wind_direction` float + NaN→None coercion |
| `VenueRoof` (StrEnum) | `indoor` / `outdoor` / `retractable` | nflverse `"dome"/"closed" → indoor`, `"outdoors" → outdoor`, `"open" → retractable` |
| `GameResult` (IntEnum) | -1 away win, 0 tie, 1 home win | |

### Canonical team data (`utils/team_data.py`)

Full metadata for all 32 teams plus `normalize_team_abbreviation(abbr)`,
which **hard-fails** on unknown abbreviations with a closest-match suggestion
via `difflib.get_close_matches`. `LA` is the Rams; `LAC` is the Chargers —
matching nflreadpy. No silent pass-through anywhere.

### Web cache tables (`api/cache.py::CACHE_SCHEMA`)

`predictions`, `feature_importances`, `backtest_metrics`,
`backtest_predictions`, `game_context`, `chart_cache` (pre-rendered
Plotly HTML), and `cache_meta` (with `last_updated`, `prediction_count`).

---

## Deep Dives

### Temporal safety: Protocol + LeakageGate + walk-forward

Temporal correctness is the whole project. Three independent systems cooperate:

1. **`FeatureBuilder` Protocol.** Every builder in `features/` satisfies a
   `@runtime_checkable` Protocol that *requires* an `as_of_datetime`
   parameter. A builder that forgets the parameter fails conformance
   checks; one that reads beyond it fails the next layer.
2. **`LeakageGate`.** Three guards: per-builder `check_time_fence`,
   combined-matrix `validate_combined_matrix` (keyword scan for
   post-game features), `check_elo_ordering` (chronological within each
   season). Violations raise `LeakageViolation` and write a JSON
   diagnostic to `outputs/diagnostics/leakage_<timestamp>.json`.
3. **Walk-forward splits.** `models/temporal.py::TemporalSplitConfig`
   validates that `train_seasons < hp_val_season < holdout_seasons` with
   no overlap; `WalkForwardSplitter` does expanding-window splits within
   the holdout. No random CV anywhere.

### Market blending: pre-2018 tuning isolation

`models/blending.py` blends WP in log-odds space (`scipy.special.logit` /
`expit`) and ATS / O/U linearly in point space. The non-obvious decision is
*where* weights are tuned: strictly pre-2018 seasons, stored as
`TUNING_SEASONS` in `models/blending_data.py`. The 2021-2024 backtest
window is therefore temporally disjoint from the weight-tuning window —
a stricter guarantee than the backtest alone. `DynamicBlendWeights`
extends this with a per-target sigmoid schedule in week-of-season and
per-target gating that falls back to static weights where dynamic does
not empirically beat static.

### UIAP-01 architectural boundary

The `api/` package must never import from `models/`, `features/`, or
`ratings/`. Enforced by `tests/api/test_import_guard.py`, which walks
every Python file under `api/` as an AST and flags imports —
including function-scoped lazy imports — from any forbidden package.
The test fails CI the moment the boundary is violated, preserving the
API's clean startup envelope (milliseconds, no heavy imports) as a hard
invariant.

### Friday orchestrator: three separate gate concepts

`pipeline/orchestrator.py::FridayPipeline` runs 18 sequential steps
(8 data, 10 predictions) each wired through `pipeline/steps.py` with a
**deferred-import adapter pattern** — every `step_*()` imports its
script inside the function body, which avoids argparse collisions at
module load. Three gate concepts are deliberately kept apart:
`StalenessGate` (season + data freshness, bypassed by `--force`),
`PipelineHealthChecker` (environment integrity, preflight + postrun,
never bypassed, advisory under `--force`), and `PipelineAlertManager`
(exactly one alert per outcome). Retry is scoped via tenacity to
transient exception types only (`ConnectionError`, `TimeoutError`,
`OSError`, `httpx.HTTPStatusError`); the execution log at
`logs/friday_pipeline.json` is written atomically.

### API concurrency envelope

`api/main.py::lifespan` opens exactly one shared read-only DuckDB
connection in `app.state.db_conn`. `api/dependencies.py::get_db` runs a
`SELECT 1` health check per request and atomically reconnects under a
`threading.RLock` on failure. `api/services.py` wraps hot-path queries
in a module-level `cachetools.TTLCache(maxsize=128, ttl=300)` with
deep-copy on read and write plus an explicit `RLock`. This all assumes
**single-worker uvicorn** (`--workers 1`). The module docstring says so
plainly; multi-worker gunicorn is available in
`deployment/gunicorn.conf.py` but would invalidate the shared-connection
and shared-cache invariants — a deliberate concurrency envelope, not a
bug.

---

## Project Structure

```
nfl-predict/
  api/                     # FastAPI serving HTML + JSON from the DuckDB web cache
    main.py                #   Lifespan: shared read-only DuckDB connection
    dependencies.py        #   get_db() DI, RLock reconnect, Jinja2Blocks loader
    services.py            #   DataService: TTLCache(maxsize=128, ttl=300) + RLock
    cache.py               #   CACHE_SCHEMA + populate_cache() — builds web_cache.duckdb
    charts/                #   Plotly chart generators (pure render layer)
    routes/{pages,fragments,api_export,health}.py  # HTML, HTMX, CSV/JSON, /health

  backtest/                # Walk-forward backtest + reporting + CLV
    engine.py, walkforward.py          # Orchestrator + SeasonSplit + ordering guards
    metrics.py, simulation.py          # Brier decomposition; flat + quarter-Kelly
    clv_tracking.py, report.py, tune.py

  conf/
    settings.py, config.yaml           # Pydantic BaseSettings + YAML config

  data/
    storage.py             # DuckDB + Parquet IO; tz normalization at write boundaries
    schemas.py             # Pydantic v2 — GameSchema, OddsSchema, WeatherSchema
    quality_gates.py       # validate_bronze_to_silver / validate_silver_to_gold
    bronze/, silver/, gold/            # Layered lakehouse
    optuna/                # SQLite stores for resumable Optuna studies
    baselines/{v1.0,v2.0}/ # Pinned backtests for A/B regression
    venues.json            # Static venue metadata (483 lines)
    nfl_predictions.duckdb # Analytics DB (~37 MB)
    web_cache.duckdb       # Read-only API cache (~6 MB)

  features/                # All builders satisfy FeatureBuilder Protocol
    protocol.py            #   @runtime_checkable Protocol with as_of_datetime
    validation.py          #   LeakageGate (hard-fail) + FeatureValidator (report-only)
    normalization.py       #   Expanding-window Z-score + prior-season bootstrap
    elo_features.py, team_form.py, weather.py, contextual.py,
    market_anchors.py, opponent_adj.py, qb_tracking.py

  models/
    trainers/              # v2.0 Optuna-driven trainers (base, wp, ats, ou)
    temporal.py, tuning.py # WalkForwardSplitter + OptunaTuner (TPE + Hyperband)
    calibrate.py           # IsotonicRegression primary, Platt fallback, ECE
    blending.py            # BlendWeights + DynamicBlendWeights + MarketBlender
    blending_data.py       # TUNING_SEASONS (pre-2018) — strict temporal isolation
    clv.py                 # Probability CLV + line CLV (primary quality metric)
    prediction_pipeline.py # Artifact-based loading; no trainer imports
    artifacts.py, train.py

  pipeline/                # v2.0 Friday orchestrator (Phase 14)
    orchestrator.py        # FridayPipeline: 18 steps with retry + logging
    steps.py               # StepDefinitions with deferred imports
    staleness.py, health.py, execution_log.py, alert.py

  ratings/elo.py           # FiveThirtyEight-style Elo: MoV K-factor, per-season HFA,
                           # divisional factor 0.54, snapshot-then-update

  scripts/                 # CLI entry points for pipeline steps + operations
    friday_pipeline.py, ingest_{games,odds,weather}.py,
    build_*.py, populate_cache.py, validate_*.py

  tests/
    unit/        # 47 files: trainers, builders, leakage, Elo, quality gates, ...
    integration/ # 19 files: end-to-end pipeline, idempotency, smoke tests
    api/         # 12 files including UIAP-01 import guard
    ui/          # HTML snapshot tests via BeautifulSoup

  utils/                   # Cross-cutting
    team_data.py           #   32 teams; normalize_team_abbreviation (hard-fail)
    date_utils.py          #   get_snapshot_time("Friday 18:00") + tz helpers
    logging_config.py      #   structlog with sensitive-data filter + request IDs
    probability_utils.py, kelly_criterion.py

  web/                     # Frontend: Jinja2 + Tailwind v4 + HTMX
    templates/             #   base.html + pages/ + components/
    static/input.css       #   Tailwind v4 source with @theme { NFL palette }

  deployment/              # Docker + Nginx + Gunicorn + cron
    Dockerfile, docker-compose.yml, docker-entrypoint.sh
    gunicorn.conf.py, uvicorn.conf.py, nginx.conf, security.py
    crontab.txt            #   Friday 5pm + 6pm ET (predates Phase 14)
    setup_scheduling.py    #   Cross-platform scheduler installer

  tools/tailwindcss.exe    # Vendored Tailwind v4 CLI binary
  Makefile                 # 40+ targets: snapshot, backtest, tune-blend, serve, train
  pyproject.toml           # Dependencies, Ruff, Pyright, pytest, coverage
  .github/workflows/friday-production.yml  # Optional GH Actions backup scheduler
```

---

## Testing

| Suite | File count | What it covers |
|-------|-----------:|----------------|
| `tests/unit/` | 47 | Trainers, feature builders, leakage gate, Elo correctness + no-leakage, quality gates, temporal splits, CLV, static + dynamic blending, Optuna tuning, timezone handling, pipeline health + orchestrator + staleness + alerts + execution log, betting simulation, QB tracking, opponent adjustment, backtest engine + metrics + report |
| `tests/integration/` | 19 | End-to-end pipeline, backtest comparison + report, data completeness, Elo convergence, idempotency, lift validation, nflreadpy + Open-Meteo smoke tests, Phase 15 integration, prediction pipeline, scheduling setup, training pipeline |
| `tests/api/` | 12 | Pages, fragments, exports, cache headers, caching, connection management, error responses, health endpoint, **UIAP-01 import guard** |
| `tests/ui/` | HTML snapshot | BeautifulSoup-based snapshot tests over rendered pages |

Seven test files use Hypothesis for property-based testing (betting
simulation, blending data, blend tuning, baseline capture, backtest
comparison + report, market blending, health endpoint). `make test-quick`
runs the fast subset; `make dev-test` runs the full suite with coverage
and HTML report.

---

## Deployment

**What actually serves production.**

- FastAPI app under a **single-worker** uvicorn (via `gunicorn` with
  `UvicornWorker`, `preload_app = True`).
- Nginx in front for TLS termination, rate limiting (`api_limit 10r/s`,
  `health_limit 2r/s`, `static_limit 20r/s`), connection limiting
  (20/IP), CSP/HSTS/OCSP stapling, and static asset caching.
- DuckDB on local disk — one read-only connection shared by the app
  (`data/web_cache.duckdb`).
- A scheduler (cron on Linux, Task Scheduler on Windows, optional
  GitHub Actions fallback at `.github/workflows/friday-production.yml`)
  that triggers the Friday orchestrator.
- Security headers middleware (`deployment/security.py`) layered into
  the FastAPI stack.
- Docker: multi-stage image (`builder → runtime → development →
  testing`) with a non-root `nflpredict` user and a `/health`-based
  container healthcheck.

**What `docker-compose.yml` also defines.** The compose file includes
Redis, Postgres, Prometheus, Grafana, and an ELK stack
(Elasticsearch + Logstash + Kibana). These are optional extras
scaffolded during earlier exploration — the minimum runtime needs only
the API, Nginx, DuckDB, and the scheduler. This is called out as a
limitation below to avoid overstating operational maturity.

---

## Current Limitations

1. **v2.0 retrain did not pass gating.** The v2.0 tuning pass produced
   new candidate artifacts that failed per-target gating, so production
   continues to run the v1.0 models and the static (pre-dynamic) blend
   artifact `blend_20260324_023118`. `DynamicBlendWeights` is fully
   implemented in code and under test — it will activate the moment a
   future retrain clears gating.
2. **Deployment docs partially stale.** `deployment/crontab.txt` and
   `deployment/README.md` both predate the unified Friday orchestrator
   introduced in Phase 14. The crontab still references
   `friday_data_update.py` / `friday_predictions_run.py`, which were
   replaced by `scripts/friday_pipeline.py` + `pipeline/orchestrator.py`.
   Code is current; the docs need a refresh.
3. **Single-worker concurrency envelope.** The shared DuckDB connection
   and module-level `TTLCache` require `--workers 1`. Multi-worker
   gunicorn would invalidate those invariants. This is documented in
   the `api/main.py` module docstring and is a deliberate envelope, not
   an accidental limit.
4. **docker-compose ships more than is used.** Redis, Postgres, and the
   ELK stack are optional scaffolding, not part of the minimum
   deployment. Describing the compose file as "the production stack"
   would overstate what's actually wired up.
5. **Phases 17-18 are not started.** There is no betting dashboard
   page and no season tracking page. The nav shows four tabs because
   four tabs are what exist.
6. **2025 season data is partial.** 2002-2024 seasons are fully
   ingested and exercised. 2025 Bronze / Silver captures exist but the
   current-season pipeline has only been run in partial exercises, not
   in anger for a full season.
7. **Legacy trainer modules coexist with new ones.** `models/train_wp.py`,
   `models/train_ats.py`, and `models/train_ou.py` stay because
   `prediction_pipeline.py` still imports `ResidualDistributionConverter`
   and `TotalDistributionConverter` from them. A future refactor can
   move the converters out, but today both paths are present.

---

## Getting Started

```bash
uv sync                           # install everything (no pip, no venv dance)
cp deployment/production.env .env # template; fill ODDS_API_KEY and anything else required

make snapshot                     # ingest -> Silver -> Gold for the current week
make train                        # train WP, ATS, O/U with Optuna tuning
make backtest                     # walk-forward backtest over 2021-2024
make build-cache                  # (re)build data/web_cache.duckdb
make serve                        # FastAPI at http://localhost:8000
make test-quick                   # fast test subset; `make dev-test` for the full suite
```

`make help` prints the full target list (40+ commands). Every target
runs through `uv run`, so nothing depends on an activated virtualenv.

---

## Conclusion

This project exists to answer a narrow question honestly: given a
Friday 18:00 ET snapshot of games, odds, and weather, what are my
best-guess WP / ATS / O/U forecasts, how much should I trust them,
and how much of that trust is borrowed from the market? The answer
is delivered by a lakehouse with hard-fail quality gates, a feature
pipeline with three independent leakage checks, walk-forward-only
model training with Optuna, a market blend tuned on temporally
disjoint seasons, and a read-only web cache served behind an
architecturally quarantined API. The product is intentionally modest
in scope (one user, one tab set, one Friday snapshot per week) and
correspondingly opinionated about correctness.

## License

MIT
