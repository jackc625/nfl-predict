# NFL Prediction System

A pre-game NFL prediction system that produces calibrated Win Probability (WP),
Against the Spread (ATS), and Over/Under (O/U) forecasts for every weekly matchup.
Models are trained strictly walk-forward, blended with the market in the
mathematically appropriate space (log-odds for probabilities, point space for
spreads and totals), and served from a read-only DuckDB cache by a FastAPI +
HTMX + Tailwind v4 web app.

The project is both a personal decision-support tool and a portfolio piece.
v1.0 MVP shipped on 2026-03-28, v2.0 (accuracy, automation, and polish) shipped
on 2026-05-26, and v2.1 "Trust & Reproducibility" (documentation, audit, and
diagnosis -- not new features) shipped on 2026-06-01. The current milestone,
v3.0 "Accuracy & Profitability" (gate hardening, gated re-fits, O/U
monetization, new signals, productization), is in progress.

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
  2021-2024 backtest window is never seen during weight selection (weights
  are tuned on pre-2018 seasons; even the 2018-2020 training window
  post-dates them) -- a harder constraint than the backtest itself.

---

## Status and Scope

**Shipped (v1.0 MVP, 2026-03-28).** Data pipeline, feature engineering,
model training, walk-forward backtest, market blending, and the web
dashboard are all in production.

**Shipped (v2.0 accuracy / automation / polish, 2026-05-26).** Phases
11-18 are complete: Elo rebuild with snapshot-then-update, Optuna-driven
retraining infrastructure, `DynamicBlendWeights` (week-of-season sigmoid
schedule, adopted per-target via gating), a unified Friday orchestrator
with retry and health gates, data-quality monitoring, the model-insights
page, the betting dashboard, and the season-tracking page.

**Shipped (v2.1 Trust & Reproducibility, 2026-06-01).** A trust/audit
milestone with NO new product features: converge on one canonical,
verified, documented pipeline; forensically audit the data and feature
engineering for correctness and leakage; honestly diagnose real model
accuracy; and verify, then explain, the automation. Phases 19-23 are
complete. See `STATE-OF-SYSTEM.md` for the trustworthy/fixed/deferred
summary and `MODEL-DIAGNOSIS.md` for the per-target accuracy verdict.

**In progress (v3.0 Accuracy & Profitability).** Phases 24-31: a
significance-tested per-target deploy gate as a hard block (24), the first
gated re-fit activation (25), the O/U CLV-to-ROI divergence diagnosis (26),
the O/U monetization chain (27), new injury / snap / situational signals
screened into gold (28), a budget-gated line-movement signal (29), a
second gated re-fit on the widened gold that measured each feature group's
contribution before deploying anything (30), and productization -- the weekly
+EV bet list and the milestone-close profitability readout (31). All eight
phases are complete.

**What is explicitly *not* in scope.** Automated bet placement (legal
complexity -- informational only), in-game / real-time predictions,
player props, DFS optimization, multi-user access, native mobile apps.
None of these exist in the codebase.

**What happened in Phase 25 (the first gated activation).** WP and ATS were
re-fit on the canonical Elo gold and passed the per-target non-regression
deploy gate (ATS via one documented fix-cycle), so they were promoted
through the gate (D25-18). O/U RETAINED the v1.0 pre-Elo model: its re-fit
failed the gate and was honestly refused (D25-14) because the v1.0 O/U
carries a stronger line-CLV edge the non-regression floor protects. The
v2.0 retrain that originally failed gating (D-17) is the history that
motivated the hardened gate; the Phase-25 activation is recorded in
`ACTIVATION-READOUT.md` (per-target CLV before/after + the deployed/retained
2x2).

**What serves production now (post Phase-30 gated re-fit).** Phase 30 rebuilt
gold in four separately-attributed rungs, measured each new feature group's
contribution against a rule frozen before any number existed, then ran the
same per-target gate again:

| Target | Serving | Phase-30 gate outcome |
|--------|---------|-----------------------|
| WP | `wp_20260824_113325` | PASSED -- promoted (the phase's one production change) |
| ATS | `ats_20260605_220128` | REFUSED -- the Phase-25 re-fit is RETAINED |
| O/U | `ou_20260326_163930` | REFUSED -- the v1.0 pre-Elo model is RETAINED |
| blend | `blend_dynamic_20260606_020635` | not gated here -- unchanged |

The dynamic blend (D-19) is live for all three targets. Two refusals out of
three is the gate working as designed, in the same D25-14 lineage: both
candidates measured worse than what was already serving, and the frozen gate
said so before either could ship. A refusal is a RESULT, and the numbers
behind both refusals are published in the same shape as the promotion's. On
feature groups: `snap` and `situational` were KEPT and `injury` was DROPPED
after measurably hurting a target -- these were screened in Phase 28 and only
BINDINGLY ruled on in Phase 30, so a group reaching gold is not a group that
was deployed. The full record, including what the phase deliberately left
open, is `GATED-REFIT-READOUT.md`. See also "Current Limitations" and
`MODEL-DIAGNOSIS.md`, the frozen v2.1 diagnosis that recommended the first
gated re-fit.

**What Phase 31 closed, and what it deliberately did not.** Phase 31 shipped
the weekly +EV bet list on a new `/bets` page -- ranked, sized in units, with
an EV band, served entirely from precomputed cache blobs -- and spent the
single unburned 2025 season on one pre-registered profitability measurement.
**It deployed no model.** No artifact was re-fit or promoted, the blend was
not changed, and `artifacts/latest.json` is byte-unchanged by the phase: the
production swap surface was left untouched on purpose, so the measurement
would be a measurement of what is actually serving.

**No target came out PROFITABLE_CLEAN.** The win-probability and totals targets
returned small positive flat-stake returns whose pre-registered ROI p-values do
not clear alpha, and the spread target returned a measured loss. The one
strongly significant result in the whole run is a closing-line-value result on
the win-probability target, and closing-line value is not profitability -- that
divergence is the phase's headline finding rather than a return. The rule was
frozen before any 2025 number existed and the split cannot be re-run. The full
per-target record, together with every disclosure the owner accepted, is
`PROFITABILITY-READOUT.md`.

---

## Key Features

### Prediction outputs
- **Win Probability (WP)** -- calibrated home-team win probability from a
  `LogisticRegression` + `StandardScaler` + `PlattCalibrator` pipeline.
  Calibration is fit on the HP-validation fold, never on training data.
- **Against the Spread (ATS)** -- predicted home margin from an `XGBRegressor`,
  converted to cover probabilities via an empirical `ResidualDistributionConverter`.
- **Over/Under (O/U)** -- predicted total points from an `XGBRegressor`,
  converted to over/under probabilities via a `TotalDistributionConverter`.

### Data lakehouse
- **Bronze** -- append-only timestamped Parquet snapshots from every ingestion.
- **Silver** -- cleaned, schema-validated, latest-wins tables, Hive-partitioned
  by `season=YYYY/` and `snapshot_ts=...` (URL-encoded ET ISO-8601), so the
  Friday 18:00 ET snapshot is a first-class storage attribute.
- **Gold** -- one feature matrix per target: `features_wp.parquet`,
  `features_ats.parquet`, `features_ou.parquet`.
- **Pydantic v2 quality gates** at every layer boundary. One bad row fails
  the whole batch (intentional -- silent row-skipping causes downstream
  corruption that is harder to debug than a hard failure).

### Feature engineering
Elo, rolling team form (EPA / success / pace / red zone / third-down),
market anchors (opening + Friday-snapshot lines; closing lines *never* read
into features), weather (indoor games zeroed), contextual (rest, travel,
divisional, surface mismatch), composite QB quality
(`0.7*z(rolling_qb_epa) + 0.3*z(rolling_cpoe)`), and opponent-adjusted
EPA. Normalization is **expanding-window** grouped by season with
`min_periods=4` and prior-season bootstrap, explicitly replacing the
leakier within-season Z-score approach.

### Temporal safety
Three independent layers: `features/protocol.py::FeatureBuilder`
(`@runtime_checkable` Protocol with mandatory `as_of_datetime`);
`features/validation.py::LeakageGate` (per-builder time-fence,
combined-matrix keyword scan, Elo chronological ordering -- violations
raise `LeakageViolation` and emit a JSON diagnostic); and
`models/temporal.py::TemporalSplitConfig.validate()` which raises if
train / HP-validation / holdout seasons overlap or violate ordering.

### Market blending
WP blended in log-odds space via `scipy.special.logit` / `expit`;
ATS / O/U blended linearly in point space. Weights tuned strictly on
**pre-2018 seasons** (`TUNING_SEASONS` in `models/blending_data.py`) --
temporally disjoint from the 2021-2024 backtest window.
`DynamicBlendWeights` (Phase 13) adds a per-target sigmoid schedule in
week-of-season with per-target gating; it is live for all three targets
(`blend_dynamic_20260606_020635`). Phase 30 re-ran the blend comparison
against the newly serving models, in a throwaway copy of the artifacts
tree, and recorded that the gating outcome would now prefer static for WP
and ATS -- by margins of 1.3e-4 and 2.8e-5, which reads as
indistinguishable rather than harmful. That finding was recorded and NOT
acted on; changing the blend is a separate gated decision.

### Walk-forward backtesting
Expanding-window walk-forward over 2021-2024, per-season retraining.
`backtest/metrics.py::brier_decomposition` implements Murphy (1973)
reliability / resolution / uncertainty by hand.
`backtest/simulation.py::BettingSimulator` runs flat-stake *and*
quarter-Kelly side by side, with `SLIPPAGE_POINTS = 0.5` and -110 vig on
ATS / O/U. CLV (`models/clv.py`) -- probability and line -- is the
primary quality metric.

### Web dashboard
FastAPI monolith with a shared read-only DuckDB connection. Jinja2
templates rendered through `jinja2-fragments`'s `Jinja2Blocks` so
full-page requests and HTMX fragment swaps share the same templates.
Six pages (**This Week**, **Performance**, **Backtest**, **Insights**,
**Betting**, **Season**), game detail drill-down (`/games/{id}`), HTMX
partials, CSV + JSON exports, and a `/health` endpoint.

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
   | data/bronze/  (append-only, timestamped parquet -- raw source snapshots)               |
   +---------------------------------------------------------------------------------------+
              |  validate_bronze_to_silver (Pydantic v2; 1 bad row fails the batch)
              v
   +---------------------------------------------------------------------------------------+
   | data/silver/  (Hive-partitioned: season=YYYY/, snapshot_ts=YYYY-MM-DDT18%3A00...)       |
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
   |   WP  : LogReg + StandardScaler + Platt scaling (calibrated on HP-val fold)           |
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
   |   Pages:     /, /performance, /backtest, /insights, /betting, /season, /bets,        |
   |              /games/{id}                                                              |
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

`models/prediction_pipeline.py` imports *only* `load_model_artifact` -- no
trainer classes on the prediction path. The manifest at `artifacts/latest.json`
points at the production versions. After the Phase-25 gated activation those were
`wp_20260605_215552` (re-fit, activated), `ats_20260605_220128` (re-fit, activated
via fix-cycle), `ou_20260326_163930` (v1.0, retained -- gate refused the re-fit),
and `blend_dynamic_20260606_020635`; that record is `ACTIVATION-READOUT.md`.
After the Phase-30 gated re-fit exactly one key moved: WP is now
`wp_20260824_113325`. ATS, O/U and the blend pointer are unchanged, because the
Phase-30 ATS and O/U candidates were REFUSED by the gate and their incumbents
retained. The per-target before/after and both pre-swap mappings are in
`ACTIVATION-READOUT.md` and `GATED-REFIT-READOUT.md`.

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
| WP model | scikit-learn `LogisticRegression` + Platt-scaled calibration | `>=1.8, <2` |
| ATS / O/U model | XGBoost `XGBRegressor` | `>=3.2, <4` |
| Hyperparameter tuning | Optuna (TPE + Hyperband, SQLite studies) | `==4.8.0` (pinned exact) |
| Web framework | FastAPI | `>=0.115, <1` |
| ASGI server | uvicorn (single-worker envelope) | uvicorn `>=0.34, <1` |
| Templating | Jinja2 + `jinja2-fragments` (`Jinja2Blocks`) | `jinja2-fragments>=1.11` |
| Frontend interactivity | HTMX 2.0.4 (CDN) | -- |
| Charts | Plotly (Python pre-render + JS CDN) | `plotly>=6.0, <7` |
| Styling | Tailwind CSS v4 (vendored CLI at `tools/tailwindcss.exe`) | -- |
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
| Gold | `data/gold/` | Parquet (one file per target) | Feature matrices ready for training -- 194 / 195 / 194 columns (wp / ats / ou) over 6,499 rows spanning 2002-2025 after the Phase-30 rebuild |
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
| `WeatherSchema` | Silver-layer weather | Open-Meteo shape + `wind_direction` float + NaN->None coercion |
| `VenueRoof` (StrEnum) | `indoor` / `outdoor` / `retractable` | nflverse `"dome"/"closed" -> indoor`, `"outdoors" -> outdoor`, `"open" -> retractable` |
| `GameResult` (IntEnum) | -1 away win, 0 tie, 1 home win | |

### Canonical team data (`utils/team_data.py`)

Full metadata for all 32 teams plus `normalize_team_abbreviation(abbr)`,
which **hard-fails** on unknown abbreviations with a closest-match suggestion
via `difflib.get_close_matches`. `LA` is the Rams; `LAC` is the Chargers --
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
window is therefore temporally disjoint from the weight-tuning window --
a stricter guarantee than the backtest alone. `DynamicBlendWeights`
extends this with a per-target sigmoid schedule in week-of-season and
per-target gating that falls back to static weights where dynamic does
not empirically beat static.

### UIAP-01 architectural boundary

The `api/` package must never import from `models/`, `features/`, or
`ratings/`. Enforced by `tests/api/test_import_guard.py`, which walks
every Python file under `api/` as an AST and flags imports --
including function-scoped lazy imports -- from any forbidden package.
The test fails CI the moment the boundary is violated, preserving the
API's clean startup envelope (milliseconds, no heavy imports) as a hard
invariant.

### Friday orchestrator: three separate gate concepts

`pipeline/orchestrator.py::FridayPipeline` runs 18 sequential steps
(8 data, 10 predictions) each wired through `pipeline/steps.py` with a
**deferred-import adapter pattern** -- every `step_*()` imports its
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
plainly; running multiple workers would invalidate the shared-connection
and shared-cache invariants -- a deliberate concurrency envelope, not a
bug.

---

## Project Structure

```
nfl-predict/
  api/                     # FastAPI serving HTML + JSON from the DuckDB web cache
    main.py                #   Lifespan: shared read-only DuckDB connection
    dependencies.py        #   get_db() DI, RLock reconnect, Jinja2Blocks loader
    services.py            #   DataService: TTLCache(maxsize=128, ttl=300) + RLock
    cache.py               #   CACHE_SCHEMA + populate_cache() -- builds web_cache.duckdb
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
    schemas.py             # Pydantic v2 -- GameSchema, OddsSchema, WeatherSchema
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
    calibrate.py           # Platt + isotonic calibrators, ECE (WP trainer selects Platt)
    blending.py            # BlendWeights + DynamicBlendWeights + MarketBlender
    blending_data.py       # TUNING_SEASONS (pre-2018) -- strict temporal isolation
    clv.py                 # Probability CLV + line CLV (primary quality metric)
    prediction_pipeline.py # Artifact-based loading; no trainer imports
    artifacts.py, train.py

  pipeline/                # v2.0 Friday orchestrator (Phase 14)
    orchestrator.py        # FridayPipeline: 19 steps with retry + logging
    steps.py               # StepDefinitions with deferred imports
    staleness.py, health.py, execution_log.py, alert.py

  ratings/elo.py           # FiveThirtyEight-style Elo: MoV K-factor, per-season HFA,
                           # divisional factor 0.54, snapshot-then-update

  scripts/                 # CLI entry points for pipeline steps + operations
    friday_pipeline.py, ingest_{games,odds,weather}.py,
    build_*.py, populate_cache.py, validate_*.py

  tests/
    unit/        # trainers, builders, leakage, Elo, quality gates, doc guards
    integration/ # end-to-end pipeline, idempotency, promote/rollback, smoke tests
    api/         # pages, fragments, exports, health, UIAP-01 import guard
    phase30_state.py  # the tracked home for Phase-30 constants (see the readout)

  utils/                   # Cross-cutting
    team_data.py           #   32 teams; normalize_team_abbreviation (hard-fail)
    date_utils.py          #   get_snapshot_time("Friday 18:00") + tz helpers
    logging_config.py      #   structlog with sensitive-data filter + request IDs
    probability_utils.py, kelly_criterion.py

  web/                     # Frontend: Jinja2 + Tailwind v4 + HTMX
    templates/             #   base.html + pages/ + components/
    static/input.css       #   Tailwind v4 source with @theme { NFL palette }

  deployment/              # Windows Task Scheduler setup (scheduling only)
    setup_scheduling.py    #   Registers the Friday task from the committed XML
    windows_scheduler.xml  #   Task definition (Fri 18:00 local = 6 PM ET)
    README.md              #   Scheduling notes (Phase 19 trimmed)

  tools/tailwindcss.exe    # Vendored Tailwind v4 CLI binary
  Makefile                 # Thin 9-target wrapper over PIPELINE.md (train, backtest,
                           #   backtest-blend, predict, build-cache, serve,
                           #   friday-production, test, lint)
  pyproject.toml           # Dependencies, Ruff, Pyright, pytest, coverage
```

---

## Testing

File counts below are a point-in-time reading taken 2026-08-24, at the close
of Phase 30; they are not pinned by a test, because a guard on a growing
count turns red on every new test file for no correctness reason.

| Suite | File count | What it covers |
|-------|-----------:|----------------|
| `tests/unit/` | 95 | Trainers, feature builders, leakage gate, Elo correctness + no-leakage, quality gates, temporal splits, CLV, static + dynamic blending, Optuna tuning, timezone handling, pipeline health + orchestrator + staleness + alerts + execution log, betting simulation, QB tracking, opponent adjustment, backtest engine + metrics + report, the feature-group gate, and the committed content guards for every repo-root document |
| `tests/integration/` | 37 | End-to-end pipeline, backtest comparison + report, data completeness, Elo convergence, idempotency, lift validation, nflreadpy + Open-Meteo smoke tests, prediction pipeline, scheduling setup, training pipeline, the promotion/rollback paths, and the Phase-30 rebuild controls |
| `tests/api/` | 14 | Pages, fragments, exports, cache headers, caching, connection management, error responses, health endpoint, **UIAP-01 import guard** |

The full suite runs green with **0 failed**. The passing count is not quoted
here, because it moves every time a guard is added and a stale count in the
front door is exactly the drift these guards exist to prevent. The number that
IS worth reading is the **7 `xfail`s**: those are deliberate quarantines --
published anchors awaiting a re-ratification Phase 30 did not perform -- and
that count is the mechanical proof those seven items are still open, not a
count of broken tests. See `GATED-REFIT-READOUT.md` for what each one is
waiting on.

Property-based testing with Hypothesis is used in one file today
(`tests/unit/test_market_blending.py`). An earlier version of this README
claimed seven; that count did not survive the suite's growth and is
corrected here rather than carried. Run the full suite with `make test`
(= `uv run pytest tests/unit tests/integration tests/api -q`); for a faster
loop, `uv run pytest tests/unit -q`.

---

## Deployment

**What actually serves production today.**

- FastAPI app under a **single-worker** uvicorn run locally
  (`uv run uvicorn api.main:app --host 0.0.0.0 --port 8000`). The
  shared read-only DuckDB connection and module-level `TTLCache`
  require `--workers 1` (see the API concurrency envelope above).
- DuckDB on local disk -- one read-only connection shared by the app
  (`data/web_cache.duckdb`).
- The only automation is the **Windows Task Scheduler** Friday run that
  triggers the orchestrator (`scripts/friday_pipeline.py`); see
  `AUTOMATION.md` for what it does and `deployment/` for the task setup.

**Hosted deployment is deliberately out of scope for now.** An earlier
exploratory Docker / Nginx / Gunicorn / docker-compose stack (with
optional Redis, Postgres, Prometheus, Grafana, and ELK extras) was
**deleted in Phase 19** -- it overstated operational maturity and was
not part of the real runtime. Packaging the app for a hosted server is
a planned future milestone (see `.planning/PROJECT.md` Out-of-Scope);
the architecture is kept deployment-friendly so that work stays small.

---

## Current Limitations

1. **O/U still serves the v1.0 model, refused by the gate TWICE.**
   In Phase 25, WP and ATS were re-fit on the canonical Elo gold and
   promoted through the per-target non-regression deploy gate (ATS via one
   documented fix-cycle); O/U's re-fit FAILED the gate and was retained on
   the v1.0 pre-Elo model -- a more-accurate O/U regressor predicts totals
   closer to the market, shrinking its line-CLV below the v1.0 edge the
   non-regression floor protects (honest refusal, D25-14). Phase 30 re-ran
   the gate on the rebuilt, widened gold and reached the same answer in the
   same shape: the O/U candidate's point error IMPROVED (MAE -1.23) while
   its line-CLV got substantially WORSE (-0.487 paired, p = 9.2e-12), so it
   was refused again. The Phase-30 ATS candidate was refused too (-0.213
   paired, p = 0.0375), leaving the Phase-25 ATS artifact serving. Only WP
   moved. The single pre-registered fix-cycle lever went UNSPENT for both
   failing targets, and not by oversight -- both candidates were already
   trained on their incumbent's exact selection window, which is precisely
   what the lever would have done. The dynamic blend (D-19) is live for all
   three targets. The earlier v2.0 retrain that failed gating on every
   target (D-17) is the history that motivated the hardened gate. See
   `GATED-REFIT-READOUT.md` for the Phase-30 record (including the numbers
   behind both refusals), `ACTIVATION-READOUT.md` for the Phase-25
   activation, and `MODEL-DIAGNOSIS.md` (DIAG-05) / `STATE-OF-SYSTEM.md`
   for the frozen v2.1 diagnosis that recommended the first re-fit.
2. **Single-worker concurrency envelope.** The shared DuckDB connection
   and module-level `TTLCache` require `--workers 1`. Running multiple
   workers would invalidate those invariants. This is documented in
   the `api/main.py` module docstring and is a deliberate envelope, not
   an accidental limit.
3. **WP ships with a negative absolute CLV, and that is not a typo.**
   The deploy gate runs in `non_regression` mode: it asks whether a
   candidate is not WORSE than the incumbent, not whether it is positive in
   absolute terms. WP's pooled probability CLV after the Phase-30 promotion
   is -0.0380 -- an improvement of +0.0061 on the incumbent (p = 0.0142),
   and still negative. Removing closing-line-value leakage is not the same
   thing as having a market edge, and this project keeps those two bars
   apart deliberately rather than quoting the flattering one. The honest
   per-target profitability verdict was discharged in Phase 31 on the clean
   2025 split, including the O/U verdict Phase 30 could not reach: no target
   is `PROFITABLE_CLEAN`, and WP's positive closing-line value did not become
   a distinguishable return. See `PROFITABILITY-READOUT.md`.
4. **Legacy trainer modules coexist with new ones.** `models/train_wp.py`,
   `models/train_ats.py`, and `models/train_ou.py` stay because
   `prediction_pipeline.py` still imports `ResidualDistributionConverter`
   and `TotalDistributionConverter` from them. A future refactor can
   move the converters out, but today both paths are present.

---

## Getting Started

The operating shell is **PowerShell on Windows 11**; every command runs
through `uv run`, so nothing depends on an activated virtualenv. The
canonical run sequence lives in `PIPELINE.md`, and `RUNBOOK.md` is the
operator runbook (setup, each operation, how to tell whether a run
succeeded, troubleshooting, recovery).

```powershell
uv sync                           # install everything (no pip, no venv dance)
Copy-Item .env.example .env       # template; fill ODDS_API_KEY and anything else required

uv run python scripts/train_models.py --target all      # train WP, ATS, O/U (Optuna)
uv run python scripts/run_backtest.py                   # walk-forward backtest 2021-2024
uv run python scripts/populate_cache.py                 # (re)build data/web_cache.duckdb
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000  # FastAPI at http://localhost:8000
```

The same steps are available as the thin Makefile targets `make train`,
`make backtest`, `make build-cache`, and `make serve` (see `PIPELINE.md`
for the full 8-stage sequence). Run the tests with `make test` (=
`uv run pytest tests/unit tests/integration tests/api -q`) or, for a
faster loop, `uv run pytest tests/unit -q`.

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

<!-- old-rule-addendum-2026-09-15 -->

## Old-rule addendum (2026-09-15)

This section was added on 2026-09-15. Nothing above it has been changed: every number, table and
heading is exactly as it was first published.

Phase 33.2 found that the model inputs behind the results in this document were defective, in five
ways. Weather observed after each game stood in for the forecast that was actually available the day
before kickoff. Closing betting lines, which are only known at kickoff, were fed into the models as
inputs. Feature builders took their cutoff from one global clock instead of each game's own lock
time. The opponent adjustment never actually ran. Early-season placeholder values read as exactly
league average, with nothing to say they were placeholders.

This document stays in the record, unedited, because deleting it would be worse: it would hide what
was claimed and when. Read it as history, not as a measure of how well the system works.

**Built under the old rule on inputs later found defective; not evidence.** Only the 2026 season,
recorded live under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2.

What this covers in this document: the serving-model table, the gate outcomes and the 2025 profitability summary it reports are all old-rule; the description of how the system is built is not a result and is not covered by this label.
