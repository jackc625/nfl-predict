# PIPELINE.md — Canonical Run Sequence

This is the single source of truth for how to run the NFL prediction system end to
end. There are **7 stages**: ingest -> features -> train -> backtest -> predict ->
build-cache -> serve. Each stage below gives:

- the **one canonical command** (runnable directly in PowerShell via `uv run`),
- the **live entry point** it resolves to, and
- the **expected output** it produces (so each stage's result is checkable).

Per the consolidation decisions there is **no full-chain mega-command**: you run the
stages independently, in order. The thin `Makefile` mirrors these exact commands so
the two never drift. For the automated weekly subset, see "Current-week / weekly run"
at the end.

All commands assume the repo root as the working directory and `uv` as the runtime
(`uv sync` first if you have not installed dependencies). Windows-native PowerShell is
the supported shell; there is no Unix-shell layer between you and the system.

---

## The 7 canonical stages

### 1. Ingest

Pull raw games, odds, and weather into Bronze, then upsert to Silver.

```powershell
uv run python scripts/ingest_games.py --season <YEAR>
uv run python scripts/ingest_odds.py --season <YEAR>
uv run python scripts/ingest_weather.py --season <YEAR>
```

- **Live entry point:** `scripts/ingest_games.py`, `scripts/ingest_odds.py`,
  `scripts/ingest_weather.py` (shared `--season` / `--seasons` / `--week` / `--weeks`
  / `--current` / `--all` flags via `utils/ingestion_args.py`).
- **Produces:** `data/bronze/*.parquet` (append-only, timestamped) then
  `data/silver/{games,odds_snapshot,weather}.parquet` (schema-validated, latest-wins).

### 2. Features

Build each feature component, then assemble the per-target Gold matrices. Use
`--all-seasons` for a full historical build, or `--season <YEAR>` for a single season.

```powershell
uv run python scripts/build_elo.py --all-seasons
uv run python scripts/build_team_form.py --all-seasons
uv run python scripts/build_contextual.py --season <YEAR>
uv run python scripts/build_weather.py --season <YEAR>
uv run python scripts/build_market_anchors.py --season <YEAR>
uv run python scripts/build_features.py --season <YEAR>
```

- **Live entry point:** `scripts/build_*.py`.
- **Produces:** `data/silver/elo_game_snapshots.parquet`, `data/silver/team_form_features.parquet`,
  contextual / weather / market-anchor feature tables, then the Gold matrices
  `data/gold/features_wp.parquet`, `data/gold/features_ats.parquet`,
  `data/gold/features_ou.parquet`.

> Note: `build_elo.py` and `build_team_form.py` accept `--all-seasons` (and
> `--current` for the current season). The contextual/weather/market-anchor/features
> builders take `--season` and `--week`. The commands above are for the
> historical/full build; for the current-week build use the orchestrator (see below).

### 3. Train

Train all three models (WP, ATS, O/U) with walk-forward temporal validation.

```powershell
uv run python scripts/train_models.py --target all
```

- **Live entry point:** `scripts/train_models.py` (thin wrapper over
  `models.train.main`; pass `--target wp|ats|ou` for a single target).
- **Produces:** `artifacts/{target}_<UTCtimestamp>/` model directories plus the
  `artifacts/latest.json` manifest that points at the production versions.

### 4. Backtest

Run the walk-forward backtest across 2021-2024 with an interactive HTML report. Add
`--blend` for the market-blended run.

```powershell
uv run python scripts/run_backtest.py
uv run python scripts/run_backtest.py --blend
```

- **Live entry point:** `scripts/run_backtest.py` (thin wrapper over `backtest.run`).
- **Produces:** `outputs/backtest/backtest_report.html`,
  `outputs/backtest/predictions_all.csv`, `outputs/backtest/season_metrics.csv`,
  `outputs/backtest/betting_simulation.csv`, `outputs/backtest/metrics_summary.json`.

### 5. Predict

Generate predictions for a specific season/week using the trained artifacts.

```powershell
uv run python scripts/generate_current_week_predictions.py --season <YEAR> --week <WEEK>
```

- **Live entry point:** `scripts/generate_current_week_predictions.py` (`--season` and
  `--week` are required; `--no-blend` skips market blending).
- **Produces:** `outputs/predictions/predictions_<YEAR>_week<WEEK>.csv` and `.json`,
  plus `outputs/predictions/game_context_<YEAR>_week<WEEK>.csv`.

### 6. Build cache

Build the read-only DuckDB web cache the API serves from. This stage sits **between
predict and serve** because, under the UIAP-01 boundary, the FastAPI app reads ONLY
from `data/web_cache.duckdb` — skip this and `serve` shows stale or empty data.

```powershell
uv run python scripts/populate_cache.py
```

- **Live entry point:** `scripts/populate_cache.py` -> `api.cache.populate_cache`.
- **Produces:** `data/web_cache.duckdb` (~6 MB; the only data source the API reads).

### 7. Serve

Start the FastAPI app + web UI (single worker — the shared DuckDB connection and
in-process cache require `--workers 1`).

```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Live entry point:** `api/main.py:app`.
- **Produces:** FastAPI at http://localhost:8000 — the six top-nav pages `/`
  (This Week), `/performance`, `/backtest`, `/insights`, `/betting`, and
  `/season`, plus the `/games/{id}` game-detail drill-down and a `/health`
  endpoint.

---

## Current-week / weekly run

The per-stage commands above are for **historical / full builds** (they use
`--season` / `--all-seasons`, which the scripts actually accept). The build scripts do
**not** expose a per-stage current-week flag; the current season is built via
`--current` on the Elo/form builders, and the full current-week run goes through the
orchestrator below.

The coherent **current-week** path is the Friday orchestrator, which calls the builder
classes directly and runs the data + predictions subset:

```powershell
uv run python scripts/friday_pipeline.py --log-level INFO
```

- **Live entry point:** `scripts/friday_pipeline.py` ->
  `pipeline.orchestrator.FridayPipeline` (18-step registry in `pipeline/steps.py`).
- **Produces:** refreshed Silver/Gold for the current week, current-week predictions
  and recommendations, exports, and a run log at `logs/friday_pipeline.json`.
- **Scope:** ingest games/weather -> data QA -> build Elo/form/contextual/weather ->
  ingest odds -> market anchors -> build features -> validate features/models ->
  generate predictions/recommendations -> export -> validate outputs. It does **not**
  train, backtest, or rebuild the web cache — run stages 3, 4, and 6 above for those.
