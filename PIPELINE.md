# PIPELINE.md — Canonical Run Sequence

This is the single source of truth for how to run the NFL prediction system end to
end. There are **8 stages**: ingest -> features -> train -> promote -> backtest ->
predict -> build-cache -> serve. Each stage below gives:

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

## The 8 canonical stages

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
uv run python scripts/build_features.py --all-seasons
```

- **Live entry point:** `scripts/build_*.py`.
- **Produces:** `data/silver/elo_game_snapshots.parquet`, `data/silver/team_form_features.parquet`,
  contextual / weather / market-anchor feature tables, then the Gold matrices
  `data/gold/features_wp.parquet`, `data/gold/features_ats.parquet`,
  `data/gold/features_ou.parquet`.

> Note: `build_elo.py` and `build_team_form.py` accept `--all-seasons` (and
> `--current` for the current season). The contextual/weather/market-anchor
> builders take `--season` and `--week`. `build_features.py` takes `--all-seasons`
> (the full historical build, which is also what a bare invocation does) OR
> `--season` / `--week` (a scoped build); the two are mutually exclusive. The
> commands above are for the historical/full build; for the current-week build use
> the orchestrator (see below).

> Note: `build_features.py` WRITES gold by default. `--no-save` is the real off
> switch for a read-only build (the bare `--save` flag cannot turn saving off --
> it is `store_true` with `default=True`). Gold currently stands at 194 / 195 / 194
> columns (wp / ats / ou) over 6,499 rows spanning 2002-2025, after the four
> attributed rebuild rungs recorded in `GATED-REFIT-READOUT.md`.

> Note -- **the two write modes are not interchangeable.** `--all-seasons` (and the
> bare invocation) REPLACES the gold tables: the frame the build produced becomes
> the table in both DuckDB and Parquet. That is what lets a full rebuild DROP a
> column, which is what the Phase-30 rung-3 narrowing needed. A scoped
> `--season <YEAR>` / `--week <N>` build MERGES instead -- latest-wins on `game_id`,
> every other season preserved -- because it only ever carried its own slice. A
> scoped build therefore cannot change the gold schema, and one that tries is
> refused before it writes rather than silently resurrecting the dropped columns
> as all-null. Use `--all-seasons` for a historical rebuild; do not reach for
> `--season` to get one.

### 3. Train

Train all three models (WP, ATS, O/U) with walk-forward temporal validation.

```powershell
uv run python scripts/train_models.py --target all
```

- **Live entry point:** `scripts/train_models.py` (thin wrapper over
  `models.train.main`; pass `--target wp|ats|ou` for a single target).
- **Produces:** `artifacts/{target}_<UTCtimestamp>/` model directories ONLY.
  Training does **not** touch production: `models.artifacts.save_model_artifact`
  defaults `update_latest=False` and the trainer never overrides it (since Plan
  24-01), so a train run writes versioned candidate dirs but never rewrites the
  `artifacts/latest.json` manifest. The manifest is written ONLY by the Promote
  stage below (`update_manifest`, the sole per-key swapper) and by the blend-mode
  re-validation's `save_blend_artifacts` (the one sanctioned `latest.json['blend']`
  writer). To deploy a freshly trained candidate, run the Promote stage; it is the
  gate that decides, per target, whether a candidate may replace production.

### 4. Promote

Score the freshly trained candidates against the FROZEN per-target deploy gate
(`config/gate.toml`) and, on `--promote`, conditionally swap ONLY the gate-passing
targets into production. This is the ONLY path that rewrites `artifacts/latest.json`
target keys -- training (stage 3) never does. Dry-run by default (scores + prints
the per-target 2x2, swaps nothing); pass `--promote` to perform the conditional swap.

```powershell
uv run python -m scripts.promote_models
uv run python -m scripts.promote_models --promote --skip-train
```

- **Live entry point:** `scripts/promote_models.py` -> `models.deploy_gate` +
  `models.artifacts.update_manifest` (the sole per-key swapper). The straight
  re-fit trains candidates into a staging dir, the gate re-scores the deployed
  baseline for the paired non-regression delta, and each passing target's
  gate-scored artifact dir is copied verbatim into production before its manifest
  key is swapped (byte-identical deploy).
- **Produces (dry-run):** the per-target 2x2 readout (candidate vs the FROZEN
  `[baseline.*]` block in `config/gate.toml`: pooled + per-season CLV
  non-regression, secondary metrics) and a non-zero exit code if any target FAILS
  the gate -- observable to CI. No production change. The frozen baseline describes
  the DEPLOYED incumbent, not v1.0: it was re-pointed at the deployed set in Phase 25
  (D25-11) and re-frozen twice more in Phase 30 -- once before the gate ran, because
  the gold rebuild had moved the values it was measured on, and once after the
  promotion so it describes the end state.
- **Produces (`--promote`):** the conditional per-target swap -- ONLY gate-passing
  targets are copied into `artifacts/` and pointed at by `artifacts/latest.json`;
  a FAILING target keeps its existing production entry (honest refusal is a valid
  outcome). The prior version dirs are retained, so the swap is reversible (see
  RUNBOOK.md "Rollback").

> **Arming the run you actually reviewed (`--skip-train`).** A bare `--promote`
> re-trains the candidates into staging FIRST, so the artifact that ships is not
> the artifact the dry run scored. The safer two-step sequence -- and the one the
> Phase-30 armed run used -- is: run the bare dry-run (it trains into staging and
> scores), review the printed 2x2, then arm with
> `--promote --skip-train`, which REUSES the exact gate-scored staging directories
> instead of re-training. The run prints a staleness warning naming each staged
> directory and its timestamp; read it rather than suppress it. Note that the armed
> run exits NON-ZERO whenever ANY gated target failed, which is correct reporting
> for a partial pass -- Phase 30 promoted WP and refused ATS and O/U, and exited 1.
> The per-target record of that run is `GATED-REFIT-READOUT.md`; the Phase-25
> activation before it is `ACTIVATION-READOUT.md`.

> Bootstrap note (clean checkout): the gated Promote path REQUIRES a pre-existing
> `artifacts/latest.json` (it re-scores the deployed baseline for the paired
> non-regression delta, raising an actionable `FileNotFoundError` when the manifest
> is absent). On a fresh checkout with no manifest, mint the FIRST one with a
> one-time `update_manifest` per target after the first train (see RUNBOOK.md
> "Setup from a fresh checkout"); every subsequent deploy goes through this gated
> Promote stage.

### 5. Backtest

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

### 6. Predict

Generate predictions for a specific season/week using the trained artifacts.

```powershell
uv run python scripts/generate_current_week_predictions.py --season <YEAR> --week <WEEK>
```

- **Live entry point:** `scripts/generate_current_week_predictions.py` (`--season` and
  `--week` are required; `--no-blend` skips market blending).
- **Produces:** `outputs/predictions/predictions_<YEAR>_week<WEEK>.csv` and `.json`,
  plus `outputs/predictions/game_context_<YEAR>_week<WEEK>.csv`.

### 7. Build cache

Build the read-only DuckDB web cache the API serves from. This stage sits **between
predict and serve** because, under the UIAP-01 boundary, the FastAPI app reads ONLY
from `data/web_cache.duckdb` — skip this and `serve` shows stale or empty data.

```powershell
uv run python scripts/populate_cache.py
```

- **Live entry point:** `scripts/populate_cache.py` -> `api.cache.populate_cache`.
- **Produces:** `data/web_cache.duckdb` (~6 MB; the only data source the API reads).

### 8. Serve

Start the FastAPI app + web UI (single worker — the shared DuckDB connection and
in-process cache require `--workers 1`).

```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **Recompile the stylesheet when a template introduces a new utility class.** The app serves
> `web/static/css/tailwind-compiled.css`, which is generated from `web/static/input.css` by the
> VENDORED Tailwind v4 CLI. There is **no `make` target for it** and no stage of its own: it is
> not part of the 8-stage sequence, the Makefile does not wrap it, and nothing in the build
> notices when it is stale. That omission is why compiled-style drift exists in this repository --
> a template can introduce a utility class that was never compiled, and the page then renders
> unstyled with no error anywhere. After editing anything under `web/templates/`, run:
>
> ```powershell
> ./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
> ```
>
> A template change that reuses only classes already present in the compiled sheet needs no
> recompile; when in doubt, run it -- it is idempotent and takes about a second.

- **Live entry point:** `api/main.py:app`.
- **Produces:** FastAPI at http://localhost:8000 — the seven top-nav pages `/`
  (This Week), `/performance`, `/backtest`, `/insights`, `/betting`,
  `/season` and `/bets`, plus the `/games/{id}` game-detail drill-down and a
  `/health` endpoint.

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
  `pipeline.orchestrator.FridayPipeline` (19-step registry in `pipeline/steps.py`).
- **Produces:** refreshed Silver/Gold for the current week, current-week predictions
  and recommendations, exports, a rebuilt `data/web_cache.duckdb`, and a run log at
  `logs/friday_pipeline.json`.
- **Scope:** ingest games/weather -> data QA -> build Elo/form/contextual/weather ->
  ingest odds -> market anchors -> build features -> validate features/models ->
  generate predictions/recommendations -> export -> validate outputs -> **populate the
  web cache**. It does **not** train, promote, or backtest — run stages 3 (train), 4
  (promote) and 5 (backtest) above for those.
- **Stage 7 is now part of the weekly run (changed by plan 31-18).** This document
  previously said the orchestrator does not rebuild the web cache; that is no longer
  true. `populate_web_cache` is the LAST registry entry, runs after the recommendation
  and export steps, and is registered `critical=False` so a cache failure degrades the
  run instead of discarding prediction work that already succeeded. Running stage 7 by
  hand remains the recovery path, and it is the exact command the `/bets` stale-cache
  refusal tells the reader to re-run. See `AUTOMATION.md` section 4.
