# RUNBOOK.md -- Operator Runbook (Phase 23)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 23 (Documentation, Runbook & State-of-System)
**Authored:** 2026-05-31
**Scope:** operator-task layer (DOC-01) + Architecture & Data-Flow section (DOC-02).

> This is the operator runbook: how to set up from a fresh checkout, perform every
> common operation, tell whether a run succeeded, troubleshoot, and recover. It is the
> TASK layer. It BUILDS ON the canonical command reference rather than restating it:
> `PIPELINE.md` owns the per-stage commands (this runbook mirrors them, link-don't-
> duplicate), `AUTOMATION.md` owns the Friday automation behavior (this runbook links it,
> does not re-explain the 18 steps), and `README.md` owns the single architecture diagram
> (the DOC-02 section below cross-references it, no second diagram). HARD BOUNDARY
> (D-01/D-07): this is documentation only -- no model re-fit, no canonical-gold rebuild.
> Every command below carries a verification-basis label so the "verified" claim is honest
> and tiered: SAFE/inference-only commands were run live this session ("verified live
> 2026-05-31"); DESTRUCTIVE commands are cited to the Phase 20 AUDIT-01 stage-runner
> evidence and deliberately NOT re-run ("verified via AUDIT-01 stage runner; not re-run").
>
> ASCII only (no emoji, per CLAUDE.md / the Windows cp1252 console constraint). Arrows are
> `->`, dashes are `--`, quotes are straight.

---

## How to read the verification-basis labels

Every operation below ends with a label stating HOW its command was verified, and when:

- **verified live 2026-05-31** -- the command was actually run during this session's D-02
  SAFE-command verification pass and exited 0. These are the SAFE / idempotent /
  inference-only commands (they load existing artifacts, write only gitignored output, or
  list-without-executing). No model was re-fit and no gold was rebuilt.
- **verified via AUDIT-01 stage runner; not re-run to preserve the D-01 no-re-fit/no-rebuild
  boundary** -- the command is DESTRUCTIVE (it re-fits artifacts, rebuilds canonical gold,
  or hits a live external API). It is NOT re-run here. Its correctness is cited to the
  Phase 20 AUDIT-01 stage-runner evidence (`tests/integration/test_audit_stage_runner.py`
  and `AUDIT-REPORT.md`), where the `--help` surface exited 0 and the destructive behavior
  was characterized without being repeated.

The split exists because a trust milestone must not claim "verified" for a command it could
never safely run. A "verified live" label NEVER appears on `train_models.py`, a real
`build_features.py` build (it writes gold by default -- there is no dry mode), or live ingest
-- those are AUDIT-01-cited by definition.

---

## Setup from a fresh checkout

A fresh clone has the code and the bundled model artifacts, but NO data. Every data and
output location is gitignored, so it does not travel with the checkout:

- `data/` -- Bronze/Silver/Gold parquet, the DuckDB files (gitignored).
- `outputs/` -- predictions, backtest reports (gitignored).
- `artifacts/` -- model directories + `latest.json` manifest (the bundled v1.0 production
  artifacts ARE present in the checkout; new ones from a re-fit would be gitignored).
- `logs/` -- the Friday run log (gitignored).

Steps:

1. Install dependencies (uv is the runtime; PowerShell is the supported shell):

   ```powershell
   uv sync
   ```

2. Create your local environment file from the template (NEVER commit the real `.env`):

   ```powershell
   Copy-Item .env.example .env
   ```

   `.env.example` is the real, committed template. Fill in any API keys you need (The Odds
   API for live odds ingest). For inference-only operation against existing data you do not
   need live keys.

3. Get data. A fresh checkout has none. Either:
   - run the **Ingest** -> **Build features** stages below to build Bronze/Silver/Gold from
     source (live ingest needs keys and is DESTRUCTIVE -- see that section), or
   - if you already have a populated `data/` from a prior run, the bundled artifacts +
     existing gold are enough to run **Predict**, **Build cache**, and **Serve** directly.

`PIPELINE.md` is the canonical 7-stage run sequence; the per-operation sections below match
its commands (the Build-features stage is summarized to its final assembly step -- see that
section and `PIPELINE.md` for the full component-build sequence).

---

## Common operations

There are 8 common operations. Each section gives the `uv run` PowerShell command, how to
tell it succeeded, and a verification-basis label. The canonical commands match `PIPELINE.md`;
for "Build features" this runbook summarizes the stage down to its final assembly step
(`build_features.py`) -- `PIPELINE.md` lists the full per-component build sequence (build_elo,
build_team_form, build_contextual, build_weather, build_market_anchors, then build_features).
When in doubt, `PIPELINE.md` is the command source of truth.

### 1. Ingest

Pull raw games, odds, and weather into Bronze, then upsert to Silver.

```powershell
uv run python scripts/ingest_games.py --season <YEAR>
uv run python scripts/ingest_odds.py --season <YEAR>
uv run python scripts/ingest_weather.py --season <YEAR>
```

- **Succeeded when:** `data/bronze/*.parquet` gains new timestamped snapshots and
  `data/silver/{games,odds_snapshot,weather}.parquet` are refreshed (schema-validated,
  latest-wins). A single bad row fails the whole Pydantic-validated batch -- look in
  `data/bronze/` for the timestamped snapshot and re-run.
- **DESTRUCTIVE:** hits live external APIs (nflreadpy, The Odds API, Open-Meteo) and mutates
  Bronze/Silver. The Odds API is offseason/costly, so this is not pulled here.
- **Verification basis:** verified via AUDIT-01 stage runner; not re-run to preserve the
  D-01 no-re-fit/no-rebuild boundary (AUDIT-REPORT.md Stage 1: `--help` exit 0 for all
  three ingest scripts; AUDIT-05 confirmed 2024 source == disk without a live pull).

### 2. Build features

Build each feature component, then assemble the per-target Gold matrices.

```powershell
uv run python scripts/build_features.py --season <YEAR>
```

- **WARNING -- `build_features.py` writes gold by DEFAULT; there is no dry mode.** `--save`
  is defined `action="store_true"` with `default=True` and there is NO `--no-save`, so the
  flag is effectively always on: any real `--season`/`--all-seasons` build materializes the
  canonical gold matrices (`data/gold/features_{wp,ats,ou}.parquet`). That is a GOLD REBUILD,
  forbidden under D-07. The `build_*.py --all-seasons` feeders likewise rebuild the full
  historical feature tables. The only non-destructive invocation is `--help`. Do NOT run a
  real build to "test the features stage" -- there is no way to validate the assembly without
  writing gold.
- **Succeeded when (a sanctioned rebuild, NOT this milestone):** the build completes without a
  LeakageGate failure and the three gold matrices are written; a LeakageGate failure instead
  lands a diagnostic at `outputs/diagnostics/leakage_<ts>.json`.
- **Verification basis:** verified via AUDIT-01 stage runner (`--help` exit 0,
  AUDIT-REPORT.md Stage 2f); the script has no dry-run, so a real build is DESTRUCTIVE and was
  NOT re-run (the single D-10 Wave-3 rebuild is the cited evidence; D-01/D-07).

### 3. Train

Train all three models (WP, ATS, O/U) with walk-forward temporal validation.

```powershell
uv run python scripts/train_models.py --target all
```

- **Succeeded when:** new `artifacts/{target}_<UTCtimestamp>/` directories appear and
  `artifacts/latest.json` is updated to point at them.
- **DESTRUCTIVE:** this overwrites the production artifacts and the `latest.json` manifest --
  it is a re-fit, which D-07 forbids in this milestone. Pass `--target wp|ats|ou` for a
  single target.
- **Verification basis:** verified via AUDIT-01 stage runner; not re-run (a re-fit, forbidden
  by D-07) (AUDIT-REPORT.md Stage 3: `--help` exit 0, "NOT executed -- D-01 no re-train").

### 4. Backtest

Run the walk-forward backtest across 2021-2024 with an interactive HTML report. Add
`--blend` for the market-blended run.

```powershell
uv run python scripts/run_backtest.py
uv run python scripts/run_backtest.py --blend
```

- **Succeeded when:** `outputs/backtest/backtest_report.html`, `predictions_all.csv`,
  `season_metrics.csv`, `betting_simulation.csv`, and `metrics_summary.json` are written.
- **SAFE:** loads existing artifacts and fits per-fold backtest models internally (NOT a
  deployed-artifact re-fit); writes only gitignored `outputs/backtest/`.
- **Verification basis:** verified live 2026-05-31 (both the plain run and `--blend` exited 0
  this session; outputs are gitignored).

### 5. Predict

Generate predictions for a specific season/week using the trained artifacts.

```powershell
uv run python scripts/generate_current_week_predictions.py --season <YEAR> --week <WEEK>
```

- **Succeeded when:** `outputs/predictions/predictions_<YEAR>_week<WEEK>.csv` and `.json`
  plus `game_context_<YEAR>_week<WEEK>.csv` are written; `wp_prob` must fall in `[0,1]`.
  Pass `--no-blend` to skip market blending.
- **SAFE:** loads the existing v1.0 artifacts (no train); writes only gitignored
  `outputs/predictions/`.
- **Verification basis:** verified live 2026-05-31 -- ran `--season 2024 --week 18` this
  session: exit 0, 16 games, loaded the v1.0 artifacts
  (`wp_20260327_114739` / `ats_20260326_163724` / `ou_20260326_163930`), wrote
  `predictions_2024_week18.csv` + `game_context_2024_week18.csv`, no re-fit, no live API.

### 6. Build cache

Build the read-only DuckDB web cache the API serves from. This sits BETWEEN predict and
serve: under the UIAP-01 boundary the FastAPI app reads ONLY from `data/web_cache.duckdb`,
so skipping this leaves `serve` showing stale or empty data.

```powershell
uv run python scripts/populate_cache.py
```

- **Succeeded when:** `data/web_cache.duckdb` (~6 MB) is refreshed -- the only data source
  the API reads.
- **SAFE:** a downstream DATA refresh from gold + backtest outputs; NOT a re-fit. The file is
  gitignored, so the refresh mutates only a gitignored file.
- **Verification basis:** verified live 2026-05-31 (exit 0 this session; refreshed the
  gitignored `data/web_cache.duckdb` -- 1139 predictions, 6263 game_context rows).

### 7. Serve

Start the FastAPI app + web UI (single worker -- the shared DuckDB connection and in-process
cache require `--workers 1`).

```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Succeeded when:** FastAPI is reachable at http://localhost:8000 with the six nav pages
  `/`, `/performance`, `/backtest`, `/insights`, `/betting`, `/season`, the `/games/{id}`
  detail drill-down, and a `/health` endpoint that
  returns HTTP 200. `/health` returns 200 for BOTH a healthy and a `degraded` status; in the
  offseason it is expected to report `status: "degraded"` (a data-freshness state) while
  still returning 200 with `cache_ready: true` and `all_models_exist: true`.
- **SAFE:** read-only DuckDB connection; per the UIAP-01 boundary the API does not import
  `models/`, `features/`, or `ratings/` and loads no model.
- **Verification basis:** verified live 2026-05-31 -- started uvicorn this session and reached
  http://localhost:8000/health: HTTP 200, `cache_ready: true`, `all_models_exist: true`,
  `status: "degraded"` (the expected offseason data-freshness state), then stopped it.

### 8. Run automation (the Friday pipeline)

The weekly run is the Friday orchestrator -- the current-week subset of the stages above
(ingest -> features -> predict -> validate; it does NOT train, backtest, or rebuild the
cache). Manual dry-run to inspect the steps without executing anything:

```powershell
uv run python scripts/friday_pipeline.py --dry-run
```

- **Do NOT re-explain the automation here.** `AUTOMATION.md` is the single source of truth
  for HOW the Friday automation runs: what fires and when, the 5-phase orchestrator flow, the
  18 steps, where outputs and logs land, the offseason no-op, and -- most importantly -- the
  canonical "did Friday succeed?" signal (the D-04 success triad: `log.status` in
  `{success, degraded}` AND the predictions CSV exists). Read `AUTOMATION.md` for all of that.
- **For the Windows Task Scheduler setup** (registering / re-pointing the
  `NFL_Predict_Pipeline` scheduled task), see `deployment/README.md`. The runbook does not
  absorb the scheduling detail -- `deployment/README.md` owns it.
- **Verification basis (the `--dry-run`):** verified live 2026-05-31 -- exit 0 this session;
  listed all 18 steps and executed nothing (`--dry-run` bypasses the offseason no-op so the
  steps can be inspected year-round).
- **Verification basis (a live scheduled run):** verified via AUDIT-01 / AUTO-03 -- the owner
  registered and ran the scheduled task (`Last Result = 0`); see `AUTOMATION.md` Section 9.
  Not re-run here.

---

## Architecture & Data Flow

This is the DOC-02 operator view: a "what do I look at when stage X misbehaves?" map.
Per D-03 the data-flow / architecture explanation lives HERE inside the runbook -- there is
no standalone `ARCHITECTURE.md`. For the single canonical diagram (the data-flow pipeline and
the model/feature architecture), cross-reference the **Architecture** section of `README.md` --
do NOT expect a second diagram here. The text below is the operator's reads/writes/where-to-
look table that complements that diagram.

The system is the 7-stage pipeline from `PIPELINE.md`:
ingest -> features -> train -> backtest -> predict -> build-cache -> serve. Every data
location is gitignored, so on a fresh checkout you build it from ingest (see Setup above).

| Stage | Reads | Writes | Where to look on failure |
|-------|-------|--------|--------------------------|
| Ingest | live APIs (nflreadpy, Odds API, Open-Meteo) | `data/bronze/*.parquet` (append-only) -> `data/silver/{games,odds_snapshot,weather}.parquet` | `data/bronze/` timestamped snapshots; a Pydantic quality-gate error fails the whole batch (1 bad row fails all) |
| Features | silver | `data/silver/` (elo, team_form, contextual/weather/market tables) -> `data/gold/features_{wp,ats,ou}.parquet` | `data/gold/` (3 matrices); the LeakageGate diagnostic at `outputs/diagnostics/leakage_<ts>.json` |
| Train | gold | `artifacts/{target}_{ts}/` + `artifacts/latest.json` | `artifacts/latest.json` manifest -- it points at the production versions |
| Backtest | gold + artifacts | `outputs/backtest/*.{html,csv,json}` | `outputs/backtest/backtest_report.html` + `metrics_summary.json` |
| Predict | gold + artifacts (via `latest.json`) | `outputs/predictions/predictions_<S>_week<W>.{csv,json}` + `game_context_*.csv` | `outputs/predictions/`; `wp_prob` must be in `[0,1]` |
| Build cache | gold + artifacts + backtest outputs | `data/web_cache.duckdb` (~6 MB, the read-only API source) | UIAP-01: the API reads ONLY this file; skip it -> serve shows stale/empty |
| Serve | `data/web_cache.duckdb` | -- (serves http://localhost:8000) | the `/health` endpoint; single-worker envelope (`--workers 1`) |
| Friday automation | the current-week subset of the above | `outputs/predictions/` + `logs/friday_pipeline.json` | the success triad (`log.status` in `{success, degraded}` + predictions CSV exists) -- see `AUTOMATION.md` Section 6 |

**The two DuckDB files are different -- keep them distinct:**

- `data/web_cache.duckdb` (~6 MB) -- the API-FACING cache. The FastAPI app reads ONLY this
  file (the UIAP-01 boundary). It is rebuilt by `populate_cache.py`. If the UI looks stale or
  empty, rebuild this and restart serve.
- `data/nfl_predictions.duckdb` (~37 MB) -- the ANALYTICS / ad-hoc store. It is never on the
  request path; the API does not read it. Do not confuse a stale analytics DB with the cache.

**Data layers in one line:** Bronze is the raw append-only source snapshot; Silver is the
cleaned, schema-validated, latest-wins tables; Gold is the per-target feature matrices the
models consume. The Friday 18:00 ET snapshot partition under `data/silver/` is what makes any
prediction reproducible from the same input snapshot (see `README.md`).

---

## Troubleshooting & Recovery

Keyed off each stage's success signal above:

- **`serve` returns nothing / pages are empty or stale.** The cache is missing or stale.
  Rebuild it (`populate_cache.py`) and restart serve -- the API reads ONLY
  `data/web_cache.duckdb` (UIAP-01). Confirm `/health` returns 200 with `cache_ready: true`.
- **`/health` reports `status: "degraded"` in the offseason.** Expected. `/health` returns
  200 for both healthy and degraded; offseason degraded is a data-freshness state, not a
  failure, as long as `cache_ready: true` and `all_models_exist: true`.
- **`predict` produces no file / `wp_prob` out of `[0,1]`.** Check that gold exists
  (`data/gold/features_{wp,ats,ou}.parquet`) and that `artifacts/latest.json` resolves to the
  production artifact dirs. Re-run predict for the target week.
- **`build_features` fails the LeakageGate.** Read the diagnostic at
  `outputs/diagnostics/leakage_<ts>.json`; a leakage failure means a feature referenced
  future data. Note that `build_features.py` writes gold by default (there is no `--no-save`),
  so running it at all is a GOLD REBUILD forbidden under D-07 -- do NOT re-run it to "force it
  through."
- **Ingest fails on a single bad row.** The Pydantic quality gate fails the whole batch on one
  bad row. Inspect the timestamped snapshot in `data/bronze/` and re-run the ingest for that
  season/week.
- **Friday automation: did it succeed?** Use the D-04 success triad in `AUTOMATION.md`
  Section 6: `log.status` in `{success, degraded}` AND
  `outputs/predictions/predictions_<S>_week<W>.csv` exists. Exit code is 0 for BOTH success
  and degraded; only `failed` exits 1. The log is `logs/friday_pipeline.json`
  (single-file overwrite -- it reflects the MOST RECENT run only).
- **Recovery is re-running the stage, not editing data by hand.** Every stage is
  re-runnable; Silver upserts are latest-wins and the gitignored outputs are regenerated.
  Recovering does NOT mean a re-fit or a gold rebuild -- those are the DESTRUCTIVE operations
  above and stay out of routine operation (D-07).

---

## Cross-references

- **`PIPELINE.md`** -- the canonical 7-stage run sequence (ingest -> features -> train ->
  backtest -> predict -> build-cache -> serve). Owns the per-stage commands this runbook
  mirrors; it is the command source of truth.
- **`AUTOMATION.md`** -- the Friday automation explanation: what fires and when, the 18
  orchestrator steps, where logs land, and the D-04 "did Friday succeed?" success triad. The
  "Run automation" operation links it instead of re-explaining it.
- **`README.md`** -- the portfolio-facing system narrative and the single architecture
  diagram the DOC-02 section above cross-references (no duplicate diagram lives here).
- **`MODEL-DIAGNOSIS.md`** -- the honest per-target accuracy diagnosis (WP/ATS = CEILING,
  O/U = MIXED) and the production-vs-backtest population distinction. Consult it before
  reasoning about model quality; the deployed v1.0 artifacts differ from the backtest models.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit: the adopted
  canonical gold, the FIX-01 cluster, the AUDIT-01 stage-runner evidence cited by the
  DESTRUCTIVE-command labels above, and the deferred findings.
- **`STATE-OF-SYSTEM.md`** -- the consolidated state-of-the-system registry: what is
  trustworthy now, what was fixed this milestone, and the single list of deferred items.
- **`deployment/README.md`** -- the Windows Task Scheduler setup for the Friday automation
  (registering / re-pointing the `NFL_Predict_Pipeline` task). The "Run automation" operation
  links it for scheduling.

---

*Phase 23 -- Documentation, Runbook & State-of-System. DOC-01 operator runbook + DOC-02
Architecture & Data-Flow section. Link-don't-duplicate; every command verification-basis-
labeled. ASCII only (no emoji, per CLAUDE.md).*
