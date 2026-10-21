# PIPELINE.md -- Canonical Run Sequence

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
uv run python scripts/build_elo.py --all-seasons --full-rebuild
uv run python scripts/build_team_form.py --all-seasons
uv run python scripts/build_contextual.py --all-seasons
uv run python scripts/build_weather.py --all-seasons
uv run python scripts/build_features.py --all-seasons
```

- **Live entry point:** `scripts/build_*.py`.
- **Produces:** `data/silver/elo_game_snapshots.parquet`, `data/silver/team_form_features.parquet`,
  contextual / weather feature tables, then the Gold matrices
  `data/gold/features_wp.parquet`, `data/gold/features_ats.parquet`,
  `data/gold/features_ou.parquet`.

> Note: `build_elo.py` and `build_team_form.py` accept `--all-seasons` (and
> `--current` for the current season). **`build_elo.py` requires `--full-rebuild`
> alongside any historical build (`--all-seasons` / `--seasons` / `--season`) before
> it will write.** A historical build REPLACES the Elo tables, and replace-all
> against 24 seasons of burn-in is a destructive operation that has to be asked for
> rather than reached by default (Plan 33-03, T-33-12). `--current` takes the live
> append path instead, which upserts the named season and leaves prior seasons
> alone, so it needs no such flag. `build_contextual.py` and `build_weather.py`
> take `--season` / `--week` for a scoped build and `--all-seasons` for the full
> historical one; the two are mutually exclusive, and `--all-seasons` is an
> explicit NAMING alias for what omitting `--season` already did (Plan 33.1-07
> Ruling N3 -- the full build is asked for by name rather than reached by
> omission, so the documented command is literally runnable). The market-anchor
> builder still takes only `--season` and `--week`.
> `build_features.py` takes `--all-seasons`
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

#### 6a. Sub-step: generate the weekly bet list

> **Pre-cutover history (Phase 34).** Since the forward-ledger cutover on 2026-10-05 this
> sub-step REFUSES (`ForwardRowsMovedToLedgerError`): forward rows come only from the daily
> run's recommend step, and a game past its lock with no row stays unbet. See "The daily run
> and its ledger steps" below. The text that follows describes the pre-cutover path.

The `/bets` page serves a ranked +EV bet list. Its rows come from a durable artifact that
must be produced BEFORE the cache is built. This is a sub-step of Predict rather than a
stage of its own: it consumes the same week's model output and produces a recommendation
artifact, so it belongs to the predict boundary, and the canonical sequence stays at eight
stages.

```powershell
uv run python scripts/generate_bet_list.py --season <YEAR> --week <WEEK>
```

- **Live entry point:** `scripts/generate_bet_list.py` -> `backtest.weekly_bet_list.generate_weekly_bet_list`.
  Omit `--season`/`--week` and it resolves the current NFL week -- the same resolver the
  Friday orchestrator's `generate_recommendations` step uses. Supply them TOGETHER or not
  at all.
- **Produces:** `outputs/bet_list/bet_list.parquet` and `outputs/bet_list/bet_tracker.json`.
- **This is the only producer an operator can run BY HAND -- it is NOT the only producer.**
  Step 15 of the Friday orchestrator (`generate_recommendations`) writes the SAME
  `outputs/bet_list/` through the SAME facade; before this script existed it was the only
  writer, with no command an operator could run. Both callers are pinned, count exact in
  both directions, by `tests/unit/test_bet_list_entry_point.py`. So a hand run and the next
  scheduled run MERGE into one artifact under the D31-18 per-game freeze fence: forward rows
  whose own game freeze has already passed are carried forward whole and everything else is
  replaced. Hand-generate a corrected list on a Friday afternoon and expect the evening's
  scheduled run to revisit every row that is not yet frozen.

### 7. Build cache

Build the read-only DuckDB web cache the API serves from. This stage sits **between
predict and serve** because, under the UIAP-01 boundary, the FastAPI app reads ONLY
from `data/web_cache.duckdb` -- skip this and `serve` shows stale or empty data.

```powershell
uv run python scripts/populate_cache.py
```

- **Live entry point:** `scripts/populate_cache.py` -> `api.cache.populate_cache`.
- **Produces:** `data/web_cache.duckdb` (~6 MB; the only data source the API reads).
- **PRECONDITION (pre-cutover): run sub-step 6a first.** Since the cutover the forward rows
  are copied from `ledger/` and 6a is refused, so run this stage directly. Before it, for the
  bet list this stage was a pure COPY step over `outputs/bet_list/`. Skipping 6a does NOT
  make it fail: the population still exits 0
  and still writes the schedule-derived navigation and per-game freeze tables, but the
  `bet_list` table is created EMPTY with no per-week populated-at marker -- and a freeze
  present beside an absent marker is exactly the condition `/bets` treats as stale, so the
  page then REFUSES the current week. Following this document top to bottom without 6a
  reaches a served site with a refused `/bets`, and nothing in the population output says
  why.

### 8. Serve

Start the FastAPI app + web UI (single worker -- the shared DuckDB connection and
in-process cache require `--workers 1`).

```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **Recompile the stylesheet when a template introduces a new utility class.** The app serves
> `web/static/css/tailwind-compiled.css`, which is generated from `web/static/input.css` by the
> VENDORED Tailwind v4 CLI. There is **no `make` target for it** and no stage of its own: it is
> not part of the 8-stage sequence and the Makefile does not wrap it. That omission is why
> compiled-style drift exists in this repository -- a template can introduce a utility class
> that was never compiled, and the page then renders unstyled with no error. Since Phase 36 the
> website push notices: before it deploys, it compiles the stylesheet into a temporary file and
> refuses (reason `stylesheet_out_of_step`) when that fresh compile differs from the committed
> sheet, so a forgotten recompile shows up as a failed push (a `FAILED: website push` on the
> phone). The laptop's own server still does not notice. The fix is still the hand recompile
> (D-17); the push never writes the committed sheet. After editing anything under
> `web/templates/`, run:
>
> ```powershell
> ./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
> ```
>
> A template change that reuses only classes already present in the compiled sheet needs no
> recompile; when in doubt, run it -- it is idempotent and takes about a second.

- **Live entry point:** `api/main.py:app`.
- **Produces:** FastAPI at http://localhost:8000 -- the five top-nav pages `/`
  (This Week), `/bets`, `/season`, `/track-record` and `/how-it-works`, plus the
  `/games/{id}` game-detail drill-down and a `/health` endpoint. The retired URLs
  `/performance`, `/backtest`, `/insights` and `/betting` answer with a 301 to the merged
  page, query string kept.

---

## Current-week / weekly run

The per-stage commands above are for **historical / full builds** (they use
`--season` / `--all-seasons`, which the scripts actually accept). The build scripts do
**not** expose a per-stage current-week flag; the current season is built via
`--current` on the Elo/form builders, and the full current-week run goes through the
orchestrator below.

The Friday orchestrator below WAS the coherent **current-week** path. Since Plan 33.2-28 the
scheduled path is the daily run, and since the forward-ledger cutover (2026-10-05) only the
daily run writes forward bets: the orchestrator's `generate_recommendations` step logs the
cutover and writes nothing, so a manual run still refreshes data, predictions, exports and the
cache but produces no bet rows. See "The daily run and its ledger steps" below. The
orchestrator calls the builder classes directly and runs the data + predictions subset:

```powershell
uv run python scripts/friday_pipeline.py --log-level INFO
```

- **Live entry point:** `scripts/friday_pipeline.py` ->
  `pipeline.orchestrator.FridayPipeline` (23-step registry in `pipeline/steps.py`).
- **Produces:** refreshed Silver/Gold for the current week, current-week predictions
  and recommendations, exports, a rebuilt `data/web_cache.duckdb`, and a run log at
  `logs/friday_pipeline.json`.
- **Scope:** ingest games/weather -> data QA -> build Elo/form/contextual/weather ->
  ingest odds -> market anchors -> build features -> validate features/models ->
  generate predictions/recommendations -> export -> validate outputs -> **populate the
  web cache**. It does **not** train, promote, or backtest -- run stages 3 (train), 4
  (promote) and 5 (backtest) above for those.
- **Stage 7 is now part of the weekly run (changed by plan 31-18).** This document
  previously said the orchestrator does not rebuild the web cache; that is no longer
  true. `populate_web_cache` is the LAST registry entry, runs after the recommendation
  and export steps, and is registered `critical=False` so a cache failure degrades the
  run instead of discarding prediction work that already succeeded. Running stage 7 by
  hand remains the recovery path, and it is the exact command the `/bets` stale-cache
  refusal tells the reader to re-run. See `AUTOMATION.md` section 4.

### The daily run and its ledger steps (the scheduled path)

Since Plan 33.2-28 the scheduled task runs the DAILY run at 5 PM ET, not the Friday
orchestrator; `AUTOMATION.md` section 1 explains it. Since the forward-ledger cutover on
2026-10-05 (Phase 34) the daily run also keeps the ledger. Its sequence, in order:

```powershell
uv run python scripts/daily_lock_pipeline.py
```

1. **Passed-lock refusal**, before any request (at or after 18:00 ET it records `lock_passed`
   and collects and predicts nothing). Since Phase 37 that run does not simply stop: it still
   attempts a due re-fit (step 8), refreshing the schedule once first when an A/A week or a
   re-fit date is due (so a binding A/A week is judged on today's schedule), and
   makes no request on any other day (D-17). Then it goes straight to the website push and the
   notifications; it has no settle pass and no ledger sync, so whatever the re-fit recorded
   is backed up by the next run's sync.
2. **Run id.** One id is bound for the whole run and printed as `LEDGER_RUN_ID=`; every ledger
   event below carries it.
3. **Schedule refresh** -- results, moved games and new playoff rounds arrive here.
4. **Closing-trigger regeneration** -- the `NFL_Predict_Closing` wake task is re-registered with
   triggers for the next 8 days of real kickoffs (`CLOSING_TRIGGERS=`). Every refreshed day,
   whether or not tomorrow has games; a skip or failure is recorded and never fails the run.
5. **Settle pass** -- one ledger write that grades settled rows from the refreshed scores,
   appends any correction entry a corrected score is owed, and finalizes closing-line columns
   (`LEDGER_SETTLE=`). A failed settle is recorded and never stops tonight's decisions.
6. **Select tomorrow's games.** None: the ledger sync (step 9) backs up the settle pass's
   write, the re-fit side step (step 8) runs next, then the web
   cache is rebuilt once if anything changed -- `WEB_CACHE_AFTER_REFIT=` when the re-fit step
   wrote records, otherwise `WEB_CACHE_AFTER_SETTLE=` when the settle pass changed the ledger, so
   the new grades reach `/bets` -- then a `no_games` record, the website push and the
   notifications.
7. **The run proper** -- collection, the full-history gold build, prediction -- through the
   daily step registry. Its recommend step is the **ledger write**: tomorrow's forward bets are
   stamped and appended to `ledger/forward_2026.jsonl` (first pick stands: a re-run that decides
   a stored row differently is refused and printed as `LEDGER_REFUSED=`), and the web cache is
   then rebuilt from the ledger.

   The same ledger write carries the frozen arm (Phase 37, D-04, RFIT-03). For every NEW live
   row it writes a shadow twin (`arm` = `shadow`), re-decided from the same inputs with the
   pinned row-19 models, blend and converter in `backtest/shadow_pin_2026.py` and the row-19 bet
   rule, in the same write, live rows first. A shadow failure -- including too little time left
   before the slate's deadline -- never touches a live row: each new live row left without a twin
   is a hole, recorded in `ledger/refit/shadow_holes.jsonl` by exception class only and sent as
   one FAILED push (folded into the run's first failed push when there is one). Shadow rows are
   never shown as bets: `/bets` and the Saturday email read the live arm only, and only the
   CSV/JSON export carries them, each with its `arm`.
8. **Re-fit side step** (Phase 37, D-15, D-17). It runs on every day that got past the picks: a
   prediction day whose run ended `success`, `finished_with_skips` or `degraded` (after the
   ledger write), a `no_games` day (step 6) and a `lock_passed` day (step 1). It never runs on
   `--dry-run`, on a refused run date (`RUN_REFUSED=`), on a run that failed, or after Ctrl+C.
   It always runs right after the ledger sync (step 9) and before the cache rebuild, the website
   push and the notifications, so that day's push and email report it.
   - **Calendar only.** Whether anything is due comes from the pre-registered date lists
     (`backtest/refit_preregistration_2026.py`, `ROLLING-REFIT-PREREGISTRATION.md`), the clock
     and its own records, never from a metric. A day with nothing due takes no lock and writes
     nothing.
   - **One attempt at most.** Otherwise it works under the attempt lock
     (`ledger/refit/attempt.lock`; a second run finds it held and reports `busy`): it closes an
     install a crash interrupted, records every A/A week now due (only until a clean week puts a
     date list in force), closes older dates `not run`, then makes at most one attempt -- the full-history gold build,
     training, the re-measure, the integrity checks and the swap through `artifacts/latest.json`.
   - **Contract line.** `REFIT= <status> date=<date or none> check=<check or none>`, where the
     status is one of `waiting`, `not_due`, `no_refit_this_season`, `ran`, `ran late`,
     `refused`, `crashed` or `busy` (`refit.hook.REFIT_STATUSES`), or `REFIT= failed <Class>`
     when the step itself raised (followed by `at ledger/refit/<file> line <n>` when a re-fit
     record would not read; `RUNBOOK.md` has the repair). When it wrote records the web cache is
     rebuilt once more,
     printed as `WEB_CACHE_AFTER_REFIT=`.
   - **Never the exit code.** A refused or crashed re-fit never changes the run's exit code; its
     date stays open and the next daily run retries it with the same cutoff. Every problem sends
     one `FAILED:` push. New models serve from the next daily run.
   - Its records are listed in `RUNBOOK.md`, "Forward ledger".
9. **Ledger sync**, once per run: a new anchor commit on `ledger-anchor` when the head moved,
   pushed to GitHub, then the private backup committed and pushed (`LEDGER_SYNC=`). On a day that
   reaches the re-fit side step it runs right after the day's ledger write and BEFORE the re-fit
   (D-15 as amended by the owner on 2026-10-21), so a long attempt -- or a laptop unplugged during
   one -- never holds back the anchor and backup of the day's rows; whatever the re-fit records is
   backed up by the next run's sync. A run that never reaches the re-fit (the offseason, a
   pipeline that failed, a run that raised) syncs at the end instead; a `lock_passed` run and a
   dry run have no sync. A failed push is
   recorded in `logs/ledger_runs.jsonl` and retried by the next sync; it never changes the run's
   exit code.

   The resulting order on a prediction day that got past the picks: the ledger write (step 7),
   the ledger sync, the re-fit side step, the web cache rebuild when the re-fit wrote records, the
   website push, the notifications (failed pushes, the bets push, the Saturday email) and the
   watcher check-in last. A `no_games` day: the settle pass, the ledger sync, the re-fit side step,
   the one cache rebuild when anything changed, the `no_games` record, the website push, the notifications and the
   check-in.

Forward rows now come ONLY from this run. `scripts/generate_bet_list.py` (sub-step 6a) refuses
to write forward rows after the cutover (`ForwardRowsMovedToLedgerError`), and the Friday
orchestrator's `generate_recommendations` step logs the cutover and writes nothing. The script
has no replay mode, so 6a is refused outright; `outputs/bet_list/` keeps only the 2025 replay rows. The owner procedures -- verify, replay, restore, recording a real bet and
a manual `uv run python -m scripts.sync_ledger` -- are in `RUNBOOK.md`, "Forward ledger".
