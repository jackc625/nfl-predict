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
> does not re-explain the 19 steps), and `README.md` owns the single architecture diagram
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

A fresh clone has the code, but NO data AND NO model artifacts. Every data, output, AND
artifact location is gitignored, so none of it travels with the checkout:

- `data/` -- Bronze/Silver/Gold parquet, the DuckDB files (gitignored).
- `outputs/` -- predictions, backtest reports (gitignored).
- `artifacts/` -- model directories + the `latest.json` manifest are ALL gitignored
  (`.gitignore` blankets `artifacts/` and tracks only `artifacts/.gitkeep`). `git ls-files
  artifacts/` returns nothing. A fresh checkout therefore has NO model dirs and NO
  `latest.json` -- they must be (re)built locally. (A correction: an earlier version of this
  runbook claimed the bundled v1.0 production artifacts shipped inside the checkout; that was
  FALSE against the actual `.gitignore` and is corrected here -- nothing under `artifacts/`
  is tracked.)
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
   - if you already have a populated `data/` from a prior run, existing gold + locally-built
     artifacts (next step) are enough to run **Predict**, **Build cache**, and **Serve**.

4. Get model artifacts AND the first manifest (clean-checkout bootstrap). Because
   `artifacts/` is fully gitignored, a fresh checkout has no model dirs and no
   `artifacts/latest.json`. Train the models (**Train** operation below) to write the
   versioned candidate dirs, then mint the FIRST manifest with a one-time per-target
   `update_manifest` call:

   ```powershell
   uv run python -c "from models.artifacts import update_manifest; [update_manifest(t, '<target>_<UTCtimestamp>') for t in ()]"
   ```

   Concretely, after `train_models.py --target all` writes (for example)
   `artifacts/wp_<ts>/`, `artifacts/ats_<ts>/`, `artifacts/ou_<ts>/`, point the manifest at
   each by its real dir name:

   ```powershell
   uv run python -c "from models.artifacts import update_manifest; update_manifest('wp', 'wp_<ts>'); update_manifest('ats', 'ats_<ts>'); update_manifest('ou', 'ou_<ts>')"
   ```

   This is the SANCTIONED first-manifest bootstrap (no new flag, no bootstrap script --
   `models.artifacts.update_manifest` already exists and is exactly the one-time call).
   Why a manual bootstrap is needed: training defaults `update_latest=False` (since Plan
   24-01) so it never writes the manifest, and the gated **Promote** operation REQUIRES a
   pre-existing `latest.json` (it re-scores the deployed baseline for the paired
   non-regression delta and raises an actionable `FileNotFoundError` if the manifest is
   absent). That is a deliberate chicken-and-egg: the gate needs a prior baseline, so it
   cannot mint the very first manifest. After this one-time bootstrap, EVERY subsequent
   deploy goes through the gated **Promote** operation -- never `update_manifest` by hand
   again.

`PIPELINE.md` is the canonical 8-stage run sequence; the per-operation sections below match
its commands (the Build-features stage is summarized to its final assembly step -- see that
section and `PIPELINE.md` for the full component-build sequence).

---

## Common operations

There are 11 common operations. Each section gives the `uv run` PowerShell command, how to
tell it succeeded, and a verification-basis label. The canonical commands match `PIPELINE.md`;
for "Build features" this runbook summarizes the stage down to its final assembly step
(`build_features.py`) -- `PIPELINE.md` lists the full per-component build sequence (build_elo,
build_team_form, build_contextual, build_weather, then build_features).
When in doubt, `PIPELINE.md` is the command source of truth. Operations 1-8 follow the
8-stage PIPELINE order (Promote is stage 4, between Train and Backtest); Rollback (operation
9) reverses a Promote, Run automation (operation 10) is the Friday orchestrator, and Rebuild
gold (operation 11) is the attributed full-history rebuild Phase 30 added.

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
uv run python scripts/build_features.py --all-seasons
```

- **WARNING -- `build_features.py` writes gold by DEFAULT.** `--save` is defined
  `action="store_true"` with `default=True`, so the bare flag can never turn saving OFF;
  any real `--season`/`--all-seasons` build materializes the canonical gold matrices
  (`data/gold/features_{wp,ats,ou}.parquet`). That is a GOLD REBUILD, forbidden under D-07.
  The `build_*.py --all-seasons` feeders likewise rebuild the full historical feature tables.
- **`--all-seasons` REPLACES gold; `--season <YEAR>` MERGES into it.** The full build
  carried every season, so its frame IS the table and is written as such -- that is the
  only mode that can change the gold schema (it is what the Phase-30 rung-3 column drop
  used). A scoped `--season` / `--week` build carried only that slice, so it is merged
  latest-wins on `game_id` with every other season preserved, and a scoped build whose
  schema differs from gold's is REFUSED before it writes. Reach for `--all-seasons`, never
  `--season`, when what you want is a historical rebuild.
- **The read-only build IS available now: pass `--no-save`.** An earlier version of this
  runbook recorded that the save flag was a no-op with no way to disable the gold write. That
  defect is FIXED in the code: `scripts/build_features.py` defines a real `--no-save`
  (`action="store_false"`, `dest="save"`) beside an explaining comment, so a build can be run
  without writing gold. `--help` remains non-destructive. A build WITHOUT `--no-save` is still
  a gold rebuild -- the off switch has to be passed, not assumed.
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

- **Succeeded when:** new `artifacts/{target}_<UTCtimestamp>/` candidate directories appear.
  Training writes versioned CANDIDATE dirs ONLY -- it does NOT touch production: since Plan
  24-01 `save_model_artifact` defaults `update_latest=False` and the trainer never overrides
  it, so a train run never rewrites `artifacts/latest.json`. To deploy a freshly trained
  candidate, run the **Promote** operation below (the gate decides per target). Pass
  `--target wp|ats|ou` for a single target.
- **DESTRUCTIVE:** this is a re-fit (it produces new model artifacts), which D-07 forbids in
  the v2.1 trust milestone. (It is the core sanctioned operation in v3.0, run THROUGH the
  Promote gate, not the bare manifest overwrite this note once described.)
- **Verification basis:** verified via AUDIT-01 stage runner; not re-run (a re-fit, forbidden
  by D-07) (AUDIT-REPORT.md Stage 3: `--help` exit 0, "NOT executed -- D-01 no re-train").

### 4. Promote models

Score the freshly trained candidates against the FROZEN per-target deploy gate
(`config/gate.toml`) and, on `--promote`, conditionally swap ONLY the gate-passing targets
into production. This is the sole sanctioned path that rewrites the `artifacts/latest.json`
target keys; training (operation 3) never does. Dry-run by default (prints the per-target
2x2, swaps nothing); pass `--promote` to perform the conditional swap.

```powershell
uv run python -m scripts.promote_models
uv run python -m scripts.promote_models --promote --skip-train
```

- **Arm the run you REVIEWED.** A bare `--promote` re-trains the candidates into staging
  first, so the artifact that ships is not the artifact the dry run scored. The sequence to
  use is: bare dry-run (trains into staging and scores) -> read the printed 2x2 ->
  `--promote --skip-train`, which REUSES the exact gate-scored staging directories. The armed
  run prints a staleness warning naming each staged directory and its timestamp; read it, do
  not suppress it. This is the sequence the Phase-30 armed run used.
- **Succeeded when (dry-run):** the per-target 2x2 readout prints (candidate vs the FROZEN
  `[baseline.*]` block in `config/gate.toml`: pooled + per-season CLV non-regression and the
  secondary metrics) and the process exits NON-ZERO if any target FAILS the gate (the hard
  block is observable to CI). Nothing in production changes. The frozen baseline describes the
  DEPLOYED incumbent, not v1.0 -- it was re-pointed at the deployed set in Phase 25 (D25-11)
  and re-frozen twice more in Phase 30, once before the gate ran because the gold rebuild had
  moved the values it was measured on, and once after the promotion so it describes the end
  state. A non-zero exit on a PARTIAL pass is correct reporting, not an error to suppress.
- **Succeeded when (`--promote`):** ONLY gate-passing targets are copied into `artifacts/`
  and pointed at by `artifacts/latest.json` (verify with
  `uv run python -c "import json,pathlib; print(pathlib.Path('artifacts/latest.json').read_text())"`);
  a FAILING target keeps its existing production entry (honest refusal is a valid outcome).
  The prior version dirs are retained (never deleted), so the swap is reversible (see
  **Rollback** below). The deployed artifact is byte-identical to the gate-scored staging
  artifact.
- **DESTRUCTIVE on `--promote`:** it rewrites `artifacts/latest.json` target keys and copies
  staged artifact dirs into production. The bare dry-run (no `--promote`) is SAFE
  (scores + prints only, swaps nothing). Requires a pre-existing `artifacts/latest.json`
  (the gate re-scores the deployed baseline); on a clean checkout, mint the first manifest
  via the one-time bootstrap in "Setup from a fresh checkout" above.
- **Verification basis:** verified via the Phase 25 owner-attended armed run (D25-18) --
  the dry-run 2x2 then the armed `--promote` ran this session and activated WP + ATS while
  retaining v1.0 OU; the full record (pre/post manifest content + sha256, the gate 2x2, the
  fix-cycle deltas) is in `ACTIVATION-READOUT.md`.
- **Verification basis (Phase 30, the most recent armed run):** the same two-step sequence ran
  again on 2026-08-24 against the rebuilt gold, this time arming with
  `--promote --skip-train`. WP PASSED and was promoted to `wp_20260824_113325`; ATS and O/U
  FAILED the gate and RETAINED their incumbents (`ats_20260605_220128` and
  `ou_20260326_163930`) byte-unchanged. The run exited 1, which is the correct partial-pass
  report. Two of three targets refusing is the gate doing its job, in the same D25-14 lineage
  as the Phase-25 O/U refusal -- a refusal is a RESULT, not a failed run. The per-target
  numbers behind BOTH the promotion and the two refusals are in `GATED-REFIT-READOUT.md`.

### 5. Backtest

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

### 6. Predict

Generate predictions for a specific season/week using the trained artifacts.

```powershell
uv run python scripts/generate_current_week_predictions.py --season <YEAR> --week <WEEK>
```

- **Succeeded when:** `outputs/predictions/predictions_<YEAR>_week<WEEK>.csv` and `.json`
  plus `game_context_<YEAR>_week<WEEK>.csv` are written; `wp_prob` must fall in `[0,1]`.
  Pass `--no-blend` to skip market blending.
- **SAFE:** loads the deployed artifacts via `artifacts/latest.json` (no train); writes only
  gitignored `outputs/predictions/`. As of the Phase 25 activation the deployed set was the
  activated WP + ATS re-fits on canonical Elo gold plus the retained v1.0 O/U (see
  `ACTIVATION-READOUT.md`). After the Phase-30 gated re-fit the deployed set is WP
  `wp_20260824_113325` (promoted in Phase 30), ATS `ats_20260605_220128` (the Phase-25 re-fit,
  RETAINED -- its Phase-30 candidate was refused) and O/U `ou_20260326_163930` (the v1.0
  pre-Elo model, RETAINED through both gates); see `GATED-REFIT-READOUT.md`. Predict always
  follows whatever `latest.json` points at, so it needs no update when a pointer moves.
- **Verification basis:** verified live 2026-05-31 -- ran `--season 2024 --week 18` that
  session: exit 0, 16 games, loaded the THEN-deployed v1.0 artifacts
  (`wp_20260327_114739` / `ats_20260326_163724` / `ou_20260326_163930`), wrote
  `predictions_2024_week18.csv` + `game_context_2024_week18.csv`, no re-fit, no live API.
  (That dated run predates the Phase 25 swap; the deployed WP/ATS versions have since changed
  per `ACTIVATION-READOUT.md` -- predict resolves them through `latest.json` either way.)

### 7. Build cache

Build the read-only DuckDB web cache the API serves from. This sits BETWEEN predict and
serve: under the UIAP-01 boundary the FastAPI app reads ONLY from `data/web_cache.duckdb`,
so skipping this leaves `serve` showing stale or empty data.

```powershell
uv run python scripts/populate_cache.py
```

- **Succeeded when:** `data/web_cache.duckdb` (~6 MB) is refreshed -- the only data source
  the API reads.
- **This is also what populates `/bets` (Phase 31).** The same run materializes the weekly
  bet-list rows and the precomputed realized-vs-expected tracker blocks from
  `outputs/bet_list/` -- a directory operation 12 below is the only HAND-RUNNABLE producer of,
  NOT the only producer: step 15 of the Friday orchestrator (`generate_recommendations`) writes
  it through the same facade. For the
  bet list this operation is a pure COPY step, so run operation 12 FIRST. `/bets` computes
  nothing on the request path, so a week whose bet-list blob is missing or stale is
  HARD-BLOCKED by the page rather than served from a partial cache -- and the refusal message
  names both commands, in order, as the recovery path.
- **Skipping operation 12 does not make this operation fail.** It still exits 0 and still
  writes the schedule-derived navigation and per-game freeze tables, but the `bet_list` table
  is created EMPTY with no per-week populated-at marker, and a freeze present beside an absent
  marker is exactly the condition `/bets` treats as stale -- so the page REFUSES the current
  week. The log line to look for is the warning naming `scripts/generate_bet_list.py`.
- **SAFE:** a downstream DATA refresh from gold + backtest outputs; NOT a re-fit. The file is
  gitignored, so the refresh mutates only a gitignored file.
- **A rebuild holds the publish lock (Phase 36).** The website push and every cache rebuild share
  one lock, so `scripts/populate_cache.py` refuses (`POPULATE_CACHE= refused`, exit 1) while a
  push runs; wait for the push to finish and run it again.
- **Verification basis:** verified live 2026-05-31 (exit 0 this session; refreshed the
  gitignored `data/web_cache.duckdb` -- 1139 predictions, 6263 game_context rows).

### 8. Serve

Start the FastAPI app + web UI (single worker -- the shared DuckDB connection and in-process
cache require `--workers 1`).

```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Succeeded when:** FastAPI is reachable at http://localhost:8000 with the seven nav pages
  `/`, `/performance`, `/backtest`, `/insights`, `/betting`, `/season`, `/bets`, the
  `/games/{id}` detail drill-down, and a `/health` endpoint that
  returns HTTP 200. `/health` returns 200 for BOTH a healthy and a `degraded` status; in the
  offseason it is expected to report `status: "degraded"` (a data-freshness state) while
  still returning 200 with `cache_ready: true` and `all_models_exist: true`.
- **SAFE:** read-only DuckDB connection; per the UIAP-01 boundary the API does not import
  `models/`, `features/`, or `ratings/` and loads no model.
- **The "Out of date" bar and the data-format notice appear locally too (Phase 36),** under the
  same rules as on the hosted copy; see "The "Out of date" bar and the data-format notice" in the
  Website section below.
- **Recompile the stylesheet after ANY template edit that introduces a new utility class.**
  There is no `make` target for this and no stage of its own. Since Phase 36 the website push
  notices a stale sheet and refuses (`stylesheet_out_of_step`), but the laptop's own server does
  not: a template can use a class that was never compiled and the local page renders unstyled
  with no error:

  ```powershell
  ./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
  ```

  It is idempotent and takes about a second, so run it when in doubt. `PIPELINE.md` stage 8
  owns the full explanation. **A known instance is OPEN right now:** `lg:grid-cols-7` is
  absent from `web/static/css/tailwind-compiled.css`, so `/betting`'s KPI grid is unstyled at
  the large breakpoint. It is pre-existing and recorded rather than fixed
  (`PROFITABILITY-READOUT.md` section 7e).
- **Verification basis:** verified live 2026-05-31 -- started uvicorn this session and reached
  http://localhost:8000/health: HTTP 200, `cache_ready: true`, `all_models_exist: true`,
  `status: "degraded"` (the expected offseason data-freshness state), then stopped it.

### 9. Rollback (reverse a Promote)

Reverse a production swap by re-pointing the `artifacts/latest.json` manifest keys back to
the PRIOR version dirs. `update_manifest` is the sole per-key swapper, so a rollback is a
pure manifest restore -- the prior artifact dirs are retained (never deleted), so no re-fit
and no copy is needed. There is no `--rollback` CLI flag: rollback is the same per-key
`update_manifest` call used elsewhere, run once per key against the recorded pre-swap mapping.

The pre-swap (and post-swap) manifest version map + sha256 are recorded in
`ACTIVATION-READOUT.md` (the D25-17 rollback record) and, for the Phase-30 swap, in
`GATED-REFIT-READOUT.md`, so the rollback target is a verifiable, written-down mapping rather
than a guess.

**To roll back the Phase-30 swap (the most recent one).** Phase 30 moved exactly ONE key: WP,
from `wp_20260605_215552` to `wp_20260824_113325`. ATS, O/U and the blend pointer were not
touched, so reversing Phase 30 is a single-key restore:

```powershell
uv run python -c "from models.artifacts import update_manifest; update_manifest('wp', 'wp_20260605_215552')"
```

The pre-swap and post-swap manifest sha256 are published as `MANIFEST_SHA256_BEFORE` and
`MANIFEST_SHA256_AFTER` in the tracked `tests/phase30_state.py` and in
`GATED-REFIT-READOUT.md`, so the restore is checkable rather than assumed.

**To roll back the Phase 25 activation to the pre-swap v1.0 mapping** (this reverses BOTH
swaps if applied after the single-key restore above):

```powershell
uv run python -c "from models.artifacts import update_manifest; update_manifest('wp', 'wp_20260327_114739'); update_manifest('ats', 'ats_20260326_163724'); update_manifest('ou', 'ou_20260326_163930')"
```

If the post-swap blend-mode re-validation rewrote the blend pointer (it did this session, via
`save_blend_artifacts` -- the one sanctioned `latest.json['blend']` write outside the
promotion swap), the blend key can be restored the same way:

```powershell
uv run python -c "from models.artifacts import update_manifest; update_manifest('blend', 'blend_dynamic_20260526_194510')"
```

- **Succeeded when:** `artifacts/latest.json` equals the recorded pre-swap mapping. Verify by
  parsed-equality + sha256 against the `ACTIVATION-READOUT.md` record:
  `uv run python -c "import json,pathlib; print(pathlib.Path('artifacts/latest.json').read_text())"`
  (the pre-swap sha256 is `14f9093edab0e7cffef968621b5d73f29534082f23add4253c6b61e8fc8b2beb`).
  `load_model_artifact('wp'|'ats'|'ou')` must all resolve after the restore.
- **DESTRUCTIVE:** it rewrites `artifacts/latest.json` (the sole production swap surface). It
  is reversible in turn -- re-applying the post-swap mapping re-activates the swap -- because
  every version dir is retained.
- **Verification basis:** verified via the committed `tests/integration/test_rollback.py` --
  it seeds the recorded pre-swap mapping, simulates the swap, restores via `update_manifest`,
  and asserts PARSED-EQUALITY + a recorded-seed sha256 + a `Path.replace` atomicity guard
  (D25-17). Not re-run by hand against production here.

### 10. Run automation (the Friday pipeline)

The weekly run is the Friday orchestrator -- the current-week subset of the stages above
(ingest -> features -> predict -> validate; it does NOT train or backtest). **The weekly path
CHANGED in Phase 31:** the orchestrator now also selects the week's +EV bet list through
`BetSelector`, writes the two durable `outputs/bet_list/` artifacts, and REBUILDS the web
cache as its last registry entry, so the bet list the site serves after a Friday run is this
run's. That last step is registered non-critical, so a cache failure degrades the run instead
of discarding prediction work that already succeeded; running operation 7 by hand is then the
documented recovery. Earlier revisions of this runbook said the orchestrator does not rebuild
the cache -- that statement is superseded, not merely out of date. Manual dry-run to inspect
the steps without executing anything:

```powershell
uv run python scripts/friday_pipeline.py --dry-run
```

- **Do NOT re-explain the automation here.** `AUTOMATION.md` is the single source of truth
  for HOW the Friday automation runs: what fires and when, the 5-phase orchestrator flow, the
  19 steps, where outputs and logs land, the offseason no-op, and -- most importantly -- the
  canonical "did Friday succeed?" signal (the D-04 success triad: `log.status` in
  `{success, degraded}` AND the predictions CSV exists). Read `AUTOMATION.md` for all of that.
- **For the Windows Task Scheduler setup** (registering / re-pointing the
  `NFL_Predict_Pipeline` scheduled task), see `deployment/README.md`. The runbook does not
  absorb the scheduling detail -- `deployment/README.md` owns it.
- **Verification basis (the `--dry-run`):** verified live 2026-05-31 -- exit 0 this session;
  listed all 19 steps and executed nothing (`--dry-run` bypasses the offseason no-op so the
  steps can be inspected year-round).
- **Verification basis (a live scheduled run):** verified via AUDIT-01 / AUTO-03 -- the owner
  registered and ran the scheduled task (`Last Result = 0`); see `AUTOMATION.md` Section 9.
  Not re-run here.

### 11. Rebuild gold (attributed)

A full-history gold rebuild, judged mechanically against a signature declared BEFORE the
rebuild runs. This is the Phase-30 discipline and it exists because a rebuild run for several
reasons at once makes every moved column unattributable. Capture a fingerprint, rebuild for
exactly ONE named cause, then have the judge attribute the diff to that cause:

```powershell
uv run python scripts/fingerprint_gold.py --out outputs/fingerprints/before.json
uv run python scripts/build_features.py --all-seasons
uv run python scripts/fingerprint_gold.py --out outputs/fingerprints/after.json
uv run python scripts/fingerprint_gold.py --compare outputs/fingerprints/before.json outputs/fingerprints/after.json --attribute-rung 2
```

- **Succeeded when:** the attribution judge EXITS 0 -- every moved column is attributed to the
  rung's one named cause and `unattributed` is empty. Judge by the exit code, not by reading
  the printed report, and do not pipe the command through anything that masks `$?`. Exit 1 is
  a BLOCKING verdict, exit 3 a non-blocking finding, exit 0 clean.
- **Attribution is NOT health.** A rung whose signature attributes every changed column can
  still have destroyed columns silently -- that happened in Phase 30, where a rung attributed
  perfectly cleanly while a bound-fitting cascade had flattened 18 columns. Re-measure
  per-column health (newly constant, newly all-NaN, newly constant within any season) against
  a committed pre-rebuild fixture as well. Measurement beats attribution.
- **Count widths off the parquet, not off the build's summary print.** The build's own
  `Features:` line is low by a constant and was never used for any published Phase-30 number.
- **DESTRUCTIVE:** this rewrites `data/gold/features_{wp,ats,ou}.parquet`. It reads
  `nflreadpy` LIVE with no cache, so an upstream revision landing between two rungs is
  indistinguishable from the rung's own named cause by inspection alone -- publish the build
  timestamps so that hypothesis stays checkable. Pass `--no-save` to `build_features.py` for a
  read-only build that writes no gold.
- **Verification basis:** verified via the Phase-30 four-rung rebuild ladder (2026-08-22) --
  four rungs plus a reproduction re-run, each judged by exit code; gold moved from 209/210/209
  at 6,263 rows to 194/195/194 at 6,499 rows. The per-rung causes, timestamps, verdicts and
  the reproduction result are in `GATED-REFIT-READOUT.md`. Not re-run here.

### 12. Generate the weekly bet list

Produce the durable artifacts the `/bets` page ultimately serves. This is STAGE ONE OF TWO:
operation 7 above copies these artifacts into the web cache, and from a cold start it cannot
do anything useful until they exist. Appended as operation 12 rather than inserted next to
operation 7 so the eleven existing heading anchors do not renumber (the Phase-30 precedent).

```powershell
uv run python scripts/generate_bet_list.py --season <YEAR> --week <WEEK>
```

Omit `--season`/`--week` and it resolves the current NFL week using the same resolver the
Friday orchestrator step uses. Supply them TOGETHER or not at all -- one alone would pair an
explicit value with a resolved one and select a week nobody asked for, which the command
refuses by name.

- **Succeeded when:** exit 0 and BOTH files are written --
  `outputs/bet_list/bet_list.parquet` and `outputs/bet_list/bet_tracker.json`. The pair is
  indivisible: the parquet feeds the ranked list and the JSON feeds the realized-vs-expected
  tracker, and a run that produced only the parquet would leave the tracker permanently empty
  while the page still looked correct. The closing log line names both paths.
- **Then run operation 7.** Generation alone does not change the served page: `/bets` reads
  the DuckDB cache, never these files. The running server picks a rebuilt cache up on the
  next request, so no restart is needed.
- **YOU ARE NOT THE ONLY WRITER of `outputs/bet_list/`.** Step 15 of the Friday orchestrator
  (`generate_recommendations`) writes the same directory through the same facade -- this
  command is the only HAND-RUNNABLE producer, which is a narrower claim. The two merge into
  one artifact under the D31-18 per-game freeze fence: forward rows whose own game freeze has
  already passed are carried forward whole, and everything else is replaced. So a corrected
  list generated by hand on a Friday afternoon will have its not-yet-frozen rows revisited by
  the evening's scheduled run. Pass `--output-dir` if you want a run that cannot be
  overwritten.
- **SAFE:** writes ONLY under `outputs/bet_list/` (or `--output-dir`). It re-fits nothing,
  promotes nothing and touches no model artifact -- it reads the deployed artifacts through
  `artifacts/latest.json` and the pre-registered tune-only fit, and refuses loudly if either
  is absent rather than defaulting a threshold nobody swept for. No replay mode is exposed;
  the command runs FORWARD only.
- **Verification basis:** verified live 2026-09-10 -- ran
  `--season 2025 --week 3 --output-dir <temp>` that session into a TEMPORARY directory (never
  the default `outputs/bet_list/`, which is inside the protected tree): exit 0, 16 scheduled
  games, 48 candidates, 16 selected, wrote `bet_list.parquet` (19,237 bytes, 48 rows) and
  `bet_tracker.json` (237 bytes, 1 forward block), loading the deployed
  `wp_20260824_113325` / `ats_20260605_220128` / `ou_20260326_163930`. No re-fit, no
  promotion, and `git status --porcelain data/ config/ artifacts/ outputs/` empty afterwards.

---

## Architecture & Data Flow

This is the DOC-02 operator view: a "what do I look at when stage X misbehaves?" map.
Per D-03 the data-flow / architecture explanation lives HERE inside the runbook -- there is
no standalone `ARCHITECTURE.md`. For the single canonical diagram (the data-flow pipeline and
the model/feature architecture), cross-reference the **Architecture** section of `README.md` --
do NOT expect a second diagram here. The text below is the operator's reads/writes/where-to-
look table that complements that diagram.

The system is the 8-stage pipeline from `PIPELINE.md`:
ingest -> features -> train -> promote -> backtest -> predict -> build-cache -> serve. Every
data location is gitignored, so on a fresh checkout you build it from ingest (see Setup above).

| Stage | Reads | Writes | Where to look on failure |
|-------|-------|--------|--------------------------|
| Ingest | live APIs (nflreadpy, Odds API, Open-Meteo) | `data/bronze/*.parquet` (append-only) -> `data/silver/{games,odds_snapshot,weather}.parquet` | `data/bronze/` timestamped snapshots; a Pydantic quality-gate error fails the whole batch (1 bad row fails all) |
| Features | silver | `data/silver/` (elo, team_form, contextual/weather/market tables) -> `data/gold/features_{wp,ats,ou}.parquet` | `data/gold/` (3 matrices); the LeakageGate diagnostic at `outputs/diagnostics/leakage_<ts>.json` |
| Train | gold | `artifacts/{target}_{ts}/` candidate dirs ONLY (NOT `latest.json` -- `update_latest=False` since Plan 24-01) | the new `artifacts/{target}_{ts}/` candidate dirs; training never rewrites the manifest |
| Promote | gold + candidate artifacts + frozen `config/gate.toml` baseline | (on `--promote`) gate-passing target dirs copied into `artifacts/` + `artifacts/latest.json` target keys swapped (`update_manifest`, the sole per-key swapper) | the per-target 2x2 readout (non-zero exit on FAIL); `artifacts/latest.json`; `ACTIVATION-READOUT.md` (Phase 25) and `GATED-REFIT-READOUT.md` (Phase 30) for the pre/post manifest records |
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
  future data. Note that `build_features.py` writes gold by default, so re-running it bare is a
  GOLD REBUILD -- do NOT re-run it to "force it through." Pass `--no-save` if you need to
  reproduce the failure without writing gold; that flag is real (added in Phase 30), and the
  earlier note here claiming no such switch existed was true of the code at the time and is
  no longer.
- **Ingest fails on a single bad row.** The Pydantic quality gate fails the whole batch on one
  bad row. Inspect the timestamped snapshot in `data/bronze/` and re-run the ingest for that
  season/week.
- **Friday automation: did it succeed?** Use the D-04 success triad in `AUTOMATION.md`
  Section 6: `log.status` in `{success, degraded}` AND
  `outputs/predictions/predictions_<S>_week<W>.csv` exists. Exit code is 0 for BOTH success
  and degraded; only `failed` exits 1. The log is `logs/friday_pipeline.json`
  (single-file overwrite -- it reflects the MOST RECENT run only).
- **A test run fails with `PRODUCTION STORE WRITE GUARD` while a scheduled task ran.** The
  closing wake task (`NFL_Predict_Closing`, 16 to 30 reading windows a month) writes
  `data/silver/odds_snapshot.parquet` and `logs/ledger_runs.jsonl`, and the 17:00 ET daily run
  writes `ledger/` and both run records under `logs/`. The suite's write guard cannot tell those
  bytes from a test's, so a session that overlaps one fails, naming whichever test module was
  running; a store caught mid-write can also stop the run with `CANNOT PROVE THE CONTENT`. When a
  run record grew during the failing module, the report starts with
  `CONCURRENT PRODUCTION WRITER LIKELY` and names the new records. Check the task's time, then
  re-run that module outside the task windows. The guard still fails in this case on purpose: it
  cannot separate the two writers, and passing would hide a real test write made in the same
  window. Do not loosen the guard to get past it.
- **Recovery is re-running the stage, not editing data by hand.** Every stage is
  re-runnable; Silver upserts are latest-wins and the gitignored outputs are regenerated.
  Recovering does NOT mean a re-fit or a gold rebuild -- those are the DESTRUCTIVE operations
  above and stay out of routine operation (D-07).

---

## Forward ledger

Added 2026-10-05 (Phase 34), after the old-rule addendum at the end of this file was written; it
is a set of operating procedures, not a result. Since the cutover that evening, every 2026 forward
bet the daily run decides is written to an append-only ledger instead of `outputs/bet_list/`. This
section covers where the record lives, how to restore it, how to keep its pushes working, how to
record a real bet and how to check it. The scheduled side -- the closing wake task, the Odds API
credit budget, the deploy keys and the run log -- is explained in `AUTOMATION.md`; the daily run's
ledger steps are in `PIPELINE.md`. Every command runs in PowerShell from the repository root.

### Where the record lives

- `ledger/forward_2026.jsonl` -- one line per entry: every forward row and every correction entry,
  chained by sha256 from a published genesis hash. A row is never edited. Grading, the closing line
  and a real fill each fill their own empty columns once, and a score corrected after grading is
  recorded as a new correction entry, never as an edit. Since Phase 37 it holds both arms: the
  live rows and the frozen arm's shadow twins, told apart by `arm` (`live` or `shadow`). Only the
  live arm is ever a bet or counted in the verdict.
- `ledger/snapshots/` -- the exact inputs each decision run scored (candidates, schedule, gold
  rows), so a row can be replayed later.
- `ledger/artifacts/` and `ledger/recipes/` -- the ledger's own copies of every model, blend and
  converter directory a row was scored with, and of the chain-fit record
  (`ledger/recipes/<sha256>.json`). Replay needs nothing from `artifacts/` or `outputs/`.
- `ledger/refit/` (Phase 37) -- the re-fit records:
  - `ledger/refit/dates.jsonl` -- one line per cadence-date event: `attempt`, `install_intent`
    and `closed`;
  - `ledger/refit/aa_weeks.jsonl` -- the A/A week results (did the frozen arm match the live arm
    for a whole week);
  - `ledger/refit/bet_rules/<sha256>.json` -- each re-fit generation's re-measured bet rule,
    written once and named by its own hash;
  - `ledger/refit/shadow_holes.jsonl` -- the frozen-arm holes (a live row written without a
    shadow twin, with the reason);
  - `ledger/refit/attempt.lock` -- held while a re-fit attempt runs; never backed up.

  Every record file is append-only (a bet-rule file is written once and never rewritten) and is
  carried by the private backup with the rest of `ledger/`. `PIPELINE.md`, daily run step 8, says
  when they are written.
- `ledger/` is itself a git repository (branch `main`). This public repository ignores it, and it
  is pushed to the PRIVATE repository `jackc625/nfl-predict-ledger`. Every push is a commit, so the
  backup keeps a dated history and a damaged local file can never overwrite the last good copy.
- The PUBLIC repository's branch `ledger-anchor` holds one two-line file, `ANCHOR`: the ledger's
  head hash and its row count. No game, pick, line or stake is ever published. A ruleset (id
  24536696) forbids deleting or force-pushing that branch, so the published history of heads
  cannot be rewritten.
- `logs/ledger_runs.jsonl` -- one line per ledger event: appends, refusals, settles, pushes and
  their failures (`AUTOMATION.md`, "Ledger run log").

Go-live was 2026-10-05 at about 17:47 ET. The ledger started with the 87 rows the old writer had
produced for weeks 3 and 4 (42 and 45), labelled `pre_verdict`. The first anchor is commit
`ec71c42` on `ledger-anchor`; the first backup push is commit `53b9d38` on the backup's `main`.
The 2025 replay rows stay in `outputs/bet_list/`. The 2026 verdict counts weeks 5-22 only, the live
arm only, on the corrected outcome; that scope is declared in `backtest/verdict_scope_2026.py`,
committed and pushed before week 5's first lock (Wednesday 2026-10-07, 18:00 ET).

Forward rows now come only from the daily run. `scripts/generate_bet_list.py` and the manual
Friday pipeline refuse to write forward rows (`ForwardRowsMovedToLedgerError`), so operation 12
above no longer produces the forward record -- the ledger is that record now.

### Restore the ledger from the private backup

Use this when `ledger/` is damaged or lost: verify fails on the chain or the anchor, the file is
gone, or the disk was replaced. Since 2026-10-12 the daily run also refuses to write when the ledger
is shorter than its anchor or the file is zero-length (for example after a power cut mid-write);
follow these steps then too, and never start a new ledger in its place. It rebuilds
`ledger/` from the private backup alone.

1. Stop the scheduled tasks, so nothing writes while you work. Use an ELEVATED terminal ("Run as
   administrator"): both tasks run with the highest privileges, and a normal terminal may answer
   "Access is denied".

   ```powershell
   schtasks /change /tn NFL_Predict_Pipeline /disable
   schtasks /change /tn NFL_Predict_Closing /disable
   ```

2. Move the damaged directory aside. Never delete it: it is the evidence of what went wrong, and
   you will compare it with the restored copy. Moving it outside the repository keeps it from being
   mistaken for project files.

   ```powershell
   Move-Item ledger "$HOME\ledger_damaged_$(Get-Date -Format yyyyMMdd_HHmm)"
   ```

3. Clone the backup into `ledger`, signed in as yourself (your normal GitHub sign-in; the deploy
   keys are for the scheduled pushes, not for this):

   ```powershell
   git clone --branch main https://github.com/jackc625/nfl-predict-ledger.git ledger
   ```

   If your GitHub account has an SSH key, the SSH form does the same:
   `git clone --branch main git@github.com:jackc625/nfl-predict-ledger.git ledger`. Always pass
   `--branch main`: the backup's history lives on `main`, and naming it means the clone cannot
   check out an empty default branch.

4. On a NEW machine only, the public repository's local `ledger-anchor` branch is missing too, and
   verify treats the local anchor as authoritative. If `git show-ref refs/heads/ledger-anchor`
   prints nothing, copy it from GitHub with `git fetch origin ledger-anchor:ledger-anchor`.

5. Verify. It must print `VERIFY_RESULT= PASS` (warning lines are allowed; a `FAILURE=` line is
   not):

   ```powershell
   uv run python -m scripts.verify_ledger
   ```

   Verify includes the D-19 settled-result check: every settled row is re-graded from the official
   scores in `data/silver/games.parquet`, so that file must exist. On a fresh machine, ingest the
   season's games first (`uv run python scripts/ingest_games.py --season 2026`), or point verify at
   a silver store that has them with `--silver-dir <dir>`. The re-grade also reads the chain-fit
   record. If `outputs/row19/neutral_hfa_chain_fit.json` is absent (it is gitignored, so a fresh
   checkout lacks it), pass the ledger's own copy:
   `--chain-fit-path ledger/recipes/<sha256>.json`.

6. Spot-check a replay of one week, 5 or later. It must end with `REPLAY_FAIL= 0`:

   ```powershell
   uv run python -m scripts.replay_ledger --week <n>
   ```

   Weeks 3 and 4 were migrated without stored inputs, so their rows print `not_replayable`. That
   is expected; it is reported and never counted as a pass.

7. Re-enable the tasks only after BOTH checks pass (elevated terminal again):

   ```powershell
   schtasks /change /tn NFL_Predict_Pipeline /enable
   schtasks /change /tn NFL_Predict_Closing /enable
   ```

If verify or replay fails on the restored copy, leave the tasks disabled. The backup and the anchor
disagree, and that has to be understood before anything is written again.

### If pushes start failing

Every daily run that changes the ledger syncs once: right after the day's ledger write and before
the re-fit side step on a day that reaches it (owner ruling 2026-10-21), otherwise at its end. It
commits a new anchor when the head moved and pushes it to `ledger-anchor`, then commits and
pushes the backup. A failed push never
fails the run: it is printed on the `LEDGER_SYNC=` line, recorded as an `anchor_push` or
`backup_push` event with `ok` false in `logs/ledger_runs.jsonl`, and retried by the next sync,
which carries the current head. A missing or locked backup repository is recorded the same way and
never fails the run either.

Verify shows a stuck push with two lines:

- `REMOTE_ANCHOR_BEHIND_BY= N` -- the GitHub anchor is N rows behind the local ledger.
- `BACKUP_BEHIND_BY= N` -- N backup commits are not on GitHub yet (`BACKUP_UNCOMMITTED= N`
  counts changed files not yet committed).

Both read 0 after a good sync. They are warnings, not failures; a number that keeps growing day
after day means GitHub is refusing the pushes.

The usual cause is a deploy key that has disappeared. The two keys --
`C:/Users/jackc/.ssh/nfl_ledger_anchor_ed25519` for the public repository and
`C:/Users/jackc/.ssh/nfl_ledger_backup_ed25519` for the private one -- were registered with `gh`,
and GitHub deletes keys added by `gh` when that `gh` login is revoked. Logging `gh` out removes
them. The key files on disk are untouched; only the registration is gone. To re-add them:

1. `gh auth status` -- if you are not logged in, `gh auth login`.
2. `uv run python -m scripts.setup_ledger_remote register-deploy-keys` -- adds each key,
   write-scoped, to its own repository (`gh repo deploy-key add <key>.pub -R <repo>
   --allow-write --title ...`), skips a key that is still registered, and reads both back.
3. `uv run python -m scripts.setup_ledger_remote verify` -- must print `SETUP_READBACK_MATCH= True`.
4. `uv run python -m scripts.sync_ledger` -- pushes whatever is waiting. Read `ANCHOR_PUSHED=`,
   `BACKUP_PUSHED=` and the two `_ERROR=` lines. `SYNC_RESULT= DONE` means it ran; a push that
   still failed is DONE too, with its error named. Then re-run verify: both `_BEHIND_BY=` lines
   should read 0.

If a key FILE is missing, `register-deploy-keys` refuses and names it. `create-keys` makes new keys
(it refuses to overwrite an existing file); then repeat steps 2-4. `sync_ledger` exits 1 only when
the ledger cannot be read, its chain is broken, or it is shorter than its anchor (`SYNC_REFUSED=`):
nothing is ever published from a broken or shortened ledger, so restore from the backup first.

### A re-fit record that will not read (a torn line)

The re-fit records under `ledger/refit/` are read strictly: a line that is not a complete, valid
record is never treated as "no records". What you see while one is bad:

- every daily run prints `REFIT= failed RefitRecordFormatError at ledger/refit/<file> line <n>`
  and sends a `FAILED: model re-fit crashed` push naming that record, that class and this
  repair, until the file is repaired (the path and the line number only, never the line's
  content);
- while the bad line is in `dates.jsonl` or `aa_weeks.jsonl`, the web cache build leaves out the
  Season page's comparison, which reads both (the page shows "Comparison not available"), and
  builds everything else as usual;
- once a re-fit generation is in force, a bad line in `dates.jsonl` other than a torn last line
  stops the whole web cache build, because the website's edge bands read that file too. What
  happens to the website then depends on the day. On a prediction day the run ends `degraded` and
  the website is not pushed (`rebuild_incomplete`). On a no-games or `lock_passed` day the run
  keeps its status: when a cache rebuild was attempted that day (the settle pass changed the
  ledger, or the re-fit wrote a record) the push is refused the same way, and when no rebuild was
  due the website is pushed with the previous cache;
- the day's live and shadow decisions are still made and written when the only bad line is a torn
  last line of `dates.jsonl` (it is left out of that read and logged); a bad line anywhere else in
  `dates.jsonl` stops them once a re-fit generation is in force.

The error names the file and the line: `refit record ledger/refit/<file> line <n> <problem>;
<advice>`. Repair it outside 16:30-18:15 ET, so no daily run is writing:

1. Copy the file aside, outside the repository, as the restore procedure moves a damaged ledger:

   ```powershell
   Copy-Item ledger\refit\<file name> "$HOME\<file name>_damaged_$(Get-Date -Format yyyyMMdd_HHmm)"
   ```

2. If the named line is the LAST line of the file and ends in no newline, the advice reads "a torn
   last line is a crash mid-append: copy the file aside, then delete that one line". Delete only
   that line, in a plain text editor, and save. Change nothing else. Nothing is ever appended onto
   a torn line (the append is refused by name), so that line holds only the torn bytes.
3. Confirm both files read with the same checks the daily run and the cache build apply (the
   field checks the writer applies, not only "is it JSON"). This must print two counts, not an
   error:

   ```powershell
   uv run python -c "from refit.records import read_date_events, read_aa_records; print('DATES=', len(read_date_events('ledger')), 'AA=', len(read_aa_records('ledger')))"
   ```

   The next daily run then prints its normal `REFIT=` line.

If the named line ends in a newline (any line but an unterminated last one), the advice says it "is
damage, not a torn append": a crash mid-append can only leave the last line unterminated. Change
nothing; compare the file with the private backup (the restore procedure above) first. A file under `bet_rules/` that no longer hashes to its name
raises the same error with its own message; it is written once and never edited, so treat it as
damage too and compare it with the backup.

### Recording a real bet

Paper rows carry empty fill columns. When you actually place a bet, record it on its row:

```powershell
uv run python -m scripts.record_fill --game-id <game_id> --season 2026 --week <n> --target ats --sportsbook draftkings --line -3.5 --odds -110 --stake-dollars 50 --filled-at 2026-10-18T12:01:00-04:00
```

- The row is named by its full key: `--game-id`, `--season`, `--week`, `--target` (`wp`, `ats` or
  `ou`) and `--arm` (default `live`).
- Give at least one of `--sportsbook`, `--line`, `--odds`, `--stake-dollars` and `--filled-at`.
  `--filled-at` must carry a UTC offset; a time without one is refused, never guessed.
- **Each value can be set once.** A second write to a column that already holds a value is refused
  (`FILL_REFUSED=`), even with the same value, and the ledger is left byte-identical. Check the
  numbers before you run it: there is no edit.
- Success prints `FILL_RECORDED=` and then the sync's lines. A fill changes the ledger directory,
  so the backup is pushed straight away (no anchor commit: the head does not move). A failed push
  is printed, not fatal; `uv run python -m scripts.sync_ledger` retries it.
- Exit codes: 0 recorded, 1 refused by the ledger, 2 bad input.

### Verify and replay

`uv run python -m scripts.verify_ledger` only reads; it never repairs anything. Exit 0 is PASS
(warnings allowed), 1 is any failure, 2 means the ledger file cannot be read. Its lines:

| Line | What it means |
|------|---------------|
| `LEDGER_ENTRIES=`, `HEAD_HASH=`, `CHAIN_OK=` | The entry count, the head hash, and whether the chain recomputes from the genesis hash. A broken chain adds `FIRST_BROKEN_SEQ=` and `FIRST_BROKEN_KEY=`, naming the first entry that fails. |
| `LOCAL_ANCHOR_ROWS=`, `LOCAL_ANCHOR_OK=`, `UNANCHORED_ENTRIES=` | The local `ledger-anchor` commit is authoritative. A missing anchor while rows exist, or an anchored head that is not the ledger's hash at that count, fails -- this is what catches a truncated file. Rows newer than the anchor are a warning; the next sync anchors them. |
| `REMOTE_ANCHOR_BEHIND_BY=` or `REMOTE_ANCHOR=` | The GitHub anchor, read anonymously: how many rows it is behind, or `unreachable` / `skipped`. Warnings. |
| `BACKUP_BEHIND_BY=`, `BACKUP_UNCOMMITTED=` (or `BACKUP=`) | Backup lag. Warnings. |
| `VERDICT_DECLARED=`, `VERDICT_START_WEEK=`, `VERDICT_STAMPS_OK=`, `VERDICT_LABELS_OK=`, `CORRECTIONS_OK=` | The rules of the committed verdict scope (W = 5): every live 2026 row carries the label its week is given; rows in weeks 5-22 carry every stamp, a registered recipe and `fill-v1`; every correction entry names a real row. |
| `REGRADE_CHECKED=`, `REGRADE_OK=` | D-19: every settled row (win, loss, push) is re-graded from the official scores and compared with its in-force outcome -- its latest correction entry if it has one, otherwise its own grade. |
| `REGRADE_MISMATCH=` | One line per settled row whose stored grade, payout or realized units disagree with the official score or with its in-force correction: `<seq> <key> <field> stored=<value> regraded=<value> <reason>`. The reason is `regrade_differs`, `correction_differs` (a correction is in force and the row still disagrees), `no_recorded_score` or `no_strategy_for_target`. Any mismatch fails. |
| `WARNING=`, `FAILURE=` | One line per warning or failure, in words. |
| `LOCAL_RESULT=` | PASS or FAIL over every check made on this machine: the chain, the local anchor, the verdict rules and the re-grade. |
| `EXTERNAL_RESULT=` | The GitHub anchor on its own: `VERIFIED`, `BEHIND`, `UNREACHABLE`, `SKIPPED` (with `--skip-remote`) or `DISAGREES`. Only `DISAGREES` fails: GitHub holds a head the local ledger does not have, which means the local file was rewritten or cut short. |
| `VERIFY_RESULT=` | The overall answer. |

A `REGRADE_MISMATCH=` line means a stored result no longer matches the public score. Never edit the
ledger by hand to make it go away. If the official score was corrected after grading, the daily
run's settle pass appends a correction entry once the revision detector flags that game, so re-run
verify after the next daily run. If the mismatch persists, or the score never changed, treat it as
damage and compare against the backup (the restore procedure above).

`uv run python -m scripts.replay_ledger` re-derives each row from its stored snapshot and the
ledger's own artifact copies, and compares every decision field: categories exactly, numbers within
1e-9. Narrow it with `--week`, `--season` or `--game-id` (repeatable). It prints one `REPLAY=` line
per row, then `REPLAY_PASS=`, `REPLAY_NOT_REPLAYABLE=` and `REPLAY_FAIL=`. A pre-verdict row with
no snapshot -- the 87 migrated rows -- is `not_replayable`; a verdict row with no snapshot fails.
Exit 0 means no row failed, 1 that a row failed, 2 that the ledger cannot be read or its chain is
broken (replay never runs on an unverified ledger).

### The one-time migration

Done once, on 2026-10-05, by Plan 34-19. Do NOT run it again. The order was: publish, verify, and
only then flip the switch.

1. `outputs/bet_list/` was copied to `outputs/bet_list_pre_migration/` (kept as a record).
2. `uv run python -m scripts.migrate_forward_rows --apply` moved the old writer's 87 forward rows
   into the empty ledger exactly as stored -- arm `live`, every stamp NULL, labelled
   `pre_verdict`, chained from the genesis hash -- and then rewrote `outputs/bet_list/` to hold only
   the 234 2025 replay rows.
3. `uv run python -m scripts.sync_ledger`, with the switch still off, pushed the first anchor
   (`ec71c42`) and the first backup (`53b9d38`); both were read back from GitHub.
4. `uv run python -m scripts.setup_ledger_remote create-anchor-ruleset` protected `ledger-anchor`
   (ruleset 24536696), and `setup_ledger_remote verify` printed `SETUP_READBACK_MATCH= True`.
5. `uv run python -m scripts.verify_ledger` printed `VERIFY_RESULT= PASS`.
6. Only then was the switch turned on -- `FORWARD_ROWS_GO_TO_LEDGER = True` in
   `forward_ledger/cutover.py`, in a commit of its own (`300e44e`) -- and the web cache rebuilt.
7. The owner installed `NFL_Predict_Closing` from an elevated terminal (`AUTOMATION.md`).

The dry run, `uv run python -m scripts.migrate_forward_rows` with no flag, only prints what
`--apply` would do. `--apply` now refuses (`MIGRATION_REFUSED= AlreadyMigratedError`), because the
ledger already holds entries. `--finish` exists only to complete an outputs rewrite interrupted
inside step 2. There is no way back: once the remote `ledger-anchor` exists, undoing the migration
is forbidden. The test suite runs with the switch OFF unless a test turns it on
(`tests/conftest.py`), so no test writes forward rows the way production does.

---

## Notifications

Added 2026-10-15 (Phase 35). The system reaches you in three ways: a phone push through ntfy (two
kinds only, failed and bets), one email every in-season Saturday through Resend, and an outside
watcher (healthchecks.io) that alerts you by push AND by email when the daily run has not checked in
by 18:30 ET. How they work is explained in `AUTOMATION.md` section 7; this section is what you do.
Every command runs in PowerShell from the repository root. This guide names the five `.env`
variables only: never paste a value into a chat, a commit, an issue or a log.

### One-time setup

1. **ntfy (the phone push).** Install the ntfy app on your phone. Generate the topic in your OWN
   terminal, so it never appears anywhere else:

   ```powershell
   uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   That prints 43 random characters; at least 32 are needed, because ntfy topics are public and
   anyone who knows the name can read and post to it. In the app, subscribe to that topic on
   `https://ntfy.sh` and set the subscription's display name to `NFL Predict`. The push titles
   carry no app name, so that display name is what tells you who sent them. The Android app and
   the ntfy web app have this setting; the iPhone app does not, so on an iPhone the pushes arrive
   under the ntfy app's own name.
2. **Resend (the weekly email).** Create a free Resend account with your own email address, then
   create an API key. The free tier sends from `onboarding@resend.dev` and delivers only to the
   account's own address, so the recipient must be the address you signed up with.
3. **healthchecks.io (the outside watcher).** In a free account:
   - create one check (any schedule: the setup command in step 5 sets it) and copy its ping URL,
     which must be the uuid form `https://hc-ping.com/<uuid>`;
   - create a READ-WRITE API key (Project Settings -> API Access); a read-only key can neither
     pause the check nor read it back;
   - create an ntfy integration (server `https://ntfy.sh`, the same topic) named exactly
     `NFL Predict ntfy`;
   - create an email integration to your own address named exactly `NFL Predict email`, and
     confirm it if healthchecks.io asks you to.

   The healthchecks.io API cannot create integrations, so these are by hand. The names matter:
   the setup command finds the two integrations by name and refuses if either is missing or
   duplicated.
4. **`.env`.** Put the five values in `.env` (never commit it; `.env.example` lists the names with
   empty values):

   | Env var | Value |
   |---------|-------|
   | `NTFY_TOPIC` | the topic from step 1 |
   | `RESEND_API_KEY` | the Resend API key |
   | `DIGEST_TO_ADDRESS` | the address the Resend account was created with |
   | `HEALTHCHECKS_PING_URL` | the check's `https://hc-ping.com/<uuid>` ping URL |
   | `HEALTHCHECKS_API_KEY` | the read-write API key |

   Delete the lines `ENABLE_EMAIL_REPORTS=...` and `ENABLE_SLACK_NOTIFICATIONS=...` if they are
   there. `Settings` no longer declares those two flags and refuses keys it does not declare, so a
   leftover line stops the daily run from loading its settings. Do not add `WATCHER_TEST_WITHHOLD_DATE` yet. Until a value is set, every send on its
   channel is recorded as a `missing_secret` failure in `logs/notifications.jsonl`; nothing
   crashes.
5. **Configure the watcher and read it back:**

   ```powershell
   uv run python -m scripts.setup_watcher
   ```

   It sets the check to cron `0 17 * * *` in `America/New_York`, grace 5400 seconds (late at
   17:00 ET, down at 18:30 ET), `manual_resume` false and both integrations attached, then reads
   it back and prints one `<field>= <want> | <got> | MATCH|MISMATCH` line each for `schedule`,
   `tz`, `grace`, `manual_resume` and `channels`. Expect `WATCHER_READBACK_MATCH= True` and exit
   0. Add `--verify-only` to read it back without changing anything. A
   `WATCHER_SETUP_REFUSED=` line names what to fix: a missing or duplicated integration name, a
   check that does not exist (HTTP 404 -- the command never creates one), a refused key (HTTP 401
   or 403: it must be the read-write key), or a ping URL that is not the uuid form (all exit 1).
   Exit 2 means `HEALTHCHECKS_PING_URL` or `HEALTHCHECKS_API_KEY` is not set.
6. **Send the TEST messages** (next subsection) and check that all three arrive.

### The TEST command

```powershell
uv run python -m scripts.notify_test
```

It sends three messages through the real sender, each labelled `TEST - ` so it can never be
mistaken for a real one:

- `TEST - FAILED: notification test` -- a top-priority push saying it is a test and no action is
  needed;
- `TEST - <n> new bets: notification test` -- a normal-priority push listing the live bets of the
  web cache's current slate (`TEST - 0 new bets` when the cache holds none, `TEST - bets unknown`
  when no cache could be read);
- the TEST weekly email -- the real email built from the current web cache, its subject starting
  `TEST - NFL Predict`.

It prints `NOTIFY_TEST_FAILED=`, `NOTIFY_TEST_BETS=` and `NOTIFY_TEST_DIGEST=`, each
`delivered <receipt id>` or `failed <kind>`. Exit 0 means all three were accepted; exit 1 means at
least one was not (`failed missing_secret`: that channel's `.env` value is unset;
`failed http_<status>`: the provider refused it). Then confirm on your phone and in your inbox that
each one arrived. The command never checks in with the watcher, and it leaves exactly three lines in
`logs/notifications.jsonl` with `source` `test_command`, priorities 5, 3 and null.

### The watcher test day (once)

This proves the watcher really alerts when the laptop goes quiet.

1. Pick an in-season ET date whose NEXT day has no game (a day the 17:00 ET run records
   `no_games`), and add this line to `.env`:

   ```text
   WATCHER_TEST_WITHHOLD_DATE=YYYY-MM-DD
   ```

2. That day's 17:00 ET run works as normal but withholds its check-in, printing
   `WATCHER_CHECKIN= withheld (test day)` and recording a `withheld` check-in in
   `logs/notifications.jsonl`. The check goes down at 18:30 ET: expect the watcher's alert on your
   phone AND by email by 18:45 ET.
3. Then DELETE the line from `.env`. It only ever acts on that one date, so a stale line is
   harmless, but it is a test switch, not a setting.

The run withholds only when every condition holds: the exact date, in season, zero games tomorrow.
If one fails it checks in normally and prints `WATCHER_TEST_DAY= not withheld: <reason>` (for
example `games tomorrow=13`); pick another date. The next day's normal check-in brings the check
back up.

### What each message means

Push titles are "what happened + the count", so the lock screen alone tells the story. Every failed
push carries `What failed:`, `Games:`, often `Error:` (one line, with any secret replaced by
`<redacted>` and any URL cut to its host), and `Next:`.

| Message | What it means | What to do |
|---------|---------------|------------|
| `FAILED: <n> games not predicted` (top priority) | The daily run stopped, started after the lock, or left games out; `Games:` names them. | Before the 6 PM ET lock, re-run `uv run python -m scripts.daily_lock_pipeline` (the `Next:` line says so, with the lock time). At or after the lock those games can no longer be predicted: read the `Error:` line and `logs/daily_lock_runs.jsonl` and fix the cause before tomorrow's run. |
| `FAILED: daily run, tomorrow's games unknown` (top priority) | The run failed and tomorrow's games could not be read either. | As above; check that the schedule refresh works (`AUTOMATION.md` section 1). |
| `FAILED: daily run degraded (<n> steps failed)` (top priority) | Predictions were made, but a non-critical step failed; `Steps:` names it. | Look at that step. If it was `populate_web_cache`, rebuild the cache by hand (operation 7). |
| `FAILED: ledger settle`, `FAILED: ledger closing finalization deferred` | Grading, corrections or closing-line finalization did not complete. Predictions are not affected. | Nothing to run now: the next daily run retries it. |
| `FAILED: closing line schedule`, `FAILED: closing line schedule read back wrong` | The closing-line wake task's triggers were not registered, or did not read back as expected. | Run `uv run python deployment/setup_scheduling.py --verify-installed`; if the task is missing, re-install it with `--install-closing` from an elevated terminal (`AUTOMATION.md`, "Closing-line wake task"). |
| `FAILED: ledger anchor push`, `FAILED: ledger backup push` | A ledger push to GitHub failed; it is retried by the next sync. | Run `uv run python -m scripts.sync_ledger`; if it keeps failing, follow "If pushes start failing" above. |
| `FAILED: ledger sync refused` | The end-of-run sync refused to publish or did not complete. | Run `uv run python -m scripts.verify_ledger`, then follow "Restore the ledger from the private backup" above. |
| `FAILED: closing line capture` | A closing-line reading failed. Those games' closing columns stay empty with reason `capture_failed`. | Nothing to re-run. If the push adds that the failure could not be written to `logs/ledger_runs.jsonl`, the weekly email will not list it. |
| `FAILED: weekly email` | The Saturday email was not delivered, or could not be built. The kind is in brackets: `http_<status>` (Resend refused it or was down for all three attempts), `transport_<ExceptionClass>` (a network failure), `no_receipt`, `missing_secret` (`RESEND_API_KEY` or `DIGEST_TO_ADDRESS` is unset), or `build_<ExceptionClass>` (the email could not be built). | The email is not re-sent automatically; this week's list is on `/bets`. `build_ValueError` usually means a malformed line in `logs/notifications.jsonl` (the email reads it for the `(update)` mark and the previous email's date, and a malformed log is never read as empty). Find the line with `uv run python -c "from notify.log import read_notifications; read_notifications()"` -- the error names the file and line number -- and fix or remove that one line. |
| `<n> new bets: <season> week <N>, <day>` | The run newly recorded live bets for that slate; one line each: matchup, bet type, pick, stake in units, EV band. A re-run lists only bets new to it. | Nothing. It is a record, not betting advice. |
| Email `NFL Predict - <season> week <N> - <n> bets - run clean` | The weekly summary. `RUN FAILED - ...` names what broke in its status header; ` (update)` marks a second email for the same day (a re-run); `bets unknown` and a stale notice mean the web cache was not current, never zero bets. | Read it. Its data-change section may name the acknowledge command (below). |
| The watcher's alert (from healthchecks.io, through `NFL Predict ntfy` and `NFL Predict email`) | The daily run did not check in by 18:30 ET -- the laptop was off or asleep at 17:00, the run died outright, or healthchecks.io could not be reached -- or it checked in as a failure. | Open the check's page on healthchecks.io. No ping that day: the run never reported; check the laptop and `logs/daily_lock_runs.jsonl`. A `/fail` ping: its body is the run summary. `failed pushes=` with fewer delivered than sent, `bets push=failed` or `digest=failed` means a message was lost: read its `failure_kind` in `logs/notifications.jsonl`. `unlogged receipts=` above 0 means a message reached you but its receipt could not be written to `logs/notifications.jsonl` (when the write raised, the run printed `NOTIFICATION_LOG= failed <ExceptionClass>`): check that `logs/` is writable and nothing is holding `logs/.notifications.lock`. |

Out of season the watcher check is paused by every daily run, so it sends nothing; the first run
from 7 days before the next season's first lock day resumes it. A message that was not delivered is
never re-sent by a later run.

### Reading logs/notifications.jsonl

`logs/notifications.jsonl` is the one place to read what reached you. It is append-only and holds
one JSON line per message (not per attempt):

| Field | Meaning |
|-------|---------|
| `run_id`, `run_date_et` | The daily run's id (null for the closing task, the TEST command and a run that started after the lock) and the ET date it was sent for. |
| `source` | `daily_run`, `closing_task` or `test_command`. |
| `event_class` | `failed`, `bets`, `test`, `digest`, `checkin` or `pause`. |
| `priority` | 5 (top) or 3 (normal) for a push; null for an email, check-in or pause. |
| `attempts` | Attempts made, at most 3 (0 for a withheld check-in or an email that could not be built). |
| `outcome` | `delivered`, `failed`, or `withheld` (the watcher test day). |
| `receipt_id` | The provider's message id when delivered. |
| `failure_kind` | `missing_secret`, `no_receipt`, `invalid_url`, `http_<status>`, `transport_<ExceptionClass>` or `build_<ExceptionClass>`. |
| `sent_at` | When the record was written (UTC). |

It never holds a topic, URL, key, recipient or message body. Read the latest lines with
`Get-Content logs/notifications.jsonl -Tail 20`.

### Acknowledging a sealed upstream change

The weekly email's `UPSTREAM DATA CHANGES AND CORRECTED GRADES` section lists every sealed-season
change the latest probe line still holds unacknowledged at WARNING or louder, for example
`schedules 2025: the upstream file changed after it was pinned (content digest moved)`. Runs are
unaffected -- they read pinned bytes -- but the email keeps listing the change until you rule on it.
Look at the change, then record your ruling:

```powershell
uv run python -m scripts.acknowledge_upstream_change --dataset schedules --season 2025 --reason "<why the change is accepted>"
```

- It copies the observed values from the latest probe line's finding for that pair (you never
  type a digest) and writes them onto the pair's entry in `config/upstream_pin.sealed.lock` with
  who ruled (`--ruled-by`, default `owner`), when (UTC) and your REQUIRED reason. The write is
  atomic: an interrupted run leaves the previous lock intact.
- It is NOT a re-freeze: `config/upstream_pin.json` is never opened, and every run keeps reading
  the pinned bytes. A second ruling on the same pair supersedes the first and keeps it on record. A
  later, different move is a new finding again.
- Success prints `ACK_RECORDED= <dataset> <season>` and `ACK_NEXT=`. Then COMMIT
  `config/upstream_pin.sealed.lock`: the acknowledgement is a committed owner ruling. From the next
  probe line on, the change reads as acknowledged and the email stops listing it.
- `ACK_REFUSED=` names why it refused: no finding for the pair in the latest probe line, a pair the
  lock does not track, or a lock or probe log that cannot be read (exit 1); a blank reason is exit
  2.
- Sealed seasons only. Live-season (2026) changes have no acknowledgement: each is listed in the
  email once, the week it is recorded, marked `fixed by ledger correction` when a stored correction
  belongs to it.

---

## Website (hosted dashboard)

Added 2026-10-18 (Phase 36). The dashboard is also published to Vercel, so you can read it on your
phone while the laptop is off. The laptop still does all the work: the hosted copy only serves the
web cache the laptop last pushed, and it never contacts the laptop. How the push fits into the daily
run is explained in `AUTOMATION.md` ("Website push"); this section is what you do. Every command
runs in PowerShell from the repository root. This guide names the two `.env` variables only: never
paste a value into a chat, a commit, an issue or a log.

### What it is

- The Vercel project `nfl-predict` in the team `jack-cutraras-projects` (the Pro plan you already
  have; no add-ons, no extra cost).
- Its address is the production URL named by `PRODUCTION_URL` in `pipeline/site_push.py` (the
  stable `*.vercel.app` alias; there is no custom domain).
- Every page sits behind Vercel's own sign-in (Vercel Authentication, scope All Deployments). A
  visitor who is not signed in to a Vercel account with access to the team sees Vercel's login page,
  never the dashboard. That includes `/health` and the `/static` files.
- It is the same app as the laptop's, serving the same pages, fragments, exports and `/health`. The
  hosted copy is refreshed after every daily 17:00 ET run that did not fail, game day or not (a
  daily heartbeat), and whenever you push by hand.

### One-time setup

1. **The dedicated token.** Go to https://vercel.com/account/settings/tokens -> Create. Scope it to
   the `jack-cutraras-projects` team, set the expiry to about one year, and name it
   `nfl-predict website push`. Put it in `.env` on one line `VERCEL_TOKEN=` followed by the value,
   and nowhere else. The push hands it to the Vercel CLI only through the CLI's environment, never
   on a command line. Your own `vercel login` session is never used. Accepted (D-01): Vercel cannot
   limit a token to one project, so this token can also reach the team's other projects.
2. **The protection bypass.** In Vercel: project `nfl-predict` -> Settings -> Deployment Protection
   -> Protection Bypass for Automation -> add a secret. Put it in `.env` on one line
   `VERCEL_BYPASS_SECRET=` followed by the value. It is how the laptop reads the protected `/health`
   back after a push (D-15): it is sent only in the `x-vercel-protection-bypass` request header,
   never in a URL.
3. **Write down the token's expiry date here** (YYYY-MM-DD):

   Token expires: 2027-10-19

4. **Which CLI the push runs.** The push runs the Vercel CLI installed globally under nvm Node
   24.18.0, by absolute path: `NODE_EXE` and `VERCEL_CLI_JS` in `pipeline/site_push.py` (D-16).
   The scheduled task has no user PATH, so a bare `vercel` would not be found. If that Node version
   is ever uninstalled, install the CLI under the Node you keep and update those two constants;
   until then every push fails with `cli_missing`.
5. **The switch.** Unless `config/site_push_enabled.json` exists and reads `"enabled": true`, every
   push -- daily or by hand -- only records `not_pushed` / `push_not_enabled` in the push log and
   sends nothing to Vercel. Committing that file is the LAST setup step, after the protection has
   been proven and after your go-ahead.

### Never change Deployment Protection

Vercel Authentication with scope All Deployments was proven once, at setup, by recorded live
refusals: the production URL, a per-deployment URL, `/health` and a `/static` file all refused a
request with no credential, and a wrong bypass value was refused too. Nothing re-checks it after
that. No push reads the setting back. So the Deployment Protection setting must never be changed:
never switch it to "production stays public" or any narrower scope, never create a share link, and
never make `/health` public to ease a check (D-02). If it is ever changed by mistake, put it back,
then run the refusal check in "Checking the live site from the laptop" below.

### Opening it on the phone

Open the production URL in the phone's browser. The first time, Vercel shows its login page: sign
in with the Vercel account that owns the team. Vercel then keeps the phone signed in, so later
visits open the dashboard directly. If the login page comes back, sign in again. It works the same
over cellular as on home Wi-Fi, and it does not need the laptop to be on.

### Push now by hand

```powershell
uv run python -m scripts.push_site
```

- **When:** after a `FAILED: website push` phone push, once you have fixed what it named (for
  example a renewed token, or a recompiled stylesheet); or whenever you want the phone copy
  refreshed without re-running the day.
- It goes through the daily push's own routine: the same switch, the same refusals, the same fresh
  publish id, the same read-back proof and the same push-log record, with source `manual`.
- It prints the routine's `SITE_PUSH=` line, then `PUSH_SITE= <outcome> <publish id | reason>`.
  Exit 0 means the push succeeded and was read back from the live site; exit 1 means anything else.
- It refuses at once (`lock_held`) while another push or a cache rebuild is running. Never run it
  between 16:30 and 18:15 ET: the 17:00 run's cache rebuild and its push hold the lock then, and a
  long manual push can make that rebuild time out (the run is then degraded and its own push
  refuses).

### Reading logs/site_pushes.jsonl

`logs/site_pushes.jsonl` is append-only and holds one JSON line per push attempt, daily or manual,
including attempts that did not push:

| Field | Meaning |
|-------|---------|
| `time_utc` | When the attempt finished (UTC). |
| `publish_id` | The fresh id this attempt minted and stamped into the cache (null if it stopped before that). |
| `run_id` | The daily run's id; null for a manual push. |
| `source` | `daily` or `manual`. |
| `outcome` | `succeeded`, `failed`, `refused` or `not_pushed`. |
| `reason` | Why it did not succeed (table below); null on success. |
| `deployment_id`, `deployment_url` | The Vercel deployment the CLI reported, and its own per-deployment URL. |
| `local_sha256` | The SHA-256 of the cache file as it was uploaded. |
| `readback` | What the last read of the live `/health` saw: `matched`, `old_id`, `digest_mismatch`, `no_id`, `http_<status>` or `transport_<ExceptionClass>`. On a `pages_failed` push, the first page that failed instead: `page_<route>_http_<status>`, `page_<route>_no_marker` or `page_<route>_transport_<ExceptionClass>`, where `<route>` is the path with `/` turned into `_` (`root` for the home page). |
| `data_format_found`, `data_format_expected` | The cache's data-format version and the one the code expects, once read. |
| `code_commit` | The commit the deployed code came from (null if the attempt stopped before deploying, or git could not say). The push uploads the files on disk, not a commit, so it ends in `+dirty` when `api/`, `web/`, `utils/`, `conf/` or an uploaded root file had uncommitted changes: the hosted site then runs code that is in no commit. Commit or discard the change, then push by hand. |

A push is `succeeded` only when the live `/health` returned this attempt's publish id AND the
SHA-256 of the file it serves equals `local_sha256`, AND each page it then fetches once -- the five
nav pages, `/fragments/games` and `/api/export/json` -- answers as the laptop does (the same check
as "Checking the live site from the laptop" below). The CLI's exit code is never the evidence. The
log never holds a token, a bypass value or a URL secret. Read the latest lines with
`Get-Content logs/site_pushes.jsonl -Tail 20`.

| Reason | What it means | What to do |
|--------|---------------|------------|
| `push_not_enabled` (`not_pushed`) | The switch file is absent or does not read `"enabled": true` (setup not finished, or paused). | Nothing, unless you meant to push: see "Pausing every push". |
| `failed`, `run_failed` (`not_pushed`) | The daily run itself failed, so it did not push. | Fix the run (its own `FAILED:` push says how); the next good run pushes. |
| `lock_held` (`refused`) | Another push or a cache rebuild held the lock. | Wait for it to finish, then push by hand. |
| `rebuild_incomplete`, `cache_missing` (`refused`) | This run's cache rebuild did not finish, or `data/web_cache.duckdb` is missing. | Rebuild the cache (operation 7), then push by hand. |
| `stylesheet_out_of_step` (`refused`) | A fresh compile of the stylesheet differs from the committed `web/static/css/tailwind-compiled.css`. | Run the recompile in operation 8, commit the sheet, then push by hand. |
| `stylesheet_unverifiable` (`refused`) | The stylesheet could not be compiled to check it, usually because `tools/tailwindcss.exe` is missing. | Restore `tools/tailwindcss.exe`, then push by hand. |
| `format_mismatch_rebuild_failed`, `format_mismatch_after_rebuild` (`refused`) | The cache's data-format version did not match the code; the push rebuilt the cache once and the rebuild failed or still did not match. | Rebuild the cache (operation 7) and read its errors, then push by hand. |
| `missing_token`, `missing_bypass_secret` (`failed`) | That `.env` value is not set. | Add it (One-time setup), then push by hand. |
| `deploy_exit_<n>` (`failed`) | The Vercel CLI refused the deploy. Most often the token has expired or been revoked. | Check the token; renew it if it expired (below), then push by hand. |
| `cli_missing`, `deploy_timeout`, `deploy_output_unparseable` (`failed`) | The CLI was not where `NODE_EXE` says, ran longer than 15 minutes, or printed something the push could not read. | `cli_missing`: see One-time setup step 4. Otherwise push by hand again. |
| `readback_failed` (`failed`) | The deploy ran, but the live `/health` never showed this attempt's publish id and digest within about two minutes. | Push by hand again, then check `/health` (below). |
| `pages_failed` (`failed`) | `/health` read back, but a page on the live site failed; `readback` names the first one (for example `page_bets_http_500`). The phone copy is broken even though it carries the new data. | Run the live check (below) to see every route, fix the code, then push by hand. |
| any other name, such as `OSError` | An unexpected error inside the push, named by its exception class only. | Push by hand again. If the same class repeats, it names what broke. |

### What "FAILED: website push" means

A failed or refused push sends one normal-priority `FAILED: website push` phone push naming the
outcome and the reason (both data-format versions on a format refusal), and changes nothing else:
the run's exit code, its ledger rows and its outputs are the same as if the push had worked. After
a failed push:

- the phone copy still shows the last good push, and grows the "Out of date" bar once that push is
  more than 36 hours old;
- the laptop's own copy already carries the failed attempt's publish stamp (the stamp is written
  before the upload), so the laptop shows NO bar even though the phone copy is stale (D-14). Do not
  read the laptop's pages as proof the phone is current.

Fix what the reason names (table above), then push by hand.

### Pausing every push

To stop every push, daily and manual, change `"enabled": true` to `"enabled": false` in
`config/site_push_enabled.json` and commit it:

```powershell
git add config/site_push_enabled.json
git commit -m "pause the website push"
```

Every run then records `not_pushed` / `push_not_enabled` and sends nothing. Only exactly `true`
pushes: `false`, a missing file, a string such as `"true"` or a file that is not valid JSON all
pause it. The push reads the file on disk, so the pause takes effect at the next run even before
the commit. To resume, set it back to `"enabled": true`, commit, and push by hand once to confirm
it reads back.

### Renewing the token (yearly)

Before the date on the `Token expires:` line above:

1. Create a new token exactly as in One-time setup step 1.
2. Replace the `VERCEL_TOKEN=` line in `.env` with the new value.
3. Push by hand (`uv run python -m scripts.push_site`) and confirm `PUSH_SITE= succeeded`.
4. Update the `Token expires:` line in this file and commit it.
5. Delete the old token at https://vercel.com/account/settings/tokens.

A token that expires first shows up as a `FAILED: website push` with a `deploy_exit_<n>` reason.

### Old copies of the site

Vercel keeps production deployments for 30 days, plus always the latest 20 and the one serving the
production URL (D-10). Older ones can no longer be opened. The push log still keeps every
attempt's publish id, deployment id, per-deployment URL and digest for the whole season, so each
push stays on record after its deployment is gone.

### Checking the live site from the laptop

```powershell
uv run python -m scripts.fetch_hosted_routes
```

It fetches every route of a fixed list -- the five nav pages, a game page, an unknown game, the
four retired redirects, the four fragments, both exports, `/health`, the stylesheet and a font --
from the production URL, sending the bypass value only in its header. A route passes only when it
answers exactly as the laptop does (same status, same redirect target) and carries its marker: a
page's title, a redirect's target, the right content type, and for `/health` the publish id and
digest of the newest succeeded push in the push log. So a Vercel login page that answers 200 fails.
It prints one `ROUTE= <path> status=<code> marker=<yes|no>` line per route and ends with
`FETCH_HOSTED= pass <n>` (exit 0) or `FETCH_HOSTED= fail <k> of <n>` (exit 1). Options:
`--game-id` picks the game page (default: the first in the local cache), `--publish-id` names the
push `/health` must show, and `--record FILE` also writes the lines to a file. Do not run it within
two minutes of 18:00 ET: a game locking mid-run changes a page between two requests.

To check that strangers are still refused (after a protection scare, for example):

```powershell
uv run python -m scripts.fetch_hosted_routes --refusals --deployment-url <a per-deployment URL from the push log>
```

It sends no credential to the production URL, that deployment URL, `/health` and the stylesheet,
and prints one `REFUSAL= <url> status=<code> content=<none|LEAKED>` line each, with the status line
and a few safe headers (never a cookie). Every line must read `content=none` with status 401, 403
or a redirect to vercel.com, ending `FETCH_HOSTED= pass 4`. Add `--wrong-bypass` to send a random
wrong bypass value instead of none; it must be refused the same way.

### The "Out of date" bar and the data-format notice

- **The orange "Out of date" bar** sits above the menu on every page, on the phone copy and on the
  laptop, when the newest push is more than 36 hours old: "Out of date: last update <weekday, date,
  time ET>. The laptop has not reported since." With no push on record it reads "Out of date: no
  push recorded." The pages stay readable. It means the laptop has not pushed for a day and a half:
  check that it was on or asleep (not shut down) at 17:00 ET, read `logs/site_pushes.jsonl`, and
  push by hand. **Expected right now:** from the day the bar shipped until the first real push, the
  laptop's own pages read "Out of date: no push recorded." -- the laptop's cache has never been
  pushed, and the bar follows the same rule on both copies. It goes away after the first push.
  A cache rebuild (the 17:00 run's, or operation 7 by hand) keeps the last push's publish id and
  time, so after a rebuild the laptop's bar and `/health` still describe the last push, not "no
  push recorded".
- **The data-format notice** is a plain page answered with HTTP 503 on every page, fragment and
  export: "This copy of the site can't show its data yet. The pages expect data format <n>; the data
  file is format <m>." It means the code and the cache were built for different versions. Locally,
  rebuild the cache (operation 7); the server picks it up on the next request with no restart. On
  the phone copy, the next push fixes it (it rebuilds a mismatched cache before deploying); push by
  hand to fix it sooner. `/health` names both versions.

---

## Season close (February 2027): the binding readout

Added 2026-10-21 (Phase 38), after the old-rule addendum at the end of this file was written; it is
an operating procedure, not a result. **Verification basis:** verified on fixtures by
`tests/unit/test_season_readout_binding.py`; the real binding run happens once, in February 2027.

### What it is

The one binding run of the 2026 money verdict. How the verdict is judged -- the bets that count,
the test, the significance level, the correction over the three targets and the verdict words -- was
frozen before the rest of the season's results existed, in `SEASON-VERDICT-PREREGISTRATION.md` and
`backtest/verdict_test_2026.py`. The binding run applies that frozen rule to the whole graded
season and writes one document, `docs/records/SEASON-2026-READOUT.md`. It runs once. There is no
second attempt.

### When to run it

After the Super Bowl (Sunday 2027-02-14) has been graded by the Monday 2027-02-15 17:00 ET daily
run. Run it on **Tuesday 2027-02-16**.

On any day before that, use the preview instead. It shows the season's raw figures only (counts,
units and flat ROI, labelled "season to date -- not the verdict"), never a p-value or a verdict
word, and it writes nothing:

```powershell
uv run python -m scripts.season_readout
```

### The command

From the repository root, in PowerShell:

```powershell
uv run python -m scripts.season_readout --binding
```

It takes no output-path option: the document always lands at `docs/records/SEASON-2026-READOUT.md`
under the repository root. On success it prints `BINDING_WRITTEN= <path>` and one
`VERDICT_<TARGET>= <word>` line per target, and exits 0.

### If it refuses

Before computing or writing anything it checks nine conditions, numbered 0 to 8, in the order
below, and stops at the first one that is not met. It then prints `BINDING_REFUSED= <sentence>`
naming that condition and exits 3. Nothing is written. What to do depends on which one it names.

**Restore the ledger first.**

0. The ledger holds no entry, the repository has no `ledger-anchor` commit, or the ledger does
   not extend that anchored head. An absent, emptied or stale-backup ledger still passes the chain
   check, so this is what stops the one run being spent on it. Never run the binding run while
   `ledger/` is moved aside (step 2 of "Restore the ledger from the private backup" above): finish
   the restore, confirm `uv run python -m scripts.verify_ledger` passes, then run it again.

**Wait for the next 17:00 ET daily run, then try again.** These clear on their own once the daily
run has graded the bets and recorded the scores:

1. A counted bet of a played game is still ungraded.
2. An in-scope game lacks a final score.
3. No played week-22 game is in the schedule yet (the Super Bowl result has not reached it).

**Already done -- do not run it again.**

4. The record already exists, or its path appears anywhere in git history. The binding run has
   already happened. Never delete or move the document to run it again: the history check refuses
   that too, and a second run would be a second look at the same season.

**Repository integrity -- stop and investigate, never work around it.** These mean the rule or the
repository is not what was frozen. Do not edit, re-commit or re-clone anything to make one go away;
find out what happened first and take it to the owner:

5. The R6 lock commit (the commit that added `backtest/verdict_test_2026.py`) cannot be found or
   proven: the clone is shallow, or the commit history names is not the witnessed lock `922b010`
   (committed before the 2026-10-28 deadline, touching exactly the two rule files), for example
   after rewritten or squashed history. Run it from a full clone of the real repository.
6. A commit after the lock edited `SEASON-VERDICT-PREREGISTRATION.md` or
   `backtest/verdict_test_2026.py` without naming the lock as INVALIDATED, or one of those two
   files has an uncommitted change.
7. A frozen value (alpha 0.05, bootstrap B 2000, seed 2704, family size 3) no longer equals its
   pin.

   The same applies to any refusal that names a failed git or path check ("a git check failed, so
   the run refuses"): the run fails closed rather than guess.

**Fix the named store first, then try again.**

8. The re-fit comparison (`scripts.compare_refit_arms`) could not run. The refusal quotes its exit
   code and its first output line; fix what that names.

   Exit 2 (not 3) means a store could not be read at all: `LEDGER_UNREADABLE=` (the ledger, or its
   chain does not verify -- see "Restore the ledger from the private backup" above) or
   `STORE_UNREADABLE=` (the silver schedule, the run log or the skip records). Fix the named store
   and run it again.

### Reading the result

Each target gets one verdict word, printed with its 2026 meaning (the meanings are fixed in
`backtest/verdict_test_2026.py`, `VERDICT_TOKEN_MEANINGS_2026`):

- `PROFITABLE_CLEAN` -- at least one counted bet, flat ROI over the season strictly above 0, and
  the one-sided ROI p-value after the correction over the three targets strictly below alpha.
- `UNPROFITABLE_CLEAN` -- at least one counted bet and a flat ROI of 0 or less. A measured result,
  not a failure to measure.
- `INCONCLUSIVE_CLEAN` -- a positive flat ROI that cannot be told apart from zero. Never called
  profitable.
- `UNDISCHARGEABLE_NO_BETS` -- the target was priced but none of its counted rows is a bet.
- `UNDISCHARGEABLE_NO_CHAIN` -- the target was never priced inside the verdict scope.

The document also reports coverage: per week, the games scheduled, decided, bet and missed, and the
season's total of missed games. Closing-line value is not part of the verdict and is not in the
document.

### Committing it

The new record must be registered in the same commit that adds it, or the readout-label test fails
on the next run:

1. Add `SEASON-2026-READOUT.md` to `NO_PREFIX_NUMBERS_REASONS` in
   `tests/unit/test_old_rule_labels.py`, with a one-sentence reason (a 2026 forward result under the
   day-before lock, so the old-rule addendum does not describe it).
2. Add `SEASON-2026-READOUT.md` to `RECORDS` in `tests/doc_locations.py`.
3. Run the label tests:

   ```powershell
   uv run python -m pytest tests/unit/test_old_rule_labels.py tests/unit/test_phase33_no_zero_failures_claim.py -q -p no:cacheprovider
   ```

4. Stage the three files by name (`docs/records/SEASON-2026-READOUT.md`,
   `tests/unit/test_old_rule_labels.py`, `tests/doc_locations.py`), commit, and push the normal way.

---

## Cross-references

- **`PIPELINE.md`** -- the canonical 8-stage run sequence (ingest -> features -> train ->
  promote -> backtest -> predict -> build-cache -> serve). Owns the per-stage commands this
  runbook mirrors; it is the command source of truth.
- **`AUTOMATION.md`** -- the Friday automation explanation: what fires and when, the 18
  orchestrator steps, where logs land, and the D-04 "did Friday succeed?" success triad. The
  "Run automation" operation links it instead of re-explaining it. Its section 7 explains the
  notification paths (Phase 35) that the "Notifications" section above operates.
- **`README.md`** -- the portfolio-facing system narrative and the single architecture
  diagram the DOC-02 section above cross-references (no duplicate diagram lives here).
- **`MODEL-DIAGNOSIS.md`** -- the frozen v2.1 honest per-target accuracy diagnosis (WP/ATS =
  CEILING, O/U = MIXED) and the production-vs-backtest population distinction. Consult it
  before reasoning about model quality. It is the point-in-time v2.1 diagnosis; the
  Phase-25 activation that followed is recorded in `ACTIVATION-READOUT.md` (below).
- **`ACTIVATION-READOUT.md`** -- the Phase 25 activation record: the gated re-fit that
  swapped WP + ATS onto the canonical Elo gold while retaining v1.0 O/U, with per-target CLV
  before/after, the deployed/retained 2x2, and the pre/post manifest state (the Rollback
  operation's source of the pre-swap mapping + sha256).
- **`GATED-REFIT-READOUT.md`** -- the Phase 30 gated re-fit record: the four attributed
  rebuild rungs, the frozen feature-group selection rule and its three verdicts (injury
  DROPPED, snap and situational KEPT), the per-target deploy outcome (WP promoted, ATS and
  O/U REFUSED and their incumbents retained), the two gate-baseline re-freezes, the blend
  re-check that changed nothing in production, and the registers and quarantines the phase
  deliberately left open. The Rollback operation's source of the Phase-30 pre-swap mapping.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit: the adopted
  canonical gold, the FIX-01 cluster, the AUDIT-01 stage-runner evidence cited by the
  DESTRUCTIVE-command labels above, and the deferred findings.
- **`PROFITABILITY-READOUT.md`** -- the Phase 31 milestone close: the per-target 2025
  clean-split profitability verdict, what is deployed and what was retained, the absolute
  closing-line value per target, and every disclosure the owner accepted. It is also where
  the five deliberately-red tests are explained, so an operator who runs the suite and sees
  them can tell an expected red from a regression.
- **`STATE-OF-SYSTEM.md`** -- the consolidated state-of-the-system registry: what is
  trustworthy now, what was fixed this milestone, and the single list of deferred items.
- **`SEASON-VERDICT-PREREGISTRATION.md`** -- the frozen rule for the 2026 money verdict, which
  the "Season close (February 2027)" section above runs once after the season is graded.
- **`deployment/README.md`** -- the Windows Task Scheduler setup for the Friday automation
  (registering / re-pointing the `NFL_Predict_Pipeline` task). The "Run automation" operation
  links it for scheduling.

---

*Phase 23 -- Documentation, Runbook & State-of-System. DOC-01 operator runbook + DOC-02
Architecture & Data-Flow section. Link-don't-duplicate; every command verification-basis-
labeled. ASCII only (no emoji, per CLAUDE.md).*

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

What this covers in this document: the model verdicts it summarises in its cross-references (the diagnosis ratings and the promotion and retention outcomes) are old-rule; the operating procedures themselves are not results and are not covered by this label.
