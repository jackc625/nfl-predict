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
- **Recompile the stylesheet after ANY template edit that introduces a new utility class.**
  There is no `make` target for this and no stage of its own; nothing in the build notices
  when the compiled sheet is stale, so a template can use a class that was never compiled and
  the page renders unstyled with no error anywhere:

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
  recorded as a new correction entry, never as an edit.
- `ledger/snapshots/` -- the exact inputs each decision run scored (candidates, schedule, gold
  rows), so a row can be replayed later.
- `ledger/artifacts/` and `ledger/recipes/` -- the ledger's own copies of every model, blend and
  converter directory a row was scored with, and of the chain-fit record
  (`ledger/recipes/<sha256>.json`). Replay needs nothing from `artifacts/` or `outputs/`.
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

Use this when `ledger/` is damaged or lost: verify fails on the chain, the file is gone, or the
disk was replaced. It rebuilds `ledger/` from the private backup alone.

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

Every daily run that changes the ledger ends with a sync. It commits a new anchor when the head
moved and pushes it to `ledger-anchor`, then commits and pushes the backup. A failed push never
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
the ledger cannot be read or its chain is broken (`SYNC_REFUSED=`): nothing is ever published from
a broken chain, so restore from the backup first.

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

## Cross-references

- **`PIPELINE.md`** -- the canonical 8-stage run sequence (ingest -> features -> train ->
  promote -> backtest -> predict -> build-cache -> serve). Owns the per-stage commands this
  runbook mirrors; it is the command source of truth.
- **`AUTOMATION.md`** -- the Friday automation explanation: what fires and when, the 18
  orchestrator steps, where logs land, and the D-04 "did Friday succeed?" success triad. The
  "Run automation" operation links it instead of re-explaining it.
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
