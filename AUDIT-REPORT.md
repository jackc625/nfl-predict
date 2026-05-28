# AUDIT-REPORT.md -- Data & Feature Correctness Audit (Phase 20)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 20 -- Data & Feature Correctness Audit
**As-found baseline seeded:** 2026-05-28 (Wave 2 / plan 20-02)
**Finalized:** _pending Wave 3 / plan 20-06_

> This is the forensic deliverable for Phase 20 (D-12). It records the system's true
> **as-found** state -- a per-stage pass/fail baseline captured **before any Wave-3 fix**
> (D-10 catalog-then-batch) -- plus the cataloged findings, the documented 2025 currency
> gap, the resolved weather-provider doc drift, the deferred non-correctness findings, and
> a placeholder for the single end-of-phase gold rebuild's metric moves. Stages that error
> are recorded honestly as FAIL findings here; they are **not** fixed in this wave.
>
> Status tags are ASCII `[PASS]` / `[FAIL]` / `[PARTIAL]` (no emoji, per CLAUDE.md).
>
> Downstream waves 20-03 / 20-04 / 20-05 emit their detailed findings to
> `outputs/diagnostics/` files; 20-06 folds those plus the gold-rebuild metric moves into
> this report's final form. Phase 23's runbook + state-of-system summary cites this report.

---

## Scope of the as-found baseline (Wave 2)

The Wave-2 baseline (this section) was produced by the re-runnable harness
`tests/integration/test_audit_stage_runner.py` (D-12 deliverable a). Per D-06 it exercises
BOTH stage paths:

- **(a) Historical / full path** -- each of PIPELINE.md's 7 canonical stages has an
  importable live entry point whose `--help` / `main` is callable exit-0. This proves the
  stage entry points run. The **single full historical gold rebuild is Wave 3's job**
  (D-10); this wave does NOT rebuild gold.
- **(b) Current-week dry-run** -- the incremental code paths (`--current` / `--season
  --week`) run end-to-end against the most-recent fully-completed week (the stand-in
  "current week" = **season 2024, week 18**, the latest fully-settled week with
  definitely-known answers, 16 games on disk). This proves the incremental paths -- where
  the `cb61042` silent break and the CR-01 abort lived -- run without a live game. No
  re-training (D-01); the predict path LOADS artifacts.

Wave-1 (plan 20-01) already landed the CR-01 cluster blocker fix inline (the only inline
fix permitted by D-10): `step_validate_models`' `critical=True` gate now passes on
correctly-trained artifacts, which is what unblocks the predict stage from running
end-to-end. Everything else surfaced here is **cataloged, not fixed**.

---

## Per-stage pass/fail (as-found)

### (a) Historical / full path -- PIPELINE.md 7 canonical stages

Evidence command pattern: `uv run python scripts/<entry>.py --help` (exit 0), driven by
`TestHistoricalStageEntryPoints` in the harness. Stage 7 (serve) is verified by importing
`api.main:app`. "Runs end-to-end" here = entry point is importable and its CLI is callable
without an import/argparse error; the full historical rebuild that materializes new gold is
Wave 3's single batch run (D-10), recorded under "Gold rebuild + metric moves" below.

| # | PIPELINE.md stage | Entry point(s) | Evidence | Status |
|---|-------------------|----------------|----------|--------|
| 1 | Ingest (games / odds / weather) | `scripts/ingest_games.py`, `ingest_odds.py`, `ingest_weather.py` | `--help` exit 0 (x3) | `[PASS]` |
| 2a | Features -- Elo | `scripts/build_elo.py` | `--help` exit 0 | `[PASS]` |
| 2b | Features -- team form | `scripts/build_team_form.py` | `--help` exit 0 | `[PASS]` |
| 2c | Features -- contextual | `scripts/build_contextual.py` | `--help` exit 0 | `[PASS]` |
| 2d | Features -- weather | `scripts/build_weather.py` | `--help` exit 0 | `[PASS]` |
| 2e | Features -- market anchors | `scripts/build_market_anchors.py` | `--help` exit 0 | `[PASS]` |
| 2f | Features -- Silver->Gold assembly | `scripts/build_features.py` | `--help` exit 0 | `[PASS]` |
| 3 | Train (from gold) | `scripts/train_models.py` | `--help` exit 0 (NOT executed -- D-01 no re-train) | `[PASS]` |
| 4 | Backtest | `scripts/run_backtest.py` | `--help` exit 0 | `[PASS]` |
| 5 | Predict | `scripts/generate_current_week_predictions.py` | `--help` exit 0 + dry-run below | `[PASS]` |
| 6 | Build cache | `scripts/populate_cache.py` | `--help` exit 0 | `[PASS]` |
| 7 | Serve | `api/main.py:app` | `from api.main import app` importable; `app.routes` present | `[PASS]` |

### (b) Current-week incremental dry-run paths (D-06)

| Path | Entry / call | Evidence | Result observed | Status |
|------|--------------|----------|-----------------|--------|
| Predict (stand-in current week) | `generate_and_write(season=2024, week=18, output_dir=<tmp>, artifacts_dir="artifacts")` | harness `test_generate_and_write_produces_nonempty_predictions` | 16 games; `predictions_2024_week18.csv` written, 16 rows, non-empty (`wp_prob`/`ats_prediction`/`ou_prediction` present); `game_context_2024_week18.csv` written; all 3 model artifacts LOADED (no train); market blend applied (16 blended) | `[PASS]` |
| Predict -- no-train guard | `run_predictions("artifacts", 2024, 18)` with `BaseTrainer.train_and_evaluate` patched to raise | harness `test_predict_dry_run_does_not_train` | All 3 targets returned predictions; no trainer invoked (D-01 honored) | `[PASS]` |
| Elo incremental | `scripts/build_elo.py --current --no-save` | harness `test_build_elo_current_dry_run_exits_zero` | exit 0; processed current season (2025) chronologically, 78 games, ratings validated; `--no-save` => no silver mutation (re-runnable) | `[PASS]` |
| Team-form incremental | `TeamFormBuilder().build_for_current_week()` (the `build_team_form.py --current` code path; silver write patched out for idempotency) | harness `test_build_team_form_current_path_runs` | Ran end-to-end; returned 64 rolling team-form rows for 32 teams; attempted `team_game_stats` silver write (intercepted) | `[PASS]` (with findings F-01, F-02 below) |

**Baseline verdict:** AUDIT-01 holds as-found -- every stage's live entry point runs, and
both the predict and the two incremental builder paths run end-to-end against the stand-in
current week without a live game and without re-training. Two **non-blocking** findings on
the team-form `--current` path are cataloged below (they did not stop the path from
running, so they are batch-fix candidates per D-10, not Wave-1 inline blockers).

---

## Findings (as-found)

> Cataloged here, **not fixed in this wave** (D-10). Correctness findings are Wave-3
> batch-fix candidates (FIX-01); each carries file:line + a recommendation. Wave-3 triage
> decides which are in correctness scope vs deferred.

### F-01 -- Team-form `--current` path uses the DEPRECATED `build_team_form_features`

- **Where:** `scripts/build_team_form.py:358` (`build_for_current_week`) ->
  `scripts/build_team_form.py:88` -> `features/team_form.py:779` `build_team_form_features`
  (decorated with `warnings.warn(... DeprecationWarning ...)` at `features/team_form.py:799`).
- **Finding:** The current-week incremental team-form build runs through the **deprecated**
  `build_team_form_features`, whose own docstring says to use `build_features` instead
  because the deprecated path does **not** enforce the `as_of_datetime` time-fence. The
  canonical historical/Gold path (`build_features.py`) uses the time-fenced builder; the
  `--current` path does not. Observed at runtime as a `DeprecationWarning` emitted on every
  `--current` run.
- **Risk:** The current-week path -- the one that produces live Friday predictions -- bypasses
  the LeakageGate time-fence that the historical path enforces. AUDIT-04 (Wave 20-04) must
  confirm whether this introduces a real leakage exposure for current-week features or
  whether the rolling-average computation is incidentally safe.
- **Recommendation:** Wave-3 / AUDIT-04 -- route `build_for_current_week` through the
  time-fenced `build_features` path (or confirm + document that the deprecated path is
  leakage-safe for the current-week rolling window). Correctness-scoped (FIX-01) IF leakage
  exposure is confirmed.

### F-02 -- Team-form `--current` appends NON-IDEMPOTENTLY to `team_game_stats` silver

- **Where:** `features/team_form.py:819` `save_dataframe(team_stats_df, "team_game_stats",
  layer="silver", partition_cols=["season"])` (partitioned save appends rather than
  replaces). Reached from every `build_team_form_features` call (the `--current` path).
- **Finding:** Re-running the team-form build (current-week or otherwise) **appends** to the
  `team_game_stats` silver table instead of upserting/replacing. Measured on disk
  2026-05-28: `team_game_stats` holds **136,012 rows** of which **127,922 are duplicate
  `(game_id, team)` pairs** -- i.e. the table is dominated by re-computed duplicate rows
  accumulated across many historical builds.
- **Risk:** A non-idempotent silver write violates the Silver "latest-wins upsert" invariant
  (PROJECT.md). Downstream rolling-average computation reads this table; duplicate team-game
  rows could double-count plays into rolling EPA/success-rate features. AUDIT-02 (Wave 20-03)
  must check whether `calculate_rolling_averages` de-duplicates before aggregating (if it
  does, the impact is bloat-only; if not, the duplicates corrupt team-form features).
- **Recommendation:** Wave-3 -- change the `team_game_stats` write to an upsert/replace
  (mirror the Silver latest-wins convention) and de-duplicate the existing on-disk table.
  Correctness-scoped (FIX-01) IF AUDIT-02 confirms the duplicates reach the feature math;
  otherwise a robustness/bloat cleanup. NOTE: the audit harness itself patches this write
  out so the harness stays re-runnable -- it does not contribute to the duplication.

### F-03 -- `get_current_nfl_week()` returns `(2025, 22)` in the 2026 offseason

- **Where:** `utils/date_utils.py` `get_current_nfl_week` (returns `(2025, 22)` as of
  2026-05-28); consumed by `build_elo.py --current` (season 2025) and
  `build_team_form.py --current` (`build_for_current_week` targets 2025 week 22).
- **Finding:** During the offseason the "current week" resolves to a post-season placeholder
  (week 22) of the 2025 season. The `--current` paths therefore operate on 2025 (which is a
  documented partial season -- see Currency Gap below). This is expected offseason behavior,
  recorded for transparency: the dry-run's "current week" is not a live game week.
- **Risk:** None for the audit (the dry-run deliberately uses 2024 wk18 as its known-answer
  stand-in). Flagged only so Wave 21 (automation audit) accounts for offseason week
  resolution when it verifies the live Friday orchestrator.
- **Recommendation:** No action this phase. Phase 21 (AUTO-01..04) owns current-week
  resolution semantics in the live orchestrator.

---

## Data currency / freshness

### 2025 partial-season currency gap (D-05) -- documented, NOT backfilled

- **As-found coverage (verified on disk 2026-05-28):**
  - **Gold** (`features_wp` / `features_ats` / `features_ou`): 2025 present for **weeks 1-4
    only = 49 rows**. Full gold span is **2002-2025, 6263 rows**.
  - **Silver `games`:** 2025 present for **weeks 1-5 = 78 rows**.
- **Disposition (D-05 / AUDIT-05):** The 2025 season is treated as an **explained currency
  gap**: "2025 ingested through ~week 4 (gold) / week 5 (silver games), then frozen; the
  remaining 2025 weeks are NOT refreshed and will NOT be backfilled this phase." This
  satisfies AUDIT-05's "or any gap is documented and explained." Backfilling 2025 is a data
  expansion bordering on model work and is deferred to the future re-fit milestone, NOT done
  here. **Do NOT backfill 2025 in Phase 20.**
- **Note:** This corrects the earlier brief assumption "gold spans 2018-2024; 2025 absent."
  The live data shows gold spans 2002-2025 and 2025 is *partial*, not missing.

### Weather provider doc drift -- RESOLVED: Open-Meteo (not Meteostat)

- The stale codebase maps (`.planning/codebase/INTEGRATIONS.md`, dated 2026-03-18) name
  **Meteostat**; PROJECT.md names **Open-Meteo**. Resolution verified from live code:
  `scripts/ingest_weather.py:37` hits `https://archive-api.open-meteo.com/v1/archive`, and
  `tests/integration/test_openmeteo_smoke.py` confirms the live endpoint. **There is no
  Meteostat code in the repo.** The canonical weather provider is the **Open-Meteo
  Historical Weather API**. The weather hand-trace (Wave 20-04, D-08 area 3) verifies feature
  values against the Open-Meteo archive.

### Source-freshness diff (AUDIT-05) -- _pending Wave 20-05_

The nflreadpy vs on-disk games diff and the Open-Meteo single-game re-fetch (D-04: NO live
odds pull -- The Odds API is offseason / costly) are produced by Wave 20-05 and folded in by
Wave 20-06.

---

## Gold rebuild + metric moves (D-02) -- _placeholder, filled by Wave 20-06_

> Per D-10, Wave 3 performs the **single** end-of-phase gold rebuild (Elo / team_form
> `--all-seasons` -> contextual / weather / market `--season` -> `build_features`) followed
> by one `run_backtest.py` (+ `--blend`) re-run. Per D-02, the corrected gold is **ADOPTED
> as canonical regardless of whether metrics rise or fall**, but **no metric move is accepted
> silently**: every before/after move in `outputs/backtest/metrics_summary.json` and
> `season_metrics.csv` must be EXPLAINED here. An *unexplained* move blocks adoption pending
> investigation. Adoption is NOT gated on "no regression" (this differs from the 260524-svu
> precedent).

| Metric | Before (old gold) | After (rebuilt gold) | Move | Explanation | Adopted? |
|--------|-------------------|----------------------|------|-------------|----------|
| _WP Brier_ | _TBD_ | _TBD_ | _TBD_ | _Wave 20-06_ | _TBD_ |
| _WP log-loss_ | _TBD_ | _TBD_ | _TBD_ | _Wave 20-06_ | _TBD_ |
| _ATS CLV_ | _TBD_ | _TBD_ | _TBD_ | _Wave 20-06_ | _TBD_ |
| _O/U CLV_ | _TBD_ | _TBD_ | _TBD_ | _Wave 20-06_ | _TBD_ |
| _gold row count_ | 6263 | _TBD_ | _TBD_ | _Wave 20-06_ | _TBD_ |

**Reminder (D-01):** the gold rebuild + backtest re-run is the full extent of the cascade.
The deployed `artifacts/latest.json` models are **NEVER re-fit** this phase; the
production-vs-backtest mismatch is DIAG-05's subject (Phase 22).

---

## Non-correctness findings (deferred)

> Real findings that are **out of FIX-01 correctness scope**. Per D-11 these are **CAPTURED
> here, NOT fixed** this phase. They are robustness / hygiene / security-shape items for a
> future milestone or the GSD backlog.

### D-11-A -- Broad `except Exception` catches

- **Where:** `pipeline/health.py` (multiple: `:159,:209,:221,:256,:278,:321,:339,:392,:443`),
  `pipeline/execution_log.py:74`, `pipeline/orchestrator.py:175` (`# noqa: BLE001 -- must
  catch all to record in log`), `pipeline/model_validation.py:98`.
- **Finding:** Broad `except Exception` swallows in the pipeline package. Several are
  deliberate graceful-degradation / log-and-continue boundaries (the orchestrator's
  per-step catch is intentional and annotated); others are candidates for narrowing to
  operation-specific exception tuples (the pattern `data/storage.py` already adopted in
  Phase 15, where 18/19 broad catches were narrowed). PROJECT.md lists "no broad exception
  swallowing" as a validated v1.0 requirement, so the remaining pipeline catches are drift.
- **Disposition:** Capture only (D-11). Narrowing is a future robustness pass.

### D-11-B -- Silent in-memory-DB fallback

- **Where:** `data/storage.py:58-60` -- when `db_path` is falsy, `connect()` silently does
  `duckdb.connect()` (in-memory) and logs `"Connected to DuckDB in-memory"` at INFO.
- **Finding:** A misconfigured / empty `db_path` silently yields an ephemeral in-memory
  database instead of failing fast. Writes would appear to succeed but vanish on process
  exit -- a silent-data-loss shape. The canonical path always passes a file `db_path`, so this
  is latent, not currently triggered.
- **Disposition:** Capture only (D-11). A future hardening would make an unexpected empty
  `db_path` an explicit error (or at least a WARNING) rather than a silent INFO fallback.

### D-11-C -- SQL built by string interpolation (injection-shaped)

- **Where:** `data/storage.py:131` (`CREATE TABLE {sanitized_name} AS SELECT * FROM
  {temp_table}`), `:890`/`:892` (`SELECT {', '.join(sanitized_columns)} FROM
  {sanitized_name}` / `SELECT * FROM {sanitized_name}`), `:1085` (`SELECT COUNT(*) ... FROM
  {sanitized_name}`).
- **Finding:** Table and column identifiers are interpolated into SQL via f-strings rather
  than parameterized. **Mitigating control present:** an allowlist sanitizer
  (`data/storage.py:189` raises `ValueError "Invalid table name ... Only alphanumeric
  characters and underscores allowed"`, plus a reserved-word check at `:204`) gates the
  interpolated names, and all inputs are first-party table/column names (not user input).
  So the *exploitable* risk is low, but the **shape** is injection-prone string-built SQL
  (DuckDB identifiers cannot be bound as parameters, which is why this pattern exists).
- **Disposition:** Capture only (D-11; STRIDE Tampering, accepted/deferred per the plan's
  threat register T-20-DEFER). Out of FIX-01 correctness scope.

### D-11-D -- Stale `artifacts/models/*.pkl` stub files (cleanup)

- **Where:** `artifacts/models/wp_model.pkl`, `ats_model.pkl`, `ou_model.pkl` -- exist as
  2025-09-16 stub placeholders (731 / 6693 / 6693 bytes), distinct from the real
  `latest.json`-resolved artifacts under `artifacts/{target}_{ts}/` (2026-03).
- **Finding:** These stubs are the WRONG convention that the Wave-1 CR-01/IN-02 fix
  (plan 20-01) repointed `pipeline/health.py` + `staleness.py` + `model_validation.py` AWAY
  from. After that fix the stubs are **non-canonical and unreferenced by the pipeline**, but
  they were intentionally left on disk in 20-01 for capture here (per 20-01 SUMMARY +
  STATE.md). Nothing now reads them.
- **Disposition:** Capture only (D-11). Safe-to-delete cleanup, deferred (a Phase-19-style
  repo-hygiene removal rather than a correctness fix). Do NOT delete blindly mid-audit;
  schedule as a tidy-up once Wave-3 confirms no lingering reference.

---

## Cross-references

- **Re-runnable harness (D-12 deliverable a):** `tests/integration/test_audit_stage_runner.py`
- **Wave-1 blocker fix (CR-01 cluster, FIX-01 / D-03):** plan `20-01-SUMMARY.md`
- **Phase decisions (D-01..D-13):** `.planning/phases/20-data-feature-correctness-audit/20-CONTEXT.md`
- **Canonical run sequence:** `PIPELINE.md`
- **Wave-3 diagnostics (pending):** `outputs/diagnostics/` (AUDIT-02/03/04/05), folded in by
  plan 20-06.

---

*Phase 20 -- Data & Feature Correctness Audit. As-found baseline seeded 2026-05-28 (Wave 2).
Finalized by Wave 3 / plan 20-06.*
