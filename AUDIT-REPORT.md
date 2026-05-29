# AUDIT-REPORT.md -- Data & Feature Correctness Audit (Phase 20)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 20 -- Data & Feature Correctness Audit
**As-found baseline seeded:** 2026-05-28 (Wave 2 / plan 20-02)
**Finalized:** 2026-05-28 (Wave 3 / plan 20-06)

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

## Wave-3 correctness triage (FIX-01, D-10) -- as-applied

> Wave 3 (plan 20-06, Task 1) triaged every cataloged finding above plus the four Wave-2
> diagnostics (`outputs/diagnostics/audit_integrity.md`, `audit_handtrace.md`,
> `audit_freshness.md`, `audit_features.txt`, `audit_leakage.txt`) into CORRECTNESS
> (in FIX-01 scope, fixed) vs NON-CORRECTNESS (deferred, D-11). The honest outcome:
> **no feature-math correctness bug in FIX-01 scope was found** -- every cataloged feature
> finding (F-01, F-03, F-LEAK-01, F-ZERO-01, F-VALIDATOR-01) is either confirmed not to
> reach the gold feature math, or is an out-of-scope robustness/hygiene item (D-11). No
> feature/builder math was changed.
>
> **ONE enabling correctness fix landed under D-13 (owner-authorized, Task 2a):** the
> reproducibility defect behind finding F-02 -- non-idempotent per-season silver writes that
> let tables collide / append across runs (weather_features ~1048x, games_with_elo ~30.7x,
> contextual ~4.9x, team_game_stats 127,922 dup rows) -- was fixed at the storage layer
> (single-file `replace_mode` writes; dropped `partition_cols` on contextual/weather/market;
> the partition reader scoped to `{layer}/{table}/` so it can no longer read a sibling
> table's partitions, which had been feeding `weather_features` the `odds_snapshot` rows).
> The on-disk bloated tables were de-duped (commit `25c364f`). This is an ENABLING
> correctness fix for the D-10 single rebuild: the corrected silver is the cause of every
> gold value-hash change and every backtest metric move recorded below. It is scoped to
> data/storage correctness ONLY -- NO new features, NO re-tuning, NO algorithm swap, and
> NEVER a deployed-artifact re-fit (D-01). F-02 was thereby promoted from "deferred bloat"
> (its as-found Wave-2 disposition) to "fixed" because reproducibility is the v2.1
> milestone's namesake goal (D-13 owner authorization, OPTION A).
>
> The single gold rebuild + backtest re-run (D-10, exactly once) then ran on the corrected
> silver to materialize the adopted canonical gold (D-02).

| Finding | Triage verdict | Basis | Disposition |
|---------|----------------|-------|-------------|
| F-01 (team-form `--current` deprecated non-time-fenced builder) | NON-correctness for gold | AUDIT-04 (20-04) confirmed the canonical historical/Gold path uses the time-fenced `build_features` (`expanding_normalize` at `build_features.py:951,970`; deprecated `normalize_features_within_seasons:586` never called). F-01 is a CURRENT-WEEK-path concern, not a gold-leakage exposure. | Deferred; current-week routing semantics are Phase 21 (AUTO-01..04) scope. NOT fixed (no gold impact). |
| F-02 (non-idempotent `team_game_stats` append, 127,922 dup rows) | Enabling correctness (reproducibility) -- PROMOTED | AUDIT-02 (20-03) confirmed the duplicates do NOT corrupt the final gold *values* (all 3 matrices have 6263 clean rows, correct widths 156/157/156, zero all-null columns -- the gold assembly de-duplicates). BUT Task 2a found the same non-idempotent-write class broke build REPRODUCIBILITY across the wider silver (weather_features ~1048x growth + a partition reader that returned a sibling table's rows). Reproducibility is the v2.1 namesake goal. | FIXED at the storage layer under D-13 (owner OPTION A), commit `25c364f`. Scoped to data/storage correctness; NO artifact re-fit (D-01). On-disk bloat de-duped. |
| F-03 (`get_current_nfl_week()` = (2025,22) offseason) | NON-correctness | Expected offseason behavior; the dry-run deliberately used 2024 wk18 as its known-answer stand-in. | Phase 21 owns live week resolution. NOT fixed. |
| F-INTEG-01 (stale `16 if week<=18` completeness) | NON-correctness (robustness) | Only mis-labels Silver completeness percentages; never produces a wrong gold value (the AUDIT-02 gold-integrity check deliberately does not assert against it). | Deferred (D-11). NOT fixed. |
| F-INTEG-02 (odds/weather silver carry no team cols) | Informational | Tables key by `game_id` only; team coverage verified transitively. Not a defect. | No action. |
| F-LEAK-01 (`home_margin` label-sibling in features_ats) | NON-correctness (benign) | Confirmed live: `home_margin` is in `build_features.py:1001` `exclude_cols`, so it never enters `feature_cols`; it is excluded from the deployed ATS 25-feature list; the build-time gate runs on `combined_features` BEFORE targets are added. Zero false positives reach a model. | Deferred capture (D-11; the `LEAKAGE_KEYWORDS` substring brittleness is captured, not redesigned). NOT fixed. |
| F-ZERO-01 (20 all-zero gold columns) | NON-correctness (bloat) | 19/20 reach NO deployed model feature_list; the one that does (`heat_impact_score`, in the WP 20-feature list) is INERT (a constant feature contributes zero signal). Dead/bloat schema, not gold-value corruption. | Deferred cleanup (D-11). NOT fixed. |
| F-VALIDATOR-01 (stale validator naming + `_z` substring collision) | NON-correctness | Report-only `FeatureValidator` FAILs are stale-schema artifacts (completeness expects v1.0 raw names vs the current `home_*`/`away_*` schema; `_z` substring collides on `red_zone_td_rate`). Not data defects. | Run+harden only; do NOT rebuild the validator (RESEARCH key insight). NOT fixed. |

**Wave-3 correctness verdict:** [PASS] -- clean feature-math audit + one enabling
reproducibility fix. No FIX-01-scoped feature/builder MATH bug was found; the only Wave-3 code
change is the D-13 enabling storage-layer fix above (idempotent silver writes + de-dup, commit
`25c364f`), which is what makes the single rebuild reproducible and explains every metric move
below. The Wave-1 CR-01 cluster blocker fix (plan 20-01) remains unchanged. The D-11
non-correctness items remain captured below, NOT fixed. No deployed model artifact was re-fit
(D-01). `uv run pytest tests/unit -q` -> 762 passed (green after the fix + triage, no
regression).

---

## Post-review correctness fixes (CR-01, WR-01, WR-02)

> A follow-up deep code review of the phase-20 fix cluster (`20-REVIEW.md`,
> 2026-05-28) surfaced one Critical + two Warning correctness defects that the
> earlier waves had not yet caught. All three were fixed under the same
> correctness-only scope (NO new features, NO refactors beyond the fix, and
> NEVER a deployed-artifact re-fit -- D-01). The single D-10 gold rebuild was
> already done and was NOT re-triggered; these are code-path fixes only. Each
> landed as its own atomic commit. (Finding IDs below are `20-REVIEW.md` IDs,
> distinct from the F-* as-found IDs above.)

| Review ID | Severity | File | Fix | Commit |
|-----------|----------|------|-----|--------|
| CR-01 | Critical | `scripts/build_features.py` (gold save) | The per-season/current-week gold write used `partition_cols=["season"] if target_season else None`, re-introducing the shared-root partitioned-append antipattern that `25c364f` eradicated from every silver builder (`pq.write_to_dataset` writes `season=YYYY/` dirs into the SHARED `data/gold/` root where `features_wp/ats/ou` collide and each run appends a NEW hash-named parquet). Dropped `partition_cols` so BOTH paths write a single self-contained file via `save_dataframe`'s default `append_mode=True` `game_id` latest-wins dedup (mirrors `build_weather.py`/`build_contextual.py`/`pipeline/steps.py`). Current-week write is now idempotent; full-rebuild path still writes the complete single file (verified on disk: gold holds 3 single files, `features_wp` 6263 rows, span 2002-2025, width 156 -- unchanged). | `0269a15` |
| WR-02 | Warning | `features/market_anchors.py` (`identify_snapshot_lines`) | The deprecated snapshot cutoff was built at `hour=18` then localized to **UTC**, yielding `Friday 18:00 UTC` = `Friday 14:00 ET` (2 PM ET) -- 4 hours before the documented "Friday 6 PM ET" freeze. On disk every `snapshot_ts` is exactly `18:00 ET` = `22:00 UTC`, so comparing `22:00 UTC` snapshots against an `18:00 UTC` cutoff evaluated False and emptied the snapshot set on the orchestrator path. Localized the naive cutoff to ET (`utils.date_utils.ET`, America/New_York) and picked the target Friday from an ET-localized `latest_date`. Verified on disk: the fixed cutoff (`2024-09-13T18:00:00-04:00`) admits 1583 pre-cutoff rows including the 18:00 ET snapshots, vs an empty set before. Ruff auto-dropped the now-unused `UTC` import. | `dcc7883` |
| WR-01 | Warning | `features/team_form.py` (`build_team_form_features`) | `team_game_stats` was written unconditionally with `replace_mode=True` using only the seasons passed in. The `--current`/orchestrator path (`build_for_current_week`) passes a 3-season subset `[current-2, current-1, current]`, so a weekly current-week run shrank the on-disk `team_game_stats` from the full 2002-2024 history down to 3 seasons -- a silent data-loss regression. Fix: SKIP the persisted `team_game_stats` write on the incremental path (detected via `target_season` AND `target_week` set). Rationale: the full-rebuild path (`build_for_seasons`) is the canonical producer of the COMPLETE table and keeps `replace_mode=True`; no code reads `team_game_stats` from disk (`build_features.py` recomputes per-game stats in-memory via `get_per_game_stats()`), so the current-week run need not mutate it; and an append alternative is unsafe because the sibling rolling table `team_form_features` has no `game_id` (so `save_dataframe`'s `game_id` dedup would not apply and a plain append would duplicate). Idempotency now holds within a fixed seasons argument. This supersedes the as-found finding **F-02**'s `--current` data-loss aspect on `team_game_stats`. The integration test `test_build_team_form_current_path_runs`, which had encoded the OLD buggy "current-week path writes `team_game_stats`" behavior, was updated to assert the corrected "no persisted write on the incremental path." | `8f08163` |

**Verification:** `uv run ruff check` clean on all three files + the updated test;
each commit passed the pre-commit hooks (ruff + ruff-format). `uv run pytest
tests/unit -q` -> **762 passed, 0 failed** (after the D-11-F test correction
below). Relevant integration suites green: `test_audit_stage_runner.py`
(18 passed), `test_idempotency.py` + `test_audit_stage_runner.py` (24 passed),
`test_prediction_pipeline.py` (14 passed) -- confirming the `build_features`
gold change and the team-form `--current` change did not break the
current-week paths. No deployed model artifact was re-fit (D-01).

A fourth post-review fix (`bfe11c1`) corrected the `elo_game_snapshots` N/A test
assertion that `25c364f` had regressed (see D-11-F below) -- the snapshot data is
correct (2018-2025); the test assertion was wrong.

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

### Source-freshness diff (AUDIT-05) -- FOLDED IN (Wave 20-05)

Produced by Wave 20-05 (`outputs/diagnostics/audit_freshness.md`), folded in here:

- **Games (nflreadpy vs on-disk silver), season 2024:** source 285 games == on-disk 285;
  0 MISSING, 0 EXTRA. [PASS] -- no settled source game is missing from disk.
- **Weather (Open-Meteo single-game re-fetch), 2024_W06_SF@SEA (outdoor):** on-disk
  temp 56.4F / wind 12.5mph / precip 0.0mm vs live re-fetch 57.4F / 11.4mph / 0.0mm.
  [PASS] -- order-of-magnitude consistent within re-analysis tolerance.
- **Odds (D-04: NO live pull -- The Odds API is offseason / 500 req/hr / costly historical
  endpoint):** snapshot-timestamp confirmation only. Silver `odds_snapshot` = 1856 rows;
  latest freeze `snapshot_ts` = 2024-09-19 18:00 ET.
- **Determinism (D-09 double-run):** `generate_and_write` run twice on the fixed 2024 wk18
  snapshot is value-identical (`assert_frame_equal` on the game_id-sorted prediction frames).
  Artifacts were LOADED, NOT re-trained (D-01). Seeds confirmed pinned (random_state=42 across
  WP/ATS/OU; OU `np.random.seed` at `models/train_ou.py:374`); `n_jobs=-1` is a noted-but-
  out-of-scope train-time variable (XGBoost hist is deterministic for fixed data, and the
  prediction path is single-threaded inference -- confirmed empirically by the double-run).

---

## Gold rebuild + metric moves (D-02) -- FINALIZED (Wave 20-06)

> Per D-10, Wave 3 performed the **single** end-of-phase gold rebuild (Elo / team_form
> `--all-seasons` -> contextual / weather / market `--season` -> `build_features`) on the
> de-bloated, idempotent silver (the D-13 enabling fix above), followed by one
> `run_backtest.py` (+ `--blend`) re-run. Per D-02, the corrected gold is **ADOPTED as
> canonical regardless of whether metrics rise or fall**, but **no metric move is accepted
> silently**: every before/after move in `outputs/backtest/metrics_summary.json` and
> `season_metrics.csv` is EXPLAINED below. Adoption is NOT gated on "no regression" (this
> differs from the 260524-svu precedent).

### Single root cause for EVERY move

There is exactly **one** driver behind every metric move: the D-13 silver de-dup replaced the
bloated / cross-contaminated `weather_features` table (which had grown ~1048x and whose
partition reader was returning `odds_snapshot` rows) with a clean one-row-per-game weather
table. The gold dimensions are UNCHANGED (6263 rows, widths 156/157/156, span 2002-2025), but
the per-game weather values feeding the LEFT JOIN changed, so the three gold value-hashes
changed (wp `f4ce831e`->`939b4118`, ats `7b1b030d`->`4fac1b82`, ou `1bb6adcd`->`56a58fdc`).
The per-fold backtest models (fit INTERNALLY by the backtest, NOT a production re-fit) then
moved accordingly. **No move is UNEXPLAINED.**

### Per-target BEFORE -> AFTER (2021-2024 walk-forward backtest)

| Target | Metric | Before (old gold) | After (rebuilt gold) | Move | Direction | Explanation | Adopted? |
|--------|--------|-------------------|----------------------|------|-----------|-------------|----------|
| WP | accuracy | 0.66462 | 0.66725 | +0.00263 | better | clean weather feeding the per-fold WP LogReg+isotonic | YES |
| WP | MAE | 0.44091 | 0.43125 | -0.00967 | better | same clean-weather driver | YES |
| ATS | MAE | 10.04096 | 10.12177 | +0.08081 | worse | same driver; honest move adopted regardless of direction (D-02) | YES |
| ATS | R2 | 0.15067 | 0.14593 | -0.00474 | worse | same driver; adopted (no-regression NOT a gate) | YES |
| ATS | RMSE | 12.99319 | 13.02697 | +0.03378 | worse | same driver; adopted | YES |
| OU | MAE | 10.57858 | 10.49883 | -0.07974 | better | same driver | YES |
| OU | R2 | 0.04655 | 0.05485 | +0.00831 | better | same driver | YES |
| OU | RMSE | 13.27015 | 13.21305 | -0.05710 | better | same driver | YES |
| SIM | flat-stake ROI | -0.02031 | -0.00018 | +0.02013 | better | win_rate 0.5516 -> 0.5677; +4 bets cross the edge threshold (3158 -> 3162) on clean weather | YES |
| BLEND | wp CLV | -0.00347 | -0.00207 | +0.00140 | better | WP dynamic blend improved; ATS/OU blend delta 0.0 (gated to static per D-19) | YES |
| gold | row count | 6263 | 6263 | 0 | unchanged | rebuild is value-only; dimensions identical | YES |

- **WP**: both metrics improved.
- **ATS**: all three metrics moved slightly worse. Per D-02 this is an HONEST move (explained
  by the single clean-weather driver) and is ADOPTED -- adoption is NOT gated on no-regression.
- **OU**: all three metrics improved.
- **SIM / BLEND**: flat-stake ROI and WP-blend CLV both improved; ATS/OU blend unchanged (D-19
  static gating).
- **UNEXPLAINED moves: NONE.** Every move traces to the one D-13 clean-weather driver.

### Task-3 adoption decision (D-02): ADOPT

The owner reviewed the annotated per-target table above and, with **every move EXPLAINED by
the single clean-weather driver and ZERO unexplained moves**, selected **OPTION: ADOPT**. The
corrected (rebuilt) gold is now the **canonical on-disk data**, regardless of metric direction
(the ATS slight regression is an honest finding documented here, NOT reverted). This mirrors
the 260524-svu rebuild+adopt+confirm precedent but, per D-02, does **not** gate on
no-regression.

### Web-cache verification (UIAP-01, D-01) -- cache REFRESHED

The served `data/web_cache.duckdb` (last built 2026-05-26) was compared against the rebuilt
gold / re-run backtest. Served values **DID change** as a result of the rebuild -- e.g.
`headline_clv` wp -0.01645 -> -0.00207, ats 0.87013 -> 1.07622; flat-stake ROI -2.03143 ->
-0.01849 (pct), win_rate 55.161 -> 56.768; and the `game_context` Elo/weather display fields.
Because served values changed, the cache was **refreshed via
`uv run python scripts/populate_cache.py`** (a downstream DATA refresh from the adopted gold +
re-run backtest outputs, NOT a model re-fit -- D-01; UIAP-01's "no request-path math" is
preserved: the cache still holds only precomputed values). Post-refresh the cache serves the
adopted values (predictions 1139 rows; game_context 6263 rows; flat ROI -0.01849, win_rate
56.768; headline_clv wp -0.00207). **No deployed model artifact was re-fit.**

**Reminder (D-01):** the gold rebuild + backtest re-run (+ this cache refresh) is the full
extent of the cascade. The deployed `artifacts/latest.json` models are **NEVER re-fit** this
phase; the production-vs-backtest mismatch is DIAG-05's subject (Phase 22).

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

### D-11-E -- `build_market_anchors.py` deprecated consensus path schema mismatch

- **Where:** `features/market_anchors.py` `create_consensus_lines:499` reads
  `group["ml_home"]` (`:519`) while the consensus-output records it feeds (`:675`, `:700`)
  emit `opening_ml_home` / `snapshot_ml_home`. The deprecated `build_market_anchors.py`
  pre-build path that calls `create_consensus_lines` (`:624`, `:629`) would raise
  `KeyError 'ml_home'` (the column was renamed to `opening_ml_home` / `snapshot_ml_home`
  upstream).
- **Finding:** This is a latent `KeyError` on a **deprecated, non-canonical** market-anchor
  path. It is NOT on the canonical gold path: the live `build_features.py` build constructs
  the market-anchor features **on the fly** (it does not invoke `create_consensus_lines`), so
  the rebuilt gold this phase never touched this code. Surfaced during Wave-3 triage while
  confirming the market builder's contribution to the rebuild.
- **Disposition:** Capture only (D-11). Out of FIX-01 correctness scope because it cannot
  affect the canonical gold (dead path on the build_features-on-the-fly market route). A
  future cleanup either repairs the consensus column references or removes the deprecated
  path. NOT fixed this phase.

### D-11-F -- RESOLVED: elo-snapshot N/A test assertion regression (was, not a data gap)

- **Where:** `tests/unit/test_audit_trace_leakage_elo.py`
  (`test_pre_burn_in_games_have_no_spurious_elo_snapshot`).
- **Finding (corrected):** `25c364f` rewrote this test to assert
  `elo_game_snapshots["season"].min() == 2002`, citing a `20-RESEARCH.md`
  Pitfall-4 claim that "gold AND elo_game_snapshots span 2002-2025." That
  conflated two DIFFERENT spans: gold `features_wp` does span 2002-2025 (6263
  rows, inline burn-in Elo computed for every game, pre-2018 inclusive), but the
  `elo_game_snapshots` table legitimately spans only **2018-2025 (1991 rows)** --
  snapshots are recorded solely where market/odds context exists; pre-2018 is
  N/A by design. The 2018-2025 span is the VERIFIED-CORRECT behavior, independently
  confirmed by the 20-04 deep hand-trace (8 seasons / 3982 team-games = 1991 games)
  and the 260523-tp5 quick-task ("N/A for pre-2018 snapshot-less games"). So the
  data was right and the `25c364f` assertion was the regression (all three of its
  rewritten assertions failed or were vacuous against the real data).
- **Resolution (`bfe11c1`):** Corrected the test to the verified invariant --
  snapshots start at 2018, no pre-2018 gold game carries a snapshot (the real N/A
  check), and snapshot-window (2018+) gold games do have snapshots. No data/builder
  change (D-01/D-10 untouched); `uv run pytest tests/unit -q` -> 762 passed.
- **Note:** This was surfaced by the deep code review's investigation of the
  pass->fail after the regression suite re-ran; the underlying `elo_game_snapshots`
  artifact (2018-2025) needs no rebuild and no backfill -- it is correct as-is.

### D-11-G -- 20-REVIEW.md deferred findings (WR-03..07, IN-01..04)

> These are the explicitly-deferred items from the follow-up review
> (`20-REVIEW.md`); per the fix scope they are CAPTURED here, code left
> untouched. (Review IDs, distinct from the F-* as-found IDs.)

- **WR-03** -- `features/team_form.py:665-666`: the `build_features`
  `kickoff_et` time-fence guard is dead code (the `calculate_team_game_stats`
  output schema has no `kickoff_et` column, so the guard never fires). Latent
  leakage-safety dead code (the primary `week < target_week` fence still
  holds). Pre-existing. Deferred: either join `kickoff_et` so the guard
  actually fences, or remove the dead guard + docstring claim.
- **WR-04** -- `features/market_anchors.py:724-730`: market-efficiency features
  gate on `"opening_probs" in locals() and "snapshot_probs" in locals()`, a
  fragile cross-iteration `locals()`-membership idiom that could read a prior
  game's stale probs if the guard structure ever changes (in practice both are
  recomputed when both data dicts are present, so the stale value is currently
  overwritten before use). Pre-existing. Deferred: initialize
  `opening_probs={}`/`snapshot_probs={}` per iteration and gate on non-empty
  dicts.
- **WR-05** -- `data/storage.py:821-827` (write side) + on-disk
  `data/silver/snapshot_ts=*/`: orphaned `snapshot_ts=`/`season=` partition
  directories remain in the shared silver root after the `replace_mode` fix;
  `ParquetManager.exists()` (`:591-595`) scans the parent dir for ANY
  `*.parquet`-containing subdir and can false-positive on a sibling table's
  leftover partitions. Reads are correct today (single file takes precedence in
  `load()`). Deferred: delete the orphaned dirs and/or tighten `exists()` to the
  same table-scoped logic `load()` uses.
- **WR-06** -- `pipeline/health.py:191-195`: `check_data_freshness` subtracts a
  naive `datetime.now()` from a potentially tz-aware `pd.to_datetime(latest)`,
  raising `TypeError` that the per-table `except Exception` swallows into
  `is_fresh: False` -- a healthy-but-tz-aware table silently reported stale.
  Pre-existing. Deferred: normalize both sides to UTC-aware before subtracting
  (mirror `scripts/data_qa.py:147-154`).
- **WR-07** -- `scripts/build_market_anchors.py:200,204`: two emoji (CROSS MARK
  U+274C, WHITE HEAVY CHECK MARK U+2705) in print statements violate the
  CLAUDE.md no-emoji rule. Pre-existing (present at base `f519bda`). Noted, not
  mass-edited this pass. Deferred: replace with ASCII tags (`[CRITICAL]` /
  `[PASS]`).
- **IN-01** -- `scripts/build_weather.py:113,139` (DEGREE SIGN U+00B0),
  `scripts/data_qa.py:1093` (BULLET U+2022): non-ASCII (not emoji) source
  characters that break the `isascii()` discipline the audit harnesses enforce
  and can corrupt on Windows cp1252 consoles. Pre-existing. Deferred: ASCII
  equivalents (`deg F` / `*`) if strict-ASCII output is desired.
- **IN-02** -- `pipeline/staleness.py:326-327`: `_check_model_age` round-trips
  the age comparison (`age_days * 86400 > model_age_days * 86400`), an
  unnecessary multiply-back-out equivalent to `age_days > model_age_days`.
  Harmless, obscures intent. Deferred: simplify.
- **IN-03** -- `scripts/data_qa.py:34-38`: `GOLD_FEATURE_MATRICES` hardcodes
  exact column counts (156/157/156) and `test_audit_integrity.py` asserts
  against them; any legitimate future feature change fails with a non-obvious
  `schema_width_ok=False`. Accepted as an intentional audit tripwire; noted as a
  known brittleness trade-off. Deferred: optionally widen to a documented range
  or emit a clearer message.
- **IN-04** -- `pipeline/staleness.py:321` vs `pipeline/health.py:256`: the two
  graceful-degradation paths resolving models via `get_latest_artifact_path`
  handle the identical contract inconsistently (staleness narrows to
  `(OSError, json.JSONDecodeError)`; health uses bare `except Exception`,
  whitelisted via the `BLE001` per-file ignore). Deferred: align health's inner
  resolver catch to the narrow tuple (the outer broad catch can remain for the
  unbounded `stat()`/import surface). Related to D-11-A.

---

## Cross-references

- **Re-runnable harness (D-12 deliverable a):** `tests/integration/test_audit_stage_runner.py`
- **Wave-1 blocker fix (CR-01 cluster, FIX-01 / D-03):** plan `20-01-SUMMARY.md`
- **Phase decisions (D-01..D-13):** `.planning/phases/20-data-feature-correctness-audit/20-CONTEXT.md`
- **Canonical run sequence:** `PIPELINE.md`
- **Wave-2/3 diagnostics (folded in):** `outputs/diagnostics/audit_integrity.md` (AUDIT-02),
  `audit_handtrace.md` + `audit_leakage.txt` + `audit_features.txt` (AUDIT-03/04),
  `audit_freshness.md` (AUDIT-05/D-09) -- all folded into this report by plan 20-06.
- **Enabling reproducibility fix (D-13):** commit `25c364f` (idempotent silver writes +
  de-dup); single rebuild + backtest re-run on the corrected silver (plan 20-06).

---

*Phase 20 -- Data & Feature Correctness Audit. As-found baseline seeded 2026-05-28 (Wave 2).
Finalized by Wave 3 / plan 20-06 on 2026-05-28: one D-13 enabling reproducibility fix, one gold
rebuild ADOPTED as canonical (D-02, every move explained), cache refreshed (UIAP-01 preserved,
no artifact re-fit per D-01), all Wave-2 diagnostics folded in, D-11 deferred items captured.*
