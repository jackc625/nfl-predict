# AUTOMATION.md -- Scheduled Automation Explanation (Phase 21; daily run since Plan 33.2-28)

> **Changed 2026-09-24 (Plan 33.2-28):** the scheduled task no longer runs once a week on
> Friday. It runs EVERY DAY at 5 PM ET and predicts tomorrow's games before today's 6 PM ET
> lock (section 1). The weekly Friday run is retired as a scheduled job;
> `scripts/friday_pipeline.py` remains as a manual tool. Sections 2-7 still describe the
> orchestrator both runs share; sections 8-9 are the Phase 21 record and are kept as history.
>
> **Added 2026-10-05 (Phase 34):** the forward bet ledger went live. Four sections after
> section 10 explain its scheduled side: the closing-line wake task `NFL_Predict_Closing`, the
> Odds API credit budget, the ledger's push credentials and rulesets, and its run log. The daily
> run's ledger steps are listed in `PIPELINE.md`; the owner procedures (restore, re-adding the
> deploy keys, recording a real bet, verify and replay) are in `RUNBOOK.md`, "Forward ledger".
>
> **Changed 2026-10-15 (Phase 35):** alerts are no longer log-only. Section 7 now describes the
> phone push, the Saturday email and the outside watcher; the owner's setup steps, the TEST
> command and what each message means are in `RUNBOOK.md`, "Notifications".
>
> **Added 2026-10-18 (Phase 36):** the daily run now ends by pushing the dashboard to Vercel.
> "Website push", before the cross-references, explains it; the owner procedures are in
> `RUNBOOK.md`, "Website (hosted dashboard)".

This is the single source of truth for HOW the Friday automation runs and how to
tell whether a run succeeded. It explains what fires, when, what each of the 18
orchestrator steps does, where outputs and logs land, and the canonical "did Friday
succeed?" signal. It is the EXPLANATION layer only: Phase 23's runbook + state-of-system
summary will CITE this file for operational detail, and this file in turn CITES
`AUDIT-REPORT.md` (the Phase 20 data/feature-correctness audit handoff) and `PIPELINE.md`
(the canonical 7-stage run sequence the orchestrator wraps). It documents the
now-verified-and-fixed reality after Phase 21 Waves 1-2 (the durable AUTO-01 guard,
the offseason no-op, the three mechanical fixes, and the scheduling reconciliation),
not intentions.

Status tags are ASCII `[PASS]` / `[FAIL]` (no emoji, per CLAUDE.md).

---

## 1. What runs + when

The automation is a single Windows Task Scheduler task, `NFL_Predict_Pipeline`,
that fires once a DAY (Plan 33.2-28; until then it fired once a week, on Friday).

- **Trigger:** every day at `17:00` LOCAL = `5 PM ET`, one hour before the `18:00`
  (`6 PM ET`) lock. Every game locks at 18:00 ET on the ET calendar day before its
  kickoff (`utils/game_lock.py`), so all of tomorrow's games share one lock: 18:00 ET
  today. The `StartBoundary` in `deployment/windows_scheduler.xml`
  (`2026-09-24T17:00:00`, a `ScheduleByDay` trigger with `DaysInterval=1`) carries NO
  timezone offset, so `schtasks` interprets it in **machine-LOCAL time**. `17:00` local
  equals `5 PM ET` **only while the machine stays on Eastern Time** (the D-09 caveat). A
  daily run has 365 chances a year to be wrong rather than 52, so
  `setup_scheduling.py --install` now REFUSES to install on a machine whose Windows time
  zone is not Eastern. A zone changed after install can still mis-time the run, but the
  run's own passed-lock refusal compares absolute UTC instants, so it cannot collect or
  predict after a lock.
- **Finish before 6 PM (owner ruling 2026-09-23):** the WHOLE run -- collection, the
  gold build, prediction and emission -- must finish before the lock, so every emitted
  row was computed before its game locked. Measured at about 14.5 minutes, plus about
  12.4 more when one bad game forces the build to run again (27 minutes together). The
  one-hour lead leaves 33 minutes spare. The time of day is asserted by parsing the XML
  (`tests/unit/test_scheduler_xml_unchanged.py`, `MIN_COLLECTION_MARGIN_MINUTES`), not
  only by its byte pin. A run too slow for the lock is refused per game, never back-dated.
- **Canonical command:** `uv run python scripts/daily_lock_pipeline.py`
  (the `<Exec>` in the XML: `<Command>` = the absolute `uv.exe` path +
  `<Arguments>run python scripts/daily_lock_pipeline.py</Arguments>`).
- **Principal:** the owner user `jackc` with `LogonType=S4U` and
  `RunLevel=HighestAvailable` (NOT the SYSTEM SID `S-1-5-18`, which cannot see the
  owner's `uv` / `.venv` / `.env`). S4U runs whether or not the owner is interactively
  logged on, which the `WakeToRun` evening run needs. `LogonType=S4U` is the
  CONFIRMED working principal (D-08 register-and-verify, Section 9). Because the S4U
  logon does NOT load the owner's USER PATH, the `<Command>` must be the ABSOLUTE
  `uv.exe` path (`C:\Users\jackc\AppData\Roaming\Python\Python313\Scripts\uv.exe`) -- a
  bare `uv` fails under S4U with `0x80070002` "file not found". The remedy was the
  absolute path, not switching off S4U.
- **WorkingDirectory:** `C:\Users\jackc\Code\nfl-predict`. The committed XML is
  machine-specific (this path + the `UserId` principal) and must be re-pointed if the
  repo moves to a different machine or user.
- **Robustness settings (XML):** `MultipleInstancesPolicy=IgnoreNew`,
  `StartWhenAvailable=false`, `RunOnlyIfNetworkAvailable=true`, `WakeToRun=true`,
  `StopIfGoingOnBatteries=true`, `ExecutionTimeLimit=PT2H`, `AllowHardTerminate=true`,
  `AllowStartOnDemand=true`, `Priority=7`. The XML is UTF-16 LE with a BOM and CRLF line
  endings, and is byte-pinned (`tests/unit/test_scheduler_xml_unchanged.py`).
  `StartWhenAvailable` is FALSE on purpose (it was true for the weekly run): a trigger
  missed while the machine slept or was off is simply missed, because running it on wake
  could be hours after the lock. The next day's run predicts its own games at its own lock.
- **Installer (single source of truth, D-07):** the task is registered by
  `uv run python deployment/setup_scheduling.py --install` (elevated), which checks that
  `scripts/daily_lock_pipeline.py` exists and the machine is on Eastern time (`--test`
  separately rehearses the entry point with `--dry-run` and the Odds API key blanked),
  then runs
  `schtasks /create /tn NFL_Predict_Pipeline /xml deployment/windows_scheduler.xml /f`
  (idempotent overwrite) and cleans up the old split tasks (`NFL_Predict_DataUpdate`,
  `NFL_Predict_Predictions`). The task name did not change when the run went daily, so the
  install overwrote the weekly definition in place and no Friday task was left behind. The
  installer installs the committed XML so the installer and the definition can never
  drift; there is no second scheduling mechanism (no PowerShell `Register-ScheduledTask`,
  no cron).
- **Read-back (no admin needed):** `uv run python deployment/setup_scheduling.py
  --verify-installed` exports the installed task, parses it and the committed XML, and
  prints one `FIELD= committed | installed | MATCH` line each for the trigger, start time,
  `Command`, `Arguments` and `StartWhenAvailable`, then `READBACK_MATCH= True|False`
  (True also needs exactly one NFL task installed). The 2026-09-24 result is in section 9.

### What one daily run does

`scripts/daily_lock_pipeline.py` (Plan 33.2-27), in order:

0. **Passed-lock refusal, before any request.** If the run starts at or after 18:00 ET on
   its run date, every one of tomorrow's games has already locked: it appends a
   `lock_passed` record to `logs/daily_lock_runs.jsonl`, makes ZERO network requests and
   exits 0. A missed day is never back-filled.
1. **Schedule refresh.** Captures what nflverse serves now and ingests the season's
   schedule from that capture, so results, moved games and each new playoff round arrive
   with no manual step. The season is read from the recorded schedule and rolls over on
   its own.
2. **Select tomorrow's games** -- the games whose lock is today's 18:00 ET. None (most
   weekdays): a `no_games` record in `logs/daily_lock_runs.jsonl`, exit 0, nothing built.
3. **The run proper**, through the same orchestrator as section 2
   (`FridayPipeline(steps=pipeline.daily_steps.build_daily_step_registry(slate))`), with
   its own 21-step registry in three stages:
   - **Collection** -- tomorrow's weather forecasts, the season's snaps and injuries, and
     tomorrow's odds from The Odds API (real requests against the API key's quota), then
     `close_collection`, which stamps `captured_at_utc` and refuses the slate if
     collection itself ended after the lock.
   - **Build** -- data QA, Elo (with a flagged provisional week-start row for every unplayed
     game of the week), team form and weather features, then the **FULL-HISTORY gold build,
     every night** (owner ruling 2026-09-23, "Rebuild everything nightly": a one-season
     build diverges from the full build on 139 of 186 numeric columns, so only the full
     build is the real one), then the leakage scan, gold currency and model loading.
   - **Predict and emit** -- tomorrow's predictions merged into the week's file with three
     stamps (`captured_at_utc`, `information_cutoff_utc` = the lock, `computed_at_utc`),
     the bet list, the exports, validation and the web cache.

   A defective row refuses only its own game (the live-skip rule), never tomorrow's slate.
   The run prints `RUN_STATUS=` and a `SKIPPED <game>` line per dropped game.
4. **The website push (Phase 36)** -- after the ledger sync and before the phone pushes, every
   run that did not fail pushes the dashboard to Vercel (a run that failed records "not pushed"
   instead). See "Website push" near the end of this document.

`--date YYYY-MM-DD` runs as of another ET day; `--dry-run` runs the schedule refresh and
the collection stage under a recording write sink, writing nothing (it stops before the
build, which cannot read captures that were never saved).

### CLI modes (the manual Friday tool)

`scripts/friday_pipeline.py` is no longer scheduled; it remains a manual operator tool.
It accepts (the phase flags are mutually exclusive):

| Flag | Effect |
|------|--------|
| (none) | Full pipeline: all 23 steps (DATA + PREDICTIONS). |
| `--data-only` | DATA phase only (the 11 DATA-phase steps). |
| `--predictions-only` | PREDICTIONS phase only (the 12 PREDICTIONS-phase steps); logs a "ensure data artifacts are fresh" warning. |
| `--dry-run` | List the steps that would execute (filtered by mode); execute nothing; exit 0. Works year-round: `--dry-run` bypasses the offseason no-op short-circuit so steps can be inspected out of season without `--force` (WR-04). |
| `--force` | Bypass the pre-flight staleness/season checks AND the offseason no-op short-circuit; pre-flight health becomes advisory. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity (default INFO). |

> Note: until Plan 33.2-28 the scheduled task ran this script's FULL pipeline (no phase
> flag). The scheduled task now runs `scripts/daily_lock_pipeline.py` instead; every mode
> in this table is a manual / out-of-season operator tool.

---

## 2. The 5-phase orchestrator flow

The CLI resolves the mode from its flags and delegates to
`pipeline.orchestrator.FridayPipeline(force, mode).run()`. `run()` is a fixed 5-phase
flow:

- **A. Pre-flight staleness gate** (`pipeline/staleness.py`, bypassed by `--force`):
  a season-window check plus four freshness signals (week-valid, data-freshness,
  partial-run, model-age warn-only). Staleness *warnings* fire exactly one
  `alert_staleness_warning` (WARNING) and continue; a staleness *failure* sets
  `status="failed"`, fires `alert_pipeline_failure` (CRITICAL) and raises.
- **B. Pre-flight health check** (`pipeline/health.py`, ALWAYS runs): 3 read-only
  checks (database connectivity, model artifacts, disk space). Without `--force`,
  `unhealthy` aborts (failure alert + raise). With `--force`, `unhealthy` is advisory:
  it appends the warning `"Pre-flight health: unhealthy (forced)"` and continues.
- **C. Step execution:** the 23-step registry (`pipeline/steps.py`), phase-filtered by
  mode, run in order. The log is written atomically after each step (incremental
  snapshot). A **critical** step failure sets `status="failed"`, fires
  `alert_pipeline_failure` (CRITICAL) and raises immediately. A **non-critical** step
  failure appends a warning and continues; the run ends `degraded`. A **per-game
  information-time refusal** on a live run (Plan 33.2-03, D33.2-05) is neither: the caught
  games are dropped, recorded in the append-only `config/skip_records.jsonl`, and the step
  is re-run on the rest; the run ends `finished_with_skips` (it wins over `degraded`).
- **D. Post-run health check** (6 checks; ADVISORY ONLY): never changes `status` or the
  exit code. An `unhealthy` post-run result only appends `"Post-run health check:
  unhealthy"` to warnings.
- **E. Completion alerting** (exactly ONE alert per outcome): `failed` ->
  `alert_pipeline_failure` (CRITICAL); `finished_with_skips` ->
  `alert_finished_with_skips` (WARNING); `degraded` -> `alert_degraded_completion`
  (WARNING); `success` -> `alert_pipeline_success` (INFO).

### Offseason no-op short-circuit (D-06)

A live, scheduled, **unforced** Friday run during the NFL offseason would otherwise hit
the staleness season gate, set `status="failed"`, fire a CRITICAL "Pipeline Failed"
alert, and raise -- a false alarm that erodes trust (crying wolf). To prevent this,
`scripts/friday_pipeline.py::main` detects the offseason and returns `0` as a clean INFO
no-op **BEFORE** constructing `FridayPipeline`, so no orchestrator alert path is reached.
"Offseason" is read from the RECORDED SCHEDULE by the one resolver both sites call,
`utils.current_slate.resolve_current_slate` (Plan 33.2-24 step 24c), so the CLI
short-circuit and `StalenessGate.check_season` can never disagree: no slate is open
before a season's opener lock day, or after a season whose Super Bowl is recorded while
the next season has not begun. `--force` deliberately bypasses the short-circuit so the
operator can still run out of season (e.g. a one-time forced run against a
completed-week stand-in). A live unforced offseason run therefore exits 0 as a no-op with
**no CRITICAL alert**.

Step 24c retired the calendar window this used to compute (`[computed opener Thursday,
+ 22 weeks]`, resolved via `get_current_nfl_week()`), which was wrong at both ends: it
opened after the Wednesday 2026 opener's Tuesday lock, and it closed on 2026-02-05, three
days before the 2026-02-08 Super Bowl. One case is deliberately NOT a no-op: when the
schedule cannot name the slate -- the calendar season has turned (August onward) but its
schedule is not recorded in silver `games`, or a playoff store is stale by more than one
round -- the run exits 1 with an ERROR naming `uv run python -m scripts.ingest_games
--season <season>`, still before any orchestrator exists (so no CRITICAL alert). A silent
no-op there would skip a live slate, which is the failure step 24c removed.

---

## 3. The 23 orchestrator steps

The registry (`pipeline.steps.build_step_registry`) is exactly 23 `StepDefinition`
entries: 11 in the DATA phase, 12 in the PREDICTIONS phase. (`build_market_anchors` was
retired in the 33.2 review, batch 3: the silver table it wrote was read by nothing.) Each step uses deferred
imports (inside the function body) to avoid argparse collisions and module-level side
effects. `critical=True` means a failure aborts the run; `retryable=True` means transient
errors trigger retry.

| # | Step | Phase | Critical | Retryable | What it does |
|---|------|-------|----------|-----------|--------------|
| 1 | `capture_live_season` | DATA | yes | yes (3) | Capture what nflverse is serving RIGHT NOW into the append-only live zone, BEFORE anything consumes it. Calls `run_capture` directly with the run's resolved (season, week). |
| 2 | `ingest_games` | DATA | yes | yes (3) | Ingest current-week games via nflreadpy. |
| 3 | `ingest_weather` | DATA | no | yes (3) | Ingest weather forecasts via Open-Meteo (async). |
| 4 | `ingest_snaps` | DATA | no | yes (3) | Capture the season's player snap counts (Plan 33.2-15, D33.2-16): every row passes `SnapCountSchema` through `validate_bronze_to_silver` and carries the release asset's `upstream_captured_at` (capture provenance; a snap count's information time is its game's end). |
| 5 | `ingest_injuries` | DATA | no | yes (3) | Capture the season's injury reports (Plan 33.2-15, D33.2-16): schema-gated through `InjurySchema`; `upstream_captured_at` is a row's information time only where it is at or before that game's lock. The stamp lookup refuses on a rate limit rather than stamping the fetch instant. |
| 6 | `data_qa` | DATA | yes | no | Run data-quality validation; fail if checks failed. |
| 7 | `build_elo` | DATA | yes | no | Update Elo ratings for the current season. |
| 8 | `build_team_form` | DATA | yes | no | Build team-form metrics for the current week. |
| 9 | `build_contextual` | DATA | yes | no | Build contextual features (travel, rest, venue). |
| 10 | `build_weather_features` | DATA | yes | no | Build weather-based features for outdoor games. |
| 11 | `verify_data_artifacts` | DATA | yes | no | Verify the DATA-boundary silver artifacts exist AND carry a row for the current (season, week). Gold is NOT checked here -- it is built in the PREDICTIONS phase, so a currency check here would report an ordering fact as a stale artifact. |
| 12 | `ingest_odds` | PREDICTIONS | yes | yes (3) | Capture the odds snapshot from The Odds API. |
| 13 | `build_features` | PREDICTIONS | yes | no | Assemble the unified per-target gold feature matrices. |
| 14 | `validate_features` | PREDICTIONS | yes | no | Validate features for data leakage / quality. |
| 15 | `verify_gold_currency` | PREDICTIONS | yes | no | Verify the three gold matrices carry a row for the current (season, week). R9's "gold has no rows for this week" refusal, at the first point in the run where gold exists. |
| 16 | `validate_models` | PREDICTIONS | yes | no | Validate WP/ATS/OU models are available + loadable (via `artifacts/latest.json`). |
| 17 | `generate_predictions` | PREDICTIONS | yes | no | Generate current-week predictions (loads artifacts, applies market blend). |
| 18 | `verify_prediction_currency` | PREDICTIONS | yes | no | Verify the prediction file's ROWS are the current week, not only its filename. Runs before anything consumes it. |
| 19 | `generate_recommendations` | PREDICTIONS | yes | no | Select the week's +EV bet list through `BetSelector` and write the durable bet-list artifacts. |
| 20 | `export_artifacts` | PREDICTIONS | yes | no | Export the predictions CSV to JSON. |
| 21 | `validate_predictions` | PREDICTIONS | yes | no | Validate the prediction file (non-empty, required columns, `wp_prob` in [0,1]). |
| 22 | `verify_output_files` | PREDICTIONS | no | no | Verify the expected output files exist (advisory; warns on missing). |
| 23 | `populate_web_cache` | PREDICTIONS | no | no | Rebuild `data/web_cache.duckdb` so the served bet list is this run's. |

> Note: retry is handled by `tenacity` (`Retrying` with `wait_exponential` backoff) and
> fires ONLY on `TRANSIENT_EXCEPTIONS` (`ConnectionError`, `TimeoutError`, `OSError`,
> and `httpx.HTTPStatusError` when httpx is available). Local / validation errors
> (`ValueError`, `RuntimeError`, etc.) are NOT retried -- they fail fast. Only the six
> network-touching steps (`capture_live_season`, `ingest_games`, `ingest_weather`,
> `ingest_snaps`, `ingest_injuries`, `ingest_odds`) are retryable.

---

## 4. Where outputs land

All current-week prediction artifacts are written under `outputs/predictions/`
(via the single `pipeline.steps._predictions_output_dir()` helper):

- `outputs/predictions/predictions_<season>_week<week>.csv` -- the prediction matrix.
- `outputs/predictions/predictions_<season>_week<week>.json` -- the same, JSON.
- `outputs/predictions/game_context_<season>_week<week>.csv` -- per-game context.

The `generate_recommendations` step writes its own two artifacts under `outputs/bet_list/`,
NOT under `outputs/predictions/`:

- `outputs/bet_list/bet_list.parquet` -- the week's +EV bet list, one row per (game, target),
  live or suppressed, selected through `backtest.bet_selector.BetSelector`.
- `outputs/bet_list/bet_tracker.json` -- the precomputed realized-versus-expected tracker
  blocks, one per honesty class.

Both are CACHE SOURCE artifacts: the cache population step reads them into the temp build.
They are also the durable home of forward recommendation history -- a forward row whose game
freeze has passed is never rewritten by a later run.

> RETIRED (plan 31-17). The step used to write
> `outputs/predictions/recommendations_<season>_week<week>.json`, derived from a confidence
> tier with no expected value, no sizing and no suppression. It had NO code consumer and is
> gone; nothing reads or writes it.

### The web cache IS rebuilt by the orchestrator (changed by plan 31-18)

This boundary MOVED. Until plan 31-18 the orchestrator did not rebuild the web cache and this
document said so; that statement is now false and has been replaced by this section.

`populate_web_cache` is step **23**, the LAST entry in the registry. It rebuilds
`data/web_cache.duckdb` from the model artifacts, the backtest outputs, the gold/silver layers
and the two `outputs/bet_list/` artifacts, so the bet list the site serves after a Friday run is
the one that run selected rather than whatever a previous manual `scripts/populate_cache.py`
invocation left behind.

Three properties of that registration are load-bearing and are each pinned by a committed test
(`tests/unit/test_step_registry_order.py`, `tests/integration/test_orchestrator_degraded_cache.py`):

- **It runs LAST** -- strictly after `generate_recommendations` (which writes the durable
  bet-list artifact) and after `export_artifacts`, so the blob it reads is the one this run wrote,
  and after both validation steps, so a cache is never published from predictions that failed
  validation. The position is asserted by INDEX, so a future insertion cannot silently move it.
- **It is `critical=False`.** A cache failure sets `status="degraded"` and the run continues; the
  prediction steps' results are still recorded. Failing the run would discard good prediction
  output because a downstream convenience failed.
- **It needs no new alert code.** A degraded completion already routes to
  `alert_degraded_completion`, which is a log line. On the daily run (since Phase 35) a degraded
  run also sends a top-priority failed push to the owner's phone (section 7); the manual Friday
  tool sends nothing. Either way the real protection against a silently stale bet list is the
  `/bets` **hard-block** -- the page refuses to serve a week whose bet-list populated-at marker
  predates that week's latest per-game line freeze. A failed run leaves that marker unadvanced,
  which is what makes the block fire deterministically.

The orchestrator still does NOT train, promote, or backtest -- those remain the separate
PIPELINE.md stages 3, 4 and 5. Stage 7 (build cache) is now ALSO run by the orchestrator; running
it manually remains the documented recovery path and is the command the `/bets` refusal names.

---

## 5. Where logs land

The execution log is written to `logs/friday_pipeline.json`.

- **Single-file overwrite -- NO per-run history.** Each run overwrites the same file;
  there is no per-run archive or rolling history. The log reflects the MOST RECENT run
  only.
- **Written atomically** by `pipeline.execution_log.write_execution_log_atomic`
  (temp file + `os.replace`), so a crash mid-write never leaves a partial JSON file.
  The log is snapshotted after each step (incremental), so a mid-run inspection shows
  progress so far.
- **Shape:** `ExecutionLog` (status, start_time, end_time, season, week,
  total_duration_ms, forced, mode, pid, `steps[]` of `StepLogEntry`, `warnings[]`,
  `error`).

The daily run writes that same file on every night it has games (it runs the same
orchestrator). A night that predicts nothing -- no games tomorrow, or the lock already
passed -- returns before any orchestrator exists, so it leaves `logs/friday_pipeline.json`
as the last real run wrote it and instead appends one line (`run_date_et`, `outcome` =
`no_games` or `lock_passed`, `game_ids`, `recorded_at`) to `logs/daily_lock_runs.jsonl`.

Every daily run (and every manual push) also appends exactly one line to
`logs/site_pushes.jsonl`, the website push log (Phase 36): one record per push attempt,
including the attempts that did not push, never rewritten. Its fields and reasons are explained
in `RUNBOOK.md`, "Reading logs/site_pushes.jsonl"; how the push runs is in "Website push" below.

> Note: per-run log history is a deferred FUTURE enhancement (D-11). It is deliberately
> NOT added in this phase (documentation / verification only); the single-file overwrite
> is documented as-is.

---

## 6. How to tell whether a run succeeded -- the D-04 success-signal triad

A Friday run resolves to exactly one of three observable outcomes. The canonical signal
is a TRIAD: the `log.status`, the matching alert level, and the presence of the
predictions output file. These three map to distinct, observable signals (proven by the
durable AUTO-01 test added in Plan 21-04).

| Outcome | `log.status` | Alert method (level) | Exit code | Predictions file |
|---------|--------------|----------------------|-----------|------------------|
| success | `success` | `alert_pipeline_success` (INFO) | 0 | present |
| degraded | `degraded` | `alert_degraded_completion` (WARNING) | 0 | present (a non-critical step failed) |
| finished with skips | `finished_with_skips` | `alert_finished_with_skips` (WARNING) | 0 | present, with NO row for each dropped game (`skipped_games` in the log; the durable record is `config/skip_records.jsonl`) |
| failed | `failed` | `alert_pipeline_failure` (CRITICAL) | 1 | may be absent |

Stated plainly:

> **"Friday succeeded"** == `log.status` in `{success, degraded}` AND
> `outputs/predictions/predictions_<S>_week<W>.csv` exists (non-empty, `wp_prob` in
> `[0,1]`).

For the daily run the same triad applies on a night with games, and `finished_with_skips`
also exits `0`. A night with nothing to predict exits `0` with no predictions file; its
signal is the new `no_games` / `lock_passed` line in `logs/daily_lock_runs.jsonl`. Since Phase 35
the daily run also tells the owner directly: every outcome other than clean sends a top-priority
failed push, and every in-season run checks in with the outside watcher (section 7).

The exit code is `0` for BOTH `success` and `degraded` (a degraded run still produced
predictions; only a non-critical step like weather or output-file verification failed),
and `1` only for `failed`. The AUTO-01 keystone test
(`test_orchestrator_predictions_phase_e2e` in
`tests/integration/test_friday_prediction_step.py`) drives the REAL
`FridayPipeline(mode="predictions-only", force=True).run()` through the REAL step
registry (no stubbed registry -- that stubbing was the `cb61042` blind spot) and asserts
`log.status == "success"`, the predictions CSV exists, `game_id` is present, and
`wp_prob` is in `[0,1]` -- proving the triad's `success` branch produces a real,
leakage-safe predictions file. The registry itself is unstubbed, but two of the
PREDICTIONS-phase step BODIES are no-op'd (`step_ingest_odds`, `step_build_features`;
`step_build_market_anchors` has since been retired) so the committed gold supplies their inputs; the remaining
generate/validate/export/verify steps run for real (matching the test's docstring). An owner-confirmed one-time live forced run produced a
real predictions file offline (16 rows, `wp_prob` 0.176-0.867, all market rows matched
from the on-disk odds snapshot with NO live pull) with NO model artifact re-fit.

---

## 7. Notifications -- phone push, weekly email and an outside watcher (Phase 35)

Until Phase 35 every alert was a log line on the laptop, so nothing reached the owner. Since then
the system reaches the owner in three ways, all built in the `notify/` package:

- **Phone push** through ntfy (`https://ntfy.sh`), to the owner's secret topic. Exactly two kinds:
  **failed** and **bets** (plus the TEST message the owner sends on purpose).
- **One weekly email** through Resend, sent every in-season week by the Saturday run.
- **An outside watcher**, a healthchecks.io check that alerts the owner by push AND by email when
  the daily run has not checked in by 18:30 ET.

A message counts as delivered only when the provider answers with its own receipt: an ntfy message
id, a Resend email id, or healthchecks.io's `OK` body. A success answer with no receipt is a
failure. The owner's setup steps, the TEST command and what each message means are in
`RUNBOOK.md`, "Notifications".

### Who sends, and when

- **The daily run** (`scripts/daily_lock_pipeline.py`) sends at its very END: after the bet-list and
  ledger writes, after the web cache rebuild and after the ledger sync, on the returning path, on
  the raising path, and on the `lock_passed` early return. The order is: the failed pushes (top
  priority first), then the bets push, then the weekly email (on its digest day only), then the
  watcher check-in LAST. Sending can never change the run's exit code, a prediction, the bet list
  or a ledger byte, and a failing send never stops the next one.
- It sends **nothing** on `--dry-run`, on a refused `--date`, or on Ctrl+C. A process killed
  outright (power loss, out of memory) sends nothing either; the missing check-in is then the
  alert.
- **The closing-line task** sends only its own failed push for a failed reading, and never checks in
  ("Closing-line wake task" below).
- **The manual `scripts/friday_pipeline.py` tool sends nothing.**

### The two push events

| Event | ntfy priority | Sent when | Title (examples) |
|-------|---------------|-----------|------------------|
| failed | 5 (top) | Any 17:00 ET daily-run outcome other than clean: orchestrator status `failed`, `degraded` or `finished_with_skips`; a `run_failed` record (a crash anywhere, including before the orchestrator exists, such as the schedule refresh); `lock_passed` with at least one game tomorrow (or tomorrow's games unknown). One message per run. | `FAILED: 3 games not predicted`, `FAILED: daily run degraded (1 step failed)`, `FAILED: daily run, tomorrow's games unknown` |
| failed | 3 (normal) | One per ledger or closing failure: a settle pass that raised (`FAILED: ledger settle`) or deferred closing finalization (`FAILED: ledger closing finalization deferred`); a closing-trigger registration that failed or skipped with `closing_task_not_installed` / `machine_not_eastern` (`FAILED: closing line schedule`), or registered but read back wrong (`FAILED: closing line schedule read back wrong`); a failed anchor push (`FAILED: ledger anchor push`) or backup push (`FAILED: ledger backup push`); a refused or broken sync (`FAILED: ledger sync refused`); each failed closing reading (`FAILED: closing line capture`); a weekly email that was not delivered or not built (`FAILED: weekly email`). | as listed |
| bets | 3 (normal) | The run newly appended at least one live-arm, `status = live` bet to the forward ledger. It lists only those rows, as written, in `/bets` order, one line each: matchup, bet type, pick and line, stake in units, EV band. | `3 new bets: 2026 week 6, Sunday` |

- **Clean outcomes send nothing:** `success`, `no_games`, `offseason`, `lock_passed` with zero games
  tomorrow, a closing-trigger skip because the closing task was running, and a closing reading
  that ended `no_games`, `skipped` or `captured`.
- **Failed comes first.** A run that is not clean and also appended bets sends the failed push,
  then the bets push. A re-run of a slate lists only bets new to that run; an identical re-run
  sends no bets push.
- **No suppression.** There is no cooldown, dedupe window or rate limit: two failures in one night
  are two pushes. The push router refuses every event class other than failed, bets and test.
- **What a failed push says:** `What failed:`, `Games:` (the games left without a prediction,
  `unknown` when they cannot be read, or `none (predictions are not affected)` for a ledger or
  closing failure), `Error:` (one line: every configured secret replaced by `<redacted>`, every
  URL cut to its scheme and host, at most 160 characters) and `Next:`. For the daily run, before
  the lock the next action is the re-run command `uv run python -m scripts.daily_lock_pipeline`;
  at or after the lock it says those games can no longer be predicted.
- Bodies are plain ASCII, at most 4,000 bytes; a longer list is cut and ends `... and N more`, so
  ntfy never turns it into an attachment. No title carries an app name: the owner names the ntfy
  subscription "NFL Predict" in the app.

### The weekly email

- **Which run sends it:** the run whose tomorrow holds the week's Sunday games -- the Saturday
  17:00 ET run. A week with no Sunday game falls back to the run before its LAST slate day. It is
  sent on that run every week, playoffs included (`2026 week 19` keeps its number), and ALSO when
  that run failed, including a Saturday run that recorded `lock_passed`. If the recorded schedule
  cannot be read, a Saturday run still sends it (`RUN FAILED - bets unknown`); on any other
  weekday no email is attempted.
- **Subject:** `NFL Predict - <season> week <N> - <n> bets - run clean`, or
  `NFL Predict - <season> week <N> - RUN FAILED - <n> bets` when the run was not clean. ` (update)`
  is appended when a digest for the same run date was already delivered (a re-run sends one more
  email). No CLV figure ever appears in the subject or the status header.
- **Contents, in order:** (1) a status header -- `Run clean.` or `RUN FAILED:` with what broke;
  (2) the week's live bets decided so far, ranked as on `/bets`, with EV band and stake; (3) every
  game of the week as `bet`, `no bet`, `not predicted` (its lock passed with no row), or
  `picked <day> evening` (its lock is still ahead); (4) `0 bets this week` when the list is empty;
  (5) last week's actual vs expected return (the block `/bets` shows above the forward record);
  (6) the season-to-date tracker totals; (7) the week's skipped or missed closing readings,
  first-pick refusals and closing-schedule read-back failures; (8) the upstream data changes and
  corrected grades (below); (9) the same "not betting advice" line `/bets` carries, last.
- **Read, never recomputed.** Every figure is read from the web cache through the same getters
  `/bets` uses. The cache counts as current when the week's bet-list populated-at marker is at or
  after this run's start. An absent, unreadable or older cache, or a bet list it cannot read, gives
  a stale notice and `bets unknown` -- never `0 bets`. Last week's block says `not available` when
  its stored rows cannot be read (as on `/bets`), never `nothing graded`.
- **Upstream data changes and corrected grades.** The section lists the latest probe line's
  unacknowledged sealed-season changes at WARNING or louder, the live season's WARNING-or-louder
  verdicts recorded since the previous delivered digest (marked `fixed by ledger correction` when a
  stored ledger correction belongs to the change), and the graded live bets whose result was
  corrected since then. A source it cannot read is named as unreadable, never shown as nothing.
  It says runs are unaffected, because every run reads pinned bytes, and names the acknowledge
  command, `uv run python -m scripts.acknowledge_upstream_change`. That command records an owner
  ruling for ONE sealed-season change on `config/upstream_pin.sealed.lock`; it is not a re-freeze
  (`config/upstream_pin.json` is never touched), and live-season changes have no acknowledgement.
  A data change never sends a push.
- **When it cannot be sent or built:** after 3 attempts, or when building it raised, one
  normal-priority `FAILED: weekly email` push names the failure kind (`http_<status>`,
  `transport_<ExceptionClass>`, `no_receipt`, `missing_secret`, or `build_<ExceptionClass>` for an
  email that could not be built), and the run's check-in goes out as a failure. The email is not
  re-sent automatically by a later run.

### The outside watcher (healthchecks.io)

- **The check:** cron `0 17 * * *` in `America/New_York` (the zone, so it follows daylight saving),
  grace 5400 seconds: the check goes late at 17:00 ET and down -- and alerts -- at 18:30 ET. Its
  alert channels are configured on the watcher itself: the integrations named `NFL Predict ntfy`
  (the same ntfy topic) and `NFL Predict email`. `manual_resume` is false, so a paused check
  resumes on its next check-in. The settings are applied and read back field by field by
  `uv run python -m scripts.setup_watcher` (update-only, it never creates a check;
  `--verify-only` reads back without updating). It prints one
  `<field>= <want> | <got> | MATCH|MISMATCH` line each for `schedule`, `tz`, `grace`,
  `manual_resume` and `channels`, then `WATCHER_READBACK_MATCH= True|False`.
- **The check-in:** once, at the very end of every in-season daily run, whatever the outcome
  (`success`, `no_games`, `failed`, `lock_passed` alike). Its body is a short ASCII run summary,
  for example `NFL Predict daily run 2026-10-17: outcome=success; games tomorrow=13; bets
  appended=3; failed pushes=0/0; bets push=delivered; digest=delivered; unlogged receipts=0`, so
  the check's history page reads like a run log. It goes to the check's `/fail` URL when any
  message stayed undelivered after its attempts, or when a delivered message's receipt could not be
  written to the notification log; the watcher then alerts through its own channels. Only this one
  end-of-run step checks in: the TEST command, the closing-line task and `friday_pipeline.py`
  never do. If healthchecks.io cannot be reached, the missing check-in alerts at 18:30 ET -- a
  false alarm beats silence.
- **The season window,** read once at the run's start from the recorded schedule: in from 7 days
  before the season's first lock day; out from the day after the later of the day the Super Bowl
  result is recorded and the day after the Super Bowl, until 7 days before the next season's first
  lock day. Out of season every daily run pauses the check (idempotent, with the read-write API
  key) and never pings it; a failed pause is recorded in the notification log and not escalated.
  The first in-window check-in resumes it. A run that cannot read the window (an unreadable
  schedule) checks in as if in season.
- **The one watcher test day:** the optional `.env` line `WATCHER_TEST_WITHHOLD_DATE=YYYY-MM-DD`
  withholds the check-in on that exact ET date only, and only when that day is in season with
  zero games tomorrow; the run prints `WATCHER_CHECKIN= withheld (test day)` and records a
  `withheld` check-in. When the date matches but a condition fails, it prints
  `WATCHER_TEST_DAY= not withheld: <reason>` and checks in normally. Any other date ignores the
  line. Delete it after the test (`RUNBOOK.md`).
- The run prints `WATCHER_CHECKIN=` with one of `delivered`, `delivered as failure`,
  `failed <kind>`, `paused (out of season)`, `pause failed <kind>` or `withheld (test day)`.

### Retries and the notification log

- **Retries:** at most 3 attempts per message. Only transient failures are retried -- a network
  error or timeout, HTTP 429, any 5xx (and Resend's 409) -- after about 10 s, then about 30 s, with
  a 15 s timeout per request. A permanent failure (any other 4xx, a missing secret, a success with
  no receipt) fails at once after 1 attempt. A missing secret is a recorded failure
  (`missing_secret`), never a skip, and never crashes the run.
- **`logs/notifications.jsonl`** (gitignored) gets one JSON line per MESSAGE, not per attempt:
  `run_id`, `run_date_et`, `source` (`daily_run`, `closing_task` or `test_command`),
  `event_class` (`failed`, `bets`, `test`, `digest`, `checkin` or `pause`), `priority` (5, 3, or
  null for an email, check-in or pause), `attempts`, `outcome` (`delivered`, `failed` or
  `withheld`), `receipt_id`, `failure_kind` and `sent_at`. The shape is closed: a record never
  holds a topic, URL, key, recipient or message body. Each line is appended in one write under a
  lock file, so the daily run and the closing task never interleave. It is the one place to read
  what reached the owner. A dry run writes nothing to it.
- **Console lines** on the daily run: `NOTIFY_FAILED=` and `NOTIFY_BETS=` (`delivered <receipt id>`
  or `failed <kind>`, then the title; `NOTIFY_BETS= none` when there is no bets push),
  `NOTIFY_DIGEST=` (`delivered <receipt id>`, `failed <kind>` or `not a digest day`), and the
  `WATCHER_CHECKIN=` line above. A delivered message whose record could not be written also prints
  `NOTIFICATION_LOG= failed <ExceptionClass>`.

### The secrets

Five values in `.env`, read through `conf.settings.Settings` as `SecretStr`, so a repr, log line or
traceback never prints them; a mistyped key fails validation naming the field, never its value.
`.env.example` lists the names with empty values. Never commit a value.

| Env var | What it is |
|---------|------------|
| `NTFY_TOPIC` | The ntfy topic: at least 32 random characters. ntfy topics are public by design, so the name itself is the secret. |
| `RESEND_API_KEY` | The Resend API key. |
| `DIGEST_TO_ADDRESS` | The weekly email's recipient. Resend's free tier sends from `onboarding@resend.dev` only to the Resend account's own address, so it must be that address. Treated as a secret: never in a log record. |
| `HEALTHCHECKS_PING_URL` | The check's ping URL, `https://hc-ping.com/<uuid>`. It embeds the check uuid. |
| `HEALTHCHECKS_API_KEY` | The healthchecks.io project's READ-WRITE API key, used only to pause the check out of season and by the setup command's update and read-back. |

Plus the optional, non-secret `WATCHER_TEST_WITHHOLD_DATE` (the one watcher test day). The ntfy
server and the Resend sender are code constants, not `.env` values.

### The TEST command

`uv run python -m scripts.notify_test` sends, through the real sender, one
`TEST - FAILED: notification test` push at priority 5, one `TEST - <n> new bets: notification test`
push at priority 3 (the web cache's current-slate live bets), and one TEST digest email (the real
weekly email from the current cache, its subject starting `TEST - NFL Predict`). It prints
`NOTIFY_TEST_FAILED=`, `NOTIFY_TEST_BETS=` and `NOTIFY_TEST_DIGEST=`, each `delivered <receipt id>`
or `failed <kind>`, and leaves three notification-log records with source `test_command`. Exit 0
when all three were delivered, 1 otherwise. It never checks in with the watcher.

### The old alert layer is not the notification path (D-05)

The orchestrator's own alert layer, `pipeline/alert.py`, still writes exactly one log line per run
outcome (the alert column of section 6's table) through
`utils.alert_manager.AlertManager.create_alert`. It is log-only by design and is NOT the
notification path: it is shared with the push-free manual `scripts/friday_pipeline.py` tool, so it
is never rerouted into `notify/`. Phase 35 deleted the old email and Slack sending code, the
cooldown and hourly-limit logic and the unused alert methods from `utils/alert_manager.py`, and the
unread `monitoring.notifications` block of `conf/config.yaml`. The `.env` flags
`ENABLE_EMAIL_REPORTS` and `ENABLE_SLACK_NOTIFICATIONS` are no longer declared in `Settings`, which
refuses keys it does not declare: a leftover line naming either stops every settings load, so it
must not be in `.env`. Sections 9 and 10 keep
the Phase 21 record of why the old channels were inert.

---

## 8. Findings (FIXED this phase)

> The D-12 automation-audit findings record. This section catalogs what was FIXED in
> Phase 21 (correctness / wiring only -- documentation / verification milestone, no model
> re-fit and no gold rebuild, D-01). Mirrors `AUDIT-REPORT.md`'s F-NN / D-11-x entry
> shape. Commit hashes are the atomic per-task commits.

### F-01 -- Team-form `--current` rerouted onto the time-fenced builder

- **Where:** `scripts/build_team_form.py::build_for_current_week`.
- **Fix:** rerouted off the deprecated, non-time-fenced `build_team_form_features` onto
  the time-fenced `TeamFormCalculator.build_features` Protocol method, so the live
  current-week build no longer emits a `DeprecationWarning`. Current-week rolling rows
  are persisted to silver `team_form_features` via a `(target_season, target_week)`-keyed
  upsert that preserves the full on-disk history (the WR-01 invariant -- no season-span
  shrink, no append-merge growth). The operative leakage guard remains the
  `week < target_week` filter in `calculate_rolling_averages`.
- **Commit:** `a01aa7a`.

### D-11-D -- Stale `artifacts/models/*.pkl` stubs deleted

- **Where:** `artifacts/models/{wp,ats,ou}_model.pkl`.
- **Fix:** deleted the three stale stub files (the wrong convention the Phase 20 CR-01
  fix repointed health / staleness / model_validation AWAY from). They were UNTRACKED, so
  removed by filesystem delete (Phase 19-02 precedent). `latest.json` + the real
  `artifacts/{target}_{ts}/` dirs are untouched; nothing on the Friday path reads the
  stubs.
- **Commit:** `f90966f`.

### D-11-E -- Market-anchor `create_consensus_lines` KeyError fixed

- **Where:** `features/market_anchors.py::create_consensus_lines`.
- **Fix:** fixed the latent `KeyError 'ml_home'` (the column was renamed upstream to
  `opening_ml_home` / `snapshot_ml_home`) by making the median computation prefix-aware
  so both the opening- and snapshot-line callers work.
- **Live + critical (the fix prevents a Friday-run crash).** `create_consensus_lines`
  is reached LIVE on every Friday run by the critical PREDICTIONS step 10,
  `step_build_market_anchors` (`pipeline/steps.py:211`), which calls the deprecated
  `MarketAnchorFeaturesCalculator.build_market_anchor_features`
  (`features/market_anchors.py:589`), which in turn calls `create_consensus_lines`
  (`features/market_anchors.py:649,654`) for both the opening- and snapshot-line frames.
  Before this fix, that call raised `KeyError 'ml_home'`, which would have aborted the
  critical step and failed the whole Friday run with a CRITICAL "Pipeline Failed" alert.
  The fix therefore prevents a live critical-step crash -- it is load-bearing, not cosmetic.
- **Still no gold value change.** The "no gold value changes" half holds: the canonical
  gold matrix is built on the fly by the SEPARATE FeatureBuilder Protocol method
  `MarketAnchorFeaturesCalculator.build_features` (`features/market_anchors.py:937`),
  invoked by step 11 `build_features` (`scripts/build_features.py`), which does NOT read
  the silver `market_anchor_features` table that step 10 writes. So step 10's silver
  output is unconsumed by gold and the fix changes no gold value -- but the path is LIVE
  and CRITICAL, not dead. (That step 10 writes an unconsumed silver table is a separate
  deferred cleanup; see Section 10.)
- **Commit:** `f90966f`.

### AUTO-02-F1 -- Post-run prediction-pipeline health glob corrected

- **Where:** `pipeline/health.py::check_prediction_pipeline`.
- **Fix:** the post-run check globbed `current_predictions*.parquet` -- a pattern the real
  run never produces -- so it ALWAYS reported "No prediction files found" even on a
  successful run. Changed the glob to `predictions_*_week*.csv` (via the shared
  `_predictions_output_dir()` helper) and read CSV. The check is advisory-only (it never
  blocks Friday), but it can now actually report `healthy`, strengthening the D-04
  "predictions file present" signal.
- **Commit:** `1ba2405`.

### D-06 -- Offseason no-op short-circuit (no crying wolf)

- **Where:** `scripts/friday_pipeline.py::main`.
- **Fix:** an unforced offseason run now returns 0 as a clean INFO no-op BEFORE
  constructing `FridayPipeline`, so it never hits the staleness season gate and never
  fires a false CRITICAL "Pipeline Failed" alert. The offseason window mirrors
  `StalenessGate.check_season` exactly; `--force` bypasses. No new `log.status` was added
  (Option A, smallest blast radius).
- **Commit:** `cc66264`.
- **Superseded in mechanism by Plan 33.2-24 step 24c:** both sites now read "offseason"
  from the recorded schedule (`utils.current_slate`) rather than a calendar window; the
  no-op contract itself is unchanged (see section 2, "Offseason no-op short-circuit").

### D-07..D-10 -- Windows scheduling reconciled into one drift-proof story

- **Where:** `deployment/windows_scheduler.xml`, `deployment/setup_scheduling.py`.
- **Fix:** corrected the canonical XML -- owner principal `jackc` + `LogonType=S4U` +
  `RunLevel=HighestAvailable` (NOT the SYSTEM SID, D-10); `StartBoundary` restored to
  `18:00` local = 6 PM ET (undoing the Phase-19 5 PM / 17:00 consolidation, D-09);
  `<Exec>` `<Command>` changed from the hardcoded `.venv\Scripts\python.exe` to run the
  pipeline via `uv` (D-10). Made `setup_scheduling.py` a single installer that registers
  THAT XML via `schtasks /create /xml ... /f` (D-07), removing the dead Linux/mac cron
  branch. No second automation mechanism.
- **Commits:** `45e2b84` (XML), `bfcf108` (installer), `6b72e41` (tests).
- **D-08 register-and-verify follow-up fixes** (discovered at the D-08 owner checkpoint;
  these correct the 21-03 artifacts so the task actually registers and launches):
  - `02e8b6a` -- the XML header comment contained illegal `--` double-hyphens (inside the
    `--platform` / `--install` install instructions), which is forbidden inside an XML
    comment, so `schtasks /create /xml` rejected the file ("The task XML is malformed.
    (9,55) incorrect comment syntax"). Rephrased the comment to be `--`-free and added an
    XML well-formedness + no-`--`-in-comment regression test (the prior 21-03 test only
    string-matched content and never PARSED the XML -- that gap is now closed).
  - `2061907` -- the bare `<Command>uv</Command>` failed under S4U with `0x80070002`
    ("file not found") because the S4U logon does NOT load the owner's USER PATH (`uv` lives
    at `C:\Users\jackc\AppData\Roaming\Python\Python313\Scripts\uv.exe`, on the User Path,
    not the Machine Path). Owner-approved fix: set `<Command>` to the ABSOLUTE `uv.exe`
    path and KEEP `LogonType=S4U` (runs whether logged on or off, no stored password). The
    `<Arguments>` stay `run python scripts/friday_pipeline.py --log-level INFO`.
- **Final state (D-08 verified):** `LogonType=S4U` is the confirmed working principal once
  the `<Command>` is the absolute `uv.exe` path -- the remedy was the absolute path, NOT
  switching to `InteractiveTokenOrPassword`. The owner re-ran (elevated)
  `uv run python deployment/setup_scheduling.py --install` ("Task 'NFL_Predict_Pipeline'
  created successfully"), then `schtasks /run` (SUCCESS) and `schtasks /query ... /v`:
  Run As User = `jackc` (NOT SYSTEM), Task To Run resolves the absolute `uv.exe` running
  `scripts/friday_pipeline.py`, Schedule = Weekly / FRI / 6:00 PM, Next Run Time =
  9/18/2026 6:00 PM (season-anchored, correct for the offseason), State = Enabled,
  `Last Result = 0` after `/run` (the run launched `uv.exe` under S4U and exited 0 via the
  D-06 offseason no-op, so no `logs/friday_pipeline.json` is written -- the no-op returns
  before the orchestrator is constructed). Details in Section 9.

### AUTO-01 -- Durable end-to-end orchestrator guard

- **Where:** `tests/integration/test_friday_prediction_step.py`
  (`test_orchestrator_predictions_phase_e2e`).
- **Fix:** added the durable `cb61042` guard -- a permanent, non-mocked, offseason-safe
  test that drives the REAL `FridayPipeline(mode="predictions-only", force=True).run()`
  through the REAL step registry (deliberately NO `build_step_registry` patch and NO
  `make_mock_step`, since those ARE the aliasing blind spot) and proves a leakage-safe
  predictions file is produced. Plus an owner-confirmed one-time live forced run, offline,
  with no model re-fit.
- **Commit:** `a9e8105`.

---

## 9. Owner-executed verifications (D-08 scheduling, D-05 alerts)

Two verifications complete the automation audit on the owner's machine. They require
admin / a local `.env` and cannot be CI-automated. Both are now DONE and recorded here.

### D-08 (AUTO-03 completion) -- scheduled task registered + verified [PASS]

The owner registered `NFL_Predict_Pipeline` (elevated) via
`uv run python deployment/setup_scheduling.py --install` ("Task 'NFL_Predict_Pipeline'
created successfully"; the old split task names were cleaned up), ran it once with
`schtasks /run` (SUCCESS), and confirmed it via
`schtasks /query /tn NFL_Predict_Pipeline /fo LIST /v`:

| Field | Verified value |
|-------|----------------|
| Run As User | `jackc` (NOT SYSTEM) |
| Logon Mode | Interactive/Background (= `LogonType=S4U`) |
| Task To Run | `uv run python scripts/friday_pipeline.py --log-level INFO` (via the absolute `uv.exe`) |
| Start In | `C:\Users\jackc\Code\nfl-predict` |
| Schedule | Weekly, Days = FRI, Start Time = 6:00 PM, Start Date = 9/11/2026 (Friday) |
| Next Run Time | 9/11/2026 6:00 PM (the boundary date is itself the first Friday of the 2026 season; season-anchored -- correct for the offseason) |
| State | Enabled |
| Stop-after | 02:00:00 (`ExecutionTimeLimit=PT2H`) |
| Last Result | `0` after `schtasks /run` (Last Run Time 5/29/2026 1:00 PM) |

The `Last Result = 0` confirms the task launched `uv.exe` under S4U and exited 0 via the
D-06 offseason no-op (no `logs/friday_pipeline.json` is written because the no-op returns
before the orchestrator is constructed -- correct offseason behavior). Reaching this
required the two follow-up fixes in Section 8 (`02e8b6a` malformed-comment, `2061907`
absolute `uv.exe` path). **Final working `LogonType` = S4U** (the remedy for the
`0x80070002` PATH failure was the absolute `uv.exe` path, NOT switching to
`InteractiveTokenOrPassword`).

> **Note (WR-03):** the committed `windows_scheduler.xml` `StartBoundary` was corrected
> from the Saturday `2026-09-12T18:00:00` to the Friday `2026-09-11T18:00:00` (the first
> Friday of the 2026 season), so the anchor weekday now matches the `DaysOfWeek=Friday`
> trigger. The owner's ALREADY-REGISTERED OS task still carries the old `9/12` boundary;
> re-installing (`uv run python deployment/setup_scheduling.py --install`) to pick up the
> corrected boundary is OPTIONAL / non-urgent -- the registered task already fires on
> Fridays (the Saturday anchor only ever shifted the first eligible fire to the next
> Friday), so behavior is unchanged either way.

The table above is the Phase 21 WEEKLY task and is kept as history; the next subsection
supersedes it.

### Plan 33.2-28 -- the daily task installed and read back [PASS]

On 2026-09-24 the owner re-ran `uv run python deployment/setup_scheduling.py --install`
(elevated). The same task name was overwritten in place, so the weekly Friday task is gone.
`--verify-installed`, re-run read-only afterwards, printed:

| Field | Committed | Installed | Result |
|-------|-----------|-----------|--------|
| Trigger | `CalendarTrigger ScheduleByDay(DaysInterval=1)` | same | MATCH |
| StartTime | `17:00:00` | same | MATCH |
| Command | the absolute `uv.exe` path | same | MATCH |
| Arguments | `run python scripts/daily_lock_pipeline.py` | same | MATCH |
| StartWhenAvailable | `false` | `false` | MATCH |

`NFL_TASKS= ['NFL_Predict_Pipeline']`, `READBACK_MATCH= True`. `schtasks /query /v` reports
Schedule Type Daily, Start Time 5:00:00 PM, Start Date 9/24/2026, Run As User `jackc`,
State Enabled, Status Ready, Next Run Time 9/24/2026 5:00:00 PM. The owner authorised it
to run for real every day ("Install and run for real", 2026-09-24).

**Liveness limits, stated plainly:** the trigger fires in machine-local time (see section 1);
a night the machine is off at 5 PM, or asleep and not woken (`WakeToRun` needs Windows wake
timers allowed), is a missed night, never a late one; and a run whose laptop is unplugged
mid-run is stopped (`StopIfGoingOnBatteries=true`).

### D-05 (alert path) -- documented-only; email + Slack are inert [PASS]

> Superseded by Phase 35: the inert email/Slack code and its config block were deleted and rebuilt
> as `notify/` (section 7). The record below is kept as history.

The owner did NOT enable email/Slack. A read-only diagnostic of `utils/alert_manager.py`
+ `conf/settings.py` (recorded in Section 7) empirically confirmed that **email AND Slack
are currently inert and cannot fire via config alone**: `AlertManager` reads the
non-existent `self.settings.monitoring` (so `monitoring_config == {}` always),
`MonitoringConfig` declares no `notifications` field, and `Settings` lacks the SMTP fields
the email path references. The console/log alert always fires and IS the working behavior;
the default stays log-only. Enabling email/Slack requires the deferred code changes in
Section 10 -- it is NOT a matter of populating config keys.

---

## 10. Non-correctness findings (deferred)

> Real findings that are OUT of correctness scope for this documentation/verification
> phase. Per the HARD BOUNDARY (D-01/D-12) these are CAPTURED here, NOT fixed. They are
> robustness / hygiene items for a future milestone or the backlog. (Pointer: these
> mirror and extend the `AUDIT-REPORT.md` deferred record.)

- **WR-03 -- dead `kickoff_et` time-fence guard.**
  `features/team_form.py:665-666`: the `build_features` `kickoff_et < as_of_datetime`
  guard is dead code because the `calculate_team_game_stats` output schema has no
  `kickoff_et` column, so the guard never fires. The operative leakage protection on the
  team-form path is the `week < target_week` filter inside `calculate_rolling_averages`.
  Pre-existing and latent (not a live leakage exposure). Deferred: either join
  `kickoff_et` so the guard actually fences, or remove the dead guard + the docstring
  claim. Documented inline at the F-01 routing site.
- **WR-04 -- fragile `locals()`-membership idiom.**
  `features/market_anchors.py:724-730`: market-efficiency features gate on
  `"opening_probs" in locals() and "snapshot_probs" in locals()`, a fragile
  cross-iteration `locals()`-membership idiom that could read a prior game's stale probs
  if the guard structure ever changes (in practice both are recomputed when both data
  dicts are present, so the stale value is currently overwritten before use).
  Pre-existing. Deferred: initialize `opening_probs={}` / `snapshot_probs={}` per
  iteration and gate on non-empty dicts.
- **D-11-A -- broad `except Exception` drift in the pipeline package.**
  Several deliberate graceful-degradation / log-and-continue boundaries use a broad
  `except Exception` (annotated `# noqa: BLE001`). Live sites observed this phase:
  `scripts/friday_pipeline.py::main` (top-level CLI catch, ~line 149, "top-level CLI must
  catch all") and `pipeline/orchestrator.py::_execute_step` (~line 175, "must catch all
  to record in log"). These per-step / top-level catches are intentional (they must
  record any failure in the log / exit cleanly). Others in `pipeline/health.py` /
  `execution_log.py` / `model_validation.py` are narrowing candidates. PROJECT.md lists
  "no broad exception swallowing" as a validated v1.0 requirement, so the remaining
  catches are drift. Deferred: a future robustness pass narrows the non-intentional ones
  to operation-specific exception tuples (the pattern `data/storage.py` adopted in
  Phase 15). The intentional CLI / per-step catches stay.
- **ALERT-WIRING -- email/Slack alert channels are inert (D-05 finding).**
  Superseded by Phase 35: the inert email/Slack code and its config block were deleted and rebuilt
  as `notify/` (section 7); the original record follows.
  `utils/alert_manager.py` + `conf/settings.py`: the email and Slack alert channels cannot
  fire via configuration today (verified empirically in Section 7). Three independent
  breaks: (1) `AlertManager.__init__` reads `self.settings.monitoring`, which does not
  exist (the YAML monitoring block is at `self.settings.config.monitoring`), so
  `self.monitoring_config == {}` always and every level resolves to `["console"]`;
  (2) `conf.settings.MonitoringConfig` declares no `notifications` field, so Pydantic
  (extra=ignore) drops `config.yaml`'s `monitoring.notifications` block even via the
  correct path; (3) `conf.settings.Settings` lacks the SMTP fields
  (`email_from`, `smtp_host`, `smtp_port`, `smtp_use_tls`, `smtp_username`,
  `smtp_password`) that `AlertManager._send_email_alert` references (would `AttributeError`
  if reached). Captured, NOT fixed (documentation/verification milestone -- HARD BOUNDARY
  D-01/D-12; no new feature). 3-part remedy for a future robustness milestone:
  (a) declare a `MonitoringConfig.notifications` sub-model (or have `AlertManager` read the
  raw YAML); (b) fix `AlertManager` to read `self.settings.config.monitoring` instead of
  the non-existent `self.settings.monitoring`; (c) add the SMTP fields to `Settings`
  (sourced from `.env`, never committed). Until then, alerts are console/log only and the
  default log-only behavior is the working behavior.
- **STEP10-UNCONSUMED-SILVER -- CLOSED in the 33.2 review, batch 3: the step, its writer and
  the unread silver table are retired.** The original record follows.
  `pipeline/steps.py:211` (`step_build_market_anchors`) saves the deprecated
  `build_market_anchor_features` output to silver `market_anchor_features`, but the
  canonical gold is built on the fly by the separate Protocol `build_features`
  (`features/market_anchors.py:937`), which does not read that silver table. So the
  silver write is unconsumed by gold (see Section 8 D-11-E). The path is still LIVE and
  CRITICAL -- a crash there fails the Friday run -- so it cannot simply be removed.
  Deferred: reroute step 10 onto the Protocol `build_features` builder (and drop the
  unconsumed silver write) in a future cleanup; OUT of this documentation/verification
  phase's scope.
- **MA-MEDIAN-INT-TRUNC -- median moneyline truncated by `int()`.**
  `features/market_anchors.py:1037-1038`: the compressed `build_features` path computes
  `ml_home_med = int(snap_ml_home_vals.median())`, truncating toward zero; for an even
  number of books the median can be a half-integer (e.g. -127.5 -> -127), slightly
  biasing the implied probability. Pre-existing (outside this phase's diff). Deferred:
  round instead of truncate, or devig from the median raw probability rather than the
  median moneyline.
- **UV-PATH-MAINT -- absolute `uv.exe` path is Python-major-version-pinned (maintenance
  note).** `deployment/windows_scheduler.xml`: the scheduled task's `<Command>` is the
  absolute `C:\Users\jackc\AppData\Roaming\Python\Python313\Scripts\uv.exe` (required
  because the S4U logon does not load the owner USER PATH, see Section 8 `2061907`). The
  `Python313` segment is version-pinned: on a Python MAJOR upgrade (e.g. to Python 3.14)
  the per-user Scripts directory moves to `Python314\` and `uv.exe` will no longer resolve
  at the committed path. Maintenance action on a Python major upgrade: re-point the
  `<Command>` to the new `Scripts\uv.exe` path and re-run
  `uv run python deployment/setup_scheduling.py --install` to re-register the task.

For the broader deferred robustness/hygiene catalog (D-11-B SQL-string-build,
silent in-memory-DB fallback, the `20-REVIEW.md` WR-05..07 / IN-01..04 items, etc.) see
`AUDIT-REPORT.md`'s "Non-correctness findings (deferred)" section.

---

## Closing-line wake task (NFL_Predict_Closing)

Added 2026-10-05 (Phase 34). A second scheduled task, beside `NFL_Predict_Pipeline`, takes the
closing-line reading every forward bet is later compared against (forward closing-line value,
which is report-only).

- **What it runs:** `uv run python -m scripts.capture_closing_lines` (the absolute `uv.exe` path,
  working directory the repository). It takes no arguments. It picks the unplayed games kicking
  off in the next 60 minutes; with none it records `no_games` and makes no request, so it spends
  nothing. Otherwise it reads the remaining credits (free), applies the credit rule below, requests
  ONE odds board and appends it to `data/silver/odds_snapshot.parquet` as closing rows. It prints
  one `CLOSING_CAPTURE=` line and writes exactly one `closing_capture` event: `no_games`,
  `skipped` (with the reason), `captured` or `failed`. The odds store has one lock,
  `data/silver/.odds_snapshot.lock`, shared with the daily run's capture, so the two never write at
  once. A closing row never decides a bet: every decision reader judges a row by its capture time
  against the lock, and closing rows are captured after it.
- **A failed reading pushes (Phase 35).** A `failed` reading sends one normal-priority push,
  `FAILED: closing line capture`, naming the games, and the task still exits 1; `no_games`,
  `skipped` and `captured` send nothing, and two failures in one night are two pushes. The failure
  event and the push are independent: if the failure cannot even be written to
  `logs/ledger_runs.jsonl` (the task prints `CLOSING_CAPTURE_LOG= failed <ExceptionClass>`), the
  push still goes and says so, because the weekly email reads the run log and will not list that
  failure. A push that raises prints `NOTIFY= failed <ExceptionClass>` and changes nothing else.
  This task never checks in with the outside watcher, which watches the daily run only, so a
  window MISSED because the laptop was off sends nothing; it shows up in the weekly email's
  closing section instead.
- **When it wakes:** only at real kickoff windows taken from the schedule, never on a repeating
  timer. A window opens 10 minutes before the earliest kickoff it serves and covers every kickoff
  within 60 minutes of that start, so 4:05 and 4:25 PM games share one reading. Each window is one
  `TimeTrigger` whose `EndBoundary` is 30 minutes after its start, so a trigger that did not fire
  in time never fires late, and `StartWhenAvailable` is false, so a missed window is simply missed,
  never run on wake.
- **AC power only:** `WakeToRun` is true, and Windows allows wake timers on AC power only, so a
  reading is taken only while the laptop is plugged in. The principal and settings are the daily
  task's (S4U logon as `jackc`, `HighestAvailable`, `IgnoreNew`, both battery flags false); the
  one difference is a 20-minute `ExecutionTimeLimit`.
- **Triggers regenerated every day from real kickoffs:** the committed template,
  `deployment/windows_closing_scheduler.xml`, carries no trigger. Every day the 5 PM daily run
  refreshes the schedule, it re-registers the task with one trigger per window over the next 8
  days (`schtasks /create /xml logs/closing_task_generated.xml /f`) and reads it back. It does so
  whether or not tomorrow has games, and it never fails the run: the outcome is printed as
  `CLOSING_TRIGGERS= <outcome> <reason>` and recorded as a `closing_triggers` event. It skips with
  `closing_task_not_installed`, `closing_task_running` or `machine_not_eastern`. Each registration
  covers 8 days, so one missed or failed daily run loses nothing. Since Phase 35 a registration
  that `failed`, or skipped with `closing_task_not_installed` or `machine_not_eastern`, sends one
  normal-priority push (`FAILED: closing line schedule`), and so does one that registered but did
  not read back as expected (`FAILED: closing line schedule read back wrong`, also listed in the
  weekly email); a skip because the closing task was running sends nothing.
- **The S4U task regenerates the triggers itself (measured).** The S4U push smoke on 2026-10-05
  (`tests/phase34_state.py`, `P34_S4U_PUSH_SMOKE`) showed the S4U context can create, read back
  and delete a scheduled task. So no owner-run weekly install is needed; the fallback planned for
  an S4U context that could not register tasks is not in use.
- **Install and read back:** the first install, or any re-install, is an owner step in an ELEVATED
  terminal: `uv run python deployment/setup_scheduling.py --install-closing`. It refuses a machine
  that is not on Eastern time, reads the silver schedule, registers the next 8 days and reads the
  task back. Read it back at any time with
  `uv run python deployment/setup_scheduling.py --verify-installed`, which works unelevated and
  prints `NFL_TASKS=`, `CLOSING_TRIGGERS= <expected> | <installed> | MATCH`,
  `CLOSING_READBACK_MATCH= True` and `READBACK_MATCH= True`. The trigger comparison is against the
  windows due at the moment you run it, so a `MISMATCH` shortly after a window opened only means
  that window has passed; the next daily run re-registers. Remove the task with
  `schtasks /delete /tn NFL_Predict_Closing /f` (elevated).
- **Elevation, stated once:** registering or deleting an S4U + `HighestAvailable` task -- `--install`,
  `--install-closing`, any S4U smoke task, `schtasks /delete` -- needs an elevated ("Run as
  administrator") terminal; an agent or normal session gets "Access is denied". Nothing INSIDE
  the running tasks needs elevation: both deploy-key pushes and the daily trigger regeneration run
  under S4U with no owner action.
- **Installed** by the owner from an elevated terminal on 2026-10-05 at 17:52 ET, read back with 7
  triggers and `CLOSING_READBACK_MATCH= True`; its first wake was that evening at 20:05 ET.

A game whose reading was skipped or missed keeps NULL closing columns with a recorded reason
(`credit_reserve`, `credit_header_unreadable`, `capture_failed`, `capture_empty` when the reading
ran but wrote no row for the game, else `capture_missed`); the settle pass finalizes them once,
after the game's result is recorded. A `captured` event carries the evidence for an empty or
partial board: `board_events`, `board_games` (each event's listed commence time), `rows_by_game`,
`empty_game_ids` and the transform's `unmatched` / `in_play` / `no_kickoff` lists.

## Odds API credit budget

The Odds API key is the free Starter plan: 500 credits a month. One board (moneyline, spread and
total, one region) costs 3 credits. The daily decision capture spends one board on every decision
day -- a day whose next ET day has a game -- and each closing reading window spends one more.

The written budget, measured by `tests/unit/test_credit_budget.py` on the recorded 2026 schedule
(272 games, weeks 1-18; run it with `-rP` to print the table):

| Month | Decision days | Decision credits | Closing windows | Closing credits | Total | Worst case (decision x3 + closing) |
|---|---|---|---|---|---|---|
| 2026-09 | 11 | 33 | 16 | 48 | 81 | 147 |
| 2026-10 | 13 | 39 | 25 | 75 | 114 | 192 |
| 2026-11 | 15 | 45 | 30 | 90 | 135 | 225 |
| 2026-12 | 15 | 45 | 26 | 78 | 123 | 213 |
| 2027-01 | 3 | 9 | 5 | 15 | 24 | 42 |

Every month is under 500, worst case included, with at least 2.2x margin. The worst case assumes
every decision day spends its full 3-board retry bound. The busiest month is November: 135 credits
planned, 225 at worst.

**Playoffs are not in the table.** The recorded schedule held weeks 1-18 only, because playoff
weeks 19-22 were not yet seeded, so the January row covers weeks 17-18 alone and February is
absent. The research estimate for the full January is about 93 credits, plus about 6 for the Super
Bowl in February -- both far under the quota.

**The reserve rule: the decision capture always comes first.** Before each closing board, the
reading checks that the credits left after it still cover every remaining decision capture this
calendar month at the retry bound: reserve = 3 credits x 3 boards x the month's remaining decision
days, today included. When `remaining - 3` is below the reserve, the reading is skipped with
`credit_reserve`. The decision capture never consults this rule.

**The credit count is read, never computed, and reading it costs 0 credits.** The remaining count
comes from the `x-requests-remaining` header of the free `/v4/sports` endpoint, read just before
each board. It is never worked out from our own spending, because the provider resets the quota
"on the first of every month" without saying at what hour or in what time zone. An absent or
unreadable header skips the reading with `credit_header_unreadable`: it fails closed and spends
nothing.

## Ledger pushes and credentials

The ledger is published to two places, each with its own SSH deploy key (GitHub refuses one deploy
key on two repositories):

| Key file (owner-only, outside the repo) | Registered on | What the code pushes with it | Public fingerprint |
|---|---|---|---|
| `C:/Users/jackc/.ssh/nfl_ledger_anchor_ed25519` | `jackc625/nfl-predict` (public), write access | only `ledger-anchor`, by the one refspec `refs/heads/ledger-anchor:refs/heads/ledger-anchor`, never forced | `SHA256:xsqcWlikkdWs9BMB/BwTwIPCdO8wGbbDbav2ueMg8n4` |
| `C:/Users/jackc/.ssh/nfl_ledger_backup_ed25519` | `jackc625/nfl-predict-ledger` (private), write access | only the backup's `main` (`HEAD:refs/heads/main`), never forced | `SHA256:SYq1qgchjEPGiE7AeHC2E1WjF3jt0uzjdfmf9SoTCDQ` |

- **The push can only use that key.** Each push runs Windows OpenSSH by absolute path
  (`C:/Windows/System32/OpenSSH/ssh.exe`) with that one key and no agent (`IdentitiesOnly`),
  `BatchMode`, and `StrictHostKeyChecking=yes` against GitHub's host keys pinned in
  `C:/Users/jackc/.ssh/nfl_ledger_github_known_hosts`. It cannot fall back to the owner's own
  credentials. `origin` stays HTTPS for the owner's normal work; ledger pushes name the SSH URL
  explicitly. Measured from the real S4U context on 2026-10-05: GitHub greeted each key as its
  repository, not as the owner, and both pushes landed.
- **The keys have no passphrase,** because a scheduled task cannot type one, so their file
  permissions are the protection: `JACKSLAPTOP\jackc` alone has access. They are never committed
  or printed.
- **The keys are tied to the owner's `gh` login.** They were added with `gh repo deploy-key add
  --allow-write`, and GitHub deletes keys added by `gh` when that login is revoked, so logging `gh`
  out removes them. The symptom is a growing `REMOTE_ANCHOR_BEHIND_BY=` / `BACKUP_BEHIND_BY=` in
  verify; `RUNBOOK.md`, "If pushes start failing", re-adds them.
- **The `master` ruleset** (`master: owner-only updates (ledger deploy key refused)`, id 24502158)
  puts update, deletion and non-fast-forward rules on the public `master`, with only the repository
  admin role allowed to bypass. No deploy key is on the bypass list, so GitHub itself refuses a
  deploy-key push to `master`; the owner's normal pushes go through as admin (GitHub prints
  "Bypassed rule violations"). It was verified by reading it back from the API, never by a test
  push.
- **The `ledger-anchor` ruleset** (id 24536696, created right after the first anchor push) forbids
  deleting or force-pushing that branch, with no bypass actor at all, so the published history of
  heads cannot be rewritten.
- **The private backup repository has NO ruleset.** GitHub Free offers none on private
  repositories. That is NOT the same protection as the public branches have: the backup's only
  protection is that no ledger push ever forces, so every earlier backup commit stays reachable on
  `main`. The public anchor is the outside check on it -- a backup that had been rewritten would
  disagree with the published head, and verify after a restore would fail.
- **Failures are recorded and retried, never fatal.** Every push outcome is a run-log event
  (`anchor_push`, `backup_push`, with `ok` and the error). A failed push, an unreachable GitHub or a
  missing or locked backup repository never fails the daily run; the next sync carries the current
  head. Retry by hand with `uv run python -m scripts.sync_ledger`; audit with
  `uv run python -m scripts.verify_ledger`. Only a broken chain refuses a sync (`sync_refused`),
  because nothing is ever published from a broken chain.
- **Each failure also reaches the owner's phone (Phase 35).** At the end of the daily run, each
  failed anchor push (`FAILED: ledger anchor push`), each failed backup push
  (`FAILED: ledger backup push`) and a refused or broken sync (`FAILED: ledger sync refused`) sends
  one normal-priority push, read from the sync's error fields; so does a failed or skipped closing
  schedule (`FAILED: closing line schedule`, "Closing-line wake task" above) and a settle pass that
  raised or deferred closing finalization ("Ledger run log" below). The push's next action names
  `uv run python -m scripts.sync_ledger` for a push failure, and for a refused sync
  `uv run python -m scripts.verify_ledger` and then the restore procedure in `RUNBOOK.md`. A day
  with no new row makes no ledger push, so it sends no such message.
- **Read the whole setup back** with `uv run python -m scripts.setup_ledger_remote verify`: keys,
  file permissions, pinned host keys, the backup's visibility, both deploy keys, both rulesets and
  the nested `ledger/` repository. It ends `SETUP_READBACK_MATCH= True`.

## Ledger run log

`logs/ledger_runs.jsonl` (gitignored) gets one JSON line per ledger event, appended and never
rewritten. Each line carries `event`, `at_utc` and `run_id`, plus the event's own facts --
outcomes, counts, row keys and hashes, never a key or a secret. The event names are a closed list;
a name outside it is refused, so a typo cannot create a second spelling a reader would miss:

| Event | Written when |
|-------|--------------|
| `append` | the daily run's recommend step appended forward rows (counts of appended, identical and refused rows) |
| `first_pick_refusal` | a re-run decided a row differently from the stored one; the first pick stands (also printed as `LEDGER_REFUSED=`) |
| `settle` | the settle pass wrote grades, corrections or closing values (with the counts), or failed (`ok` false, with the error); a pass with nothing owed writes nothing |
| `correction_appended` | a score corrected after grading changed a row's outcome, so a correction entry was appended |
| `correction_label_withdrawn` | a graded game's corrected score is now missing; recorded for the owner, nothing appended |
| `closing_finalized` | a row's closing columns were set: the closing values, or NULL with a reason |
| `anchor_commit`, `anchor_push` | the sync committed a new anchor (only when the head moved) and pushed it |
| `backup_commit`, `backup_push` | the sync committed the changed `ledger/` directory and pushed the backup |
| `sync_refused` | the sync refused to publish (broken chain or unreadable ledger) |
| `closing_triggers` | the daily run re-registered (or skipped re-registering) the closing wake task |
| `closing_capture` | one closing wake ran: `no_games`, `skipped`, `captured` or `failed` |
| `fill_recorded` | the owner recorded a real bet's fill (`scripts.record_fill`) |
| `migration` | the one-time migration of the old writer's rows (2026-10-05) |
| `cutover` | reserved in the list for the go-live switch; nothing writes it today (the switch was a commit) |

**`run_id` ties one daily run together.** The daily run binds one id for its whole run, prints it
as `LEDGER_RUN_ID=`, and every ledger event it writes -- the append, the settle pass and the
end-of-run sync's anchor and backup events -- carries that id. So "did this run's rows reach
GitHub?" is answered by finding the run's `append` and then its `anchor_push` and `backup_push`
with the same `run_id`. Events written outside a daily run (an owner CLI, a closing wake) carry
`run_id` null. Every sync event also carries the `head_hash` and `entry_count` it published.

**The notification log is a separate file.** What reached the owner -- each push, email,
check-in and pause -- is recorded in `logs/notifications.jsonl` (section 7), never here. A settle
pass that raised (`settle` with `ok` false) or deferred closing finalization also sends one
normal-priority push (`FAILED: ledger settle`, `FAILED: ledger closing finalization deferred`).

**A failure is recorded by its exception class name only** in the events the daily run writes
itself when a side step raises -- a failed `settle` or `closing_triggers` -- and in the closing
task's failed `closing_capture` (`error_type`), never by its message, which can carry a URL with a
key in it (Phase 35). The ledger sync's own events (`anchor_commit`, `anchor_push`,
`backup_commit`, `backup_push`, `sync_refused`) keep the error text they always carried.

### One ledger write the run log does not hold (2026-10-05, disclosed)

The private backup's commit `b84cb7f` (2026-10-05 17:58:30 ET) has no `settle`,
`closing_finalized` or `backup_commit` line in `logs/ledger_runs.jsonl`. It was a TEST run, not a
production writer: `tests/unit/test_daily_lock_run.py` was re-run against the working tree in the
minutes after the cutover commit flipped the switch ON and before the suite pinned it OFF
(commit `1871840`, 18:30 ET). A test in it that does not set the switch itself ran the real
settle pass and sync on the production `ledger/` while the file's fixture had redirected the run
log into a temporary directory, so the events went there. Every production ledger writer records its events. Two fingerprints in the
stored rows identify it, and neither can be corrected (closing and grading halves move from NULL
once):

- **7 week-4 rows carry `graded_at` 2026-09-26T14:00:00+00:00**, the test's fixed run instant,
  not the real grading time (5:58 PM ET on 2026-10-05). Their outcomes and payouts re-grade
  clean from the silver scores (D-19).
- **42 week-4 rows carry `closing_null_reason` `capture_missed`** (the other 42 changed rows, all of
  week 3, are `no_bet_side`). Every one of those games kicked off before `NFL_Predict_Closing`
  was installed (17:52 ET that day), so for them `capture_missed` means no closing reading was ever scheduled,
  not that the machine slept. The next real settle would have written the same reason. The three
  `2026_W04_ATL@NO` rows also say `capture_missed`: that reading ran and wrote no row, before the
  `capture_empty` reason existed.

Every row it touched is a week-3 or week-4 `pre_verdict` row, never counted in the verdict.

## Website push

Added 2026-10-18 (Phase 36). The daily run ends by publishing the dashboard to Vercel, so the
owner can read it on the phone while the laptop is off. One routine,
`pipeline.site_push.push_site`, does every push; the manual command
`uv run python -m scripts.push_site` calls the same routine. The owner's setup, the push log and
what each reason means are in `RUNBOOK.md`, "Website (hosted dashboard)".

### The switch comes first

Until `config/site_push_enabled.json` is committed, every run records one `not_pushed` /
`push_not_enabled` line in `logs/site_pushes.jsonl` and sends nothing: no secret is read, no lock
is taken, nothing is rebuilt or stamped, the Vercel CLI is not run, and no phone push is sent.
The owner commits that file as the last setup step. `git rm config/site_push_enabled.json` and a
commit pause every push, daily and manual, the same way.

### Where it runs, and which runs push

- **Inside `run_daily`** (`scripts/daily_lock_pipeline.py`): on the main path after the ledger
  sync (`_sync_ledger`) and immediately before the end-of-run notifier, whether the day returned
  or raised; and on the `lock_passed` branch immediately before that branch's own notify point.
  So the push has finished -- succeeded or failed -- before the phone pushes are decided, and a
  successful run's bets push never arrives before the site carries those bets.
- **Outcomes that push:** `offseason`, `no_games`, `lock_passed`, `success`,
  `finished_with_skips` and `degraded` -- quiet days included, so the push is a daily heartbeat.
- **Outcomes that do not push:** an orchestrator `failed` and a `run_failed` record. They write
  one `not_pushed` line whose reason is that status, with the run id, so a failed day can never
  refresh the site.
- **Never:** on `--dry-run`, on a refused `--date`, or on Ctrl+C -- no push and no record.

### One attempt, in order

1. The switch (above).
2. The secrets, `VERCEL_TOKEN` and `VERCEL_BYPASS_SECRET` from `.env`. A missing one is a
   failed push (`missing_token`, `missing_bypass_secret`) before the lock.
3. The publish lock, `logs/.cache_publish.lock`. The daily push waits up to 600 s for it; the
   manual push refuses at once (`lock_held`).
4. The refusals: this run's cache rebuild did not complete, or `data/web_cache.duckdb` is missing
   (`rebuild_incomplete`, `cache_missing`); the stylesheet is out of step with the templates or
   cannot be checked (`stylesheet_out_of_step`, `stylesheet_unverifiable`); the cache's
   data-format version does not match the code's. A format mismatch first HEALS itself (D-03):
   the push rebuilds the cache once, under the lock it already holds, and carries on when the
   rebuilt cache matches; it refuses only when that rebuild raises
   (`format_mismatch_rebuild_failed`) or still does not match (`format_mismatch_after_rebuild`).
5. A fresh publish id (uuid4) and push time (UTC), minted for this attempt and stamped into the
   cache's `cache_meta` table on a COPY of the cache, which is then swapped in.
6. The SHA-256 of the stamped file, then ONE production deploy of the code and the cache together.
   The deploy uploads the files on disk, so the record's `code_commit` names HEAD, ending in
   `+dirty` when an uploaded folder or root file has uncommitted changes. It is recorded, not
   refused: the push has exactly the refusals in step 4.
7. The read-back: up to 13 GETs of the production `/health`, 10 s apart, sending the bypass value
   only in the `x-vercel-protection-bypass` header and never following a redirect. The push counts
   as succeeded only when `/health` returns this attempt's publish id AND the SHA-256 of the file
   it serves equals the local digest. The CLI's exit code is never the evidence. A read-back that
   never matches is a failed push (`readback_failed`): no rollback and no second deploy. Once
   `/health` matches, the push GETs each of the five nav pages, `/fragments/games` and
   `/api/export/json` once, through the same client and header, judged by the live check's own
   route list (`scripts/fetch_hosted_routes.py`). `/health` uses a much smaller code path than the
   pages, so the first real push read back while `/`, `/bets` and `/fragments/games` answered 500;
   now any page that fails makes the push failed (`pages_failed`).
8. Exactly one record in `logs/site_pushes.jsonl` and one `SITE_PUSH= <outcome> <publish id |
   reason>` console line. Anything unexpected that raises after the switch is still recorded, as
   a failed push named by its exception class only.

### One lock for the push and every rebuild

The push and every web-cache rebuild hold the same publish lock: the daily run's
`populate_web_cache` step, the settle-day rebuild on a no-game day, and
`scripts/populate_cache.py` (which refuses with `POPULATE_CACHE= refused` and exit 1 while a push
holds it). So a push never uploads a cache that is being replaced underneath it, and a manual push
refuses while a rebuild runs.

### A failed push reaches the phone and changes nothing else

A failed or refused push sends one normal-priority (3) `FAILED: website push` through the existing
failed event class, after the run's other failed messages. It names the outcome and the reason,
both data-format versions on a format refusal, the exception class when the routine itself raised,
and the next step (`uv run python -m scripts.push_site`, or wait for the next 5 PM run). A
succeeded push and a `not_pushed` record send nothing. There is no third push kind: the router
still accepts only failed, bets and test, and a run that appended live bets still sends its bets
push too. The push never changes the run's exit code, its ledger rows or its saved outputs -- a
committed test runs the same day with the push succeeding, failing and raising and compares every
byte.

### The check-in and its timing

The healthchecks.io check-in is unchanged: it is still the last thing the run does. The push runs
before it, so it can delay the check-in by at most its own worst case: a 600 s lock wait, a 900 s
deploy bound, 13 read-back attempts (about 130 s), seven page checks (at most 105 s) and a 45 s
self-heal rebuild -- about 30 minutes.
The 2026-10-17 prediction day ran 17:00-17:14 ET, so even the worst case checks in around
17:45 ET, well before the watcher's 18:30 ET down time. A push that is switched off returns in
milliseconds.

### Under the S4U scheduled task

- The push runs the Vercel CLI by absolute path: `node.exe` and the CLI's `vc.js` under nvm Node
  24.18.0 (`NODE_EXE` and `VERCEL_CLI_JS` in `pipeline/site_push.py`). The S4U logon has no user
  PATH, so a bare `vercel` would fail with the same `0x80070002` "file not found" that a bare `uv`
  did (section 8).
- `VERCEL_TOKEN` is read from `.env` through `Settings` and handed to the CLI only in the child
  process's environment -- never on the command line, in the task XML, a log or the push log.
- The CLI runs with `--global-config` pointing at the gitignored `tools/vercel-global-config`, so
  the owner's interactive `vercel login` is never used.
- No task re-registration is needed: the existing `NFL_Predict_Pipeline` task runs the push as
  part of the daily run, and `deployment/windows_scheduler.xml` is unchanged.

---

## Cross-references

- **`AUDIT-REPORT.md`** -- the Phase 20 Data & Feature Correctness Audit handoff: the
  adopted canonical gold (D-02), the FIX-01 fixes, and the deferred findings record this
  document extends.
- **`PIPELINE.md`** -- the canonical 7-stage run sequence
  (ingest -> features -> train -> backtest -> predict -> build-cache -> serve) that the
  Friday orchestrator wraps for the current-week subset. The orchestrator does the
  ingest -> features -> predict -> validate work; train, backtest, and build-cache are
  the separate manual stages.
- **Phase 23 (forthcoming)** -- the operational runbook + state-of-system summary will
  CITE this file. AUTOMATION.md is the EXPLANATION; the runbook is the operational
  procedure. This file does not duplicate runbook scope.

---

*Phase 21 -- Automation Audit & Explanation. AUTO-04 end-to-end automation explanation +
the D-12 FIX-vs-deferred findings record. ASCII only (no emoji, per CLAUDE.md).*
