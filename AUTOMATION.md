# AUTOMATION.md -- Friday Automation Explanation (Phase 21)

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

The Friday automation is a single Windows Task Scheduler task, `NFL_Predict_Pipeline`,
that fires once a week.

- **Trigger:** Friday `18:00` LOCAL = `6 PM ET`. The `StartBoundary` in
  `deployment/windows_scheduler.xml` carries NO timezone offset
  (`2026-09-11T18:00:00`), so `schtasks` interprets it in **machine-LOCAL time**.
  `18:00` local equals `6 PM ET` **only while the machine stays on Eastern Time**
  (the D-09 caveat). If the machine is moved to another timezone, the 6 PM ET freeze
  is no longer honored and the trigger must be re-pointed.
- **Canonical command:** `uv run python scripts/friday_pipeline.py --log-level INFO`
  (the `<Exec>` in the XML: `<Command>` = the absolute `uv.exe` path +
  `<Arguments>run python scripts/friday_pipeline.py --log-level INFO</Arguments>`).
- **Principal:** the owner user `jackc` with `LogonType=S4U` and
  `RunLevel=HighestAvailable` (NOT the SYSTEM SID `S-1-5-18`, which cannot see the
  owner's `uv` / `.venv` / `.env`). S4U runs whether or not the owner is interactively
  logged on, which the `WakeToRun` Friday-evening run needs. `LogonType=S4U` is the
  CONFIRMED working principal (D-08 register-and-verify, Section 9). Because the S4U
  logon does NOT load the owner's USER PATH, the `<Command>` must be the ABSOLUTE
  `uv.exe` path (`C:\Users\jackc\AppData\Roaming\Python\Python313\Scripts\uv.exe`) -- a
  bare `uv` fails under S4U with `0x80070002` "file not found". The remedy was the
  absolute path, not switching off S4U.
- **WorkingDirectory:** `C:\Users\jackc\Code\nfl-predict`. The committed XML is
  machine-specific (this path + the `UserId` principal) and must be re-pointed if the
  repo moves to a different machine or user.
- **Robustness settings (XML):** `MultipleInstancesPolicy=IgnoreNew`,
  `StartWhenAvailable=true`, `RunOnlyIfNetworkAvailable=true`, `WakeToRun=true`,
  `ExecutionTimeLimit=PT2H`, `AllowHardTerminate=true`, `AllowStartOnDemand=true`,
  `Priority=7`. The XML is UTF-16 encoded.
- **Installer (single source of truth, D-07):** the task is registered by
  `uv run python deployment/setup_scheduling.py --install`, which runs
  `schtasks /create /tn NFL_Predict_Pipeline /xml deployment/windows_scheduler.xml /f`
  (idempotent overwrite) and cleans up the old split tasks (`NFL_Predict_DataUpdate`,
  `NFL_Predict_Predictions`). The installer installs the committed XML so the installer
  and the definition can never drift; there is no second scheduling mechanism (no
  PowerShell `Register-ScheduledTask`, no cron).

### CLI modes

`scripts/friday_pipeline.py` accepts (the phase flags are mutually exclusive):

| Flag | Effect |
|------|--------|
| (none) | Full pipeline: all 22 steps (DATA + PREDICTIONS). |
| `--data-only` | DATA phase only (the 9 DATA-phase steps). |
| `--predictions-only` | PREDICTIONS phase only (the 13 PREDICTIONS-phase steps); logs a "ensure data artifacts are fresh" warning. |
| `--dry-run` | List the steps that would execute (filtered by mode); execute nothing; exit 0. Works year-round: `--dry-run` bypasses the offseason no-op short-circuit so steps can be inspected out of season without `--force` (WR-04). |
| `--force` | Bypass the pre-flight staleness/season checks AND the offseason no-op short-circuit; pre-flight health becomes advisory. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity (default INFO). |

> Note: the scheduled task runs the FULL pipeline (no phase flag). The
> `--predictions-only` / `--data-only` / `--dry-run` / `--force` modes are operator
> tools for manual / out-of-season runs.

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
- **C. Step execution:** the 22-step registry (`pipeline/steps.py`), phase-filtered by
  mode, run in order. The log is written atomically after each step (incremental
  snapshot). A **critical** step failure sets `status="failed"`, fires
  `alert_pipeline_failure` (CRITICAL) and raises immediately. A **non-critical** step
  failure appends a warning and continues; the run ends `degraded`.
- **D. Post-run health check** (6 checks; ADVISORY ONLY): never changes `status` or the
  exit code. An `unhealthy` post-run result only appends `"Post-run health check:
  unhealthy"` to warnings.
- **E. Completion alerting** (exactly ONE alert per outcome): `failed` ->
  `alert_pipeline_failure` (CRITICAL); `degraded` -> `alert_degraded_completion`
  (WARNING); `success` -> `alert_pipeline_success` (INFO).

### Offseason no-op short-circuit (D-06)

A live, scheduled, **unforced** Friday run during the NFL offseason would otherwise hit
the staleness season gate, set `status="failed"`, fire a CRITICAL "Pipeline Failed"
alert, and raise -- a false alarm that erodes trust (crying wolf). To prevent this,
`scripts/friday_pipeline.py::main` detects the offseason and returns `0` as a clean INFO
no-op **BEFORE** constructing `FridayPipeline`, so no orchestrator alert path is reached.
The offseason window mirrors `StalenessGate.check_season` exactly
(`[season_start, season_start + 22 weeks]`, resolved via `get_current_nfl_week()` in ET),
so the CLI short-circuit and the season gate can never disagree. `--force` deliberately
bypasses the short-circuit so the operator can still run out of season (e.g. a one-time
forced run against a completed-week stand-in). A live unforced offseason run therefore
exits 0 as a no-op with **no CRITICAL alert**.

---

## 3. The 22 orchestrator steps

The registry (`pipeline.steps.build_step_registry`) is exactly 22 `StepDefinition`
entries: 9 in the DATA phase, 13 in the PREDICTIONS phase. Each step uses deferred
imports (inside the function body) to avoid argparse collisions and module-level side
effects. `critical=True` means a failure aborts the run; `retryable=True` means transient
errors trigger retry.

| # | Step | Phase | Critical | Retryable | What it does |
|---|------|-------|----------|-----------|--------------|
| 1 | `capture_live_season` | DATA | yes | yes (3) | Capture what nflverse is serving RIGHT NOW into the append-only live zone, BEFORE anything consumes it. Calls `run_capture` directly with the run's resolved (season, week). |
| 2 | `ingest_games` | DATA | yes | yes (3) | Ingest current-week games via nflreadpy. |
| 3 | `ingest_weather` | DATA | no | yes (3) | Ingest weather forecasts via Open-Meteo (async). |
| 4 | `data_qa` | DATA | yes | no | Run data-quality validation; fail if checks failed. |
| 5 | `build_elo` | DATA | yes | no | Update Elo ratings for the current season. |
| 6 | `build_team_form` | DATA | yes | no | Build team-form metrics for the current week. |
| 7 | `build_contextual` | DATA | yes | no | Build contextual features (travel, rest, venue). |
| 8 | `build_weather_features` | DATA | no | no | Build weather-based features for outdoor games. |
| 9 | `verify_data_artifacts` | DATA | yes | no | Verify the DATA-boundary silver artifacts exist AND carry a row for the current (season, week). Gold is NOT checked here -- it is built in the PREDICTIONS phase, so a currency check here would report an ordering fact as a stale artifact. |
| 10 | `ingest_odds` | PREDICTIONS | yes | yes (3) | Capture the odds snapshot from The Odds API. |
| 11 | `build_market_anchors` | PREDICTIONS | yes | no | Build market-anchor features from the odds snapshot. |
| 12 | `build_features` | PREDICTIONS | yes | no | Assemble the unified per-target gold feature matrices. |
| 13 | `validate_features` | PREDICTIONS | yes | no | Validate features for data leakage / quality. |
| 14 | `verify_gold_currency` | PREDICTIONS | yes | no | Verify the three gold matrices carry a row for the current (season, week). R9's "gold has no rows for this week" refusal, at the first point in the run where gold exists. |
| 15 | `validate_models` | PREDICTIONS | yes | no | Validate WP/ATS/OU models are available + loadable (via `artifacts/latest.json`). |
| 16 | `generate_predictions` | PREDICTIONS | yes | no | Generate current-week predictions (loads artifacts, applies market blend). |
| 17 | `verify_prediction_currency` | PREDICTIONS | yes | no | Verify the prediction file's ROWS are the current week, not only its filename. Runs before anything consumes it. |
| 18 | `generate_recommendations` | PREDICTIONS | yes | no | Select the week's +EV bet list through `BetSelector` and write the durable bet-list artifacts. |
| 19 | `export_artifacts` | PREDICTIONS | yes | no | Export the predictions CSV to JSON. |
| 20 | `validate_predictions` | PREDICTIONS | yes | no | Validate the prediction file (non-empty, required columns, `wp_prob` in [0,1]). |
| 21 | `verify_output_files` | PREDICTIONS | no | no | Verify the expected output files exist (advisory; warns on missing). |
| 22 | `populate_web_cache` | PREDICTIONS | no | no | Rebuild `data/web_cache.duckdb` so the served bet list is this run's. |

> Note: retry is handled by `tenacity` (`Retrying` with `wait_exponential` backoff) and
> fires ONLY on `TRANSIENT_EXCEPTIONS` (`ConnectionError`, `TimeoutError`, `OSError`,
> and `httpx.HTTPStatusError` when httpx is available). Local / validation errors
> (`ValueError`, `RuntimeError`, etc.) are NOT retried -- they fail fast. Only the four
> network-touching steps (`capture_live_season`, `ingest_games`, `ingest_weather`,
> `ingest_odds`) are retryable.

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

`populate_web_cache` is step **22**, the LAST entry in the registry. It rebuilds
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
  `alert_degraded_completion`. Note honestly what that buys: alerts are log-only by default and
  the email/Slack channels are inert (section 6), so the real protection against a silently stale
  bet list is the `/bets` **hard-block** -- the page refuses to serve a week whose bet-list
  populated-at marker predates that week's latest per-game line freeze. A failed run leaves that
  marker unadvanced, which is what makes the block fire deterministically.

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
| failed | `failed` | `alert_pipeline_failure` (CRITICAL) | 1 | may be absent |

Stated plainly:

> **"Friday succeeded"** == `log.status` in `{success, degraded}` AND
> `outputs/predictions/predictions_<S>_week<W>.csv` exists (non-empty, `wp_prob` in
> `[0,1]`).

The exit code is `0` for BOTH `success` and `degraded` (a degraded run still produced
predictions; only a non-critical step like weather or output-file verification failed),
and `1` only for `failed`. The AUTO-01 keystone test
(`test_orchestrator_predictions_phase_e2e` in
`tests/integration/test_friday_prediction_step.py`) drives the REAL
`FridayPipeline(mode="predictions-only", force=True).run()` through the REAL step
registry (no stubbed registry -- that stubbing was the `cb61042` blind spot) and asserts
`log.status == "success"`, the predictions CSV exists, `game_id` is present, and
`wp_prob` is in `[0,1]` -- proving the triad's `success` branch produces a real,
leakage-safe predictions file. The registry itself is unstubbed, but three of the ten
PREDICTIONS-phase step BODIES are no-op'd (`step_ingest_odds`, `step_build_market_anchors`,
`step_build_features`) so the committed gold supplies their inputs; the remaining
generate/validate/export/verify steps run for real (matching the test's docstring). An owner-confirmed one-time live forced run produced a
real predictions file offline (16 rows, `wp_prob` 0.176-0.867, all market rows matched
from the on-disk odds snapshot with NO live pull) with NO model artifact re-fit.

---

## 7. Alerts -- console/log ONLY in the current code (D-05, empirically verified)

Alerts are **log-only.** Every alert always fires to the console/log channel
(`_send_console_alert` + the `logger.{info,warning,error,critical}` line in
`AlertManager.create_alert`), so a normal run produces its alert with NO configuration.
**Email and Slack are currently INERT** -- they cannot fire at all via `conf/config.yaml`
today. This is a code-wiring gap, NOT a missing-secret problem: even with every key set,
no email or Slack message would be sent. The D-05 verification (empirical, read-only) is
recorded in Section 9; the alert-wiring gap is captured (NOT fixed) in Section 10 with the
3-part remedy.

The orchestrator's four alert methods (`pipeline/alert.py`) each map to exactly ONE event
and level: failure -> CRITICAL, success -> INFO, staleness -> WARNING, degraded ->
WARNING. They wrap `utils.alert_manager.AlertManager`, whose `send_alert` is supposed to
route each alert to the channels configured for its severity, gated on the enable flags.

### Why email/Slack are inert (the verified root cause)

A `conf/config.yaml` `monitoring.notifications` block (recipients, webhook URL, per-level
channel lists) exists in spirit, but the running code never reads it. Three independent
breaks, each sufficient on its own, keep email/Slack dark:

1. **`AlertManager` reads a non-existent attribute.** `AlertManager.__init__` sets
   `self.monitoring_config` from `self.settings.monitoring` -- but `conf.settings.Settings`
   has NO `monitoring` attribute (the YAML monitoring block lives under
   `self.settings.config.monitoring`). So `hasattr(self.settings, "monitoring")` is `False`
   and `self.monitoring_config == {}` ALWAYS. `send_alert` then resolves
   `notification_levels.get(<level>, ["console"])` to `["console"]` for EVERY level --
   CRITICAL and INFO alike route to `["console"]` only, regardless of any config.
2. **`MonitoringConfig` does not declare `notifications`.** Even via the correct path
   (`self.settings.config.monitoring`), `conf.settings.MonitoringConfig` declares only
   `enable_health_checks`, `health_check_interval`, `data_quality`, and
   `model_performance` -- there is NO `notifications` field. Pydantic (extra=ignore) DROPS
   `config.yaml`'s `monitoring.notifications` block, so `model_dump()` would never expose
   `enable_email` / `enable_slack` / `email_recipients` / `slack_webhook_url` /
   `notification_levels` even if break (1) were fixed.
3. **`Settings` lacks the SMTP fields.** The email SEND path
   (`AlertManager._send_email_alert`) references `self.settings.email_from`,
   `self.settings.smtp_host`, `self.settings.smtp_port`, `self.settings.smtp_use_tls`,
   `self.settings.smtp_username`, and `self.settings.smtp_password` -- NONE of which are
   defined on `Settings`. If the email channel were ever reached it would raise
   `AttributeError`. (The `.env` flags `ENABLE_EMAIL_REPORTS` / `ENABLE_SLACK_NOTIFICATIONS`
   that DO exist on `Settings` are unrelated -- `AlertManager` never reads them.)

Net: the console/log alert (always fires) is the real, working behavior. **Both email AND
Slack are currently inert** -- enabling them is NOT possible by configuration alone; it
requires the code changes listed in Section 10. The rate-limit / cooldown logic in
`should_send_alert` (`cooldown_minutes` default 30, `max_alerts_per_hour` default 20) reads
the same empty `monitoring_config`, so it falls back to those defaults.

> Note (D-05 / honesty): an earlier draft of this section implied "Slack works as
> documented; email needs SMTP settings." That was inaccurate -- in the current code BOTH
> email and Slack are inert. The honest statement is: alerts are console/log ONLY, and
> enabling email/Slack requires the deferred code changes in Section 10. The default is
> log-only and a normal run requires no configuration.

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

### D-05 (alert path) -- documented-only; email + Slack are inert [PASS]

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
- **STEP10-UNCONSUMED-SILVER -- step 10 writes a silver table no downstream step reads.**
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
