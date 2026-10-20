# ROLLING RE-FIT PRE-REGISTRATION -- the 2026 frozen-recipe re-fit and the frozen-arm comparison

**Status:** FROZEN. This document and `backtest/refit_preregistration_2026.py` are ONE
pre-registration in two files, landed in ONE commit containing nothing else. This is the
human-readable half; the constants module is the half the code reads. Every value written in both
places is the same value.

**What this document is for.** From the first re-fit date to the Super Bowl, the live arm's three
models (WP, ATS, O/U) are re-fitted every 4 NFL weeks by one fixed recipe with zero discretion, and
go live when pre-registered integrity checks pass. Beside them, a frozen shadow arm keeps today's
row-19 models and bet rule. At season close one pre-registered comparison answers whether
re-fitting beat standing still. That answer is only worth anything if every rule that governs a
number this phase produces -- which days re-fitting happens, how the models are trained, how the
bet-rule numbers are re-measured, what a new model must pass, and how "did it help?" is judged --
was fixed in committed source BEFORE the first number it governs existed. This document exists so
that fact is checkable by a reader, not merely asserted by an author (RFIT-01, D-03, D-23).

**Editing after the commit does not fix a bug -- it destroys the evidence.** A value that is wrong
here is wrong for the whole 2026 re-fit. The only legitimate response is a NEW, VISIBLY LATER
CORRECTIVE COMMIT whose message explicitly INVALIDATES this pre-registration by naming its commit
sha. Never an edit in place, and never a quiet supersession. A later commit that touches either
file without naming this one fails a test.

**The anchor is recorded outside the files it witnesses.** Neither this document nor
`backtest/refit_preregistration_2026.py` records its own content hash or its own commit. A file
that must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential: writing the
hash changes the bytes the hash was computed over, so no fixed point exists. The witness lands in
LATER commits, in a test-state module and in a production module the re-fit reads before training.

```
preregistration_anchor_recorded_in: tests/phase37_state.py, refit/witness.py
preregistration_anchor_slots: P37_PREREGISTRATION_WITNESS, PREREGISTRATION_SHA256, PREREGISTRATION_COMMITTED_AT
preregistration_paths: ROLLING-REFIT-PREREGISTRATION.md, backtest/refit_preregistration_2026.py
preregistration_deadline: 2026-10-28T17:00:00-04:00
```

The deadline is week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (21:00 UTC).
This commit's committer date must precede it, and the commit must be on GitHub's `master` before
it, or week 8 cannot be the A/A week (D-03).

---

## 1. What is frozen, and what is not

**The RECIPE is frozen, not the weights (D40-06).** Each re-fit produces new model weights, newly
picked inputs and newly measured bet-rule numbers. What cannot move is the procedure that produces
them: the calendar, the training scope, the input pool and caps, the fixed settings, the
calibration rule, the re-measure procedure, the integrity checks and the comparison.

**Out of scope, all season:**

- No hyperparameter search of any kind: no Optuna study, no tuning, no settings change (RFIT-07).
- The EV floors are fixed at 0.0 / 0.0 / 0.01 (WP / ATS / O/U) and are never re-picked.
- No new signal family, and the excluded groups (injury, situational, snap) are never re-opened.
- No gate baseline is re-frozen. `config/gate.toml`, `models/deploy_gate.py` and
  `scripts/promote_models.py` stay byte-unchanged, and their two tripwires stay red.
- The Phase 34 verdict declaration (`backtest/verdict_scope_2026.py`: weeks 5-22 count, the live
  arm counts, the shadow arm is never pooled) is unchanged.
- `LATEST_COMPLETED_SEASON` and the backtest season partition (`conf/season_partition.py`) are
  unchanged; the re-fit uses its own cutoff-relative rule below.
- No shadow row is ever back-filled for a week before the shadow start.

## 2. The two arms (RFIT-03, RFIT-04, R6, R8, D-38)

**The live arm** decides with the generation in force: whatever `artifacts/latest.json` names at
the moment of the decision. Generation 0 is the row-19 models; each successful re-fit installs the
next generation.

**The frozen (shadow) arm** is pinned to the row-19 models and bet rule for the whole season, by
the committed pin `backtest/shadow_pin_2026.py`:

| Pinned item | Value |
|---|---|
| WP model | `wp_20261004_050223` |
| ATS model | `ats_20261004_050228` |
| O/U model | `ou_20261004_050232` |
| Blend | `blend_20261004_050521` |
| Converter the blend is bound to | `market_probability_20260923_195443` |
| Recipe | `recipe-2026-row19-v1` |
| Chain-fit run record | `outputs/row19/neutral_hfa_chain_fit.json` |
| Its sha256 | `fab1118b8093bf55fd1cd4d7d8ba80eff3f95b7bd9537c4cf4f3a6b2a6ce2906` |

Nothing writes the pin. The shadow arm never reads or writes `artifacts/latest.json`: it names its
models only through the pin.

**Same process, same inputs.** Both arms are decided in the same daily run, from the same candidate
frame, at the same decision time, with the same reproduction key. The shadow arm re-scores the live
decision's own candidates and gold rows with the pinned models; it never builds a second frame.
Both arms' new rows land in ONE atomic ledger write, live rows first, then shadow rows. A shadow row
is written only beside a NEW live row of the same game and target.

**A shadow failure never costs a live row.** If the shadow arm fails, the live rows are written
exactly as they would have been without it, and every live row left without a twin is recorded as
a hole naming the exception class only. When building the twins would leave less than 120 seconds
(`forward_ledger.shadow.SHADOW_COMMIT_HEADROOM`) before the slate's publish deadline, the twins are
dropped and recorded as holes (`ShadowDeadlineHeadroomError`), so the live write always goes ahead.
A hole is sent as a FAILED push. A comparison uses paired games only; a week with no shadow rows at
all is recorded and excluded.

**Week-7 shadow rows (D-38).** The frozen arm went live on master on 2026-10-20, ahead of week 7's
decision runs (the first is Wed 2026-10-21 17:00 ET; D-38 required it by the Sat 2026-10-24 run),
so week-7 shadow rows exist as a production dress rehearsal. They stay in the ledger permanently.
They are generation 0, never tested, never counted, and never shown as bets.

## 3. Generations (D-19, D-20, RFIT-08, R11)

**One recipe id for every re-fit generation: `recipe-2026-refit-v1`.** Each generation's own
numbers (its model ids and its re-measured bet rule) are named through the generation records under
`ledger/refit/`, never by a new recipe id. Every row decided by the pinned row-19 models -- every
shadow row, and every live row from week 5 until the first re-fit installs -- carries `recipe-2026-row19-v1`;
every live row decided by a re-fit generation carries `recipe-2026-refit-v1`.

**A row's generation is DERIVED from its own stamped model id; no ledger column is added.** The
immutable column set (`IMMUTABLE_COLUMNS_V1`, `canon_v` 1) and every existing chain hash stay
untouched.

- The pinned row-19 model ids are generation 0. Every shadow row is generation 0.
- Generation N is the ids named by the last install intent of the N-th cadence date, in date order,
  whose last install intent counts. It counts when the date closed ran or ran late, or when the date
  is still open and no refused attempt follows that intent. An installed or a crashed attempt after
  it, or no attempt at all, does not stop it counting, because a crash can come after the swap wrote
  `latest.json`; so the ids `latest.json` serves always name exactly one generation.
- A date closed not run consumes no number.
- A row with NULL stamps (weeks 3-4, decided before the row-19 models) has no generation and stays
  out of the comparison.
- A row decided before a mid-week catch-up keeps the generation that decided it (first pick
  stands). No existing row is ever rewritten.

## 4. The calendar (D-01, D-02, D-05, D-06, D-33)

**The formula.** From Tue 2026-11-03 onward, each cadence Tuesday (the last Tuesday strictly
before an NFL week's first game) evaluates whether the NFL week that just finished was a clean A/A
week (section 5). The first Tuesday that finds one is re-fit date 1. Every later date is the
cadence Tuesday of the NFL week 4 weeks on, up to and including week 22. Only weeks 8 through 16 are
ever evaluated; week 7 cannot qualify.

**A/A evaluation Tuesdays** (each is the cadence Tuesday of the following NFL week):

| A/A week | Evaluated on |
|---|---|
| week 8 | Tue 2026-11-03 |
| week 9 | Tue 2026-11-10 |
| week 10 | Tue 2026-11-17 |
| week 11 | Tue 2026-11-24 |
| week 12 | Tue 2026-12-01 |
| week 13 | Tue 2026-12-08 |
| week 14 | Tue 2026-12-15 |
| week 15 | Tue 2026-12-22 |
| week 16 | Tue 2026-12-29 |

**The nine date lists.** The first clean A/A week selects one list, and that list is in force for
the rest of the season. Each date is shown with the NFL week whose first game it precedes.

| First clean A/A week | Re-fit dates (cadence Tuesdays) |
|---|---|
| 8 | 2026-11-03 (week 9), 2026-12-01 (week 13), 2026-12-29 (week 17), 2027-01-26 (week 21) |
| 9 | 2026-11-10 (week 10), 2026-12-08 (week 14), 2027-01-05 (week 18), 2027-02-09 (week 22) |
| 10 | 2026-11-17 (week 11), 2026-12-15 (week 15), 2027-01-12 (week 19) |
| 11 | 2026-11-24 (week 12), 2026-12-22 (week 16), 2027-01-19 (week 20) |
| 12 | 2026-12-01 (week 13), 2026-12-29 (week 17), 2027-01-26 (week 21) |
| 13 | 2026-12-08 (week 14), 2027-01-05 (week 18), 2027-02-09 (week 22) |
| 14 | 2026-12-15 (week 15), 2027-01-12 (week 19) |
| 15 | 2026-12-22 (week 16), 2027-01-19 (week 20) |
| 16 | 2026-12-29 (week 17), 2027-01-26 (week 21) |

**The cutoff.** Each date's scheduled cutoff is 17:00 America/New_York on that Tuesday (D-05). DST
ends Sun 2026-11-01, so every listed cutoff is 22:00 UTC. Every instant goes through the
`America/New_York` zone, never a fixed offset. A catch-up or a retry uses the same cutoff.

**The evidence the dates were derived from.** Weeks 8-18: each week's first kickoff in the recorded
2026 schedule (`data/silver/games.parquet`, read 2026-10-20). Weeks 19-22: the published
2026-season playoff calendar (Wild Card, Divisional, Conference Championships, Super Bowl LXI after
a bye week); those rounds are absent from the recorded schedule until each is set.

| NFL week | First game (UTC, or ET date for playoffs) | First game, ET | Cadence Tuesday |
|---|---|---|---|
| week 8 | 2026-10-30T00:15:00+00:00 | Thu Oct 29 20:15 EDT | (the A/A week; no re-fit date) |
| week 9 | 2026-11-06T01:15:00+00:00 | Thu Nov 5 20:15 EST | Tue 2026-11-03 |
| week 10 | 2026-11-13T01:15:00+00:00 | Thu Nov 12 20:15 EST | Tue 2026-11-10 |
| week 11 | 2026-11-20T01:15:00+00:00 | Thu Nov 19 20:15 EST | Tue 2026-11-17 |
| week 12 | 2026-11-26T01:00:00+00:00 | Wed Nov 25 20:00 EST | Tue 2026-11-24 |
| week 13 | 2026-12-04T01:15:00+00:00 | Thu Dec 3 20:15 EST | Tue 2026-12-01 |
| week 14 | 2026-12-11T01:15:00+00:00 | Thu Dec 10 20:15 EST | Tue 2026-12-08 |
| week 15 | 2026-12-18T01:15:00+00:00 | Thu Dec 17 20:15 EST | Tue 2026-12-15 |
| week 16 | 2026-12-25T01:15:00+00:00 | Thu Dec 24 20:15 EST | Tue 2026-12-22 |
| week 17 | 2027-01-01T01:15:00+00:00 | Thu Dec 31 20:15 EST | Tue 2026-12-29 |
| week 18 | 2027-01-10T18:00:00+00:00 | Sun Jan 10 13:00 EST (recorded placeholder) | Tue 2027-01-05 |
| week 19 | 2027-01-16 | Sat Jan 16 (Wild Card) | Tue 2027-01-12 |
| week 20 | 2027-01-23 | Sat Jan 23 (Divisional) | Tue 2027-01-19 |
| week 21 | 2027-01-31 | Sun Jan 31 (Conference Championships) | Tue 2027-01-26 |
| week 22 | 2027-02-14 | Sun Feb 14 (Super Bowl LXI) | Tue 2027-02-09 |

**Dates are never derived at run time.** The lists above are literals. A game moved later never
moves a date; it only changes which games precede a cutoff.

**A list keeps its last date (D-06)** even when that period holds only a few playoff games. That
re-fit runs, and its period is reported as too few to test (section 11).

**The end rule (D-33).** If no week 8-16 is clean, no re-fit runs in 2026, and the season close
says so plainly. As the constants module states it:

no week 8-16 clean: no re-fit runs in 2026, and the season close says so

## 5. The A/A rule (D-02, D-03, R7)

**A clean A/A week** is a completed NFL week, evaluated on its evaluation Tuesday (section 4), in
which every live row has a shadow twin -- the same `(game_id, season, week, target)` -- identical in
every immutable column, in the envelope `kind` and `canon_v`, in the grading half and in the
closing half, outside this exclusion list:

`arm`, envelope `seq`, envelope `chain_hash`, `fill_sportsbook`, `fill_line`, `fill_odds`,
`fill_stake_dollars`, `fill_at_utc`.

Nothing else may differ. The exclusion list is never widened to make a week pass; a mismatch is
fixed in the shadow code.

**A week is COMPLETE only when every game of that NFL week in the recorded schedule kicked off
before its evaluation Tuesday's 17:00 ET cutoff.** A week with a game at or after that instant (a
postponement) is recorded not clean with reason `week_not_complete`, and the schedule waits exactly
as it does for a mismatch, so no row can join a week after its result is binding.

**Further rules:**

- A week with zero live rows does not qualify.
- Runs missed in both arms (the laptop off: no rows in either arm) do not void a week.
- The pre-registration commit's committer date must precede the week's first decision run (its
  first lock, 18:00 ET on the day before its first game, minus one hour). Week 8's is Wed
  2026-10-28 17:00 ET.
- A mismatch records the first mismatched game, target and field, sends one FAILED push, and the
  schedule waits for the next clean week (it does not refuse a re-fit; no re-fit is due yet).
- Week 7 is never evaluated.
- After the first re-fit installs, the A/A check is never applied again.

The A/A rule reads only an engineering fact -- did both arms agree for a whole week -- never any
result.

## 6. When a re-fit runs (RFIT-02, D-15, D-17, R2, R9, R10)

- **The trigger reads only** the date lists, the run clock, the per-date records and the A/A
  records. It reads no metric, no drift figure, no CLV and no comparison output.
- **At most one re-fit per daily run.** A date with zero newly finished games still re-fits.
- **Where in the daily run:** after the day's ledger write (live and shadow rows), before the cache
  rebuild, the website push, the notifications and the check-in, so that day's push and email report
  the outcome. New models serve from the next daily run.
- **Which runs:** a due re-fit is attempted on a run that ends `success`, `finished_with_skips`,
  `degraded`, `no_games` or `lock_passed`; never on a dry run, a refused run date or Ctrl+C. A
  `lock_passed` run refreshes the schedule before a due attempt. A run that crashes before the
  re-fit point does not attempt it.
- **Due:** a date is due at a daily run when its cutoff has passed, it has no successful attempt,
  and the next date's cutoff has not passed.
- **On time vs late:** a date that succeeds on the daily run of its own ET day ran on time;
  otherwise it ran late, recorded with both times.
- **Retry:** a failed or refused date is retried at every following daily run with the identical
  cutoff, until one attempt succeeds or the next date arrives. When the next date arrives, the older
  unfinished date closes as not run and only the newer one is attempted.
- **A date with a successful attempt is never attempted again.**
- **Never edit to get through:** the recipe, the integrity checks and the bet-rule numbers are
  never edited to get a re-fit through. A genuine code or data fix between attempts is allowed and
  shows as the attempt's recorded code commit.
- **One attempt lock per ledger:** two concurrent invocations cannot both attempt the same date.
- **One durable, append-only record per date:** its status (`ran`, `ran late`, `refused`, `failed`,
  or not run before the next date), its scheduled cutoff, every attempt (time, code commit, reason
  or exception class) and the before and after ids.

## 7. Data preparation (D-37)

Every attempt first runs the full nightly build: Elo with provisional rows only for the day's slate,
team form, weather features, the full-history gold build leaving out every unplayed game that locks
after the run date's 18:00 ET lock, and the leakage scan. Training labels come from silver scores
by the gold label formula (WP: home score greater than away score; ATS: home minus away; O/U: home
plus away), asserted equal to gold's label wherever gold has one. No provisional Elo snapshot may
back a training row.

## 8. Training (R3, D-26, RFIT-07)

**Eligibility.** A game is a training row when it kicked off strictly before the cutoff and has a
final score, from season 2002 through the cutoff (2026 games included). A game kicking off exactly
at the cutoff is excluded. Every excluded game is counted with its reason, and each target's row
count must equal the finished games from 2002 to the cutoff minus the recorded exclusions.

**Eligibility is read at the date's FIRST attempt.** A game that kicked off before the cutoff
without a final score at that attempt stays excluded and counted at every later attempt of the same
date (a retry, or a catch-up after a refusal), so a result arriving between attempts never changes
that date's training rows.

**Row order.** Rows are sorted by (season, week, game_id) before anything else.

**The input pool.** Gold minus the standing excluded groups (`injury`, `situational`, `snap`, the
ratified verdict in `config/group_gate_verdict.toml`, normalized sha256
`06e147083ad2d11026a45d2587b75851a95ff2306465b997e13ebee8dc949112`) and the baseline excludes it
names. Inputs are re-picked on every training row by the trainers' own selection rule, within the
caps.

| Target | Input cap |
|---|---|
| WP | 20 |
| ATS | 25 |
| O/U | 25 |

**The fixed settings** (each trainer's own defaults, unchanged all season):

| Target | Model | Settings |
|---|---|---|
| WP | logistic regression | max_iter 1000, solver lbfgs, C 1.0, random_state 42 |
| ATS | XGBoost regressor | n_estimators 200, max_depth 4, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 1.0, random_state 42, verbosity 0, n_jobs -1 |
| O/U | XGBoost regressor | n_estimators 200, max_depth 4, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 1.0, random_state 42, verbosity 0, n_jobs -1 |

**Thread count 1.** Every fit runs with the thread count in force pinned to 1, and the count is
recorded; a run at any other count is refused.

**No search of any kind.** No Optuna study, no tuning call, no settings change.

**The calibration rule (D-26).** The calibration season is the latest season wholly before the
cutoff: 2025 for every 2026 cutoff. A helper model is trained by this same section on every season
from 2002 through the season before the calibration season (2002-2024 for a 2026 cutoff), with its
inputs picked on the helper's own training window, and the calibrator is fitted on the helper's
predictions of the calibration season only. The WP calibrator is Platt scaling with an isotonic
fallback; the ATS and O/U residual converter (a normal distribution over residuals) is fitted by
the same rule. The final model is then fitted, with inputs re-picked, on every training row before
the cutoff. Cutoff-relative form, for the re-measure folds: a fold predicting season S calibrates
on S-1 with a helper trained on 2002 through S-2.

Owner note (D-26): this calibration rule differs from the row-19 models' (they were calibrated on
2023 with inputs picked on 2002-2022), so a re-fit generation changes two things at once versus the
frozen arm: the data it learned from, and the season its calibration was fitted on. The owner chose
this knowingly. The frozen arm is unaffected.

## 9. The re-measure of the bet rule and the blend (R4, D-22, D-27, D-28, D-29, D-30, D-36)

Each generation re-measures the numbers that describe its models -- error spread, lean, edge bands,
WP calibration verdict and blend weights -- by the frozen row-19 procedure, changing only what this
record changes. The EV floors stay fixed.

**Folds.** Each historical season S of 2017-2024 is predicted by a model built by section 8 with
cutoff = S's first kickoff, one season per fold. Each finished 2026 week before the generation's
cutoff is predicted by a model built by section 8 with cutoff = that week's first kickoff, one week
per fold. Every prediction comes from a model version that never saw the game. Nothing at or after
the generation's cutoff enters any measurement.

**The corpus.** The 2020-2024 games with an owned pre-lock line in `odds_timeline`, plus the
finished 2026 games with an owned pre-lock line in silver `odds_snapshot`; in both, the line used is
the latest line timed at or before that game's own lock. A game with no pre-lock line is excluded
and counted. A 2020 WP row has no prior-fold converter, so WP's historical corpus is 2021-2024, and
the 2020 WP rows are excluded and counted, as today.

**2025 never enters a re-measured number.** It is the spent single-use hold and has no free
pre-lock source. (2025 games are training rows in section 8, and 2025 is the calibration season,
which reads results only and never a line.)

**The residual pool.** Every game of the seed seasons 2017-2019 plus every corpus game of 2020-2024
and of 2026 -- today's pool, extended. Owner ruling A (2026-10-20, D-27) states the population
exactly; outside the seed seasons, only games with an owned pre-lock line enter, and a game without
one stays out of the lean and the error spread exactly as it does today:

lean pool: every game of the seed seasons 2017-2019, plus only the 2020-2024 and finished 2026 games that have an owned pre-lock line (the corpus); the error spread uses the same 2020-2024 and 2026 corpus games; a 2020-2024 or 2026 game without an owned pre-lock line enters neither (owner ruling 2026-10-20, D-27)

**The lean (D-27)** is the pooled mean residual (actual minus prediction) of that whole pool, over
games kicked off before the generation's cutoff.

**The error spread (D-28)** is the frozen estimator (the sample standard deviation, ddof 1) over the
bias-corrected tune blocks 2020-2024 and 2026, each block's residuals corrected by the lean of
strictly prior seasons (the 2026 block by the 2017-2024 lean), so no game's own miss feeds its own
correction. WP fits no error spread (D31-07).

**The WP calibration gate (D-29)** is re-run per generation on WP's tune rows. A fail fires the
registered D31-07 fallback (prior-season probability-scale bias correction) for that generation's
WP bets; a pass leaves WP priced on its calibrated probability.

**The 2026 WP market side (D-30).** Finished 2026 games in the WP re-measure are converted through
the `slope_beta` of `market_probability_20260923_195443`, fitted on 2020-2024, strictly before
2026.

**Edge bands.** WP keeps the anchor pair 0.05 / 0.02. ATS and O/U pairs are numpy linear-method
quantiles of their own absolute-edge distributions reproducing WP's band shares, rounded to 4
decimals. The O/U edge is a ratio of the market total, with the total floored at 30.

**The 100-row floor.** A target with fewer than 100 usable re-measure rows is an integrity refusal
(`remeasure_complete`), never a no-bets generation.

**EV floors** are 0.0 / 0.0 / 0.01 (WP / ATS / O/U) in every generation.

**Blend weights.** One per target, re-tuned by `backtest.tune.run_blend_tuning` on the same
predictions and corpus, with the new generation's three model ids as its sources and bound to the
same converter, `market_probability_20260923_195443`. No tracked file is edited to do this.

**Recorded per generation.** Every re-measured number, its windows, its row counts and its
excluded-game counts are recorded for each generation under `ledger/refit/bet_rules/`, and the
generation's rows name them.

## 10. Integrity checks and the swap (R5, D-25, D-34, RFIT-06)

A re-fit goes live only if it passes every check below, in this order. The first failure refuses
the attempt, records `refused` naming that check, and leaves `artifacts/latest.json` byte-unchanged
with the previous generation still serving.

1. `recipe_fingerprint`: the running checkout's two record files hash to the pre-registered
   witness; a difference refuses before any training.
2. `aa_proof`: for the first re-fit only, the date list in force was selected by a clean A/A week
   recorded under section 5.
3. `training_scope`: every target's training rows follow section 8 (kickoff strictly before the
   cutoff, final score, no provisional Elo snapshot, labels equal to silver) and its row count
   equals the finished games minus the recorded exclusions.
4. `input_pool_and_caps`: every selected input comes from the frozen pool and each target stays
   within its cap.
5. `fixed_settings`: every model carries exactly the fixed settings, the recorded thread count is 1,
   and no search ran.
6. `out_of_sample_calibration`: the WP calibrator and the ATS and O/U residual converters were fitted
   only on predictions of games their model did not train on, by section 8's calibration rule.
7. `remeasure_complete`: section 9 finished for all three targets with at least 100 usable rows
   each, and its numbers are recorded.
8. `scores_frame`: each new model loads, its feature list is a subset of the gold columns, and it
   returns one finite value per row of the integrity frame (WP within [0, 1]).
9. `dry_stamp_and_validate`: a decision over the integrity frame through the generation-aware bet
   rule, the ledger's stamping step and its new-row validation, with NO write, accepts the new ids,
   proving the next decision run can stamp them.
10. `clean_code_commit`: before the install intent is written, the attempt's recorded code commit
    must be a full 40-hex commit with no uncommitted change to the tracked code. When git cannot
    name it, or the code has uncommitted changes, the install is refused by `clean_code_commit`,
    nothing is written to the install record or to `latest.json`, the date stays open, and the next
    daily run retries it with the identical cutoff (D-05, R10). The reason: the install intent is a
    permanent record, and the first re-fit's code version must be a real commit that descends from
    this record.
11. `bundle_validation`: the swap's own validation of the four artifacts through the serving
    loaders, including that the new blend's source ids equal the three new model ids.

**The integrity frame (D-34, D-25).** The stored 2026 gold rows of the NFL week that just finished,
plus every gold row of a slate built that same day (a catch-up on a prediction day). Nothing is
built for a future slate.

**The swap.** The install intent (the new ids and the generation's bet-rule record) is recorded
first; then `models.artifacts.replace_manifest` installs the four pointers (WP, ATS, O/U, blend) at
once. It is the only swap surface. A crash between the swap and the closing record leaves
`latest.json` entirely old or entirely new, and a completed date never swaps again.

**The gate relationship (RFIT-06), verbatim:**

gate-free by design; the shadow arm is the control and the season close is the verdict

There is no performance test before a re-fit goes live. `config/gate.toml`,
`models/deploy_gate.py` and `scripts/promote_models.py` are untouched by this phase, and their two
tripwires stay red.

## 11. The comparison (D-07, D-08, D-09, D-10, D-11, D-18, D-31, D-32, R12)

**What is scored (D-18).** Each arm's own stored model prediction, the ledger's immutable
`model_value` (WP `model_prob`, ATS `model_spread`, O/U `model_total`), never the market-blended
value.

**Per-game loss.** WP: log loss of the model's home-win probability, clipped to [1e-15, 1 - 1e-15],
against y = 1 when the home score is greater than the away score and 0 otherwise (a tie is 0,
D-31). ATS: the absolute error in points of the predicted home margin against the final home margin
(home minus away). O/U: the absolute error in points of the predicted total against the final
total.

**Pairing.** Only games both arms predicted count. Per game, gap = live loss minus frozen loss. A
game belongs to the generation of its live row; no test ever pools games from two generations.

**The test (D-07).** A paired t-test on the per-game loss gaps, two-sided: the Diebold-Mariano test
in its independent-games form, computed as the one-sample t-test of the gaps against 0. No
resampling is used, so no resampling block crosses a generation boundary.

**Minimum count (D-09).** A (target, generation) cell is tested only with at least 30 paired games.
Below that, it prints its count and "too few to test", with no p-value. If every gap in a cell is
exactly zero, it reports no difference, with no p-value.

**Correction (D-08, D-32).** Alpha 0.05, two-sided, with Benjamini-Hochberg across the family of
every (target, generation >= 1) cell with at least 30 paired games; the family size m is that count,
passed explicitly.

**The roll-up per target (D-10).** After the correction: at least one period significantly better
(negative mean gap) and none significantly worse is `HELPED`; at least one significantly worse and
none better is `HURT`; both is `MIXED`; neither is `NO DETECTABLE DIFFERENCE`.

**Secondary figures (D-11).** The pooled figure -- the same paired t-test over all games from the
tested periods -- is printed after the per-period figures, labelled secondary, reports its own
p-value, sits outside the correction family and never changes the word. Flat ROI per arm is
reported as a secondary figure with no test, in the comparison command only. Generation 0 is
reported but never tested (its gap is zero by construction). CLV appears nowhere in the deciding
output.

**When.** The pre-registered test runs once, at season close (Phase 38). The comparison panel on
the Season page shows raw numbers only all season -- live error, frozen error, the gap, which arm is
ahead and the paired-game count -- with no p-value, no confidence range and no "significant" word
(D-12, D-13). Nothing acts on the panel.

## 12. Display (D-36)

The website's edge-band display follows the re-measured edge-band thresholds of the generation in
force, so a game's displayed band is the band its live bet rule used.
