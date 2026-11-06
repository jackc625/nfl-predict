# MID-SEASON SWITCH RECORD -- the day-before system retires from 2026 week 13

**Status:** FROZEN. This document and `backtest/midseason_switch_2026.py` are ONE record in two
files, landed in ONE commit containing nothing else. This is the human-readable half; the
constants module is the half the code reads. Every value written in both places is the same value.

**What this record is for.** Study 1a measured that the repaired v2 recipe is more accurate than
the current recipe on the pooled 2017-2024 replay, and that neither earned any weight against the
day-before line: every bet type earned weight 0. So from 2026 week 13 the old day-before betting
system stops recording decisions, the v1 calendar re-fit stops, and the v2 models are installed
once, by a recorded owner command. Several frozen 2026 records assumed the old system would run to
the Super Bowl. This record says, before week 13 begins, exactly which of their statements no
longer hold and why, so a reader in February can check that the change was decided before its
effect existed and not chosen after.

**Editing after the commit does not fix a bug -- it destroys the evidence.** A value that is wrong
here is wrong for the rest of 2026. The only legitimate response is a NEW, VISIBLY LATER
CORRECTIVE COMMIT whose message explicitly INVALIDATES this record by naming its commit sha. Never
an edit in place, and never a quiet supersession.

**The anchor is recorded outside the files it witnesses.** Neither this document nor
`backtest/midseason_switch_2026.py` records its own content hash or its own commit. A file that
must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential: writing the hash
changes the bytes the hash was computed over, so no fixed point exists. The witness lands in a
LATER commit, in a test-state module, which also witnesses `models/v2_recipe.py` (section 4).

```
preregistration_anchor_recorded_in: tests/midseason_switch_state.py
preregistration_anchor_slots: SWITCH_RECORD_WITNESS, SWITCH_RECORD_COMMITTED_AT, SWITCH_RECORD_DEADLINE_UTC, V2_RECIPE_FILE_WITNESS
preregistration_paths: MIDSEASON-SWITCH-RECORD.md, backtest/midseason_switch_2026.py
preregistration_deadline: 2026-12-01T17:00:00-05:00
```

The deadline is Tue 2026-12-01 17:00 America/New_York (22:00 UTC), the cutoff of re-fit date 2
(section 3). It falls after week 12's last lock (Sun 2026-11-29 18:00 ET) and before week 13's
first lock (Wed 2026-12-02 18:00 ET), both measured on the recorded 2026 schedule
(`data/silver/games.parquet`, read 2026-11-06). This commit's committer date must precede it, and
this commit and the code that reads it must be on GitHub's `master` before it.

---

## 1. What changes from 2026 week 13

**The switch is keyed on the NFL week, not the clock.** Every 2026 game of week 13 or later
(`DAY_BEFORE_RETIRED_FROM = (2026, 13)`) is outside the old system, from Tue 2026-12-01 17:00 ET
on. That instant falls between week 12's last lock and week 13's first, so no NFL week is decided
by two systems.

- **No live row.** No `arm="live"` row is written to the forward ledger for a 2026 game of week 13
  or later.
- **So no shadow twin.** A shadow row is written only beside a new live row of the same game and
  target (`ROLLING-REFIT-PREREGISTRATION.md` section 2), so the frozen shadow arm stops after week
  12 by itself. It is not re-pinned.
- **No new arm.** This record adds no arm value of any kind.
- **The day-before window places no bets.** The v2 models earned weight 0 on all three bet types
  (section 2), and a weight of 0 means no day-before bets for that bet type.

The retirement does not depend on the install: if v2 never installs, no live row is written from
week 13 all the same, and the current recipe earned weight 0 as well.

**What stays as it was.** Every ledger row written before week 13 stays unchanged and keeps being
graded as before. No existing row is rewritten.

## 2. Why

**Study 1a** is the walk-forward replay of the fixed-input v2 recipe on the repaired gold against
the current live recipe, re-fitted at every fold on the live gold, with one fold per season
2017-2024 and then one per finished 2026 week (replay cutoff 2026-11-06T03:20:04.972329+00:00).
Its report, `.planning/research/2026-11-05-why-not-profitable/study-1a-report.md`, is an untracked
working file, so the figures this record rests on are copied here.

**The install bar.** Pooled 2017-2024 walk-forward loss, lower is better; v2 installs only if it is
at least as good as the current recipe on every target:

| Target | Metric | Current | v2 | Games |
|---|---|---|---|---|
| WP (win / moneyline) | log loss | 0.6410 | 0.6287 | 2209 |
| ATS (spread) | MAE | 10.504 | 10.204 | 2209 |
| O/U (total) | MAE | 11.034 | 10.741 | 2209 |

v2 clears it on all three. The report's caveat: the v2 inputs were fixed before the replay but were
informed by the audit's measurements on the same 2017-2024 seasons, so a small optimistic bias in
v2's install margin is possible. On the 121 finished 2026 games (reported, never a veto) v2 was
worse on WP (0.6509 vs 0.6495) and ATS (10.050 vs 9.942) and better on O/U (11.259 vs 11.541).

**The earned weights.** Against the day-before line (the 2020-2024 Friday consensus plus the 2026
live captures): the slope of the realized result on the model's departure from the line. A weight
above 0 is earned only where that slope is positive at one-sided p < 0.05:

| Target | Model | Games | Slope | One-sided p | Fitted blend weight | Earned weight |
|---|---|---|---|---|---|---|
| WP | v2 | 1151 | -0.0825 | 0.699 | 0.00 | 0.00 |
| ATS | v2 | 1406 | +0.0038 | 0.487 | 0.00 | 0.00 |
| O/U | v2 | 1406 | +0.0755 | 0.258 | 0.00 | 0.00 |
| WP | current | 1151 | -0.0801 | 0.722 | 0.00 | 0.00 |
| ATS | current | 1406 | +0.0121 | 0.442 | 0.00 | 0.00 |
| O/U | current | 1406 | +0.1307 | 0.0342 | 0.00 | 0.00 |

v2 earns 0 / 0 / 0 (WP / ATS / O/U), `V2_EARNED_WEIGHTS`. The current O/U model is the one test of
six whose slope passes, but its fitted blend weight is 0, so it earns 0 as well. On the same games,
v2 on its own trails the line on every target: WP log loss 0.6375 vs 0.6165, ATS MAE 10.046 vs
9.754, O/U MAE 10.564 vs 10.293.

**The owner rulings.**

- **2026-11-05, ruling B:** each change goes live as soon as it passes a replay on past seasons;
  the season freeze is lifted for these changes; the February readout becomes a before/after.
- **2026-11-06, the lean switch:** Study 1a gave all three bet types weight 0 and nothing is
  re-fitted after the switch, so the day-before window places no bets for the rest of 2026 either
  way. The day-before pricing machinery (best price across books, quarter-Kelly sizing, a weekly
  cap, the no-contradiction rule, a `market` arm, `fill-v2`, a market bet-rule kind) is therefore
  not built now.

## 3. The v1 calendar re-fit is paused

**The list in force.** Week 8 was the first clean A/A week, so the date list for first clean week 8
in `ROLLING-REFIT-PREREGISTRATION.md` section 4 is in force: 2026-11-03 (week 9), 2026-12-01
(week 13), 2026-12-29 (week 17), 2027-01-26 (week 21). Date 1, 2026-11-03, ran and installed
generation 1 (`wp_20261103_221538`, `ats_20261103_221544`, `ou_20261103_221550`,
`blend_20261103_221911`), which decides the live rows of weeks 9-12.

**Dates 2-4 close `not run`, paused by this record** (`REFIT_PAUSED_DATES`):

| Re-fit date | Cutoff | Precedes NFL week |
|---|---|---|
| 2: 2026-12-01 | Tue 17:00 America/New_York (22:00 UTC) | 13 |
| 3: 2026-12-29 | Tue 17:00 America/New_York (22:00 UTC) | 17 |
| 4: 2027-01-26 | Tue 17:00 America/New_York (22:00 UTC) | 21 |

When each falls due, the daily run's re-fit side step closes it `not run`, once, carrying the
reason `MIDSEASON-SWITCH-RECORD.md section 3 (owner ruling B, 2026-11-05)` (`REFIT_PAUSE_RULING`).
Nothing trains, no model is saved and `artifacts/latest.json` is not touched. The frozen date list
and `refit/cadence.py` are not edited; the pause is read from this record's constants.

**No further 2026 re-fit runs**, of the v1 recipe or any other. The v2 models are trained once, at
their install (section 4), and never re-fitted in 2026.

**Why paused.** Any of these dates after the switch would retrain the old recipe and overwrite the
v2 models. The pause holds whether or not v2 installs, and it rests on owner ruling B (2026-11-05),
which lifted the season freeze for these changes. Re-fit date 2's cutoff is this record's deadline,
so the pause is in force before the first paused date can fall due.

## 4. The v2 install

**The recipe.** `recipe-2026-v2` (`INSTALLED_RECIPE`) is `models/v2_recipe.py` as last committed,
at commit `6652baf60baec2934b345a0a6e08848ec1e1ab1c` ("feat(v2): fixed-input linear recipe and
walk-forward trainer"). Its inputs were chosen once, by hand, and are never re-picked:

| Target | Label | Inputs | Model |
|---|---|---|---|
| WP | `home_win` | `elo_diff`, `success_rate_net`, `opp_adj_epa_net`, `qb_value_diff`, `mkt_rating_diff`, `hfa_used` | logistic regression, C 1.0, max_iter 1000 |
| ATS | `home_margin` | the same six as WP | ridge regression, alpha 1.0 |
| O/U | `total_points` | `league_points_level`, `mkt_total_sum`, `off_epa_sum`, `def_epa_sum`, `pace_sum`, `wind_mph` | ridge regression, alpha 1.0 |

Each model is median imputation, then standardisation, then the linear model.

**Training.** Once, at the install, on every game from season 2002 that kicked off strictly before
the install instant and has a final score. The install instant is the training cutoff. No search
of any kind.

**Blend weights.** The install tunes one blend weight per target and refuses unless every tuned
weight equals the earned weight of section 2 (`V2_EARNED_WEIGHTS`: 0 / 0 / 0). A tuned weight other
than 0 would be a new finding for the owner to rule on, never something to install silently.

**Bet rule.** The install writes a bet-rule record for the v2 generation with no EV floors and no
edge bands: a model with weight 0 has no edge to band.

**The record and the swap.** Every install is recorded, append-only, in
`ledger/refit/owner_installs.jsonl`: the install intent before the swap, then the installed event,
or a refused event. `artifacts/latest.json` is written only by `models.artifacts.replace_manifest`,
never by hand. The recorded install becomes the next generation.

**When.** Only after week 12's last lock (Sun 2026-11-29 18:00 ET). While any 2026 game of weeks
1-12 is still before its lock, the install refuses and writes nothing, so no week-12 decision can
be stamped with a v2 model. It is one owner command; once v2 is in force, a second install
refuses.

**Gold.** The same command rebuilds the shared gold matrices with the v2 feature builder, and from
then on the nightly gold build follows the recipe in force, so the v1 models never score v2 gold.
An install that fails after the v2 gold is built rebuilds the v1 gold before it stops.

**No ledger row is decided by v2 under this record.** The install waits for week 12's last lock,
and from week 13 no live row is written (section 1).

## 5. The February readout

- **The R6 money verdict** (`SEASON-VERDICT-PREREGISTRATION.md` and `backtest/verdict_test_2026.py`,
  commit 922b010) grades the live arm over weeks 5-12 by its frozen rule, unchanged: the same test,
  the same three-test family, the same alpha and the same verdict words. Only the span is shorter,
  because no live row exists after week 12.
- **The coverage table ends at week 12** (`R6_COVERAGE_END_WEEK`) and cites this record, instead of
  reporting weeks 13-22 as missed.
- **A counted live row in week 13 or later refuses the run.** Such a row would mean the retirement
  leaked and this record is false.
- **The re-fit comparison** (`ROLLING-REFIT-PREREGISTRATION.md` section 11, commit e25884d) covers
  generation 1 against generation 0 over weeks 9-12, by its frozen test, minimum count and
  correction. No later generation decides a live row, so there is nothing else to compare.

## 6. What this record INVALIDATES

None of the files below is edited; each stays byte-unchanged. From this record on, these
statements in them no longer hold:

| Commit | Files | What no longer holds |
|---|---|---|
| `922b010b70d973a21945c15ca88067dae4149ad5` | `SEASON-VERDICT-PREREGISTRATION.md`, `backtest/verdict_test_2026.py` | In part: "the whole live 2026 season" is now weeks 5-12. Everything else in it stands. |
| `c3a33993160769b2ab9a8c2cf7cd2a24ad19be10` | `backtest/verdict_scope_2026.py` | In part: "The verdict is the LIVE arm's: the bets the system actually recommended." no longer holds from week 13, where the system records no day-before decision. The declared weeks 5-22 stay; weeks 13-22 hold no counted row. |
| `e25884da8356b9ec76970c4c14f87a797d2a32d7` | `ROLLING-REFIT-PREREGISTRATION.md`, `backtest/refit_preregistration_2026.py` | Section 1 (one frozen recipe for the season): from the install the serving models are `recipe-2026-v2`. Section 3 (generations): the owner install is a generation outside the cadence-date numbering. Section 4 (the calendar): dates 2-4 close `not run` (section 3 here). Section 6 (when a re-fit runs): a paused date that falls due is closed `not run`, not attempted. Section 9 (the re-measured bet rule and its EV floors): the v2 generation's bet rule has no EV floors and no edge bands. Section 11 (the comparison): limited to generation 1 against generation 0 over weeks 9-12. |

## 7. What this record does not do

- No `fill-v2` pricing convention: `fill-v1` stands.
- No `market` arm and no `early` arm.
- No new pre-registration: no new test, family or binding readout for 2026. This record fixes a
  switch, not a test.
- No day-before pricing machinery (owner ruling 2026-11-06).

Each of those is for a later record, which will name what it changes.
