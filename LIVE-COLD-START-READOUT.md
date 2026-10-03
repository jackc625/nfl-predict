# LIVE-COLD-START-READOUT.md -- Phase 33's closing record: the first live 2026 weeks, checked

**Phase 33, Plan 33-18. Requirement COLD-01; owner decisions D33-37, D33-38, D33-39, D33-40.
Written 2026-10-03, after the scheduled run of that evening.**

This document reports what the real, scheduled daily runs of 2026 did, and what was checked about
them. It recomputes nothing: every number below is read from `tests/phase33_state.py` (the
Phase-33 state manifest) or from another committed file, and the permanent guard
`tests/unit/test_phase33_readout_md.py` holds the two together.

It is written under the day-before lock: each game's information locks at 6 PM Eastern on the day
before its kickoff (`utils/game_lock.py`). Only the 2026 season, recorded live under that rule,
counts as evidence of whether the system works (D33.2-07). Nothing here is a claim about
accuracy or profit.

---

## 1. What this phase set out to prove, and what it proves

**In plain words.** Phase 33 had to show that a real 2026 week goes through the whole system
correctly: the team ratings (Elo) that reach the models are the real ratings, not a filler value,
and every prediction and every bet decision is made before its game's lock. That is COLD-01.

**What it shows.** Every slate a scheduled run reached was predicted completely: 14 of 14 games
for the 2026-09-26 lock, and 14 of 14 for the bracketed slate of the 2026-10-03 lock. Games whose
lock passed with no successful run have no prediction, by design, and are never back-filled
(section 3 names them).

## 2. The acceptance evidence: week 3's Elo, value by value

**In plain words.** The defect this phase exists to remove once produced a run that finished
"successfully" while feeding the models made-up ratings. So the run finishing is not the check.
A zero exit code is not the evidence; the numbers are.

**What was checked.** The gold tables built by the first scheduled run (build clock
`2026-09-26T21:12:02.795159+00:00`) were compared, value by value, with the ratings stored in
`data/silver/elo_game_snapshots.parquet`. The checked files carry these sha256 digests:

| File | sha256 |
|---|---|
| `data/gold/features_wp.parquet` | `92f203951ac6ac91062bc1a125a9708433e1471e2d8a136a6c384736d6a8f6c3` |
| `data/gold/features_ats.parquet` | `06c2dbfc11fbf58df2b6894aa67c0e9f24d89c90ed859cf57e0d8ebad46fcad8` |
| `data/gold/features_ou.parquet` | `28e9dfe8209e40af99d1b834d6fce39f13ba016faa99d071fc417e2cc3d4c031` |
| `data/silver/elo_game_snapshots.parquet` | `f3d91fa52220fa90d51159499a308c3dd6cea2396ca94f4727e8391dfc9f3e85` |

15 week-3 games x 3 tables x 8 columns: six ratings joined straight from the snapshot table
(`home_elo_pre` -> `home_elo`, `away_elo_pre` -> `away_elo`, and the two uncertainties, the Elo win
probability and the home-field value by the same name) plus two worked out by formula
(`elo_diff`, `elo_prob_away`). The largest difference across all 24 table-and-column pairs is
`8.881784197001252e-16`, against a tolerance of `1e-9`. 14 of the 15 checked rows were
provisional ratings for games not yet played; the other, `2026_W03_ATL@GB`, had been played.

**Why the comparison is to a recomputed value.** Gold does not store raw ratings. It clips each
column at the 1st and 99th percentiles of earlier seasons and then expresses each value as a
z-score against every value of its season locked at or before it (`scripts/build_features.py`,
`features/normalization.py`). So the check repeats that transform independently, from the
snapshot table alone, and compares the result. A raw rating would never equal a z-score.

**Why the check can fail.** It counts before it compares (every 2026 gold row has exactly one
snapshot row); adding +1 to one game's rating in memory makes it fail; shuffling both tables
changes nothing; and a z-score of exactly 0.0 is treated as a real reading, never as a "missing"
marker. The check is permanent: `tests/integration/test_live_2026_prediction_set.py` runs it on
every 2026 row from week 3 onward, so it keeps holding after each nightly rebuild.

## 3. The week-3 games: predicted, locked out, missed

**In plain words.** Week 3 had 16 games. 14 were predicted. Two were not, and neither will be.

| Game | Lock (UTC) | Why no prediction |
|---|---|---|
| `2026_W03_ATL@GB` | `2026-09-23T22:00:00+00:00` | its lock (2026-09-23 18:00 ET) passed before the daily task was installed on 2026-09-24 at 00:38 ET |
| `2026_W03_PHI@CHI` | `2026-09-27T22:00:00+00:00` | the 2026-09-27 17:00 ET run was missed while the laptop slept on battery, and a missed day is never back-filled (D33-39) |

Weeks 1 and 2 had no live run: week 1 was conceded in advance (D40-04), and the daily task went
live on 2026-09-24. The old plan for the first live week to be week 2 (D33-02) was overtaken by
events: Phase 33.2 was inserted on 2026-09-15, and the first live predictions were for the
2026-09-26 lock.

## 4. The bracketed daily run

**In plain words.** To see exactly what one real nightly run changes, a "before" picture of every
file it could touch was taken, the scheduled 5 PM run did its job untouched, and an "after"
picture was compared with a list of changes written down before the run.

**Which run, and why that one.** The bracket was meant for 2026-09-30, but the fallback run was
used. The recorded reason (`DAILY_BRACKET_FALLBACK_REASON`): on 2026-09-30 the 17:00 ET scheduled
daily run failed at data_qa (the weather completeness check expected all 16 week-4 games but the
slate-scoped ingest had fetched 1, so `2026_W04_PIT@CLE` was not predicted and is never
back-filled), and the session resumed at about 20:11 ET, after the 16:30 ET picture limit, so no
before-picture was taken and the bracket moved to the 2026-10-03 17:00 ET run.

**What was declared before the run** (commit `878116b`): the 21 steps of the live daily registry
(`pipeline/daily_steps.py`) plus the two schedule-refresh operations that run before it; five
roots (`data`, `outputs`, `artifacts`, `config`, `logs`); 26 files the run must write, 6 it may
write with a checkable condition, 1 it may remove, and 3 append-only `.jsonl` files with their
expected line counts. Nothing under `artifacts/` was declared, on purpose: the run loads the
models and never writes them.

**What the run did** (the scheduler's own run, observed 2026-10-03): started
`2026-10-03T17:00:17.418538-04:00`, ended `2026-10-03T17:19:00.098388-04:00`, status `success`,
21 of 21 steps succeeded, no game skipped; the scheduler's Last Result was 0.

**What moved.** 29 files changed in content: 23 under `data/`, 4 under `outputs/`, 1 under
`config/`, 1 under `logs/`, and none of the 206 files under `artifacts/`. No file was removed, no
undeclared file moved, and no comparison mixed a content hash with a file-date signature. The
`.jsonl` line counts moved by +1 (the capture log), 0 and 0, exactly as declared.

**The one finding.** `outputs/bet_list/bet_tracker.json` was declared as a file the run must
write, and the run did rewrite it, but its content is byte-identical: the tracker counts only
graded bets, and the 42 new rows are not graded until their games are played. A declaration that
was broader than necessary, not a defect; the declaration was not changed after the fact.

**The slate's own values.** All 14 Sunday games were predicted. Each prediction's information
cut-off equals the lock (`2026-10-03T22:00:00+00:00`); its inputs were captured at
`2026-10-03T21:00:48.459247+00:00` and it was computed at `2026-10-03T21:18:10.782578+00:00`, both
before the lock. Every slate game had a betting line from before the lock: 126 live odds rows
(9 sportsbooks x 14 games), the first predicting night with the free Odds API key.

## 5. Forward bet rows

**In plain words.** A "forward" bet row is a bet decision the live system writes for a game not
yet played. Each one carries `decided_at_utc`, the moment it was decided, and that moment must be
at or before its game's lock.

The week-3 check found 42 forward rows (14 games x 3 targets), every one decided at or before its
lock; the latest was decided 2722.29 seconds before its lock. The bracketed night wrote 42 more,
all decided at `2026-10-03T21:18:10.874141Z`, 2509.13 seconds before the lock (24 live bets, 18
suppressed). Every forward row's stored freeze time equals its game's lock. The permanent check is
`tests/integration/test_live_2026_prediction_set.py`.

**Handed to Phase 34 (D33-29's surviving concern).** Forward rows written now lack the columns
that would make each one fully attributable: `arm`, the artifact and recipe stamps, and the
reproduction key. Those columns belong to Phase 34's ledger and cannot be filled in afterwards.

## 6. What Phase 33.2 superseded

**In plain words.** Phase 33.2 found that inputs Phase 33 relied on were defective, and replaced
the models, numbers and rules built on them. Each item below is named only as a superseded record
beside what replaced it. No number from a superseded record is repeated here.

| Superseded record | Replaced by |
|---|---|
| win model `wp_20260914_221745` | `wp_20260923_172144` |
| spread model `ats_20260914_221751` | `ats_20260923_172148` |
| total model `ou_20260914_221756` | `ou_20260923_172152` |
| the held dynamic blend `blend_dynamic_20260606_020635` | `blend_20260923_212418` |
| the Plan 33-15 gate verdicts (record: `config/phase33_gate_verdict.toml`) | the corrected re-fit's labelled record, `REFIT-READOUT.md` |
| the 2026 rule frozen at `11761c7` | `9bb7568` (`COLD-START-CORRECTION.md`) |
| the Phase-31 EV floor and frozen residual SD (`ee20773`) | `8c9675e` (`EV-CHAIN-CORRECTION.md`) |
| the Friday 6 PM freeze | the day-before 6 PM lock (`utils/game_lock.py`, D33.2-01) |
| D33-15's 91 neutral-site games left unrepaired | repaired by Phase 33.1 (Plan 33.1-03) |
| D33-25's hold on gold weather for 2026 | ended at Plan 33-15; replaced by day-before forecasts |
| D33-14's blend re-score and D33-22's band table | the corrected blend and the corrected bands above |

**The threshold derivation is still circular, and it is said here plainly.** The superseded
pre-registration, `COLD-START-PREREGISTRATION.md` (frozen at `11761c7`, replaced by `9bb7568`),
recorded that its thresholds were derived on gold seasons 2021-2024, which was also the population
its label movement was measured against. The correction, `COLD-START-CORRECTION.md`, derives its
thresholds over its 2020-2024 window (2021-2024 for the win model) and measures its label
movement over the same corrected rows. So the circularity carries over: the bands are measured on
the population that set them.

## 7. The Elo re-derivation

**In plain words.** The historical team ratings were derived again from the game results, not
recovered from a saved copy: no known-good copy existed anywhere (finding F-03). The Elo work is
a re-derivation.

**What made it canonical** (Plan 33-13, generation `20260914T235426782597`): the chain starts in
2002 and ends in 2025; every season reconciles against the games table; both side tables hold
6,499 rows; every rating falls inside the band frozen before the run (980.0 to 2070.0); and the
five artifacts were anchored by content digest in the state manifest.

**Independent corroboration** (Plan 33.2-04): `audit/elo_replay.py` replayed every pre-game rating
from only the results known at each game's lock and compared 6,515 rows (38,839 cells) exactly:
0 mismatches.

**Still holding.** The content digest of the 2002-2025 slice of the snapshot table
(`P332_20_ELO_CANONICAL_SLICE_CONTENT_SHA256`, 6,499 rows) still matches after the 2026-10-03
rebuild. The nightly runs only add 2026 rows.

## 8. Dated operational findings

**In plain words.** What actually happened to the scheduled runs, night by night. Facts only.

- **2026-09-26.** The first real daily run (17:00-17:15 ET) predicted 14 of 14 Sunday games with
  no market side: the Odds API key had been deactivated since the paid month ended, and no 2026
  odds had ever been captured. The owner installed the free Starter-plan key that evening.
- **2026-09-27.** The run was missed. The laptop slept on battery from 2026-09-26 20:20 to
  2026-09-28 11:29 ET, and this machine allows wake timers on mains power only.
  `2026_W03_PHI@CHI` got no prediction and is never back-filled. Owner ruling 2026-09-28: keep the
  laptop plugged in at 17:00 ET, and stop the task being killed on battery (`bcc2256`,
  re-installed and read back as matching).
- **2026-09-28 and 2026-09-29.** No games next day; each night left its no-games record and a
  capture line.
- **2026-09-30.** The run failed at data_qa: the weather completeness check expected all 16
  week-4 games, while the nightly build fetches only the next day's slate (it had fetched 1).
  `2026_W04_PIT@CLE` got no prediction and is never back-filled. Fixed the same night, in 15
  commits `6ae5530`..`ba92c45`: `1f73e31` and `afcce78` (the weather check expects the slate, and a
  recorded forecast failure leaves only that game's weather unknown), `61a7fa1` (the games schema
  accepts unplayed games), `a1ea3a8` (the totals sanity floor admits a real 28.5 line), `da3cadc`
  (per-table quality checks are counted), `5e43a93` (the bet list's line-freshness check reads
  only a recorded capture time), `aaa3b34` (historical odds readers take one row per game),
  `516ac71` (the pin tool keeps untouched manifest fields), `97af245` (the 2025 replay badge reads
  "Old rule -- 2025, not evidence", owner ruling), `1b904a8` (training excludes the ruled-out
  feature groups by default), `3498d8e` and `ba92c45` (two stale tests brought up to date),
  `927c3ec` (a corrected comment time), `6ae5530` (the fallback reason) and `58a11ed` (below).
- **2026-09-30, D33-40.** The sealed nflverse schedules for 2024 and 2025 had changed upstream
  since 2026-09-15 in one column only, `ftn` (an outside charting provider's game id, 53 cells),
  which nothing in the pipeline reads. By owner ruling they were re-pinned to the corrected files
  (`58a11ed`); no feature, model or prediction moved. The capture log records every capture from
  2026-09-15 through 2026-09-30 as a sealed-file change, and every capture from 2026-10-01 on as
  clean.
- **2026-10-01 and 2026-10-02.** No games next day; each night left its no-games record and a
  clean capture line.
- **2026-10-03.** The bracketed run (section 4): 21 of 21 steps, 14 of 14 games, every slate game
  with a pre-lock line.

## 9. Suite state, as measured

**In plain words.** Five tests in this project fail on purpose: each records a fact the owner
accepted in Phase 30 or 31. After the last rebuild this plan observed, they were run by name, on
their own:

`5 failed, 34 warnings in 13.40s`

The five are exactly `DELIBERATE_TRIPWIRE_NODE_IDS`, each failing for its recorded reason.

**What was not run.** By the owner's ruling of 2026-09-28 ("A: Skip the count"), no whole-project
test count was taken and the whole suite was not run; only named test files and test ids were.
So this document makes no claim about the suite as a whole. Other known red tests are on record
in Phase 33.2's files and in the Plan 33-18 summary, and belong to the Phase 33 close-out's test
housekeeping; three of them were measured here and are handed on in section 10.

## 10. Open items handed forward

- **The rank-feature reproducibility hazard** -> Phase 34. Elo rank and percentile are worked out
  from the week's whole set of ratings at build time, so they can move when the build moves.
- **D33-29's attribution columns** -> Phase 34: `arm`, the artifact and recipe stamps and the
  reproduction key on forward rows (section 5).
- **Freezing published feature rows** (`research/ARCHITECTURE.md` S5.4) -> Phase 34.
- **The 47-of-68-field gate-baseline divergence**, still unexplained. Three tests outside the five
  fail on the same gap, measured 2026-10-03:
  `test_drift_tripwire_sees_frozen_seasons.py::TestTheLiveFrameWouldAbort` (two tests) and
  `test_promote_models.py::test_frozen_baseline_matches_rescore`. The frozen baseline is dead by
  the owner's standing rule; the gate's future is RFIT-06 (Phase 37) and clearing quarantines is
  CLEAN-02 (Phase 38).
- **A repo-wide roof-disagreement detector**, still deferred.
- **Renaming the stored `*_confidence` columns**, still owed.
