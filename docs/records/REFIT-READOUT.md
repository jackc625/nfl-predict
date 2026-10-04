# REFIT-READOUT.md -- the corrected models and blend, and what their past seasons show

**Phase 33.2, Plan 33.2-25. SPEC R13; owner decisions D33.2-03, D33.2-07, D33.2-10, D33.2-17.
Written 2026-09-23, before the production swap.**

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

This document is a RECORD, not a justification. By the owner's own ruling the corrected models
replace today's UNCONDITIONALLY: there is no pass/fail gate, so nothing written below decides
anything. It exists so that what goes into production is on the record, with the fold behind every
number stated plainly.

---

## 1. What was re-fitted, on what, and why nothing is compared with the old models

**What.** All three models -- win probability (WP), the spread (ATS) and the total (O/U) -- were
fitted again from scratch (Plan 33.2-23, then re-fitted once more on the same settings with the
`snap` family's coverage flag left out, section 8), and the blend that mixes each model with the
betting market was tuned again against those models (Plan 33.2-24's procedure).

**On what.** The corrected feature tables ("gold") that this phase rebuilt: every input now carries
only information that existed by 6 PM Eastern on the day before its game's kickoff; past weather is
the forecast that stood at that moment rather than what the weather later turned out to be; no
betting line is an input to any model; the early seasons carry an honest blank with a flag where
data did not exist, instead of a made-up "exactly average". All four artifacts record the same gold
generation, `484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597`.

**Why no comparison with the models they replace.** Those models were fitted on inputs this phase
later found defective -- betting lines that did not exist at prediction time, weather known only
after the game, early seasons filled with invented averages. Comparing the new models with them
would measure the defect, not the correction. The standing ruling is that they are dead, and this
document names none of them (section 9).

## 2. The four artifacts

| Key | Artifact | What it is | Gold generation it records |
|---|---|---|---|
| wp | `wp_20260923_172144` | win probability, logistic regression with a fitted calibration | `484397642530...1522597` in `metadata.json` |
| ats | `ats_20260923_172148` | spread (final home margin), XGBoost regression | `484397642530...1522597` in `metadata.json` |
| ou | `ou_20260923_172152` | total points, XGBoost regression | `484397642530...1522597` in `metadata.json` |
| blend | `blend_20260923_212418` | one fixed model weight per target, market for the rest | `484397642530...1522597` in `blend_weights.json` |

The blend is bound to the spread-to-win-probability converter `market_probability_20260923_195443`
(the converter is not a production pointer; the blend names it inside its own payload, and the
swap checks that it resolves). Every model and the blend record an OpenMP thread count of 1, the
pin that makes their numbers reproducible on any machine (Plan 33.2-22's finding).

No number in this document was computed for it. Every figure is read from the artifact or the state
record that captured it when the fit ran, and every one of those runs was pinned to one thread.

## 3. The folds -- every published number comes from a fold whose training seasons all precede it

A "fold" is one prediction exercise: fit on some seasons, then predict a later season the fit never
saw. Two sets of folds stand behind the results below.

**The holdout folds** (section 4). Each model's recipe was fixed on the early seasons -- features
chosen on 2002-2022, and each model's calibration step (the win model's probability calibration; the
spread and total models' converters from a predicted number to a probability) fitted on 2023
predictions made by a model trained on 2002-2022. Then each holdout season was predicted by a model
trained on every season before it and nothing after.

**The blend folds** (section 5). For the blend, each season 2020-2024 was predicted by refitting the
same recipe as if that season were the next one to be played: features, calibration and model all
fitted on earlier seasons only (Plan 33.2-24). For win probability the market's spread was turned
into a probability with a slope fitted on earlier owned seasons only.

| Fold | Test season | Model fitted on | Features chosen on | Calibration fitted on | Market converter fitted on (WP) |
|---|---|---|---|---|---|
| holdout-2024 | 2024 | 2002-2023 | 2002-2022 | 2023 | not used |
| holdout-2025 | 2025 | 2002-2024 | 2002-2022 | 2023 | not used |
| blend-2020 | 2020 | 2002-2019 | 2002-2017 | 2018 | none; WP is not tuned on this season |
| blend-2021 | 2021 | 2002-2020 | 2002-2018 | 2019 | 2020 |
| blend-2022 | 2022 | 2002-2021 | 2002-2019 | 2020 | 2020-2021 |
| blend-2023 | 2023 | 2002-2022 | 2002-2020 | 2021 | 2020-2022 |
| blend-2024 | 2024 | 2002-2023 | 2002-2021 | 2022 | 2020-2023 |

In every row the latest season any part of the fit touched is earlier than the season predicted.

**Two disclosures about the holdout seasons.**

- **2024 was also read by the tuning decision.** Plan 33.2-23's search for better settings was
  judged on 2024 against a random baseline. No search cleared its bar, so all three models ship on
  the trainers' standard settings, which no search chose. The settings are therefore not fitted to
  2024, but the decision to fall back to them did look at 2024, and that is stated here rather than
  left out.
- **2025 is the season the one-shot profitability readout already spent.** The 2025 figures below
  are prediction accuracy only -- no bet was selected, sized or graded -- and they re-open nothing
  that readout closed.

## 4. Results: out-of-sample accuracy, per target

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

**Win probability** -- how often the side given more than a 50% chance won, and the average gap
between the stated probability and what happened (0 would be perfect).

| Season | Games | Picked the winner | Average probability miss |
|---|---|---|---|
| 2024 | 285 | 0.666667 | 0.441823 |
| 2025 | 285 | 0.645614 | 0.453850 |

Across both seasons the calibration error -- how far the stated probabilities sit from the rates at
which those games were actually won -- is 0.085562.

**Spread** -- the average miss on the final home margin, in points; the larger typical miss (root
mean square); and R-squared, the share of the game-to-game variation in margins the model accounted
for (0 means no better than guessing that season's average margin for every game).

| Season | Games | Average miss (points) | Root-mean-square miss (points) | R-squared |
|---|---|---|---|---|
| 2024 | 285 | 10.073375 | 13.154046 | 0.170413 |
| 2025 | 285 | 10.313471 | 13.027124 | 0.150364 |

**Total** -- the same three measures for total points scored.

| Season | Games | Average miss (points) | Root-mean-square miss (points) | R-squared |
|---|---|---|---|---|
| 2024 | 285 | 10.214895 | 13.206984 | -0.003595 |
| 2025 | 285 | 11.059475 | 13.872577 | -0.007019 |

**What the total model's R-squared says, plainly.** It is below zero in both seasons: guessing that
season's own average total for every game would have missed by slightly less, in squared terms. The
corrected total model carries almost no game-by-game information about scoring on its own. Section 7
places that against what was expected before the re-fit.

## 5. Results: the corrected models against the market's pre-lock opinion

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

The blend was tuned on the line history the project owns: for each 2020-2024 game, the latest spread
and total captured at or before 6 PM Eastern on the day before kickoff. The model's side is the
blend-fold predictions of section 3. The market-alone and model-alone losses below are out of sample
for the models. The blend weight itself is one number per target chosen on these same games, so
"loss at the weight" is the result of that choice, not an out-of-sample figure for it.

| Target | Weight kept on the model | Loss measured | Market alone | Model alone | Loss at the weight | Games | Seasons |
|---|---|---|---|---|---|---|---|
| WP | 0.00 | log loss | 0.613530 | 0.645234 | 0.613530 | 1,093 | 2021-2024 |
| ATS | 0.00 | average miss, points | 9.795252 | 10.462824 | 9.795252 | 1,348 | 2020-2024 |
| O/U | 0.13 | average miss, points | 10.290987 | 10.846197 | 10.285894 | 1,348 | 2020-2024 |

**The finding, plainly.** For win probability and for the spread, the market's pre-lock opinion alone
did better than any mix with the corrected model, so the weight kept on the model is exactly zero and
the published blended number for those two targets IS the market's opinion. For the total, the model
earns a small share, 0.13, which improves the average miss by 0.005 points -- noise-sized, and no
significance is claimed. Games left out, and why, are listed in `BLEND-TUNING-READOUT.md` section 5.

## 6. Results: the random-baseline margin

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

Before any search ran, the owner fixed how much better a 1,000-trial search for model settings had to
be than 1,000 randomly chosen settings, scored once on 2024 (a season neither search saw), before its
winner could be kept -- and what happens if it is not.

| Target | Metric (lower is better) | Random settings, 2024 | Searched settings, 2024 | Gap | Bar to clear | Cleared | Settings shipped |
|---|---|---|---|---|---|---|---|
| WP | log loss | 0.599154 | 0.599154 | -0.0000004 | 0.0067 | no | standard defaults |
| ATS | average miss, points | 9.963965 | 9.931902 | 0.032063 | 0.49 | no | standard defaults |
| O/U | average miss, points | 9.923301 | 9.900250 | 0.023051 | 0.56 | no | standard defaults |

**No target cleared its bar, and the pre-registered rule did what it said.** A search that cannot beat
random choice by more than twice the measured noise has found noise, so each model ships on the
trainer's standard settings and the miss is published per target (owner ruling 2026-09-23,
`fall-back-to-defaults`). For win probability the two searches landed on the same score to six
decimal places. The later re-fit without the snap coverage flag (section 8) used the same standard
settings; the search was not run again.

## 7. What the models lost when the betting lines left -- expected before, observed after

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

**The expectation, recorded before the re-fit** (the rung-9 record in `tests/phase33_state.py`, written
when the betting lines left the inputs, and `33.2-CONTEXT.md`):

| Target | What left the inputs | Its weight in the model it left |
|---|---|---|
| WP | one line feature | 9.87% of the model's coefficient mass, rank 3 of 40 |
| ATS | two line features | 9.32% combined, ranks 3 and 4 |
| O/U | its most important input | 10.68%, rank 1 of 25 |

The largest loss was expected on the total.

**What happened, measured without any older model.**

| Target | Observed |
|---|---|
| WP | Picked the winner in 0.666667 (2024) and 0.645614 (2025) of games. Against the market's pre-lock opinion it added nothing: the blend keeps weight 0.00 on it. |
| ATS | R-squared 0.170413 and 0.150364: it carries some of the margin on its own. Against the market it added nothing: weight 0.00. |
| O/U | R-squared -0.003595 and -0.007019: on its own it is no better than a season-average guess. The blend keeps weight 0.13 on it for a 0.005-point gain. |

**Read plainly: the expectation held.** Every model lost one of its top-ranked inputs, and the total
model -- which lost its single most important input -- is the one left with the least to say. Once
the line is gone, the market's own pre-lock number beats each of the three models on its own
(section 5); only for the total does mixing in a small share of the model help at all, and then by
0.005 points.

## 8. The feature sets -- no betting line in any of them

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

| Target | Features used | Betting-line columns among them |
|---|---|---|
| WP | 20 | 0 |
| ATS | 25 | 0 |
| O/U | 25 | 0 |

The five betting-line columns (`snapshot_spread`, `snapshot_total`, `snapshot_ml_prob_home_fair`,
`spread_movement`, `total_movement`) left every gold table in rung 9 and appear in no written feature
list; `tests/unit/test_no_market_in_artifacts.py` checks the written lists, not the code's intent.
All three models also exclude the three feature families the re-measured group verdict dropped
(`injury`, `situational`, `snap`; owner ruling 2026-09-23).

**The snap coverage flag** (`home_snap_coverage` / `away_snap_coverage`) was removed with its family:
the `snap` family's definition now includes it, and the three models and the blend were re-fitted without it.

The lists as written:

- **WP (20):** away_abs_timezone_diff_hours, away_def_rolling_success_rate, away_elo,
  away_off_rolling_opp_adj_epa_per_play, away_off_rolling_opp_adj_pass_epa, away_short_rest,
  both_short_rest, hfa_used, home_def_rolling_opp_adj_epa_per_play,
  home_def_rolling_opp_adj_pass_epa, home_elo, home_off_rolling_success_rate, home_short_rest,
  is_divisional, passing_difficulty, passing_efficiency, scoring_reduction, thursday_game,
  turnover_multiplier, wind_mph.
- **ATS (25):** away_def_rolling_opp_adj_epa_per_play, away_eastward_travel, away_elo_percentile,
  away_off_rolling_opp_adj_pass_epa, away_off_rolling_pass_success_rate,
  away_off_rolling_rush_success_rate, away_off_rolling_success_rate, away_short_rest, elo_diff,
  elo_prob_away, elo_prob_home, home_def_rolling_opp_adj_epa_per_play, home_elo_percentile,
  home_elo_rank, home_elo_uncertainty, home_off_rolling_opp_adj_epa_per_play,
  home_off_rolling_opp_adj_pass_epa, home_off_rolling_opp_adj_rush_epa, home_rest_days,
  home_weather_advantage, is_snow, passing_efficiency, precip_moderate, season_progress,
  venue_outdoor.
- **O/U (25):** away_elo_percentile, away_off_rolling_neutral_pass_rate,
  away_off_rolling_pass_success_rate, away_off_rolling_success_rate, away_qb_adjustment,
  away_short_rest, ball_handling_difficulty, home_off_rolling_cpoe,
  home_off_rolling_opp_adj_epa_per_play, home_off_rolling_opp_adj_pass_epa, home_qb_adjustment,
  home_short_rest, home_weather_advantage, passing_difficulty, precip_prob, scoring_reduction,
  short_week, temp_mild, temp_very_cold, venue_outdoor, venue_retractable, weather_severity_score,
  wind_impact_score, wind_mph, wind_severe.

## 9. What is NOT here, and why

No pre-correction artifact id appears anywhere in this document, and no `config/gate.toml` baseline
appears in it either -- not as a comparator, not as a row, not as a reference point. The models,
the blend and the gate baselines that stand before this swap were all fitted or frozen on inputs this
phase found defective, so any comparison with them would measure the defect rather than the
correction, and the standing ruling is that they are dead. The one place their pointer values are
recorded is the state manifest (`tests/phase33_state.py`, the Plan 33.2-25 slot), and only so the
swap can be reversed by hand if that is ever needed.

There is no closing-line value, no bet, no return and no profitability figure here. Closing lines
did not exist when a prediction had to be made; they remain valid for grading bets and for
report-only closing-line value, and neither is part of this record.

## 10. What happens next

The owner decides whether these four artifacts become what serves. If they do, one atomic write
changes all four production pointers at once (`models.artifacts.replace_manifest`), after proving
each artifact loads the way the serving path loads it and that all four were fitted on the same
gold. From then on, the only evidence of whether any of this works is the 2026 season, recorded live
under the day-before 6 PM lock.

**Not clean evidence.** These are re-measured past seasons, built under the new rule on corrected
inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here is evidence of
accuracy or profitability.

---

*Every figure is read from `artifacts/{wp_20260923_172144,ats_20260923_172148,ou_20260923_172152}/metadata.json`,*
*`artifacts/blend_20260923_212418/blend_weights.json` and the `P332_23_*` / `P332_25_*` / `P332_25B_*`*
*slots of `tests/phase33_state.py`. `tests/unit/test_refit_readout_md.py` holds this document to those*
*records.*
