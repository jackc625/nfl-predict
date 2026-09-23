# BLEND-TUNING-READOUT.md -- one fixed blend weight per target, tuned on lines we owned

**Phase 33.2, Plan 33.2-24. Owner decisions D33.2-03, D33.2-09, D33.2-10. Measured 2026-09-23.**

**Built on re-measured past seasons; not clean evidence.** Only the 2026 season, recorded live
under the day-before 6 PM ET lock, counts (D33.2-07). What follows sets how much of the market's
opinion is mixed into each published prediction. It says nothing whatever about profitability.

---

## 1. What the blend does, in plain English

Every prediction the site publishes is a MIX of two opinions: the model's, and the betting
market's. The blend weight says how much of the model to keep. A weight of 1.00 means "publish
the model alone"; 0.00 means "publish the market alone"; 0.60 means "60 parts model, 40 parts
market".

Two things changed in this plan.

**One fixed weight per target.** The blend used to change its weight through the season, from a
formula with six tuned numbers. That formula was fitted on invented model predictions laid over
2010-2017 closing lines, and the check that kept it was itself scored against closing lines. A
closing line is the price just before kickoff; it did not exist when a prediction had to be made,
so every input to that formula is dead (D33.2-03, D33.2-07). It is retired, and there is now
exactly one weight per target, the same in week 1 as in week 18.

**The market's opinion comes from a line we owned before each game's lock.** The weights below
were tuned on the owned line history -- the spreads and totals the project bought, each stamped
with the real time it was captured -- taking, for each game, the latest line captured at or
before 6 PM ET on the day before kickoff. Never a closing line. For the win model, that spread is
turned into a win probability by the fitted converter (Plan 33.2-21), because the owned history
carries no moneyline at all.

## 2. The three fixed weights

| Target | Weight | Loss minimised | Loss at the weight | Market alone (w=0) | Model alone (w=1) | Games | Seasons |
|---|---|---|---|---|---|---|---|
| WP | 0.00 | log loss | 0.613838 | 0.613838 | 0.645184 | 1,091 | 2021-2024 |
| ATS | 0.00 | mean absolute error (points) | 9.780275 | 9.780275 | 10.430835 | 1,346 | 2020-2024 |
| O/U | 0.12 | mean absolute error (points) | 10.287936 | 10.292533 | 10.879621 | 1,346 | 2020-2024 |

The blend artifact is `blend_20260923_192155`. The weights were searched over every hundredth
from 0.00 to 1.00, and each target's weight is the one that made the blend's predictions closest
to what actually happened, measured in that model's own metric: log loss for win probability,
average absolute miss in points for the spread and the total.

## 3. What the weights mean -- two of them are a finding

**For win probability and for the spread, the market alone did better than any mix.** Both
weights landed at exactly 0.00, the edge of the search. That is a result, reported as one, not a
number the search was forced into: the old tuner searched only 0.50 to 0.70 and could never have
said this.

The reason is visible in the models' own predictions. On the 1,346 owned games, the spread
model's predictions line up with the market's spread closely (correlation 0.77) and track the
real margin less well than the spread does (0.33 against 0.44). Whatever the model knows, the
market already priced in, and the market knows more besides. The win model's predictions track
the result at 0.30. These are the corrected models -- no betting line among their inputs
(D33.2-03), default settings because no search beat its random baseline (Plan 33.2-23) -- and on
lines owned before the lock they add nothing the market did not already carry.

**For the total, the model gets a small weight, 0.12, and it barely matters.** It improves the
average miss by 0.005 points over the market alone -- far smaller than the gap between one season
and the next. No significance test was run on it and none is claimed.

Seen season by season, the weight each season would have chosen on its own:

| Target | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| WP | (not tuned) | 0.25 | 0.00 | 0.00 | 0.16 |
| ATS | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 |
| O/U | 0.00 | 0.46 | 0.00 | 0.00 | 0.40 |

Most seasons choose the market alone. The fixed weight is one number across all of them, which
is what D33.2-10 rules; a week-varying or season-varying shape may return only with evidence
measured under the new rule.

What this does to the published numbers: with the WP and ATS weights at 0.00, the blended win
probability and the blended spread shown for a game equal the market's own opinion. The model's
raw predictions are still produced and still shown beside them.

## 4. How the weights were fitted

**The corpus.** The owned line history (`odds_timeline`, 2020-2024). For every scheduled game in
those seasons, the LATEST line captured at or before that game's lock -- a line captured exactly
at the lock counts, one captured a second later does not. Where two lines carry the same capture
time, the later-recorded one is used, and a game can never contribute two rows. For 1,104 of the
1,346 games the line used was captured between 24 and 48 hours before the lock; the median is
24.1 hours. A game with no such line is left out and counted (section 5). Nothing is ever filled
in for a missing line.

**The model side: the corrected models' real walk-forward predictions.** The three corrected
models are the Plan 33.2-23 re-fit -- `wp_20260923_115808`, `ats_20260923_124120` and
`ou_20260923_133813` -- read by their recorded ids, not through the production pointer. Each
season was predicted by refitting that model's recipe as if the season were the latest one
finished, using the project's one season-partition rule:

| Predicted season | Features selected on | Calibration fitted on | Model fitted on |
|---|---|---|---|
| 2020 | 2002-2017 | 2018 | 2002-2019 |
| 2021 | 2002-2018 | 2019 | 2002-2020 |
| 2022 | 2002-2019 | 2020 | 2002-2021 |
| 2023 | 2002-2020 | 2021 | 2002-2022 |
| 2024 | 2002-2021 | 2022 | 2002-2023 |

Nothing that shaped a season's prediction saw that season. The shipped models' own feature lists
were deliberately NOT reused: they were chosen on 2002-2022, so reusing them for a 2020
prediction would let the selection see the season it predicts. What is reused is the recipe --
its settings, its excluded feature families and its selection rule.

**The market side.** For the spread and the total, the market's opinion is the owned line
itself. For win probability it is that spread converted to a probability OUT OF FOLD: each
season uses a conversion slope fitted only on EARLIER seasons (2021: 0.158554, 2022: 0.135658,
2023: 0.142952, 2024: 0.142728). The converter's final slope (0.151244), which the blend binds
for serving, was fitted on all five seasons including the game being scored; using it here would
have let each game's own result shape the market side it is compared against.

**Reproducibility.** Every fit was pinned to one processor thread and the value is recorded in
the artifact. The whole fit was run twice in two separate processes -- once as a dry run that
wrote nothing, once for real -- and produced identical weights, losses and per-season weights to
the last digit. The gold both runs read is generation
`484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597`, the same generation the three
models were fitted on; the fit refuses to run on any other.

## 5. The exclusions -- every game accounted for

1,408 games were scheduled in 2020-2024. 1,346 are in the corpus. The other 62 are left out, for
one reason, and one more class leaves the win-probability fit alone:

| Class | Games | Leaves |
|---|---|---|
| no_prelock_line | 62 | all three fits |
| no_prior_fold_converter | 255 | the WP fit only |

**`no_prelock_line` -- 62 games with no owned line at or before their lock.**

- 59 postseason games with no owned line at all: 24 wild-card, 20 divisional, 10 conference
  championship and 5 Super Bowl games. The owned history carries postseason lines only for the six
  2024 wild-card games, which are in the corpus.
- `2020_W05_BUF@TEN` -- regular season. The owned history holds lines for it, but every one was
  captured after its lock (the game was moved to a Tuesday during the 2020 COVID reschedules).
- `2024_W17_KC@PIT` and `2024_W17_BAL@HOU` -- the two 2024 Christmas Day games. The owned history
  DOES hold lines for both, some captured before the lock, but they are filed under the ids
  `2024_W16_KC@PIT` and `2024_W16_BAL@HOU` -- week 16, where the schedule has them in week 17 -- so
  they do not join. They are left out rather than re-keyed here, because the same history also
  feeds the fitted converter, and the two must read the same games. The mis-filing is recorded for
  correction.

**The six owned-history ids that name no scheduled game.** They are not exclusions -- no scheduled
game is missing because of them -- but they are listed so none is dropped silently:

- `2024_W16_KC@PIT`, `2024_W16_BAL@HOU` -- the two Christmas Day games above, filed a week early.
- `2022_W17_BUF@CIN` -- the game suspended and never completed in January 2023.
- `2024_W19_DET@LA`, `2024_W19_LAC@BAL`, `2024_W19_PIT@HOU` -- wild-card pairings priced on
  1-3 January 2025, before the final week settled the bracket. None of them was played.

**`no_prior_fold_converter` -- 255 games, the WP fit only.** Every 2020 game in the corpus. 2020 is
the first season of the owned history, so no earlier season exists to fit an out-of-fold
conversion slope on. Converting these games with the final slope instead would be the in-sample
leak described in section 4, so they are left out of the win-probability fit -- counted, never
filled with the final slope or a neighbouring season's. The spread and total fits keep them,
because those blend the line itself and never go through the converter.

## 6. The known gap: 2025

No free source has pre-lock lines for 2025, so the blend and the bets cannot be checked on 2025 at
lock time. This is a known and accepted limitation, stated to the owner when D33.2-03 was ruled,
not an oversight -- and it is why the tuning span is 2020-2024 and stops there. The owned history
begins in 2020 because the historical line service begins in mid-2020, and the forward capture
that would have covered 2025 never ran.

## 7. What is NOT here

There is no comparison against the retired week-varying blend. It was fitted on invented
predictions over closing lines and is not a comparator (SPEC R13, D33.2-07): a number measured
against it would say nothing about the new blend. Nor is there any comparison against earlier
models; the three blended here are the corrected ones and nothing older enters.

There are no edge thresholds here either. The blend artifact carries the blender's configured
thresholds unchanged, labelled as not calibrated by this fit; the 2026 thresholds are derived
separately (Plan 33.2-26).

## 8. What happens next

The new blend is written but NOT switched on: the production pointer still names the retired
week-varying blend, which both the prediction script and the website cache now REFUSE by name.
Until the production swap (Plan 33.2-25) installs this blend alongside the three corrected models,
a prediction or cache run that reaches the blend fails loudly rather than publishing a number
neither the old rule nor the new one chose. The two plans run back to back.

**Built on re-measured past seasons; not clean evidence.** Only the 2026 season, recorded live
under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2.

---

*Generated from the run of `python -m backtest.tune` on 2026-09-23.*
*The artifact it wrote is `artifacts/blend_20260923_192155/blend_weights.json`; its witness is the*
*`P332_24_*` block in `tests/phase33_state.py`, and `tests/unit/test_blend_tuning_readout_md.py`*
*holds this document's exclusion counts equal to that witness.*
