# SELECTION CENSUS -- what the production feature-selection path actually does

**Plan 30-17, Task 1. Measured 2026-08-22 on the owner-accepted rung-4 gold.**

This document exists because two prior measurements of "does the count of information-free
columns change what `SelectFromModel` selects" disagreed, and a behavioural change was about to
be made on the strength of one of them. It reports what the production path does, measured by
execution, before anything was changed.

Regenerate with:

```
uv run python -m scripts.selection_census
```

The machine-readable form is written to `outputs/selection/selection_census.json`. That
directory is gitignored (`.gitignore:26`), which is why the instrument and this readout are the
tracked homes for the result -- the same reason `tests/phase30_state.py` holds Plan 30-08's
expectations rather than `outputs/`.

---

## 0. What was measured on

| | |
|---|---|
| `data/gold/features_wp.parquet` | 194 cols, 6,499 rows, sha256 `d5fe93b16c0b9aac...` |
| `data/gold/features_ats.parquet` | 195 cols, 6,499 rows, sha256 `a08daa8de0f87b46...` |
| `data/gold/features_ou.parquet` | 194 cols, 6,499 rows, sha256 `209c5971a669fa56...` |
| Temporal config | `TemporalSplitConfig.default()` -- train 2018-2019, hp-val 2020, holdout 2021-2024 |
| scikit-learn | 1.8.0 (the installed version, read from source, not recalled) |

**This is a THIRD state of the frame, and the numbers below are not comparable to earlier ones.**
Plan 30-07's rung-3 drop removed the fifteen `line_movement` columns (209/210/209 -> 194/195/194)
and Plan 30-08's re-sync added 236 rows to season 2025 (6,263 -> 6,499). Any count from Plan
30-15 or earlier was taken on a frame that no longer exists. Nothing below is compared against
one.

## 1. Which call site runs, and what is binding

Determined by instrumenting `BaseTrainer.select_features` and running the REAL production entry
(`trainer.train_and_evaluate(gold, None, tune=False)` -- the call `backtest.signal_lift`'s
`_walkforward_clv` makes) once per target, recording the caller frame.

Exactly **one** `select_features` call occurs per production run, per target:

| Target | Trainer | Call site | Budget constant | Cap |
|---|---|---|---|---|
| WP | `WPTrainer` | `models/trainers/wp_trainer.py:263` in `train_and_evaluate` | `_WP_MAX_FEATURES` | 20 |
| ATS | `ATSTrainer` | `models/trainers/ats_trainer.py:210` in `train_and_evaluate` | `_ATS_MAX_FEATURES` | 25 |
| O/U | `OUTrainer` | `models/trainers/ou_trainer.py:213` in `train_and_evaluate` | `_OU_MAX_FEATURES` | 25 |

`models/trainers/base.py`'s own `train_and_evaluate` -- the one that passes `max_features=None` --
was **never** reached. All three concrete trainers override it. See section 6.

The selection window is identical for all three targets: **534 rows** (seasons 2018-2019) and
**180 feature candidates** after `WalkForwardSplitter._feature_cols` removes the ID columns, the
target and the six `raw_*` display columns.

| Target | Estimator | Candidates | Constant in window | Resolved threshold | Clear threshold | Selected | **Binding** |
|---|---|---|---|---|---|---|---|
| WP | `LogisticRegression` | 180 | 82 | 0.11554559502374163 | 56 | 20 | **the cap** |
| ATS | `XGBRegressor` | 180 | 82 | 0.0055555556900799274 | 85 | 25 | **the cap** |
| O/U | `XGBRegressor` | 180 | 82 | 0.0055555556900799274 | 91 | 25 | **the cap** |

**82 of 180 candidates -- 46% -- are zero-variance over the window the selector fits on.** That
is pre-existing (D30-DEFER-09) and is not this plan's to fix. It is why selection stability is a
live question here rather than an academic one.

**The threshold is a no-op today, for every target.** sklearn's rule (verified against the
installed source, `_from_model.py::_get_support_mask`) is an INTERSECTION: take the top
`max_features` by importance, then drop anything scoring below the threshold. When more features
clear the threshold than the cap admits -- 56 >= 20, 85 >= 25, 91 >= 25 -- the top-K are all
above it by construction and the threshold removes nothing. Production selection is therefore
**already effectively pure top-K**, and has been. The XGBoost threshold reproduces `1/180 =
0.00555555555...` to float32 precision, which is what the mean of gain importances normalised to
sum to 1 must be over 180 candidates -- a useful cross-check that the recorded threshold is the
mean and not something else.

## 2. The padding experiment -- the decisive measurement

Neither prior measurement performed it. For each target the input frame was padded with N
synthetic information-free columns and the production selection re-run. Three kinds, because
they ask three different questions: `constant` is zero-variance at 0.0; `constant_nonzero` is
zero-variance at 1.0 (strictly stronger for the linear target, since a column of zeros cannot
contribute to a linear score whatever coefficient it is given); `noise` is unit-variance and
independent of the target.

Numbers are the **symmetric difference against the unpadded selected set** -- how many of the
selected features changed identity. WP selects 20, ATS and O/U select 25.

| Target | const N=10 | const N=25 | const N=50 | const-nonzero N=10 | N=25 | N=50 | noise N=10 | N=25 | N=50 |
|---|---|---|---|---|---|---|---|---|---|
| WP | **0** | **0** | **0** | **0** | **0** | **0** | 4 | 4 | 8 |
| ATS | 10 | 16 | 16 | 10 | 16 | 16 | 16 | 20 | 20 |
| O/U | 18 | 14 | 14 | 18 | 14 | 14 | 18 | 20 | 32 |

Two controls make these readable. The isolated selection route used for the grid was proven to
reproduce the full production call exactly (same selected list, all three targets), and repeating
it on an unchanged frame reproduces its own result exactly -- so a moved set is count-dependence,
not run-to-run noise.

Under noise padding, synthetic columns are not merely perturbing the fit, they are **being
selected**: 1 of WP's 20 at N=50, up to 5 of ATS's 25 and 8 of O/U's 25.

## 3. The ablation -- the direction Phase 30 actually moved in

Padding asks what happens when dead columns arrive. Plan 30-07 REMOVED fifteen columns, so the
removal direction has the production precedent. Two removals were measured: fifteen
window-constant columns (sized to match rung 3) and all 82.

| Target | drop 15 constants (180 -> 165) | drop all 82 constants (180 -> 98) |
|---|---|---|
| WP | **0** | **0** |
| ATS | 8 | 18 |
| O/U | 16 | 10 |

No target currently selects a column that is constant over its own fit window, so nothing in the
ablation is explained by a dead column having been selected and then removed.

## 4. Verdict

**Production feature selection IS count-dependent, and the threshold is not the mechanism.**

* **WP is invariant to zero-variance columns** -- adding 50 or removing all 82 does not move a
  single one of its 20 features. It is sensitive only to noise columns, which have variance and
  can legitimately compete.
* **ATS and O/U are strongly count-dependent on zero-variance columns.** Adding ten columns that
  are literally constant changes 5 of ATS's 25 selected features and 9 of O/U's 25.

The mechanism is the **estimator refit**, not the threshold. `select_features` fits its scoring
model on whatever columns are present, so the column set changes the fit itself. The two XGBoost
targets run `colsample_bytree=0.8` and `subsample=0.8`: columns are resampled per tree, so adding
even a zero-variance column changes which columns each tree sees, and hence the gain importances
of the REAL features. WP's `LogisticRegression` has no such sampling, which is exactly why WP is
invariant to constants and ATS/OU are not.

**This refutes the fix the plan expected to make.** Pure top-K (`threshold=-inf` with the cap)
would change nothing here -- the cap already binds for all three targets, so the rule is already
top-K in effect. A rule change cannot remove a dependence that lives in the fit that produces the
scores the rule ranks.

## 5. Reconciling the two prior measurements

They are not two readings of one quantity. They sat on opposite sides of sklearn's intersection.

**Plan 30-15's `86 -> 89`** called `ATSTrainer.select_features` DIRECTLY with
`max_features=None`, on the 2018-2019 train leg of the frozen Plan 30-03 fixture (210 cols /
6,263 rows): 201 -> 86 selected before the `raw_*` exclusion, 195 -> 89 after. With no cap the
threshold is the ONLY filter, so the selected count IS the number of features clearing the mean,
and it moves whenever the mean moves. That measurement was correct about the uncapped rule. Re-run
on today's gold the same path returns **85 of 180 for ATS**, 56 for WP and 91 for O/U -- which are
precisely the clearing counts in section 1, as they must be.

**The orchestrator's counter-reproduction** ran ATS/`home_margin` capped at 25 at different
hyperparameters, found only two features clearing the mean threshold at all, and measured a top-25
symmetric difference of ZERO. When only two features clear, the cap never speaks: the intersection
returns those two whatever K is, and the top-25 candidate set is irrelevant. That measurement was
correct about the regime it was in -- a regime where the THRESHOLD binds, the opposite of
production's.

**Neither describes production**, which is the intersection of both and, on this gold, sits in the
cap-binding regime for every target. Section 1 measures that intersection directly.

**Both are consistent with the same underlying fact**, which neither could see: the count
sensitivity is in the fit. The uncapped probe made it visible as a moving clearing-count; the
capped counter-reproduction happened to land where two dominant features swamped everything and
the reordering had nothing to reorder.

**The D30-DEFER-13 production observation is only partly reproduced, and that is reported rather
than smoothed.** Plan 30-07 removed fifteen window-constant columns (195 -> 180 candidates) with
`_WP_MAX_FEATURES=20` unchanged and a WP `headline_clv` anchor moved +0.00103305 -> -0.00282341.
But WP selection here is invariant to exactly that move -- symmetric difference 0 for a
fifteen-constant drop AND for the full 82-column ablation. Either WP's selected set did not move at
rung 3 and the metric moved for some other reason, or some of the fifteen `line_movement` columns
were not constant over 2018-2019. The columns no longer exist in gold, so this census cannot close
it, and it does not claim to. What it does establish is that the fifteen-column removal DOES move
ATS (8 of 25) and O/U (16 of 25).

## 6. The `base.py` fallback asymmetry

`models/trainers/base.py:464` calls `select_features` with no `max_features`, i.e. uncapped --
a genuinely different selection rule from the three production overrides. On today's gold it
would select 56 / 85 / 91 features instead of 20 / 25 / 25.

**It is unreachable on any production path.** `BaseTrainer` is an ABC with five abstract methods,
so it cannot be instantiated; all three concrete subclasses -- the only ones in the codebase, and
the only values in `backtest/engine.py`'s `_TRAINER_MAP` and `backtest/signal_lift.py`'s
`_TRAINER_FOR` -- override `train_and_evaluate` and pass their own budget. The instrumented
production runs confirm it: the caller frame was a concrete trainer every time, and the base
method was never entered. The single other `BaseTrainer` subclass in the repository is
`tests/unit/test_promote_models_tuned_path.py`'s `_DummyTrainer`, which exercises
`tune_hyperparameters` and never calls `train_and_evaluate`.

It is documented at the call site rather than deleted, because deleting a base-class method to
resolve an inconsistency that has never fired is a larger change than the inconsistency warrants.

---

*Instrument: `scripts/selection_census.py`. Machine-readable output:
`outputs/selection/selection_census.json`.*

---

# ADDENDUM -- what Task 2 changed, and the after-measurement

**Plan 30-17, Task 2. Measured 2026-08-22 immediately after the change, same gold, same
config, same instrument.**

## 7. The change

Two edits in `models/trainers/base.py`, both inside `select_features`, neither touching an
estimator, a hyperparameter, or a per-target budget:

1. **A zero-variance pre-filter on the fit input.** `informative_columns(X)` withholds every
   column that does not vary over the fit window (all-NaN counts as constant) before the
   scoring model is fitted. A wholly-constant frame falls back to fitting on the frame as it
   stands rather than on nothing.
2. **`threshold="mean"` stated explicitly.** This is the rule `threshold=None` already
   resolved to for an L2 `LogisticRegression` and an `XGBRegressor`, so it is a documentation
   change and not a behavioural one -- but stating it means a future switch to an L1 penalty
   cannot silently re-resolve the rule to `1e-5` without someone deciding to.

`base.py`'s own `train_and_evaluate` was NOT changed. Its `max_features=None` asymmetry is
documented at that call site with the measurement that shows it unreachable, and a test now
fails if any concrete trainer stops overriding it.

**Why a pre-filter and not the pure top-K the plan expected.** Section 4 measured the cap
already binding for every target, so `threshold=-inf` would have been a no-op, and the
dependence lives in the fit rather than in the rule that ranks the fit's output. A pre-filter
removes the dependence at its source: identical post-filter input gives an identical fit, so
the invariance is exact by construction rather than tuned. Two options were considered and not
taken -- fitting the selection model with `colsample_bytree=1.0` (prohibited: it changes the
estimator), and replacing model-based selection with a univariate filter (architectural, and
far beyond what this plan authorises).

## 8. Count-independence, after

Same grid, same instrument, re-run on the changed code.

| Target | const N=10/25/50 | const-nonzero N=10/25/50 | drop 15 | drop all 82 | noise N=10/25/50 |
|---|---|---|---|---|---|
| WP | 0 / 0 / 0 | 0 / 0 / 0 | **0** | **0** | 4 / 4 / 8 |
| ATS | **0 / 0 / 0** | **0 / 0 / 0** | **0** | **0** | 18 / 22 / 20 |
| O/U | **0 / 0 / 0** | **0 / 0 / 0** | **0** | **0** | 22 / 16 / 26 |

Every zero-variance cell is now exactly zero, in both directions, for all three targets.
`count_dependent_on_zero_variance_columns` is `false` for wp, ats and ou.

**Noise sensitivity is unchanged and is not claimed to be fixed.** A column with variance but
no relationship to the target is indistinguishable from a weak real signal at fit time. No
pre-filter can exclude it without looking at the target, and doing so would be a different
selection rule. It is reported here and pinned as a boundary in
`tests/unit/test_feature_selection_stability.py` rather than asserted away.

The cap still binds after the change -- 36 / 35 / 48 features clear the threshold against caps
of 20 / 25 / 25 -- so the threshold remains a no-op and the rule remains top-K in effect.

## 9. What actually moved, by name

The change is behaviour-preserving for WP and moves 9 of ATS's 25 and 5 of O/U's 25.

**Cross-check:** the post-change selection is byte-identical to the PRE-change 82-column
ablation for all three targets, which is what a correct pre-filter must produce and is
therefore a real check rather than a restatement.

**WP -- unchanged.** Symmetric difference 0. All 20 features identical.

**ATS -- 9 out, 9 in (symmetric difference 18 of 25 selected).**

| Removed | Added |
|---|---|
| `away_def_rolling_opp_adj_epa_per_play` | `away_off_bye` |
| `away_eastward_travel` | `away_off_rolling_opp_adj_pass_epa` |
| `away_elo_rank` | `away_rolling_snap_share_te` |
| `away_off_rolling_opp_adj_epa_per_play` | `away_rolling_snap_share_wr` |
| `away_rolling_snap_share_db` | `away_timezone_diff_hours` |
| `home_look_ahead_spot` | `away_travel_fatigue_score` |
| `home_off_rolling_opp_adj_epa_per_play` | `home_backup_quality_delta` |
| `home_off_rolling_opp_adj_pass_epa` | `home_def_rolling_opp_adj_rush_epa` |
| `home_qb_out_flag` | `home_rolling_snap_share_rb` |

**O/U -- 5 out, 5 in (symmetric difference 10 of 25 selected).**

| Removed | Added |
|---|---|
| `away_backup_quality_delta` | `away_travel_fatigue_score` |
| `away_look_ahead_spot` | `home_def_rolling_opp_adj_epa_per_play` |
| `away_off_bye` | `home_look_ahead_spot` |
| `home_def_rolling_opp_adj_pass_epa` | `home_snap_concentration` |
| `home_rolling_snap_share_dl` | `venue_elevation_ft` |

This is a REAL behavioural change to which features ATS and O/U train on, landed deliberately
before Plan 30-09's baseline re-freeze and Plans 30-10 and 30-11's binding measurements -- the
only point at which it is cheap. Nothing was tuned, re-fitted or promoted:
`artifacts/latest.json` and all three gold parquets are byte-unchanged by sha256.

## 10. Re-running this document

`uv run python -m scripts.selection_census` now regenerates the AFTER state. The BEFORE census
is preserved alongside it as `outputs/selection/selection_census_before_fix.json`, with the
after state at `outputs/selection/selection_census_after_fix.json`.
