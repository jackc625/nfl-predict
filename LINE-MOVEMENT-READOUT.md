# Line-Movement Budget Gate Readout (Phase 29, SIG-04)

**Status:** GO/NO-GO record -- this is a budget-gate spike, not a build and not a gate ruling.
Nothing here is shipped; the line-movement signal is screened / carried to Phase 30 for the
binding gate, never described as a finished outcome.
**Scope:** price the historical-odds cost as a tiered menu (D-03), state the D-02 plausibility
(non-redundancy) argument, and record the owner's live branch decision (full backfill |
forward-collect-only | slip) BEFORE any build runs (SC1).
**Cost method (D-04):** every credit/dollar figure below is ESTIMATED from The Odds API
published per-credit pricing x the planned call count. The owner is on a FREE key today and the
historical endpoint is PAID-only; the free key 401s on `historical/`, so the endpoint is NOT
called here. The estimate is arithmetic on the VERIFIED credit formula
(`cost = 10 x markets x regions` per snapshot timestamp).
**Honesty framing (D-16):** the readout says "screened / carried to Phase 30". Phase 29 SCREENS
(on the backfill path only); Phase 30 runs the binding deploy gate. Forward-collect-only and slip
are FIRST-CLASS SUCCESS outcomes (D-14, SC3), not "incomplete" phases.

---

## 1. Tiered cost (estimated from The Odds API published pricing x planned calls; free key, historical endpoint not called -- D-04; pricing/formula/earliest-date checked 2026-06-29)

The historical credit formula is `cost = 10 x markets x regions` per snapshot timestamp
requested [VERIFIED against the-odds-api.com/liveapi/guides/v4, checked 2026-06-29]. One
historical call at a timestamp `T` returns the WHOLE NFL board at `T` -- you pay per
(timestamp x markets x regions), NOT per game (assumption A2). The earliest available historical
date is **2020-06-06** [VERIFIED, checked 2026-06-29]; the paid-plan requirement (free tier
excludes historical) is re-confirmed [VERIFIED, checked 2026-06-29]. The cheapest plan that fits
every tier below is the **20K plan at $30/month** (20,000 credits) [VERIFIED pricing, checked
2026-06-29]; the free plan is 500 credits/month and the next paid rung is 100K at $59/month.

**Call-count assumptions (flagged -- arithmetic on the VERIFIED formula):**
- A1: ~22 NFL weeks/season (18 regular + ~4 playoff weekends); full-trajectory cadence = 4
  snapshots/week (open ~Tue, intraweek Wed/Thu, late Thu/Fri, freeze Fri 6 PM ET); two-anchor
  cadence = 2 snapshots/week (open + freeze).
- A2: one historical call per timestamp returns the whole board (pay per timestamp, not per game).
  If the API instead paginated per-event the cost would rise sharply; the docs describe a
  board-wide snapshot, so the risk is low. If the owner proceeds on the backfill path, confirm A2
  with exactly ONE paid historical call (~10-20 credits) before the bulk pull.
- regions=us -> 1 region. Windows: **2020-2024** (~110 wk, adds the 2021 walk-forward training
  fold) or **2021-2024** (~88 wk, holdout-only). 2020 is fully covered since the season starts
  after 2020-06-06.

| Tier | Markets | Snaps/wk | Window | Snapshots | Credit cost (`10 x mkts x reg x snaps`) | $ (cheapest plan that fits) |
|------|---------|----------|--------|-----------|------------------------------------------|------------------------------|
| (a) full trajectory + spread+totals | 2 | 4 | 2020-2024 (110 wk) | 440 | 440 x 20 = **8,800** | $30 (20K), one month |
| (a) full trajectory + spread+totals | 2 | 4 | 2021-2024 (88 wk)  | 352 | 352 x 20 = **7,040** | $30 (20K), one month |
| (b) full trajectory + totals only   | 1 | 4 | 2020-2024 (110 wk) | 440 | 440 x 10 = **4,400** | $30 (20K), one month |
| (b) full trajectory + totals only   | 1 | 4 | 2021-2024 (88 wk)  | 352 | 352 x 10 = **3,520** | $30 (20K), one month |
| (c) two-anchor + totals only        | 1 | 2 | 2020-2024 (110 wk) | 220 | 220 x 10 = **2,200** | $30 (20K), one month |
| (c) two-anchor + totals only        | 1 | 2 | 2021-2024 (88 wk)  | 176 | 176 x 10 = **1,760** | $30 (20K), one month |

**Cost conclusion:** every tier -- including the richest (a)/2020-2024 at 8,800 credits -- fits
inside a SINGLE $30 month of the 20K plan with large headroom (even doubling the cadence to 8
snaps/week keeps Tier (a) at ~17,600 < 20,000). Adding the spreads market doubles credits but
stays in the same $30 plan, so the D-07 "spread conditional" gate is "yes, fold it in" on cost
grounds; the only reason to drop spreads would be payload/complexity, not credits. The forward
-collect path needs NO paid plan at all -- forward capture uses the REGULAR endpoint
(`cost = markets x regions`, ~1-2 credits/call, ~35 credits/month at 4 snaps/week x 2 markets),
comfortably within the free 500/month.

---

## 2. Plausibility (D-02): is line-movement non-redundant vs the freeze anchor + the Phase-28 injury signal?

**Cost is NOT the binding constraint -- plausibility is.** Section 1 shows every tier is a
trivial single $30 month, so the go/no-go pivots entirely on whether line-movement carries
information the model does not already have.

**vs the freeze market anchor (already a model feature).** The Friday-6 PM-ET freeze line is
ALREADY consumed as a feature in `features/market_anchors.py` (`MarketAnchorFeaturesCalculator`,
the `snapshot_total` / `snapshot_spread` anchors). That calculator even emits
`total_movement` / `spread_movement` columns today -- but with a single synthetic snapshot per
historical game, open == snapshot, so those movement columns are identically **0.0 for every
historical game** (a structurally dead column). Net drift (`freeze - open`) is therefore largely
drift against a line the model ALREADY sees: the freeze level is not new. The genuinely-new
information line-movement adds is (i) the **opening total LEVEL** (where the line started, before
it drifted to the known freeze) and (ii) the **path shape** (late/steam movement, reversals,
distance traveled). Net drift on its own is close to feeding the open alongside the already
-present freeze anchor; the level and the path are what could earn their place.

**vs the Phase-28 injury signal.** Phase 28 already ingests injury / snap-count / situational
signals into the widened gold. The market moves the total on the SAME injury news the Phase-28
features encode -- so part of any measured line-movement signal is the market's reaction to
information the model can already read directly from the injury feed. This is the priced-in
risk: buying a signal that is a noisier proxy for features already present.

**Plausibility verdict.** Line-movement is PLAUSIBLY non-redundant ONLY through the
opening-level + path-shape families (D-09 ii-iv), not through net drift against the freeze line.
This is a genuine but BOUNDED edge hypothesis -- exactly the kind the add-one-in CLV lift screen
(backfill path, Plan 29-07) exists to arbitrate. Per D-10 the readout records this priced-in
risk plainly and does NOT pre-decide the lift; the screen, not optimism, rules. The honest
framing carried throughout: if the owner backfills, the signal is screened and carried to
Phase 30, never asserted as an edge on the basis of this spike.

---

## 3. Decision: full backfill | forward-collect-only | slip (+ rationale)

<!-- Machine-readable branch marker. Downstream branch gating (Plans 29-02..29-08) reads this
literal token, not the prose. Task 2 (the blocking owner checkpoint) overwrites PENDING with
exactly one of: full-backfill | forward-collect-only | slip. -->
selected_branch: full-backfill

**Status:** DECIDED at the Task-2 blocking owner checkpoint -- the owner selected
**full-backfill** (build + screen) with the tiered cost menu (Section 1) and the D-02
plausibility argument (Section 2) in hand. SC1 is satisfied: the budget-gate decision is
recorded HERE, before any 29-02+ build runs.

**Owner rationale (full-backfill):**
- **Cost is not the binding constraint (D-01).** Every tier in Section 1 -- including the
  richest (a)/2020-2024 at 8,800 credits -- is a trivial single $30 month of the 20K plan
  with large headroom. With NO pre-registered dollar ceiling (D-01), the spend does not gate
  the decision.
- **The D-02 plausibility edge is genuine but bounded.** Line-movement is plausibly
  non-redundant only through the opening-total LEVEL and the path shape; net drift against the
  already-known freeze line is largely what the model already sees (the freeze anchor is a
  feature in `features/market_anchors.py`), and part of any measured movement is a noisier
  proxy for the Phase-28 injury signal the market reacts to. The edge is real enough to be
  worth measuring, not strong enough to assume.
- **The add-one-in CLV lift screen is the arbiter.** Because the edge is bounded, the owner
  declines to pre-decide it. The backfill path's in-process walk-forward CLV lift screen
  (Plan 29-07) measures the per-target line-movement lift against the activated baseline; that
  screen, not optimism, rules. The line-movement signal is SCREENED in Phase 29 and CARRIED TO
  PHASE 30 for the binding deploy gate -- never asserted as an edge on the basis of this spike.
- **Paid-key provisioning (D-04).** The owner will provision a paid ODDS_API_KEY for a
  one-month historical pull (one-month-paid-then-downgrade; not pre-committed). Plan 29-05
  pauses at its OWN blocking checkpoint before any paid historical call is made, so the bulk
  spend is gated a second time at execution.

**Downstream routing:** the full-backfill branch runs Plans 29-02, 29-03, 29-04, 29-05, 29-06,
and 29-07. (Both are now filled: Section 4 carries the Plan 29-07 lift results, and Section 5
records the honest scope-down of the verdict's strength.)

**The three branches:**
- **full-backfill** -- buy the 2020-2024 historical archive (one paid month, then downgrade --
  D-04, not pre-committed), build `LineMovementBuilder`, run the add-one-in CLV lift screen, and
  carry a measured per-target lift verdict to Phase 30. Runs Plans 29-02, 29-03, 29-04, 29-05,
  29-06, 29-07. Caveat: the 2018-2019 train window has ZERO coverage (floor 2020-06-06) ->
  neutral defaults + a coverage flag; the D-02 redundancy risk may make the lift flat.
- **forward-collect-only** -- ship the snapshot-capture pipeline now and wire the recurring
  intraweek capture into the Friday orchestrator (D-06) to accumulate 2025+ trajectories; fits
  the FREE tier. Infra lands, the lift screen slips honestly to a later milestone with a written
  caveat. Runs Plans 29-02, 29-03, 29-08.
- **slip** -- honest deferral with this written record; zero spend; a first-class success state
  (D-14). Runs only this plan (29-01).

---

## 4. Lift results (backfill path only): per-target paired CLV delta, D-05 keep/drop

**METHOD.** The lift is an add-one-in paired incremental-CLV screen run IN PROCESS by
`backtest/signal_lift.py` (the Phase-28 convention). Each leg is a per-season walk-forward re-fit
via `BaseTrainer.train_and_evaluate(tune=False)` -- train `<= Y-1`, measure `Y` -- so every CLV
reading is OUT OF SAMPLE. It is never a whole-frame re-score of the currently-serving artifacts,
which would score the holdout in sample. The two legs differ by exactly the line-movement family,
and the per-game delta is merged on `game_id`, so the comparison is PAIRED. Significance comes
from the canonical `clv_significance` / `CLV_COLUMN_FOR` / `SIGNIFICANCE_ALPHA` primitives in
`backtest/diagnose.py`; nothing is re-derived here.

**The baseline INCLUDES the kept Phase-28 groups.** The screen runs with
`baseline_exclude_groups=('line_movement',)`, so injury / snap / situational stay in the baseline
leg and the delta is line-movement incremental to the POST-Phase-28 feature set. That is what
makes the Section-2 D-02 question ("is line-movement just a noisier proxy for the injury news the
market is reacting to?") answerable at all. The baseline also still contains the Friday-6PM-ET
freeze anchor (`snapshot_total` / `snapshot_spread`) and the pre-existing, historically
identically-0.0 `total_movement` / `spread_movement` columns.

**Archive coverage actually joined into gold** (regular-season games, Plan 29-06):

| Season | Covered | Note |
|---|---|---|
| 2020 | **0 / 262** | the whole 2020 archive is orphaned by a `game_id` off-by-one (D29-06-01): every 2020 timeline id is one week high. The upstream `get_nfl_season_start` derivation was corrected in `45bff24`, but the STORED rows were written under the old rule and have not been re-keyed. ~1,780 paid rows contribute nothing today. A re-key needs NO further paid calls. |
| 2021 | 261 / 272 | |
| 2022 | 257 / 271 | |
| 2023 | 271 / 272 | |
| 2024 | 269 / 272 | |

### 4a. The canonical screen (pre-registered): `python -m backtest.signal_lift --phase 29`

Canonical temporal config -- train 2018-2019, hp-val 2020, measure 2021-2024; 1,019 paired games.

| Target | Paired CLV delta | t | p | Group columns the model used | Covered span |
|---|---|---|---|---|---|
| WP | +0.000000 | n/a | n/a | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |
| ATS | +0.147814 | +1.651 | 0.09904 | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |
| OU | -0.221188 | -2.283 | 0.02266 | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |

**D-13 rule as written on this grid: DROP** (significantly-negative on OU).

**That DROP is not evidence about line movement, and this readout will not pretend otherwise.**
Look at the last column: *not one* of the 15 line-movement columns was selected by *any* target.
No candidate model in this grid ever saw the signal. The reason is structural, not statistical:

- every trainer runs its own `SelectFromModel` pass on `config.train_seasons` **only**, and LOCKS
  that feature set for the entire holdout walk-forward;
- the canonical training window is **2018-2019**, and The Odds API historical floor is
  **2020-06-06**, so no 2018-2019 game can ever have a trajectory;
- measured on the rebuilt gold, all 15 columns are therefore **exactly constant (0.0) across all
  534 rows of the selection window** -- and constant across hp-val 2020 too, because of the
  D29-06-01 orphaning above. A constant column has zero model importance by construction and can
  never be selected.

So the three deltas above are **selection churn**: adding 15 constant columns to the candidate
pool perturbs the fitted importances and the selection threshold, and each target's leg locks a
slightly *different* set of the OTHER 25 features. The OU "significant" -0.221188 is a
measurement of that churn, not of market movement. The WP +0.000000 (with an undefined
t-statistic) is the clean control: WP's top-20 was unchanged, so its two legs were byte-identical.
The harness now reports this itself -- the run prints `grp_cols_used` per cell and a `NOT
MEASURED` banner -- so this failure mode cannot be published as a lift again.

### 4b. Coverage-window diagnostic: `python -m backtest.signal_lift --phase 29 --coverage-window`

To find out whether the signal is *measurable at all*, the same screen was re-run under a
diagnostic window that puts covered seasons in the selection fold: train 2021-2022, hp-val 2023,
measure **2024 only**; 255 paired games. Here the selector CAN see the family, and it chose it.

| Target | Paired CLV delta | t | p | Group columns the model used |
|---|---|---|---|---|
| WP | -0.001537 | -0.859 | 0.39111 | 3 / 15 (`spread_abs_travel`, `spread_range`, `total_drift`) |
| ATS | **+0.816061** | +3.562 | **0.00044** | 3 / 15 (`opening_spread`, `spread_drift_dir`, `total_range`) |
| OU | +0.281204 | +1.614 | 0.10784 | 3 / 15 (`opening_total`, `spread_late_drift`, `total_drift`) |

**D-13 rule as written on this grid: KEEP** (positive on ATS and OU, not significantly-negative
on any target).

**What this grid is NOT.** It is a diagnostic, not the pre-registered screen, and its weaknesses
are load-bearing, not decorative:

- **One measured season.** 2024 alone, 255 paired games. A single season's CLV can be regime-driven.
- **It trains on holdout seasons.** 2021-2022 selection and 2023 validation are holdout years, and
  D26-09 already records 2023-24 as partially burned. This configuration spends holdout to buy a
  measurement.
- **It was chosen after seeing the canonical result.** Six cells now exist across two windows with
  no multiple-comparison correction. The ATS p=0.00044 would clear a naive Bonferroni over six
  cells (0.0083), but a window picked post hoc is exactly the garden-of-forking-paths risk that
  Phase 30's binding gate exists to settle.

### 4c. Pre-registration (written and committed BEFORE the covered-window screen was run)

This subsection was written and committed to git ALONE, before any number from the covered
selection window existed anywhere in this repository -- and, in fact, before the window itself
existed in the codebase at all: the `--covered-selection-window` flag and the config it needs were
added in a LATER commit. Its whole value is that `git log` can check the ordering rather than
this document asserting it. The 4b diagnostic's central weakness was a window chosen after seeing
results; repeating that would make a new grid worth no more than the old one.

**1. The command, run exactly once.** Verbatim, copy-pasteable:

```
uv run python -m backtest.signal_lift --phase 29 --covered-selection-window
```

**2. The window.** Train seasons 2018, 2019, 2020; NO hp-val season; holdout / measure seasons
2021, 2022, 2023, 2024 -- the full four-season holdout. `groups=('line_movement',)`,
`baseline_exclude_groups=('line_movement',)`, `tune=False`, anchor
`BaseTrainer.train_and_evaluate(tune=False)`. This is a NON-DEFAULT configuration. The project's
canonical training window (train 2018-2019, hp-val 2020, holdout 2021-2024) is DELIBERATELY not
mutated (D-Q2): it is a project-wide gate-configuration decision that feeds Phase 30's binding
gate and every trainer, and changing it as a side effect of a measurement would be an invisible
global change made for a local reason. The consequence is stated in advance: Phase 30 must adopt a
covered selection window DELIBERATELY, or the family remains inert under the canonical window and
should be dropped (D29-07-01).

**3. The keep/drop rule.** Exactly as `decide_group_keep` implements it today
(`backtest/signal_lift.py:462-506`): KEEP if and only if the point-estimate delta is above zero on
at least one target AND no target is significantly-negative (mean below zero with
p < `SIGNIFICANCE_ALPHA`). Whatever that function returns on the headline grid is what gets
published. The rule is not re-stated in fresh prose that could be shaded after the fact -- it is
the committed function.

**4. The measurability precondition, and the pre-committed answer if it fails.** The grid is
evidence about line movement only for targets whose `n_group_columns_selected` is above zero. If
the family is STILL selected 0/15 even under a window that can see it, that is a real and complete
finding -- the family is not competitive on train-window importance even when visible -- and it
will be published as the phase's answer in the same voice as any other outcome. The Phase-30
recommendation in that case is pre-committed here as DROP, unless the family is admitted by some
route that does not depend on train-window importance.

**5. Multiplicity.** The grid is reported RAW with the existing multiplicity note. These three
cells bring the phase to nine cells across three windows with no correction applied here. The
binding BH-FDR / p<0.05 correction stays in the Phase-30 deploy gate. A single nominally
significant cell is a screening signal, not a gate ruling.

**6. The confound tell, registered before the run.** Inside the train 2018-2020 selection window
the family is exactly 0.0 for all 534 rows of 2018-2019 and real for roughly 244 of the 269 rows
of 2020. `line_movement_coverage` -- and anything collinear with it -- can therefore be selected
as a SEASON PROXY rather than as market information. Tell: if `line_movement_coverage` appears in
a target's `group_columns_selected`, that cell is declared CONFOUNDED and is not read as market
information.

**7. Limitations accepted in advance.** 2020 is a COVID season, and it sits inside the selection
window. The 2020 rows are recovered by an in-place re-key of already-purchased data, not
re-purchased. The headline rests on a non-default configuration (item 2).

**8. Null-result clause.** A flat, null, or negative grid is a COMPLETE and SUCCESSFUL outcome and
will be published as the phase's answer with the same prominence and detail as a positive one. The
7,210 credits already spent create no obligation to find a signal, and no result will be softened,
hedged, buried, or re-run until it is favourable.

**9. No further spend.** No Odds API call of any kind is made for this measurement.

**10. What will NOT change after seeing results.** The window definition, the decision rule, the
target set, and the measurement span. If any of them must change, the change will be reported as a
NEW post-hoc diagnostic with its own label, never folded into the headline grid.

### 4e. Ruling, and the D-02 question answered

**The binding constraint was never redundancy -- it was the calendar.** Section 2 predicted the
go/no-go would turn on whether line movement carries information the model does not already have.
The actual blocker is that the archive floor (2020-06-06) lands entirely *after* the canonical
feature-selection window (2018-2019). Under the configuration the project trains today, the
line-movement family is **inert**: it sits in gold, adds 15 columns to all three matrices, and can
never be selected into a model no matter how informative it is.

**On the D-02 priced-in risk: where the signal could be measured, it is not redundant.** The
priced-in worry from Section 2 was that the market moves on the same injury news the Phase-28
features already encode, making line movement a noisier proxy for signal the model can read
directly. In 4b the
baseline already contained the freeze anchor AND the Phase-28 injury / snap / situational signal,
and the models still chose line-movement columns over the alternatives and gained CLV on ATS. The
Section-2 prediction was that only the opening **LEVEL** and the **PATH** would be non-redundant,
not net drift. That is *partly* borne out: the selected columns are led by level (`opening_total`,
`opening_spread`) and path (`total_range`, `spread_range`, `spread_abs_travel`,
`spread_late_drift`), but `total_drift` -- plain net drift -- was also selected by two targets. So
the plausibility argument was directionally right and not exactly right.

**Ruling (owner-ratified, SIG-04):** **CARRY the line-movement family to Phase 30, CONDITIONAL on
the training window.** It is carried to Phase 30 for the binding gate and is not treated as an
edge on the strength of one diagnostic season. Two things must be true for Phase 30 to get a real
answer, and if neither is done the family should be dropped rather than carried indefinitely:

1. **Re-key the 2020 archive** (D29-06-01). No paid calls; recovers ~1,780 rows and 262 games.
2. **Phase 30 must select features on a window that has coverage.** If Phase 30 keeps train
   2018-2019, line movement cannot enter a candidate model and the 7,210 credits buy nothing
   further. This is a gate-configuration decision, not a feature-engineering one.

**Cost against learning, stated plainly.** The archive cost **7,210 credits** of a 20,000-credit
$30 month (12,790 unused), and it is final -- no further paid call is needed for any of the work
above. What that bought: a real 2020-2024 trajectory archive (9,957 rows, 1,359 games); the
measured fact that market movement is genuinely non-degenerate where the old `total_movement`
column was identically 0.0 (drift non-zero in 87.5% of games); one diagnostic season of evidence
that the signal is non-redundant against both the freeze anchor and the Phase-28 injury features;
and -- the finding with the most leverage -- the discovery that the project's feature-selection
window silently excludes any signal whose data floor is later than 2019. That last one applies to
every future signal purchase, not just this one. A flat or negative screen would have been an
equally complete result; what would NOT have been acceptable is publishing the 4a grid as a lift.

---

## 5. Honest scope-down / slip note (if applicable) -- carried to Phase 30, not a deploy outcome

The branch was **full-backfill** and it ran end to end: Plans 29-02 through 29-07 all landed. The
archive was bought, the builder was written with a per-game ET freeze fence and a three-part
leakage proof, gold was rebuilt to 209/210/209 columns with the family surviving `combine_features`
into all three matrices, and the lift screen was registered and run. Nothing slipped for lack of
data or lack of budget.

**What DID scope down is the strength of the verdict, and that is recorded here rather than
smoothed over.** The pre-registered screen (4a) could not measure the signal, because the archive
floor lands after the canonical feature-selection window. The measurement that could be made (4b)
rests on a single season and a post-hoc window. So Phase 29 ends with a *conditional carry* and a
named prerequisite, not with a measured four-season lift number. Two follow-ups are recorded in
the phase's `deferred-items.md` rather than being quietly done here: the 2020 re-key (no paid
calls) and the Phase-30 training-window decision.

**Framing, unchanged from Section 3.** The line-movement signal is SCREENED and carried to Phase
30 for the binding gate. It is not shipped, not serving, and not established as an edge. Phase 29
screens; Phase 30 rules. A null result would have been a first-class success outcome (D-14) and
was accepted as such in advance -- the screen, not optimism, rules (D-10).
