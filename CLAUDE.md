# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An NFL Prediction System that generates pre-game Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions. Built as a personal tool for informed NFL analysis, doubling as a portfolio project. v1.0 MVP (10 phases: foundation, data pipeline, feature engineering, model training, backtesting, market blending, web UI) shipped 2026-03-28; v2.0 (phases 11-18: Elo, hyperparameter tuning, dynamic blend, automation, dashboards) shipped 2026-05-26; v2.1 "Trust & Reproducibility" (phases 19-23: documentation, audit, and diagnosis -- NOT new features) shipped 2026-06-01. The current milestone, v3.0 "Accuracy & Profitability" (phases 24-31: gate hardening, gated re-fits, O/U monetization, new signals, productization), is in progress.

**Current Status**: v3.0 Accuracy & Profitability -- all eight phases (24-31) complete; the milestone close is `PROFITABILITY-READOUT.md`. End-to-end pipeline: data ingestion (nflreadpy, Odds API, Open-Meteo) -> Bronze/Silver/Gold layers -> temporal-safe feature engineering (FeatureBuilder Protocol, LeakageGate) -> walk-forward model training (WP LogReg+isotonic, ATS/O/U XGBoost) -> a per-target deploy gate (`config/gate.toml`, significance-tested non-regression CLV) that conditionally promotes only gate-passing candidates -> backtest reporting (2021-2024, Brier decomposition, betting simulation) -> market blending (log-odds WP, linear ATS/O/U, pre-2018 weight tuning; dynamic blend per D-19) -> FastAPI + HTMX web dashboard with game detail drill-down, model-insights, betting, and season-tracking pages plus CSV/JSON exports. v2.0 added real Elo (2002 burn-in), Optuna tuning, and a Windows Task Scheduler Friday orchestrator. v2.1 consolidated one canonical run sequence (`PIPELINE.md`), forensically audited data/feature correctness (`AUDIT-REPORT.md`), verified + explained the automation (`AUTOMATION.md`), and honestly diagnosed accuracy (`MODEL-DIAGNOSIS.md`).

PHASE-25 RECORD (the first gated activation, D25-18): WP and ATS were promoted onto re-fit artifacts trained on canonical Elo gold (they passed the per-target non-regression gate; ATS via one documented fix-cycle); O/U RETAINED the v1.0 pre-Elo model (the re-fit failed the gate and was honestly refused, D25-14, to protect v1.0's stronger O/U line-CLV edge). See `ACTIVATION-READOUT.md` for the per-target before/after.

PRODUCTION STATE (post Phase-30 gated re-fit, PROD-01): gold was rebuilt in four separately-attributed rungs to **194/195/194 columns over 6,499 rows (2002-2025)**, a frozen feature-group selection rule was measured once on it, and the same per-target gate was run again. Feature groups: `snap` KEPT and `situational` KEPT (significantly positive after Benjamini-Hochberg correction on one target each); `injury` DROPPED (significantly negative on ATS) and excluded at TRAIN time -- its 12 columns remain physically in gold, so the verdict is revisitable without another rebuild. These groups were SCREENED in Phase 28 and only BINDINGLY ruled on in Phase 30; a group present in gold is not a group that was deployed. Targets: **WP PASSED and was promoted to `wp_20260824_113325`** (paired +0.006094, p=0.0142) -- the phase's ONE production change; **ATS FAILED and `ats_20260605_220128` was RETAINED** (paired -0.212797, p=0.0375); **O/U FAILED and `ou_20260326_163930`, the v1.0 pre-Elo model, was RETAINED** (paired -0.487007, p=9.24e-12, its MAE improving while its line-CLV worsened -- the D25-14 shape exactly). The single pre-registered fix-cycle lever went UNSPENT for both failing targets because it had no unspent move, not because it was overlooked. The dynamic blend (`blend_dynamic_20260606_020635`) is live for all three targets and was NOT changed. Two refusals is the gate working, not a failed phase. WP's ABSOLUTE pooled CLV is still negative (-0.0380); the gate is a non-regression gate and never asserted a positive market edge. **Seven quarantined reproductions and six deferred registers remain OPEN** at the Phase-30 close -- the suite's standing `7 xfailed` is the mechanical proof. See `GATED-REFIT-READOUT.md` for the full record and `STATE-OF-SYSTEM.md` for the consolidated open list.

PHASE-31 RECORD (productization + the milestone close, PROD-02/03/04): the weekly +EV bet list ships on a new `/bets` page -- ranked, sized in units, with an EV band and a realized-vs-expected tracker, served entirely from precomputed cache blobs with zero computation in the request path -- taking the top nav to SEVEN pages. **PHASE 31 DEPLOYED NO MODEL.** Nothing was re-fit or promoted, the dynamic blend was not changed, and `artifacts/latest.json` is byte-unchanged: the production swap surface named in the paragraph above is still exactly what serves, which is the point -- the measurement had to be a measurement of what is actually running. The single unburned 2025 season was spent once, under a rule frozen at commit `ee20773` before any 2025 number existed (`PROFITABILITY-PREREGISTRATION.md` + `backtest/ev_chain_constants.py`), on a one-shot ledgered run that cannot be repeated. Verdict: **NO target is `PROFITABLE_CLEAN`** -- WP `INCONCLUSIVE_CLEAN` (65 bets, flat ROI +0.0144, BH-adjusted ROI p 0.6717), ATS `UNPROFITABLE_CLEAN` (161 bets, flat ROI -0.0528, ROI p 0.7606), O/U `INCONCLUSIVE_CLEAN` (41 bets, flat ROI +0.0325, BH-adjusted ROI p 0.6717); BH denominator 6 over a 24-row registry. The HEADLINE FINDING is the CLV-to-ROI DIVERGENCE, not a return: WP's report-only closing-line value over its own selected 2025 bets is strongly significantly POSITIVE (+0.0918, p 6.8e-34) while its ROI is indistinguishable from zero -- the Phase-26 O/U shape, now measured on a clean split for WP. CLV is REPORT-ONLY throughout and never entered the correction family; it is never evidence of profitability. Nine owner-accepted disclosures ride with it, including that 68 protected-window games had been graded against FABRICATED 0.0 market lines, that the frozen gate baseline diverges from a re-score in 47 of 68 fields and was deliberately NOT re-frozen, and that Phase 30's `snap`/`situational` KEPT verdicts rest on superseded gold with nothing retrained. FIVE tests are DELIBERATELY RED and must stay red -- they are the tripwires that fired on those accepted facts. See `PROFITABILITY-READOUT.md`.


## Persistent Instructions for Claude

* Always read entire files. Otherwise, you don't know what you don't know, and will end up making mistakes, duplicating code that already exists, or misunderstanding the architecture.
* Commit early and often. When working on large tasks, your task could be broken down into multiple logical milestones. After a certain milestone is completed and confirmed to be ok by the user, you should commit it. If you do not, if something goes wrong in further steps, we would need to end up throwing away all the code, which is expensive and time consuming.
* Your internal knowledgebase of libraries might not be up to date. When working with any external library, unless you are 100% sure that the library has a super stable interface, you will look up the latest syntax and usage via either Perplexity (first preference) or web search (less preferred, only use if Perplexity is not available)
* Do not say things like: "x library isn't working so I will skip it". Generally, it isn't working because you are using the incorrect syntax or patterns. This applies doubly when the user has explicitly asked you to use a specific library, if the user wanted to use another library they wouldn't have asked you to use a specific one in the first place.
* Always run linting after making major changes. Otherwise, you won't know if you've corrupted a file or made syntax errors, or are using the wrong methods, or using methods in the wrong way.
* Please organise code into separate files wherever appropriate, and follow general coding best practices about variable naming, modularity, function complexity, file sizes, commenting, etc.
* Code is read more often than it is written, make sure your code is always optimised for readability
* Unless explicitly asked otherwise, the user never wants you to do a "dummy" implementation of any given task. Never do an implementation where you tell the user: "This is how it *would* look like". Just implement the thing.
* Whenever you are starting a new task, it is of utmost importance that you have clarity about the task. You should ask the user follow up questions if you do not, rather than making incorrect assumptions.
* Do not carry out large refactors unless explicitly instructed to do so.
* When starting on a new task, you should first understand the current architecture, identify the files you will need to modify, and come up with a Plan. In the Plan, you will think through architectural aspects related to the changes you will be making, consider edge cases, and identify the best approach for the given task. Get your Plan approved by the user before writing a single line of code.
* If you are running into repeated issues with a given task, figure out the root cause instead of throwing random things at the wall and seeing what sticks, or throwing in the towel by saying "I'll just use another library / do a dummy implementation".
* You are an incredibly talented and experienced polyglot with decades of experience in diverse areas such as software architecture, system design, development, UI & UX, copywriting, and more.
* When doing UI & UX work, make sure your designs are both aesthetically pleasing, easy to use, and follow UI / UX best practices. You pay attention to interaction patterns, micro-interactions, and are proactive about creating smooth, engaging user interfaces that delight users.
* When you receive a task that is very large in scope or too vague, you will first try to break it down into smaller subtasks. If that feels difficult or still leaves you with too many open questions, push back to the user and ask them to consider breaking down the task for you, or guide them through that process. This is important because the larger the task, the more likely it is that things go wrong, wasting time and energy for everyone involved.
* If I am ever wrong, please point it out, I need honest feedback on my code.
* Do not use Emojis in code


## Architecture and Stack

The system follows a data pipeline architecture:

```
External Sources -> Ingestion (ETL) -> Data Lake (Parquet/DuckDB) -> Feature Build -> Models -> Predictions API (FastAPI) -> Web UI
```

**Data layers:**
- `data/bronze/` -- Raw snapshots (append-only, timestamped)
- `data/silver/` -- Cleaned tables (games, odds_snapshot, weather, contextual, elo_game_snapshots, team_form_features, team_game_stats; latest-wins upsert)
- `data/gold/` -- Feature matrices per target (wp, ats, ou)

**Tech stack:** Python 3.13 (uv + Ruff + pyright), pandas, DuckDB/Parquet, scikit-learn, XGBoost, FastAPI, Jinja2/Tailwind CSS v4, HTMX 2.0

**Data sources:** nflreadpy (games), The Odds API (odds), Open-Meteo (weather), static JSON (stadiums/venues)

**Key constraints:**
- Walk-forward validation only -- no random CV, no future data leakage
- Day-before-kickoff lock (D33.2-01): each game's information locks at 18:00 America/New_York on the ET calendar day before its kickoff, and information timed at the lock is admissible (at-lock counts, one second after does not). The one rule is `utils/game_lock.py`. The live run is a daily lock-time cycle, not a weekly Friday one
- Reproducibility: any prediction must be reproducible given the same input data snapshot
- Canonical team abbreviation mapping with hard-fail on unknowns


## Planning and Documentation

The `.planning/` directory is the single source of truth for project planning and state:

- `.planning/PROJECT.md` -- Project overview, requirements context, known issues, key decisions
- `.planning/ROADMAP.md` -- Development roadmap with milestone groupings and progress
- `.planning/STATE.md` -- Current position, decisions log, session continuity
- `.planning/MILESTONES.md` -- Shipped milestone history
- `.planning/RETROSPECTIVE.md` -- Living retrospective with lessons learned
- `.planning/milestones/` -- Archived milestone artifacts (roadmap, requirements, phases)
- `.planning/codebase/` -- Architecture analysis, stack details, conventions, structure mapping

**Repository documents.** Guides (`PIPELINE.md`, `RUNBOOK.md`, `AUTOMATION.md`, `METHODOLOGY.md`, `STATE-OF-SYSTEM.md`) live in `docs/guides/`; readouts, audits and diagnoses live in `docs/records/`. Six history-anchored records stay at the repo root and must never be moved, because their tamper evidence reads git history at the root path: `PROFITABILITY-PREREGISTRATION.md`, `COLD-START-PREREGISTRATION.md`, `COLD-START-CORRECTION.md`, `NEUTRAL-HFA-BET-RULE-CORRECTION.md`, `EV-CHAIN-CORRECTION.md`, `MOS-DECODE-COMPARISON.md`. Tests find a document from its bare name with `tests/doc_locations.py`.
