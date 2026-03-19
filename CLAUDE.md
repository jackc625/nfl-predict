# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An NFL Prediction System that generates pre-game Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) predictions. Built as a personal tool for informed NFL analysis, doubling as a portfolio project. Phases 1 (Foundation Hardening) and 2 (Data Pipeline Audit) are complete. Currently planning Phase 3 (Feature Engineering Correctness). 8 phases total, 6 remaining.

**Current Status**: Phases 1-2 complete. The data pipeline ingests games, odds, and weather through hardened Bronze/Silver layers with quality gates, canonical team mapping, and 32-team completeness verified across 2018-2024. Feature engineering, model training, backtesting, market blending, and UI polish remain.


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
- `data/silver/` -- Cleaned tables (games, odds_snapshot, weather, team_stats; latest-wins upsert)
- `data/gold/` -- Feature matrices per target (wp, ats, ou)

**Tech stack:** Python 3.13 (uv + Ruff + pyright), pandas, DuckDB/Parquet, scikit-learn, XGBoost, FastAPI, Jinja2/Tailwind CSS v4, HTMX 2.0

**Data sources:** nflreadpy (games), The Odds API (odds), Open-Meteo (weather), static JSON (stadiums/venues)

**Key constraints:**
- Walk-forward validation only -- no random CV, no future data leakage
- Friday 6 PM ET snapshot timing for odds/data freeze
- Reproducibility: any prediction must be reproducible given the same input data snapshot
- Canonical team abbreviation mapping with hard-fail on unknowns


## Planning and Documentation

The `.planning/` directory is the single source of truth for project planning and state:

- `.planning/PROJECT.md` -- Project overview, requirements context, known issues, key decisions
- `.planning/ROADMAP.md` -- 8-phase development roadmap with phase details and progress
- `.planning/STATE.md` -- Current position, decisions log, session continuity
- `.planning/REQUIREMENTS.md` -- Full requirements list (72 requirements across 7 categories)
- `.planning/codebase/` -- Architecture analysis, stack details, conventions, structure mapping
