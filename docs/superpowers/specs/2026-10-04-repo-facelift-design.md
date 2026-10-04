# Public repo facelift -- design

Date: 2026-10-04
Branch: `repo-facelift` (isolated worktree; the daily 18:00 ET pipeline keeps running from the
main checkout on `master`)

## Goal

Make the public GitHub page for `jackc625/nfl-predict` read as a professional, legitimate
portfolio project: a visitor should understand what the system does, see it, trust that it is
honest about results, and find the deeper documentation without wading through a wall of files.

Audience: recruiters, hiring managers and other engineers browsing the repo. The owner remains
the only user of the system itself.

## Owner decisions (2026-10-04)

1. Fresh visitor-facing README. The README guards that lock in history are DROPPED, not
   re-pointed; the old README survives in git history only.
2. Extras: screenshots, refreshed GitHub About box, a lint CI workflow with badge, a social
   preview image.
3. Move the repo-root reports into `docs/`, except the history-anchored records.
4. Retire the `docs/ must not exist` check (D-04); keep the "stale v1.0 docs stay deleted" check.

## Out of scope

- Moving any sealed / history-anchored record (see Part B).
- Editing the content of any moved document, or of any hash-locked / append-once file.
- Rewriting bare-filename prose mentions ("see PIPELINE.md") in code comments and docstrings.
- Reformatting the 7 files `ruff format --check` flags today.
- Releases / tags, pyproject version bump.
- `.planning/` (tracked files there are left exactly as they are).

## Part A -- plain fixes

- Add `LICENSE` (MIT, "Copyright (c) 2025-2026 Jack Cutrara"). The README and pyproject already
  claim MIT; GitHub currently shows no license.
- `pyproject.toml`: author becomes `{name = "Jack Cutrara"}` (placeholder email removed, so no
  address is published); `[project.urls]` fixed from `github.com/jackc/...` to
  `github.com/jackc625/...`.

## Part B -- move the reports into docs/

### Stay at the repo root (6)

Each is a record whose git history at its current path is the proof that its rule was frozen
before its results existed (a `git mv` would make the move commit the new "last-modifying
commit" and void that proof):

- `PROFITABILITY-PREREGISTRATION.md` -- `backtest/ev_chain_constants.py` PREREGISTRATION_PATHS,
  `test_preregistration_ancestry.py`, `config/profitability_2025_verdict.toml`
- `COLD-START-PREREGISTRATION.md` -- `backtest/cold_start_constants.py`,
  `test_phase33_preregistration_ancestry.py`
- `COLD-START-CORRECTION.md` -- `backtest/corrected_cold_start_constants.py`,
  `test_superseding_correction.py`
- `NEUTRAL-HFA-BET-RULE-CORRECTION.md` -- the live 2026 bet rule,
  `backtest/neutral_hfa_cold_start_constants.py`
- `EV-CHAIN-CORRECTION.md` -- readable half of the corrected EV-chain record (same family)
- `MOS-DECODE-COMPARISON.md` -- anchors `config/mos_tolerance.py` via
  `test_mos_tolerance_ancestry.py`

Root after the change: README.md, LICENSE, CLAUDE.md, the 6 records above, and the existing
config files (pyproject.toml, uv.lock, Makefile, conftest.py, .env.example, ...).

### Move (22), with `git mv` (bytes unchanged)

`docs/guides/` -- how it works and how to run it:
PIPELINE.md, RUNBOOK.md, AUTOMATION.md, METHODOLOGY.md, STATE-OF-SYSTEM.md

`docs/records/` -- the paper trail:
ACTIVATION-READOUT.md, AUDIT-REPORT.md, BLEND-TUNING-READOUT.md, CLOSING-LINE-AUDIT.md,
GATED-REFIT-READOUT.md, GROUP-VERDICT-READOUT.md, HISTORICAL-WEATHER-READOUT.md,
LINE-MOVEMENT-READOUT.md, LIVE-COLD-START-READOUT.md, MODEL-DIAGNOSIS.md,
OU-DIVERGENCE-DIAGNOSIS.md, PRECOVERAGE-SCAN.md, PROFITABILITY-READOUT.md, REFIT-READOUT.md,
SELECTION-CENSUS.md, SIGNAL-LIFT-READOUT.md, WEATHER-NULL-LIST.md

### Reference rules

1. **Working-tree reads get the new path.** Every test/script that opens a moved file
   (`REPO_ROOT / "X.md"` and equivalents) is updated to the new location.
2. **Historical reads keep the root name.** Reads of an old commit by path stay on the root
   name, because that is where the file lived at that commit:
   - `tests/unit/test_line_movement_readout_md.py` (~:705, `git show <sha> -- LINE-MOVEMENT-READOUT.md`)
   - `tests/unit/test_old_rule_labels.py` `_labelling_numstat` (commit 7b1928f, root-only filter)
   Using the new path there would make those checks silently vacuous.
3. **Bare-filename prose stays.** Comments, docstrings, and template comments that name a doc
   ("see PIPELINE.md") are left alone: the names stay unique and findable, several carrying
   files are hash-locked or append-once (`conf/season_partition.py`,
   `config/group_gate_verdict.toml`, `config/tuning_preregistration.py`,
   `config/mos_tolerance.py`, `backtest/ev_chain_constants.py`,
   `backtest/corrected_cold_start_constants.py`, `tests/phase30_state.py`,
   `tests/phase31_state.py`, `tests/phase33_state.py`), and none of the 28 docs contain
   markdown links, so nothing breaks on GitHub.
4. **Never edit a sealed or hash-locked file** to follow the move.

### Test changes for the move

- `tests/unit/test_old_rule_labels.py`
  - `tracked_root_markdown()` also lists tracked `*.md` directly under `docs/guides/` and
    `docs/records/` (the partition keeps covering every report, wherever it lives).
  - `LABELLED_READOUTS` and `NO_PREFIX_NUMBERS_REASONS` keep BARE names; a single
    name -> current-path lookup is used by `_read` and by the partition check.
  - `_labelling_numstat` and `test_it_touched_exactly_the_declared_labelled_set` keep comparing
    commit 7b1928f's ROOT names against the bare names (unchanged semantics).
- `tests/unit/test_profitability_readout_md.py`: names in `PRIOR_READOUT_SHA256` /
  `FIGURE_SOURCES` are used both as file paths and as tokens searched for in the readout text.
  Keep the names bare and resolve the path separately, so a moved source is never read as
  missing (which would wave figures through unchecked).
- Every other test that reads a moved doc: path constant updated to the new location.
- `tests/unit/test_methodology_md.py`: remove `test_docs_directory_removed` (D-04 retired by the
  owner); keep the stale-v1.0-basenames check.

### CLAUDE.md

Add one line under "Planning and Documentation": guides live in `docs/guides/`, the paper trail
in `docs/records/`, and the six history-anchored records stay at the repo root and must never be
moved. Nothing else in CLAUDE.md changes.

## Part C -- images

All images under `docs/images/`.

- **Screenshots** of the running site with real data (served from the main checkout, which has
  the gitignored `data/`; the UI there is identical to this branch), desktop viewport ~1440 wide,
  via Playwright: This Week (`/`, the hero), Bets (`/bets`), a Game detail page
  (`/games/{id}`), Track Record (`/track-record`), Season (`/season`).
  Each is checked by eye for errors, empty states, or anything that should not be public.
- **Banner** (1280x640 PNG) in the site's broadcast style: wordmark NFL/Predict, tagline, a
  hint of the UI. Used at the top of the README and as the GitHub social preview (the owner
  uploads it in Settings > Social preview; GitHub has no API for it). Its HTML source is kept in
  `docs/images/src/` so it can be regenerated.

## Part D -- the README

Target ~200-250 lines, ASCII only, no emoji.

1. Banner, title, one-line tagline.
2. Badges: lint workflow status, Python 3.13, FastAPI, XGBoost, scikit-learn, DuckDB, HTMX,
   Tailwind CSS, license MIT.
3. Hero screenshot (This Week).
4. What it does -- plain bullets: WP / spread / total forecasts, the ranked +EV bet list,
   per-game drill-down, the live track record.
5. Screenshots -- 2x2 grid (Bets, Game detail, Track Record, Season).
6. How it works -- a Mermaid flowchart (sources -> bronze/silver/gold -> features + leakage
   gate -> three models -> market blend -> deploy gate -> web cache -> FastAPI/HTMX), plus the
   day-before 18:00 ET information lock and the daily scheduled run.
7. Engineering highlights -- leakage protection, walk-forward only, the significance-tested
   deploy gate, pre-registered evaluation with git-anchored tamper evidence, the API/ML import
   boundary enforced by an AST test.
8. Results, honestly -- no target has shown a profitable edge; results from before the
   2026-09-15 fix were built on inputs later found defective and are not evidence; the 2026
   season, recorded live, is the scorecard (Season page; the Track Record page holds the
   labelled past-season backtests). Quotes NO pre-fix number. Every
   claim checked against `docs/records/LIVE-COLD-START-READOUT.md` and
   `docs/records/PROFITABILITY-READOUT.md` and current code.
9. Tech stack table, quickstart (PowerShell, `uv`), short folder map.
10. Documentation index: guides, records, and the six root records (each one line).
11. Disclaimer (informational only, not betting advice, not affiliated with the NFL); license.

### README guards (dropped, per decision 1)

- `tests/unit/test_readme_claude_reconciled.py`: remove `TestReadmeReconciled` and every
  README-specific assertion; the loops over (README, CLAUDE) become CLAUDE-only. All CLAUDE.md
  assertions stay. Module docstring updated to say so.
- `tests/unit/test_old_rule_labels.py`: README moves from `LABELLED_READOUTS` to
  `NO_PREFIX_NUMBERS_REASONS` with a written reason (the visitor-facing front page reports no
  pre-fix result). The 7b1928f numstat check still expects README among the files that commit
  labelled (it did), so that historical comparison is made against the labelled set plus
  README, with a comment.

## Part E -- lint CI

`.github/workflows/lint.yml`: on push and pull_request to master; checkout, install uv,
`uv run ruff check --no-fix --output-format=github .` with the project-pinned Ruff. Lint only:
the test suite needs the gitignored local data and cannot run on GitHub, so no test badge.
Action versions and syntax are looked up from current docs at implementation time.

## Part F -- GitHub About box and push (owner approval required for each)

Proposed description (<= 350 chars):
"Pre-game NFL win probability, spread and total forecasts: walk-forward models, a leakage-proof
day-before information lock, pre-registered evaluation, and a broadcast-style FastAPI + HTMX
dashboard."
(Revised while planning: the September 2026 rebuild replaced the models outright without a gate
comparison, so "a significance-tested deploy gate" would imply a step that did not put today's
models in place.)

Proposed topics: keep duckdb, fastapi, htmx, machine-learning, nfl, python, sports-analytics,
xgboost; add scikit-learn, sports-betting, elo-rating, walk-forward-validation, tailwindcss,
nflverse.

Applied with `gh repo edit` only after the owner approves the exact text. Pushing the branch /
master to GitHub is likewise asked for separately.

## Order and commits (one commit per step, on `repo-facelift`)

1. Spec (this file).
2. LICENSE + pyproject fixes.
3. Doc move + reference/test updates + D-04 retirement + CLAUDE.md line.
4. Screenshots + banner.
5. README + README guard changes.
6. Lint workflow.
Then merge to master (fast-forward) once verified; push and About box on approval.

## Verification

- Baseline FIRST, in this worktree, before any change: run every test file that references a
  moved doc or the README, and record which already fail here (known-red tests, tests needing
  the gitignored data).
- After step 3 and again after step 5: the same set, targeted (no bare full-suite run). Bar: no
  new failures; `test_docs_directory_removed` is gone.
- `uv run ruff check .` clean after every code step.
- README rendered and eyeballed (GitHub-flavoured markdown, Mermaid, image paths) before merge.

## Risks

- A history read silently switched to a new path would make a check vacuous rather than red.
  Mitigation: the two known history reads are named above and reviewed by hand.
- Screenshots could expose something unintended (e.g. local paths, error states). Mitigation:
  each image is reviewed before commit.
