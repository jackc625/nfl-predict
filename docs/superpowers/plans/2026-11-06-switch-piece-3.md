# The Switch (piece 3, lean) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From 2026 week 13, retire the old day-before betting system, pause the v1 calendar re-fit, and install the repaired v2 models through a recorded owner install, with the bets page, the notifier and the February readout saying plainly what changed and why.

**Architecture:** A new frozen record (`MIDSEASON-SWITCH-RECORD.md` + `backtest/midseason_switch_2026.py`) fixes the switch week, the paused re-fit dates and the installed recipe; every behaviour change reads it. The re-fit hook closes the paused dates `not run`. The daily run writes no `arm="live"` row for a 2026 game of week 13 or later. A new append-only `ledger/refit/owner_installs.jsonl` makes an owner-run install a recognised generation, and the nightly gold build follows the in-force generation's recipe, so v2 gold and v2 models change together and nothing changes on the switch day except one owner command. No new arm, no new fill convention, no new bet rule kind: the day-before window places no bets because v2 earned weight 0 on all three bet types.

**Tech Stack:** Python 3.13, uv, pandas, scikit-learn (the existing v2 recipe), FastAPI + Jinja2, pytest.

**Owner correction, 2026-11-06:** the switch moved from week 13 (Tue 2026-12-01) to week 10 (Tue 2026-11-10) (`MIDSEASON-SWITCH-RECORD.md` section 8, commit "record: correct the switch to 2026 week 10"); Tasks 1-10 below are the record of what was built for week 13, and Tasks 11-12 are updated for week 10, whose deadline is Tue 2026-11-10 17:00 ET.

**Spec:** `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-profitability-fixes-design.md` (sections 5, 6.2, 6.4, 6.5, 7 "Stage 2"), narrowed by the owner ruling below. Evidence (gitignored; read at these absolute paths): `C:\Users\jackc\code\nfl-predict\.planning\research\2026-11-05-why-not-profitable\` -- `constraints-for-midseason-changes.md` (sections a, c, d, e; cited "constraints s.x"), `code-map-model-v2.md` (cited "map"), `study-1a-report.md` (the numbers the record quotes). Line numbers below were read at master `92b2032`.

**Owner ruling, 2026-11-06 (lean switch).** Study 1a gave all three bet types weight 0 and nothing is re-fitted after the switch, so the day-before window places no bets for the rest of 2026 either way. Piece 3 therefore does NOT build the day-before pricing machinery (best price, quarter-Kelly, weekly cap, the no-contradiction rule, a `market` arm, `fill-v2`, a market bet-rule kind). Piece 4 builds pricing for the early-week board it will actually price from. The key-number cover-probability converter of design 6.2 item 6 is pricing machinery too (only a bet price reads it), so it also moves to piece 4.

**Calls made in this plan (the owner can overrule at review).**
1. **The switch is keyed on the NFL week, not the clock.** Week 12's last lock is Sun 11-29 18:00 ET and week 13's first is Wed 12-02 18:00 ET (measured), so Tue 12-01 17:00 ET falls between them and the switch week is 13. A week number needs no clock arithmetic (the one-lock-rule scan), and the live-decision tests use week-3 fixtures that a wall-clock check would break after Dec 1.
2. **The record and all code merge BEFORE Tue 12-01 17:00 ET and take effect on merge.** Before week 13 nothing visible changes; the re-fit pause must be in force before re-fit date 2 (12-01 22:00 UTC) whatever day v2 installs. The v2 install is one owner command run on or after Tue 12-01 (it refuses earlier).
3. **Gold follows the in-force generation.** The nightly build uses the v2 builder exactly when `artifacts/latest.json` names a v2 generation, and data QA expects that recipe's widths. So no switch-day code commit is needed, and v1 models can never score v2 gold (constraint 1 of the design).
4. **The install refuses unless the blend weights it tunes equal Study 1a's earned weights (0/0/0).** If more 2026 games moved a weight off 0, that is a new finding for the owner, not something to deploy or override silently.
5. **The v2 bet-rule record carries no EV floors and no edge bands.** Game cards then show no edge band (the site's existing "no honest threshold" state), because a model with weight 0 has no edge to band.
6. **A failed install rebuilds the v1 gold before it stops**, so the next daily run's data check passes with the old models.
7. **Closing captures stay on after week 12** (they are the reference the early-week work needs). The early-lines capture is untouched.
8. **The February readout refuses if any counted live row is in week 13 or later**: that would mean the retirement leaked and the record is false.
9. **The record is witnessed in a second, small PR** after the first merges (the witness test needs the record commit on `origin/master`).
10. **The two open piece-2 review notes** (`features/supplied_locks.py` accepts naive times; `walk_forward_v2(..., prediction_rows=)` crashes on a target with no rows) stay for piece 4, which reuses those functions.
11. **No ledger row carries the v2 recipe in piece 3**, so `forward_ledger/declarations.py` (`resolve_recipe`) is not taught it; piece 4 does that with its first early-week row.

## File structure

| File | New / edit | Responsibility |
|---|---|---|
| `MIDSEASON-SWITCH-RECORD.md` | new (root) | the record: what changes from week 13, why, what it invalidates |
| `backtest/midseason_switch_2026.py` | new | the record's constants (stdlib only) |
| `refit/hook.py` | edit | paused dates close `not run`, nothing trains |
| `notify/digest_sections.py` | edit | a paused close reads "paused", not "the next date arrived first" |
| `refit/records.py` | edit | `owner_installs.jsonl`: validate, append, read |
| `refit/generations.py` | edit | owner installs become generations; `in_force_recipe` |
| `refit/comparison.py` | edit | the Season panel dates an owner generation by its install |
| `scripts/build_gold_v2.py` | edit | `FeatureMatrixBuilderV2Live` for the sanctioned nightly writer |
| `pipeline/steps.py` | edit | the gold builder follows the in-force recipe |
| `scripts/data_qa.py` | edit | expected gold widths follow the in-force recipe |
| `refit/owner_install.py` | new | the install: gold, train, save, blend, bet rule, record, swap |
| `scripts/install_v2.py` | new | the owner's command |
| `pipeline/daily_steps.py` | edit | `recommend_slate` writes nothing from week 13 |
| `api/cache_schema.py` | edit | the page's own copy of the switch week |
| `api/routes/pages.py`, `web/templates/pages/bets.html`, `_bet_headliners.html` | edit | the retired state on /bets |
| `notify/end_of_run.py`, `notify/digest.py` | edit | say "retired", not "unknown" or silence |
| `scripts/season_readout.py`, `scripts/season_coverage.py` | edit | coverage ends at week 12; IN-A03; relative paths |
| `tests/live_manifest_record.py` | edit | recognise an owner-installed v2 generation |
| `tests/phase33_state.py` | append | one width-ladder slot (append protocol) |
| `tests/midseason_switch_state.py`, `tests/unit/test_midseason_switch_ancestry.py` | new (PR B) | the witness |

## Global Constraints

- Work in the worktree `C:\Users\jackc\Code\nfl-predict-v2` on a new branch `switch` from master `92b2032` (Task 1 step 1). Its `data/bronze`, `data/silver`, `data/gold`, `data/baselines`, `data/optuna` are junctions into the LIVE lake. Tasks 1-10 write nothing under `data/`, `artifacts/`, `ledger/` or `logs/`; tests write only under `tmp_path` (the autouse guard `tests/conftest.py:1285` fails any test that writes those folders). Only Task 12 writes the live lake, from the main checkout.
- Frozen, never edited: `SEASON-VERDICT-PREREGISTRATION.md`, `backtest/verdict_test_2026.py`, `ROLLING-REFIT-PREREGISTRATION.md`, `backtest/refit_preregistration_2026.py`, `backtest/fill_conventions.py`, `backtest/recipe_registry.py`, `backtest/verdict_scope_2026.py`, `backtest/shadow_pin_2026.py`, `refit/cadence.py`, the `PREREGISTRATION_*` values in `refit/witness.py`, `config/gate.toml`, `models/deploy_gate.py`, `scripts/promote_models.py`, `config/blend_sources.py`, every existing slot in `tests/phase3*_state.py` (new slots are appended under the file's APPEND PROTOCOL). Never hand-edit `artifacts/latest.json`.
- Tests: only named files or node ids, `uv run pytest <file>::<test> -q -p no:cacheprovider`. Never bare `pytest`, a directory, or the full suite. No `xfail`, no skip, no loosened assertion. A test that pins old behaviour is RE-POINTED with a dated docstring line ("RE-POINTED 2026-11-xx for MIDSEASON-SWITCH-RECORD.md: ...").
- Lint every touched file: `uv run ruff check --no-fix --no-cache <files>`, `uv run ruff format <files>`, `uv run pyright <files>` (0 errors).
- Repo scan rules (read once, obey everywhere): no dict key, keyword or column named `arm`, `recipe_id`, `model_artifact_id`, `blend_id`, `fill_convention_id` or any other ledger stamp name outside `forward_ledger/stamps.py` (use `recipe`, `v2_recipe`); no name containing `FIRST_SEASON` bound to a year; import the lock rule only as `import utils.game_lock as lock_rule`; no function calling `.weekday()` with a fixed hour; `api/` imports no `backtest`, `forward_ledger`, `scripts` or `notify`; `refit/hook.py` identifiers never contain `clv`, `drift`, `roi`, `brier`, `p_value`, `comparison`; no file under `refit/` imports optuna or `models.tuning`; no new top-level package; no new `rmtree`/`unlink`/`replace`/`rename` near an artifact, ledger or staging path outside `refit/`.
- ASCII only in code, tests, templates and documents; no emoji.
- Commit after each task on `switch`, ending every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01LVgorLUwwGjKecmY3yVBge`.
- Do not push, open a PR or merge until the owner says so. Merges to master need `gh pr merge --merge --admin` (the "owner-only updates" ruleset). PR A (Tasks 1-10) must be merged before Tue 2026-12-01 17:00 ET; if that cannot happen by Mon 11-30, stop and tell the owner (the record would need a superseding record naming a later week).
- Read ET time with PowerShell `Get-Date`, never Git Bash `TZ=`. Nothing that touches the live lake runs 17:00-17:20 ET.
- Every subagent runs on Opus (owner instruction, 2026-11-06).

## Review Focus

1. **An install that fails halfway** (gold written, models not swapped; or the blend weights refuse). Expected: `latest.json` is unchanged, the v1 gold is rebuilt, and the next daily run's data check passes. Pinned in Task 5.
2. **The install run too early**, while a week-12 game is still before its lock (live decisions would stamp a v2 id the ledger cannot resolve). Expected: the install refuses and writes nothing. Pinned in Task 5.
3. **A paused re-fit date falling due, including the last one (2027-01-26)**, which today's cadence would leave due forever. Expected: each closes `not run` with `paused_by`, once, and nothing trains. Pinned in Task 2.
4. **The Season panel and the February comparison with an owner generation that has no live rows.** Expected: no raise; `compare_refit_arms` exits 0. Pinned in Task 3.
5. **The week-13 bets page after its first lock.** Expected: the retired message, not "This week's list is missing"; the tracker reads as the retired system's record. Pinned in Task 8.

---

### Task 1: The switch record

**Files:**
- Create: `MIDSEASON-SWITCH-RECORD.md`, `backtest/midseason_switch_2026.py` (commit R, these two files only)
- Modify: `tests/unit/test_old_rule_labels.py`; Create: `tests/unit/test_midseason_switch_record.py` (commit R+1)

**Interfaces:**
- Produces, in `backtest/midseason_switch_2026.py` (stdlib only, no logic):
  - `RECORD_PATH: str = "MIDSEASON-SWITCH-RECORD.md"`
  - `DAY_BEFORE_RETIRED_FROM: tuple[int, int] = (2026, 13)` -- (season, week)
  - `R6_COVERAGE_END_WEEK: int = 12`
  - `REFIT_PAUSED_DATES: tuple[str, ...] = ("2026-12-01", "2026-12-29", "2027-01-26")`
  - `REFIT_PAUSE_RULING: str = "MIDSEASON-SWITCH-RECORD.md section 3 (owner ruling B, 2026-11-05)"`
  - `INSTALLED_RECIPE: str = "recipe-2026-v2"`
  - `V2_EARNED_WEIGHTS: Mapping[str, float]` = read-only `{"wp": 0.0, "ats": 0.0, "ou": 0.0}`
  - `PREREGISTRATION_DEADLINE_UTC: str = "2026-12-01T22:00:00+00:00"`
  - `INVALIDATES: tuple[str, ...]` = the three full shas `922b010b70d973a21945c15ca88067dae4149ad5`, `c3a33993160769b2ab9a8c2cf7cd2a24ad19be10`, `e25884da8356b9ec76970c4c14f87a797d2a32d7`

The record must state, in plain English: (1) from 2026 week 13 (Tue 2026-12-01 17:00 ET) no `arm="live"` row is written and so no shadow twin; no new arm is added; the day-before window places no bets; (2) why: Study 1a's install-bar losses (WP 0.6287 vs 0.6410, ATS 10.204 vs 10.504, O/U 10.741 vs 11.034) and earned weights 0/0/0 with their slopes and one-sided p (from `study-1a-report.md`), and the owner rulings of 2026-11-05 (B) and 2026-11-06 (lean switch); (3) the v1 re-fit dates 2-4 of list week 8 (the three dates above) close `not run`, paused by this record, and no further 2026 re-fit runs; (4) the v2 install: recipe `recipe-2026-v2` = `models/v2_recipe.py` as last committed (name its commit), its inputs per target, trained once on every eligible game 2002 to the install instant, blend weights equal to the earned weights or the install refuses, a bet-rule record with no EV floors and no edge bands, recorded in `ledger/refit/owner_installs.jsonl`, `latest.json` written only by `replace_manifest`, allowed only after week 12's last lock; (5) February: the R6 rule of 922b010 grades the live arm over weeks 5-12 unchanged, the coverage table ends at week 12 and cites this record, a counted live row in week 13 or later refuses the run; the e25884d comparison covers generation 1 against generation 0 over weeks 9-12; (6) what it INVALIDATES, per sha and section: 922b010 in part ("the whole live 2026 season" is now weeks 5-12), c3a3399 in part ("the live arm = the bets the system recommended" no longer holds from week 13), e25884d sections 1 (frozen recipe), 4 (dates 2-4), 9 (bet rule) and 11 (comparison limited to weeks 9-12); (7) what it does not do: no `fill-v2`, no `market` or `early` arm, no new pre-registration -- later records; (8) the anchor block, in the form of `ROLLING-REFIT-PREREGISTRATION.md:31-36`, naming `tests/midseason_switch_state.py` and the deadline.

- [ ] **Step 1: Create the branch and commit this plan on it**

```powershell
git -C C:\Users\jackc\Code\nfl-predict-v2 switch -c switch master
Move-Item C:\Users\jackc\code\nfl-predict\docs\superpowers\plans\2026-11-06-switch-piece-3.md C:\Users\jackc\Code\nfl-predict-v2\docs\superpowers\plans\
git -C C:\Users\jackc\Code\nfl-predict-v2 add docs/superpowers/plans/2026-11-06-switch-piece-3.md
git -C C:\Users\jackc\Code\nfl-predict-v2 commit -m "docs: implementation plan for the switch (piece 3, lean)"
```

Expected: the new commit sits on `92b2032`; the main checkout keeps no copy of the plan.

- [ ] **Step 2: Write the two record files and commit them ALONE**

```bash
git add MIDSEASON-SWITCH-RECORD.md backtest/midseason_switch_2026.py
git commit -m "record: retire the day-before system from 2026 week 13, pause the v1 re-fit, install v2

INVALIDATES 922b010b70d973a21945c15ca88067dae4149ad5 (in part: R6 grades weeks 5-12),
c3a33993160769b2ab9a8c2cf7cd2a24ad19be10 (in part: the live arm stops at week 13),
e25884da8356b9ec76970c4c14f87a797d2a32d7 (sections 1, 4, 9, 11). See MIDSEASON-SWITCH-RECORD.md."
```

Expected: `git show --name-only --format= HEAD` lists exactly the two files.

- [ ] **Step 3: Write the failing content test**

```python
def test_the_installed_recipe_is_the_v2_recipe():
    assert INSTALLED_RECIPE == V2_RECIPE_NAME

def test_the_paused_dates_are_list_8_dates_after_the_first_re_fit():
    # the list-week-8 dates in backtest/refit_preregistration_2026.py (~:211) later than 2026-11-03
    assert REFIT_PAUSED_DATES == tuple(d for d in list_8_dates() if d > "2026-11-03")

def test_coverage_ends_the_week_before_the_switch():
    assert R6_COVERAGE_END_WEEK == DAY_BEFORE_RETIRED_FROM[1] - 1

def test_every_earned_weight_is_zero():
    assert dict(V2_EARNED_WEIGHTS) == {"wp": 0.0, "ats": 0.0, "ou": 0.0}

def test_the_switch_instant_falls_between_week_12_and_week_13():
    games = load_silver_games(Path("data/silver")).query("season == 2026")   # read-only
    locks = lock_rule.lock_frame(games)
    deadline = datetime.fromisoformat(PREREGISTRATION_DEADLINE_UTC)
    assert locks[games.loc[games.week <= 12, "game_id"]].max() < deadline <= locks[games.loc[games.week == 13, "game_id"]].min()

def test_the_record_is_ascii_and_names_every_invalidated_sha():
    text = Path(RECORD_PATH).read_text(encoding="utf-8")
    assert text.isascii() and all(sha in text for sha in INVALIDATES)
```

- [ ] **Step 4: Run it to see it fail**

Run: `uv run pytest tests/unit/test_midseason_switch_record.py tests/unit/test_old_rule_labels.py -q -p no:cacheprovider`
Expected: `test_old_rule_labels.py` fails to collect (the new root markdown is tracked but unlisted); the content test passes or fails only on wiring.

- [ ] **Step 5: Register the record** in `tests/unit/test_old_rule_labels.py`: add it to `NO_PREFIX_NUMBERS_REASONS` (~:81-197) with a one-line reason, and to the sealed parameter list (~:269-277), as commits `b3e99df` and `1016f66` did. Never `LABELLED_READOUTS`.

- [ ] **Step 6: Run both test files**

Run: the Step 4 command. Expected: all PASS.

- [ ] **Step 7: Lint and commit R+1**

```bash
git add tests/unit/test_old_rule_labels.py tests/unit/test_midseason_switch_record.py
git commit -m "test: register the mid-season switch record and check it against the code"
```

---

### Task 2: Pause the v1 re-fit

**Files:**
- Modify: `refit/hook.py` (`_run_locked`, ~:667-750), `notify/digest_sections.py` (~:484-485), `docs/guides/AUTOMATION.md` (one paragraph: re-fit paused, why, where)
- Test: `tests/unit/test_refit_hook.py`, `tests/unit/test_refit_notifications.py`

**Interfaces:**
- Consumes: Task 1 `REFIT_PAUSED_DATES`, `REFIT_PAUSE_RULING`, imported into `refit/hook.py` by name (tests patch `refit.hook.REFIT_PAUSED_DATES`).
- Produces: in `_run_locked`, after `_recover_pending_install` (~:681) and before the schedule refresh (~:682-689): when `decision.due` is in `REFIT_PAUSED_DATES`, close it through the existing `_closed_event(..., _NOT_RUN, before=None, after=None)` loop (~:693-699) with one extra key `"paused_by": REFIT_PAUSE_RULING`, close `decision.close_not_run` as today, and skip the attempt (~:710-733) and the refresh. The outcome is the existing idle status `"not_due"`. `refit/cadence.py` is not touched. The digest renders a close carrying `paused_by` as `"not run; paused by the mid-season switch record."`.

- [ ] **Step 1: Write the failing tests** (small helpers built on the file's `side_step` helper, ~:218-232)

```python
def test_a_paused_due_date_closes_not_run_and_nothing_trains(tmp_path):
    outcome, steps = side_step_at(tmp_path, DEC1)          # 2026-12-01 22:05Z, recording steps
    closed = closed_events(tmp_path, "2026-12-01")
    assert [e["status"] for e in closed] == ["not run"]
    assert closed[0]["paused_by"] == REFIT_PAUSE_RULING
    assert steps.calls == [] and outcome.status == "not_due"

def test_a_paused_date_is_closed_once(tmp_path):
    side_step_at(tmp_path, DEC1); side_step_at(tmp_path, DEC1 + timedelta(days=1))
    assert len(closed_events(tmp_path, "2026-12-01")) == 1

def test_the_last_paused_date_closes_when_due(tmp_path):          # Review Focus 3
    for now in (DEC1, DEC29_2205Z, JAN26_2205Z):
        side_step_at(tmp_path, now)
    assert {d: closed_events(tmp_path, d)[0]["paused_by"] for d in REFIT_PAUSED_DATES} == dict.fromkeys(REFIT_PAUSED_DATES, REFIT_PAUSE_RULING)

# tests/unit/test_refit_notifications.py
def test_a_paused_close_reads_paused():
    assert "paused by the mid-season switch record" in date_line(closed_not_run(paused_by=REFIT_PAUSE_RULING))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_refit_hook.py tests/unit/test_refit_notifications.py -q -p no:cacheprovider`
Expected: the new tests FAIL; the two existing DEC1 tests (~:411-428, ~:445-463) still pass.

- [ ] **Step 3: Implement the pause and the digest branch**

- [ ] **Step 4: Re-point the two existing DEC1 tests** (~:411, ~:445): they test the hook's attempt mechanics, so each sets `monkeypatch.setattr("refit.hook.REFIT_PAUSED_DATES", ())` with a dated RE-POINTED line.

- [ ] **Step 5: Run the hook, wiring, cadence and notification tests**

Run: `uv run pytest tests/unit/test_refit_hook.py tests/unit/test_refit_notifications.py tests/unit/test_refit_daily_wiring.py tests/unit/test_refit_cadence.py tests/unit/test_refit_scans.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
git add refit/hook.py notify/digest_sections.py docs/guides/AUTOMATION.md tests/unit/test_refit_hook.py tests/unit/test_refit_notifications.py
git commit -m "feat(switch): the record's paused re-fit dates close not run"
```

---

### Task 3: Owner installs become generations

**Files:**
- Modify: `refit/records.py`, `refit/generations.py`, `refit/comparison.py` (`_period_starts`, ~:719-741)
- Test: Create `tests/unit/test_owner_install_records.py`

**Interfaces:**
- Produces, `refit/records.py`:
  - `OWNER_INSTALLS_FILENAME = "owner_installs.jsonl"` (under `ledger/refit/`)
  - `append_owner_install(ledger_dir: Path, event: Mapping[str, object]) -> None` and `read_owner_installs(ledger_dir: Path, *, skip_torn_tail: bool = False) -> list[dict[str, object]]`, with the same durable append, torn-tail rule and "one validator both ways" as the `dates.jsonl` functions. A missing file reads as `[]`.
  - Events (every one also carries `recorded_at_utc`): `install_intent` -- `install_date` (ET ISO date), `cutoff_utc`, `ruling`, `recipe`, `ids` {`wp`,`ats`,`ou`,`blend`,`converter`}, `bet_rule_path` (must equal `bet_rule_relpath(bet_rule_sha256)`), `bet_rule_sha256`, `code_commit` (40 hex; a `+dirty` suffix is refused), `gold_generation_digest`, `before_ids`; `installed` -- `install_date`, `after_ids`; `refused` -- `install_date`, `check`, `exception_class`. Unknown events and missing keys are refused.
- Produces, `refit/generations.py`:
  - `Generation` gains a trailing `cutoff_utc: datetime | None = None` (every existing constructor still works).
  - `installed_generations` appends, after the cadence generations and ordered by date, one generation per owner `install_intent` that no later `refused` with the same `install_date` follows: `recipe` from the record, `list_week=None`, `date_index=None`, `cutoff_utc` from the record, `bet_rule_path` from the record. Cadence generations keep `recipe=REFIT_RECIPE_ID` (:241).
  - `in_force_recipe(artifacts_dir: Path | str, ledger_dir: Path | str) -> str` = the recipe of `in_force_generation(resolve_production_artifacts(artifacts_dir), ledger_dir)`.
- Produces, `refit/comparison.py`: `_period_starts` starts a generation with `list_week is None` on the ET date of its `cutoff_utc`; a cadence generation with no date event still raises.

- [ ] **Step 1: Write the failing tests** (copy today's `ledger/refit/dates.jsonl` and its bet-rule file into `tmp_path` as the cadence history)

```python
def test_an_owner_install_is_the_newest_generation(tmp_path):
    ledger = ledger_with_owner_install(tmp_path, recipe="recipe-2026-v2", installed=True)
    newest = installed_generations(ledger)[-1]
    assert (newest.number, newest.recipe, newest.list_week) == (2, "recipe-2026-v2", None)
    assert generation_for_ids(V2_IDS, ledger).number == 2

def test_a_refused_owner_install_does_not_count(tmp_path):
    ledger = ledger_with_owner_install(tmp_path, refused=True)
    assert installed_generations(ledger)[-1].number == 1

def test_the_owner_record_refuses_a_dirty_commit_and_a_bad_bet_rule_path(tmp_path):
    with pytest.raises(RefitRecordFormatError):
        append_owner_install(tmp_path, intent(code_commit="a" * 40 + "+dirty"))
    with pytest.raises(RefitRecordFormatError):
        append_owner_install(tmp_path, intent(bet_rule_path="refit/bet_rules/other.json"))

def test_in_force_recipe_follows_latest_json(tmp_path):
    assert in_force_recipe(artifacts_naming(V2_IDS, tmp_path), ledger) == "recipe-2026-v2"

def test_the_season_panel_starts_an_owner_generation_on_its_install_date(tmp_path):
    assert period_starts_of(ledger)[2] == date(2026, 12, 1)

def test_compare_refit_arms_exits_zero_with_an_owner_generation_and_no_rows(tmp_path):   # Review Focus 4
    # the generation-1 live and shadow fixture rows the existing scripts/compare_refit_arms.py
    # tests use, plus an owner install with no rows of its own; call its entry point the way
    # those tests do
    assert run_compare_refit_arms(ledger) == 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_owner_install_records.py -q -p no:cacheprovider`
Expected: FAIL (names missing).

- [ ] **Step 3: Implement the three edits.**

- [ ] **Step 4: Run the new tests and the existing generation, record, comparison and runner tests**

Run: `uv run pytest tests/unit/test_owner_install_records.py tests/unit/test_refit_generations.py tests/unit/test_refit_records.py tests/unit/test_refit_comparison.py tests/unit/test_forward_ledger_runner.py -q -p no:cacheprovider` (use the real file names `grep -l "installed_generations\|_period_starts" tests/unit/*.py` lists)
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add refit/records.py refit/generations.py refit/comparison.py tests/unit/test_owner_install_records.py
git commit -m "feat(switch): an owner-recorded install becomes a generation"
```

---

### Task 4: Gold follows the in-force recipe

**Files:**
- Modify: `scripts/build_gold_v2.py`, `pipeline/steps.py` (`build_and_save_gold`, ~:835-869), `scripts/data_qa.py` (~:151-155 and the width check ~:872), `tests/phase33_state.py` (append one slot)
- Test: Create `tests/unit/test_gold_follows_recipe.py`; re-point `tests/integration/test_daily_scope_narrowing.py` (~:137-166, ~:207-232), `tests/integration/test_daily_run_skips.py` (~:452-456), `tests/integration/test_gold_write_scope.py` (~:1141-1188), `tests/integration/test_audit_integrity.py` (~:128-133), and every other test `grep -rn "check_gold_integrity\|current_ladder_widths\|GOLD_FEATURE_MATRICES" tests/` finds that compares widths

**Interfaces:**
- Consumes: Task 1 `INSTALLED_RECIPE`; Task 3 `in_force_recipe`; `forward_ledger.store.LEDGER_DIR`.
- Produces:
  - `scripts/build_gold_v2.py`: `check_v2_matrices(matrices: dict[str, pd.DataFrame], built_at: datetime) -> None` (the per-matrix LeakageGate with labels dropped plus `refuse_provenance_columns`, moved out of `build_gold_v2` ~:296-300, which now calls it), and `class FeatureMatrixBuilderV2Live(FeatureMatrixBuilderV2)` overriding `generate_feature_matrices` with the parent's exact signature (`excluded_game_ids` keyword-only, default `frozenset()`): `super()`, an empty result returned as is, else `add_v2_families(m, load_dataframe("games", layer="silver"), load_schedules(list(NFLVERSE_SEASONS)))` then `check_v2_matrices`. The module docstring says this class is the one sanctioned way v2 gold reaches `data/gold` (through `save_feature_matrices`), and `build_gold_v2` still never writes there.
  - `pipeline/steps.py`: `gold_builder_class() -> type[FeatureMatrixBuilder]` -- `FeatureMatrixBuilderV2Live` when `refit.generations.in_force_recipe(Path("artifacts"), LEDGER_DIR) == INSTALLED_RECIPE`, else `scripts.build_features.FeatureMatrixBuilder`; both imports inside the function. `_build_and_save_gold_with(builder_class: type[FeatureMatrixBuilder], serve_game_ids: frozenset[str], *, unplayed_through_lock: datetime | None) -> None` holds today's body; `build_and_save_gold` keeps its exact signature and calls it with `gold_builder_class()`.
  - `scripts/data_qa.py`: `GOLD_FEATURE_MATRICES` unchanged (the v1 widths, 188/188/187); `GOLD_FEATURE_MATRICES_V2 = {"features_wp": 212, "features_ats": 212, "features_ou": 211}` with a paragraph in the comment record; `expected_gold_widths(recipe: str) -> dict[str, int]`; the width check uses `expected_gold_widths(in_force_recipe(...))`.
  - `tests/phase33_state.py`: append `P332_SWITCH_V2_GOLD_WIDTH_DELTA` in the shape of the last `P332_*_GOLD_WIDTH_DELTA` slot, rung id `"10-switch-v2"`, `added` = the 24 names (the 12 `{home,away}_{off,def}_rolling_{epa_per_play,pass_epa_per_play,rush_epa_per_play}`, `home_mkt_rating`, `away_mkt_rating`, `mkt_rating_diff`, `mkt_total_sum`, `mkt_rating_coverage`, `league_points_level`, and the six `V2_DERIVED_COLUMNS`), `removed=()`, `widths_before=(188, 188, 187)`, `widths_after=(212, 212, 211)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_gold_builder_follows_the_in_force_recipe(monkeypatch):
    monkeypatch.setattr("refit.generations.in_force_recipe", lambda *a, **k: "recipe-2026-refit-v1")
    assert steps.gold_builder_class() is FeatureMatrixBuilder
    monkeypatch.setattr("refit.generations.in_force_recipe", lambda *a, **k: "recipe-2026-v2")
    assert steps.gold_builder_class() is FeatureMatrixBuilderV2Live

def test_the_live_v2_build_adds_the_families_once(monkeypatch):
    # FeatureMatrixBuilderV2.generate_feature_matrices, market_ratings, league_points_level and the
    # silver/nflverse loads patched to tiny frames, as tests/unit/test_build_gold_v2.py does
    out = FeatureMatrixBuilderV2Live().generate_feature_matrices(excluded_game_ids=frozenset())
    assert set(V2_DERIVED_COLUMNS) | {"mkt_rating_diff", "league_points_level"} <= set(out["ou"].columns)

def test_data_qa_expects_the_in_force_recipe_widths():
    assert expected_gold_widths("recipe-2026-v2") == {"features_wp": 212, "features_ats": 212, "features_ou": 211}
    assert expected_gold_widths("recipe-2026-refit-v1") == GOLD_FEATURE_MATRICES

def test_the_v2_ladder_slot_matches_data_qa():
    slot = phase33_state.P332_SWITCH_V2_GOLD_WIDTH_DELTA   # read its fields as test_data_qa_gold_width.py reads the other slots
    assert slot["widths_after"] == tuple(GOLD_FEATURE_MATRICES_V2.values())
    assert slot["widths_before"] == tuple(GOLD_FEATURE_MATRICES.values())
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_gold_follows_recipe.py -q -p no:cacheprovider`
Expected: FAIL (names missing).

- [ ] **Step 3: Implement the four edits.**

- [ ] **Step 4: Re-point the width and build-shape tests.** Fixture-based callers patch `refit.generations.in_force_recipe` to the v1 recipe (and the scope-narrowing tests keep patching the builder class they assert on). Production-reading tests (`test_gold_write_scope.py` ~:1141, `test_audit_integrity.py` ~:128) compare the measured widths with `expected_gold_widths(in_force_recipe(...))`: production gold must match the recipe in force, before and after the install.

- [ ] **Step 5: Run the new test, the re-pointed tests and the width-ladder tests**

Run: `uv run pytest tests/unit/test_gold_follows_recipe.py tests/unit/test_build_gold_v2.py tests/unit/test_data_qa_gold_width.py tests/unit/test_phase33_state_append_once.py tests/integration/test_daily_scope_narrowing.py tests/integration/test_daily_run_skips.py tests/integration/test_gold_write_scope.py tests/integration/test_audit_integrity.py tests/unit/test_refit_training.py tests/unit/test_live_skip_policy.py tests/unit/test_build_features_gold_write.py -q -p no:cacheprovider`
Expected: all PASS (production gold is still v1 and the in-force recipe is still v1).

- [ ] **Step 6: Lint and commit**

```bash
git add scripts/build_gold_v2.py pipeline/steps.py scripts/data_qa.py tests/phase33_state.py tests/unit/test_gold_follows_recipe.py <re-pointed tests>
git commit -m "feat(switch): the nightly gold and its width check follow the in-force recipe"
```

---

### Task 5: The owner install

**Files:**
- Create: `refit/owner_install.py`, `scripts/install_v2.py`
- Modify: `docs/guides/RUNBOOK.md` (a "Switch day: install v2" section: when it may run, the command, the detached launch, what success prints, what a refusal means, the checks after it)
- Test: Create `tests/unit/test_owner_install.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4; `models.v2_recipe` (`V2_INPUTS`, `V2_LABEL`, `V2_TRAIN_START`, `build_v2_estimator`); `models.v2_trainer.fit_v2`, `walk_forward_v2`; `refit.training.eligible_rows(gold, games, cutoff_utc, target).frame`; `models.artifacts.save_model_artifact(...)`, `replace_manifest`; `refit.remeasure.tune_generation_blend`, `extended_corpus`, `extended_slopes` (called as `backtest/study_1a.py:481-495` calls them); `refit.records.write_bet_rule_record`, `code_commit`; `refit.swap._code_commit_problem`; `forward_ledger.repro_key.gold_generation_key(gold_dir)`; `backtest.refit_preregistration_2026.MARKET_CONVERTER_ID`; `forward_ledger.settle.load_silver_games`; `refit/swap.py:112-213 install_generation` as the template for intent, swap, event.
- Produces, `refit/owner_install.py`:
  - `class InstallRefused(RuntimeError)`
  - `install_preconditions(games: pd.DataFrame, now_utc: datetime, in_force: str) -> None` -- refuses while any 2026 game with week < `DAY_BEFORE_RETIRED_FROM[1]` has a lock (`lock_rule.lock_frame`) after `now_utc`, and when `in_force == INSTALLED_RECIPE`.
  - `train_v2_models(gold_dir: Path, games: pd.DataFrame, cutoff_utc: datetime) -> dict[str, tuple[Pipeline, int]]` -- per target, `fit_v2` on the eligible rows from `V2_TRAIN_START`; returns the model and its training row count.
  - `v2_model_metadata(target: str, *, cutoff_utc: datetime, training_rows: int, gold_generation_digest: str, commit: str) -> dict[str, object]` -- `target`, `feature_names` (= `list(V2_INPUTS[target])`), `v2_recipe` (= `INSTALLED_RECIPE`), `cutoff_utc`, `training_rows`, `train_start`, `label`, `estimator`, `best_params` (`{"C": 1.0}` WP, `{"alpha": 1.0}` ATS/O-U), `exclude_groups` (`[]`), `thread_limit`, `gold_generation_digest`, `code_commit`, `switch_record_sha256` (`{path: normalized sha256}` of the two record files, CRLF folded to LF as `refit/witness.py:69-71` does).
  - `v2_bet_rule_payload(*, artifact_ids: Mapping[str, str], cutoff_utc: datetime, blend_weights: Mapping[str, float]) -> dict[str, object]` -- `record_kind="refit_generation_bet_rule"` (the only kind `load_bet_rule_record` reads), `refit_recipe=INSTALLED_RECIPE`, `install_path="owner"`, `artifact_ids`, `cutoff_utc`, `blend_weights`, `edge_tier_thresholds` all three `None`, per target `tune_fit` with `ev_floor_t=None`, `frozen_sd=None`, `lean=None`, `season_bias_by_season={}`, `sd_correction_bias_by_season={}`, `calibration_gate_passed=None`, and `targets` `{t: {"fallback_trigger": None}}`.
  - `@dataclass(frozen=True) class InstallOutcome: ids: dict[str, str]; bet_rule_sha256: str; blend_weights: dict[str, float]`
  - `run_owner_install(*, artifacts_dir: Path, ledger_dir: Path, gold_dir: Path, silver_dir: Path, now_utc: datetime, build_gold: Callable[[type], None]) -> InstallOutcome`. Order: preconditions; clean commit (`_code_commit_problem`); `build_gold(FeatureMatrixBuilderV2Live)`; train; save the three artifacts (WP with `preprocessing=model`, as the live WP is saved); `walk_forward_v2` at `now_utc` and `tune_generation_blend`; `InstallRefused` unless every tuned weight equals `V2_EARNED_WEIGHTS`; bet-rule record; `install_intent`; `replace_manifest` with the five ids; `installed`. A `replace_manifest` refusal appends `refused`. ANY exception after `build_gold(FeatureMatrixBuilderV2Live)` and before `replace_manifest` returns calls `build_gold(FeatureMatrixBuilder)` (the v1 gold back) and re-raises.
  - `scripts/install_v2.py`: `main(argv: list[str] | None = None) -> int`, no options; `build_gold` = `lambda cls: steps._build_and_save_gold_with(cls, frozenset(), unplayed_through_lock=datetime.now(UTC))`; prints `INSTALLED=` and the five ids, or `INSTALL_REFUSED=` and the reason (exit 2).

- [ ] **Step 1: Write the failing tests** (artifacts, ledger, gold and silver in `tmp_path`; the gold build, training and blend tune stubbed)

```python
def test_the_install_refuses_before_week_12s_last_lock():                     # Review Focus 2
    with pytest.raises(InstallRefused):
        install_preconditions(games_2026, now_utc=WEEK12_LAST_LOCK - timedelta(seconds=1), in_force="recipe-2026-refit-v1")

def test_the_install_refuses_a_second_v2_install():
    with pytest.raises(InstallRefused):
        install_preconditions(games_2026, now_utc=DEC1_15ET, in_force="recipe-2026-v2")

def test_v2_metadata_names_the_fixed_inputs_and_no_stamp_keys():
    meta = v2_model_metadata("ou", cutoff_utc=CUT, training_rows=6600, gold_generation_digest="g", commit="c" * 40)
    assert meta["feature_names"] == list(V2_INPUTS["ou"]) and "recipe_id" not in meta and "arm" not in meta

def test_the_v2_bet_rule_parses_with_no_floors_and_no_bands(tmp_path):
    path, _ = write_bet_rule_record(tmp_path, v2_bet_rule_payload(artifact_ids=V2_IDS, cutoff_utc=CUT, blend_weights=ZERO))
    fits, thresholds = load_bet_rule_record(path)
    assert dict(thresholds) == {"wp": None, "ats": None, "ou": None}

def test_a_successful_install_swaps_latest_json_and_records_it(tmp_path):
    outcome = run_owner_install(..., build_gold=spy)
    assert json.loads((tmp_path / "artifacts/latest.json").read_text())["wp"] == outcome.ids["wp"]
    assert [e["event"] for e in read_owner_installs(ledger)] == ["install_intent", "installed"]
    assert in_force_edge_thresholds(artifacts, ledger) == {"wp": None, "ats": None, "ou": None}

def test_a_nonzero_tuned_weight_refuses_and_restores_the_v1_gold(tmp_path):          # Review Focus 1
    with pytest.raises(InstallRefused):
        run_owner_install(..., build_gold=spy)          # tune stub returns ou = 0.02
    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_json_bytes(tmp_path) == BEFORE and read_owner_installs(ledger) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_owner_install.py -q -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `refit/owner_install.py` and `scripts/install_v2.py`, and write the RUNBOOK section.**

- [ ] **Step 4: Run the new tests and the scans that cover new refit/scripts code**

Run: `uv run pytest tests/unit/test_owner_install.py tests/unit/test_refit_scans.py tests/unit/test_ledger_stamping_site.py tests/unit/test_promote_models_ledger_guard.py tests/unit/test_season_floor_single_source.py tests/unit/test_one_lock_rule_source_scan.py tests/unit/test_no_market_in_artifacts.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add refit/owner_install.py scripts/install_v2.py docs/guides/RUNBOOK.md tests/unit/test_owner_install.py
git commit -m "feat(switch): the recorded owner install of the v2 models"
```

---

### Task 6: The live-manifest tests recognise a v2 install

**Files:**
- Modify: `tests/live_manifest_record.py` (`recorded_live_manifest` ~:95-118, `_date_closed` ~:85-92, `expected_cutoff` ~:132-134, `refit_metadata_problems` ~:161-195), `tests/unit/test_live_manifest_record.py` (~:73-140), `tests/unit/test_served_flags_survive_transform.py` (~:111-130, ~:186-205)

**Interfaces:**
- Consumes: Task 3 owner generations; Task 5 metadata keys.
- Produces: an owner generation counts as closed when its `installed` event exists (a hand rollback away from v2 must not quietly make the helper expect generation 1); `expected_cutoff` returns an owner generation's `cutoff_utc`; `refit_metadata_problems` checks, for `generation.recipe == INSTALLED_RECIPE`: `v2_recipe == INSTALLED_RECIPE`, `cutoff_utc` equal to the record's, `feature_names == list(V2_INPUTS[t])`, `switch_record_sha256` equal to the two record files' normalized hashes; the v1 branch is unchanged. In the served-flags test, the v1 checks apply when the in-force recipe is v1; for v2 the test asserts the WP pipeline's steps are `("impute", "scale", "model")`, that `feature_list.json == list(V2_INPUTS[t])` for every target, and that no served input is a missing-value indicator (v2 has none), with a dated RE-POINTED line saying why.

- [ ] **Step 1: Write the failing controls** in `tests/unit/test_live_manifest_record.py`

```python
def test_an_installed_owner_generation_is_the_recorded_manifest(tmp_path):
    assert recorded_live_manifest(ledger_with_v2(tmp_path, installed=True)) == V2_IDS

def test_an_owner_intent_without_installed_steps_back(tmp_path):
    assert recorded_live_manifest(ledger_with_v2(tmp_path, installed=False), latest=GEN1_IDS) == GEN1_IDS

def test_v2_metadata_with_a_wrong_input_list_is_a_problem(tmp_path):
    assert refit_metadata_problems(v2_generation, metadata_with(feature_names=["elo_diff"]))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_live_manifest_record.py -q -p no:cacheprovider`
Expected: the new controls FAIL.

- [ ] **Step 3: Implement the helper branches and re-point the served-flags test.**

- [ ] **Step 4: Run the helper's controls and every module that uses it** (they read today's live generation 1 and must stay green)

Run: `uv run pytest tests/unit/test_live_manifest_record.py tests/unit/test_served_flags_survive_transform.py tests/unit/test_season_partition_agreement.py tests/unit/test_final_fit_entry_point.py tests/unit/test_blend_revalidation_isolation.py tests/integration/test_phase33_gate_run.py tests/unit/test_temporal_display_columns.py tests/unit/test_weather_bridge_expiry.py -q -p no:cacheprovider` (correct a path with `ls tests/*/<name>` if needed)
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add tests/live_manifest_record.py tests/unit/test_live_manifest_record.py tests/unit/test_served_flags_survive_transform.py
git commit -m "test(switch): the live-manifest record recognises an owner-installed v2 generation"
```

---

### Task 7: No live decision from week 13

**Files:**
- Modify: `pipeline/daily_steps.py` (`recommend_slate`, :630)
- Test: `tests/unit/test_daily_lock_run.py` (add tests)

**Interfaces:**
- Consumes: Task 1 `DAY_BEFORE_RETIRED_FROM`.
- Produces: the first statement of `recommend_slate`: when `(slate.season, slate.week) >= DAY_BEFORE_RETIRED_FROM`, print `LEDGER_RETIRED= season=<s> week=<w> (MIDSEASON-SWITCH-RECORD.md)` and return -- before `live_skip.refuse_passed_locks`, the runner and the old writer. `forward_ledger/cutover.py` is not touched.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_week_13_slate_records_nothing(monkeypatch, capsys):
    calls = spy_on(monkeypatch, "forward_ledger.runner.record_forward_slate",
                   "backtest.weekly_bet_list.generate_weekly_bet_list", "pipeline.live_skip.refuse_passed_locks")
    recommend_slate(slate(season=2026, week=13))
    assert calls == {} and "LEDGER_RETIRED= season=2026 week=13" in capsys.readouterr().out

def test_a_week_12_slate_still_records(monkeypatch):
    calls = spy_on(monkeypatch, "forward_ledger.runner.record_forward_slate")
    recommend_slate(slate(season=2026, week=12))
    assert len(calls["record_forward_slate"]) == 1
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_daily_lock_run.py -q -p no:cacheprovider`
Expected: the week-13 test FAILS; everything else PASSES.

- [ ] **Step 3: Implement the early return.**

- [ ] **Step 4: Run the decision-path tests**

Run: `uv run pytest tests/unit/test_daily_lock_run.py tests/unit/test_forward_ledger_runner_appended.py tests/unit/test_refit_notifications.py tests/integration/test_daily_run_rehearsal.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add pipeline/daily_steps.py tests/unit/test_daily_lock_run.py
git commit -m "feat(switch): no day-before decision is recorded from 2026 week 13"
```

---

### Task 8: The bets page says the day-before system is retired

**Files:**
- Modify: `api/cache_schema.py` (beside `ARMS`, :188-190), `api/routes/pages.py` (`_build_bets_context` :596-692, `HEADLINER_STATES` :701-708, `_headliner_state` :716-735, `TRACKER_HEADINGS`/`TRACKER_CAPTIONS` :497-504), `web/templates/pages/bets.html` (after :179, before :185; the suppressed-section condition :324), `_bet_headliners.html` (find it under `web/templates/`)
- Test: `tests/api/test_bets_page.py`, `tests/api/test_this_week_headliner.py`; Create `tests/unit/test_switch_week_copies.py`

**Interfaces:**
- Consumes: Task 1 `DAY_BEFORE_RETIRED_FROM` (in tests only; `api/` may not import `backtest`).
- Produces:
  - `api/cache_schema.py`: `DAY_BEFORE_RETIRED_FROM: tuple[int, int] = (2026, 13)`, commented as the page's own copy of the record's value, pinned equal by `tests/unit/test_switch_week_copies.py`.
  - The context key `day_before_retired` = `(season, week) >= DAY_BEFORE_RETIRED_FROM` for the displayed week; when true, `bets_blocked` and `week_not_evaluated` are False and `missing_games`/`pending_games` empty. Computed in `_build_bets_context`, not in `_bet_week_coverage` (its AST guard, `test_bets_page.py:3495-3549`).
  - Headliner state `"retired"` placed ahead of `"blocked"`.
  - Copy (exact): empty state "No day-before bets from week 13 on. The day-before system was retired at the 2026 week 13 switch: the new model is more accurate than the old one but earned no weight against the betting line, so this window places no bets."; headliner "Day-before bets retired from week 13."; tracker heading, when the current slate week is 13 or later, "The retired day-before system, weeks 4-12" (the existing heading otherwise).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_switch_week_copies.py
def test_the_page_copy_equals_the_record():
    assert api.cache_schema.DAY_BEFORE_RETIRED_FROM == backtest.midseason_switch_2026.DAY_BEFORE_RETIRED_FROM

# tests/api/test_bets_page.py: a 2026 week-13 cache with no ledger rows, the clock after the
# week's first lock; request the page for that week the way the file's week-selection tests do
def test_week_13_after_its_first_lock_reads_retired_not_missing(week13_page_html):     # Review Focus 5
    assert "No day-before bets from week 13 on." in week13_page_html
    assert "This week's list is missing" not in week13_page_html
    assert "The retired day-before system, weeks 4-12" in week13_page_html

def test_week_12_is_unchanged(week12_page_html):
    assert "No day-before bets from week 13 on." not in week12_page_html
```

and a `"retired"` entry in `tests/api/test_this_week_headliner.py`'s `_BETS_PAGE_MARKERS` (~:54-60) with its fixture.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_switch_week_copies.py tests/api/test_bets_page.py tests/api/test_this_week_headliner.py -q -p no:cacheprovider`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement the constant, context, headliner state and template changes.** The forward heading and caption literals pinned at `test_bets_page.py:2068/2073` keep their text for weeks before 13; add the week-13 case beside them rather than editing them.

- [ ] **Step 4: Run the page tests and the import guards**

Run: `uv run pytest tests/unit/test_switch_week_copies.py tests/api/test_bets_page.py tests/api/test_this_week_headliner.py tests/api/test_import_guard.py tests/api/test_import_guard_bets.py tests/api/test_import_guard_ledger.py tests/api/test_import_guard_serving.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add api/cache_schema.py api/routes/pages.py web/templates <touched templates> tests/unit/test_switch_week_copies.py tests/api/test_bets_page.py tests/api/test_this_week_headliner.py
git commit -m "feat(switch): the bets page says the day-before system is retired from week 13"
```

---

### Task 9: The notifier says "retired", not "unknown"

**Files:**
- Modify: `notify/end_of_run.py` (`notify_end_of_run` :113, the bets line ~:150-155, `checkin_summary` ~:208-233), `notify/digest.py` (`collect_digest_inputs` :426; marker ~:458-463, "bets unknown" ~:516-519, game lines ~:312-349)
- Test: `tests/unit/test_notify_watcher.py`, `tests/unit/test_notify_events.py`, `tests/unit/test_notify_digest.py` (add tests)

**Interfaces:**
- Consumes: `api.cache_schema.DAY_BEFORE_RETIRED_FROM` (notify already imports `api.cache`).
- Produces: when `(report.slate_season, report.slate_week)` is at or past the constant, the run prints `NOTIFY_BETS= none (day-before window retired from 2026 week 13)` and the check-in carries `day-before=retired`; the weekly digest, for a retired week, says "No day-before bets: retired at the 2026 week 13 switch." instead of "bets unknown", and a game line says nothing about a pick time.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_retired_week_prints_why_there_are_no_bets(capsys):
    notify_end_of_run(report(slate_season=2026, slate_week=13), secrets=NO_SECRETS, now=NOW)
    assert "NOTIFY_BETS= none (day-before window retired from 2026 week 13)" in capsys.readouterr().out

def test_the_checkin_marks_the_window_retired():
    assert "day-before=retired" in checkin_summary(report(slate_season=2026, slate_week=13))

def test_the_digest_of_a_retired_week_does_not_read_unknown(tmp_path):
    # a 2026 week-13 cache with no per-week marker; the digest text through the file's existing render path
    text = digest_text_for(tmp_path, season=2026, week=13)
    assert "No day-before bets: retired at the 2026 week 13 switch." in text and "bets unknown" not in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_notify_watcher.py tests/unit/test_notify_events.py tests/unit/test_notify_digest.py -q -p no:cacheprovider`
Expected: the new tests FAIL; the existing ones (week-3 and week-16 fixtures; check the week-16 digest fixture still describes a non-retired case or re-point it with a dated line) PASS.

- [ ] **Step 3: Implement.** Keep the check-in body at or under 1000 bytes (`test_notify_watcher.py:213-231`).

- [ ] **Step 4: Run the notifier tests**

Run: the Step 2 command plus `tests/unit/test_notify_network_guard.py`. Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add notify/end_of_run.py notify/digest.py tests/unit/test_notify_watcher.py tests/unit/test_notify_events.py tests/unit/test_notify_digest.py
git commit -m "feat(switch): run messages and the weekly email say the day-before window is retired"
```

---

### Task 10: The February readout ends R6 coverage at week 12

**Files:**
- Modify: `scripts/season_readout.py` (`main` ~:1279-1303, `_coverage_section` ~:873-917, `_write_text` ~:1074-1076), `scripts/season_coverage.py` (`EvidenceFile.path` ~:204, "Silver directory" ~:1034), `docs/guides/RUNBOOK.md` (Season close section ~:1390-1455: one note on the week-12 coverage end and the new refusal)
- Test: `tests/unit/test_season_readout_binding.py`

**Interfaces:**
- Consumes: Task 1 `R6_COVERAGE_END_WEEK`, `RECORD_PATH`, imported into `scripts/season_readout.py` by name (tests patch `scripts.season_readout.R6_COVERAGE_END_WEEK`).
- Produces:
  - `classify_coverage` receives `dataclasses.replace(scope, end_week=R6_COVERAGE_END_WEEK)`; `verdict_rows`, refusal 3 (still waits for week 22) and `_scope_section` keep the frozen `scope`; the frozen value pins (~:209-222) are untouched.
  - The run prints `COVERAGE_END= week=12 reason=MIDSEASON-SWITCH-RECORD.md` and the coverage section states in one sentence that the day-before system stopped deciding at week 13 under that record, so weeks 13-22 are not "missed".
  - A new refusal before any write: a counted row (status live, week 5-22) in a week above `R6_COVERAGE_END_WEEK` refuses with `"a counted live row in week <w> is after the switch (MIDSEASON-SWITCH-RECORD.md): <game_id>"`.
  - IN-A03 (`.planning/milestones/v4.0-phases/38-standing-cleanups-season-close-readout/38-REVIEW.md:283-289`): before any write and before the one-shot binding record is spent, each rendered section is checked with `isascii()`; the first failure refuses, naming the section and the character's code point. `_write_text` keeps `encode("ascii")` as the backstop.
  - Every path the document writes is relative to `REPO_ROOT` when inside it, the raw path otherwise.

- [ ] **Step 1: Write the failing tests** (helpers built on the file's existing binding fixtures, ~:77-92)

```python
def test_coverage_ends_at_week_12_and_says_why(binding_repo):
    doc = run_binding(binding_repo)          # live bets in weeks 5-12 only
    assert "weeks 13-22" not in coverage_rows(doc) and "MIDSEASON-SWITCH-RECORD.md" in doc

def test_a_counted_row_after_the_switch_refuses(binding_repo):
    add_live_bet(binding_repo, week=13)
    assert refusal_of(run_binding(binding_repo)).startswith("a counted live row in week 13")

def test_a_non_ascii_section_refuses_before_any_write(binding_repo, monkeypatch):
    monkeypatch.setattr(season_readout, "_scope_section", lambda *a: ["Scope \u2014 weeks 5-12"])
    assert "U+2014" in refusal_of(run_binding(binding_repo)) and not readout_written(binding_repo)

def test_the_document_names_repo_relative_paths(binding_repo):
    assert "C:/" not in run_binding(binding_repo) and "data/silver" in run_binding(binding_repo)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_season_readout_binding.py -q -p no:cacheprovider`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement, then re-point the existing tests**: fixtures whose counted bets sit in week 22 (~:77-92) patch `R6_COVERAGE_END_WEEK` to 22 with a dated line; ~:1104 (an unscored week-22 game trips refusal 2) uses a week-12 game; ~:745 (`test_binding_refuses_on_todays_real_ledger`) also accepts refusal 3 ("no played week-22 game"), since once weeks 5-12 are all scored refusal 3 is the honest answer until February; ~:590 expects the relative silver path.

- [ ] **Step 4: Run the readout tests and the two RUNBOOK gate files** (they must stay green: no frozen file was edited)

Run: `uv run pytest tests/unit/test_season_readout_binding.py tests/unit/test_phase38_preregistration_ancestry.py tests/unit/test_phase34_preregistration_ancestry.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add scripts/season_readout.py scripts/season_coverage.py docs/guides/RUNBOOK.md tests/unit/test_season_readout_binding.py
git commit -m "feat(switch): R6 coverage ends at week 12; fix IN-A03 and the absolute paths"
```

- [ ] **Step 6: Branch checks, then stop for the owner (PR A)**

Run: `git -C C:\Users\jackc\Code\nfl-predict-v2 diff --name-status master...switch` -- every `M` must be a file named in Tasks 1-10; no frozen file appears. Run the scans once more: `uv run pytest tests/unit/test_season_floor_single_source.py tests/unit/test_ledger_stamping_site.py tests/unit/test_promote_models_ledger_guard.py tests/unit/test_refit_scans.py tests/unit/test_one_lock_rule_source_scan.py tests/unit/test_one_recommendation_path.py tests/unit/test_phase33_clv_report_only.py tests/unit/test_no_check_exemptions.py tests/integration/test_phase34_expected_failure_set.py -q -p no:cacheprovider` -- all PASS. Then report to the owner in plain words and ask to push, open PR A and merge it (`gh pr merge --merge --admin`) before Tue 12-01 17:00 ET, outside 17:00-17:20 ET. After the merge, fast-forward the main checkout's master.

---

### Task 11: Witness the record (PR B, after PR A is on origin/master)

**Files:**
- Create: `tests/midseason_switch_state.py`, `tests/unit/test_midseason_switch_ancestry.py`
- Modify: `tests/doc_locations.py` (`ROOT_RECORDS` ~:56-65), `CLAUDE.md` (":75 Eight history-anchored records" becomes nine, naming the new one; ASCII; keep "day before its kickoff" in Key constraints)

**Interfaces:**
- Produces, `tests/midseason_switch_state.py` (constants only, no imports; the APPEND PROTOCOL docstring of `tests/phase38_state.py:14-22`; a comment saying how each value was measured):
  - `SWITCH_RECORD_WITNESS: dict[str, dict[str, str]]` -- for each of the two record paths, `{"commit": <40 hex, git log -1 --format=%H -- <path>>, "normalized_sha256": <sha256 of the file bytes with CRLF folded to LF>}`
  - `SWITCH_RECORD_COMMITTED_AT: str` (`git show -s --format=%cI <commit>`)
  - `SWITCH_RECORD_DEADLINE_UTC: str = "2026-11-10T22:00:00+00:00"` (the corrected deadline; the witnessed commit is the 2026-11-06 correction, the last commit touching the two record files)
  - `V2_RECIPE_FILE_WITNESS: dict[str, dict[str, str]]` -- `models/v2_recipe.py` the same way
- Produces, the ancestry test, copying `tests/unit/test_phase38_preregistration_ancestry.py`'s arms: the witness covers exactly the record paths and both are tracked; (iv) each file hashes to its witness; no record contains its own hash or commit; the deadline equals `PREREGISTRATION_DEADLINE_UTC` and is UTC; (i) `git log -1 -- <path>` is the witnessed commit; (ii) a strict ancestor of HEAD; (iii) it touches exactly the two record paths; (v) `%cI` equals the witness and precedes the deadline; (vi) reachable from `refs/remotes/origin/master`; the witness file's last commit descends from the record; no quiet edit, through `uninvalidating_commits_touching_the_rule` (`tests/unit/test_phase33_preregistration_ancestry.py:186-225`); `models/v2_recipe.py` hashes to its witness and its last commit is the witnessed one. No synthetic-repo controls (that helper is already controlled in the Phase 38 test).

- [ ] **Step 1: Branch from the merged master**

```powershell
git -C C:\Users\jackc\Code\nfl-predict-v2 fetch origin
git -C C:\Users\jackc\Code\nfl-predict-v2 switch -c switch-witness origin/master
```

- [ ] **Step 2: Write the ancestry test against an empty state module, run it, see it fail**

Run: `uv run pytest tests/unit/test_midseason_switch_ancestry.py -q -p no:cacheprovider`
Expected: FAIL (witness names missing).

- [ ] **Step 3: Measure and write the state module** with the commands its comment names (PowerShell for the ET time of the measurement).

- [ ] **Step 4: Run it**

Run: the Step 2 command plus `tests/unit/test_midseason_switch_record.py tests/unit/test_old_rule_labels.py`. Expected: all PASS.

- [ ] **Step 5: Lint and commit (two commits: witness, then docs), then stop for the owner (PR B)**

```bash
git add tests/midseason_switch_state.py tests/unit/test_midseason_switch_ancestry.py
git commit -m "test: witness the mid-season switch record"
git add tests/doc_locations.py CLAUDE.md
git commit -m "docs: the mid-season switch record is the ninth history-anchored root record"
```

Ask the owner to push, open PR B and merge it (`--admin`), outside 17:00-17:20 ET. After the merge, fast-forward the main checkout's master (the scheduled daily run executes there).

---

### Task 12: Switch day -- install v2 (owner command, main checkout)

**Files:** none (runs the Task 5 command). Allowed: only once week 9's last lock (Sun 2026-11-08 18:00 ET) has passed, never 17:00-17:20 ET; best on Tue 2026-11-10, started no later than 15:00 ET (the v2 gold build, about 20 minutes, training and the walk-forward, and a possible v1 gold rebuild of about 20 minutes on a refusal must all finish before 17:00), so the 17:00 run is the first on v2. It changes the live lake and `latest.json`: run it only after the owner says go on the day.

- [ ] **Step 1: Preconditions**

Run (PowerShell): `Get-Date`; `git -C C:\Users\jackc\code\nfl-predict status --porcelain`; `git -C C:\Users\jackc\code\nfl-predict log -1 --format=%h`
Expected: ET time no later than 15:00 (the latest start); no modified or untracked file under the code paths (`refit/records.py:117-133`); HEAD is the PR B merge. Record the bytes of `artifacts/latest.json` and the modification times of `data/gold/features_{wp,ats,ou}.parquet`. Then, still before the install and outside 17:00-17:20 ET, run the Step 4 re-pointed node list (the `uv run pytest tests/integration/test_live_2026_prediction_set.py ... -q -p no:cacheprovider` command in Step 4, from the main checkout) and write down every node that is already red on v1, with its failure type. That list is Step 4's baseline: a node red here may stay red in Step 4 only with the same cause; any other red in Step 4 is new and means stop.

- [ ] **Step 2: Run the install detached** (the gold build alone is about 20 minutes; Claude Code stops long foreground jobs for memory)

```powershell
New-Item -ItemType Directory -Force C:\Users\jackc\code\nfl-predict\outputs\switch_install | Out-Null
$p = Start-Process -FilePath uv -ArgumentList 'run','python','-m','scripts.install_v2' -WorkingDirectory C:\Users\jackc\code\nfl-predict -RedirectStandardOutput C:\Users\jackc\code\nfl-predict\outputs\switch_install\install.log -RedirectStandardError C:\Users\jackc\code\nfl-predict\outputs\switch_install\install.err -PassThru -WindowStyle Hidden
$p.Id
```

Wait on the process id (Monitor with an until-loop on `Get-Process -Id`), then read `install.log`.
Expected: exit 0, `INSTALLED=` with the five ids; `latest.json` names the three new v2 ids and the new blend; `ledger/refit/owner_installs.jsonl` holds `install_intent` then `installed`. `INSTALL_REFUSED=` means stop: report the reason to the owner (a nonzero tuned weight is an owner decision; latest.json and gold are back to v1).

- [ ] **Step 3: The five tripwires and the bounded gate** (after every gold rebuild or `latest.json` swap)

```bash
uv run pytest "tests/integration/test_gate_baseline_byte_identity.py::TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::test_the_generated_block_equals_the_committed_block_byte_for_byte" "tests/integration/test_gold_rebuild_attribution.py::TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::test_no_NON_CLOCK_column_moved_in_a_protected_season" "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_every_data_column_reproduces_its_pre_resync_digest_exactly" "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_the_moved_set_is_exactly_the_build_clock" "tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields" -q -p no:cacheprovider
uv run pytest tests/integration/test_phase33_expected_failure_set.py -q -p no:cacheprovider
```

Expected: `5 failed`, each an `AssertionError`; the gate `8 passed, 1 skipped`. Any crash (KeyError, AttributeError) means stop and report.

- [ ] **Step 4: Gold and models agree, and the live-manifest tests pass on v2**

Run (working directory: the main checkout): `uv run python -c "import json,pandas as pd; m=json.load(open('artifacts/latest.json')); [print(t, set(json.load(open(f'artifacts/{m[t]}/feature_list.json'))) <= set(pd.read_parquet(f'data/gold/features_{t}.parquet').columns)) for t in ('wp','ats','ou')]"` -- three `True`. Then the Task 6 Step 4 command and `tests/integration/test_gold_write_scope.py tests/integration/test_audit_integrity.py` -- all PASS.

Then the tests re-pointed to the recipe in force by the final fix wave (2026-11-06; each verified on a scratch v2 install before the merge, so on v2 they must PASS):

```bash
uv run pytest tests/integration/test_live_2026_prediction_set.py tests/unit/test_audit_trace_epa_weather.py::TestArea2QbAdjAndCpoe::test_cpoe_in_plausible_range tests/integration/test_postlock_moves_neutralised.py::TestLiveGoldCarriesThePreMoveFacts "tests/unit/test_final_fit_entry_point.py::TestTheRecordedGenerationIsTheLaddersOwn::test_the_live_gold_is_the_generation_the_served_models_were_fitted_on" tests/unit/test_promote_models_tuned_path.py::test_the_serving_models_carry_the_ratified_exclusion tests/unit/test_promote_models_tuned_path.py::test_the_v2_serving_check_passes_the_fixed_inputs_and_catches_a_leak "tests/unit/test_temporal_display_columns.py::TestRealGold::test_the_wp_fit_survives_the_nan_the_owner_rulings_put_in_gold" "tests/integration/test_gold_rebuild_attribution.py::TestThePhase33LadderIsAttributed::test_the_level_preservation_family_is_the_builders_own_predicate" tests/integration/test_no_precoverage_constants.py tests/unit/test_build_gold_v2.py::test_build_refuses_an_absolute_path_inside_the_resolved_live_lake -q -p no:cacheprovider
```

Expected: all PASS except `test_live_2026_prediction_set.py::test_every_live_prediction_row_is_stamped_against_its_lock` if Step 1's baseline run recorded it as already red with the same cause (it is red in the worktree for reasons unrelated to the recipe). Compare every red here with the Step 1 baseline list; never accept a red that list does not hold. `test_build_refuses_an_absolute_path_inside_the_resolved_live_lake` could not be run on v2 before the merge (its live lake was the scratch gold), so it is checked here for the first time on v2.

Two nodes can only be verified after the install AND the next cache rebuild (the 17:00 run rebuilds `data/web_cache.duckdb` from the served models; before that the cache still holds what the v1 models served): run them after the Tue 11-10 17:00 run (Step 5):

```bash
uv run pytest tests/integration/test_activation_parity.py::test_part_a_cache_predictions_equal_backtest_csv tests/integration/test_activation_parity.py::test_part_b_cache_feature_importances_equal_deployed_metadata -q -p no:cacheprovider
```

Expected: both PASS. `test_part_b_cache_feature_importances_equal_deployed_metadata` is RED on the main checkout today (the generation-1 v1 re-fit artifacts carry no `feature_importances`, so the cache table is empty); the v2 artifacts write them (`refit.owner_install.v2_feature_importances`), so it turns green once the cache is rebuilt from v2. If either is still red after the rebuild, stop and report.

- [ ] **Step 5: After the Tue 11-10 17:00 ET run** (read `logs/daily_lock_runs.jsonl`'s last record and the run log): no failed step. Tuesday's run has no game tomorrow, so it takes the no-games branch: no gold build. It rebuilds the cache. No re-fit date falls due that day: the first paused date, 2026-12-01, closes `not run` with `paused_by` in `ledger/refit/dates.jsonl` when it falls due.
- [ ] **Step 5b: After the Wed 11-11 17:00 ET run** (week 10's first slate, Thursday's game; the first v2 nightly gold build and the first unplayed game through the v2 families): no failed step; the gold build used the v2 builder (212/212/211 columns, the data QA width check green); the run prints `LEDGER_RETIRED= season=2026 week=10` and `NOTIFY_BETS= none (day-before window retired from 2026 week 10)`; /bets for week 10 shows the retired message ("No day-before bets from week 10 on.") and the tracker heading "The retired day-before system, weeks 4-9".

- [ ] **Step 6: File and report**

Update `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-RESUME.md`: piece 3 done (date, PR numbers, the installed ids, tripwire result), next = piece 4's plan (early-week bets for spread and total), including the two open piece-2 review notes. `.planning/` is never committed. Then tell the owner in plain words what changed, what the site now shows, and that the day-before window will stay empty for the rest of 2026 by design.
