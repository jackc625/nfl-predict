# Study 1b (early-week line moves) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the time-stamped 2020-2024 early-week captures, whether the v2 model's disagreement with the early-week line predicts which way that line moves, and so decide per bet type whether plan 4 builds an early-week arm.

**Architecture:** A study only; nothing live changes. Three new backtest modules build (1) one row of lines per game: the earliest early-week capture, the same-source day-before capture and the nflverse close; (2) the model's inputs as of that early capture, by recomputing the four inputs that can change between the capture and the day-before lock (QB, market rating, league scoring level, wind) and checking per game that the rest cannot; (3) walk-forward v2 predictions from those inputs, two line-move regressions per target, the verdict and the report. Three v2 modules each gain one keyword-only parameter whose default keeps today's behaviour; no live code imports them.

**Tech Stack:** Python 3.13, uv, pandas, numpy, scipy, scikit-learn (through the existing v2 recipe), pytest.

**Spec:** `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-profitability-fixes-design.md` (sections 5, 6.3, 7 "Study 1b", ruling A in section 4), plus the owner rulings of 2026-11-06 below. Evidence (gitignored, read at these absolute paths): `C:\Users\jackc\code\nfl-predict\.planning\research\2026-11-05-why-not-profitable\` -- `constraints-for-midseason-changes.md` sections e and f (the early-line data facts), `code-map-model-v2.md` (signatures and traps, cited as "map Tn"), `study-1a-report.md`. The piece 1 plan `docs/superpowers/plans/2026-11-05-model-v2-study-1a.md` shows the modules this plan builds on.

**Scope.** Plan 2 of 4. Out of scope: the live reader of `data/bronze/early_lines/early_lines.jsonl`, a Tuesday information-time rule in `utils/game_lock.py`, the early arm with its pricing, records and bets page (all plan 4, and only for targets that pass here); the 2026 early boards (plan 4 grades them live).

**Owner rulings, 2026-11-06.**
1. **Pass bar, per target:** the slope must be positive at one-sided p < 0.05 against BOTH the nflverse close (the design's bar) AND the same-source day-before capture. The close comes from a different source than the early line, and a consistent difference between two sources can make a model look as if it predicts the move.
2. **The Prediction Tracker files are left out.** Their lines carry no posting time, so the inputs cannot be fenced to them. The files stay on disk.

**Calls made in this plan (the owner can overrule at review).**
1. **The early line** of a game is its earliest `odds_timeline` capture taken after every game of the season's earlier weeks has ended (kickoff + 4 h) and at most 7 days before kickoff. Measured 2026-11-06: 1,343 of the 1,408 games of 2020-2024 have one, 1,279 of them Tuesday 11:55 ET; the other 64 fall later in the week, after a rescheduled game or a missing Tuesday capture. The capture's own timestamp is the information time; no weekday arithmetic anywhere.
2. **"The same repaired model" (design 6.3):** the v2 models train walk-forward on the day-before v2 gold exactly as in Study 1a; only the rows they score are built as of the early capture.
3. **Which inputs are rebuilt.** Of the v2 inputs only QB, the market rating, the league scoring level and wind can change between the early capture and the lock. They are rebuilt; wind is unknown on Tuesday, so it is left blank and the recipe's median imputer fills it. Elo, success rate, opponent-adjusted EPA, pace and home field rest only on finished games. A per-game check (each team's previous game ended by the early capture) confirms that; a game failing it is left out and counted.
4. **QB as of the early capture** is the team's latest primary passer. The 2002-2024 depth charts are keyed by week with no publication time, so they are never read. This is conservative: the live arm will see Tuesday's charts.
5. **WP:** no early moneyline exists for 2020-2024, so its early, day-before and close market log-odds all come from the spread through the converter's prior-only season slope (log-odds = slope x spread). 2020 has no prior slope and is left out, as in Study 1a.
6. Week 1 and the postseason are included; the 7-day window keeps two-week lookahead boards out.

## File structure

| File | New / edit | Responsibility |
|---|---|---|
| `features/qb_early.py` | new | QB tracker that never reads an untimed depth chart |
| `features/market_rating.py`, `features/league_level.py` | edit | `locks=` keyword: the information time per game, default = the lock rule |
| `models/v2_trainer.py` | edit | `prediction_rows=` keyword: score supplied rows instead of gold rows |
| `backtest/study_1b_lines.py` | new | the early line, the day-before line and the close, one row per game |
| `backtest/study_1b_inputs.py` | new | the v2 input rows as of each game's early capture |
| `backtest/study_1b.py` | new | the two move tests per target, the verdict, the report |
| `scripts/run_study_1b.py` | new | the command line |

## Global Constraints

- Work in the worktree `C:\Users\jackc\Code\nfl-predict-v2` on a new branch `study-1b` from master `9608e7d` (Task 1 step 1). Its `data/bronze`, `data/silver`, `data/gold`, `data/baselines` and `data/optuna` are junctions into the LIVE lake: nothing in this plan writes under `data/`. The v2 gold `outputs/gold_v2/` (built 2026-11-05) is read, never rebuilt, so the five tripwires and the bounded gate are not due.
- Existing tracked files edited: only `features/market_rating.py`, `features/league_level.py`, `models/v2_trainer.py` and their tests `tests/unit/test_market_rating.py`, `tests/unit/test_league_level.py`, `tests/unit/test_v2_trainer.py`, each by adding one keyword-only parameter whose default `None` keeps today's result. Everything else is a new file under an existing top-level folder.
- Never call `save_dataframe`, `save_feature_matrices`, `save_model_artifact`, `BaseTrainer.save`, `MarketBlender.save_blend_artifacts`, `train_generation(save=True)` or anything in map s.6.5's "must not be called" list. Never write `artifacts/`. Output goes only to `outputs/study_1b/`, through `backtest.study_1a.refuse_data_dir`.
- Do not run the study between 17:00 and 17:20 ET (it can block the daily run's DuckDB writer, map T2), nor while a pytest session runs. Read ET time with PowerShell, never Git Bash `TZ=`.
- Tests: only named files or node ids, `uv run pytest <file>::<test> -p no:cacheprovider`. Never bare `pytest`, a directory, or the full suite. No `xfail`, no skip. Tests write only under `tmp_path`.
- Lint every touched file: `uv run ruff check --no-fix --no-cache <files>`, `uv run ruff format <files>`, `uv run pyright <files>` (0 errors).
- Import the lock rule only as `import utils.game_lock as lock_rule`. No function anywhere calls `.weekday()` / `.isoweekday()` together with a fixed hour (`tests/unit/test_one_lock_rule_source_scan.py`). Read nflverse only through `data.upstream_pin.load_schedules`.
- Flip the `odds_timeline` spread exactly once, through `models.market_probability.timeline_spread_to_home_fav_margin`. The day-before corpus `market_spread` is already home-positive: never flip it again (map T6).
- No dict key or frame column named `arm`, `recipe_id` or any other ledger stamp name; no name containing `FIRST_SEASON` bound to a year.
- ASCII only in code, tests and reports; no emoji.
- Commit after each task on `study-1b`, ending the message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01LVgorLUwwGjKecmY3yVBge`.
- Do not push, open a PR or merge until the owner has read the report (Task 7).

## Review Focus

1. **Weeks after a rescheduled game** (2020 Tuesday and Wednesday games, 2024 Christmas Wednesday). Expected: the early line is taken after that game ended; a game with no capture strictly before its day-before capture is left out and counted, never paired with the previous week's board. Pinned in Task 4.
2. **Lookahead boards** (games listed two weeks out, week 1, the gap before the Super Bowl). Expected: no capture more than 7 days before kickoff is used. Pinned in Task 4.
3. **Three sign conventions** (timeline inverted, corpus and nflverse home-positive). Expected: a home favourite of 3 reads +3 in the early, day-before and close columns. Pinned in Task 4.
4. **A Thursday game of the same week**, kicked off after the early capture but before a Sunday game's lock. Expected: its line and its points never reach the Sunday game's early market rating or league level. Pinned in Task 2.
5. **A team whose previous game ended after the early capture.** Expected: the game is left out and counted, never scored with day-before inputs. Pinned in Task 5.

---

### Task 1: A QB tracker that never reads an untimed depth chart

**Files:**
- Create: `features/qb_early.py`
- Test: `tests/unit/test_qb_early.py`

**Interfaces:**
- Consumes: `features.qb_v2.QBTrackerV2`; from `features.qb_tracking`: `DepthChartQBs` (frozen dataclass `qb1, qb2, published_at`), `DepthChartSchemaError`, `QBTracker.resolve_depth_chart_qbs(self, depth_charts, season, week, team, lock) -> DepthChartQBs` (`qb_tracking.py:549-626`), `QBTracker.build_features(self, games_df, as_of_datetime, *, target_season=None, target_week=None, lock_frame=None) -> DataFrame[game_id, team, qb_adjustment]` (`:702`).
- Produces: `class QBTrackerEarly(QBTrackerV2)` overriding only `resolve_depth_chart_qbs`. A non-empty frame with a `week` column and no `dt` column (the 2002-2024 schema) returns `DepthChartQBs(None, None, None)`, so `_game_inputs` falls back to the team's latest primary passer among games ended by the lock it was passed. Any other frame goes to `super()`: a `dt` chart is read at or before the lock, and a frame with neither column still raises `DepthChartSchemaError`. Task 5 passes it an early lock frame through `build_features(..., lock_frame=...)`.

- [ ] **Step 1: Create the branch and commit this plan on it**

```powershell
git -C C:\Users\jackc\Code\nfl-predict-v2 switch -c study-1b master
Move-Item C:\Users\jackc\code\nfl-predict\docs\superpowers\plans\2026-11-06-study-1b-early-week.md C:\Users\jackc\Code\nfl-predict-v2\docs\superpowers\plans\
git -C C:\Users\jackc\Code\nfl-predict-v2 add docs/superpowers/plans/2026-11-06-study-1b-early-week.md
git -C C:\Users\jackc\Code\nfl-predict-v2 commit -m "docs: implementation plan for Study 1b (early-week line moves)"
```

Expected: `git -C C:\Users\jackc\Code\nfl-predict-v2 log -2 --format=%h` shows the new commit above `9608e7d`, and the main checkout has no copy of the plan left (an untracked copy would block the later merge).

- [ ] **Step 2: Write the failing tests**

```python
LOCK = pd.Timestamp("2024-09-28 18:00", tz="America/New_York")

def test_a_week_keyed_chart_is_never_read():
    chart = pd.DataFrame({"season": [2024], "week": [4], "club_code": ["KC"],
                          "position": ["QB"], "depth_team": ["1"], "gsis_id": ["00-0099999"]})
    assert QBTrackerEarly().resolve_depth_chart_qbs(chart, 2024, 4, "KC", LOCK) == DepthChartQBs(None, None, None)

def test_a_timed_chart_is_read_only_up_to_the_lock():
    # two dt snapshots for KC: QB1 "00-000000A" at LOCK - 1 h, QB1 "00-000000B" at LOCK + 1 s
    assert QBTrackerEarly().resolve_depth_chart_qbs(timed_chart, 2025, 4, "KC", LOCK).qb1 == "00-000000A"

def test_a_chart_with_neither_schema_still_raises():
    with pytest.raises(DepthChartSchemaError):
        QBTrackerEarly().resolve_depth_chart_qbs(chart.drop(columns=["week"]), 2024, 4, "KC", LOCK)

def test_the_last_game_passer_replaces_a_chart_backup():
    # Inject _depth_chart_cache and _pbp_cache as tests/unit/test_qb_tracking.py::TestBuildFeatures
    # does (build small frames the way its mock_pbp_data / mock_depth_charts / mock_games_df do).
    # planted_chart = the real chart with KC's week-4 QB1 replaced by an unseen "00-0099999".
    starter = kc_week4_value(QBTrackerV2(), real_chart)
    assert kc_week4_value(QBTrackerV2(), planted_chart) == -0.5
    assert kc_week4_value(QBTrackerEarly(), planted_chart) == pytest.approx(starter)
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/unit/test_qb_early.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: features.qb_early`).

- [ ] **Step 4: Implement `QBTrackerEarly`** in `features/qb_early.py` (the one override; its docstring says why untimed charts are skipped and that the caller supplies the early lock frame).

- [ ] **Step 5: Run the new tests and the parents' tests**

Run: `uv run pytest tests/unit/test_qb_early.py tests/unit/test_qb_v2.py tests/unit/test_qb_tracking.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
git add features/qb_early.py tests/unit/test_qb_early.py
git commit -m "feat(1b): a QB tracker that never reads an untimed depth chart"
```

---

### Task 2: Market rating and league level at a supplied information time

**Files:**
- Modify: `features/market_rating.py` (`market_ratings`), `features/league_level.py` (`league_points_level`)
- Test: `tests/unit/test_market_rating.py`, `tests/unit/test_league_level.py` (add tests; reuse each file's `_game` / `_nfl` / `_prior_season` helpers)

**Interfaces:**
- Produces:
  - `market_ratings(games: pd.DataFrame, nflverse: pd.DataFrame, *, locks: pd.Series | None = None) -> pd.DataFrame`
  - `league_points_level(games: pd.DataFrame, *, locks: pd.Series | None = None) -> pd.DataFrame`
  - `locks` is a `game_id`-indexed series of tz-aware instants, the shape `lock_rule.lock_frame` returns. `None` means `lock_rule.lock_frame(...)` exactly as today. When given it must name every game the function outputs (market: every row of `games`; level: every row with a kickoff), else `ValueError` naming the first ten missing ids. Output columns are unchanged, and every `information_time` is at or before that game's supplied instant.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_market_rating.py
def test_a_supplied_earlier_time_leaves_out_a_game_kicked_off_after_it():
    # target 2024_W03_BUF@KC Sun 2024-09-22 13:00 ET (lock Sat 18:00); one week-1 and one
    # week-2 game involving BUF or KC, both kicked off before 2024-09-17; Thursday
    # 2024_W03_NYJ@NE kicks off 2024-09-19 20:15 ET with spread -6.0 (the latest kickoff
    # before the lock)
    early = pd.Timestamp("2024-09-17 11:55", tz="America/New_York")
    locks = lock_rule.lock_frame(games)
    locks[gid] = early
    default = market_ratings(games, nfl).set_index("game_id")
    moved = market_ratings(games, nfl, locks=locks).set_index("game_id")
    assert default.loc[gid, "information_time"] == thursday_kickoff_utc
    assert moved.loc[gid, "information_time"] <= early

def test_supplied_locks_must_name_every_game():
    with pytest.raises(ValueError, match="2024_W03_BUF@KC"):
        market_ratings(games, nfl, locks=locks.drop("2024_W03_BUF@KC"))

# tests/unit/test_league_level.py
def test_a_supplied_earlier_time_leaves_out_a_game_ended_after_it():
    # prior season mean 44.0; 2025_THU kicks off 2025-11-06 20:15 ET, total 50;
    # target 2025_SUN 2025-11-09 13:00 ET (lock Sat 18:00 reads the Thursday game)
    locks = lock_rule.lock_frame(games)
    locks["2025_SUN"] = pd.Timestamp("2025-11-04 11:55", tz=ET)
    default = league_points_level(games).set_index("game_id")
    moved = league_points_level(games, locks=locks).set_index("game_id")
    assert default.loc["2025_SUN", "league_points_level"] == pytest.approx((50 + 32 * 44.0) / 33)
    assert moved.loc["2025_SUN", "league_points_level"] == pytest.approx(44.0)

def test_supplied_locks_must_name_every_game():
    with pytest.raises(ValueError, match="2025_SUN"):
        league_points_level(games, locks=locks.drop("2025_SUN"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_market_rating.py tests/unit/test_league_level.py -p no:cacheprovider`
Expected: the four new tests FAIL (`TypeError: unexpected keyword argument 'locks'`); the existing ones PASS.

- [ ] **Step 3: Add the `locks` keyword to both functions.** Only the line that builds the locks changes, plus the coverage check.

- [ ] **Step 4: Run both test files and the gold-build tests (the callers are unchanged)**

Run: `uv run pytest tests/unit/test_market_rating.py tests/unit/test_league_level.py tests/unit/test_build_gold_v2.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add features/market_rating.py features/league_level.py tests/unit/test_market_rating.py tests/unit/test_league_level.py
git commit -m "feat(1b): market rating and league level at a supplied information time"
```

---

### Task 3: Walk-forward predictions from supplied rows

**Files:**
- Modify: `models/v2_trainer.py` (`walk_forward_v2`, `_walk_forward_target`)
- Test: `tests/unit/test_v2_trainer.py` (add tests; reuse `_tiny_world`)

**Interfaces:**
- Produces: `walk_forward_v2(gold_dir: Path, games: pd.DataFrame, generation_cutoff: datetime, *, prediction_rows: Mapping[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]`. `None` keeps today's behaviour. When given, it must hold a frame for every target in `V2_TARGETS` (`ValueError` otherwise), each carrying `game_id, season, week` and that target's `V2_INPUTS`. Each fold still trains on the gold's eligible rows, but predicts only the fold's games present in `prediction_rows[target]`, scored from those rows; a fold with none is skipped before fitting. Output columns, the never-trained-on check and the one-prediction-per-game check are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_supplied_rows_change_only_what_is_predicted(tmp_path):
    gold_dir, games = _tiny_world(tmp_path)
    cutoff = datetime(2026, 9, 1, tzinfo=UTC)
    default = walk_forward_v2(gold_dir, games, cutoff)
    rows = {t: pd.read_parquet(gold_dir / f"features_{t}.parquet").query("season == 2024")
            .reset_index(drop=True) for t in ("wp", "ats", "ou")}
    moved_id = rows["ats"].loc[0, "game_id"]
    rows["ats"].loc[0, "elo_diff"] += 5.0
    out = walk_forward_v2(gold_dir, games, cutoff, prediction_rows=rows)
    assert set(out["ats"]["season"]) == {2024}
    both = out["ats"].merge(default["ats"], on="game_id", suffixes=("", "_default"))
    same = both["game_id"] != moved_id
    np.testing.assert_allclose(both.loc[same, "prediction"], both.loc[same, "prediction_default"])
    assert both.loc[~same, "prediction"].iloc[0] != both.loc[~same, "prediction_default"].iloc[0]

def test_supplied_rows_must_cover_every_target(tmp_path):
    gold_dir, games = _tiny_world(tmp_path)
    rows = {"ats": pd.read_parquet(gold_dir / "features_ats.parquet")}
    with pytest.raises(ValueError):
        walk_forward_v2(gold_dir, games, datetime(2026, 9, 1, tzinfo=UTC), prediction_rows=rows)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_v2_trainer.py -p no:cacheprovider`
Expected: the two new tests FAIL (unexpected keyword); the existing four PASS.

- [ ] **Step 3: Add the `prediction_rows` keyword.**

- [ ] **Step 4: Run the trainer tests**

Run: `uv run pytest tests/unit/test_v2_trainer.py tests/unit/test_study_1a.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add models/v2_trainer.py tests/unit/test_v2_trainer.py
git commit -m "feat(1b): walk-forward v2 predictions from supplied rows"
```

---

### Task 4: One row of lines per game

**Files:**
- Create: `backtest/study_1b_lines.py`
- Test: `tests/unit/test_study_1b_lines.py`

**Interfaces:**
- Consumes: `models.market_probability.timeline_spread_to_home_fav_margin`, `OWNED_LINE_SEASONS` (2020-2024); `models.blending_data.load_tuning_period_data(silver_dir) -> PrelockTuningCorpus` (`.frame`: `game_id, snapshot_ts, market_spread` home-positive, `market_total`: the latest capture at or before the lock carrying both); `backtest.study_1a.close_lines(schedules) -> DataFrame[game_id, close_spread, close_total, close_home_prob]`; `data.upstream_pin.load_schedules`; `features.provenance.DECLARED_GAME_DURATION` (4 h). Silver `odds_timeline`: `game_id, snapshot_ts (UTC), spread (INVERTED), total`.
- Produces:
  - `EARLY_WINDOW: pd.Timedelta = pd.Timedelta(days=7)`
  - Reasons: `NO_EARLY_CAPTURE = "no_early_capture"`, `NO_DAY_BEFORE = "no_day_before_line"`, `NOT_BEFORE_DAY_BEFORE = "early_capture_not_before_day_before_line"`, `NO_CLOSE = "no_close"`
  - `LINE_COLUMNS = ("game_id", "season", "week", "early_ts", "early_spread", "early_total", "day_before_ts", "day_before_spread", "day_before_total", "close_spread", "close_total")`; every spread home-positive.
  - `@dataclass(frozen=True) class StudyLines: frame: pd.DataFrame; excluded: pd.DataFrame` (`excluded`: `game_id, season, week, reason`).
  - `early_captures(timeline: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame` with `game_id, early_ts, early_spread, early_total`. A capture of game G is eligible when `snapshot_ts` is at or after the latest end (kickoff + `DECLARED_GAME_DURATION`) of every game of G's season with a lower week, and `kickoff_G - EARLY_WINDOW <= snapshot_ts < kickoff_G`. The earliest eligible capture is taken; a game with none is absent.
  - `study_lines(timeline: pd.DataFrame, games: pd.DataFrame, day_before: pd.DataFrame, close: pd.DataFrame) -> StudyLines`. Pure. `day_before` has the corpus columns `game_id, snapshot_ts, market_spread, market_total`. Every game in `games` lands in exactly one of `frame` / `excluded`, with the first failing reason in the order `NO_EARLY_CAPTURE`, `NO_DAY_BEFORE`, `NOT_BEFORE_DAY_BEFORE` (requires `early_ts < day_before_ts`), `NO_CLOSE`.
  - `load_study_lines(silver_dir: Path, seasons: tuple[int, ...] = OWNED_LINE_SEASONS) -> StudyLines`: reads silver `odds_timeline` and `games` (seasons only), the corpus, and the close from `load_schedules(list(seasons))`.

- [ ] **Step 1: Write the failing tests**

Build `games` (`game_id, season, week, kickoff_et`), `timeline` (`game_id, snapshot_ts, spread, total`), `day_before` and `close` frames in the test. ET instants below; compare as UTC.

```python
def test_the_early_line_is_the_first_capture_after_the_previous_week_ended():
    # week 1: Sun 2024-09-08 13:00 and Mon 2024-09-09 20:15 (ends Tue 00:15)
    # target week 2 Sun 2024-09-15 13:00; captures Fri 09-06 17:55, Tue 09-10 11:55,
    # Wed 09-11 11:55, Fri 09-13 17:55 (the day-before line)
    assert row["early_ts"] == et("2024-09-10 11:55")

def test_a_rescheduled_earlier_week_game_moves_the_early_line_later():   # Review Focus 1
    # the week-1 Monday game moved to Wed 2024-09-11 20:15 (ends Thu 00:15); a Thu 09-12 11:55 capture exists
    assert row["early_ts"] == et("2024-09-12 11:55")

def test_no_capture_before_the_day_before_line_excludes_the_game():      # Review Focus 1
    # same as above without the Thursday capture: the first eligible capture IS the day-before one
    assert excluded_reason(out, TARGET) == NOT_BEFORE_DAY_BEFORE

def test_no_capture_more_than_seven_days_before_kickoff_is_used():       # Review Focus 2
    # week-1 target Sun 2024-09-08 13:00: captures Tue 08-27 11:55 (12 days) and Tue 09-03 11:55
    assert row["early_ts"] == et("2024-09-03 11:55")
    # a second week-1 game whose only capture is 12 days out
    assert excluded_reason(out, OTHER) == NO_EARLY_CAPTURE

def test_every_line_is_home_positive():                                   # Review Focus 3
    # a home favourite of 3: timeline spread -3.0, corpus market_spread +3.5, close_spread +3.0
    assert (row["early_spread"], row["day_before_spread"], row["close_spread"]) == (3.0, 3.5, 3.0)

def test_every_game_is_kept_or_excluded_exactly_once():
    assert sorted([*out.frame["game_id"], *out.excluded["game_id"]]) == sorted(games["game_id"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_study_1b_lines.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the module.** The floor per (season, week) is the running maximum of the lower weeks' ends, computed once; the window and floor are compared in UTC.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_study_1b_lines.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add backtest/study_1b_lines.py tests/unit/test_study_1b_lines.py
git commit -m "feat(1b): early, day-before and closing lines, one row per game"
```

---

### Task 5: The model's inputs as of the early capture

**Files:**
- Create: `backtest/study_1b_inputs.py`
- Test: `tests/unit/test_study_1b_inputs.py`

**Interfaces:**
- Consumes: Task 1 `QBTrackerEarly`; Task 2 `market_ratings(..., locks=)`, `league_points_level(..., locks=)`; `scripts.build_gold_v2.add_v2_derived_columns(frame)`; `lock_rule.lock_frame`, `lock_rule.ET`; `features.provenance.DECLARED_GAME_DURATION`. Gold matrices have no team columns; teams come from silver `games` (map s.1.7).
- Produces:
  - `EARLY_REPLACED_COLUMNS = ("home_qb_adjustment", "away_qb_adjustment", "home_mkt_rating", "away_mkt_rating", "mkt_rating_diff", "mkt_total_sum", "mkt_rating_coverage", "league_points_level", "wind_mph")`. `add_v2_derived_columns` then recomputes `qb_value_diff` and the other derived inputs from the per-side columns.
  - Reasons: `NOT_SETTLED = "inputs_not_settled_at_early_capture"`, `NO_GOLD_ROW = "no_gold_row"`
  - `early_lock_frame(games: pd.DataFrame, early_times: pd.Series) -> pd.Series`: `lock_rule.lock_frame(games)` with each early-timed game's value replaced by its early time (converted to `lock_rule.ET`). `ValueError` when an early time is after that game's own lock, or names a game not in `games`.
  - `settled_by_early_time(games: pd.DataFrame, early_times: pd.Series) -> pd.Series`: a bool per early-timed `game_id`, True when for both teams the latest game the team kicked off before G (any season) ended (kickoff + `DECLARED_GAME_DURATION`) at or before G's early time. A team with no earlier game counts as settled.
  - `replace_early_columns(gold: pd.DataFrame, games: pd.DataFrame, qb: pd.DataFrame, ratings: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame`: the gold rows of the games in `ratings`, with `home_/away_qb_adjustment` from `qb` (`game_id, team, qb_adjustment`, team matched to the game's home and away team), the five market columns from `ratings`, `league_points_level` from `level`, `wind_mph` NaN, then `add_v2_derived_columns`. Every other column is unchanged. `ValueError` when a game lacks a QB row for either team.
  - `@dataclass(frozen=True) class EarlyInputs: rows: dict[str, pd.DataFrame]; excluded: pd.DataFrame` (`game_id, reason`)
  - `build_early_inputs(gold_dir: Path, games: pd.DataFrame, nflverse: pd.DataFrame, early_times: pd.Series) -> EarlyInputs`. `games` is all silver games; the families run on the seasons from the first early season minus one through the last. Locks come from `early_lock_frame`. QB comes from `QBTrackerEarly().build_features(scope, datetime.now(UTC), target_season=s, target_week=w, lock_frame=locks)` once per (season, week) holding an early-timed game. Then `market_ratings(scope, nflverse, locks=locks)` and `league_points_level(scope, locks=locks)`. It raises `RuntimeError` if any early-timed game's rating or level `information_time` is after its early time. Unsettled games are excluded with `NOT_SETTLED`, and games missing from a gold matrix with `NO_GOLD_ROW`. `rows` holds one frame per target, from `gold_dir/features_{t}.parquet`.

- [ ] **Step 1: Write the failing tests**

```python
def test_early_lock_frame_replaces_only_early_timed_games():
    locks = early_lock_frame(games, pd.Series({SUN: et("2024-09-10 11:55")}))
    assert locks[SUN] == et("2024-09-10 11:55")
    assert locks[OTHER] == lock_rule.lock_frame(games)[OTHER]

def test_an_early_time_after_the_lock_is_refused():
    with pytest.raises(ValueError):
        early_lock_frame(games, pd.Series({SUN: et("2024-09-14 18:01")}))   # lock is 09-14 18:00

def test_a_team_whose_previous_game_ends_after_the_early_time_is_not_settled():   # Review Focus 5
    # KC played Wed 2020-12-02 20:00 ET (ends Thu 00:00); KC's target Sun 2020-12-06;
    # early time Tue 2020-12-01 11:55 -> not settled; a game of two teams that last played Sun 11-29 -> settled
    out = settled_by_early_time(games, early_times)
    assert not out[KC_GAME] and out[OTHER_GAME]

def test_replaced_columns_come_from_the_early_families_and_wind_is_unknown():
    # one gold row with elo_diff 87.0, wind_mph 14.0; qb home 1.2, away -0.5; ratings diff 2.5; level 46.0
    out = replace_early_columns(gold, games, qb, ratings, level)
    assert out.loc[0, "qb_value_diff"] == pytest.approx(1.7)
    assert out.loc[0, "mkt_rating_diff"] == 2.5 and out.loc[0, "league_points_level"] == 46.0
    assert np.isnan(out.loc[0, "wind_mph"]) and out.loc[0, "elo_diff"] == 87.0

def test_build_refuses_a_family_read_after_the_early_time(monkeypatch, tmp_path):
    # stub QBTrackerEarly.build_features, league_points_level, and market_ratings (whose
    # information_time is one second after the early time) inside backtest.study_1b_inputs
    with pytest.raises(RuntimeError):
        build_early_inputs(tmp_gold_dir, games, nflverse, early_times)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_study_1b_inputs.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the module.**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_study_1b_inputs.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add backtest/study_1b_inputs.py tests/unit/test_study_1b_inputs.py
git commit -m "feat(1b): v2 input rows as of each game's early capture"
```

---

### Task 6: Study 1b -- the moves, the verdict and the report

**Files:**
- Create: `backtest/study_1b.py`, `scripts/run_study_1b.py`
- Test: `tests/unit/test_study_1b.py`

**Interfaces:**
- Consumes: Task 3 `walk_forward_v2(..., prediction_rows=)`; Task 4 `load_study_lines`; Task 5 `build_early_inputs`; `backtest.market_info.information_beyond_line(outcome, line, model) -> InfoResult` (OLS of outcome - line on an intercept and model - line); `models.market_probability.oof_market_probability(frame[season, home_fav_margin], slopes)`, `load_market_probability_artifact`, `NoPriorFoldError`, `OWNED_LINE_SEASONS`; `backtest.refit_preregistration_2026.MARKET_CONVERTER_ID`; `refit.remeasure.extended_slopes`; `backtest.study_1a.TARGETS`, `live_artifacts_dir`, `refuse_data_dir`; `forward_ledger.settle.load_silver_games`; `data.upstream_pin.load_schedules`; `backtest.simulation.SLIPPAGE_POINTS` (0.5); `refit.records.code_commit`.
- Produces:
  - `STUDY_SEASONS = OWNED_LINE_SEASONS`; `PASS_ALPHA = 0.05`; `COMPARISONS = ("close", "day_before")`; `DEFAULT_OUT_DIR = Path("outputs/study_1b")`
  - `@dataclass(frozen=True) class MoveTest: info: InfoResult | None; note: str; games: int; mean_abs_move: float | None; mean_abs_departure: float | None`
  - `move_test(later: ArrayLike, early: ArrayLike, model: ArrayLike) -> MoveTest`: `information_beyond_line(later, early, model)`, so the slope is that of (later - early) on (model - early). A `ValueError` becomes `info=None` with its text as `note`. `games` and the two means (|later - early|, |model - early|) are taken over rows where all three are finite.
  - `passes(close: MoveTest, day_before: MoveTest) -> bool`: True only if both have `info` with `coef > 0` and `p_one_sided < PASS_ALPHA` (ruling 1).
  - `wp_market_log_odds(frame: pd.DataFrame, spread_column: str, slopes: Mapping[int, float]) -> np.ndarray`: the logit of `oof_market_probability` with `frame[spread_column]` as `home_fav_margin` (= season slope x spread); raises `NoPriorFoldError` like it.
  - `@dataclass(frozen=True) class TargetVerdict: close: MoveTest; day_before: MoveTest; passed: bool; share_clearing_slippage: float | None` -- ATS and O/U: the share of the close test's games where |close slope x (model - early)| >= `SLIPPAGE_POINTS`, i.e. how often a predicted move would clear the half-point of slippage; WP: None.
  - `@dataclass(frozen=True) class Study1bResult: generation_cutoff_utc: str; verdicts: dict[str, TargetVerdict]; counts: dict[str, int]; code_commit: str | None; v2_gold_manifest: dict[str, Any] | None`
  - `run_study_1b(v2_gold_dir: Path, silver_dir: Path, generation_cutoff: datetime, *, artifacts_dir: Path | None = None) -> Study1bResult`. Writes nothing. In order: `load_silver_games(silver_dir)`; `load_study_lines(silver_dir)`; `build_early_inputs(v2_gold_dir, games, load_schedules(list(range(STUDY_SEASONS[0] - 1, STUDY_SEASONS[-1] + 1))), lines.frame.set_index("game_id")["early_ts"])`; then `walk_forward_v2(v2_gold_dir, games, generation_cutoff, prediction_rows=inputs.rows)`, joined to the lines on `game_id`. ATS uses the spreads, O/U the totals, both against the ATS / O/U predictions. WP uses `wp_market_log_odds` of the early, day-before and close spreads against the logit of the WP prediction clipped to [1e-6, 1 - 1e-6]; seasons without a slope are dropped and counted as `wp_no_prior_fold_converter`. `counts` holds every exclusion reason from Tasks 4 and 5, the games with lines, and the games predicted per target.
  - `summary_lines(result: Study1bResult) -> list[str]`: at most 8 lines. One per target: passes / does NOT pass, both slopes with one-sided p and games, plus the slippage share for a passing ATS or O/U. Then which targets plan 4 may build an early arm for, and that a failed target gets none while the daily early boards keep recording.
  - `render_report(result) -> str` and `write_report(result: Study1bResult, out_dir: Path) -> None`: `study_1b.json` and `report.md`, both written as ASCII, through `refuse_data_dir(out_dir, what="the Study 1b report", default_dir=DEFAULT_OUT_DIR)`. The report gives the summary, then context notes (inputs as of the capture, QB from the last game's passer, wind blank, WP through the converter, why both comparisons must pass, PT left out by the 2026-11-06 ruling), a table per target and comparison (games, slope, SE, one-sided p, mean |move|, mean |model - early|), and the counts.
  - `scripts/run_study_1b.py: main(argv: list[str] | None = None) -> int` with `--v2-gold-dir` (default `outputs/gold_v2`), `--out-dir` (default `outputs/study_1b`), `--artifacts-dir` (default beside the live lake). It refuses a bad `--out-dir` before the run, uses the run instant as the cutoff, and prints the summary.

- [ ] **Step 1: Write the failing tests**

```python
def _moves(seed=11, n=1300):
    rng = np.random.default_rng(seed)
    early, edge = rng.normal(0, 6, n), rng.normal(0, 2.5, n)
    return early + 0.2 * edge + rng.normal(0, 1.0, n), early, early + edge

def test_a_planted_move_toward_the_model_is_found():
    later, early, model = _moves()
    r = move_test(later, early, model)
    assert r.info.coef == pytest.approx(0.2, abs=0.05) and r.info.p_one_sided < 0.05 and r.games == 1300

def test_a_missing_early_line_drops_only_that_row():
    later, early, model = _moves()
    early[3] = np.nan
    assert move_test(later, early, model).games == 1299

def test_both_comparisons_must_pass():
    ok = MoveTest(InfoResult(0.05, 0.01, 1e-6, 1300), "", 1300, 1.0, 2.0)
    flat = MoveTest(InfoResult(0.01, 0.02, 0.31, 1300), "", 1300, 0.6, 2.0)
    wrong_way = MoveTest(InfoResult(-0.05, 0.01, 1.0, 1300), "", 1300, 1.0, 2.0)
    broken = MoveTest(None, "degenerate fit", 1300, None, None)
    assert passes(ok, ok)
    assert not passes(ok, flat) and not passes(flat, ok)
    assert not passes(ok, broken) and not passes(wrong_way, ok)

def test_wp_market_log_odds_are_the_season_slope_times_the_spread():
    frame = pd.DataFrame({"season": [2021, 2021], "early_spread": [3.0, -7.0]})
    np.testing.assert_allclose(wp_market_log_odds(frame, "early_spread", {2021: 0.15}), [0.45, -1.05])
    with pytest.raises(NoPriorFoldError):
        wp_market_log_odds(frame.assign(season=2020), "early_spread", {2021: 0.15})

def test_report_is_ascii_and_states_each_verdict(tmp_path):
    write_report(fake_result(passed={"wp": False, "ats": True, "ou": False}), tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.isascii()
    assert "ATS: passes" in text and "WP: does NOT pass" in text
    assert json.loads((tmp_path / "study_1b.json").read_text())["verdicts"]["ats"]["passed"] is True
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_study_1b.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `backtest/study_1b.py` and `scripts/run_study_1b.py`.**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_study_1b.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add backtest/study_1b.py scripts/run_study_1b.py tests/unit/test_study_1b.py
git commit -m "feat(1b): Study 1b line-move tests, verdict and report"
```

---

### Task 7: Guard scans, the real run and the readout

**Files:**
- No new code. Copy the outputs to `C:\Users\jackc\code\nfl-predict\.planning\research\2026-11-05-why-not-profitable\study-1b-report.md` and `study-1b.json`.

- [ ] **Step 1: Run the repo scans a new module can trip, and the touched modules' tests**

Run: `uv run pytest tests/unit/test_season_floor_single_source.py tests/unit/test_promote_models_ledger_guard.py tests/unit/test_ledger_stamping_site.py tests/unit/test_one_lock_rule_source_scan.py tests/unit/test_one_recommendation_path.py tests/unit/test_no_global_cutoff_in_features.py tests/unit/test_upstream_pin.py tests/unit/test_qb_tracking.py tests/unit/test_qb_v2.py tests/unit/test_qb_early.py tests/unit/test_market_rating.py tests/unit/test_league_level.py tests/unit/test_build_gold_v2.py tests/unit/test_v2_trainer.py tests/unit/test_study_1a.py tests/unit/test_study_1b_lines.py tests/unit/test_study_1b_inputs.py tests/unit/test_study_1b.py -p no:cacheprovider`
Expected: all PASS. A failure in a scan means a new module broke a repo rule: fix the module, never the scan.

- [ ] **Step 2: Confirm nothing else changed**

Run: `git -C C:\Users\jackc\Code\nfl-predict-v2 diff --name-status master...study-1b`
Expected: `A` for every new file, and `M` only for the six files named in Global Constraints. Anything else: undo it.

- [ ] **Step 3: Run the study (check ET time first; never 17:00-17:20 ET)**

The QB pass covers about 105 weeks and each target fits five folds. If Claude Code stops the run for low memory, launch it detached with `Start-Process` (working directory the worktree, output redirected to a log under `outputs/study_1b/`) and wait on its process id.

Run: `uv run python -m scripts.run_study_1b`
Expected: exit 0; `outputs/study_1b/report.md` and `study_1b.json` exist. Before reading any verdict, check the wiring against the measured facts in `constraints-for-midseason-changes.md` section f2. About 1,340 games should have lines. On the ATS rows, mean |early - day-before| should be near 0.65 and mean |early - close| near 1.0; on the O/U rows, near 0.73 and 1.05. A large departure means the lines or a sign are wired wrong: stop and investigate.

- [ ] **Step 4: File the results**

Copy `report.md` and `study_1b.json` to the two paths under Files. In `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-RESUME.md`, mark piece 2 done with its one-line result and the next step (piece 3's plan). `.planning/` is never committed.

- [ ] **Step 5: Report to the owner in plain words**

For each bet type: whether it passes, the two slopes in ordinary language ("when the model disagreed with Tuesday's line by 2 points, the line moved about X points its way by kickoff"), and what plan 4 will and will not build. Do not push, open a PR or merge until the owner says so.
